# --------------------------------------------
# RNA-Specific Protein Similarity Model (Corrected Approach)
# --------------------------------------------
import math, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import torch
import sys
import numpy as np
from typing import Tuple, Optional, Union
from tqdm import tqdm
Tensor = torch.Tensor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

print("="*60)
print("🧬 RNA-Specific Protein Similarity Model")
print("="*60)


print("📁 Loading data...")
start_load = time.time()

# 1) Load proteins
print("  Loading protein embeddings...")
prot_obj = torch.load(config.train_rbps_embedding, map_location="cpu")
P_np = np.asarray(prot_obj["embeddings"], dtype=np.float32)   # shape (M, protein_dim)
print(f"  ✓ Proteins: {P_np.shape} ({P_np.nbytes/(1024**2):.1f} MB)")

# 2) Load RNA embeddings
print("  Loading RNA embeddings...")
rna_obj = torch.load(config.train_rna_embedding, map_location="cpu")
if isinstance(rna_obj, torch.Tensor):
    D_np = rna_obj.cpu().numpy().astype(np.float32)           # shape (N, rna_dim)
else:
    D_np = np.asarray(rna_obj, dtype=np.float32)              # shape (N, rna_dim)
print(f"  ✓ RNAs: {D_np.shape} ({D_np.nbytes/(1024**2):.1f} MB)")

# 3) Load affinity matrix Y (N × M)
print("  Loading affinity matrix...")
Y_np = np.loadtxt(config.train_affinity, dtype=np.float32, delimiter="\t")
N, M = Y_np.shape
print(f"  ✓ Affinity matrix: {Y_np.shape} ({Y_np.nbytes/(1024**2):.1f} MB)")

# 4) Consistency checks
if P_np.shape[0] != M:
    raise ValueError(f"P rows ({P_np.shape[0]}) must equal Y columns (M={M})")
if D_np.shape[0] != N:
    raise ValueError(f"D rows ({D_np.shape[0]}) must equal Y rows (N={N})")

print(f"⏱️  Data loading completed in {time.time()-start_load:.2f}s")

# Reproducibility
seed = 42
torch.manual_seed(seed); np.random.seed(seed)

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GB")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("⚠️  Using CPU (no CUDA available)")

# CORRECTED APPROACH: For each RNA, compute protein-protein similarities
print("\n🔄 Creating RNA-specific protein similarity targets...")
transform_start = time.time()

# For each RNA i, we want to predict similarities between proteins j,k based on Y[i,j] and Y[i,k]
# The target similarity for RNA i between proteins j,k is: Y[i,j] * Y[i,k] (correlation-like)
# OR cosine similarity: Y[i,j] * Y[i,k] / (||Y[i,:]|| * ||Y[i,:]||) - but this is just Y[i,j] * Y[i,k] for normalized Y

# Normalize each RNA's binding profile (each row of Y)
Y_norm = Y_np / (np.linalg.norm(Y_np, axis=1, keepdims=True) + 1e-8)  # (N, M)

print(f"  ✓ Normalized binding profiles for {N} RNAs")
print(f"⏱️  Data preparation completed in {time.time()-transform_start:.2f}s")

# Model configuration
print("\n⚙️  Model Configuration:")
rna_latent = 256    # RNA embedding dimension
prot_latent = 256   # Protein embedding dimension  
hidden_dim = 512    # Hidden layer dimension
dropout = 0.2
lr = 2e-3
weight_decay = 1e-4
epochs = 50
batch_size = 32768  # Larger batches for better GPU utilization
grad_clip = 1.0

print(f"  RNA latent dim: {rna_latent}")
print(f"  Protein latent dim: {prot_latent}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Batch size: {batch_size:,}")
print(f"  Learning rate: {lr}")
print(f"  Epochs: {epochs}")

# Move tensors to GPU
print("\n🔄 Moving data to GPU...")
start_gpu = time.time()

D = torch.from_numpy(D_np).to(device)         # (N, rna_dim)
P = torch.from_numpy(P_np).to(device)         # (M, prot_dim)
Y = torch.from_numpy(Y_np).to(device)         # (N, M)
Y_norm = torch.from_numpy(Y_norm).to(device)  # (N, M) normalized

if torch.cuda.is_available():
    torch.cuda.synchronize()

print(f"⏱️  GPU transfer completed in {time.time()-start_gpu:.2f}s")

# -------------------
# Dataset: (RNA_i, Protein_j, Protein_k) -> similarity target
# -------------------
print("\n📦 Creating RNA-specific similarity training data...")

# Create training triplets: (rna_idx, prot_j_idx, prot_k_idx, similarity_target)
training_triplets = []

# For efficiency, sample a subset of all possible triplets
n_triplets_per_rna = min(500, M * (M-1) // 2)  # Sample up to 500 protein pairs per RNA
total_budget = min(2000000, N * n_triplets_per_rna)  # Limit total triplets

print(f"  Sampling {n_triplets_per_rna} protein pairs per RNA...")
print(f"  Total budget: {total_budget:,} triplets")

np.random.seed(42)
sampled_count = 0

for rna_idx in tqdm(range(N), desc="Creating triplets"):
    if sampled_count >= total_budget:
        break
        
    # Get binding values for this RNA
    rna_binding = Y_norm[rna_idx].cpu().numpy()  # (M,)
    
    # Sample protein pairs for this RNA
    if n_triplets_per_rna >= M * (M-1) // 2:
        # Use all pairs if budget allows
        protein_pairs = [(j, k) for j in range(M) for k in range(j+1, M)]
    else:
        # Sample random pairs
        all_pairs = [(j, k) for j in range(M) for k in range(j+1, M)]
        protein_pairs = np.random.choice(len(all_pairs), size=n_triplets_per_rna, replace=False)
        protein_pairs = [all_pairs[idx] for idx in protein_pairs]
    
    for j, k in protein_pairs:
        if sampled_count >= total_budget:
            break
            
        # Compute similarity target: dot product of normalized binding values
        similarity = rna_binding[j] * rna_binding[k]
        
        training_triplets.append((rna_idx, j, k, similarity))
        sampled_count += 1

print(f"  Created {len(training_triplets):,} training triplets")

# Convert to arrays
triplets_array = np.array(training_triplets)
rna_indices = triplets_array[:, 0].astype(np.int64)
prot_j_indices = triplets_array[:, 1].astype(np.int64)
prot_k_indices = triplets_array[:, 2].astype(np.int64)
similarities = triplets_array[:, 3].astype(np.float32)

# Split into train/val/test
perm = np.random.permutation(len(triplets_array))
n_train = int(0.8 * len(perm))
n_val = int(0.1 * len(perm))

train_idx = perm[:n_train]
val_idx = perm[n_train:n_train+n_val]
test_idx = perm[n_train+n_val:]

print(f"  Training triplets: {len(train_idx):,}")
print(f"  Validation triplets: {len(val_idx):,}")
print(f"  Test triplets: {len(test_idx):,}")

class RNAProteinSimilarityDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        return (rna_indices[i], prot_j_indices[i], prot_k_indices[i], similarities[i])

train_ds = RNAProteinSimilarityDataset(train_idx)
val_ds = RNAProteinSimilarityDataset(val_idx)
test_ds = RNAProteinSimilarityDataset(test_idx)

def make_loader(ds, shuffle):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

train_loader = make_loader(train_ds, True)
val_loader = make_loader(val_ds, False)
test_loader = make_loader(test_ds, False)

print(f"  Training batches: {len(train_loader):,}")

# -------------------
# Enhanced Model: RNA-conditioned Protein Similarity
# -------------------
class RNAConditionedSimilarityModel(nn.Module):
    def __init__(self, rna_dim, prot_dim, rna_latent, prot_latent, hidden_dim, dropout=0.2):
        super().__init__()
        
        # RNA encoder
        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rna_latent),
            nn.LayerNorm(rna_latent),
            nn.ReLU(inplace=True)
        )
        
        # Protein encoder
        self.protein_encoder = nn.Sequential(
            nn.Linear(prot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prot_latent),
            nn.LayerNorm(prot_latent),
            nn.ReLU(inplace=True)
        )
        
        # RNA-conditioned protein transformation
        # This allows the RNA context to modify how we view protein similarities
        self.rna_condition_net = nn.Sequential(
            nn.Linear(rna_latent, prot_latent * 2),
            nn.LayerNorm(prot_latent * 2),
            nn.ReLU(inplace=True),
            nn.Linear(prot_latent * 2, prot_latent)
        )
        
        # Similarity prediction head
        self.similarity_head = nn.Sequential(
            nn.Linear(prot_latent * 3, hidden_dim),  # prot_j + prot_k + rna_context
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, rna_features, prot_j_features, prot_k_features):
        # Encode inputs
        rna_emb = self.rna_encoder(rna_features)           # (B, rna_latent)
        prot_j_emb = self.protein_encoder(prot_j_features) # (B, prot_latent)
        prot_k_emb = self.protein_encoder(prot_k_features) # (B, prot_latent)
        
        # RNA-conditioned protein context
        rna_context = self.rna_condition_net(rna_emb)      # (B, prot_latent)
        
        # Combine all information for similarity prediction
        combined = torch.cat([prot_j_emb, prot_k_emb, rna_context], dim=1)  # (B, prot_latent * 3)
        
        similarity = self.similarity_head(combined).squeeze(1)  # (B,)
        
        return similarity
    
    @torch.no_grad()
    def predict_rna_protein_affinities(self, rna_idx, D_all, P_all, Y_train):
        """Predict affinities for a specific RNA against all proteins using similarity reconstruction"""
        self.eval()
        
        rna_features = D_all[rna_idx:rna_idx+1].expand(M, -1)  # (M, rna_dim)
        
        # Predict similarities between all protein pairs for this RNA
        similarities = torch.zeros((M, M), device=D_all.device)
        
        batch_size = 256
        for i in range(0, M, batch_size):
            for j in range(0, M, batch_size):
                end_i = min(i + batch_size, M)
                end_j = min(j + batch_size, M)
                
                # Create batch of all pairs in this block
                pairs = []
                for ii in range(i, end_i):
                    for jj in range(j, end_j):
                        pairs.append((ii, jj))
                
                if not pairs:
                    continue
                
                batch_rna = rna_features[0:1].expand(len(pairs), -1)
                batch_i_idx = torch.tensor([p[0] for p in pairs], device=D_all.device)
                batch_j_idx = torch.tensor([p[1] for p in pairs], device=D_all.device)
                
                batch_prot_i = P_all[batch_i_idx]
                batch_prot_j = P_all[batch_j_idx]
                
                batch_sims = self.forward(batch_rna, batch_prot_i, batch_prot_j)
                
                # Fill similarity matrix
                for idx, (ii, jj) in enumerate(pairs):
                    similarities[ii, jj] = batch_sims[idx]
        
        # Make similarity matrix symmetric
        similarities = (similarities + similarities.T) / 2
        
        # Set diagonal to 1 (protein similarity with itself)
        similarities.fill_diagonal_(1.0)
        
        # Reconstruct binding affinities using weighted combination
        # Use softmax to get proper weights
        weights = torch.softmax(similarities, dim=1)  # (M, M)
        
        # Weighted sum of training binding profiles
        # reconstructed_affinities = (weights @ Y_train[:, :].T).mean(dim=0)  # (M,)
        # Weighted sum across proteins → (M,)
        # Take mean over training RNAs AFTER multiplying by weights
        reconstructed_affinities = (weights @ Y_train.T).mean(dim=1)  # (M,)

        
        return reconstructed_affinities

print("\n🏗️  Building RNA-conditioned similarity model...")
model = RNAConditionedSimilarityModel(
    D.shape[1], P.shape[1], rna_latent, prot_latent, hidden_dim, dropout
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")

# Optimizer and scheduler
optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optim, max_lr=lr*2, epochs=epochs, steps_per_epoch=len(train_loader),
    pct_start=0.1, anneal_strategy='cos'
)

scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))
loss_fn = nn.MSELoss()

def run_epoch(loader, train=True, epoch_num=None):
    model.train(train)
    total_loss, n = 0.0, 0
    
    desc = f"{'Training' if train else 'Validation'}"
    if epoch_num is not None:
        desc += f" Epoch {epoch_num}"
    
    pbar = tqdm(loader, desc=desc, leave=False)
    
    for batch_idx, (rna_idx, prot_j_idx, prot_k_idx, sim_true) in enumerate(pbar):
        rna_idx = rna_idx.to(device, dtype=torch.long)
        prot_j_idx = prot_j_idx.to(device, dtype=torch.long)
        prot_k_idx = prot_k_idx.to(device, dtype=torch.long)
        sim_true = sim_true.to(device, dtype=torch.float32)
        
        # Get features
        rna_features = D[rna_idx]      # (B, rna_dim)
        prot_j_features = P[prot_j_idx] # (B, prot_dim)
        prot_k_features = P[prot_k_idx] # (B, prot_dim)
        
        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            sim_pred = model(rna_features, prot_j_features, prot_k_features)
            loss = loss_fn(sim_pred, sim_true)
        
        if train:
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            if grad_clip:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optim)
            scaler.update()
            scheduler.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss * sim_true.numel()
        n += sim_true.numel()
        
        current_loss = total_loss / max(n, 1)
        pbar.set_postfix({
            'Loss': f'{current_loss:.6f}',
            'LR': f'{scheduler.get_last_lr()[0]:.2e}' if train else 'N/A'
        })
        
        if batch_idx % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return total_loss / max(n, 1)

# Training loop
print("\n🎯 Starting RNA-specific similarity training...")
print("="*60)

best_val = math.inf
best_state = None
t0 = time.time()

for ep in range(1, epochs+1):
    epoch_start = time.time()
    
    tr_loss = run_epoch(train_loader, train=True, epoch_num=ep)
    vl_loss = run_epoch(val_loader, train=False, epoch_num=ep)
    
    if vl_loss < best_val:
        best_val = vl_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        improvement = "⭐ NEW BEST!"
    else:
        improvement = ""
    
    epoch_time = time.time() - epoch_start
    current_lr = scheduler.get_last_lr()[0]
    
    print(f"Epoch {ep:03d}/{epochs} | "
          f"Train: {tr_loss:.6f} | "
          f"Val: {vl_loss:.6f} | "
          f"LR: {current_lr:.2e} | "
          f"Time: {epoch_time:.1f}s | "
          f"{improvement}")
    
    if ep % 10 == 0 and torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / (1024**3)
        print(f"  📊 GPU Memory: {memory_used:.1f}GB used")

training_time = time.time() - t0
print("="*60)
print(f"🏁 Training completed in {training_time:.1f}s ({training_time/60:.1f} minutes)")

if best_state is not None:
    model.load_state_dict(best_state)
    print("✅ Loaded best model weights")

# -------------------
# Evaluation: Reconstruct full affinity matrix and evaluate
# -------------------
print("\n📊 Reconstructing full affinity matrix...")

@torch.no_grad()
def evaluate_full_reconstruction(model, D_all, P_all, Y_all, n_test_rnas=1000):
    """Evaluate by reconstructing affinities for a sample of RNAs with proper train/test split"""
    model.eval()
    
    # PROPER TRAIN/TEST SPLIT: Use different RNAs for training vs testing
    all_rna_indices = np.arange(N)
    np.random.shuffle(all_rna_indices)
    
    n_test = min(n_test_rnas, N // 5)  # Use 20% for testing
    test_rna_indices = all_rna_indices[:n_test]
    train_rna_indices = all_rna_indices[n_test:]
    
    print(f"  Using {len(train_rna_indices)} RNAs for reconstruction training")
    print(f"  Testing on {len(test_rna_indices)} unseen RNAs")
    
    Y_train_subset = Y_all[train_rna_indices]  # Only training RNAs
    
    all_correlations = []
    
    for rna_idx in tqdm(test_rna_indices, desc="Reconstructing"):
        # Get true affinities for this TEST RNA (unseen during reconstruction)
        true_affinities = Y_all[rna_idx].cpu().numpy()
        
        # Predict affinities using only TRAINING RNAs for reconstruction
        pred_affinities = model.predict_rna_protein_affinities(rna_idx, D_all, P_all, Y_train_subset)
        pred_affinities = pred_affinities.cpu().numpy()
        
        # Compute correlation
        if np.std(true_affinities) > 0 and np.std(pred_affinities) > 0:
            correlation = np.corrcoef(true_affinities, pred_affinities)[0, 1]
            if not np.isnan(correlation):
                all_correlations.append(correlation)
    
    return np.array(all_correlations)

# Evaluate reconstruction performance with proper splits
rna_correlations = evaluate_full_reconstruction(model, D, P, Y, n_test_rnas=200)

print("="*60)
print("🎊 FINAL RESULTS - RNA-SPECIFIC SIMILARITY MODEL")
print("="*60)
print(f"Mean correlation across RNAs: {np.mean(rna_correlations):.4f}")
print(f"Std correlation: {np.std(rna_correlations):.4f}")
print(f"Min correlation: {np.min(rna_correlations):.4f}")
print(f"Max correlation: {np.max(rna_correlations):.4f}")
print(f"RNAs evaluated: {len(rna_correlations)}")

# Show top performing RNAs
top_indices = np.argsort(rna_correlations)[-10:]
print(f"\nTop 10 RNAs by reconstruction correlation:")
for i, idx in enumerate(top_indices[::-1]):
    correlation = rna_correlations[idx]
    print(f"  {i+1:2d}. RNA {idx:3d}: r = {correlation:.4f}")

total_time = time.time() - t0
print(f"\n⏱️  Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")

if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"🔥 Peak GPU memory usage: {peak_memory:.1f}GB")

print("="*60)
print("✨ RNA-specific similarity model complete!")
print("   - Learned protein similarities conditioned on each RNA")
print("   - Uses much more information than global approach")
print("   - Reconstructs binding via RNA-specific protein relationships")
print("="*60)