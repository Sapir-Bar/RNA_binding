# --------------------------------------------
# Optimized GPU-friendly RNA–Protein Affinity Model
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

print("="*60)
print("🧬 RNA-Protein Affinity Model Training")
print("="*60)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

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

# GPU setup with better utilization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GB")
    # Set memory growth and optimization flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("⚠️  Using CPU (no CUDA available)")

# Optimized hyperparams for better GPU utilization
print("\n⚙️  Model Configuration:")
latent_dim   = 256      # Increased for better GPU utilization
rna_hidden   = 512      # Increased
prot_hidden  = 512      # Increased
dropout      = 0.2      # Slightly increased for regularization
lr           = 2e-3     # Slightly higher learning rate
weight_decay = 1e-4     # Increased regularization
epochs       = 50       # More epochs with better optimization
batch_size   = 262144   # Larger batch size for better GPU utilization
grad_clip    = 1.0
train_frac, val_frac, test_frac = 0.8, 0.1, 0.1

print(f"  Latent dim: {latent_dim}")
print(f"  Hidden dims: RNA={rna_hidden}, Protein={prot_hidden}")
print(f"  Batch size: {batch_size:,}")
print(f"  Learning rate: {lr}")
print(f"  Epochs: {epochs}")

# Move tensors to GPU early and use pinned memory
print("\n🔄 Moving data to GPU...")
start_gpu = time.time()

D = torch.from_numpy(D_np).pin_memory().to(device, non_blocking=True)  # (N, d_dim)
P = torch.from_numpy(P_np).pin_memory().to(device, non_blocking=True)  # (M, p_dim)
Y = torch.from_numpy(Y_np).pin_memory().to(device, non_blocking=True)  # (N, M)

if torch.cuda.is_available():
    torch.cuda.synchronize()  # Wait for transfers to complete

print(f"⏱️  GPU transfer completed in {time.time()-start_gpu:.2f}s")
print(f"📊 Total pairs to train on: {N*M:,}")

# -------------------
# Optimized Dataset with better memory access patterns
# -------------------
print("\n📦 Creating training splits...")
idx_rows, idx_cols = torch.meshgrid(torch.arange(N, device=device), torch.arange(M, device=device), indexing='ij')
all_idx = torch.stack([idx_rows.flatten(), idx_cols.flatten()], dim=1)  # (#pairs, 2)

# Split indices into train/val/test
perm = torch.randperm(all_idx.shape[0], device=device)
n_train = int(train_frac * len(perm))
n_val   = int(val_frac   * len(perm))
train_idx = all_idx[perm[:n_train]]
val_idx   = all_idx[perm[n_train:n_train+n_val]]
test_idx  = all_idx[perm[n_train+n_val:]]

print(f"  Training pairs: {len(train_idx):,}")
print(f"  Validation pairs: {len(val_idx):,}")
print(f"  Test pairs: {len(test_idx):,}")

class OptimizedPairDataset(Dataset):
    def __init__(self, idx_pairs, Y_tensor):
        self.idx = idx_pairs.cpu()  # Keep indices on CPU for DataLoader
        self.Y = Y_tensor.cpu()     # <-- Ensure Y is on CPU

    def __len__(self): 
        return self.idx.shape[0]
    
    def __getitem__(self, k):
        i, j = self.idx[k]
        return i.item(), j.item(), self.Y[i, j].item()

train_ds = OptimizedPairDataset(train_idx, Y)
val_ds   = OptimizedPairDataset(val_idx, Y)
test_ds  = OptimizedPairDataset(test_idx, Y)

# Optimized DataLoader settings
def make_loader(ds, shuffle):
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=4,  # Increased workers for better CPU-GPU pipeline
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

train_loader = make_loader(train_ds, True)
val_loader   = make_loader(val_ds, False)
test_loader  = make_loader(test_ds, False)

print(f"  Training batches: {len(train_loader):,}")
print(f"  Validation batches: {len(val_loader):,}")

# -------------------
# Enhanced Two-tower model with better GPU utilization
# -------------------
class OptimizedTower(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),  # Better training stability
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        )
        self.ln = nn.LayerNorm(out_dim)  # stabilizes dot-products
        
        # Initialize weights for better training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        z = self.net(x)
        return self.ln(z)

class OptimizedRPADot(nn.Module):
    def __init__(self, d_dim, p_dim, latent_dim, rna_hidden, prot_hidden, dropout=0.0):
        super().__init__()
        self.rna = OptimizedTower(d_dim, rna_hidden, latent_dim, dropout)
        self.pro = OptimizedTower(p_dim, prot_hidden, latent_dim, dropout)
        
    def forward_pairs(self, D_batch, P_batch):
        # D_batch: (B, d_dim), P_batch: (B, p_dim) with aligned pairs
        z_d = self.rna(D_batch)
        z_p = self.pro(P_batch)
        # dot product per pair with temperature scaling
        return (z_d * z_p).sum(dim=1)
    
    @torch.no_grad()
    def full_matrix(self, D_all, P_all, block=8192):  # Larger blocks for better GPU util
        # Efficient full NxM prediction in blocks
        self.eval()
        
        # Precompute all protein embeddings
        print("  Computing protein embeddings...")
        Zp = []
        for s in tqdm(range(0, P_all.shape[0], block), desc="Protein blocks"):
            e = min(s + block, P_all.shape[0])
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                Zp.append(self.pro(P_all[s:e]))
        Zp = torch.cat(Zp, dim=0)  # (M, k)
        
        # Compute RNA embeddings and dot products in blocks
        print("  Computing RNA-Protein interactions...")
        preds = torch.empty((D_all.shape[0], P_all.shape[0]), device=D_all.device, dtype=torch.float32)
        
        for s in tqdm(range(0, D_all.shape[0], block), desc="RNA blocks"):
            e = min(s + block, D_all.shape[0])
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                Zd = self.rna(D_all[s:e])  # (b, k)
                preds[s:e] = Zd @ Zp.T      # (b, M)
                
        return preds

print("\n🏗️  Building model...")
model = OptimizedRPADot(D.shape[1], P.shape[1], latent_dim, rna_hidden, prot_hidden, dropout).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# Optimized optimizer and scheduler
optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, 
                          betas=(0.9, 0.999), eps=1e-8)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optim, max_lr=lr*2, epochs=epochs, steps_per_epoch=len(train_loader),
    pct_start=0.1, anneal_strategy='cos'
)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
loss_fn = nn.MSELoss()

def run_epoch(loader, train=True, epoch_num=None):
    model.train(train)
    total_loss, n = 0.0, 0
    
    desc = f"{'Training' if train else 'Validation'}"
    if epoch_num is not None:
        desc += f" Epoch {epoch_num}"
    
    pbar = tqdm(loader, desc=desc, leave=False)
    
    for batch_idx, (i_idx, j_idx, y) in enumerate(pbar):
        i_idx = torch.tensor(i_idx, device=device, dtype=torch.long)
        j_idx = torch.tensor(j_idx, device=device, dtype=torch.long)
        y = torch.tensor(y, device=device, dtype=torch.float32)

        # Index into embeddings
        D_b = D[i_idx]
        P_b = P[j_idx]
        
        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
            pred = model.forward_pairs(D_b, P_b)
            loss = loss_fn(pred, y)
        
        if train:
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            if grad_clip:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optim)
            scaler.update()
            scheduler.step()
        
        batch_loss = float(loss.item())
        total_loss += batch_loss * y.numel()
        n += y.numel()
        
        # Update progress bar
        current_loss = total_loss / max(n, 1)
        pbar.set_postfix({
            'Loss': f'{current_loss:.6f}',
            'LR': f'{scheduler.get_last_lr()[0]:.2e}' if train else 'N/A'
        })
        
        # Memory cleanup every 100 batches
        if batch_idx % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return total_loss / max(n, 1)

# Training loop with enhanced monitoring
print("\n🎯 Starting training...")
print("="*60)

best_val = math.inf
best_state = None
train_losses = []
val_losses = []
t0 = time.time()

for ep in range(1, epochs+1):
    epoch_start = time.time()
    
    # Training
    tr_loss = run_epoch(train_loader, train=True, epoch_num=ep)
    train_losses.append(tr_loss)
    
    # Validation
    vl_loss = run_epoch(val_loader, train=False, epoch_num=ep)
    val_losses.append(vl_loss)
    
    # Save best model
    if vl_loss < best_val:
        best_val = vl_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        improvement = "⭐ NEW BEST!"
    else:
        improvement = ""
    
    epoch_time = time.time() - epoch_start
    current_lr = scheduler.get_last_lr()[0]
    
    # Enhanced progress reporting
    print(f"Epoch {ep:03d}/{epochs} | "
          f"Train: {tr_loss:.6f} | "
          f"Val: {vl_loss:.6f} | "
          f"LR: {current_lr:.2e} | "
          f"Time: {epoch_time:.1f}s | "
          f"{improvement}")
    
    # GPU memory info every 10 epochs
    if ep % 10 == 0 and torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / (1024**3)
        memory_cached = torch.cuda.memory_reserved() / (1024**3)
        print(f"  📊 GPU Memory: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached")

total_time = time.time() - t0
print("="*60)
print(f"🏁 Training completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"   Best validation loss: {best_val:.6f}")

# Load best model
if best_state is not None:
    model.load_state_dict(best_state)
    print("✅ Loaded best model weights")

# -------------------
# Enhanced Evaluation with detailed progress
# -------------------
print("\n📊 Evaluating model...")

@torch.no_grad()
def pearson_per_protein_from_masked_pairs(model, test_idx):
    print("🔮 Computing full prediction matrix...")
    # Compute full prediction matrix once for speed
    Y_pred_full = model.full_matrix(D, P)  # (N, M)
    
    print("📈 Computing correlations...")
    # Collect predictions and truths for test pairs
    i_test = test_idx[:,0].cpu()
    j_test = test_idx[:,1].cpu()
    y_true = Y[i_test, j_test].cpu().numpy()
    y_pred = Y_pred_full[i_test, j_test].cpu().numpy()

    # Group by protein j, compute Pearson over its test entries
    from collections import defaultdict
    grp_true = defaultdict(list)
    grp_pred = defaultdict(list)
    
    for t, p, j in zip(y_true, y_pred, j_test.numpy()):
        grp_true[int(j)].append(t)
        grp_pred[int(j)].append(p)

    def pearson(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.size < 2: 
            return np.nan
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        if sa == 0 or sb == 0: 
            return np.nan
        return float(np.corrcoef(a, b)[0,1])

    per_protein_r = np.full(M, np.nan, dtype=np.float32)
    
    print("🧮 Computing per-protein correlations...")
    for j in tqdm(range(M), desc="Proteins"):
        if j in grp_true:
            per_protein_r[j] = pearson(grp_true[j], grp_pred[j])

    mean_r = np.nanmean(per_protein_r)
    return per_protein_r, mean_r, Y_pred_full

per_protein_r, mean_r, Y_hat = pearson_per_protein_from_masked_pairs(model, test_idx)

print("="*60)
print("🎊 FINAL RESULTS")
print("="*60)
print(f"Mean Pearson r across proteins (test): {mean_r:.4f}")

# Enhanced result analysis
valid_correlations = per_protein_r[~np.isnan(per_protein_r)]
if len(valid_correlations) > 0:
    print(f"Valid correlations: {len(valid_correlations)}/{M}")
    print(f"Correlation stats:")
    print(f"  Mean: {np.mean(valid_correlations):.4f}")
    print(f"  Std:  {np.std(valid_correlations):.4f}")
    print(f"  Min:  {np.min(valid_correlations):.4f}")
    print(f"  Max:  {np.max(valid_correlations):.4f}")
    
    # Show top performing proteins
    top_indices = np.argsort(valid_correlations)[-10:]
    protein_indices = np.where(~np.isnan(per_protein_r))[0]
    print(f"\nTop 10 proteins by correlation:")
    for i, idx in enumerate(top_indices[::-1]):
        protein_id = protein_indices[idx]
        correlation = valid_correlations[idx]
        print(f"  {i+1:2d}. Protein {protein_id:3d}: r = {correlation:.4f}")

print(f"\n⏱️  Total runtime: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} minutes)")

if torch.cuda.is_available():
    final_memory = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"🔥 Peak GPU memory usage: {final_memory:.1f}GB")

print("="*60)
print("✨ Analysis complete! Full predicted affinity matrix available as Y_hat")
print("="*60)