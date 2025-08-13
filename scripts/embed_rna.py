import torch
from multimolecule import RnaTokenizer, RnaBertModel
import os

# Disable tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load RNAbert model and tokenizer
print("Loading RNAbert model...")
tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnabert")
model = RnaBertModel.from_pretrained("multimolecule/rnabert").to(device)

# Read sequences
print("Reading sequences...")
with open('training_seqs.txt', 'r') as f:
    sequences = [line.strip() for line in f if line.strip()]

print(f"Found {len(sequences)} sequences")

# Process in batches
batch_size = 16
embeddings = []

print("Processing sequences...")
for i in range(0, len(sequences), batch_size):
    batch = sequences[i:i+batch_size]
    
    try:
        # Tokenize batch
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over sequence length
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings.cpu())
        
        if (i//batch_size + 1) % 10 == 0:
            print(f"Processed batch {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}")
            
    except Exception as e:
        print(f"Error processing batch {i//batch_size + 1}: {e}")
        continue

# Combine all embeddings
if embeddings:
    embeddings = torch.cat(embeddings, dim=0)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Save embeddings
    torch.save(embeddings, 'rnabert_embeddings.pt')
    print("Saved embeddings to rnabert_embeddings.pt")
    
    # Test loading
    loaded_embeddings = torch.load('rnabert_embeddings.pt')
    print(f"Verified: loaded embeddings shape: {loaded_embeddings.shape}")
else:
    print("No embeddings generated - all batches failed")

print("Done!")