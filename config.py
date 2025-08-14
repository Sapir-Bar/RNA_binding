train_rbps = "data/experimental/training_RBPs.txt"
train_seqs = "data/experimental/training_seqs.txt"
train_affinity = "data/experimental/training_data2.txt"
train_rbps_embedding = "data/embedding/RBPs/embeddings_train.pt"
train_rna_embedding = "data/embedding/RNA/rnabert_embeddings.pt"

test_seqs = "data/experimental/test_seqs.txt"
test_rbps = "data/experimental/test_RBPs.txt"
test_rbps_embedding = "data/embedding/RBPs/embeddings_test.pt"

train_indices = "data/experimental/train_indices.txt"
test_indices = "data/experimental/test_indices.txt"
model_path = "models/rbp_similarity_model.pth"

# Training configuration
test_size = 0.2
random_state = 42
normalize_Y = True
num_epochs = 5
batch_size = 128
l2_coefficient = 1e-4 
learning_rate = 1e-4

# bash command: CUDA_DEVICE_VISIBLE=0 python affinity_regression/main.py