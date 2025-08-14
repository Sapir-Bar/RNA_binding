import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
Tensor = torch.Tensor


class RBPSimilarityModel(nn.Module):
    """
    Neural network model for RBP-RNA binding prediction via similarity learning.
    
    Implements the approach from section 2.2.2 of the paper:
    1. Converts binding intensity prediction to similarity prediction (Y -> Y^T Y)
    2. Uses neural networks to predict RBP-RBP similarities
    3. Reconstructs binding intensities from predicted similarities
    """
    
    def __init__(
        self,
        protein_dim: int,
        rna_dim: int,
        hidden_protein_dim: int = 128,  # L in the paper
        hidden_rna_dim: int = 64,       # K in the paper
        merge_type: str = "concat",     # Type of merge operation
      
    ):
        """
        Args:
            protein_dim: Dimension of protein features (S in paper)
            rna_dim: Dimension of RNA features (Q in paper) 
            hidden_protein_dim: Hidden dimension for protein subnet (L)
            hidden_rna_dim: Hidden dimension for RNA subnet (K)
            merge_type: How to merge h_P and h_E ("concat", "hadamard", "sum", "bilinear")
        """
        super().__init__()
        
        self.protein_dim = protein_dim
        self.rna_dim = rna_dim
        self.hidden_protein_dim = hidden_protein_dim
        self.hidden_rna_dim = hidden_rna_dim
        self.merge_type = merge_type
  
        # Protein subnet: h_P = σ(W_P^T * P_{j,:} + b_P)
        self.protein_net = nn.Linear(protein_dim, hidden_protein_dim)
        
        # RNA subnet: h_E = σ(W_E^T * E_{i,:} + b_E)  
        self.rna_net = nn.Linear(rna_dim, hidden_rna_dim)
        
        # Merge layer output dimension depends on merge type
        if merge_type == "concat":
            merge_dim = hidden_protein_dim + hidden_rna_dim
        elif merge_type == "hadamard":
            # Element-wise product requires same dimensions
            assert hidden_protein_dim == hidden_rna_dim, "Hadamard merge requires equal hidden dimensions"
            merge_dim = hidden_protein_dim
        elif merge_type == "sum":
            # Sum requires same dimensions
            assert hidden_protein_dim == hidden_rna_dim, "Sum merge requires equal hidden dimensions"
            merge_dim = hidden_protein_dim
        elif merge_type == "bilinear":
            merge_dim = 1  # Bilinear produces scalar directly
        else:
            raise ValueError(f"Unknown merge_type: {merge_type}")
            
        # Final layer: Ŝ_{i,j} = w_M^T * M(h_P, h_E) + b_M
        if merge_type == "bilinear":
            self.bilinear = nn.Bilinear(hidden_protein_dim, hidden_rna_dim, 1)
            self.final_net = None
        else:
            self.final_net = nn.Linear(merge_dim, 1)
            self.bilinear = None
            
        # Store training data for reconstruction (will be set during training)
        self.register_buffer('Y_train', None)
        
    def forward(self, P_batch: Tensor, E_batch: Tensor) -> Tensor:
        """
        Forward pass to predict similarity values.
        
        Args:
            P_batch: Protein features, shape (batch_size, protein_dim)
            E_batch: RNA-derived features E = Y^T D, shape (batch_size, rna_dim)
            
        Returns:
            Predicted similarities, shape (batch_size,)
        """
        # Protein subnet: h_P = σ(W_P^T * P + b_P)
        h_P = torch.sigmoid(self.protein_net(P_batch))  # (batch_size, hidden_protein_dim)
        
        # RNA subnet: h_E = σ(W_E^T * E + b_E)  
        h_E = torch.sigmoid(self.rna_net(E_batch))      # (batch_size, hidden_rna_dim)
        
        # Merge h_P and h_E
        if self.merge_type == "concat":
            merged = torch.cat([h_P, h_E], dim=1)
        elif self.merge_type == "hadamard":
            merged = h_P * h_E  # Element-wise product
        elif self.merge_type == "sum":
            merged = h_P + h_E  # Element-wise sum
        elif self.merge_type == "bilinear":
            # Bilinear layer handles merging internally
            return self.bilinear(h_P, h_E).squeeze(-1)
        else:
            raise ValueError(f"Unknown merge_type: {self.merge_type}")
            
        # Final prediction: Ŝ = w_M^T * merged + b_M
        similarity = self.final_net(merged).squeeze(-1)  # (batch_size,)
        
        return similarity
    
    def forward_pairs(self, P: Tensor, E: Tensor) -> Tensor:
        """
        Predict similarities for all pairs of RBPs.
        
        Args:
            P: Protein features, shape (M, protein_dim)
            E: RNA features E = Y^T D, shape (M, rna_dim)
            
        Returns:
            Similarity matrix, shape (M, M)
        """
        M = P.shape[0]
        similarities = torch.zeros(M, M, device=P.device, dtype=P.dtype)
        
        # Compute all pairwise similarities
        for i in range(M):
            for j in range(M):
                # Prepare batch with protein i and RNA features j
                P_batch = P[j:j+1]  # Use protein j's features
                E_batch = E[i:i+1]  # Use RNA features for protein i
                
                sim = self.forward_similarity(P_batch, E_batch)
                similarities[i, j] = sim
                
        return similarities
########################################################################################    
    def reconstruct_binding_intensities(self, similarities: Tensor) -> Tensor:
        """
        Reconstruct binding intensities from predicted similarities.
        
        Args:
            similarities: Predicted similarities against training RBPs, shape (M_train,)
            
        Returns:
            Reconstructed binding intensities, shape (N,)
        """
        if self.Y_train is None:
            raise RuntimeError("Y_train not set. Call set_training_data() first.")
            
        if self.reconstruction_method == "weighted_sum":
            # ŷ_x = Y * ŝ_x  (Equation 6)
            binding_intensities = torch.matmul(self.Y_train, similarities)
            
        elif self.reconstruction_method == "pseudoinverse":
            # ŷ_x = (Y^T)^+ * ŝ_x  (Equation 7)
            Y_T = self.Y_train.transpose(0, 1)  # (M, N)
            Y_T_pinv = torch.linalg.pinv(Y_T)   # (N, M)
            binding_intensities = torch.matmul(Y_T_pinv, similarities)
            
        else:
            raise ValueError(f"Unknown reconstruction_method: {self.reconstruction_method}")
            
        return binding_intensities

    def predict_for_test_rbp(self, p_test: Tensor, P_train: Tensor, E_train: Tensor) -> Tensor:
        """
        Predict binding intensities for a test RBP.
        
        Args:
            p_test: Test RBP features, shape (protein_dim,)
            P_train: Training RBP features, shape (M_train, protein_dim)
            E_train: Training RNA features, shape (M_train, rna_dim)
            
        Returns:
            Predicted binding intensities, shape (N,)
        """
        M_train = P_train.shape[0]
        similarities = torch.zeros(M_train, device=p_test.device, dtype=p_test.dtype)
        
        # Compute similarities between test RBP and all training RBPs
        for i in range(M_train):
            # Use test RBP features with training RNA features
            P_batch = p_test.unsqueeze(0)      # (1, protein_dim)
            E_batch = E_train[i:i+1]           # (1, rna_dim)
            
            sim = self.forward_similarity(P_batch, E_batch)
            similarities[i] = sim
        
        # Reconstruct binding intensities
        binding_intensities = self.reconstruct_binding_intensities(similarities)
        
        return binding_intensities

def create_model(
    protein_dim: int,
    rna_dim: int,
    hidden_protein_dim: int = 128,
    hidden_rna_dim: int = 64,
    merge_type: str = "concat",
) -> RBPSimilarityModel:
    """
    Factory function to create RBPSimilarityModel.
    
    Args:
        protein_dim: Dimension of protein features
        rna_dim: Dimension of RNA features
        hidden_protein_dim: Hidden dimension for protein subnet (L in paper)
        hidden_rna_dim: Hidden dimension for RNA subnet (K in paper)  
        merge_type: Merge operation ("concat", "hadamard", "sum", "bilinear")
        
    Returns:
        Initialized model
    """
    return RBPSimilarityModel(
        protein_dim=protein_dim,
        rna_dim=rna_dim, 
        hidden_protein_dim=hidden_protein_dim,
        hidden_rna_dim=hidden_rna_dim,
        merge_type=merge_type,
    )