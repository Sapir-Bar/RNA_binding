import os
import torch
import numpy as np
from typing import Tuple, Optional, Union
Tensor = torch.Tensor

def load_data(
    protein_embed_pt: str,
    rna_embed_pt: str,
    affinity_txt: str,
    *,
    train_indices: Optional[str] = None,      # optional path to save train indices
    test_indices: Optional[str]  = None,      # optional path to save test indices
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
    test_size: float = 0.2,
    random_state: int = 42,
    normalize_Y: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    1) Load protein embeddings dict: {"ids": list, "embeddings": list} → P (M × pdim)
    2) Load RNA embeddings list → D (N × ddim)
    3) Load affinity matrix from text → Y (N × M)
    4) Optionally L2-normalize Y columns
    5) Compute E = Y^T · D → (M × ddim)
    6) Make a fixed train/test split by proteins; return:
       P_train, E_train, Y_train, P_test, E_test, Y_test, D
    7) Writes two text files containing the selected protein indices (one per line),
       in the exact order of the returned tensors (train first, test second).
    """
    # 1) proteins
    prot_obj = torch.load(protein_embed_pt, map_location="cpu")
    P_np = np.asarray(prot_obj["embeddings"], dtype=np.float32)   # shape (M, pdim)

    # 2) RNA
    rna_obj = torch.load(rna_embed_pt, map_location="cpu")
    if isinstance(rna_obj, torch.Tensor):
        D_np = rna_obj.cpu().numpy().astype(np.float32)           # shape (N, ddim)
    else:
        D_np = np.asarray(rna_obj, dtype=np.float32)              # shape (N, ddim)

    # 3) affinity Y (N × M)
    Y_np = np.loadtxt(affinity_txt, dtype=np.float32, delimiter="\t")
    N, M = Y_np.shape

    # 4) consistency checks
    if P_np.shape[0] != M:
        raise ValueError(f"P rows ({P_np.shape[0]}) must equal Y columns (M={M})")
    if D_np.shape[0] != N:
        raise ValueError(f"D rows ({D_np.shape[0]}) must equal Y rows (N={N})")

    # 5) convert to torch
    P = torch.as_tensor(P_np, dtype=dtype, device=device)     # (M, pdim)
    D = torch.as_tensor(D_np, dtype=dtype, device=device)     # (N, ddim)
    Y = torch.as_tensor(Y_np, dtype=dtype, device=device)     # (N, M)

    # 6) column-wise L2 normalize Y (safe for zero columns)
    if normalize_Y:
        col_norms = torch.linalg.norm(Y, dim=0, keepdim=True)   # (1, M)
        Y = Y / col_norms.clamp_min(1e-12)

    # 7) E = Y^T · D  → (M, ddim)
    E = Y.transpose(0, 1).matmul(D)

    # 8) fixed split by proteins
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0,1)")
    M_test = max(1, int(round(M * test_size)))

    g = torch.Generator().manual_seed(random_state)   # CPU generator for stable indices
    perm_cpu = torch.randperm(M, generator=g)         # permutation order = tensors order
    test_idx_cpu  = perm_cpu[:M_test].tolist()
    train_idx_cpu = perm_cpu[M_test:].tolist()

    # tensor indexing uses device copy
    test_idx_dev  = perm_cpu[:M_test].to(device)
    train_idx_dev = perm_cpu[M_test:].to(device)

    P_train = P.index_select(0, train_idx_dev)
    E_train = E.index_select(0, train_idx_dev)
    Y_train = Y.index_select(1, train_idx_dev)

    P_test  = P.index_select(0, test_idx_dev)
    E_test  = E.index_select(0, test_idx_dev)
    Y_test  = Y.index_select(1, test_idx_dev)

    # 9) save train/test indices to text files
    if train_indices is not None:
        with open(train_indices, "w") as f:
            for idx in train_idx_cpu:
                f.write(f"{idx}\n")

    if test_indices is not None:
        with open(test_indices, "w") as f:
            for idx in test_idx_cpu:
                f.write(f"{idx}\n")

    return P_train, E_train, Y_train, P_test, E_test, Y_test, D