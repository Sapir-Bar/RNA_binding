import torch
from typing import Dict, Iterator, Optional, Tuple, Callable

Tensor = torch.Tensor

class ProteinPairDataset:
    """
    Builds unique unordered protein index pairs (i <= j), once per epoch.
    For each pair it serves:
      - Pi = P[i], Ej = E[j]
      - Pj = P[j], Ei = E[i]
      - Sij = dot(Y[:, i], Y[:, j])  (cosine similarity if Y columns are L2-normalized)

    This dataset is vectorized: it does NOT materialize all pairs as Python tuples.
    It stores two index tensors (I, J) = triu_indices(M, M, offset=0).
    """

    def __init__(
        self,
        P: Tensor,          # (M, pdim)
        E: Tensor,          # (M, ddim)
        Y: Tensor,          # (N, M)  (assumed column-normalized)
        *,
        device: Optional[torch.device] = None,
        include_diag: bool = True,
    ):
        assert P.ndim == 2 and E.ndim == 2 and Y.ndim == 2, "P/E/Y must be 2D"
        M = P.shape[0]
        assert E.shape[0] == M and Y.shape[1] == M, "P/E rows and Y columns must all be M"

        self.P = P
        self.E = E
        self.Y = Y
        self.M = M
        self.device = device or P.device

        offset = 0 if include_diag else 1
        # Generates unique unordered pairs: (i,j) where i ≤ j
        I, J = torch.triu_indices(M, M, offset=offset)  # shape (2, K)
        # on CPU to keep memory light; moved to device when batching
        self.I = I  # first protein indices
        self.J = J  # second protein indices
        self.K = I.numel()  # number of elements in the tensor

    def __len__(self) -> int:
        return self.K

    def num_pairs(self) -> int:
        # M(M+1)/2 if include_diag else M(M-1)/2
        M = self.M
        return (M * (M + 1)) // 2

class ProteinPairLoader:
    """
    Lightweight loader that shuffles pair indices each epoch and yields vectorized batches:
      Returns dict with:
        p_left  = P[I_batch]          (B, pdim)   -> corresponds to (Pi)
        e_right = E[J_batch]          (B, ddim)   -> corresponds to (Ej)
        p_right = P[J_batch]          (B, pdim)   -> corresponds to (Pj)
        e_left  = E[I_batch]          (B, ddim)   -> corresponds to (Ei)
        s       = sum(Y[:, I]*Y[:, J], dim=0)     (B,)  -> Sij
        idx_i   = I_batch (cpu)       (B,)
        idx_j   = J_batch (cpu)       (B,)
    """

    def __init__(
        self,
        dataset: ProteinPairDataset,
        batch_size: int = 1024,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = 42,
    ):
        self.ds = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change epoch to vary the shuffling sequence deterministically."""
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[Dict[str, Tensor]]:
        K = len(self.ds)
        indices = torch.arange(K) # [0, 1, 2, ..., k-1]

        if self.shuffle:
            g = torch.Generator()
            if self.seed is not None:
                # Different stream per epoch (stable & reproducible)
                g.manual_seed(self.seed + self._epoch)
            indices = indices[torch.randperm(K, generator=g)]

        bs = self.batch_size
        if self.drop_last:
            end = (K // bs) * bs
        else:
            end = K

        P, E, Y = self.ds.P, self.ds.E, self.ds.Y
        dev = self.ds.device

        # Pre-store pair index tensors (CPU) to slice efficiently
        I_all, J_all = self.ds.I, self.ds.J

        for start in range(0, end, bs):
            sl = indices[start : min(start + bs, K)]
            # Batch pair indices on CPU (good for slicing and for saving/logging)
            I_b_cpu = I_all.index_select(0, sl) # returns the relevant indices from I_all for the current batch
            J_b_cpu = J_all.index_select(0, sl) # returns the relevant indices from J_all for the current batch

            # Move indices to device for tensor ops
            I_b = I_b_cpu.to(dev)
            J_b = J_b_cpu.to(dev)

            # Vectorized gathers
            p_left  = P.index_select(0, I_b)   # (B, pdim) -> Pi
            e_right = E.index_select(0, J_b)   # (B, ddim) -> Ej
            p_right = P.index_select(0, J_b)   # (B, pdim) -> Pj
            e_left  = E.index_select(0, I_b)   # (B, ddim) -> Ei

            # Sij = dot(Y[:, i], Y[:, j]) over probe dimension (N)
            # Y[:, I_b] -> (N, B), Y[:, J_b] -> (N, B)
            s = (Y.index_select(1, I_b) * Y.index_select(1, J_b)).sum(dim=0)  # (B,)

            yield {
                "p_left":  p_left,
                "e_right": e_right,
                "p_right": p_right,
                "e_left":  e_left,
                "s":       s,
                "idx_i":   I_b_cpu,   # keep CPU copies for bookkeeping/saving
                "idx_j":   J_b_cpu,
            }

    def __len__(self) -> int: # return number of batches
        K = len(self.ds)
        if self.drop_last:
            return K // self.batch_size
        return (K + self.batch_size - 1) // self.batch_size

class ProteinPairTrainer:
    """
    Generic trainer for a black-box model f(P, E) -> score.
    You provide:
      - model: nn.Module with forward(P_batch, E_batch) -> (B,) or (B,1)
      - optimizer: torch optimizer
      - loss_fn: callable (pred_ij, pred_ji, s) -> scalar loss
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
        *,
        device: Optional[torch.device] = None,
        grad_accum_steps: int = 1,
        log_every: int = 0,
        clip_grad_norm: Optional[float] = None,  # optional gradient clipping
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or next(model.parameters()).device
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.log_every = int(log_every)
        self.clip_grad_norm = clip_grad_norm

    def train_one_epoch(self, loader) -> float:

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss, num_batches = 0.0, 0

        for step, batch in enumerate(loader):
            p_left,  e_right = batch["p_left"],  batch["e_right"]
            p_right, e_left  = batch["p_right"], batch["e_left"]
            s = batch["s"]

            pred_ij = self.model(p_left,  e_right)   # f(Pi, Ej)
            pred_ji = self.model(p_right, e_left)    # f(Pj, Ei)

            if pred_ij.ndim > 1: pred_ij = pred_ij.squeeze(-1)
            if pred_ji.ndim > 1: pred_ji = pred_ji.squeeze(-1)

            loss = self.loss_fn(pred_ij, pred_ji, s) / self.grad_accum_steps
            loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1

            if self.log_every and (step % self.log_every == 0):
                print(f"[train] step={step} loss={loss.item() * self.grad_accum_steps:.4f}") # convert to wandb print

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        metric_fn: Optional[Callable[[Tensor, Tensor, Tensor], float]] = None
    ) -> Tuple[float, Optional[float]]:
        self.model.eval()
        total_loss, num_batches = 0.0, 0
        metric_accum, metric_count = 0.0, 0

        for batch in loader:
            p_left,  e_right = batch["p_left"],  batch["e_right"]
            p_right, e_left  = batch["p_right"], batch["e_left"]
            s = batch["s"]

            pred_ij = self.model(p_left,  e_right)
            pred_ji = self.model(p_right, e_left)
            if pred_ij.ndim > 1: pred_ij = pred_ij.squeeze(-1)
            if pred_ji.ndim > 1: pred_ji = pred_ji.squeeze(-1)

            loss = self.loss_fn(pred_ij, pred_ji, s)
            total_loss += loss.item()
            num_batches += 1

            if metric_fn is not None:
                metric_accum += float(metric_fn(pred_ij, pred_ji, s))
                metric_count += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_metric = (metric_accum / metric_count) if metric_count > 0 else None
        return avg_loss, avg_metric