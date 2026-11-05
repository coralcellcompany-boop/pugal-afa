"""Masked classifier p(y | x_S) shared by every model-based method (DESIGN.md, notes/experiments.md).

Threads: importing this module calls `set_torch_threads()`, which sets `torch.set_num_threads(1)` unless
the environment variable PUGAL_TORCH_THREADS says otherwise. We run many experiment processes in
parallel (one per config x seed), so one intra-op thread per process avoids oversubscription. Measured on
an M3 Pro (Accelerate BLAS, 4096-row chunks of the 2 x 128 MLP): 1 thread 5.2 M rows/s, 2 threads 7.4 M,
4 threads 7.9 M, so extra threads per process buy little; processes scale almost linearly.

Compute metric: `predict_proba` adds the number of rows it evaluates to `model.n_evals`. Training and the
validation loss used for early stopping do not count.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover
    from experiment.data import AFAData

THREADS_ENV = "PUGAL_TORCH_THREADS"
CHUNK_ROWS = 4096  # rows per forward pass; the fastest chunk size on one M3 core


def set_torch_threads(n: int | None = None) -> int:
    """Set torch intra-op threads (default: $PUGAL_TORCH_THREADS or 1). Returns the value set."""
    if n is None:
        n = int(os.environ.get(THREADS_ENV, "1"))
    torch.set_num_threads(n)
    return n


set_torch_threads()


class MaskedClassifier(nn.Module):
    """MLP on concat(x * m, m): 2 x 128 ReLU, dropout 0.1. forward(x, m) returns logits."""

    def __init__(self, d: int, n_classes: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d, self.n_classes, self.hidden, self.dropout = d, n_classes, hidden, dropout
        self.net = nn.Sequential(
            nn.Linear(2 * d, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self.n_evals = 0

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(x.dtype)
        return self.net(torch.cat([x * m, m], dim=-1))


@torch.no_grad()
def predict_proba(model: nn.Module, X, M, chunk: int = CHUNK_ROWS) -> torch.Tensor:
    """X, M [..., d] (unobserved values are ignored) -> probabilities [..., C] in float64.

    Eval mode, no grad, evaluated in chunks of `chunk` rows. Adds the number of rows to model.n_evals.
    Softmax is taken in float64 so that risks and their differences are not dominated by rounding.
    """
    X = torch.as_tensor(X, dtype=torch.float32)
    M = torch.as_tensor(M).bool()
    lead, d = X.shape[:-1], X.shape[-1]
    Xf, Mf = X.reshape(-1, d), M.reshape(-1, d)
    n_rows = Xf.shape[0]
    was_training = model.training
    model.eval()
    outs = [torch.softmax(model(Xf[s:s + chunk], Mf[s:s + chunk]).double(), dim=-1)
            for s in range(0, n_rows, chunk)]
    model.train(was_training)
    model.n_evals = getattr(model, "n_evals", 0) + n_rows
    if not outs:
        return torch.empty(*lead, 0, dtype=torch.float64)
    p = torch.cat(outs)
    return p.reshape(*lead, p.shape[-1])


def group_matrix(groups: list[list[int]], d: int) -> np.ndarray:
    """[n_groups, d] bool; row g marks the features of group g."""
    G = np.zeros((len(groups), d), dtype=bool)
    for g, feats in enumerate(groups):
        G[g, feats] = True
    return G


def random_group_masks(n_rows: int, groups: list[list[int]], d: int, rng: np.random.Generator) -> np.ndarray:
    """Per row: k ~ Uniform{0..n_groups}, then a uniformly random subset of k groups. Returns [n_rows, d] bool."""
    n_groups = len(groups)
    k = rng.integers(0, n_groups + 1, size=n_rows)
    rank = np.argsort(np.argsort(rng.random((n_rows, n_groups)), axis=1), axis=1)
    observed_groups = rank < k[:, None]
    return (observed_groups.astype(np.float32) @ group_matrix(groups, d).astype(np.float32)) > 0


def train_classifier(data: "AFAData", seed: int, max_epochs: int = 60, patience: int = 5,
                     batch_size: int = 256, lr: float = 1e-3, hidden: int = 128,
                     dropout: float = 0.1, val_masks_per_row: int = 4) -> MaskedClassifier:
    """Random-mask training (Adam, cross-entropy) with early stopping on a fixed masked validation set.

    Deterministic given `seed`: numpy Generator for masks and shuffling; torch RNG (init, dropout) is
    seeded inside `fork_rng`, so the caller's global torch RNG state is untouched. Returns the best
    epoch's weights in eval mode with n_evals = 0.
    """
    rng = np.random.default_rng(seed)
    d = data.X_train.shape[1]
    Xtr = torch.from_numpy(np.ascontiguousarray(data.X_train, dtype=np.float32))
    ytr = torch.from_numpy(np.asarray(data.y_train, dtype=np.int64))
    Xva = torch.from_numpy(np.repeat(np.asarray(data.X_val, dtype=np.float32), val_masks_per_row, axis=0))
    yva = torch.from_numpy(np.repeat(np.asarray(data.y_val, dtype=np.int64), val_masks_per_row, axis=0))
    Mva = torch.from_numpy(random_group_masks(len(Xva), data.groups, d, rng))
    n = len(Xtr)
    loss_fn = nn.CrossEntropyLoss()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = MaskedClassifier(d, data.n_classes, hidden, dropout)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        best_loss, best_state, bad, epochs = float("inf"), copy.deepcopy(model.state_dict()), 0, 0
        for _ in range(max_epochs):
            epochs += 1
            model.train()
            perm = torch.from_numpy(rng.permutation(n))
            M = torch.from_numpy(random_group_masks(n, data.groups, d, rng))
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                loss = loss_fn(model(Xtr[idx], M[idx]), ytr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vloss = float(loss_fn(model(Xva, Mva), yva))
            if vloss < best_loss - 1e-6:
                best_loss, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        model.load_state_dict(best_state)
    model.eval()
    model.n_evals = 0
    model.train_info = {"epochs": epochs, "best_val_loss": best_loss}
    return model


def save_classifier(model: MaskedClassifier, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"d": model.d, "n_classes": model.n_classes, "hidden": model.hidden,
                "dropout": model.dropout, "state_dict": model.state_dict(),
                "train_info": getattr(model, "train_info", {})}, path)


def load_classifier(path: Path) -> MaskedClassifier:
    blob = torch.load(Path(path), weights_only=False)
    model = MaskedClassifier(blob["d"], blob["n_classes"], blob["hidden"], blob["dropout"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    model.n_evals = 0
    model.train_info = blob.get("train_info", {})
    return model
