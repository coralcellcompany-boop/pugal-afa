"""Exact model M* for SynMix (exploratory X1, notes/experiments.md "Exploratory follow-up").

SynMix features are independent (z ~ Bernoulli(rho), x1..x19 ~ N(0, 1)), so the exact conditional
p(x_U | x_S) is the product of marginals, and the label posterior has a closed form:
    p(y = 1 | x_S) = eta + (1 - 2 eta) * [P(z = 1 | x_S) * P(b1 XOR b2 | x_S) + P(z = 0 | x_S) * P(b3 | x_S)],
with b_i = 1{x_i > 0}, eta the label-flip rate, and P(bit | x_S) = 0.5 for an unobserved feature.
Inputs are the z-normalised features; raw values are recovered with the train mean / std in AFAData.

`SynMixOracle` replaces the masked classifier (same `forward(x, m) -> logits` and `n_evals` interface);
`OracleSampler` replaces the kNN sampler: its pool is fresh draws from the generator, and the rows used for a
state are a deterministic hash of (x_S, S), so the same state always gets the same samples.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from experiment.data import SYNMIX_LABEL_NOISE, make_synmix

POOL_ROWS = 5000
_MASK64 = (1 << 63) - 1


class SynMixOracle(nn.Module):
    def __init__(self, data, rho: float, eta: float = SYNMIX_LABEL_NOISE):
        super().__init__()
        if data.norm_mu is None:
            raise ValueError("SynMixOracle needs the normalisation statistics of a synthetic dataset")
        self.register_buffer("mu", torch.as_tensor(data.norm_mu, dtype=torch.float64))
        self.register_buffer("sd", torch.as_tensor(data.norm_sd, dtype=torch.float64))
        self.rho, self.eta = float(rho), float(eta)
        self.n_classes = 2
        self.n_evals = 0

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        raw = x.double() * self.sd + self.mu
        m = m.bool()
        z = (raw[:, 0] > 0.5).double()
        b = (raw[:, 1:4] > 0).double()
        pz = torch.where(m[:, 0], z, torch.full_like(z, self.rho))
        pxor = torch.where(m[:, 1] & m[:, 2], (b[:, 0] != b[:, 1]).double(), torch.full_like(z, 0.5))
        p3 = torch.where(m[:, 3], b[:, 2], torch.full_like(z, 0.5))
        p1 = self.eta + (1 - 2 * self.eta) * (pz * pxor + (1 - pz) * p3)
        p = torch.stack([1 - p1, p1], dim=-1).clamp_min(1e-12)
        return p.log().to(x.dtype)


class OracleSampler:
    """Exact conditional sampler for SynMix: independent features -> draws from the marginals."""

    def __init__(self, data, rho: float, n: int = 16, seed: int = 0, pool_rows: int = POOL_ROWS):
        raw, _ = make_synmix(pool_rows, rho, np.random.default_rng([seed, 2]))
        pool = ((raw - data.norm_mu) / data.norm_sd).astype(np.float32)
        self.X_pool = np.ascontiguousarray(pool)
        self.pool = torch.from_numpy(self.X_pool.copy())
        self.pool_index = np.zeros(0, dtype=np.int64)     # no training row is in the pool
        self.y_pool = None
        self.n = int(n)
        g = torch.Generator().manual_seed(int(seed) + 104729)
        d = pool.shape[1]
        # odd 62-bit multipliers for the state hash and per-slot mixing constants
        self._w = torch.randint(1, 1 << 62, (2, d), generator=g, dtype=torch.long) | 1
        self._slot = torch.randint(1, 1 << 62, (2, 4096), generator=g, dtype=torch.long) | 1

    @staticmethod
    def _mix(h: torch.Tensor) -> torch.Tensor:
        h = h & _MASK64
        h = (h ^ (h >> 31)) * 0x5BD1E995
        h = h & _MASK64
        h = (h ^ (h >> 29)) * 0x27D4EB2F
        return (h ^ (h >> 32)) & _MASK64

    def neighbours_batch(self, X, M, k: int | None = None) -> torch.Tensor:
        """X, M [B, d] -> LongTensor [B, k] of pool indices; a deterministic function of (x_S, S)."""
        k = self.n if k is None else int(k)
        X = torch.as_tensor(X, dtype=torch.float32)
        M = torch.as_tensor(M).bool()
        q = torch.round(torch.where(M, X, torch.zeros_like(X)).double() * 1e5).long()
        h = ((q * self._w[0]) + (M.long() * self._w[1])).sum(1)          # wraps mod 2^64, deterministic
        h = self._mix(h)
        hk = self._mix(h[:, None] * self._slot[0, :k] + self._slot[1, :k])  # [B, k]
        return torch.remainder(hk, len(self.pool))
