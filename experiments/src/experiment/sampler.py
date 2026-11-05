"""k-nearest-neighbour empirical conditional p(x_rest | x_S) (DESIGN.md `sampler.py`).

A conditional sample of the unobserved features is a pool row's values at those features, so joint values
(x_j, x_k) come from the same row and keep their dependence. Distances are Euclidean on the observed
(z-scored) features, computed in float64. `neighbours` is a pure function of (x, m): ties are broken by
position in the pool, and the pool is shuffled by `seed`, so ties are broken at random but reproducibly.
With nothing observed, every distance is 0 and the result is the first n rows of the shuffled pool
(n fixed random rows). The same state always gets the same neighbours, which greedy-vs-PUGAL equality
and the h = 2 identity need.
"""

from __future__ import annotations

import numpy as np
import torch

MAX_POOL = 5000
_QUERY_CHUNK = 1024


class KNNSampler:
    def __init__(self, X_pool: np.ndarray, n: int = 16, seed: int = 0, *, y_pool: np.ndarray | None = None,
                 max_pool: int = MAX_POOL):
        rng = np.random.default_rng(seed)
        X_pool = np.asarray(X_pool, dtype=np.float32)
        idx = rng.permutation(len(X_pool))[:max_pool]
        self.pool_index = idx                      # rows of the given X_pool that form the pool, in pool order
        self.X_pool = np.ascontiguousarray(X_pool[idx])
        self.pool = torch.from_numpy(self.X_pool.copy())          # float32 [N, d]
        self._p64 = self.pool.double()
        self._p64_sq_t = (self._p64 ** 2).T.contiguous()          # [d, N]
        self._p64_t = self._p64.T.contiguous()
        self.y_pool = None if y_pool is None else np.asarray(y_pool, dtype=np.int64)[idx]
        self.y_pool_t = None if self.y_pool is None else torch.from_numpy(self.y_pool)
        if not 1 <= n <= len(idx):
            raise ValueError(f"n={n} must be in [1, pool size {len(idx)}]")
        self.n = n

    def neighbours_batch(self, X, M, k: int | None = None) -> torch.Tensor:
        """X, M [B, d] -> LongTensor [B, k] of pool indices, nearest first."""
        k = self.n if k is None else k
        X = torch.as_tensor(X).double()
        M = torch.as_tensor(M).double()
        out = torch.empty(X.shape[0], k, dtype=torch.long)
        for s in range(0, X.shape[0], _QUERY_CHUNK):
            x, m = X[s:s + _QUERY_CHUNK], M[s:s + _QUERY_CHUNK]
            xm = x * m
            dist = (xm * x).sum(1, keepdim=True) - 2.0 * (xm @ self._p64_t) + m @ self._p64_sq_t
            out[s:s + _QUERY_CHUNK] = self._k_smallest(dist, k)
        return out

    @staticmethod
    def _k_smallest(dist: torch.Tensor, k: int) -> torch.Tensor:
        """Same result as torch.sort(dist, stable=True).indices[:, :k] (order by distance, then pool
        position), but via topk: a full stable sort of [B, 5000] dominated LA2 time. Rows with a tie at the
        k-th distance (ambiguous membership, e.g. the empty state or discrete features) use the full sort."""
        vals, idx = torch.topk(dist, k, dim=1, largest=False, sorted=False)
        idx, perm = idx.sort(dim=1)                                  # by pool position ...
        order = vals.gather(1, perm).sort(dim=1, stable=True).indices   # ... then stably by distance
        res = idx.gather(1, order)
        kth = vals.max(1, keepdim=True).values
        tie = (dist <= kth).sum(1) > k
        if tie.any():
            res[tie] = torch.sort(dist[tie], dim=1, stable=True).indices[:, :k]
        return res

    def neighbours(self, x: np.ndarray, m: np.ndarray, k: int | None = None) -> np.ndarray:
        """Indices of the k (default n) nearest pool rows on the observed features of one state."""
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))[None]
        m = torch.as_tensor(np.asarray(m, dtype=bool))[None]
        return self.neighbours_batch(x, m, k)[0].numpy()
