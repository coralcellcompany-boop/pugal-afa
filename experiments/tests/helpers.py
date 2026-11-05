"""Shared test fixtures: small AFAData instances, an exact (Bayes-consistent) model/sampler pair, and a
cached trained toy-XOR setup."""

from __future__ import annotations

import functools
import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

try:  # the real dataclass (written in parallel); the fallback has the same fields
    from experiment.data import AFAData
except ImportError:  # pragma: no cover - only while data.py does not exist yet

    @dataclass
    class AFAData:  # type: ignore[no-redef]
        name: str
        X_train: np.ndarray
        y_train: np.ndarray
        X_val: np.ndarray
        y_val: np.ndarray
        X_test: np.ndarray
        y_test: np.ndarray
        n_classes: int
        groups: list[list[int]]
        imbalanced: bool


def _split(name, X, y, n_classes, groups=None, imbalanced=False, n_val=300, n_test=300) -> AFAData:
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    n_tr = len(X) - n_val - n_test
    d = X.shape[1]
    return AFAData(
        name=name,
        X_train=X[:n_tr], y_train=y[:n_tr],
        X_val=X[n_tr:n_tr + n_val], y_val=y[n_tr:n_tr + n_val],
        X_test=X[n_tr + n_val:], y_test=y[n_tr + n_val:],
        n_classes=n_classes,
        groups=groups if groups is not None else [[i] for i in range(d)],
        imbalanced=imbalanced,
    )


def make_xor_data(n: int = 4000, seed: int = 0) -> AFAData:
    """x1, x2, x3 ~ N(0,1); y = 1{x1>0} XOR 1{x2>0}; x3 is noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] > 0) ^ (X[:, 1] > 0)
    return _split("toy_xor", X, y, 2)


def make_grouped_data(n: int = 1500, seed: int = 0, imbalanced: bool = False) -> AFAData:
    """d = 6 features in 4 groups [[0, 1], [2], [3, 4], [5]]; label depends on several groups."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    score = X[:, 0] + 0.8 * X[:, 2] - 0.6 * X[:, 3] * X[:, 4] + 0.3 * X[:, 5]
    thr = 1.0 if imbalanced else 0.0
    y = score > thr
    return _split("grouped", X, y, 2, groups=[[0, 1], [2], [3, 4], [5]], imbalanced=imbalanced)


def make_random_data(n: int = 1500, d: int = 8, n_classes: int = 3, seed: int = 0) -> AFAData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    W = rng.standard_normal((d, n_classes))
    y = np.argmax(X @ W + 0.5 * rng.standard_normal((n, n_classes)), axis=1)
    return _split("random", X, y, n_classes)


@functools.lru_cache(maxsize=None)
def xor_setup(seed: int = 0, n_nb: int = 16):
    """(data, trained model, kNN sampler) for the toy XOR problem; cached across tests."""
    from experiment.model import train_classifier
    from experiment.sampler import KNNSampler

    data = make_xor_data(seed=seed)
    model = train_classifier(data, seed=seed)
    sampler = KNNSampler(data.X_train, n=n_nb, seed=seed, y_pool=data.y_train)
    return data, model, sampler


@functools.lru_cache(maxsize=None)
def trained_setup(kind: str, seed: int = 0, n_nb: int = 8):
    from experiment.model import train_classifier
    from experiment.sampler import KNNSampler

    data = {"grouped": make_grouped_data, "random": make_random_data}[kind](seed=seed)
    model = train_classifier(data, seed=seed, max_epochs=15)
    sampler = KNNSampler(data.X_train, n=n_nb, seed=seed, y_pool=data.y_train)
    return data, model, sampler


# ------------------------------------------------------------------ exact, Bayes-consistent model M
#
# A discrete joint p(x, y) over d = 3 binary features: combo z has weight w_z and label distribution P[z].
# ExactSampler returns, for a state (x_S), the consistent combos replicated in proportion to w_z, so its
# n samples ARE the exact conditional p(x_rest | x_S). ExactPosterior returns the exact p(y | x_S).
# Together they form one joint distribution, so the posterior is a martingale and Fact 1 holds exactly.

class ExactPosterior(nn.Module):
    def __init__(self, combos: torch.Tensor, weights: torch.Tensor, label_probs: torch.Tensor):
        super().__init__()
        self.register_buffer("combos", combos.double())
        self.register_buffer("weights", weights.double())
        self.register_buffer("label_probs", label_probs.double())
        self.n_evals = 0

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        x = x.double()
        m = m.bool()
        match = (~m[:, None, :]) | (x[:, None, :] == self.combos[None, :, :])   # [B, Z, d]
        w = match.all(-1).double() * self.weights[None, :]                       # [B, Z]
        post = (w @ self.label_probs) / w.sum(-1, keepdim=True)
        return torch.log(post)                                                    # softmax(log p) = p


class ExactSampler:
    def __init__(self, combos: np.ndarray, weights: np.ndarray):
        self.combos = combos.astype(np.float32)
        self.weights = weights.astype(np.int64)
        self.pool = torch.from_numpy(self.combos)
        self.y_pool = None
        d = combos.shape[1]
        totals = set()
        for mask in itertools.product([0, 1], repeat=d):
            obs = np.array(mask, bool)
            for vals in itertools.product([0.0, 1.0], repeat=int(obs.sum())):
                cons = np.all(self.combos[:, obs] == np.array(vals, np.float32), axis=1)
                totals.add(int(self.weights[cons].sum()))
        totals.discard(0)
        self.n = functools.reduce(math.lcm, totals)

    def neighbours_batch(self, X, M, k=None) -> torch.Tensor:
        assert k is None or k == self.n
        X = np.asarray(X, np.float32)
        M = np.asarray(M, bool)
        out = np.empty((len(X), self.n), np.int64)
        for b in range(len(X)):
            cons = np.all((~M[b]) | (self.combos == X[b]), axis=1)
            tot = self.weights[cons].sum()
            idx = np.concatenate([np.full(self.n * self.weights[z] // tot, z) for z in np.flatnonzero(cons)])
            out[b] = idx
        return torch.from_numpy(out)

    def neighbours(self, x, m, k=None) -> np.ndarray:
        return self.neighbours_batch(np.asarray(x)[None], np.asarray(m)[None], k)[0].numpy()


def exact_problem(seed: int = 0, n_classes: int = 3):
    rng = np.random.default_rng(seed)
    combos = np.array(list(itertools.product([0.0, 1.0], repeat=3)))
    weights = np.array([1, 2, 1, 3, 2, 1, 1, 2])
    label_probs = rng.dirichlet(np.ones(n_classes) * 0.7, size=len(combos))
    model = ExactPosterior(torch.tensor(combos), torch.tensor(weights), torch.tensor(label_probs))
    sampler = ExactSampler(combos, weights)
    return model, sampler, combos


def all_states(d: int, values=(0.0, 1.0)):
    """Every (x, m) for binary features: x is zero where unobserved."""
    for mask in itertools.product([False, True], repeat=d):
        m = np.array(mask, bool)
        for vals in itertools.product(values, repeat=int(m.sum())):
            x = np.zeros(d, np.float32)
            x[m] = vals
            yield x, m


def make_xor_trap_data(n: int = 5000, seed: int = 0) -> AFAData:
    """Greedy trap: with prob 0.75 y = 1{x1>0} XOR 1{x2>0}, else y = 1{x3>0}. U_1(x3) > 0 = U_1(x1) = U_1(x2)
    at the empty state, so greedy starts with x3, while the pair (x1, x2) is the best two-step plan."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    z = rng.random(n) < 0.25
    y = np.where(z, X[:, 2] > 0, (X[:, 0] > 0) ^ (X[:, 1] > 0))
    return _split("xor_trap", X, y, 2)


@functools.lru_cache(maxsize=None)
def xor_trap_setup(seed: int = 0, n_nb: int = 16):
    from experiment.model import train_classifier
    from experiment.sampler import KNNSampler

    data = make_xor_trap_data(seed=seed)
    model = train_classifier(data, seed=seed)
    sampler = KNNSampler(data.X_train, n=n_nb, seed=seed, y_pool=data.y_train)
    return data, model, sampler
