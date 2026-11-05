"""Gates G(s) deciding when PUGAL runs the rollout (notes/method.md Sec. 5; DESIGN.md `gates.py`).

Convention: `score(state, ctx)` is "higher = more reason to open"; `open_from_score(s)` = s > threshold;
`open(state, ctx)` = open_from_score(score(state, ctx)). Callers that also need the score (PUGAL, gate-AUROC
collection) call `score` ONCE and then `open_from_score`: RandomGate draws a new number on every score call.
`ctx` is the StepContext of the greedy step at s (U_1 of all unobserved groups, g, runner-up, ...), so gates
reuse those evaluations instead of paying for them again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from experiment.utility import State, StepContext, Utility

if TYPE_CHECKING:  # pragma: no cover
    from experiment.data import AFAData


class Gate:
    threshold: float = 0.0

    def score(self, state: State, ctx: StepContext) -> float:
        raise NotImplementedError

    def open_from_score(self, s: float) -> bool:
        return bool(s > self.threshold)

    def open(self, state: State, ctx: StepContext) -> bool:
        return self.open_from_score(self.score(state, ctx))


class ConstantGate(Gate):
    """Always open (True) or always closed (False). Used for tests and the always-on reduction."""

    def __init__(self, value: bool):
        self.value = bool(value)
        self.threshold = 0.5

    def score(self, state, ctx) -> float:
        return 1.0 if self.value else 0.0


# ------------------------------------------------------------------ cached gate signals

def _redundancy(state: State, ctx: StepContext) -> float:
    if "R" not in ctx.extra:
        ctx.extra["R"] = ctx.engine.redundancy(state, ctx.g, ctx.j2, u1=ctx.u1)
    return ctx.extra["R"]


def _synergy_hat(state: State, ctx: StepContext, flagged_pairs) -> float:
    """N_hat(s): Prop. 3's N(s) with (j, k) restricted to the flagged pairs (None = all pairs)."""
    key = ("N", None if flagged_pairs is None else frozenset(flagged_pairs))
    if key not in ctx.extra:
        N, per_pair = ctx.engine.synergy_term_detail(state, ctx.g, flagged_pairs, u1=ctx.u1)
        ctx.extra[key] = N
        ctx.extra["pair_syn"] = per_pair          # online per-pair synergy, read by TopMCandidates("online")
    return ctx.extra[key]


class SynergyGate(Gate):
    """G-syn: score = R(s) + N_hat(s), open iff score > eps (Prop. 3: Delta_2 <= R + N).

    flagged_pairs: set of ordered group pairs (j, k) from `screen_synergy`; None = no screen (all pairs,
    ablation `noscreen`). use_redundancy / use_synergy switch the terms off (ablations `nosyn`, `noredund`).
    """

    def __init__(self, flagged_pairs, eps: float = 0.01, use_redundancy: bool = True, use_synergy: bool = True):
        self.flagged_pairs = None if flagged_pairs is None else set(flagged_pairs)
        self.threshold = self.eps = eps
        self.use_redundancy, self.use_synergy = use_redundancy, use_synergy

    def score(self, state, ctx) -> float:
        s = 0.0
        if self.use_redundancy:
            s += _redundancy(state, ctx)
        if self.use_synergy:
            s += _synergy_hat(state, ctx, self.flagged_pairs)
        return s


class MarginGate(Gate):
    """G-margin: open iff a_1 - a_2 < tau. score = -(a_1 - a_2)."""

    def __init__(self, tau: float):
        self.tau = tau
        self.threshold = -tau

    def score(self, state, ctx) -> float:
        return -np.inf if ctx.a2 is None else -(ctx.a1 - ctx.a2)


class RandomGate(Gate):
    """G-rand: Bernoulli(p) per call. score = 1 - u with u ~ U(0, 1), open iff u < p. Deterministic given
    the seed and the order of calls."""

    def __init__(self, p: float, seed: int = 0):
        self.p = p
        self.threshold = 1.0 - p
        self.rng = np.random.default_rng(seed)

    def score(self, state, ctx) -> float:
        return 1.0 - float(self.rng.random())


class OracleGate(Gate):
    """G-oracle: score = exact Delta_2(s) from always-LA2 over all unobserved groups; open iff > eps."""

    def __init__(self, eps: float = 0.01):
        self.threshold = self.eps = eps

    def score(self, state, ctx) -> float:
        return ctx.engine.delta2(state, ctx=ctx)


# ------------------------------------------------------------------ learned gate

def gate_features(state: State, ctx: StepContext, flagged_pairs=None) -> np.ndarray:
    """[r(s), b, a_1 - a_2, R(s), N_hat(s), |S|]; b = 0 if the budget is not tracked, |S| = observed features."""
    a_gap = 0.0 if ctx.a2 is None else ctx.a1 - ctx.a2
    b = 0 if state.budget is None else state.budget
    return np.array([ctx.r, b, a_gap, _redundancy(state, ctx), _synergy_hat(state, ctx, flagged_pairs),
                     float(state.m.sum())], dtype=np.float64)


class LogisticModel:
    """L2-regularised logistic regression (Newton / IRLS on standardised features). Deterministic."""

    def __init__(self, l2: float = 1e-2, max_iter: int = 100, tol: float = 1e-8):
        self.l2, self.max_iter, self.tol = l2, max_iter, tol
        self.mu = self.sd = self.w = None

    def _design(self, F: np.ndarray) -> np.ndarray:
        Z = (np.asarray(F, np.float64) - self.mu) / self.sd
        return np.hstack([Z, np.ones((len(Z), 1))])

    def fit(self, F: np.ndarray, y: np.ndarray) -> "LogisticModel":
        F = np.asarray(F, np.float64)
        y = np.asarray(y, np.float64)
        self.mu = F.mean(0)
        self.sd = F.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        A = self._design(F)
        w = np.zeros(A.shape[1])
        reg = self.l2 * np.eye(A.shape[1])
        reg[-1, -1] = 0.0                                   # no penalty on the bias
        for _ in range(self.max_iter):
            p = 1.0 / (1.0 + np.exp(-(A @ w)))
            grad = A.T @ (p - y) + reg @ w
            H = (A * (p * (1 - p))[:, None]).T @ A + reg + 1e-9 * np.eye(len(w))
            step = np.linalg.solve(H, grad)
            w -= step
            if np.abs(step).max() < self.tol:
                break
        self.w = w
        return self

    def predict_proba(self, F: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self._design(np.atleast_2d(F)) @ self.w)))


class LearnedGate(Gate):
    """G-learn: score = model.predict_proba(gate_features(s)), open iff score > threshold."""

    def __init__(self, model, flagged_pairs=None, threshold: float = 0.5):
        self.model = model
        self.flagged_pairs = None if flagged_pairs is None else set(flagged_pairs)
        self.threshold = threshold

    def score(self, state, ctx) -> float:
        return float(self.model.predict_proba(gate_features(state, ctx, self.flagged_pairs)[None])[0])


# ------------------------------------------------------------------ offline synergy screen

def screen_synergy(model, sampler, data: "AFAData", n_states: int, quantile: float, *, risk_kind: str = "brier",
                   budget: int | None = None, seed: int = 0, engine: Utility | None = None,
                   return_scores: bool = False):
    """Offline synergy screen (notes/method.md Sec. 5).

    Runs greedy on training rows (rows outside the sampler pool when there are any, so a row is not its own
    neighbour) and, at each visited state with >= 2 unobserved groups, computes the local synergy
    E_{x_j}[(U_1(k | s + j) - U_1(k | s))_+] for every ordered pair of unobserved groups. Stops after
    n_states states. Each state costs about G * n * (1 + (G - 1) * n) classifier evaluations (G groups),
    i.e. the same as one always-LA2 step. Identical states (e.g. the empty state) are computed once.

    Returns the set of ordered pairs whose average synergy is > the `quantile` of all pair averages and > 0
    (and the [G, G] matrix of averages, NaN where never active, if return_scores).
    """
    eng = engine or Utility(model, sampler, data.groups, risk_kind)
    groups = eng.groups
    G, d = len(groups), eng.d
    budget = min(10, G - 1) if budget is None else min(budget, G - 1)
    rng = np.random.default_rng(seed)
    X = np.asarray(data.X_train, np.float32)
    in_pool = np.zeros(len(X), bool)
    pool_index = getattr(sampler, "pool_index", None)
    if pool_index is not None and len(pool_index) == len(getattr(sampler, "X_pool", [])) and \
            pool_index.max(initial=-1) < len(X):
        in_pool[pool_index] = True
    rows = np.flatnonzero(~in_pool) if (~in_pool).any() else np.arange(len(X))
    rows = rows[rng.permutation(len(rows))]

    total = np.zeros((G, G))
    count = np.zeros((G, G))
    memo: dict[bytes, tuple[dict, int]] = {}
    collected = 0
    for i in rows:
        if collected >= n_states:
            break
        st = State.empty(d, budget)
        for _ in range(budget):
            if collected >= n_states:
                break
            free = eng.unobserved(st)
            if len(free) < 2:
                break
            key = st.m.tobytes() + st.x.tobytes()
            if key not in memo:
                ctx = eng.greedy_context(st)
                pairs = [(j, k) for j in free for k in free if j != k]
                memo[key] = (eng.synergy_pairs(st, pairs, u1=ctx.u1), ctx.g)
            vals, g = memo[key]
            for (j, k), v in vals.items():
                total[j, k] += v
                count[j, k] += 1
            collected += 1
            st = st.acquire(g, X[i], groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    flagged = flag_from_scores(avg, quantile)
    return (flagged, avg) if return_scores else flagged


def flag_from_scores(avg: np.ndarray, quantile: float) -> set[tuple[int, int]]:
    """Ordered pairs whose average screen synergy is > the `quantile` of all finite pair averages and > 0."""
    valid = avg[np.isfinite(avg)]
    if not valid.size:
        return set()
    thr = float(np.quantile(valid, quantile))
    jj, kk = np.nonzero(np.isfinite(avg) & (avg > thr) & (avg > 0))
    return {(int(j), int(k)) for j, k in zip(jj, kk)}
