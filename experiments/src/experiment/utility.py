"""Acquisition utilities under the model M = (masked classifier, kNN conditional); notes/method.md Sec. 2-4.

State = (x, m, budget): x [d] float32 holds observed values and 0 elsewhere, m [d] bool, budget = acquisitions
left (None = not tracked). Actions are groups (lists of feature indices acquired together).

Conventions (used identically by every policy, so choices are comparable bit for bit):
- E_after(j | s) = mean over the n neighbour rows a of s of r(s + j(a)), where s + j(a) takes the values of
  group j from pool row a. U_1(j | s) = r(s) - E_after(j | s).
- Greedy picks argmin_j E_after(j | s) over candidates in ascending group order, first minimum on ties
  (this is argmax U_1 without the rounding of the subtraction).
- The truncated rollout Q^{(h)}(s, j) is the EXACT expectation under M over the kNN tree: acquire j
  (n branches), then h - 1 greedy steps, each step again averaging over the n neighbours of the state it is
  in, then terminal risk r. The last greedy step reuses its E_after values (min_k E_after = expected risk
  after the step), so it costs no extra evaluations.
- Neighbours of a state are a pure function of the state (sampler), so the samples a used for U_1(j | s)
  are the same pool rows that define the children s + j(a) in two-step quantities; this makes
  rollout_value(h=2) = r(s) - two_step_value exact up to float rounding.

Performance: every quantity is computed by building ALL (state, group, neighbour) rows as one tensor and
evaluating the model in CHUNK_ROWS-row chunks; there are no Python loops over neighbour rows or candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch

from experiment.model import CHUNK_ROWS, group_matrix, predict_proba
from experiment.risk import KINDS, risk


# ------------------------------------------------------------------ state

@dataclass(frozen=True, eq=False)
class State:
    x: np.ndarray            # [d] float32, 0 where unobserved
    m: np.ndarray            # [d] bool
    budget: int | None = None

    def __post_init__(self):
        m = np.asarray(self.m, dtype=bool)
        x = np.where(m, np.asarray(self.x, dtype=np.float32), np.float32(0.0)).astype(np.float32)
        object.__setattr__(self, "m", m)
        object.__setattr__(self, "x", x)

    @classmethod
    def empty(cls, d: int, budget: int | None = None) -> "State":
        return cls(np.zeros(d, np.float32), np.zeros(d, bool), budget)

    def acquire(self, g: int, x_full: np.ndarray, groups: list[list[int]]) -> "State":
        """Observe group g of the instance x_full; the budget (if tracked) drops by one."""
        m = self.m.copy()
        feats = list(groups[g])
        if m[feats].all():
            raise ValueError(f"group {g} is already acquired")
        m[feats] = True
        budget = None if self.budget is None else self.budget - 1
        return State(np.where(m, np.asarray(x_full, np.float32), np.float32(0.0)), m, budget)


@dataclass
class StepContext:
    """What the greedy step at s computes (cost 1 + |U| n evaluations); gates read from it."""
    engine: "Utility"
    state: State
    r: float                         # r(s)
    eafter: dict[int, float]         # E_after(j | s) for every unobserved group j
    u1: dict[int, float]             # U_1(j | s)
    order: list[int]                 # unobserved groups, best greedy first (E_after asc, index asc)
    extra: dict = field(default_factory=dict)   # cache for gate signals (R, N_hat, ...)

    @property
    def g(self) -> int:
        return self.order[0]

    @property
    def j2(self) -> int | None:
        return self.order[1] if len(self.order) > 1 else None

    @property
    def a1(self) -> float:
        return self.u1[self.g]

    @property
    def a2(self) -> float | None:
        return None if self.j2 is None else self.u1[self.j2]


def _first_argmin(values: dict[int, float]) -> int:
    """Key with the smallest value; ties -> smallest key."""
    return min(sorted(values), key=lambda j: values[j])


# ------------------------------------------------------------------ engine

class Utility:
    """U_1, two-step value, truncated rollout, synergy N and redundancy R under M (all batched)."""

    def __init__(self, model, sampler, groups: list[list[int]], risk_kind: str = "brier", n: int | None = None):
        if risk_kind not in KINDS:
            raise ValueError(f"unknown risk kind {risk_kind!r}")
        self.model, self.sampler, self.risk_kind = model, sampler, risk_kind
        self.groups = [list(g) for g in groups]
        self.pool = sampler.pool
        self.d = int(self.pool.shape[1])
        self.n_groups = len(self.groups)
        self.gmask_np = group_matrix(self.groups, self.d)
        self.gmask = torch.from_numpy(self.gmask_np)
        self._gmask_f = self.gmask.float()
        self.n = int(n if n is not None else sampler.n)

    # -------------------------------------------------- state helpers
    def unobserved(self, state: State, candidates: Iterable[int] | None = None) -> list[int]:
        """Unacquired groups (a group counts as acquired once all its features are observed)."""
        free = (self.gmask_np & ~state.m[None, :]).any(1)
        if candidates is not None:
            allow = np.zeros(self.n_groups, bool)
            allow[list(candidates)] = True
            free &= allow
        return np.flatnonzero(free).tolist()

    def _allow(self, candidates: Iterable[int] | None) -> torch.Tensor:
        allow = torch.ones(self.n_groups, dtype=torch.bool)
        if candidates is not None:
            allow[:] = False
            allow[list(candidates)] = True
        return allow

    def _free_batch(self, M: torch.Tensor) -> torch.Tensor:
        """[B, G] bool: group not yet acquired."""
        return ((~M).float() @ self._gmask_f.T) > 0

    @staticmethod
    def _root(state: State) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(state.x)[None], torch.from_numpy(state.m)[None]

    # -------------------------------------------------- batched primitives
    def _risk(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """r(s) for a batch of states -> float64 [B]."""
        return risk(predict_proba(self.model, X, M), self.risk_kind)

    def _nb(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        return self.sampler.neighbours_batch(X, M, k=self.n)

    def _children(self, X, M, nb, sidx, gidx) -> tuple[torch.Tensor, torch.Tensor]:
        """States s_sidx[p] + group gidx[p] with values from each neighbour row: [P * n, d]."""
        gm = self.gmask[gidx]                                      # [P, d]
        vals = self.pool[nb[sidx]]                                 # [P, n, d]
        Xc = torch.where(gm[:, None, :], vals, X[sidx][:, None, :])
        Mc = (M[sidx] | gm)[:, None, :].expand(-1, nb.shape[1], -1)
        d = X.shape[1]
        return Xc.reshape(-1, d), Mc.reshape(-1, d)

    def _eafter(self, X, M, sidx, gidx, nb=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-neighbour risks r(s + g(a)) for pairs (state sidx[p], group gidx[p]) -> ([P, n], nb)."""
        if nb is None:
            nb = self._nb(X, M)
        n = nb.shape[1]
        P = len(sidx)
        out = torch.empty(P, n, dtype=torch.float64)
        per = max(1, CHUNK_ROWS // n)
        for s in range(0, P, per):
            Xc, Mc = self._children(X, M, nb, sidx[s:s + per], gidx[s:s + per])
            out[s:s + per] = self._risk(Xc, Mc).view(-1, n)
        return out, nb

    def _pairs(self, M: torch.Tensor, allow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        allowed = self._free_batch(M) & allow[None, :]
        sidx, gidx = allowed.nonzero(as_tuple=True)                # row-major: groups ascending per state
        return sidx, gidx, allowed.any(1)

    def _greedy_value(self, X, M, steps: int, allow: torch.Tensor) -> torch.Tensor:
        """Expected terminal risk after `steps` greedy steps from each state (exact expectation) -> [B]."""
        B = X.shape[0]
        if steps == 0 or B == 0:
            return self._risk(X, M)
        sidx, gidx, has = self._pairs(M, allow)
        val = torch.empty(B, dtype=torch.float64)
        if has.any():
            E, nb = self._eafter(X, M, sidx, gidx)
            Ef = torch.full((B, self.n_groups), float("inf"), dtype=torch.float64)
            Ef[sidx, gidx] = E.mean(1)
            rows = has.nonzero(as_tuple=True)[0]
            if steps == 1:
                val[rows] = Ef[rows].min(1).values
            else:
                k = Ef[rows].argmin(1)                                 # first minimum = lowest group index
                Xc, Mc = self._children(X, M, nb, rows, k)
                val[rows] = self._greedy_value(Xc, Mc, steps - 1, allow).view(-1, nb.shape[1]).mean(1)
        if (~has).any():                                               # nothing left to acquire
            val[~has] = self._risk(X[~has], M[~has])
        return val

    # -------------------------------------------------- public API
    def risk_at(self, state: State) -> float:
        X, M = self._root(state)
        return float(self._risk(X, M)[0])

    def expected_after(self, state: State, candidates: Iterable[int] | None = None,
                       samples: bool = False) -> dict:
        """E_after(j | s) for candidate groups (default: all unobserved); per-neighbour risks if samples."""
        js = self.unobserved(state, candidates)
        if not js:
            return {}
        X, M = self._root(state)
        E, _ = self._eafter(X, M, torch.zeros(len(js), dtype=torch.long), torch.tensor(js))
        if samples:
            return {j: E[i] for i, j in enumerate(js)}
        means = E.mean(1)
        return {j: float(means[i]) for i, j in enumerate(js)}

    def one_step(self, state: State, candidates: Iterable[int] | None = None) -> dict[int, float]:
        """U_1(j | s) = r(s) - E_after(j | s). Cost 1 + |candidates| n evaluations."""
        ea = self.expected_after(state, candidates)
        if not ea:
            return {}
        r = self.risk_at(state)
        return {j: r - v for j, v in ea.items()}

    def greedy_context(self, state: State) -> StepContext:
        ea = self.expected_after(state)
        if not ea:
            raise ValueError("no unobserved group left")
        r = self.risk_at(state)
        order = sorted(ea, key=lambda j: (ea[j], j))
        return StepContext(self, state, r, ea, {j: r - v for j, v in ea.items()}, order)

    def rollout_values(self, state: State, js: Iterable[int], h: int, candidates: Iterable[int] | None = None,
                       return_samples: bool = False):
        """Q^{(h)}(s, j): expected risk after acquiring j then h - 1 greedy steps over `candidates`
        (default: all unobserved groups). Per-sample values are indexed by the neighbour rows a of s,
        which are the same for every j (paired samples)."""
        if h < 1:
            raise ValueError("h must be >= 1")
        js = self.unobserved(state, js)
        if not js:
            return ({}, {}) if return_samples else {}
        X, M = self._root(state)
        jt = torch.tensor(js)
        zeros = torch.zeros(len(js), dtype=torch.long)
        if h == 1:
            S, _ = self._eafter(X, M, zeros, jt)
        else:
            nb = self._nb(X, M)
            Xc, Mc = self._children(X, M, nb, zeros, jt)
            S = self._greedy_value(Xc, Mc, h - 1, self._allow(candidates)).view(len(js), -1)
        Q = {j: float(S[i].mean()) for i, j in enumerate(js)}
        if return_samples:
            return Q, {j: S[i] for i, j in enumerate(js)}
        return Q

    def rollout_value(self, state: State, j: int, h: int, candidates: Iterable[int] | None = None) -> float:
        return self.rollout_values(state, [j], h, candidates)[j]

    def _two_step_parts(self, state, js, candidates):
        """U_1(j | s) and the per-neighbour m_j(a) = max_k U_1(k | s + j(a)) (0 if nothing is left)."""
        X, M = self._root(state)
        jt = torch.tensor(js)
        zeros = torch.zeros(len(js), dtype=torch.long)
        E, nb = self._eafter(X, M, zeros, jt)
        r = self._risk(X, M)[0]
        Xc, Mc = self._children(X, M, nb, zeros, jt)
        rc = self._risk(Xc, Mc)
        m = (rc - self._greedy_value(Xc, Mc, 1, self._allow(candidates))).view(len(js), -1)
        return r - E.mean(1), m

    def two_step_values(self, state: State, js: Iterable[int], candidates: Iterable[int] | None = None):
        """Q_2(j) = U_1(j | s) + E_{x_j}[max_k U_1(k | s + j)] (expected two-step risk reduction)."""
        js = self.unobserved(state, js)
        if not js:
            return {}
        u, m = self._two_step_parts(state, js, candidates)
        return {j: float(u[i] + m[i].mean()) for i, j in enumerate(js)}

    def two_step_value(self, state: State, j: int, candidates: Iterable[int] | None = None) -> float:
        return self.two_step_values(state, [j], candidates)[j]

    def delta2(self, state: State, ctx: StepContext | None = None, candidates: Iterable[int] | None = None) -> float:
        """Exact two-step advantage Delta_2(s) = max_j Q_2(j) - Q_2(g) = Q^{(2)}(g) - min_j Q^{(2)}(j),
        computed with always-LA2 over all unobserved groups (or `candidates`, which must contain g)."""
        ctx = ctx or self.greedy_context(state)
        js = self.unobserved(state, candidates)
        if len(js) < 2:
            return 0.0
        Q = self.rollout_values(state, js, 2)
        return Q[ctx.g] - min(Q.values())

    # -------------------------------------------------- synergy and redundancy (Prop. 3)
    def _synergy_samples(self, state: State, pairs, u1: dict | None = None) -> dict:
        """(U_1(k | s + j(a)) - U_1(k | s))_+ for each neighbour a of s -> {(j, k): float64 [n]}."""
        free = set(self.unobserved(state))
        pairs = sorted({(int(j), int(k)) for j, k in pairs if j != k and j in free and k in free})
        if not pairs:
            return {}
        ks = sorted({k for _, k in pairs})
        if u1 is None or any(k not in u1 for k in ks):
            u1 = self.one_step(state, ks)
        js = sorted({j for j, _ in pairs})
        jpos = {j: i for i, j in enumerate(js)}
        X, M = self._root(state)
        nb = self._nb(X, M)
        Xc, Mc = self._children(X, M, nb, torch.zeros(len(js), dtype=torch.long), torch.tensor(js))
        n = nb.shape[1]
        rc = self._risk(Xc, Mc)                                        # [J * n]
        pj = np.array([jpos[j] for j, _ in pairs])
        pk = np.array([k for _, k in pairs])
        # child (j, a) has index jpos[j] * n + a; one (child, k) pair per pair and neighbour a
        sidx = torch.from_numpy((pj[:, None] * n + np.arange(n)[None, :]).reshape(-1))
        gidx = torch.from_numpy(np.repeat(pk, n))
        E, _ = self._eafter(Xc, Mc, sidx, gidx)
        u_child = (rc[sidx] - E.mean(1)).view(len(pairs), n)
        base = torch.tensor([u1[k] for k in pk], dtype=torch.float64)[:, None]
        pos = torch.clamp(u_child - base, min=0.0)
        return {p: pos[i] for i, p in enumerate(pairs)}

    def synergy(self, state: State, j: int, k: int) -> float:
        """E_{x_j}[(U_1(k | s + j) - U_1(k | s))_+]."""
        s = self._synergy_samples(state, [(j, k)])
        return float(s[(j, k)].mean()) if s else 0.0

    def synergy_pairs(self, state: State, pairs, u1: dict | None = None) -> dict:
        return {p: float(v.mean()) for p, v in self._synergy_samples(state, pairs, u1).items()}

    def synergy_term(self, state: State, g: int, pairs=None, u1: dict | None = None) -> float:
        """N(s) = max_{j != g} E_{x_j}[max_k (U_1(k | s + j) - U_1(k | s))_+], with (j, k) restricted to
        `pairs` (default: all ordered pairs of unobserved groups). Max over k INSIDE the expectation, as in
        Prop. 3; 0 if no active pair."""
        return self.synergy_term_detail(state, g, pairs, u1)[0]

    def synergy_term_detail(self, state: State, g: int, pairs=None, u1: dict | None = None):
        """(N(s), {(j, k): E_{x_j}[(U_1(k | s + j) - U_1(k | s))_+]}) over the active pairs with j != g."""
        if pairs is None:
            free = self.unobserved(state)
            pairs = [(j, k) for j in free for k in free if j != k]
        samples = self._synergy_samples(state, [(j, k) for j, k in pairs if j != g], u1)
        if not samples:
            return 0.0, {}
        by_j: dict[int, list[torch.Tensor]] = {}
        for (j, _), v in samples.items():
            by_j.setdefault(j, []).append(v)
        N = max(float(torch.stack(vs).max(0).values.mean()) for vs in by_j.values())
        return N, {p: float(v.mean()) for p, v in samples.items()}

    def two_step_table(self, state: State, ctx: StepContext | None = None) -> dict:
        """One always-LA2 pass (cost |U| n (1 + (|U| - 1) n)) giving
        - delta2: Delta_2(s), computed exactly as `delta2` (same batches);
        - gain:  [G, G] matrix, gain[j, k] = E_{x_j}[U_1(k | s + j)] (NaN where undefined);
        - q2:    {j: Q^{(2)}(s, j)}.
        Used by the gate-quality study (labels, submodularity-ratio estimate) without a second pass."""
        ctx = ctx or self.greedy_context(state)
        G = self.n_groups
        gain = np.full((G, G), np.nan)
        js = self.unobserved(state)
        if len(js) < 2:
            return {"delta2": 0.0, "gain": gain, "q2": {}}
        X, M = self._root(state)
        nb = self._nb(X, M)
        zeros = torch.zeros(len(js), dtype=torch.long)
        Xc, Mc = self._children(X, M, nb, zeros, torch.tensor(js))
        sidx, gidx, _ = self._pairs(Mc, self._allow(None))
        E, _ = self._eafter(Xc, Mc, sidx, gidx)
        B = Xc.shape[0]
        Ef = torch.full((B, G), float("inf"), dtype=torch.float64)
        Ef[sidx, gidx] = E.mean(1)
        n = nb.shape[1]
        q = Ef.min(1).values.view(len(js), n).mean(1)
        q2 = {j: float(q[i]) for i, j in enumerate(js)}
        rc = self._risk(Xc, Mc)
        u_child = (rc[:, None] - Ef).view(len(js), n, G)          # +inf -> -inf where k is not free
        mean_u = u_child.mean(1)
        for i, j in enumerate(js):
            for k in js:
                if k != j:
                    gain[j, k] = float(mean_u[i, k])
        return {"delta2": q2[ctx.g] - min(q2.values()), "gain": gain, "q2": q2}

    def redundancy(self, state: State, g: int, j2: int | None, candidates: Iterable[int] | None = None,
                   u1: dict | None = None) -> float:
        """R(s) = a_2 - E_{x_g}[max_k U_1(k | s + g)]. The inner max runs over `candidates` (default: all
        unobserved); a subset gives an upper bound on R, which keeps Delta_2 <= R + N valid."""
        if j2 is None:
            return 0.0
        if u1 is None or j2 not in u1:
            u1 = self.one_step(state, [j2])
        _, m = self._two_step_parts(state, [g], candidates)
        return float(u1[j2] - m[0].mean())
