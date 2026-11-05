"""Acquisition policies (DESIGN.md `policies.py`; PUGAL = notes/method.md Sec. 3).

Every policy implements select(state) -> group index, never returns an acquired group, and exposes the
bookkeeping attributes last_opened (bool) and last_gate_score (float | None). Model-based policies take a
`Utility` engine (model, sampler, groups, risk kind), so e.g. "greedy with log loss" is
GreedyPolicy(Utility(..., risk_kind="log")). Ties are broken by the lowest group index everywhere.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from experiment.model import predict_proba
from experiment.utility import State, StepContext, Utility, _first_argmin


def _free_groups(state: State, groups: list[list[int]]) -> list[int]:
    return [g for g, feats in enumerate(groups) if not state.m[list(feats)].all()]


class Policy:
    last_opened: bool = False
    last_gate_score: float | None = None

    def select(self, state: State) -> int:
        raise NotImplementedError


# ------------------------------------------------------------------ simple baselines

class RandomPolicy(Policy):
    """Uniform over unacquired groups; deterministic given the seed and the order of calls."""

    def __init__(self, groups: list[list[int]], seed: int = 0):
        self.groups = [list(g) for g in groups]
        self.rng = np.random.default_rng(seed)

    def select(self, state):
        free = _free_groups(state, self.groups)
        return int(free[self.rng.integers(len(free))])


class StaticPolicy(Policy):
    """Fixed order; skips acquired groups."""

    def __init__(self, order: list[int], groups: list[list[int]]):
        self.order = [int(g) for g in order]
        self.groups = [list(g) for g in groups]

    def select(self, state):
        free = set(_free_groups(state, self.groups))
        for g in self.order:
            if g in free:
                return g
        return min(free)        # order exhausted: lowest free index


def static_greedy_order(engine: Utility, X_rows: np.ndarray, budget: int | None = None) -> list[int]:
    """Non-adaptive greedy order: step t adds the group with the highest mean U_1 over the rows, given the
    groups already in the order (each row observes its own values). Use validation rows."""
    budget = engine.n_groups if budget is None else min(budget, engine.n_groups)
    X_rows = np.asarray(X_rows, np.float32)
    order: list[int] = []
    for _ in range(budget):
        m = np.zeros(engine.d, bool)
        for g in order:
            m[engine.groups[g]] = True
        free = [g for g in range(engine.n_groups) if g not in order]
        tot = np.zeros(len(free))
        for x in X_rows:
            ea = engine.expected_after(State(x, m), free)
            tot += np.array([ea[g] for g in free])
        order.append(free[int(np.argmin(tot))])         # argmin mean E_after = argmax mean U_1
    return order


class GreedyPolicy(Policy):
    """g(s) = argmax_j U_1(j | s). Cost per state: 1 + |U| n evaluations."""

    def __init__(self, engine: Utility):
        self.engine = engine
        self.last_ctx: StepContext | None = None

    def select(self, state):
        ctx = self.engine.greedy_context(state)
        self.last_ctx, self.last_opened, self.last_gate_score = ctx, False, None
        return ctx.g


# ------------------------------------------------------------------ candidate rules

class AllCandidates:
    needs_ctx = False

    def __call__(self, state: State, ctx: StepContext | None, engine: Utility) -> list[int]:
        return engine.unobserved(state)


class TopMCandidates:
    """Top-m groups by U_1(. | s), then the members of active flagged pairs, best pair first, until the set
    has `cap` groups (no cap if None).

    pair_scores ranks the pairs: a dict {(j, k): score} (e.g. the offline screen averages), "online" (the
    per-pair synergy the gate stored in ctx.extra["pair_syn"] at this state; only pairs with score >
    min_score count, and flagged_pairs=None then means every pair), or None (pair index order).
    """
    needs_ctx = True

    def __init__(self, m: int, flagged_pairs=None, cap: int | None = None, pair_scores=None,
                 min_score: float = 0.0):
        self.m, self.cap, self.pair_scores, self.min_score = m, cap, pair_scores, min_score
        self.flagged_pairs = None if flagged_pairs is None else set(flagged_pairs)

    def __call__(self, state, ctx, engine) -> list[int]:
        cands = list(ctx.order[:self.m])
        free = set(ctx.order)
        if self.pair_scores == "online":
            online = ctx.extra.get("pair_syn", {})
            pool = online if self.flagged_pairs is None else \
                {p: v for p, v in online.items() if p in self.flagged_pairs}
            scores = {p: v for p, v in pool.items() if v > self.min_score}
            pairs = list(scores)
        else:
            scores = self.pair_scores or {}
            pairs = list(self.flagged_pairs or [])
        active = [p for p in pairs if p[0] in free and p[1] in free]
        active.sort(key=lambda p: (-float(scores.get(p, 0.0)), p))
        for pair in active:
            for f in pair:
                if f in cands:
                    continue
                if self.cap is not None and len(cands) >= self.cap:
                    return sorted(cands)
                cands.append(f)
        return sorted(cands)


def _depth(h: int, budget: int | None) -> int:
    return h if budget is None else max(1, min(h, budget))


class LookaheadPolicy(Policy):
    """Always-on truncated rollout: argmin_{j in C(s)} Q^{(h)}(s, j), h capped by the remaining budget.
    h = 2 with AllCandidates is always-LA2. Cost (h = 2, all candidates): |U| n (|U| - 1) n evaluations.

    tie (exploratory X4, post hoc) decides between greedy g(s) and the argmin j*:
    - "index" (pre-registered LA2): plain argmin, ties -> smallest group index;
    - "greedy": keep g unless Q(g) - Q(j*) > TIE_TOL (ties -> greedy);
    - "ci": keep g unless Q(g) - Q(j*) exceeds the paired 95% CI half-width on the shared samples
      (= PUGALPolicy with an always-open gate, all candidates, h = 2 and the confidence fallback).
    For "greedy" and "ci" the greedy step is computed too (1 + |U| n evaluations), g is played without a
    rollout when at most one acquisition or one group is left, and last_deviated records j* != g.
    """
    TIE_TOL = 1e-9
    TIES = ("index", "greedy", "ci")

    def __init__(self, engine: Utility, h: int = 2, candidate_rule=None, tie: str = "index", z: float = 1.96):
        if tie not in self.TIES:
            raise ValueError(f"tie must be one of {self.TIES}")
        self.engine, self.h, self.tie, self.z = engine, h, tie, z
        self.candidate_rule = candidate_rule or AllCandidates()
        self.last_Q: dict | None = None
        if tie != "index":
            self.last_deviated = False

    def select(self, state):
        if self.tie == "index":
            ctx = self.engine.greedy_context(state) if self.candidate_rule.needs_ctx else None
            C = self.candidate_rule(state, ctx, self.engine)
            Q = self.engine.rollout_values(state, C, _depth(self.h, state.budget))
            self.last_Q, self.last_opened, self.last_gate_score = Q, True, None
            return _first_argmin(Q)
        ctx = self.engine.greedy_context(state)
        g = ctx.g
        self.last_deviated, self.last_gate_score = False, None
        if (state.budget is not None and state.budget < 2) or len(ctx.order) < 2:
            self.last_Q, self.last_opened = None, False
            return g
        C = sorted(set(self.candidate_rule(state, ctx, self.engine)) | {g})
        Q, S = self.engine.rollout_values(state, C, _depth(self.h, state.budget), return_samples=True)
        self.last_Q, self.last_opened = Q, True
        best = _first_argmin(Q)
        if best == g:
            return g
        if self.tie == "greedy":
            need = self.TIE_TOL
        else:
            n = S[g].numel()
            need = math.inf if n < 2 else self.z * float((S[g] - S[best]).std()) / math.sqrt(n)
        if not (Q[g] - Q[best]) > need:
            return g
        self.last_deviated = True
        return best


# ------------------------------------------------------------------ PUGAL

class PUGALPolicy(Policy):
    """Gated rollout with an adaptive horizon (notes/method.md Sec. 3).

    1. Greedy step: U_1(. | s) for all unobserved groups (StepContext).
    2. If at most one acquisition is left (rollout = greedy) or one group is left, play g(s) without
       evaluating the gate (last_opened = False).
    3. Gate score once; closed -> g(s).
    4. Open -> C(s) = candidate_rule (default: top-m by U_1 plus members of active pairs flagged by the gate)
       plus g(s). Q^{(h)} for C(s) from h = h0; while h < min(h_max, budget) and the CI of g(s) overlaps the
       CI of the best non-greedy candidate, h += 1. Pick argmin Q^{(h)}. Fallback (conf_fallback): keep g(s)
       unless the chosen candidate beats it by more than the CI half-width (and by more than eps).

    CIs: per-sample rollout values are indexed by the neighbour rows a of s, shared by every candidate
    (common random numbers). ci="paired" (default) uses the CI of the paired difference Q(g) - Q(j):
    overlap iff |mean diff| <= z sd(diff) / sqrt(n); fallback threshold = the same half-width.
    ci="separate" uses each candidate's own interval: overlap iff |Q(g) - Q(j)| <= hw_g + hw_j; fallback
    threshold = hw_j (the literal reading of "its confidence half-width").
    With conf_fallback=False and eps=0 the choice is plain argmin Q^{(h)} (equals LookaheadPolicy).
    """

    def __init__(self, engine: Utility, gate, h0: int = 2, h_max: int = 3, m: int = 8, eps: float = 0.0,
                 conf_fallback: bool = True, candidate_rule=None, z: float = 1.96, ci: str = "paired"):
        if ci not in ("paired", "separate"):
            raise ValueError("ci must be 'paired' or 'separate'")
        self.engine, self.gate = engine, gate
        self.h0, self.h_max, self.m, self.eps = h0, max(h_max, h0), m, eps
        self.conf_fallback, self.z, self.ci = conf_fallback, z, ci
        self.candidate_rule = candidate_rule or TopMCandidates(m, getattr(gate, "flagged_pairs", None))
        self.last_ctx: StepContext | None = None
        self.last_h: int | None = None
        self.last_fallback = False

    def _halfwidths(self, Sg: torch.Tensor, Sj: torch.Tensor) -> tuple[float, float]:
        """(overlap threshold, fallback threshold) for greedy g vs candidate j."""
        n = Sg.numel()
        if n < 2:
            return math.inf, math.inf
        if self.ci == "paired":
            hw = self.z * float((Sg - Sj).std()) / math.sqrt(n)
            return hw, hw
        hg = self.z * float(Sg.std()) / math.sqrt(n)
        hj = self.z * float(Sj.std()) / math.sqrt(n)
        return hg + hj, hj

    def select(self, state):
        eng = self.engine
        ctx = eng.greedy_context(state)
        self.last_ctx, self.last_h, self.last_fallback = ctx, None, False
        self.last_opened, self.last_gate_score = False, None
        g = ctx.g
        if (state.budget is not None and state.budget < 2) or len(ctx.order) < 2:
            return g
        score = self.gate.score(state, ctx)
        opened = self.gate.open_from_score(score)
        self.last_gate_score, self.last_opened = float(score), bool(opened)
        if not opened:
            return g

        C = sorted(set(self.candidate_rule(state, ctx, eng)) | {g})
        if len(C) < 2:
            return g
        h = _depth(self.h0, state.budget)
        h_cap = _depth(self.h_max, state.budget)
        while True:
            Q, S = eng.rollout_values(state, C, h, return_samples=True)
            best = _first_argmin(Q)
            rival = best if best != g else _first_argmin({j: Q[j] for j in C if j != g})
            hw_overlap, hw_fallback = self._halfwidths(S[g], S[rival])
            if h < h_cap and abs(Q[g] - Q[rival]) <= hw_overlap:
                h += 1
                continue
            break
        self.last_h = h
        if best == g:
            return g
        if self.conf_fallback or self.eps > 0:
            need = max(self.eps, hw_fallback if self.conf_fallback else -math.inf)
            if not (Q[g] - Q[best]) > need:
                self.last_fallback = True
                return g
        return best


# ------------------------------------------------------------------ AACO

class AACOPolicy(Policy):
    """AACO re-implementation (valancius2023acquisition) under a hard budget, as specified in DESIGN.md.

    At s: the k_nn nearest pool rows (with their labels) on the observed features; candidate subsets O of
    unacquired groups of size min(budget, |U|) (all of them if there are at most n_subsets, otherwise
    n_subsets distinct random ones); score(O) = mean over neighbours of the classifier loss on the
    neighbour's label, with x_S from the instance and x_O from the neighbour; no cost term (hard budget).
    Pick the best subset; acquire its member with the best single-group score. Cost per state:
    (#subsets + |O*|) * k_nn evaluations (|O*| = 1: no second stage). loss in {"ce", "01", "brier"}.
    """

    def __init__(self, engine: Utility, k_nn: int = 5, n_subsets: int = 1000, seed: int = 0, loss: str = "ce"):
        if loss not in ("ce", "01", "brier"):
            raise ValueError("loss must be 'ce', '01' or 'brier'")
        if getattr(engine.sampler, "y_pool", None) is None:
            raise ValueError("AACO needs labels: build KNNSampler(..., y_pool=y_train)")
        self.engine, self.k_nn, self.n_subsets, self.loss = engine, k_nn, n_subsets, loss
        self.rng = np.random.default_rng(seed)
        self.y_pool = torch.as_tensor(engine.sampler.y_pool, dtype=torch.long)

    def _subsets(self, free: list[int], size: int) -> list[tuple[int, ...]]:
        if math.comb(len(free), size) <= self.n_subsets:
            return list(itertools.combinations(free, size))
        seen: dict[tuple[int, ...], None] = {}
        arr = np.array(free)
        while len(seen) < self.n_subsets:
            seen.setdefault(tuple(sorted(arr[self.rng.choice(len(arr), size, replace=False)].tolist())), None)
        return list(seen)

    def _scores(self, x, m, nb, subsets) -> torch.Tensor:
        eng = self.engine
        ind = np.zeros((len(subsets), eng.n_groups), np.float32)
        for q, sub in enumerate(subsets):
            ind[q, list(sub)] = 1.0
        S = torch.from_numpy(ind @ eng.gmask_np.astype(np.float32) > 0)        # [Q, d]
        vals = eng.pool[nb]                                                     # [k, d]
        Xr = torch.where(S[:, None, :], vals[None, :, :], x[None, None, :])     # [Q, k, d]
        Mr = (m[None, :] | S)[:, None, :].expand(-1, len(nb), -1)
        p = predict_proba(eng.model, Xr, Mr)                                     # [Q, k, C]
        y = self.y_pool[nb]
        py = p.gather(-1, y[None, :, None].expand(len(subsets), -1, 1)).squeeze(-1)
        if self.loss == "ce":
            loss = -torch.log(py.clamp_min(1e-12))
        elif self.loss == "01":
            loss = (p.argmax(-1) != y[None, :]).double()
        else:
            loss = (p * p).sum(-1) - 2 * py + 1
        return loss.mean(1)

    def select(self, state):
        eng = self.engine
        free = eng.unobserved(state)
        self.last_opened, self.last_gate_score = False, None
        if len(free) == 1:
            return free[0]
        size = min(1 if state.budget is None else max(1, state.budget), len(free))
        x = torch.from_numpy(state.x)
        m = torch.from_numpy(state.m)
        nb = eng.sampler.neighbours_batch(x[None], m[None], k=self.k_nn)[0]
        subsets = self._subsets(free, size)
        best = subsets[int(torch.argmin(self._scores(x, m, nb, subsets)))]
        if len(best) == 1:
            return best[0]
        singles = [(j,) for j in best]
        return best[int(torch.argmin(self._scores(x, m, nb, singles)))]
