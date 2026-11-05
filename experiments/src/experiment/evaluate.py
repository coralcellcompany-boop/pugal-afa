"""Run a policy on the test split and compute the metrics of notes/experiments.md (DESIGN.md `evaluate.py`).

Compute accounting: `evals_per_inst` counts classifier rows evaluated inside policy.select (model.n_evals
deltas), and `sec_per_inst` is the wall time spent in policy.select. The predictions that score accuracy
after each step (one row per instance and step) are not counted: every method pays them equally.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from experiment.model import predict_proba
from experiment.utility import State, Utility

if TYPE_CHECKING:  # pragma: no cover
    from experiment.data import AFAData


def auroc(scores, labels) -> float:
    """Area under the ROC curve (Mann-Whitney, average ranks for ties); NaN if one class is absent."""
    s = np.asarray(scores, dtype=np.float64)
    lab = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(lab.sum()), int((~lab).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[lab].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    """Unweighted mean of per-class F1 over classes present in y_true or y_pred (sklearn's 'macro')."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = []
    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def run_policy(policy, data: "AFAData", model, sampler, budget: int, *, max_instances: int | None = None,
               gate=None, gate_eps: float = 0.01, engine: Utility | None = None,
               risk_kind: str = "brier") -> dict:
    """Run every test instance (or the first max_instances) for `budget` acquisitions.

    Returns acc (list over t = 1..budget), acc_t<t>, acc_auc (mean of acc), and f1 / f1_t<t> / f1_auc if
    data.imbalanced; evals_per_inst; sec_per_inst; open_rate (fraction of steps with policy.last_opened);
    n_instances; dev_rate (steps where policy.last_deviated) for policies that expose it. If `gate` is given, also gate_pairs = [(gate score, Delta_2 > gate_eps, Delta_2)] on the
    states greedy visits that still have >= 2 acquisitions left (Delta_2 = exact always-LA2 advantage) and
    gate_auroc. The gate pass uses `engine` (default: policy.engine, else a new brier engine) and is
    excluded from the compute metrics.
    """
    X = np.asarray(data.X_test, np.float32)
    y = np.asarray(data.y_test)
    if max_instances is not None:
        X, y = X[:max_instances], y[:max_instances]
    groups = data.groups
    N, d = X.shape
    if budget > len(groups):
        raise ValueError(f"budget {budget} exceeds the number of groups {len(groups)}")

    masks = np.zeros((budget, N, d), dtype=bool)
    evals, secs, opened, deviated = 0, 0.0, 0, 0
    track_dev = hasattr(policy, "last_deviated")
    for i in range(N):
        st = State.empty(d, budget)
        for t in range(budget):
            e0, t0 = model.n_evals, time.perf_counter()
            g = policy.select(st)
            secs += time.perf_counter() - t0
            evals += model.n_evals - e0
            opened += bool(getattr(policy, "last_opened", False))
            deviated += bool(getattr(policy, "last_deviated", False))
            st = st.acquire(g, X[i], groups)             # raises if g was already acquired
            masks[t, i] = st.m

    res: dict = {"n_instances": N}
    acc, f1 = [], []
    for t in range(budget):
        M = torch.from_numpy(masks[t])
        pred = predict_proba(model, torch.from_numpy(X) * M, M).argmax(-1).numpy()
        acc.append(float((pred == y).mean()))
        if data.imbalanced:
            f1.append(macro_f1(y, pred, data.n_classes))
    res["acc"] = acc
    res.update({f"acc_t{t + 1}": a for t, a in enumerate(acc)})
    res["acc_auc"] = float(np.mean(acc))
    if data.imbalanced:
        res["f1"] = f1
        res.update({f"f1_t{t + 1}": v for t, v in enumerate(f1)})
        res["f1_auc"] = float(np.mean(f1))
    res["evals_per_inst"] = evals / N
    res["sec_per_inst"] = secs / N
    res["open_rate"] = opened / (N * budget)
    if track_dev:                                     # X4: share of steps where the choice != g(s)
        res["dev_rate"] = deviated / (N * budget)

    if gate is not None:
        eng = engine or getattr(policy, "engine", None) or Utility(model, sampler, groups, risk_kind)
        pairs = []
        for i in range(N):
            st = State.empty(d, budget)
            while st.budget >= 2 and len(eng.unobserved(st)) >= 2:
                ctx = eng.greedy_context(st)
                score = float(gate.score(st, ctx))
                d2 = float(eng.delta2(st, ctx=ctx))
                pairs.append((score, bool(d2 > gate_eps), d2))
                st = st.acquire(ctx.g, X[i], groups)
        res["gate_pairs"] = pairs
        res["gate_auroc"] = auroc([p[0] for p in pairs], [p[1] for p in pairs])
    return res
