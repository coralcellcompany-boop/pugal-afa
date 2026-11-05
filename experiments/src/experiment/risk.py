"""Terminal (Bayes) risk r(s) of a predictive distribution p(y | x_S); notes/method.md Sec. 1.

All three are minima of functions linear in p, hence concave in p (Fact 1):
- brier: 1 - sum_y p_y^2   (Bayes risk of the quadratic score; main risk in notes/experiments.md)
- 01:    1 - max_y p_y     (Bayes risk of the 0-1 loss)
- log:   -sum_y p_y log p_y (entropy; risk reduction = conditional mutual information)
"""

from __future__ import annotations

import torch

KINDS = ("brier", "01", "log")


def risk(p: torch.Tensor, kind: str) -> torch.Tensor:
    """p [..., C] -> risk [...]. Computed in the dtype of p (callers pass float64)."""
    if kind == "brier":
        return 1.0 - (p * p).sum(-1)
    if kind == "01":
        return 1.0 - p.max(-1).values
    if kind == "log":
        return -torch.xlogy(p, p).sum(-1)   # 0 log 0 = 0
    raise ValueError(f"unknown risk kind {kind!r}; expected one of {KINDS}")
