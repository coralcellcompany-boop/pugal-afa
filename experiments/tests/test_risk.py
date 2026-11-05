"""Bayes-risk values checked by hand."""

from __future__ import annotations

import math

import pytest
import torch

from experiment.risk import risk


def test_uniform_binary():
    p = torch.tensor([0.5, 0.5])
    assert risk(p, "brier").item() == pytest.approx(0.5)
    assert risk(p, "01").item() == pytest.approx(0.5)
    assert risk(p, "log").item() == pytest.approx(math.log(2))


def test_certain_prediction_has_zero_risk_and_no_nan():
    p = torch.tensor([1.0, 0.0, 0.0])
    for kind in ("brier", "01", "log"):
        v = risk(p, kind)
        assert not torch.isnan(v)
        assert v.item() == pytest.approx(0.0, abs=1e-12)


def test_three_class_values():
    p = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    assert risk(p, "brier").item() == pytest.approx(1 - (0.49 + 0.04 + 0.01))
    assert risk(p, "01").item() == pytest.approx(0.3)
    expected = -(0.7 * math.log(0.7) + 0.2 * math.log(0.2) + 0.1 * math.log(0.1))
    assert risk(p, "log").item() == pytest.approx(expected)


def test_batched_shape():
    p = torch.softmax(torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(0)), -1)
    for kind in ("brier", "01", "log"):
        v = risk(p, kind)
        assert v.shape == (2, 5)
        assert torch.all(v >= 0)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        risk(torch.tensor([0.5, 0.5]), "hinge")
