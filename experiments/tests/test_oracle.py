"""Tests for experiment.oracle: exact SynMix posterior and exact conditional sampling (exploratory X1)."""

from __future__ import annotations

import numpy as np
import torch

from experiment.data import SYNMIX_LABEL_NOISE, load_dataset
from experiment.model import predict_proba
from experiment.oracle import OracleSampler, SynMixOracle

DATA = None


def _data(name="synmix_0.5", seed=0):
    from pathlib import Path
    return load_dataset(name, seed, Path(__file__).resolve().parents[1] / "data")


def _raw_to_norm(data, raw):
    return ((raw - data.norm_mu) / data.norm_sd).astype(np.float32)


def test_norm_stats_reproduce_normalised_train():
    data = _data()
    assert data.norm_mu is not None and data.norm_sd is not None
    raw = data.X_train * data.norm_sd + data.norm_mu
    # z (feature 0) is 0/1 in raw space
    assert np.allclose(np.round(raw[:, 0]), raw[:, 0], atol=1e-4)


def test_posterior_hand_cases():
    data = _data()
    orc = SynMixOracle(data, rho=0.5)
    eta = SYNMIX_LABEL_NOISE
    d = data.X_train.shape[1]
    raw = np.zeros((4, d))
    m = np.zeros((4, d), bool)
    # 0: nothing observed -> 0.5
    # 1: z = 0, x3 > 0 observed -> 1 - eta
    raw[1, 0], raw[1, 3] = 0, 1.0; m[1, [0, 3]] = True
    # 2: z = 1, x1 > 0, x2 > 0 -> XOR = 0 -> eta
    raw[2, 0], raw[2, 1], raw[2, 2] = 1, 1.0, 2.0; m[2, [0, 1, 2]] = True
    # 3: z unknown (rho = 0.5), x3 < 0, x1 > 0, x2 < 0 -> p_clean = .5 * 1 + .5 * 0 = .5
    raw[3, 1], raw[3, 2], raw[3, 3] = 1.0, -1.0, -1.0; m[3, [1, 2, 3]] = True
    X = torch.from_numpy(_raw_to_norm(data, raw))
    p = predict_proba(orc, X, torch.from_numpy(m))[:, 1].numpy()
    expect = np.array([0.5, 1 - eta, eta, 0.5])
    assert np.allclose(p, expect, atol=1e-6)
    assert orc.n_evals == 4


def test_posterior_ignores_unobserved_values():
    data = _data()
    orc = SynMixOracle(data, rho=0.5)
    X = torch.from_numpy(data.X_test[:50])
    M = torch.zeros_like(X, dtype=torch.bool)
    M[:, 3] = True
    X2 = X.clone()
    X2[:, :3] = -X2[:, :3]
    assert torch.allclose(predict_proba(orc, X, M), predict_proba(orc, X2, M))


def test_posterior_matches_labels_when_fully_observed():
    data = _data()
    orc = SynMixOracle(data, rho=0.5)
    X = torch.from_numpy(data.X_test)
    p = predict_proba(orc, X, torch.ones_like(X, dtype=torch.bool))[:, 1].numpy()
    acc = ((p > 0.5) == data.y_test).mean()
    assert abs(acc - (1 - SYNMIX_LABEL_NOISE)) < 0.02   # Bayes accuracy


def test_sampler_deterministic_and_marginal():
    data = _data()
    s = OracleSampler(data, rho=0.5, n=64, seed=0)
    X = torch.from_numpy(data.X_test[:200])
    M = torch.from_numpy(np.random.default_rng(0).random(X.shape) < 0.3)
    a, b = s.neighbours_batch(X, M), s.neighbours_batch(X, M)
    assert a.shape == (200, 64) and torch.equal(a, b)
    assert a.min() >= 0 and a.max() < len(s.pool)
    # different states get different draws
    assert not torch.equal(a[0], a[1])
    # pool rows are fresh draws from the generator, normalised with the train statistics
    raw = s.pool.numpy() * data.norm_sd + data.norm_mu
    assert abs(raw[:, 0].mean() - 0.5) < 0.03 and abs(raw[:, 1].mean()) < 0.05 and abs(raw[:, 1].std() - 1) < 0.05
    # draws are spread over the pool (not a few fixed rows)
    assert len(torch.unique(a)) > 0.5 * len(s.pool)
    assert len(s.pool_index) == 0


def test_gate_detail_metrics_hand_case():
    from experiment.core import gate_detail_metrics
    det = {"step": np.array([0, 0, 1, 1]), "a1": np.array([0.4, 0.0, 0.2, 0.1]),
           "a2": np.array([0.0, 0.0, 0.18, np.nan]), "R": np.array([0.0, 0.1, 0.3, 0.0]),
           "nhat": np.array([0.0, 0.5, 0.1, 0.0])}
    labels = np.array([False, True, True, False])
    scores = {k: np.array([0.0, 1.0, 0.5, 0.2]) for k in ("syn", "learn", "margin")}

    class L:
        w = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0])

    out = gate_detail_metrics(det, scores, labels, L())
    # relative margins: 1.0, 0 (a1 = 0), 0.1, 1.0 -> scores -1, 0, -0.1, -1 -> perfect separation
    assert out["auroc_relmargin"] == 1.0
    assert out["auroc_R"] == 1.0 and out["auroc_nhat"] == 1.0
    assert out["mean_a1_pos"] == 0.1 and abs(out["mean_a1_neg"] - 0.25) < 1e-12
    assert out["frac_positive_t0"] == 0.5 and out["auroc_syn_t1"] == 1.0
    assert out["learn_w_nhat"] == 5.0 and "learn_w_bias" not in out
