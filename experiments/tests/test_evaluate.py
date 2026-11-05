"""evaluate.run_policy: metrics, exact compute accounting, determinism, gate (score, label) pairs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiment.evaluate import auroc, macro_f1, run_policy
from experiment.gates import MarginGate
from experiment.policies import GreedyPolicy, LookaheadPolicy, PUGALPolicy
from experiment.gates import ConstantGate
from experiment.model import train_classifier
from experiment.sampler import KNNSampler
from experiment.utility import Utility

from helpers import make_grouped_data, trained_setup


def test_auroc_by_hand():
    assert auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert auroc([1, 1, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.5)     # ties count one half
    assert math.isnan(auroc([0.1, 0.2], [1, 1]))


def test_macro_f1_by_hand():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0, 1, 1, 1, 0])
    # class 0: tp=1 fp=1 fn=1 -> 0.5; class 1: tp=2 fp=1 fn=1 -> 2/3
    assert macro_f1(y, p, 2) == pytest.approx((0.5 + 2 / 3) / 2)


def test_run_policy_greedy_metrics_and_exact_evals():
    data, model, sampler = trained_setup("grouped")
    eng = Utility(model, sampler, data.groups, "brier")
    res = run_policy(GreedyPolicy(eng), data, model, sampler, budget=3, max_instances=20)
    assert res["n_instances"] == 20
    assert len(res["acc"]) == 3
    assert res["acc_auc"] == pytest.approx(np.mean(res["acc"]))
    assert res["acc_t1"] == res["acc"][0]
    n = sampler.n
    # greedy: (1 + 4n) + (1 + 3n) + (1 + 2n) rows per instance; final predictions are not counted
    assert res["evals_per_inst"] == pytest.approx(3 + 9 * n)
    assert res["open_rate"] == 0.0
    assert res["sec_per_inst"] > 0
    assert "f1" not in res
    assert all(0.0 <= a <= 1.0 for a in res["acc"])


def test_run_policy_deterministic():
    data, model, sampler = trained_setup("grouped")
    eng = Utility(model, sampler, data.groups, "brier")
    a = run_policy(PUGALPolicy(eng, MarginGate(0.05)), data, model, sampler, budget=3, max_instances=15)
    b = run_policy(PUGALPolicy(eng, MarginGate(0.05)), data, model, sampler, budget=3, max_instances=15)
    for k in a:
        if k != "sec_per_inst":
            assert a[k] == b[k], k


def test_run_policy_lookahead_open_rate_and_f1():
    data = make_grouped_data(n=1500, seed=1, imbalanced=True)
    model = train_classifier(data, seed=1, max_epochs=10)
    sampler = KNNSampler(data.X_train, n=6, seed=1)
    eng = Utility(model, sampler, data.groups, "brier")
    res = run_policy(LookaheadPolicy(eng, h=2), data, model, sampler, budget=3, max_instances=10)
    assert res["open_rate"] == 1.0
    assert len(res["f1"]) == 3 and res["f1_auc"] == pytest.approx(np.mean(res["f1"]))
    assert res["f1_t2"] == res["f1"][1]
    pug = run_policy(PUGALPolicy(eng, ConstantGate(True)), data, model, sampler, budget=3, max_instances=10)
    assert pug["open_rate"] == pytest.approx(2 / 3)   # the last step never opens (rollout = greedy)


def test_run_policy_gate_pairs():
    data, model, sampler = trained_setup("grouped")
    eng = Utility(model, sampler, data.groups, "brier")
    res = run_policy(GreedyPolicy(eng), data, model, sampler, budget=3, max_instances=8,
                     gate=MarginGate(0.05), gate_eps=0.01)
    pairs = res["gate_pairs"]
    assert len(pairs) == 8 * 2                     # states with >= 2 acquisitions left
    for score, label, d2 in pairs:
        assert d2 >= -1e-9
        assert label == (d2 > 0.01)
    assert "gate_auroc" in res
    assert math.isnan(res["gate_auroc"]) or 0.0 <= res["gate_auroc"] <= 1.0


def test_run_policy_dev_rate_only_for_tie_variants():
    data = make_grouped_data(n=1500, seed=1)
    model = train_classifier(data, seed=1, max_epochs=10)
    sampler = KNNSampler(data.X_train, n=6, seed=1)
    eng = Utility(model, sampler, data.groups, "brier")
    plain = run_policy(LookaheadPolicy(eng, h=2), data, model, sampler, budget=3, max_instances=10)
    assert "dev_rate" not in plain
    tie = run_policy(LookaheadPolicy(eng, h=2, tie="ci"), data, model, sampler, budget=3, max_instances=10)
    assert 0.0 <= tie["dev_rate"] <= tie["open_rate"] <= 2 / 3
