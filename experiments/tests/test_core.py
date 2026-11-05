"""core.run_experiment: metric names, every policy kind, determinism, caches, tuning on validation only,
random_matched dependency, gate-quality metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import experiment.core as core
from experiment.data import load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
COMMON = dict(risk="brier", n_neighbors=8, pool_size=5000, h0=2, h_max=2, m=2, cand_cap=3, conf_fallback=True)
GRID = {"eps": [0.005, 0.02], "screen_quantile": [0.5], "m": [1, 2]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(core, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(core, "CONFIG_DIR", tmp_path / "configs")
    monkeypatch.setattr(core, "TUNE_MAX_VAL", 25)
    monkeypatch.setattr(core, "SCREEN_STATES", 12)
    monkeypatch.setattr(core, "GATE_TRAIN_STATES", 12)
    monkeypatch.setattr(core, "GATEQ_MAX_TEST", 10)
    monkeypatch.setattr(core, "STATIC_ROWS", 15)
    core.clear_memory_caches()
    yield tmp_path
    core.clear_memory_caches()


def cfg(method, policy, budget=2, test_cap=12, dataset="toy_xor", **kw):
    c = {"dataset": dataset, "method": method, "budget": budget, **COMMON, "policy": policy, "test_cap": test_cap}
    c.update(kw)
    return c


def pugal_cfg(**kw):
    return cfg("pugal", "pugal", gate="synergy", hparams=dict(GRID), **kw)


# ------------------------------------------------------------------ names and policies

def test_metric_names_greedy(env):
    m = core.run_experiment(cfg("greedy", "greedy"), 0)
    for k in ("acc_t1", "acc_t2", "acc_auc", "evals_per_inst", "sec_per_inst", "open_rate", "classifier_val_acc"):
        assert k in m and isinstance(m[k], float), k
    assert "acc_t3" not in m and "f1_auc" not in m
    assert m["open_rate"] == 0.0
    assert m["acc_auc"] == pytest.approx((m["acc_t1"] + m["acc_t2"]) / 2)
    assert m["classifier_val_acc"] > 0.8
    assert (env / "runs" / "models" / "toy_xor_seed0.pt").exists()


@pytest.mark.skipif(not (DATA_DIR / "actg175.csv").exists(), reason="actg175 raw file missing")
def test_metric_names_imbalanced_f1(env):
    m = core.run_experiment(cfg("greedy", "greedy", dataset="actg175", budget=3, test_cap=8), 0)
    for k in ("f1_t1", "f1_t2", "f1_t3", "f1_auc", "acc_t3"):
        assert k in m, k


def test_budget_override(env):
    m = core.run_experiment(cfg("greedy", "greedy", budget=3, budget_override=1), 0)
    assert "acc_t1" in m and "acc_t2" not in m


@pytest.mark.parametrize("c", [
    cfg("random", "random"),
    cfg("static", "static"),
    cfg("greedy_log", "greedy", risk="log"),
    cfg("la2", "lookahead", h=2, candidates="all"),
    cfg("la3", "lookahead", h=3, candidates="topm_flagged", n_neighbors=4),
    cfg("aaco", "aaco", k_nn=5, n_subsets=20),
    cfg("pugal_margin", "pugal", gate="margin", hparams=dict(GRID)),
    cfg("pugal_learn", "pugal", gate="learned", hparams=dict(GRID)),
    cfg("pugal_noscreen", "pugal", gate="synergy", screen="all_pairs", hparams=dict(GRID)),
    cfg("pugal_nosyn", "pugal", gate="synergy", gate_terms="redundancy", hparams=dict(GRID)),
    cfg("pugal_h3", "pugal", gate="synergy", h_max=3, hparams=dict(GRID)),
    cfg("pugal_loss01", "pugal", gate="synergy", risk="01", hparams=dict(GRID)),
], ids=lambda c: c["method"])
def test_every_policy_kind_runs(env, c):
    m = core.run_experiment(c, 0)
    assert 0.0 <= m["acc_auc"] <= 1.0 and m["evals_per_inst"] >= 0
    if c["policy"] == "lookahead":
        assert m["open_rate"] == 1.0
    if c["policy"] in ("random", "static", "greedy", "aaco"):
        assert m["open_rate"] == 0.0


def test_unknown_policy_raises(env):
    with pytest.raises(ValueError):
        core.run_experiment(cfg("x", "nonsense"), 0)


# ------------------------------------------------------------------ determinism and caches

def test_determinism_same_config_and_seed(env):
    a = core.run_experiment(pugal_cfg(), 1)
    core.clear_memory_caches()                       # second run reads every cache from disk
    b = core.run_experiment(pugal_cfg(), 1)
    assert a.keys() == b.keys()
    for k in a:
        if k != "sec_per_inst":
            assert a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])), k


def test_classifier_cache_reused(env):
    core.run_experiment(cfg("greedy", "greedy"), 0)
    path = env / "runs" / "models" / "toy_xor_seed0.pt"
    mtime = path.stat().st_mtime_ns
    core.clear_memory_caches()
    core.run_experiment(cfg("greedy", "greedy"), 0)
    assert path.stat().st_mtime_ns == mtime


def test_hparam_choice_cached_and_reused(env):
    m = core.run_experiment(pugal_cfg(), 2)
    path = env / "runs" / "hparams" / "toy_xor_pugal.json"
    rec = json.loads(path.read_text())
    assert rec["seed"] == 0 and rec["n_val"] == 25
    assert len(rec["table"]) == 4                       # 2 eps x 1 quantile x 2 m
    assert (env / "runs" / "hparams" / "toy_xor_la2_val_n8.json").exists()
    assert m["hp_eps"] == rec["choice"]["eps"] and m["hp_m"] == rec["choice"]["m"]
    # the selection rule: lowest opening rate within 0.005 of LA2, ties -> higher acc-AUC
    ok = [r for r in rec["table"] if r["acc_auc"] >= rec["la2_val_acc_auc"] - 0.005]
    if ok:
        best = min(ok, key=lambda r: (r["open_rate"], -r["acc_auc"]))
        assert best["acc_auc"] == pytest.approx(next(r for r in rec["table"] if r["hparams"] == rec["choice"])["acc_auc"])
    # a different cached choice is used as is (no re-tuning)
    rec["choice"] = {"eps": 0.02, "screen_quantile": 0.5, "m": 1} if rec["choice"]["eps"] != 0.02 else \
        {"eps": 0.005, "screen_quantile": 0.5, "m": 1}
    path.write_text(json.dumps(rec))
    core.clear_memory_caches()
    m2 = core.run_experiment(pugal_cfg(), 3)
    assert m2["hp_eps"] == rec["choice"]["eps"] and m2["hp_m"] == 1


def test_tuning_never_touches_test_split(env, monkeypatch):
    seen = []
    real = core.run_policy

    def spy(policy, data, *a, **kw):
        seen.append((data.X_test.copy(), kw.get("max_instances")))
        return real(policy, data, *a, **kw)

    monkeypatch.setattr(core, "run_policy", spy)
    core.run_experiment(pugal_cfg(test_cap=5), 4)
    d0 = load_dataset("toy_xor", 0, DATA_DIR)
    d4 = load_dataset("toy_xor", 4, DATA_DIR)
    tuning = [x for x, _ in seen[:-1]]
    assert tuning and all(np.array_equal(x, d0.X_val[:25]) for x in tuning)
    assert np.array_equal(seen[-1][0], d4.X_test) and seen[-1][1] == 5   # the final run is on the test split


# ------------------------------------------------------------------ random_matched dependency

def test_random_matched_requires_pugal_result(env):
    c = cfg("pugal_rand", "pugal", gate="random_matched", hparams=dict(GRID))
    with pytest.raises(FileNotFoundError, match="toy_xor_pugal_seed0.json"):
        core.run_experiment(c, 0)
    (env / "results").mkdir()
    (env / "results" / "toy_xor_pugal_seed0.json").write_text(json.dumps({"metrics": {"open_rate": 0.25}}))
    m = core.run_experiment(c, 0)
    assert m["gate_p"] == pytest.approx(0.25 * 2 / 1)     # budget 2: only the first step consults the gate
    assert 0.0 <= m["open_rate"] <= 0.5


# ------------------------------------------------------------------ gate quality

def test_gatequality_metrics(env):
    c = cfg("gatequality", "gatequality", budget=3, hparams=dict(GRID))
    m = core.run_experiment(c, 0)
    for k in ("auroc_syn", "auroc_margin", "auroc_learn", "auroc_rand", "frac_positive", "mean_delta2", "nmi",
              "nmi_proxy", "gamma_hat", "gamma_mean", "n_states", "classifier_val_acc", "hp_eps"):
        assert k in m, k
    assert m["n_states"] == 10 * 2                     # 10 instances x steps with >= 2 acquisitions left
    assert 0.0 <= m["frac_positive"] <= 1.0 and m["mean_delta2"] >= -1e-9
    for k in ("auroc_syn", "auroc_margin", "auroc_learn", "auroc_rand"):
        assert math.isnan(m[k]) or 0.0 <= m[k] <= 1.0
    if not math.isnan(m["gamma_hat"]):
        assert 0.0 <= m["gamma_hat"] <= m["gamma_mean"] <= 1.0


# ------------------------------------------------------------------ P1: ablations reuse the tuned pugal choice

def test_ablation_reuses_pugal_choice_but_keeps_own_key(env):
    cdir = env / "configs"
    cdir.mkdir()
    base = pugal_cfg()
    lines = [f'{k} = {json.dumps(v)}' for k, v in base.items() if k != "hparams"]
    lines += ["[hparams]"] + [f"{k} = {json.dumps(v)}" for k, v in GRID.items()]
    (cdir / "toy_xor_pugal.toml").write_text("\n".join(lines) + "\n")
    choice = core.tune(core.load_config(cdir / "toy_xor_pugal.toml"))
    abl = {k: v for k, v in base.items() if k != "hparams"} | {"method": "pugal_m4", "m": 3}
    m = core.run_experiment(abl, 0)
    assert m["hp_eps"] == choice["eps"] and m["hp_screen_quantile"] == choice["screen_quantile"]
    assert m["hp_m"] == 3                                   # the ablation's own key wins
    assert not (env / "runs" / "hparams" / "toy_xor_pugal_m4.json").exists()
    nf = {k: v for k, v in base.items() if k != "hparams"} | {"method": "pugal_nofallback", "conf_fallback": False}
    m2 = core.run_experiment(nf, 0)
    assert m2["hp_m"] == choice["m"] and m2["hp_eps"] == choice["eps"]
    # pugal_margin keeps its own tuning
    assert core.uses_pugal_hparams(abl) and core.uses_pugal_hparams(nf)
    assert not core.uses_pugal_hparams(base | {"method": "pugal_margin", "gate": "margin"})
    assert not core.uses_pugal_hparams(base | {"method": "pugal_learn", "gate": "learned"})
