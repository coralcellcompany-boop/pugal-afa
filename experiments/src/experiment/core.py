"""Experiment logic: `run_experiment(config, seed) -> dict[str, float]` (DESIGN.md `core.py`).

One call = one (config, seed): load the split, get the cached classifier and kNN pool, resolve the
hyperparameters, build the policy, run it on the test split, and return flat float metrics.

Caches (under RUNS_DIR; created under an exclusive file lock and written by atomic rename, so many
processes can run in parallel):
- models/<dataset>_seed<seed>.pt                 classifier (train once per dataset x seed)
- screens/<dataset>_seed<s>_<risk>_n<n>_b<B>.npz  offline synergy screen: average pair synergy matrix; the
                                                  flagged set for any quantile is derived from it
- gates/<dataset>_seed<s>_<risk>_n<n>_b<B>.npz    learned-gate training states (greedy on training rows,
                                                  exact Delta_2 labels); ..._q<q>.npy = N_hat per quantile
- hparams/<dataset>_<method>.json                 tuned choice (seed 0, validation split), used by all seeds
- hparams/<dataset>_la2_val_n<n>.json             always-LA2 validation acc-AUC (selection reference)
The kNN pool is not stored: it is a seeded permutation of the training rows and is rebuilt identically.

Hyperparameter selection (notes/experiments.md): once per (dataset, method), on at most TUNE_MAX_VAL
validation instances of seed 0 (never the test split): the lowest opening rate whose validation acc-AUC is
within 0.005 of always-LA2's; ties -> higher acc-AUC, then grid order. If no setting is within 0.005, the
highest acc-AUC is chosen (ties -> lower opening rate). A single-point grid is taken as is (no LA2 run).
`la3` (topm_flagged), `gatequality` and every `pugal_*` method except `pugal_learn` / `pugal_margin` (i.e. the
ablations and `pugal_rand`) use the dataset's tuned `pugal` choice, keeping only their own changed key
(computed on demand if missing, so results never depend on job order); `screen = "all_pairs"` drops
screen_quantile. `test_cap` evaluates the first test_cap instances of the seed's test split (a prefix),
so every method on a dataset and seed sees the same instances.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import itertools
import json
import math
import os
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

from experiment.data import _parse_rho, load_dataset
from experiment.evaluate import auroc, run_policy
from experiment.gates import (
    LearnedGate,
    LogisticModel,
    MarginGate,
    _redundancy,
    RandomGate,
    SynergyGate,
    _synergy_hat,
    flag_from_scores,
    screen_synergy,
)
from experiment.oracle import OracleSampler, SynMixOracle
from experiment.model import load_classifier, predict_proba, save_classifier, train_classifier
from experiment.policies import (
    AACOPolicy,
    AllCandidates,
    GreedyPolicy,
    LookaheadPolicy,
    PUGALPolicy,
    RandomPolicy,
    StaticPolicy,
    TopMCandidates,
    static_greedy_order,
)
from experiment.sampler import KNNSampler
from experiment.utility import State, Utility

EXPERIMENTS = Path(__file__).resolve().parents[2]
DATA_DIR = EXPERIMENTS / "data"
RUNS_DIR = EXPERIMENTS / "runs"
RESULTS_DIR = EXPERIMENTS / "results"
CONFIG_DIR = EXPERIMENTS / "configs"

TUNE_SEED = 0
TUNE_MAX_VAL = 300        # validation instances used for hyperparameter selection
WITHIN = 0.005            # "within 0.5 pt of always-LA2"
SCREEN_STATES = 200       # greedy-visited training states for the offline synergy screen
GATE_TRAIN_STATES = 300   # greedy-visited training states for the learned gate
GATEQ_MAX_TEST = 300      # test instances for the gate-quality study
STATIC_ROWS = 200         # validation rows for the static greedy order
GAMMA_MIN_DEN = 1e-3      # pairs whose two-step gain is below this are skipped in the gamma estimate
DEFAULT_HP = {"eps": 0.01, "screen_quantile": 0.95}

_MEMO: dict = {}


def clear_memory_caches() -> None:
    _MEMO.clear()


def _memo(key, fn):
    key = (str(RUNS_DIR),) + key
    if key not in _MEMO:
        _MEMO[key] = fn()
    return _MEMO[key]


# ------------------------------------------------------------------ disk cache with lock + atomic rename

@contextlib.contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{path}.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _disk_cached(path: Path, compute, save, load):
    """Load `path` if it exists; otherwise compute under an exclusive lock and write by atomic rename."""
    if path.exists():
        return load(path)
    with _locked(path):
        if path.exists():                      # another process finished it while we waited
            return load(path)
        obj = compute()
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        save(obj, tmp)
        os.replace(tmp, path)
        return obj


def _save_json(obj, p):
    Path(p).write_text(json.dumps(obj, indent=2))


def _load_json(p):
    return json.loads(Path(p).read_text())


def _save_npz(obj: dict, p):
    with open(p, "wb") as fh:
        np.savez(fh, **obj)


def _load_npz(p) -> dict:
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def _save_npy(arr, p):
    with open(p, "wb") as fh:
        np.save(fh, arr)


def load_config(path: Path) -> dict:
    return tomllib.loads(Path(path).read_text())


# ------------------------------------------------------------------ shared resources per (dataset, seed)

def _training_rows(data, sampler, rng) -> np.ndarray:
    """Training rows outside the kNN pool (so a row is not its own neighbour), shuffled; all rows if none."""
    in_pool = np.zeros(len(data.X_train), bool)
    in_pool[sampler.pool_index] = True
    rows = np.flatnonzero(~in_pool) if (~in_pool).any() else np.arange(len(data.X_train))
    return rows[rng.permutation(len(rows))]


class Resources:
    """Data, classifier, samplers, engines, screens and learned-gate data for one (dataset, seed)."""

    def __init__(self, dataset: str, seed: int, pool_size: int = 5000, screen_budget: int = 10,
                 model_kind: str = "learned"):
        self.dataset, self.seed, self.pool_size = dataset, int(seed), int(pool_size)
        self.screen_budget = int(screen_budget)
        self.model_kind = model_kind
        self.data = _memo(("data", dataset, seed), lambda: load_dataset(dataset, seed, DATA_DIR))
        if model_kind == "learned":
            self.model = _memo(("model", dataset, seed), lambda: _disk_cached(
                RUNS_DIR / "models" / f"{dataset}_seed{seed}.pt",
                lambda: train_classifier(self.data, seed), save_classifier, load_classifier))
        elif model_kind == "oracle":   # exploratory X1: exact SynMix posterior and conditional
            self.rho = _parse_rho(dataset)
            if self.rho is None:
                raise ValueError(f"model = 'oracle' is defined for synmix_<rho> only, not {dataset!r}")
            self.model = _memo(("oracle", dataset, seed), lambda: SynMixOracle(self.data, self.rho))
        else:
            raise ValueError(f"unknown model kind {model_kind!r}")
        self.groups = self.data.groups
        self.d = self.data.X_train.shape[1]

    def _key(self, *parts):
        return (self.dataset, self.seed, self.pool_size, self.screen_budget, self.model_kind) + parts

    def sampler(self, n: int) -> KNNSampler:
        if self.model_kind == "oracle":
            return _memo(self._key("sampler", n), lambda: OracleSampler(self.data, self.rho, n=n, seed=self.seed))
        return _memo(self._key("sampler", n), lambda: KNNSampler(
            self.data.X_train, n=n, seed=self.seed, y_pool=self.data.y_train, max_pool=self.pool_size))

    def engine(self, risk: str, n: int) -> Utility:
        return _memo(self._key("engine", risk, n), lambda: Utility(self.model, self.sampler(n), self.groups, risk))

    def val_acc(self) -> float:
        def f():
            X = torch.from_numpy(np.asarray(self.data.X_val, np.float32))
            n0 = self.model.n_evals
            pred = predict_proba(self.model, X, torch.ones_like(X, dtype=torch.bool)).argmax(-1).numpy()
            self.model.n_evals = n0                       # not a policy evaluation
            return float((pred == np.asarray(self.data.y_val)).mean())
        return _memo(self._key("val_acc"), f)

    # -------------------------------------------------- offline synergy screen
    def _stem(self, risk, n):
        tag = "" if self.model_kind == "learned" else f"_{self.model_kind}"
        return f"{self.dataset}_seed{self.seed}_{risk}_n{n}_b{self.screen_budget}{tag}"

    def screen(self, risk: str, n: int) -> np.ndarray:
        def compute():
            _, avg = screen_synergy(self.model, self.sampler(n), self.data, n_states=SCREEN_STATES, quantile=0.5,
                                    budget=self.screen_budget, seed=self.seed, engine=self.engine(risk, n),
                                    return_scores=True)
            return {"avg": avg}
        return _memo(self._key("screen", risk, n), lambda: _disk_cached(
            RUNS_DIR / "screens" / f"{self._stem(risk, n)}.npz", compute, _save_npz, _load_npz)["avg"])

    def flagged(self, risk: str, n: int, q: float) -> set:
        return flag_from_scores(self.screen(risk, n), q)

    def pair_scores(self, risk: str, n: int) -> dict:
        avg = self.screen(risk, n)
        return {(int(j), int(k)): float(avg[j, k]) for j, k in zip(*np.nonzero(np.isfinite(avg)))}

    # -------------------------------------------------- learned gate
    def gate_train(self, risk: str, n: int) -> dict:
        """States visited by greedy on training rows with >= 2 acquisitions left: x, m, budget, exact Delta_2
        and the q-independent features [r, b, a1 - a2, R, |S|]."""
        def compute():
            eng = self.engine(risk, n)
            rows = _training_rows(self.data, self.sampler(n), np.random.default_rng(self.seed + 7919))
            Xs, Ms, Bs, D2, F = [], [], [], [], []
            for i in rows:
                st = State.empty(self.d, self.screen_budget)
                x = self.data.X_train[i]
                while st.budget >= 2 and len(eng.unobserved(st)) >= 2 and len(D2) < GATE_TRAIN_STATES:
                    ctx = eng.greedy_context(st)
                    R = eng.redundancy(st, ctx.g, ctx.j2, u1=ctx.u1)
                    gap = 0.0 if ctx.a2 is None else ctx.a1 - ctx.a2
                    Xs.append(st.x); Ms.append(st.m); Bs.append(st.budget)
                    D2.append(eng.delta2(st, ctx=ctx))
                    F.append([ctx.r, st.budget, gap, R, float(st.m.sum())])
                    st = st.acquire(ctx.g, x, self.groups)
                if len(D2) >= GATE_TRAIN_STATES:
                    break
            return {"x": np.array(Xs, np.float32).reshape(-1, self.d), "m": np.array(Ms, bool).reshape(-1, self.d),
                    "budget": np.array(Bs, np.int64), "delta2": np.array(D2), "base": np.array(F).reshape(-1, 5)}
        return _memo(self._key("gate_train", risk, n), lambda: _disk_cached(
            RUNS_DIR / "gates" / f"{self._stem(risk, n)}.npz", compute, _save_npz, _load_npz))

    def gate_nhat(self, risk: str, n: int, q: float) -> np.ndarray:
        def compute():
            tr = self.gate_train(risk, n)
            eng, flagged = self.engine(risk, n), self.flagged(risk, n, q)
            out = []
            for x, m, b in zip(tr["x"], tr["m"], tr["budget"]):
                st = State(x, m, int(b))
                ctx = eng.greedy_context(st)
                out.append(eng.synergy_term(st, ctx.g, flagged, u1=ctx.u1))
            return np.array(out)
        return _memo(self._key("gate_nhat", risk, n, q), lambda: _disk_cached(
            RUNS_DIR / "gates" / f"{self._stem(risk, n)}_q{q}.npy", compute, _save_npy, np.load))

    def learned_model(self, risk: str, n: int, q: float, eps: float):
        def fit():
            tr = self.gate_train(risk, n)
            b = tr["base"]
            F = np.column_stack([b[:, 0], b[:, 1], b[:, 2], b[:, 3], self.gate_nhat(risk, n, q), b[:, 4]])
            y = (tr["delta2"] > eps).astype(float)
            if len(y) == 0 or y.min() == y.max():
                return _ConstantProb(float(y.mean()) if len(y) else 0.0)
            return LogisticModel().fit(F, y)
        return _memo(self._key("learned", risk, n, q, eps), fit)


class _ConstantProb:
    """Learned-gate model when the training labels have one class."""

    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, F):
        return np.full(len(np.atleast_2d(F)), self.p)


def _resources(cfg: dict, seed: int) -> Resources:
    return Resources(cfg["dataset"], seed, cfg.get("pool_size", 5000), cfg["budget"], cfg.get("model", "learned"))


# ------------------------------------------------------------------ hyperparameters

def _grid(cfg: dict) -> list[dict]:
    grid = dict(cfg.get("hparams") or {})
    if cfg.get("screen") == "all_pairs":
        grid.pop("screen_quantile", None)
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _pugal_config(cfg: dict) -> dict | None:
    """The dataset's main PUGAL config: configs/<dataset>_pugal.toml, else derived from a config that
    carries the PUGAL grid, else None."""
    path = CONFIG_DIR / f"{cfg['dataset']}_pugal.toml"
    if path.exists():
        return load_config(path)
    if cfg.get("hparams"):
        base = {k: v for k, v in cfg.items() if k not in ("gate_terms", "screen", "candidates", "h")}
        return base | {"method": "pugal", "policy": "pugal", "gate": "synergy"}
    return None


def pugal_hparams(cfg: dict) -> tuple[dict, bool]:
    """The dataset's tuned `pugal` choice. A key the config sets differently from the pugal config (e.g.
    m = 4 in pugal_m4) is the config's own key and keeps the config's value."""
    pc = _pugal_config(cfg)
    if pc is None:
        return dict(DEFAULT_HP) | {"m": cfg.get("m", 8)}, False
    hp = tune(pc)
    own = {k for k in hp if k in cfg and k in pc and cfg[k] != pc[k]}
    return {k: (cfg[k] if k in own else v) for k, v in hp.items()}, True


OWN_TUNING = ("pugal", "pugal_learn", "pugal_margin")   # every other pugal_* method reuses pugal's choice


def uses_pugal_hparams(cfg: dict) -> bool:
    """True if the config takes the dataset's tuned `pugal` choice instead of tuning its own grid:
    gate quality, LA3 (topm_flagged) and every pugal_* method except pugal_learn / pugal_margin
    (ablations and pugal_rand change one component and keep everything else paired with main PUGAL)."""
    if cfg["policy"] == "gatequality":
        return True
    if cfg["policy"] == "lookahead" and cfg.get("candidates") == "topm_flagged":
        return True
    return cfg["policy"] == "pugal" and str(cfg.get("method", "")).startswith("pugal_") and \
        cfg["method"] not in OWN_TUNING


def hparams_for(cfg: dict) -> tuple[dict, bool]:
    """(hyperparameters, tuned?) for a config."""
    if uses_pugal_hparams(cfg):
        return pugal_hparams(cfg)
    if not cfg.get("hparams"):
        return {}, False
    return tune(cfg), True


def _budget(cfg: dict) -> int:
    return int(cfg.get("budget_override", cfg["budget"]))


def _val_view(data):
    k = min(TUNE_MAX_VAL, len(data.X_val))
    return dataclasses.replace(data, X_test=data.X_val[:k], y_test=data.y_val[:k])


def la2_val_reference(cfg: dict) -> float:
    n = int(cfg.get("n_neighbors", 16))
    path = RUNS_DIR / "hparams" / f"{cfg['dataset']}_la2_val_n{n}.json"

    def compute():
        res = _resources(cfg, TUNE_SEED)
        eng = res.engine("brier", n)
        val = _val_view(res.data)
        out = run_policy(LookaheadPolicy(eng, h=2), val, res.model, eng.sampler, _budget(cfg))
        return {"acc_auc": out["acc_auc"], "acc": out["acc"], "n_val": out["n_instances"], "seed": TUNE_SEED,
                "sec_per_inst": out["sec_per_inst"]}
    return float(_memo(("la2ref", cfg["dataset"], n, _budget(cfg)), lambda: _disk_cached(
        path, compute, _save_json, _load_json))["acc_auc"])


def select(table: list[dict], la2_acc_auc: float) -> dict:
    """Lowest opening rate within WITHIN of LA2 (ties -> higher acc-AUC, then grid order); if none is
    within, the highest acc-AUC (ties -> lower opening rate, then grid order)."""
    idx = list(range(len(table)))
    ok = [i for i in idx if table[i]["acc_auc"] >= la2_acc_auc - WITHIN - 1e-12]
    if ok:
        best = min(ok, key=lambda i: (table[i]["open_rate"], -table[i]["acc_auc"], i))
    else:
        best = min(idx, key=lambda i: (-table[i]["acc_auc"], table[i]["open_rate"], i))
    return table[best]["hparams"]


def tune(cfg: dict) -> dict:
    """Tuned hyperparameters for (dataset, method); computed once on seed 0's validation split."""
    path = RUNS_DIR / "hparams" / f"{cfg['dataset']}_{cfg['method']}.json"

    def compute():
        combos = _grid(cfg)
        if len(combos) == 1:
            return {"choice": combos[0], "table": [], "la2_val_acc_auc": None, "n_val": 0, "seed": TUNE_SEED,
                    "rule": "single grid point"}
        la2 = la2_val_reference(cfg)
        res = _resources(cfg, TUNE_SEED)
        val = _val_view(res.data)
        table = []
        for hp in combos:
            pol = build_policy(cfg | hp, res, TUNE_SEED)
            out = run_policy(pol, val, res.model, None, _budget(cfg))
            table.append({"hparams": hp, "acc_auc": out["acc_auc"], "open_rate": out["open_rate"],
                          "evals_per_inst": out["evals_per_inst"]})
        return {"choice": select(table, la2), "table": table, "la2_val_acc_auc": la2, "n_val": len(val.X_test),
                "seed": TUNE_SEED, "rule": f"lowest open_rate with acc_auc >= LA2 - {WITHIN}"}
    rec = _memo(("tune", cfg["dataset"], cfg["method"]), lambda: _disk_cached(path, compute, _save_json, _load_json))
    return dict(rec["choice"])


# ------------------------------------------------------------------ policies

def _matched_rate(cfg: dict, seed: int) -> float:
    """Gate probability for G-rand: the same dataset/seed PUGAL run's test opening rate, rescaled from 'per
    step' to 'per gate decision' (PUGAL never consults the gate on the last step: B - 1 of B steps)."""
    path = RESULTS_DIR / f"{cfg['dataset']}_pugal_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"random_matched gate needs {path} (the opening rate of {cfg['dataset']}_pugal with seed {seed}); "
            f"run that config first")
    rate = float(_load_json(path)["metrics"]["open_rate"])
    B = int(cfg["budget"])
    return min(1.0, rate * B / (B - 1)) if B > 1 else 0.0


def build_policy(cfg: dict, res: Resources, seed: int):
    """Policy for an effective config (fixed keys + chosen hyperparameters)."""
    kind = cfg["policy"]
    risk = cfg.get("risk", "brier")
    n = int(cfg.get("n_neighbors", 16))
    if kind == "random":
        return RandomPolicy(res.groups, seed=seed)
    if kind == "static":
        eng = res.engine(risk, n)
        order = static_greedy_order(eng, res.data.X_val[:STATIC_ROWS], _budget(cfg))
        return StaticPolicy(order, res.groups)
    if kind == "greedy":
        return GreedyPolicy(res.engine(risk, n))
    if kind == "aaco":
        return AACOPolicy(res.engine(risk, n), k_nn=int(cfg.get("k_nn", 5)),
                          n_subsets=int(cfg.get("n_subsets", 1000)), seed=seed)
    if kind == "lookahead":
        eng = res.engine(risk, n)
        if cfg.get("candidates", "all") == "all":
            rule = AllCandidates()
        else:  # topm_flagged: the screen and m of the dataset's tuned PUGAL
            pc = _pugal_config(cfg) or {}
            n_scr = int(pc.get("n_neighbors", 16))
            q = float(cfg.get("screen_quantile", DEFAULT_HP["screen_quantile"]))
            rule = TopMCandidates(int(cfg.get("m", 8)), res.flagged(risk, n_scr, q), cap=cfg.get("cand_cap"),
                                  pair_scores=res.pair_scores(risk, n_scr))
        return LookaheadPolicy(eng, h=int(cfg.get("h", 2)), candidate_rule=rule, tie=cfg.get("tie", "index"))
    if kind == "pugal":
        eng = res.engine(risk, n)
        eps = float(cfg.get("eps", DEFAULT_HP["eps"]))
        m = int(cfg.get("m", 8))
        all_pairs = cfg.get("screen") == "all_pairs"
        if all_pairs:
            flagged, scores = None, "online"
        else:
            q = float(cfg.get("screen_quantile", DEFAULT_HP["screen_quantile"]))
            flagged, scores = res.flagged(risk, n, q), res.pair_scores(risk, n)
        gate_kind = cfg.get("gate", "synergy")
        terms = cfg.get("gate_terms", "both")
        if gate_kind == "synergy":
            gate = SynergyGate(flagged, eps, use_redundancy=terms in ("both", "redundancy"),
                               use_synergy=terms in ("both", "synergy"))
        elif gate_kind == "margin":
            gate = MarginGate(tau=eps)
        elif gate_kind == "learned":
            gate = LearnedGate(res.learned_model(risk, n, q, eps), flagged, threshold=0.5)
        elif gate_kind == "random_matched":
            gate = RandomGate(_matched_rate(cfg, seed), seed=seed)
        else:
            raise ValueError(f"unknown gate {gate_kind!r}")
        rule = TopMCandidates(m, flagged, cap=cfg.get("cand_cap"), pair_scores=scores,
                              min_score=eps if all_pairs else 0.0)
        return PUGALPolicy(eng, gate, h0=int(cfg.get("h0", 2)), h_max=int(cfg.get("h_max", 2)), m=m,
                           conf_fallback=bool(cfg.get("conf_fallback", True)), candidate_rule=rule)
    raise ValueError(f"unknown policy {kind!r}")


# ------------------------------------------------------------------ gate quality (E1)

def gate_quality(cfg: dict, res: Resources, seed: int, hp: dict) -> dict:
    risk, n, budget = cfg.get("risk", "brier"), int(cfg.get("n_neighbors", 16)), _budget(cfg)
    eps, q = float(hp["eps"]), float(hp["screen_quantile"])
    eng = res.engine(risk, n)
    flagged = res.flagged(risk, n, q)
    gates = {"syn": SynergyGate(flagged, eps), "margin": MarginGate(tau=eps),
             "learn": LearnedGate(res.learned_model(risk, n, q, eps), flagged), "rand": RandomGate(0.5, seed=seed)}
    cap = min(GATEQ_MAX_TEST, cfg.get("test_cap", GATEQ_MAX_TEST))
    X = np.asarray(res.data.X_test[:cap], np.float32)
    scores = {k: [] for k in gates}
    d2s, gam, sum_d2, sum_n, red = [], [], [], [], []
    x2 = bool(cfg.get("x2_detail", False))      # exploratory X2: per-state detail
    det = {k: [] for k in ("step", "a1", "a2", "R", "nhat")}
    t0 = time.perf_counter()
    for x in X:
        st = State.empty(res.d, budget)
        r0 = eng.risk_at(st)
        s_d2 = s_n = 0.0
        for _ in range(budget):
            ctx = eng.greedy_context(st)
            if st.budget >= 2 and len(ctx.order) >= 2:
                tab = eng.two_step_table(st, ctx)
                for k, gate in gates.items():          # syn first: R and N_hat are cached in ctx for learn
                    scores[k].append(float(gate.score(st, ctx)))
                nhat = _synergy_hat(st, ctx, flagged)
                if x2:
                    det["step"].append(budget - st.budget)
                    det["a1"].append(float(ctx.a1))
                    det["a2"].append(float(ctx.a2) if ctx.a2 is not None else float("nan"))
                    det["R"].append(float(_redundancy(st, ctx)))
                    det["nhat"].append(float(nhat))
                d2s.append(tab["delta2"])
                s_d2 += tab["delta2"]
                s_n += nhat
                for j in ctx.order:
                    for k in ctx.order:
                        if j == k:
                            continue
                        den = ctx.u1[j] + tab["gain"][j, k]
                        if den > GAMMA_MIN_DEN:
                            gam.append(min(1.0, max(0.0, (ctx.u1[j] + ctx.u1[k]) / den)))
            st = st.acquire(ctx.g, x, res.groups)
        sum_d2.append(s_d2)
        sum_n.append(s_n)
        red.append(r0 - eng.risk_at(st))
    labels = np.array(d2s) > eps
    mean_red = float(np.mean(red)) if red else float("nan")
    out = {f"auroc_{k}": auroc(v, labels) for k, v in scores.items()}
    out |= {"frac_positive": float(labels.mean()) if len(labels) else float("nan"),
            "mean_delta2": float(np.mean(d2s)) if d2s else float("nan"),
            "nmi": float(np.mean(sum_d2) / mean_red) if mean_red > 0 else float("nan"),
            "nmi_proxy": float(np.mean(sum_n) / mean_red) if mean_red > 0 else float("nan"),
            "gamma_hat": float(np.min(gam)) if gam else float("nan"),
            "gamma_mean": float(np.mean(gam)) if gam else float("nan"),
            "n_states": float(len(d2s)), "n_instances": float(len(X)),
            "sec_per_inst": (time.perf_counter() - t0) / max(1, len(X))}
    if x2:
        out |= gate_detail_metrics({k: np.array(v) for k, v in det.items()},
                                   {k: np.array(v) for k, v in scores.items()}, labels,
                                   res.learned_model(risk, n, q, eps))
        dump = RUNS_DIR / "gatestates" / f"{cfg['dataset']}_seed{seed}.npz"
        dump.parent.mkdir(parents=True, exist_ok=True)
        _save_npz({**{k: np.array(v) for k, v in det.items()}, **{f"score_{k}": np.array(v) for k, v in
                   scores.items()}, "delta2": np.array(d2s), "eps": np.array(eps)}, dump)
    return out


LEARN_FEATURES = ("r", "b", "gap", "R", "nhat", "S")   # LearnedGate feature order (Resources.learned_model)


def gate_detail_metrics(det: dict, scores: dict, labels: np.ndarray, learned) -> dict:
    """Exploratory X2 statistics (notes/experiments.md): AUROC of R alone, N_hat alone and the relative margin
    (a1 - a2) / a1 (score = minus it; 0 when a1 = 0, i.e. no feature helps alone); AUROC of each gate by step;
    mean a1 on positive / negative states; the learned gate's standardised logistic weights."""
    a1, a2 = det["a1"], np.nan_to_num(det["a2"], nan=0.0)
    rel = np.where(a1 > 1e-12, (a1 - a2) / np.where(a1 > 1e-12, a1, 1.0), 0.0)
    out = {"auroc_R": auroc(det["R"], labels), "auroc_nhat": auroc(det["nhat"], labels),
           "auroc_relmargin": auroc(-rel, labels)}
    pos, neg = labels, ~labels
    out["mean_a1_pos"] = float(a1[pos].mean()) if pos.any() else float("nan")
    out["mean_a1_neg"] = float(a1[neg].mean()) if neg.any() else float("nan")
    for t in np.unique(det["step"]):
        sel = det["step"] == t
        out[f"frac_positive_t{int(t)}"] = float(labels[sel].mean())
        for k in ("syn", "learn", "margin"):
            out[f"auroc_{k}_t{int(t)}"] = auroc(scores[k][sel], labels[sel])
        out[f"auroc_relmargin_t{int(t)}"] = auroc(-rel[sel], labels[sel])
    w = getattr(learned, "w", None)
    if w is not None:
        out |= {f"learn_w_{name}": float(v) for name, v in zip(LEARN_FEATURES, w[:len(LEARN_FEATURES)])}
    return out


# ------------------------------------------------------------------ entry point

def run_experiment(config: dict, seed: int) -> dict[str, float]:
    cfg = dict(config)
    seed = int(seed)
    res = _resources(cfg, seed)
    hp, tuned = hparams_for(cfg)
    metrics: dict[str, float] = {"classifier_val_acc": res.val_acc()}
    for k in ("eps", "screen_quantile", "m"):
        if k in hp:
            metrics[f"hp_{k}"] = float(hp[k])
    if hp:
        metrics["hp_tuned"] = float(tuned)
    eff = cfg | hp
    if cfg["policy"] == "gatequality":
        metrics |= gate_quality(eff, res, seed, dict(DEFAULT_HP) | hp)
        return metrics
    policy = build_policy(eff, res, seed)
    if isinstance(getattr(policy, "gate", None), RandomGate):
        metrics["gate_p"] = float(policy.gate.p)
    if cfg["policy"] == "pugal" and cfg.get("screen") != "all_pairs":
        metrics["n_flagged"] = float(len(res.flagged(eff.get("risk", "brier"), int(eff.get("n_neighbors", 16)),
                                                      float(eff.get("screen_quantile", 0.95)))))
    out = run_policy(policy, res.data, res.model, None, _budget(cfg), max_instances=cfg.get("test_cap"))
    for k, v in out.items():
        if k.startswith(("acc_t", "f1_t")) or k in ("acc_auc", "f1_auc", "evals_per_inst", "sec_per_inst",
                                                     "open_rate", "dev_rate"):
            metrics[k] = float(v)
    metrics["n_instances"] = float(out["n_instances"])
    return metrics


# ------------------------------------------------------------------ CLI for the driver's warm-up / tuning jobs

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Warm caches for scripts/run_all.py")
    ap.add_argument("task", choices=["warm", "la2ref", "tune"])
    ap.add_argument("config", type=Path)
    ap.add_argument("--seed", type=int, default=TUNE_SEED)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    t0 = time.perf_counter()
    if args.task == "warm":          # classifier + the brier screen used by PUGAL / LA3 / gate quality
        res = _resources(cfg, args.seed)
        res.val_acc()
        if cfg.get("policy") in ("pugal", "gatequality") or cfg.get("candidates") == "topm_flagged":
            res.screen(cfg.get("risk", "brier"), int(cfg.get("n_neighbors", 16)))
    elif args.task == "la2ref":
        la2_val_reference(cfg)
    else:
        print(json.dumps(hparams_for(cfg)[0]))
    print(f"{args.task} {args.config.name} seed={args.seed} done in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
