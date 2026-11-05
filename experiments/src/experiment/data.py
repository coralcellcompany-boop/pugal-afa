"""Datasets for the PUGAL experiments: loading, splitting, normalisation.

Provenance and preprocessing (compared with AFABench, schutz2025afabench; its
code was read as text only, never executed)
-----------------------------------------------------------------------------
Matched:
- Diabetes, MiniBooNE, PhysioNet: the CSVs shipped in AFABench
  (extra/data/misc/*.csv). Last column ("Outcome") is the label; all other
  columns are features (45 / 50 / 41). Mean imputation + z-normalisation.
- BankMarketing (UCI 222, bank-full, 45 211 rows): the 16 UCI "Feature"
  columns, label y = 1 iff y == "yes". We drop `duration` (DEVIATION from
  AFABench, which keeps it): it is only known after the call and UCI notes it
  should be discarded for a realistic predictive model (label leakage). Missing values
  in job / education / contact / poutcome are their own category, as in
  AFABench (it label-encodes `astype(str)`, so NaN becomes the level "nan").
- ACTG175 (UCI 890, 2 139 rows): the 23 UCI "Feature" columns (ID `pidnum`
  dropped), label = `cid` (0/1). All numeric. We drop `time` (follow-up
  time; DEVIATION from AFABench, which keeps it): it is determined by the
  outcome window and leaks `cid`.
- CKD (UCI 336, 400 rows): the 24 UCI "Feature" columns, label 1 = "ckd",
  0 = "notckd" (whitespace stripped, lower-cased, as in AFABench).
  Binary string columns coded by sorted level (LabelEncoder order:
  abnormal=0/normal=1, notpresent=0/present=1, no=0/yes=1, good=0/poor=1);
  missing values stay missing and are mean-imputed, as in AFABench.
- CUBE: port of AFABench CubeDataset with the same torch RNG call order, using
  the parameters of its dataset-generation config (dataset/cube.yaml):
  informative std 0.3, non-informative mean 0.5 / std 0.3; 20 features,
  8 classes. Generator seed = run seed.
- CUBE-NM: port of CubeNMDataset (same RNG call order), n_contexts = 5,
  block size 10 -> 55 features, 8 classes. Parameters from AFABench's
  dataset/cube_nm.yaml: context std 0.1, informative std 0.3,
  non-informative mean 0.5 / std 0.3.

Different from AFABench (on purpose, per DESIGN.md / notes/experiments.md):
- Size: we generate 20 000 rows for CUBE / CUBE-NM (AFABench config: 1 000).
- CUBE-NM informative std is 0.3 (AFABench generation config); the class
  default in AFABench's code is 0.1.
- Splits: AFABench draws a plain random 60/20/20 split; we stratify by label
  for real data (synthetic data: plain random), then cap train <= 20 000,
  val <= 1 000, test <= 1 000 by random subsampling.
- Statistics: AFABench imputes and z-normalises on the FULL dataset before
  splitting; we use training-split means / stds only (no test leakage).
- Categoricals: AFABench label-encodes every string column (ordinal codes,
  BankMarketing stays at 16 features). We one-hot encode string columns with
  more than two levels and code two-level columns as 0/1. BankMarketing
  becomes 48 features; each original variable is one acquisition group, so
  the action space keeps AFABench's 16 actions.
- CKD: AFABench treats the stray value "\tno" in `dm` as a third level; we
  strip whitespace first (so `dm` is binary).
- z-normalisation also applies to the synthetic data (AFABench leaves CUBE
  raw).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# --------------------------------------------------------------------- constants

TRAIN_FRAC, VAL_FRAC = 0.6, 0.2
CAP_TRAIN, CAP_VAL, CAP_TEST = 20_000, 1_000, 1_000
N_SYNTH = 20_000  # rows generated for cube, cube_nm, synmix_<rho>
N_SMALL = 2_000  # rows for toy_xor and smoke
SYNMIX_D = 20
SYNMIX_LABEL_NOISE = 0.05
IMBALANCE_RATIO = 0.7  # imbalanced iff minority share < 0.7 / n_classes (= 35% for binary)

# CUBE / CUBE-NM parameters (AFABench extra/conf/scripts/dataset_generation/dataset/*.yaml)
CUBE_INFORMATIVE_STD = 0.3
CUBE_NONINFO_MEAN = 0.5
CUBE_NONINFO_STD = 0.3
CUBE_NM_CONTEXTS = 5
CUBE_NM_BLOCK = 10
CUBE_NM_CONTEXT_STD = 0.1

# raw files written by scripts/fetch_data.py
RAW_FILES = {
    "diabetes": "diabetes.csv",
    "miniboone": "miniboone.csv",
    "physionet": "physionet.csv",
    "bank": "bank.csv",
    "actg175": "actg175.csv",
    "ckd": "ckd.csv",
}


@dataclass
class AFAData:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray  # float32 [n, d], int64 [n]
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    groups: list[list[int]]  # acquisition actions; each is a list of feature indices
    imbalanced: bool  # True -> report macro-F1 as well
    norm_mu: np.ndarray | None = None  # train mean / std used for z-normalisation (raw = x * sd + mu)
    norm_sd: np.ndarray | None = None


# --------------------------------------------------------------------- generators


def _cube_binary_codes(n_classes: int = 8) -> torch.Tensor:
    # identical to AFABench: 3-bit code of each class, least significant bit first
    return torch.stack(
        [torch.tensor([int(b) for b in format(i, "03b")]) for i in range(n_classes)], dim=0
    ).flip(-1)


def make_cube(
    n: int,
    seed: int,
    informative_std: float = CUBE_INFORMATIVE_STD,
    noninfo_mean: float = CUBE_NONINFO_MEAN,
    noninfo_std: float = CUBE_NONINFO_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """CUBE (20 features, 8 classes); same RNG call order as AFABench CubeDataset."""
    n_classes, n_cube, n_dummy = 8, 10, 10
    rng = torch.Generator()
    rng.manual_seed(seed)
    y = torch.randint(0, n_classes, (n,), dtype=torch.int64, generator=rng)
    codes = _cube_binary_codes(n_classes)
    x_cube = torch.normal(mean=noninfo_mean, std=noninfo_std, size=(n, n_cube), generator=rng)
    x_dummy = torch.normal(mean=noninfo_mean, std=noninfo_std, size=(n, n_dummy), generator=rng)
    for i in range(n):
        lbl = int(y[i].item())
        idxs = [lbl + j for j in range(3)]
        x_cube[i, idxs] = (
            torch.normal(mean=0.0, std=informative_std, size=(3,), generator=rng) + codes[lbl]
        )
    X = torch.cat([x_cube, x_dummy], dim=1)
    return X.numpy().astype(np.float64), y.numpy().astype(np.int64)


def make_cube_nm(
    n: int,
    seed: int,
    n_contexts: int = CUBE_NM_CONTEXTS,
    context_std: float = CUBE_NM_CONTEXT_STD,
    informative_std: float = CUBE_INFORMATIVE_STD,
    noninfo_mean: float = CUBE_NONINFO_MEAN,
    noninfo_std: float = CUBE_NONINFO_STD,
    return_context: bool = False,
):
    """CUBE-NM: noisy one-hot context (first n_contexts features) + n_contexts blocks of 10.

    Same RNG call order as AFABench CubeNMDataset. Returns (X, y) or (X, y, context).
    """
    n_classes, block = 8, CUBE_NM_BLOCK
    rng = torch.Generator().manual_seed(seed)
    context = torch.randint(0, n_contexts, (n,), generator=rng)
    context_onehot = torch.nn.functional.one_hot(context, num_classes=n_contexts).float() + torch.normal(
        mean=0, std=context_std, size=(n, n_contexts), generator=rng
    )
    y = torch.randint(0, n_classes, (n,), generator=rng)
    codes = _cube_binary_codes(n_classes)
    blocks = torch.stack(
        [
            torch.normal(mean=noninfo_mean, std=noninfo_std, size=(n, block), generator=rng)
            for _ in range(n_contexts)
        ],
        dim=1,
    )  # (n, n_contexts, block)
    for i in range(n):
        ctx = int(context[i].item())
        lbl = int(y[i].item())
        idx = [(lbl + j) % block for j in range(3)]
        noise = torch.normal(mean=0.0, std=informative_std, size=(3,), generator=rng)
        blocks[i, ctx, idx] = codes[lbl].float() + noise
    X = torch.cat([context_onehot, blocks.view(n, -1)], dim=1)
    out = (X.numpy().astype(np.float64), y.numpy().astype(np.int64))
    if return_context:
        return out + (context.numpy().astype(np.int64),)
    return out


def make_synmix(n: int, rho: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """SynMix(rho): feature 0 = z ~ Bernoulli(rho), 1 = x1, 2 = x2, 3 = x3, 4..19 distractors.

    z = 1 -> y = 1{x1>0} XOR 1{x2>0};  z = 0 -> y = 1{x3>0};  then 5% of labels flipped.
    """
    z = (rng.random(n) < rho).astype(np.float64)
    cont = rng.standard_normal((n, SYNMIX_D - 1))
    x1, x2, x3 = cont[:, 0], cont[:, 1], cont[:, 2]
    y_xor = (x1 > 0) ^ (x2 > 0)
    y_single = x3 > 0
    y = np.where(z == 1, y_xor, y_single).astype(np.int64)
    flip = rng.random(n) < SYNMIX_LABEL_NOISE
    y = np.where(flip, 1 - y, y)
    X = np.column_stack([z, cont])
    return X, y


def make_toy_xor(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """x1, x2, x3 ~ N(0,1) (x3 is noise); y = 1{x1>0} XOR 1{x2>0}; no label noise."""
    X = rng.standard_normal((n, 3))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
    return X, y


# --------------------------------------------------------------------- raw real data


def _raw_path(name: str, data_dir: Path) -> Path:
    path = Path(data_dir) / RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `uv run python scripts/fetch_data.py` first")
    return path


def _labels_to_int(s: pd.Series) -> np.ndarray:
    v = pd.to_numeric(s, errors="raise").to_numpy(dtype=np.float64)
    if np.isnan(v).any() or not np.all(v == np.rint(v)):
        raise ValueError("labels must be integer-valued with no missing values")
    return v.astype(np.int64)


def _encode_features(df: pd.DataFrame, nan_as_category: bool) -> tuple[np.ndarray, list[list[int]]]:
    """Numeric columns pass through; string columns: 2 levels -> 0/1, more -> one-hot.

    Returns a float64 matrix (may contain NaN) and one group per original column.
    """
    cols: list[np.ndarray] = []
    groups: list[list[int]] = []
    j = 0
    for name in df.columns:
        s = df[name]
        if pd.api.types.is_numeric_dtype(s):
            cols.append(s.to_numpy(dtype=np.float64))
            groups.append([j])
            j += 1
            continue
        s = s.astype("object").map(lambda v: v.strip() if isinstance(v, str) else v)
        missing = s.isna().to_numpy()
        if nan_as_category:
            s = s.where(~missing, "nan")
            missing = np.zeros(len(s), dtype=bool)
        levels = sorted(s[~missing].unique().tolist())
        values = s.to_numpy()
        if len(levels) <= 2:
            code = np.full(len(s), np.nan)
            for k, lev in enumerate(levels):
                code[values == lev] = k
            cols.append(code)
            groups.append([j])
            j += 1
        else:
            group = []
            for lev in levels:
                col = (values == lev).astype(np.float64)
                col[missing] = np.nan
                cols.append(col)
                group.append(j)
                j += 1
            groups.append(group)
    return np.column_stack(cols), groups


def _load_afabench_csv(name: str, data_dir: Path):
    df = pd.read_csv(_raw_path(name, data_dir))
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    y = _labels_to_int(df.iloc[:, -1])
    return X, y, [[i] for i in range(X.shape[1])]


def _load_bank(data_dir: Path):
    df = pd.read_csv(_raw_path("bank", data_dir))
    target = df.pop("y").astype("object").map(lambda v: v.strip().lower())
    if not target.isin(["yes", "no"]).all():
        raise ValueError("bank: unexpected label values")
    y = (target == "yes").to_numpy().astype(np.int64)
    df = df.drop(columns=["duration"])  # label leakage, see header
    X, groups = _encode_features(df, nan_as_category=True)
    return X, y, groups


def _load_actg175(data_dir: Path):
    df = pd.read_csv(_raw_path("actg175", data_dir))
    y = _labels_to_int(df.pop("cid"))
    df = df.drop(columns=["pidnum", "time"])  # ID; `time` leaks the label, see header
    X, groups = _encode_features(df, nan_as_category=False)
    return X, y, groups


def _load_ckd(data_dir: Path):
    df = pd.read_csv(_raw_path("ckd", data_dir))
    target = df.pop("class").astype("object").map(lambda v: v.strip().lower() if isinstance(v, str) else v)
    mapped = target.map({"ckd": 1, "notckd": 0})
    if mapped.isna().any():
        raise ValueError("ckd: unexpected label values")
    y = mapped.to_numpy().astype(np.int64)
    X, groups = _encode_features(df, nan_as_category=False)
    return X, y, groups


_REAL_LOADERS = {
    "diabetes": lambda d: _load_afabench_csv("diabetes", d),
    "miniboone": lambda d: _load_afabench_csv("miniboone", d),
    "physionet": lambda d: _load_afabench_csv("physionet", d),
    "bank": _load_bank,
    "actg175": _load_actg175,
    "ckd": _load_ckd,
}

# --------------------------------------------------------------------- split / normalise


def _split_indices(y: np.ndarray, rng: np.random.Generator, stratify: bool):
    n = len(y)
    if not stratify:
        perm = rng.permutation(n)
        n_tr, n_va = int(TRAIN_FRAC * n), int(VAL_FRAC * n)
        return perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :]
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_tr, n_va = int(round(TRAIN_FRAC * len(idx))), int(round(VAL_FRAC * len(idx)))
        tr.append(idx[:n_tr])
        va.append(idx[n_tr : n_tr + n_va])
        te.append(idx[n_tr + n_va :])
    return tuple(rng.permutation(np.concatenate(parts)) for parts in (tr, va, te))


def _cap(idx: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if len(idx) <= cap:
        return idx
    return idx[rng.choice(len(idx), size=cap, replace=False)]


def _impute_normalise(X_tr: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Mean-impute with train means, then z-normalise with train mean / std (ddof 0)."""
    observed = ~np.isnan(X_tr)
    count = observed.sum(0)
    fill = np.where(count > 0, np.where(observed, X_tr, 0.0).sum(0) / np.maximum(count, 1), 0.0)
    out = []
    for X in (X_tr, *others):
        X = np.where(np.isnan(X), fill, X)
        out.append(X)
    mu, sd = _norm_stats(out[0])
    return [np.ascontiguousarray(((X - mu) / sd).astype(np.float32)) for X in out]


def _norm_stats(X_tr_filled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train mean and std (ddof 0; std 1 for constant columns) used by `_impute_normalise`."""
    mu = X_tr_filled.mean(0)
    sd = X_tr_filled.std(0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return mu, sd


def _is_imbalanced(y: np.ndarray, n_classes: int) -> bool:
    share = np.bincount(y, minlength=n_classes) / len(y)
    return bool(share.min() < IMBALANCE_RATIO / n_classes)


# --------------------------------------------------------------------- public API

_SYNMIX_RE = re.compile(r"^synmix_(.+)$")


def _parse_rho(name: str) -> float | None:
    m = _SYNMIX_RE.match(name)
    if m is None:
        return None
    try:
        rho = float(m.group(1))
    except ValueError:
        raise ValueError(f"bad synmix rho in {name!r}") from None
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError(f"synmix rho must be in [0, 1], got {name!r}")
    return rho


def load_dataset(name: str, seed: int, data_dir: Path) -> AFAData:
    """Build the train / val / test splits of dataset `name` for run `seed`.

    Names: cube, cube_nm, synmix_<rho>, diabetes, miniboone, physionet, bank,
    actg175, ckd, toy_xor, smoke (= synmix_0.5 with 2 000 rows).
    """
    gen_rng = np.random.default_rng([seed, 0])  # data generation (numpy generators)
    split_rng = np.random.default_rng([seed, 1])  # splits and caps
    rho = _parse_rho(name)
    stratify = False
    if name == "cube":
        X, y = make_cube(N_SYNTH, seed)
        n_classes, groups = 8, None
    elif name == "cube_nm":
        X, y = make_cube_nm(N_SYNTH, seed)
        n_classes = 8
        groups = [list(range(CUBE_NM_CONTEXTS))] + [[i] for i in range(CUBE_NM_CONTEXTS, X.shape[1])]
    elif rho is not None:
        X, y = make_synmix(N_SYNTH, rho, gen_rng)
        n_classes, groups = 2, None
    elif name == "smoke":
        X, y = make_synmix(N_SMALL, 0.5, gen_rng)
        n_classes, groups = 2, None
    elif name == "toy_xor":
        X, y = make_toy_xor(N_SMALL, gen_rng)
        n_classes, groups = 2, None
    elif name in _REAL_LOADERS:
        X, y, groups = _REAL_LOADERS[name](Path(data_dir))
        n_classes = int(y.max()) + 1
        if y.min() < 0 or len(np.unique(y)) != n_classes:
            raise ValueError(f"{name}: labels must be 0..C-1 with every class present")
        stratify = True
    else:
        raise ValueError(f"unknown dataset {name!r}")

    if groups is None:
        groups = [[i] for i in range(X.shape[1])]
    tr, va, te = _split_indices(y, split_rng, stratify)
    tr = _cap(tr, CAP_TRAIN, split_rng)
    va = _cap(va, CAP_VAL, split_rng)
    te = _cap(te, CAP_TEST, split_rng)
    X_tr, X_va, X_te = _impute_normalise(X[tr], X[va], X[te])
    # stats are recorded for the synthetic generators only (no missing values, so no imputation step)
    mu, sd = _norm_stats(X[tr]) if not np.isnan(X[tr]).any() else (None, None)
    return AFAData(
        name=name,
        X_train=X_tr,
        y_train=np.ascontiguousarray(y[tr], dtype=np.int64),
        X_val=X_va,
        y_val=np.ascontiguousarray(y[va], dtype=np.int64),
        X_test=X_te,
        y_test=np.ascontiguousarray(y[te], dtype=np.int64),
        n_classes=n_classes,
        groups=groups,
        imbalanced=_is_imbalanced(y, n_classes),
        norm_mu=mu,
        norm_sd=sd,
    )
