"""Tests for experiment.data (shapes, determinism, generators, normalisation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiment.data import (
    AFAData,
    load_dataset,
    make_cube,
    make_cube_nm,
    make_synmix,
    make_toy_xor,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# name -> (n_train, n_val, n_test, d, n_classes, n_groups)
SYNTHETIC = {
    "cube": (12000, 1000, 1000, 20, 8, 20),
    "cube_nm": (12000, 1000, 1000, 55, 8, 51),
    "synmix_0.5": (12000, 1000, 1000, 20, 2, 20),
    "synmix_0": (12000, 1000, 1000, 20, 2, 20),
    "synmix_1": (12000, 1000, 1000, 20, 2, 20),
    "toy_xor": (1200, 400, 400, 3, 2, 3),
    "smoke": (1200, 400, 400, 20, 2, 20),
}

# name -> (raw file, d, n_classes, n_groups); sizes depend on the file, caps checked separately
REAL = {
    "diabetes": ("diabetes.csv", 45, 3, 45),
    "miniboone": ("miniboone.csv", 50, 2, 50),
    "physionet": ("physionet.csv", 41, 2, 41),
    "bank": ("bank.csv", 47, 2, 15),  # duration dropped (leakage)
    "actg175": ("actg175.csv", 22, 2, 22),  # time dropped (leakage)
    "ckd": ("ckd.csv", 24, 2, 24),
}

_CACHE: dict[tuple[str, int], AFAData] = {}


def _load(name: str, seed: int = 0) -> AFAData:
    if name in REAL and not (DATA_DIR / REAL[name][0]).exists():
        pytest.skip(f"raw file for {name} missing; run scripts/fetch_data.py")
    key = (name, seed)
    if key not in _CACHE:
        _CACHE[key] = load_dataset(name, seed, DATA_DIR)
    return _CACHE[key]


def _splits(data: AFAData):
    return [
        (data.X_train, data.y_train),
        (data.X_val, data.y_val),
        (data.X_test, data.y_test),
    ]


def _check_common(data: AFAData, d: int, n_classes: int, n_groups: int) -> None:
    assert data.n_classes == n_classes
    for X, y in _splits(data):
        assert X.dtype == np.float32
        assert y.dtype == np.int64
        assert X.ndim == 2 and X.shape[1] == d
        assert y.shape == (X.shape[0],)
        assert np.isfinite(X).all(), "NaN/inf after loading"
        assert y.min() >= 0 and y.max() < n_classes
    # every class occurs in train
    assert set(np.unique(data.y_train)) == set(range(n_classes))
    # groups partition the feature indices
    assert len(data.groups) == n_groups
    flat = [i for g in data.groups for i in g]
    assert sorted(flat) == list(range(d))
    assert all(len(g) >= 1 for g in data.groups)
    assert isinstance(data.imbalanced, bool)


# ---------------------------------------------------------------- synthetic


@pytest.mark.parametrize("name", list(SYNTHETIC))
def test_synthetic_shapes(name):
    n_tr, n_va, n_te, d, c, n_g = SYNTHETIC[name]
    data = _load(name)
    assert data.name == name
    assert data.X_train.shape == (n_tr, d)
    assert data.X_val.shape == (n_va, d)
    assert data.X_test.shape == (n_te, d)
    _check_common(data, d, c, n_g)


@pytest.mark.parametrize("name", ["cube", "cube_nm", "synmix_0.5", "toy_xor", "smoke"])
def test_synthetic_not_imbalanced(name):
    assert _load(name).imbalanced is False


@pytest.mark.parametrize("name", ["smoke", "toy_xor", "cube"])
def test_determinism_same_seed(name):
    a = load_dataset(name, 3, DATA_DIR)
    b = load_dataset(name, 3, DATA_DIR)
    for (Xa, ya), (Xb, yb) in zip(_splits(a), _splits(b)):
        assert np.array_equal(Xa, Xb)
        assert np.array_equal(ya, yb)
    assert a.groups == b.groups


@pytest.mark.parametrize("name", ["smoke", "toy_xor", "cube_nm"])
def test_different_seed_differs(name):
    a = _load(name, 0)
    b = _load(name, 1)
    assert not np.array_equal(a.X_train, b.X_train)
    assert not np.array_equal(a.X_test, b.X_test)


def test_default_groups_are_singletons():
    data = _load("cube")
    assert data.groups == [[i] for i in range(20)]


def test_cube_nm_groups():
    data = _load("cube_nm")
    assert len(data.groups) == 51
    assert data.groups[0] == [0, 1, 2, 3, 4]
    assert data.groups[1:] == [[i] for i in range(5, 55)]


def test_train_only_normalisation():
    data = _load("synmix_0.5")
    mu = data.X_train.astype(np.float64).mean(0)
    sd = data.X_train.astype(np.float64).std(0)
    assert np.allclose(mu, 0.0, atol=1e-4)
    assert np.allclose(sd, 1.0, atol=1e-3)
    # val/test use train statistics, so their means are not exactly 0
    assert np.any(data.X_val.mean(0) != 0.0)
    assert np.any(data.X_test.mean(0) != 0.0)


def test_constant_feature_guarded():
    # synmix_0: z is always 0 -> zero std on train; must not produce NaN/inf
    data = _load("synmix_0")
    assert np.isfinite(data.X_train).all()
    assert np.all(data.X_train[:, 0] == 0.0)


@pytest.mark.parametrize("bad", ["nope", "synmix_1.5", "synmix_-0.1", "synmix_x"])
def test_unknown_name_raises(bad):
    with pytest.raises(ValueError):
        load_dataset(bad, 0, DATA_DIR)


def test_missing_raw_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset("ckd", 0, tmp_path)


# ---------------------------------------------------------------- generators


def test_synmix_rule_raw():
    rng = np.random.default_rng(0)
    X, y = make_synmix(40000, 0.5, rng)
    assert X.shape == (40000, 20)
    z = X[:, 0]
    assert set(np.unique(z)) == {0.0, 1.0}
    assert abs(z.mean() - 0.5) < 0.02
    xor = ((X[:, 1] > 0) ^ (X[:, 2] > 0)).astype(np.int64)
    single = (X[:, 3] > 0).astype(np.int64)
    on = z == 1
    agree_on = (y[on] == xor[on]).mean()
    agree_off = (y[~on] == single[~on]).mean()
    # 5% label flips -> ~95% agreement on each branch
    assert 0.93 < agree_on < 0.97
    assert 0.93 < agree_off < 0.97
    # x3 carries no signal on z = 1 rows, x1 alone none on any rows
    assert abs((y[on] == single[on]).mean() - 0.5) < 0.03
    assert abs((y == (X[:, 1] > 0)).mean() - 0.5) < 0.03


@pytest.mark.parametrize("rho", [0.0, 1.0])
def test_synmix_extreme_rho(rho):
    X, _ = make_synmix(1000, rho, np.random.default_rng(1))
    assert np.all(X[:, 0] == rho)


def test_synmix_rule_after_loading():
    data = _load("synmix_0.5")
    X, y = data.X_train, data.y_train
    on = X[:, 0] > 0  # z = 1 rows (normalised z is positive iff z = 1)
    assert 0.4 < on.mean() < 0.6
    xor = ((X[:, 1] > 0) ^ (X[:, 2] > 0)).astype(np.int64)
    assert (y[on] == xor[on]).mean() > 0.9


def test_toy_xor_rule_raw():
    X, y = make_toy_xor(2000, np.random.default_rng(0))
    assert X.shape == (2000, 3)
    assert np.array_equal(y, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64))


def test_cube_generator_structure():
    X, y = make_cube(8000, seed=0)
    assert X.shape == (8000, 20) and y.shape == (8000,)
    assert set(np.unique(y)) == set(range(8))
    # class k: features k, k+1, k+2 hold the 3-bit code of k (LSB first), sigma 0.3
    for k in range(8):
        bits = np.array([(k >> b) & 1 for b in range(3)], dtype=np.float64)
        rows = X[y == k]
        assert np.allclose(rows[:, k : k + 3].mean(0), bits, atol=0.05)
        assert np.allclose(rows[:, k : k + 3].std(0), 0.3, atol=0.05)
    # dummy features: N(0.5, 0.3)
    assert np.allclose(X[:, 10:].mean(0), 0.5, atol=0.03)
    assert np.allclose(X[:, 10:].std(0), 0.3, atol=0.03)


def test_cube_generator_seeded():
    a = make_cube(500, seed=7)
    b = make_cube(500, seed=7)
    c = make_cube(500, seed=8)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_cube_nm_generator_structure():
    X, y, ctx = make_cube_nm(5000, seed=0, return_context=True)
    assert X.shape == (5000, 55)
    # noisy one-hot context in the first 5 features
    assert np.array_equal(X[:, :5].argmax(1), ctx)
    # only the context block carries the code
    for k in (3, 5):
        bits = np.array([(k >> b) & 1 for b in range(3)], dtype=np.float64)
        idx = [(k + j) % 10 for j in range(3)]
        rows = (y == k) & (ctx == 2)
        block = X[rows][:, 5 + 2 * 10 : 5 + 3 * 10]
        assert np.allclose(block[:, idx].mean(0), bits, atol=0.1)
        other = X[rows][:, 5 : 5 + 10]  # block 0 is uninformative here
        assert np.allclose(other.mean(0), 0.5, atol=0.1)


# ---------------------------------------------------------------- real data


@pytest.mark.parametrize("name", list(REAL))
def test_real_shapes(name):
    _, d, c, n_g = REAL[name]
    data = _load(name)
    assert data.name == name
    _check_common(data, d, c, n_g)
    assert data.X_train.shape[0] <= 20000
    assert data.X_val.shape[0] <= 1000
    assert data.X_test.shape[0] <= 1000


@pytest.mark.parametrize("name", ["miniboone", "diabetes", "bank"])
def test_real_caps_hit(name):
    data = _load(name)
    assert data.X_train.shape[0] == 20000
    assert data.X_val.shape[0] == 1000
    assert data.X_test.shape[0] == 1000


@pytest.mark.parametrize(
    "name,imbalanced",
    [("diabetes", True), ("bank", True), ("physionet", True), ("ckd", False)],
)
def test_real_imbalance_flag(name, imbalanced):
    assert _load(name).imbalanced is imbalanced


def test_real_stratified_uncapped():
    data = _load("ckd")  # 400 rows -> 240 / 80 / 80, no caps
    assert (data.X_train.shape[0], data.X_val.shape[0], data.X_test.shape[0]) == (240, 80, 80)
    p_tr = data.y_train.mean()
    assert abs(data.y_val.mean() - p_tr) < 0.02
    assert abs(data.y_test.mean() - p_tr) < 0.02


def test_real_label_definitions():
    bank = _load("bank")
    assert 0.10 < bank.y_train.mean() < 0.13  # y = "yes" share is 11.7%
    ckd = _load("ckd")
    assert 0.60 < ckd.y_train.mean() < 0.65  # ckd = 1 (248 + 2 of 400)


def test_real_determinism_and_seed():
    a = load_dataset("ckd", 5, DATA_DIR) if (DATA_DIR / "ckd.csv").exists() else pytest.skip("ckd missing")
    b = load_dataset("ckd", 5, DATA_DIR)
    c = load_dataset("ckd", 6, DATA_DIR)
    for (Xa, ya), (Xb, yb) in zip(_splits(a), _splits(b)):
        assert np.array_equal(Xa, Xb) and np.array_equal(ya, yb)
    assert not np.array_equal(a.X_test, c.X_test)


def test_real_train_only_normalisation():
    data = _load("physionet")
    assert np.allclose(data.X_train.astype(np.float64).mean(0), 0.0, atol=1e-4)
    assert np.any(data.X_test.mean(0) != 0.0)
