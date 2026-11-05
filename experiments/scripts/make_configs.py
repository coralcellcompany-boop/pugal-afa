"""Generate experiments/configs/*.toml for the pre-registered matrix in notes/experiments.md.

Run from the experiments directory: `uv run python scripts/make_configs.py`. Idempotent: rewrites every config.
"""
from __future__ import annotations

from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

MAIN = ["cube", "cube_nm", "synmix_0.5", "diabetes", "miniboone", "physionet", "bank", "actg175", "ckd"]
SWEEP = ["synmix_0", "synmix_0.1", "synmix_0.25", "synmix_0.75", "synmix_1"]
ABLATION_DATASETS = ["cube_nm", "synmix_0.5", "physionet", "miniboone"]

BUDGET = {d: 6 for d in ["synmix_0.5", *SWEEP]} | {d: 10 for d in MAIN if not d.startswith("synmix")}

# method -> fixed settings (hparams grid is added for tuned methods)
METHODS = {
    "random": {"policy": "random"},
    "static": {"policy": "static"},
    "greedy": {"policy": "greedy"},
    "greedy_log": {"policy": "greedy", "risk": "log"},
    "la2": {"policy": "lookahead", "h": 2, "candidates": "all"},
    "la3": {"policy": "lookahead", "h": 3, "candidates": "topm_flagged", "n_neighbors": 8},
    "aaco": {"policy": "aaco", "k_nn": 5, "n_subsets": 1000},
    "pugal": {"policy": "pugal", "gate": "synergy"},
    "pugal_learn": {"policy": "pugal", "gate": "learned"},
    "pugal_margin": {"policy": "pugal", "gate": "margin"},
    "pugal_rand": {"policy": "pugal", "gate": "random_matched"},
}
ABLATIONS = {
    "pugal_nosyn": {"gate_terms": "redundancy"},
    "pugal_noredund": {"gate_terms": "synergy"},
    "pugal_noscreen": {"screen": "all_pairs"},
    "pugal_nofallback": {"conf_fallback": False},
    "pugal_h3": {"h_max": 3},
    "pugal_m4": {"m": 4},
    "pugal_m16": {"m": 16},
    "pugal_loss01": {"risk": "01"},
    "pugal_losslog": {"risk": "log"},
}
PUGAL_GRID = {"eps": [0.005, 0.01, 0.02], "screen_quantile": [0.90, 0.95, 0.99], "m": [4, 8]}
# margin gate: tau = eps, so it gets wider values (decision 2026-11-03)
MARGIN_GRID = {"eps": [0.005, 0.01, 0.02, 0.05, 0.1], "screen_quantile": [0.90, 0.95, 0.99], "m": [4, 8]}
# methods tuned on their own grid; every other pugal_* (ablations, pugal_rand) reuses pugal's tuned choice
OWN_GRID = {"pugal": PUGAL_GRID, "pugal_learn": PUGAL_GRID, "pugal_margin": MARGIN_GRID}
# P2 (decision 2026-11-03): first 500 test instances for EVERY config of these datasets (paired across methods)
TEST_CAP_500 = {"cube_nm", "diabetes", "miniboone", "physionet"}
COMMON = {"risk": "brier", "n_neighbors": 16, "pool_size": 5000, "h0": 2, "h_max": 2, "m": 8, "cand_cap": 16, "conf_fallback": True}


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    return repr(v)


def write(name: str, dataset: str, method: str, settings: dict, grid: dict | None, note: str) -> None:
    lines = [f"# {note}", f'dataset = "{dataset}"', f'method = "{method}"', f"budget = {BUDGET[dataset]}"]
    if dataset in TEST_CAP_500 and "test_cap" not in settings:
        settings = settings | {"test_cap": 500}
    lines += [f"{k} = {toml_value(v)}" for k, v in settings.items()]
    if grid:
        lines += ["", "[hparams]"] + [f"{k} = {toml_value(v)}" for k, v in grid.items()]
    (CONFIGS / f"{name}.toml").write_text("\n".join(lines) + "\n")


def settings_for(method: str, base: dict) -> dict:
    s = dict(COMMON)
    s.update(base)
    return s


def main() -> None:
    CONFIGS.mkdir(exist_ok=True)
    for old in CONFIGS.glob("*.toml"):
        old.unlink()
    n = 0
    for d in MAIN:
        for method, base in METHODS.items():
            grid = OWN_GRID.get(method)
            write(f"{d}_{method}", d, method, settings_for(method, base), grid, "E2 main comparison"); n += 1
        write(f"{d}_gatequality", d, "gatequality", settings_for("gq", {"policy": "gatequality"}), PUGAL_GRID,
              "E1 gate AUROC vs exact LA2 labels on greedy-visited states"); n += 1
    for d in SWEEP:
        for method in ["greedy", "la2", "pugal", "pugal_rand"]:
            grid = OWN_GRID.get(method)
            write(f"{d}_{method}", d, method, settings_for(method, METHODS[method]), grid, "E5 rho sweep"); n += 1
        write(f"{d}_gatequality", d, "gatequality", settings_for("gq", {"policy": "gatequality"}), PUGAL_GRID,
              "E1/E5 gate AUROC on the rho sweep"); n += 1
    for d in ABLATION_DATASETS:
        for abl, delta in ABLATIONS.items():
            s = settings_for("pugal", METHODS["pugal"]); s.update(delta)
            # no grid: reuses the dataset's tuned pugal choice; only the ablated key changes (P1)
            write(f"{d}_{abl}", d, abl, s, None, "E4 ablation (pugal's tuned eps/screen_quantile/m)"); n += 1
    BUDGET["smoke"] = 3
    write("smoke", "smoke", "pugal", settings_for("pugal", METHODS["pugal"]) | {"n_neighbors": 8},
          {"eps": [0.01], "screen_quantile": [0.95], "m": [4]}, "E0 smoke test, never in tables"); n += 1
    BUDGET["miniboone"] = 10
    write("smoke_miniboone", "miniboone", "la2", settings_for("la2", METHODS["la2"]) | {"budget_override": 2, "test_cap": 20},
          None, "E0 smoke on the largest dataset with the slowest method, never in tables"); n += 1
    print(f"wrote {n} configs to {CONFIGS}")


if __name__ == "__main__":
    main()
