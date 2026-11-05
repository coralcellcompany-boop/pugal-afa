"""Paper figures from experiments/results/*.json -> paper/figures/*.pdf (vector).

Palette: dataviz reference categorical slots 1-4 in fixed order (validated: light mode, adjacent pairs pass;
two slots below 3:1 contrast, so every line carries a direct label and a distinct marker).
Each method keeps its colour in every figure (colour follows the entity, not its rank).
Run from the experiments directory: `uv run python scripts/plot_figures.py`. Missing results are skipped.
"""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT.parent / "paper" / "figures"

SERIES = {  # method -> (label, colour, marker); fixed order = fixed colour
    "greedy": ("Greedy", "#2a78d6", "o"),
    "la2": ("Always-LA2", "#eb6834", "s"),
    "pugal": ("PUGAL", "#1baf7a", "D"),
    "pugal_rand": ("Random gate", "#eda100", "^"),
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RHOS = ["0", "0.1", "0.25", "0.5", "0.75", "1"]

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 2.0, "lines.markersize": 5, "pdf.fonttype": 42,
})


def load(config: str) -> list[dict]:
    return [json.loads(Path(p).read_text())["metrics"] for p in sorted(glob.glob(str(RESULTS / f"{config}_seed*.json")))]


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=INK)


def fig_rho_sweep() -> None:
    """E5: accuracy and compute vs rho (two panels, one axis each)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.4))
    drawn = False
    for method, (label, color, marker) in SERIES.items():
        xs, acc, acc_sd, ev = [], [], [], []
        for rho in RHOS:
            name = f"synmix_{rho}_{method}"
            runs = load(name)
            if not runs:
                continue
            m, s = mean_std([r["acc_auc"] for r in runs])
            xs.append(float(rho)); acc.append(m); acc_sd.append(s)
            ev.append(statistics.mean(r["evals_per_inst"] for r in runs))
        if not xs:
            continue
        drawn = True
        a1.errorbar(xs, acc, yerr=acc_sd, color=color, marker=marker, capsize=2, label=label)
        a2.plot(xs, ev, color=color, marker=marker, label=label)
        label_end(a2, xs[-1], ev[-1], label, color)
    if not drawn:
        plt.close(fig); return
    a1.set_xlabel(r"fraction of XOR instances $\rho$"); a1.set_ylabel("acc-AUC (mean over budget)")
    a2.set_xlabel(r"fraction of XOR instances $\rho$"); a2.set_ylabel("classifier evaluations / instance")
    a2.set_yscale("log")
    a1.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "rho_sweep.pdf"); plt.close(fig)
    print("wrote rho_sweep.pdf")


def fig_budget_curves(datasets=("cube_nm", "physionet")) -> None:
    """E2: accuracy after each acquisition, one panel per dataset."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.4, 2.4))
    drawn = False
    for ax, ds in zip(axes, datasets):
        for method, (label, color, marker) in SERIES.items():
            runs = load(f"{ds}_{method}")
            if not runs:
                continue
            ts = sorted(int(k[5:]) for k in runs[0] if k.startswith("acc_t"))
            means = [statistics.mean(r[f"acc_t{t}"] for r in runs) for t in ts]
            ax.plot(ts, means, color=color, marker=marker, label=label, markevery=2)
            drawn = True
        ax.set_title({"cube_nm": "CUBE-NM", "physionet": "PhysioNet"}.get(ds, ds), fontsize=8, color=INK)
        ax.set_xlabel("features acquired"); ax.set_ylabel("test accuracy")
    if not drawn:
        plt.close(fig); return
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "budget_curves.pdf"); plt.close(fig)
    print("wrote budget_curves.pdf")


def fig_nmi() -> None:
    """E3: non-myopia index vs accuracy gain of always-LA2 over greedy (one series, labelled points)."""
    names = {"cube": "CUBE", "cube_nm": "CUBE-NM", "synmix_0.5": "SynMix(0.5)", "diabetes": "Diabetes",
             "miniboone": "MiniBooNE", "physionet": "PhysioNet", "bank": "Bank", "actg175": "ACTG175", "ckd": "CKD"}
    xs, ys, labels = [], [], []
    for ds, label in names.items():
        gq, la2, gr = load(f"{ds}_gatequality"), load(f"{ds}_la2"), load(f"{ds}_greedy")
        if not (gq and la2 and gr):
            continue
        xs.append(statistics.mean(r["nmi"] for r in gq))
        ys.append(statistics.mean(r["acc_auc"] for r in la2) - statistics.mean(r["acc_auc"] for r in gr))
        labels.append(label)
    if len(xs) < 3:
        return
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.scatter(xs, ys, color=SERIES["pugal"][1], s=30, edgecolor="#fcfcfb", linewidth=1, zorder=3)
    for x, y, t in zip(xs, ys, labels):
        ax.annotate(t, (x, y), xytext=(4, 2), textcoords="offset points", fontsize=7, color=INK)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("non-myopia index (NMI)"); ax.set_ylabel("acc-AUC: always-LA2 minus greedy")
    fig.tight_layout()
    fig.savefig(FIGS / "nmi.pdf"); plt.close(fig)
    print("wrote nmi.pdf")


PARETO = {"greedy": SERIES["greedy"], "la2": SERIES["la2"], "la3": ("Always-LA3", "#e87ba4", "v"),
          "pugal": SERIES["pugal"]}   # slot 5 (magenta) for LA3; validated with slots 1-3 (CVD floor band ->
                                      # every series also has its own marker, and the numbers are in tables)
NAMES = {"cube": "CUBE", "cube_nm": "CUBE-NM", "synmix_0.5": "SynMix(0.5)", "diabetes": "Diabetes",
         "miniboone": "MiniBooNE", "physionet": "PhysioNet", "bank": "Bank", "actg175": "ACTG175", "ckd": "CKD"}


def fig_pareto(datasets, out: str, ncols: int, width: float = 6.4, row_h: float = 1.9) -> None:
    """Accuracy vs compute: acc-AUC (mean +- std over seeds) against classifier evaluations per instance
    (log scale), one panel per dataset, one point per method."""
    nrows = -(-len(datasets) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, row_h * nrows + 0.5), squeeze=False)
    drawn = False
    for ax, ds in zip(axes.flat, datasets):
        for method, (label, color, marker) in PARETO.items():
            runs = load(f"{ds}_{method}")
            if not runs:
                continue
            m, s = mean_std([r["acc_auc"] for r in runs])
            ev = statistics.mean(r["evals_per_inst"] for r in runs)
            ax.errorbar([ev], [m], yerr=[s], color=color, marker=marker, markersize=6, capsize=2,
                        markeredgecolor="#fcfcfb", markeredgewidth=0.8, linestyle="none", label=label, zorder=3)
            drawn = True
        ax.set_xscale("log")
        ax.set_title(NAMES.get(ds, ds), fontsize=8, color=INK)
        ax.margins(x=0.25, y=0.25)
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    if not drawn:
        plt.close(fig); return
    for ax in axes[-1]:
        ax.set_xlabel("classifier evaluations / instance")
    for ax in axes[:, 0]:
        ax.set_ylabel("acc-AUC")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93 if nrows == 1 else 0.96))
    fig.savefig(FIGS / out); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_rho_sweep()
    fig_budget_curves()
    fig_nmi()
    fig_pareto(["cube_nm", "synmix_0.5", "physionet"], "pareto_main.pdf", ncols=3)
    fig_pareto(list(NAMES), "pareto_all.pdf", ncols=3)
