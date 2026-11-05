"""Run every config x seed in a process pool (notes/experiments.md; DESIGN.md).

    uv run python scripts/run_all.py [--workers 10] [--only 'cube*'] [--smoke] [--dry-run] [--seeds 0 1 2 3 4]
    uv run python scripts/run_all.py --measure      # re-measure the per-dataset timing table (prints JSON)

Jobs and order (a job starts only when its dependencies have finished):
  warm   <dataset> seed N   classifier + brier synergy screen  (python -m experiment.core warm)
  la2ref <dataset>          always-LA2 validation acc-AUC, seed 0 (selection reference)
  tune   <dataset>_<method> hyperparameter choice on seed 0 validation (after warm seed 0 and la2ref)
  run    <config> seed N    `uv run python run.py <config> --seed N` (after warm, its tune, the dataset's pugal
                            tune if it uses PUGAL's choice, and for pugal_rand the same-seed pugal run)
Ready jobs start longest-estimate first. Each subprocess gets OMP_NUM_THREADS=1 and PUGAL_TORCH_THREADS=1.
Run jobs whose results/<config>_seed<N>.json exists are skipped. Logs: runs/logs/<job>.log.
"""

from __future__ import annotations

import argparse
import fnmatch
import heapq
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
LOGS = ROOT / "runs" / "logs"
SWEEP = {"synmix_0", "synmix_0.1", "synmix_0.25", "synmix_0.75", "synmix_1"}

# Measured with `--measure` on an Apple M3 Pro, 10 processes in parallel (1 torch thread each), seed 0,
# averaged over the states of greedy trajectories of 2 test instances. Seconds per state unless noted.
# Keys: n_test, budget, groups, train_s (classifier training, s), g (greedy step), la2, roll8 / roll16
# (h = 2 rollout over 8 / 16 candidates), roll16_h3 (h = 3, 16 candidates), la3 (n = 8, 16 candidates,
# h = 3), aaco (1000 subsets), R (redundancy), nflag (N_hat over 5% of pairs), nall (N over all pairs),
# table (two-step table = exact Delta_2 + pair gains), screen (one screened state).
TIMING = {}   # filled from --measure output below
TIMING_FILE = Path(__file__).with_name("timing_table.json")

OPEN_RATE_GUESS = 0.30          # assumed PUGAL opening rate for estimates (H2's pre-registered ceiling)
PROC_OVERHEAD = 6.0             # python + torch import + data/model load per subprocess, seconds


# ------------------------------------------------------------------ jobs

@dataclass
class Job:
    name: str
    kind: str                      # warm | la2ref | tune | run
    cmd: list[str]
    log: Path
    deps: list[str] = field(default_factory=list)
    result: Path | None = None
    est: float = 0.0
    block: str = "setup"
    done: bool = False


def _core():
    sys.path.insert(0, str(ROOT / "src"))
    import experiment.core as core
    return core


def _block(stem: str, cfg: dict) -> str:
    if stem.startswith("smoke"):
        return "E0 smoke"
    if cfg["policy"] == "gatequality":
        return "E1 gatequality"
    if cfg["dataset"] in SWEEP:
        return "E5 sweep"
    if cfg["method"].startswith("pugal_") and cfg["method"] not in ("pugal_learn", "pugal_margin", "pugal_rand"):
        return "E4 ablation"
    return f"E2 {cfg['method']}"


def build_jobs(only: list[str] | None, smoke: bool, seeds: list[int]) -> tuple[dict[str, Job], list[str]]:
    core = _core()
    notes = []
    paths = sorted(CONFIGS.glob("*.toml"))
    cfgs = {p.stem: core.load_config(p) for p in paths}
    selected = [s for s in cfgs if (smoke or not s.startswith("smoke"))]
    if only:
        selected = [s for s in selected if any(fnmatch.fnmatch(s, g) for g in only)]
    # pugal_rand needs the same-seed pugal result: pull in the pugal config when its results are missing
    for s in list(selected):
        c = cfgs[s]
        if c["policy"] == "pugal" and c.get("gate") == "random_matched":
            ps = f"{c['dataset']}_pugal"
            if ps not in selected and any(not (RESULTS / f"{ps}_seed{n}.json").exists() for n in seeds):
                selected.append(ps)
                notes.append(f"added {ps} (needed by {s})")
    jobs: dict[str, Job] = {}

    def py(*args):
        return ["uv", "run", "python", *args]

    def warm(dataset: str, seed: int, own: str) -> str:
        name = f"warm:{dataset}:s{seed}"
        if name not in jobs:
            src = CONFIGS / f"{dataset}_pugal.toml"
            src = src if src.exists() else CONFIGS / f"{own}.toml"
            jobs[name] = Job(name, "warm", py("-m", "experiment.core", "warm", str(src), "--seed", str(seed)),
                             LOGS / f"warm_{dataset}_seed{seed}.log")
        return name

    def tune(stem: str) -> str:
        c = cfgs[stem]
        name = f"tune:{stem}"
        if name not in jobs:
            deps = [warm(c["dataset"], 0, stem)]
            if len(core._grid(c)) > 1:
                ref = f"la2ref:{c['dataset']}"
                if ref not in jobs:
                    jobs[ref] = Job(ref, "la2ref", py("-m", "experiment.core", "la2ref", str(CONFIGS / f"{stem}.toml")),
                                    LOGS / f"la2ref_{c['dataset']}.log", deps=[warm(c["dataset"], 0, stem)])
                deps.append(ref)
            jobs[name] = Job(name, "tune", py("-m", "experiment.core", "tune", str(CONFIGS / f"{stem}.toml")),
                             LOGS / f"tune_{stem}.log", deps=deps)
        return name

    for stem in sorted(selected):
        c = cfgs[stem]
        uses_pugal = core.uses_pugal_hparams(c)
        for n in seeds:
            name = f"run:{stem}:s{n}"
            deps = [warm(c["dataset"], n, stem)]
            if c.get("hparams") and not uses_pugal:
                deps.append(tune(stem))
            pstem = f"{c['dataset']}_pugal"
            if uses_pugal and pstem in cfgs:
                deps.append(tune(pstem))
            if c.get("gate") == "random_matched":
                deps.append(f"run:{pstem}:s{n}")
            res = RESULTS / f"{stem}_seed{n}.json"
            jobs[name] = Job(name, "run", py("run.py", str(CONFIGS / f"{stem}.toml"), "--seed", str(n)),
                             LOGS / f"{stem}_seed{n}.log", deps=deps, result=res, block=_block(stem, c),
                             done=res.exists())
    for j in jobs.values():
        j.deps = [d for d in j.deps if d in jobs or not d.startswith("run:")]
        missing = [d for d in j.deps if d not in jobs]
        if missing:   # a pugal run that is not selected but whose result exists
            j.deps = [d for d in j.deps if d in jobs]
    return jobs, notes


# ------------------------------------------------------------------ estimates

def _timing() -> dict:
    if TIMING_FILE.exists():
        return json.loads(TIMING_FILE.read_text())
    return TIMING


def estimate(job: Job, cfgs: dict, T: dict) -> float:
    """Seconds for one job, from the per-dataset timing table (see TIMING)."""
    if job.kind == "warm":
        ds = job.name.split(":")[1]
        t = T.get(ds) or T.get("synmix_0.5")
        core = _core()
        return PROC_OVERHEAD + t["train_s"] + core.SCREEN_STATES * t["screen"]
    stem = job.name.split(":")[1]
    if job.kind == "la2ref":
        t = T.get(stem) or T.get("synmix_0.5")
        return PROC_OVERHEAD + min(300, t["n_val"]) * t["budget"] * t["la2"]
    c = cfgs[stem]
    t = T.get(c["dataset"]) or T.get("synmix_0.5")
    B = int(c.get("budget_override", c["budget"]))
    roll = 0.5 * (t["roll8"] + t["roll16"])
    rho = OPEN_RATE_GUESS

    def pugal_state(cc) -> float:
        gate = cc.get("gate", "synergy")
        sg = 0.0 if gate in ("margin", "random_matched") else t["R"] + t["nflag"]
        if cc.get("screen") == "all_pairs":
            sg = t["R"] + t["nall"]
        if cc.get("gate_terms") == "redundancy":
            sg = t["R"]
        r = t["roll16"] if cc.get("m") == 16 else (t["roll8"] * 0.6 if cc.get("m") == 4 else roll)
        if cc.get("h_max", 2) >= 3:
            r += 0.8 * t["roll16_h3"]
        return t["g"] + sg + rho * r * (B - 1) / B

    if job.kind == "tune":
        n_val = min(300, t["n_val"])
        core = _core()
        combos = len(core._grid(c))
        extra = GATE_TRAIN_EST(t) if c.get("gate") == "learned" else 0.0
        return PROC_OVERHEAD + combos * n_val * B * pugal_state(c) + extra
    N = min(int(c.get("test_cap", 10 ** 9)), t["n_test"])
    pol = c["policy"]
    per = {"random": 0.0005, "static": 0.0005, "greedy": t["g"], "aaco": t["aaco"]}.get(pol)
    if pol == "lookahead":
        per = t["la2"] if c.get("candidates") == "all" else t["la3"]
    if pol == "pugal":
        per = pugal_state(c)
    extra = 0.0
    if pol == "static":
        extra = 200 * B * t["g"]
    if c.get("risk") in ("01", "log") and pol == "pugal":
        extra += 200 * t["screen"]          # screen with another risk, once per seed
    if c.get("gate") == "learned":
        extra += GATE_TRAIN_EST(t)
    if pol == "gatequality":
        Ng = min(300, t["n_test"])
        return PROC_OVERHEAD + Ng * B * t["g"] + Ng * (B - 1) * (t["table"] + t["R"] + t["nflag"]) + GATE_TRAIN_EST(t)
    return PROC_OVERHEAD + N * B * per + extra


def GATE_TRAIN_EST(t: dict) -> float:
    return 300 * (t["la2"] + t["R"] + t["g"])


def simulate(jobs: dict[str, Job], workers: int) -> float:
    """Makespan of the dependency-aware longest-first schedule with the estimated durations."""
    done = {n for n, j in jobs.items() if j.done}
    pending = {n for n in jobs if n not in done}
    clock, running = 0.0, []                     # heap of (finish time, name)
    while pending or running:
        ready = sorted((n for n in pending if all(d in done for d in jobs[n].deps)), key=lambda n: -jobs[n].est)
        while ready and len(running) < workers:
            n = ready.pop(0)
            pending.discard(n)
            heapq.heappush(running, (clock + jobs[n].est, n))
        if not running:
            break                                # unsatisfiable dependencies
        clock, n = heapq.heappop(running)
        done.add(n)
    return clock


# ------------------------------------------------------------------ execution

def run(jobs: dict[str, Job], workers: int) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"OMP_NUM_THREADS": "1", "PUGAL_TORCH_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env.pop("VIRTUAL_ENV", None)
    done = {n for n, j in jobs.items() if j.done}
    failed: set[str] = set()
    pending = {n for n in jobs if n not in done}
    total = len(pending)
    finished = 0
    t_start = time.time()

    def execute(job: Job) -> tuple[str, int, float]:
        t0 = time.time()
        with open(job.log, "w") as fh:
            rc = subprocess.run(job.cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc == 0 and job.result is not None and not job.result.exists():
            rc = 99
        return job.name, rc, time.time() - t0

    print(f"{total} jobs to run ({len(done)} already done), {workers} workers", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        running = {}
        while pending or running:
            for n in sorted(pending):
                if any(d in failed for d in jobs[n].deps):
                    pending.discard(n)
                    failed.add(n)
                    finished += 1
                    print(f"[{finished}/{total}] SKIP {n}: a dependency failed", flush=True)
            ready = sorted((n for n in pending if all(d in done for d in jobs[n].deps)), key=lambda n: -jobs[n].est)
            for n in ready:
                if len(running) >= workers:
                    break
                pending.discard(n)
                running[pool.submit(execute, jobs[n])] = n
            if not running:
                if pending:
                    print(f"stuck: {len(pending)} jobs have unmet dependencies", flush=True)
                break
            fin, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in fin:
                del running[f]
                name, rc, sec = f.result()
                finished += 1
                if rc == 0:
                    done.add(name)
                    print(f"[{finished}/{total}] ok   {name} {sec:.0f}s  (elapsed {(time.time() - t_start) / 60:.1f} min)",
                          flush=True)
                else:
                    failed.add(name)
                    print(f"[{finished}/{total}] FAIL {name} rc={rc} after {sec:.0f}s, log: {jobs[name].log}", flush=True)
    print(f"finished: {len(done)} ok, {len(failed)} failed or skipped", flush=True)
    for n in sorted(failed):
        print(f"  failed: {n}  ({jobs[n].log})")
    return 1 if failed else 0


# ------------------------------------------------------------------ timing measurement

def measure_one(dataset: str) -> dict:
    import numpy as np

    core = _core()
    from experiment.policies import AACOPolicy
    from experiment.utility import State

    cfg = core.load_config(CONFIGS / f"{dataset}_pugal.toml")
    t0 = time.perf_counter()
    res = core._resources(cfg, 0)                     # trains (and caches) the seed-0 classifier if needed
    train_s = time.perf_counter() - t0                # ~0 if the classifier was already cached
    eng, eng8 = res.engine("brier", 16), res.engine("brier", 8)
    B, G = int(cfg["budget"]), len(res.groups)
    rng = np.random.default_rng(0)
    acc = {k: [] for k in ("g", "la2", "roll8", "roll16", "roll16_h3", "la3", "aaco", "R", "nflag", "nall",
                           "table", "screen")}

    def tm(key, fn):
        t = time.perf_counter()
        out = fn()
        acc[key].append(time.perf_counter() - t)
        return out

    aaco = AACOPolicy(eng, 5, 1000, seed=0)
    for i in range(2):
        x = res.data.X_test[i]
        st = State.empty(res.d, B)
        for step in range(B):
            ctx = tm("g", lambda: eng.greedy_context(st))
            free = ctx.order
            if st.budget >= 2 and len(free) >= 2:
                tm("la2", lambda: eng.rollout_values(st, free, 2))
                tm("roll8", lambda: eng.rollout_values(st, free[:8], 2))
                tm("roll16", lambda: eng.rollout_values(st, free[:16], 2))
                if i == 0 and step < 3:
                    tm("roll16_h3", lambda: eng.rollout_values(st, free[:16], 3))
                    tm("la3", lambda: eng8.rollout_values(st, free[:16], 3))
                tm("R", lambda: eng.redundancy(st, ctx.g, ctx.j2, u1=ctx.u1))
                pairs = [(j, k) for j in free for k in free if j != k]
                pick = rng.choice(len(pairs), max(1, len(pairs) // 20), replace=False)
                tm("nflag", lambda: eng.synergy_term(st, ctx.g, [pairs[p] for p in pick], u1=ctx.u1))
                tm("nall", lambda: eng.synergy_term(st, ctx.g, None, u1=ctx.u1))
                tm("table", lambda: eng.two_step_table(st, ctx))
                tm("screen", lambda: eng.synergy_pairs(st, pairs, u1=ctx.u1))
                tm("aaco", lambda: aaco.select(st))
            st = st.acquire(ctx.g, x, res.groups)
    out = {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}
    out |= {"n_test": len(res.data.X_test), "n_val": len(res.data.X_val), "budget": B, "groups": G,
            "d": res.d, "train_s": float(train_s), "train_epochs": (getattr(res.model, "train_info", {}) or {}).get("epochs")}
    return out


def measure_all(workers: int) -> dict:
    core = _core()
    datasets = sorted({core.load_config(p)["dataset"] for p in CONFIGS.glob("*_pugal.toml")} - {"smoke"} - SWEEP)
    env = os.environ | {"OMP_NUM_THREADS": "1", "PUGAL_TORCH_THREADS": "1"}
    env.pop("VIRTUAL_ENV", None)
    with ThreadPoolExecutor(workers) as pool:
        outs = pool.map(lambda ds: subprocess.run(
            ["uv", "run", "python", str(Path(__file__)), "--measure-one", ds], cwd=ROOT, env=env,
            capture_output=True, text=True), datasets)
        table = {}
        for ds, o in zip(datasets, outs):
            if o.returncode != 0:
                print(f"measure {ds} failed:\n{o.stderr[-2000:]}", file=sys.stderr)
                continue
            table[ds] = json.loads(o.stdout.strip().splitlines()[-1])
    return table


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only", nargs="+", help="config-name globs, e.g. 'cube_*' '*_la2'")
    ap.add_argument("--smoke", action="store_true", help="include configs whose name starts with smoke")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--measure", action="store_true", help="measure the timing table and write it next to this script")
    ap.add_argument("--measure-one", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.measure_one:
        print(json.dumps(measure_one(args.measure_one)))
        return 0
    if args.measure:
        table = measure_all(args.workers)
        TIMING_FILE.write_text(json.dumps(table, indent=1))
        print(json.dumps(table, indent=1))
        return 0

    jobs, notes = build_jobs(args.only, args.smoke, args.seeds)
    for n in notes:
        print("note:", n)
    T = _timing()
    core = _core()
    cfgs = {p.stem: core.load_config(p) for p in CONFIGS.glob("*.toml")}
    for j in jobs.values():
        j.est = estimate(j, cfgs, T) if T else 0.0
    if args.dry_run:
        todo = [j for j in jobs.values() if not j.done]
        for j in sorted(todo, key=lambda j: (["warm", "la2ref", "tune", "run"].index(j.kind), j.name)):
            print(f"{j.kind:6s} {j.name:48s} est {j.est / 60:7.1f} min  deps={len(j.deps)}")
        print(f"\n{len(todo)} jobs to run ({sum(j.done for j in jobs.values())} already done); "
              f"assumed PUGAL opening rate {OPEN_RATE_GUESS}")
        blocks: dict[str, list[Job]] = {}
        for j in todo:
            blocks.setdefault(j.block, []).append(j)
        print(f"\n{'block':22s} {'jobs':>5s} {'CPU-h':>8s} {'wall-h @' + str(args.workers):>10s} {'longest job (min)':>18s}")
        for b in sorted(blocks):
            cpu = sum(j.est for j in blocks[b])
            print(f"{b:22s} {len(blocks[b]):5d} {cpu / 3600:8.1f} {cpu / 3600 / args.workers:10.2f} "
                  f"{max(j.est for j in blocks[b]) / 60:18.1f}")
        cpu = sum(j.est for j in todo)
        print(f"{'TOTAL':22s} {len(todo):5d} {cpu / 3600:8.1f} {cpu / 3600 / args.workers:10.2f}")
        print(f"simulated makespan with dependencies: {simulate(jobs, args.workers) / 3600:.2f} h at "
              f"{args.workers} workers")
        return 0
    return run(jobs, args.workers)


if __name__ == "__main__":
    sys.exit(main())
