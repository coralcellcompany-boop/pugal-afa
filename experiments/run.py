"""Run one experiment from a TOML config and write results/<run_id>.json."""
from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiment.core import run_experiment


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one seeded experiment from a TOML config.")
    ap.add_argument("config", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    args = ap.parse_args()

    config = tomllib.loads(args.config.read_text())
    name = args.config.stem
    set_seeds(args.seed)
    started = datetime.now(timezone.utc)
    metrics = run_experiment(config, args.seed)
    run_id = f"{name}_seed{args.seed}"   # deterministic: a re-run overwrites its own file
    record = {
        "run_id": run_id,
        "timestamp": started.isoformat(timespec="seconds"),
        "config_name": name,
        "config": config,
        "seed": args.seed,
        "metrics": metrics,
        "meta": {
            "git_commit": git_commit(),
            "python": sys.version.split()[0],
            "host": platform.node(),
            "duration_s": round((datetime.now(timezone.utc) - started).total_seconds(), 3),
        },
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / f"{run_id}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"{run_id}: {json.dumps(metrics)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
