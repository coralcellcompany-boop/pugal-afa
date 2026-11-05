# Experiments

Config-driven, seeded runs. Each run writes one JSON file into `results/` with the config,
seed, git commit and metrics. Tables in the paper are generated from these files, never typed by hand.

```bash
cd papers/<slug>/experiments
uv run python run.py configs/baseline.toml --seed 0     # repeat for seeds 1, 2
uv run python run.py configs/small-data.toml --seed 0
cd ../../..
uv run results-table papers/<slug>/experiments/results/*.json --metric mse --metric weight_error \
   --lower-is-better mse --lower-is-better weight_error --out papers/<slug>/paper/tables/main_results.tex
```

- Put the real method in `src/experiment/core.py` (`run_experiment(config, seed) -> dict[str, float]`); keep the signature.
- Add dependencies with `uv add <package>` inside this folder.
- Large artifacts go to `runs/` and datasets to `data/` (both gitignored). Results JSON is small and committed.
- For GPU runs use Runpod (see the experiment-runner skill); copy the `results/*.json` files back here.
- Re-running the same config and seed overwrites its results file (run ids are deterministic).
