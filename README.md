# Knowing When to Look Ahead: Gated Non-Myopic Active Feature Acquisition

Code, configurations, results and pre-registration for the paper by Hossam Hamouda (Coralcell Limited).

Active feature acquisition chooses, for each test instance, which feature to measure next under a budget. Greedy policies are cheap but miss features that are informative only jointly; non-myopic policies plan at every step, although on most steps they would choose what greedy chooses. We propose PUGAL, which acts greedily by default and runs a short greedy rollout only when a gate opens. For any gate, the expected final risk under the decision model equals greedy's minus the rollout advantage collected at the opened states, so opening never hurts in expectation; a redundancy--synergy certificate bounds the two-step advantage and defines the gate. In a pre-registered study on six real and three synthetic tabular datasets, a paired test finds no difference between PUGAL and always-on two-step lookahead on any dataset, while PUGAL uses 2.8 to 11.1 times fewer classifier evaluations (31.5 to 128.3 times more than greedy); it beats greedy on the two datasets built with feature interactions and loses on none. Each pre-registered hypothesis failed at least one of its thresholds: the certificate gate is outperformed by a learned gate, the gain over greedy on one synthetic family is small, and a dataset-level non-myopia index does not rank the real datasets. An exploratory analysis shows why always-on lookahead can lose to greedy: it acts on differences that are not real, from ties in the truncated objective and from Monte-Carlo noise; a confidence rule removes these losses, and the gate then costs no accuracy on the datasets we test.

## Contents
- `experiments/`: the experiment code (a [uv](https://docs.astral.sh/uv/) project), configs (`experiments/configs/*.toml`),
  unit tests, the scripts that generate every table and figure, and every results JSON used in the paper
  (`experiments/results/`). See `experiments/README.md`.
- `PREREGISTRATION_AND_LOG.md`: the experiment plan with hypotheses and numeric falsification statements (written
  before the runs), the change log, run notes, the exploratory follow-up plans with their decision rules, and all
  verdicts.

## Reproduce
```bash
cd experiments
uv sync
uv run python scripts/fetch_data.py                   # downloads the datasets
uv run python run.py configs/cube_nm_pugal.toml --seed 0
uv run python scripts/run_all.py --workers 8          # all configs x seeds (skips existing results)
uv run pytest
```
The `meta.host` field of the results JSON files was replaced by "redacted"; no other field was changed.
