# PUGAL experiment code: design and interfaces

Theory and notation: `../notes/method.md`. Plan: `../notes/experiments.md`. Everything runs on CPU with PyTorch.
All randomness flows from one `seed` (numpy `Generator`, `torch.Generator`); same seed ⇒ identical results.

## Modules (`src/experiment/`)

### `data.py`
```python
@dataclass
class AFAData:
    name: str
    X_train: np.ndarray; y_train: np.ndarray      # float32 [n, d], int64 [n]
    X_val: np.ndarray;   y_val: np.ndarray
    X_test: np.ndarray;  y_test: np.ndarray
    n_classes: int
    groups: list[list[int]]   # acquisition actions; each is a list of feature indices. Default: [[0], [1], ..., [d-1]]
    imbalanced: bool          # True → report macro-F1 as well

def load_dataset(name: str, seed: int, data_dir: Path) -> AFAData
```
Names: `cube`, `cube_nm`, `synmix_<rho>` (e.g. `synmix_0.5`), `diabetes`, `miniboone`, `physionet`, `bank`,
`actg175`, `ckd`, `toy_xor` (tiny, for tests), `smoke` (alias of a 2 000-row synmix_0.5).
Splits 60/20/20 (stratified for real data), then caps: train ≤ 20 000, val ≤ 1 000, test ≤ 1 000.
z-normalise with train statistics; mean-impute missing values with train means; categorical columns one-hot
(bank) — record the resulting feature count. `cube_nm`: the n_c context features form ONE group.

### `model.py`
```python
class MaskedClassifier(nn.Module):          # MLP on concat(x * m, m); 2 x 128 ReLU, dropout 0.1
def train_classifier(data: AFAData, seed: int, max_epochs=60, patience=5) -> MaskedClassifier
def predict_proba(model, X: Tensor, M: Tensor) -> Tensor    # [batch, n_classes], eval mode, no grad, batched
```
Random-mask training: per row, sample k ~ Uniform{0..d} then a uniformly random subset of size k (by group).
A global counter `model.n_evals` is incremented by the batch size on every `predict_proba` call (compute metric).

### `sampler.py`
```python
class KNNSampler:
    def __init__(self, X_pool: np.ndarray, n: int, seed: int)   # pool ≤ 5000 train rows
    def neighbours(self, x: np.ndarray, m: np.ndarray) -> np.ndarray   # indices of n nearest pool rows on observed features (all rows if nothing observed: n random rows)
```
A conditional sample of unobserved features = the pool row's values at those features (so joint samples keep
dependence). Batched version over many states is welcome.

### `risk.py`
```python
def risk(p: Tensor, kind: str) -> Tensor   # kind in {"brier", "01", "log"}; p [..., C] → [...]
#   brier: 1 - sum p^2 ; 01: 1 - max p ; log: -sum p log p
```

### `utility.py`
State = (x_obs: np.ndarray[d], m: np.ndarray[d] bool). Helpers that, given model, sampler and risk kind, compute:
- `one_step(state, candidates) -> dict[group, U1]`: U_1(j) = r(s) - mean_over_neighbour_rows r(s ⊕ j).
- `two_step_value(state, j, candidates) -> Q2(j)`: U_1(j) + mean over neighbour rows of max_k U_1(k | s ⊕ j),
  where the inner U_1 uses neighbours recomputed for the new state.
- `rollout_value(state, j, h, candidates)`: expected risk after acquiring j then h-1 greedy steps (truncated
  rollout); Q^{(h)} in the notes. h = 2 must equal r(s) - Q2(j) (test this identity).
- `synergy(state, j, k)`: E_{x_j}[ (U_1(k | s ⊕ j) - U_1(k | s))_+ ].
- `redundancy(state, g, j2)`: a_2 - E_{x_g}[ max_k U_1(k | s ⊕ g) ]  (R(s) in Prop. 3).
Everything batched with torch; no Python loops over neighbour rows inside hot paths.

### `policies.py`
```python
class Policy:  # stateless per instance
    def select(self, state) -> group_index   # never returns an already acquired group
    # bookkeeping attributes: self.last_opened: bool, self.last_gate_score: float | None
```
Implement: `RandomPolicy`, `StaticPolicy(order)`, `GreedyPolicy(risk_kind)`, `LookaheadPolicy(h, candidate_rule)`
(always-on), `AACOPolicy(k_nn=5, n_subsets=1000)` (per valancius2023acquisition: at each step evaluate random
candidate subsets of unobserved groups with the kNN rows, score = mean loss of the classifier on the neighbours
with the subset added, plus nothing for cost under hard budget; pick the best subset; acquire its member with the
best single-feature score), and `PUGALPolicy(gate, h0=2, h_max=3, m=8, eps, conf_fallback=True)` implementing
Sec. 3 of notes/method.md (greedy unless gate opens; when open, truncated rollout over C(s) = top-m by U_1 ∪
features in active flagged synergy pairs; escalate h while CIs overlap; fall back to greedy unless the best
candidate beats greedy by more than its CI half-width).

### `gates.py`
`SynergyGate(flagged_pairs, eps)` (score = R(s) + max over active flagged pairs of synergy),
`MarginGate(tau)` (score = -(a1 - a2)), `RandomGate(p, seed)`, `OracleGate(eps)` (score = exact Δ_2 via
always-LA2), `LearnedGate(model)` (logistic regression on state features [r(s), b, a1-a2, R, N̂, |S|]).
Each gate exposes `score(state, ctx) -> float` and `open(state, ctx) -> bool`.
`screen_synergy(model, sampler, data, n_states, quantile) -> set[(j, k)]` builds the offline screen from states
visited by greedy on training rows.

### `evaluate.py`
```python
def run_policy(policy, data, model, sampler, budget) -> dict
```
Runs every test instance to `budget`, returns per-step accuracy (and macro-F1 if imbalanced), normalised
acc-AUC, evaluations per instance, seconds per instance, opening rate, and (when asked) the list of
(gate score, Δ_2 label) pairs for RQ1.

### `core.py`
`run_experiment(config, seed) -> dict[str, float]` reads `dataset`, `method`, `[hparams]`; builds data, trains
the classifier (cached on disk under `runs/` keyed by dataset+seed), builds the sampler and the policy, runs
`run_policy`, and returns flat numeric metrics: `acc_t1..acc_tB`, `f1_t1..` (if imbalanced), `acc_auc`, `f1_auc`,
`evals_per_inst`, `sec_per_inst`, `open_rate`, `gate_auroc` (gate-quality configs only), `nmi` (where computed).

## Tests (`tests/`)
Data shapes and determinism; risk values by hand; U_1 ≥ 0 under the model (Fact 1) on a toy problem;
the identity rollout_value(h=2) = r - Q2; synergy > 0 for the XOR pair and ≈ 0 for independent features on
`toy_xor`; greedy never re-acquires; PUGAL with an always-closed gate equals greedy and with an always-open gate
equals LookaheadPolicy on the same candidates.
