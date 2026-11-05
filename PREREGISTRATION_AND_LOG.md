# Experiment plan (pre-registration)

<!-- Written 2026-11-03 before any experiment code. Changes after the first real run are logged at the bottom with a reason. -->

## Hypotheses under test
- H1 (RQ1: can a cheap signal predict when lookahead changes the decision) → E1 gate-quality study.
- H2 (RQ2: does gating keep always-lookahead accuracy at lower compute) → E2 main comparison, E4 ablations.
- H3 (RQ3: does a dataset-level index predict where non-myopic beats greedy) → E3 cross-dataset study.

## Datasets
All features z-normalised on the training split. Splits: train / validation / test, re-drawn per seed;
test is capped at 1000 instances and validation at 1000; training at 20 000 rows.

| Name | Size | Features / classes | Split | Licence | Source / citation key |
|---|---|---|---|---|---|
| CUBE (σ = 0.3) | 20 000 generated | 20 / 8 | 60/20/20 | generated (MIT code) | schutz2025afabench (generator ported) |
| CUBE-NM (n_c = 5, context group acquired as one action) | 20 000 generated | 55 features, 51 actions / 8 | 60/20/20 | generated (MIT code) | schutz2025afabench |
| SynMix(ρ), ρ ∈ {0, 0.1, 0.25, 0.5, 0.75, 1} | 20 000 generated each | 20 / 2 | 60/20/20 | ours | this paper (Sec. 5.1) |
| Diabetes (NHANES) | 92 062 | 45 / 3 | stratified | as shipped in AFABench (MIT repo) | kachuee2019cost |
| MiniBooNE | 130 064 | 50 / 2 | stratified | UCI, CC BY 4.0 | roe2005boosted |
| PhysioNet 2012 (in-hospital mortality) | 12 000 | 41 / 2 | stratified | PhysioNet open access | silva2012predicting |
| BankMarketing | 45 211 | 16 / 2 | stratified | UCI, CC BY 4.0 | moro2014bank |
| ACTG175 | 2 139 | 23 / 2 | stratified | UCI, CC BY 4.0 | hammer1996aids |
| CKD | 400 | 24 / 2 | stratified | UCI, CC BY 4.0 | rubini2015chronic |
| Fashion-MNIST (stress test, appendix only) | 70 000 | 784 / 10 | standard | MIT | TODO cite via lit-bib before use |

**SynMix(ρ).** Binary label. Feature z ∈ {0, 1} (context, P(z=1) = ρ); pair (x_1, x_2); single x_3; 16 Gaussian
distractors. If z = 1, y = 1{x_1 > 0} XOR 1{x_2 > 0}; if z = 0, y = 1{x_3 > 0}; 5% label noise. Greedy is
optimal when ρ = 0 and fails on the z = 1 instances; ρ is the fraction of instances that need lookahead.
Budget B = 6. This is the controlled stress test for H1 and the compute-saving curve.

Real datasets follow AFABench preprocessing (mean imputation, z-normalisation). CSVs for Diabetes, MiniBooNE and
PhysioNet come from the AFABench repository (MIT); BankMarketing, ACTG175 and CKD from UCI. A fetch script
downloads them into `experiments/data/` (gitignored) and records SHA-256 hashes.

## Shared components (identical for every method on a dataset and seed)
- **Classifier**: MLP on [x ⊙ m, m], 2 × 128 ReLU, dropout 0.1, trained with random masks (number observed
  uniform in 0..d), Adam 1e-3, batch 256, ≤ 60 epochs, early stopping on validation loss (patience 5).
- **Conditional sampler**: k-nearest-neighbour empirical conditional on the observed features (Euclidean on
  z-scored values) over a pool of ≤ 5000 training rows; n = 16 neighbours; joint values (x_j, x_k) are drawn from
  the same neighbour row so dependence is kept.
- **Risk**: Brier/Gini Bayes risk r = 1 − Σ_y p(y)² (main; strictly concave, bounded in [0, 1]);
  ablations: 0-1 risk and log loss (entropy, the CMI objective).
- **Budget**: hard budget, unit costs, anytime evaluation: one trajectory to B_max (10 for CUBE, CUBE-NM and real
  datasets; 6 for SynMix) gives the whole accuracy–budget curve. Rollout depth does not depend on remaining budget
  beyond the cap h ≤ B_max − t.

## Baselines
| Name | Citation key | Why it is a fair comparison |
|---|---|---|
| Random | — | floor |
| Static greedy order (features ranked by mean one-step utility on validation) | schutz2025afabench (static PT-S family) | non-adaptive reference |
| Greedy (Brier) — PUGAL's base | gadgil2023estimating (same kind of loss-reduction utility) | same model, sampler and budget; the "never look ahead" arm |
| Greedy (log loss / CMI) | covert2023learning, gadgil2023estimating | the objective of GDFS and DIME, computed with the same model |
| Always-LA2 (2-step truncated rollout, all candidates) | kobayashi2025learning (fixed lookahead) | the "always look ahead" arm; same model and sampler |
| Always-LA3 (3-step, top-m + flagged candidates, n = 8) | — | deeper always-on arm |
| AACO (re-implemented: 5-NN, random subset search, 1000 subsets per step) | valancius2023acquisition | strongest inference-time non-myopic method; same classifier and kNN pool |
| GDFS-DM, DIME-DM (AFABench code, same splits) | covert2023learning, gadgil2023estimating | learned greedy policies; run only if the AFABench code reproduces on our splits within 2 working days; otherwise their AFABench-reported numbers appear in an appendix table, not in the main comparison |

Fairness: all model-based methods share one trained classifier and one kNN pool per dataset × seed; no method gets
extra hyperparameter trials (each tuned with the same 6-point grid on validation, seed 0). NM-PPG and BRiG-AFA are
discussed but not run (learned policies with a different training protocol; NM-PPG code availability unknown,
BRiG-AFA code is public and is a stretch baseline on CUBE-NM only).

## Metrics
| Metric | Higher or lower is better | Why this metric |
|---|---|---|
| Accuracy at budget t, t = 1..B_max | higher | AFABench's metric for balanced datasets |
| Macro-F1 at budget t (PhysioNet, BankMarketing, MiniBooNE, CKD, ACTG175) | higher | AFABench's metric for imbalanced datasets |
| Normalised area under the accuracy–budget curve (mean over t = 1..B_max) | higher | one number per run for tables and tests |
| Classifier evaluations per test instance (exact count) | lower | hardware-independent compute |
| Wall-clock seconds per test instance (M3 Pro CPU) | lower | practical compute |
| Opening rate (fraction of steps with G = 1) | lower | where the compute goes |
| Gate AUROC for 1{Δ_2(s) > ε} on states visited by greedy (labels from exact LA2), ε = 0.01 | higher | RQ1 |
| Non-myopia index NMI and its Spearman correlation with acc-AUC(LA2) − acc-AUC(greedy) | higher | RQ3 |
Derived: "compute at equal accuracy" = evaluations per instance needed to reach LA2's acc-AUC within 0.5 pt.

## Experiments
| Id | Tests | Config file | Seeds | Expected compute |
|---|---|---|---|---|
| E0 | pipeline sanity (smallest dataset, tiny budget) | smoke.toml | 1 | < 1 min |
| E1 | H1: gate AUROC (G-syn, G-learn, G-margin, G-rand) vs exact LA2 labels | `<dataset>_gatequality.toml` on all datasets | 5 | ~0.5 CPU-h total |
| E2 | H2: accuracy–budget curves, compute, opening rate for all methods | `<dataset>_<method>.toml` | 5 | ~20 CPU-h total (LA2 and AACO dominate) |
| E3 | H3: NMI vs gain; compare against empirical submodularity ratio and greedy-gap estimates | computed from E1/E2 outputs | 5 | negligible |
| E4 | ablations (below) on CUBE-NM, SynMix(0.5), PhysioNet, MiniBooNE | `<dataset>_pugal_<ablation>.toml` | 5 | ~5 CPU-h |
| E5 | SynMix ρ-sweep: compute saving vs ρ | `synmix<ρ>_<method>.toml` | 5 | ~3 CPU-h |
| E6 | stress test Fashion-MNIST (appendix) | `fmnist_<method>.toml` | 3 (deterministic given the classifier; see note) | ~4 CPU-h |

Seeds: 5 everywhere (the classifier is learned). E6 uses 3 seeds only if the pilot std of acc-AUC < 0.5 pt;
otherwise E6 is dropped rather than seeds cut.

## Ablations
| Ablation | What is removed | Expected |
|---|---|---|
| `nosyn` | gate uses R(s) only | misses XOR states: accuracy drops on CUBE-NM / SynMix, unchanged elsewhere |
| `noredund` | gate uses N̂(s) only | small accuracy drop where redundancy matters |
| `noscreen` | no offline screen: synergy checked on all unobserved pairs | same accuracy, much higher gate cost |
| `nofallback` | no confidence-bound fallback | more non-greedy moves, higher variance, possible small drops on real data |
| `fixedh2` | depth fixed at 2 (no escalation) | small drop on CUBE-NM where the useful plan needs 3 steps |
| `m4`, `m16` | candidate set size m | m = 4 cheaper, small drop; m = 16 costlier, no gain |
| `loss01`, `losslog` | risk function | 0-1 has flat utilities (ties); log loss = CMI objective |
| `randgate` | gate replaced by Bernoulli(p) at matched opening rate | Theorem 1 says no harm; captures much less improvement |

## Compute budget and stopping rule
Local Apple M3 Pro (12 cores, 36 GB), CPU PyTorch, runs parallelised over processes. Estimated ~33 CPU-hours
(~4 wall-clock hours on 10 parallel workers) plus 20% spare. No GPU and no Runpod spend ($0). Stopping rule: if
E1 on CUBE-NM and SynMix(0.5) gives G-syn AUROC < 0.70, stop E2 and switch the main gate to G-learn (log the
change here). Hyperparameters (ε ∈ {0.005, 0.01, 0.02}, screen threshold at the {90, 95, 99}th percentile of
pairwise synergy, m ∈ {4, 8}) are chosen once per dataset on the validation split with seed 0 by the rule:
lowest opening rate whose validation acc-AUC is within 0.5 pt of always-LA2; then all seeds run with the chosen
values. The test split is never used for selection.

A "win" is a paired comparison over the 5 seeds (same seed = same split, classifier and kNN pool): the mean
paired difference exceeds its standard deviation and the Wilcoxon signed-rank test over all dataset × seed pairs
is significant at 0.05. "Within" means |mean difference| ≤ max(1 pt, pooled std).

## What result would falsify each hypothesis
- H1 is false if G-syn's AUROC for 1{Δ_2 > 0.01} is below 0.80 on CUBE-NM or on SynMix(0.5), or if it does not
  exceed G-margin's AUROC by at least 0.05 on at least 3 of the 6 real datasets.
- H2 is false if PUGAL's acc-AUC is below always-LA2's by more than max(1 pt, pooled std) on 2 or more of the 9
  main datasets, or if its opening rate exceeds 30% on 4 or more of the 6 real datasets, or if it does not beat
  greedy's acc-AUC by more than the pooled std on CUBE-NM and on SynMix(ρ ≥ 0.5).
- H3 is false if the Spearman correlation between NMI and acc-AUC(LA2) − acc-AUC(greedy) over the 9 main
  datasets is below 0.6, or if it is not higher than the correlation of the empirical adaptive submodularity
  ratio estimate.

## Change log
- (none yet)
- 2026-11-03 (before any run): BankMarketing drops `duration` and ACTG175 drops `time` (both leak the label; AFABench keeps them). Bank now has 15 actions (47 one-hot features grouped per variable), ACTG175 has 22 features. CUBE-NM uses informative σ = 0.3 (AFABench experiment config). Splits are stratified and preprocessing uses train-split statistics only (AFABench uses the full dataset). CKD stray "\tno" level merged into "no".
- 2026-11-03 (before any full run, after a 150-instance timing pilot on synmix_0.5 with untuned settings): main PUGAL uses h_max = 2 (depth-3 escalation cost 1.54M evaluations per instance, about 4× always-LA2, because exact-expectation rollouts grow geometrically with depth); escalation to h = 3 becomes ablation `pugal_h3`, and `pugal_fixedh2` is dropped (it equals main). Candidate set capped at 2m. Fallback rule: paired 95% normal CI on shared neighbour samples (the Hoeffding version never leaves greedy at n = 16). E1 cost re-estimated at ~17 CPU-hours (exact LA2 on every greedy-visited state), ~2 wall-clock hours on 10 workers. Gate oracle uses Δ_2 (two-step) labels.
- 2026-11-03 (before the full run; reason: measured cost ~241 CPU-hours vs ~33 planned): (1) ablations reuse the dataset's tuned PUGAL settings and change exactly one component (no separate tuning); (2) the test set is capped at the first 500 instances of each seed's test split for ALL methods on cube_nm, diabetes, miniboone and physionet (paired comparisons kept); (3) the margin gate gets its own threshold grid {0.005, 0.01, 0.02, 0.05, 0.1}; (4) the candidate cap is fixed at 16 for every m. Estimated ~120 CPU-hours, ~12 wall-clock hours on 10 workers.
- 2026-11-04 (during the full run, no change to code, configs or selection): from about 06:00 the machine was shared with an unrelated Kaggle workload (17 processes at normal priority, the GPU also busy); our processes ran at nice 5 until 13:10, then nice 0 (user ran `sudo renice`). The Mac slept from 09:40 to 11:29 (lid closed; processes paused, uptime unbroken). The run was also frozen for about 1 h at 07:15 (SIGSTOP/SIGCONT) while moving to cloud compute was considered; it stayed local. Effect: results unchanged (deterministic given seed; perf_counter = mach_absolute_time does not count sleep), but wall-clock seconds per instance depend on the load at run time, so the paper uses classifier-evaluation counts as the compute metric and calls seconds indicative.

## Observations during the run (analysis notes; protocol unchanged)
- 2026-11-05: the full run finished at 06:03 with 961/961 jobs ok and none failed (1833 min wall clock). The Mac
  slept once more, for 197 s at 21:16 on 2026-11-04 (battery at 1%, then back on AC); processes paused and resumed,
  results are unaffected. Kaggle was idle from the evening of 2026-11-04.
- 2026-11-05: gate-quality AUROC is undefined (NaN) when a seed has no state with Δ_2 > ε (SynMix(0) all seeds,
  ACTG175 seeds 1 and 3). results-table now skips non-finite values; the caption's seed count shows how many
  seeds define the cell.
- 2026-11-04, first gate-quality results (diabetes, 3 of 5 seeds): the margin gate's AUROC is far below 0.5. Code checked: score = −(a1 − a2) as designed (gates.py MarginGate), so this is not a sign bug. Likely cause (to verify): scale confound. Δ2, ε and the margin are all in absolute utility units; early states have large utilities, large margins and large Δ2, and late states have small ones, so "small margin" selects late states where Δ2 is small. Plan: report the pre-registered margin gate as is; add a clearly labelled post-hoc analysis (AUROC by step index, and a relative margin (a1 − a2)/a1) so the reader can judge whether the margin baseline is weak or merely mis-scaled. Do not replace the pre-registered gate.
- 2026-11-04: gamma_hat (the minimum over pairs, matching the definition of the adaptive submodularity ratio) is 0 on the first gate-quality runs, which makes it a degenerate predictor for H3. Plan: report H3 against both gamma_hat (pre-registered) and gamma_mean (mean over pairs, a stronger baseline already computed by core.py), so the comparison is not won by default.

## Exploratory follow-up (post hoc; planned 2026-11-05 after the full run)

Status: **exploratory, designed after seeing the E1–E5 results.** It does not change the pre-registered verdicts
(H1 false on the SynMix(0.5) AUROC clause, H2 false on the "beats greedy on SynMix(ρ ≥ 0.5)" clause, H3 false).
Its results are reported in a separate, labelled subsection and appendix, never mixed into the H1–H3 tables.
The decision rules below are fixed before any X run.

Observations that motivate it (from paper/tables, commit 2fc1d20):
- always-LA2 is below greedy on SynMix(0.5) and SynMix(1), where the generator has a known XOR structure, while
  PUGAL is not below greedy anywhere;
- the learned gate has higher AUROC than the synergy gate on most datasets and opens less often;
- the margin gate's AUROC is far below 0.5 on several datasets;
- the LA2 − greedy acc-AUC gap is near zero on 8 of 9 main datasets, so the H3 correlation is computed over a
  very narrow range.

| Id | Question | Design | Seeds | Compute |
|---|---|---|---|---|
| X1 | Is LA2's loss on SynMix caused by the learned model (posterior + kNN conditional) or by Monte-Carlo noise? | SynMix(0.5), SynMix(1); methods greedy, la2, pugal (tuned settings reused); 2×2 factorial: model ∈ {learned, oracle}, n ∈ {16, 64}. Oracle = exact Bayes posterior of the generator (5% label noise included, applied to the normalised features via the stored train mean/std) and exact sampling of unobserved features from their true marginals (features are independent in SynMix). | 5 | ~25–35 CPU-h (LA2 at n = 64 dominates) |
| X2 | Why does the learned gate beat the synergy gate, and why is the margin AUROC below 0.5? | Re-run gate quality on CUBE-NM, SynMix(0.5), Diabetes, PhysioNet with a per-state dump (Δ_2, step index, a1, a2, R, N̂, each gate score) and the learned gate's standardised coefficients. Statistics: AUROC by step index; AUROC of R alone, N̂ alone, relative margin (a1 − a2)/a1; coefficient table. | 5 | ~4 CPU-h |
| X3 | Is the weak NMI correlation a range problem? | No new runs: Spearman(NMI, LA2 − greedy) and Spearman(NMI, PUGAL − greedy) over the SynMix ρ-sweep (6 values: 0, 0.1, 0.25, 0.5, 0.75, 1; the plan first said 7 by mistake) and over the 9 main datasets; plus NMI vs the X1 oracle-model gain. | — | 0 |

Decision rules (fixed before running):
- X1, "model mismatch": under the oracle model at n = 64, LA2 − greedy acc-AUC ≥ −(pooled std) on both SynMix
  datasets, while with the learned model at n = 64 it stays below −(pooled std) on at least one.
- X1, "Monte-Carlo noise": with the learned model, moving n from 16 to 64 raises LA2 − greedy by more than the
  pooled std on both SynMix datasets.
- Both, neither, or mixed outcomes are reported as they are.
- X2, "scale confound of the margin gate": the relative-margin AUROC exceeds 0.5 by more than its seed std on at
  least 3 of the 4 datasets.
- X2, "learned gate = recalibrated synergy": the AUROC of N̂ alone is within 0.05 of the learned gate on at least
  3 of the 4 datasets.
- X3 is descriptive only (no threshold); with 7 points it cannot support a claim beyond "consistent / not
  consistent with".

Machine: the local Mac (Apple M3 Pro), 10 workers; no cloud spend. Estimated 30–40 CPU-h, about 4–6 hours
wall-clock with the machine otherwise idle.

Run record (X1–X3): launched 2026-11-05 from the working tree before commit 4647306, so the X1/X2 results carry
meta.git_commit = d6850eb while their code is the one committed in 4647306. 173 of 175 jobs ran first time; the
two diabetes_gatequality_x2 seeds 0 and 1 failed on a missing runs/gatestates directory (fixed: the dump now
creates it) and were re-run.

X2 verdicts (2026-11-06; all 5 seeds; source paper/tables/x2_gates.tex and results/*_gatequality_x2_seed*.json):
- "Learned gate = recalibrated synergy": **not supported.** AUROC of N̂ alone minus the learned gate's is −0.116
  (CUBE-NM), −0.233 (SynMix(0.5)), −0.039 (Diabetes), −0.093 (PhysioNet): within 0.05 on 1 of 4 datasets (rule: 3).
  The learned gate uses information beyond N̂ (it also sees r(s), b, the margin, R and |S|).
- "Scale confound of the margin gate": **supported by the rule, weak on the dataset that motivated it.** Relative
  margin AUROC − 0.5 exceeds its seed std on CUBE-NM (0.143 vs 0.034), SynMix(0.5) (0.097 vs 0.068) and PhysioNet
  (0.137 vs 0.113), not on Diabetes (0.039 vs 0.115): 3 of 4. On Diabetes, where the margin gate is most inverted
  (AUROC 0.17), mean a1 is larger on positive states (0.061) than on negative ones (0.019), as the scale story
  predicts, but normalising by a1 only brings the AUROC to chance level, not above it.
- Correction of the preliminary reading given to the author on 2026-11-05 ("the learned gate is mostly the synergy
  term, better calibrated"): not supported with all seeds.

X1 verdicts (2026-11-06; all 5 seeds; LA2 − greedy acc-AUC with pooled std = sqrt of the mean of the two
variances; source paper/tables/x1_model.tex and results/synmix_*_{la2,greedy}_{learned,oracle}_n*_seed*.json):
- "Model mismatch": **not supported.** Under the exact model at n = 64, LA2 − greedy is −0.042 (pooled std 0.030)
  on SynMix(0.5), below −(pooled std); on SynMix(1) it is 0.000 (LA2 and greedy take the same groups there, see X4).
- "Monte-Carlo noise": **supported, but it does not close the gap.** With the learned model, n 16 → 64 raises
  LA2 − greedy from −0.035 to −0.012 on SynMix(0.5) and from −0.062 to −0.041 on SynMix(1), i.e. by 0.022 and 0.021,
  more than the pooled std of either comparison (≤ 0.017); LA2 stays below greedy at n = 64 on both.
- Correction of the preliminary reading given to the author on 2026-11-05 (2–3 seeds, "neither model nor n
  matters"): with all seeds, n matters for the learned model. Consistent with X4: LA2 acts on differences inside
  the sampling noise; more samples shrink that noise, a confidence rule removes such moves.

### X4: LA2 tie-breaking (post hoc; planned 2026-11-05 after X1 finished)

Status: **exploratory, designed after seeing the X1 results.** Same reporting rules as X1–X3.

Observation that motivates it (results/synmix_1_{greedy,la2,pugal}_oracle_n16_seed0.json, acc_t1..acc_t3;
and a probe of Q^(2) at the state "x1 observed" under the oracle model): with the exact model, Q^(2)(x2) equals
Q^(2) of a useless group (both reach the XOR at depth 2), and LookaheadPolicy breaks the tie by the smallest group
index, so LA2 acquires the useless group z first and reaches the XOR answer one step later than PUGAL. The
truncated objective is indifferent to *when* within the horizon information arrives, but acc-AUC averages over
budgets. PUGAL's confidence fallback keeps g(s) in such ties.

| Id | Question | Design | Seeds | Compute |
|---|---|---|---|---|
| X4 | Does LA2's loss on SynMix come from tie-breaking ("procrastination") rather than from lookahead itself? | always-LA2 with two decision rules: `la2_tie` (keep g(s) unless Q^(2)(g) − min Q^(2) > 1e-9) and `la2_ci` (keep g(s) unless the gap exceeds the paired 95% CI half-width; = PUGAL with an always-open gate, all candidates, h = 2). Learned model, n = 16 on SynMix(0.5), SynMix(1), CUBE-NM; oracle model, n = 16 on SynMix(0.5), SynMix(1). New metric dev_rate = share of steps where the choice differs from g(s). | 5 | ~12 CPU-h (CUBE-NM dominates) |

Decision rules (fixed before running):
- "Procrastination explains the SynMix loss": with the learned model, la2_ci − greedy acc-AUC ≥ −(pooled std) on
  both SynMix datasets (la2 is below −(pooled std) on both), and under the oracle model la2_tie − greedy ≥
  −(pooled std) on both.
- "The fix keeps lookahead's gain": on CUBE-NM, la2_ci − la2 acc-AUC ≥ −(pooled std). Reported either way.
- Outcomes are reported as they are; X4 does not change any pre-registered verdict.

X4 verdicts (2026-11-06; 65 jobs, all 5 seeds; pooled std = sqrt of the mean of the two variances):
- "Procrastination explains the SynMix loss": **supported.** Learned model: la2_ci − greedy = +0.006 on
  SynMix(0.5) (0.821 ± 0.008 vs 0.815 ± 0.010) and −0.001 on SynMix(1) (0.860 ± 0.010 vs 0.861 ± 0.010), both
  ≥ −(pooled std). Oracle model: la2_tie − greedy = 0.000 on SynMix(0.5) (0.836 vs 0.836) and +0.073 on SynMix(1)
  (0.874 vs 0.801). Refinement: with the learned model exact ties are rare, so la2_tie = la2 to three decimals,
  yet LA2 leaves g(s) on 36% / 32% of steps (dev_rate of la2_tie) and la2_ci only on 2% / 5%: the learned-model
  loss comes from acting on near-ties within Monte-Carlo noise, the oracle-model loss from exact ties broken by
  index. Both are "a non-greedy move whose advantage is not real".
- "The fix keeps lookahead's gain": **not supported.** CUBE-NM la2_ci − la2 = −0.051 (0.520 ± 0.041 vs
  0.571 ± 0.025), below −(pooled std) = −0.034. The CI rule is too conservative where lookahead helps.
- Unplanned but direct: la2_ci and PUGAL (same rule, gated) match to within 0.001–0.002 acc-AUC on all three
  datasets, while PUGAL uses 5–11× fewer classifier evaluations; the accuracy gap of PUGAL to LA2 on CUBE-NM is
  the confidence rule's, not the gate's.

## Verdicts (pre-registered H1–H3; full run of 2026-11-05, commit 2fc1d20)
Sources: paper/tables/gate_quality.tex, main_acc.tex, open_rate.tex, h3_correlation.tex; exact seed means from
results/*.json.
- **H1: not supported (falsified on one clause).** G-syn AUROC is 0.89 on CUBE-NM (≥ 0.80) but 0.79 on
  SynMix(0.5) (< 0.80). G-syn exceeds G-margin by ≥ 0.05 on 5 of 6 real datasets (not on ACTG175). Additional:
  the learned gate has a higher AUROC than G-syn on most datasets.
- **H2: not supported (falsified on one clause).** PUGAL is below always-LA2 by more than max(1 pt, pooled std)
  on 1 of 9 datasets (CUBE-NM), fewer than 2: passes. Opening rate > 30% on 3 of 6 real datasets (MiniBooNE,
  PhysioNet, CKD), fewer than 4: passes. PUGAL beats greedy by more than the pooled std on CUBE-NM and SynMix(0.75)
  but not on SynMix(0.5) or SynMix(1): fails. Additional: always-LA2 is below greedy on SynMix(0.5) and SynMix(1).
- **H3: not supported.** Spearman(NMI, LA2 − greedy) over the 9 main datasets is 0.32 (< 0.6). It is higher than
  the submodularity-ratio estimates (γ̂ constant, no correlation; −γ̄ 0.20).
