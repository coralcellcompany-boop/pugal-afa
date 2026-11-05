#!/usr/bin/env bash
# Regenerate every results table of the paper from results/*.json (no number is typed by hand).
# Run from the repository root:  papers/pugal-afa/experiments/scripts/make_tables.sh [OUT_DIR]
# OUT_DIR defaults to papers/pugal-afa/paper/tables. Point it elsewhere to preview partial results.
set -euo pipefail
EXP=papers/pugal-afa/experiments
R=$EXP/results
OUT=${1:-papers/pugal-afa/paper/tables}
mkdir -p "$OUT"
RT="uv run results-table"

DS_MAIN="cube,cube_nm,synmix_0.5,diabetes,miniboone,physionet,bank,actg175,ckd"
DS_IMB="miniboone,physionet,bank,actg175,ckd"
DS_ABL="cube_nm,synmix_0.5,physionet,miniboone"
METHODS="random,static,greedy,greedy_log,aaco,la2,la3,pugal,pugal_learn,pugal_margin,pugal_rand"
PUGALS="pugal,pugal_learn,pugal_margin,pugal_rand"
ABL="pugal,pugal_nosyn,pugal_noredund,pugal_noscreen,pugal_nofallback,pugal_m4,pugal_m16,pugal_h3,pugal_loss01,pugal_losslog"
LABELS=(
  --label-map cube=CUBE --label-map cube_nm=CUBE-NM --label-map "synmix_0.5=SynMix(0.5)" --label-map diabetes=Diabetes
  --label-map miniboone=MiniBooNE --label-map physionet=PhysioNet --label-map bank=Bank --label-map actg175=ACTG175
  --label-map ckd=CKD
  --label-map "synmix_0=SynMix(0)" --label-map "synmix_0.1=SynMix(0.1)" --label-map "synmix_0.25=SynMix(0.25)"
  --label-map "synmix_0.75=SynMix(0.75)" --label-map "synmix_1=SynMix(1)"
  --label-map random=Random --label-map static=Static --label-map "greedy=Greedy (Brier)" --label-map "greedy_log=Greedy (log loss)"
  --label-map aaco=AACO --label-map la2=Always-LA2 --label-map la3=Always-LA3 --label-map "pugal=PUGAL (synergy gate)"
  --label-map "pugal_learn=PUGAL (learned gate)" --label-map "pugal_margin=PUGAL (margin gate)" --label-map "pugal_rand=PUGAL (random gate)"
  --label-map "pugal_nosyn=without synergy term" --label-map "pugal_noredund=without redundancy term"
  --label-map "pugal_noscreen=without offline screen" --label-map "pugal_nofallback=without CI fallback"
  --label-map "pugal_m4=\$m = 4\$" --label-map "pugal_m16=\$m = 16\$" --label-map "pugal_h3=escalate to \$h = 3\$"
  --label-map "pugal_loss01=0-1 risk" --label-map "pugal_losslog=log-loss risk"
  --label-map acc_auc=acc-AUC --label-map f1_auc=macro-F1-AUC --label-map evals_per_inst="Classifier evaluations per instance"
  --label-map open_rate="Opening rate" --label-map sec_per_inst="Seconds per instance (indicative)"
  --label-map auroc_syn=Synergy --label-map auroc_learn=Learned --label-map auroc_margin=Margin --label-map auroc_rand=Random
  --label-map nmi=NMI --label-map nmi_proxy="NMI proxy" --label-map gamma_hat="\$\\hat\\gamma\$ (min)"
  --label-map gamma_mean="\$\\bar\\gamma\$ (mean)" --label-map frac_positive="Share of states with \$\\Delta_2 > \\varepsilon\$"
)
grid() {  # grid <metric> <rows> <cols> <out> <caption> [extra args]
  local metric=$1 rows=$2 cols=$3 out=$4 cap=$5; shift 5
  $RT "$R"/*_seed*.json --metric "$metric" --group method --columns dataset --rows "$rows" --cols "$cols" \
      "${LABELS[@]}" --caption "$cap" --label "tab:${out%.tex}" --out "$OUT/$out" "$@"
}
grid acc_auc "$METHODS" "$DS_MAIN" main_acc.tex "Accuracy averaged over budgets 1 to the largest budget."
grid f1_auc "$METHODS" "$DS_IMB" main_f1.tex "Macro-F1 averaged over budgets, imbalanced datasets."
grid evals_per_inst "$METHODS" "$DS_MAIN" main_evals.tex "Compute." --lower-is-better evals_per_inst --compact --precision 1
grid sec_per_inst "$METHODS" "$DS_MAIN" main_secs.tex "Wall-clock time (indicative; shared machine)." --lower-is-better sec_per_inst --precision 1
grid open_rate "$PUGALS" "$DS_MAIN" open_rate.tex "Share of steps where the gate opened."
grid acc_auc "$ABL" "$DS_ABL" ablation_acc.tex "Ablations, accuracy."
grid evals_per_inst "$ABL" "$DS_ABL" ablation_evals.tex "Ablations, compute." --lower-is-better evals_per_inst --compact --precision 1

$RT "$R"/*_gatequality_seed*.json --metric auroc_syn --metric auroc_learn --metric auroc_margin --metric auroc_rand \
    --group dataset --bold row --first-header Dataset "${LABELS[@]}" --label tab:gate_quality --out "$OUT/gate_quality.tex" \
    --caption "RQ1: AUROC of each gate's score for the label \$\\Delta_2(s) > \\varepsilon\$ on greedy-visited states."
$RT "$R"/*_gatequality_seed*.json --metric nmi --metric nmi_proxy --metric gamma_hat --metric gamma_mean --metric frac_positive \
    --group dataset --bold none --first-header Dataset "${LABELS[@]}" --label tab:nmi --out "$OUT/nmi.tex" \
    --caption "RQ3: non-myopia index and adaptive submodularity ratio estimates."
uv run python "$EXP/scripts/make_compute_table.py" --out "$OUT/compute.tex"
uv run python papers/pugal-afa/experiments/scripts/make_h3_table.py "$OUT/h3_correlation.tex"
uv run --project "$EXP" python "$EXP/scripts/make_stats_table.py" "$OUT/paired_tests.tex"

# ---------------------------------------------------------------- exploratory follow-up (post hoc; X1, X2, X4)
XLAB=(
  --label-map "synmix_1=SynMix(1)" --label-map "synmix_0.5=SynMix(0.5)" --label-map cube_nm=CUBE-NM
  --label-map cube_nm=CUBE-NM --label-map diabetes=Diabetes --label-map physionet=PhysioNet
  --label-map "greedy_learned_n16=Greedy, learned, \$n=16\$" --label-map "greedy_learned_n64=Greedy, learned, \$n=64\$"
  --label-map "greedy_oracle_n16=Greedy, exact, \$n=16\$" --label-map "greedy_oracle_n64=Greedy, exact, \$n=64\$"
  --label-map "la2_learned_n16=LA2, learned, \$n=16\$" --label-map "la2_learned_n64=LA2, learned, \$n=64\$"
  --label-map "la2_oracle_n16=LA2, exact, \$n=16\$" --label-map "la2_oracle_n64=LA2, exact, \$n=64\$"
  --label-map "pugal_learned_n16=PUGAL, learned, \$n=16\$" --label-map "pugal_learned_n64=PUGAL, learned, \$n=64\$"
  --label-map "pugal_oracle_n16=PUGAL, exact, \$n=16\$" --label-map "pugal_oracle_n64=PUGAL, exact, \$n=64\$"
  --label-map greedy=Greedy --label-map la2=Always-LA2 --label-map "la2_tie=LA2, ties to greedy"
  --label-map "la2_ci=LA2, CI rule" --label-map pugal=PUGAL
  --label-map "la2_tie_oracle_n16=LA2, ties to greedy" --label-map "la2_ci_oracle_n16=LA2, CI rule"
  --label-map acc_auc=acc-AUC --label-map dev_rate="Share of non-greedy steps"
  --label-map auroc_learn=Learned --label-map auroc_syn="Synergy (\$R+\\hat N\$)" --label-map auroc_nhat="\$\\hat N\$ alone"
  --label-map auroc_R="\$R\$ alone" --label-map auroc_margin=Margin --label-map auroc_relmargin="Relative margin"
)
X1ROWS="greedy_learned_n16,greedy_learned_n64,greedy_oracle_n16,greedy_oracle_n64,la2_learned_n16,la2_learned_n64,la2_oracle_n16,la2_oracle_n64,pugal_learned_n16,pugal_learned_n64,pugal_oracle_n16,pugal_oracle_n64"
$RT "$R"/synmix_*_seed*.json --metric acc_auc --group method --columns dataset --rows "$X1ROWS" --cols "synmix_0.5,synmix_1" \
    "${XLAB[@]}" --bold none --label tab:x1 --out "$OUT/x1_model.tex" \
    --caption "Exploratory (post hoc) X1: acc-AUC with the learned model or the exact SynMix model, and \$n\$ sampled values."
X4ROWS="greedy,la2,la2_tie,la2_ci,pugal"
$RT "$R"/*_seed*.json --metric acc_auc --group method --columns dataset --rows "$X4ROWS" --cols "synmix_0.5,synmix_1,cube_nm" \
    "${XLAB[@]}" --label tab:x4_acc --out "$OUT/x4_acc.tex" \
    --caption "Exploratory (post hoc) X4: acc-AUC of always-LA2 with three decision rules (learned model)."
$RT "$R"/*_seed*.json --metric dev_rate --group method --columns dataset --rows "la2_tie,la2_ci" --cols "synmix_0.5,synmix_1,cube_nm" \
    "${XLAB[@]}" --bold none --label tab:x4_dev --out "$OUT/x4_dev.tex" \
    --caption "Exploratory (post hoc) X4: share of steps where always-LA2 leaves the greedy choice."
$RT "$R"/*_seed*.json --metric evals_per_inst --group method --columns dataset --rows "$X4ROWS" --cols "synmix_0.5,synmix_1,cube_nm" \
    "${XLAB[@]}" --lower-is-better evals_per_inst --compact --precision 1 --label tab:x4_evals --out "$OUT/x4_evals.tex" \
    --caption "Exploratory (post hoc) X4: classifier evaluations per instance."
$RT "$R"/*_oracle_n16_seed*.json --metric acc_auc --group method --columns dataset \
    --rows "greedy_oracle_n16,la2_oracle_n16,la2_tie_oracle_n16,la2_ci_oracle_n16,pugal_oracle_n16" --cols "synmix_0.5,synmix_1" \
    "${XLAB[@]}" --label tab:x4_oracle --out "$OUT/x4_oracle.tex" \
    --caption "Exploratory (post hoc) X4 with the exact SynMix model (\$n = 16\$): acc-AUC."
$RT "$R"/*_gatequality_x2_seed*.json --metric auroc_learn --metric auroc_syn --metric auroc_nhat --metric auroc_R \
    --metric auroc_margin --metric auroc_relmargin --group dataset --bold row --first-header Dataset "${XLAB[@]}" \
    --label tab:x2 --out "$OUT/x2_gates.tex" \
    --caption "Exploratory (post hoc) X2: AUROC of gate scores and their components for \$\\Delta_2(s) > \\varepsilon\$."
uv run python "$EXP/scripts/make_budget_table.py" "$OUT/budget_to_greedy.tex"
uv run python "$EXP/scripts/make_ratio_table.py" "$OUT/compute_ratio.tex"
uv run python "$EXP/scripts/make_final_table.py" "$OUT/final_acc.tex"
