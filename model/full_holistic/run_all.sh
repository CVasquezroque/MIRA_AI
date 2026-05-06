#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

MODE="full"
RESULTS_DIR="model/full_holistic/results_full_train"
TOP_N_MODELS_TO_SAVE="3"

if ! command -v conda >/dev/null 2>&1; then
  echo "[run_all] conda is not available in PATH." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DL-env

RUN_DATA_AUDIT="${RUN_DATA_AUDIT:-1}"
RUN_BASELINE_SEARCH="${RUN_BASELINE_SEARCH:-1}"
RUN_BALANCE_GATE="${RUN_BALANCE_GATE:-1}"
RUN_ADVANCED_FEATURES="${RUN_ADVANCED_FEATURES:-1}"
RUN_ANOMALY_RECENCY="${RUN_ANOMALY_RECENCY:-1}"
RUN_IMBALANCE_ENSEMBLES="${RUN_IMBALANCE_ENSEMBLES:-1}"
RUN_HYPERPARAMETER_TUNING="${RUN_HYPERPARAMETER_TUNING:-0}"
RUN_CASCADE_FILTER="${RUN_CASCADE_FILTER:-1}"
RUN_RIFF_RULES="${RUN_RIFF_RULES:-1}"
RUN_OPERATIONAL_THRESHOLDS="${RUN_OPERATIONAL_THRESHOLDS:-1}"
RUN_TOPK="${RUN_TOPK:-1}"
RUN_SHAP="${RUN_SHAP:-0}"
RUN_FAIRNESS="${RUN_FAIRNESS:-1}"
RUN_FEATURE_ABLATION="${RUN_FEATURE_ABLATION:-0}"
RUN_ANOMALY_COMPARISON="${RUN_ANOMALY_COMPARISON:-0}"
RUN_CALIBRATION="${RUN_CALIBRATION:-0}"
RUN_STABILITY="${RUN_STABILITY:-0}"
RUN_FINAL_REPORT="${RUN_FINAL_REPORT:-1}"
RUN_CATBOOST_REFIT="${RUN_CATBOOST_REFIT:-0}"

BASE_CMD=("python" "model/full_holistic/run_stage.py" "--mode" "$MODE" "--results-dir" "$RESULTS_DIR" "--top-n-models-to-save" "$TOP_N_MODELS_TO_SAVE")
if [[ "${INCLUDE_EXPENSIVE_ENSEMBLES:-0}" == "1" ]]; then
  BASE_CMD+=("--include-expensive-ensembles")
fi
if [[ "${INCLUDE_ANOMALY_REFIT:-0}" == "1" ]]; then
  BASE_CMD+=("--include-anomaly-refit")
fi

run_stage() {
  local stage="$1"
  echo "[run_all] stage=${stage} mode=${MODE}"
  "${BASE_CMD[@]}" --stage "$stage"
}

[[ "$RUN_DATA_AUDIT" == "1" ]] && run_stage "data-audit"
[[ "$RUN_BASELINE_SEARCH" == "1" ]] && run_stage "baseline-search"
[[ "$RUN_BALANCE_GATE" == "1" ]] && run_stage "balance-gate"
[[ "$RUN_ADVANCED_FEATURES" == "1" ]] && run_stage "advanced-features-gate"
[[ "$RUN_ANOMALY_RECENCY" == "1" ]] && run_stage "anomaly-recency-gate"
[[ "$RUN_IMBALANCE_ENSEMBLES" == "1" ]] && run_stage "imbalance-ensemble-gate"
[[ "$RUN_HYPERPARAMETER_TUNING" == "1" ]] && run_stage "hyperparameter-tuning-gate"
[[ "$RUN_CASCADE_FILTER" == "1" ]] && run_stage "cascade-filter"
[[ "$RUN_RIFF_RULES" == "1" ]] && run_stage "riff-rules"
[[ "$RUN_OPERATIONAL_THRESHOLDS" == "1" ]] && run_stage "operational-thresholds"
[[ "$RUN_TOPK" == "1" ]] && run_stage "topk"
[[ "$RUN_SHAP" == "1" ]] && run_stage "shap"
[[ "$RUN_FAIRNESS" == "1" ]] && run_stage "fairness"
[[ "$RUN_FEATURE_ABLATION" == "1" ]] && run_stage "feature-ablation"
[[ "$RUN_ANOMALY_COMPARISON" == "1" ]] && run_stage "anomaly-comparison"
[[ "$RUN_CALIBRATION" == "1" ]] && run_stage "calibration"
[[ "$RUN_STABILITY" == "1" ]] && run_stage "stability"
[[ "$RUN_FINAL_REPORT" == "1" ]] && run_stage "final-report"
[[ "$RUN_CATBOOST_REFIT" == "1" ]] && run_stage "catboost-refit"
