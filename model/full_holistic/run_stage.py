from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


if __package__ in {None, ""}:
    PROJECT_DIR = Path(__file__).resolve().parents[2]
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

from model.full_holistic.config import apply_cli_overrides, config_for_mode
from model.full_holistic.paths import resolve_results_dir
from model.full_holistic.utils.io import DependencyError, prepare_results_dir


STAGE_MODULES = {
    "data-audit": "model.full_holistic.data.audit",
    "baseline-search": "model.full_holistic.models.baseline",
    "balance-gate": "model.full_holistic.models.balancing",
    "advanced-features-gate": "model.full_holistic.models.advanced",
    "anomaly-recency-gate": "model.full_holistic.models.anomaly_recency",
    "imbalance-ensemble-gate": "model.full_holistic.models.imbalance_ensemble",
    "hyperparameter-tuning-gate": "model.full_holistic.models.hyperparameter_tuning",
    "cascade-filter": "model.full_holistic.evaluation.cascade",
    "riff-rules": "model.full_holistic.evaluation.riff",
    "operational-thresholds": "model.full_holistic.evaluation.thresholds",
    "topk": "model.full_holistic.evaluation.topk",
    "shap": "model.full_holistic.evaluation.shap_analysis",
    "fairness": "model.full_holistic.evaluation.fairness",
    "feature-ablation": "model.full_holistic.evaluation.ablation",
    "anomaly-comparison": "model.full_holistic.evaluation.anomaly_comparison",
    "calibration": "model.full_holistic.evaluation.calibration",
    "stability": "model.full_holistic.evaluation.stability",
    "final-report": "model.full_holistic.reporting.final_report",
    "catboost-refit": "model.full_holistic.models.refit",
    "umap-error-analysis": "model.full_holistic.data_viz.umap_error_analysis",
}


CORE_PIPELINE = [
    "data-audit",
    "baseline-search",
    "balance-gate",
    "advanced-features-gate",
    "anomaly-recency-gate",
    "imbalance-ensemble-gate",
    "hyperparameter-tuning-gate",
    "cascade-filter",
    "riff-rules",
    "operational-thresholds",
    "topk",
    "fairness",
    "final-report",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one modular full-holistic fraud modeling stage.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--stage", required=True, choices=sorted([*STAGE_MODULES.keys(), "full-pipeline"]))
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to model/full_holistic/results_full_train.",
    )
    parser.add_argument("--train-rows", default=None, help="Training row cap. Use 'all' for full train.")
    parser.add_argument("--tuning-train-rows", default=None, help="Inner tuning train row cap. Use 'all' for all rows.")
    parser.add_argument("--tuning-valid-rows", default=None, help="Inner tuning validation row cap. Use 'all' for all rows.")
    parser.add_argument(
        "--tuning-trials",
        type=int,
        default=None,
        help="Optuna trials per tuned family for hyperparameter-tuning-gate.",
    )
    parser.add_argument(
        "--include-expensive-ensembles",
        action="store_true",
        help="Include optional voting/stacking baseline additions where supported.",
    )
    parser.add_argument(
        "--include-anomaly-refit",
        action="store_true",
        help="Only with --stage catboost-refit: also refit CatBoost with appended anomaly scores.",
    )
    parser.add_argument(
        "--top-n-models-to-save",
        type=int,
        default=3,
        help="Save fitted joblib artifacts only for the top N candidates per modeling stage.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite only the selected stage folder. Never resets the whole results directory.",
    )
    return parser.parse_args()


def run_one(stage: str, config, results_dir: Path, args: argparse.Namespace) -> None:
    module = importlib.import_module(STAGE_MODULES[stage])
    module.run(
        config,
        results_dir,
        force=args.force,
        include_anomaly_refit=args.include_anomaly_refit,
        top_n_models_to_save=args.top_n_models_to_save,
    )


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(config_for_mode(args.mode), args)
    results_dir = resolve_results_dir(args.results_dir)
    prepare_results_dir(results_dir)

    try:
        if args.stage == "full-pipeline":
            for stage in CORE_PIPELINE:
                print(f"[run-stage] Starting {stage} ({args.mode})...", flush=True)
                run_one(stage, config, results_dir, args)
        else:
            run_one(args.stage, config, results_dir, args)
    except DependencyError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
