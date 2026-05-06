from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "model"
FULL_HOLISTIC_DIR = MODEL_DIR / "full_holistic"
DEFAULT_RESULTS_DIR = FULL_HOLISTIC_DIR / "results_full_train"
DATA_PATH = PROJECT_DIR / "data_banca" / "Base.csv"


STAGE_DIRS: dict[str, str] = {
    "data-audit": "00_data_audit",
    "baseline-search": "01_baseline_search",
    "balance-gate": "02_balance_gate",
    "advanced-features-gate": "03_advanced_features_gate",
    "anomaly-recency-gate": "04_anomaly_recency_gate",
    "imbalance-ensemble-gate": "05_imbalance_ensemble_gate",
    "hyperparameter-tuning-gate": "05b_hyperparameter_tuning_gate",
    "cascade-filter": "06_cascade_filter",
    "riff-rules": "07_riff_rules",
    "operational-thresholds": "08_operational_thresholds",
    "topk": "09_topk",
    "shap": "10_shap",
    "fairness": "11_fairness",
    "feature-ablation": "12_feature_ablation",
    "anomaly-comparison": "13_anomaly_comparison",
    "calibration": "14_calibration",
    "stability": "15_stability",
    "final-report": "16_final_report",
    "catboost-refit": "17_catboost_refit",
}


def resolve_results_dir(value: str | Path | None) -> Path:
    if value is None:
        return DEFAULT_RESULTS_DIR
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def stage_dir(results_dir: Path, stage: str) -> Path:
    try:
        name = STAGE_DIRS[stage]
    except KeyError as exc:
        raise KeyError(f"Unknown stage: {stage}") from exc
    return results_dir / name
