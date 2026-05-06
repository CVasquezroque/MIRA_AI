from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from model.full_holistic.data.context import load_context_optional
from model.full_holistic.paths import STAGE_DIRS
from model.full_holistic.registry import load_candidate_registry, merge_candidate_registry
from model.full_holistic.reporting.decision_table import run as run_decision_table
from model.full_holistic.reporting.figures import generate_report_figures, markdown_figure_section
from model.full_holistic.utils.io import optional_read_csv, prepare_stage_dir
from model.full_holistic.utils.reporting import markdown_table, not_run


def _first_existing(results_dir: Path, *relative_paths: str) -> Path | None:
    for relative in relative_paths:
        path = results_dir / relative
        if path.exists():
            return path
    return None


def _table_section(path: Path | None, columns: list[str] | None = None, max_rows: int = 12) -> str:
    if path is None:
        return not_run()
    frame = pd.read_csv(path)
    if columns is not None:
        existing = [column for column in columns if column in frame.columns]
        frame = frame[existing] if existing else frame
    return markdown_table(frame.head(max_rows).round(6), max_rows=max_rows)


def _text_context(context) -> str:
    if context is None:
        return "Data context was not available when this report was generated."
    return (
        f"- Train months: `{context.train_months}`\n"
        f"- Validation month: `{context.valid_month}`\n"
        f"- Test month: `{context.test_month}`\n"
        f"- Fraud prevalence train / validation / test: `{context.train_prevalence:.6f}` / "
        f"`{context.valid_prevalence:.6f}` / `{context.test_prevalence:.6f}`"
    )


def run(config, results_dir: Path, *, force: bool = False, **_) -> None:
    del config
    output_dir = prepare_stage_dir(results_dir, "final-report", force=force)
    merge_candidate_registry(results_dir)
    candidates = load_candidate_registry(results_dir, required=True)
    ranked = pd.DataFrame([{key: value for key, value in row.items() if key != "spec"} for row in candidates])
    ranked = ranked.sort_values("validation_pr_auc", ascending=False)
    decision_table = run_decision_table(None, results_dir, force=False)
    recommended = decision_table[decision_table["recommended_final_model"]].iloc[0]
    best_pr_auc = decision_table[decision_table["best_by_pr_auc"]].iloc[0]
    figures = generate_report_figures(results_dir, decision_table)
    context = load_context_optional(results_dir)
    candidate_columns = [
        column
        for column in ["model", "stage", "model_family", "validation_pr_auc", "test_pr_auc", "test_roc_auc"]
        if column in ranked.columns
    ]

    threshold_path = _first_existing(results_dir, "threshold_policy_test_metrics.csv")
    low_fpr_path = _first_existing(results_dir, "low_fpr_sweep_test_metrics.csv", f"{STAGE_DIRS['operational-thresholds']}/low_fpr_sweep_test_metrics.csv")
    topk_path = _first_existing(results_dir, "topk_alert_metrics.csv")
    cascade_path = _first_existing(results_dir, f"{STAGE_DIRS['cascade-filter']}/cascade_test_metrics.csv")
    riff_path = _first_existing(results_dir, f"{STAGE_DIRS['riff-rules']}/riff_test_metrics.csv")
    imbalance_path = _first_existing(results_dir, f"{STAGE_DIRS['imbalance-ensemble-gate']}/imbalance_ensemble_test_metrics.csv")
    tuning_path = _first_existing(results_dir, f"{STAGE_DIRS['hyperparameter-tuning-gate']}/tuned_vs_fixed_catboost_fpr5.csv")
    shap_path = _first_existing(
        results_dir,
        "top_features_by_mean_abs_shap.csv",
        f"{STAGE_DIRS['shap']}/top_features_by_mean_abs_shap.csv",
        f"{STAGE_DIRS['shap']}/05_shap_top_features_summary.csv",
    )
    fairness_path = _first_existing(results_dir, "fairness_disparity_summary.csv", f"{STAGE_DIRS['fairness']}/fairness_disparity_summary.csv")
    ablation_path = _first_existing(results_dir, "feature_ablation_metrics.csv", f"{STAGE_DIRS['feature-ablation']}/feature_ablation_metrics.csv")
    anomaly_path = _first_existing(results_dir, "anomaly_score_comparison.csv", f"{STAGE_DIRS['anomaly-comparison']}/anomaly_score_comparison.csv")
    recency_path = _first_existing(results_dir, f"{STAGE_DIRS['anomaly-recency-gate']}/recency_strategy_comparison.csv")
    calibration_path = _first_existing(results_dir, "07_calibration_comparison.csv", f"{STAGE_DIRS['calibration']}/07_calibration_comparison.csv")
    stability_path = _first_existing(results_dir, "stability_seed_metrics.csv", f"{STAGE_DIRS['stability']}/stability_seed_metrics.csv")
    bootstrap_path = _first_existing(results_dir, "bootstrap_confidence_intervals.csv", f"{STAGE_DIRS['stability']}/bootstrap_confidence_intervals.csv")

    report = f"""# Holistic Fraud Modeling Report

## 1. Executive Summary

- Strongest ranking candidate: `{best_pr_auc['model']}` with validation PR-AUC `{best_pr_auc['validation_pr_auc']:.6f}` and test PR-AUC `{best_pr_auc['test_pr_auc']:.6f}`.
- Recommended final model: `{recommended['model']}`.
- Final recommendation status: `{recommended['deployment_readiness']}`.
- Optional sections that were skipped are marked as: This analysis was not run.

## 2. Dataset And Temporal Split

{_text_context(context)}

## 3. Candidate Registry

{markdown_table(ranked[candidate_columns].head(12).round(6))}

## 4. Final Candidate Decision Table

{markdown_table(decision_table.head(12).round(6))}

## 5. Decision Figures

{markdown_figure_section(figures)}

## 6. Threshold Policy Comparison

{_table_section(threshold_path, ["model", "threshold_policy", "feasible", "precision", "fdr", "recall_tpr", "fpr", "alert_rate"], max_rows=20)}

## 7. Top-K Alert Prioritization

{_table_section(topk_path, max_rows=20)}

## 8. Low-FPR Sweep

{_table_section(low_fpr_path, ["model", "fpr_cap_label", "threshold", "alerts", "tp", "fp", "fn", "tn", "precision", "fdr", "recall_tpr", "fpr", "alert_rate", "lift", "fp_per_tp"], max_rows=30)}

## 9. Cascade CatBoost/Stage-1 To TP/FP Filter

{_table_section(cascade_path, max_rows=20)}

## 10. RIFF-Style Low-FPR Rules

{_table_section(riff_path, max_rows=20)}

## 11. Undersample Imbalance Ensembles

{_table_section(imbalance_path, max_rows=20)}

## 12. Hyperparameter Tuning Gate

{_table_section(tuning_path, ["model", "stage", "model_family", "validation_pr_auc", "test_pr_auc", "test_threshold_precision", "test_threshold_fdr", "test_threshold_recall_tpr", "test_threshold_fpr", "test_precision_top1pct", "test_recall_top1pct", "test_fprcap_0.05_max_fpr_gap"], max_rows=20)}

## 13. SHAP Interpretability

{_table_section(shap_path, max_rows=20)}

## 14. Fairness By Protected Group

{_table_section(fairness_path, max_rows=20)}

## 15. Feature Ablation

{_table_section(ablation_path, max_rows=20)}

## 16. Anomaly Score Comparison

{_table_section(anomaly_path, max_rows=20)}

## 17. Recency And Temporal Robustness

{_table_section(recency_path, ["model", "model_family", "train_strategy", "anomaly_policy", "validation_pr_auc", "test_pr_auc"], max_rows=20)}

## 18. Calibration

{_table_section(calibration_path, max_rows=20)}

## 19. Stability And Uncertainty

{_table_section(stability_path, max_rows=20)}

{_table_section(bootstrap_path, max_rows=20)}

## 20. Notes

- Each stage reads persisted inputs and writes its own folder under `results_full_train`.
- The final report is intentionally tolerant of skipped optional experiments.
- Thresholds are selected on validation and then reported on test when the threshold stage has been run.
"""
    report_path = output_dir / "holistic_report.md"
    report_path.write_text(report, encoding="utf-8")
    shutil.copy2(report_path, results_dir / "holistic_report.md")
    print(f"[final-report] Saved final report in: {report_path}")
