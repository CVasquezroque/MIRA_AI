from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from model.full_holistic.constants import MAIN_FPR_CAP
from model.full_holistic.paths import STAGE_DIRS
from model.full_holistic.registry import load_candidate_registry
from model.full_holistic.utils.io import optional_read_csv, prepare_stage_dir


def _append_missing_models(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    if fallback.empty:
        return primary
    if primary.empty or "model" not in primary.columns:
        return fallback.copy()
    if "model" not in fallback.columns:
        return primary
    missing = fallback[~fallback["model"].isin(primary["model"])]
    if missing.empty:
        return primary
    return pd.concat([primary, missing], ignore_index=True, sort=False)


def _threshold_rows_from_tuning(tuning_comparison: pd.DataFrame) -> pd.DataFrame:
    if tuning_comparison.empty:
        return pd.DataFrame()
    required = {"model", "fpr_cap", "test_threshold_precision", "test_threshold_fdr", "test_threshold_recall_tpr", "test_threshold_fpr"}
    if not required.issubset(tuning_comparison.columns):
        return pd.DataFrame()
    rows = tuning_comparison[tuning_comparison["fpr_cap"].astype(float).round(8).eq(round(MAIN_FPR_CAP, 8))].copy()
    if rows.empty:
        return pd.DataFrame()
    rename = {
        "test_threshold_precision": "precision",
        "test_threshold_fdr": "fdr",
        "test_threshold_recall_tpr": "recall_tpr",
        "test_threshold_fpr": "fpr",
        "test_threshold_alert_rate": "alert_rate",
        "selected_threshold": "threshold",
    }
    common_cols = ["model", "stage", "model_family", *rename.keys()]
    existing = [column for column in common_cols if column in rows.columns]
    result = rows[existing].rename(columns=rename)
    result["split"] = "test"
    result["threshold_policy"] = "valid_global_5pct_fpr"
    result["threshold_selected_on"] = "validation"
    result["feasible"] = result["fpr"].astype(float) <= MAIN_FPR_CAP
    return result


def _topk_rows_from_tuning(tuning_comparison: pd.DataFrame) -> pd.DataFrame:
    if tuning_comparison.empty or "model" not in tuning_comparison.columns:
        return pd.DataFrame()
    rows = []
    for _, row in tuning_comparison.iterrows():
        for pct, suffix in [(0.005, "top0_5pct"), (0.01, "top1pct"), (0.05, "top5pct")]:
            precision_col = f"test_precision_{suffix}"
            recall_col = f"test_recall_{suffix}"
            lift_col = f"test_lift_{suffix}"
            if precision_col not in tuning_comparison.columns:
                continue
            rows.append(
                {
                    "model": row["model"],
                    "split": "test",
                    "topk_pct": pct,
                    "precision_at_k": row.get(precision_col, np.nan),
                    "recall_at_k": row.get(recall_col, np.nan),
                    "lift_at_k": row.get(lift_col, np.nan),
                }
            )
    return pd.DataFrame(rows)


def _fairness_rows_from_tuning(tuning_comparison: pd.DataFrame) -> pd.DataFrame:
    if tuning_comparison.empty or "model" not in tuning_comparison.columns:
        return pd.DataFrame()
    gap_col = f"test_fprcap_{MAIN_FPR_CAP:g}_max_fpr_gap"
    if gap_col not in tuning_comparison.columns:
        gap_col = "test_fprcap_0.05_max_fpr_gap"
    if gap_col not in tuning_comparison.columns:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "model": tuning_comparison["model"],
            "feature_policy": "with_housing_status",
            "threshold_policy": "valid_global_5pct_fpr",
            "attribute": "max_over_tuning_groups",
            "max_fpr_gap": tuning_comparison[gap_col],
            "max_tpr_gap": np.nan,
            "equalized_odds_difference": np.nan,
        }
    )


def build_decision_table(
    candidates: list[dict],
    threshold_test_metrics: pd.DataFrame,
    topk_frame: pd.DataFrame,
    fairness_disparity: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    holistic_all_metrics: pd.DataFrame | None = None,
    tuning_comparison: pd.DataFrame | None = None,
) -> pd.DataFrame:
    holistic_all_metrics = pd.DataFrame() if holistic_all_metrics is None else holistic_all_metrics
    tuning_comparison = pd.DataFrame() if tuning_comparison is None else tuning_comparison
    ranked = pd.DataFrame([{key: value for key, value in row.items() if key != "spec"} for row in candidates])
    if ranked.empty:
        raise RuntimeError("Missing candidate registry. Please run --stage baseline-search first.")
    ranked = ranked.sort_values("validation_pr_auc", ascending=False)
    table = ranked.head(12).copy()

    if not holistic_all_metrics.empty and {"split", "threshold_policy", "model"}.issubset(holistic_all_metrics.columns):
        threshold_test_metrics = _append_missing_models(
            threshold_test_metrics,
            holistic_all_metrics[
                (holistic_all_metrics["split"] == "test")
                & (holistic_all_metrics["threshold_policy"] == "valid_global_5pct_fpr")
            ].copy(),
        )
    threshold_test_metrics = _append_missing_models(threshold_test_metrics, _threshold_rows_from_tuning(tuning_comparison))
    topk_frame = _append_missing_models(topk_frame, _topk_rows_from_tuning(tuning_comparison))
    fairness_disparity = _append_missing_models(fairness_disparity, _fairness_rows_from_tuning(tuning_comparison))

    if not threshold_test_metrics.empty and {"split", "threshold_policy", "model"}.issubset(threshold_test_metrics.columns):
        main = threshold_test_metrics[
            (threshold_test_metrics["split"] == "test")
            & (threshold_test_metrics["threshold_policy"] == "valid_global_5pct_fpr")
        ][["model", "precision", "fdr", "recall_tpr", "fpr", "alert_rate"]].rename(
            columns={
                "precision": "main_precision",
                "fdr": "main_fdr",
                "recall_tpr": "main_recall_tpr",
                "fpr": "main_fpr",
                "alert_rate": "main_alert_rate",
            }
        )
        table = table.merge(main, on="model", how="left")

        business = threshold_test_metrics[
            (threshold_test_metrics["split"] == "test")
            & (
                threshold_test_metrics["threshold_policy"].isin(
                    ["valid_business_fdr30", "valid_business_fdr30_fallback_closest_to_target"]
                )
            )
        ].copy()
        if not business.empty:
            business = business.sort_values(["model", "feasible"], ascending=[True, False]).groupby("model").head(1)
            business = business[["model", "threshold_policy", "feasible", "precision", "fdr", "recall_tpr"]].rename(
                columns={
                    "threshold_policy": "business_policy",
                    "feasible": "business_fdr30_feasible",
                    "precision": "business_precision",
                    "fdr": "business_fdr",
                    "recall_tpr": "business_recall_tpr",
                }
            )
            table = table.merge(business, on="model", how="left")

    if not topk_frame.empty and {"split", "topk_pct", "model"}.issubset(topk_frame.columns):
        topk = topk_frame[(topk_frame["split"] == "test") & (topk_frame["topk_pct"].isin([0.01, 0.05]))].pivot_table(
            index="model",
            columns="topk_pct",
            values=["precision_at_k", "recall_at_k", "lift_at_k"],
            aggfunc="first",
        )
        if not topk.empty:
            topk.columns = [f"{metric}_top{int(topk_pct * 100)}pct" for metric, topk_pct in topk.columns]
            table = table.merge(topk.reset_index(), on="model", how="left")

    if not fairness_disparity.empty and {"feature_policy", "threshold_policy", "model"}.issubset(fairness_disparity.columns):
        fairness = fairness_disparity[
            (fairness_disparity["feature_policy"] == "with_housing_status")
            & (fairness_disparity["threshold_policy"] == "valid_global_5pct_fpr")
        ].groupby("model", as_index=False).agg(
            worst_max_fpr_gap=("max_fpr_gap", "max"),
            worst_max_tpr_gap=("max_tpr_gap", "max"),
            worst_equalized_odds_difference=("equalized_odds_difference", "max"),
        )
        table = table.merge(fairness, on="model", how="left")

    if not calibration_frame.empty and {"split", "model", "brier_score"}.issubset(calibration_frame.columns):
        calibration = calibration_frame[calibration_frame["split"] == "test"].groupby("model", as_index=False).agg(
            best_brier_score=("brier_score", "min")
        )
        table = table.merge(calibration, on="model", how="left")

    defaults = {
        "main_precision": np.nan,
        "main_fdr": np.nan,
        "main_recall_tpr": np.nan,
        "main_fpr": np.nan,
        "main_alert_rate": np.nan,
        "business_policy": "not_run",
        "business_fdr30_feasible": False,
        "business_precision": np.nan,
        "business_fdr": np.nan,
        "business_recall_tpr": np.nan,
        "precision_at_k_top1pct": np.nan,
        "recall_at_k_top1pct": np.nan,
        "lift_at_k_top1pct": np.nan,
        "precision_at_k_top5pct": np.nan,
        "recall_at_k_top5pct": np.nan,
        "lift_at_k_top5pct": np.nan,
        "worst_max_fpr_gap": np.nan,
        "worst_max_tpr_gap": np.nan,
        "worst_equalized_odds_difference": np.nan,
        "best_brier_score": np.nan,
    }
    for column, value in defaults.items():
        if column not in table.columns:
            table[column] = value

    table["interpretability"] = np.where(
        table["model_family"].isin(["Voting", "Stacking"]),
        "lower",
        np.where(table["model_family"] == "Logistic Regression", "high", "medium"),
    )
    table["feature_complexity"] = table["stage"].map(
        {
            "baseline_randomsearch": 1,
            "baseline_catboost": 1,
            "baseline_ensemble": 3,
            "balance_gate": 2,
            "advanced_gate": 3,
            "anomaly_recency_gate": 4,
            "imbalance_ensemble_gate": 3,
            "catboost_refit": 2,
        }
    ).fillna(3)
    table["deployment_readiness"] = np.where(
        table["business_fdr30_feasible"].fillna(False)
        & (table["worst_equalized_odds_difference"].fillna(0) <= 0.10),
        "deploy_candidate",
        np.where(table["main_fpr"].fillna(1.0) <= MAIN_FPR_CAP, "pilot_candidate", "not_ready"),
    )

    benef_cols = ["validation_pr_auc", "test_pr_auc", "main_recall_tpr", "lift_at_k_top1pct"]
    harm_cols = ["main_fdr", "worst_max_fpr_gap", "feature_complexity"]
    table["decision_score"] = 0.0
    for column in benef_cols:
        table["decision_score"] += table[column].rank(ascending=False, method="average", na_option="bottom")
    for column in harm_cols:
        table["decision_score"] += table[column].rank(ascending=True, method="average", na_option="bottom")

    best_pr_auc = table.sort_values(["validation_pr_auc", "test_pr_auc"], ascending=False).iloc[0]["model"]
    best_recall = table.sort_values("main_recall_tpr", ascending=False, na_position="last").iloc[0]["model"]
    feasible_rows = table[table["business_fdr30_feasible"].fillna(False)]
    best_business = (
        feasible_rows.sort_values(["business_recall_tpr", "business_precision"], ascending=False).iloc[0]["model"]
        if not feasible_rows.empty
        else None
    )
    best_topk = table.sort_values("lift_at_k_top1pct", ascending=False, na_position="last").iloc[0]["model"]
    best_fair = table.sort_values(["worst_equalized_odds_difference", "validation_pr_auc"], ascending=[True, False], na_position="last").iloc[0]["model"]
    simple_table = table[~table["model_family"].isin(["Voting", "Stacking"])].copy()
    recommendation_pool = simple_table if not simple_table.empty else table.copy()
    operationally_evaluated = recommendation_pool[
        recommendation_pool[["main_precision", "main_fdr", "main_recall_tpr", "main_fpr"]].notna().all(axis=1)
    ].copy()
    if not operationally_evaluated.empty:
        readiness_order = {"deploy_candidate": 0, "pilot_candidate": 1, "not_ready": 2}
        operationally_evaluated["_readiness_order"] = operationally_evaluated["deployment_readiness"].map(readiness_order).fillna(3)
        recommended = operationally_evaluated.sort_values(
            ["_readiness_order", "decision_score", "validation_pr_auc", "test_pr_auc"],
            ascending=[True, True, False, False],
        ).iloc[0]["model"]
    else:
        recommended = recommendation_pool.sort_values(["validation_pr_auc", "test_pr_auc"], ascending=False).iloc[0]["model"]

    table["best_by_pr_auc"] = table["model"] == best_pr_auc
    table["best_by_recall_at_fpr5"] = table["model"] == best_recall
    table["best_under_fdr30"] = table["model"] == best_business if best_business else False
    table["best_by_topk_alert_prioritization"] = table["model"] == best_topk
    table["best_by_fairness_adjusted_interpretation"] = table["model"] == best_fair
    table["recommended_final_model"] = table["model"] == recommended
    return table


def run(config, results_dir: Path, *, force: bool = False, **_) -> pd.DataFrame:
    del config
    candidates = load_candidate_registry(results_dir, required=True)
    output_dir = prepare_stage_dir(results_dir, "final-report", force=force)
    threshold_test_metrics = optional_read_csv(results_dir / "threshold_policy_test_metrics.csv")
    topk_frame = optional_read_csv(results_dir / "topk_alert_metrics.csv")
    fairness_disparity = optional_read_csv(results_dir / "fairness_disparity_summary.csv")
    calibration_frame = optional_read_csv(results_dir / "07_calibration_comparison.csv")
    holistic_all_metrics = optional_read_csv(results_dir / "holistic_all_metrics.csv")
    tuning_comparison = optional_read_csv(results_dir / STAGE_DIRS["hyperparameter-tuning-gate"] / "tuned_vs_fixed_catboost_fpr5.csv")
    table = build_decision_table(
        candidates,
        threshold_test_metrics,
        topk_frame,
        fairness_disparity,
        calibration_frame,
        holistic_all_metrics,
        tuning_comparison,
    )
    path = output_dir / "final_candidate_decision_table.csv"
    table.to_csv(path, index=False)
    shutil.copy2(path, results_dir / "final_candidate_decision_table.csv")
    return table
