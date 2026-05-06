from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from model.full_holistic.data.context import load_context
from model.full_holistic.registry import load_candidate_registry, load_scores
from model.full_holistic.reporting.figures import model_names_from_candidates, write_fairness_figures
from model.full_holistic.utils.io import DependencyError, prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger
from model.full_holistic.utils.metrics import make_age_group, make_income_group
from model.full_holistic.utils.reporting import markdown_table


SELECTED_POLICIES = {
    "valid_global_5pct_fpr",
    "valid_business_fdr30",
    "valid_business_fdr30_fallback_closest_to_target",
    "valid_joint_business",
    "valid_joint_business_pareto_closest_joint_gap",
    "valid_cost_sensitive_10_to_1",
}


def _safe_rate(numerator, denominator, default=np.nan):
    return float(numerator / denominator) if denominator else float(default)


def _groups_for_rows(context, row_ids: pd.Series) -> dict[str, pd.Series]:
    test = context.test_eval
    if row_ids.isin(test.index).all():
        aligned = test.loc[row_ids]
    else:
        aligned = test.reset_index(drop=True).loc[row_ids.reset_index(drop=True)]
    return {
        "housing_status": aligned["housing_status"].fillna("Unknown").astype(str),
        "employment_status": aligned["employment_status"].fillna("Unknown").astype(str),
        "customer_age_group": make_age_group(aligned["customer_age"]),
        "income_group": make_income_group(aligned["income"]),
    }


def _evaluate_groups(model_name: str, model_family: str, threshold_policy: str, threshold: float, score_frame: pd.DataFrame, groups: dict[str, pd.Series]) -> list[dict]:
    predictions = (score_frame["score_raw"].astype(float).to_numpy() >= float(threshold)).astype(int)
    rows = []
    for attribute, group_values in groups.items():
        frame = pd.DataFrame(
            {
                "y_true": score_frame["y_true"].astype(int).to_numpy(),
                "prediction": predictions,
                "group": pd.Series(group_values).fillna("Unknown").astype(str).to_numpy(),
            }
        )
        for group_name, group_frame in frame.groupby("group", dropna=False):
            y_group = group_frame["y_true"].to_numpy()
            pred_group = group_frame["prediction"].to_numpy()
            tn = int(((y_group == 0) & (pred_group == 0)).sum())
            fp = int(((y_group == 0) & (pred_group == 1)).sum())
            fn = int(((y_group == 1) & (pred_group == 0)).sum())
            tp = int(((y_group == 1) & (pred_group == 1)).sum())
            precision = _safe_rate(tp, tp + fp, default=0.0)
            recall = _safe_rate(tp, tp + fn, default=0.0)
            fpr = _safe_rate(fp, fp + tn)
            rows.append(
                {
                    "model": model_name,
                    "model_family": model_family,
                    "feature_policy": "with_housing_status",
                    "threshold_policy": threshold_policy,
                    "attribute": attribute,
                    "group": group_name,
                    "group_size": int(len(group_frame)),
                    "fraud_prevalence": float(y_group.mean()),
                    "alert_rate": _safe_rate(tp + fp, len(group_frame), default=0.0),
                    "precision": precision,
                    "fdr": 1.0 - precision,
                    "recall_tpr": recall,
                    "fnr": _safe_rate(fn, fn + tp),
                    "fpr": fpr,
                    "tnr": _safe_rate(tn, tn + fp),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                }
            )
    return rows


def _disparity_rows(group_frame: pd.DataFrame) -> list[dict]:
    rows = []
    for (model_name, feature_policy, threshold_policy, attribute), frame in group_frame.groupby(
        ["model", "feature_policy", "threshold_policy", "attribute"]
    ):
        max_fpr = frame["fpr"].max()
        min_fpr = frame["fpr"].min()
        max_tpr = frame["recall_tpr"].max()
        min_tpr = frame["recall_tpr"].min()
        max_alert = frame["alert_rate"].max()
        min_alert = frame["alert_rate"].min()
        approval_rate = 1.0 - frame["alert_rate"]
        non_zero_fpr = frame["fpr"][frame["fpr"] > 0]
        rows.append(
            {
                "model": model_name,
                "feature_policy": feature_policy,
                "threshold_policy": threshold_policy,
                "attribute": attribute,
                "max_fpr_gap": max_fpr - min_fpr,
                "max_tpr_gap": max_tpr - min_tpr,
                "fpr_ratio_worst_to_best": _safe_rate(non_zero_fpr.max(), non_zero_fpr.min()) if not non_zero_fpr.empty else np.nan,
                "tpr_ratio_worst_to_best": _safe_rate(min_tpr, max_tpr),
                "equal_opportunity_difference": max_tpr - min_tpr,
                "equalized_odds_difference": max(max_fpr - min_fpr, max_tpr - min_tpr),
                "disparate_impact_ratio_alert_positive": _safe_rate(min_alert, max_alert),
                "disparate_impact_ratio_approval_positive": _safe_rate(approval_rate.min(), approval_rate.max()),
                "worst_fpr_group": frame.sort_values("fpr", ascending=False).iloc[0]["group"],
                "lowest_tpr_group": frame.sort_values("recall_tpr", ascending=True).iloc[0]["group"],
            }
        )
    return rows


def run(config, results_dir: Path, *, force: bool = False, **_) -> None:
    context = load_context(results_dir)
    candidates = load_candidate_registry(results_dir, required=True)
    threshold_path = results_dir / "threshold_policy_test_metrics.csv"
    if not threshold_path.exists():
        raise DependencyError("Missing threshold policy metrics. Please run --stage operational-thresholds first.")
    threshold_test_metrics = pd.read_csv(threshold_path)
    scores = load_scores(results_dir, required=True)
    output_dir = prepare_stage_dir(results_dir, "fairness", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "Fairness Run Log")

    top_candidates = candidates[: min(8, config.fairness_top_n)]
    if not (output_dir / "fairness_disparity_summary.csv").exists():
        selected = threshold_test_metrics[
            (threshold_test_metrics["split"] == "test")
            & (threshold_test_metrics["threshold_policy"].isin(SELECTED_POLICIES))
            & (~threshold_test_metrics["threshold"].isna())
        ].copy()
        group_rows = []
        overall_rows = []
        for candidate in top_candidates:
            model_scores = scores[(scores["model"] == candidate["model"]) & (scores["split"] == "test")].sort_values("row_id")
            if model_scores.empty:
                continue
            groups = _groups_for_rows(context, model_scores["row_id"])
            candidate_thresholds = selected[selected["model"] == candidate["model"]]
            for _, threshold_row in candidate_thresholds.iterrows():
                overall_rows.append(
                    {
                        "model": candidate["model"],
                        "model_family": candidate["model_family"],
                        "feature_policy": "with_housing_status",
                        **threshold_row.to_dict(),
                    }
                )
                group_rows.extend(
                    _evaluate_groups(
                        candidate["model"],
                        candidate["model_family"],
                        threshold_row["threshold_policy"],
                        threshold_row["threshold"],
                        model_scores,
                        groups,
                    )
                )
        if not group_rows:
            raise DependencyError("Missing score rows for fairness audit. Please run --stage baseline-search first.")
        overall = pd.DataFrame(overall_rows)
        fairness_by_group = pd.DataFrame(group_rows)
        fairness_disparity = pd.DataFrame(_disparity_rows(fairness_by_group))
        overall.to_csv(output_dir / "06_housing_status_overall_metrics.csv", index=False)
        fairness_by_group.to_csv(output_dir / "fairness_by_group.csv", index=False)
        fairness_by_group.to_csv(output_dir / "06_housing_status_group_audit.csv", index=False)
        fairness_disparity.to_csv(output_dir / "fairness_disparity_summary.csv", index=False)
        report = "# Fairness Report\n\nPersisted-score fairness audit. The optional without-`housing_status` refit comparison was not run in this stage.\n\n"
        report += "## Housing Status Disparities\n\n"
        housing = fairness_disparity[fairness_disparity["attribute"] == "housing_status"]
        report += markdown_table(housing.round(6)) if not housing.empty else "_No housing_status disparity rows available._"
        (output_dir / "06_fairness_report.md").write_text(report + "\n", encoding="utf-8")
        (output_dir / "06_housing_status_fairness_report.md").write_text(report + "\n", encoding="utf-8")
    fairness_by_group_path = output_dir / "fairness_by_group.csv"
    fairness_disparity_path = output_dir / "fairness_disparity_summary.csv"
    if fairness_by_group_path.exists() and fairness_disparity_path.exists():
        try:
            write_fairness_figures(
                pd.read_csv(fairness_by_group_path),
                pd.read_csv(fairness_disparity_path),
                output_dir,
                model_names=model_names_from_candidates(top_candidates, top_n=8),
                focus_model=top_candidates[0]["model"] if top_candidates else None,
            )
        except Exception as exc:
            logger.write("Fairness Figures Skipped", repr(exc))

    for name in [
        "06_housing_status_overall_metrics.csv",
        "fairness_by_group.csv",
        "06_housing_status_group_audit.csv",
        "fairness_disparity_summary.csv",
        "06_fairness_report.md",
        "06_housing_status_fairness_report.md",
    ]:
        path = output_dir / name
        if path.exists():
            shutil.copy2(path, results_dir / name)
    print(f"[fairness] Saved fairness artifacts in: {output_dir}")
