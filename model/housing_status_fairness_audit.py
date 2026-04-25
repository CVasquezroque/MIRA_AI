"""
Compare fraud models with and without housing_status and audit group errors.

This is a fairness-oriented diagnostic, not a deployment approval. The goal is
to see whether removing housing_status materially changes model performance and
whether false positive / false negative behavior varies across housing groups.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone

from advanced_feature_modeling import (
    RANDOM_STATE,
    TARGET,
    catboost_scores,
    evaluate_scores,
    fit_catboost,
    make_raw_features,
    make_target_frequency_pipeline,
    make_xgboost,
    model_scores,
    split_before_preprocessing,
    stratified_training_sample,
    threshold_at_fpr,
)


RESULTS_DIR = Path("model/results/fairness_housing")
FIGURES_DIR = RESULTS_DIR / "figures"
SENSITIVE_COLUMN = "housing_status"

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


def safe_rate(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def markdown_table(frame):
    """Small markdown table writer to avoid optional tabulate dependency."""
    if frame.empty:
        return "_No rows available._"

    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_float_dtype(clean[column]):
            clean[column] = clean[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
        else:
            clean[column] = clean[column].astype(str)

    headers = clean.columns.tolist()
    rows = clean.values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]

    def format_row(values):
        return "| " + " | ".join(
            str(value).ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def evaluate_group_metrics(y_true, scores, groups, threshold):
    predictions = (scores >= threshold).astype(int)
    frame = pd.DataFrame(
        {
            "y_true": pd.Series(y_true).to_numpy(),
            "score": scores,
            "prediction": predictions,
            "group": pd.Series(groups).fillna("Unknown").astype(str).to_numpy(),
        }
    )

    rows = []
    for group, group_frame in frame.groupby("group", dropna=False):
        y_group = group_frame["y_true"].to_numpy()
        pred_group = group_frame["prediction"].to_numpy()

        tn = int(((y_group == 0) & (pred_group == 0)).sum())
        fp = int(((y_group == 0) & (pred_group == 1)).sum())
        fn = int(((y_group == 1) & (pred_group == 0)).sum())
        tp = int(((y_group == 1) & (pred_group == 1)).sum())

        legitimate_count = tn + fp
        fraud_count = tp + fn
        alert_count = tp + fp
        rows.append(
            {
                "housing_status": group,
                "n": len(group_frame),
                "fraud_count": fraud_count,
                "legitimate_count": legitimate_count,
                "fraud_rate": safe_rate(fraud_count, len(group_frame)),
                "alert_count": alert_count,
                "alert_rate": safe_rate(alert_count, len(group_frame)),
                "precision": safe_rate(tp, tp + fp),
                "recall_tpr": safe_rate(tp, tp + fn),
                "fpr": safe_rate(fp, fp + tn),
                "fnr": safe_rate(fn, fn + tp),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "small_group_warning": len(group_frame) < 500 or fraud_count < 20,
            }
        )

    return pd.DataFrame(rows).sort_values("n", ascending=False)


def train_and_score_experiment(
    model_family,
    feature_policy,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    scale_pos_weight,
):
    if model_family == "CatBoost":
        fitted = fit_catboost(X_train, y_train, X_valid, y_valid, scale_pos_weight)
        valid_scores = catboost_scores(fitted, X_valid)
        test_scores = catboost_scores(fitted, X_test)
    elif model_family == "XGBoost focal":
        pipeline = make_target_frequency_pipeline(make_xgboost(standard=False))
        fitted = clone(pipeline)
        fitted.fit(X_train, y_train)
        valid_scores = model_scores(fitted, X_valid)
        test_scores = model_scores(fitted, X_test)
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    experiment_name = f"{model_family} | {feature_policy}"
    return experiment_name, fitted, valid_scores, test_scores


def plot_group_rates(group_audit):
    plot_data = group_audit[
        (group_audit["split"] == "test")
        & (group_audit["threshold_policy"] == "valid_global_5pct_fpr")
    ].copy()

    for metric, label in [("fpr", "False Positive Rate"), ("fnr", "False Negative Rate")]:
        grid = sns.catplot(
            data=plot_data,
            x="housing_status",
            y=metric,
            hue="feature_policy",
            col="model_family",
            kind="bar",
            height=4.2,
            aspect=1.2,
            sharey=False,
            palette="Set2",
        )
        grid.set_axis_labels("Housing status", label)
        grid.set_titles("{col_name}")
        grid.fig.suptitle(f"Test {label} by Housing Status", y=1.03)
        grid.tight_layout()
        grid.savefig(FIGURES_DIR / f"test_{metric}_by_housing_status.png", dpi=150)
        plt.close(grid.fig)


def build_delta_tables(overall_metrics, group_audit):
    key_cols = ["model_family", "split", "threshold_policy"]

    with_overall = overall_metrics[overall_metrics["feature_policy"] == "with_housing_status"]
    without_overall = overall_metrics[overall_metrics["feature_policy"] == "without_housing_status"]
    overall_delta = with_overall.merge(
        without_overall,
        on=key_cols,
        suffixes=("_with_housing", "_without_housing"),
    )
    for metric in ["pr_auc", "roc_auc", "precision", "recall_tpr", "fpr"]:
        overall_delta[metric + "_delta_with_minus_without"] = (
            overall_delta[metric + "_with_housing"]
            - overall_delta[metric + "_without_housing"]
        )

    group_key_cols = key_cols + ["housing_status"]
    with_group = group_audit[group_audit["feature_policy"] == "with_housing_status"]
    without_group = group_audit[group_audit["feature_policy"] == "without_housing_status"]
    group_delta = with_group.merge(
        without_group,
        on=group_key_cols,
        suffixes=("_with_housing", "_without_housing"),
    )
    for metric in ["alert_rate", "precision", "recall_tpr", "fpr", "fnr"]:
        group_delta[metric + "_delta_with_minus_without"] = (
            group_delta[metric + "_with_housing"]
            - group_delta[metric + "_without_housing"]
        )

    return overall_delta, group_delta


def write_report(overall_metrics, overall_delta, group_delta, metadata):
    deployment_rows = overall_metrics[
        (overall_metrics["split"] == "test")
        & (overall_metrics["threshold_policy"] == "valid_global_5pct_fpr")
    ].copy()
    deployment_rows = deployment_rows.sort_values(["model_family", "feature_policy"])

    group_focus = group_delta[
        (group_delta["split"] == "test")
        & (group_delta["threshold_policy"] == "valid_global_5pct_fpr")
    ].copy()
    group_focus["abs_fpr_delta"] = group_focus["fpr_delta_with_minus_without"].abs()
    group_focus["abs_fnr_delta"] = group_focus["fnr_delta_with_minus_without"].abs()
    top_group_shift = group_focus.sort_values(
        ["model_family", "abs_fpr_delta", "abs_fnr_delta"],
        ascending=[True, False, False],
    ).groupby("model_family").head(3)

    deployment_table = markdown_table(
        deployment_rows[
            [
                "model_family",
                "feature_policy",
                "threshold",
                "precision",
                "recall_tpr",
                "fpr",
                "pr_auc",
                "roc_auc",
                "tp",
                "fp",
                "tn",
                "fn",
            ]
        ].round(6)
    )
    overall_delta_table = markdown_table(
        overall_delta[
            (overall_delta["split"] == "test")
            & (overall_delta["threshold_policy"] == "valid_global_5pct_fpr")
        ][
            [
                "model_family",
                "precision_delta_with_minus_without",
                "recall_tpr_delta_with_minus_without",
                "fpr_delta_with_minus_without",
                "pr_auc_delta_with_minus_without",
                "roc_auc_delta_with_minus_without",
            ]
        ].round(6)
    )
    top_group_shift_table = markdown_table(
        top_group_shift[
            [
                "model_family",
                "housing_status",
                "n_with_housing",
                "fraud_count_with_housing",
                "fpr_with_housing",
                "fpr_without_housing",
                "fpr_delta_with_minus_without",
                "fnr_with_housing",
                "fnr_without_housing",
                "fnr_delta_with_minus_without",
                "small_group_warning_with_housing",
            ]
        ].round(6)
    )

    report = f"""# Housing Status Fairness Audit

This audit compares candidate fraud models trained **with** and **without**
`housing_status`. It then evaluates false positive and false negative behavior
by the original `housing_status` groups.

This is a diagnostic step before any deployment recommendation. It does not make
causal claims about housing status.

## Setup

- Sensitive/proxy variable audited: `housing_status`
- Chronological split: train months {metadata['train_months']}, validation month {metadata['valid_month']}, test month {metadata['test_month']}
- Training sample rows: {metadata['train_sample_rows']:,}
- Removed constant columns: {metadata['unusable_columns']}
- Threshold policy used for the main audit: choose one global threshold on validation with FPR <= 5%, then apply that threshold to test.

## Overall Test Metrics at Validation-Selected 5% FPR Threshold

{deployment_table}

## Overall Change When Housing Status Is Included

Positive deltas mean the model trained with `housing_status` scored higher than
the model trained without it.

{overall_delta_table}

## Largest Group-Level Shifts on Test

These rows show where including `housing_status` changed group-level FPR/FNR the
most. Small groups are flagged in the CSV outputs because their rates are noisy.

{top_group_shift_table}

## How To Read The Audit

- FPR answers: among legitimate customers in this group, how many were incorrectly flagged?
- FNR answers: among fraud cases in this group, how many were missed?
- Alert rate answers: how often this group is sent to review/rejection at the chosen threshold?
- Small groups such as rare housing codes can have unstable rates; avoid overinterpreting them.

## Recommendation Before Deployment

Do not recommend deployment from model quality metrics alone. Compare:

- business lift from keeping `housing_status`,
- group-level FPR/FNR gaps,
- whether similar performance can be achieved without `housing_status`,
- whether any proxy variables recreate the same disparities even after removal.

The next review should include a domain/legal fairness review and threshold
selection based on acceptable false positive cost and customer impact.
"""
    (RESULTS_DIR / "housing_status_fairness_report.md").write_text(report, encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("Loading data...")
    data = pd.read_csv("data_banca/Base.csv")
    train, valid, test, train_months, valid_month, test_month = split_before_preprocessing(data)
    train_sample = stratified_training_sample(train)
    print(f"Training sample rows: {len(train_sample):,}")

    unusable_columns = [
        column
        for column in train.columns
        if column != TARGET and train[column].nunique(dropna=False) <= 1
    ]
    print(f"Unusable constant columns removed: {unusable_columns}")

    y_train = train_sample[TARGET].copy()
    y_valid = valid[TARGET].copy()
    y_test = test[TARGET].copy()

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / max(positive_count, 1)
    print(f"Training-sample fraud rate: {positive_count / len(y_train):.4%}")
    print(f"scale_pos_weight: {scale_pos_weight:.3f}")

    model_families = ["CatBoost", "XGBoost focal"]
    feature_policies = {
        "with_housing_status": [],
        "without_housing_status": [SENSITIVE_COLUMN],
    }

    overall_rows = []
    group_rows = []

    for model_family in model_families:
        for feature_policy, extra_drop_columns in feature_policies.items():
            drop_columns = [TARGET] + unusable_columns + extra_drop_columns
            X_train = make_raw_features(train_sample, drop_columns)
            X_valid = make_raw_features(valid, drop_columns)
            X_test = make_raw_features(test, drop_columns)

            print(f"\nTraining {model_family} | {feature_policy}...")
            experiment_name, fitted, valid_scores, test_scores = train_and_score_experiment(
                model_family,
                feature_policy,
                X_train,
                y_train,
                X_valid,
                y_valid,
                X_test,
                scale_pos_weight,
            )

            thresholds = {
                "default_0_50": 0.50,
                "valid_global_5pct_fpr": threshold_at_fpr(y_valid, valid_scores, max_fpr=0.05),
            }

            for split_name, y_split, scores, frame in [
                ("validation", y_valid, valid_scores, valid),
                ("test", y_test, test_scores, test),
            ]:
                for threshold_policy, threshold in thresholds.items():
                    metrics = evaluate_scores(y_split, scores, threshold=threshold)
                    overall_rows.append(
                        {
                            "model": experiment_name,
                            "model_family": model_family,
                            "feature_policy": feature_policy,
                            "split": split_name,
                            "threshold_policy": threshold_policy,
                            "threshold": threshold,
                            **metrics,
                        }
                    )

                    group_metrics = evaluate_group_metrics(
                        y_split,
                        scores,
                        frame[SENSITIVE_COLUMN],
                        threshold=threshold,
                    )
                    group_metrics["model"] = experiment_name
                    group_metrics["model_family"] = model_family
                    group_metrics["feature_policy"] = feature_policy
                    group_metrics["split"] = split_name
                    group_metrics["threshold_policy"] = threshold_policy
                    group_metrics["threshold"] = threshold
                    group_rows.append(group_metrics)

    overall_metrics = pd.DataFrame(overall_rows)
    group_audit = pd.concat(group_rows, ignore_index=True)
    overall_delta, group_delta = build_delta_tables(overall_metrics, group_audit)

    overall_metrics.to_csv(RESULTS_DIR / "housing_status_overall_metrics.csv", index=False)
    group_audit.to_csv(RESULTS_DIR / "housing_status_group_audit.csv", index=False)
    overall_delta.to_csv(RESULTS_DIR / "housing_status_overall_delta.csv", index=False)
    group_delta.to_csv(RESULTS_DIR / "housing_status_group_delta.csv", index=False)

    plot_group_rates(group_audit)

    metadata = {
        "train_months": train_months,
        "valid_month": valid_month,
        "test_month": test_month,
        "train_sample_rows": len(train_sample),
        "scale_pos_weight": scale_pos_weight,
        "unusable_columns": unusable_columns,
        "sensitive_column": SENSITIVE_COLUMN,
    }
    (RESULTS_DIR / "housing_status_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_report(overall_metrics, overall_delta, group_delta, metadata)

    print("\nOverall test metrics at validation-selected 5% FPR threshold:")
    print(
        overall_metrics[
            (overall_metrics["split"] == "test")
            & (overall_metrics["threshold_policy"] == "valid_global_5pct_fpr")
        ]
        .sort_values(["model_family", "feature_policy"])
        .to_string(index=False)
    )
    print(f"\nSaved housing-status audit artifacts in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
