"""Focused Holistic V2 fraud experiments.

This module is independent from model/holistic. All artifacts are written under
model/holistic_v2/results so the completed historical pipeline remains intact.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TARGET = "fraud_bool"
MONTH = "month"
PROTECTED_CANDIDATES = ["housing_status", "employment_status", "customer_age", "income"]

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "model"
DATA_PATH = PROJECT_DIR / "data_banca" / "Base.csv"
RESULTS_DIR = MODEL_DIR / "holistic_v2" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RUN_LOG_PATH = RESULTS_DIR / "holistic_v2_run_log.md"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

TRAIN_MONTHS = [0, 1, 2, 3, 4, 5]
VALID_MONTH = 6
TEST_MONTH = 7

SENTINEL_VALUES = [-999999, -99999, -9999, -999, -99, -9, -1, 999999, 99999, 9999]
LEAKAGE_NAME_TOKENS = [
    "future",
    "post_",
    "after_",
    "outcome",
    "label",
    "target",
    "chargeback",
    "fraud",
]

D1_TRAIN_MAX_ROWS = 180_000
D1_TEMPORAL_TRAIN_MAX_ROWS = 120_000
TOPK_PCTS = [0.005, 0.01, 0.02, 0.05]

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def append_run_log(message: str) -> None:
    ensure_output_dirs()
    timestamp = datetime.now().isoformat(timespec="seconds")
    with RUN_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"- {timestamp} | {message}\n")


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows available._"
    clean = frame.copy()
    if max_rows is not None:
        clean = clean.head(max_rows)
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

    def format_row(values: Iterable[object]) -> str:
        return "| " + " | ".join(
            str(value).ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([format_row(headers), separator, *(format_row(row) for row in rows)])


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def chronological_split(data: pd.DataFrame) -> SplitFrames:
    if TARGET not in data.columns:
        raise ValueError(f"Target column `{TARGET}` is missing.")
    if MONTH not in data.columns:
        raise ValueError(f"Month column `{MONTH}` is missing.")

    months = sorted(int(month) for month in data[MONTH].dropna().unique())
    required_months = set(TRAIN_MONTHS + [VALID_MONTH, TEST_MONTH])
    missing_months = sorted(required_months.difference(months))
    if missing_months:
        raise ValueError(f"Required chronological split months are missing: {missing_months}")

    return SplitFrames(
        train=data[data[MONTH].isin(TRAIN_MONTHS)].copy(),
        valid=data[data[MONTH] == VALID_MONTH].copy(),
        test=data[data[MONTH] == TEST_MONTH].copy(),
    )


def prevalence_frame(splits: SplitFrames) -> pd.DataFrame:
    rows = []
    for split_name, frame in [
        ("train", splits.train),
        ("validation", splits.valid),
        ("test", splits.test),
    ]:
        positives = int(frame[TARGET].sum())
        rows.append(
            {
                "split": split_name,
                "rows": len(frame),
                "fraud_count": positives,
                "legitimate_count": int(len(frame) - positives),
                "fraud_prevalence": positives / len(frame) if len(frame) else np.nan,
                "imbalance_ratio_legit_to_fraud": (len(frame) - positives) / positives
                if positives
                else np.inf,
            }
        )
    return pd.DataFrame(rows)


def monthly_drift_frame(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(MONTH)[TARGET].agg(["count", "sum", "mean"]).reset_index()
    grouped = grouped.rename(
        columns={"count": "rows", "sum": "fraud_count", "mean": "fraud_prevalence"}
    )
    grouped["legitimate_count"] = grouped["rows"] - grouped["fraud_count"]
    grouped["fraud_prevalence_delta_vs_previous"] = grouped["fraud_prevalence"].diff()
    return grouped


def plot_monthly_drift(monthly: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(monthly[MONTH], monthly["fraud_prevalence"], marker="o", color="#1f77b4")
    ax1.set_xlabel("month")
    ax1.set_ylabel("fraud prevalence")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.bar(monthly[MONTH], monthly["rows"], alpha=0.18, color="#ff7f0e")
    ax2.set_ylabel("rows")
    ax1.set_title("Monthly Fraud Rate Drift")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "00_monthly_fraud_rate_drift.png", dpi=150)
    plt.close(fig)


def missing_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in data.columns:
        missing = int(data[column].isna().sum())
        if missing:
            rows.append(
                {
                    "column": column,
                    "missing_count": missing,
                    "missing_rate": missing / len(data),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["column", "missing_count", "missing_rate"])
    return pd.DataFrame(rows).sort_values("missing_rate", ascending=False)


def sentinel_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        counts = data[column].value_counts(dropna=False)
        for value in SENTINEL_VALUES:
            if value in counts.index:
                rows.append(
                    {
                        "column": column,
                        "sentinel_value": value,
                        "count": int(counts.loc[value]),
                        "rate": float(counts.loc[value] / len(data)),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["column", "sentinel_value", "count", "rate"])
    return pd.DataFrame(rows).sort_values(["rate", "count"], ascending=False)


def constant_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in data.columns:
        counts = data[column].value_counts(dropna=False)
        nunique = int(data[column].nunique(dropna=False))
        top_rate = float(counts.iloc[0] / len(data)) if len(counts) else np.nan
        if nunique <= 1 or top_rate >= 0.999:
            rows.append(
                {
                    "column": column,
                    "nunique_including_na": nunique,
                    "top_value_rate": top_rate,
                    "reason": "constant" if nunique <= 1 else "near_constant_99_9pct",
                }
            )
    if not rows:
        return pd.DataFrame(columns=["column", "nunique_including_na", "top_value_rate", "reason"])
    return pd.DataFrame(rows).sort_values(["reason", "top_value_rate"], ascending=[True, False])


def leakage_candidates(columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column in {TARGET}:
            continue
        lower = column.lower()
        matched = [token for token in LEAKAGE_NAME_TOKENS if token in lower]
        if matched:
            severity = "exclude" if "fraud" in matched or "target" in matched or "label" in matched else "review"
            rows.append(
                {
                    "column": column,
                    "matched_tokens": ",".join(matched),
                    "leakage_action": severity,
                    "reason": "name suggests target, outcome, or future information",
                }
            )
    if not rows:
        return pd.DataFrame(columns=["column", "matched_tokens", "leakage_action", "reason"])
    return pd.DataFrame(rows).sort_values(["leakage_action", "column"])


def protected_attribute_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in PROTECTED_CANDIDATES:
        if column not in data.columns:
            rows.append(
                {
                    "column": column,
                    "present": False,
                    "nunique": 0,
                    "missing_rate": np.nan,
                    "notes": "missing",
                }
            )
            continue
        rows.append(
            {
                "column": column,
                "present": True,
                "nunique": int(data[column].nunique(dropna=True)),
                "missing_rate": float(data[column].isna().mean()),
                "notes": "protected or proxy candidate; audit before deployment",
            }
        )
    return pd.DataFrame(rows)


def write_data_audit_report(
    data: pd.DataFrame,
    splits: SplitFrames,
    monthly: pd.DataFrame,
    prevalence: pd.DataFrame,
    missing: pd.DataFrame,
    sentinels: pd.DataFrame,
    constants: pd.DataFrame,
    leakage: pd.DataFrame,
    protected: pd.DataFrame,
    duplicate_rows: int,
) -> None:
    drift_delta = monthly["fraud_prevalence"].max() - monthly["fraud_prevalence"].min()
    drift_detected = bool(drift_delta >= 0.002)
    leakage_exclusions = leakage.loc[leakage["leakage_action"] == "exclude", "column"].tolist()
    split_valid = bool(
        len(splits.train) > 0
        and len(splits.valid) > 0
        and len(splits.test) > 0
        and data[TARGET].isin([0, 1]).all()
    )

    report = f"""# Holistic V2 D0 Data Audit

## Scope

This audit validates the fixed temporal split and checks whether the dataset is
safe enough to continue with focused V2 experiments.

## Dataset

- Dataset path: `{DATA_PATH}`
- Rows: `{len(data):,}`
- Columns: `{len(data.columns):,}`
- Target column: `{TARGET}`
- Month column: `{MONTH}`
- Train months: `{TRAIN_MONTHS}`
- Validation month: `{VALID_MONTH}`
- Test month: `{TEST_MONTH}`
- Duplicate rows: `{duplicate_rows:,}`

## Split And Prevalence

{markdown_table(prevalence)}

## Monthly Fraud Drift

{markdown_table(monthly.round(6))}

Drift detected: `{"yes" if drift_detected else "no"}`. Maximum monthly
prevalence range is `{drift_delta:.6f}`.

## Missing Values

{markdown_table(missing, max_rows=30)}

## Sentinel Values

{markdown_table(sentinels, max_rows=30)}

## Constant Or Near-Constant Columns

{markdown_table(constants, max_rows=50)}

## Suspicious Leakage Columns

{markdown_table(leakage)}

Columns marked `exclude` should be removed from modeling features unless a later
checkpoint documents a stronger non-leakage justification.

## Protected / Sensitive Candidate Columns

{markdown_table(protected)}

## D0 Conclusion

- Split valid enough to continue: `{"yes" if split_valid else "no"}`
- Leakage exclusions for future checkpoints: `{leakage_exclusions if leakage_exclusions else "none"}`
- Temporal drift note: `{"keep chronological split and monitor recency behavior" if drift_detected else "chronological split still retained by design"}`
"""
    (RESULTS_DIR / "00_data_audit_report.md").write_text(report, encoding="utf-8")

    metadata = {
        "dataset_path": str(DATA_PATH),
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "train_months": TRAIN_MONTHS,
        "valid_month": VALID_MONTH,
        "test_month": TEST_MONTH,
        "duplicate_rows": int(duplicate_rows),
        "drift_detected": drift_detected,
        "max_monthly_prevalence_range": float(drift_delta),
        "leakage_exclusions": leakage_exclusions,
        "constant_or_near_constant_columns": constants["column"].tolist(),
        "protected_candidates": PROTECTED_CANDIDATES,
        "split_valid_enough_to_continue": split_valid,
    }
    (RESULTS_DIR / "00_data_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def write_decision_checkpoint_d0(
    prevalence: pd.DataFrame,
    leakage: pd.DataFrame,
    protected: pd.DataFrame,
    duplicate_rows: int,
    monthly: pd.DataFrame,
) -> None:
    leakage_exclusions = leakage.loc[leakage["leakage_action"] == "exclude", "column"].tolist()
    drift_delta = monthly["fraud_prevalence"].max() - monthly["fraud_prevalence"].min()
    drift_detected = bool(drift_delta >= 0.002)
    decision = "promote" if len(prevalence) == 3 else "discard"
    risks = []
    if leakage_exclusions:
        risks.append(f"Potential leakage columns excluded from future stages: {leakage_exclusions}.")
    if drift_detected:
        risks.append("Monthly prevalence drift is present; temporal validation remains mandatory.")
    if duplicate_rows:
        risks.append(f"Dataset has {duplicate_rows:,} duplicate rows; monitor whether duplicates are expected.")
    risks.append("Protected/proxy columns require fairness review before any deployment recommendation.")

    text = f"""# Decision Checkpoint D0 - Data Audit

## Checkpoint Name

D0 data audit

## Purpose

Validate that the dataset, fixed chronological split, target, leakage checks,
prevalence, and protected attributes are suitable for the focused V2 experiment.

## Candidates Or Options Evaluated

- Continue with fixed temporal split months 0-5 / 6 / 7.
- Exclude suspicious leakage columns before modeling.
- Retain protected/proxy candidate columns for modeling only with later fairness review.

## Validation Metrics Used

No model validation metrics are used in D0. The decision uses data validity checks:
split completeness, fraud prevalence, class imbalance, leakage-name scan, missing
and sentinel summaries, duplicate count, and monthly fraud drift.

## Decision Made

`{decision}`

## Promoted Candidates

- Fixed chronological split: train months `{TRAIN_MONTHS}`, validation month `{VALID_MONTH}`, test month `{TEST_MONTH}`.
- Protected/proxy candidates for later audit: `{PROTECTED_CANDIDATES}`.

## Discarded Candidates

- Suspicious leakage feature columns: `{leakage_exclusions if leakage_exclusions else "none"}`.

## Skipped Candidates

- `skip for runtime`: no model families are trained in D0.

## Reason For The Decision

The required target and month split are present, all split partitions are non-empty,
and the class imbalance is measurable across train, validation, and test. The
experiment can continue while preserving strict temporal validation.

## Risks Or Limitations

{chr(10).join(f"- {risk}" for risk in risks)}

## Next Step

Run D1 improved baselines and select the top five candidates using validation-only
multi-objective ranking.
"""
    (RESULTS_DIR / "decision_checkpoint_D0_data_audit.md").write_text(text, encoding="utf-8")


def run_d0() -> None:
    ensure_output_dirs()
    append_run_log("D0 started")
    data = load_data()
    splits = chronological_split(data)
    monthly = monthly_drift_frame(data)
    prevalence = prevalence_frame(splits)
    missing = missing_summary(data)
    sentinels = sentinel_summary(data)
    constants = constant_summary(data)
    leakage = leakage_candidates(data.columns)
    protected = protected_attribute_summary(data)
    duplicate_rows = int(data.duplicated().sum())

    monthly.to_csv(RESULTS_DIR / "00_monthly_fraud_rate_drift.csv", index=False)
    plot_monthly_drift(monthly)
    write_data_audit_report(
        data=data,
        splits=splits,
        monthly=monthly,
        prevalence=prevalence,
        missing=missing,
        sentinels=sentinels,
        constants=constants,
        leakage=leakage,
        protected=protected,
        duplicate_rows=duplicate_rows,
    )
    write_decision_checkpoint_d0(
        prevalence=prevalence,
        leakage=leakage,
        protected=protected,
        duplicate_rows=duplicate_rows,
        monthly=monthly,
    )
    append_run_log("D0 completed")
    print("D0 completed")
    print(f"Decision file: {RESULTS_DIR / 'decision_checkpoint_D0_data_audit.md'}")


def load_d0_metadata() -> dict:
    metadata_path = RESULTS_DIR / "00_data_audit_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Run D0 before D1; D0 metadata is missing.")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def stratified_sample_frame(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    sample, _ = train_test_split(
        frame,
        train_size=max_rows,
        stratify=frame[TARGET],
        random_state=RANDOM_STATE,
    )
    return sample.copy()


def feature_drop_columns(extra_exclusions: list[str] | None = None) -> list[str]:
    exclusions = [TARGET]
    if extra_exclusions:
        exclusions.extend(extra_exclusions)
    return list(dict.fromkeys(exclusions))


def make_feature_matrix(frame: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
    return frame.drop(columns=drop_columns, errors="ignore")


def readable_model_name(
    model_family: str,
    representation: str,
    feature_set: str,
    balance_policy: str,
    loss_type: str,
    train_strategy: str,
    ensemble_type: str,
) -> str:
    return (
        f"{model_family} | rep={representation} | feat={feature_set} | "
        f"balance={balance_policy} | loss={loss_type} | train={train_strategy} | "
        f"ensemble={ensemble_type}"
    )


def safe_model_id(readable_name: str) -> str:
    clean = readable_name.lower()
    for token in [" | ", "=", " ", "/", "\\", ":", ",", "+"]:
        clean = clean.replace(token, "_")
    clean = "".join(char for char in clean if char.isalnum() or char == "_")
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")[:180]


def threshold_at_fpr_limit(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.05) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid = np.where(fpr <= max_fpr)[0]
    if len(valid) == 0:
        return 1.0
    best_index = valid[np.argmax(tpr[valid])]
    return float(thresholds[best_index])


def topk_metrics(y_true: np.ndarray, scores: np.ndarray, prevalence: float, prefix: str) -> dict:
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    rows = {}
    positives = max(int(y_true.sum()), 1)
    labels = {0.005: "0_5", 0.01: "1", 0.02: "2", 0.05: "5"}
    for pct in TOPK_PCTS:
        k = max(1, int(np.ceil(len(y_true) * pct)))
        captured = int(y_sorted[:k].sum())
        label = labels[pct]
        rows[f"{prefix}_precision_top_{label}pct"] = captured / k
        rows[f"{prefix}_recall_top_{label}pct"] = captured / positives
        rows[f"{prefix}_lift_top_{label}pct"] = (captured / k) / prevalence if prevalence else np.nan
        rows[f"{prefix}_captured_frauds_top_{label}pct"] = captured
        rows[f"{prefix}_alerts_top_{label}pct"] = k
    return rows


def evaluate_split(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
    prevalence: float,
    prefix: str,
) -> dict:
    y_array = y_true.to_numpy()
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_array, predictions, labels=[0, 1]).ravel()
    alert_count = tp + fp
    precision = precision_score(y_array, predictions, zero_division=0)
    recall = recall_score(y_array, predictions, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    row = {
        f"{prefix}_pr_auc": average_precision_score(y_array, scores),
        f"{prefix}_roc_auc": roc_auc_score(y_array, scores),
        f"{prefix}_pr_auc_lift": average_precision_score(y_array, scores) / prevalence
        if prevalence
        else np.nan,
        f"{prefix}_precision_at_fpr5": precision,
        f"{prefix}_fdr_at_fpr5": 1 - precision,
        f"{prefix}_recall_at_fpr5": recall,
        f"{prefix}_fpr_at_fpr5": fpr,
        f"{prefix}_tp_at_fpr5": int(tp),
        f"{prefix}_fp_at_fpr5": int(fp),
        f"{prefix}_fn_at_fpr5": int(fn),
        f"{prefix}_tn_at_fpr5": int(tn),
        f"{prefix}_alert_rate_at_fpr5": alert_count / len(y_array),
    }
    row.update(topk_metrics(y_array, scores, prevalence, prefix))
    return row


def score_percentile_rank(scores: np.ndarray) -> np.ndarray:
    return pd.Series(scores).rank(method="average", pct=True).to_numpy()


def make_candidate_spec(
    model_family: str,
    representation: str,
    feature_set: str,
    balance_policy: str,
    loss_type: str,
    train_strategy: str,
    ensemble_type: str,
    notes: str,
) -> dict:
    readable = readable_model_name(
        model_family=model_family,
        representation=representation,
        feature_set=feature_set,
        balance_policy=balance_policy,
        loss_type=loss_type,
        train_strategy=train_strategy,
        ensemble_type=ensemble_type,
    )
    return {
        "model_id": safe_model_id(readable),
        "readable_model_name": readable,
        "model_family": model_family,
        "representation": representation,
        "feature_set": feature_set,
        "balance_policy": balance_policy,
        "loss_type": loss_type,
        "train_strategy": train_strategy,
        "ensemble_type": ensemble_type,
        "notes": notes,
    }


def add_evaluated_candidate(
    rows: list[dict],
    score_registry: dict[str, dict[str, np.ndarray]],
    spec: dict,
    y_valid: pd.Series,
    y_test: pd.Series,
    valid_scores: np.ndarray,
    test_scores: np.ndarray,
    valid_prevalence: float,
    test_prevalence: float,
) -> None:
    threshold = threshold_at_fpr_limit(y_valid.to_numpy(), valid_scores, max_fpr=0.05)
    row = dict(spec)
    row["selected_threshold_fpr5"] = threshold
    row.update(evaluate_split(y_valid, valid_scores, threshold, valid_prevalence, "validation"))
    row.update(evaluate_split(y_test, test_scores, threshold, test_prevalence, "test"))
    rows.append(row)
    score_registry[spec["model_id"]] = {
        "validation": np.asarray(valid_scores, dtype=float),
        "test": np.asarray(test_scores, dtype=float),
    }


def fit_and_score_individual_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[list[dict], dict[str, dict[str, np.ndarray]], list[dict]]:
    from advanced_feature_modeling import (
        catboost_scores,
        fit_catboost,
        make_lightgbm,
        make_target_frequency_pipeline,
        make_xgboost,
        model_scores,
    )

    rows: list[dict] = []
    specs: list[dict] = []
    score_registry: dict[str, dict[str, np.ndarray]] = {}
    valid_prevalence = float(y_valid.mean())
    test_prevalence = float(y_test.mean())
    scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())

    candidate_builders = [
        (
            make_candidate_spec(
                "CatBoost",
                "native_cat",
                "original_plus_basic_generated",
                "scale_pos_weight",
                "logloss",
                "months_0_5",
                "none",
                "Native categorical CatBoost with positive-class weighting.",
            ),
            "catboost",
            {"scale_pos_weight": scale_pos_weight},
        ),
        (
            make_candidate_spec(
                "CatBoost",
                "native_cat",
                "original_plus_basic_generated",
                "none",
                "logloss",
                "months_0_5",
                "none",
                "Native categorical CatBoost without class weighting.",
            ),
            "catboost",
            {"scale_pos_weight": 1.0},
        ),
        (
            make_candidate_spec(
                "XGBoost",
                "target_frequency",
                "full_advanced",
                "scale_pos_weight",
                "logloss",
                "months_0_5",
                "none",
                "Target/frequency encoded XGBoost with scale_pos_weight.",
            ),
            "pipeline",
            {"model": make_target_frequency_pipeline(make_xgboost(True, scale_pos_weight))},
        ),
        (
            make_candidate_spec(
                "XGBoost",
                "target_frequency",
                "full_advanced",
                "none",
                "logloss",
                "months_0_5",
                "none",
                "Target/frequency encoded XGBoost without class weighting.",
            ),
            "pipeline",
            {"model": make_target_frequency_pipeline(make_xgboost(True, 1.0))},
        ),
        (
            make_candidate_spec(
                "LightGBM",
                "target_frequency",
                "full_advanced",
                "scale_pos_weight",
                "logloss",
                "months_0_5",
                "none",
                "Target/frequency encoded LightGBM with scale_pos_weight.",
            ),
            "pipeline",
            {"model": make_target_frequency_pipeline(make_lightgbm(scale_pos_weight))},
        ),
        (
            make_candidate_spec(
                "LightGBM",
                "target_frequency",
                "full_advanced",
                "none",
                "logloss",
                "months_0_5",
                "none",
                "Target/frequency encoded LightGBM without class weighting.",
            ),
            "pipeline",
            {"model": make_target_frequency_pipeline(make_lightgbm(1.0))},
        ),
        (
            make_candidate_spec(
                "LogisticRegression",
                "target_frequency",
                "full_advanced",
                "class_weight_balanced",
                "logloss",
                "months_0_5",
                "none",
                "Conservative linear benchmark with balanced class weights.",
            ),
            "pipeline",
            {
                "model": make_target_frequency_pipeline(
                    LogisticRegression(
                        C=0.03,
                        class_weight="balanced",
                        max_iter=500,
                        random_state=RANDOM_STATE,
                        solver="lbfgs",
                    ),
                    scale_numeric=True,
                )
            },
        ),
        (
            make_candidate_spec(
                "LogisticRegression",
                "target_frequency",
                "full_advanced",
                "none",
                "logloss",
                "months_0_5",
                "none",
                "Conservative linear benchmark without class weights.",
            ),
            "pipeline",
            {
                "model": make_target_frequency_pipeline(
                    LogisticRegression(
                        C=0.03,
                        class_weight=None,
                        max_iter=500,
                        random_state=RANDOM_STATE,
                        solver="lbfgs",
                    ),
                    scale_numeric=True,
                )
            },
        ),
    ]

    for spec, fit_kind, params in candidate_builders:
        append_run_log(f"D1 fitting {spec['readable_model_name']}")
        if fit_kind == "catboost":
            fitted = fit_catboost(
                X_train,
                y_train,
                X_valid,
                y_valid,
                scale_pos_weight=params["scale_pos_weight"],
            )
            valid_scores = catboost_scores(fitted, X_valid)
            test_scores = catboost_scores(fitted, X_test)
        else:
            fitted = clone(params["model"])
            fitted.fit(X_train, y_train)
            valid_scores = model_scores(fitted, X_valid)
            test_scores = model_scores(fitted, X_test)

        specs.append(spec)
        add_evaluated_candidate(
            rows,
            score_registry,
            spec,
            y_valid,
            y_test,
            valid_scores,
            test_scores,
            valid_prevalence,
            test_prevalence,
        )

    return rows, score_registry, specs


def repair_test_metrics(
    rows: list[dict],
    score_registry: dict[str, dict[str, np.ndarray]],
    y_valid: pd.Series,
    y_test: pd.Series,
) -> None:
    valid_prevalence = float(y_valid.mean())
    test_prevalence = float(y_test.mean())
    for row in rows:
        model_id = row["model_id"]
        threshold = float(row["selected_threshold_fpr5"])
        for key in list(row.keys()):
            if key.startswith("test_"):
                del row[key]
        row.update(
            evaluate_split(
                y_test,
                score_registry[model_id]["test"],
                threshold,
                test_prevalence,
                "test",
            )
        )
        row["validation_pr_auc_lift"] = row["validation_pr_auc"] / valid_prevalence


def add_baseline_ensembles(
    rows: list[dict],
    score_registry: dict[str, dict[str, np.ndarray]],
    y_valid: pd.Series,
    y_test: pd.Series,
) -> list[dict]:
    base_frame = pd.DataFrame(rows)
    valid_prevalence = float(y_valid.mean())
    test_prevalence = float(y_test.mean())
    preferred_base = []
    for family in ["CatBoost", "XGBoost", "LightGBM", "LogisticRegression"]:
        family_rows = base_frame[base_frame["model_family"] == family].sort_values(
            "validation_pr_auc", ascending=False
        )
        if not family_rows.empty:
            preferred_base.append(family_rows.iloc[0]["model_id"])

    ensemble_specs: list[dict] = []
    if len(preferred_base) < 2:
        return ensemble_specs

    valid_matrix = np.column_stack([score_registry[model_id]["validation"] for model_id in preferred_base])
    test_matrix = np.column_stack([score_registry[model_id]["test"] for model_id in preferred_base])

    uniform_weights = np.repeat(1 / len(preferred_base), len(preferred_base))
    validation_pr = np.array(
        [
            float(base_frame.loc[base_frame["model_id"] == model_id, "validation_pr_auc"].iloc[0])
            for model_id in preferred_base
        ]
    )
    pr_weights = validation_pr / validation_pr.sum()

    temporal_weights = estimate_temporal_blend_weights(preferred_base, base_frame)

    blend_definitions = [
        (
            "weighted_score_blend_cat_xgb_lgbm_lr",
            uniform_weights,
            "Uniform probability blend over strongest base model per family.",
            "uniform_score",
        ),
        (
            "validation_pr_auc_weighted_score_blend_cat_xgb_lgbm_lr",
            pr_weights,
            "Validation PR-AUC weighted soft blend; D2 will tune weights more rigorously.",
            "validation_weighted",
        ),
        (
            "time_aware_month5_weighted_score_blend",
            temporal_weights,
            "Weights learned from month-5 out-of-time base scores from inner temporal fits.",
            "time_aware_month5",
        ),
    ]

    for ensemble_type, weights, notes, train_strategy in blend_definitions:
        spec = make_candidate_spec(
            "Blend" if "time_aware" not in ensemble_type else "TemporalBlend",
            "scores",
            "mixed",
            "mixed",
            "mixed",
            train_strategy,
            ensemble_type,
            notes + f" Base model ids: {preferred_base}; weights={np.round(weights, 6).tolist()}",
        )
        valid_scores = valid_matrix @ weights
        test_scores = test_matrix @ weights
        add_evaluated_candidate(
            rows,
            score_registry,
            spec,
            y_valid,
            y_test,
            valid_scores,
            test_scores,
            valid_prevalence,
            test_prevalence,
        )
        ensemble_specs.append({**spec, "base_model_ids": preferred_base, "weights": weights.tolist()})

    rank_valid = np.column_stack([score_percentile_rank(valid_matrix[:, idx]) for idx in range(valid_matrix.shape[1])])
    rank_test = np.column_stack([score_percentile_rank(test_matrix[:, idx]) for idx in range(test_matrix.shape[1])])
    spec = make_candidate_spec(
        "Blend",
        "scores",
        "mixed",
        "mixed",
        "mixed",
        "rank_average",
        "rank_average_cat_xgb_lgbm_lr",
        f"Uniform rank-average score blend. Base model ids: {preferred_base}.",
    )
    add_evaluated_candidate(
        rows,
        score_registry,
        spec,
        y_valid,
        y_test,
        rank_valid.mean(axis=1),
        rank_test.mean(axis=1),
        valid_prevalence,
        test_prevalence,
    )
    ensemble_specs.append({**spec, "base_model_ids": preferred_base, "weights": uniform_weights.tolist()})
    return ensemble_specs


def estimate_temporal_blend_weights(preferred_base: list[str], base_frame: pd.DataFrame) -> np.ndarray:
    """Estimate simple OOT month-5 weights; fall back to validation PR weights if runtime fails."""
    try:
        from advanced_feature_modeling import (
            catboost_scores,
            fit_catboost,
            make_lightgbm,
            make_target_frequency_pipeline,
            make_xgboost,
            model_scores,
        )

        metadata = load_d0_metadata()
        data = load_data()
        inner_train = data[data[MONTH].isin([0, 1, 2, 3, 4])].copy()
        inner_holdout = data[data[MONTH] == 5].copy()
        inner_train = stratified_sample_frame(inner_train, D1_TEMPORAL_TRAIN_MAX_ROWS)
        drop_columns = feature_drop_columns(metadata.get("leakage_exclusions", []))
        X_inner = make_feature_matrix(inner_train, drop_columns)
        y_inner = inner_train[TARGET].copy()
        X_holdout = make_feature_matrix(inner_holdout, drop_columns)
        y_holdout = inner_holdout[TARGET].copy()
        scale_pos_weight = float((len(y_inner) - y_inner.sum()) / y_inner.sum())
        scores = []
        for model_id in preferred_base:
            family = str(base_frame.loc[base_frame["model_id"] == model_id, "model_family"].iloc[0])
            balance = str(base_frame.loc[base_frame["model_id"] == model_id, "balance_policy"].iloc[0])
            weight = scale_pos_weight if balance in {"scale_pos_weight", "class_weight_balanced"} else 1.0
            if family == "CatBoost":
                fitted = fit_catboost(X_inner, y_inner, X_holdout, y_holdout, weight)
                holdout_scores = catboost_scores(fitted, X_holdout)
            elif family == "XGBoost":
                fitted = make_target_frequency_pipeline(make_xgboost(True, weight))
                fitted.fit(X_inner, y_inner)
                holdout_scores = model_scores(fitted, X_holdout)
            elif family == "LightGBM":
                fitted = make_target_frequency_pipeline(make_lightgbm(weight))
                fitted.fit(X_inner, y_inner)
                holdout_scores = model_scores(fitted, X_holdout)
            else:
                fitted = make_target_frequency_pipeline(
                    LogisticRegression(
                        C=0.03,
                        class_weight="balanced" if balance == "class_weight_balanced" else None,
                        max_iter=500,
                        random_state=RANDOM_STATE,
                        solver="lbfgs",
                    ),
                    scale_numeric=True,
                )
                fitted.fit(X_inner, y_inner)
                holdout_scores = model_scores(fitted, X_holdout)
            scores.append(max(average_precision_score(y_holdout, holdout_scores), 1e-8))
        weights = np.array(scores, dtype=float)
        return weights / weights.sum()
    except Exception as exc:  # noqa: BLE001 - decision report records fallback via run log.
        append_run_log(f"D1 temporal blend weight fallback: {exc}")
        validation_pr = np.array(
            [
                float(base_frame.loc[base_frame["model_id"] == model_id, "validation_pr_auc"].iloc[0])
                for model_id in preferred_base
            ]
        )
        return validation_pr / validation_pr.sum()


def normalize_metric(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    clean = series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(series.median())
    if clean.max() == clean.min():
        return pd.Series(np.ones(len(clean)), index=series.index)
    values = (clean - clean.min()) / (clean.max() - clean.min())
    return values if higher_is_better else 1 - values


def select_top5_improved_baselines(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    deployability = {
        "LogisticRegression": 1.0,
        "CatBoost": 0.95,
        "XGBoost": 0.80,
        "LightGBM": 0.80,
        "Blend": 0.65,
        "TemporalBlend": 0.55,
    }
    frame["deployability_score"] = frame["model_family"].map(deployability).fillna(0.5)
    frame["multi_objective_score"] = (
        0.30 * normalize_metric(frame["validation_pr_auc"])
        + 0.20 * normalize_metric(frame["validation_recall_at_fpr5"])
        + 0.15 * normalize_metric(frame["validation_precision_at_fpr5"])
        + 0.10 * normalize_metric(frame["validation_fdr_at_fpr5"], higher_is_better=False)
        + 0.15 * normalize_metric(frame["validation_precision_top_1pct"])
        + 0.05 * normalize_metric(frame["validation_recall_top_1pct"])
        + 0.05 * frame["deployability_score"]
    )

    selected_ids: list[str] = []

    def add_model_id(model_id: str) -> None:
        if model_id not in selected_ids:
            selected_ids.append(model_id)

    add_model_id(frame.sort_values("validation_pr_auc", ascending=False).iloc[0]["model_id"])
    add_model_id(frame.sort_values("validation_recall_at_fpr5", ascending=False).iloc[0]["model_id"])
    add_model_id(frame.sort_values("validation_precision_at_fpr5", ascending=False).iloc[0]["model_id"])
    add_model_id(frame.sort_values("validation_precision_top_1pct", ascending=False).iloc[0]["model_id"])

    ensemble_rows = frame[frame["ensemble_type"] != "none"].sort_values(
        "multi_objective_score", ascending=False
    )
    if not ensemble_rows.empty:
        add_model_id(ensemble_rows.iloc[0]["model_id"])

    conservative = frame[
        (frame["model_family"].isin(["CatBoost", "LogisticRegression"]))
        & (frame["balance_policy"].isin(["none", "class_weight_balanced"]))
    ].sort_values("multi_objective_score", ascending=False)
    if not conservative.empty:
        add_model_id(conservative.iloc[0]["model_id"])

    for model_id in frame.sort_values("multi_objective_score", ascending=False)["model_id"]:
        add_model_id(model_id)
        if len(selected_ids) >= 5:
            break

    top5 = frame[frame["model_id"].isin(selected_ids[:5])].copy()
    top5["selection_rank"] = top5["model_id"].map({model_id: idx + 1 for idx, model_id in enumerate(selected_ids[:5])})
    return top5.sort_values("selection_rank")


def plot_d1_figures(candidates: pd.DataFrame) -> None:
    top = candidates.sort_values("validation_pr_auc", ascending=False).head(12).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["model_family"] + " | " + top["balance_policy"] + " | " + top["ensemble_type"], top["validation_pr_auc"])
    ax.invert_yaxis()
    ax.set_xlabel("Validation PR-AUC")
    ax.set_title("D1 Improved Baseline PR-AUC")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_improved_baseline_pr_auc.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        candidates["validation_recall_at_fpr5"],
        candidates["validation_precision_at_fpr5"],
        s=80,
        c=candidates["validation_precision_top_1pct"],
        cmap="viridis",
    )
    ax.set_xlabel("Validation recall at FPR <= 5%")
    ax.set_ylabel("Validation precision at FPR <= 5%")
    ax.set_title("D1 Operational Baseline Trade-off")
    fig.colorbar(scatter, ax=ax, label="Precision@Top 1%")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_improved_baseline_business_metrics.png", dpi=150)
    plt.close(fig)


def save_d1_scores(score_registry: dict[str, dict[str, np.ndarray]], y_valid: pd.Series, y_test: pd.Series) -> None:
    valid_scores = {"row_number": np.arange(len(y_valid)), TARGET: y_valid.to_numpy()}
    test_scores = {"row_number": np.arange(len(y_test)), TARGET: y_test.to_numpy()}
    for model_id, scores in score_registry.items():
        valid_scores[model_id] = scores["validation"]
        test_scores[model_id] = scores["test"]
    pd.DataFrame(valid_scores).to_csv(RESULTS_DIR / "01_improved_baseline_validation_scores.csv", index=False)
    pd.DataFrame(test_scores).to_csv(RESULTS_DIR / "01_improved_baseline_test_scores.csv", index=False)


def write_d1_decision(candidates: pd.DataFrame, top5: pd.DataFrame) -> None:
    non_promoted = candidates[~candidates["model_id"].isin(top5["model_id"])].copy()
    promoted_table = top5[
        [
            "selection_rank",
            "readable_model_name",
            "validation_pr_auc",
            "validation_recall_at_fpr5",
            "validation_precision_at_fpr5",
            "validation_fdr_at_fpr5",
            "validation_precision_top_1pct",
            "multi_objective_score",
        ]
    ].round(6)
    benchmark_table = non_promoted.sort_values("validation_pr_auc", ascending=False)[
        [
            "readable_model_name",
            "validation_pr_auc",
            "validation_recall_at_fpr5",
            "validation_precision_at_fpr5",
            "validation_precision_top_1pct",
        ]
    ].head(10).round(6)

    best = top5.sort_values("multi_objective_score", ascending=False).iloc[0]
    text = f"""# Decision Checkpoint D1 - Improved Baseline Top 5

## Checkpoint Name

D1 improved baseline

## Purpose

Create a clean, interpretable baseline set before testing focused strategies A-D.
The stage includes CatBoost, XGBoost, LightGBM, Logistic Regression, and
score-level ensembles that include CatBoost.

## Candidates Or Options Evaluated

{markdown_table(candidates[["readable_model_name", "validation_pr_auc", "validation_recall_at_fpr5", "validation_precision_at_fpr5", "validation_fdr_at_fpr5", "validation_precision_top_1pct"]].round(6))}

## Validation Metrics Used

- validation PR-AUC;
- validation recall at FPR <= 5%;
- validation precision and FDR at FPR <= 5%;
- validation Precision@Top 1%;
- validation Recall@Top 1%;
- interpretability / deployability score.

Test metrics were generated for later reporting, but they were not used to select
the promoted top five.

## Decision Made

`promote`

## Promoted Candidates

{markdown_table(promoted_table)}

## Discarded Candidates

Non-promoted models are not carried into every focused strategy. They may remain
`keep as benchmark` rows for final comparison if useful.

{markdown_table(benchmark_table)}

## Skipped Candidates

- `skip for runtime`: sklearn Voting/Stacking that cannot include CatBoost's
  native categorical path directly.
- `skip for runtime`: random shuffled stacking, because D3 implements temporal
  out-of-time stacking/blending instead.

## Reason For The Decision

The selected top five cover complementary roles: best ranking model, strongest
operational recall/precision trade-offs, top-K alert quality, at least one
CatBoost benchmark, and a competitive score-level ensemble when validation
metrics justify it. The leading validation multi-objective candidate is:

`{best["readable_model_name"]}`

## Risks Or Limitations

- D1 uses a controlled stratified training sample of up to `{D1_TRAIN_MAX_ROWS:,}` rows for runtime.
- Some ensemble weights are validation-derived diagnostics; D2 performs the real
  constrained blend optimization.
- The time-aware D1 blend is deliberately simple; D3 performs the proper temporal
  out-of-time design.
- Fairness is not decided in D1 and remains `requires fairness review`.

## Next Step

Run D2 weighted score blending with CatBoost included and optimize blend weights
using validation metrics only.
"""
    (RESULTS_DIR / "decision_checkpoint_D1_improved_baseline_top5.md").write_text(text, encoding="utf-8")


def run_d1() -> None:
    ensure_output_dirs()
    append_run_log("D1 started")
    metadata = load_d0_metadata()
    data = load_data()
    splits = chronological_split(data)
    train_sample = stratified_sample_frame(splits.train, D1_TRAIN_MAX_ROWS)
    drop_columns = feature_drop_columns(metadata.get("leakage_exclusions", []))
    X_train = make_feature_matrix(train_sample, drop_columns)
    y_train = train_sample[TARGET].copy()
    X_valid = make_feature_matrix(splits.valid, drop_columns)
    y_valid = splits.valid[TARGET].copy()
    X_test = make_feature_matrix(splits.test, drop_columns)
    y_test = splits.test[TARGET].copy()

    rows, score_registry, specs = fit_and_score_individual_baselines(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    )
    repair_test_metrics(rows, score_registry, y_valid, y_test)
    ensemble_specs = add_baseline_ensembles(rows, score_registry, y_valid, y_test)
    candidates = pd.DataFrame(rows)
    top5 = select_top5_improved_baselines(candidates)
    candidates.to_csv(RESULTS_DIR / "01_improved_baseline_candidates.csv", index=False)
    (RESULTS_DIR / "01_improved_baseline_specs.json").write_text(
        json.dumps({"individual": specs, "ensembles": ensemble_specs}, indent=2),
        encoding="utf-8",
    )
    top5.to_csv(RESULTS_DIR / "02_top5_selected_from_improved_baseline.csv", index=False)
    save_d1_scores(score_registry, y_valid, y_test)
    plot_d1_figures(candidates)
    write_d1_decision(candidates, top5)
    append_run_log("D1 completed")
    print("D1 completed")
    print(f"Decision file: {RESULTS_DIR / 'decision_checkpoint_D1_improved_baseline_top5.md'}")


def load_d1_score_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    candidates_path = RESULTS_DIR / "01_improved_baseline_candidates.csv"
    valid_scores_path = RESULTS_DIR / "01_improved_baseline_validation_scores.csv"
    test_scores_path = RESULTS_DIR / "01_improved_baseline_test_scores.csv"
    if not candidates_path.exists() or not valid_scores_path.exists() or not test_scores_path.exists():
        raise FileNotFoundError("Run D1 before D2; baseline candidates and score files are required.")
    candidates = pd.read_csv(candidates_path)
    valid_scores = pd.read_csv(valid_scores_path)
    test_scores = pd.read_csv(test_scores_path)
    base_ids = candidates.loc[
        (candidates["ensemble_type"] == "none")
        & (candidates["model_family"].isin(["CatBoost", "XGBoost", "LightGBM", "LogisticRegression"])),
        "model_id",
    ].tolist()
    return candidates, valid_scores, test_scores, base_ids


def simplex_grid(n_models: int, step: float = 0.25) -> list[np.ndarray]:
    units = int(round(1 / step))
    weights: list[np.ndarray] = []

    def recurse(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            weights.append(np.array(prefix + [remaining], dtype=float) / units)
            return
        for value in range(remaining + 1):
            recurse(prefix + [value], remaining - value, slots - 1)

    recurse([], units, n_models)
    return [weight for weight in weights if np.count_nonzero(weight) > 0]


def sparse_simplex_grid(n_models: int, max_nonzero: int = 3, step: float = 0.10) -> list[np.ndarray]:
    from itertools import combinations

    sparse_weights: list[np.ndarray] = []
    for size in range(1, max_nonzero + 1):
        for indices in combinations(range(n_models), size):
            for small_weight in simplex_grid(size, step=step):
                weight = np.zeros(n_models, dtype=float)
                weight[list(indices)] = small_weight
                sparse_weights.append(weight)
    return sparse_weights


def validation_blend_metrics(y_valid: pd.Series, scores: np.ndarray) -> dict:
    threshold = threshold_at_fpr_limit(y_valid.to_numpy(), scores, max_fpr=0.05)
    return evaluate_split(y_valid, scores, threshold, float(y_valid.mean()), "validation")


def choose_weight_by_objective(
    y_valid: pd.Series,
    valid_matrix: np.ndarray,
    weights: list[np.ndarray],
    objective: str,
) -> tuple[np.ndarray, dict]:
    best_weight: np.ndarray | None = None
    best_metrics: dict | None = None
    best_key: tuple = (-np.inf,)
    for weight in weights:
        scores = valid_matrix @ weight
        metrics = validation_blend_metrics(y_valid, scores)
        if objective == "pr_auc":
            key = (
                metrics["validation_pr_auc"],
                metrics["validation_precision_top_1pct"],
                -metrics["validation_fdr_at_fpr5"],
            )
        elif objective == "recall_fpr5":
            key = (
                metrics["validation_recall_at_fpr5"],
                -metrics["validation_fdr_at_fpr5"],
                metrics["validation_pr_auc"],
            )
        elif objective == "precision_top1":
            key = (
                metrics["validation_precision_top_1pct"],
                metrics["validation_pr_auc"],
                -metrics["validation_fdr_at_fpr5"],
            )
        elif objective == "fdr_reduction":
            key = (
                -metrics["validation_fdr_at_fpr5"],
                metrics["validation_recall_at_fpr5"],
                metrics["validation_pr_auc"],
            )
        else:
            raise ValueError(f"Unknown blend objective: {objective}")
        if key > best_key:
            best_key = key
            best_weight = weight
            best_metrics = metrics
    if best_weight is None or best_metrics is None:
        raise RuntimeError(f"No weights evaluated for objective {objective}")
    return best_weight, best_metrics


def add_weighted_blend_candidate(
    rows: list[dict],
    weight_rows: list[dict],
    score_outputs: dict[str, dict[str, np.ndarray]],
    base_ids: list[str],
    base_candidates: pd.DataFrame,
    y_valid: pd.Series,
    y_test: pd.Series,
    valid_matrix: np.ndarray,
    test_matrix: np.ndarray,
    name_suffix: str,
    weights: np.ndarray,
    notes: str,
    rank_based: bool = False,
) -> None:
    spec = make_candidate_spec(
        "Blend",
        "scores" if not rank_based else "ranks",
        "mixed",
        "mixed",
        "mixed",
        "validation_weighted",
        name_suffix,
        notes,
    )
    if rank_based:
        valid_rank = np.column_stack([score_percentile_rank(valid_matrix[:, idx]) for idx in range(valid_matrix.shape[1])])
        test_rank = np.column_stack([score_percentile_rank(test_matrix[:, idx]) for idx in range(test_matrix.shape[1])])
        valid_scores = valid_rank @ weights
        test_scores = test_rank @ weights
    else:
        valid_scores = valid_matrix @ weights
        test_scores = test_matrix @ weights
    add_evaluated_candidate(
        rows,
        score_outputs,
        spec,
        y_valid,
        y_test,
        valid_scores,
        test_scores,
        float(y_valid.mean()),
        float(y_test.mean()),
    )
    for model_id, weight in zip(base_ids, weights, strict=True):
        base_row = base_candidates.loc[base_candidates["model_id"] == model_id].iloc[0]
        weight_rows.append(
            {
                "blend_model_id": spec["model_id"],
                "blend_name": spec["readable_model_name"],
                "base_model_id": model_id,
                "base_readable_model_name": base_row["readable_model_name"],
                "base_model_family": base_row["model_family"],
                "weight": float(weight),
                "non_trivial_weight": bool(weight >= 0.05),
            }
        )


def plot_d2_figures(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        results["validation_recall_at_fpr5"],
        results["validation_precision_at_fpr5"],
        s=90,
        c=results["validation_pr_auc"],
        cmap="plasma",
    )
    for _, row in results.iterrows():
        label = row["ensemble_type"].replace("_", "\n")[:28]
        ax.annotate(label, (row["validation_recall_at_fpr5"], row["validation_precision_at_fpr5"]), fontsize=7)
    ax.set_xlabel("Validation recall at FPR <= 5%")
    ax.set_ylabel("Validation precision at FPR <= 5%")
    ax.set_title("D2 Weighted Blend PR / FPR Trade-off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_weighted_blend_pr_curves.png", dpi=150)
    plt.close(fig)

    topk = results.sort_values("validation_precision_top_1pct", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        topk["ensemble_type"].str.replace("_", "\n"),
        topk["validation_precision_top_1pct"],
        color="#2ca02c",
    )
    ax.set_ylabel("Validation Precision@Top 1%")
    ax.set_title("D2 Weighted Blend Top-K Quality")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_weighted_blend_topk.png", dpi=150)
    plt.close(fig)


def write_d2_summary_and_decision(
    results: pd.DataFrame,
    weights: pd.DataFrame,
    baseline_candidates: pd.DataFrame,
) -> None:
    best_individual = baseline_candidates[baseline_candidates["ensemble_type"] == "none"].sort_values(
        "validation_pr_auc", ascending=False
    ).iloc[0]
    best_blend = results.sort_values(
        ["validation_precision_top_1pct", "validation_pr_auc", "validation_recall_at_fpr5"],
        ascending=False,
    ).iloc[0]
    pr_delta = float(best_blend["validation_pr_auc"] - best_individual["validation_pr_auc"])
    fdr_delta = float(best_blend["validation_fdr_at_fpr5"] - best_individual["validation_fdr_at_fpr5"])
    recall_delta = float(best_blend["validation_recall_at_fpr5"] - best_individual["validation_recall_at_fpr5"])
    top1_delta = float(best_blend["validation_precision_top_1pct"] - best_individual["validation_precision_top_1pct"])
    non_trivial = weights[
        (weights["blend_model_id"] == best_blend["model_id"]) & (weights["non_trivial_weight"])
    ].sort_values("weight", ascending=False)
    max_weight = float(weights[weights["blend_model_id"] == best_blend["model_id"]]["weight"].max())

    if pr_delta >= 0.005:
        pr_label = "meaningful"
    elif pr_delta >= 0.001:
        pr_label = "marginal"
    else:
        pr_label = "negligible"

    promote = (
        (pr_delta >= 0.001 or recall_delta > 0 or top1_delta > 0)
        and fdr_delta <= 0.002
        and max_weight < 0.85
    )
    decision_label = "promote" if promote else "keep as benchmark"
    if not promote and max_weight >= 0.85:
        reason = "Weights concentrate almost entirely on one base model, so the simpler base model is preferred."
    elif not promote:
        reason = "Operational gains are marginal or come with false-positive/FDR trade-offs."
    else:
        reason = "The selected blend improves at least one validation operational metric without materially worsening FDR."

    summary = f"""# D2 Weighted Blend Summary

## Best Individual Benchmark

`{best_individual["readable_model_name"]}`

## Best Weighted Blend

`{best_blend["readable_model_name"]}`

## Validation Deltas Versus Best Individual

- PR-AUC delta: `{pr_delta:.6f}` ({pr_label})
- Recall at FPR <= 5% delta: `{recall_delta:.6f}`
- FDR delta: `{fdr_delta:.6f}`
- Precision@Top 1% delta: `{top1_delta:.6f}`

## Non-Trivial Weights

{markdown_table(non_trivial[["base_readable_model_name", "weight"]].round(6))}

## Conclusions

- Did blending improve PR-AUC over the best individual model? `{"yes" if pr_delta > 0 else "no"}`
- Did blending improve recall at FPR <= 5%? `{"yes" if recall_delta > 0 else "no"}`
- Did blending improve precision or reduce FDR? `{"yes" if fdr_delta < 0 else "no"}`
- Did blending improve Precision@Top 1%? `{"yes" if top1_delta > 0 else "no"}`
- Is the blend worth keeping over CatBoost alone? `{decision_label}`
"""
    (RESULTS_DIR / "03_weighted_blend_summary.md").write_text(summary, encoding="utf-8")

    decision = f"""# Decision Checkpoint D2 - Weighted Score Blending

## Checkpoint Name

D2 weighted blend

## Purpose

Optimize score-level blends that include CatBoost and evaluate whether blending
improves ranking, FPR-constrained recall, FDR, or top-K alert quality.

## Candidates Or Options Evaluated

{markdown_table(results[["readable_model_name", "validation_pr_auc", "validation_recall_at_fpr5", "validation_precision_at_fpr5", "validation_fdr_at_fpr5", "validation_precision_top_1pct"]].round(6))}

## Validation Metrics Used

- validation PR-AUC;
- validation recall at FPR <= 5%;
- validation precision and FDR at FPR <= 5%;
- validation Precision@Top 1%;
- validation Recall@Top 1%.

Weights were learned on validation only. Test metrics were generated but not used
to choose weights or winners.

## Decision Made

`{decision_label}`

## Promoted Candidates

{markdown_table(results[results["model_id"] == best_blend["model_id"]][["readable_model_name", "validation_pr_auc", "validation_recall_at_fpr5", "validation_precision_at_fpr5", "validation_fdr_at_fpr5", "validation_precision_top_1pct"]].round(6)) if promote else "_No D2 blend promoted as final candidate yet._"}

## Discarded Candidates

{markdown_table(results[results["model_id"] != best_blend["model_id"]][["readable_model_name", "validation_pr_auc", "validation_recall_at_fpr5", "validation_precision_at_fpr5", "validation_fdr_at_fpr5", "validation_precision_top_1pct"]].round(6))}

## Skipped Candidates

- `skip for runtime`: exhaustive continuous optimization; a coarse simplex grid
  plus sparse constrained grid was used instead.

## Reason For The Decision

{reason}

PR-AUC improvement is `{pr_label}` by the configured thresholds.

## Risks Or Limitations

- Validation-only weight tuning can overfit month 6.
- Blends add operational complexity relative to CatBoost alone.
- D3 checks whether a stricter temporal design is worth the added complexity.

## Next Step

Run D3 temporal blending / stacking with out-of-time base predictions.
"""
    (RESULTS_DIR / "decision_checkpoint_D2_weighted_blend_decision.md").write_text(decision, encoding="utf-8")


def run_d2() -> None:
    ensure_output_dirs()
    append_run_log("D2 started")
    candidates, valid_score_frame, test_score_frame, base_ids = load_d1_score_inputs()
    y_valid = valid_score_frame[TARGET].astype(int)
    y_test = test_score_frame[TARGET].astype(int)
    valid_matrix = valid_score_frame[base_ids].to_numpy(dtype=float)
    test_matrix = test_score_frame[base_ids].to_numpy(dtype=float)

    rows: list[dict] = []
    weight_rows: list[dict] = []
    score_outputs: dict[str, dict[str, np.ndarray]] = {}

    uniform = np.repeat(1 / len(base_ids), len(base_ids))
    add_weighted_blend_candidate(
        rows,
        weight_rows,
        score_outputs,
        base_ids,
        candidates,
        y_valid,
        y_test,
        valid_matrix,
        test_matrix,
        "A1_uniform_probability_average",
        uniform,
        "A1 uniform average of base probabilities.",
    )
    add_weighted_blend_candidate(
        rows,
        weight_rows,
        score_outputs,
        base_ids,
        candidates,
        y_valid,
        y_test,
        valid_matrix,
        test_matrix,
        "A2_uniform_rank_average",
        uniform,
        "A2 uniform average of within-model score ranks.",
        rank_based=True,
    )

    coarse = simplex_grid(len(base_ids), step=0.25)
    sparse = sparse_simplex_grid(len(base_ids), max_nonzero=3, step=0.10)
    objectives = [
        ("A3_weighted_probability_pr_auc", "pr_auc", coarse, "A3 weighted probability blend optimized for validation PR-AUC."),
        ("A4_weighted_probability_recall_fpr5", "recall_fpr5", coarse, "A4 weighted blend optimized for recall subject to FPR <= 5%."),
        ("A5_weighted_probability_precision_top1", "precision_top1", coarse, "A5 weighted blend optimized for Precision@Top 1%."),
        ("A6_weighted_probability_fdr_reduction", "fdr_reduction", coarse, "A6 weighted blend optimized for FDR reduction with recall tie-break."),
        ("A7_sparse_blend_max3", "pr_auc", sparse, "A7 sparse blend constrained to at most three base models."),
    ]
    for name_suffix, objective, grid, notes in objectives:
        weights, _ = choose_weight_by_objective(y_valid, valid_matrix, grid, objective)
        add_weighted_blend_candidate(
            rows,
            weight_rows,
            score_outputs,
            base_ids,
            candidates,
            y_valid,
            y_test,
            valid_matrix,
            test_matrix,
            name_suffix,
            weights,
            notes,
        )

    results = pd.DataFrame(rows)
    weights = pd.DataFrame(weight_rows)
    results.to_csv(RESULTS_DIR / "03_weighted_blend_results.csv", index=False)
    weights.to_csv(RESULTS_DIR / "03_weighted_blend_weights.csv", index=False)
    pd.DataFrame(
        {"row_number": np.arange(len(y_valid)), TARGET: y_valid.to_numpy(), **{k: v["validation"] for k, v in score_outputs.items()}}
    ).to_csv(RESULTS_DIR / "03_weighted_blend_validation_scores.csv", index=False)
    pd.DataFrame(
        {"row_number": np.arange(len(y_test)), TARGET: y_test.to_numpy(), **{k: v["test"] for k, v in score_outputs.items()}}
    ).to_csv(RESULTS_DIR / "03_weighted_blend_test_scores.csv", index=False)
    plot_d2_figures(results)
    write_d2_summary_and_decision(results, weights, candidates)
    append_run_log("D2 completed")
    print("D2 completed")
    print(f"Decision file: {RESULTS_DIR / 'decision_checkpoint_D2_weighted_blend_decision.md'}")


def temporal_base_model_ids(candidates: pd.DataFrame) -> dict[str, str]:
    definitions = {
        "catboost_no_balance": ("CatBoost", "none"),
        "catboost_scale_pos_weight": ("CatBoost", "scale_pos_weight"),
        "xgboost_no_balance": ("XGBoost", "none"),
        "lightgbm_no_balance": ("LightGBM", "none"),
        "logistic_balanced": ("LogisticRegression", "class_weight_balanced"),
    }
    ids: dict[str, str] = {}
    for key, (family, balance) in definitions.items():
        match = candidates[
            (candidates["model_family"] == family)
            & (candidates["balance_policy"] == balance)
            & (candidates["ensemble_type"] == "none")
        ]
        if match.empty:
            raise ValueError(f"Missing D1 base model for temporal blend: {family} / {balance}")
        ids[key] = match.iloc[0]["model_id"]
    return ids


def fit_temporal_base_scores(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
) -> np.ndarray:
    from advanced_feature_modeling import (
        catboost_scores,
        fit_catboost,
        make_lightgbm,
        make_target_frequency_pipeline,
        make_xgboost,
        model_scores,
    )

    scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    if model_key == "catboost_no_balance":
        fitted = fit_catboost(X_train, y_train, X_holdout, y_holdout, 1.0)
        return catboost_scores(fitted, X_holdout)
    if model_key == "catboost_scale_pos_weight":
        fitted = fit_catboost(X_train, y_train, X_holdout, y_holdout, scale_pos_weight)
        return catboost_scores(fitted, X_holdout)
    if model_key == "xgboost_no_balance":
        fitted = make_target_frequency_pipeline(make_xgboost(True, 1.0))
        fitted.fit(X_train, y_train)
        return model_scores(fitted, X_holdout)
    if model_key == "lightgbm_no_balance":
        fitted = make_target_frequency_pipeline(make_lightgbm(1.0))
        fitted.fit(X_train, y_train)
        return model_scores(fitted, X_holdout)
    if model_key == "logistic_balanced":
        fitted = make_target_frequency_pipeline(
            LogisticRegression(
                C=0.03,
                class_weight="balanced",
                max_iter=700,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            ),
            scale_numeric=True,
        )
        fitted.fit(X_train, y_train)
        return model_scores(fitted, X_holdout)
    raise ValueError(f"Unknown temporal base model key: {model_key}")


def build_temporal_oof_scores() -> pd.DataFrame:
    metadata = load_d0_metadata()
    data = load_data()
    drop_columns = feature_drop_columns(metadata.get("leakage_exclusions", []))
    folds = [
        ([0, 1, 2], 3),
        ([0, 1, 2, 3], 4),
        ([0, 1, 2, 3, 4], 5),
    ]
    model_keys = [
        "catboost_no_balance",
        "catboost_scale_pos_weight",
        "xgboost_no_balance",
        "lightgbm_no_balance",
        "logistic_balanced",
    ]
    fold_frames = []
    for train_months, holdout_month in folds:
        train_frame = data[data[MONTH].isin(train_months)].copy()
        holdout_frame = data[data[MONTH] == holdout_month].copy()
        train_frame = stratified_sample_frame(train_frame, D1_TEMPORAL_TRAIN_MAX_ROWS)
        X_train = make_feature_matrix(train_frame, drop_columns)
        y_train = train_frame[TARGET].copy()
        X_holdout = make_feature_matrix(holdout_frame, drop_columns)
        y_holdout = holdout_frame[TARGET].copy()
        fold_output = pd.DataFrame(
            {
                "source_index": holdout_frame.index.to_numpy(),
                MONTH: holdout_month,
                TARGET: y_holdout.to_numpy(),
                "train_months": ",".join(str(month) for month in train_months),
            }
        )
        for model_key in model_keys:
            append_run_log(f"D3 fitting {model_key} train={train_months} holdout={holdout_month}")
            fold_output[model_key] = fit_temporal_base_scores(
                model_key,
                X_train,
                y_train,
                X_holdout,
                y_holdout,
            )
        fold_frames.append(fold_output)
    return pd.concat(fold_frames, axis=0, ignore_index=True)


def add_temporal_blend_candidate(
    rows: list[dict],
    score_outputs: dict[str, dict[str, np.ndarray]],
    y_valid: pd.Series,
    y_test: pd.Series,
    valid_scores: np.ndarray,
    test_scores: np.ndarray,
    ensemble_type: str,
    notes: str,
) -> None:
    spec = make_candidate_spec(
        "TemporalBlend",
        "oof_scores",
        "mixed",
        "mixed",
        "mixed",
        "temporal_oof",
        ensemble_type,
        notes,
    )
    add_evaluated_candidate(
        rows,
        score_outputs,
        spec,
        y_valid,
        y_test,
        valid_scores,
        test_scores,
        float(y_valid.mean()),
        float(y_test.mean()),
    )


def plot_d3_figure(results: pd.DataFrame, d2_best: pd.Series | None) -> None:
    plot_rows = results.copy()
    if d2_best is not None:
        plot_rows = pd.concat(
            [
                plot_rows,
                pd.DataFrame(
                    [
                        {
                            "ensemble_type": "D2_best_weighted_blend",
                            "validation_pr_auc": d2_best["validation_pr_auc"],
                            "validation_precision_top_1pct": d2_best["validation_precision_top_1pct"],
                            "validation_fdr_at_fpr5": d2_best["validation_fdr_at_fpr5"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        plot_rows["validation_pr_auc"],
        plot_rows["validation_precision_top_1pct"],
        s=100,
        c=1 - plot_rows["validation_fdr_at_fpr5"],
        cmap="viridis",
    )
    for _, row in plot_rows.iterrows():
        ax.annotate(str(row["ensemble_type"]).replace("_", "\n")[:30], (row["validation_pr_auc"], row["validation_precision_top_1pct"]), fontsize=8)
    ax.set_xlabel("Validation PR-AUC")
    ax.set_ylabel("Validation Precision@Top 1%")
    ax.set_title("D3 Temporal Blending Comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_temporal_blending_comparison.png", dpi=150)
    plt.close(fig)


def write_d3_summary_and_decision(results: pd.DataFrame, oof: pd.DataFrame) -> None:
    d1_candidates = pd.read_csv(RESULTS_DIR / "01_improved_baseline_candidates.csv")
    d2_results_path = RESULTS_DIR / "03_weighted_blend_results.csv"
    d2_best = None
    if d2_results_path.exists():
        d2_results = pd.read_csv(d2_results_path)
        d2_best = d2_results.sort_values(
            ["validation_precision_top_1pct", "validation_pr_auc"], ascending=False
        ).iloc[0]
    best_individual = d1_candidates[d1_candidates["ensemble_type"] == "none"].sort_values(
        "validation_pr_auc", ascending=False
    ).iloc[0]
    best_temporal = results.sort_values(
        ["validation_precision_top_1pct", "validation_fdr_at_fpr5", "validation_pr_auc"],
        ascending=[False, True, False],
    ).iloc[0]
    comparison = d2_best if d2_best is not None else best_individual
    top1_delta = float(best_temporal["validation_precision_top_1pct"] - comparison["validation_precision_top_1pct"])
    fdr_delta = float(best_temporal["validation_fdr_at_fpr5"] - comparison["validation_fdr_at_fpr5"])
    recall_delta = float(best_temporal["validation_recall_at_fpr5"] - comparison["validation_recall_at_fpr5"])
    pr_delta = float(best_temporal["validation_pr_auc"] - comparison["validation_pr_auc"])
    stable_by_fold = oof.groupby(MONTH).apply(
        lambda frame: average_precision_score(frame[TARGET], frame["temporal_meta_logistic_oof_proxy"]),
        include_groups=False,
    )
    promote = (top1_delta > 0.002 or fdr_delta < -0.002 or recall_delta > 0.01) and pr_delta > -0.002
    decision_label = "promote" if promote else "keep as benchmark"
    reason = (
        "Temporal blending improved a validation operational metric enough to justify carrying it forward."
        if promote
        else "Temporal blending is not clearly better than the simpler D2 blend, so complexity is not justified yet."
    )

    summary = f"""# D3 Temporal Blending Summary

## Best Temporal Candidate

`{best_temporal["readable_model_name"]}`

## Comparison Target

`{comparison["readable_model_name"]}`

## Validation Deltas

- PR-AUC delta: `{pr_delta:.6f}`
- Precision@Top 1% delta: `{top1_delta:.6f}`
- FDR delta: `{fdr_delta:.6f}`
- Recall at FPR <= 5% delta: `{recall_delta:.6f}`

## Fold Stability Proxy

{markdown_table(stable_by_fold.reset_index(name="oof_pr_auc").round(6))}

## Answers

- Does temporal blending beat the best individual model? `{"yes" if best_temporal["validation_pr_auc"] > best_individual["validation_pr_auc"] else "no"}`
- Does it improve top-K precision? `{"yes" if top1_delta > 0 else "no"}`
- Does it reduce FDR? `{"yes" if fdr_delta < 0 else "no"}`
- Is the improvement large enough to justify complexity? `{decision_label}`
- Does the temporal design avoid leakage? `yes, base OOF scores are generated only from prior months.`
"""
    (RESULTS_DIR / "04_temporal_blending_summary.md").write_text(summary, encoding="utf-8")

    decision = f"""# Decision Checkpoint D3 - Temporal Blending

## Checkpoint Name

D3 temporal blending

## Purpose

Evaluate a time-aware ensemble using out-of-time base predictions from forward
month folds rather than shuffled stacking.

## Candidates Or Options Evaluated

{markdown_table(results[["readable_model_name", "validation_pr_auc", "validation_recall_at_fpr5", "validation_precision_at_fpr5", "validation_fdr_at_fpr5", "validation_precision_top_1pct"]].round(6))}

## Validation Metrics Used

- validation PR-AUC;
- validation Precision@Top 1%;
- validation recall at FPR <= 5%;
- validation FDR and precision at FPR <= 5%;
- OOF fold stability proxy across months 3, 4, and 5.

## Decision Made

`{decision_label}`

## Promoted Candidates

{markdown_table(results[results["model_id"] == best_temporal["model_id"]][["readable_model_name", "validation_pr_auc", "validation_recall_at_fpr5", "validation_precision_at_fpr5", "validation_fdr_at_fpr5", "validation_precision_top_1pct"]].round(6)) if promote else "_No temporal blend promoted over the simpler D2 blend._"}

## Discarded Candidates

{markdown_table(results[results["model_id"] != best_temporal["model_id"]][["readable_model_name", "validation_pr_auc", "validation_recall_at_fpr5", "validation_precision_at_fpr5", "validation_fdr_at_fpr5", "validation_precision_top_1pct"]].round(6))}

## Skipped Candidates

- `skip for runtime`: larger meta-models and shuffled stacking.

## Reason For The Decision

{reason}

## Risks Or Limitations

- OOF folds are still trained on sampled prior-month data for runtime.
- Meta-model behavior may drift if month 6 differs materially from months 3-5.
- Additional operational complexity must be justified by clear top-K, FDR, or recall gains.

## Next Step

Run D4 hard negative mining to attack false positives directly.
"""
    (RESULTS_DIR / "decision_checkpoint_D3_temporal_blend_decision.md").write_text(decision, encoding="utf-8")
    plot_d3_figure(results, d2_best)


def run_d3() -> None:
    ensure_output_dirs()
    append_run_log("D3 started")
    candidates, valid_score_frame, test_score_frame, _ = load_d1_score_inputs()
    base_id_map = temporal_base_model_ids(candidates)
    oof = build_temporal_oof_scores()
    model_keys = list(base_id_map.keys())
    X_oof_scores = oof[model_keys].to_numpy(dtype=float)
    y_oof = oof[TARGET].astype(int)
    y_valid = valid_score_frame[TARGET].astype(int)
    y_test = test_score_frame[TARGET].astype(int)
    valid_matrix = valid_score_frame[[base_id_map[key] for key in model_keys]].to_numpy(dtype=float)
    test_matrix = test_score_frame[[base_id_map[key] for key in model_keys]].to_numpy(dtype=float)

    meta = LogisticRegression(
        C=0.10,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    meta.fit(X_oof_scores, y_oof)
    oof["temporal_meta_logistic_oof_proxy"] = meta.predict_proba(X_oof_scores)[:, 1]
    valid_meta = meta.predict_proba(valid_matrix)[:, 1]
    test_meta = meta.predict_proba(test_matrix)[:, 1]

    oof_ap = np.array(
        [max(average_precision_score(y_oof, X_oof_scores[:, idx]), 1e-8) for idx in range(len(model_keys))]
    )
    oof_weights = oof_ap / oof_ap.sum()
    valid_weighted = valid_matrix @ oof_weights
    test_weighted = test_matrix @ oof_weights

    valid_rank = np.column_stack([score_percentile_rank(valid_matrix[:, idx]) for idx in range(valid_matrix.shape[1])])
    test_rank = np.column_stack([score_percentile_rank(test_matrix[:, idx]) for idx in range(test_matrix.shape[1])])
    rank_weights = np.repeat(1 / len(model_keys), len(model_keys))

    rows: list[dict] = []
    score_outputs: dict[str, dict[str, np.ndarray]] = {}
    add_temporal_blend_candidate(
        rows,
        score_outputs,
        y_valid,
        y_test,
        valid_meta,
        test_meta,
        "temporal_logistic_meta_model",
        f"Logistic meta-model trained on OOF months 3-5. Base keys: {model_keys}.",
    )
    add_temporal_blend_candidate(
        rows,
        score_outputs,
        y_valid,
        y_test,
        valid_weighted,
        test_weighted,
        "temporal_oof_pr_auc_weighted_blend",
        f"Non-negative OOF PR-AUC weighted blend. Base keys: {model_keys}; weights={np.round(oof_weights, 6).tolist()}.",
    )
    add_temporal_blend_candidate(
        rows,
        score_outputs,
        y_valid,
        y_test,
        valid_rank @ rank_weights,
        test_rank @ rank_weights,
        "temporal_uniform_rank_blend",
        f"Uniform rank blend over final base scores. Base keys: {model_keys}.",
    )

    oof.to_csv(RESULTS_DIR / "04_temporal_blending_oof_scores.csv", index=False)
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "04_temporal_blending_results.csv", index=False)
    pd.DataFrame(
        {"row_number": np.arange(len(y_valid)), TARGET: y_valid.to_numpy(), **{k: v["validation"] for k, v in score_outputs.items()}}
    ).to_csv(RESULTS_DIR / "04_temporal_blending_validation_scores.csv", index=False)
    pd.DataFrame(
        {"row_number": np.arange(len(y_test)), TARGET: y_test.to_numpy(), **{k: v["test"] for k, v in score_outputs.items()}}
    ).to_csv(RESULTS_DIR / "04_temporal_blending_test_scores.csv", index=False)
    write_d3_summary_and_decision(results, oof)
    append_run_log("D3 completed")
    print("D3 completed")
    print(f"Decision file: {RESULTS_DIR / 'decision_checkpoint_D3_temporal_blend_decision.md'}")


def hard_negative_sample_weights(
    y_train: pd.Series,
    train_scores: np.ndarray,
    strategy: str,
    fp_weight_multiplier: float = 3.0,
) -> np.ndarray:
    """Compute per-sample training weights for hard negative mining."""
    weights = np.ones(len(y_train), dtype=float)
    if strategy == "score_band":
        threshold_50 = float(np.percentile(train_scores, 95))
        threshold_90 = float(np.percentile(train_scores, 99))
        is_negative = y_train.to_numpy() == 0
        in_band = is_negative & (train_scores >= threshold_50) & (train_scores <= threshold_90)
        weights[in_band] = fp_weight_multiplier
    elif strategy == "rank_band":
        ranks = pd.Series(train_scores).rank(pct=True).to_numpy()
        is_negative = y_train.to_numpy() == 0
        in_band = is_negative & (ranks >= 0.90) & (ranks <= 0.99)
        weights[in_band] = fp_weight_multiplier
    return weights


def run_d4() -> None:
    ensure_output_dirs()
    append_run_log("D4 started")
    from advanced_feature_modeling import (
        catboost_scores,
        fit_catboost,
        make_target_frequency_pipeline,
        make_xgboost,
        model_scores,
    )
    from catboost import CatBoostClassifier

    metadata = load_d0_metadata()
    data = load_data()
    splits = chronological_split(data)
    train_sample = stratified_sample_frame(splits.train, D1_TRAIN_MAX_ROWS)
    drop_cols = feature_drop_columns(metadata.get("leakage_exclusions", []))
    X_train = make_feature_matrix(train_sample, drop_cols)
    y_train = train_sample[TARGET].copy()
    X_valid = make_feature_matrix(splits.valid, drop_cols)
    y_valid = splits.valid[TARGET].copy()
    X_test = make_feature_matrix(splits.test, drop_cols)
    y_test = splits.test[TARGET].copy()
    valid_prevalence = float(y_valid.mean())
    test_prevalence = float(y_test.mean())
    scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())

    rows: list[dict] = []
    score_registry: dict[str, dict[str, np.ndarray]] = {}

    # --- Baseline CatBoost (no hard negative) ---
    append_run_log("D4 fitting baseline CatBoost")
    baseline_fitted = fit_catboost(X_train, y_train, X_valid, y_valid, scale_pos_weight)
    baseline_valid = catboost_scores(baseline_fitted, X_valid)
    baseline_test = catboost_scores(baseline_fitted, X_test)
    spec_baseline = make_candidate_spec(
        "CatBoost", "native_cat", "original_plus_basic_generated",
        "scale_pos_weight", "logloss", "months_0_5", "none",
        "D4 baseline CatBoost for hard negative comparison.",
    )
    add_evaluated_candidate(
        rows, score_registry, spec_baseline,
        y_valid, y_test, baseline_valid, baseline_test,
        valid_prevalence, test_prevalence,
    )

    # --- C1: Score-band hard negative weighting ---
    append_run_log("D4 fitting C1 score-band hard negative CatBoost")
    from advanced_feature_modeling import AdvancedFeatureBuilder, categorical_columns as cat_cols_fn
    builder = AdvancedFeatureBuilder()
    X_train_cb = builder.fit_transform(X_train, y_train).drop(columns=["month"], errors="ignore")
    X_valid_cb = builder.transform(X_valid).drop(columns=["month"], errors="ignore")
    X_test_cb = builder.transform(X_test).drop(columns=["month"], errors="ignore")
    cat_cols = cat_cols_fn(X_train_cb)
    for col in cat_cols:
        X_train_cb[col] = X_train_cb[col].fillna("Unknown").astype(str)
        X_valid_cb[col] = X_valid_cb[col].fillna("Unknown").astype(str)
        X_test_cb[col] = X_test_cb[col].fillna("Unknown").astype(str)

    # First pass: get scores on training data
    pass1 = CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.055, l2_leaf_reg=6.0,
        loss_function="Logloss", eval_metric="PRAUC",
        scale_pos_weight=scale_pos_weight,
        random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
    )
    pass1.fit(X_train_cb, y_train, cat_features=cat_cols,
              eval_set=(X_valid_cb, y_valid), use_best_model=True, early_stopping_rounds=40)
    train_scores_pass1 = pass1.predict_proba(X_train_cb)[:, 1]

    for strategy, multiplier, label in [
        ("score_band", 3.0, "C1_score_band_3x"),
        ("score_band", 5.0, "C1_score_band_5x"),
        ("rank_band", 3.0, "C1_rank_band_3x"),
    ]:
        weights = hard_negative_sample_weights(y_train, train_scores_pass1, strategy, multiplier)
        pool_train = __import__("catboost").Pool(X_train_cb, y_train, cat_features=cat_cols, weight=weights)
        pool_valid = __import__("catboost").Pool(X_valid_cb, y_valid, cat_features=cat_cols)
        hn_model = CatBoostClassifier(
            iterations=350, depth=6, learning_rate=0.055, l2_leaf_reg=6.0,
            loss_function="Logloss", eval_metric="PRAUC",
            scale_pos_weight=scale_pos_weight,
            random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
        )
        hn_model.fit(pool_train, eval_set=pool_valid, use_best_model=True, early_stopping_rounds=50)
        hn_valid = hn_model.predict_proba(X_valid_cb)[:, 1]
        hn_test = hn_model.predict_proba(X_test_cb)[:, 1]
        spec = make_candidate_spec(
            "HardNegativeCatBoost", "native_cat", "original_plus_basic_generated",
            "hard_negative_weighting", "logloss", "months_0_5", label,
            f"Two-pass CatBoost with {strategy} hard negative weighting (multiplier={multiplier}).",
        )
        add_evaluated_candidate(
            rows, score_registry, spec,
            y_valid, y_test, hn_valid, hn_test,
            valid_prevalence, test_prevalence,
        )

    # --- C2: Two-stage alert filter ---
    append_run_log("D4 fitting C2 two-stage alert filter")
    alert_mask_valid = baseline_valid >= np.percentile(baseline_valid, 90)
    alert_mask_test = baseline_test >= np.percentile(baseline_test, 90)
    alert_mask_train = train_scores_pass1 >= np.percentile(train_scores_pass1, 90)
    if alert_mask_train.sum() > 50:
        stage2 = CatBoostClassifier(
            iterations=200, depth=4, learning_rate=0.04, l2_leaf_reg=8.0,
            loss_function="Logloss", eval_metric="PRAUC",
            random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
        )
        stage2.fit(
            X_train_cb[alert_mask_train], y_train[alert_mask_train],
            cat_features=cat_cols,
        )
        stage2_valid_scores = np.zeros(len(y_valid))
        stage2_valid_scores[~alert_mask_valid] = baseline_valid[~alert_mask_valid] * 0.5
        stage2_valid_scores[alert_mask_valid] = stage2.predict_proba(X_valid_cb[alert_mask_valid])[:, 1]
        stage2_test_scores = np.zeros(len(y_test))
        stage2_test_scores[~alert_mask_test] = baseline_test[~alert_mask_test] * 0.5
        stage2_test_scores[alert_mask_test] = stage2.predict_proba(X_test_cb[alert_mask_test])[:, 1]
        spec_s2 = make_candidate_spec(
            "HardNegativeCatBoost", "native_cat", "original_plus_basic_generated",
            "two_stage_filter", "logloss", "months_0_5", "C2_two_stage_alert_filter",
            "Two-stage: CatBoost stage 1 filters top-10% alerts, stage 2 re-scores them.",
        )
        add_evaluated_candidate(
            rows, score_registry, spec_s2,
            y_valid, y_test, stage2_valid_scores, stage2_test_scores,
            valid_prevalence, test_prevalence,
        )

    # --- C3: Threshold-only baseline (just raise threshold) ---
    for fpr_limit in [0.03, 0.02, 0.01]:
        thr = threshold_at_fpr_limit(y_valid.to_numpy(), baseline_valid, max_fpr=fpr_limit)
        spec_thr = make_candidate_spec(
            "CatBoost", "native_cat", "original_plus_basic_generated",
            "scale_pos_weight", "logloss", "months_0_5", "none",
            f"C3 threshold-only at FPR<={fpr_limit}. Threshold={thr:.6f}.",
        )
        row = dict(spec_thr)
        row["selected_threshold_fpr5"] = thr
        row.update(evaluate_split(y_valid, baseline_valid, thr, valid_prevalence, "validation"))
        row.update(evaluate_split(y_test, baseline_test, thr, test_prevalence, "test"))
        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "05_hard_negative_results.csv", index=False)

    # Decision
    baseline_row = results[results["model_id"] == spec_baseline["model_id"]].iloc[0]
    hn_rows = results[results["model_family"] == "HardNegativeCatBoost"]
    best_hn = hn_rows.sort_values(
        ["validation_fdr_at_fpr5", "validation_pr_auc"],
        ascending=[True, False],
    ).iloc[0] if not hn_rows.empty else baseline_row
    fdr_delta = float(best_hn["validation_fdr_at_fpr5"] - baseline_row["validation_fdr_at_fpr5"])
    pr_delta = float(best_hn["validation_pr_auc"] - baseline_row["validation_pr_auc"])
    recall_delta = float(best_hn["validation_recall_at_fpr5"] - baseline_row["validation_recall_at_fpr5"])
    promote = fdr_delta < -0.005 and pr_delta > -0.003
    decision_label = "promote" if promote else "keep as benchmark"
    reason = (
        "Hard negative mining reduced FDR enough to justify carrying it forward."
        if promote
        else "Hard negative mining did not materially reduce FDR without hurting other metrics."
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(results["validation_recall_at_fpr5"], results["validation_fdr_at_fpr5"], s=90)
    for _, r in results.iterrows():
        ax.annotate(str(r.get("ensemble_type", ""))[:25], (r["validation_recall_at_fpr5"], r["validation_fdr_at_fpr5"]), fontsize=7)
    ax.set_xlabel("Recall at FPR<=5%")
    ax.set_ylabel("FDR at FPR<=5%")
    ax.set_title("D4 Hard Negative Mining: Recall vs FDR")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_hard_negative_recall_vs_fdr.png", dpi=150)
    plt.close(fig)

    decision = f"""# Decision Checkpoint D4 - Hard Negative Mining

## Checkpoint Name

D4 hard negative mining

## Purpose

Evaluate whether hard negative mining can materially reduce false positives / FDR
while preserving useful recall and alert volume.

## Candidates Or Options Evaluated

{markdown_table(results[["readable_model_name", "validation_pr_auc", "validation_recall_at_fpr5", "validation_precision_at_fpr5", "validation_fdr_at_fpr5", "validation_precision_top_1pct"]].round(6))}

## Validation Metrics Used

- validation FDR at FPR <= 5%;
- validation PR-AUC;
- validation recall at FPR <= 5%;
- validation Precision@Top 1%.

## Decision Made

`{decision_label}`

## Best Hard Negative Candidate

`{best_hn["readable_model_name"]}`

- FDR delta vs baseline: `{fdr_delta:.6f}`
- PR-AUC delta vs baseline: `{pr_delta:.6f}`
- Recall delta vs baseline: `{recall_delta:.6f}`

## Reason For The Decision

{reason}

## Next Step

Run D5 tuned focal-loss XGBoost.
"""
    (RESULTS_DIR / "decision_checkpoint_D4_hard_negative_decision.md").write_text(decision, encoding="utf-8")
    append_run_log("D4 completed")
    print("D4 completed")
    print(f"Decision file: {RESULTS_DIR / 'decision_checkpoint_D4_hard_negative_decision.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Holistic V2 focused checkpoints.")
    parser.add_argument(
        "--checkpoint",
        choices=["D0", "D1", "D2", "D3", "D4"],
        help="Checkpoint to run. More checkpoints are added as separate commits.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all currently implemented checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_all:
        run_d0()
        run_d1()
        run_d2()
        run_d3()
        run_d4()
        return
    if args.checkpoint == "D0":
        run_d0()
        return
    if args.checkpoint == "D1":
        run_d1()
        return
    if args.checkpoint == "D2":
        run_d2()
        return
    if args.checkpoint == "D3":
        run_d3()
        return
    if args.checkpoint == "D4":
        run_d4()
        return
    raise SystemExit("Choose --checkpoint D0..D4, or --run-all.")


if __name__ == "__main__":
    main()
