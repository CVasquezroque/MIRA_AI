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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Holistic V2 focused checkpoints.")
    parser.add_argument(
        "--checkpoint",
        choices=["D0", "D1"],
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
        return
    if args.checkpoint == "D0":
        run_d0()
        return
    if args.checkpoint == "D1":
        run_d1()
        return
    raise SystemExit("Choose --checkpoint D0, --checkpoint D1, or --run-all.")


if __name__ == "__main__":
    main()
