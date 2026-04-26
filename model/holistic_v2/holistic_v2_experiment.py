"""Focused Holistic V2 fraud experiments.

This module is independent from model/holistic. All artifacts are written under
model/holistic_v2/results so the completed historical pipeline remains intact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Holistic V2 focused checkpoints.")
    parser.add_argument(
        "--checkpoint",
        choices=["D0"],
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
    if args.run_all or args.checkpoint == "D0":
        run_d0()
        return
    raise SystemExit("Choose --checkpoint D0 or --run-all.")


if __name__ == "__main__":
    main()
