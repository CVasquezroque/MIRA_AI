from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from model.full_holistic.constants import TARGET
from model.full_holistic.paths import DATA_PATH
from model.full_holistic.paths import stage_dir
from model.full_holistic.utils.io import DependencyError, require_file
from model.full_holistic.utils.serialization import dump_joblib, load_joblib


@dataclass
class DataContext:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    train_sample: pd.DataFrame
    valid_eval: pd.DataFrame
    test_eval: pd.DataFrame
    train_months: list[int] | None
    valid_month: int | None
    test_month: int | None
    unusable_columns: list[str]
    scale_pos_weight: float
    original_feature_columns: list[str]
    original_categorical_columns: list[str]
    original_numeric_columns: list[str]
    original_binary_columns: list[str]
    train_prevalence: float
    valid_prevalence: float
    test_prevalence: float


def context_path(results_dir: Path) -> Path:
    return stage_dir(results_dir, "data-audit") / "context.joblib"


def metadata_path(results_dir: Path) -> Path:
    return stage_dir(results_dir, "data-audit") / "context_metadata.json"


def save_context(results_dir: Path, context, config) -> None:
    path = context_path(results_dir)
    dump_joblib(context, path)
    metadata = {
        "config": asdict(config),
        "train_months": context.train_months,
        "valid_month": context.valid_month,
        "test_month": context.test_month,
        "unusable_columns": context.unusable_columns,
        "fraud_prevalence": {
            "train": context.train_prevalence,
            "validation": context.valid_prevalence,
            "test": context.test_prevalence,
        },
    }
    metadata_path(results_dir).write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def load_context(results_dir: Path):
    path = require_file(
        context_path(results_dir),
        "Missing data context. Please run --stage data-audit first.",
    )
    return load_joblib(path)


def load_context_optional(results_dir: Path):
    try:
        return load_context(results_dir)
    except DependencyError:
        return None


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        dtype_name = str(frame[column].dtype)
        if dtype_name in {"object", "category", "str", "string"} or dtype_name.startswith("string["):
            columns.append(column)
    return columns


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    categorical = set(categorical_columns(frame))
    return [column for column in frame.columns if column not in categorical]


def temporal_split(data: pd.DataFrame):
    if "month" not in data.columns or data["month"].nunique(dropna=True) < 4:
        raise ValueError("Temporal split requires a `month` column with at least four distinct months.")
    months = [int(month) for month in sorted(data["month"].dropna().unique())]
    test_month = months[-1]
    valid_month = months[-2]
    train_months = months[:-2]
    train = data[data["month"].isin(train_months)].copy()
    valid = data[data["month"] == valid_month].copy()
    test = data[data["month"] == test_month].copy()
    return train, valid, test, train_months, valid_month, test_month


def sample_frame(frame: pd.DataFrame, max_rows: int | None, random_state: int = 42) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.copy()
    positive = frame[frame[TARGET] == 1]
    negative = frame[frame[TARGET] == 0]
    pos_take = min(len(positive), max(1, int(max_rows * max(float(frame[TARGET].mean()), 0.02))))
    neg_take = max_rows - pos_take
    sampled = pd.concat(
        [
            positive.sample(n=pos_take, random_state=random_state) if len(positive) > pos_take else positive,
            negative.sample(n=min(neg_take, len(negative)), random_state=random_state) if len(negative) > neg_take else negative,
        ],
        axis=0,
    )
    return sampled.sample(frac=1.0, random_state=random_state).copy()


def build_context(config) -> tuple[DataContext, pd.DataFrame]:
    data = pd.read_csv(DATA_PATH)
    train, valid, test, train_months, valid_month, test_month = temporal_split(data)
    train_sample = sample_frame(train, config.train_rows)
    valid_eval = sample_frame(valid, config.eval_rows)
    test_eval = sample_frame(test, config.eval_rows)
    feature_frame = train.drop(columns=[TARGET], errors="ignore")
    cat_cols = categorical_columns(feature_frame)
    num_cols = numeric_columns(feature_frame)
    binary_cols = [
        column
        for column in feature_frame.columns
        if column not in cat_cols and feature_frame[column].dropna().nunique() <= 2
    ]
    positive = int(train_sample[TARGET].sum())
    negative = int(len(train_sample) - positive)
    scale_pos_weight = negative / max(positive, 1)
    context = DataContext(
        train=train,
        valid=valid,
        test=test,
        train_sample=train_sample,
        valid_eval=valid_eval,
        test_eval=test_eval,
        train_months=train_months,
        valid_month=valid_month,
        test_month=test_month,
        unusable_columns=[],
        scale_pos_weight=scale_pos_weight,
        original_feature_columns=feature_frame.columns.tolist(),
        original_categorical_columns=cat_cols,
        original_numeric_columns=num_cols,
        original_binary_columns=binary_cols,
        train_prevalence=float(train[TARGET].mean()),
        valid_prevalence=float(valid[TARGET].mean()),
        test_prevalence=float(test[TARGET].mean()),
    )
    drift = (
        data.groupby("month", as_index=False)[TARGET]
        .agg(rows="size", frauds="sum", fraud_prevalence="mean")
        .sort_values("month")
    )
    return context, drift
