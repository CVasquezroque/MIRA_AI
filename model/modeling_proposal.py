"""
Simple modeling proposal built from the EDA findings.

This is intentionally not a production pipeline. It compares a few readable
preprocessing variants and five model families using fraud-focused metrics.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TARGET = "fraud_bool"
RESULTS_DIR = Path("model/results")

# Based on the EDA and Diccionario.xlsx. Do not convert every -1 blindly.
SENTINEL_TO_NA = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

# Missingness itself may be useful in fraud detection, so compare with/without these flags.
MISSING_FLAG_COLUMNS = [
    "prev_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
]

# IQR flags are only for continuous or count-like variables, not binary columns.
OUTLIER_COLUMNS = [
    "proposed_credit_limit",
    "session_length_in_minutes",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
]

# Candidate variables from the EDA. The script still checks train skewness first.
LOG_CANDIDATES = [
    "proposed_credit_limit",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "session_length_in_minutes",
]

SENSITIVE_REVIEW_COLUMNS = ["housing_status", "employment_status"]

PREPROCESSING_VARIANTS = [
    {
        "variant": "median_imputation_only",
        "add_missing_flags": False,
        "add_outlier_flags": False,
        "add_log_features": False,
    },
    {
        "variant": "add_missing_flags",
        "add_missing_flags": True,
        "add_outlier_flags": False,
        "add_log_features": False,
    },
    {
        "variant": "add_outlier_flags",
        "add_missing_flags": False,
        "add_outlier_flags": True,
        "add_log_features": False,
    },
    {
        "variant": "missing_and_outlier_flags",
        "add_missing_flags": True,
        "add_outlier_flags": True,
        "add_log_features": False,
    },
    {
        "variant": "missing_outlier_log1p",
        "add_missing_flags": True,
        "add_outlier_flags": True,
        "add_log_features": True,
    },
]


def split_before_preprocessing(data):
    """Prefer chronological split because the EDA found monthly drift."""
    if "month" in data.columns and data["month"].nunique() >= 3:
        months = [int(month) for month in sorted(data["month"].dropna().unique())]
        test_month = months[-1]
        valid_month = months[-2]
        train_months = months[:-2]

        train = data[data["month"].isin(train_months)].copy()
        valid = data[data["month"] == valid_month].copy()
        test = data[data["month"] == test_month].copy()

        print("Using chronological split based on month.")
        print(f"Train months: {train_months}")
        print(f"Validation month: {valid_month}")
        print(f"Test month: {test_month}")
        return train, valid, test

    print("Month split is not feasible, so using stratified random split.")
    train, temp = train_test_split(
        data,
        test_size=0.30,
        stratify=data[TARGET],
        random_state=RANDOM_STATE,
    )
    valid, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp[TARGET],
        random_state=RANDOM_STATE,
    )
    return train.copy(), valid.copy(), test.copy()


def prepare_feature_variant(raw_splits, variant):
    """Apply one preprocessing variant without fitting on validation/test."""
    prepared = {name: X.copy() for name, X in raw_splits.items()}

    if variant["add_missing_flags"]:
        # Create flags before replacing -1. This keeps missingness as a possible signal.
        for X in prepared.values():
            for column in MISSING_FLAG_COLUMNS:
                if column in X.columns:
                    X[column + "_was_missing"] = (X[column] == -1).astype(int)

    for X in prepared.values():
        for column in SENTINEL_TO_NA:
            if column in X.columns:
                X[column] = X[column].replace(-1, np.nan)

    iqr_bounds = {}
    if variant["add_outlier_flags"]:
        # Fit IQR limits on train only, then reuse them for validation/test.
        X_train = prepared["train"]
        for column in OUTLIER_COLUMNS:
            if column not in X_train.columns or X_train[column].nunique(dropna=True) <= 2:
                continue

            values = X_train[column].dropna()
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            if iqr > 0:
                iqr_bounds[column] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        for X in prepared.values():
            for column, (lower, upper) in iqr_bounds.items():
                flag_name = column + "_is_iqr_outlier"
                X[flag_name] = ((X[column] < lower) | (X[column] > upper)).astype(int)

    log_columns = []
    if variant["add_log_features"]:
        # Select log1p features from train only, and only for positive long-tailed variables.
        X_train = prepared["train"]
        for column in LOG_CANDIDATES:
            if column not in X_train.columns:
                continue

            values = X_train[column].dropna()
            if values.empty or values.min() < 0:
                continue

            median = values.median()
            p99 = values.quantile(0.99)
            long_tail = median > 0 and p99 / median >= 5
            strong_skew = abs(values.skew()) >= 1

            if long_tail or strong_skew:
                log_columns.append(column)

        for X in prepared.values():
            for column in log_columns:
                # log1p works with zeros, while plain log does not.
                X[column + "_log1p"] = np.log1p(X[column])

    details = {
        "missing_flags": [column + "_was_missing" for column in MISSING_FLAG_COLUMNS]
        if variant["add_missing_flags"]
        else [],
        "outlier_flags": [column + "_is_iqr_outlier" for column in iqr_bounds],
        "log_features": [column + "_log1p" for column in log_columns],
    }

    return prepared["train"], prepared["valid"], prepared["test"], details


def build_preprocessors(X_train):
    categorical_features = []
    for column in X_train.columns:
        dtype_name = str(X_train[column].dtype)
        if dtype_name in ["object", "category", "str", "string"] or dtype_name.startswith("string["):
            categorical_features.append(column)

    numeric_features = [column for column in X_train.columns if column not in categorical_features]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    logistic_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # RobustScaler is a good default here because the EDA found outliers.
            ("scaler", RobustScaler()),
        ]
    )

    tree_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    logistic_preprocessor = ColumnTransformer(
        transformers=[
            ("num", logistic_numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", tree_numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return logistic_preprocessor, tree_preprocessor


def build_models(logistic_preprocessor, tree_preprocessor, y_train):
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / max(positive_count, 1)

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", logistic_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=400,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    DecisionTreeClassifier(
                        class_weight="balanced",
                        max_depth=8,
                        min_samples_leaf=100,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=30,
                        max_depth=10,
                        min_samples_leaf=100,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=30,
                        max_depth=10,
                        min_samples_leaf=100,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        boosting_model = XGBClassifier(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        boosting_name = "XGBoost"
    except ImportError:
        try:
            from lightgbm import LGBMClassifier

            boosting_model = LGBMClassifier(
                n_estimators=60,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            boosting_name = "LightGBM"
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier

            boosting_model = HistGradientBoostingClassifier(
                max_iter=60,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=RANDOM_STATE,
            )
            boosting_name = "HistGradientBoosting"
            print("Install xgboost or lightgbm to use a stronger gradient boosting library.")

    models[boosting_name] = Pipeline(
        steps=[
            ("preprocess", tree_preprocessor),
            ("model", boosting_model),
        ]
    )

    return models


def evaluate_predictions(y_true, scores, threshold=0.50):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    try:
        pr_auc = average_precision_score(y_true, scores)
    except ValueError:
        pr_auc = np.nan

    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = np.nan

    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall_tpr": recall_score(y_true, predictions, zero_division=0),
        "fpr": fpr,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def threshold_table(y_true, scores):
    rows = []

    for threshold in np.arange(0.05, 0.96, 0.05):
        metrics = evaluate_predictions(y_true, scores, threshold)
        metrics["threshold"] = threshold
        rows.append(metrics)

    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data_banca/Base.csv...")
    data = pd.read_csv("data_banca/Base.csv")
    print(f"Loaded data: {data.shape[0]:,} rows and {data.shape[1]:,} columns.")

    train, valid, test = split_before_preprocessing(data)
    print(f"Split sizes: train={len(train):,}, valid={len(valid):,}, test={len(test):,}")

    # Remove only clearly unusable columns, using train after the split.
    # Low individual correlation is not enough to remove a variable.
    unusable_columns = [
        column
        for column in train.columns
        if column != TARGET and train[column].nunique(dropna=False) <= 1
    ]

    y_train = train[TARGET].copy()
    y_valid = valid[TARGET].copy()
    y_test = test[TARGET].copy()

    drop_columns = [TARGET] + unusable_columns
    if "month" in data.columns:
        # Use month for chronological evaluation. Keep it out of the first baseline feature set.
        drop_columns.append("month")

    raw_splits = {
        "train": train.drop(columns=drop_columns, errors="ignore"),
        "valid": valid.drop(columns=drop_columns, errors="ignore"),
        "test": test.drop(columns=drop_columns, errors="ignore"),
    }

    print(f"Removed columns: {unusable_columns}")
    print(f"Sentinel columns converted to NaN when present: {SENTINEL_TO_NA}")
    print(
        "Imputation is always fitted inside each training pipeline. The comparison is "
        "median/mode imputation alone versus imputation plus missingness indicators."
    )
    print(f"Sensitive review columns to compare with/without later: {SENSITIVE_REVIEW_COLUMNS}")

    all_validation_rows = []
    best_model = None
    best_model_name = None
    best_variant_name = None
    best_valid_scores = None
    best_pr_auc = -np.inf
    best_test_data = None

    for variant in PREPROCESSING_VARIANTS:
        variant_name = variant["variant"]
        print(f"\nPreprocessing variant: {variant_name}")

        X_train, X_valid, X_test, details = prepare_feature_variant(raw_splits, variant)
        print(f"  Missingness flags: {details['missing_flags']}")
        print(f"  IQR outlier flags: {details['outlier_flags']}")
        print(f"  log1p features: {details['log_features']}")

        logistic_preprocessor, tree_preprocessor = build_preprocessors(X_train)
        models = build_models(logistic_preprocessor, tree_preprocessor, y_train)

        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            model.fit(X_train, y_train)

            valid_scores = model.predict_proba(X_valid)[:, 1]
            metrics = evaluate_predictions(y_valid, valid_scores, threshold=0.50)
            metrics["variant"] = variant_name
            metrics["model"] = model_name
            all_validation_rows.append(metrics)

            if metrics["pr_auc"] > best_pr_auc:
                best_pr_auc = metrics["pr_auc"]
                best_model = model
                best_model_name = model_name
                best_variant_name = variant_name
                best_valid_scores = valid_scores
                best_test_data = X_test

    validation_results = pd.DataFrame(all_validation_rows).sort_values("pr_auc", ascending=False)
    validation_path = RESULTS_DIR / "validation_model_comparison.csv"
    validation_results.to_csv(validation_path, index=False)

    print("\nValidation metrics at threshold 0.50, sorted by PR-AUC:")
    print(validation_results.to_string(index=False))
    print(f"\nSaved validation comparison to {validation_path}")

    thresholds = threshold_table(y_valid, best_valid_scores)
    thresholds_path = RESULTS_DIR / "threshold_candidates_best_model.csv"
    thresholds.to_csv(thresholds_path, index=False)

    print(f"\nBest validation combination: {best_model_name} + {best_variant_name}")
    print("\nThreshold candidates for the best validation combination:")
    print(thresholds.to_string(index=False))
    print(f"\nSaved threshold candidates to {thresholds_path}")

    print(
        "\nUse the validation threshold table to define a score-based flow: "
        "low scores for automatic approval, medium scores for analyst review, "
        "and high scores for rejection or stronger verification."
    )
    print(
        "Choose thresholds by balancing fraud loss reduction, false positives, "
        "and customer experience. Do not rely only on the default 0.50 threshold."
    )

    # After choosing thresholds on validation, run one final check on test.
    test_scores = best_model.predict_proba(best_test_data)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_scores, threshold=0.50)
    test_metrics["model"] = best_model_name
    test_metrics["variant"] = best_variant_name
    test_path = RESULTS_DIR / "test_metrics_best_model.csv"
    pd.DataFrame([test_metrics]).to_csv(test_path, index=False)

    print(f"\nFinal test check for {best_model_name} + {best_variant_name} at threshold 0.50:")
    print(pd.DataFrame([test_metrics]).to_string(index=False))
    print(f"\nSaved final test metrics to {test_path}")


if __name__ == "__main__":
    main()
