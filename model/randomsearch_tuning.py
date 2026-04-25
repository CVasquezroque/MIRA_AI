"""
Deeper model tuning with RandomizedSearchCV.

This script keeps the EDA-driven decisions, but makes the comparison richer:
- preprocessing options are tuned inside the pipeline to avoid leakage,
- models are tuned with RandomizedSearchCV, not GridSearchCV,
- LightGBM and a soft-voting ensemble are included when available,
- validation/test confusion matrices, classification reports, and ROC curves are saved.

The tuning step uses a stratified sample from the training months so it remains
practical on a 1M-row dataset. Final selected models are refit on the full
training period before validation/test evaluation.
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
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TARGET = "fraud_bool"
RESULTS_DIR = Path("model/results/tuning")

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

TUNE_TRAIN_MAX_ROWS = 90_000
TUNE_VALID_MAX_ROWS = 35_000

SENTINEL_TO_NA = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

MISSING_FLAG_COLUMNS = [
    "prev_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
]

OUTLIER_COLUMNS = [
    "proposed_credit_limit",
    "session_length_in_minutes",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
]

LOG_CANDIDATES = [
    "proposed_credit_limit",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "session_length_in_minutes",
]


class FraudFeatureBuilder(BaseEstimator, TransformerMixin):
    """EDA-driven feature options fitted only on the training fold."""

    def __init__(
        self,
        add_missing_flags=True,
        add_outlier_flags=True,
        add_log_features=True,
        skew_threshold=1.0,
        long_tail_ratio=5.0,
    ):
        self.add_missing_flags = add_missing_flags
        self.add_outlier_flags = add_outlier_flags
        self.add_log_features = add_log_features
        self.skew_threshold = skew_threshold
        self.long_tail_ratio = long_tail_ratio

    def fit(self, X, y=None):
        X_clean = self._replace_sentinels(X.copy())
        self.iqr_bounds_ = {}
        self.log_columns_ = []

        if self.add_outlier_flags:
            for column in OUTLIER_COLUMNS:
                if column not in X_clean.columns or X_clean[column].nunique(dropna=True) <= 2:
                    continue

                values = X_clean[column].dropna()
                if values.empty:
                    continue

                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    self.iqr_bounds_[column] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        if self.add_log_features:
            for column in LOG_CANDIDATES:
                if column not in X_clean.columns:
                    continue

                values = X_clean[column].dropna()
                if values.empty or values.min() < 0:
                    continue

                median = values.median()
                p99 = values.quantile(0.99)
                long_tail = median > 0 and p99 / median >= self.long_tail_ratio
                strong_skew = abs(values.skew()) >= self.skew_threshold

                if long_tail or strong_skew:
                    self.log_columns_.append(column)

        return self

    def transform(self, X):
        X_out = X.copy()

        if self.add_missing_flags:
            for column in MISSING_FLAG_COLUMNS:
                if column in X_out.columns:
                    X_out[column + "_was_missing"] = (X_out[column] == -1).astype(int)

        X_out = self._replace_sentinels(X_out)

        if self.add_outlier_flags:
            for column, (lower, upper) in self.iqr_bounds_.items():
                flag_name = column + "_is_iqr_outlier"
                X_out[flag_name] = ((X_out[column] < lower) | (X_out[column] > upper)).astype(int)

        if self.add_log_features:
            for column in self.log_columns_:
                X_out[column + "_log1p"] = np.log1p(X_out[column])

        return X_out

    def _replace_sentinels(self, X):
        for column in SENTINEL_TO_NA:
            if column in X.columns:
                X[column] = X[column].replace(-1, np.nan)
        return X


def categorical_columns(X):
    columns = []
    for column in X.columns:
        dtype_name = str(X[column].dtype)
        if dtype_name in ["object", "category", "str", "string"] or dtype_name.startswith("string["):
            columns.append(column)
    return columns


def numeric_columns(X):
    categorical = set(categorical_columns(X))
    return [column for column in X.columns if column not in categorical]


def split_before_preprocessing(data):
    if "month" in data.columns and data["month"].nunique() >= 4:
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
        return train, valid, test, train_months, valid_month, test_month

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
    return train.copy(), valid.copy(), test.copy(), None, None, None


def stratified_sample(data, max_rows):
    if len(data) <= max_rows:
        return data.copy()

    sample, _ = train_test_split(
        data,
        train_size=max_rows,
        stratify=data[TARGET],
        random_state=RANDOM_STATE,
    )
    return sample.copy()


def make_tuning_sample(train, train_months):
    if train_months is not None and len(train_months) >= 2:
        inner_valid_month = train_months[-1]
        inner_train_months = train_months[:-1]
        inner_train = train[train["month"].isin(inner_train_months)].copy()
        inner_valid = train[train["month"] == inner_valid_month].copy()
        print(f"RandomizedSearchCV inner train months: {inner_train_months}")
        print(f"RandomizedSearchCV inner validation month: {inner_valid_month}")
    else:
        inner_train, inner_valid = train_test_split(
            train,
            test_size=0.25,
            stratify=train[TARGET],
            random_state=RANDOM_STATE,
        )

    inner_train = stratified_sample(inner_train, TUNE_TRAIN_MAX_ROWS)
    inner_valid = stratified_sample(inner_valid, TUNE_VALID_MAX_ROWS)
    tuning_data = pd.concat([inner_train, inner_valid], axis=0).reset_index(drop=True)
    test_fold = np.r_[
        np.full(len(inner_train), -1, dtype=int),
        np.zeros(len(inner_valid), dtype=int),
    ]
    print(f"Tuning sample: train={len(inner_train):,}, validation={len(inner_valid):,}")
    return tuning_data, PredefinedSplit(test_fold)


def make_raw_features(frame, drop_columns):
    return frame.drop(columns=drop_columns, errors="ignore")


def build_preprocessor(scale_numeric):
    numeric_pipeline_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_pipeline_steps.append(("scaler", RobustScaler()))

    numeric_pipeline = Pipeline(steps=numeric_pipeline_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def make_pipeline(model, scale_numeric):
    return Pipeline(
        steps=[
            ("features", FraudFeatureBuilder()),
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def get_model_spaces(scale_pos_weight):
    spaces = {}

    spaces["Logistic Regression"] = {
        "estimator": make_pipeline(
            LogisticRegression(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=300,
                solver="lbfgs",
            ),
            scale_numeric=True,
        ),
        "params": {
            "features__add_missing_flags": [True, False],
            "features__add_outlier_flags": [True, False],
            "features__add_log_features": [True, False],
            "model__C": loguniform(0.01, 10),
        },
        "n_iter": 6,
    }

    spaces["Decision Tree"] = {
        "estimator": make_pipeline(
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            scale_numeric=False,
        ),
        "params": {
            "features__add_missing_flags": [True, False],
            "features__add_outlier_flags": [True, False],
            "features__add_log_features": [True, False],
            "model__max_depth": [4, 6, 8, 10, 12, None],
            "model__min_samples_leaf": randint(50, 500),
            "model__min_samples_split": randint(100, 1000),
            "model__ccp_alpha": [0.0, 0.00001, 0.0001, 0.001],
        },
        "n_iter": 8,
    }

    spaces["Random Forest"] = {
        "estimator": make_pipeline(
            RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale_numeric=False,
        ),
        "params": {
            "features__add_missing_flags": [True, False],
            "features__add_outlier_flags": [True, False],
            "features__add_log_features": [True, False],
            "model__n_estimators": randint(50, 111),
            "model__max_depth": [8, 10, 12, 16, None],
            "model__min_samples_leaf": randint(50, 350),
            "model__max_features": ["sqrt", "log2", 0.5],
        },
        "n_iter": 5,
    }

    spaces["Extra Trees"] = {
        "estimator": make_pipeline(
            ExtraTreesClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale_numeric=False,
        ),
        "params": {
            "features__add_missing_flags": [True, False],
            "features__add_outlier_flags": [True, False],
            "features__add_log_features": [True, False],
            "model__n_estimators": randint(50, 111),
            "model__max_depth": [8, 10, 12, 16, None],
            "model__min_samples_leaf": randint(50, 350),
            "model__max_features": ["sqrt", "log2", 0.5],
        },
        "n_iter": 5,
    }

    try:
        from xgboost import XGBClassifier

        spaces["XGBoost"] = {
            "estimator": make_pipeline(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                scale_numeric=False,
            ),
            "params": {
                "features__add_missing_flags": [True, False],
                "features__add_outlier_flags": [True, False],
                "features__add_log_features": [True, False],
                "model__n_estimators": randint(60, 161),
                "model__max_depth": randint(3, 7),
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__subsample": uniform(0.7, 0.3),
                "model__colsample_bytree": uniform(0.7, 0.3),
                "model__min_child_weight": randint(1, 10),
                "model__gamma": uniform(0.0, 3.0),
                "model__reg_lambda": loguniform(0.5, 10),
                "model__scale_pos_weight": [
                    scale_pos_weight * 0.75,
                    scale_pos_weight,
                    scale_pos_weight * 1.25,
                ],
            },
            "n_iter": 6,
        }
    except ImportError:
        print("XGBoost is not available; skipping it.")

    try:
        from lightgbm import LGBMClassifier

        spaces["LightGBM"] = {
            "estimator": make_pipeline(
                LGBMClassifier(
                    objective="binary",
                    random_state=RANDOM_STATE,
                    n_jobs=4,
                    verbose=-1,
                    force_col_wise=True,
                    device_type="cpu",
                ),
                scale_numeric=False,
            ),
            "params": {
                "features__add_missing_flags": [True, False],
                "features__add_outlier_flags": [True, False],
                "features__add_log_features": [True, False],
                "model__n_estimators": randint(50, 121),
                "model__num_leaves": randint(15, 50),
                "model__max_depth": [3, 5, 8, -1],
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__subsample": uniform(0.7, 0.3),
                "model__colsample_bytree": uniform(0.7, 0.3),
                "model__min_child_samples": randint(80, 300),
                "model__reg_lambda": loguniform(0.1, 10),
                "model__scale_pos_weight": [
                    scale_pos_weight * 0.75,
                    scale_pos_weight,
                    scale_pos_weight * 1.25,
                ],
            },
            "n_iter": 4,
        }
    except ImportError:
        print("LightGBM is not available; skipping it.")

    return spaces


def evaluate_scores(y_true, scores, threshold=0.50):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall_tpr": recall_score(y_true, predictions, zero_division=0),
        "fpr": fpr,
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def model_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def params_to_jsonable(params):
    clean = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean


def tune_single_model(name, space, X_tune, y_tune, cv):
    print(f"\nTuning {name} with RandomizedSearchCV...")
    search = RandomizedSearchCV(
        estimator=space["estimator"],
        param_distributions=space["params"],
        n_iter=space["n_iter"],
        scoring="average_precision",
        cv=cv,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        search.fit(X_tune, y_tune)

    print(f"Best tuning PR-AUC for {name}: {search.best_score_:.6f}")
    print(f"Best params for {name}: {search.best_params_}")
    return search


def save_cv_results(searches):
    rows = []
    for model_name, search in searches.items():
        results = pd.DataFrame(search.cv_results_)
        results["model"] = model_name
        rows.append(results)

    all_results = pd.concat(rows, axis=0, ignore_index=True)
    all_results.to_csv(RESULTS_DIR / "randomsearch_cv_results.csv", index=False)


def save_best_params(searches):
    best_params = {
        model_name: {
            "best_score_pr_auc": float(search.best_score_),
            "best_params": params_to_jsonable(search.best_params_),
        }
        for model_name, search in searches.items()
    }
    with open(RESULTS_DIR / "randomsearch_best_params.json", "w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=2)


def plot_roc_curves(score_rows, split_name):
    plt.figure(figsize=(8, 6))
    for row in score_rows:
        fpr, tpr, _ = roc_curve(row["y_true"], row["scores"])
        label = f"{row['model']} (AUC={row['roc_auc']:.3f})"
        plt.plot(fpr, tpr, linewidth=1.6, label=label)

    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    plt.title(f"ROC Curves - {split_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"roc_{split_name.lower()}.png", dpi=150)
    plt.close()


def plot_confusion_matrices(score_rows, split_name, threshold=0.50):
    cols = 3
    rows = int(np.ceil(len(score_rows) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for axis, row in zip(axes, score_rows):
        predictions = (row["scores"] >= threshold).astype(int)
        matrix = confusion_matrix(row["y_true"], predictions, labels=[0, 1])
        display = ConfusionMatrixDisplay(matrix, display_labels=["Legit", "Fraud"])
        display.plot(ax=axis, values_format="d", colorbar=False)
        axis.set_title(f"{row['model']}\nthreshold={threshold:.2f}")

    for axis in axes[len(score_rows) :]:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"confusion_matrices_{split_name.lower()}.png", dpi=150)
    plt.close()


def save_classification_reports(score_rows, split_name, threshold=0.50):
    report_rows = []
    matrix_rows = []

    for row in score_rows:
        predictions = (row["scores"] >= threshold).astype(int)
        report = classification_report(
            row["y_true"],
            predictions,
            labels=[0, 1],
            target_names=["legitimate", "fraud"],
            output_dict=True,
            zero_division=0,
        )
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                report_rows.append(
                    {
                        "split": split_name,
                        "model": row["model"],
                        "class_or_average": label,
                        **metrics,
                    }
                )

        tn, fp, fn, tp = confusion_matrix(row["y_true"], predictions, labels=[0, 1]).ravel()
        matrix_rows.append(
            {
                "split": split_name,
                "model": row["model"],
                "threshold": threshold,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

    pd.DataFrame(report_rows).to_csv(
        RESULTS_DIR / f"classification_report_{split_name.lower()}.csv",
        index=False,
    )
    pd.DataFrame(matrix_rows).to_csv(
        RESULTS_DIR / f"confusion_matrices_{split_name.lower()}.csv",
        index=False,
    )


def make_voting_classifier(best_estimators):
    preferred = []
    for model_name in ["Logistic Regression", "LightGBM", "XGBoost", "Random Forest", "Extra Trees"]:
        if model_name in best_estimators:
            preferred.append((model_name.lower().replace(" ", "_"), clone(best_estimators[model_name])))
        if len(preferred) == 3:
            break

    if len(preferred) < 2:
        return None

    return VotingClassifier(estimators=preferred, voting="soft", n_jobs=-1)


def tune_voting_classifier(voting, X_tune, y_tune, cv):
    weight_options = [
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
    ]
    if len(voting.estimators) == 2:
        weight_options = [(1, 1), (2, 1), (1, 2), (3, 1), (1, 3)]

    print("\nTuning Voting Classifier weights with RandomizedSearchCV...")
    search = RandomizedSearchCV(
        estimator=voting,
        param_distributions={"weights": weight_options},
        n_iter=min(4, len(weight_options)),
        scoring="average_precision",
        cv=cv,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_tune, y_tune)
    print(f"Best tuning PR-AUC for Voting Classifier: {search.best_score_:.6f}")
    print(f"Best params for Voting Classifier: {search.best_params_}")
    return search


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    data = pd.read_csv("data_banca/Base.csv")
    print(f"Loaded data: {data.shape[0]:,} rows, {data.shape[1]:,} columns")

    train, valid, test, train_months, valid_month, test_month = split_before_preprocessing(data)
    print(f"Split sizes: train={len(train):,}, valid={len(valid):,}, test={len(test):,}")

    unusable_columns = [
        column
        for column in train.columns
        if column != TARGET and train[column].nunique(dropna=False) <= 1
    ]
    drop_columns = [TARGET] + unusable_columns
    if "month" in data.columns:
        drop_columns.append("month")

    X_train = make_raw_features(train, drop_columns)
    y_train = train[TARGET].copy()
    X_valid = make_raw_features(valid, drop_columns)
    y_valid = valid[TARGET].copy()
    X_test = make_raw_features(test, drop_columns)
    y_test = test[TARGET].copy()

    tuning_data, predefined_cv = make_tuning_sample(train, train_months)
    X_tune = make_raw_features(tuning_data, drop_columns)
    y_tune = tuning_data[TARGET].copy()

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / max(positive_count, 1)
    print(f"scale_pos_weight from full training split: {scale_pos_weight:.3f}")

    model_spaces = get_model_spaces(scale_pos_weight)
    searches = {}
    best_estimators_from_search = {}

    for model_name, space in model_spaces.items():
        search = tune_single_model(model_name, space, X_tune, y_tune, predefined_cv)
        searches[model_name] = search
        best_estimators_from_search[model_name] = search.best_estimator_

    voting = make_voting_classifier(best_estimators_from_search)
    if voting is not None:
        voting_search = tune_voting_classifier(voting, X_tune, y_tune, predefined_cv)
        searches["Voting Classifier"] = voting_search
        best_estimators_from_search["Voting Classifier"] = voting_search.best_estimator_

    save_cv_results(searches)
    save_best_params(searches)

    validation_rows = []
    test_rows = []
    validation_score_rows = []
    test_score_rows = []

    for model_name, estimator in best_estimators_from_search.items():
        print(f"\nRefitting best {model_name} on full training period...")
        final_model = clone(estimator)
        final_model.fit(X_train, y_train)

        valid_scores = model_scores(final_model, X_valid)
        test_scores = model_scores(final_model, X_test)

        valid_metrics = evaluate_scores(y_valid, valid_scores)
        valid_metrics["model"] = model_name
        validation_rows.append(valid_metrics)

        test_metrics = evaluate_scores(y_test, test_scores)
        test_metrics["model"] = model_name
        test_rows.append(test_metrics)

        validation_score_rows.append(
            {
                "model": model_name,
                "y_true": y_valid,
                "scores": valid_scores,
                "roc_auc": valid_metrics["roc_auc"],
            }
        )
        test_score_rows.append(
            {
                "model": model_name,
                "y_true": y_test,
                "scores": test_scores,
                "roc_auc": test_metrics["roc_auc"],
            }
        )

    validation_results = pd.DataFrame(validation_rows).sort_values("pr_auc", ascending=False)
    test_results = pd.DataFrame(test_rows).sort_values("pr_auc", ascending=False)
    validation_results.to_csv(RESULTS_DIR / "tuned_validation_metrics.csv", index=False)
    test_results.to_csv(RESULTS_DIR / "tuned_test_metrics.csv", index=False)

    plot_roc_curves(validation_score_rows, "Validation")
    plot_roc_curves(test_score_rows, "Test")
    plot_confusion_matrices(validation_score_rows, "Validation")
    plot_confusion_matrices(test_score_rows, "Test")
    save_classification_reports(validation_score_rows, "Validation")
    save_classification_reports(test_score_rows, "Test")

    print("\nTuned validation metrics at threshold 0.50:")
    print(validation_results.to_string(index=False))
    print("\nTuned test metrics at threshold 0.50:")
    print(test_results.to_string(index=False))
    print(f"\nSaved tuning artifacts in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
