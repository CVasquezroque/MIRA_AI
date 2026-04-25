"""
Advanced fraud modeling experiments.

This script extends the previous EDA/modeling work without replacing it. It:
- creates plots that justify ratio and interaction features,
- compares one-hot encoding with temporal target/frequency encoding,
- tries CatBoost,
- compares standard log-loss training with a focal-style XGBoost objective,
- tries a simple temporal stacking ensemble,
- saves SHAP and permutation-importance summaries for the best tree model.

The run is intentionally an experiment, not a production pipeline.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


RANDOM_STATE = 42
TARGET = "fraud_bool"
RESULTS_DIR = Path("model/results/advanced")
FIGURES_DIR = RESULTS_DIR / "figures"
TRAIN_SAMPLE_MAX_ROWS = 250_000
SHAP_SAMPLE_ROWS = 2_000
PERMUTATION_SAMPLE_ROWS = 8_000

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

RATIO_FEATURES = {
    "velocity_6h_to_24h": ("velocity_6h", "velocity_24h"),
    "velocity_24h_to_4w": ("velocity_24h", "velocity_4w"),
    "zip_count_4w_to_velocity_4w": ("zip_count_4w", "velocity_4w"),
    "dob_emails_to_zip_count": ("date_of_birth_distinct_emails_4w", "zip_count_4w"),
    "branch_count_to_zip_count": ("bank_branch_count_8w", "zip_count_4w"),
    "credit_limit_to_income": ("proposed_credit_limit", "income"),
}

INTERACTION_COLUMNS = [
    "device_os__source",
    "email_free__source",
    "phone_valid_combo",
    "payment_type__credit_limit_bin",
    "email_similarity_bin__email_free",
    "session_length_bin__source",
]

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


def safe_divide(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


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


def stratified_training_sample(train):
    if len(train) <= TRAIN_SAMPLE_MAX_ROWS:
        return train.copy()

    sample, _ = train_test_split(
        train,
        train_size=TRAIN_SAMPLE_MAX_ROWS,
        stratify=train[TARGET],
        random_state=RANDOM_STATE,
    )
    return sample.copy()


def make_raw_features(frame, drop_columns):
    return frame.drop(columns=drop_columns, errors="ignore")


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


class AdvancedFeatureBuilder(BaseEstimator, TransformerMixin):
    """Feature additions motivated by the EDA and the BAF fraud setting."""

    def __init__(
        self,
        add_missing_flags=True,
        add_outlier_flags=True,
        add_log_features=True,
        add_ratio_features=True,
        add_interaction_features=True,
    ):
        self.add_missing_flags = add_missing_flags
        self.add_outlier_flags = add_outlier_flags
        self.add_log_features = add_log_features
        self.add_ratio_features = add_ratio_features
        self.add_interaction_features = add_interaction_features

    def fit(self, X, y=None):
        X_clean = self._replace_sentinels(X.copy())
        self.iqr_bounds_ = {}
        self.log_columns_ = []
        self.bin_edges_ = {}

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
                long_tail = median > 0 and p99 / median >= 5.0
                strong_skew = abs(values.skew()) >= 1.0
                if long_tail or strong_skew:
                    self.log_columns_.append(column)

        if self.add_interaction_features:
            self.bin_edges_["proposed_credit_limit"] = self._quantile_edges(
                X_clean.get("proposed_credit_limit")
            )
            self.bin_edges_["name_email_similarity"] = self._quantile_edges(
                X_clean.get("name_email_similarity")
            )
            self.bin_edges_["session_length_in_minutes"] = self._quantile_edges(
                X_clean.get("session_length_in_minutes")
            )

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
                X_out[column + "_is_iqr_outlier"] = (
                    (X_out[column] < lower) | (X_out[column] > upper)
                ).astype(int)

        if self.add_log_features:
            for column in self.log_columns_:
                X_out[column + "_log1p"] = np.log1p(X_out[column])

        if self.add_ratio_features:
            for feature_name, (numerator, denominator) in RATIO_FEATURES.items():
                if numerator in X_out.columns and denominator in X_out.columns:
                    X_out[feature_name] = safe_divide(X_out[numerator], X_out[denominator])

        if self.add_interaction_features:
            X_out = self._add_interactions(X_out)

        return X_out

    def _replace_sentinels(self, X):
        for column in SENTINEL_TO_NA:
            if column in X.columns:
                X[column] = X[column].replace(-1, np.nan)
        return X

    def _quantile_edges(self, series):
        if series is None:
            return None
        values = series.dropna()
        if values.nunique() < 2:
            return None
        quantiles = values.quantile([0, 0.25, 0.50, 0.75, 1.0]).to_numpy()
        quantiles = np.unique(quantiles)
        if len(quantiles) < 3:
            return None
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        return quantiles

    def _bin(self, X, column, label):
        edges = self.bin_edges_.get(column)
        if edges is None or column not in X.columns:
            return pd.Series("missing", index=X.index, dtype="object")
        binned = pd.cut(X[column], bins=edges, include_lowest=True, duplicates="drop")
        return binned.astype("string").fillna("missing").astype(str)

    def _add_interactions(self, X):
        if {"device_os", "source"}.issubset(X.columns):
            X["device_os__source"] = X["device_os"].astype(str) + "__" + X["source"].astype(str)

        if {"email_is_free", "source"}.issubset(X.columns):
            X["email_free__source"] = (
                X["email_is_free"].astype(str) + "__" + X["source"].astype(str)
            )

        if {"phone_home_valid", "phone_mobile_valid"}.issubset(X.columns):
            X["phone_valid_combo"] = (
                X["phone_home_valid"].astype(str) + "__" + X["phone_mobile_valid"].astype(str)
            )

        if {"payment_type", "proposed_credit_limit"}.issubset(X.columns):
            limit_bin = self._bin(X, "proposed_credit_limit", "credit_limit_bin")
            X["payment_type__credit_limit_bin"] = X["payment_type"].astype(str) + "__" + limit_bin

        if {"name_email_similarity", "email_is_free"}.issubset(X.columns):
            similarity_bin = self._bin(X, "name_email_similarity", "email_similarity_bin")
            X["email_similarity_bin__email_free"] = (
                similarity_bin + "__" + X["email_is_free"].astype(str)
            )

        if {"session_length_in_minutes", "source"}.issubset(X.columns):
            session_bin = self._bin(X, "session_length_in_minutes", "session_length_bin")
            X["session_length_bin__source"] = session_bin + "__" + X["source"].astype(str)

        return X


class TemporalTargetFrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Target/frequency encoding with temporal fit_transform.

    During model training, each month receives encodings learned only from earlier
    months. During validation/test transform, mappings learned from the full
    training period are used. This reduces target leakage compared with plain
    target encoding on the same rows being fitted.
    """

    def __init__(self, smoothing=50.0, min_samples_leaf=20, month_column="month"):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.month_column = month_column

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y, index=X.index)
        self.global_mean_ = float(y.mean())
        self.category_columns_ = categorical_columns(X)
        self.columns_to_drop_ = list(self.category_columns_)
        if self.month_column in X.columns:
            self.columns_to_drop_.append(self.month_column)
        self.mappings_ = {}
        self.freq_mappings_ = {}

        for column in self.category_columns_:
            category = X[column].fillna("Unknown").astype(str)
            stats = (
                pd.DataFrame({"category": category, "target": y})
                .groupby("category")["target"]
                .agg(["mean", "count"])
            )
            smooth = self._smooth(stats["mean"], stats["count"])
            self.mappings_[column] = smooth.to_dict()
            self.freq_mappings_[column] = (stats["count"] / len(X)).to_dict()
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        X = X.copy()
        y = pd.Series(y, index=X.index)
        encoded = X.copy()

        if self.month_column in X.columns and X[self.month_column].nunique(dropna=True) > 1:
            months = sorted(X[self.month_column].dropna().unique())
            for column in self.category_columns_:
                encoded[column + "_target_mean"] = self.global_mean_
                encoded[column + "_frequency"] = 0.0

                for month in months:
                    current_mask = X[self.month_column] == month
                    previous_mask = X[self.month_column] < month
                    if previous_mask.sum() < self.min_samples_leaf:
                        continue

                    previous_category = X.loc[previous_mask, column].fillna("Unknown").astype(str)
                    current_category = X.loc[current_mask, column].fillna("Unknown").astype(str)
                    stats = (
                        pd.DataFrame(
                            {
                                "category": previous_category,
                                "target": y.loc[previous_mask],
                            }
                        )
                        .groupby("category")["target"]
                        .agg(["mean", "count"])
                    )
                    smooth = self._smooth(stats["mean"], stats["count"])
                    freq = stats["count"] / previous_mask.sum()
                    encoded.loc[current_mask, column + "_target_mean"] = (
                        current_category.map(smooth).fillna(self.global_mean_).to_numpy()
                    )
                    encoded.loc[current_mask, column + "_frequency"] = (
                        current_category.map(freq).fillna(0.0).to_numpy()
                    )
        else:
            encoded = self.transform(X)

        return encoded.drop(columns=self.columns_to_drop_, errors="ignore")

    def transform(self, X):
        X = X.copy()
        encoded = X.copy()
        for column in self.category_columns_:
            category = X[column].fillna("Unknown").astype(str)
            encoded[column + "_target_mean"] = (
                category.map(self.mappings_[column]).fillna(self.global_mean_).astype(float)
            )
            encoded[column + "_frequency"] = (
                category.map(self.freq_mappings_[column]).fillna(0.0).astype(float)
            )
        return encoded.drop(columns=self.columns_to_drop_, errors="ignore")

    def _smooth(self, means, counts):
        weight = 1 / (1 + np.exp(-(counts - self.min_samples_leaf) / self.smoothing))
        return self.global_mean_ * (1 - weight) + means * weight


class DataFrameMedianImputer(BaseEstimator, TransformerMixin):
    """Median imputer that preserves column names for SHAP/permutation importance."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
        self.columns_ = X.columns.tolist()
        self.medians_ = X.median(numeric_only=True)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy().replace([np.inf, -np.inf], np.nan)
        X = X.reindex(columns=self.columns_)
        return X.fillna(self.medians_).fillna(0.0)


class DataFrameRobustScaler(BaseEstimator, TransformerMixin):
    """RobustScaler wrapper that returns a DataFrame."""

    def __init__(self):
        self.scaler = RobustScaler()

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = X.columns.tolist()
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).reindex(columns=self.columns_)
        return pd.DataFrame(self.scaler.transform(X), columns=self.columns_, index=X.index)


def make_feature_evidence(train):
    print("Creating feature evidence plots...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    builder = AdvancedFeatureBuilder(
        add_missing_flags=False,
        add_outlier_flags=False,
        add_log_features=False,
        add_ratio_features=True,
        add_interaction_features=True,
    )
    X_train = train.drop(columns=[TARGET], errors="ignore")
    X_plot = builder.fit_transform(X_train)
    plot_data = X_plot.copy()
    plot_data[TARGET] = train[TARGET].to_numpy()

    ratio_rows = []
    ratio_features = [column for column in RATIO_FEATURES if column in plot_data.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes = axes.reshape(-1)
    base_rate = plot_data[TARGET].mean() * 100

    for axis, column in zip(axes, ratio_features):
        values = plot_data[[column, TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
        if values[column].nunique() < 4:
            axis.axis("off")
            continue
        values["bin"] = pd.qcut(values[column], q=10, duplicates="drop")
        summary = (
            values.groupby("bin", observed=True)[TARGET]
            .agg(fraud_rate="mean", count="size")
            .reset_index()
        )
        summary["feature"] = column
        summary["bin_label"] = summary["bin"].astype(str)
        ratio_rows.append(summary[["feature", "bin_label", "fraud_rate", "count"]])

        x = np.arange(len(summary))
        axis.bar(x, summary["fraud_rate"] * 100, color="#3b82f6", alpha=0.78)
        axis.axhline(base_rate, color="#111827", linestyle="--", linewidth=1)
        axis.set_title(column)
        axis.set_xlabel("Quantile bin")
        axis.set_ylabel("Fraud rate (%)")
        axis.set_xticks(x)
        axis.set_xticklabels([str(i + 1) for i in x])

    for axis in axes[len(ratio_features) :]:
        axis.axis("off")

    fig.suptitle("Fraud Rate by Ratio Feature Quantile - Training Months", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "strategy_1_ratio_fraud_rates.png", dpi=150)
    plt.close()

    if ratio_rows:
        pd.concat(ratio_rows, ignore_index=True).to_csv(
            RESULTS_DIR / "strategy_1_ratio_bin_fraud_rates.csv",
            index=False,
        )

    interaction_specs = [
        ("device_os", "source", "Device OS x Source", "Source", "Device OS"),
        ("payment_type", "proposed_credit_limit", "Payment Type x Credit Limit", "Credit limit bin", "Payment type"),
        ("email_is_free", "name_email_similarity", "Free Email x Name/Email Similarity", "Name/email similarity bin", "Free email"),
        ("phone_home_valid", "phone_mobile_valid", "Home Phone x Mobile Phone Valid", "Mobile phone valid", "Home phone valid"),
        ("source", "session_length_in_minutes", "Source x Session Length", "Session length bin", "Source"),
        ("device_os", "email_is_free", "Device OS x Free Email", "Free email", "Device OS"),
    ]

    def short_tick_label(value):
        text = str(value)
        if text.lower() in ["nan", "missing"]:
            return "missing"
        if (text.startswith("(") or text.startswith("[")) and "," in text:
            left, right = text.strip("()[]").split(",", 1)
            try:
                left_number = float(left)
                right_number = float(right)
            except ValueError:
                return text
            return f"{left_number:.3g}-{right_number:.3g}"
        return text

    heatmap_rows = []
    fig, axes = plt.subplots(3, 2, figsize=(18, 20), constrained_layout=True)
    axes = axes.reshape(-1)
    for axis, (row_col, col_col, title, x_label, y_label) in zip(axes, interaction_specs):
        if row_col not in plot_data.columns or col_col not in plot_data.columns:
            axis.axis("off")
            continue

        temp = plot_data[[row_col, col_col, TARGET]].copy()
        if temp[col_col].dtype.kind in "if" and temp[col_col].nunique(dropna=True) > 6:
            temp[col_col] = pd.qcut(temp[col_col], q=4, duplicates="drop").astype(str)
        else:
            temp[col_col] = temp[col_col].astype(str)
        temp[row_col] = temp[row_col].astype(str)

        rate = temp.pivot_table(
            index=row_col,
            columns=col_col,
            values=TARGET,
            aggfunc="mean",
            observed=True,
        )
        counts = temp.pivot_table(
            index=row_col,
            columns=col_col,
            values=TARGET,
            aggfunc="size",
            observed=True,
        )
        for idx in rate.index:
            for col in rate.columns:
                if pd.notna(rate.loc[idx, col]):
                    heatmap_rows.append(
                        {
                            "interaction": title,
                            "row_value": idx,
                            "column_value": col,
                            "fraud_rate": rate.loc[idx, col],
                            "count": counts.loc[idx, col],
                        }
                    )

        sns.heatmap(
            rate * 100,
            ax=axis,
            cmap="YlOrRd",
            linewidths=0.4,
            cbar=True,
            cbar_kws={"shrink": 0.78, "label": "Fraud rate (%)"},
        )
        axis.set_title(title, fontsize=14, pad=12)
        axis.set_xlabel(x_label, fontsize=12, labelpad=10)
        axis.set_ylabel(y_label, fontsize=12, labelpad=10)
        axis.set_xticklabels(
            [short_tick_label(label.get_text()) for label in axis.get_xticklabels()],
            rotation=35,
            ha="right",
            fontsize=10,
        )
        axis.set_yticklabels(
            [short_tick_label(label.get_text()) for label in axis.get_yticklabels()],
            rotation=0,
            fontsize=10,
        )

    fig.suptitle(
        "Fraud Rate Heatmaps for Candidate Interaction Features - Training Months",
        fontsize=18,
    )
    plt.savefig(FIGURES_DIR / "strategy_2_interaction_heatmaps.png", dpi=170, bbox_inches="tight")
    plt.close()

    pd.DataFrame(heatmap_rows).to_csv(
        RESULTS_DIR / "strategy_2_interaction_fraud_rates.csv",
        index=False,
    )


def make_onehot_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    return Pipeline(
        [
            ("features", AdvancedFeatureBuilder()),
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def make_target_frequency_pipeline(model, scale_numeric=False):
    steps = [
        ("features", AdvancedFeatureBuilder()),
        ("target_frequency", TemporalTargetFrequencyEncoder()),
        ("imputer", DataFrameMedianImputer()),
    ]
    if scale_numeric:
        steps.append(("scaler", DataFrameRobustScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def focal_binary_objective(alpha=0.75, gamma=2.0):
    """XGBoost custom objective for binary focal-style loss."""

    def objective(y_true, predt):
        p = np.clip(expit(predt), 1e-7, 1 - 1e-7)
        y = y_true.astype(float)
        q = 1 - p

        grad_pos = alpha * (gamma * p * (q**gamma) * np.log(p) - q ** (gamma + 1))
        grad_neg = (1 - alpha) * (
            -gamma * (p**gamma) * q * np.log(q) + p ** (gamma + 1)
        )
        grad = np.where(y == 1, grad_pos, grad_neg)

        hess_pos = alpha * (
            gamma * (q**gamma) * np.log(p)
            - (gamma**2) * p * (q ** (gamma - 1)) * np.log(p)
            + (2 * gamma + 1) * (q**gamma)
        ) * p * q
        hess_neg = (1 - alpha) * (
            -(gamma**2) * (p ** (gamma - 1)) * q * np.log(q)
            + gamma * (p**gamma) * np.log(q)
            + (2 * gamma + 1) * (p**gamma)
        ) * p * q
        hess = np.where(y == 1, hess_pos, hess_neg)
        return grad, np.maximum(hess, 1e-7)

    return objective


def make_xgboost(standard=True, scale_pos_weight=1.0):
    from xgboost import XGBClassifier

    objective = "binary:logistic" if standard else focal_binary_objective(alpha=0.75, gamma=2.0)
    return XGBClassifier(
        objective=objective,
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=180,
        max_depth=4,
        learning_rate=0.055,
        subsample=0.88,
        colsample_bytree=0.76,
        min_child_weight=3,
        gamma=2.34,
        reg_lambda=0.60,
        scale_pos_weight=scale_pos_weight if standard else 1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_lightgbm(scale_pos_weight):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="binary",
        n_estimators=160,
        learning_rate=0.04,
        max_depth=8,
        num_leaves=17,
        min_child_samples=220,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_lambda=5.25,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbose=-1,
        force_col_wise=True,
        device_type="cpu",
    )


def make_logistic():
    return LogisticRegression(
        C=0.03,
        class_weight="balanced",
        max_iter=500,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )


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


def threshold_at_fpr(y_true, scores, max_fpr=0.05):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid = np.where(fpr <= max_fpr)[0]
    if len(valid) == 0:
        return 0.50
    best_index = valid[np.argmax(tpr[valid])]
    return float(thresholds[best_index])


def model_scores(model, X):
    return model.predict_proba(X)[:, 1]


def fit_catboost(X_train, y_train, X_valid, y_valid, scale_pos_weight):
    from catboost import CatBoostClassifier

    builder = AdvancedFeatureBuilder()
    X_train_cb = builder.fit_transform(X_train, y_train).drop(columns=["month"], errors="ignore")
    X_valid_cb = builder.transform(X_valid).drop(columns=["month"], errors="ignore")

    cat_cols = categorical_columns(X_train_cb)
    for column in cat_cols:
        X_train_cb[column] = X_train_cb[column].fillna("Unknown").astype(str)
        X_valid_cb[column] = X_valid_cb[column].fillna("Unknown").astype(str)

    model = CatBoostClassifier(
        iterations=350,
        depth=6,
        learning_rate=0.055,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights=None,
        scale_pos_weight=scale_pos_weight,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        X_train_cb,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_valid_cb, y_valid),
        use_best_model=True,
        early_stopping_rounds=50,
    )

    return {
        "model": model,
        "builder": builder,
        "cat_cols": cat_cols,
    }


def catboost_scores(fitted, X):
    X_cb = fitted["builder"].transform(X).drop(columns=["month"], errors="ignore")
    for column in fitted["cat_cols"]:
        if column in X_cb.columns:
            X_cb[column] = X_cb[column].fillna("Unknown").astype(str)
    return fitted["model"].predict_proba(X_cb)[:, 1]


def make_stacking_pipeline(scale_pos_weight):
    estimators = [
        ("xgb", make_target_frequency_pipeline(make_xgboost(standard=True, scale_pos_weight=scale_pos_weight))),
        ("lgbm", make_target_frequency_pipeline(make_lightgbm(scale_pos_weight=scale_pos_weight))),
        ("lr", make_target_frequency_pipeline(make_logistic(), scale_numeric=True)),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=300),
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
    )
    return stack


def run_experiments(X_train, y_train, X_valid, y_valid, X_test, y_test, scale_pos_weight):
    experiments = [
        {
            "name": "XGBoost one-hot + advanced features + standard logloss",
            "type": "pipeline",
            "model": make_onehot_pipeline(make_xgboost(standard=True, scale_pos_weight=scale_pos_weight)),
        },
        {
            "name": "XGBoost target/frequency + advanced features + standard logloss",
            "type": "pipeline",
            "model": make_target_frequency_pipeline(
                make_xgboost(standard=True, scale_pos_weight=scale_pos_weight)
            ),
        },
        {
            "name": "XGBoost target/frequency + advanced features + focal-style loss",
            "type": "pipeline",
            "model": make_target_frequency_pipeline(make_xgboost(standard=False)),
        },
        {
            "name": "LightGBM target/frequency + advanced features + standard logloss",
            "type": "pipeline",
            "model": make_target_frequency_pipeline(make_lightgbm(scale_pos_weight=scale_pos_weight)),
        },
        {
            "name": "Logistic target/frequency + advanced features + standard logloss",
            "type": "pipeline",
            "model": make_target_frequency_pipeline(make_logistic(), scale_numeric=True),
        },
        {
            "name": "CatBoost native categoricals + advanced features + standard logloss",
            "type": "catboost",
            "model": None,
        },
        {
            "name": "Stacking XGB + LGBM + Logistic target/frequency",
            "type": "pipeline",
            "model": make_stacking_pipeline(scale_pos_weight),
        },
    ]

    validation_rows = []
    test_rows = []
    score_rows = []
    fitted_models = {}

    for index, experiment in enumerate(experiments, start=1):
        print(f"\n[{index}/{len(experiments)}] Training {experiment['name']}...")
        if experiment["type"] == "catboost":
            fitted = fit_catboost(X_train, y_train, X_valid, y_valid, scale_pos_weight)
            valid_scores = catboost_scores(fitted, X_valid)
            test_scores = catboost_scores(fitted, X_test)
        else:
            fitted = clone(experiment["model"])
            fitted.fit(X_train, y_train)
            valid_scores = model_scores(fitted, X_valid)
            test_scores = model_scores(fitted, X_test)

        valid_metrics = evaluate_scores(y_valid, valid_scores)
        valid_metrics["model"] = experiment["name"]
        validation_rows.append(valid_metrics)

        chosen_threshold = threshold_at_fpr(y_valid, valid_scores, max_fpr=0.05)
        test_metrics = evaluate_scores(y_test, test_scores)
        test_metrics["model"] = experiment["name"]
        test_metrics["validation_threshold_for_5pct_fpr"] = chosen_threshold
        test_at_fpr = evaluate_scores(y_test, test_scores, threshold=chosen_threshold)
        for key, value in test_at_fpr.items():
            test_metrics["test_at_valid_5pct_fpr_" + key] = value
        test_rows.append(test_metrics)

        score_rows.append(
            {
                "model": experiment["name"],
                "y_valid": y_valid,
                "valid_scores": valid_scores,
                "y_test": y_test,
                "test_scores": test_scores,
                "valid_pr_auc": valid_metrics["pr_auc"],
                "valid_roc_auc": valid_metrics["roc_auc"],
            }
        )
        fitted_models[experiment["name"]] = fitted

    validation_results = pd.DataFrame(validation_rows).sort_values("pr_auc", ascending=False)
    test_results = pd.DataFrame(test_rows).sort_values("pr_auc", ascending=False)
    validation_results.to_csv(RESULTS_DIR / "advanced_validation_metrics.csv", index=False)
    test_results.to_csv(RESULTS_DIR / "advanced_test_metrics.csv", index=False)
    plot_model_curves(score_rows)
    return validation_results, test_results, score_rows, fitted_models


def plot_model_curves(score_rows):
    plt.figure(figsize=(8, 6))
    for row in sorted(score_rows, key=lambda item: item["valid_pr_auc"], reverse=True):
        fpr, tpr, _ = roc_curve(row["y_test"], row["test_scores"])
        plt.plot(fpr, tpr, linewidth=1.5, label=f"{row['model'][:32]} AUC={roc_auc_score(row['y_test'], row['test_scores']):.3f}")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    plt.title("Advanced Models - Test ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "advanced_test_roc.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    for row in sorted(score_rows, key=lambda item: item["valid_pr_auc"], reverse=True):
        precision, recall, _ = precision_recall_curve(row["y_test"], row["test_scores"])
        pr_auc = average_precision_score(row["y_test"], row["test_scores"])
        plt.plot(recall, precision, linewidth=1.5, label=f"{row['model'][:32]} PR={pr_auc:.3f}")
    plt.title("Advanced Models - Test Precision Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "advanced_test_precision_recall.png", dpi=150)
    plt.close()


def save_explainability(best_model_name, fitted_model, X_valid, y_valid):
    if not isinstance(fitted_model, Pipeline) or "target/frequency" not in best_model_name:
        print("Skipping SHAP/permutation plots because the selected model is not the target/frequency tree pipeline.")
        return

    print(f"Creating SHAP and permutation-importance plots for {best_model_name}...")
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(SHAP_SAMPLE_ROWS, len(X_valid))
    shap_indices = rng.choice(len(X_valid), size=sample_size, replace=False)
    X_sample = X_valid.iloc[shap_indices].copy()

    pre_model = fitted_model[:-1]
    model = fitted_model[-1]
    X_prepared = pre_model.transform(X_sample)

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_prepared)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        shap_abs = np.abs(shap_values).mean(axis=0)
        shap_summary = (
            pd.DataFrame({"feature": X_prepared.columns, "mean_abs_shap": shap_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(25)
        )
        shap_summary.to_csv(RESULTS_DIR / "shap_top_features.csv", index=False)

        plt.figure(figsize=(8, 7))
        sns.barplot(data=shap_summary, y="feature", x="mean_abs_shap", color="#2563eb")
        plt.title("Top SHAP Contributions - Validation Sample")
        plt.xlabel("Mean absolute SHAP value")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_top_features.png", dpi=150)
        plt.close()
    except Exception as exc:
        print(f"SHAP failed, continuing with permutation importance only: {exc}")

    perm_size = min(PERMUTATION_SAMPLE_ROWS, len(X_valid))
    perm_indices = rng.choice(len(X_valid), size=perm_size, replace=False)
    X_perm = X_valid.iloc[perm_indices].copy()
    y_perm = y_valid.iloc[perm_indices].copy()

    permutation = permutation_importance(
        fitted_model,
        X_perm,
        y_perm,
        scoring="average_precision",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    permutation_summary = (
        pd.DataFrame(
            {
                "feature": X_perm.columns,
                "importance_mean": permutation.importances_mean,
                "importance_std": permutation.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .head(25)
    )
    permutation_summary.to_csv(RESULTS_DIR / "permutation_importance_raw_features.csv", index=False)

    plt.figure(figsize=(8, 7))
    sns.barplot(data=permutation_summary, y="feature", x="importance_mean", color="#16a34a")
    plt.title("Permutation Importance on Raw Validation Features")
    plt.xlabel("Drop in validation PR-AUC")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "permutation_importance_raw_features.png", dpi=150)
    plt.close()


def write_report(validation_results, test_results, train_sample_rows, train_fraud_rate):
    best_validation = validation_results.iloc[0]
    best_test = test_results.iloc[0]
    report = f"""# Advanced Feature and Modeling Experiments

This iteration extends the previous EDA-driven pipeline. It does not replace the earlier
baseline/tuning results.

## Why Strategy 1: Ratio Features

The ratio plots in `figures/strategy_1_ratio_fraud_rates.png` compare fraud rates across
training-month quantiles. They are meant to check whether relative behavior has signal,
not to claim causality. The most useful candidates are ratios between short-window and
long-window activity, and ratios that normalize local concentration signals.

Examples tested:

- `velocity_6h / velocity_24h`
- `velocity_24h / velocity_4w`
- `zip_count_4w / velocity_4w`
- `date_of_birth_distinct_emails_4w / zip_count_4w`
- `bank_branch_count_8w / zip_count_4w`
- `proposed_credit_limit / income`

## Why Strategy 2: Interaction Features

The heatmaps in `figures/strategy_2_interaction_heatmaps.png` show fraud-rate variation
for pairs of features that are plausible in account-opening fraud: device/channel,
email/name consistency, phone validation, session behavior, and requested credit limit.
These are association checks from EDA, not causal claims.

Examples tested:

- `device_os x source`
- `email_is_free x source`
- `phone_home_valid x phone_mobile_valid`
- `payment_type x proposed_credit_limit_bin`
- `name_email_similarity_bin x email_is_free`
- `session_length_bin x source`

## Strategy 3: More Robust Target/Frequency Encoding

The new encoder adds two columns per categorical feature:

- a smoothed historical fraud-rate encoding,
- a category-frequency encoding.

To reduce leakage, training rows are encoded month-by-month using only earlier months
when `month` is available. Validation and test use mappings fitted on the training period.
This is stricter than fitting target encoding once on all training rows and then letting
each row carry information from its own label.

## Standard Objective vs Focal-Style Objective

Most baseline models use a standard binary log-loss objective. In plain language, the
model is trained to assign good probabilities overall. Class weights or
`scale_pos_weight` can make fraud mistakes more expensive, but the loss is still the
standard log-loss shape.

The focal-style XGBoost objective changes that training pressure: easier examples receive
less weight, while harder examples receive more attention. This can improve minority-class
capture, but it can also hurt calibration or increase false positives, so it must be
compared on validation/test metrics instead of assumed better.

## Run Setup

- Training sample: {train_sample_rows:,} rows from chronological training months.
- Training-sample fraud rate: {train_fraud_rate:.4%}.
- Validation and test months were not resampled.
- Step 7 anomaly scores and Step 8 recency weighting were intentionally left for the next iteration.

## Best Results

Best validation PR-AUC:

- Model: `{best_validation['model']}`
- PR-AUC: {best_validation['pr_auc']:.6f}
- ROC-AUC: {best_validation['roc_auc']:.6f}
- Precision @ 0.50: {best_validation['precision']:.6f}
- Recall @ 0.50: {best_validation['recall_tpr']:.6f}
- FPR @ 0.50: {best_validation['fpr']:.6f}

Best test PR-AUC:

- Model: `{best_test['model']}`
- PR-AUC: {best_test['pr_auc']:.6f}
- ROC-AUC: {best_test['roc_auc']:.6f}
- Precision @ 0.50: {best_test['precision']:.6f}
- Recall @ 0.50: {best_test['recall_tpr']:.6f}
- FPR @ 0.50: {best_test['fpr']:.6f}

## Explainability

The script saves:

- `shap_top_features.csv` and `figures/shap_top_features.png` when SHAP supports the best tree pipeline.
- `permutation_importance_raw_features.csv` and `figures/permutation_importance_raw_features.png` as a model-agnostic check.

SHAP is useful for global contribution patterns and local explanations. Permutation
importance is easier to explain to non-technical stakeholders because it measures how
much validation PR-AUC drops when a raw feature is shuffled.

Important fairness note: if socioeconomic variables such as `housing_status` or
`employment_status` appear important, that is not automatically a reason to keep or
remove them. It is a reason to run a comparison with and without those fields and audit
group-level false positive/false negative behavior before recommending deployment.
"""
    (RESULTS_DIR / "advanced_modeling_report.md").write_text(report, encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("Loading data...")
    data = pd.read_csv("data_banca/Base.csv")
    train, valid, test, train_months, valid_month, test_month = split_before_preprocessing(data)
    train_sample = stratified_training_sample(train)
    print(f"Train sample for advanced experiments: {len(train_sample):,}")

    unusable_columns = [
        column
        for column in train.columns
        if column != TARGET and train[column].nunique(dropna=False) <= 1
    ]
    print(f"Unusable constant columns removed: {unusable_columns}")

    make_feature_evidence(train)

    drop_columns = [TARGET] + unusable_columns
    X_train = make_raw_features(train_sample, drop_columns)
    y_train = train_sample[TARGET].copy()
    X_valid = make_raw_features(valid, drop_columns)
    y_valid = valid[TARGET].copy()
    X_test = make_raw_features(test, drop_columns)
    y_test = test[TARGET].copy()

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / max(positive_count, 1)
    print(f"Training-sample fraud rate: {positive_count / len(y_train):.4%}")
    print(f"scale_pos_weight: {scale_pos_weight:.3f}")

    validation_results, test_results, score_rows, fitted_models = run_experiments(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        scale_pos_weight,
    )

    print("\nValidation metrics sorted by PR-AUC:")
    print(validation_results.to_string(index=False))
    print("\nTest metrics sorted by PR-AUC:")
    print(test_results.to_string(index=False))

    explainable_names = [
        model_name
        for model_name in validation_results["model"]
        if "target/frequency" in model_name and ("XGBoost" in model_name or "LightGBM" in model_name)
    ]
    if explainable_names:
        explainable_name = explainable_names[0]
        save_explainability(explainable_name, fitted_models[explainable_name], X_valid, y_valid)
    else:
        best_model_name = validation_results.iloc[0]["model"]
        save_explainability(best_model_name, fitted_models[best_model_name], X_valid, y_valid)

    write_report(
        validation_results,
        test_results,
        train_sample_rows=len(train_sample),
        train_fraud_rate=positive_count / len(y_train),
    )

    metadata = {
        "train_months": train_months,
        "valid_month": valid_month,
        "test_month": test_month,
        "train_sample_rows": len(train_sample),
        "scale_pos_weight": scale_pos_weight,
        "unusable_columns": unusable_columns,
    }
    (RESULTS_DIR / "advanced_run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved advanced experiment artifacts in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
