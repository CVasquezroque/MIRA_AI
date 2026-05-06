from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

from model.full_holistic.constants import TARGET


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

LOG_CANDIDATES = [
    "income",
    "proposed_credit_limit",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "zip_count_4w",
    "session_length_in_minutes",
]


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        dtype_name = str(frame[column].dtype)
        if dtype_name in {"object", "category", "str", "string"} or dtype_name.startswith("string["):
            columns.append(column)
    return columns


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    cats = set(categorical_columns(frame))
    return [column for column in frame.columns if column not in cats]


def make_raw_features(frame: pd.DataFrame, drop_sensitive: bool = False) -> pd.DataFrame:
    drop_cols = [TARGET]
    if drop_sensitive:
        drop_cols.extend(["housing_status", "employment_status", "customer_age", "income"])
    return frame.drop(columns=drop_cols, errors="ignore").copy()


class AdvancedFeatureBuilder(BaseEstimator, TransformerMixin):
    """EDA-motivated feature builder used by autonomous stages."""

    def __init__(self, add_missing_flags=True, add_log_features=True, add_ratio_features=True, add_interactions=True):
        self.add_missing_flags = add_missing_flags
        self.add_log_features = add_log_features
        self.add_ratio_features = add_ratio_features
        self.add_interactions = add_interactions

    def fit(self, X, y=None):
        X = self._replace_sentinels(pd.DataFrame(X).copy())
        self.log_columns_ = []
        for column in LOG_CANDIDATES:
            if column not in X.columns:
                continue
            values = pd.to_numeric(X[column], errors="coerce").dropna()
            if values.empty or values.min() < 0:
                continue
            median = values.median()
            p99 = values.quantile(0.99)
            if (median > 0 and p99 / median >= 5.0) or abs(values.skew()) >= 1.0:
                self.log_columns_.append(column)
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        if self.add_missing_flags:
            for column in MISSING_FLAG_COLUMNS:
                if column in X_out.columns:
                    X_out[f"{column}_was_missing"] = (X_out[column] == -1).astype(int)
        X_out = self._replace_sentinels(X_out)
        if self.add_log_features:
            for column in self.log_columns_:
                if column in X_out.columns:
                    X_out[f"log1p_{column}"] = np.log1p(pd.to_numeric(X_out[column], errors="coerce").clip(lower=0))
        if self.add_ratio_features:
            self._add_ratios(X_out)
        if self.add_interactions:
            self._add_interactions(X_out)
        return X_out.replace([np.inf, -np.inf], np.nan)

    def _replace_sentinels(self, X: pd.DataFrame) -> pd.DataFrame:
        for column in SENTINEL_TO_NA:
            if column in X.columns:
                X[column] = X[column].replace(-1, np.nan)
        return X

    def _add_ratios(self, X: pd.DataFrame) -> None:
        pairs = [
            ("velocity_6h", "velocity_24h", "velocity_6h_to_24h"),
            ("velocity_24h", "velocity_4w", "velocity_24h_to_4w"),
            ("credit_risk_score", "proposed_credit_limit", "risk_score_to_credit_limit"),
            ("session_length_in_minutes", "device_distinct_emails_8w", "session_length_per_device_email"),
            ("current_address_months_count", "customer_age", "address_months_to_age"),
        ]
        for numerator, denominator, name in pairs:
            if numerator in X.columns and denominator in X.columns:
                den = pd.to_numeric(X[denominator], errors="coerce").replace(0, np.nan)
                X[name] = pd.to_numeric(X[numerator], errors="coerce") / den

    def _add_interactions(self, X: pd.DataFrame) -> None:
        for left, right, name in [
            ("device_os", "source", "device_os__source"),
            ("email_is_free", "source", "email_free__source"),
            ("phone_home_valid", "phone_mobile_valid", "phone_valid_combo"),
            ("payment_type", "employment_status", "payment_type__employment_status"),
        ]:
            if left in X.columns and right in X.columns:
                X[name] = X[left].fillna("Unknown").astype(str) + "__" + X[right].fillna("Unknown").astype(str)


class TemporalTargetFrequencyEncoder(BaseEstimator, TransformerMixin):
    """Temporal target/frequency encoder.

    Training rows are encoded month-by-month from prior months only when possible;
    validation and test use mappings learned from the full training period.
    """

    def __init__(self, smoothing=50.0, min_samples_leaf=20, month_column="month"):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.month_column = month_column

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(y, index=X.index).astype(float)
        self.global_mean_ = float(y.mean())
        self.category_columns_ = categorical_columns(X)
        self.columns_to_drop_ = list(self.category_columns_)
        if self.month_column in X.columns:
            self.columns_to_drop_.append(self.month_column)
        self.mappings_ = {}
        self.freq_mappings_ = {}
        for column in self.category_columns_:
            category = X[column].fillna("Unknown").astype(str)
            stats = pd.DataFrame({"category": category, "target": y}).groupby("category")["target"].agg(["mean", "count"])
            self.mappings_[column] = self._smooth(stats["mean"], stats["count"]).to_dict()
            self.freq_mappings_[column] = (stats["count"] / len(X)).to_dict()
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        X = pd.DataFrame(X).copy()
        y = pd.Series(y, index=X.index).astype(float)
        encoded = X.copy()
        if self.month_column in X.columns and X[self.month_column].nunique(dropna=True) > 1:
            months = sorted(X[self.month_column].dropna().unique())
            for column in self.category_columns_:
                encoded[f"{column}_target_mean"] = self.global_mean_
                encoded[f"{column}_frequency"] = 0.0
                for month in months:
                    current = X[self.month_column] == month
                    previous = X[self.month_column] < month
                    if previous.sum() < self.min_samples_leaf:
                        continue
                    prev_category = X.loc[previous, column].fillna("Unknown").astype(str)
                    stats = pd.DataFrame({"category": prev_category, "target": y.loc[previous]}).groupby("category")["target"].agg(["mean", "count"])
                    smooth = self._smooth(stats["mean"], stats["count"])
                    freq = stats["count"] / previous.sum()
                    cur_category = X.loc[current, column].fillna("Unknown").astype(str)
                    encoded.loc[current, f"{column}_target_mean"] = cur_category.map(smooth).fillna(self.global_mean_).to_numpy()
                    encoded.loc[current, f"{column}_frequency"] = cur_category.map(freq).fillna(0.0).to_numpy()
        else:
            encoded = self.transform(X)
        return encoded.drop(columns=self.columns_to_drop_, errors="ignore")

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        encoded = X.copy()
        for column in self.category_columns_:
            category = X[column].fillna("Unknown").astype(str)
            encoded[f"{column}_target_mean"] = category.map(self.mappings_[column]).fillna(self.global_mean_).astype(float)
            encoded[f"{column}_frequency"] = category.map(self.freq_mappings_[column]).fillna(0.0).astype(float)
        return encoded.drop(columns=self.columns_to_drop_, errors="ignore")

    def _smooth(self, means, counts):
        weight = 1 / (1 + np.exp(-(counts - self.min_samples_leaf) / self.smoothing))
        return self.global_mean_ * (1 - weight) + means * weight


class DataFrameMedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
        self.columns_ = X.columns.tolist()
        self.medians_ = X.median(numeric_only=True)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan).reindex(columns=self.columns_)
        return X.fillna(self.medians_).fillna(0.0)


class DataFrameRobustScaler(BaseEstimator, TransformerMixin):
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
