"""
Progressive holistic fraud modeling workflow.

This script rebuilds the former hollistic analysis with a staged approach:

1. randomized baseline tuning,
2. balancing expansion for strong baseline families,
3. advanced feature expansion,
4. anomaly/recency expansion,
5. SHAP for top models,
6. housing_status fairness audit for top candidates.

It intentionally avoids a full cartesian product. Each stage promotes a limited
number of candidates to the next stage.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.special import logit
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "model"
RESULTS_DIR = MODEL_DIR / "results" / "holistic"
DATA_PATH = PROJECT_DIR / "data_banca" / "Base.csv"

sys.path.insert(0, str(MODEL_DIR))

from randomsearch_tuning import FraudFeatureBuilder as BaselineFeatureBuilder  # noqa: E402
from advanced_feature_modeling import (  # noqa: E402
    AdvancedFeatureBuilder,
    DataFrameMedianImputer,
    DataFrameRobustScaler,
    TemporalTargetFrequencyEncoder,
    categorical_columns,
    evaluate_scores,
    make_lightgbm,
    make_logistic,
    make_raw_features,
    make_stacking_pipeline,
    make_target_frequency_pipeline,
    make_xgboost,
    model_scores,
    numeric_columns,
    split_before_preprocessing,
    threshold_at_fpr,
)
from housing_status_fairness_audit import evaluate_group_metrics, markdown_table as _markdown_table  # noqa: E402


RANDOM_STATE = 42
TARGET = "fraud_bool"
SENSITIVE_COLUMN = "housing_status"
MAIN_FPR_CAP = 0.05

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LocalOutlierFactor was fitted with feature names",
)


def markdown_table(frame: pd.DataFrame) -> str:
    clean = frame.copy()
    for column in clean.columns:
        clean[column] = clean[column].map(
            lambda value: str(value).replace("|", "/") if not pd.isna(value) else value
        )
    return _markdown_table(clean)


@dataclass
class RunConfig:
    mode: str
    tuning_train_rows: int
    tuning_valid_rows: int
    train_rows: int
    eval_rows: int | None
    baseline_n_iter: int
    catboost_n_iter: int
    top_n_baseline_to_balance: int
    top_n_to_advanced: int
    top_n_to_anomaly: int
    fairness_top_n: int
    shap_top_n: int
    shap_sample_rows: int
    anomaly_legit_rows: int
    lof_legit_rows: int
    autoencoder_legit_rows: int
    include_expensive_ensembles: bool


def config_for_mode(mode: str) -> RunConfig:
    if mode == "smoke":
        return RunConfig(
            mode=mode,
            tuning_train_rows=2_500,
            tuning_valid_rows=1_000,
            train_rows=6_000,
            eval_rows=2_500,
            baseline_n_iter=1,
            catboost_n_iter=1,
            top_n_baseline_to_balance=2,
            top_n_to_advanced=2,
            top_n_to_anomaly=1,
            fairness_top_n=1,
            shap_top_n=1,
            shap_sample_rows=100,
            anomaly_legit_rows=1_000,
            lof_legit_rows=800,
            autoencoder_legit_rows=800,
            include_expensive_ensembles=False,
        )

    if mode == "full":
        return RunConfig(
            mode=mode,
            tuning_train_rows=90_000,
            tuning_valid_rows=35_000,
            train_rows=180_000,
            eval_rows=None,
            baseline_n_iter=6,
            catboost_n_iter=5,
            top_n_baseline_to_balance=10,
            top_n_to_advanced=10,
            top_n_to_anomaly=6,
            fairness_top_n=10,
            shap_top_n=3,
            shap_sample_rows=1_500,
            anomaly_legit_rows=25_000,
            lof_legit_rows=15_000,
            autoencoder_legit_rows=18_000,
            include_expensive_ensembles=True,
        )

    raise ValueError(f"Unknown mode: {mode}")


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


class DecisionLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "# Progressive Holistic Run Log\n\n"
            "This file is written as the pipeline advances. It records decisions, "
            "promotion gates, and key outputs.\n\n",
            encoding="utf-8",
        )

    def write(self, title: str, body: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"## {title}\n\n{body.strip()}\n\n")


class AnomalyScoreAppender(BaseEstimator, TransformerMixin):
    """Append anomaly scores fitted only on legitimate training rows."""

    def __init__(
        self,
        legit_sample_rows: int,
        lof_sample_rows: int,
        autoencoder_sample_rows: int,
    ):
        self.legit_sample_rows = legit_sample_rows
        self.lof_sample_rows = lof_sample_rows
        self.autoencoder_sample_rows = autoencoder_sample_rows

    def fit(self, X, y):
        self.preprocessor_ = Pipeline(
            [
                ("features", AdvancedFeatureBuilder()),
                ("target_frequency", TemporalTargetFrequencyEncoder()),
                ("imputer", DataFrameMedianImputer()),
                ("scaler", DataFrameRobustScaler()),
            ]
        )
        prepared = self.preprocessor_.fit_transform(X, y)
        prepared = pd.DataFrame(prepared).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        y_series = pd.Series(y, index=prepared.index)
        legit = prepared[y_series == 0]

        iso_sample = self._sample(legit, self.legit_sample_rows)
        lof_sample = self._sample(legit, self.lof_sample_rows)
        ae_sample = self._sample(legit, self.autoencoder_sample_rows)

        self.isolation_forest_ = IsolationForest(
            n_estimators=80,
            max_samples=min(10_000, len(iso_sample)),
            contamination="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.isolation_forest_.fit(iso_sample)

        self.lof_ = LocalOutlierFactor(
            n_neighbors=35,
            novelty=True,
            contamination="auto",
            n_jobs=-1,
        )
        self.lof_.fit(lof_sample)

        hidden_units = max(8, min(32, prepared.shape[1] // 2))
        self.autoencoder_ = MLPRegressor(
            hidden_layer_sizes=(hidden_units,),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=35,
            early_stopping=True,
            validation_fraction=0.12,
            random_state=RANDOM_STATE,
        )
        self.autoencoder_.fit(ae_sample, ae_sample)
        return self

    def transform(self, X):
        prepared = self.preprocessor_.transform(X)
        prepared = pd.DataFrame(prepared).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        reconstructed = self.autoencoder_.predict(prepared)

        X_out = X.copy()
        X_out["isolation_forest_anomaly_score"] = -self.isolation_forest_.score_samples(prepared)
        X_out["lof_anomaly_score"] = -self.lof_.score_samples(prepared)
        X_out["autoencoder_reconstruction_error"] = np.mean(
            (prepared.to_numpy() - reconstructed) ** 2,
            axis=1,
        )
        return X_out

    def _sample(self, frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if len(frame) <= max_rows:
            return frame
        return frame.sample(n=max_rows, random_state=RANDOM_STATE)


def safe_name(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")[:120]


def sample_frame(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.copy()
    sample, _ = train_test_split(
        frame,
        train_size=max_rows,
        stratify=frame[TARGET],
        random_state=RANDOM_STATE,
    )
    return sample.copy()


def scale_pos_weight_from_y(y, sample_weight=None) -> float:
    y_arr = pd.Series(y).to_numpy()
    if sample_weight is None:
        neg = (y_arr == 0).sum()
        pos = (y_arr == 1).sum()
    else:
        weights = np.asarray(sample_weight)
        neg = weights[y_arr == 0].sum()
        pos = weights[y_arr == 1].sum()
    return float(neg / max(pos, 1e-12))


def load_context(config: RunConfig, logger: DecisionLogger) -> DataContext:
    data = pd.read_csv(DATA_PATH)
    train, valid, test, train_months, valid_month, test_month = split_before_preprocessing(data)
    unusable_columns = [
        column
        for column in train.columns
        if column != TARGET and train[column].nunique(dropna=False) <= 1
    ]

    train_sample = sample_frame(train, config.train_rows)
    valid_eval = sample_frame(valid, config.eval_rows)
    test_eval = sample_frame(test, config.eval_rows)
    scale_pos_weight = scale_pos_weight_from_y(train_sample[TARGET])

    month_drift = (
        pd.concat([train, valid, test])
        .groupby("month")[TARGET]
        .agg(count="size", fraud_rate="mean")
        .reset_index()
    )
    month_drift.to_csv(RESULTS_DIR / "00_monthly_fraud_rate_drift.csv", index=False)

    logger.write(
        "00 Data Audit",
        "\n".join(
            [
                f"- Data rows: {len(data):,}.",
                f"- Train rows: {len(train):,}; validation rows: {len(valid):,}; test rows: {len(test):,}.",
                f"- Run train sample rows: {len(train_sample):,}.",
                f"- Validation eval rows: {len(valid_eval):,}; test eval rows: {len(test_eval):,}.",
                f"- Train months: {train_months}; validation month: {valid_month}; test month: {test_month}.",
                f"- Removed constant columns: {unusable_columns}.",
                f"- scale_pos_weight on run train sample: {scale_pos_weight:.4f}.",
            ]
        ),
    )

    return DataContext(
        train=train,
        valid=valid,
        test=test,
        train_sample=train_sample,
        valid_eval=valid_eval,
        test_eval=test_eval,
        train_months=train_months,
        valid_month=valid_month,
        test_month=test_month,
        unusable_columns=unusable_columns,
        scale_pos_weight=scale_pos_weight,
    )


def make_inner_tuning_frames(context: DataContext, config: RunConfig):
    train = context.train
    train_months = context.train_months
    if train_months is not None and len(train_months) >= 2:
        inner_valid_month = train_months[-1]
        inner_train_months = train_months[:-1]
        inner_train = train[train["month"].isin(inner_train_months)].copy()
        inner_valid = train[train["month"] == inner_valid_month].copy()
    else:
        inner_train, inner_valid = train_test_split(
            train,
            test_size=0.25,
            stratify=train[TARGET],
            random_state=RANDOM_STATE,
        )

    inner_train = sample_frame(inner_train, config.tuning_train_rows)
    inner_valid = sample_frame(inner_valid, config.tuning_valid_rows)
    tuning_data = pd.concat([inner_train, inner_valid], axis=0).reset_index(drop=True)
    test_fold = np.r_[
        np.full(len(inner_train), -1, dtype=int),
        np.zeros(len(inner_valid), dtype=int),
    ]
    return inner_train, inner_valid, tuning_data, PredefinedSplit(test_fold)


def baseline_drop_columns(context: DataContext, drop_sensitive: bool = False) -> list[str]:
    drop_columns = [TARGET] + context.unusable_columns
    if "month" not in drop_columns:
        drop_columns.append("month")
    if drop_sensitive:
        drop_columns.append(SENSITIVE_COLUMN)
    return drop_columns


def advanced_drop_columns(context: DataContext, drop_sensitive: bool = False) -> list[str]:
    drop_columns = [TARGET] + context.unusable_columns
    if drop_sensitive:
        drop_columns.append(SENSITIVE_COLUMN)
    return drop_columns


def build_onehot_preprocessor(scale_numeric: bool, dense: bool = False) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", RobustScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
    ]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_columns),
            ("cat", Pipeline(categorical_steps), categorical_columns),
        ],
        sparse_threshold=0.0 if dense else 0.3,
    )


def make_baseline_pipeline(model, scale_numeric: bool, dense: bool = False):
    pipeline_cls = ImbPipeline if dense else Pipeline
    return pipeline_cls(
        steps=[
            ("features", BaselineFeatureBuilder()),
            ("preprocess", build_onehot_preprocessor(scale_numeric=scale_numeric, dense=dense)),
            ("model", model),
        ]
    )


def make_baseline_spaces(scale_pos_weight: float, n_iter: int) -> dict[str, dict]:
    spaces: dict[str, dict] = {}
    spaces["Logistic Regression"] = {
        "family": "Logistic Regression",
        "estimator": make_baseline_pipeline(
            LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=350),
            scale_numeric=True,
        ),
        "params": {
            "features__add_missing_flags": [True, False],
            "features__add_outlier_flags": [True, False],
            "features__add_log_features": [True, False],
            "model__C": loguniform(0.01, 10),
        },
        "n_iter": n_iter,
    }
    spaces["Decision Tree"] = {
        "family": "Decision Tree",
        "estimator": make_baseline_pipeline(
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
        "n_iter": n_iter,
    }
    spaces["Random Forest"] = {
        "family": "Random Forest",
        "estimator": make_baseline_pipeline(
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
            "model__n_estimators": randint(50, 121),
            "model__max_depth": [8, 10, 12, 16, None],
            "model__min_samples_leaf": randint(50, 350),
            "model__max_features": ["sqrt", "log2", 0.5],
        },
        "n_iter": n_iter,
    }

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    spaces["XGBoost"] = {
        "family": "XGBoost",
        "estimator": make_baseline_pipeline(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
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
            "model__n_estimators": randint(60, 181),
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
        "n_iter": n_iter,
    }
    spaces["LightGBM"] = {
        "family": "LightGBM",
        "estimator": make_baseline_pipeline(
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
            "model__n_estimators": randint(50, 151),
            "model__num_leaves": randint(15, 55),
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
        "n_iter": n_iter,
    }
    return spaces


def score_fitted_model(fitted, X, kind: str = "pipeline"):
    if kind == "catboost_native":
        return catboost_native_scores(fitted, X)
    if hasattr(fitted, "predict_proba"):
        return fitted.predict_proba(X)[:, 1]
    return fitted.decision_function(X)


def catboost_native_scores(fitted: dict, X: pd.DataFrame):
    X_cb = fitted["builder"].transform(X).drop(columns=["month"], errors="ignore")
    for column in fitted["cat_cols"]:
        if column in X_cb.columns:
            X_cb[column] = X_cb[column].fillna("Unknown").astype(str)
    return fitted["model"].predict_proba(X_cb)[:, 1]


def fit_catboost_native(
    X_train,
    y_train,
    X_valid,
    y_valid,
    scale_pos_weight: float,
    params: dict | None = None,
    sample_weight=None,
):
    from catboost import CatBoostClassifier

    params = params or {}
    builder = AdvancedFeatureBuilder()
    X_train_cb = builder.fit_transform(X_train, y_train).drop(columns=["month"], errors="ignore")
    X_valid_cb = builder.transform(X_valid).drop(columns=["month"], errors="ignore")
    cat_cols = categorical_columns(X_train_cb)
    for column in cat_cols:
        X_train_cb[column] = X_train_cb[column].fillna("Unknown").astype(str)
        X_valid_cb[column] = X_valid_cb[column].fillna("Unknown").astype(str)

    model = CatBoostClassifier(
        iterations=params.get("iterations", 260),
        depth=params.get("depth", 6),
        learning_rate=params.get("learning_rate", 0.055),
        l2_leaf_reg=params.get("l2_leaf_reg", 6.0),
        loss_function="Logloss",
        eval_metric="PRAUC",
        scale_pos_weight=params.get("scale_pos_weight", scale_pos_weight),
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        X_train_cb,
        y_train,
        cat_features=cat_cols,
        sample_weight=sample_weight,
        eval_set=(X_valid_cb, y_valid),
        use_best_model=True,
        early_stopping_rounds=params.get("early_stopping_rounds", 45),
    )
    return {"model": model, "builder": builder, "cat_cols": cat_cols}


def evaluate_candidate(
    all_metrics: list[dict],
    fitted_registry: dict[str, dict],
    *,
    model_name: str,
    stage: str,
    model_family: str,
    feature_set: str,
    balance_policy: str,
    train_strategy: str,
    anomaly_policy: str,
    fitted,
    model_kind: str,
    X_valid: pd.DataFrame,
    y_valid,
    X_test: pd.DataFrame,
    y_test,
    spec: dict,
) -> dict:
    valid_scores = score_fitted_model(fitted, X_valid, model_kind)
    test_scores = score_fitted_model(fitted, X_test, model_kind)
    selected_threshold = threshold_at_fpr(y_valid, valid_scores, max_fpr=MAIN_FPR_CAP)

    common = {
        "model": model_name,
        "stage": stage,
        "model_family": model_family,
        "feature_set": feature_set,
        "balance_policy": balance_policy,
        "train_strategy": train_strategy,
        "anomaly_policy": anomaly_policy,
    }
    for split_name, y_true, scores in [
        ("validation", y_valid, valid_scores),
        ("test", y_test, test_scores),
    ]:
        for policy, threshold in [
            ("default_0_50", 0.50),
            ("valid_global_5pct_fpr", selected_threshold),
        ]:
            metrics = evaluate_scores(y_true, scores, threshold=threshold)
            all_metrics.append(
                {
                    **common,
                    "split": split_name,
                    "threshold_policy": policy,
                    "threshold": threshold,
                    **metrics,
                }
            )

    candidate = {
        **common,
        "selected_threshold": float(selected_threshold),
        "validation_pr_auc": float(average_precision_score(y_valid, valid_scores)),
        "validation_roc_auc": float(roc_auc_score(y_valid, valid_scores)),
        "test_pr_auc": float(average_precision_score(y_test, test_scores)),
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "spec": spec,
    }
    fitted_registry[model_name] = {
        "fitted": fitted,
        "model_kind": model_kind,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "valid_scores": valid_scores,
        "X_test": X_test,
        "y_test": y_test,
        "test_scores": test_scores,
        "candidate": candidate,
    }
    return candidate


def run_randomized_baselines(
    context: DataContext,
    config: RunConfig,
    logger: DecisionLogger,
    all_metrics: list[dict],
    fitted_registry: dict[str, dict],
) -> list[dict]:
    print("[01] Running randomized baseline search...", flush=True)
    logger.write(
        "01 Baseline Randomized Search - Start",
        "Comparing Logistic Regression, Decision Tree, Random Forest, XGBoost, "
        "and LightGBM with EDA-driven preprocessing options inside the pipeline. "
        "The search score is PR-AUC on an inner chronological validation fold.",
    )
    _, _, tuning_data, predefined_cv = make_inner_tuning_frames(context, config)

    drop_columns = baseline_drop_columns(context)
    X_tune = make_raw_features(tuning_data, drop_columns)
    y_tune = tuning_data[TARGET].copy()
    X_train = make_raw_features(context.train_sample, drop_columns)
    y_train = context.train_sample[TARGET].copy()
    X_valid = make_raw_features(context.valid_eval, drop_columns)
    y_valid = context.valid_eval[TARGET].copy()
    X_test = make_raw_features(context.test_eval, drop_columns)
    y_test = context.test_eval[TARGET].copy()

    searches = {}
    tuned_estimators = {}
    candidates = []
    cv_rows = []
    best_params = {}
    spaces = make_baseline_spaces(context.scale_pos_weight, config.baseline_n_iter)

    for model_name, space in spaces.items():
        print(f"[01] Tuning {model_name}...", flush=True)
        started = time.time()
        search = RandomizedSearchCV(
            estimator=space["estimator"],
            param_distributions=space["params"],
            n_iter=space["n_iter"],
            scoring="average_precision",
            cv=predefined_cv,
            refit=True,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=0,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(X_tune, y_tune)
        searches[model_name] = search
        best_params[model_name] = search.best_params_

        cv_result = pd.DataFrame(search.cv_results_)
        cv_result["model_family"] = model_name
        cv_result["stage"] = "baseline_randomsearch"
        cv_result["elapsed_seconds"] = time.time() - started
        cv_rows.append(cv_result)

        fitted = clone(search.best_estimator_)
        fitted.fit(X_train, y_train)
        model_label = f"baseline_randomsearch | {model_name}"
        tuned_estimators[model_name] = fitted
        candidates.append(
            evaluate_candidate(
                all_metrics,
                fitted_registry,
                model_name=model_label,
                stage="baseline_randomsearch",
                model_family=model_name,
                feature_set="baseline_eda_onehot",
                balance_policy="model_default_weighting",
                train_strategy="full_0_5_sample",
                anomaly_policy="without_anomaly_scores",
                fitted=fitted,
                model_kind="pipeline",
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                y_test=y_test,
                spec={
                    "type": "baseline_randomsearch",
                    "model_family": model_name,
                    "best_params": search.best_params_,
                    "drop_sensitive": False,
                },
            )
        )

    if cv_rows:
        pd.concat(cv_rows, ignore_index=True).to_csv(
            RESULTS_DIR / "01_baseline_randomsearch_cv_results.csv",
            index=False,
        )
    (RESULTS_DIR / "01_baseline_randomsearch_best_params.json").write_text(
        json.dumps(best_params, indent=2, default=str),
        encoding="utf-8",
    )

    print("[01] Fitting CatBoost/ensemble baseline additions...", flush=True)
    candidates.extend(
        run_baseline_catboost_and_ensembles(
            context,
            config,
            all_metrics,
            fitted_registry,
            tuned_estimators,
        )
    )

    baseline_table = pd.DataFrame(candidates).sort_values("validation_pr_auc", ascending=False)
    baseline_table.drop(columns=["spec"]).to_csv(RESULTS_DIR / "01_baseline_candidates.csv", index=False)
    logger.write(
        "01 Baseline Randomized Search - Result",
        "Top baseline candidates by validation PR-AUC:\n\n"
        + markdown_table(
            baseline_table[
                [
                    "model",
                    "model_family",
                    "validation_pr_auc",
                    "validation_roc_auc",
                    "test_pr_auc",
                    "test_roc_auc",
                ]
            ]
            .head(12)
            .round(6)
        ),
    )
    return candidates


def run_baseline_catboost_and_ensembles(
    context: DataContext,
    config: RunConfig,
    all_metrics: list[dict],
    fitted_registry: dict[str, dict],
    tuned_estimators: dict[str, object],
) -> list[dict]:
    candidates = []
    drop_columns = advanced_drop_columns(context)
    X_train = make_raw_features(context.train_sample, drop_columns)
    y_train = context.train_sample[TARGET].copy()
    X_valid = make_raw_features(context.valid_eval, drop_columns)
    y_valid = context.valid_eval[TARGET].copy()
    X_test = make_raw_features(context.test_eval, drop_columns)
    y_test = context.test_eval[TARGET].copy()

    cat_params_grid = catboost_param_candidates(config.catboost_n_iter)
    best_cat = None
    best_score = -np.inf
    for params in cat_params_grid:
        fitted = fit_catboost_native(
            X_train,
            y_train,
            X_valid,
            y_valid,
            context.scale_pos_weight,
            params=params,
        )
        score = average_precision_score(y_valid, catboost_native_scores(fitted, X_valid))
        if score > best_score:
            best_score = score
            best_cat = (params, fitted)

    if best_cat is not None:
        params, fitted = best_cat
        candidates.append(
            evaluate_candidate(
                all_metrics,
                fitted_registry,
                model_name="baseline_catboost | CatBoost native",
                stage="baseline_catboost",
                model_family="CatBoost",
                feature_set="advanced_native_categoricals",
                balance_policy="scale_pos_weight",
                train_strategy="full_0_5_sample",
                anomaly_policy="without_anomaly_scores",
                fitted=fitted,
                model_kind="catboost_native",
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                y_test=y_test,
                spec={
                    "type": "catboost_native",
                    "model_family": "CatBoost",
                    "params": params,
                    "drop_sensitive": False,
                },
            )
        )

    if not config.include_expensive_ensembles:
        return candidates

    base_for_ensemble = [
        (name.lower().replace(" ", "_"), clone(model))
        for name, model in tuned_estimators.items()
        if name in ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
    ][:3]
    if len(base_for_ensemble) >= 2:
        X_train_base = make_raw_features(context.train_sample, baseline_drop_columns(context))
        X_valid_base = make_raw_features(context.valid_eval, baseline_drop_columns(context))
        X_test_base = make_raw_features(context.test_eval, baseline_drop_columns(context))
        y_train_base = context.train_sample[TARGET].copy()
        y_valid_base = context.valid_eval[TARGET].copy()
        y_test_base = context.test_eval[TARGET].copy()

        voting = VotingClassifier(estimators=base_for_ensemble, voting="soft", n_jobs=-1)
        voting.fit(X_train_base, y_train_base)
        candidates.append(
            evaluate_candidate(
                all_metrics,
                fitted_registry,
                model_name="baseline_ensemble | Voting",
                stage="baseline_ensemble",
                model_family="Voting",
                feature_set="baseline_eda_onehot",
                balance_policy="inherits_base_estimators",
                train_strategy="full_0_5_sample",
                anomaly_policy="without_anomaly_scores",
                fitted=voting,
                model_kind="pipeline",
                X_valid=X_valid_base,
                y_valid=y_valid_base,
                X_test=X_test_base,
                y_test=y_test_base,
                spec={
                    "type": "voting_baseline",
                    "model_family": "Voting",
                    "base_families": [name for name, _ in base_for_ensemble],
                    "drop_sensitive": False,
                },
            )
        )

        stacking = StackingClassifier(
            estimators=base_for_ensemble,
            final_estimator=LogisticRegression(class_weight="balanced", max_iter=300),
            stack_method="predict_proba",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=1,
        )
        stacking.fit(X_train_base, y_train_base)
        candidates.append(
            evaluate_candidate(
                all_metrics,
                fitted_registry,
                model_name="baseline_ensemble | Stacking",
                stage="baseline_ensemble",
                model_family="Stacking",
                feature_set="baseline_eda_onehot",
                balance_policy="inherits_base_estimators",
                train_strategy="full_0_5_sample",
                anomaly_policy="without_anomaly_scores",
                fitted=stacking,
                model_kind="pipeline",
                X_valid=X_valid_base,
                y_valid=y_valid_base,
                X_test=X_test_base,
                y_test=y_test_base,
                spec={
                    "type": "stacking_baseline",
                    "model_family": "Stacking",
                    "base_families": [name for name, _ in base_for_ensemble],
                    "drop_sensitive": False,
                },
            )
        )

    return candidates


def catboost_param_candidates(n_iter: int) -> list[dict]:
    if n_iter <= 1:
        return [
            {
                "iterations": 80,
                "depth": 4,
                "learning_rate": 0.06,
                "l2_leaf_reg": 6.0,
                "early_stopping_rounds": 20,
            }
        ]

    rng = np.random.default_rng(RANDOM_STATE)
    choices = []
    for _ in range(max(1, n_iter)):
        choices.append(
            {
                "iterations": int(rng.choice([160, 220, 280, 350])),
                "depth": int(rng.choice([4, 5, 6, 7])),
                "learning_rate": float(rng.choice([0.035, 0.05, 0.065, 0.08])),
                "l2_leaf_reg": float(rng.choice([3.0, 6.0, 9.0, 12.0])),
                "early_stopping_rounds": 40,
            }
        )
    return choices


def model_families_from_top(candidates: list[dict], top_n: int) -> list[str]:
    families = []
    for row in sorted(candidates, key=lambda item: item["validation_pr_auc"], reverse=True):
        family = row["model_family"]
        if family not in families:
            families.append(family)
        if len(families) >= top_n:
            break
    return families


def make_sampler(policy: str):
    if policy == "random_undersampling":
        return RandomUnderSampler(sampling_strategy=0.25, random_state=RANDOM_STATE)
    if policy == "random_oversampling":
        return RandomOverSampler(sampling_strategy=0.10, random_state=RANDOM_STATE)
    if policy == "smote":
        return SMOTE(sampling_strategy=0.10, k_neighbors=3, random_state=RANDOM_STATE)
    return None


def make_balanced_pipeline(model_family: str, policy: str, scale_pos_weight: float):
    sampler = make_sampler(policy)
    if model_family == "Logistic Regression":
        model = LogisticRegression(
            C=0.03,
            class_weight="balanced" if policy == "class_weight" else None,
            max_iter=400,
            random_state=RANDOM_STATE,
        )
        pipeline = make_baseline_pipeline(model, scale_numeric=True, dense=True)
    elif model_family == "Decision Tree":
        model = DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=120,
            class_weight="balanced" if policy == "class_weight" else None,
            random_state=RANDOM_STATE,
        )
        pipeline = make_baseline_pipeline(model, scale_numeric=False, dense=True)
    elif model_family == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=90,
            max_depth=12,
            min_samples_leaf=70,
            max_features="sqrt",
            class_weight="balanced_subsample" if policy == "class_weight" else None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        pipeline = make_baseline_pipeline(model, scale_numeric=False, dense=True)
    elif model_family == "XGBoost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_estimators=140,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.88,
            colsample_bytree=0.76,
            min_child_weight=3,
            gamma=2.34,
            reg_lambda=0.60,
            scale_pos_weight=scale_pos_weight if policy == "class_weight" else 1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        pipeline = make_baseline_pipeline(model, scale_numeric=False, dense=True)
    elif model_family == "LightGBM":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            objective="binary",
            n_estimators=120,
            learning_rate=0.04,
            max_depth=8,
            num_leaves=17,
            min_child_samples=269,
            subsample=0.75,
            colsample_bytree=0.81,
            reg_lambda=5.25,
            scale_pos_weight=scale_pos_weight if policy == "class_weight" else 1.0,
            random_state=RANDOM_STATE,
            n_jobs=4,
            verbose=-1,
            force_col_wise=True,
            device_type="cpu",
        )
        pipeline = make_baseline_pipeline(model, scale_numeric=False, dense=True)
    else:
        return None

    if sampler is not None:
        pipeline.steps.insert(-1, ("sampler", sampler))
    return pipeline


def run_balancing_gate(
    context: DataContext,
    config: RunConfig,
    logger: DecisionLogger,
    all_metrics: list[dict],
    fitted_registry: dict[str, dict],
    baseline_candidates: list[dict],
) -> list[dict]:
    print("[02] Running balancing gate...", flush=True)
    promoted_families = model_families_from_top(
        baseline_candidates,
        config.top_n_baseline_to_balance,
    )
    policies = ["no_balance", "class_weight", "random_undersampling", "random_oversampling", "smote"]
    drop_columns = baseline_drop_columns(context)
    X_train = make_raw_features(context.train_sample, drop_columns)
    y_train = context.train_sample[TARGET].copy()
    X_valid = make_raw_features(context.valid_eval, drop_columns)
    y_valid = context.valid_eval[TARGET].copy()
    X_test = make_raw_features(context.test_eval, drop_columns)
    y_test = context.test_eval[TARGET].copy()

    candidates = []
    skipped = []
    for family in promoted_families:
        print(f"[02] Expanding balance policies for {family}...", flush=True)
        if family in ["Voting", "Stacking"]:
            skipped.append(f"{family}: ensemble balance expansion is skipped; base models carry balance.")
            continue

        if family == "CatBoost":
            for policy in ["no_balance", "class_weight"]:
                params = {"scale_pos_weight": context.scale_pos_weight if policy == "class_weight" else 1.0}
                fitted = fit_catboost_native(
                    make_raw_features(context.train_sample, advanced_drop_columns(context)),
                    y_train,
                    make_raw_features(context.valid_eval, advanced_drop_columns(context)),
                    y_valid,
                    context.scale_pos_weight,
                    params=params,
                )
                label = f"balance_gate | CatBoost | {policy}"
                candidates.append(
                    evaluate_candidate(
                        all_metrics,
                        fitted_registry,
                        model_name=label,
                        stage="balance_gate",
                        model_family="CatBoost",
                        feature_set="advanced_native_categoricals",
                        balance_policy=policy,
                        train_strategy="full_0_5_sample",
                        anomaly_policy="without_anomaly_scores",
                        fitted=fitted,
                        model_kind="catboost_native",
                        X_valid=make_raw_features(context.valid_eval, advanced_drop_columns(context)),
                        y_valid=y_valid,
                        X_test=make_raw_features(context.test_eval, advanced_drop_columns(context)),
                        y_test=y_test,
                        spec={
                            "type": "catboost_native",
                            "model_family": "CatBoost",
                            "params": params,
                            "balance_policy": policy,
                        },
                    )
                )
            skipped.append("CatBoost: sampler policies skipped for native categorical training.")
            continue

        for policy in policies:
            pipeline = make_balanced_pipeline(family, policy, context.scale_pos_weight)
            if pipeline is None:
                skipped.append(f"{family}: no balancing factory for {policy}.")
                continue
            fitted = clone(pipeline)
            fitted.fit(X_train, y_train)
            label = f"balance_gate | {family} | {policy}"
            candidates.append(
                evaluate_candidate(
                    all_metrics,
                    fitted_registry,
                    model_name=label,
                    stage="balance_gate",
                    model_family=family,
                    feature_set="baseline_eda_onehot_dense",
                    balance_policy=policy,
                    train_strategy="full_0_5_sample",
                    anomaly_policy="without_anomaly_scores",
                    fitted=fitted,
                    model_kind="pipeline",
                    X_valid=X_valid,
                    y_valid=y_valid,
                    X_test=X_test,
                    y_test=y_test,
                    spec={
                        "type": "balance_gate",
                        "model_family": family,
                        "balance_policy": policy,
                    },
                )
            )

    table = pd.DataFrame(candidates).sort_values("validation_pr_auc", ascending=False)
    if not table.empty:
        table.drop(columns=["spec"]).to_csv(RESULTS_DIR / "02_balancing_candidates.csv", index=False)
    logger.write(
        "02 Balancing Gate",
        "Promoted baseline families: "
        + ", ".join(promoted_families)
        + ".\n\nSkipped notes:\n"
        + "\n".join(f"- {item}" for item in skipped)
        + "\n\nTop balancing candidates:\n\n"
        + markdown_table(
            table[
                [
                    "model",
                    "model_family",
                    "balance_policy",
                    "validation_pr_auc",
                    "test_pr_auc",
                ]
            ]
            .head(12)
            .round(6)
        )
        if not table.empty
        else "No balancing candidates were produced.",
    )
    return candidates


def make_advanced_model(model_family: str, scale_pos_weight: float):
    if model_family == "Logistic Regression":
        return make_target_frequency_pipeline(make_logistic(), scale_numeric=True), "pipeline"
    if model_family == "Decision Tree":
        model = DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=8,
            min_samples_leaf=120,
            random_state=RANDOM_STATE,
        )
        return make_target_frequency_pipeline(model), "pipeline"
    if model_family == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=110,
            max_depth=12,
            min_samples_leaf=70,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        return make_target_frequency_pipeline(model), "pipeline"
    if model_family == "XGBoost":
        return make_target_frequency_pipeline(
            make_xgboost(standard=True, scale_pos_weight=scale_pos_weight)
        ), "pipeline"
    if model_family == "LightGBM":
        return make_target_frequency_pipeline(make_lightgbm(scale_pos_weight=scale_pos_weight)), "pipeline"
    if model_family == "Stacking":
        return make_stacking_pipeline(scale_pos_weight), "pipeline"
    if model_family == "Voting":
        estimators = [
            ("xgb", make_target_frequency_pipeline(make_xgboost(standard=True, scale_pos_weight=scale_pos_weight))),
            ("lgbm", make_target_frequency_pipeline(make_lightgbm(scale_pos_weight=scale_pos_weight))),
            ("lr", make_target_frequency_pipeline(make_logistic(), scale_numeric=True)),
        ]
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1), "pipeline"
    return None, None


def run_advanced_feature_gate(
    context: DataContext,
    config: RunConfig,
    logger: DecisionLogger,
    all_metrics: list[dict],
    fitted_registry: dict[str, dict],
    previous_candidates: list[dict],
) -> list[dict]:
    print("[03] Running advanced feature gate...", flush=True)
    promoted_families = model_families_from_top(previous_candidates, config.top_n_to_advanced)
    drop_columns = advanced_drop_columns(context)
    X_train = make_raw_features(context.train_sample, drop_columns)
    y_train = context.train_sample[TARGET].copy()
    X_valid = make_raw_features(context.valid_eval, drop_columns)
    y_valid = context.valid_eval[TARGET].copy()
    X_test = make_raw_features(context.test_eval, drop_columns)
    y_test = context.test_eval[TARGET].copy()

    candidates = []
    skipped = []
    for family in promoted_families:
        print(f"[03] Training advanced candidate for {family}...", flush=True)
        if family == "CatBoost":
            fitted = fit_catboost_native(
                X_train,
                y_train,
                X_valid,
                y_valid,
                context.scale_pos_weight,
            )
            kind = "catboost_native"
        else:
            model, kind = make_advanced_model(family, context.scale_pos_weight)
            if model is None:
                skipped.append(f"{family}: no advanced factory.")
                continue
            fitted = clone(model)
            fitted.fit(X_train, y_train)

        label = f"advanced_gate | {family}"
        candidates.append(
            evaluate_candidate(
                all_metrics,
                fitted_registry,
                model_name=label,
                stage="advanced_gate",
                model_family=family,
                feature_set="ratios_interactions_target_frequency",
                balance_policy="model_default_weighting",
                train_strategy="full_0_5_sample",
                anomaly_policy="without_anomaly_scores",
                fitted=fitted,
                model_kind=kind,
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                y_test=y_test,
                spec={
                    "type": "advanced_gate",
                    "model_family": family,
                    "balance_policy": "model_default_weighting",
                },
            )
        )

    table = pd.DataFrame(candidates).sort_values("validation_pr_auc", ascending=False)
    if not table.empty:
        table.drop(columns=["spec"]).to_csv(RESULTS_DIR / "03_advanced_candidates.csv", index=False)
    logger.write(
        "03 Advanced Feature Gate",
        "Promoted families: "
        + ", ".join(promoted_families)
        + ". Features include ratio features, interaction features, and temporal "
        "target/frequency encoding.\n\nSkipped notes:\n"
        + "\n".join(f"- {item}" for item in skipped)
        + "\n\nTop advanced candidates:\n\n"
        + markdown_table(
            table[["model", "model_family", "validation_pr_auc", "test_pr_auc"]]
            .head(12)
            .round(6)
        )
        if not table.empty
        else "No advanced candidates were produced.",
    )
    return candidates


def make_train_strategy_frame(context: DataContext, config: RunConfig, strategy: str):
    if strategy == "full_0_5":
        sample = sample_frame(context.train, config.train_rows)
        return sample, None
    if strategy == "full_0_5_recency_weighted":
        sample = sample_frame(context.train, config.train_rows)
        months = sample["month"].astype(float)
        span = max(months.max() - months.min(), 1.0)
        weights = 0.5 + (months - months.min()) / span
        return sample, (weights / weights.mean()).to_numpy()
    if strategy == "recent_3_5":
        recent = context.train[context.train["month"].isin([3, 4, 5])].copy()
        sample = sample_frame(recent, config.train_rows)
        return sample, None
    raise ValueError(f"Unknown train strategy: {strategy}")


def run_anomaly_recency_gate(
    context: DataContext,
    config: RunConfig,
    logger: DecisionLogger,
    all_metrics: list[dict],
    fitted_registry: dict[str, dict],
    previous_candidates: list[dict],
) -> list[dict]:
    print("[04] Running anomaly/recency gate...", flush=True)
    promoted_families = [
        family
        for family in model_families_from_top(previous_candidates, config.top_n_to_anomaly)
        if family in ["XGBoost", "LightGBM", "CatBoost", "Random Forest", "Logistic Regression", "Stacking", "Voting"]
    ]
    strategies = ["full_0_5", "full_0_5_recency_weighted", "recent_3_5"]
    anomaly_options = [False, True]
    X_valid_base = make_raw_features(context.valid_eval, advanced_drop_columns(context))
    y_valid = context.valid_eval[TARGET].copy()
    X_test_base = make_raw_features(context.test_eval, advanced_drop_columns(context))
    y_test = context.test_eval[TARGET].copy()

    candidates = []
    for strategy in strategies:
        print(f"[04] Strategy {strategy}...", flush=True)
        train_frame, sample_weight = make_train_strategy_frame(context, config, strategy)
        y_train = train_frame[TARGET].copy()
        X_train_base = make_raw_features(train_frame, advanced_drop_columns(context))
        spw = scale_pos_weight_from_y(y_train, sample_weight)

        for use_anomaly in anomaly_options:
            if use_anomaly:
                appender = AnomalyScoreAppender(
                    legit_sample_rows=config.anomaly_legit_rows,
                    lof_sample_rows=config.lof_legit_rows,
                    autoencoder_sample_rows=config.autoencoder_legit_rows,
                )
                appender.fit(X_train_base, y_train)
                X_train = appender.transform(X_train_base)
                X_valid = appender.transform(X_valid_base)
                X_test = appender.transform(X_test_base)
                anomaly_policy = "with_anomaly_scores"
            else:
                X_train = X_train_base
                X_valid = X_valid_base
                X_test = X_test_base
                anomaly_policy = "without_anomaly_scores"

            for family in promoted_families:
                print(f"[04] Training {family} with {strategy} / {anomaly_policy}...", flush=True)
                if family == "CatBoost":
                    fitted = fit_catboost_native(
                        X_train,
                        y_train,
                        X_valid,
                        y_valid,
                        spw,
                        sample_weight=sample_weight,
                    )
                    kind = "catboost_native"
                else:
                    model, kind = make_advanced_model(family, spw)
                    if model is None:
                        continue
                    fitted = clone(model)
                    fit_params = {}
                    if sample_weight is not None and family not in ["Stacking", "Voting"]:
                        fit_params["model__sample_weight"] = sample_weight
                    fitted.fit(X_train, y_train, **fit_params)

                label = f"anomaly_recency_gate | {family} | {strategy} | {anomaly_policy}"
                candidates.append(
                    evaluate_candidate(
                        all_metrics,
                        fitted_registry,
                        model_name=label,
                        stage="anomaly_recency_gate",
                        model_family=family,
                        feature_set="advanced_plus_optional_anomaly",
                        balance_policy="model_default_weighting",
                        train_strategy=strategy,
                        anomaly_policy=anomaly_policy,
                        fitted=fitted,
                        model_kind=kind,
                        X_valid=X_valid,
                        y_valid=y_valid,
                        X_test=X_test,
                        y_test=y_test,
                        spec={
                            "type": "anomaly_recency_gate",
                            "model_family": family,
                            "train_strategy": strategy,
                            "use_anomaly": use_anomaly,
                        },
                    )
                )

    table = pd.DataFrame(candidates).sort_values("validation_pr_auc", ascending=False)
    if not table.empty:
        table.drop(columns=["spec"]).to_csv(RESULTS_DIR / "04_anomaly_recency_candidates.csv", index=False)
    logger.write(
        "04 Anomaly And Recency Gate",
        "Promoted families: "
        + ", ".join(promoted_families)
        + ". Strategies: full_0_5, full_0_5_recency_weighted, recent_3_5. "
        "Anomaly scores are fitted only on legitimate training rows.\n\n"
        + markdown_table(
            table[
                [
                    "model",
                    "model_family",
                    "train_strategy",
                    "anomaly_policy",
                    "validation_pr_auc",
                    "test_pr_auc",
                ]
            ]
            .head(12)
            .round(6)
        )
        if not table.empty
        else "No anomaly/recency candidates were produced.",
    )
    return candidates


def top_validation_candidates(candidates: list[dict], top_n: int) -> list[dict]:
    return sorted(candidates, key=lambda row: row["validation_pr_auc"], reverse=True)[:top_n]


def run_shap_explainability(
    config: RunConfig,
    logger: DecisionLogger,
    candidates: list[dict],
    fitted_registry: dict[str, dict],
) -> None:
    print("[05] Running SHAP explainability...", flush=True)
    try:
        import shap
    except Exception as exc:
        logger.write("05 SHAP Explainability", f"SHAP unavailable: {exc}")
        return

    rows = []
    failures = []
    for candidate in top_validation_candidates(candidates, config.shap_top_n):
        model_name = candidate["model"]
        info = fitted_registry.get(model_name)
        if info is None:
            failures.append({"model": model_name, "reason": "missing fitted registry entry"})
            continue

        try:
            X_sample = sample_frame(info["X_valid"].assign(**{TARGET: info["y_valid"].to_numpy()}), config.shap_sample_rows)
            y_sample = X_sample[TARGET].copy()
            X_sample = X_sample.drop(columns=[TARGET])
            if info["model_kind"] == "catboost_native":
                prepared = info["fitted"]["builder"].transform(X_sample).drop(columns=["month"], errors="ignore")
                for column in info["fitted"]["cat_cols"]:
                    if column in prepared.columns:
                        prepared[column] = prepared[column].fillna("Unknown").astype(str)
                explainer = shap.TreeExplainer(info["fitted"]["model"])
                shap_values = explainer.shap_values(prepared)
                feature_names = prepared.columns
            elif isinstance(info["fitted"], (Pipeline, ImbPipeline)):
                final_model = info["fitted"].steps[-1][1]
                prepared, feature_names = transformed_features_for_explainability(
                    info["fitted"],
                    X_sample,
                )
                model_class = final_model.__class__.__name__
                if model_class in [
                    "XGBClassifier",
                    "LGBMClassifier",
                    "RandomForestClassifier",
                    "DecisionTreeClassifier",
                ]:
                    explainer = shap.TreeExplainer(final_model)
                    shap_values = explainer.shap_values(prepared)
                elif model_class == "LogisticRegression":
                    explainer = shap.LinearExplainer(final_model, prepared)
                    shap_values = explainer.shap_values(prepared)
                else:
                    failures.append({"model": model_name, "reason": f"unsupported final model {model_class}"})
                    continue
            else:
                failures.append({"model": model_name, "reason": "unsupported model object for SHAP"})
                continue

            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
            if hasattr(shap_values, "values"):
                shap_values = shap_values.values
            shap_abs = np.abs(shap_values).mean(axis=0)
            summary = (
                pd.DataFrame({"feature": feature_names, "mean_abs_shap": shap_abs})
                .sort_values("mean_abs_shap", ascending=False)
                .head(40)
            )
            path = RESULTS_DIR / f"05_shap_top_features_{safe_name(model_name)}.csv"
            summary.to_csv(path, index=False)
            for _, row in summary.head(15).iterrows():
                rows.append(
                    {
                        "model": model_name,
                        "feature": row["feature"],
                        "mean_abs_shap": row["mean_abs_shap"],
                    }
                )
        except Exception as exc:
            failures.append({"model": model_name, "reason": repr(exc)})

    if rows:
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "05_shap_top_features_summary.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(RESULTS_DIR / "05_shap_failures.csv", index=False)

    logger.write(
        "05 SHAP Explainability",
        f"Attempted SHAP on top {config.shap_top_n} validation candidates. "
        f"Successful model-feature rows: {len(rows)}. Failures: {len(failures)}.",
    )


def transformed_features_for_explainability(fitted, X_sample: pd.DataFrame):
    X_work = X_sample.copy()
    feature_names = None
    for step_name, step in fitted.steps[:-1]:
        if hasattr(step, "transform"):
            X_work = step.transform(X_work)
            if hasattr(step, "get_feature_names_out"):
                try:
                    feature_names = step.get_feature_names_out()
                except Exception:
                    feature_names = None
            elif isinstance(X_work, pd.DataFrame):
                feature_names = X_work.columns.astype(str)
        elif hasattr(step, "fit_resample"):
            # Samplers are used only during fit. They must not alter validation rows.
            continue

    if hasattr(X_work, "toarray"):
        X_work = X_work.toarray()
    prepared = pd.DataFrame(X_work)
    if feature_names is None or len(feature_names) != prepared.shape[1]:
        feature_names = prepared.columns.astype(str)
    else:
        feature_names = pd.Index(feature_names).astype(str)
    prepared.columns = feature_names
    return prepared, prepared.columns


def refit_candidate_for_fairness(
    spec: dict,
    context: DataContext,
    config: RunConfig,
    drop_sensitive: bool,
):
    spec_type = spec.get("type")
    family = spec.get("model_family")
    if spec_type == "baseline_randomsearch":
        drop_columns = baseline_drop_columns(context, drop_sensitive=drop_sensitive)
        X_train = make_raw_features(context.train_sample, drop_columns)
        y_train = context.train_sample[TARGET].copy()
        X_valid = make_raw_features(context.valid_eval, drop_columns)
        X_test = make_raw_features(context.test_eval, drop_columns)
        space = make_baseline_spaces(context.scale_pos_weight, 1)[family]
        model = clone(space["estimator"])
        model.set_params(**spec.get("best_params", {}))
        model.fit(X_train, y_train)
        return model, "pipeline", X_valid, X_test

    if spec_type in ["catboost_native"]:
        drop_columns = advanced_drop_columns(context, drop_sensitive=drop_sensitive)
        X_train = make_raw_features(context.train_sample, drop_columns)
        y_train = context.train_sample[TARGET].copy()
        X_valid = make_raw_features(context.valid_eval, drop_columns)
        y_valid = context.valid_eval[TARGET].copy()
        X_test = make_raw_features(context.test_eval, drop_columns)
        fitted = fit_catboost_native(
            X_train,
            y_train,
            X_valid,
            y_valid,
            context.scale_pos_weight,
            params=spec.get("params", {}),
        )
        return fitted, "catboost_native", X_valid, X_test

    if spec_type == "balance_gate":
        drop_columns = baseline_drop_columns(context, drop_sensitive=drop_sensitive)
        X_train = make_raw_features(context.train_sample, drop_columns)
        y_train = context.train_sample[TARGET].copy()
        X_valid = make_raw_features(context.valid_eval, drop_columns)
        X_test = make_raw_features(context.test_eval, drop_columns)
        model = make_balanced_pipeline(
            family,
            spec.get("balance_policy", "no_balance"),
            context.scale_pos_weight,
        )
        model.fit(X_train, y_train)
        return model, "pipeline", X_valid, X_test

    if spec_type in ["advanced_gate", "voting_baseline", "stacking_baseline"]:
        drop_columns = advanced_drop_columns(context, drop_sensitive=drop_sensitive)
        X_train = make_raw_features(context.train_sample, drop_columns)
        y_train = context.train_sample[TARGET].copy()
        X_valid = make_raw_features(context.valid_eval, drop_columns)
        X_test = make_raw_features(context.test_eval, drop_columns)
        mapped_family = family
        if spec_type == "voting_baseline":
            mapped_family = "Voting"
        if spec_type == "stacking_baseline":
            mapped_family = "Stacking"
        model, kind = make_advanced_model(mapped_family, context.scale_pos_weight)
        if model is None:
            return None, None, None, None
        model = clone(model)
        model.fit(X_train, y_train)
        return model, kind, X_valid, X_test

    if spec_type == "anomaly_recency_gate":
        strategy = spec.get("train_strategy", "full_0_5")
        train_frame, sample_weight = make_train_strategy_frame(context, config, strategy)
        y_train = train_frame[TARGET].copy()
        X_train = make_raw_features(train_frame, advanced_drop_columns(context, drop_sensitive=drop_sensitive))
        X_valid = make_raw_features(context.valid_eval, advanced_drop_columns(context, drop_sensitive=drop_sensitive))
        X_test = make_raw_features(context.test_eval, advanced_drop_columns(context, drop_sensitive=drop_sensitive))
        if spec.get("use_anomaly", False):
            appender = AnomalyScoreAppender(
                legit_sample_rows=config.anomaly_legit_rows,
                lof_sample_rows=config.lof_legit_rows,
                autoencoder_sample_rows=config.autoencoder_legit_rows,
            )
            appender.fit(X_train, y_train)
            X_train = appender.transform(X_train)
            X_valid = appender.transform(X_valid)
            X_test = appender.transform(X_test)
        spw = scale_pos_weight_from_y(y_train, sample_weight)
        if family == "CatBoost":
            fitted = fit_catboost_native(
                X_train,
                y_train,
                X_valid,
                context.valid_eval[TARGET].copy(),
                spw,
                sample_weight=sample_weight,
            )
            return fitted, "catboost_native", X_valid, X_test
        model, kind = make_advanced_model(family, spw)
        if model is None:
            return None, None, None, None
        model = clone(model)
        fit_params = {}
        if sample_weight is not None and family not in ["Stacking", "Voting"]:
            fit_params["model__sample_weight"] = sample_weight
        model.fit(X_train, y_train, **fit_params)
        return model, kind, X_valid, X_test

    return None, None, None, None


def run_housing_fairness_audit(
    context: DataContext,
    config: RunConfig,
    logger: DecisionLogger,
    candidates: list[dict],
) -> None:
    print("[06] Running housing_status fairness audit...", flush=True)
    top = top_validation_candidates(candidates, config.fairness_top_n)
    overall_rows = []
    group_rows = []
    y_valid = context.valid_eval[TARGET].copy()
    y_test = context.test_eval[TARGET].copy()

    for candidate in top:
        for feature_policy, drop_sensitive in [
            ("with_housing_status", False),
            ("without_housing_status", True),
        ]:
            fitted, kind, X_valid, X_test = refit_candidate_for_fairness(
                candidate["spec"],
                context,
                config,
                drop_sensitive=drop_sensitive,
            )
            if fitted is None:
                continue
            valid_scores = score_fitted_model(fitted, X_valid, kind)
            test_scores = score_fitted_model(fitted, X_test, kind)
            threshold = threshold_at_fpr(y_valid, valid_scores, max_fpr=MAIN_FPR_CAP)
            metrics = evaluate_scores(y_test, test_scores, threshold=threshold)
            overall_rows.append(
                {
                    "source_model": candidate["model"],
                    "model_family": candidate["model_family"],
                    "feature_policy": feature_policy,
                    "threshold_policy": "valid_global_5pct_fpr",
                    "threshold": threshold,
                    **metrics,
                }
            )
            group_metrics = evaluate_group_metrics(
                y_test,
                test_scores,
                context.test_eval[SENSITIVE_COLUMN],
                threshold=threshold,
            )
            group_metrics["source_model"] = candidate["model"]
            group_metrics["model_family"] = candidate["model_family"]
            group_metrics["feature_policy"] = feature_policy
            group_metrics["threshold_policy"] = "valid_global_5pct_fpr"
            group_metrics["threshold"] = threshold
            group_rows.append(group_metrics)

    if not overall_rows:
        logger.write("06 Housing Status Fairness", "No fairness rows were produced.")
        return

    overall = pd.DataFrame(overall_rows)
    groups = pd.concat(group_rows, ignore_index=True)
    overall.to_csv(RESULTS_DIR / "06_housing_status_overall_metrics.csv", index=False)
    groups.to_csv(RESULTS_DIR / "06_housing_status_group_audit.csv", index=False)

    delta_rows = []
    for (source_model, model_family), frame in overall.groupby(["source_model", "model_family"]):
        pivot = frame.set_index("feature_policy")
        if {"with_housing_status", "without_housing_status"}.issubset(pivot.index):
            with_row = pivot.loc["with_housing_status"]
            without_row = pivot.loc["without_housing_status"]
            delta_rows.append(
                {
                    "source_model": source_model,
                    "model_family": model_family,
                    "precision_delta_with_minus_without": with_row["precision"] - without_row["precision"],
                    "recall_tpr_delta_with_minus_without": with_row["recall_tpr"] - without_row["recall_tpr"],
                    "fpr_delta_with_minus_without": with_row["fpr"] - without_row["fpr"],
                    "pr_auc_delta_with_minus_without": with_row["pr_auc"] - without_row["pr_auc"],
                    "roc_auc_delta_with_minus_without": with_row["roc_auc"] - without_row["roc_auc"],
                }
            )
    delta = pd.DataFrame(delta_rows)
    if not delta.empty:
        delta.to_csv(RESULTS_DIR / "06_housing_status_overall_delta.csv", index=False)

    ba_rows = groups[groups["housing_status"] == "BA"].copy()
    report = f"""# Housing Status Fairness Audit - Progressive Holistic Run

Top candidates audited: {len(top)}

Threshold policy: choose one global threshold on validation with FPR <= 5%, then
apply that threshold to test.

## Overall Metrics

{markdown_table(overall[['source_model', 'model_family', 'feature_policy', 'precision', 'recall_tpr', 'fpr', 'pr_auc', 'roc_auc']].round(6))}

## Overall Delta With Minus Without Housing Status

{markdown_table(delta.round(6)) if not delta.empty else '_No paired rows available._'}

## BA Group Rows

{markdown_table(ba_rows[['source_model', 'model_family', 'feature_policy', 'n', 'fraud_count', 'alert_rate', 'precision', 'recall_tpr', 'fpr', 'fnr']].round(6))}

## Interpretation

This is diagnostic, not deployment approval. Compare predictive lift with
group-level FPR/FNR shifts before recommending use of `housing_status`.
"""
    (RESULTS_DIR / "06_housing_status_fairness_report.md").write_text(report, encoding="utf-8")
    logger.write(
        "06 Housing Status Fairness",
        f"Audited {len(top)} top candidates with and without `housing_status`. "
        "Saved overall, group, delta, and markdown report artifacts.",
    )


def run_calibration_check(
    logger: DecisionLogger,
    candidates: list[dict],
    fitted_registry: dict[str, dict],
) -> None:
    best = top_validation_candidates(candidates, 1)
    if not best:
        return
    best = best[0]
    info = fitted_registry[best["model"]]
    valid_scores = info["valid_scores"]
    test_scores = info["test_scores"]
    y_valid = info["y_valid"]
    y_test = info["y_test"]
    calibrator = LogisticRegression(max_iter=300)
    calibrator.fit(logit(np.clip(valid_scores, 1e-6, 1 - 1e-6)).reshape(-1, 1), y_valid)
    valid_cal = calibrator.predict_proba(logit(np.clip(valid_scores, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]
    test_cal = calibrator.predict_proba(logit(np.clip(test_scores, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]

    rows = []
    for split, y_true, raw, cal in [
        ("validation", y_valid, valid_scores, valid_cal),
        ("test", y_test, test_scores, test_cal),
    ]:
        for score_type, scores in [("raw", raw), ("platt_calibrated_on_validation", cal)]:
            rows.append(
                {
                    "model": best["model"],
                    "split": split,
                    "score_type": score_type,
                    "brier_score": brier_score_loss(y_true, scores),
                    "mean_score": float(np.mean(scores)),
                    "observed_fraud_rate": float(np.mean(y_true)),
                    "pr_auc": average_precision_score(y_true, scores),
                    "roc_auc": roc_auc_score(y_true, scores),
                }
            )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "07_calibration_comparison.csv", index=False)
    logger.write(
        "07 Calibration",
        f"Platt scaling check saved for best validation candidate: `{best['model']}`.",
    )


def write_all_outputs(
    config: RunConfig,
    context: DataContext,
    all_metrics: list[dict],
    candidates: list[dict],
) -> None:
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(RESULTS_DIR / "holistic_all_metrics.csv", index=False)
    main = metrics[metrics["threshold_policy"] == "valid_global_5pct_fpr"].copy()
    main[main["split"] == "validation"].to_csv(
        RESULTS_DIR / "holistic_validation_main_threshold.csv",
        index=False,
    )
    main[main["split"] == "test"].to_csv(
        RESULTS_DIR / "holistic_test_main_threshold.csv",
        index=False,
    )
    candidate_frame = pd.DataFrame([{k: v for k, v in row.items() if k != "spec"} for row in candidates])
    candidate_frame.to_csv(RESULTS_DIR / "holistic_candidate_ranking.csv", index=False)
    specs = {
        row["model"]: row["spec"]
        for row in candidates
    }
    (RESULTS_DIR / "holistic_candidate_specs.json").write_text(
        json.dumps(specs, indent=2, default=str),
        encoding="utf-8",
    )
    metadata = {
        "config": asdict(config),
        "train_months": context.train_months,
        "valid_month": context.valid_month,
        "test_month": context.test_month,
        "unusable_columns": context.unusable_columns,
        "threshold_policy": f"validation global FPR <= {MAIN_FPR_CAP:.2%}",
    }
    (RESULTS_DIR / "holistic_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )


def write_final_report(logger: DecisionLogger, candidates: list[dict]) -> None:
    ranked = pd.DataFrame([{k: v for k, v in row.items() if k != "spec"} for row in candidates])
    ranked = ranked.sort_values("validation_pr_auc", ascending=False)
    best_validation = ranked.iloc[0]
    best_test = ranked.sort_values("test_pr_auc", ascending=False).iloc[0]

    report = f"""# Holistic Fraud Modeling Report

This run uses a progressive gate design rather than a full cartesian product.

## Best Validation Candidate

- Model: `{best_validation['model']}`
- Validation PR-AUC: {best_validation['validation_pr_auc']:.6f}
- Validation ROC-AUC: {best_validation['validation_roc_auc']:.6f}
- Test PR-AUC: {best_validation['test_pr_auc']:.6f}
- Test ROC-AUC: {best_validation['test_roc_auc']:.6f}

## Best Test PR-AUC Candidate

- Model: `{best_test['model']}`
- Validation PR-AUC: {best_test['validation_pr_auc']:.6f}
- Test PR-AUC: {best_test['test_pr_auc']:.6f}
- Test ROC-AUC: {best_test['test_roc_auc']:.6f}

## Top 15 Candidates

{markdown_table(ranked[['model', 'stage', 'model_family', 'feature_set', 'balance_policy', 'train_strategy', 'anomaly_policy', 'validation_pr_auc', 'test_pr_auc', 'test_roc_auc']].head(15).round(6))}

## Main Artifacts

- `progressive_decision_log.md`
- `holistic_all_metrics.csv`
- `holistic_candidate_ranking.csv`
- `01_baseline_candidates.csv`
- `02_balancing_candidates.csv`
- `03_advanced_candidates.csv`
- `04_anomaly_recency_candidates.csv`
- `05_shap_top_features_summary.csv`
- `06_housing_status_fairness_report.md`
- `07_calibration_comparison.csv`
"""
    (RESULTS_DIR / "holistic_report.md").write_text(report, encoding="utf-8")
    logger.write(
        "08 Final Report",
        "Final report written to `holistic_report.md`.",
    )


def reset_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in RESULTS_DIR.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    reset_results_dir()
    config = config_for_mode(args.mode)
    logger = DecisionLogger(RESULTS_DIR / "progressive_decision_log.md")
    logger.write(
        "Run Config",
        "```json\n" + json.dumps(asdict(config), indent=2) + "\n```",
    )

    context = load_context(config, logger)
    all_metrics: list[dict] = []
    fitted_registry: dict[str, dict] = {}

    baseline_candidates = run_randomized_baselines(
        context,
        config,
        logger,
        all_metrics,
        fitted_registry,
    )
    balancing_candidates = run_balancing_gate(
        context,
        config,
        logger,
        all_metrics,
        fitted_registry,
        baseline_candidates,
    )
    candidates_after_balance = baseline_candidates + balancing_candidates
    advanced_candidates = run_advanced_feature_gate(
        context,
        config,
        logger,
        all_metrics,
        fitted_registry,
        candidates_after_balance,
    )
    candidates_after_advanced = candidates_after_balance + advanced_candidates
    anomaly_candidates = run_anomaly_recency_gate(
        context,
        config,
        logger,
        all_metrics,
        fitted_registry,
        candidates_after_advanced,
    )
    all_candidates = candidates_after_advanced + anomaly_candidates

    write_all_outputs(config, context, all_metrics, all_candidates)
    run_shap_explainability(config, logger, all_candidates, fitted_registry)
    run_housing_fairness_audit(context, config, logger, all_candidates)
    run_calibration_check(logger, all_candidates, fitted_registry)
    write_final_report(logger, all_candidates)

    print(f"Saved holistic artifacts in: {RESULTS_DIR}")
    print("Top validation candidates:")
    ranked = pd.DataFrame([{k: v for k, v in row.items() if k != "spec"} for row in all_candidates])
    print(
        ranked.sort_values("validation_pr_auc", ascending=False)
        .head(10)[["model", "validation_pr_auc", "test_pr_auc", "test_roc_auc"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
