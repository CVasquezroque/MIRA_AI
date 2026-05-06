from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier

from model.full_holistic.constants import MAIN_FPR_CAP, RANDOM_STATE, TARGET
from model.full_holistic.data.context import sample_frame
from model.full_holistic.features.engineering import (
    AdvancedFeatureBuilder,
    DataFrameMedianImputer,
    DataFrameRobustScaler,
    TemporalTargetFrequencyEncoder,
    categorical_columns,
    make_raw_features,
    numeric_columns,
)
from model.full_holistic.utils.metrics import compute_threshold_metrics, safe_rate
from model.full_holistic.utils.thresholds import threshold_at_fpr


def fit_with_filtered_warnings(estimator, X, y=None, **fit_kwargs):
    """Fit estimators while suppressing expected library noise.

    The pipeline logs metrics and stage artifacts explicitly. These filters only
    silence non-fatal sklearn/LightGBM warnings that otherwise spam long runs.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        )
        if y is None:
            return estimator.fit(X, **fit_kwargs)
        return estimator.fit(X, y, **fit_kwargs)


def make_onehot_preprocessor(X: pd.DataFrame, scale_numeric: bool = False, dense: bool = False) -> ColumnTransformer:
    cats = categorical_columns(X)
    nums = numeric_columns(X)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", RobustScaler()))
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=not dense)
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), nums),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)]), cats),
        ],
        remainder="drop",
        sparse_threshold=0.0 if dense else 0.3,
    )


def make_model_family(model_family: str, scale_pos_weight: float = 1.0, random_state: int = RANDOM_STATE):
    if model_family == "Logistic Regression":
        return LogisticRegression(
            class_weight="balanced",
            C=0.05,
            max_iter=2000,
            solver="lbfgs",
            random_state=random_state,
        )
    if model_family == "Decision Tree":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=80, class_weight="balanced", random_state=random_state)
    if model_family == "Random Forest":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=60,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_family == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_estimators=160,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.88,
            colsample_bytree=0.78,
            min_child_weight=3,
            gamma=1.5,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_family == "LightGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            n_estimators=150,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            min_child_samples=180,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=4.0,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=4,
            verbose=-1,
            force_col_wise=True,
        )
    raise ValueError(f"Unsupported model family: {model_family}")


def make_baseline_pipeline(model_family: str, X_reference: pd.DataFrame, scale_pos_weight: float) -> Pipeline:
    model = make_model_family(model_family, scale_pos_weight)
    return Pipeline([("preprocess", make_onehot_preprocessor(X_reference, scale_numeric=model_family == "Logistic Regression")), ("model", model)])


def make_advanced_pipeline(model_family: str, scale_pos_weight: float) -> Pipeline:
    model = make_model_family(model_family, scale_pos_weight)
    steps = [
        ("features", AdvancedFeatureBuilder()),
        ("target_frequency", TemporalTargetFrequencyEncoder()),
        ("imputer", DataFrameMedianImputer()),
    ]
    if model_family == "Logistic Regression":
        steps.append(("scaler", DataFrameRobustScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _prepare_catboost_native_frame(
    X: pd.DataFrame,
    *,
    builder: AdvancedFeatureBuilder | None,
    fit_builder: bool = False,
    y=None,
) -> tuple[pd.DataFrame, list[str]]:
    if builder is None:
        transformed = pd.DataFrame(X).copy()
    else:
        transformed = builder.fit_transform(X, y) if fit_builder else builder.transform(X)
    transformed = transformed.drop(columns=["month"], errors="ignore")
    cats = categorical_columns(transformed)
    for column in cats:
        transformed[column] = transformed[column].fillna("Unknown").astype(str)
    return transformed, cats


def fit_catboost_native(
    X_train,
    y_train,
    X_valid,
    y_valid,
    scale_pos_weight: float,
    sample_weight=None,
    random_state: int = RANDOM_STATE,
    use_advanced_features: bool = True,
):
    from catboost import CatBoostClassifier

    builder = AdvancedFeatureBuilder() if use_advanced_features else None
    X_train_cb, cat_cols = _prepare_catboost_native_frame(X_train, builder=builder, fit_builder=True, y=y_train)
    X_valid_cb, _ = _prepare_catboost_native_frame(X_valid, builder=builder)
    model = CatBoostClassifier(
        iterations=220,
        depth=5,
        learning_rate=0.06,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        eval_metric="PRAUC",
        scale_pos_weight=scale_pos_weight,
        random_seed=random_state,
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
        early_stopping_rounds=40,
    )
    return {
        "model": model,
        "builder": builder,
        "cat_cols": cat_cols,
        "feature_columns": X_train_cb.columns.astype(str).tolist(),
        "catboost_feature_mode": "advanced_feature_builder" if use_advanced_features else "raw_native_features",
        "drop_columns": ["month"],
    }


def score_model(fitted, X: pd.DataFrame, model_kind: str):
    if model_kind == "catboost_native":
        builder = fitted.get("builder")
        if builder is None:
            X_cb = pd.DataFrame(X).copy()
        else:
            X_cb = builder.transform(X)
        X_cb = X_cb.drop(columns=fitted.get("drop_columns", ["month"]), errors="ignore")
        feature_columns = fitted.get("feature_columns")
        if feature_columns is not None:
            X_cb = X_cb.reindex(columns=feature_columns)
        for column in fitted["cat_cols"]:
            if column in X_cb.columns:
                X_cb[column] = X_cb[column].fillna("Unknown").astype(str)
        proba = fitted["model"].predict_proba(X_cb)
        classes = getattr(fitted["model"], "classes_", None)
        if classes is not None and 1 in list(classes):
            return proba[:, list(classes).index(1)]
        return proba[:, 1]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        )
        if hasattr(fitted, "predict_proba"):
            proba = fitted.predict_proba(X)
            classes = getattr(fitted, "classes_", None)
            if classes is None and hasattr(fitted, "named_steps"):
                classes = getattr(fitted.steps[-1][1], "classes_", None)
            if classes is not None and 1 in list(classes):
                return proba[:, list(classes).index(1)]
            return proba[:, 1]
        return fitted.decision_function(X)


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
    runtime_seconds: float | None = None,
) -> dict:
    valid_scores = score_model(fitted, X_valid, model_kind)
    test_scores = score_model(fitted, X_test, model_kind)
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
    for split_name, y_true, scores in [("validation", y_valid, valid_scores), ("test", y_test, test_scores)]:
        for policy, threshold in [("default_0_50", 0.50), ("valid_global_5pct_fpr", selected_threshold)]:
            all_metrics.append({**common, "split": split_name, "threshold_policy": policy, **compute_threshold_metrics(y_true, scores, threshold)})
    validation_pr_auc = float(average_precision_score(y_valid, valid_scores))
    test_pr_auc = float(average_precision_score(y_test, test_scores))
    candidate = {
        **common,
        "selected_threshold": float(selected_threshold),
        "validation_pr_auc": validation_pr_auc,
        "validation_roc_auc": float(roc_auc_score(y_valid, valid_scores)),
        "validation_pr_auc_lift": safe_rate(validation_pr_auc, float(pd.Series(y_valid).mean())),
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "test_pr_auc_lift": safe_rate(test_pr_auc, float(pd.Series(y_test).mean())),
        "runtime_seconds": runtime_seconds,
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


def train_candidate_for_family(context, config, model_family: str, *, feature_mode: str, stage: str, label_prefix: str, balance_policy: str = "model_default_weighting", train_frame: pd.DataFrame | None = None, sample_weight=None):
    train_frame = sample_frame(context.train, config.train_rows) if train_frame is None else train_frame
    y_train = train_frame[TARGET].copy()
    y_valid = context.valid_eval[TARGET].copy()
    y_test = context.test_eval[TARGET].copy()
    X_train = make_raw_features(train_frame)
    X_valid = make_raw_features(context.valid_eval)
    X_test = make_raw_features(context.test_eval)
    started = time.time()
    if model_family == "CatBoost":
        fitted = fit_catboost_native(
            X_train,
            y_train,
            X_valid,
            y_valid,
            context.scale_pos_weight,
            sample_weight=sample_weight,
            use_advanced_features=feature_mode == "advanced",
        )
        kind = "catboost_native"
    else:
        pipeline = make_baseline_pipeline(model_family, X_train, context.scale_pos_weight) if feature_mode == "baseline" else make_advanced_pipeline(model_family, context.scale_pos_weight)
        fitted = clone(pipeline)
        fit_kwargs = {}
        if sample_weight is not None and feature_mode == "advanced":
            fit_kwargs["model__sample_weight"] = sample_weight
        fit_with_filtered_warnings(fitted, X_train, y_train, **fit_kwargs)
        kind = "pipeline"
    runtime = time.time() - started
    if model_family == "CatBoost":
        feature_set = "catboost_native_raw_no_generated_features" if feature_mode == "baseline" else "catboost_native_missing_log_ratio_interactions"
    else:
        feature_set = "baseline_eda_onehot" if feature_mode == "baseline" else "ratios_interactions_target_frequency"
    return {
        "model_name": f"{label_prefix} | {model_family}",
        "stage": stage,
        "model_family": model_family,
        "feature_set": feature_set,
        "balance_policy": balance_policy,
        "train_strategy": "full_0_5" if config.train_rows is None else f"sample_{config.train_rows}",
        "anomaly_policy": "without_anomaly_scores",
        "fitted": fitted,
        "model_kind": kind,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test,
        "runtime_seconds": runtime,
        "spec": {
            "type": stage,
            "model_family": model_family,
            "feature_mode": feature_mode,
            "feature_set": feature_set,
            "catboost_feature_mode": fitted.get("catboost_feature_mode") if model_family == "CatBoost" else None,
            "balance_policy": balance_policy,
        },
    }
