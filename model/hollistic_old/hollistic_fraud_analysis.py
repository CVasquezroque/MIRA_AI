"""
Holistic fraud modeling analysis.

This script consolidates the previous work and adds two new experiment families:

1. Anomaly scores as features:
   - Isolation Forest score
   - Local Outlier Factor novelty score
   - simple autoencoder-style reconstruction error with MLPRegressor

2. Temporal drift / recency strategies:
   - train months 0-5, unweighted
   - train months 0-5, more weight on recent months
   - train months 3-5 only

The output is intentionally analytical rather than production-oriented. It writes
comparison tables, curves, confusion matrices, SHAP beeswarm plots, calibration
diagnostics, and a markdown report to model/results/hollistic.
"""

from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import logit
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


MODEL_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = MODEL_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from advanced_feature_modeling import (  # noqa: E402
    RANDOM_STATE,
    TARGET,
    AdvancedFeatureBuilder,
    DataFrameMedianImputer,
    DataFrameRobustScaler,
    TemporalTargetFrequencyEncoder,
    catboost_scores,
    categorical_columns,
    evaluate_scores,
    fit_catboost,
    make_lightgbm,
    make_raw_features,
    make_target_frequency_pipeline,
    make_xgboost,
    model_scores,
    split_before_preprocessing,
    stratified_training_sample,
    threshold_at_fpr,
)
from housing_status_fairness_audit import evaluate_group_metrics, markdown_table  # noqa: E402


RESULTS_DIR = MODEL_DIR / "results" / "hollistic"
FIGURES_DIR = RESULTS_DIR / "figures"
TRAIN_SAMPLE_MAX_ROWS = 180_000
ANOMALY_LEGIT_SAMPLE_ROWS = 25_000
LOF_LEGIT_SAMPLE_ROWS = 15_000
AUTOENCODER_LEGIT_SAMPLE_ROWS = 18_000
SHAP_SAMPLE_ROWS = 1_500
PERMUTATION_THRESHOLD_FPR = 0.05
SENSITIVE_COLUMN = "housing_status"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LocalOutlierFactor was fitted with feature names",
)


class AnomalyScoreAppender(BaseEstimator, TransformerMixin):
    """Fit anomaly detectors on legitimate training rows and append scores."""

    def __init__(
        self,
        legit_sample_rows=ANOMALY_LEGIT_SAMPLE_ROWS,
        lof_sample_rows=LOF_LEGIT_SAMPLE_ROWS,
        autoencoder_sample_rows=AUTOENCODER_LEGIT_SAMPLE_ROWS,
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

        iso_sample = self._sample_rows(legit, self.legit_sample_rows)
        lof_sample = self._sample_rows(legit, self.lof_sample_rows)
        autoencoder_sample = self._sample_rows(legit, self.autoencoder_sample_rows)

        self.isolation_forest_ = IsolationForest(
            n_estimators=90,
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
        self.autoencoder_.fit(autoencoder_sample, autoencoder_sample)
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

    def _sample_rows(self, frame, max_rows):
        if len(frame) <= max_rows:
            return frame
        return frame.sample(n=max_rows, random_state=RANDOM_STATE)


def markdown_percent(value):
    if pd.isna(value):
        return ""
    return f"{value:.3%}"


def stratified_sample(frame, max_rows):
    if len(frame) <= max_rows:
        return frame.copy()
    sample, _ = train_test_split(
        frame,
        train_size=max_rows,
        stratify=frame[TARGET],
        random_state=RANDOM_STATE,
    )
    return sample.copy()


def make_train_strategy_frame(train, strategy):
    if strategy == "full_0_5":
        sample = stratified_sample(train, TRAIN_SAMPLE_MAX_ROWS)
        sample_weight = None
    elif strategy == "full_0_5_recency_weighted":
        sample = stratified_sample(train, TRAIN_SAMPLE_MAX_ROWS)
        months = sample["month"].astype(float)
        min_month = months.min()
        max_month = months.max()
        span = max(max_month - min_month, 1.0)
        sample_weight = 0.5 + (months - min_month) / span
        sample_weight = sample_weight / sample_weight.mean()
        sample_weight = sample_weight.to_numpy()
    elif strategy == "recent_3_5":
        recent = train[train["month"].isin([3, 4, 5])].copy()
        sample = stratified_sample(recent, TRAIN_SAMPLE_MAX_ROWS)
        sample_weight = None
    else:
        raise ValueError(f"Unknown train strategy: {strategy}")
    return sample, sample_weight


def add_anomaly_scores_if_needed(X_train, y_train, X_valid, X_test, use_anomaly):
    if not use_anomaly:
        return X_train.copy(), X_valid.copy(), X_test.copy(), None

    print("  Fitting anomaly score features on legitimate training rows...")
    appender = AnomalyScoreAppender()
    appender.fit(X_train, y_train)
    return (
        appender.transform(X_train),
        appender.transform(X_valid),
        appender.transform(X_test),
        appender,
    )


def model_scale_pos_weight(y_train, sample_weight=None):
    y = pd.Series(y_train).to_numpy()
    if sample_weight is None:
        neg = (y == 0).sum()
        pos = (y == 1).sum()
    else:
        weights = np.asarray(sample_weight)
        neg = weights[y == 0].sum()
        pos = weights[y == 1].sum()
    return float(neg / max(pos, 1e-12))


def build_supervised_model(model_family, scale_pos_weight):
    if model_family == "XGBoost":
        model = make_xgboost(standard=True, scale_pos_weight=scale_pos_weight)
        model.set_params(n_estimators=150)
        return make_target_frequency_pipeline(model)

    if model_family == "LightGBM":
        model = make_lightgbm(scale_pos_weight=scale_pos_weight)
        model.set_params(n_estimators=140)
        return make_target_frequency_pipeline(model)

    raise ValueError(f"Pipeline model not supported: {model_family}")


def fit_catboost_holistic(X_train, y_train, X_valid, y_valid, scale_pos_weight, sample_weight):
    from catboost import CatBoostClassifier

    builder = AdvancedFeatureBuilder()
    X_train_cb = builder.fit_transform(X_train, y_train).drop(columns=["month"], errors="ignore")
    X_valid_cb = builder.transform(X_valid).drop(columns=["month"], errors="ignore")

    cat_cols = categorical_columns(X_train_cb)
    for column in cat_cols:
        X_train_cb[column] = X_train_cb[column].fillna("Unknown").astype(str)
        X_valid_cb[column] = X_valid_cb[column].fillna("Unknown").astype(str)

    model = CatBoostClassifier(
        iterations=260,
        depth=6,
        learning_rate=0.055,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        eval_metric="PRAUC",
        scale_pos_weight=scale_pos_weight,
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
        early_stopping_rounds=45,
    )
    return {"model": model, "builder": builder, "cat_cols": cat_cols}


def catboost_holistic_scores(fitted, X):
    X_cb = fitted["builder"].transform(X).drop(columns=["month"], errors="ignore")
    for column in fitted["cat_cols"]:
        if column in X_cb.columns:
            X_cb[column] = X_cb[column].fillna("Unknown").astype(str)
    return fitted["model"].predict_proba(X_cb)[:, 1]


def fit_and_score_model(
    model_family,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    scale_pos_weight,
    sample_weight,
):
    if model_family == "CatBoost":
        fitted = fit_catboost_holistic(
            X_train,
            y_train,
            X_valid,
            y_valid,
            scale_pos_weight,
            sample_weight,
        )
        valid_scores = catboost_holistic_scores(fitted, X_valid)
        test_scores = catboost_holistic_scores(fitted, X_test)
    else:
        fitted = build_supervised_model(model_family, scale_pos_weight)
        fit_params = {}
        if sample_weight is not None:
            fit_params["model__sample_weight"] = sample_weight
        fitted.fit(X_train, y_train, **fit_params)
        valid_scores = model_scores(fitted, X_valid)
        test_scores = model_scores(fitted, X_test)

    return fitted, valid_scores, test_scores


def evaluate_at_thresholds(y_valid, valid_scores, y_test, test_scores):
    threshold = threshold_at_fpr(y_valid, valid_scores, max_fpr=PERMUTATION_THRESHOLD_FPR)
    valid_metrics = evaluate_scores(y_valid, valid_scores, threshold=0.50)
    valid_metrics["threshold"] = 0.50
    valid_metrics["threshold_policy"] = "default_0_50"

    valid_fpr_metrics = evaluate_scores(y_valid, valid_scores, threshold=threshold)
    valid_fpr_metrics["threshold"] = threshold
    valid_fpr_metrics["threshold_policy"] = "valid_global_5pct_fpr"

    test_metrics = evaluate_scores(y_test, test_scores, threshold=0.50)
    test_metrics["threshold"] = 0.50
    test_metrics["threshold_policy"] = "default_0_50"

    test_fpr_metrics = evaluate_scores(y_test, test_scores, threshold=threshold)
    test_fpr_metrics["threshold"] = threshold
    test_fpr_metrics["threshold_policy"] = "valid_global_5pct_fpr"
    return [valid_metrics, valid_fpr_metrics], [test_metrics, test_fpr_metrics], threshold


def run_holistic_experiments(train, valid, test):
    unusable_columns = [
        column
        for column in train.columns
        if column != TARGET and train[column].nunique(dropna=False) <= 1
    ]
    print(f"Unusable constant columns removed: {unusable_columns}")

    X_valid_raw = make_raw_features(valid, [TARGET] + unusable_columns)
    y_valid = valid[TARGET].copy()
    X_test_raw = make_raw_features(test, [TARGET] + unusable_columns)
    y_test = test[TARGET].copy()

    model_families = ["XGBoost", "LightGBM", "CatBoost"]
    train_strategies = ["full_0_5", "full_0_5_recency_weighted", "recent_3_5"]
    anomaly_options = [False, True]

    validation_rows = []
    test_rows = []
    score_rows = []
    fitted_models = {}
    anomaly_snapshots = []

    total = len(model_families) * len(train_strategies) * len(anomaly_options)
    index = 0
    for train_strategy in train_strategies:
        train_frame, sample_weight = make_train_strategy_frame(train, train_strategy)
        y_train = train_frame[TARGET].copy()
        X_train_raw = make_raw_features(train_frame, [TARGET] + unusable_columns)

        for use_anomaly in anomaly_options:
            anomaly_label = "with_anomaly_scores" if use_anomaly else "without_anomaly_scores"
            X_train, X_valid, X_test, appender = add_anomaly_scores_if_needed(
                X_train_raw,
                y_train,
                X_valid_raw,
                X_test_raw,
                use_anomaly,
            )
            if use_anomaly:
                anomaly_snapshot = X_train[
                    [
                        "isolation_forest_anomaly_score",
                        "lof_anomaly_score",
                        "autoencoder_reconstruction_error",
                    ]
                ].copy()
                anomaly_snapshot[TARGET] = y_train.to_numpy()
                anomaly_snapshot["train_strategy"] = train_strategy
                anomaly_snapshots.append(anomaly_snapshot.sample(
                    n=min(30_000, len(anomaly_snapshot)),
                    random_state=RANDOM_STATE,
                ))

            spw = model_scale_pos_weight(y_train, sample_weight)
            for model_family in model_families:
                index += 1
                model_name = f"{model_family} | {train_strategy} | {anomaly_label}"
                print(f"\n[{index}/{total}] Training {model_name}...")
                fitted, valid_scores, test_scores = fit_and_score_model(
                    model_family,
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    X_test,
                    scale_pos_weight=spw,
                    sample_weight=sample_weight,
                )

                valid_metric_list, test_metric_list, selected_threshold = evaluate_at_thresholds(
                    y_valid,
                    valid_scores,
                    y_test,
                    test_scores,
                )

                for metrics in valid_metric_list:
                    validation_rows.append(
                        {
                            "model": model_name,
                            "model_family": model_family,
                            "train_strategy": train_strategy,
                            "anomaly_policy": anomaly_label,
                            **metrics,
                        }
                    )
                for metrics in test_metric_list:
                    test_rows.append(
                        {
                            "model": model_name,
                            "model_family": model_family,
                            "train_strategy": train_strategy,
                            "anomaly_policy": anomaly_label,
                            **metrics,
                        }
                    )

                score_rows.append(
                    {
                        "model": model_name,
                        "model_family": model_family,
                        "train_strategy": train_strategy,
                        "anomaly_policy": anomaly_label,
                        "selected_threshold": selected_threshold,
                        "y_valid": y_valid,
                        "valid_scores": valid_scores,
                        "y_test": y_test,
                        "test_scores": test_scores,
                    }
                )
                fitted_models[model_name] = {
                    "fitted": fitted,
                    "X_valid": X_valid,
                    "X_test": X_test,
                    "y_valid": y_valid,
                    "y_test": y_test,
                }

    validation_results = pd.DataFrame(validation_rows)
    test_results = pd.DataFrame(test_rows)
    validation_results.to_csv(RESULTS_DIR / "hollistic_validation_metrics.csv", index=False)
    test_results.to_csv(RESULTS_DIR / "hollistic_test_metrics.csv", index=False)

    if anomaly_snapshots:
        anomaly_snapshot = pd.concat(anomaly_snapshots, ignore_index=True)
        anomaly_snapshot.to_csv(RESULTS_DIR / "anomaly_score_training_snapshot.csv", index=False)
        plot_anomaly_score_distributions(anomaly_snapshot)

    return validation_results, test_results, score_rows, fitted_models, unusable_columns


def plot_anomaly_score_distributions(anomaly_snapshot):
    from matplotlib.patches import Patch

    features = [
        "isolation_forest_anomaly_score",
        "lof_anomaly_score",
        "autoencoder_reconstruction_error",
    ]
    feature_labels = {
        "isolation_forest_anomaly_score": "Isolation Forest",
        "lof_anomaly_score": "Local Outlier Factor",
        "autoencoder_reconstruction_error": "Autoencoder error",
    }
    strategy_labels = {
        "full_0_5": "Train months 0-5",
        "full_0_5_recency_weighted": "0-5, recency weighted",
        "recent_3_5": "Recent months 3-5",
    }
    strategies = [strategy for strategy in strategy_labels if strategy in anomaly_snapshot["train_strategy"].unique()]
    palette = {"legitimate": "#3b6fb6", "fraud": "#e9783a"}

    fig, axes = plt.subplots(
        len(features),
        len(strategies),
        figsize=(6.2 * len(strategies), 4.2 * len(features)),
        sharex=False,
        sharey=False,
    )
    axes = np.atleast_2d(axes)

    for row_index, feature in enumerate(features):
        for col_index, strategy in enumerate(strategies):
            axis = axes[row_index, col_index]
            panel = anomaly_snapshot[anomaly_snapshot["train_strategy"] == strategy].copy()
            panel = panel[[feature, TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
            panel["class_label"] = np.where(panel[TARGET] == 1, "fraud", "legitimate")

            upper = panel[feature].quantile(0.995)
            lower = panel[feature].quantile(0.005)
            panel["score_for_plot"] = panel[feature].clip(lower=lower, upper=upper)

            for class_label in ["legitimate", "fraud"]:
                class_panel = panel[panel["class_label"] == class_label]
                if class_panel["score_for_plot"].nunique() < 2:
                    continue
                sns.kdeplot(
                    data=class_panel,
                    x="score_for_plot",
                    ax=axis,
                    fill=True,
                    alpha=0.28,
                    linewidth=1.5,
                    color=palette[class_label],
                    warn_singular=False,
                )

            if row_index == 0:
                axis.set_title(strategy_labels[strategy], fontsize=13, pad=12)
            if col_index == 0:
                axis.set_ylabel(f"{feature_labels[feature]}\nDensity", fontsize=12)
            else:
                axis.set_ylabel("Density", fontsize=10)
            if row_index == len(features) - 1:
                axis.set_xlabel("Score, clipped to panel 0.5%-99.5%", fontsize=11)
            else:
                axis.set_xlabel("")
            axis.tick_params(axis="both", labelsize=10)

    legend_handles = [
        Patch(facecolor=palette["legitimate"], edgecolor=palette["legitimate"], alpha=0.28, label="legitimate"),
        Patch(facecolor=palette["fraud"], edgecolor=palette["fraud"], alpha=0.28, label="fraud"),
    ]
    fig.legend(
        handles=legend_handles,
        title="Class",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Anomaly Score Distributions by Class", fontsize=20, y=0.995)
    fig.subplots_adjust(top=0.88, hspace=0.38, wspace=0.28)
    fig.savefig(FIGURES_DIR / "anomaly_score_distributions.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparisons(validation_results, test_results, score_rows):
    main_test = test_results[test_results["threshold_policy"] == "valid_global_5pct_fpr"].copy()
    main_valid = validation_results[
        validation_results["threshold_policy"] == "valid_global_5pct_fpr"
    ].copy()

    plt.figure(figsize=(13, 6))
    plot_data = main_test.sort_values("pr_auc", ascending=False)
    sns.barplot(
        data=plot_data,
        x="pr_auc",
        y="model",
        hue="anomaly_policy",
        dodge=False,
        palette="Set2",
    )
    plt.title("Holistic Test PR-AUC by Model and Strategy")
    plt.xlabel("PR-AUC")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hollistic_test_pr_auc_comparison.png", dpi=150)
    plt.close()

    top_rows = sorted(
        score_rows,
        key=lambda row: average_precision_score(row["y_valid"], row["valid_scores"]),
        reverse=True,
    )[:8]

    plt.figure(figsize=(8, 6))
    for row in top_rows:
        fpr, tpr, _ = roc_curve(row["y_test"], row["test_scores"])
        roc_auc = roc_auc_score(row["y_test"], row["test_scores"])
        plt.plot(fpr, tpr, linewidth=1.5, label=f"{row['model'][:36]} AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    plt.title("Top Holistic Models - Test ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hollistic_test_roc_top8.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    for row in top_rows:
        precision, recall, _ = precision_recall_curve(row["y_test"], row["test_scores"])
        pr_auc = average_precision_score(row["y_test"], row["test_scores"])
        plt.plot(recall, precision, linewidth=1.5, label=f"{row['model'][:36]} PR={pr_auc:.3f}")
    plt.title("Top Holistic Models - Test Precision Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hollistic_test_precision_recall_top8.png", dpi=150)
    plt.close()

    plot_confusion_matrices(top_rows)
    main_valid.to_csv(RESULTS_DIR / "hollistic_validation_main_threshold.csv", index=False)
    main_test.to_csv(RESULTS_DIR / "hollistic_test_main_threshold.csv", index=False)


def plot_confusion_matrices(score_rows):
    cols = 2
    rows = int(np.ceil(len(score_rows) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    matrix_rows = []
    for axis, row in zip(axes, score_rows):
        predictions = (row["test_scores"] >= row["selected_threshold"]).astype(int)
        matrix = confusion_matrix(row["y_test"], predictions, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        matrix_rows.append(
            {
                "model": row["model"],
                "threshold": row["selected_threshold"],
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
            ax=axis,
        )
        axis.set_title(row["model"][:54])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")

    for axis in axes[len(score_rows) :]:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hollistic_confusion_matrices_test_top8.png", dpi=150)
    plt.close()
    pd.DataFrame(matrix_rows).to_csv(
        RESULTS_DIR / "hollistic_confusion_matrices_test_top8.csv",
        index=False,
    )


def fit_platt_calibrator(y_calib, scores):
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    features = logit(clipped).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=300)
    calibrator.fit(features, y_calib)
    return calibrator


def calibrated_scores(calibrator, scores):
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    return calibrator.predict_proba(logit(clipped).reshape(-1, 1))[:, 1]


def expected_calibration_error(y_true, scores, bins=10):
    frame = pd.DataFrame({"y": y_true, "score": scores})
    frame["bin"] = pd.qcut(frame["score"], q=bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        n=("y", "size"),
        observed_rate=("y", "mean"),
        mean_score=("score", "mean"),
    )
    return float(
        (grouped["n"] / grouped["n"].sum() * (grouped["observed_rate"] - grouped["mean_score"]).abs()).sum()
    )


def run_calibration(best_row, score_rows, valid, test):
    best_scores = next(row for row in score_rows if row["model"] == best_row["model"])
    calibrator = fit_platt_calibrator(best_scores["y_valid"], best_scores["valid_scores"])
    valid_calibrated = calibrated_scores(calibrator, best_scores["valid_scores"])
    test_calibrated = calibrated_scores(calibrator, best_scores["test_scores"])

    rows = []
    for split_name, y_true, raw_scores, calibrated in [
        ("validation_month_6", best_scores["y_valid"], best_scores["valid_scores"], valid_calibrated),
        ("test_month_7", best_scores["y_test"], best_scores["test_scores"], test_calibrated),
    ]:
        for score_type, scores in [("raw", raw_scores), ("platt_calibrated_on_validation", calibrated)]:
            rows.append(
                {
                    "model": best_row["model"],
                    "split": split_name,
                    "score_type": score_type,
                    "brier_score": brier_score_loss(y_true, scores),
                    "ece_10_bins": expected_calibration_error(y_true, scores, bins=10),
                    "mean_score": float(np.mean(scores)),
                    "observed_fraud_rate": float(np.mean(y_true)),
                    "pr_auc": average_precision_score(y_true, scores),
                    "roc_auc": roc_auc_score(y_true, scores),
                }
            )

    calibration = pd.DataFrame(rows)
    calibration.to_csv(RESULTS_DIR / "calibration_comparison.csv", index=False)

    month_rows = []
    for split_name, frame, y_true, raw_scores, calibrated in [
        ("validation", valid, best_scores["y_valid"], best_scores["valid_scores"], valid_calibrated),
        ("test", test, best_scores["y_test"], best_scores["test_scores"], test_calibrated),
    ]:
        for score_type, scores in [("raw", raw_scores), ("platt_calibrated_on_validation", calibrated)]:
            month_rows.append(
                {
                    "split": split_name,
                    "month": int(frame["month"].iloc[0]),
                    "score_type": score_type,
                    "n": len(frame),
                    "observed_fraud_rate": float(np.mean(y_true)),
                    "mean_score": float(np.mean(scores)),
                    "brier_score": brier_score_loss(y_true, scores),
                    "ece_10_bins": expected_calibration_error(y_true, scores),
                }
            )
    pd.DataFrame(month_rows).to_csv(RESULTS_DIR / "month_level_calibration.csv", index=False)

    plot_calibration_curves(best_scores["y_test"], best_scores["test_scores"], test_calibrated)


def plot_calibration_curves(y_test, raw_scores, calibrated_scores_):
    rows = []
    for label, scores in [("raw", raw_scores), ("platt_calibrated_on_validation", calibrated_scores_)]:
        frame = pd.DataFrame({"y": y_test, "score": scores})
        frame["bin"] = pd.qcut(frame["score"], q=10, duplicates="drop")
        grouped = frame.groupby("bin", observed=True).agg(
            observed_rate=("y", "mean"),
            mean_score=("score", "mean"),
            n=("y", "size"),
        )
        grouped["score_type"] = label
        rows.append(grouped.reset_index(drop=True))

    plot_data = pd.concat(rows, ignore_index=True)
    plt.figure(figsize=(7, 6))
    sns.lineplot(
        data=plot_data,
        x="mean_score",
        y="observed_rate",
        hue="score_type",
        marker="o",
    )
    max_value = max(plot_data["mean_score"].max(), plot_data["observed_rate"].max())
    plt.plot([0, max_value], [0, max_value], color="black", linestyle="--", linewidth=1)
    plt.title("Test Calibration Curve for Best Holistic Model")
    plt.xlabel("Mean predicted score")
    plt.ylabel("Observed fraud rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "best_model_test_calibration_curve.png", dpi=150)
    plt.close()


def save_shap_beeswarm(best_tree_model_name, fitted_info):
    print(f"Creating SHAP beeswarm for {best_tree_model_name}...")
    fitted = fitted_info["fitted"]
    if not isinstance(fitted, Pipeline):
        print("  Selected explainability model is not a sklearn pipeline; skipping beeswarm.")
        return

    rng = np.random.default_rng(RANDOM_STATE)
    X_valid = fitted_info["X_valid"]
    sample_size = min(SHAP_SAMPLE_ROWS, len(X_valid))
    sample_indices = rng.choice(len(X_valid), size=sample_size, replace=False)
    X_sample = X_valid.iloc[sample_indices].copy()

    pre_model = fitted[:-1]
    model = fitted[-1]
    X_prepared = pre_model.transform(X_sample)

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_prepared)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        explanation = shap.Explanation(
            values=shap_values,
            base_values=np.repeat(np.asarray(explainer.expected_value).reshape(-1)[-1], len(X_prepared)),
            data=X_prepared.to_numpy(),
            feature_names=X_prepared.columns.tolist(),
        )
        plt.figure(figsize=(9, 8))
        shap.plots.beeswarm(explanation, max_display=25, show=False)
        plt.title("SHAP Beeswarm - Best Holistic Tree Pipeline")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_beeswarm_best_tree_pipeline.png", dpi=150)
        plt.close()

        shap_abs = np.abs(shap_values).mean(axis=0)
        (
            pd.DataFrame({"feature": X_prepared.columns, "mean_abs_shap": shap_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .to_csv(RESULTS_DIR / "shap_beeswarm_feature_summary.csv", index=False)
        )
    except Exception as exc:
        print(f"  SHAP beeswarm failed: {exc}")


def copy_prior_artifacts():
    copies = {
        MODEL_DIR / "results" / "advanced" / "figures" / "strategy_1_ratio_fraud_rates.png":
            FIGURES_DIR / "prior_strategy_1_ratio_fraud_rates.png",
        MODEL_DIR / "results" / "advanced" / "figures" / "strategy_2_interaction_heatmaps.png":
            FIGURES_DIR / "prior_strategy_2_interaction_heatmaps.png",
        MODEL_DIR / "results" / "fairness_housing" / "figures" / "test_fpr_by_housing_status.png":
            FIGURES_DIR / "prior_fairness_test_fpr_by_housing_status.png",
        MODEL_DIR / "results" / "fairness_housing" / "figures" / "test_fnr_by_housing_status.png":
            FIGURES_DIR / "prior_fairness_test_fnr_by_housing_status.png",
        MODEL_DIR / "results" / "fairness_housing" / "housing_status_fairness_report.md":
            RESULTS_DIR / "prior_housing_status_fairness_report.md",
        MODEL_DIR / "results" / "fairness_housing" / "housing_status_group_audit.csv":
            RESULTS_DIR / "prior_housing_status_group_audit.csv",
        MODEL_DIR / "results" / "fairness_housing" / "housing_status_group_delta.csv":
            RESULTS_DIR / "prior_housing_status_group_delta.csv",
        MODEL_DIR / "results" / "fairness_housing" / "housing_status_overall_metrics.csv":
            RESULTS_DIR / "prior_housing_status_overall_metrics.csv",
    }
    for source, destination in copies.items():
        if source.exists():
            shutil.copy2(source, destination)


def aggregate_previous_results():
    rows = []
    sources = [
        ("baseline_proposal", MODEL_DIR / "results" / "validation_model_comparison.csv", "validation"),
        ("baseline_best_test", MODEL_DIR / "results" / "test_metrics_best_model.csv", "test"),
        ("randomsearch_tuning", MODEL_DIR / "results" / "tuning" / "tuned_validation_metrics.csv", "validation"),
        ("randomsearch_tuning", MODEL_DIR / "results" / "tuning" / "tuned_test_metrics.csv", "test"),
        ("balancing", MODEL_DIR / "results" / "balancing" / "balancing_validation_metrics.csv", "validation"),
        ("balancing", MODEL_DIR / "results" / "balancing" / "balancing_test_metrics.csv", "test"),
        ("advanced", MODEL_DIR / "results" / "advanced" / "advanced_validation_metrics.csv", "validation"),
        ("advanced", MODEL_DIR / "results" / "advanced" / "advanced_test_metrics.csv", "test"),
        ("fairness_housing", MODEL_DIR / "results" / "fairness_housing" / "housing_status_overall_metrics.csv", "both"),
    ]
    for source_name, path, split in sources:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "split" not in frame.columns:
            frame["split"] = split
        frame["source"] = source_name
        rows.append(frame)
    if rows:
        combined = pd.concat(rows, axis=0, ignore_index=True, sort=False)
        combined.to_csv(RESULTS_DIR / "prior_results_combined.csv", index=False)


def write_report(validation_results, test_results, score_rows, train, valid, test, unusable_columns):
    main_valid = validation_results[validation_results["threshold_policy"] == "valid_global_5pct_fpr"]
    main_test = test_results[test_results["threshold_policy"] == "valid_global_5pct_fpr"]
    best_valid = main_valid.sort_values("pr_auc", ascending=False).iloc[0]
    matching_test = main_test[main_test["model"] == best_valid["model"]].iloc[0]
    best_test = main_test.sort_values("pr_auc", ascending=False).iloc[0]

    drift_table = (
        pd.concat([train, valid, test], axis=0)
        .groupby("month")[TARGET]
        .agg(["count", "mean"])
        .rename(columns={"mean": "fraud_rate"})
        .reset_index()
    )
    drift_table.to_csv(RESULTS_DIR / "monthly_fraud_rate_drift.csv", index=False)

    anomaly_delta = (
        main_test.pivot_table(
            index=["model_family", "train_strategy"],
            columns="anomaly_policy",
            values=["pr_auc", "roc_auc", "recall_tpr", "fpr"],
            aggfunc="first",
        )
    )
    anomaly_delta.columns = ["__".join(column).strip() for column in anomaly_delta.columns]
    for metric in ["pr_auc", "roc_auc", "recall_tpr", "fpr"]:
        with_col = f"{metric}__with_anomaly_scores"
        without_col = f"{metric}__without_anomaly_scores"
        if with_col in anomaly_delta and without_col in anomaly_delta:
            anomaly_delta[f"{metric}_delta_anomaly_minus_no_anomaly"] = (
                anomaly_delta[with_col] - anomaly_delta[without_col]
            )
    anomaly_delta = anomaly_delta.reset_index()
    anomaly_delta.to_csv(RESULTS_DIR / "anomaly_feature_deltas.csv", index=False)

    recency_table = (
        main_test.groupby("train_strategy")[["pr_auc", "roc_auc", "recall_tpr", "fpr"]]
        .max()
        .reset_index()
    )
    recency_table.to_csv(RESULTS_DIR / "recency_strategy_summary.csv", index=False)

    report = f"""# Hollistic Fraud Modeling Analysis

This folder consolidates the earlier EDA/modeling/fairness work and adds two
new experiment families:

- anomaly scores as supervised model features,
- recency-aware training strategies for temporal drift.

The spelling `hollistic` follows the requested folder name.

## Data Split

- Train months: 0-5
- Validation month: 6
- Test month: 7
- Constant columns removed: {unusable_columns}

Monthly fraud-rate drift:

{markdown_table(drift_table.round(6))}

## New Anomaly Score Features

The anomaly features were fitted only on training data, using legitimate
training rows as the reference population:

- `isolation_forest_anomaly_score`
- `lof_anomaly_score`
- `autoencoder_reconstruction_error`

These scores do not replace the supervised fraud model. They are appended as
extra "rarity" signals and then evaluated inside XGBoost, LightGBM, and CatBoost.

## Recency Strategies

- `full_0_5`: train on months 0-5.
- `full_0_5_recency_weighted`: train on months 0-5, giving higher sample weight to later months.
- `recent_3_5`: train only on months 3-5.

## Main Results

Main threshold policy: choose one global threshold on validation with FPR <= 5%,
then apply that threshold to test.

Best validation model:

- `{best_valid['model']}`
- Validation PR-AUC: {best_valid['pr_auc']:.6f}
- Validation ROC-AUC: {best_valid['roc_auc']:.6f}
- Validation recall: {best_valid['recall_tpr']:.6f}
- Validation FPR: {best_valid['fpr']:.6f}

Same selected model on test:

- Test PR-AUC: {matching_test['pr_auc']:.6f}
- Test ROC-AUC: {matching_test['roc_auc']:.6f}
- Test precision: {matching_test['precision']:.6f}
- Test recall: {matching_test['recall_tpr']:.6f}
- Test FPR: {matching_test['fpr']:.6f}

Best test PR-AUC model:

- `{best_test['model']}`
- Test PR-AUC: {best_test['pr_auc']:.6f}
- Test ROC-AUC: {best_test['roc_auc']:.6f}
- Test precision: {best_test['precision']:.6f}
- Test recall: {best_test['recall_tpr']:.6f}
- Test FPR: {best_test['fpr']:.6f}

## Anomaly Feature Delta

Positive deltas mean adding anomaly scores improved that metric.

{markdown_table(anomaly_delta[['model_family', 'train_strategy', 'pr_auc_delta_anomaly_minus_no_anomaly', 'roc_auc_delta_anomaly_minus_no_anomaly', 'recall_tpr_delta_anomaly_minus_no_anomaly', 'fpr_delta_anomaly_minus_no_anomaly']].round(6))}

## Recency Summary

Best metric achieved by each train strategy across model/anomaly settings:

{markdown_table(recency_table.round(6))}

## Fairness Context

The prior housing-status audit is copied into this folder as reference plots,
CSV tables, and markdown report.
Earlier results showed that `housing_status` added predictive lift but increased
group-level FPR for `housing_status = BA`. That means a deployment recommendation
should not be made from PR-AUC alone.

## Calibration

The best validation model is also calibrated with Platt scaling on validation
month 6 and evaluated on test month 7. This is a probability calibration check,
not a new ranking model. See:

- `calibration_comparison.csv`
- `month_level_calibration.csv`
- `figures/best_model_test_calibration_curve.png`

## Key Artifacts

- `hollistic_validation_metrics.csv`
- `hollistic_test_metrics.csv`
- `hollistic_test_main_threshold.csv`
- `anomaly_feature_deltas.csv`
- `recency_strategy_summary.csv`
- `prior_results_combined.csv`
- `figures/hollistic_test_roc_top8.png`
- `figures/hollistic_test_precision_recall_top8.png`
- `figures/hollistic_confusion_matrices_test_top8.png`
- `figures/shap_beeswarm_best_tree_pipeline.png`
- `figures/anomaly_score_distributions.png`
"""
    (RESULTS_DIR / "hollistic_report.md").write_text(report, encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("Loading data...")
    data = pd.read_csv(PROJECT_DIR / "data_banca" / "Base.csv")
    train, valid, test, train_months, valid_month, test_month = split_before_preprocessing(data)

    validation_results, test_results, score_rows, fitted_models, unusable_columns = run_holistic_experiments(
        train,
        valid,
        test,
    )
    plot_metric_comparisons(validation_results, test_results, score_rows)

    main_valid = validation_results[
        validation_results["threshold_policy"] == "valid_global_5pct_fpr"
    ].copy()
    best_valid = main_valid.sort_values("pr_auc", ascending=False).iloc[0]
    run_calibration(best_valid, score_rows, valid, test)

    tree_candidates = main_valid[
        main_valid["model_family"].isin(["XGBoost", "LightGBM"])
    ].sort_values("pr_auc", ascending=False)
    if not tree_candidates.empty:
        shap_model_name = tree_candidates.iloc[0]["model"]
        save_shap_beeswarm(shap_model_name, fitted_models[shap_model_name])

    copy_prior_artifacts()
    aggregate_previous_results()
    write_report(validation_results, test_results, score_rows, train, valid, test, unusable_columns)

    metadata = {
        "train_months": train_months,
        "valid_month": valid_month,
        "test_month": test_month,
        "train_sample_max_rows": TRAIN_SAMPLE_MAX_ROWS,
        "anomaly_legit_sample_rows": ANOMALY_LEGIT_SAMPLE_ROWS,
        "lof_legit_sample_rows": LOF_LEGIT_SAMPLE_ROWS,
        "autoencoder_legit_sample_rows": AUTOENCODER_LEGIT_SAMPLE_ROWS,
        "threshold_policy": f"validation global FPR <= {PERMUTATION_THRESHOLD_FPR:.2%}",
    }
    (RESULTS_DIR / "hollistic_run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    main_test = test_results[test_results["threshold_policy"] == "valid_global_5pct_fpr"]
    print("\nTop holistic test results at validation-selected threshold:")
    print(
        main_test.sort_values("pr_auc", ascending=False)
        .head(12)
        .to_string(index=False)
    )
    print(f"\nSaved hollistic artifacts in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
