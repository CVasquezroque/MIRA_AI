"""
Compare class imbalance strategies for fraud detection.

This script intentionally keeps the feature preparation fixed so the comparison
focuses on balancing methods:
- no balancing,
- class_weight / scale_pos_weight,
- random undersampling,
- random oversampling,
- SMOTE,
- SMOTE plus weighting.

SMOTE and the other samplers are inside an imbalanced-learn pipeline. They are
applied only during model fitting, never to validation or test data.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from randomsearch_tuning import (
    RANDOM_STATE,
    TARGET,
    FraudFeatureBuilder,
    categorical_columns,
    make_raw_features,
    numeric_columns,
    split_before_preprocessing,
)


RESULTS_DIR = Path("model/results/balancing")
TRAIN_SAMPLE_MAX_ROWS = 120_000
THRESHOLD = 0.50

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


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


def build_preprocessor(scale_numeric):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", RobustScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            # Dense output keeps SMOTE behavior predictable on this moderate-width feature matrix.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=0.0,
    )


def build_pipeline(model, scale_numeric, sampler=None):
    steps = [
        (
            "features",
            FraudFeatureBuilder(
                add_missing_flags=True,
                add_outlier_flags=True,
                add_log_features=True,
            ),
        ),
        ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
    ]

    if sampler is not None:
        steps.append(("sampler", sampler))

    steps.append(("model", model))
    return ImbPipeline(steps=steps)


def model_scores(model, X):
    return model.predict_proba(X)[:, 1]


def evaluate_scores(y_true, scores, threshold=THRESHOLD):
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


def make_sampler(strategy_name):
    if strategy_name == "random_undersampling":
        return RandomUnderSampler(sampling_strategy=0.25, random_state=RANDOM_STATE)
    if strategy_name == "random_oversampling":
        return RandomOverSampler(sampling_strategy=0.10, random_state=RANDOM_STATE)
    if strategy_name in ["smote", "smote_plus_weight"]:
        return SMOTE(sampling_strategy=0.10, k_neighbors=3, random_state=RANDOM_STATE)
    return None


def make_logistic(strategy_name):
    class_weight = "balanced" if strategy_name in ["class_weight", "smote_plus_weight"] else None
    model = LogisticRegression(
        C=0.03,
        class_weight=class_weight,
        max_iter=400,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    return build_pipeline(model, scale_numeric=True, sampler=make_sampler(strategy_name))


def make_random_forest(strategy_name):
    class_weight = "balanced_subsample" if strategy_name in ["class_weight", "smote_plus_weight"] else None
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=70,
        max_features="sqrt",
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return build_pipeline(model, scale_numeric=False, sampler=make_sampler(strategy_name))


def make_extra_trees(strategy_name):
    class_weight = "balanced" if strategy_name in ["class_weight", "smote_plus_weight"] else None
    model = ExtraTreesClassifier(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=70,
        max_features="sqrt",
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return build_pipeline(model, scale_numeric=False, sampler=make_sampler(strategy_name))


def make_xgboost(strategy_name, scale_pos_weight):
    from xgboost import XGBClassifier

    if strategy_name == "class_weight":
        model_weight = scale_pos_weight
    elif strategy_name == "smote_plus_weight":
        # A mild extra weight avoids double-counting the original imbalance too aggressively.
        model_weight = 3.0
    else:
        model_weight = 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=140,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.88,
        colsample_bytree=0.76,
        min_child_weight=3,
        gamma=2.34,
        reg_lambda=0.60,
        scale_pos_weight=model_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return build_pipeline(model, scale_numeric=False, sampler=make_sampler(strategy_name))


def make_lightgbm(strategy_name, scale_pos_weight):
    from lightgbm import LGBMClassifier

    if strategy_name == "class_weight":
        model_weight = scale_pos_weight
    elif strategy_name == "smote_plus_weight":
        model_weight = 3.0
    else:
        model_weight = 1.0

    model = LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.04,
        max_depth=8,
        num_leaves=17,
        min_child_samples=269,
        subsample=0.75,
        colsample_bytree=0.81,
        reg_lambda=5.25,
        scale_pos_weight=model_weight,
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbose=-1,
        force_col_wise=True,
        device_type="cpu",
    )
    return build_pipeline(model, scale_numeric=False, sampler=make_sampler(strategy_name))


def build_experiments(scale_pos_weight):
    strategies = [
        "no_balance",
        "class_weight",
        "random_undersampling",
        "random_oversampling",
        "smote",
        "smote_plus_weight",
    ]

    makers = {
        "Logistic Regression": lambda strategy: make_logistic(strategy),
        "Random Forest": lambda strategy: make_random_forest(strategy),
        "Extra Trees": lambda strategy: make_extra_trees(strategy),
        "XGBoost": lambda strategy: make_xgboost(strategy, scale_pos_weight),
        "LightGBM": lambda strategy: make_lightgbm(strategy, scale_pos_weight),
    }

    experiments = []
    for model_name, maker in makers.items():
        for strategy in strategies:
            experiments.append(
                {
                    "model": model_name,
                    "balancing": strategy,
                    "pipeline": maker(strategy),
                }
            )

    return experiments


def plot_roc(score_rows, split_name):
    top_rows = sorted(score_rows, key=lambda row: row["pr_auc"], reverse=True)[:10]
    plt.figure(figsize=(8, 6))
    for row in top_rows:
        fpr, tpr, _ = roc_curve(row["y_true"], row["scores"])
        label = f"{row['model']} + {row['balancing']} (AUC={row['roc_auc']:.3f})"
        plt.plot(fpr, tpr, linewidth=1.4, label=label)

    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    plt.title(f"Top ROC Curves by PR-AUC - {split_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"roc_{split_name.lower()}_top10.png", dpi=150)
    plt.close()


def plot_confusion_matrices(score_rows, split_name):
    top_rows = sorted(score_rows, key=lambda row: row["pr_auc"], reverse=True)[:10]
    cols = 2
    rows = int(np.ceil(len(top_rows) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 4.2 * rows))
    axes = np.array(axes).reshape(-1)

    for axis, row in zip(axes, top_rows):
        predictions = (row["scores"] >= THRESHOLD).astype(int)
        matrix = confusion_matrix(row["y_true"], predictions, labels=[0, 1])
        display = ConfusionMatrixDisplay(matrix, display_labels=["Legit", "Fraud"])
        display.plot(ax=axis, values_format="d", colorbar=False)
        axis.set_title(f"{row['model']} + {row['balancing']}")

    for axis in axes[len(top_rows) :]:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"confusion_matrices_{split_name.lower()}_top10.png", dpi=150)
    plt.close()


def save_reports(score_rows, split_name):
    report_rows = []
    matrix_rows = []

    for row in score_rows:
        predictions = (row["scores"] >= THRESHOLD).astype(int)
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
                        "balancing": row["balancing"],
                        "class_or_average": label,
                        **metrics,
                    }
                )

        tn, fp, fn, tp = confusion_matrix(row["y_true"], predictions, labels=[0, 1]).ravel()
        matrix_rows.append(
            {
                "split": split_name,
                "model": row["model"],
                "balancing": row["balancing"],
                "threshold": THRESHOLD,
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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    data = pd.read_csv("data_banca/Base.csv")

    train, valid, test, train_months, valid_month, test_month = split_before_preprocessing(data)
    print(f"Split sizes: train={len(train):,}, valid={len(valid):,}, test={len(test):,}")

    train_sample = stratified_training_sample(train)
    print(f"Training sample for balancing comparison: {len(train_sample):,} rows")
    print(
        "SMOTE, over-sampling and under-sampling are fitted only on this training sample. "
        "Validation/test months stay untouched."
    )

    unusable_columns = [
        column
        for column in train.columns
        if column != TARGET and train[column].nunique(dropna=False) <= 1
    ]

    drop_columns = [TARGET] + unusable_columns
    if "month" in data.columns:
        drop_columns.append("month")

    X_train = make_raw_features(train_sample, drop_columns)
    y_train = train_sample[TARGET].copy()
    X_valid = make_raw_features(valid, drop_columns)
    y_valid = valid[TARGET].copy()
    X_test = make_raw_features(test, drop_columns)
    y_test = test[TARGET].copy()

    negative_count = (train_sample[TARGET] == 0).sum()
    positive_count = (train_sample[TARGET] == 1).sum()
    scale_pos_weight = negative_count / max(positive_count, 1)
    print(f"Training sample fraud rate: {positive_count / len(train_sample):.4%}")
    print(f"scale_pos_weight from training sample: {scale_pos_weight:.3f}")

    experiments = build_experiments(scale_pos_weight)

    validation_rows = []
    test_rows = []
    validation_score_rows = []
    test_score_rows = []

    for index, experiment in enumerate(experiments, start=1):
        model_name = experiment["model"]
        balancing = experiment["balancing"]
        pipeline = experiment["pipeline"]

        print(f"\n[{index}/{len(experiments)}] Training {model_name} with {balancing}...")
        fitted = clone(pipeline)
        fitted.fit(X_train, y_train)

        valid_scores = model_scores(fitted, X_valid)
        test_scores = model_scores(fitted, X_test)

        valid_metrics = evaluate_scores(y_valid, valid_scores)
        valid_metrics["model"] = model_name
        valid_metrics["balancing"] = balancing
        validation_rows.append(valid_metrics)

        test_metrics = evaluate_scores(y_test, test_scores)
        test_metrics["model"] = model_name
        test_metrics["balancing"] = balancing
        test_rows.append(test_metrics)

        validation_score_rows.append(
            {
                "model": model_name,
                "balancing": balancing,
                "y_true": y_valid,
                "scores": valid_scores,
                "pr_auc": valid_metrics["pr_auc"],
                "roc_auc": valid_metrics["roc_auc"],
            }
        )
        test_score_rows.append(
            {
                "model": model_name,
                "balancing": balancing,
                "y_true": y_test,
                "scores": test_scores,
                "pr_auc": test_metrics["pr_auc"],
                "roc_auc": test_metrics["roc_auc"],
            }
        )

    validation_results = pd.DataFrame(validation_rows).sort_values("pr_auc", ascending=False)
    test_results = pd.DataFrame(test_rows).sort_values("pr_auc", ascending=False)

    validation_results.to_csv(RESULTS_DIR / "balancing_validation_metrics.csv", index=False)
    test_results.to_csv(RESULTS_DIR / "balancing_test_metrics.csv", index=False)

    plot_roc(validation_score_rows, "Validation")
    plot_roc(test_score_rows, "Test")
    plot_confusion_matrices(validation_score_rows, "Validation")
    plot_confusion_matrices(test_score_rows, "Test")
    save_reports(validation_score_rows, "Validation")
    save_reports(test_score_rows, "Test")

    print("\nValidation metrics at threshold 0.50:")
    print(validation_results.to_string(index=False))
    print("\nTest metrics at threshold 0.50:")
    print(test_results.to_string(index=False))
    print(f"\nSaved balancing comparison artifacts in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
