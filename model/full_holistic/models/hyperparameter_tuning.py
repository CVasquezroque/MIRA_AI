from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from model.full_holistic.constants import RANDOM_STATE, TARGET
from model.full_holistic.data.context import load_context, sample_frame
from model.full_holistic.features.engineering import AdvancedFeatureBuilder, categorical_columns, make_raw_features
from model.full_holistic.models.common import (
    evaluate_candidate,
    fit_with_filtered_warnings,
    make_onehot_preprocessor,
    score_model,
)
from model.full_holistic.paths import STAGE_DIRS
from model.full_holistic.registry import merge_candidate_registry, save_candidate_artifacts
from model.full_holistic.reporting.figures import write_tuning_figures
from model.full_holistic.utils.io import DependencyError, prepare_stage_dir
from model.full_holistic.utils.metrics import compute_threshold_metrics, make_age_group, make_income_group, topk_rows
from model.full_holistic.utils.thresholds import threshold_at_fpr


FAMILIES = ["CatBoost", "XGBoost", "LightGBM", "Logistic Regression"]
TOPK_LEVELS = [0.005, 0.01, 0.05]
FPR_CAPS = [0.005, 0.01, 0.05]
FAIRNESS_GROUPS = ["housing_status", "customer_age_group", "income_group", "employment_status"]


def _require_optuna():
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna
    except ImportError as exc:
        raise DependencyError(
            "Missing Optuna. Install it in DL-env with: python -m pip install optuna"
        ) from exc


def _inner_temporal_split(context) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_months = list(context.train_months or sorted(context.train["month"].dropna().unique()))
    if len(train_months) < 2:
        raise DependencyError("Hyperparameter tuning requires at least two training months for inner temporal validation.")
    inner_train_months = train_months[:-1]
    inner_valid_month = train_months[-1]
    inner_train = context.train[context.train["month"].isin(inner_train_months)].copy()
    inner_valid = context.train[context.train["month"] == inner_valid_month].copy()
    if inner_train.empty or inner_valid.empty:
        raise DependencyError("Inner temporal split is empty. Check month values in the data context.")
    return inner_train, inner_valid


def _scale_pos_weight(frame: pd.DataFrame) -> float:
    positives = int(frame[TARGET].sum())
    negatives = int(len(frame) - positives)
    return negatives / max(positives, 1)


def _sample_params(trial, family: str, scale_pos_weight: float) -> dict:
    if family == "Logistic Regression":
        return {
            "C": trial.suggest_float("C", 0.005, 2.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }
    if family == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 420, step=40),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.14, log=True),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 12.0, log=True),
            "scale_pos_weight": scale_pos_weight * trial.suggest_float("scale_pos_weight_factor", 0.50, 2.0),
        }
    if family == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 380, step=40),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.14, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 60, 300, step=20),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 12.0, log=True),
            "scale_pos_weight": scale_pos_weight * trial.suggest_float("scale_pos_weight_factor", 0.50, 2.0),
        }
    if family == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 120, 400, step=40),
            "depth": trial.suggest_int("depth", 4, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.14, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 14.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.5),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.5),
            "scale_pos_weight": scale_pos_weight * trial.suggest_float("scale_pos_weight_factor", 0.60, 1.60),
        }
    raise ValueError(f"Unsupported tuning family: {family}")


def _make_pipeline(family: str, X_reference: pd.DataFrame, params: dict) -> Pipeline:
    if family == "Logistic Regression":
        model = LogisticRegression(
            C=float(params["C"]),
            class_weight=params["class_weight"],
            max_iter=3000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    elif family == "XGBoost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )
    elif family == "LightGBM":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=4,
            verbose=-1,
            force_col_wise=True,
            **params,
        )
    else:
        raise ValueError(f"Unsupported pipeline family: {family}")
    return Pipeline(
        [
            ("preprocess", make_onehot_preprocessor(X_reference, scale_numeric=family == "Logistic Regression")),
            ("model", model),
        ]
    )


def _prepare_catboost_frame(builder: AdvancedFeatureBuilder, X: pd.DataFrame, *, fit: bool, y=None) -> tuple[pd.DataFrame, list[str]]:
    transformed = builder.fit_transform(X, y) if fit else builder.transform(X)
    transformed = transformed.drop(columns=["month"], errors="ignore")
    cats = categorical_columns(transformed)
    for column in cats:
        transformed[column] = transformed[column].fillna("Unknown").astype(str)
    return transformed, cats


def _fit_catboost(
    X_train: pd.DataFrame,
    y_train,
    X_valid: pd.DataFrame | None,
    y_valid,
    params: dict,
    *,
    use_inner_early_stopping: bool,
):
    from catboost import CatBoostClassifier

    builder = AdvancedFeatureBuilder()
    X_train_cb, cat_cols = _prepare_catboost_frame(builder, X_train, fit=True, y=y_train)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        **params,
    )
    fit_kwargs = {"cat_features": cat_cols}
    if use_inner_early_stopping and X_valid is not None and y_valid is not None:
        X_valid_cb, _ = _prepare_catboost_frame(builder, X_valid, fit=False)
        fit_kwargs.update(
            {
                "eval_set": (X_valid_cb, y_valid),
                "use_best_model": True,
                "early_stopping_rounds": 40,
            }
        )
    model.fit(X_train_cb, y_train, **fit_kwargs)
    return {"model": model, "builder": builder, "cat_cols": cat_cols}


def _fit_family(
    family: str,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    params: dict,
    *,
    catboost_inner_early_stop: bool,
):
    y_train = train_frame[TARGET].astype(int)
    X_train = make_raw_features(train_frame)
    if family == "CatBoost":
        X_valid = make_raw_features(valid_frame) if valid_frame is not None else None
        y_valid = valid_frame[TARGET].astype(int) if valid_frame is not None else None
        return _fit_catboost(X_train, y_train, X_valid, y_valid, params, use_inner_early_stopping=catboost_inner_early_stop), "catboost_native"
    pipeline = clone(_make_pipeline(family, X_train, params))
    fit_with_filtered_warnings(pipeline, X_train, y_train)
    return pipeline, "pipeline"


def _topk_metric_values(model_name: str, split: str, y_true, scores) -> dict:
    values = {}
    for row in topk_rows(model_name, split, y_true, scores, TOPK_LEVELS):
        suffix = row["topk_label"].replace("top_", "top").replace(".0", "").replace(".", "_")
        values[f"precision_{suffix}"] = row["precision_at_k"]
        values[f"fdr_{suffix}"] = row["fdr_at_k"]
        values[f"recall_{suffix}"] = row["recall_at_k"]
        values[f"lift_{suffix}"] = row["lift_at_k"]
        values[f"fp_per_tp_{suffix}"] = row["fp_per_tp_at_k"]
        values[f"captured_frauds_{suffix}"] = row["captured_frauds"]
    return values


def _top1_summary(y_true, scores) -> dict:
    row = topk_rows("_", "inner_validation", y_true, scores, [0.01])[0]
    return {
        "precision_top1pct": row["precision_at_k"],
        "recall_top1pct": row["recall_at_k"],
        "fdr_top1pct": row["fdr_at_k"],
        "lift_top1pct": row["lift_at_k"],
        "fp_per_tp_top1pct": row["fp_per_tp_at_k"],
    }


def _group_frame(frame: pd.DataFrame) -> pd.DataFrame:
    groups = pd.DataFrame(index=frame.index)
    if "housing_status" in frame.columns:
        groups["housing_status"] = frame["housing_status"].astype("string").fillna("Unknown")
    if "employment_status" in frame.columns:
        groups["employment_status"] = frame["employment_status"].astype("string").fillna("Unknown")
    if "customer_age" in frame.columns:
        groups["customer_age_group"] = make_age_group(frame["customer_age"])
    if "income" in frame.columns:
        groups["income_group"] = make_income_group(frame["income"])
    return groups


def _fairness_gaps(frame: pd.DataFrame, y_true, scores, threshold: float, prefix: str = "fairness") -> dict:
    y_arr = pd.Series(y_true, index=frame.index).astype(int)
    pred = pd.Series((np.asarray(scores) >= threshold).astype(int), index=frame.index)
    groups = _group_frame(frame)
    result = {}
    all_fpr_gaps = []
    all_alert_gaps = []
    for column in FAIRNESS_GROUPS:
        if column not in groups:
            continue
        rows = []
        for _, mask in groups[column].groupby(groups[column], dropna=False).groups.items():
            idx = list(mask)
            y_g = y_arr.loc[idx].to_numpy()
            p_g = pred.loc[idx].to_numpy()
            negatives = int((y_g == 0).sum())
            fp = int(((p_g == 1) & (y_g == 0)).sum())
            alerts = int(p_g.sum())
            rows.append({"fpr": fp / max(negatives, 1), "alert_rate": alerts / max(len(y_g), 1)})
        if not rows:
            continue
        fprs = [row["fpr"] for row in rows]
        alert_rates = [row["alert_rate"] for row in rows]
        fpr_gap = float(max(fprs) - min(fprs))
        alert_gap = float(max(alert_rates) - min(alert_rates))
        result[f"{prefix}_{column}_fpr_gap"] = fpr_gap
        result[f"{prefix}_{column}_alert_rate_gap"] = alert_gap
        all_fpr_gaps.append(fpr_gap)
        all_alert_gaps.append(alert_gap)
    result[f"{prefix}_max_fpr_gap"] = float(max(all_fpr_gaps)) if all_fpr_gaps else np.nan
    result[f"{prefix}_max_alert_rate_gap"] = float(max(all_alert_gaps)) if all_alert_gaps else np.nan
    return result


def _guardrail_metrics(frame: pd.DataFrame, y_true, scores) -> dict:
    threshold = threshold_at_fpr(y_true, scores, max_fpr=0.05)
    metrics = compute_threshold_metrics(y_true, scores, threshold)
    fairness = _fairness_gaps(frame, y_true, scores, threshold, prefix="inner")
    max_fpr_gap = fairness.get("inner_max_fpr_gap", np.nan)
    max_alert_gap = fairness.get("inner_max_alert_rate_gap", np.nan)
    feasible = (
        metrics["fpr"] <= 0.05 + 1e-12
        and (math.isnan(max_fpr_gap) or max_fpr_gap <= 0.20)
        and (math.isnan(max_alert_gap) or max_alert_gap <= 0.20)
    )
    return {
        "threshold_fpr5": threshold,
        "fpr_at_fpr5": metrics["fpr"],
        "fdr_at_fpr5": metrics["fdr"],
        "recall_at_fpr5": metrics["recall_tpr"],
        "precision_at_fpr5": metrics["precision"],
        "guardrail_feasible": bool(feasible),
        **fairness,
    }


def _trial_objective(family: str, inner_train: pd.DataFrame, inner_valid: pd.DataFrame, scale_pos_weight: float):
    y_inner_valid = inner_valid[TARGET].astype(int)
    X_inner_valid = make_raw_features(inner_valid)

    def objective(trial):
        params = _sample_params(trial, family, scale_pos_weight)
        started = time.time()
        fitted, kind = _fit_family(
            family,
            inner_train,
            inner_valid if family == "CatBoost" else None,
            params,
            catboost_inner_early_stop=True,
        )
        scores = score_model(fitted, X_inner_valid, kind)
        top1 = _top1_summary(y_inner_valid, scores)
        pr_auc = float(average_precision_score(y_inner_valid, scores))
        roc_auc = float(roc_auc_score(y_inner_valid, scores))
        guardrails = _guardrail_metrics(inner_valid, y_inner_valid, scores)
        penalty = 0.0
        if guardrails["fpr_at_fpr5"] > 0.05:
            penalty += 0.05
        if guardrails.get("inner_max_fpr_gap", np.nan) > 0.20:
            penalty += 0.02
        if guardrails.get("inner_max_alert_rate_gap", np.nan) > 0.20:
            penalty += 0.02
        objective_value = (
            guardrails["recall_at_fpr5"]
            + 0.25 * top1["precision_top1pct"]
            + 0.05 * top1["recall_top1pct"]
            + 0.01 * pr_auc
            - 0.10 * guardrails["fdr_at_fpr5"]
            - penalty
        )
        trial.set_user_attr("params", params)
        trial.set_user_attr("runtime_seconds", time.time() - started)
        trial.set_user_attr("inner_pr_auc", pr_auc)
        trial.set_user_attr("inner_roc_auc", roc_auc)
        for key, value in {**top1, **guardrails}.items():
            trial.set_user_attr(key, value)
        return objective_value

    return objective


def _trial_rows(study, family: str) -> list[dict]:
    rows = []
    for trial in study.trials:
        if trial.value is None:
            continue
        row = {
            "family": family,
            "trial_number": trial.number,
            "objective_value": trial.value,
            **trial.user_attrs,
        }
        row.update({f"param_{key}": value for key, value in trial.user_attrs.get("params", {}).items()})
        rows.append(row)
    return rows


def _select_best_trial(study):
    completed = [trial for trial in study.trials if trial.value is not None]
    if not completed:
        raise DependencyError("Hyperparameter tuning did not complete any valid trials.")
    return sorted(
        completed,
        key=lambda trial: (
            bool(trial.user_attrs.get("guardrail_feasible", False)),
            float(trial.user_attrs.get("precision_top1pct", -1.0)),
            float(trial.user_attrs.get("inner_pr_auc", -1.0)),
            float(trial.user_attrs.get("recall_top1pct", -1.0)),
        ),
        reverse=True,
    )[0]


def _comparison_rows(model: str, family: str, stage: str, valid_frame: pd.DataFrame, valid_scores, test_frame: pd.DataFrame, test_scores) -> list[dict]:
    rows = []
    y_valid = valid_frame[TARGET].astype(int)
    y_test = test_frame[TARGET].astype(int)
    base = {
        "model": model,
        "stage": stage,
        "model_family": family,
        "validation_pr_auc": float(average_precision_score(y_valid, valid_scores)),
        "validation_roc_auc": float(roc_auc_score(y_valid, valid_scores)),
        "test_pr_auc": float(average_precision_score(y_test, test_scores)),
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        **{f"validation_{key}": value for key, value in _topk_metric_values(model, "validation", y_valid, valid_scores).items()},
        **{f"test_{key}": value for key, value in _topk_metric_values(model, "test", y_test, test_scores).items()},
    }
    for cap in FPR_CAPS:
        threshold = threshold_at_fpr(y_valid, valid_scores, max_fpr=cap)
        valid_metrics = compute_threshold_metrics(y_valid, valid_scores, threshold)
        test_metrics = compute_threshold_metrics(y_test, test_scores, threshold)
        fairness = _fairness_gaps(test_frame, y_test, test_scores, threshold, prefix=f"test_fprcap_{cap:g}")
        row = {
            **base,
            "threshold_selection_split": "validation",
            "fpr_cap": cap,
            "selected_threshold": threshold,
            **{f"validation_threshold_{key}": value for key, value in valid_metrics.items()},
            **{f"test_threshold_{key}": value for key, value in test_metrics.items()},
            **fairness,
        }
        rows.append(row)
    return rows


def _load_fixed_catboost_comparison(results_dir: Path, valid_frame: pd.DataFrame, test_frame: pd.DataFrame) -> list[dict]:
    baseline_dir = results_dir / STAGE_DIRS["baseline-search"]
    candidates_path = baseline_dir / "candidates.csv"
    valid_scores_path = baseline_dir / "validation_scores.csv"
    test_scores_path = baseline_dir / "test_scores.csv"
    if not (candidates_path.exists() and valid_scores_path.exists() and test_scores_path.exists()):
        return []
    candidates = pd.read_csv(candidates_path)
    catboost_rows = candidates[candidates["model_family"].eq("CatBoost")].copy()
    if catboost_rows.empty:
        return []
    catboost = catboost_rows.sort_values("validation_pr_auc", ascending=False).iloc[0]
    valid_scores_frame = pd.read_csv(valid_scores_path)
    test_scores_frame = pd.read_csv(test_scores_path)
    valid_scores = valid_scores_frame[valid_scores_frame["model"].eq(catboost["model"])]["score_raw"].to_numpy()
    test_scores = test_scores_frame[test_scores_frame["model"].eq(catboost["model"])]["score_raw"].to_numpy()
    if len(valid_scores) != len(valid_frame) or len(test_scores) != len(test_frame):
        return []
    return _comparison_rows(catboost["model"], "CatBoost", str(catboost["stage"]), valid_frame, valid_scores, test_frame, test_scores)


def run(config, results_dir: Path, *, force: bool = False, top_n_models_to_save: int = 3, **_) -> None:
    optuna = _require_optuna()
    stage = "hyperparameter-tuning-gate"
    stage_dir = prepare_stage_dir(results_dir, stage, force=force)
    context = load_context(results_dir)

    inner_train_full, inner_valid_full = _inner_temporal_split(context)
    inner_train = sample_frame(inner_train_full, config.tuning_train_rows)
    inner_valid = sample_frame(inner_valid_full, config.tuning_valid_rows)
    final_train = sample_frame(context.train, config.train_rows)
    valid_frame = context.valid_eval.copy()
    test_frame = context.test_eval.copy()
    scale_pos_weight = _scale_pos_weight(inner_train)

    trials = max(1, int(config.hyperparameter_tuning_trials))
    trial_rows: list[dict] = []
    candidates: list[dict] = []
    all_metrics: list[dict] = []
    fitted_registry: dict[str, dict] = {}
    comparison_rows: list[dict] = []

    for family in FAMILIES:
        print(f"[hyperparameter-tuning-gate] tuning {family} ({trials} trials)", flush=True)
        sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"full_holistic_{family}")
        study.optimize(_trial_objective(family, inner_train, inner_valid, scale_pos_weight), n_trials=trials, show_progress_bar=False)
        trial_rows.extend(_trial_rows(study, family))
        best_trial = _select_best_trial(study)
        best_params = best_trial.user_attrs["params"]

        started = time.time()
        fitted, kind = _fit_family(
            family,
            final_train,
            None,
            best_params,
            catboost_inner_early_stop=False,
        )
        runtime = time.time() - started
        feature_set = "catboost_native_advanced_features" if family == "CatBoost" else "baseline_eda_onehot"
        model_name = f"hyperparameter_tuned | {family}"
        payload = {
            "model_name": model_name,
            "stage": stage,
            "model_family": family,
            "feature_set": feature_set,
            "balance_policy": "validation_selected_class_weight_or_scale_pos_weight",
            "train_strategy": "full_0_5_after_inner_temporal_tuning" if config.train_rows is None else f"sample_{config.train_rows}_after_inner_temporal_tuning",
            "anomaly_policy": "without_anomaly_scores",
            "fitted": fitted,
            "model_kind": kind,
            "X_valid": make_raw_features(valid_frame),
            "y_valid": valid_frame[TARGET].astype(int),
            "X_test": make_raw_features(test_frame),
            "y_test": test_frame[TARGET].astype(int),
            "runtime_seconds": runtime,
            "spec": {
                "type": stage,
                "model_family": family,
                "feature_set": feature_set,
                "inner_tuning_train_months": list(inner_train_full["month"].dropna().sort_values().unique()),
                "inner_tuning_validation_month": int(inner_valid_full["month"].iloc[0]),
                "outer_validation_month": context.valid_month,
                "test_month": context.test_month,
                "objective": "maximize inner validation recall@FPR<=5% with precision@top1%, recall@top1%, PR-AUC, FDR, and fairness guardrails",
                "guardrails": "FPR<=5% plus max FPR and alert-rate fairness gap <=0.20 on inner validation when feasible",
                "best_inner_trial": int(best_trial.number),
                "best_inner_objective_value": float(best_trial.value),
                "best_inner_metrics": {key: value for key, value in best_trial.user_attrs.items() if key != "params"},
                "best_params": best_params,
            },
        }
        candidate = evaluate_candidate(all_metrics, fitted_registry, **payload)
        candidates.append(candidate)
        info = fitted_registry[model_name]
        comparison_rows.extend(
            _comparison_rows(
                model_name,
                family,
                stage,
                valid_frame,
                info["valid_scores"],
                test_frame,
                info["test_scores"],
            )
        )

    pd.DataFrame(trial_rows).to_csv(stage_dir / "tuning_trials.csv", index=False)
    save_candidate_artifacts(
        stage_dir=stage_dir,
        candidates=candidates,
        all_metrics=all_metrics,
        fitted_registry=fitted_registry,
        context=context,
        top_n_models_to_save=top_n_models_to_save,
    )

    comparison_rows = _load_fixed_catboost_comparison(results_dir, valid_frame, test_frame) + comparison_rows
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(stage_dir / "tuned_vs_fixed_catboost.csv", index=False)
    if not comparison.empty:
        fpr5 = comparison[comparison["fpr_cap"].eq(0.05)].copy()
        fpr5 = fpr5.dropna(axis=1, how="all")
        fpr5.to_csv(stage_dir / "tuned_vs_fixed_catboost_fpr5.csv", index=False)
        try:
            write_tuning_figures(fpr5, stage_dir)
        except Exception as exc:
            print(f"[hyperparameter-tuning-gate] tuning figures skipped: {exc!r}", flush=True)
        best_tuned = fpr5[fpr5["stage"].eq(stage)].sort_values(
            ["validation_precision_top1pct", "validation_pr_auc", "validation_recall_top1pct"],
            ascending=False,
        ).head(1)
        fixed = fpr5[fpr5["model"].str.contains("CatBoost", case=False, na=False) & ~fpr5["stage"].eq(stage)].head(1)
        if not best_tuned.empty and not fixed.empty:
            delta = float(best_tuned.iloc[0]["validation_precision_top1pct"] - fixed.iloc[0]["validation_precision_top1pct"])
            print(
                "[hyperparameter-tuning-gate] best tuned vs fixed CatBoost validation precision@top1% delta: "
                f"{delta:.6f}",
                flush=True,
            )

    merge_candidate_registry(results_dir)
    print(f"[hyperparameter-tuning-gate] Saved {len(candidates)} tuned candidates in: {stage_dir}")
