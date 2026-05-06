from __future__ import annotations

import math
import time
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from model.full_holistic.constants import TARGET
from model.full_holistic.data.context import load_context
from model.full_holistic.features.engineering import make_raw_features
from model.full_holistic.models.common import fit_with_filtered_warnings, make_advanced_pipeline, make_baseline_pipeline, score_model
from model.full_holistic.utils.io import prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger
from model.full_holistic.utils.metrics import compute_threshold_metrics, safe_rate
from model.full_holistic.utils.reporting import markdown_table


def _top_mask(scores, pct: float) -> np.ndarray:
    k = max(1, int(math.ceil(len(scores) * pct)))
    order = np.argsort(np.asarray(scores, dtype=float))[::-1]
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:k]] = True
    return mask


def _fit_stage1(X_train, y_train, X_valid, y_valid, scale_pos_weight: float):
    try:
        from model.full_holistic.models.common import fit_catboost_native

        return fit_catboost_native(X_train, y_train, X_valid, y_valid, scale_pos_weight), "catboost_native", "CatBoost"
    except Exception:
        model = make_advanced_pipeline("Random Forest", scale_pos_weight)
        fitted = clone(model)
        fit_with_filtered_warnings(fitted, X_train, y_train)
        return fitted, "pipeline", "Random Forest"


def _make_filter(family: str, X_reference, scale_pos_weight: float):
    if family == "Logistic Regression":
        return make_baseline_pipeline("Logistic Regression", X_reference, scale_pos_weight), "pipeline"
    if family == "LightGBM":
        return make_advanced_pipeline("LightGBM", scale_pos_weight), "pipeline"
    if family == "CatBoost":
        return None, "catboost_native"
    return make_baseline_pipeline("Random Forest", X_reference, scale_pos_weight), "pipeline"


def _evaluate_cascade(model_name: str, split: str, y_true, stage1_scores, stage2_scores, top_risk_pct: float, final_top_pct: float) -> dict:
    base_mask = _top_mask(stage1_scores, top_risk_pct)
    final_count = max(1, int(math.ceil(len(stage1_scores) * final_top_pct)))
    stage2_rank = np.full(len(stage1_scores), -np.inf)
    stage2_rank[base_mask] = stage2_scores[base_mask]
    order = np.argsort(stage2_rank)[::-1]
    final_mask = np.zeros(len(stage1_scores), dtype=bool)
    final_mask[order[:final_count]] = True
    threshold = float(stage2_rank[order[final_count - 1]])
    metrics = compute_threshold_metrics(y_true, final_mask.astype(float), threshold=0.5)
    base_metrics = compute_threshold_metrics(y_true, base_mask.astype(float), threshold=0.5)
    return {
        "model": model_name,
        "split": split,
        "top_risk_pct": top_risk_pct,
        "final_top_pct": final_top_pct,
        "stage2_threshold_or_rank_cutoff": threshold,
        "base_alerts_before_filter": base_metrics["alerts"],
        "base_precision_before_filter": base_metrics["precision"],
        "base_fdr_before_filter": base_metrics["fdr"],
        "base_recall_before_filter": base_metrics["recall_tpr"],
        "base_fpr_before_filter": base_metrics["fpr"],
        **metrics,
    }


def run(config, results_dir: Path, *, force: bool = False, **_) -> None:
    context = load_context(results_dir)
    output_dir = prepare_stage_dir(results_dir, "cascade-filter", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "Cascade Filter Run Log")
    started = time.time()

    train = context.train.copy()
    months = sorted(train["month"].dropna().unique())
    oof_parts = []
    for month in months[1:]:
        history = train[train["month"] < month].copy()
        holdout = train[train["month"] == month].copy()
        if history.empty or holdout.empty or history[TARGET].nunique() < 2:
            continue
        X_hist = make_raw_features(history)
        y_hist = history[TARGET]
        X_hold = make_raw_features(holdout)
        fitted, kind, _ = _fit_stage1(X_hist, y_hist, X_hold, holdout[TARGET], context.scale_pos_weight)
        scores = score_model(fitted, X_hold, kind)
        part = holdout.copy()
        part["score_stage1_oof"] = scores
        oof_parts.append(part)
    if not oof_parts:
        raise RuntimeError("Cascade filter could not build temporal out-of-fold stage-1 scores inside train.")
    oof = pd.concat(oof_parts, axis=0)
    train_queue_mask = _top_mask(oof["score_stage1_oof"], config.cascade_top_risk_pct)
    filter_train = oof.loc[train_queue_mask].copy()
    filter_y = filter_train[TARGET]
    filter_X = make_raw_features(filter_train.drop(columns=["score_stage1_oof"], errors="ignore"))
    filter_X["score_stage1"] = filter_train["score_stage1_oof"].to_numpy()

    X_train_full = make_raw_features(context.train)
    y_train_full = context.train[TARGET]
    X_valid = make_raw_features(context.valid_eval)
    y_valid = context.valid_eval[TARGET]
    X_test = make_raw_features(context.test_eval)
    y_test = context.test_eval[TARGET]
    stage1, stage1_kind, stage1_family = _fit_stage1(X_train_full, y_train_full, X_valid, y_valid, context.scale_pos_weight)
    valid_stage1 = score_model(stage1, X_valid, stage1_kind)
    test_stage1 = score_model(stage1, X_test, stage1_kind)

    filter_families = ["Logistic Regression"]
    for family, package in [("LightGBM", "lightgbm"), ("CatBoost", "catboost")]:
        if find_spec(package) is not None:
            filter_families.append(family)
        else:
            logger.write("Filter Skipped", f"`{family}` filter skipped because `{package}` is unavailable.")
    rows_valid, rows_test = [], []
    scored_validation = pd.DataFrame({"row_id": context.valid_eval.index, "month": context.valid_eval["month"], "y_true": y_valid, "score_stage1": valid_stage1})
    scored_test = pd.DataFrame({"row_id": context.test_eval.index, "month": context.test_eval["month"], "y_true": y_test, "score_stage1": test_stage1})
    for family in filter_families:
        X_ref = filter_X.copy()
        if family == "CatBoost":
            from model.full_holistic.models.common import fit_catboost_native

            fitted2 = fit_catboost_native(X_ref, filter_y, X_ref, filter_y, context.scale_pos_weight)
            kind2 = "catboost_native"
        else:
            model2, kind2 = _make_filter(family, X_ref, context.scale_pos_weight)
            fitted2 = clone(model2)
            fit_with_filtered_warnings(fitted2, X_ref, filter_y)
        valid_filter_X = X_valid.copy()
        valid_filter_X["score_stage1"] = valid_stage1
        test_filter_X = X_test.copy()
        test_filter_X["score_stage1"] = test_stage1
        valid_stage2 = score_model(fitted2, valid_filter_X, kind2)
        test_stage2 = score_model(fitted2, test_filter_X, kind2)
        model_name = f"cascade | {stage1_family} -> {family}"
        rows_valid.append(_evaluate_cascade(model_name, "validation", y_valid, valid_stage1, valid_stage2, config.cascade_top_risk_pct, config.cascade_final_top_pct))
        rows_test.append(_evaluate_cascade(model_name, "test", y_test, test_stage1, test_stage2, config.cascade_top_risk_pct, config.cascade_final_top_pct))
        scored_validation[f"score_stage2_{family.lower().replace(' ', '_')}"] = valid_stage2
        scored_test[f"score_stage2_{family.lower().replace(' ', '_')}"] = test_stage2

    valid_frame = pd.DataFrame(rows_valid)
    test_frame = pd.DataFrame(rows_test)
    valid_frame["runtime_seconds"] = time.time() - started
    test_frame["runtime_seconds"] = time.time() - started
    valid_frame.to_csv(output_dir / "cascade_validation_metrics.csv", index=False)
    test_frame.to_csv(output_dir / "cascade_test_metrics.csv", index=False)
    scored_validation.to_csv(output_dir / "cascade_scored_validation.csv", index=False)
    scored_test.to_csv(output_dir / "cascade_scored_test.csv", index=False)
    summary = f"""# Cascade Filter Summary

This stage is a two-step human-review diagnostic, not an automatic blocker.

Stage 1 produces a broad top-risk queue. Stage 2 is trained only on temporal out-of-fold high-risk train rows, so the filter never learns from stage-1 scores generated by a model that saw the same rows.

## Test Metrics

{markdown_table(test_frame.round(6))}
"""
    (output_dir / "cascade_summary.md").write_text(summary, encoding="utf-8")
    logger.write("Cascade Result", f"Saved {len(test_frame)} cascade variants.")
    print(f"[cascade-filter] Saved cascade artifacts in: {output_dir}")
