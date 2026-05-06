from __future__ import annotations

from pathlib import Path

import pandas as pd

from model.full_holistic.constants import TARGET
from model.full_holistic.data.context import load_context
from model.full_holistic.features.engineering import make_raw_features
from model.full_holistic.models.common import evaluate_candidate, fit_catboost_native
from model.full_holistic.registry import merge_candidate_registry, read_json, write_json
from model.full_holistic.utils.io import prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger
from model.full_holistic.registry import save_candidate_artifacts


def run(
    config,
    results_dir: Path,
    *,
    force: bool = False,
    include_anomaly_refit: bool = False,
    **_,
) -> None:
    context = load_context(results_dir)
    output_dir = prepare_stage_dir(results_dir, "catboost-refit", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "CatBoost Refit Run Log")
    all_metrics: list[dict] = []
    fitted_registry: dict[str, dict] = {}
    candidates = []
    X_train = make_raw_features(context.train)
    y_train = context.train[TARGET]
    X_valid = make_raw_features(context.valid_eval)
    y_valid = context.valid_eval[TARGET]
    X_test = make_raw_features(context.test_eval)
    y_test = context.test_eval[TARGET]
    fitted = fit_catboost_native(X_train, y_train, X_valid, y_valid, context.scale_pos_weight)
    candidates.append(
        evaluate_candidate(
            all_metrics,
            fitted_registry,
            model_name="catboost_refit | CatBoost native full train",
            stage="catboost_refit",
            model_family="CatBoost",
            feature_set="advanced_native_categoricals",
            balance_policy="scale_pos_weight",
            train_strategy="full_train_months_0_5",
            anomaly_policy="without_anomaly_scores",
            fitted=fitted,
            model_kind="catboost_native",
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            spec={"type": "catboost_refit", "model_family": "CatBoost", "include_anomaly_refit": include_anomaly_refit},
        )
    )
    save_candidate_artifacts(
        stage_dir=output_dir,
        candidates=candidates,
        all_metrics=all_metrics,
        fitted_registry=fitted_registry,
        context=context,
        top_n_models_to_save=1,
    )
    pd.DataFrame([{k: v for k, v in row.items() if k != "spec"} for row in candidates]).to_csv(output_dir / "catboost_refit_candidates.csv", index=False)
    merge_candidate_registry(results_dir)
    logger.write("CatBoost Refit Result", "Saved autonomous CatBoost full-train refit candidate.")
    print(f"[catboost-refit] Saved CatBoost refit artifacts in: {output_dir}")
