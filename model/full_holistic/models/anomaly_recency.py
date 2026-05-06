from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import IsolationForest

from model.full_holistic.constants import TARGET
from model.full_holistic.data.context import load_context
from model.full_holistic.features.engineering import make_raw_features
from model.full_holistic.models.common import evaluate_candidate, fit_with_filtered_warnings, make_advanced_pipeline
from model.full_holistic.registry import load_candidates_for_stages, merge_candidate_registry, save_candidate_artifacts
from model.full_holistic.utils.io import DependencyError, prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger


def run(config, results_dir: Path, *, force: bool = False, top_n_models_to_save: int = 3, **_) -> None:
    context = load_context(results_dir)
    previous_candidates = load_candidates_for_stages(
        results_dir,
        ["baseline-search", "balance-gate", "advanced-features-gate"],
        required=True,
    )
    if not previous_candidates:
        raise DependencyError(
            "Missing candidate registry. Please run --stage baseline-search first, or skip stages that depend on it."
        )
    output_dir = prepare_stage_dir(results_dir, "anomaly-recency-gate", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "Anomaly Recency Gate Run Log")
    all_metrics: list[dict] = []
    fitted_registry: dict[str, dict] = {}
    families = []
    supported_families = {"Logistic Regression", "Random Forest", "XGBoost", "LightGBM"}
    for row in previous_candidates:
        family = row["model_family"]
        if family not in families and family in supported_families:
            families.append(family)
        if len(families) >= config.top_n_to_anomaly:
            break
    candidates = []
    strategies = ["full_0_5", "full_0_5_recency_weighted", "recent_3_5"]
    for strategy in strategies:
        train_frame = context.train.copy()
        sample_weight = None
        if strategy == "recent_3_5" and context.train_months:
            recent_months = context.train_months[-3:]
            train_frame = train_frame[train_frame["month"].isin(recent_months)].copy()
        if config.train_rows is not None and len(train_frame) > config.train_rows:
            train_frame = train_frame.sample(n=config.train_rows, random_state=42)
        if strategy == "full_0_5_recency_weighted":
            months = train_frame["month"].astype(float)
            span = max(float(months.max() - months.min()), 1.0)
            sample_weight = (0.5 + (months - months.min()) / span).to_numpy()
            sample_weight = sample_weight / np.mean(sample_weight)
        X_train = make_raw_features(train_frame)
        y_train = train_frame[TARGET]
        X_valid = make_raw_features(context.valid_eval)
        y_valid = context.valid_eval[TARGET]
        X_test = make_raw_features(context.test_eval)
        y_test = context.test_eval[TARGET]
        for use_anomaly in [False, True]:
            X_train_use = X_train.copy()
            X_valid_use = X_valid.copy()
            X_test_use = X_test.copy()
            anomaly_policy = "without_anomaly_scores"
            if use_anomaly:
                numeric = X_train_use.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                legit = numeric.loc[y_train == 0]
                if len(legit) > config.anomaly_legit_rows:
                    legit = legit.sample(n=config.anomaly_legit_rows, random_state=42)
                iso = IsolationForest(n_estimators=80, max_samples=min(10_000, len(legit)), contamination="auto", random_state=42, n_jobs=-1)
                iso.fit(legit)
                for frame, target in [(X_train_use, numeric), (X_valid_use, X_valid_use.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)), (X_test_use, X_test_use.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0))]:
                    frame["isolation_forest_anomaly_score"] = -iso.score_samples(target.reindex(columns=numeric.columns, fill_value=0.0))
                anomaly_policy = "with_isolation_forest_anomaly_score"
            for family in families:
                pipeline = make_advanced_pipeline(family, context.scale_pos_weight)
                fitted = clone(pipeline)
                fit_kwargs = {}
                if sample_weight is not None:
                    fit_kwargs["model__sample_weight"] = sample_weight
                fit_with_filtered_warnings(fitted, X_train_use, y_train, **fit_kwargs)
                candidates.append(
                    evaluate_candidate(
                        all_metrics,
                        fitted_registry,
                        model_name=f"anomaly_recency_gate | {family} | {strategy} | {anomaly_policy}",
                        stage="anomaly_recency_gate",
                        model_family=family,
                        feature_set="advanced_plus_optional_anomaly",
                        balance_policy="model_default_weighting",
                        train_strategy=strategy,
                        anomaly_policy=anomaly_policy,
                        fitted=fitted,
                        model_kind="pipeline",
                        X_valid=X_valid_use,
                        y_valid=y_valid,
                        X_test=X_test_use,
                        y_test=y_test,
                        spec={"type": "anomaly_recency_gate", "model_family": family, "train_strategy": strategy, "use_anomaly": use_anomaly},
                    )
                )
    if candidates:
        import pandas as pd

        pd.DataFrame([{k: v for k, v in row.items() if k != "spec"} for row in candidates]).to_csv(output_dir / "recency_strategy_comparison.csv", index=False)
    save_candidate_artifacts(
        stage_dir=output_dir,
        candidates=candidates,
        all_metrics=all_metrics,
        fitted_registry=fitted_registry,
        context=context,
        top_n_models_to_save=top_n_models_to_save,
    )
    merge_candidate_registry(results_dir)
    logger.write("Anomaly Recency Result", f"Trained {len(candidates)} candidates across recency/anomaly variants.")
    print(f"[anomaly-recency-gate] Saved {len(candidates)} candidates in: {output_dir}")
