from __future__ import annotations

from pathlib import Path

from model.full_holistic.data.context import load_context
from model.full_holistic.features.engineering import make_raw_features
from model.full_holistic.models.common import evaluate_candidate, fit_with_filtered_warnings, make_baseline_pipeline
from model.full_holistic.registry import load_stage_candidates, merge_candidate_registry, save_candidate_artifacts
from model.full_holistic.constants import TARGET
from model.full_holistic.utils.io import prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger
from sklearn.base import clone


BASELINE_MISSING = "Missing baseline candidates. Please run --stage baseline-search first, or skip stages that depend on it."


def run(config, results_dir: Path, *, force: bool = False, top_n_models_to_save: int = 3, **_) -> None:
    context = load_context(results_dir)
    baseline_candidates = load_stage_candidates(results_dir, "baseline-search", BASELINE_MISSING)
    output_dir = prepare_stage_dir(results_dir, "balance-gate", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "Balance Gate Run Log")
    all_metrics: list[dict] = []
    fitted_registry: dict[str, dict] = {}
    promoted = []
    for row in baseline_candidates:
        family = row["model_family"]
        if family not in promoted and family in {"Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM"}:
            promoted.append(family)
        if len(promoted) >= config.top_n_baseline_to_balance:
            break
    policies = ["class_weight"]
    try:
        from imblearn.over_sampling import RandomOverSampler, SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.under_sampling import RandomUnderSampler

        samplers = {
            "random_undersampling": RandomUnderSampler(sampling_strategy=0.25, random_state=42),
            "random_oversampling": RandomOverSampler(sampling_strategy=0.10, random_state=42),
            "smote": SMOTE(sampling_strategy=0.10, k_neighbors=3, random_state=42),
        }
    except Exception:
        samplers = {}
        logger.write("Resampling Skipped", "`imblearn` is unavailable; only class-weighted balance candidates were trained.")
    candidates = []
    X_train = make_raw_features(context.train_sample)
    y_train = context.train_sample[TARGET]
    X_valid = make_raw_features(context.valid_eval)
    y_valid = context.valid_eval[TARGET]
    X_test = make_raw_features(context.test_eval)
    y_test = context.test_eval[TARGET]
    for family in promoted:
        base = make_baseline_pipeline(family, X_train, context.scale_pos_weight)
        for policy in policies:
            fitted = clone(base)
            fit_with_filtered_warnings(fitted, X_train, y_train)
            candidates.append(
                evaluate_candidate(
                    all_metrics,
                    fitted_registry,
                    model_name=f"balance_gate | {family} | {policy}",
                    stage="balance_gate",
                    model_family=family,
                    feature_set="baseline_eda_onehot",
                    balance_policy=policy,
                    train_strategy="train_sample",
                    anomaly_policy="without_anomaly_scores",
                    fitted=fitted,
                    model_kind="pipeline",
                    X_valid=X_valid,
                    y_valid=y_valid,
                    X_test=X_test,
                    y_test=y_test,
                    spec={"type": "balance_gate", "model_family": family, "balance_policy": policy},
                )
            )
        for policy, sampler in samplers.items():
            fitted = ImbPipeline(base.steps[:-1] + [("sampler", sampler), base.steps[-1]])
            fit_with_filtered_warnings(fitted, X_train, y_train)
            candidates.append(
                evaluate_candidate(
                    all_metrics,
                    fitted_registry,
                    model_name=f"balance_gate | {family} | {policy}",
                    stage="balance_gate",
                    model_family=family,
                    feature_set="baseline_eda_onehot",
                    balance_policy=policy,
                    train_strategy="train_sample",
                    anomaly_policy="without_anomaly_scores",
                    fitted=fitted,
                    model_kind="pipeline",
                    X_valid=X_valid,
                    y_valid=y_valid,
                    X_test=X_test,
                    y_test=y_test,
                    spec={"type": "balance_gate", "model_family": family, "balance_policy": policy},
                )
            )
    save_candidate_artifacts(
        stage_dir=output_dir,
        candidates=candidates,
        all_metrics=all_metrics,
        fitted_registry=fitted_registry,
        context=context,
        top_n_models_to_save=top_n_models_to_save,
    )
    merge_candidate_registry(results_dir)
    logger.write("Balance Gate Result", f"Promoted families: {', '.join(promoted)}.")
    print(f"[balance-gate] Saved {len(candidates)} candidates in: {output_dir}")
