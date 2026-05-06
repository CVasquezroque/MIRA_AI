from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

from model.full_holistic.data.context import load_context
from model.full_holistic.models.common import evaluate_candidate, train_candidate_for_family
from model.full_holistic.registry import merge_candidate_registry, save_candidate_artifacts
from model.full_holistic.utils.io import prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger


def run(config, results_dir: Path, *, force: bool = False, top_n_models_to_save: int = 3, **_) -> None:
    context = load_context(results_dir)
    output_dir = prepare_stage_dir(results_dir, "baseline-search", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "Baseline Search Run Log")
    all_metrics: list[dict] = []
    fitted_registry: dict[str, dict] = {}
    candidates = []
    families = ["Logistic Regression", "Decision Tree", "Random Forest"]
    for optional_family, package in [("XGBoost", "xgboost"), ("LightGBM", "lightgbm"), ("CatBoost", "catboost")]:
        if find_spec(package) is not None:
            families.append(optional_family)
        else:
            logger.write("Optional Model Skipped", f"`{optional_family}` was skipped because its dependency is unavailable.")
    for family in families:
        payload = train_candidate_for_family(
            context,
            config,
            family,
            feature_mode="baseline",
            stage="baseline_search",
            label_prefix="baseline",
        )
        candidates.append(evaluate_candidate(all_metrics, fitted_registry, **payload))
    save_candidate_artifacts(
        stage_dir=output_dir,
        candidates=candidates,
        all_metrics=all_metrics,
        fitted_registry=fitted_registry,
        context=context,
        top_n_models_to_save=top_n_models_to_save,
    )
    merge_candidate_registry(results_dir)
    if candidates:
        logger.write("Baseline Result", f"Saved {len(candidates)} autonomous baseline candidates.")
    print(f"[baseline-search] Saved {len(candidates)} candidates in: {output_dir}")
