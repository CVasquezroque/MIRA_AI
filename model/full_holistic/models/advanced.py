from __future__ import annotations

from pathlib import Path

from model.full_holistic.data.context import load_context
from model.full_holistic.models.balancing import BASELINE_MISSING
from model.full_holistic.models.common import evaluate_candidate, train_candidate_for_family
from model.full_holistic.registry import (
    load_candidates_for_stages,
    load_stage_candidates,
    merge_candidate_registry,
    save_candidate_artifacts,
)
from model.full_holistic.utils.io import prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger


def run(config, results_dir: Path, *, force: bool = False, top_n_models_to_save: int = 3, **_) -> None:
    context = load_context(results_dir)
    load_stage_candidates(results_dir, "baseline-search", BASELINE_MISSING)
    previous_candidates = load_candidates_for_stages(
        results_dir,
        ["baseline-search", "balance-gate"],
        required=True,
    )
    output_dir = prepare_stage_dir(results_dir, "advanced-features-gate", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "Advanced Feature Gate Run Log")
    all_metrics: list[dict] = []
    fitted_registry: dict[str, dict] = {}
    supported_families = {"Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost"}
    families = []
    for row in previous_candidates:
        family = row["model_family"]
        if family not in families and family in supported_families:
            families.append(family)
        if len(families) >= config.top_n_to_advanced:
            break
    candidates = []
    for family in families:
        payload = train_candidate_for_family(
            context,
            config,
            family,
            feature_mode="advanced",
            stage="advanced_gate",
            label_prefix="advanced_gate",
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
    logger.write("Advanced Feature Gate Result", f"Promoted families: {', '.join(families)}.")
    print(f"[advanced-features-gate] Saved {len(candidates)} candidates in: {output_dir}")
