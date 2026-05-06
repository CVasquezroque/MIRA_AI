from __future__ import annotations

import shutil
from pathlib import Path

from model.full_holistic.data.context import load_context
from model.full_holistic.features.engineering import make_raw_features
from model.full_holistic.models.common import evaluate_candidate, fit_with_filtered_warnings, make_advanced_pipeline
from model.full_holistic.reporting.figures import write_ablation_figures
from model.full_holistic.utils.io import prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger
from model.full_holistic.constants import TARGET
from sklearn.base import clone
import pandas as pd


def run(config, results_dir: Path, *, force: bool = False, **_) -> None:
    context = load_context(results_dir)
    output_dir = prepare_stage_dir(results_dir, "feature-ablation", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "Feature Ablation Run Log")
    feature_sets = {
        "original_with_sensitive": False,
        "original_without_sensitive": True,
    }
    rows = []
    for name, drop_sensitive in feature_sets.items():
        X_train = make_raw_features(context.train_sample, drop_sensitive=drop_sensitive)
        y_train = context.train_sample[TARGET]
        X_valid = make_raw_features(context.valid_eval, drop_sensitive=drop_sensitive)
        y_valid = context.valid_eval[TARGET]
        X_test = make_raw_features(context.test_eval, drop_sensitive=drop_sensitive)
        y_test = context.test_eval[TARGET]
        for family in ["Logistic Regression", "Random Forest"]:
            fitted = clone(make_advanced_pipeline(family, context.scale_pos_weight))
            fit_with_filtered_warnings(fitted, X_train, y_train)
            all_metrics, registry = [], {}
            candidate = evaluate_candidate(
                all_metrics,
                registry,
                model_name=f"feature_ablation | {name} | {family}",
                stage="feature_ablation",
                model_family=family,
                feature_set=name,
                balance_policy="model_default_weighting",
                train_strategy="train_sample",
                anomaly_policy="without_anomaly_scores",
                fitted=fitted,
                model_kind="pipeline",
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                y_test=y_test,
                spec={"type": "feature_ablation", "feature_set": name, "model_family": family},
            )
            rows.append({k: v for k, v in candidate.items() if k != "spec"})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "feature_ablation_metrics.csv", index=False)
    write_ablation_figures(frame, output_dir)
    (output_dir / "feature_ablation_summary.md").write_text("# Feature Ablation Summary\n\nAutonomous sensitivity check over protected/sensitive feature removal.\n", encoding="utf-8")
    logger.write("Feature Ablation Result", f"Saved {len(frame)} ablation rows.")
    for name in ["feature_ablation_metrics.csv", "feature_ablation_summary.md"]:
        path = output_dir / name
        if path.exists():
            shutil.copy2(path, results_dir / name)
    print(f"[feature-ablation] Saved {len(frame)} ablation rows in: {output_dir}")
