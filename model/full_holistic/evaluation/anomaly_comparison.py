from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
import numpy as np

from model.full_holistic.constants import TARGET
from model.full_holistic.data.context import load_context
from model.full_holistic.features.engineering import make_raw_features
from model.full_holistic.models.common import evaluate_candidate, fit_with_filtered_warnings, make_advanced_pipeline
from model.full_holistic.utils.io import prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger


def run(config, results_dir: Path, *, force: bool = False, **_) -> None:
    context = load_context(results_dir)
    output_dir = prepare_stage_dir(results_dir, "anomaly-comparison", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "Anomaly Comparison Run Log")
    X_train = make_raw_features(context.train_sample)
    y_train = context.train_sample[TARGET]
    X_valid = make_raw_features(context.valid_eval)
    y_valid = context.valid_eval[TARGET]
    X_test = make_raw_features(context.test_eval)
    y_test = context.test_eval[TARGET]
    numeric = X_train.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    legit = numeric.loc[y_train == 0]
    if len(legit) > config.anomaly_legit_rows:
        legit = legit.sample(n=config.anomaly_legit_rows, random_state=42)
    iso = IsolationForest(n_estimators=80, max_samples=min(10_000, len(legit)), contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(legit)
    rows = []
    for method in ["none", "isolation_forest"]:
        X_train_use, X_valid_use, X_test_use = X_train.copy(), X_valid.copy(), X_test.copy()
        if method == "isolation_forest":
            for frame in [X_train_use, X_valid_use, X_test_use]:
                nframe = frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                frame["isolation_forest_anomaly_score"] = -iso.score_samples(nframe.reindex(columns=numeric.columns, fill_value=0.0))
        for family in ["Logistic Regression", "Random Forest"]:
            fitted = clone(make_advanced_pipeline(family, context.scale_pos_weight))
            fit_with_filtered_warnings(fitted, X_train_use, y_train)
            all_metrics, registry = [], {}
            candidate = evaluate_candidate(
                all_metrics,
                registry,
                model_name=f"anomaly_comparison | {family} | {method}",
                stage="anomaly_comparison",
                model_family=family,
                feature_set="advanced_plus_optional_anomaly",
                balance_policy="model_default_weighting",
                train_strategy="train_sample",
                anomaly_policy=method,
                fitted=fitted,
                model_kind="pipeline",
                X_valid=X_valid_use,
                y_valid=y_valid,
                X_test=X_test_use,
                y_test=y_test,
                spec={"type": "anomaly_comparison", "model_family": family, "anomaly_method": method},
            )
            row = {k: v for k, v in candidate.items() if k != "spec"}
            row["anomaly_method"] = method
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "anomaly_score_comparison.csv", index=False)
    (output_dir / "anomaly_score_summary.md").write_text("# Anomaly Score Summary\n\nCompares supervised models with and without IsolationForest anomaly score.\n", encoding="utf-8")
    logger.write("Anomaly Comparison Result", f"Saved {len(frame)} rows.")
    for name in ["anomaly_score_comparison.csv", "anomaly_score_summary.md"]:
        path = output_dir / name
        if path.exists():
            shutil.copy2(path, results_dir / name)
    print(f"[anomaly-comparison] Saved {len(frame)} anomaly comparison rows in: {output_dir}")
