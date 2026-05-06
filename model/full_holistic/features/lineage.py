from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from model.full_holistic.data.context import load_context
from model.full_holistic.utils.io import prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger


def run(config, results_dir: Path, *, force: bool = False, **_) -> None:
    del config
    context = load_context(results_dir)
    output_dir = prepare_stage_dir(results_dir, "final-report", force=False)
    logger = StageLogger(output_dir / "feature_lineage_log.md", "Feature Lineage Log")
    rows = []
    protected = {"housing_status", "employment_status", "customer_age", "income"}
    for feature in context.original_feature_columns:
        rows.append(
            {
                "feature": feature,
                "original_or_generated": "original",
                "feature_group": "protected_or_sensitive_candidate" if feature in protected else "original_raw",
                "source_feature_or_features": feature,
                "notes": "Original input column.",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "feature_lineage.csv", index=False)
    logger.write("Feature Lineage", f"Saved lineage for {len(frame)} original columns.")
    path = output_dir / "feature_lineage.csv"
    if path.exists():
        shutil.copy2(path, results_dir / "feature_lineage.csv")
