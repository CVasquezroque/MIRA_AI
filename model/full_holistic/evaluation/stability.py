from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from model.full_holistic.registry import fitted_registry_from_scores, load_candidate_registry
from model.full_holistic.utils.metrics import compute_threshold_metrics
from model.full_holistic.utils.io import prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger


def run(config, results_dir: Path, *, force: bool = False, **_) -> None:
    del config
    candidates = load_candidate_registry(results_dir, required=True)
    output_dir = prepare_stage_dir(results_dir, "stability", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "Stability Run Log")
    fitted_registry = fitted_registry_from_scores(results_dir, candidates)
    best = candidates[0]
    info = fitted_registry[best["model"]]
    rng = np.random.default_rng(42)
    rows = []
    y = pd.Series(info["y_test"]).astype(int).to_numpy()
    scores = pd.Series(info["test_scores"]).astype(float).to_numpy()
    for _ in range(200):
        idx = rng.integers(0, len(y), len(y))
        rows.append(compute_threshold_metrics(y[idx], scores[idx], best["selected_threshold"]))
    bootstrap_frame = pd.DataFrame(rows)[["pr_auc", "roc_auc", "precision", "recall_tpr", "fpr", "fdr"]].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    bootstrap_frame["n_bootstrap"] = 200
    seed_frame = pd.DataFrame()
    seed_frame.to_csv(output_dir / "stability_seed_metrics.csv", index=False)
    bootstrap_frame.to_csv(output_dir / "bootstrap_confidence_intervals.csv", index=False)
    (output_dir / "stability_summary.md").write_text("# Stability Summary\n\nBootstrap uncertainty from persisted test scores for the top validation candidate.\n", encoding="utf-8")
    logger.write("Stability Result", f"Saved bootstrap intervals for `{best['model']}`.")
    for name in ["stability_seed_metrics.csv", "bootstrap_confidence_intervals.csv", "stability_summary.md"]:
        path = output_dir / name
        if path.exists():
            shutil.copy2(path, results_dir / name)
    print(f"[stability] Saved {len(seed_frame)} seed rows and {len(bootstrap_frame)} bootstrap rows in: {output_dir}")
