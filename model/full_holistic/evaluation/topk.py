from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from model.full_holistic.constants import TOPK_LEVELS
from model.full_holistic.registry import load_candidate_registry, load_scores
from model.full_holistic.reporting.figures import model_names_from_candidates, write_topk_figures
from model.full_holistic.utils.metrics import topk_rows
from model.full_holistic.utils.io import prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger


def run(config, results_dir: Path, *, force: bool = False, **_) -> None:
    del config
    candidates = load_candidate_registry(results_dir, required=True)
    scores = load_scores(results_dir, required=True)
    output_dir = prepare_stage_dir(results_dir, "topk", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "Top-K Run Log")

    rows = []
    score_groups = {model: frame for model, frame in scores.groupby("model")}
    for candidate in candidates:
        group = score_groups.get(candidate["model"])
        if group is None:
            continue
        for split in ["validation", "test"]:
            split_frame = group[group["split"] == split].sort_values("row_id")
            if split_frame.empty:
                continue
            rows.extend(topk_rows(candidate["model"], split, split_frame["y_true"], split_frame["score_raw"], TOPK_LEVELS))
    frame = pd.DataFrame(rows)
    metrics_path = output_dir / "topk_alert_metrics.csv"
    frame.to_csv(metrics_path, index=False)
    try:
        write_topk_figures(frame, output_dir, model_names=model_names_from_candidates(candidates, top_n=8))
    except Exception as exc:
        logger.write("Top-K Figures Skipped", repr(exc))
    test_top1 = frame[(frame["split"] == "test") & (frame["topk_pct"] == 0.01)].sort_values("recall_at_k", ascending=False) if not frame.empty else pd.DataFrame()
    lines = [
        "# Top-K Alert Summary",
        "",
        "Top-K metrics are computed from persisted ranked fraud scores.",
        "",
    ]
    if not test_top1.empty:
        best = test_top1.iloc[0]
        lines.append(
            f"- At top 1%, `{best['model']}` captures {best['recall_at_k']:.4f} recall with precision {best['precision_at_k']:.4f} and lift {best['lift_at_k']:.2f}x."
        )
    summary_path = output_dir / "topk_alert_summary.md"
    summary_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    for path in [metrics_path, summary_path]:
        shutil.copy2(path, results_dir / path.name)
    logger.write("Top-K Alerts", "Saved top-K metrics from persisted candidate scores.")
    print(f"[topk] Saved top-K metrics in: {output_dir}")
