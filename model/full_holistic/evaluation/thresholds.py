from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from model.full_holistic.registry import load_candidate_registry, load_scores
from model.full_holistic.reporting.figures import model_names_from_candidates, write_threshold_figures
from model.full_holistic.utils.metrics import compute_threshold_metrics, threshold_sweep_frame
from model.full_holistic.utils.thresholds import best_row_under_fpr, determine_threshold_policies, low_fpr_policies
from model.full_holistic.utils.io import prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger
from model.full_holistic.utils.reporting import markdown_table


def run(config, results_dir: Path, *, force: bool = False, **_) -> None:
    del config
    candidates = load_candidate_registry(results_dir, required=True)
    scores = load_scores(results_dir, required=True)
    output_dir = prepare_stage_dir(results_dir, "operational-thresholds", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "Operational Thresholds Run Log")

    validation_rows = []
    test_rows = []
    low_validation_rows = []
    low_test_rows = []
    summary_lines = [
        "# Threshold Policy Trade-off Summary",
        "",
        "Thresholds are selected on validation and applied once to test.",
        "",
    ]
    score_groups = {model: frame for model, frame in scores.groupby("model")}
    for candidate in candidates:
        model_name = candidate["model"]
        group = score_groups.get(model_name)
        if group is None:
            continue
        valid = group[group["split"] == "validation"].sort_values("row_id")
        test = group[group["split"] == "test"].sort_values("row_id")
        if valid.empty or test.empty:
            continue
        validation_sweep = threshold_sweep_frame(valid["y_true"], valid["score_raw"])
        policies = determine_threshold_policies(validation_sweep)
        validation_lookup = {}
        for policy in policies:
            validation_row = {
                **{key: candidate.get(key) for key in ["model", "stage", "model_family", "feature_set", "balance_policy", "train_strategy", "anomaly_policy"]},
                "split": "validation",
                "threshold_policy": policy["policy_name"],
                "threshold_selected_on": "validation",
                "feasible": bool(policy["feasible"]),
                "selection_notes": policy["selection_notes"],
                **(compute_threshold_metrics(valid["y_true"], valid["score_raw"], policy["threshold"]) if pd.notna(policy["threshold"]) else {}),
            }
            validation_rows.append(validation_row)
            validation_lookup[policy["policy_name"]] = validation_row
            test_rows.append(
                {
                    **{key: candidate.get(key) for key in ["model", "stage", "model_family", "feature_set", "balance_policy", "train_strategy", "anomaly_policy"]},
                    "split": "test",
                    "threshold_policy": policy["policy_name"],
                    "threshold_selected_on": "validation",
                    "feasible": bool(policy["feasible"]),
                    "selection_notes": policy["selection_notes"],
                    **(compute_threshold_metrics(test["y_true"], test["score_raw"], policy["threshold"]) if pd.notna(policy["threshold"]) else {}),
                }
            )
        for policy in low_fpr_policies():
            low_row = best_row_under_fpr(validation_sweep, policy["fpr_cap"])
            threshold = float(low_row["threshold"]) if low_row is not None else float("nan")
            common = {
                **{key: candidate.get(key) for key in ["model", "stage", "model_family", "feature_set", "balance_policy", "train_strategy", "anomaly_policy"]},
                "threshold_policy": policy["policy_name"],
                "fpr_cap": policy["fpr_cap"],
                "fpr_cap_label": policy["label"],
                "threshold_selected_on": "validation",
                "threshold": threshold,
                "feasible": low_row is not None,
            }
            if low_row is not None:
                low_validation_rows.append({**common, "split": "validation", **compute_threshold_metrics(valid["y_true"], valid["score_raw"], threshold)})
                low_test_rows.append({**common, "split": "test", **compute_threshold_metrics(test["y_true"], test["score_raw"], threshold)})
        if candidate == candidates[0] and "valid_global_5pct_fpr" in validation_lookup:
            row = validation_lookup["valid_global_5pct_fpr"]
            summary_lines.extend(
                [
                    f"## {model_name}",
                    "",
                    f"- Under `valid_global_5pct_fpr`: precision {row['precision']:.4f}, FDR {row['fdr']:.4f}, recall {row['recall_tpr']:.4f}, FPR {row['fpr']:.4f}, alerts {int(row['alert_count'])}.",
                    "",
                ]
            )

    validation_frame = pd.DataFrame(validation_rows)
    test_frame = pd.DataFrame(test_rows)
    low_validation_frame = pd.DataFrame(low_validation_rows)
    low_test_frame = pd.DataFrame(low_test_rows)
    validation_path = output_dir / "threshold_policy_validation_metrics.csv"
    test_path = output_dir / "threshold_policy_test_metrics.csv"
    low_validation_path = output_dir / "low_fpr_sweep_validation_metrics.csv"
    low_test_path = output_dir / "low_fpr_sweep_test_metrics.csv"
    validation_frame.to_csv(validation_path, index=False)
    test_frame.to_csv(test_path, index=False)
    low_validation_frame.to_csv(low_validation_path, index=False)
    low_test_frame.to_csv(low_test_path, index=False)
    low_summary = [
        "# Low-FPR Sweep Summary",
        "",
        "Thresholds are selected on validation at progressively stricter FPR caps and then applied unchanged to test.",
        "",
    ]
    if not low_test_frame.empty:
        best = low_test_frame.sort_values(["fpr_cap", "validation_pr_auc" if "validation_pr_auc" in low_test_frame.columns else "recall_tpr"], ascending=[True, False]).head(20)
        cols = ["model", "fpr_cap_label", "threshold", "alerts", "tp", "fp", "precision", "fdr", "recall_tpr", "fpr", "alert_rate", "lift", "fp_per_tp"]
        low_summary.append(markdown_table(best[[col for col in cols if col in best.columns]].round(6)))
    low_summary_path = output_dir / "low_fpr_sweep_summary.md"
    low_summary_path.write_text("\n".join(low_summary).strip() + "\n", encoding="utf-8")
    try:
        write_threshold_figures(
            low_test_frame,
            test_frame,
            output_dir,
            model_names=model_names_from_candidates(candidates, top_n=8),
        )
    except Exception as exc:
        logger.write("Operational Threshold Figures Skipped", repr(exc))
    (output_dir / "threshold_policy_tradeoff_summary.md").write_text(
        "\n".join(summary_lines).strip() + "\n",
        encoding="utf-8",
    )
    for path in [validation_path, test_path, low_validation_path, low_test_path, low_summary_path, output_dir / "threshold_policy_tradeoff_summary.md"]:
        shutil.copy2(path, results_dir / path.name)
    logger.write("Operational Thresholds", "Saved validation/test threshold policy metrics from persisted candidate scores.")
    print(f"[operational-thresholds] Saved threshold metrics for {test_frame['model'].nunique() if not test_frame.empty else 0} models.")
