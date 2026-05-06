from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from model.full_holistic.data.context import build_context, save_context
from model.full_holistic.reporting.figures import write_data_audit_figures
from model.full_holistic.utils.io import prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger
from model.full_holistic.utils.reporting import markdown_table


def run(config, results_dir: Path, *, force: bool = False, **_) -> None:
    output_dir = prepare_stage_dir(results_dir, "data-audit", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "Data Audit Run Log")
    logger.write("Run Config", "```json\n" + json.dumps(asdict(config), indent=2) + "\n```")
    context, drift = build_context(config)
    save_context(results_dir, context, config)
    drift.to_csv(output_dir / "00_monthly_fraud_rate_drift.csv", index=False)
    split_frame = drift.copy()
    split_frame["split"] = split_frame["month"].map(
        lambda month: "train" if month in context.train_months else ("validation" if month == context.valid_month else "test")
    )
    figures = write_data_audit_figures(split_frame, output_dir, context=context)
    figure_section = "\n\n".join(f"![{figure['title']}]({figure['path']})" for figure in figures) or "_No figures generated._"
    report = f"""# Data Audit

The modular pipeline uses a chronological split and keeps test untouched for final evaluation.

- Train months: `{context.train_months}`
- Validation month: `{context.valid_month}`
- Test month: `{context.test_month}`
- Train rows: `{len(context.train):,}`
- Validation rows: `{len(context.valid):,}`
- Test rows: `{len(context.test):,}`
- Train prevalence: `{context.train_prevalence:.6f}`
- Validation prevalence: `{context.valid_prevalence:.6f}`
- Test prevalence: `{context.test_prevalence:.6f}`
- Scale positive weight from train sample: `{context.scale_pos_weight:.6f}`

## Monthly Drift

{markdown_table(split_frame.round(6))}

## Figures

{figure_section}
"""
    (output_dir / "00_data_audit_report.md").write_text(report, encoding="utf-8")
    logger.write(
        "Data Context Checkpoint",
        f"Saved reusable context checkpoint to `{output_dir / 'context.joblib'}`.",
    )
    print(f"[data-audit] Saved context and audit artifacts in: {output_dir}")
