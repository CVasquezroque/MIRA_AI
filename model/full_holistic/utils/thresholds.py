from __future__ import annotations

import numpy as np
import pandas as pd

from model.full_holistic.constants import BUSINESS_MAX_FDR, COST_RATIOS, LOW_FPR_CAPS, MAIN_FPR_CAP
from model.full_holistic.utils.metrics import threshold_sweep_frame


def threshold_at_fpr(y_true, scores, max_fpr: float = 0.05) -> float:
    sweep = threshold_sweep_frame(y_true, scores)
    row = best_row_under_fpr(sweep, max_fpr)
    if row is None:
        return float(np.max(scores))
    return float(row["threshold"])


def best_row_under_fpr(sweep: pd.DataFrame, max_fpr: float):
    if sweep.empty:
        return None
    eligible = sweep[sweep["fpr"] <= max_fpr].copy()
    if eligible.empty:
        eligible = sweep.sort_values(["fpr", "recall_tpr", "precision"], ascending=[True, False, False]).head(1)
    return eligible.sort_values(["recall_tpr", "precision", "threshold"], ascending=[False, False, False]).iloc[0]


def best_row_under_fdr(sweep: pd.DataFrame, max_fdr: float):
    if sweep.empty:
        return None
    eligible = sweep[sweep["fdr"] <= max_fdr].copy()
    if eligible.empty:
        return None
    return eligible.sort_values(["recall_tpr", "precision", "threshold"], ascending=[False, False, False]).iloc[0]


def cost_sensitive_row(sweep: pd.DataFrame, cost_ratio: tuple[int, int]):
    if sweep.empty:
        return None
    c_fn, c_fp = cost_ratio
    scored = sweep.assign(total_cost=c_fn * sweep["fn"] + c_fp * sweep["fp"])
    return scored.sort_values(["total_cost", "recall_tpr", "precision"], ascending=[True, False, False]).iloc[0]


def determine_threshold_policies(validation_sweep: pd.DataFrame) -> list[dict]:
    rows = []
    row = best_row_under_fpr(validation_sweep, MAIN_FPR_CAP)
    rows.append(
        {
            "policy_name": "valid_global_5pct_fpr",
            "threshold": float(row["threshold"]) if row is not None else np.nan,
            "feasible": row is not None,
            "selection_notes": "Maximize validation recall subject to FPR <= 5%.",
            "cost_ratio": None,
        }
    )
    business = best_row_under_fdr(validation_sweep, BUSINESS_MAX_FDR)
    rows.append(
        {
            "policy_name": "valid_business_fdr30",
            "threshold": float(business["threshold"]) if business is not None else np.nan,
            "feasible": business is not None,
            "selection_notes": "Maximize validation recall subject to FDR <= 30%.",
            "cost_ratio": None,
        }
    )
    for cost_ratio in COST_RATIOS:
        cost_row = cost_sensitive_row(validation_sweep, cost_ratio)
        rows.append(
            {
                "policy_name": f"valid_cost_sensitive_{cost_ratio[0]}_to_{cost_ratio[1]}",
                "threshold": float(cost_row["threshold"]) if cost_row is not None else np.nan,
                "feasible": cost_row is not None,
                "selection_notes": f"Minimize validation total cost with C_FN:C_FP = {cost_ratio[0]}:{cost_ratio[1]}.",
                "cost_ratio": cost_ratio,
            }
        )
    return rows


def low_fpr_policies() -> list[dict]:
    return [
        {
            "policy_name": f"valid_low_fpr_{str(cap * 100).replace('.', '_')}pct",
            "fpr_cap": cap,
            "label": f"FPR<={cap:.2%}",
        }
        for cap in LOW_FPR_CAPS
    ]
