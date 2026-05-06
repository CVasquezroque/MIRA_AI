from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def safe_rate(numerator, denominator, default=np.nan) -> float:
    return float(numerator / denominator) if denominator else float(default)


def _safe_pr_auc(y_true, scores) -> float:
    y_arr = pd.Series(y_true).astype(int).to_numpy()
    if len(np.unique(y_arr)) < 2:
        return float("nan")
    return float(average_precision_score(y_arr, np.asarray(scores, dtype=float)))


def _safe_roc_auc(y_true, scores) -> float:
    y_arr = pd.Series(y_true).astype(int).to_numpy()
    if len(np.unique(y_arr)) < 2:
        return float("nan")
    return float(roc_auc_score(y_arr, np.asarray(scores, dtype=float)))


def compute_threshold_metrics(y_true, scores, threshold=0.50) -> dict[str, float]:
    y_arr = pd.Series(y_true).astype(int).to_numpy()
    score_arr = np.asarray(scores, dtype=float)
    predictions = (score_arr >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, predictions, labels=[0, 1]).ravel()
    prevalence = float(y_arr.mean()) if len(y_arr) else 0.0
    precision = safe_rate(tp, tp + fp, default=0.0)
    recall = safe_rate(tp, tp + fn, default=0.0)
    fpr = safe_rate(fp, fp + tn, default=0.0)
    fnr = safe_rate(fn, fn + tp, default=0.0)
    tnr = safe_rate(tn, tn + fp, default=0.0)
    alert_rate = safe_rate(tp + fp, len(y_arr), default=0.0)
    pr_auc = _safe_pr_auc(y_arr, score_arr)
    roc_auc = _safe_roc_auc(y_arr, score_arr)
    fp_per_tp = safe_rate(fp, tp, default=np.inf if fp else 0.0)
    return {
        "threshold": float(threshold),
        "alerts": int(tp + fp),
        "alert_count": int(tp + fp),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision,
        "fdr": 1.0 - precision,
        "recall_tpr": recall,
        "fpr": fpr,
        "fnr": fnr,
        "tnr": tnr,
        "specificity_tnr": tnr,
        "alert_rate": alert_rate,
        "fraud_prevalence": prevalence,
        "lift": safe_rate(precision, prevalence),
        "precision_lift": safe_rate(precision, prevalence),
        "fp_per_tp": fp_per_tp,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "pr_auc_lift": safe_rate(pr_auc, prevalence),
        "n_obs": int(len(y_arr)),
    }


def threshold_sweep_frame(y_true, scores) -> pd.DataFrame:
    y_arr = pd.Series(y_true).astype(int).to_numpy()
    score_arr = np.asarray(scores, dtype=float)
    if len(y_arr) == 0:
        return pd.DataFrame()
    order = np.argsort(score_arr)[::-1]
    sorted_scores = score_arr[order]
    sorted_y = y_arr[order]
    total_pos = int(sorted_y.sum())
    total_neg = int(len(sorted_y) - total_pos)
    tp_cum = np.cumsum(sorted_y)
    fp_cum = np.cumsum(1 - sorted_y)
    change_idx = np.where(np.diff(sorted_scores) != 0)[0]
    last_idx = np.r_[change_idx, len(sorted_scores) - 1]
    tp = tp_cum[last_idx]
    fp = fp_cum[last_idx]
    thresholds = sorted_scores[last_idx]
    fn = total_pos - tp
    tn = total_neg - fp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = tp / max(total_pos, 1)
    fpr = fp / max(total_neg, 1)
    alert_count = tp + fp
    prevalence = float(y_arr.mean())
    frame = pd.DataFrame(
        {
            "threshold": thresholds.astype(float),
            "tp": tp.astype(int),
            "fp": fp.astype(int),
            "fn": fn.astype(int),
            "tn": tn.astype(int),
            "alerts": alert_count.astype(int),
            "alert_count": alert_count.astype(int),
            "precision": precision,
            "fdr": 1.0 - precision,
            "recall_tpr": recall,
            "fpr": fpr,
            "alert_rate": alert_count / len(sorted_y),
            "fraud_prevalence": prevalence,
            "n_obs": int(len(sorted_y)),
        }
    )
    frame["lift"] = frame["precision"].map(lambda value: safe_rate(value, prevalence))
    frame["precision_lift"] = frame["lift"]
    frame["fp_per_tp"] = np.divide(frame["fp"], frame["tp"], out=np.full(len(frame), np.inf), where=frame["tp"] > 0)
    return frame


def topk_rows(model_name: str, split: str, y_true, scores, topk_levels: list[float]) -> list[dict]:
    y_arr = pd.Series(y_true).astype(int).to_numpy()
    score_arr = np.asarray(scores, dtype=float)
    order = np.argsort(score_arr)[::-1]
    y_sorted = y_arr[order]
    prevalence = float(y_arr.mean()) if len(y_arr) else 0.0
    rows = []
    for topk in topk_levels:
        k = max(1, int(math.ceil(len(y_arr) * topk)))
        positives = int(y_sorted[:k].sum())
        precision = positives / k
        recall = positives / max(int(y_arr.sum()), 1)
        rows.append(
            {
                "model": model_name,
                "split": split,
                "topk_pct": topk,
                "topk_label": f"top_{topk * 100:.1f}pct",
                "k": k,
                "precision_at_k": precision,
                "fdr_at_k": 1.0 - precision,
                "recall_at_k": recall,
                "lift_at_k": safe_rate(precision, prevalence),
                "fp_per_tp_at_k": safe_rate(k - positives, positives, default=np.inf if k - positives else 0.0),
                "fraud_prevalence": prevalence,
                "captured_frauds": positives,
            }
        )
    return rows


def make_age_group(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=[0, 20, 30, 40, 50, np.inf], labels=["10-20", "21-30", "31-40", "41-50", "51+"]).astype("string").fillna("Unknown")


def make_income_group(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=[-np.inf, 0.3, 0.6, np.inf], labels=["low_0.1_0.3", "mid_0.4_0.6", "high_0.7_0.9"]).astype("string").fillna("Unknown")
