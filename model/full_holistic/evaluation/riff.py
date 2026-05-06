from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, _tree

from model.full_holistic.constants import LOW_FPR_CAPS, TARGET
from model.full_holistic.data.context import load_context
from model.full_holistic.features.engineering import make_raw_features
from model.full_holistic.utils.io import prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger
from model.full_holistic.utils.metrics import compute_threshold_metrics
from model.full_holistic.utils.reporting import markdown_table


def _numeric_xy(frame: pd.DataFrame):
    X = make_raw_features(frame).select_dtypes(include=[np.number])
    y = frame[TARGET].astype(int)
    return X, y


def _extract_rules(tree: DecisionTreeClassifier, feature_names: list[str]) -> list[dict]:
    rules = []
    tree_ = tree.tree_

    def walk(node: int, conditions: list[tuple[str, str, float]]):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            counts = tree_.value[node][0]
            total = counts.sum()
            frauds = counts[1] if len(counts) > 1 else 0
            rules.append(
                {
                    "rule_id": len(rules),
                    "conditions": " AND ".join(f"{f} {op} {v:.6g}" for f, op, v in conditions) or "ALL",
                    "leaf_samples": int(total),
                    "leaf_fraud_rate": float(frauds / max(total, 1)),
                    "path": conditions,
                }
            )
            return
        feature = feature_names[tree_.feature[node]]
        threshold = float(tree_.threshold[node])
        walk(tree_.children_left[node], conditions + [(feature, "<=", threshold)])
        walk(tree_.children_right[node], conditions + [(feature, ">", threshold)])

    walk(0, [])
    return rules


def _rule_mask(X: pd.DataFrame, path: list[tuple[str, str, float]]) -> np.ndarray:
    mask = np.ones(len(X), dtype=bool)
    for feature, op, threshold in path:
        if op == "<=":
            mask &= X[feature].to_numpy() <= threshold
        else:
            mask &= X[feature].to_numpy() > threshold
    return mask


def _rule_metrics(y_true, mask):
    return compute_threshold_metrics(y_true, mask.astype(float), threshold=0.5)


def _greedy_select(candidate_rules: pd.DataFrame, masks: dict[int, np.ndarray], y_selection, fpr_cap: float) -> tuple[list[int], dict]:
    selected: list[int] = []
    active = np.zeros(len(y_selection), dtype=bool)
    ordered = candidate_rules.sort_values(["selection_precision", "selection_tp"], ascending=[False, False])
    for _, row in ordered.iterrows():
        rule_id = int(row["rule_id"])
        proposed = active | masks[rule_id]
        metrics = _rule_metrics(y_selection, proposed)
        if metrics["fpr"] <= fpr_cap and metrics["alerts"] > active.sum():
            selected.append(rule_id)
            active = proposed
    return selected, _rule_metrics(y_selection, active)


def run(config, results_dir: Path, *, force: bool = False, **_) -> None:
    context = load_context(results_dir)
    output_dir = prepare_stage_dir(results_dir, "riff-rules", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "RIFF-Style Rules Run Log")
    train_months = sorted(context.train["month"].dropna().unique())
    induction_months = train_months[:-1]
    selection_month = train_months[-1]
    induction = context.train[context.train["month"].isin(induction_months)].copy()
    selection = context.train[context.train["month"] == selection_month].copy()
    X_ind, y_ind = _numeric_xy(induction)
    X_sel, y_sel = _numeric_xy(selection)
    X_val, y_val = _numeric_xy(context.valid_eval)
    X_test, y_test = _numeric_xy(context.test_eval)
    columns = X_ind.columns.tolist()
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("tree", DecisionTreeClassifier(max_depth=config.riff_max_depth, min_samples_leaf=config.riff_min_leaf, class_weight="balanced", random_state=42)),
        ]
    )
    pipe.fit(X_ind, y_ind)
    imputer = pipe.named_steps["imputer"]
    tree = pipe.named_steps["tree"]
    X_sel_imp = pd.DataFrame(imputer.transform(X_sel), columns=columns, index=X_sel.index)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=columns, index=X_val.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=columns, index=X_test.index)
    candidates = []
    masks_selection = {}
    for rule in _extract_rules(tree, columns):
        mask = _rule_mask(X_sel_imp, rule["path"])
        metrics = _rule_metrics(y_sel, mask)
        rule.update(
            {
                "selection_support": int(mask.sum()),
                "selection_tp": metrics["tp"],
                "selection_fp": metrics["fp"],
                "selection_precision": metrics["precision"],
                "selection_fdr": metrics["fdr"],
                "selection_recall_tpr": metrics["recall_tpr"],
                "selection_fpr": metrics["fpr"],
                "selected": False,
            }
        )
        candidates.append(rule)
        masks_selection[int(rule["rule_id"])] = mask
    candidate_frame = pd.DataFrame(candidates).drop(columns=["path"])
    selected_rows = []
    validation_rows = []
    test_rows = []
    for fpr_cap in LOW_FPR_CAPS[:4]:
        selected_ids, selection_metrics = _greedy_select(candidate_frame, masks_selection, y_sel, fpr_cap)
        active_val = np.zeros(len(y_val), dtype=bool)
        active_test = np.zeros(len(y_test), dtype=bool)
        for rule in candidates:
            if int(rule["rule_id"]) in selected_ids:
                active_val |= _rule_mask(X_val_imp, rule["path"])
                active_test |= _rule_mask(X_test_imp, rule["path"])
                selected_rows.append({**{k: v for k, v in rule.items() if k != "path"}, "fpr_cap": fpr_cap, "selected": True})
        validation_rows.append({"fpr_cap": fpr_cap, **_rule_metrics(y_val, active_val)})
        test_rows.append({"fpr_cap": fpr_cap, **_rule_metrics(y_test, active_test)})
    candidate_frame.to_csv(output_dir / "riff_candidate_rules.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output_dir / "riff_selected_rules.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(output_dir / "riff_validation_metrics.csv", index=False)
    test_frame = pd.DataFrame(test_rows)
    test_frame.to_csv(output_dir / "riff_test_metrics.csv", index=False)
    summary = f"""# RIFF-Style Low-FPR Rules Summary

This is RIFF-inspired, not a full reproduction of a specific paper. It trains a class-weighted tree on induction months, extracts high-fraud leaves as readable rules, evaluates them on a separate train selection month, and greedily selects rules subject to low-FPR caps before final validation/test evaluation.

## Test Metrics

{markdown_table(test_frame.round(6))}
"""
    (output_dir / "riff_summary.md").write_text(summary, encoding="utf-8")
    logger.write("RIFF-Style Result", f"Extracted {len(candidate_frame)} candidate rules and selected rules for {len(LOW_FPR_CAPS[:4])} FPR caps.")
    print(f"[riff-rules] Saved RIFF-style artifacts in: {output_dir}")

