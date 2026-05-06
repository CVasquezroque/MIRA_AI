from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from model.full_holistic.constants import MAIN_FPR_CAP
from model.full_holistic.registry import fitted_registry_from_scores, load_candidate_registry
from model.full_holistic.utils.metrics import compute_threshold_metrics
from model.full_holistic.utils.io import prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger
from model.full_holistic.utils.thresholds import threshold_at_fpr


def run(config, results_dir: Path, *, force: bool = False, **_) -> None:
    del config
    candidates = load_candidate_registry(results_dir, required=True)
    output_dir = prepare_stage_dir(results_dir, "calibration", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "Calibration Run Log")
    fitted_registry = fitted_registry_from_scores(results_dir, candidates)
    rows = []
    for candidate in candidates[:3]:
        info = fitted_registry[candidate["model"]]
        valid_scores = np.asarray(info["valid_scores"], dtype=float)
        test_scores = np.asarray(info["test_scores"], dtype=float)
        y_valid = pd.Series(info["y_valid"]).astype(int).to_numpy()
        y_test = pd.Series(info["y_test"]).astype(int).to_numpy()
        platt = LogisticRegression(max_iter=300)
        platt.fit(logit(np.clip(valid_scores, 1e-6, 1 - 1e-6)).reshape(-1, 1), y_valid)
        valid_platt = platt.predict_proba(logit(np.clip(valid_scores, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]
        test_platt = platt.predict_proba(logit(np.clip(test_scores, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(valid_scores, y_valid)
        valid_isotonic = isotonic.predict(valid_scores)
        test_isotonic = isotonic.predict(test_scores)
        valid_bundle = {"raw": valid_scores, "sigmoid_platt": valid_platt, "isotonic": valid_isotonic}
        test_bundle = {"raw": test_scores, "sigmoid_platt": test_platt, "isotonic": test_isotonic}
        for split, y_true, bundle in [("validation", y_valid, valid_bundle), ("test", y_test, test_bundle)]:
            for method, scores in bundle.items():
                threshold = threshold_at_fpr(y_valid, valid_bundle[method], max_fpr=MAIN_FPR_CAP)
                metrics = compute_threshold_metrics(y_true, scores, threshold)
                rows.append(
                    {
                        "model": candidate["model"],
                        "split": split,
                        "calibration_method": method,
                        "brier_score": float(brier_score_loss(y_true, scores)),
                        "mean_score": float(np.mean(scores)),
                        "observed_fraud_rate": float(np.mean(y_true)),
                        "pr_auc": float(average_precision_score(y_true, scores)),
                        "roc_auc": float(roc_auc_score(y_true, scores)),
                        "selected_threshold_fpr5": threshold,
                        "test_precision_at_fpr5_threshold": metrics["precision"],
                        "test_recall_at_fpr5_threshold": metrics["recall_tpr"],
                        "test_fpr_at_fpr5_threshold": metrics["fpr"],
                        "test_fdr_at_fpr5_threshold": metrics["fdr"],
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "07_calibration_comparison.csv", index=False)
    logger.write("Calibration Result", f"Saved calibration rows for {min(3, len(candidates))} top candidates.")
    path = output_dir / "07_calibration_comparison.csv"
    if path.exists():
        shutil.copy2(path, results_dir / path.name)
    print(f"[calibration] Saved {len(frame)} calibration rows in: {output_dir}")
