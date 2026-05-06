from __future__ import annotations

import inspect
import time
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from model.full_holistic.constants import MAIN_FPR_CAP, RANDOM_STATE, TARGET, TOPK_LEVELS
from model.full_holistic.data.context import load_context, sample_frame
from model.full_holistic.features.engineering import make_raw_features
from model.full_holistic.models.common import (
    evaluate_candidate,
    fit_with_filtered_warnings,
    make_onehot_preprocessor,
    score_model,
)
from model.full_holistic.registry import load_stage_candidates, merge_candidate_registry, save_candidate_artifacts
from model.full_holistic.utils.io import DependencyError, prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger
from model.full_holistic.utils.metrics import compute_threshold_metrics, topk_rows
from model.full_holistic.utils.reporting import markdown_table
from model.full_holistic.utils.thresholds import threshold_at_fpr


CATBOOST_MISSING = "CatBoost comparison was not found. Please run --stage baseline-search first."


class ApproximateUndersampleEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Fallback only when `imbalanced-ensemble` is unavailable.

    This class is intentionally labeled as an approximation. Production-quality
    SPE, Balance Cascade, and EasyEnsemble experiments should use `imbens`.
    """

    def __init__(self, method: str, n_estimators: int = 8, random_state: int = RANDOM_STATE):
        self.method = method
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).astype(int).reset_index(drop=True)
        rng = np.random.default_rng(self.random_state)
        pos_idx = np.flatnonzero(y.to_numpy() == 1)
        neg_pool = np.flatnonzero(y.to_numpy() == 0)
        self.estimators_ = []
        hard_scores = np.zeros(len(neg_pool), dtype=float)
        for i in range(self.n_estimators):
            if len(pos_idx) == 0 or len(neg_pool) == 0:
                break
            n_neg = min(len(neg_pool), max(len(pos_idx), 1))
            if self.method == "self_paced_ensemble" and i > 0:
                order = np.argsort(hard_scores)[::-1]
                hard_take = min(len(order), n_neg // 2)
                hard_idx = neg_pool[order[:hard_take]]
                remaining = np.setdiff1d(neg_pool, hard_idx, assume_unique=False)
                random_idx = rng.choice(remaining, size=min(n_neg - hard_take, len(remaining)), replace=False) if len(remaining) else np.array([], dtype=int)
                neg_idx = np.concatenate([hard_idx, random_idx])
            else:
                neg_idx = rng.choice(neg_pool, size=n_neg, replace=len(neg_pool) < n_neg)
            idx = np.concatenate([pos_idx, neg_idx])
            rng.shuffle(idx)
            estimator = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=self.random_state + i)
            fit_with_filtered_warnings(estimator, X.iloc[idx], y.iloc[idx])
            self.estimators_.append(estimator)
            if self.method == "balance_cascade":
                neg_scores = _positive_proba(estimator, X.iloc[neg_pool])
                keep = neg_scores >= 0.5
                if keep.sum() >= max(len(pos_idx), 10):
                    neg_pool = neg_pool[keep]
            elif self.method == "self_paced_ensemble":
                hard_scores = _positive_proba(estimator, X.iloc[neg_pool])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        if not self.estimators_:
            raise RuntimeError("Fallback undersample ensemble has no fitted estimators.")
        probs = np.mean([_positive_proba(estimator, X) for estimator in self.estimators_], axis=0)
        return np.column_stack([1.0 - probs, probs])


def _positive_proba(estimator, X) -> np.ndarray:
    proba = estimator.predict_proba(X)
    classes = getattr(estimator, "classes_", None)
    if classes is None and hasattr(estimator, "named_steps"):
        classes = getattr(estimator.steps[-1][1], "classes_", None)
    if classes is not None and 1 in list(classes):
        return proba[:, list(classes).index(1)]
    return proba[:, 1]


def _candidate_class_names():
    if find_spec("imbens") is None:
        return None
    from imbens.ensemble import BalanceCascadeClassifier, EasyEnsembleClassifier, SelfPacedEnsembleClassifier

    return {
        "EasyEnsemble": EasyEnsembleClassifier,
        "Balance Cascade": BalanceCascadeClassifier,
        "Self-paced Ensemble": SelfPacedEnsembleClassifier,
    }


def _accepted_kwargs(cls, params: dict) -> dict:
    signature = inspect.signature(cls)
    accepted = set(signature.parameters)
    return {key: value for key, value in params.items() if key in accepted}


def _make_imbens_estimator(label: str, cls, params: dict, n_estimators: int):
    base_params = {
        "n_estimators": n_estimators,
        "random_state": RANDOM_STATE,
        "estimator": DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=RANDOM_STATE),
        "base_estimator": DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=RANDOM_STATE),
    }
    base_params.update(params)
    accepted = _accepted_kwargs(cls, base_params)
    ignored = sorted(set(params) - set(accepted))
    return cls(**accepted), ignored


def _make_fallback_estimator(method: str, params: dict, n_estimators: int):
    return ApproximateUndersampleEnsembleClassifier(method=method, n_estimators=n_estimators), sorted(params)


def _parameter_trials(label: str) -> list[tuple[str, dict]]:
    if label == "Self-paced Ensemble":
        return [("default", {}), ("search_k_bins_1", {"k_bins": 1}), ("search_k_bins_10", {"k_bins": 10})]
    if label == "Balance Cascade":
        return [("default", {}), ("search_replacement_true", {"replacement": True}), ("search_replacement_false", {"replacement": False})]
    if label == "EasyEnsemble":
        return [
            ("default", {}),
            ("search_max_samples_0_5_max_features_0_5", {"max_samples": 0.5, "max_features": 0.5}),
            ("search_max_samples_0_5_max_features_1_0", {"max_samples": 0.5, "max_features": 1.0}),
            ("search_max_samples_1_0_max_features_0_5", {"max_samples": 1.0, "max_features": 0.5}),
            ("search_max_samples_1_0_max_features_1_0", {"max_samples": 1.0, "max_features": 1.0}),
        ]
    raise ValueError(f"Unsupported imbalance ensemble label: {label}")


def _make_model(label: str, params: dict, X_reference: pd.DataFrame, n_estimators: int, logger: StageLogger):
    classes = _candidate_class_names()
    method_map = {
        "EasyEnsemble": "easy_ensemble",
        "Balance Cascade": "balance_cascade",
        "Self-paced Ensemble": "self_paced_ensemble",
    }
    if classes is None:
        estimator, ignored = _make_fallback_estimator(method_map[label], params, n_estimators)
        implementation = "fallback_approximation"
        logger.write(
            "Fallback Approximation Used",
            "`imbalanced-ensemble` is unavailable. Install it in `DL-env` with `pip install imbalanced-ensemble` to use official `imbens` estimators.",
        )
    else:
        estimator, ignored = _make_imbens_estimator(label, classes[label], params, n_estimators)
        implementation = "official_imbens"
        if ignored:
            logger.write("Unsupported Search Params", f"`{label}` ignored unsupported parameters: `{ignored}`.")
    pipeline = Pipeline(
        [
            ("preprocess", make_onehot_preprocessor(X_reference, scale_numeric=False)),
            ("model", estimator),
        ]
    )
    return pipeline, implementation, ignored


def _classes_report(fitted) -> str:
    classes = getattr(fitted, "classes_", None)
    if classes is None and hasattr(fitted, "named_steps"):
        classes = getattr(fitted.steps[-1][1], "classes_", None)
    if classes is None:
        return "missing"
    return ",".join(str(item) for item in list(classes))


def _score_diagnostics(y_true, scores) -> dict:
    roc_auc = float(roc_auc_score(y_true, scores))
    inverted_roc_auc = float(roc_auc_score(y_true, 1.0 - np.asarray(scores, dtype=float)))
    return {
        "roc_auc": roc_auc,
        "inverted_roc_auc": inverted_roc_auc,
        "score_inversion_suspected": bool(roc_auc < 0.5 and inverted_roc_auc > 0.5),
        "inverted_pr_auc": float(average_precision_score(y_true, 1.0 - np.asarray(scores, dtype=float))),
    }


def _operational_rows(model_name: str, y_valid, valid_scores, y_test, test_scores):
    threshold = threshold_at_fpr(y_valid, valid_scores, MAIN_FPR_CAP)
    valid_row = {
        "model": model_name,
        "split": "validation",
        "threshold_policy": "valid_global_5pct_fpr",
        **compute_threshold_metrics(y_valid, valid_scores, threshold),
    }
    test_row = {
        "model": model_name,
        "split": "test",
        "threshold_policy": "valid_global_5pct_fpr",
        **compute_threshold_metrics(y_test, test_scores, threshold),
    }
    for row, y_true, scores in [(valid_row, y_valid, valid_scores), (test_row, y_test, test_scores)]:
        for item in topk_rows(model_name, row["split"], y_true, scores, TOPK_LEVELS):
            if item["topk_pct"] in {0.005, 0.01}:
                label = "top0_5pct" if item["topk_pct"] == 0.005 else "top1pct"
                row[f"precision_{label}"] = item["precision_at_k"]
                row[f"fdr_{label}"] = item["fdr_at_k"]
                row[f"recall_{label}"] = item["recall_at_k"]
                row[f"lift_{label}"] = item["lift_at_k"]
                row[f"fp_per_tp_{label}"] = item["fp_per_tp_at_k"]
        row.update({f"score_diag_{key}": value for key, value in _score_diagnostics(y_true, scores).items()})
    return valid_row, test_row


def _catboost_comparison(results_dir: Path) -> dict:
    try:
        rows = load_stage_candidates(results_dir, "baseline-search", CATBOOST_MISSING)
    except DependencyError:
        return {}
    for row in rows:
        if row.get("model") == "baseline | CatBoost":
            return row
    return {}


def run(config, results_dir: Path, *, force: bool = False, top_n_models_to_save: int = 3, **_) -> None:
    context = load_context(results_dir)
    output_dir = prepare_stage_dir(results_dir, "imbalance-ensemble-gate", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "Imbalance Ensemble Gate Run Log")
    final_train_frame = sample_frame(context.train, config.train_rows)
    X_train = make_raw_features(final_train_frame)
    y_train = final_train_frame[TARGET]
    search_frame = sample_frame(context.train, config.tuning_train_rows)
    X_search = make_raw_features(search_frame)
    y_search = search_frame[TARGET]
    X_valid = make_raw_features(context.valid_eval)
    y_valid = context.valid_eval[TARGET]
    X_test = make_raw_features(context.test_eval)
    y_test = context.test_eval[TARGET]
    n_estimators = config.imbalance_ensemble_estimators

    all_metrics: list[dict] = []
    fitted_registry: dict[str, dict] = {}
    candidates = []
    validation_rows = []
    test_rows = []
    trial_rows = []

    for label in ["EasyEnsemble", "Balance Cascade", "Self-paced Ensemble"]:
        trial_payloads = []
        for trial_name, params in _parameter_trials(label):
            started = time.time()
            model, implementation, ignored = _make_model(label, params, X_search, n_estimators, logger)
            fitted = clone(model)
            fit_with_filtered_warnings(fitted, X_search, y_search)
            runtime = time.time() - started
            valid_scores = score_model(fitted, X_valid, "pipeline")
            validation_pr_auc = float(average_precision_score(y_valid, valid_scores))
            valid_diag = _score_diagnostics(y_valid, valid_scores)
            trial_row = {
                "method": label,
                "trial_name": trial_name,
                "implementation": implementation,
                "params": params,
                "ignored_params": ignored,
                "classes_": _classes_report(fitted),
                "validation_pr_auc": validation_pr_auc,
                "validation_roc_auc": valid_diag["roc_auc"],
                "validation_inverted_roc_auc": valid_diag["inverted_roc_auc"],
                "validation_score_inversion_suspected": valid_diag["score_inversion_suspected"],
                "runtime_seconds": runtime,
            }
            trial_rows.append(trial_row)
            trial_payloads.append((trial_row, params, implementation, ignored))

        default_payload = next(payload for payload in trial_payloads if payload[0]["trial_name"] == "default")
        tuned_payload = max(trial_payloads, key=lambda payload: payload[0]["validation_pr_auc"])
        selected_payloads = [("default", default_payload), ("validation_selected", tuned_payload)]

        for selection_label, payload in selected_payloads:
            trial_row, params, implementation, ignored = payload
            started = time.time()
            model, implementation, ignored = _make_model(label, params, X_train, n_estimators, logger)
            fitted = clone(model)
            fit_with_filtered_warnings(fitted, X_train, y_train)
            runtime = time.time() - started
            valid_scores = score_model(fitted, X_valid, "pipeline")
            test_scores = score_model(fitted, X_test, "pipeline")
            final_classes = _classes_report(fitted)
            final_valid_diag = _score_diagnostics(y_valid, valid_scores)
            final_test_diag = _score_diagnostics(y_test, test_scores)
            model_name = f"imbalance_ensemble_gate | {label} | {selection_label}"
            candidate = evaluate_candidate(
                all_metrics,
                fitted_registry,
                model_name=model_name,
                stage="imbalance_ensemble_gate",
                model_family=label,
                feature_set="baseline_eda_onehot",
                balance_policy=trial_row["trial_name"],
                train_strategy="train_sample_temporal_validation_selected" if selection_label == "validation_selected" else "train_sample_default",
                anomaly_policy="without_anomaly_scores",
                fitted=fitted,
                model_kind="pipeline",
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                y_test=y_test,
                spec={
                    "type": "imbalance_ensemble_gate",
                    "method": label,
                    "selection": selection_label,
                    "trial_name": trial_row["trial_name"],
                    "implementation": implementation,
                    "params": params,
                    "ignored_params": ignored,
                    "classes_": final_classes,
                    "search_classes_": trial_row["classes_"],
                    "search_validation_pr_auc": trial_row["validation_pr_auc"],
                    "search_validation_roc_auc": trial_row["validation_roc_auc"],
                    "final_validation_roc_auc": final_valid_diag["roc_auc"],
                    "final_test_roc_auc": final_test_diag["roc_auc"],
                    "final_test_inverted_roc_auc": final_test_diag["inverted_roc_auc"],
                    "final_test_score_inversion_suspected": final_test_diag["score_inversion_suspected"],
                },
                runtime_seconds=runtime,
            )
            candidates.append(candidate)
            valid_row, test_row = _operational_rows(model_name, y_valid, valid_scores, y_test, test_scores)
            valid_row.update({"method": label, "selection": selection_label, "implementation": implementation, "classes_": final_classes})
            test_row.update({"method": label, "selection": selection_label, "implementation": implementation, "classes_": final_classes})
            validation_rows.append(valid_row)
            test_rows.append(test_row)

    save_candidate_artifacts(
        stage_dir=output_dir,
        candidates=candidates,
        all_metrics=all_metrics,
        fitted_registry=fitted_registry,
        context=context,
        top_n_models_to_save=top_n_models_to_save,
    )
    candidate_frame = pd.DataFrame([{k: v for k, v in row.items() if k != "spec"} for row in candidates])
    candidate_frame.to_csv(output_dir / "imbalance_ensemble_candidates.csv", index=False)
    trial_frame = pd.DataFrame(trial_rows)
    trial_frame.to_csv(output_dir / "imbalance_ensemble_validation_search.csv", index=False)
    validation_frame = pd.DataFrame(validation_rows)
    test_frame = pd.DataFrame(test_rows)
    validation_frame.to_csv(output_dir / "imbalance_ensemble_validation_metrics.csv", index=False)
    test_frame.to_csv(output_dir / "imbalance_ensemble_test_metrics.csv", index=False)

    catboost = _catboost_comparison(results_dir)
    best_ensemble = max(candidates, key=lambda row: row["validation_pr_auc"]) if candidates else None
    catboost_note = "CatBoost baseline was not available for comparison."
    if catboost and best_ensemble:
        catboost_note = (
            f"Best imbalance ensemble by validation PR-AUC: `{best_ensemble['model']}` "
            f"val PR-AUC `{best_ensemble['validation_pr_auc']:.6f}`, test PR-AUC `{best_ensemble['test_pr_auc']:.6f}`. "
            f"CatBoost baseline val PR-AUC `{float(catboost['validation_pr_auc']):.6f}`, "
            f"test PR-AUC `{float(catboost['test_pr_auc']):.6f}`. "
            "Treat the imbalance ensemble as competitive only if it closes this validation gap and improves operational metrics."
        )
    implementation_note = (
        "Official `imbalanced-ensemble` (`imbens`) estimators were used."
        if any(row.get("implementation") == "official_imbens" for row in trial_rows)
        else "This run used fallback approximations because `imbalanced-ensemble` was unavailable in the active environment."
    )
    inversion_rows = [row for row in trial_rows if row["validation_score_inversion_suspected"]]
    inversion_note = (
        "No score inversion was detected."
        if not inversion_rows
        else f"Score inversion suspected for {len(inversion_rows)} trial(s); inspect `imbalance_ensemble_validation_search.csv`."
    )
    summary = f"""# Imbalance Ensemble Gate Summary

{implementation_note}

This stage audits `classes_` and always scores the probability assigned to `fraud_bool=1`.
It also reports whether `1 - score` would produce ROC-AUC > 0.5 when the original ROC-AUC is below 0.5.

## Validation Search

The search uses the temporal validation month only. Test is never used to select parameters.

- Self-paced Ensemble: `k_bins in [1, 10]`
- Balance Cascade: `replacement in [True, False]`
- EasyEnsemble: `max_samples in [0.5, 1.0]`, `max_features in [0.5, 1.0]`

{inversion_note}

## CatBoost Comparison

{catboost_note}

## Test Metrics

{markdown_table(test_frame.round(6))}
"""
    (output_dir / "imbalance_ensemble_summary.md").write_text(summary, encoding="utf-8")
    merge_candidate_registry(results_dir)
    logger.write("Imbalance Ensemble Result", f"Saved {len(candidates)} selected/default candidates and {len(trial_rows)} validation-search trials.")
    print(f"[imbalance-ensemble-gate] Saved {len(candidates)} candidates in: {output_dir}")
