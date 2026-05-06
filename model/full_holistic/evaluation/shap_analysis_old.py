from __future__ import annotations

import shutil
from pathlib import Path
import re

import numpy as np
import pandas as pd

from model.full_holistic.data.context import load_context
from model.full_holistic.registry import load_candidate_registry, load_model_artifacts
from model.full_holistic.reporting.figures import figure_dir, write_shap_group_figures
from model.full_holistic.utils.io import DependencyError, prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).lower()).strip("_")
    return clean[:90] or "model"


def _plot_shap_outputs(shap, values: np.ndarray, prepared: pd.DataFrame, output_dir: Path, model_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slug = _slug(model_name)
    output_dir = figure_dir(output_dir)
    max_display = min(25, prepared.shape[1])
    shap.summary_plot(values, prepared, plot_type="bar", max_display=max_display, show=False)
    plt.title("Global SHAP Importance")
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_global_importance_{slug}.png", dpi=160, bbox_inches="tight")
    plt.close()

    shap.summary_plot(values, prepared, max_display=max_display, show=False)
    plt.title("SHAP Beeswarm")
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_beeswarm_{slug}.png", dpi=160, bbox_inches="tight")
    plt.close()


def _group_contribution_rows(model_name: str, raw_sample: pd.DataFrame, feature_names, values: np.ndarray) -> list[dict]:
    if "housing_status" not in raw_sample.columns:
        return []
    group_values = raw_sample["housing_status"].fillna("Unknown").astype(str).reset_index(drop=True)
    shap_frame = pd.DataFrame(np.abs(values), columns=pd.Index(feature_names).astype(str))
    shap_frame["group"] = group_values
    rows = []
    top_features = shap_frame.drop(columns=["group"]).mean().sort_values(ascending=False).head(20).index
    for group_name, group_frame in shap_frame.groupby("group", dropna=False):
        means = group_frame[top_features].mean().sort_values(ascending=False)
        for feature, value in means.items():
            rows.append(
                {
                    "model": model_name,
                    "group_attribute": "housing_status",
                    "group": group_name,
                    "feature": feature,
                    "mean_abs_shap": float(value),
                    "n_rows": int(len(group_frame)),
                }
            )
    return rows


def run(config, results_dir: Path, *, force: bool = False, **_) -> None:
    context = load_context(results_dir)
    candidates = load_candidate_registry(results_dir, required=True)
    output_dir = prepare_stage_dir(results_dir, "shap", force=force)
    logger = StageLogger(output_dir / "progressive_decision_log.md", "SHAP Run Log")
    selected_candidates = candidates[: max(config.shap_top_n, 1)]
    decision_path = results_dir / "final_candidate_decision_table.csv"
    if decision_path.exists():
        decision_table = pd.read_csv(decision_path)
        if {"model", "recommended_final_model"}.issubset(decision_table.columns):
            recommended = decision_table[decision_table["recommended_final_model"].astype(str).str.lower().eq("true")]
            if not recommended.empty:
                recommended_name = recommended.iloc[0]["model"]
                selected_candidates = [candidate for candidate in candidates if candidate["model"] == recommended_name] + [
                    candidate for candidate in selected_candidates if candidate["model"] != recommended_name
                ]
    top_names = {candidate["model"] for candidate in selected_candidates}
    artifacts = load_model_artifacts(results_dir, top_names)
    if not artifacts:
        raise DependencyError(
            "Missing saved model artifacts for SHAP. Re-run a modeling stage with --top-n-models-to-save > 0."
        )
    try:
        import shap
    except Exception as exc:
        logger.write("SHAP Unavailable", f"SHAP dependency is unavailable: {exc!r}")
        return
    rows = []
    group_rows = []
    failures = []
    for candidate in selected_candidates:
        artifact = artifacts.get(candidate["model"])
        if artifact is None:
            continue
        fitted = artifact["fitted"]
        X_valid = artifact["X_valid"]
        if len(X_valid) > config.shap_sample_rows:
            X_sample = X_valid.sample(n=config.shap_sample_rows, random_state=42)
        else:
            X_sample = X_valid.copy()
        try:
            if artifact["model_kind"] == "catboost_native":
                prepared = fitted["builder"].transform(X_sample).drop(columns=["month"], errors="ignore")
                for column in fitted["cat_cols"]:
                    if column in prepared.columns:
                        prepared[column] = prepared[column].fillna("Unknown").astype(str)
                explainer = shap.TreeExplainer(fitted["model"])
                values = explainer.shap_values(prepared)
                feature_names = prepared.columns.astype(str)
            else:
                final_model = fitted.steps[-1][1] if hasattr(fitted, "steps") else fitted
                prepared = X_sample
                if hasattr(fitted, "steps"):
                    for _, step in fitted.steps[:-1]:
                        if hasattr(step, "transform"):
                            prepared = step.transform(prepared)
                if hasattr(prepared, "toarray"):
                    prepared = prepared.toarray()
                prepared = pd.DataFrame(prepared)
                explainer = shap.Explainer(final_model, prepared)
                values = explainer(prepared).values
                feature_names = prepared.columns.astype(str)
            if isinstance(values, list):
                values = values[-1]
            values = np.asarray(values)
            prepared_frame = pd.DataFrame(prepared, columns=pd.Index(feature_names).astype(str))
            _plot_shap_outputs(shap, values, prepared_frame, output_dir, candidate["model"])
            group_rows.extend(_group_contribution_rows(candidate["model"], X_sample, feature_names, values))
            summary = pd.DataFrame({"model": candidate["model"], "feature": feature_names, "mean_abs_shap": np.abs(values).mean(axis=0)})
            summary = summary.sort_values("mean_abs_shap", ascending=False)
            summary.to_csv(output_dir / f"shap_top_features_{candidate['model'].replace(' ', '_').replace('|', '_')[:80]}.csv", index=False)
            rows.extend(summary.head(40).to_dict("records"))
        except Exception as exc:
            failures.append({"model": candidate["model"], "reason": repr(exc)})
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "top_features_by_mean_abs_shap.csv", index=False)
        shutil.copy2(output_dir / "top_features_by_mean_abs_shap.csv", results_dir / "top_features_by_mean_abs_shap.csv")
    if group_rows:
        group_path = output_dir / "shap_group_contribution_summary.csv"
        group_frame = pd.DataFrame(group_rows)
        group_frame.to_csv(group_path, index=False)
        write_shap_group_figures(group_frame, output_dir)
        shutil.copy2(group_path, results_dir / "shap_group_contribution_summary.csv")
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "shap_failures.csv", index=False)
    print(f"[shap] Saved SHAP artifacts in: {output_dir}")
