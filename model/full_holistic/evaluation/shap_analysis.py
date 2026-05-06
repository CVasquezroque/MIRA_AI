from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from model.full_holistic.data.context import load_context
from model.full_holistic.registry import load_candidate_registry, load_model_artifacts
from model.full_holistic.reporting.figures import figure_dir, write_shap_group_figures
from model.full_holistic.utils.io import DependencyError, prepare_stage_dir
from model.full_holistic.utils.logging import StageLogger


CATEGORICAL_GROUP_COLUMNS = [
    "housing_status",
    "device_os",
    "employment_status",
    "payment_type",
    "source",
]
REPORTABLE_ANONYMOUS_RATIO_MAX = 0.30
REPORTABLE_MIN_NAMED_FEATURES = 3


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).lower()).strip("_")
    return clean[:90] or "model"


def _clean_feature_name(name: object) -> str:
    """Make transformed sklearn feature names readable without losing parent information."""
    text = str(name)
    text = text.replace("preprocess__", "")
    text = re.sub(r"^(numeric|categorical|remainder)__", "", text)
    text = text.replace("onehot__", "")
    text = text.replace("imputer__", "")
    text = text.replace("scaler__", "")
    return text


def _is_anonymous_feature_name(name: object) -> bool:
    text = str(name).strip()
    if not text:
        return True
    if text.isdigit():
        return True
    if re.fullmatch(r"x\d+", text.lower()):
        return True
    if re.fullmatch(r"feature_?\d+", text.lower()):
        return True
    return False


def _feature_names_are_reportable(feature_names: Iterable[object]) -> tuple[bool, str]:
    names = [str(name) for name in feature_names]
    if len(names) < REPORTABLE_MIN_NAMED_FEATURES:
        return False, "too_few_features"
    anonymous = sum(_is_anonymous_feature_name(name) for name in names)
    ratio = anonymous / max(len(names), 1)
    if ratio > REPORTABLE_ANONYMOUS_RATIO_MAX:
        return False, f"too_many_anonymous_feature_names:{ratio:.2f}"
    return True, "feature_names_interpretable"


def _get_step_feature_names(step, input_names: list[str], n_output: int) -> list[str]:
    names = None
    if hasattr(step, "get_feature_names_out"):
        try:
            names = step.get_feature_names_out(input_names)
        except TypeError:
            try:
                names = step.get_feature_names_out()
            except Exception:
                names = None
        except Exception:
            names = None
    if names is None and hasattr(step, "columns_"):
        try:
            names = list(step.columns_)
        except Exception:
            names = None
    if names is None and len(input_names) == n_output:
        names = input_names
    if names is None:
        names = [f"feature_{idx}" for idx in range(n_output)]
    return [_clean_feature_name(name) for name in list(names)]


def _as_dataframe(data, columns: list[str], index=None) -> pd.DataFrame:
    if hasattr(data, "toarray"):
        data = data.toarray()
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        frame.columns = [_clean_feature_name(col) for col in frame.columns]
        return frame
    array = np.asarray(data)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if len(columns) != array.shape[1]:
        columns = [f"feature_{idx}" for idx in range(array.shape[1])]
    return pd.DataFrame(array, columns=pd.Index(columns).astype(str), index=index)


def _prepare_pipeline_input(fitted, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Run all preprocessing steps before the final estimator while preserving names."""
    prepared = X_sample.copy()
    feature_names = list(prepared.columns.astype(str))
    index = prepared.index
    steps = fitted.steps[:-1] if hasattr(fitted, "steps") else []
    for _, step in steps:
        if not hasattr(step, "transform"):
            continue
        transformed = step.transform(prepared)
        if isinstance(transformed, pd.DataFrame):
            prepared = transformed.copy()
            prepared.columns = [_clean_feature_name(col) for col in prepared.columns]
            feature_names = list(prepared.columns.astype(str))
            index = prepared.index
            continue
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed)
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)
        feature_names = _get_step_feature_names(step, feature_names, transformed.shape[1])
        prepared = pd.DataFrame(transformed, columns=feature_names, index=index)
    return prepared


def _final_estimator(fitted):
    return fitted.steps[-1][1] if hasattr(fitted, "steps") else fitted


def _normalize_shap_values(values) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        # Binary/multiclass explainers may return (n, features, classes).
        values = values[:, :, -1]
    return values


def _plot_shap_outputs(
    shap,
    values: np.ndarray,
    prepared: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    *,
    reportable_beeswarm: bool,
    beeswarm_reason: str,
) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slug = _slug(model_name)
    output_dir = figure_dir(output_dir)
    max_display = min(25, prepared.shape[1])
    manifest_rows: list[dict] = []

    shap.summary_plot(values, prepared, plot_type="bar", max_display=max_display, show=False)
    plt.title("Global SHAP Importance")
    plt.tight_layout()
    global_path = output_dir / f"shap_global_importance_{slug}.png"
    plt.savefig(global_path, dpi=170, bbox_inches="tight")
    plt.close()
    manifest_rows.append(
        {
            "model": model_name,
            "figure_type": "global_importance",
            "path": global_path.as_posix(),
            "filename": global_path.name,
            "reportable": True,
            "reason": "global_importance_always_reportable",
        }
    )

    # Save beeswarm only as a final-report figure when feature names are interpretable.
    # Otherwise, save a diagnostic copy with a different filename so the final report does not pick it by accident.
    beeswarm_name = f"shap_beeswarm_{slug}.png" if reportable_beeswarm else f"shap_beeswarm_diagnostic_{slug}.png"
    beeswarm_path = output_dir / beeswarm_name
    shap.summary_plot(values, prepared, max_display=max_display, show=False)
    plt.title("SHAP Beeswarm")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=170, bbox_inches="tight")
    plt.close()
    manifest_rows.append(
        {
            "model": model_name,
            "figure_type": "beeswarm",
            "path": beeswarm_path.as_posix(),
            "filename": beeswarm_path.name,
            "reportable": bool(reportable_beeswarm),
            "reason": beeswarm_reason,
        }
    )
    return manifest_rows


def _infer_parent_feature(feature: str, raw_columns: Iterable[str]) -> str:
    clean = _clean_feature_name(feature)
    raw = sorted([str(col) for col in raw_columns], key=len, reverse=True)
    for parent in raw:
        if clean == parent:
            return parent
        if clean.startswith(parent + "_") or clean.startswith(parent + "__"):
            return parent
        if clean.startswith(parent + "="):
            return parent
    return clean


def _grouped_feature_importance_rows(model_name: str, raw_sample: pd.DataFrame, feature_names: list[str], values: np.ndarray) -> list[dict]:
    parents = [_infer_parent_feature(feature, raw_sample.columns) for feature in feature_names]
    shap_frame = pd.DataFrame(values, columns=pd.Index(feature_names).astype(str))
    parent_frame = pd.DataFrame(index=shap_frame.index)
    for parent in dict.fromkeys(parents):
        cols = [feature for feature, candidate_parent in zip(feature_names, parents) if candidate_parent == parent]
        parent_frame[parent] = shap_frame[cols].sum(axis=1)
    mean_abs = parent_frame.abs().mean().sort_values(ascending=False)
    mean_signed = parent_frame.mean().reindex(mean_abs.index)
    return [
        {
            "model": model_name,
            "parent_feature": feature,
            "mean_abs_shap": float(mean_abs.loc[feature]),
            "mean_shap": float(mean_signed.loc[feature]),
            "n_rows": int(len(parent_frame)),
        }
        for feature in mean_abs.index
    ]


def _categorical_group_contribution_rows(model_name: str, raw_sample: pd.DataFrame, feature_names: list[str], values: np.ndarray) -> list[dict]:
    available_group_columns = [column for column in CATEGORICAL_GROUP_COLUMNS if column in raw_sample.columns]
    if not available_group_columns:
        return []
    shap_frame = pd.DataFrame(values, columns=pd.Index(feature_names).astype(str)).reset_index(drop=True)
    abs_frame = shap_frame.abs()
    top_features = abs_frame.mean().sort_values(ascending=False).head(20).index
    rows: list[dict] = []
    for group_attribute in available_group_columns:
        group_values = raw_sample[group_attribute].fillna("Unknown").astype(str).reset_index(drop=True)
        temp = shap_frame[top_features].copy()
        temp_abs = abs_frame[top_features].copy()
        temp["group"] = group_values
        temp_abs["group"] = group_values
        for group_name, group_abs_frame in temp_abs.groupby("group", dropna=False):
            group_signed_frame = temp[temp["group"].eq(group_name)]
            means_abs = group_abs_frame.drop(columns=["group"]).mean().sort_values(ascending=False)
            means_signed = group_signed_frame.drop(columns=["group"]).mean().reindex(means_abs.index)
            for feature, value in means_abs.items():
                rows.append(
                    {
                        "model": model_name,
                        "group_attribute": group_attribute,
                        "group": group_name,
                        "feature": feature,
                        "mean_abs_shap": float(value),
                        "mean_shap": float(means_signed.loc[feature]),
                        "n_rows": int(len(group_abs_frame)),
                    }
                )
    return rows


def _save_grouped_feature_plot(grouped_rows: list[dict], output_dir: Path, model_name: str) -> dict | None:
    if not grouped_rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(grouped_rows)
    frame = frame[frame["model"].eq(model_name)].sort_values("mean_abs_shap", ascending=False).head(25)
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    ax.barh(frame["parent_feature"].astype(str)[::-1], frame["mean_abs_shap"].astype(float)[::-1])
    ax.set_title("Grouped SHAP Importance")
    ax.set_xlabel("Mean absolute SHAP after grouping transformed features")
    path = figure_dir(output_dir) / f"shap_grouped_feature_importance_{_slug(model_name)}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {
        "model": model_name,
        "figure_type": "grouped_feature_importance",
        "path": path.as_posix(),
        "filename": path.name,
        "reportable": True,
        "reason": "grouped_transformed_features_to_parent_features",
    }


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

    rows: list[dict] = []
    group_rows: list[dict] = []
    grouped_feature_rows: list[dict] = []
    figure_manifest: list[dict] = []
    failures: list[dict] = []

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
                feature_names = list(prepared.columns.astype(str))
                prepared_frame = prepared.copy()
            else:
                final_model = _final_estimator(fitted)
                prepared_frame = _prepare_pipeline_input(fitted, X_sample)
                feature_names = list(prepared_frame.columns.astype(str))
                explainer = shap.Explainer(final_model, prepared_frame)
                values = explainer(prepared_frame).values

            values = _normalize_shap_values(values)
            if values.shape[1] != len(feature_names):
                raise ValueError(
                    f"SHAP value width ({values.shape[1]}) does not match feature_names ({len(feature_names)})."
                )
            prepared_frame = _as_dataframe(prepared_frame, feature_names, index=X_sample.index)
            reportable_beeswarm, beeswarm_reason = _feature_names_are_reportable(feature_names)
            figure_manifest.extend(
                _plot_shap_outputs(
                    shap,
                    values,
                    prepared_frame,
                    output_dir,
                    candidate["model"],
                    reportable_beeswarm=reportable_beeswarm,
                    beeswarm_reason=beeswarm_reason,
                )
            )

            group_rows.extend(_categorical_group_contribution_rows(candidate["model"], X_sample, feature_names, values))
            model_grouped_rows = _grouped_feature_importance_rows(candidate["model"], X_sample, feature_names, values)
            grouped_feature_rows.extend(model_grouped_rows)
            grouped_plot = _save_grouped_feature_plot(model_grouped_rows, output_dir, candidate["model"])
            if grouped_plot:
                figure_manifest.append(grouped_plot)

            summary = pd.DataFrame(
                {
                    "model": candidate["model"],
                    "feature": feature_names,
                    "mean_abs_shap": np.abs(values).mean(axis=0),
                    "mean_shap": values.mean(axis=0),
                    "reportable_feature_name": [not _is_anonymous_feature_name(name) for name in feature_names],
                }
            ).sort_values("mean_abs_shap", ascending=False)
            summary.to_csv(output_dir / f"shap_top_features_{_slug(candidate['model'])}.csv", index=False)
            rows.extend(summary.head(40).to_dict("records"))
        except Exception as exc:
            failures.append({"model": candidate["model"], "reason": repr(exc)})

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "top_features_by_mean_abs_shap.csv", index=False)
        shutil.copy2(output_dir / "top_features_by_mean_abs_shap.csv", results_dir / "top_features_by_mean_abs_shap.csv")
    if grouped_feature_rows:
        grouped_path = output_dir / "shap_grouped_feature_importance.csv"
        pd.DataFrame(grouped_feature_rows).to_csv(grouped_path, index=False)
        shutil.copy2(grouped_path, results_dir / "shap_grouped_feature_importance.csv")
    if group_rows:
        group_path = output_dir / "shap_group_contribution_summary.csv"
        group_frame = pd.DataFrame(group_rows)
        group_frame.to_csv(group_path, index=False)
        group_figures = write_shap_group_figures(group_frame, output_dir)
        for figure in group_figures:
            figure_manifest.append(
                {
                    "model": figure.get("model", ""),
                    "figure_type": "group_contribution",
                    "group_attribute": figure.get("group_attribute", ""),
                    "path": figure["path"],
                    "filename": Path(figure["path"]).name,
                    "reportable": True,
                    "reason": "categorical_group_contribution_summary",
                }
            )
        shutil.copy2(group_path, results_dir / "shap_group_contribution_summary.csv")
    if figure_manifest:
        manifest_frame = pd.DataFrame(figure_manifest)
        manifest_frame.to_csv(output_dir / "shap_figure_manifest.csv", index=False)
        shutil.copy2(output_dir / "shap_figure_manifest.csv", results_dir / "shap_figure_manifest.csv")
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "shap_failures.csv", index=False)
    print(f"[shap] Saved SHAP artifacts in: {output_dir}")
