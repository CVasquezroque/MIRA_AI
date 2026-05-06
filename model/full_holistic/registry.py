from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from model.full_holistic.paths import STAGE_DIRS
from model.full_holistic.utils.io import DependencyError
from model.full_holistic.utils.serialization import dump_joblib, load_joblib


MODELING_STAGES = [
    "baseline-search",
    "balance-gate",
    "advanced-features-gate",
    "anomaly-recency-gate",
    "imbalance-ensemble-gate",
    "hyperparameter-tuning-gate",
    "catboost-refit",
]


def safe_name(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")[:120] or "model"


def candidate_without_spec(candidate: dict) -> dict:
    return {key: value for key, value in candidate.items() if key != "spec"}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _aligned_score_rows(model_name: str, split: str, frame: pd.DataFrame, y_true, scores) -> pd.DataFrame:
    y_series = pd.Series(y_true)
    if y_series.index.isin(frame.index).all():
        aligned = frame.loc[y_series.index]
    else:
        aligned = frame.reset_index(drop=True)
        y_series = y_series.reset_index(drop=True)
    result = pd.DataFrame(
        {
            "model": model_name,
            "split": split,
            "row_id": aligned.index.to_numpy(),
            "month": aligned["month"].to_numpy() if "month" in aligned.columns else pd.NA,
            "y_true": y_series.astype(int).to_numpy(),
            "fraud_bool": y_series.astype(int).to_numpy(),
            "score_raw": pd.Series(scores, dtype="float64").to_numpy(),
        }
    )
    return result


def save_candidate_artifacts(
    *,
    stage_dir: Path,
    candidates: list[dict],
    all_metrics: list[dict],
    fitted_registry: dict[str, dict],
    context,
    top_n_models_to_save: int,
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    model_dir = stage_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = [candidate_without_spec(candidate) for candidate in candidates]
    candidate_frame = pd.DataFrame(candidate_rows)
    if not candidate_frame.empty and "validation_pr_auc" in candidate_frame.columns:
        candidate_frame = candidate_frame.sort_values("validation_pr_auc", ascending=False)
    candidate_frame.to_csv(stage_dir / "candidates.csv", index=False)

    specs = {candidate["model"]: candidate.get("spec", {}) for candidate in candidates}
    write_json(stage_dir / "candidate_specs.json", specs)

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(stage_dir / "metrics.csv", index=False)
    if not metrics.empty and "split" in metrics.columns:
        metrics[metrics["split"] == "validation"].to_csv(stage_dir / "validation_metrics.csv", index=False)
        metrics[metrics["split"] == "test"].to_csv(stage_dir / "test_metrics.csv", index=False)

    validation_score_frames = []
    test_score_frames = []
    for candidate in candidates:
        info = fitted_registry.get(candidate["model"])
        if info is None:
            continue
        validation_score_frames.append(
            _aligned_score_rows(
                candidate["model"],
                "validation",
                context.valid_eval,
                info["y_valid"],
                info["valid_scores"],
            )
        )
        test_score_frames.append(
            _aligned_score_rows(
                candidate["model"],
                "test",
                context.test_eval,
                info["y_test"],
                info["test_scores"],
            )
        )

    validation_scores = pd.concat(validation_score_frames, ignore_index=True) if validation_score_frames else pd.DataFrame()
    test_scores = pd.concat(test_score_frames, ignore_index=True) if test_score_frames else pd.DataFrame()
    validation_scores.to_csv(stage_dir / "validation_scores.csv", index=False)
    test_scores.to_csv(stage_dir / "test_scores.csv", index=False)
    pd.concat([validation_scores, test_scores], ignore_index=True).to_csv(stage_dir / "scores.csv", index=False)

    saved_models: dict[str, str] = {}
    if top_n_models_to_save > 0 and not candidate_frame.empty:
        selected_models = candidate_frame.head(top_n_models_to_save)["model"].tolist()
        recommended = candidate_frame.sort_values(["validation_pr_auc", "test_pr_auc"], ascending=False).head(1)["model"].tolist()
        for model_name in dict.fromkeys(selected_models + recommended):
            info = fitted_registry.get(model_name)
            if info is None:
                continue
            filename = f"{safe_name(model_name)}.joblib"
            artifact = {
                "fitted": info.get("fitted"),
                "model_kind": info.get("model_kind"),
                "X_valid": info.get("X_valid"),
                "y_valid": info.get("y_valid"),
                "valid_scores": info.get("valid_scores"),
                "X_test": info.get("X_test"),
                "y_test": info.get("y_test"),
                "test_scores": info.get("test_scores"),
                "candidate": info.get("candidate"),
            }
            dump_joblib(artifact, model_dir / filename)
            saved_models[model_name] = str(Path("models") / filename)
    write_json(stage_dir / "model_artifacts.json", saved_models)


def _stage_paths(results_dir: Path) -> list[Path]:
    paths = []
    for stage in MODELING_STAGES:
        directory_name = STAGE_DIRS.get(stage)
        if directory_name:
            paths.append(results_dir / directory_name)
    return paths


def merge_candidate_registry(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    candidate_frames = []
    specs: dict[str, dict] = {}
    score_frames = []
    metric_frames = []

    for stage_path in _stage_paths(results_dir):
        candidate_path = stage_path / "candidates.csv"
        if candidate_path.exists():
            frame = pd.read_csv(candidate_path)
            if not frame.empty:
                candidate_frames.append(frame)
        specs.update(read_json(stage_path / "candidate_specs.json"))
        score_path = stage_path / "scores.csv"
        if score_path.exists():
            score_frame = pd.read_csv(score_path)
            if not score_frame.empty:
                score_frames.append(score_frame)
        metrics_path = stage_path / "metrics.csv"
        if metrics_path.exists():
            metric_frame = pd.read_csv(metrics_path)
            if not metric_frame.empty:
                metric_frames.append(metric_frame)

    registry = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    if not registry.empty:
        registry = registry.drop_duplicates(subset=["model"], keep="last")
        registry = registry.sort_values("validation_pr_auc", ascending=False)
    registry.to_csv(results_dir / "candidate_registry.csv", index=False)
    registry.to_csv(results_dir / "holistic_candidate_ranking.csv", index=False)
    write_json(results_dir / "candidate_specs.json", specs)
    write_json(results_dir / "holistic_candidate_specs.json", specs)

    scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    if not scores.empty:
        scores = scores.drop_duplicates(subset=["model", "split", "row_id"], keep="last")
    scores.to_csv(results_dir / "candidate_scores.csv", index=False)
    if not scores.empty:
        scores[scores["split"] == "validation"].to_csv(results_dir / "validation_scores.csv", index=False)
        scores[scores["split"] == "test"].to_csv(results_dir / "test_scores.csv", index=False)

    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    metrics.to_csv(results_dir / "holistic_all_metrics.csv", index=False)
    if not metrics.empty and {"threshold_policy", "split"}.issubset(metrics.columns):
        main = metrics[metrics["threshold_policy"] == "valid_global_5pct_fpr"].copy()
        main[main["split"] == "validation"].to_csv(results_dir / "holistic_validation_main_threshold.csv", index=False)
        main[main["split"] == "test"].to_csv(results_dir / "holistic_test_main_threshold.csv", index=False)
    return registry, specs


def load_candidate_registry(results_dir: Path, required: bool = True) -> list[dict]:
    registry_path = results_dir / "candidate_registry.csv"
    if not registry_path.exists():
        merge_candidate_registry(results_dir)
    if not registry_path.exists() or registry_path.stat().st_size == 0:
        if required:
            raise DependencyError(
                "Missing candidate registry. Please run --stage baseline-search first, or skip stages that depend on it."
            )
        return []
    frame = pd.read_csv(registry_path)
    if frame.empty:
        if required:
            raise DependencyError(
                "Missing candidate registry. Please run --stage baseline-search first, or skip stages that depend on it."
            )
        return []
    specs = read_json(results_dir / "candidate_specs.json")
    rows = frame.to_dict("records")
    for row in rows:
        row["spec"] = specs.get(row["model"], {})
    return rows


def load_stage_candidates(results_dir: Path, stage: str, message: str) -> list[dict]:
    directory = results_dir / STAGE_DIRS[stage]
    path = directory / "candidates.csv"
    if not path.exists():
        raise DependencyError(message)
    frame = pd.read_csv(path)
    if frame.empty:
        raise DependencyError(message)
    specs = read_json(directory / "candidate_specs.json")
    rows = frame.to_dict("records")
    for row in rows:
        row["spec"] = specs.get(row["model"], {})
    return rows


def load_candidates_for_stages(results_dir: Path, stages: list[str], required: bool = True) -> list[dict]:
    rows: list[dict] = []
    seen_models: set[str] = set()
    for stage in stages:
        directory = results_dir / STAGE_DIRS[stage]
        path = directory / "candidates.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        specs = read_json(directory / "candidate_specs.json")
        for row in frame.sort_values("validation_pr_auc", ascending=False).to_dict("records"):
            if row["model"] in seen_models:
                continue
            row["spec"] = specs.get(row["model"], {})
            rows.append(row)
            seen_models.add(row["model"])
    if required and not rows:
        raise DependencyError(
            "Missing upstream candidate outputs. Please run the required previous stages first."
        )
    return rows


def load_scores(results_dir: Path, required: bool = True) -> pd.DataFrame:
    path = results_dir / "candidate_scores.csv"
    if not path.exists():
        merge_candidate_registry(results_dir)
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise DependencyError(
                "Missing candidate scores. Please run --stage baseline-search first, or skip stages that depend on it."
            )
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty and required:
        raise DependencyError(
            "Missing candidate scores. Please run --stage baseline-search first, or skip stages that depend on it."
        )
    return frame


def load_model_artifacts(results_dir: Path, model_names: set[str] | None = None) -> dict[str, dict]:
    artifacts: dict[str, dict] = {}
    for stage_path in _stage_paths(results_dir):
        mapping = read_json(stage_path / "model_artifacts.json")
        for model_name, relative_path in mapping.items():
            if model_names is not None and model_name not in model_names:
                continue
            artifact_path = stage_path / relative_path
            if artifact_path.exists():
                artifacts[model_name] = load_joblib(artifact_path)
    return artifacts


def fitted_registry_from_scores(results_dir: Path, candidates: list[dict]) -> dict[str, dict]:
    scores = load_scores(results_dir, required=True)
    by_candidate = {candidate["model"]: candidate for candidate in candidates}
    fitted_registry: dict[str, dict] = {}
    for model_name, group in scores.groupby("model"):
        candidate = by_candidate.get(model_name)
        if candidate is None:
            continue
        valid = group[group["split"] == "validation"].sort_values("row_id")
        test = group[group["split"] == "test"].sort_values("row_id")
        fitted_registry[model_name] = {
            "fitted": None,
            "model_kind": "scores_only",
            "candidate": candidate,
            "y_valid": valid["y_true"].astype(int),
            "valid_scores": valid["score_raw"].astype(float),
            "y_test": test["y_true"].astype(int),
            "test_scores": test["score_raw"].astype(float),
        }
    return fitted_registry
