from __future__ import annotations

import math
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from model.full_holistic.constants import RANDOM_STATE, TARGET
from model.full_holistic.paths import DATA_PATH


MODEL_NAME = "baseline | CatBoost"
OUTPUT_FOLDER = "umap_error_analysis"
MAX_LOW_RISK_TN = 5_000
MAX_SVD_COMPONENTS = 40
TOPK_LEVELS = (0.005, 0.01, 0.05)
SENTINEL_TO_NA = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]


def _require_umap():
    try:
        from umap import UMAP
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("UMAP is unavailable. Install `umap-learn` in DL-env to run this diagnostic stage.") from exc
    return UMAP


def _load_catboost_scores(results_dir: Path) -> pd.DataFrame:
    scores_path = results_dir / "test_scores.csv"
    if not scores_path.exists():
        scores_path = results_dir / "candidate_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError("Missing test scores. Expected `test_scores.csv` or `candidate_scores.csv` in results dir.")
    scores = pd.read_csv(scores_path)
    scores = scores[(scores["split"] == "test") & (scores["model"] == MODEL_NAME)].copy()
    if scores.empty:
        raise ValueError(f"No test scores found for `{MODEL_NAME}`.")
    return scores.sort_values("row_id").reset_index(drop=True)


def _load_threshold(results_dir: Path, scores: pd.DataFrame) -> float:
    registry_path = results_dir / "candidate_registry.csv"
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
        row = registry[registry["model"] == MODEL_NAME]
        if not row.empty and "selected_threshold" in row.columns:
            return float(row.iloc[0]["selected_threshold"])
    metrics_path = results_dir / "threshold_policy_test_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        row = metrics[(metrics["model"] == MODEL_NAME) & (metrics["threshold_policy"] == "valid_global_5pct_fpr")]
        if not row.empty:
            return float(row.iloc[0]["threshold"])
    raise ValueError("Missing CatBoost operational threshold. Run baseline-search or operational-thresholds first.")


def _merge_original_rows(scores: pd.DataFrame) -> pd.DataFrame:
    usecols = None
    original = pd.read_csv(DATA_PATH, usecols=usecols)
    original = original.reset_index(names="row_id")
    if "row_id" in scores.columns:
        merged = scores.merge(original, on="row_id", how="left", suffixes=("_score", ""))
        missing = int(merged[TARGET].isna().sum()) if TARGET in merged.columns else len(merged)
        if missing:
            raise ValueError(f"`row_id` merge failed for {missing} test rows.")
        print("[umap-error-analysis] merged scores with Base.csv using row_id.")
        return merged
    print("[umap-error-analysis] row_id was not available; assuming test scores preserve Base.csv test row order.")
    test_original = original[original["month"] == int(scores["month"].iloc[0])].copy().reset_index(drop=True)
    if len(test_original) != len(scores):
        raise ValueError("Cannot safely align test scores by row order; row counts differ.")
    return pd.concat([scores.reset_index(drop=True), test_original.reset_index(drop=True)], axis=1)


def _assign_labels(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["catboost_score"] = frame["score_raw"].astype(float)
    frame["predicted_positive"] = frame["catboost_score"] >= threshold
    y = frame[TARGET].astype(int)
    frame["error_type"] = np.select(
        [
            (y == 1) & frame["predicted_positive"],
            (y == 0) & frame["predicted_positive"],
            (y == 1) & ~frame["predicted_positive"],
            (y == 0) & ~frame["predicted_positive"],
        ],
        ["TP", "FP", "FN", "TN"],
        default="unknown",
    )
    n = len(frame)
    order = frame["catboost_score"].rank(method="first", ascending=False)
    for pct in TOPK_LEVELS:
        k = max(1, int(math.ceil(n * pct)))
        frame[f"top_{pct:g}"] = order <= k
    return frame


def _sample_rows(frame: pd.DataFrame) -> pd.DataFrame:
    include = frame[TARGET].astype(int).eq(1)
    for pct in TOPK_LEVELS:
        include |= frame[f"top_{pct:g}"]
    include |= frame["error_type"].isin(["TP", "FP", "FN"])
    selected = frame[include].copy()
    low_risk_tn = frame[(~include) & (frame["error_type"] == "TN")].copy()
    if len(low_risk_tn) > MAX_LOW_RISK_TN:
        low_risk_tn = low_risk_tn.sample(n=MAX_LOW_RISK_TN, random_state=RANDOM_STATE)
    return pd.concat([selected, low_risk_tn], axis=0).drop_duplicates("row_id").sample(frac=1.0, random_state=RANDOM_STATE)


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    leakage = {
        TARGET,
        "y_true",
        "fraud_bool_score",
        "fraud_bool",
        "model",
        "split",
        "score_raw",
        "catboost_score",
        "predicted_positive",
        "error_type",
        "row_id",
        "alert_count",
    }
    leakage.update({col for col in frame.columns if col.startswith("top_")})
    leakage.update({col for col in frame.columns if col.endswith("_score")})
    features = frame.drop(columns=[col for col in leakage if col in frame.columns], errors="ignore").copy()
    for column in SENTINEL_TO_NA:
        if column in features.columns:
            features[column] = features[column].replace(-1, np.nan)
    return features


def _embed(features: pd.DataFrame) -> tuple[np.ndarray, bool]:
    from scipy import sparse
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, RobustScaler

    UMAP = _require_umap()
    nums = [col for col in features.columns if is_numeric_dtype(features[col])]
    cats = [col for col in features.columns if col not in nums]
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=True)
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]), nums),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)]), cats),
        ],
        remainder="drop",
    )
    encoded = preprocess.fit_transform(features)
    reduced = False
    n_features = encoded.shape[1]
    if n_features > 50:
        n_components = min(MAX_SVD_COMPONENTS, n_features - 1, encoded.shape[0] - 1)
        encoded = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE).fit_transform(encoded)
        reduced = True
    elif sparse.issparse(encoded):
        encoded = encoded.toarray()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state.*")
        embedding = UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=RANDOM_STATE).fit_transform(encoded)
    return embedding, reduced


def _scatter(ax, x, y, **kwargs):
    ax.scatter(x, y, s=7, linewidths=0, alpha=0.72, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])


def _save_fraud_plot(frame: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    colors = frame[TARGET].astype(int).map({0: "#8a96a3", 1: "#d62728"})
    _scatter(ax, frame["umap_x"], frame["umap_y"], c=colors)
    ax.set_title("UMAP: fraud vs legit")
    for label, color in [("legit", "#8a96a3"), ("fraud", "#d62728")]:
        ax.scatter([], [], c=color, s=24, label=label)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "umap_01_fraud_vs_legit.png")
    plt.close(fig)


def _save_score_plot(frame: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    sc = ax.scatter(frame["umap_x"], frame["umap_y"], c=frame["catboost_score"], cmap="viridis", s=7, linewidths=0, alpha=0.72)
    for pct, color, size in [(0.05, "#ffffff", 16), (0.01, "#ffbf00", 25), (0.005, "#ff2d55", 36)]:
        col = f"top_{pct:g}"
        subset = frame[frame[col]]
        ax.scatter(subset["umap_x"], subset["umap_y"], facecolors="none", edgecolors=color, s=size, linewidths=0.7, label=f"top {pct:.1%}")
    ax.set_title("UMAP: CatBoost score and top-k regions")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="CatBoost score")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "umap_02_score_and_topk.png")
    plt.close(fig)


def _save_error_plot(frame: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"TN": "#b8c0cc", "FP": "#ff7f0e", "FN": "#1f77b4", "TP": "#d62728"}
    order = ["TN", "FP", "FN", "TP"]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    for label in order:
        subset = frame[frame["error_type"] == label]
        if subset.empty:
            continue
        ax.scatter(subset["umap_x"], subset["umap_y"], c=colors[label], s=8 if label == "TN" else 13, linewidths=0, alpha=0.55 if label == "TN" else 0.84, label=label)
    ax.set_title("UMAP: threshold error types")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "umap_03_error_types.png")
    plt.close(fig)


def _save_anomaly_plot(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    anomaly_cols = [
        col
        for col in ["isolation_forest_anomaly_score", "lof_anomaly_score", "autoencoder_reconstruction_error"]
        if col in frame.columns
    ]
    if not anomaly_cols:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
        ax.scatter(frame["umap_x"], frame["umap_y"], c="#b8c0cc", s=7, linewidths=0, alpha=0.55)
        ax.text(0.5, 0.5, "No anomaly score columns found", transform=ax.transAxes, ha="center", va="center", fontsize=13)
        ax.set_title("UMAP: anomaly scores")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(output_dir / "umap_04_anomaly_scores.png")
        plt.close(fig)
        return []
    n = len(anomaly_cols)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5), dpi=160, squeeze=False)
    for ax, col in zip(axes.ravel(), anomaly_cols):
        values = pd.to_numeric(frame[col], errors="coerce")
        sc = ax.scatter(frame["umap_x"], frame["umap_y"], c=values, cmap="magma", s=7, linewidths=0, alpha=0.72)
        ax.set_title(col)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_dir / "umap_04_anomaly_scores.png")
    plt.close(fig)
    return anomaly_cols


def _nn_summary(frame: pd.DataFrame) -> dict:
    from sklearn.neighbors import NearestNeighbors

    coords = frame[["umap_x", "umap_y"]].to_numpy()
    n_neighbors = min(16, len(frame))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    idx = nn.kneighbors(coords, return_distance=False)[:, 1:]
    fraud = frame[TARGET].astype(int).to_numpy()
    error = frame["error_type"].to_numpy()
    fraud_rows = np.where(fraud == 1)[0]
    fp_rows = np.where(error == "FP")[0]
    top5 = frame["top_0.05"].to_numpy(dtype=bool)
    overall_std = float(np.mean(np.std(coords, axis=0)))
    top5_std = float(np.mean(np.std(coords[top5], axis=0))) if top5.any() else np.nan
    fraud_legit_neighbor_share = float(np.mean(fraud[idx[fraud_rows]] == 0)) if len(fraud_rows) else np.nan
    fp_tp_neighbor_share = float(np.mean(error[idx[fp_rows]] == "TP")) if len(fp_rows) else np.nan
    return {
        "fraud_legit_neighbor_share": fraud_legit_neighbor_share,
        "fp_tp_neighbor_share": fp_tp_neighbor_share,
        "top5_dispersion_ratio": top5_std / overall_std if overall_std else np.nan,
    }


def _print_summary(frame: pd.DataFrame, threshold: float, reduced: bool, anomaly_cols: list[str]) -> None:
    counts = {
        "sample_size": int(len(frame)),
        "fraud": int(frame[TARGET].astype(int).sum()),
        "legit": int((frame[TARGET].astype(int) == 0).sum()),
        "error_types": frame["error_type"].value_counts().to_dict(),
        "top_0.5pct": int(frame["top_0.005"].sum()),
        "top_1pct": int(frame["top_0.01"].sum()),
        "top_5pct": int(frame["top_0.05"].sum()),
    }
    nn = _nn_summary(frame)
    fraud_mixed = nn["fraud_legit_neighbor_share"] >= 0.55
    fp_overlap = nn["fp_tp_neighbor_share"] >= 0.08
    topk_concentrated = nn["top5_dispersion_ratio"] < 0.75
    print("[umap-error-analysis] summary")
    print(f"- Model: {MODEL_NAME}")
    print(f"- Operational threshold: {threshold:.6f} selected outside this stage; UMAP does not tune thresholds.")
    print(f"- Final sample size/counts: {counts}")
    print(f"- Encoded matrix used dimensionality reduction before UMAP: {reduced}")
    print(f"- Fraud clustering: {'mixed with legitimate cases' if fraud_mixed else 'some local clustering visible'}; fraud-neighbor legit share={nn['fraud_legit_neighbor_share']:.3f}.")
    print(f"- FP/TP overlap: {'visual overlap likely' if fp_overlap else 'limited TP-neighbor signal in 2D'}; FP-neighbor TP share={nn['fp_tp_neighbor_share']:.3f}.")
    print(f"- High-risk top-k region: {'concentrated' if topk_concentrated else 'diffuse'}; top5 dispersion ratio={nn['top5_dispersion_ratio']:.3f}.")
    if anomaly_cols:
        print(f"- Anomaly scores plotted: {', '.join(anomaly_cols)}. Inspect alignment visually; no causal claim is made.")
    else:
        print("- Anomaly scores: no anomaly score columns were available in the merged test rows.")
    print("- Interpretation: this is a 2D diagnostic projection only, not model or threshold selection.")
    print("- Follow-up: use plots to guide targeted error slices; current evidence mainly supports top-k/low-FPR review rather than claims of clean separability.")


def run(config, results_dir: Path, *, force: bool = False, **_) -> None:
    output_dir = results_dir / OUTPUT_FOLDER
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = _load_catboost_scores(results_dir)
    threshold = _load_threshold(results_dir, scores)
    merged = _assign_labels(_merge_original_rows(scores), threshold)
    sampled = _sample_rows(merged)
    features = _feature_frame(sampled)
    embedding, reduced = _embed(features)
    sampled = sampled.copy()
    sampled["umap_x"] = embedding[:, 0]
    sampled["umap_y"] = embedding[:, 1]
    _save_fraud_plot(sampled, output_dir)
    _save_score_plot(sampled, output_dir)
    _save_error_plot(sampled, output_dir)
    anomaly_cols = _save_anomaly_plot(sampled, output_dir)
    _print_summary(sampled, threshold, reduced, anomaly_cols)
