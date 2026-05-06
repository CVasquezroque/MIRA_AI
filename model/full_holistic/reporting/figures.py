from __future__ import annotations

from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd

from model.full_holistic.constants import MAIN_FPR_CAP
from model.full_holistic.data.context import load_context_optional
from model.full_holistic.paths import STAGE_DIRS
from model.full_holistic.utils.io import optional_read_csv


TOP_N = 8
PALETTE = [
    "#2F6F9F",
    "#5A8F57",
    "#C47A2C",
    "#8E5EA2",
    "#52796F",
    "#B65D4B",
    "#4D7EA8",
    "#7A6C5D",
]
SPLIT_COLORS = {"train": "#2F6F9F", "validation": "#C47A2C", "test": "#B65D4B"}
BENEFIT_METRICS = {"validation_pr_auc", "test_pr_auc", "main_precision", "main_recall_tpr", "precision_at_k_top1pct", "lift_at_k_top1pct"}
HARM_METRICS = {"main_fdr", "main_fpr", "worst_max_fpr_gap", "worst_equalized_odds_difference"}
METRIC_LABELS = {
    "validation_pr_auc": "Validation PR-AUC",
    "test_pr_auc": "Test PR-AUC",
    "main_precision": "Precision @ FPR<=5%",
    "main_fdr": "FDR @ FPR<=5%",
    "main_recall_tpr": "Recall @ FPR<=5%",
    "main_fpr": "Test FPR",
    "precision_at_k_top1pct": "Precision @ Top 1%",
    "lift_at_k_top1pct": "Lift @ Top 1%",
    "worst_max_fpr_gap": "Max FPR gap",
}


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="notebook")
    except Exception:
        sns = None
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 14,
            "axes.titleweight": "bold",
        }
    )
    return plt, sns


def figure_dir(stage_dir: Path) -> Path:
    path = stage_dir / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def wrap_label(value: str, *, width: int = 24, max_chars: int = 70) -> str:
    text = str(value).replace(" | ", " / ").replace("_", " ")
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def slug(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).lower()).strip("_")
    return clean[:90] or "figure"


def save_figure(fig, output_dir: Path, filename: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    return path.as_posix()


def read_first(results_dir: Path, *relative_paths: str) -> pd.DataFrame:
    for relative in relative_paths:
        frame = optional_read_csv(results_dir / relative)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def model_names_from_candidates(candidates, *, top_n: int = TOP_N) -> list[str]:
    names = []
    for candidate in candidates[:top_n]:
        if isinstance(candidate, dict):
            names.append(str(candidate.get("model")))
        else:
            names.append(str(candidate))
    return [name for name in names if name and name != "None"]


def top_models_from_decision_table(decision_table: pd.DataFrame, n: int = TOP_N) -> list[str]:
    if decision_table.empty or "model" not in decision_table.columns:
        return []
    sort_cols = [col for col in ["recommended_final_model", "validation_pr_auc", "test_pr_auc"] if col in decision_table.columns]
    ascending = [False] + [False] * (len(sort_cols) - 1)
    return decision_table.sort_values(sort_cols, ascending=ascending).head(n)["model"].astype(str).tolist()


def recommended_model(decision_table: pd.DataFrame) -> str | None:
    if decision_table.empty or "recommended_final_model" not in decision_table.columns:
        return None
    recommended = decision_table[decision_table["recommended_final_model"].astype(str).str.lower().eq("true")]
    if recommended.empty:
        return str(decision_table.iloc[0]["model"]) if "model" in decision_table.columns else None
    return str(recommended.iloc[0]["model"])


def _colors_for(names: list[str]) -> dict[str, str]:
    return {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(names)}


def _legend_outside(ax, *, columns: int = 1) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, ncol=columns)


def _existing_figure(results_dir: Path, stage: str, filename: str) -> str | None:
    path = results_dir / STAGE_DIRS[stage] / "figures" / filename
    return path.as_posix() if path.exists() else None


def _data_for_report(results_dir: Path, decision_table: pd.DataFrame) -> dict[str, object]:
    return {
        "context": load_context_optional(results_dir),
        "drift": read_first(results_dir, f"{STAGE_DIRS['data-audit']}/00_monthly_fraud_rate_drift.csv"),
        "threshold": read_first(results_dir, "threshold_policy_test_metrics.csv", f"{STAGE_DIRS['operational-thresholds']}/threshold_policy_test_metrics.csv"),
        "low_fpr": read_first(results_dir, "low_fpr_sweep_test_metrics.csv", f"{STAGE_DIRS['operational-thresholds']}/low_fpr_sweep_test_metrics.csv"),
        "topk": read_first(results_dir, "topk_alert_metrics.csv", f"{STAGE_DIRS['topk']}/topk_alert_metrics.csv"),
        "fairness_group": read_first(results_dir, "fairness_by_group.csv", f"{STAGE_DIRS['fairness']}/fairness_by_group.csv"),
        "fairness_disparity": read_first(results_dir, "fairness_disparity_summary.csv", f"{STAGE_DIRS['fairness']}/fairness_disparity_summary.csv"),
        "shap": read_first(results_dir, "top_features_by_mean_abs_shap.csv", f"{STAGE_DIRS['shap']}/top_features_by_mean_abs_shap.csv"),
        "shap_group": read_first(results_dir, "shap_group_contribution_summary.csv", f"{STAGE_DIRS['shap']}/shap_group_contribution_summary.csv"),
        "ablation": read_first(results_dir, "feature_ablation_metrics.csv", f"{STAGE_DIRS['feature-ablation']}/feature_ablation_metrics.csv"),
        "tuning": read_first(results_dir, f"{STAGE_DIRS['hyperparameter-tuning-gate']}/tuned_vs_fixed_catboost_fpr5.csv"),
        "decision": decision_table,
        "results_dir": results_dir,
    }


def write_data_audit_figures(drift: pd.DataFrame, output_dir: Path, *, context=None) -> list[dict[str, str]]:
    if drift.empty or "month" not in drift.columns:
        return []
    rate_col = "fraud_rate" if "fraud_rate" in drift.columns else "fraud_prevalence"
    if rate_col not in drift.columns:
        return []
    plt, _ = setup_matplotlib()
    frame = drift.copy()
    if "split" not in frame.columns:
        frame["split"] = "train"
        if context is not None:
            frame["split"] = frame["month"].map(
                lambda month: "train" if month in context.train_months else ("validation" if month == context.valid_month else "test")
            )
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bars = ax.bar(
        frame["month"].astype(str),
        frame[rate_col].astype(float) * 100,
        color=[SPLIT_COLORS.get(split, "#777777") for split in frame["split"]],
    )
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_title("Fraud Prevalence By Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Fraud prevalence (%)")
    ax.margins(y=0.14)
    path = save_figure(fig, figure_dir(output_dir), "01_fraud_prevalence_by_month.png")
    plt.close(fig)
    return [{"title": "Fraud prevalence by month", "path": path, "caption": "Monthly fraud prevalence with temporal split coloring."}]


def write_threshold_figures(
    low_fpr_frame: pd.DataFrame,
    threshold_frame: pd.DataFrame,
    output_dir: Path,
    *,
    model_names: list[str] | None = None,
    focus_model: str | None = None,
    include_top_tradeoff: bool = True,
    include_focus_tradeoff: bool = True,
    include_confusion: bool = True,
) -> list[dict[str, str]]:
    plt, sns = setup_matplotlib()
    generated: list[dict[str, str]] = []
    fig_dir = figure_dir(output_dir)
    selected = model_names or []
    colors = _colors_for(selected)

    if include_top_tradeoff and not low_fpr_frame.empty and {"model", "fpr_cap", "precision", "recall_tpr", "fdr"}.issubset(low_fpr_frame.columns):
        frame = low_fpr_frame[low_fpr_frame["model"].isin(selected)].copy() if selected else low_fpr_frame.copy()
        if not frame.empty:
            fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), sharex=True)
            for model, group in frame.groupby("model"):
                group = group.sort_values("fpr_cap")
                label = wrap_label(model, width=22, max_chars=55)
                color = colors.get(model)
                axes[0].plot(group["fpr_cap"] * 100, group["precision"], marker="o", label=label, color=color)
                axes[1].plot(group["fpr_cap"] * 100, group["recall_tpr"], marker="o", label=label, color=color)
                axes[2].plot(group["fpr_cap"] * 100, group["fdr"], marker="o", label=label, color=color)
            for ax, title in zip(axes, ["Precision", "Recall", "FDR"]):
                ax.set_title(title)
                ax.set_xlabel("Validation FPR cap (%)")
                ax.set_ylim(bottom=0)
            _legend_outside(axes[-1])
            path = save_figure(fig, fig_dir, "low_fpr_tradeoff_top8.png")
            generated.append({"title": "Low-FPR trade-off across top candidates", "path": path, "caption": "Validation-selected FPR caps applied to test."})
            plt.close(fig)

    if include_focus_tradeoff and focus_model and not low_fpr_frame.empty and {"model", "fpr_cap", "precision", "recall_tpr", "fdr"}.issubset(low_fpr_frame.columns):
        frame = low_fpr_frame[low_fpr_frame["model"].eq(focus_model)].sort_values("fpr_cap")
        if not frame.empty:
            fig, ax = plt.subplots(figsize=(8.2, 4.8))
            ax.plot(frame["fpr_cap"] * 100, frame["precision"], marker="o", label="Precision", color=PALETTE[0])
            ax.plot(frame["fpr_cap"] * 100, frame["recall_tpr"], marker="o", label="Recall", color=PALETTE[1])
            ax.plot(frame["fpr_cap"] * 100, frame["fdr"], marker="o", label="FDR", color=PALETTE[5])
            ax.set_title("Final Model Threshold Trade-off")
            ax.set_xlabel("Validation FPR cap (%)")
            ax.set_ylabel("Test metric")
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False)
            path = save_figure(fig, fig_dir, "final_model_tradeoff_curve.png")
            generated.append({"title": "Final model threshold trade-off", "path": path, "caption": f"Precision, recall, and FDR for {wrap_label(focus_model, width=48)}."})
            plt.close(fig)

    needed = {"model", "threshold_policy", "tp", "fp", "fn", "tn"}
    if include_confusion and not threshold_frame.empty and needed.issubset(threshold_frame.columns):
        frame = threshold_frame[
            threshold_frame["threshold_policy"].eq("valid_global_5pct_fpr")
            & (threshold_frame["model"].isin(selected) if selected else True)
        ].copy()
        if not frame.empty:
            n = min(TOP_N, len(frame))
            cols = 4
            rows = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(13.5, 3.4 * rows))
            axes = np.atleast_1d(axes).ravel()
            for ax, (_, row) in zip(axes, frame.head(n).iterrows()):
                matrix = np.array([[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]])
                if sns:
                    sns.heatmap(
                        matrix,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        cbar=False,
                        ax=ax,
                        xticklabels=["Pred 0", "Pred 1"],
                        yticklabels=["True 0", "True 1"],
                        annot_kws={"fontsize": 8},
                    )
                else:
                    ax.imshow(matrix, cmap="Blues")
                ax.set_title(wrap_label(row["model"], width=20, max_chars=52), fontsize=9)
            for ax in axes[n:]:
                ax.axis("off")
            fig.suptitle("Test Confusion Matrices @ Validation FPR <= 5%", y=1.02)
            path = save_figure(fig, fig_dir, "confusion_matrices_top8_test.png")
            generated.append({"title": "Confusion matrices for top candidates", "path": path, "caption": "Top candidates evaluated on test at the validation-selected 5% FPR policy."})
            plt.close(fig)
    return generated


def write_topk_figures(topk_frame: pd.DataFrame, output_dir: Path, *, model_names: list[str] | None = None) -> list[dict[str, str]]:
    if topk_frame.empty or not {"model", "split", "topk_pct", "precision_at_k", "recall_at_k", "lift_at_k", "captured_frauds"}.issubset(topk_frame.columns):
        return []
    plt, _ = setup_matplotlib()
    fig_dir = figure_dir(output_dir)
    selected = model_names or sorted(topk_frame["model"].astype(str).unique())[:TOP_N]
    frame = topk_frame[topk_frame["split"].eq("test") & topk_frame["model"].isin(selected)].copy()
    if frame.empty:
        return []
    colors = _colors_for(selected)
    generated: list[dict[str, str]] = []

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.7), sharex=True)
    for model, group in frame.groupby("model"):
        group = group.sort_values("topk_pct")
        label = wrap_label(model, width=22, max_chars=55)
        color = colors.get(model)
        axes[0].plot(group["topk_pct"] * 100, group["precision_at_k"], marker="o", label=label, color=color)
        axes[1].plot(group["topk_pct"] * 100, group["recall_at_k"], marker="o", label=label, color=color)
        axes[2].plot(group["topk_pct"] * 100, group["lift_at_k"], marker="o", label=label, color=color)
    for ax, title in zip(axes, ["Precision", "Recall", "Lift"]):
        ax.set_title(title)
        ax.set_xlabel("Top-k alert budget (%)")
        ax.set_ylim(bottom=0)
    _legend_outside(axes[-1])
    path = save_figure(fig, fig_dir, "topk_precision_recall_lift_curves.png")
    generated.append({"title": "Top-k precision, recall, and lift", "path": path, "caption": "Decision metrics as alert budget increases."})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for model, group in frame.groupby("model"):
        group = group.sort_values("topk_pct")
        ax.plot(group["topk_pct"] * 100, group["captured_frauds"], marker="o", label=wrap_label(model, width=22, max_chars=55), color=colors.get(model))
    ax.set_title("Fraud Captured Versus Alert Budget")
    ax.set_xlabel("Top-k alert budget (%)")
    ax.set_ylabel("Captured frauds")
    _legend_outside(ax)
    path = save_figure(fig, fig_dir, "fraud_captured_vs_alert_budget.png")
    generated.append({"title": "Fraud captured versus alert budget", "path": path, "caption": "Absolute fraud capture count at each top-k budget."})
    plt.close(fig)
    return generated


def write_fairness_figures(
    fairness_by_group: pd.DataFrame,
    fairness_disparity: pd.DataFrame,
    output_dir: Path,
    *,
    model_names: list[str] | None = None,
    focus_model: str | None = None,
) -> list[dict[str, str]]:
    plt, sns = setup_matplotlib()
    generated: list[dict[str, str]] = []
    fig_dir = figure_dir(output_dir)
    selected = model_names or []

    if not fairness_by_group.empty and {"model", "attribute", "group", "alert_rate", "fpr", "recall_tpr", "threshold_policy"}.issubset(fairness_by_group.columns):
        candidate_model = focus_model if focus_model in set(fairness_by_group["model"].astype(str)) else None
        if candidate_model is None:
            candidate_model = str(fairness_by_group.iloc[0]["model"])
        housing = fairness_by_group[
            fairness_by_group["model"].eq(candidate_model)
            & fairness_by_group["threshold_policy"].eq("valid_global_5pct_fpr")
            & fairness_by_group["attribute"].eq("housing_status")
        ].copy()
        if not housing.empty:
            housing = housing.sort_values("alert_rate", ascending=False)
            fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
            for ax, col, title, color in zip(axes, ["alert_rate", "fpr", "recall_tpr"], ["Alert Rate", "FPR", "Recall"], [PALETTE[0], PALETTE[5], PALETTE[1]]):
                ax.bar(housing["group"].astype(str), housing[col].astype(float), color=color)
                ax.set_title(title)
                ax.tick_params(axis="x", rotation=45)
                ax.set_ylim(bottom=0)
            fig.suptitle(f"Fairness By Housing Status: {wrap_label(candidate_model, width=52)}", y=1.04)
            path = save_figure(fig, fig_dir, "fairness_housing_status_rates.png")
            generated.append({"title": "Fairness by housing status", "path": path, "caption": "Alert rate, FPR, and recall by housing status."})
            plt.close(fig)

        alerts = fairness_by_group[
            fairness_by_group["model"].eq(candidate_model)
            & fairness_by_group["threshold_policy"].eq("valid_global_5pct_fpr")
        ].copy()
        if not alerts.empty:
            alerts["label"] = alerts["attribute"].astype(str) + ": " + alerts["group"].astype(str)
            alerts = alerts.sort_values("alert_rate", ascending=False).head(18)
            fig, ax = plt.subplots(figsize=(9, 6.3))
            ax.barh(alerts["label"].map(lambda x: wrap_label(x, width=30, max_chars=58))[::-1], alerts["alert_rate"].astype(float)[::-1], color=PALETTE[0])
            ax.set_title("Highest Group Alert Rates")
            ax.set_xlabel("Alert rate")
            path = save_figure(fig, fig_dir, "fairness_alert_rate_by_group.png")
            generated.append({"title": "Group alert-rate summary", "path": path, "caption": "Highest alert-rate groups for the audited model."})
            plt.close(fig)

    if not fairness_disparity.empty and {"model", "threshold_policy", "attribute", "max_fpr_gap"}.issubset(fairness_disparity.columns):
        frame = fairness_disparity[fairness_disparity["threshold_policy"].eq("valid_global_5pct_fpr")].copy()
        if selected:
            frame = frame[frame["model"].isin(selected)]
        if not frame.empty:
            pivot = frame.pivot_table(index="model", columns="attribute", values="max_fpr_gap", aggfunc="max")
            pivot.index = [wrap_label(x, width=24, max_chars=60) for x in pivot.index]
            fig, ax = plt.subplots(figsize=(10.2, 6.3))
            if sns:
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="OrRd", ax=ax, cbar_kws={"label": "Max FPR gap"})
            else:
                ax.imshow(pivot.fillna(0), aspect="auto")
            ax.set_title("Fairness Disparity Across Main Candidates")
            ax.set_xlabel("Protected group")
            ax.set_ylabel("")
            path = save_figure(fig, fig_dir, "fairness_disparity_top8.png")
            generated.append({"title": "Fairness disparity across main candidates", "path": path, "caption": "Maximum FPR gap by protected attribute."})
            plt.close(fig)
    return generated


def _desirability_heatmap_frame(table: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = table.set_index("model")[columns].apply(pd.to_numeric, errors="coerce")
    score = raw.copy()
    for col in raw.columns:
        values = raw[col]
        min_value = values.min(skipna=True)
        max_value = values.max(skipna=True)
        if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
            score[col] = 0.5
            continue
        normalized = (values - min_value) / (max_value - min_value)
        score[col] = 1.0 - normalized if col in HARM_METRICS else normalized
    return raw, score


def write_decision_summary_figures(
    decision_table: pd.DataFrame,
    threshold_frame: pd.DataFrame,
    output_dir: Path,
    *,
    top_models: list[str],
) -> list[dict[str, str]]:
    if decision_table.empty:
        return []
    plt, sns = setup_matplotlib()
    fig_dir = figure_dir(output_dir)
    generated: list[dict[str, str]] = []
    table = decision_table.head(TOP_N).copy()
    labels = table["model"].map(lambda value: wrap_label(value, width=24, max_chars=62))

    if {"validation_pr_auc", "main_recall_tpr"}.issubset(table.columns):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharey=True)
        axes[0].barh(labels, table["validation_pr_auc"].astype(float), color=PALETTE[0])
        axes[1].barh(labels, table["main_recall_tpr"].astype(float), color=PALETTE[1])
        axes[0].set_title("Validation PR-AUC")
        axes[1].set_title(f"Test Recall @ FPR <= {MAIN_FPR_CAP:.0%}")
        for ax in axes:
            ax.invert_yaxis()
            ax.set_xlabel("Metric value")
        path = save_figure(fig, fig_dir, "candidate_ranking_pr_auc_recall_fpr5.png")
        generated.append({"title": "Candidate ranking by PR-AUC and recall", "path": path, "caption": "Ranking contrasts model discrimination with operational fraud capture."})
        plt.close(fig)

    heatmap_cols = ["main_precision", "main_fdr", "main_recall_tpr", "main_fpr", "precision_at_k_top1pct", "lift_at_k_top1pct", "worst_max_fpr_gap"]
    heatmap_cols = [col for col in heatmap_cols if col in table.columns]
    if heatmap_cols and "model" in table.columns:
        raw, score = _desirability_heatmap_frame(table, heatmap_cols)
        raw.index = [wrap_label(x, width=24, max_chars=62) for x in raw.index]
        score.index = raw.index
        score.columns = [METRIC_LABELS.get(col, col) for col in score.columns]
        annot = raw.copy()
        annot.columns = score.columns
        fig, ax = plt.subplots(figsize=(11.5, 6.6))
        if sns:
            sns.heatmap(score, annot=annot, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Column-normalized desirability"})
        else:
            ax.imshow(score.fillna(0.5), aspect="auto")
        ax.set_title("Operational Metrics By Candidate")
        ax.set_xlabel("")
        ax.set_ylabel("")
        path = save_figure(fig, fig_dir, "operational_metrics_desirability_heatmap_top8.png")
        generated.append({"title": "Operational metrics heatmap", "path": path, "caption": "Cell color is normalized per metric so higher color intensity is always better; annotations show raw values."})
        plt.close(fig)

    generated.extend(
        write_threshold_figures(
            pd.DataFrame(),
            threshold_frame,
            output_dir,
            model_names=top_models,
            include_top_tradeoff=False,
            include_focus_tradeoff=False,
            include_confusion=True,
        )
    )
    return generated


def write_ablation_figures(ablation_frame: pd.DataFrame, output_dir: Path) -> list[dict[str, str]]:
    if ablation_frame.empty or not {"feature_set", "validation_pr_auc", "test_pr_auc"}.issubset(ablation_frame.columns):
        return []
    plt, _ = setup_matplotlib()
    frame = ablation_frame.copy()
    family = frame["model_family"].astype(str) if "model_family" in frame.columns else pd.Series(["model"] * len(frame))
    frame["label"] = frame["feature_set"].astype(str) + "\n" + family
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.bar(x - 0.18, frame["validation_pr_auc"].astype(float), width=0.36, color=PALETTE[0], label="Validation PR-AUC")
    ax.bar(x + 0.18, frame["test_pr_auc"].astype(float), width=0.36, color=PALETTE[1], label="Test PR-AUC")
    ax.set_xticks(x, frame["label"].map(lambda value: wrap_label(value, width=18, max_chars=46)), rotation=25, ha="right")
    ax.set_title("Feature-family Ablation")
    ax.set_ylabel("PR-AUC")
    ax.legend(frameon=False)
    path = save_figure(fig, figure_dir(output_dir), "feature_family_ablation_pr_auc.png")
    plt.close(fig)
    return [{"title": "Feature-family ablation plot", "path": path, "caption": "PR-AUC sensitivity to feature-family removal or policy changes."}]


def write_shap_group_figures(group_frame: pd.DataFrame, output_dir: Path, *, focus_model: str | None = None) -> list[dict[str, str]]:
    required = {"model", "group", "feature", "mean_abs_shap"}
    if group_frame.empty or not required.issubset(group_frame.columns):
        return []
    plt, sns = setup_matplotlib()
    model = focus_model if focus_model in set(group_frame["model"].astype(str)) else str(group_frame.iloc[0]["model"])
    frame = group_frame[group_frame["model"].eq(model)].copy()
    if frame.empty:
        return []
    if "group_attribute" not in frame.columns:
        frame["group_attribute"] = "housing_status"

    generated: list[dict[str, str]] = []
    fig_dir = figure_dir(output_dir)
    preferred_attributes = ["housing_status", "device_os", "employment_status", "payment_type", "source"]
    available_attributes = [attr for attr in preferred_attributes if attr in set(frame["group_attribute"].astype(str))]
    available_attributes.extend([attr for attr in sorted(frame["group_attribute"].astype(str).unique()) if attr not in available_attributes])

    for group_attribute in available_attributes:
        attr_frame = frame[frame["group_attribute"].astype(str).eq(group_attribute)].copy()
        if attr_frame.empty:
            continue
        top_features = attr_frame.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False).head(12).index
        plot_frame = attr_frame[attr_frame["feature"].isin(top_features)].copy()
        if plot_frame.empty:
            continue
        if "n_rows" in plot_frame.columns:
            sizes = plot_frame.groupby("group")["n_rows"].max().to_dict()
            plot_frame["group_label"] = plot_frame["group"].astype(str).map(lambda value: f"{value} (n={int(sizes.get(value, 0))})")
        else:
            plot_frame["group_label"] = plot_frame["group"].astype(str)
        pivot = plot_frame.pivot_table(index="group_label", columns="feature", values="mean_abs_shap", aggfunc="mean")
        if pivot.empty:
            continue
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        pivot.columns = [wrap_label(col, width=18, max_chars=42) for col in pivot.columns]
        fig, ax = plt.subplots(figsize=(11.8, max(4.8, 0.45 * len(pivot) + 2.2)))
        if sns:
            sns.heatmap(pivot, cmap="YlGnBu", ax=ax, cbar_kws={"label": "Mean absolute SHAP"})
        else:
            ax.imshow(pivot.fillna(0), aspect="auto")
        ax.set_title(f"SHAP Contribution Patterns By {group_attribute}")
        ax.set_xlabel("Feature")
        ax.set_ylabel(group_attribute)
        filename = f"shap_group_contribution_heatmap_{slug(group_attribute)}.png"
        path = save_figure(fig, fig_dir, filename)
        plt.close(fig)
        generated.append(
            {
                "title": f"SHAP contribution patterns by {group_attribute}",
                "path": path,
                "caption": f"Mean absolute SHAP values by {group_attribute} group with group support shown when available.",
                "model": model,
                "group_attribute": group_attribute,
            }
        )
    return generated


def write_tuning_figures(comparison_frame: pd.DataFrame, output_dir: Path) -> list[dict[str, str]]:
    if comparison_frame.empty or not {"model", "test_pr_auc", "test_threshold_recall_tpr", "test_precision_top1pct"}.issubset(comparison_frame.columns):
        return []
    plt, sns = setup_matplotlib()
    fig_dir = figure_dir(output_dir)
    frame = comparison_frame.sort_values("test_pr_auc", ascending=False).copy()
    labels = frame["model"].map(lambda value: wrap_label(value, width=24, max_chars=60))
    generated: list[dict[str, str]] = []

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9), sharey=True)
    for ax, col, title, color in zip(
        axes,
        ["test_pr_auc", "test_threshold_recall_tpr", "test_precision_top1pct"],
        ["Test PR-AUC", "Recall @ FPR<=5%", "Precision @ Top 1%"],
        [PALETTE[0], PALETTE[1], PALETTE[2]],
    ):
        ax.barh(labels, frame[col].astype(float), color=color)
        ax.set_title(title)
        ax.invert_yaxis()
    path = save_figure(fig, fig_dir, "tuned_vs_fixed_model_comparison.png")
    generated.append({"title": "Tuned versus fixed model comparison", "path": path, "caption": "Tuned families compared with the fixed CatBoost reference."})
    plt.close(fig)

    heatmap_cols = [
        "test_pr_auc",
        "test_threshold_precision",
        "test_threshold_fdr",
        "test_threshold_recall_tpr",
        "test_threshold_fpr",
        "test_precision_top1pct",
        "test_lift_top1pct",
    ]
    heatmap_cols = [col for col in heatmap_cols if col in frame.columns]
    if heatmap_cols:
        renamed = frame.rename(
            columns={
                "test_threshold_precision": "main_precision",
                "test_threshold_fdr": "main_fdr",
                "test_threshold_recall_tpr": "main_recall_tpr",
                "test_threshold_fpr": "main_fpr",
                "test_precision_top1pct": "precision_at_k_top1pct",
                "test_lift_top1pct": "lift_at_k_top1pct",
            }
        )
        mapped_cols = [
            {"test_pr_auc": "test_pr_auc"}.get(col, {
                "test_threshold_precision": "main_precision",
                "test_threshold_fdr": "main_fdr",
                "test_threshold_recall_tpr": "main_recall_tpr",
                "test_threshold_fpr": "main_fpr",
                "test_precision_top1pct": "precision_at_k_top1pct",
                "test_lift_top1pct": "lift_at_k_top1pct",
            }.get(col, col))
            for col in heatmap_cols
        ]
        raw, score = _desirability_heatmap_frame(renamed[["model", *mapped_cols]].copy(), mapped_cols)
        raw.index = [wrap_label(x, width=24, max_chars=60) for x in raw.index]
        score.index = raw.index
        score.columns = [METRIC_LABELS.get(col, col) for col in score.columns]
        annot = raw.copy()
        annot.columns = score.columns
        fig, ax = plt.subplots(figsize=(10.5, 5.6))
        if sns:
            sns.heatmap(score, annot=annot, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Column-normalized desirability"})
        else:
            ax.imshow(score.fillna(0.5), aspect="auto")
        ax.set_title("Tuning Operational Heatmap")
        path = save_figure(fig, fig_dir, "tuning_operational_desirability_heatmap.png")
        generated.append({"title": "Tuning operational heatmap", "path": path, "caption": "Normalized tuning comparison across operational metrics."})
        plt.close(fig)
    return generated


def _report_or_stage_path(results_dir: Path, stage: str, filename: str, fallback: str | None = None) -> str | None:
    path = _existing_figure(results_dir, stage, filename)
    return path or fallback


def generate_report_figures(results_dir: Path, decision_table: pd.DataFrame) -> list[dict[str, str]]:
    output_dir = results_dir / STAGE_DIRS["final-report"]
    try:
        setup_matplotlib()
    except Exception:
        return []
    data = _data_for_report(results_dir, decision_table)
    top_models = top_models_from_decision_table(decision_table)
    focus_model = recommended_model(decision_table)
    generated: list[dict[str, str]] = []

    prevalence_stage = _existing_figure(results_dir, "data-audit", "01_fraud_prevalence_by_month.png")
    prevalence_fallback = [] if prevalence_stage else write_data_audit_figures(data["drift"], output_dir, context=data["context"])
    generated.append(
        {
            "title": "Fraud prevalence by month",
            "path": prevalence_stage or (prevalence_fallback[0]["path"] if prevalence_fallback else None),
            "caption": "Monthly fraud prevalence with train/validation/test split coloring.",
        }
    )

    generated.extend(write_decision_summary_figures(decision_table, data["threshold"], output_dir, top_models=top_models))
    generated.extend(
        write_threshold_figures(
            data["low_fpr"],
            pd.DataFrame(),
            output_dir,
            model_names=top_models,
            focus_model=focus_model,
            include_top_tradeoff=False,
            include_focus_tradeoff=True,
            include_confusion=False,
        )
    )

    topk_stage_paths = {
        "topk_precision_recall_lift_curves.png": _existing_figure(results_dir, "topk", "topk_precision_recall_lift_curves.png"),
        "fraud_captured_vs_alert_budget.png": _existing_figure(results_dir, "topk", "fraud_captured_vs_alert_budget.png"),
    }
    topk_fallback = [] if all(topk_stage_paths.values()) else write_topk_figures(data["topk"], output_dir, model_names=top_models)
    for title, filename, index in [
        ("Top-k precision, recall, and lift", "topk_precision_recall_lift_curves.png", 0),
        ("Fraud captured versus alert budget", "fraud_captured_vs_alert_budget.png", 1),
    ]:
        fallback = topk_fallback[index]["path"] if len(topk_fallback) > index else None
        path = topk_stage_paths.get(filename) or fallback
        if path:
            generated.append({"title": title, "path": path, "caption": "Top-k alert budget diagnostics on test."})

    fairness_stage_paths = {
        "fairness_housing_status_rates.png": _existing_figure(results_dir, "fairness", "fairness_housing_status_rates.png"),
        "fairness_disparity_top8.png": _existing_figure(results_dir, "fairness", "fairness_disparity_top8.png"),
    }
    fairness_fallback = [] if all(fairness_stage_paths.values()) else write_fairness_figures(data["fairness_group"], data["fairness_disparity"], output_dir, model_names=top_models, focus_model=focus_model)
    for title, filename, caption in [
        ("Fairness by housing status", "fairness_housing_status_rates.png", "Alert rate, FPR, and recall by housing status."),
        ("Fairness disparity across main candidates", "fairness_disparity_top8.png", "Maximum FPR gap by protected attribute."),
    ]:
        fallback = next((item["path"] for item in fairness_fallback if Path(item["path"]).name == filename), None)
        path = fairness_stage_paths.get(filename) or fallback
        if path:
            generated.append({"title": title, "path": path, "caption": caption})

    shap_items = _shap_report_figures(results_dir, data, output_dir, focus_model)
    generated.extend(shap_items)

    ablation_stage = _existing_figure(results_dir, "feature-ablation", "feature_family_ablation_pr_auc.png")
    ablation_fallback = [] if ablation_stage else write_ablation_figures(data["ablation"], output_dir)
    ablation_path = ablation_stage or (ablation_fallback[0]["path"] if ablation_fallback else None)
    if ablation_path:
        generated.append({"title": "Feature-family ablation", "path": ablation_path, "caption": "PR-AUC sensitivity to feature-family changes."})

    tuning_stage = _existing_figure(results_dir, "hyperparameter-tuning-gate", "tuned_vs_fixed_model_comparison.png")
    tuning_fallback = [] if tuning_stage else write_tuning_figures(data["tuning"], output_dir)
    tuning_path = tuning_stage or (tuning_fallback[0]["path"] if tuning_fallback else None)
    if tuning_path:
        generated.append({"title": "Tuned versus fixed model comparison", "path": tuning_path, "caption": "Tuned model families compared with the fixed CatBoost reference."})

    return [item for item in generated if item.get("path")]


def _read_shap_figure_manifest(results_dir: Path) -> pd.DataFrame:
    for relative in ["shap_figure_manifest.csv", f"{STAGE_DIRS['shap']}/shap_figure_manifest.csv"]:
        path = results_dir / relative
        if path.exists():
            frame = optional_read_csv(path)
            if not frame.empty:
                return frame
    return pd.DataFrame()


def _manifest_path(value: object, results_dir: Path) -> str | None:
    if value is None or pd.isna(value):
        return None
    path = Path(str(value))
    if path.exists():
        return path.as_posix()
    candidate = results_dir / str(value)
    if candidate.exists():
        return candidate.as_posix()
    return str(value)


def _shap_report_figures(results_dir: Path, data: dict[str, object], output_dir: Path, focus_model: str | None) -> list[dict[str, str]]:
    generated: list[dict[str, str]] = []
    manifest = _read_shap_figure_manifest(results_dir)
    if not manifest.empty and {"figure_type", "path", "reportable"}.issubset(manifest.columns):
        frame = manifest.copy()
        frame["reportable"] = frame["reportable"].astype(str).str.lower().isin(["true", "1", "yes"])
        frame = frame[frame["reportable"]]
        if focus_model and "model" in frame.columns:
            focused = frame[frame["model"].astype(str).eq(focus_model)]
            if not focused.empty:
                frame = focused
        priority = ["global_importance", "grouped_feature_importance", "beeswarm", "group_contribution"]
        for figure_type in priority:
            matches = frame[frame["figure_type"].astype(str).eq(figure_type)].copy()
            if matches.empty:
                continue
            if figure_type == "group_contribution" and "group_attribute" in matches.columns:
                preferred = ["housing_status", "device_os", "employment_status"]
                ordered = []
                for attr in preferred:
                    subset = matches[matches["group_attribute"].astype(str).eq(attr)]
                    if not subset.empty:
                        ordered.append(subset.iloc[0])
                if not ordered and not matches.empty:
                    ordered.append(matches.iloc[0])
                rows = ordered[:3]
            else:
                rows = [matches.iloc[0]]
            for row in rows:
                path = _manifest_path(row.get("path"), results_dir)
                if not path:
                    continue
                title_map = {
                    "global_importance": "Global SHAP importance",
                    "grouped_feature_importance": "Grouped SHAP importance",
                    "beeswarm": "SHAP beeswarm",
                    "group_contribution": f"SHAP contribution patterns by {row.get('group_attribute', 'group')}",
                }
                caption_map = {
                    "global_importance": "Mean absolute SHAP values for the final model.",
                    "grouped_feature_importance": "Transformed features grouped back to parent feature names when possible.",
                    "beeswarm": "Distribution of SHAP contributions for the final model; shown only when feature names are interpretable.",
                    "group_contribution": "Mean absolute SHAP values by categorical group with denominator context when available.",
                }
                generated.append({"title": title_map.get(figure_type, figure_type), "path": path, "caption": caption_map.get(figure_type, "SHAP diagnostic figure.")})
        if generated:
            return generated

    shap_dir = results_dir / STAGE_DIRS["shap"] / "figures"
    if focus_model:
        model_slug = slug(focus_model)
        for title, pattern, caption in [
            ("Global SHAP importance", f"shap_global_importance_{model_slug}*.png", "Mean absolute SHAP values for the final model."),
            ("SHAP beeswarm", f"shap_beeswarm_{model_slug}*.png", "Distribution of SHAP contributions for the final model."),
        ]:
            matches = sorted(shap_dir.glob(pattern)) if shap_dir.exists() else []
            matches = [path for path in matches if "diagnostic" not in path.name]
            if matches:
                generated.append({"title": title, "path": matches[0].as_posix(), "caption": caption})
    if generated:
        for path in sorted(shap_dir.glob("shap_group_contribution_heatmap*.png"))[:3] if shap_dir.exists() else []:
            generated.append({"title": "SHAP contribution patterns by group", "path": path.as_posix(), "caption": "Mean absolute SHAP values by categorical group."})
        return generated

    shap_summary = data["shap"]
    if isinstance(shap_summary, pd.DataFrame) and not shap_summary.empty and {"model", "feature", "mean_abs_shap"}.issubset(shap_summary.columns):
        plt, _ = setup_matplotlib()
        model = focus_model if focus_model in set(shap_summary["model"].astype(str)) else str(shap_summary.iloc[0]["model"])
        frame = shap_summary[shap_summary["model"].eq(model)].sort_values("mean_abs_shap", ascending=False).head(20)
        if not frame.empty:
            fig, ax = plt.subplots(figsize=(8.8, 6.8))
            ax.barh(frame["feature"].astype(str).map(lambda x: wrap_label(x, width=28, max_chars=58))[::-1], frame["mean_abs_shap"].astype(float)[::-1], color=PALETTE[0])
            ax.set_title("Global SHAP Importance")
            ax.set_xlabel("Mean absolute SHAP")
            path = save_figure(fig, figure_dir(output_dir), "global_shap_importance_final_model.png")
            generated.append({"title": "Global SHAP importance", "path": path, "caption": "Mean absolute SHAP values from available SHAP summary artifacts."})
            plt.close(fig)
    shap_group = data["shap_group"]
    if isinstance(shap_group, pd.DataFrame) and not shap_group.empty:
        generated.extend(write_shap_group_figures(shap_group, output_dir, focus_model=focus_model))
    return generated


def markdown_figure_section(figures: list[dict[str, str]]) -> str:
    if not figures:
        return "No report figures were generated from the available artifacts."
    lines: list[str] = []
    seen: set[str] = set()
    for figure in figures:
        path = figure["path"]
        if path in seen:
            continue
        seen.add(path)
        lines.append(f"### {figure['title']}")
        lines.append("")
        lines.append(f"![{figure['title']}]({path})")
        caption = figure.get("caption")
        if caption:
            lines.append("")
            lines.append(f"_{caption}_")
        lines.append("")
    return "\n".join(lines).strip()
