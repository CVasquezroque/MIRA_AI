import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def guardar(ruta):
    plt.tight_layout()
    plt.savefig(ruta, dpi=140, bbox_inches="tight")
    plt.close()


def graficar_balance(target_counts, target_percent, ruta):
    plt.figure(figsize=(5.5, 4))
    ax = sns.barplot(x=target_counts.index.astype(str), y=target_counts.values, color="#4c78a8")
    ax.set_title("Balance de fraud_bool")
    ax.set_xlabel("fraud_bool")
    ax.set_ylabel("Filas")

    for index, value in enumerate(target_counts.values):
        ax.text(index, value, f"{target_percent.iloc[index]:.2f}%", ha="center", va="bottom", fontsize=9)

    guardar(ruta)


def graficar_tasas_categoricas(categorical_plots, fraud_rate, ruta):
    fig, axes = plt.subplots(len(categorical_plots), 1, figsize=(8.5, 4 * len(categorical_plots)))
    if len(categorical_plots) == 1:
        axes = [axes]

    for axis, item in zip(axes, categorical_plots):
        column, table = item
        plot_data = table.copy()
        plot_data[column] = plot_data[column].astype(str)

        sns.barplot(data=plot_data, y=column, x="fraud_rate_pct", ax=axis, color="#72b7b2")
        axis.axvline(fraud_rate * 100, color="#e45756", linestyle="--", linewidth=1)
        axis.set_title(f"Tasa de fraude por {column}")
        axis.set_xlabel("Tasa de fraude (%)")
        axis.set_ylabel("")

    guardar(ruta)


def graficar_spearman(spearman, ruta):
    plot_data = spearman.head(12).sort_values("spearman_corr")
    plt.figure(figsize=(8.5, 5.5))
    colors = np.where(plot_data["spearman_corr"] >= 0, "#4c78a8", "#f58518")

    plt.barh(plot_data["column"], plot_data["spearman_corr"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Correlacion Spearman con fraud_bool")
    plt.xlabel("Correlacion Spearman")
    plt.ylabel("")

    guardar(ruta)


def graficar_bins_numericos(data, top_numeric, target, fraud_rate, ruta):
    fig, axes = plt.subplots(len(top_numeric), 1, figsize=(8.5, 4 * len(top_numeric)))
    if len(top_numeric) == 1:
        axes = [axes]

    for axis, column in zip(axes, top_numeric):
        valid = data[[column, target]].dropna().copy()

        if valid[column].nunique() < 5:
            binned = valid.groupby(column)[target].agg(["count", "mean"]).reset_index()
            binned["bin"] = binned[column].astype(str)
        else:
            valid["bin"] = pd.qcut(valid[column], q=10, duplicates="drop")
            binned = valid.groupby("bin", observed=True)[target].agg(["count", "mean"]).reset_index()
            binned["bin"] = binned["bin"].astype(str)

        binned["fraud_rate_pct"] = binned["mean"] * 100
        sns.lineplot(data=binned, x="bin", y="fraud_rate_pct", marker="o", ax=axis, color="#54a24b")
        axis.axhline(fraud_rate * 100, color="#e45756", linestyle="--", linewidth=1)
        axis.set_title(f"Tasa de fraude por intervalos de {column}")
        axis.set_xlabel("")
        axis.set_ylabel("Tasa de fraude (%)")
        axis.tick_params(axis="x", rotation=35)

    guardar(ruta)


def graficar_boxplots_numericos(data, columnas, target, ruta):
    fig, axes = plt.subplots(len(columnas), 1, figsize=(8.5, 3.4 * len(columnas)))
    if len(columnas) == 1:
        axes = [axes]

    for axis, column in zip(axes, columnas):
        valid = data[[column, target]].dropna().copy()
        lower = valid[column].quantile(0.01)
        upper = valid[column].quantile(0.99)
        filtered = valid[(valid[column] >= lower) & (valid[column] <= upper)]

        sns.boxplot(data=filtered, x=target, y=column, ax=axis, color="#bab0ac", showfliers=False)
        axis.set_title(f"{column}: comparacion central entre fraude y no fraude")
        axis.set_xlabel("fraud_bool")
        axis.set_ylabel(column)

    guardar(ruta)


def graficar_matriz_correlacion(correlation_matrix, ruta):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        cbar_kws={"label": "Spearman"},
    )
    plt.title("Matriz de correlacion Spearman entre variables numericas")
    guardar(ruta)


def graficar_tendencia_mensual(month_summary, ruta):
    fig, axis_1 = plt.subplots(figsize=(8.5, 4.5))

    sns.lineplot(data=month_summary, x="month", y="fraud_rate_pct", marker="o", ax=axis_1, color="#e45756")
    axis_1.set_ylabel("Tasa de fraude (%)")
    axis_1.set_xlabel("month")
    axis_1.set_title("Volumen y tasa de fraude por mes")

    axis_2 = axis_1.twinx()
    sns.barplot(data=month_summary, x="month", y="count", ax=axis_2, color="#4c78a8", alpha=0.25)
    axis_2.set_ylabel("Filas")

    guardar(ruta)
