import numpy as np
import pandas as pd


def columnas_de_texto(data):
    return data.select_dtypes(exclude=np.number).columns.tolist()


def tabla_tasa_fraude(data, columna, target, min_count):
    tabla = (
        data.groupby(columna, dropna=False)[target]
        .agg(["count", "sum", "mean"])
        .rename(columns={"sum": "frauds", "mean": "fraud_rate"})
        .reset_index()
    )
    tabla["share_pct"] = tabla["count"] / len(data) * 100
    tabla["fraud_rate_pct"] = tabla["fraud_rate"] * 100
    tabla = tabla[tabla["count"] >= min_count].copy()
    tabla = tabla.sort_values("fraud_rate", ascending=False)
    return tabla


def spearman_con_rangos(data, columna, target):
    valid = data[[columna, target]].dropna()
    if valid[columna].nunique() <= 1:
        return np.nan

    x_rank = valid[columna].rank(method="average")
    y_rank = valid[target].rank(method="average")
    return x_rank.corr(y_rank)


def matriz_spearman(data, columnas):
    ranked = data[columnas].rank(method="average")
    return ranked.corr()
