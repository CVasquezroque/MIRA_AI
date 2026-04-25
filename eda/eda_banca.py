import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns

from utils.eda_utils import columnas_de_texto, matriz_spearman, spearman_con_rangos, tabla_tasa_fraude
from utils.graficos import (
    graficar_balance,
    graficar_bins_numericos,
    graficar_boxplots_numericos,
    graficar_matriz_correlacion,
    graficar_spearman,
    graficar_tasas_categoricas,
    graficar_tendencia_mensual,
)


def main():
    base_path = "data_banca/Base.csv"
    dictionary_path = "data_banca/Diccionario.xlsx"
    figures_dir = "eda/figures"
    target = "fraud_bool"

    sns.set_theme(style="whitegrid")

    print("\n" + "=" * 80)
    print("1. Carga inicial")
    print("=" * 80)
    print("Se cargan solo los dos archivos necesarios: Base.csv y Diccionario.xlsx.")

    df = pd.read_csv(base_path)
    dictionary = pd.read_excel(dictionary_path)

    print(f"Base.csv: {df.shape[0]:,} filas y {df.shape[1]:,} columnas.")
    print(f"Diccionario.xlsx: {dictionary.shape[0]:,} filas documentadas.")
    print("\nPrimeras filas:")
    print(df.head(3).to_string(index=False))

    print("\n" + "=" * 80)
    print("2. Comparacion con el diccionario")
    print("=" * 80)

    dictionary_columns = dictionary.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    descriptions = {}
    for _, row in dictionary.iterrows():
        column = str(row.iloc[0]).strip()
        description = str(row.iloc[1]) if len(row) > 1 else ""
        descriptions[column] = description

    data_columns = df.columns.tolist()
    not_documented = []
    for column in data_columns:
        if column not in dictionary_columns:
            not_documented.append(column)

    documented_but_missing = []
    for column in dictionary_columns:
        if column not in data_columns:
            documented_but_missing.append(column)

    if len(not_documented) == 0:
        print("Todas las columnas de Base.csv aparecen en el diccionario.")
    else:
        print("Columnas en Base.csv que no aparecen en el diccionario:")
        print(not_documented)

    if len(documented_but_missing) == 0:
        print("Todas las columnas del diccionario aparecen en Base.csv.")
    else:
        print("Columnas documentadas que no estan en Base.csv:")
        print(documented_but_missing)

    if len(not_documented) == 0 and len(documented_but_missing) == 0:
        print("Decision: el diccionario se puede usar como guia para revisar calidad de datos.")
    else:
        print("Decision: continuar, pero no sobreinterpretar columnas sin documentacion completa.")

    print("\n" + "=" * 80)
    print("3. Calidad de datos")
    print("=" * 80)

    duplicate_count = df.duplicated().sum()
    duplicate_pct = duplicate_count / len(df) * 100
    print(f"Filas duplicadas exactas: {duplicate_count:,} ({duplicate_pct:.4f}%).")

    if duplicate_count == 0:
        print("Decision: no se elimina nada por duplicados.")
        data = df.copy()
    elif duplicate_pct < 0.5:
        print("Decision: los duplicados son pocos; se reportan, pero no se eliminan para este EDA.")
        data = df.copy()
    else:
        print("Decision: hay suficientes duplicados para afectar resumenes; se analiza una copia sin duplicados.")
        data = df.drop_duplicates().copy()

    regular_missing = data.isna().sum()
    regular_missing = regular_missing[regular_missing > 0].sort_values(ascending=False)

    if regular_missing.empty:
        print("\nNo se encontraron valores NaN normales.")
        print("Decision: revisar codigos ocultos de ausencia, como -1 o textos tipo 'unknown'.")
    else:
        print("\nValores NaN normales:")
        print((regular_missing / len(data) * 100).round(3).rename("missing_pct").to_string())

    object_columns = columnas_de_texto(data)
    missing_tokens = {"", "na", "n/a", "nan", "none", "null", "missing", "unknown", "?"}
    hidden_text_missing = {}

    for column in object_columns:
        normalized = data[column].astype(str).str.strip().str.lower()
        count = normalized.isin(missing_tokens).sum()
        if count > 0:
            hidden_text_missing[column] = count

    if len(hidden_text_missing) == 0:
        print("No se encontraron codigos textuales obvios de ausencia en variables categoricas.")
    else:
        print("Codigos textuales de ausencia encontrados:")
        for column, count in hidden_text_missing.items():
            print(f"{column}: {count:,}")

    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    minus_one_counts = (data[numeric_columns] == -1).sum()
    minus_one_counts = minus_one_counts[minus_one_counts > 0].sort_values(ascending=False)

    if minus_one_counts.empty:
        print("\nNo se encontraron valores numericos -1.")
    else:
        print("\nColumnas numericas con valores -1:")
        print(minus_one_counts.to_string())

    sentinel_candidates = [
        "prev_address_months_count",
        "current_address_months_count",
        "bank_months_count",
        "session_length_in_minutes",
        "device_distinct_emails_8w",
    ]

    hidden_minus_one_columns = []
    missing_flags = []
    sentinel_rows = []

    for column in minus_one_counts.index:
        if column not in sentinel_candidates:
            continue

        description = descriptions.get(column, "").replace("−", "-").lower()
        dictionary_says_missing = "-1" in description and ("faltante" in description or "missing" in description)
        negative_is_not_logical = column in ["session_length_in_minutes", "device_distinct_emails_8w"]

        sentinel_mask = data[column] == -1
        pct_total = sentinel_mask.mean() * 100
        pct_legit = sentinel_mask[data[target] == 0].mean() * 100
        pct_fraud = sentinel_mask[data[target] == 1].mean() * 100
        difference = pct_fraud - pct_legit

        if dictionary_says_missing or negative_is_not_logical:
            decision = "convertir a NaN"
            hidden_minus_one_columns.append(column)
        else:
            decision = "preservar"

        if abs(difference) >= 0.25 and decision == "convertir a NaN":
            flag_name = column + "_was_missing"
            data[flag_name] = sentinel_mask.astype(int)
            missing_flags.append(flag_name)
            decision = decision + " y crear flag"

        sentinel_rows.append(
            {
                "variable": column,
                "sentinel_pct_total": pct_total,
                "sentinel_pct_legit": pct_legit,
                "sentinel_pct_fraud": pct_fraud,
                "fraud_minus_legit_pp": difference,
                "decision": decision,
            }
        )

    if len(sentinel_rows) > 0:
        print("\nRevision especifica de sentinels -1 sospechosos:")
        sentinel_table = pd.DataFrame(sentinel_rows)
        print(
            sentinel_table.round(
                {
                    "sentinel_pct_total": 3,
                    "sentinel_pct_legit": 3,
                    "sentinel_pct_fraud": 3,
                    "fraud_minus_legit_pp": 3,
                }
            ).to_string(index=False)
        )
        print(
            "Decision: no se eliminan filas. Los -1 justificados se convierten a NaN; "
            "si la ausencia cambia entre fraude y no fraude, se conserva tambien un flag."
        )
    else:
        print("No hubo sentinels -1 en las variables sospechosas principales.")

    other_minus_one_columns = []
    for column in minus_one_counts.index:
        if column not in sentinel_candidates:
            other_minus_one_columns.append(column)

    if len(other_minus_one_columns) > 0:
        print("\nColumnas con -1 que se preservan porque el diccionario no lo marca como faltante:")
        print(other_minus_one_columns)

    if len(hidden_minus_one_columns) > 0:
        hidden_minus_one_columns = sorted(set(hidden_minus_one_columns))
        print("\nSe reemplaza -1 por NaN solo en estas columnas justificadas:")
        print(hidden_minus_one_columns)
        for column in hidden_minus_one_columns:
            data[column] = data[column].replace(-1, np.nan)
    else:
        print("Decision: no hay reemplazos de -1 suficientemente justificados.")

    constant_columns = []
    for column in data.columns:
        if data[column].nunique(dropna=True) <= 1:
            constant_columns.append(column)

    if len(constant_columns) > 0:
        print("\nColumnas con variacion nula o casi nula:")
        print(constant_columns)
        print("Decision: se reportan, pero no se usan en graficos de relacion con fraude.")

    binary_columns = [
        target,
        "email_is_free",
        "phone_home_valid",
        "phone_mobile_valid",
        "has_other_cards",
        "foreign_request",
        "keep_alive_session",
        "device_fraud_count",
    ]
    print("\nRevision rapida de columnas binarias documentadas:")
    for column in binary_columns:
        if column in data.columns:
            values = sorted(data[column].dropna().unique().tolist())
            if set(values).issubset({0, 1}):
                print(f"{column}: valores esperados {values}")
            else:
                print(f"{column}: valores inusuales {values}")

    negative_after_missing = []
    for column in numeric_columns:
        if column != target and data[column].dropna().min() < 0:
            negative_after_missing.append(column)

    if len(negative_after_missing) > 0:
        print("\nVariables que aun tienen valores negativos despues de tratar ausencias ocultas:")
        print(negative_after_missing)
        print(
            "Decision: se conservan porque su comportamiento continuo o el diccionario sugieren que no son "
            "automaticamente errores."
        )

    print("\n" + "=" * 80)
    print("4. Variable objetivo: fraud_bool")
    print("=" * 80)

    target_counts = data[target].value_counts(dropna=False).sort_index()
    target_percent = target_counts / len(data) * 100
    print(pd.DataFrame({"count": target_counts, "percent": target_percent.round(3)}).to_string())

    fraud_rate = data[target].mean()
    print(f"\nTasa global de fraude: {fraud_rate:.3%}.")

    if fraud_rate < 0.05:
        print(
            "Decision: la clase fraude es muy minoritaria; conviene priorizar tasas de fraude "
            "y comparaciones relativas, no solo conteos."
        )
    else:
        print("Decision: aunque se revisaran conteos, las tasas siguen siendo utiles para comparar grupos.")

    graficar_balance(target_counts, target_percent, figures_dir + "/01_balance_fraud_bool.png")

    print("\n" + "=" * 80)
    print("5. Resumen de variables")
    print("=" * 80)

    low_cardinality_numeric = []
    for column in numeric_columns:
        if column != target and data[column].nunique(dropna=True) <= 12:
            low_cardinality_numeric.append(column)

    continuous_numeric = []
    for column in numeric_columns:
        if column == target:
            continue
        if column in low_cardinality_numeric:
            continue
        if column in constant_columns:
            continue
        continuous_numeric.append(column)

    print(f"Variables categoricas de texto: {object_columns}")
    print(f"Variables numericas de baja cardinalidad tratadas como categorias: {low_cardinality_numeric}")
    print(f"Variables numericas continuas para distribucion/correlacion: {continuous_numeric}")

    dominated_columns = []
    for column in object_columns + low_cardinality_numeric:
        top_share = data[column].value_counts(dropna=False, normalize=True).iloc[0]
        if top_share >= 0.95:
            dominated_columns.append((column, top_share))

    if len(dominated_columns) > 0:
        print("\nVariables dominadas por una sola categoria:")
        for column, top_share in dominated_columns:
            print(f"{column}: la categoria mas frecuente concentra {top_share:.2%} de filas")
        print("Decision: se resumen, pero se evita gastar graficos si no aportan contraste claro.")

    print("\nResumen numerico de variables continuas:")
    print(data[continuous_numeric].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T.round(3).to_string())

    print("\nResumen de variables categoricas:")
    for column in object_columns:
        unique_count = data[column].nunique(dropna=False)
        print(f"\n{column}: {unique_count} categorias")
        if unique_count <= 15:
            print(data[column].value_counts(dropna=False).to_string())
        else:
            print("Tiene demasiadas categorias; se muestran solo las 10 mas frecuentes.")
            print(data[column].value_counts(dropna=False).head(10).to_string())

    print("\n" + "=" * 80)
    print("6. Relaciones categoricas con fraude")
    print("=" * 80)

    categorical_candidates = []
    for column in object_columns + low_cardinality_numeric:
        if column != target and column not in constant_columns:
            categorical_candidates.append(column)

    min_category_count = max(100, int(len(data) * 0.0005))
    important_categorical = [
        "payment_type",
        "employment_status",
        "housing_status",
        "source",
        "device_os",
    ]
    profiled_categorical = []

    print("Primero se revisan las variables categoricas principales documentadas.")
    for column in important_categorical:
        if column not in data.columns:
            continue

        table = tabla_tasa_fraude(data, column, target, 1)
        counts_by_target = pd.crosstab(data[column], data[target], dropna=False)
        pct_by_target = counts_by_target.div(counts_by_target.sum(axis=0), axis=1) * 100

        for value in [0, 1]:
            if value not in counts_by_target.columns:
                counts_by_target[value] = 0
                pct_by_target[value] = 0

        profile = table[[column, "count", "frauds", "fraud_rate_pct"]].copy()
        profile = profile.merge(
            counts_by_target[[0, 1]].reset_index().rename(columns={0: "count_legit", 1: "count_fraud"}),
            on=column,
            how="left",
        )
        profile = profile.merge(
            pct_by_target[[0, 1]].reset_index().rename(columns={0: "pct_legit", 1: "pct_fraud"}),
            on=column,
            how="left",
        )

        if len(profile) > 10:
            profile = profile.sort_values("count", ascending=False).head(10)
            print(f"\n{column}: muchas categorias; se muestran las 10 mas frecuentes.")
        else:
            print(f"\n{column}: distribucion por clase y tasa de fraude.")

        print(
            profile[
                [
                    column,
                    "count",
                    "count_legit",
                    "pct_legit",
                    "count_fraud",
                    "pct_fraud",
                    "fraud_rate_pct",
                ]
            ]
            .round({"pct_legit": 2, "pct_fraud": 2, "fraud_rate_pct": 3})
            .to_string(index=False)
        )

        small_categories = profile[profile["count"] < min_category_count]
        if len(small_categories) > 0:
            print("Nota: las categorias con muestra pequena se reportan, pero no se sobreinterpretan.")

        profiled_categorical.append(column)

    interesting_categorical = []

    for column in categorical_candidates:
        table = tabla_tasa_fraude(data, column, target, min_category_count)

        if len(table) <= 1:
            print(f"{column}: no tiene suficiente contraste despues de filtrar categorias pequenas.")
            continue

        rate_range = table["fraud_rate"].max() - table["fraud_rate"].min()

        if rate_range >= max(0.002, fraud_rate * 0.35):
            interesting_categorical.append((column, rate_range, table))
            if column not in profiled_categorical:
                print(f"\n{column}: las tasas de fraude varian lo suficiente para revisar.")
                print(
                    table[[column, "count", "share_pct", "frauds", "fraud_rate_pct"]]
                    .head(10)
                    .round({"share_pct": 2, "fraud_rate_pct": 3})
                    .to_string(index=False)
                )
            else:
                print(f"{column}: ya fue perfilada arriba y se conserva como variable categorica relevante.")
        else:
            print(f"{column}: diferencias debiles; no se fuerza interpretacion.")

    interesting_categorical = sorted(interesting_categorical, key=lambda item: item[1], reverse=True)
    categorical_plots = []
    for column, rate_range, table in interesting_categorical:
        if data[column].nunique(dropna=False) <= 15 and len(categorical_plots) < 3:
            categorical_plots.append((column, table))

    if len(categorical_plots) > 0:
        graficar_tasas_categoricas(
            categorical_plots,
            fraud_rate,
            figures_dir + "/02_tasas_fraude_categoricas.png",
        )
    else:
        print("No se generaron graficos categoricos porque no habia contrastes legibles suficientes.")

    print("\n" + "=" * 80)
    print("7. Variables booleanas")
    print("=" * 80)

    boolean_columns = [
        "email_is_free",
        "phone_home_valid",
        "phone_mobile_valid",
        "keep_alive_session",
        "has_other_cards",
        "foreign_request",
    ]
    meaningful_boolean = []

    print("Estas variables se revisan aparte porque no corresponde aplicarles analisis IQR de outliers.")
    for column in boolean_columns:
        if column not in data.columns:
            continue

        values = sorted(data[column].dropna().unique().tolist())
        if not set(values).issubset({0, 1}):
            print(f"{column}: no es estrictamente binaria, se omite de esta seccion.")
            continue

        table = tabla_tasa_fraude(data, column, target, 1)
        counts_by_target = pd.crosstab(data[column], data[target], dropna=False)
        pct_by_target = counts_by_target.div(counts_by_target.sum(axis=0), axis=1) * 100

        for value in [0, 1]:
            if value not in counts_by_target.index:
                counts_by_target.loc[value] = 0
                pct_by_target.loc[value] = 0
            if value not in counts_by_target.columns:
                counts_by_target[value] = 0
                pct_by_target[value] = 0

        profile = table[[column, "count", "frauds", "fraud_rate_pct"]].copy()
        profile = profile.merge(
            counts_by_target[[0, 1]].reset_index().rename(columns={0: "count_legit", 1: "count_fraud"}),
            on=column,
            how="left",
        )
        profile = profile.merge(
            pct_by_target[[0, 1]].reset_index().rename(columns={0: "pct_legit", 1: "pct_fraud"}),
            on=column,
            how="left",
        )

        rate_for_zero = table.loc[table[column] == 0, "fraud_rate_pct"]
        rate_for_one = table.loc[table[column] == 1, "fraud_rate_pct"]
        if len(rate_for_zero) > 0 and len(rate_for_one) > 0:
            rate_difference = float(rate_for_one.iloc[0] - rate_for_zero.iloc[0])
        else:
            rate_difference = 0

        print(f"\n{column}: distribucion por clase y tasa de fraude.")
        print(
            profile[[column, "count", "count_legit", "pct_legit", "count_fraud", "pct_fraud", "fraud_rate_pct"]]
            .sort_values(column)
            .round({"pct_legit": 2, "pct_fraud": 2, "fraud_rate_pct": 3})
            .to_string(index=False)
        )

        if abs(rate_difference) >= 0.25:
            meaningful_boolean.append((column, rate_difference))
            print(f"Decision: diferencia visible en tasa de fraude entre 0 y 1 ({rate_difference:.3f} pp).")
        else:
            print("Decision: diferencia pequena; se reporta sin forzar interpretacion.")

    print("\n" + "=" * 80)
    print("8. Comparacion numerica entre fraude y no fraude")
    print("=" * 80)

    selected_numerical = [
        "proposed_credit_limit",
        "income",
        "name_email_similarity",
        "customer_age",
        "credit_risk_score",
        "session_length_in_minutes",
        "bank_months_count",
        "prev_address_months_count",
        "current_address_months_count",
        "zip_count_4w",
        "velocity_6h",
        "velocity_24h",
        "velocity_4w",
    ]
    numerical_to_review = []
    for column in selected_numerical:
        if column in data.columns and column not in constant_columns:
            numerical_to_review.append(column)

    print(
        "Se comparan percentiles y medianas porque la media sola puede ocultar asimetrias "
        "o colas largas."
    )
    numerical_summary_rows = []
    skewed_numerical = []

    for column in numerical_to_review:
        values_all = data[column].dropna()
        if len(values_all) > 0:
            median_all = values_all.median()
            p99_all = values_all.quantile(0.99)
            if median_all > 0 and p99_all / median_all >= 5:
                skewed_numerical.append(column)

        for class_value in [0, 1]:
            values = data.loc[data[target] == class_value, column].dropna()
            if len(values) == 0:
                continue

            numerical_summary_rows.append(
                {
                    "variable": column,
                    "fraud_bool": class_value,
                    "count": len(values),
                    "mean": values.mean(),
                    "p01": values.quantile(0.01),
                    "q1": values.quantile(0.25),
                    "median": values.median(),
                    "q3": values.quantile(0.75),
                    "p99": values.quantile(0.99),
                }
            )

    numerical_summary = pd.DataFrame(numerical_summary_rows)
    print(numerical_summary.round(3).to_string(index=False))

    if len(skewed_numerical) > 0:
        print("\nVariables con colas largas o asimetria clara segun p99/mediana:")
        print(skewed_numerical)
        print("Decision: para estas variables se prefieren percentiles y graficos filtrados, no graficos saturados.")

    boxplot_columns = []
    for column in ["proposed_credit_limit", "credit_risk_score", "name_email_similarity", "session_length_in_minutes"]:
        if column in numerical_to_review and len(boxplot_columns) < 4:
            boxplot_columns.append(column)

    if len(boxplot_columns) > 0:
        print("\nSe guarda un boxplot filtrado entre p1 y p99 para evitar que colas extremas tapen la comparacion central.")
        graficar_boxplots_numericos(
            data,
            boxplot_columns,
            target,
            figures_dir + "/05_boxplots_numericos_filtrados.png",
        )

    print("\n" + "=" * 80)
    print("9. Outliers exploratorios")
    print("=" * 80)

    outlier_columns = [
        "proposed_credit_limit",
        "session_length_in_minutes",
        "zip_count_4w",
        "velocity_6h",
        "velocity_24h",
        "velocity_4w",
    ]
    outlier_flags = []
    outlier_rows = []

    print(
        "En fraude, los outliers pueden ser senal y no ruido. Por eso se crean flags, "
        "pero no se eliminan observaciones."
    )

    for column in outlier_columns:
        if column not in data.columns:
            continue
        if data[column].nunique(dropna=True) <= 2:
            continue

        values = data[column].dropna()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            print(f"{column}: IQR igual a cero; no se crea flag de outlier.")
            continue

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        flag_name = column + "_is_iqr_outlier"
        outlier_mask = (data[column] < lower_limit) | (data[column] > upper_limit)
        if outlier_mask.sum() == 0:
            print(f"{column}: no se detectan outliers IQR; no se crea flag.")
            continue

        data[flag_name] = outlier_mask.astype(int)
        outlier_flags.append(flag_name)

        table = tabla_tasa_fraude(data, flag_name, target, 1)
        normal_rate = table.loc[table[flag_name] == 0, "fraud_rate_pct"]
        outlier_rate = table.loc[table[flag_name] == 1, "fraud_rate_pct"]
        if len(normal_rate) > 0 and len(outlier_rate) > 0:
            normal_rate_value = float(normal_rate.iloc[0])
            outlier_rate_value = float(outlier_rate.iloc[0])
            relation = "similar"
            if outlier_rate_value > normal_rate_value + 0.25:
                relation = "mayor"
            elif outlier_rate_value < normal_rate_value - 0.25:
                relation = "menor"
        else:
            normal_rate_value = np.nan
            outlier_rate_value = np.nan
            relation = "no comparable"

        p99 = values.quantile(0.99)
        p99_mask = data[column] > p99
        p99_rate = data.loc[p99_mask, target].mean() * 100 if p99_mask.sum() > 0 else np.nan

        outlier_rows.append(
            {
                "variable": column,
                "iqr_outlier_pct": data[flag_name].mean() * 100,
                "fraud_rate_non_outlier": normal_rate_value,
                "fraud_rate_outlier": outlier_rate_value,
                "outlier_vs_non_outlier": relation,
                "p99": p99,
                "fraud_rate_above_p99": p99_rate,
            }
        )

    if len(outlier_rows) > 0:
        outlier_summary = pd.DataFrame(outlier_rows)
        print(outlier_summary.round(3).to_string(index=False))
        print(
            "Decision: preservar valores originales y usar estos flags solo como posibles senales "
            "para modelado posterior."
        )
    else:
        print("No se crearon flags de outlier para las variables revisadas.")

    print("\n" + "=" * 80)
    print("10. Correlaciones y multicolinealidad")
    print("=" * 80)

    spearman_rows = []
    for column in continuous_numeric:
        corr = spearman_con_rangos(data, column, target)
        if not pd.isna(corr):
            spearman_rows.append({"column": column, "spearman_corr": corr, "abs_corr": abs(corr)})

    spearman = pd.DataFrame(spearman_rows)

    if spearman.empty:
        print("No hubo variables continuas adecuadas para correlacion Spearman.")
    else:
        spearman = spearman.sort_values("abs_corr", ascending=False)
        print("Variables numericas con mayor correlacion Spearman absoluta frente a fraud_bool:")
        print(spearman.head(12).round(4).to_string(index=False))
        print(
            "Nota: una correlacion individual baja no vuelve inutil a una variable; en fraude pueden existir "
            "patrones no lineales o combinaciones entre variables."
        )

        strongest_corr = spearman.iloc[0]["abs_corr"]
        if strongest_corr < 0.03:
            print("Decision: las correlaciones son debiles; sirven como pistas, no como relaciones fuertes.")
        else:
            print("Decision: revisar las variables numericas principales con tasas de fraude por bins.")

        graficar_spearman(spearman, figures_dir + "/03_spearman_numericas.png")

        top_numeric = spearman.head(3)["column"].tolist()
        graficar_bins_numericos(
            data,
            top_numeric,
            target,
            fraud_rate,
            figures_dir + "/04_bins_numericos_fraude.png",
        )

    correlation_columns = []
    for column in numerical_to_review:
        if column in data.columns and data[column].nunique(dropna=True) > 1:
            correlation_columns.append(column)

    high_corr_pairs = []
    if len(correlation_columns) >= 2:
        correlation_matrix = matriz_spearman(data, correlation_columns)
        print("\nPares de variables numericas con alta correlacion Spearman absoluta:")
        for i in range(len(correlation_columns)):
            for j in range(i + 1, len(correlation_columns)):
                col_a = correlation_columns[i]
                col_b = correlation_columns[j]
                corr_value = correlation_matrix.loc[col_a, col_b]
                if abs(corr_value) >= 0.65:
                    high_corr_pairs.append((col_a, col_b, corr_value))

        if len(high_corr_pairs) > 0:
            for col_a, col_b, corr_value in high_corr_pairs:
                print(f"{col_a} vs {col_b}: {corr_value:.3f}")
        else:
            print("No se encontraron pares por encima del umbral 0.65.")

        graficar_matriz_correlacion(
            correlation_matrix,
            figures_dir + "/06_matriz_correlacion_numericas.png",
        )

    velocity_columns = []
    for column in ["velocity_6h", "velocity_24h", "velocity_4w"]:
        if column in data.columns:
            velocity_columns.append(column)

    if len(velocity_columns) == 3:
        velocity_corr = matriz_spearman(data, velocity_columns)
        print("\nChequeo especifico de multicolinealidad entre variables velocity:")
        print(velocity_corr.round(3).to_string())

        velocity_high_corr = False
        for i in range(len(velocity_columns)):
            for j in range(i + 1, len(velocity_columns)):
                if abs(velocity_corr.iloc[i, j]) >= 0.70:
                    velocity_high_corr = True

        if velocity_high_corr:
            print(
                "Decision: no se elimina ninguna velocity en EDA. En modelado se podria comparar conservar todas, "
                "retener la mas util o reducir dimensionalidad."
            )
        else:
            print("Decision: no hay evidencia fuerte para tratar estas variables como redundantes en EDA.")

    print("\n" + "=" * 80)
    print("11. Analisis temporal por month")
    print("=" * 80)

    temporal_drift = False
    if "month" in data.columns:
        month_summary = (
            data.groupby("month")[target]
            .agg(["count", "sum", "mean"])
            .rename(columns={"sum": "frauds", "mean": "fraud_rate"})
            .reset_index()
            .sort_values("month")
        )
        month_summary["fraud_rate_pct"] = month_summary["fraud_rate"] * 100
        print("Volumen y tasa de fraude por mes:")
        print(month_summary[["month", "count", "frauds", "fraud_rate_pct"]].round(3).to_string(index=False))

        month_rate_range = month_summary["fraud_rate_pct"].max() - month_summary["fraud_rate_pct"].min()
        if month_rate_range >= 0.30:
            temporal_drift = True
            print(
                f"Decision: la tasa de fraude cambia {month_rate_range:.3f} puntos porcentuales entre meses; "
                "conviene considerar validacion temporal en modelado."
            )
        else:
            print(
                f"Decision: la variacion mensual es {month_rate_range:.3f} puntos porcentuales; "
                "no parece una deriva fuerte bajo este umbral."
            )

        temporal_variables = []
        for column in ["velocity_6h", "velocity_24h", "proposed_credit_limit", "credit_risk_score"]:
            if column in data.columns:
                temporal_variables.append(column)

        if len(temporal_variables) > 0:
            print("\nMedianas mensuales de variables seleccionadas:")
            print(data.groupby("month")[temporal_variables].median().round(3).to_string())

        graficar_tendencia_mensual(month_summary, figures_dir + "/07_tendencia_mensual.png")
    else:
        print("No existe columna month; se omite analisis temporal.")

    print("\n" + "=" * 80)
    print("12. Resumen final exploratorio")
    print("=" * 80)

    print("A. Principales hallazgos exploratorios")
    print(f"- Filas analizadas: {len(data):,}. Columnas originales: {df.shape[1]:,}.")
    print(f"- La tasa de fraude observada es {fraud_rate:.3%}; el target esta fuertemente desbalanceado.")
    print(f"- Duplicados exactos en la base original: {duplicate_count:,}.")

    missing_after = data.isna().sum()
    missing_after = missing_after[missing_after > 0].sort_values(ascending=False)
    if missing_after.empty:
        print("- No quedaron valores faltantes tras la revision.")
    else:
        print("- Valores faltantes despues de reemplazar sentinels justificados:")
        print((missing_after / len(data) * 100).round(3).rename("missing_pct").to_string())

    if len(missing_flags) > 0:
        print(f"- Se crearon flags de ausencia informativa: {missing_flags}.")

    if len(interesting_categorical) > 0:
        print("- Variables categoricas o de baja cardinalidad con diferencias claras en tasa de fraude:")
        for column, rate_range, table in interesting_categorical[:5]:
            row = table.iloc[0]
            print(
                f"  {column}: categoria {row[column]} con {row['fraud_rate_pct']:.3f}% "
                f"de fraude en {int(row['count']):,} filas."
            )

    if len(meaningful_boolean) > 0:
        print("- Variables booleanas con diferencia visible entre 0 y 1:")
        for column, difference in meaningful_boolean:
            print(f"  {column}: diferencia de {difference:.3f} puntos porcentuales.")

    if not spearman.empty:
        print("- Variables numericas con mayor asociacion monotona exploratoria:")
        for _, row in spearman.head(5).iterrows():
            print(f"  {row['column']}: Spearman {row['spearman_corr']:.4f}.")

    if len(outlier_flags) > 0:
        print(f"- Se crearon flags exploratorios de outliers: {outlier_flags}.")

    if temporal_drift:
        print("- La tasa de fraude cambia por mes; hay senal exploratoria de posible deriva temporal.")
    else:
        print("- No se observo una deriva temporal fuerte bajo el umbral usado.")

    print("\nB. Implicaciones para preparacion de datos y modelado futuro")
    if len(constant_columns) > 0:
        print(f"- Remover columnas constantes o sin variacion util, por ejemplo: {constant_columns}.")

    if len(hidden_minus_one_columns) > 0:
        print(f"- Convertir sentinels -1 justificados a NaN: {hidden_minus_one_columns}.")

    if len(missing_flags) > 0:
        print("- Mantener los flags de ausencia porque la ausencia parece asociarse con fraud_bool.")

    if len(outlier_flags) > 0:
        print("- Preservar outliers; pueden ser senales de fraude. Usar flags opcionales en vez de borrar filas.")

    print("- Codificar variables categoricas como payment_type, employment_status, housing_status, source y device_os.")
    print("- Tratar el desbalance de clases con metricas y validacion adecuadas para fraude.")
    print(
        "- Tener cuidado con variables socioeconomicas o sensibles indirectas, como income, housing_status "
        "o customer_age; pueden requerir revision de sesgo antes de uso productivo."
    )

    if len(high_corr_pairs) > 0:
        print("- Revisar multicolinealidad en modelado; no se eliminaron variables automaticamente durante el EDA.")

    if temporal_drift:
        print("- Considerar una particion cronologica o validacion temporal para evaluar modelos futuros.")

    print(
        "\nEstas observaciones son exploratorias. Describen patrones de la base, "
        "pero no prueban causalidad ni deben leerse como explicaciones definitivas del fraude."
    )

    print("\n" + "=" * 80)
    print("13. Propuesta de preparacion de datos y modelado")
    print("=" * 80)

    print(
        "Esta propuesta usa los hallazgos anteriores como guia. No elimina variables solo por "
        "correlacion individual baja, porque el fraude puede aparecer en patrones no lineales o "
        "en combinaciones de variables."
    )

    print("\nA. Definir target, features y columnas removidas")
    print("- Target: fraud_bool.")
    if len(constant_columns) > 0:
        print(f"- Remover del set de features columnas sin variacion util: {constant_columns}.")
    else:
        print("- No se detectaron columnas constantes para remover por ahora.")
    print(
        "- Si device_fraud_count sigue siendo constante, se excluye del modelado. "
        "No se excluyen variables solo porque su correlacion individual con fraud_bool sea baja."
    )
    print(
        "- La columna month se usa principalmente para separar train/valid/test cuando hay deriva temporal; "
        "si se incluye como feature, conviene justificarlo segun el uso futuro del modelo."
    )

    print("\nB. Dividir antes de preprocesar")
    if temporal_drift:
        print(
            "- Como el EDA encontro cambios mensuales en la tasa de fraude, se recomienda una particion cronologica: "
            "meses tempranos para train, el mes siguiente para validacion y el ultimo mes para test."
        )
    elif "month" in data.columns:
        print(
            "- Aunque la deriva mensual no fue fuerte bajo el umbral exploratorio, month existe; "
            "se puede comparar una particion cronologica con una particion estratificada."
        )
    else:
        print(
            "- Si no hay variable temporal usable, usar una particion estratificada para conservar la proporcion "
            "de fraude en train, validacion y test."
        )
    print(
        "- Cualquier imputador, codificador, escalador, limite IQR o seleccion de variables log debe ajustarse "
        "solo con train para evitar fuga de informacion."
    )

    print("\nC. Ausencias ocultas y sentinels")
    if len(hidden_minus_one_columns) > 0:
        print(f"- Convertir -1 a NaN solo en columnas justificadas por diccionario o patron claro: {hidden_minus_one_columns}.")
    else:
        print("- No se justificaron reemplazos generales de -1; revisar el diccionario antes de cambiarlos.")
    print(
        "- No se eliminan filas por tener sentinels. Para ausencias potencialmente informativas se crean flags, "
        "por ejemplo prev_address_months_count_was_missing, bank_months_count_was_missing y "
        "session_length_in_minutes_was_missing."
    )

    print("\nD. Imputacion")
    print(
        "- Numericas: imputar con mediana ajustada en train, porque varias variables tienen colas largas "
        "u outliers exploratorios."
    )
    print(
        "- Categoricas: imputar con una categoria explicita 'Unknown' o con la moda, segun tenga sentido "
        "para la variable. El imputador tambien se ajusta solo con train."
    )

    print("\nE. Outliers y transformaciones")
    if len(outlier_flags) > 0:
        print(f"- Mantener valores originales y crear flags IQR para variables continuas o de conteo: {outlier_flags}.")
    else:
        print("- Si se crean flags IQR, aplicarlos solo a variables continuas o de conteo, no a binarias.")
    print(
        "- No borrar outliers por defecto: en fraude pueden representar comportamiento anomalos util para el modelo."
    )
    if len(skewed_numerical) > 0:
        print(f"- Evaluar log1p solo en variables positivas con asimetria fuerte, candidatas desde el EDA: {skewed_numerical}.")
    else:
        print(
            "- Evaluar log1p solo si train muestra asimetria fuerte o colas largas en variables como "
            "proposed_credit_limit, zip_count_4w, velocity_6h, velocity_24h, velocity_4w o "
            "session_length_in_minutes."
        )
    print(
        "- Usar log1p en vez de log porque permite transformar valores cero sin generar infinitos. "
        "No se aplica a todas las numericas."
    )

    print("\nF. Encoding, escalado y variables sensibles")
    print(
        "- Codificar categoricas como payment_type, employment_status, housing_status, source y device_os con "
        "one-hot encoding y handle_unknown='ignore'."
    )
    print(
        "- Escalar solo para modelos que lo necesitan, especialmente Logistic Regression. "
        "RobustScaler es razonable aqui por la presencia de colas largas."
    )
    print(
        "- Modelos de arbol como Random Forest, XGBoost o LightGBM normalmente no requieren escalado."
    )
    print(
        "- No remover automaticamente housing_status o employment_status, pero comparar modelos con y sin estas "
        "variables y auditar fairness antes de cualquier uso productivo."
    )

    print("\nG. Desbalance, modelos y metricas")
    print(
        "- Manejar el desbalance dentro del entrenamiento: class_weight='balanced' en Logistic Regression, "
        "class_weight en Random Forest y scale_pos_weight en XGBoost o LightGBM."
    )
    print(
        "- SMOTE o undersampling pueden compararse luego, pero nunca antes del split porque eso produciria "
        "data leakage."
    )
    print(
        "- Evaluar al menos Logistic Regression como baseline interpretable, Random Forest como baseline no lineal "
        "y XGBoost o LightGBM como boosting mas fuerte. El modelo mas complejo no se asume mejor."
    )
    print(
        "- Usar Precision, Recall/TPR, FPR, PR-AUC, ROC-AUC y matriz de confusion. "
        "No enfocar accuracy porque el fraude es minoritario."
    )
    print(
        "- Precision indica que tan confiables son las alertas; Recall/TPR indica cuanto fraude real se detecta; "
        "FPR indica cuantos clientes legitimos se marcan incorrectamente."
    )

    print("\nH. Umbrales y flujo de decision")
    print(
        "- No depender del umbral 0.5. Usar validacion para elegir umbrales de score: bajo para aprobacion "
        "automatica, medio para revision y alto para rechazo o verificacion reforzada."
    )
    print(
        "- La seleccion debe balancear reduccion de perdida por fraude, falsos positivos y experiencia del cliente."
    )

    print("\nI. Resumen operativo de la propuesta")
    print("- Removido: target de X, columnas constantes como device_fraud_count si no varia, y month si solo se usa para split.")
    print("- Transformado: sentinels -1 justificados a NaN; log1p solo en numericas positivas con colas largas.")
    print("- Flags: ausencias potencialmente informativas y outliers IQR en variables continuas/de conteo.")
    print("- Codificado: categoricas principales con one-hot encoding y handle_unknown='ignore'.")
    print("- Escalado: solo para Logistic Regression, preferiblemente con RobustScaler; arboles sin escalado obligatorio.")
    print("- Desbalance: pesos de clase o scale_pos_weight dentro del entrenamiento.")
    print("- Evaluacion: metricas de fraude y matriz de confusion con la misma estrategia de validacion.")
    print("- Umbrales: seleccionados en validacion para separar aprobacion, revision y rechazo/verificacion.")
    print("\nSe deja un ejemplo simple de implementacion en model/modeling_proposal.py.")
    print(f"\nGraficos guardados en: {figures_dir}")


if __name__ == "__main__":
    main()
