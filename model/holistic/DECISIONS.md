# Holistic Fraud Modeling Decision Log

Este archivo documenta decisiones de diseno antes de ejecutar el nuevo pipeline.
El script tambien escribe reportes por etapa en `model/results/holistic`.

## 2026-04-25 - Reset Controlado

- Se mantiene `model/hollistic_old` como referencia del script anterior.
- Se mantiene `model/results/hollistic` como referencia de resultados previos.
- Se crea `model/holistic` para la nueva implementacion.
- Se inicializa Git para versionar etapas de trabajo con commits separados.

## 2026-04-25 - Estrategia Progresiva

El objetivo del usuario es incluir RandomizedSearch, modelos clasicos, boosting,
ensembles, balanceo, features avanzadas, anomaly scores, SHAP y fairness. Hacer
un producto cartesiano completo seria caro y dificil de leer, asi que se adopta
una estrategia por compuertas:

1. **Baseline gate**: comparar familias principales con preprocessing EDA-driven.
2. **Balancing gate**: aplicar estrategias de balanceo al top de baselines.
3. **Advanced feature gate**: aplicar ratios, interacciones y target/frequency
   encoding al top posterior a balanceo.
4. **Temporal/anomaly gate**: probar anomaly scores y recency strategies sobre
   candidatos ya competitivos.
5. **Diagnostics gate**: SHAP top 3 y fairness `housing_status` top 10.

## Variables Sensibles

`housing_status` se trata como variable sensible/proxy. No se elimina
automaticamente porque tiene lift predictivo, pero se audita con y sin la columna
y por grupo antes de cualquier recomendacion de despliegue.

## Balanceo

Se comparan:

- `no_balance`;
- `class_weight` o `scale_pos_weight`;
- `random_undersampling`;
- `random_oversampling`;
- `smote`.

Los samplers se aplican dentro de `imblearn.Pipeline` para evitar leakage. Para
CatBoost nativo, los samplers que requieren matriz numerica one-hot pueden ser
omitidos o corridos como variante compatible si el pipeline la define.

## Anomaly Scores

Los anomaly scores se entrenan solo con filas legitimas del training sample:

- Isolation Forest;
- Local Outlier Factor con `novelty=True`;
- error de reconstruccion tipo autoencoder con `MLPRegressor`.

Se tratan como features auxiliares para modelos supervisados, no como reemplazo
del modelo principal.

