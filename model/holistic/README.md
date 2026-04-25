# Holistic Fraud Modeling

Nueva suite progresiva para deteccion de fraude de apertura de cuentas.

La carpeta anterior `hollistic_old` queda como referencia historica. Esta version
usa la ortografia `holistic` y consolida en un flujo incremental:

1. auditoria de datos, columnas constantes, sentinels y split temporal;
2. baselines con busqueda aleatoria de modelos y preprocessing;
3. expansion de balanceo solo sobre candidatos fuertes;
4. feature engineering avanzado con ratios, interacciones y target/frequency encoding;
5. anomaly scores y estrategias temporales de recency;
6. SHAP explainability para los 3 mejores modelos interpretables;
7. fairness audit de `housing_status` para los 10 mejores candidatos.

## Principio De Diseno

No se prueba todo contra todo. El pipeline avanza por compuertas:

- primero se corre una comparacion amplia pero moderada de baselines;
- luego se toma el top N y se le agregan variantes de balanceo;
- despues se toma el nuevo top N y se prueban features avanzadas;
- finalmente se aplican anomaly scores, recency weighting, SHAP y fairness solo a
  los mejores candidatos.

Esto evita un grid explosivo, mantiene la comparacion trazable y permite explicar
por que cada etapa existe.

## Ejecucion

Usar el entorno del proyecto:

```bash
/home/legion_carlos/miniconda3/envs/DL-env/bin/python model/holistic/holistic_fraud_analysis.py --mode smoke
```

Para una corrida mas seria:

```bash
/home/legion_carlos/miniconda3/envs/DL-env/bin/python model/holistic/holistic_fraud_analysis.py --mode full
```

Los artefactos se guardan en `model/results/holistic`.

## Convenciones

- Train: meses `0-5`.
- Validacion: mes `6`.
- Test: mes `7`.
- Threshold principal: umbral global elegido en validacion con FPR <= 5%, aplicado
  sin reajuste al test.
- Metrica de ranking principal: PR-AUC.
- Accuracy no se usa como metrica principal por el desbalance.

