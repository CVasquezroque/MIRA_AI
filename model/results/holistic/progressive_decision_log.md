# Progressive Holistic Run Log

This file is written as the pipeline advances. It records decisions, promotion gates, and key outputs.

## Run Config

```json
{
  "mode": "smoke",
  "tuning_train_rows": 2500,
  "tuning_valid_rows": 1000,
  "train_rows": 6000,
  "eval_rows": 2500,
  "baseline_n_iter": 1,
  "catboost_n_iter": 1,
  "top_n_baseline_to_balance": 2,
  "top_n_to_advanced": 2,
  "top_n_to_anomaly": 1,
  "fairness_top_n": 1,
  "shap_top_n": 1,
  "shap_sample_rows": 100,
  "anomaly_legit_rows": 1000,
  "lof_legit_rows": 800,
  "autoencoder_legit_rows": 800,
  "include_expensive_ensembles": false
}
```

## 00 Data Audit

- Data rows: 1,000,000.
- Train rows: 794,989; validation rows: 108,168; test rows: 96,843.
- Run train sample rows: 6,000.
- Validation eval rows: 2,500; test eval rows: 2,500.
- Train months: [0, 1, 2, 3, 4, 5]; validation month: 6; test month: 7.
- Removed constant columns: ['device_fraud_count'].
- scale_pos_weight on run train sample: 95.7742.

## 01 Baseline Randomized Search - Start

Comparing Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM with EDA-driven preprocessing options inside the pipeline. The search score is PR-AUC on an inner chronological validation fold.

## 01 Baseline Randomized Search - Result

Top baseline candidates by validation PR-AUC:

| model                                       | model_family        | validation_pr_auc | validation_roc_auc | test_pr_auc | test_roc_auc |
| ------------------------------------------- | ------------------- | ----------------- | ------------------ | ----------- | ------------ |
| baseline_randomsearch / Logistic Regression | Logistic Regression | 0.07529           | 0.765517           | 0.138065    | 0.8426       |
| baseline_randomsearch / Random Forest       | Random Forest       | 0.059072          | 0.770455           | 0.100314    | 0.81599      |
| baseline_randomsearch / XGBoost             | XGBoost             | 0.053541          | 0.70593            | 0.056061    | 0.767993     |
| baseline_catboost / CatBoost native         | CatBoost            | 0.052615          | 0.756572           | 0.115162    | 0.814882     |
| baseline_randomsearch / LightGBM            | LightGBM            | 0.04361           | 0.697736           | 0.056647    | 0.79558      |
| baseline_randomsearch / Decision Tree       | Decision Tree       | 0.020377          | 0.63094            | 0.029352    | 0.655304     |

## 02 Balancing Gate

Promoted baseline families: Logistic Regression, Random Forest.

Skipped notes:


Top balancing candidates:

| model                                                     | model_family        | balance_policy       | validation_pr_auc | test_pr_auc |
| --------------------------------------------------------- | ------------------- | -------------------- | ----------------- | ----------- |
| balance_gate / Logistic Regression / random_oversampling  | Logistic Regression | random_oversampling  | 0.091877          | 0.128568    |
| balance_gate / Logistic Regression / smote                | Logistic Regression | smote                | 0.089416          | 0.118959    |
| balance_gate / Logistic Regression / class_weight         | Logistic Regression | class_weight         | 0.082665          | 0.121022    |
| balance_gate / Random Forest / smote                      | Random Forest       | smote                | 0.073151          | 0.099645    |
| balance_gate / Random Forest / no_balance                 | Random Forest       | no_balance           | 0.070252          | 0.084566    |
| balance_gate / Logistic Regression / no_balance           | Logistic Regression | no_balance           | 0.065403          | 0.111902    |
| balance_gate / Random Forest / class_weight               | Random Forest       | class_weight         | 0.063634          | 0.090304    |
| balance_gate / Random Forest / random_oversampling        | Random Forest       | random_oversampling  | 0.057154          | 0.082984    |
| balance_gate / Logistic Regression / random_undersampling | Logistic Regression | random_undersampling | 0.048838          | 0.069721    |
| balance_gate / Random Forest / random_undersampling       | Random Forest       | random_undersampling | 0.048462          | 0.040803    |

## 03 Advanced Feature Gate

Promoted families: Logistic Regression, Random Forest. Features include ratio features, interaction features, and temporal target/frequency encoding.

Skipped notes:


Top advanced candidates:

| model                               | model_family        | validation_pr_auc | test_pr_auc |
| ----------------------------------- | ------------------- | ----------------- | ----------- |
| advanced_gate / Random Forest       | Random Forest       | 0.060932          | 0.105635    |
| advanced_gate / Logistic Regression | Logistic Regression | 0.059569          | 0.102558    |

## 04 Anomaly And Recency Gate

Promoted families: Logistic Regression. Strategies: full_0_5, full_0_5_recency_weighted, recent_3_5. Anomaly scores are fitted only on legitimate training rows.

| model                                                                                           | model_family        | train_strategy            | anomaly_policy         | validation_pr_auc | test_pr_auc |
| ----------------------------------------------------------------------------------------------- | ------------------- | ------------------------- | ---------------------- | ----------------- | ----------- |
| anomaly_recency_gate / Logistic Regression / recent_3_5 / with_anomaly_scores                   | Logistic Regression | recent_3_5                | with_anomaly_scores    | 0.082724          | 0.087193    |
| anomaly_recency_gate / Logistic Regression / recent_3_5 / without_anomaly_scores                | Logistic Regression | recent_3_5                | without_anomaly_scores | 0.079572          | 0.085951    |
| anomaly_recency_gate / Logistic Regression / full_0_5_recency_weighted / without_anomaly_scores | Logistic Regression | full_0_5_recency_weighted | without_anomaly_scores | 0.061006          | 0.099953    |
| anomaly_recency_gate / Logistic Regression / full_0_5_recency_weighted / with_anomaly_scores    | Logistic Regression | full_0_5_recency_weighted | with_anomaly_scores    | 0.060629          | 0.108114    |
| anomaly_recency_gate / Logistic Regression / full_0_5 / without_anomaly_scores                  | Logistic Regression | full_0_5                  | without_anomaly_scores | 0.059569          | 0.102558    |
| anomaly_recency_gate / Logistic Regression / full_0_5 / with_anomaly_scores                     | Logistic Regression | full_0_5                  | with_anomaly_scores    | 0.059038          | 0.108135    |

## 05 SHAP Explainability

Attempted SHAP on top 1 validation candidates. Successful model-feature rows: 15. Failures: 0.

## 06 Housing Status Fairness

Audited 1 top candidates with and without `housing_status`. Saved overall, group, delta, and markdown report artifacts.

## 07 Calibration

Platt scaling check saved for best validation candidate: `balance_gate | Logistic Regression | random_oversampling`.

## 08 Final Report

Final report written to `holistic_report.md`.

