# Hollistic Fraud Modeling Analysis

This folder consolidates the earlier EDA/modeling/fairness work and adds two
new experiment families:

- anomaly scores as supervised model features,
- recency-aware training strategies for temporal drift.

The spelling `hollistic` follows the requested folder name.

## Data Split

- Train months: 0-5
- Validation month: 6
- Test month: 7
- Constant columns removed: ['device_fraud_count']

Monthly fraud-rate drift:

| month | count  | fraud_rate |
| ----- | ------ | ---------- |
| 0     | 132440 | 0.011326   |
| 1     | 127620 | 0.009387   |
| 2     | 136979 | 0.008746   |
| 3     | 150936 | 0.009222   |
| 4     | 127691 | 0.011371   |
| 5     | 119323 | 0.011825   |
| 6     | 108168 | 0.013405   |
| 7     | 96843  | 0.014746   |

## New Anomaly Score Features

The anomaly features were fitted only on training data, using legitimate
training rows as the reference population:

- `isolation_forest_anomaly_score`
- `lof_anomaly_score`
- `autoencoder_reconstruction_error`

These scores do not replace the supervised fraud model. They are appended as
extra "rarity" signals and then evaluated inside XGBoost, LightGBM, and CatBoost.

## Recency Strategies

- `full_0_5`: train on months 0-5.
- `full_0_5_recency_weighted`: train on months 0-5, giving higher sample weight to later months.
- `recent_3_5`: train only on months 3-5.

## Main Results

Main threshold policy: choose one global threshold on validation with FPR <= 5%,
then apply that threshold to test.

Best validation model:

- `CatBoost | recent_3_5 | with_anomaly_scores`
- Validation PR-AUC: 0.172083
- Validation ROC-AUC: 0.885295
- Validation recall: 0.518621
- Validation FPR: 0.049748

Same selected model on test:

- Test PR-AUC: 0.193662
- Test ROC-AUC: 0.885047
- Test precision: 0.162379
- Test recall: 0.481793
- Test FPR: 0.037195

Best test PR-AUC model:

- `CatBoost | full_0_5_recency_weighted | with_anomaly_scores`
- Test PR-AUC: 0.206803
- Test ROC-AUC: 0.889390
- Test precision: 0.158278
- Test recall: 0.502101
- Test FPR: 0.039962

## Anomaly Feature Delta

Positive deltas mean adding anomaly scores improved that metric.

| model_family | train_strategy            | pr_auc_delta_anomaly_minus_no_anomaly | roc_auc_delta_anomaly_minus_no_anomaly | recall_tpr_delta_anomaly_minus_no_anomaly | fpr_delta_anomaly_minus_no_anomaly |
| ------------ | ------------------------- | ------------------------------------- | -------------------------------------- | ----------------------------------------- | ---------------------------------- |
| CatBoost     | full_0_5                  | -0.000506                             | 0.003046                               | 0.016106                                  | 0.003427                           |
| CatBoost     | full_0_5_recency_weighted | 0.002663                              | 0.003030                               | 0.023109                                  | 0.002295                           |
| CatBoost     | recent_3_5                | -0.009924                             | -0.003397                              | -0.006303                                 | -0.001750                          |
| LightGBM     | full_0_5                  | -0.007131                             | -0.005657                              | -0.007703                                 | 0.001939                           |
| LightGBM     | full_0_5_recency_weighted | -0.002695                             | -0.003111                              | 0.006303                                  | 0.001949                           |
| LightGBM     | recent_3_5                | -0.004996                             | -0.002086                              | 0.005602                                  | 0.007200                           |
| XGBoost      | full_0_5                  | 0.001922                              | 0.001352                               | -0.002101                                 | 0.000210                           |
| XGBoost      | full_0_5_recency_weighted | -0.004071                             | -0.000616                              | 0.003501                                  | 0.001394                           |
| XGBoost      | recent_3_5                | 0.000534                              | 0.000337                               | 0.042017                                  | 0.009024                           |

## Recency Summary

Best metric achieved by each train strategy across model/anomaly settings:

| train_strategy            | pr_auc   | roc_auc  | recall_tpr | fpr      |
| ------------------------- | -------- | -------- | ---------- | -------- |
| full_0_5                  | 0.202698 | 0.890200 | 0.508403   | 0.041618 |
| full_0_5_recency_weighted | 0.206803 | 0.889390 | 0.504202   | 0.041775 |
| recent_3_5                | 0.203586 | 0.890584 | 0.534314   | 0.049384 |

## Fairness Context

The prior housing-status audit is copied into this folder as reference plots,
CSV tables, and markdown report.
Earlier results showed that `housing_status` added predictive lift but increased
group-level FPR for `housing_status = BA`. That means a deployment recommendation
should not be made from PR-AUC alone.

## Calibration

The best validation model is also calibrated with Platt scaling on validation
month 6 and evaluated on test month 7. This is a probability calibration check,
not a new ranking model. See:

- `calibration_comparison.csv`
- `month_level_calibration.csv`
- `figures/best_model_test_calibration_curve.png`

## Key Artifacts

- `hollistic_validation_metrics.csv`
- `hollistic_test_metrics.csv`
- `hollistic_test_main_threshold.csv`
- `anomaly_feature_deltas.csv`
- `recency_strategy_summary.csv`
- `prior_results_combined.csv`
- `figures/hollistic_test_roc_top8.png`
- `figures/hollistic_test_precision_recall_top8.png`
- `figures/hollistic_confusion_matrices_test_top8.png`
- `figures/shap_beeswarm_best_tree_pipeline.png`
- `figures/anomaly_score_distributions.png`
