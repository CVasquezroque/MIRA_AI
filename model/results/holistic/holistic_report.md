# Holistic Fraud Modeling Report

This run uses a progressive gate design rather than a full cartesian product.

## Best Validation Candidate

- Model: `balance_gate | Logistic Regression | random_oversampling`
- Validation PR-AUC: 0.091877
- Validation ROC-AUC: 0.782048
- Test PR-AUC: 0.128568
- Test ROC-AUC: 0.833394

## Best Test PR-AUC Candidate

- Model: `baseline_randomsearch | Logistic Regression`
- Validation PR-AUC: 0.075290
- Test PR-AUC: 0.138065
- Test ROC-AUC: 0.842600

## Top 15 Candidates

| model                                                                                           | stage                 | model_family        | feature_set                          | balance_policy          | train_strategy            | anomaly_policy         | validation_pr_auc | test_pr_auc | test_roc_auc |
| ----------------------------------------------------------------------------------------------- | --------------------- | ------------------- | ------------------------------------ | ----------------------- | ------------------------- | ---------------------- | ----------------- | ----------- | ------------ |
| balance_gate / Logistic Regression / random_oversampling                                        | balance_gate          | Logistic Regression | baseline_eda_onehot_dense            | random_oversampling     | full_0_5_sample           | without_anomaly_scores | 0.091877          | 0.128568    | 0.833394     |
| balance_gate / Logistic Regression / smote                                                      | balance_gate          | Logistic Regression | baseline_eda_onehot_dense            | smote                   | full_0_5_sample           | without_anomaly_scores | 0.089416          | 0.118959    | 0.828335     |
| anomaly_recency_gate / Logistic Regression / recent_3_5 / with_anomaly_scores                   | anomaly_recency_gate  | Logistic Regression | advanced_plus_optional_anomaly       | model_default_weighting | recent_3_5                | with_anomaly_scores    | 0.082724          | 0.087193    | 0.781414     |
| balance_gate / Logistic Regression / class_weight                                               | balance_gate          | Logistic Regression | baseline_eda_onehot_dense            | class_weight            | full_0_5_sample           | without_anomaly_scores | 0.082665          | 0.121022    | 0.833054     |
| anomaly_recency_gate / Logistic Regression / recent_3_5 / without_anomaly_scores                | anomaly_recency_gate  | Logistic Regression | advanced_plus_optional_anomaly       | model_default_weighting | recent_3_5                | without_anomaly_scores | 0.079572          | 0.085951    | 0.779647     |
| baseline_randomsearch / Logistic Regression                                                     | baseline_randomsearch | Logistic Regression | baseline_eda_onehot                  | model_default_weighting | full_0_5_sample           | without_anomaly_scores | 0.07529           | 0.138065    | 0.8426       |
| balance_gate / Random Forest / smote                                                            | balance_gate          | Random Forest       | baseline_eda_onehot_dense            | smote                   | full_0_5_sample           | without_anomaly_scores | 0.073151          | 0.099645    | 0.744917     |
| balance_gate / Random Forest / no_balance                                                       | balance_gate          | Random Forest       | baseline_eda_onehot_dense            | no_balance              | full_0_5_sample           | without_anomaly_scores | 0.070252          | 0.084566    | 0.773765     |
| balance_gate / Logistic Regression / no_balance                                                 | balance_gate          | Logistic Regression | baseline_eda_onehot_dense            | no_balance              | full_0_5_sample           | without_anomaly_scores | 0.065403          | 0.111902    | 0.807705     |
| balance_gate / Random Forest / class_weight                                                     | balance_gate          | Random Forest       | baseline_eda_onehot_dense            | class_weight            | full_0_5_sample           | without_anomaly_scores | 0.063634          | 0.090304    | 0.787668     |
| anomaly_recency_gate / Logistic Regression / full_0_5_recency_weighted / without_anomaly_scores | anomaly_recency_gate  | Logistic Regression | advanced_plus_optional_anomaly       | model_default_weighting | full_0_5_recency_weighted | without_anomaly_scores | 0.061006          | 0.099953    | 0.826217     |
| advanced_gate / Random Forest                                                                   | advanced_gate         | Random Forest       | ratios_interactions_target_frequency | model_default_weighting | full_0_5_sample           | without_anomaly_scores | 0.060932          | 0.105635    | 0.778956     |
| anomaly_recency_gate / Logistic Regression / full_0_5_recency_weighted / with_anomaly_scores    | anomaly_recency_gate  | Logistic Regression | advanced_plus_optional_anomaly       | model_default_weighting | full_0_5_recency_weighted | with_anomaly_scores    | 0.060629          | 0.108114    | 0.833558     |
| anomaly_recency_gate / Logistic Regression / full_0_5 / without_anomaly_scores                  | anomaly_recency_gate  | Logistic Regression | advanced_plus_optional_anomaly       | model_default_weighting | full_0_5                  | without_anomaly_scores | 0.059569          | 0.102558    | 0.827797     |
| advanced_gate / Logistic Regression                                                             | advanced_gate         | Logistic Regression | ratios_interactions_target_frequency | model_default_weighting | full_0_5_sample           | without_anomaly_scores | 0.059569          | 0.102558    | 0.827797     |

## Main Artifacts

- `progressive_decision_log.md`
- `holistic_all_metrics.csv`
- `holistic_candidate_ranking.csv`
- `01_baseline_candidates.csv`
- `02_balancing_candidates.csv`
- `03_advanced_candidates.csv`
- `04_anomaly_recency_candidates.csv`
- `05_shap_top_features_summary.csv`
- `06_housing_status_fairness_report.md`
- `07_calibration_comparison.csv`
