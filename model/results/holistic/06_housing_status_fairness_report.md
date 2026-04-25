# Housing Status Fairness Audit - Progressive Holistic Run

Top candidates audited: 1

Threshold policy: choose one global threshold on validation with FPR <= 5%, then
apply that threshold to test.

## Overall Metrics

| source_model                                             | model_family        | feature_policy         | precision | recall_tpr | fpr      | pr_auc   | roc_auc  |
| -------------------------------------------------------- | ------------------- | ---------------------- | --------- | ---------- | -------- | -------- | -------- |
| balance_gate / Logistic Regression / random_oversampling | Logistic Regression | with_housing_status    | 0.170213  | 0.216216   | 0.015834 | 0.128568 | 0.833394 |
| balance_gate / Logistic Regression / random_oversampling | Logistic Regression | without_housing_status | 0.118644  | 0.189189   | 0.021112 | 0.106963 | 0.82456  |

## Overall Delta With Minus Without Housing Status

| source_model                                             | model_family        | precision_delta_with_minus_without | recall_tpr_delta_with_minus_without | fpr_delta_with_minus_without | pr_auc_delta_with_minus_without | roc_auc_delta_with_minus_without |
| -------------------------------------------------------- | ------------------- | ---------------------------------- | ----------------------------------- | ---------------------------- | ------------------------------- | -------------------------------- |
| balance_gate / Logistic Regression / random_oversampling | Logistic Regression | 0.051569                           | 0.027027                            | -0.005278                    | 0.021604                        | 0.008833                         |

## BA Group Rows

| source_model                                             | model_family        | feature_policy         | n   | fraud_count | alert_rate | precision | recall_tpr | fpr      | fnr  |
| -------------------------------------------------------- | ------------------- | ---------------------- | --- | ----------- | ---------- | --------- | ---------- | -------- | ---- |
| balance_gate / Logistic Regression / random_oversampling | Logistic Regression | with_housing_status    | 488 | 20          | 0.061475   | 0.2       | 0.3        | 0.051282 | 0.7  |
| balance_gate / Logistic Regression / random_oversampling | Logistic Regression | without_housing_status | 488 | 20          | 0.04918    | 0.208333  | 0.25       | 0.040598 | 0.75 |

## Interpretation

This is diagnostic, not deployment approval. Compare predictive lift with
group-level FPR/FNR shifts before recommending use of `housing_status`.
