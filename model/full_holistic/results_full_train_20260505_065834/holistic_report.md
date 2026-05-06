# Holistic Fraud Modeling Report

## 1. Executive Summary

- Strongest ranking candidate: `hyperparameter_tuned | XGBoost` with validation PR-AUC `0.182158` and test PR-AUC `0.215888`.
- Recommended final model: `hyperparameter_tuned | CatBoost`.
- Final recommendation status: `deploy_candidate`.
- Optional sections that were skipped are marked as: This analysis was not run.

## 2. Dataset And Temporal Split

- Train months: `[0, 1, 2, 3, 4, 5]`
- Validation month: `6`
- Test month: `7`
- Fraud prevalence train / validation / test: `0.010253` / `0.013405` / `0.014746`

## 3. Candidate Registry

| model                                                                             | stage                      | model_family | validation_pr_auc | test_pr_auc | test_roc_auc |
| --------------------------------------------------------------------------------- | -------------------------- | ------------ | ----------------- | ----------- | ------------ |
| hyperparameter_tuned / XGBoost                                                    | hyperparameter-tuning-gate | XGBoost      | 0.182158          | 0.215888    | 0.895181     |
| advanced_gate / CatBoost                                                          | advanced_gate              | CatBoost     | 0.182134          | 0.218422    | 0.895868     |
| balance_gate / XGBoost / class_weight                                             | balance_gate               | XGBoost      | 0.181142          | 0.205965    | 0.894311     |
| baseline / XGBoost                                                                | baseline_search            | XGBoost      | 0.181142          | 0.205965    | 0.894311     |
| baseline / CatBoost                                                               | baseline_search            | CatBoost     | 0.180242          | 0.217709    | 0.896679     |
| anomaly_recency_gate / XGBoost / full_0_5 / with_isolation_forest_anomaly_score   | anomaly_recency_gate       | XGBoost      | 0.177784          | 0.212653    | 0.894708     |
| anomaly_recency_gate / XGBoost / full_0_5 / without_anomaly_scores                | anomaly_recency_gate       | XGBoost      | 0.177167          | 0.212117    | 0.894701     |
| advanced_gate / XGBoost                                                           | advanced_gate              | XGBoost      | 0.177167          | 0.212117    | 0.894701     |
| hyperparameter_tuned / CatBoost                                                   | hyperparameter-tuning-gate | CatBoost     | 0.176810          | 0.208517    | 0.890696     |
| anomaly_recency_gate / XGBoost / recent_3_5 / with_isolation_forest_anomaly_score | anomaly_recency_gate       | XGBoost      | 0.176716          | 0.210390    | 0.892355     |
| anomaly_recency_gate / XGBoost / recent_3_5 / without_anomaly_scores              | anomaly_recency_gate       | XGBoost      | 0.176585          | 0.200728    | 0.891611     |
| balance_gate / XGBoost / random_oversampling                                      | balance_gate               | XGBoost      | 0.176480          | 0.194081    | 0.891628     |

## 4. Final Candidate Decision Table

| model                                                                             | stage                      | model_family | feature_set                                    | balance_policy                                       | train_strategy                       | anomaly_policy                      | selected_threshold | validation_pr_auc | validation_roc_auc | validation_pr_auc_lift | test_pr_auc | test_roc_auc | test_pr_auc_lift | runtime_seconds | main_precision | main_fdr | main_recall_tpr | main_fpr | main_alert_rate | business_policy      | business_fdr30_feasible | business_precision | business_fdr | business_recall_tpr | lift_at_k_top1pct | lift_at_k_top5pct | precision_at_k_top1pct | precision_at_k_top5pct | recall_at_k_top1pct | recall_at_k_top5pct | worst_max_fpr_gap | worst_max_tpr_gap | worst_equalized_odds_difference | best_brier_score | interpretability | feature_complexity | deployment_readiness | decision_score | best_by_pr_auc | best_by_recall_at_fpr5 | best_under_fdr30 | best_by_topk_alert_prioritization | best_by_fairness_adjusted_interpretation | recommended_final_model |
| --------------------------------------------------------------------------------- | -------------------------- | ------------ | ---------------------------------------------- | ---------------------------------------------------- | ------------------------------------ | ----------------------------------- | ------------------ | ----------------- | ------------------ | ---------------------- | ----------- | ------------ | ---------------- | --------------- | -------------- | -------- | --------------- | -------- | --------------- | -------------------- | ----------------------- | ------------------ | ------------ | ------------------- | ----------------- | ----------------- | ---------------------- | ---------------------- | ------------------- | ------------------- | ----------------- | ----------------- | ------------------------------- | ---------------- | ---------------- | ------------------ | -------------------- | -------------- | -------------- | ---------------------- | ---------------- | --------------------------------- | ---------------------------------------- | ----------------------- |
| hyperparameter_tuned / XGBoost                                                    | hyperparameter-tuning-gate | XGBoost      | baseline_eda_onehot                            | validation_selected_class_weight_or_scale_pos_weight | full_0_5_after_inner_temporal_tuning | without_anomaly_scores              | 0.885196           | 0.182158          | 0.893281           | 13.588744              | 0.215888    | 0.895181     | 14.640948        | 15.929364       | 0.156250       | 0.843750 | 0.553221        | 0.044710 | 0.052208        | valid_business_fdr30 | True                    | 0.888889           | 0.111111     | 0.005602            | 21.625927         | 10.908449         | 0.318885               | 0.160851               | 0.216387            | 0.545518            | 0.177104          | 1.000000          | 1.000000                        |                  | medium           | 3.000000           | pilot_candidate      | 31.500000      | True           | True                   | False            | False                             | False                                    | False                   |
| advanced_gate / CatBoost                                                          | advanced_gate              | CatBoost     | catboost_native_missing_log_ratio_interactions | model_default_weighting                              | full_0_5                             | without_anomaly_scores              | 0.788399           | 0.182134          | 0.890867           | 13.586931              | 0.218422    | 0.895868     | 14.812782        | 79.970047       | 0.163755       | 0.836245 | 0.525210        | 0.040140 | 0.047293        | valid_business_fdr30 | True                    | 0.916667           | 0.083333     | 0.007703            | 22.115835         | 10.768418         | 0.326109               | 0.158786               | 0.221289            | 0.538515            | 0.163102          | 1.000000          | 1.000000                        |                  | medium           | 3.000000           | pilot_candidate      | 21.500000      | False          | False                  | False            | True                              | False                                    | False                   |
| balance_gate / XGBoost / class_weight                                             | balance_gate               | XGBoost      | baseline_eda_onehot                            | class_weight                                         | train_sample                         | without_anomaly_scores              | 0.796552           | 0.181142          | 0.891288           | 13.512923              | 0.205965    | 0.894311     | 13.967949        |                 | 0.161112       | 0.838888 | 0.531513        | 0.041419 | 0.048646        | valid_business_fdr30 | True                    | 0.888889           | 0.111111     | 0.005602            | 20.856072         | 10.768418         | 0.307534               | 0.158786               | 0.208683            | 0.538515            | 0.177980          | 1.000000          | 1.000000                        |                  | medium           | 2.000000           | pilot_candidate      | 43.500000      | False          | False                  | False            | False                             | False                                    | False                   |
| baseline / XGBoost                                                                | baseline_search            | XGBoost      | baseline_eda_onehot                            | model_default_weighting                              | full_0_5                             | without_anomaly_scores              | 0.796552           | 0.181142          | 0.891288           | 13.512923              | 0.205965    | 0.894311     | 13.967949        | 12.135715       | 0.161112       | 0.838888 | 0.531513        | 0.041419 | 0.048646        | valid_business_fdr30 | True                    | 0.888889           | 0.111111     | 0.005602            | 20.856072         | 10.768418         | 0.307534               | 0.158786               | 0.208683            | 0.538515            | 0.177980          | 1.000000          | 1.000000                        |                  | medium           | 3.000000           | pilot_candidate      | 47.500000      | False          | False                  | False            | False                             | False                                    | False                   |
| baseline / CatBoost                                                               | baseline_search            | CatBoost     | catboost_native_raw_no_generated_features      | model_default_weighting                              | full_0_5                             | without_anomaly_scores              | 0.787401           | 0.180242          | 0.891478           | 13.445823              | 0.217709    | 0.896679     | 14.764428        | 90.137995       | 0.162513       | 0.837487 | 0.534314        | 0.041209 | 0.048481        | valid_business_fdr30 | True                    | 1.000000           | 0.000000     | 0.004202            | 21.695914         | 10.838434         | 0.319917               | 0.159818               | 0.217087            | 0.542017            | 0.167751          | 1.000000          | 1.000000                        |                  | medium           | 3.000000           | pilot_candidate      | 25.500000      | False          | False                  | False            | False                             | False                                    | False                   |
| anomaly_recency_gate / XGBoost / full_0_5 / with_isolation_forest_anomaly_score   | anomaly_recency_gate       | XGBoost      | advanced_plus_optional_anomaly                 | model_default_weighting                              | full_0_5                             | with_isolation_forest_anomaly_score | 0.804124           | 0.177784          | 0.890598           | 13.262425              | 0.212653    | 0.894708     | 14.421521        |                 | 0.158814       | 0.841186 | 0.547619        | 0.043410 | 0.050845        | valid_business_fdr30 | True                    | 1.000000           | 0.000000     | 0.003501            | 21.765900         | 10.922452         | 0.320949               | 0.161057               | 0.217787            | 0.546218            | 0.183066          | 0.714127          | 0.714127                        |                  | medium           | 4.000000           | pilot_candidate      | 44.500000      | False          | False                  | False            | False                             | False                                    | False                   |
| anomaly_recency_gate / XGBoost / full_0_5 / without_anomaly_scores                | anomaly_recency_gate       | XGBoost      | advanced_plus_optional_anomaly                 | model_default_weighting                              | full_0_5                             | without_anomaly_scores              | 0.801017           | 0.177167          | 0.890467           | 13.216430              | 0.212117    | 0.894701     | 14.385157        |                 | 0.161221       | 0.838779 | 0.528711        | 0.041168 | 0.048357        | valid_business_fdr30 | True                    | 0.850000           | 0.150000     | 0.011905            | 20.926059         | 10.768418         | 0.308566               | 0.158786               | 0.209384            | 0.538515            | 0.173330          | 0.689655          | 0.689655                        |                  | medium           | 4.000000           | pilot_candidate      | 47.500000      | False          | False                  | False            | False                             | True                                     | False                   |
| advanced_gate / XGBoost                                                           | advanced_gate              | XGBoost      | ratios_interactions_target_frequency           | model_default_weighting                              | full_0_5                             | without_anomaly_scores              | 0.801017           | 0.177167          | 0.890467           | 13.216430              | 0.212117    | 0.894701     | 14.385157        | 9.807311        | 0.161221       | 0.838779 | 0.528711        | 0.041168 | 0.048357        | valid_business_fdr30 | True                    | 0.850000           | 0.150000     | 0.011905            | 20.926059         | 10.768418         | 0.308566               | 0.158786               | 0.209384            | 0.538515            | 0.173330          | 0.689655          | 0.689655                        |                  | medium           | 3.000000           | pilot_candidate      | 42.500000      | False          | False                  | False            | False                             | False                                    | False                   |
| hyperparameter_tuned / CatBoost                                                   | hyperparameter-tuning-gate | CatBoost     | catboost_native_advanced_features              | validation_selected_class_weight_or_scale_pos_weight | full_0_5_after_inner_temporal_tuning | without_anomaly_scores              | 0.783026           | 0.176810          | 0.884682           | 13.189802              | 0.208517    | 0.890696     | 14.141031        | 66.833668       | 0.159704       | 0.840296 | 0.514006        | 0.040476 | 0.047458        | valid_business_fdr30 | True                    | 1.000000           | 0.000000     | 0.007003            | 21.345980         | 10.516361         | 0.314757               | 0.155069               | 0.213585            | 0.525910            | 0.142756          |                   |                                 |                  | medium           | 3.000000           | deploy_candidate     | 47.500000      | False          | False                  | False            | False                             | False                                    | True                    |
| anomaly_recency_gate / XGBoost / recent_3_5 / with_isolation_forest_anomaly_score | anomaly_recency_gate       | XGBoost      | advanced_plus_optional_anomaly                 | model_default_weighting                              | recent_3_5                           | with_isolation_forest_anomaly_score | 0.806563           | 0.176716          | 0.888269           | 13.182766              | 0.210390    | 0.892355     | 14.268049        |                 | 0.148767       | 0.851233 | 0.549020        | 0.047016 | 0.054418        | valid_business_fdr30 | True                    | 0.840000           | 0.160000     | 0.014706            | 21.206006         | 10.600380         | 0.312693               | 0.156308               | 0.212185            | 0.530112            |                   |                   |                                 |                  | medium           | 4.000000           | deploy_candidate     | 58.500000      | False          | False                  | True             | False                             | False                                    | False                   |
| anomaly_recency_gate / XGBoost / recent_3_5 / without_anomaly_scores              | anomaly_recency_gate       | XGBoost      | advanced_plus_optional_anomaly                 | model_default_weighting                              | recent_3_5                           | without_anomaly_scores              | 0.803177           | 0.176585          | 0.888042           | 13.173031              | 0.200728    | 0.891611     | 13.612821        |                 | 0.159545       | 0.840455 | 0.520308        | 0.041021 | 0.048088        | valid_business_fdr30 | True                    | 1.000000           | 0.000000     | 0.004202            | 20.366164         | 10.502358         | 0.300310               | 0.154863               | 0.203782            | 0.525210            |                   |                   |                                 |                  | medium           | 4.000000           | deploy_candidate     | 73.500000      | False          | False                  | False            | False                             | False                                    | False                   |
| balance_gate / XGBoost / random_oversampling                                      | balance_gate               | XGBoost      | baseline_eda_onehot                            | random_oversampling                                  | train_sample                         | without_anomaly_scores              | 0.968919           | 0.176480          | 0.889105           | 13.165161              | 0.194081    | 0.891628     | 13.162064        |                 | 0.162754       | 0.837246 | 0.509804        | 0.039250 | 0.046188        | valid_business_fdr30 | True                    | 0.727273           | 0.272727     | 0.005602            | 19.876256         | 10.432342         | 0.293086               | 0.153830               | 0.198880            | 0.521709            |                   |                   |                                 |                  | medium           | 2.000000           | deploy_candidate     | 62.500000      | False          | False                  | False            | False                             | False                                    | False                   |

## 5. Decision Figures

### Fraud prevalence by month

![Fraud prevalence by month](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/00_data_audit/figures/01_fraud_prevalence_by_month.png)

_Monthly fraud prevalence with train/validation/test split coloring._

### Candidate ranking by PR-AUC and recall

![Candidate ranking by PR-AUC and recall](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/16_final_report/figures/candidate_ranking_pr_auc_recall_fpr5.png)

_Ranking contrasts model discrimination with operational fraud capture._

### Operational metrics heatmap

![Operational metrics heatmap](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/16_final_report/figures/operational_metrics_desirability_heatmap_top8.png)

_Cell color is normalized per metric so higher color intensity is always better; annotations show raw values._

### Confusion matrices for top candidates

![Confusion matrices for top candidates](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/16_final_report/figures/confusion_matrices_top8_test.png)

_Top candidates evaluated on test at the validation-selected 5% FPR policy._

### Final model threshold trade-off

![Final model threshold trade-off](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/16_final_report/figures/final_model_tradeoff_curve.png)

_Precision, recall, and FDR for hyperparameter tuned / CatBoost._

### Top-k precision, recall, and lift

![Top-k precision, recall, and lift](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/09_topk/figures/topk_precision_recall_lift_curves.png)

_Top-k alert budget diagnostics on test._

### Fraud captured versus alert budget

![Fraud captured versus alert budget](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/09_topk/figures/fraud_captured_vs_alert_budget.png)

_Top-k alert budget diagnostics on test._

### Fairness by housing status

![Fairness by housing status](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/11_fairness/figures/fairness_housing_status_rates.png)

_Alert rate, FPR, and recall by housing status._

### Fairness disparity across main candidates

![Fairness disparity across main candidates](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/11_fairness/figures/fairness_disparity_top8.png)

_Maximum FPR gap by protected attribute._

### Global SHAP importance

![Global SHAP importance](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/10_shap/figures/shap_global_importance_hyperparameter_tuned_catboost.png)

_Mean absolute SHAP values for the final model._

### Grouped SHAP importance

![Grouped SHAP importance](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/10_shap/figures/shap_grouped_feature_importance_hyperparameter_tuned_catboost.png)

_Transformed features grouped back to parent feature names when possible._

### SHAP beeswarm

![SHAP beeswarm](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/10_shap/figures/shap_beeswarm_hyperparameter_tuned_catboost.png)

_Distribution of SHAP contributions for the final model; shown only when feature names are interpretable._

### SHAP contribution patterns by housing_status

![SHAP contribution patterns by housing_status](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/10_shap/figures/shap_group_contribution_heatmap_housing_status.png)

_Mean absolute SHAP values by categorical group with denominator context when available._

### SHAP contribution patterns by device_os

![SHAP contribution patterns by device_os](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/10_shap/figures/shap_group_contribution_heatmap_device_os.png)

_Mean absolute SHAP values by categorical group with denominator context when available._

### SHAP contribution patterns by employment_status

![SHAP contribution patterns by employment_status](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/10_shap/figures/shap_group_contribution_heatmap_employment_status.png)

_Mean absolute SHAP values by categorical group with denominator context when available._

### Feature-family ablation

![Feature-family ablation](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/12_feature_ablation/figures/feature_family_ablation_pr_auc.png)

_PR-AUC sensitivity to feature-family changes._

### Tuned versus fixed model comparison

![Tuned versus fixed model comparison](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/05b_hyperparameter_tuning_gate/figures/tuned_vs_fixed_model_comparison.png)

_Tuned model families compared with the fixed CatBoost reference._

## 6. Threshold Policy Comparison

| model                                 | threshold_policy             | feasible | precision | fdr      | recall_tpr | fpr      | alert_rate |
| ------------------------------------- | ---------------------------- | -------- | --------- | -------- | ---------- | -------- | ---------- |
| hyperparameter_tuned / XGBoost        | valid_global_5pct_fpr        | True     | 0.156250  | 0.843750 | 0.553221   | 0.044710 | 0.052208   |
| hyperparameter_tuned / XGBoost        | valid_business_fdr30         | True     | 0.888889  | 0.111111 | 0.005602   | 0.000010 | 0.000093   |
| hyperparameter_tuned / XGBoost        | valid_cost_sensitive_5_to_1  | True     | 0.307477  | 0.692523 | 0.230392   | 0.007766 | 0.011049   |
| hyperparameter_tuned / XGBoost        | valid_cost_sensitive_10_to_1 | True     | 0.207358  | 0.792642 | 0.434174   | 0.024839 | 0.030875   |
| hyperparameter_tuned / XGBoost        | valid_cost_sensitive_25_to_1 | True     | 0.118838  | 0.881162 | 0.624650   | 0.069318 | 0.077507   |
| hyperparameter_tuned / XGBoost        | valid_cost_sensitive_50_to_1 | True     | 0.073365  | 0.926635 | 0.759804   | 0.143625 | 0.152711   |
| advanced_gate / CatBoost              | valid_global_5pct_fpr        | True     | 0.163755  | 0.836245 | 0.525210   | 0.040140 | 0.047293   |
| advanced_gate / CatBoost              | valid_business_fdr30         | True     | 0.916667  | 0.083333 | 0.007703   | 0.000010 | 0.000124   |
| advanced_gate / CatBoost              | valid_cost_sensitive_5_to_1  | True     | 0.369343  | 0.630657 | 0.177171   | 0.004528 | 0.007073   |
| advanced_gate / CatBoost              | valid_cost_sensitive_10_to_1 | True     | 0.233469  | 0.766531 | 0.400560   | 0.019682 | 0.025299   |
| advanced_gate / CatBoost              | valid_cost_sensitive_25_to_1 | True     | 0.120888  | 0.879112 | 0.625350   | 0.068061 | 0.076278   |
| advanced_gate / CatBoost              | valid_cost_sensitive_50_to_1 | True     | 0.086211  | 0.913789 | 0.721989   | 0.114531 | 0.123489   |
| balance_gate / XGBoost / class_weight | valid_global_5pct_fpr        | True     | 0.161112  | 0.838888 | 0.531513   | 0.041419 | 0.048646   |
| balance_gate / XGBoost / class_weight | valid_business_fdr30         | True     | 0.888889  | 0.111111 | 0.005602   | 0.000010 | 0.000093   |
| balance_gate / XGBoost / class_weight | valid_cost_sensitive_5_to_1  | True     | 0.327522  | 0.672478 | 0.184174   | 0.005659 | 0.008292   |
| balance_gate / XGBoost / class_weight | valid_cost_sensitive_10_to_1 | True     | 0.216707  | 0.783293 | 0.376050   | 0.020343 | 0.025588   |
| balance_gate / XGBoost / class_weight | valid_cost_sensitive_25_to_1 | True     | 0.115482  | 0.884518 | 0.637255   | 0.073049 | 0.081369   |
| balance_gate / XGBoost / class_weight | valid_cost_sensitive_50_to_1 | True     | 0.073329  | 0.926671 | 0.758403   | 0.143437 | 0.152505   |
| baseline / XGBoost                    | valid_global_5pct_fpr        | True     | 0.161112  | 0.838888 | 0.531513   | 0.041419 | 0.048646   |
| baseline / XGBoost                    | valid_business_fdr30         | True     | 0.888889  | 0.111111 | 0.005602   | 0.000010 | 0.000093   |

## 7. Top-K Alert Prioritization

| model                          | split      | topk_pct | topk_label  | k     | precision_at_k | fdr_at_k | recall_at_k | lift_at_k | fp_per_tp_at_k | fraud_prevalence | captured_frauds |
| ------------------------------ | ---------- | -------- | ----------- | ----- | -------------- | -------- | ----------- | --------- | -------------- | ---------------- | --------------- |
| hyperparameter_tuned / XGBoost | validation | 0.005000 | top_0.5pct  | 541   | 0.353050       | 0.646950 | 0.131724    | 26.337036 | 1.832461       | 0.013405         | 191             |
| hyperparameter_tuned / XGBoost | validation | 0.010000 | top_1.0pct  | 1082  | 0.267098       | 0.732902 | 0.199310    | 19.925140 | 2.743945       | 0.013405         | 289             |
| hyperparameter_tuned / XGBoost | validation | 0.020000 | top_2.0pct  | 2164  | 0.208872       | 0.791128 | 0.311724    | 15.581597 | 3.787611       | 0.013405         | 452             |
| hyperparameter_tuned / XGBoost | validation | 0.050000 | top_5.0pct  | 5409  | 0.135145       | 0.864855 | 0.504138    | 10.081640 | 6.399453       | 0.013405         | 731             |
| hyperparameter_tuned / XGBoost | validation | 0.100000 | top_10.0pct | 10817 | 0.088379       | 0.911621 | 0.659310    | 6.592982  | 10.314854      | 0.013405         | 956             |
| hyperparameter_tuned / XGBoost | test       | 0.005000 | top_0.5pct  | 485   | 0.397938       | 0.602062 | 0.135154    | 26.987061 | 1.512953       | 0.014746         | 193             |
| hyperparameter_tuned / XGBoost | test       | 0.010000 | top_1.0pct  | 969   | 0.318885       | 0.681115 | 0.216387    | 21.625927 | 2.135922       | 0.014746         | 309             |
| hyperparameter_tuned / XGBoost | test       | 0.020000 | top_2.0pct  | 1937  | 0.250903       | 0.749097 | 0.340336    | 17.015577 | 2.985597       | 0.014746         | 486             |
| hyperparameter_tuned / XGBoost | test       | 0.050000 | top_5.0pct  | 4843  | 0.160851       | 0.839149 | 0.545518    | 10.908449 | 5.216945       | 0.014746         | 779             |
| hyperparameter_tuned / XGBoost | test       | 0.100000 | top_10.0pct | 9685  | 0.098709       | 0.901291 | 0.669468    | 6.694194  | 9.130753       | 0.014746         | 956             |
| advanced_gate / CatBoost       | validation | 0.005000 | top_0.5pct  | 541   | 0.351201       | 0.648799 | 0.131034    | 26.199146 | 1.847368       | 0.013405         | 190             |
| advanced_gate / CatBoost       | validation | 0.010000 | top_1.0pct  | 1082  | 0.273567       | 0.726433 | 0.204138    | 20.407756 | 2.655405       | 0.013405         | 296             |
| advanced_gate / CatBoost       | validation | 0.020000 | top_2.0pct  | 2164  | 0.211645       | 0.788355 | 0.315862    | 15.788433 | 3.724891       | 0.013405         | 458             |
| advanced_gate / CatBoost       | validation | 0.050000 | top_5.0pct  | 5409  | 0.134221       | 0.865779 | 0.500690    | 10.012682 | 6.450413       | 0.013405         | 726             |
| advanced_gate / CatBoost       | validation | 0.100000 | top_10.0pct | 10817 | 0.088287       | 0.911713 | 0.658621    | 6.586085  | 10.326702      | 0.013405         | 955             |
| advanced_gate / CatBoost       | test       | 0.005000 | top_0.5pct  | 485   | 0.400000       | 0.600000 | 0.135854    | 27.126891 | 1.500000       | 0.014746         | 194             |
| advanced_gate / CatBoost       | test       | 0.010000 | top_1.0pct  | 969   | 0.326109       | 0.673891 | 0.221289    | 22.115835 | 2.066456       | 0.014746         | 316             |
| advanced_gate / CatBoost       | test       | 0.020000 | top_2.0pct  | 1937  | 0.255034       | 0.744966 | 0.345938    | 17.295669 | 2.921053       | 0.014746         | 494             |
| advanced_gate / CatBoost       | test       | 0.050000 | top_5.0pct  | 4843  | 0.158786       | 0.841214 | 0.538515    | 10.768418 | 5.297789       | 0.014746         | 769             |
| advanced_gate / CatBoost       | test       | 0.100000 | top_10.0pct | 9685  | 0.100568       | 0.899432 | 0.682073    | 6.820235  | 8.943532       | 0.014746         | 974             |

## 8. Low-FPR Sweep

| model                                 | fpr_cap_label | threshold | alerts | tp  | fp   | fn   | tn    | precision | fdr      | recall_tpr | fpr      | alert_rate | lift      | fp_per_tp |
| ------------------------------------- | ------------- | --------- | ------ | --- | ---- | ---- | ----- | --------- | -------- | ---------- | -------- | ---------- | --------- | --------- |
| hyperparameter_tuned / XGBoost        | FPR<=0.25%    | 0.985243  | 397    | 162 | 235  | 1266 | 95180 | 0.408060  | 0.591940 | 0.113445   | 0.002463 | 0.004099   | 27.673528 | 1.450617  |
| hyperparameter_tuned / XGBoost        | FPR<=0.50%    | 0.978266  | 690    | 247 | 443  | 1181 | 94972 | 0.357971  | 0.642029 | 0.172969   | 0.004643 | 0.007125   | 24.276602 | 1.793522  |
| hyperparameter_tuned / XGBoost        | FPR<=1.00%    | 0.966147  | 1244   | 361 | 883  | 1067 | 94532 | 0.290193  | 0.709807 | 0.252801   | 0.009254 | 0.012846   | 19.680080 | 2.445983  |
| hyperparameter_tuned / XGBoost        | FPR<=2.00%    | 0.943913  | 2312   | 541 | 1771 | 887  | 93644 | 0.233997  | 0.766003 | 0.378852   | 0.018561 | 0.023874   | 15.868996 | 3.273567  |
| hyperparameter_tuned / XGBoost        | FPR<=3.00%    | 0.923544  | 3258   | 654 | 2604 | 774  | 92811 | 0.200737  | 0.799263 | 0.457983   | 0.027291 | 0.033642   | 13.613403 | 3.981651  |
| hyperparameter_tuned / XGBoost        | FPR<=5.00%    | 0.885196  | 5056   | 790 | 4266 | 638  | 91149 | 0.156250  | 0.843750 | 0.553221   | 0.044710 | 0.052208   | 10.596442 | 5.400000  |
| advanced_gate / CatBoost              | FPR<=0.25%    | 0.957380  | 337    | 143 | 194  | 1285 | 95221 | 0.424332  | 0.575668 | 0.100140   | 0.002033 | 0.003480   | 28.777043 | 1.356643  |
| advanced_gate / CatBoost              | FPR<=0.50%    | 0.941231  | 645    | 248 | 397  | 1180 | 95018 | 0.384496  | 0.615504 | 0.173669   | 0.004161 | 0.006660   | 26.075461 | 1.600806  |
| advanced_gate / CatBoost              | FPR<=1.00%    | 0.914916  | 1145   | 352 | 793  | 1076 | 94622 | 0.307424  | 0.692576 | 0.246499   | 0.008311 | 0.011823   | 20.848615 | 2.252841  |
| advanced_gate / CatBoost              | FPR<=2.00%    | 0.874613  | 2143   | 529 | 1614 | 899  | 93801 | 0.246850  | 0.753150 | 0.370448   | 0.016916 | 0.022129   | 16.740697 | 3.051040  |
| advanced_gate / CatBoost              | FPR<=3.00%    | 0.841512  | 2998   | 622 | 2376 | 806  | 93039 | 0.207472  | 0.792528 | 0.435574   | 0.024902 | 0.030957   | 14.070152 | 3.819936  |
| advanced_gate / CatBoost              | FPR<=5.00%    | 0.788399  | 4580   | 750 | 3830 | 678  | 91585 | 0.163755  | 0.836245 | 0.525210   | 0.040140 | 0.047293   | 11.105441 | 5.106667  |
| balance_gate / XGBoost / class_weight | FPR<=0.25%    | 0.953471  | 354    | 148 | 206  | 1280 | 95209 | 0.418079  | 0.581921 | 0.103641   | 0.002159 | 0.003655   | 28.352965 | 1.391892  |
| balance_gate / XGBoost / class_weight | FPR<=0.50%    | 0.939550  | 619    | 221 | 398  | 1207 | 95017 | 0.357027  | 0.642973 | 0.154762   | 0.004171 | 0.006392   | 24.212613 | 1.800905  |
| balance_gate / XGBoost / class_weight | FPR<=1.00%    | 0.916753  | 1120   | 332 | 788  | 1096 | 94627 | 0.296429  | 0.703571 | 0.232493   | 0.008259 | 0.011565   | 20.102964 | 2.373494  |
| balance_gate / XGBoost / class_weight | FPR<=2.00%    | 0.880094  | 2147   | 491 | 1656 | 937  | 93759 | 0.228691  | 0.771309 | 0.343838   | 0.017356 | 0.022170   | 15.509203 | 3.372709  |
| balance_gate / XGBoost / class_weight | FPR<=3.00%    | 0.850423  | 3033   | 602 | 2431 | 826  | 92984 | 0.198483  | 0.801517 | 0.421569   | 0.025478 | 0.031319   | 13.460590 | 4.038206  |
| balance_gate / XGBoost / class_weight | FPR<=5.00%    | 0.796552  | 4711   | 759 | 3952 | 669  | 91463 | 0.161112  | 0.838888 | 0.531513   | 0.041419 | 0.048646   | 10.926189 | 5.206851  |
| baseline / XGBoost                    | FPR<=0.25%    | 0.953471  | 354    | 148 | 206  | 1280 | 95209 | 0.418079  | 0.581921 | 0.103641   | 0.002159 | 0.003655   | 28.352965 | 1.391892  |
| baseline / XGBoost                    | FPR<=0.50%    | 0.939550  | 619    | 221 | 398  | 1207 | 95017 | 0.357027  | 0.642973 | 0.154762   | 0.004171 | 0.006392   | 24.212613 | 1.800905  |
| baseline / XGBoost                    | FPR<=1.00%    | 0.916753  | 1120   | 332 | 788  | 1096 | 94627 | 0.296429  | 0.703571 | 0.232493   | 0.008259 | 0.011565   | 20.102964 | 2.373494  |
| baseline / XGBoost                    | FPR<=2.00%    | 0.880094  | 2147   | 491 | 1656 | 937  | 93759 | 0.228691  | 0.771309 | 0.343838   | 0.017356 | 0.022170   | 15.509203 | 3.372709  |
| baseline / XGBoost                    | FPR<=3.00%    | 0.850423  | 3033   | 602 | 2431 | 826  | 92984 | 0.198483  | 0.801517 | 0.421569   | 0.025478 | 0.031319   | 13.460590 | 4.038206  |
| baseline / XGBoost                    | FPR<=5.00%    | 0.796552  | 4711   | 759 | 3952 | 669  | 91463 | 0.161112  | 0.838888 | 0.531513   | 0.041419 | 0.048646   | 10.926189 | 5.206851  |
| baseline / CatBoost                   | FPR<=0.25%    | 0.958785  | 344    | 145 | 199  | 1283 | 95216 | 0.421512  | 0.578488 | 0.101541   | 0.002086 | 0.003552   | 28.585750 | 1.372414  |
| baseline / CatBoost                   | FPR<=0.50%    | 0.942853  | 625    | 234 | 391  | 1194 | 95024 | 0.374400  | 0.625600 | 0.163866   | 0.004098 | 0.006454   | 25.390770 | 1.670940  |
| baseline / CatBoost                   | FPR<=1.00%    | 0.917309  | 1148   | 355 | 793  | 1073 | 94622 | 0.309233  | 0.690767 | 0.248599   | 0.008311 | 0.011854   | 20.971355 | 2.233803  |
| baseline / CatBoost                   | FPR<=2.00%    | 0.876459  | 2146   | 524 | 1622 | 904  | 93793 | 0.244175  | 0.755825 | 0.366947   | 0.016999 | 0.022160   | 16.559286 | 3.095420  |
| baseline / CatBoost                   | FPR<=3.00%    | 0.842981  | 2975   | 619 | 2356 | 809  | 93059 | 0.208067  | 0.791933 | 0.433473   | 0.024692 | 0.030720   | 14.110542 | 3.806139  |
| baseline / CatBoost                   | FPR<=5.00%    | 0.787401  | 4695   | 763 | 3932 | 665  | 91483 | 0.162513  | 0.837487 | 0.534314   | 0.041209 | 0.048481   | 11.021202 | 5.153342  |

## 9. Cascade CatBoost/Stage-1 To TP/FP Filter

| model                                     | split | top_risk_pct | final_top_pct | stage2_threshold_or_rank_cutoff | base_alerts_before_filter | base_precision_before_filter | base_fdr_before_filter | base_recall_before_filter | base_fpr_before_filter | threshold | alerts | alert_count | tp  | fp  | fn   | tn    | precision | fdr      | recall_tpr | fpr      | fnr      | tnr      | specificity_tnr | alert_rate | fraud_prevalence | lift      | precision_lift | fp_per_tp | pr_auc   | roc_auc  | pr_auc_lift | n_obs | runtime_seconds |
| ----------------------------------------- | ----- | ------------ | ------------- | ------------------------------- | ------------------------- | ---------------------------- | ---------------------- | ------------------------- | ---------------------- | --------- | ------ | ----------- | --- | --- | ---- | ----- | --------- | -------- | ---------- | -------- | -------- | -------- | --------------- | ---------- | ---------------- | --------- | -------------- | --------- | -------- | -------- | ----------- | ----- | --------------- |
| cascade / CatBoost -> Logistic Regression | test  | 0.050000     | 0.010000      | 0.677610                        | 4843                      | 0.158786                     | 0.841214               | 0.538515                  | 0.042698               | 0.500000  | 969    | 969         | 320 | 649 | 1108 | 94766 | 0.330237  | 0.669763 | 0.224090   | 0.006802 | 0.775910 | 0.993198 | 0.993198        | 0.010006   | 0.014746         | 22.395782 | 22.395782      | 2.028125  | 0.085444 | 0.608644 | 5.794573    | 96843 | 573.354684      |
| cascade / CatBoost -> LightGBM            | test  | 0.050000     | 0.010000      | 0.939476                        | 4843                      | 0.158786                     | 0.841214               | 0.538515                  | 0.042698               | 0.500000  | 969    | 969         | 306 | 663 | 1122 | 94752 | 0.315789  | 0.684211 | 0.214286   | 0.006949 | 0.785714 | 0.993051 | 0.993051        | 0.010006   | 0.014746         | 21.415966 | 21.415966      | 2.166667  | 0.079255 | 0.603669 | 5.374850    | 96843 | 573.354684      |
| cascade / CatBoost -> CatBoost            | test  | 0.050000     | 0.010000      | 0.955326                        | 4843                      | 0.158786                     | 0.841214               | 0.538515                  | 0.042698               | 0.500000  | 969    | 969         | 317 | 652 | 1111 | 94763 | 0.327141  | 0.672859 | 0.221989   | 0.006833 | 0.778011 | 0.993167 | 0.993167        | 0.010006   | 0.014746         | 22.185821 | 22.185821      | 2.056782  | 0.084094 | 0.607578 | 5.703015    | 96843 | 573.354684      |

## 10. RIFF-Style Low-FPR Rules

| fpr_cap  | threshold | alerts | alert_count | tp  | fp   | fn   | tn    | precision | fdr      | recall_tpr | fpr      | fnr      | tnr      | specificity_tnr | alert_rate | fraud_prevalence | lift     | precision_lift | fp_per_tp | pr_auc   | roc_auc  | pr_auc_lift | n_obs |
| -------- | --------- | ------ | ----------- | --- | ---- | ---- | ----- | --------- | -------- | ---------- | -------- | -------- | -------- | --------------- | ---------- | ---------------- | -------- | -------------- | --------- | -------- | -------- | ----------- | ----- |
| 0.002500 | 0.500000  | 195    | 195         | 7   | 188  | 1421 | 95227 | 0.035897  | 0.964103 | 0.004902   | 0.001970 | 0.995098 | 0.998030 | 0.998030        | 0.002014   | 0.014746         | 2.434465 | 2.434465       | 26.857143 | 0.014849 | 0.501466 | 1.007032    | 96843 |
| 0.005000 | 0.500000  | 663    | 663         | 12  | 651  | 1416 | 94764 | 0.018100  | 0.981900 | 0.008403   | 0.006823 | 0.991597 | 0.993177 | 0.993177        | 0.006846   | 0.014746         | 1.227461 | 1.227461       | 54.250000 | 0.014774 | 0.500790 | 1.001911    | 96843 |
| 0.010000 | 0.500000  | 740    | 740         | 23  | 717  | 1405 | 94698 | 0.031081  | 0.968919 | 0.016106   | 0.007515 | 0.983894 | 0.992485 | 0.992485        | 0.007641   | 0.014746         | 2.107833 | 2.107833       | 31.173913 | 0.015009 | 0.504296 | 1.017843    | 96843 |
| 0.020000 | 0.500000  | 2134   | 2134        | 107 | 2027 | 1321 | 93388 | 0.050141  | 0.949859 | 0.074930   | 0.021244 | 0.925070 | 0.978756 | 0.978756        | 0.022036   | 0.014746         | 3.400395 | 3.400395       | 18.943925 | 0.017398 | 0.526843 | 1.179862    | 96843 |

## 11. Undersample Imbalance Ensembles

| model                                                               | split | threshold_policy      | threshold | alerts | alert_count | tp  | fp   | fn   | tn    | precision | fdr      | recall_tpr | fpr      | fnr      | tnr      | specificity_tnr | alert_rate | fraud_prevalence | lift      | precision_lift | fp_per_tp | pr_auc   | roc_auc  | pr_auc_lift | n_obs | precision_top0_5pct | fdr_top0_5pct | recall_top0_5pct | lift_top0_5pct | fp_per_tp_top0_5pct | precision_top1pct | fdr_top1pct | recall_top1pct | lift_top1pct | fp_per_tp_top1pct | score_diag_roc_auc | score_diag_inverted_roc_auc | score_diag_score_inversion_suspected | score_diag_inverted_pr_auc | method              | selection           | implementation  | classes_ |
| ------------------------------------------------------------------- | ----- | --------------------- | --------- | ------ | ----------- | --- | ---- | ---- | ----- | --------- | -------- | ---------- | -------- | -------- | -------- | --------------- | ---------- | ---------------- | --------- | -------------- | --------- | -------- | -------- | ----------- | ----- | ------------------- | ------------- | ---------------- | -------------- | ------------------- | ----------------- | ----------- | -------------- | ------------ | ----------------- | ------------------ | --------------------------- | ------------------------------------ | -------------------------- | ------------------- | ------------------- | --------------- | -------- |
| imbalance_ensemble_gate / EasyEnsemble / default                    | test  | valid_global_5pct_fpr | 0.789991  | 5029   | 5029        | 618 | 4411 | 810  | 91004 | 0.122887  | 0.877113 | 0.432773   | 0.046230 | 0.567227 | 0.953770 | 0.953770        | 0.051929   | 0.014746         | 8.333873  | 8.333873       | 7.137540  | 0.133520 | 0.858604 | 9.054969    | 96843 | 0.272165            | 0.727835      | 0.092437         | 18.457472      | 2.674242            | 0.227038          | 0.772962    | 0.154062       | 15.397100    | 3.404545          | 0.858604           | 0.141396                    | False                                | 0.007927                   | EasyEnsemble        | default             | official_imbens | 0,1      |
| imbalance_ensemble_gate / EasyEnsemble / validation_selected        | test  | valid_global_5pct_fpr | 0.699715  | 4994   | 4994        | 650 | 4344 | 778  | 91071 | 0.130156  | 0.869844 | 0.455182   | 0.045527 | 0.544818 | 0.954473 | 0.954473        | 0.051568   | 0.014746         | 8.826832  | 8.826832       | 6.683077  | 0.144846 | 0.868793 | 9.823066    | 96843 | 0.317526            | 0.682474      | 0.107843         | 21.533717      | 2.149351            | 0.241486          | 0.758514    | 0.163866       | 16.376915    | 3.141026          | 0.868793           | 0.131207                    | False                                | 0.007866                   | EasyEnsemble        | validation_selected | official_imbens | 0,1      |
| imbalance_ensemble_gate / Balance Cascade / default                 | test  | valid_global_5pct_fpr | 0.657912  | 4476   | 4476        | 104 | 4372 | 1324 | 91043 | 0.023235  | 0.976765 | 0.072829   | 0.045821 | 0.927171 | 0.954179 | 0.954179        | 0.046219   | 0.014746         | 1.575735  | 1.575735       | 42.038462 | 0.018788 | 0.453298 | 1.274160    | 96843 | 0.070103            | 0.929897      | 0.023810         | 4.754197       | 13.264706           | 0.041280          | 0.958720    | 0.028011       | 2.799473     | 23.225000         | 0.453298           | 0.546702                    | True                                 | 0.015895                   | Balance Cascade     | default             | official_imbens | 0,1      |
| imbalance_ensemble_gate / Balance Cascade / validation_selected     | test  | valid_global_5pct_fpr | 0.646993  | 3090   | 3090        | 132 | 2958 | 1296 | 92457 | 0.042718  | 0.957282 | 0.092437   | 0.031001 | 0.907563 | 0.968999 | 0.968999        | 0.031907   | 0.014746         | 2.897047  | 2.897047       | 22.409091 | 0.033722 | 0.474637 | 2.286909    | 96843 | 0.107216            | 0.892784      | 0.036415         | 7.271125       | 8.326923            | 0.073271          | 0.926729    | 0.049720       | 4.969064     | 12.647887         | 0.474637           | 0.525363                    | True                                 | 0.015667                   | Balance Cascade     | validation_selected | official_imbens | 0,1      |
| imbalance_ensemble_gate / Self-paced Ensemble / default             | test  | valid_global_5pct_fpr | 0.639306  | 4329   | 4329        | 684 | 3645 | 744  | 91770 | 0.158004  | 0.841996 | 0.478992   | 0.038202 | 0.521008 | 0.961798 | 0.961798        | 0.044701   | 0.014746         | 10.715404 | 10.715404      | 5.328947  | 0.184167 | 0.868708 | 12.489725   | 96843 | 0.364948            | 0.635052      | 0.123950         | 24.749792      | 1.740113            | 0.282766          | 0.717234    | 0.191877       | 19.176388    | 2.536496          | 0.868708           | 0.131292                    | False                                | 0.007847                   | Self-paced Ensemble | default             | official_imbens | 0,1      |
| imbalance_ensemble_gate / Self-paced Ensemble / validation_selected | test  | valid_global_5pct_fpr | 0.655821  | 4876   | 4876        | 691 | 4185 | 737  | 91230 | 0.141715  | 0.858285 | 0.483894   | 0.043861 | 0.516106 | 0.956139 | 0.956139        | 0.050350   | 0.014746         | 9.610686  | 9.610686       | 6.056440  | 0.162057 | 0.866254 | 10.990272   | 96843 | 0.315464            | 0.684536      | 0.107143         | 21.393888      | 2.169935            | 0.260062          | 0.739938    | 0.176471       | 17.636678    | 2.845238          | 0.866254           | 0.133746                    | False                                | 0.007866                   | Self-paced Ensemble | validation_selected | official_imbens | 0,1      |

## 12. Hyperparameter Tuning Gate

| model                                      | stage                      | model_family        | validation_pr_auc | test_pr_auc | test_threshold_precision | test_threshold_fdr | test_threshold_recall_tpr | test_threshold_fpr | test_precision_top1pct | test_recall_top1pct | test_fprcap_0.05_max_fpr_gap |
| ------------------------------------------ | -------------------------- | ------------------- | ----------------- | ----------- | ------------------------ | ------------------ | ------------------------- | ------------------ | ---------------------- | ------------------- | ---------------------------- |
| baseline / CatBoost                        | baseline_search            | CatBoost            | 0.180242          | 0.217709    | 0.162513                 | 0.837487           | 0.534314                  | 0.041209           | 0.319917               | 0.217087            | 0.167751                     |
| hyperparameter_tuned / CatBoost            | hyperparameter-tuning-gate | CatBoost            | 0.176810          | 0.208517    | 0.159704                 | 0.840296           | 0.514006                  | 0.040476           | 0.314757               | 0.213585            | 0.142756                     |
| hyperparameter_tuned / XGBoost             | hyperparameter-tuning-gate | XGBoost             | 0.182158          | 0.215888    | 0.156250                 | 0.843750           | 0.553221                  | 0.044710           | 0.318885               | 0.216387            | 0.177104                     |
| hyperparameter_tuned / LightGBM            | hyperparameter-tuning-gate | LightGBM            | 0.166244          | 0.192851    | 0.150243                 | 0.849757           | 0.498599                  | 0.042205           | 0.301342               | 0.204482            | 0.160368                     |
| hyperparameter_tuned / Logistic Regression | hyperparameter-tuning-gate | Logistic Regression | 0.151966          | 0.183884    | 0.152982                 | 0.847018           | 0.457983                  | 0.037950           | 0.283798               | 0.192577            | 0.163813                     |

## 13. SHAP Interpretability

| model                           | feature                          | mean_abs_shap | mean_shap | reportable_feature_name |
| ------------------------------- | -------------------------------- | ------------- | --------- | ----------------------- |
| hyperparameter_tuned / CatBoost | device_os                        | 0.437619      | -0.035104 | True                    |
| hyperparameter_tuned / CatBoost | has_other_cards                  | 0.393067      | 0.009888  | True                    |
| hyperparameter_tuned / CatBoost | phone_home_valid                 | 0.358787      | -0.018147 | True                    |
| hyperparameter_tuned / CatBoost | name_email_similarity            | 0.331635      | 0.063136  | True                    |
| hyperparameter_tuned / CatBoost | keep_alive_session               | 0.322498      | -0.010322 | True                    |
| hyperparameter_tuned / CatBoost | current_address_months_count     | 0.321310      | 0.019698  | True                    |
| hyperparameter_tuned / CatBoost | prev_address_months_count        | 0.310504      | 0.024160  | True                    |
| hyperparameter_tuned / CatBoost | income                           | 0.310120      | 0.093908  | True                    |
| hyperparameter_tuned / CatBoost | housing_status                   | 0.248814      | 0.015254  | True                    |
| hyperparameter_tuned / CatBoost | intended_balcon_amount           | 0.235381      | 0.003271  | True                    |
| hyperparameter_tuned / CatBoost | customer_age                     | 0.220175      | 0.004721  | True                    |
| hyperparameter_tuned / CatBoost | bank_branch_count_8w             | 0.201860      | 0.003390  | True                    |
| hyperparameter_tuned / CatBoost | employment_status                | 0.182484      | 0.027487  | True                    |
| hyperparameter_tuned / CatBoost | credit_risk_score                | 0.145995      | 0.036652  | True                    |
| hyperparameter_tuned / CatBoost | email_is_free                    | 0.133240      | -0.017006 | True                    |
| hyperparameter_tuned / CatBoost | email_free__source               | 0.122325      | -0.016670 | True                    |
| hyperparameter_tuned / CatBoost | velocity_4w                      | 0.112862      | 0.025170  | True                    |
| hyperparameter_tuned / CatBoost | date_of_birth_distinct_emails_4w | 0.096123      | 0.040293  | True                    |
| hyperparameter_tuned / CatBoost | days_since_request               | 0.090066      | -0.013785 | True                    |
| hyperparameter_tuned / CatBoost | velocity_24h                     | 0.086618      | -0.063703 | True                    |

## 14. Fairness By Protected Group

| model                    | feature_policy      | threshold_policy             | attribute          | max_fpr_gap | max_tpr_gap | fpr_ratio_worst_to_best | tpr_ratio_worst_to_best | equal_opportunity_difference | equalized_odds_difference | disparate_impact_ratio_alert_positive | disparate_impact_ratio_approval_positive | worst_fpr_group | lowest_tpr_group |
| ------------------------ | ------------------- | ---------------------------- | ------------------ | ----------- | ----------- | ----------------------- | ----------------------- | ---------------------------- | ------------------------- | ------------------------------------- | ---------------------------------------- | --------------- | ---------------- |
| advanced_gate / CatBoost | with_housing_status | valid_business_fdr30         | customer_age_group | 0.000360    | 0.030864    | 1.000000                | 0.000000                | 0.030864                     | 0.030864                  | 0.000000                              | 0.997961                                 | 51+             | 10-20            |
| advanced_gate / CatBoost | with_housing_status | valid_business_fdr30         | employment_status  | 0.000014    | 0.011905    | 1.000000                | 0.000000                | 0.011905                     | 0.011905                  | 0.000000                              | 0.999558                                 | CA              | CB               |
| advanced_gate / CatBoost | with_housing_status | valid_business_fdr30         | housing_status     | 0.000027    | 0.011123    | 1.000000                | 0.000000                | 0.011123                     | 0.011123                  | 0.000000                              | 0.999479                                 | BC              | BB               |
| advanced_gate / CatBoost | with_housing_status | valid_business_fdr30         | income_group       | 0.000016    | 0.008636    | 1.000000                | 0.000000                | 0.008636                     | 0.008636                  | 0.000000                              | 0.999829                                 | high_0.7_0.9    | mid_0.4_0.6      |
| advanced_gate / CatBoost | with_housing_status | valid_cost_sensitive_10_to_1 | customer_age_group | 0.078338    | 0.456754    | 20.408183               | 0.237174                | 0.456754                     | 0.456754                  | 0.045211                              | 0.893668                                 | 51+             | 10-20            |
| advanced_gate / CatBoost | with_housing_status | valid_cost_sensitive_10_to_1 | employment_status  | 0.033063    | 0.428571    | 19.540119               | 0.000000                | 0.428571                     | 0.428571                  | 0.035920                              | 0.952243                                 | CC              | CF               |
| advanced_gate / CatBoost | with_housing_status | valid_cost_sensitive_10_to_1 | housing_status     | 0.086036    | 0.528365    | 81.224799               | 0.000000                | 0.528365                     | 0.528365                  | 0.000000                              | 0.893233                                 | BA              | BG               |
| advanced_gate / CatBoost | with_housing_status | valid_cost_sensitive_10_to_1 | income_group       | 0.020307    | 0.159005    | 4.439832                | 0.628025                | 0.159005                     | 0.159005                  | 0.233317                              | 0.974173                                 | high_0.7_0.9    | low_0.1_0.3      |
| advanced_gate / CatBoost | with_housing_status | valid_global_5pct_fpr        | customer_age_group | 0.134855    | 0.508949    | 13.881044               | 0.289227                | 0.508949                     | 0.508949                  | 0.067081                              | 0.833128                                 | 51+             | 10-20            |
| advanced_gate / CatBoost | with_housing_status | valid_global_5pct_fpr        | employment_status  | 0.071208    | 0.595238    | 15.519903               | 0.000000                | 0.595238                     | 0.595238                  | 0.055068                              | 0.909412                                 | CC              | CG               |
| advanced_gate / CatBoost | with_housing_status | valid_global_5pct_fpr        | housing_status     | 0.163102    | 1.000000    | 39.250097               | 0.000000                | 1.000000                     | 1.000000                  | 0.000000                              | 0.812845                                 | BA              | BG               |
| advanced_gate / CatBoost | with_housing_status | valid_global_5pct_fpr        | income_group       | 0.038676    | 0.149992    | 3.791224                | 0.728607                | 0.149992                     | 0.149992                  | 0.270832                              | 0.954379                                 | high_0.7_0.9    | low_0.1_0.3      |
| advanced_gate / XGBoost  | with_housing_status | valid_business_fdr30         | customer_age_group | 0.000360    | 0.030864    | 11.495683               | 0.000000                | 0.030864                     | 0.030864                  | 0.000000                              | 0.997961                                 | 51+             | 10-20            |
| advanced_gate / XGBoost  | with_housing_status | valid_business_fdr30         | employment_status  | 0.000041    | 0.013389    | 1.000000                | 0.000000                | 0.013389                     | 0.013389                  | 0.000000                              | 0.999558                                 | CA              | CB               |
| advanced_gate / XGBoost  | with_housing_status | valid_business_fdr30         | housing_status     | 0.000109    | 0.017798    | 4.073292                | 0.000000                | 0.017798                     | 0.017798                  | 0.000000                              | 0.999062                                 | BA              | BB               |
| advanced_gate / XGBoost  | with_housing_status | valid_business_fdr30         | income_group       | 0.000047    | 0.013817    | 1.000000                | 0.000000                | 0.013817                     | 0.013817                  | 0.000000                              | 0.999705                                 | high_0.7_0.9    | mid_0.4_0.6      |
| advanced_gate / XGBoost  | with_housing_status | valid_cost_sensitive_10_to_1 | customer_age_group | 0.068476    | 0.409001    | 18.897083               | 0.224414                | 0.409001                     | 0.409001                  | 0.048175                              | 0.908015                                 | 51+             | 10-20            |
| advanced_gate / XGBoost  | with_housing_status | valid_cost_sensitive_10_to_1 | employment_status  | 0.048143    | 0.464286    | 107.984869              | 0.000000                | 0.464286                     | 0.464286                  | 0.000000                              | 0.936424                                 | CC              | CF               |
| advanced_gate / XGBoost  | with_housing_status | valid_cost_sensitive_10_to_1 | housing_status     | 0.083684    | 0.496107    | 73.361241               | 0.000000                | 0.496107                     | 0.496107                  | 0.000000                              | 0.896987                                 | BA              | BG               |
| advanced_gate / XGBoost  | with_housing_status | valid_cost_sensitive_10_to_1 | income_group       | 0.018862    | 0.134431    | 4.391279                | 0.654831                | 0.134431                     | 0.134431                  | 0.237601                              | 0.976200                                 | high_0.7_0.9    | low_0.1_0.3      |

## 15. Feature Ablation

| model                                                               | stage            | model_family        | feature_set                | balance_policy          | train_strategy | anomaly_policy         | selected_threshold | validation_pr_auc | validation_roc_auc | validation_pr_auc_lift | test_pr_auc | test_roc_auc | test_pr_auc_lift | runtime_seconds |
| ------------------------------------------------------------------- | ---------------- | ------------------- | -------------------------- | ----------------------- | -------------- | ---------------------- | ------------------ | ----------------- | ------------------ | ---------------------- | ----------- | ------------ | ---------------- | --------------- |
| feature_ablation / original_with_sensitive / Logistic Regression    | feature_ablation | Logistic Regression | original_with_sensitive    | model_default_weighting | train_sample   | without_anomaly_scores | 0.812025           | 0.162295          | 0.884487           | 12.106948              | 0.195334    | 0.887706     | 13.246985        |                 |
| feature_ablation / original_with_sensitive / Random Forest          | feature_ablation | Random Forest       | original_with_sensitive    | model_default_weighting | train_sample   | without_anomaly_scores | 0.633807           | 0.150817          | 0.875864           | 11.250722              | 0.173599    | 0.875194     | 11.773026        |                 |
| feature_ablation / original_without_sensitive / Logistic Regression | feature_ablation | Logistic Regression | original_without_sensitive | model_default_weighting | train_sample   | without_anomaly_scores | 0.788888           | 0.126059          | 0.853073           | 9.403849               | 0.158432    | 0.861447     | 10.744389        |                 |
| feature_ablation / original_without_sensitive / Random Forest       | feature_ablation | Random Forest       | original_without_sensitive | model_default_weighting | train_sample   | without_anomaly_scores | 0.639078           | 0.119724          | 0.850832           | 8.931226               | 0.146115    | 0.853353     | 9.909112         |                 |

## 16. Anomaly Score Comparison

This analysis was not run.

## 17. Recency And Temporal Robustness

| model                                                                                                        | model_family        | train_strategy            | anomaly_policy                      | validation_pr_auc | test_pr_auc |
| ------------------------------------------------------------------------------------------------------------ | ------------------- | ------------------------- | ----------------------------------- | ----------------- | ----------- |
| anomaly_recency_gate / XGBoost / full_0_5 / without_anomaly_scores                                           | XGBoost             | full_0_5                  | without_anomaly_scores              | 0.177167          | 0.212117    |
| anomaly_recency_gate / LightGBM / full_0_5 / without_anomaly_scores                                          | LightGBM            | full_0_5                  | without_anomaly_scores              | 0.165343          | 0.207728    |
| anomaly_recency_gate / Random Forest / full_0_5 / without_anomaly_scores                                     | Random Forest       | full_0_5                  | without_anomaly_scores              | 0.150817          | 0.173599    |
| anomaly_recency_gate / Logistic Regression / full_0_5 / without_anomaly_scores                               | Logistic Regression | full_0_5                  | without_anomaly_scores              | 0.162295          | 0.195334    |
| anomaly_recency_gate / XGBoost / full_0_5 / with_isolation_forest_anomaly_score                              | XGBoost             | full_0_5                  | with_isolation_forest_anomaly_score | 0.177784          | 0.212653    |
| anomaly_recency_gate / LightGBM / full_0_5 / with_isolation_forest_anomaly_score                             | LightGBM            | full_0_5                  | with_isolation_forest_anomaly_score | 0.168691          | 0.205502    |
| anomaly_recency_gate / Random Forest / full_0_5 / with_isolation_forest_anomaly_score                        | Random Forest       | full_0_5                  | with_isolation_forest_anomaly_score | 0.150869          | 0.174033    |
| anomaly_recency_gate / Logistic Regression / full_0_5 / with_isolation_forest_anomaly_score                  | Logistic Regression | full_0_5                  | with_isolation_forest_anomaly_score | 0.168290          | 0.200250    |
| anomaly_recency_gate / XGBoost / full_0_5_recency_weighted / without_anomaly_scores                          | XGBoost             | full_0_5_recency_weighted | without_anomaly_scores              | 0.175669          | 0.209945    |
| anomaly_recency_gate / LightGBM / full_0_5_recency_weighted / without_anomaly_scores                         | LightGBM            | full_0_5_recency_weighted | without_anomaly_scores              | 0.169001          | 0.206000    |
| anomaly_recency_gate / Random Forest / full_0_5_recency_weighted / without_anomaly_scores                    | Random Forest       | full_0_5_recency_weighted | without_anomaly_scores              | 0.147944          | 0.171871    |
| anomaly_recency_gate / Logistic Regression / full_0_5_recency_weighted / without_anomaly_scores              | Logistic Regression | full_0_5_recency_weighted | without_anomaly_scores              | 0.162937          | 0.195545    |
| anomaly_recency_gate / XGBoost / full_0_5_recency_weighted / with_isolation_forest_anomaly_score             | XGBoost             | full_0_5_recency_weighted | with_isolation_forest_anomaly_score | 0.174488          | 0.209610    |
| anomaly_recency_gate / LightGBM / full_0_5_recency_weighted / with_isolation_forest_anomaly_score            | LightGBM            | full_0_5_recency_weighted | with_isolation_forest_anomaly_score | 0.167428          | 0.200514    |
| anomaly_recency_gate / Random Forest / full_0_5_recency_weighted / with_isolation_forest_anomaly_score       | Random Forest       | full_0_5_recency_weighted | with_isolation_forest_anomaly_score | 0.146261          | 0.173007    |
| anomaly_recency_gate / Logistic Regression / full_0_5_recency_weighted / with_isolation_forest_anomaly_score | Logistic Regression | full_0_5_recency_weighted | with_isolation_forest_anomaly_score | 0.168690          | 0.199541    |
| anomaly_recency_gate / XGBoost / recent_3_5 / without_anomaly_scores                                         | XGBoost             | recent_3_5                | without_anomaly_scores              | 0.176585          | 0.200728    |
| anomaly_recency_gate / LightGBM / recent_3_5 / without_anomaly_scores                                        | LightGBM            | recent_3_5                | without_anomaly_scores              | 0.155923          | 0.196249    |
| anomaly_recency_gate / Random Forest / recent_3_5 / without_anomaly_scores                                   | Random Forest       | recent_3_5                | without_anomaly_scores              | 0.148214          | 0.175160    |
| anomaly_recency_gate / Logistic Regression / recent_3_5 / without_anomaly_scores                             | Logistic Regression | recent_3_5                | without_anomaly_scores              | 0.162900          | 0.189874    |

## 18. Calibration

This analysis was not run.

## 19. Stability And Uncertainty

This analysis was not run.

This analysis was not run.

## 20. Notes

- Each stage reads persisted inputs and writes its own folder under `results_full_train`.
- The final report is intentionally tolerant of skipped optional experiments.
- Thresholds are selected on validation and then reported on test when the threshold stage has been run.
