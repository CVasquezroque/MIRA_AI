# Decision Checkpoint D1 - Improved Baseline Top 5

## Checkpoint Name

D1 improved baseline

## Purpose

Create a clean, interpretable baseline set before testing focused strategies A-D.
The stage includes CatBoost, XGBoost, LightGBM, Logistic Regression, and
score-level ensembles that include CatBoost.

## Candidates Or Options Evaluated

| readable_model_name                                                                                                                                        | validation_pr_auc | validation_recall_at_fpr5 | validation_precision_at_fpr5 | validation_fdr_at_fpr5 | validation_precision_top_1pct |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------- | ---------------------------- | ---------------------- | ----------------------------- |
| CatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=scale_pos_weight | loss=logloss | train=months_0_5 | ensemble=none                | 0.170692          | 0.512414                  | 0.122466                     | 0.877534               | 0.263401                      |
| CatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=none | loss=logloss | train=months_0_5 | ensemble=none                            | 0.173015          | 0.509655                  | 0.121746                     | 0.878254               | 0.269871                      |
| XGBoost | rep=target_frequency | feat=full_advanced | balance=scale_pos_weight | loss=logloss | train=months_0_5 | ensemble=none                           | 0.169369          | 0.519310                  | 0.123828                     | 0.876172               | 0.258780                      |
| XGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=logloss | train=months_0_5 | ensemble=none                                       | 0.168792          | 0.502759                  | 0.120456                     | 0.879544               | 0.261553                      |
| LightGBM | rep=target_frequency | feat=full_advanced | balance=scale_pos_weight | loss=logloss | train=months_0_5 | ensemble=none                          | 0.163167          | 0.492414                  | 0.118114                     | 0.881886               | 0.266174                      |
| LightGBM | rep=target_frequency | feat=full_advanced | balance=none | loss=logloss | train=months_0_5 | ensemble=none                                      | 0.167058          | 0.514483                  | 0.123265                     | 0.876735               | 0.260628                      |
| LogisticRegression | rep=target_frequency | feat=full_advanced | balance=class_weight_balanced | loss=logloss | train=months_0_5 | ensemble=none           | 0.155680          | 0.495862                  | 0.119933                     | 0.880067               | 0.252311                      |
| LogisticRegression | rep=target_frequency | feat=full_advanced | balance=none | loss=logloss | train=months_0_5 | ensemble=none                            | 0.147468          | 0.481379                  | 0.115793                     | 0.884207               | 0.246765                      |
| Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=uniform_score | ensemble=weighted_score_blend_cat_xgb_lgbm_lr                         | 0.174215          | 0.522759                  | 0.124856                     | 0.875144               | 0.269871                      |
| Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=validation_pr_auc_weighted_score_blend_cat_xgb_lgbm_lr | 0.174297          | 0.521379                  | 0.125249                     | 0.874751               | 0.269871                      |
| TemporalBlend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=time_aware_month5 | ensemble=time_aware_month5_weighted_score_blend           | 0.174363          | 0.520690                  | 0.124917                     | 0.875083               | 0.269871                      |
| Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=rank_average | ensemble=rank_average_cat_xgb_lgbm_lr                                  | 0.174894          | 0.515862                  | 0.123249                     | 0.876751               | 0.268022                      |

## Validation Metrics Used

- validation PR-AUC;
- validation recall at FPR <= 5%;
- validation precision and FDR at FPR <= 5%;
- validation Precision@Top 1%;
- validation Recall@Top 1%;
- interpretability / deployability score.

Test metrics were generated for later reporting, but they were not used to select
the promoted top five.

## Decision Made

`promote`

## Promoted Candidates

| selection_rank | readable_model_name                                                                                                                                        | validation_pr_auc | validation_recall_at_fpr5 | validation_precision_at_fpr5 | validation_fdr_at_fpr5 | validation_precision_top_1pct | multi_objective_score |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------- | ---------------------------- | ---------------------- | ----------------------------- | --------------------- |
| 1              | Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=rank_average | ensemble=rank_average_cat_xgb_lgbm_lr                                  | 0.174894          | 0.515862                  | 0.123249                     | 0.876751               | 0.268022                      | 0.880309              |
| 2              | Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=uniform_score | ensemble=weighted_score_blend_cat_xgb_lgbm_lr                         | 0.174215          | 0.522759                  | 0.124856                     | 0.875144               | 0.269871                      | 0.964692              |
| 3              | Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=validation_pr_auc_weighted_score_blend_cat_xgb_lgbm_lr | 0.174297          | 0.521379                  | 0.125249                     | 0.874751               | 0.269871                      | 0.969299              |
| 4              | CatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=none | loss=logloss | train=months_0_5 | ensemble=none                            | 0.173015          | 0.509655                  | 0.121746                     | 0.878254               | 0.269871                      | 0.821013              |
| 5              | TemporalBlend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=time_aware_month5 | ensemble=time_aware_month5_weighted_score_blend           | 0.174363          | 0.520690                  | 0.124917                     | 0.875083               | 0.269871                      | 0.952932              |

## Discarded Candidates

Non-promoted models are not carried into every focused strategy. They may remain
`keep as benchmark` rows for final comparison if useful.

| readable_model_name                                                                                                                              | validation_pr_auc | validation_recall_at_fpr5 | validation_precision_at_fpr5 | validation_precision_top_1pct |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | ------------------------- | ---------------------------- | ----------------------------- |
| CatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=scale_pos_weight | loss=logloss | train=months_0_5 | ensemble=none      | 0.170692          | 0.512414                  | 0.122466                     | 0.263401                      |
| XGBoost | rep=target_frequency | feat=full_advanced | balance=scale_pos_weight | loss=logloss | train=months_0_5 | ensemble=none                 | 0.169369          | 0.519310                  | 0.123828                     | 0.258780                      |
| XGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=logloss | train=months_0_5 | ensemble=none                             | 0.168792          | 0.502759                  | 0.120456                     | 0.261553                      |
| LightGBM | rep=target_frequency | feat=full_advanced | balance=none | loss=logloss | train=months_0_5 | ensemble=none                            | 0.167058          | 0.514483                  | 0.123265                     | 0.260628                      |
| LightGBM | rep=target_frequency | feat=full_advanced | balance=scale_pos_weight | loss=logloss | train=months_0_5 | ensemble=none                | 0.163167          | 0.492414                  | 0.118114                     | 0.266174                      |
| LogisticRegression | rep=target_frequency | feat=full_advanced | balance=class_weight_balanced | loss=logloss | train=months_0_5 | ensemble=none | 0.155680          | 0.495862                  | 0.119933                     | 0.252311                      |
| LogisticRegression | rep=target_frequency | feat=full_advanced | balance=none | loss=logloss | train=months_0_5 | ensemble=none                  | 0.147468          | 0.481379                  | 0.115793                     | 0.246765                      |

## Skipped Candidates

- `skip for runtime`: sklearn Voting/Stacking that cannot include CatBoost's
  native categorical path directly.
- `skip for runtime`: random shuffled stacking, because D3 implements temporal
  out-of-time stacking/blending instead.

## Reason For The Decision

The selected top five cover complementary roles: best ranking model, strongest
operational recall/precision trade-offs, top-K alert quality, at least one
CatBoost benchmark, and a competitive score-level ensemble when validation
metrics justify it. The leading validation multi-objective candidate is:

`Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=validation_pr_auc_weighted_score_blend_cat_xgb_lgbm_lr`

## Risks Or Limitations

- D1 uses a controlled stratified training sample of up to `180,000` rows for runtime.
- Some ensemble weights are validation-derived diagnostics; D2 performs the real
  constrained blend optimization.
- The time-aware D1 blend is deliberately simple; D3 performs the proper temporal
  out-of-time design.
- Fairness is not decided in D1 and remains `requires fairness review`.

## Next Step

Run D2 weighted score blending with CatBoost included and optimize blend weights
using validation metrics only.
