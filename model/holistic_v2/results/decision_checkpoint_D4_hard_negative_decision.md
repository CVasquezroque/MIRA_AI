# Decision Checkpoint D4 - Hard Negative Mining

## Checkpoint Name

D4 hard negative mining

## Purpose

Evaluate whether hard negative mining can materially reduce false positives / FDR
while preserving useful recall and alert volume.

## Candidates Or Options Evaluated

| readable_model_name                                                                                                                                                          | validation_pr_auc | validation_recall_at_fpr5 | validation_precision_at_fpr5 | validation_fdr_at_fpr5 | validation_precision_top_1pct |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------- | ---------------------------- | ---------------------- | ----------------------------- |
| CatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=scale_pos_weight | loss=logloss | train=months_0_5 | ensemble=none                                  | 0.170692          | 0.512414                  | 0.122466                     | 0.877534               | 0.263401                      |
| HardNegativeCatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=hard_negative_weighting | loss=logloss | train=months_0_5 | ensemble=C1_score_band_3x   | 0.172433          | 0.519310                  | 0.125375                     | 0.874625               | 0.261553                      |
| HardNegativeCatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=hard_negative_weighting | loss=logloss | train=months_0_5 | ensemble=C1_score_band_5x   | 0.171519          | 0.520690                  | 0.124096                     | 0.875904               | 0.266174                      |
| HardNegativeCatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=hard_negative_weighting | loss=logloss | train=months_0_5 | ensemble=C1_rank_band_3x    | 0.172718          | 0.515862                  | 0.123087                     | 0.876913               | 0.275416                      |
| HardNegativeCatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=two_stage_filter | loss=logloss | train=months_0_5 | ensemble=C2_two_stage_alert_filter | 0.024466          | 0.093793                  | 0.024908                     | 0.975092               | 0.035120                      |
| CatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=scale_pos_weight | loss=logloss | train=months_0_5 | ensemble=none                                  | 0.170692          | 0.412414                  | 0.157410                     | 0.842590               | 0.263401                      |
| CatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=scale_pos_weight | loss=logloss | train=months_0_5 | ensemble=none                                  | 0.170692          | 0.339310                  | 0.189888                     | 0.810112               | 0.263401                      |
| CatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=scale_pos_weight | loss=logloss | train=months_0_5 | ensemble=none                                  | 0.170692          | 0.236552                  | 0.243781                     | 0.756219               | 0.263401                      |

## Validation Metrics Used

- validation FDR at FPR <= 5%;
- validation PR-AUC;
- validation recall at FPR <= 5%;
- validation Precision@Top 1%.

## Decision Made

`keep as benchmark`

## Best Hard Negative Candidate

`HardNegativeCatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=hard_negative_weighting | loss=logloss | train=months_0_5 | ensemble=C1_score_band_3x`

- FDR delta vs baseline: `-0.002909`
- PR-AUC delta vs baseline: `0.001742`
- Recall delta vs baseline: `0.006897`

## Reason For The Decision

Hard negative mining did not materially reduce FDR without hurting other metrics.

## Next Step

Run D5 tuned focal-loss XGBoost.
