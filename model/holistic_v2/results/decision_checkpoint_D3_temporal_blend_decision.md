# Decision Checkpoint D3 - Temporal Blending

## Checkpoint Name

D3 temporal blending

## Purpose

Evaluate a time-aware ensemble using out-of-time base predictions from forward
month folds rather than shuffled stacking.

## Candidates Or Options Evaluated

| readable_model_name                                                                                                                         | validation_pr_auc | validation_recall_at_fpr5 | validation_precision_at_fpr5 | validation_fdr_at_fpr5 | validation_precision_top_1pct |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------- | ---------------------------- | ---------------------- | ----------------------------- |
| TemporalBlend | rep=oof_scores | feat=mixed | balance=mixed | loss=mixed | train=temporal_oof | ensemble=temporal_logistic_meta_model       | 0.174063          | 0.519310                  | 0.124586                     | 0.875414               | 0.267098                      |
| TemporalBlend | rep=oof_scores | feat=mixed | balance=mixed | loss=mixed | train=temporal_oof | ensemble=temporal_oof_pr_auc_weighted_blend | 0.173809          | 0.517931                  | 0.123682                     | 0.876318               | 0.265250                      |
| TemporalBlend | rep=oof_scores | feat=mixed | balance=mixed | loss=mixed | train=temporal_oof | ensemble=temporal_uniform_rank_blend        | 0.175167          | 0.515172                  | 0.122882                     | 0.877118               | 0.270795                      |

## Validation Metrics Used

- validation PR-AUC;
- validation Precision@Top 1%;
- validation recall at FPR <= 5%;
- validation FDR and precision at FPR <= 5%;
- OOF fold stability proxy across months 3, 4, and 5.

## Decision Made

`keep as benchmark`

## Promoted Candidates

_No temporal blend promoted over the simpler D2 blend._

## Discarded Candidates

| readable_model_name                                                                                                                         | validation_pr_auc | validation_recall_at_fpr5 | validation_precision_at_fpr5 | validation_fdr_at_fpr5 | validation_precision_top_1pct |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------- | ---------------------------- | ---------------------- | ----------------------------- |
| TemporalBlend | rep=oof_scores | feat=mixed | balance=mixed | loss=mixed | train=temporal_oof | ensemble=temporal_logistic_meta_model       | 0.174063          | 0.519310                  | 0.124586                     | 0.875414               | 0.267098                      |
| TemporalBlend | rep=oof_scores | feat=mixed | balance=mixed | loss=mixed | train=temporal_oof | ensemble=temporal_oof_pr_auc_weighted_blend | 0.173809          | 0.517931                  | 0.123682                     | 0.876318               | 0.265250                      |

## Skipped Candidates

- `skip for runtime`: larger meta-models and shuffled stacking.

## Reason For The Decision

Temporal blending is not clearly better than the simpler D2 blend, so complexity is not justified yet.

## Risks Or Limitations

- OOF folds are still trained on sampled prior-month data for runtime.
- Meta-model behavior may drift if month 6 differs materially from months 3-5.
- Additional operational complexity must be justified by clear top-K, FDR, or recall gains.

## Next Step

Run D4 hard negative mining to attack false positives directly.
