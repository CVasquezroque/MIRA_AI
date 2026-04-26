# Decision Checkpoint D2 - Weighted Score Blending

## Checkpoint Name

D2 weighted blend

## Purpose

Optimize score-level blends that include CatBoost and evaluate whether blending
improves ranking, FPR-constrained recall, FDR, or top-K alert quality.

## Candidates Or Options Evaluated

| readable_model_name                                                                                                                        | validation_pr_auc | validation_recall_at_fpr5 | validation_precision_at_fpr5 | validation_fdr_at_fpr5 | validation_precision_top_1pct |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | ------------------------- | ---------------------------- | ---------------------- | ----------------------------- |
| Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A1_uniform_probability_average         | 0.174176          | 0.520690                  | 0.124137                     | 0.875863               | 0.264325                      |
| Blend | rep=ranks | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A2_uniform_rank_average                 | 0.174902          | 0.521379                  | 0.124322                     | 0.875678               | 0.265250                      |
| Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A3_weighted_probability_pr_auc         | 0.175970          | 0.513103                  | 0.122691                     | 0.877309               | 0.269871                      |
| Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A4_weighted_probability_recall_fpr5    | 0.169230          | 0.525517                  | 0.125288                     | 0.874712               | 0.259704                      |
| Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A5_weighted_probability_precision_top1 | 0.173387          | 0.517241                  | 0.123376                     | 0.876624               | 0.278189                      |
| Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A6_weighted_probability_fdr_reduction  | 0.174917          | 0.521379                  | 0.125937                     | 0.874063               | 0.265250                      |
| Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A7_sparse_blend_max3                   | 0.176485          | 0.515172                  | 0.123267                     | 0.876733               | 0.268022                      |

## Validation Metrics Used

- validation PR-AUC;
- validation recall at FPR <= 5%;
- validation precision and FDR at FPR <= 5%;
- validation Precision@Top 1%;
- validation Recall@Top 1%.

Weights were learned on validation only. Test metrics were generated but not used
to choose weights or winners.

## Decision Made

`promote`

## Promoted Candidates

| readable_model_name                                                                                                                        | validation_pr_auc | validation_recall_at_fpr5 | validation_precision_at_fpr5 | validation_fdr_at_fpr5 | validation_precision_top_1pct |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | ------------------------- | ---------------------------- | ---------------------- | ----------------------------- |
| Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A5_weighted_probability_precision_top1 | 0.173387          | 0.517241                  | 0.123376                     | 0.876624               | 0.278189                      |

## Discarded Candidates

| readable_model_name                                                                                                                       | validation_pr_auc | validation_recall_at_fpr5 | validation_precision_at_fpr5 | validation_fdr_at_fpr5 | validation_precision_top_1pct |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------- | ---------------------------- | ---------------------- | ----------------------------- |
| Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A1_uniform_probability_average        | 0.174176          | 0.520690                  | 0.124137                     | 0.875863               | 0.264325                      |
| Blend | rep=ranks | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A2_uniform_rank_average                | 0.174902          | 0.521379                  | 0.124322                     | 0.875678               | 0.265250                      |
| Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A3_weighted_probability_pr_auc        | 0.175970          | 0.513103                  | 0.122691                     | 0.877309               | 0.269871                      |
| Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A4_weighted_probability_recall_fpr5   | 0.169230          | 0.525517                  | 0.125288                     | 0.874712               | 0.259704                      |
| Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A6_weighted_probability_fdr_reduction | 0.174917          | 0.521379                  | 0.125937                     | 0.874063               | 0.265250                      |
| Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A7_sparse_blend_max3                  | 0.176485          | 0.515172                  | 0.123267                     | 0.876733               | 0.268022                      |

## Skipped Candidates

- `skip for runtime`: exhaustive continuous optimization; a coarse simplex grid
  plus sparse constrained grid was used instead.

## Reason For The Decision

The selected blend improves at least one validation operational metric without materially worsening FDR.

PR-AUC improvement is `negligible` by the configured thresholds.

## Risks Or Limitations

- Validation-only weight tuning can overfit month 6.
- Blends add operational complexity relative to CatBoost alone.
- D3 checks whether a stricter temporal design is worth the added complexity.

## Next Step

Run D3 temporal blending / stacking with out-of-time base predictions.
