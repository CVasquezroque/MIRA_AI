# D3 Temporal Blending Summary

## Best Temporal Candidate

`TemporalBlend | rep=oof_scores | feat=mixed | balance=mixed | loss=mixed | train=temporal_oof | ensemble=temporal_uniform_rank_blend`

## Comparison Target

`Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A5_weighted_probability_precision_top1`

## Validation Deltas

- PR-AUC delta: `0.001780`
- Precision@Top 1% delta: `-0.007394`
- FDR delta: `0.000494`
- Recall at FPR <= 5% delta: `-0.002069`

## Fold Stability Proxy

| month | oof_pr_auc |
| ----- | ---------- |
| 3     | 0.153554   |
| 4     | 0.177270   |
| 5     | 0.176512   |

## Answers

- Does temporal blending beat the best individual model? `yes`
- Does it improve top-K precision? `no`
- Does it reduce FDR? `no`
- Is the improvement large enough to justify complexity? `keep as benchmark`
- Does the temporal design avoid leakage? `yes, base OOF scores are generated only from prior months.`
