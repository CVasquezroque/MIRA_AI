# D2 Weighted Blend Summary

## Best Individual Benchmark

`CatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=none | loss=logloss | train=months_0_5 | ensemble=none`

## Best Weighted Blend

`Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A5_weighted_probability_precision_top1`

## Validation Deltas Versus Best Individual

- PR-AUC delta: `0.000372` (negligible)
- Recall at FPR <= 5% delta: `0.007586`
- FDR delta: `-0.001629`
- Precision@Top 1% delta: `0.008318`

## Non-Trivial Weights

| base_readable_model_name                                                                                                                         | weight   |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| CatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=scale_pos_weight | loss=logloss | train=months_0_5 | ensemble=none      | 0.500000 |
| CatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=none | loss=logloss | train=months_0_5 | ensemble=none                  | 0.250000 |
| LogisticRegression | rep=target_frequency | feat=full_advanced | balance=class_weight_balanced | loss=logloss | train=months_0_5 | ensemble=none | 0.250000 |

## Conclusions

- Did blending improve PR-AUC over the best individual model? `yes`
- Did blending improve recall at FPR <= 5%? `yes`
- Did blending improve precision or reduce FDR? `yes`
- Did blending improve Precision@Top 1%? `yes`
- Is the blend worth keeping over CatBoost alone? `promote`
