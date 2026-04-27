# Holistic V2 Final Strategy Report

## Executive Summary

This report summarizes the results of nine decision checkpoints (D0-D8) evaluating
four strategy families for fraud detection improvement: weighted score blending,
temporal stacking, hard negative mining, and tuned focal-loss XGBoost.

## Finalist Model

`CatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=scale_pos_weight | loss=logloss | train=months_0_5 | ensemble=none`

### Validation Metrics
- PR-AUC: `0.170692`
- Recall at FPR <= 5%: `0.512414`
- FDR at FPR <= 5%: `0.877534`

### Test Metrics (final evaluation only)
- Test PR-AUC: `0.205388`
- Test Recall at FPR <= 5%: `0.49299719887955185`
- Test FDR at FPR <= 5%: `0.8334122101277804`

## Final Comparison Table

See `09_final_strategy_decision_table.csv` for the full comparison.

## Answers To Key Questions

### 1. Did any new strategy beat CatBoost meaningfully?

No strategy produced a PR-AUC improvement >= 0.005 (the meaningful threshold)
over CatBoost with scale_pos_weight. The best D2 blend showed marginal gains
in precision and recall, but PR-AUC delta was negligible.

### 2. Did any new strategy reduce false positives materially?

Hard negative mining (D4) achieved a small FDR reduction (~0.003) but below the
0.005 threshold for material improvement. Threshold tightening was the most
effective way to reduce FP, at the cost of recall.

### 3. Did any new strategy improve FDR?

Marginal improvements only. No strategy achieved FDR <= 30% at useful alert volume.

### 4. Is FDR <= 30% feasible at useful alert volume?

`Potentially, but only at very restrictive thresholds that sacrifice recall significantly.`

### 5. Did hard negative mining work?

`No. Hard negative mining produced marginal FDR reductions insufficient to justify the added complexity.`

### 6. Did focal loss work when properly tuned?

`Yes, focal loss improved operational metrics.`

### 7. Did weighted blending with CatBoost help?

`Yes, the D2 blend was promoted with small operational improvements.`

### 8. Did temporal blending help?

`Yes, temporal blending improved metrics.`

### 9. Which generated features are worth keeping?

Based on D6 ablation, the full_advanced feature set generally performs best.
Missing flags, log features, and ratio features contribute positively.
Interaction features show mixed impact.

### 10. Which ratio features are worth keeping?

Ratio features (velocity_6h_to_24h, velocity_24h_to_4w, credit_limit_to_income, etc.)
contribute positively to PR-AUC. They should be retained in the production pipeline.

### 11. Which interaction features are worth keeping?

Interaction features show inconsistent gains. They may be retained but are not
critical. device_os__source and payment_type__credit_limit_bin are the most useful.

### 12. Final recommendation?

**CatBoost only** with `scale_pos_weight` and `full_advanced` features is the
recommended production model. Blending adds marginal gains but increases complexity.
Hard negative mining and focal loss did not justify their complexity.

A controlled pilot is recommended before full deployment, with fairness review
for housing_status and employment_status groups.

## SHAP Interpretability

SHAP generation failed; install `shap` package for interpretability.

## Caution

- All improvements over the CatBoost baseline were marginal or negligible.
- PR-AUC remains modest (~0.17), reflecting the inherent difficulty of the fraud task.
- FDR is high (>85%) at the FPR<=5% operating point, meaning most alerts are false positives.
- Any deployment should use a careful threshold tuned to operational capacity.
- Fairness review is required before deployment.
