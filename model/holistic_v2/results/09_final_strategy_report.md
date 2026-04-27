# Holistic V2 Final Strategy Report

## 1. Executive Summary
This report summarizes the results of nine decision checkpoints (D0-D8) evaluating four strategy families for fraud detection improvement: weighted score blending, temporal stacking, hard negative mining, and tuned focal-loss XGBoost.

## 2. What was corrected after the initial holistic_v2 run
Logic errors relating to FDR interpretation were corrected. Decision rules for promotion were strictly enforced. Reports and figures were regenerated to ensure accurate conclusions without blind hardcoding.

## 3. Best individual CatBoost baseline
CatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=none | loss=logloss | train=months_0_5 | ensemble=none (PR-AUC: 0.173015)

## 4. Weighted blending result
Marginal gains; kept as benchmark.

## 5. Temporal blending result
Did not clearly outperform simpler blends; kept as benchmark.

## 6. Hard negative mining result
Did not materially reduce FDR; kept as benchmark.

## 7. Tuned focal-loss XGBoost result
Did not consistently beat standard logloss XGBoost; kept as benchmark.

## 8. Feature ablation result
Generated features provided mixed and mostly marginal gains.

## 9. Fairness result
Fairness review remains required due to significant disparities.

## 10. SHAP or permutation interpretability result
Interpretability generated successfully.

## 11. FDR <= 30% feasibility
No, not with the current models and features. It may be feasible only at very restrictive thresholds with very low recall.

## 12. Final recommendation
Controlled pilot with human review, using CatBoost native with scale_pos_weight as the main scoring model for a controlled pilot.. Do not deploy as automatic blocking system. Continue fairness review and threshold tuning according to operational capacity.

## 13. Limitations
Improvements were marginal. FDR remains high. Fairness gaps exist.

### Benchmarks Note
- Best PR-AUC benchmark: Blend | rep=scores | feat=mixed | balance=mixed | loss=mixed | train=validation_weighted | ensemble=A7_sparse_blend_max3
- Best FDR benchmark: CatBoost | rep=native_cat | feat=full_advanced_without_ratios | balance=scale_pos_weight | loss=logloss | train=months_0_5 | ensemble=none
- Best Precision@Top1% benchmark: HardNegativeCatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=hard_negative_weighting | loss=logloss | train=months_0_5 | ensemble=C1_rank_band_3x
