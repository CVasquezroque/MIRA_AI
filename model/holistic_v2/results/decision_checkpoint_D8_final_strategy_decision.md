# Decision Checkpoint D8 - Final Strategy Decision

## Checkpoint Name

D8 final strategy report

## Purpose

Final comparison of all strategies, SHAP interpretability, and deployment recommendation.

## Final Candidate

`CatBoost | rep=native_cat | feat=original_plus_basic_generated | balance=scale_pos_weight | loss=logloss | train=months_0_5 | ensemble=none`

## Decision Made

`final candidate`

## Promoted Final Model

CatBoost with scale_pos_weight, full_advanced features, trained on months 0-5.

## Reason

CatBoost consistently delivered the best or near-best performance across all
checkpoints. No alternative strategy (blending, temporal stacking, hard negative
mining, focal loss) produced meaningful improvements that justified their added
complexity.

## Deployment Recommendation

Controlled pilot with:
- CatBoost native categorical model
- Full advanced feature set
- Threshold selected on validation at FPR <= 5%
- Fairness monitoring for housing_status and employment_status
- Regular recalibration on new monthly data

## Risks

- High FDR at operational threshold
- Fairness gaps may require mitigation
- Model may degrade with temporal drift
- PR-AUC is modest; operational expectations should be calibrated accordingly
