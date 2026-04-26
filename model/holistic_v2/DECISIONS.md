# Holistic V2 Decision Log

This module is intentionally separate from `model/holistic`, which remains the
historical completed pipeline.

## 2026-04-26 - Experiment Scope

The prior holistic run found CatBoost native to be the strongest stable ranking
candidate. Holistic V2 therefore avoids reopening unlimited model search and
focuses on four targeted hypotheses:

- weighted score blending can include CatBoost and optimize operational metrics;
- temporal blending can reduce leakage risk versus shuffled stacking;
- hard negative mining may reduce false positives directly;
- focal-loss XGBoost should be treated as a tuned family, not a fixed one-off.

## Decision Discipline

Each checkpoint writes a decision markdown file under
`model/holistic_v2/results/` and must be committed separately. Decisions use
validation metrics only. Test metrics may be reported for final evaluation but
must not be used to pick winners before D8.

Allowed decision labels:

- promote
- keep as benchmark
- discard
- skip for runtime
- not operationally meaningful
- requires fairness review
- final candidate

## Output Isolation

All new outputs must stay under `model/holistic_v2/results/`. No checkpoint may
write into `model/holistic/results/`.
