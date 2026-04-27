# Decision Checkpoint D5 - Focal Loss Summary

## Deltas vs Standard XGBoost
- PR-AUC delta: `0.001382`
- Precision@Top1% delta: `-0.001848`
- FDR delta: `0.001760`

## Final D5 decision
`keep as benchmark`

Focal loss showed at most marginal improvements over standard XGBoost and did not solve the false-alert problem. It should not replace CatBoost or be promoted as an operational improvement.
