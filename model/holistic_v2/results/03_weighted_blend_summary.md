# Decision Checkpoint D2 - Weighted Blending Summary

- Best individual CatBoost baseline PR-AUC: `0.173015`
- Best weighted blend validation PR-AUC: `0.176485`
- Best weighted blend validation Precision@Top1%: `0.268022`
- Best weighted blend validation FDR: `0.876733`

## Deltas vs CatBoost
- PR-AUC delta: `0.003470` (marginal)
- FDR delta: `-0.001521` (marginal)
- Precision@Top1% delta: `-0.001848` (negligible)

## Final D2 decision
`keep as benchmark`

Weighted blending produced at most marginal ranking improvements. It did not materially reduce FDR or false-positive burden. Keep as benchmark, not as final operational replacement.
