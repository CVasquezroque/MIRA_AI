# Decision Checkpoint D3 - Temporal Blending Summary

- Did temporal blending beat best individual CatBoost in validation PR-AUC? `Yes` (Delta: 0.002153)
- Did it beat best D2 weighted blend in validation PR-AUC? `No` (Delta: -0.001318)
- Did it improve Precision@Top1%? `No`
- Did it reduce FDR? `No`
- Did it improve recall at FPR <= 5%? `No`
- Is the added complexity justified? `No`

## Final D3 decision
`keep as benchmark`

Temporal blending did not clearly outperform the simpler weighted blend. Keep as benchmark. Do not promote as final model.
