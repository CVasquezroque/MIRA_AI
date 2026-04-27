# Decision Checkpoint D6 - CatBoost Feature Ablation Summary

> **Note:** Due to runtime constraints, this ablation was run over CatBoost only, not all top 5 models.

- Does full_advanced meaningfully beat original_only? `No` (-0.000688, negligible)
- Does full_advanced_without_ratios beat full_advanced? `Yes` (0.001660, marginal)
- Do ratios_all improve over original_only? `No` (-0.002690, negligible)
- Do interactions_all improve over original_only? `No` (-0.001196, negligible)
- Does removing sensitive columns cause a large drop? `Yes` (-0.026251)
- Are generated features essential or only marginal? `Marginal`

## Final D6 decision
`promote`

Generated features provide mixed and mostly marginal gains. Original-only remains highly competitive. Ratio features are not clearly essential. Interaction features are inconsistent. Sensitive/proxy variables are important for performance, so fairness review is mandatory.
