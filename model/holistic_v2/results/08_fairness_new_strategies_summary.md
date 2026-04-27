# Decision Checkpoint D7 - Fairness Audit Summary

> **Note:** Current fairness audit covers only the final CatBoost candidate. Fairness for other strategies remains pending.

- Which group has highest FPR? `housing_status: BA` (0.1990)
- Which group has lowest TPR? `employment_status: CF` (0.0000)
- Which group has highest alert rate? `housing_status: BA` (0.2202)
- Max FPR gap: `0.1930`
- Max TPR gap: `1.0000`
- DIR alert: `0.0289`
- Is fairness review required? `Yes`

## Final D7 decision
`requires fairness review`

Fairness review remains required. The strong performance drop when removing sensitive/proxy variables in D6 indicates that protected or proxy variables may be important drivers of model behavior.
