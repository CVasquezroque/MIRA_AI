# RIFF-Style Low-FPR Rules Summary

This is RIFF-inspired, not a full reproduction of a specific paper. It trains a class-weighted tree on induction months, extracts high-fraud leaves as readable rules, evaluates them on a separate train selection month, and greedily selects rules subject to low-FPR caps before final validation/test evaluation.

## Test Metrics

| fpr_cap  | threshold | alerts | alert_count | tp  | fp   | fn   | tn    | precision | fdr      | recall_tpr | fpr      | fnr      | tnr      | specificity_tnr | alert_rate | fraud_prevalence | lift     | precision_lift | fp_per_tp | pr_auc   | roc_auc  | pr_auc_lift | n_obs |
| -------- | --------- | ------ | ----------- | --- | ---- | ---- | ----- | --------- | -------- | ---------- | -------- | -------- | -------- | --------------- | ---------- | ---------------- | -------- | -------------- | --------- | -------- | -------- | ----------- | ----- |
| 0.002500 | 0.500000  | 195    | 195         | 7   | 188  | 1421 | 95227 | 0.035897  | 0.964103 | 0.004902   | 0.001970 | 0.995098 | 0.998030 | 0.998030        | 0.002014   | 0.014746         | 2.434465 | 2.434465       | 26.857143 | 0.014849 | 0.501466 | 1.007032    | 96843 |
| 0.005000 | 0.500000  | 663    | 663         | 12  | 651  | 1416 | 94764 | 0.018100  | 0.981900 | 0.008403   | 0.006823 | 0.991597 | 0.993177 | 0.993177        | 0.006846   | 0.014746         | 1.227461 | 1.227461       | 54.250000 | 0.014774 | 0.500790 | 1.001911    | 96843 |
| 0.010000 | 0.500000  | 740    | 740         | 23  | 717  | 1405 | 94698 | 0.031081  | 0.968919 | 0.016106   | 0.007515 | 0.983894 | 0.992485 | 0.992485        | 0.007641   | 0.014746         | 2.107833 | 2.107833       | 31.173913 | 0.015009 | 0.504296 | 1.017843    | 96843 |
| 0.020000 | 0.500000  | 2134   | 2134        | 107 | 2027 | 1321 | 93388 | 0.050141  | 0.949859 | 0.074930   | 0.021244 | 0.925070 | 0.978756 | 0.978756        | 0.022036   | 0.014746         | 3.400395 | 3.400395       | 18.943925 | 0.017398 | 0.526843 | 1.179862    | 96843 |
