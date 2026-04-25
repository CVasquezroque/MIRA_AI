# Housing Status Fairness Audit

This audit compares candidate fraud models trained **with** and **without**
`housing_status`. It then evaluates false positive and false negative behavior
by the original `housing_status` groups.

This is a diagnostic step before any deployment recommendation. It does not make
causal claims about housing status.

## Setup

- Sensitive/proxy variable audited: `housing_status`
- Chronological split: train months [0, 1, 2, 3, 4, 5], validation month 6, test month 7
- Training sample rows: 250,000
- Removed constant columns: ['device_fraud_count']
- Threshold policy used for the main audit: choose one global threshold on validation with FPR <= 5%, then apply that threshold to test.

## Overall Test Metrics at Validation-Selected 5% FPR Threshold

| model_family  | feature_policy         | threshold | precision | recall_tpr | fpr      | pr_auc   | roc_auc  | tp  | fp   | tn    | fn  |
| ------------- | ---------------------- | --------- | --------- | ---------- | -------- | -------- | -------- | --- | ---- | ----- | --- |
| CatBoost      | with_housing_status    | 0.758477  | 0.160116  | 0.502101   | 0.039417 | 0.205903 | 0.889644 | 717 | 3761 | 91654 | 711 |
| CatBoost      | without_housing_status | 0.759002  | 0.159438  | 0.460784   | 0.036357 | 0.192457 | 0.881311 | 658 | 3469 | 91946 | 770 |
| XGBoost focal | with_housing_status    | 0.311310  | 0.158981  | 0.528711   | 0.041859 | 0.207997 | 0.893693 | 755 | 3994 | 91421 | 673 |
| XGBoost focal | without_housing_status | 0.309461  | 0.154982  | 0.470588   | 0.038401 | 0.192449 | 0.882305 | 672 | 3664 | 91751 | 756 |

## Overall Change When Housing Status Is Included

Positive deltas mean the model trained with `housing_status` scored higher than
the model trained without it.

| model_family  | precision_delta_with_minus_without | recall_tpr_delta_with_minus_without | fpr_delta_with_minus_without | pr_auc_delta_with_minus_without | roc_auc_delta_with_minus_without |
| ------------- | ---------------------------------- | ----------------------------------- | ---------------------------- | ------------------------------- | -------------------------------- |
| CatBoost      | 0.000678                           | 0.041317                            | 0.003060                     | 0.013446                        | 0.008334                         |
| XGBoost focal | 0.003999                           | 0.058123                            | 0.003459                     | 0.015548                        | 0.011388                         |

## Largest Group-Level Shifts on Test

These rows show where including `housing_status` changed group-level FPR/FNR the
most. Small groups are flagged in the CSV outputs because their rates are noisy.

| model_family  | housing_status | n_with_housing | fraud_count_with_housing | fpr_with_housing | fpr_without_housing | fpr_delta_with_minus_without | fnr_with_housing | fnr_without_housing | fnr_delta_with_minus_without | small_group_warning_with_housing |
| ------------- | -------------- | -------------- | ------------------------ | ---------------- | ------------------- | ---------------------------- | ---------------- | ------------------- | ---------------------------- | -------------------------------- |
| CatBoost      | BA             | 19182          | 899                      | 0.158836         | 0.104578            | 0.054258                     | 0.359288         | 0.456062            | -0.096774                    | False                            |
| CatBoost      | BB             | 23213          | 166                      | 0.013971         | 0.026381            | -0.012409                    | 0.704819         | 0.614458            | 0.090361                     | False                            |
| CatBoost      | BE             | 12327          | 54                       | 0.003748         | 0.015563            | -0.011815                    | 0.870370         | 0.870370            | 0.000000                     | False                            |
| XGBoost focal | BA             | 19182          | 899                      | 0.186731         | 0.109719            | 0.077011                     | 0.293660         | 0.451613            | -0.157953                    | False                            |
| XGBoost focal | BB             | 23213          | 166                      | 0.008895         | 0.028941            | -0.020046                    | 0.740964         | 0.620482            | 0.120482                     | False                            |
| XGBoost focal | BE             | 12327          | 54                       | 0.002852         | 0.016622            | -0.013770                    | 0.925926         | 0.796296            | 0.129630                     | False                            |

## How To Read The Audit

- FPR answers: among legitimate customers in this group, how many were incorrectly flagged?
- FNR answers: among fraud cases in this group, how many were missed?
- Alert rate answers: how often this group is sent to review/rejection at the chosen threshold?
- Small groups such as rare housing codes can have unstable rates; avoid overinterpreting them.

## Recommendation Before Deployment

Do not recommend deployment from model quality metrics alone. Compare:

- business lift from keeping `housing_status`,
- group-level FPR/FNR gaps,
- whether similar performance can be achieved without `housing_status`,
- whether any proxy variables recreate the same disparities even after removal.

The next review should include a domain/legal fairness review and threshold
selection based on acceptable false positive cost and customer impact.
