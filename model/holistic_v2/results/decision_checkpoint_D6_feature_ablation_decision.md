# Decision Checkpoint D6 - Feature Ablation

## Checkpoint Name

D6 top five feature ablations

## Purpose

Run individual feature ablations over CatBoost to determine which feature groups
contribute meaningfully.

## Candidates Evaluated

| feature_set                        | validation_pr_auc | validation_recall_at_fpr5 | validation_precision_at_fpr5 | validation_fdr_at_fpr5 | validation_precision_top_1pct | pr_auc_delta_vs_full | top1_delta_vs_full |
| ---------------------------------- | ----------------- | ------------------------- | ---------------------------- | ---------------------- | ----------------------------- | -------------------- | ------------------ |
| full_advanced                      | 0.170692          | 0.512414                  | 0.122466                     | 0.877534               | 0.263401                      | 0.000000             | 0.000000           |
| original_only                      | 0.171380          | 0.513103                  | 0.122530                     | 0.877470               | 0.267098                      | 0.000688             | 0.003697           |
| original_without_sensitive         | 0.138512          | 0.459310                  | 0.111056                     | 0.888944               | 0.228281                      | -0.032179            | -0.035120          |
| missing_flags_only                 | 0.170172          | 0.498621                  | 0.119524                     | 0.880476               | 0.263401                      | -0.000519            | 0.000000           |
| outlier_flags_only                 | 0.171430          | 0.513103                  | 0.122894                     | 0.877106               | 0.254159                      | 0.000739             | -0.009242          |
| log_features_only                  | 0.169380          | 0.511724                  | 0.122949                     | 0.877051               | 0.258780                      | -0.001311            | -0.004621          |
| ratios_only                        | 0.168690          | 0.496552                  | 0.120724                     | 0.879276               | 0.268022                      | -0.002002            | 0.004621           |
| interactions_only                  | 0.170184          | 0.514483                  | 0.122677                     | 0.877323               | 0.260628                      | -0.000507            | -0.002773          |
| full_advanced_without_ratios       | 0.172351          | 0.518621                  | 0.123868                     | 0.876132               | 0.268022                      | 0.001660             | 0.004621           |
| full_advanced_without_interactions | 0.169454          | 0.506207                  | 0.121002                     | 0.878998               | 0.252311                      | -0.001238            | -0.011091          |
| full_advanced_without_sensitive    | 0.144441          | 0.472414                  | 0.114357                     | 0.885643               | 0.231978                      | -0.026251            | -0.031423          |

## Key Findings

- Best feature set: `full_advanced_without_ratios` (PR-AUC: 0.172351)
- Worst feature set: `original_without_sensitive` (PR-AUC: 0.138512)
- Full advanced PR-AUC: `0.170692`
- Removing sensitive columns impact: `-0.026251` PR-AUC
- Removing ratios impact: `0.001660` PR-AUC
- Removing interactions impact: `-0.001238` PR-AUC

## Decision Made

`promote`

## Reason

Feature ablation identifies which features contribute and which can be safely removed.
Full advanced or full advanced without sensitive should be promoted as the production
feature set depending on fairness review in D7.

## Next Step

Run D7 fairness audit for finalists.
