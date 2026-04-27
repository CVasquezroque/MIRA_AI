# Decision Checkpoint D5 - Tuned Focal-Loss XGBoost

## Checkpoint Name

D5 tuned focal loss

## Purpose

Evaluate a controlled grid of focal-loss XGBoost (alpha x gamma) against standard
logloss XGBoost baselines.

## Grid Evaluated

- alpha: [0.25, 0.5, 0.75, 0.9]
- gamma: [1.0, 2.0, 3.0, 4.0]
- Total focal candidates: 16

## Candidates Or Options Evaluated

| readable_model_name                                                                                                                | validation_pr_auc | validation_recall_at_fpr5 | validation_precision_at_fpr5 | validation_fdr_at_fpr5 | validation_precision_top_1pct |
| ---------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------- | ---------------------------- | ---------------------- | ----------------------------- |
| XGBoost | rep=target_frequency | feat=full_advanced | balance=scale_pos_weight | loss=logloss | train=months_0_5 | ensemble=none   | 0.169369          | 0.519310                  | 0.123828                     | 0.876172               | 0.258780                      |
| XGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=logloss | train=months_0_5 | ensemble=none               | 0.168792          | 0.502759                  | 0.120456                     | 0.879544               | 0.261553                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.25_g1.0 | train=months_0_5 | ensemble=none | 0.162359          | 0.503448                  | 0.120402                     | 0.879598               | 0.268022                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.25_g2.0 | train=months_0_5 | ensemble=none | 0.159149          | 0.502069                  | 0.120470                     | 0.879530               | 0.260628                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.25_g3.0 | train=months_0_5 | ensemble=none | 0.158240          | 0.498621                  | 0.119861                     | 0.880139               | 0.261553                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.25_g4.0 | train=months_0_5 | ensemble=none | 0.154282          | 0.495862                  | 0.119396                     | 0.880604               | 0.255083                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.5_g1.0 | train=months_0_5 | ensemble=none  | 0.166958          | 0.504828                  | 0.121152                     | 0.878848               | 0.264325                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.5_g2.0 | train=months_0_5 | ensemble=none  | 0.164211          | 0.508966                  | 0.121722                     | 0.878278               | 0.265250                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.5_g3.0 | train=months_0_5 | ensemble=none  | 0.161054          | 0.505517                  | 0.120877                     | 0.879123               | 0.266174                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.5_g4.0 | train=months_0_5 | ensemble=none  | 0.158038          | 0.506207                  | 0.121563                     | 0.878437               | 0.264325                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.75_g1.0 | train=months_0_5 | ensemble=none | 0.165861          | 0.509655                  | 0.121686                     | 0.878314               | 0.262477                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.75_g2.0 | train=months_0_5 | ensemble=none | 0.166542          | 0.507586                  | 0.121332                     | 0.878668               | 0.268022                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.75_g3.0 | train=months_0_5 | ensemble=none | 0.163253          | 0.505517                  | 0.121438                     | 0.878562               | 0.260628                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.75_g4.0 | train=months_0_5 | ensemble=none | 0.158364          | 0.499310                  | 0.119868                     | 0.880132               | 0.261553                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.9_g1.0 | train=months_0_5 | ensemble=none  | 0.170751          | 0.509655                  | 0.122068                     | 0.877932               | 0.256932                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.9_g2.0 | train=months_0_5 | ensemble=none  | 0.168704          | 0.508276                  | 0.121617                     | 0.878383               | 0.261553                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.9_g3.0 | train=months_0_5 | ensemble=none  | 0.162249          | 0.504828                  | 0.121292                     | 0.878708               | 0.260628                      |
| FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.9_g4.0 | train=months_0_5 | ensemble=none  | 0.154740          | 0.503448                  | 0.121041                     | 0.878959               | 0.254159                      |

## Best Focal Candidate

`FocalXGBoost | rep=target_frequency | feat=full_advanced | balance=none | loss=focal_a0.75_g2.0 | train=months_0_5 | ensemble=none`

- PR-AUC delta vs best standard: `-0.002827`
- FDR delta: `0.002496`
- Recall delta: `-0.011724`
- Precision@Top 1% delta: `0.009242`

## Decision Made

`promote`

## Reason For The Decision

Focal-loss XGBoost improved at least one operational metric versus standard logloss.

## Next Step

Run D6 feature ablations over the top 5 baseline candidates.
