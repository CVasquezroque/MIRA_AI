# Full Holistic Modular Pipeline

This workspace is autonomous. It does not import or execute the old monolithic workspace.

Stages are run through `run_stage.py`; each stage writes only its own folder
inside `results_full_train`, and `--force` clears only that stage folder.

## Core Stages

- `data-audit`: loads `data_banca/Base.csv`, uses temporal train/validation/test split, and saves `context.joblib`.
- `baseline-search`: trains autonomous baseline models and saves candidates, scores, specs, and selected joblib artifacts.
- `balance-gate`: trains class-weight/resampling variants for top baseline families.
- `advanced-features-gate`: trains autonomous advanced-feature pipelines.
- `anomaly-recency-gate`: tests recency weighting/recent-only training and optional IsolationForest score.
- `imbalance-ensemble-gate`: tests class-imbalance ensembles, preferring official `imbalanced-ensemble` implementations when available.
- `hyperparameter-tuning-gate`: tunes CatBoost, XGBoost, LightGBM, and Logistic Regression with Optuna while keeping the fixed-parameter stages as reproducible reference baselines. The inner tuning split uses train months except the last train month for fitting and the last train month for validation; outer validation and test remain untouched for tuning.
- `cascade-filter`: trains a stage-1 ranker and a temporal out-of-fold top-risk TP/FP filter.
- `riff-rules`: extracts RIFF-style low-FPR rules from tree leaves.
- `operational-thresholds`: selects thresholds on validation, including low-FPR sweep, and applies them to test.
- `topk`: computes ranked alert metrics.
- `final-report`: combines available artifacts and marks skipped optional analyses as `This analysis was not run.`

## Leakage Discipline

Thresholds, rules, cascade filters, and candidate choices use train or validation
only. Test is used only for final evaluation.

## Hyperparameter Tuning

Run the tuning gate independently:

```bash
python model/full_holistic/run_stage.py --mode full --stage hyperparameter-tuning-gate --tuning-trials 20
```

Use `--tuning-trials` to control Optuna trials per family. The primary selection
objective is aligned to operational fraud review: inner validation recall at
FPR<=5%, precision/recall at top 1%, PR-AUC, FDR, and fairness/FPR guardrails.
In `--mode full`, tuning uses the full inner temporal train and validation
splits unless row caps are explicitly provided.
