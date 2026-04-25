# Advanced Feature and Modeling Experiments

This iteration extends the previous EDA-driven pipeline. It does not replace the earlier
baseline/tuning results.

## Why Strategy 1: Ratio Features

The ratio plots in `figures/strategy_1_ratio_fraud_rates.png` compare fraud rates across
training-month quantiles. They are meant to check whether relative behavior has signal,
not to claim causality. The most useful candidates are ratios between short-window and
long-window activity, and ratios that normalize local concentration signals.

Examples tested:

- `velocity_6h / velocity_24h`
- `velocity_24h / velocity_4w`
- `zip_count_4w / velocity_4w`
- `date_of_birth_distinct_emails_4w / zip_count_4w`
- `bank_branch_count_8w / zip_count_4w`
- `proposed_credit_limit / income`

## Why Strategy 2: Interaction Features

The heatmaps in `figures/strategy_2_interaction_heatmaps.png` show fraud-rate variation
for pairs of features that are plausible in account-opening fraud: device/channel,
email/name consistency, phone validation, session behavior, and requested credit limit.
These are association checks from EDA, not causal claims.

Examples tested:

- `device_os x source`
- `email_is_free x source`
- `phone_home_valid x phone_mobile_valid`
- `payment_type x proposed_credit_limit_bin`
- `name_email_similarity_bin x email_is_free`
- `session_length_bin x source`

## Strategy 3: More Robust Target/Frequency Encoding

The new encoder adds two columns per categorical feature:

- a smoothed historical fraud-rate encoding,
- a category-frequency encoding.

To reduce leakage, training rows are encoded month-by-month using only earlier months
when `month` is available. Validation and test use mappings fitted on the training period.
This is stricter than fitting target encoding once on all training rows and then letting
each row carry information from its own label.

## Standard Objective vs Focal-Style Objective

Most baseline models use a standard binary log-loss objective. In plain language, the
model is trained to assign good probabilities overall. Class weights or
`scale_pos_weight` can make fraud mistakes more expensive, but the loss is still the
standard log-loss shape.

The focal-style XGBoost objective changes that training pressure: easier examples receive
less weight, while harder examples receive more attention. This can improve minority-class
capture, but it can also hurt calibration or increase false positives, so it must be
compared on validation/test metrics instead of assumed better.

## Run Setup

- Training sample: 250,000 rows from chronological training months.
- Training-sample fraud rate: 1.0252%.
- Validation and test months were not resampled.
- Step 7 anomaly scores and Step 8 recency weighting were intentionally left for the next iteration.

## Best Results

Best validation PR-AUC:

- Model: `CatBoost native categoricals + advanced features + standard logloss`
- PR-AUC: 0.176375
- ROC-AUC: 0.886974
- Precision @ 0.50: 0.060973
- Recall @ 0.50: 0.767586
- FPR @ 0.50: 0.160620

Best test PR-AUC:

- Model: `XGBoost target/frequency + advanced features + focal-style loss`
- PR-AUC: 0.207997
- ROC-AUC: 0.893693
- Precision @ 0.50: 0.428977
- Recall @ 0.50: 0.105742
- FPR @ 0.50: 0.002107

## Explainability

The script saves:

- `shap_top_features.csv` and `figures/shap_top_features.png` when SHAP supports the best tree pipeline.
- `permutation_importance_raw_features.csv` and `figures/permutation_importance_raw_features.png` as a model-agnostic check.

SHAP is useful for global contribution patterns and local explanations. Permutation
importance is easier to explain to non-technical stakeholders because it measures how
much validation PR-AUC drops when a raw feature is shuffled.

Important fairness note: if socioeconomic variables such as `housing_status` or
`employment_status` appear important, that is not automatically a reason to keep or
remove them. It is a reason to run a comparison with and without those fields and audit
group-level false positive/false negative behavior before recommending deployment.
