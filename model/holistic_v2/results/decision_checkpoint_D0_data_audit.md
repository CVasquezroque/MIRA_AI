# Decision Checkpoint D0 - Data Audit

## Checkpoint Name

D0 data audit

## Purpose

Validate that the dataset, fixed chronological split, target, leakage checks,
prevalence, and protected attributes are suitable for the focused V2 experiment.

## Candidates Or Options Evaluated

- Continue with fixed temporal split months 0-5 / 6 / 7.
- Exclude suspicious leakage columns before modeling.
- Retain protected/proxy candidate columns for modeling only with later fairness review.

## Validation Metrics Used

No model validation metrics are used in D0. The decision uses data validity checks:
split completeness, fraud prevalence, class imbalance, leakage-name scan, missing
and sentinel summaries, duplicate count, and monthly fraud drift.

## Decision Made

`promote`

## Promoted Candidates

- Fixed chronological split: train months `[0, 1, 2, 3, 4, 5]`, validation month `6`, test month `7`.
- Protected/proxy candidates for later audit: `['housing_status', 'employment_status', 'customer_age', 'income']`.

## Discarded Candidates

- Suspicious leakage feature columns: `['device_fraud_count']`.

## Skipped Candidates

- `skip for runtime`: no model families are trained in D0.

## Reason For The Decision

The required target and month split are present, all split partitions are non-empty,
and the class imbalance is measurable across train, validation, and test. The
experiment can continue while preserving strict temporal validation.

## Risks Or Limitations

- Potential leakage columns excluded from future stages: ['device_fraud_count'].
- Monthly prevalence drift is present; temporal validation remains mandatory.
- Protected/proxy columns require fairness review before any deployment recommendation.

## Next Step

Run D1 improved baselines and select the top five candidates using validation-only
multi-objective ranking.
