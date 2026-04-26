# Holistic V2 D0 Data Audit

## Scope

This audit validates the fixed temporal split and checks whether the dataset is
safe enough to continue with focused V2 experiments.

## Dataset

- Dataset path: `C:\Users\LEGION CARLOS\Desktop\Home\BREIT\MIRA_AI\data_banca\Base.csv`
- Rows: `1,000,000`
- Columns: `32`
- Target column: `fraud_bool`
- Month column: `month`
- Train months: `[0, 1, 2, 3, 4, 5]`
- Validation month: `6`
- Test month: `7`
- Duplicate rows: `0`

## Split And Prevalence

| split      | rows   | fraud_count | legitimate_count | fraud_prevalence | imbalance_ratio_legit_to_fraud |
| ---------- | ------ | ----------- | ---------------- | ---------------- | ------------------------------ |
| train      | 794989 | 8151        | 786838           | 0.010253         | 96.532695                      |
| validation | 108168 | 1450        | 106718           | 0.013405         | 73.598621                      |
| test       | 96843  | 1428        | 95415            | 0.014746         | 66.817227                      |

## Monthly Fraud Drift

| month | rows   | fraud_count | fraud_prevalence | legitimate_count | fraud_prevalence_delta_vs_previous |
| ----- | ------ | ----------- | ---------------- | ---------------- | ---------------------------------- |
| 0     | 132440 | 1500        | 0.011326         | 130940           |                                    |
| 1     | 127620 | 1198        | 0.009387         | 126422           | -0.001939                          |
| 2     | 136979 | 1198        | 0.008746         | 135781           | -0.000641                          |
| 3     | 150936 | 1392        | 0.009222         | 149544           | 0.000477                           |
| 4     | 127691 | 1452        | 0.011371         | 126239           | 0.002149                           |
| 5     | 119323 | 1411        | 0.011825         | 117912           | 0.000454                           |
| 6     | 108168 | 1450        | 0.013405         | 106718           | 0.001580                           |
| 7     | 96843  | 1428        | 0.014746         | 95415            | 0.001340                           |

Drift detected: `yes`. Maximum monthly
prevalence range is `0.006000`.

## Missing Values

_No rows available._

## Sentinel Values

| column                       | sentinel_value | count  | rate     |
| ---------------------------- | -------------- | ------ | -------- |
| prev_address_months_count    | -1             | 712920 | 0.712920 |
| bank_months_count            | -1             | 253635 | 0.253635 |
| current_address_months_count | -1             | 4254   | 0.004254 |
| session_length_in_minutes    | -1             | 2015   | 0.002015 |
| credit_risk_score            | -1             | 488    | 0.000488 |
| device_distinct_emails_8w    | -1             | 359    | 0.000359 |
| credit_risk_score            | -9             | 268    | 0.000268 |
| credit_risk_score            | -99            | 42     | 0.000042 |

## Constant Or Near-Constant Columns

| column             | nunique_including_na | top_value_rate | reason   |
| ------------------ | -------------------- | -------------- | -------- |
| device_fraud_count | 1                    | 1.000000       | constant |

## Suspicious Leakage Columns

| column             | matched_tokens | leakage_action | reason                                               |
| ------------------ | -------------- | -------------- | ---------------------------------------------------- |
| device_fraud_count | fraud          | exclude        | name suggests target, outcome, or future information |

Columns marked `exclude` should be removed from modeling features unless a later
checkpoint documents a stronger non-leakage justification.

## Protected / Sensitive Candidate Columns

| column            | present | nunique | missing_rate | notes                                                 |
| ----------------- | ------- | ------- | ------------ | ----------------------------------------------------- |
| housing_status    | True    | 7       | 0.000000     | protected or proxy candidate; audit before deployment |
| employment_status | True    | 7       | 0.000000     | protected or proxy candidate; audit before deployment |
| customer_age      | True    | 9       | 0.000000     | protected or proxy candidate; audit before deployment |
| income            | True    | 9       | 0.000000     | protected or proxy candidate; audit before deployment |

## D0 Conclusion

- Split valid enough to continue: `yes`
- Leakage exclusions for future checkpoints: `['device_fraud_count']`
- Temporal drift note: `keep chronological split and monitor recency behavior`
