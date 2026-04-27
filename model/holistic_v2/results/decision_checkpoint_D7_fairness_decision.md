# Decision Checkpoint D7 - Fairness Audit

## Checkpoint Name

D7 finalist fairness audit

## Purpose

Evaluate group-level fairness metrics for the finalist CatBoost model across
housing_status, employment_status, age group, and income group.

## Fairness Metrics

| group_attribute    | group_value  | group_size | fraud_prevalence | alert_rate | precision | fdr      | tpr_recall | fpr      |
| ------------------ | ------------ | ---------- | ---------------- | ---------- | --------- | -------- | ---------- | -------- |
| housing_status     | BA           | 20718      | 0.042813         | 0.220195   | 0.134809  | 0.865191 | 0.693348   | 0.199032 |
| housing_status     | BB           | 33162      | 0.006694         | 0.019360   | 0.087227  | 0.912773 | 0.252252   | 0.017790 |
| housing_status     | BC           | 32762      | 0.006990         | 0.016788   | 0.096364  | 0.903636 | 0.231441   | 0.015277 |
| housing_status     | BD           | 3727       | 0.007513         | 0.051784   | 0.062176  | 0.937824 | 0.428571   | 0.048932 |
| housing_status     | BE           | 17599      | 0.004773         | 0.006364   | 0.062500  | 0.937500 | 0.083333   | 0.005995 |
| housing_status     | BF           | 176        | 0.000000         | 0.034091   | 0.000000  | 1.000000 |            | 0.034091 |
| housing_status     | BG           | 24         | 0.000000         | 0.083333   | 0.000000  | 1.000000 |            | 0.083333 |
| employment_status  | CA           | 83827      | 0.014530         | 0.063679   | 0.119895  | 0.880105 | 0.525452   | 0.056870 |
| employment_status  | CB           | 11563      | 0.008735         | 0.030788   | 0.101124  | 0.898876 | 0.356436   | 0.027918 |
| employment_status  | CC           | 4246       | 0.027791         | 0.072774   | 0.210356  | 0.789644 | 0.550847   | 0.059109 |
| employment_status  | CD           | 2004       | 0.000499         | 0.010978   | 0.045455  | 0.954545 | 1.000000   | 0.010484 |
| employment_status  | CE           | 2298       | 0.002611         | 0.009138   | 0.047619  | 0.952381 | 0.166667   | 0.008726 |
| employment_status  | CF           | 4187       | 0.001433         | 0.004777   | 0.000000  | 1.000000 | 0.000000   | 0.004784 |
| employment_status  | CG           | 43         | 0.000000         | 0.023256   | 0.000000  | 1.000000 |            | 0.023256 |
| customer_age_group | 26-35        | 30959      | 0.009981         | 0.033302   | 0.110572  | 0.889428 | 0.368932   | 0.029918 |
| customer_age_group | 36-45        | 28613      | 0.014259         | 0.069619   | 0.106928  | 0.893072 | 0.522059   | 0.063074 |
| customer_age_group | 46-55        | 15951      | 0.023447         | 0.118801   | 0.121372  | 0.878628 | 0.614973   | 0.106888 |
| customer_age_group | 56+          | 5161       | 0.038171         | 0.152877   | 0.181242  | 0.818758 | 0.725888   | 0.130137 |
| customer_age_group | <=25         | 27484      | 0.005894         | 0.013099   | 0.119444  | 0.880556 | 0.265432   | 0.011602 |
| income_group       | (0.099, 0.4] | 29350      | 0.007632         | 0.021397   | 0.114650  | 0.885350 | 0.321429   | 0.019089 |
| income_group       | (0.4, 0.7]   | 28862      | 0.009667         | 0.033677   | 0.110082  | 0.889918 | 0.383513   | 0.030263 |
| income_group       | (0.7, 0.9]   | 49956      | 0.018957         | 0.089419   | 0.126259  | 0.873741 | 0.595565   | 0.079638 |

## Summary

- Max FPR gap across any attribute: `0.193037`
- Max TPR gap across any attribute: `1.000000`
- Min Disparate Impact Ratio (alert): `0.028902`
- Requires further fairness review: `yes`

## Decision Made

`requires fairness review`

## Reason

Significant group-level disparities were found. The model should not be deployed without fairness mitigation.

## Next Step

Run D8 final strategy report.
