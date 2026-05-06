# Data Audit

The modular pipeline uses a chronological split and keeps test untouched for final evaluation.

- Train months: `[0, 1, 2, 3, 4, 5]`
- Validation month: `6`
- Test month: `7`
- Train rows: `794,989`
- Validation rows: `108,168`
- Test rows: `96,843`
- Train prevalence: `0.010253`
- Validation prevalence: `0.013405`
- Test prevalence: `0.014746`
- Scale positive weight from train sample: `96.532695`

## Monthly Drift

| month | rows   | frauds | fraud_prevalence | split      |
| ----- | ------ | ------ | ---------------- | ---------- |
| 0     | 132440 | 1500   | 0.011326         | train      |
| 1     | 127620 | 1198   | 0.009387         | train      |
| 2     | 136979 | 1198   | 0.008746         | train      |
| 3     | 150936 | 1392   | 0.009222         | train      |
| 4     | 127691 | 1452   | 0.011371         | train      |
| 5     | 119323 | 1411   | 0.011825         | train      |
| 6     | 108168 | 1450   | 0.013405         | validation |
| 7     | 96843  | 1428   | 0.014746         | test       |

## Figures

![Fraud prevalence by month](C:/Users/LEGION CARLOS/Desktop/Home/BREIT/MIRA_AI/model/full_holistic/results_full_train/00_data_audit/figures/01_fraud_prevalence_by_month.png)
