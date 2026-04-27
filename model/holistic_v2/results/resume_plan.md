# Holistic V2 Resume Plan

## Inspection Date

2026-04-26T23:55 CDT

## Git Log (model/holistic_v2)

```
8005218 feat: add holistic v2 weighted blending       (D2 committed)
90d7d5c feat: add holistic v2 improved baselines      (D1 committed)
b563331 feat: initialize holistic v2 data audit       (D0 committed)
```

## Checkpoint Status

| Checkpoint | Commit | Outputs Present | Decision File | Real Conclusions | Status |
|---|---|---|---|---|---|
| D0 data audit | b563331 ✅ | ✅ all present | ✅ decision_checkpoint_D0_data_audit.md | ✅ real | **Complete** |
| D1 improved baselines | 90d7d5c ✅ | ✅ all present | ✅ decision_checkpoint_D1_improved_baseline_top5.md | ✅ real | **Complete** |
| D2 weighted blending | 8005218 ✅ | ✅ all present | ✅ decision_checkpoint_D2_weighted_blend_decision.md | ✅ real | **Complete** |
| D3 temporal blending | ❌ no commit | ✅ all present (untracked) | ✅ decision_checkpoint_D3_temporal_blend_decision.md | ✅ real (keep as benchmark) | **Outputs exist, needs commit** |
| D4 hard negative mining | ❌ | ❌ | ❌ | ❌ | **Missing** |
| D5 focal loss | ❌ | ❌ | ❌ | ❌ | **Missing** |
| D6 feature ablation | ❌ | ❌ | ❌ | ❌ | **Missing** |
| D7 fairness | ❌ | ❌ | ❌ | ❌ | **Missing** |
| D8 final strategy | ❌ | ❌ | ❌ | ❌ | **Missing** |

## Uncommitted Work

### Modified (tracked):
- `model/holistic_v2/holistic_v2_experiment.py` — 396 lines added (D3 run_d3() code)
- `model/holistic_v2/results/holistic_v2_run_log.md` — D3 log entries appended

### Untracked:
- `model/holistic_v2/results/04_temporal_blending_oof_scores.csv` (~51 MB — large, should be gitignored)
- `model/holistic_v2/results/04_temporal_blending_results.csv`
- `model/holistic_v2/results/04_temporal_blending_summary.md`
- `model/holistic_v2/results/04_temporal_blending_test_scores.csv` (~6 MB)
- `model/holistic_v2/results/04_temporal_blending_validation_scores.csv` (~7 MB)
- `model/holistic_v2/results/decision_checkpoint_D3_temporal_blend_decision.md`
- `model/holistic_v2/results/figures/04_temporal_blending_comparison.png`

## D3 Decision Summary

The D3 temporal blend decision was `keep as benchmark`. The temporal uniform rank blend
achieved PR-AUC 0.175 and Precision@Top1% 0.271, but did not clearly beat the simpler
D2 weighted blend. Real metrics and real conclusions are present.

## Script Checkpoint Support

The committed script (HEAD) supports: `--checkpoint D0, D1, D2, --run-all`
The working copy adds: `--checkpoint D3`
Checkpoints D4–D8 are not yet implemented in the script.

## Files Already Generated (in results/)

### Committed:
- 00_data_audit_metadata.json, 00_data_audit_report.md, 00_monthly_fraud_rate_drift.csv
- 01_improved_baseline_candidates.csv, 01_improved_baseline_specs.json
- 01_improved_baseline_test_scores.csv, 01_improved_baseline_validation_scores.csv
- 02_top5_selected_from_improved_baseline.csv
- 03_weighted_blend_results.csv, 03_weighted_blend_summary.md
- 03_weighted_blend_test_scores.csv, 03_weighted_blend_validation_scores.csv, 03_weighted_blend_weights.csv
- decision_checkpoint_D0, D1, D2 .md files
- figures: 00_monthly_fraud_rate_drift.png, 01_*.png, 03_*.png

### Uncommitted (D3):
- 04_temporal_blending_*.csv, 04_temporal_blending_summary.md
- decision_checkpoint_D3_temporal_blend_decision.md
- figures/04_temporal_blending_comparison.png

## Files Missing

- D4: 05_hard_negative_*.csv, decision_checkpoint_D4_hard_negative_decision.md
- D5: 06_focal_loss_*.csv, decision_checkpoint_D5_focal_loss_decision.md
- D6: 07_feature_ablation_*.csv, decision_checkpoint_D6_feature_ablation_decision.md
- D7: 08_fairness_*.csv, decision_checkpoint_D7_fairness_decision.md
- D8: 09_final_strategy_*.csv, 09_final_strategy_report.md, decision_checkpoint_D8_final_strategy_decision.md

## Large Files Handling

The following score CSV files are very large and should be added to .gitignore:
- *_oof_scores.csv (51 MB)
- *_validation_scores.csv (7–27 MB)
- *_test_scores.csv (6–24 MB)

Note: these are already tracked for D1/D2 commits. Going forward, new large score files
should be gitignored and the .gitignore updated before D3 commit.

## Resume Plan

1. **First action**: Update .gitignore for large holistic_v2 score CSVs, then commit D3 work:
   ```
   git add model/holistic_v2
   git commit -m "feat: add holistic v2 temporal blending"
   ```

2. **Then implement and run D4** (hard negative mining)
3. **Then implement and run D5** (tuned focal loss)
4. **Then implement and run D6** (feature ablation)
5. **Then implement and run D7** (fairness audit)
6. **Then implement and run D8** (final strategy report)

## Exact Command to Resume

```bash
# Step 1: commit D3
git add model/holistic_v2
git commit -m "feat: add holistic v2 temporal blending"

# Step 2: implement D4, then run
python model/holistic_v2/holistic_v2_experiment.py --checkpoint D4
```

## Last Completed Checkpoint: D3 (outputs exist, not committed)
## Last Committed Checkpoint: D2 (8005218)
## Next Checkpoint to Run: D4 (hard negative mining)
