# Holistic V2 Focused Fraud Experiment

`model/holistic_v2` is an isolated follow-up experiment to the completed
`model/holistic` pipeline. It does not modify or overwrite historical holistic
outputs.

The experiment focuses on four strategy families:

1. weighted score blending with CatBoost included;
2. time-aware blending / temporal stacking;
3. hard negative mining to reduce false positives;
4. tuned focal-loss XGBoost.

All generated artifacts are written to:

```text
model/holistic_v2/results/
model/holistic_v2/results/figures/
```

## Checkpoints

Run one checkpoint at a time:

```bash
python model/holistic_v2/holistic_v2_experiment.py --checkpoint D0
```

The script may also support `--run-all`, but each decision checkpoint is meant to
be reviewed and committed separately.

## Split Discipline

- Train: months `0-5`
- Validation: month `6`
- Test: month `7`

Validation is used for decisions, thresholds, weights, hyperparameters, feature
sets, and candidate promotion. Test is reserved for final evaluation.
