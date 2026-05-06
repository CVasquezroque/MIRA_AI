from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    mode: str
    tuning_train_rows: int | None
    tuning_valid_rows: int | None
    train_rows: int | None
    eval_rows: int | None
    baseline_n_iter: int
    catboost_n_iter: int
    top_n_baseline_to_balance: int
    top_n_to_advanced: int
    top_n_to_anomaly: int
    fairness_top_n: int
    shap_top_n: int
    shap_sample_rows: int
    anomaly_legit_rows: int
    lof_legit_rows: int
    autoencoder_legit_rows: int
    include_expensive_ensembles: bool
    cascade_top_risk_pct: float
    cascade_final_top_pct: float
    riff_max_depth: int
    riff_min_leaf: int
    imbalance_ensemble_estimators: int
    hyperparameter_tuning_trials: int


def config_for_mode(mode: str) -> RunConfig:
    if mode == "smoke":
        return RunConfig(
            mode=mode,
            tuning_train_rows=2_500,
            tuning_valid_rows=1_000,
            train_rows=6_000,
            eval_rows=2_500,
            baseline_n_iter=1,
            catboost_n_iter=1,
            top_n_baseline_to_balance=2,
            top_n_to_advanced=2,
            top_n_to_anomaly=1,
            fairness_top_n=1,
            shap_top_n=1,
            shap_sample_rows=100,
            anomaly_legit_rows=1_000,
            lof_legit_rows=800,
            autoencoder_legit_rows=800,
            include_expensive_ensembles=False,
            cascade_top_risk_pct=0.05,
            cascade_final_top_pct=0.01,
            riff_max_depth=4,
            riff_min_leaf=25,
            imbalance_ensemble_estimators=5,
            hyperparameter_tuning_trials=2,
        )
    if mode == "full":
        return RunConfig(
            mode=mode,
            tuning_train_rows=None,
            tuning_valid_rows=None,
            train_rows=None,
            eval_rows=None,
            baseline_n_iter=6,
            catboost_n_iter=5,
            top_n_baseline_to_balance=10,
            top_n_to_advanced=10,
            top_n_to_anomaly=6,
            fairness_top_n=10,
            shap_top_n=3,
            shap_sample_rows=1_500,
            anomaly_legit_rows=25_000,
            lof_legit_rows=15_000,
            autoencoder_legit_rows=18_000,
            include_expensive_ensembles=True,
            cascade_top_risk_pct=0.05,
            cascade_final_top_pct=0.01,
            riff_max_depth=5,
            riff_min_leaf=120,
            imbalance_ensemble_estimators=12,
            hyperparameter_tuning_trials=20,
        )
    raise ValueError(f"Unknown mode: {mode}")


def parse_row_limit(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"all", "none", "null", "full"}:
        return None
    parsed = int(normalized)
    if parsed <= 0:
        return None
    return parsed


def apply_cli_overrides(config: RunConfig, args) -> RunConfig:
    if args.train_rows is not None:
        config.train_rows = parse_row_limit(args.train_rows)
    if args.tuning_train_rows is not None:
        config.tuning_train_rows = parse_row_limit(args.tuning_train_rows)
    if args.tuning_valid_rows is not None:
        config.tuning_valid_rows = parse_row_limit(args.tuning_valid_rows)
    if args.include_expensive_ensembles:
        config.include_expensive_ensembles = True
    if getattr(args, "tuning_trials", None) is not None:
        config.hyperparameter_tuning_trials = int(args.tuning_trials)
    return config
