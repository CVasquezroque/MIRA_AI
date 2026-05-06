RANDOM_STATE = 42
TARGET = "fraud_bool"
SENSITIVE_COLUMN = "housing_status"
MAIN_FPR_CAP = 0.05
BUSINESS_MAX_FDR = 0.30
BUSINESS_MIN_PRECISION = 1.0 - BUSINESS_MAX_FDR
COST_RATIOS = [(5, 1), (10, 1), (25, 1), (50, 1)]
LOW_FPR_CAPS = [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]
TOPK_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.10]
PROTECTED_COLUMNS = {"housing_status", "employment_status", "customer_age", "income"}
ANOMALY_COLUMNS = [
    "isolation_forest_anomaly_score",
    "lof_anomaly_score",
    "autoencoder_reconstruction_error",
]
