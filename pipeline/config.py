"""
Central config: column names, exam-type detection, category constants.
"""

# Raw CSV columns
COL_YEAR        = "Year"
COL_ROUND       = "Round"
COL_INSTITUTE   = "Institute"
COL_PROGRAM     = "Academic Program Name"
COL_QUOTA       = "Quota"
COL_SEAT_TYPE   = "Seat Type"
COL_GENDER      = "Gender"
COL_OPEN_RANK   = "Opening Rank"
COL_CLOSE_RANK  = "Closing Rank"

# Derived column added during loading
COL_EXAM_TYPE   = "Exam Type"   # "advanced" | "mains"

# Exam-type detection (matched against institute name, case-insensitive)
# IITs use JEE Advanced ranks; everything else uses JEE Mains ranks.
IIT_KEYWORDS = ["indian institute of technology", "iit "]

# Prediction target year (set to next year before running predict) 
PREDICT_YEAR = 2026

# Minimum historical data points to fit a trend model per combination
MIN_YEARS_FOR_TREND = 2

# Ensemble weight for combining year-trend vs round-ratio predictions
# 0.0 = pure round-ratio scaling; 1.0 = pure per-round year trend
# Tuned via grid search on JoSAA 2024: w=1.0 gives MAE 3,174 vs 3,708 at w=0.5.
# With trend_model="median", w=1.0 means "use per-round historical median directly"
# which is strictly better than going through the ratio estimation step.
ENSEMBLE_WEIGHT = 1.0

# Default trend model for the year-signal component
# Multi-year validation (2022-2025) shows GP RBF outperforms SVR RBF every year
# (avg MAE 2,667 vs 2,742) via Bayesian posterior mean reversion; no sklearn
# optimisation overhead: one np.linalg.solve per slot.
DEFAULT_TREND_MODEL = "gp_rbf"

# Rounds to predict 
JOSAA_ROUNDS = [1, 2, 3, 4, 5, 6]
CSAB_ROUNDS  = [1, 2]          # CSAB typically has 2 special rounds
ALL_ROUNDS   = JOSAA_ROUNDS    # default; overridden per source in CLI

# CSAB quota name normalisation (current-year page uses full names)
# Historical archive uses abbreviations; current-year page uses full strings.
# Map everything to the abbreviation so all years share the same key space.
CSAB_QUOTA_NORM: dict[str, str] = {
    "all india":            "AI",
    "home state":           "HS",
    "other state":          "OS",
    "home state for goa":   "GO",
    "jammu & kashmir (ut)": "JK",
    "jammu and kashmir":    "JK",
    "ladakh (ut)":          "LA",
    "ladakh":               "LA",
}

# Data source configs
import os

SOURCES = {
    "josaa": {
        "csv":        os.path.join(os.path.dirname(__file__), "..", "data", "josaa_ranks.csv"),
        "model":      "josaa_model.pkl",
        "round_col":  "Round",
        "rounds":     JOSAA_ROUNDS,
        "trend_model": "mlp_ensemble",  # GP-MLP ensemble (tuned arch) beats GP RBF avg 2,608 vs 2,667
        "mlp_hidden":  (512, 256, 128),
        "mlp_dropout": 0.15,
        "blend_alpha": 0.5,             # confirmed optimal on 2024 validation
        # safe ≤ 0.80xpred, match ≤ 1.00xpred, reach ≤ 1.20xpred
        "safe_threshold":  0.80,
        "reach_threshold": 1.20,
        "disclaimer": None,
    },
    "csab": {
        "csv":        os.path.join(os.path.dirname(__file__), "..", "data", "csab_ranks.csv"),
        "model":      "csab_model.pkl",
        "round_col":  "Special Round",
        "rounds":     CSAB_ROUNDS,
        "trend_model": "mlp_ensemble",   # GP-MLP ensemble cuts MAE ~15% vs pure GP RBF
        "mlp_hidden":  (256, 128, 64),   # default arch (CSAB dataset is smaller)
        "mlp_dropout": 0.2,
        "blend_alpha": 0.7,              # LOO-calibrated (2023-2024 avg): more robust than single-year 0.9
        # Widen thresholds - CSAB MAE (~42k) is still ~15x JOSAA MAE (~3k),
        # so tight safe/reach bands would be misleading.
        "safe_threshold":  0.60,
        "reach_threshold": 1.50,
        "disclaimer": (
            "CSAB NOTICE: These predictions apply only to institutes with "
            "leftover seats after JOSAA counselling is complete. An institute "
            "will NOT appear in CSAB if all its seats were filled during JOSAA. "
            "CSAB closing ranks are highly variable year-to-year (model MAE ~50,000); "
            "treat all categories as rough guidance only."
        ),
    },
}

# Model artifact paths
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "josaa_model.pkl")   # default (JOSAA)

# Actual round data available for the current counselling year (JoSAA only).
# Keys are round numbers; values are paths to the corresponding CSV files.
# To add R2 when it drops: add  2: os.path.join(DATA_DIR, "Round2-2026.csv")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CURRENT_ROUND_DATA: dict[int, str] = {
    1: os.path.join(DATA_DIR, "Round1-2026.csv"),
}
