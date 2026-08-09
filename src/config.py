"""
Central configuration for the Edge-AI Predictive Maintenance + Live Digital Twin project.

All paths, column names and tunable thresholds live here so the rest of the
codebase never hardcodes a "magic" value.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models_store"
EVAL_DIR = ROOT_DIR / "evaluation"

TRAIN_FILE = DATA_DIR / "train_FD001.csv"
TEST_FILE = DATA_DIR / "test_FD001.csv"
RUL_FILE = DATA_DIR / "RUL_FD001.csv"

RUL_MODEL_PATH = MODELS_DIR / "rul_model.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
FEATURE_LIST_PATH = MODELS_DIR / "feature_list.joblib"
HEALTHY_BASELINE_PATH = MODELS_DIR / "healthy_baseline.joblib"

# --------------------------------------------------------------------------
# Dataset schema  (NASA C-MAPSS FD001 turbofan degradation simulation)
# --------------------------------------------------------------------------
INDEX_COLS = ["unit_nr", "time_cycles"]
SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_RAW_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Sensors that are effectively constant / non-informative in FD001 (single
# operating condition). Dropping them is a standard, well documented step
# in C-MAPSS literature (Saxena et al., 2008) and keeps the "edge device"
# feature set small and cheap to compute in real time.
DROP_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                 "sensor_16", "sensor_18", "sensor_19"]

USED_SENSORS = [c for c in SENSOR_COLS if c not in DROP_SENSORS]

# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
ROLLING_WINDOW = 5          # cycles used for rolling mean/std (edge-friendly, small buffer)
RUL_CLIP = 125              # standard piecewise-linear RUL clipping (Heimes 2008)
HEALTHY_FRACTION = 0.3      # first 30% of an engine's life is treated as "healthy" baseline

# --------------------------------------------------------------------------
# Health scoring
# --------------------------------------------------------------------------
# Composite health index weights (must sum to 1.0)
W_RUL = 0.5
W_ANOMALY = 0.3
W_DEVIATION = 0.2

HEALTH_WARNING_THRESHOLD = 60   # score < 60  -> WARNING
HEALTH_CRITICAL_THRESHOLD = 35  # score < 35  -> CRITICAL

# --------------------------------------------------------------------------
# Streaming simulation
# --------------------------------------------------------------------------
DEFAULT_STREAM_INTERVAL_SEC = 0.15   # simulated edge sampling interval
FLEET_SIZE = 6                        # number of engines streamed concurrently in the demo
