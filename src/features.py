"""
Feature engineering shared by offline training and the real-time edge
inference engine. Using the SAME functions in both places guarantees
train/serve consistency -- a common real-world pitfall in predictive
maintenance systems.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def add_rolling_features(df: pd.DataFrame, window: int = config.ROLLING_WINDOW) -> pd.DataFrame:
    """
    Add rolling mean/std per engine for the selected sensors. This mimics
    what a resource-constrained edge device would keep in a small circular
    buffer (no need to store full history).
    """
    df = df.sort_values(["unit_nr", "time_cycles"]).copy()
    grouped = df.groupby("unit_nr")[config.USED_SENSORS]

    roll_mean = grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
    roll_std = grouped.transform(lambda s: s.rolling(window, min_periods=1).std().fillna(0.0))

    roll_mean.columns = [f"{c}_rmean" for c in config.USED_SENSORS]
    roll_std.columns = [f"{c}_rstd" for c in config.USED_SENSORS]

    out = pd.concat([df, roll_mean, roll_std], axis=1)
    return out


def get_feature_columns() -> list[str]:
    """Final feature set used by every model (order matters -> stored + reused)."""
    base = config.USED_SENSORS + config.SETTING_COLS
    rmean = [f"{c}_rmean" for c in config.USED_SENSORS]
    rstd = [f"{c}_rstd" for c in config.USED_SENSORS]
    return base + rmean + rstd


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: raw dataframe -> engineered feature matrix (no scaling)."""
    feat_df = add_rolling_features(df)
    cols = get_feature_columns()
    return feat_df[cols]


def compute_healthy_baseline(train_df: pd.DataFrame,
                              healthy_fraction: float = config.HEALTHY_FRACTION) -> dict:
    """
    Build a per-sensor "healthy" mean/std baseline using only the earliest
    `healthy_fraction` of each engine's life (when degradation is assumed
    negligible). Used later to compute a live sensor-deviation score for
    the health index -- a lightweight physics-free proxy for condition
    monitoring commonly used in industrial anomaly detection.
    """
    df = train_df.copy()
    max_cycle = df.groupby("unit_nr")["time_cycles"].transform("max")
    healthy_mask = df["time_cycles"] <= (healthy_fraction * max_cycle)
    healthy_df = df[healthy_mask]

    baseline = {
        "mean": healthy_df[config.USED_SENSORS].mean(),
        "std": healthy_df[config.USED_SENSORS].std().replace(0, 1e-6),
    }
    return baseline


def sensor_deviation_score(row: pd.Series, baseline: dict) -> float:
    """
    Average absolute z-score of current sensor readings vs. the healthy
    baseline, i.e. how far the engine currently is from "normal" behaviour.
    Higher = more deviated.
    """
    z = (row[config.USED_SENSORS] - baseline["mean"]) / baseline["std"]
    return float(np.abs(z).mean())
