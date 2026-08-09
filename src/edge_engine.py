"""
Edge Inference Engine.

Simulates inference running on an edge device or edge gateway.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from src import config, features, health_scoring


@dataclass
class Prediction:
    unit_nr: int
    time_cycles: int
    rul_pred: float
    true_rul: float | None
    anomaly_flag: bool
    anomaly_raw: float
    deviation: float
    health_score: float
    status: str
    latency_ms: float
    sensors: dict = field(default_factory=dict)


class EdgeInferenceEngine:

    def __init__(self, window: int = config.ROLLING_WINDOW):
        self.window = window

        self.rul_model = joblib.load(
            config.RUL_MODEL_PATH
        )

        self.anomaly_model = joblib.load(
            config.ANOMALY_MODEL_PATH
        )

        self.scaler = joblib.load(
            config.SCALER_PATH
        )

        self.feature_cols = joblib.load(
            config.FEATURE_LIST_PATH
        )

        self.baseline = joblib.load(
            config.HEALTHY_BASELINE_PATH
        )

        self._buffers: dict[int, deque] = {}

    def _get_buffer(self, unit_nr: int) -> deque:

        if unit_nr not in self._buffers:
            self._buffers[unit_nr] = deque(
                maxlen=self.window
            )

        return self._buffers[unit_nr]

    def _build_feature_row(
        self,
        unit_nr: int,
        row: pd.Series,
    ) -> pd.DataFrame:

        buf = self._get_buffer(unit_nr)

        buf.append(
            row[config.USED_SENSORS].astype(float)
        )

        window_df = pd.DataFrame(
            list(buf)
        )

        r_mean = window_df.mean()

        r_std = window_df.std(
            ddof=1
        ).fillna(0.0)

        feat = {}

        for c in config.USED_SENSORS:
            feat[c] = float(row[c])

        for c in config.SETTING_COLS:
            feat[c] = float(row[c])

        for c in config.USED_SENSORS:
            feat[f"{c}_rmean"] = float(
                r_mean[c]
            )

        for c in config.USED_SENSORS:
            feat[f"{c}_rstd"] = float(
                r_std[c]
            )

        return pd.DataFrame(
            [feat]
        )[self.feature_cols]

    def predict(
        self,
        unit_nr: int,
        time_cycles: int,
        row: pd.Series,
        true_rul: float | None = None,
    ) -> Prediction:

        t0 = time.perf_counter()

        feat_row = self._build_feature_row(
            unit_nr,
            row,
        )

        X = self.scaler.transform(
            feat_row
        )

        rul_pred = float(
            self.rul_model.predict(X)[0]
        )

        rul_pred = float(
            np.clip(
                rul_pred,
                0,
                config.RUL_CLIP,
            )
        )

        anomaly_raw = float(
            self.anomaly_model.decision_function(X)[0]
        )

        anomaly_flag = bool(
            self.anomaly_model.predict(X)[0] == -1
        )

        deviation = features.sensor_deviation_score(
            row,
            self.baseline,
        )

        health = health_scoring.compute_health(
            rul_pred,
            anomaly_raw,
            deviation,
        )

        latency_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        return Prediction(
            unit_nr=unit_nr,
            time_cycles=time_cycles,
            rul_pred=round(
                rul_pred,
                1,
            ),
            true_rul=true_rul,
            anomaly_flag=anomaly_flag,
            anomaly_raw=round(
                anomaly_raw,
                4,
            ),
            deviation=round(
                deviation,
                3,
            ),
            health_score=health.health_score,
            status=health.status,
            latency_ms=round(
                latency_ms,
                3,
            ),
            sensors={
                c: float(row[c])
                for c in config.USED_SENSORS
            },
        )