"""
Composite Health Index (0-100) combining three independent signals:

  1. RUL component       -- how much useful life the ML regressor predicts is left
  2. Anomaly component    -- how "outlier-like" the current behaviour is (Isolation Forest)
  3. Deviation component  -- how far current sensors are from the engine's own healthy baseline

Combining several signals (rather than relying on a single model) is a
standard defensive-design pattern in industrial condition monitoring
systems: it degrades gracefully if any single model is wrong or noisy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import config


@dataclass
class HealthResult:
    health_score: float
    status: str
    rul_component: float
    anomaly_component: float
    deviation_component: float


def _rul_to_score(rul_pred: float, rul_cap: int = config.RUL_CLIP) -> float:
    """Map predicted RUL (cycles) linearly onto a 0-100 scale."""
    return float(np.clip((rul_pred / rul_cap) * 100.0, 0, 100))


def _anomaly_to_score(anomaly_raw: float) -> float:
    """
    IsolationForest.decision_function returns roughly [-0.5, 0.5], with
    NEGATIVE values meaning "more anomalous". We map it onto 0-100 where
    100 = perfectly normal, 0 = highly anomalous, using a logistic squash
    so it's robust to outliers in the raw score.
    """
    squashed = 1.0 / (1.0 + np.exp(-12.0 * anomaly_raw))  # sigmoid centered at 0
    return float(np.clip(squashed * 100.0, 0, 100))


def _deviation_to_score(deviation: float, cap: float = 6.0) -> float:
    """Map mean |z-score| deviation (0 = identical to healthy baseline) to 0-100."""
    return float(np.clip((1 - deviation / cap) * 100.0, 0, 100))


def compute_health(rul_pred: float, anomaly_raw: float, deviation: float) -> HealthResult:
    rul_c = _rul_to_score(rul_pred)
    anomaly_c = _anomaly_to_score(anomaly_raw)
    dev_c = _deviation_to_score(deviation)

    score = (config.W_RUL * rul_c +
             config.W_ANOMALY * anomaly_c +
             config.W_DEVIATION * dev_c)

    if score < config.HEALTH_CRITICAL_THRESHOLD:
        status = "CRITICAL"
    elif score < config.HEALTH_WARNING_THRESHOLD:
        status = "WARNING"
    else:
        status = "HEALTHY"

    return HealthResult(
        health_score=round(score, 1),
        status=status,
        rul_component=round(rul_c, 1),
        anomaly_component=round(anomaly_c, 1),
        deviation_component=round(dev_c, 1),
    )
