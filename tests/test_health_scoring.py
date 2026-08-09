import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, health_scoring


def test_healthy_engine_scores_high():
    # High predicted RUL, normal (positive) anomaly score, near-zero deviation
    result = health_scoring.compute_health(rul_pred=120, anomaly_raw=0.15, deviation=0.2)
    assert result.status == "HEALTHY"
    assert result.health_score > config.HEALTH_WARNING_THRESHOLD


def test_failing_engine_scores_low():
    # Near-zero RUL, negative (anomalous) score, large sensor deviation
    result = health_scoring.compute_health(rul_pred=2, anomaly_raw=-0.3, deviation=5.5)
    assert result.status == "CRITICAL"
    assert result.health_score < config.HEALTH_CRITICAL_THRESHOLD


def test_health_score_is_monotonic_in_rul():
    low = health_scoring.compute_health(rul_pred=10, anomaly_raw=0.0, deviation=1.0)
    high = health_scoring.compute_health(rul_pred=100, anomaly_raw=0.0, deviation=1.0)
    assert high.health_score > low.health_score


def test_health_score_bounded_0_100():
    for rul in [-50, 0, 60, 125, 500]:
        for anomaly in [-1.0, 0.0, 1.0]:
            for dev in [0.0, 3.0, 20.0]:
                result = health_scoring.compute_health(rul, anomaly, dev)
                assert 0.0 <= result.health_score <= 100.0


def test_status_thresholds_consistent():
    assert config.HEALTH_CRITICAL_THRESHOLD < config.HEALTH_WARNING_THRESHOLD
