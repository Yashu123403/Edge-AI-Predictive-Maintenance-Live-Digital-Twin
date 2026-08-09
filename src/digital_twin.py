"""
Live Digital Twin state layer.

Each physical engine gets a lightweight "twin": an in-memory object that
mirrors its most recent readings, predictions and alert history. The
dashboard reads ONLY from these twins -- it never touches the ML models
or the raw stream directly. This separation (stream -> edge inference ->
twin state -> UI) mirrors a real industrial architecture where the twin
is the single source of truth consumed by any number of visualization
or downstream (e.g. CMMS/ERP alerting) clients.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from src.edge_engine import Prediction

HISTORY_LEN = 60


@dataclass
class TwinState:
    unit_nr: int
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    alerts: deque = field(default_factory=lambda: deque(maxlen=20))
    latest: Prediction | None = None
    cycles_seen: int = 0

    def update(self, pred: Prediction):
        prev_status = self.latest.status if self.latest else None
        self.latest = pred
        self.history.append(pred)
        self.cycles_seen += 1

        if pred.status in ("WARNING", "CRITICAL") and pred.status != prev_status:
            self.alerts.appendleft(
                f"[cycle {pred.time_cycles}] Engine {pred.unit_nr} -> {pred.status} "
                f"(health={pred.health_score}, RUL~{pred.rul_pred}c, anomaly={pred.anomaly_flag})"
            )


class DigitalTwinFleet:
    """Owns one TwinState per engine unit and exposes fleet-level views."""

    def __init__(self):
        self._twins: dict[int, TwinState] = {}

    def update(self, pred: Prediction) -> TwinState:
        if pred.unit_nr not in self._twins:
            self._twins[pred.unit_nr] = TwinState(unit_nr=pred.unit_nr)
        twin = self._twins[pred.unit_nr]
        twin.update(pred)
        return twin

    def get(self, unit_nr: int) -> TwinState | None:
        return self._twins.get(unit_nr)

    def all_units(self) -> list[int]:
        return sorted(self._twins.keys())

    def fleet_summary(self) -> list[dict]:
        rows = []
        for uid in self.all_units():
            t = self._twins[uid]
            if t.latest is None:
                continue
            rows.append({
                "Engine": t.unit_nr,
                "Cycle": t.latest.time_cycles,
                "Predicted RUL": t.latest.rul_pred,
                "Health Score": t.latest.health_score,
                "Status": t.latest.status,
                "Anomaly": "YES" if t.latest.anomaly_flag else "no",
                "Latency (ms)": t.latest.latency_ms,
            })
        return rows
