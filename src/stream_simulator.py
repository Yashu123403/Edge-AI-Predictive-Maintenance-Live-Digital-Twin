"""
Simulates a live fleet of edge devices streaming sensor telemetry.

Instead of inventing synthetic waveforms, this simulator REPLAYS genuine
NASA C-MAPSS sensor trajectories cycle-by-cycle, at a configurable
interval, round-robin across several engines at once -- exactly what a
fleet of real turbofan-mounted edge sensors would look like streaming
into a monitoring backend. This is the standard, honest way to simulate
"live" IoT data from a public dataset without fabricating readings.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from src import config


@dataclass
class SensorReading:
    unit_nr: int
    time_cycles: int
    row: pd.Series
    true_rul: float | None = None


class SensorStreamSimulator:
    """Round-robins cycle-by-cycle readings across a subset of engines."""

    def __init__(self, df: pd.DataFrame, fleet_size: int = config.FLEET_SIZE,
                 interval_sec: float = config.DEFAULT_STREAM_INTERVAL_SEC,
                 seed: int = 42):
        self.interval_sec = interval_sec
        rng = pd.Series(df["unit_nr"].unique()).sample(
            n=min(fleet_size, df["unit_nr"].nunique()), random_state=seed
        )
        self.unit_ids = sorted(rng.tolist())
        self.streams = {
            uid: df[df.unit_nr == uid].sort_values("time_cycles").reset_index(drop=True)
            for uid in self.unit_ids
        }
        self._pointers = {uid: 0 for uid in self.unit_ids}

    def is_exhausted(self) -> bool:
        return all(self._pointers[uid] >= len(self.streams[uid]) for uid in self.unit_ids)

    def step(self) -> list[SensorReading]:
        """Advance every active engine by exactly one cycle. Returns the
        batch of new readings produced this tick (engines that already
        finished their trajectory are skipped)."""
        batch = []
        for uid in self.unit_ids:
            ptr = self._pointers[uid]
            stream = self.streams[uid]
            if ptr >= len(stream):
                continue
            row = stream.iloc[ptr]
            true_rul = float(row["RUL"]) if "RUL" in row else None
            batch.append(SensorReading(unit_nr=uid, time_cycles=int(row.time_cycles),
                                        row=row, true_rul=true_rul))
            self._pointers[uid] += 1
        return batch

    def run(self, max_steps: int | None = None) -> Iterator[list[SensorReading]]:
        """Generator form: yields one batch (one tick across the fleet) at a time,
        sleeping `interval_sec` between ticks to emulate real-time streaming."""
        steps = 0
        while not self.is_exhausted():
            if max_steps is not None and steps >= max_steps:
                break
            yield self.step()
            steps += 1
            time.sleep(self.interval_sec)
