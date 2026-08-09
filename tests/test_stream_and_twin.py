import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import data_loader
from src.digital_twin import DigitalTwinFleet
from src.edge_engine import EdgeInferenceEngine
from src.stream_simulator import SensorStreamSimulator


def _small_test_df():
    _, test_df = data_loader.get_labeled_train_test()
    return test_df[test_df.unit_nr.isin([1, 2, 3])]


def test_stream_simulator_round_robin_order():
    df = _small_test_df()
    sim = SensorStreamSimulator(df, fleet_size=3, interval_sec=0.0, seed=0)
    batch1 = sim.step()
    # first tick should return exactly one reading per engine, all at cycle 1
    assert len(batch1) == 3
    assert all(r.time_cycles == 1 for r in batch1)


def test_stream_simulator_exhausts():
    df = _small_test_df()
    sim = SensorStreamSimulator(df, fleet_size=3, interval_sec=0.0, seed=0)
    steps = 0
    for _ in sim.run():
        steps += 1
        if steps > 500:
            break  # safety valve
    assert sim.is_exhausted()
    assert steps > 0


def test_digital_twin_updates_and_history():
    df = _small_test_df()
    sim = SensorStreamSimulator(df, fleet_size=1, interval_sec=0.0, seed=0)
    engine = EdgeInferenceEngine()
    fleet = DigitalTwinFleet()

    n = 0
    for batch in sim.run(max_steps=10):
        for reading in batch:
            pred = engine.predict(reading.unit_nr, reading.time_cycles, reading.row, reading.true_rul)
            fleet.update(pred)
            n += 1

    uid = sim.unit_ids[0]
    twin = fleet.get(uid)
    assert twin is not None
    assert twin.cycles_seen == n
    assert len(twin.history) == n
    assert twin.latest is not None
    assert 0 <= twin.latest.health_score <= 100


def test_fleet_summary_matches_active_units():
    df = _small_test_df()
    sim = SensorStreamSimulator(df, fleet_size=2, interval_sec=0.0, seed=1)
    engine = EdgeInferenceEngine()
    fleet = DigitalTwinFleet()

    for batch in sim.run(max_steps=5):
        for reading in batch:
            pred = engine.predict(reading.unit_nr, reading.time_cycles, reading.row, reading.true_rul)
            fleet.update(pred)

    summary = fleet.fleet_summary()
    assert len(summary) == len(sim.unit_ids)
    for row in summary:
        assert row["Status"] in ("HEALTHY", "WARNING", "CRITICAL")
