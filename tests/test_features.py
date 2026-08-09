import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config, features


def _toy_df():
    rows = []
    for cycle in range(1, 8):
        row = {"unit_nr": 1, "time_cycles": cycle}
        for s in config.SETTING_COLS:
            row[s] = 0.0
        for s in config.SENSOR_COLS:
            row[s] = float(cycle)  # simple increasing signal
        rows.append(row)
    return pd.DataFrame(rows)


def test_add_rolling_features_shape():
    df = _toy_df()
    out = features.add_rolling_features(df, window=3)
    for c in config.USED_SENSORS:
        assert f"{c}_rmean" in out.columns
        assert f"{c}_rstd" in out.columns
    assert len(out) == len(df)


def test_rolling_mean_correctness():
    df = _toy_df()
    out = features.add_rolling_features(df, window=3)
    # sensor value == cycle number, so rolling mean at cycle 5 over window 3 (3,4,5) == 4
    row = out[out.time_cycles == 5].iloc[0]
    assert abs(row["sensor_2_rmean"] - 4.0) < 1e-9


def test_feature_columns_no_duplicates():
    cols = features.get_feature_columns()
    assert len(cols) == len(set(cols))


def test_build_feature_matrix_no_nans():
    df = _toy_df()
    X = features.build_feature_matrix(df)
    assert not X.isnull().values.any()


def test_healthy_baseline_and_deviation_zero_for_healthy_row():
    df = _toy_df()
    df["RUL"] = 100
    baseline = features.compute_healthy_baseline(df, healthy_fraction=1.0)
    # a row exactly at the mean should have ~0 deviation
    mean_row = df.iloc[0].copy()
    for s in config.USED_SENSORS:
        mean_row[s] = baseline["mean"][s]
    dev = features.sensor_deviation_score(mean_row, baseline)
    assert dev < 1e-6
