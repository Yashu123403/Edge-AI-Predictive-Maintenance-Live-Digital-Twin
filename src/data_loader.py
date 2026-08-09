"""
Loads the NASA C-MAPSS FD001 turbofan degradation dataset and attaches
Remaining-Useful-Life (RUL) labels.

Dataset source (public, genuine, not synthetic):
NASA Prognostics Center of Excellence, "Turbofan Engine Degradation
Simulation Data Set" (Saxena & Goebel, 2008), subset FD001:
100 engines, single operating condition, single fault mode (HPC degradation).
"""
from __future__ import annotations

import pandas as pd

from src import config


def load_train() -> pd.DataFrame:
    """Load the training set (100 engines, run-to-failure trajectories)."""
    df = pd.read_csv(config.TRAIN_FILE)
    return df


def load_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the test set (100 engines, truncated before failure) and the
    ground-truth RUL at the point of truncation."""
    test_df = pd.read_csv(config.TEST_FILE)
    rul_df = pd.read_csv(config.RUL_FILE)
    rul_df["unit_nr"] = rul_df.index + 1
    return test_df, rul_df


def add_train_rul(df: pd.DataFrame, clip: int = config.RUL_CLIP) -> pd.DataFrame:
    """
    For run-to-failure training trajectories, RUL at cycle t is simply
    (max_cycle_for_engine - t). Clipped to `clip` because engines are
    healthy and behave near-identically far away from failure (Heimes 2008
    piecewise-linear RUL target — standard practice for C-MAPSS).
    """
    df = df.copy()
    max_cycle = df.groupby("unit_nr")["time_cycles"].transform("max")
    df["RUL"] = max_cycle - df["time_cycles"]
    if clip is not None:
        df["RUL"] = df["RUL"].clip(upper=clip)
    return df


def add_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame,
                  clip: int = config.RUL_CLIP) -> pd.DataFrame:
    """
    Test trajectories are truncated before failure. The true RUL at the
    LAST observed cycle of each engine is given in RUL_FD001.csv. We
    back-compute RUL for every earlier cycle by adding the remaining
    cycles until that last observed point.
    """
    df = test_df.copy()
    last_cycle = df.groupby("unit_nr")["time_cycles"].transform("max")
    rul_at_end = df["unit_nr"].map(rul_df.set_index("unit_nr")["RUL"])
    df["RUL"] = (last_cycle - df["time_cycles"]) + rul_at_end
    if clip is not None:
        df["RUL"] = df["RUL"].clip(upper=clip)
    return df


def get_labeled_train_test():
    """Convenience wrapper returning fully-labeled train/test DataFrames."""
    train_df = add_train_rul(load_train())
    test_raw, rul_df = load_test()
    test_df = add_test_rul(test_raw, rul_df)
    return train_df, test_df
