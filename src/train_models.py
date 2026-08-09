"""
Offline training pipeline.

Trains two models on the C-MAPSS FD001 training set:

  1. RandomForestRegressor  -> predicts Remaining Useful Life (RUL) in cycles
  2. IsolationForest        -> unsupervised anomaly detector fit ONLY on
                                data from each engine's early "healthy" life

Also fits a StandardScaler on the feature matrix and stores the healthy
sensor baseline used for the deviation component of the health score.

Run:  python -m src.train_models
"""
from __future__ import annotations

import time

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src import config, data_loader, features


def train():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading and labeling C-MAPSS FD001 ...")
    train_df, test_df = data_loader.get_labeled_train_test()
    print(f"      train engines={train_df.unit_nr.nunique()}  rows={len(train_df)}")
    print(f"      test  engines={test_df.unit_nr.nunique()}  rows={len(test_df)}")

    print("[2/6] Engineering features (rolling mean/std, window="
          f"{config.ROLLING_WINDOW}) ...")
    X_train_raw = features.build_feature_matrix(train_df)
    y_train = train_df["RUL"].values

    X_test_raw = features.build_feature_matrix(test_df)
    y_test = test_df["RUL"].values

    print("[3/6] Fitting StandardScaler ...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print("[4/6] Training RandomForestRegressor (RUL prediction) ...")
    t0 = time.time()
    rul_model = RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    rul_model.fit(X_train, y_train)
    print(f"      trained in {time.time()-t0:.1f}s")

    preds = rul_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    print(f"      Test RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")

    print("[5/6] Training IsolationForest (anomaly detection) on healthy data ...")
    max_cycle = train_df.groupby("unit_nr")["time_cycles"].transform("max")
    healthy_mask = train_df["time_cycles"] <= (config.HEALTHY_FRACTION * max_cycle)
    X_healthy = scaler.transform(X_train_raw[healthy_mask.values])

    anomaly_model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    anomaly_model.fit(X_healthy)

    print("[6/6] Computing healthy sensor baseline + saving artifacts ...")
    baseline = features.compute_healthy_baseline(train_df)

    joblib.dump(rul_model, config.RUL_MODEL_PATH)
    joblib.dump(anomaly_model, config.ANOMALY_MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(features.get_feature_columns(), config.FEATURE_LIST_PATH)
    joblib.dump(baseline, config.HEALTHY_BASELINE_PATH)

    print(f"\nSaved models to {config.MODELS_DIR}")
    print("Done.")

    return {"rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    train()
