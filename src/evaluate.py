"""
Generates evaluation/metrics_report.md from the held-out C-MAPSS test set.

Reports:
  - RUL regression:  RMSE, MAE, R^2, and the NASA PHM08 asymmetric scoring
    function (penalizes late predictions -- i.e. predicting MORE life than
    an engine actually has -- far more heavily than early predictions,
    because a machine that fails before you expect it to is much worse
    than one you replace a bit early).
  - Anomaly detection: since C-MAPSS has no labeled fault-onset timestamps,
    we evaluate against the standard literature proxy "RUL <= 30 cycles"
    as an approximate impending-failure window and report precision/recall
    of the anomaly flag against that proxy. This is clearly documented as
    an approximation, not a fabricated ground truth.
  - Edge inference latency distribution (p50/p95/p99) measured live.

Run:  python -m src.evaluate
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              precision_score, r2_score, recall_score)

from src import config, data_loader, features
from src.edge_engine import EdgeInferenceEngine


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA PHM08 challenge scoring function (Saxena et al., 2008).
    d = pred - true.  d<0 (early / conservative prediction) is penalized
    gently; d>0 (late / optimistic prediction, i.e. told the operator the
    machine had more life than it did) is penalized much more heavily,
    reflecting real safety cost asymmetry in maintenance decisions.
    """
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(score))


def evaluate_rul(engine: EdgeInferenceEngine, test_df: pd.DataFrame) -> dict:
    """
    Reports two views, both standard in the C-MAPSS literature:
      - 'all_cycles'   : RMSE/MAE/R2 over every streamed cycle (continuous
                          monitoring view -- what the live dashboard actually sees)
      - 'final_cycle'  : RMSE/MAE/R2/PHM08 evaluated only at each engine's
                          LAST observed test cycle (the standard C-MAPSS
                          FD001 benchmark convention used to compare against
                          published papers, since PHM08 scoring explodes if
                          summed across thousands of far-from-failure cycles).
    """
    X = features.build_feature_matrix(test_df)
    X_scaled = engine.scaler.transform(X)
    y_true_all = test_df["RUL"].values
    y_pred_all = np.clip(engine.rul_model.predict(X_scaled), 0, config.RUL_CLIP)

    all_cycles = {
        "rmse": float(np.sqrt(mean_squared_error(y_true_all, y_pred_all))),
        "mae": float(mean_absolute_error(y_true_all, y_pred_all)),
        "r2": float(r2_score(y_true_all, y_pred_all)),
        "n_samples": len(y_true_all),
    }

    last_idx = test_df.groupby("unit_nr")["time_cycles"].idxmax()
    final_df = test_df.loc[last_idx]
    X_final = features.build_feature_matrix(test_df).loc[last_idx]
    X_final_scaled = engine.scaler.transform(X_final)
    y_true_final = final_df["RUL"].values
    y_pred_final = np.clip(engine.rul_model.predict(X_final_scaled), 0, config.RUL_CLIP)

    final_cycle = {
        "rmse": float(np.sqrt(mean_squared_error(y_true_final, y_pred_final))),
        "mae": float(mean_absolute_error(y_true_final, y_pred_final)),
        "r2": float(r2_score(y_true_final, y_pred_final)),
        "phm08_score": phm08_score(y_true_final, y_pred_final),
        "n_samples": len(y_true_final),
    }

    return {"all_cycles": all_cycles, "final_cycle": final_cycle}


def evaluate_anomaly_proxy(engine: EdgeInferenceEngine, test_df: pd.DataFrame,
                            rul_threshold: int = 30) -> dict:
    X = features.build_feature_matrix(test_df)
    X_scaled = engine.scaler.transform(X)
    anomaly_pred = (engine.anomaly_model.predict(X_scaled) == -1).astype(int)

    proxy_label = (test_df["RUL"].values <= rul_threshold).astype(int)

    precision = precision_score(proxy_label, anomaly_pred, zero_division=0)
    recall = recall_score(proxy_label, anomaly_pred, zero_division=0)

    return {
        "proxy_definition": f"RUL <= {rul_threshold} cycles treated as impending-failure window",
        "precision": float(precision),
        "recall": float(recall),
        "flagged_rate": float(anomaly_pred.mean()),
        "proxy_positive_rate": float(proxy_label.mean()),
    }


def evaluate_latency(engine: EdgeInferenceEngine, test_df: pd.DataFrame,
                      n_samples: int = 300) -> dict:
    sample = test_df.sort_values(["unit_nr", "time_cycles"]).groupby("unit_nr").head(20)
    sample = sample.head(n_samples)

    latencies = []
    for _, row in sample.iterrows():
        t0 = time.perf_counter()
        engine.predict(int(row.unit_nr), int(row.time_cycles), row, float(row.RUL))
        latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
        "n_samples": len(arr),
    }


def main():
    print("Loading test data + trained models ...")
    _, test_df = data_loader.get_labeled_train_test()
    engine = EdgeInferenceEngine()

    print("Evaluating RUL regression ...")
    rul_metrics = evaluate_rul(engine, test_df)

    print("Evaluating anomaly detector (proxy labels) ...")
    anomaly_metrics = evaluate_anomaly_proxy(engine, test_df)

    print("Measuring live edge-inference latency ...")
    latency_metrics = evaluate_latency(engine, test_df)

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.EVAL_DIR / "metrics_report.md"

    with open(report_path, "w") as f:
        f.write("# Evaluation Report — Edge-AI Predictive Maintenance\n\n")
        f.write("Dataset: NASA C-MAPSS FD001 test split (100 engines, 13,096 cycles)\n\n")

        f.write("## RUL Regression (RandomForestRegressor)\n\n")
        f.write("### All streamed cycles (continuous monitoring view)\n")
        ac = rul_metrics["all_cycles"]
        f.write(f"- RMSE: **{ac['rmse']:.2f} cycles**\n")
        f.write(f"- MAE: **{ac['mae']:.2f} cycles**\n")
        f.write(f"- R²: **{ac['r2']:.3f}**\n")
        f.write(f"- Evaluated on {ac['n_samples']} test cycles\n\n")

        f.write("### Final cycle per engine (standard FD001 benchmark convention)\n")
        fc = rul_metrics["final_cycle"]
        f.write(f"- RMSE: **{fc['rmse']:.2f} cycles**\n")
        f.write(f"- MAE: **{fc['mae']:.2f} cycles**\n")
        f.write(f"- R²: **{fc['r2']:.3f}**\n")
        f.write(f"- NASA PHM08 asymmetric score: **{fc['phm08_score']:.1f}** "
                f"(lower is better; penalizes late/optimistic predictions more heavily "
                f"than early/conservative ones)\n")
        f.write(f"- Evaluated on {fc['n_samples']} engines (1 final cycle each)\n\n")

        f.write("## Anomaly Detection (IsolationForest) — proxy evaluation\n\n")
        f.write(f"- {anomaly_metrics['proxy_definition']}\n")
        f.write(f"- Precision: **{anomaly_metrics['precision']:.3f}**\n")
        f.write(f"- Recall: **{anomaly_metrics['recall']:.3f}**\n")
        f.write(f"- Fraction of cycles flagged anomalous: {anomaly_metrics['flagged_rate']:.3f}\n")
        f.write(f"- Fraction of cycles actually in proxy failure window: "
                f"{anomaly_metrics['proxy_positive_rate']:.3f}\n\n")

        f.write("## Edge Inference Latency (single-reading, single-core)\n\n")
        f.write(f"- p50: **{latency_metrics['p50_ms']:.2f} ms**\n")
        f.write(f"- p95: **{latency_metrics['p95_ms']:.2f} ms**\n")
        f.write(f"- p99: **{latency_metrics['p99_ms']:.2f} ms**\n")
        f.write(f"- max: {latency_metrics['max_ms']:.2f} ms\n")
        f.write(f"- Measured over {latency_metrics['n_samples']} live predict() calls\n\n")

        f.write("## Notes / Honesty disclosures\n\n")
        f.write("- Sensor stream is a genuine replay of NASA C-MAPSS data, not fabricated.\n")
        f.write("- Anomaly precision/recall use RUL<=30 as an approximate proxy label because "
                "C-MAPSS has no explicit fault-onset annotation; this is standard practice in "
                "the PdM literature but is an approximation, not ground truth for anomalies.\n")
        f.write("- Latency figures reflect scikit-learn inference on this dev machine's CPU, "
                "not a physically deployed edge device (per project scope: software-only).\n")

    print(f"\nSaved report to {report_path}")
    print("\n--- Summary ---")
    print("RUL (all cycles):", rul_metrics["all_cycles"])
    print("RUL (final cycle / benchmark):", rul_metrics["final_cycle"])
    print(anomaly_metrics)
    print(latency_metrics)


if __name__ == "__main__":
    main()
