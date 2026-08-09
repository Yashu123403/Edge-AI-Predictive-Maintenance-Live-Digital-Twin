# Evaluation Report — Edge-AI Predictive Maintenance

Dataset: NASA C-MAPSS FD001 test split (100 engines, 13,096 cycles)

## RUL Regression (RandomForestRegressor)

### All streamed cycles (continuous monitoring view)
- RMSE: **16.72 cycles**
- MAE: **11.67 cycles**
- R²: **0.632**
- Evaluated on 13096 test cycles

### Final cycle per engine (standard FD001 benchmark convention)
- RMSE: **17.96 cycles**
- MAE: **12.81 cycles**
- R²: **0.799**
- NASA PHM08 asymmetric score: **1003.6** (lower is better; penalizes late/optimistic predictions more heavily than early/conservative ones)
- Evaluated on 100 engines (1 final cycle each)

## Anomaly Detection (IsolationForest) — proxy evaluation

- RUL <= 30 cycles treated as impending-failure window
- Precision: **0.176**
- Recall: **1.000**
- Fraction of cycles flagged anomalous: 0.144
- Fraction of cycles actually in proxy failure window: 0.025

## Edge Inference Latency (single-reading, single-core)

- p50: **38.52 ms**
- p95: **41.47 ms**
- p99: **45.14 ms**
- max: 49.73 ms
- Measured over 300 live predict() calls

## Notes / Honesty disclosures

- Sensor stream is a genuine replay of NASA C-MAPSS data, not fabricated.
- Anomaly precision/recall use RUL<=30 as an approximate proxy label because C-MAPSS has no explicit fault-onset annotation; this is standard practice in the PdM literature but is an approximation, not ground truth for anomalies.
- Latency figures reflect scikit-learn inference on this dev machine's CPU, not a physically deployed edge device (per project scope: software-only).
