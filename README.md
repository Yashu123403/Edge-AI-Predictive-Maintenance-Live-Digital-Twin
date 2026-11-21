# Edge-AI Predictive Maintenance + Live Digital Twin

A software-only, real-time simulation of an **industrial predictive maintenance system**:
a simulated fleet of sensor-equipped machines streams live telemetry, an edge-style inference
engine runs Remaining-Useful-Life (RUL) prediction + anomaly detection +
composite health scoring on every reading, and a live Streamlit dashboard renders each
machine's **digital twin** in real time.
No hardware, no Jetson/Pi/ESP32, no physical sensors — runs entirely on a normal
Windows/Linux laptop in VS Code.
This project translates that exact pattern — **stream → edge inference → health
score → digital twin → dashboard** — into a runnable, testable Python codebase.

## Key Features

- Real-time multi-engine sensor-stream simulation
- Per-engine RUL prediction using Random Forest
- Isolation Forest anomaly detection
- Composite 0–100 health scoring
- Per-engine Digital Twin state management
- Live Streamlit fleet dashboard
- Engine-level RUL, health, anomaly, alert, and sensor-history visualization
- Bounded per-engine feature buffers for edge-style inference
- Automated evaluation and unit testing


## 1. Architecture


NASA C-MAPSS FD001            (public turbofan degradation simulation dataset, replayed as a live stream)
        │
        ▼
SensorStreamSimulator          round-robins cycle-by-cycle readings across a
(src/stream_simulator.py)      simulated fleet of N engines, like N edge devices
        │  SensorReading       representing independent edge-connected machines
        ▼
EdgeInferenceEngine             maintains a small rolling buffer per engine (bounded
(src/edge_engine.py)            memory, edge-feasible), builds the SAME feature vector
        │                       used at training time, then runs:
        │                         • RandomForestRegressor  -> predicted RUL (cycles)
        │                         • IsolationForest        -> anomaly flag + score
        │                         • sensor-deviation score  -> distance from healthy baseline
        │  Prediction (+ latency_ms)
        ▼
health_scoring.compute_health   combines all three signals into one 0-100
(src/health_scoring.py)         Health Index + HEALTHY / WARNING / CRITICAL status
        │
        ▼
DigitalTwinFleet                 owns one TwinState per engine: rolling history,
(src/digital_twin.py)            alert log, latest prediction — the single source
        │                        of truth for any UI
        ▼
Streamlit Dashboard               live fleet table, per-engine gauge, RUL/health/
(dashboard/app.py)                 sensor trend charts, alert feed


**Design principle:** the UI never touches the ML models directly — it only reads
`DigitalTwinFleet` state. This mirrors how a real plant dashboard sits on top of a twin
service rather than embedding inference logic, and is what lets you swap the UI (web,
mobile, Grafana, Power BI) without touching the ML layer.

**Train/serve consistency:** `src/features.py` is imported by BOTH the offline training
script and the real-time edge engine, so the exact same feature computation runs in
both places — a common real-world PdM bug (train/serve skew) is designed out.



## 2. Dataset
**NASA C-MAPSS Turbofan Engine Degradation Simulation, subset FD001**
(Saxena & Goebel, 2008, NASA Prognostics Center of Excellence) — a widely used public benchmark for predictive maintenance and Remaining Useful Life (RUL) research.

* 100 turbofan engines with run-to-failure degradation trajectories
* Single operating condition and single fault mode (HPC degradation)
* 3 operational settings + 21 sensor channels covering temperatures, pressures, speeds, fuel flow ratios, and related measurements
* 20,631 training cycles / 13,096 test cycles
* Small enough to include in the repository (~5 MB as CSV), so no external dataset download is required at runtime
* Included under `data/` as:

  * `train_FD001.csv`
  * `test_FD001.csv`
  * `RUL_FD001.csv`

**Important:** C-MAPSS is a **simulated turbofan degradation dataset**, not measurements collected from physical aircraft engines. This project uses the publicly available simulated trajectories and replays them as a live sensor stream to demonstrate an end-to-end predictive-maintenance and Digital Twin pipeline.

This makes the project reproducible while using a well-established benchmark dataset rather than inventing synthetic sensor readings.


## 3. Folder structure

edge-ai-predictive-maintenance/
├── data/
│   ├── train_FD001.csv          # 100 engines, run-to-failure
│   ├── test_FD001.csv           # 100 engines, truncated trajectories
│   └── RUL_FD001.csv            # ground-truth RUL at truncation point
├── src/
│   ├── config.py                # all paths, thresholds, schema in one place
│   ├── data_loader.py           # load + RUL labeling
│   ├── features.py              # rolling features, healthy baseline, deviation score
│   ├── train_models.py          # offline training -> models_store/*.joblib
│   ├── edge_engine.py           # real-time per-reading inference (the "edge AI")
│   ├── health_scoring.py        # composite 0-100 health index
│   ├── stream_simulator.py      # replays dataset as a live multi-engine stream
│   ├── digital_twin.py          # per-engine live state (the "digital twin")
│   └── evaluate.py              # generates evaluation/metrics_report.md
├── dashboard/
│   └── app.py                   # Streamlit live digital-twin dashboard
├── models_store/                # trained model artifacts (generated by train_models.py)
├── evaluation/
│   └── metrics_report.md        # generated evaluation report
├── tests/
│   ├── test_features.py
│   ├── test_health_scoring.py
│   └── test_stream_and_twin.py
├── run_pipeline.py              # CLI: train / evaluate / simulate / dashboard
├── requirements.txt
└── README.md


## 4. Setup & run instructions (VS Code, Windows or Linux)

# 1. Clone / open the folder in VS Code, then create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the models (RUL regressor + anomaly detector). Takes ~2 minutes on a laptop CPU.
python run_pipeline.py train

# 4. (Optional) Generate the evaluation report
python run_pipeline.py evaluate

# 5a. Quick headless demo in the terminal (no browser needed)
python run_pipeline.py simulate --fleet-size 5 --steps 80

# 5b. Full live Digital Twin dashboard
streamlit run dashboard/app.py
# opens at http://localhost:8501 — click "Start" in the sidebar

Models are saved to `models_store/` after step 3 and are reused by everything else —
you only need to train once.

## 5. Testing

pip install pytest   # already in requirements.txt
python -m pytest tests/ -v

14 unit tests covering:
- Feature engineering correctness (rolling-mean math verified against a hand-computed
  toy signal, no-NaN guarantee, no duplicate feature columns)
- Health-scoring logic (bounds, monotonicity in RUL, correct HEALTHY/WARNING/CRITICAL
  classification for known-good and known-bad inputs)
- Streaming + digital twin integration (round-robin ordering, exhaustion, twin history
  length matches ticks processed, fleet summary matches active units)

All 14 currently pass (`14 passed in ~4s`).



## 6. Evaluation metrics (actual results on the FD001 test split)

Full report: [`evaluation/metrics_report.md`](evaluation/metrics_report.md) — generated
by `python run_pipeline.py evaluate`, not hand-written.

**RUL Regression (RandomForestRegressor, 250 trees)**

| Metric | All streamed cycles | Final cycle per engine (benchmark convention) |
|---|---|---|
| RMSE | 16.72 cycles | 17.96 cycles |
| MAE | 11.67 cycles | 12.81 cycles |
| R² | 0.632 | 0.799 |
| NASA PHM08 score | — | 1003.6 (lower is better) |

This provides an honest classical-ML baseline, with clear opportunities for improvement through
richer feature engineering and sequence/deep-learning models.

**Anomaly Detection (IsolationForest)** — evaluated against the standard literature
proxy `RUL ≤ 30 cycles` = "impending failure window" (C-MAPSS has no explicit
fault-onset label, so this proxy is used and disclosed, not invented as ground truth):

| Metric                        | Value |
| ----------------------------- | ----- |
| Recall                        | 1.000 |
| Precision                     | 0.176 |
| % of cycles flagged anomalous | 14.4% |

Recall of 1.0 means the detector flagged every test cycle labeled by the evaluation
proxy as being within the final 30 cycles. The low precision reflects the trade-off
associated with an unsupervised anomaly detector configured for high recall.


**Edge inference latency** (single reading, single CPU core, this dev laptop):

| Percentile | Latency |
|---|---|
| p50 | 38.5 ms |
| p95 | 41.5 ms |
| p99 | 45.1 ms |


## 7. Honesty notes 
- All sensor data is replayed directly from the public NASA C-MAPSS dataset; no
  synthetic sensor values are generated.
- Anomaly detection precision/recall use an approximate proxy label (`RUL ≤ 30`)
  because this dataset has no explicit fault-onset annotation; this is disclosed in
  `evaluation/metrics_report.md`, not hidden.
- "Edge AI" here means the *software architecture and constraints* of edge inference
  (bounded memory, per-reading latency budget, lightweight model and dependency
  footprint) — it runs on a laptop CPU, not physical edge hardware, per the project's software-only constraint.

  AUTHOR
  YASHU A.B
