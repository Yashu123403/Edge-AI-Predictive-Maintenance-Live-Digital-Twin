# Edge-AI Predictive Maintenance + Live Digital Twin

A software-only, real-time simulation of an **industrial predictive maintenance system**:
a simulated fleet of edge-mounted sensors streams live telemetry, an on-device-style
inference engine runs Remaining-Useful-Life (RUL) prediction + anomaly detection +
composite health scoring on every reading, and a live Streamlit dashboard renders each
machine's **digital twin** in real time.

Built around the ER&D domains LTTS actively sells and hires for: **Industrial AI, Edge AI,
IoT, Predictive Maintenance and Digital Twins** (see "Why this maps to LTTS" below).

No hardware, no Jetson/Pi/ESP32, no physical sensors — runs entirely on a normal
Windows/Linux laptop in VS Code.

---

## 1. Why this maps to LTTS (research summary)

LTTS's public engineering literature (ltts.com/services, ltts.com/resources, and its
2025 Altair Digital Twin CoE / Bentley iTwin / Microsoft partnerships) repeatedly
centers on the same stack this project implements:

- **Digital Twins for predictive maintenance and RUL** — LTTS explicitly describes a
  "Digital Twin for Line Operations" that does "predictive maintenance... using
  real-time and historical data" and an oil-rig digital twin built on 50+ ML algorithms.
- **Edge/cloud + AI for manufacturing** — LTTS's Manufacturing Engineering page lists
  "digital twins, digital threads, IoT, edge/cloud platforms, AI/GenAI" together as one
  connected stack, applied to predictive maintenance and OEE.
- **AI-enabled health monitoring** — LTTS markets "AI-enabled Aircraft Health Monitoring"
  that "transforms reactive maintenance into predictive solutions."
- **AI-Powered Vision/Edge Inspection** — LTTS's edge-intelligence product line for
  manufacturing defect detection uses the same "sensor → edge inference → decision"
  pattern this project implements for condition monitoring instead of vision.

This project translates that exact pattern — **stream → edge inference → health
score → digital twin → dashboard** — into a runnable, testable Python codebase.

---

## 2. Architecture

```
NASA C-MAPSS FD001            (genuine public sensor dataset, replayed as a live stream)
        │
        ▼
SensorStreamSimulator          round-robins cycle-by-cycle readings across a
(src/stream_simulator.py)      simulated fleet of N engines, like N edge devices
        │  SensorReading
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
```

**Design principle:** the UI never touches the ML models directly — it only reads
`DigitalTwinFleet` state. This mirrors how a real plant dashboard sits on top of a twin
service rather than embedding inference logic, and is what lets you swap the UI (web,
mobile, Grafana, PowerBI) without touching the ML layer.

**Train/serve consistency:** `src/features.py` is imported by BOTH the offline training
script and the real-time edge engine, so the exact same feature computation runs in
both places — a common real-world PdM bug (train/serve skew) is designed out.

---

## 3. Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation, subset FD001**
(Saxena & Goebel, 2008, NASA Prognostics Center of Excellence — the standard public
benchmark dataset for predictive maintenance / RUL research).

- 100 engines, run-to-failure trajectories, single operating condition, single fault
  mode (HPC degradation)
- 3 operational settings + 21 sensor channels (temperatures, pressures, speeds, fuel
  flow ratios, etc. — genuine simulated turbofan telemetry)
- 20,631 training cycles / 13,096 test cycles — small enough to ship in the repo (~5 MB
  as CSV), no external download needed at run time
- Already included under `data/` as clean CSVs (`train_FD001.csv`, `test_FD001.csv`,
  `RUL_FD001.csv`)

This is not a toy/synthetic dataset — it is the same dataset used in dozens of
published PdM papers and Kaggle competitions, which makes your results and metrics
independently verifiable and comparable to published baselines.

---

## 4. Folder structure

```
ltts-edge-ai-predictive-maintenance/
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
```

---

## 5. Setup & run instructions (VS Code, Windows or Linux)

```bash
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
```

Models are saved to `models_store/` after step 3 and are reused by everything else —
you only need to train once.

---

## 6. Testing

```bash
pip install pytest   # already in requirements.txt
python -m pytest tests/ -v
```

14 unit tests covering:
- Feature engineering correctness (rolling-mean math verified against a hand-computed
  toy signal, no-NaN guarantee, no duplicate feature columns)
- Health-scoring logic (bounds, monotonicity in RUL, correct HEALTHY/WARNING/CRITICAL
  classification for known-good and known-bad inputs)
- Streaming + digital twin integration (round-robin ordering, exhaustion, twin history
  length matches ticks processed, fleet summary matches active units)

All 14 currently pass (`14 passed in ~4s`).

---

## 7. Evaluation metrics (actual results on the FD001 test split)

Full report: [`evaluation/metrics_report.md`](evaluation/metrics_report.md) — generated
by `python run_pipeline.py evaluate`, not hand-written.

**RUL Regression (RandomForestRegressor, 250 trees)**

| Metric | All streamed cycles | Final cycle per engine (benchmark convention) |
|---|---|---|
| RMSE | 16.72 cycles | 17.96 cycles |
| MAE | 11.67 cycles | 12.81 cycles |
| R² | 0.632 | 0.799 |
| NASA PHM08 score | — | 1003.6 (lower is better) |

These are in the expected range for a RandomForest baseline on FD001 without deep
feature engineering (published deep-learning approaches on this exact benchmark
typically reach RMSE ≈ 12–16 with LSTMs/CNNs, so this is an honest, competitive
classical-ML baseline you can discuss both the strengths and the improvement path of).

**Anomaly Detection (IsolationForest)** — evaluated against the standard literature
proxy `RUL ≤ 30 cycles` = "impending failure window" (C-MAPSS has no explicit
fault-onset label, so this proxy is used and disclosed, not invented as ground truth):

| Metric | Value |
|---|---|
| Recall | 1.000 |
| Precision | 0.176 |
| % of cycles flagged anomalous | 14.4% |

Recall of 1.0 means the anomaly detector never misses an engine entering its final 30
cycles; the low precision is an honest and expected characteristic of unsupervised
anomaly detection tuned for high recall, and is exactly the kind of precision/recall
trade-off worth discussing in an interview (see Q&A below).

**Edge inference latency** (single reading, single CPU core, this dev laptop):

| Percentile | Latency |
|---|---|
| p50 | 38.5 ms |
| p95 | 41.5 ms |
| p99 | 45.1 ms |

---

## 8. Likely LTTS interview questions (and how this project answers them)

**System design / architecture**
- *"Walk me through your architecture end to end."* → stream → edge feature buffer →
  RUL regressor + anomaly detector → composite health score → digital twin state →
  dashboard; explain why the UI never calls the models directly.
- *"Why a digital twin layer instead of just calling the model from the UI?"* →
  decoupling: twin state is the single source of truth, multiple consumers (dashboard,
  alerting, CMMS integration) could read it without duplicating inference logic or
  re-running predictions.
- *"How would you actually deploy the 'edge' part on real hardware (Jetson/PLC gateway)?"*
  → talk about exporting the sklearn model via ONNX / skl2onnx, quantization, reducing
  rolling-window size, and why this project deliberately kept the model dependency-light
  (numpy/pandas/sklearn only) specifically so it's portable to a constrained runtime later.

**ML / data science**
- *"Why RandomForest and not an LSTM for RUL?"* → interpretability, fast CPU-only
  training/inference (no GPU needed, matches "no hardware" constraint), strong tabular
  baseline; be ready to discuss how an LSTM/1D-CNN over raw cycles could likely reduce
  RMSE further at the cost of latency and interpretability.
- *"Why clip RUL at 125 cycles?"* → piecewise-linear RUL target (Heimes 2008): engines
  behave near-identically while healthy, so training on unclipped decreasing RUL far
  from failure adds noise without signal.
- *"How did you validate you're not leaking test data into training?"* → RUL/anomaly
  models are fit only on `train_FD001`; the IsolationForest is fit only on each
  training engine's early "healthy" cycles; scaler is fit on train and applied to test.
- *"Why is anomaly precision only 17%?"* → contamination hyperparameter (0.05) trades
  recall for precision; discuss tuning `contamination`, adding a persistence rule
  (flag only if anomalous for k consecutive cycles) to cut false positives in a real
  deployment.
- *"What's the PHM08 score and why not just use RMSE?"* → asymmetric penalty: predicting
  MORE life than an engine actually has is penalized far more than predicting less,
  because a machine failing before its predicted maintenance window is a safety/cost
  issue — RMSE treats both directions equally, which doesn't match real maintenance risk.

**Software engineering**
- *"How did you keep training and serving consistent?"* → `src/features.py` is the
  single shared feature-computation module imported by both `train_models.py` and
  `edge_engine.py` — same functions, same feature order, same rolling-window logic.
- *"How is this tested?"* → pytest suite covering feature math, health-score bounds/
  monotonicity, and full streaming+twin integration; show `python -m pytest -v`.
- *"How would you scale this to thousands of real machines?"* → discuss swapping the
  in-process round-robin simulator for a real message broker (Kafka/MQTT), the
  DigitalTwinFleet for a proper twin service (e.g. Azure Digital Twins / AWS IoT
  TwinMaker) backed by a time-series DB, and the dashboard for a proper multi-user
  web frontend — while keeping the same edge-inference contract.

---

## 9. Truthful resume bullets

> Only use bullets that are true for what you actually ran/observed. Fill in your own
> RMSE/latency numbers if you retrain with different hyperparameters.

- Designed and built a modular Python **Edge-AI Predictive Maintenance system** with a
  live Digital Twin dashboard, simulating real-time multi-sensor IoT streaming, ML-based
  Remaining-Useful-Life prediction, unsupervised anomaly detection, and composite health
  scoring — architecture aligned to industrial predictive-maintenance and digital-twin
  patterns (stream → edge inference → twin state → dashboard).
- Trained a RandomForest regression model for turbofan Remaining-Useful-Life prediction
  on the NASA C-MAPSS FD001 dataset (100 engines, 20K+ cycles), achieving RMSE ≈ 17.96
  cycles and R² ≈ 0.80 on the standard final-cycle benchmark, plus an IsolationForest
  anomaly detector reaching 100% recall on an impending-failure proxy window.
- Built a real-time edge-inference engine maintaining bounded per-device rolling buffers
  and measuring live inference latency (p50 ≈ 39 ms, p95 ≈ 42 ms) to emulate constrained-
  hardware deployment characteristics without requiring physical devices.
- Implemented a live multi-engine Digital Twin state layer and a Streamlit dashboard with
  fleet overview, per-engine health gauges, RUL/health trend charts, and automatic
  WARNING/CRITICAL alerting, decoupled from the ML layer via a clean twin-state
  interface.
- Wrote a 14-test pytest suite covering feature-engineering correctness, health-scoring
  logic, and end-to-end stream/twin integration, and an automated evaluation script
  producing an industry-standard NASA PHM08 asymmetric scoring report.

---

## 10. Honesty notes (read before presenting this)

- All sensor data is a genuine replay of the public NASA C-MAPSS dataset — nothing is
  invented or randomly generated to look realistic.
- Anomaly detection precision/recall use an approximate proxy label (`RUL ≤ 30`)
  because this dataset has no explicit fault-onset annotation; this is disclosed in
  `evaluation/metrics_report.md`, not hidden.
- "Edge AI" here means the *software architecture and constraints* of edge inference
  (bounded memory, per-reading latency budget, lightweight dependency-light model) —
  it runs on a laptop CPU, not physical edge hardware, per the project's own
  no-hardware constraint.
