"""
Live Digital Twin Dashboard (Streamlit).

Run with:
streamlit run dashboard/app.py

Architecture on screen:
SensorStreamSimulator -> EdgeInferenceEngine -> DigitalTwinFleet -> this UI

The UI never talks to the ML models directly -- it only reads twin state,
mirroring how a real plant-floor dashboard would sit on top of a digital
twin service rather than embedding inference logic itself.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, data_loader
from src.digital_twin import DigitalTwinFleet
from src.edge_engine import EdgeInferenceEngine
from src.stream_simulator import SensorStreamSimulator


# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Edge-AI Predictive Maintenance | Live Digital Twin",
    layout="wide",
    page_icon="🛠️",
)

STATUS_COLOR = {
    "HEALTHY": "#2ecc71",
    "WARNING": "#f39c12",
    "CRITICAL": "#e74c3c",
}


# --------------------------------------------------------------------------
# Cached / session-persistent objects
# --------------------------------------------------------------------------

@st.cache_resource
def load_engine():
    return EdgeInferenceEngine()


@st.cache_data
def load_data():
    _, test_df = data_loader.get_labeled_train_test()
    return test_df


def init_session(fleet_size: int, seed: int):
    """Create a fresh simulator and digital twin fleet."""

    st.session_state.sim = SensorStreamSimulator(
        load_data(),
        fleet_size=fleet_size,
        interval_sec=0.0,
        seed=seed,
    )

    st.session_state.fleet = DigitalTwinFleet()

    st.session_state.running = False
    st.session_state.paused = False
    st.session_state.tick = 0


# --------------------------------------------------------------------------
# Initialize session
# --------------------------------------------------------------------------

if "sim" not in st.session_state:
    init_session(config.FLEET_SIZE, seed=42)

# Safety for existing Streamlit sessions

if "running" not in st.session_state:
    st.session_state.running = False

if "paused" not in st.session_state:
    st.session_state.paused = False

if "tick" not in st.session_state:
    st.session_state.tick = 0


edge_engine = load_engine()


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------

st.sidebar.title("🛠️ Control Panel")

st.sidebar.caption(
    "Edge-AI Predictive Maintenance + Live Digital Twin"
)

st.sidebar.markdown(
    "**Dataset:** NASA C-MAPSS FD001 "
    "(100 turbofan engines, replayed as a live fleet)"
)

fleet_size = st.sidebar.slider(
    "Fleet size (concurrent simulated engines)",
    2,
    10,
    config.FLEET_SIZE,
)

speed = st.sidebar.select_slider(
    "Stream speed",
    options=["Slow", "Medium", "Fast", "Max"],
    value="Fast",
)

seed = st.sidebar.number_input(
    "Random seed (choose which engines stream)",
    value=42,
    step=1,
)

speed_map = {
    "Slow": 0.6,
    "Medium": 0.3,
    "Fast": 0.1,
    "Max": 0.0,
}

interval = speed_map[speed]


# --------------------------------------------------------------------------
# Start / Pause / Resume / Restart controls
# --------------------------------------------------------------------------

# First row: Start and Pause

col1, col2 = st.sidebar.columns(2)

if col1.button(
    "▶ Start",
    use_container_width=True,
):
    st.session_state.running = True
    st.session_state.paused = False

if col2.button(
    "⏸ Pause",
    use_container_width=True,
):
    st.session_state.running = False
    st.session_state.paused = True
    st.rerun()


# Second row: Resume and Restart

col3, col4 = st.sidebar.columns(2)

if col3.button(
    "▶ Resume",
    use_container_width=True,
):
    st.session_state.running = True
    st.session_state.paused = False

if col4.button(
    "🔄 Restart",
    use_container_width=True,
):
    init_session(fleet_size, seed)
    st.rerun()


# --------------------------------------------------------------------------
# Simulation controls
# --------------------------------------------------------------------------

steps_per_click = st.sidebar.slider(
    "Cycles advanced per refresh",
    1,
    10,
    1,
)


# --------------------------------------------------------------------------
# Visible simulation status
# --------------------------------------------------------------------------

if st.session_state.running:
    st.sidebar.success("🟢 Simulation Running")

elif st.session_state.paused:
    st.sidebar.warning("🟡 Simulation Paused")

else:
    st.sidebar.info("🔵 Simulation Ready")

st.sidebar.divider()

st.sidebar.markdown(
    "**Pipeline:** Sensor Stream → Edge Feature Buffer → "
    "RUL Regressor + Isolation Forest Anomaly Detector → "
    "Composite Health Score → Digital Twin"
)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("Edge-AI Predictive Maintenance — Live Digital Twin")

st.caption(
    "Software-only simulation: replays genuine NASA C-MAPSS turbofan sensor "
    "trajectories as a live streaming fleet, runs on-the-fly ML inference "
    "(RUL + anomaly detection), and mirrors each engine's state in a digital twin."
)

fleet: DigitalTwinFleet = st.session_state.fleet
sim: SensorStreamSimulator = st.session_state.sim


# --------------------------------------------------------------------------
# Advance the simulation
# --------------------------------------------------------------------------

if st.session_state.running and not sim.is_exhausted():

    for _ in range(steps_per_click):

        if sim.is_exhausted():
            break

        batch = sim.step()

        for reading in batch:

            pred = edge_engine.predict(
                reading.unit_nr,
                reading.time_cycles,
                reading.row,
                reading.true_rul,
            )

            fleet.update(pred)

            # --------------------------------------------------------------
            # TEMPORARY ANOMALY DEBUG OUTPUT
            # --------------------------------------------------------------
            print(
                f"Engine {pred.unit_nr} | "
                f"Cycle {pred.time_cycles} | "
                f"Anomaly Score {pred.anomaly_raw} | "
                f"Anomaly {pred.anomaly_flag}"
            )

        st.session_state.tick += 1


# --------------------------------------------------------------------------
# Fleet overview
# --------------------------------------------------------------------------

summary = fleet.fleet_summary()

top1, top2, top3, top4 = st.columns(4)

n_critical = sum(
    1 for r in summary
    if r["Status"] == "CRITICAL"
)

n_warning = sum(
    1 for r in summary
    if r["Status"] == "WARNING"
)

n_healthy = sum(
    1 for r in summary
    if r["Status"] == "HEALTHY"
)

avg_latency = (
    sum(r["Latency (ms)"] for r in summary) / len(summary)
) if summary else 0.0


top1.metric(
    "Engines streaming",
    len(sim.unit_ids),
)

top2.metric(
    "Critical / Warning",
    f"{n_critical} / {n_warning}",
)

top3.metric(
    "Healthy",
    n_healthy,
)

top4.metric(
    "Avg edge inference latency",
    f"{avg_latency:.1f} ms",
)


# --------------------------------------------------------------------------
# Fleet table
# --------------------------------------------------------------------------

st.subheader("Fleet Overview")

if summary:

    fleet_df = pd.DataFrame(summary)

    def _style_status(val):

        color = STATUS_COLOR.get(
            val,
            "#999999",
        )

        return (
            f"background-color: {color}22; "
            f"color: {color}; "
            f"font-weight: 600;"
        )

    st.dataframe(
        fleet_df.style.map(
            _style_status,
            subset=["Status"],
        ),
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Press ▶ Start in the sidebar to begin "
        "streaming simulated sensor data."
    )


st.divider()


# --------------------------------------------------------------------------
# Per-engine digital twin detail
# --------------------------------------------------------------------------

if summary:

    ids = fleet.all_units()

    selected = st.selectbox(
        "Inspect engine (digital twin detail)",
        ids,
        format_func=lambda x: f"Engine {x}",
    )

    twin = fleet.get(selected)
    latest = twin.latest


    # ----------------------------------------------------------------------
    # Status and metrics
    # ----------------------------------------------------------------------

    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:

        if latest.status == "HEALTHY":
            status_color = "green"

        elif latest.status == "WARNING":
            status_color = "orange"

        else:
            status_color = "red"

        st.markdown(
            f"### Status: :{status_color}[{latest.status}]"
        )

        st.metric(
            "Health Score",
            f"{latest.health_score} / 100",
        )

        st.metric(
            "Predicted RUL",
            f"{latest.rul_pred} cycles",
        )

        if latest.true_rul is not None:

            st.caption(
                "Ground-truth RUL (for evaluation only): "
                f"{latest.true_rul:.0f} cycles"
            )

        st.metric(
            "Anomaly flag",
            "⚠️ YES"
            if latest.anomaly_flag
            else "✅ no",
        )


    # ----------------------------------------------------------------------
    # Health gauge
    # ----------------------------------------------------------------------

    with c2:

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=latest.health_score,
                title={
                    "text": "Health Index"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "color": STATUS_COLOR.get(
                            latest.status,
                            "#3498db",
                        )
                    },
                    "steps": [
                        {
                            "range": [
                                0,
                                config.HEALTH_CRITICAL_THRESHOLD,
                            ],
                            "color": "rgba(231, 76, 60, 0.2)",
                        },
                        {
                            "range": [
                                config.HEALTH_CRITICAL_THRESHOLD,
                                config.HEALTH_WARNING_THRESHOLD,
                            ],
                            "color": "rgba(243, 156, 18, 0.2)",
                        },
                        {
                            "range": [
                                config.HEALTH_WARNING_THRESHOLD,
                                100,
                            ],
                            "color": "rgba(46, 204, 113, 0.2)",
                        },
                    ],
                },
            )
        )

        gauge.update_layout(
            height=260,
            margin=dict(
                l=20,
                r=20,
                t=40,
                b=10,
            ),
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
        )


    # ----------------------------------------------------------------------
    # Recent alerts
    # ----------------------------------------------------------------------

    with c3:

        st.markdown("**Recent Alerts**")

        if twin.alerts:

            for alert in twin.alerts:
                st.write("🔔", alert)

        else:

            st.write(
                "No alerts yet — engine has stayed within healthy range."
            )


    # ----------------------------------------------------------------------
    # Historical charts
    # ----------------------------------------------------------------------

    hist = list(twin.history)

    if len(hist) > 1:

        cycles = [
            h.time_cycles
            for h in hist
        ]

        rul_pred = [
            h.rul_pred
            for h in hist
        ]

        rul_true = [
            h.true_rul
            for h in hist
        ]

        health = [
            h.health_score
            for h in hist
        ]


        chart_col1, chart_col2 = st.columns(2)


        # ------------------------------------------------------------------
        # RUL chart
        # ------------------------------------------------------------------

        with chart_col1:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=cycles,
                    y=rul_pred,
                    name="Predicted RUL",
                    line=dict(
                        color="#3498db"
                    ),
                )
            )

            if any(
                r is not None
                for r in rul_true
            ):

                fig.add_trace(
                    go.Scatter(
                        x=cycles,
                        y=rul_true,
                        name="True RUL",
                        line=dict(
                            color="#95a5a6",
                            dash="dot",
                        )
                    )
                )

            fig.update_layout(
                title="Remaining Useful Life over time",
                height=320,
                xaxis_title="Cycle",
                yaxis_title="RUL (cycles)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ------------------------------------------------------------------
        # Health chart
        # ------------------------------------------------------------------

        with chart_col2:

            fig2 = go.Figure()

            fig2.add_trace(
                go.Scatter(
                    x=cycles,
                    y=health,
                    name="Health Score",
                    line=dict(
                        color="#9b59b6"
                    ),
                    fill="tozeroy",
                )
            )

            fig2.add_hline(
                y=config.HEALTH_WARNING_THRESHOLD,
                line_dash="dash",
                line_color="orange",
            )

            fig2.add_hline(
                y=config.HEALTH_CRITICAL_THRESHOLD,
                line_dash="dash",
                line_color="red",
            )

            fig2.update_layout(
                title="Composite Health Score over time",
                height=320,
                xaxis_title="Cycle",
                yaxis_title="Health (0-100)",
            )

            st.plotly_chart(
                fig2,
                use_container_width=True,
            )


        # ------------------------------------------------------------------
        # Live sensor trends
        # ------------------------------------------------------------------

        st.markdown(
            "**Live sensor trends (key sensors)**"
        )

        sensor_cols = st.columns(3)

        key_sensors = [
            "sensor_2",
            "sensor_4",
            "sensor_11",
        ]

        for i, sensor_name in enumerate(key_sensors):

            vals = [
                h.sensors.get(sensor_name)
                for h in hist
            ]

            fig3 = go.Figure()

            fig3.add_trace(
                go.Scatter(
                    x=cycles,
                    y=vals,
                    line=dict(
                        color="#16a085"
                    ),
                )
            )

            fig3.update_layout(
                title=sensor_name,
                height=220,
                margin=dict(
                    l=10,
                    r=10,
                    t=40,
                    b=10,
                ),
            )

            sensor_cols[i].plotly_chart(
                fig3,
                use_container_width=True,
            )


# --------------------------------------------------------------------------
# Auto-refresh while running
# --------------------------------------------------------------------------

if st.session_state.running and not sim.is_exhausted():

    if interval > 0:
        time.sleep(interval)

    st.rerun()


elif st.session_state.running and sim.is_exhausted():

    st.success(
        "Stream complete — all simulated engines "
        "reached end of trajectory."
    )

    st.session_state.running = False
    st.session_state.paused = False