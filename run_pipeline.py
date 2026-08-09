"""
Single CLI entrypoint for the whole project.

Usage:
    python run_pipeline.py train              # train RUL + anomaly models
    python run_pipeline.py evaluate            # generate evaluation/metrics_report.md
    python run_pipeline.py simulate            # headless console demo of the live stream
    python run_pipeline.py dashboard           # convenience wrapper that reminds you the
                                                # correct command (Streamlit apps must be
                                                # launched via `streamlit run`, not python)
"""
import argparse
import sys

from src import config, data_loader
from src.digital_twin import DigitalTwinFleet
from src.edge_engine import EdgeInferenceEngine
from src.stream_simulator import SensorStreamSimulator


def cmd_train():
    from src.train_models import train
    train()


def cmd_evaluate():
    from src.evaluate import main as eval_main
    eval_main()


def cmd_simulate(args):
    if not config.RUL_MODEL_PATH.exists():
        print("No trained model found. Run `python run_pipeline.py train` first.")
        sys.exit(1)

    print(f"Starting headless console simulation "
          f"(fleet_size={args.fleet_size}, steps={args.steps}) ...\n")

    _, test_df = data_loader.get_labeled_train_test()
    sim = SensorStreamSimulator(test_df, fleet_size=args.fleet_size,
                                 interval_sec=args.interval, seed=args.seed)
    engine = EdgeInferenceEngine()
    fleet = DigitalTwinFleet()

    tick = 0
    for batch in sim.run(max_steps=args.steps):
        tick += 1
        for reading in batch:
            pred = engine.predict(reading.unit_nr, reading.time_cycles,
                                   reading.row, reading.true_rul)
            twin = fleet.update(pred)
            marker = {"HEALTHY": " ", "WARNING": "!", "CRITICAL": "X"}[pred.status]
            print(f"[{marker}] tick={tick:04d} engine={pred.unit_nr:03d} "
                  f"cycle={pred.time_cycles:04d} RUL={pred.rul_pred:6.1f} "
                  f"health={pred.health_score:5.1f} status={pred.status:8s} "
                  f"anomaly={'Y' if pred.anomaly_flag else 'n'} "
                  f"latency={pred.latency_ms:5.1f}ms")

    print("\n--- Final fleet summary ---")
    for row in fleet.fleet_summary():
        print(row)


def cmd_dashboard():
    print("Streamlit apps must be launched directly, e.g.:\n")
    print("    streamlit run dashboard/app.py\n")


def main():
    parser = argparse.ArgumentParser(description="Edge-AI Predictive Maintenance pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Train RUL regressor + anomaly detector")
    sub.add_parser("evaluate", help="Run evaluation and write metrics_report.md")

    sim_p = sub.add_parser("simulate", help="Headless console live-stream demo")
    sim_p.add_argument("--fleet-size", type=int, default=config.FLEET_SIZE, dest="fleet_size")
    sim_p.add_argument("--steps", type=int, default=60)
    sim_p.add_argument("--interval", type=float, default=0.0)
    sim_p.add_argument("--seed", type=int, default=42)

    sub.add_parser("dashboard", help="Show the command to launch the Streamlit dashboard")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train()
    elif args.command == "evaluate":
        cmd_evaluate()
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "dashboard":
        cmd_dashboard()


if __name__ == "__main__":
    main()
