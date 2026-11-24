from __future__ import annotations
import argparse
from benchmark.data_protocol import build as build_data_protocol
from benchmark.supervised import run_benchmark as run_supervised
from benchmark.supervised import tune_models
from benchmark.hybrid_experiments import run as run_hybrid_experiments
from benchmark.unknown_attack import run as run_unknown_attack
from benchmark.analysis import run_all as run_analyses
def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid IDS reproducible benchmark")
    parser.add_argument(
        "--phase",
        choices=[
            "data", "tune", "supervised", "hybrid", "unknown",
            "analysis", "all"
        ],
        default="all",
        help="Experiment phase to run.",
    )
    parser.add_argument("--config", default="benchmark/config.yaml")
    args = parser.parse_args()
    if args.phase in {"data", "all"}:
        build_data_protocol(args.config)
    if args.phase in {"tune", "all"}:
        tune_models(args.config)
    if args.phase in {"supervised", "all"}:
        run_supervised(args.config)
    if args.phase in {"hybrid", "all"}:
        run_hybrid_experiments(args.config)
    if args.phase in {"unknown", "all"}:
        run_unknown_attack(args.config)
    if args.phase in {"analysis", "all"}:
        run_analyses()
if __name__ == "__main__":
    main()
