"""Small installed CLI; a demo never masquerades as a research experiment."""

import argparse
import json
import logging
import os
import uuid
from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cruxrnn", description="Reproducible sequence-similarity research"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Tiny end-to-end CPU demonstration; not research findings"
    )
    demo.add_argument("--output", default="outputs/v2/demo")
    demo.add_argument("--resume", action="store_true")
    run = commands.add_parser("run", help="Execute an original/expanded/exploratory v2 recipe")
    run.add_argument("--config", default="configs/v2/main.json")
    run.add_argument(
        "--experiment",
        default="all",
        choices=(
            "all",
            "main",
            "architecture",
            "capacity",
            "granularity",
            "forgetting",
            "noise",
            "dynamics",
        ),
    )
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--device", choices=("cpu", "cuda"))
    run.add_argument("--resume", action="store_true")
    cell = commands.add_parser("cell", help="Custom exploratory pair without editing source code")
    cell.add_argument("--config", default="configs/v2/smoke.json")
    cell.add_argument("--architecture", choices=("rnn", "gru", "lstm"), default="rnn")
    cell.add_argument("--hidden-size", type=int, default=7)
    cell.add_argument("--distance", type=int, default=4)
    cell.add_argument("--run-id", type=int, default=0)
    cell.add_argument("--namespace", default="cruxrnn_custom_exploratory")
    cell.add_argument("--output", default="outputs/v2/custom")
    cell.add_argument("--resume", action="store_true")
    for name in ("analyze", "figures", "verify"):
        subparser = commands.add_parser(name)
        subparser.add_argument("--config", default="configs/v2/main.json")
    original = commands.add_parser("original", help="Print the preserved v1 reproduction commands")
    original.add_argument(
        "--execute", action="store_true", help="Verify/resume the original frozen v1 experiment"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "demo":
        from .engine import run_study

        config = ExperimentConfig(
            namespace="cruxrnn_v2_demo",
            output_directory=args.output,
            repetitions=1,
            distances=(1, 7),
            pre_per_sequence=2,
            train_per_sequence=40,
            eval_per_sequence=2,
            d_examples=20,
            eval_every=1,
            bootstrap_samples=1000,
            exploratory_repetitions=1,
            noise_levels=(0.0, 0.1, 0.6),
        )
        record = run_study(config, workers=1, resume=args.resume, research=False)
        print(
            json.dumps(
                {
                    "label": "Illustrative pipeline demo; no scientific inference",
                    "completed_cells": len(record["successes"]),
                    "new_main_models": sum(
                        row["models"] for row in record["successes"] if row["status"] == "completed"
                    ),
                    "seconds": record["wall_seconds"],
                },
                indent=2,
            )
        )
    elif args.command == "run":
        config = load_config(args.config)
        if args.device:
            config = replace(config, device=args.device)
        if config.device == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from .engine import run_study

        run_study(config, args.experiment, args.workers, args.resume, args.config)
    elif args.command == "cell":
        from .engine import execute_cell
        from .reproducibility import configure_backend, execution_identity, utc_now, write_json

        if args.run_id < 0:
            parser.error("run-id must be nonnegative")
        original_config = load_config(args.config)
        if Path(args.output).resolve() == Path(original_config.output_directory).resolve():
            parser.error("Custom cells require a separate output directory")
        config = replace(
            original_config,
            namespace=args.namespace,
            output_directory=args.output,
            repetitions=1,
            distances=(args.distance,),
            exploratory_repetitions=0,
        )
        if config.device == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        configure_backend(config.device)
        identity = execution_identity(require_committed=False)
        job = {
            "run_id": args.run_id,
            "distance": args.distance,
            "architecture": args.architecture,
            "hidden_size": args.hidden_size,
            "study": "custom_exploratory",
            "schedules": ["blocked", "interleaved"],
        }
        started = utc_now()
        result = execute_cell(config, job, identity, args.resume)
        receipt = {
            **identity,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "role": "Custom exploratory cell, outside the prespecified main suite",
            "configuration": config.as_dict(),
            "job": job,
            "result": result,
        }
        write_json(
            Path(args.output) / "custom_execution_records" / (uuid.uuid4().hex + ".json"), receipt
        )
        print(json.dumps({"label": receipt["role"], **result}, indent=2))
    elif args.command == "original":
        command = "scripts/run_experiment.py --config configs/final_proposed.json --resume"
        print(command)
        if args.execute:
            import subprocess
            import sys

            subprocess.run([sys.executable] + command.split(), check=True)
    else:
        from .analysis import analyze, verify

        config = load_config(args.config)
        if args.command == "verify":
            print(json.dumps(verify(config), indent=2))
        elif args.command == "analyze":
            analyze(config)
        else:
            from .visualization import figures

            figures(config)
    if args.command == "demo":
        print(f"Saved complete data and checkpoints under {Path(args.output).resolve()}")
