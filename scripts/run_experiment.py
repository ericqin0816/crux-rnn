"""Run phases, retain failures, and skip only verified complete cells on resume."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from motor_rnn import SOURCE_COMMIT
from motor_rnn.data import ORDERS, balanced_assignment
from motor_rnn.experiment import execute_cell
from motor_rnn.provenance import file_hash, implementation_hash, write_json
from execution_provenance import begin_execution, finish_execution, require_phase_authorization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="Explicitly override checkpoint metadata; records a deviation")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gate_path = Path("provenance/execution_gate.json")
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {"final_training_allowed": False}
    require_phase_authorization(config, gate)
    assert config["source_commit"] == SOURCE_COMMIT
    if config["phase"] == "final":
        frozen = config_path.with_suffix(".sha256").read_text().strip()
        assert frozen == file_hash(config_path), "Frozen final config was altered"
        assert config["implementation_sha256"] == implementation_hash(), "Training implementation changed after freeze"
    if args.force:
        with open("reports/deviations.md", "a", encoding="utf-8") as f:
            f.write("\nExplicit --force requested for {}. Checkpoint metadata override enabled.\n".format(config_path))
    torch.set_num_threads(config.get("cpu_threads", 1))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(config["output_directory"])
    root.mkdir(parents=True, exist_ok=True)
    record_path, execution_record = begin_execution(root, config, config_path, sys.argv)
    ids = config["run_ids"]
    distances = config["distances"]
    if config["phase"] == "final":
        assignment = balanced_assignment(ids, distances)
    else:
        assignment = {str(k): {str(r): "ABC" if config["mode"] == "source_exact" else ORDERS[i % 6] for i, r in enumerate(ids)} for k in distances}
    saved_assignment = root / "blocked_order_assignment.json"
    if saved_assignment.exists():
        assert json.loads(saved_assignment.read_text()) == assignment, "Saved blocked order assignment changed"
    else:
        write_json(saved_assignment, assignment)
    frozen_config = root / "phase_config.json"
    if frozen_config.exists():
        assert json.loads(frozen_config.read_text()) == config, "Phase config changed"
    else:
        write_json(frozen_config, config)
    chosen_ids = ids[:args.limit] if args.limit is not None else ids
    begin = time.perf_counter()
    failed = 0
    successes, failures = [], []
    for run_id in chosen_ids:
        for distance in distances:
            cell_config = {**config, "run_id": run_id, "distance": distance, "blocked_order": assignment[str(distance)][str(run_id)]}
            directory = root / "runs" / "run_{:03d}_k{}".format(run_id, distance)
            try:
                status = execute_cell(cell_config, directory, args.resume, args.force)
                successes.append({**status, "result_directory": str(directory), "manifest_sha256": file_hash(directory / "manifest.json"),
                                  "completion_record_sha256": file_hash(directory / "complete.json")})
                print(json.dumps(status), flush=True)
                with open(root / "events.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(status) + "\n")
            except Exception:
                failed += 1
                failure = {"run_id": run_id, "distance": distance, "traceback": traceback.format_exc(),
                           "mechanical_failure": True, "exclusion": False}
                failures.append(failure)
                with open(root / "failures.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(failure) + "\n")
                print(json.dumps(failure), flush=True)
                finish_execution(record_path, execution_record, successes, failures, time.perf_counter() - begin)
                # Retain failed cell and stop; do not quietly change the design.
                raise
    wall_seconds = time.perf_counter() - begin
    finish_execution(record_path, execution_record, successes, failures, wall_seconds)
    write_json(root / "last_execution.json", {"command": sys.argv, "config_file_sha256": file_hash(config_path),
                                               "wall_seconds": wall_seconds, "failed_cells": failed,
                                               "requested_run_ids": chosen_ids, "distances": distances})


if __name__ == "__main__":
    main()
