"""CruxRNN: curate existing verified outputs and prepare an unexecuted proposal."""
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from motor_rnn.data import balanced_assignment
from motor_rnn.experiment import verify_done
from motor_rnn.provenance import file_hash, implementation_hash, write_json

BASELINE = "2673c9ca11ca9fc701a1d13129f0cb371385ea70"
TITLE = "CruxRNN: Sequence Similarity, Practice Scheduling, and Interference in a Minimal Recurrent Neural Network"
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
review_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(source), str(destination))


historical_files = {"RUN_REPORT.md": "prior_RUN_REPORT.md", "reports/results.md": "prior_results.md",
                    "reports/plain_english.md": "prior_plain_english.md", "notebooks/walkthrough.ipynb": "prior_walkthrough.ipynb"}
history = Path("reports/history")
history.mkdir(parents=True, exist_ok=True)
historical_hashes = {}
for source, name in historical_files.items():
    content = subprocess.check_output(["git", "show", BASELINE + ":" + source])
    destination = history / name
    if destination.exists():
        assert destination.read_bytes() == content
    else:
        destination.write_bytes(content)
    historical_hashes[destination.as_posix()] = hashlib.sha256(content).hexdigest()

stages = []
for phase, configuration, publication in (("source_exact", "source_exact", "source_reproduction"), ("smoke", "smoke", "smoke"), ("pilot", "pilot", "pilot")):
    path = Path("configs", configuration + ".json")
    config = json.loads(path.read_text(encoding="utf-8"))
    root = Path(config["output_directory"])
    orders = json.loads((root / "blocked_order_assignment.json").read_text(encoding="utf-8"))
    target = Path("results", publication)
    target.mkdir(parents=True, exist_ok=True)
    cells = []
    for run_id in config["run_ids"]:
        for k in config["distances"]:
            directory = root / "runs" / "run_{:03d}_k{}".format(run_id, k)
            cfg = {**config, "run_id": run_id, "distance": k, "blocked_order": orders[str(k)][str(run_id)]}
            assert verify_done(directory, cfg), "Required cell is incomplete"
            done = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            copied_manifest = target / "manifests" / (directory.name + ".json")
            copy(directory / "manifest.json", copied_manifest)
            cells.append({"run_id": run_id, "distance": k, "directory": directory.as_posix(),
                          "manifest_in_repository": copied_manifest.as_posix(), "manifest_sha256": file_hash(copied_manifest),
                          "config_hash": done["config_hash"], "data_sha256": manifest["data_sha256"],
                          "complete_record_sha256": file_hash(directory / "complete.json"),
                          "artifact_hashes": done["files"], "elapsed_seconds": done["elapsed_seconds"],
                          "artifact_bytes": sum((directory / name).stat().st_size for name in done["files"])})
    analysis = root / "analysis"
    names = ["run_level.csv", "execution_times.csv", "verification.json", "design_verification.json"]
    if phase == "source_exact":
        names += ["summary.csv", "sequence_overlap_summary.csv", "sequence_overlap_totals.json"]
        for label in ("first_five", "full_100"):
            for file in (root / "comparison" / label).iterdir():
                if file.is_file():
                    copy(file, target / label / file.name)
    else:
        names += ["metrics.csv", "training.csv", "development.csv", "D_development.csv", "primary_summary.csv",
                  "paired_differences.csv", "secondary_summary.csv", "threshold_reached_counts.csv", "position_summary.csv"]
        for file in (root / "figures").iterdir():
            if file.suffix in (".png", ".pdf"):
                copy(file, target / "figures" / file.name)
    for name in names:
        source = analysis / name
        if source.exists():
            copy(source, target / ("raw_metrics.csv" if name == "metrics.csv" else name))
    copy(root / "phase_config.json", target / "phase_config.json")
    copy(root / "blocked_order_assignment.json", target / "blocked_order_assignment.json")
    copy(root / "last_execution.json", target / "original_last_execution.json")
    failures = (root / "failures.jsonl").read_text().splitlines() if (root / "failures.jsonl").exists() else []
    stage = {"project": "CruxRNN", "phase": phase, "generating_git_commit": None,
             "generating_git_commit_status": "Not recorded before historical execution; do not infer from later commits",
             "first_post_execution_snapshot_commit": BASELINE, "review_verification_code_commit": review_commit,
             "review_verified_at_utc": now, "original_started_at_utc": None, "original_finished_at_utc": None,
             "config_file": path.as_posix(), "config_file_sha256": file_hash(path), "seed_namespace": config["namespace"],
             "environment_sha256": file_hash("outputs/environment/system_info.json"),
             "pip_freeze_sha256": file_hash("outputs/environment/pip_freeze.txt"),
             "completed_cells": len(cells), "failure_records": len(failures), "excluded_cells": 0,
             "cell_runtime_seconds": sum(c["elapsed_seconds"] for c in cells),
             "raw_run_artifact_bytes": sum(c["artifact_bytes"] for c in cells), "cells": cells}
    write_json(target / "artifact_index.json", stage)
    stages.append({key: value for key, value in stage.items() if key != "cells"})

for name in ("system_info.json", "pip_freeze.txt", "benchmark.json"):
    copy(Path("outputs/environment", name), Path("provenance/environment", name))
copy("outputs/provenance/source_files.json", "provenance/author_source_files.json")
write_json("provenance/stages.json", {"project": "CruxRNN", "review_recorded_at_utc": now,
                                     "historical_snapshot_commit": BASELINE, "historical_document_sha256": historical_hashes,
                                     "numerical_implementation_sha256": implementation_hash(), "stages": stages,
                                     "prior_final_execution": {"completed_cells": 120, "namespace": "final_v1",
                                                               "config_sha256": file_hash("configs/final.json"),
                                                               "generating_git_commit": None,
                                                               "first_post_execution_snapshot_commit": BASELINE,
                                                               "outputs_preserved_at": "outputs/final", "new_final_execution_in_this_review": False}})

directory = Path("outputs/pilot/runs/run_000_k2")
with np.load(directory / "data.npz") as data, np.load(directory / "orders.npz") as orders:
    example = {"manifest": json.loads((directory / "manifest.json").read_text(encoding="utf-8")),
               "pair_checks": json.loads((directory / "pair_checks.json").read_text(encoding="utf-8")),
               "train_cue_shape": list(data["train_x"].shape), "final_probe_cue_shape": list(data["final_noisy_x"].shape),
               "training_example_cue": data["train_x"][0, 0].tolist(), "training_example_target": data["train_y"][0].tolist(),
               "blocked_first_ids": orders["blocked"][:12].tolist(), "interleaved_first_ids": orders["interleaved"][:12].tolist(),
               "example_id_multisets_verified_equal": bool(np.array_equal(np.sort(orders["blocked"]), np.sort(orders["interleaved"]))) }
write_json("results/pilot/example.json", example)

old = json.loads(Path("configs/final.json").read_text(encoding="utf-8"))
proposal = {**old, "project": "CruxRNN", "title": TITLE, "repository_name": "crux-rnn",
            "namespace": "cruxrnn_final_proposed_v1", "output_directory": "outputs/cruxrnn_final_proposed_v1"}
assert implementation_hash() == old["implementation_sha256"]
assert not Path(proposal["output_directory"]).exists(), "Proposed final directory must not have been executed"
path = Path("configs/final_proposed.json")
if path.exists():
    assert json.loads(path.read_text(encoding="utf-8")) == proposal, "Never change an existing proposal silently"
else:
    write_json(path, proposal)
checksum = file_hash(path)
marker = path.with_suffix(".sha256")
if marker.exists():
    assert marker.read_text().strip() == checksum
else:
    marker.write_text(checksum + "\n", encoding="utf-8")
assignment = balanced_assignment(proposal["run_ids"], proposal["distances"])
assignment_path = Path("configs/final_proposed_blocked_orders.json")
if assignment_path.exists():
    assert json.loads(assignment_path.read_text()) == assignment
else:
    write_json(assignment_path, assignment)
write_json("provenance/final_proposal.json", {"status": "proposed_only_not_executed", "project": "CruxRNN",
                                            "prepared_at_utc": now, "proposal_configuration": path.as_posix(),
                                            "config_sha256": checksum, "assignment_sha256": file_hash(assignment_path),
                                            "seed_namespace": proposal["namespace"], "historical_config_unchanged": file_hash("configs/final.json"),
                                            "approval_required": True, "final_data_generated": False,
                                            "scientific_parameters_changed": False,
                                            "non_scientific_differences": ["project identity fields", "new unused seed namespace", "new output directory"]})
print(json.dumps({"curated_completed_cells": {s["phase"]: s["completed_cells"] for s in stages},
                  "proposal_sha256": checksum, "proposed_final_started": False,
                  "historical_final_execution_disclosed": True}, indent=2))
