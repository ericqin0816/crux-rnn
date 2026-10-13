"""CruxRNN approval gate and contemporaneous provenance for future executions."""
import datetime
import json
import subprocess
import uuid
from pathlib import Path
from motor_rnn.provenance import canonical_hash, file_hash, implementation_hash, write_json


def require_phase_authorization(config, gate):
    if config["phase"] == "final" and not gate.get("final_training_allowed", False):
        raise PermissionError("CruxRNN final execution is blocked pending explicit user approval. No final seeds were launched.")


def git_execution_identity():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError("Commit tracked implementation/config/documentation changes before running a new experiment phase.")
    return commit


def begin_execution(root, config, config_path, command):
    commit = git_execution_identity()
    environment = Path("outputs/environment/system_info.json")
    freeze = Path("outputs/environment/pip_freeze.txt")
    if not environment.exists() or not freeze.exists():
        raise FileNotFoundError("Capture environment information before an experiment phase")
    identity = uuid.uuid4().hex
    path = root / "execution_records" / (identity + ".json")
    record = {"project": "CruxRNN", "execution_id": identity, "phase": config["phase"],
              "code_git_commit": commit, "working_tree_clean_before_execution": True,
              "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "command": command, "config_file_sha256": file_hash(config_path),
              "config_canonical_sha256": canonical_hash(config), "implementation_sha256": implementation_hash(),
              "environment_sha256": file_hash(environment), "environment": json.loads(environment.read_text(encoding="utf-8")),
              "pip_freeze_sha256": file_hash(freeze), "seed_namespace": config["namespace"],
              "state": "started", "success_records": [], "failure_records": []}
    write_json(path, record)
    return path, record


def finish_execution(path, record, successes, failures, wall_seconds):
    record.update({"finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "success_records": successes, "failure_records": failures, "wall_seconds": wall_seconds,
                   "state": "failed" if failures else "completed"})
    write_json(path, record)
