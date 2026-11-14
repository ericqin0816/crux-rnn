"""Named seeds, byte fingerprints and honest contemporaneous execution identity."""

import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from motor_rnn.provenance import (
    canonical_hash,
    file_hash,
    seed,
    seed_bundle,
    state_hash,
    write_json,
)

__all__ = [
    "canonical_hash",
    "file_hash",
    "seed",
    "seed_bundle",
    "state_hash",
    "write_json",
    "utc_now",
    "configure_backend",
    "set_seed",
    "execution_identity",
    "numeric_fingerprint",
]
NUMERICAL_FILES = (
    "config.py",
    "models.py",
    "sequences.py",
    "schedules.py",
    "training.py",
    "evaluation.py",
    "engine.py",
    "reproducibility.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_backend(device: str = "cpu") -> None:
    """One Torch thread per worker avoids oversubscription of tiny recurrent ops."""
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; select --device cpu")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def numeric_fingerprint() -> str:
    root = Path(__file__).parent
    from motor_rnn.provenance import implementation_hash

    return canonical_hash(
        {
            "v2": {name: file_hash(root / name) for name in NUMERICAL_FILES},
            "frozen_v1_helpers": implementation_hash(),
        }
    )


def execution_identity(config_path=None, require_committed: bool = True) -> dict:
    """Verify numerical bytes against Git; record unrelated documentation edits.

    A demo may run from a dirty tree and is explicitly labeled accordingly.
    Research requires numerical source, config and plan to be committed.
    """
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        if require_committed:
            raise RuntimeError("Research execution requires a versioned Git checkout") from error
        commit, status = None, ["Source checkout unavailable"]
    paths = [Path(__file__).parent / name for name in NUMERICAL_FILES]
    paths += list(Path("src/motor_rnn").glob("*.py"))
    if config_path:
        from .config import load_config

        paths += [Path(config_path), Path(load_config(str(config_path)).plan)]
    if require_committed:
        for path in paths:
            relative = path.resolve().relative_to(Path.cwd()).as_posix()
            committed = subprocess.check_output(["git", "show", commit + ":" + relative])
            # Git may normalize text line endings; numerical source is pinned LF.
            if committed.replace(b"\r\n", b"\n") != path.read_bytes().replace(b"\r\n", b"\n"):
                raise RuntimeError(
                    f"Commit numerical/config/plan changes before running: {relative}"
                )
    return {
        "git_commit": commit,
        "git_status": status,
        "working_tree_clean": not status,
        "numerical_source_committed": require_committed,
        "numeric_code_sha256": numeric_fingerprint(),
        "recorded_at_utc": utc_now(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cuda_available": torch.cuda.is_available(),
        "torch_threads_per_worker": 1,
        "torch_interop_threads_per_worker": 1,
    }
