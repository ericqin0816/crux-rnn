"""Stable hashes, atomic writes, and deterministic named random streams."""
import hashlib
import json
import os
from pathlib import Path

STREAMS = (
    "initialization", "pretraining_sequences", "pretraining_noise",
    "sequence_family_generation", "training_noise", "interleaving_permutation",
    "development_noise", "final_noise", "generalization_generation",
    "generalization_noise", "D_selection", "D_noise", "bootstrap_analysis",
)


def seed(namespace, run_id, distance, stream):
    label = "motor-rnn-v1|{}|{}|{}|{}".format(namespace, run_id, distance, stream)
    return int.from_bytes(hashlib.blake2b(label.encode(), digest_size=4).digest(), "little")


def seed_bundle(namespace, run_id, distance):
    return {name: seed(namespace, run_id, distance, name) for name in STREAMS}


def canonical_hash(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(temp), str(path))


def state_hash(model):
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def implementation_hash():
    root = Path(__file__).parent
    files = ["__init__.py", "provenance.py", "model.py", "data.py", "metrics.py", "training.py", "experiment.py"]
    return canonical_hash({name: file_hash(root / name) for name in files})
