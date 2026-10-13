"""SGD with atomic, hash-guarded checkpoints and exact restart state."""
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .metrics import evaluate
from .provenance import state_hash


def optimizer_for(model, learning_rate=0.02):
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)


def rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save_checkpoint(path, model, optimizer, update, metadata, losses, probes, elapsed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**metadata, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
               "current_update": update, "rng_states": rng_state(), "losses": losses,
               "probes": probes, "elapsed_seconds": elapsed, "state_hash": state_hash(model)}
    tmp = path.with_suffix(".tmp")
    torch.save(payload, str(tmp))
    os.replace(str(tmp), str(path))


def load_checkpoint(path, model, optimizer, metadata, force=False):
    payload = torch.load(str(path), map_location="cpu")
    for key in ("config_hash", "data_manifest_hash", "run_id", "distance", "schedule", "blocked_order", "source_commit", "stage"):
        if payload.get(key) != metadata.get(key) and not force:
            raise ValueError("Checkpoint mismatch for " + key)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    assert state_hash(model) == payload["state_hash"]
    restore_rng(payload["rng_states"])
    return payload


def step(model, optimizer, x, y):
    model.train()
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, 4), y.reshape(-1))
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train(model, optimizer, x, y, order, metadata, checkpoint_path,
          probes=None, changed=(), batch_size=20, eval_every=25,
          checkpoint_every=50, resume=False, force=False, stop_after=None):
    assert len(order) == len(x) == len(y)
    assert len(order) % batch_size == 0
    assert np.array_equal(np.sort(order), np.arange(len(x)))
    total = len(order) // batch_size
    update, losses, history, prior_elapsed = 0, [], [], 0.0
    if resume and Path(checkpoint_path).exists():
        saved = load_checkpoint(checkpoint_path, model, optimizer, metadata, force)
        update, losses, history, prior_elapsed = saved["current_update"], saved["losses"], saved["probes"], saved["elapsed_seconds"]
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.long)
    start = time.perf_counter()
    def probe(at):
        for split, (probe_x, probe_y) in (probes or {}).items():
            for row in evaluate(model, probe_x, probe_y, changed):
                if row["group"] in ("all", "A", "B", "C"):
                    history.append({"update": at, "split": split, **row})
    if update == 0 and not history:
        probe(0)
    for index in range(update, total):
        ids = order[index * batch_size:(index + 1) * batch_size]
        loss = step(model, optimizer, x_tensor[ids], y_tensor[ids])
        update = index + 1
        losses.append({"update": update, "training_cross_entropy": loss})
        if update % eval_every == 0 or update == total:
            probe(update)
        if update % checkpoint_every == 0 or update == total or update == stop_after:
            save_checkpoint(checkpoint_path, model, optimizer, update, metadata, losses, history,
                            prior_elapsed + time.perf_counter() - start)
        if update == stop_after:
            break
    return {"current_update": update, "losses": losses, "probes": history,
            "elapsed_seconds": prior_elapsed + time.perf_counter() - start}
