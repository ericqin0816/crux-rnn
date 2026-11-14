"""One-pass SGD, fixed development probes and atomically resumable checkpoints."""

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .evaluation import evaluate
from .reproducibility import state_hash


def optimizer_for(model: torch.nn.Module, learning_rate: float, momentum: float = 0):
    """Construct fresh SGD with the actual validated settings."""
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


def step(model, optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    model.train()
    loss = F.cross_entropy(model(x).reshape(-1, 4), y.reshape(-1))
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss; retain the failed cell")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train(
    model,
    optimizer,
    x: np.ndarray,
    y: np.ndarray,
    order: np.ndarray,
    checkpoint_path: Path,
    metadata: dict,
    batch_size: int = 20,
    eval_every: int = 25,
    probes=None,
    changed=(),
    resume: bool = False,
    stop_after=None,
) -> dict:
    """Train one pass; checkpoints retain optimizer/RNG/history and recipe guards."""
    if len(order) != len(x) or len(x) != len(y) or len(order) % batch_size:
        raise ValueError("Training arrays/order must align and divide into complete batches")
    if not np.array_equal(np.sort(order), np.arange(len(x))):
        raise ValueError("Training order must be a permutation of every example ID")
    checkpoint_path = Path(checkpoint_path)
    losses, history, update, elapsed = [], [], 0, 0.0
    device = next(model.parameters()).device
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError(f"Checkpoint exists; use --resume: {checkpoint_path}")
        saved = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        if saved["metadata"] != metadata:
            raise ValueError("Checkpoint metadata/recipe mismatch; no override is provided")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        if state_hash(model) != saved["state_hash"]:
            raise ValueError("Checkpoint model hash does not match")
        torch.set_rng_state(saved["torch_rng"].cpu())
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        losses, history, update, elapsed = (
            saved["losses"],
            saved["probes"],
            saved["update"],
            saved["elapsed_seconds"],
        )
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.long, device=device)
    total = len(order) // batch_size
    if update > total or len(losses) != update:
        raise ValueError("Checkpoint update/history exceeds the configured training budget")
    started = time.perf_counter()

    def probe(at):
        for split, (probe_x, probe_y) in (probes or {}).items():
            for row in evaluate(model, probe_x, probe_y, changed):
                if row["group"] in ("all", "A", "B", "C"):
                    history.append({"update": at, "split": split, **row})

    if update == 0 and not history:
        probe(0)
    for index in range(update, total):
        ids = torch.as_tensor(
            order[index * batch_size : (index + 1) * batch_size], dtype=torch.long, device=device
        )
        loss = step(model, optimizer, x_tensor[ids], y_tensor[ids])
        update = index + 1
        losses.append({"update": update, "cross_entropy": loss})
        if update % eval_every == 0 or update == total:
            probe(update)
        if update % 50 == 0 or update == total or update == stop_after:
            payload = {
                "metadata": metadata,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "state_hash": state_hash(model),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "update": update,
                "losses": losses,
                "probes": history,
                "elapsed_seconds": elapsed + time.perf_counter() - started,
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(payload, str(temporary))
            os.replace(str(temporary), str(checkpoint_path))
        if update == stop_after:
            break
    return {
        "update": update,
        "losses": losses,
        "probes": history,
        "elapsed_seconds": elapsed + time.perf_counter() - started,
    }
