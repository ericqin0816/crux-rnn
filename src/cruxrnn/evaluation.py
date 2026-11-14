"""Evaluation never trains; groups are explicit and invalid empty masks stay NaN."""

import numpy as np
import torch

from motor_rnn.metrics import summarize, sustained_threshold
from motor_rnn.provenance import state_hash

__all__ = ["evaluate", "sustained_threshold"]


def evaluate(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    changed=(),
    main_groups: bool = True,
    batch_size: int = 1024,
) -> list:
    """Score clean/noisy arrays; A/B/C slices require three equal-sized blocks."""
    if len(x) == 0 or len(x) != len(y) or batch_size < 1:
        raise ValueError("Evaluation requires nonempty aligned arrays and a positive batch size")
    if main_groups and len(y) % 3:
        raise ValueError("A/B/C evaluation requires three equally sized groups")
    before = state_hash(model)
    was_training = model.training
    device = next(model.parameters()).device
    model.eval()
    try:
        with torch.no_grad():
            chunks = [
                model(
                    torch.as_tensor(
                        x[start : start + batch_size], dtype=torch.float32, device=device
                    )
                ).cpu()
                for start in range(0, len(x), batch_size)
            ]
            rows = summarize(torch.cat(chunks), y, changed)
    finally:
        model.train(was_training)
    if state_hash(model) != before:
        raise RuntimeError("Evaluation changed model parameters")
    if not main_groups:
        rows = [row for row in rows if row["group"] not in ("A", "B", "C", "shared", "differing")]
    return rows
