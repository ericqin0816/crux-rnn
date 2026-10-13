"""Action and whole-sequence metrics; copies are never statistical subjects."""
import numpy as np
import torch
from torch.nn import functional as F
from .provenance import state_hash


def summarize(logits, targets, changed_positions=()):
    logits = torch.as_tensor(logits)
    targets = torch.as_tensor(targets, dtype=torch.long)
    correct = (logits.argmax(-1) == targets).cpu().numpy()
    losses = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1), reduction="none").reshape(-1, 7).cpu().numpy()
    changed = np.asarray(changed_positions, dtype=int)
    shared = np.setdiff1d(np.arange(7), changed)
    def group(mask):
        return {"action_accuracy": float(correct[:, mask].mean()) if len(mask) else float("nan"),
                "action_count": int(correct.shape[0] * len(mask)),
                "cross_entropy": float(losses[:, mask].mean()) if len(mask) else float("nan")}
    rows = [{"group": "all", **group(np.arange(7)), "sequence_accuracy": float(correct.all(axis=1).mean())}]
    rows += [{"group": "position_{}".format(i + 1), **group([i]), "sequence_accuracy": float("nan")} for i in range(7)]
    rows += [{"group": name, **group(mask), "sequence_accuracy": float("nan")} for name, mask in (("shared", shared), ("differing", changed))]
    if len(targets) % 3 == 0:
        n = len(targets) // 3
        for i, letter in enumerate("ABC"):
            sl = slice(i * n, (i + 1) * n)
            rows.append({"group": letter, "action_accuracy": float(correct[sl].mean()),
                         "action_count": n * 7, "cross_entropy": float(losses[sl].mean()),
                         "sequence_accuracy": float(correct[sl].all(axis=1).mean())})
    return rows


def evaluate(model, x, y, changed_positions=(), batch_size=1024, main_groups=True):
    before = state_hash(model)
    previous_training = model.training
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunks.append(model(torch.as_tensor(x[start:start + batch_size], dtype=torch.float32)).cpu())
        rows = summarize(torch.cat(chunks), y, changed_positions)
    model.train(previous_training)
    assert state_hash(model) == before, "Evaluation changed model parameters"
    if not main_groups:
        rows = [row for row in rows if row["group"] not in "ABC"]
    return rows


def sustained_threshold(curve, threshold=0.9, checkpoints=3):
    streak = 0
    for i, (update, accuracy) in enumerate(curve):
        streak = streak + 1 if accuracy >= threshold else 0
        if streak >= checkpoints:
            return {"threshold_start": int(curve[i - checkpoints + 1][0]), "threshold_confirmed": int(update)}
    return {"threshold_start": "not_reached", "threshold_confirmed": "not_reached"}
