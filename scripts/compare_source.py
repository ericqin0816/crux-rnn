"""Read author arrays locally; publish shape/aggregate comparisons, no copies."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from motor_rnn.provenance import file_hash, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int)
parser.add_argument("--label", default="current")
args = parser.parse_args()
author = Path("external/author_repo/results_lr_0_02_save")
root = Path("outputs/source_exact")
dirs = sorted(p.parent for p in (root / "runs").glob("*/complete.json"))
if args.limit:
    dirs = dirs[:args.limit]
assert dirs
arrays = {}
for directory in dirs:
    with np.load(directory / "source_arrays.npz") as values:
        for key in values.files:
            arrays.setdefault(key, []).append(values[key])
arrays = {key: np.stack(value) for key, value in arrays.items()}
mapping = {"blocked_training": "loss_array_blocked_array.npy", "interleaved_training": "loss_array_random_array.npy",
           "blocked_retention": "loss_retention_array_blocked_array.npy", "interleaved_retention": "loss_retention_array_random_array.npy",
           "blocked_generalization": "loss_test_array_blocked_array.npy", "interleaved_generalization": "loss_test_array_random_array.npy",
           "pre_generalization": "loss_test_array_pre_array.npy"}
records, saved, comparisons = [], {}, []
for path in sorted(author.glob("*.npy")):
    a = np.load(path, allow_pickle=False)
    saved[path.name] = a
    records.append({"file": path.name, "shape": list(a.shape), "sha256": file_hash(path),
                    "mean": float(a.mean()), "finite": bool(np.isfinite(a).all())})
for name, filename in mapping.items():
    local, original = arrays[name], saved[filename]
    assert local.shape[1:] == original.shape[1:], "Per-run shape differs from reference"
    comparisons.append({"metric": name, "new_runs": local.shape[0], "author_runs": original.shape[0],
                        "new_shape": str(local.shape), "author_shape": str(original.shape),
                        "new_mean_loss": float(local.mean()), "author_mean_loss": float(original.mean())})
target = root / "comparison" / args.label
target.mkdir(parents=True, exist_ok=True)
pd.DataFrame(comparisons).to_csv(target / "aggregate_comparison.csv", index=False)
write_json(target / "author_array_audit.json", records)
directions = {}
for label, a in (("new", arrays), ("author", {name: saved[filename] for name, filename in mapping.items()})):
    switches = [float(a["blocked_training"][:, index].mean() - a["blocked_training"][:, index - 1].mean()) for index in (250, 500)]
    directions[label] = {"blocked_training_loss_lower": bool(a["blocked_training"].mean() < a["interleaved_training"].mean()),
                         "blocked_switch_loss_increases": switches,
                         "IP_retention_loss_lower": bool(a["interleaved_retention"].mean() < a["blocked_retention"].mean()),
                         "IP_generalization_loss_lower": bool(a["interleaved_generalization"].mean() < a["blocked_generalization"].mean()),
                         "retention_loss_ABC_blocked": a["blocked_retention"].reshape(len(a["blocked_retention"]), 3, 5000).mean(axis=(0, 2)).tolist(),
                         "retention_loss_ABC_IP": a["interleaved_retention"].reshape(len(a["interleaved_retention"]), 3, 5000).mean(axis=(0, 2)).tolist()}
for kind in ("noisy", "pruned", "interf"):
    b = saved["loss_retention_{}_array_blocked.npy".format(kind)]
    ip = saved["loss_retention_{}_array_random.npy".format(kind)]
    directions["author"][kind + "_last_sequence_loss_increase"] = {"blocked": float((b[:, -1] - b[:, 0]).mean()), "IP": float((ip[:, -1] - ip[:, 0]).mean())}
directions["new"]["optional_vulnerability_rerun"] = False
write_json(target / "qualitative_directions.json", directions)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, label, values in ((axes[0], "New source-compatible (n={})".format(len(dirs)), arrays),
                          (axes[1], "Authors' saved (n=100)", {name: saved[filename] for name, filename in mapping.items()})):
    for name, caption, color in (("blocked_training", "Blocked", "#276AA0"), ("interleaved_training", "Interleaved", "#D95C25")):
        ax.plot(np.arange(1, 751), values[name].mean(0), label=caption, color=color)
    ax.set_title(label)
    ax.set_xlabel("SGD update")
    for boundary in (250, 500):
        ax.axvline(boundary, color="gray", ls=":")
axes[0].set_ylabel("Training cross-entropy")
axes[0].legend()
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(target / ("training_comparison." + suffix), dpi=180, bbox_inches="tight")
plt.close(fig)
print(pd.DataFrame(comparisons).to_string(index=False))
print(json.dumps(directions, indent=2))
