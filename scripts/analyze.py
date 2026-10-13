"""Verify outputs; bootstrap simulation IDs; draw original Matplotlib figures."""
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
import torch
from motor_rnn.data import ORDERS
from motor_rnn.experiment import verify_done
from motor_rnn.provenance import seed, file_hash, write_json

COLORS = {"blocked": "#276AA0", "interleaved": "#D95C25"}
LABELS = {"blocked": "Blocked", "interleaved": "Interleaved"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 180, "axes.grid": True, "grid.alpha": .18})


def collect(config):
    root = Path(config["output_directory"])
    assignments = json.loads((root / "blocked_order_assignment.json").read_text())
    frames = {name: [] for name in ("run_level", "metrics", "training", "development", "D_development")}
    completed, missing, runtimes = [], [], []
    for run_id in config["run_ids"]:
        for k in config["distances"]:
            directory = root / "runs" / "run_{:03d}_k{}".format(run_id, k)
            cell = {**config, "run_id": run_id, "distance": k, "blocked_order": assignments[str(k)][str(run_id)]}
            if not verify_done(directory, cell):
                missing.append({"run_id": run_id, "distance": k})
                continue
            expected = 3 * config["train_per_sequence"] // 20
            for schedule in ("blocked", "interleaved"):
                checkpoint = torch.load(str(directory / (schedule + "_main.pt")), map_location="cpu")
                assert checkpoint["current_update"] == expected
                assert len(checkpoint["losses"]) == expected
                for key in ("rng_states", "config_hash", "data_manifest_hash", "source_commit", "run_id", "schedule", "blocked_order"):
                    assert key in checkpoint
                if config["mode"] == "controlled_similarity":
                    checkpoints = sorted(set([0, expected] + list(range(config["eval_every"], expected + 1, config["eval_every"]))))
                    assert len(checkpoint["probes"]) == len(checkpoints) * 8
                    assert sorted({r["update"] for r in checkpoint["probes"]}) == checkpoints
                    dc = torch.load(str(directory / (schedule + "_D.pt")), map_location="cpu")
                    assert dc["current_update"] == config["D_examples"] // 20
                    assert len(dc["probes"]) == (config["D_examples"] // 20 + 1) * 8
            for name in frames:
                path = directory / (name + ".csv")
                if path.exists():
                    df = pd.read_csv(path)
                    df["run_id"] = run_id
                    df["distance"] = k
                    frames[name].append(df)
            completed.append({"run_id": run_id, "distance": k})
            done = json.loads((directory / "complete.json").read_text())
            runtimes.append({"run_id": run_id, "distance": k, "seconds": done["elapsed_seconds"]})
    if not completed:
        raise RuntimeError("No verified completed runs")
    merged = {name: pd.concat(values, ignore_index=True) for name, values in frames.items() if values}
    target = root / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    for name, frame in merged.items():
        frame.to_csv(target / (name + ".csv"), index=False)
    pd.DataFrame(runtimes).to_csv(target / "execution_times.csv", index=False)
    failures = (root / "failures.jsonl").read_text().splitlines() if (root / "failures.jsonl").exists() else []
    write_json(target / "verification.json", {"completed": completed, "missing": missing, "failure_records": len(failures),
                                              "config_file": config, "all_completed_artifact_hashes_verified": True,
                                              "measured_cell_runtime_seconds": float(sum(r["seconds"] for r in runtimes))})
    return merged, target, missing


def bootstrap(config, runs, target):
    paired = runs.pivot(index=["run_id", "distance"], columns="schedule", values="final_noisy_action_accuracy").reset_index()
    assert not paired[["blocked", "interleaved"]].isna().any().any()
    paired["IP_minus_blocked_pp"] = 100 * (paired["interleaved"] - paired["blocked"])
    paired.to_csv(target / "paired_differences.csv", index=False)
    matrix = paired.pivot(index="run_id", columns="distance", values="IP_minus_blocked_pp").dropna()
    if config["phase"] == "final":
        assert matrix.shape == (30, 4), "Final inference requires all 30 complete replicate IDs"
    distances = matrix.columns.to_numpy(dtype=float)
    values = matrix.to_numpy()
    bs_seed = seed(config["namespace"], "analysis", 0, "bootstrap_analysis")
    rng = np.random.RandomState(bs_seed)
    indices = rng.randint(0, len(values), size=(10000, len(values)))
    resampled_means = values[indices].mean(1)
    x = distances - distances.mean()
    slopes = (resampled_means @ x) / (x @ x)
    slope = float((values.mean(0) @ x) / (x @ x))
    summary = []
    for i, k in enumerate(distances):
        lo, hi = np.percentile(resampled_means[:, i], [2.5, 97.5])
        summary.append({"distance": int(k), "paired_runs": len(values), "mean_difference_pp": float(values[:, i].mean()),
                        "median_difference_pp": float(np.median(values[:, i])), "ci_low_pp": float(lo), "ci_high_pp": float(hi),
                        "blocked_noisy_accuracy_pct": float(100 * runs[(runs.distance == k) & (runs.schedule == "blocked")].final_noisy_action_accuracy.mean()),
                        "IP_noisy_accuracy_pct": float(100 * runs[(runs.distance == k) & (runs.schedule == "interleaved")].final_noisy_action_accuracy.mean())})
    summary = pd.DataFrame(summary)
    summary.to_csv(target / "primary_summary.csv", index=False)
    np.savez_compressed(target / "bootstrap.npz", replicate_ids=matrix.index.to_numpy(), indices=indices,
                        categorical_means_pp=resampled_means, slopes_pp_per_hamming=slopes)
    pd.DataFrame({"run_id": matrix.index, "slope_pp_per_hamming": (values @ x) / (x @ x)}).to_csv(target / "replicate_slopes.csv", index=False)
    low, high = np.percentile(slopes, [2.5, 97.5])
    write_json(target / "trend.json", {"slope_pp_per_hamming": slope, "ci_low": float(low), "ci_high": float(high),
                                       "intercept_pp": float(values.mean() - slope * distances.mean()), "bootstrap_samples": 10000,
                                       "bootstrap_seed": bs_seed, "bootstrap_unit": "whole replicate IDs, with all four distances",
                                       "replicate_count": len(values), "scientific_inference": config["phase"] == "final"})
    secondary_columns = ["final_clean_action_accuracy", "final_noisy_sequence_accuracy", "final_cross_entropy",
                         "generalization_clean_accuracy", "generalization_noisy_accuracy", "forgetting_clean", "forgetting_noisy"]
    secondary = runs.groupby(["distance", "schedule"])[secondary_columns].mean()
    secondary.to_csv(target / "secondary_summary.csv")
    threshold = runs.groupby(["distance", "schedule"])["threshold_start"].apply(lambda vals: int(sum(str(v) != "not_reached" for v in vals)))
    threshold.to_csv(target / "threshold_reached_counts.csv", header=["reached_count"])
    return summary, paired


def savefig(fig, directory, name):
    fig.savefig(directory / (name + ".png"), bbox_inches="tight")
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def mean_curve(frame, schedule, group="all", split="noisy"):
    selected = frame[(frame.schedule == schedule) & (frame.group == group) & (frame.split == split)]
    return selected.groupby("update").action_accuracy.agg(["mean", "std", "count"])


def controlled_figures(config, frames, target, summary, paired):
    directory = target.parent / "figures"
    directory.mkdir(exist_ok=True)
    ks = config["distances"]
    title_suffix = " (pipeline smoke; not findings)" if config["phase"] == "smoke" else (" (pilot; diagnostic only)" if config["phase"] == "pilot" else "")
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True, sharey=True)
    for col, k in enumerate(ks):
        data = frames["development"][frames["development"].distance == k]
        for row, split in enumerate(("noisy", "clean")):
            ax = axes[row, col]
            for schedule in COLORS:
                curve = mean_curve(data, schedule, split=split)
                ax.plot(curve.index, 100 * curve["mean"], color=COLORS[schedule], label=LABELS[schedule])
                if curve["count"].max() > 1:
                    se = curve["std"].fillna(0) / np.sqrt(curve["count"])
                    ax.fill_between(curve.index, 100 * (curve["mean"] - se), 100 * (curve["mean"] + se), color=COLORS[schedule], alpha=.15)
            for boundary in (250, 500):
                if 3 * config["train_per_sequence"] // 20 >= boundary:
                    ax.axvline(boundary, color="gray", ls=":", lw=.8)
            ax.set_title("k={} | {} probes".format(k, split))
            ax.set_ylim(0, 102)
            if col == 0:
                ax.set_ylabel("Action accuracy (%)")
            if row == 1:
                ax.set_xlabel("Main SGD updates")
    axes[0, 0].legend()
    fig.suptitle("Fixed development probes, equally weighted A/B/C" + title_suffix)
    fig.tight_layout()
    savefig(fig, directory, "learning_curves")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    runs = frames["run_level"]
    for ax, metric, caption in zip(axes, ("final_noisy_action_accuracy", "final_clean_action_accuracy"), ("Fresh noisy final", "Clean final")):
        for schedule in COLORS:
            grouped = runs[runs.schedule == schedule].groupby("distance")[metric].mean()
            ax.plot(grouped.index, 100 * grouped, "o-", color=COLORS[schedule], label=LABELS[schedule])
        ax.set_title(caption)
        ax.set_xticks(ks)
        ax.set_xlabel("Pairwise Hamming distance")
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("Action accuracy (%)")
    axes[0].legend()
    fig.suptitle("Final trained-sequence performance" + title_suffix)
    fig.tight_layout()
    savefig(fig, directory, "final_accuracy")
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, k in enumerate(ks):
        vals = paired[paired.distance == k].IP_minus_blocked_pp.to_numpy()
        jitter = np.linspace(-.15, .15, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color="#798895", s=20, alpha=.6)
        row = summary[summary.distance == k].iloc[0]
        ax.errorbar(i, row.mean_difference_pp, yerr=[[row.mean_difference_pp - row.ci_low_pp], [row.ci_high_pp - row.mean_difference_pp]],
                    fmt="o", color="#8D327D", capsize=5, lw=2, zorder=4)
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_xticks(range(len(ks)), labels=ks)
    ax.set_xlabel("Pairwise Hamming distance (categorical)")
    ax.set_ylabel("Interleaved − blocked (percentage points)")
    ax.set_title("Fresh noisy final: run pairs and bootstrap 95% CIs" + title_suffix)
    fig.tight_layout()
    savefig(fig, directory, "paired_differences")
    fig, axes = plt.subplots(3, 4, figsize=(15, 8), sharex=True, sharey=True)
    d_frame = frames["D_development"]
    for col, k in enumerate(ks):
        data = d_frame[d_frame.distance == k]
        for row, letter in enumerate("ABC"):
            ax = axes[row, col]
            for schedule in COLORS:
                curve = mean_curve(data, schedule, group=letter)
                ax.plot(curve.index, 100 * curve["mean"], color=COLORS[schedule], label=LABELS[schedule])
            ax.set_title("k={} | sequence {}".format(k, letter))
            if col == 0:
                ax.set_ylabel("Noisy dev accuracy (%)")
            if row == 2:
                ax.set_xlabel("D SGD updates")
            ax.set_ylim(0, 102)
    axes[0, 0].legend()
    fig.suptitle("Old A/B/C during identical D training; D similarity observed" + title_suffix)
    fig.tight_layout()
    savefig(fig, directory, "D_forgetting")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    metric_frame = frames["metrics"]
    for ax, split, title in zip(axes, ("final_noisy", "final_clean"), ("Fresh noisy final", "Clean final")):
        selected = metric_frame[(metric_frame.stage == "before_D") & (metric_frame.split == split)]
        for schedule in COLORS:
            for group, style in (("shared", "--"), ("differing", "-")):
                grouped = selected[(selected.schedule == schedule) & (selected.group == group)].groupby("distance").action_accuracy.mean()
                ax.plot(grouped.index, 100 * (1 - grouped), "o" + style, color=COLORS[schedule], label=LABELS[schedule] + " / " + group)
        ax.set_title(title)
        ax.set_xticks(ks)
        ax.set_xlabel("Pairwise Hamming distance")
        ax.set_ylim(0, 100)
        ax.annotate("k=7: no shared positions", xy=(.04, .96), xycoords="axes fraction", ha="left", va="top", fontsize=8)
    axes[0].set_ylabel("Action error (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Errors at shared and differing positions" + title_suffix)
    fig.tight_layout()
    savefig(fig, directory, "position_errors")
    pos = metric_frame[(metric_frame.stage == "before_D") & metric_frame.group.isin(["shared", "differing"])]
    pos.groupby(["distance", "schedule", "split", "group"])[["action_accuracy", "action_count"]].mean().to_csv(target / "position_summary.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    frames, target, missing = collect(config)
    if config["mode"] == "controlled_similarity":
        summary, paired = bootstrap(config, frames["run_level"], target)
        controlled_figures(config, frames, target, summary, paired)
        print(summary.to_string(index=False))
        print((target / "trend.json").read_text())
    else:
        frames["run_level"].groupby("schedule").mean(numeric_only=True).to_csv(target / "summary.csv")
    print("Verified complete cells:", len(frames["run_level"]) // 2, "; missing:", len(missing))


if __name__ == "__main__":
    main()
