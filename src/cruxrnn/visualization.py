"""Original research figures derived from verified, saved seed-level outputs."""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import ExperimentConfig
from .engine import cell_directory, make_jobs
from .models import SequenceModel, parameter_count
from .reproducibility import file_hash, utc_now, write_json

COLORS = {
    "blocked": "#0072B2",
    "interleaved": "#D55E00",
    "large_blocks": "#009E73",
    "small_blocks": "#CC79A7",
    "cyclic": "#8A6D00",
}
LABELS = {
    "blocked": "Blocked",
    "interleaved": "Shuffled IP",
    "large_blocks": "Blocks of 1000",
    "small_blocks": "Blocks of 100",
    "cyclic": "Trial cycling",
}


def save(fig, directory: Path, name: str) -> None:
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def _box(axis, x, y, text, width=1.8, color="#E4EFF5"):
    axis.add_patch(
        FancyBboxPatch(
            (x, y), width, 0.65, boxstyle="round,pad=0.1", facecolor=color, edgecolor="#304B5B"
        )
    )
    axis.text(x + width / 2, y + 0.325, text, ha="center", va="center", fontsize=10)


def _arrow(axis, first, last, curve=0):
    axis.add_patch(
        FancyArrowPatch(
            first,
            last,
            arrowstyle="-|>",
            mutation_scale=12,
            connectionstyle=f"arc3,rad={curve}",
            color="#304B5B",
        )
    )


def diagrams(config: ExperimentConfig, directory: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 3.5))
    axis.set(xlim=(-0.3, 12), ylim=(-0.7, 2.7))
    axis.axis("off")
    _box(axis, 0, 0.8, "10 unique bases\nPretrain once")
    _box(axis, 3, 1.5, "Clone: blocked")
    _box(axis, 3, 0.1, "Clone: shuffled IP", color="#F8E7DC")
    _box(axis, 6, 0.8, "Fresh held-out cues\nBefore D")
    _box(axis, 9, 0.8, "Identical D training\nScore old A/B/C", width=2.3)
    for endpoint in ((3, 1.8), (3, 0.4)):
        _arrow(axis, (1.9, 1.1), endpoint)
    for start in ((4.9, 1.8), (4.9, 0.4)):
        _arrow(axis, start, (6, 1.1))
    _arrow(axis, (7.9, 1.1), (9, 1.1))
    axis.text(
        6,
        -0.4,
        "Within a replicate: same bases, noisy examples, initial weights and update budget; order differs.",
        ha="center",
        fontsize=10,
    )
    axis.set_title("Paired experimental design · one independent family per run × distance")
    save(fig, directory, "experimental_design")
    fig, axis = plt.subplots(figsize=(11, 3.6))
    axis.set(xlim=(-0.3, 10.4), ylim=(-0.5, 2.6))
    axis.axis("off")
    _box(axis, 0, 0.7, "Complete cue\n7 features")
    _box(axis, 3, 0.7, "Recurrent state\nh hidden units")
    _box(axis, 6, 0.7, "Linear readout\n4 raw logits")
    _arrow(axis, (1.9, 1.02), (3, 1.02))
    _arrow(axis, (4.9, 1.02), (6, 1.02))
    _arrow(axis, (7, 0.65), (3.8, 0.65), 0.6)
    axis.text(5.5, -0.2, "Linear 4→7 feedback; repeat for seven action outputs", ha="center")
    _arrow(axis, (3.2, 1.45), (4.6, 1.45), -0.8)
    axis.text(3.9, 2.05, "State resets for each example", ha="center", fontsize=9)
    axis.set_title("Complete-cue decoding architecture · RNN / GRU / LSTM recurrence")
    save(fig, directory, "model_architecture")
    jobs = make_jobs(config, "main")
    first = next(job for job in jobs if len(job["schedules"]) == 5)
    with np.load(cell_directory(config, first) / "orders.npz") as archive:
        fig, axes = plt.subplots(5, 1, figsize=(11, 5.2), sharex=True)
        family_colors = np.array([[0, 0.45, 0.7], [0.9, 0.45, 0], [0, 0.65, 0.45]])
        for axis, schedule in zip(axes, first["schedules"]):
            families = archive[schedule] // config.train_per_sequence
            batches = families.reshape(-1, config.batch_size)
            proportions = np.stack([(batches == label).mean(1) for label in range(3)], axis=1)
            axis.imshow(
                (proportions @ family_colors)[None, :, :],
                aspect="auto",
                extent=(0, len(batches), 0, 1),
            )
            axis.set_yticks([])
            axis.set_ylabel(LABELS[schedule], rotation=0, ha="right", va="center", fontsize=9)
        axes[-1].set_xlabel("SGD update (color encodes actual A/B/C minibatch proportions)")
        fig.suptitle("Recorded schedules · identical example IDs and exposure, different ordering")
        fig.tight_layout()
        save(fig, directory, "practice_schedules")
    fig, axes = plt.subplots(1, len(config.distances), figsize=(15, 3.2), squeeze=False)
    for axis, k in zip(axes[0], config.distances):
        job = next(job for job in jobs if job["distance"] == k and job["run_id"] == 0)
        manifest = json.loads((cell_directory(config, job) / "manifest.json").read_text())
        bases = np.array(
            [manifest["main_sequences"][letter]["actions_0_based"] for letter in "ABC"]
        )
        axis.imshow(bases, vmin=0, vmax=3, cmap="viridis", aspect="equal")
        for row in range(3):
            for col in range(7):
                axis.text(
                    col,
                    row,
                    str(bases[row, col]),
                    ha="center",
                    va="center",
                    color="white" if bases[row, col] < 2 else "black",
                    fontsize=8,
                )
        axis.set_title(f"k={k}")
        axis.set_xticks([])
        axis.set_yticks(range(3), labels=list("ABC"))
    fig.suptitle("Recorded run-0 sequence families · all three pairwise distances equal k")
    fig.tight_layout()
    save(fig, directory, "sequence_similarity")


def figures(config: ExperimentConfig) -> None:
    """Generate twelve PNG/SVG figure types; uncertainty bands are labeled SE."""
    analysis = Path(config.output_directory) / "analysis"
    verification = json.loads((analysis / "verification.json").read_text())
    if verification["missing_cells"]:
        raise ValueError("Figures require a complete verified dataset")
    directory = Path(config.output_directory) / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "svg.fonttype": "none",
        }
    )
    diagrams(config, directory)
    main = pd.read_csv(analysis / "main_summary.csv")
    paired = pd.read_csv(analysis / "paired_differences.csv")
    runs = pd.read_csv(analysis / "run_level.csv")
    effects = pd.read_csv(analysis / "variant_effects.csv")
    variants = pd.read_csv(analysis / "variants.csv")
    curves = pd.read_csv(analysis / "learning_curves.csv")
    primary = runs[(runs.architecture == "rnn") & (runs.hidden_size == 7)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, metric, title in zip(
        axes, ("action_accuracy", "clean_accuracy"), ("Fresh noisy final", "Clean final")
    ):
        for schedule in ("blocked", "interleaved"):
            group = (
                primary[primary.schedule == schedule]
                .groupby("distance")[metric]
                .agg(["mean", "std", "count"])
            )
            axis.errorbar(
                group.index,
                100 * group["mean"],
                yerr=100 * group["std"] / np.sqrt(group["count"]),
                fmt="o-",
                color=COLORS[schedule],
                label=LABELS[schedule],
                capsize=3,
            )
        axis.set(
            title=title, xlabel="Pairwise Hamming distance", xticks=config.distances, ylim=(0, 102)
        )
    axes[0].set_ylabel("Action accuracy (%)")
    axes[0].legend()
    fig.suptitle("Expanded RNN · means ± SE across independent IDs")
    fig.tight_layout()
    save(fig, directory, "main_accuracy")
    ks = [k for k in (1, 2, 4, 7) if k in config.distances]
    fig, axes = plt.subplots(2, len(ks), figsize=(14, 6), sharex=True, sharey=True, squeeze=False)
    for column, k in enumerate(ks):
        for row, split in enumerate(("noisy", "clean")):
            axis = axes[row, column]
            part = curves[
                (curves.architecture == "rnn")
                & (curves.hidden_size == 7)
                & (curves.distance == k)
                & (curves.stage == "main")
                & (curves.split == split)
                & (curves.group == "all")
            ]
            for schedule in ("blocked", "interleaved"):
                series = part[part.schedule == schedule]
                axis.plot(
                    series["update"],
                    100 * series["mean"],
                    color=COLORS[schedule],
                    label=LABELS[schedule],
                )
                axis.fill_between(
                    series["update"],
                    100 * (series["mean"] - series.se),
                    100 * (series["mean"] + series.se),
                    color=COLORS[schedule],
                    alpha=0.16,
                )
            axis.set(title=f"k={k} · {split}", ylim=(0, 102))
            if row == 1:
                axis.set_xlabel("Main SGD update")
            if column == 0:
                axis.set_ylabel("Action accuracy (%)")
    axes[0, 0].legend()
    fig.suptitle("Fixed A/B/C development probes · means ± SE")
    fig.tight_layout()
    save(fig, directory, "learning_curves")
    fig, axis = plt.subplots(figsize=(10, 4.5))
    for index, k in enumerate(config.distances):
        values = paired[paired.distance == k].difference_pp
        axis.scatter(
            index + np.linspace(-0.18, 0.18, len(values)), values, s=11, color="#667883", alpha=0.45
        )
        value = main[main.distance == k].iloc[0]
        axis.errorbar(
            index,
            value.difference_pp,
            yerr=[
                [value.difference_pp - value.ci_low_pp],
                [value.ci_high_pp - value.difference_pp],
            ],
            fmt="o",
            color="#882255",
            capsize=5,
            lw=2,
        )
    axis.axhline(0, color="black", ls="--", lw=1)
    axis.set(
        xticks=range(len(config.distances)),
        xticklabels=config.distances,
        xlabel="Hamming distance",
        ylabel="IP − blocked (pp)",
        title="Every seed pair and pointwise bootstrap 95% CI",
    )
    fig.tight_layout()
    save(fig, directory, "paired_effects")
    fig, axis = plt.subplots(figsize=(9, 4.7))
    architecture_effects = effects[(effects.architecture != "rnn") | (effects.hidden_size == 7)]
    for (architecture, hidden), part in architecture_effects.groupby(
        ["architecture", "hidden_size"]
    ):
        count = parameter_count(SequenceModel(architecture, int(hidden)))
        axis.errorbar(
            part.distance,
            part.difference_pp,
            yerr=[part.difference_pp - part.ci_low_pp, part.ci_high_pp - part.difference_pp],
            marker="o",
            capsize=3,
            label=f"{architecture.upper()} h={hidden} ({count} params)",
        )
    axis.axhline(0, color="black", ls="--", lw=1)
    axis.set(
        xlabel="Hamming distance",
        ylabel="IP − blocked (pp)",
        title="Architecture and approximate parameter matching · matched subset",
    )
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, directory, "architecture_comparison")
    # Separate capacity curves so architecture figure stays legible.
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for hidden, part in effects[effects.architecture == "rnn"].groupby("hidden_size"):
        axis.errorbar(
            part.distance,
            part.difference_pp,
            yerr=[part.difference_pp - part.ci_low_pp, part.ci_high_pp - part.difference_pp],
            marker="o",
            capsize=3,
            label=f"h={hidden}",
        )
    axis.axhline(0, color="black", ls="--", lw=1)
    axis.set(
        xlabel="Hamming distance",
        ylabel="IP − blocked (pp)",
        title="Capacity sensitivity · same training budget, matched subset",
    )
    axis.legend()
    fig.tight_layout()
    save(fig, directory, "capacity_sensitivity")
    fig, axis = plt.subplots(figsize=(9, 4.5))
    part = variants[(variants.architecture == "rnn") & (variants.hidden_size == 7)]
    for schedule, series in part.groupby("schedule"):
        axis.plot(
            series.distance,
            100 * series.mean_accuracy,
            "o-",
            color=COLORS[schedule],
            label=LABELS[schedule],
        )
    axis.set(
        xlabel="Hamming distance",
        ylabel="Fresh-noisy action accuracy (%)",
        ylim=(0, 102),
        title="Practice granularity · same examples and update count",
    )
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, directory, "practice_granularity")
    fig, axes = plt.subplots(3, len(ks), figsize=(14, 8), sharex=True, sharey=True, squeeze=False)
    for column, k in enumerate(ks):
        for row, letter in enumerate("ABC"):
            axis = axes[row, column]
            part = curves[
                (curves.architecture == "rnn")
                & (curves.hidden_size == 7)
                & (curves.distance == k)
                & (curves.stage == "D")
                & (curves.split == "noisy")
                & (curves.group == letter)
            ]
            for schedule in ("blocked", "interleaved"):
                series = part[part.schedule == schedule]
                axis.plot(
                    series["update"],
                    100 * series["mean"],
                    color=COLORS[schedule],
                    label=LABELS[schedule],
                )
            axis.set(title=f"k={k} · {letter}", ylim=(0, 102))
            if row == 2:
                axis.set_xlabel("D SGD update")
            if column == 0:
                axis.set_ylabel("Old-sequence accuracy (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Interference from identical D practice · per-sequence development means")
    fig.tight_layout()
    save(fig, directory, "forgetting_interference")
    noise = pd.read_csv(analysis / "noise_summary.csv")
    fig, axes = plt.subplots(1, len(ks), figsize=(14, 3.6), sharey=True, squeeze=False)
    for axis, k in zip(axes[0], ks):
        part = noise[
            (noise.architecture == "rnn") & (noise.hidden_size == 7) & (noise.distance == k)
        ]
        for schedule in ("blocked", "interleaved"):
            series = part[part.schedule == schedule]
            axis.errorbar(
                series.noise_sd,
                100 * series["mean"],
                yerr=100 * series["std"] / np.sqrt(series["count"]),
                fmt="o-",
                color=COLORS[schedule],
                label=LABELS[schedule],
                capsize=3,
            )
        axis.set(title=f"k={k}", xlabel="Held-out cue noise SD", ylim=(0, 102))
    axes[0, 0].set_ylabel("Action accuracy (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Noise robustness · new evaluation sets; training noise fixed at .1 · ±SE")
    fig.tight_layout()
    save(fig, directory, "noise_robustness")
    write_json(
        directory / "figure_manifest.json",
        {
            "created_at_utc": utc_now(),
            "visualization_source_sha256": file_hash(__file__),
            "data_namespace": config.namespace,
            "data_source": "verified v2 outputs only",
            "figures": {
                path.name: file_hash(path)
                for path in sorted(directory.iterdir())
                if path.suffix in (".png", ".svg")
            },
        },
    )
    if config.namespace == "cruxrnn_v2_main_20261114":
        shutil.copytree(str(directory), "results/v2/figures", dirs_exist_ok=True)
