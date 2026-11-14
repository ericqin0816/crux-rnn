"""Verify saved experiments and aggregate seed-level results without pooling v1."""

import gzip
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from motor_rnn.data import assert_no_overlap, hamming

from .config import ExperimentConfig
from .engine import cell_directory, make_jobs, verify_complete
from .reproducibility import canonical_hash, file_hash, seed, seed_bundle, utc_now, write_json
from .schedules import make_order
from .statistics import holm, paired_summary, similarity_interaction

KEYS = ["architecture", "hidden_size", "distance", "schedule"]


def verify(config: ExperimentConfig) -> dict:
    """Audit every expected cell/data/order/checkpoint before conducting inference."""
    missing, completed, models, d_models, producing = [], 0, 0, 0, set()
    for job in make_jobs(config):
        directory = cell_directory(config, job)
        done = verify_complete(directory)
        if not done:
            missing.append(directory.name)
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        if canonical_hash(manifest["recipe"]["configuration"]) != canonical_hash(config.as_dict()):
            raise ValueError("Cell configuration differs from the planned study")
        expected = seed_bundle(config.namespace, job["run_id"], job["distance"])
        if any(manifest["seeds"][key] != value for key, value in expected.items()):
            raise ValueError("Saved seed assignments differ from the recipe")
        with np.load(directory / "data.npz") as archive, np.load(
            directory / "orders.npz"
        ) as orders:
            assert_no_overlap(
                {name: archive[name] for name in ("main_bases", "pre_bases", "gen_bases", "D_base")}
            )
            bases = archive["main_bases"]
            if [hamming(bases[i], bases[j]) for i, j in ((0, 1), (0, 2), (1, 2))] != [
                job["distance"]
            ] * 3:
                raise ValueError("Exact Hamming constraints were violated")
            if not np.array_equal(
                archive["train_y"], np.repeat(bases, config.train_per_sequence, axis=0)
            ):
                raise ValueError("Training targets were altered")
            if np.array_equal(archive["dev_noisy_x"], archive["final_noisy_x"]):
                raise ValueError("Development/final noise overlap")
            for schedule in job["schedules"]:
                saved_order = orders[schedule]
                if not np.array_equal(
                    np.sort(saved_order), np.arange(3 * config.train_per_sequence)
                ):
                    raise ValueError("Unequal schedule exposure")
                # Reconstruct the actual deterministic order independently.
                from .schedules import blocked_order

                expected_order = make_order(
                    config.train_per_sequence,
                    schedule,
                    blocked_order(job["run_id"], job["distance"]),
                    manifest["seeds"]["interleaving_permutation"],
                )
                if not np.array_equal(saved_order, expected_order):
                    raise ValueError("Schedule differs from its saved assignment")
        rows = pd.read_csv(directory / "run_level.csv")
        if len(rows) != len(job["schedules"]) or rows.initial_state_sha256.nunique() != 1:
            raise ValueError("Paired initializations or model counts differ")
        if not (rows.main_updates == 3 * config.train_per_sequence // config.batch_size).all():
            raise ValueError("Main update budget differs")
        if not (rows.d_updates == config.d_examples // config.batch_size).all():
            raise ValueError("D update budget differs")
        for schedule in job["schedules"]:
            for stage, updates in (
                ("main", 3 * config.train_per_sequence // config.batch_size),
                ("D", config.d_examples // config.batch_size),
            ):
                checkpoint = torch.load(
                    str(directory / f"{schedule}_{stage}.pt"),
                    map_location="cpu",
                    weights_only=False,
                )
                if checkpoint["update"] != updates or len(checkpoint["losses"]) != updates:
                    raise ValueError("Checkpoint update/history differs")
        metrics = pd.read_csv(directory / "metrics.csv")
        if job["distance"] == 7:
            shared = metrics[metrics.group == "shared"]
            if not shared.action_accuracy.isna().all() or not (shared.action_count == 0).all():
                raise ValueError("k=7 shared metrics must be NaN with zero count")
        for schedule in job["schedules"]:
            before = metrics[
                (metrics.schedule == schedule)
                & (metrics.stage == "before_D")
                & (metrics.split == "final_noisy")
            ]
            after = metrics[
                (metrics.schedule == schedule)
                & (metrics.stage == "after_D")
                & (metrics.split == "final_noisy")
            ]
            overall = before[before.group == "all"].action_accuracy.iloc[0]
            if not np.isclose(
                overall, before[before.group.isin(list("ABC"))].action_accuracy.mean()
            ):
                raise ValueError("A/B/C weighting differs")
            forgetting = overall - after[after.group == "all"].action_accuracy.iloc[0]
            if not np.isclose(forgetting, rows[rows.schedule == schedule].forgetting_noisy.iloc[0]):
                raise ValueError("Forgetting identity differs")
        completed += 1
        models += done["main_models"]
        d_models += done["D_adapted_clones"]
        producing.add(done["completing_git_commit"])
    failure_path = Path(config.output_directory) / "failures.jsonl"
    report = {
        "verified_at_utc": utc_now(),
        "namespace": config.namespace,
        "expected_cells": len(make_jobs(config)),
        "completed_cells": completed,
        "main_models": models,
        "D_adapted_clones": d_models,
        "missing_cells": missing,
        "failure_records": len(failure_path.read_text().splitlines())
        if failure_path.exists()
        else 0,
        "excluded_cells": 0,
        "producing_git_commits": sorted(producing),
        "config_sha256": canonical_hash(config.as_dict()),
        "all_completed_artifact_hashes_valid": True,
    }
    write_json(Path(config.output_directory) / "analysis/verification.json", report)
    return report


def _accumulate(accumulator, frame: pd.DataFrame, keys: list, column: str) -> None:
    for row in frame[keys + [column]].itertuples(index=False, name=None):
        value = row[-1]
        if not math.isfinite(value):
            continue
        item = accumulator.setdefault(row[:-1], [0.0, 0.0, 0])
        item[0] += value
        item[1] += value * value
        item[2] += 1


def _summary(accumulator, keys: list) -> pd.DataFrame:
    rows = []
    for key, (total, square, count) in sorted(accumulator.items()):
        variance = (
            max(0, (square - total * total / count) / (count - 1)) if count > 1 else float("nan")
        )
        rows.append(
            dict(
                zip(keys, key),
                mean=total / count,
                sd=math.sqrt(variance),
                n=count,
                se=math.sqrt(variance / count),
            )
        )
    return pd.DataFrame(rows)


def _pair(frame, metric="action_accuracy") -> pd.DataFrame:
    result = (
        frame[frame.schedule.isin(["blocked", "interleaved"])]
        .pivot(index=["run_id", "distance"], columns="schedule", values=metric)
        .reset_index()
    )
    if result[["blocked", "interleaved"]].isna().any().any():
        raise ValueError("A paired cell is missing; incomplete pairs cannot be dropped")
    result["difference_pp"] = 100 * (result.interleaved - result.blocked)
    return result


def analyze(config: ExperimentConfig) -> None:
    """Produce planned tables, corrected tests, seed distributions and curve SEs."""
    verification = verify(config)
    if verification["missing_cells"]:
        raise ValueError(
            "The study is incomplete; report missing cells before interpreting results"
        )
    if config.repetitions < 2:
        raise ValueError("A one-ID demo does not support inferential statistics")
    root = Path(config.output_directory)
    target = root / "analysis"
    run_frames, metric_frames, curves, losses = [], [], {}, {}
    curve_keys = KEYS + ["stage", "split", "group", "update"]
    loss_keys = KEYS + ["stage", "update"]
    with gzip.open(target / "manifests.jsonl.gz", "wt", encoding="utf-8") as manifests, gzip.open(
        target / "artifact_index.jsonl.gz", "wt", encoding="utf-8"
    ) as index:
        for job in make_jobs(config):
            directory = cell_directory(config, job)
            run_frames.append(pd.read_csv(directory / "run_level.csv"))
            metric_frames.append(pd.read_csv(directory / "metrics.csv"))
            _accumulate(
                curves, pd.read_csv(directory / "curves.csv"), curve_keys, "action_accuracy"
            )
            _accumulate(losses, pd.read_csv(directory / "losses.csv"), loss_keys, "cross_entropy")
            manifests.write(
                (directory / "manifest.json").read_text().strip().replace("\n", "") + "\n"
            )
            index.write(
                json.dumps(
                    {
                        "directory": directory.as_posix(),
                        "completion_sha256": file_hash(directory / "complete.json"),
                        "completion": json.loads((directory / "complete.json").read_text()),
                    }
                )
                + "\n"
            )
    runs = pd.concat(run_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    runs.to_csv(target / "run_level.csv", index=False)
    metrics.to_csv(target / "raw_metrics.csv.gz", index=False, compression="gzip")
    _summary(curves, curve_keys).to_csv(target / "learning_curves.csv", index=False)
    _summary(losses, loss_keys).to_csv(target / "training_loss.csv", index=False)
    primary = runs[(runs.architecture == "rnn") & (runs.hidden_size == 7)]
    paired = _pair(primary)
    paired.to_csv(target / "paired_differences.csv", index=False)
    rng = np.random.RandomState(seed(config.namespace, "analysis", 0, "bootstrap_analysis"))
    samples = config.bootstrap_samples
    summaries = []
    for distance, frame in paired.groupby("distance"):
        summaries.append(
            {"distance": distance, **paired_summary(frame.blocked, frame.interleaved, rng, samples)}
        )
    summary = pd.DataFrame(summaries)
    matrix = paired.pivot(index="run_id", columns="distance", values="difference_pp")
    if matrix.shape != (config.repetitions, len(config.distances)) or matrix.isna().any().any():
        raise ValueError("Primary matrix differs from the prespecified independent sample size")
    original_distances = [k for k in (1, 2, 4, 7) if k in matrix.columns]
    interactions = {}
    for label, distances in (
        ("original_four", original_distances),
        ("all_distances", list(matrix.columns)),
    ):
        result = similarity_interaction(matrix[distances].to_numpy(), distances, rng, samples)
        indices, slopes = result.pop("bootstrap_indices"), result.pop("bootstrap_slopes")
        np.savez_compressed(
            target / f"bootstrap_{label}.npz",
            indices=indices,
            slopes=slopes,
            replicate_ids=matrix.index.to_numpy(),
        )
        interactions[label] = result
    primary_mask = summary.distance.isin(original_distances)
    family = summary.loc[primary_mask, "p_sign_flip"].tolist() + [
        interactions["original_four"]["p_slope_sign_flip"],
        interactions["original_four"]["p_categorical_interaction"],
    ]
    adjusted = holm(family)
    summary.loc[primary_mask, "p_holm_primary_family"] = adjusted[:-2]
    interactions["original_four"]["p_slope_holm"] = float(adjusted[-2])
    interactions["original_four"]["p_categorical_holm"] = float(adjusted[-1])
    adjusted_all = holm(
        summary.p_sign_flip.tolist()
        + [
            interactions["all_distances"]["p_slope_sign_flip"],
            interactions["all_distances"]["p_categorical_interaction"],
        ]
    )
    summary["p_holm_all_distances_family"] = adjusted_all[:-2]
    interactions["all_distances"]["p_slope_holm"] = float(adjusted_all[-2])
    interactions["all_distances"]["p_categorical_holm"] = float(adjusted_all[-1])
    summary.to_csv(target / "main_summary.csv", index=False)
    write_json(target / "interactions.json", interactions)
    subset = runs[
        (runs.run_id < config.exploratory_repetitions) & runs.distance.isin(original_distances)
    ]
    variants = subset.groupby(KEYS).action_accuracy.agg(["mean", "std", "count"]).reset_index()
    variants.rename(columns={"mean": "mean_accuracy", "std": "sd_accuracy", "count": "n"}).to_csv(
        target / "variants.csv", index=False
    )
    effects = []
    for (arch, hidden), frame in subset.groupby(["architecture", "hidden_size"]):
        for distance, part in _pair(frame).groupby("distance"):
            effects.append(
                {
                    "architecture": arch,
                    "hidden_size": hidden,
                    "distance": distance,
                    **paired_summary(part.blocked, part.interleaved, rng, samples),
                }
            )
    effect_frame = pd.DataFrame(effects)
    effect_frame.to_csv(target / "variant_effects.csv", index=False)
    reference = _pair(subset[(subset.architecture == "rnn") & (subset.hidden_size == 7)])
    contrasts = []
    for (arch, hidden), frame in subset.groupby(["architecture", "hidden_size"]):
        if arch == "rnn" and hidden == 7:
            continue
        merged = _pair(frame).merge(
            reference,
            on=["run_id", "distance"],
            suffixes=("_variant", "_reference"),
            validate="one_to_one",
        )
        for distance, part in merged.groupby("distance"):
            difference = (
                part.difference_pp_variant.to_numpy() - part.difference_pp_reference.to_numpy()
            )
            # paired_summary converts fractions to pp, so divide these pp by 100.
            result = paired_summary(np.zeros(len(part)), difference / 100, rng, samples)
            contrasts.append(
                {
                    "family": "architecture" if arch != "rnn" else "capacity",
                    "architecture": arch,
                    "hidden_size": hidden,
                    "distance": distance,
                    **result,
                }
            )
    contrast_frame = pd.DataFrame(contrasts)
    if len(contrast_frame):
        for _, indices in contrast_frame.groupby("family").groups.items():
            contrast_frame.loc[indices, "p_holm_family"] = holm(
                contrast_frame.loc[indices, "p_sign_flip"]
            )
    contrast_frame.to_csv(target / "variant_contrasts.csv", index=False)
    granularity = []
    baseline = subset[(subset.architecture == "rnn") & (subset.hidden_size == 7)]
    for distance, frame in baseline.groupby("distance"):
        pivot = frame.pivot(index="run_id", columns="schedule", values="action_accuracy")
        for schedule in ("large_blocks", "small_blocks", "cyclic", "interleaved"):
            granularity.append(
                {
                    "distance": distance,
                    "schedule": schedule,
                    **paired_summary(pivot.blocked, pivot[schedule], rng, samples),
                }
            )
    granularity = pd.DataFrame(granularity)
    if len(granularity):
        granularity["p_holm_family"] = holm(granularity.p_sign_flip)
    granularity.to_csv(target / "granularity_effects.csv", index=False)
    metrics.groupby(KEYS + ["stage", "split", "group"])[
        ["action_accuracy", "sequence_accuracy", "cross_entropy", "action_count"]
    ].agg(["mean", "std", "count"]).to_csv(target / "secondary_summary.csv")
    noise = metrics[
        (metrics.stage == "before_D")
        & (metrics.group == "all")
        & metrics.split.str.startswith("robustness_")
    ].copy()
    noise["noise_sd"] = noise.split.str.replace("robustness_", "", regex=False).astype(float)
    noise.groupby(KEYS + ["noise_sd"]).action_accuracy.agg(
        ["mean", "std", "count"]
    ).reset_index().to_csv(target / "noise_summary.csv", index=False)
    forgetting = (
        metrics[(metrics.split == "final_noisy") & metrics.group.isin(["all", "A", "B", "C"])]
        .pivot(index=KEYS + ["run_id", "group"], columns="stage", values="action_accuracy")
        .reset_index()
    )
    forgetting["forgetting_pp"] = 100 * (forgetting.before_D - forgetting.after_D)
    forgetting.to_csv(target / "forgetting_distributions.csv", index=False)
    thresholds = runs.copy()
    thresholds["threshold_update"] = pd.to_numeric(thresholds.threshold_start, errors="coerce")
    thresholds.groupby(KEYS).threshold_update.agg(["count", "median"]).reset_index().rename(
        columns={"count": "reached", "median": "median_update_reached_only"}
    ).to_csv(target / "thresholds.csv", index=False)
    write_json(
        target / "analysis_provenance.json",
        {
            "producing_git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "config_sha256": canonical_hash(config.as_dict()),
            "completed_at_utc": utc_now(),
            "namespace": config.namespace,
            "statistics_source_sha256": file_hash(Path(__file__).with_name("statistics.py")),
            "analysis_source_sha256": file_hash(__file__),
            "bootstrap_samples": samples,
            "v1_data_pooled": False,
            "independent_unit": "replicate ID",
        },
    )
    publication = (
        Path("results/v2")
        if config.namespace == "cruxrnn_v2_main_20261114"
        else root / "publication"
    )
    publication.mkdir(parents=True, exist_ok=True)
    for path in target.iterdir():
        if path.suffix in (".csv", ".json", ".gz"):
            shutil.copyfile(str(path), str(publication / path.name))
    for name in ("configuration.json", "blocked_orders.json", "pip_freeze.txt"):
        shutil.copyfile(str(root / name), str(publication / name))
    shutil.copytree(
        str(root / "execution_records"), str(publication / "execution_records"), dirs_exist_ok=True
    )
    print(summary.to_string(index=False))
    print(json.dumps(interactions, indent=2))
