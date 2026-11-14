"""Independent paired cells, explicit provenance, guarded resume and parallel CPU runs."""

import copy
import json
import logging
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from motor_rnn.data import copies, hamming, sequence_description

from .config import ExperimentConfig
from .evaluation import evaluate, sustained_threshold
from .models import SequenceModel, parameter_count
from .reproducibility import (
    canonical_hash,
    configure_backend,
    execution_identity,
    file_hash,
    seed,
    seed_bundle,
    set_seed,
    state_hash,
    utc_now,
    write_json,
)
from .schedules import SCHEDULES, blocked_order, make_order
from .sequences import generate
from .training import optimizer_for, train

LOG = logging.getLogger(__name__)
ORIGINAL_DISTANCES = (1, 2, 4, 7)


def make_jobs(config: ExperimentConfig, experiment: str = "all") -> list:
    """Reuse main RNN references for every exploratory matched comparison."""
    if experiment not in (
        "all",
        "main",
        "architecture",
        "capacity",
        "granularity",
        "forgetting",
        "noise",
        "dynamics",
    ):
        raise ValueError(f"Unknown experiment: {experiment}")
    jobs = []
    if experiment in ("all", "main", "granularity", "forgetting", "noise", "dynamics"):
        for run_id in range(config.repetitions):
            for distance in config.distances:
                extended = (
                    run_id < config.exploratory_repetitions and distance in ORIGINAL_DISTANCES
                )
                jobs.append(
                    {
                        "run_id": run_id,
                        "distance": distance,
                        "architecture": "rnn",
                        "hidden_size": 7,
                        "study": "main",
                        "schedules": list(SCHEDULES) if extended else ["blocked", "interleaved"],
                    }
                )
    if experiment in ("all", "architecture", "capacity"):
        variants = []
        if experiment in ("all", "architecture"):
            variants += [
                (arch, hidden, "architecture") for arch in ("gru", "lstm") for hidden in (7, 3)
            ]
        if experiment in ("all", "capacity"):
            variants += [("rnn", hidden, "capacity") for hidden in (16, 32, 64)]
        for arch, hidden, study in variants:
            for run_id in range(config.exploratory_repetitions):
                for distance in config.distances:
                    if distance in ORIGINAL_DISTANCES:
                        jobs.append(
                            {
                                "run_id": run_id,
                                "distance": distance,
                                "architecture": arch,
                                "hidden_size": hidden,
                                "study": study,
                                "schedules": ["blocked", "interleaved"],
                            }
                        )
    return jobs


def cell_directory(config: ExperimentConfig, job: dict) -> Path:
    return (
        Path(config.output_directory)
        / "runs"
        / f"{job['architecture']}_h{job['hidden_size']}_r{job['run_id']:03d}_k{job['distance']}"
    )


def verify_complete(directory: Path, recipe_hash=None) -> dict:
    """Verify every saved artifact; a completed cell can never be silently rerun."""
    path = directory / "complete.json"
    if not path.exists():
        return {}
    done = json.loads(path.read_text(encoding="utf-8"))
    if recipe_hash is not None and done["recipe_sha256"] != recipe_hash:
        raise ValueError(f"Completed cell recipe mismatch: {directory}")
    for name, checksum in done["files"].items():
        if not (directory / name).is_file() or file_hash(directory / name) != checksum:
            raise ValueError(f"Corrupt/missing completed artifact: {directory / name}")
    return done


def execute_cell(config: ExperimentConfig, job: dict, identity: dict, resume=False) -> dict:
    """One data family and pretrained initialization, cloned into every schedule."""
    directory = cell_directory(config, job)
    recipe = {
        "configuration": config.as_dict(),
        "job": job,
        "numeric_code_sha256": identity["numeric_code_sha256"],
    }
    recipe_hash = canonical_hash(recipe)
    existing = verify_complete(directory, recipe_hash)
    if existing:
        if not resume:
            raise FileExistsError(f"Complete cell exists; use --resume: {directory}")
        return {
            "status": "verified_existing",
            "directory": directory.as_posix(),
            "models": len(job["schedules"]),
        }
    started, clock = utc_now(), time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    seeds = seed_bundle(config.namespace, job["run_id"], job["distance"])
    seeds.update(
        {
            f"robustness_{level:g}": seed(
                config.namespace, job["run_id"], job["distance"], f"robustness_{level:g}"
            )
            for level in config.noise_levels
        }
    )
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["recipe_sha256"] != recipe_hash or not resume:
            raise ValueError("Existing incomplete cell requires --resume with the unchanged recipe")
        if file_hash(directory / "data.npz") != manifest["data_sha256"]:
            raise ValueError("Saved master data were altered")
        with np.load(directory / "data.npz", allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
    else:
        data = generate(config, job["distance"], seeds)
        for level in config.noise_levels:
            name = f"robustness_{level:g}"
            data[name + "_x"], data[name + "_y"] = copies(
                data["main_bases"],
                config.eval_per_sequence,
                np.random.RandomState(seeds[name]),
                level,
            )
        np.savez_compressed(directory / "data.npz", **data)
        manifest = {
            "recipe": recipe,
            "recipe_sha256": recipe_hash,
            "seeds": seeds,
            "data_sha256": file_hash(directory / "data.npz"),
            "started_at_utc": started,
            "initial_producing_git_commit": identity["git_commit"],
            "main_sequences": {
                name: sequence_description(base) for name, base in zip("ABC", data["main_bases"])
            },
            "pretraining_bases": data["pre_bases"].tolist(),
            "generalization_bases": data["gen_bases"].tolist(),
            "D_base": data["D_base"].tolist(),
            "changed_positions": data["changed_positions"].tolist(),
            "D_distances": [hamming(data["D_base"][0], base) for base in data["main_bases"]],
        }
        write_json(manifest_path, manifest)
    with open(directory / "attempts.jsonl", "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"started_at_utc": started, "git_commit": identity["git_commit"], "resume": resume}
            )
            + "\n"
        )
    metadata = {"recipe_sha256": recipe_hash, "data_sha256": manifest["data_sha256"]}
    set_seed(seeds["initialization"])
    model = SequenceModel(job["architecture"], job["hidden_size"]).to(config.device)

    def fit(target, x_name, order, stage, schedule, probes=None):
        return train(
            target,
            optimizer_for(target, config.learning_rate, config.momentum),
            data[x_name + "_x"],
            data[x_name + "_y"],
            order,
            directory / f"{schedule}_{stage}.pt",
            {**metadata, "stage": stage, "schedule": schedule},
            batch_size=config.batch_size,
            eval_every=config.d_eval_every if stage == "D" else config.eval_every,
            probes=probes,
            changed=data["changed_positions"],
            resume=resume,
        )

    fit(model, "pre", np.arange(len(data["pre_x"])), "pretraining", "shared")
    initial_hash = state_hash(model)
    probes = {
        split: (data[f"dev_{split}_x"], data[f"dev_{split}_y"]) for split in ("clean", "noisy")
    }
    orders = {
        name: make_order(
            config.train_per_sequence,
            name,
            blocked_order(job["run_id"], job["distance"]),
            seeds["interleaving_permutation"],
        )
        for name in job["schedules"]
    }
    np.savez_compressed(directory / "orders.npz", **orders)
    metrics, run_rows, curves, losses = [], [], [], []
    for schedule in job["schedules"]:
        target = copy.deepcopy(model)
        if state_hash(target) != initial_hash:
            raise RuntimeError("Paired initialization differs")
        main = fit(target, "train", orders[schedule], "main", schedule, probes)
        if main["update"] != 3 * config.train_per_sequence // config.batch_size:
            raise RuntimeError("Incomplete main training budget")
        identity_fields = {
            **{key: value for key, value in job.items() if key != "schedules"},
            "schedule": schedule,
        }
        losses += [{**identity_fields, "stage": "main", **row} for row in main["losses"]]
        curves += [{**identity_fields, "stage": "main", **row} for row in main["probes"]]

        def score(scoring_model, stage, split, main_groups=True, fields=identity_fields):
            rows = evaluate(
                scoring_model,
                data[split + "_x"],
                data[split + "_y"],
                data["changed_positions"],
                main_groups,
            )
            metrics.extend({**fields, "stage": stage, "split": split, **row} for row in rows)
            return rows[0]

        noisy, clean = (
            score(target, "before_D", "final_noisy"),
            score(target, "before_D", "final_clean"),
        )
        gen_clean, gen_noisy = (
            score(target, "before_D", "gen_clean", False),
            score(target, "before_D", "gen_noisy", False),
        )
        for level in config.noise_levels:
            score(target, "before_D", f"robustness_{level:g}")
        curve = [
            (row["update"], row["action_accuracy"])
            for row in main["probes"]
            if row["split"] == "noisy" and row["group"] == "all"
        ]
        row = {
            **identity_fields,
            "parameter_count": parameter_count(target),
            "initial_state_sha256": initial_hash,
            "action_accuracy": noisy["action_accuracy"],
            "clean_accuracy": clean["action_accuracy"],
            "sequence_accuracy": noisy["sequence_accuracy"],
            "cross_entropy": noisy["cross_entropy"],
            "generalization_clean": gen_clean["action_accuracy"],
            "generalization_noisy": gen_noisy["action_accuracy"],
            "main_updates": main["update"],
            "main_seconds": main["elapsed_seconds"],
            **sustained_threshold(curve, config.threshold, config.threshold_checkpoints),
        }
        # Keep the intact main model and adapt only a clone to D.
        main_state = state_hash(target)
        main_target, target = target, copy.deepcopy(target)
        interference = fit(target, "D", np.arange(len(data["D_x"])), "D", schedule, probes)
        curves += [{**identity_fields, "stage": "D", **item} for item in interference["probes"]]
        losses += [{**identity_fields, "stage": "D", **item} for item in interference["losses"]]
        after_noisy, after_clean = (
            score(target, "after_D", "final_noisy"),
            score(target, "after_D", "final_clean"),
        )
        if state_hash(main_target) != main_state:
            raise RuntimeError("D training modified the original main model")
        row.update(
            forgetting_noisy=noisy["action_accuracy"] - after_noisy["action_accuracy"],
            forgetting_clean=clean["action_accuracy"] - after_clean["action_accuracy"],
            d_updates=interference["update"],
        )
        run_rows.append(row)
    for name, rows in (
        ("metrics", metrics),
        ("run_level", run_rows),
        ("curves", curves),
        ("losses", losses),
    ):
        pd.DataFrame(rows).to_csv(directory / f"{name}.csv", index=False)
    paths = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.name != "complete.json" and p.suffix != ".tmp"
    )
    done = {
        "recipe_sha256": recipe_hash,
        "job": job,
        "started_at_utc": manifest["started_at_utc"],
        "finished_at_utc": utc_now(),
        "completing_git_commit": identity["git_commit"],
        "elapsed_seconds_this_attempt": time.perf_counter() - clock,
        "main_models": len(job["schedules"]),
        "D_adapted_clones": len(job["schedules"]),
        "pretrained_models": 1,
        "initial_state_sha256": initial_hash,
        "files": {p.name: file_hash(p) for p in paths},
    }
    write_json(directory / "complete.json", done)
    verify_complete(directory, recipe_hash)
    return {
        "status": "completed",
        "directory": directory.as_posix(),
        "models": len(job["schedules"]),
        "elapsed_seconds": done["elapsed_seconds_this_attempt"],
    }


def run_study(
    config: ExperimentConfig,
    experiment: str = "all",
    workers: int = 1,
    resume: bool = False,
    config_path=None,
    research: bool = True,
) -> dict:
    """Run a committed recipe; successes remain valid even when another cell fails."""
    if workers < 1 or (config.device == "cuda" and workers != 1):
        raise ValueError("Use positive workers; CUDA runs require one worker")
    configure_backend(config.device)
    identity = execution_identity(config_path, research)
    root = Path(config.output_directory)
    root.mkdir(parents=True, exist_ok=True)
    config_file = root / "configuration.json"
    if config_file.exists() and canonical_hash(
        json.loads(config_file.read_text())
    ) != canonical_hash(config.as_dict()):
        raise ValueError("Output directory belongs to a different configuration")
    write_json(config_file, config.as_dict())
    jobs = make_jobs(config, experiment)
    assignment = {
        str(k): {str(run): blocked_order(run, k) for run in range(config.repetitions)}
        for k in config.distances
    }
    write_json(root / "blocked_orders.json", assignment)
    execution_path = root / "execution_records" / (uuid.uuid4().hex + ".json")
    record = {
        **identity,
        "started_at_utc": utc_now(),
        "config_sha256": canonical_hash(config.as_dict()),
        "config_file_sha256": file_hash(config_path) if config_path else None,
        "plan_sha256": file_hash(config.plan) if Path(config.plan).exists() else None,
        "command": sys.argv,
        "experiment": experiment,
        "workers": workers,
        "device": config.device,
        "expected_cells_this_selection": len(jobs),
        "state": "running",
        "successes": [],
        "failures": [],
    }
    write_json(execution_path, record)
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    )
    (root / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=configure_backend, initargs=(config.device,)
    ) as pool:
        tasks = {pool.submit(execute_cell, config, job, identity, resume): job for job in jobs}
        for future in as_completed(tasks):
            job = tasks[future]
            try:
                result = future.result()
                record["successes"].append(result)
                LOG.info(
                    "%s %d/%d %s",
                    experiment,
                    len(record["successes"]),
                    len(jobs),
                    result["directory"],
                )
            except Exception:
                failure = {
                    "job": job,
                    "utc": utc_now(),
                    "traceback": traceback.format_exc(),
                    "excluded": False,
                }
                record["failures"].append(failure)
                with open(root / "failures.jsonl", "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure) + "\n")
                LOG.exception("Cell failed and was retained: %s", job)
            write_json(execution_path, record)
    record.update(
        finished_at_utc=utc_now(),
        wall_seconds=time.perf_counter() - started,
        state="failed" if record["failures"] else "completed",
    )
    write_json(execution_path, record)
    if record["failures"]:
        raise RuntimeError(
            f"{len(record['failures'])} retained cell failures; inspect {execution_path}"
        )
    return record
