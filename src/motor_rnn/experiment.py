"""Execute one immutable run-by-distance cell and verify its saved artifacts."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from . import SOURCE_COMMIT
from .data import save_data, schedule_orders
from .metrics import evaluate, sustained_threshold
from .model import MotorRNN
from .provenance import canonical_hash, file_hash, seed_bundle, state_hash, write_json
from .training import optimizer_for, train


def verify_done(directory, config):
    done_path = directory / "complete.json"
    if not done_path.exists():
        return False
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done["config_hash"] != canonical_hash(config):
        raise ValueError("Completed run config mismatch")
    for name, expected in done["files"].items():
        if not (directory / name).exists() or file_hash(directory / name) != expected:
            raise ValueError("Completed artifact corrupt or missing: " + name)
    return True


def per_example_loss(model, x, y):
    values = []
    before = state_hash(model)
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            logits = model(torch.as_tensor(x[start:start + 1024]))
            loss = F.cross_entropy(logits.reshape(-1, 4), torch.as_tensor(y[start:start + 1024]).reshape(-1), reduction="none")
            values.append(loss.reshape(-1, 7).mean(1).numpy())
    assert state_hash(model) == before
    return np.concatenate(values)


def source_vulnerability(model, data, seeds, schedule):
    """Optional source protocols: last sequence only; accumulated perturbations."""
    rows = []
    suffix = "_" + schedule
    from .provenance import seed
    torch.manual_seed(seed(str(seeds["initialization"]), 0, 0, "weight_noise" + suffix))
    retention = (data["train_x"][-5000:], data["train_y"][-5000:])
    generalization = (data["gen_noisy_x"], data["gen_noisy_y"])
    for kind in ("weight_noise", "pruning", "interference"):
        perturbed = copy.deepcopy(model)
        opt = optimizer_for(perturbed)
        # Independent source interference stimuli per schedule, as in source.
        rng = np.random.RandomState(seed(str(seeds["D_noise"]), 0, 0, kind + suffix))
        from .data import copies
        base = rng.randint(4, size=(1, 7))
        dx, dy = copies(base, 2000, rng, 0.1)
        order = rng.permutation(2000)
        from .training import step
        limit = 101 if kind == "interference" else 100
        for at in range(limit):
            if at:
                if kind == "interference":
                    ids = order[(at - 1) * 20:at * 20]
                    step(perturbed, opt, torch.as_tensor(dx[ids]), torch.as_tensor(dy[ids]))
                else:
                    with torch.no_grad():
                        for p in perturbed.parameters():
                            if kind == "weight_noise":
                                p.add_(torch.randn_like(p) * 0.1)
                            else:
                                p.mul_(torch.rand_like(p) > 0.05)
            for split, pair in (("last_training_sequence", retention), ("generalization", generalization)):
                rows.append({"kind": kind, "step": at, "split": split,
                             "cross_entropy": float(per_example_loss(perturbed, *pair).mean())})
    return rows


def execute_cell(config, directory, resume=False, force=False):
    directory = Path(directory)
    if verify_done(directory, config):
        if not resume:
            raise FileExistsError("Run already complete; use --resume")
        return {"status": "verified_existing", "run_id": config["run_id"], "distance": config["distance"]}
    begin = time.perf_counter()
    seeds = seed_bundle(config["namespace"], config["run_id"], config["distance"])
    if (directory / "manifest.json").exists():
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config_hash"] != canonical_hash(config):
            raise ValueError("Existing data config differs; retain it and start a separate phase")
        assert file_hash(directory / "data.npz") == manifest["data_sha256"], "Data file corrupt"
        with np.load(directory / "data.npz", allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
    else:
        data, manifest = save_data(directory, config, seeds)
    controlled = config["mode"] == "controlled_similarity"
    metadata = {"config_hash": canonical_hash(config), "data_manifest_hash": file_hash(directory / "manifest.json"),
                "run_id": config["run_id"], "distance": config["distance"], "blocked_order": config["blocked_order"],
                "source_commit": SOURCE_COMMIT}
    torch.manual_seed(seeds["initialization"])
    initial = MotorRNN(source_exact=True)
    pre = train(initial, optimizer_for(initial), data["pre_x"], data["pre_y"], np.arange(len(data["pre_x"])),
                {**metadata, "schedule": "pretraining", "stage": "pretraining"}, directory / "pretraining.pt",
                resume=resume, force=force)
    assert pre["current_update"] == 10 * config["pre_per_sequence"] // 20
    torch.save(initial.state_dict(), str(directory / "pretrained_weights.pt"))
    models = {name: copy.deepcopy(initial) for name in ("blocked", "interleaved")}
    hashes = {name: state_hash(model) for name, model in models.items()}
    assert len(set(hashes.values())) == 1
    orders = schedule_orders(config["train_per_sequence"], config["blocked_order"], seeds["interleaving_permutation"])
    np.savez_compressed(directory / "orders.npz", **orders)
    pair_checks = {"initial_state_hashes": hashes, "identical_example_id_multisets": True,
                   "exposures_each_sequence": config["train_per_sequence"], "updates_each_schedule": len(data["train_x"]) // 20,
                   "fresh_identical_sgd": {"lr": 0.02, "momentum": 0.0}}
    write_json(directory / "pair_checks.json", pair_checks)
    probes = {split: (data["dev_" + split + "_x"], data["dev_" + split + "_y"]) for split in ("clean", "noisy")} if controlled else None
    metrics, training_log, dev_log, d_log, run_rows, vuln = [], [], [], [], [], []
    source_arrays = {}
    if not controlled:
        source_arrays["pre_generalization"] = per_example_loss(initial, data["gen_noisy_x"], data["gen_noisy_y"])
    def evaluate_split(model, schedule, stage, name, prefix, main_groups=True):
        rows = evaluate(model, data[prefix + "_x"], data[prefix + "_y"], data["changed_positions"], main_groups=main_groups)
        for row in rows:
            metrics.append({"run_id": config["run_id"], "distance": config["distance"], "schedule": schedule,
                            "stage": stage, "split": name, **row})
        return next(row for row in rows if row["group"] == "all")
    for schedule, model in models.items():
        main = train(model, optimizer_for(model), data["train_x"], data["train_y"], orders[schedule],
                     {**metadata, "schedule": schedule, "stage": "main"}, directory / (schedule + "_main.pt"),
                     probes=probes, changed=data["changed_positions"], eval_every=config["eval_every"],
                     checkpoint_every=50, resume=resume, force=force)
        assert main["current_update"] == len(data["train_x"]) // 20
        assert len(main["losses"]) == main["current_update"]
        for row in main["losses"]:
            training_log.append({"schedule": schedule, **row})
        for row in main["probes"]:
            dev_log.append({"schedule": schedule, **row})
        if controlled:
            noisy = evaluate_split(model, schedule, "before_D", "final_noisy", "final_noisy")
            clean = evaluate_split(model, schedule, "before_D", "final_clean", "final_clean")
        else:
            noisy = evaluate_split(model, schedule, "source_retention", "training_noisy", "train")
            clean = {"action_accuracy": float("nan")}
            source_arrays[schedule + "_training"] = np.asarray([row["training_cross_entropy"] for row in main["losses"]])
            source_arrays[schedule + "_retention"] = per_example_loss(model, data["train_x"], data["train_y"])
            source_arrays[schedule + "_generalization"] = per_example_loss(model, data["gen_noisy_x"], data["gen_noisy_y"])
        gen_clean = evaluate_split(model, schedule, "before_D", "generalization_clean", "gen_clean", False)
        gen_noisy = evaluate_split(model, schedule, "before_D", "generalization_noisy", "gen_noisy", False)
        row = {"run_id": config["run_id"], "distance": config["distance"], "schedule": schedule,
               "blocked_order": config["blocked_order"], "main_updates": main["current_update"],
               "training_cross_entropy": float(np.mean([r["training_cross_entropy"] for r in main["losses"]])),
               "final_noisy_action_accuracy": noisy["action_accuracy"], "final_clean_action_accuracy": clean["action_accuracy"],
               "final_noisy_sequence_accuracy": noisy["sequence_accuracy"], "final_cross_entropy": noisy["cross_entropy"],
               "generalization_clean_accuracy": gen_clean["action_accuracy"], "generalization_noisy_accuracy": gen_noisy["action_accuracy"],
               "generalization_cross_entropy": gen_noisy["cross_entropy"], "main_seconds": main["elapsed_seconds"]}
        if controlled:
            curve = [(r["update"], r["action_accuracy"]) for r in main["probes"] if r["split"] == "noisy" and r["group"] == "all"]
            row.update(sustained_threshold(curve))
            disturbed = copy.deepcopy(model)
            assert state_hash(disturbed) == state_hash(model)
            d_training = train(disturbed, optimizer_for(disturbed), data["D_x"], data["D_y"], np.arange(len(data["D_x"])),
                               {**metadata, "schedule": schedule, "stage": "D"}, directory / (schedule + "_D.pt"),
                               probes=probes, changed=data["changed_positions"], eval_every=1, checkpoint_every=50,
                               resume=resume, force=force)
            assert d_training["current_update"] == config["D_examples"] // 20
            for r in d_training["probes"]:
                d_log.append({"schedule": schedule, **r})
            for r in d_training["losses"]:
                training_log.append({"schedule": schedule + "_D", **r})
            after_noisy = evaluate_split(disturbed, schedule, "after_D", "final_noisy", "final_noisy")
            after_clean = evaluate_split(disturbed, schedule, "after_D", "final_clean", "final_clean")
            row.update({"forgetting_noisy": noisy["action_accuracy"] - after_noisy["action_accuracy"],
                        "forgetting_clean": clean["action_accuracy"] - after_clean["action_accuracy"],
                        "D_updates": d_training["current_update"], "D_seconds": d_training["elapsed_seconds"]})
        elif config.get("source_vulnerability", False):
            for r in source_vulnerability(model, data, seeds, schedule):
                vuln.append({"schedule": schedule, **r})
        run_rows.append(row)
    for name, rows in (("metrics", metrics), ("training", training_log), ("development", dev_log),
                       ("D_development", d_log), ("run_level", run_rows), ("source_vulnerability", vuln)):
        if rows:
            pd.DataFrame(rows).to_csv(directory / (name + ".csv"), index=False)
    if source_arrays:
        np.savez_compressed(directory / "source_arrays.npz", **source_arrays)
    files = [p.name for p in directory.iterdir() if p.is_file() and p.name != "complete.json" and not p.name.endswith(".tmp")]
    done = {"status": "complete", "config_hash": canonical_hash(config), "run_id": config["run_id"],
            "distance": config["distance"], "elapsed_seconds": time.perf_counter() - begin,
            "main_updates_each": len(data["train_x"]) // 20, "files": {name: file_hash(directory / name) for name in sorted(files)}}
    write_json(directory / "complete.json", done)
    assert verify_done(directory, config)
    return {key: done[key] for key in ("status", "run_id", "distance", "elapsed_seconds")}
