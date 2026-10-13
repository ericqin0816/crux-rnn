"""An independent artifact audit, including the analysis resampling unit."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import torch
from motor_rnn.data import assert_no_overlap, hamming, ORDERS
from motor_rnn.experiment import verify_done
from motor_rnn.provenance import implementation_hash, file_hash, write_json, seed_bundle

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
path = Path(args.config)
config = json.loads(path.read_text())
root = Path(config["output_directory"])
assignment = json.loads((root / "blocked_order_assignment.json").read_text())
if config["phase"] == "final":
    assert file_hash(path) == path.with_suffix(".sha256").read_text().strip()
    assert implementation_hash() == config["implementation_sha256"]
    for k in config["distances"]:
        orders = list(assignment[str(k)].values())
        assert {s: orders.count(s) for s in ORDERS} == {s: 5 for s in ORDERS}
all_seeds = set()
count, byte_count = 0, 0
for run_id in config["run_ids"]:
    for k in config["distances"]:
        directory = root / "runs" / "run_{:03d}_k{}".format(run_id, k)
        cfg = {**config, "run_id": run_id, "distance": k, "blocked_order": assignment[str(k)][str(run_id)]}
        assert verify_done(directory, cfg), "Incomplete required cell"
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["seeds"] == seed_bundle(config["namespace"], run_id, k)
        for value in manifest["seeds"].values():
            assert value not in all_seeds, "Unexpected seed collision"
            all_seeds.add(value)
        assert manifest["seeds"]["training_noise"] != manifest["seeds"]["development_noise"] != manifest["seeds"]["final_noise"]
        with np.load(directory / "data.npz") as data, np.load(directory / "orders.npz") as orders:
            assert data["train_x"].shape == (3 * config["train_per_sequence"], 1, 7)
            assert data["train_y"].shape == (3 * config["train_per_sequence"], 7)
            assert data["pre_x"].shape == (10 * config["pre_per_sequence"], 1, 7)
            assert np.array_equal(np.sort(orders["blocked"]), data["train_ids"])
            assert np.array_equal(np.sort(orders["interleaved"]), data["train_ids"])
            for schedule in ("blocked", "interleaved"):
                assert np.array_equal(np.bincount(data["train_family"][orders[schedule]]), [config["train_per_sequence"]] * 3)
            assert np.array_equal(data["train_y"], np.repeat(data["main_bases"], config["train_per_sequence"], axis=0))
            if config["mode"] == "controlled_similarity":
                assert_no_overlap({key: data[key] for key in ("main_bases", "pre_bases", "gen_bases", "D_base")})
                assert [hamming(data["main_bases"][i], data["main_bases"][j]) for i, j in ((0, 1), (0, 2), (1, 2))] == [k] * 3
                for split in ("dev", "final"):
                    assert data[split + "_noisy_x"].shape == (3 * config["eval_per_sequence"], 1, 7)
                    assert data[split + "_clean_x"].shape == (3, 1, 7)
                    assert np.array_equal(data[split + "_clean_x"][:, 0], data["main_bases"] + 1)
                    assert np.array_equal(data[split + "_noisy_y"], np.repeat(data["main_bases"], config["eval_per_sequence"], axis=0))
                assert not np.array_equal(data["dev_noisy_x"], data["final_noisy_x"])
                assert data["gen_noisy_x"].shape == (100 * config["eval_per_sequence"], 1, 7)
                assert data["gen_clean_x"].shape == (100, 1, 7)
                assert data["D_x"].shape == (config["D_examples"], 1, 7)
                assert np.array_equal(data["D_y"], np.repeat(data["D_base"], config["D_examples"], axis=0))
        checks = json.loads((directory / "pair_checks.json").read_text())
        assert len(set(checks["initial_state_hashes"].values())) == 1
        raw = pd.read_csv(directory / "metrics.csv")
        run_level = pd.read_csv(directory / "run_level.csv")
        assert len(run_level) == 2
        for schedule in ("blocked", "interleaved"):
            if config["mode"] == "controlled_similarity":
                for split in ("final_clean", "final_noisy"):
                    for stage in ("before_D", "after_D"):
                        part = raw[(raw.schedule == schedule) & (raw.stage == stage) & (raw.split == split)]
                        overall = part[part.group == "all"].action_accuracy.iloc[0]
                        equally_weighted = part[part.group.isin(list("ABC"))].action_accuracy.mean()
                        assert np.isclose(overall, equally_weighted, atol=1e-12)
                        shared = part[part.group == "shared"].iloc[0]
                        if k == 7:
                            assert shared.action_count == 0 and np.isnan(shared.action_accuracy)
                        before = raw[(raw.schedule == schedule) & (raw.stage == "before_D") & (raw.split == split) & (raw.group == "all")].action_accuracy.iloc[0]
                        after = raw[(raw.schedule == schedule) & (raw.stage == "after_D") & (raw.split == split) & (raw.group == "all")].action_accuracy.iloc[0]
                        forgetting_column = "forgetting_clean" if split == "final_clean" else "forgetting_noisy"
                        assert np.isclose(run_level[run_level.schedule == schedule][forgetting_column].iloc[0], before - after)
                development = pd.read_csv(directory / "development.csv")
                for split in ("clean", "noisy"):
                    frame = development[(development.schedule == schedule) & (development.split == split)]
                    for at in frame["update"].unique():
                        assert set(frame[frame["update"] == at].group) == {"all", "A", "B", "C"}
                if config["phase"] != "smoke":
                    assert {0, 250, 500, 750}.issubset(set(development["update"]))
        count += 1
        byte_count += sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
if config["mode"] == "controlled_similarity":
    with np.load(root / "analysis" / "bootstrap.npz") as saved:
        paired = pd.read_csv(root / "analysis" / "paired_differences.csv")
        matrix = paired.pivot(index="run_id", columns="distance", values="IP_minus_blocked_pp")
        assert saved["indices"].shape == (10000, len(config["run_ids"]))
        assert np.array_equal(saved["replicate_ids"], matrix.index.to_numpy())
        recalculated = matrix.to_numpy()[saved["indices"]].mean(axis=1)
        np.testing.assert_allclose(recalculated, saved["categorical_means_pp"], rtol=0, atol=1e-12)
        x = matrix.columns.to_numpy(dtype=float)
        x = x - x.mean()
        slope = (recalculated @ x) / (x @ x)
        np.testing.assert_allclose(slope, saved["slopes_pp_per_hamming"], rtol=0, atol=1e-12)
report = {"verified_cells": count, "missing_cells": 0, "distinct_saved_seed_values": len(all_seeds),
          "all_data_and_complete_file_hashes_valid": True, "config_sha256": file_hash(path),
          "run_artifact_bytes": byte_count, "no_statistical_subjects_other_than_run_ids": True}
write_json(root / "analysis" / "design_verification.json", report)
print(json.dumps(report, indent=2))
