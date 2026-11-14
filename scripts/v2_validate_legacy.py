"""Retrain four saved v1 pairs with v2 and compare every final weight bitwise."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from cruxrnn.evaluation import evaluate
from cruxrnn.models import SequenceModel
from cruxrnn.reproducibility import (
    configure_backend,
    execution_identity,
    state_hash,
    utc_now,
    write_json,
)
from cruxrnn.training import optimizer_for, train


def main():
    configure_backend()
    identity = execution_identity("configs/v2/main.json")
    root = Path("outputs/v2/parity")
    if (root / "verification.json").exists():
        raise FileExistsError(
            "Parity already recorded; retain it rather than replacing a successful check"
        )
    started, clock = utc_now(), time.perf_counter()
    records = []
    for distance in (1, 2, 4, 7):
        original = Path(f"outputs/cruxrnn_final_proposed_v1/runs/run_000_k{distance}")
        manifest = json.loads((original / "manifest.json").read_text())
        with np.load(original / "data.npz") as saved:
            data = {name: saved[name] for name in saved.files}
        with np.load(original / "orders.npz") as saved:
            orders = {name: saved[name] for name in saved.files}
        torch.manual_seed(manifest["seeds"]["initialization"])
        initial = SequenceModel()
        directory = root / f"run_000_k{distance}"
        metadata = {"validation": "v1 numerical parity", "distance": distance}
        train(
            initial,
            optimizer_for(initial, 0.02),
            data["pre_x"],
            data["pre_y"],
            np.arange(len(data["pre_x"])),
            directory / "pre.pt",
            {**metadata, "stage": "pre"},
        )
        for schedule in ("blocked", "interleaved"):
            model = copy.deepcopy(initial)
            result = train(
                model,
                optimizer_for(model, 0.02),
                data["train_x"],
                data["train_y"],
                orders[schedule],
                directory / (schedule + ".pt"),
                {**metadata, "schedule": schedule, "stage": "main"},
            )
            reference = torch.load(
                str(original / (schedule + "_main.pt")), map_location="cpu", weights_only=False
            )
            if state_hash(model) != reference["state_hash"]:
                raise AssertionError(f"Final weights differ for k={distance}, {schedule}")
            np.testing.assert_array_equal(
                [row["cross_entropy"] for row in result["losses"]],
                [row["training_cross_entropy"] for row in reference["losses"]],
            )
            score = evaluate(
                model, data["final_noisy_x"], data["final_noisy_y"], data["changed_positions"]
            )[0]
            records.append(
                {
                    "distance": distance,
                    "schedule": schedule,
                    "weights_bitwise_equal": True,
                    "all_750_losses_bitwise_equal": True,
                    "state_sha256": state_hash(model),
                    "final_noisy_action_accuracy": score["action_accuracy"],
                }
            )
    write_json(
        root / "verification.json",
        {
            **identity,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "wall_seconds": time.perf_counter() - clock,
            "validation_main_models": 8,
            "independent_new_scientific_runs": 0,
            "results": records,
        },
    )
    print(
        json.dumps(
            {
                "all_weights_and_losses_bitwise_equal": True,
                "validation_main_models": 8,
                "wall_seconds": time.perf_counter() - clock,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
