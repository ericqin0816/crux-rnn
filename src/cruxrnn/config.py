"""Validated operational configuration for independent v2 experiments."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Training budgets count examples, with one pass and equal schedule exposure."""

    namespace: str = "cruxrnn_v2_main_20261114"
    output_directory: str = "outputs/v2/main"
    repetitions: int = 100
    distances: tuple = (1, 2, 3, 4, 5, 6, 7)
    pre_per_sequence: int = 100
    train_per_sequence: int = 5000
    eval_per_sequence: int = 100
    d_examples: int = 2000
    batch_size: int = 20
    learning_rate: float = 0.02
    momentum: float = 0.0
    noise_sd: float = 0.1
    eval_every: int = 25
    d_eval_every: int = 1
    threshold: float = 0.9
    threshold_checkpoints: int = 3
    bootstrap_samples: int = 10000
    exploratory_repetitions: int = 20
    noise_levels: tuple = (0.0, 0.1, 0.3, 0.6, 1.0)
    device: str = "cpu"
    plan: str = "docs/experiment_plan_v2.md"

    def __post_init__(self) -> None:
        integers = (
            "repetitions",
            "pre_per_sequence",
            "train_per_sequence",
            "eval_per_sequence",
            "d_examples",
            "batch_size",
            "eval_every",
            "d_eval_every",
            "threshold_checkpoints",
            "bootstrap_samples",
        )
        for name in integers:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.distances or len(set(self.distances)) != len(self.distances):
            raise ValueError("distances must be nonempty and distinct")
        if any(type(k) is not int or not 1 <= k <= 7 for k in self.distances):
            raise ValueError("Each Hamming distance must be an integer from 1 through 7")
        for examples in (10 * self.pre_per_sequence, 3 * self.train_per_sequence, self.d_examples):
            if examples % self.batch_size:
                raise ValueError("Every training stage must contain complete batches")
        for name in ("learning_rate", "momentum", "noise_sd", "threshold"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.momentum < 0 or self.noise_sd < 0:
            raise ValueError("Learning rate must be positive; momentum/noise nonnegative")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must lie in [0,1]")
        if (
            type(self.exploratory_repetitions) is not int
            or not 0 <= self.exploratory_repetitions <= self.repetitions
        ):
            raise ValueError("exploratory_repetitions must be between zero and repetitions")
        if len(set(self.noise_levels)) != len(self.noise_levels) or any(
            not math.isfinite(x) or x < 0 for x in self.noise_levels
        ):
            raise ValueError("noise_levels must be distinct, finite and nonnegative")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda; CUDA is optional and never assumed")
        if not self.namespace or not self.output_directory:
            raise ValueError("A distinct namespace and output directory are required")
        directory = Path(self.output_directory).resolve()
        v1 = [Path("outputs/final").resolve(), Path("outputs/cruxrnn_final_proposed_v1").resolve()]
        if any(directory == old or old in directory.parents for old in v1):
            raise ValueError("v2 may not overwrite a historical v1 experiment directory")

    def as_dict(self) -> dict:
        """Return a JSON-compatible recipe with every operational setting."""
        return asdict(self)


def load_config(path: str) -> ExperimentConfig:
    """Reject unknown fields and invalid budgets before allocating data/models."""
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("distances", "noise_levels"):
        if key in values:
            values[key] = tuple(values[key])
    try:
        return ExperimentConfig(**values)
    except TypeError as error:
        raise ValueError(f"Unknown or malformed configuration: {error}") from error
