"""Controls for new scientific functionality and frozen-v1 equivalence."""

import copy
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

from cruxrnn.config import ExperimentConfig, load_config
from cruxrnn.engine import execute_cell, make_jobs, verify_complete
from cruxrnn.evaluation import evaluate
from cruxrnn.models import SequenceModel, parameter_count
from cruxrnn.reproducibility import execution_identity, seed_bundle, state_hash
from cruxrnn.schedules import ORDERS, SCHEDULES, blocked_order, make_order
from cruxrnn.sequences import family, generate
from cruxrnn.training import optimizer_for, train
from motor_rnn.data import generate as legacy_generate
from motor_rnn.data import hamming
from motor_rnn.model import MotorRNN

torch.set_num_threads(1)


class V2DesignTests(unittest.TestCase):
    def tiny(self, output="outputs/v2/unit"):
        return ExperimentConfig(
            namespace="v2_unit",
            output_directory=output,
            repetitions=2,
            exploratory_repetitions=0,
            distances=(1, 7),
            pre_per_sequence=2,
            train_per_sequence=20,
            eval_per_sequence=2,
            d_examples=20,
            eval_every=1,
            bootstrap_samples=100,
            noise_levels=(0, 0.1),
        )

    def test_all_seven_distances(self):
        for k in range(1, 8):
            for run in range(20):
                bases, changed = family(np.random.RandomState(run), k)
                self.assertEqual(len(changed), k)
                self.assertEqual(
                    [hamming(bases[i], bases[j]) for i, j in ((0, 1), (0, 2), (1, 2))], [k] * 3
                )
        for invalid in (0, 8, 1.5, True):
            with self.assertRaises(ValueError):
                family(np.random.RandomState(0), invalid)

    def test_generator_matches_v1_at_original_distances(self):
        cfg = ExperimentConfig()
        legacy = json.loads(Path("configs/final_proposed.json").read_text())
        for k in (1, 2, 4, 7):
            seeds = seed_bundle("parity", 0, k)
            old = legacy_generate({**legacy, "distance": k}, seeds)
            new = generate(cfg, k, seeds)
            for key, value in new.items():
                np.testing.assert_array_equal(value, old[key])

    def test_unique_disjoint_bases_and_unchanged_labels(self):
        cfg = self.tiny()
        for k in range(1, 8):
            data = generate(cfg, k, seed_bundle("v2_unit", 0, k))
            bases = np.concatenate(
                [data[key] for key in ("main_bases", "pre_bases", "gen_bases", "D_base")]
            )
            self.assertEqual(len(bases), len({tuple(row) for row in bases}))
            np.testing.assert_array_equal(
                data["train_y"], np.repeat(data["main_bases"], 20, axis=0)
            )
            self.assertFalse(np.array_equal(data["dev_noisy_x"], data["final_noisy_x"]))

    def test_schedule_exposure_and_determinism(self):
        for name in SCHEDULES:
            for count in (23, 100, 5000):
                ids = make_order(count, name, "CAB", 42)
                np.testing.assert_array_equal(np.sort(ids), np.arange(3 * count))
                np.testing.assert_array_equal(np.bincount(ids // count), [count] * 3)
                np.testing.assert_array_equal(ids, make_order(count, name, "CAB", 42))
        np.testing.assert_array_equal(
            make_order(3, "cyclic", "ABC", 0), [0, 3, 6, 1, 4, 7, 2, 5, 8]
        )
        with self.assertRaises(ValueError):
            make_order(1, "bogus", "ABC", 0)

    def test_near_balance_exactly_100_replicates(self):
        for k in range(1, 8):
            counts = [sum(blocked_order(run, k) == order for run in range(100)) for order in ORDERS]
            self.assertEqual(sum(counts), 100)
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_architecture_shapes_and_parameter_counts(self):
        expected = {"rnn": 179, "gru": 403, "lstm": 515}
        for architecture in expected:
            for hidden in (3, 7, 16, 32, 64):
                model = SequenceModel(architecture, hidden)
                self.assertEqual(tuple(model(torch.zeros(2, 1, 7)).shape), (2, 7, 4))
                if hidden == 7:
                    self.assertEqual(parameter_count(model), expected[architecture])
        with self.assertRaises(ValueError):
            SequenceModel("unknown")

    def test_rnn_output_and_gradient_bitwise_parity(self):
        torch.manual_seed(232)
        legacy = MotorRNN()
        torch.manual_seed(232)
        new = SequenceModel()
        x = torch.randn(20, 1, 7)
        torch.testing.assert_close(legacy(x), new(x), atol=0, rtol=0)
        legacy(x).sum().backward()
        new(x).sum().backward()
        for old, current in zip(legacy.parameters(), new.parameters()):
            torch.testing.assert_close(old.grad, current.grad, atol=0, rtol=0)

    def test_paired_initialization_and_configured_optimizer(self):
        original = SequenceModel("gru", 7)
        first, second = copy.deepcopy(original), copy.deepcopy(original)
        self.assertEqual(state_hash(first), state_hash(second))
        optimizer = optimizer_for(first, 0.123, 0.25)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.123)
        self.assertEqual(optimizer.param_groups[0]["momentum"], 0.25)

    def test_evaluation_groups_and_no_weight_mutation(self):
        cfg = self.tiny()
        data = generate(cfg, 7, seed_bundle("eval", 0, 7))
        model = SequenceModel("lstm", 3)
        before = state_hash(model)
        rows = evaluate(model, data["final_noisy_x"], data["final_noisy_y"], range(7))
        shared = next(row for row in rows if row["group"] == "shared")
        self.assertTrue(np.isnan(shared["action_accuracy"]))
        self.assertEqual(shared["action_count"], 0)
        self.assertEqual(state_hash(model), before)
        self.assertTrue(model.training)
        rows = evaluate(model, data["final_noisy_x"], data["final_noisy_y"], main_groups=False)
        self.assertFalse({row["group"] for row in rows} & {"A", "B", "C", "shared", "differing"})

    def test_configuration_rejects_invalid_and_historical_paths(self):
        cfg = self.tiny()
        for changes in (
            {"distances": (1, 1)},
            {"distances": (8,)},
            {"learning_rate": float("nan")},
            {"batch_size": 7},
            {"repetitions": True},
            {"threshold": 2},
            {"output_directory": "outputs/final/subdir"},
            {"device": "auto"},
        ):
            with self.assertRaises(ValueError):
                replace(cfg, **changes)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"unknown":2}')
            with self.assertRaises(ValueError):
                load_config(str(path))

    def test_new_training_resume_is_bitwise_exact(self):
        cfg = self.tiny()
        data = generate(cfg, 1, seed_bundle("resume", 0, 1))
        torch.manual_seed(7)
        initial = SequenceModel()
        full, partial = copy.deepcopy(initial), copy.deepcopy(initial)
        order = make_order(20, "interleaved", "ABC", 7)
        metadata = {"recipe": "unit"}
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.pt", Path(directory) / "b.pt"
            whole = train(
                full,
                optimizer_for(full, 0.02),
                data["train_x"],
                data["train_y"],
                order,
                a,
                metadata,
            )
            train(
                partial,
                optimizer_for(partial, 0.02),
                data["train_x"],
                data["train_y"],
                order,
                b,
                metadata,
                stop_after=1,
            )
            resumed = SequenceModel()
            result = train(
                resumed,
                optimizer_for(resumed, 0.02),
                data["train_x"],
                data["train_y"],
                order,
                b,
                metadata,
                resume=True,
            )
            self.assertEqual(state_hash(full), state_hash(resumed))
            self.assertEqual(whole["losses"], result["losses"])
            with self.assertRaises(ValueError):
                train(
                    resumed,
                    optimizer_for(resumed, 0.02),
                    data["train_x"],
                    data["train_y"],
                    order,
                    b,
                    {"recipe": "wrong"},
                    resume=True,
                )

    def test_end_to_end_cell_and_corruption_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.tiny(directory)
            job = make_jobs(cfg, "main")[0]
            identity = execution_identity(require_committed=False)
            result = execute_cell(cfg, job, identity)
            path = Path(result["directory"])
            done = verify_complete(path)
            self.assertEqual(done["main_models"], 2)
            self.assertEqual(
                execute_cell(cfg, job, identity, resume=True)["status"], "verified_existing"
            )
            with self.assertRaises(FileExistsError):
                execute_cell(cfg, job, identity)
            (path / "run_level.csv").write_text("corrupt")
            with self.assertRaises(ValueError):
                verify_complete(path)

    def test_planned_counts_and_reuse(self):
        jobs = make_jobs(ExperimentConfig())
        self.assertEqual(len(jobs), 1260)
        self.assertEqual(sum(len(job["schedules"]) for job in jobs), 2760)
        self.assertEqual(
            len(
                {
                    (job["architecture"], job["hidden_size"], job["run_id"], job["distance"])
                    for job in jobs
                }
            ),
            len(jobs),
        )
