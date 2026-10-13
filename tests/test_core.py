import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from motor_rnn.data import (ORDERS, assert_no_overlap, balanced_assignment, copies,
                            family, generate, hamming, schedule_orders)
from motor_rnn.metrics import evaluate, summarize, sustained_threshold
from motor_rnn.model import MotorRNN
from motor_rnn.provenance import canonical_hash, seed_bundle, state_hash
from motor_rnn.training import optimizer_for, train

torch.set_num_threads(1)


class DesignTests(unittest.TestCase):
    def test_pairwise_hamming(self):
        rng = np.random.RandomState(12)
        for k in (1, 2, 4, 7):
            for _ in range(50):
                bases, changed = family(rng, k)
                self.assertEqual(len(changed), k)
                self.assertEqual([hamming(bases[i], bases[j]) for i, j in ((0, 1), (0, 2), (1, 2))], [k] * 3)

    def test_forbidden_sequence_overlap(self):
        cfg = json.loads(Path("configs/smoke.json").read_text())
        for k in (1, 2, 4, 7):
            data = generate({**cfg, "distance": k}, seed_bundle("test", 0, k))
            assert_no_overlap({name: data[name] for name in ("main_bases", "pre_bases", "gen_bases", "D_base")})
        with self.assertRaises(AssertionError):
            assert_no_overlap({"first": np.zeros((1, 7), dtype=int), "second": np.zeros((1, 7), dtype=int)})

    def test_identical_paired_initial_weights(self):
        model = MotorRNN()
        blocked, ip = copy.deepcopy(model), copy.deepcopy(model)
        self.assertEqual(state_hash(blocked), state_hash(ip))
        self.assertEqual(optimizer_for(blocked).state_dict(), optimizer_for(ip).state_dict())

    def test_equal_exposure(self):
        for order in ORDERS:
            schedules = schedule_orders(5000, order, 16)
            self.assertTrue(np.array_equal(np.sort(schedules["blocked"]), np.sort(schedules["interleaved"])))
            for ids in schedules.values():
                np.testing.assert_array_equal(np.bincount(ids // 5000), [5000] * 3)

    def test_equal_optimizer_update_counts(self):
        base, _ = family(np.random.RandomState(31), 2)
        x, y = copies(base, 20, np.random.RandomState(23), 0.1)
        init = MotorRNN()
        meta = self.metadata()
        with tempfile.TemporaryDirectory() as tmp:
            for schedule, ids in schedule_orders(20, "CAB", 15).items():
                model = copy.deepcopy(init)
                optimizer = optimizer_for(model)
                calls = []
                original = optimizer.step
                def counted_step():
                    calls.append(1)
                    return original()
                optimizer.step = counted_step
                result = train(model, optimizer, x, y, ids, meta, Path(tmp) / (schedule + ".pt"))
                self.assertEqual(len(calls), 3)
                self.assertEqual(result["current_update"], 3)
        self.assertEqual(15000 // 20, 750)

    def test_targets_unchanged_by_noise(self):
        bases = np.asarray([[0, 1, 2, 3, 0, 1, 2]])
        clean_x, clean_y = copies(bases, 20, np.random.RandomState(1), 0)
        noisy_x, noisy_y = copies(bases, 20, np.random.RandomState(2), 100)
        np.testing.assert_array_equal(clean_y, noisy_y)
        np.testing.assert_array_equal(clean_x[:, 0], clean_y + 1)
        self.assertFalse(np.array_equal(clean_x, noisy_x))

    def test_known_metrics(self):
        targets = np.zeros((3, 7), dtype=np.int64)
        logits = np.zeros((3, 7, 4), dtype=np.float32)
        logits[0, :, 0] = 10
        logits[1, :3, 0] = 10
        logits[1, 3:, 1] = 10
        logits[2, :, 1] = 10
        rows = {row["group"]: row for row in summarize(logits, targets, [0, 1, 2, 3, 4, 5, 6])}
        self.assertAlmostEqual(rows["all"]["action_accuracy"], 10 / 21)
        self.assertAlmostEqual(rows["all"]["sequence_accuracy"], 1 / 3)
        self.assertEqual(rows["shared"]["action_count"], 0)
        self.assertTrue(np.isnan(rows["shared"]["action_accuracy"]))
        zeros = summarize(np.zeros((1, 7, 4), dtype=np.float32), np.zeros((1, 7), dtype=np.int64))
        self.assertAlmostEqual(zeros[0]["cross_entropy"], np.log(4), places=6)

    def test_evaluation_does_not_modify_weights(self):
        model = MotorRNN()
        bases, _ = family(np.random.RandomState(9), 4)
        x, y = copies(bases, 2, np.random.RandomState(5), 0.1)
        before = state_hash(model)
        evaluate(model, x, y, [0, 1, 2, 3])
        self.assertEqual(before, state_hash(model))
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_forward_shape_and_source_timing(self):
        for exact, expected_calls in ((True, 8), (False, 7)):
            model = MotorRNN(source_exact=exact)
            calls = []
            handle = model.recurrence.register_forward_hook(lambda module, args, result: calls.append(args[0].shape))
            out = model(torch.ones(20, 1, 7))
            handle.remove()
            self.assertEqual(tuple(out.shape), (20, 7, 4))
            self.assertEqual(len(calls), expected_calls)
            self.assertTrue(all(tuple(shape) == (20, 1, 7) for shape in calls))
            torch.testing.assert_close(out, model(torch.ones(20, 1, 7)), rtol=0, atol=0)

    def test_author_model_numerical_parity_when_available(self):
        import importlib.util
        path = Path("external/author_repo/utils/NetworkModels.py")
        if not path.exists():
            self.skipTest("Audit-only local reference not present")
        spec = importlib.util.spec_from_file_location("author_model_audit", path)
        author = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(author)
        torch.manual_seed(232)
        reference = author.MotorLearningRNN(7, 7, 4, 7)
        torch.manual_seed(232)
        own = MotorRNN()
        x = torch.randn(20, 1, 7)
        a, b = reference(x), own(x)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        a.sum().backward()
        b.sum().backward()
        for left, right in zip(reference.parameters(), own.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)

    def test_final_blocked_order_balance(self):
        assignments = balanced_assignment(list(range(30)), [1, 2, 4, 7])
        for orders in assignments.values():
            self.assertEqual({order: list(orders.values()).count(order) for order in ORDERS}, {order: 5 for order in ORDERS})

    def test_resume_matches_uninterrupted_tiny_run(self):
        base, changed = family(np.random.RandomState(31), 2)
        x, y = copies(base, 40, np.random.RandomState(44), 0.1)
        initial = MotorRNN()
        full, interrupted = copy.deepcopy(initial), copy.deepcopy(initial)
        probes = {"noisy": copies(base, 2, np.random.RandomState(8), 0.1)}
        meta = self.metadata()
        with tempfile.TemporaryDirectory() as tmp:
            whole = train(full, optimizer_for(full), x, y, np.arange(len(x)), meta, Path(tmp) / "full.pt",
                          probes=probes, changed=changed, eval_every=1)
            train(interrupted, optimizer_for(interrupted), x, y, np.arange(len(x)), meta, Path(tmp) / "part.pt",
                  probes=probes, changed=changed, eval_every=1, stop_after=3)
            resumed = MotorRNN()
            continued = train(resumed, optimizer_for(resumed), x, y, np.arange(len(x)), meta, Path(tmp) / "part.pt",
                              probes=probes, changed=changed, eval_every=1, resume=True)
            self.assertEqual(state_hash(full), state_hash(resumed))
            self.assertEqual(whole["losses"], continued["losses"])
            self.assertEqual(whole["probes"], continued["probes"])
            with self.assertRaises(ValueError):
                train(MotorRNN(), optimizer_for(MotorRNN()), x, y, np.arange(len(x)),
                      {**meta, "config_hash": "wrong"}, Path(tmp) / "part.pt", resume=True)

    def test_seed_namespace_separation(self):
        bundles = [seed_bundle(phase, 0, 1) for phase in ("smoke_v1", "pilot_v1", "final_v1")]
        values = [seed for bundle in bundles for seed in bundle.values()]
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual(bundles[0], seed_bundle("smoke_v1", 0, 1))

    def test_sustained_threshold(self):
        self.assertEqual(sustained_threshold([(0, .89), (25, .91), (50, .92), (75, .90)]),
                         {"threshold_start": 25, "threshold_confirmed": 75})
        self.assertEqual(sustained_threshold([(0, .95), (25, .80), (50, .96)])["threshold_start"], "not_reached")

    @staticmethod
    def metadata():
        return {"config_hash": canonical_hash({"test": 1}), "data_manifest_hash": "tiny-data-hash", "run_id": 0,
                "distance": 2, "schedule": "tiny", "blocked_order": "ABC", "source_commit": "test", "stage": "main"}


if __name__ == "__main__":
    unittest.main()
