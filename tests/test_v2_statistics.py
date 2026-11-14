"""Numerical checks for pairing, multiplicity and interaction invariants."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from cruxrnn.statistics import holm, paired_summary, sign_flip, similarity_interaction


class StatisticsTests(unittest.TestCase):
    def test_holm_known_values(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_zero_difference_is_not_significant(self):
        result = paired_summary(np.ones(10), np.ones(10), np.random.RandomState(1), 1000)
        self.assertEqual(result["p_sign_flip"], 1)
        self.assertEqual(result["ci_low_pp"], 0)
        self.assertIsNone(result["paired_dz"])
        self.assertEqual(result["dz_bootstrap_undefined_count"], 1000)

    def test_pairing_and_sd(self):
        first = np.array([0.1, 0.2, 0.3, 0.4])
        second = first + np.array([0.01, 0.02, 0.03, 0.04])
        result = paired_summary(first, second, np.random.RandomState(2), 1000)
        self.assertAlmostEqual(result["difference_pp"], 2.5)
        self.assertAlmostEqual(result["sd_difference_pp"], np.std([1, 2, 3, 4], ddof=1))
        self.assertLessEqual(result["ci_low_pp"], 2.5)
        self.assertGreaterEqual(result["ci_high_pp"], 2.5)

    def test_missing_pairs_rejected(self):
        with self.assertRaises(ValueError):
            paired_summary([0.1, np.nan], [0.2, 0.3], np.random.RandomState(0))
        with self.assertRaises(ValueError):
            paired_summary([0.1], [0.2], np.random.RandomState(0))

    def test_interaction_removes_constant_schedule_effect(self):
        result = similarity_interaction(
            np.full((20, 4), 15.0), [1, 2, 4, 7], np.random.RandomState(3), 1000
        )
        self.assertAlmostEqual(result["slope_pp_per_hamming"], 0)
        self.assertEqual(result["p_categorical_interaction"], 1)

    def test_known_slope_and_whole_row_bootstrap(self):
        x = np.array([1, 2, 4, 7])
        values = np.arange(12)[:, None] + 3 * x
        result = similarity_interaction(values, x, np.random.RandomState(4), 1000)
        self.assertAlmostEqual(result["slope_pp_per_hamming"], 3)
        self.assertEqual(result["bootstrap_indices"].shape, (1000, 12))
        np.testing.assert_allclose(result["bootstrap_slopes"], 3)
        self.assertLess(result["p_slope_sign_flip"], 0.05)

    def test_seed_reproducibility(self):
        a = sign_flip(np.arange(20), np.random.RandomState(10), 1000)
        b = sign_flip(np.arange(20), np.random.RandomState(10), 1000)
        self.assertEqual(a, b)

    def test_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            holm([-0.1, 0.2])
