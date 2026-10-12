import math
from pathlib import Path
import unittest
import numpy as np
import shapely
import torch

from rockcut.benchmark import summarize, result_row
from rockcut.local import LocalOptimizer
from rockcut.problem import Problem
from rockcut.seeds import destroy_repair, random_layout, shelf_seeds
from rockcut.solver import SearchConfig, VARIANTS, solve
from rockcut.verify import validate


class ExperimentalGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_random_centers_are_inside_rotated_triangle_and_reproducible(self):
        p = Problem([[1, -2], [8, 4], [-2, 7]], 1)
        a = random_layout(p, 200, np.random.default_rng(19))
        b = random_layout(p, 200, np.random.default_rng(19))
        np.testing.assert_array_equal(a, b)
        self.assertTrue(shapely.contains_xy(p.polygon, a[:, 0], a[:, 1]).all())
        self.assertTrue(((a[:, 2] >= 0) & (a[:, 2] < np.pi)).all())

    def test_diverse_shelves_include_multiple_orientations(self):
        p = Problem.load("examples/slab.json")
        catalog = shelf_seeds(p, limit=8, diverse=True)
        angles = np.array([q[0, 2] for q in catalog if len(q)])
        self.assertGreater(len(angles), 3)
        for i, angle in enumerate(angles):
            self.assertTrue(np.all(abs((angles[i+1:] - angle + np.pi/2) % np.pi - np.pi/2) >= 0.035))

    def test_group_repair_preserves_count_unmoved_slots_and_input(self):
        p = Problem.load("examples/slab.json")
        q = shelf_seeds(p, limit=1)[0]
        before = q.copy()
        after = destroy_repair(p, q, np.random.default_rng(4))
        np.testing.assert_array_equal(q, before)
        self.assertEqual(after.shape, q.shape)
        self.assertTrue(np.isfinite(after).all())
        moved = np.any(after != before, axis=1).sum()
        self.assertGreaterEqual(moved, 1)
        self.assertLessEqual(moved, 8)
        expired = destroy_repair(p, q, np.random.default_rng(4), deadline=0)
        np.testing.assert_array_equal(expired, q)

    def test_adaptive_weights_boost_decay_cap_and_reset(self):
        p = Problem([[0, 0], [20, 0], [20, 20], [0, 20]], 1)
        local = LocalOptimizer(p)
        q = np.array([[0, 0, 0.2], [1, 0.3, 0.5]])
        before = local.penalty(torch.tensor(q[None])).item()
        local.update_weights(q)
        self.assertGreater(local.penalty(torch.tensor(q[None])).item(), before)
        for _ in range(30):
            self.assertLessEqual(local.update_weights(q), 8)
        weight = local.pair_weights.item()
        local.update_weights(np.array([[-3, 0, 0], [3, 0, 0]]))
        self.assertLess(local.pair_weights.item(), weight)
        local.reset_weights()
        self.assertAlmostEqual(local.penalty(torch.tensor(q[None])).item(), before)
        self.assertFalse(validate(p, p.denormalize(q))["accepted"])

    def test_weighted_gradient_matches_finite_difference(self):
        p = Problem([[0, 0], [10, 0], [10, 10], [0, 10]], 1)
        local = LocalOptimizer(p)
        q = torch.tensor([[[0, 0, 0.23], [1.8, 0.4, 0.61]]], dtype=torch.float64, requires_grad=True)
        local.update_weights(q.detach().numpy()[0])
        self.assertTrue(torch.autograd.gradcheck(lambda x: local.penalty(x, smooth=0.02), (q,), atol=1e-5))

    def test_every_variant_keeps_independent_acceptance_on_known_tiling(self):
        p = Problem([[0, 0], [6, 0], [6, 2], [0, 2]], 1)
        for variant, settings in VARIANTS.items():
            with self.subTest(variant=variant):
                r = solve(p, SearchConfig(grid_seed='off', seconds=3, seconds_per_n=1, **settings))
                self.assertEqual(r["n"], 4)
                self.assertTrue(validate(p, r["state"])["accepted"])


class BenchmarkMetricTests(unittest.TestCase):
    def test_failed_attempts_are_censored_and_paired_by_case_seed(self):
        rows = [
            dict(case_id="a", variant="baseline", seed=1, n=4, target_n=5, success=False, time_to_target=None, elapsed_seconds=20),
            dict(case_id="a", variant="baseline", seed=2, n=4, target_n=5, success=False, time_to_target=None, elapsed_seconds=20),
            dict(case_id="a", variant="repair", seed=1, n=5, target_n=5, success=True, time_to_target=4, elapsed_seconds=20),
            dict(case_id="a", variant="repair", seed=2, n=4, target_n=5, success=False, time_to_target=None, elapsed_seconds=20),
        ]
        summary = summarize(rows, ["baseline", "repair"], ["a"])
        original, repair = summary["by_case"]
        self.assertIsNone(original["median_success_time"])
        self.assertIsNone(original["ert_seconds"])
        self.assertEqual(repair["median_success_time"], 4)
        self.assertEqual(repair["ert_seconds"], 24)
        self.assertEqual(summary["overall"][1]["paired_wins"], 1)
        self.assertEqual(summary["overall"][1]["paired_ties"], 1)


if __name__ == "__main__":
    unittest.main()
