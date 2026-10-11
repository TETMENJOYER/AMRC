import json
import tempfile
import unittest
import contextlib
import io
from pathlib import Path
import numpy as np
import torch

from rockcut.problem import Problem
from rockcut.geometry import area_score, corners
from rockcut.local import LocalOptimizer
from rockcut.tabu import equivalent, TabuMemory
from rockcut.verify import validate
from rockcut.solver import SearchConfig, solve
from rockcut.seeds import insert_one
from rockcut.cli import main as cli_main


def square(side=10):
    return Problem([[0, 0], [side, 0], [side, side], [0, side]], 1)


class GeometryTests(unittest.TestCase):
    def test_default_area_and_upper_bound(self):
        p = Problem.load("examples/slab.json")
        self.assertAlmostEqual(p.area, 830000)
        self.assertEqual(p.area_upper_bound, 27)

    def test_invalid_polygons_and_dimensions(self):
        for vertices in ([[0, 0], [2, 2], [0, 2], [2, 0]], [[0, 0], [1, 0], [2, 0]]):
            with self.assertRaises(ValueError):
                Problem(vertices, 1)
        with self.assertRaises(ValueError):
            Problem([[0, 0], [1, 0], [0, 1]], 0)

    def test_clockwise_and_closed_polygon(self):
        p = Problem([[0, 0], [0, 4], [4, 4], [4, 0], [0, 0]], 1)
        self.assertTrue(p.polygon.exterior.is_ccw)
        self.assertEqual(len(p.vertices), 4)

    def test_unsupported_production_settings(self):
        data = square().to_dict()
        for key, value in [("kerf", 1), ("allowance", 2), ("aspect_ratio", 2), ("touching_allowed", False)]:
            with self.assertRaises(ValueError):
                Problem.from_dict({**data, key: value})

    def test_crossed_rectangles_area(self):
        p = square()
        score = area_score(p, np.array([[0, 0, 0], [0, 0, np.pi / 2]]))
        self.assertAlmostEqual(score.overlap, 1)
        self.assertEqual(score.outside, 0)

    def test_coincident_rectangles_are_invalid(self):
        p = square()
        state = np.array([[5, 5, 0], [5, 5, np.pi]])
        check = validate(p, state)
        self.assertFalse(check["accepted"])
        self.assertAlmostEqual(check["overlap_area_sum"], 3)

    def test_edge_and_corner_contact_allowed(self):
        p = square()
        state = np.array([[1.5, 0.5, 0], [4.5, 0.5, 0], [4.5, 1.5, 0]])
        self.assertEqual(validate(p, state)["status"], "feasible")

    def test_outside_area(self):
        p = square()
        state = np.array([[1, 0.5, 0]])
        check = validate(p, state)
        self.assertAlmostEqual(check["outside_area_sum"], 0.5)
        self.assertAlmostEqual(check["max_boundary_violation"], 0.5)

    def test_near_feasible_is_not_accepted(self):
        p = square()
        state = np.array([[1.5, 0.5, 0], [4.5 - 1e-9, 0.5, 0]])
        check = validate(p, state, length_tolerance=1e-6, area_tolerance=1e-6)
        self.assertEqual(check["status"], "near_feasible")
        self.assertFalse(check["accepted"])

    def test_area_tolerance_cannot_hide_linear_violation(self):
        check = validate(square(), np.array([[1.5 - 1e-5, 0.5, 0]]),
                         length_tolerance=1e-8, area_tolerance=10)
        self.assertEqual(check["status"], "infeasible")

    def test_empty_and_nonfinite_state(self):
        self.assertTrue(validate(square(), np.empty((0, 3)))["accepted"])
        with self.assertRaises(ValueError):
            validate(square(), [[np.nan, 0, 0]])

    def test_rigid_transform_invariance(self):
        p = square()
        state = np.array([[3, 3, 0.1], [7, 7, 0.7]])
        angle = 0.43
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        translation = np.array([300, -200])
        moved = Problem(p.vertices @ rot.T + translation, 1)
        q = state.copy()
        q[:, :2] = q[:, :2] @ rot.T + translation
        q[:, 2] += angle
        self.assertTrue(validate(moved, q)["accepted"])

    def test_normalization_roundtrip(self):
        p = Problem.load("examples/slab.json")
        state = np.array([[310, 200, 0.3], [600, 430, 1.2]])
        np.testing.assert_allclose(p.denormalize(p.normalize(state)), state)

    def test_incremental_insertion_equals_full_scoring(self):
        p = square()
        state = np.array([[0, 0, 0], [0.1, 0.2, 0.3]])
        fast = insert_one(p, state, np.random.default_rng(9), choices=20)
        rng = np.random.default_rng(9)
        candidates = []
        for _ in range(20):
            xy = rng.uniform(p.normalized_vertices.min(0), p.normalized_vertices.max(0))
            angle = rng.uniform(0, np.pi) if rng.random() < 0.4 else rng.choice([0, np.pi / 2])
            candidates.append(np.vstack((state, [*xy, angle])))
        expected = min(candidates, key=lambda q: area_score(p, q).total)
        np.testing.assert_array_equal(fast, expected)


class TabuTests(unittest.TestCase):
    def setUp(self):
        self.state = np.array([[0, 0, 0.01], [3, 2, 1.1], [7, 5, 2.2]])

    def test_permutations_and_half_turns(self):
        b = self.state[[2, 0, 1]].copy()
        b[:, 2] += np.pi
        self.assertTrue(equivalent(self.state, b, 0.01, 0.01))

    def test_wrap_boundary(self):
        a = np.array([[0, 0, 0.001]])
        b = np.array([[0, 0, np.pi - 0.001]])
        self.assertTrue(equivalent(a, b, 0.01, 0.01))
        b[0, 2] = np.pi / 2
        self.assertFalse(equivalent(a, b, 0.01, 0.01))

    def test_full_matching_not_greedy(self):
        a = np.array([[0, 0, 0], [0.1, 0, 0]])
        b = np.array([[0.05, 0, 0], [-0.1, 0, 0]])
        self.assertTrue(equivalent(a, b, 0.11, 0.01))

    def test_one_large_move_is_not_averaged_away(self):
        b = self.state.copy()
        b[0, 0] += 0.1
        self.assertFalse(equivalent(self.state, b, 0.04, 0.01))

    def test_expiration_and_copy(self):
        memory = TabuMemory(tenure=2)
        q = self.state.copy()
        memory.add(q, 0)
        q[0, 0] += 10
        self.assertTrue(memory.contains(self.state, 1))
        self.assertFalse(memory.contains(self.state, 2))


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gradient_finite_difference(self):
        local = LocalOptimizer(square())
        q = torch.tensor([[[0, 0, 0.23], [1.8, 0.4, 0.61]]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z: local.penalty(z, smooth=0.02), (q,), atol=1e-5))
        self.assertTrue(torch.autograd.gradcheck(lambda z: local.penalty(z), (q,), atol=1e-5))

    def test_sat_agrees_with_polygon_geometry(self):
        p = square()
        local = LocalOptimizer(p, margin=0)
        rng = np.random.default_rng(42)
        batch = rng.uniform([-6, -6, 0], [6, 6, np.pi], size=(100, 3, 3))
        values = local.penalty(torch.tensor(batch)).detach().numpy()
        for q, value in zip(batch, values):
            self.assertEqual(value > 1e-10, area_score(p, q).total > 1e-10)

    def test_local_correction_reaches_strict_feasibility(self):
        p = square()
        local = LocalOptimizer(p)
        state = np.array([[0, 0, 0.1], [2.4, 0.1, -0.1]])
        corrected = local.polish(local.relax(state[None], 100)[0], 150)
        self.assertLess(area_score(p, corrected).total, area_score(p, state).total)
        self.assertTrue(validate(p, p.denormalize(corrected))["accepted"])

    def test_adam_candidates_are_independent(self):
        local = LocalOptimizer(square())
        a = np.array([[0, 0, 0.2], [1.7, 0.3, 0.6]])
        b = np.array([[0.4, 0.8, 0.5], [2.1, 1.4, 0.2]])
        alone = local.relax(a[None], 20)[0]
        batched = local.relax(np.stack((a, b)), 20)[0]
        np.testing.assert_allclose(alone, batched, atol=1e-10)

    def test_known_tiling_and_export_roundtrip(self):
        p = Problem([[0, 0], [6, 0], [6, 2], [0, 2]], 1)
        result = solve(p, SearchConfig(seconds=4, seconds_per_n=1, batch_size=2, adam_steps=10))
        self.assertEqual(result["n"], 4)
        self.assertTrue(result["area_bound_reached"])
        self.assertFalse(result["optimality_proven"])
        loaded = json.loads(json.dumps(result))
        self.assertTrue(validate(Problem.from_dict(loaded["problem"]), loaded["state"])["accepted"])

    def test_fixed_count_zero(self):
        result = solve(square(), SearchConfig(seconds=1), target_n=0)
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["stop_reason"], "target_reached")

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            SearchConfig(seconds=-1)
        with self.assertRaises(ValueError):
            solve(square(), target_n=1000)

    def test_failed_search_does_not_claim_impossibility_or_reduce_bound(self):
        p = Problem([[0, 0], [20, 0], [20, 0.5], [0, 0.5]], 1)
        result = solve(p, SearchConfig(seconds=0.5, seconds_per_n=0.3,
                                      batch_size=2, adam_steps=10, polish_iterations=10), target_n=1)
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["area_upper_bound"], 3)
        self.assertFalse(result["optimality_proven"])
        self.assertTrue(result["validation"]["accepted"])
        self.assertEqual(result["stop_reason"], "search_budget_exhausted")

    def test_cli_rejects_tampered_coordinates_despite_saved_accepted_flag(self):
        data = {"problem": square().to_dict(), "n": 2, "angle_unit": "radians",
                "state": [[3, 3, 0], [3, 3, 0]], "validation": {"accepted": True}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(["verify", str(path)]), 2)


if __name__ == "__main__":
    unittest.main()
