"""Exercise count retries and contact search through real geometry acceptance."""
import unittest
from unittest.mock import patch

import numpy as np

from rockcut.problem import Problem
from rockcut.solver import SearchConfig, solve
from rockcut.tabu import TabuMemory


class ControlledLocal:
    def __init__(self, succeed_on=None, clock=None):
        self.succeed_on = succeed_on
        self.clock = clock
        self.calls = 0
        self.margin = 1e-5
        self.margins = []

    def reset_weights(self):
        pass

    def relax(self, batch, steps, deadline):
        self.calls += 1
        self.margins.append(self.margin)
        if self.clock is not None:
            self.clock[0] += 0.1
        if self.succeed_on is not None and self.calls >= self.succeed_on:
            return np.zeros_like(batch)  # Exact fit after denormalization.
        return batch.copy()

    def polish(self, state, iterations, deadline):
        return state.copy()


class SearchPolicyTests(unittest.TestCase):
    def run_controlled(self, local, **overrides):
        problem = Problem([[0, 0], [3, 0], [3, 1], [0, 1]], 1)
        options = dict(seconds=5, seconds_per_n=1, batch_size=1, adam_steps=1,
                       polish_iterations=1, iterations_per_n=1, tabu_tenure=2)
        options.update(overrides)
        with (patch("rockcut.solver.LocalOptimizer", return_value=local),
              patch("rockcut.solver.shelf_seeds", return_value=[np.empty((0, 3))]),
              patch("rockcut.solver.fill_gaps", side_effect=lambda p, q, deadline: q),
              patch("rockcut.solver.insert_one", return_value=np.array([[10., 0., 0.]])),
              patch("rockcut.solver.TabuMemory", wraps=TabuMemory) as memory):
            result = solve(problem, SearchConfig(**options), target_n=1)
            memory_count = memory.call_count
        return result, memory_count

    def test_baseline_stops_after_first_failed_attempt(self):
        local = ControlledLocal(succeed_on=2)
        result, _ = self.run_controlled(local)
        self.assertEqual(local.calls, 1)
        self.assertEqual(result["n"], 0)
        self.assertFalse(result["best_unaccepted_candidate"]["validation"]["accepted"])

    def test_retries_preserve_tabu_and_age_it_across_attempts(self):
        # Same invalid layout: remembered at iteration 0, rejected at 1,
        # expired at 2. The fourth attempt then produces an actual exact fit.
        local = ControlledLocal(succeed_on=4)
        result, memory_count = self.run_controlled(local, retry_failed_n=True)
        self.assertEqual(memory_count, 1)
        self.assertEqual(result["search_counters"]["n_restarts"], 3)
        self.assertEqual([a["tabu_rejections"] for a in result["attempts"][:3]], [0, 1, 0])
        self.assertEqual(result["n"], 1)
        self.assertTrue(result["validation"]["accepted"])
        self.assertIsNone(result["best_unaccepted_candidate"])
        self.assertFalse(result["optimality_proven"])

    def test_retries_respect_total_budget(self):
        clock = [0.0]
        local = ControlledLocal(clock=clock)
        with patch("rockcut.solver.time.monotonic", side_effect=lambda: clock[0]):
            result, _ = self.run_controlled(local, seconds=0.25, seconds_per_n=0.1,
                                             retry_failed_n=True)
        self.assertEqual(local.calls, 3)
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["stop_reason"], "search_budget_exhausted")
        self.assertEqual(result["area_upper_bound"], 1)
        self.assertLessEqual(result["elapsed_seconds"], 0.35)

    def test_margin_reaches_zero_without_relaxing_acceptance(self):
        local = ControlledLocal()
        result, _ = self.run_controlled(local, anneal_margin=True,
                                         iterations_per_n=5, seconds_per_n=5)
        self.assertEqual(len(local.margins), 5)
        self.assertGreater(local.margins[0], 0)
        self.assertEqual(local.margins[-1], 0)
        self.assertTrue(all(a >= b for a, b in zip(local.margins, local.margins[1:])))
        self.assertEqual(result["n"], 0)
        self.assertFalse(result["best_unaccepted_candidate"]["validation"]["accepted"])


if __name__ == "__main__":
    unittest.main()
