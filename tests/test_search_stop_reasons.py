"""Search exit diagnostics must describe the stopping rule, not failed N as impossible."""
import unittest
from unittest.mock import patch

import numpy as np

from rockcut.problem import Problem
from rockcut.solver import SearchConfig, solve


class ControlledLocal:
    def __init__(self, clock, duration=0., succeed_after=None):
        self.clock = clock
        self.duration = duration
        self.succeed_after = succeed_after
        self.calls = 0
        self.margin = 1e-5

    def reset_weights(self):
        pass

    def relax(self, batch, steps, deadline):
        self.calls += 1
        self.clock[0] += self.duration
        return (np.zeros_like(batch) if self.succeed_after is not None
                and self.calls >= self.succeed_after else batch.copy())

    def polish(self, state, iterations, deadline):
        return state.copy()


class SearchStopReasonTests(unittest.TestCase):
    def controlled(self, *, duration=0., succeed_after=None, initial=None,
                   target=1, seeds=None, insertion=None, accept=None, **options):
        problem = Problem([[0, 0], [3, 0], [3, 2], [0, 2]], 1)
        clock = [0.]
        local = ControlledLocal(clock, duration, succeed_after)
        settings = dict(seconds=5., seconds_per_n=1., batch_size=1,
                        adam_steps=1, polish_iterations=1, iterations_per_n=1,
                        device='cpu', threads=1, alignment_seconds=0.)
        settings.update(options)
        if insertion is None:
            insertion = lambda p, q, rng: np.vstack((q, [10., 0., 0.]))
        with (patch('rockcut.solver.time.monotonic', side_effect=lambda: clock[0]),
              patch('rockcut.solver.LocalOptimizer', return_value=local),
              patch('rockcut.solver.shelf_seeds', return_value=[] if seeds is None else seeds),
              patch('rockcut.solver.fill_gaps', side_effect=lambda p, q, deadline: q),
              patch('rockcut.solver.insert_one', side_effect=insertion)):
            result = solve(problem, SearchConfig(grid_seed='off', **settings), target_n=target,
                           initial_state=initial,
                           acceptance_check=None if accept is None else lambda q: accept(q, clock))
        return result, local

    def assert_honest_failure(self, result, reason):
        self.assertEqual(result['stop_reason'], reason)
        self.assertEqual(result['history'][-1]['stop_reason'], reason)
        self.assertEqual(result['attempts'][-1]['status'], 'not_found_within_budget')
        self.assertTrue(result['validation']['accepted'])
        self.assertFalse(result['optimality_proven'])
        self.assertEqual(result['area_upper_bound'], 2)

    def test_first_failed_n_iteration_limit_leaves_global_time_unused(self):
        result, local = self.controlled()
        self.assert_honest_failure(result, 'first_failed_n_iteration_limit')
        self.assertEqual(local.calls, 1)
        self.assertEqual(result['n'], 0)
        self.assertEqual(result['stop_detail']['failed_n'], 1)
        self.assertEqual(result['stop_detail']['remaining_seconds'], 5.)
        self.assertEqual([h['event'] for h in result['history']],
                         ['start', 'attempt', 'iteration', 'finished'])
        self.assertFalse(result['best_unaccepted_candidate']['validation']['accepted'])

    def test_attempt_deadline_is_distinct_from_global_deadline(self):
        result, local = self.controlled(duration=.2, seconds_per_n=.1, iterations_per_n=2)
        self.assert_honest_failure(result, 'first_failed_n_attempt_budget_exhausted')
        self.assertEqual(local.calls, 1)
        self.assertEqual(result['attempts'][0]['iterations'], 1)
        self.assertEqual(result['stop_detail']['failure_reason'], 'attempt_budget_exhausted')
        self.assertAlmostEqual(result['stop_detail']['remaining_seconds'], 4.8)

    def test_retries_reach_actual_global_deadline(self):
        result, local = self.controlled(duration=.1, seconds=.25, seconds_per_n=.1,
                                       retry_failed_n=True)
        self.assert_honest_failure(result, 'global_budget_exhausted')
        self.assertEqual(local.calls, 3)
        self.assertEqual(result['search_counters']['n_restarts'], 2)
        self.assertEqual(result['stop_detail']['remaining_seconds'], 0.)

    def test_retry_success_reports_target_and_keeps_legacy_failure_record(self):
        result, local = self.controlled(succeed_after=2, retry_failed_n=True)
        self.assertEqual(result['stop_reason'], 'target_reached')
        self.assertEqual(result['n'], 1)
        self.assertTrue(result['validation']['accepted'])
        self.assertEqual(local.calls, 2)
        self.assertEqual([a['status'] for a in result['attempts']],
                         ['not_found_within_budget', 'feasible'])
        self.assertIsNone(result['stop_detail']['failed_n'])
        self.assertIsNone(result['stop_detail']['failure_reason'])

    def test_target_and_area_success_keep_existing_reason_precedence(self):
        warm = np.array([[1.5, .5, 0.]])
        result, local = self.controlled(initial=warm)
        self.assertEqual(result['stop_reason'], 'target_reached')
        self.assertEqual(local.calls, 0)
        np.testing.assert_array_equal(result['state'], warm)
        seeds = [np.array([[0., -.5, 0.], [0., .5, 0.]])]
        for target in (None, 2):
            with self.subTest(target=target):
                result, local = self.controlled(target=target, seeds=seeds)
                self.assertEqual(result['stop_reason'], 'area_upper_bound_reached')
                self.assertEqual(result['n'], 2)
                self.assertTrue(result['validation']['accepted'])
                self.assertEqual(local.calls, 0)

    def test_final_audit_time_cannot_relabel_first_failed_n(self):
        calls = []
        def accept(state, clock):
            calls.append(len(state))
            if len(calls) == 2:  # Initial empty state, then final saved-state audit.
                clock[0] += 10.
            return {'accepted': True}
        result, _ = self.controlled(accept=accept)
        self.assert_honest_failure(result, 'first_failed_n_iteration_limit')
        self.assertEqual(calls, [0, 0])
        self.assertGreater(result['search_elapsed_seconds'], result['config']['seconds'])
        self.assertEqual(result['stop_detail']['search_phase_elapsed_seconds'], 0.)
        self.assertEqual(result['stop_detail']['remaining_seconds'], 5.)

    def test_failed_n_does_not_replace_the_verified_warm_state(self):
        warm = np.array([[1.5, .5, 0.]])
        result, _ = self.controlled(initial=warm, target=2)
        self.assert_honest_failure(result, 'first_failed_n_iteration_limit')
        self.assertEqual(result['n'], 1)
        self.assertEqual(result['stop_detail']['failed_n'], 2)
        np.testing.assert_array_equal(result['state'], warm)

    def test_no_full_initial_proposal_has_a_separate_reason(self):
        # A failed proposal that overshoots N is not an exhausted time budget.
        result, local = self.controlled(insertion=lambda p, q, rng: np.ones((2, 3)))
        self.assert_honest_failure(result, 'first_failed_n_initialization_failed')
        self.assertEqual(local.calls, 0)
        self.assertEqual(result['attempts'][0]['iterations'], 0)
        self.assertEqual(result['stop_detail']['remaining_seconds'], 5.)


if __name__ == '__main__':
    unittest.main()
