"""Rigid initialization must retain a copied physical best and strict geometry."""
import unittest
from unittest.mock import patch

import numpy as np

from rockcut.block import Block
from rockcut.cut_plan import parallel_plan
from rockcut.packing3d import verify_layout
from rockcut.problem import Problem
from rockcut.solver import SearchConfig, solve
from rockcut.verify import check_candidate


PARAMETERS = {'angle_radians': 0., 'phase_x': 0., 'phase_y': 0.}


class ControlledLocal:
    def __init__(self, mutate=False, success=None):
        self.margin = 1e-5
        self.calls = 0
        self.mutate = mutate
        self.success = success

    def reset_weights(self):
        pass

    def relax(self, batch, steps, deadline):
        self.calls += 1
        if self.mutate:
            batch[:] = 100.
        if self.success is not None:
            return np.repeat(self.success[None, :, :], len(batch), axis=0)
        return batch.copy()

    def polish(self, state, iterations, deadline):
        return state.copy()


class GridFallbackTests(unittest.TestCase):
    def problem_and_warm(self):
        p = Problem([[0, 0], [20, 0], [20, 6], [0, 6]], .7)
        warm = np.array([[1.2345678901234567, 1.987654321012345, .01234567890123456],
                         [3.0123456789012345, 3.456789012345678, .23456789012345678]])
        self.assertTrue(check_candidate(p, warm)['accepted'])
        # Ensure the fixture exposes the precision loss the physical companion fixes.
        self.assertNotEqual(p.denormalize(p.normalize(warm)).tobytes(), warm.tobytes())
        return p, warm

    def controlled(self, problem, *, warm=None, target=None, grid=None, checker=None,
                   shelves=None, mutate_search=False, configure=None, use_default_grid=False,
                   successful_physical=None):
        clock = [0.]
        local = ControlledLocal(mutate_search, None if successful_physical is None
                                else problem.normalize(successful_physical))
        options = dict(seconds=5., seconds_per_n=1., batch_size=1, adam_steps=1,
                       polish_iterations=1, iterations_per_n=1, device='cpu', threads=1,
                       alignment_seconds=0., grid_seed='rigid', grid_seconds=1.)
        options.update(configure or {})
        if use_default_grid:
            options.pop('grid_seed')

        def proposal(p, q, rng):
            if mutate_search:
                q[:] = 100.
            return np.vstack((q, [100., 100., 0.]))

        def grid_call(p, deadline, **kwargs):
            return iter(()) if grid is None else grid(p, deadline, clock)

        def shelf_call(p, **kwargs):
            return [] if shelves is None else shelves(p, clock, kwargs)

        with (patch('rockcut.solver.time.monotonic', side_effect=lambda: clock[0]),
              patch('rockcut.solver.LocalOptimizer', return_value=local),
              patch('rockcut.solver.rigid_grid_candidates', side_effect=grid_call) as generator,
              patch('rockcut.solver.shelf_seeds', side_effect=shelf_call),
              patch('rockcut.solver.fill_gaps', side_effect=lambda p, q, deadline: q),
              patch('rockcut.solver.insert_one', side_effect=proposal)):
            result = solve(problem, SearchConfig(**options), initial_state=warm,
                           target_n=target, acceptance_check=checker)
        return result, local, generator

    def assert_retained(self, result, expected, source, before_n=None):
        state = np.asarray(result['state'], dtype=np.float64).reshape(-1, 3)
        expected = np.asarray(expected, dtype=np.float64).reshape(-1, 3)
        self.assertEqual(state.tobytes(), expected.tobytes())
        self.assertEqual(result['n'], len(expected))
        self.assertTrue(result['validation']['accepted'])
        self.assertFalse(result['optimality_proven'])
        info = result['best_retention']
        self.assertEqual(info['source'], source)
        self.assertEqual(info['before_n'], len(expected) if before_n is None else before_n)
        self.assertEqual(info['after_n'], result['n'])
        self.assertEqual(info['returned_reason'], result['stop_reason'])
        self.assertTrue(info['count_preserved'])

    def test_global_deadline_returns_exact_warm_better_than_grid(self):
        p, warm = self.problem_and_warm()
        def grid(problem, deadline, clock):
            yield warm[:1].copy(), PARAMETERS.copy()
            clock[0] = 6.
        result, local, _ = self.controlled(p, warm=warm, target=3, grid=grid)
        self.assert_retained(result, warm, 'verified_warm_start')
        self.assertEqual(result['stop_reason'], 'global_budget_exhausted')
        self.assertEqual(local.calls, 0)
        self.assertEqual(result['grid_initialization']['best_grid_n'], 1)
        self.assertEqual(result['grid_initialization']['stop_reason'], 'grid_quota_exhausted')

    def test_first_failed_n_returns_exact_warm_with_time_remaining(self):
        p, warm = self.problem_and_warm()
        result, local, _ = self.controlled(p, warm=warm, target=3)
        self.assert_retained(result, warm, 'verified_warm_start')
        self.assertEqual(local.calls, 1)
        self.assertEqual(result['stop_reason'], 'first_failed_n_iteration_limit')
        self.assertGreater(result['stop_detail']['remaining_seconds'], 0)

    def test_grid_and_shelf_ties_do_not_replace_warm(self):
        p, warm = self.problem_and_warm()
        tie = np.array([[2.5, 4.5, 0.], [6.7, 4.5, 0.]])
        self.assertTrue(check_candidate(p, tie)['accepted'])
        result, _, _ = self.controlled(p, warm=warm, target=3,
            grid=lambda p, deadline, clock: iter([(tie.copy(), PARAMETERS.copy())]),
            shelves=lambda p, clock, kwargs: [p.normalize(tie)])
        self.assert_retained(result, warm, 'verified_warm_start')
        self.assertEqual(result['grid_initialization']['best_grid_n'], 2)
        self.assertEqual([h['source'] for h in result['history'] if h['event'] == 'feasible'],
                         ['verified_warm_start'])

    def test_mutating_input_alias_and_search_starts_cannot_damage_saved_warm(self):
        p, warm = self.problem_and_warm(); expected = warm.copy()
        def grid(problem, deadline, clock):
            warm[:] = 500.
            return iter(())
        result, local, _ = self.controlled(p, warm=warm, target=3, grid=grid, mutate_search=True)
        self.assertFalse(np.array_equal(warm, expected))
        self.assertEqual(local.calls, 1)
        self.assert_retained(result, expected, 'verified_warm_start')

    def test_grid_buffer_reuse_cannot_damage_accepted_physical_best(self):
        p = Problem([[0, 0], [6, 0], [6, 2], [0, 2]], 1)
        buffer = np.array([[1.5, .5, 0.]])
        expected = buffer.copy()
        def grid(problem, deadline, clock):
            yield buffer, PARAMETERS.copy()
            buffer[:] = 100.
        result, _, _ = self.controlled(p, grid=grid)
        self.assertFalse(np.array_equal(buffer, expected))
        self.assert_retained(result, expected, 'rigid_grid', before_n=0)

    def test_accepted_n_plus_one_updates_the_same_physical_best(self):
        p, warm = self.problem_and_warm()
        improved = np.vstack((warm, [10.5, 5., 0.]))
        self.assertTrue(check_candidate(p, improved)['accepted'])
        result, local, _ = self.controlled(p, warm=warm, target=3,
                                          successful_physical=improved)
        # Local search authorizes the denormalized optimized candidate, not the
        # warm-start companion. The final companion must track that new best.
        verified_improved = p.denormalize(p.normalize(improved))
        self.assert_retained(result, verified_improved, 'tabu_gradient', before_n=2)
        self.assertEqual(result['stop_reason'], 'target_reached')
        self.assertEqual(local.calls, 1)
        self.assertEqual(result['attempts'][0]['status'], 'feasible')

    def test_grid_larger_than_explicit_target_is_capped_before_acceptance(self):
        p = Problem([[0, 0], [9, 0], [9, 1], [0, 1]], 1)
        candidate = np.array([[1.5, .5, 0.], [4.5, .5, 0.], [7.5, .5, 0.]])
        self.assertTrue(check_candidate(p, candidate)['accepted'])
        result, local, _ = self.controlled(p, warm=candidate[:1], target=2,
            grid=lambda p, deadline, clock: iter([(candidate, PARAMETERS.copy())]))
        self.assert_retained(result, candidate[:2], 'rigid_grid', before_n=1)
        self.assertEqual(result['stop_reason'], 'target_reached')
        self.assertEqual(local.calls, 0)
        self.assertLessEqual(result['grid_initialization']['best_grid_n'], result['target_n'])
        self.assertEqual(result['grid_initialization']['candidates'][0]['verified_n'], 2)

    def test_any_nonempty_outside_cell_is_removed_before_grid_acceptance(self):
        p = Problem([[0, 0], [6, 0], [6, 2], [0, 2]], 1)
        candidate = np.array([[1.5, .5, 0.], [4.5+1e-9, .5, 0.]])
        self.assertFalse(check_candidate(p, candidate)['accepted'])
        result, _, _ = self.controlled(p,
            grid=lambda p, deadline, clock: iter([(candidate, PARAMETERS.copy())]))
        self.assert_retained(result, candidate[:1], 'rigid_grid', before_n=0)
        visited = result['grid_initialization']['candidates'][0]
        self.assertEqual((visited['original_n'], visited['verified_n']), (2, 1))
        self.assertTrue(check_candidate(p, result['state'])['accepted'])

    def test_exact_boundary_and_pair_contacts_are_still_allowed(self):
        p = Problem([[0, 0], [6, 0], [6, 1], [0, 1]], 1)
        candidate = np.array([[1.5, .5, 0.], [4.5, .5, 0.]])
        result, local, _ = self.controlled(p,
            grid=lambda p, deadline, clock: iter([(candidate, PARAMETERS.copy())]))
        self.assert_retained(result, candidate, 'rigid_grid', before_n=0)
        self.assertEqual(result['stop_reason'], 'area_upper_bound_reached')
        self.assertEqual(local.calls, 0)

    def test_2d_cover_cannot_hide_full_thickness_exit_in_tapered_source(self):
        # Remove x+z>6 from a 6x4x2 box: bottom width6, top width4, volume40.
        block = Block.from_dict({'schema_version': 1, 'units': 'mm',
            'base': {'type': 'box', 'size': [6, 4, 2]},
            'subtract': [{'type': 'box', 'center': [9, 2, 3],
                          'size': [20, 8, 6*np.sqrt(2)], 'rotation_degrees': [0, 45, 0]}]})
        self.assertAlmostEqual(block.solid.volume(), 40., places=9)
        plan = parallel_plan(block, [0, 0, 1], 2)
        self.assertEqual(len(plan.specs), 1)
        p = Problem([[0, 0], [6, 0], [6, 4], [0, 4]], 1)
        candidate = np.array([[4.5, .5, 0.]])
        self.assertTrue(check_candidate(p, candidate)['accepted'])
        seen = []
        def checker(state):
            check = verify_layout(block, plan.specs[0], 1, state, slab_local=plan.pieces[0])
            seen.append((len(state), check['accepted']))
            return check
        bad = checker(candidate)
        self.assertFalse(bad['accepted'])
        self.assertTrue(bad['whole_thickness_verified'])
        result, _, _ = self.controlled(p, checker=checker,
            grid=lambda p, deadline, clock: iter([(candidate, PARAMETERS.copy())]))
        self.assert_retained(result, np.empty((0, 3)), 'empty', before_n=0)
        self.assertIn((1, False), seen)
        self.assertEqual(result['grid_initialization']['best_grid_n'], 0)
        self.assertTrue(result['validation']['additional_check']['whole_thickness_verified'])

    def test_impossible_thin_region_returns_zero_without_positive_count_promise(self):
        p = Problem([[0, 0], [20, 0], [20, .5], [0, .5]], 1)
        self.assertGreater(p.area_upper_bound, 0)
        result, _, _ = self.controlled(p,
            grid=lambda p, deadline, clock: iter([(np.array([[1.5, .25, 0.]]), PARAMETERS.copy())]))
        self.assert_retained(result, np.empty((0, 3)), 'empty', before_n=0)
        self.assertEqual(result['grid_initialization']['best_grid_n'], 0)

    def test_explicit_off_does_not_call_grid(self):
        p = Problem([[0, 0], [6, 0], [6, 1], [0, 1]], 1)
        warm = np.array([[1.5, .5, 0.]])
        def forbidden(*args):
            raise AssertionError('off mode invoked rigid grid')
        result, _, generator = self.controlled(p, warm=warm, target=1,
            grid=forbidden, configure={'grid_seed': 'off'})
        generator.assert_not_called()
        self.assertNotIn('grid_initialization', result)
        self.assertNotIn('best_retention', result)
        self.assertEqual(result['n'], 1)

    def test_default_grid_retains_exact_warm_on_worse_or_tied_candidates(self):
        p, warm = self.problem_and_warm()
        def grid(problem, deadline, clock):
            yield warm[:1].copy(), PARAMETERS.copy()
            yield warm.copy(), PARAMETERS.copy()
            clock[0] = 6.
        result, _, generator = self.controlled(p, warm=warm,
            grid=grid, use_default_grid=True)
        generator.assert_called_once()
        self.assert_retained(result, warm, 'verified_warm_start')
        self.assertEqual(result['config']['grid_seed'], 'rigid')
        self.assertEqual(result['grid_initialization']['best_grid_n'], len(warm))
        self.assertEqual([c['verified_n'] for c in result['grid_initialization']['candidates']], [1, 2])

    def test_rigid_rejects_target_below_warm_but_off_keeps_legacy_truncation(self):
        p, warm = self.problem_and_warm()
        with self.assertRaisesRegex(ValueError, 'warm|discard|target'):
            self.controlled(p, warm=warm, target=1)
        result, _, _ = self.controlled(p, warm=warm, target=1, configure={'grid_seed': 'off'})
        self.assertEqual(result['n'], 1)
        self.assertEqual(result['stop_reason'], 'target_reached')

    def test_generator_exception_is_not_masked_by_an_existing_valid_warm_start(self):
        p, warm = self.problem_and_warm()
        def broken(*args):
            raise RuntimeError('deliberate grid enumeration bug')
        with self.assertRaisesRegex(RuntimeError, 'grid enumeration bug'):
            self.controlled(p, warm=warm, target=3, grid=broken)

    def test_external_validator_program_exception_is_not_successful_fallback(self):
        p = Problem([[0, 0], [6, 0], [6, 1], [0, 1]], 1)
        candidate = np.array([[1.5, .5, 0.], [4.5, .5, 0.]])
        def broken_check(state):
            if len(state) == 2:
                raise RuntimeError('deliberate source checker bug')
            return {'accepted': True}
        with self.assertRaisesRegex(RuntimeError, 'source checker bug'):
            self.controlled(p, warm=candidate[:1], checker=broken_check,
                grid=lambda p, deadline, clock: iter([(candidate, PARAMETERS.copy())]))

    def test_grid_allocation_is_inside_the_shared_initialization_budget(self):
        p = Problem([[0, 0], [6, 0], [6, 1], [0, 1]], 1)
        deadlines = []
        def grid(problem, deadline, clock):
            deadlines.append(deadline)
            yield np.array([[1.5, .5, 0.]]), PARAMETERS.copy()
            clock[0] = .31
        result, _, _ = self.controlled(p, grid=grid, target=1, configure={'seconds': 4.})
        self.assertEqual(deadlines, [.3])
        info = result['grid_initialization']
        self.assertEqual(info['requested_seconds'], 1.)
        self.assertEqual(info['allocated_seconds'], .3)
        self.assertLessEqual(info['allocated_seconds'], result['config']['seconds']*.15)
        self.assertEqual(info['stop_reason'], 'grid_quota_exhausted')
        self.assertAlmostEqual(info['overrun_seconds'], .01)
        self.assertEqual(len(info['candidates']), 1)
        self.assertEqual(result['stop_reason'], 'target_reached')


if __name__ == '__main__':
    unittest.main()
