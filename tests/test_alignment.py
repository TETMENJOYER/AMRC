import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from shapely.geometry import box, MultiPolygon

from rockcut.alignment import (AlignmentConfig, ConvergenceMonitor, GroupedLayout,
                              angle_groups, angles_not_worse, align_layout, regularity)
from rockcut.block import Block, Frame
from rockcut.slab3d import SlabSpec
from rockcut.packing3d import verify_layout
from rockcut.problem import Problem
from rockcut.postprocess import align_result
from rockcut.solver import SearchConfig, solve


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.problem = Problem([[0, 0], [20, 0], [20, 15], [0, 15]], 1)
        self.state = np.array([[4, 4, .15], [9, 4.3, -.12], [4.1, 9, .09], [9, 9.2, -.08]])

    def test_gradient_improves_spread_angles_without_rebuilding_or_losing_count(self):
        result = align_layout(self.problem, self.state, AlignmentConfig(seconds=8, steps=100, rebuild=False))
        self.assertTrue(result['alignment']['improved'])
        self.assertLess(result['alignment']['after']['score'], result['alignment']['before']['score']*.2)
        self.assertTrue(result['validation']['accepted'])
        self.assertEqual(result['n'], 4)
        self.assertGreater(result['alignment']['counters']['gradient_steps'], 0)

    def test_complete_reorganization_preserves_count_and_input(self):
        original = self.state.copy()
        result = align_layout(self.problem, self.state, AlignmentConfig(seconds=3, steps=20))
        self.assertTrue(result['alignment']['improved'])
        self.assertEqual(len(result['state']), len(original))
        np.testing.assert_array_equal(self.state, original)
        np.testing.assert_array_equal(result['alignment']['original_state'], original)
        self.assertTrue(result['validation']['accepted'])

    def test_additional_3d_check_rejects_rebuild_in_a_covering_but_larger_region(self):
        block = Block.from_dict({'schema_version': 1, 'units': 'mm',
            'base': {'type': 'box', 'size': [13, 13, 4]}})
        spec = SlabSpec(Frame.from_dict({}), 1, 3)
        checker = lambda q: verify_layout(block, spec, 1, q)
        # A tempting, regular, strictly 2D-valid proposal lies outside source.
        proposal = self.problem.normalize(np.array([[16, 2, 0], [16, 4, 0], [16, 6, 0], [16, 8, 0]]))
        with patch('rockcut.alignment.shelf_seeds', return_value=[proposal]):
            result = align_layout(self.problem, self.state, AlignmentConfig(seconds=4, steps=1), checker)
        self.assertGreater(result['alignment']['counters']['rejected_geometry'], 0)
        self.assertEqual(result['n'], 4)
        self.assertTrue(result['validation']['additional_check']['accepted'])
        self.assertTrue(checker(result['state'])['accepted'])

    def test_unaccepted_initial_layout_is_rejected_instead_of_salvaged(self):
        with self.assertRaisesRegex(ValueError, 'strictly verified'):
            align_layout(self.problem, np.vstack((self.state, self.state[0])))
        with self.assertRaisesRegex(ValueError, 'strictly verified'):
            align_layout(self.problem, self.state, acceptance_check=lambda q: {'accepted': False})

    def test_already_aligned_contact_grid_is_preserved_exactly(self):
        p = Problem([[0, 0], [6, 0], [6, 2], [0, 2]], 1)
        state = np.array([[1.5, .5, 0], [4.5, .5, 0], [1.5, 1.5, 0], [4.5, 1.5, 0]])
        result = align_layout(p, state)
        np.testing.assert_array_equal(result['state'], state)
        self.assertFalse(result['alignment']['improved'])
        self.assertEqual(result['alignment']['counters']['gradient_steps'], 0)
        self.assertTrue(result['validation']['accepted'])

    def test_empty_and_disconnected_regions(self):
        self.assertEqual(align_layout(self.problem, [])['n'], 0)
        p = Problem.from_region(MultiPolygon([box(0, 0, 6, 3), box(10, 0, 16, 3)]), 1)
        state = [[2, 1, 0], [12, 1, 0]]
        result = align_layout(p, state, AlignmentConfig(seconds=1, steps=10))
        self.assertEqual(result['n'], 2)
        self.assertTrue(result['validation']['accepted'])

    def test_metrics_ignore_permutation_translation_rotation_and_uniform_scaling(self):
        reference = regularity(self.problem, self.state)
        permuted = regularity(self.problem, self.state[[3, 1, 0, 2]])
        theta = .73
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        moved = self.state.copy()
        moved[:, :2] = self.state[:, :2]@rot.T+40
        moved[:, 2] += theta
        other = regularity(self.problem, moved)
        scaled = Problem(self.problem.vertices*7, 7)
        large = self.state.copy(); large[:, :2] *= 7
        resized = regularity(scaled, large)
        for field in ('score', 'direction_dispersion', 'mean_angle_error_degrees'):
            self.assertAlmostEqual(reference[field], permuted[field], places=12)
            self.assertAlmostEqual(reference[field], other[field], places=12)
            self.assertAlmostEqual(reference[field], resized[field], places=12)

    def test_orthogonal_angles_are_one_direction_family(self):
        metrics = regularity(self.problem, [[3, 3, 0], [8, 3, np.pi/2]])
        self.assertAlmostEqual(metrics['direction_dispersion'], 0)
        self.assertEqual(metrics['direction_families_1deg'], 1)

    def test_invalid_config_and_state(self):
        for kwargs in ({'seconds': 0}, {'seconds': float('nan')}, {'steps': 0}, {'rebuild': 1}):
            with self.assertRaises(ValueError):
                AlignmentConfig(**kwargs)
        with self.assertRaises(ValueError):
            align_layout(self.problem, [[2, 2, float('nan')]])

    def test_saved_external_verification_cannot_be_silently_dropped(self):
        saved = {'problem': self.problem.to_dict(), 'n': 4, 'state': self.state.tolist(),
                 'angle_unit': 'radians', 'validation': {'additional_check': {'accepted': True}}}
        before = copy.deepcopy(saved)
        with self.assertRaisesRegex(ValueError, 'original checker'):
            align_result(saved)
        self.assertEqual(saved, before)

    def test_solver_second_stage_runs_after_count_search(self):
        result = solve(self.problem, SearchConfig(grid_seed='off', seconds=.1, alignment_seconds=1),
                       target_n=4, initial_state=self.state)
        self.assertEqual(result['n'], 4)
        self.assertTrue(result['alignment']['count_preserved'])
        self.assertTrue(result['validation']['accepted'])
        self.assertEqual(result['alignment']['stage'], 'post_search_fixed_count')

    def test_whole_block_alignment_retains_plan_balance_and_every_slab_count(self):
        from rockcut.block_solver import BlockSearchConfig, solve_block
        from rockcut.postprocess import align_block_result
        from rockcut.verify_block import verify_block_result
        inputs = {'schema_version': 1, 'kind': 'block_cutting_problem',
            'block': {'schema_version': 1, 'units': 'mm', 'base': {'type': 'box', 'size': [6, 2, 7]}},
            'product': {'short_side': 1, 'thickness': 3}, 'directions': [[0, 0, 1]]}
        original = solve_block(inputs, BlockSearchConfig(grid_seed='off', seconds=2, refine_pairs=0,
                               coarse_seconds_per_slab=.1, fine_seconds_per_slab=.1, alignment_seconds=.1))
        self.assertEqual(original['history'][-1]['event'], 'block_finished')
        self.assertTrue(original['alignment']['count_preserved'])
        saved = copy.deepcopy(original)
        result = align_block_result(original, AlignmentConfig(seconds=1))
        self.assertGreater(result['n_total'], 0)
        self.assertEqual(result['plan'], saved['plan'])
        self.assertEqual(result['volume_balance'], saved['volume_balance'])
        self.assertEqual([r['n'] for r in result['slabs']], [r['n'] for r in saved['slabs']])
        self.assertEqual(original, saved)
        self.assertTrue(verify_block_result(json.loads(json.dumps(result)))['accepted'])

    def test_cli_align_exports_recheckable_result(self):
        from rockcut.cli import main
        from rockcut.verify import validate
        result = solve(self.problem, SearchConfig(grid_seed='off', seconds=.1), target_n=4, initial_state=self.state)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'input.json').write_text(json.dumps(result), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(['align', str(root/'input.json'), '--output', str(root/'out'), '--seconds', '1',
                               '--method-policy', 'always'])
            self.assertEqual(status, 0)
            exported = json.loads((root/'out/result.json').read_text(encoding='utf-8'))
            self.assertEqual(exported['n'], result['n'])
            self.assertEqual(exported['alignment']['method_policy'], 'always')
            self.assertTrue(validate(self.problem, exported['state'])['accepted'])
            self.assertTrue((root/'out/report.html').exists())
            self.assertTrue((root/'out/placements.csv').exists())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_gradient_retains_full_3d_verified_count(self):
        # Cold CUDA library/kernel loading can exceed the small search budget.
        # Exercise the same kernels before timing the actual improvement test.
        align_layout(self.problem, self.state,
            AlignmentConfig(seconds=30, steps=1, device='cuda', rebuild=False, stop_mode='time'))
        block = Block.from_dict({'schema_version': 1, 'units': 'mm',
            'base': {'type': 'box', 'size': [20, 15, 4]}})
        spec = SlabSpec(Frame.from_dict({}), 1, 3)
        result = align_layout(self.problem, self.state,
            AlignmentConfig(seconds=8, steps=80, device='cuda', rebuild=False),
            lambda q: verify_layout(block, spec, 1, q))
        self.assertTrue(result['alignment']['improved'])
        self.assertEqual(result['n'], 4)
        self.assertTrue(result['validation']['additional_check']['accepted'])


class AngleProtectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.problem = Problem([[0, 0], [20, 0], [20, 15], [0, 15]], 1)

    def test_edge_reward_cannot_buy_an_angle_regression(self):
        original = np.array([[4, 4, 0.], [8, 4.4, 0.]])
        tempting = np.array([[4, 4, 0.], [7.1, 4, .001]])
        before, worse_angles = regularity(self.problem, original), regularity(self.problem, tempting)
        self.assertLess(worse_angles['score'], before['score'])
        self.assertFalse(angles_not_worse(worse_angles, before))
        with patch('rockcut.alignment.shelf_seeds', return_value=[self.problem.normalize(tempting)]):
            result = align_layout(self.problem, original, AlignmentConfig(steps=10, max_rounds=3))
        self.assertGreater(result['alignment']['counters']['rejected_angle_regressions'], 0)
        self.assertTrue(angles_not_worse(result['alignment']['after'], before))
        self.assertLessEqual(result['alignment']['after']['max_angle_error_degrees'], 1e-8)
        self.assertEqual(result['alignment']['phases'][0]['completed_rounds'], 0)
        self.assertEqual(result['n'], 2)

    def test_angle_priority_may_improve_angles_even_with_a_larger_legacy_score(self):
        original = np.array([[4, 4, .01], [7.5, 4, -.01]])
        aligned = np.array([[4, 4, 0.], [8, 4.4, 0.]])
        checker = lambda q: {'accepted': bool(np.array_equal(q, original) or np.array_equal(q, aligned))}
        with patch('rockcut.alignment.shelf_seeds', return_value=[self.problem.normalize(aligned)]):
            result = align_layout(self.problem, original, AlignmentConfig(steps=1), checker)
        np.testing.assert_array_equal(result['state'], aligned)
        self.assertTrue(result['alignment']['improved'])
        self.assertGreater(result['alignment']['after']['score'], result['alignment']['before']['score'])
        self.assertEqual(result['alignment']['after']['mean_angle_error_degrees'], 0)
        self.assertTrue(result['validation']['additional_check']['accepted'])

    def test_group_partition_handles_orthogonal_turns_and_circle_seam(self):
        q = np.array([[2, 2, 0], [5, 2, np.pi/2], [8, 2, np.pi/2-1e-12],
                      [3, 7, .3], [7, 7, .3+np.pi/2]])
        labels, bases, turns = angle_groups(q)
        self.assertEqual(len(bases), 2)
        np.testing.assert_array_equal(labels[:3], np.repeat(labels[0], 3))
        self.assertEqual(labels[3], labels[4])
        perm = np.array([4, 2, 0, 3, 1])
        other = angle_groups(q[perm])[0]
        np.testing.assert_array_equal((labels[:, None] == labels)[perm][:, perm], other[:, None] == other)
        moved = q.copy(); moved[:, 2] += .71
        rotated = angle_groups(moved)[0]
        np.testing.assert_array_equal(labels[:, None] == labels, rotated[:, None] == rotated)
        self.assertTrue(np.isfinite(bases).all())

    def test_group_parameters_keep_parallel_items_coupled_during_descent(self):
        q = np.array([[4, 4, .1], [8, 4, .1+np.pi/2], [4, 9, .35], [9, 9, .35+np.pi/2]])
        model = GroupedLayout(q, 'angles', 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        for _ in range(10):
            optimizer.zero_grad()
            model.values().square().sum().backward()
            optimizer.step()
        angles = model.values().detach()[0, :, 2].numpy()
        self.assertAlmostEqual(angles[1]-angles[0], np.pi/2, places=12)
        self.assertAlmostEqual(angles[3]-angles[2], np.pi/2, places=12)
        self.assertEqual(len(model.rotation), 2)

    def test_edge_phase_common_rotation_preserves_all_relative_group_angles(self):
        q = np.array([[4, 4, .1], [8, 4, .1+np.pi/2], [4, 9, .35], [9, 9, .35+np.pi/2]])
        model = GroupedLayout(q, 'edges', 'cpu')
        with torch.no_grad():
            model.rotation.add_(.4)
        actual = model.values().detach()[0].numpy()
        np.testing.assert_allclose(actual[:, 2]-actual[0, 2], q[:, 2]-q[0, 2], atol=1e-14)
        self.assertEqual(len(model.rotation), 1)

    def test_geometric_repair_cannot_unlock_individual_angles(self):
        from rockcut.local import LocalOptimizer
        import time
        q = np.array([[4, 4, .1], [4.2, 4, .1+np.pi/2], [10, 9, .1]])
        model = GroupedLayout(q, 'edges', 'cpu')
        repaired = model.repair(LocalOptimizer(self.problem), time.monotonic()+10)
        delta = (repaired[:, 2]-repaired[0, 2])-(q[:, 2]-q[0, 2])
        np.testing.assert_allclose(delta, 0., atol=1e-14)

    def test_full_angle_phase_and_edge_phase_do_not_regress_any_protected_metric(self):
        q = np.array([[4, 4, .15], [9, 4.3, -.12], [4.1, 9, .09], [9, 9.2, -.08]])
        result = align_layout(self.problem, q, AlignmentConfig(steps=60, rebuild=False))
        a = result['alignment']
        self.assertTrue(angles_not_worse(a['after'], a['before']))
        self.assertTrue(angles_not_worse(a['phases'][1]['after'], a['phases'][1]['before']))
        self.assertEqual([p['phase'] for p in a['phases']], ['angles', 'edges'])
        self.assertEqual(a['completed_rounds'], sum(p['completed_rounds'] for p in a['phases']))
        self.assertEqual(result['n'], len(q))


class ConvergenceTests(unittest.TestCase):
    def test_default_is_metric_stopping_without_a_time_cap(self):
        cfg = AlignmentConfig()
        self.assertIsNone(cfg.seconds)
        self.assertEqual(cfg.stop_mode, 'convergence')
        self.assertEqual(cfg.method_policy, 'adaptive')

    def test_one_failed_round_does_not_stop_a_later_improvement(self):
        monitor = ConvergenceMonitor(1., AlignmentConfig(seconds=None))
        self.assertFalse(monitor.observe(1.))
        self.assertFalse(monitor.observe(.8))
        self.assertFalse(monitor.observe(.7))
        self.assertFalse(monitor.observe(.7))
        self.assertFalse(monitor.observe(.7))
        self.assertTrue(monitor.observe(.7))
        self.assertAlmostEqual(monitor.history[-1]['window_log_gain'], 0.)

    def test_relative_test_works_across_orders_of_magnitude(self):
        for scale in (1e-8, 1., 1e8):
            monitor = ConvergenceMonitor(scale, AlignmentConfig(seconds=None, absolute_gain=0))
            self.assertFalse(monitor.observe(scale*.9995))
            self.assertFalse(monitor.observe(scale*.999))
            self.assertTrue(monitor.observe(scale*.9985))
            self.assertAlmostEqual(monitor.history[-1]['window_relative_gain'], .0015)

    def test_absolute_threshold_handles_near_zero_with_finite_logs(self):
        monitor = ConvergenceMonitor(1e-10, AlignmentConfig(seconds=None))
        self.assertFalse(monitor.observe(9e-11))
        self.assertFalse(monitor.observe(8e-11))
        self.assertTrue(monitor.observe(0.))
        self.assertTrue(np.isfinite(monitor.history[-1]['window_log_gain']))
        extreme = ConvergenceMonitor(1e300, AlignmentConfig())
        extreme.observe(1e200); extreme.observe(1e100); extreme.observe(0.)
        self.assertTrue(np.isfinite(extreme.history[-1]['window_log_gain']))

    def test_gradual_substantial_improvements_do_not_trigger_plateau(self):
        monitor = ConvergenceMonitor(1., AlignmentConfig(seconds=None))
        for score in (0.8, 0.6, 0.5, 0.4, 0.3, 0.2):
            self.assertFalse(monitor.observe(score))

    def test_invalid_and_increasing_best_scores_are_rejected(self):
        monitor = ConvergenceMonitor(1., AlignmentConfig(seconds=None))
        for score in (float('nan'), float('inf'), -1., 1.1):
            with self.assertRaises(ValueError):
                monitor.observe(score)
        for kwargs in ({'window': 0}, {'relative_gain': -.1}, {'absolute_gain': float('nan')},
                       {'max_rounds': 0}, {'stop_mode': 'unknown'}, {'score_floor': 0}):
            with self.assertRaises(ValueError):
                AlignmentConfig(**kwargs)

    def test_rejected_gradient_improvements_cannot_reset_the_verified_plateau(self):
        torch.set_num_threads(1)
        p = Problem([[0, 0], [20, 0], [20, 15], [0, 15]], 1)
        original = np.array([[4, 4, .15], [9, 4.3, -.12], [4.1, 9, .09], [9, 9.2, -.08]])
        result = align_layout(p, original, AlignmentConfig(seconds=None, steps=10, rebuild=False),
                              lambda q: {'accepted': bool(np.array_equal(q, original))})
        self.assertEqual(result['alignment']['stop_reason'], 'plateau')
        self.assertEqual(result['alignment']['completed_rounds'], 6)
        self.assertEqual([p['completed_rounds'] for p in result['alignment']['phases']], [3, 3])
        self.assertEqual([p['metric'] for p in result['alignment']['phases']], ['angle_error_score', 'edge_error'])
        np.testing.assert_array_equal(result['state'], original)
        self.assertGreater(result['alignment']['counters']['rejected_geometry'], 0)
        self.assertTrue(result['validation']['accepted'])

    def test_round_safety_limit_is_not_reported_as_convergence(self):
        torch.set_num_threads(1)
        p = Problem([[0, 0], [20, 0], [20, 15], [0, 15]], 1)
        original = np.array([[4, 4, .15], [9, 4.3, -.12]])
        result = align_layout(p, original, AlignmentConfig(seconds=None, steps=1, rebuild=False, max_rounds=1),
                              lambda q: {'accepted': bool(np.array_equal(q, original))})
        self.assertEqual(result['alignment']['stop_reason'], 'round_limit')
        self.assertEqual(result['alignment']['completed_rounds'], 1)

    def test_time_safety_limit_does_not_claim_a_verified_plateau(self):
        p = Problem([[0, 0], [20, 0], [20, 15], [0, 15]], 1)
        original = [[4, 4, .15], [9, 4.3, -.12]]
        clock_values = iter([0.]+[5.]*100)
        with patch('rockcut.alignment.time.monotonic', side_effect=lambda: next(clock_values)):
            result = align_layout(p, original, AlignmentConfig(seconds=1, rebuild=False))
        self.assertEqual(result['alignment']['stop_reason'], 'time_safety_limit')
        self.assertEqual(result['alignment']['completed_rounds'], 0)
        self.assertEqual(result['alignment']['convergence_history'], [])
        np.testing.assert_array_equal(result['state'], original)
        self.assertTrue(result['validation']['accepted'])

    def test_saved_block_without_time_limit_uses_plateau_policy(self):
        from rockcut.block_solver import BlockSearchConfig, solve_block
        from rockcut.postprocess import align_block_result
        inputs = {'schema_version': 1, 'kind': 'block_cutting_problem',
            'block': {'schema_version': 1, 'units': 'mm', 'base': {'type': 'box', 'size': [6, 2, 7]}},
            'product': {'short_side': 1, 'thickness': 3}, 'directions': [[0, 0, 1]]}
        source = solve_block(inputs, BlockSearchConfig(grid_seed='off', seconds=2, refine_pairs=0))
        result = align_block_result(source, AlignmentConfig(seconds=None))
        self.assertEqual(result['n_total'], source['n_total'])
        self.assertEqual(result['alignment']['slabs_not_processed'], 0)
        self.assertIsNone(result['alignment']['budget_seconds'])
        self.assertTrue(result['validation']['accepted'])


if __name__ == '__main__':
    unittest.main()
