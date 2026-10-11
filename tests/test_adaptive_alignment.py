import unittest
from unittest.mock import patch

import numpy as np
import torch

from rockcut.adaptive_alignment import GradientController, would_stall
from rockcut.alignment import AlignmentConfig, ConvergenceMonitor, align_layout, angles_not_worse
from rockcut.block import Block, Frame
from rockcut.packing3d import verify_layout
from rockcut.problem import Problem
from rockcut.slab3d import SlabSpec


class AdaptiveAlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = AlignmentConfig(method_policy='adaptive')
        self.state = np.array([[3., 3., 0.], [7., 3.4, 0.]])
        self.metrics = {'edge_error': .1, 'exact_direction_groups': 1}
        self.controller = GradientController(self.cfg, 'edges', 'edge_error')

    def attempt(self, accepted=None, completed=True, round_number=1):
        self.controller.begin(self.state, self.metrics)
        self.controller.finish(round_number, 'active', accepted or {'grouped_gradient': 0, 'grouped_gradient_repaired': 0},
                               self.metrics, .1, completed)

    def pause(self):
        for i in range(3):
            self.attempt(round_number=i+1)
        self.assertTrue(self.controller.paused)

    def test_lbfgs_success_keeps_chain_active_even_when_adam_is_empty(self):
        self.attempt(); self.attempt()
        self.attempt({'grouped_gradient': 0, 'grouped_gradient_repaired': 1})
        self.assertFalse(self.controller.paused)
        self.assertEqual(self.controller.failures, 0)

    def test_partial_attempt_does_not_prove_method_stagnation(self):
        for _ in range(5):
            self.attempt(completed=False)
        self.assertFalse(self.controller.paused)
        self.assertEqual(self.controller.failures, 0)

    def test_successful_probe_resumes_and_phase_controllers_are_independent(self):
        self.pause()
        angle_controller = GradientController(self.cfg, 'angles', 'angle_error_score')
        self.assertFalse(angle_controller.paused)
        shifted = self.state.copy(); shifted[:, 0] += .1
        metrics = dict(self.metrics, edge_error=.09)
        reason = self.controller.decide(shifted, metrics, 4)
        self.controller.begin(shifted, metrics)
        self.controller.finish(4, reason, {'grouped_gradient': 0, 'grouped_gradient_repaired': 1},
                               dict(metrics, edge_error=.08), .1)
        self.assertFalse(self.controller.paused)
        self.assertEqual(self.controller.decide(shifted, metrics, 5), 'active')
        self.assertEqual(self.controller.summary()['resumptions'], 1)
        self.assertEqual(self.controller.failures, 0)

    def test_identical_state_does_not_trigger_periodic_deterministic_repetition(self):
        self.pause()
        for round_number in range(4, 12):
            self.controller.skip(round_number)
        self.assertIsNone(self.controller.decide(self.state, self.metrics, 12, last_round=True))

    def test_tiny_changes_do_not_restart_but_periodic_probe_returns(self):
        self.pause()
        shifted = self.state.copy(); shifted[0, 0] += .001
        metrics = dict(self.metrics, edge_error=.09999)
        for i in range(3):
            self.assertIsNone(self.controller.decide(shifted, metrics, i+4))
            self.controller.skip(i+4)
        self.assertEqual(self.controller.decide(shifted, metrics, 7), 'periodic_probe')
        self.controller.begin(shifted, metrics)
        self.controller.finish(7, 'periodic_probe', {'grouped_gradient': 0, 'grouped_gradient_repaired': 0}, metrics, .1)
        self.assertTrue(self.controller.paused)
        self.assertEqual(self.controller.skipped, 0)

    def test_significant_external_change_and_plateau_confirmation(self):
        self.pause()
        shifted = self.state.copy(); shifted[:, 0] += .1
        self.assertEqual(self.controller.decide(shifted, dict(self.metrics, edge_error=.09), 4),
                         'substantial_external_change')
        tiny = self.state.copy(); tiny[0, 0] += 1e-8
        self.assertIsNone(self.controller.decide(tiny, self.metrics, 4))
        self.assertEqual(self.controller.decide(tiny, self.metrics, 4, confirmation=True), 'plateau_confirmation')
        self.assertEqual(self.controller.decide(tiny, self.metrics, 12, last_round=True), 'round_limit_probe')
        self.assertIsNone(self.controller.decide(self.state, self.metrics, 4, confirmation=True))

    def test_preview_matches_phase_monitor_without_recording_partial_history(self):
        monitor = ConvergenceMonitor(.1, self.cfg)
        for score in (.099, .098, .098, .098, .098):
            count = len(monitor.history)
            preview = would_stall(monitor, score)
            self.assertEqual(len(monitor.history), count)
            self.assertEqual(preview, monitor.observe(score))

    def test_resumed_invalid_geometry_and_angle_proposals_cannot_replace_fallback(self):
        problem = Problem([[0, 0], [20, 0], [20, 15], [0, 15]], 1)
        original = np.array([[4, 4, 0.], [8, 4.4, 0.]])
        block = Block.from_dict({'schema_version': 1, 'units': 'mm', 'base': {'type': 'box', 'size': [13, 13, 4]}})
        spec = SlabSpec(Frame.from_dict({}), 1, 3)
        outside = np.array([[16, 2, 0.], [16, 4, 0.]])
        bad_angles = outside.copy(); bad_angles[1, 2] = .001
        geometric_failures = []
        def checker(q):
            audit = verify_layout(block, spec, 1, q)
            if not audit['accepted']:
                geometric_failures.append(q.copy())
            audit['accepted'] = audit['accepted'] and np.array_equal(q, original)
            return audit
        def schedule(controller, state, metrics, round_number, **kwargs):
            return None if round_number == 2 else 'periodic_probe' if round_number == 3 else 'active'
        with patch('torch.optim.Adam.step', lambda optimizer: None), \
             patch('rockcut.alignment.GroupedLayout.repair', side_effect=[problem.normalize(outside), problem.normalize(bad_angles)]), \
             patch.object(GradientController, 'decide', schedule):
            result = align_layout(problem, original, AlignmentConfig(method_policy='adaptive', steps=1, rebuild=False), checker)
        np.testing.assert_array_equal(result['state'], original)
        self.assertEqual(result['n'], 2)
        self.assertTrue(result['validation']['additional_check']['accepted'])
        self.assertTrue(angles_not_worse(result['alignment']['after'], result['alignment']['before']))
        self.assertGreater(result['alignment']['counters']['rejected_geometry'], 0)
        self.assertGreater(result['alignment']['counters']['rejected_angle_regressions'], 0)
        self.assertTrue(geometric_failures)

    def test_time_policy_still_runs_all_methods_once(self):
        controller = GradientController(AlignmentConfig(method_policy='adaptive', stop_mode='time'), 'edges', 'edge_error')
        self.assertFalse(controller.adaptive)
        self.assertEqual(controller.decide(self.state, self.metrics, 1), 'always')

    def test_bad_scheduler_settings_are_rejected(self):
        for kwargs in ({'method_policy': 'never'}, {'method_window': 0}, {'probe_interval': 0},
                       {'method_relative_gain': float('nan')}, {'method_center_rms': -.1}):
            with self.assertRaises(ValueError):
                AlignmentConfig(**kwargs)

    def test_search_interfaces_validate_and_preserve_previous_method_policy(self):
        from rockcut.solver import SearchConfig
        from rockcut.block_solver import BlockSearchConfig
        for config_type in (SearchConfig, BlockSearchConfig):
            self.assertEqual(config_type().alignment_method_policy, 'adaptive')
            self.assertEqual(config_type(alignment_method_policy='always').alignment_method_policy, 'always')
            with self.assertRaises(ValueError):
                config_type(alignment_method_policy='never')


if __name__ == '__main__':
    unittest.main()
