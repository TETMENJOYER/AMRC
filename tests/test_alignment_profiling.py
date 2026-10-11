import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from rockcut import postprocess
from rockcut.alignment import AlignmentConfig
from rockcut.problem import Problem
from scripts.profile_alignment import AlignmentProfiler, instrument, MethodMonitor, method_stopping


class AlignmentProfilingTests(unittest.TestCase):
    def test_instrumentation_preserves_layout_history_and_stops(self):
        torch.set_num_threads(1)
        problem = Problem([[0, 0], [20, 0], [20, 15], [0, 15]], 1)
        source = {'problem': problem.to_dict(), 'angle_unit': 'radians', 'n': 4,
                  'state': [[4, 4, .15], [9, 4.3, -.12], [4.1, 9, .09], [9, 9.2, -.08]]}
        cfg = AlignmentConfig(steps=10, max_rounds=2, rebuild=False, device='cpu')
        expected = postprocess.align_result(source, cfg)
        profiler = AlignmentProfiler()
        with tempfile.TemporaryDirectory() as directory:
            with instrument(profiler, Path(directory)):
                actual = postprocess.align_result(source, cfg)
        np.testing.assert_array_equal(actual['state'], expected['state'])
        for key in ('before', 'after', 'counters', 'stop_reason', 'completed_rounds'):
            self.assertEqual(actual['alignment'][key], expected['alignment'][key])
        clean = lambda rows: [{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in rows]
        self.assertEqual(clean(actual['alignment']['history']), clean(expected['alignment']['history']))
        self.assertEqual(clean(actual['alignment']['convergence_history']), clean(expected['alignment']['convergence_history']))
        self.assertEqual(sum(r['proposals'] for r in profiler.methods.values()),
                         actual['alignment']['counters']['proposals'])
        self.assertEqual(sum(r['accepted'] for r in profiler.methods.values()),
                         actual['alignment']['counters']['accepted_improvements'])
        self.assertFalse(profiler.stack)

    def test_experimental_monitor_preserves_active_gradient_and_geometry_checks(self):
        torch.set_num_threads(1)
        problem = Problem([[0, 0], [20, 0], [20, 15], [0, 15]], 1)
        source = {'problem': problem.to_dict(), 'angle_unit': 'radians', 'n': 4,
                  'state': [[4, 4, .15], [9, 4.3, -.12], [4.1, 9, .09], [9, 9.2, -.08]]}
        cfg = AlignmentConfig(steps=10, max_rounds=2, rebuild=False, device='cpu')
        calls = []
        def checker(state):
            calls.append(np.array(state))
            return {'accepted': True}
        expected = postprocess.align_result(source, cfg, checker)
        count = len(calls)
        calls.clear()
        monitor = MethodMonitor(999)
        with tempfile.TemporaryDirectory() as directory:
            with method_stopping(monitor, Path(directory)):
                actual = postprocess.align_result(source, cfg, checker)
        np.testing.assert_array_equal(actual['state'], expected['state'])
        self.assertEqual(actual['alignment']['counters'], expected['alignment']['counters'])
        self.assertEqual(len(calls), count)
        self.assertTrue(monitor.rows)
        self.assertTrue(all(not row['skipped'] for row in monitor.rows))


if __name__ == '__main__':
    unittest.main()
