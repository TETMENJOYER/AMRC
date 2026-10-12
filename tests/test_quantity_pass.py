"""Queue and quota behavior, using real strict 2D/full-volume acceptance."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from rockcut.block import Block
from rockcut.cut_plan import parallel_plan
from rockcut.problem import Problem
from rockcut.quantity_pass import QuantityPassConfig, improve_candidate


def fixture(count=2):
    block = Block.from_dict({'schema_version': 1, 'units': 'mm',
        'base': {'type': 'box', 'size': [9, 3, count], 'center': [4.5, 1.5, count/2]}})
    plan = parallel_plan(block, [0, 0, 1], 1)
    problems = [Problem([[0, 0], [9, 0], [9, 3], [0, 3]], 1) for _ in plan.specs]
    records = [{'index': index, 'spec': spec.to_dict(), 'n': 2,
        'state': [[1.5, .5, 0.], [4.5, .5, 0.]], 'status': 'baseline',
        'search': {'attempts': [{'n': 3, 'status': 'not_found_within_budget',
                                'best_area_penalty': float(index+1)}]}}
        for index, spec in enumerate(plan.specs)]
    return block, {'plan': plan, 'problems': problems, 'records': records,
                   'machine_setup': {'frozen': 'source setup'}}


def result(initial, state=None, attempts=None):
    state = initial.copy() if state is None else state
    return {'n': len(state), 'state': np.asarray(state).tolist(),
        'attempts': attempts or [], 'stop_reason': 'search_budget_exhausted',
        'validation': {'accepted': True}}


class QuantityPassTests(unittest.TestCase):
    def test_failed_or_reduced_result_leaves_every_record_unchanged(self):
        block, candidate = fixture()
        before = copy.deepcopy(candidate['records'])
        clock = [0.]

        def search(problem, config, **kwargs):
            clock[0] += .1
            # A dishonest/mock solver must not replace an incumbent with fewer.
            return result(kwargs['initial_state'], kwargs['initial_state'][:1])

        with patch('rockcut.quantity_pass.time.monotonic', side_effect=lambda: clock[0]), \
             patch('rockcut.quantity_pass.solve', side_effect=search):
            summary = improve_candidate(block, candidate, 1, QuantityPassConfig(grid_seed='off'), 20.)
        self.assertEqual(candidate['records'], before)
        self.assertEqual(summary['before_n'], summary['after_n'])
        self.assertTrue(all(r['stop_reason'] == 'unexpected_candidate_count' for r in summary['visits']))

    def test_invalid_addition_rejected_despite_saved_accepted_flag(self):
        block, candidate = fixture(1)
        before = copy.deepcopy(candidate['records'])

        def search(problem, config, **kwargs):
            return result(kwargs['initial_state'], np.vstack((kwargs['initial_state'], [50., 50., 0.])))

        with patch('rockcut.quantity_pass.solve', side_effect=search):
            summary = improve_candidate(block, candidate, 1, QuantityPassConfig(grid_seed='off'), 1e20)
        self.assertEqual(candidate['records'], before)
        self.assertEqual(summary['visits'][0]['stop_reason'], 'independent_geometry_rejection')

    def test_full_assigned_piece_check_rejects_2d_valid_source_valid_addition(self):
        block, candidate = fixture(1)
        candidate['records'][0].update(n=1, state=[[1.5, .5, 0.]])
        candidate['plan'].pieces[0] = Block.from_dict({'schema_version': 1, 'units': 'mm',
            'base': {'type': 'box', 'size': [3, 3, 1], 'center': [1.5, 1.5, .5]}}).solid
        before = copy.deepcopy(candidate['records'])

        def search(problem, config, **kwargs):
            return result(kwargs['initial_state'], np.vstack((kwargs['initial_state'], [4.5, .5, 0.])))

        with patch('rockcut.quantity_pass.solve', side_effect=search):
            summary = improve_candidate(block, candidate, 1, QuantityPassConfig(grid_seed='off'), 1e20)
        self.assertEqual(candidate['records'], before)
        self.assertFalse(summary['visits'][0]['accepted_addition'])

    def test_early_success_releases_quota_and_next_target_waits_for_other_slabs(self):
        block, candidate = fixture()
        plan_before = copy.deepcopy(candidate['plan'].to_dict())
        setup_before = copy.deepcopy(candidate['machine_setup'])
        pieces_before = list(candidate['plan'].pieces)
        clock = [0.]
        cfgs = []

        def search(problem, config, **kwargs):
            cfgs.append(config)
            clock[0] += 1.
            initial = kwargs['initial_state']
            addition = [7.5, .5, 0.] if len(initial) == 2 else [1.5, 1.5, 0.]
            return result(initial, np.vstack((initial, addition)))

        with patch('rockcut.quantity_pass.time.monotonic', side_effect=lambda: clock[0]), \
             patch('rockcut.quantity_pass.solve', side_effect=search):
            summary = improve_candidate(block, candidate, 1, QuantityPassConfig(grid_seed='off'), 20.)
        self.assertEqual([r['slab_index'] for r in summary['visits']], [0, 1, 0, 1])
        self.assertEqual([r['target_n'] for r in summary['visits']], [3, 3, 4, 4])
        self.assertEqual(summary['final_counts'], [4, 4])
        self.assertEqual(summary['visits'][0]['released_quota_seconds'], 11.)
        self.assertEqual(summary['remaining_seconds'], 16.)
        self.assertEqual(candidate['plan'].to_dict(), plan_before)
        self.assertEqual(candidate['machine_setup'], setup_before)
        self.assertTrue(all(a is b for a, b in zip(candidate['plan'].pieces, pieces_before)))
        self.assertTrue(all(c.diverse_starts and c.group_repair and c.adaptive_penalties and c.retry_failed_n
                            and c.alignment_seconds == 0 for c in cfgs))

    def test_retries_stop_at_absolute_budget_and_log_unvisited_slabs(self):
        block, candidate = fixture()
        clock = [0.]
        attempts = []

        def search(problem, config, **kwargs):
            self.assertTrue(config.retry_failed_n)
            end = clock[0]+config.seconds
            while clock[0] < end:
                spent = min(config.seconds_per_n, end-clock[0])
                clock[0] += spent
                attempts.append({'n': kwargs['target_n'], 'status': 'not_found_within_budget',
                                 'seconds': spent, 'iterations': 1})
            return result(kwargs['initial_state'], attempts=attempts)

        with patch('rockcut.quantity_pass.time.monotonic', side_effect=lambda: clock[0]), \
             patch('rockcut.quantity_pass.solve', side_effect=search) as search_mock:
            summary = improve_candidate(block, candidate, 1, QuantityPassConfig(grid_seed='off'), 5.)
        self.assertEqual(search_mock.call_count, 1)
        self.assertEqual([a['seconds'] for a in attempts], [3., 2.])
        self.assertEqual(summary['not_visited_due_budget'], [1])
        self.assertEqual(summary['stop_reason'], 'global_budget_exhausted')
        self.assertEqual(summary['visits'][0]['stop_reason'], 'global_budget_exhausted')

    def test_long_and_retry_have_equal_visit_quota_and_other_features(self):
        quotas = []
        for policy in ('long', 'retry'):
            block, candidate = fixture(1)
            clock = [0.]

            def search(problem, config, **kwargs):
                quotas.append((config.seconds, config.seconds_per_n))
                self.assertEqual(config.retry_failed_n, policy == 'retry')
                self.assertTrue(config.diverse_starts and config.group_repair and config.adaptive_penalties)
                clock[0] += config.seconds
                return result(kwargs['initial_state'])

            with patch('rockcut.quantity_pass.time.monotonic', side_effect=lambda: clock[0]), \
                 patch('rockcut.quantity_pass.solve', side_effect=search):
                improve_candidate(block, candidate, 1, QuantityPassConfig(grid_seed='off', policy=policy), 60.)
        self.assertEqual(quotas, [(12., 12.), (12., 3.)])

    def test_generic_priority_uses_penalty_then_gap_without_fixed_slab_numbers(self):
        block, candidate = fixture(3)
        candidate['records'][0]['search']['attempts'][0]['best_area_penalty'] = .2
        candidate['records'][1]['search']['attempts'][0]['best_area_penalty'] = .01
        candidate['records'][2]['search'] = {}
        with patch('rockcut.quantity_pass.solve') as mock:
            summary = improve_candidate(block, candidate, 1, QuantityPassConfig(grid_seed='off'), -1.)
        mock.assert_not_called()
        self.assertEqual([r['slab_index'] for r in summary['queue']], [1, 0, 2])
        self.assertEqual(summary['not_visited_due_budget'], [1, 0, 2])

    def test_initial_validation_consumes_search_budget(self):
        block, candidate = fixture(1)
        clock = [0.]
        from rockcut.quantity_pass import check_candidate

        def audit(*args, **kwargs):
            clock[0] += 2.
            return check_candidate(*args, **kwargs)

        with patch('rockcut.quantity_pass.time.monotonic', side_effect=lambda: clock[0]), \
             patch('rockcut.quantity_pass.check_candidate', side_effect=audit), \
             patch('rockcut.quantity_pass.solve') as mock:
            summary = improve_candidate(block, candidate, 1, QuantityPassConfig(grid_seed='off'), 1.)
        mock.assert_not_called()
        self.assertEqual(summary['preparation_seconds'], 2.)
        self.assertEqual(summary['global_deadline_overrun_seconds'], 1.)
        self.assertEqual(summary['stop_reason'], 'global_budget_exhausted')

    def test_invalid_warm_state_raises_without_modifying_record(self):
        block, candidate = fixture(1)
        candidate['records'][0]['state'][0] = [50., 50., 0.]
        before = copy.deepcopy(candidate['records'])
        with patch('rockcut.quantity_pass.solve') as mock:
            with self.assertRaisesRegex(ValueError, 'independently valid initial'):
                improve_candidate(block, candidate, 1, QuantityPassConfig(grid_seed='off'), 1e20)
        mock.assert_not_called()
        self.assertEqual(candidate['records'], before)

    def test_preparation_quota_exhaustion_continues_other_slabs(self):
        block, candidate = fixture()
        clock = [0.]
        from rockcut.quantity_pass import check_candidate

        def audit(*args, **kwargs):
            clock[0] += 13.
            return check_candidate(*args, **kwargs)

        with patch('rockcut.quantity_pass.time.monotonic', side_effect=lambda: clock[0]), \
             patch('rockcut.quantity_pass.check_candidate', side_effect=audit), \
             patch('rockcut.quantity_pass.solve') as mock:
            summary = improve_candidate(block, candidate, 1, QuantityPassConfig(grid_seed='off'), 60.)
        mock.assert_not_called()
        self.assertEqual(summary['preparation_quota_skipped_slabs'], [0, 1])
        self.assertEqual(summary['remaining_seconds'], 34.)
        self.assertEqual(summary['stop_reason'], 'queue_completed')

    def test_reported_budget_exhaustion_with_time_remaining_has_honest_visit_stop(self):
        block, candidate = fixture(1)
        clock = [0.]

        def search(problem, config, **kwargs):
            clock[0] += 1.
            return result(kwargs['initial_state'], attempts=[{'n': 3, 'status': 'not_found_within_budget',
                'iterations': config.iterations_per_n}])

        with patch('rockcut.quantity_pass.time.monotonic', side_effect=lambda: clock[0]), \
             patch('rockcut.quantity_pass.solve', side_effect=search):
            summary = improve_candidate(block, candidate, 1, QuantityPassConfig(grid_seed='off', policy='long'), 60.)
        self.assertEqual(summary['visits'][0]['reported_stop'], 'search_budget_exhausted')
        self.assertEqual(summary['visits'][0]['stop_reason'], 'attempt_iteration_limit')
        self.assertEqual(summary['remaining_seconds'], 59.)

    def test_config_and_frozen_candidate_validation(self):
        for options in ({'policy': 'unknown'}, {'slab_seconds': 0}, {'attempt_seconds': float('nan')},
                        {'max_visits_per_slab': 0}, {'max_visits_per_slab': True}, {'device': 'missing'},
                        {'seed': -1}, {'iterations_per_n': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                QuantityPassConfig(grid_seed='off', **options)
        block, candidate = fixture(1)
        candidate['records'][0]['spec']['upper'] += .1
        with self.assertRaisesRegex(ValueError, 'frozen plan'):
            improve_candidate(block, candidate, 1, QuantityPassConfig(grid_seed='off'), 1e20)


if __name__ == '__main__':
    unittest.main()
