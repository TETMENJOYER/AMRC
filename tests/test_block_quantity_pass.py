"""Block integration keeps optional quantity work inside its declared budget."""
import copy
import unittest
from unittest.mock import patch

from rockcut.block_solver import BlockSearchConfig, solve_block
from rockcut.verify_block import verify_block_result


class BlockQuantityPassTests(unittest.TestCase):
    def setUp(self):
        self.inputs={'schema_version':1,'kind':'block_cutting_problem',
            'block':{'schema_version':1,'units':'mm','base':{'type':'box','size':[6,2,7]}},
            'product':{'short_side':1,'aspect_ratio':3,'thickness':3,'kerf':0,'allowance':0},
            'cut_policy':{'kind':'parallel_support_start','initial_trim_allowed':False,'kerf':0},
            'directions':[[0,0,1]]}

    def config(self, **extra):
        return BlockSearchConfig(grid_seed='off', seconds=1.,coarse_seconds_per_slab=.05,
            fine_seconds_per_slab=.05,refine_pairs=0,device='cpu',alignment_seconds=0,**extra)

    def test_disabled_mode_never_calls_second_pass(self):
        with patch('rockcut.quantity_pass.improve_candidate',side_effect=AssertionError('optional pass invoked')):
            result=solve_block(self.inputs,self.config())
        self.assertNotIn('quantity_pass',result)
        self.assertNotIn('search_stages',result)
        self.assertTrue(verify_block_result(result)['accepted'])

    def test_enabled_pass_uses_same_absolute_deadline_and_preserves_plan(self):
        snapshot={}
        def pass_stub(block,candidate,side,config,deadline,emit):
            from rockcut import block_solver
            snapshot.update(plan=copy.deepcopy(candidate['plan'].to_dict()),
                states=copy.deepcopy(candidate['records']), deadline=deadline,now=block_solver.time.monotonic())
            return {'before_n':sum(r['n'] for r in candidate['records']),
                'after_n':sum(r['n'] for r in candidate['records']),'stop_reason':'queue_completed'}
        with patch('rockcut.quantity_pass.improve_candidate',side_effect=pass_stub):
            result=solve_block(self.inputs,self.config(quantity_pass='retry',quantity_reserve_seconds=.2))
        self.assertEqual(result['plan'],snapshot['plan'])
        self.assertEqual(result['inputs'],self.inputs)
        self.assertEqual([s['n'] for s in result['slabs']],[s['n'] for s in snapshot['states']])
        self.assertLessEqual(snapshot['deadline']-snapshot['now'],.25)
        self.assertEqual(result['search_stages']['ordinary_budget_seconds'],.8)
        self.assertEqual(result['search_stages']['total_budget_seconds'],1.)
        self.assertEqual(result['quantity_pass']['requested_reserve_seconds'],.2)
        self.assertTrue(verify_block_result(result)['accepted'])

    def test_invalid_reserve_cannot_add_hidden_time(self):
        for params in ({'quantity_pass':'retry'}, {'quantity_pass':'off','quantity_reserve_seconds':1},
                       {'quantity_pass':'retry','quantity_reserve_seconds':360,'seconds':360},
                       {'quantity_pass':'retry','quantity_reserve_seconds':-1},
                       {'quantity_pass':'long','quantity_reserve_seconds':float('nan')}):
            with self.subTest(params=params),self.assertRaises(ValueError):
                BlockSearchConfig(grid_seed='off', **params)


if __name__=='__main__':
    unittest.main()
