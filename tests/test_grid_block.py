import unittest
from unittest.mock import patch
import numpy as np
from rockcut.block_solver import BlockSearchConfig, solve_block
from rockcut.quantity_pass import QuantityPassConfig
from rockcut.verify_block import verify_block_result


class GridBlockIntegrationTests(unittest.TestCase):
    def test_default_coarse_grid_is_checked_and_passed_to_inner_search(self):
        inputs={'schema_version':1,'kind':'block_cutting_problem',
            'block':{'schema_version':1,'units':'mm','base':{'type':'box','size':[6,2,7]}},
            'product':{'short_side':1,'aspect_ratio':3,'thickness':3,'kerf':0,'allowance':0},
            'directions':[[0,0,1]]}
        state=np.array([[1.5,.5,0],[4.5,.5,0],[1.5,1.5,0]])
        calls=[]
        def search(problem,cfg,initial_state,acceptance_check):
            calls.append(cfg.grid_seed)
            check=acceptance_check(initial_state)
            self.assertTrue(check['accepted'])
            return {'n':len(initial_state),'state':initial_state.tolist(),
                'elapsed_seconds':0.,'attempts':[],'stop_reason':'test',
                'grid_initialization':{},'best_retention':{}}
        with patch('rockcut.block_solver.rigid_grid_candidates',return_value=iter([(state,{})])), \
             patch('rockcut.block_solver.shelf_seeds',return_value=[]), \
             patch('rockcut.block_solver.solve',side_effect=search):
            result=solve_block(inputs,BlockSearchConfig(seconds=2,coarse_seconds_per_slab=.1,
                fine_seconds_per_slab=.1,refine_pairs=0,device='cpu'))
        self.assertEqual(result['n_total'],3)
        self.assertTrue(verify_block_result(result)['accepted'])
        self.assertTrue(calls)
        self.assertEqual(set(calls),{'rigid'})
        self.assertTrue(any('coarse_grid' in s for s in result['slabs']))

    def test_quantity_pass_forwards_default_and_off_without_changing_combined(self):
        for options, expected in (({}, 'rigid'), ({'grid_seed':'off'}, 'off')):
            with self.subTest(options=options):
                cfg=QuantityPassConfig(grid_seconds=.3,**options).search_config(2.,101)
                self.assertEqual(cfg.grid_seed,expected)
                self.assertEqual(cfg.grid_seconds,.3)
                self.assertTrue(cfg.diverse_starts and cfg.group_repair and cfg.adaptive_penalties)

    def test_grid_configuration_rejects_invalid_policy_and_quota(self):
        for constructor in (BlockSearchConfig,QuantityPassConfig):
            for options in ({'grid_seed':'unexpected'},{'grid_seconds':0.},{'grid_seconds':float('nan')}):
                with self.subTest(constructor=constructor,options=options),self.assertRaises(ValueError):
                    constructor(**options)
