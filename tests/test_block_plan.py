import copy
import json
import unittest
import contextlib
import io
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch

from rockcut.block import Block,Frame
from rockcut.cut_plan import parallel_plan,component_volumes
from rockcut.block_solver import BlockSearchConfig,solve_block,directions
from rockcut.verify_block import verify_block_result,block_inputs
from rockcut.cli import main as cli_main
from rockcut.problem import Problem
from rockcut.solver import SearchConfig,solve
from rockcut.block_generator import demo_experiment
from rockcut.slab3d import build_slab
from rockcut.packing3d import verify_layout


def cube_input(height=7):
    return {'schema_version':1,'kind':'block_cutting_problem',
            'block':{'schema_version':1,'units':'mm','name':'analytic_block',
                     'base':{'type':'box','size':[6,2,height]}},
            'product':{'short_side':1,'aspect_ratio':3,'thickness':3,'kerf':0,'allowance':0},
            'cut_policy':{'kind':'parallel_support_start','initial_trim_allowed':False,'kerf':0},
            'directions':[[0,0,1]]}


class BlockPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.result=solve_block(cube_input(),BlockSearchConfig(seconds=4,refine_pairs=0,
                              coarse_seconds_per_slab=.2,fine_seconds_per_slab=.2))

    def test_sequence_has_nominal_thickness_and_volume_balance(self):
        block,_,t=block_inputs(cube_input())
        plan=parallel_plan(block,[0,0,1],t)
        self.assertEqual(len(plan.specs),2)
        self.assertAlmostEqual(sum(p.volume() for p in plan.pieces)+plan.remainder.volume(),84)
        self.assertAlmostEqual(plan.remainder.volume(),12)
        self.assertEqual([s.thickness for s in plan.specs],[3,3])
        self.assertAlmostEqual(plan.steps[0]['after_volume_mm3'],48)
        self.assertAlmostEqual(plan.steps[1]['before_volume_mm3'],48)
        self.assertTrue(all(s['cut_performed'] for s in plan.steps))

    def test_reverse_start_keeps_opposite_edge_remainder(self):
        block,_,t=block_inputs(cube_input())
        plan=parallel_plan(block,[0,0,-1],t)
        world=plan.frame.world_solid(plan.remainder)
        bounds=world.bounding_box()
        self.assertAlmostEqual(bounds[2],0)
        self.assertAlmostEqual(bounds[5],1)
        self.assertAlmostEqual(world.volume(),12)

    def test_final_complete_piece_requires_no_additional_cut(self):
        block,_,t=block_inputs(cube_input(6))
        plan=parallel_plan(block,[0,0,1],t)
        self.assertEqual(plan.to_dict()['cut_count'],1)
        self.assertEqual(plan.steps[-1]['kind'],'take_final_piece')
        self.assertTrue(plan.remainder.is_empty())

    def test_split_fragments_are_all_preserved(self):
        source=cube_input();source['block']['base']['size']=[10,10,10]
        source['block']['subtract']=[{'type':'box','center':[5,5,9],'size':[2,12,8]}]
        block,_,_=block_inputs(source)
        plan=parallel_plan(block,[0,0,1],2)
        self.assertEqual(len(component_volumes(plan.pieces[3])),2)
        self.assertEqual(len(plan.steps[2]['remaining_component_volumes_mm3']),2)
        self.assertAlmostEqual(sum(p.volume() for p in plan.pieces),block.solid.volume())

    def test_whole_pipeline_known_count_and_volume_yield(self):
        r=self.result
        self.assertEqual(r['n_total'],8)
        self.assertEqual([s['n'] for s in r['slabs']],[4,4])
        self.assertAlmostEqual(r['yield_by_volume'],72/84)
        self.assertTrue(r['validation']['accepted'])
        self.assertLess(r['volume_balance']['error_mm3'],1e-10)
        self.assertFalse(r['optimality_proven'])

    def test_slab_with_empty_full_thickness_region_is_accounted(self):
        source=cube_input()
        source['product']['thickness']=8
        source['block']['base']={'type':'prism','vertices':[[0,0],[10,0],[10,10],[0,10]],
                                 'height':10,'top_shift':[30,0]}
        r=solve_block(source,BlockSearchConfig(seconds=2,refine_pairs=0))
        self.assertEqual(r['n_total'],0)
        self.assertEqual(len(r['slabs']),1)
        self.assertTrue(r['validation']['accepted'])
        self.assertAlmostEqual(r['volume_balance']['unused_in_slabs_mm3'],800)
        self.assertAlmostEqual(r['volume_balance']['final_remainder_mm3'],200)

    def test_thinner_than_one_slab_returns_whole_block_as_remainder(self):
        r=solve_block(cube_input(2),BlockSearchConfig(seconds=1,refine_pairs=0))
        self.assertEqual(r['n_total'],0)
        self.assertEqual(len(r['slabs']),0)
        self.assertAlmostEqual(r['volume_balance']['final_remainder_mm3'],24)
        self.assertTrue(r['validation']['accepted'])

    def test_direction_selected_by_total_products_not_one_section(self):
        source=cube_input();source['directions']=[[0,1,0],[0,0,1]]
        r=solve_block(source,BlockSearchConfig(seconds=4,refine_pairs=0,coarse_seconds_per_slab=.2))
        self.assertEqual(r['n_total'],8)
        np.testing.assert_allclose(r['plan']['normal'],[0,0,1])

    def test_tiny_budget_retains_complete_plan_and_balance(self):
        r=solve_block(cube_input(),BlockSearchConfig(seconds=.00001,refine_pairs=0))
        self.assertEqual(len(r['slabs']),2)
        self.assertTrue(r['validation']['accepted'])
        self.assertLessEqual(r['n_total'],8)
        self.assertAlmostEqual(r['volume_balance']['unallocated_volume_mm3'],84-r['n_total']*9)

    def test_audit_ignores_saved_polygons_world_vertices_and_flags(self):
        r=copy.deepcopy(self.result)
        r['validation']={'accepted':False,'slabs':[]}
        for s in r['slabs']:
            s['problem']=None;s['geometry']={};s['world_vertices']=[[999,999,999]]
        self.assertTrue(verify_block_result(r)['accepted'])

    def test_audit_rejects_intersecting_products(self):
        r=copy.deepcopy(self.result)
        r['slabs'][0]['state'][1]=r['slabs'][0]['state'][0].copy()
        self.assertFalse(verify_block_result(r)['accepted'])

    def test_audit_rejects_changed_cut_count_or_balance(self):
        for edit in ('count','cut','remainder','yield'):
            r=copy.deepcopy(self.result)
            if edit=='count':r['n_total']+=1
            if edit=='cut':r['plan']['cuts'][0]['plane_offset_mm']+=.1
            if edit=='remainder':r['volume_balance']['final_remainder_mm3']+=1
            if edit=='yield':r['yield_by_volume']=1
            self.assertFalse(verify_block_result(r)['accepted'])

    def test_audit_rejects_reassigned_slab_or_missing_record(self):
        r=copy.deepcopy(self.result)
        r['slabs'][1]['spec']=r['slabs'][0]['spec']
        with self.assertRaises(ValueError):verify_block_result(r)
        r=copy.deepcopy(self.result);r['slabs'].pop()
        with self.assertRaises(ValueError):verify_block_result(r)

    def test_cli_rejects_tampering_even_with_saved_acceptance(self):
        r=copy.deepcopy(self.result);r['n_total']+=1;r['validation']['accepted']=True
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'result.json';path.write_text(json.dumps(r),encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(['verify',str(path)]),2)

    def test_configuration_and_unagreed_production_options_rejected(self):
        for kw in ({'seconds':0},{'direction_count':0},{'refine_pairs':-1},{'variant':'unknown'}):
            with self.assertRaises(ValueError):BlockSearchConfig(**kw)
        for field in ('kerf','allowance','thickness'):
            source=cube_input();source['product'][field]=-1
            with self.assertRaises(ValueError):block_inputs(source)
        source=cube_input();source['cut_policy']['initial_trim_allowed']=True
        with self.assertRaises(ValueError):block_inputs(source)
        block,_,_=block_inputs(cube_input())
        with self.assertRaises(ValueError):directions(block,2,1,[[0,0,0]])
        a=directions(block,12,1);b=directions(block,12,1)
        np.testing.assert_allclose(a,b)

    def test_verified_warm_start_is_retained_and_invalid_one_is_not_accepted(self):
        p=Problem([[0,0],[6,0],[6,2],[0,2]],1)
        valid=np.array([[1.5,.5,0],[4.5,.5,0],[1.5,1.5,0],[4.5,1.5,0]])
        r=solve(p,SearchConfig(seconds=.1),initial_state=valid)
        self.assertEqual(r['n'],4)
        self.assertTrue(any(e.get('source')=='verified_warm_start' for e in r['history']))
        bad=np.tile([1.5,.5,0],(4,1))
        r=solve(p,SearchConfig(seconds=.1),initial_state=bad)
        self.assertTrue(any(e.get('event')=='seed_subset' and e.get('retained_n')==1 for e in r['history']))
        self.assertTrue(r['validation']['accepted'])

    def test_reverse_irregular_block_keeps_full_geometry_after_normalization(self):
        block=Block.from_dict(demo_experiment()['block'])
        plan=parallel_plan(block,[0,0,-1],30)
        slab=build_slab(block,plan.specs[0])
        p=Problem.from_region(slab.footprint,100)
        self.assertTrue(p.polygon.is_valid)
        self.assertTrue(p.physical_region.equals(slab.footprint))
        self.assertAlmostEqual(p.area,slab.footprint.area)
        for component in p.normalization_discarded_components:
            xmin,ymin,xmax,ymax=component['bounds']
            self.assertLess(min(xmax-xmin,ymax-ymin),100)

    def test_numerically_failed_direction_does_not_destroy_a_valid_complete_plan(self):
        with patch('rockcut.block_solver.Problem.from_region',side_effect=ValueError('numerical contour failure')):
            r=solve_block(cube_input(),BlockSearchConfig(seconds=1,refine_pairs=0))
        self.assertEqual(r['n_total'],0)
        self.assertTrue(r['validation']['accepted'])
        self.assertEqual(len(r['slabs']),2)
        self.assertFalse(r['candidates'][0]['geometry_complete'])
        self.assertIsNone(r['candidates'][0]['area_upper_bound'])
        self.assertTrue(any(e['event']=='direction_geometry_failure' for e in r['history']))

    def test_numerical_csg_failure_rejects_candidate_without_claiming_zero_residual(self):
        block,side,t=block_inputs(cube_input())
        spec=parallel_plan(block,[0,0,1],t).specs[0]
        with patch('rockcut.packing3d.validate_product',side_effect=ValueError('invalid residual volume')):
            audit=verify_layout(block,spec,side,[[1.5,.5,0]])
        self.assertFalse(audit['accepted'])
        self.assertIsNone(audit['missing_volume_sum_mm3'])
        self.assertEqual(audit['products'][0]['status'],'numerical_verification_failed')

    def test_products_must_fit_the_assigned_piece_not_only_the_original_block(self):
        block,side,t=block_inputs(cube_input())
        plan=parallel_plan(block,[0,0,1],t)
        state=[[1.5,.5,0]]
        self.assertTrue(verify_layout(block,plan.specs[0],side,state,slab_local=plan.pieces[0])['accepted'])
        self.assertFalse(verify_layout(block,plan.specs[0],side,state,slab_local=plan.pieces[1])['accepted'])


if __name__=='__main__':unittest.main()
