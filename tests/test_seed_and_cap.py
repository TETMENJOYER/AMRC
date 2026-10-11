import unittest
from unittest.mock import patch
import numpy as np
import manifold3d as m3d

from rockcut.block import Block, Frame, mesh_arrays
from rockcut.plane_clip import clip_z
from rockcut.cut_plan import parallel_plan
from rockcut.problem import Problem
from rockcut.verify import validate, check_candidate, verified_subset
from rockcut.packing3d import verify_layout
from rockcut.solver import SearchConfig, solve
from rockcut.block_solver import BlockSearchConfig, solve_block


class CapTests(unittest.TestCase):
    def test_cut_cap_is_exact_and_surviving_source_vertices_are_unchanged(self):
        source=m3d.Manifold.cube((8,9,10)).rotate((11,27,13)).translate((123.456,789.012,345.678))
        offset=347.0155409298938
        raw=source.trim_by_plane((0,0,1),offset)
        before,_=mesh_arrays(source)
        clipped,note=clip_z(source,offset,True)
        vertices,faces=mesh_arrays(clipped)
        self.assertGreater(note['corrected_cap_vertices'],0)
        self.assertLess(note['maximum_roundoff_correction_mm'],1e-11)
        self.assertGreaterEqual(vertices[:,2].min(),offset)
        cap=faces[np.all(np.abs(vertices[faces][:,:,2]-offset)<1e-11,axis=1)]
        self.assertGreater(len(cap),0)
        self.assertTrue(np.all(vertices[np.unique(cap),2]==offset))
        kept=set(map(tuple,vertices))
        for vertex in before[before[:,2]>offset]:
            self.assertIn(tuple(vertex),kept)
        self.assertAlmostEqual(clipped.volume(),raw.volume(),places=9)

    def test_existing_surface_is_never_snapped_to_a_nearby_cut(self):
        source=m3d.Manifold.cube((3,2,7))
        below_top=float(np.nextafter(7.,0.))
        raw=source.trim_by_plane((0,0,-1),-below_top)
        clipped,note=clip_z(source,below_top,False)
        self.assertEqual(note['corrected_cap_vertices'],0)
        np.testing.assert_array_equal(mesh_arrays(clipped)[0],mesh_arrays(raw)[0])

    def test_sequential_rotated_cuts_preserve_volume_and_cap_planes(self):
        block=Block.from_dict({'schema_version':1,'units':'mm','base':{'type':'box','size':[8,9,10],
            'center':[123.456,789.012,345.678],'rotation_degrees':[11,27,13]}})
        plan=parallel_plan(block,[0,0,1],2)
        self.assertAlmostEqual(sum(p.volume() for p in plan.pieces)+plan.remainder.volume(),720,places=8)
        for piece,spec in zip(plan.pieces,plan.specs):
            vertices,_=mesh_arrays(piece)
            self.assertGreaterEqual(vertices[:,2].min(),spec.lower)
            self.assertLessEqual(vertices[:,2].max(),spec.upper)


class SeedSubsetTests(unittest.TestCase):
    def setUp(self):
        self.problem=Problem([[0,0],[12,0],[12,2],[0,2]],1)
        self.state=np.array([[1.5,.5,0],[4.5,.5,0],[7.5,.5,0]])
        self.block=Block.from_dict({'schema_version':1,'units':'mm','base':{'type':'box','size':[12,2,7]}})
        self.plan=parallel_plan(self.block,[0,0,1],3)

    def test_retains_valid_products_after_a_boundary_failure(self):
        bad=np.vstack((self.state,[20,.5,0]))
        kept,check=verified_subset(bad,lambda q:check_candidate(self.problem,q))
        np.testing.assert_array_equal(kept,self.state)
        self.assertTrue(check['accepted'])

    def test_removes_a_conflicting_product_and_rechecks_whole_subset(self):
        bad=np.vstack((self.state,self.state[1]))
        kept,check=verified_subset(bad,lambda q:check_candidate(self.problem,q))
        np.testing.assert_array_equal(kept,self.state)
        self.assertTrue(validate(self.problem,kept)['accepted'])

    def test_numeric_product_failure_keeps_other_3d_verified_products(self):
        from rockcut.packing3d import validate_product
        def product_check(block,spec,placement,side):
            if placement[0]==4.5:
                raise ValueError('invalid numerical CSG residual')
            return validate_product(block,spec,placement,side)
        seen=[]
        def accept(q):
            seen.append(len(q))
            return verify_layout(self.block,self.plan.specs[0],1,q,slab_local=self.plan.pieces[0])
        with patch('rockcut.packing3d.validate_product',side_effect=product_check):
            kept,check=verified_subset(self.state,lambda q:check_candidate(self.problem,q,accept))
        np.testing.assert_array_equal(kept,self.state[[0,2]])
        self.assertEqual(seen,[3,2])
        self.assertTrue(check['additional_check']['accepted'])

    def test_wrong_assigned_slab_keeps_no_products(self):
        accept=lambda q:verify_layout(self.block,self.plan.specs[0],1,q,slab_local=self.plan.pieces[1])
        kept,check=verified_subset(self.state,lambda q:check_candidate(self.problem,q,accept))
        self.assertEqual(len(kept),0)
        self.assertTrue(check['accepted'])

    def test_product_outside_actual_source_is_removed_despite_a_covering_2d_region(self):
        smaller=Block.from_dict({'schema_version':1,'units':'mm','base':{'type':'box','size':[6,2,7]}})
        plan=parallel_plan(smaller,[0,0,1],3)
        accept=lambda q:verify_layout(smaller,plan.specs[0],1,q,slab_local=plan.pieces[0])
        self.assertTrue(validate(self.problem,self.state)['accepted'])
        kept,check=verified_subset(self.state,lambda q:check_candidate(self.problem,q,accept))
        np.testing.assert_array_equal(kept,self.state[:2])
        self.assertTrue(check['additional_check']['accepted'])

    def test_unexplained_or_inconsistent_rejection_is_not_salvaged(self):
        for diagnostic in ({'accepted':False},
                           {'accepted':False,'products':[{'i':0,'accepted':True}]}):
            accept=lambda q,d=diagnostic:d if len(q) else {'accepted':True}
            kept,check=verified_subset(self.state,lambda q:check_candidate(self.problem,q,accept))
            self.assertEqual(len(kept),0)
            self.assertTrue(check['accepted'])

    def test_false_product_flags_do_not_bypass_full_reverification(self):
        calls=[]
        def accept(q):
            calls.append(len(q))
            return {'accepted':len(q)==0,'products':[{'i':i,'accepted':i!=1} for i in range(len(q))]}
        kept,check=verified_subset(self.state,lambda q:check_candidate(self.problem,q,accept))
        self.assertEqual(len(kept),0)
        self.assertTrue(check['accepted'])
        self.assertIn(2,calls)

    def test_solver_retains_checked_subset_as_warm_start(self):
        accept=lambda q:verify_layout(self.block,self.plan.specs[0],1,q,slab_local=self.plan.pieces[0])
        state=np.vstack((self.state,[20,.5,0]))
        result=solve(self.problem,SearchConfig(seconds=.01),initial_state=state,acceptance_check=accept)
        self.assertGreaterEqual(result['n'],3)
        self.assertTrue(result['validation']['additional_check']['accepted'])
        self.assertTrue(any(e.get('event')=='seed_subset' and e['retained_n']==3 for e in result['history']))

    def test_coarse_block_stage_retains_valid_part_of_a_rejected_seed(self):
        source={'schema_version':1,'kind':'block_cutting_problem',
                'block':{'schema_version':1,'units':'mm','base':{'type':'box','size':[6,2,7]}},
                'product':{'short_side':1,'thickness':3},'directions':[[0,0,1]]}
        seed=np.array([[1.5,.5,0],[4.5,.5,0],[1.5,1.5,0],[100,100,0]])
        with patch('rockcut.block_solver.shelf_seeds',side_effect=lambda p,**kwargs:[p.normalize(seed)]), \
             patch('rockcut.block_solver.fill_gaps',side_effect=lambda p,q,deadline:q):
            result=solve_block(source,BlockSearchConfig(seconds=1,refine_pairs=0,
                coarse_seconds_per_slab=.1,fine_seconds_per_slab=.01))
        self.assertTrue(result['validation']['accepted'])
        self.assertGreaterEqual(result['n_total'],6)
        for slab in result['slabs']:
            self.assertEqual(slab['coarse_seed'],{'original_n':4,'retained_n':3})


if __name__=='__main__':
    unittest.main()
