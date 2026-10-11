import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import shapely
import torch
from shapely.geometry import Polygon, MultiPolygon, box, mapping

from rockcut.problem import Problem
from rockcut.geometry import area_score
from rockcut.local import LocalOptimizer
from rockcut.seeds import shelf_seeds, random_layout, center_region
from rockcut.verify import validate
from rockcut.solver import SearchConfig, solve
from rockcut.packing3d import solve_slab, verify_slab_result
from rockcut.block_generator import demo_experiment
from rockcut.cli import main as cli_main


def dent():
    return Polygon([[0,0],[8,0],[8,5],[5,5],[5,4.2],[4,4.2],[4,5],[0,5]])


def l_shape():
    return Polygon([[0,0],[6,0],[6,2],[3,2],[3,4],[0,4]])


class NonconvexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_vertex_input_accepts_concavity_without_hull_substitution(self):
        p = Problem(np.asarray(l_shape().exterior.coords)[:-1], 1)
        self.assertFalse(p.is_convex)
        self.assertAlmostEqual(p.area, 18)
        self.assertEqual(p.area_upper_bound, 6)
        q = np.array([[4.5,3.5,0]])
        self.assertEqual(validate(p,q)['outside_area_sum'],3)

    def test_all_four_corners_inside_do_not_hide_external_indentation(self):
        p = Problem.from_region(dent(), 1)
        state = np.array([[4.5,4.25,0]])
        check = validate(p,state)
        self.assertTrue(all(p.physical_region.covers(shapely.Point(v)) for v in check['rectangle_vertices'][0]))
        self.assertFalse(check['accepted'])
        self.assertAlmostEqual(check['outside_area_sum'], .55)
        penalty = LocalOptimizer(p,margin=0).penalty(torch.tensor(p.normalize(state)[None])).item()
        self.assertGreater(penalty,0)

    def test_excluded_2d_interior_and_components_roundtrip(self):
        # Pure 2D domains: excluded regions are not cavities in a generated block.
        for region in (box(0,0,8,5).difference(box(4,2,4.2,2.2)),
                       MultiPolygon([box(0,0,3,2),box(6,0,9,2)])):
            p = Problem.from_region(region,1)
            restored = Problem.from_dict(json.loads(json.dumps(p.to_dict())))
            self.assertTrue(restored.physical_region.equals(region))
            self.assertAlmostEqual(restored.area,region.area)
        p = Problem.from_region(box(0,0,8,5).difference(box(4,2,4.2,2.2)),1)
        state = np.array([[4,2,0]])
        self.assertFalse(validate(p,state)['accepted'])
        self.assertAlmostEqual(area_score(p,p.normalize(state)).outside,.04)
        self.assertGreater(LocalOptimizer(p,margin=0).penalty(torch.tensor(p.normalize(state)[None])).item(),0)

    def test_forbidden_triangles_cover_complement_and_sat_matches_area(self):
        rng = np.random.default_rng(91)
        for region in (dent(),l_shape(),box(0,0,8,5).difference(box(4,2,4.2,2.2)),
                       MultiPolygon([box(0,0,3,2),box(6,0,9,2)])):
            p = Problem.from_region(region,1)
            triangles = shapely.polygons(p.obstacle_triangles)
            complement = p.polygon.convex_hull.difference(p.polygon)
            self.assertLess(shapely.union_all(triangles).symmetric_difference(complement).area,1e-10)
            batch = rng.uniform([-6,-6,0],[6,6,np.pi],size=(60,2,3))
            loss = LocalOptimizer(p,margin=0).penalty(torch.tensor(batch)).detach().numpy()
            for q, value in zip(batch,loss):
                self.assertEqual(value>1e-10,area_score(p,q).total>1e-10)

    def test_boundary_gradient_matches_finite_difference(self):
        p = Problem.from_region(dent(),1)
        q = torch.tensor(p.normalize([[5.5,4.7,.23]])[None],dtype=torch.float64,requires_grad=True)
        local = LocalOptimizer(p)
        for smooth in (0,.02):
            self.assertTrue(torch.autograd.gradcheck(lambda z: local.penalty(z,smooth=smooth),(q,),atol=1e-5))

    def test_local_correction_escapes_external_indentation(self):
        p = Problem.from_region(dent(),1)
        q = p.normalize([[4.5,4.25,.1]])
        local = LocalOptimizer(p)
        fixed = local.polish(local.relax(q[None],100)[0],100)
        self.assertTrue(validate(p,p.denormalize(fixed))['accepted'])

    def test_random_centers_respect_nonconvex_holes_and_all_components(self):
        for region in (dent(),box(0,0,8,5).difference(box(4,2,6,4)),
                       MultiPolygon([box(0,0,3,2),box(6,0,9,2)])):
            p = Problem.from_region(region,1)
            a = random_layout(p,500,np.random.default_rng(7))
            b = random_layout(p,500,np.random.default_rng(7))
            np.testing.assert_array_equal(a,b)
            self.assertTrue(shapely.contains_xy(p.polygon,a[:,0],a[:,1]).all())
            if region.geom_type=='MultiPolygon':
                self.assertTrue((p.denormalize(a)[:,0]>6).any())
                self.assertTrue((p.denormalize(a)[:,0]<3).any())

    def test_fixed_angle_center_domain_agrees_with_full_rectangle(self):
        p = Problem.from_region(dent(),1)
        angle=.37
        domain=center_region(p,angle)
        rng=np.random.default_rng(18)
        xy=rng.uniform([-5,-4],[5,4],size=(150,2))
        rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        q=np.column_stack((xy @ rot.T,np.full(len(xy),angle)))
        from rockcut.geometry import corners
        np.testing.assert_array_equal(shapely.contains_xy(domain,xy[:,0],xy[:,1]),
                                      shapely.covers(p.polygon,shapely.polygons(corners(q))))

    def test_known_nonconvex_and_disconnected_tilings(self):
        for region,count in ((l_shape(),6),(MultiPolygon([box(0,0,3,2),box(6,0,9,2)]),4)):
            p=Problem.from_region(region,1)
            seeds=shelf_seeds(p,limit=2,margin=0)
            self.assertTrue(any(len(q)==count and validate(p,p.denormalize(q))['accepted'] for q in seeds))
            result=solve(p,SearchConfig(seconds=4,seconds_per_n=1,batch_size=2,adam_steps=10))
            self.assertEqual(result['n'],count)
            self.assertTrue(result['validation']['accepted'])
            self.assertFalse(result['optimality_proven'])

    def test_invalid_regions_and_ambiguous_input_are_rejected(self):
        for region in (Polygon(),Polygon([[0,0],[2,2],[0,2],[2,0]]),shapely.LineString([[0,0],[1,1]])):
            with self.assertRaises(ValueError):
                Problem.from_region(region,1)
        data=Problem.from_region(l_shape(),1).to_dict()
        with self.assertRaises(ValueError):
            Problem.from_dict({**data,'vertices':[[0,0],[1,0],[0,1]]})

    def test_slab_pipeline_and_independent_verification_ignore_saved_region(self):
        result=solve_slab(demo_experiment(),SearchConfig(seconds=4,seconds_per_n=1,batch_size=2,adam_steps=10),target_n=2)
        self.assertEqual(result['n'],2)
        self.assertTrue(result['validation3d']['accepted'])
        result['problem']['geometry']=mapping(box(-1e6,-1e6,1e6,1e6))
        self.assertTrue(verify_slab_result(result)['accepted'])
        result['state'][0]=[975,450,0]
        result['validation3d']['accepted']=True
        self.assertFalse(verify_slab_result(result)['accepted'])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'result.json'
            path.write_text(json.dumps(result),encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(['verify',str(path)]),2)

    def test_slab_verification_checks_pair_interiors_and_declared_count(self):
        result=solve_slab(demo_experiment(),SearchConfig(seconds=4,batch_size=2,adam_steps=10),target_n=2)
        result['state']=[[300,250,0],[300,250,0]]
        audit=verify_slab_result(result)
        self.assertFalse(audit['accepted'])
        self.assertEqual(len(audit['conflicting_pairs']),1)
        result['n']=3
        with self.assertRaises(ValueError):
            verify_slab_result(result)

    def test_additional_rejection_does_not_become_a_feasible_incumbent(self):
        p=Problem.from_region(l_shape(),1)
        result=solve(p,SearchConfig(seconds=1,seconds_per_n=.2,batch_size=2,
                     adam_steps=10,polish_iterations=10,iterations_per_n=1),target_n=1,
                     acceptance_check=lambda q: {'accepted':len(q)==0})
        self.assertEqual(result['n'],0)
        self.assertTrue(result['validation']['accepted'])
        self.assertNotEqual(result['stop_reason'],'target_reached')


if __name__=='__main__':
    unittest.main()
