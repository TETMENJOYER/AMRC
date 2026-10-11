import copy
import unittest

import numpy as np
import trimesh

from rockcut.block import Block
from rockcut.mesh_import import close_lower_plane, mesh_payload, decode_mesh
from rockcut.cut_plan import parallel_plan
from rockcut.slab3d import build_slab, SlabSpec
from rockcut.block import Frame


def open_box():
    mesh = trimesh.creation.box(bounds=[[0, 0, 0], [6, 4, 7]])
    mesh.update_faces(~np.all(mesh.vertices[mesh.faces, 2] == 0, axis=1))
    mesh.remove_unreferenced_vertices()
    return mesh


class MeshImportTests(unittest.TestCase):
    def test_closure_has_analytic_volume_and_planar_bottom(self):
        mesh, note = close_lower_plane(open_box(), axis=2, clearance=1)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertEqual(mesh.euler_number, 2)
        self.assertAlmostEqual(mesh.volume, 6*4*6)
        self.assertTrue(np.all(mesh.vertices[np.unique(mesh.faces[-note['cap_triangles']:]), 2] == 1))

    def test_retains_concavity(self):
        import manifold3d as m
        from rockcut.block import mesh_arrays
        solid = m.CrossSection([[[0,0],[6,0],[6,2],[2,2],[2,6],[0,6]]]).extrude(7)
        v, f = mesh_arrays(solid)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        mesh.update_faces(~np.all(mesh.vertices[mesh.faces, 2] == 0, axis=1))
        closed, _ = close_lower_plane(mesh, axis=2, clearance=1)
        self.assertAlmostEqual(closed.volume, 20*6)
        self.assertLess(closed.volume, closed.convex_hull.volume)

    def test_source_vertices_are_not_moved(self):
        source = open_box(); before = source.vertices.copy()
        mesh, _ = close_lower_plane(source, axis=2, clearance=1)
        np.testing.assert_array_equal(source.vertices, before)
        for point in before[before[:, 2] > 1]:
            self.assertTrue(np.any(np.all(mesh.vertices == point, axis=1)))

    def test_closure_rejects_boundary_above_model(self):
        with self.assertRaises(ValueError):
            close_lower_plane(open_box(), axis=2, clearance=8)

    def test_closure_rejects_closed_source(self):
        with self.assertRaises(ValueError):
            close_lower_plane(trimesh.creation.box(), axis=2)

    def test_embedded_mesh_roundtrip_and_source_slab(self):
        mesh, _ = close_lower_plane(open_box(), axis=2, clearance=1)
        block = Block.from_dict({'schema_version':1,'units':'mm','base':mesh_payload(mesh)})
        self.assertAlmostEqual(block.solid.volume(), 144)
        plan = parallel_plan(block, [0,0,1], 3)
        self.assertEqual(len(plan.specs),2)
        self.assertAlmostEqual(sum(p.volume() for p in plan.pieces),144)
        slab = build_slab(block, SlabSpec(Frame.from_dict({'normal':[0,0,1]}),1,4))
        self.assertAlmostEqual(slab.footprint.area,24)

    def test_rejects_checksum_tampering(self):
        base=mesh_payload(trimesh.creation.box()); base['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'checksum'):
            decode_mesh(base)

    def test_rejects_open_mesh(self):
        with self.assertRaises(ValueError):
            decode_mesh(mesh_payload(open_box()))

    def test_rejects_fractional_or_negative_or_large_indices(self):
        mesh=trimesh.creation.box()
        for value in (.5,-1,9999):
            faces=mesh.faces.astype(float if value==.5 else int)
            faces[0,0]=value
            with self.assertRaises(ValueError):
                decode_mesh({'vertices':mesh.vertices.tolist(),'faces':faces.tolist()})

    def test_rejects_nonfinite_vertices_and_inverted_surface(self):
        mesh=trimesh.creation.box()
        data={'vertices':mesh.vertices.tolist(),'faces':mesh.faces.tolist()}
        bad=copy.deepcopy(data);bad['vertices'][0][0]=float('nan')
        with self.assertRaises(ValueError): decode_mesh(bad)
        bad=copy.deepcopy(data);bad['faces']=np.asarray(bad['faces'])[:,::-1].tolist()
        with self.assertRaises(ValueError): decode_mesh(bad)

    def test_rejects_enclosed_cavity_and_through_tunnel(self):
        import manifold3d as m
        from rockcut.block import mesh_arrays
        cube=m.Manifold.cube([10,10,10])
        for cutter in (m.Manifold.cube([2,2,2]).translate([4,4,4]),
                       m.Manifold.cube([2,2,12]).translate([4,4,-1])):
            v,f=mesh_arrays(cube-cutter)
            with self.assertRaises(ValueError):
                decode_mesh({'vertices':v.tolist(),'faces':f.tolist()})

    def test_full_cutting_and_repeated_audit_use_embedded_source(self):
        from rockcut.block_solver import solve_block, BlockSearchConfig
        from rockcut.verify_block import verify_block_result
        mesh,_=close_lower_plane(open_box(),axis=2,clearance=1)
        source={'schema_version':1,'kind':'block_cutting_problem',
                'block':{'schema_version':1,'units':'mm','base':mesh_payload(mesh)},
                'product':{'short_side':1,'thickness':3,'aspect_ratio':3},
                'directions':[[0,0,1]]}
        result=solve_block(source,BlockSearchConfig(seconds=2,refine_pairs=0,
                            coarse_seconds_per_slab=.3,fine_seconds_per_slab=.3))
        self.assertEqual(result['n_total'],16)
        self.assertAlmostEqual(result['yield_by_volume'],1)
        self.assertTrue(verify_block_result(result)['accepted'])
