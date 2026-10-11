import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon, box

from rockcut.block import Block, Frame, mesh_arrays
from rockcut.block_generator import demo_experiment
from rockcut.slab3d import SlabSpec, build_slab
from rockcut.geometry3d import run_experiment, recheck_result
from rockcut.verify3d import validate_product


def recipe(cutters=()):
    return {"schema_version": 1, "units": "mm", "name": "analytic_cube", "base": {"type": "box", "size": [10, 10, 10]},
            "subtract": list(cutters)}


def cutter(center, size):
    return {"type": "box", "center": center, "size": size}


def slab_spec(lo=1, hi=9):
    return SlabSpec(Frame.from_dict({}), lo, hi)


class Geometry3DTests(unittest.TestCase):
    def test_prism_known_area_volume_and_closed_mesh(self):
        block = Block.from_dict(recipe())
        slab = build_slab(block, slab_spec())
        self.assertAlmostEqual(slab.footprint.area, 100)
        self.assertLess(slab.footprint.symmetric_difference(box(0, 0, 10, 10)).area, 1e-10)
        self.assertAlmostEqual(slab.solid.volume(), 800)
        self.assertAlmostEqual(slab.diagnostics['partition_error_mm3'], 0, places=10)
        self.assertTrue(block.diagnostics()['watertight'])
        self.assertEqual(mesh_arrays(block.solid)[0].dtype, np.float64)

    def test_hidden_side_notch_has_analytic_footprint(self):
        block = Block.from_dict(recipe([cutter([9, 5, 3], [4, 4, 1])]))
        slab = build_slab(block, slab_spec())
        expected = box(0, 0, 10, 10).difference(box(7, 3, 10, 7))
        self.assertLess(slab.footprint.symmetric_difference(expected).area, 1e-10)
        self.assertAlmostEqual(slab.sampled_intersection.area, 100)
        self.assertAlmostEqual(slab.diagnostics['unsafe_sampled_area_mm2'], 12)
        check = validate_product(block, slab.spec, [8, 5, 0], 1, corner_diagnostics=True)
        self.assertFalse(check['accepted'])
        self.assertTrue(check['all_corners_strictly_inside_diagnostic'])
        self.assertAlmostEqual(check['missing_volume_mm3'], 2.5)

    def test_cavities_through_holes_and_separate_blocks_are_rejected(self):
        for cut in (cutter([5, 5, 3], [1, .5, 1]),
                    cutter([5, 5, 5], [1, 1, 12]),
                    cutter([5, 5, 3], [12, 12, 1])):
            with self.assertRaisesRegex(ValueError, 'without enclosed cavities'):
                Block.from_dict(recipe([cut]))

    def test_disconnected_footprint_not_filled_with_convex_hull(self):
        block = Block.from_dict(recipe([cutter([5, 5, 9], [2, 12, 8])]))
        slab = build_slab(block, slab_spec())
        self.assertEqual(slab.footprint.geom_type, 'MultiPolygon')
        self.assertEqual(slab.diagnostics['components'], 2)
        self.assertAlmostEqual(slab.footprint.area, 80)
        self.assertFalse(validate_product(block, slab.spec, [5, 5, 0], 1)['accepted'])

    def test_empty_common_footprint_is_normal_result(self):
        block = Block.from_dict({'schema_version': 1, 'units': 'mm',
            'base': {'type': 'prism', 'height': 10, 'top_shift': [30, 0],
                     'vertices': [[0, 0], [10, 0], [10, 10], [0, 10]]}})
        slab = build_slab(block, slab_spec())
        self.assertTrue(slab.footprint.is_empty)
        self.assertGreater(slab.solid.volume(), 0)
        self.assertFalse(validate_product(block, slab.spec, [5, 5, 0], 1)['accepted'])

    def test_absent_slab_and_coplanar_boundary_do_not_crash(self):
        block = Block.from_dict(recipe())
        for lower, upper in ((10, 11), (-2, -1)):
            slab = build_slab(block, slab_spec(lower, upper))
            self.assertTrue(slab.solid.is_empty())
            self.assertTrue(slab.footprint.is_empty)
        whole = build_slab(block, slab_spec(0, 10))
        self.assertAlmostEqual(whole.footprint.area, 100)

    def test_boundary_contact_allowed_but_small_residual_rejected(self):
        block = Block.from_dict(recipe())
        self.assertTrue(validate_product(block, slab_spec(0, 10), [1.5, .5, 0], 1)['accepted'])
        check = validate_product(block, slab_spec(0, 10), [1.5 - 1e-6, .5, 0], 1)
        self.assertFalse(check['accepted'])
        self.assertAlmostEqual(check['missing_volume_mm3'], 1e-5, places=10)

    def test_thin_defect_is_not_ignored_by_area_threshold(self):
        block = Block.from_dict(recipe([cutter([9, 5, 3], [4, 4, 1e-4])]))
        slab = build_slab(block, slab_spec())
        self.assertAlmostEqual(slab.footprint.area, 88)
        check = validate_product(block, slab.spec, [8, 5, 0], 1)
        self.assertFalse(check['accepted'])
        self.assertAlmostEqual(check['missing_volume_mm3'], .00025, places=10)

    def test_cutout_touching_slab_face_from_outside_does_not_penalize(self):
        block = Block.from_dict(recipe([cutter([5, 5, .5], [2, 2, 1])]))
        slab = build_slab(block, slab_spec(1, 9))
        self.assertAlmostEqual(slab.footprint.area, 100)
        self.assertTrue(validate_product(block, slab.spec, [5, 5, 0], 1)['accepted'])

    def test_rigid_transform_preserves_area_and_full_volume_checks(self):
        source = recipe([cutter([9, 5, 3], [4, 4, 1])])
        rotation = Rotation.from_euler('xyz', [17, -23, 31], degrees=True).as_matrix()
        frame = Frame(np.array([100., -200., 300.]), rotation)
        source['pose'] = frame.to_dict()
        block = Block.from_dict(source)
        slab = build_slab(block, SlabSpec(frame, 1, 9))
        self.assertAlmostEqual(slab.footprint.area, 88, places=7)
        check = validate_product(block, slab.spec, [8, 5, 0], 1)
        self.assertFalse(check['accepted'])
        self.assertAlmostEqual(check['missing_volume_mm3'], 2.5, places=8)
        self.assertTrue(validate_product(block, slab.spec, [3, 7, .2], 1)['accepted'])

    def test_arbitrary_normal_through_cube(self):
        block = Block.from_dict(recipe())
        frame = Frame.from_dict({'normal': [1, 0, 0]})
        slab = build_slab(block, SlabSpec(frame, 2, 7))
        self.assertAlmostEqual(slab.footprint.area, 100)
        self.assertAlmostEqual(slab.solid.volume(), 500)

    def test_volume_validator_does_not_call_footprint_builder(self):
        block = Block.from_dict(recipe())
        with patch('rockcut.slab3d.build_slab', side_effect=AssertionError('must not sample')):
            self.assertTrue(validate_product(block, slab_spec(), [5, 5, 0], 1)['accepted'])

    def test_default_validation_never_scans_corner_winding(self):
        block = Block.from_dict(recipe())
        with patch('rockcut.verify3d.winding_numbers', side_effect=AssertionError('expensive diagnostic')):
            check = validate_product(block, slab_spec(), [5, 5, 0], 1)
            outside = validate_product(block, slab_spec(), [.5, .5, 0], 1)
        self.assertTrue(check['accepted'])
        self.assertFalse(outside['accepted'])
        self.assertFalse(check['corner_diagnostics_computed'])
        self.assertIsNone(check['all_corners_strictly_inside_diagnostic'])

    def test_corner_diagnostics_do_not_change_strict_volume_decisions(self):
        block = Block.from_dict(recipe([cutter([9, 5, 3], [4, 4, 1])]))
        for state in ([3, 5, 0], [8, 5, 0], [1.5, .5, 0], [1.5-1e-6, .5, 0]):
            plain = validate_product(block, slab_spec(), state, 1)
            diagnostic = validate_product(block, slab_spec(), state, 1, corner_diagnostics=True)
            self.assertTrue(diagnostic['corner_diagnostics_computed'])
            for field in ('accepted', 'missing_volume_mm3', 'product_volume_mm3', 'method',
                          'nonempty_residual_is_rejected', 'world_vertices'):
                self.assertEqual(plain[field], diagnostic[field])

    def test_invalid_inputs_are_rejected(self):
        for lo, hi in ((1, 1), (2, 1), (0, float('nan'))):
            with self.assertRaises(ValueError):
                slab_spec(lo, hi)
        with self.assertRaises(ValueError):
            Frame.from_dict({'normal': [0, 0, 0]})
        with self.assertRaises(ValueError):
            Frame([0, 0, 0], np.diag([1, 1, -1]))
        bad = recipe(); bad['base']['size'] = [-1, 10, 10]
        with self.assertRaises(ValueError):
            Block.from_dict(bad)
        for state, side in (([float('nan'), 0, 0], 1), ([1, 1, 0], 0), ([1, 1], 1)):
            with self.assertRaises(ValueError):
                validate_product(Block.from_dict(recipe()), slab_spec(), state, side)

    def test_demo_is_reproducible_nonconvex_and_homogeneous(self):
        source = demo_experiment()
        self.assertEqual(source, demo_experiment())
        self.assertNotEqual(source, demo_experiment(123))
        result, _, _ = run_experiment(source)
        self.assertTrue(result['all_expectations_met'])
        self.assertTrue(result['all_footprint_checks_agree'])
        self.assertEqual(sum(t['accepted'] for t in result['trial_results']), 4)
        self.assertNotIn('subtract', source['block'])
        polygon = Polygon(source['block']['base']['vertices'])
        self.assertFalse(polygon.equals(polygon.convex_hull))
        self.assertEqual(result['block_check']['boundary_shells'], 1)
        self.assertEqual(result['block_check']['euler_number'], 2)
        self.assertEqual(result['slab_geometry']['holes'], 0)
        self.assertFalse(result['trial_results'][3]['accepted'])

    def test_reverification_ignores_saved_geometry_and_flags(self):
        result, _, _ = run_experiment(demo_experiment())
        restored = json.loads(json.dumps(result))
        restored['effective_region'] = {'type': 'Feature', 'geometry': None}
        for trial in restored['trial_results']:
            trial['accepted'] = True
        audit = recheck_result(restored)
        self.assertTrue(audit['all_expectations_met'])
        self.assertFalse(audit['trials'][3]['accepted'])
        restored['inputs']['trials'][3]['state'] = [300, 250, 0]
        self.assertFalse(recheck_result(restored)['all_expectations_met'])


if __name__ == '__main__':
    unittest.main()
