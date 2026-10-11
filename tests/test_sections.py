import unittest
import numpy as np
from shapely.geometry import Polygon

from rockcut.problem import Problem
from rockcut.sections import Section, SlabSections, validate_sections, verify_result


def box(x0, y0, x1, y1):
    return Problem([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 1)


class SectionTests(unittest.TestCase):
    def model(self):
        return SlabSections([Section(0, box(0, 0, 10, 10)),
                             Section(1, box(2, 0, 12, 10)),
                             Section(2, box(0, 2, 10, 12))])

    def test_intersection_preserves_registration(self):
        model = self.model()
        common = model.effective_problem()
        self.assertEqual(common.area, 64)
        self.assertEqual(Polygon(common.vertices).bounds, (2, 2, 10, 10))
        for section in model.sections:
            self.assertTrue(Polygon(section.problem.vertices).covers(Polygon(common.vertices)))

    def test_not_minimum_area_section_or_union(self):
        model = self.model()
        # All three sections have area 100, but none is safe on its own.
        state = np.array([[1.5, .5, 0]])
        check = validate_sections(model, state)
        self.assertTrue(check['sections'][0]['accepted'])
        self.assertFalse(check['accepted'])
        self.assertEqual(model.effective_problem().area, 64)

    def test_feasible_in_every_section(self):
        check = validate_sections(self.model(), np.array([[5, 5, .2]]))
        self.assertTrue(check['accepted'])
        self.assertEqual(len(check['sections']), 3)
        self.assertFalse(check['continuous_thickness_verified'])

    def test_empty_or_line_intersection_rejected(self):
        for third in (box(20, 0, 30, 10), box(10, 0, 20, 10)):
            model = SlabSections([Section(0, box(0, 0, 10, 10)), Section(1, third)])
            with self.assertRaises(ValueError):
                model.effective_problem()

    def test_invalid_heights_and_units_rejected(self):
        for heights in ((0, 0), (1, 0), (0, float('nan'))):
            with self.assertRaises(ValueError):
                SlabSections([Section(heights[0], box(0, 0, 10, 10)),
                              Section(heights[1], box(0, 0, 10, 10))])
        other = box(0, 0, 10, 10)
        other.units = 'cm'
        with self.assertRaises(ValueError):
            SlabSections([Section(0, box(0, 0, 10, 10)), Section(1, other)])

    def test_export_roundtrip_and_recompute_validation(self):
        model = self.model()
        restored = SlabSections.from_dict(model.to_dict())
        self.assertEqual(restored.effective_problem().area, 64)
        data = {'section_model': model.to_dict(), 'state': [[1.5, .5, 0]],
                'angle_unit': 'radians', 'n': 1, 'section_validation': {'accepted': True},
                'problem': box(0, 0, 100, 100).to_dict()}
        self.assertFalse(verify_result(data)['accepted'])
        data['state'] = [[5, 5, 0]]
        self.assertTrue(verify_result(data)['accepted'])
        data['n'] = 2
        with self.assertRaises(ValueError):
            verify_result(data)

    def test_no_implicit_alignment_or_interpolation(self):
        data = self.model().to_dict()
        for key, value in [('coordinate_system', 'local_xy'), ('between_sections', 'linear')]:
            with self.assertRaises(ValueError):
                SlabSections.from_dict({**data, key: value})

    def test_demo_has_three_different_convex_sections(self):
        model = SlabSections.load('examples/slab_sections.json')
        self.assertEqual([len(s.problem.vertices) for s in model.sections], [7, 6, 7])
        self.assertGreater(model.effective_problem().area, 0)
        self.assertLess(model.effective_problem().area, min(s.problem.area for s in model.sections))


if __name__ == '__main__':
    unittest.main()
