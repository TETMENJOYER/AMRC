"""Physical rigid-grid proposals preserve global phases and strict cell covers."""
import math
import unittest
from unittest.mock import patch

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon, box

from rockcut.grid import rigid_grid_candidates
from rockcut.problem import Problem
from rockcut.verify import check_candidate


class RigidGridTests(unittest.TestCase):
    def test_both_axis_orientations_are_first_and_contact_grid_is_strict(self):
        problem = Problem([[0, 0], [6, 0], [6, 2], [0, 2]], 1)
        candidates = rigid_grid_candidates(problem, math.inf, angle_steps=2)
        horizontal, first = next(candidates)
        vertical, second = next(candidates)
        self.assertEqual(first['angle_radians'], 0.)
        self.assertEqual(second['angle_radians'], math.pi/2)
        self.assertEqual(first['retained_cells'], 4)
        self.assertEqual(first['generated_cells'], 4)
        self.assertTrue(check_candidate(problem, horizontal)['accepted'])
        self.assertTrue(check_candidate(problem, vertical)['accepted'])
        self.assertEqual(first['phase_x'], first['phase_y'])
        self.assertEqual(first['margin'], 0.)

    def test_global_xy_phases_and_angle_are_shared_across_every_row(self):
        problem = Problem([[10, 20], [34, 20], [34, 32], [10, 32]], 2)
        trials = list(rigid_grid_candidates(problem, math.inf, angle_steps=2))
        self.assertEqual(len(trials), 32)
        phases = {(p['phase_x'], p['phase_y']) for _, p in trials if p['angle_radians'] == 0.}
        self.assertEqual(phases, {(x, y) for x in (0., .5, .25, .75) for y in (0., .5, .25, .75)})
        for state, parameters in trials:
            angle = parameters['angle_radians']
            co, si = (1., 0.) if angle == 0. else (0., 1.)
            rotation = np.array([[co, -si], [si, co]])
            vertices = (problem.vertices-problem.vertices[0]) @ rotation
            low = vertices.min(0)
            centers = (state[:, :2]-problem.vertices[0]) @ rotation
            indices = (centers-low-[3., 1.]-np.array([parameters['phase_x']*6,
                parameters['phase_y']*2]))/[6., 2.]
            np.testing.assert_allclose(indices, np.rint(indices), atol=1e-12)
            np.testing.assert_array_equal(state[:, 2], np.full(len(state), angle))
            self.assertTrue(check_candidate(problem, state)['accepted'])

    def test_inside_center_does_not_authorize_cell_crossing_a_concave_boundary(self):
        region = Polygon([[0, 0], [6, 0], [6, 2], [4, 2], [4, .8], [2, .8], [2, 2], [0, 2]])
        problem = Problem.from_region(region, 1)
        self.assertTrue(shapely.contains_xy(region, 1.5, .5))
        state, params = next(rigid_grid_candidates(problem, math.inf))
        self.assertEqual(params['generated_cells'], 4)
        self.assertFalse(np.any(np.all(state[:, :2] == [1.5, .5], axis=1)))
        self.assertTrue(check_candidate(problem, state)['accepted'])

    def test_disjoint_components_share_one_grid_and_gap_cells_are_removed(self):
        region = MultiPolygon([box(0, 0, 3, 1), box(6, 0, 9, 1)])
        problem = Problem.from_region(region, 1)
        state, params = next(rigid_grid_candidates(problem, math.inf))
        np.testing.assert_array_equal(state[:, :2], [[1.5, .5], [7.5, .5]])
        self.assertEqual(params['generated_cells'], 3)
        self.assertEqual(params['retained_cells'], 2)
        self.assertTrue(check_candidate(problem, state)['accepted'])
        self.assertTrue(problem.physical_region.equals(region))

    def test_holes_are_preserved_and_entire_hole_crossing_cell_is_removed(self):
        region = Polygon(box(0, 0, 9, 3).exterior.coords, [box(3, 1, 6, 2).exterior.coords])
        problem = Problem.from_region(region, 1)
        state, params = next(rigid_grid_candidates(problem, math.inf))
        self.assertEqual(params['generated_cells'], 9)
        self.assertEqual(params['retained_cells'], 8)
        self.assertFalse(np.any(np.all(state[:, :2] == [4.5, 1.5], axis=1)))
        self.assertTrue(check_candidate(problem, state)['accepted'])

    def test_small_violation_is_not_converted_to_coverage(self):
        problem = Problem.from_region(box(0, 0, 3, np.nextafter(1., 0.)), 1)
        state, params = next(rigid_grid_candidates(problem, math.inf))
        self.assertEqual(len(state), 0)
        self.assertTrue(params['complete'])

    def test_positive_margin_is_only_a_proposal_gap(self):
        problem = Problem.from_region(box(0, 0, 10, 5), 1)
        state, params = next(rigid_grid_candidates(problem, math.inf, margin=.1))
        self.assertTrue(check_candidate(problem, state)['accepted'])
        self.assertGreater(len(state), 1)
        np.testing.assert_allclose(np.diff(np.unique(state[:, 0])), 3.2)
        np.testing.assert_allclose(np.diff(np.unique(state[:, 1])), 1.2)
        self.assertEqual(params['margin'], .1)

    def test_uniform_nonaxis_angles_use_physical_whole_cell_coverage(self):
        from shapely.affinity import rotate, translate
        region = translate(rotate(box(0, 0, 18, 9), 45, origin=(0, 0)), 1000, -2000)
        problem = Problem.from_region(region, 1)
        trials = list(rigid_grid_candidates(problem, math.inf, margin=.01, angle_steps=4, phase_steps=2))
        self.assertEqual({p['angle_radians'] for _, p in trials}, {0., math.pi/2, math.pi/4, 3*math.pi/4})
        self.assertTrue(any(len(state) > 1 for state, p in trials if p['angle_radians'] == math.pi/4))
        for state, params in trials:
            np.testing.assert_array_equal(state[:, 2], np.full(len(state), params['angle_radians']))
            # Every retained cell must independently satisfy the physical
            # validator even at nonaxis angles and a nonzero world origin.
            for cell in state:
                self.assertTrue(check_candidate(problem, cell[None])['accepted'])

    def test_deadline_yields_independently_valid_partial_grid_with_honest_metadata(self):
        problem = Problem.from_region(box(0, 0, 6, 100), 1)
        clock = [0.]
        original_covers = shapely.covers

        def covers(*args, **kwargs):
            clock[0] += 1.
            return original_covers(*args, **kwargs)

        with patch('rockcut.grid.time.monotonic', side_effect=lambda: clock[0]), \
             patch('rockcut.grid.shapely.covers', side_effect=covers):
            candidates = list(rigid_grid_candidates(problem, .5))
        self.assertEqual(len(candidates), 1)
        state, params = candidates[0]
        self.assertEqual(len(state), 2)
        self.assertEqual(params['generated_cells'], 2)
        self.assertFalse(params['complete'])
        self.assertEqual(params['planned_cells'], 200)
        self.assertEqual(params['deadline_overrun_seconds'], .5)
        self.assertTrue(check_candidate(problem, state)['accepted'])

    def test_expired_deadline_does_not_build_any_cells(self):
        problem = Problem.from_region(box(0, 0, 6, 3), 1)
        with patch('rockcut.grid.time.monotonic', return_value=1.), \
             patch('rockcut.grid.shapely.covers') as covered:
            self.assertEqual(list(rigid_grid_candidates(problem, .5)), [])
        covered.assert_not_called()

    def test_invalid_controls_propagate(self):
        problem = Problem.from_region(box(0, 0, 6, 3), 1)
        for kwargs in ({'angle_steps': 0}, {'phase_steps': True}, {'phase_steps': -1},
                       {'margin': -.1}, {'margin': math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                list(rigid_grid_candidates(problem, math.inf, **kwargs))
        with self.assertRaises(ValueError):
            list(rigid_grid_candidates(problem, math.nan))


if __name__ == '__main__':
    unittest.main()
