"""Stream physical rigid-grid proposals; callers own final 2D/3D acceptance."""
import math
import time

import numpy as np
import shapely
from shapely.affinity import translate


def _diagonal_pairs(first_count, second_count, reverse_first=False):
    """Fair finite Cartesian prefix without materializing the Cartesian product."""
    for total in range(first_count+second_count-1):
        low, high = max(0, total-second_count+1), min(total, first_count-1)
        indices = range(high, low-1, -1) if reverse_first else range(low, high+1)
        for first in indices:
            yield first, total-first


def _phases(count):
    indices = [0]
    for numerator, denominator in ((1, 2), (1, 4), (3, 4)):
        if count*numerator % denominator == 0:
            index = count*numerator//denominator
            if index not in indices:
                indices.append(index)
    indices.extend(index for index in range(count) if index not in indices)
    return [index/count for index in indices]


def _directions(count):
    angles = [0., math.pi/2]
    for index in range(count):
        angle = index*math.pi/count
        if not any(abs(angle-old) < 4*np.finfo(float).eps for old in angles):
            angles.append(angle)
    return angles


def rigid_grid_candidates(problem, deadline, margin=0., angle_steps=18, phase_steps=4):
    """Yield (physical state, metadata) from one angle and two GLOBAL offsets.

    Phases are fractions of the nominal 3s by s cell; pitch is (3+2m)s by
    (1+2m)s. Margin m is a proposal gap in normalized units, never a tolerance.
    Both axis orientations precede remaining uniform angles. Deadline checks
    bracket rows/chunks; a started GEOS call may overrun softly. Single cells
    must satisfy strict covers. Pair-interior and full-source/assigned-piece
    predicates remain with the caller, especially for rotated contact grids.
    No state is accepted here and no physical domain is repaired or expanded.
    """
    for name, value in (('angle_steps', angle_steps), ('phase_steps', phase_steps)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('margin must be finite and nonnegative')
    if math.isnan(deadline) or deadline == -math.inf:
        raise ValueError('deadline must be finite or positive infinity')
    if time.monotonic() >= deadline:
        return
    # Translate exactly as the independent physical validator does. Rotation
    # affects only proposal bounds; coverage uses the full unrotated region.
    preparation_started = time.monotonic()
    reference = problem.vertices[0]
    region = translate(problem.physical_region, -reference[0], -reference[1])
    coordinates = shapely.get_coordinates(problem.physical_region)-reference
    side = problem.short_side
    width, height = 3*side, side
    pitch_x, pitch_y = (3+2*margin)*side, (1+2*margin)*side
    angles, phases = _directions(angle_steps), _phases(phase_steps)
    pairs = list(_diagonal_pairs(len(phases), len(phases)))
    preparation_seconds = time.monotonic()-preparation_started
    bounds_cache = {}
    sx, sy = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])
    for angle_index, phase_index in _diagonal_pairs(len(angles), len(pairs), reverse_first=True):
        if time.monotonic() >= deadline:
            return
        started = time.monotonic()
        angle = angles[angle_index]
        co, si = math.cos(angle), math.sin(angle)
        if abs(co) < 4*np.finfo(float).eps:
            co = 0.
        if abs(si) < 4*np.finfo(float).eps:
            si = 0.
        rotation = np.array([[co, -si], [si, co]])
        if angle_index not in bounds_cache:
            rotated = coordinates @ rotation
            bounds_cache[angle_index] = rotated.min(0), rotated.max(0)
        low, high = bounds_cache[angle_index]
        ix, iy = pairs[phase_index]
        phase_x, phase_y = phases[ix], phases[iy]
        xstart = low[0]+width/2+margin*side+phase_x*width
        ystart = low[1]+height/2+margin*side+phase_y*height
        xmax, ymax = high[0]-width/2-margin*side, high[1]-height/2-margin*side
        columns = max(0, math.floor((xmax-xstart)/pitch_x)+1)
        rows = max(0, math.floor((ymax-ystart)/pitch_y)+1)
        retained, generated, filtering_seconds = [], 0, 0.
        complete = True
        for row in range(rows if columns else 0):
            if time.monotonic() >= deadline:
                complete = False
                break
            y = ystart+row*pitch_y
            for offset in range(0, columns, 128):
                if time.monotonic() >= deadline:
                    complete = False
                    break
                xs = xstart+np.arange(offset, min(columns, offset+128))*pitch_x
                local = np.column_stack((xs, np.full(len(xs), y)))
                centers = local @ rotation.T+reference
                state = np.column_stack((centers, np.full(len(centers), angle)))
                # Match the validator's left-associated corner arithmetic.
                px = centers[:, 0, None]-reference[0]+sx*width/2*co-sy*height/2*si
                py = centers[:, 1, None]-reference[1]+sx*width/2*si+sy*height/2*co
                filter_started = time.monotonic()
                keep = shapely.covers(region, shapely.polygons(np.stack((px, py), axis=-1)))
                filtering_seconds += time.monotonic()-filter_started
                generated += len(state)
                if np.any(keep):
                    retained.append(state[keep])
            if not complete:
                break
        physical = np.vstack(retained) if retained else np.empty((0, 3), dtype=float)
        elapsed = time.monotonic()-started
        yield physical, {'angle_radians': angle, 'phase_x': phase_x, 'phase_y': phase_y,
            'margin': margin, 'complete': complete, 'generated_cells': generated,
            'retained_cells': len(physical), 'planned_cells': rows*columns,
            'generating_seconds': elapsed-filtering_seconds, 'filtering_seconds': filtering_seconds,
            'preparation_seconds': preparation_seconds if angle_index == phase_index == 0 else 0.,
            'deadline_overrun_seconds': max(0., time.monotonic()-deadline)}
