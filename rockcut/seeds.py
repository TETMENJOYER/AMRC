"""Reproducible shelves for warm starts, then stochastic insertion/moves."""
from __future__ import annotations

import math
import time
import numpy as np
import shapely
from shapely.geometry import Polygon, LineString
from .problem import Problem
from .geometry import corners, area_score


def shelf_seeds(problem: Problem, limit: int = 8, deadline: float = math.inf,
                margin: float = 1e-5, diverse: bool = False) -> list[np.ndarray]:
    if not problem.is_convex:
        return region_shelf_seeds(problem, limit, deadline, margin, diverse)
    v = problem.normalized_vertices
    edges = np.roll(v, -1, axis=0) - v
    angles = np.unique(np.r_[np.linspace(0, np.pi, 18, endpoint=False),
                             np.arctan2(edges[:, 1], edges[:, 0]) % np.pi])
    layouts = [np.empty((0, 3))]
    for angle in angles:
        if time.monotonic() >= deadline:
            break
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        local_vertices = v @ rot
        normals = problem.normals @ rot
        bounds = problem.offsets - 1.5 * abs(normals[:, 0]) - 0.5 * abs(normals[:, 1]) - margin
        ymin, ymax = local_vertices[:, 1].min() + 0.5 + margin, local_vertices[:, 1].max() - 0.5 - margin
        for phase in (0, 0.25, 0.5, 0.75):
            for alignment in (0, 0.5, 1):
                states = []
                for y in np.arange(ymin + phase, ymax + 1e-12, 1 + 2 * margin):
                    lo, hi = -np.inf, np.inf
                    for normal, bound in zip(normals, bounds):
                        rhs = bound - normal[1] * y
                        if abs(normal[0]) < 1e-12:
                            if rhs < 0:
                                lo, hi = 1, 0
                                break
                        elif normal[0] > 0:
                            hi = min(hi, rhs / normal[0])
                        else:
                            lo = max(lo, rhs / normal[0])
                    if lo > hi or not np.isfinite([lo, hi]).all():
                        continue
                    count = int(np.floor((hi - lo) / (3 + 2 * margin))) + 1
                    start = lo + alignment * (hi - lo - (count - 1) * (3 + 2 * margin))
                    for x in start + np.arange(count) * (3 + 2 * margin):
                        cx, cy = np.array([x, y]) @ rot.T
                        states.append([cx, cy, angle])
                layouts.append(np.array(states, dtype=float).reshape(-1, 3))
    layouts.sort(key=len, reverse=True)
    if diverse:
        selected, orientations = [], []
        for layout in layouts:
            if not len(layout):
                continue
            angle = layout[0, 2]
            if any(abs((angle - old + np.pi / 2) % np.pi - np.pi / 2) < 0.035
                   for old in orientations):
                continue
            selected.append(layout)
            orientations.append(angle)
            if len(selected) == limit:
                break
        return selected or [np.empty((0, 3))]
    return layouts[:limit]


def center_region(problem: Problem, angle: float, margin: float = 0):
    """Exact polygonal center domain at a fixed angle, in rotated coordinates.

    Erode the convex bound by rectangle support, then exclude Minkowski sums
    of forbidden triangles and the reflected rectangle. No vertex sampling.
    """
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = np.asarray(problem.polygon.convex_hull.exterior.coords)[:-1] @ rot
    normals = problem.normals @ rot
    bounds = problem.offsets - (1.5 + margin) * abs(normals[:, 0]) - (.5 + margin) * abs(normals[:, 1])
    for normal, bound in zip(normals, bounds):
        clipped = []
        for a, b in zip(points, np.roll(points, -1, axis=0)):
            da, db = a @ normal - bound, b @ normal - bound
            if da <= 0:
                clipped.append(a)
            if (da <= 0) != (db <= 0):
                clipped.append(a + da / (da - db) * (b - a))
        points = np.asarray(clipped).reshape(-1, 2)
        if len(points) < 3:
            return Polygon()
    domain = Polygon(points)
    offsets = np.array([[-1.5-margin, -.5-margin], [1.5+margin, -.5-margin],
                        [1.5+margin, .5+margin], [-1.5-margin, .5+margin]])
    # Keep each triangle's coordinate arithmetic unchanged. Vectorize only GEOS
    # object creation/hulls, avoiding one Python MultiPoint wrapper per obstacle.
    forbidden_points = [((triangle @ rot)[:, None, :] + offsets[None]).reshape(-1, 2)
                        for triangle in problem.obstacle_triangles]
    if forbidden_points:
        forbidden = shapely.convex_hull(shapely.multipoints(forbidden_points))
        domain = domain.difference(shapely.union_all(forbidden))
    return domain


def region_shelf_seeds(problem, limit, deadline, margin, diverse):
    edges = []
    components = problem.polygon.geoms if problem.polygon.geom_type == "MultiPolygon" else [problem.polygon]
    for polygon in components:
        for ring in [polygon.exterior, *polygon.interiors]:
            v = np.asarray(ring.coords)[:-1]
            edges.extend(np.roll(v, -1, axis=0) - v)
    edges = np.asarray(edges)
    angles = np.unique(np.r_[np.linspace(0, np.pi, 18, endpoint=False),
                             np.arctan2(edges[:, 1], edges[:, 0]) % np.pi])
    layouts = [np.empty((0, 3))]
    for angle in angles:
        if time.monotonic() >= deadline:
            break
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        if margin == 0:
            # A legal center set can collapse to a line in an exact-width arm.
            # Polygon booleans discard such lines; test contact grids directly.
            v = problem.normalized_vertices @ rot
            low, high = v.min(0), v.max(0)
            for phase in (0, .5):
                xs = np.arange(low[0] + 1.5, high[0] - 1.5 + 1e-12, 3)
                ys = np.arange(low[1] + .5 + phase, high[1] - .5 + 1e-12, 1)
                xy = np.array([[x, y] for y in ys for x in xs], dtype=float).reshape(-1, 2)
                q = np.column_stack((xy @ rot.T, np.full(len(xy), angle)))
                if len(q):
                    q = q[shapely.covers(problem.polygon, shapely.polygons(corners(q)))]
                layouts.append(q)
        domain = center_region(problem, angle, margin)
        if domain.is_empty or domain.area <= 0:
            continue
        xmin, ymin, xmax, ymax = domain.bounds
        # The center domain is immutable for this angle. All three alignment
        # choices intersect exactly the same horizontal rows; cache their bounds
        # locally, never across angles, problems, margins, or calls.
        row_sections = {}
        for phase in (0, .25, .5, .75):
            for alignment in (0, .5, 1):
                if time.monotonic() >= deadline:
                    break
                states = []
                for y in np.arange(ymin + phase, ymax + 1e-12, 1 + 2 * margin):
                    if y not in row_sections:
                        section = domain.intersection(LineString([[xmin-1, y], [xmax+1, y]]))
                        pieces = list(section.geoms) if hasattr(section, "geoms") else [section]
                        row_sections[y] = [piece.bounds for piece in pieces
                                           if not piece.is_empty and piece.geom_type in ("LineString", "Point")]
                    for lo, _, hi, _ in row_sections[y]:
                        count = int(np.floor((hi - lo) / (3 + 2 * margin))) + 1
                        start = lo + alignment * (hi - lo - (count - 1) * (3 + 2 * margin))
                        for x in start + np.arange(count) * (3 + 2 * margin):
                            states.append([*(np.array([x, y]) @ rot.T), angle])
                q = np.asarray(states, dtype=float).reshape(-1, 3)
                # Preserve exact whole-polygon acceptance despite roundoff in erosion.
                if len(q):
                    keep = shapely.covers(problem.polygon, shapely.polygons(corners(q)))
                    q = q[keep]
                layouts.append(q)
    layouts.sort(key=len, reverse=True)
    if diverse:
        selected, orientations = [], []
        for q in layouts:
            if not len(q) or any(abs((q[0, 2]-a+np.pi/2) % np.pi - np.pi/2) < .035 for a in orientations):
                continue
            selected.append(q); orientations.append(q[0, 2])
            if len(selected) == limit:
                break
        return selected or [np.empty((0, 3))]
    return layouts[:limit]


def fill_gaps(problem: Problem, state: np.ndarray, deadline: float = math.inf) -> np.ndarray:
    """Add rectangles only after exact predicates; this is just an initializer."""
    result = list(state.copy())
    shapes = list(shapely.polygons(corners(state))) if len(state) else []
    low, high = problem.normalized_vertices.min(0), problem.normalized_vertices.max(0)
    for angle in (0, np.pi / 2):
        for y in np.arange(low[1] + 0.5, high[1], 0.25):
            if time.monotonic() >= deadline:
                return np.array(result).reshape(-1, 3)
            for x in np.arange(low[0] + 0.5, high[0], 0.25):
                candidate = np.array([[x, y, angle]])
                shape = shapely.polygons(corners(candidate))[0]
                if not problem.polygon.covers(shape):
                    continue
                if shapes and not shapely.relate_pattern(shapes, shape, "F********").all():
                    continue
                result.append(candidate[0])
                shapes.append(shape)
    return np.array(result, dtype=float).reshape(-1, 3)


def insert_one(problem: Problem, state: np.ndarray, rng: np.random.Generator,
               choices: int = 24) -> np.ndarray:
    if choices < 1:
        raise ValueError("choices must be positive")
    low, high = problem.normalized_vertices.min(0), problem.normalized_vertices.max(0)
    proposals = []
    for _ in range(choices):
        xy = rng.uniform(low, high)
        angle = rng.uniform(0, np.pi) if rng.random() < 0.4 else rng.choice([0, np.pi / 2])
        proposals.append([*xy, angle])
    proposals = np.asarray(proposals)
    shapes = shapely.polygons(corners(proposals))
    values = shapely.area(shapely.difference(shapes, problem.polygon))
    if len(state):
        fixed = shapely.polygons(corners(state))
        values += shapely.area(shapely.intersection(shapes[:, None], fixed[None, :])).sum(axis=1)
    # All existing-pair areas and existing outside areas are constant across
    # these proposals. Evaluate only the contribution of the inserted rectangle.
    return np.vstack((state, proposals[np.argmin(values)]))


def perturb(problem: Problem, state: np.ndarray, rng: np.random.Generator,
            strength: float = 1.0) -> np.ndarray:
    q = state.copy()
    if len(q) == 0:
        return q
    conflicts = area_score(problem, q).per_rectangle + 0.03
    anchor = rng.choice(len(q), p=conflicts / conflicts.sum())
    kind = rng.integers(5)
    if kind == 0:
        indices = [anchor]
        q[indices, :2] += rng.normal(0, 0.65 * strength, (1, 2))
        q[indices, 2] += rng.normal(0, 0.3 * strength)
    elif kind == 1:
        distances = np.linalg.norm(q[:, :2] - q[anchor, :2], axis=1)
        indices = np.argsort(distances)[:min(len(q), int(rng.integers(2, 6)))]
        q[indices, :2] += rng.normal(0, 0.45 * strength, (len(indices), 2))
        q[indices, 2] += rng.normal(0, 0.22 * strength, len(indices))
    elif kind == 2:
        q[anchor, :2] = rng.uniform(problem.normalized_vertices.min(0), problem.normalized_vertices.max(0))
        q[anchor, 2] = rng.uniform(0, np.pi)
    elif kind == 3:
        q[anchor, 2] += rng.choice([-1, 1]) * np.pi / 2 + rng.normal(0, 0.03)
        q[anchor, :2] += rng.normal(0, 0.15, 2)
    else:
        q[:, :2] += rng.normal(0, 0.18 * strength, (len(q), 2))
        q[:, 2] += rng.normal(0, 0.07 * strength, len(q))
    q[:, 2] %= np.pi
    return q


def random_layout(problem: Problem, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform centers in a convex polygon, using area-weighted fan triangles."""
    if not problem.is_convex:
        triangles = problem.sampling_triangles
        a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        areas = abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        indices = rng.choice(len(areas), size=n, p=areas / areas.sum())
        xy = np.einsum("ni,nij->nj", rng.dirichlet([1, 1, 1], size=n), triangles[indices])
        return np.column_stack((xy, rng.uniform(0, np.pi, n)))
    v = problem.normalized_vertices
    a, b = v[1:-1] - v[0], v[2:] - v[0]
    areas = abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    indices = rng.choice(len(areas), size=n, p=areas / areas.sum())
    triangles = np.stack((np.broadcast_to(v[0], (n, 2)), v[indices + 1], v[indices + 2]), axis=1)
    xy = np.einsum("ni,nij->nj", rng.dirichlet([1, 1, 1], size=n), triangles)
    return np.column_stack((xy, rng.uniform(0, np.pi, n)))


def destroy_repair(problem: Problem, state: np.ndarray, rng: np.random.Generator,
                   deadline: float = math.inf) -> np.ndarray:
    """Reinsert a conflict-centered cluster, preserving every rectangle's slot."""
    if not len(state):
        return state.copy()
    conflicts = area_score(problem, state).per_rectangle + 0.03
    anchor = rng.choice(len(state), p=conflicts / conflicts.sum())
    count = min(len(state), int(rng.integers(3, 9)))
    distances = np.linalg.norm(state[:, :2] - state[anchor, :2], axis=1)
    removed = np.argsort(distances)[:count]
    rng.shuffle(removed)
    working = np.delete(state, removed, axis=0)
    result = state.copy()
    for slot in removed:
        if time.monotonic() >= deadline:
            # Abandon an incomplete reconstruction, rather than lose rectangles.
            return state.copy()
        working = insert_one(problem, working, rng, choices=32)
        result[slot] = working[-1]
    return result
