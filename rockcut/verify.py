"""Independent CPU validator in original physical units.

This module deliberately does not import the optimizer, its penalties, or corners.
Strict GEOS predicates decide acceptance; tolerance-only candidates stay unaccepted.
"""
from __future__ import annotations

import math
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
from .problem import Problem


def check_candidate(problem, state, acceptance_check=None):
    check = validate(problem, state)
    if check["accepted"] and acceptance_check is not None:
        additional = acceptance_check(state)
        check["additional_check"] = additional
        if not additional["accepted"]:
            check.update(accepted=False, status="rejected_by_additional_check")
    return check


def verified_subset(state, check_layout):
    """Remove invalid seed products and conflicts, then verify the retained layout.

    Only validators with explicit per-product diagnostics permit salvage. An
    unexplained rejection falls back to the empty layout. This heuristic never
    upgrades a failed predicate or bypasses final verification.
    """
    candidate = np.asarray(state, dtype=float).reshape(-1, 3).copy()
    for _ in range(3):
        check = check_layout(candidate)
        if check["accepted"]:
            return candidate, check
        diagnostic = check.get("additional_check", check)
        if "products" in diagnostic:
            products = diagnostic["products"]
            if len(products) != len(candidate) or [p.get("i") for p in products] != list(range(len(candidate))):
                break
            keep = {p["i"] for p in products if p["accepted"]}
        elif "outside_indices" in diagnostic:
            keep = set(range(len(candidate))) - set(diagnostic["outside_indices"])
        else:
            break
        conflicts = [(p["i"], p["j"]) for p in diagnostic.get("conflicting_pairs", [])]
        while True:
            active = [(i, j) for i, j in conflicts if i in keep and j in keep]
            if not active:
                break
            degree = {i: 0 for i in keep}
            for i, j in active:
                degree[i] += 1; degree[j] += 1
            keep.remove(max(keep, key=lambda i: (degree[i], i)))
        if len(keep) >= len(candidate):
            break
        candidate = candidate[sorted(keep)]
    empty = np.empty((0, 3))
    check = check_layout(empty)
    if not check["accepted"]:
        raise RuntimeError("Verification rejected the empty fallback layout")
    return empty, check


def validate(problem: Problem, physical_state: np.ndarray,
             length_tolerance: float | None = None,
             area_tolerance: float | None = None) -> dict:
    length_tolerance = problem.short_side * 1e-7 if length_tolerance is None else length_tolerance
    area_tolerance = problem.short_side**2 * 1e-9 if area_tolerance is None else area_tolerance
    if not (math.isfinite(length_tolerance) and length_tolerance >= 0 and
            math.isfinite(area_tolerance) and area_tolerance >= 0):
        raise ValueError("Validation tolerances must be finite and nonnegative")
    state = np.asarray(physical_state, dtype=float)
    if state.size == 0:
        state = np.empty((0, 3))
    if state.ndim != 2 or state.shape[1] != 3 or not np.isfinite(state).all():
        raise ValueError("State must be a finite (N, 3) array")
    # Translate for numerical stability, retaining physical units and dimensions.
    reference = problem.vertices[0]
    slab_points = problem.vertices - reference
    slab = translate(problem.physical_region, -reference[0], -reference[1])
    a, b = 3 * problem.short_side, problem.short_side
    polygons, vertices, original_vertices = [], [], []
    for x, y, theta in state:
        theta = theta % math.pi
        co, si = math.cos(theta), math.sin(theta)
        if abs(co) < 4 * np.finfo(float).eps:
            co = 0.0
        if abs(si) < 4 * np.finfo(float).eps:
            si = 0.0
        points = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            points.append([x - reference[0] + sx * a / 2 * co - sy * b / 2 * si,
                           y - reference[1] + sx * a / 2 * si + sy * b / 2 * co])
        points = np.array(points)
        vertices.append(points)
        original_vertices.append((points + reference).tolist())
        polygons.append(Polygon(points))
    if problem.is_convex:
        edges = np.roll(slab_points, -1, axis=0) - slab_points
        normals = np.array([[e[1], -e[0]] for e in edges])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        offsets = np.sum(normals * slab_points, axis=1)
    outside_areas, boundary_depths, outside_indices = [], [], []
    for i, (p, vs) in enumerate(zip(polygons, vertices)):
        outside_areas.append(float(p.difference(slab).area))
        if problem.is_convex:
            depth = float(max(0, np.max(vs @ normals.T - offsets)))
        else:
            # Diagnostic only: detect excluded interiors even when all corners
            # are inside. Strict covers() below decides acceptance, not this proxy.
            depth = max(math.sqrt(outside_areas[-1]), max(slab.distance(Point(v)) for v in vs))
        boundary_depths.append(depth)
        if not slab.covers(p):
            outside_indices.append(i)
    conflicts, pair_areas, depths = [], [], []
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            pi, pj = polygons[i], polygons[j]
            area = float(pi.intersection(pj).area)
            pair_areas.append(area)
            if not pi.relate_pattern(pj, "F********"):
                candidates = []
                for vs in (vertices[i], vertices[j]):
                    for k in (0, 1):
                        axis = vs[k + 1] - vs[k]
                        axis = axis / np.linalg.norm(axis)
                        p, q = vertices[i] @ axis, vertices[j] @ axis
                        candidates.append(min(p.max() - q.min(), q.max() - p.min()))
                depth = float(max(0, min(candidates)))
                conflicts.append({"i": i, "j": j, "area": area, "penetration": depth})
                depths.append(depth)
    total_overlap, total_outside = sum(pair_areas), sum(outside_areas)
    max_depth, max_outside = max(depths, default=0.0), max(boundary_depths, default=0.0)
    strict = not conflicts and not outside_indices
    near = (total_overlap <= area_tolerance and total_outside <= area_tolerance
            and max_depth <= length_tolerance and max_outside <= length_tolerance)
    return {
        "status": "feasible" if strict else "near_feasible" if near else "infeasible",
        "accepted": strict,
        "n": len(state), "overlap_area_sum": total_overlap, "outside_area_sum": total_outside,
        "max_pair_overlap_area": max(pair_areas, default=0.0),
        "max_penetration": max_depth, "max_boundary_violation": max_outside,
        "boundary_violation_metric": "halfplane_depth" if problem.is_convex else "max_corner_distance_or_sqrt_outside_area",
        "length_tolerance": length_tolerance, "area_tolerance": area_tolerance,
        "outside_indices": outside_indices, "conflicting_pairs": conflicts,
        "rectangle_vertices": original_vertices,
        "arithmetic": "float64 / GEOS; numerical geometric verification, not a formal certificate",
        "units": problem.units,
    }
