"""Independent from footprint construction: subtract each FULL product from K.

The CSG backend is shared with construction. Analytical regression cases test
that backend; this is numerical verification, not an exact-arithmetic certificate.
"""
from __future__ import annotations

import math
import numpy as np
import manifold3d as m3d

from .block import Block, checked, mesh_arrays
from .slab3d import SlabSpec


def product_geometry(spec: SlabSpec, state, short_side: float):
    values = np.asarray(state, dtype=np.float64)
    short_side = float(short_side)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Product state must be finite x, y, theta")
    if not math.isfinite(short_side) or short_side <= 0:
        raise ValueError("short_side must be positive")
    x, y, angle = values
    angle %= math.pi
    c, s = math.cos(angle), math.sin(angle)
    if abs(c) < 4 * np.finfo(float).eps:
        c = 0.
    if abs(s) < 4 * np.finfo(float).eps:
        s = 0.
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    translation = np.array([x, y, (spec.lower + spec.upper) / 2])
    local = m3d.Manifold.cube((3 * short_side, short_side, spec.thickness), center=True)
    local = local.transform(np.column_stack((rotation, translation)))
    solid = spec.frame.world_solid(local)
    corners = np.array([[sx * 1.5 * short_side, sy * .5 * short_side, sz * .5 * spec.thickness]
                        for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])
    world_corners = spec.frame.to_world(corners @ rotation.T + translation)
    rectangle = np.array([[-1.5, -.5], [1.5, -.5], [1.5, .5], [-1.5, .5]]) * short_side
    rectangle = rectangle @ rotation[:2, :2].T + [x, y]
    return checked(solid, "product", allow_empty=False), world_corners, rectangle


def winding_numbers(solid: m3d.Manifold, points: np.ndarray) -> np.ndarray:
    """Solid-angle diagnostic only. Boundary points are ambiguous; never used for acceptance."""
    vertices, faces = mesh_arrays(solid)
    triangles = vertices[faces]
    results = []
    for point in np.asarray(points):
        a, b, c = np.moveaxis(triangles - point, 1, 0)
        la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (la * lb * lc + np.einsum("ij,ij->i", a, b) * lc
                       + np.einsum("ij,ij->i", b, c) * la + np.einsum("ij,ij->i", c, a) * lb)
        results.append(float(np.sum(2 * np.arctan2(numerator, denominator)) / (4 * np.pi)))
    return np.asarray(results)


def validate_product(block: Block, spec: SlabSpec, state, short_side: float,
                     *, corner_diagnostics: bool = False) -> dict:
    """Strict CSG acceptance; expensive corner diagnostics are explicitly opt-in.

    Diagnostics neither replace nor influence the full-volume subtraction.
    Ordinary search/audit calls avoid scanning the source mesh for each corner.
    """
    product, corners, _ = product_geometry(spec, state, short_side)
    missing = checked(product - block.solid, "product outside block")
    volume = missing.volume()
    # A nonempty residual is rejected even when its volume is very small.
    accepted = missing.is_empty()
    winding = winding_numbers(block.solid, corners) if corner_diagnostics else None
    return {"accepted": accepted, "status": "feasible" if accepted else "infeasible",
            "missing_volume_mm3": volume, "product_volume_mm3": product.volume(),
            "world_vertices": corners.tolist(),
            "corner_diagnostics_computed": winding is not None,
            "corner_winding_numbers": winding.tolist() if winding is not None else None,
            "all_corners_strictly_inside_diagnostic": (bool(np.all(np.abs(winding - 1) < 1e-7))
                                                      if winding is not None else None),
            "method": "full_product_minus_original_block",
            "acceptance_uses_corners_or_samples": False,
            "nonempty_residual_is_rejected": True,
            "backend_tolerance_mm": max(block.solid.get_tolerance(), product.get_tolerance()),
            "arithmetic": "Manifold float64 CSG; independent of the derived 2D footprint"}
