"""Optimize one full-thickness slab; verify products against the source solid."""
from __future__ import annotations

import copy
import numpy as np
from shapely.geometry import Polygon

from .block import Block, Frame, checked
from .slab3d import SlabSpec, build_slab
from .problem import Problem
from .solver import solve
from .verify3d import product_geometry, validate_product


def source_geometry(source):
    product = source["product"]
    if product.get("aspect_ratio", 3) != 3 or product.get("kerf", 0) != 0 or product.get("allowance", 0) != 0:
        raise ValueError("Packing uses aspect ratio 3, zero kerf and allowance")
    side = float(product["short_side"])
    if not np.isfinite(side) or side <= 0:
        raise ValueError("short_side must be finite and positive")
    return Block.from_dict(source["block"]), SlabSpec.from_dict(source["slab"]), side


def verify_slab_result(result):
    """Ignore saved footprint, rectangle vertices and all acceptance flags."""
    if result.get("kind") != "slab_packing_result" or result.get("angle_unit") != "radians":
        raise ValueError("Expected a slab packing result with radian angles")
    state = np.asarray(result["state"], dtype=float)
    if state.size == 0:
        state = np.empty((0, 3))
    if state.ndim != 2 or state.shape[1] != 3 or not np.isfinite(state).all() or result["n"] != len(state):
        raise ValueError("Invalid placements or declared count")
    block, spec, side = source_geometry(result["source3d"])
    return verify_layout(block, spec, side, state)


def verify_layout(block, spec, side, state, slab_local=None):
    """Check an entire parallel-slab layout using already rebuilt source geometry."""
    state = np.asarray(state, dtype=float).reshape(-1, 3)
    if not np.isfinite(state).all():
        raise ValueError("Placements must be finite")
    products, polygons = [], []
    for i, placement in enumerate(state):
        try:
            check = validate_product(block, spec, placement, side)
            if slab_local is not None:
                # Rebuild directly in the slab frame: a world→local round trip
                # can add a spurious microscopic Z excursion at a cut plane.
                local_spec=SlabSpec(Frame.from_dict({}),spec.lower,spec.upper)
                local=product_geometry(local_spec,placement,side)[0]
                in_piece=checked(local-slab_local,'product outside separated slab').is_empty()
                check['in_separated_slab']=in_piece
                check['accepted']=check['accepted'] and in_piece
        except ValueError as exc:
            # A numerically invalid CSG residual is an unaccepted candidate,
            # never a small-volume exception that permits a product.
            check = {'accepted':False,'status':'numerical_verification_failed',
                     'missing_volume_mm3':None,'error':str(exc)}
        products.append({"i": i, **check})
        polygons.append(Polygon(product_geometry(spec, placement, side)[2]))
    conflicts = []
    for i, pi in enumerate(polygons):
        for j in range(i + 1, len(polygons)):
            if not pi.relate_pattern(polygons[j], "F********"):
                conflicts.append({"i": i, "j": j, "area_mm2": float(pi.intersection(polygons[j]).area)})
    return {"accepted": all(p["accepted"] for p in products) and not conflicts,
            "n": len(state), "products": products, "conflicting_pairs": conflicts,
            "missing_volume_sum_mm3": (sum(p["missing_volume_mm3"] for p in products)
                                       if all(p['missing_volume_mm3'] is not None for p in products) else None),
            "whole_thickness_verified": True,
            "in_separated_slab": (all(p.get('in_separated_slab',False) for p in products)
                                  if slab_local is not None else None),
            "scope": "source_3d_block_and_all_pair_interiors_in_the_shared_slab_frame",
            "arithmetic": "float64 / Manifold + GEOS; shared numerical kernels, not formal certification"}


def solve_slab(inputs, config=None, progress=None, target_n=None):
    source = {key: copy.deepcopy(inputs[key]) for key in ("block", "slab", "product")}
    block, spec, side = source_geometry(source)
    slab = build_slab(block, spec)
    problem = Problem.from_region(slab.footprint, side, block.name + "_slab", "mm")
    def accept(state):
        return verify_layout(block, spec, side, state)

    result = solve(problem, config, progress, target_n, acceptance_check=accept)
    result.update(kind="slab_packing_result", source3d=source, slab_geometry=slab.diagnostics)
    result["validation3d"] = verify_slab_result(result)
    if not result["validation3d"]["accepted"]:
        raise RuntimeError("Full 3D verification rejected the optimized slab layout")
    return result
