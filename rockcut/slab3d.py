"""Full-thickness footprint of a nonconvex solid, without sampling in z."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import manifold3d as m3d
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

from .block import Block, Frame, checked, mesh_arrays


@dataclass
class SlabSpec:
    frame: Frame
    lower: float
    upper: float

    def __post_init__(self):
        self.lower, self.upper = float(self.lower), float(self.upper)
        if not np.isfinite([self.lower, self.upper]).all() or self.upper <= self.lower:
            raise ValueError("Slab requires finite lower < upper")

    @classmethod
    def from_dict(cls, data: dict) -> "SlabSpec":
        return cls(Frame.from_dict(data.get("frame", {})), data["lower"], data["upper"])

    def to_dict(self) -> dict:
        return {"frame": self.frame.to_dict(), "lower": self.lower, "upper": self.upper}

    @property
    def thickness(self) -> float:
        return self.upper - self.lower


def polygonal(geometry):
    if geometry.is_empty:
        return Polygon()
    if geometry.geom_type not in ("Polygon", "MultiPolygon") or not geometry.is_valid:
        raise ValueError("Expected valid polygonal area; no automatic geometry repair")
    return geometry


def project_volume(solid: m3d.Manifold):
    """Project a CLOSED volume by the union of all projected boundary triangles.

    Every vertical line meeting a bounded solid meets its boundary. Using the
    complete boundary therefore includes internal cavities and disconnected parts.
    """
    if solid.is_empty():
        return Polygon()
    vertices, faces = mesh_arrays(solid)
    triangles = []
    for face in faces:
        points = vertices[face, :2]
        # Only exactly zero-area projections are discarded, not small features.
        cross = ((points[1, 0] - points[0, 0]) * (points[2, 1] - points[0, 1])
                 - (points[1, 1] - points[0, 1]) * (points[2, 0] - points[0, 0]))
        if cross != 0:
            triangles.append(Polygon(points))
    return polygonal(unary_union(triangles))


def sample_section(solid: m3d.Manifold, height: float):
    """Display-only section; Manifold's topmost coplanar face is excluded.

    Convert nonintersecting nested contours with even/odd fill, preserving islands
    within holes. These sample sections never authorize product acceptance.
    """
    contours = [Polygon(p) for p in solid.slice(height).to_polygons() if len(p) >= 3]
    region = Polygon()
    for contour in sorted(contours, key=lambda p: -p.area):
        if not contour.is_valid:
            raise ValueError("Invalid sample contour")
        region = region.symmetric_difference(contour)
    return polygonal(region)


@dataclass
class SlabGeometry:
    spec: SlabSpec
    local_block: m3d.Manifold
    solid: m3d.Manifold
    footprint: object
    sample_heights: list[float]
    sample_regions: list
    sampled_intersection: object
    diagnostics: dict


def build_slab(block: Block, spec: SlabSpec) -> SlabGeometry:
    local = spec.frame.local_solid(block.solid)
    vertices, _ = mesh_arrays(local)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    span = hi - lo
    pad = max(1., float(span.max()) * .01)
    x0, y0 = lo[:2] - pad
    x1, y1 = hi[:2] + pad
    surrounding = m3d.Manifold.cube((x1 - x0, y1 - y0, spec.thickness)).translate((x0, y0, spec.lower))
    slab = checked(local ^ surrounding, "slab intersection")
    absent = checked(surrounding - local, "absent material")
    forbidden = project_volume(absent)
    footprint = polygonal(box(x0, y0, x1, y1).difference(forbidden))
    lower = checked(local.trim_by_plane((0, 0, -1), -spec.lower), "lower remainder")
    upper = checked(local.trim_by_plane((0, 0, 1), spec.upper), "upper remainder")
    volume_error = abs(local.volume() - lower.volume() - slab.volume() - upper.volume())
    if volume_error > 1e-9 * max(1, local.volume()):
        raise ValueError("Slab volume partition failed")
    heights = [spec.lower, (spec.lower + spec.upper) / 2, spec.upper]
    samples = [sample_section(local, z) for z in heights]
    naive = samples[0]
    for sample in samples[1:]:
        naive = naive.intersection(sample)
    naive = polygonal(naive)
    parts = list(footprint.geoms) if footprint.geom_type == "MultiPolygon" else ([] if footprint.is_empty else [footprint])
    diagnostics = {
        "method": "bounding_prism_minus_projection_of_absent_volume",
        "uses_sampled_sections_for_acceptance": False,
        "whole_thickness_evaluated": True,
        "area_mm2": footprint.area, "components": len(parts),
        "holes": sum(len(p.interiors) for p in parts),
        "sample_heights_mm": heights,
        "sampled_intersection_area_mm2": naive.area,
        "unsafe_sampled_area_mm2": naive.difference(footprint).area,
        "block_volume_mm3": local.volume(), "slab_volume_mm3": slab.volume(),
        "lower_remainder_volume_mm3": lower.volume(), "upper_remainder_volume_mm3": upper.volume(),
        "partition_error_mm3": volume_error,
        "arithmetic": "float64 / Manifold CSG / GEOS polygon union; numerical geometry",
        "manifold_tolerance_mm": local.get_tolerance(),
        "samples_note": "Display only; a topmost coplanar face is excluded by the slicing backend"
    }
    return SlabGeometry(spec, local, slab, footprint, heights, samples, naive, diagnostics)


def footprint_geojson(slab: SlabGeometry) -> dict:
    return {"type": "Feature", "geometry": mapping(slab.footprint),
            "properties": {"coordinate_system": "slab_local_xy", "units": "mm",
                           "slab": slab.spec.to_dict(), **slab.diagnostics}}
