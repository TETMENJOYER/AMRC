from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import shapely
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, MultiPolygon, mapping, shape


def triangulate_region(region) -> np.ndarray:
    """Constrained triangles preserve concave edges, holes and all components."""
    if region.is_empty:
        return np.empty((0, 3, 2))
    triangles = list(shapely.constrained_delaunay_triangles(region).geoms)
    rebuilt = shapely.union_all(triangles)
    if rebuilt.symmetric_difference(region).area > 1e-10 * max(1., region.area):
        raise ValueError("Constrained triangulation failed the area coverage check")
    return np.array([np.asarray(t.exterior.coords)[:-1] for t in triangles], dtype=float).reshape(-1, 3, 2)


@dataclass
class Problem:
    vertices: np.ndarray
    short_side: float
    name: str = "slab"
    units: str = "mm"
    region: object | None = None

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64).copy()
        self.short_side = float(self.short_side)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError("vertices must have shape (M, 2)")
        if len(self.vertices) > 1 and np.array_equal(self.vertices[0], self.vertices[-1]):
            self.vertices = self.vertices[:-1]
        if len(self.vertices) < 3 or not np.isfinite(self.vertices).all():
            raise ValueError("A polygon requires at least three finite vertices")
        if not np.isfinite(self.short_side) or self.short_side <= 0:
            raise ValueError("short_side must be finite and positive")
        if len(np.unique(self.vertices, axis=0)) != len(self.vertices):
            raise ValueError("Repeated polygon vertices are not allowed")
        physical = Polygon(self.vertices) if self.region is None else self.region
        if isinstance(physical, dict):
            physical = shape(physical)
        if (physical.geom_type not in ("Polygon", "MultiPolygon") or physical.is_empty
                or not physical.is_valid or physical.area <= 0
                or not np.isfinite(shapely.get_coordinates(physical)).all()):
            raise ValueError("The region must be a valid nonempty Polygon or MultiPolygon")
        components = list(physical.geoms) if physical.geom_type == "MultiPolygon" else [physical]
        all_vertices = np.vstack([np.asarray(p.exterior.coords)[:-1] for p in components])
        self.origin = all_vertices.mean(axis=0)
        v = (self.vertices - self.origin) / self.short_side
        p = affine_transform(physical, [1/self.short_side, 0, 0, 1/self.short_side,
                                        -self.origin[0]/self.short_side, -self.origin[1]/self.short_side])
        self.normalization_discarded_components = []
        self.search_physical_region = physical
        if self.region is not None and not p.is_valid:
            # GEOS booleans can leave separate strips thinner than one float ULP.
            # Rescaling can collapse such a strip. It cannot contain any product:
            # every rotated 3:1 rectangle spans >= short_side on both XY axes.
            # Keep the FULL physical geometry for serialization/final acceptance;
            # omit only unrepresentable, provably unusable search components.
            retained=[]; physical_retained=[]
            for component in components:
                normalized=shapely.transform(component, lambda xy:(xy-self.origin)/self.short_side)
                if not normalized.is_valid or normalized.area<=0:
                    xmin,ymin,xmax,ymax=component.bounds
                    if min(xmax-xmin,ymax-ymin)>=self.short_side:
                        raise ValueError("A product-sized component became invalid during normalization")
                    self.normalization_discarded_components.append({'area':component.area,'bounds':component.bounds})
                else:
                    retained.append(normalized)
                    physical_retained.append(component)
            p=retained[0] if len(retained)==1 else MultiPolygon(retained)
            self.search_physical_region=(physical_retained[0] if len(physical_retained)==1
                                         else MultiPolygon(physical_retained))
        if not p.is_valid or p.area <= 0:
            raise ValueError("The polygon must be valid and have positive area")
        self.is_convex = p.equals(p.convex_hull)
        if physical.geom_type == "Polygon" and not physical.exterior.is_ccw:
            self.vertices = self.vertices[::-1].copy()
            v = v[::-1].copy()
        self.physical_region = physical
        # Retain the original arithmetic on convex, vertex-only inputs.
        self.polygon = Polygon(v) if self.region is None else p
        self.normalized_vertices = (all_vertices - self.origin) / self.short_side
        if self.is_convex:
            self.normalized_vertices = v
        hull_vertices = v if self.is_convex else np.asarray(p.convex_hull.exterior.coords)[:-1]
        if not Polygon(hull_vertices).exterior.is_ccw:
            hull_vertices = hull_vertices[::-1]
        edges = np.roll(hull_vertices, -1, axis=0) - hull_vertices
        self.normals = np.column_stack((edges[:, 1], -edges[:, 0]))
        self.normals /= np.linalg.norm(self.normals, axis=1, keepdims=True)
        self.offsets = np.einsum("mi,mi->m", self.normals, hull_vertices)
        self.obstacle_triangles = triangulate_region(self.polygon.convex_hull.difference(self.polygon))
        self._sampling_triangles = None

    @property
    def sampling_triangles(self):
        if self._sampling_triangles is None:
            self._sampling_triangles = triangulate_region(self.polygon)
        return self._sampling_triangles

    @classmethod
    def from_region(cls, region, short_side: float, name="slab", units="mm") -> "Problem":
        if isinstance(region, dict):
            region = shape(region.get("geometry", region))
        if region.is_empty or region.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError("The region must have positive polygonal area")
        first = max(region.geoms,key=lambda p:p.area) if region.geom_type == "MultiPolygon" else region
        return cls(np.asarray(first.exterior.coords)[:-1], short_side, name, units, region)

    @property
    def area(self) -> float:
        return float(self.physical_region.area)

    @property
    def product_area(self) -> float:
        return 3 * self.short_side**2

    @property
    def area_upper_bound(self) -> int:
        # A tiny upward allowance avoids an invalid upper bound at an integer
        # caused by floating-point polygon area arithmetic. It can only loosen it.
        ratio = self.area / self.product_area
        return int(np.floor(ratio + 1e-12 * max(1.0, ratio)))

    def denormalize(self, state: np.ndarray) -> np.ndarray:
        result = np.asarray(state, dtype=float).reshape(-1, 3).copy()
        result[:, :2] = result[:, :2] * self.short_side + self.origin
        result[:, 2] %= np.pi
        return result

    def normalize(self, state: np.ndarray) -> np.ndarray:
        result = np.asarray(state, dtype=float).reshape(-1, 3).copy()
        result[:, :2] = (result[:, :2] - self.origin) / self.short_side
        return result

    def to_dict(self) -> dict:
        data = {"name": self.name, "units": self.units,
                "short_side": self.short_side, "aspect_ratio": 3, "kerf": 0,
                "allowance": 0, "touching_allowed": True}
        if self.region is not None:
            data["geometry"] = mapping(self.physical_region)
        else:
            data["vertices"] = self.vertices.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Problem":
        if data.get("aspect_ratio", 3) != 3:
            raise ValueError("This version supports aspect_ratio = 3 only")
        if data.get("kerf", 0) != 0 or data.get("allowance", 0) != 0:
            raise ValueError("Nonzero kerf/allowance requires a production geometry model")
        if data.get("touching_allowed", True) is not True:
            raise ValueError("This version assumes touching is allowed")
        if "geometry" in data:
            if "vertices" in data:
                raise ValueError("Specify either vertices or geometry, not both")
            return cls.from_region(data["geometry"], data["short_side"], data.get("name", "slab"), data.get("units", "mm"))
        return cls(data["vertices"], data["short_side"], data.get("name", "slab"), data.get("units", "mm"))

    @classmethod
    def load(cls, path: str | Path) -> "Problem":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))
