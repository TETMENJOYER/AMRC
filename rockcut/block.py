"""Closed synthetic/imported solids and rigid coordinate frames, in millimetres."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np
import manifold3d as m3d
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon


def vector(value, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite vector of length 3")
    return a


def checked(solid: m3d.Manifold, name: str, allow_empty: bool = True) -> m3d.Manifold:
    if solid.status() != m3d.Error.NoError:
        raise ValueError(f"{name}: Manifold status {solid.status()}")
    volume = solid.volume()
    if not np.isfinite(volume) or volume < 0:
        raise ValueError(f"{name}: invalid volume")
    if not allow_empty and (solid.is_empty() or volume <= 0):
        raise ValueError(f"{name}: positive volume required")
    return solid


def mesh_arrays(solid: m3d.Manifold) -> tuple[np.ndarray, np.ndarray]:
    mesh = solid.to_mesh64()
    return np.asarray(mesh.vert_properties[:, :3], dtype=np.float64), np.asarray(mesh.tri_verts, dtype=np.int64)


@dataclass
class Frame:
    origin: np.ndarray
    basis: np.ndarray  # columns u, v, n; world = origin + basis @ local

    def __post_init__(self):
        self.origin = vector(self.origin, "origin").copy()
        self.basis = np.asarray(self.basis, dtype=np.float64).copy()
        if (self.basis.shape != (3, 3) or not np.isfinite(self.basis).all()
                or not np.allclose(self.basis.T @ self.basis, np.eye(3), rtol=0, atol=1e-12)
                or not np.isclose(np.linalg.det(self.basis), 1, rtol=0, atol=1e-12)):
            raise ValueError("basis must be an orthonormal right-handed 3x3 matrix")

    @classmethod
    def from_dict(cls, data: dict) -> "Frame":
        if "basis" in data:
            return cls(data.get("origin", [0, 0, 0]), data["basis"])
        n = vector(data.get("normal", [0, 0, 1]), "normal")
        length = np.linalg.norm(n)
        if length < 1e-15:
            raise ValueError("normal must be nonzero")
        n = n / length
        hint = vector(data.get("x_direction", [1, 0, 0] if abs(n[0]) < .9 else [0, 1, 0]), "x_direction")
        u = hint - n * np.dot(n, hint)
        if np.linalg.norm(u) < 1e-12:
            raise ValueError("x_direction must not be parallel to normal")
        u /= np.linalg.norm(u)
        return cls(data.get("origin", [0, 0, 0]), np.column_stack((u, np.cross(n, u), n)))

    def to_dict(self) -> dict:
        return {"origin": self.origin.tolist(), "basis": self.basis.tolist()}

    def to_world(self, points) -> np.ndarray:
        return np.asarray(points) @ self.basis.T + self.origin

    def to_local(self, points) -> np.ndarray:
        return (np.asarray(points) - self.origin) @ self.basis

    def world_solid(self, solid: m3d.Manifold) -> m3d.Manifold:
        return checked(solid.transform(np.column_stack((self.basis, self.origin))), "world transform")

    def local_solid(self, solid: m3d.Manifold) -> m3d.Manifold:
        return checked(solid.transform(np.column_stack((self.basis.T, -self.basis.T @ self.origin))), "local transform")


def box_solid(spec: dict) -> m3d.Manifold:
    if spec.get("type", "box") != "box":
        raise ValueError("Only box cutters are supported in stage 1")
    size = vector(spec["size"], "box size")
    center = vector(spec.get("center", size / 2), "box center")
    if np.any(size <= 0):
        raise ValueError("Box dimensions must be positive")
    angles = vector(spec.get("rotation_degrees", [0, 0, 0]), "box rotation")
    rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    solid = m3d.Manifold.cube(size, center=True).transform(np.column_stack((rotation, center)))
    return checked(solid, "box", allow_empty=False)


@dataclass
class Block:
    spec: dict
    solid: m3d.Manifold

    @classmethod
    def from_dict(cls, data: dict) -> "Block":
        """Rebuild a closed solid from a CSG recipe or an embedded validated mesh."""
        if data.get("schema_version") != 1 or data.get("units") != "mm":
            raise ValueError("Block requires schema_version = 1 and units = mm")
        base = data["base"]
        if base["type"] == "box":
            solid = box_solid(base)
        elif base["type"] == "prism":
            vertices = np.asarray(base["vertices"], dtype=np.float64)
            if vertices.ndim != 2 or vertices.shape[1] != 2 or not np.isfinite(vertices).all():
                raise ValueError("Invalid prism vertices")
            polygon = Polygon(vertices)
            if not polygon.is_valid or polygon.area <= 0:
                raise ValueError("The generator's prism base must be a valid simple polygon")
            if not polygon.exterior.is_ccw:
                vertices = vertices[::-1]
            height = float(base["height"])
            if not np.isfinite(height) or height <= 0:
                raise ValueError("Prism height must be positive")
            solid = m3d.CrossSection([vertices]).extrude(height)
            shear = np.asarray(base.get("top_shift", [0, 0]), dtype=float)
            if shear.shape != (2,) or not np.isfinite(shear).all():
                raise ValueError("Invalid prism top_shift")
            transform = np.eye(3, 4)
            transform[:2, 2] = shear / height
            solid = checked(solid.transform(transform), "prism", allow_empty=False)
        elif base["type"] == "mesh":
            from .mesh_import import decode_mesh
            mesh = decode_mesh(base)
            solid = checked(m3d.Manifold(m3d.Mesh64(
                np.ascontiguousarray(mesh.vertices, dtype=np.float64),
                np.ascontiguousarray(mesh.faces, dtype=np.uint64))), "imported mesh", allow_empty=False)
            if abs(solid.volume()-mesh.volume) > 1e-9*max(1., mesh.volume):
                raise ValueError("Imported mesh volume changed during conversion")
        else:
            raise ValueError("Unsupported base type")
        for cutter in data.get("subtract", []):
            solid = checked(solid - box_solid(cutter), "CSG subtraction")
        if "pose" in data:
            solid = Frame.from_dict(data["pose"]).world_solid(solid)
        block = cls(copy.deepcopy(data), checked(solid, "block", allow_empty=False))
        block.diagnostics()
        return block

    @property
    def name(self) -> str:
        return str(self.spec.get("name", "synthetic_block"))

    def diagnostics(self) -> dict:
        import trimesh
        vertices, faces = mesh_arrays(self.solid)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        volume = self.solid.volume()
        independent_volume = float(mesh.volume)
        error = abs(volume - independent_volume)
        if not mesh.is_watertight or not mesh.is_winding_consistent or error > 1e-9 * max(1, volume):
            raise ValueError("Generated mesh failed closure/orientation/volume checks")
        shells = mesh.split(only_watertight=False)
        if len(shells) != 1 or mesh.euler_number != 2:
            raise ValueError("Block must be one solid without enclosed cavities or through holes")
        return {"name": self.name, "volume_mm3": volume, "mesh_volume_mm3": independent_volume,
                "material_model": "homogeneous_solid", "boundary_shells": len(shells),
                "euler_number": int(mesh.euler_number),
                "volume_disagreement_mm3": error, "watertight": bool(mesh.is_watertight),
                "winding_consistent": bool(mesh.is_winding_consistent), "vertices": len(vertices),
                "triangles": len(faces), "bounds_mm": list(self.solid.bounding_box()),
                "manifold_tolerance_mm": self.solid.get_tolerance(),
                "source": ("validated embedded mesh; source preprocessing recorded in provenance"
                           if self.spec['base']['type']=='mesh' else "validated synthetic CSG recipe")}
