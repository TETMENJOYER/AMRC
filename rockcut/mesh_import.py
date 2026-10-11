"""Explicit mesh import and planar lower closure; no convex hull or remeshing."""
import base64
import hashlib
import json
import zlib

import numpy as np
import shapely
from shapely.geometry import Polygon
import trimesh


def exact_weld(mesh):
    vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    faces = inverse[mesh.faces]
    used, inverse = np.unique(faces, return_inverse=True)
    return trimesh.Trimesh(vertices=vertices[used], faces=inverse.reshape(-1, 3), process=False)


def mesh_payload(mesh):
    raw = json.dumps({'vertices': mesh.vertices.tolist(), 'faces': mesh.faces.tolist()},
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    return {'type': 'mesh', 'encoding': 'zlib-json-base64',
            'sha256': hashlib.sha256(raw).hexdigest(),
            'data': base64.b64encode(zlib.compress(raw)).decode('ascii')}


def decode_mesh(base):
    if 'encoding' in base:
        if base['encoding'] != 'zlib-json-base64':
            raise ValueError('Unsupported mesh encoding')
        raw = zlib.decompress(base64.b64decode(base['data'], validate=True))
        if hashlib.sha256(raw).hexdigest() != base['sha256']:
            raise ValueError('Mesh payload checksum mismatch')
        base = json.loads(raw)
    vertices = np.asarray(base['vertices'], dtype=np.float64)
    faces_raw = np.asarray(base['faces'])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError('Mesh requires finite XYZ vertices')
    if (faces_raw.ndim != 2 or faces_raw.shape[1] != 3 or len(faces_raw) < 4
            or not np.issubdtype(faces_raw.dtype, np.integer)
            or faces_raw.min() < 0 or faces_raw.max() >= len(vertices)):
        raise ValueError('Mesh requires valid integer triangle indices')
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces_raw.astype(np.int64), process=False)
    if (not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0
            or mesh.euler_number != 2 or len(mesh.split(only_watertight=False)) != 1
            or not mesh.nondegenerate_faces().all()
            or len(np.unique(np.sort(mesh.faces, axis=1), axis=0)) != len(mesh.faces)):
        raise ValueError('Mesh must be a closed, consistently oriented single solid without cavities or tunnels')
    return mesh


def boundary_loops(mesh):
    edges, counts = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Nonmanifold edges in input surface')
    adjacency = {}
    for a, b in edges[counts == 1]:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError('Boundary is not a collection of simple closed loops')
    remaining = set(adjacency)
    loops = []
    while remaining:
        start = min(remaining)
        loop, previous, node = [], None, start
        while True:
            if node not in remaining:
                if node != start:
                    raise ValueError('Invalid boundary cycle')
                break
            remaining.remove(node); loop.append(node)
            neighbors = adjacency[node]
            other = neighbors[0] if neighbors[0] != previous else neighbors[1]
            previous, node = node, other
        loops.append(loop)
    return loops


def close_lower_plane(mesh, axis=1, clearance=.001):
    """Clip above all existing open edges, then triangulate the new flat cap.

    Coordinates are unchanged above the plane. Each cut edge gets one cached
    intersection vertex, eliminating cracks without coordinate rounding.
    """
    if axis not in (0, 1, 2) or not np.isfinite(clearance) or clearance <= 0:
        raise ValueError('Invalid closure axis/clearance')
    mesh = exact_weld(mesh)
    if not mesh.is_winding_consistent or not mesh.nondegenerate_faces().all():
        raise ValueError('Input surface has invalid orientation or degenerate faces')
    original_loops = boundary_loops(mesh)
    if not original_loops:
        raise ValueError('No open lower boundary to close')
    ids = np.concatenate(original_loops)
    offset = float(mesh.vertices[ids, axis].max()+clearance)
    if offset >= mesh.vertices[:, axis].max():
        raise ValueError('Closure plane would remove the whole model')
    points = mesh.vertices.tolist()
    intersections = {}
    def crossing(a, b):
        if mesh.vertices[a, axis] == offset:
            return a
        if mesh.vertices[b, axis] == offset:
            return b
        edge = tuple(sorted((a, b)))
        if edge not in intersections:
            va, vb = mesh.vertices[list(edge)]
            point = va+(offset-va[axis])/(vb[axis]-va[axis])*(vb-va)
            point[axis] = offset
            intersections[edge] = len(points)
            points.append(point.tolist())
        return intersections[edge]
    clipped_faces = []
    for face in mesh.faces:
        inside = mesh.vertices[face, axis] >= offset
        if inside.all():
            clipped_faces.append(face.tolist()); continue
        if not inside.any():
            continue
        polygon = []
        for a, b in zip(face, np.roll(face, -1)):
            ia, ib = mesh.vertices[[a, b], axis] >= offset
            if ia:
                polygon.append(int(a))
            if ia != ib:
                polygon.append(crossing(int(a), int(b)))
        polygon = list(dict.fromkeys(polygon))
        clipped_faces.extend([polygon[0], polygon[j], polygon[j+1]] for j in range(1, len(polygon)-1))
    surface = exact_weld(trimesh.Trimesh(vertices=points, faces=clipped_faces, process=False))
    loops = boundary_loops(surface)
    if not loops:
        raise ValueError('No boundary after lower clipping')
    axes = [i for i in range(3) if i != axis]
    caps = []
    cap_regions = []
    for loop in loops:
        xy = surface.vertices[loop][:, axes]
        if not np.all(surface.vertices[loop, axis] == offset):
            raise ValueError('An open edge survived outside the closure plane')
        polygon = Polygon(xy)
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError('Invalid planar cap contour')
        if any(polygon.intersects(p) for p in cap_regions):
            raise ValueError('Nested or intersecting cap contours are not supported')
        cap_regions.append(polygon)
        lookup = {tuple(p): int(i) for p, i in zip(xy, loop)}
        triangles = list(shapely.constrained_delaunay_triangles(polygon).geoms)
        if shapely.union_all(triangles).symmetric_difference(polygon).area > polygon.area*1e-12:
            raise ValueError('Cap triangulation failed area coverage')
        for triangle in triangles:
            face = [lookup[tuple(p)] for p in np.asarray(triangle.exterior.coords)[:3]]
            va, vb, vc = surface.vertices[face]
            if np.cross(vb-va, vc-va)[axis] > 0:
                face[1], face[2] = face[2], face[1]
            caps.append(face)
    closed = trimesh.Trimesh(vertices=surface.vertices,
                           faces=np.vstack((surface.faces, caps)), process=False)
    decode_mesh({'vertices': closed.vertices.tolist(), 'faces': closed.faces.tolist()})
    return closed, {'axis': axis, 'plane_offset_glb_units': offset,
                    'clearance_glb_units': clearance, 'original_boundary_loops': len(original_loops),
                    'cap_loops': len(loops), 'cap_triangles': len(caps),
                    'cap_area_glb_units_squared': sum(p.area for p in cap_regions),
                    'removed_vertical_extent_glb_units': offset-mesh.vertices[:, axis].min(),
                    'retained_side_triangles': len(surface.faces),
                    'operation': 'clip lower open surface and add planar constrained-triangulated cap',
                    'cap_is_measured_surface': False, 'convex_hull_used': False,
                    'surface_simplification_used': False}
