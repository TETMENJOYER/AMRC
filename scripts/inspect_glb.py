"""Read-only GLB geometry audit. No hole filling or geometric approximation."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import manifold3d as m3d
import numpy as np
import trimesh


def inspect(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    blob = source.read_bytes()
    magic, version, length = struct.unpack_from('<III', blob)
    if magic != 0x46546c67 or version != 2 or length != len(blob):
        raise ValueError('Invalid GLB v2 container')
    chunk_length, chunk_type = struct.unpack_from('<II', blob, 12)
    if chunk_type != 0x4e4f534a:
        raise ValueError('Missing JSON chunk')
    document = json.loads(blob[20:20+chunk_length])
    scene = trimesh.load(source, force='scene', process=False)
    if any(not isinstance(g, trimesh.Trimesh) for g in scene.geometry.values()):
        raise ValueError('Scene contains non-triangle geometry')
    mesh = scene.to_geometry()  # Apply all scene transforms, retain all objects.
    if not np.isfinite(mesh.vertices).all():
        raise ValueError('Non-finite coordinates')
    vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    faces = inverse[mesh.faces]
    welded = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    edges, counts = np.unique(np.sort(welded.edges, axis=1), axis=0, return_counts=True)
    boundary = edges[counts == 1]
    adjacency = {}
    for a, b in boundary:
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))
    unseen = set(adjacency)
    loops = []
    while unseen:
        pending = [unseen.pop()]
        component = []
        while pending:
            node = pending.pop()
            component.append(node)
            for other in adjacency[node]:
                if other in unseen:
                    unseen.remove(other)
                    pending.append(other)
        points = vertices[component]
        _, _, basis = np.linalg.svd(points-points.mean(0), full_matrices=False)
        distances = np.abs((points-points.mean(0)) @ basis[-1])
        loops.append({'vertices': len(component),
                      'closed_simple_loop': all(len(adjacency[n]) == 2 for n in component),
                      'bounds_glb_units': np.stack((points.min(0), points.max(0))).tolist(),
                      'best_fit_plane_normal': basis[-1].tolist(),
                      'plane_max_deviation_glb_units': float(distances.max()),
                      'plane_rms_deviation_glb_units': float(np.sqrt(np.mean(distances**2)))})
    solid = m3d.Manifold(m3d.Mesh64(np.ascontiguousarray(vertices),
                                  np.ascontiguousarray(faces, dtype=np.uint64)))
    accepted = (welded.is_watertight and welded.is_winding_consistent
                and welded.euler_number == 2 and solid.status() == m3d.Error.NoError
                and not solid.is_empty() and solid.volume() > 0)
    audit = {
        'source_file': str(source), 'source_bytes': len(blob),
        'sha256': hashlib.sha256(blob).hexdigest(), 'asset': document.get('asset'),
        'scene_transforms_applied': True, 'geometry_objects': len(scene.geometry),
        'triangles': len(faces), 'vertices_before_welding': len(mesh.vertices),
        'vertices_after_exact_welding': len(vertices),
        'welding_rule': 'identical XYZ only; no displacement, rounding, filling or deletion',
        'bounds_glb_units': welded.bounds.tolist(), 'extents_glb_units': welded.extents.tolist(),
        'nominal_extents_mm_if_gltf_meters': (welded.extents*1000).tolist(),
        'physical_scale_confirmed_by_measurement': False,
        'watertight': bool(welded.is_watertight),
        'winding_consistent': bool(welded.is_winding_consistent),
        'surface_components': len(welded.split(only_watertight=False)),
        'euler_number': int(welded.euler_number), 'boundary_edges': len(boundary),
        'nonmanifold_edges_more_than_two_faces': int(np.sum(counts > 2)),
        'degenerate_faces': int(np.sum(~welded.nondegenerate_faces())),
        'duplicate_faces': len(faces)-len(np.unique(np.sort(faces, axis=1), axis=0)),
        'boundary_components': loops, 'manifold_status': str(solid.status()),
        'self_intersections': 'not independently checked; closure already fails',
        'accepted_as_cutting_solid': bool(accepted),
        'volume': None if not accepted else float(solid.volume()),
        'volume_note': 'An open surface does not define the required solid volume.'}
    (output/'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    # Render without textures so actual surface geometry and open edges are visible.
    display = vertices[:, [0, 2, 1]].copy()
    display[:, 1] *= -1  # GLB Y up -> display Z up, preserving handedness.
    triangles = display[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    illumination = .55 + .45*np.abs(normals @ (np.array([1., -1., 2.])/np.sqrt(6)))
    colors = illumination[:, None]*np.array([.80, .77, .70])[None, :]
    fig = plt.figure(figsize=(12, 6.5))
    low, high = display.min(0), display.max(0)
    center, radius = (low+high)/2, max(high-low)*.55
    for index, (elevation, azimuth) in enumerate(((25, -55), (-30, -55)), 1):
        ax = fig.add_subplot(1, 2, index, projection='3d')
        ax.add_collection3d(Poly3DCollection(triangles, facecolor=colors,
            edgecolor='none', linewidth=0, alpha=1, rasterized=True))
        ax.add_collection3d(Line3DCollection(display[boundary], colors='#cf1935', linewidths=1.5))
        ax.set(xlim=(center[0]-radius, center[0]+radius),
               ylim=(center[1]-radius, center[1]+radius),
               zlim=(center[2]-radius, center[2]+radius))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_title('Вид сверху' if index == 1 else 'Вид снизу', pad=25)
        ax.set_xlabel('GLB X'); ax.set_ylabel('-GLB Z'); ax.set_zlabel('GLB Y')
    fig.suptitle(f'Гранитный блок: {len(faces):,} треугольников\nКрасным показан край незамкнутой поверхности', y=.98)
    fig.subplots_adjust(left=.02, right=.95, top=.80, bottom=.09, wspace=.18)
    fig.savefig(output/'boundary-preview.png', dpi=150)
    plt.close(fig)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('--output', default='outputs/granite-inspection')
    args = parser.parse_args()
    inspect(args.source, args.output)
