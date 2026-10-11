"""Prepare the supplied real scan with an explicitly reconstructed planar bottom."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

from rockcut.block import Block
from rockcut.mesh_import import close_lower_plane, mesh_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root/'outputs/granite-closed'
    folder.mkdir(parents=True, exist_ok=True)
    original = root/'examples/models/granite_block_original.glb'
    original.parent.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    raw = source.read_bytes()
    if original.exists() and original.read_bytes() != raw:
        raise FileExistsError('Preserved source differs; choose a new source filename')
    if not original.exists():
        shutil.copyfile(source, original)
    length, _ = struct.unpack_from('<II', raw, 12)
    asset = json.loads(raw[20:20+length])['asset']
    scene = trimesh.load(source, force='scene', process=False)
    if any(not isinstance(g, trimesh.Trimesh) for g in scene.geometry.values()):
        raise ValueError('Non-triangle scene geometry')
    closed, operation = close_lower_plane(scene.to_geometry(), axis=1, clearance=.001)
    # Use the GLB metric convention as an explicitly unconfirmed physical scale.
    vertices = closed.vertices[:, [0, 2, 1]].copy()*1000
    vertices[:, 1] *= -1
    translation = -vertices.min(0)
    vertices += translation
    prepared = trimesh.Trimesh(vertices=vertices, faces=closed.faces, process=False)
    provenance = {'source_sha256': hashlib.sha256(raw).hexdigest(), 'source_asset': asset,
                  'source_file': 'examples/models/granite_block_original.glb',
                  'scale_mm_per_glb_unit': 1000., 'physical_scale_confirmed': False,
                  'world_transform': 'mm XYZ = 1000*(GLB X,-GLB Z,GLB Y)+translation_mm',
                  'translation_mm': translation.tolist(), 'closure': operation,
                  'material_model': 'assumed homogeneous internally solid stone',
                  'self_intersection_audit': 'no dedicated all-triangle intersection audit; topology and CSG tests below'}
    block_data = {'schema_version': 1, 'units': 'mm', 'name': 'granite_scan_planar_bottom',
                  'base': mesh_payload(prepared), 'provenance': provenance}
    block = Block.from_dict(block_data)
    diagnostics = block.diagnostics()
    audit = {'accepted': True, 'provenance': provenance, 'closed_mesh': diagnostics,
             'cap_planarity_error_mm': float(np.abs(prepared.vertices[
                 np.unique(prepared.faces[-operation['cap_triangles']:]), 2]).max()),
             'original_file_unchanged': hashlib.sha256(source.read_bytes()).hexdigest()==provenance['source_sha256']}
    inputs = {'schema_version': 1, 'kind': 'block_cutting_problem', 'block': block_data,
              'product': {'short_side': 100, 'aspect_ratio': 3, 'thickness': 30, 'kerf': 0, 'allowance': 0},
              'cut_policy': {'kind': 'parallel_support_start', 'initial_trim_allowed': False, 'kerf': 0},
              'directions': [[0, 0, 1], [1, 0, 0], [0, 1, 0]]}
    path = root/'examples/granite_block_cutting.json'
    path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding='utf-8')
    (folder/'closure-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    prepared.export(folder/'granite_closed_mm.obj')
    glb = prepared.copy(); glb.vertices /= 1000
    glb.export(folder/'granite_closed.glb')
    fig = plt.figure(figsize=(12, 6.5))
    cap_count = operation['cap_triangles']
    triangles = prepared.vertices[prepared.faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    colors = (.55+.45*np.abs(normals @ (np.array([1., -1., 2.])/np.sqrt(6))))[:, None]*np.array([.8, .77, .7])
    colors[-cap_count:]=np.array([.32,.61,.77])
    for index, elevation in enumerate((25, -30), 1):
        ax = fig.add_subplot(1, 2, index, projection='3d')
        angle=np.radians(-55); elev=np.radians(elevation)
        view=np.array([np.cos(elev)*np.cos(angle),np.cos(elev)*np.sin(angle),np.sin(elev)])
        visible=normals @ view > 0
        ax.add_collection3d(Poly3DCollection(triangles[visible], facecolor=colors[visible], edgecolor='none'))
        ax.set_proj_type('ortho')
        low, high = prepared.bounds
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
        ax.set_box_aspect(high-low); ax.view_init(elev=elevation, azim=-55)
        ax.set_xlabel('X, мм'); ax.set_ylabel('Y, мм'); ax.set_zlabel('Z, мм')
        ax.set_title('Вид сверху' if index == 1 else 'Вид снизу', pad=25)
    fig.suptitle('Замкнутый гранитный блок\nСиним показано реконструированное плоское основание', y=.98)
    fig.subplots_adjust(left=.02, right=.95, top=.80, bottom=.08, wspace=.18)
    fig.savefig(folder/'closed-preview.png', dpi=150)
    plt.close(fig)
    print(json.dumps({'input': str(path), 'audit': audit}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
