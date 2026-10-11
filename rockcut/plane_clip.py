"""Construct parallel cut caps on their prescribed coordinate plane.

Only newly generated vertices of axis-aligned cap faces are corrected for
floating-point roundoff. Original source vertices are never snapped.
"""
import numpy as np
import manifold3d as m3d

from .block import checked, mesh_arrays


def clip_z(solid, offset, keep_above):
    before,_=mesh_arrays(solid)
    normal=(0,0,1) if keep_above else (0,0,-1)
    distance=offset if keep_above else -offset
    clipped=checked(solid.trim_by_plane(normal,distance),'parallel plane clip')
    note={'corrected_cap_vertices':0,'maximum_roundoff_correction_mm':0.}
    if clipped.is_empty():
        return clipped,note
    vertices,faces=mesh_arrays(clipped)
    scale=max(1.,abs(float(offset)),float(np.abs(vertices).max()))
    roundoff=32*np.finfo(float).eps*scale
    triangles=vertices[faces]
    near=np.all(np.abs(triangles[:,:,2]-offset)<=roundoff,axis=1)
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    planar=np.linalg.norm(cross[:,:2],axis=1)<=32*np.finfo(float).eps*np.linalg.norm(cross,axis=1)
    indices=np.unique(faces[near & planar])
    indices=indices[vertices[indices,2]!=offset]
    if not len(indices):
        return clipped,note
    original=set(map(tuple,before))
    changes=[int(i) for i in indices if tuple(vertices[i]) not in original]
    if not changes:
        return clipped,note
    note={'corrected_cap_vertices':len(changes),
          'maximum_roundoff_correction_mm':float(np.max(abs(vertices[changes,2]-offset)))}
    vertices=vertices.copy()
    vertices[changes,2]=offset
    rebuilt=m3d.Manifold(m3d.Mesh64(np.ascontiguousarray(vertices),np.ascontiguousarray(faces,dtype=np.uint64)))
    rebuilt=checked(rebuilt,'canonical parallel cap',allow_empty=False)
    if abs(rebuilt.volume()-clipped.volume())>1e-9*max(1.,clipped.volume()):
        raise ValueError('Cap roundoff correction changed cut volume')
    return rebuilt,note
