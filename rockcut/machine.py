"""One stable rigid setup on a horizontal table, followed by vertical cuts.

Convex hulls here describe support contacts only; the stone stays nonconvex.
Stability is ideal rigid-body statics for homogeneous material, before cutting.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, Point
import trimesh

from .block import Frame, mesh_arrays, vector


def machine_policy(inputs):
    policy = inputs.get('machine')
    if policy is None:
        return None
    allowed = {'kind','support_mode','support_margin_mm','contact_tolerance_mm','max_supports','yaw_angles_degrees'}
    if not isinstance(policy, dict) or set(policy)-allowed:
        raise ValueError('Invalid machine policy fields')
    if policy.get('kind') != 'vertical_fixed_setup':
        raise ValueError('Machine requires vertical_fixed_setup')
    if policy.get('support_mode', 'fixed') not in ('fixed','stable_faces'):
        raise ValueError('Support mode must be fixed or stable_faces')
    for key, default in (('support_margin_mm',0.),('contact_tolerance_mm',1e-6)):
        value = policy.get(key, default)
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid {key}')
    if policy.get('contact_tolerance_mm',1e-6) <= 0:
        raise ValueError('Contact tolerance must be positive')
    count = policy.get('max_supports',3)
    if isinstance(count,bool) or not isinstance(count,int) or count < 1:
        raise ValueError('max_supports must be a positive integer')
    angles = policy.get('yaw_angles_degrees')
    if angles is not None and (not isinstance(angles,list) or not angles
                              or any(isinstance(a,bool) or not isinstance(a,(int,float)) or not math.isfinite(a) for a in angles)):
        raise ValueError('Yaw angles must be a nonempty finite list')
    if 'directions' in inputs:
        raise ValueError('Use machine yaw angles instead of unconstrained directions')
    return policy


@dataclass
class SourceMass:
    vertices: np.ndarray
    center: np.ndarray
    tolerance: float

    @classmethod
    def from_block(cls, block, policy):
        vertices, faces = mesh_arrays(block.solid)
        center = np.asarray(trimesh.Trimesh(vertices=vertices,faces=faces,process=False).center_mass)
        scale = max(1.,float(np.ptp(vertices,axis=0).max()))
        tolerance = float(policy.get('contact_tolerance_mm',scale*1e-9))
        if tolerance > scale * 1e-7:
            raise ValueError('Contact tolerance is only for numerical roundoff, not roughness')
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('Invalid volume center of mass')
        return cls(vertices,center,tolerance)


def rotation_for_support(outward, yaw_degrees):
    outward = vector(outward,'support outward normal')
    length = np.linalg.norm(outward)
    if not np.isclose(length,1.,rtol=0,atol=1e-10):
        raise ValueError('Support normal must be unit length')
    base = Frame.from_dict({'normal':-outward}).basis.T
    yaw = math.radians(float(yaw_degrees))
    c,s = math.cos(yaw),math.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]]) @ base


def contact_audit(mass, frame, policy):
    positions = frame.to_world(mass.vertices)
    lowest = float(positions[:,2].min())
    if abs(lowest) > mass.tolerance:
        raise ValueError('Block must touch the table at z=0 without penetrating it')
    contacts = positions[positions[:,2] <= lowest+mass.tolerance,:2]
    support = MultiPoint(contacts).convex_hull
    com = frame.to_world(mass.center)
    inside = support.geom_type == 'Polygon' and support.area > 0 and support.contains(Point(com[:2]))
    margin = float(Point(com[:2]).distance(support.boundary)) if inside else 0.
    required = float(policy.get('support_margin_mm',0))
    accepted = inside and margin > required + mass.tolerance
    return {'accepted':bool(accepted),'center_of_mass_source_mm':mass.center.tolist(),
            'center_of_mass_machine_mm':com.tolist(),'support_contact_count':len(contacts),
            'support_polygon_machine_xy_mm':(np.asarray(support.exterior.coords)[:-1].tolist()
                                            if support.geom_type=='Polygon' else []),
            'support_area_mm2':float(support.area),'stability_margin_mm':margin,
            'required_margin_mm':required,'contact_tolerance_mm':mass.tolerance,
            'minimum_z_mm':lowest,'scope':'initial homogeneous rigid block only; retaining bridges assumed'}


def make_setup(mass, policy, outward, yaw, support_id):
    rotation = rotation_for_support(outward,yaw)
    minimum_z = float((mass.vertices @ rotation.T)[:,2].min())
    frame = Frame([0,0,-minimum_z],rotation)
    audit = contact_audit(mass,frame,policy)
    if not audit['accepted']:
        raise ValueError('Initial block setup is unstable or below its required margin')
    return {'kind':'vertical_fixed_setup','source_to_machine':frame.to_dict(),
            'support_source_normal':np.asarray(outward).tolist(),'support_id':support_id,
            'yaw_degrees':float(yaw),'blade_normal_machine':[1.,0.,0.],
            'block_moves_during_series':False,'separation':'retaining_bridges_then_mechanical',
            'initial_stability':audit}


def setup_candidates(block, inputs, count):
    policy = machine_policy(inputs)
    mass = SourceMass.from_block(block,policy)
    if policy.get('support_mode','fixed') == 'fixed':
        bases = [make_setup(mass,policy,np.array([0.,0.,-1.]),0.,'fixed_base')]
        discovered = 1
    else:
        hull = ConvexHull(mass.vertices)
        # Coplanar hull triangles define the same table placement.
        planes = np.unique(np.round(hull.equations,10),axis=0)
        bases = []
        for i, plane in enumerate(planes):
            normal = plane[:3] / np.linalg.norm(plane[:3])
            try:
                bases.append(make_setup(mass,policy,normal,0.,f'support_{i}'))
            except ValueError as exc:
                if 'unstable' not in str(exc):
                    raise
        bases.sort(key=lambda s:(s['initial_stability']['support_area_mm2'],
                                 s['initial_stability']['stability_margin_mm']),reverse=True)
        discovered = len(bases)
        bases = bases[:policy.get('max_supports',3)]
        if not bases:
            raise ValueError('No stable initial support faces found')
    specified = policy.get('yaw_angles_degrees')
    angles = (specified if specified is not None else
              np.linspace(0,360,max(1,math.ceil(count/len(bases))),endpoint=False))
    candidates = []
    for angle in angles:
        for base in bases:
            setup = make_setup(mass,policy,base['support_source_normal'],angle,base['support_id'])
            frame = Frame.from_dict(setup['source_to_machine'])
            normal = frame.basis.T @ np.array([1.,0.,0.])
            # Duplicate normals on different support faces remain different setups.
            if not any(s['support_id']==setup['support_id'] and np.linalg.norm(n-normal)<1e-9
                       for n,s in candidates):
                candidates.append((normal,setup))
            if specified is None and len(candidates)>=count:
                break
        if specified is None and len(candidates)>=count:
            break
    return candidates, {'stable_supports_found':discovered,'stable_supports_retained':len(bases),
                        'setups_generated':len(candidates),'selection':'largest support areas first'}


def verify_machine_setup(block, inputs, setup, cut_normal):
    policy = machine_policy(inputs)
    if policy is None:
        if setup is not None:
            raise ValueError('Machine setup without machine policy')
        return None
    if not isinstance(setup,dict) or setup.get('kind')!='vertical_fixed_setup':
        raise ValueError('Missing fixed machine setup')
    if (setup.get('block_moves_during_series') is not False
            or setup.get('separation')!='retaining_bridges_then_mechanical'):
        raise ValueError('Machine requires one setup and retaining bridges throughout the series')
    frame = Frame.from_dict(setup['source_to_machine'])
    yaw = float(setup['yaw_degrees'])
    if not math.isfinite(yaw):
        raise ValueError('Invalid yaw')
    outward = vector(setup['support_source_normal'],'support normal')
    if policy.get('support_mode','fixed')=='fixed' and not np.allclose(outward,[0,0,-1],rtol=0,atol=1e-10):
        raise ValueError('Fixed base policy prohibits tipping the block')
    if not np.allclose(frame.basis,rotation_for_support(outward,yaw),rtol=0,atol=1e-10):
        raise ValueError('Saved support/yaw differ from machine rotation')
    if policy.get('yaw_angles_degrees') is not None and not any(
            abs((yaw-float(a)+180)%360-180)<1e-8 for a in policy['yaw_angles_degrees']):
        raise ValueError('Yaw is outside the allowed input angles')
    blade = vector(setup['blade_normal_machine'],'blade normal')
    normal = vector(cut_normal,'cut normal')
    if (not np.allclose(blade,[1,0,0],rtol=0,atol=1e-12)
            or not np.allclose(frame.basis @ normal,blade,rtol=0,atol=1e-10)):
        raise ValueError('Cut must be vertical and aligned with the fixed machine blade')
    audit = contact_audit(SourceMass.from_block(block,policy),frame,policy)
    if not audit['accepted']:
        raise ValueError('Initial setup failed independent stability verification')
    return {**audit,'vertical_cuts':True,'single_fixed_setup':True,
            'bridges_strength_verified':False,'bridge_breakage_allowance_mm':None}
