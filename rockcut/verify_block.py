"""Rebuild all cuts and products from the source; ignore saved packing geometry."""
import numpy as np

from .block import Block, Frame, checked
from .cut_plan import parallel_plan
from .packing3d import verify_layout
from .verify3d import product_geometry
from .machine import machine_policy, verify_machine_setup


def block_inputs(data):
    if data.get("schema_version") != 1 or data.get("kind") != "block_cutting_problem":
        raise ValueError("Expected schema_version=1, kind=block_cutting_problem")
    p = data['product']
    if p.get('aspect_ratio', 3) != 3 or any(p.get(k, 0) != 0 for k in ('kerf', 'allowance')):
        raise ValueError("Product requires aspect ratio 3 and zero kerf/allowance")
    side, thickness = float(p['short_side']), float(p['thickness'])
    if not np.isfinite([side, thickness]).all() or min(side, thickness) <= 0:
        raise ValueError("Product dimensions must be positive")
    policy = data.get('cut_policy', {})
    if (policy.get('kind', 'parallel_support_start') != 'parallel_support_start'
            or policy.get('initial_trim_allowed', False) is not False
            or policy.get('kerf', 0) != 0):
        raise ValueError("Only parallel support start, no initial trim, zero first-stage kerf")
    machine_policy(data)
    return Block.from_dict(data['block']), side, thickness


def verify_block_result(result):
    if result.get('kind') != 'block_cutting_result' or result.get('angle_unit') != 'radians':
        raise ValueError("Expected block_cutting_result with radian angles")
    block, side, thickness = block_inputs(result['inputs'])
    saved = result['plan']
    machine_audit = verify_machine_setup(block,result['inputs'],result.get('machine_setup'),saved['normal'])
    plan = parallel_plan(block, saved['normal'], thickness)
    frame = Frame.from_dict(saved['frame'])
    if not np.allclose(frame.basis, plan.frame.basis, rtol=0, atol=1e-12) or not np.array_equal(frame.origin, plan.frame.origin):
        raise ValueError("Saved frame differs from the reconstructed cut frame")
    checks = []
    scale = max(1., block.solid.volume())
    vtol = 1e-9 * scale
    if len(result['slabs']) != len(plan.specs) or len(saved['cuts']) != len(plan.steps):
        raise ValueError("Missing or duplicate slab/cut records")
    for key, actual in (('support_lower_mm', plan.lower), ('support_upper_mm', plan.upper), ('thickness_mm', thickness)):
        if not np.isclose(saved[key], actual, rtol=0, atol=1e-9):
            raise ValueError("Saved cut support or thickness changed")
    if saved.get('initial_trim_allowed') is not False or saved.get('first_stage_kerf_mm') != 0:
        raise ValueError("Saved cut policy is incompatible")
    total = 0
    produced = 0.
    slab_unused = 0.
    cuts_ok = True
    cuts_ok = cuts_ok and saved.get('slab_count') == len(plan.specs)
    cuts_ok = cuts_ok and saved.get('cut_count') == sum(s['cut_performed'] for s in plan.steps)
    cuts_ok = cuts_ok and np.isclose(saved.get('remainder_volume_mm3',np.nan),plan.remainder.volume(),rtol=0,atol=vtol)
    for i, (record, spec, piece, step) in enumerate(zip(result['slabs'], plan.specs, plan.pieces, plan.steps)):
        if record['index'] != i or record.get('angle_unit') != 'radians':
            raise ValueError("Invalid slab order or angle unit")
        stored_spec = record['spec']
        if (not np.isclose(stored_spec['lower'], spec.lower, rtol=0, atol=1e-9)
                or not np.isclose(stored_spec['upper'], spec.upper, rtol=0, atol=1e-9)
                or not np.allclose(Frame.from_dict(stored_spec['frame']).basis, frame.basis, rtol=0, atol=1e-12)
                or not np.array_equal(Frame.from_dict(stored_spec['frame']).origin, frame.origin)):
            raise ValueError("Slab range/frame changed")
        stored_cut = saved['cuts'][i]
        for key in ('slab_index', 'kind', 'cut_performed'):
            cuts_ok = cuts_ok and stored_cut[key] == step[key]
        for key in ('plane_offset_mm', 'before_volume_mm3', 'slab_volume_mm3', 'after_volume_mm3'):
            tol = 1e-9 if key == 'plane_offset_mm' else vtol
            cuts_ok = cuts_ok and np.isclose(stored_cut[key], step[key], rtol=0, atol=tol)
        cuts_ok = cuts_ok and np.allclose(stored_cut['plane_normal_world'], step['plane_normal_world'], rtol=0, atol=1e-12)
        cuts_ok = cuts_ok and np.isclose(record.get('slab_volume_mm3',np.nan),piece.volume(),rtol=0,atol=vtol)
        state = np.asarray(record['state'], dtype=float)
        if state.size == 0:
            state = np.empty((0, 3))
        if state.ndim != 2 or state.shape[1] != 3 or record['n'] != len(state):
            raise ValueError("Invalid placements or slab count")
        audit = verify_layout(block, spec, side, state, slab_local=piece)
        in_piece = audit['in_separated_slab']
        pv = len(state) * 3 * side**2 * thickness
        unused = piece.volume() - pv
        checks.append({'index': i, 'accepted': audit['accepted'] and in_piece and unused >= -vtol,
                       'n': len(state), 'in_separated_slab': in_piece,
                       'slab_volume_mm3': piece.volume(), 'product_volume_mm3': pv,
                       'unused_volume_mm3': max(0., unused), **{'layout': audit}})
        total += len(state); produced += pv; slab_unused += max(0., unused)
    initial = block.solid.volume()
    remainder = plan.remainder.volume()
    error = abs(initial - produced - slab_unused - remainder)
    count_ok = result['n_total'] == total
    count_ok = count_ok and np.isclose(result.get('product_volume_mm3',np.nan),3*side**2*thickness,rtol=0,atol=vtol)
    count_ok = count_ok and np.isclose(result.get('yield_by_volume',np.nan),produced/initial,rtol=0,atol=1e-12)
    balance = {'initial_volume_mm3': initial, 'product_volume_mm3': produced,
               'unused_in_slabs_mm3': slab_unused, 'final_remainder_mm3': remainder,
               'kerf_volume_mm3': 0., 'initial_trim_volume_mm3': 0.,
               'unallocated_volume_mm3': max(0., initial - produced),
               'error_mm3': error, 'tolerance_mm3': vtol}
    saved_balance = result.get('volume_balance', {})
    balance_matches = all(np.isfinite(saved_balance.get(k, np.nan)) and
                          abs(saved_balance[k] - balance[k]) <= vtol
                          for k in balance if k not in ('tolerance_mm3',))
    accepted = all(c['accepted'] for c in checks) and cuts_ok and count_ok and error <= vtol and balance_matches
    return {'accepted': bool(accepted), 'n_total': total, 'slab_count': len(checks),
            'cut_count': sum(s['cut_performed'] for s in plan.steps), 'cuts_valid': bool(cuts_ok),
            'declared_count_valid': bool(count_ok), 'saved_balance_matches': bool(balance_matches),
            'yield_by_volume': produced / initial, 'volume_balance': balance, 'slabs': checks,
            'scope': 'full_source_block_products_and_sequential_parallel_cuts',
            'arithmetic': 'float64 / Manifold + GEOS; numerical shared kernels',
            'cross_slab_nonoverlap': 'verified disjoint parallel intervals; touching allowed',
            **({'machine':machine_audit} if machine_audit is not None else {})}
