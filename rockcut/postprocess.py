"""Reorganize saved results independently of the count search."""
import copy
import time
from dataclasses import asdict, replace

from .alignment import AlignmentConfig, align_layout
from .problem import Problem


def align_block_result(result, config=None, progress=None):
    from .verify_block import block_inputs, verify_block_result
    from .cut_plan import parallel_plan
    from .packing3d import verify_layout
    cfg = config or AlignmentConfig()
    original_audit = verify_block_result(result)
    if not original_audit['accepted']:
        raise ValueError('Cannot align an unaccepted source block result')
    output = copy.deepcopy(result)
    block, side, thickness = block_inputs(output['inputs'])
    plan = parallel_plan(block, output['plan']['normal'], thickness)
    started = time.monotonic()
    deadline = started+cfg.seconds if cfg.seconds is not None else float('inf')
    jobs = [i for i, r in enumerate(output['slabs']) if r['n'] and r.get('problem')]
    rows = []
    for position, i in enumerate(jobs):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            break
        r = output['slabs'][i]
        problem = Problem.from_dict(r['problem'])
        allowance = (None if cfg.seconds is None else remaining if cfg.stop_mode == 'convergence'
                     else remaining/max(1, len(jobs)-position))
        inner = replace(cfg, seconds=allowance)
        aligned = align_layout(problem, r['state'], inner,
            lambda q, k=i: verify_layout(block, plan.specs[k], side, q, slab_local=plan.pieces[k]))
        r.update(aligned)
        rows.append({'slab_index': i, 'n': r['n'], 'improved': aligned['alignment']['improved'],
                     'stop_reason': aligned['alignment']['stop_reason'],
                     'completed_rounds': aligned['alignment']['completed_rounds'],
                     'angle_guard_passed': aligned['alignment']['angle_guard_passed'],
                     'phase_stops': {p['phase']: p['stop_reason'] for p in aligned['alignment']['phases']},
                     'before': aligned['alignment']['before'], 'after': aligned['alignment']['after']})
        if progress:
            progress({'event': 'aligned_slab', **rows[-1]})
    output['validation'] = verify_block_result(output)
    if not output['validation']['accepted'] or output['n_total'] != result['n_total']:
        raise RuntimeError('Independent aligned block verification failed')
    output['alignment'] = {'stage': 'post_search_fixed_count', 'n_total': output['n_total'],
                           'count_preserved': True, 'slabs': rows,
                           'policy': 'angles_then_edges_with_shared_group_angles',
                           'method_policy': cfg.method_policy,
                           'angle_guard_passed': all(r['angle_guard_passed'] for r in rows),
                           'slabs_not_processed': len(jobs)-len(rows),
                           'config': asdict(cfg),
                           'stop_reason': 'time_safety_limit' if len(rows) < len(jobs) else 'all_slabs_processed',
                           'elapsed_seconds': time.monotonic()-started,
                           'budget_seconds': cfg.seconds, 'cut_schedule_verified': False}
    return output


def align_result(result, config=None, acceptance_check=None, progress=None):
    if result.get('kind') == 'block_cutting_result':
        if acceptance_check is not None:
            raise ValueError('Block postprocessing rebuilds its own 3D checker')
        return align_block_result(result, config, progress)
    if result.get('angle_unit') != 'radians' or result.get('n') != len(result['state']):
        raise ValueError('Invalid saved count or angle unit')
    output = copy.deepcopy(result)
    if result.get('kind') == 'slab_packing_result':
        from .packing3d import source_geometry, verify_layout, verify_slab_result
        block, spec, side = source_geometry(result['source3d'])
        acceptance_check = lambda q: verify_layout(block, spec, side, q)
    elif (result.get('validation', {}).get('additional_check') is not None or result.get('validation3d') is not None) and acceptance_check is None:
        raise ValueError('Saved result has additional verification: provide its original checker (CLI --control)')
    aligned = align_layout(Problem.from_dict(result['problem']), result['state'], config, acceptance_check, progress)
    output.update(aligned)
    if result.get('kind') == 'slab_packing_result':
        output['validation3d'] = verify_slab_result(output)
        if not output['validation3d']['accepted']:
            raise RuntimeError('Saved aligned slab rejected by independent 3D audit')
    return output
