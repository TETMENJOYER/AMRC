"""Optional bounded N+1 visits on a live, already constructed parallel plan.

The caller owns the absolute search deadline and the final block audit/balance.
Only independently accepted target layouts replace individual slab records.
"""
from collections import deque
from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from .packing3d import verify_layout
from .solver import SearchConfig, VARIANTS, solve
from .verify import check_candidate


@dataclass(frozen=True)
class QuantityPassConfig:
    policy: str = 'retry'
    slab_seconds: float = 12.
    attempt_seconds: float = 3.
    seed: int = 20261009
    batch_size: int = 6
    adam_steps: int = 100
    polish_iterations: int = 80
    polish_candidates: int = 3
    iterations_per_n: int = 40
    device: str = 'auto'
    threads: int = 1
    max_visits_per_slab: int = 2
    grid_seed: str = 'rigid'
    grid_seconds: float = 1.

    def __post_init__(self):
        if self.policy not in ('retry', 'long'):
            raise ValueError('quantity policy must be retry or long')
        for name in ('slab_seconds', 'attempt_seconds'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if type(self.max_visits_per_slab) is not int or self.max_visits_per_slab < 1:
            raise ValueError('max_visits_per_slab must be a positive integer')
        if self.device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('device must be auto, cpu or cuda')
        self.search_config(self.slab_seconds, self.seed)

    def search_config(self, seconds, seed):
        return SearchConfig(seconds=seconds,
            seconds_per_n=min(self.attempt_seconds, seconds) if self.policy == 'retry' else seconds,
            seed=seed, batch_size=self.batch_size, adam_steps=self.adam_steps,
            polish_iterations=self.polish_iterations, polish_candidates=self.polish_candidates,
            iterations_per_n=self.iterations_per_n, device=self.device, threads=self.threads,
            alignment_seconds=0., retry_failed_n=self.policy == 'retry',
            grid_seed=self.grid_seed,grid_seconds=self.grid_seconds, **VARIANTS['combined'])


def _failed_penalty(record):
    target = record['n'] + 1
    for attempt in reversed((record.get('search') or {}).get('attempts', [])):
        value = attempt.get('best_area_penalty')
        if (attempt.get('n') == target and attempt.get('status') != 'feasible'
                and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
            return float(value)
    return None


def _failure_stop(result, cfg, now, visit_deadline, deadline):
    if now >= deadline:
        return 'global_budget_exhausted'
    if now >= visit_deadline:
        return 'slab_quota_exhausted'
    attempts = result.get('attempts', [])
    if attempts and attempts[-1].get('iterations', 0) >= cfg.iterations_per_n:
        return 'attempt_iteration_limit'
    # Older solver versions label early failed attempts as budget exhaustion.
    reported = result.get('stop_reason', 'target_not_found')
    return 'one_attempt_failed' if reported == 'search_budget_exhausted' else reported


def improve_candidate(block, candidate, side, config, deadline, emit=None):
    """Mutate accepted N+1 slab records and return a fully serializable protocol.

    Failed visits do not repeat in this pass. Successful visits append their next
    target after all pending slabs, with a finite per-slab visit cap. Thus early
    success returns its unused time to the same absolute-budget queue. Bounds
    and old failed penalties order attempts; they never authorize acceptance.
    """
    cfg = config
    if not isinstance(cfg, QuantityPassConfig):
        raise TypeError('config must be QuantityPassConfig')
    if not math.isfinite(deadline) or not math.isfinite(side) or side <= 0:
        raise ValueError('deadline must be finite and side must be positive')
    started = time.monotonic()
    records, problems, plan = candidate['records'], candidate['problems'], candidate['plan']
    if not (len(records) == len(problems) == len(plan.specs) == len(plan.pieces)):
        raise ValueError('candidate does not match its frozen plan')
    events, visits, preparation_skipped = [], [], []

    def log(event, **fields):
        row = {'event': event, 'elapsed_seconds': time.monotonic()-started, **fields}
        events.append(row)
        if emit is not None:
            emit(event, **{k: v for k, v in row.items() if k != 'event'})

    initial_counts = [record['n'] for record in records]
    eligible = []
    for index, (record, problem, spec) in enumerate(zip(records, problems, plan.specs)):
        if record['index'] != index or record['spec'] != spec.to_dict():
            raise ValueError('slab record differs from the frozen plan')
        if problem is not None and problem.area_upper_bound > record['n']:
            penalty = _failed_penalty(record)
            eligible.append({'slab_index': index, 'before_n': record['n'], 'target_n': record['n']+1,
                'area_upper_bound': problem.area_upper_bound, 'gap': problem.area_upper_bound-record['n'],
                'last_failed_penalty': penalty})
    eligible.sort(key=lambda row: (row['last_failed_penalty'] is None,
        row['last_failed_penalty'] if row['last_failed_penalty'] is not None else math.inf,
        -row['gap'], row['slab_index']))
    queue = deque(row['slab_index'] for row in eligible)
    counts = {index: 0 for index in queue}
    preparation_seconds = time.monotonic()-started
    log('quantity_pass_start', policy=cfg.policy, before_n=sum(initial_counts), queue=eligible,
        remaining_seconds=max(0., deadline-time.monotonic()), preparation_seconds=preparation_seconds)
    while queue and time.monotonic() < deadline:
        index = queue.popleft()
        record, problem = records[index], problems[index]
        before = record['n']
        target = before+1
        visit_started = time.monotonic()
        remaining_before = max(0., deadline-visit_started)
        quota = min(cfg.slab_seconds, remaining_before)
        visit_deadline = min(deadline, visit_started+quota)
        initial = np.array(record['state'], dtype=float, copy=True).reshape(-1, 3)
        if len(initial) != before:
            raise ValueError('declared initial count differs from the warm state')
        spec, piece = plan.specs[index], plan.pieces[index]

        def checker(state):
            return verify_layout(block, spec, side, state, slab_local=piece)

        if not check_candidate(problem, initial, checker)['accepted']:
            raise ValueError('quantity pass requires an independently valid initial state')
        initial_audit_seconds = time.monotonic()-visit_started
        preparation_seconds += initial_audit_seconds
        allowance = min(deadline, visit_deadline)-time.monotonic()
        if allowance <= 0:
            if time.monotonic() >= deadline:
                queue.appendleft(index)
                log('quantity_pass_preparation_budget_exhausted', slab_index=index,
                    initial_audit_seconds=initial_audit_seconds, remaining_seconds=0.)
                break
            # An unusually costly audit exhausted this visit's quota only.
            # Continue other slabs without repeatedly auditing the same one.
            preparation_skipped.append(index)
            log('quantity_pass_preparation_quota_exhausted', slab_index=index,
                initial_audit_seconds=initial_audit_seconds,
                remaining_seconds=max(0., deadline-time.monotonic()))
            continue
        counts[index] += 1
        inner = cfg.search_config(allowance, cfg.seed+len(visits)+1)
        log('quantity_pass_visit', slab_index=index, visit=counts[index], before_n=before,
            target_n=target, quota_seconds=quota, solver_allowance_seconds=allowance,
            remaining_seconds=max(0., deadline-time.monotonic()), queued_slabs=list(queue))
        search_started = time.monotonic()
        result = solve(problem, inner, target_n=target, initial_state=initial, acceptance_check=checker)
        search_wall_seconds = time.monotonic()-search_started
        accepted = False
        rejection = None
        final_audit_seconds = 0.
        # Saved acceptance flags do not authorize replacement, even for N+1.
        if result.get('n') == target:
            audit_started = time.monotonic()
            try:
                state = np.asarray(result['state'], dtype=float)
                if state.shape != (target, 3) or not np.isfinite(state).all():
                    rejection = 'invalid_candidate_shape_or_coordinates'
                else:
                    accepted = bool(check_candidate(problem, state, checker)['accepted'])
                    if not accepted:
                        rejection = 'independent_geometry_rejection'
            except (ValueError, TypeError):
                rejection = 'invalid_candidate_shape_or_coordinates'
            final_audit_seconds = time.monotonic()-audit_started
        elif result.get('n') != before:
            rejection = 'unexpected_candidate_count'
        now = time.monotonic()
        real_stop = ('target_reached' if accepted else rejection or
                     _failure_stop(result, inner, now, visit_deadline, deadline))
        row = {'slab_index': index, 'visit': counts[index], 'seed': inner.seed, 'policy': cfg.policy,
            'before_n': before, 'target_n': target, 'after_n': target if accepted else before,
            'accepted_addition': accepted, 'quota_seconds': quota, 'solver_allowance_seconds': allowance,
            'inner_config': asdict(inner), 'initial_audit_seconds': initial_audit_seconds,
            'search_wall_seconds': search_wall_seconds, 'final_audit_seconds': final_audit_seconds,
            'elapsed_seconds': now-visit_started, 'released_quota_seconds': max(0., quota-(now-visit_started)),
            'remaining_before_seconds': remaining_before, 'remaining_seconds': max(0., deadline-now),
            'global_deadline_overrun_seconds': max(0., now-deadline),
            'quota_overrun_seconds': max(0., now-visit_deadline),
            'attempts': result.get('attempts', []), 'reported_stop': result.get('stop_reason'),
            'stop_reason': real_stop}
        if cfg.grid_seed == 'rigid':
            row.update(grid_initialization=result.get('grid_initialization'),
                       best_retention=result.get('best_retention'))
        visits.append(row)
        if accepted:
            record.update(n=target, state=state.tolist(), status='quantity_pass_verified',
                quantity_pass_search=row.copy())
            if target < problem.area_upper_bound and counts[index] < cfg.max_visits_per_slab:
                queue.append(index)
        log('quantity_pass_visit_finished', **{k: v for k, v in row.items() if k != 'elapsed_seconds'},
            visit_elapsed_seconds=row['elapsed_seconds'])
    remaining = max(0., deadline-time.monotonic())
    pending = [{'slab_index': index, 'next_target_n': records[index]['n']+1,
                'completed_visits': counts[index]} for index in queue]
    reason = 'global_budget_exhausted' if queue and not remaining else 'queue_completed'
    summary = {'config': asdict(cfg), 'before_n': sum(initial_counts),
        'after_n': sum(record['n'] for record in records), 'initial_counts': initial_counts,
        'final_counts': [record['n'] for record in records], 'queue': eligible, 'visits': visits,
        'pending': pending, 'not_visited_due_budget': [row['slab_index'] for row in pending if not row['completed_visits']],
        'preparation_quota_skipped_slabs': preparation_skipped,
        'preparation_seconds': preparation_seconds, 'elapsed_seconds': time.monotonic()-started,
        'remaining_seconds': remaining, 'global_deadline_overrun_seconds': max(0., time.monotonic()-deadline),
        'stop_reason': reason}
    log('quantity_pass_finished', before_n=summary['before_n'], after_n=summary['after_n'],
        stop_reason=reason, pending=pending, remaining_seconds=remaining)
    summary['events'] = events
    return summary
