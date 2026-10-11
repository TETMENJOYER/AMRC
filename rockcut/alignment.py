"""Second stage: reorganize a verified layout with its count held fixed.

    Soft regularity terms guide proposals; independent geometry alone authorizes
    saved placements. An immutable original and the best verified state survive
    infeasible gradient trajectories and failed complete reorganizations.
"""
from dataclasses import asdict, dataclass
import math
import time

import numpy as np
import torch

from .local import LocalOptimizer, resolve_device
from .seeds import shelf_seeds
from .verify import check_candidate
from .adaptive_alignment import GradientController, would_stall


@dataclass
class AlignmentConfig:
    seconds: float | None = None
    steps: int = 180
    device: str = 'auto'
    rebuild: bool = True
    stop_mode: str = 'convergence'
    window: int = 3
    min_rounds: int = 3
    max_rounds: int = 12
    relative_gain: float = .002
    absolute_gain: float = 1e-7
    score_floor: float = 1e-12
    method_policy: str = 'adaptive'
    method_window: int = 3
    probe_interval: int = 3
    method_relative_gain: float = .01
    method_absolute_gain: float = 1e-6
    method_center_rms: float = .025
    method_angle_rms: float = .002

    def __post_init__(self):
        if self.seconds is not None and (not math.isfinite(self.seconds) or self.seconds <= 0):
            raise ValueError('alignment seconds must be finite and positive')
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError('alignment steps must be a positive integer')
        if not isinstance(self.rebuild, bool):
            raise ValueError('alignment rebuild must be boolean')
        if self.stop_mode not in ('convergence', 'time'):
            raise ValueError('alignment stop_mode must be convergence or time')
        if self.method_policy not in ('always', 'adaptive'):
            raise ValueError('alignment method_policy must be always or adaptive')
        for name in ('window', 'min_rounds', 'max_rounds', 'method_window', 'probe_interval'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'alignment {name} must be a positive integer')
        for name in ('relative_gain', 'absolute_gain', 'method_relative_gain', 'method_absolute_gain',
                     'method_center_rms', 'method_angle_rms'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'alignment {name} must be finite and nonnegative')
        if not math.isfinite(self.score_floor) or self.score_floor <= 0:
            raise ValueError('alignment score_floor must be finite and positive')
        resolve_device(self.device)


class ConvergenceMonitor:
    """Monitor only verified best scores across COMPLETE mixed-method rounds.

    Stop when S[k-window]-S[k] <= abs_tol + rel_tol*S[k-window].
    A log ratio is recorded safely near zero, without changing the objective.
    Neither rejected states nor a changing geometric penalty enter this test.
    """
    def __init__(self, initial_score, config):
        if not math.isfinite(initial_score) or initial_score < 0:
            raise ValueError('initial regularity score must be finite and nonnegative')
        self.config = config
        self.scores = [float(initial_score)]
        self.history = []

    def observe(self, verified_best_score):
        if (not math.isfinite(verified_best_score) or verified_best_score < 0
                or verified_best_score > self.scores[-1]+1e-12):
            raise ValueError('convergence needs a finite nonincreasing verified best score')
        self.scores.append(float(verified_best_score))
        round_number = len(self.scores)-1
        row = {'round': round_number, 'score': float(verified_best_score),
               'round_gain': self.scores[-2]-self.scores[-1],
               'window_gain': None, 'window_relative_gain': None,
               'window_log_gain': None, 'window_gain_threshold': None, 'stalled': False}
        cfg = self.config
        if round_number >= max(cfg.window, cfg.min_rounds):
            reference = self.scores[-cfg.window-1]
            gain = max(0., reference-verified_best_score)
            threshold = cfg.absolute_gain+cfg.relative_gain*reference
            row.update(window_gain=gain, window_relative_gain=gain/max(reference, cfg.score_floor),
                       window_log_gain=math.log(reference+cfg.score_floor)-math.log(verified_best_score+cfg.score_floor),
                       window_gain_threshold=threshold, stalled=gain <= threshold)
        self.history.append(row)
        return row['stalled']


def _frame(q):
    z = np.exp(4j*q[:, 2]).mean() if len(q) else 1+0j
    phi = (np.angle(z)/4) % (np.pi/2) if abs(z) > 1e-12 else (min(q[:, 2] % (np.pi/2)) if len(q) else 0.)
    rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    return float(phi), rotation, float(abs(z))


def _neighbors(q):
    i, j = np.triu_indices(len(q), 1)
    dist2 = np.sum((q[i, :2]-q[j, :2])**2, axis=1)
    near = dist2 <= 25.
    return i[near], j[near], np.exp(-dist2[near]/16.)


def _line_gaps(xy, extents, i, j):
    # Four signed edge combinations on each of the two orthogonal axes.
    signs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    return (xy[i]-xy[j])[:, None, :] + signs[None, :, 0, None]*extents[i, None] - signs[None, :, 1, None]*extents[j, None]


# Arithmetic allowances only; these are not manufacturing angle tolerances.
ANGLE_ROUNDOFF_DEGREES = 1e-8
DISPERSION_ROUNDOFF = 1e-12


def angles_not_worse(candidate, reference):
    """A better edge term cannot buy degradation of any protected angle metric."""
    return (candidate['angle_error_score'] <= reference['angle_error_score']+1e-12
            and candidate['direction_dispersion'] <= reference['direction_dispersion']+DISPERSION_ROUNDOFF
            and candidate['mean_angle_error_degrees'] <= reference['mean_angle_error_degrees']+ANGLE_ROUNDOFF_DEGREES
            and candidate['max_angle_error_degrees'] <= reference['max_angle_error_degrees']+ANGLE_ROUNDOFF_DEGREES
            and candidate['direction_families_1deg'] <= reference['direction_families_1deg']
            and candidate['exact_direction_groups'] <= reference['exact_direction_groups'])


def angle_improved(candidate, reference):
    return (candidate['angle_error_score'] < reference['angle_error_score']-1e-12
            or candidate['direction_dispersion'] < reference['direction_dispersion']-DISPERSION_ROUNDOFF
            or candidate['direction_families_1deg'] < reference['direction_families_1deg']
            or candidate['exact_direction_groups'] < reference['exact_direction_groups'])


def angle_groups(state):
    """Exact numerical direction families on the circle modulo 90 degrees.

    Group membership uses 1e-10 radians, never the loose reporting threshold.
    Nearby but distinct directions remain separate until a verified merge.
    Handles the 0/90 seam and is invariant to item permutations.
    """
    theta = np.asarray(state, dtype=float).reshape(-1, 3)[:, 2]
    period = np.pi/2
    if not len(theta):
        return np.empty(0, dtype=int), np.empty(0), np.empty(0, dtype=int)
    angles = theta % period
    order = np.argsort(angles, kind='stable')
    gaps = np.diff(np.r_[angles[order], angles[order[0]]+period])
    order = np.roll(order, -int(np.argmax(gaps))-1)
    labels, bases = np.empty(len(theta), dtype=int), []
    for i in order:
        difference = (angles[i]-bases[-1]+period/2) % period-period/2 if bases else math.inf
        if abs(difference) > 1e-10:
            bases.append(float(angles[i]))
        labels[i] = len(bases)-1
    bases = np.asarray(bases)
    turns = np.rint((theta-bases[labels])/period).astype(int)
    return labels, bases, turns


class GroupedLayout:
    """Train centers and shared group angles, never independent item angles.

    In the edge phase all group angle differences are frozen: one common
    rotation is the sole angular degree of freedom, including during repair.
    """
    def __init__(self, normalized_state, phase, device):
        labels, bases, turns = angle_groups(normalized_state)
        self.labels = torch.tensor(labels, dtype=torch.long, device=device)
        self.bases = torch.tensor(bases, dtype=torch.float64, device=device)
        self.turns = torch.tensor(turns*np.pi/2, dtype=torch.float64, device=device)
        self.xy = torch.nn.Parameter(torch.tensor(normalized_state[:, :2], dtype=torch.float64, device=device))
        self.phase = phase
        self.rotation = torch.nn.Parameter(self.bases.clone() if phase == 'angles'
                                           else torch.zeros(1, dtype=torch.float64, device=device))

    def parameters(self):
        return [self.xy, self.rotation]

    def values(self):
        bases = self.rotation if self.phase == 'angles' else self.bases+self.rotation[0]
        theta = bases[self.labels]+self.turns
        return torch.cat((self.xy, theta[:, None]), dim=1)[None]

    def repair(self, local, deadline):
        # Geometric polishing uses the SAME constrained representation.
        optimizer = torch.optim.LBFGS(self.parameters(), max_iter=40,
                                     tolerance_grad=1e-10, tolerance_change=1e-12,
                                     line_search_fn='strong_wolfe')
        def closure():
            if time.monotonic() >= deadline:
                raise TimeoutError
            optimizer.zero_grad(set_to_none=True)
            loss = local.penalty(self.values()).sum()
            loss.backward()
            return loss
        try:
            optimizer.step(closure)
        except TimeoutError:
            pass
        return self.values().detach()[0].cpu().numpy().copy()


def regularity(problem, state):
    """Permutation-invariant geometric proxies, not an executable cut schedule.

    Position errors are normalized by the short side. Orthogonal orientations
    belong to the same direction family. Contact/acceptance tolerances are never
    derived from these cosmetic grouping thresholds.
    """
    state = np.asarray(state, dtype=float).reshape(-1, 3)
    q = problem.normalize(state)
    phi, rot, coherence = _frame(q)
    d = q[:, 2]-phi
    extents = np.column_stack((1.5*np.abs(np.cos(d))+.5*np.abs(np.sin(d)),
                              1.5*np.abs(np.sin(d))+.5*np.abs(np.cos(d))))
    i, j, weights = _neighbors(q)
    gaps = _line_gaps(q[:, :2]@rot, extents, i, j)
    errors = np.minimum(1., np.min(gaps**2, axis=(1, 2))) if len(i) else np.empty(0)
    edge_error = float(np.average(errors, weights=weights)) if len(errors) else 0.
    angles = np.sort(q[:, 2] % (np.pi/2))
    cyclic = np.diff(np.r_[angles, angles[0]+np.pi/2]) if len(angles) else np.empty(0)
    families = max(1, int(np.sum(cyclic > np.deg2rad(1)))) if len(angles) else 0
    angular_error = np.abs((d+np.pi/4) % (np.pi/2)-np.pi/4)
    # Count infinite support-line groups only for reporting. It omits segment
    # connectivity, access and cutting order, and is not the number of saw moves.
    aligned = angular_error <= np.deg2rad(1)
    xy = q[:, :2]@rot
    line_groups = int(4*np.sum(~aligned))
    for axis in (0, 1):
        offsets = np.sort(np.r_[xy[aligned, axis]-extents[aligned, axis],
                                xy[aligned, axis]+extents[aligned, axis]])
        if len(offsets):
            line_groups += 1+int(np.sum(np.diff(offsets) > .005))
    dispersion = max(0., 1.-coherence)
    return {'score': dispersion+.2*edge_error, 'direction_dispersion': dispersion,
            'edge_error': edge_error,
            'angle_error_score': float(angular_error.mean()*4/np.pi) if len(q) else 0.,
            'edge_alignment_rms_mm': math.sqrt(edge_error)*problem.short_side,
            'mean_angle_error_degrees': float(np.rad2deg(angular_error).mean()) if len(q) else 0.,
            'max_angle_error_degrees': float(np.rad2deg(angular_error).max()) if len(q) else 0.,
            'direction_families_1deg': families, 'support_line_groups_proxy': line_groups,
            'exact_direction_groups': len(angle_groups(q)[1]),
            'dominant_direction_degrees': float(np.rad2deg(phi)),
            'metric_scope': 'Orthogonal direction coherence and nearby edge alignment; saw schedule not verified.'}


def align_layout(problem, state, config=None, acceptance_check=None, progress=None):
    cfg = config or AlignmentConfig()
    started = time.monotonic()
    deadline = started+cfg.seconds if cfg.seconds is not None else math.inf
    original = np.array(state, dtype=float, copy=True)
    if original.size == 0:
        original = np.empty((0, 3))
    if original.ndim != 2 or original.shape[1] != 3 or not np.isfinite(original).all():
        raise ValueError('alignment needs a finite physical (N, 3) layout')
    n = len(original)
    check = check_candidate(problem, original, acceptance_check)
    if not check['accepted']:
        raise ValueError('alignment requires a strictly verified starting layout')
    best = original.copy()
    before = regularity(problem, best)
    best_metrics = before.copy()
    history, phases = [], []
    counters = {'gradient_steps': 0, 'proposals': 0, 'accepted_improvements': 0,
                'rejected_geometry': 0, 'rejected_angle_regressions': 0}
    device = resolve_device(cfg.device)
    seeds = None
    edge_anchor = None
    method_counts, method_times = {}, {}

    def consider_impl(candidate, source, phase):
        nonlocal best, best_metrics, check
        candidate = np.array(candidate, dtype=float, copy=True)
        if candidate.shape != original.shape or not np.isfinite(candidate).all():
            return
        counters['proposals'] += 1
        metrics = regularity(problem, candidate)
        anchors = [before, best_metrics]
        if phase == 'edges':
            anchors.append(edge_anchor)
        if not all(angles_not_worse(metrics, a) for a in anchors):
            counters['rejected_angle_regressions'] += 1
            return
        if phase == 'angles':
            improves = angle_improved(metrics, best_metrics)
        else:
            improves = metrics['edge_error'] < best_metrics['edge_error']-1e-12
        if not improves:
            return
        candidate_check = check_candidate(problem, candidate, acceptance_check)
        if not candidate_check['accepted']:
            counters['rejected_geometry'] += 1
            return
        best, best_metrics, check = candidate.copy(), metrics, candidate_check
        counters['accepted_improvements'] += 1
        entry = {'source': source, 'phase': phase, 'n': n, 'score': metrics['score'],
                 'angle_error_score': metrics['angle_error_score'], 'edge_error': metrics['edge_error'],
                 'mean_angle_error_degrees': metrics['mean_angle_error_degrees'],
                 'max_angle_error_degrees': metrics['max_angle_error_degrees'],
                 'direction_dispersion': metrics['direction_dispersion'],
                 'direction_families_1deg': metrics['direction_families_1deg'],
                 'exact_direction_groups': metrics['exact_direction_groups'],
                 'elapsed_seconds': time.monotonic()-started}
        history.append(entry)
        if progress:
            progress(entry)
        return True

    def consider(candidate, source, phase):
        started_proposal = time.perf_counter()
        metric = 'angle_error_score' if phase == 'angles' else 'edge_error'
        previous = best_metrics[metric]
        count = counters['proposals']
        accepted = consider_impl(candidate, source, phase)
        if counters['proposals'] > count:
            row = method_counts.setdefault(phase+'/'+source, {'proposals': 0, 'accepted': 0,
                'verified_target_gain': 0., 'proposal_seconds': 0.})
            row['proposals'] += 1
            row['accepted'] += bool(accepted)
            row['verified_target_gain'] += previous-best_metrics[metric] if accepted else 0.
            row['proposal_seconds'] += time.perf_counter()-started_proposal

    def record_time(phase, method, started_method):
        key = phase+'/'+method
        method_times[key] = method_times.get(key, 0.)+time.perf_counter()-started_method

    def run_gradient(phase):
        adam_started = time.perf_counter()
        steps_before = counters['gradient_steps']
        local = LocalOptimizer(problem, device, margin=0)
        initial = problem.normalize(best)
        phi, _, _ = _frame(initial)
        i, j, weights = _neighbors(initial)
        ii = torch.tensor(i, dtype=torch.long, device=device)
        jj = torch.tensor(j, dtype=torch.long, device=device)
        ww = torch.tensor(weights, dtype=torch.float64, device=device)
        signs = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=torch.float64, device=device)
        grouped = GroupedLayout(initial, phase, device)
        optimizer = torch.optim.Adam(grouped.parameters(), lr=.008)
        gradient_deadline = (deadline-cfg.seconds*.2 if cfg.stop_mode == 'time' and cfg.seconds is not None else deadline)
        for step in range(cfg.steps):
            if time.monotonic() >= gradient_deadline:
                break
            optimizer.zero_grad(set_to_none=True)
            q = grouped.values()
            frame_angle = phi if phase == 'angles' else phi+grouped.rotation[0]
            angles = q[0, :, 2]-frame_angle
            if phase == 'angles':
                # A true angular objective, without a reward for edge changes.
                objective = torch.atan2((4*angles).sin(), (4*angles).cos()).abs().mean()/np.pi
            else:
                co, si = frame_angle.cos(), frame_angle.sin()
                rr = torch.stack((torch.stack((co, -si)), torch.stack((si, co))))
                xy = q[0, :, :2]@rr
                extents = torch.stack((1.5*angles.cos().abs()+.5*angles.sin().abs(),
                                       1.5*angles.sin().abs()+.5*angles.cos().abs()), dim=1)
                if len(i):
                    gaps = (xy[ii]-xy[jj])[:, None, :] + signs[None, :, 0, None]*extents[ii, None] - signs[None, :, 1, None]*extents[jj, None]
                    objective = (gaps.square().flatten(1).min(1).values.clamp_max(1)*ww).sum()/ww.sum()
                else:
                    objective = q.sum()*0
            multiplier = 100.*(10.**(2*step/max(1, cfg.steps-1)))
            loss = multiplier*local.penalty(q).sum()+objective
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(grouped.parameters(), 10.)
            optimizer.step()
            counters['gradient_steps'] += 1
            if step % 10 == 9 or step == cfg.steps-1:
                candidate = grouped.values().detach()[0].cpu().numpy().copy()
                consider(problem.denormalize(candidate), 'grouped_gradient', phase)
        record_time(phase, 'grouped_adam', adam_started)
        if time.monotonic() < gradient_deadline:
            repair_started = time.perf_counter()
            candidate = grouped.repair(local, gradient_deadline)
            consider(problem.denormalize(candidate), 'grouped_gradient_repaired', phase)
            record_time(phase, 'grouped_lbfgs', repair_started)
        return counters['gradient_steps']-steps_before == cfg.steps and time.monotonic() < gradient_deadline

    def run_snaps(phase):
        if time.monotonic() >= deadline:
            return
        if phase == 'angles':
            # Merge entire exact groups towards another/common direction.
            # Individual members never rotate independently within a group.
            qn = problem.normalize(best)
            labels, bases, turns = angle_groups(qn)
            dominant, _, _ = _frame(qn)
            targets = np.unique(np.r_[dominant, 0., bases])
            for g in range(len(bases)):
                for target in targets:
                    if time.monotonic() >= deadline:
                        return
                    mask = labels == g
                    candidate = best.copy()
                    candidate[mask, 2] = (target+turns[mask]*np.pi/2) % np.pi
                    consider(candidate, 'verified_group_angle_snap', phase)
        else:
            # Translate one item onto a nearby edge; all angles stay bitwise fixed.
            phi, rot, _ = _frame(problem.normalize(best))
            for k in range(n):
                if time.monotonic() >= deadline:
                    return
                qn = problem.normalize(best)
                xy = qn[:, :2]@rot
                d = qn[:, 2]-phi
                extents = np.column_stack((1.5*np.abs(np.cos(d))+.5*np.abs(np.sin(d)),
                                          1.5*np.abs(np.sin(d))+.5*np.abs(np.cos(d))))
                neighbors = np.flatnonzero((np.sum((qn[:, :2]-qn[k, :2])**2, axis=1) <= 25.) & (np.arange(n) != k))
                if not len(neighbors):
                    continue
                gaps = _line_gaps(xy, extents, np.full(len(neighbors), k), neighbors)
                for axis in (0, 1):
                    options = gaps[:, :, axis].ravel()
                    options = options[(np.abs(options) > 1e-8) & (np.abs(options) <= .15)]
                    if len(options):
                        delta = float(options[np.argmin(np.abs(options))])
                        candidate = best.copy()
                        candidate[k, :2] -= delta*problem.short_side*rot[:, axis]
                        consider(candidate, 'verified_edge_snap', phase)

    for phase, metric in (('angles', 'angle_error_score'), ('edges', 'edge_error')):
        phase_before = best_metrics.copy()
        if phase == 'edges':
            edge_anchor = phase_before.copy()
        monitor = ConvergenceMonitor(phase_before[metric], cfg)
        method_monitor = GradientController(cfg, phase, metric)
        def attempt_gradient(round_number, attempt_reason):
            method_monitor.begin(problem.normalize(best), best_metrics)
            count, start = len(history), time.perf_counter()
            complete = run_gradient(phase)
            accepted = {source: sum(h['source'] == source for h in history[count:])
                        for source in ('grouped_gradient', 'grouped_gradient_repaired')}
            method_monitor.finish(round_number, attempt_reason, accepted, best_metrics,
                                  time.perf_counter()-start, complete)
            return sum(accepted.values()) > 0
        completed = 0
        reason = 'round_limit'
        # Skip an already attained angular/edge target without spending rounds.
        if not n or best_metrics[metric] <= cfg.score_floor:
            reason = 'regularity_target_reached'
        elif time.monotonic() >= deadline:
            reason = 'time_safety_limit'
        else:
            if cfg.rebuild:
                if seeds is None:
                    rebuild_deadline = (min(deadline, started+cfg.seconds*.4)
                                        if cfg.stop_mode == 'time' and cfg.seconds is not None else deadline)
                    seeds = shelf_seeds(problem, limit=8, deadline=rebuild_deadline, margin=1e-5)
                for seed in seeds:
                    if time.monotonic() >= deadline:
                        break
                    if len(seed) >= n:
                        consider(problem.denormalize(seed[:n]), 'complete_reorganization', phase)
            rounds_limit = 1 if cfg.stop_mode == 'time' else cfg.max_rounds
            for round_index in range(rounds_limit):
                if best_metrics[metric] <= cfg.score_floor:
                    reason = 'regularity_target_reached'
                    break
                if time.monotonic() >= deadline:
                    reason = 'time_safety_limit'
                    break
                round_number = round_index+1
                # A stalled window must be confirmed from the same state that
                # the ordinary policy would use BEFORE its edge/angle snaps.
                # Probing after the snaps changes the LBFGS basin (L_25).
                attempt_reason = method_monitor.decide(problem.normalize(best), best_metrics, round_number,
                    last_round=round_number == rounds_limit,
                    confirmation=would_stall(monitor, best_metrics[metric]))
                if attempt_reason is not None:
                    attempt_gradient(round_number, attempt_reason)
                else:
                    method_monitor.skip(round_number)
                snaps_started = time.perf_counter()
                run_snaps(phase)
                record_time(phase, 'angle_snap' if phase == 'angles' else 'edge_snap', snaps_started)
                if time.monotonic() >= deadline:
                    reason = 'time_safety_limit'
                    break
                # Verified improvements can only move the window away from
                # plateau. Thus its boundary probe belongs before this round,
                # never after a potentially trajectory-changing snap sweep.
                completed += 1
                stalled = monitor.observe(best_metrics[metric])
                monitor.history[-1].update(phase=phase, metric=metric, elapsed_seconds=time.monotonic()-started)
                if best_metrics[metric] <= cfg.score_floor:
                    reason = 'regularity_target_reached'
                    break
                if cfg.stop_mode == 'convergence' and stalled:
                    reason = 'plateau'
                    break
                if cfg.stop_mode == 'time':
                    reason = 'single_pass_finished'
                    break
            else:
                reason = 'round_limit'
        phases.append({'phase': phase, 'metric': metric, 'before': phase_before, 'after': best_metrics.copy(),
                       'completed_rounds': completed, 'stop_reason': reason,
                       'gradient_controller': method_monitor.summary(),
                       'convergence_history': monitor.history})
    reasons = [p['stop_reason'] for p in phases]
    if 'time_safety_limit' in reasons:
        stop_reason = 'time_safety_limit'
    elif 'round_limit' in reasons:
        stop_reason = 'round_limit'
    elif all(r == 'regularity_target_reached' for r in reasons):
        stop_reason = 'regularity_target_reached'
    elif 'single_pass_finished' in reasons:
        stop_reason = 'single_pass_finished'
    else:
        stop_reason = 'plateau'
    check = check_candidate(problem, best, acceptance_check)
    if not check['accepted'] or len(best) != n or not angles_not_worse(best_metrics, before):
        raise RuntimeError('Final independent alignment verification or angle guard failed')
    return {'state': best.tolist(), 'n': n, 'validation': check,
            'alignment': {'config': asdict(cfg), 'device_used': device, 'original_state': original.tolist(),
                          'before': before, 'after': best_metrics,
                          'improved': angle_improved(best_metrics, before) or best_metrics['edge_error'] < before['edge_error']-1e-12,
                          'count_preserved': True, 'elapsed_seconds': time.monotonic()-started,
                          'policy': 'angles_then_edges_with_shared_group_angles',
                          'score_role': 'Legacy weighted score is diagnostic; phase objectives decide acceptance and stopping.',
                          'angle_guard_passed': True, 'phases': phases,
                          'method_policy': cfg.method_policy,
                          'methods': method_counts, 'method_inclusive_seconds': method_times,
                          'angle_roundoff_degrees': ANGLE_ROUNDOFF_DEGREES,
                          'stop_reason': stop_reason,
                          'completed_rounds': sum(p['completed_rounds'] for p in phases),
                          'convergence_history': [h for p in phases for h in p['convergence_history']],
                          'stopping_score': 'verified angle error in angle phase; verified edge error in edge phase',
                          'history': history, 'counters': counters, 'search_budget_unchanged': True,
                          'stage': 'post_search_fixed_count', 'cut_schedule_verified': False}}
