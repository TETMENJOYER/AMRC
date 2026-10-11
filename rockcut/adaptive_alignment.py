"""Schedule deterministic grouped-gradient attempts from verified best states.

Adam and its LBFGS correction form one trajectory, but their verified yields
are recorded separately. Only an entirely completed, unproductive trajectory
counts toward suspension. This scheduler never authorizes a candidate.
"""
import math
import numpy as np


class GradientController:
    def __init__(self, config, phase, metric):
        self.config, self.phase, self.metric = config, phase, metric
        self.adaptive = config.method_policy == 'adaptive' and config.stop_mode == 'convergence'
        self.failures = 0
        self.paused = False
        self.skipped = 0
        self.last_state = None
        self.reference = None
        self.last_groups = None
        self.events = []

    def changed(self, state):
        return self.last_state is not None and not np.array_equal(state, self.last_state)

    def significant_change(self, state, metrics):
        if self.reference is None or not self.changed(state):
            return False
        gain = self.reference-metrics[self.metric]
        threshold = max(self.config.method_absolute_gain,
                        self.config.method_relative_gain*self.reference)
        center_rms = float(np.sqrt(np.mean(np.sum((state[:, :2]-self.last_state[:, :2])**2, axis=1))))
        delta = (state[:, 2]-self.last_state[:, 2]+np.pi/2) % np.pi-np.pi/2
        angle_rms = float(np.sqrt(np.mean(delta**2)))
        groups_changed = metrics['exact_direction_groups'] < self.last_groups
        return gain >= threshold and (center_rms >= self.config.method_center_rms
                                      or angle_rms >= self.config.method_angle_rms or groups_changed)

    def decide(self, state, metrics, round_number, last_round=False, confirmation=False):
        if not self.adaptive:
            return 'always'
        if not self.paused:
            return 'active'
        if not self.changed(state):
            return None  # Deterministic repetition of the identical state has no new information.
        if confirmation:
            return 'plateau_confirmation'
        if last_round:
            return 'round_limit_probe'
        if self.significant_change(state, metrics):
            return 'substantial_external_change'
        if self.skipped >= self.config.probe_interval:
            return 'periodic_probe'
        return None

    def skip(self, round_number):
        self.skipped += 1
        self.events.append({'round': round_number, 'action': 'skip', 'paused': self.paused,
                            'consecutive_skips': self.skipped})

    def begin(self, state, metrics):
        self.last_state = np.array(state, copy=True)
        self.reference = float(metrics[self.metric])
        self.last_groups = metrics['exact_direction_groups']

    def finish(self, round_number, reason, accepted, metrics, seconds, completed=True):
        resumed = self.paused
        if sum(accepted.values()):
            self.failures, self.paused = 0, False
        elif completed:
            self.failures += 1
            self.paused = self.adaptive and self.failures >= self.config.method_window
        self.skipped = 0
        self.events.append({'round': round_number, 'action': 'probe' if resumed else 'run',
            'reason': reason, 'completed': completed, 'accepted_by_method': accepted,
            'verified_target_gain': self.reference-float(metrics[self.metric]),
            'seconds': seconds, 'consecutive_unproductive_attempts': self.failures, 'paused': self.paused})

    def summary(self):
        return {'requested_policy': self.config.method_policy,
                'effective_policy': 'adaptive' if self.adaptive else 'always',
                'phase': self.phase, 'metric': self.metric,
                'attempts': sum(e['action'] != 'skip' for e in self.events),
                'skips': sum(e['action'] == 'skip' for e in self.events),
                'resumptions': sum(e['action'] == 'probe' for e in self.events),
                'events': self.events}


def would_stall(monitor, score):
    """Preview the unchanged phase rule without adding a provisional history row."""
    cfg = monitor.config
    next_round = len(monitor.scores)
    if next_round < max(cfg.window, cfg.min_rounds):
        return False
    reference = monitor.scores[-cfg.window]
    return max(0., reference-score) <= cfg.absolute_gain+cfg.relative_gain*reference
