"""Optional nested wall-time accounting; leaves search decisions unchanged."""
from contextlib import ExitStack, contextmanager
from functools import wraps
import time
from unittest.mock import patch


class PhaseProfiler:
    def __init__(self):
        self.rows = {}
        self.stack = []

    @contextmanager
    def span(self, name):
        frame = [time.perf_counter(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - frame[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            row = self.rows.setdefault(name, {'calls': 0, 'inclusive_seconds': 0., 'exclusive_seconds': 0.})
            row['calls'] += 1
            row['inclusive_seconds'] += elapsed
            row['exclusive_seconds'] += max(0., elapsed - frame[1])

    def wrap(self, name, function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with self.span(name):
                return function(*args, **kwargs)
        return wrapped

    @contextmanager
    def instrument(self):
        from . import block_solver, solver, verify, verify_block, packing3d, report_block
        from .local import LocalOptimizer
        from .tabu import TabuMemory
        targets = [
            (block_solver, 'block_inputs', 'source_loading'),
            (verify_block, 'block_inputs', 'source_loading'),
            (report_block, 'block_inputs', 'source_loading'),
            (block_solver, 'parallel_plan', 'cut_geometry'),
            (verify_block, 'parallel_plan', 'cut_geometry'),
            (block_solver, 'build_slab', 'slab_footprints'),
            (block_solver, 'setup_candidates', 'stable_setups'),
            (block_solver, 'verify_layout', 'validation_3d'),
            (verify_block, 'verify_layout', 'validation_3d'),
            (packing3d, 'verify_layout', 'validation_3d'),
            (block_solver, 'verify_block_result', 'final_audit_other'),
            (verify, 'validate', 'validation_2d'),
            (solver, 'area_score', 'exact_area_scoring'),
            (LocalOptimizer, 'relax', 'adam'),
            (LocalOptimizer, 'polish', 'lbfgs'),
            (TabuMemory, 'contains', 'tabu'),
            (TabuMemory, 'add', 'tabu'),
        ]
        for module in (block_solver, solver):
            for name in ('shelf_seeds', 'fill_gaps'):
                targets.append((module, name, 'initial_layouts'))
        for name in ('insert_one', 'perturb', 'random_layout', 'destroy_repair'):
            targets.append((solver, name, 'proposals'))
        with ExitStack() as patches:
            for owner, attribute, category in targets:
                patches.enter_context(patch.object(owner, attribute, self.wrap(category, getattr(owner, attribute))))
            yield self

    def summary(self):
        total = sum(r['exclusive_seconds'] for r in self.rows.values())
        return {'accounted_wall_seconds': total,
                'phases': [{'phase': name, **row,
                            'exclusive_fraction': row['exclusive_seconds'] / total if total else 0.}
                           for name, row in sorted(self.rows.items(), key=lambda item: -item[1]['exclusive_seconds'])],
                'method': 'Nested synchronous wall timers; exclusive totals avoid double counting. CPU run only.'}
