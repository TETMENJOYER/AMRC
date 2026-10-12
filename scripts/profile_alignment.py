"""Reproducible fixed-N alignment timings; diagnostic instrumentation is in memory.

Run from the project root. Never reuses a populated output directory. CUDA spans
measure host wall time (including waits); only run boundaries synchronize CUDA.
"""
import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager, ExitStack
import copy
import cProfile
from dataclasses import asdict
import hashlib
import inspect
import io
import json
from pathlib import Path
import pstats
import tempfile
import time
import types
from unittest.mock import patch

import numpy as np
import torch

from rockcut import alignment, packing3d, postprocess, verify
from rockcut.alignment import AlignmentConfig, angles_not_worse, regularity
from rockcut.local import LocalOptimizer
from rockcut.performance import load_controls, source_context, environment, sync
from rockcut.problem import Problem
from rockcut.profiling import PhaseProfiler
from rockcut.report import write_report


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AlignmentProfiler(PhaseProfiler):
    def __init__(self):
        super().__init__()
        self.methods = defaultdict(lambda: {'proposals': 0, 'accepted': 0, 'verified_target_gain': 0.})

    @contextmanager
    def proposal(self, source, phase, metrics, history):
        key = f'{phase}/{source}'
        metric = 'angle_error_score' if phase == 'angles' else 'edge_error'
        previous, count = metrics[metric], len(history)
        self.methods[key]['proposals'] += 1
        with self.span('proposal/' + key):
            yield
        if len(history) > count:
            self.methods[key]['accepted'] += 1
            self.methods[key]['verified_target_gain'] += previous-history[-1][metric]

    def summary(self):
        result = super().summary()
        result['method'] = ('Nested host wall timers; exclusive times are disjoint. CUDA synchronized '
                            'at run boundaries only; these are not CUDA kernel durations.')
        result['methods'] = dict(self.methods)
        return result


def timed_statement(node, expression):
    return ast.With(items=[ast.withitem(context_expr=ast.parse(expression, mode='eval').body)], body=[node])


class Instrument(ast.NodeTransformer):
    """Add spans without changing expressions, loop order, or acceptance rules."""
    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if node.name == 'consider':
            declarations = [n for n in node.body if isinstance(n, ast.Nonlocal)]
            body = [n for n in node.body if not isinstance(n, ast.Nonlocal)]
            node.body = declarations + [ast.With(items=[ast.withitem(context_expr=ast.parse(
                '__profile.proposal(source, phase, best_metrics, history)', mode='eval').body)], body=body)]
        elif node.name == 'run_round':
            node.body = [ast.With(items=[ast.withitem(context_expr=ast.parse(
                "__profile.span('round/' + phase)", mode='eval').body)], body=node.body)]
        return node

    def visit_For(self, node):
        node = self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.target.id == 'step':
            return timed_statement(node, "__profile.span('grouped_adam/' + phase)")
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        call = node.value
        if isinstance(call, ast.Call):
            if isinstance(call.func, ast.Name) and call.func.id == 'check_candidate':
                name = ast.unparse(call.args[1])
                if name in ('original', 'best'):
                    return timed_statement(node, f"__profile.span('{name}_audit')")
            if isinstance(call.func, ast.Attribute) and call.func.attr == 'repair':
                return timed_statement(node, "__profile.span('grouped_lbfgs/' + phase)")
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        # The snap branch can be identified by the nearby explanatory layout code.
        if ast.unparse(node.test) == "phase == 'angles'" and any(
                isinstance(n, ast.Name) and n.id == 'targets' for n in ast.walk(node)):
            node.body = [ast.With(items=[ast.withitem(context_expr=ast.parse(
                "__profile.span('group_angle_snap/' + phase)", mode='eval').body)], body=node.body)]
            node.orelse = [ast.With(items=[ast.withitem(context_expr=ast.parse(
                "__profile.span('edge_snap/' + phase)", mode='eval').body)], body=node.orelse)]
        return node


class MethodMonitor:
    """Experimental policy: stop Adam+repair after W rounds without acceptance."""
    def __init__(self, window):
        self.window = window
        self.misses = defaultdict(int)
        self.rows = []

    def enabled(self, phase):
        enabled = self.misses[phase] < self.window
        if not enabled:
            self.rows.append({'phase': phase, 'skipped': True})
        return enabled

    @contextmanager
    def gradient(self, phase, history):
        count, started = len(history), time.perf_counter()
        yield
        accepted = sum(h['source'] in ('grouped_gradient', 'grouped_gradient_repaired')
                       for h in history[count:])
        self.misses[phase] = 0 if accepted else self.misses[phase]+1
        self.rows.append({'phase': phase, 'accepted': accepted,
                          'consecutive_unproductive_rounds': self.misses[phase],
                          'seconds': time.perf_counter()-started, 'skipped': False})


class StopMethod(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if node.name == 'run_round':
            split = next(i for i, statement in enumerate(node.body)
                         if isinstance(statement, ast.If)
                         and ast.unparse(statement.test) == 'time.monotonic() >= deadline')
            body = ast.With(items=[ast.withitem(context_expr=ast.parse(
                '__method_monitor.gradient(phase, history)', mode='eval').body)], body=node.body[:split])
            node.body = [ast.If(test=ast.parse('__method_monitor.enabled(phase)', mode='eval').body,
                                body=[body], orelse=[]), *node.body[split:]]
        elif node.name == 'run_gradient':
            body = ast.With(items=[ast.withitem(context_expr=ast.parse(
                '__method_monitor.gradient(phase, history)', mode='eval').body)], body=node.body)
            node.body = [ast.If(test=ast.parse('__method_monitor.enabled(phase)', mode='eval').body,
                                body=[body], orelse=[ast.Return(value=ast.Constant(value=True))])]
        return node


@contextmanager
def method_stopping(monitor, artifact_folder):
    tree = StopMethod().visit(ast.parse(inspect.getsource(alignment.align_layout)))
    ast.fix_missing_locations(tree)
    (artifact_folder/'experimental-align-layout.py').write_text(ast.unparse(tree), encoding='utf-8')
    namespace = dict(alignment.__dict__, __method_monitor=monitor)
    exec(compile(tree, str(artifact_folder/'experimental-align-layout.py'), 'exec'), namespace)
    with patch.object(postprocess, 'align_layout', namespace['align_layout']):
        yield


@contextmanager
def instrument(profiler, artifact_folder):
    tree = Instrument().visit(ast.parse(inspect.getsource(alignment.align_layout)))
    ast.fix_missing_locations(tree)
    transformed = ast.unparse(tree)
    (artifact_folder/'instrumented-align-layout.py').write_text(transformed, encoding='utf-8')
    namespace = dict(alignment.__dict__, __profile=profiler)
    exec(compile(tree, str(artifact_folder/'instrumented-align-layout.py'), 'exec'), namespace)
    with ExitStack() as patches:
        for owner, name, category in (
            (alignment, 'regularity', 'metrics'), (alignment, '_neighbors', 'neighbors'),
            (alignment, 'angle_groups', 'angle_groups'), (alignment, 'shelf_seeds', 'rows'),
            (LocalOptimizer, 'penalty', 'geometric_penalty'),
            (alignment.GroupedLayout, 'repair', 'lbfgs_compute'),
            (verify, 'validate', 'validation_2d'), (packing3d, 'verify_layout', 'validation_3d')):
            wrapped = profiler.wrap(category, getattr(owner, name))
            patches.enter_context(patch.object(owner, name, wrapped))
            if owner is alignment:
                namespace[name] = wrapped
        patches.enter_context(patch.object(postprocess, 'align_layout', namespace['align_layout']))
        yield


def audit(problem, source, result, checker):
    assert result['n'] == source['n'] == len(result['state'])
    assert result['problem'] == source['problem']
    assert result['alignment']['original_state'] == source['state']
    for key in source.keys()-{'state', 'n', 'validation', 'validation3d', 'alignment'}:
        assert result[key] == source[key], key
    assert result['alignment']['cut_schedule_verified'] is False
    validation = verify.check_candidate(problem, result['state'], checker)
    assert validation['accepted']
    before, after = regularity(problem, source['state']), regularity(problem, result['state'])
    assert angles_not_worse(after, before)
    previous = before
    for row in result['alignment']['history']:
        assert angles_not_worse(row, previous)
        if row['phase'] == 'edges':
            assert row['edge_error'] < previous['edge_error']-1e-12
        previous = row
    for phase in result['alignment']['phases']:
        from rockcut.alignment import ConvergenceMonitor
        # Replaying a score history is device independent; CPU-only audit hosts
        # must also be able to inspect results produced on CUDA.
        monitor_config = dict(result['alignment']['config'], device='cpu')
        monitor = ConvergenceMonitor(phase['before'][phase['metric']], AlignmentConfig(**monitor_config))
        for row in phase['convergence_history']:
            assert monitor.observe(row['score']) == row['stalled']
        if phase['stop_reason'] == 'plateau':
            assert monitor.history[-1]['stalled']
    for key in after:
        if isinstance(after[key], (int, float)):
            assert np.isclose(after[key], result['alignment']['after'][key], rtol=0, atol=1e-12)
    return {'passed': True, 'n': result['n'], 'validation': validation,
            'angle_guard': True, 'history_and_stops_rechecked': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', default=['granite_slab_14', 'granite_slab_22'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--mode', choices=['nested', 'ordinary', 'cprofile'], default='nested')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--steps', type=int, default=180)
    parser.add_argument('--max-rounds', type=int, default=12)
    parser.add_argument('--no-rebuild', action='store_true')
    parser.add_argument('--shelf-snapshot', type=Path,
                        help='replay saved shelf implementation in this process without changing core files')
    parser.add_argument('--method-stall-window', type=int, default=0,
                        help='separate experimental policy, ordinary timings only; 0 keeps baseline')
    parser.add_argument('--method-policy', choices=['always', 'adaptive'], default='always')
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    if args.method_stall_window < 0 or (args.method_stall_window and args.mode != 'ordinary'):
        raise ValueError('method-stall-window must be nonnegative and requires ordinary mode')
    root = args.output
    if root.exists() and any(root.iterdir()):
        raise ValueError('Refusing to overwrite populated output')
    root.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str((root/'tmp').resolve())
    Path(tempfile.tempdir).mkdir(exist_ok=True)
    torch.set_num_threads(1)
    if args.shelf_snapshot:
        snapshot = types.ModuleType('rockcut._shelf_profile_snapshot')
        snapshot.__package__ = 'rockcut'
        snapshot.__file__ = str(args.shelf_snapshot.resolve())
        exec(compile(args.shelf_snapshot.read_text(encoding='utf-8'), snapshot.__file__, 'exec'), snapshot.__dict__)
        alignment.shelf_seeds = snapshot.shelf_seeds
    cfg = AlignmentConfig(seconds=None, steps=args.steps, max_rounds=args.max_rounds,
                          rebuild=not args.no_rebuild, device=args.device,
                          method_policy='always' if args.method_stall_window else args.method_policy)
    configuration = {'config': asdict(cfg), 'mode': args.mode, 'repeats': args.repeats,
                     'experimental_method_stall_window': args.method_stall_window,
                     'shelf_snapshot': str(args.shelf_snapshot.resolve()) if args.shelf_snapshot else None,
                     'shelf_snapshot_sha256': digest(args.shelf_snapshot) if args.shelf_snapshot else None,
                     'ids': args.ids, 'environment': environment(),
                     'alignment_module': str(Path(alignment.__file__).resolve()),
                     'code_sha256': {str(p): digest(p) for p in [Path(__file__), *Path('rockcut').glob('*.py')]},
                     'suite_sha256': digest('benchmarks/performance/suite.json'),
                     'scope': 'full convergence' if args.steps == 180 and args.max_rounds == 12 and not args.no_rebuild
                              else 'short diagnostic; not full alignment or evidence of plateau'}
    save(root/'configuration.json', configuration)
    _, cases = load_controls('benchmarks/performance')
    rows = []
    for cid in args.ids:
        case = next(c for c in cases if c['id'] == cid)
        input_path = Path('benchmarks/performance/cases') / f'{cid}.json'
        input_hash = digest(input_path)
        source = {'problem': case['problem'], 'state': case['reference_state'],
                  'n': case['reference_n'], 'angle_unit': 'radians'}
        problem = Problem.from_dict(source['problem'])
        # Frozen controls contain geometry, not a solver report. Supply the
        # physical-area fields required by the normal export. No search runs
        # here; measured alignment times live in the measurement record.
        used_area = source['n'] * problem.product_area
        source.update(area_upper_bound=problem.area_upper_bound,
                      utilization=used_area / problem.area,
                      unused_area=problem.area - used_area,
                      elapsed_seconds=0., search_elapsed_seconds=0.,
                      optimality_proven=False)
        source_copy = copy.deepcopy(source)
        started = time.perf_counter()
        block, spec, piece = source_context(case, Path('benchmarks/performance'), {})
        restoration = time.perf_counter()-started
        checker = lambda q: packing3d.verify_layout(block, spec, problem.short_side, q, slab_local=piece)
        # Warmup uses the same full volumetric checker.
        started = time.perf_counter()
        postprocess.align_result(source, AlignmentConfig(steps=10, max_rounds=1, rebuild=False,
                                  device=args.device), checker)
        sync(args.device)
        warmup = time.perf_counter()-started
        for repeat in range(args.repeats):
            folder = root/f'{cid}-{repeat+1}'
            folder.mkdir()
            profiler = AlignmentProfiler()
            monitor = MethodMonitor(args.method_stall_window) if args.method_stall_window else None
            cp = cProfile.Profile() if args.mode == 'cprofile' else None
            with ExitStack() as context:
                if monitor:
                    context.enter_context(method_stopping(monitor, folder))
                if args.mode == 'nested':
                    context.enter_context(instrument(profiler, folder))
                sync(args.device)
                started = time.perf_counter()
                if cp:
                    cp.enable()
                with profiler.span('alignment_total'):
                    result = postprocess.align_result(source, cfg, checker)
                if cp:
                    cp.disable()
                sync(args.device)
                elapsed = time.perf_counter()-started
            if monitor:
                result['alignment']['experimental_method_stop'] = {
                    'policy': 'stop_gradient_after_unaccepted_rounds',
                    'window': monitor.window, 'rounds': monitor.rows,
                    'scope': 'separate experiment; not the default alignment policy'}
            if cp:
                cp.dump_stats(str(folder/'cpu-profile.pstats'))
                stream = io.StringIO()
                pstats.Stats(cp, stream=stream).sort_stats('cumtime').print_stats(60)
                (folder/'cpu-profile.txt').write_text(stream.getvalue(), encoding='utf-8')
            with profiler.span('serialization'):
                save(folder/'result.json', result)
                restored = read(folder/'result.json')
            with profiler.span('serialized_audit'):
                check = audit(problem, source, restored, checker)
                save(folder/'audit.json', check)
            with profiler.span('export'):
                write_report(result, folder/'export')
            with profiler.span('export_audit'):
                exported = read(folder/'export/result.json')
                export_check = audit(problem, source, exported, checker)
                import csv
                with (folder/'export/placements.csv').open(encoding='utf-8-sig', newline='') as f:
                    csv_rows = list(csv.DictReader(f))
                csv_state = [[float(r['center_x']), float(r['center_y']), float(r['angle_radians'])] for r in csv_rows]
                assert csv_state == exported['state']
                assert verify.check_candidate(problem, csv_state, checker)['accepted']
                save(folder/'export-audit.json', export_check)
            assert source == source_copy and digest(input_path) == input_hash
            summary = {'id': cid, 'device': args.device, 'mode': args.mode, 'repeat': repeat+1,
                       'scope': configuration['scope'], 'input_sha256': digest(input_path),
                       'configuration_sha256': digest(root/'configuration.json'),
                       'result_sha256': digest(folder/'result.json'),
                       'restoration_seconds': restoration, 'warmup_seconds': warmup,
                       'alignment_wall_seconds': elapsed, 'alignment': result['alignment'],
                       'profile': profiler.summary(), 'audits_passed': True, 'path': str(folder)}
            save(folder/'measurement.json', summary)
            rows.append(summary)
            save(root/'summary.json', rows)
            print(json.dumps({'id': cid, 'mode': args.mode, 'repeat': repeat+1, 'device': args.device,
                              'seconds': elapsed, 'after': result['alignment']['after'],
                              'profile': profiler.summary()}, ensure_ascii=True), flush=True)
    save(root/'checks.json', {'passed': True, 'runs': len(rows),
                              'source_hashes': {r['id']: r['input_sha256'] for r in rows}})


if __name__ == '__main__':
    main()
