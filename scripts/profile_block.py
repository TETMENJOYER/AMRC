"""Profile a full block search, mandatory audit and report without changing inputs."""
import argparse
import cProfile
import json
from pathlib import Path
import platform
import pstats
import time
import numpy as np
import torch

from rockcut.block_solver import BlockSearchConfig, solve_block
from rockcut.local import LocalOptimizer
from rockcut.problem import Problem
from rockcut.profiling import PhaseProfiler
from rockcut.report_block import write_block_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/performance/profile'))
    parser.add_argument('--seconds', type=float)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding='utf-8-sig'))
    config = BlockSearchConfig(**reference['config'])
    config.device = 'cpu'
    if args.seconds is not None:
        config.seconds = args.seconds
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    warm = LocalOptimizer(Problem([[0,0],[12,0],[12,4],[0,4]],1))
    warm.relax(np.array([[[0,0,.1],[2,0,.2]]]), 5)
    profiler, functions = PhaseProfiler(), cProfile.Profile()
    started = time.perf_counter()
    functions.enable()
    try:
        with profiler.instrument():
            with profiler.span('other_search'):
                result = solve_block(reference['inputs'], config,
                    progress=lambda row: print(json.dumps(row, ensure_ascii=True), flush=True))
            with profiler.span('report_and_exports'):
                report = write_block_report(result, args.output / 'run')
    finally:
        functions.disable()
        functions.dump_stats(str(args.output / 'functions.prof'))
    stats = pstats.Stats(functions)
    top = sorted(stats.stats.items(), key=lambda item: -item[1][3])[:80]
    data = {**profiler.summary(), 'total_wall_seconds': time.perf_counter() - started,
            'reference': str(args.reference), 'profiled_result': str(report),
            'python': platform.python_version(), 'torch': torch.__version__,
            'n': result['n_total'], 'accepted': result['validation']['accepted'],
            'search_seconds': result['search_seconds'], 'audit_seconds': result['verification_seconds'],
            'budget_seconds': config.seconds,
            'timing_caveat': 'cProfile and timers add overhead. The time-budgeted search may explore a different set of candidates.',
            'functions': [{'file': f[0], 'line': f[1], 'function': f[2], 'calls': v[1],
                           'self_seconds': v[2], 'cumulative_seconds': v[3]} for f,v in top]}
    (args.output / 'profile.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    with (args.output / 'functions.txt').open('w', encoding='utf-8') as stream:
        pstats.Stats(functions, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(80)
    print(json.dumps({'profile': str(args.output / 'profile.json'), 'n': data['n'], 'accepted': data['accepted']}), flush=True)


if __name__ == '__main__':
    main()
