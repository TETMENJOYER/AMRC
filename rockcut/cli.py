from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from .problem import Problem
from .verify import validate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid tabu + gradient rectangle packing")
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("solve", help="Find a feasible layout within a time budget")
    search.add_argument("problem", type=Path)
    search.add_argument("--output", type=Path, default=Path("outputs/latest"))
    search.add_argument("--seconds", type=float, default=120)
    search.add_argument("--seconds-per-n", type=float, default=45)
    search.add_argument("--seed", type=int, default=20261008)
    search.add_argument("--batch-size", type=int, default=12)
    search.add_argument("--adam-steps", type=int, default=180)
    search.add_argument("--polish-iterations", type=int, default=120)
    search.add_argument("--iterations-per-n", type=int, default=100)
    search.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    search.add_argument("--threads", type=int, default=1)
    search.add_argument('--alignment-seconds', type=float, default=None,
                        help='Optional safety time limit; default: no time cap, 0 disables (block: total limit)')
    search.add_argument('--alignment-stop', choices=['convergence', 'time'], default='convergence')
    search.add_argument('--alignment-method-policy', choices=['always', 'adaptive'], default='adaptive',
                        help='Adaptive gradient pauses, or the previous always-run policy')
    search.add_argument('--alignment-no-time-limit', action='store_true', help='Stop on plateau/round limit without a time cap')
    search.add_argument("--target-n", type=int, help="Attempt exactly this count")
    search.add_argument("--variant", choices=["baseline", "diverse", "repair", "adaptive", "polish_all", "combined",
                                             "restarts", "anneal"],
                        default="baseline", help="Experimental search strategy (default: original baseline)")
    slab = sub.add_parser("slab", parents=[search], add_help=False,
                          help="Pack the full-thickness region from a block/slab/product JSON")
    block = sub.add_parser("block", parents=[search], add_help=False,
                           help="Search parallel cuts and layouts, optionally in one stable vertical machine setup")
    block.add_argument('--directions',type=int,default=12)
    block.add_argument('--shortlist',type=int,default=3)
    block.add_argument('--coarse-seconds',type=float,default=.25)
    block.add_argument('--slab-seconds',type=float,default=2)
    block.add_argument('--fine-plans',type=int,default=2)
    block.add_argument('--refine-pairs',type=int,default=2)
    check = sub.add_parser("verify", help="Independently recheck an exported JSON result")
    check.add_argument("result", type=Path)
    align = sub.add_parser('align', help='Reorganize a saved layout without rerunning count search')
    align.add_argument('result', type=Path)
    align.add_argument('--output', type=Path, default=Path('outputs/aligned'))
    align.add_argument('--seconds', type=float, default=None, help='Optional safety time limit; default: no time cap')
    align.add_argument('--no-time-limit', action='store_true')
    align.add_argument('--stop-mode', choices=['convergence', 'time'], default='convergence')
    align.add_argument('--relative-gain', type=float, default=.002, help='Window relative improvement threshold (default .002 = 0.2 percent)')
    align.add_argument('--absolute-gain', type=float, default=1e-7)
    align.add_argument('--window', type=int, default=3, help='Number of complete rounds in the stagnation window')
    align.add_argument('--max-rounds', type=int, default=12, help='Protective round limit per phase')
    align.add_argument('--steps', type=int, default=180)
    align.add_argument('--method-policy', choices=['always', 'adaptive'], default='adaptive',
                       help='Adaptive gradient pauses, or the previous always-run policy')
    align.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    align.add_argument('--control', help='Frozen control ID, required for control results with external 3D verification')
    align.add_argument('--suite', type=Path, default=Path('benchmarks/performance'))
    args = parser.parse_args(argv)
    try:
        if args.command == 'align':
            from .alignment import AlignmentConfig
            from .postprocess import align_result
            data = json.loads(args.result.read_text(encoding='utf-8-sig'))
            checker = None
            if args.control:
                from .performance import load_controls, source_context
                from .packing3d import verify_layout
                _, cases = load_controls(args.suite)
                matches = [c for c in cases if c['id'] == args.control]
                if len(matches) != 1 or data['problem'] != matches[0]['problem']:
                    raise ValueError('Control missing or does not match saved problem')
                block, spec, piece = source_context(matches[0], args.suite, {})
                side = Problem.from_dict(data['problem']).short_side
                checker = lambda q: verify_layout(block, spec, side, q, slab_local=piece)
            result = align_result(data, AlignmentConfig(seconds=None if args.no_time_limit else args.seconds,
                steps=args.steps, device=args.device, stop_mode=args.stop_mode, relative_gain=args.relative_gain,
                absolute_gain=args.absolute_gain, window=args.window, max_rounds=args.max_rounds,
                method_policy=args.method_policy), checker)
            if data.get('kind') == 'block_cutting_result':
                from .report_block import write_block_report
                report = write_block_report(result, args.output)
            else:
                from .report import write_report
                report = write_report(result, args.output)
            print(f'Report: {report.resolve()}')
            return 0
        if args.command == "verify":
            data = json.loads(args.result.read_text(encoding="utf-8-sig"))
            if data.get('kind')=='block_cutting_result':
                from .verify_block import verify_block_result
                result=verify_block_result(data)
                print(json.dumps(result,ensure_ascii=False,indent=2))
                return 0 if result['accepted'] else 2
            if data.get("kind") == "slab_packing_result":
                from .packing3d import verify_slab_result
                result = verify_slab_result(data)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0 if result["accepted"] else 2
            if data.get("angle_unit") != "radians":
                raise ValueError("Expected angle_unit = radians")
            state = np.asarray(data["state"], dtype=float).reshape(-1, 3)
            if data["n"] != len(state):
                raise ValueError("Declared count differs from the number of placements")
            result = validate(Problem.from_dict(data["problem"]), state)
            # Do not trust a saved 'accepted' flag or saved rectangle vertices.
            print(json.dumps({k: v for k, v in result.items() if k != "rectangle_vertices"}, ensure_ascii=False, indent=2))
            return 0 if result["accepted"] else 2
        from .solver import SearchConfig, solve, VARIANTS
        from .report import write_report
        cfg = SearchConfig(seed=args.seed, seconds=args.seconds, seconds_per_n=args.seconds_per_n,
                           batch_size=args.batch_size, adam_steps=args.adam_steps,
                           polish_iterations=args.polish_iterations, iterations_per_n=args.iterations_per_n,
                           device=args.device, threads=args.threads,
                           alignment_seconds=None if args.alignment_no_time_limit else args.alignment_seconds,
                           alignment_stop_mode=args.alignment_stop,
                           alignment_method_policy=args.alignment_method_policy,
                           **VARIANTS[args.variant])
        def progress(event):
            print(json.dumps(event, ensure_ascii=False), flush=True)

        if args.command=='block':
            if args.target_n is not None:
                raise ValueError('--target-n applies to one slab; block search maximizes the total count')
            from .block_solver import BlockSearchConfig,solve_block
            from .report_block import write_block_report
            inputs=json.loads(args.problem.read_text(encoding='utf-8-sig'))
            block_cfg=BlockSearchConfig(seconds=args.seconds,seed=args.seed,direction_count=args.directions,
                shortlist=args.shortlist,coarse_seconds_per_slab=args.coarse_seconds,
                fine_seconds_per_slab=args.slab_seconds,fine_plans=args.fine_plans,refine_pairs=args.refine_pairs,
                batch_size=args.batch_size,adam_steps=args.adam_steps,polish_iterations=args.polish_iterations,
                iterations_per_n=args.iterations_per_n,variant=args.variant,device=args.device,threads=args.threads,
                alignment_seconds=None if args.alignment_no_time_limit else args.alignment_seconds,
                alignment_stop_mode=args.alignment_stop, alignment_method_policy=args.alignment_method_policy)
            result=solve_block(inputs,block_cfg,progress)
            report=write_block_report(result,args.output)
            print(f'Report: {report.resolve()}')
            return 0
        if args.command == "slab":
            from .packing3d import solve_slab
            inputs = json.loads(args.problem.read_text(encoding="utf-8-sig"))
            result = solve_slab(inputs, cfg, progress, args.target_n)
        else:
            result = solve(Problem.load(args.problem), cfg, progress, args.target_n)
        report = write_report(result, args.output)
        print(f"Report: {report.resolve()}")
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
