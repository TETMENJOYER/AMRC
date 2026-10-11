import argparse
from pathlib import Path
from rockcut.performance import device_benchmarks, quality_benchmarks

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',type=Path,default=Path('benchmarks/performance'))
    parser.add_argument('--output',type=Path,default=Path('outputs/performance/devices'))
    parser.add_argument('--devices',nargs='+',choices=['cpu','cuda'],default=['cpu','cuda'])
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--adam-steps',type=int,default=20)
    parser.add_argument('--seeds',type=int,nargs='+',default=[101,202])
    parser.add_argument('--solver-seconds',type=float,default=6)
    parser.add_argument('--mode',choices=['all','micro','quality'],default='all')
    args=parser.parse_args()
    if args.mode in ('all','micro'):
        device_benchmarks(args.suite,args.output,args.devices,args.repeats,args.adam_steps)
    if args.mode in ('all','quality'):
        quality_benchmarks(args.suite,args.output,args.devices,args.seeds,args.solver_seconds)
