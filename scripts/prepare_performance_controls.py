import argparse
from pathlib import Path
from rockcut.performance import build_controls

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build_controls(args.reference,args.output)
