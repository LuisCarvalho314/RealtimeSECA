#!/usr/bin/env python3
"""Run all deterministic synthetic factors sequentially."""
import argparse
from pathlib import Path
import subprocess
import sys

from synthetic import SCENARIOS, generate

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=10)
    args = parser.parse_args()
    generate(args.input_dir)
    for name in SCENARIOS:
        output = args.output_dir / name
        subprocess.run([sys.executable, str(HERE/'run.py'), '--manifest',
                        str(args.input_dir/name/'manifest.json'), '--batches', '12',
                        '--repetitions', str(args.repetitions), '--warmup', '1',
                        '--output', str(output)], check=True)
        for script in ('summarize.py', 'plot.py'):
            subprocess.run([sys.executable, str(HERE/script), str(output)], check=True)
        subprocess.run([sys.executable, str(HERE/'verify_inputs.py'), '--manifest',
                        str(args.input_dir/name/'manifest.json'), '--batches', '12',
                        '--output', str(output/'input-verification.json')], check=True)


if __name__ == '__main__':
    main()
