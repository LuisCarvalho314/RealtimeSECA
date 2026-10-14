#!/usr/bin/env python3
"""Report metric deltas or a qualified Table 3 comparison, without declaring parity."""
import argparse
import json
from evaluation.comparison import compare_runs, compare_published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left')
    parser.add_argument('right', nargs='?')
    parser.add_argument('--published', action='store_true')
    args = parser.parse_args()
    if args.published == bool(args.right):
        parser.error('provide a second run or --published')
    result = compare_published(args.left) if args.published else compare_runs(args.left, args.right)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
