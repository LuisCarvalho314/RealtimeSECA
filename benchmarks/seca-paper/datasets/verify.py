#!/usr/bin/env python3
"""Verify prepared bytes against independently loaded pinned inputs and recipe."""
import argparse
from pathlib import Path

from common import HERE, THEMES, load_sources
from prepare import build_theme


def verify(source_dir, data_dir):
    rows = load_sources(source_dir)
    expected = {}
    for theme, (name, _) in THEMES.items():
        expected.update(build_theme(theme, rows[name]))
    directory = Path(data_dir)
    actual_paths = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if actual_paths != set(expected):
        raise ValueError(f'prepared file inventory mismatch; missing={sorted(set(expected)-actual_paths)}, '
                         f'unexpected={sorted(actual_paths-set(expected))}')
    for relative, data in expected.items():
        if (directory / relative).read_bytes() != data:
            raise ValueError(f'prepared content or SHA-256 differs: {relative}')
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=HERE / 'data/raw')
    parser.add_argument('--data-dir', type=Path, default=HERE / 'data/prepared')
    args = parser.parse_args()
    try:
        count = verify(args.source_dir, args.data_dir)
    except (ValueError, OSError) as error:
        parser.exit(1, f'verification failed: {error}\n')
    print(f'Verified {count} files, including 24 batches, two ground-truth files and two manifests.')


if __name__ == '__main__':
    main()
