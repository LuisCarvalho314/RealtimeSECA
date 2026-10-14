#!/usr/bin/env python3
"""Execute the fixed G/H research reconstruction; no parameter search."""
import argparse
import csv
import gzip
import json
import subprocess
from pathlib import Path
from evaluation.comparison import PUBLISHED
from evaluation.io import digest, read_json, write_json
from recompute import recompute
from run import run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def preserve(run_dir, evidence):
    """Keep ID-only assignments and scores in Git; corpus and labels stay ignored."""
    evidence.mkdir(parents=True, exist_ok=False)
    metadata = read_json(run_dir / 'run.json')
    write_json(evidence / 'run.json', metadata)
    (evidence / 'scores.csv').write_bytes((run_dir / 'scores.csv').read_bytes())
    cycles = []
    for path in sorted(run_dir.glob('assignments-*.json')):
        index = int(path.stem.split('-')[1])
        data = path.read_bytes()
        (evidence / (path.name + '.gz')).write_bytes(gzip.compress(data, mtime=0))
        cycles.append({'cycle': index+1, **read_json(run_dir / f'diagnostics-{index:04}.json'),
                       'assignment_sha256': digest(path)})
    write_json(evidence / 'cycles.json', cycles)
    write_json(evidence / 'artifact-hashes.json', {
        str(path.relative_to(run_dir)): digest(path)
        for path in sorted(run_dir.rglob('*')) if path.is_file()})
    write_json(evidence / 'independent-recomputation.json', recompute(run_dir))
    algorithm = 'SECA' if metadata['configuration']['engine']['memory_mode'] == 'Full' else 'SECA-Light'
    key = (metadata['manifest']['theme'], algorithm, metadata['clustering_procedure'])
    with (run_dir / 'scores.csv').open() as stream:
        scores = list(csv.DictReader(stream))
    comparisons = []
    for metric, published in zip(('nmi', 'bcubed_f1', 'purity'), PUBLISHED[key]):
        values = [float(row[metric]) for row in scores if row[metric]]
        # Never present an average over only the scoreable cycles as Table 3 parity.
        reproduced = sum(values)/12 if len(values) == 12 else None
        comparisons.append({'metric': metric, 'published': published, 'reproduced': reproduced,
                            'absolute_difference': abs(published-reproduced) if reproduced is not None else None,
                            'scoreable_cycles': len(values)})
    result = {'theme': key[0], 'algorithm': algorithm, 'method': key[2],
              'status': metadata['status'], 'learning_cycles': metadata['learning_cycles_completed'],
              'assignment_cycles': metadata['assignment_cycles_completed'],
              'published_comparison': comparisons, 'parity_claim': False,
              'aggregation': 'unweighted mean of all 12 current-batch cycle scores; null if any blocked',
              'source': '2023 published Table 3, p. 127025; final PDF independently verified'}
    write_json(evidence / 'summary.json', result)
    return result


def execute(theme, output, evidence, engine):
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    summaries = []
    for mode in ('Full', 'SlidingWindow'):
        for method in ('direct', 'indirect'):
            name = f'{theme}-{mode}-{method}'
            directory = output / name
            print(f'Starting {name}', flush=True)
            run(HERE / f'datasets/data/prepared/{theme}/manifest.json', HERE / f'config/{mode}.json',
                directory, method, engine, adapter=['python3', str(HERE / 'clustering/cluster.py'), '{request}'],
                engine_revision=revision, continue_blocked=True)
            summaries.append(preserve(directory, evidence / name))
            print(f"Saved {name}: {summaries[-1]['status']}", flush=True)
    write_json(evidence / f'{theme}-summary.json', summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--theme', choices=['G', 'H'], required=True)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--evidence', type=Path, default=HERE / 'evidence')
    parser.add_argument('--engine', type=Path, default=ROOT / 'target/debug/examples/paper_stream')
    args = parser.parse_args()
    execute(args.theme, args.output, args.evidence, args.engine)
