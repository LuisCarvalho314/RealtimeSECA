#!/usr/bin/env python3
"""Verify persistence and diagnose source-order sensitivity using public exports."""
import argparse
import subprocess
from pathlib import Path
from clustering.cluster import direct, indirect
from evaluation.io import read_json, write_json


def command(engine, run_dir, output, state=None, paths=None):
    inputs = paths or sorted((run_dir / 'inputs').glob('batch-*.json'))
    args = [str(engine), 'timeline-many', *map(str, inputs), '--config',
            str(run_dir / 'engine-config.json'), '--out-dir', str(output)]
    if state:
        args.extend(['--state-in', str(state)])
    subprocess.run(args, check=True)


def resume(engine, run_dir):
    result = []
    for index in (0, 5):
        output = run_dir / f'resumed-{index:04}'
        inputs = sorted((run_dir / 'inputs').glob('batch-*.json'))[:index+2]
        command(engine, run_dir, output, run_dir / f'trees/snapshot-{index:04}.json', inputs)
        for tree in sorted(output.glob('tree_batch_*.json')):
            cycle = int(tree.stem.split('_')[-1])
            snapshot = f'snapshot-{cycle:04}.json'
            if read_json(tree) != read_json(run_dir / 'trees' / tree.name):
                raise ValueError(f'resumed tree differs at {cycle}')
            if read_json(output / snapshot) != read_json(run_dir / 'trees' / snapshot):
                raise ValueError(f'resumed snapshot differs at {cycle}')
            result.append({'resumed_after_cycle': index+1, 'cycle': cycle+1,
                           'tree_equal': True, 'snapshot_equal': True})
    write_json(run_dir / 'resume-check.json', result)


def ordering(engine, run_dir):
    output = run_dir / 'ordering'
    output.mkdir()
    paths = []
    for index in range(2):
        batch = read_json(run_dir / f'inputs/batch-{index:04}.json')
        batch['sources'].reverse()
        path = output / f'batch-{index:04}.json'
        write_json(path, batch)
        paths.append(path)
    command(engine, run_dir, output / 'trees', paths=paths)
    results = []
    for index in range(2):
        batch = read_json(paths[index])
        population = [s['source_id'] for s in batch['sources']]
        tokens = {s['source_id']: s['tokens'] for s in batch['sources']}
        original = read_json(run_dir / f'trees/tree_batch_{index:04}.json')
        reversed_tree = read_json(output / f'trees/tree_batch_{index:04}.json')
        threshold = read_json(run_dir / 'run.json')['configuration']['clustering_settings']['contextual_threshold']
        for method in ('direct', 'indirect'):
            def assignments(tree):
                return (direct(tree, index, population) if method == 'direct' else
                        indirect(tree, index, population, tokens, threshold))
            def signatures(assignment):
                groups = [frozenset(c['source_ids']) for c in assignment['clusters']]
                return {s: frozenset(g for g in groups if s in g) for s in population}
            a, b = assignments(original), assignments(reversed_tree)
            sa, sb = signatures(a), signatures(b)
            results.append({'cycle': index+1, 'method': method, 'population': len(population),
                'documents_with_changed_membership_sets': sum(sa[s] != sb[s] for s in population),
                'original_clusters': len(a['clusters']), 'reversed_clusters': len(b['clusters']),
                'original_unassigned': len(a['unassigned_source_ids']),
                'reversed_unassigned': len(b['unassigned_source_ids'])})
    write_json(run_dir / 'ordering-check.json', {'operation': 'reverse source order within first two batches',
        'comparison': 'event member sets, independent of node/cluster IDs; measures changed co-memberships, not a document error rate',
        'cycles': results})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--engine', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ordering', action='store_true')
    args = parser.parse_args()
    if args.resume:
        resume(args.engine.resolve(), args.run.resolve())
    if args.ordering:
        ordering(args.engine.resolve(), args.run.resolve())
