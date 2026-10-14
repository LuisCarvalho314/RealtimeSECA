#!/usr/bin/env python3
"""Independent set-based recomputation from saved assignments, without metrics.py."""
import argparse
import csv
import math
from pathlib import Path
from evaluation.io import read_json, write_json


def score(assignment, truth):
    labels = {row['source_id']: row['label'] for row in truth}
    population = set(assignment['evaluation_source_ids'])
    clusters = [set(c['source_ids']) for c in assignment['clusters'] if c['source_ids']]
    if any(sum(s in c for c in clusters) != 1 for s in population):
        return dict.fromkeys(('nmi', 'bcubed_precision', 'bcubed_recall', 'bcubed_f1', 'purity'))
    if not population:
        return dict.fromkeys(('nmi', 'bcubed_precision', 'bcubed_recall', 'bcubed_f1', 'purity'))
    classes = [{s for s in population if labels[s] == label}
               for label in sorted({labels[s] for s in population})]
    n = len(population)
    entropy = sum(-len(c)/n * math.log(len(c)/n) for c in clusters + classes)
    mi = sum(len(c & t)/n * math.log(n*len(c & t)/(len(c)*len(t)))
             for c in clusters for t in classes if c & t)
    precision = sum(len(c & t)**2/len(c) for c in clusters for t in classes)/n
    recall = sum(len(c & t)**2/len(t) for c in clusters for t in classes)/n
    f1 = sum(2*len(c & t)**2/(len(c)+len(t)) for c in clusters for t in classes)/n
    purity = sum(max(len(c & t) for t in classes) for c in clusters)/n
    return dict(nmi=2*mi/entropy if entropy else None, bcubed_precision=precision,
                bcubed_recall=recall, bcubed_f1=f1, purity=purity)


def recompute(directory):
    directory = Path(directory)
    truth = read_json(directory / 'ground-truth.json')
    with (directory / 'scores.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    checks = []
    for row in rows:
        cycle = int(row['cycle'])
        result = score(read_json(directory / f'assignments-{cycle-1:04}.json'), truth)
        for metric, value in result.items():
            saved = float(row[metric]) if row[metric] else None
            if (value is None) != (saved is None) or (value is not None and abs(value-saved) > 1e-12):
                raise ValueError(f'cycle {cycle}: {metric} disagrees: {saved} versus {value}')
        checks.append({'cycle': cycle, 'scores': result})
    return {'independent_set_calculation': True, 'cycles_verified': len(checks), 'cycles': checks}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = recompute(args.directory)
    write_json(args.directory / 'independent-recomputation.json', result)
    print(f"Verified {result['cycles_verified']} saved cycles independently")
