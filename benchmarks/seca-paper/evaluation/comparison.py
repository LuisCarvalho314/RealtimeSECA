"""Compare portable scores; internal HKT/node/cluster names are irrelevant."""
import csv
import math
from collections import defaultdict
from pathlib import Path
from .io import SCORES, read_json

# Table 3, final published PDF p. 127025, independently visually verified. Fractions, ordered NMI / BCubed F1 / purity.
PUBLISHED = {
    ('G', 'SECA', 'direct'): (0.8621, 0.5463, 0.7167),
    ('G', 'SECA', 'indirect'): (0.8830, 0.6303, 0.8036),
    ('G', 'SECA-Light', 'direct'): (0.8594, 0.5537, 0.7073),
    ('G', 'SECA-Light', 'indirect'): (0.8793, 0.6274, 0.7920),
    ('H', 'SECA', 'direct'): (0.7775, 0.5305, 0.7553),
    ('H', 'SECA', 'indirect'): (0.8752, 0.7114, 0.8918),
    ('H', 'SECA-Light', 'direct'): (0.8430, 0.6432, 0.8144),
    ('H', 'SECA-Light', 'indirect'): (0.8885, 0.7408, 0.8903),
}


def load_run(directory):
    directory = Path(directory)
    metadata = read_json(directory / 'run.json')
    if metadata['status'] != 'complete':
        raise ValueError('comparison requires completed runs')
    with (directory / 'scores.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    indexed = {}
    for row in rows:
        key = (row['theme'], row['algorithm'], row['clustering_procedure'], int(row['cycle']))
        if key in indexed:
            raise ValueError('duplicate score cycle')
        for metric in SCORES:
            value = float(row[metric]) if row[metric] else None
            if value is not None and not math.isfinite(value):
                raise ValueError('non-finite score')
            row[metric] = value
        indexed[key] = row
    if not indexed or len(indexed) != len(metadata['manifest']['batches']):
        raise ValueError('incomplete cycle scores')
    return metadata, indexed


def compare_runs(left, right):
    a, left_rows = load_run(left)
    b, right_rows = load_run(right)
    if left_rows.keys() != right_rows.keys():
        raise ValueError('run cycle/variant keys differ')
    contexts = ['dataset_hashes', 'metric_definitions', 'configuration', 'clustering_procedure', 'mock', 'smoke']
    changes = [field for field in contexts if a[field] != b[field]]
    differences = []
    for key in sorted(left_rows):
        x, y = left_rows[key], right_rows[key]
        for metric in SCORES:
            delta = y[metric] - x[metric] if x[metric] is not None and y[metric] is not None else None
            differences.append({'theme': key[0], 'algorithm': key[1], 'clustering_procedure': key[2],
                                'cycle': key[3], 'metric': metric, 'left': x[metric], 'right': y[metric],
                                'difference': delta})
    return {'units': 'fractions; differences are right minus left', 'context_changes': changes,
            'left_revision': a['git_revision'], 'right_revision': b['git_revision'],
            'left_engine_revision': a.get('engine_git_revision'),
            'right_engine_revision': b.get('engine_git_revision'),
            'differences': differences}


def compare_published(directory):
    metadata, rows = load_run(directory)
    groups = defaultdict(list)
    for key, row in rows.items():
        groups[key[:3]].append((key[3], row))
    result = []
    for key, cycles in sorted(groups.items()):
        if sorted(c for c, _ in cycles) != list(range(1, 13)):
            raise ValueError('Table 3 comparison requires all 12 cycles')
        if key not in PUBLISHED:
            raise ValueError('variant not reported in Table 3')
        for metric, published in zip(('nmi', 'bcubed_f1', 'purity'), PUBLISHED[key]):
            values = [r[metric] for _, r in sorted(cycles)]
            reproduced = math.fsum(values)/12 if all(v is not None for v in values) else None
            result.append({'theme': key[0], 'algorithm': key[1], 'clustering_procedure': key[2],
                           'metric': metric, 'published': published, 'reproduced': reproduced,
                           'difference': reproduced - published if reproduced is not None else None,
                           'absolute_difference': abs(reproduced - published) if reproduced is not None else None})
    return {'units': 'fractions', 'source': 'SECA published 2023 Table 3, p. 127025; PDF SHA256 dacd87d997e81820079d85d2f4fd3466705a87b7d5839ba6b98ccc5208f44947',
            'aggregation': 'unweighted cycle mean; paper does not fully specify averaging population',
            'parity_claim': False, 'purpose': metadata['configuration']['purpose'], 'mock': metadata['mock'],
            'comparisons': result}
