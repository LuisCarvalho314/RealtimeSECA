"""Strict JSON and portable, deterministic result persistence."""
import csv
import hashlib
import json
from pathlib import Path

SCORES = ['nmi', 'bcubed_precision', 'bcubed_recall', 'bcubed_f1', 'purity']
FIELDS = ['theme', 'algorithm', 'clustering_procedure', 'cycle',
          'evaluated_document_count', 'predicted_cluster_count', 'ground_truth_event_count'] + SCORES


def parse_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def read_json(path):
    return parse_json(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_scores(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
