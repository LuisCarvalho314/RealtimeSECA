#!/usr/bin/env python3
"""Deterministic stress streams; factors are input controls, not claimed tree depth."""
import argparse
import hashlib
import json
import random
from pathlib import Path

SCENARIOS = {
    'base': dict(per_batch=100, vocabulary=50, overlap=.5, levels=2, drift=0),
    'more-documents': dict(per_batch=200, vocabulary=50, overlap=.5, levels=2, drift=0),
    'larger-vocabulary': dict(per_batch=100, vocabulary=200, overlap=.5, levels=2, drift=0),
    'higher-overlap': dict(per_batch=100, vocabulary=50, overlap=.85, levels=2, drift=0),
    'deeper-input': dict(per_batch=100, vocabulary=50, overlap=.5, levels=5, drift=0),
    'frequent-drift': dict(per_batch=100, vocabulary=50, overlap=.5, levels=2, drift=.75),
}


def generate(root, batches=12):
    for name, p in SCENARIOS.items():
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        rng = random.Random(90210)
        records = []
        for batch in range(batches):
            sources = []
            for index in range(p['per_batch']):
                cluster = index % 4
                words = set()
                if rng.random() < p['overlap']:
                    for level in range(p['levels']):
                        words.add(f'cluster{cluster}-level{level}')
                for _ in range(3):
                    words.add(f'background{rng.randrange(p["vocabulary"])}')
                if p['drift'] and rng.random() < p['drift']:
                    words.add(f'epoch{batch}-cluster{cluster}')
                sid = f'synthetic:{name}:{batch:02}:{index:04}'
                sources.append({'source_id': sid, 'batch_index': batch,
                                'tokens': sorted(words), 'text': None,
                                'timestamp_unix_ms': None, 'metadata': None})
            path = folder / f'batch-{batch:02}.json'
            raw = json.dumps({'batch_index': batch, 'sources': sources},
                             separators=(',', ':')).encode()
            path.write_bytes(raw)
            records.append({'batch_index': batch, 'path': path.name,
                            'documents': len(sources), 'sha256': hashlib.sha256(raw).hexdigest()})
        manifest = {'theme': f'synthetic-{name}', 'source_variant': 'generated-seed-90210',
                    'reconstruction': {'recipe': p, 'random_seed': 90210,
                                       'status': 'synthetic stress stream'},
                    'batches': records}
        (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batches', type=int, default=12)
    args = parser.parse_args()
    generate(args.output_dir, args.batches)
