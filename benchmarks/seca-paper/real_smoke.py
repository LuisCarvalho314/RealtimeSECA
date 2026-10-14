#!/usr/bin/env python3
"""Select two disjoint repeated texts without consulting annotations; preserve IDs."""
import argparse
from collections import defaultdict
from pathlib import Path
from evaluation.contracts import load_manifest
from evaluation.io import digest, write_json


def prepare(manifest_path, output):
    manifest, batches, truth, _ = load_manifest(manifest_path)
    groups = defaultdict(list)
    for batch in batches:
        for source in batch:
            groups[tuple(source['tokens'])].append(source)
    selected, vocabulary = [], set()
    for tokens, sources in groups.items():
        if len(sources) >= 4 and not vocabulary.intersection(tokens):
            selected.append(sources[:4])
            vocabulary.update(tokens)
            if len(selected) == 2:
                break
    if len(selected) != 2:
        raise ValueError('real corpus lacks two disjoint repeated texts')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    entries, identities = [], set()
    for index in range(2):
        sources = [dict(source, batch_index=index) for group in selected for source in group[index*2:index*2+2]]
        identities.update(source['source_id'] for source in sources)
        path = output / f'batch-{index:04}.json'
        write_json(path, {'batch_index': index, 'sources': sources})
        entries.append({'batch_index': index, 'path': path.name, 'sha256': digest(path)})
    write_json(output / 'truth.json', [row for row in truth if row['source_id'] in identities])
    write_json(output / 'manifest.json', {'schema_version': 1, 'theme': 'G',
        'path_base': 'manifest_directory', 'batches': entries, 'ground_truth_path': 'truth.json',
        'ground_truth_sha256': digest(output / 'truth.json'), 'parent_manifest_sha256': digest(manifest_path),
        'source': manifest['source'], 'selection': 'first two repeated texts with disjoint token sets; four original documents per text; no labels consulted'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.manifest, args.output)
