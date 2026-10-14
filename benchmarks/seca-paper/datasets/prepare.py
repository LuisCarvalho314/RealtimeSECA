#!/usr/bin/env python3
"""Build reconstructed G/H SourceBatch JSON and isolated ground truth."""
import argparse
import os
import tempfile
from pathlib import Path

from common import (HERE, LOCK, PREPROCESSING, THEMES, encode, load_sources,
                    sha256, source_id, statistics)


def build_theme(theme, rows):
    name, size = THEMES[theme]
    count = size * 12
    if len(rows) < count:
        raise ValueError(f'{theme}: expected at least {count} documents, got {len(rows)}')
    selected = rows[:count]  # Explicit, reported reconstruction; never sort or shuffle.
    files, batches = {}, []
    ground_truth = [{'source_id': source_id(name, r['Id']), 'label': str(r['clusterNo'])}
                    for r in selected]
    truth_path = f'{theme}/ground-truth.json'
    files[truth_path] = encode(ground_truth)
    for index in range(12):
        records = selected[index * size:(index + 1) * size]
        sources = [{
            'source_id': source_id(name, r['Id']), 'batch_index': index,
            'tokens': r['textCleaned'].split(), 'text': r['textCleaned'],
            'timestamp_unix_ms': None, 'metadata': None,
        } for r in records]
        path = f'{theme}/batch-{index:02d}.json'
        files[path] = encode({'batch_index': index, 'sources': sources})
        batches.append({'batch_index': index, 'path': path, 'documents': size,
                        'source_line_start': index * size + 1,
                        'source_line_end': (index + 1) * size,
                        'sha256': sha256(files[path])})
    manifest = {
        'schema_version': 1, 'theme': theme,
        'original_dataset_name': 'News' if theme == 'G' else 'Tweets',
        'source_variant': name, 'path_base': 'selected_benchmark_data_directory',
        'source': {'repository': LOCK['repository'], 'revision': LOCK['revision'],
                   'files': LOCK['files']},
        'documents': count, 'batch_size': size, 'batch_count': 12,
        'batches': batches, 'ground_truth_path': truth_path,
        'ground_truth_sha256': sha256(files[truth_path]),
        'preprocessing': PREPROCESSING,
        'source_id_convention': f'mstream:{name}:<original Id verbatim>',
        'label_convention': 'clusterNo as decimal string, without relabeling',
        'reconstruction': {
            'selection': 'reconstructed', 'ordering': 'reconstructed',
            'exact_paper_reconstruction': False,
            'method': f'First {count} physical JSONL records of {name}, in published file order, contiguous batches.',
            'discarded_tail_documents': len(rows) - count,
            'variant_evidence': 'Prefix label count and min/max event sizes match SECA Table 2; variant is inferred, not confirmed.',
            'uncertainty': 'Paper gives no ordered ID list, exact variant filename or truncation procedure; aggregate agreement does not establish exact identity or order.',
        },
        'selected_statistics': statistics(selected),
    }
    files[f'{theme}/manifest.json'] = encode(manifest)
    return files


def prepare(source_dir, output_dir):
    rows = load_sources(source_dir)  # Verify every full file before constructing outputs.
    files = {}
    for theme, (name, _) in THEMES.items():
        files.update(build_theme(theme, rows[name]))
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f'output directory already exists: {output_dir}; choose a fresh directory')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Staging prevents input/output failures from exposing incomplete benchmark data.
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix='.seca-') as stage:
        for relative, data in files.items():
            path = Path(stage) / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        os.rename(stage, output_dir)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=HERE / 'data/raw')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'data/prepared')
    args = parser.parse_args()
    try:
        prepare(args.source_dir, args.output_dir)
    except (ValueError, OSError) as error:
        parser.exit(1, f'preparation failed: {error}\n')
    for theme, (name, size) in THEMES.items():
        print(f'{theme}: RECONSTRUCTED {name}: 12 x {size}; '
              f'explicitly excluded tail: {LOCK["files"][name]["statistics"]["documents"] - size * 12}')
    print(f'Prepared in {args.output_dir}; exact paper reconstruction is NOT established.')


if __name__ == '__main__':
    main()
