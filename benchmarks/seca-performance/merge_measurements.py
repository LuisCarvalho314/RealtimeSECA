#!/usr/bin/env python3
"""Combine same-host C# measurements with a corrected Rust adapter run."""
import argparse
import json
from pathlib import Path


def read(directory):
    provenance = json.loads((directory/'provenance.json').read_text())
    rows = [json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
    return provenance, rows


def main(base, rust, output):
    old_meta, old_rows = read(base)
    new_meta, new_rows = read(rust)
    for field in ['git_sha', 'platform', 'machine', 'processor', 'logical_cpu_count',
                  'rustc', 'cargo', 'rust_profile', 'config_sha256', 'repetitions']:
        if old_meta[field] != new_meta[field]:
            raise ValueError(f'cannot combine different {field}')
    if old_meta['dataset'] != new_meta['dataset']:
        raise ValueError('cannot combine different datasets')
    csharp = [r for r in old_rows if r['engine'] == 'csharp-original-assembly']
    rust_rows = [r for r in new_rows if r['engine'] in ('v1', 'v2')]
    expected = old_meta['repetitions'] * len(old_meta['dataset']['batch_sha256'])
    if len(csharp) != expected or len(rust_rows) != 2*expected:
        raise ValueError('incomplete measurements')
    meta = dict(old_meta)
    for field in ['rust_adapter_sha256', 'rust_binary_sha256']:
        meta.pop(field, None)
        if field in new_meta:
            meta[field] = new_meta[field]
    meta['measurement_passes'] = {
        'csharp': 'original three-way pass; unmodified original assembly adapter',
        'rust': 'separate same-host pass after moving owned SourceBatch into SecaEngine outside caller clone'}
    output.mkdir(parents=True, exist_ok=True)
    (output/'provenance.json').write_text(json.dumps(meta, indent=2)+'\n')
    (output/'runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in csharp + rust_rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('base', type=Path)
    p.add_argument('rust', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    main(a.base, a.rust, a.output)
