"""Staged reference execution; keep successful cycles and surfaced failures."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from build import sha
from inputs import load


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def run(build, data, output, timeout):
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'input.json', data)
    provenance = json.loads((build / 'provenance.json').read_text())
    command = ['mono', str(build / 'checkpoint.exe'), str(output / 'input.json'),
               str(output / 'cycles')]
    started = time.monotonic()
    with (output / 'stderr.log').open('w') as stderr, (output / 'stdout.log').open('w') as stdout:
        try:
            completed = subprocess.run(command, env=os.environ | dict(MONO_PATH=provenance['mono_path']),
                                       stdout=stdout, stderr=stderr, timeout=timeout)
            code, status = completed.returncode, 'complete' if completed.returncode == 0 else 'reference_failure'
        except subprocess.TimeoutExpired:
            code, status = None, 'resource_timeout'
    snapshots = sorted((output / 'cycles').glob('cycle-*.json'))
    result = dict(status=status, exit_code=code, command=command,
                  elapsed_seconds=time.monotonic()-started, timeout_seconds=timeout, requested_cycles=len(data['batches']),
                  completed_cycles=len(snapshots),
                  first_incomplete_cycle=len(snapshots)+1 if status != 'complete' else None,
                  input_sha256=sha(output / 'input.json'),
                  checkpoint_binary_sha256=provenance['checkpoint_binary'],
                  compiled_sources=provenance['compiled_sources'],
                  cycle_hashes={p.name: sha(p) for p in snapshots},
                  stderr_sha256=sha(output / 'stderr.log'))
    write(output / 'execution.json', result)
    print(output.name, status, len(snapshots), 'cycles', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = json.loads((args.build / 'provenance.json').read_text())
    root = Path(provenance['reference_root'])
    for name in ['stable', 'reconstruct']:
        data = json.loads((root / f'tools/seca-reference-harness/fixtures/{name}.json').read_text())
        run(args.build, data, args.output / f'golden-{name}', 60)
    failure = Path(__file__).resolve().parents[3] / 'crates/realtime-seca-core/tests/fixtures/parity_context_metrics/shared_candidate.input.json'
    run(args.build, json.loads(failure.read_text()), args.output / 'golden-shared-candidate-failure', 60)
    converted = {}
    for theme in ['G', 'H']:
        manifest, documents, _, hashes, data, mapping = load(args.prepared / theme / 'manifest.json')
        converted[theme] = data
        write(args.output / f'{theme}-mapping.json', mapping)
        write(args.output / f'{theme}-dataset.json', dict(manifest=manifest, dataset_hashes=hashes))
    data = converted['G']
    run(args.build, data | dict(batches=[data['batches'][0][:32]]), args.output / 'news-small', args.timeout)
    run(args.build, data | dict(batches=[data['batches'][0]]), args.output / 'news-first', args.timeout)
    run(args.build, data | dict(batches=data['batches'][:3]), args.output / 'news-multiple', args.timeout)
    for theme, data in converted.items():
        run(args.build, data, args.output / f'{theme}-full', args.timeout)


if __name__ == '__main__':
    main()
