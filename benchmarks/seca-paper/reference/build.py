"""Verify the existing harness, then compile output-only checkpoint instrumentation."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, output):
    output.mkdir(parents=True, exist_ok=False)
    harness = root / 'tools/seca-reference-harness'
    with (output / 'verification.log').open('w') as log:
        for command in ([str(harness / 'build.sh')],
                        ['python3', str(harness / 'verify.py')]):
            subprocess.run(command, stdout=log, stderr=log, check=True)
    reference = root / 'references/seca-parity-build'
    generated = reference / '.harness-generated'
    libraries = reference / 'SECA/bin/Debug'
    dependencies = ['SourceCleanerEngine.exe', 'Newtonsoft.Json.dll', 'Accord.dll',
                    'Accord.Math.dll', 'Accord.Math.Core.dll']
    command = ['mcs', '-langversion:7.2', f'-out:{output / "checkpoint.exe"}',
               '-r:System.Core', '-r:System.Data', '-r:System.Windows.Forms',
               '-r:System.Drawing']
    command += [f'-r:{libraries / name}' for name in dependencies]
    command += [str(p) for p in sorted(generated.glob('*.cs'))]
    command += [str(HERE / 'Checkpoint.cs'), str(harness / 'Execution.cs')]
    with (output / 'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=log, check=True)
    provenance = {
        'upstream_revision': None, 'upstream_revision_note': 'Supplied archive has no Git metadata',
        'reference_root': str(root), 'mono_path': str(libraries),
        'compiled_sources': json.loads((generated / 'source-hashes.json').read_text()),
        'generated_sources': {p.name: sha(p) for p in sorted(generated.glob('*.cs'))},
        'dependencies': {name: sha(libraries / name) for name in dependencies},
        'harness': {p.name: sha(p) for p in sorted(harness.iterdir()) if p.is_file()},
        'checkpoint_source': sha(HERE / 'Checkpoint.cs'),
        'checkpoint_binary': sha(output / 'checkpoint.exe'),
        'mono': subprocess.check_output(['mono', '--version'], text=True),
        'adaptations': ['Existing documented parity harness adaptations unchanged',
                        'Write compact per-cycle JSON after the original completed lifecycle',
                        'Do not retain all prior snapshots; atomic rename each completed output'],
    }
    original = root / 'references/seca-csharp/SECA/SECA/SECA'
    provenance['original_sources'] = {name: sha(original / name)
        for name in provenance['compiled_sources']}
    portal = root / 'references/seca-csharp/SECA/SECA/Portal/RetrospectiveAdministrator.cs'
    provenance['light_lifecycle_sources'] = {str(portal.relative_to(root)): sha(portal)}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.reference_root.resolve(), args.output.resolve())
