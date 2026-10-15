"""Fingerprint local payloads; publish only ID-only, authored or hash evidence."""
import argparse
import json
import subprocess
from pathlib import Path

from build import sha
from reference_execute import write
from inputs import load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def package(build, runs, normalized, evidence, prepared):
    provenance = json.loads((build / 'provenance.json').read_text())
    refroot = Path(provenance['reference_root'])
    original = refroot / 'references/seca-csharp/SECA/SECA/SECA'
    provenance['original_sources'] = {name: sha(original / name)
                                      for name in provenance['compiled_sources']}
    portal = refroot / 'references/seca-csharp/SECA/SECA/Portal/RetrospectiveAdministrator.cs'
    provenance['light_lifecycle_sources'] = {str(portal.relative_to(refroot)): sha(portal)}
    registered = json.loads((ROOT / 'crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/provenance.json').read_text())
    if any(registered.get(name) != digest for name, digest in provenance['compiled_sources'].items()):
        raise ValueError('C# compiled sources differ from executable goldens')
    provenance['compiled_sources_match_existing_goldens'] = True
    provenance['benchmark_base'] = '459117c'
    provenance['shared_scoring_sources'] = {str(p.relative_to(ROOT)): sha(p) for p in [
        HERE.parent / 'clustering/cluster.py', HERE.parent / 'evaluation/metrics.py']}
    provenance['benchmark_tooling'] = {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.is_file()}
    provenance['rust_core_changed'] = bool(subprocess.check_output(
        ['git', 'diff', '459117c', '--', 'crates/realtime-seca-core', 'crates/realtime-seca-cli'], cwd=ROOT))
    if provenance['rust_core_changed']:
        raise ValueError('reference benchmark must not change the Rust engine/tooling')
    write(evidence / 'provenance.json', provenance)
    for name in ['verification.log', 'build.log']:
        (evidence / name).write_bytes((build / name).read_bytes())
    stages = []
    for directory in sorted(p for p in runs.iterdir() if p.is_dir()):
        metadata = json.loads((directory / 'execution.json').read_text())
        if sha(directory / 'input.json') != metadata['input_sha256']:
            raise ValueError('Execution input changed')
        if metadata['checkpoint_binary_sha256'] != provenance['checkpoint_binary']:
            raise ValueError('Execution binary differs from verified build')
        for name, digest in metadata['cycle_hashes'].items():
            if sha(directory / 'cycles' / name) != digest:
                raise ValueError('Successful reference checkpoint changed')
        metadata['timeout_seconds'] = metadata.get('timeout_seconds',
            60 if directory.name.startswith('golden-') else 1800)
        stages.append(dict(stage=directory.name, **metadata))
        if directory.name not in ['G-full', 'H-full']:
            (evidence / f'{directory.name}-stderr.log').write_bytes((directory / 'stderr.log').read_bytes())
    write(evidence / 'stages.json', stages)
    for theme in ['G', 'H']:
        _, _, _, _, data, mapping = load(prepared / theme / 'manifest.json')
        if data != json.loads((runs / f'{theme}-full/input.json').read_text()):
            raise ValueError('Reference input differs from prepared manifest')
        if mapping != json.loads((runs / f'{theme}-mapping.json').read_text()):
            raise ValueError('Reference identity mapping changed')
    thresholds = {}
    for theme in ['G', 'H']:
        entries = []
        for file in sorted((runs / f'{theme}-full/cycles').glob('*.json')):
            snapshot = json.loads(file.read_text())['snapshots'][0]
            values = {tuple(d['errors'][key]['option1_Thresh'] for key in [
                'alphaError', 'betaError', 'wordImportanceError']) for d in snapshot.get('decisions', [])}
            if values and values != {(0.1, 0.2, 0.3)}:
                raise ValueError('Reference effective thresholds differ from requested values')
            entries.append(dict(cycle=snapshot['cycle'],
                                decisions=len(snapshot.get('decisions', [])), effective_thresholds=list(values)))
        thresholds[theme] = entries
    write(evidence / 'effective-thresholds.json', thresholds)
    prefixes = {name: {p.name: p.read_bytes() == (runs / 'G-full/cycles' / p.name).read_bytes()
                       for p in sorted((runs / name / 'cycles').glob('*.json'))}
                for name in ['news-first', 'news-multiple']}
    if not all(all(v.values()) for v in prefixes.values()):
        raise ValueError('Repeated News prefixes differ')
    write(evidence / 'repeated-prefixes.json', prefixes)
    payloads = {str(root): {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
                for root in [build, runs, normalized]}
    write(evidence / 'local-artifact-hashes.json', payloads)
    write(evidence / 'artifact-hashes.json', {p.name: sha(p) for p in sorted(evidence.iterdir())
                                            if p.is_file() and p.name != 'artifact-hashes.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ['build', 'runs', 'normalized', 'evidence', 'prepared']:
        parser.add_argument('--'+option, type=Path, required=True)
    args = parser.parse_args()
    package(args.build, args.runs, args.normalized, args.evidence, args.prepared)
