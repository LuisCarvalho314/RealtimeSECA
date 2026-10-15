#!/usr/bin/env python3
"""Small, standard-library SECA experiment runner. See docs/benchmark-reproduction.md."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCK = json.loads((HERE / 'datasets.json').read_text())
SETTINGS = dict(alpha=.7, beta=.5, branchMinimum=1, alphaError=.1,
                betaError=.2, wordImportanceError=.3)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def validate(data):
    """Reject input outside the shared executable contract before either process."""
    assert set(data) == set(SETTINGS) | {'batches'}, 'unexpected/missing settings'
    for key in SETTINGS:
        value = data[key]
        assert type(value) in (int, float), key
        assert value >= 0 and (key == 'branchMinimum' or value <= 1), key
    assert type(data['branchMinimum']) is int
    assert data['batches'], 'no batches'
    ids = set()
    for batch in data['batches']:
        assert batch, 'empty batch excluded by reference contract'
        for doc in batch:
            assert set(doc) == {'id', 'words'}
            sid, words = doc['id'], doc['words']
            assert type(sid) is int and 0 < sid <= 2147483647 and sid not in ids
            ids.add(sid)
            assert words and len(set(words)) == len(words), 'empty/duplicate tokens'
            assert all(isinstance(w, str) and w and w == w.strip() for w in words)


def public_input(theme, source_dir, download, smoke=False):
    spec = LOCK[theme]
    path = source_dir / spec['file']
    url = (f"https://raw.githubusercontent.com/jackyin12/MStream/{LOCK['revision']}"
           f"/MStream/data/{spec['file']}")
    if not path.exists() and download:
        payload = urllib.request.urlopen(url, timeout=60).read()
        assert digest(payload) == spec['sha256'], 'download hash mismatch'
        source_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    payload = path.read_bytes()
    assert digest(payload) == spec['sha256'], 'cached dataset hash mismatch'
    rows = [json.loads(line) for line in payload.splitlines()]
    limit = 75 if smoke else spec['documents']
    selected = rows[:limit]
    assert len(selected) == limit
    assert len({r['Id'] for r in rows}) == len(rows)
    # Positive reference IDs are a reversible ordinal mapping; labels stay out of input.
    docs = [dict(id=i + 1, words=list(dict.fromkeys(r['textCleaned'].split())))
            for i, r in enumerate(selected)]
    batch_size = 25 if smoke else spec['batch_size']
    data = dict(SETTINGS, batches=[docs[i:i + batch_size]
                                 for i in range(0, len(docs), batch_size)])
    provenance = dict(classification='comparable_with_deviations', url=url,
                      upstream_sha256=digest(payload), variant=spec['file'],
                      selection='physical-record prefix; original file order',
                      excluded_tail=len(rows) - len(selected),
                      documents=len(docs), batch_size=batch_size, batches=len(data['batches']),
                      scope='75-document smoke subset' if smoke else 'full chosen G/H reconstruction',
                      labels=len({r['clusterNo'] for r in selected}),
                      event_size_range=[min(sum(r['clusterNo'] == label for r in selected) for label in {r['clusterNo'] for r in selected}),
                                        max(sum(r['clusterNo'] == label for r in selected) for label in {r['clusterNo'] for r in selected})],
                      original_ids_sha256=digest(json.dumps([r['Id'] for r in selected]).encode()))
    # Reversible mapping retained locally only; no dataset/annotations redistributed.
    mapping = [dict(id=i + 1, source=f"mstream:{spec['file']}:{r['Id']}",
                    label=r['clusterNo']) for i, r in enumerate(selected)]
    return data, provenance, mapping


def execute(command, timeout, prefix):
    stdout, stderr = Path(str(prefix) + '.stdout'), Path(str(prefix) + '.stderr')
    status = 'ok'
    code = None
    with stdout.open('wb') as out, stderr.open('wb') as err:
        try:
            result = subprocess.run(command, stdout=out, stderr=err, timeout=timeout,
                                    cwd=ROOT, check=False)
            code = result.returncode
            status = 'ok' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            status = 'timeout'
        except OSError as error:
            status = 'unavailable'
            err.write(str(error).encode())
    return dict(status=status, exit_code=code, stdout_sha256=digest(stdout.read_bytes()),
                stderr_sha256=digest(stderr.read_bytes()),
                error=(stderr.read_text(errors='replace')[:1000] + '\n...\n' +
                       stderr.read_text(errors='replace')[-1000:]) if status != 'ok' else None), stdout


def topology_rust(tree):
    hkts = tree['hkts']
    nodes = tree['nodes']
    def scope(h):
        result = []
        for nid in h['node_ids']:
            node = next(n for n in nodes if n['node_id'] == nid)
            children = [c for c in hkts if c['parent_node_id'] == nid]
            assert len(children) <= 1
            result.append(dict(words=sorted('<>' if w['word_id'] == -1 else w['token']
                                           for w in node['words']),
                               sources=sorted(s['external_source_id'] for s in node['sources']),
                               refuge=node['is_refuge_node'],
                               child=scope(children[0]) if children else None))
        return sorted(result, key=lambda n: json.dumps(n, sort_keys=True))
    roots = [h for h in hkts if h['parent_node_id'] == 0]
    assert len(roots) == 1
    return scope(roots[0])


def topology_csharp(output, snapshot):
    vocab = {v['id']: v['word'] for v in output['vocabulary']}
    tree = snapshot['tree']
    def scope(h):
        result = []
        for node in h['nodes']:
            children = [c for c in tree if c['parentNodeId'] == node['id']]
            assert len(children) <= 1
            result.append(dict(words=sorted('<>' if w == -1 else vocab[w] for w in node['words']),
                               sources=sorted(str(s) for s in node['sources']),
                               refuge=-1 in node['words'],
                               child=scope(children[0]) if children else None))
        return sorted(result, key=lambda n: json.dumps(n, sort_keys=True))
    roots = [h for h in tree if h['parentNodeId'] == 0]
    assert len(roots) == 1
    return scope(roots[0])


def characteristics(topology):
    result = dict(hkts=0, nodes=0, refuge_nodes=0, source_memberships=0,
                  descriptor_occurrences=0, max_depth=0, represented_sources=0)
    sources = set()
    def visit(nodes, depth):
        result['hkts'] += 1
        result['max_depth'] = max(result['max_depth'], depth)
        for node in nodes:
            result['nodes'] += 1
            result['refuge_nodes'] += int(node['refuge'])
            result['source_memberships'] += len(node['sources'])
            result['descriptor_occurrences'] += 0 if node['refuge'] else len(node['words'])
            sources.update(node['sources'])
            if node['child'] is not None:
                visit(node['child'], depth + 1)
    visit(topology, 0)
    result['represented_sources'] = len(sources)
    return result


def measure(language, output_path, expected_cycles):
    if language == 'rust':
        snapshots = [json.loads(line) for line in output_path.read_text().splitlines()]
        assert [s['cycle'] for s in snapshots] == list(range(expected_cycles))
        topology = topology_rust(snapshots[-1]['tree'])
        triggers = sum(len(s['report']['reconstructed_hkt_ids'])
                       for s in snapshots[1:])
    else:
        output = json.loads(output_path.read_text())
        snapshots = output['snapshots']
        assert len(snapshots) == expected_cycles, 'incomplete reference output'
        topology = topology_csharp(output, snapshots[-1])
        triggers = sum(d['triggered'] for s in snapshots for d in s.get('decisions', []))
    return dict(characteristics(topology), reconstruction_targets=triggers), topology


def reference_provenance(reference):
    paths = list((reference / 'tools/seca-reference-harness').glob('*'))
    paths += list((reference / 'tools/seca-reference-harness/.build').glob('harness.exe'))
    paths += list((reference / 'references/seca-parity-build/.harness-generated').glob('source-hashes.json'))
    paths += [reference / 'references/seca-parity-build/SECA/bin/Debug' / name
              for name in ('SourceCleanerEngine.exe', 'Newtonsoft.Json.dll', 'Accord.dll',
                           'Accord.Math.dll', 'Accord.Math.Core.dll')]
    return {str(p.relative_to(reference)): digest(p.read_bytes())
            for p in sorted(paths) if p.is_file()}


def run_case(name, data, provenance, args, raw):
    row = dict(case=name, provenance=provenance, configuration={**SETTINGS, **data},
               classification=provenance['classification'],
               missing_measurements=['NMI', 'BCubed F1', 'purity', 'event recall'],
               missing_reason='Direct/indirect event clustering is not implemented in Rust or exposed by the reference harness.')
    row['configuration'].pop('batches')
    payload = (json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    row['input_sha256'] = digest(payload)
    path = raw / f'{name}.input.json'
    path.write_bytes(payload)
    try:
        validate(data)
    except (AssertionError, TypeError, KeyError) as error:
        row['exclusion'] = str(error)
        row['rust'] = row['csharp'] = dict(status='excluded')
        return row
    observations = {}
    commands = dict(rust=[str(args.rust.resolve()), str(path.resolve())],
                    csharp=[str(args.reference / 'tools/seca-reference-harness/run.sh'),
                            str(path.resolve())])
    for language, command in commands.items():
        result, output_path = execute(command, args.timeout, raw / f'{name}.{language}')
        row[language] = result
        if result['status'] != 'ok':
            lines = Path(str(raw / f'{name}.{language}') + '.stderr').read_text(errors='replace').splitlines()
            result['failure_summary'] = next((line for line in lines if 'Exception:' in line), None)
        if language == 'rust':
            result['emitted_cycles'] = output_path.read_bytes().count(b'\n')
        if result['status'] == 'ok':
            try:
                result['metrics'], observations[language] = measure(language, output_path, len(data['batches']))
                if args.repeat:
                    repeat, _ = execute(command, args.timeout, raw / f'{name}.{language}.repeat')
                    result['repeat_status'] = repeat['status']
                    result['deterministic_repeat'] = (repeat['status'] == 'ok' and
                                                      repeat['stdout_sha256'] == result['stdout_sha256'])
            except (ValueError, AssertionError, KeyError, StopIteration) as error:
                result.update(status='invalid_output', error=str(error))
        print(f"{name} {language}: {result['status']}", flush=True)
    if len(observations) == 2:
        row['final_labelled_topology_equal'] = observations['rust'] == observations['csharp']
        row['differences'] = {}
        for key, rust in row['rust']['metrics'].items():
            reference = row['csharp']['metrics'][key]
            row['differences'][key] = dict(rust_minus_csharp=rust - reference,
                                           relative_to_csharp=(rust - reference) / reference if reference else None)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['fixtures', 'public', 'all'], default='all')
    parser.add_argument('--case', action='append', help='exact case name; repeatable')
    parser.add_argument('--source-dir', type=Path, default=HERE / 'data')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--reference', type=Path, default=Path(os.environ.get('SECA_REFERENCE_ROOT', str(ROOT))))
    parser.add_argument('--rust', type=Path, default=ROOT / 'target/release/examples/benchmark_repro')
    parser.add_argument('--output', type=Path, required=True, help='new directory; never overwrite results')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--repeat', action='store_true', help='repeat successful runs; compare output bytes')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    args.reference = args.reference.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output / 'raw'
    raw.mkdir()
    cases = []
    fixtures = ROOT / 'crates/realtime-seca-core/tests/fixtures'
    if args.suite in ('all', 'fixtures'):
        for family in ('parity_context_metrics', 'parity_hkt_reconstruction'):
            pattern = '*.input.json' if family == 'parity_context_metrics' else '*/input.json'
            for path in sorted((fixtures / family).glob(pattern)):
                label = path.name.removesuffix('.input.json') if family == 'parity_context_metrics' else path.parent.name
                cases.append((f'{family}-{label}', path))
    if args.suite in ('all', 'public'):
        cases.extend((theme, None) for theme in ('G', 'H', 'G-smoke', 'H-smoke'))
    if args.case:
        unknown = set(args.case) - {name for name, _ in cases}
        if unknown:
            parser.error(f'unknown cases: {sorted(unknown)}')
        cases = [(name, path) for name, path in cases if name in args.case]
    report = dict(schema_version=1, engine_revision=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        tracked_changes=subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines(),
        platform=platform.platform(), python=sys.version, timeout_seconds=args.timeout,
        retention='Full / C# isLightSECA=false; no forgetting',
        reference_kind='instrumented serial original classes; not unmodified application',
        reference_sha256=reference_provenance(args.reference),
        runner_sha256=digest(Path(__file__).read_bytes()),
        cargo_lock_sha256=digest((ROOT / 'Cargo.lock').read_bytes()) if (ROOT / 'Cargo.lock').exists() else None,
        rust_toolchain=subprocess.check_output(['rustc', '--version'], text=True).strip(),
        rust_adapter_sha256=digest((ROOT / 'crates/realtime-seca-core/examples/benchmark_repro.rs').read_bytes()),
        rust_executable_sha256=digest(args.rust.read_bytes()) if args.rust.exists() else None,
        results=[])
    for name, path in cases:
        try:
            if path:
                data = json.loads(path.read_text())
                provenance = dict(classification='synthetic_characterization',
                                  path=str(path.relative_to(ROOT)), sha256=digest(path.read_bytes()))
            else:
                data, provenance, mapping = public_input(name.split('-')[0], args.source_dir, args.download, name.endswith('-smoke'))
                save(raw / f'{name}.identities.json', mapping)
            report['results'].append(run_case(name, data, provenance, args, raw))
        except (OSError, ValueError, AssertionError) as error:
            report['results'].append(dict(case=name, status='unavailable', reason=str(error)))
        save(args.output / 'report.json', report)
    with (args.output / 'metrics.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['case', 'language', 'status', 'metric', 'value', 'rust_minus_csharp', 'relative_to_csharp'])
        for row in report['results']:
            for language in ('rust', 'csharp'):
                result = row.get(language, dict(status=row.get('status', 'unavailable')))
                for metric, value in (result.get('metrics') or {'unavailable': None}).items():
                    diff = row.get('differences', {}).get(metric, {})
                    writer.writerow([row['case'], language, result['status'], metric, value,
                                     diff.get('rust_minus_csharp'), diff.get('relative_to_csharp')])
    failures = any(row.get('status') == 'unavailable' or any(
        row.get(lang, {}).get('status') not in ('ok', 'excluded') or
        row.get(lang, {}).get('deterministic_repeat') is False
        for lang in ('rust', 'csharp')) for row in report['results'])
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
