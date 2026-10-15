"""Cross-engine representation and quality comparison through the shared adapter."""
import argparse
import gzip
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

from build import sha
from reference_execute import write
from inputs import load
from normalize import canonical, normalize, statistics
import cluster

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.metrics import evaluate, ScientificBlocker, validate_assignments

HERE = Path(__file__).resolve().parent
PAPER = {'G': {'direct': [.8621, .5463, .7167], 'indirect': [.8830, .6303, .8036]},
         'H': {'direct': [.7775, .5305, .7553], 'indirect': [.8752, .7114, .8918]}}
METRICS = ['nmi', 'bcubed_f1', 'purity']


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def compress(path, value):
    with path.open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0, filename='') as zipped:
        zipped.write(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def pairs(assignment):
    return set(itertools.chain.from_iterable(
        itertools.combinations(sorted(c['source_ids']), 2) for c in assignment['clusters']))


def agreement(left, right):
    population = left['evaluation_source_ids']
    if set(population) != set(right['evaluation_source_ids']):
        raise ValueError('different evaluation population')
    a, b = pairs(left), pairs(right)
    total = len(population)*(len(population)-1)//2
    # Retain the full family of event member sets per document, including overlap.
    def families(assignment):
        result = {s: [] for s in population}
        for c in assignment['clusters']:
            members = tuple(sorted(c['source_ids']))
            for s in members:
                result[s].append(members)
        return {s: sorted(values) for s, values in result.items()}
    fa, fb = families(left), families(right)
    changed = [s for s in population if fa[s] != fb[s]]
    return dict(population='current_batch', documents=len(population), unordered_pairs=total,
                both_comembers=len(a & b), csharp_only_comembers=len(a-b), rust_only_comembers=len(b-a),
                both_separate=total-len(a | b), changed_membership_families=len(changed),
                changed_source_ids=changed)


def score(tree, documents, index, method, truth):
    population = [s['source_id'] for s in documents]
    request = dict(tree=tree, batch_index=index, evaluation_source_ids=population,
                   source_tokens={s['source_id']: s['tokens'] for s in documents},
                   contextual_threshold='1/2', method=method)
    try:
        assignment = cluster.evaluate(request)
    except ValueError as error:
        return dict(status='clustering_input_failure', reason=str(error)), None
    memberships, _ = validate_assignments(assignment, truth)
    diagnostics = dict(overlapping_documents=sum(len(v)>1 for v in memberships.values()),
                       unassigned_documents=sum(not v for v in memberships.values()),
                       predicted_cluster_count=len(assignment['clusters']))
    try:
        return dict(status='scored', metrics=evaluate(assignment, truth), **diagnostics), assignment
    except ScientificBlocker as error:
        return dict(status='scientific_blocker', reason=str(error), **diagnostics), assignment


def differences(csharp, rust):
    def nodes(tree):
        return Counter((n['is_refuge_node'], tuple(sorted(w['token'] for w in n['words'])),
                        tuple(sorted(s['external_source_id'] for s in n['sources']))) for n in tree['nodes'])
    a, b = nodes(csharp), nodes(rust)
    return dict(unmatched_csharp_nodes=[dict(refuge=k[0], descriptors=k[1], sources=k[2], count=v)
                                        for k, v in (a-b).items()],
                unmatched_rust_nodes=[dict(refuge=k[0], descriptors=k[1], sources=k[2], count=v)
                                     for k, v in (b-a).items()])


def scopes(tree, decisions, reference=False):
    hkts, nodes, _, _, memberships, descriptors = cluster._index(tree)
    def path(hid):
        parent = hkts[hid]['parent_node_id']
        if not parent:
            return []
        return path(nodes[parent]['hkt_id']) + [dict(refuge=nodes[parent]['is_refuge_node'],
                                                    descriptors=sorted(descriptors[parent]))]
    result = []
    for d in decisions:
        hid = d['hktId'] if reference else d['hkt_id']
        result.append(dict(ancestor_descriptor_path=path(hid),
                           scope_sources_sha256=fingerprint(sorted(set().union(
                               *(memberships[n] for n in hkts[hid]['node_ids'])))),
                           triggered=d['triggered'] if reference else d['should_reconstruct']))
    return result


def representation_evidence(tree):
    hkts, nodes, _, _, memberships, descriptors = cluster._index(tree)
    violations = []
    for hid, hkt in hkts.items():
        parent = hkt['parent_node_id']
        if parent:
            for nid in hkt['node_ids']:
                excess = memberships[nid] - memberships[parent]
                if excess:
                    violations.append(dict(parent_node_id=parent, parent_is_refuge=nodes[parent]['is_refuge_node'], child_node_id=nid,
                                           excess_source_ids=sorted(excess)))
    return dict(scope_violations=violations,
                hkts=[dict(hkt_id=h['hkt_id'], parent_node_id=h['parent_node_id'],
                           node_ids=h['node_ids']) for h in hkts.values()],
                nodes=[dict(node_id=nid, refuge=n['is_refuge_node'],
                            descriptor_sha256=fingerprint(sorted(descriptors[nid])),
                            descriptor_count=len(descriptors[nid]),
                            source_ids=sorted(memberships[nid])) for nid, n in nodes.items()])


def node_co_membership(left, right, population):
    # Compare ordinary node relations for the explicit current-batch population.
    # This is a learned-representation diagnostic, not an event assignment.
    def relation(tree):
        return pairs(dict(clusters=[dict(source_ids=sorted(
            {s['external_source_id'] for s in n['sources']} & set(population)))
            for n in tree['nodes'] if not n['is_refuge_node']]))
    a, b = relation(left), relation(right)
    total = len(population)*(len(population)-1)//2
    return dict(population='current_batch', unordered_pairs=total,
                both_comembers=len(a & b), csharp_only_comembers=len(a-b),
                rust_only_comembers=len(b-a), both_separate=total-len(a | b))


def compare_cycle(theme, index, snapshots, trees, mapping, documents, truth, evidence, local, previous):
    rust = json.loads((trees / f'tree_batch_{index:04}.json').read_text())
    row = dict(cycle=index+1, rust=statistics(rust), reference_completed=index in snapshots)
    report = json.loads((trees / f'report-{index:04}.json').read_text())
    rust_decisions = report.get('decision_diagnostics', [])
    recorded = json.loads((HERE.parent / 'evidence/final' / f'{theme}-Full-indirect' / 'artifact-hashes.json').read_text())
    if sha(trees / f'report-{index:04}.json') != recorded[f'trees/report-{index:04}.json']:
        raise ValueError('Rust decision report differs from preserved benchmark')
    row['rust']['reconstruction_decisions'] = len(rust_decisions)
    row['rust']['reconstructed_scopes'] = sum(d['should_reconstruct'] for d in rust_decisions)
    if index in snapshots:
        raw = json.loads(snapshots[index].read_text())
        snapshot = raw['snapshots'][0]
        csharp = normalize(raw, snapshot, mapping)
        write(local / f'{theme}-tree-{index:04}.json', csharp)
        write(local / f'{theme}-representation-{index:04}.json', differences(csharp, rust))
        row['csharp'] = statistics(csharp)
        row['csharp']['reconstruction_decisions'] = len(snapshot.get('decisions', []))
        row['csharp']['reconstructed_scopes'] = sum(d['triggered'] for d in snapshot.get('decisions', []))
        row['representation'] = dict(
            topology_and_membership_equal=canonical(csharp)==canonical(rust),
            descriptor_topology_equal=canonical(csharp, False)==canonical(rust, False),
            csharp_canonical_sha256=fingerprint(canonical(csharp)),
            rust_canonical_sha256=fingerprint(canonical(rust)),
            csharp_descriptor_topology_sha256=fingerprint(canonical(csharp, False)),
            rust_descriptor_topology_sha256=fingerprint(canonical(rust, False)))
        row['representation']['ordinary_node_co_membership'] = node_co_membership(
            csharp, rust, [d['source_id'] for d in documents[index]])
        for engine, tree in [('csharp', csharp), ('rust', rust)]:
            ev = representation_evidence(tree)
            compress(evidence / f'{theme}-{engine}-nodes-{index:04}.json.gz', ev)
            row[engine]['scope_violation_count'] = len(ev['scope_violations'])
        if index:
            observed = dict(csharp=scopes(previous['csharp'], snapshot.get('decisions', []), True),
                            rust=scopes(previous['rust'], rust_decisions))
            write(local / f'{theme}-scopes-{index:04}.json', observed)
            row['reconstruction_scope_evidence_sha256'] = fingerprint(observed)
            write(evidence / f'{theme}-scopes-{index:04}.json', {engine: [
                d | dict(ancestor_descriptor_path=[
                    dict(refuge=p['refuge'], descriptors_sha256=fingerprint(p['descriptors']))
                    for p in d['ancestor_descriptor_path']]) for d in decisions]
                for engine, decisions in observed.items()})
    for method in ['direct', 'indirect']:
        row[method] = {}
        rs, ra = score(rust, documents[index], index, method, truth)
        saved = HERE.parent / 'evidence/final' / f'{theme}-Full-{method}' / f'assignments-{index:04}.json.gz'
        with gzip.open(saved, 'rt') as stream:
            if ra != json.load(stream):
                raise ValueError('Rust re-evaluation changed preserved assignments')
        if method == 'direct':
            memberships, _ = validate_assignments(ra, truth)
            compress(evidence / f'{theme}-rust-direct-documents-{index:04}.json.gz', memberships)
        row[method]['rust'] = rs
        if index in snapshots:
            cs, ca = score(csharp, documents[index], index, method, truth)
            row[method]['csharp'] = cs
            if ca is not None:
                if method == 'direct':
                    memberships, _ = validate_assignments(ca, truth)
                    compress(evidence / f'{theme}-csharp-direct-documents-{index:04}.json.gz', memberships)
                compress(evidence / f'{theme}-csharp-{method}-{index:04}.json.gz', ca)
            if ca is not None and ra is not None:
                row[method]['agreement'] = agreement(ca, ra)
        row[method]['metrics_comparison'] = []
        for name, published in zip(METRICS, PAPER[theme][method]):
            r = rs.get('metrics', {}).get(name)
            c = row[method].get('csharp', {}).get('metrics', {}).get(name)
            row[method]['metrics_comparison'].append(dict(metric=name,
                published_12_cycle_mean=published, rust=r, csharp=c,
                absolute_csharp_rust=None if c is None or r is None else abs(c-r),
                rust_minus_published=None if r is None else r-published,
                csharp_minus_published=None if c is None else c-published))
    if index in snapshots:
        previous['csharp'] = csharp
    previous['rust'] = rust
    return row


def compare_theme(theme, prepared, runs, rust_local, evidence, local):
    manifest, documents, truth, hashes, data, mapping = load(prepared / theme / 'manifest.json')
    if mapping != json.loads((runs / f'{theme}-mapping.json').read_text()):
        raise ValueError('stored identity mapping changed')
    compress(evidence / f'{theme}-source-mapping.json.gz', mapping)
    write(evidence / f'{theme}-dataset.json', dict(manifest=manifest, dataset_hashes=hashes))
    reference = runs / f'{theme}-full'
    execution = json.loads((reference / 'execution.json').read_text())
    if json.loads((reference / 'input.json').read_text()) != data or sha(reference / 'input.json') != execution['input_sha256']:
        raise ValueError('Reference execution input differs from converted manifest')
    for name, digest in execution['cycle_hashes'].items():
        if sha(reference / 'cycles' / name) != digest:
            raise ValueError('Reference successful snapshot changed')
    write(evidence / f'{theme}-execution.json', execution)
    (evidence / f'{theme}-stderr.log').write_bytes((reference / 'stderr.log').read_bytes())
    rust_evidence = HERE.parent / 'evidence/final' / f'{theme}-Full-indirect'
    old = json.loads((rust_evidence / 'run.json').read_text())
    if old['dataset_hashes'] != hashes or old['manifest_sha256'] != sha(prepared / theme / 'manifest.json'):
        raise ValueError('Rust evidence dataset mismatch')
    trees = rust_local / f'{theme}-Full-indirect/trees'
    # Compare against exactly the immutable trees used to obtain existing scores.
    for name, digest in old['learning_evidence']['tree_hashes'].items():
        if sha(trees / name) != digest:
            raise ValueError('Rust learned export differs from recorded benchmark')
    rows, previous = [], {}
    snapshots = {json.loads(p.read_text())['snapshots'][0]['cycle']-1: p
                 for p in (reference / 'cycles').glob('cycle-*.json')}
    for index in range(12):
        rows.append(compare_cycle(theme, index, snapshots, trees, mapping, documents,
                                  truth, evidence, local, previous))
        print(theme, 'compared cycle', index+1, flush=True)
    aggregates = {}
    for method in ['direct', 'indirect']:
        aggregates[method] = []
        for name, published in zip(METRICS, PAPER[theme][method]):
            matched = [r for r in rows if all(r[method].get(engine, {}).get('metrics', {}).get(name) is not None
                                             for engine in ['csharp', 'rust'])]
            c = mean(r[method]['csharp']['metrics'][name] for r in matched) if matched else None
            r = mean(r[method]['rust']['metrics'][name] for r in matched) if matched else None
            full = mean(row[method]['rust']['metrics'][name] for row in rows) if all(
                row[method]['rust'].get('metrics', {}).get(name) is not None for row in rows) else None
            aggregates[method].append(dict(metric=name, published=published,
                paired_cycles=[row['cycle'] for row in matched], csharp_paired_mean=c, rust_paired_mean=r,
                absolute_csharp_rust=None if c is None else abs(c-r),
                csharp_12_cycle_mean=c if len(matched)==12 else None, rust_12_cycle_mean=full,
                csharp_paired_minus_published=None if c is None else c-published,
                rust_paired_minus_published=None if r is None else r-published,
                rust_full_minus_published=None if full is None else full-published))
    summary = dict(theme=theme, engine='accumulated SECA', reference_execution_status=execution['status'],
                   completed_reference_cycles=execution['completed_cycles'],
                   first_substantive_divergence_cycle=next((r['cycle'] for r in rows
                     if r.get('representation', {}).get('topology_and_membership_equal') is False), None),
                   aggregates=aggregates,
                   interpretation='C# engine evaluated using the shared reconstructed clustering procedure',
                   aggregate_policy='Unweighted current-batch cycle means; paired scoreable cycles are not a full-stream estimate')
    write(evidence / f'{theme}-cycles.json', rows)
    write(evidence / f'{theme}-summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ['prepared', 'runs', 'rust-local', 'evidence', 'local']:
        parser.add_argument('--'+option, type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=False)
    args.local.mkdir(parents=True, exist_ok=False)
    summaries = [compare_theme(t, args.prepared, args.runs, args.rust_local, args.evidence, args.local)
                 for t in ['G', 'H']]
    write(args.evidence / 'summary.json', summaries)
    write(args.evidence / 'artifact-hashes.json', {p.name: sha(p) for p in sorted(args.evidence.iterdir()) if p.is_file()})


if __name__ == '__main__':
    main()
