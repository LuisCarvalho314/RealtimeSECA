"""Run isolated engines, reuse paper clustering/scoring, normalize tree IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/seca-paper'))
from clustering.cluster import indirect, direct
from evaluation.metrics import evaluate, ScientificBlocker


def topology(tree):
    nodes = {n['node_id']: n for n in tree['nodes']}
    children = {}
    for h in tree['hkts']:
        children.setdefault(h['parent_node_id'], []).append(h)
    def visit(parent):
        return sorted([
            (tuple(sorted(w['token'] for w in nodes[n]['words'])),
             tuple(sorted(s['external_source_id'] for s in nodes[n]['sources'])),
             visit(n))
            for h in children.get(parent, []) for n in h['node_ids']
        ])
    return visit(0)


def pair_errors(assignment, truth):
    labels = {r['source_id']: r['label'] for r in truth}
    predicted = {s: c['cluster_id'] for c in assignment['clusters'] for s in c['source_ids']}
    cells = Counter((p, labels[s]) for s, p in predicted.items())
    ps, ts = Counter(predicted.values()), Counter(labels[s] for s in predicted)
    pairs = lambda counts: sum(n*(n-1)//2 for n in counts.values())
    true_positive = pairs(cells)
    return {'false_merge_pairs': pairs(ps)-true_positive,
            'fragmented_truth_pairs': pairs(ts)-true_positive}


def run(data, output, cycles):
    output.mkdir(parents=True, exist_ok=False)
    batches = sorted(data.glob('batch-*.json'))[:cycles]
    truth = json.loads((data/'ground-truth.json').read_text())
    rows = []
    for mode in ['Full', 'SlidingWindow']:
        config = json.loads((ROOT/f'benchmarks/seca-paper/config/{mode}.json').read_text())['engine']
        path = output/f'{mode}.json'
        path.write_text(json.dumps(config))
        for engine in ['v1', 'v2']:
            destination = output/f'{mode}-{engine}'
            resource = output/f'{mode}-{engine}-resource.txt'
            command = ['/usr/bin/time', '-v', '-o', str(resource),
                       str(ROOT/'target/release/examples/compare_stream'), engine,
                       str(path), str(destination), *map(str,batches)]
            subprocess.run(command, check=True)
            for index, batch_path in enumerate(batches):
                result = json.loads((destination/f'cycle-{index:02}.json').read_text())
                batch = json.loads(batch_path.read_text())
                population = [s['source_id'] for s in batch['sources']]
                tokens = {s['source_id']:s['tokens'] for s in batch['sources']}
                assignment = indirect(result['tree'], index, population, tokens, '1/2')
                (destination/f'assignments-{index:02}.json').write_text(json.dumps(assignment))
                raw_direct = direct(result['tree'], index, population)
                counts = Counter(s for c in raw_direct['clusters'] for s in c['source_ids'])
                row = {'mode':mode,'engine':engine,'cycle':index,'seconds':result['seconds'],
                       'hkts':len(result['tree']['hkts']),'nodes':len(result['tree']['nodes']),
                       'retained':len(result['tree']['source_legend']),
                       'direct_overlap_documents':sum(n>1 for n in counts.values()),
                       'direct_unassigned':len(raw_direct['unassigned_source_ids'])}
                try:
                    row |= evaluate(assignment, truth) | pair_errors(assignment, truth)
                except ScientificBlocker as error:
                    row['scientific_blocker'] = str(error)
                rows.append(row)
        for index in range(len(batches)):
            a,b = [json.loads((output/f'{mode}-{e}'/f'cycle-{index:02}.json').read_text()) for e in ['v1','v2']]
            v1=a['decisions']['decision_diagnostics']; v2=b['decisions']
            rows.append({'mode':mode,'cycle':index,'normalized_topology_equal':topology(a['tree'])==topology(b['tree']),
                         'v1_inspected':len(v1),'v2_inspected':len(v2),
                         'v1_reconstructed':sum(d['should_reconstruct'] for d in v1),
                         'v2_reconstructed':sum(d['reconstructed'] for d in v2)})
    provenance = {'dataset':str(data),'cycles':len(batches),
                  'batch_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in batches},
                  'results':rows}
    (output/'results.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--cycles',type=int,default=12)
    args=parser.parse_args()
    run(args.data.resolve(),args.output.resolve(),args.cycles)
