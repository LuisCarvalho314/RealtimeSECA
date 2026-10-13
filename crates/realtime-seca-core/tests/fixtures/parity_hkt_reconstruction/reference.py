"""Execute the existing C# harness twice; save actual output and normalized goldens."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def normalize(output, snapshot):
    vocab = {v['id']: v['word'] for v in output['vocabulary']}
    tree = snapshot['tree']
    visited = set()

    def scope(h, path):
        assert h['id'] not in visited, 'duplicate/cyclic attachment'
        visited.add(h['id'])
        nodes = []
        for index, n in enumerate(h['nodes']):
            children = [c for c in tree if c['parentNodeId'] == n['id']]
            assert len(children) <= 1
            nodes.append(dict(words=sorted('<>' if w == -1 else vocab[w] for w in n['words']),
                              sources=sorted(str(s) for s in n['sources']),
                              refuge=-1 in n['words'],
                              child=scope(children[0], f'{path}/{index}') if children else None))
        return dict(path=path, expected=sorted(vocab[w['wordId']] for w in h['expectedWords']), nodes=nodes)

    roots = [h for h in tree if h['parentNodeId'] == 0]
    assert len(roots) == 1
    topology = scope(roots[0], 'root')
    assert len(visited) == len(tree), 'unreachable live HKT'
    return topology


def execute(name, batches, **settings):
    counter = 0
    rows = []
    for batch in batches:
        docs = []
        for words in batch:
            counter += 1
            docs.append(dict(id=counter, words=words.split()))
        rows.append(docs)
    data = dict(alpha=.7, beta=.5, branchMinimum=1, alphaError=.1,
                betaError=.13, wordImportanceError=.3, batches=rows)
    data.update(settings)
    folder = HERE / name
    folder.mkdir(exist_ok=True)
    input_path = folder / 'input.json'
    input_path.write_text(json.dumps(data, indent=2) + '\n')
    command = [str(ROOT / 'tools/seca-reference-harness/run.sh'), str(input_path)]
    a = subprocess.run(command, capture_output=True, check=True)
    b = subprocess.run(command, capture_output=True, check=True)
    assert a.stdout == b.stdout, name
    (folder / 'csharp.json').write_bytes(a.stdout)
    output = json.loads(a.stdout)
    if name == 'child_reconstruct':
        before, after = [s['tree'] for s in output['snapshots'][:2]]
        root = next(h for h in before if h['parentNodeId'] == 0)
        sibling_parent = root['nodes'][1]['id']
        sibling = next(h for h in before if h['parentNodeId'] == sibling_parent)
        assert sibling == next(h for h in after if h['id'] == sibling['id'])
        decisions = output['snapshots'][1]['decisions']
        assert sum(d['triggered'] for d in decisions) == 1
        assert next(d for d in decisions if d['triggered'])['topologyAtDecision']['parentNodeId'] == root['nodes'][0]['id']
    if name in ('root_reconstruct', 'new_dominant', 'sequential'):
        decisions = output['snapshots'][1]['decisions']
        assert len(decisions) == 1 and decisions[0]['triggered']
    snapshots = []
    for i, snapshot in enumerate(output['snapshots']):
        decisions = []
        if i:
            previous = output['snapshots'][i-1]['tree']
            def path_of(hid):
                h = next(h for h in previous if h['id'] == hid)
                if h['parentNodeId'] == 0:
                    return 'root'
                parent = next(p for p in previous if any(n['id'] == h['parentNodeId'] for n in p['nodes']))
                idx = next(j for j,n in enumerate(parent['nodes']) if n['id'] == h['parentNodeId'])
                return f"{path_of(parent['id'])}/{idx}"
            decisions = sorted([dict(path=path_of(d['hktId']), triggered=d['triggered']) for d in snapshot['decisions']], key=lambda d:d['path'])
        snapshots.append(dict(topology=normalize(output, snapshot), decisions=decisions))
    (folder / 'csharp.normalized.json').write_text(json.dumps(snapshots, indent=2) + '\n')
    print(name, len(snapshots), 'snapshots')


if __name__ == '__main__':
    execute('single_root', [['a b','a b'], ['a b'], ['a b']])
    execute('one_prominent', [['a','a'], ['a'], ['a']])
    execute('competing', [['a','b','a b'], ['a b'], ['a']], beta=.75, branchMinimum=99)
    execute('ties_ab', [['a b'], ['a b'], ['a b']], branchMinimum=99)
    execute('ties_ba', [['b a'], ['b a'], ['b a']], branchMinimum=99)
    execute('merge_many', [['a b c','a b c'], ['a b c'], ['a b c']])
    execute('separate', [['a','a','b','b'], ['a b'], ['b']], beta=.75, branchMinimum=99)
    execute('refuge', [['a','a','a','b','b'], ['b'], ['b']], alpha=.8, alphaError=1, betaError=1, wordImportanceError=1)
    execute('unmatched', [['a','a'], ['x'], ['a']], branchMinimum=99, alphaError=1, betaError=1, wordImportanceError=1)
    execute('child', [['a b','a b','a','a'], ['a b'], ['a b']], alphaError=1, betaError=1, wordImportanceError=1)
    sibling = ['a b']*4 + ['a c']*4 + ['a']*2 + ['d e']*4 + ['d f']*4 + ['d']*2
    execute('siblings', [sibling, ['a b'], ['d e']], alphaError=1, betaError=1, wordImportanceError=1)
    execute('child_reconstruct', [sibling, ['a x']*3+['a']*7, ['a b']], alphaError=.11, betaError=.2, wordImportanceError=.2)
    execute('root_reconstruct', [['a b','a'], ['x']*4, ['x']], alphaError=.11, betaError=.2, wordImportanceError=.2)
    execute('new_dominant', [['a b','a b'], ['c']*3, ['c']])
    execute('sequential', [['a b','a b'], ['c']*3, ['c'], ['d']*8, ['d']])
    execute('collision', [['a c','a','b c','b'], ['a b c'], ['a b c']], alpha=1, branchMinimum=99, alphaError=1, betaError=1, wordImportanceError=1)
    execute('collision_reordered', [['b c','b','a c','a'], ['a b c'], ['a b c']], alpha=1, branchMinimum=99, alphaError=1, betaError=1, wordImportanceError=1)
    execute('branch_boundary', [['a b','a'], ['a b'], ['a']], branchMinimum=2, alphaError=1, betaError=1, wordImportanceError=1)
    execute('overlap_recursive', [['a b','a b','a b c','a b c','a c','a c','a'], ['a b c'], ['a b c']], alpha=.8, beta=.75, alphaError=1, betaError=1, wordImportanceError=1)
    execute('collision_true_tie', [['a','b','a c','b c'], ['a b c'], ['a b c']], alpha=1, branchMinimum=99, alphaError=1, betaError=1, wordImportanceError=1)
    execute('frequency_vs_vocabulary', [['a b','a b','b','z','z','z','z'], ['z'], ['a b']], alpha=.5, beta=1, branchMinimum=99, alphaError=1, betaError=1, wordImportanceError=1)
    hashes = json.loads((ROOT/'references/seca-parity-build/.harness-generated/source-hashes.json').read_text())
    for name in ('Program.cs', 'Execution.cs', 'prepare.py', 'build.sh', 'run.sh'):
        hashes['harness/'+name] = hashlib.sha256((ROOT/'tools/seca-reference-harness'/name).read_bytes()).hexdigest()
    (HERE/'provenance.json').write_text(json.dumps(hashes, indent=2)+'\n')
