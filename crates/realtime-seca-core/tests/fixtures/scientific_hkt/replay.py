"""Replay each fixed input twice through the existing serial C# harness.
Does not change input fixtures, parity goldens, or Rust observations.
"""
from pathlib import Path
import json, subprocess, hashlib
HERE = Path(__file__).resolve().parent
root = HERE.parents[4]
dest = HERE

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


for p in sorted(dest.iterdir()):
    if not p.is_dir(): continue
    name = p.name
    command=[str(root/'tools/seca-reference-harness/run.sh'),str(p/'input.json')]
    a=subprocess.run(command,capture_output=True,check=True); b=subprocess.run(command,capture_output=True,check=True);assert a.stdout==b.stdout
    (p/'csharp.json').write_bytes(a.stdout)
    out=json.loads(a.stdout); paths={}; snaps=[]
    for s in out['snapshots']:
        decisions=sorted([dict(path=paths[d['hktId']],triggered=d['triggered']) for d in s.get('decisions',[])],key=lambda d:d['path'])
        top=normalize(out,s);paths={}
        def walk(h,path):
            paths[h['id']]=path
            for i,node in enumerate(h['nodes']):
                child=next((c for c in s['tree'] if c['parentNodeId']==node['id']),None)
                if child:walk(child,f'{path}/{i}')
        walk(next(h for h in s['tree'] if h['parentNodeId']==0),'root')
        snaps.append(dict(topology=top,decisions=decisions))
    (p/'csharp.normalized.json').write_text(json.dumps(snaps,indent=2)+'\n')
    print(name)
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'tools/seca-reference-harness').glob('*') if p.is_file()}
hashes.update(json.loads((root/'tools/seca-reference-harness/.build/source-hashes.json').read_text()))
(dest/'provenance.json').write_text(json.dumps(hashes,indent=2)+'\n')
provenance = json.loads((dest/'provenance.json').read_text())
for relative in ['crates/realtime-seca-core/src/tree/hkt_builder.rs', 'crates/realtime-seca-core/src/engine/rebuild.rs', 'crates/realtime-seca-core/src/engine/baseline.rs', 'crates/realtime-seca-core/tests/scientific_hkt_contract.rs', 'crates/realtime-seca-core/tests/fixtures/scientific_hkt/replay.py']:
    provenance[relative] = hashlib.sha256((root/relative).read_bytes()).hexdigest()
provenance['environment'] = dict(mono=subprocess.check_output(['mono','--version'],text=True).splitlines()[0],
    rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
(dest/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
