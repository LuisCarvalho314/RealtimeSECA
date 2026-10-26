#!/usr/bin/env python3
"""Read-only archived container fixtures, preserved as normalized reproducible inputs."""
import json,hashlib
from pathlib import Path
from evaluate import HERE,BASE,guard
from evaluate_evolution import scenario,run,CAND
ARCHIVE=Path('/home/azureuser/risklive-dev/.container-test/seca-validation/data/runtime/seca')
def prepare():
    prepared=[]
    for gamma in (3,7,30):
        vocab={};sources={};batches=[];provenance=[]
        for day in ('14','15'):
            path=ARCHIVE/f'relevant_news_data_{gamma}d_2026-01-{day}_batch.json'
            raw=path.read_bytes();batch=json.loads(raw)
            provenance.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()})
            if not vocab:
                for s in batch['sources']:
                    for w in s['tokens']:
                        w=w.strip()
                        if w:vocab.setdefault(w,len(vocab)+1)
            else:
                for w in sorted({w.strip() for s in batch['sources'] for w in s['tokens'] if w.strip()}):vocab.setdefault(w,len(vocab)+1)
            docs=[(sources.setdefault(s['source_id'],len(sources)+1),[vocab[w.strip()] for w in s['tokens'] if w.strip()]) for s in batch['sources']]
            batches.append((batch['batch_index'],docs))
        name,data=scenario(f'container-validation-{gamma}d',batches[0][1],batches[1:],gamma=gamma,minimum=10)
        prepared.append({'name':name,'input':data,'provenance':provenance,'source_count':[len(x[1]) for x in batches]})
    return prepared

def main():
    guard();saved=HERE/'raw/archived-container-inputs.json'
    data=json.loads(saved.read_text()) if saved.exists() else prepare()
    saved.write_text(json.dumps(data,indent=2)+'\n')
    results=[]
    for d in data:
        b,c=run(BASE,d['input']),run(CAND,d['input'])
        equal=b['states']==c['states'];assert equal,d['name']
        results.append({'case':d['name'],'source_count':d['source_count'],'exact_state_equal':equal,'baseline':b,'candidate':c})
    (HERE/'raw/archived-container-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('3 archived container fixture sequences passed exact normalized state comparison')
if __name__=='__main__':main()
