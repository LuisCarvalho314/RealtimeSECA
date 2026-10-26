#!/usr/bin/env python3
"""Differential normalized evolution; word IDs follow production registration order."""
import json,subprocess,random
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE/'evaluation-baseline/target/release/seca-exploration-baseline'
CAND=HERE/'evaluation-baseline/target/release/candidate_stream'
def rows(docs):return [dict(source=s,word=w) for s,words in docs for w in words]
def scenario(name,initial,batches,**kw):
    return name,dict(construction=dict(records=rows(initial),alpha=kw.pop('alpha',.7),beta=kw.pop('beta',.5),branchMinimum=kw.pop('minimum',1)),initial_index=0,gamma=kw.pop('gamma',None),alpha_error_threshold=kw.pop('ae',1.),beta_error_threshold=kw.pop('be',1.),importance_error_threshold=kw.pop('ie',1.),batches=[dict(index=i,sources=[dict(id=s,words=words) for s,words in docs]) for i,docs in batches],**kw)
def cases():
    initial=[(1,[1,2]),(2,[1,2]),(3,[3]),(4,[3,4])]
    yield scenario('stable-overlap',initial,[(1,[(5,[1,2]),(6,[3,4])]),(2,[(7,[1,2,3])])])
    yield scenario('empty-and-expiry',initial,[(1,[]),(2,[]),(3,[(1,[1,2])])],gamma=2)
    yield scenario('deduplication',initial,[(1,[(1,[4]),(5,[1]),(5,[3])]),(2,[(5,[4]),(6,[2])])])
    yield scenario('candidate-evidence',initial,[(1,[(5,[1,5]),(6,[1,5])]),(2,[(7,[1,5]),(8,[3,5])]),(3,[(9,[1,5])])])
    yield scenario('previously-expected',[(1,[1,2]),(2,[1,2]),(3,[1])],[(1,[(4,[1,2])]),(2,[(5,[2])])],alpha=.5)
    yield scenario('repeated-reconstruction',initial,[(1,[(5,[5]),(6,[5]),(7,[5])]),(2,[(8,[6]),(9,[6]),(10,[6])]),(3,[(11,[1])])],ae=0.,be=0.,ie=0.)
    yield scenario('rollback-order',initial,[(2,[(5,[1])]),(1,[(5,[1])]),(1,[(6,[1])]),(2,[])])
    yield scenario('window-candidate-readmission',initial,[(1,[(5,[1,5])]),(2,[(6,[1,5])]),(3,[(1,[1,5])]),(4,[])],gamma=2,ae=.2,be=.2,ie=.2)

    yield scenario('disjoint-child-reconstruction',[(1,[1,2]),(2,[1,2]),(3,[1]),(4,[1]),(5,[3,4]),(6,[3,4]),(7,[3]),(8,[3]),(9,[7,8]),(10,[7,8]),(11,[7]),(12,[7])],[(1,[(13,[1,5]),(14,[1,5]),(15,[1,5]),(16,[3,6]),(17,[3,6]),(18,[3,6])])],ie=.1)
    yield scenario('parent-suppression',[(1,[1,2]),(2,[1,2]),(3,[1]),(4,[1])],[(1,[(5,[3]),(6,[3]),(7,[3]),(8,[3])]),(2,[(9,[4]),(10,[4]),(11,[4]),(12,[4])])],ie=.1)
    rng=random.Random(92071)
    for trial in range(100):
        # Seed every vocabulary ID in order, preserving production registration.
        initial=[(1,list(range(1,8)))]+[(s,[w for w in range(1,8) if rng.random()<.35]) for s in range(2,9)]
        batches=[]
        for index in range(1,5):
            docs=[(index*10+j,[w for w in range(1,8) if rng.random()<.35]) for j in range(rng.randrange(5))]
            batches.append((index,docs))
        yield scenario('random-stream-'+str(trial),initial,batches,alpha=rng.choice([.3,.7,1.]),beta=rng.choice([0.,.5,1.]),minimum=2,gamma=rng.choice([None,2,3]),ae=rng.choice([0.,.2,1.]),be=rng.choice([0.,.2,1.]),ie=rng.choice([0.,.2,1.]))

def run(binary,data):
    p=subprocess.run([str(binary)],input=json.dumps(data),text=True,capture_output=True,timeout=120)
    if p.returncode:return {'process_error':p.stderr,'exit':p.returncode}
    return json.loads(p.stdout)
def main():
    from evaluate import guard
    guard()
    out=[]
    for name,data in cases():
        b=run(BASE,data);c=run(CAND,data)
        result={'case':name,'input':data,'baseline':b,'candidate':c}
        if 'states' in b and 'states' in c:
            result['state_equal']=[x==y for x,y in zip(b['states'],c['states'])]
            result['state_lengths_equal']=len(b['states'])==len(c['states'])==len(data['batches'])+1
            result['update_lengths_equal']=len(b['updates'])==len(c['updates'])==len(data['batches'])
            result['decision_equal']=[]
            for bu,cu in zip(b['updates'],c['updates']):
                same=bu['ok']==cu['ok']
                if same and bu['ok']:
                    ds=cu['decisions'];bs=bu['report']['decision_diagnostics']
                    same=len(ds)==len(bs) and all(all(bd[k]==cd[k] for k in cd) for bd,cd in zip(bs,ds))
                    same=same and sorted(d['hkt_id'] for d in ds if d['should_reconstruct'])==bu['report']['reconstructed_hkt_ids']
                elif same:same=bu['unchanged'] and cu['unchanged']
                result['decision_equal'].append(same)
        out.append(result)
    print(json.dumps(out,indent=2))
    assert all(r.get('state_lengths_equal') and r.get('update_lengths_equal') and all(r.get('state_equal',[])) and all(r.get('decision_equal',[])) for r in out), 'Evolution differential mismatch; inspect captured results'
if __name__=='__main__':main()
