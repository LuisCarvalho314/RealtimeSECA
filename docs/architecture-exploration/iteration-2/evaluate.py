#!/usr/bin/env python3
"""Independent process adapter comparison; no candidate imports or implementation calls."""
import json,random,subprocess,statistics,hashlib,platform
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=HERE/'evaluation-baseline/target/release/seca-exploration-baseline'
CAND=HERE/'rust-candidates/target/release/seca-research-candidates'
def records(docs):
    ids={}
    return [{'source':int(d['id']),'word':ids.setdefault(w,len(ids)+1)} for d in docs for w in dict.fromkeys(d['words'])]
def run(binary,data,extra=()):
    p=subprocess.run(['/usr/bin/time','-f','%M',str(binary),*extra],input=json.dumps(data),text=True,capture_output=True,check=True,timeout=120)
    out=json.loads(p.stdout);out['peak_rss_kib']=int(p.stderr.strip().splitlines()[-1]);return out

def cases():
    for group in ('scientific_hkt','parity_hkt_reconstruction'):
        for p in sorted((ROOT/'crates/realtime-seca-core/tests/fixtures'/group).glob('*/input.json')):
            d=json.loads(p.read_text());yield group+'/'+p.parent.name,dict(records=records(d['batches'][0]),alpha=d.get('alpha',.7),beta=d.get('beta',.5),branchMinimum=d.get('branchMinimum',1))
    rng=random.Random(20711)
    for i in range(300):
        n=rng.choice([1,3,12,30]);v=rng.choice([1,5,8]);density=rng.choice([.1,.35,.8])
        rows=[dict(source=s,word=w) for s in range(1,n+1) for w in range(1,v+1) if rng.random()<density]
        if not rows: rows=[dict(source=1,word=1)]
        rng.shuffle(rows)
        yield 'random-'+str(i),dict(records=rows,alpha=rng.choice([0.,.3,.7,1.]),beta=rng.choice([0.,.5,1.]),branchMinimum=rng.choice([1,2,4]))
def topology(scope,depth=1):
    if scope is None:return {'hkts':0,'nodes':0,'depth':0}
    children=[topology(n['child'],depth+1) for n in scope['nodes'] if n['child'] is not None]
    return {'hkts':1+sum(c['hkts'] for c in children),'nodes':len(scope['nodes'])+sum(c['nodes'] for c in children),'depth':max([depth]+[c['depth'] for c in children])}

def workloads():
    news=json.loads((ROOT/'crates/realtime-seca-core/tests/data/large_batch.json').read_text())
    for n in (25,100,250):
        docs=[dict(id=i+1,words=d['tokens']) for i,d in enumerate(news['sources'][:n])]
        yield 'archived-news-'+str(len(docs)),dict(records=records(docs),alpha=.7,beta=.5,branchMinimum=10)
    for n,v,density,minimum in ((100,40,.1,100),(1000,40,.1,1000),(1000,8,.5,20),(100,8,.5,4)):
        rng=random.Random(717);rows=[dict(source=s,word=w) for s in range(1,n+1) for w in range(1,v+1) if rng.random()<density]
        yield f'controlled-{n}-{v}-{density}-{minimum}',dict(records=rows,alpha=.7,beta=.5,branchMinimum=minimum)
def guard():
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()=="e0ca91eb62e1e5843ddb47d98ec71a388a67f6d5", "Run comparisons from pinned production baseline; see baseline.md"
    subprocess.run(["git","diff","--exit-code","HEAD","--","crates"],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)

def main():
    guard()
    checks=[]
    for name,data in cases():
        (HERE/'raw/current-construction-input.json').write_text(json.dumps({'case':name,'input':data},indent=2))
        b=run(BASE,data);row={'case':name}
        for impl,flags in [('sets',()),('masks',('--masks',))]:
            c=run(CAND,data,flags);row[impl]={'tree':b['tree']==c['tree'],'full':b['full']==c['full']}
            if b['full']!=c['full']:
                (HERE/'raw'/f'{name.replace("/","-")}-{impl}-mismatch.json').write_text(json.dumps({'input':data,'baseline':b,'candidate':c},indent=2))
        checks.append(row)
        (HERE/'raw/construction-checks.json').write_text(json.dumps(checks,indent=2))
    measurements=[]
    for name,data in workloads():
        row={'workload':name,'sources':len({r['source'] for r in data['records']}),'words':len({r['word'] for r in data['records']}),'relations':len(data['records']),'parameters':{k:v for k,v in data.items() if k!='records'},'input_sha256':hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()}
        for impl,binary,flags in [('baseline',BASE,()),('sets',CAND,()),('masks',CAND,('--masks',))]:
            trials=[run(binary,data,flags) for _ in range(5)]
            if impl=='baseline':
                row['topology']=topology(trials[0]['full'])
                expected_full=trials[0]['full']
            row[impl+'_exact_result_equal']=all(t['full']==expected_full for t in trials)
            assert row[impl+'_exact_result_equal'], 'workload differential mismatch'
            assert all(t['full']==trials[0]['full'] for t in trials), 'nondeterministic repeated trial'
            row[impl]={'core_seconds':[t['seconds'] for t in trials],'total_seconds':[t['total_seconds'] for t in trials],'peak_rss_kib':[t['peak_rss_kib'] for t in trials]}
            row[impl]['median_ms']=statistics.median(row[impl]['total_seconds'])*1000
        for impl in ('sets','masks'):row[impl]['ratio']=row[impl]['median_ms']/row['baseline']['median_ms']
        measurements.append(row)
        (HERE/'raw/construction-measurements.json').write_text(json.dumps(measurements,indent=2))
    print(json.dumps({'seed':20711,'platform':platform.platform(),'checks':checks,'measurements':measurements},indent=2))
    assert len(checks)==343 and all(c[k]['full'] and c[k]['tree'] for c in checks for k in ('sets','masks')), 'Construction mismatch; inspect captured results'
if __name__=='__main__':
    (HERE/'raw').mkdir(exist_ok=True);main()
