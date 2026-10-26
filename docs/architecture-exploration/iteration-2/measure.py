#!/usr/bin/env python3
"""Untimed allocation passes and comparable process_batch timings, separate from JSON adapters."""
import json,subprocess,statistics,hashlib
from pathlib import Path
from evaluate import HERE,BASE,workloads,guard
from evaluate_evolution import scenario
PROBE=HERE/'evaluation-baseline/target/release/probe'
STREAM=HERE/'evaluation-baseline/target/release/candidate_stream'
def run(binary,data,args=()):
    p=subprocess.run(['/usr/bin/time','-f','%M',str(binary),*args],input=json.dumps(data),text=True,capture_output=True,check=True,timeout=120)
    result=json.loads(p.stdout);result['peak_rss_kib']=int(p.stderr.strip().splitlines()[-1]);return result

def streams():
    # Two stable disjoint descriptor groups, low/high novelty and 20 batches.
    for gamma in (None,3):
        for changing in (False,True):
            initial=[(s,[1,2] if s%2 else [3,4]) for s in range(1,41)]
            batches=[]
            for index in range(1,21):
                docs=[]
                for j in range(40):
                    words=[1,2] if j%2 else [3,4]
                    if changing: words=words+[5+index]
                    docs.append((index*100+j,words))
                batches.append((index,docs))
            name=('full' if gamma is None else 'window3')+('-reconstruction' if changing else '-stable')
            yield scenario(name,initial,batches,minimum=10,gamma=gamma,ae=0. if changing else 1.,be=0. if changing else 1.,ie=0. if changing else 1.)

def main():
    guard();alloc=[]
    for name,data in workloads():
        row={'workload':name,'input_sha256':hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()}
        for mode in ('production','sets','masks'):row[mode]=run(PROBE,data,[mode])
        alloc.append(row)
    (HERE/'raw/allocation-results.json').write_text(json.dumps(alloc,indent=2)+'\n')
    observations=[]
    for name,data in streams():
        trials=[]
        for trial in range(5):
            results={}
            # Alternate order across independent process trials.
            for impl,binary in ([('production',BASE),('sets',STREAM)] if trial%2==0 else [('sets',STREAM),('production',BASE)]):
                results[impl]=run(binary,data)
            b,c=results['production'],results['sets']
            assert b['states']==c['states'],name
            item={'trial':trial,'state_equal':True}
            for impl,result in results.items():
                item[impl]={'update_seconds':result['update_seconds'],'peak_rss_kib':result['peak_rss_kib']}
            trials.append(item)
        observations.append({'workload':name,'input':data,'trials':trials})
        (HERE/'raw/stream-measurements.json').write_text(json.dumps(observations,indent=2)+'\n')
    print(json.dumps({'allocations':alloc,'streams':observations},indent=2))
if __name__=='__main__':main()
