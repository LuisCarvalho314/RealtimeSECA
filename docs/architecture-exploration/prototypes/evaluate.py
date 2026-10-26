"""Common construction experiment. Run from any directory; requires built Rust adapter."""
import importlib.util
import hashlib
import json
from pathlib import Path
import random
import os
import statistics
import subprocess
import sys
import time
import tracemalloc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


direct = load('direct', HERE / 'direct/seca.py')
bitset = load('bitset', HERE / 'bitset/seca_bitset.py')
binary = HERE / 'rust-baseline/target/release/seca-exploration-baseline'
simplified = os.environ.get('SECA_SIMPLIFIED_BINARY')


def records(documents):
    words = {}
    return [{'source': int(d['id']), 'word': words.setdefault(w, len(words) + 1)}
            for d in documents for w in dict.fromkeys(d['words'])]


def rust(rows, a, b, minimum):
    process = subprocess.run(['/usr/bin/time', '-f', '%M', str(binary)], input=json.dumps({'records': rows,
        'alpha': a, 'beta': b, 'branchMinimum': minimum}), text=True,
        capture_output=True, check=True, timeout=60)
    result = json.loads(process.stdout)
    result['peak_rss_kib'] = int(process.stderr.strip())
    return result


def candidate(name, rows, a, b, minimum):
    if name == 'direct':
        return direct.build(rows, a, b, minimum)
    return bitset.Incidence.from_records((r['source'], r['word']) for r in rows).build(a, b, minimum)


def run():
    cases = []
    for group in ('scientific_hkt', 'parity_hkt_reconstruction'):
        for p in sorted((ROOT / 'crates/realtime-seca-core/tests/fixtures' / group).glob('*/input.json')):
            data = json.loads(p.read_text())
            cases.append((group + '/' + p.parent.name, records(data['batches'][0]),
                          data.get('alpha', .7), data.get('beta', .5), data.get('branchMinimum', 1)))
    rng = random.Random(711)
    for i in range(100):
        rows = [{'source': s, 'word': w} for s in range(1, 13) for w in range(1, 9)
                if rng.random() < .35]
        rng.shuffle(rows)
        cases.append((f'random-{i}', rows, rng.choice([.3,.7,1.]), rng.choice([0.,.5,1.]), 2))
    checks = []
    for label, rows, a, b, minimum in cases:
        expected = rust(rows, a, b, minimum)['tree']
        check = {'case': label, **{name: candidate(name, rows, a, b, minimum) == expected
                                   for name in ('direct', 'bitset')}}
        if simplified:
            saved = globals()['binary']
            globals()['binary'] = Path(simplified)
            check['simplified'] = rust(rows, a, b, minimum)['tree'] == expected
            globals()['binary'] = saved
        checks.append(check)
    news = json.loads((ROOT / 'crates/realtime-seca-core/tests/data/large_batch.json').read_text())
    workloads = []
    for count in (25, 100, 250):
        docs = [{'id': i + 1, 'words': d['tokens']} for i,d in enumerate(news['sources'][:count])]
        workloads.append((f'news-{len(docs)}', records(docs), .7, .5, 10))
    for count in (100, 1000):
        rows = [{'source': s, 'word': w} for s in range(1,count+1)
                for w in range(1,41) if (s * w) % 13 < 5]
        workloads.append((f'controlled-{count}', rows, .7, .5, 10))
    measurements = []
    for label, rows, a, b, minimum in workloads:
        baseline = [rust(rows,a,b,minimum) for _ in range(3)]
        row = {'workload':label, 'sources':len({r['source'] for r in rows}),
               'words':len({r['word'] for r in rows}), 'relations':len(rows),
               'rust_core_median_ms':statistics.median(x['seconds'] for x in baseline)*1000}
        row['rust_total_median_ms'] = statistics.median(x['total_seconds'] for x in baseline)*1000
        row['rust_peak_rss_kib'] = max(x['peak_rss_kib'] for x in baseline)
        if simplified:
            saved = globals()['binary']
            globals()['binary'] = Path(simplified)
            other = [rust(rows,a,b,minimum) for _ in range(3)]
            globals()['binary'] = saved
            row['simplified'] = {'median_ms':statistics.median(x['total_seconds'] for x in other)*1000,
                                 'peak_rss_kib':max(x['peak_rss_kib'] for x in other),
                                 'matches_rust':other[0]['tree'] == baseline[0]['tree']}
        for name in ('direct','bitset'):
            times=[]
            for _ in range(3):
                start=time.perf_counter()
                result=candidate(name,rows,a,b,minimum)
                times.append((time.perf_counter()-start)*1000)
            tracemalloc.start()
            candidate(name,rows,a,b,minimum)
            _,peak=tracemalloc.get_traced_memory()
            tracemalloc.stop()
            row[name]={'median_ms':statistics.median(times),'python_peak_bytes':peak,
                       'matches_rust':result == baseline[0]['tree']}
            probe = subprocess.run(['/usr/bin/time','-f','%M',sys.executable,str(Path(__file__)),
                                    '--worker',name], input=json.dumps([rows,a,b,minimum]),
                                   text=True,capture_output=True,check=True,timeout=60)
            row[name]['peak_rss_kib'] = int(probe.stderr.strip())
        measurements.append(row)
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT/'crates/realtime-seca-core/src').rglob('*.rs')}
    return {'contract':'ordered construction words, sources, word_sources, child; no lifecycle parity',
            'python':sys.version,'platform':sys.platform,'source_sha256':hashes,
            'seed':711,'checks':checks,'measurements':measurements}


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        candidate(sys.argv[2], *json.load(sys.stdin))
    else:
        print(json.dumps(run(), indent=2))
