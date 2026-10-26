"""Summarize executed comparison evidence; no inferred/unevaluated scores."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics


def peers(assignment):
    return {s: frozenset(c['source_ids']) for c in assignment['clusters'] for s in c['source_ids']}


def summarize(directory):
    run=json.loads((directory/'results.json').read_text())
    groups=defaultdict(list)
    for row in run['results']:
        if 'engine' in row:
            groups[row['mode'],row['engine']].append(row)
    summary=[]
    for (mode,engine), rows in groups.items():
        resource=(directory/f'{mode}-{engine}-resource.txt').read_text()
        record={'mode':mode,'engine':engine,'cycles':len(rows),
                'timed_update_export_seconds':sum(r['seconds'] for r in rows),
                'process_peak_rss_kib':int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',resource)[1]),
                'final_hkts':rows[-1]['hkts'],'final_nodes':rows[-1]['nodes'],
                'final_retained':rows[-1]['retained']}
        for metric in ['nmi','bcubed_precision','bcubed_recall','bcubed_f1','purity','false_merge_pairs','fragmented_truth_pairs']:
            values=[r[metric] for r in rows if metric in r and r[metric] is not None]
            record['mean_'+metric]=statistics.mean(values) if len(values)==len(rows) else None
        summary.append(record)
    comparisons=[]
    for mode in ['Full','SlidingWindow']:
        for cycle in range(run['cycles']):
            left,right=[peers(json.loads((directory/f'{mode}-{e}'/f'assignments-{cycle:02}.json').read_text())) for e in ['v1','v2']]
            comparisons.append({'mode':mode,'cycle':cycle,
                                'changed_indirect_comember_sets':sum(left.get(s)!=right.get(s) for s in left.keys()|right.keys())})
    return {'summary':summary,'assignment_comparisons':comparisons}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize(args.directory),indent=2))
