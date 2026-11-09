#!/usr/bin/env python3
"""Regenerate aggregates from raw per-run observations."""
import csv
import json
import sys
from pathlib import Path
from run import summarize


def main(directory):
    rows = [json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
    summary = summarize(rows)
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (directory/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    by_engine = {s['engine']: s for s in summary}
    ratios = {}
    for label, a, b in [('csharp-to-v1','csharp-original-assembly','v1'),
                        ('v1-to-v2','v1','v2'),
                        ('csharp-to-v2','csharp-original-assembly','v2')]:
        if a in by_engine and b in by_engine:
            ratios[label] = {
                'algorithm_speedup': by_engine[a]['total_median_seconds']/by_engine[b]['total_median_seconds'],
                'peak_rss_ratio': by_engine[a]['peak_rss_median_kib']/by_engine[b]['peak_rss_median_kib']}
    (directory/'ratios.json').write_text(json.dumps(ratios, indent=2)+'\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
