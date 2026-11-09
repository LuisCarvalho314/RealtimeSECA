#!/usr/bin/env python3
"""Dependency-free SVG plots from measured JSONL rows."""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

COLORS = {'csharp-original-assembly': '#bb513b', 'v1': '#2764a5', 'v2': '#22835b'}


def median_series(rows, key, cumulative=False):
    groups = defaultdict(list)
    by_run = defaultdict(list)
    for row in rows:
        by_run[(row['engine'], row['repetition'])].append(row)
    for (engine, _), run in by_run.items():
        running = 0
        for row in sorted(run, key=lambda x: x['batch']):
            value = row.get(key)
            if value is None:
                continue
            if cumulative:
                running += value
                value = running
            groups[(engine, row['documents'])].append(value)
    out = defaultdict(list)
    for (engine, documents), values in groups.items():
        out[engine].append((documents, statistics.median(values)))
    return {engine: sorted(points) for engine, points in out.items()}


def svg(path, title, ylabel, series):
    width, height = 860, 500
    left, right, top, bottom = 75, 25, 50, 65
    points = [p for values in series.values() for p in values]
    if not points:
        return
    xmax = max(x for x, _ in points)
    ymax = max(y for _, y in points) * 1.08 or 1
    sx = lambda x: left + x / xmax * (width-left-right)
    sy = lambda y: height-bottom-y / ymax * (height-top-bottom)
    esc = lambda s: str(s).replace('&', '&amp;').replace('<', '&lt;')
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="28" font-size="20" font-family="sans-serif">{esc(title)}</text>',
             f'<path d="M {left} {top} V {height-bottom} H {width-right}" fill="none" stroke="#333"/>',
             f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif">Processed documents</text>',
             f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-family="sans-serif">{esc(ylabel)}</text>']
    for tick in range(5):
        value = tick*ymax/4
        y = sy(value)
        parts += [f'<path d="M {left} {y:.1f} H {width-right}" stroke="#ddd"/>',
                  f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2g}</text>']
    for engine, values in series.items():
        color = COLORS.get(engine, '#555')
        path_data = ' '.join(('M' if i == 0 else 'L') + f'{sx(x):.1f},{sy(y):.1f}'
                             for i, (x, y) in enumerate(values))
        parts.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        parts += [f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}"/>' for x, y in values]
    for i, engine in enumerate(series):
        parts.append(f'<text x="{width-220}" y="{top+18*i}" font-family="sans-serif" font-size="13" fill="{COLORS.get(engine, "#555")}">{esc(engine)}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts)+'\n')


def main(directory):
    rows = [json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
    charts = directory/'plots'
    charts.mkdir(exist_ok=True)
    for filename, title, key, unit, cumulative in [
        ('cumulative-runtime.svg', 'Cumulative algorithm time', 'algorithm_seconds', 'Seconds', True),
        ('batch-latency.svg', 'Incremental batch latency', 'algorithm_seconds', 'Seconds', False),
        ('peak-memory.svg', 'Process peak resident memory', 'rss_after_batch_kib', 'KiB', False),
        ('hkts.svg', 'Live HKTs', 'hkts', 'HKTs', False),
        ('nodes.svg', 'Live nodes', 'nodes', 'Nodes', False),
        ('reconstruction-frequency.svg', 'Reconstructed HKT targets per batch', 'reconstructed_hkts', 'Targets', False),
    ]:
        series = median_series(rows, key, cumulative)
        if filename == 'batch-latency.svg':
            series = {k: [(x,y) for x,y in v if x > min(p[0] for p in v)] for k,v in series.items()}
        svg(charts/filename, title, unit, series)
    v1 = dict(median_series(rows, 'algorithm_seconds', True).get('v1', []))
    v2 = dict(median_series(rows, 'algorithm_seconds', True).get('v2', []))
    svg(charts/'v1-v2-speedup.svg', 'Rust v1 to v2 cumulative speedup', 'Time ratio',
        {'v1/v2': [(x, v1[x]/v2[x]) for x in sorted(v1.keys() & v2.keys()) if v2[x] > 0]})


if __name__ == '__main__':
    main(Path(sys.argv[1]))
