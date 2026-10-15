"""Render per-cycle representation and shared-evaluator comparison tables."""
import argparse
import csv
import json
from pathlib import Path

from build import sha
from reference_execute import write


def percentage(value):
    return '—' if value is None else f'{100*value:.2f}'


def csv_file(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def render(evidence):
    quality, representations, aggregate = [], [], []
    text = ['# Shared reconstructed evaluator: C# versus Rust', '',
            'Scores are percentages; differences are percentage points. Published values are',
            '12-cycle means, including in per-cycle rows. — denotes an unavailable metric.', '',
            '| Theme / metric | Scoreable paired cycles | C# paired | Rust paired | Abs. Δ | Published | C# paired − published | Rust paired − published | Rust full |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for theme in ['G', 'H']:
        summary = json.loads((evidence / f'{theme}-summary.json').read_text())
        cycles = json.loads((evidence / f'{theme}-cycles.json').read_text())
        for method, rows in summary['aggregates'].items():
            for row in rows:
                aggregate.append(dict(theme=theme, method=method, **row))
                if method == 'indirect':
                    values = [row[k] for k in ['csharp_paired_mean', 'rust_paired_mean',
                        'absolute_csharp_rust', 'published', 'csharp_paired_minus_published',
                        'rust_paired_minus_published', 'rust_12_cycle_mean']]
                    text.append(f"| {theme} / {row['metric']} | {','.join(map(str,row['paired_cycles']))} | " +
                                ' | '.join(map(percentage, values)) + ' |')
        for row in cycles:
            representations.append(dict(theme=theme, cycle=row['cycle'], **{
                engine+'_'+key: row.get(engine, {}).get(key) for engine in ['csharp', 'rust']
                for key in ['hkt_count', 'node_count', 'hierarchy_depth', 'ordinary_node_count',
                            'refuge_node_count', 'reconstruction_decisions', 'reconstructed_scopes',
                            'scope_violation_count']},
                descriptor_topology_equal=row.get('representation', {}).get('descriptor_topology_equal'),
                topology_and_membership_equal=row.get('representation', {}).get('topology_and_membership_equal')))
            for method in ['direct', 'indirect']:
                for metric in row[method]['metrics_comparison']:
                    quality.append(dict(theme=theme, cycle=row['cycle'], method=method, **metric,
                                        csharp_status=row[method].get('csharp', {}).get('status', 'not_completed'),
                                        rust_status=row[method]['rust']['status']))
    text += ['', 'Partial paired means are not estimates of missing cycles or replacements for the published average.', '',
             '| Theme / cycle | C# NMI | Rust NMI | C# F1 | Rust F1 | C# purity | Rust purity | C# status |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for theme in ['G', 'H']:
        for row in json.loads((evidence / f'{theme}-cycles.json').read_text()):
            vals = [row['indirect'].get(e, {}).get('metrics', {}).get(m)
                    for m in ['nmi', 'bcubed_f1', 'purity'] for e in ['csharp', 'rust']]
            status = row['indirect'].get('csharp', {}).get('status', 'not_completed')
            text.append(f"| {theme} / {row['cycle']} | " + ' | '.join(map(percentage, vals)) + f' | {status} |')
    csv_file(evidence / 'per-cycle-metrics.csv', quality)
    csv_file(evidence / 'per-cycle-representations.csv', representations)
    csv_file(evidence / 'aggregate-metrics.csv', aggregate)
    (evidence / 'comparison.md').write_text('\n'.join(text) + '\n')
    write(evidence / 'artifact-hashes.json', {p.name: sha(p) for p in sorted(evidence.iterdir())
                                            if p.is_file() and p.name != 'artifact-hashes.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    render(parser.parse_args().evidence)
