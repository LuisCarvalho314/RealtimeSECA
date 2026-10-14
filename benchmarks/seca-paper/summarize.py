#!/usr/bin/env python3
"""Generate the eight-row published comparison without inventing blocked scores."""
import argparse
from pathlib import Path
from evaluation.io import read_json, write_json


def summarize(evidence):
    rows = [read_json(evidence / f'{theme}-{mode}-{method}/summary.json')
            for theme in ('G', 'H') for mode in ('Full', 'SlidingWindow')
            for method in ('direct', 'indirect')]
    header = '| Theme / method | Published NMI | Reproduced | Abs. Δ | Published BCubed F1 | Reproduced | Abs. Δ | Published purity | Reproduced | Abs. Δ |'
    table = [header, '|---|' + '---:|' * 9]
    for row in rows:
        values = []
        for metric in row['published_comparison']:
            values.extend('—' if metric[key] is None else f'{100*metric[key]:.2f}'
                          for key in ('published', 'reproduced', 'absolute_difference'))
        table.append('| ' + ' / '.join((row['theme'], row['algorithm'], row['method'])) + ' | ' + ' | '.join(values) + ' |')
    text = '\n'.join(table) + '\n\nScores are percentages; absolute differences are percentage points. — means no defined full-population aggregate. Direct overlaps remain unconverted. All reproduced scores are research reconstructions, not exact paper replication.\n'
    (evidence / 'published-comparison.md').write_text(text)
    write_json(evidence / 'published-comparison.json', rows)
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    print(summarize(parser.parse_args().evidence))
