#!/usr/bin/env python3
"""Recheck historical identities/order using a local MStream Git clone."""
import argparse
import datetime
import json
import subprocess
from pathlib import Path

from common import HERE, encode, load_sources, sha256, source_id


def audit(checkout, source_dir):
    reference = json.loads((HERE / 'historical-audit.json').read_text())
    current = load_sources(source_dir)
    identities = []
    for old, expected in reference['files'].items():
        data = subprocess.check_output(['git', '-C', str(checkout), 'show',
                                        reference['revision'] + ':' + expected['path']])
        if sha256(data) != expected['sha256']:
            raise ValueError(f'{old}: historical SHA-256 mismatch')
        rows = [json.loads(line) for line in data.splitlines()]
        name = expected['current_variant']
        new = current[name]
        facts = {
            'path': expected['path'], 'sha256': sha256(data), 'documents': len(rows),
            'current_variant': name,
            'same_ordered_cleaned_text': [r['textCleaned'] for r in rows] == [r['textCleaned'] for r in new],
            'same_ordered_labels': [r['clusterNo'] for r in rows] == [r['clusterNo'] for r in new],
            'same_ordered_ids': [str(r['tweetId']) for r in rows] == [r['Id'] for r in new],
            'legacy_id_field': 'tweetId',
            'duplicate_legacy_ids': len(rows) - len({str(r['tweetId']) for r in rows}),
        }
        if old.startswith('TREC'):
            times = [datetime.datetime.strptime(r['timeStr'], '%a %b %d %H:%M:%S %z %Y')
                     for r in rows]
            descents = sum(a > b for a, b in zip(times, times[1:]))
            facts.update({'timestamp_field': 'timeStr', 'timestamp_descents': descents,
                          'chronologically_nondecreasing': descents == 0})
        if facts != expected:
            raise ValueError(f'{old}: historical audit differs from reference')
        for previous, published in zip(rows, new):
            identities.append({'source_id': source_id(name, published['Id']),
                               'legacy_tweetId': str(previous['tweetId']),
                               'legacy_timeStr': previous.get('timeStr')})
    return encode(identities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-checkout', required=True, type=Path)
    parser.add_argument('--source-dir', type=Path, default=HERE / 'data/raw')
    parser.add_argument('--identity-output', type=Path, default=HERE / 'data/historical-identities.json')
    args = parser.parse_args()
    try:
        data = audit(args.upstream_checkout, args.source_dir)
        if args.identity_output.exists():
            raise ValueError(f'identity output already exists: {args.identity_output}')
        args.identity_output.parent.mkdir(parents=True, exist_ok=True)
        args.identity_output.write_bytes(data)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'historical audit failed: {error}\n')
    print('Verified four historical corpora; preserved native identities separately from learner inputs.')


if __name__ == '__main__':
    main()
