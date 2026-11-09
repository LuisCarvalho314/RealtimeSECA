#!/usr/bin/env python3
"""Audit the adapters' documented token-set conversions against shared batches."""
import argparse
import hashlib
import json
from pathlib import Path
from run import prepare


def digest(records):
    h = hashlib.sha256()
    for batch, sid, words in records:
        h.update(json.dumps([batch, sid, sorted(words)], ensure_ascii=False,
                            separators=(',', ':')).encode())
        h.update(b'\n')
    return h.hexdigest()


def verify(paths):
    word_ids = {}
    inverse_words = {}
    records = {'shared_json': [], 'csharp_rows': [], 'rust_v1': [], 'rust_v2': []}
    batch_counts = []
    for path in paths:
        batch = json.loads(Path(path).read_bytes())
        for source in batch['sources']:
            sid = source['source_id']
            words = set(source['tokens'])
            csharp_ids = []
            for word in source['tokens']:
                if word not in word_ids:
                    word_ids[word] = len(word_ids) + 1
                    inverse_words[word_ids[word]] = word
                if word_ids[word] not in csharp_ids:
                    csharp_ids.append(word_ids[word])
            csharp_words = {inverse_words[w] for w in csharp_ids}
            v1_words = set(source['tokens'])
            v2_words = set(source['tokens'])
            if not (words == csharp_words == v1_words == v2_words):
                raise ValueError(f'adapter token set mismatch: {sid}')
            for key, value in [('shared_json', words), ('csharp_rows', csharp_words),
                               ('rust_v1', v1_words), ('rust_v2', v2_words)]:
                records[key].append((batch['batch_index'], sid, value))
        batch_counts.append(len(batch['sources']))
    hashes = {key: digest(value) for key, value in records.items()}
    if len(set(hashes.values())) != 1:
        raise ValueError('adapter input hashes differ')
    return {'ordered_token_sets_sha256_by_adapter': hashes,
            'batch_documents': batch_counts, 'documents': sum(batch_counts),
            'vocabulary': len(word_ids)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', choices=['G', 'H'], default='G')
    parser.add_argument('--batches', type=int, default=12)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths, provenance = prepare(args.theme, args.batches, args.manifest)
    result = verify(paths)
    if result['ordered_token_sets_sha256_by_adapter']['shared_json'] != provenance['ordered_token_sets_sha256']:
        raise ValueError('verification digest differs from runner digest')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
