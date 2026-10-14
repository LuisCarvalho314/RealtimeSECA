"""Pinned source verification and lossless whitespace tokenization (stdlib only)."""
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / 'sources.json').read_text(encoding='utf-8'))
THEMES = {'G': ('News', 740), 'H': ('Tweets-T', 2000)}
PREPROCESSING = (
    'Use upstream textCleaned verbatim; split on whitespace with str.split(); '
    'preserve token order and repetitions; no stemming, stop-word removal, '
    'case folding, deduplication or additional cleaning.'
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def parse_rows(data, name):
    rows, seen = [], set()
    for number, line in enumerate(data.decode('utf-8').splitlines(), 1):
        try:
            row = json.loads(line, object_pairs_hook=unique_object)
            if not isinstance(row, dict) or set(row) != {'Id', 'clusterNo', 'textCleaned'}:
                raise ValueError('expected only Id, clusterNo, textCleaned')
            identity, label, text = row['Id'], row['clusterNo'], row['textCleaned']
            if not isinstance(identity, str) or not identity or identity != identity.strip():
                raise ValueError('invalid original Id')
            if identity in seen:
                raise ValueError(f'duplicate Id: {identity}')
            if type(label) is not int or label < 1:
                raise ValueError('clusterNo must be a positive integer')
            if not isinstance(text, str) or not text.split():
                raise ValueError('empty or invalid textCleaned')
            seen.add(identity)
            rows.append(row)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f'{name}:{number}: {error}') from error
    return rows


def statistics(rows):
    texts = [r['textCleaned'] for r in rows]
    labels = Counter(r['clusterNo'] for r in rows)
    return {
        'documents': len(rows), 'labels': len(labels),
        'duplicate_ids': len(rows) - len({r['Id'] for r in rows}),
        'duplicate_text_excess': len(texts) - len(set(texts)),
        'duplicate_label_text_excess': len(rows) - len({(r['clusterNo'], r['textCleaned']) for r in rows}),
        'minimum_label_documents': min(labels.values()),
        'maximum_label_documents': max(labels.values()),
        'ordered_ids_sha256': sha256(encode([r['Id'] for r in rows])),
        'ordered_records_sha256': sha256(encode(rows)),
        'token_count': sum(len(t.split()) for t in texts),
    }


def verify_bytes(data, name):
    expected = LOCK['files'][name]
    actual = sha256(data)
    if actual != expected['sha256']:
        raise ValueError(f'{name}: SHA-256 mismatch: expected {expected["sha256"]}, got {actual}')
    rows = parse_rows(data, name)
    if statistics(rows) != expected['statistics']:
        raise ValueError(f'{name}: source statistics differ from pinned provenance')
    return rows


def load_sources(directory):
    result = {}
    for name in LOCK['files']:
        path = Path(directory) / name
        if not path.is_file():
            raise ValueError(f'missing source file: {path}; run download.py first')
        result[name] = verify_bytes(path.read_bytes(), name)
    return result


def source_id(name, original_id):
    # Variant qualification matters: Tweets and Tweets-T reuse IDs for different texts.
    return f'mstream:{name}:{original_id}'
