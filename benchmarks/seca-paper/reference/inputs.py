"""Preserve ordered preprocessed tokens and map identities to positive integers."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.contracts import load_manifest

PARAMETERS = dict(alpha=.7, beta=.5, alphaError=.1, betaError=.2,
                  wordImportanceError=.3, branchMinimum=10)


def convert(documents):
    mapping, batches, seen = [], [], set()
    for index, batch in enumerate(documents):
        converted = []
        for position, source in enumerate(batch):
            sid, tokens = source['source_id'], source['tokens']
            if sid in seen or source['batch_index'] != index:
                raise ValueError('duplicate identity or incorrect batch index')
            if not tokens or any(not isinstance(t, str) or not t.strip() for t in tokens):
                raise ValueError('reference cannot represent empty or invalid token evidence')
            seen.add(sid)
            # One row per distinct source-word relation, as in Rust ingestion.
            # This does not change the archived ordered token list.
            words = list(dict.fromkeys(tokens))
            identity = len(mapping) + 1
            mapping.append(dict(csharp_source_id=identity, external_source_id=sid,
                                batch_index=index, document_position=position,
                                ordered_position=identity - 1,
                                original_tokens_sha256=hashlib.sha256(
                                    json.dumps(tokens, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
                                token_count=len(tokens), distinct_word_count=len(words)))
            converted.append(dict(id=identity, words=words))
        batches.append(converted)
    verify(documents, mapping, batches)
    return PARAMETERS | dict(batches=batches), mapping


def verify(documents, mapping, batches):
    expected = [(i, j, s) for i, b in enumerate(documents) for j, s in enumerate(b)]
    if len(expected) != len(mapping) or len(batches) != len(documents):
        raise ValueError('mapping population mismatch')
    if any(len(a) != len(b) for a, b in zip(documents, batches)):
        raise ValueError('batch population mismatch')
    for number, ((i, j, source), row) in enumerate(zip(expected, mapping), 1):
        if (row['csharp_source_id'], row['external_source_id'], row['batch_index'],
            row['document_position'], row['ordered_position']) != (
                number, source['source_id'], i, j, number - 1):
            raise ValueError('mapping identity/order mismatch')
        tokens = source['tokens']
        token_hash = hashlib.sha256(json.dumps(tokens, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        if (row['token_count'], row['distinct_word_count'], row['original_tokens_sha256']) != (
                len(tokens), len(set(tokens)), token_hash):
            raise ValueError('mapping token evidence mismatch')
        if batches[i][j] != dict(id=number, words=list(dict.fromkeys(tokens))):
            raise ValueError('source-word relation mismatch')


def load(path):
    manifest, documents, truth, hashes = load_manifest(path)
    data, mapping = convert(documents)
    return manifest, documents, truth, hashes, data, mapping
