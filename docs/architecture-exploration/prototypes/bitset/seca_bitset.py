"""Incidence-matrix SECA construction and staged error kernel; Python stdlib only."""
import json
import math
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Incidence:
    sources: tuple
    words: tuple
    columns: tuple
    rows: tuple

    @classmethod
    def from_records(cls, records):
        records = tuple((int(s), int(w)) for s, w in records)
        if any(w <= 0 for _, w in records):
            raise ValueError("word IDs must be positive; -1 is reserved for refuge")
        sources = tuple(sorted({s for s, _ in records}))
        words = tuple(sorted({w for _, w in records}))
        source_index = {s: i for i, s in enumerate(sources)}
        word_index = {w: i for i, w in enumerate(words)}
        columns = [0] * len(words)
        for s, w in records:
            columns[word_index[w]] |= 1 << source_index[s]
        # Retain row rank only for observable frequency tie ordering, not evidence.
        rows = tuple((source_index[s], word_index[w]) for s, w in records)
        return cls(sources, words, tuple(columns), rows)

    def decode(self, mask):
        return [s for i, s in enumerate(self.sources) if mask & (1 << i)]

    def build(self, alpha=0.7, beta=0.5, branch_minimum=1):
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (alpha, beta)):
            raise ValueError("alpha and beta must be finite and in [0,1]")
        if branch_minimum < 0:
            raise ValueError("branch minimum must be nonnegative")

        def scope(source_mask, available):
            columns = {i: self.columns[i] & source_mask for i in available}
            columns = {i: bits for i, bits in columns.items() if bits}
            if not columns:
                return []
            counts = {i: bits.bit_count() for i, bits in columns.items()}
            # Earliest surviving input row resolves prominent ties, including child scopes.
            rank = {}
            for r, (s, w) in enumerate(self.rows):
                if w in columns and source_mask & (1 << s):
                    rank.setdefault(w, r)
            first = min(columns, key=lambda i: (-counts[i], rank[i]))
            expected = [i for i in sorted(columns)
                        if counts[i] / counts[first] >= alpha]
            members = [columns[first]]
            node_words = [[first]]
            for i in expected:
                if i == first:
                    continue
                scores = [(columns[i] & m).bit_count() / m.bit_count() for m in members]
                best = max(range(len(scores)), key=scores.__getitem__)
                if scores[best] >= beta:
                    members[best] |= columns[i]
                    node_words[best].append(i)
                else:
                    members.append(columns[i])
                    node_words.append([i])
            covered = 0
            total = 0
            for m in members:
                covered |= m
            for m in columns.values():
                total |= m
            refuge = total & ~covered
            if refuge:
                members.append(refuge)
                node_words.append([])
            result = []
            for ws, mask in zip(node_words, members):
                remaining = tuple(i for i in columns if i not in ws)
                child = scope(mask, remaining) if mask.bit_count() > branch_minimum else []
                result.append({"words": sorted(self.words[i] for i in ws) if ws else [-1],
                               "sources": self.decode(mask),
                               "word_sources": {str(self.words[i]): self.decode(columns[i]) for i in sorted(ws)},
                               "child": child})
            return result

        return scope((1 << len(self.sources)) - 1, tuple(range(len(self.words))))


def scope_errors(nodes, candidates, old_words, alpha, beta):
    """Pure staged Rust metric semantics, not candidate admission or update control.

    Each node: members, prior {word:mask}, added {word:mask}, assigned [word].
    Candidates: {word:(retained_count,incoming_count)} after strength admission.
    Source bits must use one shared dense index; incoming and retained identities
    are disjoint, as required by the production input contract.
    """
    counts = [old.bit_count() + n.get("added", {}).get(w, 0).bit_count()
              for n in nodes for w, old in n["prior"].items()]
    counts += [a + b for a, b in candidates.values()]
    prominent = max([1] + counts)
    denominator = max(1, sum(counts))
    metrics = {}
    for n in nodes:
        for w in sorted(set(n["prior"]) | set(n.get("assigned", []))):
            updated = n["prior"].get(w, 0) | n.get("added", {}).get(w, 0)
            count = updated.bit_count()
            metrics[w] = (count / prominent,
                          (updated & n["members"]).bit_count() / max(1, n["members"].bit_count()),
                          count / denominator)
    for w, (a, b) in candidates.items():
        metrics.setdefault(w, ((a + b) / prominent, 0.0, (a + b) / denominator))
    ordered = sorted(set(old_words))
    size = len(ordered)
    ae = sum(max(0.0, alpha - metrics.get(w, (0, 0, 0))[0]) for w in ordered)
    be = sum(max(0.0, beta - metrics.get(w, (0, 0, 0))[1]) for w in ordered)
    wi = 1.0 - sum(metrics.get(w, (0, 0, 0))[2] for w in ordered)
    return {"alpha_error": ae / size if size else 0.0,
            "beta_error": be / size if size else 0.0,
            "word_importance_error": min(1.0, max(0.0, wi)), "word_metrics": metrics}


def from_documents(documents):
    vocabulary = {}
    records = []
    for d in documents:
        for word in d["words"]:
            vocabulary.setdefault(word, len(vocabulary) + 1)
            records.append((d["id"], vocabulary[word]))
    return Incidence.from_records(records), vocabulary


if __name__ == "__main__":
    data = json.load(sys.stdin)
    if "records" in data:
        matrix = Incidence.from_records(data["records"])
        vocabulary = None
    else:
        matrix, vocabulary = from_documents(data["batches"][0])
    print(json.dumps({"tree": matrix.build(data.get("alpha", .7), data.get("beta", .5),
                                           data.get("branchMinimum", 1)),
                      "vocabulary": vocabulary}, sort_keys=True))
