"""Pure nested SECA construction and metric kernels; no production dependencies."""
from dataclasses import dataclass, replace
import json
import sys


@dataclass(frozen=True)
class Node:
    words: frozenset
    sources: frozenset
    child: object = None


@dataclass(frozen=True)
class Hkt:
    nodes: tuple
    expected: frozenset


@dataclass(frozen=True)
class Scope:
    memberships: tuple
    rows: tuple

    def __iter__(self):
        return iter(self.memberships)

    def __bool__(self):
        return bool(self.memberships)


def restrict(scope, node):
    return relation((source, word) for source, word in scope.rows
                    if word not in node.words and source in node.sources)


def relation(rows):
    """Rows are ordered (source, positive integer word) pairs, duplicates harmless."""
    rows = tuple(dict.fromkeys(rows))
    words = {}
    for source, word in rows:
        words.setdefault(word, set()).add(source)
    return Scope(tuple((word, frozenset(sources)) for word, sources in words.items()), rows)


def construct(scope, alpha=.7, beta=.5, branch_minimum=1):
    """Persistent values replace destructive row removal and mirrored ID indexes."""
    if not scope:
        return None
    prominent, sources = max(scope, key=lambda pair: len(pair[1]))
    maximum = len(sources)
    admitted = frozenset(w for w, ss in scope if len(ss) / maximum >= alpha)
    nodes = (Node(frozenset((prominent,)), sources),)
    memberships = dict(scope)
    for word in sorted(admitted - {prominent}):
        sources = memberships[word]
        scores = tuple(len(sources & n.sources) / len(n.sources) for n in nodes)
        best = max(range(len(nodes)), key=lambda i: scores[i])
        if scores[best] >= beta:
            old = nodes[best]
            merged = Node(old.words | {word}, old.sources | sources)
            nodes = nodes[:best] + (merged,) + nodes[best + 1:]
        else:
            nodes += (Node(frozenset((word,)), sources),)
    uncovered = frozenset().union(*(ss for _, ss in scope)) - frozenset().union(*(n.sources for n in nodes))
    if uncovered:
        nodes += (Node(frozenset((-1,)), uncovered),)

    def refine(node):
        if len(node.sources) <= branch_minimum:
            return node
        child_scope = restrict(scope, node)
        return replace(node, child=construct(child_scope, alpha, beta, branch_minimum))

    return Hkt(tuple(map(refine, nodes)), admitted - {prominent})


def errors(old_words, measurements, alpha=.7, beta=.5):
    """Measurements ordered by word: (strength1, eligibility1, importance1).

    Caller must implement contextual mapping/admission before producing these.
    No rounding: strict threshold decisions belong to the caller.
    """
    old = tuple(old_words)
    divisor = max(1, len(old))
    return {
        'alpha': sum(max(0., alpha - measurements[w][0]) for w in old) / divisor,
        'beta': sum(max(0., beta - measurements[w][1]) for w in old) / divisor,
        'importance': min(1., max(0., 1. - sum(measurements[w][2] for w in old))),
    }


def input_relation(data):
    vocabulary = {}
    rows = []
    for source in data['batches'][0]:
        for token in source['words']:
            word = vocabulary.setdefault(token, len(vocabulary) + 1)
            rows.append((str(source['id']), word))
    return relation(rows), {v: k for k, v in vocabulary.items()}


def normalized(tree, vocabulary, path='root'):
    if tree is None:
        return None
    return {'path': path, 'nodes': [
        {'words': sorted(vocabulary[w] for w in node.words if w != -1),
         'sources': sorted(node.sources), 'refuge': -1 in node.words,
         'child': normalized(node.child, vocabulary, f'{path}/{i}')}
        for i, node in enumerate(tree.nodes)]}


def run(data):
    scope, vocabulary = input_relation(data)
    return normalized(construct(scope, data.get('alpha', .7), data.get('beta', .5),
                                data.get('branchMinimum', 1)), vocabulary)


def build(records, alpha=.7, beta=.5, branch_minimum=1):
    """Shared evaluation API: nested canonical nodes with scoped word indexes."""
    scope = relation((row['source'], row['word']) for row in records)
    tree = construct(scope, alpha, beta, branch_minimum)

    def project(current, local_scope):
        if current is None:
            return []
        memberships = dict(local_scope)
        result = []
        for node in current.nodes:
            child_scope = restrict(local_scope, node)
            result.append({'words': sorted(node.words), 'sources': sorted(node.sources),
                           'word_sources': {str(w): sorted(memberships[w])
                                            for w in sorted(node.words) if w != -1},
                           'child': project(node.child, child_scope)})
        return result

    return project(tree, scope)


if __name__ == '__main__':
    with open(sys.argv[1]) as handle:
        print(json.dumps(run(json.load(handle)), indent=2, sort_keys=True))
