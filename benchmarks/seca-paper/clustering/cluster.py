"""SECA VII-D postprocessing; see docs/seca-clustering-evaluation.md for assumptions."""

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _strings(values, name):
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{name} must be a list of nonempty strings")
    return set(values)


def _unique(rows, key):
    result = {}
    for row in rows:
        identity = _integer(row[key], key, 1)
        if identity in result:
            raise ValueError(f"duplicate {key}: {identity}")
        result[identity] = row
    return result


def _legends(tree):
    sources, external_ids = {}, set()
    for row in tree["source_legend"]:
        identity = row["internal_source_id"]
        if type(identity) is not int or identity in sources:
            raise ValueError("invalid or duplicate internal source ID")
        external = row["external_source_id"]
        _strings([external], "source identity")
        if external == "<UNKNOWN_SOURCE>" or external in external_ids:
            raise ValueError("unresolved or duplicate external source ID")
        sources[identity] = external
        external_ids.add(external)
    words, tokens = {}, set()
    for row in tree["word_legend"]:
        identity = _integer(row["word_id"], "word_id", 1)
        if identity in words:
            raise ValueError("duplicate word ID")
        token = row["token"]
        _strings([token], "word token")
        if token in tokens:
            raise ValueError("duplicate vocabulary token")
        words[identity] = token
        tokens.add(token)
    return sources, words


def _node_evidence(node, source_legend, word_legend):
    if type(node["is_refuge_node"]) is not bool:
        raise ValueError("is_refuge_node must be boolean")
    sources = set()
    for ref in node["sources"]:
        identity = ref["internal_source_id"]
        if type(identity) is not int or identity not in source_legend:
            raise ValueError("unknown internal source ID")
        external = ref["external_source_id"]
        if source_legend[identity] != external:
            raise ValueError("source reference disagrees with legend")
        sources.add(external)
    words = set()
    sentinels = 0
    for ref in node["words"]:
        identity = ref["word_id"]
        if type(identity) is not int:
            raise ValueError("word ID must be an integer")
        if identity == -1:
            sentinels += 1
            if ref["token"] != "<REFUGE>":
                raise ValueError("invalid refuge marker")
        elif identity not in word_legend or ref["token"] != word_legend[identity]:
            raise ValueError("word reference disagrees with legend")
        else:
            words.add(ref["token"])
    if node["is_refuge_node"]:
        if words or sentinels != 1:
            raise ValueError("refuge must contain only its marker")
    elif sentinels or not words:
        raise ValueError("ordinary node must contain descriptors, no refuge marker")
    return sources, words


def _index(tree):
    """Validate and index the public export without consulting serialized state."""
    hkts = _unique(tree["hkts"], "hkt_id")
    nodes = _unique(tree["nodes"], "node_id")
    source_legend, word_legend = _legends(tree)
    memberships, descriptors = {}, {}
    for identity, node in nodes.items():
        _integer(node["hkt_id"], "hkt_id", 1)
        memberships[identity], descriptors[identity] = _node_evidence(
            node, source_legend, word_legend
        )
    children, roots, owned = {}, [], set()
    for identity, hkt in hkts.items():
        parent = _integer(hkt["parent_node_id"], "parent_node_id")
        if parent == 0:
            roots.append(identity)
        elif parent not in nodes:
            raise ValueError("missing HKT attachment node")
        else:
            children.setdefault(parent, []).append(identity)
        for nid in hkt["node_ids"]:
            _integer(nid, "node_id", 1)
            if nid not in nodes or nid in owned or nodes[nid]["hkt_id"] != identity:
                raise ValueError("inconsistent node ownership")
            owned.add(nid)
    if len(roots) != 1 or owned != set(nodes):
        raise ValueError("tree needs one Seed-HKT and exactly one owner per node")
    return hkts, nodes, children, roots[0], memberships, descriptors


def _events(tree):
    hkts, nodes, children, seed, memberships, descriptors = _index(tree)
    seen, events = set(), {}
    # The nearest ordinary ancestor owns contextual descendants. A node reached
    # through only refuge ancestors starts an independent Seed/Orphan event.
    pending = [(seed, None, None)]
    while pending:
        hid, event, parent = pending.pop()
        if hid in seen:
            raise ValueError("cycle or shared HKT")
        seen.add(hid)
        for nid in sorted(hkts[hid]["node_ids"]):
            if parent is not None and not memberships[nid] <= memberships[parent]:
                raise ValueError("child sources must be a subset of parent sources")
            owner = event
            if not nodes[nid]["is_refuge_node"]:
                if owner is None:
                    owner = nid
                    events[owner] = []
                events[owner].append(nid)
            for child in children.get(nid, []):
                pending.append((child, owner, nid))
    if seen != set(hkts):
        raise ValueError("unreachable or cyclic HKT")
    return events, memberships, descriptors


def _population(batch_index, evaluation_source_ids):
    _integer(batch_index, "batch_index")
    population = _strings(evaluation_source_ids, "evaluation_source_ids")
    if len(population) != len(evaluation_source_ids):
        raise ValueError("evaluation population contains duplicate identities")
    return population


def _assignment(batch_index, population, clusters):
    rows = [
        {"cluster_id": "event-" + "-".join(map(str, roots)),
         "source_ids": sorted(sources)}
        for roots, sources in sorted(clusters.items()) if sources
    ]
    rows.sort(key=lambda row: row["cluster_id"])
    assigned = set().union(*(set(row["source_ids"]) for row in rows))
    return {"batch_index": batch_index, "evaluation_source_ids": sorted(population),
            "clusters": rows, "unassigned_source_ids": sorted(population - assigned)}


def direct(tree, batch_index, evaluation_source_ids):
    """Select Seed/Orphan memberships; retain overlapping event assignments."""
    population = _population(batch_index, evaluation_source_ids)
    events, memberships, _ = _events(tree)
    return _assignment(batch_index, population, {
        (root,): memberships[root] & population for root in events
    })


def contextual_proxies(tree):
    """Equation 13, reducing repeated descriptors by maximum (documented assumption)."""
    events, memberships, descriptors = _events(tree)
    proxies = {}
    for root, branch in sorted(events.items()):
        denominator = len(memberships[root])
        if not denominator:
            continue
        weights = {}
        for nid in branch:
            value = Fraction(len(memberships[nid]), denominator)
            for word in sorted(descriptors[nid]):
                weights[word] = max(weights.get(word, Fraction(0)), value)
        proxies[(root,)] = weights
    return proxies


def _threshold(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Fraction)):
        raise ValueError("contextual_threshold must be an explicit number in (0, 1]")
    try:
        cutoff = Fraction(str(value))
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("invalid contextual_threshold") from error
    if not 0 < cutoff <= 1:
        raise ValueError("contextual_threshold must be in (0, 1]")
    return cutoff


def merge_proxies(proxies, contextual_threshold):
    """Merge connected components of original top-word overlap, taking max weights."""
    cutoff = _threshold(contextual_threshold)
    top = {root: {word for word, value in weights.items() if value >= cutoff}
           for root, weights in proxies.items()}
    remaining, merged = set(proxies), {}
    while remaining:
        pending, component = [min(remaining)], set()
        while pending:
            root = pending.pop()
            if root in component:
                continue
            component.add(root)
            remaining.remove(root)
            pending.extend(other for other in sorted(remaining)
                           if top[root] & top[other])
        identity = tuple(sorted(nid for root in component for nid in root))
        weights = {}
        for root in sorted(component):
            for word, value in proxies[root].items():
                weights[word] = max(weights.get(word, Fraction(0)), value)
        merged[identity] = dict(sorted(weights.items()))
    return dict(sorted(merged.items()))


def proximity(weights, source_words):
    """Equation 14: one contribution per matching source word, no normalization."""
    return sum((weights.get(word, Fraction(0)) for word in set(source_words)), Fraction(0))


def indirect(tree, batch_index, evaluation_source_ids, source_tokens, contextual_threshold):
    """Single documented interpretation of VII-D; published-score parity is blocked."""
    population = _population(batch_index, evaluation_source_ids)
    if not isinstance(source_tokens, dict) or not population <= source_tokens.keys():
        raise ValueError("source_tokens must contain every evaluation source ID")
    tokens = {sid: _strings(source_tokens[sid], f"tokens for {sid}")
              for sid in sorted(population)}
    proxies = merge_proxies(contextual_proxies(tree), contextual_threshold)
    clusters = {root: set() for root in proxies}
    for sid in sorted(population):
        best_root, best_value = None, Fraction(0)
        for root, weights in proxies.items():
            value = proximity(weights, tokens[sid])
            if value > best_value:
                best_root, best_value = root, value
        if best_root is not None:
            clusters[best_root].add(sid)
    return _assignment(batch_index, population, clusters)


def evaluate(request):
    """Shared JSON request adapter. Extra fields (including labels) are never used."""
    args = (request["tree"], request["batch_index"], request["evaluation_source_ids"])
    if request["method"] == "direct":
        return direct(*args)
    if request["method"] == "indirect":
        return indirect(*args, request["source_tokens"], request["contextual_threshold"])
    raise ValueError("method must be direct or indirect")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate(json.loads(args.request.read_text(encoding="utf-8")))
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"clustering: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
