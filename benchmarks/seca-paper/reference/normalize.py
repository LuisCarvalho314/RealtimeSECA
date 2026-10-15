"""Read-only translation of actual C# topology to the existing clustering export."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'clustering'))
import cluster


def normalize(output, snapshot, mapping):
    vocabulary = {v['id']: v['word'] for v in output['vocabulary']}
    sources = {v['csharp_source_id']: v['external_source_id'] for v in mapping}
    if len(vocabulary) != len(output['vocabulary']) or len(sources) != len(mapping):
        raise ValueError('duplicate word/source identities')

    def word(identity):
        return dict(word_id=identity, token='<REFUGE>' if identity == -1 else vocabulary[identity])

    hkts, nodes, used = [], [], set()
    for hkt in snapshot['tree']:
        hkts.append(dict(hkt_id=hkt['id'], parent_node_id=hkt['parentNodeId'],
                         node_ids=[n['id'] for n in hkt['nodes']],
                         expected_words=[word(w['wordId']) for w in hkt['expectedWords']]))
        for node in hkt['nodes']:
            used.update(node['sources'])
            nodes.append(dict(node_id=node['id'], hkt_id=hkt['id'],
                              is_refuge_node=-1 in node['words'],
                              words=[word(w) for w in node['words']],
                              sources=[dict(internal_source_id=s, external_source_id=sources[s])
                                       for s in node['sources']]))
    result = dict(hkts=hkts, nodes=nodes,
                  source_legend=[dict(internal_source_id=s, external_source_id=sources[s])
                                 for s in sorted(used)],
                  word_legend=[word(w) for w in sorted(vocabulary)])
    # Structure validation deliberately does not repair broken reference scope.
    cluster._index(result)
    return result


def canonical(tree, include_sources=True):
    """Unordered labelled tree: incidental IDs and sibling enumeration are excluded."""
    hkts, nodes, children, root, memberships, descriptors = cluster._index(tree)
    seen = set()

    def scope(hid):
        if hid in seen:
            raise ValueError('cyclic/shared HKT')
        seen.add(hid)
        rows = []
        for nid in hkts[hid]['node_ids']:
            rows.append((nodes[nid]['is_refuge_node'], tuple(sorted(descriptors[nid])),
                         tuple(sorted(memberships[nid])) if include_sources else (),
                         tuple(sorted(scope(c) for c in children.get(nid, [])))))
        return tuple(sorted(rows))

    result = scope(root)
    if len(seen) != len(hkts):
        raise ValueError('unreachable HKT')
    return result


def statistics(tree):
    hkts, nodes, children, root, _, _ = cluster._index(tree)
    depths, pending = [], [(root, 1)]
    seen = set()
    while pending:
        hid, depth = pending.pop()
        if hid in seen:
            raise ValueError('cyclic/shared HKT')
        seen.add(hid)
        depths.append(depth)
        for nid in hkts[hid]['node_ids']:
            pending.extend((c, depth + 1) for c in children.get(nid, []))
    if seen != set(hkts):
        raise ValueError('unreachable HKT')
    refuge = sum(n['is_refuge_node'] for n in nodes.values())
    return dict(hkt_count=len(hkts), node_count=len(nodes), hierarchy_depth=max(depths),
                ordinary_node_count=len(nodes)-refuge, refuge_node_count=refuge)
