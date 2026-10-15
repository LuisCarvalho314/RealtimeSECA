"""Identify identical baseline parent scopes affected by the native branch boundary."""
import argparse
import json
from pathlib import Path

from normalize import normalize
import cluster
from reference_execute import write


def attachments(tree):
    hkts, nodes, children, root, memberships, descriptors = cluster._index(tree)
    result = {}
    def visit(hid, ancestors):
        for nid in hkts[hid]['node_ids']:
            label = (nodes[nid]['is_refuge_node'], tuple(sorted(descriptors[nid])),
                     tuple(sorted(memberships[nid])))
            path = ancestors + (label,)
            if path in result:
                raise ValueError('ambiguous identical node paths')
            result[path] = bool(children.get(nid))
            for child in children.get(nid, []):
                visit(child, path)
    visit(root, ())
    return result


def analyze(runs, rust, output):
    result = {}
    for theme in ['G', 'H']:
        mapping = json.loads((runs / f'{theme}-mapping.json').read_text())
        raw = json.loads((runs / f'{theme}-full/cycles/cycle-0001.json').read_text())
        csharp = normalize(raw, raw['snapshots'][0], mapping)
        other = json.loads((rust / f'{theme}-Full-indirect/trees/tree_batch_0000.json').read_text())
        a, b = attachments(csharp), attachments(other)
        matched = a.keys() & b.keys()
        boundary = [path for path in matched if a[path] and not b[path] and len(path[-1][2]) == 10]
        result[theme] = dict(matched_parent_nodes=len(matched),
                             csharp_only_branches_at_exact_minimum=len(boundary),
                             matched_ten_source_scopes=[list(path[-1][2]) for path in sorted(boundary)],
                             interpretation='Native >=10 versus >10 explains branch presence at these otherwise identical parent paths; not an attribution of all divergence')
    write(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ['runs', 'rust', 'output']:
        parser.add_argument('--'+option, type=Path, required=True)
    args = parser.parse_args()
    analyze(args.runs, args.rust, args.output)
