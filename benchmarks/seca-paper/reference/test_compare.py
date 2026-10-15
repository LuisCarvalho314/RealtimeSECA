"""Guard ID-independent comparisons, overlap retention, and invalid-scope evidence."""
import json
import unittest
from pathlib import Path

from compare_engines import agreement, differences, representation_evidence, score
from normalize import canonical, normalize

FIXTURE = Path(__file__).resolve().parents[3] / 'crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction'


class ComparisonTests(unittest.TestCase):
    def test_membership_families_and_pairs_ignore_event_ids_and_preserve_overlap(self):
        left = dict(evaluation_source_ids=['a', 'b', 'c'], clusters=[
            dict(cluster_id='1', source_ids=['a', 'b']), dict(cluster_id='2', source_ids=['b', 'c'])])
        right = dict(evaluation_source_ids=['c', 'b', 'a'], clusters=[
            dict(cluster_id='99', source_ids=['c', 'b']), dict(cluster_id='10', source_ids=['b', 'a'])])
        result = agreement(left, right)
        self.assertEqual(result['both_comembers'], 2)
        self.assertEqual(result['both_separate'], 1)
        self.assertEqual(result['changed_source_ids'], [])
        right['clusters'].pop()
        result = agreement(left, right)
        self.assertEqual(result['csharp_only_comembers'], 1)
        self.assertEqual(result['changed_membership_families'], 2)

    def test_identical_overlapping_events_retain_membership_multiplicity(self):
        left = dict(evaluation_source_ids=['a', 'b'], clusters=[
            dict(cluster_id='1', source_ids=['a', 'b']),
            dict(cluster_id='2', source_ids=['a', 'b'])])
        right = dict(evaluation_source_ids=['a', 'b'], clusters=left['clusters'][:1])
        result = agreement(left, right)
        self.assertEqual(result['both_comembers'], 1)
        self.assertEqual(result['changed_membership_families'], 2)

    def test_order_and_allocator_ids_are_not_scientific_differences(self):
        output = json.loads((FIXTURE / 'siblings/csharp.json').read_text())
        mapping = [dict(csharp_source_id=i, external_source_id=str(i)) for i in range(1, 23)]
        tree = normalize(output, output['snapshots'][0], mapping)
        other = json.loads(json.dumps(tree))
        for h in other['hkts']:
            h['hkt_id'] += 100
            h['parent_node_id'] = h['parent_node_id'] + 200 if h['parent_node_id'] else 0
            h['node_ids'] = [n+200 for n in reversed(h['node_ids'])]
        for n in other['nodes']:
            n['node_id'] += 200
            n['hkt_id'] += 100
            n['words'].reverse()
            n['sources'].reverse()
        other['hkts'].reverse()
        other['nodes'].reverse()
        self.assertEqual(canonical(tree), canonical(other))
        self.assertEqual(differences(tree, other), dict(unmatched_csharp_nodes=[], unmatched_rust_nodes=[]))

    def test_native_scope_failure_keeps_excess_id_and_blocks_both_methods(self):
        output = json.loads((FIXTURE / 'refuge/csharp.json').read_text())
        mapping = [dict(csharp_source_id=i, external_source_id=str(i)) for i in range(1, 8)]
        tree = normalize(output, output['snapshots'][1], mapping)
        violations = representation_evidence(tree)['scope_violations']
        self.assertTrue(violations)
        self.assertTrue(all(v['parent_is_refuge'] for v in violations))
        self.assertIn('6', violations[0]['excess_source_ids'])
        for method in ['direct', 'indirect']:
            result, assignment = score(tree, [dict(source_id='6', tokens=['b'])], 1, method,
                                       [dict(source_id='6', label='one')])
            self.assertEqual(result['status'], 'clustering_input_failure')
            self.assertIsNone(assignment)


if __name__ == '__main__':
    unittest.main()
