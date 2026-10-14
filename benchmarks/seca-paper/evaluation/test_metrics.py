import copy
import math
import unittest
from evaluation.metrics import ScientificBlocker, evaluate, validate_assignments

TRUTH = [{'source_id': s, 'label': t} for s, t in zip('abcd', 'AABB')]


def assignment(clusters):
    return {'batch_index': 0, 'evaluation_source_ids': list('abcd'),
            'clusters': [{'cluster_id': str(i), 'source_ids': list(c)} for i, c in enumerate(clusters)],
            'unassigned_source_ids': sorted(set('abcd') - set(''.join(clusters)))}


class MetricsTests(unittest.TestCase):
    def assert_scores(self, clusters, nmi, precision, recall, f1, purity):
        result = evaluate(assignment(clusters), TRUTH)
        for name, expected in zip(('nmi', 'bcubed_precision', 'bcubed_recall', 'bcubed_f1', 'purity'),
                                  (nmi, precision, recall, f1, purity)):
            self.assertAlmostEqual(result[name], expected, places=14)

    def test_perfect(self):
        self.assert_scores(['ab', 'cd'], 1, 1, 1, 1, 1)

    def test_merged(self):
        self.assert_scores(['abcd'], 0, .5, 1, 2/3, .5)

    def test_fragmented(self):
        self.assert_scores(['a', 'b', 'c', 'd'], 2/3, 1, .5, 2/3, 1)

    def test_mixed_document_f1_not_harmonic_mean_of_means(self):
        # a,b: p=2/3,r=1,F1=4/5; c:1/3,1/2,2/5; d:1,1/2,2/3.
        self.assert_scores(['abc', 'd'],
            2*(.5*math.log(4/3)+.25*math.log(2/3)+.25*math.log(2))/
            (math.log(2)-.75*math.log(.75)-.25*math.log(.25)),
            2/3, .75, 2/3, .75)

    def test_relabeling_and_order_are_irrelevant(self):
        a = assignment(['abc', 'd'])
        b = copy.deepcopy(a)
        b['clusters'][0]['cluster_id'] = 'ground-truth-looking-but-arbitrary'
        b['clusters'].reverse()
        b['evaluation_source_ids'].reverse()
        self.assertEqual(evaluate(a, TRUTH), evaluate(b, list(reversed(TRUTH))))

    def test_empty_clusters_ignored(self):
        self.assertEqual(evaluate(assignment(['ab', 'cd', '']), TRUTH), evaluate(assignment(['ab', 'cd']), TRUTH))

    def test_empty_population_and_degenerate(self):
        a = {'batch_index': 0, 'evaluation_source_ids': [], 'clusters': [], 'unassigned_source_ids': []}
        self.assertIsNone(evaluate(a, [])['nmi'])
        self.assertIsNone(evaluate(assignment(['abcd']), [{'source_id': s, 'label': 'A'} for s in 'abcd'])['nmi'])

    def test_overlap_valid_at_boundary_but_scoring_blocked(self):
        a = assignment(['ab', 'bcd'])
        validate_assignments(a, TRUTH)
        with self.assertRaises(ScientificBlocker):
            evaluate(a, TRUTH)

    def test_unassigned_scoring_blocked(self):
        for clusters in ([], ['ab']):
            with self.assertRaises(ScientificBlocker):
                evaluate(assignment(clusters), TRUTH)

    def test_invalid_boundaries(self):
        cases = []
        a = assignment(['ab', 'cd']); a['clusters'][0]['source_ids'].append('a'); cases.append(a)
        a = assignment(['ab', 'cd']); a['clusters'][0]['source_ids'].append('unknown'); cases.append(a)
        a = assignment(['ab', 'cd']); a['unassigned_source_ids'] = ['a']; cases.append(a)
        a = assignment(['ab', 'cd']); a['evaluation_source_ids'].append('a'); cases.append(a)
        a = assignment(['ab', 'cd']); a['clusters'][1]['cluster_id'] = '0'; cases.append(a)
        for a in cases:
            with self.subTest(a=a), self.assertRaises(ValueError):
                evaluate(a, TRUTH)
        with self.assertRaises(ValueError):
            evaluate(assignment(['ab', 'cd']), TRUTH[:-1])
        with self.assertRaises(ValueError):
            evaluate(assignment(['ab', 'cd']), TRUTH + TRUTH[:1])
        with self.assertRaises(ScientificBlocker):
            evaluate(assignment(['ab', 'cd']), [{'source_id': s, 'label': ['A', 'B']} for s in 'abcd'])


if __name__ == '__main__':
    unittest.main()
