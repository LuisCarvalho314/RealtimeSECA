import unittest
from seca_bitset import Incidence, scope_errors


class BitsetTests(unittest.TestCase):
    def test_duplicate_rows_do_not_count_twice(self):
        a = Incidence.from_records([(1, 1), (2, 1), (1, 2)])
        b = Incidence.from_records([(1, 1), (1, 1), (2, 1), (1, 2)])
        self.assertEqual(a.build(), b.build())

    def test_refuge_refinement_and_exclusive_cutoff(self):
        matrix = Incidence.from_records([(1, 1), (2, 1), (3, 2)])
        tree = matrix.build(branch_minimum=1)
        self.assertEqual([n['words'] for n in tree], [[1], [-1]])
        self.assertEqual(tree[1]['child'], [])
        self.assertEqual(matrix.build(branch_minimum=0)[1]['child'][0]['words'], [2])

    def test_greedy_source_union_changes_later_denominator(self):
        records = [(1, 1), (2, 1), (1, 2), (3, 2), (2, 3), (4, 3)]
        tree = Incidence.from_records(records).build(alpha=1, beta=.5, branch_minimum=10)
        self.assertEqual([n['words'] for n in tree], [[1, 2], [3]])
        self.assertEqual(tree[0]['sources'], [1, 2, 3])

    def test_equal_score_uses_first_node(self):
        records = [(1, 1), (2, 2), (1, 3), (2, 3)]
        # Word 3 is prominent, use distinct equal-frequency columns instead.
        records = [(1, 1), (2, 1), (3, 2), (4, 2), (1, 3), (3, 3)]
        tree = Incidence.from_records(records).build(alpha=1, beta=.5, branch_minimum=10)
        self.assertEqual([n['words'] for n in tree], [[1, 3], [2]])

    def test_original_surviving_row_breaks_child_frequency_tie(self):
        records = [(2, 2), (1, 1), (2, 1), (1, 3)]
        tree = Incidence.from_records(records).build(alpha=.7, branch_minimum=0)
        self.assertEqual(tree[0]['child'][0]['words'], [2])

    def test_metric_stable_and_novel_evidence(self):
        nodes = [{'members': 7, 'prior': {1: 3, 2: 3}, 'added': {1: 4, 2: 4}}]
        metrics = scope_errors(nodes, {}, [1, 2], .7, .5)
        self.assertEqual([metrics[k] for k in ('alpha_error', 'beta_error', 'word_importance_error')], [0., 0., 0.])
        metrics = scope_errors([{'members': 3, 'prior': {1: 3, 2: 3}}], {3: (0, 3)}, [1, 2], .7, .5)
        self.assertAlmostEqual(metrics['alpha_error'], .7 - 2 / 3)
        self.assertEqual(metrics['beta_error'], 0)
        self.assertAlmostEqual(metrics['word_importance_error'], 3 / 7)

    def test_empty_metric_scope_and_input_validation(self):
        self.assertEqual(Incidence.from_records([]).build(), [])
        self.assertEqual(scope_errors([], {}, [], .7, .5)['word_importance_error'], 1.)
        with self.assertRaises(ValueError):
            Incidence.from_records([(1, -1)])
        with self.assertRaises(ValueError):
            Incidence.from_records([]).build(alpha=float('nan'))


if __name__ == '__main__':
    unittest.main()
