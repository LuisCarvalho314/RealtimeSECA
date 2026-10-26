"""Independent equation checks. No Rust, C#, or construction prototype imports.

Run from repository root; writes only iteration-2/scientific/exact_results.json
and discrepancy_ledger.json. Historical observations are read-only evidence.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def shared_measurements():
    nodes = [{1, 2, 3, 4, 5, 6}, {4, 5, 6, 7}]
    x = {1, 4, 7}
    counts = [len(n & x) for n in nodes]
    old_counts = [6, 4] + counts
    return {
        'scope_x_df': len(x), 'local_x_df': counts,
        'unique_word_wi': 1 - F(6 + 4 + len(x), 6 + 4 + len(x)),
        'operational_wi': 1 - F(6 + 4 + counts[-1], sum(old_counts)),
        'scope_alpha_error': max(F(0), F(45, 100) - F(len(x), 6)) / 3,
        'operational_alpha_error': max(F(0), F(45, 100) - F(counts[-1], 6)) / 3,
        'worst_node_beta_error': max(F(0), *(F(3, 5) - F(len(n & x), len(n)) for n in nodes)) / 3,
        'last_node_beta_error': max(F(0), F(3, 5) - F(len(nodes[-1] & x), len(nodes[-1]))) / 3,
    }


def calculations():
    return {
        'shared': shared_measurements(),
        'expected_history_wi': [F(2 * k, 2 + k + 2 * k) for k in range(1, 5)],
        'candidate_incoming_beta': [F(2, 6), F(1, 7), F(1, 8), F(1, 9)],
        'candidate_historical_beta': [F(4, 6), F(5, 7), F(6, 8), F(7, 9)],
        'candidate_denominator': {'node_union': F(2, 6), 'prominent': F(2, 5)},
        'frequency_order': {'b_into_a': F(2, 2), 'a_into_b': F(2, 3)},
        'collision_tie': [F(1, 2), F(1, 2)],
        'scope_vs_adopted_union_alpha': {'scoped': F(0), 'adopted_only': F(1, 18)},
        'precision': {'raw_wi': F(1, 30001), 'rounded4': F(0), 'threshold': F(3, 100000)},
    }


def structural(tree):
    if tree is None:
        return None
    return {'path': tree['path'], 'nodes': [
        {'words': [w for w in n['words'] if w not in ('-1', '<>')],
         'sources': n['sources'], 'refuge': n['refuge'],
         'child': structural(n['child'])} for n in tree['nodes']]}


def ledger():
    rows = []
    fixtures = ROOT / 'crates/realtime-seca-core/tests/fixtures'
    for group in ('parity_hkt_reconstruction', 'scientific_hkt'):
        for path in sorted((fixtures / group).glob('*/input.json')):
            rust = path.parent / 'rust.observed.json'
            csharp = path.parent / 'csharp.normalized.json'
            if not (rust.exists() and csharp.exists()):
                continue
            r = structural(json.loads(rust.read_text())[0]['topology'])
            c = structural(json.loads(csharp.read_text())[0]['topology'])
            if r != c:
                rows.append({'fixture': f'{group}/{path.parent.name}',
                             'input': json.loads(path.read_text()),
                             'csharp_baseline': c, 'rust_baseline': r})
    return rows


class Equations(unittest.TestCase):
    def test_shared_word_mass_without_new_vocabulary(self):
        s = shared_measurements()
        self.assertEqual(s['unique_word_wi'], 0)
        self.assertEqual(s['operational_wi'], F(1, 7))
    def test_shared_alpha(self):
        s = shared_measurements()
        self.assertEqual(s['scope_alpha_error'], 0)
        self.assertEqual(s['operational_alpha_error'], F(7, 180))
    def test_shared_beta(self):
        s = shared_measurements()
        self.assertEqual(s['worst_node_beta_error'], F(4, 45))
        self.assertEqual(s['last_node_beta_error'], F(1, 30))
    def test_expected_history(self):
        self.assertEqual(calculations()['expected_history_wi'], [F(2, 5), F(1, 2), F(6, 11), F(4, 7)])
    def test_candidate_history_changes_admission(self):
        c = calculations()
        self.assertTrue(all(x < F(1, 2) for x in c['candidate_incoming_beta']))
        self.assertTrue(all(x >= F(1, 2) for x in c['candidate_historical_beta']))
    def test_asymmetric_order_changes_grouping(self):
        c = calculations()['frequency_order']
        self.assertEqual(c['b_into_a'], 1)
        self.assertLess(c['a_into_b'], 1)
    def test_node_union_denominator(self):
        c = calculations()['candidate_denominator']
        self.assertLess(c['node_union'], F(2, 5))
        self.assertEqual(c['prominent'], F(2, 5))
    def test_precision_trigger(self):
        c = calculations()['precision']
        self.assertGreater(c['raw_wi'], c['threshold'])
        self.assertLess(c['rounded4'], c['threshold'])
    def test_branch_equality_is_not_residual_support(self):
        parent, residual, minimum = {1, 2}, {1}, 2
        self.assertTrue(len(parent) >= minimum)
        self.assertFalse(len(parent) > minimum)
        self.assertFalse(len(residual) >= minimum)
    def test_refuge_is_complement_and_matching_overlaps(self):
        universe = {1, 2, 3, 4}
        a, b = {1, 3}, {2, 3}
        self.assertEqual(universe - (a | b), {4})
        self.assertEqual(a & b, {3})
    def test_recursive_scope_retains_explicit_intersection(self):
        b, c = {1, 2, 3, 4}, {3, 4, 5, 6}
        self.assertEqual(b & c, {3, 4})
        remaining = {'b', 'c'} - {'b', 'c'}
        self.assertEqual(remaining, set())
    def test_exact_collision_tie_has_no_unique_argmax(self):
        a, b, c = {1, 3}, {2, 4}, {3, 4}
        scores = [F(len(n & c), len(n)) for n in (a, b)]
        self.assertEqual(scores, [F(1, 2), F(1, 2)])
    def test_all_fifteen_captured_disagreements_accounted(self):
        self.assertEqual(len(ledger()), 15)
    def test_duplicate_source_membership_is_a_set(self):
        self.assertEqual(len(set([1, 1, 2])), 2)


if __name__ == '__main__':
    (HERE / 'exact_results.json').write_text(json.dumps(calculations(), default=str, indent=2) + '\n')
    (HERE / 'discrepancy_ledger.json').write_text(json.dumps(ledger(), indent=2) + '\n')
    unittest.main()
