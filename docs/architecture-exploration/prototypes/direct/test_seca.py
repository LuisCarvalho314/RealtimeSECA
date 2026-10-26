import json
from pathlib import Path
import unittest
from seca import construct, relation, errors, run, build

ROOT = Path(__file__).resolve().parents[4]


def structural(tree):
    if tree is None:
        return None
    return {'path': tree['path'], 'nodes': [
        {'words': [w for w in n['words'] if w not in ('-1', '<>')],
         'sources': n['sources'], 'refuge': n['refuge'],
         'child': structural(n['child'])} for n in tree['nodes']]}


class DirectTests(unittest.TestCase):
    def test_frozen_state_and_duplicate_relations(self):
        rows = [(1, 1), (1, 2), (2, 1)]
        self.assertEqual(construct(relation(rows)), construct(relation(rows + rows)))
        state = construct(relation(rows))
        with self.assertRaises(AttributeError):
            state.nodes = ()

    def test_no_refinement_without_new_words(self):
        self.assertIsNone(construct(relation([(1, 1), (2, 1)])).nodes[0].child)

    def test_shared_api_scoped_indexes(self):
        result = build([{'source': 1, 'word': 1}, {'source': 1, 'word': 2},
                        {'source': 2, 'word': 1}], .7, .5, 1)
        self.assertEqual(result, [{'words': [1], 'sources': [1, 2],
                                  'word_sources': {'1': [1, 2]},
                                  'child': [{'words': [2], 'sources': [1],
                                             'word_sources': {'2': [1]}, 'child': []}]}])

    def test_common_seed711_random0(self):
        data = json.loads(Path(__file__).with_name('random0_regression.json').read_text())
        self.assertEqual(build(data['records'], data['alpha'], data['beta'],
                               data['branchMinimum']), data['expected'])

    def test_child_tie_uses_first_surviving_row(self):
        rows = [(1, 1), (2, 2), (3, 1), (2, 9), (3, 9), (4, 9)]
        tree = construct(relation(rows), .7, 1., 0)
        self.assertEqual([n.words for n in tree.nodes[0].child.nodes],
                         [frozenset({2}), frozenset({1})])

    def test_empty_and_full_precision_metrics(self):
        self.assertEqual(errors([], {}), {'alpha': 0, 'beta': 0, 'importance': 1})
        actual = errors([1, 2], {1: (2/3, 1., 2/7), 2: (2/3, 1., 2/7)})
        self.assertAlmostEqual(actual['alpha'], .7 - 2/3)
        self.assertAlmostEqual(actual['importance'], 3/7)

    def test_locked_rust_baseline_fixtures(self):
        fixtures = ROOT / 'crates/realtime-seca-core/tests/fixtures'
        for group in ('parity_hkt_reconstruction', 'scientific_hkt'):
            for path in sorted((fixtures / group).glob('*/input.json')):
                observed = path.parent / 'rust.observed.json'
                if observed.exists():
                    with self.subTest(fixture=str(path.parent)):
                        expected = json.loads(observed.read_text())[0]['topology']
                        self.assertEqual(run(json.loads(path.read_text())), structural(expected))


if __name__ == '__main__':
    unittest.main()
