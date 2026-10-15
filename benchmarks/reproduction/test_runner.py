"""Contract tests independent of the core algorithm and optional datasets."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.data = dict(runner.SETTINGS, batches=[[dict(id=1, words=['a'])]])

    def test_identity_tokens_and_empty_failure(self):
        runner.validate(self.data)
        for batches in ([], [[]], [[dict(id=0, words=['a'])]],
                        [[dict(id=1, words=['a', 'a'])]],
                        [[dict(id=1, words=[' a'])]],
                        [[dict(id=1, words=['a']), dict(id=1, words=['b'])]]):
            with self.subTest(batches=batches), self.assertRaises(AssertionError):
                runner.validate(dict(self.data, batches=batches))

    def test_adapters_remove_ids_order_but_preserve_attachment(self):
        rust = dict(hkts=[dict(hkt_id=1, parent_node_id=0, node_ids=[4, 5]),
                          dict(hkt_id=2, parent_node_id=4, node_ids=[6])],
                    nodes=[dict(node_id=i, words=[dict(word_id=w, token=t)],
                                sources=[dict(external_source_id=str(s))], is_refuge_node=w == -1)
                           for i, w, t, s in [(4, 1, 'a', 1), (5, -1, None, 2), (6, 2, 'b', 1)]])
        output = dict(vocabulary=[dict(id=20, word='a'), dict(id=30, word='b')])
        snapshot = dict(tree=[dict(id=10, parentNodeId=0, nodes=[
            dict(id=50, words=[-1], sources=[2]), dict(id=40, words=[20], sources=[1])]),
            dict(id=11, parentNodeId=40, nodes=[dict(id=60, words=[30], sources=[1])])])
        a = runner.topology_rust(rust)
        b = runner.topology_csharp(output, snapshot)
        self.assertEqual(a, b)
        self.assertEqual(runner.characteristics(a), dict(hkts=2, nodes=3, refuge_nodes=1,
            source_memberships=3, descriptor_occurrences=2, max_depth=1, represented_sources=2))
        changed = copy.deepcopy(snapshot)
        changed['tree'][1]['parentNodeId'] = 50
        self.assertNotEqual(a, runner.topology_csharp(output, changed))

    def test_failure_and_timeout_are_not_measurements(self):
        import sys
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'test.rust'
            result, path = runner.execute([sys.executable, '-c', 'raise SystemExit(3)'], 2, prefix)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['exit_code'], 3)
            self.assertEqual(path.name, 'test.rust.stdout')
            result, _ = runner.execute([sys.executable, '-c', 'import time; time.sleep(5)'], .01, prefix)
            self.assertEqual(result['status'], 'timeout')
            self.assertNotIn('metrics', result)

    def test_committed_fixture_inventory_is_complete(self):
        fixtures = runner.ROOT / 'crates/realtime-seca-core/tests/fixtures'
        self.assertEqual(len(list((fixtures / 'parity_context_metrics').glob('*.input.json'))), 19)
        self.assertEqual(len(list((fixtures / 'parity_hkt_reconstruction').glob('*/input.json'))), 21)

    def test_hash_mismatch_refuses_cached_public_data(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'News').write_text('{}\n')
            with self.assertRaisesRegex(AssertionError, 'hash mismatch'):
                runner.public_input('G', Path(directory), False)


if __name__ == '__main__':
    unittest.main()
