"""Translation and checkpoint regression tests using existing executable goldens."""
import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from inputs import convert, verify
from normalize import canonical, normalize, statistics
import cluster

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction'


def golden_tree(scope):
    return tuple(sorted((n['refuge'], tuple(w for w in n['words'] if w != '<>'),
                         tuple(n['sources']), (golden_tree(n['child']),) if n['child'] else ())
                        for n in scope['nodes']))


class TranslationTests(unittest.TestCase):
    def test_every_golden_snapshot_preserves_labelled_attachments_and_memberships(self):
        count = 0
        for folder in sorted(FIXTURES.iterdir()):
            if not folder.is_dir():
                continue
            output = json.loads((folder / 'csharp.json').read_text())
            goldens = json.loads((folder / 'csharp.normalized.json').read_text())
            data = json.loads((folder / 'input.json').read_text())
            mapping = [dict(csharp_source_id=d['id'], external_source_id=str(d['id']))
                       for batch in data['batches'] for d in batch]
            for snapshot, golden in zip(output['snapshots'], goldens):
                with self.subTest(fixture=folder.name, cycle=snapshot['cycle']):
                    before = copy.deepcopy(output)
                    tree = normalize(output, snapshot, mapping)
                    self.assertEqual(canonical(tree), golden_tree(golden['topology']))
                    self.assertEqual(output, before)
                    self.assertGreater(statistics(tree)['hkt_count'], 0)
                    count += 1
        self.assertEqual(count, 65)

    def test_missing_identity_and_broken_ownership_fail(self):
        output = json.loads((FIXTURES / 'single_root/csharp.json').read_text())
        with self.assertRaises(KeyError):
            normalize(output, output['snapshots'][0], [])
        broken = copy.deepcopy(output)
        broken['snapshots'][0]['tree'][0]['parentNodeId'] = 999
        with self.assertRaises(ValueError):
            normalize(broken, broken['snapshots'][0], [
                dict(csharp_source_id=i, external_source_id=str(i)) for i in range(1, 5)])

    def test_stale_refuge_scope_is_preserved_and_shared_adapter_rejects_it(self):
        output = json.loads((FIXTURES / 'refuge/csharp.json').read_text())
        tree = normalize(output, output['snapshots'][1], [
            dict(csharp_source_id=i, external_source_id=str(i)) for i in range(1, 8)])
        with self.assertRaisesRegex(ValueError, 'subset'):
            cluster.direct(tree, 1, ['6'])

    def test_distinct_relations_preserve_first_appearance_and_external_order(self):
        documents = [[dict(source_id='zero:0000', batch_index=0, tokens=['b', 'a', 'b']),
                      dict(source_id='zero:0001', batch_index=0, tokens=['a'])],
                     [dict(source_id='next', batch_index=1, tokens=['c'])]]
        data, mapping = convert(documents)
        self.assertEqual(data['batches'][0][0], dict(id=1, words=['b', 'a']))
        self.assertEqual(documents[0][0]['tokens'], ['b', 'a', 'b'])
        self.assertEqual(mapping[2]['csharp_source_id'], 3)
        mapping[0]['external_source_id'] = 'wrong'
        with self.assertRaises(ValueError):
            verify(documents, mapping, data['batches'])

    @unittest.skipUnless(os.environ.get('SECA_CHECKPOINT_BUILD'), 'requires local reference build')
    def test_checkpoint_has_identical_successful_snapshots_and_preserves_failure_prefix(self):
        build = Path(os.environ['SECA_CHECKPOINT_BUILD'])
        provenance = json.loads((build / 'provenance.json').read_text())
        env = os.environ | dict(MONO_PATH=provenance['mono_path'])
        for folder in sorted(FIXTURES.iterdir()):
            if not folder.is_dir():
                continue
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'cycles'
                completed = subprocess.run(['mono', str(build / 'checkpoint.exe'),
                                            str(folder / 'input.json'), str(out)],
                                           env=env, capture_output=True)
                self.assertEqual(completed.returncode, 0, completed.stderr.decode())
                original = json.loads((folder / 'csharp.json').read_text())
                snapshots = [json.loads(p.read_text())['snapshots'][0]
                             for p in sorted(out.glob('cycle-*.json'))]
                self.assertEqual(snapshots, original['snapshots'], folder.name)
        contexts = ROOT / 'crates/realtime-seca-core/tests/fixtures/parity_context_metrics'
        for golden in sorted(contexts.glob('*.csharp.json')):
            fixture = golden.with_name(golden.name.replace('.csharp.json', '.input.json'))
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'cycles'
                completed = subprocess.run(['mono', str(build / 'checkpoint.exe'), str(fixture), str(out)],
                                           env=env, capture_output=True)
                self.assertEqual(completed.returncode, 0, completed.stderr.decode())
                snapshots = [json.loads(p.read_text())['snapshots'][0]
                             for p in sorted(out.glob('cycle-*.json'))]
                self.assertEqual(snapshots, json.loads(golden.read_text())['snapshots'], golden.name)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'cycles'
            fixture = ROOT / 'crates/realtime-seca-core/tests/fixtures/parity_context_metrics/shared_candidate.input.json'
            completed = subprocess.run(['mono', str(build / 'checkpoint.exe'), str(fixture), str(out)],
                                       env=env, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(len(list(out.glob('cycle-*.json'))), 1)
            self.assertIn(b'same key', completed.stderr)


if __name__ == '__main__':
    unittest.main()
