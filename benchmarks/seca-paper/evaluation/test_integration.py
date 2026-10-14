import os
import sys
import tempfile
import unittest
from pathlib import Path
from evaluation.io import digest, read_json, write_json
from evaluation.make_fixture import make_fixture
from evaluation.metrics import evaluate

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from recompute import recompute, score
from run import run


class IntegrationTests(unittest.TestCase):
    def test_independent_metrics_and_missing_membership(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            truth = read_json(root / 'truth.json')
            assignment = read_json(root / 'assignments-0000.json')
            expected = evaluate(assignment, truth)
            for metric, value in score(assignment, truth).items():
                self.assertAlmostEqual(value, expected[metric])
            assignment['clusters'][0]['source_ids'].pop()
            self.assertTrue(all(v is None for v in score(assignment, truth).values()))

    @unittest.skipUnless(os.environ.get('SECA_TEST_ENGINE'), 'requires built Rust CLI')
    def test_saved_learning_is_checked_before_rescoring(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = root / 'fixture'
            make_fixture(fixture)
            args = (fixture / 'manifest.json', fixture / 'config-Full.json')
            first, second = root / 'first', root / 'second'
            a = run(*args, first, 'direct', os.environ['SECA_TEST_ENGINE'], mock=fixture, smoke=True)
            b = run(*args, second, 'direct', os.environ['SECA_TEST_ENGINE'], mock=fixture,
                    smoke=True, learning_from=first)
            self.assertEqual(a, b)
            self.assertEqual(read_json(second / 'run.json')['learning_evidence']['run_sha256'], digest(first / 'run.json'))
            prior = read_json(first / 'run.json')
            prior['engine_input_hashes']['batch-0000.json'] = 'changed'
            write_json(first / 'run.json', prior)
            with self.assertRaisesRegex(ValueError, 'saved learning evidence differs'):
                run(*args, root / 'bad', 'direct', os.environ['SECA_TEST_ENGINE'],
                    mock=fixture, smoke=True, learning_from=first)
            self.assertEqual(read_json(root / 'bad/run.json')['status'], 'failed')

    @unittest.skipUnless(os.environ.get('SECA_TEST_ENGINE'), 'requires built Rust CLI')
    def test_blocked_cycle_does_not_discard_later_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = root / 'fixture'
            make_fixture(fixture)
            manifest = read_json(fixture / 'manifest.json')
            manifest['batches'] = []
            truth = []
            for index in range(12):
                source = f'original-{index}'
                path = fixture / f'batch-{index:04}.json'
                write_json(path, {'batch_index': index, 'sources': [
                    {'source_id': source, 'batch_index': index, 'tokens': ['evidence']}]})
                manifest['batches'].append({'batch_index': index, 'path': path.name, 'sha256': digest(path)})
                truth.append({'source_id': source, 'label': str(index % 2)})
            write_json(fixture / 'truth.json', truth)
            manifest['ground_truth_sha256'] = digest(fixture / 'truth.json')
            write_json(fixture / 'manifest.json', manifest)
            config = read_json(fixture / 'config-Full.json')
            config['purpose'] = 'research'
            write_json(fixture / 'config.json', config)
            adapter = root / 'adapter.py'
            adapter.write_text('import json,sys\nr=json.load(sys.stdin)\nprint(json.dumps(dict(batch_index=r["batch_index"], evaluation_source_ids=r["evaluation_source_ids"], clusters=[], unassigned_source_ids=r["evaluation_source_ids"])))\n')
            output = root / 'run'
            rows = run(fixture / 'manifest.json', fixture / 'config.json', output, 'direct',
                       os.environ['SECA_TEST_ENGINE'], adapter=[sys.executable, str(adapter)], continue_blocked=True)
            metadata = read_json(output / 'run.json')
            self.assertEqual(metadata['status'], 'scientifically_blocked')
            self.assertEqual(metadata['assignment_cycles_completed'], 12)
            self.assertEqual(len(metadata['scientific_blockers']), 12)
            self.assertEqual(recompute(output)['cycles_verified'], 12)
            self.assertTrue(all(row['nmi'] is None for row in rows))
            self.assertTrue((output / 'assignments-0011.json').is_file())


if __name__ == '__main__':
    unittest.main()
