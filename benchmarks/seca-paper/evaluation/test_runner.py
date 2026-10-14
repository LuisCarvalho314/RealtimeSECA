import copy
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from evaluation.adapters import command_assignments
from evaluation.comparison import compare_runs, compare_published
from evaluation.contracts import load_manifest, population, validate_config
from evaluation.io import digest, parse_json, read_json, write_json, write_scores, SCORES
from evaluation.make_fixture import make_fixture
from evaluation.metrics import evaluate, DEFINITIONS

RUNNER_PATH = Path(__file__).resolve().parents[1] / 'run.py'
spec = importlib.util.spec_from_file_location('seca_runner', RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        make_fixture(self.root / 'fixture')
        self.fixture = self.root / 'fixture'
        self.config = read_json(self.fixture / 'config-Full.json')

    def test_configuration(self):
        validate_config(self.config)
        for mutate in (lambda c: c['engine']['seca_thresholds'].pop('beta'),
                       lambda c: c['engine']['hkt_builder'].pop('similarity_threshold'),
                       lambda c: c['parameter_sources'].pop('gamma'),
                       lambda c: c.update(purpose='paper'),
                       lambda c: c.update(evaluation_population='implicit'),
                       lambda c: c['engine'].update(memory_mode='SlidingWindow'),
                       lambda c: c['engine'].update(max_batches_in_memory=2)):
            c = copy.deepcopy(self.config); mutate(c)
            with self.assertRaises(ValueError):
                validate_config(c)
        validate_config(read_json(self.fixture / 'config-SlidingWindow.json'))

    def test_manifest_and_hashes(self):
        manifest, documents, truth, hashes = load_manifest(self.fixture / 'manifest.json', True)
        self.assertEqual(len(documents), 2)
        self.assertEqual(len(truth), 4)
        self.assertEqual(len(hashes), 3)
        with self.assertRaises(ValueError):
            load_manifest(self.fixture / 'manifest.json')
        batch = self.fixture / manifest['batches'][0]['path']
        batch.write_text(batch.read_text() + ' ')
        with self.assertRaises(ValueError):
            load_manifest(self.fixture / 'manifest.json', True)

    def test_manifest_sequence_identity_and_leakage(self):
        for kind in ('sequence', 'duplicate', 'leakage', 'truth', 'unknown_source'):
            make_fixture(self.fixture)
            manifest = read_json(self.fixture / 'manifest.json')
            batch_path = self.fixture / 'batch-0001.json'
            batch = read_json(batch_path)
            if kind == 'sequence':
                manifest['batches'].reverse()
            elif kind == 'duplicate':
                batch['sources'][0]['source_id'] = 'fixture-0-0'
            elif kind == 'leakage':
                batch['sources'][0]['metadata'] = {'label': 'alpha'}
            elif kind == 'unknown_source':
                batch['sources'][0]['label'] = 'alpha'
            elif kind == 'truth':
                truth_path = self.fixture / 'truth.json'
                write_json(truth_path, read_json(truth_path)[:-1])
                manifest['ground_truth_sha256'] = digest(truth_path)
            write_json(batch_path, batch)
            manifest['batches'][1 if kind != 'sequence' else 0]['sha256'] = digest(batch_path)
            write_json(self.fixture / 'manifest.json', manifest)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                load_manifest(self.fixture / 'manifest.json', True)

    def test_agent1_manifest_path_base(self):
        theme_dir = self.fixture / 'G'
        make_fixture(theme_dir)
        manifest = read_json(theme_dir / 'manifest.json')
        manifest['path_base'] = 'selected_benchmark_data_directory'
        for entry in manifest['batches']:
            entry['path'] = 'G/' + entry['path']
        manifest['ground_truth_path'] = 'G/truth.json'
        write_json(theme_dir / 'manifest.json', manifest)
        self.assertEqual(len(load_manifest(theme_dir / 'manifest.json', True)[1]), 2)

    def test_file_command_adapter(self):
        script = self.root / 'adapter.py'
        script.write_text('import pathlib,sys\nprint(pathlib.Path(sys.argv[1]).read_text())\n')
        request = {'method': 'direct', 'source_tokens': {'source': ['word']}}
        self.assertEqual(command_assignments([sys.executable, str(script), '{request}'], request), request)

    def test_evaluation_population_separate_from_learning_window(self):
        _, documents, _, _ = load_manifest(self.fixture / 'manifest.json', True)
        config = read_json(self.fixture / 'config-SlidingWindow.json')
        self.assertEqual(len(population(documents, 1, config)), 2)
        config['evaluation_population'] = 'cumulative'
        self.assertEqual(len(population(documents, 1, config)), 4)
        config['evaluation_population'] = 'retained_window'
        self.assertEqual(len(population(documents, 1, config)), 2)
        self.assertEqual(len(documents[0]), 2)

    def test_strict_serialization(self):
        with self.assertRaises(ValueError):
            parse_json('{"x":1,"x":2}')
        with self.assertRaises(ValueError):
            parse_json('{"x":NaN}')
        path = self.root / 'result.json'
        write_json(path, {'nmi': None, 'value': 2/3})
        self.assertEqual(read_json(path), {'nmi': None, 'value': 2/3})
        first = path.read_bytes(); write_json(path, read_json(path))
        self.assertEqual(first, path.read_bytes())

    def completed_run(self, name, count=2):
        path = self.root / name; path.mkdir()
        manifest = read_json(self.fixture / 'manifest.json')
        manifest['batches'] = [{}] * count
        write_json(path / 'run.json', {'status': 'complete', 'manifest': manifest,
            'configuration': self.config, 'dataset_hashes': {}, 'metric_definitions': DEFINITIONS,
            'clustering_procedure': 'direct', 'mock': True, 'smoke': True, 'git_revision': name})
        truth = read_json(self.fixture / 'truth.json')
        scores = evaluate(read_json(self.fixture / 'assignments-0000.json'), truth)
        rows = [{'theme': 'G', 'algorithm': 'SECA', 'clustering_procedure': 'direct', 'cycle': i+1,
                 **scores} for i in range(count)]
        write_scores(path / 'scores.csv', rows)
        return path, rows

    def test_comparison(self):
        a, _ = self.completed_run('a')
        b, rows = self.completed_run('b')
        self.assertTrue(all(d['difference'] == 0 for d in compare_runs(a, b)['differences']))
        rows[1]['nmi'] = .75
        write_scores(b / 'scores.csv', rows)
        differences = compare_runs(a, b)['differences']
        self.assertEqual([d['difference'] for d in differences if d['cycle'] == 2 and d['metric'] == 'nmi'], [-.25])
        metadata = read_json(b / 'run.json'); metadata['status'] = 'failed'; write_json(b / 'run.json', metadata)
        with self.assertRaises(ValueError):
            compare_runs(a, b)

    def test_published_comparison(self):
        short, _ = self.completed_run('short')
        with self.assertRaises(ValueError):
            compare_published(short)
        complete, _ = self.completed_run('complete', 12)
        result = compare_published(complete)
        self.assertEqual(result['comparisons'][0]['published'], .8621)
        self.assertFalse(result['parity_claim'])
        self.assertTrue(result['mock'])

    def test_mock_assignment_evaluation(self):
        _, _, truth, _ = load_manifest(self.fixture / 'manifest.json', True)
        self.assertEqual(evaluate(read_json(self.fixture / 'assignments-0000.json'), truth)['purity'], 1)

    @unittest.skipUnless(os.environ.get('SECA_TEST_ENGINE'), 'set SECA_TEST_ENGINE for real engine smoke')
    def test_real_engine_mock_smoke(self):
        for mode in ('Full', 'SlidingWindow'):
            for procedure in ('direct', 'indirect'):
                output = self.root / f'{mode}-{procedure}'
                rows = runner.run(self.fixture / 'manifest.json', self.fixture / f'config-{mode}.json',
                                  output, procedure, os.environ['SECA_TEST_ENGINE'], mock=self.fixture, smoke=True)
                self.assertEqual(len(rows), 2)
                self.assertEqual(read_json(output / 'run.json')['status'], 'complete')
                tree = read_json(output / 'trees/tree_batch_0001.json')
                self.assertEqual(len(tree['source_legend']), 4 if mode == 'Full' else 2)
                self.assertEqual(len(read_json(output / 'ground-truth.json')), 4)
                self.assertEqual(read_json(output / 'assignments-0001.json')['batch_index'], 1)

    @unittest.skipUnless(os.environ.get('SECA_TEST_ENGINE'), 'set SECA_TEST_ENGINE for real engine smoke')
    def test_scientific_failure_preserves_raw_memberships(self):
        assignments = read_json(self.fixture / 'assignments-0000.json')
        assignments['clusters'].append({'cluster_id': 'overlap', 'source_ids': ['fixture-0-0']})
        write_json(self.fixture / 'assignments-0000.json', assignments)
        output = self.root / 'blocked'
        with self.assertRaises(ValueError):
            runner.run(self.fixture / 'manifest.json', self.fixture / 'config-Full.json', output,
                       'direct', os.environ['SECA_TEST_ENGINE'], mock=self.fixture, smoke=True)
        self.assertEqual(read_json(output / 'run.json')['status'], 'scientifically_blocked')
        self.assertEqual(read_json(output / 'assignments-0000.json'), assignments)
        with self.assertRaises(ValueError):
            compare_runs(output, output)


if __name__ == '__main__':
    unittest.main()
