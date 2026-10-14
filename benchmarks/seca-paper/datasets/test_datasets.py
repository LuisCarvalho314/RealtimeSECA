"""Regression tests use all four real pinned files; download first. No network in tests."""
import collections
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audit_history import audit
from common import (HERE, LOCK, THEMES, encode, load_sources, parse_rows,
                    source_id)
from download import download
from prepare import build_theme, prepare
from verify import verify


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = Path(os.environ.get('SECA_DATASET_SOURCE_DIR', HERE / 'data/raw'))
        cls.rows = load_sources(cls.raw)  # Missing public sources fail; never skip validation.
        cls.files = {}
        for theme, (name, _) in THEMES.items():
            cls.files.update(build_theme(theme, cls.rows[name]))

    def test_exact_sizes_ids_labels_and_no_label_leakage(self):
        all_ids = set()
        for theme, (name, size) in THEMES.items():
            manifest = json.loads(self.files[f'{theme}/manifest.json'])
            truth = json.loads(self.files[manifest['ground_truth_path']])
            self.assertEqual(len(manifest['batches']), 12)
            self.assertEqual(len(truth), 12 * size)
            observed = []
            for index, entry in enumerate(manifest['batches']):
                batch = json.loads(self.files[entry['path']])
                self.assertEqual(set(batch), {'batch_index', 'sources'})
                self.assertEqual(batch['batch_index'], index)
                self.assertEqual(len(batch['sources']), size)
                for record in batch['sources']:
                    self.assertEqual(set(record), {'source_id', 'batch_index', 'tokens', 'text',
                                                  'timestamp_unix_ms', 'metadata'})
                    self.assertEqual(record['batch_index'], index)
                    self.assertIsNone(record['metadata'])
                    self.assertIsNone(record['timestamp_unix_ms'])
                    self.assertNotIn(record['source_id'], all_ids)
                    all_ids.add(record['source_id'])
                    observed.append(record)
            expected = self.rows[name][:12 * size]
            self.assertEqual([r['source_id'] for r in observed],
                             [source_id(name, r['Id']) for r in expected])
            for record, original, label in zip(observed, expected, truth):
                self.assertEqual(record['text'], original['textCleaned'])
                self.assertEqual(record['tokens'], original['textCleaned'].split())
                self.assertEqual(label, {'source_id': record['source_id'],
                                         'label': str(original['clusterNo'])})
                self.assertTrue(label['label'])
            self.assertFalse(manifest['reconstruction']['exact_paper_reconstruction'])
            self.assertEqual(manifest['reconstruction']['discarded_tail_documents'],
                             len(self.rows[name]) - size * 12)

    def test_source_sha256_and_order(self):
        for name, entry in LOCK['files'].items():
            data = (self.raw / name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry['sha256'])
            self.assertEqual(len(data), entry['bytes'])
            original = [json.loads(line) for line in data.splitlines()]
            self.assertEqual(self.rows[name], original)

    def test_variants_are_not_interchangeable(self):
        for base in ('News', 'Tweets'):
            def multiset(rows):
                return collections.Counter((r['clusterNo'], r['textCleaned']) for r in rows)
            self.assertEqual(multiset(self.rows[base]), multiset(self.rows[base + '-T']))
            self.assertNotEqual(self.rows[base], self.rows[base + '-T'])
        def id_map(name):
            return {r['Id']: (r['clusterNo'], r['textCleaned']) for r in self.rows[name]}
        self.assertEqual(id_map('News'), id_map('News-T'))
        self.assertNotEqual(id_map('Tweets'), id_map('Tweets-T'))
        for name, count, labels, minimum, maximum in (
                ('News', 8880, 152, 2, 354), ('News-T', 8880, 98, 3, 430),
                ('Tweets', 24000, 255, 1, 607), ('Tweets-T', 24000, 227, 1, 607)):
            counts = collections.Counter(r['clusterNo'] for r in self.rows[name][:count])
            self.assertEqual((len(counts), min(counts.values()), max(counts.values())),
                             (labels, minimum, maximum))

    def test_duplicates_and_repeated_tokens_are_preserved(self):
        for theme, (name, size) in THEMES.items():
            original = self.rows[name][:12 * size]
            self.assertGreater(len(original) - len({r['textCleaned'] for r in original}), 0)
            records = [r for i in range(12)
                       for r in json.loads(self.files[f'{theme}/batch-{i:02d}.json'])['sources']]
            self.assertEqual(collections.Counter(r['text'] for r in records),
                             collections.Counter(r['textCleaned'] for r in original))
            self.assertTrue(any(len(r['tokens']) != len(set(r['tokens'])) for r in records))

    def test_determinism_and_tracked_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / 'one', Path(directory) / 'two'
            prepare(self.raw, first)
            prepare(self.raw, second)
            self.assertEqual(verify(self.raw, first), 28)
            self.assertEqual(verify(self.raw, second), 28)
            for path, data in self.files.items():
                self.assertEqual((first / path).read_bytes(), data)
                self.assertEqual((second / path).read_bytes(), data)
            for theme in THEMES:
                self.assertEqual((HERE / f'manifests/{theme}.json').read_bytes(),
                                 self.files[f'{theme}/manifest.json'])
            with self.assertRaisesRegex(ValueError, 'already exists'):
                prepare(self.raw, first)
            self.assertEqual(verify(self.raw, first), 28)

    def test_manifest_paths_and_generated_checksums(self):
        for theme in THEMES:
            manifest = json.loads(self.files[f'{theme}/manifest.json'])
            entries = [(b['path'], b['sha256']) for b in manifest['batches']]
            entries.append((manifest['ground_truth_path'], manifest['ground_truth_sha256']))
            for path, digest in entries:
                self.assertFalse(Path(path).is_absolute())
                self.assertNotIn('..', Path(path).parts)
                self.assertEqual(hashlib.sha256(self.files[path]).hexdigest(), digest)
            self.assertEqual(manifest['source']['revision'], LOCK['revision'])

    def test_corrupt_missing_and_reordered_sources_fail_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, out = Path(directory) / 'raw', Path(directory) / 'out'
            raw.mkdir()
            for name in LOCK['files']:
                (raw / name).write_bytes((self.raw / name).read_bytes())
            for name in LOCK['files']:
                path = raw / name
                original = path.read_bytes()
                for bad in (original + b' ', b'not json', b'\n'.join(reversed(original.splitlines()))):
                    path.write_bytes(bad)
                    with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                        prepare(raw, out)
                    self.assertFalse(out.exists())
                path.unlink()
                with self.assertRaisesRegex(ValueError, 'missing source file'):
                    prepare(raw, out)
                self.assertFalse(out.exists())
                path.write_bytes(original)

    def test_strict_parser_and_whitespace(self):
        row = {'Id': '000001', 'clusterNo': 42, 'textCleaned': 'A\tB  B\nC'}
        parsed = parse_rows(json.dumps(row).encode(), 'fixture')
        self.assertEqual(parsed[0]['textCleaned'].split(), ['A', 'B', 'B', 'C'])
        for invalid in ({**row, 'clusterNo': True}, {**row, 'clusterNo': ''},
                        {**row, 'textCleaned': ' '}, {**row, 'Id': ''}, {**row, 'extra': 'x'}):
            with self.assertRaises(ValueError):
                parse_rows(json.dumps(invalid).encode(), 'fixture')
        with self.assertRaisesRegex(ValueError, 'duplicate Id'):
            parse_rows((json.dumps(row) + '\n' + json.dumps(row)).encode(), 'fixture')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            parse_rows(b'{"Id":"a","Id":"b"}', 'fixture')
        with self.assertRaisesRegex(ValueError, 'expected at least'):
            build_theme('G', parsed)

    def test_downloader_pinned_urls_cache_and_corruption(self):
        calls = []
        def fetch(url):
            calls.append(url)
            name = url.rsplit('/', 1)[-1]
            self.assertIn('/' + LOCK['revision'] + '/', url)
            return (self.raw / name).read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / 'raw'
            download(raw, fetch)
            self.assertEqual(len(calls), 4)
            download(raw, lambda _: self.fail('valid cache should avoid network'))
            (raw / 'News').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                download(raw, fetch)
            self.assertEqual((raw / 'News').read_bytes(), b'corrupt')
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / 'raw'
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                download(raw, lambda _: b'corrupt')
            self.assertFalse(raw.exists())

    def test_output_failures_do_not_publish_partial_data(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out'
            with patch('prepare.os.rename', side_effect=OSError('simulated publish failure')):
                with self.assertRaisesRegex(OSError, 'simulated'):
                    prepare(self.raw, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_prepared_corruption_missing_extra_and_labels_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out'
            prepare(self.raw, output)
            path = output / 'G/batch-00.json'
            original = path.read_bytes()
            batch = json.loads(original)
            batch['sources'][0]['metadata'] = {'label': '65'}
            path.write_bytes(encode(batch))
            with self.assertRaisesRegex(ValueError, 'prepared content'):
                verify(self.raw, output)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
                verify(self.raw, output)
            path.write_bytes(original)
            (output / 'extra.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
                verify(self.raw, output)

    def test_historical_native_ids_chronology_and_corruption(self):
        current = [{'Id': '000001', 'clusterNo': 1, 'textCleaned': 'first'},
                   {'Id': '000002', 'clusterNo': 2, 'textCleaned': 'second'}]
        old = [{'tweetId': '30195992632496128', 'clusterNo': 1, 'textCleaned': 'first',
                'timeStr': 'Wed Jan 26 09:31:00 +0000 2011'},
               {'tweetId': '30195992632496129', 'clusterNo': 2, 'textCleaned': 'second',
                'timeStr': 'Wed Jan 26 02:42:24 +0000 2011'}]
        data = ('\n'.join(json.dumps(row) for row in old) + '\n').encode()
        facts = {'path': 'MStream/data/TREC-T', 'sha256': hashlib.sha256(data).hexdigest(),
                 'documents': 2, 'current_variant': 'Tweets-T',
                 'same_ordered_cleaned_text': True, 'same_ordered_labels': True,
                 'same_ordered_ids': False, 'legacy_id_field': 'tweetId',
                 'duplicate_legacy_ids': 0, 'timestamp_field': 'timeStr',
                 'timestamp_descents': 1, 'chronologically_nondecreasing': False}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = {'revision': 'pinned-history', 'files': {'TREC-T': facts}}
            (root / 'historical-audit.json').write_bytes(encode(reference))
            with patch('audit_history.HERE', root), \
                    patch('audit_history.load_sources', return_value={'Tweets-T': current}), \
                    patch('audit_history.subprocess.check_output', return_value=data) as git:
                identities = json.loads(audit(root, root))
                self.assertIn('pinned-history:MStream/data/TREC-T', git.call_args.args[0])
                self.assertEqual(identities[0]['source_id'], 'mstream:Tweets-T:000001')
                self.assertEqual(identities[0]['legacy_tweetId'], old[0]['tweetId'])
                self.assertEqual(identities[1]['legacy_timeStr'], old[1]['timeStr'])
                self.assertEqual(set(identities[0]), {'source_id', 'legacy_tweetId', 'legacy_timeStr'})
                git.return_value = b'corrupt'
                with self.assertRaisesRegex(ValueError, 'historical SHA-256 mismatch'):
                    audit(root, root)
                git.return_value = data
                current[0]['clusterNo'] = 2
                with self.assertRaisesRegex(ValueError, 'historical audit differs'):
                    audit(root, root)

    def test_cli_missing_sources_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out'
            result = subprocess.run([sys.executable, str(HERE / 'prepare.py'),
                                     '--source-dir', directory, '--output-dir', str(output)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('missing source file', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
