"""Focused checks for input equivalence and manifest rejection."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from run import prepare
from merge_measurements import main as merge_measurements
from verify_inputs import verify


class InputVerificationTests(unittest.TestCase):
    def test_duplicate_tokens_preserve_one_document_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory) / 'batch.json'
            batch.write_text(json.dumps({'batch_index': 0, 'sources': [
                {'source_id': 'document:1', 'batch_index': 0, 'tokens': ['a', 'b', 'a']},
                {'source_id': 'document:2', 'batch_index': 0, 'tokens': ['b', 'c']}]}))
            result = verify([str(batch)])
            self.assertEqual(result['documents'], 2)
            self.assertEqual(result['vocabulary'], 3)
            self.assertEqual(len(set(result['ordered_token_sets_sha256_by_adapter'].values())), 1)

    def test_corrupt_batch_fails_before_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            batch = folder / 'batch.json'
            batch.write_text(json.dumps({'batch_index': 0, 'sources': [
                {'source_id': 'document:1', 'batch_index': 0, 'tokens': ['a']}]}))
            manifest = folder / 'manifest.json'
            manifest.write_text(json.dumps({'theme': 'synthetic', 'source_variant': 'test',
                'reconstruction': {}, 'batches': [{'batch_index': 0, 'path': 'batch.json',
                'documents': 1, 'sha256': hashlib.sha256(batch.read_bytes()).hexdigest()}]}))
            paths, _ = prepare('G', 1, manifest)
            self.assertEqual(paths, [str(batch)])
            batch.write_text(batch.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'input hash mismatch'):
                prepare('G', 1, manifest)

    def test_merge_uses_hashes_recorded_with_rust_run(self):
        with tempfile.TemporaryDirectory() as directory:
            base, rust, output = (Path(directory) / name for name in ('base', 'rust', 'merged'))
            base.mkdir()
            rust.mkdir()
            meta = {key: 'same' for key in ('git_sha', 'platform', 'machine', 'processor',
                    'logical_cpu_count', 'rustc', 'cargo', 'rust_profile', 'config_sha256')}
            meta.update(dataset={'batch_sha256': ['batch']}, repetitions=1,
                        rust_adapter_sha256='old-source')
            (base / 'provenance.json').write_text(json.dumps(meta))
            (rust / 'provenance.json').write_text(json.dumps({
                **meta, 'rust_adapter_sha256': 'measured-source',
                'rust_binary_sha256': 'measured-binary'}))
            (base / 'runs.jsonl').write_text(json.dumps({'engine': 'csharp-original-assembly'}) + '\n')
            (rust / 'runs.jsonl').write_text(''.join(json.dumps({'engine': engine}) + '\n'
                    for engine in ('v1', 'v2')))
            merge_measurements(base, rust, output)
            merged = json.loads((output / 'provenance.json').read_text())
            self.assertEqual(merged['rust_adapter_sha256'], 'measured-source')
            self.assertEqual(merged['rust_binary_sha256'], 'measured-binary')


if __name__ == '__main__':
    unittest.main()
