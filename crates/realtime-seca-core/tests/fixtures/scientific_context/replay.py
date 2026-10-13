"""Verify owned C# captures without modifying the shared harness or goldens."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
for path in sorted(HERE.glob('*.input.json')):
    name = path.name.removesuffix('.input.json')
    runs = [subprocess.run([str(ROOT / 'tools/seca-reference-harness/run.sh'), str(path)],
                          cwd=ROOT, capture_output=True, text=True) for _ in range(2)]
    for result in runs:
        if name == 'shared_asymmetric':
            assert result.returncode != 0 and not result.stdout.strip()
            assert 'same key' in result.stderr.lower()
        else:
            assert result.returncode == 0, result.stderr
            expected_errors = {
                'expected_history': [.4, .3333, 0, .5714],
                'expected_trigger': [.4, .3333],
                'historical_beta': [.4, 0, .4286, .4],
                'historical_beta_trigger': [.4, 0, 0, 0],
            }
            snapshots = json.loads(result.stdout)['snapshots'][1:]
            for snapshot, expected in zip(snapshots, expected_errors[name], strict=True):
                assert len(snapshot['decisions']) == 1
                errors = snapshot['decisions'][0]['errors']
                assert errors['alphaError']['option1_error'] == 0
                assert errors['betaError']['option1_error'] == 0
                assert errors['wordImportanceError']['option1_error'] == expected
            assert json.loads(result.stdout) == json.loads((HERE / (name + '.csharp.json')).read_text())
    print(name, 'verified twice')
