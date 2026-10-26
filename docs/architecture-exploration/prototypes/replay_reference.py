"""Read-only verification of captured C# construction outputs; never recapture."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
results = []
fixtures = ROOT / 'crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction'
for path in sorted(fixtures.glob('*/input.json')):
    observed = path.parent / 'csharp.json'
    if not observed.exists():
        continue
    process = subprocess.run([str(ROOT / 'tools/seca-reference-harness/run.sh'), str(path)],
                             capture_output=True, timeout=30)
    matches = process.returncode == 0 and json.loads(process.stdout) == json.loads(observed.read_text())
    results.append({'fixture':path.parent.name, 'exit':process.returncode, 'matches':matches})
print(json.dumps(results, indent=2))
if not all(row['matches'] for row in results):
    raise SystemExit(1)
