"""Verify actual reference outputs; --capture explicitly replaces audit goldens."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
capture = "--capture" in sys.argv
for path in sorted(HERE.glob("*.input.json")):
    name = path.name.removesuffix(".input.json")
    runs = [subprocess.run(
        [str(ROOT / "tools/seca-reference-harness/run.sh"), str(path)],
        cwd=ROOT, capture_output=True, text=True) for _ in range(2)]
    assert runs[0].returncode == runs[1].returncode, name
    if runs[0].returncode:
        marker = "same key" if name == "shared_candidate" else "nonempty"
        assert marker.lower() in runs[0].stderr.lower(), (name, runs[0].stderr)
        assert not runs[0].stdout.strip(), name
        if capture:
            (HERE / f"{name}.stderr.txt").write_text("".join(line.rstrip() + "\n" for line in runs[0].stderr.splitlines()))
        print(name, "reference failure:", marker)
        continue
    assert runs[0].stdout == runs[1].stdout, name
    output = json.dumps(json.loads(runs[0].stdout), separators=(",", ":")) + "\n"
    golden = HERE / f"{name}.csharp.json"
    if capture:
        golden.write_text(output)
        (HERE / f"{name}.stderr.txt").write_text("".join(line.rstrip() + "\n" for line in runs[0].stderr.splitlines()))
    else:
        assert golden.read_text() == output, name
    print(name, hashlib.sha256(output.encode()).hexdigest())
if capture:
    hashes = ROOT / "references/seca-parity-build/.harness-generated/source-hashes.json"
    (HERE / "source-hashes.json").write_text(hashes.read_text())
