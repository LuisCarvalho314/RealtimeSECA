"""Clustering command protocol; ground truth is never sent to adapters."""
import json
import subprocess
import tempfile
from pathlib import Path
from .io import write_json
from .io import read_json


def mock_assignments(directory, request):
    """Replay authored fixtures only; this is not a clustering algorithm."""
    return read_json(directory / f"assignments-{request['batch_index']:04}.json")


def command_assignments(command, request):
    with tempfile.TemporaryDirectory(prefix='seca-clustering-') as directory:
        path = Path(directory) / 'request.json'
        write_json(path, request)
        expanded = [str(path) if arg == '{request}' else arg for arg in command]
        completed = subprocess.run(expanded, input=json.dumps(request, allow_nan=False),
                                   text=True, capture_output=True, check=True)
    # Same strict parser as artifact inputs, without filesystem staging.
    from .io import parse_json
    return parse_json(completed.stdout)
