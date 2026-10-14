#!/usr/bin/env python3
"""Download all four variants at a pinned commit; verify before installing any file."""
import argparse
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from common import HERE, LOCK, verify_bytes


def download(directory, fetch=None):
    def fetch_url(url):
        request = urllib.request.Request(url, headers={'User-Agent': 'seca-dataset-preparation/1'})
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()

    fetch = fetch or fetch_url
    directory = Path(directory)
    verified = {}
    for name, entry in LOCK['files'].items():
        path = directory / name
        url = f'https://raw.githubusercontent.com/jackyin12/MStream/{LOCK["revision"]}/{entry["path"]}'
        data = path.read_bytes() if path.exists() else fetch(url)
        verify_bytes(data, name)  # A corrupt cache fails; it is never silently replaced.
        verified[name] = data
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in verified.items():
        if (directory / name).exists():
            continue
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, directory / name)
        finally:
            temporary_path.unlink(missing_ok=True)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=HERE / 'data/raw')
    args = parser.parse_args()
    try:
        downloaded = download(args.source_dir)
    except (ValueError, OSError, urllib.error.URLError) as error:
        parser.exit(1, f'download failed: {error}\n')
    print(f'Verified {len(downloaded)} pinned source files in {args.source_dir}')


if __name__ == '__main__':
    main()
