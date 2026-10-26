"""Build the same read-only adapter against another existing Rust checkout.

Only writes the specified output directory; does not edit the checkout.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('checkout', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
here = Path(__file__).resolve().parent
args.output.mkdir(parents=True, exist_ok=True)
(args.output / 'src').mkdir(exist_ok=True)
shutil.copyfile(here / 'rust-baseline/src/main.rs', args.output / 'src/main.rs')
manifest = (here / 'rust-baseline/Cargo.toml').read_text()
manifest = manifest.replace('"../../../../crates/realtime-seca-core"',
                            json.dumps(str(args.checkout.resolve() / 'crates/realtime-seca-core')))
(args.output / 'Cargo.toml').write_text(manifest)
subprocess.run(['cargo', 'build', '--offline', '--release', '--manifest-path',
                str(args.output / 'Cargo.toml')], check=True)
print(args.output.resolve() / 'target/release/seca-exploration-baseline')
