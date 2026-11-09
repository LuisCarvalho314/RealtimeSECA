#!/usr/bin/env python3
"""Sequential, same-host, algorithm-only three-way stream benchmark."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ORIGINAL = Path(os.environ.get('SECA_ORIGINAL_ASSEMBLY_DIR',
                               '/Users/lcarv/Downloads/SECA 2/SECA/SECA/bin/Release'))
FIELDS = ['engine', 'repetition', 'batch', 'documents', 'algorithm_seconds',
          'retained_documents', 'hkts', 'nodes', 'refuge_nodes', 'max_depth',
          'source_memberships', 'inspected_hkts', 'reconstructed_hkts',
          'removed_hkts', 'peak_rss_kib', 'process_wall_seconds']


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def prepare(theme, limit, custom_manifest=None):
    manifest_path = custom_manifest or ROOT / f'benchmarks/seca-paper/datasets/manifests/{theme}.json'
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent if custom_manifest else ROOT / 'benchmarks/seca-paper/datasets/data/prepared'
    batches = []
    canonical = hashlib.sha256()
    raw_hashes = []
    seen = set()
    vocabulary = set()
    for item in manifest['batches'][:limit]:
        path = base / item['path']
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item['sha256']:
            raise ValueError(f'input hash mismatch: {path}')
        batch = json.loads(raw)
        if batch['batch_index'] != item['batch_index'] or len(batch['sources']) != item['documents']:
            raise ValueError(f'batch metadata mismatch: {path}')
        for source in batch['sources']:
            sid = source['source_id']
            if sid in seen or source['batch_index'] != batch['batch_index']:
                raise ValueError(f'duplicate or misplaced source: {sid}')
            seen.add(sid)
            words = sorted(set(source['tokens']))
            vocabulary.update(words)
            canonical.update(json.dumps([batch['batch_index'], sid, words],
                                        ensure_ascii=False, separators=(',', ':')).encode())
            canonical.update(b'\n')
        batches.append(str(path))
        raw_hashes.append(digest)
    return batches, {'theme': manifest.get('theme', theme), 'manifest_sha256': hashlib.sha256(
        manifest_path.read_bytes()).hexdigest(),
        'batch_sha256': raw_hashes, 'ordered_token_sets_sha256': canonical.hexdigest(),
        'documents': len(seen), 'vocabulary': len(vocabulary),
        'source_variant': manifest['source_variant'], 'selection': manifest['reconstruction']}


def build(engines):
    subprocess.run(['cargo', 'build', '--release', '-p', 'realtime-seca-v2',
                    '--example', 'performance_stream'], cwd=ROOT, check=True)
    if 'csharp' not in engines:
        return
    if not ORIGINAL.joinpath('SECA.exe').exists():
        raise FileNotFoundError(ORIGINAL / 'SECA.exe')
    out = HERE / '.build'
    out.mkdir(exist_ok=True)
    subprocess.run(['mcs', '-optimize+', f'-out:{out / "OriginalAssemblyRunner.exe"}',
                    f'-r:{ORIGINAL / "SECA.exe"}',
                    f'-r:{ORIGINAL / "SourceCleanerEngine.exe"}',
                    f'-r:{ORIGINAL / "Newtonsoft.Json.dll"}',
                    str(HERE / 'OriginalAssemblyRunner.cs')], check=True)


def measure(engine, batches, config):
    if engine == 'csharp':
        argv = ['mono', str(HERE / '.build/OriginalAssemblyRunner.exe'), str(config), *batches]
        env = dict(os.environ, MONO_PATH=str(ORIGINAL), SECA_INTERNAL_WARMUP='1')
    else:
        argv = [str(ROOT / 'target/release/examples/performance_stream'), engine, str(config), *batches]
        env = os.environ.copy()
    started = time.monotonic()
    proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = proc.communicate()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        raise
    wall = time.monotonic() - started
    if proc.returncode:
        raise RuntimeError(f'{engine} exited {proc.returncode}: {stderr[-3000:]}')
    rows = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
    if len(rows) != len(batches):
        raise RuntimeError(f'{engine} emitted {len(rows)} of {len(batches)} rows; stderr={stderr[-1000:]}')
    for row in rows:
        row['peak_rss_kib'] = max(r['rss_after_batch_kib'] for r in rows)
        row['process_wall_seconds'] = wall
    return rows


def summarize(rows):
    summary = []
    engines = dict.fromkeys(row['engine'] for row in rows)
    for engine in engines:
        group = [r for r in rows if r['engine'] == engine]
        reps = sorted(set(r['repetition'] for r in group))
        totals = [sum(r['algorithm_seconds'] for r in group if r['repetition'] == rep) for rep in reps]
        cpu_totals = [sum(r['algorithm_cpu_seconds'] for r in group if r['repetition'] == rep) for rep in reps]
        updates = [r['algorithm_seconds'] for r in group if r['batch'] > 0]
        baselines = [r['algorithm_seconds'] for r in group if r['batch'] == 0]
        memory = [max(r['peak_rss_kib'] for r in group if r['repetition'] == rep) for rep in reps]
        process_walls = [next(r['process_wall_seconds'] for r in group if r['repetition'] == rep) for rep in reps]
        stream_walls = [max(r['stream_wall_seconds'] for r in group if r['repetition'] == rep) for rep in reps]
        p95 = lambda v: sorted(v)[min(len(v)-1, int(.95 * len(v)))]
        final = [r for r in group if r['batch'] == max(x['batch'] for x in group)]
        summary.append({'engine': engine, 'repetitions': len(reps),
                        'baseline_median_seconds': statistics.median(baselines),
                        'baseline_mean_seconds': statistics.mean(baselines),
                        'baseline_std_seconds': statistics.stdev(baselines) if len(baselines)>1 else None,
                        'baseline_p95_seconds': p95(baselines),
                        'total_median_seconds': statistics.median(totals),
                        'total_mean_seconds': statistics.mean(totals),
                        'total_std_seconds': statistics.stdev(totals) if len(totals)>1 else None,
                        'total_p95_seconds': p95(totals),
                        'total_cpu_median_seconds': statistics.median(cpu_totals),
                        'cpu_to_wall_ratio_median': statistics.median(c/t for c,t in zip(cpu_totals, totals)),
                        'process_wall_median_seconds': statistics.median(process_walls),
                        'process_wall_mean_seconds': statistics.mean(process_walls),
                        'process_wall_std_seconds': statistics.stdev(process_walls) if len(process_walls)>1 else None,
                        'process_wall_p95_seconds': p95(process_walls),
                        'stream_wall_median_seconds': statistics.median(stream_walls),
                        'stream_wall_mean_seconds': statistics.mean(stream_walls),
                        'stream_wall_std_seconds': statistics.stdev(stream_walls) if len(stream_walls)>1 else None,
                        'stream_wall_p95_seconds': p95(stream_walls),
                        'batch_median_seconds': statistics.median(updates),
                        'batch_mean_seconds': statistics.mean(updates),
                        'batch_std_seconds': statistics.stdev(updates) if len(updates)>1 else None,
                        'batch_p95_seconds': p95(updates),
                        'peak_rss_median_kib': statistics.median(memory),
                        'peak_rss_mean_kib': statistics.mean(memory),
                        'peak_rss_std_kib': statistics.stdev(memory) if len(memory)>1 else None,
                        'peak_rss_p95_kib': p95(memory),
                        'final_hkts_median': statistics.median(r['hkts'] for r in final),
                        'final_nodes_median': statistics.median(r['nodes'] for r in final),
                        'final_source_memberships_median': statistics.median(r['source_memberships'] for r in final),
                        'total_reconstructed_targets_median': (statistics.median(
                            sum(r['reconstructed_hkts'] for r in group if r['repetition'] == rep)
                            for rep in reps) if all(r['reconstructed_hkts'] is not None for r in group) else None)})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', choices=['G', 'H'], default='G')
    parser.add_argument('--mode', choices=['Full', 'SlidingWindow'], default='Full')
    parser.add_argument('--batches', type=int, default=12)
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--engines', nargs='+', choices=['csharp', 'v1', 'v2'], default=['csharp', 'v1', 'v2'])
    parser.add_argument('--manifest', type=Path, help='custom manifest with batch paths relative to itself')
    parser.add_argument('--config', type=Path, help='override configuration JSON')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode != 'Full' and 'csharp' in args.engines:
        parser.error('C# Light memory semantics differ; no equivalent adapter is available')
    if args.repetitions < 1 or args.batches < 1:
        parser.error('positive repetitions and batches required')
    args.output.mkdir(parents=True, exist_ok=True)
    batches, provenance = prepare(args.theme, args.batches, args.manifest)
    config = args.config or ROOT / f'benchmarks/seca-v2/evidence/{args.mode}-config.json'
    build(args.engines)
    meta = {'git_sha': command('git', 'rev-parse', 'HEAD'), 'platform': platform.platform(),
            'machine': platform.machine(), 'processor': platform.processor(),
            'logical_cpu_count': os.cpu_count(),
            'rustc': command('rustc', '--version'), 'cargo': command('cargo', '--version'),
            'rust_profile': 'release',
            'config': json.loads(config.read_text()), 'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
            'warmup': args.warmup, 'repetitions': args.repetitions,
            'engines': args.engines, 'dataset': provenance,
            'threading': 'C# original Parallel.ForEach default Mono thread pool; Rust single thread',
            'rust_adapter_sha256': hashlib.sha256((ROOT / 'crates/realtime-seca-v2/examples/performance_stream.rs').read_bytes()).hexdigest(),
            'rust_binary_sha256': hashlib.sha256((ROOT / 'target/release/examples/performance_stream').read_bytes()).hexdigest()}
    if 'csharp' in args.engines:
        meta['csharp_build'] = 'original precompiled Release SECA.exe, mcs optimized adapter; in-process first-two-batch JIT warmup'
        meta['mono'] = command('mono', '--version').splitlines()[0]
        meta['mcs'] = command('mcs', '--version')
        meta['csharp_assembly_sha256'] = hashlib.sha256((ORIGINAL / 'SECA.exe').read_bytes()).hexdigest()
        meta['csharp_source_sha256'] = {
            name: hashlib.sha256((ORIGINAL.parent.parent / 'SECA_DataClass' / name).read_bytes()).hexdigest()
            for name in ['SECAAlgorithm.cs', 'HKT.cs', 'Node.cs', 'AlphaError.cs',
                         'BetaError.cs', 'WordImportanceError.cs']}
    (args.output / 'provenance.json').write_text(json.dumps(meta, indent=2) + '\n')
    rows = []
    for engine in args.engines:
        for rep in range(-args.warmup, args.repetitions):
            current = measure(engine, batches, config)
            if rep >= 0:
                for row in current:
                    row['repetition'] = rep
                    rows.append(row)
                (args.output / 'runs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            print(f'{engine} {rep}: {sum(r["algorithm_seconds"] for r in current):.3f}s', flush=True)
    summary = summarize(rows)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (args.output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


if __name__ == '__main__':
    main()
