#!/usr/bin/env python3
"""Run one theme/memory mode/procedure; never tune or launch a hidden matrix."""
import argparse
import subprocess
import shutil
from pathlib import Path
from evaluation.adapters import command_assignments, mock_assignments
from evaluation.contracts import load_manifest, population, validate_config
from evaluation.io import digest, read_json, write_json, write_scores
from evaluation.metrics import DEFINITIONS, evaluate, ScientificBlocker, validate_assignments

ROOT = Path(__file__).resolve().parents[2]


def run(manifest_path, config_path, output, procedure, engine, mock=None, adapter=None, smoke=False, engine_revision=None, continue_blocked=False, learning_from=None):
    if continue_blocked and (mock or smoke):
        raise ValueError('continue-blocked is for full research runs only')
    config = validate_config(read_json(config_path))
    if config['purpose'] == 'mock-smoke' and not mock:
        raise ValueError('mock-smoke requires fixture adapter')
    if mock and config['purpose'] != 'mock-smoke':
        raise ValueError('mock output must be identified as mock-smoke')
    if procedure not in ('direct', 'indirect') or bool(mock) == bool(adapter):
        raise ValueError('choose direct/indirect and exactly one clustering adapter')
    manifest, documents, truth, hashes = load_manifest(manifest_path, smoke)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    engine = Path(engine).resolve()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    metadata = {'status': 'running', 'configuration': config, 'manifest': manifest,
                'dataset_hashes': hashes, 'manifest_sha256': digest(manifest_path),
                'git_revision': revision, 'engine_git_revision': engine_revision or revision, 'git_status': dirty, 'engine_sha256': digest(engine),
                'clustering_procedure': procedure, 'metric_definitions': DEFINITIONS,
                'adapter_command': adapter,
                'adapter_file_hashes': {arg: digest(arg) for arg in (adapter or []) if Path(arg).is_file()}, 'mock': bool(mock), 'smoke': smoke}
    write_json(output / 'run.json', metadata)
    rows = []
    blockers = []
    try:
        inputs = output / 'inputs'
        inputs.mkdir()
        paths = []
        for index, sources in enumerate(documents):
            path = inputs / f'batch-{index:04}.json'
            # Only tokenized evidence enters engine, never evaluator labels/history.
            clean = [{k: s[k] for k in ('source_id', 'batch_index', 'tokens')} for s in sources]
            write_json(path, {'batch_index': index, 'sources': clean})
            paths.append(str(path))
        metadata['engine_input_hashes'] = {Path(p).name: digest(p) for p in paths}
        write_json(output / 'ground-truth.json', truth)
        write_json(output / 'engine-config.json', config['engine'])
        command = [str(engine), 'timeline-many', *paths, '--config', str(output / 'engine-config.json'),
                   '--out-dir', str(output / 'trees')]
        metadata['engine_command'] = command
        if learning_from:
            previous = Path(learning_from).resolve()
            prior = read_json(previous / 'run.json')
            if (prior.get('learning_cycles_completed') != len(documents) or
                    prior['configuration']['engine'] != config['engine'] or
                    prior['dataset_hashes'] != hashes or prior['engine_sha256'] != digest(engine) or
                    prior['engine_input_hashes'] != metadata['engine_input_hashes']):
                raise ValueError('saved learning evidence differs from requested engine/data/configuration')
            metadata['engine_git_revision'] = prior['engine_git_revision']
            metadata['learning_evidence'] = {'run_sha256': digest(previous / 'run.json'),
                'tree_hashes': {p.name: digest(p) for p in sorted((previous / 'trees').glob('tree_batch_*.json'))}}
            if len(metadata['learning_evidence']['tree_hashes']) != len(documents):
                raise ValueError('saved learning evidence lacks cycles')
            shutil.copytree(previous / 'trees', output / 'trees')
            for name in ('engine.stdout', 'engine.stderr'):
                shutil.copyfile(previous / name, output / name)
        else:
            with (output / 'engine.stdout').open('w') as stdout, (output / 'engine.stderr').open('w') as stderr:
                completed = subprocess.run(command, text=True, stdout=stdout, stderr=stderr)
            completed.check_returncode()
        for index in range(len(documents)):
            sources = population(documents, index, config)
            tree = read_json(output / 'trees' / f'tree_batch_{index:04}.json')
            retained_config = dict(config, evaluation_population='retained_window')
            expected_retained = {s['source_id'] for s in population(documents, index, retained_config)}
            actual_retained = {s['external_source_id'] for s in tree['source_legend']}
            if actual_retained != expected_retained:
                raise ValueError(f'cycle {index+1}: exported retained identity population differs from memory contract')
            request = {'schema_version': 1, 'batch_index': index, 'method': procedure,
                       'settings': config['clustering_settings'],
                       'tree': tree,
                       'evaluation_source_ids': [s['source_id'] for s in sources],
                       'source_tokens': {s['source_id']: [t.strip() for t in s['tokens'] if t.strip()] for s in sources}}
            if procedure == 'indirect':
                if not mock and 'contextual_threshold' not in config['clustering_settings']:
                    raise ValueError('indirect adapter requires explicit contextual_threshold')
                request['contextual_threshold'] = config['clustering_settings'].get('contextual_threshold')
            assignments = (mock_assignments(Path(mock), request) if mock else
                           command_assignments(adapter, request))
            write_json(output / f'assignments-{index:04}.json', assignments)
            if assignments['batch_index'] != index or set(assignments['evaluation_source_ids']) != set(request['evaluation_source_ids']):
                raise ValueError('adapter changed requested cycle or evaluation population')
            memberships, labels = validate_assignments(assignments, truth)
            diagnostics = {'population': len(memberships),
                           'assigned': sum(bool(c) for c in memberships.values()),
                           'unassigned': sum(not c for c in memberships.values()),
                           'overlapping': sum(len(c) > 1 for c in memberships.values()),
                           'memberships': sum(len(c) for c in memberships.values()),
                           'maximum_memberships': max(map(len, memberships.values()), default=0),
                           'predicted_clusters': sum(bool(c['source_ids']) for c in assignments['clusters'])}
            write_json(output / f'diagnostics-{index:04}.json', diagnostics)
            try:
                scores = evaluate(assignments, truth)
            except ScientificBlocker as error:
                if not continue_blocked:
                    raise
                blockers.append({'cycle': index + 1, 'reason': str(error)})
                scores = {'evaluated_document_count': len(memberships),
                          'predicted_cluster_count': diagnostics['predicted_clusters'],
                          'ground_truth_event_count': len({labels[s] for s in memberships}),
                          **dict.fromkeys(('nmi', 'bcubed_precision', 'bcubed_recall', 'bcubed_f1', 'purity'))}
            rows.append({'theme': manifest['theme'], 'algorithm': 'SECA' if config['engine']['memory_mode'] == 'Full' else 'SECA-Light',
                         'clustering_procedure': procedure, 'cycle': index + 1, **scores})
            write_scores(output / 'scores.csv', rows)
        metadata['learning_cycles_completed'] = len(documents)
        metadata['assignment_cycles_completed'] = len(rows)
        metadata['scientific_blockers'] = blockers
        metadata['status'] = 'scientifically_blocked' if blockers else 'complete'
    except Exception as error:
        metadata['status'] = 'scientifically_blocked' if isinstance(error, ScientificBlocker) else 'failed'
        metadata['error'] = str(error)
        raise
    finally:
        write_json(output / 'run.json', metadata)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--procedure', choices=['direct', 'indirect'], required=True)
    parser.add_argument('--engine', required=True, type=Path)
    parser.add_argument('--engine-revision', required=True, help='Git revision used to build engine binary')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--learning-from', type=Path, help='rescore saved learning with matching engine/data/configuration')
    parser.add_argument('--continue-blocked', action='store_true', help='preserve every research cycle; undefined scores remain null')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mock-assignments', type=Path)
    group.add_argument('--adapter-command', nargs='+')
    args = parser.parse_args()
    run(args.manifest, args.config, args.output, args.procedure, args.engine,
        args.mock_assignments, args.adapter_command, args.smoke, args.engine_revision, args.continue_blocked, args.learning_from)


if __name__ == '__main__':
    main()
