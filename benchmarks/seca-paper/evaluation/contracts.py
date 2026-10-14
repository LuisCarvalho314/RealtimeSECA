"""Versioned manifest and configuration boundary (no Rust defaults)."""
from pathlib import Path
from .io import digest, read_json
from .metrics import unique_ids

PAPER_THRESHOLDS = dict(alpha=0.7, beta=0.5, alpha_error_threshold=0.1,
                        beta_error_threshold=0.2, word_importance_error_threshold=0.3)


def validate_config(config):
    required = {'engine', 'evaluation_population', 'clustering_settings', 'parameter_sources', 'purpose'}
    if set(config) != required:
        raise ValueError(f'configuration requires exactly {sorted(required)}')
    if config['purpose'] not in ('mock-smoke', 'research'):
        raise ValueError('paper reproduction blocked: population, overlap, gamma and clustering settings unresolved')
    if config['evaluation_population'] not in ('current_batch', 'retained_window', 'cumulative'):
        raise ValueError('explicit evaluation population required')
    engine = config['engine']
    if set(engine) != {'hkt_builder', 'seca_thresholds', 'memory_mode', 'max_batches_in_memory'}:
        raise ValueError('all engine fields must be explicit')
    if engine['seca_thresholds'] != PAPER_THRESHOLDS:
        raise ValueError('use reported paper thresholds; do not tune against results')
    builder = engine['hkt_builder']
    if set(builder) != {'minimum_threshold_against_max_word_count', 'similarity_threshold',
                        'minimum_number_of_sources_to_create_branch_for_node'}:
        raise ValueError('all HKT construction fields required')
    minimum = builder['minimum_number_of_sources_to_create_branch_for_node']
    if (builder['minimum_threshold_against_max_word_count'] != 0.7 or
            builder['similarity_threshold'] != 0.5 or type(minimum) is not int or minimum < 1):
        raise ValueError('invalid construction settings')
    gamma = engine['max_batches_in_memory']
    if engine['memory_mode'] == 'Full':
        if gamma is not None:
            raise ValueError('Full mode requires null gamma')
    elif engine['memory_mode'] == 'SlidingWindow':
        if type(gamma) is not int or gamma < 1:
            raise ValueError('SlidingWindow requires explicit positive integer gamma')
    else:
        raise ValueError('unknown memory mode')
    sources = config['parameter_sources']
    for key in ('branch_minimum', 'gamma', 'clustering_settings', 'evaluation_population'):
        if not isinstance(sources.get(key), str) or not sources[key].strip():
            raise ValueError(f'record evidence or explicit unresolved research choice for {key}')
    if not isinstance(config['clustering_settings'], dict):
        raise ValueError('explicit clustering settings object required')
    return config


def load_manifest(path, smoke=False):
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get('schema_version') != 1 or manifest.get('theme') not in ('G', 'H'):
        raise ValueError('manifest requires schema_version=1 and theme G or H')
    batches = manifest['batches']
    if not batches or (not smoke and len(batches) != 12):
        raise ValueError('benchmark requires 12 ordered batches (smoke may use fewer)')
    base = manifest.get('path_base')
    if base == 'selected_benchmark_data_directory':
        root = path.parent.parent
    elif base == 'manifest_directory':
        root = path.parent
    else:
        raise ValueError('unknown manifest path_base')
    hashes = {}
    documents = []
    seen = set()
    for index, entry in enumerate(batches):
        batch_path = (root / entry['path']).resolve()
        if not batch_path.is_relative_to(root):
            raise ValueError('batch path escapes dataset directory')
        if entry['batch_index'] != index or digest(batch_path) != entry['sha256']:
            raise ValueError('batch sequence or SHA256 mismatch')
        batch = read_json(batch_path)
        if batch['batch_index'] != index:
            raise ValueError('batch content index mismatch')
        ids = unique_ids([s['source_id'] for s in batch['sources']], 'batch sources')
        if ids & seen:
            raise ValueError('source IDs must be globally unique across batches')
        for source in batch['sources']:
            if source['batch_index'] != index:
                raise ValueError('source batch index mismatch')
            # timeline-many scopes synthetic CSV row IDs; reject to preserve identity.
            source_id = source['source_id']
            if source_id.startswith('row_') and source_id[4:].isdigit():
                raise ValueError('namespace synthetic row IDs before engine invocation')
            if not isinstance(source['tokens'], list) or any(not isinstance(t, str) for t in source['tokens']):
                raise ValueError('tokens must be a string list')
            if set(source) - {'source_id', 'batch_index', 'tokens', 'text', 'timestamp_unix_ms', 'metadata'}:
                raise ValueError('unexpected source fields; ground truth must remain separate')
            if source.get('metadata') is not None:
                raise ValueError('benchmark engine inputs must omit metadata (label leakage boundary)')
        seen |= ids
        documents.append(batch['sources'])
        hashes[entry['path']] = entry['sha256']
    truth_path = (root / manifest['ground_truth_path']).resolve()
    if not truth_path.is_relative_to(root):
        raise ValueError('truth path escapes dataset directory')
    if digest(truth_path) != manifest['ground_truth_sha256']:
        raise ValueError('ground truth SHA256 mismatch')
    truth = read_json(truth_path)
    truth_ids = unique_ids([r['source_id'] for r in truth], 'ground truth')
    if truth_ids != seen:
        raise ValueError('ground truth must match dataset population exactly')
    hashes[manifest['ground_truth_path']] = manifest['ground_truth_sha256']
    return manifest, documents, truth, hashes


def population(documents, index, config):
    boundary = config['evaluation_population']
    start = index if boundary == 'current_batch' else 0
    if boundary == 'retained_window' and config['engine']['memory_mode'] == 'SlidingWindow':
        start = max(0, index + 1 - config['engine']['max_batches_in_memory'])
    return [source for batch in documents[start:index + 1] for source in batch]
