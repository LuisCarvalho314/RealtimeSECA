"""Generate a tiny orchestration fixture, never a replacement G/H dataset."""
from pathlib import Path
from .contracts import PAPER_THRESHOLDS
from .io import write_json, digest


def make_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    truth = []
    batches = []
    for index in range(2):
        sources = [{'source_id': f'fixture-{index}-{j}', 'batch_index': index,
                    'tokens': [word, 'shared']} for j, word in enumerate(('alpha', 'beta'))]
        filename = f'batch-{index:04}.json'
        write_json(directory / filename, {'batch_index': index, 'sources': sources})
        batches.append({'batch_index': index, 'path': filename, 'sha256': digest(directory / filename)})
        truth.extend({'source_id': s['source_id'], 'label': s['tokens'][0]} for s in sources)
        write_json(directory / f'assignments-{index:04}.json', {
            'batch_index': index, 'evaluation_source_ids': [s['source_id'] for s in sources],
            'clusters': [{'cluster_id': f'mock-{j}', 'source_ids': [s['source_id']]} for j, s in enumerate(sources)],
            'unassigned_source_ids': []})
    write_json(directory / 'truth.json', truth)
    write_json(directory / 'manifest.json', {'schema_version': 1, 'theme': 'G', 'batches': batches,
        'path_base': 'manifest_directory', 'ground_truth_path': 'truth.json',
        'ground_truth_sha256': digest(directory / 'truth.json')})
    for mode in ('Full', 'SlidingWindow'):
        write_json(directory / f'config-{mode}.json', {
            'purpose': 'mock-smoke', 'evaluation_population': 'current_batch', 'clustering_settings': {},
            'parameter_sources': {key: 'Synthetic smoke choice; publication value unresolved' for key in
                                  ('branch_minimum', 'gamma', 'clustering_settings', 'evaluation_population')},
            'engine': {'hkt_builder': {'minimum_threshold_against_max_word_count': .7,
                                      'similarity_threshold': .5,
                                      'minimum_number_of_sources_to_create_branch_for_node': 10},
                       'seca_thresholds': PAPER_THRESHOLDS, 'memory_mode': mode,
                       'max_batches_in_memory': 1 if mode == 'SlidingWindow' else None}})


if __name__ == '__main__':
    make_fixture(Path(__file__).parent / 'fixtures')
