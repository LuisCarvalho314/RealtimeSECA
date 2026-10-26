"""Compare immutable baseline construction against locked observations; never edit them."""
import json
from pathlib import Path
from seca import run
from test_seca import ROOT, structural


def evaluate():
    outcomes = []
    for group in ('parity_hkt_reconstruction', 'scientific_hkt'):
        for path in sorted((ROOT / 'crates/realtime-seca-core/tests/fixtures' / group).glob('*/input.json')):
            result = run(json.loads(path.read_text()))
            row = {'fixture': f'{group}/{path.parent.name}'}
            for label, filename in [('rust', 'rust.observed.json'), ('csharp', 'csharp.normalized.json')]:
                observed = path.parent / filename
                if observed.exists():
                    row[label] = result == structural(json.loads(observed.read_text())[0]['topology'])
            outcomes.append(row)
    return {'scope': 'baseline ordered structural projection only; excludes expectedWords, IDs, top_words, all updates',
            'results': outcomes,
            'totals': {label: {'pass': sum(row.get(label) is True for row in outcomes),
                               'fail': sum(row.get(label) is False for row in outcomes)}
                       for label in ('rust', 'csharp')}}


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))
