"""Evaluate the direct reducer on actual C# observations, not invented metrics.

This tests only error aggregation. It does NOT test mapping/admission or derive
new memberships, and cannot establish either prototype's update parity.
"""
import json
from evaluate import ROOT, direct


def run():
    results = []
    for path in sorted((ROOT / 'crates/realtime-seca-core/tests/fixtures/parity_context_metrics').glob('*.csharp.json')):
        captured = json.loads(path.read_text())
        for snapshot in captured['snapshots'][1:]:
            for decision in snapshot.get('decisions', []):
                old = sorted({w for n in decision['topologyAtDecision']['nodes'] for w in n['words'] if w != -1})
                metrics = {w['wordId']: (
                    w['values']['numberOfSourcesAfterNewBatchOverNumberOfSourcesOfNewProminWordInHKT'],
                    w['values']['numberOfIntersectedSourcesWithNewProminWordInNodeOverNumSourcesOfNewProminWordInNode'],
                    w['values']['precentageOfSourcesInHKTInNewBatch']) for w in decision['wordMetrics']}
                errors = direct.errors(old, metrics, captured['parameters']['alpha'], captured['parameters']['beta'])
                expected = {key:decision['errors'][name]['option1_error'] for key,name in
                            [('alpha','alphaError'),('beta','betaError'),('importance','wordImportanceError')]}
                results.append({'fixture':path.stem,'cycle':snapshot['cycle'], 'hkt':decision['hktId'],
                    'matches_rounding_envelope':all(abs(errors[k]-expected[k]) <= .000050000001 for k in errors)})
    return {'scope':'direct reducer over captured C# measurements only; no update parity', 'checks':results}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
