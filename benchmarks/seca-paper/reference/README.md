# C# reference benchmark

This tooling executes the adapted original C# algorithm classes under Mono, not
an uninstrumented Windows application. It adds per-cycle output checkpointing,
without changing reference scientific calculations or the Rust engine.

Read [the report](../../../docs/seca-reference-benchmark.md) before interpreting
scores. `local/` contains restricted binaries and token-bearing inputs/trees and
is ignored. `evidence/` contains ID-only outputs, descriptor hashes and reports.

Run from the repository root, substituting your existing reference installation
and prepared benchmark paths. All output directories must be fresh.

```bash
REF=/home/azureuser/risklive-dev/experimental
PREPARED=/tmp/seca-paper-benchmark/benchmarks/seca-paper/datasets/data/prepared
RUST_RESULTS=/tmp/seca-paper-benchmark/benchmarks/seca-paper/results/final
BASE="$PWD/benchmarks/seca-paper/reference/local"

python3 benchmarks/seca-paper/reference/build.py \
  --reference-root "$REF" --output "$BASE/build-new"
SECA_CHECKPOINT_BUILD="$BASE/build-new" \
  python3 -m unittest discover -s benchmarks/seca-paper/reference -v
python3 benchmarks/seca-paper/reference/reference_execute.py \
  --prepared "$PREPARED" --build "$BASE/build-new" --output "$BASE/runs-new"
python3 benchmarks/seca-paper/reference/compare_engines.py \
  --prepared "$PREPARED" --runs "$BASE/runs-new" \
  --rust-local "$RUST_RESULTS" \
  --evidence "$BASE/evidence-new" --local "$BASE/normalized-new"
python3 benchmarks/seca-paper/reference/package_evidence.py \
  --build "$BASE/build-new" --runs "$BASE/runs-new" \
  --normalized "$BASE/normalized-new" --evidence "$BASE/evidence-new" \
  --prepared "$PREPARED"
python3 benchmarks/seca-paper/reference/analyze_boundaries.py \
  --runs "$BASE/runs-new" --rust "$RUST_RESULTS" \
  --output "$BASE/evidence-new/boundary-analysis.json"
python3 benchmarks/seca-paper/reference/tables.py "$BASE/evidence-new"
```

`build.py` first rebuilds and verifies the existing harness. The local reference
root needs the original archive, compilable copy, five existing dependencies and
`tools/seca-reference-harness/`; nothing is downloaded or redistributed. The
compiled algorithm classes use the existing `prepare.py` and `Execution.cs`.
`Checkpoint.cs` is an adapted copy of the authored parity wrapper only.

`reference_execute.py` records process exit status, input hash, reference source
and binary hashes, every completed cycle, and full stderr. A surfaced exception
or explicit resource timeout ends that stream; no continuation from partial
mutation occurs. The default timeout is 1,800 seconds per stage and is recorded
as a resource failure if reached. It is not a scientific termination criterion.

Prepared batch hashes and truth hashes must match their manifests. Conversion
keeps source/document/token order, assigns integer IDs by first ordered appearance,
and emits distinct source-word relations in first-token order. Original repeated
tokens remain in the prepared dataset and their ordered hashes in the mapping.
No extra cleaning occurs and no labels enter either engine or clustering adapter.

`compare_engines.py` reads immutable existing Rust verbose exports. It verifies
all tree and decision-report hashes against the existing benchmark evidence,
then re-evaluates both trees with the exact existing `cluster.evaluate` function
from `benchmarks/seca-paper/clustering/cluster.py`. The metric implementation is
the existing `evaluation.metrics.evaluate`. Rust assignments must equal their
preserved gzip evidence exactly. Recreate missing Rust exports with the existing
benchmark reproduction commands in `docs/seca-paper-benchmark.md`.

Raw C# snapshots retain actual memberships, pending maps, metrics, pre-rebuild
observations and completed topology. Normalized exports use only actual source
memberships. They do not incorporate pending sources or infer missing metrics.
Generated node/HKT IDs remain in exports for attachments; comparisons discard
these IDs and sort semantic sibling subtrees. Canonical comparisons exclude
expected-set persistence differences, which are retained in raw evidence.

`evidence/*-nodes-*.json.gz` preserves node attachments, ordinary/refuge identity,
actual external source memberships and descriptor fingerprints. Exact descriptor
strings, normalized trees and unmatched-node details stay in `local/`; their
hashes are inventoried. This follows the existing dataset policy of keeping
preprocessed corpora and token-bearing exports outside Git. Direct/indirect
assignments that the shared adapter can generate are retained without altering
overlap. Invalid reference scope produces no invented assignment.

The ordinary-node co-membership diagnostic compares current-batch pairs, excluding
refuge nodes. Event co-membership diagnostics use shared-adapter assignments and
ignore event IDs. Per-document membership-family comparisons retain multiple
member sets. Reconstruction scopes have ancestor descriptor fingerprints and
prior scoped-source fingerprints; different tree scopes are not forcibly paired.
Published per-cycle scores are unavailable: per-cycle tables show the published
12-cycle mean as context, not a recovered per-cycle target. Partial paired means
are explicitly separated from full-stream means.
