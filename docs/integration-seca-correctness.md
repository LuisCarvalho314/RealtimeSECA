# SECA correctness integration

2026-10-13. Integration base: `01a644f` (`origin/main`). Separate branch:
`integration/seca-correctness`. One integration commit combines completed work:

| Agent branch | Reviewed commit | Integrated scope |
| --- | --- | --- |
| `fix/seca-ingestion` | `643bf1c` | Article identities, explicit-offset timestamps, CSV/restart regressions and ingestion contract |
| `fix/seca-snapshot-validation` | `d00ec33` | Vocabulary references, retained-batch order/window validation, corruption/continuation regressions and state contract |
| `research/seca-context-semantics` | `6102267` | Investigation report, synthetic captures and characterization tests |
| `research/seca-hkt-construction` | `c6cc0ac` | Investigation report, synthetic captures and characterization tests |

No live agents were registered when integration began. All four named branches
had their completed commits. Engineering worktrees were clean. The shared
research checkout contained unrelated uncommitted work and local reference
artifacts; integration used a clean isolated worktree and only the named commits.

CSV ingestion now gives trimmed URLs stable identities and anonymous rows unique
conversion-scoped identities. Explicit-offset calendar timestamps are parsed;
ambiguous local timestamps remain unset. Snapshots reject unknown vocabulary
references, expired SlidingWindow evidence, future batches and unordered retained
batches, while valid Full-memory historical evidence remains accepted.

Integration additionally replaces the deprecated stop-word enum alias with the
equivalent `"en"` language code, supported by both cached 0.10.0 and freshly
resolved 0.10.1. This addresses a warnings-denied build failure without changing
the selected English stop-word list. No metric algorithm, context update, HKT
builder, reconstruction path or threshold was changed.

## Scientific review

Both reports distinguish executed observations from hand-derived proposed
outputs. Their scientific tests faithfully preserve current behavior, including
known discrepancies. Shared-candidate C# duplicate-key failures provide no
successful decision oracle; the reports do not invent one. HKT tie choices and
branch equality are explicit recommendations where papers are underspecified.
The contextual worst-node beta reduction and descriptor persistence need review
as proposed semantics, not reference bug fixes.

[Scientific decisions pending](scientific-decisions-pending.md) records E01/E02/
E03/E05/E06/E07 recommendations, rationale, confidence and minimal future Rust
changes. E04 retains the full node-union denominator. Author clarification is
still desirable for shared-word reduction, expectations, recursive scope and the
branch counted set. No proposed scientific change is approved by this integration.

## Verification

Independent checks in `/tmp/seca-correctness-integration`:

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | Passed |
| `cargo test --workspace` | 163 passed, 0 failed, 2 intentionally ignored; completed including all large-dataset tests |
| `cargo clippy --workspace --all-targets -- -D warnings` | Passed after the equivalent stop-word code adjustment |
| CLI crate rerun after final ingestion adjustment | 3 unit + 4 CSV workflow tests passed, including restart/replay/deduplication persistence regression |
| Focused executable Rust parity suites | 20 context + 12 HKT passed, 2 intentionally ignored |
| Focused scientific/snapshot suites | 4 context + 4 HKT + 12 snapshot validation passed |
| Existing external C# harness verification | Passed: hand-calculated metrics/topology, continuation, repeatability and invalid-input checks |
| Existing executable C# fixture replays | 19 context inputs and 21 HKT inputs verified, each twice against existing captures; known reference failures checked as failures |
| New scientific C# fixture replays | 5 context + 22 HKT inputs verified twice against raw captures without changing evidence |
| Supplemental CLI process persistence checks | 6 passed: byte-identical restart replay; duplicate batch index, malformed input, configuration mismatch, corrupt snapshot and tree-write failure all rejected without altering persistent outputs or leaving temporary state |
| `git diff --check` and publication text/path audit | Passed |

The workspace run began before the stop-word alias adjustment. Final CLI tests,
formatting and Clippy were rerun after it; core production code did not change
during the run. Large-dataset integration took 235.85s; batch processing took
70.92s. Dependency resolution selected stop-words 0.10.1, chrono 0.4.45 and uuid
1.28.0. The repository ignores its lockfile, so these versions describe this run,
not a newly pinned dependency contract.

Initial failures: sandbox registry DNS access prevented the first workspace
invocation; it was rerun with approved access. Initial Clippy failed on two uses
of the deprecated stop-word alias; the equivalent language-code change resolved
both. There are no outstanding test/build failures. Neither strict ignored
assertion was unignored or rewritten, nor were discrepancy goldens forced into
agreement. They retain the precision and HKT topology differences documented by
the prior executable audit.

Reproduction commands for committed Rust checks:

```sh
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p realtime-seca-cli
cargo test -p realtime-seca-core --test parity_context_metrics --test parity_hkt_reconstruction
cargo test -p realtime-seca-core --test snapshot_validation --test scientific_context_contract --test scientific_hkt_contract
```

External C# verification used the existing local harness without copying it into
integration. Existing parity replays use `SECA_REFERENCE_ROOT` as documented in
the executable summary. The scientific captures were verified with a temporary
read-only runner, because the HKT research replay script recaptures files; those
goldens and provenance were left intact. Supplemental CLI checks used temporary
synthetic batches and files outside the repository. Changed production files
remain below 400 lines; the existing CSV conversion function was reviewed and
kept cohesive without unrelated restructuring.

## Publication contents

Production changes are confined to CLI `Cargo.toml`, `src/ingestion.rs` and core
`src/engine/validation.rs`. Added tests are CLI `tests/csv_workflow.rs` and core
`tests/snapshot_validation.rs`, `tests/scientific_context_contract.rs`, and
`tests/scientific_hkt_contract.rs`. Scientific evidence is confined to
`tests/fixtures/scientific_context/` and `tests/fixtures/scientific_hkt/` in the
core crate. Documentation adds the ingestion contract, both research reports,
pending decisions and this integration report, and updates `docs/state.md`.
The complete file inventory follows below.

The integration diff contains UTF-8 source/evidence text only. New scientific
inputs use synthetic tokens a/b/c/d/q/x/z. No old C# source, binaries, reference
tree, harness, build products or restricted dataset is added. The pre-existing
`tests/data/large_batch.json` news fixture is unchanged; this check does not
establish its licensing. Local reference sources/binaries were used externally
for replay and are not redistributed. Ignored `Cargo.lock` and `target/` remain
local. Existing strict parity test files are unchanged, including both intentional
`#[ignore]` annotations. Main is not committed or pushed by this integration.

## Complete changed-file inventory

145 files relative to integration base:

- `crates/realtime-seca-cli/Cargo.toml`
- `crates/realtime-seca-cli/src/ingestion.rs`
- `crates/realtime-seca-cli/tests/csv_workflow.rs`
- `crates/realtime-seca-core/src/engine/validation.rs`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/csharp-source-hashes.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/expected_history.csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/expected_history.input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/expected_history.rust.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/expected_trigger.csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/expected_trigger.input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/expected_trigger.rust.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/historical_beta.csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/historical_beta.input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/historical_beta.rust.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/historical_beta_trigger.csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/historical_beta_trigger.input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/historical_beta_trigger.rust.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/provenance.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/replay.py`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/shared_asymmetric.csharp.stderr.txt`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/shared_asymmetric.input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/shared_asymmetric.rust.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_context/shared_sets.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_2_baseline/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_2_baseline/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_2_baseline/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_2_baseline/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_2_baseline/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_2_rebuilt/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_2_rebuilt/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_2_rebuilt/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_2_rebuilt/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_2_rebuilt/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_3_baseline/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_3_baseline/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_3_baseline/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_3_baseline/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_3_baseline/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_3_rebuilt/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_3_rebuilt/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_3_rebuilt/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_3_rebuilt/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_3_rebuilt/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_4_baseline/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_4_baseline/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_4_baseline/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_4_baseline/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_4_baseline/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_4_rebuilt/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_4_rebuilt/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_4_rebuilt/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_4_rebuilt/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/boundary_4_rebuilt/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/branch_boundary/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/branch_boundary/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/branch_boundary/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/branch_boundary/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/branch_boundary/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/branch_boundary_reversed/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/branch_boundary_reversed/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/branch_boundary_reversed/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/branch_boundary_reversed/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/branch_boundary_reversed/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie_rebuilt/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie_rebuilt/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie_rebuilt/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie_rebuilt/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie_rebuilt/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie_rebuilt_reversed/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie_rebuilt_reversed/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie_rebuilt_reversed/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie_rebuilt_reversed/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie_rebuilt_reversed/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie_reversed/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie_reversed/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie_reversed/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie_reversed/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie_reversed/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary_rebuilt/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary_rebuilt/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary_rebuilt/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary_rebuilt/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary_rebuilt/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary_rebuilt_reversed/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary_rebuilt_reversed/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary_rebuilt_reversed/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary_rebuilt_reversed/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary_rebuilt_reversed/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary_reversed/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary_reversed/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary_reversed/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary_reversed/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary_reversed/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_child_rebuilt/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_child_rebuilt/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_child_rebuilt/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_child_rebuilt/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_child_rebuilt/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_child_rebuilt_reversed/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_child_rebuilt_reversed/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_child_rebuilt_reversed/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_child_rebuilt_reversed/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_child_rebuilt_reversed/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive_rebuilt/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive_rebuilt/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive_rebuilt/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive_rebuilt/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive_rebuilt/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive_rebuilt_reversed/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive_rebuilt_reversed/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive_rebuilt_reversed/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive_rebuilt_reversed/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive_rebuilt_reversed/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive_reversed/comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive_reversed/csharp.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive_reversed/csharp.normalized.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive_reversed/input.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive_reversed/rust.observed.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/permutation-comparison.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/provenance.json`
- `crates/realtime-seca-core/tests/fixtures/scientific_hkt/replay.py`
- `crates/realtime-seca-core/tests/scientific_context_contract.rs`
- `crates/realtime-seca-core/tests/scientific_hkt_contract.rs`
- `crates/realtime-seca-core/tests/snapshot_validation.rs`
- `docs/cli-ingestion.md`
- `docs/integration-seca-correctness.md`
- `docs/research-context-semantics.md`
- `docs/research-hkt-construction.md`
- `docs/scientific-decisions-pending.md`
- `docs/state.md`
