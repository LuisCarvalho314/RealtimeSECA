# SECALight structural simplification

Base: latest GitHub `main` verified on 2026-10-22,
`0992ee3c9b8ab86b1e028f94fa5bc0403fec22cd` (approved audit and behavioural oracle).
Branch: `refactor/secalight-architecture`; worktree:
`/home/azureuser/secalight-refactor`.

Shell fetch failed because DNS was unavailable. The connected GitHub API supplied
the signed merge payload; reconstructing the Git object reproduced its exact SHA.
Its tree is `d7767520424626a4ec676e2f04126eb047613615`, also the tree of the local
prerequisite integration commit `00e97b7`. Baseline tests and the performance
binary were built from that identical integration tree. No remote refs changed.

## Audit assessment and decisions

The [architectural audit](seca-simplification-audit.md) correctly identified
unnecessary construction and planning ownership. Inspection confirmed that
production ancestry only ever adds attachment-node descriptors with zero count
payloads to two maps, while its rejected map remains empty. One exclusion set
therefore preserves reachable behaviour. The synthetic private rejected-count test
was removed with that unreachable path; genuine exclusion assertions were adapted,
and new tests exercise accumulated exclusions through three actual tree levels.

The work implements F1, F4a/F4b, the batch-summary and pending-merge portions of
F3, arithmetic replacement IDs from F7, small identity/export/pruning cleanups
from F8, and the shared timeline/owned chunking/conditional snapshot portions of
F10. It does not claim completion of every audit recommendation.

The retained node lookup and token payloads have real consumers. The runtime
mirror remains a schema-4 consistency invariant. The inverse identity map supports
collision-safe admission; retained source order influences reconstruction ties.
The outer engine clone protects failures after earlier candidate mutations. These
were preserved rather than replaced with new rollback or persistence machinery.
Public synthetic-token fallbacks and existing restore acceptance rules remain.
The audit's E02/E03 proposal conflict is still a scientific review question.

## Measured complexity

Physical Rust source lines, including blank lines and comments, compared directly
with `0992ee3`. Core excludes files below `src/**/tests/`; CLI counts include its
unchanged embedded test modules, matching the audit's CLI convention. No test,
document, fixture, or example deletion contributes to these reductions.

| Area | Before | After | Net removed |
| --- | ---: | ---: | ---: |
| Core production `src` | 3,494 | 3,253 | 241 (6.90%) |
| CLI `src`, including embedded tests | 1,106 | 1,100 | 6 (0.54%) |
| Combined | 4,600 | 4,353 | 247 (5.37%) |
| Builder alone (subset of core) | 530 | 436 | 94 (17.74%) |

The latest base's core count differs from the audit's earlier 3,484-line base.
Counts are physical lines, not executable statements or a cyclomatic metric.

Removed representations and synchronization obligations:

- Build-wide `hkt_input_scopes` archive and unused recursion-level argument.
- Mutable builder `nodes_by_id` and collision/top-word/first-node/refuge mirror
  updates. Finished local HKTs now generate the public mirror once at return.
- Positional source-word BTreeMaps, conversion helper, and collected deletion keys.
  Frequency-ordered vectors retain row order through deletion and recursion.
- Root/child HKT clones used solely to recurse after insertion. Explicit counters
  preserve allocation order while completed HKTs are inserted after recursion.
- Three ancestry maps and their count tuples, replaced by one descriptor set.
- Trigger boolean, target vector and four parallel scope/ancestry maps, replaced
  by one ordered map of target records. Reports derive IDs and flags from it.
- `ComputedBatchWordStats` and its module. Input preparation returns the existing
  public summary plus frequencies and uses their ordered keys to admit vocabulary.
- Two remapping tables; replacement adds offsets to owned hierarchy IDs and derives
  its public node mirror from those remapped nodes.
- Pending-membership cloning into an intermediate vector; maps move with
  `mem::take`, and source sets merge with `append`.
- A repeated prior-membership clone, redundant legend writes/removals and export
  sorts, gamma conversion, one duplicated timeline execution/export/manifest loop,
  chunk payload clones and unconditional baseline snapshot serialization.

No dependency, configurable policy, alternative update path, or public type was
introduced. `ReconstructionTarget` is a private concrete record holding provenance
and exclusions together; it replaces the parallel plan fields.

## Preserved behaviour and added evidence

The canonical lifecycle remains preparation → top-down mapping/admission/errors →
accepted-parent mutation and child inspection → selected reconstruction → forgetting
→ final diagnostics → transaction commit. No metric formulas, thresholds, admission
operators, DF tie rules, branch-minimum strictness, or descriptor persistence change.
Construction still recurses over original inputs, not the residual rows proposed
by E05. Caller-supplied root DF and public `SourceWordRecord` fields remain intact.

New private regressions cover accumulated attachment exclusions over three levels,
two disjoint triggered children with ascending reported target IDs, JSON restore
continuation, and rollback after accepted-parent mutation and input registration.
The failure test deliberately invalidates the private builder configuration after
valid construction: reconstruction then fails at builder validation, and the whole
serialized engine must remain identical. This exercises a later failure boundary
without adding production fault injection; it is not allocation-failure coverage.

The locked behavioural oracle, capture program, all archived schema-4 snapshots,
all parity fixtures and the algorithm specification are unchanged. Its strict
public, scientific-structure and archived-continuation checks pass after each
significant core slice. Private stage tests follow the smaller representation;
assertions about real source membership/exclusion and continuation remain.

## Validation

All final checks passed on Rust 1.92.0 / Linux x86_64 with the unchanged lockfile:

- `cargo fmt --all -- --check`.
- `cargo test --locked --workspace`: 191 passed, zero failed, two existing strict
  C# parity tests ignored. The identical baseline tree passed 190 with the same
  two skips; the net increase is two new tests minus one impossible private test.
- `cargo clippy --locked --workspace --all-targets -- -D warnings`.
- `cargo build --locked --workspace --examples --bins`.
- `cargo test --locked -p realtime-seca-core --test behaviour_oracle`: all three
  independent checks passed, including archived schema-4 continuation.
- Python standard-library unittest discovery: evaluation 25, clustering 21,
  pinned datasets 13, reproduction contracts 5; all 64 passed, none skipped.
  Evaluation used `SECA_TEST_ENGINE` pointing to the freshly built CLI, including
  Full/SlidingWindow integration. Dataset tests used the local pinned files at
  `/tmp/seca-mstream-upstream/MStream/data`; they verified their own hashes.
- `git diff --check`; locked oracle, capture and fixture paths have no diff from
  the base.

Focused unit, parity, stream, invariant and/or snapshot suites plus the locked
oracle were run between significant implementation slices. The complete original
large-data suite passed both before refactoring and on the final implementation.
The external C# executable was not rerun; committed reference-characterization
fixtures were exercised with their documented intentional differences.

Python reproduction commands, from the repository root unless noted:

```sh
# From benchmarks/seca-paper:
SECA_TEST_ENGINE=/absolute/path/to/target/debug/realtime-seca-cli \
  python -m unittest discover -s evaluation -p 'test_*.py'
python -m unittest discover -s clustering -p 'test_*.py'
# From the repository root:
SECA_DATASET_SOURCE_DIR=/path/to/pinned/MStream/data \
  python -m unittest discover -s benchmarks/seca-paper/datasets -p 'test_*.py'
python -m unittest discover -s benchmarks/reproduction -p 'test_*.py'
```

## Performance comparison

The unchanged historical public-API `benchmark_stream.rs` from
`315eb3a347d953e938dcc123a1fd27530d7fa022` was temporarily restored in the
baseline and refactored worktrees, built in release mode with the locked
workspace, and removed afterward. See [the original harness contract](performance.md).
The compared implementation is `fdbe308a881a24a216a2f2ebefe992b7aa9368dd`;
subsequent changes only add documentation and measurement evidence.

Three fresh-process runs per version/configuration alternate version order.
Each constructs a baseline then processes 30 batches of 100 sources, vocabulary
256, seed 42 and branch minimum 32. Stable uses overlap .95, no novel anchors and
thresholds 1; changing uses overlap .5, novel anchors every two batches and
thresholds 0. All three thresholds use the indicated value. These are supported
parameters exercising accepted updates and repeated reconstruction, not production
default thresholds or default branching. Input generation and measurement probes
are outside `process_batch` timing; diagnostic work inside it is included.

| Stream | Median update before → after (ms) | Change | p95 before → after (ms) |
| --- | ---: | ---: | ---: |
| Full, stable | 30.851 → 30.408 | −1.4% | 54.368 → 53.969 |
| Full, changing | 97.947 → 89.910 | −8.2% | 196.483 → 169.713 |
| gamma=3, stable | 10.684 → 10.352 | −3.1% | 11.219 → 11.153 |
| gamma=3, changing | 43.856 → 39.829 | −9.2% | 73.079 → 69.237 |

Table cells are medians across three per-run statistics. Per-run update medians
average the two middle values; p95 is nearest-rank within each run. Baseline construction fell from 2.745–2.778 ms to 1.969–2.025 ms
across the four configuration medians (27–29%). Stable runs reconstructed zero
updates; changing runs reconstructed all 30. Snapshot byte lengths, vocabulary,
active source counts, topology counts, reconstruction flags and target counts
matched exactly at every batch in all 12 paired runs. This comparison is broader
than checking performance alone, but these fields are not a full state oracle.

Median process HWM before → after was 24,336 → 23,572 KiB (Full stable),
29,412 → 29,516 (Full changing), 5,248 → 5,360 (window stable), and
8,456 → 8,244 (window changing). These mixed small differences do not establish a
memory improvement. HWM includes allocator retention and earlier measurement
probes, not just live engine storage. No allocation-count instrumentation was added.

Linux x86_64, Rust 1.92.0, standard release optimization, shared host, no CPU
pinning or exclusive CPU access; workspace tests overlapped measurements. Small
stable-stream differences may be noise. The reconstruction-heavy improvement is
observed on these workloads, not a latency guarantee or a scientific-quality claim.
No arbitrary-depth/wide-tree or default-branching performance claim is made.

[Raw per-batch CSV](secalight-simplification-performance.csv) preserves all 720
observations and timings; [machine-readable summary](secalight-simplification-performance.json)
includes trial values, provenance and SHA-256 hashes of compared observation fields.
To reproduce, temporarily restore the historical example in both isolated
checkouts, build with `cargo build --locked --release -p realtime-seca-core
--example benchmark_stream`, and execute three times per configuration:

```sh
./target/release/examples/benchmark_stream --batches 30 --sources 100 \
  --vocabulary 256 --seed 42 --branch-min 32 --gamma 0 \
  --overlap 0.95 --novel-every 0 --threshold 1
```

Use gamma 3 for SlidingWindow and the changing parameters described above.
Compare the same CSV fields and alternate baseline/refactored execution order.

## Remaining opportunities and limits

Runtime node payload duplication and per-scope cloned parallel maps still exist.
A private single-owner hierarchy needs an explicit schema-4 adapter that preserves
all accepted shapes and rejects stale/extra mirrors; moving that risk into this
patch would not simplify its review. Borrowed scope access and operation-local
adjacency indexes remain useful next steps. Validation must retain multiple HKTs
sharing a parent node where old snapshots currently permit them.

Context and metrics still repeat parent-scope DF computation. Their other old-word
counts use different evidence and cannot be unified casually. Vocabulary reversal
and normalization repeat across scope operations. Construction still makes one
mutable row copy per scope and recursive child dataset copies; the build-wide
archive and duplicated HKT payloads are gone.

Final diagnostic comparison still clones the engine and traverses the final tree.
Arbitrary public diagnostic batches and accepted snapshots can reach synthetic
word fallbacks; deleting them or strengthening load validation would change a
boundary contract. The CLI still computes separate verbose exports for simultaneous
printing and dumping. Its persistence/fsync path was left intact.

The frozen oracle covers one ordered synthetic stream in two memory modes, not
the whole input space or scientific correctness. Two existing strict C# parity
assertions remain deliberately ignored. E01–E03 and E05–E07 remain unapproved;
their proposals are unchanged and none is implemented here. No new scientific or
snapshot-compatibility proposal was needed for these changes.
