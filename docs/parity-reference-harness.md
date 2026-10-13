# Executable C# SECA reference harness

Run from the repository root (Mono `mcs`/`mono`, Python 3 and Bash required):

```sh
tools/seca-reference-harness/build.sh
tools/seca-reference-harness/run.sh tools/seca-reference-harness/fixtures/stable.json > /tmp/stable.json
tools/seca-reference-harness/run.sh tools/seca-reference-harness/fixtures/reconstruct.json > /tmp/reconstruct.json
python3 tools/seca-reference-harness/verify.py
```

The existing compilable reference must be present, with its dependencies in
`references/seca-parity-build/SECA/bin/Debug/`: `SourceCleanerEngine.exe`,
`Newtonsoft.Json.dll`, `Accord.dll`, `Accord.Math.dll`, `Accord.Math.Core.dll`.
No downloads or package restore are performed. Compiler warnings inherited from
the reference (including unreachable swallowed-exception bodies) are expected.
Generated source stays in the separate build copy's ignored `.harness-generated/`;
the executable stays in this directory's ignored `.build/`. Do not commit or
redistribute generated reference source, binaries, or the original reference.
Source SHA-256 hashes are recorded in `.harness-generated/source-hashes.json`.

## Input

JSON contains an ordered `batches` array. The first batch builds the baseline;
each subsequent batch runs a contextual update. Each batch contains ordered
objects `{"id":1,"words":["a","b"]}`. IDs must be positive integers unique
across the entire run. Batches and word arrays must be nonempty. Tokens are
already preprocessed, compared exactly (case sensitive), and must be unique
within a document. No stemming, normalization, tokenization or ML runs here.
Word IDs are allocated by first appearance across source and token order.
Source-word rows are stably sorted by descending batch document frequency;
record order resolves ties. There is one row per source-word pair.

Optional top-level settings and defaults:

| Setting | Default |
| --- | --- |
| alpha | 0.7 |
| beta | 0.5 |
| branchMinimum | 1 |
| alphaError | 0.1 |
| betaError | 0.13 |
| wordImportanceError | 0.3 |

All three selected error options are Option1. The original code also calculates
other options; their raw outputs and effective thresholds are included. This
minimal harness executes accumulated SECA (`isLightSECA=false`), without
forgetting. It does not claim to validate SECA-Light's retention boundary.

## Entry points and cycle lifecycle

`SECAAlgorithm.buildTree(Dictionary<int, SourceWord>, int)` invokes the original
HKT construction and branching. `updateTree(isDeploy:true, ...)` performs the
original top-down contextual mapping, candidate admission, error evaluation,
and selective reconstruction. `clearMemory(..., isLightSECA:false)` removes
marked dead HKTs, and `copyDataDictionary` appends incoming rows to global
history after the update. Rebuilt HKTs' `isState1` flag is reset before the next
cycle, so they participate as state 0. There is no database reload between cycles.

`HKT.computeBatchErrorsForCurrentHKTLevel` owns metric calculation;
`HKT.isAccepted` makes the trigger decision. `Node` exposes word/source maps,
pending memberships and per-node expected words. Construction's expected-word
set is a local variable, not persistent diagnostic state: baseline exports show
only the exposed dynamic expected-word map, which may be empty. Baseline word
metrics are unavailable and are not fabricated.

## JSON schema version 1

The output has `schemaVersion`, execution `mode`, `parameters` (including the
input batches), `vocabulary`, and ordered `snapshots`.

Each snapshot has `cycle` and `tree`. Updates also have `decisions` and
`removedHktIds`. Each HKT in `tree` exposes `id`, `parentNodeId` (0 means root),
`expectedWords`, and ordered `nodes`. Each node exposes `id`, `words` (-1 means
refuge), `sources`, `wordSources`, `pendingSources`, `pendingWordSources`, and
`expectedWords`. These are raw state, including pending maps left by the reference.
Parent attachments and final nodes encode reconstructed subtree topology.

Each decision captures the evaluated HKT before reconstruction clears its nodes:
`hktId`, `accepted`, `triggered`, `topologyAtDecision`, `errors`, and
`wordMetrics`. Suppressed descendants have no decision entry. Uninspected new
HKTs have no invented metrics. `wordMetrics` entries use the actual dictionary
key as `wordId` and preserve the raw original object as `values`: the reference
leaves some object's `wordId`, `nodeId`, and `hktId` fields at zero. Likewise
`errors.hktId` may be zero; use the enclosing decision ID.

Relevant raw metric names:

| Quantity | `wordMetrics[].values` property |
| --- | --- |
| Strength state 0 | numberOfSourcesBeforeNewBatchOverNumberOfSourcesOfOldProminWordInHKT |
| Strength state 1 | numberOfSourcesAfterNewBatchOverNumberOfSourcesOfNewProminWordInHKT |
| Eligibility state 0 | numberOfIntersectedSourcesWithOldProminWordInNodeOverNumSourcesOfOldProminWordInNode |
| Eligibility state 1 | numberOfIntersectedSourcesWithNewProminWordInNodeOverNumSourcesOfNewProminWordInNode |
| Importance state 0 | precentageOfSourcesInHKTInOldBatch |
| Importance state 1 | precentageOfSourcesInHKTInNewBatch |

`errors.alphaError`, `betaError`, and `wordImportanceError` contain `option1_error`
and `option1_Thresh`, along with other original option fields. Original spelling,
rounding, zero defaults and nonfinite values are preserved; Newtonsoft represents
NaN/Infinity as strings. Reference logs go to stderr; stdout contains JSON only.
A surfaced error exits nonzero without emitting partial JSON.

## Harness instrumentation and environment differences

`prepare.py` generates execution adaptations from the compilable copy. It does
not alter either reference source tree's existing files or the Rust engine.
Calculations, formulas, rounding and admission/trigger comparisons remain original.

- Replace `Parallel.ForEach` in compiled SECA classes with synchronous iteration.
  This avoids concurrent mutation and schedule-dependent IDs/order. It cannot
  reproduce race-dependent behavior of the original application's parallel run.
  SourceCleanerEngine is referenced as a compiled assembly; its internal loops
  remain unchanged (frequency recomputation can log redundant-add messages).
- Replace GUID allocation with independent sequential HKT/node IDs and sequential
  node sorting IDs. IDs are deterministic per process, not original application IDs.
- Disable construction of the unused persistence `DataHelper`. Its database methods
  remain compiled but are never called. No DataSource.txt or database is required.
- Insert a read-only observation immediately before `isAccepted`; snapshot existing
  metrics and topology without recalculating them.
- Surface exceptions normally swallowed in HKT/Node/SECAAlgorithm catch blocks.
  This changes failure reporting, not successful scientific calculations.

The harness compiles the original algorithm classes with these adaptations rather
than calling the uninstrumented SECA.exe: that executable cannot expose the
pre-reconstruction diagnostics and has nondeterministic execution controls.
Dependencies are referenced from compiled assemblies. Windows Forms types remain
assembly dependencies because unused reference persistence signatures mention them;
no forms, message loop, database, ML preprocessing, or external services execute.
This is an execution reference for comparison, not evidence that Rust and C# agree.

## Manually checked examples

`stable.json`: baseline sources 1 and 2 both contain a,b. At alpha=.7, beta=.5,
there is one root node with a,b and no descendants. Source 3 also contains a,b:
D(a)=D(b)=3; strengths and eligibilities are 1; importances are 3/6=.5;
all Option1 errors are zero, so no reconstruction occurs. The next cycle adds
source 4 and preserves the same ratios and topology.

`reconstruct.json`: the same baseline receives three c-only sources. Updated
prominent count is 3; old strengths are 2/3 and old eligibilities remain 1.
Alpha-Error is .7-2/3 rounded to .0333; Beta-Error is zero. Importance denominator
is 2+2+3=7; old importance sum is 4/7, so WI-Error is 3/7 rounded to .4286.
Only WI exceeds its threshold .3. Reconstruction puts c (three sources) at the
root, and sources 1,2 into a refuge node. Its child contains a,b. A subsequent
c-only source is accepted into the reconstructed root. `verify.py` asserts these
values/topologies, repeated byte-identical output, and invalid-input failure.
