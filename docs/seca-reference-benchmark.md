# C# reference versus Rust: Themes G and H

## Scope and measured answer

This experiment compares accumulated SECA on the identical prepared G/H inputs,
using the existing reconstructed Python clustering and metric procedures. Base:
`origin/integration/seca-paper-benchmark`, fetched and verified at `459117c`.
Worktree: `/tmp/seca-reference-benchmark`; branch:
`experiments/seca-reference-benchmark`. No Rust learning, scientific C# formulas,
existing benchmark evidence or golden fixture was changed.

Both C# streams completed all 12 learning cycles. Some reference snapshots cannot
be evaluated by the unchanged shared clustering adapter because child memberships
exceed their parent refuge memberships. Comparable quality can therefore be
assessed only on jointly scoreable cycles; a full-stream quality-equivalence claim
is unavailable. Direct overlaps also prevent the existing hard-partition metrics.
On those jointly scoreable cycles, the engines have close descriptive quality: paired-cycle mean differences are 0.08–0.46 percentage points across the three metrics. This does not establish full-stream equivalence. The numerical results and affected cycles are reported below.

These are **C# engine evaluated using the shared reconstructed clustering
procedure** results. They are not recovered published outputs of the original
Windows application, nor an equivalence proof for SECA or SECA-Light.

## Reference identity and environment

The reference is the supplied adapted original C# algorithm classes under Mono
6.8.0.105. Its archive has no upstream Git metadata; **the upstream C# revision is
unknown**. Source-content hashes, rather than an enclosing Rust repository revision,
identify it. [Provenance](../benchmarks/seca-paper/reference/evidence/provenance.json)
records all compiled source hashes, corresponding original archive hashes, the
five binary dependency hashes, generated instrumentation hashes, original harness
hashes, checkpoint wrapper/binary hashes and the exact shared scoring file hashes.
All compiled-input source hashes match the existing executable parity register.

Original and compilable algorithm classes are unchanged. DataHelper has the
previously documented Mono compatibility changes to C# 8 using declarations; its
persistence constructor remains disabled. The source-to-binary correspondence of
the supplied SourceCleanerEngine dependency remains unverified.

The existing local harness was built and verified **before experimental changes**:
manual metrics/topology, subsequent updates, deterministic repeats and invalid-input
failure passed, with 22 inherited compiler warnings. The checkpoint wrapper uses
that same generated reference and `Execution.cs`. Existing adaptations serialize
algorithm-class parallel loops, allocate deterministic incidental IDs, disable
unused persistence, expose observations and surface normally swallowed exceptions.
They do not reproduce the uninstrumented application's execution scheduling or
post-exception continuation. SourceCleanerEngine's compiled internal frequency
loops remain parallel; their redundant-add diagnostics remain in full stderr logs.
They did not surface a G/H exception. They are not silently discarded or presented
as a completely serial application.

The only new reference execution adaptation is **output checkpointing**. The wrapper
writes compact JSON atomically after each completed original cycle, including
actual topology and original decision/metric observations. It releases previous
snapshot objects instead of retaining the entire stream until final stdout JSON.
No scientific formula, admission comparison, reconstruction decision, branch
control or accumulated lifecycle was changed. Checkpoint snapshots equal all
21 HKT fixtures' 65 raw snapshots and all 17 successful context goldens exactly.
A shared-candidate failure test confirms that the successful baseline survives
while no failed-cycle snapshot is emitted.

## Identical inputs and native configuration

Prepared G/H manifests and all 28 prepared files were independently verified
against the pinned raw files. G is the ordered 8,880-document News prefix in
12 batches of 740. H is the ordered 24,000-document Tweets-T prefix in 12 batches
of 2,000. Their original-paper selection/order uncertainty is unchanged; see
[the original benchmark report](seca-paper-benchmark.md).

External identities retain the original namespace and zero padding. C# positive
integer IDs are assigned by ordered document position, beginning at 1, separately
for each stream. The committed gzip mappings record the bijection, batch index,
within-batch position, overall position, original ordered-token hash, original
token count and distinct-word count. Repeated texts remain different sources.
Mappings are regenerated and checked during comparison. The engine receives one
row per distinct source-word pair, preserving first token occurrence; the prepared
original ordered tokens, including repetitions, remain unchanged. This reflects
Rust ingestion's distinct memberships and the existing C# harness contract, not
additional linguistic preprocessing.

| Setting | Both engines / interpretation |
|---|---|
| Alpha / Beta | 0.7 / 0.5 |
| AlphaError / BetaError / WordImportanceError | 0.1 / 0.2 / 0.3 |
| Branch minimum argument | 10 |
| Memory | Rust Full; C# accumulated, `isLightSECA=false` |
| Error option | C# Option1 for all three errors |
| Evaluated population | complete current batch |
| Indirect contextual cutoff | exact inclusive `1/2` |
| Metrics | existing `seca-hard-v1` NMI, document-mean BCubed F1, purity |

C#'s Option1 BetaError default is 0.13; the explicit benchmark value 0.2 is passed
instead. Every captured G/H decision's effective thresholds were verified as
0.1/0.2/0.3. Construction's minimum significant-word support remains its existing
constructor default of 1. The branch argument has different native meanings:
**C# branches at support ≥10; Rust at support >10** (E06). Neither implementation
was adjusted to hide this difference. C# rounds errors to four decimals, while
Rust preserves its existing full-precision decision comparisons (E08).

## Execution and retained evidence

Stages include original golden verification, all executable golden checkpoint
comparisons, the first 32 News documents, the entire first News batch, three News
batches and both complete streams. The first-batch and three-batch News outputs
are byte-identical to their corresponding full-G outputs. Goldens and corpus
learning outputs are retained; there is no continuation after a reference exception.

Both complete streams succeeded: G took approximately 119 seconds and H 671
seconds in this environment. These timings describe feasibility only; this is not
an optimized or controlled performance benchmark. The configured resource limit
was 1,800 seconds per corpus stage and was not reached. H was executed once;
full-stream deterministic replay is not claimed. [Stage metadata](../benchmarks/seca-paper/reference/evidence/stages.json)
records successful cycle counts, input hashes, binary/source fingerprints and
stderr hashes. The existing shared-candidate fixture independently reproduced
its duplicate-key exception after one completed baseline (E01); its input hash,
reference hashes, stderr and successful prefix are preserved.

Raw inputs, raw C# cycle checkpoints, normalized trees, descriptor strings,
unmatched-node details and reconstruction-scope paths remain under ignored
`benchmarks/seca-paper/reference/local/`. Their hashes are inventoried. Restricted
C# classes and dependencies are not in Git. Following the existing corpus policy,
token-bearing trees and dataset payloads are also kept locally. Committed evidence
includes ID-only memberships, topology attachments, descriptor fingerprints,
identity mappings, reconstruction-scope fingerprints, assignments, summaries and
CSV tables. Exact descriptor comparisons are performed on tokens, not only hashes;
the hashes allow publication without redistributing the corpus vocabulary payload.

## Representation comparison and scientific interpretation

Translation preserves ordinary/refuge nodes, descriptor words, HKT attachments and
actual source memberships. It never unions pending maps into actual memberships.
Raw pending maps, dynamic expected sets, errors and pre-reconstruction topology
remain available in checkpoints. Baseline metrics and suppressed-descendant
observations are not invented. The final topology is the actual state after each
completed reconstruction/update lifecycle.

Canonical tree comparison recursively sorts semantic sibling subtrees and removes
incidental IDs. It compares descriptor/refuge topology separately from labelled
source memberships. Word ordering, node ordering and allocator IDs are not treated
as scientific differences. Construction expected-set persistence differences
remain raw observations and are excluded from the labelled topology equality
claim, consistent with E12.

Both themes first diverge substantively at **cycle 1**. The branch-boundary analysis
finds 8 G and 13 H otherwise identical ten-source parent paths with a C# child
branch and no Rust child. This establishes an E06 contribution to baseline
construction divergence. It does not explain every descriptor/membership difference
or authorize a correction. E05 recursive-scope and E07 greedy-order differences
remain plausible contributors already established on small fixtures; these corpus
runs do not isolate their individual effects.

Reconstruction decision counts and triggered counts are recorded per cycle for
each engine. Affected scopes use prior ancestor descriptors and prior scoped source
sets, with ID-independent fingerprints; exact paths are retained locally. Different
learned contexts are not forcibly aligned. A disagreement in reconstruction count
alone cannot isolate E08 precision, E09 stale expectation, or another metric effect.
No corpus-wide causal attribution to those discrepancies is established.

For each cycle, ID-only node exports preserve actual source memberships and
attachments. Ordinary-node source co-membership compares every unordered pair in
the current batch, independently of node IDs; event co-membership and per-document
direct membership families are compared wherever the shared adapter succeeds.
Families retain overlaps. Unscoreable snapshots still have representation evidence,
but no fabricated event assignment. [Per-cycle representation CSV](../benchmarks/seca-paper/reference/evidence/per-cycle-representations.csv)
contains HKT/node counts, depth, ordinary/refuge counts, decisions, triggers, scope
violations and equality classifications.

The invalid-scope observations are consistent with **E10**: child sources absent
from their parent refuge. Rust's complete refuge union is scientifically justified
by the documented unmatched-source contract; it is not changed for C# parity.
Every captured G/H scope violation has a refuge parent. Their exact excess source
identities are retained in compressed node evidence. The unchanged shared adapter
rejects these native trees with `child sources must be a subset of parent sources`.
No parent repair, descendant union, validation bypass or alternate C# clustering
procedure was introduced. This is a reference-state/evaluation-input failure,
not an engine process exception and not a measured zero clustering score.

## Shared clustering quality

Both engines are evaluated by the exact existing `cluster.evaluate` implementation
in `benchmarks/seca-paper/clustering/cluster.py` and the existing metric evaluator.
Identical original current-batch tokens, cutoff, population and definitions are
used. No ground truth enters clustering. Rust's saved tree/report hashes are
checked against the original benchmark, and all regenerated Rust assignments
must equal the preserved assignment evidence exactly.

Published values are the existing independently verified 2023 Table 3 values.
Original published per-cycle scores are unavailable. Per-cycle differences from
the published mean provide context, not recovered per-cycle targets. Partial
paired means are reported separately from 12-cycle aggregates and do not estimate
missing C# cycles. See [the generated comparison](../benchmarks/seca-paper/reference/evidence/comparison.md),
[per-cycle metric CSV](../benchmarks/seca-paper/reference/evidence/per-cycle-metrics.csv)
and [aggregate metric CSV](../benchmarks/seca-paper/reference/evidence/aggregate-metrics.csv).

| Theme / metric | Paired cycles | Published | C# paired | Rust paired | Abs. C#–Rust Δ | C# paired − published | Rust paired − published | Rust full |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G / nmi | 1, 2 | 88.30 | 86.48 | 86.25 | 0.22 | -1.82 | -2.05 | 87.17 |
| G / bcubed_f1 | 1, 2 | 63.03 | 59.55 | 59.33 | 0.23 | -3.48 | -3.70 | 60.89 |
| G / purity | 1, 2 | 80.36 | 76.28 | 75.88 | 0.41 | -4.08 | -4.48 | 79.77 |
| H / nmi | 1, 2, 3, 5 | 87.52 | 83.43 | 83.05 | 0.37 | -4.09 | -4.47 | 85.25 |
| H / bcubed_f1 | 1, 2, 3, 5 | 71.14 | 66.90 | 66.82 | 0.08 | -4.24 | -4.32 | 71.05 |
| H / purity | 1, 2, 3, 5 | 89.18 | 90.80 | 90.34 | 0.46 | 1.62 | 1.16 | 93.05 |

Scores are percentages; Δ and signed differences are percentage points. C# full-stream means are undefined in both themes. The paired and published populations cover different cycle sets.

G cycles 1–2 differ by 0.22/0.23/0.41 points in NMI/F1/purity. H cycles 1–3 and 5 differ by 0.37/0.08/0.46 points. C# paired means are slightly higher on these observed cycles. Individual cycle gaps reach 1.44 points (H cycle 1 F1); the differences between means can cancel differences in opposite directions. Without a predefined equivalence margin, original per-cycle scores or complete C# scoring, these are descriptive proximity results, not statistical equivalence.

The representation evolution below includes every successful learning cycle. Pairs are C#/Rust; “violations” counts C# child-node edges with excess memberships. Rust has zero violations in every cycle.

| Theme / cycle | HKTs | Nodes | HKT depth | Reconstructed scopes | Violations |
|---|---:|---:|---:|---:|---:|
| G / 1 | 42/34 | 263/240 | 8/8 | 0/0 | 0 |
| G / 2 | 102/96 | 516/498 | 9/9 | 6/5 | 0 |
| G / 3 | 117/103 | 619/575 | 11/11 | 30/20 | 92 |
| G / 4 | 196/203 | 910/933 | 12/11 | 10/11 | 45 |
| G / 5 | 269/304 | 1120/1250 | 13/11 | 36/6 | 81 |
| G / 6 | 292/322 | 1251/1370 | 14/13 | 47/46 | 158 |
| G / 7 | 320/349 | 1405/1539 | 16/13 | 48/53 | 204 |
| G / 8 | 412/408 | 1756/1786 | 14/13 | 33/33 | 119 |
| G / 9 | 431/519 | 1859/2222 | 15/13 | 52/32 | 235 |
| G / 10 | 491/537 | 2059/2342 | 16/14 | 55/58 | 223 |
| G / 11 | 533/576 | 2232/2509 | 17/14 | 52/59 | 258 |
| G / 12 | 570/619 | 2394/2699 | 17/15 | 51/49 | 293 |
| H / 1 | 219/200 | 814/721 | 11/11 | 0/0 | 0 |
| H / 2 | 461/413 | 1889/1667 | 12/12 | 1/1 | 0 |
| H / 3 | 659/615 | 2904/2593 | 13/13 | 4/4 | 0 |
| H / 4 | 856/789 | 3795/3369 | 14/14 | 5/5 | 10 |
| H / 5 | 1036/972 | 4628/4190 | 14/14 | 1/1 | 0 |
| H / 6 | 1279/1194 | 5702/5129 | 15/15 | 9/9 | 8 |
| H / 7 | 1508/1422 | 6495/5897 | 15/15 | 6/6 | 10 |
| H / 8 | 1756/1650 | 7812/6982 | 15/15 | 8/7 | 13 |
| H / 9 | 1932/1814 | 8744/7865 | 16/16 | 6/6 | 9 |
| H / 10 | 2135/2017 | 9678/8658 | 16/16 | 13/12 | 7 |
| H / 11 | 2380/2235 | 10560/9544 | 16/16 | 25/20 | 35 |
| H / 12 | 2599/2391 | 11288/10185 | 18/18 | 54/53 | 78 |

No complete labelled topology/membership snapshot matches across engines in either theme.

| Theme / cycle | C# direct overlaps | Rust direct overlaps | Documents with different direct membership families |
|---|---:|---:|---:|
| G / 1 | 60 | 60 | 4 |
| G / 2 | 73 | 60 | 53 |
| H / 1 | 153 | 152 | 12 |
| H / 2 | 206 | 204 | 12 |
| H / 3 | 200 | 202 | 21 |
| H / 5 | 178 | 179 | 11 |

Direct assignment retains every overlap on valid snapshots. Rust direct metrics
remain undefined in every cycle because the population is not a hard partition.
C# direct metrics are likewise never converted to hard clustering; valid assignments
and explicit per-document membership lists are preserved. Invalid C# scopes produce
no direct or indirect shared-adapter assignment. Those limitations preclude any
published-direct or full-stream C# quality comparison.

## SECA-Light availability

The archive does contain `clearMemory(..., isLightSECA=true)` and
`removeSourcesFromMemory`; absence of C# forgetting code is **not** claimed.
Inspection found a lifecycle mismatch with simply enabling that flag in the
accumulated harness. The desktop retrospective controller appends incoming rows,
saves/updates the database and then clears memory. Persistence also resets rebuilt
`isState1` flags. The existing accumulated harness clears dead structures before
copying incoming rows and explicitly resets flags before the next update. These
orders are immaterial for its established accumulated path but not interchangeable
when removing memberships.

The native source filter is `cycleId > currentCycle - (noCyclesToSaveInMemory - 1)`.
With correctly appended current rows, a setting of 3 therefore retains two cycle
cohorts after forgetting, whereas Rust gamma=3 retains three. Calling native
forgetting before appending current rows would additionally remove incoming node
memberships from its active-source set. The native cleanup marks empty structures,
while some node-removal code is commented out. The application saves its visible
state before forgetting; the Rust benchmark exports its committed retained state.

A faithful, independently validated original Light observation/continuation
contract and original experiment retention configuration have not been established.
**SECA-Light comparison is unavailable in this benchmark.** No flag-only reuse,
new C# forgetting policy, gamma adjustment or guessed snapshot phase was introduced.
The accumulated comparison is complete with its explicit scoring limitations.
The lifecycle/controller, original/compiled cleanup and persistence source hashes
are recorded so this missing executable validation is reviewable.

## Reproduction and validation

[Tooling README](../benchmarks/seca-paper/reference/README.md) gives complete local
build, verification, staged execution, translation, comparison and evidence commands.
Missing prepared datasets and Rust trees are regenerated with the unchanged commands
in [the existing benchmark report](seca-paper-benchmark.md). New output paths are
required, preserving previous successful outputs and failures.

Validation: 9 focused reference/comparison tests, 21 existing clustering tests,
25 existing evaluator/integration tests (including real Rust CLI execution), and
13 real pinned-dataset tests passed. All 28 prepared files independently verified.
`cargo fmt --all -- --check` passed. No Rust source/tooling, module boundary or state
ownership changed; workspace tests/strict Clippy were not rerun for this Python/C#
harness-only addition. Their existing benchmark results remain unchanged, not
claimed as new validation. No parameter tuning or performance optimization occurred.

Unresolved assumptions remain: unknown upstream C# revision/dependency build,
adapted execution versus the full Windows application, original dataset selection
and ordering, native branch-boundary interpretation, original clustering procedure,
cutoff, evaluated population, averaging convention, missing original per-cycle
quality data and independently unvalidated Light lifecycle. Close partial scores
cannot resolve those uncertainties or establish full scientific parity.
