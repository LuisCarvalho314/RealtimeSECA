# Comparative evaluation

## Scope and reproducibility

Investigation date: 2026-10-26. The starting checkout is
`research/seca-hkt-construction`, HEAD `c6cc0ac3edaf7aa2c902d2e6d86f63e9cfc9accc`,
with pre-existing uncommitted validation, invariant-test, scientific-context and
documentation changes. It is **not clean main**. Those changes were preserved.
All new implementation work is under this directory; no production/reference
behaviour or locked fixtures were changed. No branches were merged or published.

The independently implemented prototypes exercise recursive CA construction, the
core used by both initialization and selected subtree reconstruction, plus error
kernels. They are meaningful computational experiments, **not complete SECA
implementations**. Candidate admission, top-down update orchestration, selective
replacement/ID propagation, forgetting, schema-4 restoration, reporting and
transaction guarantees remain unimplemented. The user-permitted small-experiment
fallback is used because implementing those faithfully requires another complete
state machine and resolving the documented contextual ambiguities. No full
SECA-Light equivalence or production rewrite saving is claimed.

The existing simplified Rust checkout at
`/home/azureuser/secalight-simplify`, clean commit
`d7e6e657b6acdc354a2a4bdfe2a672510bf555cb`, was also tested through the identical
read-only construction adapter. Other experimental worktrees were not treated as
validated alternatives. The adapter invokes the public production builder; it is
a baseline, not a third independent prototype. Its package is outside the root
workspace and depends on the core by path.

See [prototypes/README.md](prototypes/README.md) for commands and scope. Raw results:
[results.json](results.json), [direct fixture differential](prototypes/direct/results.json),
[reference replay](reference_replay.json), [metric aggregation](metric_results.json).
The common runner records Python version and source SHA-256 hashes. Timings are
three-trial medians from the same machine; Linux `/usr/bin/time` measures fresh
process peak RSS. These are exploratory, unpinned runs alongside regression tests,
not production capacity or statistically established speed ratios.

## Correctness evidence

| Evaluation | Current Rust | Simplified Rust | Immutable sets | Incidence masks |
|---|---|---|---|---|
| 43 locked Rust baseline topology projections | Baseline | 43/43 through live adapter | 43/43 | 43/43 |
| 100 shuffled synthetic incidence cases, seed 711 | Baseline | 100/100 | 100/100 after correction | 100/100 |
| Five larger common workloads including news | Baseline | 5/5 | 5/5 | 5/5 |
| Legacy C# baseline structural projection | Known differences | Not independently projected to C# | 28/43 agree, 15 differ | Matches same Rust adapter on all 43; inherits those discrepancies |
| Complete evolution/retention/restart | Existing tests pass | Not evaluated here | Not implemented | Not implemented |

Common construction comparisons include ordered descriptors, source membership,
per-word membership, and recursive attachment structure. They deliberately exclude
numeric topology IDs, `top_words`, expected-word export fields and all update
snapshots. Locked structural projections also exclude per-word memberships; the
live adapter adds that comparison. A baseline input from a reconstruction fixture
does **not** test that fixture's reconstruction sequence in a prototype.

The independent agents initially chose different ordering representations. The
first immutable implementation matched all 43 locked fixtures but failed 38/100
randomized cases and three larger workloads. Grouping words lost which word had
the earliest surviving source-word row in a child scope. Immutable ordered pairs
now accompany set memberships; a minimal counterexample and the failing random
case are regression tests. The corrected implementation passes all 143 checks
and five workloads. This is concrete evidence that reducing state without
preserving ordering can change semantics, and that locked examples alone are
insufficient. Neither source permutation nor token permutation is a valid general
metamorphic invariant. Duplicate incidence does leave the construction unchanged.

There are seven direct and seven mask unit tests, covering immutability,
duplicate incidence, branch boundaries, refuge, asymmetric union denominators,
ties, recursive ordering and basic metrics. The direct error reducer also matches
21 actual captured C# decisions within the four-decimal rounding envelope when
fed **C#'s own measurements**. That verifies aggregation, not measurement derivation
or candidate admission. The mask kernel derives staged measurements in its unit
tests but has no general captured-update adapter; update parity remains unknown.

The C# harness self-check passed. All 21 existing construction golden executions
were rerun read-only and matched their captured JSON. The context replay passed,
including its expected empty-input and duplicate-key failure cases; it verifies
repeated deterministic execution. Existing Rust parity suites passed their
characterized differences. Two strict legacy parity tests remain deliberately
ignored by the existing suite; they must not be described as passes. We did not
change their oracle, tolerances or fixture data.

The [semantic specification](semantic_specification.md) and prior executed audits
explain discrepancies: precision before triggers, stale expectations, candidate
beta denominator, refuge evidence, inclusive branch boundary, collision ties,
noncollision selection and recursive scopes. These are baseline interpretation
differences, distinct from the prototype's corrected row-order bug. Shared-word
collapse and candidate history remain scientific uncertainties. C# agreement is
not proof of correspondence to the paper.

## Measured construction time

Milliseconds, median of three. Rust total includes incidence-count preparation,
builder and canonical projection; Python includes relation/matrix preparation,
construction and projection. JSON parsing/encoding, process startup, persistence
and the full SECA update are excluded from time in all columns. Current Rust's
builder-only timer is also in raw results. Rust retains extra production fields
and mirrored indexes; the projection normalizes outputs but does not eliminate
that extra work. Python/Rust language differences prevent predicting the speed of
a future Rust set or mask implementation from these numbers.

| Input | Sources / words / distinct relations | Current Rust | Simplified Rust | Sets (Python) | Masks (Python) |
|---|---:|---:|---:|---:|---:|
| News prefix | 25 / 535 / 743 | 0.47 | 0.48 | 1.50 | 1.14 |
| News prefix | 100 / 1339 / 2862 | 4.10 | 4.06 | 13.00 | 11.37 |
| News prefix | 250 / 2648 / 6955 | 10.37 | 10.40 | 40.15 | 31.64 |
| Controlled overlapping | 100 / 40 / 1707 | 9.24 | 9.34 | 14.65 | 18.81 |
| Controlled overlapping | 1000 / 40 / 17208 | 88.52 | 88.54 | 129.32 | 195.96 |

News input is the repository's checked-in `tests/data/large_batch.json`, with
repeated tokens deduplicated and the first 25/100/250 records retained. It is a
news-derived fixture, not a newly obtained representative production sample;
its coverage is unknown. Controlled cases use the same deterministic relation
generator for all candidates. Parameters are alpha .7, beta .5, branch cutoff 10.
Random correctness cases also exercise alpha .3/.7/1, beta 0/.5/1 and cutoff 2.

These experiments show neither replacement is currently faster on these complete
construction projections. Masks are faster than sets on the news prefixes, slower
on the denser controlled cases. Original-row ordering scans and decoding remain
significant in this mask prototype. No parallelism or acceleration was introduced.
Scaling also depends on recursive topology, not merely source count.

## Memory and structural evidence

| Workload | Rust peak RSS KiB | Simplified RSS KiB | Sets RSS KiB | Masks RSS KiB | Sets traced peak MiB | Masks traced peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| news-25 | 3376 | 3444 | 18548 | 18368 | .44 | .20 |
| news-100 | 5988 | 5944 | 20928 | 19852 | 1.85 | .84 |
| news-250 | 11040 | 10932 | 24608 | 22244 | 4.12 | 2.14 |
| controlled-100 | 5924 | 5980 | 19796 | 19036 | .99 | .39 |
| controlled-1000 | 32112 | 32196 | 30352 | 25612 | 7.27 | 2.48 |

RSS includes executable/interpreter, input JSON and output allocations; the Python
worker imports both candidate modules. It is not exclusive algorithm heap use.
`tracemalloc` is measured in a separate untimed pass and covers peak traced Python
allocations, including normalized output, not total allocations over the run.
There is no equivalent Rust allocator trace here. Masks reduce traced peak memory
relative to corrected sets on every measured workload; this does not establish
retained streaming memory, snapshot size, allocator churn or a Rust memory saving.

Current core: 23 non-test Rust source files, 3504 physical lines; simplified core:
23 files, 3494 lines. Current builder: 530 lines. Each Python implementation module
is 142 lines including its adapters, metric kernel and CLI; neither includes
production integration. Counts include blanks/comments and exclude tests; they
are descriptive, not equivalent-feature rewrite estimates. Core dependencies are
serde, serde_json, thiserror; both prototypes use only Python's standard library.

| Dimension | Current / simplified Rust | Immutable set model | Incidence mask model |
|---|---|---|---|
| State ownership | Ordered topology plus mirrored nodes, retained payloads and contextual indexes | Nested immutable scopes, row ordering and source sets; no global mirror | One immutable incidence table, dense local source positions, scoped masks and ordered row ranks |
| Transformations | Records, cached DF, mutable row maps, scoped copies, topology maps | Ordered relation to sets to nested result; child pair filtering remains | Columns once; child intersections and popcounts; rows still scanned for ties |
| Maintainability | Full operational coverage; more stage/state synchronization | Mathematics visible, easy isolated tests; tuple copies and retained ranks need explanation | Small primitive vocabulary; indexing, bit complements and source decoding need care |
| Integration | Existing RiskLive JSON/CLI/snapshot contracts | A Rust implementation and complete state/report adapters required | Same plus dense-ID mapping and retention/reindexing rules |
| Failure | Clone-based update isolation and restore checks tested | Pure construction can be discarded on failure; no update transaction implemented | Immutable input and local computation can be discarded; no update transaction implemented |
| Scalability | Full-path synthetic evidence in prior performance report | Membership allocation and repeated scoped copies | Dense wide columns, row-rank scans, decoded output and deep recursion |
| Migration risk | Lowest if retained | High for complete rewrite; medium for isolated builder substitution after export parity | High; source-index lifecycle adds substantial correctness exposure |

All prototypes can overflow Python recursion limits on sufficiently deep input;
none is production hardened. Construction has finite progress because ordinary
branches remove descriptors and refuge restricts source scope, but this is not a
practical depth or resource bound. Out-of-memory recovery and hostile-input
limits were not tested. No claim is made about incremental expiry recovery.

## Full-system evidence and remaining work

The pre-experiment full workspace test run passed (the large-data suite took
about 230 seconds). Post-experiment results and required checks are recorded in
[validation.md](validation.md). Existing [performance.md](../performance.md)
contains earlier full update measurements: diagnostics replay can dominate clone
cost and gamma does not bound learned vocabulary/topology. Those are historical
measurements, not this investigation's new prototype timings. Current experiments
do not profile end-to-end production RiskLive or attribute its cost to construction.

Before replacement can be recommended: implement contextual mapping/admission,
expose comparable per-word measurements without changing decisions, test all
update snapshots, overlapping descriptors and candidates, selective child rebuild,
ID/sibling preservation, expiry/readmission, empty batches, fault rollback and
JSON restart equivalence. Compare exact exports as well as structural projections.
Use archived RiskLive input sequences, pin trials, and measure allocation counts,
long-run retention and final diagnostics separately. No estimate of full rewrite
runtime, engineering duration or total production LOC is supported yet.
