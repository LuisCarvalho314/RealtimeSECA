# First-principles V2 implementation report

Base: fetched origin/main e0ca91eb62e1e5843ddb47d98ec71a388a67f6d5.
Branch: research/seca-v2-first-principles. Worktree: /tmp/seca-v2-first-principles.
A sibling worktree was initially created, then moved to /tmp to allow sandboxed
writes. The original dirty checkout and all other worktrees remain unchanged.
No merge or push was performed. V1 core sources are unmodified.

## Delivered implementation

[Specification](specification.md), [state/architecture](architecture.md),
[crate](../../crates/realtime-seca-v2), and
[comparison harness](../../benchmarks/seca-v2/README.md).
The scientific specification preceded Rust implementation. E01–E07, C# construction
and metric source, existing behavioral fixtures and the paper benchmark protocol
were examined. C# was not rerun; this task's executable legacy comparison is against
clean origin/main Rust V1. Existing C# evidence remains labelled as prior evidence.

The algorithm has two dynamic authoritative structures: the owned recursive tree
and retained source payloads. Node source sets describe contextual membership;
retained source words are independent evidence, not a second node representation.
There are zero independently mutable indexes or node-cache synchronization rules.
One update walk directly replaces disjoint triggered scopes; SECA-Light adds one
forgetting walk. Temporary per-scope word-source sets, candidates and exclusions
are derived and dropped. No second reconstruction planning/execution pass exists.
There is no persistent vocabulary or expectation cache: learned words live in
nodes and eligible expectations are computed from evidence.

Core: 498 physical Rust lines in three files, including API, types, continuation
validation and formatting; 20 functions/methods, no traits beyond Serde's required
Deserialize implementation. The recursive constructor is 67 lines and evolution
is 57 lines; mapping is a separate 25-line set operation. None exceeds 80 lines. The 429-line mathematical test file was reviewed as
one cohesive set of explicit, hand-calculated examples; production files remain
under 225 lines.
For context, V1 engine/tree files total 2,982 physical lines including diagnostics
and snapshot machinery, excluding config/types and tests. These counts cover
different reporting surfaces and are not a scientific performance metric.
The reduction is principally removal of mirrored nodes, numeric ID registries,
pending context stages, scope snapshots, rebuild plans, remapping and transaction
clones. Input validation completes before mutation; remaining work is infallible
apart from unrecoverable resource failures. There is no production V1 dependency.

## Discrepancy classification

- **Intentional scientific corrections:** distinct-source scope DF and unique
  importance vocabulary; historical candidate beta; persistent accepted adoption;
  frequency-first construction; full-level exclusions at every construction entry.
  Focused hand-derived and V1 comparison tests demonstrate these changes. No V1
  goldens were rewritten to make differences disappear.
- **Explicit unresolved interpretations:** worst-node shared-word beta reduction;
  descriptor-only W0; inclusive residual-bearing branch support (V1 counts total
  parent sources exclusively, C# inclusively); lexical/first-node ties; empty-scope
  skips; gamma boundary; promotion of an existing descendant descriptor. Each
  choice is specified with alternatives and discriminating tests. Residual support
  is an additional deliberate departure from the earlier proposal to retain total
  parent support, grounded in the original CA branch paragraph.
- **Intentional representation/lifecycle choice:** tokenless sources remain in
  refuge, including at child scopes, to preserve exact source-scope coverage.
  Derived expected words need no persistence because future admission recomputes
  from retained evidence. Old learned descriptors remain after forgetting.
- **Observed quality regression:** Tweets Full mean BCubed F1 falls from
  71.0519% to 70.8128%, with lower recall despite slightly higher NMI/purity.
  Tweets Light F1 falls from 75.6879% to 75.4970%. Both are measured
  regressions under this protocol; their causes remain unresolved.
- **Algorithmic/lifecycle regressions found:** none in the executed focused tests.
  This is not proof of equivalence. Every real benchmark topology differs from V1
  even after ID/order normalization. Aggregate quality and differing reconstruction
  counts cannot isolate the causal contribution of each scientific choice; those
  compound differences remain unresolved rather than being blanket-certified.

The benchmark does not certify unchanged event assignments: co-member differences
are recorded for each cycle. It also does not establish universal quality gains.
Remaining author-level scientific questions are enumerated in specification.md;
V2 is a working research implementation, not an approved production migration.

## Validation

`cargo fmt --all -- --check`: passed.
`cargo test --workspace --offline`: 201 passed, zero failed, two existing ignored
strict C# parity tests (rounding and known construction discrepancies). The full
run included the long large-dataset regression tests. Final V2-specific tests:
13 mathematical/lifecycle and three V1 comparison tests passed after the final
changes. `cargo clippy --workspace --all-targets --offline -- -D warnings`: passed.
Existing clustering tests: 21 passed. Evaluation tests with SECA_TEST_ENGINE set
to the built CLI: 25 passed, no skips. No new C# execution or exact published
experiment replication is claimed. Offline Cargo used locally cached dependencies.

## Executed scientific benchmark

Two archived prepared themes: G/News (8,880 sources, 12 x 740) and H/Tweets-T
(24,000 sources, 12 x 2,000); Full and gamma=3 Light. Eight engine/mode/theme
streams, 96 measured cycles. Same ordered inputs, alpha=.7, beta=.5, branch
minimum=10, error limits .1/.2/.3; indirect contextual merge threshold 1/2.
Current-batch scoring; arithmetic means over 12 cycles. This reconstructs the
repository protocol, not the original published experiment.

| Theme | Mode | Engine | NMI % | B3 precision % | B3 recall % | B3 F1 % | Purity % |
|---|---|---|---:|---:|---:|---:|---:|
| G | Full | v1 | 87.1674 | 74.7380 | 60.7815 | 60.8876 | 79.7748 |
| G | Full | v2 | 87.3765 | 75.6642 | 60.5942 | 61.1667 | 80.4617 |
| G | SlidingWindow | v1 | 86.5879 | 72.7048 | 60.5942 | 59.8558 | 78.0068 |
| G | SlidingWindow | v2 | 86.7215 | 73.3005 | 60.2679 | 59.9190 | 78.3559 |
| H | Full | v1 | 85.2491 | 90.6564 | 64.7321 | 71.0519 | 93.0500 |
| H | Full | v2 | 85.2761 | 90.8918 | 64.4519 | 70.8128 | 93.2583 |
| H | SlidingWindow | v1 | 87.5554 | 91.1727 | 70.8902 | 75.6879 | 93.6208 |
| H | SlidingWindow | v2 | 87.6116 | 91.4332 | 70.5375 | 75.4970 | 93.8292 |

Per-cycle mean false-merge / fragmented-truth pairs (not percentages):

| Theme | Mode | V1 merge / fragmentation | V2 merge / fragmentation |
|---|---|---:|---:|
| G | Full | 959.92 / 1638.42 | 921.67 / 1653.67 |
| G | SlidingWindow | 1094.67 / 1640.83 | 1056.42 / 1654.42 |
| H | Full | 3466.00 / 46419.67 | 3359.58 / 46746.00 |
| H | SlidingWindow | 4079.92 / 38699.50 | 3938.50 / 39242.58 |

| Theme | Mode | V1 seconds / peak MiB | V2 seconds / peak MiB | V1 final HKTs / nodes | V2 final HKTs / nodes |
|---|---|---:|---:|---:|---:|
| G | Full | 38.021 / 181.3 | 1.938 / 161.3 | 619 / 2699 | 704 / 3145 |
| G | SlidingWindow | 14.235 / 62.1 | 0.962 / 42.4 | 257 / 1101 | 291 / 1251 |
| H | Full | 141.995 / 629.1 | 7.716 / 545.1 | 2391 / 10185 | 2473 / 11273 |
| H | SlidingWindow | 62.749 / 167.2 | 3.355 / 130.4 | 854 / 3303 | 862 / 3575 |

Timing includes update and tree export/value conversion, with standard V1
diagnostics. RSS is process-wide and includes file/serialization work; V2 checked
round trips run every cycle. Single release-mode runs, with possible workspace-test
contention: these are observed end-to-end costs, not isolated algorithm benchmarks.
No optimization was added. Quality changes must not be traded away for speed.

G Full: normalized topology differs in 12/12 cycles; reconstructed scopes V1/V2 total 372/434; changed indirect co-member sets 80–292 documents per batch.

G SlidingWindow: normalized topology differs in 12/12 cycles; reconstructed scopes V1/V2 total 192/260; changed indirect co-member sets 80–426 documents per batch.

H Full: normalized topology differs in 12/12 cycles; reconstructed scopes V1/V2 total 124/127; changed indirect co-member sets 758–1445 documents per batch.

H SlidingWindow: normalized topology differs in 12/12 cycles; reconstructed scopes V1/V2 total 32/35; changed indirect co-member sets 589–1370 documents per batch.


V2 improves News aggregate NMI/F1/purity but lowers recall and increases
fragmentation. Tweets Full F1 declines by about 0.24 percentage points and Light
F1 by about 0.19 points. Precision and purity gains do not erase these regressions.
These measurements support a compact executable prototype, not a claim that the
scientific tradeoffs are settled or that V2 is ready to replace V1.

Raw per-cycle scores, dataset hashes, normalized comparison outcomes, compressed
ID-only assignments, resource measurements and source provenance are under
[benchmarks/seca-v2/evidence](../../benchmarks/seca-v2/evidence). Full trees and
raw decisions remain in /tmp/seca-v2-benchmark-G and /tmp/seca-v2-benchmark-H;
the harness regenerates them. See its README for exact commands and definitions.

Direct memberships remain overlapping and were not converted to hard partitions.
Per-batch overlap/unassigned document ranges and final retained source counts:

G/Full/v1: direct overlaps 58–88, direct unassigned 0–0, final retained 8880.

G/Full/v2: direct overlaps 60–87, direct unassigned 0–0, final retained 8880.

G/SlidingWindow/v1: direct overlaps 59–80, direct unassigned 0–0, final retained 2220.

G/SlidingWindow/v2: direct overlaps 60–80, direct unassigned 0–0, final retained 2220.

H/Full/v1: direct overlaps 143–287, direct unassigned 0–0, final retained 24000.

H/Full/v2: direct overlaps 148–289, direct unassigned 0–0, final retained 24000.

H/SlidingWindow/v1: direct overlaps 50–233, direct unassigned 0–0, final retained 6000.

H/SlidingWindow/v2: direct overlaps 49–254, direct unassigned 0–0, final retained 6000.
