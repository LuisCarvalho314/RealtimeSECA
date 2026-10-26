# Evolution validation

Construction equivalence and evolution equivalence are different claims. This experiment does not replace `SecaEngine` or route an experimental update through production calculation. Production engine tests establish the operational comparison contract; they do not establish an independent candidate's scientific validity.

## Required comparison boundary

The common JSON adapter accepts ordered normalized source-word relations, alpha, beta and branch minimum. Its `full` result compares exact HKT and node IDs, attachment IDs, expected-word sets, ordered nodes, descriptor sets, source sets, per-word contextual source sets, top-word sets and recursively attached child scopes. Refuge is represented explicitly by descriptor `-1`. Scope membership is recoverable as the union of node memberships. Construction has no pending evidence, which is always empty in production after construction; the candidate does not store redundant empty pending maps. `is_state1` is excluded (uniformly false in production construction); lifecycle annotation compatibility is unsupported. Engine vocabulary/external source legends, reporting, snapshots and allocator continuation are outside this adapter. Thus full-construction agreement cannot imply schema-4 snapshot compatibility.

## Lifecycle requirements and evidence

Production has tests for full-memory retention, sliding-window expiry, dead-node preservation until reconstruction, candidate admission, overlapping contextual source measurements, deterministic trigger plans, repeated reconstruction, upward source propagation, ancestor exclusions, pending-state merging, snapshot validation and source provenance. The exact baseline suite result is recorded in `baseline.md`.

The completed candidate lifecycle experiment and its executable boundary are documented below. Cases absent from that experiment are **unsupported**, not passing. In particular, no constructor differential test counts as a test of candidate alpha/beta/word-importance errors or selective reconstruction.

| Requirement | Evidence required before claiming complete equivalence |
|---|---|
| Validation and deduplication | Invalid duplicates fail before deduplication; first valid occurrence wins; retained external identities are ignored |
| Registration | Persistent vocabulary IDs, deterministic external source hashing with collision probing, forgotten-source readmission |
| Traversal and mapping | Top-down context restrictions, overlapping descriptors, refuge and ancestor exclusions |
| Candidates and errors | Successive-batch evidence, unassigned expected words, exact metrics and strict trigger thresholds |
| Reconstruction | Parent suppression, multiple disjoint targets, unaffected sibling payload and ID preservation, monotonic allocator continuation |
| Forgetting | Every evidence index and identity map pruned; vocabulary/topology retained; empty batches advance cutoff |
| Transactions | Invalid input and late failure leave the complete previous state unchanged |
| Restart | Equivalent continuation from production schema-4 state and equivalent observable IDs |

## State ownership findings

The existing `hkts_by_id` owns topology; `nodes_by_id` mirrors whole node payloads. Production updates in `context.rs` synchronize both during apply/merge and forgetting synchronizes both after pruning. A single owned recursive hierarchy makes this particular equality invariant unnecessary. Removing the cache in production still requires proving performance and public-state compatibility, and replacing snapshot validation that currently checks the mirror.

Sets alone do not remove historical evidence requirements. Node descriptors remain fixed during accepted updates, while candidate memberships and expected words may evolve independently. Forgetting deliberately preserves topology and vocabulary even if a node becomes empty. Reconstructing from only the surviving node source union cannot recover all contextual token evidence, especially for expected words absent from old descriptors. Retained token payloads or an equivalent global source-word incidence index remain necessary.

A transaction clone is an independent design decision from the builder's representation. An owned hierarchy reduces synchronization work but does not prove that validation, metrics, partial subtree replacement, reporting and forgetting cannot fail halfway through. Removing cloning requires a complete staging or undo strategy; no such production-equivalent strategy has been validated here.

## Executed native evolution evidence

Final `evaluate_evolution.py` runs **110 independent sequences, 426 attempted
updates and 823 scope decisions** against the production engine. Ten named cases
cover stable overlapping memberships, empty expiry/readmission, retained and
within-batch duplicates, candidate evidence across batches, previously expected
words, repeated reconstruction, invalid sequence rollback, sliding-window
candidate history, two disjoint child targets with an unaffected third sibling,
and parent-trigger suppression. Another100 seed92071 streams vary alpha/beta,
thresholds, empty batches, Full/gamma2/gamma3 and four successive batches.

All resulting normalized hierarchy states match **exactly** after every attempted
batch, including failed attempts; all823 error triples and trigger decisions
match as f64 values without tolerance. All topology and attachment IDs, ordered
nodes, descriptors, expected words, contextual memberships, refuge and top words
match. The runner asserts update/state lengths and fails on missing output or any
mismatch. Complete inputs, states and production reports are in
[`raw/evolution.json`](raw/evolution.json). Both engines serialize and restore
between batches: production uses its schema-4 validator; the candidate uses its
own serde state. This proves continuation for each representation on these
sequences, **not cross-loading their snapshot formats**.

An adapter issue initially flagged two random sequences: production's
`reconstructed_hkt_ids` reports selected targets even when the scoped evidence
is empty and no replacement is installed. Candidate `replacements` records only
installed replacements. The corrected comparison checks selected decision IDs
against that production report; actual replacement topology IDs remain checked
through exact resulting state. This preserves rather than erases the distinction.
No algorithmic mismatch was found in these sequences.

Twelve independent candidate integration tests derive invariants from explicit
sets rather than production outputs. They cover overlapping routing, exact
candidate-history importance2/5,1/2,6/11, independent alpha/beta deficits,
strict threshold equality, parent suppression and nonrecycled allocators, two
child replacements with the third sibling unchanged, nested expiry, learned
empty-node preservation, readmission, invalid-before-dedup rollback, normalized
serde continuation and unsupported empty-root rollback after registration.
Five construction unit tests pass separately. Their log is retained in `raw/`.

Three read-only archived **container validation fixture** sequences, labelled
3d/7d/30d, also match exact normalized states. Each has only two baseline sources
and eight incoming sources, with shared fixture tokens and repeated row identities.
They are not representative production retention workloads. The six source files
are hashed in `raw/archived-container-inputs.json`; reproducible normalized inputs
and output states are retained. No live model, database or published output was
changed. Available archives do not establish long-run30-day density, retention
cost or actual RiskLive event utility.

| Requested case | Candidate result / limitation |
|---|---|
| No-change, multiple batches, empty batch | Passed exact normalized state/decisions |
| Repeated reconstruction and parent suppression | Passed; fresh IDs and no child evaluation under selected parent |
| Multiple disjoint targets and unaffected sibling | Passed exact IDs, annotations and memberships |
| Overlapping node memberships | Passed; sources route to every matching node |
| Shared descriptor memberships at one HKT | Independently calculated scientifically; not installed through a public lifecycle trace |
| Expected words and successive candidate evidence | Passed targeted plus randomized sequences |
| Full and sliding retention, expiry/readmission | Passed; nested memberships expire while topology/vocabulary/allocators persist |
| Failed sequence/invalid normalized input | Passed complete candidate-state rollback; production report failure strings excluded |
| Failure after normalized vocabulary registration | Passed empty-root unsupported boundary rollback |
| Failure after accepted-parent mutation or mid-replacement | **Untested** candidate failure boundary; production has its own regression |
| Restart continuation | Passed each engine's own state format; schema-4 interoperability **unsupported** |
| External identities, FNV collisions, string normalization and per-source batch validation | **Unsupported** at normalized candidate boundary |
| Arbitrary accepted historical snapshot shapes, `is_state1` and viewer observations | **Unsupported** exact public compatibility |
| Allocation failure, stack exhaustion, allocator overflow | **Untested/unrecovered** |
| Real3/7/30-day retained production stream | **Unavailable**; container fixture validation is insufficient |

The candidate now implements a complete normalized mathematical update path,
including selective reconstruction and forgetting; it is not merely an initial
builder or an evidence-only mapping simulation. It still lacks production's
complete input/persistence/report boundary. Neither the missing boundary work nor
the production lifecycle code is counted as candidate functionality by delegation.
There is no call from the candidate algorithm into production computation.

## Reproduction

Run from the pinned production checkout described in `baseline.md`:

```sh
cargo build --offline --release --manifest-path docs/architecture-exploration/iteration-2/evaluation-baseline/Cargo.toml
python3 docs/architecture-exploration/iteration-2/evaluate_evolution.py > /tmp/seca-evolution.json
python3 docs/architecture-exploration/iteration-2/archive_check.py
cargo test --offline --manifest-path docs/architecture-exploration/iteration-2/rust-candidates/Cargo.toml
```

The evaluator package's additional candidate dependency belongs only to comparison
and allocation probes; neither candidate library nor production library depends on
the other. Production and candidate transition timing use separate binaries with
ordinary System allocators. Counting-allocator probes run separately and untimed.
Normalized numeric source IDs are mapped through production legends; numeric token
spellings are padded for lexical registration. Targeted/randomized inputs control
initial registration ordering explicitly. Arbitrary already-assigned normalized
IDs are not a claim of equivalence to arbitrary external token registration order.
