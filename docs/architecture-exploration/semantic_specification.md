# SECA computational specification

This is a representation-independent reconstruction for the exploration, not a
replacement for the approved [scientific contract](../algorithm.md). Evidence:
current source at `c6cc0ac`, original and compilable C# under `references/`, the
locked `tools/seca-reference-harness`, and the executed
[metrics](../parity-context-metrics.md) and
[construction](../parity-hkt-reconstruction.md) audits. The C# desktop application
is not itself a deterministic mathematical specification. Its serial harness is
the principal legacy behavioural oracle; current Rust is the operational contract.
They disagree in documented cases, so “parity” must always name its target.

## Problem and values

Given an ordered sequence of batches of token sets, maintain a hierarchy of
contextual word groups and overlapping document memberships. Detect sufficient
change in a context to reconstruct that context and its descendants; otherwise
retain its learned descriptors. This is neither a partition into news events nor
a generic hierarchical clustering objective. The repository implements hierarchy
evolution, not the paper's subsequent direct/indirect event clustering evaluation.

A source has an external identity, batch index and token set. Token repetition
within a source has no weight. Input order matters for vocabulary IDs and ties;
token strings are trimmed and compared case-sensitively after upstream NLP.
Parameters are strength admission alpha, node admission beta, branch cutoff,
three error thresholds and retention gamma (or accumulated Full memory).

A context/HKT contains an ordered list of nodes. Each ordinary node has descriptor
words, a source set and contextual per-word source sets; a refuge node represents
unmatched sources. Child contexts refine an attachment node's source scope.
Expected words are candidates, not automatically descriptors. State also retains
source-word evidence, source provenance, stable vocabulary and monotonic topology
identities. Exact maps, objects, cached nodes and pending fields are not essential
mathematical state. Outputs are updated topology/memberships, decisions and their
errors, plus enough state to reproduce the next update. RiskLive additionally
requires stable tree/report JSON and resumable schema-4 snapshots.

## Construction as an ordered operation

For a source scope R, define S(w) as its distinct sources containing w and
D(w)=|S(w)|. Select potential descriptors with D(w)/max D >= alpha. The first
highest-frequency word seeds a node; stable source-word record order breaks its
ties. Process the other potential descriptors in ascending vocabulary-ID order.
Assign each to the existing node maximizing |S(w) intersect N|/|N| when this is
>= beta, otherwise start a node. Strict improvement chooses the first equal
maximum. Assignment expands N by union, so later scores depend on earlier words.
This is an asymmetric, sequential greedy grouping, not Jaccard clustering.

Remove adopted word rows, not their sources. Sources left without ordinary
membership become refuge. For each node whose source count exceeds the branch
minimum, build a child from its source scope without that attachment node's words;
refuge retains all its scoped words. Recompute scoped document frequencies and
repeat. Preserve the current Rust pre-construction scope at recursive levels.
The C# recursion and inclusive branch cutoff are different in characterized cases.
Top words are export annotations, not an extra clustering stage.

## One update

1. Validate source/batch consistency before deduplication. Keep the first occurrence
   of an identity within a batch and ignore identities still retained. Register
   previously unseen normalized words without changing old descriptor vocabulary.
2. Visit contexts top-down, root with the whole new batch and children with only
   sources matched to their attachment node. Match a source to every ordinary
   node having any descriptor in it; otherwise route it to existing refuge.
   Skip scopes with no incoming evidence.
3. Freeze W0 as the union of old ordinary descriptors. Form updated descriptor
   memberships by old/new union. Obtain candidate counts from retained parent
   scope plus incoming evidence, exclude ancestor-adopted words, admit by strength,
   and recheck against the updated prominent count. Candidate node admission uses
   incoming overlap divided by the updated full node union; passing beta does not
   silently turn candidates into W0 descriptors.
4. Let c(n,w) count the updated contextual membership of descriptor w in node n.
   Let k(w) count an alpha-admitted candidate's retained-plus-incoming evidence.
   Set P=max(1, all c and k), T=max(1, sum of all descriptor occurrences c plus
   all candidate k). For a measured word in node n, strength=c/P,
   eligibility=|word sources intersect updated N(n)|/max(1,|N(n)|), importance=c/T.
   Candidate-only measurements use k/P, zero eligibility and k/T. A word measured
   in multiple nodes is overwritten deterministically in ascending node-ID order;
   T nevertheless counts descriptor occurrences separately. This is an operational
   convention, not a proof that the paper's word-set formula intends duplication.
5. Errors are mean max(0,alpha-strength) and mean max(0,beta-eligibility) over W0,
   and clamp(1-sum importance over W0,0,1). Empty W0 gives zero first two errors.
   If any full-precision error strictly exceeds its threshold, select that subtree
   for reconstruction and do not inspect its descendants. Otherwise expose updated
   memberships for child scope processing and retain learned topology.
6. Reconstruct selected, disjoint subtrees from retained evidence plus exactly the
   triggering incoming scope, excluding ancestor descriptors. Preserve unaffected
   IDs/topology; assign fresh nonrecycled IDs to replacements and propagate parent
   source unions. This is selective evolution, not a complete-window rebuild.
7. After evolution, remove source evidence older than current-gamma+1. Empty batches
   advance this boundary. Retain learned words/topology and allocators. Commit the
   whole update only on success; failure must leave the prior model unchanged.

These formulas require contextual memberships: using one global word frequency
for every descriptor can change the result. See `engine/metrics.rs` and
`engine/evidence.rs` for the operational distinction.

## Invariants and compatibility

One reachable root; every child has one live attachment; word membership is a
subset of node membership; every member references retained evidence; descriptor
and expected-word IDs resolve; identities are unique in retained evidence; all
committed pending memberships are empty. Restart must reproduce subsequent reports
and topology exactly. Gamma bounds batches, not learned topology, vocabulary or
sources per batch. No assumption of single-node document assignment is valid.

RiskLive's actual consumer is `../src/services/seca_timeline.py`: separate 3d/7d/30d
models, subprocess `update`, JSON snapshots in SQLite, independent lifetime
receipt ledgers, archived tree/report pairs and final-topology viewer diagnostics.
Its atomic multi-model transaction and publication recovery are infrastructure,
not SECA equations. A production replacement must preserve those boundaries and
report availability semantics or explicitly version/replay them. Viewer observations
can be computed separately; their extra full-tree traversal is not needed to decide
which scopes reconstruct. A tree export alone cannot resume the algorithm.

## Unresolved reference/scientific differences

| Case | Legacy C# | Current operational Rust |
|---|---|---|
| Errors near threshold | Rounded to four decimals before comparison | Full precision |
| Consecutive expected words | Can retain stale counts | Recomputes accumulated evidence |
| Candidate beta | Prominent-word source denominator | Full node union denominator |
| Refuge update | Can omit new unmatched membership | Commits mapped refuge union |
| Branch equality | Inclusive cutoff | Exclusive cutoff |
| Equal collision score | Last qualifying node | First qualifying node |
| Noncollision creation | Next remaining frequency-first word | Current admitted word in ID order |
| Recursive scope | Removes level descriptors in deeper C# scopes | Attachment-node exclusion with pre-construction scope |
| Shared candidate | Duplicate-key failure | Deterministic word-map overwrite |
| Empty batch | Harness rejects it | Valid; advances retention |

The locked fixtures must stay untouched. Tests should assert agreement on shared
behaviour and explicitly assert known differences. Silently adding C# defects to
an “equivalent” rewrite would violate the current contract. The paper does not
resolve shared-descriptor accounting, W0 excluding expectations, historical
candidate beta numerator, exact gamma boundary, or all ordering/recursive-scope
choices. Existing research proposals are proposals, not approved semantic changes.

Mathematically necessary work is scoped incidence counting, ordered greedy
construction, memberships, threshold decisions, selective replacement and
post-update expiry. Source-word row cloning, mirrored node ownership, string-to-ID
remapping per scope, temporary object lifecycles, full-model cloning and separate
viewer traversal are implementation decisions. They can change if the same
state transition, failure guarantees and public observations are demonstrated.
