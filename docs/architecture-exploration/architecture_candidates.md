# Architecture candidates

All equivalent candidates target [the reconstructed contract](semantic_specification.md),
including ordered greedy construction and selective updates. None may substitute
“rebuild the latest window” for evolution: forgotten evidence can leave learned
structure, and preserved siblings/IDs are observable. The alternatives below change
representation or computation, rather than module names. Complexity and resource
predictions are hypotheses until the common experiments in [comparison.md](comparison.md).

## 1. Direct set transformation — selected as prototype A

Represent a scoped input as word-to-source ordered sets; nodes are small records
of descriptors and memberships. Implement construction directly as ordered set
intersections/unions and recursion, without source-word row objects, mutable
row deletion, duplicated global node ownership or stage object lifecycle. A pure
`construct(scope, parameters)` returns the tree; an eventual `update(old,batch)`
would return a candidate state and report before commit.

This deliberately favors explicit operations and standard collections over
indices or generalized policies. Scoped frequencies become set lengths. A local
word map and chosen-node accumulator replace repeated row scans and transformations.
Ordering, contextual memberships and recursive scope remain explicit arguments.
Expected implementation complexity is low for construction, moderate for complete
evolution and public reports. It integrates naturally into Rust without another
runtime. Tree sets still allocate heavily; purity alone does not eliminate copies.
The prototype evaluates the construction kernel only, not complete update parity.

## 2. Algebraic incidence masks — selected as prototype B

Treat source-word incidence as a Boolean matrix. Give retained sources dense
local positions and represent each word column, node membership and scope as a
bit mask. Scoped evidence is column AND scope; frequencies and collision scores
are popcounts; unions are OR; ancestor exclusion clears candidate columns.
Construction operates on masks directly rather than converting back to sets per
operation. Ordered greedy assignment remains a scalar control dependency.

This eliminates per-membership tree allocations and makes the computational
primitives explicit. It is a fundamentally different storage and execution model,
although it cannot remove semantic order dependence. Dense masks cost O(V*N/word)
space even for rare words; sparse/compressed masks may be preferable. Dense local
IDs must remain distinct from stable external IDs, with explicit restore/expiry
mapping. A production arena could own topology once and export ordered maps only
at the boundary. Implementation complexity is moderate; remapping and snapshot
migration are its main new risks. The experiment uses dependency-free masks, not
an unvalidated library and adapter stack.

A mature substitution exists for the set primitive: Rust
[RoaringBitmap](https://docs.rs/roaring/latest/roaring/bitmap/struct.RoaringBitmap.html)
supports cardinality-only intersections/unions. Its documentation establishes the
primitive operation, not end-to-end SECA equivalence or an advantage on RiskLive.
Source IDs require a compact mapping or a 64-bit representation; density and
retention behavior must be benchmarked before choosing compressed storage.

## 3. Relational evidence with ordered greedy driver

Store distinct `(batch,source,word)` rows, `(node,source)` and `(node,word)` relations.
Compute scope frequency and overlaps through joins/grouping; a small ordered
host loop performs node selection, membership expansion and top-down suppression.
An embedded query engine can evaluate all candidate overlaps in a scope as a
batch rather than issue repeated procedural searches. State is relations plus
ordered topology and identity metadata, committed transactionally.

This is attractive if evidence is already queried relationally for other analysis.
[DuckDB recursive CTEs](https://www.duckdb.org/docs/current/sql/query_syntax/with)
can express traversals and intermediate relations. That documentation does not
establish that a recursive query simplifies greedy stateful assignment: every
accepted word changes later denominators and ties. SQL order must be explicit,
COUNT DISTINCT must match source-set semantics, and query-engine float reduction
order must not silently change triggers. SQL/FFI/deployment/schema overhead plus
host control flow is expected to exceed the small batch kernel's benefit. Reject
for the first prototype round; no database merely to replace ordinary sets.
RiskLive's existing SQLite archive is a persistence boundary, not evidence that
SECA should execute inside it.

## 4. Immutable versioned state and path replacement

Represent each context as an immutable record with immutable evidence indexes;
`step(state,batch)` yields a new root plus a report. Accepted changes replace only
touched paths, reconstruction installs a new subtree, and expiry replaces affected
evidence chunks. Readers hold a consistent version, and discarding an incomplete
version gives rollback without cloning every unaffected subtree.

This changes ownership, transaction strategy and execution model rather than just
using const references. It can simplify failure isolation and concurrent readers.
Persistent maps or reference-counted chunks add allocation/reference-count overhead;
source expiry touches many nodes, and shared retained evidence can keep old versions
alive accidentally. Stable public IDs and explicit version cleanup remain necessary.
Pure functions with ordinary owned values are useful immediately (candidate 1);
a full persistent data-structure architecture is deferred until clone/memory
measurements justify its larger migration surface. It does not remove the
algorithm's contextual evidence or greedy dependency.

## 5. Incremental materialized counts with expiry deltas

Maintain an inverted index from source to affected nodes and per-scope word counts.
New/expired source events update those indexes and counts; a batch barrier triggers
top-down error decisions and reconstruction. Keep the current update-before-expiry
ordering: expirations cannot be applied early just because delta processing makes
that convenient. Topology rebuild invalidates associated derived indexes.

This can avoid repeatedly scanning retained token payloads. It does not require
RiskLive to become a streaming system; daily batches can supply deltas. It adds
mutable derived state and difficult provenance/rollback requirements. One source
can appear in multiple overlapping branches, so a global delta count is insufficient.
Stale cache bugs can alter reconstruction without obvious crashes. Expected
complexity is high relative to batch scanning; defer while latency is secondary.
It becomes plausible only if representative profiling identifies retained evidence
scans as a dominant production cost.

## 6. Bipartite graph/hypergraph computation

Use a bipartite graph of sources and words, with node contexts as induced scoped
subgraphs/hyperedges. Neighborhood intersection gives eligibility and masks;
subgraph restriction defines branches. A graph store can share adjacency evidence
between contexts and derive memberships lazily. This removes replicated contextual
incidence tables at the cost of repeated scoped adjacency intersections.

The exact version is effectively an alternative incidence engine, potentially
useful when downstream co-occurrence queries already need graphs. Generic graph
libraries add little to ordered set algebra and often require ID adapters.
[NetworkX's community algorithms](https://networkx.org/documentation/stable/reference/algorithms/community.html)
optimize partition/modularity or other graph objectives; those are not this
asymmetric, ordered, overlapping SECA grouping. Replacing SECA with Louvain,
connected components, hierarchical clustering or formal concept enumeration is an
algorithmic variant, not a library drop-in. No equivalent complete mature SECA
library was identified in this bounded search; this is not proof none exists.
Reject generic graph infrastructure for the prototypes; retain algebraic adjacency
primitives in candidate 2.

## 7. Event replay and fresh-window construction

An alternative is to store an append-only normalized batch log and reconstruct
state by replay, keeping only checkpoints as optimization. This is semantically
equivalent if replay executes the exact ordered SECA transition, and simplifies
recovery/version comparisons; it trades runtime and growing archive requirements
for less mutable authoritative state. It does not remove the transition itself.
RiskLive already optionally archives inputs, so replay is useful migration
infrastructure rather than a superior new core solver.

Building a fresh CA tree from only the last gamma batches instead eliminates
learned zero-support topology and ignores threshold history; it can dramatically
simplify the product, but is an algorithmic variant. It would require separate
news-analysis utility evaluation and revised public contracts. Reject as a
replacement claim, not as a future scientific/product experiment.

## Assumptions challenged and consequences

| Inherited assumption | Finding |
|---|---|
| Many stage objects are essential | No; scoped evidence/counts and an explicit transition suffice in principle. Full reports still need an adapter. |
| Existing control flow is mathematically required | Ordered greedy word assignment, top-down suppression and expiry after evolution are required; row conversion, pending-object merge mechanics and a second viewer sweep are not. |
| One source belongs to one cluster | False; overlaps and refuge recursion are essential. |
| Every structure must be mutable | False; pure construction and immutable transaction results are feasible. Mutable local accumulators remain simpler than copying on every word. |
| Full rebuild is equivalent to selective evolution | False; topology, sibling IDs and history differ. |
| Parallelism is desirable | Unproved. Greedy node unions are sequential; independent scope counts could parallelize, but deterministic reduction/ID order must be enforced. Batch latency does not justify concurrency by itself. |
| Generic policies/frameworks help | No demonstrated alternative production policies; avoid them. |
| More cache state is necessary for performance | Unmeasured. Direct incidence masks can remove row transformations before adding caches. |
| Directly coding the paper settles semantics | False; its ordering, shared-word accounting and construction scope omissions need explicit decisions. |
| Fewer lines proves maintainability | False; a small construction prototype omits lifecycle, restore, diagnostics and rollback work. Compare equivalent feature sets. |

The selected pair attacks representation first: explicit set mathematics versus
word-parallel incidence algebra. It provides an executable disagreement detector
and tests whether row machinery is necessary, while keeping incomplete lifecycle
work visible. Relational, persistent-state and delta architectures have distinct
merits but insufficient expected benefit to justify two more infrastructure-heavy
prototypes before that smaller experiment.
