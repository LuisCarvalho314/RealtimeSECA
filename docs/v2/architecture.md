# V2 architecture and state contract

The [scientific specification](specification.md) owns algorithm semantics.
V1 architecture.md/state.md continue to govern V1 unchanged.

`Model` owns configuration, one recursive `Hkt`, retained token payloads keyed by
external identity, and the next batch position. `Hkt` owns ordered nodes; each
`Node` owns its descriptors, source identity set, and optional boxed child HKT.
There are no node IDs, mirrored nodes, parent indexes, word IDs, vocabulary cache,
word-membership indexes, persistent expectations, pending memberships or plans.
A child represents the one residual dataset of its parent. General multiple-child
storage would have no operation creating a second child under this specification.

`recursive::construct` builds any scope from evidence and ancestor exclusions.
`recursive::evolve` maps incoming sources, computes errors, replaces immediately
or admits candidates and descends. `map_incoming` is a small local set operation;
`frequencies` derives temporary source sets. `forget` intersects memberships after
one batch-level evidence eviction. metrics.rs contains pure set calculations.
lib.rs validates input, owns state, and provides checked Serde continuation.
No production dependency on V1 exists. Only dev/example code imports its public
API for comparison; the scientific core has no filesystem or reporting framework.

A successful update covers the complete root source scope. Every child source is
in its parent's membership; every source reference resolves to retained evidence.
At most one refuge exists per HKT. Ordinary descriptors are nonempty token sets;
empty descriptors denote refuge without a marker token. Learned structures survive
zero support. Construction exclusions are path-local, not a global word blacklist.
Accepted parent promotion may leave an old descendant adoption, as explicitly
specified; restoration must permit that state. Candidate/source metrics derive
freshly from payloads, never from number of node memberships.

Input validation, including malformed duplicates, completes before mutation.
Thereafter update has no recoverable error branch, so no transaction clone or
rollback structures are required. Allocation failure/stack exhaustion is not
handled as a recoverable error. Source identity is exact; tokens must already be
normalized and nonempty; sets remove repetitions. First identity wins until it is
forgotten. Deduplication precedes post-update forgetting: an identity still present
at inspection is ignored even if it expires at this batch's end. Invalid batch sequence/configuration/input returns an error without
changing state. Consecutive batch numbers count empty observations toward gamma.

Serde serializes only canonical model state. Deserialize validates config,
retention provenance/window, reference resolution, parent subset relationships,
refuge uniqueness and scope coverage. This is a research continuation format,
without V1 snapshot compatibility or a future-stable schema promise. Reports are
returned values, not retained model state. Path indices identify an inspected old
scope for that update; replacement preserves the attachment but may reorder nodes.

There is one top-down evolution pass, construction only at disjoint triggered
scopes, and one forgetting walk for Light. Frequencies and candidate/old-word sets
are derived per inspected HKT. This deliberately favors comprehensible direct
recursion over persistent indexing. Recursion depth is bounded by consumed token
vocabulary/source reduction at construction, but not by a fixed defensive limit.
