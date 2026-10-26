# Recommendation

Retain the current production path. The native experiments justify continued
work on a bounded builder alternative and provide a useful independent evolution
oracle; they do not yet justify adoption or a broader rewrite. This conclusion
rests on measured allocation/performance trade-offs and compatibility obligations,
not on deployment status or the alternatives merely being incomplete.

Scientific correctness remains a separate question. Matching current production
`e0ca91e` preserves its interpretations; it does not resolve the manuscript's
ambiguities. See [scientific_equivalence.md](scientific_equivalence.md).

## Ranked assessment and answers

1. **Keep production and retain the native direct engine as an independent research
   oracle.** Both construction kernels match343 normalized construction cases,
   including fields excluded by Iteration1. The direct lifecycle is executable
   through selective reconstruction and forgetting, rather than a construction-only
   proposal. See [validation](evolution_validation.md) for the final stream counts,
   exact fields, error tolerance and exclusions. This is stronger operational
   evidence, not a publication correctness certificate.
2. **Prefer ordered scoped incidence and one owned hierarchy as the next bounded
   architectural candidate.** It most directly expresses SECA's set operations,
   overlapping contexts and ordered greedy dependencies. Direct sets simplify the
   *expression* of construction: no row deletion or synchronized hierarchy mirror.
   They have not demonstrated a material production-equivalent simplification:
   production already removed the old construction mirror and input archive, the
   candidate still duplicates membership/rank keys locally, and public adapters
   remain to be implemented. Its larger news cases allocate substantially more
   than production and are slower. A small source count alone is not an adoption
   argument. See [native architecture](rust_architectures.md) and
   [measurements](benchmark_results.md).
3. **Keep masks as a focused kernel option, not a selected production representation.**
   They win on some controlled inputs, but the mixed set/mask implementation adds
   columns and node masks while retaining set storage; it does not improve the
   available news workloads consistently. No real retained3d/7d/30d evidence
   establishes that its density or churn favors dense or compressed bitmaps.
   A fully indexed evidence engine remains a distinct experiment. Adding a bitmap
   dependency is not justified by these results.

The complete normalized lifecycle demonstrates that the representation can support
more than the builder. Single hierarchy ownership removes a genuine mirror equality
invariant across mapping, replacement and forgetting. It does not remove retained
evidence, contextual measurements, deterministic target ordering or transactions.
Candidate update timings exclude production's external-ID/token registration,
public reports, final-topology diagnostics and validated persistence; their ratios
cannot be called a whole-engine speedup. A broader rewrite's net maintainability
benefit is still unquantified after accounting for these necessary boundaries.

**Essential decisions:** sequential candidate adoption, deterministic seed/tie
order, context-local word/source evidence, overlapping routing, learned descriptors
separate from expectations, top-down trigger suppression, exact reconstruction
incoming scope and ancestor exclusions, unaffected siblings, nonrecycled IDs,
update-before-expiry, and atomic failure semantics. The current *particular*
clone mechanism is not mathematically essential, but its guarantee is. Public
snapshot, diagnostic and identity behavior is operationally essential when
compatibility is claimed, even where it is not part of the mathematics.

**Structures that are not mathematically necessary:** the runtime full-node mirror,
materialized empty pending maps in a staged implementation, mutable row deletion,
and repeated parent-scope DF computation. The first can become a boundary view
only with an adapter covering all accepted production snapshot shapes, including
multiple children on one attachment. The last requires demonstrating safe reuse
across evolving scopes; an unvalidated cache would add more obligations than it
removes. Old build-wide archives/mirrors are already absent from latest production
and cannot be counted as new savings. Retained source tokens or equivalent incidence
evidence and source-word ordering information are necessary, not cleanup targets.

**Unresolved science:** W0's temporal treatment of unassigned expectations; unique
word versus node-occurrence importance mass; shared-word metric reduction; scope
versus contextual descriptor DF; historical candidate beta and descriptor persistence;
frequency versus vocabulary traversal; recursive exclusion and branch support/cutoff;
and forgetting's precise boundary. The independently verified node-union beta
normalization, set-valued counting, stale-evidence correction and full-precision
comparisons have stronger support than those unresolved choices. The scientific
report distinguishes verified arithmetic from observed behavior and proposals.

**Production decision:** no production changes now. The smallest safe next step
is to retain these independent differential tests and use them to review a bounded
builder patch that eliminates the candidate's extra allocations and supplies exact
production output projection. Such a patch must pass the locked behavior and
schema-4 continuation oracles through actual reconstruction, not only constructor
comparisons. The existing implementation wins the present deployment decision
because of the measured news/allocation trade-off and the outstanding integration
contract, not simply because it already exists.

**Further work before semantic changes:** resolve the named equations jointly
with candidate persistence and contextual source definitions; independently replay
the published case study and verify the final paper against the accessible accepted
manuscript; obtain representative archived RiskLive token streams and retained states
for all three model horizons; test late failures after accepted-parent mutation;
and assess long-run allocation, expiry and reconstruction behavior. The tiny container
validation archives and the checked-in news prefix are useful fixtures, not a
representative production sample. A scientific variant such as level exclusion or
frequency traversal must be evaluated separately from a compatible builder rewrite.

## Algorithmic structure: equivalent representations versus variants

**Ordered grouping.** Source intersection and union can be indexed, batched,
vectorized or represented as masks. Node choice remains sequential because every
adoption changes the denominator seen by the next candidate. For example let
S(a)={1,2}, S(b)={2,3}, beta=1/2 and admit both. With a first, b joins at1/2 and
the union becomes{1,2,3}. Add source4 containing a: now the seed has three
sources and b scores1/3, causing separate nodes on reconstruction. Updating only
the previous node's memberships cannot reproduce that split. Swapping seed or
candidate order can change all later assignments. Ordinary clustering, simultaneous
assignment or merge-only adjustment therefore changes the algorithm in general.
An equivalent optimization must preserve the ordered decisions or prove its
shortcuts cannot change their comparisons. No such general shortcut was proved.

**Selective reconstruction.** Replaying greedy construction over a triggered
scope is a well-defined way to recover its exact descriptors and memberships.
A transformation model could cache prefix decisions and replay from the earliest
changed comparison; finding that point and invalidating dependencies are additional
state obligations. In the worst case the first decision changes. This is a
possible optimization, not evidence that reconstruction is avoidable. A fresh
whole-window hierarchy also differs: accepted siblings retain descriptors and
IDs, forgotten evidence can leave learned empty nodes, and a triggered parent
suppresses all its child decisions. These are observable evolution properties.
Persistent path replacement can preserve those properties without physically
cloning siblings, but persistence does not itself establish the replacement's
scientific semantics or schema-4 compatibility.

**Historical evidence.** One immutable or append-only ordered source-word index
can share evidence across levels. Each context still needs its source scope,
ancestor exclusions, learned descriptors and contextual memberships. A single
global count cannot replace intersections with each scope. Expiry must run after
trigger/reconstruction decisions, and source identity may be readmitted after
expiry. Evidence chunks shared by old versions must actually become releasable;
otherwise a nominal sliding window does not bound retained memory. Incremental
counts and expiry deltas trade repeated scanning for invalidation and rollback
contracts; the present experiments do not justify adding those caches.

**Overlapping hierarchy and DAGs.** A source-word incidence graph naturally
represents evidence once. Overlap does not require shared hierarchy ownership:
multiple context-local nodes may reference the same sources. Merging two hierarchy
nodes solely because their descriptors or memberships currently match can change
parent scope, future incoming routing, ancestry exclusions, trigger suppression,
public IDs and subsequent reconstruction. A DAG with separate contextual node
identities and shared evidence can be representationally equivalent; a DAG that
identifies those nodes changes the state transition unless stronger equivalence
is proved. Generic graph clustering replaces the objective. No speculative graph
framework was added.

## Adoption gates

A bounded builder replacement may proceed only after exact construction fields,
ordered topology IDs, expected/top-word annotations, reconstruction entry ordering,
and existing locked public/snapshot continuation oracles pass using the replacement.
The replacement must be one canonical path once adopted. Keep old fixtures read-only
and retain named legacy differences. It also needs bounded resource behaviour and
reviewed helper/function size, plus repeated release measurements with all adapter
costs counted. A clear reduction in production-equivalent state and code can
justify adoption without a latency win; a tiny kernel excluding adapters cannot.

A broader engine rewrite additionally requires exact public reports/diagnostics,
external source and vocabulary registration, schema-4 accepted-state validation,
late failure rollback, monotonic allocators, retained evidence equivalence and
3d/7d/30d shadow replay through representative archived RiskLive streams. Neither
normalized IDs nor matching tree exports certify these boundaries. A scientific
change needs a separately approved mathematical contract, equation witnesses,
legacy discrepancy characterization and multi-batch revalidation. Resolving one
formula while leaving related candidate persistence or reduction ambiguous is
not sufficient.
