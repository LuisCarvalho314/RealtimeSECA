# Recommendation

Retain the current production implementation. The investigation supports replacing
some computational representations, but does not support a complete rewrite's
cost or risk today. The strongest finding is that construction can be expressed
as a small ordered set/mask computation without mutable source-word row deletion
or mirrored topology ownership. Neither prototype yet proves the complete
state transition or RiskLive continuation/report contracts.

## Ranked choices

1. **Keep the current canonical SECA-Light path and use these experiments as an
   independent differential oracle.** Resolve scientific/reference discrepancies
   before selecting a new compatibility target. The tested simplified Rust
   checkout is materially the same construction baseline, not an independently
   superior computational architecture. Its measured differences are too small
   and uncontrolled to justify switching.
2. **If a rewrite is pursued, prefer a direct Rust batch computation with explicit
   immutable inputs and one owned hierarchy.** Ordered source-word incidence,
   scoped set operations and pure error calculation make the semantics visible.
   Use local mutable accumulators where clearer; persistent collections are not
   necessary to obtain a pure state-transition interface. Prototype A demonstrates
   construction, not an estimated 142-line production engine. A bounded builder
   replacement could be validated before attempting the much larger update path.
3. **Keep incidence masks as the competing implementation for a memory-driven
   decision.** They consistently reduced traced Python peak allocations in these
   cases, but had workload-dependent speed and require local-to-stable ID handling.
   Test a Rust standard bit vector or a compressed bitmap only when real density
   and lifecycle measurements justify the dependency. No mask library replaces
   the SECA controller or resolves scientific ambiguity.
4. **Defer relational engines, persistent collection infrastructure and maintained
   delta indexes.** They add contracts and operational machinery without a measured
   benefit for this batch-oriented consumer. Generic community detection and
   fresh-window clustering are algorithmic variants and need separate news-analysis
   utility evaluation; they are not compatible replacement shortcuts.

## Answers to the architectural questions

The current high-level architecture is appropriate: a deterministic computational
core, tokenized batch boundary, selective hierarchy evolution, atomic state
transition and infrastructure owned by the caller fit RiskLive. Its precise
in-memory representation is not mathematically necessary. Source-word row records
with duplicated frequencies, temporary child datasets and two synchronized node
owners can be replaced. The prototypes demonstrate this for construction, while
also exposing how seemingly incidental row order affects results.

The existing implementation can plausibly become simpler without changing the
important properties. Eliminate destructive row-removal bookkeeping and repeated
DF recomputation over duplicated row objects; derive counts from scoped incidence.
Store nodes once and derive lookup/export projections at boundaries if the full
update contract proves it safe. Remove stage-specific intermediate objects when
ordinary explicit values suffice. These are candidate production changes, not
changes authorized or made by this exploration. The current core already lacks
the old compatibility policies and fallback engines; do not reintroduce them.

Keep the necessary ordering: greedy node merges change later denominators;
top-down triggers suppress children; reconstruction precedes forgetting. Retain
contextual per-word membership, stable vocabulary and IDs, original descriptors
for W0, exact incoming reconstruction scope and sibling preservation. A single
global frequency matrix cannot silently replace contextual descriptor counts.

Viewer diagnostics are separate observations, not another mathematical update.
The existing repeated clone/mapping/traversal is a promising stage to eliminate
through shared read-only evidence computation **if** the same final-topology
coverage/availability can be preserved. The historical performance report supports
investigating that cost before removing transactional cloning. Current prototypes
do not validate such fusion. Whole-state clone removal is not a priority without
equivalent failure isolation and evidence that it matters.

A fundamentally different complete implementation is **not yet demonstrated to
be preferable**. A direct ordered set architecture is the clearest candidate for
maintainability; incidence algebra is the strongest alternative storage model.
The measured Python prototypes are slower than Rust construction on the tested
inputs. That does not predict Rust rewrite performance, and latency is secondary
anyway. Their small size does not include the difficult continuation/report work.

## What production adoption would involve

First freeze an explicit target: current operational Rust semantics with named
C# discrepancies, or a newly approved scientific revision. Keep the locked C#
oracle independent. Add differential access to temporary word measurements and
candidate memberships so disagreement can be located before final topology.

Next implement one complete pure batch transition over an owned hierarchy and
retained evidence, using either sets or masks throughout. Construction and selected
reconstruction must share that same implementation. Preserve transactional discard
on any error, selective subtree identity rules and update-before-expiry behavior.
Avoid parallel node assignment and generic alternative-policy frameworks.

Then provide exact tree/report and schema-4-compatible persistence projections,
or explicitly version snapshots and replay archived tokenized batches. A tree
export cannot migrate the missing evidence. RiskLive's independent 3d/7d/30d
models, receipt ledgers, SQLite transaction, subprocess exit behavior, report
archives and viewer diagnostics must continue to work. Run side-by-side shadow
replays without publication before any cutover. Remove the old production path
only after equivalence is established; there should still be one canonical update
path after migration.

The main risks are recursive tie order (already caught experimentally), candidate
history and scope, duplicate descriptor counting, full-precision trigger boundaries,
expiry changing dense indices, nonrecycled IDs, rollback, and snapshot/report
compatibility. Benefits could include less duplicated evidence, fewer synchronized
structures and more directly testable mathematics. The present evidence does not
quantify whole-engine maintenance savings or demonstrate that those benefits repay
a full rewrite. A focused computational-kernel experiment is justified; a production
rewrite is not currently justified.
