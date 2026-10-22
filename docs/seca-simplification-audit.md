# SECALight production simplification audit

Base: `main` at `d7e6e657b6acdc354a2a4bdfe2a672510bf555cb`.
Investigation only; no production changes. Isolated worktree:
`/tmp/seca-simplification-audit`, branch `audit/seca-simplification`.
References and line numbers below describe this revision.

## Assessment

A substantially smaller implementation is feasible without changing the science.
The best targets are duplicated node ownership, three parallel ancestry maps,
construction-wide retention of recursive inputs, and repeated conversion of the
same evidence. The engine already has one update path and no policy framework;
introducing traits, strategies, an arena library, or a general pipeline would
move it away from the objective.

The conservative first programme should target **roughly 250–420 net production
lines removed**, mainly from ancestry/planning, builder temporaries, and local
copying. A more invasive second programme can remove runtime node duplication and
the diagnostic engine clone. Its benefit is primarily fewer owned payloads and
synchronisation obligations, not guaranteed additional line savings. These are
review estimates, not measured patches or performance results. Do not promise a
50% code reduction: much of this small core is necessary algorithm, validation,
and public serialization.

Counts from `wc -l`/source enumeration, including comments and blank lines:

| Area | Existing lines | Interpretation |
|---|---:|---|
| Core `src`, excluding tests | 3,484 | Includes contracts, config, errors and exports |
| Mapping/context/evidence/metrics/trigger | 1,114 | Main update staging and traversal |
| Baseline/rebuild/builder | 1,070 | Conversion, construction, replacement |
| Diagnostics | 174 | Full observation sweep and private engine clone |
| Forgetting/snapshotting/validation | 397 | Mostly necessary boundary work |
| CLI `src`, including embedded tests | 1,106 | Timeline file alone is 381 lines, including 77 test lines |

Read [repository rules](../AGENTS.md), [algorithm](algorithm.md),
[architecture](architecture.md), [state](state.md),
[pending decisions](scientific-decisions-pending.md), and the supplemental
[proposals](scientific-decision-proposals.md). The two decision documents are not
fully aligned: pending E03 proposes descriptor persistence after acceptance,
whereas the supplemental E02 discussion retains provisional assignments until
reconstruction. Neither is approval. Preserve current behaviour and resolve this
scientifically before implementing either interpretation.

## Existing execution and ownership

`SecaEngine` owns configuration, optional topology, append-only word vocabulary,
retained token batches, two identity directions, batch provenance, monotonically
advancing allocators, and last-report summaries
([engine/mod.rs:33](../crates/realtime-seca-core/src/engine/mod.rs#L33)).
`hkts_by_id -> Hkt.nodes` owns nodes, while `nodes_by_id` owns complete copies of
the same nodes. Synchronisation is manual in construction, context updates,
ancestor propagation, replacement, forgetting, and diagnostic views.

Baseline validates and deduplicates sources, clones the engine, converts tokens
to source-word rows, builds the tree, retains a stripped batch, constructs a
report, then installs the candidate. Updates also clone first. Inside the
candidate, `prepare_batch` validates order and source records before deduplication,
registers source identities, computes the public batch summary, and admits new
word IDs. Traversal computes retained source-word sets once, then visits scopes
top-down. Each scope is frozen into copied maps; incoming sources map to every
matching ordinary node, or refuge if none matches. Candidate admission and beta
assignment produce a temporary stage. Metrics use old descriptors as W0 and
updated evidence for measurements. Strict threshold violations record a target
and stop descent. Accepted scopes stage memberships, recurse, then merge them.

After traversal the batch is retained. Disjoint targets are reconstructed from
retained target members plus their exact triggering incoming scopes, excluding
ancestor words. Replacement remaps fresh IDs and propagates sources upward.
Forgetting follows reconstruction. Final-topology observations and the report
are completed before the candidate replaces the original
([process_batch_inner](../crates/realtime-seca-core/src/engine/mod.rs#L119)).

The engine is single-owner Rust already. The impediment to simpler borrowing is
that stage methods borrow all of `self`, recursive traversal mutates it, and tree
nodes have two owners. Frozen W0 is necessary; copying every membership map is
not intrinsically necessary. Borrow an immutable scope while computing an owned
update, release that borrow, mutate, and recurse using owned child IDs/scopes.
Do not hold tree references across replacement or recursive mutation.

Snapshots clone the complete engine into a schema-4 envelope. Load validates the
envelope and all retained-state invariants. Tree exports deliberately cannot
resume processing. CLI `update` writes the tree, writes and fsyncs a temporary
snapshot, renames it, then fsyncs the parent directory. The rename is the state
commit point; this is not a multi-file transaction. An error after rename (for
example directory fsync or stdout failure) can occur after state replacement.

## Findings

### F1. Ancestry and trigger plans contain demonstrably redundant state

References: [AncestorContext and RecursiveTriggerPlan](../crates/realtime-seca-core/src/engine/trigger.rs#L237),
[child ancestry construction](../crates/realtime-seca-core/src/engine/evidence.rs#L120),
[admit_candidate_words](../crates/realtime-seca-core/src/engine/context.rs#L120),
[scoped reconstruction](../crates/realtime-seca-core/src/engine/rebuild.rs#L133).

Production starts with default ancestry. Its only augmentation inserts the same
attachment-node descriptors into `ancestor_words` and `ancestor_accepted_words`,
with counts `(0,0)`. Rejected ancestry is never populated. The three-map/count
model describes possible historical behaviours, not the actual update path.
`with_sets` exists for internal tests; tests inject rejected words and synthetic
states that no production caller creates.

Use one `BTreeSet<i32>` of excluded ancestor descriptors, retaining exactly the
current attachment-node exclusion rule. Delete the rejected-count admission path,
zero count payloads, duplicate accepted map, unused child-builder arguments, and
`with_sets`. Replace tests of impossible ancestry states with reachable exclusion
and multi-level continuation tests; retain their genuine exclusion assertions.
This is deletion of scaffolding, not approval of E05's broader level exclusion.

The plan has a boolean, target vector, scoped-index map, and three ancestry maps
that must agree. It saves ancestry even for accepted scopes, although only targets
consume it. Use a private ordered map from target ID to a small owned target
record containing incoming indexes and the exclusion set. Derive boolean and
reported target IDs from its keys; keep decision diagnostics separately in their
existing traversal order. Each HKT has one parent and is visited once in valid
state; target `contains`, union-on-repeat, and final `dedup` are redundant for this
traversal. Preserve ascending target execution order because it determines fresh
IDs. Estimated combined net deletion: **70–110 lines**.

### F2. Scope snapshots are parallel representations of one HKT

References: [snapshot_hkt_scope](../crates/realtime-seca-core/src/engine/scope_mapping.rs#L5),
[scope types](../crates/realtime-seca-core/src/engine/scope_mapping.rs#L161),
[update staging](../crates/realtime-seca-core/src/engine/context.rs#L103).

One snapshot contains separate maps of node descriptors, sources, and word-source
sets, plus node lists and child links. The key agreement is implicit; later
functions use `get`/default paths that obscure missing-node mistakes. Borrowing
`&Hkt` and its nodes during the immutable calculation would express the actual
relationship directly. The old descriptor view must survive until metrics are
finished; it need not survive into child recursion. Own only the stage and child
work that cross the mutation boundary. A fallback is a single local `Vec<Node>`
snapshot, which is still simpler than parallel maps, but retain borrowing as the
target. `MappedHktScopeState.hkt_id` is never read in production and can disappear.
Its scoped-index copy can be borrowed or moved into a target only on a trigger.

`snapshot_hkt_scope` scans every HKT to find children for every visited scope.
Diagnostics do the same for the whole final tree; restore validation scans all
HKTs for every node, and subtree collection scans all HKTs per popped HKT.
Construct one operation-local parent-node-to-children index, ordered to preserve
each caller's traversal. Validation must retain *all* children: the existing
snapshot code's single-child map overwrites if several HKTs share a parent;
restore validation does not explicitly reject that shape. Do not silently add a
new uniqueness invariant or drop reachable children while optimising validation.
Avoid a new serialized adjacency cache and another synchronisation problem.
Estimated net deletion for snapshot simplification and shared local traversal
work: **35–65 lines**, with larger allocation/traversal benefits than line savings.

### F3. Evidence conversions repeat, but two old-word counts are not equivalent

References: [mapping](../crates/realtime-seca-core/src/engine/scope_mapping.rs#L65),
[evidence](../crates/realtime-seca-core/src/engine/evidence.rs#L4),
[metrics](../crates/realtime-seca-core/src/engine/metrics.rs#L5),
[batch stats](../crates/realtime-seca-core/src/engine/batch_stats.rs#L15),
[input](../crates/realtime-seca-core/src/engine/input.rs#L3).

Token-to-ID maps are rebuilt in mapping for every scope, both token directions
are cloned in candidate admission, and the reverse direction is rebuilt again
for incoming memberships. Sources are repeatedly trimmed and deduplicated for
mapping, incoming word memberships, retained evidence, batch stats, and rebuilding.
Build a borrowed vocabulary reverse index once per traversal and source word sets
once per batch operation; keep them local. Avoid adding a persistent token cache
until evidence proves it worthwhile. Preserve vocabulary allocation order:
baseline is encounter order, newly registered update tokens are lexical order.

`build_state0_df_in_parent_scope` is run in both context and metrics for the same
scope; return/pass that result alongside the stage instead. Context also builds
old descriptor memberships by intersecting retained payloads and node membership,
while metrics reads persisted `word_source_ids`. Those are **different evidence
sources**, even if many fixtures happen to agree. Share the calculation only
where the source is identical. Replacing both with one ostensibly canonical count
would cross E01/E03 and can change accepted/restored-state behaviour.

`ComputedBatchWordStats` repeats the five public summary fields, which input
copies field by field. Return the existing summary and the DF map (a tuple is
sufficient), and use DF keys to find new tokens rather than scanning tokens again.
The DF map is read by tests and can be useful to input; it is not justification
for a new generic statistics object. In `merge_state1_updates_for_hkt`, use
`std::mem::take` to move pending maps/sets instead of collecting a cloned vector.
In metrics, the first clone of `prior` is only cloned again into `updated`; borrow
prior and allocate the union once. Preserve BTree iteration and summation order,
strict comparisons, floors and clamping. Combined estimate: **30–55 net lines**.

### F4. Hierarchy construction retains inputs globally for a local recursion

References: [build_full_tree](../crates/realtime-seca-core/src/tree/hkt_builder.rs#L44),
[create_hkt](../crates/realtime-seca-core/src/tree/hkt_builder.rs#L88),
[create_branches](../crates/realtime-seca-core/src/tree/hkt_builder.rs#L198),
[BuilderState](../crates/realtime-seca-core/src/tree/hkt_builder.rs#L441).

The builder is 530 lines. `hkt_input_scopes` saves a cloned original dataset for
every created HKT until the entire build ends; recursion clones that entry again.
The caller already has the original root input or `sorted_scope`. Carry that
original slice through recursive calls instead and remove the map. Do not pass
the post-adoption residual dataset: that would implement pending E05 accidentally.
Remove unused `current_hkt_level`, the repeated nonempty guard, and the fallback
scope clone. These have no algorithmic role.

Construct each owned HKT locally, recursively finish its children and top words,
then insert it. Use explicit local ID counters: current allocator methods derive
IDs from map lengths, so moving insertion after recursion without new counters
would reuse IDs. Preserve the exact allocation sequence, including depth-first
branch order. Build the public `nodes_by_id` mirror once from finished HKTs at
return instead of modifying both copies on every collision and resynchronising
at the end. This removes internal mirror mutation without changing the public
`HktBuildOutput` shape. It also avoids root/child HKT clones solely to satisfy
borrowing around `BuilderState`.

`records_to_map_by_source_word_id` actually keys rows by their sorted *position*,
not `record.source_word_id`. A `Vec<SourceWordRecord>` with stable `retain` gives
the same remaining row order and first entry. Replace the BTreeMap and collected
removal-key vector; inline `first_entry` and the first-word node wrapper where
that makes the flow clearer. Keep original row IDs for frequency ties and preserve
ascending expected-word-ID processing after the first word. Do not canonicalise
by token (E07). `find_best_collided_node_index`, scoped DF, and refuge calculation
remain genuine CA operations, even if each performs a scan.

Estimate: **90–145 net lines** across these builder steps, to be verified by a
prototype. Memory improvement is removal of the build-wide scope archive, beyond
unavoidable active recursion scopes and final output. Keep the public builder
validation: callers can instantiate it independently of `SecaEngine`.

### F5. Node duplication is the largest ownership problem, with a real API cost

References: [Node/Hkt](../crates/realtime-seca-core/src/tree/models.rs#L17),
[HktBuildOutput](../crates/realtime-seca-core/src/tree/hkt_builder.rs#L6),
[apply/merge](../crates/realtime-seca-core/src/engine/context.rs#L5),
[propagate_sources_to_parent_chain](../crates/realtime-seca-core/src/engine/rebuild.rs#L6),
[forgetting](../crates/realtime-seca-core/src/engine/forgetting.rs#L3),
[restore checks](../crates/realtime-seca-core/src/engine/validation.rs#L5).

A parent lookup needs a node location, not another owned node and its membership
sets. Prefer runtime ownership in ordered `Hkt.nodes`, with an ID-to-`(hkt_id,
node_position)` location map if repeated parent lookup warrants it. That locator
changes only when topology changes; memberships mutate one object. Child links
can remain operation-local. There is no need for `Rc<RefCell<Node>>`, unsafe
self-references, graph traits, or a custom generational arena.

However `tree` is a public module, `Node`/`Hkt`/`HktBuildOutput` have public fields,
and the engine derives public Serialize/Deserialize. Removing `nodes_by_id` or
changing `Hkt.nodes` directly breaks Rust clients and schema-4 state. Retain public
construction output and snapshot shape. A private engine hierarchy can store
nodes once; boundary serialization must emit the existing mirror, and decoding
must compare both supplied copies **before** constructing the runtime hierarchy.
Do not regenerate the mirror on input and thereby accept stale-cache corruption.
Direct engine serde is also public, not only `snapshot()`.

This is the only substantial representation split worth considering here: a
fixed existing wire contract and one simpler private owner. It is not free.
Budget approximately **40–80 removed mutation/remapping lines**, offset by perhaps
**60–120 lines** of concrete boundary conversion/serde and locator maintenance.
These figures are uncertain and excluded from the first-programme savings.
Reject the implementation if it grows into a generic storage layer. Even with a
small line increase, removing one full node payload copy from every engine clone
can justify it after measurement. Builder-local F4 should land first and delivers
some benefit without this compatibility cost.

Pending node maps (`source_ids_new_from_batches`,
`word_source_ids_new_from_batches`) also live in public serialized Node, although
successful state requires both empty. Moving them to recursion-local stages is
plausible, but acceptance/child/merge ordering must remain identical. Do not merge
early merely because it is easier to borrow. Keeping the empty fields at the
boundary may be required; delete their runtime use only in a separate measured
slice. First take the low-risk `mem::take` improvement in F3.

### F6. The diagnostic clone can be removed; the observation sweep cannot

References: [display_diagnostics](../crates/realtime-seca-core/src/engine/diagnostics.rs#L4),
[display_scope_recursive](../crates/realtime-seca-core/src/engine/diagnostics.rs#L81),
[observation regressions](../crates/realtime-seca-core/src/engine/tests/stream.rs#L79).

Every update clones the candidate engine again, removes incoming sources from its
retained batches and both node copies, rebuilds retained word sets, and performs
a second mapping/context/metric traversal. Baseline also pays this clone and
mapping cost even though every metric is unavailable without a comparison.

The sweep is a real public report consumer. New scopes after reconstruction have
no decision measurement, so reusing decision diagnostics would give false or
missing final observations. Trigger provenance and display coverage must remain
separate. Existing tests explicitly cover three decisions versus 260 display
scopes. Do not make observation optional on the existing `process_batch` API or
move errors past the transactional commit.

Start with the no-comparison path: enumerate final HKTs, count retained source
unions, return None metrics with the existing reason, and sort by ID. No cloned
engine or recursive mapping is needed for that path. For comparisons, pass an
excluded incoming-ID set through a small concrete borrowed calculation view,
filtering old memberships and retained payloads without copying the whole engine.
Preserve current details: incoming IDs are removed from old node evidence and
parent membership; payloads exclude them; final counts include them; mapping uses
final descriptors and incoming tokens; unavailable reasons and IDs are unchanged.
Do not instead report metrics on a pre-update tree. Filtered views may add lines;
prototype against the current function as an oracle before replacing it. This is
primarily removal of a whole-engine allocation, not an assured deletion win.

### F7. Reconstruction has small removable representations, but scope is essential

References: [build_scoped_source_word_records_for_hkt](../crates/realtime-seca-core/src/engine/rebuild.rs#L133),
[remap_subtree_ids](../crates/realtime-seca-core/src/engine/rebuild.rs#L314),
[baseline conversion](../crates/realtime-seca-core/src/engine/baseline.rs#L129).

Each target rescans retained batches, rebuilds vocabulary lookup, and clones three
ancestor maps. F1 eliminates the maps; a per-operation source lookup can remove
repeated retained scans for many disjoint targets. Preserve original retained
batch/source/token order when producing rows: iterating source IDs instead would
change tie-breaking. Do not concatenate the entire current batch; exact scoped
indexes and `(source,word)` deduplication prevent sibling leakage/double counting.

ID maps in `remap_subtree_ids` only implement `old + offset`. Mutate IDs directly,
including parent attachments, preserving the zero-root special case and current
allocator advancement. Do not add fresh-ID allocation as another builder policy.
A tiny config-to-builder constructor call can remove duplicated engine call-site
boilerplate, but it is convenience, not an architectural abstraction.
Estimated net deletion: **25–45 lines**, partially overlapping F5 if both land.

`SourceWordRecord.word` is copied into every relation but unused by the builder;
`source_word_id` and repeated DF look redundant too. All three are public fields:
word text remains an interface, row IDs govern ties, and DF is an explicit builder
input. Do not delete those fields under this mandate. Internally borrowing rows
or using compact tuples is possible only if it saves more complexity than the
conversion costs. Do not recompute caller-supplied root DF in the public builder
without treating that as a changed contract.

### F8. Persistence and identity redundancy mostly buys something

References: [snapshot/load](../crates/realtime-seca-core/src/engine/snapshotting.rs#L4),
[validation](../crates/realtime-seca-core/src/engine/validation.rs#L5),
[source registration](../crates/realtime-seca-core/src/engine/sources.rs#L16),
[forgetting](../crates/realtime-seca-core/src/engine/forgetting.rs#L3).

Keep the outer candidate clone. Validation, builder, reconstruction and reporting
can fail after earlier candidate mutations; an undo log or persistent collection
would be new infrastructure. `snapshot()` also promises an owned snapshot that
survives further updates. A borrowed serialization helper might avoid that clone
for CLI writes but must not replace the public owned method. Optimise the payload
being cloned first.

Retained tokens are not redundant with contextual word memberships: reconstruction
needs unassigned words and original source order. Source unions are not generally
derivable from descriptor memberships (refuge, mapped evidence, and propagated
sources). Expected words differ from descriptors; top words and `is_state1` are
public export data. Empty learned nodes/HKTs after forgetting are live model
knowledge, not garbage. Full memory mode remains a supported interface.

The inverse identity map and per-batch provenance duplicate information from
retained batches, but provide collision-safe admission and direct forgetting.
Keep them for now; deleting them would trade a small implementation for repeated
scans and new wire compatibility work. Keep the append-only vocabulary and
allocator counters; recomputing them from live topology would recycle IDs.

Small deletions are safe: source registration inserts the forward legend again
after `stable_source_id` already established it; forgetting removes the same
legend entry twice. The `usize` gamma conversion only feeds a zero check, while
the actual cutoff uses u32. Simplify this internal arithmetic without weakening
constructor/load checks or accepting missing gamma. `Result<Option<usize>>`
for pruning can become an internal count returning zero in Full mode; preserve
all public results. Existing exports iterate BTreeMaps and then sort by the same
keys, so those final sorts are redundant. Estimated net deletion: **15–25 lines**.
Validation itself is only 118 lines: preserve cross-checks; optimise traversal,
not the protection. Envelope config/index duplication is intentional tamper and
consistency checking under schema 4.

### F9. Some apparently dead vocabulary paths are publicly reachable

References: [synthetic candidates](../crates/realtime-seca-core/src/engine/context.rs#L128),
[synthetic rebuild words](../crates/realtime-seca-core/src/engine/rebuild.rs#L197),
[public display API](../crates/realtime-seca-core/src/engine/diagnostics.rs#L4).

Normal prepared updates register all incoming normalized tokens before mapping,
so negative synthetic word IDs and cloned per-scope token legends are unnecessary
for that input. But public `display_diagnostics(Some(batch))` does not prepare or
register the supplied batch. Its synthetic-candidate behaviour is reachable.
Also load validation does not check every retained token against the word legend;
an accepted snapshot can contain token payloads absent from that legend, reaching
the rebuild synthetic fallback. The FNV fallback similarly is not a substitute
for collision-probed registered IDs.

Do not globally delete these paths because private tests alone appear to use
them. First characterise arbitrary public display batches and accepted snapshot
continuation. Borrow the registered lookup in the canonical update while retaining
fallback handling where the existing boundary admits it. Strengthening load
validation to reject previously accepted states is a separate contract decision,
not a free structural refactor. No savings for deleting these paths are included.

### F10. CLI duplication is narrower than its historical comments imply

References: [timeline commands](../crates/realtime-seca-cli/src/timeline.rs#L15),
[pre-grouped timeline](../crates/realtime-seca-cli/src/timeline.rs#L192),
[baseline command](../crates/realtime-seca-cli/src/commands.rs#L6),
[update persistence](../crates/realtime-seca-cli/src/update.rs#L6),
[CSV ingestion](../crates/realtime-seca-cli/src/ingestion.rs#L63),
[server](../crates/realtime-seca-cli/src/server.rs#L5).

Two timeline commands repeat engine creation, first-baseline/subsequent-update,
verbose export, file naming and manifest writing. Share that execution loop with
ordinary concrete functions taking prepared batches and output paths. Keep input
preparation distinct: chunking balances one input; timeline-many preserves unequal
batches, validates before output creation, and namespaces legacy `row_N` IDs.
Even though the current converter emits `csv-row:<uuid>:...`, old JSON inputs
still exercise `row_N`; existing tests require the compatibility rule. Deleting
it changes public identity semantics. Preserve empty-input differences and both
commands' stdout/manifest values.

`commands::run_baseline_command` unconditionally clones and serializes a snapshot
even without `--snapshot-out`; do that work inside the option branch. Reuse one
verbose export when both dump and print are requested. Chunking can consume
records rather than cloning text, metadata and tokens, while keeping its public
CLI behaviour and empty trailing chunks. Repeated argument parsing is verbose,
but a new parser framework/dependency is not the simplification priority.
Estimate **35–65 net CLI lines**, additional to the core estimate, chiefly from
the common timeline loop. Avoid a universal command runner.

Core imports no filesystem, CSV, HTTP or presentation dependencies. That boundary
is sound. Update's stronger SECA-Light/matching-alpha-beta checks are application
policy, not a duplicate core algorithm. Keep atomic state writing local to the
application; do not consolidate it with baseline's ordinary export writes in a
helper that loses fsync/rename semantics. `TreeResponse(Arc<...>)` is a small
serialization bridge avoiding a whole export clone; replacing it merely to erase
a wrapper can increase copying. Leave it unless serde's existing Arc support is
already enabled and demonstrably makes the wrapper unnecessary.

## Module boundaries after deletion

Keep baseline construction, selective replacement, forgetting, snapshot/export,
restore validation, and observational reporting as recognisable operations. Their
different mutation and failure boundaries justify separate modules. Keep the
builder independent of the engine and the CLI independent of algorithm internals.

The existing stage types are misplaced: `trigger.rs` owns `HktUpdateStage`,
`WordChangeMetrics` and `PaperScopeMetrics`, which context/metrics/diagnostics
consume. Define each next to its calculation, or eliminate it if it only forwards
fields. `HktUpdateStage` has a real role (provisional evidence before a decision);
`MappedHktScopeState` has a real role (multi-node incoming assignment); neither
needs to become a trait or pipeline stage interface. The current `PaperScopeMetrics`
wrapper just carries three computed numbers: retain a small concrete value type,
remove unused parameters/dead-code suppression, and avoid fallible signatures for
pure calculations that cannot return an error. Keep public diagnostic Options:
unavailable display values are meaningful, even if decision metrics are complete.

After F3, fold `batch_stats.rs` into input preparation and consider folding the
63-line `sources.rs` there too: registration and admission are one identity
boundary, and the combined file should remain manageable. Keep metrics separate
from context so the equations remain easy to audit. Mapping should work over
borrowed nodes, and evidence functions should take the specific collections they
read rather than borrow the entire engine. Consolidate repeated loops and drop
obsolete parameter plumbing before moving files. Combining all five pipeline
modules into a 1,114-line engine implementation would obscure the algorithm and
save no implementation complexity. Module count is not the success measure.

## Prioritised, independently testable implementation slices

Net estimates below include replacement code, exclude new tests, and overlap
where stated. Implement each as a reviewable change; do not combine scientific
corrections with them.

| Order | Slice and expected reduction | Correctness risk | Required regression gate |
|---|---|---|---|
| 1 | F1 ancestry/target consolidation; 70–110 core lines | Lost ancestor exclusion, changed target order/ID allocation, tests relying on impossible private states | Multi-level ancestor exclusion, parent-trigger suppression, two disjoint triggered children, sibling preservation; exact reports and next-update tree across restore |
| 2 | F4a remove build-wide scope archive and unused recursion level; approximately 25–45 of F4's 90–145 lines | Passing residual instead of original input changes E05 behaviour | Existing overlap-recursion, refuge recursion, baseline/rebuild entry and tie goldens; compare every word/source/attachment and top_words |
| 3 | F4b local owned HKT construction, explicit counters, mirror once at return, Vec row retention; remaining approximately 65–100 F4 lines | Changed allocation timing, row order, collision order or top-word propagation | Public builder fixtures, equal-DF source-row ties, word-ID ordering, collision equality, strict branch minimum, full output mirror equality, large baseline determinism |
| 4 | F3 identical-evidence reuse, move pending data, batch-summary consolidation; 30–55 lines | Conflating retained intersections and persisted word memberships; changing floats or ID admission order | Historical candidate alpha/beta asymmetry, expected-only histories, shared-candidate overwrite, duplicate tokens, accepted parent/child sequencing, summary known/new counts, exact-threshold fixtures |
| 5 | F7 arithmetic ID remapping and F8 local persistence/identity cleanup; combined 40–70 lines before overlaps | Recycled IDs, root attachment loss, lost empty-batch window progress | Child/root replacement lineage, monotonically fresh IDs across deletion/restart, gamma 1/3 and saturating cutoff, Full mode, duplicate and forgotten/re-admitted identity, corruption rejection |
| 6 | F2 borrow scopes and index adjacency per operation; 35–65 lines | Borrow-driven reordering, missing children, loss of frozen W0 | All stage and stream tests; preserve current node-order overwrite; valid deep/wide trees, multiple children per attachment where load currently accepts them, cycles/orphans still rejected |
| 7 | F10 common timeline execution and conditional exports; 35–65 CLI lines | Changed daily grouping, legacy identities, destructive-output validation timing | Existing timeline/CSV workflows, unequal daily batches, old row_N and new namespaces, bad input produces no output, flag/output manifests; state write failure paths unchanged |
| 8 | F6a baseline observation fast path; small unbudgeted deletion | Wrong reason, count or final ID order | Full display result equality on ordinary/refuge/empty-evidence topology and no model mutation |
| 9 | F5 private single-owner engine hierarchy with unchanged public/wire types; line-neutral or modest growth | Schema/API regression; decoder hides corrupt mirrors; stale node locations after replacement | Compile public builder/struct consumers; schema-4 and direct-engine JSON round trips; stale/extra mirrors rejected; all update, prune and replacement invariants; old/new readers continue identically |
| 10 | F6b borrowed filtered comparison view; no promised line saving | Incoming counted twice, incorrect final topology, changed unavailable cases, observation error after commit | Full diagnostic equality against old implementation after rebuilding/pruning, arbitrary public incoming batches, three-decisions/260-scopes case, ensure reports cannot affect model decisions |
| Later | F5 pending maps local to recursion, only if visibility proof is straightforward | Early merge changes child context or ancestor propagation | Inspect memberships before/after each recursion phase in focused tests; exact continuation and empty pending maps after success/failure |

Slices 1–6 total approximately 265–445 lines arithmetically; overlapping snapshot,
ancestry and mirror edits make **250–420** a more credible combined budget.
Do not count F5's removed mirrors twice, or claim any of these estimates as a
completed reduction. Measure serialized size, allocation counts/peak memory and
baseline/update latency separately on deep/wide and sparse-trigger fixtures before
and after ownership/diagnostic work. Runtime speedups cannot be inferred from LOC.

## Regression strategy and approval boundaries

Existing coverage provides useful executable preservation contracts:

- `engine/tests/{context,metrics,mapping,reconstruction,stream,forgetting,snapshots}.rs`
  covers stages, current formulas, ordering and reporting. Private test APIs should
  shrink with deleted scaffolding; meaningful behavioural assertions must survive.
- `tests/seca_light_stream.rs` covers restart equivalence, selective child rebuild,
  sibling preservation, duplicates, empty windows and failure isolation.
- `tests/snapshot_validation.rs` and `tests/seca_light_invariants.rs` corrupt wire
  state and check retained boundaries,
  vocabulary references, provenance and continuation; keep rejection assertions.
- `tests/parity_context_metrics.rs`, `tests/parity_hkt_reconstruction.rs`, and the
  scientific observation suites lock current discrepancies and order dependence.
  Two strict C# parity checks are intentionally ignored. Do not enable them by
  changing semantics or regenerate observations to make refactors green.
- `tests/large_dataset_regression.rs` and CLI `tests/csv_workflow.rs` exercise size
  and ingestion persistence. Historical names such as `process_batch_skeleton.rs`
  and `paper_scaffold` do not mean their current assertions are dead code.

For structural patches, compare serialized resumable state, public exports,
reports (including order, reasons and None values), and the *next* update after
JSON round-trip, not just immediate topology. Where a private representation
changes, compare the old wire shape. Add a late-stage failure regression (after
accepted-parent mutation, reconstruction work, or diagnostics setup) because a
sequence-validation failure alone cannot prove transactional rollback. CLI fault
coverage should distinguish pre-rename failure from post-rename errors; an already
committed snapshot must remain loadable. There is no need to invent a production
fault-injection framework for this audit.

Safe structural work includes replacing positional maps with stable vectors,
removing proven duplicate ancestry, moving rather than cloning owned temporary
sets, borrowing vocabulary, retaining one local construction scope, and deriving
public report flags from one target collection. These must preserve existing
results even when the current science is questionable.

Scientific changes requiring approval include unique-word denominator/DF or
shared-word beta reduction (E01), changing W0 or expectation persistence (E02/E03),
using retained candidate beta history (E03), excluding all containing-HKT words
from descendants (E05), admitting equality at the branch minimum (E06), token/DF
ordering or exact-rational collision decisions (E07), or changing empty-scope,
gamma, floor, precision, refuge, threshold or deduplication semantics. A smaller
hierarchy resulting from E05 is not a structural optimisation. Review the linked
pending decisions and update the scientific specification before such changes.

Public/wire changes have a separate approval boundary even when scientifically
neutral: deleting `SourceWordRecord` fields, `Node` pending fields, output
`is_state1`/top_words, schema envelope duplication, the legacy row_N ingestion
rule, or any previously accepted load shape cannot be bundled as harmless cleanup.
This plan preserves them. No removed selector, full/hybrid fallback, archived
subtree mechanism, tombstone model, or pluggable policy framework remains in the
inspected production path to delete; do not invent a replacement for absent code.

## Audit verification

The local clone initially lacked the requested merge commit. The GitHub connector
provided its signed commit data; the exact commit object was recovered and
verified by its Git object hash. Its tree is
`2aa07b71dbec9dd2a0692b3e597d711749424ca5`, identical to the locally available
`074f400` parent. The isolated worktree is checked out at the requested merge
commit, not a substituted parent. No remote refs were moved.

Executed in the isolated worktree with Rust 1.92.0 and the unchanged lockfile:

- `cargo fmt --all -- --check`: passed.
- `cargo test --locked --offline --workspace`: passed; 187 tests passed, zero
  failed, two known strict-parity assertions ignored. Includes large-data,
  scientific observation, CLI CSV, stream and snapshot corruption suites.
- `cargo clippy --locked --offline --workspace --all-targets -- -D warnings`:
  passed.
- `git diff --check`: passed. All 49 local document links resolve and source line
  anchors are within the referenced files.
- Tracked production files, configuration, lockfile and test fixtures are identical
  to the base. The only worktree addition is this document.

This report makes static complexity estimates; no implementation prototype,
performance benchmark, new scientific oracle, or external C# replay was performed.
