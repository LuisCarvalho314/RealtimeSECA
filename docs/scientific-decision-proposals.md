# Scientific decision proposals

Status: proposals for researcher acceptance; no scientific contract or algorithm
is changed by this document. Reviewed on 2026-10-15 from fetched `origin/main`
`01a644f3a7411bd4de4ad172555153fbcf525cc2`, in the isolated branch
`audit/seca-scientific-decisions`. Only this document is a deliverable.

The key recommendations are unique-source aggregation within each HKT (E01),
removal of all containing-HKT descriptors before descendant construction (E05),
an inclusive branch minimum (E06), and an explicit frequency-first deterministic
construction order (E07). Shared-word beta reduction remains a scientific choice;
a concrete conservative proposal is supplied below, not attributed to the paper.

## Evidence and reading boundaries

**Published evidence.** [S23: 2023 SECA accepted manuscript](https://www.researchgate.net/publication/375503068_Real-Time_Event_Detection_Using_Self-Evolving_Contextual_Analysis_SECA_Approach),
[DOI 10.1109/ACCESS.2023.3331219](https://doi.org/10.1109/ACCESS.2023.3331219),
sections V, VII-A/B/C, Figures 1–3 and equations 1–12 were read directly. The
accessible text is the accepted author version, not an independently checked
final typeset version. [CA20: Aziz and Starkey, IEEE Access 8, 17722–17733](https://www.researchgate.net/publication/337879072_Predicting_Supervise_Machine_Learning_Performances_for_Sentiment_Analysis_Using_Contextual-Based_Approaches),
[DOI 10.1109/ACCESS.2019.2958702](https://doi.org/10.1109/ACCESS.2019.2958702),
section III-A, pp. 17725–17726, steps 1–5 and the following branch paragraph,
provides an accessible original CA construction specification. The 2020 thesis
cited as S23 reference 47 was not obtained; this report does not claim to have
read it. The university S23 publication landing page was accessible, but was not
used as a substitute for algorithm text.

**Repository evidence.** Read [algorithm.md](algorithm.md),
[consolidated evidence register](parity-executable-summary.md),
[context audit](parity-context-metrics.md),
[construction audit](parity-hkt-reconstruction.md), and architecture/state contracts.
The consolidated register supplies E01–E13 identities. Historic claims in the
individual audits are superseded by their executable sections and consolidation.

C# references below identify original files beneath
`references/seca-csharp/SECA/SECA/SECA/SECA_DataClass/` in the supplied local
reference archive; it is intentionally not redistributed. Rust links identify
tracked source at the base commit. C# execution means the existing serial,
instrumented harness, not the original concurrent desktop application. Archive
revision and compiled preprocessing dependency provenance remain unknown, as
recorded in the consolidation. No C# source, harness, production Rust, defaults
or existing golden fixtures were edited.

CM denotes [context fixtures](../crates/realtime-seca-core/tests/fixtures/parity_context_metrics/);
HKT denotes [construction fixtures](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/).
All named fixture evidence below was replayed in this work without capture or
regeneration. New examples explicitly labelled **independent** are mathematical
or structural oracles, not new successful C# runs or public-API Rust observations.
Confidence measures scientific support, not just certainty that implementations
differ. “High” does not mean a researcher has accepted the proposal.

## Decision table

| Issue | Paper interpretation | C# behaviour | Rust behaviour | Recommendation | Confidence | Required tests |
|---|---|---|---|---|---|---|
| E01 | S23 (5),(11),(12) use word frequencies; (8) is node-specific | Per-node counts; duplicate metric key throws | Per-node denominator; last word metric overwrites | Unique sources per word per HKT for alpha/WI; retain node beta evidence; propose worst-node deficit per old word | High: reject overwrite/double count; medium: scope; low: beta reduction | A1–A5 |
| E02 | S23 (6),(9),(12): old words; (11): old plus expectations | Old descriptors only | Old descriptors only | Keep W0 fixed until reconstruction; candidates remain outside it across accepted updates | Medium | A4, C2 |
| E03/E04 | S23 (3),(8): word/node source intersection | Incoming candidate overlap / prominent sources | Incoming candidate overlap / node union | Keep node union; propose retained plus incoming overlap for beta, consistent with alpha | Medium for history; high for union | C1–C3 |
| E05 | CA20 branch paragraph excludes words already appearing in containing HKT | Recursive child uses residual rows; seed/rebuild entry differs | Reuses pre-construction rows; excludes only attachment-node words | Exclude all containing-HKT descriptors at every construction entry; preserve independent sibling contexts | Medium-high | S1–S4 |
| E06 | CA20 has a sufficiency threshold; no explicit integer comparator | `>= minimum` | `> minimum` | Inclusive minimum with nonempty residual evidence; retain current total-node count interpretation pending residual-count decision | Medium for inclusive convention; low for published count basis | B1–B3 |
| E07 | CA20 starts with highest, then second-highest frequency; exact ties unspecified | Hash traversal / last maximum; noncollision can create another word | Expected IDs ascending / first maximum; record-order frequency ties | Descending scoped DF then token bytes; first-created node on collision tie; canonical relation order | High: frequency priority; medium: deterministic conventions | O1–O4 |
| E08 | No rounding requirement in S23 equations | Four-decimal errors before trigger | Full precision, strict `>` | Preserve precision; correct misleading reason formatting separately | High | P1 |
| E09 | Accumulated evidence in S23 VII-C | Stale existing expectation count | Recomputes retained plus incoming | Preserve Rust | High under E02 | C2 |
| E10 | S23 V-B refuge represents unmatched sources | Stale accepted refuge parent | Persists unmatched memberships | Preserve Rust | High | R1 |
| E11 | S23 (7) uses node sources | State0 uses prominent sources | No equivalent public State0 metric | C# defect; do not port; no demonstrated Rust defect | High | P2 |
| E12 | No specified object IDs or stored candidate layout | Dynamic candidates; runtime sibling order | Construction candidates persisted; stable IDs/order | Incidental when labelled attachments and continuation agree; isolate E07 semantic ordering | High for IDs; medium for candidate storage | O4, C2 |
| E13 | S23 mapping supplies new evidence; empty evaluation unspecified | Extra empty-child inspections; harness rejects empty batches | Skips empty scopes | Preserve skip; empty/zero-evidence reconstruction interpretation unresolved | Medium | Z1 |

## E01 — frequency and node eligibility need different identities

**Source rule.** S23 equations (5) and (11) normalize word-source counts;
(8) intersects word and node sources, and (12) sums old-word importance.
Equations (6)/(9) average old-word deficits. None specifies how multiple nodes
owning the same word collapse into one beta measurement. [S23](https://www.researchgate.net/publication/375503068_Real-Time_Event_Detection_Using_Self-Evolving_Contextual_Analysis_SECA_Approach).

**Established implementation facts.** `HKT.cs:393` sums descriptor counts by node
and adds admitted expected counts; `Node.cs:180` computes contextual measurements;
`HKT.cs:287` merges with `Dictionary.Add`. CM `shared_candidate` reproduces a
duplicate-key exception with no decision JSON. Rust
[metrics.rs:8](../crates/realtime-seca-core/src/engine/metrics.rs#L8) sums each
node's descriptor count, adds candidates once, then `result.insert(word, ...)`
overwrites earlier measurements; W0 is a word-ID union. This is deterministic
for a given ID order, not invariant under equivalent node reorderings. CM's
shared-candidate test verifies existing overwrite behavior, not a correct
shared-descriptor oracle. A normal baseline processes each descriptor once at
one HKT; repeated descriptors in one container are not established as reachable
through these successful engine fixtures. Candidate assignment to several nodes
is reachable. The recommendation must handle that case and explicitly define
any accepted/restored repeated-descriptor state, without claiming a new reachable
production failure.

**Independent counterexample A.** Consider an already represented HKT with old
nodes `{a,x}` and `{b,x}`. Node sources are respectively `{1,2,3,4}` and
`{3,4,5,6,7,8}`; a appears on the first set, b on the second, and x on
`{1,3,4,5,6}`. Contextual x counts are 3 and 4, but its unique HKT count is 5.
There are no admitted new words; W0 is `{a,b,x}`. This is a measurement input,
not a claim the current constructor creates this topology.

| Calculation | x from first node wins | x from second node wins | Proposed unique HKT counts |
|---|---:|---:|---:|
| Importance denominator | 4+3+6+4=17 | 17 | 4+6+5=15 |
| Sum of old importances | 13/17 | 14/17 | 1 |
| WI error | 4/17 | 3/17 | 0 |
| x strength, prominent count 6 | 3/6 | 4/6 | 5/6 |
| Alpha error, alpha=.7 | 7/90 | 1/45 | 1/90 |
| x node eligibility | 3/4 | 4/6 | Both retained separately |
| Beta error, beta=.75 | 0 | 1/36 | 1/36 under proposed reduction |

Thus at WI threshold .2, other error thresholds 1, exchanging node processing
order alone flips reconstruction: 4/17>.2 but 3/17<.2. Positive WI with no new
word is an accounting artifact. At beta threshold .02 the overwrite also hides
or exposes one node's violation. This cannot be repaired by selecting first,
last, largest, or average **importance** while retaining the mixed denominator.
An occurrence-based normalization would instead define a different observation
universe, and must change numerator, denominator and W0 together; it is not a
literal repair of the current word-based equations.

**Recommendation A, proposed equations.** Let S(H,t) be the retained evidence
eligible for that HKT at measurement time: all retained plus incoming sources
at seed; parent-adopted retained plus mapped incoming sources at a child. Include
accepted pending parent memberships before descending. Exclude ancestor-ineligible
candidate words according to the selected construction rule. Define:

```text
F(H,w,t) = {s in S(H,t): token w occurs in s}
D_H(w)   = |F(H,w,1)|
W0       = union of old ordinary-node descriptors (no refuge sentinel)
C1       = alpha-admitted candidates, excluding W0
W1       = W0 union C1
Q        = max({1} union {D_H(v): v in W1})
T        = max(1, sum(D_H(v) for v in W1))
strength(w)   = D_H(w) / Q
importance(w) = D_H(w) / T
eligibility(w,z) = |F(H,w,1) intersect Sources(z,1)| / max(1, |Sources(z,1)|)
```

Count a source once per word, not once per node; different words can each count
the same source. Recompute from retained token evidence or an equivalent complete
source-set index: merely summing counts cannot deduplicate overlaps, and merely
unioning existing node indexes can miss occurrences elsewhere in the HKT scope.
For example x on `{1,2}` in one node and `{2,3}` in another has unique count 3,
not sum 4 or maximum 2. Sources containing w but not adopted into its current
owning node still contribute to HKT-level DF under this proposal; they contribute
to eligibility only through intersection. This is a deliberate scope decision,
not an assertion that existing contextual indexes are already complete.

**Researcher decision A-beta.** Recommend, if one old-word average is retained,
`deficit(w)=max over old owning nodes z of max(0,beta-eligibility(w,z))`, then
average over unique W0. This checks every ownership rule without giving a repeated
word extra weight. It agrees with current equations for single ownership.
The paper does not distinguish this from averaging deficits over `(word,node)`
pairs or assigning one exclusive owner. Those yield different thresholds: in A,
pair averaging gives `(1/12)/4=1/48`, versus proposed `1/36`. Choosing maximum
is a conservative scientific extension; no implementation correction should
silently select it. Confidence is low for author intent, despite clear order
invariance and an explicit executable oracle.

**Independent multilevel example A-child.** H1 parent sources `{1,2}` and H2
parent sources `{2,3,4}` contain x on `{1,2,3}` globally. D_H1(x)=2 and
D_H2(x)=2, not global 3; their overlap source 2 is legitimately evidence in both
separate contexts. Never deduplicate across HKTs or pool same-depth containers.
A word may remain meaningful in independent branches; E05 only prevents reuse
of words already adopted on that branch's containing-HKT sequence.

**Required tests.** A1: independent source-set oracle for A, overlap `{1,2}` /
`{2,3}`, identical and disjoint word memberships; WI=0 without new vocabulary.
A2: reorder nodes and renumber IDs; all three errors/decisions invariant, including
thresholds .2 and .02 above. A3: separate child scopes as A-child, plus a token
outside its owning node but inside HKT scope. A4: x admitted to two nodes must
count once in W1, remain outside W0, and never panic; beta-rejected candidates
still affect T. A5: empty retained frequencies after forgetting, missing descriptor
evidence, and round-trip continuation preserve documented zero conventions,
transaction rollback and both node indexes. New oracles must be separate from
unchanged historical parity goldens; public aggregate tests alone do not observe
all temporary node measurements.

## E05 — descendants must use residual containing-HKT vocabulary

**Source rule.** CA20 III-A, p.17726, branch paragraph says “which is not
appeared in HKT 1” when describing descendant words. Its prose also refers to
removing node words, so the wording is imperfect. The containing-HKT exclusion
is stronger evidence than the earlier audit's absence of a construction source.
[CA20](https://www.researchgate.net/publication/337879072_Predicting_Supervise_Machine_Learning_Performances_for_Sentiment_Analysis_Using_Contextual-Based_Approaches).
S23 V-A alone does not settle sibling removal.

**Established behavior and important C# qualification.** Rust
[hkt_builder.rs:106](../crates/realtime-seca-core/src/tree/hkt_builder.rs#L106)
retains input before adoption; `create_branches:198` filters only the attachment
node's descriptors. C# `HKT.createBranch:691–727` calls `addNode` then recursively
passes its mutated local dictionary. However `SECAAlgorithm.build:66–82` branches
from the original seed dictionary, not the mutated local copy; `reConstructHKT:613–627`
likewise branches from the unmodified replacement input. Consequently C# does
**not** implement a uniform all-HKT exclusion rule. Copying its recursion verbatim
would leave a baseline/rebuild entry inconsistency.

**Independent structural counterexample and executed anchor.** HKT
`overlap_recursive` uses documents `ab;ab;abc;abc;ac;ac;a`, alpha=.8, beta=.75,
minimum1. Root a has DF7; b and c each have DF4, so they move to the a-child.
There b and c split: overlap2/4=.5<.75. Reusing the input below b yields c on
sources3,4; reusing it below c yields b on the same sources. Both have two sources,
so the boundary discrepancy cannot explain their existence. The executed fixture
has those Rust grandchildren and no C# grandchildren. They reinterpret two peer
concepts already consumed in one context as refinements of each other without new
residual vocabulary. That increases depth and future inspected/triggered scopes.

**Recommendation S.** For each constructed HKT H, collect adopted ordinary
words A(H); remove all their relations from its input before deriving any child
scope. For each ordinary or refuge node z, intersect that residual dataset with
z's source set, recompute DF, and recurse. Carry the exclusion through the path,
including the selected reconstruction root. Do not delete source identity or
parent membership merely because its residual word rows become empty. Preserve
words in independent branches whose containing-HKT paths have not adopted them.
This follows the stronger CA20 rule and removes structurally redundant descendants;
it is not a recommendation to remove same tokens everywhere in the tree.

The update candidate filter must use the same path exclusions if this is accepted;
otherwise an accepted child can re-admit its parent's sibling descriptor and undo
the construction decision. Current `evidence.rs:build_child_ancestor_context_from_parent_node`
collects only attachment-node descriptors. This follow-through needs a coordinated
contract change, not a second construction/update mode.

**Required tests.** S1: retain the existing overlap golden as history; a separate
scientific oracle has root a / child b,c and no grandchildren. S2: perform equivalent
construction at seed, recursive child and selected replacement root; all exclude
containing-HKT adopted words identically. Include the simpler seed scope
`b;b;bc;bc;c;c` at alpha1, beta.75, minimum1 to expose C#'s entry exception.
S3: a residual d not adopted at H still produces a child; refuge still uses its
residual member evidence; two independent child contexts may both use d.
S4: later child updates cannot re-admit excluded sibling/ancestor words; verify
replacement attachment, unchanged sibling IDs, parent source sets, mirror equality,
rollback on failure and restore continuation.

## E06 — inclusive minimum, with a separate count-basis question

**Source rule and interpretation limit.** CA20 III-A's branch paragraph checks
sufficiency of remaining sources/words. It does not provide an integer `>` versus
`>=` rule; its residual-source wording also does not prove that total parent
membership is the intended count. S23 Figure1 requires remaining words, without
settling this configurable integer boundary. Do not confuse branch minimum with
S23's inclusive alpha/beta admission equations (2)/(3).

**Established behavior.** C# `HKT.createBranch:691` uses total node sources
`>= minimum`; Rust [hkt_builder.rs:209](../crates/realtime-seca-core/src/tree/hkt_builder.rs#L209)
requires total node sources `> minimum`. Both require nonempty child records.
HKT `branch_boundary` replays `ab;a` at minimum2, alpha.7, beta.5: C# creates b
below a and Rust does not; later child inspection paths differ.

**Independent boundary table and recommendation B.** Treat a parameter named
minimum as the smallest eligible count: `|Sources(z)| >= minimum`, plus nonempty
residual records after S. At minimum2: count1 fails, count2 passes, count3 passes.
For the fixture the parent has two sources but residual b occurs on only one;
this makes the count-basis question observable rather than hidden. Recommend the
inclusive **total-node** convention as the narrowly scoped correction supported
by C# and the parameter meaning. It is a convention consistent with construction,
not a theorem from CA20. If researcher intent instead requires minimum residual
sources, the same fixture must have no child even with `>=`; that is an additional
scientific correction affecting both languages. Resolve that question explicitly
before claiming literal CA20 threshold fidelity.

Use the same builder criterion for baseline, ordinary/refuge descendants and
reconstruction. A qualifying count with no residual word must never create an
empty HKT. Minimum0 makes any nonempty residual scope eligible, without creating
empty descendants or changing the configured default.

**Required tests.** B1: counts m-1,m,m+1 for m=2, with at least m residual-bearing
sources where possible; isolate comparison from missing evidence. B2: separately
test total2/residual1 above, total2/residual0, minimum0/1, refuge and all-adopted
scopes. B3: repeat exact-boundary cases through selected root and child reconstruction,
checking next-cycle inspection paths and no empty/unreachable HKT.

## E07 — deterministic greedy construction, without hash enumeration

**Source rule.** CA20 III-A steps1–2 start with the highest, then second-highest
word frequency. Step5 compares a subsequent word against existing nodes. It does
not fix exact frequency or collision ties. [CA20](https://www.researchgate.net/publication/337879072_Predicting_Supervise_Machine_Learning_Performances_for_Sentiment_Analysis_Using_Contextual-Based_Approaches).

**Established behavior.** C# `findMyExpectedWords:946` forms a HashSet from
frequency-sorted rows; `addOtherNodes:863` takes its enumeration and the last
maximum collision. Scoped ConcurrentDictionary enumeration can alter ties.
Its noncollision path creates the first remaining row's word but removes the
inspected word: when these differ the transition is incoherent; no successful
fixture here independently proves that failure path. Rust
[hkt_builder.rs:126](../crates/realtime-seca-core/src/tree/hkt_builder.rs#L126)
uses ascending vocabulary IDs for expected words after seeding by frequency;
`:388` selects the first maximum, `:468` breaks DF ties by original record ID.
Rust creates/removes the inspected word coherently. Input permutations and
historical vocabulary IDs can therefore change its topology.

**Independent numeric/structural examples with executed anchors.** HKT
`frequency_vs_vocabulary` has `ab;ab;b;z;z;z;z`, alpha.5, beta1. Frequencies
z4,b3,a2 require z,b,a order under the proposed frequency rule. a overlaps b by
2/3<1, giving separate z;b;a nodes. Current Rust visits a before b: b overlaps
a's two sources by 2/2=1 and merges them. The executed reference yields the
separate nodes. Sorting the final output cannot undo the changed membership.

HKT `collision_true_tie` has `a;b;ac;bc`, alpha1, beta.5. All DF=2; once a and b
nodes exist, c overlaps each by1/2. The executed reference chooses b, Rust a.
Both obey beta; neither exact tie preference follows from the source. Choosing a
expands its union to3 sources, changes later collision denominators and gives c
a different future child context. HKT `siblings` also has e/f order differences
with equal labelled attachments; that observation alone is representational.

**Recommendation O (portable convention, explicitly proposed).**

1. Deduplicate source-word pairs after existing identity/normalization validation.
   Recompute scoped distinct DF once for each HKT's input. Process admitted words
   by descending DF, then ascending normalized token UTF-8 bytes. Freeze that word
   order during this HKT's construction; node source unions evolve greedily.
2. For each word compute overlap/current-node-source-count against every existing
   ordinary node. Choose the largest score meeting beta; exact ties choose the
   earliest-created node. Node creation order is derived from step1, so the tie
   key does not depend on runtime IDs. Compare ratios exactly by integer cross
   products with safe width for ordering; do not introduce epsilon ties.
3. If no node qualifies, create a node for that same inspected word. Union its
   sources and remove that word's rows. Never create one word and remove another.
4. Derive relation order by `(token UTF-8 bytes, external source-ID UTF-8 bytes)`;
   relation IDs are storage identities, not scientific priorities. Node traversal
   is creation order with refuge last. Build child scopes with S and recompute DF.
   Vocabulary IDs may remain append-only/first-seen; resolve to tokens for comparison.

This chooses token order rather than encounter/record order so a permutation of
identical documents or token lists within the **same batch** does not alter labelled
hierarchy. Batch order and source identity/deduplication rules remain substantive;
there is no proposed invariance to repartitioning a stream or changing text
normalization. Preserve asymmetric eligibility, source unions, single-node greedy
construction assignment and multi-node update mapping. This does not preserve all
existing outputs: it deliberately preserves the construction rules while replacing
accidental ordering. No parallel/hash scheduling or C# bucket order becomes policy.
First-created collision tie preserves Rust's coherent convention once the word
order is canonical; last-created has no stronger scientific support.

**Required tests.** O1: z4,b3,a2 yields separate z;b;a, with word IDs adversarially
ordered. O2: true tie yields a/c and b under lexical a,b,c; include unequal scores
and exact beta equality. O3: permute documents, tokens and record IDs; vary prior
vocabulary allocation and serialize/restore; compare labelled hierarchy, attachment,
memberships and subsequent decisions, not allocator values. Include non-ASCII tokens
and rational scores with different denominators. O4: baseline and rebuilt identical
scopes produce equivalent labelled hierarchy; preserve independent sibling identity,
refuge position, both node indexes and source-set contents. Keep historical goldens
unchanged and add explicit proposal oracles in a later implementation change.

## E02/E03/E04 — dependent vocabulary and evidence decisions

**Proposed E02 lifecycle.** Construction consumes its temporary candidates into
descriptor nodes. Let W0 at every inspection be descriptors of that current
learned HKT, fixed before mapping. Let C1 be eligible non-descriptor expectations
recomputed from evidence; beta assignment is provisional and does not promote
words into W0 on an accepted update. Reconstruction installs a new descriptor
vocabulary; the next cycle uses that vocabulary. A previously unassigned candidate
stays novel relative to this learned container. This preserves current Rust and
observable C# behavior and avoids stored construction metadata changing WI.
S23's distinction between baseline novelty and per-batch states motivates this
interpretation, but does not fully specify expectation promotion across cycles.

**Independent consequence and anchor.** CM `expected_trigger`:
`a;a -> ax;x -> ax;x`, alpha.5, beta.9, minimum1000, WI threshold.4.
At the second update counts a4,x4 imply T=8. Descriptor-only W0 gives WI=.5
and triggers; promoting old expectation x into W0 gives WI=0 and accepts.
C# actually gives 1/3 and accepts because its x count is stale2; this is E09,
not evidence supporting promotion. Rust gives .5 and reconstructs. An expectation's
cached presence alone must not settle W0. This is why E01 needs a disjoint C1
and why E09's correctness is conditional on the selected W0 lifecycle.

**Proposed E03/E04 evidence.** Use retained plus incoming candidate occurrences
within the current node's updated source set, divided by that full node union.
CM `historical_candidate` (`ax;ax;a;a -> ax;ax`, alpha.6, beta.5) has a6,x4.
Incoming-only beta2/6 fails; full-history4/6 passes. Both implementations currently
use incoming-only evidence; both count history for alpha. This asymmetry makes
admission depend on batch partitioning even with identical retained evidence.
History is scientifically preferable under the accumulated state interpretation,
but S23 does not spell out this candidate-assignment intermediate step.
CM `node_union` separately isolates denominator choice:2/5=.4 passes C# at
beta.4, while2/6 fails Rust; node sources, rather than prominent-word sources,
are the appropriate denominator under the proposed rule.

Immediate errors can remain equal: candidates affect T after alpha regardless
of beta and stay outside W0. A later change in stored candidate membership or
scope may change continuation, but neither fixture proves a later trigger effect
from beta admission alone. Do not use aggregate equality as intermediate-state
proof, or promise a topology change not executed.

**Required tests.** C1: historical4/6 versus incoming2/6; capture node assignment
with focused internal tests, also after forgetting removes the historical x rows.
C2: consecutive expectations, assignment/rejection, strength recheck/removal,
reconstruction promotion, restart and the .4 WI boundary above. C3: parent-mapped
child history only, no sibling leakage; node-union fixture and multiple-node x
admission; global HKT candidate DF and node intersection remain distinct.

## E08–E13 — classification and targeted follow-through

These classifications distinguish a C# defect from a demonstrated Rust defect.
Scientific recommendations are not requests for exact C# parity.

| Issue | Classification / counterexample / end effect | Action and required tests |
|---|---|---|
| E08 | Justified precision difference to preserve. CM alpha/beta threshold:1/12>.0833, but C# rounds to.0833 and accepts. Rust's trigger is correct under full precision. `trigger.rs:201` displays both operands as .0833: an actual Rust diagnostic defect, not a scientific error. | P1: retain strict comparison; later display enough digits to explain trigger. Test equality, adjacent thresholds, half-way decimal cases, and WI separately. S23 (6),(9),(12) do not prescribe rounding. |
| E09 | Actual C# stale-evidence defect under E02; justified Rust difference. `expected_trigger` has x4, not cached x2; WI .5 versus1/3 at threshold.4 changes reconstruction. `HKT.setExpectedWordsInHKT:514–545` only inserts missing keys. | Preserve Rust recomputation; C2 covers retained, forgotten and restarted expectations. S23 VII-C accumulated evidence supports freshness. |
| E10 | Actual C# membership defect; justified Rust difference. HKT refuge baseline `a;a;a;b;b` then b leaves C# parent refuge{4,5}, child{4,5,6}; Rust parent includes6. Immediate old-word errors can agree while ancestry is stale. | Preserve complete refuge union. R1: force later parent/child-scoped reconstruction after source6, check retained evidence, attachment, subset ancestry, mirror equality and forgetting. S23 V-B refuge definition supplies the source-set interpretation. |
| E11 | Actual C# State0 denominator defect against S23 (7); not a demonstrated Rust defect. CM node_union b has2/3 with prominent sources versus required3/4 on the baseline node union; State1 is5/6. Option1 uses State1, so this field does not cause the observed immediate decision. | P2: any State0 metric exposed later must use old node union and intersections; keep it out of Option1 decisions. No claim of executed intermediate Rust State0 parity. |
| E12 | Incidental representation where labelled memberships/attachments and continuation agree: random/monotonic IDs, f,e versus e,f in independent siblings, baseline construction sets versus dynamic candidates. Unresolved if stored candidates alter future admission. E07 already covers order that changes greedy construction. | Preserve incidental differences. O4/C2 compare subsequent decisions through restore and reconstruction before declaring candidate storage harmless. Do not change state schema solely for snapshot equality. |
| E13 | Justified operational skip for zero incoming scope; unresolved no-evidence semantics. Rust skips an empty child, C# executes extra accepted inspections in replayed cases. Harness empty input fails before the algorithm; it proves no empty-update formula. Zero retained descriptors could yield WI=1 with floored denominators, so skipping and evaluating are not universally equivalent. | Z1: Full-mode empty update preserves learned context; SlidingWindow empty batch still advances forgetting. Add an internal zero-evidence selected-plan case and verify no partial subtree removal, mirror divergence or lost parent attachment. Preserve transactional Rust behavior; researcher must define zero-evidence trigger semantics if reachable. |

## Verification performed and limits

Executed from this isolated base checkout, using the supplied local reference:

```sh
SECA_REFERENCE_ROOT=/home/azureuser/risklive-dev/experimental python3 crates/realtime-seca-core/tests/fixtures/parity_context_metrics/replay.py
SECA_REFERENCE_ROOT=/home/azureuser/risklive-dev/experimental python3 crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/reference.py
cargo test --offline -p realtime-seca-core --test parity_context_metrics --test parity_hkt_reconstruction
git diff --check
```

Context replay: 19 inputs twice; 17 successful outputs match goldens; expected
shared-candidate and empty-input failures reproduced. HKT replay:21 inputs,
65 snapshots, twice with unchanged goldens. Rust:32 passed, two known strict-parity
tests ignored by the existing suites. No claim that ignored parity checks pass.
No architecture or algorithm changed, so workspace-wide build/clippy were not
required or run. The focused tests compile the core and execute both evidence
adapters. The independent arithmetic below was checked separately with exact
rational calculations. A standalone Python set/ratio construction calculation also
verified the E05 overlap residual, E06 adjacent boundaries, E07 frequency and
collision examples, and the historical candidate fractions. These checks do not
execute the proposed algorithm in production.

```python
from fractions import Fraction as F
assert 1 - F(4+6+3,17) == F(4,17)
assert 1 - F(4+6+4,17) == F(3,17)
assert 1 - F(4+6+5,15) == 0
assert ((F(7,10)-F(4,6)) + (F(7,10)-F(3,6))) / 3 == F(7,90)
assert 2 * (F(7,10)-F(4,6)) / 3 == F(1,45)
assert (F(7,10)-F(4,6)) / 3 == F(1,90)
assert (F(3,4)-F(4,6)) / 3 == F(1,36)
assert (F(3,4)-F(4,6)) / 4 == F(1,48)
```

Acceptance of a recommendation is not an implementation: follow-up must update
`algorithm.md` and applicable state/architecture contracts, add the specified
independent regression oracles, and retain one canonical update/construction
path. Preserve immutable historical parity fixtures as evidence of the base
revision. A changed output requires scientific justification, not golden rewriting.

## Questions that require researcher judgement

1. **A / A-beta:** accept unique-source HKT-level DF for strength/importance,
   including occurrences outside a word's owning node? For beta, accept worst
   owning-node deficit per unique old word, or require a specified occurrence
   average/exclusive ownership? The available paper does not choose that reduction.
2. **S:** accept containing-HKT word exclusion at seed, every descendant, update
   candidate admission and reconstruction, despite the narrower S23 wording and
   C#'s inconsistent seed/rebuild entries?
3. **B:** accept inclusive total-node membership minimum plus nonempty residual
   evidence, or is the intended minimum the number of residual-bearing sources?
   Total2/residual1 at minimum2 separates these interpretations.
4. **O:** accept frequency priority plus lexical token ties and first-created
   collision ties as the portable convention? The exact tie policy is a decision,
   not a recoverable C# hash enumeration rule.
5. **C:** confirm descriptor-only W0 until reconstruction, and full retained plus
   incoming evidence for candidate beta? These make repeated expectations and
   candidate admission coherent but are not fully specified in S23.

These five decisions suffice to authorize narrowly scoped scientific corrections;
no further broad source-code audit is needed to choose among the stated rules.
Zero-evidence reconstruction remains a separate conditional question, not a reason
to import C# failure behavior or block decisions on nonempty contexts.
