# Scientific equivalence: three contracts, not one oracle

Status: research evidence, 2026-10-26. No production formula, threshold, reference
source or historical fixture changed. Operational target is current main
`e0ca91eb62e1e5843ddb47d98ec71a388a67f6d5`; see [baseline](baseline.md).
Construction agreement with this target proves operational compatibility only.
It cannot prove the target implements the published mathematics.

## Sources and independently established claims

**A — publication.** Directly consulted the accessible full-text author manuscript
of [SECA](https://www.researchgate.net/publication/375503068_Real-Time_Event_Detection_Using_Self-Evolving_Contextual_Analysis_SECA_Approach),
DOI 10.1109/ACCESS.2023.3331219, sections V, VII-A/B/C and equations 4–12.
It describes word/source sets, node-specific eligibility, old-word error sums,
expected words in updated importance, top-down reconstruction and post-update
forgetting. It does not settle shared-word reduction, temporal expectation
classification, candidate-history admission, or deterministic ordering.
The accessible copy identifies itself as an accepted manuscript; final PDF
equivalence and the published case-study outputs have not been independently verified.

Directly consulted the original
[CA construction paper](https://www.researchgate.net/publication/337879072_Predicting_Supervise_Machine_Learning_Performances_for_Sentiment_Analysis_Using_Contextual-Based_Approaches),
DOI 10.1109/ACCESS.2019.2958702, III-A, pp.17725–17726. Its sequential
frequency-based construction and remaining-level-word branch description support
frequency traversal and level exclusion. It discusses sufficient residual sources
without specifying an integer cutoff operator. Its older strict strength
inequality is not a reason to replace SECA's later admission rule. Portable ties
remain an engineering choice.

**B — legacy.** Inspected original `HKT.cs`, `Node.cs`, `SECAAlgorithm.cs` in the
preserved original checkout's `references/seca-csharp`, and captured serial harness
observations in `scientific_hkt`, `scientific_context`, and
`parity_hkt_reconstruction`. The original desktop's parallel collection behavior
is not interchangeable with deterministic serial Mono captures. In `HKT.cs`,
`createBranch` uses `>=`; `addOtherNodes` chooses the last equal maximum and
creates `sourceWordList.First()` on noncollision while removing the inspected
word. The first branch expansion receives an original scope, deeper expansions
receive mutated rows. These are observed implementation choices, not equations.

**C — operational.** Read [algorithm](../../algorithm.md),
[architecture](../../architecture.md), [state](../../state.md), builder,
`engine/metrics.rs`, evidence, context, reconstruction and forgetting paths.
Rust chooses first maximum, ascending vocabulary-ID candidates after a
frequency-first seed, attachment-only exclusion, strict branch cutoff,
descriptor-only W0, contextual descriptor counts and refreshed candidate history.
The existing pending proposals are not approved contracts.

## Comparison boundaries and evidence strength

The historical 43-case result compares **baseline only**: ordered descriptors,
source memberships, refuge and recursive attachment. It excludes public IDs,
expected words, top words, per-word memberships and all later updates. The new
native Rust adapter strengthens construction comparison separately; it does not
retroactively strengthen historical C# evidence. Historical `*_rebuilt` names
still describe a baseline projection when counted among these 43.

[discrepancy_ledger.json](scientific/discrepancy_ledger.json) contains literal
ordered inputs and both captured structures for **every one of the 15** mismatches.
It is generated read-only from the locked captures, without importing any builder.
Exactly 28/43 projections agree. Those 15 mismatches are repeated manifestations
of five mechanisms, not fifteen independent scientific defects:

| Captured fixture (group prefix omitted where repeated) | Rule below | C# versus Rust baseline difference |
|---|---|---|
| parity_hkt_reconstruction/branch_boundary | D1 | b child at equality versus absent |
| parity_hkt_reconstruction/child_reconstruct | D2 | d children f,e versus e,f only |
| parity_hkt_reconstruction/collision_true_tie | D3 | c joins b versus a |
| parity_hkt_reconstruction/frequency_vs_vocabulary | D4 | z;b;a versus z;a,b |
| parity_hkt_reconstruction/overlap_recursive | D5 | a/[b;c] versus reciprocal descendants |
| parity_hkt_reconstruction/siblings | D2 | d children f,e versus e,f only |
| scientific_hkt/boundary_3_baseline | D1 | b child at equality versus absent |
| scientific_hkt/branch_boundary | D1 | b child at equality versus absent |
| scientific_hkt/branch_boundary_reversed | D1 | same under reversed records |
| scientific_hkt/collision_true_tie | D3 | c joins b versus a |
| scientific_hkt/frequency_vs_vocabulary | D4 | z;b;a versus z;a,b |
| scientific_hkt/overlap_child_rebuilt | D5 | reciprocal b/c descendants only in Rust |
| scientific_hkt/overlap_child_rebuilt_reversed | D5 | same with child order c,b |
| scientific_hkt/overlap_recursive | D5 | reciprocal b/c descendants only in Rust |
| scientific_hkt/overlap_recursive_reversed | D5 | same with child order c,b |

## Minimal construction counterexamples and recommendations

Notation: each numbered item is one distinct source, words are token sets;
`a/[b;c]` means a's child HKT has separate b,c nodes. All examples are literal
reproducers already present in historical fixtures unless labelled calculation.
Thresholds not listed are disabled at 1; branch minimum99 suppresses recursion.

**D1 — minimum boundary.** Input `1:ab, 2:a`, alpha .7, beta .5, minimum2.
DF(a)=2, DF(b)=1, so only a is root descriptor; residual b exists. C# creates
`a:{1,2}/b:{1}`; Rust creates only a. Independent calculation isolates `2>=2`
versus `2>2`. This is an **underspecified choice**, not a publication-proved Rust
bug. Recommendation: review inclusive minimum with explicit parent-union wording;
also resolve whether the gate should count residual sources (here 1) rather than
all parent sources (2). Preserve strict production comparison pending approval.
The minimum3 below/equal/above sweep and rebuilt sweeps in the prior scientific
fixtures isolate this from admission effects.

**D2 — equal-frequency enumeration.** Minimal ordering witness: two disjoint
words `1:e,2:f`, same DF, beta1. The first encountered row seeds the first node.
The larger siblings fixture supplies equal-frequency e/f child scopes: C# f,e,
Rust e,f with identical descriptors/memberships after ignoring sibling order.
No mathematical rule orders these ties. Classification: **underspecified choice**.
Recommendation: preserve operational order for replacement validation; separately
review canonical token-byte ties. Do not call these two ordering-only mismatches
clustering or membership disagreements. They can still affect IDs and later ties.

**D3 — collision maximum tie.** `1:a,2:b,3:ac,4:bc`, alpha1, beta.5.
After a and b exist, c scores exactly 1/2 against each. C# produces
`a:{1,3};bc:{2,3,4}`; Rust `ac:{1,3,4};b:{2,4}`. Independent rational scores
cannot choose between equal maxima. Classification: **underspecified choice**.
Recommendation: explicit earliest-created maximum is coherent and deterministic;
no scientific rationale establishes C#'s last-enumerated maximum as mandatory.
Input permutation is not a valid general topology invariant for this greedy rule.

**D4 — traversal and noncollision selection.** `1:ab,2:ab,3:b,4:z,5:z,6:z,7:z`,
alpha.5, beta1. Scoped DF(z,b,a)=(4,3,2). C# produces z;b;a, Rust z;a,b.
After z, Rust's ID order creates a then merges b since 2/2=1. Frequency order
creates b then rejects a since 2/3<1. The CA text favors frequency traversal:
classify Rust's ID traversal as an **unresolved scientific question**,
with semantic correction requiring approval and lifecycle tests. C# noncollision
creation of an unrelated first row while removing the inspected candidate is a
separate **implementation bug** whenever those diverge. Captured serial frequency
order usually masks it; these captures do not establish a standalone runtime
failure for that branch. Recommendation: always create the inspected candidate;
review frequency traversal independently, retaining Rust's coherent selection.

**D5 — recursive scope.** `1:ab,2:ab,3:abc,4:abc,5:ac,6:ac,7:a`, alpha.8,
beta.75, minimum1. Root a, next level b sources{1,2,3,4}, c{3,4,5,6}; overlap
is exactly{3,4}. C# baseline is a/[b;c]. Rust also constructs c below b and b
below c with that overlap. Independent intersection proves what the extra nodes
encode; it cannot decide the intended vocabulary exclusion. Original CA favors
level exclusion; later SECA predecessor-node wording admits attachment exclusion.
Classification: **unresolved scientific interpretation**, with a **legacy
inconsistency**: selective child reconstruction can reintroduce reciprocal nodes
because its entry scope differs from deeper recursion. Recommendation: one
reviewed exclusion rule at every entry; do not reproduce C# mutation accidents.
Retain raw evidence even if descriptors are excluded. Published support does not
justify globally forbidding a word from appearing in different contexts.

**D6 — refuge and multiple membership.** Construction witness `a×3,b×2`,
alpha.8,beta.5,minimum1: both engines construct a and refuge{b sources}, with b
below refuge. Refuge is a complement of the union of ordinary memberships, not
an exclusive partition. Update witness baseline `1:a,2:d`, then `3:ad`:
both ordinary nodes receive source3 (legacy captured multiple-match case).
This follows independent union/intersection calculations and SECA's overlapping
node representation. The captured `parity_hkt_reconstruction/refuge` update supplies source6:b to
baseline a:{1,2,3},refuge:{4,5}, with a b-child under refuge. C# leaves refuge
at{4,5} while its child becomes{4,5,6}; Rust commits{4,5,6} to both. Independent
complement routing gives source6 to refuge and the source union is{4,5,6}.
Rust's committed refuge union is therefore a **justified correction** to dropped
legacy evidence, with a concrete parent/child membership inconsistency in C#.  Recommendation: preserve multi-match
routing and refuge evidence; do not infer document partitions from hierarchy shape.
Construction parity does not establish legacy refuge-update correctness.

**D7 — descriptors versus construction expectations.** `1:ab,2:ab` gives a,b
in one node, while Rust construction expected set retains b after removing seed a;
C#'s captured `single_root` baseline has `expectedWords:[]`, while Rust retains
expected b. Both have the same descriptor node{a,b} and per-word sources{1,2}.
Thus legacy expected bookkeeping is observably different even in this two-source
case; descriptors are the same. The words used to build
a node are descriptors regardless of a leftover candidate cache. Classification:
**underspecified software choice**, not a topology disagreement. Recommendation: compare
exact expected fields against Rust where compatibility is claimed, but never
identify a construction cache with the paper's persistent dynamic expectations.

## Independent update calculations and unresolved questions

[exact_checks.py](scientific/exact_checks.py) uses Python integer sets and
`fractions.Fraction` only. It calls neither implementation nor prior prototypes.
[exact_results.json](scientific/exact_results.json) records exact outputs. These
checks establish arithmetic consequences of competing definitions, not approval
of a new scientific definition. Existing executed scientific-context captures
provide the labelled C#/Rust observations below.

**U1 — W0 expectations.** Baseline `a×2`, then distinct `ax,x` sources per batch,
alpha.6,beta.9. After k=1..4 updates D(a)=2+k,D(x)=2k. Descriptor-only W0={a}
gives WI=2/5,1/2,6/11,4/7 (Rust captures). C# gives .4,.3333,0,.5714 because
its expectation count becomes stale, disappears, then is reinserted. Including x
in W0 from the next cycle gives WI=0, but equation8 supplies no adopting node for
unassigned x; giving it zero eligibility would add beta error .9/2=.45.
At WI threshold.4, Rust reconstructs cycle3 and C# does not. Classification:
C# stale evidence is a **bug**; W0 temporal interpretation is **unresolved**.
Recommendation: retain descriptor-only W0 operationally; obtain a scientific rule
for unassigned expected eligibility before changing it. Legacy staleness proves
nothing about that rule.

**U2 — shared descriptors and importance.** Independent stable-state example:
node1={1,2,3,4,5,6}, descriptors{a,x}; node2={4,5,6,7}, descriptors{b,x};
a occurs in node1, b in node2, x globally{1,4,7}. Node-local x counts are2,2.
Unique scoped-word denominator is6+4+3=13 and unchanged W0={a,b,x} yields WI0.
Rust's incidence denominator6+4+2+2=14 and last-node x count2 yield WI1/7.
At alpha.45 scope x strength3/6 passes; local x2/6 gives mean deficit7/180.
At beta.6, x eligibilities are1/3 and1/2: worst-node reduction gives4/45;
last-node overwrite gives1/30. Reversing node order changes the latter.
This is **unresolved science with a demonstrated set/multiset inconsistency**.
No public captured Rust lifecycle installs this stable shared-descriptor state;
it is an independent mathematical witness, not a claimed reachable engine trace.
C# shared candidate fixture instead throws duplicate-key before decision output.
Recommendation: specify one vocabulary counting rule and separate (node,word)
eligibility; worst-node reduction is a proposal, not an equation in the paper.
Do not silently patch denominator alone or label a C# exception a numeric result.

**U3 — contextual DF scope.** node1={1,2},descriptors{a,x}; node2={3,4,5},
descriptors{b}; x globally{1,3,4}. Scope DF(x)=3, adopted-membership DF(x)=1.
At alpha.5, scope strength passes but adopted-only mean alpha deficit is1/18.
Rust derives old descriptors contextually and candidates from parent scope;
C# code also distinguishes these scopes. Neither is a proved unique reading of
paper Sources(w). Classification: **unresolved source-scope definition**.
Recommendation: review scope DF versus adopted-union DF explicitly; retain
per-node intersections regardless of chosen aggregate.

**U4 — candidate beta numerator and denominator.** Baseline `ax×2,a×2`, alpha.6,
beta.5; next `ax×2`, then three singleton `ax` batches. x starts below alpha.
Accumulated (a,x) counts become(6,4),(7,5),(8,6),(9,7). Rust and C# use incoming
candidate overlap:2/6,1/7,1/8,1/9, all failing beta; history intersections give
4/6,5/7,6/8,7/9, all passing. Rust WI captures2/5,5/12,3/7,7/16;
C# .4,0,.4286,.4 also reflects stale expectations. With WI threshold.41,
history-based **persistent** adoption would avoid Rust's cycle3 trigger. Merely
changing numerator cannot claim that result because current accepted candidates
remain expectations rather than new descriptors. Classification: **unresolved
admission lifecycle**, historical symmetry is compelling but not independently
proved by the paper's candidate wording. Recommendation: review history and
persistence jointly, with multi-batch/forgetting tests.
Separately, the node_union captured case has overlap2,node-union6,prominent5:
2/6 fails beta.4, C# candidate2/5 passes. Equation8 supports node-union denominator;
Rust's denominator is a **justified correction**, not proof of its numerator.

**U5 — successive candidate evidence.** `shared_asymmetric` baseline a×3,b×2;
then ax×3,bx×2; then ax,bx yields Rust WI1/3,7/19; C# duplicate-key failure in
first update. Candidate x counts5 then7 arise from distinct retained-plus-new
sources, independent of transient node assignment. In `expected_history` these
refresh every cycle. Classification: refreshed accumulated evidence is a
**justified correction** to stale C# bookkeeping. Whether all historical
unassigned words should be rediscovered absent current-batch occurrence is a
separate unresolved discovery policy; the captures do not prove completeness.

**U6 — precision.** Independent unchanged old count30000 plus new candidate count1
gives WI1/30001. At threshold3/100000 the exact equation triggers, four-decimal
rounding produces0 and accepts. C# rounds errors before comparison; Rust uses
full precision. Classification: **justified correction** to follow equations,
with intentional legacy incompatibility. This tiny witness checks arithmetic,
not executable admission on a 30001-source fixture. Strict `>` error triggering
and inclusive admission remain separate operators.

**U7 — forgetting boundary and empty evidence.** Publication wording does not
uniquely fix whether current-gamma is retained; Rust retains exactly gamma batches,
forgetting after evolution, and empty batches advance expiry. Serial harness
rejects empty batches, so no cross-language empty-batch result exists. Empty W0
zero alpha/beta and denominator floors are operational totalizations, not paper
proofs. Classification: **underspecified boundary/software contracts**.
Recommendation: preserve tested Rust retention, rollback and restart behavior;
seek explicit scientific wording before a boundary change.

## What is established and what is not

Verified independently: distinct-source set arithmetic; asymmetric greedy
path-dependence; equality of collision scores; differences caused by node-union
versus prominent denominator; history versus incoming numerator; shared-word
mass inconsistency under an incidence denominator with unique-word aggregation;
and exact consequences of expectation and rounding alternatives.

Behaviourally established by preserved observations: the 15 baseline discrepancies
and labelled scientific update sequences. Newly executed equation tests do not
constitute fresh C# execution. The parent investigation records any current Rust
or harness replays separately. Stable shared descriptors, published case-study
reproduction, final-manuscript equivalence, candidate rediscovery completeness
and a uniquely mandated shared-word reduction remain unproved.

Keep three test families distinct: (1) mathematical set/equation invariants,
(2) frozen legacy observations, including expected failures, (3) current ordered
Rust transition/identity contracts. Do not weaken a legacy fixture to certify an
alternative or promote a pending proposal into an invariant. No production
semantic change is justified solely by matching either computational prototype.

## Reproduction

```sh
python3 docs/architecture-exploration/iteration-2/scientific/exact_checks.py
```

Executed: **14 passed**, no failures. This regenerates only iteration-2 JSON.
Its ledger assertion requires exactly15 preserved baseline mismatches; no locked
capture is written. Scientific review and native implementation evaluation are
separate deliverables; full lifecycle coverage belongs in
[evolution_validation.md](evolution_validation.md).
