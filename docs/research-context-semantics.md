# Scientific context and metrics contract investigation

2026-10-13. Research only; no production algorithm or canonical specification
changed. Owned fixtures: `crates/realtime-seca-core/tests/fixtures/scientific_context/`.
Owned characterization: `scientific_context_contract.rs`. Branch:
`research/seca-context-semantics`.

## Evidence and limits

The requested `docs/parity-executable-summary.md` was absent. Read the executable
section of [parity-context-metrics.md](parity-context-metrics.md), its golden inputs,
outputs and shared-candidate stderr instead. Baseline checkout has pre-existing
integration edits; results characterize that working tree, not clean main.
`provenance.json` fingerprints the measured production and harness files;
`csharp-source-hashes.json` fingerprints compiled reference inputs. Existing
shared fixtures were neither recaptured nor edited.

Paper: Al Sulaimani and Starkey, IEEE Access 11, 127011–127034,
[DOI](https://doi.org/10.1109/ACCESS.2023.3331219), verified against the accessible
[author manuscript](https://www.researchgate.net/publication/375503068_Real-Time_Event_Detection_Using_Self-Evolving_Contextual_Analysis_SECA_Approach).
The institutional final-version page returned 401; final-PDF equivalence is not
claimed. Relevant source locations are Section V/Figure 1, VII-A/Figure 3,
VII-B equations 4–12, VII-C.

Source findings (paraphrased): expected words satisfy alpha; beta examines node
source intersections. Equations 5 and 11 use word source counts; equation 8 uses
node source counts. Equations 6, 9 and 12 sum over state-0 words. Equation 11's
state-1 vocabulary includes expected words. VII-C removes outdated evidence after
updating. None specifies how shared words collapse across nodes or whether an
unassigned expectation becomes an old word next cycle. Historical evidence is
consistent with the accumulated model, but candidate admission timing remains
underspecified. These findings support the proposal below without proving its
extensions uniquely mandated by the paper.

C# observations come from the existing serial, instrumented harness, with original
formulas and pre-decision snapshots. Rust observations come from public decision
diagnostics and exported topology. Temporary Rust assignment maps are unavailable:
claims about them are source inspection plus arithmetic, not executed intermediate
parity. Both executions use accumulated memory; sliding-window semantics are
specified below but not cross-language tested here.

## Proposed canonical source-set specification

This section is an explicit proposed contract, not a production requirement.
Let R be retained sources available at inspection and B the incoming HKT scope.
For the root the scope is the full batch; a child receives only sources adopted
by its parent node. Let P_h be retained sources in that same parent scope (root:
all retained sources). Token identity is canonical and document membership is a
set. Ancestor-adopted words are excluded from candidate consideration.

For each ordinary node z, let T_z^0 be its descriptor set and N_z^0 its retained
source membership. Freeze these before any assignment. Let M_z(B) be incoming
sources mapped to z using old descriptors, and define:

```text
S_h^1 = P_h ∪ B
F_h^1(w) = {s ∈ S_h^1 : w occurs in s}
N_z^1 = N_z^0 ∪ M_z(B)
J_z^1(w) = F_h^1(w) ∩ N_z^1
T0 = ⋃ordinary z T_z^0                  (unique descriptor vocabulary)
O0 = {(z,w) : w ∈ T_z^0}               (descriptor incidences)
D_h^1(w) = |F_h^1(w)|                  (distinct scoped sources, once per word)
q = argmax_w D_h^1(w)                  (among eligible old/candidate words)
C1 = {w outside T0 and ancestors : D_h^1(w)/D_h^1(q) ≥ α}
W0 = T0
W1 = T0 ∪ C1
strength1(w) = D_h^1(w)/D_h^1(q)
eligibility1(z,w) = |J_z^1(w)|/|N_z^1|
importance1(w) = D_h^1(w)/Σv∈W1 D_h^1(v)
A = Σw∈W0 max(0,α-strength1(w)) / |W0|
B_error = Σw∈W0 maxz:(z,w)∈O0 max(0,β-eligibility1(z,w)) / |W0|
I = 1 - Σw∈W0 importance1(w)
assignment(z,w) ⇔ w ∈ C1 and eligibility1(z,w) ≥ β
```

Candidate discovery must include retained, currently unassigned scoped tokens,
not just previously persisted expected IDs or words appearing in this batch.
Compute q from old descriptors plus all non-ancestor scoped candidate tokens,
then apply alpha; a maximum-count token necessarily passes alpha. Rechecking an
old expected key uses fresh sets, never a cached count. Retaining an expected key
is bookkeeping, not membership in W0. A below-alpha expected token contributes
nothing to W1; old descriptors remain in W1 even if they now fail alpha.

Persist a beta assignment into the accepted node descriptor set and its full
retained-plus-incoming word membership; it enters W0 on the next update. On a
rejected update, reconstruct from the same scoped evidence. This persistence is a
necessary part of this proposal: a transient assignment that never becomes a
node descriptor cannot deliver the proposed future behavior. Refuge markers have
no word metric. Sets prevent duplicate source or vocabulary contributions.

Empty W0 gives A=B_error=0. If the importance total is zero, use I=0 when W1=W0
and I=1 when positive new-word evidence exists; the latter has a positive total
by construction. Zero node evidence gives eligibility=0; zero prominent evidence
gives strength=0. Skip empty incoming scopes as the existing lifecycle specifies.
After successful update, apply the existing retention window; next inspection
intersects historical sets with surviving evidence. Learned descriptors survive
forgetting, so their zero support can be detected on a later nonempty inspection.

Global HKT DF is selected over union-of-descriptor-membership DF. The latter,
`|⋃z:w∈T_z J_z(w)|`, deduplicates sources but misses a word in a sibling/refuge
source where it was not adopted as a descriptor. This creates representation-
dependent strength and importance. Scope DF treats old and candidate words alike.
It still preserves contextuality through the parent scope and node intersection.

Worst-node beta deficit is a deliberate extension: every adoption must satisfy
its own node rule; a strong adoption cannot hide a weak adoption. It retains one
contribution per word and equation 9's unique-word normalization. Incidence mean,
per-word mean and best-node eligibility are alternatives, not equivalent formulas.
A dedicated `(node,word)` map is needed regardless of which extension is chosen.

## Vocabulary classification and recommendation

| Category at inspection | W0 | W1 | Node eligibility |
|---|---|---|---|
| Existing ordinary descriptor | yes, once | yes, once | each old adoption |
| Newly alpha-admitted candidate | no | yes | every candidate node |
| Beta-assigned during this update | no | yes | retained plus incoming |
| Persisted expected but unassigned | no | only if freshly alpha-admitted | admission only |
| Historical, currently unassigned token | no | only if freshly alpha-admitted | admission only |
| Accepted beta assignment from earlier cycle | yes | yes | old adoption |

E01: use scoped distinct word DF, unique vocabularies and importance denominator;
keep node eligibility separate. **High confidence** in deduplication and separation;
**medium confidence** in full-parent-scope DF rather than adopted-membership union;
**medium-low confidence** in worst-node beta reduction. The paper leaves the last
two choices unresolved; author clarification would increase confidence.

E02: retain descriptor-only W0. **Medium confidence.** Including expectations is a
plausible temporal reading of equation 11, but equation 8 supplies no node for an
unassigned word. Giving it eligibility zero, a best hypothetical node or skipping
it each invents a further rule. Descriptor-only W0 preserves a coherent common
old vocabulary across all three errors. This is a recommendation, not a proof
that paper terminology excludes expectations across cycles.

E03: candidate beta should use retained plus incoming intersections, recomputed on
each update, and accepted assignment must persist. **Medium-high confidence.**
Incoming-only numerator over an accumulated denominator penalizes small batches
and disagrees with the same eligibility rule used for existing words.

E04: keep full node-union denominator. **High confidence.** Equation 8 and its
node-specific interpretation support it; no new evidence warrants substituting
prominent-word sources. The existing `node_union` golden is a useful independent
check: incoming x overlap2, node union6, prominent support5; beta=.4 rejects under
2/6 and admits under C#'s 2/5. Immediate WI=.2 is unchanged, but persistent adoption
would affect later W0 and mapping.

## Independently calculated datasets and executed results

All rows below are literal ordered fixture documents; `×k` means k distinct IDs.
Thresholds are 1 unless explicitly stated. Branch minimum1000 suppresses children.
All observed alpha/beta errors are zero; only WI is shown. Raw full C# observations,
Rust diagnostics/topology and failure stderr are retained alongside each input.

| Fixture | Baseline; subsequent batches | Exact Rust WI, cycles2 onward | Actual C# WI | Topology |
|---|---|---|---|---|
| shared_asymmetric | a×3,b×2; ax×3,bx×2; ax,bx | 1/3,7/19 | duplicate-key failure in cycle2 | Rust keeps a,b nodes |
| expected_history | a×2; ax,x repeated four times | 2/5,1/2,6/11,4/7 | .4,.3333,0,.5714 | both keep a only |
| expected_trigger | a×2; ax,x twice; threshold .4 | 2/5,1/2 | .4,.3333 | Rust reconstructs cycle3 into a and x nodes; C# keeps a |
| historical_beta | ax×2,a×2; ax×2; ax; ax; ax | 2/5,5/12,3/7,7/16 | .4,0,.4286,.4 | both keep a only |
| historical_beta_trigger | same; threshold .39 | 2/5,0,0,0 | .4,0,0,0 | both reconstruct cycle2 into one a,x node |

Expected history arithmetic: at cycles2–5, D(a)=3,4,5,6 and D(x)=2,4,6,8.
Thus I=D(x)/(D(a)+D(x)). Incoming overlap is one each cycle; accumulated node
membership is3,4,5,6. Even full-history x overlap1,2,3,4 never passes beta=.9.
Expectation-inclusive W0 would instead give I=0 from cycle3, and, assigning
unassigned x eligibility zero, B_error=.9/2=.45. With beta threshold1 and WI
threshold.4 that alternative avoids the Rust cycle3 reconstruction. Its missing
node rule is consequential, not cosmetic.

C# stores x count2 through cycle3, removes x at cycle4 and reinserts it at cycle5;
its per-word importance/count captures demonstrate this. These observations
cannot establish a valid alternative W0 contract. Agreement again at cycle5 is
coincidental (reinserted historical/current count happens to be8).

Historical beta arithmetic: baseline x2/a4=.5<alpha=.6, so x is not a descriptor.
Cycle2 has a6,x4; subsequent counts are (7,5),(8,6),(9,7). Incoming beta is
2/6,1/7,1/8,1/9; historical beta is4/6,5/7,6/8,7/9. All historical values pass.5;
all incoming values fail. Under the proposal, cycle2 assigns x; cycle3 W0={a,x},
I=0 and both strength and eligibility remain above thresholds through cycle5.
At WI threshold .41, incoming-only reconstructs cycle3 (5/12>.41), while the
proposal accepts every update. This is an independently calculated counterfactual,
not a modified-production execution. With threshold .39 both actual engines
reconstruct immediately, and the resulting a,x node persists through three later
updates: the fixture proves adoption's future effect once reconstruction installs
it. Production currently keeps accepted candidates as expectations, including
those that transiently pass beta; persistence needs explicit authorization.

Shared asymmetric execution: cycle2 has a6,b4,x5 and node-local x3,x2;
both local incoming beta ratios equal .5. Actual I=5/15 measures only old a,b,
so it does not expose the temporary x measurement selected by node iteration.
Cycle3 gives a7,b5,x7 and I=7/19. C# fails before emitting any decision JSON;
there is no C# decision oracle for shared candidates. Accepted Rust topology
still contains only a,b, so this API cannot install a stable shared descriptor.
The following stable-state case is independently calculated, not injected into
production or claimed as cross-language execution.

## Asymmetric stable shared-descriptor counterexample

The independent `shared_sets.json` supplies node source sets z1={1,2,3,4,5,6},
z2={4,5,6,7}; descriptors {a,x} and {b,x}; a sources=z1, b sources=z2,
x sources={1,4,7}. All seven sources belong to this root scope. Thus node-local
x counts are2,2, global x DF3, descriptor counts a6,b4,x2,x2.

For W0={a,b,x}, the canonical denominator13 gives importances6/13,4/13,3/13
and I=0. The current multiset denominator14 with last-node x2 gives
I=1-(6+4+2)/14=1/7. A WI threshold .1 therefore accepts the proposal and rejects
the multiset/overwrite alternative, despite no new vocabulary. Counting source4
twice instead gives x4 and denominator14, I=0 but changes strength and importance.
A descriptor-incidence vocabulary could also force I=0, at the cost of giving
shared words multiple votes in alpha and beta.

At alpha=.45, scoped x strength3/6=.5 passes; last-node x strength2/6 fails,
yielding A=(.45-1/3)/3=7/180. Alpha threshold .02 separates decisions when
other thresholds are1. At beta=.6, x eligibility is1/3 in z1 and1/2 in z2.
Worst-node per-word B_error=4/45; per-word mean B_error=11/180; last-node
B_error=1/30. Beta threshold .07 separates worst-node from both alternatives.
Reversing node order changes last-node beta error to4/45. These are asymmetric
eligibilities even though both local counts happen to be2.

A second scope test distinguishes scope DF from adopted-membership union:
z1={1,2}, z2={3,4,5}, descriptors {a,x},{b}; a={1,2}, b={3,4,5},
x={1,3,4}. Scope D(x)=3, adopted-union D(x)=1. At alpha=.5, scoped x passes,
adopted-union x fails with A=(.5-1/3)/3=1/18. Threshold .04 separates decisions.
This supports selecting a single source scope before selecting metrics.

## Minimal proposed implementation change (not implemented)

1. Build scoped word source sets once from retained and incoming evidence; count
   each distinct vocabulary word once. Use those sets for q, strength, importance
   and candidate admission. Preserve separate node intersections.
2. Store eligibility by (node,word); reduce beta deficits explicitly by the chosen
   worst-node rule. Freeze descriptor union W0 before candidates are assigned.
3. Replace incoming-only candidate intersection with retained-plus-incoming
   intersection. Persist accepted assignments and their historical memberships;
   re-evaluate historical unassigned vocabulary under fresh alpha.

No compatibility mode, fallback reconstruction, new public diagnostics or
production changes are needed for this proposal. Items1–3 require a separate
scientific approval and production regression suite, particularly child scopes,
forgetting, ancestor exclusion and atomic failure invariants. Author confirmation
of shared-word reduction and expected-word persistence remains desirable.

## Reproduction and validation

```sh
python3 crates/realtime-seca-core/tests/fixtures/scientific_context/replay.py
cargo test --offline -p realtime-seca-core --test scientific_context_contract
# Explicit capture of new Rust evidence only:
CAPTURE_SCIENTIFIC_CONTEXT=1 cargo test --offline -p realtime-seca-core --test scientific_context_contract
```

The C# replay verifies each successful capture twice and the shared failure's
exit status, empty stdout and duplicate-key class twice. Stack addresses in stderr
are not deterministic. Rust replay checks complete captured diagnostics and
exported topology for all five sequences. Independent source-set tests verify
union/occurrence differences and historical beta across four consecutive cycles.
No unresolved scientific alternative is asserted as a production requirement.

Completed focused checks: scientific context tests **4 passed**; existing context
parity tests **20 passed, 1 intentional precision-discrepancy test ignored**;
C# harness verification passed; all five owned C# cases verified twice (including
the expected shared-key failure). Owned Rust test passes rustfmt check and the
working tree passes `git diff --check`. No architecture changed, so workspace
architectural checks are outside this investigation's scope.

A supplementary `cargo test --offline -p realtime-seca-core` run passed all
74 unit tests and three large-dataset tests. It was interrupted while the two
large-dataset stream tests were still running; the full-suite result is therefore
**incomplete**, not a pass. The completed focused suites above are the validation
for these research-only additions.
