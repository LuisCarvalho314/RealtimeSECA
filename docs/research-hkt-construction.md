# Scientific HKT construction contract: E05, E06, E07

Status: research recommendation, **not an approved production contract** (2026-10-13).
Production construction and reconstruction are unchanged. No compatibility mode,
alternate builder, reconstruction strategy, or configuration was introduced.

## Evidence and limits

Read the executable audit in [parity-hkt-reconstruction.md](parity-hkt-reconstruction.md),
[algorithm.md](algorithm.md), [architecture.md](architecture.md), and [state.md](state.md).
Requested `docs/parity-executable-summary.md` was absent from this checkout; no
replacement summary was assumed. Investigation started at `0176dfe` with unrelated
uncommitted integration changes in validation, invariant tests and these three
contract documents, plus untracked reference/harness directories. They were preserved.
Measurements describe this supplied checkout, not clean main.

Primary scientific sources inspected:

- Abdul Aziz and Starkey, [original CA construction paper](https://www.researchgate.net/publication/337879072_Predicting_Supervise_Machine_Learning_Performances_for_Sentiment_Analysis_Using_Contextual-Based_Approaches),
  DOI 10.1109/ACCESS.2019.2958702, III-A, steps 1–5 and branch paragraph,
  pp. 17725–17726. It ranks words by source frequency, compares successive potential
  words with existing nodes, and describes branching using remaining words absent
  from the parent HKT. The branch paragraph tests sources with remaining words for
  sufficiency; it supplies no numeric branch minimum or equality operator. Earlier
  strict inequalities concern word strength/similarity, not this integer cutoff.
- Al Sulaimani and Starkey, [2023 SECA accepted manuscript](https://www.researchgate.net/publication/375503068_Real-Time_Event_Detection_Using_Self-Evolving_Contextual_Analysis_SECA_Approach),
  V-A/B, Figure 1, VII-A. It identifies the prominent word by highest source count,
  constructs children from remaining parent-context words, and reconstructs selected
  HKTs and descendants. It leaves sibling exclusion, equal-frequency order and
  collision ties unspecified. Its wording about the predecessor node is less
  specific than the original construction paper about exclusion at the HKT level.
- [2021 CA classification paper](https://www.researchgate.net/publication/356012935_Short_Text_Classification_Using_Contextual_Analysis),
  Figure 4 and associated discussion: children express other important words in
  parent-node sources; recurring tokens can express different contexts. This is
  evidence against a global once-only vocabulary rule, not a definition of sibling
  reuse in overlapping branches.

The original C# `HKT.cs`, `Node.cs`, `SECAAlgorithm.cs` and the existing serial
harness were inspected. `HKT.addNode/removeUsedWord/addOtherNodes` removes adopted
word rows; `createBranch` passes the resulting local dictionary into recursion.
`SECAAlgorithm.build` and `reConstructHKT`, however, pass the **original copied
scope** to the first branch expansion. Rust `create_hkt` saves the pre-construction
scope, `create_branches` excludes only the individual attachment node's words,
and candidate traversal uses ascending vocabulary IDs. Collision comparison uses
strict improvement, preserving the first maximum. See fixture provenance hashes.

C# evidence is actual execution of the existing instrumented serial reference,
run twice with byte-identical stdout per fixture. Collection behavior belongs to
that Mono harness; race-dependent Windows execution is not reproduced. Rust uses
the public engine, Full memory, matching parameters, and validates snapshot
restoration after every cycle. Token/source identity is preserved; incidental IDs
are omitted from normalized topology. Ordered nodes and source memberships are
retained. Expected-word diagnostics are excluded only from topology comparisons;
raw observations still contain them. No published case-study reproduction or
performance claim is made.

## E05 — recursive scope

Exact fixture: [overlap_recursive/input.json](../crates/realtime-seca-core/tests/fixtures/scientific_hkt/overlap_recursive/input.json).
Parameters: alpha .8, beta .75, minimum 1; all error thresholds 1.
Baseline `1:a b; 2:a b; 3:a b c; 4:a b c; 5:a c; 6:a c; 7:a`;
updates `8:a b c`, then `9:a b c`.

C# baseline: `a:{1..7}/[b:{1,2,3,4}; c:{3,4,5,6}]`.
Rust baseline: `a:{1..7}/[b:{1,2,3,4}/[c:{3,4}]; c:{3,4,5,6}/[b:{3,4}]]`.
Each update adds its shared source to both b/c nodes and Rust's reciprocal descendants.
Minimum 1 with two shared sources isolates this difference from E06.

Scientific interpretation: a descendant's **sources** must refine its attachment
node's sources; its available vocabulary must refine the parent's available
vocabulary after level descriptors are removed. A child's node descriptors need
not be a subset of its parent's descriptors: that would prevent new subtopics.
The original CA HKT-level exclusion provides the stronger rationale for removing
all descriptors adopted at the current HKT before constructing any children.
Sibling-adopted words therefore cannot reappear immediately below another sibling.
A word may still appear in distinct contexts on branches whose ancestors never
adopted it. This exclusion is contextual, not global across the tree.

In this fixture the reciprocal branches encode the same intersection {3,4} twice.
They are redundant as descriptors of the level's already represented topics, but
are **not devoid of information**: they explicitly expose b–c co-occurrence.
Pruning sacrifices that explicit intersection view. Source sets and underlying
source-word evidence preserve it for later analysis; excluding branch rows must
never delete retained evidence. A stronger requirement to enumerate every
co-occurrence is not established by the published CA representation.

Consequences: baseline shrinks from four HKTs/five nodes to two HKTs/three nodes,
reducing redundant inspection scopes. Selective reconstruction changes the
replacement's depth and subsequent possible trigger paths. Critically, simply
matching C# recursive mutation fails consistency:

- `overlap_recursive_rebuilt` triggers the root; C# still prunes reciprocal branches
  below a, while Rust retains them (the extra q refuge also exposes E06).
- `overlap_child_rebuilt` starts with the seven overlap sources plus `8..14:d`.
  Update `15..18:a x; 19..25:a` uses error thresholds 1/1/.2. Both engines trigger
  **only root/0**, preserve the d sibling, and replace a's child with b, c, x.
  **C# now introduces both reciprocal descendants**, matching Rust's descendants.
  Its first expansion uses the unconsumed reconstruction scope. Follow-up
  `26:a b c` maps to both reciprocal branches in both engines without another trigger.
  Raw C# checks verify the d sibling retains its node object and ID.

Recommendation: consume all actual non-refuge node descriptors at every HKT,
including Seed-HKT and the root of any rebuilt subtree. Intersect the residual
rows with each attachment node's source membership, recompute scoped DF, then
construct children. Preserve source payloads and memberships. Use the same rule
regardless of where a subtree enters construction. Ancestor descriptor exclusion
remains mandatory in reconstruction; never resurrect consumed ancestor vocabulary
merely because retained payloads still contain it.

Confidence: medium-high for level exclusion from original CA wording; high for
executed discrepancies and the reconstruction exception. The later SECA wording
admits a parent-node-only reading; author clarification or independent case-study
reproduction remains desirable. Whether accepted-update candidate admission must
also exclude sibling descriptors needs a separate lifecycle investigation before
claiming this invariant holds throughout evolution. This report specifies construction.

## E06 — minimum branch boundary

Exact fixture: [branch_boundary/input.json](../crates/realtime-seca-core/tests/fixtures/scientific_hkt/branch_boundary/input.json).
Alpha .7, beta .5, minimum 2; errors 1/1/1.
Baseline `1:a b; 2:a`; updates `3:a b`, `4:a`.
C#: `a:{1,2}/[b:{1}]`; Rust: `a:{1,2}`. C# updates b to {1,3};
Rust has no child to inspect. Neither accepted update constructs the absent branch
when parent membership later exceeds the minimum.

Controlled sweep uses minimum **3** and source counts 2, 3, 4. Minimum 3 avoids
merging a and b at count 1, which would confound the original minimum-2 sweep.
Baseline rows: `1:a b`, followed by `2..n:a`; alpha .7, beta .5.
Reconstruction rows: baseline `1:q`; incoming `2:a b`, followed by `3..n+1:a`;
errors 1/1/.1 trigger the root in both languages. Follow-up is `n+2:a`.

| Parent source count | C# baseline/rebuilt b child | Rust baseline/rebuilt b child |
| --- | --- | --- |
| minimum − 1 = 2 | absent | absent |
| minimum = 3 | present | absent |
| minimum + 1 = 4 | present | present |

Scientific interpretation: a value named a minimum sufficient source count
naturally admits equality. The published branch description establishes
sufficiency, but **does not prove an inclusive integer operator**. Its count is
sources with remaining words, whereas both implementations gate on the whole
node's source union. Here only one source has b even when the parent has 3 or 4:
this distinction is an additional unresolved methodological issue, not evidence
for changing the gate's denominator in this narrow correction.

Recommendation: `node.source_count >= minimum`, with nonempty residual evidence
required, consistently at baseline and reconstruction. Treat the existing setting
as minimum parent-context support, explicitly documenting this operational
interpretation after review. Do not silently relabel it a residual-source minimum.
Topology gains children at equality; later updates gain inspection/trigger paths
and reconstructed descendants. No retroactive child creation on accepted updates
is proposed. Confidence: high for measurements, medium for inclusive recommendation;
low for claiming the publication specifies the numeric boundary or counted set.

## E07 — ordering and collision ties

Exact [collision_true_tie/input.json](../crates/realtime-seca-core/tests/fixtures/scientific_hkt/collision_true_tie/input.json):
alpha 1, beta .5, minimum 99, errors 1/1/1.
`1:a; 2:b; 3:a c; 4:b c`; updates `5:a b c`, `6:a b c`.
DF(a)=DF(b)=DF(c)=2. After creating a and b, c intersects each at 1/2.
C#: `[a:{1,3}; b,c:{2,3,4}]`; Rust: `[a,c:{1,3,4}; b:{2,4}]`.
The choice changes memberships, not just IDs or order. Reversing both document
and token order with IDs fixed makes both execute `[b,c:{2,3,4}; a:{1,3}]`:
C# encounters c before the simultaneous collision exists. A last-maximum rule
alone cannot explain or stabilize this reversal.

Exact [frequency_vs_vocabulary/input.json](../crates/realtime-seca-core/tests/fixtures/scientific_hkt/frequency_vs_vocabulary/input.json):
alpha .5, beta 1, minimum 99, errors 1/1/1.
`1:a b; 2:a b; 3:b; 4:z; 5:z; 6:z; 7:z`; updates `8:z`, `9:a b`.
DF(z)=4, DF(b)=3, DF(a)=2. C#: `[z:{4,5,6,7}; b:{1,2,3}; a:{1,2}]`;
Rust: `[z:{4,5,6,7}; a,b:{1,2,3}]`. Rust creates a before b;
b then scores 2/2 and merges, whereas a against b scores 2/3 and stays separate.
Reversal makes Rust select z,b,a and agree with C#. Root-reconstruction variants
prepend `1:q`, shift later source IDs by one, and use WI threshold .1;
both trigger root, reproducing these differing node choices after rebuilding.
Reversing rebuilt inputs leaves Rust unchanged in these two fixtures: retained
history is reconstructed in internal source order and vocabulary IDs persist.
This stability in a particular reconstruction is not baseline order invariance.

Recommended portable deterministic contract:

1. Count distinct sources per token **within the current available scope**. Rank
   tokens by descending DF; use this ranking for prominence and candidate order.
2. Resolve equal DF by ascending exact UTF-8 token bytes (case sensitive). Input
   is already normalized; no locale collation, fresh normalization or ID ordering.
3. Visit each strength-admitted candidate once in that order. Scan existing nodes
   in creation order; evaluate the current source unions, including prior merges.
4. Choose the greatest eligible intersection/node-source-count ratio. Resolve
   exactly equal ratios to the earliest-created node. Compare rational scores by
   cross multiplication with overflow-safe arithmetic; do not use epsilon ties.
   Preserve existing beta admission semantics pending a separate numeric audit.
5. Construct the first node from the first ranked token, and on no collision
   construct the **candidate being evaluated**, never an unrelated first row.
6. At each recursive scope recompute DF after filtering residual words and sources,
   rerank with the same token key, and traverse attachment nodes in creation order.
   Put refuge last. Source-row ordering may be canonicalized by token rank and
   exact external source-ID bytes but must not decide word or node ties.

This chooses Rust's first collision maximum once construction order is canonical;
there is no scientific reason to favor C#'s last enumerated maximum. Frequency
ranking is supported by original CA; byte ordering and first maximum are explicit
engineering decisions supplying missing determinism, not published scientific facts.

Hand-derived proposed outputs: true tie becomes `[a,c:{1,3,4}; b:{2,4}]` in
both permutations; frequency fixture becomes `[z:{4,5,6,7}; b:{1,2,3}; a:{1,2}]`;
overlap becomes `a/[b;c]` without reciprocal descendants. These predictions have
**not** been executed through a modified builder. No alternate implementation was
created. Greedy construction is path dependent, so ordering changes can alter
future source routing, eligibility denominators, error values and selective targets.
Confidence: high for the need for scoped frequency traversal and measured input
sensitivity; medium for the proposed tie contract, which the papers do not prescribe.

## Prioritized minimal correction proposals (review required)

1. E07 frequency candidate traversal plus canonical token tie key and explicit
   first maximum: removes allocation/collection dependencies before evaluating
   any downstream topology correction. Reuse this ordering at every construction
   entry and recursion; do not change stored vocabulary IDs.
2. E05 residual level vocabulary in the existing builder: exclude all adopted
   descriptors on every expansion, without C#'s first-expansion exception. Confirm
   accepted-update/reconstruction eligibility remains compatible with the intended
   vocabulary boundary before elevating it to a persistent state invariant.
3. E06 change only the integer comparison to inclusive, documenting parent-context
   support and the unresolved residual-source count definition.

All proposals use the canonical builder, with no engine strategy changes. Each
requires reviewed scientific/architecture documentation and focused production
regressions before implementation. Existing parity goldens must remain actual
reference evidence: new behavior may intentionally disagree with C#.

## Reproduction and executed results

Dedicated evidence: [scientific_hkt](../crates/realtime-seca-core/tests/fixtures/scientific_hkt).
Every case contains exact ordered input, raw C#, normalized C#, separately labelled
Rust observations and equality classifications. Reversal preserves IDs while
reversing documents and token arrays in every batch. Replay requires the existing
reference/harness dependencies; none are redistributed here.

```sh
# Existing harness must already be built (see its README).
python3 crates/realtime-seca-core/tests/fixtures/scientific_hkt/replay.py
cargo test -p realtime-seca-core --test scientific_hkt_contract
cargo test -p realtime-seca-core --test parity_hkt_reconstruction
# Explicitly record only scientific Rust observations after reviewed changes:
SECA_RECORD_SCIENTIFIC_HKT=1 cargo test -p realtime-seca-core --test scientific_hkt_contract scientific_construction_observations
```

Passing observation tests means the discrepancies are recorded faithfully, not
that production implements the proposed contract. Focused assertions cover the
baseline/rebuilt cutoff sweep, reciprocal depth, selective target and untouched
sibling; every Rust snapshot is restored to check state consistency.

Executed 22 fixtures / 60 snapshots per language.
Scientific tests: 4 passed. Existing HKT parity tests: 12 passed,
1 intentionally ignored strict comparison. Workspace formatting check and focused
Clippy (`-D warnings`) passed. No architectural change; full workspace tests were
not required or run.

### Input permutation measurement

Each cell gives equality per cycle, ordered topology / topology allowing sibling
permutation. The latter still compares descriptors, memberships and attachments;
it removes paths and sorts siblings recursively. Y/N are measurements of existing
engines, not of the recommended contract.

| Fixture | C# | Rust |
| --- | --- | --- |
| branch_boundary | YYY / YYY | YYY / YYY |
| collision_true_tie | NNN / YYY | NNN / NNN |
| collision_true_tie_rebuilt | YNN / YYY | YYY / YYY |
| frequency_vs_vocabulary | YYY / YYY | NNN / NNN |
| frequency_vs_vocabulary_rebuilt | YYY / YYY | YYY / YYY |
| overlap_child_rebuilt | NNN / YYY | NNN / YYY |
| overlap_recursive | NNN / YYY | NNN / YYY |
| overlap_recursive_rebuilt | YNN / YYY | YNN / YYY |

The rebuilt overlap-child comparison retains reciprocal intersections under both
permutations; reversed ordering changes their traversal order. The complete
cycle-level C#/Rust classifications are in each `comparison.json`; permutation
measurements are in `permutation-comparison.json`.
