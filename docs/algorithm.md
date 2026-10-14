# Scientific specification

Source: Al Sulaimani and Starkey, *Real-Time Event Detection Using Self-Evolving
Contextual Analysis (SECA) Approach*, IEEE Access 11 (2023), 127011–127034,
[DOI](https://doi.org/10.1109/ACCESS.2023.3331219).
Section V defines the HKT representation; VII-A/Figure 3 defines evolution;
VII-B defines change metrics; VII-C adds forgetting. Construction builds on the
Contextual Analysis work cited by the paper. The production engine implements
hierarchy evolution. The paper's direct/indirect event clustering is evaluated
separately by the [experimental benchmark](seca-paper-benchmark.md);
its reconstruction assumptions do not change the contracts below.

## Paper contract

HKTs contain word/source nodes. Child-HKTs refine parent nodes; refuge nodes
collect unmatched sources. Construction uses strength ≥ alpha and node
eligibility ≥ beta (equations 2–3). Map the entire batch at Seed-HKT and only
parent-adopted sources at child scopes. Multiple non-refuge matches are possible.

State 0 precedes an update; state 1 includes new evidence. Let D(w) count distinct
sources, q be the updated prominent word, z the word's node, W0 the old vocabulary,
and W1 the updated vocabulary including expected words:

```text
strength1(w)   = D(w) / D(q)                       (5)
eligibility1(w)= |Sources(w) ∩ Sources(z)| / D(z)  (8)
importance1(w) = D(w) / Σ(v ∈ W1) D(v)             (11)
Alpha-Error    = Σ(w ∈ W0) max(0, α-strength1(w)) / |W0| (6)
Beta-Error     = Σ(w ∈ W0) max(0, β-eligibility1(w)) / |W0| (9)
WI-Error       = 1 - Σ(w ∈ W0) importance1(w)       (12)
```

Inspect HKTs top-down. Any error exceeding its threshold reconstructs that HKT
and descendants; otherwise inspect children recursively. SECA-Light subsequently
discards outdated source evidence according to gamma.

## Repository interpretations

These choices specify this implementation; they are not additional paper rules.

- Strength and importance use retained plus incoming evidence. Incoming document
  frequency counts a token once per source. Word eligibility uses the updated
  node source union, including the node's earlier memberships.
- Existing node words define W0. Previously expected words remain candidates
  rather than becoming old node words. Candidates participate in the importance
  denominator only after strength admission; their presence alone does not mutate
  old vocabulary before evaluating errors.
- Candidate admission excludes ancestor-adopted words. Candidates are checked
  again against the updated prominent count before beta admission to nodes.
  Accepted parents expose updated memberships before descending; merging pending
  memberships follows child processing.
- A triggered subtree is built from its retained members and the exact incoming
  scope that caused the trigger. Ancestor words are excluded from its dataset.
  Parent source membership is propagated up the chain. Siblings preserve IDs
  and topology.
- Error comparisons use full precision and strict `>`; admission uses `>=`.
  Empty incoming scopes are skipped. Empty W0 produces zero alpha/beta error.
  Count denominators are floored at one when evidence is empty. Observation
  reports mark unsupported comparisons unavailable instead of asserting change.
- The retained window is exactly gamma batches including the current batch:
  indices `current - gamma + 1` through `current`, saturating at zero. Empty
  batches count toward the window. Forgetting removes source evidence, but does
  not independently remove learned words, nodes, or HKTs.
- Source identity, token normalization, ID allocation, atomic persistence, and
  configuration validation are software contracts documented in [state.md](state.md).

## Ambiguities and scientific follow-up

The paper's textual cutoff (“before” current minus gamma) and its bounded-batch
case study leave a boundary ambiguity. The exact-gamma interpretation above is
intentional and covered by tests.

The implementation inherits two choices requiring a focused scientific audit:
W0 excludes unassigned expected words, and the importance denominator aggregates
node word counts. If a word can occur in several nodes at one level, its frequency
can therefore be counted more than once. The measurement map resolves such a
word deterministically by node order. This refactor preserves these choices;
it does not claim they have been proved equivalent to the paper's set notation.

Candidate beta admission currently counts incoming word memberships against the
updated node source union. Earlier occurrences of a candidate may thus influence
scope strength without contributing to this node admission numerator. Establish
reference fixtures before changing that interpretation.

The builder sorts every input by descending document frequency, with source-word
record order breaking ties. Reconstruction previously omitted this prerequisite;
that defect is corrected rather than preserved as a scientific interpretation.

Construction tie ordering, refuge recursion, and ancestor exclusion retain the
existing builder behavior. Independent reproduction of the referenced CA
construction algorithm and published case study remains scientific validation
work. No alternative error metrics or full-rebuild fallback are supported.
