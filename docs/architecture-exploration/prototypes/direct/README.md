# Direct immutable/set prototype

This is a standalone Python standard-library computational experiment. Construction
is a fold over a word/source relation into immutable nested `Hkt` and `Node` values.
It retains immutable ordered source/word pair tuples for tie provenance, and never
constructs mutable row objects, deletes adopted rows, allocates global IDs, maintains
a mirrored node lookup, retains builder scope caches, or clones an entire engine.
Each child receives the intersection of its parent membership and the scope relation,
excluding the parent's descriptor words. Source memberships are immutable sets.

The greedy fold remains sequential: its current union is the next collision
comparison's denominator. Replacing that fold with connected components or a
single static similarity matrix changes the result. Independence from the original
control flow eliminates bookkeeping, not this mathematical dependency.

Run from the repository root:

```sh
python3 -m unittest discover -s docs/architecture-exploration/prototypes/direct -v
python3 docs/architecture-exploration/prototypes/direct/evaluate.py
python3 docs/architecture-exploration/prototypes/direct/seca.py tools/seca-reference-harness/fixtures/stable.json
```

The JSON command builds **only the first batch**. `build(records, alpha, beta,
branch_minimum)` exposes the shared numeric source/word API for cross-prototype
experiments. `construct` and `errors` are independently callable kernels.

## Evidence and limitations

Seven tests pass, including 43 locked Rust baseline structural projections across
`parity_hkt_reconstruction` and `scientific_hkt`. The same projection agrees with
28 C# baselines and disagrees with 15; `results.json` names every outcome. These
are comparisons against existing executed observations, not freshly executed C#
parity. No reference, oracle, or fixture was modified. Structural projection
preserves ordered nodes, descriptor sets, source sets, refuge and child attachments;
it excludes IDs, `expectedWords`, presentation top words, and dynamic state.

The prototype follows current Rust's specified ascending admitted-word order,
first maximum collision, exclusive branch cutoff, and pre-adoption recursive
scope. Existing C# differences in these policies remain visible. Matching Rust
therefore does not establish C# equivalence or scientific validity of those choices.

The error kernel accepts explicitly supplied post-mapping strength, eligibility,
and importance measurements; tests exercise the stable and dominant-new-word
fractions without rounding. It does not itself implement contextual mapping,
candidate admission, measurement derivation, selective reconstruction, forgetting,
source identity, resumable snapshots, allocation, transaction recovery, or reporting.
Empty old vocabulary yields zero alpha/beta and WI=1 per literal formula; whether
such a comparison is available must be decided by its caller.

This demonstrates a materially different representation for construction, not a
complete SECA replacement. Production Rust could use nested owned values and
borrowed immutable scopes while retaining a thin allocator/index adapter. Persisting
immutable update paths instead of cloning the complete state requires a separate
transaction and lifecycle experiment. Python timings are not estimates of a Rust
rewrite. Set intersections allocate per child; overlapping hierarchies can duplicate
evidence across sibling branches. No bound on that inherent expansion was proved.

## Differentially discovered ordering defect

The first grouped-set prototype passed all 43 existing baselines but failed 38/100
shuffled seed-711 cases and several larger workloads. Group order alone lost the
first surviving row when a child excludes sources. Immutable `Scope` now retains
ordered deduplicated pairs alongside memberships, and derives child ordering from
the filtered pair sequence. This is necessary semantic provenance, despite the
extra temporary representation. A small focused counterexample and the common
runner's random-0 case guard the correction. `random0_regression.json` is a new
observation from the read-only Rust adapter, not a modified reference fixture.

After correction, the common evaluator matches Rust for all 143 cases (43 locked
baselines plus 100 seeded randomized cases) and all five measured workloads
(news prefixes 25/100/250, controlled 100/1000). This remains construction-only
evidence. The original locked-only result was insufficient to establish ordering
correctness; the defect demonstrates the value of independent random differential
inputs rather than treating existing fixture success as comprehensive parity.
