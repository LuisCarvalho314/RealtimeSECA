# Remaining C#–Rust v2 discrepancies

This inventory follows the News G first-cycle construction investigation. It
classifies **observed differences**, not proposed compatibility changes. A
*scientific* difference changes which evidence, words, nodes, or sources the
contextual model uses; it needs a model-level decision. An *engineering*
difference concerns a reproducibility rule or state/implementation defect not
specified by the publications. An engineering difference can still change
descriptors and later scientific measurements.

The baseline comparison uses the saved C# reference snapshot and a verified
first-wins Rust v2 build from commit `7fc0ba5c0df4b00f00245414de19ee9ad29fb763`.
The source batch contains 740
News G documents with alpha 0.7 and beta 0.5. The C# snapshot SHA-256 is
`9d941e7c3fdd754554888860a6f8735396c793e791b179119824a58aba521037`;
the canonical Rust cycle-0 tree SHA-256 is
`1cc62e6ab654ebb987034bc3878abda4216d2ba4859ff6467aa7d89d31f58d88`.
[Matched-HKT data](evidence/seca-v2-ordering/news-g-hkt-comparison.csv)
shows 42 matched paths, 7 ordinary descriptor partitions that differ, and one
matched path with a different observed source union. Those counts are baseline
observations; they do not represent seven independent causes.

## First-cycle matched HKT differences

| Path, abbreviated | Measured distinction | Classification and confidence |
| --- | --- | --- |
| `black / friday` | Five words all have DF 4; the traced original C# methods process `sale, cyber, monday, deal, thanksgiving`, while v2 uses lexical order. A Rust replay of the C# order matches every beta score, descriptor, and source membership. | **Engineering tie policy, proven** for the directly executed case. It changes the greedy scientific grouping. C# has no portable tie rule. |
| `kanye|west / kardashian|kim` | Same 12-source union. Rust admits sibling words `rogen,seth` (DF 5); C# admits `james,parody` (DF 3). Excluding the sibling words reduces the local maximum from 5 to 4, allowing the DF-3 words through alpha. | **Scientific scope/exclusion choice, strongly supported.** Rust v2 excludes path descriptors; the C# baseline consumes descriptors adopted at the current HKT before child construction. |
| `REFUGE / REFUGE / report` | Same 12 sources; Rust additionally adopts `nsa,porn`, sibling descriptors absent from the C# child. | **Scientific sibling-exclusion choice, strongly supported.** |
| `REFUGE / REFUGE / seahawks` | Same 10 sources; Rust adopts sibling word `year` in the `seahawks` child. | **Scientific sibling-exclusion choice, strongly supported.** |
| `REFUGE / REFUGE / year` | Same 13 sources; Rust adopts sibling word `seahawks` in the `year` child. | **Scientific sibling-exclusion choice, strongly supported.** |
| `kanye|west / franco|rogen|seth / bound|james|parody|video` | C# has 10 observed sources; Rust has those plus source 646. Descendant vocabularies differ. | **Downstream scientific scope/membership divergence.** Its independent cause is unproved; upstream descriptor and source routing already differ. |
| six-level nested refuge path | Same observed source union and descriptor vocabulary, but several words join different nodes. | **Ordering/collision tie hypothesis.** No candidate-level trace has isolated the first difference. |

The `black / friday` and nested refuge rows share vocabulary and observed scope,
but only the first has a direct C# construction trace and counterfactual Rust replay.
The sibling-word rows are instances of the same model question: whether
child construction excludes only descriptors on its ancestry path, as v2 does
(`crates/realtime-seca-v2/src/recursive.rs:69–79`), or all words adopted by the
current HKT, as in the C# baseline (`HKT.cs:661–662, 687–719`). The original CA
construction description supports level-wide remaining words; later SECA wording
also admits a parent-node reading. This remains a **scientific** decision rather
than a reason to emulate C# collection iteration.

## Later cycles and other known classes

[Cycle observations](evidence/seca-v2-ordering/news-g-cycle-by-cycle.csv) show
that first-cycle descriptor differences precede all updates. In cycle 2, both
snapshots cover the root sources, while matched paths and reconstruction decisions
already differ. By cycle 3, C# omits 297 incoming sources from its stored root
membership; all 297 appear in descendants, and closing parent memberships over
descendants removes the root refuge difference. C# records 30 triggers versus
Rust's 20. This is an **engineering hierarchy-consistency symptom with scientific
consequences**. The snapshots do not reveal the first within-update state change,
so neither the ten-trigger gap nor the 297-source gap is assigned a specific cause.
Aligned event-level scopes, matches, errors, and reconstruction flags are still
needed for that attribution.

Controlled C#–Rust fixture audits in
[parity-context-metrics.md](parity-context-metrics.md) and
[parity-hkt-reconstruction.md](parity-hkt-reconstruction.md) establish further
classes, with different applicability to v2:

| Difference | Type | Current assessment |
| --- | --- | --- |
| Candidate history, old-versus-expected vocabulary, repeated words across nodes, and beta admission source sets | **Scientific semantics** | The SECA equations do not settle every lifecycle or aggregation choice. Rust v2's `metrics.rs:37–94` uses unique scoped word sets and old descriptor vocabulary; its direct reconstruction path in `recursive.rs:120–135` differs from C#'s accepted candidate-assignment lifecycle. News G trigger differences cannot be assigned to one of these rules from snapshots alone. |
| Minimum branch support | **Scientific counted-set ambiguity** | C# tests parent node source count; v2 tests residual child scope length (`recursive.rs:72–75`). An inclusive numeric boundary alone does not settle which set is counted. |
| Equal beta scores between existing nodes | **Engineering tie choice** | C# selects the last enumerated maximum; v2 keeps the first strict maximum (`recursive.rs:33–43`). This may alter topology, and node iteration itself can depend on earlier scientific choices. No News G path in this inventory isolates such a tie. |
| C# four-decimal error rounding versus Rust full precision | **Engineering numeric policy** | Controlled alpha/beta threshold fixtures change trigger decisions. There is no published four-decimal requirement; v2 uses full precision (`recursive.rs:125–127`). |
| C# cached expected-word counts and duplicate-key failure on shared candidates | **Observed implementation behavior** | Controlled fixtures show a changed WI trigger and a surfaced exception. Their causal contribution to News G has not been established. |
| C# refuge membership not committed at an accepted parent, and output-only IDs/order | **Engineering state/representation** | In controlled fixtures, Rust retains unmatched refuge membership and C# can leave its parent stale. IDs and sibling display order are incidental only where labelled memberships and later behavior agree. |

The explicit UTF-8 tie comparator and its regression test make the current Rust
ordering portable. They intentionally leave every scientific choice in this
inventory unchanged. Further correction needs a scoped causal trace for each
remaining News G path and an approved model rule for HKT-wide exclusion,
branch support, and update metrics.
