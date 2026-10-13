# Executable C#–Rust HKT and reconstruction audit

## Current evidence (2026-10-13)

This section supersedes the historical source-only audit retained below. The existing
reference harness was available, compiled, and executed. **Full behavioural parity
is not established.** Twenty-one deterministic fixtures (65 snapshots) compare
actual C# execution with the public Rust engine in `MemoryMode::Full`. Golden C#
outputs are never manufactured from Rust. No production code was changed.

The audit started at HEAD `9e70e2136dea237f1e922352df0ae24d4441c2fc` on
`integration/seca-validation`, with unrelated uncommitted integration changes.
Local `main` is `42c07c098c1fadb931a4e7b5b6920ee9561dabc5`; consequently this is
an audit of the supplied integration checkout, **not a clean-main measurement**.
Those changes were preserved. Work is committed on `agent2/executable-hkt-parity`.

Scientific reference: [2023 SECA accepted author manuscript](https://www.researchgate.net/publication/375503068_Real-Time_Event_Detection_Using_Self-Evolving_Contextual_Analysis_SECA_Approach),
sections V and VII-A, Figures 1–3, equations 2–3. The university final-publication
endpoint returned HTTP 401. The manuscript describes word/source nodes, child
attachments, refuge for sources with no sibling-word match, strength/eligibility
admission, and top-down reconstruction. Seed-HKT receives all incoming sources;
children receive parent-adopted sources. Triggering reconstructs the selected HKT
and descendants, rather than continuing inspection below it. It does not establish
a portable collection tie order, a branch-size cutoff, or whether sibling words
may recur below overlapping nodes. Those scientific questions remain unresolved.

## C#–Rust implementation map

C# algorithm paths are under `references/seca-csharp/SECA/SECA/SECA/SECA_DataClass/`;
the harness compiles the corresponding files under `references/seca-parity-build/SECA/`.
Rust paths below are relative to `crates/realtime-seca-core/src/`.

| Original C# methods | Rust counterpart | Role |
| --- | --- | --- |
| `Tokenizer.createSourceWordListInMemory`, `SourceWord.updateWordNoOfSources`; harness `Program.Rows` | `engine/baseline.rs` conversion, `tree/hkt_builder.rs:update_document_frequencies` | Ordered source-word relations and scoped DF |
| `SECAAlgorithm.buildTree/build` | `SecaEngine::build_baseline_tree`, `HktBuilder::build_full_tree` | Seed construction |
| `HKT.findMyExpectedWords`, `addNode`, `addOtherNodes`, `getSources`, `removeUsedWord` | `create_hkt`, expected-word selection, collision selection | Prominence, strength, eligibility, source unions and node creation |
| `HKT.addRefugeNodes`, `createBranches/createBranch`, scoped dataset helpers | `create_branches` | Refuge, descendants, attachment and stopping |
| `HKT.matchNewSourcesToHKT`, `Node.matchNewSourcesToNode`, `SECAAlgorithm.getSourceWordDatasetForNodeChildHKT` | `engine/scope_mapping.rs`, `engine/context.rs` | Multi-node mapping, unmatched sources, child evidence |
| `SECAAlgorithm.updateTree`, `HKT.isAccepted` | `engine/trigger.rs` | Top-down inspection, disjoint reconstruction targets |
| `SECAAlgorithm.reConstructHKT`, `AddSourcesWordsToMyParentNode`, `setDeadHKTs/removeDeadHKTs` | `engine/rebuild.rs` | Scoped historical/incoming evidence, ancestor exclusion, replacement and parent propagation |

## Comparison methodology and reproducibility

Run from the repository root:

```sh
tools/seca-reference-harness/build.sh
python3 tools/seca-reference-harness/verify.py
python3 crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/reference.py
cargo test -p realtime-seca-core --test parity_hkt_reconstruction
# Intentionally failing strict comparison, kept visible:
cargo test -p realtime-seca-core --test parity_hkt_reconstruction strict_executed_csharp_parity -- --ignored
```

Each fixture directory contains ordered `input.json`, actual `csharp.json`,
`csharp.normalized.json`, separately labelled `rust.observed.json`, and
`comparison.json`. The Python script runs the existing harness twice and requires
byte-identical stdout before saving C# evidence. `provenance.json` records reference
source hashes and harness source/script hashes. Mono is 6.8.0.105. The three principal
algorithm files (`HKT.cs`, `Node.cs`, `SECAAlgorithm.cs`) were byte-identical between
the original and compilable reference trees. `prepare.py` was inspected: it
serializes parallel traversal, substitutes deterministic allocators, disables unused
persistence construction, observes before acceptance, and surfaces swallowed failures.
Formulas and successful algorithm paths remain original. This establishes parity
against the serial instrumented reference, not race-dependent Windows execution.

The observer captures the evaluated HKT before reconstruction; decisions refer to
old IDs, which are mapped to paths in the preceding snapshot. Tree output comes
from live `hktList`, not an independent recreation. `verify.py` also checks manual
metrics/topology, stable follow-ups, deterministic repeats, and invalid-input failure.
Harness cycle reset and accumulated cleanup/history append were verified in Program.

Rust uses the same alpha/beta for builder and updates, all three supplied error
thresholds, the identical branch minimum, ordered tokens/documents, globally unique
source IDs and identical batch boundaries. Full memory excludes forgetting.
Public exports and decision diagnostics are used, and each resulting Rust snapshot
is restored to validate its state. No test-only production API was introduced.

Normalization represents an HKT recursively at `root/node-index/...`, with ordered
nodes, sorted token sets, sorted external source membership sets, explicit refuge
identity, child attachments and HKT expected-word sets. IDs and display metadata
are incidental; node order is retained because greedy merging can depend on it.
Every live HKT must be reachable exactly once, with one root and at most one child
per node. Token identity remains case sensitive. Source membership sorting is
lexical in both languages and does not alter algorithm execution order.

C# exposed HKT expected words are **dynamic** candidates; baseline construction's
local expected set is discarded. Rust persists construction candidates, including
already assigned words, until update recomputation. Full comparison preserves this
difference. `topology_equal` separately compares structure/memberships without
expected fields; `expected_and_topology_equal` compares those fields too, and
`decisions_equal` compares all observed inspection paths and triggers. Raw C# node
expected words, per-word/pending memberships and metrics remain in `csharp.json`;
Rust's public topology export has no equivalent node candidate field. No equality
of those unavailable internal fields is claimed. Detailed metric arithmetic belongs
to the other audit.

Tests assert current Rust observations and the exact per-cycle parity matrix;
passing means discrepancies remain accurately recorded, not universal parity.
`SECA_RECORD_RUST_AUDIT=1` intentionally regenerates only Rust observations and the
matrix after a reviewed engine change. It never writes C# expectations. The strict
comparison remains an ignored, explicitly executable failing test.

## Findings and minimal recommendations

- **Verified behavioural parity:** single prominent word; ordinary multi-node source
  mapping; merged-word node unions; no new refuge when an unmatched source arrives
  at an accepted HKT lacking refuge; child mapping; root replacement, dominant-word
  selection, ancestor exclusion and subsequent stable updates; two successive root
  reconstructions. The tested child reconstruction selects only `root/0`, replaces
  its evidence with b/c/x, retains its a attachment and preserves the d sibling's
  topology, memberships and IDs in each language. The next a/b update triggers the
  root in both engines, yielding the same structure/memberships.
- **Confirmed Rust discrepancy / unresolved scientific ambiguity — branch boundary:**
  C# `HKT.createBranch` uses `count >= minimum`; Rust skips `count <= minimum`.
  `branch_boundary` has two a sources, one also b, minimum 2: only C# creates the b
  child. The minimal parity correction is an inclusive cutoff, affecting baseline,
  reconstruction descendants and future inspection. Resolve the scientific minimum
  contract before changing it.
- **Confirmed Rust discrepancy / unresolved scientific ambiguity — greedy ordering:**
  C# iterates frequency-derived expected words and resolves equal node collision
  scores to the last enumerated maximum. Rust processes ascending vocabulary IDs
  and keeps the first maximum. `collision_true_tie` selects b/c in C#, a/c in Rust;
  `frequency_vs_vocabulary` creates z,b,a in C#, z,{a,b} in Rust. Minimal reference
  parity corrections are frequency/record-order expected-word processing and last
  maximum collision selection. They affect source assignment, baseline and rebuilt
  structures and subsequent updates. Do not copy unspecified hash iteration as a
  scientific rule. The older `collision` fixture is not a true simultaneous tie:
  c is visited before b exists, so both engines agree there.
- **Confirmed ordering difference:** `siblings` has C# child order f,e below d,
  while Rust has e,f, caused by C# scoped ConcurrentDictionary enumeration versus
  Rust original row ordering. Memberships match by labels, but ordered topology
  deliberately fails. No ID-induced attachment defect was observed. A portable
  scientific ordering rule is preferable to emulating runtime hash buckets.
- **Confirmed Rust discrepancy / unresolved scientific ambiguity — recursive scope:**
  C# recursively passes child construction's post-adoption rows; Rust reuses the
  pre-construction child scope. `overlap_recursive` makes Rust grandchildren c below
  b and b below c for shared sources 3,4; C# stops after b/c. This uses minimum 1
  and two shared sources, isolating it from the cutoff discrepancy. A minimal parity
  correction would remove all words adopted at that HKT from recursive evidence.
  It affects depth, memberships, descendants rebuilt and subsequent inspection.
- **Original C# deviation from paper refuge semantics / deliberate Rust difference:**
  `refuge` maps source 6 (b) to the refuge child but leaves the parent refuge's
  persistent sources at {4,5}; Rust persists {4,5,6}. C# `matchNewSourcesToHKT`
  skips refuge nodes; child dataset selection uses unmatched rows, and accepted
  membership merging only adds memberships for node words. Refuge has no real word
  to merge. Its child nevertheless gains source 6. Rust's explicit refuge union
  better represents the definition's unmatched documents. Preserve the Rust union;
  do not introduce stale C# parent membership. This difference can affect later
  parent-scoped evidence and therefore later reconstruction.
- **Deliberate implementation difference — empty child inspection:** C# recursively
  evaluates empty scopes; Rust skips them. In the tested stable follow-ups these
  extra C# decisions are accepted and topology is unchanged. This is observable
  decision inequality, not a reconstruction-target difference. No correction is
  justified by these successful cases; empty-scope triggering remains untested.
- **Deliberate representation difference — expected words:** stored construction
  sets differ initially and in newly reconstructed/uninspected descendants; many
  accepted updates converge to empty candidate sets. Do not erase the difference
  from golden equality. Specify construction versus dynamic candidates before any
  state-model change. No reconstruction defect attributable solely to persistence
  was demonstrated.

Historical source-only warnings about C# first-row creation paired with another
word's removal, shared-row concurrency, empty reconstruction failure and duplicate
relations are not established by these successful executions. Inputs exclude
duplicate pairs; harness rejects branch minimum 0 and empty batches. The old Rust
minimum-0 test is a Rust-only regression, not executable reference evidence.
Potential empty-evidence failure and more complex candidate admission require
separate fixtures; no claims are inferred from overall node counts.

## Fixture results

Notation: `words:{source IDs}/[child nodes]`; ordered nodes are separated by `;`.
Sources are sequential IDs assigned across ordered batches. Input and exact JSON
for each case are linked. `T/E/D` gives topology-membership, full expected-topology,
and inspection-decision equality for each cycle (`Y`/`N`). Parameters are
`alpha/beta/branchMinimum; alphaError/betaError/wordImportanceError`.

### branch_boundary

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/branch_boundary/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/branch_boundary/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/branch_boundary/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/branch_boundary/rust.observed.json). Parameters: `0.7/0.5/2; 1/1/1`.

Batches: [1:a b, 2:a] → [3:a b] → [4:a].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a:{1,2}/[b:{1}]]` | `[a:{1,2}]` | N/N/Y |
| 2 | `[a:{1,2,3}/[b:{1,3}]]` | `[a:{1,2,3}]` | N/N/N |
| 3 | `[a:{1,2,3,4}/[b:{1,3}]]` | `[a:{1,2,3,4}]` | N/N/N |

### child

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/child/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/child/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/child/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/child/rust.observed.json). Parameters: `0.7/0.5/1; 1/1/1`.

Batches: [1:a b, 2:a b, 3:a, 4:a] → [5:a b] → [6:a b].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a:{1,2,3,4}/[b:{1,2}]]` | `[a:{1,2,3,4}/[b:{1,2}]]` | Y/Y/Y |
| 2 | `[a:{1,2,3,4,5}/[b:{1,2,5}]]` | `[a:{1,2,3,4,5}/[b:{1,2,5}]]` | Y/Y/Y |
| 3 | `[a:{1,2,3,4,5,6}/[b:{1,2,5,6}]]` | `[a:{1,2,3,4,5,6}/[b:{1,2,5,6}]]` | Y/Y/Y |

### child_reconstruct

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/child_reconstruct/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/child_reconstruct/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/child_reconstruct/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/child_reconstruct/rust.observed.json). Parameters: `0.7/0.5/1; 0.11/0.2/0.2`.

Batches: [1:a b, 2:a b, 3:a b, 4:a b, 5:a c, 6:a c, 7:a c, 8:a c, 9:a, 10:a, 11:d e, 12:d e, 13:d e, 14:d e, 15:d f, 16:d f, 17:d f, 18:d f, 19:d, 20:d] → [21:a x, 22:a x, 23:a x, 24:a, 25:a, 26:a, 27:a, 28:a, 29:a, 30:a] → [31:a b].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a:{1,10,2,3,4,5,6,7,8,9}/[b:{1,2,3,4}; c:{5,6,7,8}]; d:{11,12,13,14,15,16,17,18,19,20}/[f:{15,16,17,18}; e:{11,12,13,14}]]` | `[a:{1,10,2,3,4,5,6,7,8,9}/[b:{1,2,3,4}; c:{5,6,7,8}]; d:{11,12,13,14,15,16,17,18,19,20}/[e:{11,12,13,14}; f:{15,16,17,18}]]` | N/N/Y |
| 2 | `[a:{1,10,2,21,22,23,24,25,26,27,28,29,3,30,4,5,6,7,8,9}/[b:{1,2,3,4}; c:{5,6,7,8}; x:{21,22,23}]; d:{11,12,13,14,15,16,17,18,19,20}/[f:{15,16,17,18}; e:{11,12,13,14}]]` | `[a:{1,10,2,21,22,23,24,25,26,27,28,29,3,30,4,5,6,7,8,9}/[b:{1,2,3,4}; c:{5,6,7,8}; x:{21,22,23}]; d:{11,12,13,14,15,16,17,18,19,20}/[e:{11,12,13,14}; f:{15,16,17,18}]]` | N/N/N |
| 3 | `[a:{1,10,2,21,22,23,24,25,26,27,28,29,3,30,31,4,5,6,7,8,9}/[b:{1,2,3,31,4}; c:{5,6,7,8}; <>:{21,22,23}/[x:{21,22,23}]]; <>:{11,12,13,14,15,16,17,18,19,20}/[d:{11,12,13,14,15,16,17,18,19,20}/[e:{11,12,13,14}; f:{15,16,17,18}]]]` | `[a:{1,10,2,21,22,23,24,25,26,27,28,29,3,30,31,4,5,6,7,8,9}/[b:{1,2,3,31,4}; c:{5,6,7,8}; <>:{21,22,23}/[x:{21,22,23}]]; <>:{11,12,13,14,15,16,17,18,19,20}/[d:{11,12,13,14,15,16,17,18,19,20}/[e:{11,12,13,14}; f:{15,16,17,18}]]]` | Y/N/Y |

### collision

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/collision/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/collision/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/collision/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/collision/rust.observed.json). Parameters: `1/0.5/99; 1/1/1`.

Batches: [1:a c, 2:a, 3:b c, 4:b] → [5:a b c] → [6:a b c].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a,c:{1,2,3}; b:{3,4}]` | `[a,c:{1,2,3}; b:{3,4}]` | Y/N/Y |
| 2 | `[a,c:{1,2,3,5}; b:{3,4,5}]` | `[a,c:{1,2,3,5}; b:{3,4,5}]` | Y/Y/Y |
| 3 | `[a,c:{1,2,3,5,6}; b:{3,4,5,6}]` | `[a,c:{1,2,3,5,6}; b:{3,4,5,6}]` | Y/Y/Y |

### collision_reordered

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/collision_reordered/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/collision_reordered/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/collision_reordered/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/collision_reordered/rust.observed.json). Parameters: `1/0.5/99; 1/1/1`.

Batches: [1:b c, 2:b, 3:a c, 4:a] → [5:a b c] → [6:a b c].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[b,c:{1,2,3}; a:{3,4}]` | `[b,c:{1,2,3}; a:{3,4}]` | Y/N/Y |
| 2 | `[b,c:{1,2,3,5}; a:{3,4,5}]` | `[b,c:{1,2,3,5}; a:{3,4,5}]` | Y/Y/Y |
| 3 | `[b,c:{1,2,3,5,6}; a:{3,4,5,6}]` | `[b,c:{1,2,3,5,6}; a:{3,4,5,6}]` | Y/Y/Y |

### collision_true_tie

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/collision_true_tie/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/collision_true_tie/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/collision_true_tie/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/collision_true_tie/rust.observed.json). Parameters: `1/0.5/99; 1/1/1`.

Batches: [1:a, 2:b, 3:a c, 4:b c] → [5:a b c] → [6:a b c].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a:{1,3}; b,c:{2,3,4}]` | `[a,c:{1,3,4}; b:{2,4}]` | N/N/Y |
| 2 | `[a:{1,3,5}; b,c:{2,3,4,5}]` | `[a,c:{1,3,4,5}; b:{2,4,5}]` | N/N/Y |
| 3 | `[a:{1,3,5,6}; b,c:{2,3,4,5,6}]` | `[a,c:{1,3,4,5,6}; b:{2,4,5,6}]` | N/N/Y |

### competing

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/competing/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/competing/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/competing/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/competing/rust.observed.json). Parameters: `0.7/0.75/99; 0.1/0.13/0.3`.

Batches: [1:a, 2:b, 3:a b] → [4:a b] → [5:a].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a:{1,3}; b:{2,3}]` | `[a:{1,3}; b:{2,3}]` | Y/N/Y |
| 2 | `[a:{1,3,4}; b:{2,3,4}]` | `[a:{1,3,4}; b:{2,3,4}]` | Y/Y/Y |
| 3 | `[a:{1,3,4,5}; b:{2,3,4}]` | `[a:{1,3,4,5}; b:{2,3,4}]` | Y/Y/Y |

### frequency_vs_vocabulary

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/frequency_vs_vocabulary/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/frequency_vs_vocabulary/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/frequency_vs_vocabulary/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/frequency_vs_vocabulary/rust.observed.json). Parameters: `0.5/1/99; 1/1/1`.

Batches: [1:a b, 2:a b, 3:b, 4:z, 5:z, 6:z, 7:z] → [8:z] → [9:a b].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[z:{4,5,6,7}; b:{1,2,3}; a:{1,2}]` | `[z:{4,5,6,7}; a,b:{1,2,3}]` | N/N/Y |
| 2 | `[z:{4,5,6,7,8}; b:{1,2,3}; a:{1,2}]` | `[z:{4,5,6,7,8}; a,b:{1,2,3}]` | N/N/Y |
| 3 | `[z:{4,5,6,7,8}; b:{1,2,3,9}; a:{1,2,9}]` | `[z:{4,5,6,7,8}; a,b:{1,2,3,9}]` | N/N/Y |

### merge_many

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/merge_many/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/merge_many/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/merge_many/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/merge_many/rust.observed.json). Parameters: `0.7/0.5/1; 0.1/0.13/0.3`.

Batches: [1:a b c, 2:a b c] → [3:a b c] → [4:a b c].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a,b,c:{1,2}]` | `[a,b,c:{1,2}]` | Y/N/Y |
| 2 | `[a,b,c:{1,2,3}]` | `[a,b,c:{1,2,3}]` | Y/Y/Y |
| 3 | `[a,b,c:{1,2,3,4}]` | `[a,b,c:{1,2,3,4}]` | Y/Y/Y |

### new_dominant

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/new_dominant/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/new_dominant/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/new_dominant/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/new_dominant/rust.observed.json). Parameters: `0.7/0.5/1; 0.1/0.13/0.3`.

Batches: [1:a b, 2:a b] → [3:c, 4:c, 5:c] → [6:c].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a,b:{1,2}]` | `[a,b:{1,2}]` | Y/N/Y |
| 2 | `[c:{3,4,5}; <>:{1,2}/[a,b:{1,2}]]` | `[c:{3,4,5}; <>:{1,2}/[a,b:{1,2}]]` | Y/N/Y |
| 3 | `[c:{3,4,5,6}; <>:{1,2}/[a,b:{1,2}]]` | `[c:{3,4,5,6}; <>:{1,2}/[a,b:{1,2}]]` | Y/N/N |

### one_prominent

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/one_prominent/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/one_prominent/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/one_prominent/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/one_prominent/rust.observed.json). Parameters: `0.7/0.5/1; 0.1/0.13/0.3`.

Batches: [1:a, 2:a] → [3:a] → [4:a].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a:{1,2}]` | `[a:{1,2}]` | Y/Y/Y |
| 2 | `[a:{1,2,3}]` | `[a:{1,2,3}]` | Y/Y/Y |
| 3 | `[a:{1,2,3,4}]` | `[a:{1,2,3,4}]` | Y/Y/Y |

### overlap_recursive

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/overlap_recursive/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/overlap_recursive/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/overlap_recursive/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/overlap_recursive/rust.observed.json). Parameters: `0.8/0.75/1; 1/1/1`.

Batches: [1:a b, 2:a b, 3:a b c, 4:a b c, 5:a c, 6:a c, 7:a] → [8:a b c] → [9:a b c].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a:{1,2,3,4,5,6,7}/[b:{1,2,3,4}; c:{3,4,5,6}]]` | `[a:{1,2,3,4,5,6,7}/[b:{1,2,3,4}/[c:{3,4}]; c:{3,4,5,6}/[b:{3,4}]]]` | N/N/Y |
| 2 | `[a:{1,2,3,4,5,6,7,8}/[b:{1,2,3,4,8}; c:{3,4,5,6,8}]]` | `[a:{1,2,3,4,5,6,7,8}/[b:{1,2,3,4,8}/[c:{3,4,8}]; c:{3,4,5,6,8}/[b:{3,4,8}]]]` | N/N/N |
| 3 | `[a:{1,2,3,4,5,6,7,8,9}/[b:{1,2,3,4,8,9}; c:{3,4,5,6,8,9}]]` | `[a:{1,2,3,4,5,6,7,8,9}/[b:{1,2,3,4,8,9}/[c:{3,4,8,9}]; c:{3,4,5,6,8,9}/[b:{3,4,8,9}]]]` | N/N/N |

### refuge

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/refuge/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/refuge/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/refuge/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/refuge/rust.observed.json). Parameters: `0.8/0.5/1; 1/1/1`.

Batches: [1:a, 2:a, 3:a, 4:b, 5:b] → [6:b] → [7:b].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a:{1,2,3}; <>:{4,5}/[b:{4,5}]]` | `[a:{1,2,3}; <>:{4,5}/[b:{4,5}]]` | Y/Y/Y |
| 2 | `[a:{1,2,3}; <>:{4,5}/[b:{4,5,6}]]` | `[a:{1,2,3}; <>:{4,5,6}/[b:{4,5,6}]]` | N/N/Y |
| 3 | `[a:{1,2,3}; <>:{4,5}/[b:{4,5,6,7}]]` | `[a:{1,2,3}; <>:{4,5,6,7}/[b:{4,5,6,7}]]` | N/N/Y |

### root_reconstruct

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/root_reconstruct/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/root_reconstruct/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/root_reconstruct/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/root_reconstruct/rust.observed.json). Parameters: `0.7/0.5/1; 0.11/0.2/0.2`.

Batches: [1:a b, 2:a] → [3:x, 4:x, 5:x, 6:x] → [7:x].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a:{1,2}/[b:{1}]]` | `[a:{1,2}/[b:{1}]]` | Y/Y/Y |
| 2 | `[x:{3,4,5,6}; <>:{1,2}/[a:{1,2}/[b:{1}]]]` | `[x:{3,4,5,6}; <>:{1,2}/[a:{1,2}/[b:{1}]]]` | Y/Y/Y |
| 3 | `[x:{3,4,5,6,7}; <>:{1,2}/[a:{1,2}/[b:{1}]]]` | `[x:{3,4,5,6,7}; <>:{1,2}/[a:{1,2}/[b:{1}]]]` | Y/Y/N |

### separate

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/separate/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/separate/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/separate/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/separate/rust.observed.json). Parameters: `0.7/0.75/99; 0.1/0.13/0.3`.

Batches: [1:a, 2:a, 3:b, 4:b] → [5:a b] → [6:b].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a:{1,2}; b:{3,4}]` | `[a:{1,2}; b:{3,4}]` | Y/N/Y |
| 2 | `[a:{1,2,5}; b:{3,4,5}]` | `[a:{1,2,5}; b:{3,4,5}]` | Y/Y/Y |
| 3 | `[a:{1,2,5}; b:{3,4,5,6}]` | `[a:{1,2,5}; b:{3,4,5,6}]` | Y/Y/Y |

### sequential

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/sequential/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/sequential/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/sequential/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/sequential/rust.observed.json). Parameters: `0.7/0.5/1; 0.1/0.13/0.3`.

Batches: [1:a b, 2:a b] → [3:c, 4:c, 5:c] → [6:c] → [7:d, 8:d, 9:d, 10:d, 11:d, 12:d, 13:d, 14:d] → [15:d].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a,b:{1,2}]` | `[a,b:{1,2}]` | Y/N/Y |
| 2 | `[c:{3,4,5}; <>:{1,2}/[a,b:{1,2}]]` | `[c:{3,4,5}; <>:{1,2}/[a,b:{1,2}]]` | Y/N/Y |
| 3 | `[c:{3,4,5,6}; <>:{1,2}/[a,b:{1,2}]]` | `[c:{3,4,5,6}; <>:{1,2}/[a,b:{1,2}]]` | Y/N/N |
| 4 | `[d:{10,11,12,13,14,7,8,9}; <>:{1,2,3,4,5,6}/[c:{3,4,5,6}; <>:{1,2}/[a,b:{1,2}]]]` | `[d:{10,11,12,13,14,7,8,9}; <>:{1,2,3,4,5,6}/[c:{3,4,5,6}; <>:{1,2}/[a,b:{1,2}]]]` | Y/N/Y |
| 5 | `[d:{10,11,12,13,14,15,7,8,9}; <>:{1,2,3,4,5,6}/[c:{3,4,5,6}; <>:{1,2}/[a,b:{1,2}]]]` | `[d:{10,11,12,13,14,15,7,8,9}; <>:{1,2,3,4,5,6}/[c:{3,4,5,6}; <>:{1,2}/[a,b:{1,2}]]]` | Y/N/N |

### siblings

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/siblings/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/siblings/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/siblings/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/siblings/rust.observed.json). Parameters: `0.7/0.5/1; 1/1/1`.

Batches: [1:a b, 2:a b, 3:a b, 4:a b, 5:a c, 6:a c, 7:a c, 8:a c, 9:a, 10:a, 11:d e, 12:d e, 13:d e, 14:d e, 15:d f, 16:d f, 17:d f, 18:d f, 19:d, 20:d] → [21:a b] → [22:d e].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a:{1,10,2,3,4,5,6,7,8,9}/[b:{1,2,3,4}; c:{5,6,7,8}]; d:{11,12,13,14,15,16,17,18,19,20}/[f:{15,16,17,18}; e:{11,12,13,14}]]` | `[a:{1,10,2,3,4,5,6,7,8,9}/[b:{1,2,3,4}; c:{5,6,7,8}]; d:{11,12,13,14,15,16,17,18,19,20}/[e:{11,12,13,14}; f:{15,16,17,18}]]` | N/N/Y |
| 2 | `[a:{1,10,2,21,3,4,5,6,7,8,9}/[b:{1,2,21,3,4}; c:{5,6,7,8}]; d:{11,12,13,14,15,16,17,18,19,20}/[f:{15,16,17,18}; e:{11,12,13,14}]]` | `[a:{1,10,2,21,3,4,5,6,7,8,9}/[b:{1,2,21,3,4}; c:{5,6,7,8}]; d:{11,12,13,14,15,16,17,18,19,20}/[e:{11,12,13,14}; f:{15,16,17,18}]]` | N/N/N |
| 3 | `[a:{1,10,2,21,3,4,5,6,7,8,9}/[b:{1,2,21,3,4}; c:{5,6,7,8}]; d:{11,12,13,14,15,16,17,18,19,20,22}/[f:{15,16,17,18}; e:{11,12,13,14,22}]]` | `[a:{1,10,2,21,3,4,5,6,7,8,9}/[b:{1,2,21,3,4}; c:{5,6,7,8}]; d:{11,12,13,14,15,16,17,18,19,20,22}/[e:{11,12,13,14,22}; f:{15,16,17,18}]]` | N/N/N |

### single_root

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/single_root/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/single_root/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/single_root/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/single_root/rust.observed.json). Parameters: `0.7/0.5/1; 0.1/0.13/0.3`.

Batches: [1:a b, 2:a b] → [3:a b] → [4:a b].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a,b:{1,2}]` | `[a,b:{1,2}]` | Y/N/Y |
| 2 | `[a,b:{1,2,3}]` | `[a,b:{1,2,3}]` | Y/Y/Y |
| 3 | `[a,b:{1,2,3,4}]` | `[a,b:{1,2,3,4}]` | Y/Y/Y |

### ties_ab

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/ties_ab/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/ties_ab/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/ties_ab/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/ties_ab/rust.observed.json). Parameters: `0.7/0.5/99; 0.1/0.13/0.3`.

Batches: [1:a b] → [2:a b] → [3:a b].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a,b:{1}]` | `[a,b:{1}]` | Y/N/Y |
| 2 | `[a,b:{1,2}]` | `[a,b:{1,2}]` | Y/Y/Y |
| 3 | `[a,b:{1,2,3}]` | `[a,b:{1,2,3}]` | Y/Y/Y |

### ties_ba

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/ties_ba/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/ties_ba/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/ties_ba/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/ties_ba/rust.observed.json). Parameters: `0.7/0.5/99; 0.1/0.13/0.3`.

Batches: [1:b a] → [2:b a] → [3:b a].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a,b:{1}]` | `[a,b:{1}]` | Y/N/Y |
| 2 | `[a,b:{1,2}]` | `[a,b:{1,2}]` | Y/Y/Y |
| 3 | `[a,b:{1,2,3}]` | `[a,b:{1,2,3}]` | Y/Y/Y |

### unmatched

[Input](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/unmatched/input.json), [C# raw](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/unmatched/csharp.json), [normalized C#](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/unmatched/csharp.normalized.json), [Rust](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/unmatched/rust.observed.json). Parameters: `0.7/0.5/99; 1/1/1`.

Batches: [1:a, 2:a] → [3:x] → [4:a].

| Cycle | C# | Rust | T/E/D |
| --- | --- | --- | --- |
| 1 | `[a:{1,2}]` | `[a:{1,2}]` | Y/Y/Y |
| 2 | `[a:{1,2}]` | `[a:{1,2}]` | Y/Y/Y |
| 3 | `[a:{1,2,4}]` | `[a:{1,2,4}]` | Y/Y/Y |

## Build and test results

Harness build passed (22 inherited compiler warnings). `verify.py` passed.
All 21 reference fixtures were byte-identical across two executions per fixture.
`cargo fmt --all -- --check` passed. Focused integration: 12 passed, one explicitly
ignored strict parity test. Running that test with `--ignored` fails as expected
(exit 101), first on branch-boundary structure; this is retained evidence.
Focused clippy with `-D warnings` passed. Core library tests: 74 passed.
No architecture changed; full workspace tests were not run.

The existing file exceeds 400 lines after adding the audit. Its earlier 11 tests
were preserved; new execution helpers are cohesive and confined to an internal
test module. No module boundaries or production state ownership changed.


---

The revision-specific report from main follows unchanged. Its earlier execution-availability statements are superseded by the executable evidence above and the authoritative summary.

# HKT construction and selective reconstruction parity audit

Historical audit of **15814d1**. Test source is retained at commit `432c47c`;
clean-main confirmation and scientific limitations are summarized separately.
See [the consolidated audit](audit-2026-10.md) for revision reconciliation.


## Scope and evidence

Audited checkout: `15814d1d636a853acbf0ca950d44cce4c534b3e4` (initial branch
`fix/seca-light-stream`). Local `main` is `42c07c098c1fadb931a4e7b5b6920ee9561dabc5`;
this audit preserves the supplied checkout rather than discarding its newer
implementation. Work is on `audit/hkt-reconstruction-parity`. Unrelated CLI edits
were present during the audit and are excluded from this commit. Other workers
committed CLI and state-invariant changes in the shared checkout while this audit
ran; the audit commit stages only the two paths requested here.

No production Rust, C# source, or scientific specification was changed. Tests use
public builder/engine APIs. `dotnet`, `mono`, `csc`, and `mcs` are absent from PATH;
therefore **all C# outputs below are source-inferred**, conditional on stated
collection enumeration, and none are runtime-verified cross-language results.
Rust outputs are verified by `tests/parity_hkt_reconstruction.rs`. Tests pin the
current Rust contract, including discrepancies; passing does not establish parity.

Scientific reference: Al Sulaimani and Starkey, [2023 SECA paper](https://doi.org/10.1109/ACCESS.2023.3331219),
sections V and VII-A, Figure 3, equations 2–3. The accessible
[author manuscript](https://www.researchgate.net/publication/375503068_Real-Time_Event_Detection_Using_Self-Evolving_Contextual_Analysis_SECA_Approach)
was inspected; the university publication page could not be fetched (401).
Detailed error arithmetic is outside this audit.

## Component mapping

C# paths below are relative to `references/seca-csharp/SECA/SECA/`;
Rust paths are relative to `crates/realtime-seca-core/src/`.

| C# component | Rust component |
| --- | --- |
| `SourceCleanerEngine/BLClass/Tokenizer.cs:createSourceWordListInMemory` | `engine/baseline.rs:convert_batch_to_source_word_records` |
| `SourceCleanerEngine/DataClass/SourceWord.cs:updateWordNoOfSources` | `tree/hkt_builder.rs:update_document_frequencies` |
| `SECA/SECA_DataClass/HKT.cs:addNode`, `findMyExpectedWords` | `HktBuilder::create_hkt`, `find_expected_words_general` |
| `HKT.cs:addOtherNodes`, `getSources`, `removeUsedWord` | collision selection, node creation, word-row removal in builder |
| `HKT.cs:addRefugeNodes`, `createBranches`, `createBranch` | refuge construction and `create_branches` |
| `HKT.cs:setSourceWordListForCurrentNode`, `setSourceWordListForRefugeNode` | child-scope filtering and frequency recomputation |
| `SECA/SECA_DataClass/Node.cs`, `HKT.cs` | `tree/models.rs:Node`, `Hkt` |
| `SECAAlgorithm.cs:updateTree`, child dataset helpers | `engine/trigger.rs`, `engine/scope_mapping.rs` |
| `SECAAlgorithm.cs:reConstructHKT`, `AddSourcesWordsToMyParentNode` | `engine/rebuild.rs` |
| `SECAAlgorithm.cs:setDeadHKTs`, `removeDeadHKTs` | subtree collection/removal and replacement |

## Baseline construction comparison

Both operate on source/word relations, seed with the first row's word, admit
words at strength >= alpha, use intersection / existing node source count >= beta,
merge source unions, and create refuge from sources absent from all ordinary nodes.
Removing an adopted word removes its rows, not all rows of its sources. A source
can remain in several nodes; each baseline word is processed once at a level.

Rust assigns vocabulary IDs by first token encounter, trims tokens, validates and
deduplicates source identities, and counts distinct sources per word. C# tokenizer
assigns vocabulary from a `HashSet<string>` and uses NLP output. Its frequency
helper counts rows, although node membership uses distinct sources. Duplicate
source/word rows therefore affect C# admission but not Rust document frequency.
End-to-end tokenizer parity was not tested; fixtures use explicit relations.

Rust always sorts by descending cached frequency then `source_word_id`, reindexes
rows, and iterates expected words in ascending word ID. C# `build(Dictionary)`
assumes already sorted input; tokenizer and reconstruction explicitly sort.
`findMyExpectedWords` scans all rows; Rust stops at the first below-threshold row
because sorting is guaranteed. C# iterates a HashSet of expected words. Child
records come from a ConcurrentDictionary, so equal-frequency child ordering is
not a portable first-encounter contract. Rust child tie ordering retains original
record IDs. Fixed input ordering is deterministic in Rust, but permutations of
sources/tokens or inherited vocabulary IDs can change greedy topology.

C# `addOtherNodes` (lines 863–935) creates a noncolliding node using the **first
remaining row's word**, but removes rows of the **inspected expected word**.
Rust creates/removes the inspected word. These are equivalent only when expected
iteration agrees with remaining-row order. C# chooses the last enumerated node
at the maximum collision score; Rust chooses the first created node. Node union
changes later denominators, making both word and collision order consequential.

Rust branches only when source count > configured minimum. C# `createBranch`
(line 691) branches when count >= minimum. For ordinary nodes both select
parent-member rows excluding parent words; refuge includes all member rows. Both
recompute scoped frequencies and attach children to the originating node.
However, C# passes the **mutated** local dataset into further recursion after
`addNode`; adopted words from the entire child HKT have been removed. Rust stores
and recursively uses the **pre-construction** scope, excluding only the particular
parent node's words. Thus a word assigned to a sibling node can reappear in a
Rust descendant when source memberships overlap; C# removes it from the recursive
dataset. This is a confirmed source-level scope difference, not runtime-verified
C# topology. C# also mutates shared SourceWord frequency objects in parallel
branches; overlapping branch scopes introduce a potential race. Rust clones rows.

C# construction expected words are a temporary local set; dynamic expected words
start empty. Rust stores the strength-admitted set minus the prominent word,
including words already assigned to nodes. These fields cannot be compared as
identical scientific sets or raw snapshots. Rust `top_words` are presentation
metadata, excluded from normalized structural equality.

## Reconstruction comparison

The paper's VII-A/Figure 3 specifies top-down inspection: use the full incoming
batch at Seed-HKT, parent-adopted sources at children, and reconstruct a triggered
container and descendants instead of inspecting those descendants. Both traversal
paths follow this pattern. Rust explicitly skips empty incoming scopes; C# may
call child traversal with an empty dataset. Error outcomes for empty datasets are
not inferred here.

C# `reConstructHKT` (lines 554–628) unions historical rows whose sources belong
to the selected HKT with the exact child incoming dataset, excludes ancestor
words, recomputes frequencies, sorts, and rebuilds with the old parent attachment.
Rust performs the corresponding scoped operation using retained token payloads
and trigger provenance; it deduplicates source/word pairs. Retained history means
current memory contents, not lifetime history. Full memory is used by these tests
to isolate reconstruction from forgetting.

C# ancestor filtering uses `ancestoWords`; Rust excludes the union of all,
accepted, and rejected ancestor maps. Their equivalence depends on the ancestor
context builder preserving those inclusions. Baseline branch filtering alone is
not equivalent to this explicit reconstruction filter.

C# propagates filtered incoming source IDs upward; Rust propagates all source IDs
represented in the filtered replacement dataset. In valid preexisting ancestry,
historical IDs should already belong to ancestors, so this larger union is
normally redundant. Neither path adds a source with no remaining nonancestor row
through this propagation step. The child fixture checks the parent union and
absence of sibling historical sources in the replacement.

Rust removes precisely the selected subtree from both indexes and allocates fresh
monotonic IDs. C# removes descendants via `setDeadHKTs`/caller cleanup, uses random
IDs, and may retain archive objects/logical removal metadata. Compare only live
reachable hierarchy; archives and ID allocation are incidental. Accepted sibling
scopes may receive their own memberships during an update; preservation means
absence of replacement or unintended mutation, not necessarily universal byte
identity. Our child fixture has no sibling incoming scope and verifies its exact
normalized contents and surviving IDs.

Rust skips a selected rebuild when ancestor exclusion yields no records, keeping
the original subtree. C# continues removal then calls `addNode` on empty input;
`findMyExpectedWords.First()` throws. The outer update catch swallows exceptions,
so a partially removed tree is possible. This is a source-inferred failure-path
difference. An empty batch is **not** an empty reconstruction dataset: it skips
inspection. There is no public API to inject a selected empty trigger plan;
that particular failure path is not runtime-covered by these integration tests.

## Deterministic datasets and normalized comparisons

Notation: `w:{sources}` is a node, `<>` is refuge, `/` attaches a child HKT.
Integer words in builder tests are explicit vocabulary identities; source labels
are exact integers or external IDs. No HKT/node IDs enter normalized comparisons.
C# predictions use frequency order for expected words and creation order for
nodes unless otherwise stated. HashSet/ConcurrentDictionary enumeration prevents
claiming these predictions are universal C# outputs.

| Case and parameters | Dataset | Rust verified output | C# source-inferred comparison |
| --- | --- | --- | --- |
| Single root; alpha .7, beta .5, min 1 | 1:[1,2], 2:[1,2] | `{1,2}:{1,2}`, no child; expected={2} | Same node, local expected set discarded |
| Competing words; .7/.75/min99 | 1:[1], 2:[2], 3:[1,2] | `1:{1,3}`, `2:{2,3}` | Same ordinary memberships |
| Prominent tie; .7/.5/min99 | 1:[1,2] versus 1:[2,1] | Same merged node; expected={2} versus {1} | First row seeds prominent; no persistent baseline expected set |
| Collision tie; 1/.5/min99 | 1:[1,3], 2:[1], 3:[2,3], 4:[2] | `{1,3}:{1,2,3}`, `2:{3,4}` | With node order 1 then 2: `1:{1,2}`, `{2,3}:{1,3,4}`; tie attaches to last |
| Word-order mismatch; .5/1/min99 | 1–4:[3], 5–6:[2,1], 7:[2] | `3:{1,2,3,4}`, `{1,2}:{5,6,7}` | Frequency iteration 3,2,1: separate nodes `3`, `2`, `1`; source overlap remains |
| Refuge; .8/.5/min1 | 1–3:[1], 4–5:[2] | `1:{1,2,3}`, `<>:{4,5}/2:{4,5}` | Same hierarchy; descendant recursion empties after adoption |
| Branch boundary; .7/.5 | 1:[1,2], 2:[1], min2 versus min1 | min2: root only; min1: `1:{1,2}/2:{1}` | min2 already creates child; at min1 same child |
| Two siblings; .7/.5/min1 | s0–9: a; s10–19: d; first four per group add b/e, next four add c/f | Root a/d; children b/c and e/f | Same, barring child tie order; normalized labels agree |
| One child reconstructed | Previous baseline; n0–9:[a], n0–2 also x | Only a-child replaced, containing b/c/x evidence; no a in child; d-child preserved | Dataset scope and attachment agree; exact trigger outcome not executed |
| Root reconstructed/new dominant | o1:[a,b], o2:[a]; x0–3:[x] | Root x plus refuge old evidence; all old HKT IDs removed; only root inspected | Sorted replacement selects x; below-refuge depth can differ due cutoff |
| Incoming multiple matches | a1:[a], d1:[d]; both:[a,d] | both belongs to both root nodes; no reconstruction | Mapping permits same memberships |
| Reconstruction then no change | Root replacement above; x4:[x] | No further reconstruction; HKT IDs survive | Expected conditional on equivalent trigger configuration; not executed |
| Empty incoming scope | Multiple-match case followed by empty batch | Normalized hierarchy and memberships unchanged in Full mode | Empty traversal behavior not asserted |
| Overlapping multilevel scope; .8/.75/min0 | 1–6 carry 0; 1–3 also 1; 3–5 also 2 | Root 0 / child nodes 1 and 2 / each has the other word below it, source={3} | Child construction removes both words before recursion: no grandchildren |
| Empty builder input | No relations | No HKT returned | `First()` throws; not a valid baseline reference tree |

The collision-tie test also reorders the same source relations to seed word 2
instead of word 1: Rust then merges word 3 with word 2. This is a verified
topology change caused solely by equal-frequency input ordering.

`normalized` in the integration tests recursively records sorted token word sets,
external source sets, child attachment, and expected tokens. It sorts sibling
representations and omits allocators, state flags, top words, and internal hashes.
The report supplies conditional C# structures; tests do not pretend to execute a
second implementation. Fresh versus preserved IDs are checked separately where
replacement identity matters. Builder tests also check the mirrored node cache.

## Discrepancies, causes, and minimal recommendations

1. **Branch boundary:** literal `>` versus `>=` mismatch. Decide the scientific
   minimum contract, then align the comparison and config documentation with a
   boundary regression. The 2023 SECA description does not settle this cutoff.
2. **Expected-word ordering and collision ties:** ordered Rust sets versus C#
   hash collection enumeration/last maximum. A deterministic tie policy needs
   explicit scientific approval; do not copy unspecified hash iteration. Test
   permutations and greedy membership effects before choosing an ordering.
3. **Noncolliding C# node selection:** first-row creation paired with removal of
   another word is fragile. Preserve Rust's coherent inspected-word behavior;
   document divergence rather than introduce the C# defect for parity.
4. **Recursive dataset:** C# passes post-adoption rows, Rust pre-adoption scope.
   Decide whether sibling descriptors may recur below another parent with shared
   sources using the original CA references. Add an overlapping multilevel
   cross-language fixture before changing Rust scope semantics.
5. **Frequency inputs:** row count versus distinct source count. Paper sources
   are sets; distinct-source Rust counts are supported. Normalize duplicate pairs
   in any future C# harness rather than change Rust to count repetitions.
6. **Expected-word persistence:** distinguish construction candidates from
   dynamic unassigned expected words when specifying cross-language comparison.
   Do not rename/remove Rust state without resolving its update contract.
7. **Empty replacement:** preserve transactional state and define the intended
   no-evidence outcome explicitly. Rust's skip is safer than C# partial removal;
   add an internal trigger-plan test separately if this path becomes reachable.
8. **Parallel mutable C# rows:** use copied scope records in a future harness and
   record runtime/version/enumeration. A serial reimplementation would test an
   interpretation, not the unmodified original's concurrency behavior.

The paper supports strength/eligibility admission and targeted top-down evolution,
but delegates baseline construction to earlier CA work. It does not resolve
collection ordering, collision ties, reuse of sibling words below a node,
minimum branch size, persistent expected-word representation, or empty replacement.
These remain scientific ambiguities; no production correction is implemented.

## Execution results

`cargo fmt --all -- --check`: passed.

`cargo test -p realtime-seca-core --test parity_hkt_reconstruction`: **11 passed**.

`cargo clippy -p realtime-seca-core --all-targets -- -D warnings`: passed.

`cargo test -p realtime-seca-core --lib`: **74 passed**.

`cargo test -p realtime-seca-core --test seca_light_stream`: **12 passed**.

`cargo test -p realtime-seca-core`: interrupted after several minutes (exit 130).
All 74 unit tests and three large-dataset baseline tests passed; two chunked
large-dataset tests were still running. No failures observed; this is not a
completed full-suite pass. A separate process-batch integration run was also
interrupted (exit 130) with 12 tests passed and its large-dataset sliding-window
test still running. Workspace-wide tests were not run; no architectural changes
were made.

 Initial focused execution: nine passed and one
fixture expectation failed. The word-order fixture correctly produced `{1,2}`
because adding word 2 covers every source of the already created word-1 node.
The expectation was corrected to capture that order-dependent merge.
