# SECALight behavioural regression oracle

This is a compatibility record of unmodified `d7e6e65`, not a proof of scientific
correctness. E01–E03 and E05–E07 remain pending. No C# goldens, production code,
or schema definitions change. Ordinary CI needs only the existing Rust tests.

The input, expected observations and four schema-4 snapshots are committed under
`crates/realtime-seca-core/tests/fixtures/behaviour_oracle/`. Capture provenance is
in `provenance.json`; validation and deliberate-failure evidence is in
`validation.md` there. The harness uses public construction, update, export,
reporting and snapshot APIs, with no private algorithm helpers or reimplementation
of metric formulas. Its independence comes from archived baseline outputs: a
future implementation cannot establish equivalence merely by agreeing with its
own restarted copy.

## Comparison contract

`tests/behaviour_oracle.rs` runs three independently named checks:

- **Strict public compatibility:** exact parsed JSON values for the full update
  report (including trigger and display diagnostics, reasons and notes), both tree
  exports, explanation, word statistics, stored batch count, schema and batch
  position, vocabulary, identity/provenance indexes and retained token batches.
  IDs, ordering, nullability and strings are preserved. Object key order and JSON
  whitespace are immaterial. Floats have no rounding or tolerance. This deliberately
  freezes observed public output, including presentation details; a failure here
  alone does not establish a scientific change.
- **Scientific structure:** rooted HKT/node attachments, ordered nodes, descriptor
  and top-word tokens, expected words, refuge status, external source memberships,
  and contextual word memberships (including non-descriptor candidates). Decisions
  retain scope, output scope, reconstruction targets, reasons, counts and all three
  trigger errors. Only allocation IDs are replaced with rooted node-position paths,
  word IDs with exact tokens, and source hashes with exact external identities.
  Membership sets are sorted by external identity; no memberships are removed or
  merged. Descriptors, node order and decisions are not sorted away. A typed refuge
  marker avoids colliding with a real token. Traversal checks every HKT is reached.
- **Archived continuation:** restore the baseline implementation's schema-4 JSON
  snapshots after batches 2 and 6, in each mode, and compare every subsequent report,
  export and structural observation against the frozen uninterrupted run. Repeat
  JSON restoration after each continued update. Immediate round trips compare the
  whole snapshot exactly. Current captures also verify round trips at every step.

Strict ID equality is an engineering compatibility fence. The structural check
separately answers whether topology, membership, identity and decisions survived
an intentional allocation change. Neither test may be silently relaxed to make a
simplification pass. Snapshot layouts are archived as **readable old inputs**;
the harness does not demand that every future freshly emitted snapshot have the
same private field layout. Schema-4 loading and continuation must still work.

Before batch 3 the harness submits a valid prefix followed by an empty source
identity, requires rejection and exact parsed-JSON equality of the entire serialized
snapshot, records the error text, and then submits the valid batch. This detects
leaked vocabulary, identity, membership and allocator changes observable in state
or continuation. Comparisons report the first differing JSON path.

## Coverage and audit

The new stream reuses the two-topic construction/selective-update recipe from
`seca_light_stream.rs`; it does not copy the existing threshold and corruption
case matrices. The same nine small batches run with Full memory and gamma=3:

| Batch | Frozen observation |
| --- | --- |
| 1 | Baseline with two topic branches, descriptors, expectations, memberships and identities |
| 2 | Root accepted, one child reconstructed, root membership propagated and other sibling preserved; first checkpoint |
| 3 | Rejected malformed update followed by valid input; retained identity ignored, first duplicate wins, trimmed/repeated/empty tokens normalized; root reconstruction |
| 4–6 | Empty batches; Full retains evidence, gamma=3 removes batches 1–3 exactly, vocabulary/topology persist; second checkpoint after complete expiry |
| 7 | Old source identity ignored in Full and readmitted with new evidence in gamma=3 |
| 8 | New dominant evidence; different selective/root reconstruction decisions in the two modes |
| 9 | Accepted continuation with learned vocabulary and stable hierarchy |

Existing coverage remains responsible for these complementary cases:

| Existing suite | Reused coverage; reason it is insufficient alone |
| --- | --- |
| `seca_light_stream.rs` (12 tests) | Persistence, 30 updates, selective child/sibling preservation, forgetting, restart, duplicates and failure isolation. Assertions mostly check selected properties or compare two executions of the current engine. |
| `seca_light_invariants.rs` | Full serialized source/topology/cache/allocator invariants, repeated reconstruction, gamma 1/3 readmission, atomic rejection and corruption. Valid invariants alone allow a changed but internally consistent algorithm. |
| `snapshot_validation.rs` | Schema-4 corruption, window boundaries including saturation, Full history, token references and restart. Current-engine round trips do not test reading archived baseline state. |
| `parity_context_metrics.rs` | Executed C# fixture comparisons and explicit known differences, strict precision boundaries, expected/candidate history, shared-word and node-union cases. Already supplies the small metric edge-case matrix. |
| `parity_hkt_reconstruction.rs` | 21 executable fixture cases covering construction, ties, collisions, recursion, root/child reconstruction and sequential updates. Already freezes Rust topology observations and C# difference classifications. Its optional `SECA_RECORD_RUST_AUDIT` writer is separate from this oracle. |

The C# strict parity tests remain ignored with their existing documented reasons;
reference disagreement is neither hidden nor resolved here. The new oracle never
uses the external C# executable, a recording environment flag, `fs::write`, or an
expected-file path in its capture program. Expected JSON is compiled into the test
with `include_str!`. Running tests cannot update it.

## Capture and review

To inspect fresh observations without modifying fixtures:

```sh
cargo run --locked -p realtime-seca-core --example capture_behaviour_oracle > /tmp/seca-observed.json
cargo test --locked -p realtime-seca-core --test behaviour_oracle
```

The example only prints to stdout. Baseline capture was run twice in separate
processes and compared byte for byte. Changing expected values requires an
explicit version-control edit with a new provenance record and review of both
comparison layers. Do not recapture from a changed implementation and call the
result baseline equivalence. There is no automatic accept/update mode.

## Remaining gaps

This is one deterministic ordered synthetic stream in two memory modes, not an
exhaustive input space. It does not establish permutation invariance, portability
of ties, collision-probing coverage, arbitrary-depth selective propagation,
long-horizon drift, or equivalence for every scientific proposal. Existing parity
and invariant suites cover selected related cases, not general proofs. The stream
does not force every trigger reason independently or simulate failures after
validation/reporting; it has no filesystem crash/concurrent-writer injection.
It excludes ingestion, HTTP, performance and clustering quality. Exact floating
comparisons were validated on Linux x86_64 with the pinned toolchain; other
platforms may expose numerical differences requiring investigation. Successful
regression equivalence preserves these observations and known limitations, and
must not be presented as scientific correctness or approval of pending decisions.
