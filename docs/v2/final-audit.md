# SECA v2 specification-freeze audit

**Audit basis:** `7fc0ba5c0df4b00f00245414de19ee9ad29fb763`, 27 October 2026. The v2 implementation was unchanged during this audit. This is an audit of the explicit v2 interpretation, not a claim of exact C# output parity.

## Verified evidence

- Production `recursive::construct` ranks admitted words by descending scoped document frequency and exact UTF-8 token bytes. Its beta comparison uses strict `>` between overlap ratios, so an equal best score keeps the first-created eligible node. The source file was unchanged from the basis commit for the benchmark build.
- News G Full was rerun over 11,109 upstream MStream News records in 16 batches from a release build of that commit. The first three prepared batches each contain 740 documents. The prepared first-batch SHA-256 is `aecee350c56d5402e5c552c8a4ce7f827063b4f2f01cfd2d5974f3dbca60a005`; the saved C# reference SHA-256 is `9d941e7c3fdd754554888860a6f8735396c793e791b179119824a58aba521037`. The first-cycle canonical Rust tree SHA-256 is `1cc62e6ab654ebb987034bc3878abda4216d2ba4859ff6467aa7d89d31f58d88`. [Provenance and all three tree hashes](../evidence/seca-v2-ordering/provenance.json) record the reproduction.
- The previous committed [cycle CSV](../evidence/seca-v2-ordering/news-g-cycle-by-cycle.csv) had reported 8, 20, and 14 matched-path descriptor mismatches. It matched the saved temporary **last-wins** trace row for row; probing that run reproduced the previous 42-row HKT CSV exactly. The verified first-wins build gives **7, 20, and 13**, respectively. Both derived CSVs were regenerated. The extra first-cycle mismatch was `REFUGE / REFUGE / ordinary:deal`.
- First-cycle comparison has 42 matched HKT paths, 7 different ordinary descriptor partitions, and one path with a different observed source union. In cycle 3, the C# root omits 297 incoming sources from its stored membership; all 297 are present in descendants. C# records 30 reconstruction triggers and v2 records 20. These are snapshot observations, not a causal attribution of the trigger gap.
- Full News G v2 completed all 16 cycles: final tree 926 HKTs and 3,850 nodes, 11,109 retained documents, 575 reconstructed scopes, mean B-cubed F1 `0.640057320638627`. Timing and peak-memory comparisons are outside this freeze audit; the sandbox prevented macOS `time -l` from querying `sysctl`.

## Causal status and scientific choices

The [original C# candidate-order trace](../evidence/seca-v2-ordering/news-g-black-friday-csharp-construction-trace.json) for `black → friday` proves the local order mechanism: the original compiled reference and traced source copy agree; replaying its candidate order in Rust reproduces the observed beta scores and node memberships. [Construction-order analysis](../seca-v2-construction-order.md) records the interpretation. Sibling descriptor exclusion is strongly supported as the cause of several other baseline vocabulary differences, but those paths lack a direct event-level causal trace. The nested refuge partition, later trigger differences, and the first within-update cause of C# parent-membership gaps remain unresolved. A C# discrepancy alone is not evidence of a bug.

The residual-bearing branch minimum, ancestry-path exclusion, fresh evidence frequencies, first-wins beta ties, and parent–child membership consistency are explicit v2 rules. The papers do not uniquely settle the branch counted set, sibling exclusion, or equal-frequency order. Eight strict C# parity fixtures are active; four (`branch_boundary`, `overlap_recursive`, `collision_true_tie`, `refuge`) are intentionally ignored and document these output differences.

## Implementation review and preprocessing

For ordinary child recursion, at least one descriptor is added to the ancestry exclusion set, so the eligible vocabulary decreases. A Refuge child adds no descriptor, but it cannot recurse with the identical source scope: any nonempty residual scope has a maximum-frequency eligible word, at least that word is admitted for valid `alpha ∈ [0,1]`, and its source IDs leave the next refuge. Thus a consecutive Refuge chain strictly decreases its finite scope. No termination defect was found and no regression test or algorithm change was justified.

The benchmark adapter receives externally prepared MStream `textCleaned`, splits it on whitespace, and passes token lists to v2. V2 converts them to token sets. The adapter does no further stemming, lemmatization, stop-word filtering, case folding, or cleaning. The upstream preparation method for `textCleaned` is not established here.

## Freeze disposition

**Correctness blockers:** none identified in the audited v2 contract. This does not establish exact C# parity or settle publication ambiguities.

**Optional engineering improvements:** stronger state-restoration validation, longer SECA-Light window and empty-batch checks, and reproducible memory measurement outside this sandbox.

**Future scientific experiments:** instrument aligned C# and v2 event-level scopes, matches, candidate histories and errors; test alternative branch counted sets and sibling exclusion; evaluate quality and calibration over longer streams. These experiments may motivate a later versioned scientific decision, but do not change this freeze candidate.

## Verification

Run from the repository root after the evidence and documentation changes:

```sh
cargo fmt --all -- --check
cargo test -p realtime-seca-v2
cargo clippy -p realtime-seca-v2 --all-targets -- -D warnings
```

All three commands passed in the writable audit checkout. `cargo test -p realtime-seca-v2` ran 32 passing tests: 3 comparison, 2 construction-order, 8 active C# parity, and 19 mathematics tests; 4 C# parity tests were intentionally ignored. No test failed. The format check produced no diff, and clippy completed with `-D warnings`.
