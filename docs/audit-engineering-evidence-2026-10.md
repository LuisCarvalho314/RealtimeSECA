# Engineering audit evidence, October 2026

This report supplements the two historical parity reports and
[performance.md](performance.md). Target: fetched main
`2bcd8edf045b38c1b02437f7672178b42991ac07`. No production fix is included.

## State agent evidence

Commit `342e72caed6f99ce37c512f5da2b21f96a038520`
contains `crates/realtime-seca-core/tests/seca_light_invariants.rs` (18 tests).
The agent supplied no separate Markdown report or committed execution log. Its
assertions must not be treated as passing merely because they were committed.

During consolidation the exact committed test file was copied to an external
Cargo harness, depending on the **unchanged clean-main core** by path. This avoids
adding tests or altering implementation files in the documentation commit.

```bash
mkdir -p /tmp/seca-audit-evidence/tests
git show 342e72c:crates/realtime-seca-core/tests/seca_light_invariants.rs \
  > /tmp/seca-audit-evidence/tests/seca_light_invariants.rs
```

These reproduction commands require the original local agent Git objects, which
are not published by this documentation-only push. The test outcomes and corruption
datasets above/below are preserved here independently of transient logs.

Harness `/tmp/seca-audit-evidence/Cargo.toml`:

```toml
[package]
name = "seca-audit-evidence"
version = "0.0.0"
edition = "2021"
[dependencies]
realtime-seca-core = { path = "/tmp/seca-consolidated-audit/crates/realtime-seca-core" }
serde_json = "1"
```

```bash
cargo test --offline --manifest-path /tmp/seca-audit-evidence/Cargo.toml \
  --test seca_light_invariants
```

Result: **14 passed, 4 failed**, exit 101. All four failed at the assertion
`corrupted snapshot was accepted`:

| Test | Injected corruption | Observed result |
|---|---|---|
| `snapshot_rejects_invalid_expected_word` | Unknown expected word ID 2147483647 | Restore accepted |
| `snapshot_rejects_invalid_top_word` | Unknown presentation top-word ID | Restore accepted |
| `snapshot_rejects_invalid_membership_word` | Unknown word-membership key, with valid source members | Restore accepted |
| `snapshot_rejects_expired_retained_evidence` | Consistent envelope/current index 5, gamma=3, still retaining batch 1 | Restore accepted |

The 14 passing cases cover cache completeness, root uniqueness, child attachment,
unknown descriptor words, memberships inside node sources, retained source references,
inverse identity consistency, live allocator bounds, schema rejection, invalid-update
isolation, initialization overwrite rejection, duplicate first-wins, expired-ID
readmission, append-only vocabulary, repeated reconstruction, fresh IDs and JSON
restart equivalence. These fixtures support those specific cases, not every
possible corrupt snapshot or arbitrary stream.

Restore code: `crates/realtime-seca-core/src/engine/validation.rs`,
`validate_restored_state`; envelope checks: `engine/snapshotting.rs`, `load_snapshot`.
The failures predate the audit; the tests reveal missing checks rather than creating
invalid state during a normal update. No successful post-fix result is claimed here.

The HKT agent's unchanged test file from `432c47c` was run in the same external
harness: **11 passed, 0 failed**. The file is retrieved analogously; command:

```bash
cargo test --offline --manifest-path /tmp/seca-audit-evidence/Cargo.toml \
  --test parity_hkt_reconstruction
```

Local raw logs: `/tmp/seca-audit-state.log`, `/tmp/seca-audit-hkt.log`.
They are session artifacts, not committed dependencies of this report.

## CSV agent evidence and clean-main reproduction

Commit `a5a485f6fa3e202996d3b9cdfd7f4af60d661a64`
contains production changes and four CLI workflow regression tests; the production
changes are **excluded** from this audit commit. There is no separate committed
agent Markdown report. The four tests were executed on the integration checkout
before the reporting-only task and passed. That checkout includes the agent fix;
this is evidence for the proposed fix, not the behavior of current main.

Clean main was built with:

```bash
cargo build --offline --manifest-path /tmp/seca-consolidated-audit/Cargo.toml \
  -p realtime-seca-cli
```

Two one-row CSVs, independently converted with batch indexes 0 and 1:

```csv
Title,URL,Timestamp
market news,https://example.org/a,1970-01-01T00:00:00Z
```

The second file uses URL `https://example.org/b`; title and timestamp are unchanged.
Both conversions returned `source_id = "row_000001"` and
`timestamp_unix_ms = null`, while preserving URL/Timestamp as metadata. Persistent
CLI updates with SlidingWindow gamma=3 and matching builder/update alpha=beta=.5
returned:

| Input | sources_processed | active_source_count |
|---|---:|---:|
| First file, initialize batch 0 | 1 | 1 |
| Second file, restore and update batch 1 | 0 | 1 |

Thus a distinct URL article is silently discarded by retained-ID deduplication.
Source: `crates/realtime-seca-cli/src/ingestion.rs`,
`convert_csv_to_source_batch` and `parse_timestamp_to_unix_ms`;
core `engine/input.rs`, `prepare_batch`.

Reproduction files live under `/tmp/seca-audit-evidence/`: `a.csv`, `b.csv`, their
converted JSON, `config.json`, state/tree and update reports. Commands use
`from-csv INPUT OUTPUT --batch-index INDEX`, then
`update INPUT --config CONFIG --state-out STATE --dump-tree-verbose TREE`, with
`--state-in STATE` for the second invocation. Current CLI requires SECA-Light
configuration and the tree output flag; initial attempts without these requirements
failed and are not evidence of an engine defect.

The agent's proposed scheme uses trimmed `url:` identities and
`csv-row:<conversion UUID>:<batch>:<row>` fallbacks, plus explicit-offset timestamp
parsing. It avoids cross-file fallback collisions and supports saved-JSON replay;
reconverting anonymous articles gives new identities. Aliased URLs and lifetime
article deduplication remain application questions. No fix was integrated here.

## Validation scope

The historical context agent ran on old main `42c07c0`, not current main. Its
reported 154-test run preceded its final nineteenth fixture. The HKT/performance
agents ran core checks on `15814d1`, which is included in current main; their
large-dataset runs were interrupted. These results cannot be combined into a
workspace-wide pass. Integration checks begun under the earlier task were stopped
when the reporting-only task replaced it; their modified checkout does not validate
this audit target. Clean-main checks above are the consolidation execution evidence.
