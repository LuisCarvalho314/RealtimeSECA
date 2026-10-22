# Validation evidence — 2026-10-22

Environment: Linux x86_64, Rust 1.92.0, Python 3.11.4. Baseline and expected
content hashes are in `provenance.json`. Production sources and both existing
parity fixture directories compare unchanged against `d7e6e65`.

## Deliberate perturbation, then restoration

Temporarily changed only `tests/support/behaviour_oracle.rs` before constructing
the engine:

```diff
-    for (name, config) in stream.configs {
+    for (name, mut config) in stream.configs {
+        config.seca_thresholds.alpha_error_threshold = 0.09; // TEST-ONLY PERTURBATION
```

The committed input still specifies 0.11. This test-only override exercises the
real engine: batch 2's root alpha error is 0.09999999999999998, so root
reconstruction replaces acceptance plus selective child reconstruction.

Command:

```sh
cargo test --offline --locked -p realtime-seca-core --test behaviour_oracle
```

Observed exit status 101; archived continuation passed, and both fresh-run
comparison tests failed:

```text
strict_public_contract_matches_frozen_baseline ... FAILED
length at full/1/strict/report/decision_diagnostics
  left: 1
 right: 2
scientific_structure_matches_frozen_baseline ... FAILED
length at full/1/structural/decisions
  left: 1
 right: 2
test result: FAILED. 1 passed; 2 failed; 0 ignored
```

Indices in diagnostic paths are zero-based (step 1 is batch 2). Archived
continuation passes because it reads the original archived configuration; the
probe changes fresh engine construction only.

A separate temporary report-boundary probe set batch 9's
`report.reconstruction_triggered = true` before observation. Both comparison tests
failed at `full/8/strict/report/reconstruction_triggered` and
`full/8/structural/reconstruction_triggered` (actual true, expected false).
This narrower probe demonstrates reporting sensitivity without changing topology.

Restored the original observer after each probe. Neither probe touched production
code or expected values. The restored focused run passed all three tests. A fresh
stdout capture was byte-identical to `expected.json`; its SHA-256 remains
`f074b97e6895ae892b44b562580f9d5adf4dd85a8766a7d25118480ab6e9fef5`.

## Existing checks

Successful commands (offline Cargo uses the already cached pinned dependencies):

```sh
cargo fmt --all -- --check
cargo clippy --offline --locked --workspace --all-targets -- -D warnings
cargo build --offline --locked --workspace --examples --bins
SECA_DATASET_SOURCE_DIR=/home/azureuser/risklive-dev/seca-baseline/benchmarks/seca-paper/datasets/data/raw \
  python3 -m unittest discover -s benchmarks/seca-paper/datasets -v
python3 -m unittest discover -s benchmarks/seca-paper/clustering -v
python3 -m unittest discover -s benchmarks/reproduction -v
SECA_TEST_ENGINE="$PWD/target/debug/realtime-seca-cli" \
  python3 -m unittest discover -s benchmarks/seca-paper/evaluation -t benchmarks/seca-paper -v
```

Python: datasets 13, clustering 21, external-reference runner contracts 5,
evaluation including real-engine integration 25; all passed, no skips. Dataset
files were read from the existing cache and verified by the suite; no network
download or external C# harness was used. No research worktree files were changed.

The complete `cargo test --offline --locked --workspace` run finished successfully:
190 passed, 0 failed, 2 existing strict C# parity tests ignored for their documented
known discrepancies. This includes the large-dataset regressions and all three
new oracle tests. The restored focused oracle run also passed 3/3.
