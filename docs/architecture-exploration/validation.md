# Validation record

2026-10-26, current working checkout at `c6cc0ac` with the pre-existing changes
identified in [comparison.md](comparison.md). No production edits were made by
this investigation.

| Check | Result |
|---|---|
| `cargo test --workspace` before experiments | Passed: 171 passed, two existing strict legacy parity tests ignored |
| `cargo test --workspace` after experiments | Passed: 171 passed, same two ignored |
| `cargo fmt --all -- --check` before/after | Passed |
| `cargo clippy --workspace --all-targets -- -D warnings` before/after | Passed |
| C# harness `verify.py` | Passed metrics, topology, subsequent cycles, deterministic repeats and invalid-input failure |
| Context fixture `replay.py` | Passed, including expected reference failures |
| Read-only construction reference re-execution | 21/21 captured JSON outputs unchanged |
| Python direct tests | Seven passed |
| Python mask tests | Seven passed |
| Common live-builder differential | Each candidate and simplified Rust: 143/143 construction cases; five/five larger workloads |
| Direct error aggregation over captured C# measurements | 21/21 within four-decimal rounding envelope |
| Standalone Rust adapter release build | Passed offline |
| Standalone adapter formatting and Clippy | Passed |
| `git diff --check` | Passed |

Post-workspace results by suite: CLI unit 3; CSV workflow 4; core unit 74;
large dataset 5; context parity 20 plus one ignored; HKT parity 12 plus one
ignored; scientific context 13; scientific HKT 4; HKT ordering 4; state invariants
20; streaming 12. Large-data tests took about 229 seconds and scientific-context
tests about 71 seconds. These are test durations, not performance benchmarks.

The two ignored tests deliberately expose known C# differences. Successful
characterization tests do not establish strict cross-language equivalence.
Neither prototype ran full evolution parity: those stages are absent. There are
no prototype persistence or failure-injection claims. The available simplified
checkout was evaluated only for construction, not its complete workspace suite.

Source provenance is in [results.json](results.json). Execution logs for the
post-workspace suite and context replay are in `/tmp/seca-exploration-post-tests.log`
and `/tmp/seca-exploration-context-replay.log` in this environment; these paths are
ephemeral, while the commands and summarized outcomes are retained here.

No local paper PDF was found in the RiskLive checkout. The paper-derived equations
were checked against the repository's scientific specification and existing
executed research audits; direct DOI access failed through the web tool in this
run. Independent full-paper/case-study reproduction remains uncompleted scientific
validation, not something established by the prototype comparisons. Library
assessments use linked primary documentation in [architecture_candidates.md](architecture_candidates.md).
