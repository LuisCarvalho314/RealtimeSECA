# SECA architecture research, iteration 2

Start with [baseline.md](baseline.md), [scientific_equivalence.md](scientific_equivalence.md),
[rust_architectures.md](rust_architectures.md), [evolution_validation.md](evolution_validation.md),
[benchmark_results.md](benchmark_results.md), and [recommendation.md](recommendation.md).
The new executable implementations are research only. They neither modify production
nor call its computations internally. The production-only evaluation adapter is a
separate process and package.

## Reproduce against the correct production version

This directory may be viewed from the older research branch. **Do not benchmark
against that branch's production source.** Use an isolated checkout pinned to the
production commit recorded in baseline.md, and copy this directory into its
`docs/architecture-exploration/iteration-2/`. Import the previous research material
using the command in baseline.md if absent. Keep the older fixtures unchanged.

Commands below run from that pinned repository root. They use existing cached
Rust dependencies; omit `--offline` only if dependency retrieval is needed.

```sh
cargo test --locked --workspace
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --offline --manifest-path docs/architecture-exploration/iteration-2/rust-candidates/Cargo.toml
cargo fmt --manifest-path docs/architecture-exploration/iteration-2/rust-candidates/Cargo.toml -- --check
cargo clippy --offline --manifest-path docs/architecture-exploration/iteration-2/rust-candidates/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path docs/architecture-exploration/iteration-2/rust-candidates/Cargo.toml
cargo build --offline --release --manifest-path docs/architecture-exploration/iteration-2/evaluation-baseline/Cargo.toml
python3 docs/architecture-exploration/iteration-2/scientific/exact_checks.py
python3 docs/architecture-exploration/iteration-2/evaluate.py
python3 docs/architecture-exploration/iteration-2/evaluate_evolution.py
```

The benchmark report specifies final measurement commands and boundaries. Evaluation
scripts write only this iteration's results. Source hashes pin the measured code;
changing candidate code requires rebuilding and rerunning the comparisons.

Standalone candidate construction accepts JSON on stdin:

```json
{"records":[{"source":1,"word":1},{"source":2,"word":1},{"source":1,"word":2}],"alpha":0.7,"beta":0.5,"branchMinimum":2}
```

Run `rust-candidates/target/release/seca-research-candidates`, adding `--masks` for
the competing construction kernel. Lifecycle requests use `construction`,
`initial_index`, `batches`, `gamma`, and the three error thresholds; complete
examples are retained in the raw differential results. These normalized numeric
IDs deliberately exclude production token normalization, external-source hashing
and public snapshot compatibility. Rust integration tests are executable API
examples. The candidate's serde round trip is a research restart format, not a
schema-4 production snapshot.

Read-only C# execution requires the separately preserved Mono reference harness
and assemblies. This investigation does not redistribute or edit them. The
captured reference results and exact reproduction commands are documented in
baseline.md and the existing reference-harness report.
