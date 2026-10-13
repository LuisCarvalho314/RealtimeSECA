# Development rules

- Read [docs/algorithm.md](docs/algorithm.md) before changing algorithmic behavior.
- Read [docs/architecture.md](docs/architecture.md) before changing module boundaries
  or state ownership; preserve [docs/state.md](docs/state.md) invariants.
- Preserve paper correspondence: paper → scientific specification → architecture
  → Rust. Report unresolved scientific ambiguities; never silently change semantics.
- Maintain one canonical SECA-Light update path. Do not restore placeholder metrics,
  full/hybrid rebuild fallbacks, compatibility policies, or experimental scaffolding.
- Prefer deleting unnecessary code to adding abstractions. Keep modules cohesive
  and functions understandable; review files over 400 lines and functions over 80.
- Keep algorithmic logic independent of ingestion, filesystem I/O, HTTP, and presentation.
- Make narrowly scoped changes unless explicitly tasked with restructuring.
- Add focused regression tests for behavior changes, including state/failure invariants.
- Update the relevant scientific, architecture, or state document when its contract
  changes. Do not duplicate those documents here.
- Run relevant builds/tests before reporting completion. For architectural changes,
  run `cargo fmt --all -- --check`, `cargo test --workspace`, and
  `cargo clippy --workspace --all-targets -- -D warnings`; report limitations accurately.
