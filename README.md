# RealtimeSECA

Rust contextual analysis and persistent SECA-Light engine. The `realtime-seca-core` crate builds a baseline once, maps successive ordered source batches onto existing HKTs, measures paper-defined changes, selectively replaces subtrees and forgets old source memberships without resetting the learned tree.

See [SECA-Light lifecycle and CLI](docs/seca-light.md). Existing `baseline`, `timeline`, `timeline-many` and CSV conversion commands remain available. RiskLive's production stream uses `update` rather than independently rebuilding daily baselines.

```sh
cargo test -p realtime-seca-core
cargo test -p realtime-seca-cli
cargo build --release -p realtime-seca-cli
```
