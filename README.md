# RealtimeSECA

A Rust implementation of streaming hierarchical contextual analysis based on
[Al Sulaimani and Starkey (2023)](https://doi.org/10.1109/ACCESS.2023.3331219).
The engine builds a Hierarchical Knowledge Tree (HKT), maps tokenized sources,
measures contextual change, selectively reconstructs affected subtrees, and
persists the resulting model. SECA accumulates source evidence; SECA-Light uses
the same update algorithm with a gamma-sized batch retention window.

## Build and validate

```sh
cargo build --release --workspace
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## Persistent SECA-Light updates

```sh
cargo run -p realtime-seca-cli -- update batch-0.json \
  --config seca_light_config.json --state-out model.json \
  --dump-tree-verbose tree-0.json
cargo run -p realtime-seca-cli -- update batch-1.json \
  --config seca_light_config.json --state-in model.json --state-out model.json \
  --dump-tree-verbose tree-1.json
```

The first nonempty batch constructs the baseline. Subsequent batch indices must
be consecutive. An empty update advances the retention window. Supply stable
source IDs; duplicates among retained sources are ignored. Input JSON contains
`batch_index` and `sources`; each source contains `source_id`, `batch_index`,
`tokens`, and optional `text`, `timestamp_unix_ms`, and `metadata`.
The engine expects tokenized input; tokenization policy belongs to ingestion.

`seca_light_config.json` selects gamma=3 and matching construction/update alpha
and beta. Reports distinguish trigger decisions on previous scopes from
observations on the resulting tree. State files contain the complete resumable
model; tree exports are presentation artifacts. The CLI stages and atomically
renames the state file. Serialize concurrent writers externally.

Snapshots use schema **4**. Earlier snapshots require explicit replay of archived
input. Configuration now has three error thresholds; legacy option selectors
and placeholder policies are rejected. Copy the supplied configurations when
migrating. Keep the same configuration throughout a persistent stream.

Other CLI commands provide baseline construction, timelines, CSV ingestion,
console presentation, and serving an exported tree. Run the binary without
arguments for usage. The library also supports accumulated SECA evidence through
`MemoryMode::Full`; reconstruction remains selective in both retention modes.

Read [the scientific specification](docs/algorithm.md),
[the implementation map](docs/architecture.md), and
[the state invariants](docs/state.md). [AGENTS.md](AGENTS.md) governs development.
