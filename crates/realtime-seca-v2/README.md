# Realtime SECA V2

Independent recursive research implementation. See
[specification](../../docs/v2/specification.md),
[ownership and invariants](../../docs/v2/architecture.md), and
[comparison procedure](../../benchmarks/seca-v2/README.md).

```rust
use realtime_seca_v2::{Config, Model, Source};
let mut model = Model::new(Config { window: Some(3), ..Config::default() })?;
let source = Source { id: "doc-1".into(), words: ["quake".into()].into() };
let baseline_decisions = model.update(0, vec![source])?;
assert!(baseline_decisions.is_empty());
let decisions = model.update(1, vec![])?; // empty batch advances retention
let tree = model.root();
# Ok::<(), String>(())
```

Inputs are already normalized, case-sensitive token sets. Source IDs must be
nonempty and batch positions consecutive from zero. Returned decisions contain
full-precision errors and the inspected scope's node-index path. The model owns
the resulting tree; callers borrow it with `root()`. Serialize `Model` using Serde
and deserialize it to resume with invariant validation. No filesystem API,
legacy snapshot format, event clustering, or application ingestion is in the core.

## Command-line interface

```sh
seca-v2-cli update --batch batch.json --config config.json --state state.json
seca-v2-cli show-last --state state.json
```

`batch.json` has `{"batch":0,"sources":[{"id":"article-001","words":["nuclear","reactor"]}]}`.
`config.json` is the V2 `Config` JSON shape, for example
`{"alpha":0.7,"beta":0.5,"error_limits":{"alpha":0.1,"beta":0.2,"importance":0.3},"minimum_branch_sources":2,"window":3}`.
`window` is `null` for full SECA or a positive gamma for SECA-Light. These values
are examples, not prescribed RiskLive settings. RiskLive supplies normalized,
case-sensitive words. The CLI collects them into V2 token sets and calls V2's
validation and update logic. Source IDs and tokens must be nonempty; batches
must be consecutive from zero. Duplicate IDs follow V2's first-retained-source
rule. RiskLive owns identity deduplication beyond the forgetting window.

The state file is one JSON object with `engine: "realtime-seca-v2"`, `version: 1`,
`model` (the complete resumable V2 model), and `last_result` containing `batch`
and native V2 `decisions`. Incompatible versions, engines, invalid models and
configuration mismatches are rejected. The state file is the authoritative
committed output. `show-last` reconstructs the last report from it if the
process fails after committing state but before stdout is delivered.

On success, `update` emits one JSON line on stdout with `schema_version: 1`,
`batch`, `next_batch`, `retained_source_count`, `root` (recursive native V2 HKT),
and `decisions` (each has a structural `path`, full-precision `errors`, and
`reconstructed`). Paths are positions in an HKT, not durable node identities.
Errors go to stderr. The CLI writes a temporary file beside the state file,
syncs it, then atomically replaces the state file; a failed update or write
before replacement leaves the previous state intact. A directory sync error
after replacement can be reported even though the new state is committed;
inspect `show-last` in that case. The caller must prevent concurrent writers
to the same state path. RiskLive can later store the state and report together
in its own transaction.
