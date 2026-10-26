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
