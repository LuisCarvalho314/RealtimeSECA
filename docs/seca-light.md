# SECA-Light

Reference: Al Sulaimani and Starkey (2023), IEEE Access, [DOI 10.1109/ACCESS.2023.3331219](https://doi.org/10.1109/ACCESS.2023.3331219), sections V–VII.

## Persistent update

```sh
realtime-seca-cli update batch-0.json --config seca_light_config.json --state-out model.json --dump-tree-verbose tree-0.json
realtime-seca-cli update batch-1.json --config seca_light_config.json --state-in model.json --state-out model.json --dump-tree-verbose tree-1.json
```

The initial nonempty batch constructs a baseline. Subsequent calls restore the complete engine from the schema-3 snapshot and use targeted reconstruction. Batch indices must increase; dates have no meaning inside the engine. Source IDs must be stable and unique; duplicate active IDs are ignored. The caller supplies a stream-wide ingestion ledger if previously forgotten IDs must never be reintroduced.

The CLI requires SlidingWindow, positive `max_batches_in_memory` (gamma), paper Option1 metrics, and matching builder/update alpha and beta. `seca_light_config.json` demonstrates the paper case-study thresholds and gamma=3. General library defaults remain available for older callers; they do not automatically configure the production Light update path.

State is written by temporary-file/fsync/rename. Batch processing clones the engine before mutation so errors leave the prior instance intact. Serialize concurrent writers externally. Unknown snapshot versions and mismatched configurations fail; legacy schema-2 metadata-only snapshots cannot resume. To change configuration, explicitly replay archived tokenized batches into separate state. The caller owns snapshot history and input archiving.

## Paper equations and evolution

Word strength is scoped document frequency divided by maximum word document frequency. Node eligibility is the fraction of node sources containing that word. Alpha-Error averages `max(0, alpha-strength1)` over the old vocabulary; Beta-Error averages `max(0, beta-eligibility1)`. Word-Importance-Error is one minus the updated normalized importance of the old vocabulary, with old and newly eligible words in the denominator. Comparisons use full precision and strict greater-than thresholds. Options2/3 are legacy alternatives, not production paper equations.

Traversal starts at Seed-HKT and passes parent-adopted source scopes to children. A significant error replaces that HKT and descendants and stops descending there. Otherwise the structure survives and child containers are inspected. Rebuilding considers all retained batches and the new batch, excluding words adopted on the ancestor path. Candidate vocabulary is kept separate from old vocabulary until a reconstruction decision is made.

After update, retain exactly the latest gamma batches. Pruning removes expired source payloads, legend entries, node/word membership and pending memberships and updates counts and mirrored indexes. Raw text, metadata and timestamps are not retained in active payloads. Empty batches after bootstrap advance the window without resetting topology. Empty scopes do not trigger errors from undefined denominators.

Forgetting source memory does not independently delete learned nodes or HKTs. A later targeted reconstruction may remove obsolete structure. Gamma bounds batches; it does not bound tree size, vocabulary or sources in a single batch. Historical removed-subtree payloads are not retained inside Light state; save output snapshots externally. The paper's cutoff wording is ambiguous by one batch; this implementation interprets its “up to gamma batches” as exactly gamma including the current batch.

## Implementation and tests

`engine/baseline.rs` builds once; `scope_mapping.rs` maps; `trigger.rs` computes/evaluates changes top-down; `rebuild.rs` handles selective replacement; `engine/mod.rs` commits updates and prunes; `snapshotting.rs` persists full state. The CLI `src/update.rs` exposes that engine, and the core `examples/update.rs` includes the same module for offline fixture verification.

`tests/seca_light_stream.rs` covers persisted continuation, no-change topology, child-only rebuild, subtree shrinkage, gamma=3 over many batches, restart equivalence, duplicates, empty batches, failure isolation and rejection of old schemas. Reports include inspected/reconstructed HKT IDs, sources added/forgotten, active count and detailed metrics/threshold reasons in notes.
