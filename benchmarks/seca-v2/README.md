# Independent V1/V2 comparison

Build the research example (V1 is a dev dependency only):

```bash
cargo build -p realtime-seca-v2 --release --example compare_stream
python3 benchmarks/seca-v2/compare.py PREPARED_G NEW_OUTPUT_G
python3 benchmarks/seca-v2/compare.py PREPARED_H NEW_OUTPUT_H
python3 benchmarks/seca-v2/summarize.py NEW_OUTPUT_G
```

Prepared data follows `benchmarks/seca-paper/datasets/prepare.py`; see
`docs/seca-paper-benchmark.md` for pinned upstream bytes, preparation and scientific
limits. Actual executed paths and hashes are in evidence/*.json. Outputs must be
new directories. Both engines receive identical ordered files and configs from
the existing benchmark. Full and gamma=3 Light each run 12 batches by default.
V1 uses its public API only; V2 independently converts the same token inputs.

`cycle-NN.json` preserves verbose tree, decisions and timing. Structural comparison
sorts descriptor/source-labelled recursive nodes, eliminating incidental IDs and
sibling order. Assignment comparison uses document co-member sets, not event IDs.
Raw engine decisions retain their original scope identities/paths; reconstructed
scope counts are compared without pretending structurally different trees share
an inspection universe. Mathematical and explicit discrepancy tests run with
`cargo test -p realtime-seca-v2`.

The existing indirect clustering adapter and evaluation library score current
batch populations with contextual merge threshold 1/2. Direct overlap/unassigned
counts remain diagnostic: no invented hard-partition conversion. Pairwise false
merges count predicted co-members with different truth labels; fragmentation
counts same-truth pairs assigned to different events. Results are reconstructed
protocol measurements, not exact replication or causal isolation of E01–E07.

GNU `time -v` (`gtime` on macOS, `/usr/bin/time` on Linux) measures process peak
RSS independently for each engine/mode.
Per-cycle timings include update plus public export/JSON-value conversion; V1
also computes its standard diagnostics and V2 does not reproduce them. Input
parsing, disk output and V2 checked serialization round trips are outside those
timings, but inside process RSS. No claim of pure-core speed or memory is made.
Runs are single observations and may contend with workspace tests. Compare
scientific quality separately from runtime. Raw evidence can be regenerated;
compact results, assignment archives, resources and source/data hashes are kept.
