# SECA-Light streaming performance baseline

This is a public-API, release-build synthetic baseline, not a real-news validation
or a real-time readiness claim. No production implementation was changed.

## Environment

Measured on 2026-10-13 at engine revision
`15814d1d636a853acbf0ca950d44cce4c534b3e4` (the supplied checkout,
branch `audit/hkt-reconstruction-parity`; not independently verified against remote
main). CPU: AMD EPYC 7763 under Microsoft virtualization, 8 visible logical CPUs,
4 cores / 2 threads per core, one NUMA node, reported 2.445 GHz, 32 MiB L3.
Host RAM: 31 GiB, no swap. Cgroup v1 reports unlimited memory and CPU quota.
Ubuntu 20.04.6 LTS, Linux `5.15.0-1089-azure`, x86_64; rustc 1.92.0
(`ded5c06cf`, LLVM 21.1.3). Cargo's standard optimized release profile;
no CPU pinning or exclusive machine access. Core tests overlapped part of the
matrix: wall-clock timings are exploratory, with possible contention. Each
configuration has one fresh process, not repeated independent trials. Quantiles
summarize sequential updates, not a stationary distribution or confidence interval.

## Workload and measurement contract

`crates/realtime-seca-core/examples/benchmark_stream.rs` uses only public APIs and
existing serde_json. Batch zero initializes the baseline, followed by **30 or 90
updates** (31 or 91 total batches). All sources have unique batch-qualified IDs;
there is no deduplication shortcut. SplitMix64 with seed 42 generates four equally
represented sectors. A source has two common tokens, one sector/topic anchor, and
20 draws with replacement from its sector pool; tokens are sorted and deduplicated.
There is no text, timestamp, or metadata payload, tokenizer, network or disk I/O
inside the timed API operations. This models tokenized short news items, not their
full ingestion cost.

The vocabulary parameter (256 here) is the size of the **available sector word
pool**, rounded down to a multiple of four, plus common/topic tokens. Not all pool
words necessarily appear in a small batch. Each pool slides by
`round((vocabulary / 4) * (1 - overlap))` positions per batch: overlap is a nominal
pool overlap, not a guarantee of observed document vocabulary overlap. Stable uses
0.95 (three new positions per sector per update); changing uses 0.5 (32). On every
second changing batch, four fresh topic anchors replace the usual anchors; normal
anchors return next batch. `--novel-every 0` disables this variation.

All runs use construction alpha/beta 0.5, update alpha/beta 0.5, and branch minimum
32; unlike the production default of 1, this limits tiny synthetic branches.
`--branch-min 1` reproduces the default branching configuration. Stable runs set
all three error thresholds to 1 (minimal reconstruction); changing runs set them
to 0 (frequent reconstruction). These are explicit supported engine parameters,
not alternative update paths. A matched changing-vocabulary threshold comparison
below separates threshold effects from vocabulary effects. Actual reconstruction
frequency is measured; zero thresholds do not force a reconstruction when all
available errors are zero.

Gamma=3 retains at most three batches of source evidence after pruning. Full
(`--gamma 0`) accumulates all evidence. These have **different memory and model
semantics**; gamma is not a semantics-preserving performance switch. Vocabulary
and learned topology persist under gamma pruning. Reconstruction precedes pruning,
so it can temporarily operate on four batches of evidence.

CSV fields:

- `process_ms`: the complete `process_batch`, including transaction clone,
  trigger traversal, reconstruction, pruning, diagnostics, commit and old-state
  disposal. Input generation and the copy used for diagnostic replay are excluded.
- `diagnostics_ms`: separate public `display_diagnostics(Some(&incoming))` replay
  on the just-committed engine; includes its clone and traversal. It repeats the
  reporting work with the same final state and input, but allocator/cache conditions
  differ from its invocation inside process_batch. It is an estimate, not an exact
  decomposition of the process timer.
- `clone_ms`: standalone public engine Clone on post-update state; destruction is
  outside the timer. The transaction's input state is smaller/different, so this
  is a cost probe, not its exact internal duration.
- `snapshot_ms`: public `snapshot()` creation (including engine clone).
  `json_ms`: compact `serde_json::to_vec` on that snapshot, timed separately.
  Snapshot persistence cost before filesystem work is their sum. JSON allocation
  is included; destruction, export, CSV output and filesystem writes are excluded.
- `snapshot_bytes`: serialized compact schema-4 snapshot length; `active_sources`,
  `vocabulary`, `hkts`, `nodes`: report and untimed verbose tree export counts.
- `reconstructed`, `targets`: whether this batch reconstructed and number of
  selected roots, not the number of newly created HKTs.
- `rss_kib`, `hwm_kib`: Linux `/proc/self/status` VmRSS and VmHWM immediately
  after processing, before additional probes/exports. Empty on unsupported systems.
  HWM includes earlier probes, and RSS includes allocator-retained storage and the
  diagnostic input copy. Neither measures just engine-owned memory or instantaneous
  peak inside one stage. Extra probes may affect subsequent allocator/cache state.

Median, p95 and max use nearest-rank order statistics. Baseline construction is
reported separately on stderr and excluded from update quantiles. Individual
mapping, metrics, pruning and reconstruction stage timers are unavailable through
public APIs; no interfaces were added. Diagnostics clone and traversal cannot be
split further with this benchmark.

## Reproduction

Run from the repository root. The following reproduces the matrix sequentially,
with a 90-second **whole-process** limit per configuration (including probes and
serialization), not an update deadline. CSV preserves completed updates when a
run times out; stderr summary is absent on incomplete runs.

```bash
cargo build --release -p realtime-seca-core --example benchmark_stream
mkdir -p /tmp/seca-performance
for batches in 30 90; do
  for sources in 25 100 400; do
    for workload in stable novel; do
      if [ "$workload" = stable ]; then
        overlap=0.95; novel=0; threshold=1
      else
        overlap=0.5; novel=2; threshold=0
      fi
      for gamma in 3 0; do
        name="$workload-$batches-$sources-g$gamma"
        timeout 90s ./target/release/examples/benchmark_stream \
          --batches "$batches" --sources "$sources" --vocabulary 256 \
          --overlap "$overlap" --novel-every "$novel" \
          --gamma "$gamma" --threshold "$threshold" --branch-min 32 --seed 42 \
          > "/tmp/seca-performance/$name.csv" \
          2> "/tmp/seca-performance/$name.log"
      done
    done
  done
 done
for threshold in 0 1; do
  timeout 90s ./target/release/examples/benchmark_stream \
    --batches 30 --sources 100 --vocabulary 256 --overlap 0.5 \
    --novel-every 2 --gamma 3 --seed 42 --branch-min 32 \
    --threshold "$threshold" \
    > "/tmp/seca-performance/ablation-$threshold.csv" \
    2> "/tmp/seca-performance/ablation-$threshold.log"
done
```

For exact token stability use `--overlap 1 --novel-every 0`. For the initial
production-branching pilot use `--branch-min 1 --batches 30 --sources 100
--overlap 0.95 --novel-every 0 --gamma 3 --threshold 0.1 --seed 42`.
Its baseline took 5304.769 ms; no update completed before manual cancellation.
That pilot had no predefined timeout, so no numeric update bound is inferred.
The matrix does not establish performance for default branching or default error
thresholds (0.1, 0.13, 0.3); `--threshold` intentionally sets all three alike.

## Results

All latencies are milliseconds. “Completed” counts updates, excluding baseline. Partial rows are censored at the whole-process timeout and **not comparable full-run quantiles**. Stable = mostly stable vocabulary/minimal reconstruction; novel = changing vocabulary/frequent reconstruction.

| Workload | Updates requested/completed | Sources/batch | Memory | Process median | p95 | Max | Reconstructed batches | Final sources | Vocabulary | HKTs/nodes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| stable | 30/30 | 25 | gamma=3 | 3.68 | 4.15 | 4.15 | 0/30 | 75 | 620 | 1/1 |
| stable | 30/30 | 25 | Full | 9.47 | 17.22 | 17.76 | 0/30 | 775 | 620 | 1/1 |
| novel | 30/30 | 25 | gamma=3 | 8.20 | 13.09 | 13.54 | 0/30 | 75 | 4057 | 1/1 |
| novel | 30/30 | 25 | Full | 15.53 | 29.38 | 30.06 | 0/30 | 775 | 4057 | 1/1 |
| stable | 30/30 | 100 | gamma=3 | 15.88 | 16.75 | 16.82 | 0/30 | 300 | 622 | 2/5 |
| stable | 30/30 | 100 | Full | 45.74 | 84.75 | 87.57 | 0/30 | 3100 | 622 | 2/5 |
| novel | 30/30 | 100 | gamma=3 | 60.04 | 104.16 | 108.61 | 30/30 | 300 | 4162 | 6/152 |
| novel | 30/30 | 100 | Full | 150.24 | 249.87 | 272.46 | 30/30 | 3100 | 4162 | 8/435 |
| stable | 30/30 | 400 | gamma=3 | 211.68 | 360.68 | 374.78 | 0/30 | 1200 | 622 | 39/1149 |
| stable | 30/30 | 400 | Full | 851.74 | 1120.15 | 1214.97 | 0/30 | 12400 | 622 | 39/1149 |
| novel | 30/30 | 400 | gamma=3 | 1283.30 | 1740.86 | 1931.66 | 30/30 | 1200 | 4162 | 180/5850 |
| novel | 30/18 | 400 | Full | — | — | — | 18/18 | 7600 | 2602 | 732/22003 |
| stable | 90/90 | 25 | gamma=3 | 4.57 | 6.00 | 6.27 | 0/90 | 75 | 1339 | 1/1 |
| stable | 90/90 | 25 | Full | 26.12 | 50.01 | 51.74 | 0/90 | 2275 | 1339 | 1/1 |
| novel | 90/90 | 25 | gamma=3 | 18.83 | 33.63 | 35.77 | 0/90 | 75 | 11708 | 1/1 |
| novel | 90/90 | 25 | Full | 45.30 | 90.63 | 94.76 | 0/90 | 2275 | 11708 | 1/1 |
| stable | 90/90 | 100 | gamma=3 | 17.82 | 20.99 | 22.47 | 0/90 | 300 | 1342 | 2/5 |
| stable | 90/90 | 100 | Full | 84.61 | 247.78 | 258.32 | 0/90 | 9100 | 1342 | 2/5 |
| novel | 90/90 | 100 | gamma=3 | 96.92 | 155.36 | 229.69 | 90/90 | 300 | 11962 | 6/166 |
| novel | 90/90 | 100 | Full | 316.49 | 719.47 | 846.12 | 90/90 | 9100 | 11962 | 11/1180 |
| stable | 90/90 | 400 | gamma=3 | 120.04 | 313.47 | 336.34 | 0/90 | 1200 | 1342 | 39/1149 |
| stable | 90/84 | 400 | Full | — | — | — | 0/84 | 34000 | 1270 | 39/1149 |
| novel | 90/49 | 400 | gamma=3 | — | — | — | 49/49 | 1200 | 6630 | 183/5827 |
| novel | 90/25 | 400 | Full | — | — | — | 25/25 | 10400 | 3510 | 951/28778 |

### Separate costs

Medians are separate order statistics; their ratios are descriptive and their sums need not equal a measured combined median. Snapshot + JSON is outside process_batch. All configurations, including censored prefixes, are listed to retain observed cost evidence.

| Configuration (workload-updates-sources-memory) | Diagnostics median | Clone median | Snapshot creation median | JSON median | JSON p95 | JSON max | Final snapshot MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| novel-30-100-g0 | 67.92 | 2.33 | 2.57 | 2.71 | 4.30 | 4.54 | 3.094 |
| novel-30-100-g3 | 25.98 | 0.77 | 0.72 | 0.73 | 0.87 | 1.19 | 0.467 |
| novel-30-25-g0 | 7.52 | 0.63 | 0.66 | 0.63 | 1.14 | 1.25 | 0.616 |
| novel-30-25-g3 | 3.64 | 0.27 | 0.26 | 0.19 | 0.30 | 0.31 | 0.155 |
| novel-30-400-g0 | 1889.64 | 46.76 | 67.03 | 71.60 | 144.91 | 144.91 | 45.142 |
| novel-30-400-g3 | 853.05 | 15.78 | 17.47 | 18.06 | 23.38 | 24.07 | 9.674 |
| novel-90-100-g0 | 128.35 | 4.59 | 5.37 | 5.31 | 10.88 | 17.40 | 9.040 |
| novel-90-100-g3 | 52.17 | 0.86 | 0.85 | 0.93 | 1.33 | 1.58 | 0.692 |
| novel-90-25-g0 | 21.93 | 1.70 | 1.83 | 1.90 | 3.51 | 3.80 | 1.839 |
| novel-90-25-g3 | 8.44 | 0.46 | 0.49 | 0.47 | 0.78 | 0.91 | 0.361 |
| novel-90-400-g0 | 1161.85 | 29.01 | 39.32 | 39.45 | 92.59 | 98.01 | 59.604 |
| novel-90-400-g3 | 755.45 | 11.39 | 10.36 | 10.55 | 14.18 | 16.08 | 9.597 |
| stable-30-100-g0 | 22.77 | 2.15 | 2.03 | 2.27 | 3.85 | 4.47 | 2.265 |
| stable-30-100-g3 | 6.78 | 0.46 | 0.47 | 0.37 | 0.46 | 0.47 | 0.237 |
| stable-30-25-g0 | 4.63 | 0.59 | 0.57 | 0.56 | 0.97 | 1.25 | 0.517 |
| stable-30-25-g3 | 1.47 | 0.15 | 0.15 | 0.11 | 0.14 | 0.17 | 0.066 |
| stable-30-400-g0 | 493.83 | 16.87 | 21.16 | 34.68 | 82.06 | 87.67 | 22.309 |
| stable-30-400-g3 | 110.92 | 2.20 | 2.34 | 2.53 | 6.08 | 6.95 | 1.289 |
| stable-90-100-g0 | 43.22 | 3.16 | 3.72 | 3.60 | 12.82 | 13.26 | 6.701 |
| stable-90-100-g3 | 7.71 | 0.49 | 0.49 | 0.47 | 0.56 | 1.36 | 0.257 |
| stable-90-25-g0 | 12.93 | 1.45 | 1.53 | 1.60 | 2.90 | 3.00 | 1.523 |
| stable-90-25-g3 | 1.93 | 0.17 | 0.17 | 0.16 | 0.19 | 0.21 | 0.084 |
| stable-90-400-g0 | 328.27 | 15.40 | 20.70 | 27.97 | 41.15 | 43.19 | 38.305 |
| stable-90-400-g3 | 59.29 | 1.06 | 1.20 | 1.52 | 4.10 | 5.54 | 1.311 |

### Memory and snapshot trends

RSS/HWM are MiB converted from Linux KiB; snapshots are MiB converted from bytes. These observations include benchmark overhead and allocator retention.

| Workload | Sources/batch | Memory | Run length | Observed batch | Active sources | Vocabulary | Snapshot MiB | RSS MiB | HWM MiB | Process ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stable | 100 | gamma=3 | 90 | 1 | 200 | 274 | 0.154 | 4.09 | 4.09 | 11.52 |
| stable | 100 | gamma=3 | 90 | 30 | 300 | 622 | 0.237 | 5.19 | 5.19 | 17.00 |
| stable | 100 | gamma=3 | 90 | 90 | 300 | 1342 | 0.257 | 5.69 | 5.69 | 20.75 |
| stable | 100 | Full | 90 | 1 | 200 | 274 | 0.154 | 4.20 | 4.20 | 11.09 |
| stable | 100 | Full | 90 | 30 | 3100 | 622 | 2.265 | 23.30 | 23.30 | 57.32 |
| stable | 100 | Full | 90 | 90 | 9100 | 1342 | 6.701 | 58.53 | 58.53 | 257.76 |
| novel | 100 | gamma=3 | 90 | 1 | 200 | 390 | 0.331 | 5.00 | 5.00 | 33.69 |
| novel | 100 | gamma=3 | 90 | 30 | 300 | 4162 | 0.467 | 8.09 | 8.23 | 73.33 |
| novel | 100 | gamma=3 | 90 | 90 | 300 | 11962 | 0.692 | 11.93 | 11.93 | 99.72 |
| novel | 100 | Full | 90 | 1 | 200 | 390 | 0.331 | 4.89 | 4.89 | 32.16 |
| novel | 100 | Full | 90 | 30 | 3100 | 4162 | 3.094 | 27.18 | 27.18 | 173.65 |
| novel | 100 | Full | 90 | 90 | 9100 | 11962 | 9.040 | 82.99 | 82.99 | 719.47 |
| stable | 400 | gamma=3 | 30 | 1 | 800 | 274 | 4.054 | 20.13 | 20.13 | 323.68 |
| stable | 400 | gamma=3 | 30 | 30 | 1200 | 622 | 1.289 | 34.14 | 34.14 | 168.94 |
| stable | 400 | Full | 30 | 1 | 800 | 274 | 4.054 | 20.27 | 20.27 | 478.13 |
| stable | 400 | Full | 30 | 30 | 12400 | 622 | 22.309 | 123.43 | 145.28 | 1214.97 |
| novel | 400 | gamma=3 | 30 | 1 | 800 | 390 | 8.177 | 35.58 | 35.58 | 1108.57 |
| novel | 400 | gamma=3 | 30 | 30 | 1200 | 4162 | 9.674 | 95.98 | 95.98 | 1695.83 |
| novel | 400 | Full | 30 | 1 | 800 | 390 | 8.177 | 35.55 | 35.55 | 947.74 |
| novel | 400 | Full | 30 | 18 | 7600 | 2602 | 45.142 | 231.81 | 274.59 | 3875.00 |
| stable | 400 | gamma=3 | 90 | 1 | 800 | 274 | 4.054 | 20.28 | 20.28 | 316.83 |
| stable | 400 | gamma=3 | 90 | 30 | 1200 | 622 | 1.289 | 31.98 | 31.98 | 106.39 |
| stable | 400 | gamma=3 | 90 | 90 | 1200 | 1342 | 1.311 | 31.98 | 31.98 | 128.40 |
| stable | 400 | Full | 90 | 1 | 800 | 274 | 4.054 | 20.39 | 20.39 | 316.43 |
| stable | 400 | Full | 90 | 30 | 12400 | 622 | 22.309 | 125.39 | 147.87 | 452.58 |
| stable | 400 | Full | 90 | 84 | 34000 | 1270 | 38.305 | 239.07 | 277.58 | 972.99 |
| novel | 400 | gamma=3 | 90 | 1 | 800 | 390 | 8.177 | 35.64 | 35.64 | 633.61 |
| novel | 400 | gamma=3 | 90 | 30 | 1200 | 4162 | 9.674 | 97.99 | 97.99 | 1089.88 |
| novel | 400 | gamma=3 | 90 | 49 | 1200 | 6630 | 9.597 | 97.99 | 97.99 | 1401.75 |
| novel | 400 | Full | 90 | 1 | 800 | 390 | 8.177 | 35.72 | 35.72 | 606.40 |
| novel | 400 | Full | 90 | 25 | 10400 | 3510 | 59.604 | 309.53 | 367.82 | 3799.02 |

### Matched reconstruction threshold comparison

Changing vocabulary, 30 updates, 100 sources, gamma=3, identical seed and input; only all three error thresholds differ. Reconstruction changes subsequent topology, so this comparison measures the resulting whole update paths, not pure reconstruction stage time.

| Threshold | Process median/p95/max ms | Diagnostics median ms | Reconstruction batches | Final HKTs/nodes | Snapshot MiB |
|---:|---:|---:|---:|---:|---:|
| 0 | 42.82/70.81/78.85 | 19.62 | 30/30 | 6/152 | 0.467 |
| 1 | 16.38/22.24/22.80 | 7.34 | 0/30 | 2/5 | 0.315 |

## Observed costs and practical limits

The matched threshold experiment measured 42.82 ms median with reconstruction
in every update versus 16.38 ms with none. The final trees differed (152 versus
5 nodes), and diagnostics replay rose from 7.34 to 19.62 ms. This establishes a
cost difference for the complete reconstruction-enabled path, not an isolated
reconstruction timer. At 25 sources, even threshold zero produced **no**
reconstruction: the one-node topology and its available error measurements matter.
“Novel” names the input workload; actual reconstruction counts govern comparisons.

Final-tree diagnostics are the strongest directly measured performance target.
For the 30-update stable 400-source Full run, replay median was 493.83 ms versus
851.74 ms for process_batch; in the novel gamma=3 run it was 853.05 ms versus
1283.30 ms. These are separate measurements, so they cannot be subtracted to
produce exact stage timings. They nevertheless establish that the operation is
substantial. It includes a clone, topology sweep, input remapping and metric work;
this benchmark does not identify the hottest function within that operation.

Engine cloning is measurable but much smaller than diagnostics in those runs:
16.87 and 15.78 ms median respectively. Snapshot creation was 21.16 and 17.47 ms;
JSON encoding added 34.68 and 18.06 ms outside the update. Neither snapshot work
nor standalone cloning dominates the observed processing latency. JSON output
becomes a material additional persistence cost for large Full snapshots (up to
144.91 ms observed in the censored 30-update novel Full run). Filesystem durability
and restore cost were not measured.

Scaling is not just proportional to incoming source count. Stable gamma=3
30-update medians were 3.68, 15.88 and 211.68 ms for 25, 100 and 400 sources, with
1, 5 and 1149 final nodes. Novel gamma=3 at 400 sources reached 5850 nodes and
1931.66 ms maximum. The default-branching pilot's 5.3-second baseline is additional
evidence that construction configuration and resulting topology need explicit
workload coverage. The internal builder/reconstruction/metrics costs cannot be
attributed individually through this example.

Repeated 100-source gamma=3 updates completed both horizons: maximum process
latency was 22.47 ms for stable and 229.69 ms for novel in the 90-update runs.
These measured configurations are plausible for repeated batches arriving at
intervals comfortably above the measured processing plus persistence and ingestion
cost. They are not evidence for production defaults, real-news inputs, sustained
unbounded operation, latency guarantees or any particular real-time deadline.
The 400-source novel run already has second-scale latency. Four matrix runs were
censored: novel 30/400/Full at 18 updates, stable 90/400/Full at 84, novel
90/400/gamma=3 at 49, and novel 90/400/Full at 25. Timeout includes explicit
measurement probes, so it is not a 90-second bound on engine update time alone.

Cross-run timing variability is visible: stable gamma=3 at 400 sources had a
211.68 ms median over 30 updates and 120.04 ms over 90; the identical 30-update
novel 100-source gamma=3 input measured 60.04 ms in the matrix and 42.82 ms in the
later ablation. Concurrent tests, scheduling, allocator/cache state and changing
state across horizons prevent a precise causal attribution. Structural counts and
snapshot sizes at identical prefixes agree; timing differences should be revisited
with repeated trials on an idle, pinned environment before setting an SLO.

## Memory growth

Full accumulation grew predictably in retained sources, but serialized state and
RSS depend on memberships and topology as well as payload count. For stable
100-source Full, batches 1/30/90 retained 200/3100/9100 sources, snapshots grew
0.154/2.265/6.701 MiB, and RSS grew 4.20/23.30/58.53 MiB. Novel Full at the same
horizon ended at 9.040 MiB and 82.99 MiB RSS. The censored novel 400-source Full
run reached 59.604 MiB serialized and 309.53 MiB RSS at batch 25.

Gamma=3 bounds retained sources to 300 or 1200, not all state. At 100 sources,
stable snapshots grew from 0.237 MiB at batch 30 to 0.257 MiB at 90; novel grew
0.467 to 0.692 MiB. Novel vocabulary grew from 4162 to 11962 entries even though
only 300 sources remained. RSS rose from 8.09 to 11.93 MiB. This follows the
append-only vocabulary contract and does not establish a leak. Longer novel
streams remain an open growth question.

Pruning can reduce serialized membership size without shrinking topology or
returning allocator storage to the OS: stable 400-source gamma=3 snapshots fell
from 4.054 MiB at batch 1 to 1.289 MiB at batch 30 while RSS rose. In the 90-update
run RSS then stayed at 31.98 MiB through batches 30 and 90. Thus neither current
snapshot size nor RSS alone estimates the live engine's exclusive heap size.
No per-allocation profiling was performed.

## Recommended performance work, ranked by measured evidence

1. **Prioritize final-tree diagnostics.** Its public replay consumes hundreds of
   milliseconds in the large cases and is much larger than a standalone clone.
   Profile that operation first and preserve its observational/reporting contract
   when optimizing it. The evidence supports the operation as a target, not a
   particular unmeasured caching or data-structure change.
2. **Investigate construction and reconstruction on large topologies.** The
   matched threshold comparison and node-count scaling establish substantial
   whole-path cost. Obtain internal stage profiles before selecting an
   implementation optimization; preserve canonical SECA semantics and selective
   subtree replacement. Changing branch limits or reconstruction thresholds is a
   model change, not an equivalent implementation optimization.
3. **Prioritize snapshot creation/encoding only for persistence-heavy workloads.**
   They are separately measured and increase with serialized state, but are smaller
   than processing/diagnostics in this matrix. Measure durable writes at the
   application boundary before ranking end-to-end persistence optimizations.

Transaction clone elimination is **not supported as the first priority** by these
measurements. Any future change must retain failure isolation. Gamma selection
must follow required evidence-retention semantics; choosing gamma=3 instead of
Full is not an equivalent algorithmic optimization. No optimization was implemented
and no unmeasured speedup is claimed.

## Validation and artifacts

```bash
cargo fmt --all -- --check
cargo test -p realtime-seca-core --example benchmark_stream
cargo test -p realtime-seca-core --lib --test seca_light_stream
cargo clippy -p realtime-seca-core --example benchmark_stream -- -D warnings
```

Release build and smoke run passed; example tests: 2 passed; core unit tests:
74 passed; streaming integration tests: 12 passed. Formatting and example clippy
passed. A broader `cargo test -p realtime-seca-core` invocation was interrupted
while two large-dataset streaming tests were still running (three of that suite's
five tests had passed); this is not a full-suite pass. No architecture or
production logic was changed, and workspace builds/tests are not claimed.

Per-batch CSV and parameter/summary logs are available in this session under
`/tmp/seca-performance`; only the assigned example and this report are committed.
The reproduction commands recreate those artifacts. Table process quantiles use
`sort(process_ms)[ceil(fraction * completed_count) - 1]`, with fractions
0.5/0.95/1.0; incomplete runs omit full-run process quantiles. All final state
figures refer to the last completed update, including censored runs.
