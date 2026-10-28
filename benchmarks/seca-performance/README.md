# Three-way SECA computational benchmark

This benchmark invokes the original **Release `SECA.exe` assembly's** public
`SECAAlgorithm` methods, `SecaEngine` (Rust v1), and the independent `Model`
(Rust v2). The C# application entry point only initializes WinForms; the adapter
does not use its database, GUI, tokenizer, or serial research harness. Original
algorithm classes and their `Parallel.ForEach` calls are unmodified. The
adapter's token-to-integer encoding, word-frequency preparation, and JSON parsing
finish before the first timed call. The original C# project declares
`Optimize=true` for its Release configuration and targets .NET Framework 4.7.2;
the local Release assembly is a prerequisite and is not redistributed.

## Reproduce

From the repository root on macOS with Mono 6.12, `mcs`, Rust, and Python 3:

Set `SECA_ORIGINAL_ASSEMBLY_DIR` to the original project's `bin/Release`
directory when it is not at the runner's default path. C# runs require both
the Release assemblies there and the `SECA_DataClass` sources two directories
above for provenance; Rust-only runs do not.

```sh
python3 benchmarks/seca-performance/run.py \
  --theme G --mode Full --batches 12 --warmup 1 --repetitions 10 \
  --output benchmarks/seca-performance/results/G-Full
python3 benchmarks/seca-performance/plot.py benchmarks/seca-performance/results/G-Full
python3 benchmarks/seca-performance/verify_inputs.py --theme G \
  --output benchmarks/seca-performance/results/G-Full/input-verification.json
```

`--theme H` selects Tweets H; `--mode SlidingWindow --engines v1 v2` runs the
gamma-three Rust Light comparison. The script rejects C# Light because its
forgetting policy is different. The `--batches` option produces increasing
stream-length experiments from the same prefix. Do not run competitors together;
the runner executes every engine and repetition sequentially. The `.build/`
directory is ignored. Each output directory receives provenance, raw JSONL
measurements, summary JSON and CSV. `plot.py` generates SVG figures without
third-party Python packages. Summary p95 uses nearest rank; standard deviation
is sample SD.

Synthetic stress inputs can be regenerated without external data:

```sh
python3 benchmarks/seca-performance/run_synthetic.py \
  --input-dir /private/tmp/seca-synthetic \
  --output-dir benchmarks/seca-performance/results/synthetic
```

The six synthetic scenarios separately increase document count, vocabulary
size, token overlap, input word levels, or batch vocabulary drift against a
fixed seed. Actual hierarchy depth and reconstruction counts are *outputs*;
the generated word levels and drift are controlled input factors, not promises
that a particular algorithm will construct a particular shape. Generated
batch files are deliberately kept outside Git.

The runner verifies every selected batch's SHA-256 against the committed
MStream-derived G/H manifest. It also verifies batch index, document count,
source-ID uniqueness, document order, token-set vocabulary size, and a canonical
SHA-256 of ordered `(batch, source ID, sorted unique tokens)` records. All three
adapters load the exact same batch paths. News G uses 8,880 records from `News`
in 12 batches of 740; Tweets H uses 24,000 records from `Tweets-T` in 12 batches
of 2,000. These are documented reconstructions, not recovered exact paper
partitions; see `docs/seca-benchmark-datasets.md`.

The C# API requires integer source and word IDs. Its adapter assigns each
external document ID and token a stable, collision-free integer in first-appearance
order across the whole stream; inverse mapping recovers the exact external IDs
and token sets. Rust receives the external strings directly. `verify_inputs.py`
independently reconstructs these conversions and rejects any per-document
token-set mismatch, then compares order-sensitive hashes for all three paths.

## Timing and measurements

`algorithm_seconds` brackets each public baseline or update call. C# update
timing includes original `updateTree`, original `clearMemory` and original
`copyDataDictionary` state continuation; Rust v1 includes its normal report
diagnostics and transactional clone; Rust v2 includes its normal update.
Input parsing, row/set conversion, serialization and process startup are
outside this timer. Observation of tree shape, retained evidence and RSS occurs
after each timed call. `stream_wall_seconds` is elapsed time since the measured
stream began, including algorithm calls and after-batch observations, and
excluding preprocessing, process startup and C# warm-up. Process wall time
includes all parsing and observations and the C# internal warm-up, so use
stream wall or algorithm time for direct speedups.
For C#, `algorithm_cpu_seconds` uses Mono `Process.TotalProcessorTime`; for Rust,
it uses `getrusage` user plus system CPU time. CPU time can exceed wall time for
the original parallel C# algorithm. `rss_after_batch_kib` is the process's
resident high-water mark at the batch boundary (Mono `PeakWorkingSet64`, Rust
`getrusage.ru_maxrss`). This includes runtime, JIT, retained prepared inputs,
algorithm state and measurement adapter; it is **not** hierarchy size or isolated
model memory. `peak_rss_kib` is the maximum of these batch-boundary readings.
Very short transient peaks between boundaries may be missed by Mono reporting.

`source_memberships` sums node source-set sizes; a source may appear in multiple
nodes. `retained_documents` counts retained identities. C# reports
`removed_hkts` from the original dead-HKT bag. Those include descendants, so
the C# reconstructed *target* count and inspected count remain null. Rust
reconstruction counts are target decisions, not removed descendants. Exact
reconstruction duration is unavailable without changing algorithm internals;
the plots show frequency only. No per-HKT timing normalization is inferred.

Each measured C# process first warms its JIT on cloned input for the baseline
and first incremental update, drops that temporary engine, and runs GC before
the measured stream. This warm-up is separate from the runner's process-level
`--warmup` repetitions. Process wall time includes this internal warm-up;
algorithm times do not. The C# code uses `Parallel.ForEach` with default Mono thread-pool scheduling;
Rust v1 and v2 are single-threaded. The C# adapter discards diagnostic
`Console.WriteLine` output inside timed calls to avoid terminal I/O, leaving
computations intact. C# source enumeration and concurrent mutation can make
tree counts vary across runs. The C# application originally uses a SQL-backed
input and `SourceCleanerEngine` preprocessing; this benchmark supplies the same
already-preprocessed MStream token sets to all three algorithms. It does not
claim equivalence to the original application's ingestion or default dataset.

Configuration is the existing `benchmarks/seca-v2/evidence/Full-config.json` or
`SlidingWindow-config.json`: alpha 0.7, beta 0.5, branch minimum 10, alpha
error 0.1, beta error 0.2, importance error 0.3. C# uses Option1 error choices.
Its original four-decimal error rounding, candidate handling, tie behavior,
source caches and concurrency remain original; Rust v1 and v2 follow their
documented contracts. The three trees are **different computational workloads**.

The original C# assembly has no Git metadata in its local source directory;
`provenance.json` records the exact assembly SHA-256 instead of inventing a
source commit. The Rust Git SHA, compiler/runtime versions, config and dataset
hashes are recorded there. New runs also record the Rust adapter source and
compiled Rust binary hashes. Results from another machine must be summarized
separately; direct speedup ratios require same-host runs.

The committed News G Full evidence combines a C# pass and a corrected Rust
pass on the same host. `merge_measurements.py` checks matching source revision,
machine, dataset/config hashes and repetition count before combining their
raw rows; the combined provenance names both passes. The corrected Rust
adapter moves an owned `SourceBatch` into v1, so no caller clone is timed.
