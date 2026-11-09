# Three-way SECA computational performance

## Scope and evidence

The primary comparison uses the reconstructed MStream News G stream: 8,880
documents in twelve ordered batches of 740. All three algorithms consumed the
same prepared JSON batch files after SHA-256 and token-set verification. The
original C# **Release assembly**, invoked through its public `SECAAlgorithm`
methods under Mono, is the C# comparator. This is not the serial instrumented
harness and not the WinForms application entry point. Rust v1 uses `SecaEngine`;
Rust v2 uses its independent recursive `Model`. All configurations have alpha
0.7, beta 0.5, branch minimum 10, and error limits 0.1/0.2/0.3.

The original application normally draws from SQL and its own preprocessing
components. The benchmark bypasses those steps to supply the recovered
MStream `textCleaned` tokens directly. The C# API's integer source/word IDs are
stable bijective encodings of the same external strings. Each engine receives
identical document order, token sets, batch boundaries and vocabulary. The
News G prefix is an explicit reconstruction, not a proven exact paper stream.

Timings enclose only baseline/update calls. C# update timing includes its
original continuation steps (`clearMemory` and `copyDataDictionary`); v1
includes its transactional clone and normal diagnostics; v2 includes only its
normal update. Parsing, set/row conversion, output serialization, shape
inspection and process startup occur outside algorithm timing. C# receives an
in-process baseline/first-update JIT warm-up on copied input, then a GC before
measurement. Rust Release builds are ahead-of-time compiled. Each competitor
and repetition runs sequentially. `stream_wall_seconds` includes post-batch
observations but excludes startup, preprocessing and C# JIT warm-up; process
wall also includes those stages.

The process resident-memory high-water mark includes prepared input,
runtime/JIT, algorithm state and adapter allocations. It does not represent
hierarchy size or isolate retained model memory. C# executes the original
`Parallel.ForEach` code under the default Mono thread pool; Rust v1 and v2 are
single-threaded. CPU seconds and the CPU/wall ratio expose that difference.

## Primary comparison: News G, Full SECA

Ten measured full-stream repetitions follow one discarded full-stream warm-up
per engine. Times below are seconds. Batch statistics use the 110 incremental
update observations per engine. RSS is the median of ten per-process peaks.
The C# observations come from the original three-way pass; the Rust observations
come from a later pass on the same machine and exact inputs after removing a
caller-side batch clone from the Rust timer. The combined provenance records
both passes. No measurements from different machines are combined.

| Metric | Original C# assembly | Rust v1 | Rust v2 |
| --- | ---: | ---: | ---: |
| Baseline construction, median | 0.3267 | 0.01269 | 0.005533 |
| Total algorithm time, median | 44.339 | 17.730 | 0.6499 |
| Total algorithm time, mean ± SD | 45.232 ± 5.650 | 17.722 ± 0.317 | 0.6509 ± 0.00599 |
| Total algorithm time, p95 | 57.071 | 18.298 | 0.6611 |
| Median incremental batch latency | 3.944 | 1.485 | 0.05627 |
| p95 incremental batch latency | 7.035 | 3.296 | 0.1053 |
| Measured stream wall time, median | 44.408 | 17.735 | 0.6535 |
| Throughput, documents / median algorithm second | 200 | 501 | 13,664 |
| Peak process RSS, median MiB | 139.0 | 59.39 | 61.20 |
| Final HKTs, median | 580 | 619 | 726 |
| Final nodes, median | 2,439.5 | 2,699 | 3,085 |
| Final source-membership entries, median | 58,743.5 | 68,277 | 69,734 |
| Reconstructed HKT targets, total median | unavailable | 372 | 413 |

Using the **median full-stream algorithm time** consistently:

| Comparison | Speedup (A time / B time) | Peak RSS ratio (A / B) |
| --- | ---: | ---: |
| Original C# → Rust v1 | 2.50× | 2.34× |
| Rust v1 → Rust v2 | 27.28× | 0.971× |
| Original C# → Rust v2 | 68.22× | 2.27× |

The original C# run used a median 265.5 CPU seconds for 44.3 elapsed
algorithm seconds (CPU/wall ratio 6.09); v1 and v2 each used approximately
one CPU second per elapsed second. C# therefore consumed substantially more
parallel CPU work while still having higher elapsed latency. Its final HKT
counts ranged from 570 to 614 and final node counts from 2,409 to 2,592
across the ten measured repetitions. V1 and v2 each produced stable counts
across their repetitions. The original C# concurrent execution introduces a
variable computational workload as well as timing variation.

### A. Original C# to Rust v1

V1 is 2.50× faster in median total algorithm time and has a 2.34× lower
median process RSS. It finishes with more HKTs, nodes and source-membership
entries than the C# median. The speedup therefore does not arise simply from
producing a smaller final hierarchy. The original C# path is parallel and
uses different candidate/error accounting and reconstruction details; the
result cannot be credited solely to Rust or to a single architectural change.

### B. Rust v1 to Rust v2

V2 is 27.28× faster while constructing a larger final hierarchy and
triggering more HKT reconstruction targets. Both are single-threaded Rust;
v1's transactional clone, lookup mirror, retained indexes and final diagnostic
sweep versus v2's direct recursive state and sets are concrete implementation
differences. V2 also changes scientific choices for child scopes, tie ordering,
candidate handling and ancestry exclusion. The measured gain establishes
the benefit of the **whole v2 path**, not the isolated cost of any one data
structure or formula. V1's median process peak is 1.80 MiB below v2's on G
Full; process RSS also includes their different prepared-input representations.

### C. Original C# to Rust v2

V2's total algorithm time is 68.22× shorter and the original C# process peak
is 2.27× v2's. V2's final HKT/node counts exceed the C# median by about 25%
and 26%, respectively. The gain reflects both different semantics/workloads
and implementation efficiency; no equal-tree control isolates those effects.

V2 has **more** final HKTs (17.3%), nodes (14.3%), source-membership entries
(2.1%) and reconstructed targets (11.0%) than v1 on this stream. Its 27.28×
time gain therefore is not explained by constructing a smaller final tree.
This does not quantify per-scope work or prove that each reconstruction is
equally expensive. It does show a large end-to-end computational benefit of
the v2 implementation and scientific contract taken together. V1 has the
lowest measured process peak on G Full (59.39 MiB), versus v2's 61.20 MiB;
the original C# peak is 2.34× v1's and 2.27× v2's in this environment.

Across News G stream prefixes, median cumulative v1/v2 speedup rises from
2.29× at 740 documents to 9.91× at 1,480, 18.69× at 4,440 and 27.28× at
8,880. This is measured scaling within one stream, not an asymptotic
complexity claim. All three accumulated more HKTs and memberships as batches
arrived; v2's elapsed time grew most favorably relative to the alternatives.

The [cumulative runtime](results/G-Full/plots/cumulative-runtime.svg),
[incremental latency](results/G-Full/plots/batch-latency.svg),
[memory](results/G-Full/plots/peak-memory.svg),
[HKT](results/G-Full/plots/hkts.svg),
[node](results/G-Full/plots/nodes.svg),
[reconstruction frequency](results/G-Full/plots/reconstruction-frequency.svg)
and [v1/v2 prefix speedup](results/G-Full/plots/v1-v2-speedup.svg) plots show
the median per-batch trajectories. Exact reconstruction cost is unavailable
from the public APIs and is not inferred from overall batch latency.

## Tweets H, Full SECA

The reconstructed Tweets H stream has 24,000 documents in twelve ordered
batches of 2,000 and 11,610 distinct tokens. Ten measured repetitions per
engine follow one discarded warm-up. Times are seconds; RSS is the median
per-process high-water mark.

| Metric | Original C# assembly | Rust v1 | Rust v2 |
| --- | ---: | ---: | ---: |
| Baseline construction, median | 1.216 | 0.09151 | 0.02190 |
| Total algorithm time, median | 360.517 | 77.030 | 2.7208 |
| Total algorithm time, mean ± SD | 380.543 ± 69.659 | 77.217 ± 1.085 | 2.708 ± 0.0417 |
| Total algorithm time, p95 | 489.772 | 79.605 | 2.777 |
| Median incremental batch latency | 28.909 | 6.737 | 0.2492 |
| p95 incremental batch latency | 80.638 | 14.502 | 0.4065 |
| Measured stream wall time, median | 360.625 | 77.048 | 2.731 |
| Throughput, documents / median algorithm second | 66.6 | 311.6 | 8,821 |
| Peak process RSS, median MiB | 356.46 | 170.50 | 180.64 |
| Final HKTs, median | 2,669.5 | 2,391 | 2,581 |
| Final nodes, median | 11,427.5 | 10,185 | 10,830 |
| Final source-membership entries, median | 244,138.5 | 240,045 | 242,860 |
| Reconstructed HKT targets, total median | unavailable | 124 | 127 |

| Comparison | Speedup (A time / B time) | Peak RSS ratio (A / B) |
| --- | ---: | ---: |
| Original C# → Rust v1 | 4.68× | 2.09× |
| Rust v1 → Rust v2 | 28.31× | 0.944× |
| Original C# → Rust v2 | 132.50× | 1.97× |

V2 finishes with 8.0% more HKTs, 6.3% more nodes, 1.2% more source-membership
entries and three more reconstructed targets than v1. Its 28.31× v1→v2
speedup cannot be explained by a smaller final hierarchy. Against C#, v2
finishes with about 3.3% fewer HKTs and 5.2% fewer nodes, so different work
may contribute to the 132.50× total gain. These counts cannot quantify that
contribution; source-scope sizes and reconstruction costs are not equal across
engines. C# used a median 2,456 CPU seconds per stream, a CPU/wall ratio of
6.78, and had a much wider elapsed-time SD than either Rust implementation.
The lowest measured Full-SECA process peak is v1's 170.50 MiB; v2 is 180.64
MiB and C# is 356.46 MiB. The memory comparison still includes each runtime
and prepared-input representation.

The H [runtime](results/H-Full/plots/cumulative-runtime.svg),
[batch latency](results/H-Full/plots/batch-latency.svg),
[memory](results/H-Full/plots/peak-memory.svg),
[HKT](results/H-Full/plots/hkts.svg),
[node](results/H-Full/plots/nodes.svg),
[reconstruction frequency](results/H-Full/plots/reconstruction-frequency.svg)
and [v1/v2 prefix speedup](results/H-Full/plots/v1-v2-speedup.svg) figures are
generated from the same raw H runs.

## Scientific and workload differences

C# rounds its Option1 errors to four decimals and uses mutable dictionaries,
concurrent collections, candidate caches, random GUID IDs and concurrent
traversal. Its output can vary with thread scheduling. V1 uses stable identities,
explicit scope mapping, retained indexes, transactional cloning, scoped
reconstruction and a post-update diagnostic sweep. V2 uses direct recursive
construction/evolution, one evidence map and `BTreeSet` source operations; it
recomputes document frequencies from retained evidence in each inspected
scope. V2 deliberately uses residual-bearing child thresholds, ancestry-path
word exclusion, and stable lexical tie ordering. The three implementations
therefore perform different amounts of work and may construct different HKTs.
No speedup here can be attributed solely to a programming language.

For SECA-Light, the original C# `clearMemory` cutoff retains only
`gamma-1` cycles at its strict boundary and can delete empty topology. Rust v1
and v2 retain exactly `gamma` batches and preserve learned descriptors/topology
when forgetting. C# Light is excluded from a numerically equivalent gamma-three
comparison. A C# Light timing would be a separately labelled policy comparison.

### News G, gamma-three SECA-Light

Ten measured Rust repetitions after one warm-up give median full-stream
algorithm times of **7.819 s** (v1) and **0.3478 s** (v2), a **22.48×** v1→v2
speedup. Median process peak RSS is 35.13 MiB (v1) and 33.34 MiB (v2), a
v1/v2 memory ratio of 1.054. V1 ends with 257 HKTs, 1,101 nodes, 13,902
source-membership entries and 192 reconstructed targets; v2 ends with 293,
1,236, 14,226 and 217 respectively. Both retain the gamma-three source
window. Here v2 is faster and uses less process memory despite a larger final
tree and more triggered targets. These figures compare the two Rust contracts,
not an equivalent C# forgetting policy.

### Tweets H, gamma-three SECA-Light

On the 24,000-document H stream, ten measured Rust repetitions give median
full-stream algorithm times of **35.318 s** (v1) and **1.2791 s** (v2), a
**27.61×** speedup. Median process peak RSS is **94.10 MiB** (v1) and
**87.59 MiB** (v2), a 1.074 v1/v2 ratio. Final HKTs/nodes are 854/3,303
for v1 and 909/3,475 for v2; reconstructed targets total 32 and 33.
V2 again has a larger final hierarchy but lower elapsed time and memory under
the same gamma-three retention rule.

The C# public API does not expose exact inspected-HKT counts, reconstructed
*target* counts or time spent inside reconstruction. Its dead-HKT bag contains
removed descendants as well as triggers, so it is recorded separately. Rust
v1/v2 expose inspected and triggered targets. No exact reconstruction-duration
figure is reported. Per-HKT timing is not used as a universal normalization:
scope sizes and source memberships vary substantially.

## Reproducibility and limits

`results/G-Full/provenance.json` records source SHA, original assembly hash,
dataset/config hashes, runtime/compiler versions, machine, warm-up and
repetition counts. `runs.jsonl` preserves every batch measurement. The summary
and ratios can be regenerated with `summarize.py`; SVG figures with `plot.py`.
The original C# source directory has no Git repository, so a source commit SHA
cannot honestly be supplied for it. The exact Release assembly SHA-256 and
source-file hashes identify the local comparator.

The saved measurements predate benchmark-source and Rust-binary hashes in the
runner. In particular, the G Full merged provenance records Rust adapter SHA-256
`44640a1e…`, while the adapter consolidated here has SHA-256 `29c7b24a…`.
The raw observations remain valid historical measurements, but their exact
Rust adapter build cannot be reconstructed from this commit alone. Future runs
record the adapter and Rust binary hashes at measurement time.

The original code's concurrent dictionary/set mutations and random IDs can
change its hierarchy across repetitions. Median latency is robust to some
variation, but the changing hierarchy is itself part of the workload. The
reported node/HKT counts and source-membership entries allow that variation to
be seen; they do not causally isolate language, data-structure or scientific
semantics costs. Source-level profiling was not performed. Optimization
opportunities identified from code inspection are hypotheses until separately
profiled and tested.

## Bottlenecks and optimization hypotheses

The measured evidence establishes the total update-time distributions,
per-batch latency curves, CPU/wall ratios, RSS curves, hierarchy sizes and Rust
trigger counts. It does **not** isolate cycles in individual functions.

Code inspection suggests that v1's candidate transaction clones the retained
engine, maintains an owned hierarchy plus node lookup mirror and retained
indexes, and runs a final diagnostic clone/traversal. Its map/context/rebuild
pipeline also allocates temporary sets and recursively traverses selected
scopes. These are plausible contributors to its 17.73 s G Full and H
Light times, but the benchmark does not assign a measured percentage to any.
The C# original enumerates dictionaries and collections repeatedly, allocates
temporary source/word collections, runs parallel loops and retains candidate
caches; its 6.09 median CPU/wall ratio on G shows substantial parallel work,
but there is no garbage-collection allocation profile in this study.

For v2, the three highest-priority **candidates for profiling and subsequent
optimization** are:

1. Repeated `frequencies(scope, evidence)` construction in every inspected or
   rebuilt HKT. The same retained source/word evidence is scanned along
   recursive paths, with owned strings inserted into temporary maps and sets.
2. String-keyed `BTreeSet` intersections, residual-scope construction and
   source-ID cloning at child traversal. An interned-ID representation or
   reusable scope buffer could reduce allocations if profiling confirms this
   dominates.
3. Recursive child construction after a trigger, which recomputes frequencies
   and fresh source sets for descendants. Passing reusable counts or views
   through construction may avoid repeated work while preserving the current
   exact scope and ancestor-exclusion semantics.

These are ordered by visible repeated work and potential reach, **not measured
function-level cost**. No algorithm or benchmark-targeted production
optimization was made. Reconstruction frequency is measured; reconstruction
duration and allocation hot spots remain unmeasured.
