# Native Rust benchmark results

Baseline is current production main `e0ca91e`, not the old research branch.
Both native candidates agree exactly on343 construction cases (43 captured plus
300 seed20711 random cases), plus every measured workload and repeated trial.
The direct normalized engine also agrees on110 evolution sequences; correctness
boundaries are in [evolution_validation.md](evolution_validation.md).
Scientific validity is assessed separately in [scientific_equivalence.md](scientific_equivalence.md).

## Measurement boundaries

Rust1.92.0, Linux x86_64, Cargo default release optimization (opt-level3), no LTO,
CPU pinning or target-cpu override. Standalone lockfiles use the same serde,
serde_json and thiserror versions as production. Five independent fresh-process
runs per implementation/workload; construction order is grouped by implementation,
stream order alternates. Shared-host load is uncontrolled, so small differences
are not conclusive. No warmup or best-trial selection was used.

Construction timings include incidence preparation and construction through the
owned native result. Production includes document-frequency preparation and its
public node cache; candidates include ordered incidence/rank preparation and,
for masks, scope-local source indexing and mask conversion. JSON parsing,
canonical projection/serialization and process startup are excluded for all.
These are computational representation comparisons, not drop-in API benchmarks:
no candidate-to-production HktBuildOutput/schema-4 materialization is included.
Candidate recursive ownership itself differs from production's public return type.

## Construction timings

Milliseconds, median of five independent runs. Ratios are candidate/production;
less than1 means lower measured elapsed time at the stated boundary. Density is
relations/(sources×words). Depth counts the root as1.

| Workload | Sources/words/relations | Density | HKTs/nodes/depth | Production ms | Sets ms (ratio) | Masks ms (ratio) |
|---|---:|---:|---:|---:|---:|---:|
| archived-news-25 | 25/535/743 | 0.056 | 2/5/2 | 0.355 | 0.286 (0.80) | 0.356 (1.00) |
| archived-news-100 | 100/1339/2862 | 0.021 | 14/42/6 | 2.541 | 2.716 (1.07) | 3.475 (1.37) |
| archived-news-250 | 250/2648/6955 | 0.011 | 21/93/5 | 6.620 | 7.450 (1.13) | 9.393 (1.42) |
| controlled-100-40-0.1-100 | 100/40/427 | 0.107 | 1/14/1 | 0.191 | 0.095 (0.50) | 0.109 (0.57) |
| controlled-1000-40-0.1-1000 | 990/40/4040 | 0.102 | 1/38/1 | 2.592 | 1.580 (0.61) | 0.939 (0.36) |
| controlled-1000-8-0.5-20 | 998/8/4016 | 0.503 | 199/315/7 | 16.803 | 9.582 (0.57) | 11.659 (0.69) |
| controlled-100-8-0.5-4 | 100/8/411 | 0.514 | 47/78/5 | 0.769 | 0.474 (0.62) | 0.605 (0.79) |

The first three use the checked-in news-derived `large_batch.json`, not live
RiskLive retained models. Flat40-word inputs suppress recursion with the branch
minimum at source count;8-word inputs separately exercise recursion through depth7.
Controlled data use seed717. Random correctness cases vary admission, collision
and branch parameters, including0/1 boundaries. Full inputs and hashes are retained.

Direct sets improve these controlled workloads but regress the larger news
prefixes; masks win the flat1000-source case but add overhead on all news prefixes
and recursive cases relative to direct sets. This mask experiment overlays dense
collision columns on a set-owned hierarchy. It is not a fully incidence-owned
engine or a bound on the best possible bitmap implementation.

## Allocations and memory

A separate **untimed** System-allocator wrapper counts successful allocation and
reallocation calls, total requested bytes and peak additional requested live bytes.
Input JSON/typed normalized input already exists when counters reset. Native
preparation and retained owned result are included; canonical projection/output
is excluded. Counter scalars are captured before allocating result JSON. These
numbers exclude allocator overhead, fragmentation and OS mappings. Each row is
one deterministic allocation pass; raw process RSS is also retained. It is not
allocation-failure recovery or an end-to-end snapshot memory profile.

| Workload | Allocation calls production / sets / masks | Peak requested live KiB production / sets / masks |
|---|---:|---:|
| archived-news-25 | 1149 / 2930 / 4020 | 227.5 / 367.2 / 404.2 |
| archived-news-100 | 7783 / 23778 / 30935 | 773.5 / 1473.6 / 1633.8 |
| archived-news-250 | 18076 / 57343 / 74097 | 1619.8 / 2966.1 / 3356.8 |
| controlled-100-40-0.1-100 | 402 / 292 / 383 | 71.9 / 48.0 / 50.8 |
| controlled-1000-40-0.1-1000 | 4161 / 2968 / 3249 | 700.3 / 484.9 / 497.7 |
| controlled-1000-8-0.5-20 | 50809 / 34979 / 42860 | 4135.4 / 1849.2 / 1854.1 |
| controlled-100-8-0.5-4 | 2998 / 2634 / 3405 | 300.6 / 156.4 / 158.2 |

News construction allocates substantially more objects and peak requested memory
in the direct candidate, despite owning the hierarchy only once. Per-word source
sets plus source-ordinal maps and child-scope copies are real costs. Controlled
inputs favor the direct representation. These mixed results do not support a
universal memory improvement. Peak RSS from five uninstrumented construction
runs and separate allocation probes is available in raw JSON; process RSS includes
input/output, executable and allocator retention, and is not equivalent to live
algorithm storage.

## Normalized update measurements

Each fresh process constructs40 sources, then processes20 batches of40 sources.
Stable inputs alternate two fixed descriptor groups; changing inputs add a novel
word each batch and use zero error thresholds. Full and gamma3 use identical
algorithms with different retention. Five runs per configuration, alternating
implementation order. Table entries are medians across the five per-run update
medians; RSS is the median process peak. All105 resulting normalized states in
each production/candidate pairing match exactly (initial plus20 updates across
five trials). JSON conversion, state normalization and each engine's own restart
roundtrip are outside the individual update timers.

Production `process_batch` includes external identity/vocabulary work, reports
and final-topology diagnostics; candidate `process` uses normalized identities
and returns only decisions/replacements. Both include transaction cloning,
contextual measurements, reconstruction and forgetting. Both timed binaries use
plain System allocation; the counting probe is separate. Therefore these numbers
are **not production-equivalent end-to-end speedup claims**. They show the cost of
the implemented normalized computational slice and motivate measuring the missing
boundary costs before a broader decision.

| Workload | Production median update ms | Candidate median update ms | Ratio | Peak RSS KiB production / candidate |
|---|---:|---:|---:|---:|
| full-stable | 1.139 | 0.170 | 0.15 | 8696 / 5584 |
| full-reconstruction | 2.317 | 0.426 | 0.18 | 10768 / 6628 |
| window3-stable | 0.434 | 0.065 | 0.15 | 5084 / 3576 |
| window3-reconstruction | 0.922 | 0.167 | 0.18 | 6016 / 3800 |

No full native mask lifecycle is implemented. No shared-descriptor synthetic
snapshot, arbitrary-depth limit, realistic30-day retention scale, production
persistence/fsync/publication cost or RiskLive news-quality evaluation is measured.
The3d/7d/30d container archives are tiny validation fixtures, described separately;
they cannot justify production sizing or a compressed-bitmap dependency.

## Code and invariant complexity

Physical non-test Rust lines including comments/blanks: production core3253,
production builder436; candidate library927 (including the competing mask kernel
and shared models), candidate CLI27, total954. Evaluation adapters/probes add341
lines and the Python experiment drivers add further code; none is silently
counted as algorithm simplicity. Production-equivalent LOC is **not established**:
the954-line normalized engine omits input registration, schema-4 validation,
accepted historical shapes, diagnostics/report contracts and persistence adapters.
It cannot be compared to3253 as a percentage reduction.

The candidate removes the synchronized full node mirror and permanent pending
membership representation. It still owns retained source payloads, contextual
source sets, topology, vocabulary and allocators. Operation-local construction
keeps both source sets and source-to-ordinal maps with equal domains; the mask
kernel adds source-position maps and columns. Thus it removes one hierarchy
synchronization invariant but adds/retains explicit local representation obligations.
Both engines keep clone-based transaction isolation. No evidence supports
removing cloning or historical payloads, nor does a small builder kernel remove
the full lifecycle's invariants. Candidate dependencies are serde/serde_json;
production adds thiserror. No graph/bitmap/database framework was added.

An earlier evaluator invocation reported production exit137 on a stress input.
The exact input was not checkpointed before the restart; the old generator ranges,
seed and observed status are preserved in `raw/interrupted-stress.json`.
Termination cause and peak memory are unknown. Hierarchy growth is a plausible
inference, not a diagnosed cause or evidence of candidate superiority. This
provenance loss is an investigation limitation. Current bounded cases checkpoint
inputs and record actual topology. The failed invocation is excluded from timing
ratios, not relabelled as a completed measurement.

## Reproduction and raw evidence

Run from a clean production source checkout at the pinned SHA; scripts refuse a
different HEAD or modified production crates. All experiments remain standalone:

```sh
cargo build --offline --release --manifest-path docs/architecture-exploration/iteration-2/rust-candidates/Cargo.toml
cargo build --offline --release --manifest-path docs/architecture-exploration/iteration-2/evaluation-baseline/Cargo.toml
python3 docs/architecture-exploration/iteration-2/evaluate.py > /tmp/seca-construction.json
python3 docs/architecture-exploration/iteration-2/evaluate_evolution.py > /tmp/seca-evolution.json
python3 docs/architecture-exploration/iteration-2/measure.py > /tmp/seca-measurements.json
python3 docs/architecture-exploration/iteration-2/archive_check.py
```

`raw/comparison.json` and `construction-measurements.json` retain five timings and
RSS values per implementation/workload, exact-field checks and actual topology.
`raw/workload-inputs.json` preserves inputs; `allocation-results.json` preserves
allocation counts/bytes; `stream-measurements.json` preserves every update timing,
input and trial; `evolution.json` preserves all differential states and decisions.
`measurement-provenance.json` contains compiler output, source/binary hashes,
fixture hash and file-by-file line counts. Baseline/test/fmt/Clippy/build logs
remain separate. Timing differences are implementation observations on these
inputs, not asymptotic-complexity proofs or universal production predictions.
