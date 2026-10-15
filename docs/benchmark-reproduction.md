# SECA benchmark reproduction

Integration note: this is historical Full-memory Rust/C# hierarchy characterization
from `01a644f`, not the canonical Theme G/H benchmark. The canonical runner,
clustering, metrics and complete saved evidence are documented in
[seca-paper-benchmark.md](seca-paper-benchmark.md). Statements below about missing
clustering apply only to this external-reference adapter and its original checkout.
Its physical-record prefix differs from the canonical dataset manifests; do not
combine the two result populations. No scientific proposal is approved here.

This branch adds an executable comparison using public Rust APIs and the existing
external serial C# harness. **No production algorithm is changed. Full scientific
parity and reproduction of the paper's clustering scores are not established.**
The engine baseline is fetched `origin/main` `01a644f3a7411bd4de4ad172555153fbcf525cc2`.
Existing golden fixtures are inputs/evidence, never regenerated or used as a
substitute for executing C#.

## Scientific source and benchmark eligibility

Al Sulaimani and Starkey, *Real-Time Event Detection Using Self-Evolving
Contextual Analysis (SECA) Approach*, IEEE Access 11 (2023), 127011–127034,
[DOI](https://doi.org/10.1109/ACCESS.2023.3331219).
The published version was inspected, including VII-D, VIII, IX and Tables 2–4.
A cached PDF was discovered at `/tmp/seca-paper.pdf`; its SHA-256 is
`dacd87d997e81820079d85d2f4fd3466705a87b7d5839ba6b98ccc5208f44947`.
The [author institution's published version](https://abdn.elsevierpure.com/ws/portalfiles/portal/284651535/Al_Sulaimani_etal_IEEEA_Real_Time_Event_VoR.pdf)
is the legitimate paper source. Direct download returned HTTP 403 in this session;
the existing local copy and rendered Tables 2/3 were available. The PDF is not
redistributed. Algorithm and prior execution contracts remain in
[algorithm.md](algorithm.md), [parity-executable-summary.md](parity-executable-summary.md),
[parity-reference-harness.md](parity-reference-harness.md), and
[performance.md](performance.md).

| Class | Benchmark | Legitimate source / public availability | Required preparation and evaluation | Current executable status |
| --- | --- | --- | --- | --- |
| Reproducible published benchmarks | None established end to end | No exact ordered paper dataset plus complete evaluation implementation established | Requires data identity, order, clustering and metric protocol | No paper-score reproduction claimed |
| Comparable with documented deviations | G: News | Public preprocessed MStream `News`, originally 11,109 titles / 152 categories | Choose prefix 8,880, 12 batches × 740; whitespace tokens; paper uses direct/indirect cluster scores averaged across cycles | Public dataset execution supported; event scores unavailable |
| Comparable with documented deviations | H: Tweets | Public preprocessed MStream `Tweets-T`, originally 30,322 tweets / 269 categories from TREC 2011–2015 | Choose prefix 24,000, 12 × 2,000; whitespace tokens; direct/indirect evaluation | Public dataset execution supported; variant/order uncertainty and event scores unavailable |
| Comparable input subsets only | G-smoke / H-smoke | Same pinned public files | First 75 records, 3 × 25; same parameters | Explicit small workload, never substituted for full G/H |
| Not presently reproducible | A | Manually curated TREC-derived selections; no original ordered selections located | 2,280 sources, 12 × 190, two groups × three events; fixed unequal source proportions | Excluded; do not synthesize event selections from plot |
| Not presently reproducible | B | Same unavailable curation | Same counts; one group shrinks while the other emerges to equal proportions | Excluded |
| Not presently reproducible | C | Same unavailable curation | Same counts; initially equal proportions diverge | Excluded |
| Not presently reproducible | D | Same unavailable curation | Same counts; one group peaks at cycle 6, the other troughs, then equalizes | Excluded |
| Not presently reproducible | E | Same unavailable curation | Same counts; group proportions cross then diverge | Excluded |
| Not presently reproducible | F | Same unavailable curation | Same 2,280 / 190; three groups × three events, changing proportions | Excluded |
| Not presently reproducible | World Cup 2022 final case study | Authors' Twitter API collection, not an exact publicly recovered archive | About 2.3M raw / 800k filtered tweets; Dec 18 15:00–17:57 GMT; exclude retweets, non-English and manually identified spam; minute batches, gamma 3; manual live-blog comparison and random 100-tweet node summaries | Excluded: missing corpus, full filtering recipe, samples and evaluation component |
| Not measured here | Theme H emissions / duration, transparency | Paper Table 4 / qualitative assessments | CodeCarbon energy/carbon assumptions and original hardware; questionnaire/manual assessment | No equivalent instrumentation; excluded as quantitative parity tests |
| Supplemental executable characterization | 40 committed fixture inputs | Repository's public synthetic a/b/c/... inputs | 19 context inputs + 21 HKT inputs; exact committed parameters/order; Full memory | Actual Rust/C# comparison; not published benchmark evidence |

### Public dataset identity and deviations

Source: [jackyin12/MStream](https://github.com/jackyin12/MStream/tree/ac894075f2bbb93405679e237d2d5d11a325078c/MStream/data),
revision `ac894075f2bbb93405679e237d2d5d11a325078c`, the locally discovered
upstream checkout's revision. This is a dataset source only; no MStream algorithm
is executed. [datasets.json](../benchmarks/reproduction/datasets.json) pins raw
SHA-256 values. `--download` fetches those exact files; alternatively supply a
local directory. Cached files must match the raw hashes, including CRLF bytes.
Public availability does not establish redistribution permission: source texts,
labels, generated input, identities, and raw topology remain local and ignored.

Table 2 reports G as 8,880 / 152 labels / event sizes 2–354, and H as 24,000 /
227 labels / sizes 1–607. The chosen prefixes match these aggregate statistics.
This supports the selection as a **documented inference**, not exact recovery.
The paper names Tweets without specifying a variant filename or record selection.
Literal `Tweets` has 255 labels in its 24,000-record prefix, whereas `Tweets-T`
has 227. The `-T` variant changes order and must not be represented as verified
chronological data. The paper describes natural evolution for G/H; this tension
remains unresolved. Obtain the authors' ordered IDs/import scripts to settle it.

Upstream cleaned text is authoritative: no new NLP, stemming, stop-word list,
case conversion or label-dependent selection. Use Python `str.split()` followed
by stable first-occurrence token deduplication. Both executables therefore receive
one source-word relation per document/token, in the same order. Identical text
under different corpus IDs remains separate evidence. Labels are saved only in a
local identity sidecar and never passed to either algorithm. A positive ordinal
1..N maps reversibly to `mstream:<variant>:<original Id>`; the same ordinal is the
Rust external identity and C# source ID. All batches are nonempty.

## Execution and output contract

[run.py](../benchmarks/reproduction/run.py) uses Python's standard library.
[benchmark_repro.rs](../crates/realtime-seca-core/examples/benchmark_repro.rs)
is a small public-API adapter, not an alternative engine. The common JSON input
is exactly the reference harness schema: ordered `batches` of `{id,words}`, plus
`alpha`, `beta`, `branchMinimum`, `alphaError`, `betaError`,
`wordImportanceError`. Both receive byte-identical JSON. The first batch constructs
the baseline; subsequent batches update it. Invalid/empty batches, duplicate IDs,
nonpositive IDs, empty/duplicate/untrimmed tokens and invalid parameter types are
excluded before execution. Empty input exclusion is not an algorithm equivalence
claim.

Public runs set construction/update alpha=.7, beta=.5, errors=.1/.2/.3, matching
paper VIII-C.4; **.2 beta error differs from the repository default .13**.
Branch minimum=1 is explicit but not recovered from the paper. Fixtures preserve
their own exact settings. Rust uses `MemoryMode::Full`; reference uses accumulated
SECA, `isLightSECA=false`. No forgetting comparison is claimed. The paper's Light
scores are retained for context only: the existing harness does not expose an
equivalent gamma/Light run. No full/hybrid rebuild fallback or compatibility
algorithm is introduced.

Set `SECA_REFERENCE_ROOT` to a directory containing
`tools/seca-reference-harness/run.sh`, its built `.build/harness.exe`, and original
reference dependencies. The actual discovered root here was
`/home/azureuser/risklive-dev/experimental`, with supplied sources under
`references/seca-csharp` and compilable copy under `references/seca-parity-build`.
The harness is an external prerequisite, not committed here: its generation
includes adapted original source. Follow [its documented build](parity-reference-harness.md)
if locally available. The runner does not build or modify that source, and never
redistributes original C#, generated source, dependencies or executables.

This C# reference serializes parallel loops, assigns deterministic IDs, disables
unused persistence, observes actual state, and surfaces swallowed exceptions.
Successful formulas remain original; failures and races need separate
interpretation. It is **not the original unmodified Windows application**. Its
archive revision and SourceCleanerEngine source-to-binary correspondence remain
unknown. The report hashes harness files, actual harness executable, generated
source manifest and five dependencies so substitutions are visible.

Each executable gets a fresh process. `--timeout` is a per-execution whole-process
limit, including serialization; no benchmark latency claim follows from it.
Rust flushes JSONL per completed cycle. C# emits its JSON only after success.
Unsuccessful processes have no final metrics; `emitted_cycles` describes Rust
output progress only. C# empty output does not establish that no internal work
completed. Logs/raw outputs are retained locally. Missing datasets, missing
executables, failures, timeouts, invalid output and common-input exclusions are
explicit. A nonzero runner exit indicates an execution failure/missing measurement
or failed repeat, not a cross-language metric mismatch. An explicitly excluded
input does not itself cause failure. Existing output directories are refused.
`--repeat` reruns successful executions and compares stdout bytes; failures are
not interpreted as successful determinism measurements.

The report records input SHA-256, raw corpus hash, URL/revision, original ordered
ID digest, excluded tail, parameters, retention, code/executable hashes,
base revision, tracked differences, Rust version, Python/platform and process
status. The exact dependency lock used is saved as
[benchmarks/reproduction/Cargo.lock](../benchmarks/reproduction/Cargo.lock); copy it
to the repository root before building to reproduce dependency resolution.
The root lock is repository-ignored; its hash is recorded. Reproduction is semantic,
not executable-byte equality across different toolchains. After building, no
network is needed to run fixtures or cached datasets.

### Comparable HKT measurements

Only completed full runs receive these end measurements. Counts include refuges
where specified, and no expected-word representation is treated as a cluster.

| Field | Definition |
| --- | --- |
| `hkts` | Number of reachable live HKT containers |
| `nodes` | All reachable nodes, including refuge nodes |
| `refuge_nodes` | Nodes flagged refuge in Rust / containing sentinel -1 in C# |
| `descriptor_occurrences` | Sum of non-refuge node word counts; repeated word in different nodes counts again |
| `source_memberships` | Sum of node source-set sizes; a source in several nodes/levels counts several times |
| `represented_sources` | Union of source IDs in all nodes; not total ingested or retained source count |
| `max_depth` | Maximum HKT depth, root=0 |
| `reconstruction_targets` | Sum of selected reconstruction root counts across all updates; not generated HKT count |
| `final_labelled_topology_equal` | Recursive words/refuge/source memberships and attachment equality; removes incidental IDs, expected maps and sibling order |

Normalized full labelled topology is compared independently of the scalar counts:
equal counts alone cannot establish equal structure. A changed sibling order is
ignored in this *final* comparison; it may still affect future greedy behavior.
Differences in original units use signed `Rust - C#`; relative differences divide that
quantity by the C# value. For a zero C# denominator, relative difference is null,
including 0 versus 0. These are structural/descriptive metrics, not event quality.
This comparison does not assert per-word source-map or pending-state equality.
Raw JSON/JSONL retains decisions and available errors for investigation; aggregate
errors from different HKT scopes are not averaged into invented quality scores.

### Published evaluation definitions

The machine-readable [published.json](../benchmarks/reproduction/published.json)
transcribes all four SECA variants for A–H from Table 3, in percentages. They are
**averages over the experimental cycles**, not merely final-cycle measurements.
Paper VII-D direct clustering uses Seed/Orphan nodes' sources; indirect clustering
uses contextual word values (13), merging of descriptors, proximity (14), and
highest-proximity assignment. Rust implements neither evaluation stage; the
existing C# harness exports HKTs/decisions, not those cluster assignments.
A topology-to-label shortcut would be a new method and is deliberately absent.
The exact overlap/tie, merge threshold and averaging details require clarification
before a future score reproduction. No labels are reconstructed or curated here.

Purity (18) is the document-weighted dominant-category proportion per predicted
cluster. NMI (19–22) is `2 I(labels;clusters)/(H(labels)+H(clusters))`, using the
arithmetic-mean entropy normalization. BCubed (23–25) computes each document's
same-label cluster proportion and same-cluster label proportion, takes their
harmonic mean, then averages per-document F1 as described in the paper; the
harmonic mean of corpus-average precision/recall need not agree. The printed
purity variable descriptions interchange label/cluster notation; resolve against
actual evaluation code before claiming exact implementation. Coverage (26) counts
a ground-truth event detected when at least 60% of a cluster concerns that event;
Fig.6 presents G/H recall curves. No exact numeric curve digitization is invented.

| Theme | SECA direct NMI / F1 / purity % | SECA indirect | Light direct | Light indirect |
| --- | --- | --- | --- | --- |
| A | 92.73 / 89.48 / 99.82 | 92.70 / 89.53 / 99.82 | 92.74 / 89.66 / 99.69 | 91.83 / 87.91 / 99.91 |
| B | 94.58 / 91.82 / 99.87 | 94.39 / 92.64 / 99.87 | 94.56 / 92.27 / 99.61 | 93.45 / 91.06 / 99.82 |
| C | 92.80 / 90.10 / 99.87 | 91.87 / 89.32 / 99.87 | 92.43 / 90.38 / 99.65 | 91.52 / 88.42 / 99.82 |
| D | 93.16 / 91.63 / 99.83 | 93.69 / 92.22 / 99.82 | 93.51 / 92.45 / 99.65 | 93.02 / 91.34 / 99.82 |
| E | 93.08 / 90.33 / 99.91 | 93.29 / 90.77 / 99.82 | 93.79 / 91.53 / 99.74 | 92.21 / 89.26 / 99.87 |
| F | 90.96 / 84.89 / 99.64 | 90.17 / 83.45 / 99.56 | 90.99 / 85.72 / 99.55 | 89.86 / 82.98 / 99.56 |
| G | 86.21 / 54.63 / 71.67 | 88.30 / 63.03 / 80.36 | 85.94 / 55.37 / 70.73 | 87.93 / 62.74 / 79.20 |
| H | 77.75 / 53.05 / 75.53 | 87.52 / 71.14 / 89.18 | 84.30 / 64.32 / 81.44 | 88.85 / 74.08 / 89.03 |

Table 4 reports SECA 492 seconds / 1.65 grams CO2eq and SECA-Light 163 seconds /
0.56 grams for Theme H; nearby prose says 1.69 for SECA, an unresolved internal
inconsistency. These hardware/energy-dependent values are not comparison targets.
Transparency assessment is qualitative. No extra baseline algorithms are run.

## Reproduce

From the isolated worktree root, build before running:

```sh
cp benchmarks/reproduction/Cargo.lock Cargo.lock
cargo build --offline --release -p realtime-seca-core --example benchmark_repro
export SECA_REFERENCE_ROOT=/path/to/external/reference-root
python3 benchmarks/reproduction/run.py --suite fixtures --repeat --timeout 60 --output /tmp/seca-fixtures-new
python3 benchmarks/reproduction/run.py --suite public --download --repeat --timeout 60 --output /tmp/seca-public-new
```

Omit `--offline` for a first Cargo build when dependencies are not cached.
For cached dataset files use `--source-dir /path/to/MStream/data` instead of
`--download`. Network download failures remain explicit; there is no replacement
corpus. Use `--case G` or `--case H-smoke` to select individual experiments;
`--case` is repeatable and fixture names include their family prefix.
Increase the process timeout if desired, recording a new run in a fresh directory.
The output is `report.json`, `metrics.csv`, and `raw/` (local only), including
common inputs, identity sidecars, stdout and stderr. The CSV retains missing
measurement/status rows; Git normalizes committed CSV line endings to LF. Commit aggregate reports only; public text and labels
and all C# source/binaries must stay excluded.

The historical benchmark at `315eb3a` was inspected. The
`benchmark_stream.rs` from that commit measures synthetic timing/retention with different workloads
and thresholds; it cannot supply public event-quality scores. This runner reuses
the same public engine APIs and the committed parity inputs instead of importing
performance probes or obsolete engine code.

## Actual experiments

Executed on 2026-10-15 with Rust/Cargo 1.92.0 and Mono 6.8.0.105. The exact
command was:

```sh
SECA_REFERENCE_ROOT=/home/azureuser/risklive-dev/experimental python3 benchmarks/reproduction/run.py --suite all --source-dir /tmp/seca-mstream-upstream/MStream/data --output benchmarks/reproduction/results --timeout 60 --repeat
```

Committed outputs: [JSON report](../benchmarks/reproduction/results/report.json)
and [CSV metrics](../benchmarks/reproduction/results/metrics.csv).
Raw outputs/identity maps remain in that directory's ignored `raw/` subtree;
rerunning recreates them without manual processing. All 13 actual compilable
reference source hashes were separately checked against the committed context
fixture manifest and match. Binary/harness provenance is recorded in the report.

There are 44 selected cases: 40 synthetic inputs and four public-data workloads.
38 synthetic cases complete in both languages; empty input is excluded by the
shared contract; shared-candidate Rust succeeds while C# throws a duplicate-key
exception. Of the 38 paired fixture runs, 29 have equal final labelled topology,
9 differ. Both public subsets complete in both languages with unequal topology.
In the 60-second sweep, all 41 successful Rust and 40 successful C# executions were repeated with
byte-identical stdout. This is determinism evidence for successful observed runs,
not a universal claim about the original parallel application's behavior.

### Public workloads

| Workload | Rust outcome | Instrumented serial C# outcome | Published-quality comparison |
| --- | --- | --- | --- |
| G, 8,880 / 12 × 740 | Timeout at 60s; 7 emitted complete cycles | Failure, duplicate-key exception (word ID 636); no final JSON | Unavailable |
| H, 24,000 / 12 × 2,000 | Timeout at 60s; 3 emitted complete cycles | Timeout at 60s; no final JSON | Unavailable |
| G-smoke, 75 / 3 × 25 | Complete; deterministic repeat | Complete; deterministic repeat | No paper metric for this subset; clustering missing |
| H-smoke, 75 / 3 × 25 | Complete; deterministic repeat | Complete; deterministic repeat | No paper metric for this subset; clustering missing |

These deadlines bound this experiment; they do not establish a performance
ranking or that larger-timeout runs are impossible. Test/build workloads overlapped
some runs. No final metrics are calculated from a timed-out prefix, and the
small workloads do not replace G/H. A larger timeout is an explicit new experiment.

For completed public subsets, each cell below gives Rust / C# / signed difference
and relative difference with C# as denominator. Counts are final except cumulative
reconstruction targets.

| Metric | G-smoke Rust / C# | Δ / relative % | H-smoke Rust / C# | Δ / relative % |
| --- | --- | --- | --- | --- |
| hkts | 24 / 26 | -2 / -7.69% | 39 / 44 | -5 / -11.36% |
| nodes | 103 / 105 | -2 / -1.90% | 122 / 127 | -5 / -3.94% |
| refuge_nodes | 4 / 4 | +0 / +0.00% | 11 / 11 | +0 / +0.00% |
| max_depth | 2 / 2 | +0 / +0.00% | 3 / 3 | +0 / +0.00% |
| descriptor_occurrences | 455 / 457 | -2 / -0.44% | 563 / 578 | -15 / -2.60% |
| source_memberships | 227 / 229 | -2 / -0.87% | 298 / 303 | -5 / -1.65% |
| represented_sources | 75 / 75 | +0 / +0.00% | 75 / 75 | +0 / +0.00% |
| reconstruction_targets | 2 / 2 | +0 / +0.00% | 2 / 2 | +0 / +0.00% |

Both subsets represent all 75 input identities, but their word grouping and/or
attachments differ. C# has two more HKTs/nodes on News and five more on Tweets-T.
The boundary fixture below establishes one mechanism that can generate extra C#
branches at minimum=1. Recursive scope and greedy ordering also differ in known
fixtures. These are plausible contributors, not a controlled attribution of every
public topology difference. The runner does not infer scientific accuracy from
fewer nodes, or equate a node-membership union with event recall.

### Extended full-workload attempts

The full chosen G/H inputs were also run independently with a 300-second
per-executable limit, without `--repeat`. Example command (use H and a fresh
output directory for the other run):

```sh
SECA_REFERENCE_ROOT=/home/azureuser/risklive-dev/experimental python3 benchmarks/reproduction/run.py --suite public --case G --source-dir /tmp/seca-mstream-upstream/MStream/data --output benchmarks/reproduction/results-G-extended --timeout 300
```

[Extended G JSON](../benchmarks/reproduction/results-G-extended/report.json) /
[CSV](../benchmarks/reproduction/results-G-extended/metrics.csv) records a complete
12-cycle Rust run on the chosen 8,880-record News prefix. C# fails with the same
duplicate-key exception (word ID 636), without a final measurement. This successful
full Rust execution was not repeated, unlike the small paired comparisons.

| Final G characteristic | Actual Rust | C# | Difference |
| --- | ---: | --- | --- |
| HKTs | 5,641 | Unavailable: exception | Unavailable |
| Nodes | 15,072 | Unavailable | Unavailable |
| Refuge nodes | 2,218 | Unavailable | Unavailable |
| Maximum HKT depth | 14 | Unavailable | Unavailable |
| Descriptor occurrences | 32,573 | Unavailable | Unavailable |
| Source memberships | 90,165 | Unavailable | Unavailable |
| Represented sources | 8,880 | Unavailable | Unavailable |
| Cumulative reconstruction targets | 2,177 | Unavailable | Unavailable |

This provides actual Rust end characteristics for the full chosen G workload,
not NMI/F1/purity. No published score difference is calculated: clustering outputs
are missing and exact paper record identity/order is unresolved.

[Extended H JSON](../benchmarks/reproduction/results-H-extended/report.json) /
[CSV](../benchmarks/reproduction/results-H-extended/metrics.csv) records timeouts
for both implementations at 300 seconds. Rust emitted six complete cycles;
C# emitted no final JSON. Neither has a full-run end measurement. These extended
runs overlapped each other and part of the shorter sweep; no runtime comparison
is inferred. Timeout-prefix lengths can vary with machine load.

### Fixture discrepancies explaining possible structural divergence

The following actual final measurements select the cases whose final labelled
topology differs; names omit the family prefix only within this table. All exact
parameters and input hashes remain in the report. H/N means HKTs/nodes; memberships
counts source memberships across levels. Empty input and C# failure remain outside
these paired comparisons.

| Family / case | Rust H/N; memberships; reconstruction targets | C# H/N; memberships; reconstruction targets | Interpretation from prior executed evidence |
| --- | --- | --- | --- |
| context / alpha_threshold | 1/1; 6; 1 | 1/1; 6; 0 | Full precision versus four-decimal C# rounding changes trigger |
| context / beta_threshold | 1/1; 6; 1 | 1/1; 6; 0 | Full precision versus four-decimal C# rounding changes trigger |
| context / expected_trigger | 1/2; 7; 1 | 1/1; 4; 0 | C# stale expected-word counts versus fresh accumulated Rust evidence |
| context / refuge | 1/2; 5; 0 | 1/2; 4; 0 | Refuge memberships retained differently; memberships can differ with equal counts of HKTs/nodes |
| HKT / branch_boundary | 1/1; 4; 0 | 2/2; 6; 0 | Rust strict source-count cutoff versus inclusive C# cutoff |
| HKT / collision_true_tie | 1/2; 9; 0 | 1/2; 9; 0 | Different greedy collision tie resolution; equal scalar counts hide different grouping |
| HKT / frequency_vs_vocabulary | 1/2; 9; 0 | 1/3; 12; 0 | Expected-word traversal changes grouping |
| HKT / overlap_recursive | 4/5; 29; 0 | 2/3; 21; 0 | Sibling-adopted word reuse changes recursive scopes |
| HKT / refuge | 2/3; 11; 0 | 2/3; 9; 0 | Refuge memberships retained differently; memberships can differ with equal counts of HKTs/nodes |

These observations agree with the prior [discrepancy register](parity-executable-summary.md).
Equal final trees in other fixtures do not prove equal intermediate decisions or
full scientific parity. In particular, no inference about published NMI, BCubed,
purity, coverage or Light forgetting follows. No golden fixture is altered.

### Validation and remaining limitations

Executed:

```sh
python3 -m unittest discover -s benchmarks/reproduction -v
cargo build --offline --release -p realtime-seca-core --example benchmark_repro
cargo test --offline --workspace
cargo test --offline -p realtime-seca-core --example benchmark_repro
cargo fmt --all -- --check
cargo clippy --offline --workspace --all-targets -- -D warnings
cargo clippy --offline -p realtime-seca-core --all-targets -- -D warnings
git diff --check
```

The [validation record](../benchmarks/reproduction/validation.json) preserves check outcomes.
Five Python contract tests pass, covering identity/token rejection, empty batches,
fixture inventory, topology attachment/order normalization, count definitions,
process failure/timeout handling and source-hash mismatch. Workspace Rust tests:
**138 passed, 2 ignored, 0 failed**; the two ignored tests assert known strict
C# parity failures and remain unchanged. Release build, example compilation,
formatting, core/all-target Clippy and whitespace validation pass.
Workspace Clippy **fails** on two pre-existing deprecations in unchanged
`crates/realtime-seca-cli/src/ingestion.rs`: `stop_words::LANGUAGE` and
`LANGUAGE::English`, with resolved `stop-words 0.10.1`. No warning suppression or
unrelated CLI migration is included. The lockfile records the exact resolution.

The final runner and adapter are respectively under 400 lines; their entry
functions were reviewed for cohesive input/execution/output responsibilities.
Production `src/` and existing fixtures have no diff. All output artifacts staged
for publication are aggregate JSON/CSV/provenance, not corpus text, labels or
restricted source. The report's engine revision is the unmodified base engine;
the runner/adapter/executable/lock hashes identify the benchmark additions.

Still required for end-to-end paper reproduction: authors' exact ordered datasets
and selections, direct/indirect event clustering and its evaluation contract,
Light reference retention configuration, resolved scientific ambiguities, and
original C# release/dependency provenance. These are limitations rather than
fabricated measurements. This branch is a reproducible executable comparison
with bounded exclusions, not a parity certification.
