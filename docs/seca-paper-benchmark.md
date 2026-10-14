# SECA paper benchmark: Themes G and H

## Objective and reproduction status

Establish an empirical event-clustering reference for future Rust simplification,
using Al Sulaimani and Starkey, *Real-Time Event Detection Using Self-Evolving
Contextual Analysis (SECA) Approach*, IEEE Access 11 (2023), 127011–127034
([DOI](https://doi.org/10.1109/ACCESS.2023.3331219)). This benchmark tests scientific
clustering quality; exact C# implementation parity is not its acceptance criterion.

These results are a **reconstruction of the experimental protocol**, not exact
paper replication. The original ordered selections, gamma, indirect merge
threshold, and scored population have not been recovered. The user confirmed no
original experiment configuration was available. No settings were selected by
maximizing agreement with published scores.

All 12 cycles were executed for each theme in Full and SlidingWindow memory,
and both direct and indirect assignments were generated after every cycle.
Indirect scores form the empirical quality reference under the declared research
interpretation. Direct memberships overlap; the paper does not establish their
conversion to the hard partitions required by the documented metrics. Those
scores remain undefined rather than silently dropping documents or memberships.

## Integration review and scope

Base: tested correctness integration `5ece43a71a40f08854aee0c2942791ec50443201`,
which was ahead of unmerged `main`. Reviewed the three reports and complete branch
diffs before selecting files:

| Agent branch | Reviewed revision | Owned files |
|---|---|---|
| `experiments/seca-datasets` | `8c9294595e9d7e37360d62d64406908d5a19283f` | dataset scripts, locks, manifests, tests and report |
| `experiments/seca-clustering` | `a67ca52da2f69aa0aaaf4efec023e4b79d1e7dbe` | clustering adapter, tests and report |
| `experiments/seca-evaluation` | `adbd9f6866cec84dd418b9fee78439a8a5a53e9d` | evaluation, runner, comparison, fixtures, tests and report |

Ownership did not conflict. Preparation emits `SourceBatch` JSON and separate
truth; clustering accepts public verbose tree exports and caller-owned tokens;
evaluation consumes explicit populations and memberships. No agent or integration
change altered production SECA construction, context metrics, reconstruction,
forgetting, module boundaries or state ownership. Existing goldens are unchanged.
The only Rust addition is an experimental example calling the existing public
engine API and writing evidence. No additional production read-only interface
was needed. The integration corrected indirect positive ties to use semantic
proxy signatures rather than allocator IDs; the unresolved paper rule is recorded.

## Dataset provenance and preparation

Pinned source: [MStream](https://github.com/jackyin12/MStream/tree/ac894075f2bbb93405679e237d2d5d11a325078c/MStream/data)
at `ac894075f2bbb93405679e237d2d5d11a325078c`, the latest revision before the
paper's cited access date. Exact bytes, all four investigated variants, normalized
ordered hashes and historical correspondence are recorded in
[the source lock](../benchmarks/seca-paper/datasets/sources.json) and
[the dataset report](seca-benchmark-datasets.md).

| Theme | Reconstructed selection | Cycles | Labels | Raw source SHA-256 |
|---|---|---|---|---|
| G | first 8,880 records of `News`; 2,229 excluded | 12 × 740 | 152 | `f68afe2771048371a84fb97f0e8599b437b178117dd3678ec16a52c30b36cfad` |
| H | first 24,000 records of `Tweets-T`; 6,322 excluded | 12 × 2,000 | 227 | `e1b806af9992e881ff75b76efe259dff8b3483d9be11f1110a777356075070cf` |

These prefixes match Table 2's counts and event-size ranges. This does not prove
the original IDs or ordering. Literal `Tweets` gives 255 labels in its first
24,000 records; `Tweets-T` gives the reported 227. Historical native timestamps
show `Tweets` is chronological and `Tweets-T` is not. The aggregate match and
paper's natural-evolution description therefore remain in tension. H uses the
aggregate-matching variant explicitly; it is not presented as recovery of the
original chronological tweet stream.

Upstream cleaned text is whitespace-tokenized without additional cleaning,
stemming, deduplication or reordering. Original zero-padded IDs are qualified by
variant, e.g. `mstream:News:000001`; repeated texts retain their distinct identities.
No label-derived field enters engine or clustering requests. SourceBatch inputs
contain only ID, batch index and tokens at the engine boundary. Raw and prepared
hashes and all 28 generated files were verified against committed manifests.

No dataset-specific redistribution grant or repository LICENSE was found at the
pinned revision. Original text and ground-truth payloads are not retained in Git.
Only preparation code, provenance, hashes, statistics, manifests and generated
ID-only assignment evidence are committed. Original corpora, tokens, annotations,
trees and snapshots remain in ignored local directories and are regenerable.

## Parameters and learning lifecycle

| Parameter | Value | Evidence/status |
|---|---|---|
| alpha / beta | 0.7 / 0.5 | published experimental settings, VIII-C |
| AlphaError / BetaError / WIError | 0.1 / 0.2 / 0.3 | published experimental settings, VIII-C |
| branch minimum | 10, existing exclusive cutoff | repository configuration; original experiment value unresolved |
| SECA memory | Full; gamma null | accumulated evidence; canonical update path |
| SECA-Light memory | SlidingWindow; gamma 3 | existing `seca_light_config.json`; **not a verified original experiment gamma** |
| indirect merge cutoff | exact `1/2`, inclusive | research assumption; original threshold not reported |
| evaluation population | current batch, 740 / 2,000 documents | explicit reconstruction choice; original population unresolved |
| aggregate | unweighted mean of all 12 cycle scores | declared interpretation of Table 3's “average” |

Configurations are [Full](../benchmarks/seca-paper/config/Full.json) and
[SlidingWindow](../benchmarks/seca-paper/config/SlidingWindow.json). Gamma 3
means exactly the current and two preceding batches, following the existing
[state contract](state.md). The paper's literal cutoff can suggest a different
boundary; neither the gamma value nor that ambiguity was patched to fit results.

The first batch calls `build_baseline_tree` once. Every later batch calls
`process_batch` on that same engine. Reports, verbose exports and resumable
schema-4 snapshots are saved after each cycle. Exports' external source identities
are checked against the complete retained population, independently of the
scored current batch. A scoring-only pass may reuse immutable saved exports only
after checking input hashes, dataset hashes, engine binary and configuration;
its parent run and tree hashes are recorded. Its recorded engine command is a
replay recipe, not a claim that scoring reexecuted learning.

## Clustering interpretation

Direct events are ordinary Seed/Orphan nodes reached through only refuge
ancestors. Ordinary descendants refine their owning event, and an internal refuge
below an ordinary node does not start another independent event. Direct assignment
uses each event root's source set and retains multiple memberships.

Indirect proxies include ordinary contextual descendants, weighting each word by
node support / event-root support (equation 13). Repeated descriptors and merged
weights use their maximum value. Proxies sharing a descriptor at or above `1/2`
merge as connected components of the original overlap graph. Proximity sums exact
weights of distinct matching source tokens (equation 14). Maximum positive
proximity assigns one event; positive ties use the lexicographically smallest
sorted token/weight proxy signature. Zero matches remain unassigned. Presentation
`top_words` are not used as contextual descriptors. These reduction, merging and
tie decisions are documented research interpretations, not recovered paper rules;
see [the clustering report](seca-clustering-evaluation.md).

Assignments and comparisons treat event names as opaque. Changing node IDs alone
cannot change positive-tie decisions, and a focused test verifies this. Replaying
the archived source inputs, rather than restoring old implementation snapshots,
is the baseline for future architecture comparisons. No private Rust state layout
is read by the clustering adapter.

## Evaluation methodology

For complete hard partitions, NMI uses `2 MI / (H(predicted)+H(truth))` with natural
logs. BCubed precision and recall are document means, and F1 is the mean of each
document's F1, rather than the harmonic mean of aggregate precision and recall.
Purity is summed predicted-cluster majority count divided by evaluation population.
The paper's W/C explanatory notation is inconsistent; this follows its
predicted-cluster interpretation. Empty clusters contribute nothing. Empty
populations or zero-entropy NMI are undefined. Scores are fractions in JSON/CSV
and percentages in the table below.

All raw assignments are saved before scoring. Unassigned or overlapping documents
block hard-partition scores for the full population; they are not omitted,
converted to singleton events, or put in an invented cluster. No overlap metric
variant is claimed to be recovered from the paper. An independent set-intersection
calculation recomputes every saved score, with agreement to `1e-12`; this numerical
check is not a paper-parity tolerance.

## Published versus reproduced scores

The final published PDF was recovered from the
[authors' university repository](https://abdn.elsevierpure.com/ws/portalfiles/portal/284651535/Al_Sulaimani_etal_IEEEA_Real_Time_Event_VoR.pdf),
SHA-256 `dacd87d997e81820079d85d2f4fd3466705a87b7d5839ba6b98ccc5208f44947`.
Table 3 was visually checked on PDF page 15 / printed p. 127025, including all
24 G/H values. All eight method variants are reported; no reference is fabricated.

| Theme / method | Published NMI | Reproduced | Abs. Δ | Published BCubed F1 | Reproduced | Abs. Δ | Published purity | Reproduced | Abs. Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G / SECA / direct | 86.21 | — | — | 54.63 | — | — | 71.67 | — | — |
| G / SECA / indirect | 88.30 | 87.17 | 1.13 | 63.03 | 60.89 | 2.14 | 80.36 | 79.77 | 0.59 |
| G / SECA-Light / direct | 85.94 | — | — | 55.37 | — | — | 70.73 | — | — |
| G / SECA-Light / indirect | 87.93 | 86.59 | 1.34 | 62.74 | 59.86 | 2.88 | 79.20 | 78.01 | 1.19 |
| H / SECA / direct | 77.75 | — | — | 53.05 | — | — | 75.53 | — | — |
| H / SECA / indirect | 87.52 | 85.25 | 2.27 | 71.14 | 71.05 | 0.09 | 89.18 | 93.05 | 3.87 |
| H / SECA-Light / direct | 84.30 | — | — | 64.32 | — | — | 81.44 | — | — |
| H / SECA-Light / indirect | 88.85 | 87.56 | 1.29 | 74.08 | 75.69 | 1.61 | 89.03 | 93.62 | 4.59 |

Scores are percentages; absolute differences are percentage points. — means no defined full-population aggregate. Direct overlaps remain unconverted. All reproduced scores are research reconstructions, not exact paper replication.

## Observed discrepancies and scientific limits

Every cycle has nonempty predicted events and zero unassigned documents for both
procedures. Direct overlap ranges are 58–88 (News Full), 59–80 (News Light),
143–287 (Tweets Full), and 50–233 (Tweets Light), per batch. A document has at
most four direct memberships in Full / News Light, and three in Tweets Light;
there is no exploding duplication. All indirect metrics are finite.

The indirect differences are 1.13–2.27 percentage points for NMI, 0.09–2.88 for
BCubed F1, and 0.59–4.59 for purity. News scores are lower across these metrics.
Tweet purity is higher, while Full NMI is lower and Full BCubed F1 is very close.
Full Tweet mean BCubed precision is 90.66% and recall 64.73%, consistent with
fragmentation contributing to high purity. Higher purity alone is not evidence of
better overall clustering. No standard errors or original per-cycle assignments
are available to establish a statistical equivalence claim.

Reversing source order within the first two batches changes the following numbers
of documents' event co-member sets; these counts ignore event/node IDs and are
not document error rates:

| Theme / procedure | Cycle 1 | Cycle 2 | Batch population |
|---|---:|---:|---:|
| G direct | 0 | 49 | 740 |
| G indirect | 0 | 28 | 740 |
| H direct | 6 | 9 | 2,000 |
| H indirect | 11 | 97 | 2,000 |

Both memory modes give these early checks, before their windows diverge. The
node-ID tie correction removed a separate spurious sensitivity in indirect
assignment. Residual ordering sensitivity is measured, not repaired in the engine;
these two-cycle diagnostics do not establish stability of reordered full streams.
Snapshot continuation after cycles 1 and 6 reproduces the next tree and complete
state exactly in both themes and modes. Separate learning passes use identical
input order, and their complete cycle exports are byte-identical.

The score differences matter as measured quality changes; comparable numbers
alone do not establish scientific parity. Unknown population/averaging, H's
variant and order, the branch cutoff, gamma, and proxy merging can plausibly
explain deviations, but these runs do not isolate their contributions. Upstream
preprocessed bytes are fixed, so additional preprocessing variation was not
introduced here. Original preprocessing provenance itself remains incomplete.
The engine's documented expected-word, candidate-beta, importance-denominator
and construction interpretations remain potential contributors (see
[scientific decisions pending](scientific-decisions-pending.md)); no experiment
here proves they caused a particular score difference. No engine parameter or
algorithm was changed to improve agreement.

| Failure category | Evidence and interpretation |
|---|---|
| Dataset / experiment protocol | exact original selections, ordering, gamma, merge threshold and population unavailable; reconstruction is executable, exact replication is blocked |
| Clustering implementation | adapter invariants and hand-derived tests pass; allocator-ID tie dependence was corrected in experimental clustering and tested; remaining proxy rules lack original executable evidence |
| Metric implementation | hand-calculated tests, prior scikit-learn arithmetic-NMI cross-check, and independent saved-assignment recomputation agree; direct overlap scoring lacks a scientific contract |
| Engine behaviour | canonical updates complete; retained identity populations and snapshot continuation checked; known cross-language differences preserved; scores alone do not identify an engine defect |
| Resources | all required streams completed; no resource-related run failure; Full grows in runtime while Light retains the selected evidence window; no optimization was performed |

The usable quality baseline is the four indirect research runs under this exact
configuration and dataset reconstruction. Published direct quality cannot yet be
reproduced without an explicit original scoring rule. Recovering the original
ordered per-cycle IDs and experiment configuration would be necessary to upgrade
this result to exact replication.

## Validation and preserved evidence

`cargo fmt --all -- --check`, `cargo test --workspace` and
`cargo clippy --workspace --all-targets -- -D warnings` passed: 163 Rust tests,
with the two known strict parity tests still ignored by the ordinary suite.
Explicitly executing each strict test reproduces its known failure: the alpha
rounding threshold and HKT `branch_boundary` topology. Neither test nor golden
was modified to force a pass. Dataset tests: 13; clustering tests: 21; evaluation
and integration tests with the real CLI: 25.

A label-independent real-data smoke selects two disjoint repeated News texts and
four original identities for each text. Both modes and both procedures initialize,
update, produce two nonempty events, score all metrics, persist/reload assignments
and independently recompute perfect scores. Restoring the first snapshot produces
identical next-cycle exports and snapshots. This verifies the integration path;
its chosen small sample does not establish corpus-wide quality.

[Committed evidence](../benchmarks/seca-paper/evidence/final) contains per-cycle
CSV, diagnostics, ID-only gzip assignments, independent recomputations, exact
configuration, manifests, Git/binary/adapter hashes, complete local artifact
hash inventories, the published comparison, integration review and validation
records. Dependency versions are preserved in `engine-Cargo.lock`; the recorded
build uses rustc/cargo 1.92.0 and Python 3.11.4 on Linux x86_64, debug profile.
The engine was built from benchmark revision `31e1b9d`; final scoring
uses the source revision recorded in each run. Tracked source code was clean;
recorded untracked status entries are generated evidence. Reports, snapshots,
normalized engine inputs and annotations remain locally under ignored `results/`.

## Reproduction commands

Run from the checked-out integration branch root. Python's standard library and
Rust suffice; downloading the pinned sources needs network access. Outputs refuse
to overwrite prior evidence, so use fresh paths for subsequent experiments.

```bash
python3 benchmarks/seca-paper/datasets/download.py
python3 benchmarks/seca-paper/datasets/prepare.py
python3 benchmarks/seca-paper/datasets/verify.py
cp benchmarks/seca-paper/evidence/final/engine-Cargo.lock Cargo.lock
cargo build --locked -p realtime-seca-core --example paper_stream
cargo build --locked -p realtime-seca-cli

python3 benchmarks/seca-paper/real_smoke.py \
  --manifest benchmarks/seca-paper/datasets/data/prepared/G/manifest.json \
  --output /tmp/seca-smoke-new --execute

python3 benchmarks/seca-paper/execute.py --theme G \
  --output /tmp/seca-rerun --evidence /tmp/seca-rerun-evidence
python3 benchmarks/seca-paper/execute.py --theme H \
  --output /tmp/seca-rerun --evidence /tmp/seca-rerun-evidence
python3 benchmarks/seca-paper/summarize.py /tmp/seca-rerun-evidence
```

Each theme command executes all four method/memory combinations without tuning.
A scientific scoring blocker is recorded per cycle and does not discard later
assignments. Engine or malformed-evidence errors fail the run explicitly.

Recompute a committed run directly from its compressed ID-only assignments and
freshly prepared truth:

```bash
python3 benchmarks/seca-paper/recompute.py \
  benchmarks/seca-paper/evidence/final/G-Full-indirect \
  --truth benchmarks/seca-paper/datasets/data/prepared/G/ground-truth.json
```

Use the corresponding theme truth file for H. Snapshot and ordering checks on
new Full or SlidingWindow runs are reproducible with:

```bash
python3 benchmarks/seca-paper/check_run.py \
  --run /tmp/seca-rerun/G-SlidingWindow-direct \
  --engine target/debug/examples/paper_stream --resume --ordering
python3 benchmarks/seca-paper/compare.py \
  benchmarks/seca-paper/evidence/final/G-Full-indirect \
  /tmp/seca-rerun/G-Full-indirect
```

Run regression tests independently of the full benchmark:

```bash
python3 -m unittest discover -s benchmarks/seca-paper/datasets -v
python3 -m unittest discover -s benchmarks/seca-paper/clustering -v
SECA_TEST_ENGINE="$PWD/target/debug/realtime-seca-cli" \
  python3 -m unittest discover -s benchmarks/seca-paper/evaluation \
  -t benchmarks/seca-paper -v
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```
