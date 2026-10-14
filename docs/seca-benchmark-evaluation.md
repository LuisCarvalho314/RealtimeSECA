# SECA benchmark evaluation

This standard-library Python component owns evaluation and orchestration only.
It does not change Rust learning, dataset preparation or clustering procedures.
Base: `origin/integration/seca-correctness`, revision
`5ece43a71a40f08854aee0c2942791ec50443201`.

Scientific evidence: [SECA DOI](https://doi.org/10.1109/ACCESS.2023.3331219)
and the [author-posted accepted manuscript](https://www.researchgate.net/publication/375503068_Real-Time_Event_Detection_Using_Self-Evolving_Contextual_Analysis_SECA_Approach),
VIII-C4, VIII-D1, equations 18–25 and Table 3. The final PDF linked by
[University of Aberdeen](https://abdn.elsevierpure.com/en/publications/real-time-event-detection-using-self-evolving-contextual-analysis/)
returned HTTP 403 on 2026-10-14. Final-version verification remains outstanding.
No existing G/H benchmark plan was present on the base branch. The concurrent
Agent 1 manifest and Agent 2 request contracts were inspected read-only.

## Scientific status

**The runner can execute research and mock runs; paper reproduction is blocked.**
The publication does not fully specify all necessary evaluation decisions. None
is chosen by maximizing agreement with Table 3. `purpose: "paper"` is rejected.

| Question | Established evidence / unresolved consequence |
| --- | --- |
| NMI denominator | Equation 19 uses the arithmetic entropy mean: `2 MI / (H(W)+H(C))`. This is not geometric, maximum or minimum normalization. |
| Empty clusters | Zero mass contributes nothing to equations 20–22; empty predicted clusters are ignored. |
| Degenerate NMI | Both entropies zero gives 0/0. Return JSON null / blank CSV, not an invented score of 1. Empty evaluation populations likewise have undefined metrics. |
| BCubed | Equations 23–25 and the accompanying text describe document precision/recall and **mean individual-document F1**. This differs from taking the harmonic mean of corpus-average precision and recall. No overlapping BCubed variant or cited implementation is established by this section. |
| Direct overlap | VII-D direct uses Seed/Orphan source sets, allowing multiple memberships. No published conversion from these memberships to the hard partitions used by NMI is established. Retain raw overlaps and block scoring. |
| Indirect assignments | VII-D step 4 selects maximum proximity. Its ties, zero proximity, proxy merges and threshold remain unresolved; see Agent 2's clustering document when integrated. Evaluation never resolves these using labels. |
| Unassigned documents | The formulas do not establish whether to omit documents, form an extra cluster, create singletons or assign zero contributions. Each changes scores; block scoring rather than choose. |
| Ground-truth multilabel | The experimental description uses annotated categories. No multilabel rule is recovered. Reject multilabel truth; never convert it implicitly. |
| Purity | Implement the dominant **predicted cluster** category count, summed over clusters and divided by population N, as described immediately before equation 18. The equation's W/C explanatory notation reverses roles; following its literal category-majority reading instead would yield inverse purity. This inconsistency is documented, not silently treated as a new metric. |
| Scored population | Cumulative feeds and source ratios do not uniquely establish the scored population for each algorithm/cycle. Require an explicit research choice: current batch, retained window or cumulative. |
| Remaining parameters | Alpha=.7, beta=.5, AlphaError=.1, BetaError=.2, WIError=.3 are reported. Branch minimum, experimental gamma and indirect clustering settings are not recovered. Require explicit values and provenance/uncertainty text; no Rust defaults. |
| Dataset reconstruction | Preserve Agent 1's source revision, hashes, reconstruction warnings and batch order. Aggregate statistics do not prove exact paper source IDs or order. |
| Table 3 averaging | Report a clearly marked unweighted mean of all 12 cycle scores. Table 3 says average but does not fully establish weighting/population. No automatic parity claim or tolerance. |

`evaluation/metrics.py` implements the established complete hard-partition case.
For each document, with intersection count a, predicted-cluster size p and truth
category size t: precision=a/p, recall=a/t, F1=2a/(p+t). Means weight documents
equally. Purity weights clusters by their document counts. Source and predicted
cluster names are opaque independent identities. A contingency table makes the
calculation linear in documents plus occupied cells. Natural logarithms cancel
in normalized MI. Stable ordering and `math.fsum` produce repeatable output.
Scores are fractions; multiply by 100 to compare with paper percentages.

The shared assignment JSON is validated for duplicate population IDs, cluster IDs,
members within a cluster, truth source IDs, unknown members, missing truth, and
exact correspondence between explicitly unassigned IDs and missing memberships.
Extra truth outside the evaluation population is allowed. Repeated membership
**across** clusters is a valid boundary input but a scientific scoring blocker.
All raw assignments are written before scoring, including blocked inputs.

## Manifest and configuration

Consume Agent 1's schema version 1 manifest: `theme` G/H, ordered `batches`
containing `batch_index`, `path`, `sha256`; `ground_truth_path` and
`ground_truth_sha256`. Its `path_base: "selected_benchmark_data_directory"`
resolves paths relative to the parent of the theme manifest directory.
The synthetic fixture uses `path_base: "manifest_directory"`.
Paths must remain inside the selected directory. Verify every batch/truth hash,
zero-based contiguous batch order, source batch indices, globally unique source
IDs and complete truth before starting. Synthetic `row_N` IDs are rejected because
`timeline-many` scopes them; prepared MStream IDs retain identity.
Labels must remain in the separate truth JSON. Metadata and unexpected source
fields are rejected. The runner writes only source ID, batch index and tokens to
its engine input files.

See [fixture Full configuration](../benchmarks/seca-paper/evaluation/fixtures/config-Full.json)
and [fixture SlidingWindow configuration](../benchmarks/seca-paper/evaluation/fixtures/config-SlidingWindow.json).
These values are **synthetic smoke choices**, not recovered paper parameters.
Research configs must supply exactly `engine`, `evaluation_population`,
`clustering_settings`, `parameter_sources`, and `purpose: "research"`.
Every Rust config field must be explicit. Full mode uses null gamma; SlidingWindow
uses a positive integer gamma. Builder alpha/beta match .7/.5. The five reported
thresholds cannot be tuned. Provenance text must explicitly cover branch minimum,
gamma, clustering settings and evaluation population.

## Runner and clustering interface

Build the existing CLI from the source revision you intend to evaluate:

```sh
cargo build -p realtime-seca-cli --offline
SECA_TEST_ENGINE="$PWD/target/debug/realtime-seca-cli" python3 -m unittest discover \
  -s benchmarks/seca-paper/evaluation -t benchmarks/seca-paper -v
```

Run one small smoke fixture into a fresh output directory:

```sh
python3 benchmarks/seca-paper/run.py \
  --manifest benchmarks/seca-paper/evaluation/fixtures/manifest.json \
  --config benchmarks/seca-paper/evaluation/fixtures/config-Full.json \
  --engine target/debug/realtime-seca-cli --engine-revision "$(git rev-parse HEAD)" \
  --procedure direct --mock-assignments benchmarks/seca-paper/evaluation/fixtures \
  --smoke --output /tmp/seca-evaluation-smoke
```

Use the SlidingWindow fixture config for Light. Tests exercise both modes and
both procedure names with authored assignments; they **do not test a real
clustering algorithm**. G/H runs require exactly 12 batches without `--smoke`.
The integration agent owns complete experiments. This component never launches
all variants automatically.

The CLI `timeline-many` constructs one baseline from batch zero and calls
`process_batch` on the same engine for each later batch. It exports each read-only
verbose tree. The runner does not reconstruct from exports or restart per batch.
Evaluation follows the exports after the ordered learning pass; historical input
tokens remain caller-owned and are never reinjected into learning.
The CLI does not persist resumable snapshots for this command: failed benchmark
runs must be replayed into a fresh directory, not resumed from tree exports.

The adapter command receives JSON on stdin, and may use the literal argument
`{request}` to read the identical JSON from a temporary file. It returns the shared
assignment JSON on stdout. Request fields are `schema_version`, `batch_index`,
`method` (`direct`/`indirect`), `tree`, `evaluation_source_ids`, `source_tokens`,
`settings`, and (for indirect) explicit `contextual_threshold` from settings.
Tokens are trimmed and empty tokens removed to match engine normalization.
No ground-truth labels, truth paths or labels in metadata are supplied.
Once Agent 2 is integrated, invoke its file adapter with:

```sh
python3 benchmarks/seca-paper/run.py \
  --manifest /path/to/prepared/G/manifest.json --config /path/to/research-config.json \
  --engine target/debug/realtime-seca-cli --engine-revision "$(git rev-parse HEAD)" \
  --procedure indirect --output /path/to/new-run \
  --adapter-command python3 benchmarks/seca-paper/clustering/cluster.py '{request}'
```

Direct uses the same command with `--procedure direct`. No substitute clustering
implementation exists here. The mock adapter simply reads
`assignments-NNNN.json` fixtures. Using it requires `purpose: "mock-smoke"`.
Direct on cumulative Light populations will generally encounter expired,
unassigned sources: this is reported as a blocker, not repaired by restoring
expired evidence. External ground truth does not change the learning window.

## Artifacts and comparisons

Each fresh run directory contains the exact configuration, complete original
manifest and its SHA256, dataset hashes, source Git revision/status, declared engine
source revision and binary SHA256, adapter command and hashes of local adapter
file arguments, normalized engine inputs and their hashes, full ground truth,
read-only cycle trees, raw memberships, engine stdout/stderr, metric definitions
and fraction-valued per-cycle CSV. The declared engine revision is caller-attested;
binary hashing identifies its exact bytes but does not prove the build source.
Build it from the recorded revision. Dirty source state is explicitly recorded;
use clean commits for empirical regression references.

`run.json` transitions from running to complete, failed, or scientifically_blocked.
Only completed runs are compared. Partial CSV files are not completed runs.
CSV counts include only evaluated documents, nonempty predicted clusters and truth
categories represented in that population. Cycle is one-based (1–12); assignment
batch indices remain zero-based (0–11). Undefined scores are blank.

```sh
python3 benchmarks/seca-paper/compare.py /path/to/run-a /path/to/run-b
python3 benchmarks/seca-paper/compare.py /path/to/completed-12-cycle-run --published
```

The first reports right-minus-left metric deltas and configuration/dataset/metric
context changes, independently of HKT, node or predicted cluster IDs. A comparison
with different context is diagnostic, not evidence of a regression under fixed
conditions. The second compares all 12-cycle means to eight G/H SECA variants in
accepted-manuscript Table 3. BCubed precision/recall have no published values here.
Missing variants are rejected; incomplete cycles cannot be averaged. Output marks
mock/research purpose, aggregation uncertainty and pending final-PDF verification.
It never asserts parity. Original C# execution is unnecessary for recomputation or
benchmark-to-benchmark comparison.

## Verification

Run the unit suite without a Rust binary for metric/config/manifest/serialization
and comparison checks; real-engine tests skip explicitly unless `SECA_TEST_ENGINE`
is set. Hand-calculated perfect, merged, fragmented and asymmetric examples
verify F1 aggregation, NMI denominator, weighted purity and relabeling invariance.
Tests cover scientific failure preservation and separate truth/learning retention.
No numerical dependency is required. No production Rust or existing parity tests
were changed. Complete G/H experiments were not launched.

Validated on this branch: 22 Python tests passed with real-engine smoke enabled;
`cargo build -p realtime-seca-cli --offline` and all seven CLI tests passed.
The four nondegenerate NMI examples also matched installed scikit-learn 1.2.2
with explicit `average_method="arithmetic"` within 1e-14, using its
[documented normalization interface](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html).
Scikit-learn is a one-time cross-check, not a runtime or test-suite dependency.

## Integration verification

The integration agent verified the recovered final published PDF (SHA-256
`dacd87d997e81820079d85d2f4fd3466705a87b7d5839ba6b98ccc5208f44947`),
including a visual check of Table 3 on PDF page 15 / printed p. 127025.
The source comments and comparison output now identify the final version.
The earlier outstanding-final-PDF statements above record the component agent's
original evidence, and are superseded by this verification.

The integrated experimental `paper_stream` Rust example uses only public engine
methods and additionally saves reports and resumable snapshots after every cycle.
The original CLI remains supported. `--continue-blocked` preserves all research
cycles with null metrics where complete hard-partition scoring is undefined;
it does not choose a conversion policy. Exact retained identities are checked
against Full or the explicit gamma window on every export. `--learning-from`
can rescore saved learning after checking engine binary, configuration, input
hashes and dataset hashes. It records the original engine revision and tree hashes.
No tree export is treated as resumable engine state.

`recompute.py` independently calculates scores using set intersections rather
than the evaluator's contingency table. `check_run.py` compares the next update
after snapshots at cycles 1 and 6 and diagnoses within-batch source reversal.
The integrated indirect tie rule uses sorted token/weight signatures rather than
node IDs; see the clustering document. These are explicit research choices,
not newly established original experimental settings.

See [the integrated scientific report](seca-paper-benchmark.md) for the fixed
reconstruction, run status, score comparison, scientific limitations and commands.
