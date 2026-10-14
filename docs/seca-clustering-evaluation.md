# SECA event clustering evaluation

## Evidence and status (2026-10-14)

This component does not modify learning. Base: `origin/integration/seca-correctness`
at `5ece43a71a40f08854aee0c2942791ec50443201`.

Scientific source: Al Sulaimani and Starkey (2023),
[DOI](https://doi.org/10.1109/ACCESS.2023.3331219), sections V-B and VII-D,
equations 13–14. The [author-posted accepted manuscript](https://www.researchgate.net/publication/375503068_Real-Time_Event_Detection_Using_Self-Evolving_Contextual_Analysis_SECA_Approach)
was readable. The university's final-version PDF and author PDF downloads returned
HTTP 403; correspondence with the final typeset version has not been verified.

Published procedure: direct clusters are Seed/Orphan nodes' source sets. Orphans
have only refuge ancestors. Indirect proxies include branch words weighted by
node support divided by event-root support (13). Proxies sharing a top word merge;
sources select the greatest sum of matching contextual values (14).

The supplied `references/seca-csharp/SECA/` archive was searched across C#, Python
and SQL for cluster, proxy, proximity and contextual-value operations. No event
clustering implementation was located. SQL `CLUSTERED` indexes are unrelated.
HKT construction/evolution code is available, but provides no executed evidence
for these postprocessing rules. No C# clustering discrepancy or defect is proven.
No restricted reference source or binaries are included here.

**Direct is implemented. Indirect is executable under the single interpretation
below; reproduction of the published indirect results remains scientifically
blocked.** The missing experimental threshold and unresolved rules are not claimed
to be recovered from C#. These decisions were recorded before implementation.

## Contract and population

The Python standard-library module consumes `export_baseline_tree_verbose()` JSON
directly, plus `batch_index` and an explicit `evaluation_source_ids` list. It
validates tree reachability, ownership, source legends and child membership scope.
Input is read-only; failures produce no assignment. No ground-truth field is read.

Direct memberships are the exported, retained sets intersected with the requested
population. A normal child is not another event; a refuge below a normal node
does not create an orphan event. Following only refuge ancestors from Seed-HKT
identifies event roots. Refuge nodes themselves are never event clusters. Sources
in several event roots remain in several clusters. Repeated membership within a
set is deduplicated. No descendant union or hard-label conversion is applied.
Empty event clusters are omitted; the population minus assigned sources is
reported as unassigned, including sources absent from the retained export.

`batch_index` is evaluation metadata, not a filter or a source of provenance.
Callers select current-batch or retained populations explicitly. The publication
does not unambiguously fix the scored population for every reported experiment.
Recommend reporting both population policy and engine memory mode in experiment
metadata. The adapter cannot recover forgotten memberships. It must not treat
unassigned sources as an extra event or silently omit them from the population.

Cluster IDs encode event-root node IDs, never labels. IDs are deterministic for a
given export, but may change on reconstruction; they do not track events through
time. Merged IDs contain numerically sorted root IDs. Output clusters and source
IDs are sorted, and JSON keys are sorted. Export array ordering is irrelevant.

## Indirect interpretation and open scientific decisions

Only one procedure is implemented; there are no compatibility algorithms. These
are **repository interpretations requiring scientific review**, not paper claims.

| Issue | Consequence/example | Implemented interpretation and rationale |
| --- | --- | --- |
| Branch extent | Ignoring descendants would make all root descriptors weight 1 and discard hierarchical context. | Include every ordinary descendant of an event root, including descendants reached through an internal refuge. Such nodes remain contextual to that event. Ignore refuge sentinels and expected words. |
| Repeated word within a proxy | Root support 4; two descendants carry x with supports 2 and 1: possible weights are 1/2, 3/4 or 1/4. | Maximum occurrence weight, 1/2. Preserve the strongest available context without counting the same source word twice. This reduction is not specified by equation 13. |
| Top/important words and threshold | A shared word weighted 1/2 merges at cutoff 1/2 but not 3/4. A top-k rule would behave differently. | Require an explicit contextual-value cutoff in (0,1], with inclusive comparison. No default or claimed published value. Exact rational comparisons preserve ties. This interprets the unspecified threshold as contextual importance. |
| Transitive merge | A shares x with B; B shares y with C; A and C share none. | Connected components of the original top-word overlap graph merge all three. This satisfies every original pairwise merge independently of iteration order. Do not recompute top words during merging. |
| Merged contextual values | A:x=1/2 and B:x=3/4 could become 5/4, 3/4 or a recomputed ratio. | Maximum, 3/4, preserves the contextual-value scale and counts x once. Recomputing equation 13 would require a new definition of a merged root. |
| Positive assignment ties | Source x has equal score for independent events. | Choose the numerically smallest root-ID tuple. A deterministic engineering rule, not a published tie rule. Indirect remains hard clustering. |
| Zero proximity / no proxies | An empty source has equal zero proximity to all events. | Leave unassigned rather than assert an event with no matching evidence. The paper does not specify this exception to its maximum rule. |
| Zero root support after forgetting | Equation 13 would divide by zero. | Omit empty roots; reject nonempty descendants under an empty root as invalid scope. |

Proximity sums weights of **distinct** matching source words. It uses neither term
frequency, cosine/Jaccard normalization, node-word eligibility, nor a centroid.
Contextual values use full retained support, even when evaluating only the latest
batch. Restricting support to the evaluation subset would change the learned
context without a learning update. Fractions avoid floating-point tie changes.

The exported `top_words` are deliberately unused: `hkt_builder.rs::add_node_top_words`
starts with descriptors and fills from the sorted HKT scope up to ten, potentially
including words outside the node's event branch. This is not evidence for VII-D's
contextual importance threshold. Reusing it would silently import a different rule.

## Essential input and smallest proposed read-only addition

Verbose export is sufficient for direct clustering and equation 13: it includes
parent attachments, descriptors, distinct memberships and external identities.
Equation 14 additionally needs each evaluated source's complete normalized token
set. Descriptor lists and exported top words cannot reconstruct this information.
The module requires `source_tokens: {external_source_id: [token, ...]}` for indirect
clustering. Every evaluation ID must have a token entry, including empty documents.
Caller-owned tokenized `SourceBatch` inputs can supply these sets without a new
engine API. Tokens must use the same normalization as engine ingestion; the
adapter does no case folding, stemming or label-based selection.

If the integration agent needs a self-contained retained-evidence export, propose
only a read-only list `{external_source_id, batch_index, tokens}` built from retained
processed batches. No raw text, metadata, word memberships, allocator state or
private snapshot layout is needed. Nothing is added to production here. Callers
must archive evidence themselves to evaluate documents after forgetting.

## Usage and validation

Run `python3 -m unittest discover -s benchmarks/seca-paper/clustering -v`.
CLI: `python3 benchmarks/seca-paper/clustering/cluster.py request.json`, writing
one assignment JSON object to stdout; validation errors go to stderr with exit 2.
Requests have `method` (`direct` or `indirect`), `tree` (the verbose export),
`batch_index`, and `evaluation_source_ids`. Indirect additionally requires
`source_tokens` and `contextual_threshold`, preferably an exact string such as
`"1/2"`. Do not infer a threshold from labels. Record the supplied threshold and
this interpretation document alongside each run; the shared assignment output
contains only population and memberships.

Hand-derived tests explain their expected sets and fractions in docstrings. They
cover one/multiple events, true orphan chains versus internal refuge branches,
overlap, empty/unassigned populations, contextual denominators, repeated words,
threshold equality, transitive merging, merged values, hard ties, zero matches,
permutation/repeat stability and rejected malformed evidence without input mutation.
Synthetic tests validate these rules; they do not establish published-score parity.

Validation completed: all 20 unit tests passed. An offline build and execution of
`cargo run --offline -p realtime-seca-cli -- baseline sample_batch.json --config
sample_config.json --dump-tree-verbose /tmp/seca-clustering-tree.json` succeeded.
Both adapters consumed that actual export with the sample batch token sets and
accounted for all three external source IDs. Production Rust files were unchanged;
the full Rust workspace test/clippy suite was not run for this Python-only addition.
