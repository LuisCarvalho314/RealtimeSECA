# SECA paper benchmark datasets: Themes G and H

Exact paper dataset reconstruction has **not** been established. This preparation
recovers the public preprocessed corpora and creates a deterministic G/H
reconstruction. It does not run SECA or establish experimental parity. Themes
A–F, clustering, baselines and evaluation are outside its scope.

## Paper evidence and variant decision

The original published paper is Al Sulaimani and Starkey, *Real-Time Event
Detection Using Self-Evolving Contextual Analysis (SECA) Approach*, IEEE Access 11
(2023), 127011–127034, [DOI](https://doi.org/10.1109/ACCESS.2023.3331219).
The published PDF was recovered through the
[authors' university repository](https://abdn.elsevierpure.com/ws/portalfiles/portal/284651535/Al_Sulaimani_etal_IEEEA_Real_Time_Event_VoR.pdf).
Its SHA-256 is
`dacd87d997e81820079d85d2f4fd3466705a87b7d5839ba6b98ccc5208f44947`.
Section VIII-B, VIII-C.1, Table 2 (published p. 127023, PDF page 13), and
reference [51] were checked against this published version. The paper describes
preprocessed News/Tweets, whitespace tokenization, and truncation for G/H. It
cites MStream with an access date of 2022-10-21. It does not identify a variant
filename, a truncation rule, or ordered document IDs.

The following are measured results of taking a prefix of each public file:

| Variant | Full documents | Full labels | Prefix documents | Prefix labels | Prefix event size min–max |
| --- | ---: | ---: | ---: | ---: | ---: |
| News | 11,109 | 152 | 8,880 | 152 | 2–354 |
| News-T | 11,109 | 152 | 8,880 | 98 | 3–430 |
| Tweets | 30,322 | 269 | 24,000 | 255 | 1–607 |
| Tweets-T | 30,322 | 269 | 24,000 | 227 | 1–607 |

Table 2 reports G: 8,880 documents, 740 per cycle, 152 events, event sizes
2–354; H: 24,000 documents, 2,000 per cycle, 227 events, event sizes 1–607.
**We infer News for G and Tweets-T for H**, since their prefixes match all these
aggregate counts. This inference conflicts with treating the paper's generic
name “Tweets” as the literal filename `Tweets`. The public files cannot resolve
that ambiguity conclusively. Neither the aggregate match nor the paper's
reference to natural evolution proves the selected identities or sequence.

Our explicit reconstruction takes the first 8,880 physical JSONL records of
`News` and the first 24,000 of `Tweets-T`, preserving the published file order and
splitting into 12 consecutive batches. Batch indices are 0–11 (paper cycles 1–12).
The remaining 2,229 and 6,322 records respectively are excluded and reported in
both manifests and preparation output. This is a chosen prefix truncation, not a
recovered original truncation procedure. No other filtering or reordering occurs.

Precise remaining uncertainty: which tweet variant the authors imported, which
records they removed, whether truncation was a global prefix or occurred within
previous partitions, and whether their database import/query preserved file
order. An original ordered ID list for each cycle, or the authors' exact
selection/import script and dataset snapshot, is required to establish exactness.
The chosen current files have no timestamps; historical timestamps establish
that `Tweets-T` itself is not chronological (see the historical audit below). No missing original data or A–F samples have been fabricated.

## Recovered provenance

The source is [jackyin12/MStream](https://github.com/jackyin12/MStream/tree/ac894075f2bbb93405679e237d2d5d11a325078c/MStream/data),
pinned to `ac894075f2bbb93405679e237d2d5d11a325078c` (2019-12-25). This is
also the latest commit in the repository history before the paper's cited access
date. The four files were last changed in December 2018; their Git blob IDs,
last-change revisions, byte counts, raw hashes, and order-sensitive normalized
hashes are recorded in
[`sources.json`](../benchmarks/seca-paper/datasets/sources.json).
The downloaded raw bytes retain upstream CRLF newlines. SHA-256 is checked
before parsing; identical logical records with different byte encoding fail.

| File | Raw SHA-256 |
| --- | --- |
| News | `f68afe2771048371a84fb97f0e8599b437b178117dd3678ec16a52c30b36cfad` |
| News-T | `562d6a1771a4675caedb405d68371348bd7e54db130b633ef5ebd5896157ffbb` |
| Tweets | `126d6f0a4fd76abfdb76f555081ed44af4f185d8a202197906886fcf09927b98` |
| Tweets-T | `e1b806af9992e881ff75b76efe259dff8b3483d9be11f1110a777356075070cf` |

Each JSONL record contains exactly `Id`, `clusterNo`, and `textCleaned`. All four
files have unique, nonempty original IDs and nonempty cleaned texts. IDs span
`000001`–`011109` for News variants and `000001`–`030322` for tweet variants.
`News`, `Tweets`, and `Tweets-T` are in sequential ID order; `News-T` is not.
These are external corpus identifiers; the historical audit below recovers the
earlier native tweet IDs separately.
Leading zeros are preserved. Labels are positive integers; News uses 1–152,
whereas tweet labels range from 1–439 with only 269 distinct values. Labels are
not compacted or renamed.

Both variant pairs have exactly the same multiset of `(clusterNo, textCleaned)`
records. News variants retain the same ID-to-label/text mapping with different
orders. Tweet variants reuse sequential IDs for different label/text records;
the same numeric ID is therefore not a cross-variant identity. Source IDs are
`mstream:News:<Id>` for G and `mstream:Tweets-T:<Id>` for H, including the
original zero-padded ID verbatim. They contain no label or generated row identity.

A second audit checked the initial public dataset snapshot at revision
`77770aea5580b668c69f17b0b6c3d2795016987d`. Its facts and raw hashes are in
[`historical-audit.json`](../benchmarks/seca-paper/datasets/historical-audit.json).
At that revision News IDs were called `tweetId` but had the same values as today's
`Id`. Tweet files were called `TREC` and `TREC-T`; they included native `tweetId`
and `timeStr`. Current variants preserve their respective historical texts,
labels and row orders exactly, but replace native tweet IDs with sequential
six-digit IDs. Both historical tweet corpora have unique native IDs. Parsing
`timeStr` with `%a %b %d %H:%M:%S %z %Y` gives zero adjacent time descents in
`TREC`, and 15,123 in `TREC-T`.

This verifies chronological order for the public `Tweets` variant and rejects
that interpretation for `Tweets-T`. It sharpens the unresolved tension between
H's matching 227-event aggregate and the paper's natural-evolution description.
The preparation retains the IDs in the snapshot available at the cited access
date. An optional audit script writes a separate correspondence file for all
four variants, relating qualified current IDs to historical `tweetId` and
`timeStr`; it does not inject these fields into learner input or change batches.

Identical text does not imply the same source or even the same label:

| Corpus | Excess identical texts | Excess identical label/text pairs |
| --- | ---: | ---: |
| Full News (either variant) | 350 | 248 |
| Full Tweets (either variant) | 4,336 | 4,329 |
| Prepared G | 217 | 157 |
| Prepared H | 2,187 | 2,181 |

“Excess” counts total records minus distinct values, not distinct duplicated
groups. Every duplicate remains present under its original ID. Repeated tokens
inside documents also remain present.

## Preprocessing and learner boundary

The upstream MStream paper, Yin et al., *Model-based Clustering of Short Text
Streams* (KDD 2018), section 5.1,
[DOI](https://doi.org/10.1145/3219819.3220094), describes lowercase conversion,
stop-word removal and stemming. It describes the `-T` variants as topic-sorted,
then divided into 16 parts and shuffled within each part. The published files
are authoritative here: their shuffle is not rerun and its seed is not needed.
The exact earlier cleaner, stemmer and stop-word list were not established, and
no attempt is made to recreate preprocessed text from raw documents.

The scripts use `textCleaned` verbatim as optional `text`, with Python
`str.split()` for whitespace tokenization. On these source files this agrees
with splitting on ASCII spaces and discarding empty tokens: no cleaned text
contains other whitespace. Token order, spelling, case and multiplicity are
preserved. Upstream `Document.py` uses space splitting and later reduces token
frequency to binary occurrence; this preparation preserves repetitions in the
Rust token vector rather than applying that learner transformation.

The JSON matches `SourceBatch { batch_index, sources }`. Each source has
`source_id`, matching `batch_index`, `tokens`, `text`, null `timestamp_unix_ms`
and null `metadata`. No source label, `clusterNo`, original labeled JSON object,
or label-derived feature enters these files. Corpus text can naturally contain
numbers; label isolation means that annotations are never injected into input,
not that numeric tokens are removed. No CSV ingestion or additional cleaning is
used. Core normalization on later ingestion is a separate engine contract; this
preparation neither changes nor executes it.

Ground truth is a separate ordered JSON array:

```json
[{"source_id": "mstream:News:000001", "label": "65"}]
```

Every included document has exactly one label. Decimal string serialization
preserves the upstream category value without asserting event names or hierarchy.

## Manifest schema version 1

Committed reference manifests are
[`G.json`](../benchmarks/seca-paper/datasets/manifests/G.json) and
[`H.json`](../benchmarks/seca-paper/datasets/manifests/H.json).
Preparation also writes `G/manifest.json` and `H/manifest.json` alongside data.
All paths resolve from the **selected prepared benchmark data directory**
(default `benchmarks/seca-paper/datasets/data/prepared`), regardless of where a
manifest is stored. Paths use relative POSIX syntax. No machine paths or run
timestamps are serialized.

| Field | Contract |
| --- | --- |
| `schema_version` | Integer 1; incompatible changes require a new version. |
| `theme`, `original_dataset_name`, `source_variant` | G/H, paper's dataset name, and exact chosen upstream filename. |
| `path_base` | Literal `selected_benchmark_data_directory`. |
| `source` | Repository URL, pinned revision, provenance and checksums for all four investigated variants. Its file paths are relative to the upstream repository. |
| `documents`, `batch_size`, `batch_count` | Selected count, fixed batch size and 12 cycles. |
| `batches` | Ordered array; each entry has `batch_index`, relative `path`, `documents`, inclusive one-based upstream `source_line_start`/`source_line_end`, and raw output `sha256`. |
| `ground_truth_path`, `ground_truth_sha256` | Relative path and raw-byte digest of the separate annotation array. |
| `preprocessing`, `source_id_convention`, `label_convention` | Input transformation and identity/annotation rules. |
| `reconstruction` | Separate selection/ordering statuses, explicit false `exact_paper_reconstruction`, recipe, tail count, variant evidence, uncertainty. |
| `selected_statistics` | Counts, duplicates, label sizes, token count, ordered ID/record hashes. |

Normalized ordered hashes use UTF-8 JSON with `ensure_ascii=False`, two-space
indentation and one final newline. `ordered_ids_sha256` hashes the array of
original IDs, and `ordered_records_sha256` hashes the array of original objects
in original field and document order. The tweet variants share an ordered ID
hash but have different ordered record hashes, illustrating why ID order alone
cannot establish equivalence. Batch and ground-truth hashes always hash the
actual serialized bytes.

## Rebuild and validate

Run from the repository root; Python 3.9+ and its standard library suffice.
Network access is needed only for downloading:

```bash
python3 benchmarks/seca-paper/datasets/download.py
python3 benchmarks/seca-paper/datasets/prepare.py
python3 benchmarks/seca-paper/datasets/verify.py
python3 -m unittest discover -s benchmarks/seca-paper/datasets -v
```

For a new rebuild, choose a fresh output directory rather than deleting an
existing result:

```bash
python3 benchmarks/seca-paper/datasets/download.py --source-dir benchmarks/seca-paper/datasets/data/raw-fresh
python3 benchmarks/seca-paper/datasets/prepare.py --source-dir benchmarks/seca-paper/datasets/data/raw-fresh --output-dir benchmarks/seca-paper/datasets/data/prepared-fresh
python3 benchmarks/seca-paper/datasets/verify.py --source-dir benchmarks/seca-paper/datasets/data/raw-fresh --data-dir benchmarks/seca-paper/datasets/data/prepared-fresh
SECA_DATASET_SOURCE_DIR=benchmarks/seca-paper/datasets/data/raw-fresh python3 -m unittest discover -s benchmarks/seca-paper/datasets -v
```

To reproduce the optional historical audit and recover native tweet identities
from scratch, clone the public repository outside the tracked tree, then run:

```bash
git clone https://github.com/jackyin12/MStream.git benchmarks/seca-paper/datasets/data/upstream
python3 benchmarks/seca-paper/datasets/audit_history.py --upstream-checkout benchmarks/seca-paper/datasets/data/upstream
```

The script checks historical SHA-256, record correspondence, original IDs and
chronology against the committed audit, then creates the ignored
`data/historical-identities.json`. The historical source snapshot is read by
commit ID, regardless of the clone's working branch. Existing identity output
is refused. This audit is not needed to rebuild or consume G/H batches.

A custom data directory can be selected with these same flags; consumers resolve
manifest paths from that selected directory. Keep custom outputs outside Git or
inside the ignored `data/` directory. The pinned source lock and committed
manifests do not need regeneration. Tests require all four raw files and fail
clearly if they are unavailable; they never download or silently skip them.

The downloader verifies all four responses/cached files before installing any
new raw file. Corrupt cached files fail rather than being silently redownloaded.
Preparation verifies the full source files and stages all outputs before
publishing the directory. An existing output directory is refused, preserving
previous results. SHA mismatch, missing file, invalid record, repeated identity,
or incomplete source causes failure. Verification checks exact bytes and file
inventory against the pinned recipe, detecting changed annotations, label
injection, order changes, missing batches and unexpected files.

The dataset suite covers all nine requested validation properties, actual
variant differences, retained duplicates/repeated tokens, CLI errors and
publication failure. Initial validation: **13 tests passed**, and verification
accepted **28 generated files**. Rust engine files and module boundaries were
not changed; no clustering or scientific/parity tests were executed by this task.

## Redistribution

No dataset-specific redistribution grant or repository LICENSE was found in the
inspected MStream revision. Availability on GitHub alone is not treated as a
license grant. Raw downloads, generated batches and ground-truth annotations
remain in the dataset directory's Git-ignored `data/` subtree. Only scripts,
provenance/statistics, hashes, reference manifests and this report are committed.
The original papers are cited and are not redistributed. Downloading and running
the commands above reconstructs all deliverables without including text or
annotation payloads in the branch.
