# State and invariants

`SecaEngine` owns the resumable model. Algorithmic interpretations are specified
in [algorithm.md](algorithm.md); mutation boundaries are mapped in
[architecture.md](architecture.md).

## Authoritative and derived state

- `hkt_build_output.hkts_by_id` owns live topology and ordered node contents.
  `nodes_by_id` is a synchronized lookup cache. Every cache entry must equal its
  owned node; each node belongs to exactly one HKT.
- `baseline_word_legend` is the append-only vocabulary, despite its historical
  field name. Positive word IDs are stable across updates, forgetting,
  reconstruction, and restoration. `-1` denotes refuge rather than a token.
- `baseline_source_legend` maps retained internal IDs to external IDs.
  `source_id_by_url` is its inverse; the external ID need not be a URL.
  `source_ids_by_batch_index` records retained provenance.
- `processed_batches` owns retained token payloads. Raw text, metadata, and
  timestamps are stripped. Per-word memberships are contextual indexes over those
  sources and may include candidate words that are not node descriptors.
- `next_hkt_id` and `next_node_id` exceed every allocated live ID and advance after
  reconstruction. Removed IDs are never reused.
- Last update explanation and batch summary serve existing reporting APIs; they
  do not decide reconstruction. Full-tree observations live only in reports.

There are no archived subtrees, logical tombstones, duplicate reverse source
legends, or per-source batch maps in live state. Historical analysis uses external
exports/snapshots owned by the caller.

## Identity and membership invariants

Source IDs must be nonempty. Each source's batch index must equal its enclosing
batch. Validation precedes deduplication, so malformed duplicates cannot hide an
invalid update. A source contributes once: the first occurrence in a batch wins,
and already retained identities are ignored. Initial construction applies the
same within-batch rule. An identity forgotten by gamma may be admitted again;
applications requiring lifetime deduplication must maintain an ingestion ledger.

Internal source IDs use deterministic FNV-1a hashing with collision probing.
Both directions of the retained identity map agree. Every retained identity occurs
in exactly one retained batch and its corresponding provenance set. Vocabulary
maps each token to one ID; tokens are trimmed, empty tokens excluded, and repeated
tokens count once toward document frequency.

There is exactly one Seed-HKT (`parent_node_id == 0`). Every other HKT is reachable
through its parent node. Node membership contains only retained source IDs.
Word memberships are subsets of node membership. Pending membership maps must be
empty after a successful update. The nodes cache has no extra or stale entries.
The vocabulary remains valid even when its words currently have no sources.

## Temporary state

Scope snapshots, mapped incoming indexes, candidate words, word measurements,
ancestor context, scoped reconstruction records, replacement trees, and trigger
plans are local to one transaction. Ancestry contains only excluded descriptors
on attachment nodes, accumulated along the path; there are no accepted/rejected
count maps. One ordered target map keeps each reconstruction's exact incoming
scope and exclusion set together. Its keys supply both the reported target IDs
and execution order; non-emptiness supplies the reconstruction flag.

Construction keeps each unfinished HKT and its original input scope local to
recursion, and generates its public node cache only after node mutations finish.
Replacement adds allocator offsets to owned IDs and derives the replacement cache
from those remapped nodes. Persistent engine cache validation remains unchanged.
Pending source and word memberships move into committed memberships at merge.

Candidate membership is separate from old
node vocabulary while errors are measured. A triggered parent suppresses child
inspection, ensuring selected reconstruction targets do not overlap.

## Reconstruction and forgetting

Reconstruction deletes exactly the target subtree from both hierarchy maps and
installs the new hierarchy with fresh IDs and the previous parent attachment.
Parent source unions propagate upward. Neither deleted node payloads nor old
subtree copies enter resumable state.

For gamma batches, retain batch indices at least `current - (gamma - 1)`.
Forgetting removes older token payloads, source legends, inverse identities,
batch provenance, ordinary memberships, pending memberships, and word membership
entries that become empty. It preserves vocabulary, node descriptors, expected
words, topology, and allocator counters. Empty updates still advance the cutoff.
`MemoryMode::Full` skips this removal for accumulated SECA applications; it uses
the same mapping and reconstruction path.

## Transactions and snapshots

Baseline construction and updates run on a clone. Failure at validation, mapping,
building, reporting, or any later stage leaves the original engine unchanged.
An established baseline cannot be overwritten through the initialization API.
The clone cost is intentional failure isolation; replacing it needs equivalent
rollback guarantees.

Schema-4 snapshots include the complete engine, configuration, batch position,
and engine version. Restoration rejects unsupported schema/version, absent state,
or inconsistent envelope metadata. It validates topology reachability, mirrored
nodes, source memberships, identity/provenance agreement, vocabulary uniqueness,
and allocator bounds. Every descriptor, expected word, top word, and word-membership
key must resolve in the persistent vocabulary; only node descriptors may use the
`-1` refuge sentinel. Retained identities must be nonempty. Retained batches must be consecutive and
strictly increasing, not in the
future, and, in SlidingWindow mode, at or after
`current.saturating_sub(gamma.saturating_sub(1))`. Provenance and both identity maps
must agree exactly with those retained batches. Full-memory snapshots have no
gamma cutoff. Configuration changes require explicit replay.
Earlier snapshots and removed configuration selectors are rejected rather than
silently selecting different semantics.

Serde JSON round trips must preserve the next update's reports and tree exactly.
A tree export is insufficient: it omits tokens and internal continuation state.
Filesystem atomicity belongs to the CLI; concurrent writers require application
coordination. A crash may leave an uncommitted tree export, but the old model file
remains usable until the state rename succeeds.
