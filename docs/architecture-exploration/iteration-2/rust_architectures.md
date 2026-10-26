# Native Rust architectures

Both experiments are isolated in [`rust-candidates/`](rust-candidates/), a standalone
Rust package outside the production workspace. They accept ordered normalized
`(source:i64,word:i32)` incidence, alpha, beta and an exclusive branch minimum.
They do not call the production builder or engine. The separate
[`evaluation-baseline/`](evaluation-baseline/) package alone depends on production.
The comparison contract and exact exclusions are in
[evolution_validation.md](evolution_validation.md).

## Direct set construction

The authoritative result is one owned recursive `Scope` with ordered `Node`s.
Each node owns descriptors, source memberships, contextual per-word memberships,
top words and an optional owned child. Numeric HKT/node IDs are allocated once
in depth-first order and retained in this representation. There is no global
node-payload mirror, mutable row deletion, archived construction scopes or second
hierarchy owner. Construction returns a value; an error cannot install a partial
hierarchy into a caller's model. The single `Option` child covers reachable
constructor output; it cannot express every historical schema-4 accepted shape
with multiple child HKTs attached to one parent node. Removing production
structures requires an explicit adapter for those accepted shapes.

Input is grouped into word incidences with `BTreeSet<i64>` membership. Each
incidence also retains the first original input ordinal for **each source**.
That is semantically necessary ordering information, not a cached document
frequency: a child scope can exclude the source containing a word's first row.
The incidence currently keeps both a source set and a source-to-ordinal map
with equal domains. Relation preparation and child restriction must maintain that
local equality invariant. This is a real additional representation, even though
it is operation-local rather than a mirrored hierarchy. An ordinal map alone
could be authoritative, but that unmeasured variant was not substituted here.
The child intersects each word's source set with its attachment's sources and
recomputes the minimum surviving ordinal. Frequencies are set lengths. Stable
frequency/rank sorting chooses the seed, then ascending admitted word IDs drive
the greedy sequential accumulator. A candidate maximizes intersection cardinality
divided by current node cardinality; strict improvement preserves first-maximum
ties. Adoption immediately unions memberships before the next candidate.

Refuge is the uncovered source complement. Each child sees the unmodified parent
scope, with only attachment descriptors excluded. Expected-word bookkeeping and
top-word annotations follow the operational Rust contract even where that differs
from a cleaner scientific interpretation. Public construction IDs, expected words,
top words and contextual memberships are compared exactly, not only simplified
shape. This normalized contract derives accurate DF from incidence and interprets row
position as the original source-word ordinal. It does not preserve arbitrary
caller-supplied `source_word_id` values or inconsistent public `word_number_of_sources`
fields. The direct implementation was reconstructed from this contract; it is not
a line-by-line Python port.

`construct` handles grouping; `refine_node` handles scoped refinement. Input
normalization, ordering, set collision selection and annotations are explicit
ordinary functions. Standard ordered collections keep the primitive operations
visible and avoid a generic collection or policy framework. Recomputing child
incidences still allocates memberships and ranks; the implementation does not
claim to eliminate all scope copies. Recursive ownership also does not establish
a safe arbitrary-depth bound.

## Dense incidence competitor

The `--masks` variant is deliberately a smaller construction collision kernel,
not a second complete engine. At each scope it indexes the union of source IDs
once, constructs dense `Vec<u64>` columns for every word, and maintains growing
node masks. Intersection uses bitwise AND/popcount; adoption updates a node with
OR. The original scalar candidate order and first-maximum rule remain unchanged.
Dense masks contain only local positions; public stable IDs remain in the owned
set hierarchy and output. Scope ranking, child restriction and output decoding
still use the set model. Construction measurements include indexing and column
conversion; they are not hidden in an untimed setup adapter.

This improves the earlier per-candidate re-encoding experiment, but it is not an
end-to-end incidence-only representation. It intentionally retains both sets and
masks while the kernel is evaluated. Therefore its measured overhead cannot
establish the best possible performance or memory of a fully indexed engine,
and its code size is additional to the direct path. A persistent dense source
index would need explicit expiry/readmission/remapping invariants. No compressed
bitmap dependency was added: the available representative retained workloads do
not establish density or long-run churn sufficient to justify one. Ordinary dense
vectors provide an executable dependency-free comparator first.

## Independent normalized evolution

`evolution::Engine` owns one hierarchy, ordered retained batches of normalized
source word payloads, the append-only normalized vocabulary, current batch,
retention settings, error thresholds and monotonic topology allocators. It
implements validation-before-deduplication, retained-identity deduplication,
source mapping to every matching ordinary node or refuge, candidate strength
admission/rechecking, incoming-only candidate beta, contextual errors and top-down
trigger suppression. Contextual measurement is split into source mapping,
candidate admission, per-word measurement and error reduction helpers; the
scientifically disputed operational formulas are preserved rather than revised.

Accepted scopes recurse before merging new evidence. Retained parent scope is
passed explicitly, preserving the production distinction between old evidence
and incoming evidence. Trigger targets retain exactly their incoming scope and
ancestor exclusions. Targets are rebuilt in ascending old HKT-ID order with the
same direct constructor, fresh allocator positions, unaffected sibling ownership
preserved, and source propagation up the parent chain. Forgetting then removes
expired evidence and every contextual source membership while retaining learned
words/topology. Empty batches advance the window. The alternative never rebuilds
an entire window as a fallback for unsupported evolution.

A clone remains the transaction boundary: `process` mutates a private candidate
and installs it only after success. The owned hierarchy removes the mirror
synchronization obligation, but that does not make incremental rollback free.
No undo log or persistent collection was introduced without demonstrated need.
Serde roundtrip continuation is executable for this research state. It is **not**
a schema-4 adapter or a replacement for production's restore validation.

The normalized boundary deliberately excludes external string source IDs,
FNV collision probing, token trimming, string-to-vocabulary registration, source
batch-index fields, production `is_state1` annotations, diagnostics/report strings and schema-4 accepted
snapshot shapes. A wordless initial source cannot be expressed by incidence-only
construction input. Empty-root evolution is explicitly rejected; it is not
silently delegated to production. Arbitrary mutation of the public research tree
or deserialized research engine is not a validated ingestion interface. A natural unsupported empty-root update tests rollback after vocabulary
registration; failure after accepted-parent mutation or midway through replacement
remains untested. Allocation
failure, stack exhaustion and topology allocator overflow are not recovered.
These are remaining engineering obligations, not passing compatibility claims.

## Complexity and review boundary

The direct builder removes row mutation bookkeeping and duplicate hierarchy
ownership. It still needs contextual membership sets, source-ordinal ordering,
scoped incidence transformations, learned descriptors, retained evidence and
allocators. The lifecycle demonstrates why a small construction kernel alone is
not a meaningful complete rewrite line count.

At the source checkpoint before benchmark instrumentation, physical non-test
Rust lines (including comments/blanks) were: `lib.rs`290, `evolution.rs`391,
`evolution/context.rs`246 and binary driver27, total954. `lib.rs` includes the
secondary dense-mask kernel and shared models/projection, not only the direct
constructor. Its inline construction tests add60 lines; independent integration
tests are counted separately. The final measured counts and adapter cost are in
[benchmark_results.md](benchmark_results.md); instrumentation can change these
checkpoint counts. None of these totals is production-equivalent because the
public registration, persistence validation and reporting boundaries are absent.

Every source file was reviewed against the 400-line guideline. The remaining
81-line `visit` function is one cohesive ordered traversal: evaluate, stop at a
trigger, otherwise recurse in child-ID order and merge memberships. Its explicit
arguments expose the state/evidence boundary; a new generic controller would not
simplify it. Larger construction and context functions were split into concrete
mathematical helpers. Dependencies are `serde` and `serde_json`, already used by
production; no bitmap, graph, database or persistent-collection library was added.

## Validation

Standalone `cargo test --offline`, `cargo fmt -- --check` and
`cargo clippy --offline --all-targets -- -D warnings` pass; logs are in `raw/`.
Five construction unit tests include a 100-case mask/set exact-field comparison,
a minimal first-surviving-incidence regression, duplicate incidence/input
immutability, seed ties and invalid/empty input. Twelve independent evolution tests
cover the detailed lifecycle invariants described in
[evolution_validation.md](evolution_validation.md). Larger production differential
results and release measurements are reported separately; unit agreement between
two kernels sharing a controller is not independent proof of that controller.

Reproduce from repository root:

```sh
cargo test --offline --manifest-path docs/architecture-exploration/iteration-2/rust-candidates/Cargo.toml
cargo fmt --manifest-path docs/architecture-exploration/iteration-2/rust-candidates/Cargo.toml -- --check
cargo clippy --offline --manifest-path docs/architecture-exploration/iteration-2/rust-candidates/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path docs/architecture-exploration/iteration-2/rust-candidates/Cargo.toml
```
