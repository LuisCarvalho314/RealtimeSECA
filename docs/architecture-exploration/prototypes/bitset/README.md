# Bit-column incidence algebra experiment

`seca_bitset.py` represents source–word evidence as an immutable tuple of integer
columns: bit `s` of column `w` means source `s` contains word `w`. Construction
passes only a source mask and available column indices to child scopes. It never
materializes child source-word datasets, mutable membership sets, duplicated node
indexes, cached scope frequencies, or source-word records per child. Intersection,
union, difference and document frequency become AND, OR, masked complement and
population count. The sequential greedy node admission remains necessary because
a union changes the denominator of later word admissions.

Run from repository root:

```sh
python3 -m unittest discover -s docs/architecture-exploration/prototypes/bitset -v
python3 docs/architecture-exploration/prototypes/bitset/seca_bitset.py < tools/seca-reference-harness/fixtures/stable.json
```

The JSON CLI accepts the locked harness input, but **builds only its first batch**;
it does not claim update parity. Alternatively provide
`{"records":[[1,1],[2,1],[1,2]],"alpha":0.7,"beta":0.5,"branchMinimum":1}`.
`Incidence.from_records(records).build(alpha,beta,branch_minimum)` returns ordered
nested descriptors with sorted word/source IDs and a `child` descriptor per node.
Dense bit indices do not change external source IDs. Input row order remains as
small integer pairs solely because construction ties depend on the earliest
surviving original row. Duplicate source-word evidence is collapsed mathematically;
production rejects duplicate source identities before this boundary.

`scope_errors` independently computes full precision alpha, beta and importance
errors from a **prepared stage**. It preserves per-node count aggregation and
last-node word-metric overwrite in Rust. Its argument contract deliberately exposes
retained/incoming and candidate counts so it cannot silently assume that paper
set notation resolves existing ambiguities. Tests cover the stable and novel-word
metric examples algebraically, without claiming executed reference update parity.

## Scope and limitations

Implemented: complete recursive baseline hierarchy construction, greedy union and
eligibility, refuge nodes, ancestor-word exclusion by column removal, document
frequency sorting and original row ties; separate staged error kernel.

Omitted: candidate discovery/admission, incoming-scope mapping, selective rebuild
control, retention and vocabulary persistence, IDs, top-word presentation,
transactional failure handling, ingestion validation, Rust integration. Python
recursion can fail on deeply nested hierarchies. This is a computational experiment,
not a SECA-Light replacement. No reference, oracle or production file is modified.

Construction targets current Rust's **exclusive** branch cutoff, **first** maximum
eligibility tie, and explicit expected-word selection. Inspection of original
C# `HKT.addOtherNodes` found last maximum eligibility ties and, on noncollision,
selection of the remaining first frequency-ordered row's word rather than the
loop's expected word. C# also uses inclusive branch cutoffs. These are distinct
compatibility questions; a successful Rust construction comparison does not prove
cross-language equivalence. C# floating errors are rounded; this kernel returns
full precision, as the Rust/documented contract requires.

Bit columns may save substantial per-membership allocation for moderately dense
news scopes, but sparse high-cardinality vocabularies allocate through the highest
set bit. Parent-column AND processes whole machine-word ranges even for tiny
child scopes. The prototype scans original integer row pairs to recover child tie
rank; a maintained sparse row-rank index could improve this but would add state.
Any production choice needs representative density and depth measurements.
