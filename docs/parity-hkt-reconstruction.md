# HKT construction and selective reconstruction parity audit

## Scope and evidence

Audited checkout: `15814d1d636a853acbf0ca950d44cce4c534b3e4` (initial branch
`fix/seca-light-stream`). Local `main` is `42c07c098c1fadb931a4e7b5b6920ee9561dabc5`;
this audit preserves the supplied checkout rather than discarding its newer
implementation. Work is on `audit/hkt-reconstruction-parity`. Unrelated CLI edits
were present during the audit and are excluded from this commit. Other workers
committed CLI and state-invariant changes in the shared checkout while this audit
ran; the audit commit stages only the two paths requested here.

No production Rust, C# source, or scientific specification was changed. Tests use
public builder/engine APIs. `dotnet`, `mono`, `csc`, and `mcs` are absent from PATH;
therefore **all C# outputs below are source-inferred**, conditional on stated
collection enumeration, and none are runtime-verified cross-language results.
Rust outputs are verified by `tests/parity_hkt_reconstruction.rs`. Tests pin the
current Rust contract, including discrepancies; passing does not establish parity.

Scientific reference: Al Sulaimani and Starkey, [2023 SECA paper](https://doi.org/10.1109/ACCESS.2023.3331219),
sections V and VII-A, Figure 3, equations 2–3. The accessible
[author manuscript](https://www.researchgate.net/publication/375503068_Real-Time_Event_Detection_Using_Self-Evolving_Contextual_Analysis_SECA_Approach)
was inspected; the university publication page could not be fetched (401).
Detailed error arithmetic is outside this audit.

## Component mapping

C# paths below are relative to `references/seca-csharp/SECA/SECA/`;
Rust paths are relative to `crates/realtime-seca-core/src/`.

| C# component | Rust component |
| --- | --- |
| `SourceCleanerEngine/BLClass/Tokenizer.cs:createSourceWordListInMemory` | `engine/baseline.rs:convert_batch_to_source_word_records` |
| `SourceCleanerEngine/DataClass/SourceWord.cs:updateWordNoOfSources` | `tree/hkt_builder.rs:update_document_frequencies` |
| `SECA/SECA_DataClass/HKT.cs:addNode`, `findMyExpectedWords` | `HktBuilder::create_hkt`, `find_expected_words_general` |
| `HKT.cs:addOtherNodes`, `getSources`, `removeUsedWord` | collision selection, node creation, word-row removal in builder |
| `HKT.cs:addRefugeNodes`, `createBranches`, `createBranch` | refuge construction and `create_branches` |
| `HKT.cs:setSourceWordListForCurrentNode`, `setSourceWordListForRefugeNode` | child-scope filtering and frequency recomputation |
| `SECA/SECA_DataClass/Node.cs`, `HKT.cs` | `tree/models.rs:Node`, `Hkt` |
| `SECAAlgorithm.cs:updateTree`, child dataset helpers | `engine/trigger.rs`, `engine/scope_mapping.rs` |
| `SECAAlgorithm.cs:reConstructHKT`, `AddSourcesWordsToMyParentNode` | `engine/rebuild.rs` |
| `SECAAlgorithm.cs:setDeadHKTs`, `removeDeadHKTs` | subtree collection/removal and replacement |

## Baseline construction comparison

Both operate on source/word relations, seed with the first row's word, admit
words at strength >= alpha, use intersection / existing node source count >= beta,
merge source unions, and create refuge from sources absent from all ordinary nodes.
Removing an adopted word removes its rows, not all rows of its sources. A source
can remain in several nodes; each baseline word is processed once at a level.

Rust assigns vocabulary IDs by first token encounter, trims tokens, validates and
deduplicates source identities, and counts distinct sources per word. C# tokenizer
assigns vocabulary from a `HashSet<string>` and uses NLP output. Its frequency
helper counts rows, although node membership uses distinct sources. Duplicate
source/word rows therefore affect C# admission but not Rust document frequency.
End-to-end tokenizer parity was not tested; fixtures use explicit relations.

Rust always sorts by descending cached frequency then `source_word_id`, reindexes
rows, and iterates expected words in ascending word ID. C# `build(Dictionary)`
assumes already sorted input; tokenizer and reconstruction explicitly sort.
`findMyExpectedWords` scans all rows; Rust stops at the first below-threshold row
because sorting is guaranteed. C# iterates a HashSet of expected words. Child
records come from a ConcurrentDictionary, so equal-frequency child ordering is
not a portable first-encounter contract. Rust child tie ordering retains original
record IDs. Fixed input ordering is deterministic in Rust, but permutations of
sources/tokens or inherited vocabulary IDs can change greedy topology.

C# `addOtherNodes` (lines 863–935) creates a noncolliding node using the **first
remaining row's word**, but removes rows of the **inspected expected word**.
Rust creates/removes the inspected word. These are equivalent only when expected
iteration agrees with remaining-row order. C# chooses the last enumerated node
at the maximum collision score; Rust chooses the first created node. Node union
changes later denominators, making both word and collision order consequential.

Rust branches only when source count > configured minimum. C# `createBranch`
(line 691) branches when count >= minimum. For ordinary nodes both select
parent-member rows excluding parent words; refuge includes all member rows. Both
recompute scoped frequencies and attach children to the originating node.
However, C# passes the **mutated** local dataset into further recursion after
`addNode`; adopted words from the entire child HKT have been removed. Rust stores
and recursively uses the **pre-construction** scope, excluding only the particular
parent node's words. Thus a word assigned to a sibling node can reappear in a
Rust descendant when source memberships overlap; C# removes it from the recursive
dataset. This is a confirmed source-level scope difference, not runtime-verified
C# topology. C# also mutates shared SourceWord frequency objects in parallel
branches; overlapping branch scopes introduce a potential race. Rust clones rows.

C# construction expected words are a temporary local set; dynamic expected words
start empty. Rust stores the strength-admitted set minus the prominent word,
including words already assigned to nodes. These fields cannot be compared as
identical scientific sets or raw snapshots. Rust `top_words` are presentation
metadata, excluded from normalized structural equality.

## Reconstruction comparison

The paper's VII-A/Figure 3 specifies top-down inspection: use the full incoming
batch at Seed-HKT, parent-adopted sources at children, and reconstruct a triggered
container and descendants instead of inspecting those descendants. Both traversal
paths follow this pattern. Rust explicitly skips empty incoming scopes; C# may
call child traversal with an empty dataset. Error outcomes for empty datasets are
not inferred here.

C# `reConstructHKT` (lines 554–628) unions historical rows whose sources belong
to the selected HKT with the exact child incoming dataset, excludes ancestor
words, recomputes frequencies, sorts, and rebuilds with the old parent attachment.
Rust performs the corresponding scoped operation using retained token payloads
and trigger provenance; it deduplicates source/word pairs. Retained history means
current memory contents, not lifetime history. Full memory is used by these tests
to isolate reconstruction from forgetting.

C# ancestor filtering uses `ancestoWords`; Rust excludes the union of all,
accepted, and rejected ancestor maps. Their equivalence depends on the ancestor
context builder preserving those inclusions. Baseline branch filtering alone is
not equivalent to this explicit reconstruction filter.

C# propagates filtered incoming source IDs upward; Rust propagates all source IDs
represented in the filtered replacement dataset. In valid preexisting ancestry,
historical IDs should already belong to ancestors, so this larger union is
normally redundant. Neither path adds a source with no remaining nonancestor row
through this propagation step. The child fixture checks the parent union and
absence of sibling historical sources in the replacement.

Rust removes precisely the selected subtree from both indexes and allocates fresh
monotonic IDs. C# removes descendants via `setDeadHKTs`/caller cleanup, uses random
IDs, and may retain archive objects/logical removal metadata. Compare only live
reachable hierarchy; archives and ID allocation are incidental. Accepted sibling
scopes may receive their own memberships during an update; preservation means
absence of replacement or unintended mutation, not necessarily universal byte
identity. Our child fixture has no sibling incoming scope and verifies its exact
normalized contents and surviving IDs.

Rust skips a selected rebuild when ancestor exclusion yields no records, keeping
the original subtree. C# continues removal then calls `addNode` on empty input;
`findMyExpectedWords.First()` throws. The outer update catch swallows exceptions,
so a partially removed tree is possible. This is a source-inferred failure-path
difference. An empty batch is **not** an empty reconstruction dataset: it skips
inspection. There is no public API to inject a selected empty trigger plan;
that particular failure path is not runtime-covered by these integration tests.

## Deterministic datasets and normalized comparisons

Notation: `w:{sources}` is a node, `<>` is refuge, `/` attaches a child HKT.
Integer words in builder tests are explicit vocabulary identities; source labels
are exact integers or external IDs. No HKT/node IDs enter normalized comparisons.
C# predictions use frequency order for expected words and creation order for
nodes unless otherwise stated. HashSet/ConcurrentDictionary enumeration prevents
claiming these predictions are universal C# outputs.

| Case and parameters | Dataset | Rust verified output | C# source-inferred comparison |
| --- | --- | --- | --- |
| Single root; alpha .7, beta .5, min 1 | 1:[1,2], 2:[1,2] | `{1,2}:{1,2}`, no child; expected={2} | Same node, local expected set discarded |
| Competing words; .7/.75/min99 | 1:[1], 2:[2], 3:[1,2] | `1:{1,3}`, `2:{2,3}` | Same ordinary memberships |
| Prominent tie; .7/.5/min99 | 1:[1,2] versus 1:[2,1] | Same merged node; expected={2} versus {1} | First row seeds prominent; no persistent baseline expected set |
| Collision tie; 1/.5/min99 | 1:[1,3], 2:[1], 3:[2,3], 4:[2] | `{1,3}:{1,2,3}`, `2:{3,4}` | With node order 1 then 2: `1:{1,2}`, `{2,3}:{1,3,4}`; tie attaches to last |
| Word-order mismatch; .5/1/min99 | 1–4:[3], 5–6:[2,1], 7:[2] | `3:{1,2,3,4}`, `{1,2}:{5,6,7}` | Frequency iteration 3,2,1: separate nodes `3`, `2`, `1`; source overlap remains |
| Refuge; .8/.5/min1 | 1–3:[1], 4–5:[2] | `1:{1,2,3}`, `<>:{4,5}/2:{4,5}` | Same hierarchy; descendant recursion empties after adoption |
| Branch boundary; .7/.5 | 1:[1,2], 2:[1], min2 versus min1 | min2: root only; min1: `1:{1,2}/2:{1}` | min2 already creates child; at min1 same child |
| Two siblings; .7/.5/min1 | s0–9: a; s10–19: d; first four per group add b/e, next four add c/f | Root a/d; children b/c and e/f | Same, barring child tie order; normalized labels agree |
| One child reconstructed | Previous baseline; n0–9:[a], n0–2 also x | Only a-child replaced, containing b/c/x evidence; no a in child; d-child preserved | Dataset scope and attachment agree; exact trigger outcome not executed |
| Root reconstructed/new dominant | o1:[a,b], o2:[a]; x0–3:[x] | Root x plus refuge old evidence; all old HKT IDs removed; only root inspected | Sorted replacement selects x; below-refuge depth can differ due cutoff |
| Incoming multiple matches | a1:[a], d1:[d]; both:[a,d] | both belongs to both root nodes; no reconstruction | Mapping permits same memberships |
| Reconstruction then no change | Root replacement above; x4:[x] | No further reconstruction; HKT IDs survive | Expected conditional on equivalent trigger configuration; not executed |
| Empty incoming scope | Multiple-match case followed by empty batch | Normalized hierarchy and memberships unchanged in Full mode | Empty traversal behavior not asserted |
| Overlapping multilevel scope; .8/.75/min0 | 1–6 carry 0; 1–3 also 1; 3–5 also 2 | Root 0 / child nodes 1 and 2 / each has the other word below it, source={3} | Child construction removes both words before recursion: no grandchildren |
| Empty builder input | No relations | No HKT returned | `First()` throws; not a valid baseline reference tree |

The collision-tie test also reorders the same source relations to seed word 2
instead of word 1: Rust then merges word 3 with word 2. This is a verified
topology change caused solely by equal-frequency input ordering.

`normalized` in the integration tests recursively records sorted token word sets,
external source sets, child attachment, and expected tokens. It sorts sibling
representations and omits allocators, state flags, top words, and internal hashes.
The report supplies conditional C# structures; tests do not pretend to execute a
second implementation. Fresh versus preserved IDs are checked separately where
replacement identity matters. Builder tests also check the mirrored node cache.

## Discrepancies, causes, and minimal recommendations

1. **Branch boundary:** literal `>` versus `>=` mismatch. Decide the scientific
   minimum contract, then align the comparison and config documentation with a
   boundary regression. The 2023 SECA description does not settle this cutoff.
2. **Expected-word ordering and collision ties:** ordered Rust sets versus C#
   hash collection enumeration/last maximum. A deterministic tie policy needs
   explicit scientific approval; do not copy unspecified hash iteration. Test
   permutations and greedy membership effects before choosing an ordering.
3. **Noncolliding C# node selection:** first-row creation paired with removal of
   another word is fragile. Preserve Rust's coherent inspected-word behavior;
   document divergence rather than introduce the C# defect for parity.
4. **Recursive dataset:** C# passes post-adoption rows, Rust pre-adoption scope.
   Decide whether sibling descriptors may recur below another parent with shared
   sources using the original CA references. Add an overlapping multilevel
   cross-language fixture before changing Rust scope semantics.
5. **Frequency inputs:** row count versus distinct source count. Paper sources
   are sets; distinct-source Rust counts are supported. Normalize duplicate pairs
   in any future C# harness rather than change Rust to count repetitions.
6. **Expected-word persistence:** distinguish construction candidates from
   dynamic unassigned expected words when specifying cross-language comparison.
   Do not rename/remove Rust state without resolving its update contract.
7. **Empty replacement:** preserve transactional state and define the intended
   no-evidence outcome explicitly. Rust's skip is safer than C# partial removal;
   add an internal trigger-plan test separately if this path becomes reachable.
8. **Parallel mutable C# rows:** use copied scope records in a future harness and
   record runtime/version/enumeration. A serial reimplementation would test an
   interpretation, not the unmodified original's concurrency behavior.

The paper supports strength/eligibility admission and targeted top-down evolution,
but delegates baseline construction to earlier CA work. It does not resolve
collection ordering, collision ties, reuse of sibling words below a node,
minimum branch size, persistent expected-word representation, or empty replacement.
These remain scientific ambiguities; no production correction is implemented.

## Execution results

`cargo fmt --all -- --check`: passed.

`cargo test -p realtime-seca-core --test parity_hkt_reconstruction`: **11 passed**.

`cargo clippy -p realtime-seca-core --all-targets -- -D warnings`: passed.

`cargo test -p realtime-seca-core --lib`: **74 passed**.

`cargo test -p realtime-seca-core --test seca_light_stream`: **12 passed**.

`cargo test -p realtime-seca-core`: interrupted after several minutes (exit 130).
All 74 unit tests and three large-dataset baseline tests passed; two chunked
large-dataset tests were still running. No failures observed; this is not a
completed full-suite pass. A separate process-batch integration run was also
interrupted (exit 130) with 12 tests passed and its large-dataset sliding-window
test still running. Workspace-wide tests were not run; no architectural changes
were made.

 Initial focused execution: nine passed and one
fixture expectation failed. The word-order fixture correctly produced `{1,2}`
because adding word 2 covers every source of the already created word-1 node.
The expectation was corrected to capture that order-dependent merge.
