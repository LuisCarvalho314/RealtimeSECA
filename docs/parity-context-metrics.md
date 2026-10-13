# Executable context and metrics audit (2026-10-13)

This section supersedes the execution-availability and current-engine conclusions
in the historical reports below. Those reports are retained for provenance.
The harness is accessible, builds with Mono, and passes `verify.py`. This audit
executes the current working engine with `MemoryMode::Full`, not SlidingWindow.
No production Rust, scientific contract, original C#, or harness files were edited.

The starting checkout was `integration/seca-validation`, HEAD
`9e70e2136dea237f1e922352df0ae24d4441c2fc`, with pre-existing uncommitted integration
changes. It was **not clean main**. Results describe that working tree, on the
new branch `agent1/context-metrics-executable`; they must not be attributed to
clean main. `audit-hashes.json` identifies the relevant Rust and harness files;
`source-hashes.json` records the harness's compiled source inputs. HKT, Node,
SECAAlgorithm and the three error classes in the original and compilable reference
were checked byte-identical. Earlier uncommitted changes in the assigned report
and tests were preserved. Other owners' files remain outside this commit.

## C#–Rust mapping

C# paths here are relative to `references/seca-parity-build/SECA/SECA_DataClass/`.
Their original equivalents are under
`references/seca-csharp/SECA/SECA/SECA/SECA_DataClass/`.
Rust paths are relative to `crates/realtime-seca-core/src/engine/`.

| C# location | Rust location | Correspondence |
|---|---|---|
| HKT.cs:119 `computeBatchErrorsForCurrentHKTLevel` | trigger.rs:55 traversal; context.rs:111 update stage | Context before decision |
| HKT.cs:615 mapping; Node.cs:158 pending word mapping | scope_mapping.rs:65; context.rs:231 | Scope DF, source-to-node mapping |
| HKT.cs:514 `setExpectedWordsInHKT`, :360 reset | context.rs:126 admission, :277 recheck | Candidate alpha admission |
| Node.cs:85 `assignExpectedWordsToNodes`, :127 prominent selection | context.rs:327 assignment | Candidate beta; denominators differ |
| HKT.cs:393 denominator, :419 prominent count | metrics.rs:35 | Descriptor counts plus admitted candidates |
| Node.cs:180 `setChangeStatisitcsForWordsInNode` | metrics.rs:65 | Updated strength, eligibility, importance |
| HKT.cs:197 error computation, :242 current words | metrics.rs:131, :149 | Descriptor-only W0 |
| AlphaError.cs:45/:55; BetaError.cs:44/:54; WordImportanceError.cs:38/:47 | metrics.rs:149 | Equations and rounding |
| HKT.cs:94 `isAccepted` | trigger.rs:175/:206 | Strict threshold decisions |
| SECAAlgorithm.cs:390/:433 accepted recursion/merge | trigger.rs:119; context.rs:6/:72 | Child processing and accepted membership commit |
| HKT.cs:593 parent history lookup | evidence.rs parent DF | Historical candidate counts |

## Test methodology

Fixtures live in `crates/realtime-seca-core/tests/fixtures/parity_context_metrics/`.
Each `*.input.json` contains ordered batches of positive unique integer source IDs
and unique, preprocessed words per document. Rust uses the same IDs as strings,
same word arrays, batch sequence, alpha, beta, branch minimum and three thresholds.
The first batch constructs the baseline. Most fixtures use branch minimum 1000
and thresholds 1 to retain the measured scope. The child case uses minimum 1;
both builders produce the required two HKTs for this dataset, avoiding their
known exclusive/inclusive cutoff discrepancy. No random data are used.

```sh
tools/seca-reference-harness/build.sh
python3 tools/seca-reference-harness/verify.py
python3 crates/realtime-seca-core/tests/fixtures/parity_context_metrics/replay.py
cargo test --offline -p realtime-seca-core --test parity_context_metrics -- --nocapture
cargo test --offline -p realtime-seca-core
# Explicitly reproduce the labelled strict-parity failure:
cargo test --offline -p realtime-seca-core --test parity_context_metrics expected_failure_csharp_alpha_threshold_parity -- --ignored
```

`replay.py --capture` was used to capture actual reference stdout, compacted only
by JSON parsing/serialization. It preserves every raw tree, metric and error
field. Subsequent default replay verifies goldens and two byte-identical executions
per successful fixture. Failures retain stderr and **have no invented JSON golden**.
Captured stderr strips trailing line whitespace only. Logs include nondeterministic stack addresses, so failure verification checks exit
status, empty stdout and the error class rather than exact stderr bytes.

The harness's documented instrumentation serializes iteration, assigns deterministic
IDs, disables unused persistence, surfaces swallowed exceptions and observes
existing metrics immediately before `isAccepted`. It does not recalculate metrics.
These are adapted original algorithm executions, not the nondeterministic desktop
application. The original compiled preprocessing dependency remains in use.

Rust assertions consume actual `decision_diagnostics`, not final-topology display
observations. The tests independently aggregate the captured word strengths, eligibilities and
importances over captured W0 using the input alpha/beta, and verify C# threshold
comparisons against captured errors. Numerical comparisons permit only C#'s four-decimal rounding envelope
(0.00005); explicit discrepancy branches assert the observed divergent values and
decisions. Selected unchanged, child, refuge, exact-alpha and node-union fixtures
also compare retained sources against C# by external identity and descriptor counts.
C# word-level State 0/1 values and pending assignments are captured; Rust's public
report does not expose its temporary per-word metrics or assignment map. Consequently
word-level correspondence uses C# execution, independent arithmetic and Rust source
inspection, and is **not** claimed as executed intermediate Rust parity. Additional
non-mutating instrumentation would be needed to close that observational gap;
no production API was added.

Scientific reference: [2023 SECA paper DOI](https://doi.org/10.1109/ACCESS.2023.3331219)
and the accessible [author manuscript](https://www.researchgate.net/publication/375503068_Real-Time_Event_Detection_Using_Self-Evolving_Contextual_Analysis_SECA_Approach),
sections V and VII, equations 2–12. The final DOI page was inaccessible in this run.
Independent calculations below use those equations as transcribed in
[algorithm.md](algorithm.md). The manuscript distinguishes accumulated SECA from
SECA-Light and defines strict error exceedance. It does not specify four-decimal
rounding before decisions. Expected-word scope and shared-word collapse remain
scientific ambiguities rather than being settled by implementation agreement.

## Results

Notation: baseline `{a,b}×2` means two distinct documents; every input is available
as the named fixture. Errors are `(Alpha, Beta, WI)`. Unrounded Rust fractions are
shown exactly; C# values are actual rounded results. Unless noted all decisions
accept. “Metric parity” below allows documented rounding, not bitwise equality.

| Fixture | C# result | Rust result | Independent paper calculation | Parity status / trigger consequence |
|---|---|---|---|---|
| unchanged | (0,0,0), twice; sources 2→3→4 | same | DF equal; strength/eligibility=1; importance=.5 each | Verified metrics and retained membership; no trigger |
| new_word | (0,0,.3333) | (0,0,1/3) | a4,x2; denominator6; W0={a} | Metric parity; no trigger |
| below_alpha | (0,0,0); no x | same | x1/a5=.2<.7 | Verified rejection and decision |
| just_below_alpha | (0,0,0); no x | same | x2/a4=.5<.5001 | Verified near-boundary rejection |
| exact_alpha | (0,0,.3333); x pending at node | (0,0,1/3); old descriptor retained | strength=beta=.5; W0={a} | Admission correspondence; no trigger |
| below_beta | (0,0,.4); x unassigned | same | strength2/3; incoming beta1/3; denominator5 | Metric parity; no trigger |
| weakening | (.0833,.0833,0) | (1/12,1/12,0) | a6,b2; mean deficit (.5−1/3)/2 | Rounding difference; decision agrees at thresholds1 |
| expected_consecutive | first WI=.4, next .3333 | first .4, next .5 | next a4,x4; denominator8; descriptor-only W0 | Confirmed metric divergence; thresholds1 hide decision impact |
| expected_trigger | first .4 accepted, next .3333 accepted | first .4 accepted, next .5 reconstructs | same next-cycle calculation; .5>.4 | Confirmed decision divergence caused by stale C# expectation |
| shared_candidate | duplicate-key failure; no decision JSON | (0,0,1/3), accepts | a4,b4,x4; denominator12 | C# algorithm failure; final decision parity unavailable |
| child | root and child (0,0,0); memberships5/3 | same; incoming scopes2/1 | child sees only parent-matched `{a,b}` | Verified scope/membership outcome; no trigger |
| alpha_threshold | .0833 equals threshold; accepts | 1/12>.0833; alpha trigger | full-precision deficit exceeds threshold | Deliberate precision difference changes decision |
| beta_threshold | .0833 equals threshold; accepts | 1/12>.0833; beta trigger | same for eligibility | Deliberate precision difference changes decision |
| wi_threshold | .4 equals threshold; accepts | same | x2/(a3+x2)=.4 | Verified strict equality acceptance |
| low_evidence | (0,0,0) | same | one old plus one incoming a; all ratios1 | Verified low-evidence result |
| empty | harness rejects empty batch | skips inspection; accepts | no new evidence; unchanged old context | Input-contract difference; C# algorithm empty-scope behavior unavailable |
| historical_candidate | (0,0,.4); no node assignment | same | a6,x4; full-history beta4/6 passes, incoming2/6 fails | Metric parity; historical-beta scientific ambiguity persists |
| node_union | (0,0,.2); x assigned by 2/5=.4 | (0,0,.2); assignment rejected by 2/6 | a3,b5,x2; denominator10; full-node beta1/3 | Candidate assignment differs; immediate decisions agree |
| refuge | (0,0,0); refuge remains source4 only | same errors; refuge gains source5 | unmatched evidence belongs in refuge context | Membership difference; immediate decisions agree |

## Calculations, timing and discrepancies

**Verified behavioural parity within observable limits.** Both measure old
node descriptors for W0, excluding newly assigned candidates and unassigned
expectations. Updated descriptor strength divides contextual node DF by the
largest descriptor/candidate count at that HKT level. State 1 eligibility divides
word/node intersection by the updated full node union. Importance divides the
word count by the sum of descriptor counts plus alpha-admitted candidate totals,
even when beta rejects a candidate. The weakening golden records b strength and
eligibility 1/3, importance2/8, a importance6/8; independent equations give
A=B=1/12 and WI=0. The below-beta golden records a3/5 and unassigned x2/5.

Source IDs are unique in the harness. With one relation per source-word, relation
counts implement DF; Rust explicitly deduplicates repeated tokens. Repeated
words within a document are rejected by the harness, so the pre-existing Rust
repetition test establishes a Rust input contract, **not cross-language parity**.
Raw C# State 0 eligibility uses old prominent-word sources (Node.cs:223), while
State 1 uses full node union (:233–237). In node_union, b's State 0 eligibility is
2/3, while State 1 is5/6; State 0 equations in the manuscript use node sources.
This is a **C# deviation from the paper**; Option1 uses only State 1, so it does
not explain the immediate trigger result. Rust has no observable State 0 metric
report with which to claim State 0 numerical equivalence.

**Deliberate implementation difference: precision.** C# rounds all three Option1
errors with `Math.Round(...,4)` before HKT.cs:94 evaluates strict `>`; Rust
metrics.rs:149 and trigger.rs:175 preserve full precision. The alpha_threshold and
beta_threshold inputs each isolate one changed trigger reason. The explicit ignored
regression asserts C# acceptance and fails against Rust, demonstrating the mismatch.
Rust's reason string rounds display values, producing `0.0833 > threshold 0.0833`;
actual numbers differ. Diagnostic formatting is misleading but not its decision.

**C# deviation from accumulated evidence: stale expectations.** HKT.cs:545 only
inserts an absent expected key. The second expected_consecutive update retains
x's old object `(state0=0,state1=2)` despite four accumulated occurrences.
C# denominator becomes a4+x2=6, WI=1−4/6=.3333. Rust recomputes x4 from retained
parent history plus incoming scope, denominator8, WI=.5. `expected_trigger` proves
this changes reconstruction at identical WI threshold .4. Under the documented
descriptor-only W0 interpretation Rust follows accumulated evidence; changing
Rust to stale counts would contradict that contract.

**Deliberate implementation difference: candidate beta.** Node.cs:85/:109 uses
incoming candidate overlap with updated prominent-word sources. Rust
context.rs:327 uses incoming overlap with the full node source union. node_union
records C# x in node expectedWords and pendingWordSources; old metrics still
sum only a,b. Neither implementation permanently adds x to descriptor words in
this accepted path. Assignment differs even though old-word errors agree. The
historical_candidate fixture also proves C# rejects beta despite four accumulated
x occurrences; Rust's pre-existing regression confirms its incoming-only rule.
Whether earlier candidate occurrences must contribute remains **unresolved
scientific ambiguity**; it must not be inferred from agreeing aggregate errors.

**C# deviation / deliberate Rust membership difference: refuge.** C# HKT mapping
tracks nonmatching sources separately; SECAAlgorithm.cs:433 merges only old
word-index keys, leaving refuge source membership unchanged in the captured
fixture. Rust context.rs:6/:72 commits its mapped source union including refuge.
This improves retained-context completeness. It changes no immediate error because
refuge words are omitted from the three old-word sums. Future child/history effects
are not claimed from this one fixture.

**C# failure for multi-node candidates; unresolved scientific ambiguity.**
shared_candidate assigns x to both a and b nodes, then HKT.cs:287 performs
`Dictionary.Add` for duplicate x metrics. The surfaced exception originates in
that original merge, not in a fabricated diagnostic. The uninstrumented application
swallows this exception; its partially stale continuation is not reconstructed here.
Rust metrics.rs:101 inserts by word ID and deterministically overwrites by node
order. This fixture has candidate x once in the importance denominator; it does
not establish correctness for an existing descriptor repeated in multiple nodes.
The descriptor denominator loop (C# HKT.cs:393; Rust metrics.rs:35) counts each
node occurrence, while W0 is a union. The paper's set notation and scope of D(w)
remain unresolved for this case. No stable shared-descriptor golden is invented.

**Timing and propagation.** Rust input.rs registers source IDs and append-only
vocabulary before mapping and trigger evaluation, within the candidate transaction.
Temporary stage memberships enter metrics; original W0 remains immutable. Accepted
parents expose pending memberships before child traversal and merge afterward.
C# maps temporary membership, admits/rechecks candidates, assigns them and computes
metrics before acceptance (HKT.cs:119–170). Accepted children run before the merge
in SECAAlgorithm.cs:433; C# pending maps remain visible after success and are reset
on the next inspection. Global historical rows are appended by the harness after
updateTree, clearMemory and decisions. Vocabulary is allocated from ordered inputs
before each cycle. Neither language's new-token ID allocation by itself expands W0.
The child fixture verifies source5 reaches both parent and child, source6 reaches
neither child nor matched root node, and only one incoming source reaches the
child. It does not settle every ancestor candidate propagation edge case.

**Empty and zero denominators.** Low evidence is executed successfully. Empty
batches are rejected at the harness boundary; Rust skips empty scopes. C# floor-one
word denominators and empty-sequence alpha/beta divisions are source observations,
not executable zero-evidence parity. No public baseline can construct an empty
vocabulary here. This remains a coverage limitation, along with half-way rounding,
all-zero retained scopes and stable multi-node descriptors.

## Recommendations (not implemented)

No confirmed current Rust scientific defect requiring a production fix was found
in these executed fixtures. Keep full precision, accumulated expected counts and
complete refuge membership under the current scientific contract. Exact legacy
C# compatibility would require rounding before triggers, stale expected evidence,
and incomplete refuge retention; these are not justified scientific corrections.

The smallest justified follow-up is diagnostic formatting that distinguishes
full-precision values at threshold boundaries. Resolve historical candidate beta,
shared-word collapse, W0 expectations and ancestor candidate propagation against
the scientific specification before changing calculations. If compatibility rather
than scientific correctness is desired, make that objective explicit; do not add
a second execution path to emulate reference defects. Additional read-only Rust
word-level instrumentation and a harness input extension for empty scopes would
require separately authorized work outside this file ownership.

## Validation

- C# build succeeded, with 22 inherited warnings; `verify.py` passed.
- 19 fixture inputs replayed twice: 17 deterministic successful JSON outputs and
  two reproducible failures (shared candidate and empty input).
- Focused Rust suite: **20 passed, 1 ignored**, including the executable golden
  comparison covering all inputs. The ignored C# acceptance test was explicitly
  run and **failed as expected** (Rust true versus C# false).
- Core crate suite: **156 passed, 2 ignored, 0 failed** (74 unit tests and
  82 integration tests; both ignored tests document known cross-language
  differences). The final focused suite was rerun after test-helper cleanup.
- Assigned test file passed rustfmt check; `git diff --check` passed. The file
  exceeds 400 lines because existing integration regressions are preserved; the
  executable comparison uses focused configuration, membership and metric helpers.
- No architectural change; workspace architectural validation is not required.


---

The revision-specific report from main follows unchanged. Its earlier execution-availability statements are superseded by the executable evidence above and the authoritative summary.

# SECA contextual updating and metrics audit

Historical audit of **42c07c0**, not current main. The test source is retained at
commit `c6cbc61` and is not added by this documentation-only consolidation.
See [the consolidated audit](audit-2026-10.md) for revision reconciliation.


## Revision and evidence

Audited Rust: `main`, **42c07c098c1fadb931a4e7b5b6920ee9561dabc5**.
The user explicitly selected this revision after the initial checkout was found
on `fix/seca-light-stream` at `15814d1`. The supplied `AGENTS.md`,
`docs/algorithm.md`, `docs/architecture.md`, and `docs/state.md` were read there;
none of these files exists on the audited main revision. They are context for the
investigation, not evidence that main already implements their contracts.
In particular, main has no `engine/context.rs`, `evidence.rs`, or `metrics.rs`;
the corresponding functions are in `engine/trigger.rs`.
No claim below describes the feature branch's execution.

C# reference: `references/seca-csharp/SECA/SECA/`, read without modification.
C# results throughout this report are **static, unverified behavior**, never
executed parity. `command -v dotnet mono mcs csc msbuild` found none. The SECA
project is a .NET Framework 4.7.2 Windows Forms executable with a
SourceCleanerEngine project dependency and Accord libraries. Bundled binaries do
not establish a working runtime. No C# adapter was needed or created.

Scientific source: Al Sulaimani and Starkey, *Real-Time Event Detection Using
Self-Evolving Contextual Analysis (SECA) Approach*, IEEE Access 11 (2023),
127011–127034, [DOI](https://doi.org/10.1109/ACCESS.2023.3331219).
The [author manuscript](https://www.researchgate.net/publication/375503068_Real-Time_Event_Detection_Using_Self-Evolving_Contextual_Analysis_SECA_Approach)
was read at Sections VII-A/B, equations 2–12. Its displayed text identifies it as
an accepted manuscript; the final published PDF was not obtained. The
[university publication record](https://abdn.elsevierpure.com/en/publications/real-time-event-detection-using-self-evolving-contextual-analysis/)
identifies the final version. This distinction limits claims about editorial
changes to the final paper, not the source-code comparisons.

Fixtures: `crates/realtime-seca-core/tests/parity_context_metrics.rs`.
They use `SecaEngine::new`, `build_baseline_tree`, `process_batch`, tree exports,
and batch-statistics accessors. Main has no public structured decision diagnostics:
metrics are extracted from **paper diagnostic** notes, which expose four decimal
places. Placeholder notes are never used as the metric oracle. Tests explicitly
characterize current defects; passing tests do not certify scientific correctness.
They introduce no production API, production modification, or new policy.

## C# to Rust mapping

C# paths in this table are relative to `SECA/SECA_DataClass/` unless stated
otherwise. Rust paths are relative to `crates/realtime-seca-core/src/engine/`.

| Functionality | C# | Rust main |
|---|---|---|
| Top-down orchestration | `SECAAlgorithm.updateTree`, `HKT.computeBatchErrorsForCurrentHKTLevel` | `trigger.rs::evaluate_hkt_recursively_for_batch` |
| State 0 / pending State 1 | `Node.sourceIds`, `wordSourceIds`; `sourceIdsNewFromBatches`, `wordSourceIdsNewFromBatches`, `wordIdsNewFromBatches` | `tree::Node` analogous sets; `HktScopeSnapshot`, `HktUpdateStage` |
| Incoming ordinary-node mapping | `HKT.matchNewSourcesToHKT`, `Node.matchNewSourcesToNode` | `scope_mapping.rs::map_batch_into_hkt_scope` |
| Unmatched sources / refuge | `HKT.setNonMatchingSources`; `SECAAlgorithm.getSourceWordDatasetForNodeChildHKT_ForRefuge` | refuge mapping in `scope_mapping.rs`; child selection in recursive traversal |
| Word frequency | `SourceCleanerEngine/DataClass/SourceWord.updateWordNoOfSources`; `HKT.setExpectedWordsInHKT` row grouping | `map_batch_into_hkt_scope`; `build_baseline_source_word_sets` |
| Historical candidate scope | `HKT.getNoOfSourcesOfWordInParentNodeScopeInState0` | `trigger.rs::build_state0_df_in_parent_scope` |
| Initial alpha admission | `HKT.setExpectedWordsInHKT` | `trigger.rs::compute_update_stage_for_scope` |
| Updated prominent count / strength recheck | `HKT.setNumberOfSourcesOfProminentWordInHKT_Expected`, `reSetExpectedWordsInHKT` | same update-stage function |
| Candidate beta assignment | `HKT.assignExpectedWordsInHKTToNodes`, `Node.assignExpectedWordsToNodes`, `getSourcesOfNewProminWordInNode` | same update-stage function |
| Importance denominator | `HKT.setNumberOfSourcesForAllCurrentAndExpectedWordsInHKT_Expected` | `compute_word_change_metrics_for_scope` |
| Word measurements | `Node.setChangeStatisitcsForWordsInNode`, `HKT.setExpectedWordsChangeStatisitcs` | `compute_word_change_metrics_for_scope` |
| Old vocabulary / aggregation | `HKT.getCurrentWordsIdsInHKT`, `computeHKT_Errors` | `build_current_word_ids`, `compute_paper_scope_metrics_from_change_metrics` |
| Threshold decisions | `HKT.isAccepted` | `select_paper_errors_and_thresholds`, `compute_paper_shadow_trigger_reasons` |
| Parent context | `SECAAlgorithm.updateAncestorAcceptedAndRejectedWords`, `setAccecptedAndRejectedWordsInHKTFromSpecificNode` | `build_child_ancestor_context_from_parent_node` |
| Pending membership commit | accepted branch of `SECAAlgorithm.updateTree` | `apply_state1_updates_for_hkt`, `merge_state1_updates_for_hkt` |

## Equations and exact counting semantics

The independent oracle uses distinct sources and explicit vocabulary sets:

```text
D1(w) = number of distinct retained-or-incoming scoped sources containing w
q     = updated most frequent relevant word
M1(z) = old node members union incoming members
W0    = old node descriptors (alternative: include old unassigned expectations)
W1    = W0 union alpha-admitted expected words
s1(w) = D1(w) / D1(q)
e1(w) = |Sources1(w) intersection M1(z)| / |M1(z)|
i1(w) = D1(w) / sum(v in W1) D1(v)
A     = sum(w in W0) max(0, alpha - s1(w)) / |W0|
B     = sum(w in W0) max(0, beta  - e1(w)) / |W0|
I     = 1 - sum(w in W0) i1(w)
```

These are the paper's equations 5, 8, 11, 6, 9, 12 written as counting operations.
The alternative W0 definition is evaluated explicitly below, not silently chosen.

**Actual main and C# behavior:**

* Mapping allows a source in every ordinary node whose descriptors it contains.
  Only sources matching no ordinary node go to refuge. Seed-HKT gets all incoming
  sources; a child gets only its parent's matched incoming sources. C# uses
  parallel child traversal; Rust visits children by HKT ID.
* Rust normalizes with trimming, ignores empty tokens, and uses per-source sets.
  C# frequency functions count **SourceWord rows**, not distinct source IDs.
  Equality with DF requires one row per `(source, word)`. This is not guaranteed
  by the inspected in-memory tokenizer: `NLP.tokenize` inserts freshly allocated
  `Word` objects into a HashSet and `Word` has no value equality override. Repeated
  tokens can survive as separate objects and rows. Node source sets deduplicate
  memberships even when candidate row counts do not. Ingestion parity is therefore
  conditional, not confirmed for repeated C# text tokens.
* Initial candidate alpha denominator in main is the historical count of the
  first word ID of the first node, reconstructed from retained token payloads.
  C# uses the first node's first word membership count. Rust's BTreeSet chooses
  smallest word ID; C# HashSet iteration need not do so. Neither expression proves
  that this is still the true State 0 maximum after previous updates.
* Both then compute an updated maximum over node-word counts and candidate totals
  and recheck alpha. Admission is inclusive `>=`. Current node counts are local
  memberships; candidate historical counts use all retained sources at Seed-HKT,
  or the parent node's historical membership for a child. Incoming candidate counts
  use that HKT's incoming scope, including unmatched sources.
* Candidate beta uses **incoming candidate sources intersect updated node-prominent
  sources / updated node-prominent source count** in both main and C#. Historical
  candidate occurrences do not enter this numerator. This differs from applying
  the full-history node-source equation directly. It also differs from using the
  full node union when several descriptor source sets enlarge the node.
* Old-word State 1 Beta-Error uses the **full updated node union**, in both main
  and C#. The C# comment proposing this alternative was implemented in
  `Node.setChangeStatisitcsForWordsInNode`; its former prominent-word expression
  is commented out. Candidate admission and old-word measurement use different
  beta denominators. State 0 beta diagnostics still use the old prominent word;
  Option1 does not use State 0 beta.
* Importance denominator is the sum of **every node-word membership count**,
  plus every alpha-admitted candidate's historical parent-scope plus incoming DF,
  before beta assignment. Rejected-alpha words are absent; failed-beta candidates
  remain present. This is a multiset over node descriptors, not a vocabulary union.
* C# restricts error summation to `node.wordIds`, excluding pending assignments
  (`wordIdsNewFromBatches`) and unassigned expectations. Main applies
  `HktUpdateStage::apply_to_snapshot` **before error aggregation**, adding assigned
  candidates to `node_word_ids_by_node_id`. Main therefore uses W0 plus newly
  assigned candidates. Unassigned expected words remain outside W0.
* Accepted updates merge existing descriptor memberships after child traversal.
  Both main and inspected C# preserve descriptor sets rather than permanently
  promoting pending candidate IDs to descriptor words. Main merges candidate
  source indexes; the C# accepted loop visits only existing `wordSourceIds.Keys`,
  so pending candidate indexes are not merged by that loop.
* Main explicitly skips refuge membership commit in `apply_state1_updates_for_hkt`.
  C# maps refuge via `nonMatchingSources` for child delivery, and the ordinary-node
  commit loop does not add those unmatched source IDs. Thus child delivery and
  refuge's own retained source union are separate, with no demonstrated complete
  refuge-membership update in either path.
* Main propagates words in the **mutated** parent snapshot, including candidates
  temporarily beta-assigned there, as accepted ancestor words. C# propagates all
  old node descriptors plus candidate node measurements whose strength passes
  alpha. Candidates with node-level counts below alpha can therefore be handled
  differently. Neither rejects an old descriptor merely because its strength falls.
* Main carries `expected_words` forward and recalculates totals. C# carries
  `expectedWords_Dynamic` objects forward but only inserts an absent key in
  `setExpectedWordsInHKT`: old expected counts can remain stale across batches.
  Its reset clears pending node maps, not dynamic expectations or node
  `expectedWords`. Duplicate `Dictionary.Add` and swallowed exceptions further
  limit prediction of repeated-candidate runs.

### Precision, thresholds, and empty evidence

Main's default paper policy selects Option1 thresholds (`alpha_option1_threshold`,
`beta_option1_threshold`, `word_importance_option1_threshold`), not the similarly
named generic error thresholds. Tests use the active Option1 fields; they do not
alter thresholds to make implementations agree. Other selectable options are
outside the paper-defined error equations in this audit.

Main rounds aggregate Option1 errors to four places before strict `>` decisions,
as does C#. Rust implements `(value * 10000).round()/10000` (ties away from zero);
C# `Math.Round(value, 4)` uses ties to even by default. Exact decimal half-way
parity is therefore not established. The tests confirm equality acceptance for
all three errors and rejection just below the rounded value. At A=B=1/12, main
uses 0.0833: a threshold of 0.0833 accepts, although unrounded 1/12 exceeds it.
The supplied feature-branch document's full-precision policy does not describe main.

Count denominators are floored at one in main and most C# word measurements.
Main explicitly returns A=B=0 for empty W0; C# Option1 divides by the empty
sequence length, producing NaN. Both importance aggregations give I=1 when the
summed old mass is zero. C# NaN fails `>` and can suppress reconstruction.
Main does not skip the root for an empty incoming batch; it can report or trigger
on historical errors. Child traversal requires a nonempty incoming scope.
A public API fixture for an empty-word HKT / all-zero retained evidence was not
constructed: baseline rejects an empty dataset, and no production test hook or
partial snapshot surgery was introduced. Zero-denominator claims are static.

## Deterministic comparisons

Notation: `{a,b} × k` means k **different source identities**, each containing
a and b. Baseline is batch 1; update is batch 2 except the repeated-candidate row.
Errors are `(A, B, I)`. C# entries assume distinct SourceWord relations, default
Option1 selection, and no prior exception; they are static predictions.
Unless stated, branch creation is disabled by a high minimum source count,
alpha=beta=0.5, and all active error thresholds are 1 to retain context for inspection.
The baseline builder is used only to obtain state, not audited here.

| Case / dataset | Independent paper calculation | C# static, unverified | Rust main executed | Reconstruction consequence |
|---|---|---|---|---|
| Unchanged one word: old `{a}`, incoming `{a,a," a ",""}` | Distinct D(a)=2; `(0,0,0)` | `(0,0,0)` for unique rows; tokenizer duplicate caveat | `(0,0,0)`; node gains one source | No trigger |
| Unseen below alpha: old `{a}×4`, incoming `{a,x}`, alpha=.7 | x strength 1/5<.7; excluded; `(0,0,0)` | Initial 1/4<.7; excluded | `(0,0,0)`; no candidate retained | No trigger |
| Alpha pass, beta fail: old `{a}×2`, incoming `{a,x}`, `{x}` | D(a)=3,D(x)=2; alpha 2/3; beta 1/3; I=2/5 | `(0,0,.4)` | `(0,0,.4)`; x expected, no descriptor change | WI threshold .4 accepts; .3999 triggers |
| Exact alpha/beta admission: old `{a}×2`, incoming `{a,x}×2` | x strength=2/4=.5, beta=.5; W0={a}; I=2/6=1/3 | `(0,0,.3333)` | `(0,0,0)` because W0 becomes {a,x} | At WI threshold .2, paper/C# trigger; Rust does not |
| Unchanged two descriptors: old `{a,b}×2`, incoming `{a,b}` | W1=W0={a,b}; denominator 6; `(0,0,0)` | Dynamic expectations initially empty; `(0,0,0)` | b remains in construction expected set; denominator 9; `(0,0,.3333)` | At WI .3 Rust spuriously triggers |
| Old b loses strength: old `{a,b}×2`, incoming `{a}×4` | a=6,b=2; b strength and eligibility=1/3; A=B=(.5-1/3)/2=1/12; I=0 | `(.0833,.0833,0)` | `(.0833,.0833,0)`; b drops from expectation denominator | Rounded A/B equality .0833 accepts; .08329 triggers |
| Unassigned expectation over two updates: old `{a}×2`, then `{a,x}`, `{x}` twice; beta=.9 | At third batch D(a)=4,D(x)=4; descriptor-only W0 gives `(0,0,.5)`; including x in W0 would give I=0 and needs an eligibility definition for unassigned x | Persistent dynamic x can retain earlier total 2 instead of 4: `(0,0,.3333)` under this path | Synthetic x absent from historical legend; counts only incoming 2: `(0,0,.3333)`; still reported new | At WI threshold .4, full-history descriptor oracle triggers; main would not |
| Historical candidate beta: old `{a,x}×2`, `{a}×2`, incoming `{a,x}×2`; alpha=.6 | D(a)=6,D(x)=4; full-history x beta=4/6>=.5; I=.4 | Admission beta counts only incoming 2/6; fails; `(0,0,.4)` | `(0,0,.4)`; x unassigned | Admission/context differ from full-history equation; immediate old-word error agrees |
| Shared candidate: old `{a}×2`, `{b}×2`, incoming `{a,x}×2`, `{b,x}×2` | D(a)=D(b)=D(x)=4; W0={a,b}; I=1/3. Candidate's full-history node beta=.5 | x assigned to both nodes; duplicate x `Dictionary.Add` during metric merge throws; outer catch swallows; complete errors not established | Denominator a4+b4+b4+x4=16; x last-node numerator2; I=1-(4+4+2)/16=.375 | At WI .35, mathematical paper accepts and main would trigger; no C# decision oracle |
| Candidate beta versus measurement beta: old `{a,b}×2`, `{a}`, `{b}`, incoming `{b,x}×2`; alpha=beta=.4 | Node union6, prominent b5; full-node candidate beta2/6<.4; old-word `(0,0,.2)` | Candidate admitted by 2/5=.4; errors still old-word `(0,0,.2)` | Newly assigned x included: B=(.4-2/6)/3=.0222; duplicate b denominator15; `(0,.0222,.3333)` | At WI .25 Rust would trigger, paper/C# old-word errors would not |
| Parent/child scopes: old `{a,b}×2`, `{a}×2`, alpha=.7; incoming `{a,b}`, `{x}`; branches enabled | Root receives 2 sources, a-child receives 1; child b count=3; ancestor a excluded from candidate adoption | Ordinary child gets only matched source | Notes root scope=2, child scope=1; `(0,0,0)` at both | No trigger; x does not inflate child counts |
| Refuge: old `{a}×3`, `{x}`, alpha=.7; incoming `{x}` | x total2/3<.7; old metrics `(0,0,0)`; unmatched source belongs to refuge context | Unmatched set receives source; refuge retained membership commit not demonstrated | `(0,0,0)`; refuge source list unchanged | Immediate decision unchanged; retained context incomplete |
| Empty update with two descriptors: old `{a,b}×2`, incoming empty | Unchanged vocabulary gives `(0,0,0)` | With no dynamic candidates, `(0,0,0)` | `(0,0,.3333)`; root inspected despite zero scope | Historical duplicate error could trigger on an empty batch |
| Invalid sequence: after batch1, submit batch3 | No contextual mutation on rejected update | Not executed; no equivalent assertion made | Error; exported tree unchanged; batch2 still succeeds | Confirms this specific failure boundary only |

The shared row tests a word temporarily assigned to multiple nodes via the public
update API. A stable descriptor shared across multiple nodes cannot be installed
through main's public snapshot API (it contains no resumable tree). Its separate
multiset-denominator behavior is established **statically**: both sum counts per
node; Rust overwrites one word measurement in ascending node ID order, whereas
C# `Dictionary.Add` rejects a duplicate key. No guessed intermediate measurement
is asserted. Multi-node source assignment and multi-node descriptor assignment
must not be conflated.

## Findings and classification

### Confirmed Rust defects on main

1. **State 0 vocabulary is mutated before errors.** `evaluate_hkt_recursively_for_batch`
   computes measurements from the old snapshot, then uses `apply_to_snapshot` to
   add assigned candidates before aggregating errors. This violates the old-word
   summation contract and differs from C#'s `getCurrentWordsIdsInHKT`. Executed
   fixtures demonstrate a missed reconstruction at WI threshold .2.
2. **Node descriptors overlap the expected set.** Builder-produced expected IDs
   include already assigned non-first words; updating never excludes these current
   descriptors from expectations. The metrics denominator counts them twice.
   The executed unchanged `{a,b}` case demonstrates spurious reconstruction.
   This is an update/measurement set-contract defect regardless of who repairs
   construction. Baseline construction itself remains outside this audit.
3. **New vocabulary has no persistent canonical identity on accepted updates.**
   Main allocates negative synthetic candidate IDs locally but does not append
   new tokens to the word legend in the accepted path. Retained token evidence
   cannot contribute historical counts for those IDs, and the next batch still
   reports the token as new. The two-update x fixture exposes the lost historical
   count through I=.3333 instead of .5. Repeated local synthetic allocation also
   presents a possible collision between different tokens; that consequence was
   not separately executed and should not be treated as a confirmed fixture.
4. **Refuge membership does not absorb mapped unmatched sources.** The update
   stage contains refuge incoming membership but apply skips refuge nodes; the
   executed refuge fixture confirms the retained source export does not grow.
   This violates contextual membership completeness, even though the inspected
   C# accepted path has the same apparent omission. Child delivery is not proof
   of retained membership completeness.

### Paper/C# ambiguities

* **W0 and unassigned expectations.** Both source-code paths exclude unassigned
  expectations from old-word sums. The supplied scientific document explicitly
  records this as unresolved. A descriptor-only versus expectation-inclusive
  calculation differs substantially in the repeated-candidate case; assigning
  node eligibility to an unassigned word remains unspecified.
* **Vocabulary sets versus per-node frequencies.** The paper equations do not
  settle how a shared descriptor's contextual count is collapsed into a single
  word contribution. C# cannot complete its dictionary merge for duplicates;
  Rust chooses the last node silently. This is insufficient evidence of a valid
  shared-word algorithm, despite matching denominator loops.
* **Historical candidate beta.** Both implementations admit against incoming
  candidate memberships while candidate strength includes historical evidence.
  A literal full-history node eligibility admits x in the historical fixture;
  both source paths reject it. Scientific clarification is needed before choosing
  this admission rule as an intended exception.
* **Node versus prominent-source eligibility.** C# candidate admission uses
  node-prominent sources, but State 1 old-word errors use node union. These agree
  only when the chosen prominent word covers all node members. Rust main follows
  this distinction; it should not be called a Rust-only defect.
* **Which scope supplies D(w).** Old descriptor measurements use contextual
  node memberships; candidate counts use parent-scope history and HKT incoming
  scope. Different scopes can yield different counts for the same token. The
  paper's symbols do not alone define how to convert those to one shared-word DF.
* **Admission timing.** Both code paths add alpha-admitted, failed-beta candidates
  to the importance denominator before beta assignment. The rejected-alpha case
  confirms exclusion. Whether observed but unadmitted words belong in W1 is a
  scientific vocabulary-definition question, not a reason to count all new tokens.

### Deliberate implementation differences

Rust's ordered maps/sets make ties and last-node overwrites deterministic; C#
uses HashSets and parallel work. Main returns zero empty-vocabulary alpha/beta
rather than NaN. Rust's four-decimal rounding uses different tie handling from
C#; arithmetic equality at non-half-way values is confirmed only for Rust and
established statically for C#. Four-decimal aggregation itself is inherited from
C#, with a real decision difference from the unrounded equation. No alternative
policy, compatibility layer, or fallback was introduced by this investigation.

### Insufficient evidence

There is **no executed C# parity**, no validated reproduction of the published
case study, and no claim about final-PDF differences from the accepted manuscript.
C# swallowed exceptions, stale dynamic candidate counts, duplicate object tokens,
and shared-word dictionary collisions prevent declaring C# an unquestioned oracle.
The exact half-way rounding boundary, a stable shared descriptor across nodes,
zero-evidence HKT behavior through the public API, and detailed candidate
ancestor effects are static findings or unresolved cases. Main's public notes
cannot expose unrounded word-level counts, candidate beta values, chosen prominent
word IDs, or full pending membership maps. Aggregate assertions do not claim to
observe these intermediates directly.

## Confirmed parity and discrepancies

Executed Rust fixtures and source-established C# equations agree for a unique
unchanged descriptor, rejection below alpha, alpha-admitted/failed-beta candidate
importance, old-word weakening, strict rounded thresholds, parent-restricted child
mapping, and incoming-only candidate beta. This is **static correspondence plus
Rust execution**, not confirmed execution parity between languages.

Confirmed discrepancies from C# source are main's State 0 pollution, overlap
between construction expectations and current descriptors, missing canonical
identity for newly accepted tokens, different shared-word failure behavior, and
rounding ties. Differences from the mathematical oracle additionally include
historical candidate eligibility and rounding before decisions. These are
classified above rather than all being labelled Rust defects.

## Recommended minimal corrections (not implemented)

1. Preserve the original W0 snapshot through error aggregation. Apply pending
   memberships and ancestor context only after a decision. Keep the missed-trigger
   fixture, then replace its characterization assertion with I=1/3 and triggering
   once the production correction is separately authorized.
2. Enforce disjoint current-descriptor and candidate sets at the update boundary;
   do not count an old descriptor again as a candidate. This addresses the
   unchanged-vocabulary defect without restructuring the baseline builder.
3. Register incoming token IDs persistently before scope mapping, using one
   canonical identity across parent/child scopes and subsequent retained evidence.
   Commit candidate evidence with the same identity, inside the existing update
   transaction design when that contract is adopted.
4. Commit refuge source unions consistently with incoming mapping and child
   delivery. Keep historical counts and retained-source membership synchronized.
5. Resolve shared-word counting, W0 expectations, candidate historical beta,
   candidate versus node beta denominators, and admission timing scientifically
   **before** changing those semantics. Publish explicit equations for the chosen
   scope and collapse rule. Do not copy C# dictionary failures or stale counts.
6. Choose error precision intentionally. Full precision accords with the supplied
   later contract; exact C# parity would require its rounding behavior, including
   ties to even. Do not silently switch between these objectives.

No production fix, threshold adjustment to force agreement, baseline algorithm
change, or subtree replacement change was made. The generic policies and rebuild
fallbacks already present on main were neither restored nor extended.

## Validation

* `cargo test -p realtime-seca-core --test parity_context_metrics -- --nocapture`:
  initial 10 fixtures passed after resolving Cargo dependencies outside the
  network-restricted sandbox.
* Final `cargo test --offline -p realtime-seca-core --test parity_context_metrics`:
  **19 passed, 0 failed**. Parameterized tests include equality and just-below
  boundaries for alpha, beta, and importance. Test file formatted with rustfmt; its final `--check` and `git diff --check` passed.
  An initial draft of the nineteenth case failed because its baseline produced
  two nodes rather than the intended single node. The dataset was corrected to
  have two shared baseline sources; production code and comparison rules were
  unchanged, and the corrected fixture passed.
* `cargo test --offline -p realtime-seca-core`: **154 passed, 0 failed**
  (111 unit, 5 large-dataset integration, 18 parity, 20 stream/skeleton integration;
  zero-test targets and doc tests also succeeded). This run preceded the final
  nineteenth parity case, which passed in the final focused run.
* C# execution unavailable as described above; no C# build/test result asserted.
* No architectural changes, so workspace-wide architectural checks were not
  required. Existing core test code emits nine unused-import/variable warnings;
  production files were left untouched.

Generated Cargo lock and build output are local artifacts only, not part of this
commit. Documentation and the one integration-test file are the only changes.
