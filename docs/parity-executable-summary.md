# Consolidated executable C#–Rust SECA parity evidence

Consolidated on 2026-10-13 against fetched main
**1d9309274dc585f02f8a0328b8efdc323b9684cb**. This is the authoritative
cross-language evidence register for these fixtures. **Full parity is not
established. No production correction is included or scientifically authorized
by this report.** The engine should expose the algorithm, not its implementation
history. Recommendations below require subsequent scientific decisions and work.

## Revisions and provenance

Reviewed commits:

- Context/metrics: `5a3cbc7b2d1d7f270b2b5bd300b472a16bc03f58` and
  `0176dfea0cc2195c53e60cd1e333162319481ebf` (stderr whitespace normalization).
- HKT/reconstruction: `970db4b5134d12f0555da48d0b14090cad537183`.
- Both agents started from `9e70e2136dea237f1e922352df0ae24d4441c2fc`
  with unrelated uncommitted integration changes. Their original Rust execution
  is not clean-main evidence. This consolidation independently executes their
  combined Rust tests against the exact main revision above; observations match
  the committed goldens without regeneration or relaxed assertions.

**C# upstream Git revision is unknown:** the supplied archive has no Git metadata.
A Rust repository commit obtained by running Git inside that archive would identify
its enclosing repository, not the C# revision. Exact source content is identified
by [the context source manifest](../crates/realtime-seca-core/tests/fixtures/parity_context_metrics/source-hashes.json)
and [the HKT provenance manifest](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/provenance.json).
Both compiled-input manifests agree. Consolidation verified all 13 source hashes.
HKT, Node, SECAAlgorithm, all three error classes and five other data classes are
byte-identical between original and compilable reference copies. DataHelper is
not byte-identical: normalized comparison shows three C# 8 `using` declarations
changed to ordinary declarations for Mono compatibility. Persistence construction
is disabled and these database methods do not execute. This adaptation predates
consolidation; neither C# copy was edited here.

[Consolidation provenance](parity-executable-provenance.json) records all current
Rust core source hashes, harness file hashes, and hashes of the five compiled
reference dependencies. The six agent-recorded engine file hashes match clean
main; builder, baseline and rebuild files also match the supplied agent checkout.
The compiled SourceCleanerEngine dependency is executed, rather than recreated;
its source-to-binary correspondence has not been independently established.
Obtaining the original archive release/commit and dependency build provenance is
required before claiming an exact upstream C# revision.

## Execution and parameter equivalence

Both agents' executable artifacts were reviewed and independently replayed.
The existing reference builds under Mono **6.8.0.105**, with 22 inherited warnings;
`verify.py` passes manual metrics/topology, follow-up updates, deterministic repeats
and invalid-input failure. Rust is **1.92.0**. The execution protocol is preserved
in [harness documentation](parity-reference-harness.md).

This executes adapted original algorithm classes, **not the uninstrumented
Windows application**. The harness serializes parallel iteration, substitutes
deterministic IDs, disables unused persistence, surfaces swallowed exceptions,
and observes actual metrics immediately before `isAccepted`. It does not
recompute errors or replace successful admission/reconstruction formulas.
Race-dependent behavior is outside the comparison. Raw JSON fields containing
zero IDs and pending state are preserved and are not scientific outputs invented
by instrumentation. Baseline metrics and suppressed-descendant decisions are not
fabricated. Failure instrumentation changes reporting; it cannot establish the
original application's post-exception continuation.

Each input fixes ordered documents, unique positive source IDs, unique preprocessed
case-sensitive words, batch boundaries, alpha, beta, branch minimum and all three
Option1 error thresholds. Rust consumes the same relations and external source
identities, uses identical construction/update parameters, and explicitly selects
**MemoryMode::Full** in both executable adapters. C# selects accumulated SECA
(`isLightSECA=false`). These tests exclude forgetting, NLP/ingestion parity,
duplicate source-word pairs and proprietary datasets. Older Rust-only regression
helpers are supplemental evidence, not additional C# comparisons.

- Context: 19 inputs, 17 successful goldens replayed twice byte-identically;
  shared-candidate duplicate-key and empty-input rejection reproduced. Empty
  input has no C# algorithm decision and shared candidate has no successful JSON.
- HKT: 21 inputs, 65 snapshots, each execution repeated byte-identically. Raw C#,
  normalized C#, Rust observations and per-cycle equality classifications retained.
- Context tests independently recompute aggregate errors from captured C# word
  metrics. Numerical equivalence permits only the four-decimal rounding envelope
  (0.00005); decision differences are separately asserted.
- HKT normalization removes incidental IDs, sorts words/source sets, and preserves
  refuge identity, attachments and node order. Decisions map pre-reconstruction
  IDs to previous-tree paths. A new snapshot-count assertion prevents zipped
  comparisons from silently omitting a cycle.

Rust public diagnostics do not expose temporary per-word metrics or candidate
assignment maps. Candidate-assignment correspondence therefore combines executed
C# fields, Rust aggregate/public results, focused Rust regressions and source
inspection; it is not direct intermediate-state parity. Ordered equality failures
for sibling enumeration are not proof of different labelled tree structure.

## Confirmed matching behavior

Within the observed fixtures, both engines agree on unchanged old-word errors,
low-evidence errors, alpha rejection just below the boundary, admission at equality,
descriptor-only W0, inclusion of alpha-admitted candidates in the importance
denominator despite beta rejection, weakening errors within rounding, and strict
threshold equality when the represented values coincide. Ordinary merged-node
source unions and multi-node mapping agree. Child updates see parent-matched
sources and accepted memberships propagate in the tested cases.

Root reconstruction suppresses descendant inspection and includes retained plus
incoming scoped evidence. Dominant-word selection and successive reconstructions
agree in the tested cases. Child reconstruction selects only `root/0`, excludes
ancestor a from the b/c/x replacement, preserves its attachment and the d sibling's
memberships/IDs, and a subsequent update reconstructs the root in both languages.
Its untouched sibling order differs, so whole-tree ordered equality is not claimed.
No reconstruction-scope or ancestor-filtering defect is demonstrated here.

## Evidence-backed discrepancy register

Severity is scientific/behavioral risk, not an instruction to copy C#. Confidence
in an observed mismatch is separate from confidence in the scientific correction.
CM names refer to [context fixtures](../crates/realtime-seca-core/tests/fixtures/parity_context_metrics/);
HKT names refer to [construction fixtures](../crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/).
Exact ordered datasets/settings are in the named JSON inputs.

| ID / prior audit | Classification | Evidence and reconstruction/topology consequence | Severity / confidence | Scientific interpretation and recommendation |
|---|---|---|---|---|
| E01 / F05 | Shared-word scientific ambiguity; C# failure confirmed | CM `shared_candidate`: assigning x to two nodes throws duplicate-key exception in original metric merge. Rust accepts with WI=1/3. No successful C# decision to compare. Existing descriptors repeated across nodes remain untested as a stable C# golden. | High potential impact; failure confirmed, scientific correction unresolved | Node-occurrence denominator plus one overwritten measurement per word is not proved consistent with paper set sums. Specify contextual D(w), W0 union and denominator collapse; then implement one canonical aggregation. Do not emulate exceptions. |
| E02 / F06 | Verified observable W0 agreement; scientific ambiguity | Both use old node descriptors, excluding assigned/unassigned candidates from W0. Repeated candidates still affect WI; agreement does not settle whether expectations belong to later W0. | Medium; behavior confirmed, interpretation unresolved | Define old/admitted/assigned/expected vocabularies across cycles before changing W0. |
| E03 / F07 | Historical candidate scientific ambiguity | CM `historical_candidate`: full-history overlap4/6 passes beta=.5, incoming2/6 fails. C# has no assignment; Rust rule and aggregates correspond. Immediate errors match; alternate admission could alter future decisions. | Medium; C# execution and Rust rule confirmed, direct intermediate Rust comparison incomplete | Determine whether admission must include historical candidate memberships; matching errors alone do not validate it. |
| E04 / F07 | Intentional Rust denominator difference | CM `node_union`: C# candidate x passes2/5=.4 using prominent-word sources; Rust rejects2/6 using full node union. Both have WI=.2 and accept. No later topology/trigger effect demonstrated. | Medium; rule difference confirmed, assignment observation limited | Node-union denominator is documented and consistent with equation3/8 node notation; keep it unless scientific evidence changes that interpretation. |
| E05 / F08 | Confirmed cross-language recursive-scope discrepancy; scientific ambiguity | HKT `overlap_recursive`: Rust adds c grandchildren below b and b below c for shared sources3,4; C# stops after b/c. Minimum1 with two shared sources isolates cutoff. Baseline/rebuilt topology and inspected contexts differ. | Medium; topology confirmed, correction unresolved | C# removes adopted rows before recursion; Rust reuses pre-construction child scope. Consult original CA contract before deciding whether all sibling-adopted words must be removed. |
| E06 / F09 | Confirmed branch-boundary discrepancy; scientific ambiguity | HKT `branch_boundary`: source count2/minimum2 creates a b child only in C#. Later inspection paths differ; topology changes immediately. | Medium; confirmed | C# >= versus Rust >. Resolve inclusive minimum scientifically, then apply uniformly to baseline and reconstruction. |
| E07 / F11 | Confirmed greedy-order/collision discrepancy; scientific ambiguity | HKT `collision_true_tie`: C# merges b/c, Rust a/c. `frequency_vs_vocabulary`: C# z;b;a, Rust z;{a,b}. Memberships/topology change; subsequent contexts can differ. | Medium; confirmed | C# expected traversal follows frequency-derived order and last maximum; Rust expected IDs and first maximum. Specify portable frequency/record-order traversal and collision tie contract; avoid hash-bucket emulation. |
| E08 / F10 | Intentional justified precision difference | CM `alpha_threshold` and `beta_threshold`: C# rounds1/12 to.0833 and accepts at.0833; Rust full precision triggers. Each isolates one trigger reason. | Medium; confirmed decision change | Preserve strict full-precision comparisons. Improve diagnostic formatting separately: displayed `0.0833 > 0.0833` hides real values. No four-decimal scientific requirement is established. |
| E09 | C# stale-expectation deviation from accumulated evidence | CM `expected_trigger`: second x update keeps C# x2 instead of accumulated x4; denominator6 yields WI=.3333 and acceptance, Rust denominator8 yields WI=.5 and reconstruction at.4. | Medium; confirmed decision change | Under descriptor-only W0 and accumulated DF, Rust follows current specification. Preserve recomputation; never make counts stale for parity. W0 ambiguity remains separate. |
| E10 | C# refuge deviation / justified Rust retention | CM/HKT `refuge`: Rust commits incoming unmatched parent refuge sources; C# parent remains stale even while its child gains source6. Immediate old-word errors agree, memberships differ. Future reconstruction effect not executed. | Medium; membership confirmed, later impact unverified | Preserve complete refuge union, consistent with unmatched-source semantics. Test subsequent parent-scoped reconstruction. |
| E11 | C# State0 eligibility deviation from paper interpretation | CM `node_union`: C# State0 b eligibility2/3 uses prominent sources; State1 is5/6 using node union. Option1 uses State1; immediate decision unaffected. | Low for this decision path; C# field confirmed | Do not copy State0 denominator into Rust. No executed Rust State0 metric parity is available. |
| E12 | Incidental ordering/representation difference | HKT `siblings`: f,e versus e,f under d; labelled memberships/attachments match. Construction expected-set persistence also differs from C# dynamic exported candidates. | Low observed impact; confirmed | Preserve raw distinctions but do not recommend fixes for IDs, sibling collection order or stored candidates alone. No reconstruction change solely from these representations was proved. |
| E13 | Intentional empty-scope/input-contract difference | Rust skips empty child scopes; C# extra inspections accept in successful fixtures. Harness rejects empty batches before algorithm execution. | Low observed impact; confirmed successful cases, empty-trigger behavior untested | No correction justified. Do not infer empty reconstruction/zero-denominator parity from rejected input. |

Prominent-word frequency ties respect source/token order in Rust-only tests; the
executed `ties_ab`/`ties_ba` cases verify their resulting merged tree, not a public
intermediate Rust prominent-word metric. The older `collision` dataset is not a
simultaneous collision tie: c is visited before b exists. E07 relies on the separate
true-tie fixture. C# create-first/remove-other and concurrent row mutation concerns
remain historical source hypotheses, not demonstrated corrections.

## Minimal reproductions

Notation: semicolons separate distinct documents; arrows separate batches. IDs
are allocated in listed order. Unless shown otherwise all three error thresholds
are1; alpha/beta/minimum appear after each dataset. These are small reproductions,
not a proof of mathematical minimality. The full JSON fixes every parameter.

| Fixture | Dataset / parameters | Isolated result |
|---|---|---|
| CM `shared_candidate` | a;a;b;b → ax;ax;bx;bx / .5/.5/1000 | Original duplicate-key failure; Rust WI=1/3 |
| CM `historical_candidate` | ax;ax;a;a → ax;ax / .6/.5/1000 | Incoming versus historical beta numerator |
| CM `node_union` | ab;ab;a;b → bx;bx / .4/.4/1000 | Prominent versus union beta denominator |
| CM `expected_trigger` | a;a → ax;x → ax;x / .5/.9/1000, WI threshold.4 | Stale expectations flip reconstruction |
| CM `alpha_threshold`, `beta_threshold` | ab;ab → a;a;a;a / .5/.5/1000, selected error threshold.0833 | Rounded acceptance versus precise trigger |
| HKT `branch_boundary` | ab;a → ab → a / .7/.5/2 | Inclusive child creation |
| HKT `overlap_recursive` | ab;ab;abc;abc;ac;ac;a → abc → abc / .8/.75/1 | Sibling-word reuse creates extra grandchildren |
| HKT `collision_true_tie` | a;b;ac;bc → abc → abc / 1/.5/99 | Last versus first maximum changes merged node |
| HKT `frequency_vs_vocabulary` | ab;ab;b;z;z;z;z → z → ab / .5/1/99 | Expected traversal changes merging |
| CM `refuge` | a;a;a;x → x / .7/.5/1000 | Accepted parent refuge retention |

## Priorities for subsequent implementation agents

1. **Resolve shared-descriptor aggregation (E01), then W0 and historical beta
   (E02/E03).** Derive explicit source-set equations and compute independent
   expected values on multi-node/multilevel shared-word cases. Shared-candidate
   failure is not a substitute for a shared-descriptor oracle. Include asymmetric
   node counts, reordered nodes and thresholds that expose decision changes.
2. **Resolve original CA recursive scope, minimum boundary and greedy ordering
   (E05/E06/E07).** These already change HKT topology. Consult the referenced CA
   construction specification and establish deterministic scientific rules before
   proposing corrections. Test baseline and rebuilt subtrees with overlap, true
   ties, reversed input, unequal frequencies, exact minimum and adjacent counts.
3. **Preserve justified Rust behavior:** full precision, fresh accumulated candidate
   counts, complete refuge retention and documented node-union beta. Extend the
   refuge/candidate cases through later reconstruction to measure deferred effects.
   Improve threshold diagnostics as a separate presentation correction.
4. **Close observational/provenance gaps:** obtain C# archive revision/dependency
   provenance; add separately reviewed read-only Rust intermediate instrumentation;
   examine ancestor candidate propagation and historically admitted but unassigned
   words. Study empty and zero-evidence scopes without forcing harness parity.
   Test half-way rounding and WI boundaries separately. Validate final publication
   and original CA text; this consolidation uses repository scientific specification
   and the agents' manuscript analysis, with no new published-paper verification.

No new policy enum, C# compatibility path, full rebuild alternative, obsolete
structure or production instrumentation is introduced. Any future correction must
remain one canonical update path, update its scientific contract, and preserve
transactional/state invariants. No correction is justified solely by IDs or order
of independent siblings.

## Validation and commands executed

The harness remains an external local prerequisite because original C# and its
compiled dependencies must not be redistributed. The integrated replay scripts
accept `SECA_REFERENCE_ROOT`, pointing to a checkout with the unchanged harness
and supplied reference. Their default mode verifies existing goldens; HKT capture
now requires explicit `--capture`, like context capture. Rust tests need no C#
installation. [Harness protocol](parity-reference-harness.md) is included as
documentation only; it does not provide restricted source or binaries.

Executed from `/home/azureuser/risklive-dev/experimental`:

```sh
git fetch origin main
tools/seca-reference-harness/build.sh
python3 tools/seca-reference-harness/verify.py
python3 crates/realtime-seca-core/tests/fixtures/parity_context_metrics/replay.py
python3 crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/reference.py
```

Executed from isolated integration checkout `/tmp/seca-executable-integration`
on main, with only tests, fixtures and evidence documents imported:

```sh
SECA_REFERENCE_ROOT=/home/azureuser/risklive-dev/experimental python3 crates/realtime-seca-core/tests/fixtures/parity_context_metrics/replay.py
SECA_REFERENCE_ROOT=/home/azureuser/risklive-dev/experimental python3 crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/reference.py
cargo test --offline -p realtime-seca-core --test parity_context_metrics --test parity_hkt_reconstruction
cargo test --offline -p realtime-seca-core --test parity_context_metrics expected_failure_csharp_alpha_threshold_parity -- --ignored
cargo test --offline -p realtime-seca-core --test parity_hkt_reconstruction strict_executed_csharp_parity -- --ignored
cargo test --offline --workspace
cargo fmt --all
cargo fmt --all -- --check
cargo clippy --offline --workspace --all-targets -- -D warnings
git diff --check
```

Both integrated C# replay scripts pass against existing goldens and provenance.
Focused Rust tests: **32 passed, 2 ignored**, rerun after the coverage and discrepancy-acceptance
assertions. Both ignored strict comparisons were explicitly run and **failed as
expected**: precision (Rust true versus C# false), and branch-boundary topology.
These assertions remain intact; the passing characterization suites document
known differences, not universal parity. Workspace validation: **138 passed, 2 ignored, 0 failed** (2 CLI unit,
74 core unit, 5 large-dataset, 20 context, 12 HKT, 13 batch-processing and
12 stream tests). Large-dataset tests took229.79s and the batch suite70.58s.
The workspace run began before the final coverage assertions; focused suites
were rerun afterward and pass. Final formatting, clippy with warnings denied,
and whitespace checks pass. No full-paper or deployment readiness claim follows
from these results.

Files above400 lines were reviewed: the 655-line context and 637-line
HKT integration suites preserve focused historical regressions plus executable
adapters. Large fixture JSON is captured evidence rather than algorithm code.
Production source, configuration defaults and original C# are unchanged. The
historical reports retain their original revision-specific conclusions, and the
prior audit receives only a dated executable-evidence status note. Engineering,
performance, forgetting and readiness findings are outside this consolidation.

Publication validation: the staged change contains173 evidence/test text files.
All input words are synthetic a/b/c/d/e/f/x/z. The staged production-source diff
is empty; no original C# source, proprietary dataset, binary, target directory or
harness build artifact is included. One commit is published on the separate
`integration/executable-parity-summary` branch; main is not merged or pushed.
