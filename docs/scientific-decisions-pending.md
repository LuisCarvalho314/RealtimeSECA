# Scientific decisions pending

2026-10-13. Recommendations for scientific review, **not approved algorithm
changes**. Integration changes ingestion and snapshot validation only. The
canonical metric path and HKT builder remain unchanged; the two ignored strict
parity assertions remain evidence of observed discrepancies.

Read together with [context investigation](research-context-semantics.md),
[construction investigation](research-hkt-construction.md), and the
[executable evidence summary](parity-executable-summary.md). The investigations
used a supplied working tree and external serial Mono reference harness; their
provenance describes those measurements, not a clean-main reproduction. They
predate the summary's availability in their checkout. Scientific observation tests
characterize existing behavior, not the proposed implementation.

| Decision | Recommendation and rationale | Confidence | Minimal expected Rust changes after approval |
| --- | --- | --- | --- |
| E01: shared words and metric source sets | Count each word once using distinct retained-plus-incoming sources in the current parent scope. Use unique W0/W1 sets for strength and importance. Keep eligibility per `(node, word)` and reduce old-word beta deficits by the worst adopting node. Scope DF avoids losing occurrences outside an adopting node; worst-node reduction prevents a strong adoption masking a weak one. | High for deduplication and separate node eligibility; medium for full scope DF; medium-low for worst-node reduction, which the paper does not specify. | In evidence/metrics, build scoped word source sets, deduplicate vocabularies and denominator, retain node-word eligibility and explicitly reduce deficits. Preserve ancestor exclusion and failure isolation. |
| E02: old versus expected vocabulary | Keep W0 as the frozen union of ordinary descriptors. Previously expected but unassigned words enter W1 only after fresh alpha admission. Accepted assignments become W0 on the next update. Unassigned expectations have no defined adopting node for equation 8; including them in W0 requires an additional eligibility rule. | Medium. A temporal expectation-inclusive reading remains plausible; author clarification is desirable. | Freeze descriptor-only W0 before candidate handling; refresh expected counts/admission from scoped evidence and avoid treating cached expectation keys as old descriptors. Document this interpretation explicitly. |
| E03: candidate beta history and persistence | Use retained-plus-incoming word intersections over the full updated node source union, recomputing each cycle. Persist accepted beta assignments as descriptors with their full historical memberships. Incoming-only numerators penalize small batches while the denominator accumulates; transient assignments cannot provide the intended next-cycle vocabulary. | Medium-high for consistent historical intersections; descriptor persistence is an explicit proposed lifecycle change requiring review. | In context/evidence, replace incoming-only candidate intersections, discover historical unassigned candidates, and commit descriptors/memberships only after acceptance. On rejection use the existing scoped reconstruction path. Test subsequent cycles, child scopes and forgetting. |
| E05: recursive vocabulary scope | Remove all ordinary descriptors adopted at the current HKT before constructing any child; intersect residual rows with each attachment node's sources and recompute DF. Apply at baseline and every reconstruction entry. Original CA supports level exclusion; reciprocal descendants duplicate already represented sibling topics. Preserve retained token evidence. | Medium-high for level exclusion; high for observed baseline/reconstruction discrepancy. Later SECA wording also permits a parent-node-only reading. | In the existing builder, share one residual level scope across branch expansion and filter per attachment. Preserve reconstruction ancestor exclusion. No alternate builder or evidence deletion. |
| E06: branch minimum | Admit equality: parent node source count `>= minimum`, provided residual evidence is nonempty, at baseline and reconstruction. Document the setting as minimum parent-context support. A minimum naturally includes equality, but the papers do not define this integer operator. | Medium for inclusive boundary; high for executed cutoff sweep; low for asserting the publication defines either the operator or counted set. | Change the single builder comparison and document its counted set; add below/equal/above tests at baseline and reconstruction. Do not add retroactive branch creation on accepted updates. |
| E07: ordering and collision ties | Rank candidates by descending scoped distinct-source DF, then ascending exact UTF-8 token bytes. Visit once in rank order; scan nodes in creation order and choose the greatest eligible ratio, first node on exact ties. Recompute rank in each residual scope. Frequency ranking follows CA; canonical ties supply missing portable determinism. | High for frequency traversal and measured input sensitivity; medium for the engineering tie contract. | In the canonical builder, use token-based ranking rather than vocabulary allocation IDs, create the actual candidate on no collision, and compare rational collision scores with overflow-safe cross multiplication. Keep vocabulary IDs stable and beta admission semantics unchanged pending numeric review. |

## Review dependencies and unresolved choices

Approve E01/E02/E03 together: word source scope, vocabulary classification and
assignment persistence determine the same errors across cycles. The paper does
not uniquely settle full scope versus adopted-membership union DF, the reduction
of multiple node eligibilities, or the status of an unassigned old expectation.
Worst-node beta is a proposed extension, not a published formula.

For construction, review E07 first, then E05, then E06. Greedy ordering changes
memberships, later denominators and reconstruction targets, so topology changes
need lifecycle regressions rather than renamed goldens. E05 sacrifices explicit
reciprocal co-occurrence branches while preserving their underlying evidence;
whether accepted-update candidates must exclude sibling descriptors remains an
unresolved follow-up before making level exclusion a persistent state invariant.
E06 additionally leaves parent source count versus sources containing residual
words unresolved; this proposal deliberately retains the current counted set.

E04's full node-union eligibility denominator is retained (high confidence), not
a pending replacement. Full-precision error comparisons and existing thresholds
remain unchanged. Serial C# behavior is evidence, not authority for collection
order, stale candidate counts, duplicate-key failures or rounding discrepancies.

After review, update the scientific specification before production changes,
then align architecture/state contracts as needed and add focused regressions
for repeated updates, permutations, selective reconstruction, ancestor exclusion,
forgetting, restart equivalence and atomic failure. Hand-derived proposed outputs
have not been executed through a modified metric path or builder.
