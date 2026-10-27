# Recursive SECA v2 — scientific and implementation specification

**Status:** Executable research interpretation, updated 26 October 2026.  
**Implementation:** `crates/realtime-seca-v2/`.  
**Scope:** This specification describes v2, not a proposed change to `realtime-seca-core` (v1).

## 1. Purpose and sources of authority

SECA v2 independently implements recursive Hierarchical Knowledge Tree (HKT) construction and self-evolution. The objective is to reproduce the **resulting tree** of the C# reference where its output follows the published algorithm, without reproducing incidental implementation details, caches, IDs, or known inconsistencies.

Scientific foundations:

- Al Sulaimani and Starkey (2023), *Real-Time Event Detection Using Self-Evolving Contextual Analysis (SECA) Approach*, §§ V and VII-A–C, especially equations (2)–(12), DOI [10.1109/ACCESS.2023.3331219](https://doi.org/10.1109/ACCESS.2023.3331219).
- Abdul Aziz and Starkey (2020), *Predicting Supervise Machine Learning Performances for Sentiment Analysis Using Contextual-Based Approaches*, § III-A, steps 1–5 and the paragraph on child-branch construction, DOI [10.1109/ACCESS.2019.2958702](https://doi.org/10.1109/ACCESS.2019.2958702).

The papers establish the recursive word/source structure, strength and overlap criteria, conditional reconstruction, and (for SECA-Light) forgetting. They do **not** fully specify deterministic tie-breaking, exact branch-count boundaries, the lifecycle of unmatched incoming sources, or every historical candidate-accounting detail. The conventions below are therefore distinguished from claims of a uniquely recoverable original algorithm.

The C# reference is an **output oracle and diagnostic comparator**, not a requirement to mimic its intermediate mutations. Recorded fixtures are in `crates/realtime-seca-core/tests/fixtures/parity_hkt_reconstruction/`.

## 2. Data model, identity, and parameters

A `Source` has a nonempty external ID and a set of distinct, nonempty normalized tokens. Tokenization and normalization happen **before** v2 receives the source. Repeated tokens within a source count once.

The model stores:

- `retained: BTreeMap<SourceId, Evidence>`: source words and originating batch, independently of their current HKT memberships;
- `root: Hkt`: the learned hierarchy;
- `next_batch`: the next required batch index;
- `config`: construction and evolution parameters.

Each `Hkt` contains an ordered collection of `Node` values. Each node owns a descriptor-word set, a source-membership set, and optionally one child HKT. An empty descriptor set designates a **refuge**; no synthetic refuge word is stored. Node/HKT numeric IDs are not part of the scientific structure.

The current `Config::default()` is:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `alpha` | `0.7` | Minimum word strength for admission at an HKT |
| `beta` | `0.5` | Minimum source-overlap eligibility for grouping words in a node |
| `error_limits.alpha` | `0.1` | Alpha-error reconstruction limit |
| `error_limits.beta` | `0.2` | Beta-error reconstruction limit |
| `error_limits.importance` | `0.3` | Word-importance-error reconstruction limit |
| `minimum_branch_sources` | `2` | Inclusive count of residual-bearing sources needed for a child |
| `window` | `None` | Full SECA; `Some(gamma)` enables SECA-Light |

These defaults are **implementation settings**, not values uniquely dictated by the papers. In particular, the current beta-error default is `0.2`; earlier v2 code used `0.13`. All probabilities must be finite and in `[0,1]`; `minimum_branch_sources` and any supplied `gamma` must be positive.

Batches are numbered consecutively from zero. A retained source ID is immutable: the first occurrence wins and subsequent occurrences are ignored, including duplicates within a batch. After that identity is forgotten under SECA-Light, the same external ID may enter again as new evidence. Invalid batch numbers or empty IDs/tokens are rejected before insertion.

## 3. Context, vocabulary, and document frequency

For each HKT inspection or construction, define:

- \(S\): the HKT's effective source scope;
- \(X\): words excluded by actual ancestry-path descriptors (not sibling nodes);
- \(W_0\): the **existing descriptor vocabulary** of ordinary nodes in this HKT, before the update; refuge has no descriptor;
- \(F_S(w)=\{s\in S:w\in\operatorname{words}(s)\}\);
- \(D_S(w)=|F_S(w)|\), the number of **distinct source identities** containing \(w\).

Document frequency is calculated from retained source evidence **over the effective scope**, rather than reconstructed from the union of current node memberships. Consequently, a source not mapped to a node can still contribute to candidate strength and trigger a subsequent reconstruction.

The prominent count is

\[
p=\max\{D_S(w):w\notin X\ \text{or}\ w\in W_0\},
\]

with `0` if there is no eligible term. Including old descriptors in the prominent-count calculation preserves their ability to be measured if an unusual historical state has an old descriptor also present in the exclusion set. During initial construction, \(W_0=\varnothing\).

New, **provisional** candidate words are

\[
C=\{w\notin W_0\cup X:D_S(w)>0,\ D_S(w)/p\ge\alpha\}.
\]

Ratios with zero denominators evaluate to `0` in the implementation.

## 4. Baseline and reconstructed HKT construction

`recursive::construct(S, X, evidence, config)` is **the sole construction procedure**, used for both the baseline tree and triggered replacements.

1. Calculate distinct document frequencies over \(S\).
2. Admit words whose strength relative to the most frequent available term is at least \(\alpha\).
3. Process the admitted words in **descending document frequency**, breaking frequency ties by ascending token order.
4. For each word \(w\), compare its source set with each existing ordinary node \(z\) using

   \[
   \operatorname{eligibility}(z,w)
   =\frac{|F_S(w)\cap S(z)|}{|S(z)|}.
   \]

   If one or more nodes meet \(\beta\), add \(w\) to the node with the highest eligibility and union \(F_S(w)\) into that node's membership. Exact score ties choose the **first-created** qualifying node. If no node qualifies, create an ordinary node for \(w\).
5. Calculate sources in \(S\) not represented by **any** ordinary node. If nonempty, create one refuge node containing those sources. This includes sources with no admitted words or no tokens at all at the current scope.
6. For each attachment node \(z\), define child-specific exclusions \(X_z=X\cup W(z)\). Sibling descriptors remain eligible; a Refuge Node contributes no descriptors.
7. For each node \(z\), ordinary or refuge, define its **residual-bearing child scope**:

   \[
   S_{\mathrm{child}}(z)=
   \{s\in S(z):\operatorname{words}(s)\setminus X_z\ne\varnothing\}.
   \]

   Create its child HKT recursively from \((S_{\mathrm{child}}(z),X_z)\) only if

   \[
   |S_{\mathrm{child}}(z)|\ge\texttt{minimum\_branch\_sources}.
   \]

   Thus the branch minimum is **inclusive** but counts residual-bearing sources, not all sources in the parent node. No child is created from an empty residual dataset.

**Interpretation of child scope.** A source containing only words already described at higher levels terminates at its deepest meaningful parent. It is **not** carried into a child refuge merely to preserve parent-source coverage. A source with residual words that do not match an admitted descriptor, however, can enter a child refuge during construction. Original source evidence remains retained in both cases.

A source may belong to more than one ordinary sibling node. Descriptor exclusion is **cumulative along the actual ancestry path only**. A token may be reused in another branch under the same HKT, but cannot repeat on its own ancestry path.

## 5. Accepted evolution and conditional reconstruction

`Model::update(batch, sources)` retains new evidence first. Batch zero constructs the baseline. Every later batch invokes `recursive::evolve` on the root with the **entire retained source scope** and the set of newly inserted source IDs.

For an HKT with effective scope \(S\) and incoming IDs \(I\):

1. **Skip an empty incoming set.** No inspection or reconstruction decision is recorded for that HKT.
2. **Map new sources using established ordinary descriptors.** A source is added to every existing ordinary node sharing at least one descriptor token with it. Multiple ordinary matches are allowed; descriptors themselves do not change during mapping.
3. **Handle unmatched sources without making new topology.** If the HKT has an **existing** refuge, add sources that matched no ordinary node to that refuge. If no refuge exists, they remain in `retained` but have **no node membership in this HKT**. In particular, an accepted update **does not create a refuge** just because a source is unmatched.
4. Recompute document frequencies and provisional candidates using the full effective scope \(S\), including retained-but-unmapped sources. Calculate the three errors below.
5. **If any error exceeds its limit**, replace this HKT immediately with `construct(S, X, evidence, config)` and return. Its old descendants are replaced as part of construction; do not inspect them again during this update. Untargeted siblings remain intact.
6. **Otherwise accept:** retain all established descriptor words and existing topology; provisional candidates are **not promoted** to ordinary-node descriptors. Recurse into each existing child using its residual-bearing child scope and the intersection of that scope with \(I\). Each child independently evaluates its own errors and may reconstruct.

The effective child scope in step 6 uses the same `residual_scope` rule as construction:

\[
X_z=X\cup W(z),\qquad S_{\mathrm{child}}(z)=\{s\in S(z):\operatorname{words}(s)\setminus X_z\ne\varnothing\},
\qquad I_{\mathrm{child}}=I\cap S_{\mathrm{child}}(z).
\]

The key conceptual distinction is **stable learned structure versus accumulated evidence**: a source can be retained for future learning while currently having no path in the learned tree. A new ordinary descriptor or refuge can appear on reconstruction; mere mapping of an accepted update does not create new nodes.

### 5.1. Reconstruction consistency

A triggered HKT is reconstructed using exactly the **effective scope and exclusions** passed to its inspection, not just the latest batch and not stale child-node descriptors. The intended equivalence is

\[
\operatorname{Reconstruct}(H,S,X;\theta)
\equiv\operatorname{Construct}(S,X;\theta).
\]

This equality applies to **reconstruction**, not to every accepted incremental update. An accepted tree may differ from a fresh tree built over the same retained evidence because conditional reconstruction deliberately provides temporal stability.

## 6. Error measures and trigger semantics

All measures use distinct source sets and the current HKT's **old descriptor vocabulary** \(W_0\), not historical candidate caches. Let \(p\) and \(C\) be defined as in §3. Define the updated strength and ordinary-node eligibility:

\[
\operatorname{strength}(w)=\frac{D_S(w)}{p},\qquad
\operatorname{eligibility}(z,w)=\frac{|F_S(w)\cap S(z)|}{|S(z)|}.
\]

The implementation calculates:

\[
A=\begin{cases}
\frac{1}{|W_0|}\displaystyle\sum_{w\in W_0}
\max(0,\alpha-\operatorname{strength}(w)),&|W_0|>0\\
0,&|W_0|=0,
\end{cases}
\]

\[
B=\begin{cases}
\frac{1}{|W_0|}\displaystyle\sum_{w\in W_0}
\max_{z:\,w\in\operatorname{words}(z)}
\max(0,\beta-\operatorname{eligibility}(z,w)),&|W_0|>0\\
0,&|W_0|=0,
\end{cases}
\]

\[
I=\frac{\sum_{w\in C}D_S(w)}
{\sum_{w\in W_0}D_S(w)+\sum_{w\in C}D_S(w)}.
\]

Thus alpha error measures weakening of established descriptors, beta error measures violations of established within-node co-occurrence (using the **worst** deficit if a descriptor appears in multiple ordinary nodes), and word-importance error measures the relative mass of newly eligible candidates.

Reconstruction occurs precisely when

\[
A>\tau_A\quad\lor\quad B>\tau_B\quad\lor\quad I>\tau_I.
\]

Comparisons use **full floating-point precision**, without the C# reference's four-decimal rounding. Equality to a limit is **accepted**, not reconstructed. Errors and reconstruction decisions are recorded with the HKT's path at the time of inspection. Provisional candidates affect \(I\) but are never added to \(W_0\) on an accepted update.

## 7. SECA-Light, forgetting, and persistence

For `window = Some(gamma)`, after processing a batch with index \(t\), retain source evidence from batches satisfying

\[
\operatorname{batch}(s)\ge\max(0,t-(\gamma-1)).
\]

Then recursively intersect every node's membership with the remaining retained source IDs. **Forgetting does not, by itself, delete descriptors, empty nodes, or child HKTs**. An empty batch still advances the batch index and therefore the forgetting window. Full SECA (`window = None`) retains historical evidence indefinitely.

**Ordering matters:** evolution for batch \(t\) runs **before** this forgetting step; the implementation does not first trim the window and then calculate change errors. This is the current operational convention and should be tested separately if strict window-at-decision semantics are required.

The model serializes its configuration, retained evidence, tree, and next batch index. Deserialization validates parameters, retained-evidence timing, node membership subsets, ancestry-path descriptor non-repetition, child-parent scope consistency, and at most one refuge per HKT. It intentionally does **not** demand that every retained source occur in a tree node after accepted updates, because unmapped retained evidence is valid.

The current validation is permissive in another respect: it checks subset relations but does not prove that every existing descriptor-matching source was mapped into the correct node. Strengthening restoration validation without forbidding legitimate unmapped evidence is a separate follow-up.

## 8. Reference-output parity and intentional differences

`tests/csharp_tree_parity.rs` replays saved C# input batches through v2 and compares recursively normalized trees. Normalization preserves descriptor words, source memberships, refuge nodes, and parent-child attachments, while ignoring numeric IDs, candidate caches, sibling enumeration order, and internal reconstruction traces.

Based on the tested fixture set reported on 26 October 2026, these **nine fixtures match all checked cycles**:

`child`, `child_reconstruct`, `root_reconstruct`, `overlap_recursive`, `unmatched`, `one_prominent`, `merge_many`, `separate`, `frequency_vs_vocabulary`.

Three strict comparisons remain deliberately **ignored as known output differences**:

| Fixture | C# reference | v2 convention | Reason for retaining v2 |
| --- | --- | --- | --- |
| `branch_boundary` | Builds a child when the **parent's total source count** meets the branch minimum | Requires at least that many **residual-bearing** child sources | Better reflects the reduced dataset used for recursive analysis in CA 2020 |
| `collision_true_tie` | Chooses the last node attaining the maximum overlap in the recorded execution | Chooses the first-created maximum after deterministic word ordering | Neither tie rule is specified by the papers; deterministic execution is preferable to C# enumeration artifacts |
| `refuge` | An accepted update adds a source to a **refuge child** but leaves it absent from the **parent refuge** | Adds the source to the existing parent refuge and then propagates it into its eligible child | Maintains the child–parent membership subset invariant |

These ignored tests are **not** passing parity claims; they remain useful strict oracles for diagnosing deliberate divergences. They should not be silenced by changing tree normalization or rewriting the recorded C# outputs.

A further known numerical divergence is the C# reference's cached/stale expected-word frequencies in some consecutive updates. V2 recomputes document frequency from actual retained evidence. This can change reconstruction decisions; it is an intentional correction, not a C# behavioural-parity target.

## 9. Executable verification

From the repository root:

```sh
cargo fmt --all -- --check
cargo test -p realtime-seca-v2
cargo clippy -p realtime-seca-v2 --all-targets -- -D warnings

# Strict C# comparisons except the explicitly ignored known differences:
cargo test -p realtime-seca-v2 --test csharp_tree_parity -- --nocapture

# Explicitly reproduce a known divergence when required:
cargo test -p realtime-seca-v2 --test csharp_tree_parity \
  csharp_branch_boundary_tree_parity -- --ignored --nocapture
```

Particularly relevant regression tests in `tests/mathematics.rs` cover:

- admitted-word ranking and deterministic collision ties;
- residual-bearing branch and child-source filtering;
- absence of automatic refuge creation during accepted updates;
- retained-but-unmapped evidence triggering a later reconstruction;
- independent child reconstruction versus fresh construction;
- selective reconstruction without changing an unaffected sibling;
- deferred candidates, threshold boundaries, forgetting, and serialize/restore.

## 10. Open scientific and engineering questions

1. **Branch minimum:** The residual-bearing interpretation is supported by the 2020 construction description, but the publication does not completely formalize its threshold boundary; retain it as an explicit v2 choice.
2. **Candidate lifecycle:** The stored descriptor vocabulary remains unchanged on acceptance. Whether the C# reference's provisional per-node beta assignments have any **additional future structural effect** beyond v2's freshly derived candidates is not established by existing tree-output fixtures.
3. **Metric scope and calibration:** Test alpha, beta, and importance over longer realistic update sequences, especially overlapping memberships, repeated novel terms, and near-threshold reconstructions. Matching final trees for a small fixture suite does not prove all intermediate errors agree.
4. **Unmatched-source semantics:** Accepted updates preserve unmatched evidence without inventing new topology. Verify that retained-but-unmapped evidence still influences later reconstruction and is handled consistently in deeper scopes; the C# paper does not specify every such case.
5. **SECA-Light:** Clarify whether the desired scientific semantics require forgetting **before** or **after** the current batch's reconstruction decision, and test longer windows and empty updates.
6. **State validation:** Strengthen corruption detection as appropriate without reintroducing the invalid assumption that learned node memberships always cover all retained evidence.
7. **Performance:** The current implementation prioritizes transparent, mathematically explicit source sets and recursion over optimization; benchmark with realistic source volumes before introducing additional indexes or mutable caches.

**Working rule:** A discrepancy with C# is a research question, not automatically a v2 defect. Prefer the simplest deterministic implementation consistent with the published mathematics, and keep any justified output differences explicit and testable.
