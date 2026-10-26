# Recursive SECA V2 scientific specification

Status: executable research interpretation, independent of V1. This does not
approve pending changes to V1. Written before the V2 Rust types.

## Evidence

Published foundations: [SECA 2023](https://doi.org/10.1109/ACCESS.2023.3331219),
V, VII-A–C, equations 2–12; [CA 2020](https://doi.org/10.1109/ACCESS.2019.2958702),
III-A steps 1–5 and branch paragraph. Read accessible author-hosted full text;
the final SECA PDF endpoint was unavailable. Published rules establish distinct
source counts, strength/eligibility admission, three change errors, top-down
replacement, and subsequent forgetting. CA orders words by frequency, unions
sources on collision, collects unmatched sources, and recursively analyzes
residual vocabulary. Its branch paragraph discusses residual-bearing sources.
Exact ties and several temporal/overlap cases are unspecified.

Repository evidence: algorithm.md, scientific-decisions-pending.md,
scientific-decision-proposals.md, research-context-semantics.md,
research-hkt-construction.md, seca-paper-benchmark.md. C# HKT.cs addNode,
createBranch and context fixtures show inclusive parent-count gating, mutable
candidate counts and enumeration-dependent collisions. Rust V1 uses exclusive
parent-count gating, vocabulary-ID traversal, incoming-only candidate beta and
node-aggregated importance. These are observations, not mathematical authority.

## Sets and equations

A source has an exact nonempty identity and a set of normalized tokens. First
retained identity wins. HKT scope S is all evidence at root, or its containing
node's accepted membership below root. X is the set of excluded ancestor words.
F(w)={s in S: w in tokens(s)}, D(w)=|F(w)|. All counts are distinct sources.
Old descriptors W0 are frozen before mapping/admission; refuge has no descriptor.
Let p=max D(w) over nonexcluded tokens and old descriptors. Candidates
C={w outside W0 and X: D(w)>0 and D(w)/p >= alpha}; W1=W0 union C.
Mapped memberships N1(z)=N0(z) union incoming sources sharing any old descriptor;
unmatched incoming sources enter refuge. Multiple ordinary matches are allowed.

    strength(w) = D(w)/p
    eligibility(z,w) = |F(w) intersection N1(z)| / |N1(z)|
    importance(w) = D(w) / sum(v in W1) D(v)
    A = sum(w in W0) max(0, alpha-strength(w)) / |W0|
    B = sum(w in W0) max(z adopting w) max(0,beta-eligibility(z,w)) / |W0|
    I = sum(w in C) D(w) / sum(v in W1) D(v)

Zero denominators yield zero; empty W0 gives A=B=0. I is algebraically equation
12 when evidence is positive and is zero with no evidence. Strict full-precision
error > threshold triggers replacement. Pure metrics never mutate descriptors.

## Construction

Rank nonexcluded words by decreasing D then ascending UTF-8 token. Admit strength
>= alpha. For each word, choose the existing node with greatest intersection /
current node union >= beta; exact rational ties choose first-created. Union the
word's entire source set into that node; otherwise create its own node. Refuge
contains exactly unmatched sources, including tokenless sources. Remove all level
descriptors from the residual vocabulary. For each node, if at least minimum
sources contain residual tokens, recursively construct one child over that node's
full source scope. Thus one residual dataset gives one child HKT, including under
refuge; no partition into multiple child datasets is scientifically required.
The same function builds baseline and any replacement. Exclusions strictly grow
on ordinary descent; refuge descent strictly reduces the source scope whenever
residual words exist, ensuring termination.

## Evolution and retention

Insert fresh source payloads, freeze descriptors, map incoming evidence, calculate
metrics. On a trigger replace immediately from the full accepted parent scope,
then return without inspecting old descendants. Otherwise persist candidates into
every ordinary node satisfying beta against its full updated membership. Candidate
adoption adds descriptors, not retroactive source assignments. Descendants receive
only the incoming intersection with accepted parent membership. Exclude all current
level descriptors from descendant candidate discovery and reconstruction.

An accepted promotion can overlap a learned descendant descriptor. V2 preserves
that learned descendant until its own metric-triggered reconstruction; it does
not silently rebuild or delete knowledge to enforce a construction-only invariant.
Existing descriptors remain measurable even if newly excluded. This temporal
case is unresolved scientifically; alternatives are forbidding promotion or
immediately restructuring descendants. Neither is justified by Figure 3.

Empty incoming scopes skip inspection. After evolution, remove evidence older
than current-(gamma-1), intersect every membership with retained identities, and
preserve descriptors/topology. Batches are consecutive starting at zero, including
empty updates. Full retention skips this operation. Forgotten identities may return.
Expected words are derived from fresh evidence, never cached mutable state.

## Explicit interpretations and discriminating tests

E01 scope DF versus adopted-membership DF; worst-node beta versus incidence/mean:
shared asymmetric sets. E02 descriptor-only W0 versus expectations: repeated x
expectations. E03 historical beta and persistent adoption versus transient incoming
counts: ax2,a2 then ax2. E05 full-level exclusion versus parent-only: overlapping
b/c contexts under a. E06 residual-bearing >= minimum versus total-parent > or >=:
a3 with only one residual b. E07 lexical ties/first maximum versus hash/ID order:
a, b, ac, bc and z4,b3,a2. These interpretations are selected for V2 research;
author clarification remains desirable. Exact gamma and zero-evidence semantics
are also operational conventions, not claims of uniquely recovered paper intent.
