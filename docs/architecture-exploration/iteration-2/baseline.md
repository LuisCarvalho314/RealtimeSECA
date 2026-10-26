# Iteration 2 baseline

Investigation date: 2026-10-26. Production comparison is exactly remote `main`
`e0ca91eb62e1e5843ddb47d98ec71a388a67f6d5`, verified by a successful `git fetch origin`
from `LuisCarvalho314/RealtimeSECA`. The local branch named `main` is older
(`aba1be3`); it was deliberately not used. Original research HEAD is
`dffae7b88a7218ab778056c13e42b97bc9c5facb`. Merge base is
`2bcd8edf045b38c1b02437f7672178b42991ac07`; `git rev-list --left-right --count
origin/main...HEAD` reports 33 production-only and 11 research-only commits.

## Isolation and preservation

The original checkout remains on `research/seca-hkt-construction`. Existing tracked
modifications affect engine validation, invariant tests and algorithm/architecture/state
documents. Existing untracked material includes `.worktrees/`, scientific-context
fixtures/tests, context research, `references/` and `tools/`. None was discarded,
committed, rebased or merged. An initial tracked binary patch and status inventory
were saved outside the checkout, with copies in `raw/`.

An isolated local clone at `/tmp/seca-iteration-2`, branch
`research/seca-architecture-iteration-2`, is based on the exact production SHA.
A clone was used because this checkout's Git metadata lives outside its writable
root. Only `docs/architecture-exploration/` was selectively imported using
`git archive dffae7b ... | tar -x`; no historical production code was imported.
Original untracked C# reference/harness material is inspected read-only at its
existing location. The new deliverable directory can be copied into the original
research checkout without changing production files. No remote writes or history
rewrites are part of this investigation.

Reproduce the source baseline with:

```sh
git clone git@github.com:LuisCarvalho314/RealtimeSECA.git seca-iteration-2
cd seca-iteration-2
git switch --detach e0ca91eb62e1e5843ddb47d98ec71a388a67f6d5
git archive dffae7b88a7218ab778056c13e42b97bc9c5facb docs/architecture-exploration | tar -x
```

Then copy this iteration's research directory into that checkout. Standalone
research packages declare their own workspaces and do not modify production's
workspace membership or dependencies.

## Differences from the old experimental baseline

This is not an eleven-commit merge. Current main includes executable C# parity
work, scientific fixtures and pending-decision reports, ingestion and restore
corrections, paper benchmark datasets/evidence, locked public behaviour and
continuation oracles, and the validated SECA-Light simplification. These are
absent or older on the research branch. The older research working tree also
contains uncommitted corrections, so its previous test counts cannot characterize
current main.

In the construction implementation, current main removes build-wide input-scope
archives, intermediate node mirrors, positional row maps, recursion-only HKT
clones, and redundant intermediate synchronization. It preserves original
recursive input scopes, stable row tie order, exclusive branch cutoff, and
ascending candidate-word IDs. The builder falls from 530 to 436 physical lines;
core production source from 3504 in the earlier working-tree report to 3253 in
the current simplification report. New measurements count files directly rather
than assuming these historical counts.

The update path now uses one ancestry exclusion set and one ordered target map,
moves pending maps instead of copying them, derives replacement caches after ID
remapping, and reuses input frequencies. The apparent removal of ancestor
accepted/rejected count maps preserves reachable behaviour: the accepted/rejected
payload paths were unused, as documented in the simplification audit. Metrics
retain the same scientific choices; their diff removes a redundant clone only.
No parity with an older prototype is a proof of paper correctness.

## Validation before experiments

Toolchain: Rust 1.92.0 (`ded5c06cf`), Python 3.11.4, Linux x86_64. Workspace
validation runs against the isolated production tree, not the dirty research tree.
Commands and full captured outputs are retained under `raw/` at completion.

| Check | Result |
|---|---|
| `cargo fmt --all -- --check` | Passed |
| `cargo clippy --workspace --all-targets -- -D warnings` | Passed |
| `cargo test --workspace` | Passed: 191 passed, zero failed, 2 intentional strict parity skips |
| Prior direct Python unittest discovery | 7 passed |
| Prior bitset Python unittest discovery | 7 passed |
| C# harness `verify.py` | Passed manual metrics/topology, later cycles, repeatability, expected invalid-input failure |
| Read-only C# construction replay | 21/21 captured executions unchanged |
| Context fixture `replay.py` without capture flag | Passed, including expected empty-input and duplicate-key failures |

The full production suite was launched before new implementation work. A research
agent wrote the initial construction source while tests were in progress, before
receiving the baseline gate message; execution and further implementation were
held pending baseline completion. This sequencing exception did not touch the
production test tree and is recorded rather than claiming strict phase ordering.

Existing intentional strict C# parity skips must remain skips, not successful
legacy equivalence. New candidate tests and production lifecycle tests are reported
separately. The baseline does not establish scientific correctness.

The complete suite finished successfully before candidate differential execution.
Large-data tests took225.78s and scientific-context tests70.61s; these are
validation durations, not benchmark observations. The 191-pass total independently
reproduces current production validation rather than importing the old research
count. Full output, summary JSON and read-only reference replay logs are retained
in `raw/`. Reference binaries and source fixtures were not rebuilt or recaptured.
