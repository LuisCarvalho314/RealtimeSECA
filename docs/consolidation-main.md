# Main consolidation review

Starting baseline: `origin/integration/seca-paper-benchmark` at `459117c`.
The detached consolidation starts there; origin/main is already its ancestor.

## Reviewed branches

- `fix/seca-ingestion-state` (`be042f7`): recovered 24 persistence and lifecycle
  regressions. Incorporated nonempty restored identities and consecutive retained
  batches, matching existing input validation. Retained baseline vocabulary
  validation, CSV conversion namespaces, explicit-offset timestamp handling and
  last-batch envelope check. Rejected content-based anonymous deduplication and
  URL rewriting because they replace the established ingestion contract; identical
  anonymous text does not establish article identity. Duplicate checks were omitted.
- `audit/seca-scientific-decisions` (`c59babe`): retained proposals as documentation,
  linked from pending decisions. No proposal is approved or implemented.
- `experiments/seca-benchmark-repro` (`04a5ac3`): retained external Full-memory
  hierarchy characterization and recorded historical evidence. Its adapter calls
  the canonical engine; it adds no scientific implementation. Clarified population
  and scope differences from canonical Theme G/H evaluation.

`benchmarks/seca-paper` is byte-for-byte unchanged from `459117c`. Production
changes are confined to the two restore-time rejection checks. The 491-line
invariant test file remains cohesive: shared state assertions and corruption cases;
no new production abstraction or module boundary is introduced.

## Branch disposition

The following branch tips are ancestors of consolidated main and can be deleted
once any attached worktree is clean and no longer needed:

- `integration/seca-paper-benchmark`, `integration/seca-correctness`
- `fix/seca-ingestion-state`, `audit/seca-scientific-decisions`
- `experiments/seca-benchmark-repro`, `fix/seca-light-stream`
- local `audit/consolidated-2026-10`, `audit/context-metrics`,
  `test/seca-light-state-invariants`
- remote `integration/executable-parity-summary`

No branch or worktree was deleted. Other branch tips are not ancestors and must
be retained pending separate review, including `experiments/seca-reference-benchmark`,
clustering/datasets/evaluation branches, research branches, older fixes and parity
agents. `origin/master` has unique non-patch-equivalent commit `4c58fbf`; retain it.
The original research checkout and its uncommitted files were left untouched.

## Scientific limitations

The existing two ignored strict cross-language parity assertions fail when run
explicitly (threshold precision and HKT boundary/topology). The external reference
comparison also retains its C# shared-candidate failure and excludes empty input.
None is a new integration regression or authorization to change scientific semantics.
The pending scientific decisions and original dataset/protocol uncertainties remain.
