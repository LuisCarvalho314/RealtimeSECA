# Running the competing computational experiments

Run from the repository root. Python 3, Cargo with the existing dependency cache,
and Linux `/usr/bin/time` are required. Mono and the existing reference binaries
are needed only for reference replay. Nothing downloads dependencies or edits
production/reference code. The prototypes use Python's standard library only;
the separate Rust package is a read-only measurement adapter.

```sh
cargo build --offline --release --manifest-path docs/architecture-exploration/prototypes/rust-baseline/Cargo.toml
python3 -m unittest discover -s docs/architecture-exploration/prototypes/direct -v
python3 -m unittest discover -s docs/architecture-exploration/prototypes/bitset -v
python3 docs/architecture-exploration/prototypes/evaluate.py > /tmp/seca-comparison.json
python3 docs/architecture-exploration/prototypes/evaluate_metrics.py > /tmp/seca-metrics.json
python3 docs/architecture-exploration/prototypes/direct/evaluate.py
```

The shared evaluator compares 43 existing fixture baselines, 100 seed-711 random
cases and five news/controlled workloads to the current live Rust builder. It
emits individual correctness results, three-trial median timing, peak RSS and
separate peak traced Python allocations. It returns data rather than silently
tolerating mismatches: use this assertion when checking a reproduced result.

```sh
python3 - <<'PY'
import json
d = json.load(open('/tmp/seca-comparison.json'))
assert all(c['direct'] and c['bitset'] for c in d['checks'])
assert all(w[n]['matches_rust'] for w in d['measurements'] for n in ('direct', 'bitset'))
PY
```

Optional: compile the same adapter against the already available simplified
checkout without modifying that checkout. Replace the path on another machine.
Its tested revision was `d7e6e657b6acdc354a2a4bdfe2a672510bf555cb`.

```sh
python3 docs/architecture-exploration/prototypes/prepare_comparison.py /home/azureuser/secalight-simplify /tmp/seca-exploration-simplified
SECA_SIMPLIFIED_BINARY=/tmp/seca-exploration-simplified/target/release/seca-exploration-baseline python3 docs/architecture-exploration/prototypes/evaluate.py > /tmp/seca-comparison-all.json
```

The common numeric schema is ordered rows `{"source": integer, "word": positive
integer}` with alpha, beta and exclusive branch minimum. Duplicate incidence is
set-valued. Both Python implementations expose direct callable APIs documented in
their own READMEs. Root construction, child scoped counting, greedy grouping and
refuge recursion are independent implementations; neither calls production code.

**Not covered:** complete evolution, contextual candidate admission, reconstruction
controller, source hashing, stable topology IDs, top-word exports, forgetting,
snapshot restoration, final viewer observations or transaction rollback. Metric
experiments are pure kernels, not an implementation of the missing update stages.
The construction adapter deliberately omits those same features. See
[comparison.md](../comparison.md) for evidence boundaries and next tests.

Reference validation, without recapturing fixtures:

```sh
python3 tools/seca-reference-harness/verify.py
python3 docs/architecture-exploration/prototypes/replay_reference.py > /tmp/seca-reference-replay.json
python3 crates/realtime-seca-core/tests/fixtures/parity_context_metrics/replay.py
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

Do not run `reference.py` to verify construction: its capture entry point writes
fixtures. `replay_reference.py` invokes `run.sh` with each existing input,
parsed stdout, and compared it to the existing `csharp.json`, writing only
[reference_replay.json](../reference_replay.json). The locked reference remains
independent of the Rust target and of either prototype.
