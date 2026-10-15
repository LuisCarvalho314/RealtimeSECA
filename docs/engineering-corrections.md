# Ingestion and restoration corrections

Based on origin/main `01a644f`. Scientific calculations, hierarchy construction,
reconstruction, thresholds, core deduplication and forgetting are unchanged.

## CSV identity contract

Conversion chooses the first nonempty trimmed `ArticleId`, `ArticleID`, or
`article_id`, prefixed `article:`. Publishers must provide globally unique article
identifiers across their input corpus. Otherwise an absolute HTTP(S) `URL` is
parsed with the Rust `url` crate and prefixed `url:`. Normalization trims outer
whitespace, uses WHATWG URL serialization (including lowercase scheme/host,
default-port removal, empty-path `/`, and dot-segment resolution), and removes
fragments. Path case, query parameters and their ordering remain significant;
HTTP and HTTPS, `www` aliases, redirects, tracking parameters and percent-encoding
aliases are not equated by an additional policy.

Without a usable URL or article ID, `csv-content:` prefixes a JSON array of
Title, ShortSummary and Description, each with whitespace collapsed. This full
content key is deterministic across files, batches and row order and avoids
hash/delimiter collisions without a hashing dependency. It can be long. Identical
anonymous content is deliberately the same identity; edits change it. Distinct
anonymous articles with identical content cannot be distinguished without an ID.
Timestamp, API collection time, search query and file location do not affect it.

Existing input columns and original provenance metadata remain available, and
recognized article ID columns are now preserved too. Invalid/non-HTTP URLs remain
in metadata and use the content fallback. Explicit IDs take precedence over URLs;
adding/removing an explicit ID changes identity. Old saved `row_` identities are
not migrated; replay original CSVs to rebuild their state with stable identities.
The existing timeline handling of legacy synthetic JSON IDs remains unchanged.
Core deduplication still lasts only while evidence is retained.

## Timestamp contract

`Timestamp`, then `API_Timestamp` if the first is absent or unparsable, is parsed
with chrono's RFC3339 parser into Unix milliseconds. This supports the explicit
zone form of ISO8601: `Z` or numeric `+HH:MM`/`-HH:MM` offsets, including fractional
seconds (submillisecond precision is truncated). Outer whitespace is ignored.
Malformed, missing and timezone-less values produce `None`; no timezone is
invented. No additional legacy date formats are introduced. Raw values remain
in metadata. Chrono and url provide established parsing rather than custom date
or URL implementations; chrono's clock/timezone features are disabled.

## Restoration contract

Restore validates expected words, descriptor words, presentation top words and
word-membership keys against the append-only vocabulary (allowing the existing
`-1` refuge marker). Candidate memberships need not be descriptors. Existing
node/cache equality and source subset checks apply to these references.

Retained batches must be strictly ordered and contiguous, no later than the
snapshot's current index, and within `[current - gamma + 1, current]` with saturation at zero in
SlidingWindow mode. Empty batches count toward this window. Full mode permits
older evidence. Retained external IDs must be nonempty, unique, and agree with
both identity maps and provenance. Corruption is rejected without repair.

## Regression evidence

Before changes, the recovered `342e72c` invariant suite ran **14 passed, 4 failed**:
unknown expected, top and membership words and expired evidence were accepted.
Four new ingestion tests all failed: row-number cross-file collisions,
anonymous reconversion/reordering, article-ID provenance and timestamp parsing.
After changes these regressions pass. Added tests cover converted CSV ingestion
through restart/deduplication, future/duplicate/missing batches, extra provenance, empty
identities, inclusive gamma boundaries, empty updates and Full history. The
recovered suite also checks update failure isolation and restart equivalence
through repeated reconstruction. Fixtures remain small and deterministic.

Final validation on the corrected branch:

- `cargo fmt --all -- --check`: passed.
- `cargo test --workspace`: **168 passed, 0 failed, 2 pre-existing ignored**.
  All five large-dataset regressions and the large sliding-window test completed.
- `cargo clippy --workspace --all-targets -- -D warnings`: passed.

The ignored tests retain known C#/Rust rounding and executed parity differences;
this work does not resolve or change their scientific interpretation. The initial
online dependency-index attempt failed due to sandbox DNS; cached dependencies
allowed offline builds, followed by the exact validation commands above. Earlier
superseded/duplicate test runs were interrupted; the final workspace run completed.
No identified engineering defect remains unresolved within this scope.
