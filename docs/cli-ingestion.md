# CSV ingestion contract

`realtime-seca-cli from-csv INPUT.csv OUTPUT.json --batch-index N` converts
article rows into a saved `SourceBatch`. Identity and timestamp parsing belong
to ingestion and do not change SECA update or forgetting rules.

A nonempty `URL`, after trimming surrounding whitespace, produces the identity
`url:<trimmed URL>`. The URL column is taken as the publisher's article identifier;
it is not validated, canonicalized, lowercased, or resolved. Different trimmed
URLs have distinct identities, and repeated URLs share an identity even if their
text differs. The engine retains the first occurrence within a batch and ignores
identities already retained in persistent state. Forgotten identities may be
admitted again; this is not lifetime deduplication.

An absent, empty, or whitespace-only URL produces
`csv-row:<conversion UUID>:<batch index>:<one-based CSV record number>`.
Each conversion generates a fresh random UUID, avoiding row-position collisions
between independently converted files, including files using the same batch
index. Anonymous rows in one conversion remain distinct even when their text is
identical. Reconverting the same anonymous article creates a new identity and
therefore does not deduplicate it across conversions. Save and replay the
converted JSON to preserve identities and deterministic engine replay.

Timestamps accept RFC3339 with `Z` or an explicit numeric offset (for example,
`1970-01-01T01:00:00+01:00`) and the space-separated form
`YYYY-MM-DD HH:MM:SS[.fraction]+HH:MM` (also allowing negative offsets).
Surrounding whitespace is ignored for parsing. Unsupported, malformed, empty,
and timezone-free values produce no parsed timestamp; no local timezone is
assumed. The unknown-offset marker `-00:00` is also rejected. The first
successfully parsed value from `Timestamp`, then
`API_Timestamp`, supplies Unix milliseconds. Submillisecond precision is reduced
to milliseconds. Original nonempty timestamp fields remain unchanged in metadata,
including values that fail parsing. Timestamps do not control gamma forgetting.

The CLI uses `chrono` for calendar/offset validation and Unix conversion, and
`uuid` v4 for conversion scopes, avoiding custom date arithmetic or a persistent
identity database.
