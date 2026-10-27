# Rust v2 construction order

`realtime-seca-v2` constructs each HKT from its current source scope. It counts
each token once per source, admits words whose document frequency divided by the
maximum is at least alpha, and visits admitted words by descending scoped
document frequency. Equal frequencies use ascending **exact UTF-8 token bytes**.
The same order applies to baseline construction and a reconstructed HKT because
both call `recursive::construct`.

Grouping remains greedy. A candidate joins the existing node with the greatest
eligible `|candidate sources ∩ node sources| / |node sources|` score. Each node's
source union is updated before the next candidate. An equal best score keeps the
earliest created node. These choices are explicit implementation rules. The
[CA construction paper](https://doi.org/10.1109/ACCESS.2019.2958702) ranks words
by source frequency, while the [SECA paper](https://doi.org/10.1109/ACCESS.2023.3331219)
uses alpha and beta for admission and grouping; neither specifies how to order
equal-frequency words.

## News G, first cycle

At `ordinary:black / ordinary:friday`, the 18 source documents leave 71
source-word rows after excluding ancestor descriptors. `sale`, `cyber`,
`monday`, `deal`, and `thanksgiving` each have document frequency 4. Rust's
order is `cyber → deal → monday → sale → thanksgiving`. With beta 0.5, `deal`
joins `cyber` at `2/4`; after `monday` joins, `sale` scores `2/6` and starts a
new node. The resulting descriptors are `cyber|deal|monday`, `sale`, and
`thanksgiving`.

The observed original C# execution processed `sale → cyber → monday → deal →
thanksgiving` and produced `cyber|monday|sale`, `deal`, `thanksgiving`. Its child
rows are frequency sorted, but equal-frequency row order comes from a concurrent
dictionary and candidate iteration uses a hash set. Replaying that order with
Rust's grouping calculation reproduced every observed beta score and node
membership. The difference is order, not vocabulary, scope, or beta arithmetic.

The regression fixture is
[`news_g_friday_child.tsv`](../crates/realtime-seca-v2/tests/fixtures/news_g_friday_child.tsv).
[`construction_order.rs`](../crates/realtime-seca-v2/tests/construction_order.rs)
checks the exact descriptors and source memberships with input sources in both
orders. A second test uses U+E000 and U+10000 to pin the UTF-8 byte tie rule.

This change makes the existing Rust lexical tie comparator explicit as a byte
comparison and records its behavior in tests. `String::cmp` already gave the
same ordering for valid UTF-8 strings, so no output change is expected. It does
not add C# runtime compatibility behavior or alter source references.
