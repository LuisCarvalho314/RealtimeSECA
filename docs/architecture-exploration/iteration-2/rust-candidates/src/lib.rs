//! Research-only ordered contextual construction. No production algorithm calls.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub type Sources = BTreeSet<i64>;
pub mod evolution;
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Row {
    pub source: i64,
    pub word: i32,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Input {
    pub records: Vec<Row>,
    pub alpha: f64,
    pub beta: f64,
    #[serde(rename = "branchMinimum")]
    pub branch_minimum: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Scope {
    pub hkt_id: i32,
    pub parent_node_id: i32,
    pub expected_words: BTreeSet<i32>,
    pub nodes: Vec<Node>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Node {
    pub node_id: i32,
    pub hkt_id: i32,
    pub words: BTreeSet<i32>,
    pub sources: Sources,
    pub word_sources: BTreeMap<i32, Sources>,
    pub top_words: BTreeSet<i32>,
    pub child: Option<Box<Scope>>,
}

/// A word incidence and the immutable input position establishing equal-DF order.
#[derive(Clone)]
struct Incidence {
    word: i32,
    sources: Sources,
    first: usize,
    positions: BTreeMap<i64, usize>,
}

fn relation(rows: &[Row]) -> Vec<Incidence> {
    let mut by_word: BTreeMap<i32, Incidence> = BTreeMap::new();
    for (first, row) in rows.iter().enumerate() {
        let incidence = by_word.entry(row.word).or_insert_with(|| Incidence {
            word: row.word,
            sources: Sources::new(),
            first,
            positions: BTreeMap::new(),
        });
        incidence.sources.insert(row.source);
        incidence.positions.entry(row.source).or_insert(first);
    }
    by_word.into_values().collect()
}

#[derive(Default)]
struct Ids {
    hkt: i32,
    node: i32,
}

/// Validate normalized construction inputs before allocating any hierarchy.
pub fn build(input: &Input, masks: bool) -> Result<Option<Scope>, String> {
    if !(0.0..=1.0).contains(&input.alpha) || !(0.0..=1.0).contains(&input.beta) {
        return Err("alpha and beta must be finite values in [0,1]".into());
    }
    if input.records.iter().any(|r| r.word <= 0) {
        return Err("normalized word IDs must be positive".into());
    }
    let words = relation(&input.records);
    let mut ids = Ids::default();
    Ok(construct(&words, 0, input, masks, &mut ids))
}

fn new_node(word: i32, sources: Sources, hkt: i32, ids: &mut Ids) -> Node {
    ids.node += 1;
    let word_sources = if word == -1 {
        BTreeMap::new()
    } else {
        BTreeMap::from([(word, sources.clone())])
    };
    Node {
        node_id: ids.node,
        hkt_id: hkt,
        words: BTreeSet::from([word]),
        sources,
        word_sources,
        top_words: BTreeSet::new(),
        child: None,
    }
}

fn construct(
    words: &[Incidence],
    parent: i32,
    config: &Input,
    masks: bool,
    ids: &mut Ids,
) -> Option<Scope> {
    let mut ranked: Vec<_> = words.iter().collect();
    ranked.sort_by_key(|w| (std::cmp::Reverse(w.sources.len()), w.first));
    let seed = ranked.first()?;
    let maximum = seed.sources.len();
    let mut expected: BTreeSet<_> = words
        .iter()
        .filter(|w| w.sources.len() as f64 / maximum as f64 >= config.alpha)
        .map(|w| w.word)
        .collect();
    expected.remove(&seed.word);
    ids.hkt += 1;
    let hkt_id = ids.hkt;
    let mut nodes = vec![new_node(seed.word, seed.sources.clone(), hkt_id, ids)];
    let mut algebra = masks.then(|| ScopeMasks::new(words, seed.word));
    // Every membership union is visible to the following word: ordering is essential.
    for word in &expected {
        let incoming = &words[words.binary_search_by_key(word, |w| w.word).unwrap()].sources;
        let best = algebra.as_ref().map_or_else(
            || set_best(&nodes, incoming, config.beta),
            |m| m.best(*word, config.beta),
        );
        if let Some(algebra) = &mut algebra {
            algebra.adopt(*word, best);
        }
        match best {
            Some(index) => {
                let node = &mut nodes[index];
                node.words.insert(*word);
                node.sources.extend(incoming);
                node.word_sources.insert(*word, incoming.clone());
            }
            None => nodes.push(new_node(*word, incoming.clone(), hkt_id, ids)),
        }
    }
    let covered: Sources = nodes
        .iter()
        .flat_map(|n| n.sources.iter().copied())
        .collect();
    let all: Sources = words
        .iter()
        .flat_map(|w| w.sources.iter().copied())
        .collect();
    let refuge: Sources = all.difference(&covered).copied().collect();
    if !refuge.is_empty() {
        nodes.push(new_node(-1, refuge, hkt_id, ids));
    }
    for node in &mut nodes {
        refine_node(node, words, config, masks, ids);
    }
    Some(Scope {
        hkt_id,
        parent_node_id: parent,
        expected_words: expected,
        nodes,
    })
}

fn refine_node(node: &mut Node, words: &[Incidence], config: &Input, masks: bool, ids: &mut Ids) {
    if node.sources.len() <= config.branch_minimum {
        return;
    }
    // The parent scope stays immutable; exclude only this attachment's words.
    let child_words: Vec<_> = words
        .iter()
        .filter(|w| !node.words.contains(&w.word))
        .filter_map(|w| {
            let sources: Sources = w.sources.intersection(&node.sources).copied().collect();
            let positions: BTreeMap<_, _> = sources.iter().map(|s| (*s, w.positions[s])).collect();
            let first = positions.values().copied().min().unwrap_or(usize::MAX);
            (!sources.is_empty()).then_some(Incidence {
                word: w.word,
                sources,
                first,
                positions,
            })
        })
        .collect();
    if child_words.is_empty() {
        return;
    }
    if !node.words.contains(&-1) {
        node.top_words = node.words.clone();
        let mut top: Vec<_> = child_words.iter().collect();
        top.sort_by_key(|w| (std::cmp::Reverse(w.sources.len()), w.first));
        for w in top {
            if node.top_words.len() >= 10 {
                break;
            }
            node.top_words.insert(w.word);
        }
    }
    node.child = construct(&child_words, node.node_id, config, masks, ids).map(Box::new);
}

fn set_best(nodes: &[Node], incoming: &Sources, beta: f64) -> Option<usize> {
    let mut best = None;
    let mut maximum = -1.0;
    for (index, node) in nodes.iter().enumerate() {
        let score = node.sources.intersection(incoming).count() as f64 / node.sources.len() as f64;
        if score >= beta && score > maximum {
            best = Some(index);
            maximum = score;
        }
    }
    best
}

/// Scope-local incidence kernel. Stable source IDs are indexed once per scope;
/// word columns and growing node masks remain dense until that scope is complete.
struct ScopeMasks {
    columns: BTreeMap<i32, Vec<u64>>,
    nodes: Vec<Vec<u64>>,
}
impl ScopeMasks {
    fn new(words: &[Incidence], seed: i32) -> Self {
        let universe: Sources = words
            .iter()
            .flat_map(|w| w.sources.iter().copied())
            .collect();
        let positions: BTreeMap<_, _> = universe
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();
        let columns: BTreeMap<_, _> = words
            .iter()
            .map(|word| {
                let mut bits = vec![0u64; positions.len().div_ceil(64)];
                for source in &word.sources {
                    let i = positions[source];
                    bits[i / 64] |= 1 << (i % 64);
                }
                (word.word, bits)
            })
            .collect();
        let nodes = vec![columns[&seed].clone()];
        Self { columns, nodes }
    }
    fn best(&self, word: i32, beta: f64) -> Option<usize> {
        let incoming = &self.columns[&word];
        let mut best = None;
        let mut maximum = -1.0;
        for (index, node) in self.nodes.iter().enumerate() {
            let count: usize = node
                .iter()
                .zip(incoming)
                .map(|(a, b)| (a & b).count_ones() as usize)
                .sum();
            let size: usize = node.iter().map(|a| a.count_ones() as usize).sum();
            let score = count as f64 / size as f64;
            if score >= beta && score > maximum {
                best = Some(index);
                maximum = score;
            }
        }
        best
    }
    fn adopt(&mut self, word: i32, node: Option<usize>) {
        match node {
            Some(index) => {
                for (a, b) in self.nodes[index].iter_mut().zip(&self.columns[&word]) {
                    *a |= b;
                }
            }
            None => self.nodes.push(self.columns[&word].clone()),
        }
    }
}

pub fn structural(scope: &Option<Scope>) -> serde_json::Value {
    match scope {
        None => serde_json::json!([]),
        Some(scope) => serde_json::Value::Array(
            scope
                .nodes
                .iter()
                .map(|n| {
                    serde_json::json!({"words":n.words,"sources":n.sources,
            "word_sources":n.word_sources,"child":structural(&n.child.as_deref().cloned())})
                })
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn case(rows: &[(i64, i32)]) -> Input {
        Input {
            records: rows
                .iter()
                .map(|(source, word)| Row {
                    source: *source,
                    word: *word,
                })
                .collect(),
            alpha: 0.0,
            beta: 0.5,
            branch_minimum: 1,
        }
    }
    #[test]
    fn mask_and_set_preserve_every_field() {
        for seed in 0..100 {
            let rows: Vec<_> = (1..9)
                .flat_map(|s| {
                    (1..7).filter_map(move |w| {
                        ((s * 31 + w * 17 + seed * 13) % 7 < 3).then_some((s, w as i32))
                    })
                })
                .collect();
            assert_eq!(build(&case(&rows), false), build(&case(&rows), true));
        }
    }
    #[test]
    fn seed_ties_follow_input_not_word_id() {
        let scope = build(&case(&[(1, 9), (2, 1)]), false).unwrap().unwrap();
        assert_eq!(scope.nodes[0].words, BTreeSet::from([9]));
        assert_eq!(scope.expected_words, BTreeSet::from([1]));
    }
    #[test]
    fn child_seed_uses_first_surviving_incidence() {
        let mut input = case(&[(1, 2), (2, 1), (2, 3), (3, 1), (4, 1), (4, 2)]);
        input.alpha = 0.8;
        let tree = build(&input, false).unwrap().unwrap();
        let child = tree.nodes[0].child.as_ref().unwrap();
        assert_eq!(child.nodes[0].words, BTreeSet::from([3]));
        assert_eq!(Some(tree), build(&input, true).unwrap());
    }
    #[test]
    fn duplicate_incidence_is_set_valued_and_input_unchanged() {
        let input = case(&[(1, 1), (2, 1), (2, 2)]);
        let before = serde_json::to_value(&input).unwrap();
        let mut duplicate = input.clone();
        duplicate.records.push(Row { source: 2, word: 2 });
        assert_eq!(build(&input, false), build(&duplicate, false));
        assert_eq!(before, serde_json::to_value(&input).unwrap());
    }
    #[test]
    fn empty_and_invalid_are_explicit() {
        assert!(build(&case(&[]), false).unwrap().is_none());
        assert!(build(&case(&[(1, -1)]), false).is_err());
    }
}
