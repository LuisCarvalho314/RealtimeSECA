use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Rust equivalent of C# `SourceWord` used by the HKT construction logic.
/// One row = (source, word) relation plus cached frequency count in the current scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceWordRecord {
    pub source_word_id: i32,
    pub source_id: i64,
    pub word_id: i32,
    pub word: Option<String>,
    pub word_number_of_sources: usize,
}

/// Rust equivalent of C# `Node`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub node_id: i32,
    pub hkt_id: i32,
    pub word_ids: BTreeSet<i32>,
    pub source_ids: BTreeSet<i64>,
    pub word_source_ids: std::collections::BTreeMap<i32, BTreeSet<i64>>,
    pub source_ids_new_from_batches: BTreeSet<i64>,
    pub word_source_ids_new_from_batches: std::collections::BTreeMap<i32, BTreeSet<i64>>,
    pub word_ids_new_from_batches: BTreeSet<i32>,
    pub top_words: BTreeSet<i32>,
    pub words_for_display: Option<String>,
    pub number_of_sources_for_display: Option<usize>,
}

impl Node {
    pub fn new(node_id: i32, hkt_id: i32) -> Self {
        Self {
            node_id,
            hkt_id,
            word_ids: BTreeSet::new(),
            source_ids: BTreeSet::new(),
            word_source_ids: std::collections::BTreeMap::new(),
            source_ids_new_from_batches: BTreeSet::new(),
            word_source_ids_new_from_batches: std::collections::BTreeMap::new(),
            word_ids_new_from_batches: BTreeSet::new(),
            top_words: BTreeSet::new(),
            words_for_display: None,
            number_of_sources_for_display: None,
        }
    }

    pub fn is_refuge_node(&self) -> bool {
        self.word_ids.contains(&-1)
    }
}

/// Rust equivalent of C# `HKT`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hkt {
    pub hkt_id: i32,
    pub parent_node_id: i32,
    pub expected_words: BTreeSet<i32>,
    pub is_state1: bool,
    pub nodes: Vec<Node>,
}

impl Hkt {
    pub fn new(
        hkt_id: i32,
        parent_node_id: i32,
        expected_words: BTreeSet<i32>,
        is_state1: bool,
    ) -> Self {
        Self {
            hkt_id,
            parent_node_id,
            expected_words,
            is_state1,
            nodes: Vec::new(),
        }
    }
}
