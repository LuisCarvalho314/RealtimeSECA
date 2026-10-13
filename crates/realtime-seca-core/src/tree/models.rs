use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Source-word evidence used by HKT construction.
/// One row = (source, word) relation plus cached frequency count in the current scope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceWordRecord {
    pub source_word_id: i32,
    pub source_id: i64,
    pub word_id: i32,
    pub word: Option<String>,
    pub word_number_of_sources: usize,
}

/// Topic words and their contextual source memberships.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Node {
    pub node_id: i32,
    pub hkt_id: i32,
    pub word_ids: BTreeSet<i32>,
    pub source_ids: BTreeSet<i64>,
    pub word_source_ids: std::collections::BTreeMap<i32, BTreeSet<i64>>,
    pub source_ids_new_from_batches: BTreeSet<i64>,
    pub word_source_ids_new_from_batches: std::collections::BTreeMap<i32, BTreeSet<i64>>,
    pub top_words: BTreeSet<i32>,
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
            top_words: BTreeSet::new(),
        }
    }

    pub fn is_refuge_node(&self) -> bool {
        self.word_ids.contains(&-1)
    }
}

/// An HKT container attached to one parent node (zero denotes Seed-HKT).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
