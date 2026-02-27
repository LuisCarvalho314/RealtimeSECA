use super::*;
use std::collections::{BTreeMap, BTreeSet};

impl SecaEngine {
    pub(crate) fn snapshot_hkt_scope(&self, hkt_id: i32) -> Result<HktScopeSnapshot, SecaError> {
        let hkt_build_output =
            self.hkt_build_output
                .as_ref()
                .ok_or_else(|| SecaError::StateError {
                    message: "cannot snapshot HKT scope: baseline tree missing".to_string(),
                })?;

        let hkt =
            hkt_build_output
                .hkts_by_id
                .get(&hkt_id)
                .ok_or_else(|| SecaError::StateError {
                    message: format!("cannot snapshot HKT scope: HKT {} not found", hkt_id),
                })?;

        let mut non_refuge_node_ids = Vec::new();
        let mut refuge_node_id = None;
        let mut node_word_ids_by_node_id: BTreeMap<i32, BTreeSet<i32>> = BTreeMap::new();
        let mut node_source_ids_by_node_id: BTreeMap<i32, BTreeSet<i64>> = BTreeMap::new();
        let mut node_word_source_ids_by_node_id: BTreeMap<i32, BTreeMap<i32, BTreeSet<i64>>> =
            BTreeMap::new();
        let mut node_ids_in_hkt_order: Vec<i32> = Vec::new();

        for node in &hkt.nodes {
            node_ids_in_hkt_order.push(node.node_id);
            if node.is_refuge_node() {
                refuge_node_id = Some(node.node_id);
            } else {
                non_refuge_node_ids.push(node.node_id);
            }

            node_word_ids_by_node_id.insert(node.node_id, node.word_ids.clone());
            node_source_ids_by_node_id.insert(node.node_id, node.source_ids.clone());
            node_word_source_ids_by_node_id.insert(node.node_id, node.word_source_ids.clone());
        }

        non_refuge_node_ids.sort_unstable();

        let current_node_ids: BTreeSet<i32> = hkt.nodes.iter().map(|node| node.node_id).collect();

        let mut child_hkt_ids_by_parent_node_id: BTreeMap<i32, i32> = BTreeMap::new();
        for child_hkt in hkt_build_output.hkts_by_id.values() {
            if child_hkt.parent_node_id != 0 && current_node_ids.contains(&child_hkt.parent_node_id)
            {
                child_hkt_ids_by_parent_node_id.insert(child_hkt.parent_node_id, child_hkt.hkt_id);
            }
        }

        Ok(HktScopeSnapshot {
            hkt_id,
            parent_node_id: hkt.parent_node_id,
            expected_word_ids: hkt.expected_words.clone(),
            non_refuge_node_ids,
            refuge_node_id,
            node_word_ids_by_node_id,
            node_source_ids_by_node_id,
            node_word_source_ids_by_node_id,
            child_hkt_ids_by_parent_node_id,
            node_ids_in_hkt_order,
        })
    }

    pub(crate) fn map_batch_into_hkt_scope(
        &self,
        batch: &SourceBatch,
        scoped_batch_source_indexes: &BTreeSet<usize>,
        scope_snapshot: &HktScopeSnapshot,
    ) -> Result<MappedHktScopeState, SecaError> {
        let mut baseline_word_id_by_token: BTreeMap<&str, i32> = BTreeMap::new();
        for (word_id, token) in &self.baseline_word_legend {
            baseline_word_id_by_token.insert(token.as_str(), *word_id);
        }

        let mut matched_node_source_indexes_by_node_id: BTreeMap<i32, BTreeSet<usize>> =
            BTreeMap::new();

        // Aggregate scope-level diagnostics (existing behavior)
        let mut word_document_frequency_in_scope: BTreeMap<String, usize> = BTreeMap::new();
        let mut known_word_ids_in_scope: BTreeSet<i32> = BTreeSet::new();
        let mut new_tokens_in_scope: BTreeSet<String> = BTreeSet::new();

        // NEW: node-level diagnostics for paper-oriented beta / node analysis
        let mut word_document_frequency_by_node_id: BTreeMap<i32, BTreeMap<String, usize>> =
            BTreeMap::new();
        let mut known_word_ids_by_node_id: BTreeMap<i32, BTreeSet<i32>> = BTreeMap::new();
        let mut new_tokens_by_node_id: BTreeMap<i32, BTreeSet<String>> = BTreeMap::new();

        for source_index in scoped_batch_source_indexes {
            let source = batch
                .sources
                .get(*source_index)
                .ok_or_else(|| SecaError::StateError {
                    message: format!(
                        "batch source index {} out of bounds during scope mapping",
                        source_index
                    ),
                })?;

            let mut unique_tokens_in_source: BTreeSet<&str> = BTreeSet::new();
            for token in &source.tokens {
                let normalized_token = token.trim();
                if normalized_token.is_empty() {
                    continue;
                }
                unique_tokens_in_source.insert(normalized_token);
            }

            // Aggregate scope-level DF/known/new (DF per source)
            for token in &unique_tokens_in_source {
                *word_document_frequency_in_scope
                    .entry((*token).to_string())
                    .or_insert(0) += 1;

                if let Some(word_id) = baseline_word_id_by_token.get(*token) {
                    known_word_ids_in_scope.insert(*word_id);
                } else {
                    new_tokens_in_scope.insert((*token).to_string());
                }
            }

            // Determine node matches (non-refuge nodes only). Multi-node assignment allowed.
            let mut matched_non_refuge_node_ids_for_source: Vec<i32> = Vec::new();

            for node_id in &scope_snapshot.non_refuge_node_ids {
                let Some(node_word_ids) = scope_snapshot.node_word_ids_by_node_id.get(node_id)
                else {
                    continue;
                };

                let mut matches_this_node = false;

                for token in &unique_tokens_in_source {
                    if let Some(word_id) = baseline_word_id_by_token.get(*token) {
                        if node_word_ids.contains(word_id) {
                            matches_this_node = true;
                            break;
                        }
                    }
                }

                if matches_this_node {
                    matched_non_refuge_node_ids_for_source.push(*node_id);
                }
            }

            if !matched_non_refuge_node_ids_for_source.is_empty() {
                matched_non_refuge_node_ids_for_source.sort_unstable();
                matched_non_refuge_node_ids_for_source.dedup();

                for node_id in matched_non_refuge_node_ids_for_source {
                    matched_node_source_indexes_by_node_id
                        .entry(node_id)
                        .or_default()
                        .insert(*source_index);

                    // NEW: node-local DF/known/new (DF per mapped node, per source)
                    let node_df_map = word_document_frequency_by_node_id
                        .entry(node_id)
                        .or_default();
                    let node_known_set = known_word_ids_by_node_id.entry(node_id).or_default();
                    let node_new_set = new_tokens_by_node_id.entry(node_id).or_default();

                    for token in &unique_tokens_in_source {
                        *node_df_map.entry((*token).to_string()).or_insert(0) += 1;

                        if let Some(word_id) = baseline_word_id_by_token.get(*token) {
                            node_known_set.insert(*word_id);
                        } else {
                            node_new_set.insert((*token).to_string());
                        }
                    }
                }
            } else if let Some(refuge_node_id) = scope_snapshot.refuge_node_id {
                // No non-refuge node matched => map to refuge
                matched_node_source_indexes_by_node_id
                    .entry(refuge_node_id)
                    .or_default()
                    .insert(*source_index);

                // NEW: node-local DF/known/new for refuge
                let node_df_map = word_document_frequency_by_node_id
                    .entry(refuge_node_id)
                    .or_default();
                let node_known_set = known_word_ids_by_node_id.entry(refuge_node_id).or_default();
                let node_new_set = new_tokens_by_node_id.entry(refuge_node_id).or_default();

                for token in &unique_tokens_in_source {
                    *node_df_map.entry((*token).to_string()).or_insert(0) += 1;

                    if let Some(word_id) = baseline_word_id_by_token.get(*token) {
                        node_known_set.insert(*word_id);
                    } else {
                        node_new_set.insert((*token).to_string());
                    }
                }
            }
        }

        Ok(MappedHktScopeState {
            hkt_id: scope_snapshot.hkt_id,
            scoped_batch_source_indexes: scoped_batch_source_indexes.clone(),
            matched_node_source_indexes_by_node_id,
            word_document_frequency_in_scope,
            known_word_ids_in_scope,
            new_tokens_in_scope,

            // NEW fields
            word_document_frequency_by_node_id,
            known_word_ids_by_node_id,
            new_tokens_by_node_id,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct HktScopeSnapshot {
    pub(crate) hkt_id: i32,
    pub(crate) parent_node_id: i32,
    pub(crate) expected_word_ids: BTreeSet<i32>,
    pub(crate) non_refuge_node_ids: Vec<i32>,
    pub(crate) refuge_node_id: Option<i32>,
    pub(crate) node_word_ids_by_node_id: BTreeMap<i32, BTreeSet<i32>>,
    pub(crate) node_source_ids_by_node_id: BTreeMap<i32, BTreeSet<i64>>,
    pub(crate) node_word_source_ids_by_node_id: BTreeMap<i32, BTreeMap<i32, BTreeSet<i64>>>,
    pub(crate) child_hkt_ids_by_parent_node_id: BTreeMap<i32, i32>,
    pub(crate) node_ids_in_hkt_order: Vec<i32>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct MappedHktScopeState {
    #[allow(dead_code)]
    pub(crate) hkt_id: i32,
    pub(crate) scoped_batch_source_indexes: BTreeSet<usize>,
    pub(crate) matched_node_source_indexes_by_node_id: BTreeMap<i32, BTreeSet<usize>>,

    pub(crate) word_document_frequency_in_scope: BTreeMap<String, usize>,
    pub(crate) known_word_ids_in_scope: BTreeSet<i32>,
    pub(crate) new_tokens_in_scope: BTreeSet<String>,

    // NEW: node-level mapped state1 diagnostics (beta-ready data path)
    pub(crate) word_document_frequency_by_node_id: BTreeMap<i32, BTreeMap<String, usize>>,
    pub(crate) known_word_ids_by_node_id: BTreeMap<i32, BTreeSet<i32>>,
    pub(crate) new_tokens_by_node_id: BTreeMap<i32, BTreeSet<String>>,
}
