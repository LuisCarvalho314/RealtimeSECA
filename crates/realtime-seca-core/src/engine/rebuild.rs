use super::*;
use crate::tree::{Hkt, Node};
use std::collections::BTreeSet;

impl SecaEngine {
    pub(crate) fn propagate_sources_to_parent_chain(
        &mut self,
        start_parent_node_id: i32,
        source_ids: &BTreeSet<i64>,
    ) -> Result<(), SecaError> {
        let tree = self
            .hkt_build_output
            .as_mut()
            .ok_or_else(|| SecaError::StateError {
                message: "cannot propagate sources: baseline tree missing".into(),
            })?;
        let mut current = start_parent_node_id;
        while current != 0 {
            let node = tree
                .nodes_by_id
                .get_mut(&current)
                .ok_or_else(|| SecaError::StateError {
                    message: format!("missing ancestor node {current}"),
                })?;
            node.source_ids.extend(source_ids);
            let owner =
                tree.hkts_by_id
                    .get_mut(&node.hkt_id)
                    .ok_or_else(|| SecaError::StateError {
                        message: format!("missing owner for ancestor node {current}"),
                    })?;
            let owned_node = owner
                .nodes
                .iter_mut()
                .find(|n| n.node_id == current)
                .ok_or_else(|| SecaError::StateError {
                    message: format!("ancestor node {current} absent from its HKT"),
                })?;
            *owned_node = node.clone();
            current = owner.parent_node_id;
        }
        Ok(())
    }

    pub(crate) fn rebuild_selected_hkts_from_trigger_plan(
        &mut self,
        batch: &SourceBatch,
        trigger_plan: &crate::engine::trigger::RecursiveTriggerPlan,
    ) -> Result<BTreeMap<i32, i32>, SecaError> {
        let mut replacements = BTreeMap::new();
        for target_hkt_id in trigger_plan.targets.keys() {
            let target_hkt_id = *target_hkt_id;

            // Get parent_node_id with immutable access
            let parent_node_id = {
                let hkt_build_output =
                    self.hkt_build_output
                        .as_ref()
                        .ok_or_else(|| SecaError::StateError {
                            message: "cannot rebuild selected HKTs: baseline tree missing"
                                .to_string(),
                        })?;

                hkt_build_output
                    .hkts_by_id
                    .get(&target_hkt_id)
                    .map(|h| h.parent_node_id)
                    .ok_or_else(|| SecaError::StateError {
                        message: format!("cannot rebuild: HKT {} not found", target_hkt_id),
                    })?
            };

            let scoped_records =
                build_scoped_source_word_records_for_hkt(self, batch, trigger_plan, target_hkt_id)?;

            if scoped_records.is_empty() {
                continue;
            }

            let state1_source_ids: BTreeSet<i64> =
                scoped_records.iter().map(|r| r.source_id).collect();

            if parent_node_id != 0 {
                self.propagate_sources_to_parent_chain(parent_node_id, &state1_source_ids)?;
            }

            let hkt_builder = HktBuilder::new(
                self.config
                    .hkt_builder
                    .minimum_threshold_against_max_word_count,
                self.config.hkt_builder.similarity_threshold,
                self.config
                    .hkt_builder
                    .minimum_number_of_sources_to_create_branch_for_node,
            )?;

            let subtree = hkt_builder.build_full_tree(scoped_records, true)?;

            let hkt_build_output =
                self.hkt_build_output
                    .as_mut()
                    .ok_or_else(|| SecaError::StateError {
                        message: "cannot rebuild selected HKTs: baseline tree missing".to_string(),
                    })?;

            let hkt_id_offset = self.next_hkt_id - 1;
            let node_id_offset = self.next_node_id - 1;

            remove_subtree(hkt_build_output, target_hkt_id);

            let rebuilt_root = subtree
                .hkts_by_id
                .values()
                .find(|h| h.parent_node_id == 0)
                .expect("rebuilt subtree root")
                .hkt_id;
            replacements.insert(target_hkt_id, rebuilt_root + hkt_id_offset);
            let remapped =
                remap_subtree_ids(subtree, hkt_id_offset, node_id_offset, parent_node_id);
            let remapped_hkt_len = remapped.hkts_by_id.len() as i32;
            let remapped_node_len = remapped.nodes_by_id.len() as i32;
            hkt_build_output.hkts_by_id.extend(remapped.hkts_by_id);
            hkt_build_output.nodes_by_id.extend(remapped.nodes_by_id);

            self.next_hkt_id += remapped_hkt_len;
            self.next_node_id += remapped_node_len;
        }

        Ok(replacements)
    }
}

pub(crate) fn build_scoped_source_word_records_for_hkt(
    engine: &SecaEngine,
    current_batch: &SourceBatch,
    trigger_plan: &crate::engine::trigger::RecursiveTriggerPlan,
    target_hkt_id: i32,
) -> Result<Vec<SourceWordRecord>, SecaError> {
    let hkt_build_output =
        engine
            .hkt_build_output
            .as_ref()
            .ok_or_else(|| SecaError::StateError {
                message: "cannot build scoped dataset: baseline tree missing".to_string(),
            })?;

    let hkt = hkt_build_output
        .hkts_by_id
        .get(&target_hkt_id)
        .ok_or_else(|| SecaError::StateError {
            message: format!(
                "cannot build scoped dataset: HKT {} not found",
                target_hkt_id
            ),
        })?;

    // State0 sources: union of node.source_ids in the HKT
    let mut state0_source_ids: BTreeSet<i64> = BTreeSet::new();
    for node in &hkt.nodes {
        state0_source_ids.extend(node.source_ids.iter().copied());
    }

    // State1 sources: from trigger plan scoped indexes (current batch only)
    let target = trigger_plan
        .targets
        .get(&target_hkt_id)
        .cloned()
        .unwrap_or_default();

    // token -> word_id
    let mut baseline_word_id_by_token: BTreeMap<&str, i32> = BTreeMap::new();
    for (word_id, token) in &engine.baseline_word_legend {
        baseline_word_id_by_token.insert(token.as_str(), *word_id);
    }

    let mut records: Vec<SourceWordRecord> = Vec::new();
    let mut source_word_id_counter: i32 = 1;
    let mut seen_pairs: BTreeSet<(i64, i32)> = BTreeSet::new();

    let mut synthetic_word_id_counter: i32 = i32::MAX;
    let mut synthetic_word_id_by_token: BTreeMap<String, i32> = BTreeMap::new();

    let mut push_source = |source_id_str: &str, tokens: &[String]| {
        let internal_source_id = engine
            .source_id_by_url
            .get(source_id_str)
            .copied()
            .unwrap_or_else(|| SecaEngine::fnv1a_64(source_id_str));

        for token in tokens {
            let normalized = token.trim();
            if normalized.is_empty() {
                continue;
            }

            let word_id = if let Some(word_id) = baseline_word_id_by_token.get(normalized) {
                if target.excluded_words.contains(word_id) {
                    continue;
                }
                *word_id
            } else {
                *synthetic_word_id_by_token
                    .entry(normalized.to_string())
                    .or_insert_with(|| {
                        let id = synthetic_word_id_counter;
                        synthetic_word_id_counter -= 1;
                        id
                    })
            };

            if !seen_pairs.insert((internal_source_id, word_id)) {
                continue;
            }

            records.push(SourceWordRecord {
                source_word_id: source_word_id_counter,
                source_id: internal_source_id,
                word_id,
                word: Some(normalized.to_string()),
                word_number_of_sources: 0,
            });
            source_word_id_counter += 1;
        }
    };

    // State0: use stored batches to pull tokens for sources in this HKT
    for batch in &engine.processed_batches {
        for source in &batch.sources {
            let internal_source_id = engine
                .source_id_by_url
                .get(source.source_id.as_str())
                .copied()
                .unwrap_or_else(|| SecaEngine::fnv1a_64(source.source_id.as_str()));

            if state0_source_ids.contains(&internal_source_id) {
                push_source(&source.source_id, &source.tokens);
            }
        }
    }

    // State1: scoped sources from current batch
    for idx in target.source_indexes {
        if let Some(source) = current_batch.sources.get(idx) {
            push_source(&source.source_id, &source.tokens);
        }
    }

    crate::tree::hkt_builder::update_document_frequencies(&mut records);

    Ok(records)
}

fn collect_subtree_hkt_and_node_ids(
    hkt_build_output: &HktBuildOutput,
    root_hkt_id: i32,
) -> (BTreeSet<i32>, BTreeSet<i32>) {
    let mut hkt_ids: BTreeSet<i32> = BTreeSet::new();
    let mut node_ids: BTreeSet<i32> = BTreeSet::new();

    let mut queue: Vec<i32> = vec![root_hkt_id];

    while let Some(hkt_id) = queue.pop() {
        if !hkt_ids.insert(hkt_id) {
            continue;
        }

        if let Some(hkt) = hkt_build_output.hkts_by_id.get(&hkt_id) {
            for node in &hkt.nodes {
                node_ids.insert(node.node_id);
            }
        }

        for child in hkt_build_output.hkts_by_id.values() {
            if node_ids.contains(&child.parent_node_id) && !hkt_ids.contains(&child.hkt_id) {
                queue.push(child.hkt_id);
            }
        }
    }

    (hkt_ids, node_ids)
}

fn remove_subtree(hkt_build_output: &mut HktBuildOutput, root_hkt_id: i32) {
    let (hkt_ids, node_ids) = collect_subtree_hkt_and_node_ids(hkt_build_output, root_hkt_id);

    for node_id in node_ids {
        hkt_build_output.nodes_by_id.remove(&node_id);
    }
    for hkt_id in hkt_ids {
        hkt_build_output.hkts_by_id.remove(&hkt_id);
    }
}

fn remap_subtree_ids(
    subtree: HktBuildOutput,
    hkt_id_offset: i32,
    node_id_offset: i32,
    new_parent_node_id: i32,
) -> HktBuildOutput {
    let mut node_id_map: BTreeMap<i32, i32> = BTreeMap::new();
    let mut hkt_id_map: BTreeMap<i32, i32> = BTreeMap::new();

    for old_id in subtree.nodes_by_id.keys() {
        node_id_map.insert(*old_id, *old_id + node_id_offset);
    }
    for old_id in subtree.hkts_by_id.keys() {
        hkt_id_map.insert(*old_id, *old_id + hkt_id_offset);
    }

    let mut new_nodes_by_id: BTreeMap<i32, Node> = BTreeMap::new();
    for (old_id, mut node) in subtree.nodes_by_id {
        let new_id = *node_id_map.get(&old_id).unwrap();
        node.node_id = new_id;
        node.hkt_id = *hkt_id_map.get(&node.hkt_id).unwrap();
        new_nodes_by_id.insert(new_id, node);
    }

    let mut new_hkts_by_id: BTreeMap<i32, Hkt> = BTreeMap::new();
    for (old_id, mut hkt) in subtree.hkts_by_id {
        let new_id = *hkt_id_map.get(&old_id).unwrap();
        hkt.hkt_id = new_id;

        if hkt.parent_node_id == 0 {
            hkt.parent_node_id = new_parent_node_id;
        } else {
            hkt.parent_node_id = *node_id_map.get(&hkt.parent_node_id).unwrap();
        }

        for node in &mut hkt.nodes {
            node.node_id = *node_id_map.get(&node.node_id).unwrap();
            node.hkt_id = new_id;
        }

        new_hkts_by_id.insert(new_id, hkt);
    }

    HktBuildOutput {
        hkts_by_id: new_hkts_by_id,
        nodes_by_id: new_nodes_by_id,
    }
}
