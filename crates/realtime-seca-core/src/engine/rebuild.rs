use super::*;
use crate::tree::{Hkt, Node};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildMode {
    FullFromAllBatches,      // existing fallback (safe)
    SubtreeTargeted,         // next implementation
    HybridFullOnRootTrigger, // optional transitional mode
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SelectedHktRebuildRequest {
    pub(crate) target_hkt_id: i32,
    pub(crate) scoped_batch_source_indexes: BTreeSet<usize>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SelectedHktRebuildPlan {
    pub(crate) batch_index: u32,
    pub(crate) requests: Vec<SelectedHktRebuildRequest>,
}

impl SecaEngine {
    pub(crate) fn execute_rebuild_action_for_trigger_plan(
        &mut self,
        target_hkt_ids: &[i32],
    ) -> Result<(), SecaError> {
        if target_hkt_ids.is_empty() {
            return Ok(());
        }

        match self.rebuild_mode {
            RebuildMode::FullFromAllBatches => self.rebuild_from_stored_batches(),
            RebuildMode::HybridFullOnRootTrigger => {
                if self.selected_rebuild_contains_root(target_hkt_ids)? {
                    self.rebuild_from_stored_batches()
                } else {
                    // Transitional behavior: selected rebuild path still falls back internally.
                    self.rebuild_selected_hkts_from_stored_batches(target_hkt_ids)
                }
            }
            RebuildMode::SubtreeTargeted => {
                // Transitional behavior: selected rebuild path still falls back internally.
                self.rebuild_selected_hkts_from_stored_batches(target_hkt_ids)
            }
        }
    }

    pub(crate) fn rebuild_selected_hkts_from_stored_batches(
        &mut self,
        target_hkt_ids: &[i32],
    ) -> Result<(), SecaError> {
        if target_hkt_ids.is_empty() {
            return Ok(());
        }

        // Stage 3B transitional fallback:
        // Paper intent is subtree/HKT-local reconstruction, but current execution remains full rebuild.
        let _ = target_hkt_ids;
        self.rebuild_from_stored_batches()
    }

    pub(crate) fn selected_rebuild_contains_root(
        &self,
        target_hkt_ids: &[i32],
    ) -> Result<bool, SecaError> {
        let hkt_build_output =
            self.hkt_build_output
                .as_ref()
                .ok_or_else(|| SecaError::StateError {
                    message: "cannot inspect root HKT: baseline tree missing".to_string(),
                })?;

        let root_hkt_id = hkt_build_output
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id == 0)
            .map(|hkt| hkt.hkt_id)
            .ok_or_else(|| SecaError::StateError {
                message: "cannot inspect root HKT: root HKT not found".to_string(),
            })?;

        Ok(target_hkt_ids.contains(&root_hkt_id))
    }

    pub(crate) fn rebuild_from_stored_batches(&mut self) -> Result<(), SecaError> {
        if self.processed_batches.is_empty() {
            return Err(SecaError::StateError {
                message: "cannot rebuild from stored batches: no batches are stored".to_string(),
            });
        }

        let combined_batch_index = self.last_processed_batch_index.unwrap_or(0);

        let mut combined_sources = Vec::new();
        for batch in &self.processed_batches {
            combined_sources.extend(batch.sources.clone());
        }

        let combined_batch = SourceBatch {
            batch_index: combined_batch_index,
            sources: combined_sources,
        };

        let mut temporary_engine = SecaEngine::new(self.config.clone())?;
        temporary_engine.rebuild_mode = self.rebuild_mode;
        let _ = temporary_engine.build_baseline_tree(combined_batch)?;

        self.hkt_build_output = temporary_engine.hkt_build_output.clone();
        self.baseline_word_legend = temporary_engine.baseline_word_legend.clone();
        self.baseline_source_legend = temporary_engine.baseline_source_legend.clone();
        self.has_baseline = temporary_engine.has_baseline;
        self.next_hkt_id = temporary_engine.next_hkt_id;
        self.next_node_id = temporary_engine.next_node_id;
        self.rebuild_source_registration_from_processed_batches();

        Ok(())
    }

    pub(crate) fn propagate_sources_to_parent_chain(
        &mut self,
        start_parent_node_id: i32,
        source_ids: &BTreeSet<i64>,
    ) -> Result<(), SecaError> {
        let hkt_build_output =
            self.hkt_build_output
                .as_mut()
                .ok_or_else(|| SecaError::StateError {
                    message: "cannot propagate sources: baseline tree missing".to_string(),
                })?;

        let mut current_node_id = start_parent_node_id;

        while current_node_id != 0 {
            let hkt_id = match hkt_build_output.nodes_by_id.get(&current_node_id) {
                Some(_) => {
                    // extend sources on the node itself
                    let node = hkt_build_output
                        .nodes_by_id
                        .get_mut(&current_node_id)
                        .expect("node existed above; missing on mutable fetch");
                    node.source_ids.extend(source_ids.iter().copied());
                    node.hkt_id
                }
                None => break,
            };

            let parent_node_id = hkt_build_output
                .hkts_by_id
                .get(&hkt_id)
                .map(|hkt| hkt.parent_node_id)
                .unwrap_or(0);

            current_node_id = parent_node_id;
        }

        Ok(())
    }

    pub(crate) fn extract_scoped_batch_from_indexes(
        &self,
        batch: &SourceBatch,
        scoped_batch_source_indexes: &BTreeSet<usize>,
    ) -> Result<SourceBatch, SecaError> {
        let mut scoped_sources = Vec::with_capacity(scoped_batch_source_indexes.len());

        for source_index in scoped_batch_source_indexes {
            let source = batch
                .sources
                .get(*source_index)
                .ok_or_else(|| SecaError::StateError {
                    message: format!(
                        "cannot extract scoped batch: source index {} out of bounds",
                        source_index
                    ),
                })?;
            scoped_sources.push(source.clone());
        }

        Ok(SourceBatch {
            batch_index: batch.batch_index,
            sources: scoped_sources,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn build_selected_hkt_rebuild_plan(
        &self,
        batch: &SourceBatch,
        target_hkt_ids: &[i32],
    ) -> Result<SelectedHktRebuildPlan, SecaError> {
        if target_hkt_ids.is_empty() {
            return Ok(SelectedHktRebuildPlan {
                batch_index: batch.batch_index,
                requests: Vec::new(),
            });
        }

        let hkt_build_output =
            self.hkt_build_output
                .as_ref()
                .ok_or_else(|| SecaError::StateError {
                    message: "cannot build selected HKT rebuild plan: baseline tree missing"
                        .to_string(),
                })?;

        let mut deduplicated_ids: Vec<i32> = target_hkt_ids.to_vec();
        deduplicated_ids.sort_unstable();
        deduplicated_ids.dedup();

        for target_hkt_id in &deduplicated_ids {
            if !hkt_build_output.hkts_by_id.contains_key(target_hkt_id) {
                return Err(SecaError::StateError {
                    message: format!(
                        "cannot build selected HKT rebuild plan: HKT {} not found",
                        target_hkt_id
                    ),
                });
            }
        }

        let root_hkt_id = self.find_root_hkt_id()?;
        let all_source_indexes: BTreeSet<usize> = (0..batch.sources.len()).collect();

        let target_id_set: BTreeSet<i32> = deduplicated_ids.iter().copied().collect();
        let mut collected_scopes_by_hkt_id: BTreeMap<i32, BTreeSet<usize>> = BTreeMap::new();

        self.collect_selected_hkt_scopes_for_batch(
            batch,
            root_hkt_id,
            &all_source_indexes,
            &target_id_set,
            &mut collected_scopes_by_hkt_id,
        )?;

        let requests = deduplicated_ids
            .into_iter()
            .map(|target_hkt_id| SelectedHktRebuildRequest {
                target_hkt_id,
                scoped_batch_source_indexes: collected_scopes_by_hkt_id
                    .remove(&target_hkt_id)
                    .unwrap_or_default(),
            })
            .collect();

        Ok(SelectedHktRebuildPlan {
            batch_index: batch.batch_index,
            requests,
        })
    }

    pub(crate) fn rebuild_selected_hkts_from_trigger_plan(
        &mut self,
        batch: &SourceBatch,
        trigger_plan: &crate::engine::trigger::RecursiveTriggerPlan,
    ) -> Result<(), SecaError> {
        for target_hkt_id in &trigger_plan.reconstruct_hkt_ids {
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

            let max_hkt_id = hkt_build_output
                .hkts_by_id
                .keys()
                .max()
                .copied()
                .unwrap_or(0);
            let max_node_id = hkt_build_output
                .nodes_by_id
                .keys()
                .max()
                .copied()
                .unwrap_or(0);
            let archived_max_hkt_id = self
                .archived_subtrees_by_root_id
                .values()
                .flat_map(|subtree| subtree.hkts_by_id.keys().copied())
                .max()
                .unwrap_or(0);
            let archived_max_node_id = self
                .archived_subtrees_by_root_id
                .values()
                .flat_map(|subtree| subtree.nodes_by_id.keys().copied())
                .max()
                .unwrap_or(0);

            let global_max_hkt_id = max_hkt_id.max(archived_max_hkt_id);
            let global_max_node_id = max_node_id.max(archived_max_node_id);

            if self.next_hkt_id <= global_max_hkt_id {
                self.next_hkt_id = global_max_hkt_id + 1;
            }
            if self.next_node_id <= global_max_node_id {
                self.next_node_id = global_max_node_id + 1;
            }

            let hkt_id_offset = self.next_hkt_id - 1;
            let node_id_offset = self.next_node_id - 1;

            let archived_subtree = extract_subtree(hkt_build_output, target_hkt_id);
            if let Some(removed_root) = archived_subtree.hkts_by_id.get(&target_hkt_id) {
                let old_parent_node_id = removed_root.parent_node_id;
                let mut logical_removed = removed_root.clone();
                logical_removed.parent_node_id = -1;
                logical_removed.nodes.clear();
                self.logically_removed_hkts_by_id.insert(
                    target_hkt_id,
                    crate::engine::LogicalRemovedHkt {
                        hkt: logical_removed,
                        old_parent_node_id,
                    },
                );
            }
            if !archived_subtree.hkts_by_id.is_empty() {
                self.archived_subtrees_by_root_id
                    .insert(target_hkt_id, archived_subtree);
            }

            remove_subtree(hkt_build_output, target_hkt_id);

            let remapped =
                remap_subtree_ids(subtree, hkt_id_offset, node_id_offset, parent_node_id);
            let remapped_hkt_len = remapped.hkts_by_id.len() as i32;
            let remapped_node_len = remapped.nodes_by_id.len() as i32;
            hkt_build_output.hkts_by_id.extend(remapped.hkts_by_id);
            hkt_build_output.nodes_by_id.extend(remapped.nodes_by_id);

            self.next_hkt_id += remapped_hkt_len;
            self.next_node_id += remapped_node_len;
        }

        Ok(())
    }
    pub(crate) fn format_selected_hkt_rebuild_plan_note(
        &self,
        plan: &SelectedHktRebuildPlan,
    ) -> String {
        if plan.requests.is_empty() {
            return format!(
                "Selected-HKT rebuild plan (batch {}): no HKT targets",
                plan.batch_index
            );
        }

        let request_summaries = plan
            .requests
            .iter()
            .map(|request| {
                format!(
                    "HKT {} (scoped_sources={})",
                    request.target_hkt_id,
                    request.scoped_batch_source_indexes.len()
                )
            })
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "Selected-HKT rebuild plan (batch {}): {} target(s) -> {}",
            plan.batch_index,
            plan.requests.len(),
            request_summaries
        )
    }
    #[allow(dead_code)]
    fn collect_selected_hkt_scopes_for_batch(
        &self,
        batch: &SourceBatch,
        hkt_id: i32,
        scoped_batch_source_indexes: &BTreeSet<usize>,
        target_hkt_ids: &BTreeSet<i32>,
        collected: &mut BTreeMap<i32, BTreeSet<usize>>,
    ) -> Result<(), SecaError> {
        let scope_snapshot = self.snapshot_hkt_scope(hkt_id)?;
        let mapped_scope =
            self.map_batch_into_hkt_scope(batch, scoped_batch_source_indexes, &scope_snapshot)?;

        if target_hkt_ids.contains(&hkt_id) {
            collected.insert(hkt_id, mapped_scope.scoped_batch_source_indexes.clone());
        }

        let mut child_hkt_ids_with_scopes: Vec<(i32, BTreeSet<usize>)> = Vec::new();

        for node_id in &scope_snapshot.non_refuge_node_ids {
            if let Some(child_hkt_id) = scope_snapshot.child_hkt_ids_by_parent_node_id.get(node_id)
            {
                let child_scope = mapped_scope
                    .matched_node_source_indexes_by_node_id
                    .get(node_id)
                    .cloned()
                    .unwrap_or_default();

                if !child_scope.is_empty() {
                    child_hkt_ids_with_scopes.push((*child_hkt_id, child_scope));
                }
            }
        }

        child_hkt_ids_with_scopes.sort_by_key(|(child_hkt_id, _)| *child_hkt_id);

        for (child_hkt_id, child_scope) in child_hkt_ids_with_scopes {
            self.collect_selected_hkt_scopes_for_batch(
                batch,
                child_hkt_id,
                &child_scope,
                target_hkt_ids,
                collected,
            )?;
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn find_root_hkt_id(&self) -> Result<i32, SecaError> {
        let hkt_build_output =
            self.hkt_build_output
                .as_ref()
                .ok_or_else(|| SecaError::StateError {
                    message: "cannot inspect root HKT: baseline tree missing".to_string(),
                })?;

        hkt_build_output
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id == 0)
            .map(|hkt| hkt.hkt_id)
            .ok_or_else(|| SecaError::StateError {
                message: "cannot inspect root HKT: root HKT not found".to_string(),
            })
    }
    pub(crate) fn build_selected_hkt_rebuild_plan_from_trigger_plan(
        &self,
        batch: &SourceBatch,
        trigger_plan: &crate::engine::trigger::RecursiveTriggerPlan,
    ) -> Result<SelectedHktRebuildPlan, SecaError> {
        let hkt_build_output =
            self.hkt_build_output
                .as_ref()
                .ok_or_else(|| SecaError::StateError {
                    message: "cannot build selected HKT rebuild plan: baseline tree missing"
                        .to_string(),
                })?;

        let mut deduplicated_ids = trigger_plan.reconstruct_hkt_ids.clone();
        deduplicated_ids.sort_unstable();
        deduplicated_ids.dedup();

        let mut requests = Vec::with_capacity(deduplicated_ids.len());

        for target_hkt_id in deduplicated_ids {
            if !hkt_build_output.hkts_by_id.contains_key(&target_hkt_id) {
                return Err(SecaError::StateError {
                    message: format!(
                        "cannot build selected HKT rebuild plan: HKT {} not found",
                        target_hkt_id
                    ),
                });
            }

            let scoped_indexes = trigger_plan
                .reconstruct_scopes_by_hkt_id
                .get(&target_hkt_id)
                .cloned()
                .unwrap_or_default();

            if let Some(out_of_bounds_index) = scoped_indexes
                .iter()
                .find(|index| **index >= batch.sources.len())
            {
                return Err(SecaError::StateError {
                message: format!(
                    "cannot build selected HKT rebuild plan: scoped source index {} out of bounds for batch size {} (HKT {})",
                    out_of_bounds_index,
                    batch.sources.len(),
                    target_hkt_id
                ),
            });
            }

            requests.push(SelectedHktRebuildRequest {
                target_hkt_id,
                scoped_batch_source_indexes: scoped_indexes,
            });
        }

        Ok(SelectedHktRebuildPlan {
            batch_index: batch.batch_index,
            requests,
        })
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
    let scoped_indexes = trigger_plan
        .reconstruct_scopes_by_hkt_id
        .get(&target_hkt_id)
        .cloned()
        .unwrap_or_default();

    // token -> word_id
    let mut baseline_word_id_by_token: BTreeMap<&str, i32> = BTreeMap::new();
    for (word_id, token) in &engine.baseline_word_legend {
        baseline_word_id_by_token.insert(token.as_str(), *word_id);
    }

    // Ancestor filter sets
    let ancestor_words = trigger_plan
        .ancestor_words_by_hkt_id
        .get(&target_hkt_id)
        .cloned()
        .unwrap_or_default();
    let ancestor_accept = trigger_plan
        .ancestor_accepted_words_by_hkt_id
        .get(&target_hkt_id)
        .cloned()
        .unwrap_or_default();
    let ancestor_reject = trigger_plan
        .ancestor_rejected_words_by_hkt_id
        .get(&target_hkt_id)
        .cloned()
        .unwrap_or_default();

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
                if ancestor_words.contains_key(word_id)
                    || ancestor_accept.contains_key(word_id)
                    || ancestor_reject.contains_key(word_id)
                {
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
    for idx in scoped_indexes {
        if let Some(source) = current_batch.sources.get(idx) {
            push_source(&source.source_id, &source.tokens);
        }
    }

    // Fill word_number_of_sources
    let mut source_ids_by_word_id: BTreeMap<i32, BTreeSet<i64>> = BTreeMap::new();
    for record in &records {
        source_ids_by_word_id
            .entry(record.word_id)
            .or_default()
            .insert(record.source_id);
    }
    for record in &mut records {
        record.word_number_of_sources = source_ids_by_word_id
            .get(&record.word_id)
            .map(|set| set.len())
            .unwrap_or(0);
    }

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

fn extract_subtree(hkt_build_output: &HktBuildOutput, root_hkt_id: i32) -> HktBuildOutput {
    let (hkt_ids, node_ids) = collect_subtree_hkt_and_node_ids(hkt_build_output, root_hkt_id);
    let mut hkts_by_id = BTreeMap::new();
    let mut nodes_by_id = BTreeMap::new();

    for hkt_id in hkt_ids {
        if let Some(hkt) = hkt_build_output.hkts_by_id.get(&hkt_id) {
            hkts_by_id.insert(hkt_id, hkt.clone());
        }
    }

    for node_id in node_ids {
        if let Some(node) = hkt_build_output.nodes_by_id.get(&node_id) {
            nodes_by_id.insert(node_id, node.clone());
        }
    }

    HktBuildOutput {
        hkts_by_id,
        nodes_by_id,
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
