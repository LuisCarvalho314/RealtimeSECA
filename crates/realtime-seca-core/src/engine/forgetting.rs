use super::*;
impl SecaEngine {
    pub(super) fn apply_seca_light_pruning_if_enabled(
        &mut self,
        current_batch_index: u32,
    ) -> Result<Option<usize>, SecaError> {
        if self.config.memory_mode != crate::config::MemoryMode::SlidingWindow {
            return Ok(None);
        }

        let Some(gamma) = self.config.max_batches_in_memory else {
            return Ok(None);
        };

        if gamma == 0 {
            return Err(SecaError::InvalidConfiguration {
                message: "max_batches_in_memory must be > 0 when using SlidingWindow".to_string(),
            });
        }

        let window_start = current_batch_index.saturating_sub(gamma.saturating_sub(1));
        let active_source_ids = self.compute_active_sources_for_window(window_start);

        let mut pruned_source_ids: BTreeSet<i64> = BTreeSet::new();
        let stale_batch_indexes: Vec<u32> = self
            .source_ids_by_batch_index
            .keys()
            .copied()
            .filter(|batch_index| *batch_index < window_start)
            .collect();
        for stale_batch_index in stale_batch_indexes {
            if let Some(source_ids) = self.source_ids_by_batch_index.remove(&stale_batch_index) {
                pruned_source_ids.extend(source_ids);
            }
        }
        pruned_source_ids.retain(|source_id| !active_source_ids.contains(source_id));

        for source_id in &pruned_source_ids {
            if let Some(external_source_id) = self.baseline_source_legend.remove(source_id) {
                self.source_id_by_url.remove(external_source_id.as_str());
            }
        }

        self.processed_batches
            .retain(|batch| batch.batch_index >= window_start);
        let prune_tree = |tree: &mut HktBuildOutput| {
            for hkt in tree.hkts_by_id.values_mut() {
                for node in &mut hkt.nodes {
                    node.source_ids.retain(|id| active_source_ids.contains(id));
                    node.source_ids_new_from_batches
                        .retain(|id| active_source_ids.contains(id));
                    for memberships in [
                        &mut node.word_source_ids,
                        &mut node.word_source_ids_new_from_batches,
                    ] {
                        memberships.retain(|_, ids| {
                            ids.retain(|id| active_source_ids.contains(id));
                            !ids.is_empty()
                        });
                    }
                    tree.nodes_by_id.insert(node.node_id, node.clone());
                }
            }
        };
        if let Some(tree) = self.hkt_build_output.as_mut() {
            prune_tree(tree);
        }
        Ok(Some(pruned_source_ids.len()))
    }
    pub(super) fn compute_active_sources_for_window(
        &self,
        window_start_batch_index: u32,
    ) -> BTreeSet<i64> {
        let mut active_source_ids = BTreeSet::new();
        for (batch_index, source_ids) in &self.source_ids_by_batch_index {
            if *batch_index >= window_start_batch_index {
                active_source_ids.extend(source_ids.iter().copied());
            }
        }
        active_source_ids
    }
}
