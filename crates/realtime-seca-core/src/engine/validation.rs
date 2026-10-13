use super::*;

impl SecaEngine {
    /// Check resumable state at the persistence boundary, before accepting it.
    pub(super) fn validate_restored_state(&self) -> Result<(), SecaError> {
        let invalid = || SecaError::StateError {
            message: "inconsistent model state".into(),
        };
        let tree = self.hkt_build_output.as_ref().ok_or_else(invalid)?;
        let roots: Vec<_> = tree
            .hkts_by_id
            .values()
            .filter(|h| h.parent_node_id == 0)
            .collect();
        if roots.len() != 1 {
            return Err(invalid());
        }
        let mut visited = BTreeSet::new();
        let mut pending = vec![roots[0].hkt_id];
        let mut node_ids = BTreeSet::new();
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                return Err(invalid());
            }
            let hkt = tree.hkts_by_id.get(&id).ok_or_else(invalid)?;
            if hkt.hkt_id != id {
                return Err(invalid());
            }
            for node in &hkt.nodes {
                if node.hkt_id != id
                    || !node_ids.insert(node.node_id)
                    || tree.nodes_by_id.get(&node.node_id) != Some(node)
                    || node
                        .source_ids
                        .iter()
                        .any(|s| !self.baseline_source_legend.contains_key(s))
                    || node
                        .word_ids
                        .iter()
                        .any(|w| *w != -1 && !self.baseline_word_legend.contains_key(w))
                    || node
                        .word_source_ids
                        .values()
                        .any(|s| !s.is_subset(&node.source_ids))
                    || !node.source_ids_new_from_batches.is_empty()
                    || !node.word_source_ids_new_from_batches.is_empty()
                {
                    return Err(invalid());
                }
                pending.extend(
                    tree.hkts_by_id
                        .values()
                        .filter(|h| h.parent_node_id == node.node_id)
                        .map(|h| h.hkt_id),
                );
            }
        }
        if visited.len() != tree.hkts_by_id.len()
            || node_ids.len() != tree.nodes_by_id.len()
            || tree.hkts_by_id.keys().any(|id| *id >= self.next_hkt_id)
            || tree.nodes_by_id.keys().any(|id| *id >= self.next_node_id)
        {
            return Err(invalid());
        }
        let mut batches = BTreeMap::<u32, BTreeSet<i64>>::new();
        let mut source_ids = BTreeSet::new();
        for batch in &self.processed_batches {
            for source in &batch.sources {
                let id = self
                    .source_id_by_url
                    .get(&source.source_id)
                    .ok_or_else(invalid)?;
                if source.batch_index != batch.batch_index
                    || !source_ids.insert(*id)
                    || self.baseline_source_legend.get(id) != Some(&source.source_id)
                {
                    return Err(invalid());
                }
                batches.entry(batch.batch_index).or_default().insert(*id);
            }
        }
        if batches != self.source_ids_by_batch_index
            || source_ids.len() != self.baseline_source_legend.len()
            || source_ids.len() != self.source_id_by_url.len()
            || self.processed_batches.last().map(|b| b.batch_index)
                != self.last_processed_batch_index
        {
            return Err(invalid());
        }
        let tokens: BTreeSet<_> = self.baseline_word_legend.values().collect();
        if tokens.len() != self.baseline_word_legend.len()
            || self.baseline_word_legend.keys().any(|id| *id <= 0)
        {
            return Err(invalid());
        }
        Ok(())
    }
}
