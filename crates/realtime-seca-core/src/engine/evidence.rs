use super::scope_mapping::HktScopeSnapshot;
use super::trigger::AncestorContext;
use super::*;
impl SecaEngine {
    pub(super) fn build_baseline_source_word_sets(
        &self,
    ) -> Result<BTreeMap<i64, BTreeSet<i32>>, SecaError> {
        let mut baseline_word_id_by_token: BTreeMap<&str, i32> = BTreeMap::new();
        for (word_id, token) in &self.baseline_word_legend {
            baseline_word_id_by_token.insert(token.as_str(), *word_id);
        }

        let mut by_source: BTreeMap<i64, BTreeSet<i32>> = BTreeMap::new();

        for source in self
            .processed_batches
            .iter()
            .flat_map(|batch| &batch.sources)
        {
            let internal_source_id = self
                .source_id_by_url
                .get(source.source_id.as_str())
                .copied()
                .unwrap_or_else(|| SecaEngine::fnv1a_64(source.source_id.as_str()));

            let mut unique_tokens: BTreeSet<&str> = BTreeSet::new();
            for token in &source.tokens {
                let normalized = token.trim();
                if !normalized.is_empty() {
                    unique_tokens.insert(normalized);
                }
            }

            let entry = by_source.entry(internal_source_id).or_default();
            for token in unique_tokens {
                if let Some(word_id) = baseline_word_id_by_token.get(token) {
                    entry.insert(*word_id);
                }
            }
        }

        Ok(by_source)
    }
    pub(super) fn build_state0_df_in_parent_scope(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        baseline_source_word_sets: &BTreeMap<i64, BTreeSet<i32>>,
    ) -> Result<BTreeMap<i32, usize>, SecaError> {
        let mut scope_source_ids: BTreeSet<i64> = BTreeSet::new();

        if scope_snapshot.parent_node_id == 0 {
            scope_source_ids.extend(baseline_source_word_sets.keys().copied());
        } else {
            let hkt_build_output =
                self.hkt_build_output
                    .as_ref()
                    .ok_or_else(|| SecaError::StateError {
                        message: "baseline tree missing for parent-scope DF".to_string(),
                    })?;
            let parent_node = hkt_build_output
                .nodes_by_id
                .get(&scope_snapshot.parent_node_id)
                .ok_or_else(|| SecaError::StateError {
                    message: format!(
                        "cannot compute parent-scope DF: parent node {} not found",
                        scope_snapshot.parent_node_id
                    ),
                })?;
            scope_source_ids.extend(parent_node.source_ids.iter().copied());
        }

        let mut df_by_word_id: BTreeMap<i32, usize> = BTreeMap::new();

        for source_id in scope_source_ids {
            if let Some(word_ids) = baseline_source_word_sets.get(&source_id) {
                for word_id in word_ids {
                    *df_by_word_id.entry(*word_id).or_insert(0) += 1;
                }
            }
        }

        Ok(df_by_word_id)
    }
    pub(super) fn build_node_state0_sources_by_word_id(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        baseline_source_word_sets: &BTreeMap<i64, BTreeSet<i32>>,
    ) -> BTreeMap<i32, BTreeMap<i32, BTreeSet<i64>>> {
        let mut result: BTreeMap<i32, BTreeMap<i32, BTreeSet<i64>>> = BTreeMap::new();

        for (node_id, node_word_ids) in &scope_snapshot.node_word_ids_by_node_id {
            let node_source_ids = scope_snapshot
                .node_source_ids_by_node_id
                .get(node_id)
                .cloned()
                .unwrap_or_default();

            for source_id in node_source_ids {
                if let Some(word_ids_in_source) = baseline_source_word_sets.get(&source_id) {
                    for word_id in word_ids_in_source {
                        if *word_id == -1 {
                            continue;
                        }
                        if !node_word_ids.contains(word_id) {
                            continue;
                        }
                        result
                            .entry(*node_id)
                            .or_default()
                            .entry(*word_id)
                            .or_default()
                            .insert(source_id);
                    }
                }
            }
        }

        result
    }
    pub(super) fn build_child_ancestor_context_from_parent_node(
        &self,
        scope: &HktScopeSnapshot,
        parent_node_id: i32,
        ancestor: &AncestorContext,
    ) -> AncestorContext {
        let mut child = ancestor.clone();
        for word in scope
            .node_word_ids_by_node_id
            .get(&parent_node_id)
            .into_iter()
            .flatten()
            .filter(|id| **id != -1)
        {
            child.insert(*word);
        }
        child
    }
}
