use super::scope_mapping::{HktScopeSnapshot, MappedHktScopeState};
use super::trigger::*;
use super::*;
impl SecaEngine {
    pub(super) fn apply_state1_updates_for_hkt(
        &mut self,
        hkt_id: i32,
        update_stage: &HktUpdateStage,
    ) -> Result<(), SecaError> {
        let hkt_build_output =
            self.hkt_build_output
                .as_mut()
                .ok_or_else(|| SecaError::StateError {
                    message: "cannot apply state1 updates: baseline tree missing".to_string(),
                })?;

        let hkt = hkt_build_output
            .hkts_by_id
            .get_mut(&hkt_id)
            .ok_or_else(|| SecaError::StateError {
                message: format!("cannot apply state1 updates: HKT {} not found", hkt_id),
            })?;

        hkt.expected_words = update_stage.updated_expected_word_ids.clone();

        for node in &mut hkt.nodes {
            node.source_ids_new_from_batches.clear();
            node.word_source_ids_new_from_batches.clear();

            if let Some(ids) = update_stage
                .node_state1_source_ids_by_node_id
                .get(&node.node_id)
            {
                node.source_ids_new_from_batches.extend(ids.iter().copied());
            }
            if !node.is_refuge_node() {
                if let Some(state1_sources_by_word) = update_stage
                    .node_state1_sources_by_word_id
                    .get(&node.node_id)
                {
                    for word_id in node.word_ids.iter().copied().filter(|id| *id != -1) {
                        if let Some(state1_sources) = state1_sources_by_word.get(&word_id) {
                            node.word_source_ids_new_from_batches
                                .entry(word_id)
                                .or_default()
                                .extend(state1_sources.iter().copied());
                            node.source_ids_new_from_batches
                                .extend(state1_sources.iter().copied());
                        }
                    }
                }
            }

            if let Some(global_node) = hkt_build_output.nodes_by_id.get_mut(&node.node_id) {
                *global_node = node.clone();
            }
        }

        Ok(())
    }
    pub(super) fn merge_state1_updates_for_hkt(&mut self, hkt_id: i32) -> Result<(), SecaError> {
        let hkt_build_output =
            self.hkt_build_output
                .as_mut()
                .ok_or_else(|| SecaError::StateError {
                    message: "cannot merge state1 updates: baseline tree missing".to_string(),
                })?;

        let hkt = hkt_build_output
            .hkts_by_id
            .get_mut(&hkt_id)
            .ok_or_else(|| SecaError::StateError {
                message: format!("cannot merge state1 updates: HKT {} not found", hkt_id),
            })?;

        for node in &mut hkt.nodes {
            for (word_id, sources) in std::mem::take(&mut node.word_source_ids_new_from_batches) {
                node.word_source_ids
                    .entry(word_id)
                    .or_default()
                    .extend(sources);
            }

            node.source_ids
                .append(&mut node.source_ids_new_from_batches);

            if let Some(global_node) = hkt_build_output.nodes_by_id.get_mut(&node.node_id) {
                *global_node = node.clone();
            }
        }

        Ok(())
    }
    pub(super) fn compute_update_stage_for_scope(
        &self,
        batch: &SourceBatch,
        scope: &HktScopeSnapshot,
        mapped: &MappedHktScopeState,
        ancestor: &AncestorContext,
        retained_words: &BTreeMap<i64, BTreeSet<i32>>,
    ) -> Result<HktUpdateStage, SecaError> {
        let prior_df = self.build_state0_df_in_parent_scope(scope, retained_words)?;
        let prior_node_words = self.build_node_state0_sources_by_word_id(scope, retained_words);
        let mut stage = self.admit_candidate_words(scope, mapped, ancestor, &prior_df);
        self.map_incoming_word_memberships(batch, mapped, &mut stage)?;
        self.recheck_candidate_strength(scope, mapped, &prior_df, &prior_node_words, &mut stage);
        self.assign_candidates_to_nodes(scope, mapped, &mut stage);
        Ok(stage)
    }

    fn admit_candidate_words(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        mapped_scope: &MappedHktScopeState,
        ancestor: &AncestorContext,
        prior_df: &BTreeMap<i32, usize>,
    ) -> HktUpdateStage {
        let mut updated_expected_word_ids = scope_snapshot.expected_word_ids.clone();
        let mut token_to_word_id: BTreeMap<String, i32> = BTreeMap::new();
        let mut word_id_to_token: BTreeMap<i32, String> = BTreeMap::new();
        let mut next_synthetic_word_id: i32 = -2;
        for (word_id, token) in &self.baseline_word_legend {
            token_to_word_id.insert(token.clone(), *word_id);
            word_id_to_token.insert(*word_id, token.clone());
        }

        let alpha_threshold = self.config.seca_thresholds.alpha;

        let mut current_words_in_hkt: BTreeSet<i32> = BTreeSet::new();
        for word_ids in scope_snapshot.node_word_ids_by_node_id.values() {
            for word_id in word_ids {
                if *word_id != -1 {
                    current_words_in_hkt.insert(*word_id);
                }
            }
        }

        let state0_prominent_count = prior_df
            .iter()
            .filter(|(word, _)| !ancestor.contains(word))
            .map(|(word, old)| {
                let added = self
                    .baseline_word_legend
                    .get(word)
                    .and_then(|t| mapped_scope.word_document_frequency_in_scope.get(t))
                    .copied()
                    .unwrap_or(0);
                old + added
            })
            .chain(
                mapped_scope
                    .word_document_frequency_in_scope
                    .iter()
                    .filter(|(token, _)| {
                        token_to_word_id
                            .get(*token)
                            .is_none_or(|word| !ancestor.contains(word))
                    })
                    .map(|(_, count)| *count),
            )
            .max()
            .unwrap_or(0)
            .max(1);
        for (token, state1_count) in &mapped_scope.word_document_frequency_in_scope {
            let word_id = token_to_word_id.get(token).copied().unwrap_or_else(|| {
                let assigned = next_synthetic_word_id;
                next_synthetic_word_id -= 1;
                token_to_word_id.insert(token.clone(), assigned);
                word_id_to_token.insert(assigned, token.clone());
                assigned
            });
            if current_words_in_hkt.contains(&word_id) {
                continue;
            }
            let state0_count = prior_df.get(&word_id).copied().unwrap_or(0);
            let ratio = (state0_count + *state1_count) as f64 / (state0_prominent_count as f64);
            if ratio >= alpha_threshold {
                updated_expected_word_ids.insert(word_id);
            }
        }

        updated_expected_word_ids
            .retain(|word| !current_words_in_hkt.contains(word) && !ancestor.contains(word));

        HktUpdateStage {
            updated_expected_word_ids,
            token_by_word_id: word_id_to_token,
            ..Default::default()
        }
    }

    fn map_incoming_word_memberships(
        &self,
        batch: &SourceBatch,
        mapped: &MappedHktScopeState,
        stage: &mut HktUpdateStage,
    ) -> Result<(), SecaError> {
        let word_by_token: BTreeMap<_, _> = stage
            .token_by_word_id
            .iter()
            .map(|(id, token)| (token.as_str(), *id))
            .collect();
        for (node, indexes) in &mapped.matched_node_source_indexes_by_node_id {
            for index in indexes {
                let source = batch
                    .sources
                    .get(*index)
                    .ok_or_else(|| SecaError::StateError {
                        message: "source index out of bounds during context update".into(),
                    })?;
                let id = self
                    .source_id_by_url
                    .get(&source.source_id)
                    .copied()
                    .unwrap_or_else(|| Self::fnv1a_64(&source.source_id));
                stage
                    .node_state1_source_ids_by_node_id
                    .entry(*node)
                    .or_default()
                    .insert(id);
                for token in &source.tokens {
                    if let Some(word) = word_by_token.get(token.trim()) {
                        stage
                            .node_state1_sources_by_word_id
                            .entry(*node)
                            .or_default()
                            .entry(*word)
                            .or_default()
                            .insert(id);
                    }
                }
            }
        }
        Ok(())
    }

    fn recheck_candidate_strength(
        &self,
        scope: &HktScopeSnapshot,
        mapped: &MappedHktScopeState,
        prior_df: &BTreeMap<i32, usize>,
        prior_node_words: &BTreeMap<i32, BTreeMap<i32, BTreeSet<i64>>>,
        stage: &mut HktUpdateStage,
    ) {
        let candidate_counts: BTreeMap<_, _> = stage
            .updated_expected_word_ids
            .iter()
            .map(|word| {
                let old = prior_df.get(word).copied().unwrap_or(0);
                let added = stage
                    .token_by_word_id
                    .get(word)
                    .and_then(|token| mapped.word_document_frequency_in_scope.get(token))
                    .copied()
                    .unwrap_or(0);
                (*word, old + added)
            })
            .collect();
        let mut prominent = candidate_counts.values().copied().max().unwrap_or(0).max(1);
        for (node, words) in &scope.node_word_ids_by_node_id {
            if words.contains(&-1) {
                continue;
            }
            for word in words {
                let old = prior_node_words
                    .get(node)
                    .and_then(|m| m.get(word))
                    .map_or(0, BTreeSet::len);
                let added = stage
                    .node_state1_sources_by_word_id
                    .get(node)
                    .and_then(|m| m.get(word))
                    .map_or(0, BTreeSet::len);
                prominent = prominent.max(old + added);
            }
        }
        stage.updated_expected_word_ids.retain(|word| {
            candidate_counts[word] as f64 / prominent as f64 >= self.config.seca_thresholds.alpha
        });
    }

    fn assign_candidates_to_nodes(
        &self,
        scope: &HktScopeSnapshot,
        mapped: &MappedHktScopeState,
        stage: &mut HktUpdateStage,
    ) {
        for node in mapped.matched_node_source_indexes_by_node_id.keys() {
            if scope
                .node_word_ids_by_node_id
                .get(node)
                .is_some_and(|words| words.contains(&-1))
            {
                continue;
            }
            let mut members = scope
                .node_source_ids_by_node_id
                .get(node)
                .cloned()
                .unwrap_or_default();
            if let Some(added) = stage.node_state1_source_ids_by_node_id.get(node) {
                members.extend(added);
            }
            for word in &stage.updated_expected_word_ids {
                let Some(sources) = stage
                    .node_state1_sources_by_word_id
                    .get(node)
                    .and_then(|m| m.get(word))
                else {
                    continue;
                };
                if sources.is_empty() {
                    continue;
                }
                let eligibility =
                    sources.intersection(&members).count() as f64 / members.len().max(1) as f64;
                if eligibility >= self.config.seca_thresholds.beta {
                    stage
                        .assigned_expected_words_by_node_id
                        .entry(*node)
                        .or_default()
                        .insert(*word);
                }
            }
        }
    }
}
