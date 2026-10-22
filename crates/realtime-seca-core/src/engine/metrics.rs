use super::scope_mapping::{HktScopeSnapshot, MappedHktScopeState};
use super::trigger::*;
use super::*;
impl SecaEngine {
    pub(super) fn compute_word_change_metrics_for_scope(
        &self,
        scope: &HktScopeSnapshot,
        mapped: &MappedHktScopeState,
        stage: &HktUpdateStage,
        retained_words: &BTreeMap<i64, BTreeSet<i32>>,
    ) -> Result<BTreeMap<i32, WordChangeMetrics>, SecaError> {
        let prior_df = self.build_state0_df_in_parent_scope(scope, retained_words)?;
        let candidate_counts: BTreeMap<_, _> = stage
            .updated_expected_word_ids
            .iter()
            .map(|word| {
                let incoming = stage
                    .token_by_word_id
                    .get(word)
                    .or_else(|| self.baseline_word_legend.get(word))
                    .and_then(|token| mapped.word_document_frequency_in_scope.get(token))
                    .copied()
                    .unwrap_or(0);
                (*word, (prior_df.get(word).copied().unwrap_or(0), incoming))
            })
            .collect();
        let mut current_counts = Vec::new();
        for (node, words) in &scope.node_word_ids_by_node_id {
            if words.contains(&-1) {
                continue;
            }
            for word in words {
                let old = scope
                    .node_word_source_ids_by_node_id
                    .get(node)
                    .and_then(|by_word| by_word.get(word))
                    .map_or(0, BTreeSet::len);
                let added = stage
                    .node_state1_sources_by_word_id
                    .get(node)
                    .and_then(|by_word| by_word.get(word))
                    .map_or(0, BTreeSet::len);
                current_counts.push(old + added);
            }
        }
        let updated_counts = current_counts
            .iter()
            .copied()
            .chain(candidate_counts.values().map(|(old, added)| old + added));
        let importance_denominator = updated_counts.clone().sum::<usize>().max(1) as f64;
        let prominent_count = updated_counts.max().unwrap_or(0).max(1) as f64;
        let mut result = BTreeMap::new();
        for (node, words) in &scope.node_word_ids_by_node_id {
            if words.contains(&-1) {
                continue;
            }
            let mut node_words = words.clone();
            if let Some(assigned) = stage.assigned_expected_words_by_node_id.get(node) {
                node_words.extend(assigned);
            }
            let mut members = scope
                .node_source_ids_by_node_id
                .get(node)
                .cloned()
                .unwrap_or_default();
            if let Some(added) = stage.node_state1_source_ids_by_node_id.get(node) {
                members.extend(added);
            }
            for word in node_words {
                let mut updated = scope
                    .node_word_source_ids_by_node_id
                    .get(node)
                    .and_then(|by_word| by_word.get(&word))
                    .cloned()
                    .unwrap_or_default();
                if let Some(added) = stage
                    .node_state1_sources_by_word_id
                    .get(node)
                    .and_then(|by_word| by_word.get(&word))
                {
                    updated.extend(added);
                }
                result.insert(
                    word,
                    WordChangeMetrics {
                        strength: updated.len() as f64 / prominent_count,
                        eligibility: members.intersection(&updated).count() as f64
                            / members.len().max(1) as f64,
                        importance: updated.len() as f64 / importance_denominator,
                    },
                );
            }
        }
        for (word, (old, added)) in candidate_counts {
            result.entry(word).or_insert_with(|| WordChangeMetrics {
                strength: (old + added) as f64 / prominent_count,
                eligibility: 0.0,
                importance: (old + added) as f64 / importance_denominator,
            });
        }
        Ok(result)
    }
    pub(super) fn build_current_word_ids(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        _update_stage: &HktUpdateStage,
    ) -> BTreeSet<i32> {
        let mut current_word_ids: BTreeSet<i32> = BTreeSet::new();
        for word_ids in scope_snapshot.node_word_ids_by_node_id.values() {
            if word_ids.contains(&-1) {
                continue;
            }
            current_word_ids.extend(word_ids.iter().copied().filter(|id| *id != -1));
        }
        current_word_ids
    }

    pub(super) fn compute_paper_scope_metrics_from_change_metrics(
        &self,
        scope: &HktScopeSnapshot,
        _mapped: &MappedHktScopeState,
        stage: &HktUpdateStage,
        changes: &BTreeMap<i32, WordChangeMetrics>,
    ) -> Result<PaperScopeMetrics, SecaError> {
        let words = self.build_current_word_ids(scope, stage);
        let mut alpha = 0.0;
        let mut beta = 0.0;
        let mut importance = 0.0;
        for word in &words {
            if let Some(m) = changes.get(word) {
                alpha += (self.config.seca_thresholds.alpha - m.strength).max(0.0);
                beta += (self.config.seca_thresholds.beta - m.eligibility).max(0.0);
                importance += m.importance;
            } else {
                alpha += self.config.seca_thresholds.alpha;
                beta += self.config.seca_thresholds.beta;
            }
        }
        let n = words.len() as f64;
        Ok(PaperScopeMetrics {
            alpha_error: Some(if n == 0.0 { 0.0 } else { alpha / n }),
            beta_error: Some(if n == 0.0 { 0.0 } else { beta / n }),
            word_importance_error: Some((1.0 - importance).clamp(0.0, 1.0)),
        })
    }
}
