use super::trigger::*;
use super::*;
impl SecaEngine {
    pub fn display_diagnostics(
        &self,
        incoming: Option<&SourceBatch>,
    ) -> Result<Vec<crate::types::HktDisplayDiagnostics>, SecaError> {
        let Some(final_tree) = self.hkt_build_output.as_ref() else {
            return Ok(Vec::new());
        };
        let counts: BTreeMap<i32, usize> = final_tree
            .hkts_by_id
            .iter()
            .map(|(id, hkt)| {
                let members: BTreeSet<_> = hkt
                    .nodes
                    .iter()
                    .flat_map(|n| n.source_ids.iter().copied())
                    .filter(|id| self.baseline_source_legend.contains_key(id))
                    .collect();
                (*id, members.len())
            })
            .collect();
        let mut view = self.clone();
        let empty = SourceBatch {
            batch_index: 0,
            sources: Vec::new(),
        };
        let batch = incoming.unwrap_or(&empty);
        let incoming_ids: BTreeSet<_> = batch
            .sources
            .iter()
            .filter_map(|s| self.source_id_by_url.get(&s.source_id).copied())
            .collect();
        view.processed_batches.iter_mut().for_each(|b| {
            b.sources.retain(|s| {
                !view
                    .source_id_by_url
                    .get(&s.source_id)
                    .is_some_and(|id| incoming_ids.contains(id))
            })
        });
        if let Some(tree) = view.hkt_build_output.as_mut() {
            for node in tree.nodes_by_id.values_mut() {
                node.source_ids.retain(|id| !incoming_ids.contains(id));
                for ids in node.word_source_ids.values_mut() {
                    ids.retain(|id| !incoming_ids.contains(id));
                }
            }
            for hkt in tree.hkts_by_id.values_mut() {
                for node in &mut hkt.nodes {
                    *node = tree.nodes_by_id[&node.node_id].clone();
                }
            }
        }
        let retained_words = view.build_baseline_source_word_sets()?;
        let mut result = Vec::new();
        for root in view
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .values()
            .filter(|h| h.parent_node_id == 0)
        {
            view.display_scope_recursive(
                batch,
                root.hkt_id,
                &(0..batch.sources.len()).collect(),
                &AncestorContext::default(),
                incoming.is_some(),
                &counts,
                &mut result,
                &retained_words,
            )?;
        }
        result.sort_by_key(|d| d.hkt_id);
        Ok(result)
    }
    #[allow(clippy::too_many_arguments)] // Recursive observation inputs plus two output accumulators.
    pub(super) fn display_scope_recursive(
        &self,
        batch: &SourceBatch,
        id: i32,
        indexes: &BTreeSet<usize>,
        ancestor: &AncestorContext,
        comparison: bool,
        counts: &BTreeMap<i32, usize>,
        result: &mut Vec<crate::types::HktDisplayDiagnostics>,
        retained_words: &BTreeMap<i64, BTreeSet<i32>>,
    ) -> Result<(), SecaError> {
        let scope = self.snapshot_hkt_scope(id)?;
        let mapped = self.map_batch_into_hkt_scope(batch, indexes, &scope)?;
        let words = self.build_current_word_ids(&scope, &HktUpdateStage::default());
        let has_words = !words.is_empty();
        // Refuge-only evidence does not support a current-word measurement.
        let has_evidence = words.iter().any(|word| {
            scope
                .node_word_source_ids_by_node_id
                .values()
                .any(|by_word| by_word.get(word).is_some_and(|sources| !sources.is_empty()))
                || self.baseline_word_legend.get(word).is_some_and(|token| {
                    mapped
                        .word_document_frequency_in_scope
                        .get(token)
                        .copied()
                        .unwrap_or(0)
                        > 0
                })
        });
        let metrics = if comparison && has_words && has_evidence {
            let stage = self.compute_update_stage_for_scope(
                batch,
                &scope,
                &mapped,
                ancestor,
                retained_words,
            )?;
            let changes = self.compute_word_change_metrics_for_scope(
                &scope,
                &mapped,
                &stage,
                retained_words,
            )?;
            Some(self.compute_paper_scope_metrics_from_change_metrics(
                &scope, &mapped, &stage, &changes,
            )?)
        } else {
            None
        };
        result.push(crate::types::HktDisplayDiagnostics {
            hkt_id: id,
            mapped_source_count: counts[&id],
            paper_alpha_error: metrics.as_ref().and_then(|m| m.alpha_error),
            paper_beta_error: metrics.as_ref().and_then(|m| m.beta_error),
            paper_word_importance_error: metrics.as_ref().and_then(|m| m.word_importance_error),
            unavailable_reason: if !comparison {
                Some("baseline_no_comparison".into())
            } else if !has_words {
                Some("no_current_words".into())
            } else if !has_evidence {
                Some("no_applicable_evidence".into())
            } else {
                None
            },
        });
        for (parent_node, child) in &scope.child_hkt_ids_by_parent_node_id {
            let child_indexes = mapped
                .matched_node_source_indexes_by_node_id
                .get(parent_node)
                .cloned()
                .unwrap_or_default();
            let context =
                self.build_child_ancestor_context_from_parent_node(&scope, *parent_node, ancestor);
            self.display_scope_recursive(
                batch,
                *child,
                &child_indexes,
                &context,
                comparison,
                counts,
                result,
                retained_words,
            )?;
        }
        Ok(())
    }
}
