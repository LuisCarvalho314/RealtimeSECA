use super::scope_mapping::{HktScopeSnapshot, MappedHktScopeState};
use super::*;
use std::collections::{BTreeMap, BTreeSet};

impl SecaEngine {
    /// Inspect parent scopes before children; stop at each reconstruction target.
    pub(crate) fn evaluate_seca_trigger_plan_for_batch(
        &mut self,
        batch: &SourceBatch,
    ) -> Result<RecursiveTriggerPlan, SecaError> {
        let hkt_build_output =
            self.hkt_build_output
                .as_ref()
                .ok_or_else(|| SecaError::StateError {
                    message: "cannot evaluate trigger plan: baseline tree missing".to_string(),
                })?;

        let root_hkt_id = hkt_build_output
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id == 0)
            .map(|hkt| hkt.hkt_id)
            .ok_or_else(|| SecaError::StateError {
                message: "cannot evaluate trigger plan: root HKT not found".to_string(),
            })?;

        let retained_words = self.build_baseline_source_word_sets()?;
        let all_source_indexes: BTreeSet<usize> = (0..batch.sources.len()).collect();

        let mut plan = RecursiveTriggerPlan::default();

        self.evaluate_hkt_recursively_for_batch(
            batch,
            root_hkt_id,
            &all_source_indexes,
            &AncestorContext::default(),
            &mut plan,
            &retained_words,
        )?;

        Ok(plan)
    }

    fn evaluate_hkt_recursively_for_batch(
        &mut self,
        batch: &SourceBatch,
        hkt_id: i32,
        scoped_batch_source_indexes: &BTreeSet<usize>,
        ancestor: &AncestorContext,
        plan: &mut RecursiveTriggerPlan,
        retained_words: &BTreeMap<i64, BTreeSet<i32>>,
    ) -> Result<(), SecaError> {
        if scoped_batch_source_indexes.is_empty() {
            return Ok(());
        }
        let scope_snapshot = self.snapshot_hkt_scope(hkt_id)?;
        let mapped_scope =
            self.map_batch_into_hkt_scope(batch, scoped_batch_source_indexes, &scope_snapshot)?;
        let update_stage = self.compute_update_stage_for_scope(
            batch,
            &scope_snapshot,
            &mapped_scope,
            ancestor,
            retained_words,
        )?;
        let change_metrics = self.compute_word_change_metrics_for_scope(
            &scope_snapshot,
            &mapped_scope,
            &update_stage,
            retained_words,
        )?;
        // Metrics refer to state0 words; mapping must not change that vocabulary.
        let decision = self.evaluate_scope_trigger_decision(
            &scope_snapshot,
            &mapped_scope,
            &update_stage,
            &change_metrics,
        )?;

        plan.diagnostics.push(decision.clone());

        if decision.should_reconstruct {
            // One visit per scope; ordered keys also preserve reconstruction allocation order.
            plan.targets.insert(
                hkt_id,
                ReconstructionTarget {
                    source_indexes: mapped_scope.scoped_batch_source_indexes,
                    excluded_words: ancestor.clone(),
                },
            );

            // Per paper: stop descending below a triggered HKT.
            return Ok(());
        }

        // Children see the parent-adopted context.
        self.apply_state1_updates_for_hkt(hkt_id, &update_stage)?;

        let mut child_hkt_ids: Vec<(i32, BTreeSet<usize>, i32)> = Vec::new();

        let mut candidate_parent_node_ids = scope_snapshot.non_refuge_node_ids.clone();
        if let Some(refuge_node_id) = scope_snapshot.refuge_node_id {
            candidate_parent_node_ids.push(refuge_node_id);
        }
        candidate_parent_node_ids.sort_unstable();
        candidate_parent_node_ids.dedup();

        for node_id in candidate_parent_node_ids {
            if let Some(child_hkt_id) = scope_snapshot.child_hkt_ids_by_parent_node_id.get(&node_id)
            {
                let child_scope = mapped_scope
                    .matched_node_source_indexes_by_node_id
                    .get(&node_id)
                    .cloned()
                    .unwrap_or_default();

                if !child_scope.is_empty() {
                    child_hkt_ids.push((*child_hkt_id, child_scope, node_id));
                }
            }
        }

        child_hkt_ids.sort_by_key(|(child_hkt_id, _, _)| *child_hkt_id);

        for (child_hkt_id, child_scope, parent_node_id) in child_hkt_ids {
            let child_ancestor = self.build_child_ancestor_context_from_parent_node(
                &scope_snapshot,
                parent_node_id,
                ancestor,
            );

            self.evaluate_hkt_recursively_for_batch(
                batch,
                child_hkt_id,
                &child_scope,
                &child_ancestor,
                plan,
                retained_words,
            )?;
        }

        // Commit mapped memberships after processing child scopes.
        self.merge_state1_updates_for_hkt(hkt_id)?;

        Ok(())
    }

    fn trigger_reasons(&self, metrics: &PaperScopeMetrics, source_count: usize) -> Vec<String> {
        if source_count == 0 {
            return Vec::new();
        }
        let thresholds = &self.config.seca_thresholds;
        [
            (
                "word_importance_error",
                metrics.word_importance_error,
                thresholds.word_importance_error_threshold,
            ),
            (
                "alpha_error",
                metrics.alpha_error,
                thresholds.alpha_error_threshold,
            ),
            (
                "beta_error",
                metrics.beta_error,
                thresholds.beta_error_threshold,
            ),
        ]
        .into_iter()
        .filter_map(|(name, value, threshold)| {
            value
                .filter(|value| *value > threshold)
                .map(|value| format!("{name} {value:.4} > threshold {threshold:.4}"))
        })
        .collect()
    }

    pub(super) fn evaluate_scope_trigger_decision(
        &self,
        scope: &HktScopeSnapshot,
        mapped: &MappedHktScopeState,
        stage: &HktUpdateStage,
        changes: &BTreeMap<i32, WordChangeMetrics>,
    ) -> Result<crate::types::HktDecisionDiagnostics, SecaError> {
        let metrics =
            self.compute_paper_scope_metrics_from_change_metrics(scope, mapped, stage, changes)?;
        let reasons = self.trigger_reasons(&metrics, mapped.scoped_batch_source_indexes.len());
        Ok(crate::types::HktDecisionDiagnostics {
            hkt_id: scope.hkt_id,
            output_hkt_id: Some(scope.hkt_id),
            scoped_source_count: mapped.scoped_batch_source_indexes.len(),
            mapped_source_count: mapped
                .matched_node_source_indexes_by_node_id
                .iter()
                .filter(|(id, _)| scope.non_refuge_node_ids.contains(id))
                .flat_map(|(_, ids)| ids.iter().copied())
                .collect::<BTreeSet<_>>()
                .len(),
            should_reconstruct: !reasons.is_empty(),
            trigger_reasons: reasons,
            paper_alpha_error: metrics.alpha_error,
            paper_beta_error: metrics.beta_error,
            paper_word_importance_error: metrics.word_importance_error,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct RecursiveTriggerPlan {
    pub(crate) diagnostics: Vec<crate::types::HktDecisionDiagnostics>,
    pub(crate) targets: BTreeMap<i32, ReconstructionTarget>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ReconstructionTarget {
    pub(crate) source_indexes: BTreeSet<usize>,
    pub(crate) excluded_words: BTreeSet<i32>,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub(super) struct PaperScopeMetrics {
    pub(super) alpha_error: Option<f64>,
    pub(super) beta_error: Option<f64>,
    pub(super) word_importance_error: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct HktUpdateStage {
    pub(super) updated_expected_word_ids: BTreeSet<i32>,
    pub(super) assigned_expected_words_by_node_id: BTreeMap<i32, BTreeSet<i32>>,
    pub(super) node_state1_source_ids_by_node_id: BTreeMap<i32, BTreeSet<i64>>,
    pub(super) node_state1_sources_by_word_id: BTreeMap<i32, BTreeMap<i32, BTreeSet<i64>>>,
    pub(super) token_by_word_id: BTreeMap<i32, String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct WordChangeMetrics {
    pub(crate) strength: f64,
    pub(crate) eligibility: f64,
    pub(crate) importance: f64,
}

// Only descriptors on attachment nodes are excluded; this is not E05's broader exclusion.
pub(super) type AncestorContext = BTreeSet<i32>;
