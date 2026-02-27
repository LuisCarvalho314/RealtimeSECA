use super::scope_mapping::{HktScopeSnapshot, MappedHktScopeState};
use super::*;
use crate::config::TriggerPolicyMode;
use crate::engine::rebuild::SelectedHktRebuildPlan;
use std::collections::{BTreeMap, BTreeSet};

impl SecaEngine {
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

        let all_source_indexes: BTreeSet<usize> = (0..batch.sources.len()).collect();

        let mut plan = RecursiveTriggerPlan {
            batch_index: batch.batch_index,
            any_reconstruction_triggered: false,
            reconstruct_hkt_ids: Vec::new(),
            reconstruct_scopes_by_hkt_id: BTreeMap::new(),
            notes: vec!["SECA recursive trigger evaluation started".to_string()],
            ancestor_words_by_hkt_id: BTreeMap::new(),
            ancestor_accepted_words_by_hkt_id: BTreeMap::new(),
            ancestor_rejected_words_by_hkt_id: BTreeMap::new(),
        };

        self.evaluate_hkt_recursively_for_batch(
            batch,
            root_hkt_id,
            &all_source_indexes,
            &AncestorContext::default(),
            &mut plan,
        )?;

        // Keep deterministic order for downstream planning/notes/tests
        plan.reconstruct_hkt_ids.sort_unstable();
        plan.reconstruct_hkt_ids.dedup();

        if plan.any_reconstruction_triggered {
            plan.notes.push(format!(
                "SECA trigger plan: reconstruction requested for HKTs {:?}",
                plan.reconstruct_hkt_ids
            ));
        } else {
            plan.notes
                .push("SECA trigger plan: no reconstruction requested".to_string());
        }

        Ok(plan)
    }

    fn evaluate_hkt_recursively_for_batch(
        &mut self,
        batch: &SourceBatch,
        hkt_id: i32,
        scoped_batch_source_indexes: &BTreeSet<usize>,
        ancestor: &AncestorContext,
        plan: &mut RecursiveTriggerPlan,
    ) -> Result<(), SecaError> {
        let scope_snapshot = self.snapshot_hkt_scope(hkt_id)?;
        let mapped_scope =
            self.map_batch_into_hkt_scope(batch, scoped_batch_source_indexes, &scope_snapshot)?;
        let update_stage =
            self.compute_update_stage_for_scope(batch, &scope_snapshot, &mapped_scope, ancestor)?;
        let change_metrics =
            if self.config.trigger_policy_mode == TriggerPolicyMode::PaperDiagnosticScaffold {
                Some(self.compute_word_change_metrics_for_scope(
                    &scope_snapshot,
                    &mapped_scope,
                    &update_stage,
                )?)
            } else {
                None
            };
        let scope_snapshot = update_stage.apply_to_snapshot(&scope_snapshot);
        let decision = self.evaluate_scope_trigger_decision(
            &scope_snapshot,
            &mapped_scope,
            &update_stage,
            change_metrics.as_ref(),
        )?;

        plan.notes.push(format!(
            "HKT {}: scoped_sources={}, matched_nodes={}, known_words_in_scope={}, new_tokens_in_scope={}",
            hkt_id,
            mapped_scope.scoped_batch_source_indexes.len(),
            mapped_scope.matched_node_source_indexes_by_node_id.len(),
            mapped_scope.known_word_ids_in_scope.len(),
            mapped_scope.new_tokens_in_scope.len()
        ));
        if !update_stage.updated_expected_word_ids.is_empty() {
            plan.notes.push(format!(
                "HKT {} update stage: expected_words_state1={}",
                hkt_id,
                update_stage.updated_expected_word_ids.len()
            ));
        }

        if let Some(word_importance_error) = decision.word_importance_error {
            plan.notes.push(format!(
                "HKT {} metrics (placeholder): word_importance_error={:.4}",
                decision.hkt_id, word_importance_error
            ));
        }

        if let (Some(alpha_error), Some(beta_error)) = (decision.alpha_error, decision.beta_error) {
            plan.notes.push(format!(
                "HKT {} metrics (placeholder): alpha_error={:.4}, beta_error={:.4}",
                decision.hkt_id, alpha_error, beta_error
            ));
        }

        if let Some(paper_wi_error) = decision.paper_word_importance_error {
            plan.notes.push(format!(
                "HKT {} metrics (paper diagnostic): wi_error_eq12={:.4}",
                decision.hkt_id, paper_wi_error
            ));
        }

        if let Some(paper_alpha_error) = decision.paper_alpha_error {
            plan.notes.push(format!(
                "HKT {} metrics (paper diagnostic): alpha_error_eq6_scaffold={:.4}",
                decision.hkt_id, paper_alpha_error
            ));
        }

        if let Some(paper_beta_error) = decision.paper_beta_error {
            plan.notes.push(format!(
                "HKT {} metrics (paper diagnostic): beta_error_eq9_scaffold={:.4}",
                decision.hkt_id, paper_beta_error
            ));
        }

        if let Some(policy_label) = &decision.active_trigger_policy_label {
            plan.notes.push(format!(
                "HKT {} active trigger policy: {}",
                decision.hkt_id, policy_label
            ));
        }

        plan.notes.push(format!(
            "HKT {} ancestor context: accepted={}, rejected={}",
            hkt_id,
            ancestor.ancestor_accepted_words.len(),
            ancestor.ancestor_rejected_words.len(),
        ));

        plan.ancestor_accepted_words_by_hkt_id
            .insert(hkt_id, ancestor.ancestor_accepted_words.clone());
        plan.ancestor_rejected_words_by_hkt_id
            .insert(hkt_id, ancestor.ancestor_rejected_words.clone());
        plan.ancestor_words_by_hkt_id
            .insert(hkt_id, ancestor.ancestor_words.clone());

        let current_word_count = self
            .build_current_word_ids(&scope_snapshot, &update_stage)
            .len();
        let selected_thresholds = self.compute_paper_selected_thresholds(current_word_count);
        let paper_shadow_trigger_reasons = self.compute_paper_shadow_trigger_reasons(
            decision.paper_word_importance_error,
            decision.paper_alpha_error,
            decision.paper_beta_error,
            mapped_scope.scoped_batch_source_indexes.len(),
            selected_thresholds.word_importance_error_threshold,
            selected_thresholds.alpha_error_threshold,
            selected_thresholds.beta_error_threshold,
        );

        if self.config.trigger_policy_mode == TriggerPolicyMode::PaperDiagnosticScaffold {
            let paper_inputs = self.compute_paper_scope_diagnostic_inputs_from_change_metrics(
                &scope_snapshot,
                &mapped_scope,
                &update_stage,
                change_metrics
                    .as_ref()
                    .expect("change metrics must exist in paper mode"),
            );
            plan.notes
                .push(self.format_paper_scope_diagnostic_inputs_note(hkt_id, &paper_inputs));

            let beta_node_diagnostics =
                self.compute_paper_beta_node_diagnostics(&scope_snapshot, &mapped_scope);
            plan.notes
                .push(self.format_paper_beta_node_diagnostics_note(hkt_id, &beta_node_diagnostics));
        }

        plan.notes.push(format!(
            "HKT {} paper-policy shadow: would_trigger={}",
            decision.hkt_id,
            !paper_shadow_trigger_reasons.is_empty()
        ));

        for shadow_reason in &paper_shadow_trigger_reasons {
            plan.notes.push(format!(
                "HKT {} paper-policy shadow reason: {}",
                decision.hkt_id, shadow_reason
            ));
        }

        if decision.should_reconstruct {
            plan.any_reconstruction_triggered = true;

            if !plan.reconstruct_hkt_ids.contains(&hkt_id) {
                plan.reconstruct_hkt_ids.push(hkt_id);
            }

            // Preserve the exact scoped provenance that caused the trigger.
            plan.reconstruct_scopes_by_hkt_id
                .entry(hkt_id)
                .and_modify(|existing_scope| {
                    existing_scope.extend(mapped_scope.scoped_batch_source_indexes.iter().copied());
                })
                .or_insert_with(|| mapped_scope.scoped_batch_source_indexes.clone());

            for trigger_reason in &decision.trigger_reasons {
                plan.notes
                    .push(format!("HKT {} trigger reason: {}", hkt_id, trigger_reason));
            }

            // Per paper: stop descending below a triggered HKT.
            return Ok(());
        }

        // C#-parity: apply state1 mappings + expected words for accepted HKTs before descending.
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
                batch,
                &scope_snapshot,
                &mapped_scope,
                hkt_id,
                parent_node_id,
                ancestor,
            );

            self.evaluate_hkt_recursively_for_batch(
                batch,
                child_hkt_id,
                &child_scope,
                &child_ancestor,
                plan,
            )?;
        }

        // C#-parity: merge mapped sources into node state after child recursion completes.
        self.merge_state1_updates_for_hkt(hkt_id)?;

        Ok(())
    }

    fn apply_state1_updates_for_hkt(
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
            node.word_ids_new_from_batches.clear();

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

                    if let Some(assigned_expected_words) = update_stage
                        .assigned_expected_words_by_node_id
                        .get(&node.node_id)
                    {
                        for word_id in assigned_expected_words {
                            node.word_ids_new_from_batches.insert(*word_id);
                            if let Some(state1_sources) = state1_sources_by_word.get(word_id) {
                                node.word_source_ids_new_from_batches
                                    .entry(*word_id)
                                    .or_default()
                                    .extend(state1_sources.iter().copied());
                                node.source_ids_new_from_batches
                                    .extend(state1_sources.iter().copied());
                            }
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

    fn merge_state1_updates_for_hkt(&mut self, hkt_id: i32) -> Result<(), SecaError> {
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
            let pending_word_sources: Vec<(i32, BTreeSet<i64>)> = node
                .word_source_ids_new_from_batches
                .iter()
                .map(|(word_id, sources)| (*word_id, sources.clone()))
                .collect();

            for (word_id, sources) in pending_word_sources {
                node.word_source_ids
                    .entry(word_id)
                    .or_default()
                    .extend(sources);
            }

            node.source_ids
                .extend(node.source_ids_new_from_batches.iter().copied());

            node.source_ids_new_from_batches.clear();
            node.word_source_ids_new_from_batches.clear();
            node.word_ids_new_from_batches.clear();

            if let Some(global_node) = hkt_build_output.nodes_by_id.get_mut(&node.node_id) {
                *global_node = node.clone();
            }
        }

        Ok(())
    }

    pub(crate) fn evaluate_scope_trigger_decision_placeholder(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        mapped_scope: &MappedHktScopeState,
    ) -> Result<HktTriggerDecisionInternal, SecaError> {
        let scoped_source_count = mapped_scope.scoped_batch_source_indexes.len();
        let metrics = self.compute_scope_placeholder_metrics(mapped_scope);

        // Keep paper diagnostics available for notes/inspection, but do NOT use them
        // to drive placeholder trigger behavior.
        let paper_word_importance_error =
            self.compute_paper_word_importance_error_diagnostic(scope_snapshot, mapped_scope);
        let paper_alpha_error =
            self.compute_paper_alpha_error_diagnostic(scope_snapshot, mapped_scope);
        let paper_beta_error =
            self.compute_paper_beta_error_diagnostic(scope_snapshot, mapped_scope);

        let trigger_reasons =
            self.compute_placeholder_trigger_reasons(scoped_source_count, &metrics);
        let should_reconstruct = !trigger_reasons.is_empty();

        Ok(HktTriggerDecisionInternal {
            hkt_id: scope_snapshot.hkt_id,
            should_reconstruct,
            trigger_reasons,
            active_trigger_policy_label: Some("placeholder".to_string()),

            // Placeholder metrics (active for this function)
            alpha_error: Some(metrics.alpha_error),
            beta_error: Some(metrics.beta_error),
            word_importance_error: Some(metrics.word_importance_error),

            // Paper diagnostics remain diagnostic only here
            paper_word_importance_error,
            paper_alpha_error,
            paper_beta_error,
        })
    }

    fn compute_scope_placeholder_metrics(
        &self,
        mapped_scope: &MappedHktScopeState,
    ) -> ScopePlaceholderMetrics {
        let thresholds = &self.config.seca_thresholds;

        let (known_count, new_count) = self.resolve_placeholder_scope_vocab_counts(mapped_scope);

        let known_word_count = known_count as f64;
        let new_token_count = new_count as f64;
        let total_vocab_in_scope = known_word_count + new_token_count;

        let word_importance_error = if total_vocab_in_scope > 0.0 {
            new_token_count / total_vocab_in_scope
        } else {
            0.0
        };

        let alpha_estimate = if total_vocab_in_scope > 0.0 {
            known_word_count / total_vocab_in_scope
        } else {
            0.0
        };

        let beta_estimate = if total_vocab_in_scope > 0.0 {
            new_token_count / total_vocab_in_scope
        } else {
            0.0
        };

        ScopePlaceholderMetrics {
            alpha_error: (alpha_estimate - thresholds.alpha).abs(),
            beta_error: (beta_estimate - thresholds.beta).abs(),
            word_importance_error,
        }
    }

    fn compute_paper_word_importance_error_diagnostic(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        mapped_scope: &MappedHktScopeState,
    ) -> Option<f64> {
        self.compute_paper_scope_metrics(scope_snapshot, mapped_scope)
            .word_importance_error_option1
    }

    fn compute_paper_alpha_error_diagnostic(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        mapped_scope: &MappedHktScopeState,
    ) -> Option<f64> {
        self.compute_paper_scope_metrics(scope_snapshot, mapped_scope)
            .alpha_error_option1
    }

    pub(crate) fn compute_paper_beta_error_diagnostic(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        mapped_scope: &MappedHktScopeState,
    ) -> Option<f64> {
        let thresholds = &self.config.seca_thresholds;

        let mut total_violation_deviation = 0.0_f64;
        let mut total_state0_words_count = 0_usize;

        // Evaluate all nodes in the targeted HKT (non-refuge + refuge if present)
        let mut node_ids: Vec<i32> = scope_snapshot.non_refuge_node_ids.clone();
        if let Some(refuge_node_id) = scope_snapshot.refuge_node_id {
            node_ids.push(refuge_node_id);
        }
        node_ids.sort_unstable();

        for node_id in node_ids {
            let state0_node_word_ids = match scope_snapshot.node_word_ids_by_node_id.get(&node_id) {
                Some(word_ids) => word_ids,
                None => continue,
            };

            total_state0_words_count += state0_node_word_ids.len();

            // CountSources(Sources(z)) in state1 = number of mapped sources assigned to this node
            let state1_node_source_count = mapped_scope
                .matched_node_source_indexes_by_node_id
                .get(&node_id)
                .map(|sources| sources.len())
                .unwrap_or(0);

            // No mapped sources in this node in state1 => all eligibilities are zero
            if state1_node_source_count == 0 {
                for _word_id in state0_node_word_ids {
                    if thresholds.beta > 0.0 {
                        total_violation_deviation += thresholds.beta;
                    }
                }
                continue;
            }

            // Need CountSources(Sources(w) ∩ Sources(z)) in state1 for each state0 node word.
            let node_local_df_map = mapped_scope
                .word_document_frequency_by_node_id
                .get(&node_id);

            for word_id in state0_node_word_ids {
                let elig1 = if let Some(token) = self.baseline_word_legend.get(word_id) {
                    let node_local_df = node_local_df_map
                        .and_then(|m| m.get(token))
                        .copied()
                        .unwrap_or(0);

                    (node_local_df as f64) / (state1_node_source_count as f64)
                } else {
                    0.0
                }
                .clamp(0.0, 1.0);

                if elig1 < thresholds.beta {
                    total_violation_deviation += (thresholds.beta - elig1).abs();
                }
            }
        }

        if total_state0_words_count == 0 {
            Some(0.0)
        } else {
            Some((total_violation_deviation / (total_state0_words_count as f64)).clamp(0.0, 1.0))
        }
    }

    fn compute_paper_shadow_trigger_reasons(
        &self,
        paper_word_importance_error: Option<f64>,
        paper_alpha_error: Option<f64>,
        paper_beta_error: Option<f64>,
        scoped_source_count: usize,
        word_importance_error_threshold: f64,
        alpha_error_threshold: f64,
        beta_error_threshold: f64,
    ) -> Vec<String> {
        let mut trigger_reasons = Vec::new();

        if let Some(wi_error) = paper_word_importance_error {
            if wi_error > word_importance_error_threshold {
                trigger_reasons.push(format!(
                    "paper_shadow.word_importance_error {:.4} > threshold {:.4}",
                    wi_error, word_importance_error_threshold
                ));
            }
        }

        if let Some(alpha_error) = paper_alpha_error {
            if alpha_error > alpha_error_threshold {
                trigger_reasons.push(format!(
                    "paper_shadow.alpha_error {:.4} > threshold {:.4}",
                    alpha_error, alpha_error_threshold
                ));
            }
        }

        if let Some(beta_error) = paper_beta_error {
            if beta_error > beta_error_threshold {
                trigger_reasons.push(format!(
                    "paper_shadow.beta_error {:.4} > threshold {:.4}",
                    beta_error, beta_error_threshold
                ));
            }
        }

        if scoped_source_count <= 1 && !trigger_reasons.is_empty() {
            trigger_reasons.push("paper_shadow.tiny scoped batch (<=1 source)".to_string());
        }

        trigger_reasons
    }

    fn compute_placeholder_trigger_reasons(
        &self,
        scoped_source_count: usize,
        metrics: &ScopePlaceholderMetrics,
    ) -> Vec<String> {
        let thresholds = &self.config.seca_thresholds;
        let mut trigger_reasons = Vec::new();

        if metrics.word_importance_error > thresholds.word_importance_error_threshold {
            trigger_reasons.push(format!(
                "word_importance_error {:.4} > threshold {:.4} (HKT-local placeholder)",
                metrics.word_importance_error, thresholds.word_importance_error_threshold
            ));
        }

        if scoped_source_count <= 1 && !trigger_reasons.is_empty() {
            trigger_reasons.push("tiny scoped batch (<=1 source)".to_string());
        }

        trigger_reasons
    }

    pub(super) fn compute_update_stage_for_scope(
        &self,
        batch: &SourceBatch,
        scope_snapshot: &HktScopeSnapshot,
        mapped_scope: &MappedHktScopeState,
        ancestor: &AncestorContext,
    ) -> Result<HktUpdateStage, SecaError> {
        let mut updated_expected_word_ids = scope_snapshot.expected_word_ids.clone();
        let mut token_to_word_id: BTreeMap<String, i32> = BTreeMap::new();
        let mut word_id_to_token: BTreeMap<i32, String> = BTreeMap::new();
        let mut next_synthetic_word_id: i32 = -2;
        for (word_id, token) in &self.baseline_word_legend {
            token_to_word_id.insert(token.clone(), *word_id);
            word_id_to_token.insert(*word_id, token.clone());
        }

        let alpha_threshold = self.config.seca_thresholds.alpha;
        let beta_threshold = self.config.seca_thresholds.beta;

        let mut current_words_in_hkt: BTreeSet<i32> = BTreeSet::new();
        for word_ids in scope_snapshot.node_word_ids_by_node_id.values() {
            for word_id in word_ids {
                if *word_id != -1 {
                    current_words_in_hkt.insert(*word_id);
                }
            }
        }
        current_words_in_hkt.extend(ancestor.ancestor_accepted_words.keys().copied());

        let baseline_source_word_sets = self.build_baseline_source_word_sets()?;
        let state0_df_parent_scope =
            self.build_state0_df_in_parent_scope(scope_snapshot, &baseline_source_word_sets)?;
        let node_state0_sources_by_word_id =
            self.build_node_state0_sources_by_word_id(scope_snapshot, &baseline_source_word_sets);

        let hkt_build_output =
            self.hkt_build_output
                .as_ref()
                .ok_or_else(|| SecaError::StateError {
                    message: "baseline tree missing for expected-word admission".to_string(),
                })?;

        let (prominent_node_id, prominent_word_id) = hkt_build_output
            .hkts_by_id
            .get(&scope_snapshot.hkt_id)
            .and_then(|hkt| hkt.nodes.first())
            .and_then(|node| {
                node.word_ids
                    .iter()
                    .next()
                    .map(|word_id| (node.node_id, *word_id))
            })
            .unwrap_or((0, -1));

        let state0_prominent_count = node_state0_sources_by_word_id
            .get(&prominent_node_id)
            .and_then(|by_word| by_word.get(&prominent_word_id))
            .map(|sources| sources.len())
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
            let state0_count = state0_df_parent_scope.get(&word_id).copied().unwrap_or(0);
            let ratio = (state0_count + *state1_count) as f64 / (state0_prominent_count as f64);
            if ratio >= alpha_threshold {
                updated_expected_word_ids.insert(word_id);
            }
        }

        if !ancestor.ancestor_rejected_words.is_empty() {
            for (rejected_word_id, (state0_count, state1_count)) in
                &ancestor.ancestor_rejected_words
            {
                let ratio =
                    (*state0_count + *state1_count) as f64 / (state0_prominent_count as f64);
                if ratio >= alpha_threshold {
                    updated_expected_word_ids.insert(*rejected_word_id);
                }
            }
        }

        let mut normalized_tokens_by_source_index: BTreeMap<usize, BTreeSet<String>> =
            BTreeMap::new();
        for source_index in &mapped_scope.scoped_batch_source_indexes {
            let source = batch
                .sources
                .get(*source_index)
                .ok_or_else(|| SecaError::StateError {
                    message: format!(
                        "batch source index {} out of bounds during update stage",
                        source_index
                    ),
                })?;
            let mut tokens = BTreeSet::new();
            for token in &source.tokens {
                let normalized = token.trim();
                if normalized.is_empty() {
                    continue;
                }
                tokens.insert(normalized.to_string());
            }
            normalized_tokens_by_source_index.insert(*source_index, tokens);
        }

        let mut node_state1_source_ids_by_node_id: BTreeMap<i32, BTreeSet<i64>> = BTreeMap::new();
        let mut node_state1_sources_by_word_id: BTreeMap<i32, BTreeMap<i32, BTreeSet<i64>>> =
            BTreeMap::new();
        for (node_id, source_indexes) in &mapped_scope.matched_node_source_indexes_by_node_id {
            let mut source_ids = BTreeSet::new();
            for source_index in source_indexes {
                let source =
                    batch
                        .sources
                        .get(*source_index)
                        .ok_or_else(|| SecaError::StateError {
                            message: format!(
                                "batch source index {} out of bounds during update stage",
                                source_index
                            ),
                        })?;
                let internal_source_id = self
                    .source_id_by_url
                    .get(source.source_id.as_str())
                    .copied()
                    .unwrap_or_else(|| SecaEngine::fnv1a_64(source.source_id.as_str()));
                source_ids.insert(internal_source_id);

                if let Some(tokens) = normalized_tokens_by_source_index.get(source_index) {
                    for token in tokens {
                        let word_id = token_to_word_id.get(token).copied().unwrap_or_else(|| {
                            let assigned = next_synthetic_word_id;
                            next_synthetic_word_id -= 1;
                            token_to_word_id.insert(token.clone(), assigned);
                            word_id_to_token.insert(assigned, token.clone());
                            assigned
                        });
                        node_state1_sources_by_word_id
                            .entry(*node_id)
                            .or_default()
                            .entry(word_id)
                            .or_default()
                            .insert(internal_source_id);
                    }
                }
            }
            node_state1_source_ids_by_node_id.insert(*node_id, source_ids);
        }

        let mut expected_total_counts: BTreeMap<i32, usize> = BTreeMap::new();
        for word_id in &updated_expected_word_ids {
            let state0_count = state0_df_parent_scope.get(word_id).copied().unwrap_or(0);
            let state1_count = word_id_to_token
                .get(word_id)
                .and_then(|token| mapped_scope.word_document_frequency_in_scope.get(token))
                .copied()
                .unwrap_or(0);
            expected_total_counts.insert(*word_id, state0_count + state1_count);
        }

        let mut prominent_expected_count = 1usize;

        for (node_id, word_ids) in &scope_snapshot.node_word_ids_by_node_id {
            if word_ids.contains(&-1) {
                continue;
            }
            for word_id in word_ids.iter().copied().filter(|id| *id != -1) {
                let state0_count = node_state0_sources_by_word_id
                    .get(node_id)
                    .and_then(|by_word| by_word.get(&word_id))
                    .map(|sources| sources.len())
                    .unwrap_or(0);
                let state1_count = node_state1_sources_by_word_id
                    .get(node_id)
                    .and_then(|by_word| by_word.get(&word_id))
                    .map(|sources| sources.len())
                    .unwrap_or(0);
                let total = state0_count + state1_count;
                if total > prominent_expected_count {
                    prominent_expected_count = total;
                }
            }
        }

        for word_id in &updated_expected_word_ids {
            let state0_count = state0_df_parent_scope.get(word_id).copied().unwrap_or(0);
            let state1_count = word_id_to_token
                .get(word_id)
                .and_then(|token| mapped_scope.word_document_frequency_in_scope.get(token))
                .copied()
                .unwrap_or(0);
            let total = state0_count + state1_count;
            if total > prominent_expected_count {
                prominent_expected_count = total;
            }
        }

        if !expected_total_counts.is_empty() {
            updated_expected_word_ids = updated_expected_word_ids
                .into_iter()
                .filter(|word_id| {
                    let total = expected_total_counts.get(word_id).copied().unwrap_or(0);
                    (total as f64) / (prominent_expected_count as f64) >= alpha_threshold
                })
                .collect();
        }

        let mut assigned_expected_words_by_node_id: BTreeMap<i32, BTreeSet<i32>> = BTreeMap::new();

        for (node_id, _source_indexes) in &mapped_scope.matched_node_source_indexes_by_node_id {
            let node_word_ids = scope_snapshot
                .node_word_ids_by_node_id
                .get(node_id)
                .cloned()
                .unwrap_or_default();
            if node_word_ids.contains(&-1) {
                continue;
            }

            let _node_state0_sources = scope_snapshot
                .node_source_ids_by_node_id
                .get(node_id)
                .cloned()
                .unwrap_or_default();

            let mut prominent_word_sources: BTreeSet<i64> = BTreeSet::new();
            let mut max_prominent_count = 0usize;

            for word_id in node_word_ids.iter().copied().filter(|id| *id != -1) {
                let mut sources_for_word: BTreeSet<i64> = BTreeSet::new();
                if let Some(state0_sources_by_word) = node_state0_sources_by_word_id.get(node_id) {
                    if let Some(state0_sources) = state0_sources_by_word.get(&word_id) {
                        sources_for_word.extend(state0_sources.iter().copied());
                    }
                }
                if let Some(state1_sources_by_word) = node_state1_sources_by_word_id.get(node_id) {
                    if let Some(state1_sources) = state1_sources_by_word.get(&word_id) {
                        sources_for_word.extend(state1_sources.iter().copied());
                    }
                }

                let count = sources_for_word.len();
                if count > max_prominent_count {
                    max_prominent_count = count;
                    prominent_word_sources = sources_for_word;
                }
            }

            if prominent_word_sources.is_empty() {
                if let Some(node_state1_sources) = node_state1_source_ids_by_node_id.get(node_id) {
                    prominent_word_sources = node_state1_sources.clone();
                }
            }

            let denominator = prominent_word_sources.len().max(1) as f64;

            for expected_word_id in &updated_expected_word_ids {
                let sources_with_word = node_state1_sources_by_word_id
                    .get(node_id)
                    .and_then(|by_word| by_word.get(expected_word_id))
                    .cloned()
                    .unwrap_or_default();

                if sources_with_word.is_empty() {
                    continue;
                }

                let numerator = sources_with_word
                    .intersection(&prominent_word_sources)
                    .count() as f64;
                let eligibility = numerator / denominator;
                if eligibility >= beta_threshold {
                    assigned_expected_words_by_node_id
                        .entry(*node_id)
                        .or_default()
                        .insert(*expected_word_id);
                }
            }
        }

        Ok(HktUpdateStage {
            updated_expected_word_ids,
            assigned_expected_words_by_node_id,
            node_state1_source_ids_by_node_id,
            node_state1_sources_by_word_id,
            token_by_word_id: word_id_to_token,
        })
    }

    pub(super) fn compute_word_change_metrics_for_scope(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        mapped_scope: &MappedHktScopeState,
        update_stage: &HktUpdateStage,
    ) -> Result<BTreeMap<i32, WordChangeMetrics>, SecaError> {
        let alpha_threshold = self.config.seca_thresholds.alpha;
        let beta_threshold = self.config.seca_thresholds.beta;

        let baseline_source_word_sets = self.build_baseline_source_word_sets()?;
        let state0_df_parent_scope =
            self.build_state0_df_in_parent_scope(scope_snapshot, &baseline_source_word_sets)?;

        let number_of_sources_for_all_words_in_hkt_state0 = {
            let mut total = 0usize;
            for (node_id, word_ids) in &scope_snapshot.node_word_ids_by_node_id {
                if word_ids.contains(&-1) {
                    continue;
                }
                let word_source_ids = scope_snapshot
                    .node_word_source_ids_by_node_id
                    .get(node_id)
                    .cloned()
                    .unwrap_or_default();
                for word_id in word_ids.iter().copied().filter(|id| *id != -1) {
                    total += word_source_ids.get(&word_id).map(|s| s.len()).unwrap_or(0);
                }
            }
            total.max(1)
        };

        let number_of_sources_for_all_current_and_expected_words = {
            let mut total = 0usize;
            for (node_id, word_ids) in &scope_snapshot.node_word_ids_by_node_id {
                if word_ids.contains(&-1) {
                    continue;
                }
                let word_source_ids = scope_snapshot
                    .node_word_source_ids_by_node_id
                    .get(node_id)
                    .cloned()
                    .unwrap_or_default();
                let state1_sources = update_stage
                    .node_state1_sources_by_word_id
                    .get(node_id)
                    .cloned()
                    .unwrap_or_default();
                for word_id in word_ids.iter().copied().filter(|id| *id != -1) {
                    let state0_count = word_source_ids.get(&word_id).map(|s| s.len()).unwrap_or(0);
                    let state1_count = state1_sources.get(&word_id).map(|s| s.len()).unwrap_or(0);
                    total += state0_count + state1_count;
                }
            }

            for word_id in &update_stage.updated_expected_word_ids {
                let state0_count = state0_df_parent_scope.get(word_id).copied().unwrap_or(0);
                let token = update_stage
                    .token_by_word_id
                    .get(word_id)
                    .or_else(|| self.baseline_word_legend.get(word_id));
                let state1_count = token
                    .and_then(|t| mapped_scope.word_document_frequency_in_scope.get(t))
                    .copied()
                    .unwrap_or(0);
                total += state0_count + state1_count;
            }

            total.max(1)
        };

        let number_of_sources_of_prominent_word_in_hkt_state0 = {
            let mut prominent_node_id = None;
            for node_id in &scope_snapshot.node_ids_in_hkt_order {
                if let Some(word_ids) = scope_snapshot.node_word_ids_by_node_id.get(node_id) {
                    if word_ids.contains(&-1) {
                        continue;
                    }
                    prominent_node_id = Some(*node_id);
                    break;
                }
            }

            if let Some(node_id) = prominent_node_id {
                let word_source_ids = scope_snapshot
                    .node_word_source_ids_by_node_id
                    .get(&node_id)
                    .cloned()
                    .unwrap_or_default();
                let word_ids = scope_snapshot
                    .node_word_ids_by_node_id
                    .get(&node_id)
                    .cloned()
                    .unwrap_or_default();
                let prominent_word_id = word_ids.iter().copied().find(|id| *id != -1);
                prominent_word_id
                    .and_then(|word_id| word_source_ids.get(&word_id).map(|s| s.len()))
                    .unwrap_or(0)
            } else {
                0
            }
        };

        let number_of_sources_of_prominent_word_in_hkt_expected = {
            let mut max_count = 0usize;
            for (node_id, word_ids) in &scope_snapshot.node_word_ids_by_node_id {
                if word_ids.contains(&-1) {
                    continue;
                }
                let word_source_ids = scope_snapshot
                    .node_word_source_ids_by_node_id
                    .get(node_id)
                    .cloned()
                    .unwrap_or_default();
                let state1_sources = update_stage
                    .node_state1_sources_by_word_id
                    .get(node_id)
                    .cloned()
                    .unwrap_or_default();
                for word_id in word_ids.iter().copied().filter(|id| *id != -1) {
                    let state0_count = word_source_ids.get(&word_id).map(|s| s.len()).unwrap_or(0);
                    let state1_count = state1_sources.get(&word_id).map(|s| s.len()).unwrap_or(0);
                    max_count = max_count.max(state0_count + state1_count);
                }
            }

            for word_id in &update_stage.updated_expected_word_ids {
                let state0_count = state0_df_parent_scope.get(word_id).copied().unwrap_or(0);
                let token = update_stage
                    .token_by_word_id
                    .get(word_id)
                    .or_else(|| self.baseline_word_legend.get(word_id));
                let state1_count = token
                    .and_then(|t| mapped_scope.word_document_frequency_in_scope.get(t))
                    .copied()
                    .unwrap_or(0);
                max_count = max_count.max(state0_count + state1_count);
            }

            max_count.max(1)
        };

        let mut metrics_by_word_id: BTreeMap<i32, WordChangeMetrics> = BTreeMap::new();
        let mut words_already_mapped: BTreeSet<i32> = BTreeSet::new();

        for (node_id, word_ids) in &scope_snapshot.node_word_ids_by_node_id {
            if word_ids.contains(&-1) {
                continue;
            }

            let mut node_word_ids = word_ids.clone();
            if let Some(assigned) = update_stage.assigned_expected_words_by_node_id.get(node_id) {
                node_word_ids.extend(assigned.iter().copied());
            }

            let node_word_source_ids = scope_snapshot
                .node_word_source_ids_by_node_id
                .get(node_id)
                .cloned()
                .unwrap_or_default();
            let node_state1_sources = update_stage
                .node_state1_sources_by_word_id
                .get(node_id)
                .cloned()
                .unwrap_or_default();

            let sources_of_old_prominent_word_in_node = {
                let prominent_word_id = word_ids.iter().copied().find(|id| *id != -1);
                prominent_word_id
                    .and_then(|word_id| node_word_source_ids.get(&word_id).cloned())
                    .unwrap_or_default()
            };

            let current_sources_in_node = {
                let mut sources = scope_snapshot
                    .node_source_ids_by_node_id
                    .get(node_id)
                    .cloned()
                    .unwrap_or_default();
                if let Some(state1_sources) =
                    update_stage.node_state1_source_ids_by_node_id.get(node_id)
                {
                    sources.extend(state1_sources.iter().copied());
                }
                sources
            };

            for word_id in node_word_ids.iter().copied().filter(|id| *id != -1) {
                let word_source_ids = node_word_source_ids
                    .get(&word_id)
                    .cloned()
                    .unwrap_or_default();
                let mut all_sources_in_word = word_source_ids.clone();
                if let Some(state1_sources) = node_state1_sources.get(&word_id) {
                    all_sources_in_word.extend(state1_sources.iter().copied());
                }

                let number_of_sources_before_new_batch = word_source_ids.len() as f64;
                let number_of_sources_after_new_batch = all_sources_in_word.len() as f64;
                let number_of_sources_in_new_batch =
                    (all_sources_in_word.len() - word_source_ids.len()) as f64;

                let precentage_of_sources_in_hkt_in_old_batch = number_of_sources_before_new_batch
                    / (number_of_sources_for_all_words_in_hkt_state0 as f64);
                let precentage_of_sources_in_hkt_in_new_batch = number_of_sources_after_new_batch
                    / (number_of_sources_for_all_current_and_expected_words as f64);

                let number_of_sources_before_new_batch_over_number_of_sources_of_old_promin_word_in_hkt =
                    number_of_sources_before_new_batch
                        / (number_of_sources_of_prominent_word_in_hkt_state0.max(1) as f64);
                let number_of_intersected_sources_with_old_promin_word_in_node_over_num_sources_of_old_promin_word_in_node =
                    (sources_of_old_prominent_word_in_node
                        .intersection(&word_source_ids)
                        .count() as f64)
                        / (sources_of_old_prominent_word_in_node.len().max(1) as f64);

                let number_of_sources_after_new_batch_over_number_of_sources_of_new_promin_word_in_hkt =
                    number_of_sources_after_new_batch
                        / (number_of_sources_of_prominent_word_in_hkt_expected.max(1) as f64);
                let number_of_intersected_sources_with_new_promin_word_in_node_over_num_sources_of_new_promin_word_in_node =
                    (current_sources_in_node
                        .intersection(&all_sources_in_word)
                        .count() as f64)
                        / (current_sources_in_node.len().max(1) as f64);

                let old_deviation_fom_alpha_parameter = alpha_threshold
                    - number_of_sources_before_new_batch_over_number_of_sources_of_old_promin_word_in_hkt;
                let old_deviation_fom_beta_parameter = beta_threshold
                    - number_of_intersected_sources_with_old_promin_word_in_node_over_num_sources_of_old_promin_word_in_node;
                let new_deviation_fom_alpha_parameter = alpha_threshold
                    - number_of_sources_after_new_batch_over_number_of_sources_of_new_promin_word_in_hkt;
                let new_deviation_fom_beta_parameter = beta_threshold
                    - number_of_intersected_sources_with_new_promin_word_in_node_over_num_sources_of_new_promin_word_in_node;

                let erros_in_deviation_fom_alpha_parameter =
                    if new_deviation_fom_alpha_parameter > 0.0 {
                        new_deviation_fom_alpha_parameter.abs()
                    } else {
                        0.0
                    };
                let erros_in_deviation_fom_beta_parameter =
                    if new_deviation_fom_beta_parameter > 0.0 {
                        new_deviation_fom_beta_parameter.abs()
                    } else {
                        0.0
                    };

                metrics_by_word_id.insert(
                    word_id,
                    WordChangeMetrics {
                        word_id,
                        node_id: *node_id,
                        number_of_sources_before_new_batch,
                        number_of_sources_in_new_batch,
                        number_of_sources_after_new_batch,
                        number_of_sources_before_new_batch_over_number_of_sources_of_old_promin_word_in_hkt,
                        number_of_intersected_sources_with_old_promin_word_in_node_over_num_sources_of_old_promin_word_in_node,
                        number_of_sources_after_new_batch_over_number_of_sources_of_new_promin_word_in_hkt,
                        number_of_intersected_sources_with_new_promin_word_in_node_over_num_sources_of_new_promin_word_in_node,
                        old_deviation_fom_alpha_parameter,
                        old_deviation_fom_beta_parameter,
                        new_deviation_fom_alpha_parameter,
                        new_deviation_fom_beta_parameter,
                        erros_in_deviation_fom_alpha_parameter,
                        erros_in_deviation_fom_beta_parameter,
                        precentage_of_sources_in_hkt_in_old_batch,
                        precentage_of_sources_in_hkt_in_new_batch,
                    },
                );

                words_already_mapped.insert(word_id);
            }
        }

        for word_id in &update_stage.updated_expected_word_ids {
            if words_already_mapped.contains(word_id) {
                continue;
            }
            let state0_count = state0_df_parent_scope.get(word_id).copied().unwrap_or(0);
            let token = update_stage
                .token_by_word_id
                .get(word_id)
                .or_else(|| self.baseline_word_legend.get(word_id));
            let state1_count = token
                .and_then(|t| mapped_scope.word_document_frequency_in_scope.get(t))
                .copied()
                .unwrap_or(0);

            let number_of_sources_before_new_batch = 0.0;
            let number_of_sources_in_new_batch = state1_count as f64;
            let number_of_sources_after_new_batch = (state0_count + state1_count) as f64;

            let precentage_of_sources_in_hkt_in_old_batch = number_of_sources_before_new_batch
                / (number_of_sources_for_all_words_in_hkt_state0 as f64);
            let precentage_of_sources_in_hkt_in_new_batch = number_of_sources_after_new_batch
                / (number_of_sources_for_all_current_and_expected_words as f64);

            let number_of_sources_before_new_batch_over_number_of_sources_of_old_promin_word_in_hkt =
                number_of_sources_before_new_batch
                    / (number_of_sources_of_prominent_word_in_hkt_state0.max(1) as f64);
            let number_of_sources_after_new_batch_over_number_of_sources_of_new_promin_word_in_hkt =
                number_of_sources_after_new_batch
                    / (number_of_sources_of_prominent_word_in_hkt_expected.max(1) as f64);

            let old_deviation_fom_alpha_parameter = alpha_threshold
                - number_of_sources_before_new_batch_over_number_of_sources_of_old_promin_word_in_hkt;
            let new_deviation_fom_alpha_parameter = alpha_threshold
                - number_of_sources_after_new_batch_over_number_of_sources_of_new_promin_word_in_hkt;
            let erros_in_deviation_fom_alpha_parameter = if new_deviation_fom_alpha_parameter > 0.0
            {
                new_deviation_fom_alpha_parameter.abs()
            } else {
                0.0
            };

            metrics_by_word_id.insert(
                *word_id,
                WordChangeMetrics {
                    word_id: *word_id,
                    node_id: 0,
                    number_of_sources_before_new_batch,
                    number_of_sources_in_new_batch,
                    number_of_sources_after_new_batch,
                    number_of_sources_before_new_batch_over_number_of_sources_of_old_promin_word_in_hkt,
                    number_of_intersected_sources_with_old_promin_word_in_node_over_num_sources_of_old_promin_word_in_node: 0.0,
                    number_of_sources_after_new_batch_over_number_of_sources_of_new_promin_word_in_hkt,
                    number_of_intersected_sources_with_new_promin_word_in_node_over_num_sources_of_new_promin_word_in_node: 0.0,
                    old_deviation_fom_alpha_parameter,
                    old_deviation_fom_beta_parameter: 0.0,
                    new_deviation_fom_alpha_parameter,
                    new_deviation_fom_beta_parameter: 0.0,
                    erros_in_deviation_fom_alpha_parameter,
                    erros_in_deviation_fom_beta_parameter: 0.0,
                    precentage_of_sources_in_hkt_in_old_batch,
                    precentage_of_sources_in_hkt_in_new_batch,
                },
            );
        }

        Ok(metrics_by_word_id)
    }

    fn build_current_word_ids(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        _update_stage: &HktUpdateStage,
    ) -> BTreeSet<i32> {
        let mut current_word_ids: BTreeSet<i32> = BTreeSet::new();
        for (_node_id, word_ids) in &scope_snapshot.node_word_ids_by_node_id {
            if word_ids.contains(&-1) {
                continue;
            }
            current_word_ids.extend(word_ids.iter().copied().filter(|id| *id != -1));
        }
        current_word_ids
    }

    fn round4(value: f64) -> f64 {
        (value * 10000.0).round() / 10000.0
    }

    fn round3(value: f64) -> f64 {
        (value * 1000.0).round() / 1000.0
    }

    fn euclidean_distance(values_a: &[f64], values_b: &[f64]) -> f64 {
        let mut total = 0.0_f64;
        for (left, right) in values_a.iter().zip(values_b.iter()) {
            let diff = left - right;
            total += diff * diff;
        }
        total.sqrt()
    }

    fn compute_paper_selected_thresholds(
        &self,
        current_word_count: usize,
    ) -> SelectedPaperThresholds {
        let thresholds = &self.config.seca_thresholds;
        let alpha_option3_max = (current_word_count as f64).sqrt();
        let alpha_option3_threshold =
            Self::round3(thresholds.alpha_option3_threshold * alpha_option3_max);

        let alpha_error_threshold = match thresholds.selected_alpha_option {
            crate::config::AlphaErrorOption::Option1 => thresholds.alpha_option1_threshold,
            crate::config::AlphaErrorOption::Option2 => thresholds.alpha_option2_threshold,
            crate::config::AlphaErrorOption::Option3 => alpha_option3_threshold,
        };

        let beta_error_threshold = match thresholds.selected_beta_option {
            crate::config::BetaErrorOption::Option1 => thresholds.beta_option1_threshold,
            crate::config::BetaErrorOption::Option2 => thresholds.beta_option2_threshold,
            crate::config::BetaErrorOption::Option3 => thresholds.beta_option3_threshold,
        };

        let word_importance_error_threshold = match thresholds.selected_word_importance_option {
            crate::config::WordImportanceErrorOption::Option1 => {
                thresholds.word_importance_option1_threshold
            }
            crate::config::WordImportanceErrorOption::Option2 => {
                thresholds.word_importance_option2_threshold
            }
        };

        SelectedPaperThresholds {
            alpha_error_threshold,
            beta_error_threshold,
            word_importance_error_threshold,
        }
    }

    fn select_paper_errors_and_thresholds(
        &self,
        metrics: &PaperScopeMetrics,
        current_word_count: usize,
    ) -> SelectedPaperErrors {
        let thresholds = &self.config.seca_thresholds;
        let selected_thresholds = self.compute_paper_selected_thresholds(current_word_count);

        let alpha_error = match thresholds.selected_alpha_option {
            crate::config::AlphaErrorOption::Option1 => metrics.alpha_error_option1,
            crate::config::AlphaErrorOption::Option2 => metrics.alpha_error_option2,
            crate::config::AlphaErrorOption::Option3 => metrics.alpha_error_option3,
        };

        let beta_error = match thresholds.selected_beta_option {
            crate::config::BetaErrorOption::Option1 => metrics.beta_error_option1,
            crate::config::BetaErrorOption::Option2 => metrics.beta_error_option2,
            crate::config::BetaErrorOption::Option3 => metrics.beta_error_option3,
        };

        let word_importance_error = match thresholds.selected_word_importance_option {
            crate::config::WordImportanceErrorOption::Option1 => {
                metrics.word_importance_error_option1
            }
            crate::config::WordImportanceErrorOption::Option2 => {
                metrics.word_importance_error_option2
            }
        };

        SelectedPaperErrors {
            alpha_error,
            beta_error,
            word_importance_error,
            alpha_error_threshold: selected_thresholds.alpha_error_threshold,
            beta_error_threshold: selected_thresholds.beta_error_threshold,
            word_importance_error_threshold: selected_thresholds.word_importance_error_threshold,
        }
    }

    fn compute_paper_scope_diagnostic_inputs_from_change_metrics(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        mapped_scope: &MappedHktScopeState,
        update_stage: &HktUpdateStage,
        change_metrics: &BTreeMap<i32, WordChangeMetrics>,
    ) -> PaperScopeDiagnosticInputs {
        let current_word_ids = self.build_current_word_ids(scope_snapshot, update_stage);
        let mut state1_strengths = Vec::new();
        let mut state1_prominent_word_df = 0usize;

        for word_id in &current_word_ids {
            if let Some(metrics) = change_metrics.get(word_id) {
                state1_strengths.push(
                    metrics
                        .number_of_sources_after_new_batch_over_number_of_sources_of_new_promin_word_in_hkt,
                );
                state1_prominent_word_df = state1_prominent_word_df
                    .max(metrics.number_of_sources_after_new_batch.round() as usize);
            } else {
                state1_strengths.push(0.0);
            }
        }

        let alpha_threshold = self.config.seca_thresholds.alpha;
        let state1_alpha_violation_count = state1_strengths
            .iter()
            .filter(|strength1| **strength1 < alpha_threshold)
            .count();

        let state1_total_df: usize = mapped_scope
            .word_document_frequency_in_scope
            .values()
            .copied()
            .sum();

        let mut state1_mass_on_state0_vocab = 0.0_f64;
        if state1_total_df > 0 {
            for word_id in &current_word_ids {
                if let Some(word_token) = self.baseline_word_legend.get(word_id) {
                    let df = mapped_scope
                        .word_document_frequency_in_scope
                        .get(word_token)
                        .copied()
                        .unwrap_or(0);
                    state1_mass_on_state0_vocab += (df as f64) / (state1_total_df as f64);
                }
            }
        }

        let retained_expected_vocab_count = current_word_ids
            .intersection(&mapped_scope.known_word_ids_in_scope)
            .count();

        let retained_expected_vocab_ratio = if current_word_ids.is_empty() {
            0.0
        } else {
            (retained_expected_vocab_count as f64) / (current_word_ids.len() as f64)
        };

        let total_observed_scope_vocab =
            mapped_scope.known_word_ids_in_scope.len() + mapped_scope.new_tokens_in_scope.len();

        let known_vocab_ratio_in_state1 = if total_observed_scope_vocab == 0 {
            0.0
        } else {
            (mapped_scope.known_word_ids_in_scope.len() as f64)
                / (total_observed_scope_vocab as f64)
        };

        let new_vocab_ratio_in_state1 = if total_observed_scope_vocab == 0 {
            0.0
        } else {
            (mapped_scope.new_tokens_in_scope.len() as f64) / (total_observed_scope_vocab as f64)
        };

        PaperScopeDiagnosticInputs {
            state0_expected_vocab_count: current_word_ids.len(),
            state1_known_vocab_count: mapped_scope.known_word_ids_in_scope.len(),
            state1_new_vocab_count: mapped_scope.new_tokens_in_scope.len(),
            retained_expected_vocab_count,
            state1_total_df,
            state1_mass_on_state0_vocab: state1_mass_on_state0_vocab.clamp(0.0, 1.0),
            retained_expected_vocab_ratio: retained_expected_vocab_ratio.clamp(0.0, 1.0),
            known_vocab_ratio_in_state1: known_vocab_ratio_in_state1.clamp(0.0, 1.0),
            new_vocab_ratio_in_state1: new_vocab_ratio_in_state1.clamp(0.0, 1.0),
            state1_prominent_word_df,
            state1_strengths_for_state0_expected_words: state1_strengths,
            state1_alpha_violation_count,
        }
    }

    pub(super) fn compute_paper_scope_metrics_from_change_metrics(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        _mapped_scope: &MappedHktScopeState,
        update_stage: &HktUpdateStage,
        change_metrics: &BTreeMap<i32, WordChangeMetrics>,
    ) -> Result<PaperScopeMetrics, SecaError> {
        let thresholds = &self.config.seca_thresholds;
        let current_word_ids = self.build_current_word_ids(scope_snapshot, update_stage);

        let mut _strength0 = Vec::new();
        let mut strength1 = Vec::new();
        let mut _eligibility0 = Vec::new();
        let mut eligibility1 = Vec::new();
        let mut _importance0 = Vec::new();
        let mut importance1 = Vec::new();

        for word_id in &current_word_ids {
            if let Some(metrics) = change_metrics.get(word_id) {
                _strength0.push(
                    metrics
                        .number_of_sources_before_new_batch_over_number_of_sources_of_old_promin_word_in_hkt,
                );
                strength1.push(
                    metrics
                        .number_of_sources_after_new_batch_over_number_of_sources_of_new_promin_word_in_hkt,
                );
                _eligibility0.push(
                    metrics
                        .number_of_intersected_sources_with_old_promin_word_in_node_over_num_sources_of_old_promin_word_in_node,
                );
                eligibility1.push(
                    metrics
                        .number_of_intersected_sources_with_new_promin_word_in_node_over_num_sources_of_new_promin_word_in_node,
                );
                _importance0.push(metrics.precentage_of_sources_in_hkt_in_old_batch);
                importance1.push(metrics.precentage_of_sources_in_hkt_in_new_batch);
            } else {
                _strength0.push(0.0);
                strength1.push(0.0);
                _eligibility0.push(0.0);
                eligibility1.push(0.0);
                _importance0.push(0.0);
                importance1.push(0.0);
            }
        }

        let n = strength1.len() as f64;

        let (
            alpha_error_option1,
            alpha_error_option2,
            alpha_error_option3,
            beta_error_option1,
            beta_error_option2,
            beta_error_option3,
        ) = if n == 0.0 {
            (
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
            )
        } else {
            let mut total_strength_deviation = 0.0_f64;
            for value in &strength1 {
                if *value < thresholds.alpha {
                    total_strength_deviation += thresholds.alpha - *value;
                }
            }
            let alpha_error_option1 = Some(Self::round4(total_strength_deviation / n));

            let mut total_strength_difference = 0.0_f64;
            for (strength0, strength1) in _strength0.iter().zip(strength1.iter()) {
                if *strength1 < thresholds.alpha {
                    total_strength_difference += (*strength1 - *strength0).abs();
                }
            }
            let alpha_error_option2 = Some(Self::round4(total_strength_difference / n));
            let alpha_error_option3 = Some(Self::round4(Self::euclidean_distance(
                &_strength0,
                &strength1,
            )));

            let mut total_eligibility_deviation = 0.0_f64;
            for value in &eligibility1 {
                if *value < thresholds.beta {
                    total_eligibility_deviation += thresholds.beta - *value;
                }
            }
            let beta_error_option1 = Some(Self::round4(total_eligibility_deviation / n));

            let mut total_eligibility_difference = 0.0_f64;
            for (eligibility0, eligibility1) in _eligibility0.iter().zip(eligibility1.iter()) {
                if *eligibility1 < thresholds.beta {
                    total_eligibility_difference += (*eligibility1 - *eligibility0).abs();
                }
            }
            let beta_error_option2 = Some(Self::round4(total_eligibility_difference / n));
            let beta_error_option3 = Some(Self::round4(Self::euclidean_distance(
                &_eligibility0,
                &eligibility1,
            )));

            (
                alpha_error_option1,
                alpha_error_option2,
                alpha_error_option3,
                beta_error_option1,
                beta_error_option2,
                beta_error_option3,
            )
        };

        let total_importance: f64 = importance1.iter().sum();
        let word_importance_error_option1 = Some(Self::round4(1.0 - total_importance));
        let word_importance_error_option2 = Some(Self::round4(Self::euclidean_distance(
            &_importance0,
            &importance1,
        )));

        let alpha_value = if strength1.is_empty() {
            Some(0.0)
        } else {
            Some(strength1.iter().sum::<f64>() / (strength1.len() as f64))
        };
        let beta_value = if eligibility1.is_empty() {
            Some(0.0)
        } else {
            Some(eligibility1.iter().sum::<f64>() / (eligibility1.len() as f64))
        };
        let wi_retained_mass_value = Some(total_importance);
        let wi_error_value = word_importance_error_option1;

        Ok(PaperScopeMetrics {
            alpha_value,
            beta_value,
            wi_retained_mass_value,
            wi_error_value,
            alpha_error_option1,
            alpha_error_option2,
            alpha_error_option3,
            beta_error_option1,
            beta_error_option2,
            beta_error_option3,
            word_importance_error_option1,
            word_importance_error_option2,
        })
    }

    fn compute_paper_scope_diagnostic_inputs(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        mapped_scope: &MappedHktScopeState,
    ) -> PaperScopeDiagnosticInputs {
        let state1_total_df: usize = mapped_scope
            .word_document_frequency_in_scope
            .values()
            .copied()
            .sum();

        let mut state1_mass_on_state0_vocab = 0.0_f64;

        if state1_total_df > 0 {
            for word_id in &scope_snapshot.expected_word_ids {
                if let Some(word_token) = self.baseline_word_legend.get(word_id) {
                    let df = mapped_scope
                        .word_document_frequency_in_scope
                        .get(word_token)
                        .copied()
                        .unwrap_or(0);
                    state1_mass_on_state0_vocab += (df as f64) / (state1_total_df as f64);
                }
            }
        }

        let retained_expected_vocab_count = scope_snapshot
            .expected_word_ids
            .intersection(&mapped_scope.known_word_ids_in_scope)
            .count();

        let retained_expected_vocab_ratio = if scope_snapshot.expected_word_ids.is_empty() {
            0.0
        } else {
            (retained_expected_vocab_count as f64) / (scope_snapshot.expected_word_ids.len() as f64)
        };

        let total_observed_scope_vocab =
            mapped_scope.known_word_ids_in_scope.len() + mapped_scope.new_tokens_in_scope.len();

        let known_vocab_ratio_in_state1 = if total_observed_scope_vocab == 0 {
            0.0
        } else {
            (mapped_scope.known_word_ids_in_scope.len() as f64)
                / (total_observed_scope_vocab as f64)
        };

        let new_vocab_ratio_in_state1 = if total_observed_scope_vocab == 0 {
            0.0
        } else {
            (mapped_scope.new_tokens_in_scope.len() as f64) / (total_observed_scope_vocab as f64)
        };

        // Eq.(6)-oriented alpha diagnostic inputs scaffold
        // Strength1(w) = CountSources(Sources(w)) / CountSources(Sources(p_state1))
        let state1_prominent_word_df =
            self.compute_state1_prominent_word_df_in_scope(scope_snapshot, mapped_scope);

        let mut state1_strengths_for_state0_expected_words = Vec::new();

        if state1_prominent_word_df > 0 {
            for word_id in &scope_snapshot.expected_word_ids {
                if let Some(word_token) = self.baseline_word_legend.get(word_id) {
                    let df = mapped_scope
                        .word_document_frequency_in_scope
                        .get(word_token)
                        .copied()
                        .unwrap_or(0);

                    let strength1 = (df as f64) / (state1_prominent_word_df as f64);
                    state1_strengths_for_state0_expected_words.push(strength1.clamp(0.0, 1.0));
                } else {
                    state1_strengths_for_state0_expected_words.push(0.0);
                }
            }
        } else {
            state1_strengths_for_state0_expected_words
                .resize(scope_snapshot.expected_word_ids.len(), 0.0);
        }

        let alpha_threshold = self.config.seca_thresholds.alpha;
        let state1_alpha_violation_count = state1_strengths_for_state0_expected_words
            .iter()
            .filter(|strength1| **strength1 < alpha_threshold)
            .count();

        PaperScopeDiagnosticInputs {
            state0_expected_vocab_count: scope_snapshot.expected_word_ids.len(),
            state1_known_vocab_count: mapped_scope.known_word_ids_in_scope.len(),
            state1_new_vocab_count: mapped_scope.new_tokens_in_scope.len(),
            retained_expected_vocab_count,
            state1_total_df,
            state1_mass_on_state0_vocab: state1_mass_on_state0_vocab.clamp(0.0, 1.0),

            retained_expected_vocab_ratio: retained_expected_vocab_ratio.clamp(0.0, 1.0),
            known_vocab_ratio_in_state1: known_vocab_ratio_in_state1.clamp(0.0, 1.0),
            new_vocab_ratio_in_state1: new_vocab_ratio_in_state1.clamp(0.0, 1.0),

            state1_prominent_word_df,
            state1_strengths_for_state0_expected_words,
            state1_alpha_violation_count,
        }
    }

    fn format_paper_scope_diagnostic_inputs_note(
        &self,
        hkt_id: i32,
        inputs: &PaperScopeDiagnosticInputs,
    ) -> String {
        format!(
            "HKT {} paper inputs (diagnostic): expected_vocab={}, retained_expected_vocab={}, state1_known_vocab={}, state1_new_vocab={}, state1_total_df={}, state1_mass_on_state0_vocab={:.4}, state1_prominent_word_df={}, alpha_violation_count={}",
            hkt_id,
            inputs.state0_expected_vocab_count,
            inputs.retained_expected_vocab_count,
            inputs.state1_known_vocab_count,
            inputs.state1_new_vocab_count,
            inputs.state1_total_df,
            inputs.state1_mass_on_state0_vocab,
            inputs.state1_prominent_word_df,
            inputs.state1_alpha_violation_count,
        )
    }

    fn evaluate_scope_trigger_decision(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        mapped_scope: &MappedHktScopeState,
        update_stage: &HktUpdateStage,
        change_metrics: Option<&BTreeMap<i32, WordChangeMetrics>>,
    ) -> Result<HktTriggerDecisionInternal, SecaError> {
        match self.config.trigger_policy_mode {
            TriggerPolicyMode::Placeholder => {
                self.evaluate_scope_trigger_decision_placeholder(scope_snapshot, mapped_scope)
            }
            TriggerPolicyMode::PaperDiagnosticScaffold => {
                let metrics = change_metrics.ok_or_else(|| SecaError::StateError {
                    message: "missing word change metrics for paper trigger decision".to_string(),
                })?;
                self.evaluate_scope_trigger_decision_paper_scaffold(
                    scope_snapshot,
                    mapped_scope,
                    update_stage,
                    metrics,
                )
            }
        }
    }

    pub(super) fn evaluate_scope_trigger_decision_paper_scaffold(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        mapped_scope: &MappedHktScopeState,
        update_stage: &HktUpdateStage,
        change_metrics: &BTreeMap<i32, WordChangeMetrics>,
    ) -> Result<HktTriggerDecisionInternal, SecaError> {
        let paper_bundle = self.compute_paper_diagnostic_bundle_from_change_metrics(
            scope_snapshot,
            mapped_scope,
            update_stage,
            change_metrics,
        )?;

        let should_reconstruct = !paper_bundle.paper_shadow_trigger_reasons.is_empty();

        // Keep placeholder metrics available for continuity/debugging while paper formulas evolve.
        let placeholder_metrics = self.compute_scope_placeholder_metrics(mapped_scope);

        Ok(HktTriggerDecisionInternal {
            hkt_id: scope_snapshot.hkt_id,
            should_reconstruct,
            trigger_reasons: paper_bundle.paper_shadow_trigger_reasons.clone(),
            alpha_error: Some(placeholder_metrics.alpha_error),
            beta_error: Some(placeholder_metrics.beta_error),
            word_importance_error: Some(placeholder_metrics.word_importance_error),
            paper_word_importance_error: paper_bundle.paper_word_importance_error,
            paper_alpha_error: paper_bundle.paper_alpha_error,
            paper_beta_error: paper_bundle.paper_beta_error,
            active_trigger_policy_label: Some("paper_diagnostic_scaffold".to_string()),
        })
    }

    fn compute_paper_diagnostic_bundle_from_change_metrics(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        mapped_scope: &MappedHktScopeState,
        update_stage: &HktUpdateStage,
        change_metrics: &BTreeMap<i32, WordChangeMetrics>,
    ) -> Result<PaperDiagnosticBundle, SecaError> {
        let scoped_source_count = mapped_scope.scoped_batch_source_indexes.len();
        let inputs = self.compute_paper_scope_diagnostic_inputs_from_change_metrics(
            scope_snapshot,
            mapped_scope,
            update_stage,
            change_metrics,
        );

        let paper_metrics = self.compute_paper_scope_metrics_from_change_metrics(
            scope_snapshot,
            mapped_scope,
            update_stage,
            change_metrics,
        )?;

        let current_word_count = self
            .build_current_word_ids(scope_snapshot, update_stage)
            .len();
        let selected = self.select_paper_errors_and_thresholds(&paper_metrics, current_word_count);

        let paper_shadow_trigger_reasons = self.compute_paper_shadow_trigger_reasons(
            selected.word_importance_error,
            selected.alpha_error,
            selected.beta_error,
            scoped_source_count,
            selected.word_importance_error_threshold,
            selected.alpha_error_threshold,
            selected.beta_error_threshold,
        );

        Ok(PaperDiagnosticBundle {
            inputs,
            paper_word_importance_error: selected.word_importance_error,
            paper_alpha_error: selected.alpha_error,
            paper_beta_error: selected.beta_error,
            paper_shadow_trigger_reasons,
            scoped_source_count,
        })
    }

    fn compute_paper_scope_metrics(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        mapped_scope: &MappedHktScopeState,
    ) -> PaperScopeMetrics {
        let thresholds = &self.config.seca_thresholds;
        let inputs = self.compute_paper_scope_diagnostic_inputs(scope_snapshot, mapped_scope);

        // WI diagnostic (eq.12-inspired scaffold)
        let wi_retained_mass_value = if inputs.state1_total_df == 0 {
            Some(0.0)
        } else {
            Some(inputs.state1_mass_on_state0_vocab.clamp(0.0, 1.0))
        };

        let wi_error_value =
            wi_retained_mass_value.map(|retained_mass| (1.0 - retained_mass).clamp(0.0, 1.0));

        // Alpha value (diagnostic summary)
        let alpha_value = Some(inputs.retained_expected_vocab_ratio.clamp(0.0, 1.0));

        let alpha_error_option1 = {
            let n = inputs.state1_strengths_for_state0_expected_words.len();
            if n == 0 {
                Some(0.0)
            } else {
                let mut total_violation_deviation = 0.0_f64;
                for strength1 in &inputs.state1_strengths_for_state0_expected_words {
                    if *strength1 < thresholds.alpha {
                        total_violation_deviation += (thresholds.alpha - *strength1).abs();
                    }
                }
                Some((total_violation_deviation / (n as f64)).clamp(0.0, 1.0))
            }
        };

        // Paper-facing beta error from Eq.(9)-oriented word-level violation metric
        let beta_error_option1 =
            self.compute_paper_beta_error_diagnostic(scope_snapshot, mapped_scope);

        // Human-readable beta summary only (non-trigger)
        let beta_node_diagnostics =
            self.compute_paper_beta_node_diagnostics(scope_snapshot, mapped_scope);

        let active_beta_nodes: Vec<&PaperNodeBetaDiagnostic> = beta_node_diagnostics
            .iter()
            .filter(|node| {
                (node.state1_known_vocab_count_in_node + node.state1_new_vocab_count_in_node) > 0
            })
            .collect();

        let beta_value = if active_beta_nodes.is_empty() {
            Some(0.0)
        } else {
            let mean_beta = active_beta_nodes
                .iter()
                .map(|node| node.node_beta_value)
                .sum::<f64>()
                / (active_beta_nodes.len() as f64);
            Some(mean_beta.clamp(0.0, 1.0))
        };

        let word_importance_error_option1 = wi_error_value;

        PaperScopeMetrics {
            alpha_value,
            beta_value,
            wi_retained_mass_value,
            wi_error_value,
            alpha_error_option1,
            alpha_error_option2: None,
            alpha_error_option3: None,
            beta_error_option1,
            beta_error_option2: None,
            beta_error_option3: None,
            word_importance_error_option1,
            word_importance_error_option2: None,
        }
    }

    fn resolve_placeholder_scope_vocab_counts(
        &self,
        mapped_scope: &MappedHktScopeState,
    ) -> (usize, usize) {
        let aggregate_known_count = mapped_scope.known_word_ids_in_scope.len();
        let aggregate_new_count = mapped_scope.new_tokens_in_scope.len();

        if aggregate_known_count > 0 || aggregate_new_count > 0 {
            return (aggregate_known_count, aggregate_new_count);
        }

        let mut derived_known_word_ids = BTreeSet::new();
        for word_ids in mapped_scope.known_word_ids_by_node_id.values() {
            derived_known_word_ids.extend(word_ids.iter().copied());
        }

        let mut derived_new_tokens = BTreeSet::new();
        for tokens in mapped_scope.new_tokens_by_node_id.values() {
            derived_new_tokens.extend(tokens.iter().cloned());
        }

        (derived_known_word_ids.len(), derived_new_tokens.len())
    }

    fn compute_paper_beta_node_diagnostics(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        mapped_scope: &MappedHktScopeState,
    ) -> Vec<PaperNodeBetaDiagnostic> {
        let mut diagnostics = Vec::new();

        let mut node_ids: Vec<i32> = scope_snapshot.non_refuge_node_ids.clone();
        if let Some(refuge_node_id) = scope_snapshot.refuge_node_id {
            node_ids.push(refuge_node_id);
        }
        node_ids.sort_unstable();

        for node_id in node_ids {
            let state0_node_vocab_count = scope_snapshot
                .node_word_ids_by_node_id
                .get(&node_id)
                .map(|set| set.len())
                .unwrap_or(0);

            let state1_known_vocab_count_in_node = mapped_scope
                .known_word_ids_by_node_id
                .get(&node_id)
                .map(|set| set.len())
                .unwrap_or(0);

            let state1_new_vocab_count_in_node = mapped_scope
                .new_tokens_by_node_id
                .get(&node_id)
                .map(|set| set.len())
                .unwrap_or(0);

            let state1_total_observed_node_vocab =
                state1_known_vocab_count_in_node + state1_new_vocab_count_in_node;

            let node_beta_value = if state1_total_observed_node_vocab == 0 {
                0.0
            } else {
                (state1_new_vocab_count_in_node as f64) / (state1_total_observed_node_vocab as f64)
            };

            diagnostics.push(PaperNodeBetaDiagnostic {
                node_id,
                state0_node_vocab_count,
                state1_known_vocab_count_in_node,
                state1_new_vocab_count_in_node,
                node_beta_value: node_beta_value.clamp(0.0, 1.0),
            });
        }

        diagnostics
    }

    fn format_paper_beta_node_diagnostics_note(
        &self,
        hkt_id: i32,
        node_diagnostics: &[PaperNodeBetaDiagnostic],
    ) -> String {
        if node_diagnostics.is_empty() {
            return format!("HKT {} paper beta node diagnostics: no nodes", hkt_id);
        }

        let summary = node_diagnostics
            .iter()
            .map(|d| {
                format!(
                    "node {}: beta={:.4} (state0_vocab={}, known={}, new={})",
                    d.node_id,
                    d.node_beta_value,
                    d.state0_node_vocab_count,
                    d.state1_known_vocab_count_in_node,
                    d.state1_new_vocab_count_in_node
                )
            })
            .collect::<Vec<_>>()
            .join("; ");

        format!(
            "HKT {} paper beta node diagnostics (node-summary, non-trigger): {}",
            hkt_id, summary
        )
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct RecursiveTriggerPlan {
    pub(crate) batch_index: u32,
    pub(crate) any_reconstruction_triggered: bool,
    pub(crate) reconstruct_hkt_ids: Vec<i32>,
    pub(crate) reconstruct_scopes_by_hkt_id: BTreeMap<i32, BTreeSet<usize>>,
    pub(crate) notes: Vec<String>,
    pub(crate) ancestor_words_by_hkt_id: BTreeMap<i32, BTreeMap<i32, (usize, usize)>>,
    pub(crate) ancestor_accepted_words_by_hkt_id: BTreeMap<i32, BTreeMap<i32, (usize, usize)>>,
    pub(crate) ancestor_rejected_words_by_hkt_id: BTreeMap<i32, BTreeMap<i32, (usize, usize)>>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct HktTriggerDecisionInternal {
    pub(crate) hkt_id: i32,
    pub(crate) should_reconstruct: bool,
    pub(crate) trigger_reasons: Vec<String>,
    pub(crate) alpha_error: Option<f64>,
    pub(crate) beta_error: Option<f64>,
    pub(crate) word_importance_error: Option<f64>, // active placeholder WI gate
    pub(crate) paper_word_importance_error: Option<f64>, // diagnostic only
    pub(crate) paper_alpha_error: Option<f64>,     // diagnostic only (eq.6 scaffold)
    pub(crate) paper_beta_error: Option<f64>,      // diagnostic only (eq.9 scaffold)
    pub(crate) active_trigger_policy_label: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct ScopePlaceholderMetrics {
    alpha_error: f64,
    beta_error: f64,
    word_importance_error: f64,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct PaperScopeDiagnosticInputs {
    state0_expected_vocab_count: usize,
    state1_known_vocab_count: usize,
    state1_new_vocab_count: usize,
    retained_expected_vocab_count: usize,
    state1_total_df: usize,
    state1_mass_on_state0_vocab: f64,

    retained_expected_vocab_ratio: f64,
    known_vocab_ratio_in_state1: f64,
    new_vocab_ratio_in_state1: f64,

    state1_prominent_word_df: usize,
    state1_strengths_for_state0_expected_words: Vec<f64>,
    state1_alpha_violation_count: usize,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct PaperDiagnosticBundle {
    inputs: PaperScopeDiagnosticInputs,
    paper_word_importance_error: Option<f64>,
    paper_alpha_error: Option<f64>,
    paper_beta_error: Option<f64>,
    paper_shadow_trigger_reasons: Vec<String>,
    scoped_source_count: usize,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub(super) struct PaperScopeMetrics {
    alpha_value: Option<f64>,
    beta_value: Option<f64>,
    wi_retained_mass_value: Option<f64>,
    wi_error_value: Option<f64>,

    alpha_error_option1: Option<f64>,
    alpha_error_option2: Option<f64>,
    alpha_error_option3: Option<f64>,
    beta_error_option1: Option<f64>,
    beta_error_option2: Option<f64>,
    beta_error_option3: Option<f64>,
    word_importance_error_option1: Option<f64>,
    word_importance_error_option2: Option<f64>,
}

#[allow(dead_code)]
impl PaperScopeMetrics {
    pub(super) fn alpha_error_option1(&self) -> Option<f64> {
        self.alpha_error_option1
    }

    pub(super) fn beta_error_option1(&self) -> Option<f64> {
        self.beta_error_option1
    }

    pub(super) fn word_importance_error_option1(&self) -> Option<f64> {
        self.word_importance_error_option1
    }
}

#[derive(Debug, Clone, Default)]
struct SelectedPaperThresholds {
    alpha_error_threshold: f64,
    beta_error_threshold: f64,
    word_importance_error_threshold: f64,
}

#[derive(Debug, Clone, Default)]
struct SelectedPaperErrors {
    alpha_error: Option<f64>,
    beta_error: Option<f64>,
    word_importance_error: Option<f64>,
    alpha_error_threshold: f64,
    beta_error_threshold: f64,
    word_importance_error_threshold: f64,
}

#[derive(Debug, Clone, Default)]
struct PaperNodeBetaDiagnostic {
    node_id: i32,
    state0_node_vocab_count: usize,
    state1_known_vocab_count_in_node: usize,
    state1_new_vocab_count_in_node: usize,
    node_beta_value: f64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SelectedHktRebuildExecutionReport {
    pub(crate) batch_index: u32,
    pub(crate) entries: Vec<SelectedHktRebuildExecutionEntry>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SelectedHktRebuildExecutionEntry {
    pub(crate) hkt_id: i32,
    pub(crate) scoped_source_count: usize,
    pub(crate) unique_token_count: usize,
    pub(crate) known_token_count: usize,
    pub(crate) new_token_count: usize,
}

impl SecaEngine {
    pub(crate) fn build_selected_hkt_execution_report(
        &self,
        batch: &SourceBatch,
        plan: &SelectedHktRebuildPlan,
    ) -> Result<SelectedHktRebuildExecutionReport, SecaError> {
        let mut baseline_word_id_by_token = std::collections::BTreeMap::new();
        for (word_id, token) in &self.baseline_word_legend {
            baseline_word_id_by_token.insert(token.as_str(), *word_id);
        }

        let mut entries = Vec::with_capacity(plan.requests.len());

        for request in &plan.requests {
            let scoped_batch = self
                .extract_scoped_batch_from_indexes(batch, &request.scoped_batch_source_indexes)?;

            let mut unique_tokens = std::collections::BTreeSet::new();
            for source in &scoped_batch.sources {
                for token in &source.tokens {
                    let normalized = token.trim();
                    if normalized.is_empty() {
                        continue;
                    }
                    unique_tokens.insert(normalized.to_string());
                }
            }

            let mut known = 0usize;
            let mut new = 0usize;
            for token in &unique_tokens {
                if baseline_word_id_by_token.contains_key(token.as_str()) {
                    known += 1;
                } else {
                    new += 1;
                }
            }

            entries.push(SelectedHktRebuildExecutionEntry {
                hkt_id: request.target_hkt_id,
                scoped_source_count: scoped_batch.sources.len(),
                unique_token_count: unique_tokens.len(),
                known_token_count: known,
                new_token_count: new,
            });
        }

        entries.sort_by_key(|e| e.hkt_id);

        Ok(SelectedHktRebuildExecutionReport {
            batch_index: plan.batch_index,
            entries,
        })
    }

    pub(crate) fn format_selected_hkt_execution_report_note(
        &self,
        report: &SelectedHktRebuildExecutionReport,
    ) -> String {
        if report.entries.is_empty() {
            return format!(
                "Selected-HKT rebuild execution report (batch {}): no entries",
                report.batch_index
            );
        }

        let summary = report
            .entries
            .iter()
            .map(|e| {
                format!(
                    "HKT {}: scoped_sources={}, unique_tokens={}, known={}, new={}",
                    e.hkt_id,
                    e.scoped_source_count,
                    e.unique_token_count,
                    e.known_token_count,
                    e.new_token_count
                )
            })
            .collect::<Vec<_>>()
            .join("; ");

        format!(
            "Selected-HKT rebuild execution report (batch {}): {}",
            report.batch_index, summary
        )
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SelectedHktSubtreeDryRunReport {
    pub(crate) batch_index: u32,
    pub(crate) entries: Vec<SelectedHktSubtreeDryRunEntry>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SelectedHktSubtreeDryRunEntry {
    pub(crate) target_hkt_id: i32,
    pub(crate) scoped_source_count: usize,
    pub(crate) subtree_hkt_count: usize,
    pub(crate) subtree_node_count: usize,
    pub(crate) new_token_count_in_scope: usize,
}

impl SecaEngine {
    pub(crate) fn build_selected_hkt_subtree_dry_run_report(
        &self,
        plan: &SelectedHktRebuildPlan,
        trigger_plan: &crate::engine::trigger::RecursiveTriggerPlan,
    ) -> Result<SelectedHktSubtreeDryRunReport, SecaError> {
        // Use stored batches as the reconstruction memory source (consistent with current rebuild strategy).
        // This uses the latest batch index as report batch_index.
        let batch_index = plan.batch_index;
        let current_batch = self
            .processed_batches
            .last()
            .ok_or_else(|| SecaError::StateError {
                message: "cannot build subtree dry-run report: no stored batches".to_string(),
            })?;

        let mut entries = Vec::with_capacity(plan.requests.len());

        for request in &plan.requests {
            let scoped_records = crate::engine::rebuild::build_scoped_source_word_records_for_hkt(
                self,
                current_batch,
                trigger_plan,
                request.target_hkt_id,
            )?;

            if scoped_records.is_empty() {
                entries.push(SelectedHktSubtreeDryRunEntry {
                    target_hkt_id: request.target_hkt_id,
                    scoped_source_count: 0,
                    subtree_hkt_count: 0,
                    subtree_node_count: 0,
                    new_token_count_in_scope: 0,
                });
                continue;
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

            let subtree = hkt_builder.build_full_tree(scoped_records.clone(), true)?;
            let subtree_hkt_count = subtree.hkts_by_id.len();
            let subtree_node_count = subtree.nodes_by_id.len();

            // Use existing scope stats to estimate new tokens in this scope (relative to current baseline vocab).
            let mut unique_tokens = std::collections::BTreeSet::new();
            for record in &scoped_records {
                if let Some(word) = &record.word {
                    unique_tokens.insert(word.clone());
                }
            }

            let mut new_tokens = 0usize;
            for token in &unique_tokens {
                if !self
                    .baseline_word_legend
                    .values()
                    .any(|t| t.as_str() == token.as_str())
                {
                    new_tokens += 1;
                }
            }

            entries.push(SelectedHktSubtreeDryRunEntry {
                target_hkt_id: request.target_hkt_id,
                scoped_source_count: scoped_records
                    .iter()
                    .map(|r| r.source_id)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                subtree_hkt_count,
                subtree_node_count,
                new_token_count_in_scope: new_tokens,
            });
        }

        entries.sort_by_key(|e| e.target_hkt_id);

        Ok(SelectedHktSubtreeDryRunReport {
            batch_index,
            entries,
        })
    }

    pub(crate) fn format_selected_hkt_subtree_dry_run_report_note(
        &self,
        report: &SelectedHktSubtreeDryRunReport,
    ) -> String {
        if report.entries.is_empty() {
            return format!(
                "Selected-HKT subtree dry-run report (batch {}): no entries",
                report.batch_index
            );
        }

        let summary = report
            .entries
            .iter()
            .map(|e| {
                format!(
                    "HKT {}: scoped_sources={}, subtree_hkts={}, subtree_nodes={}, new_tokens_in_scope={}",
                    e.target_hkt_id,
                    e.scoped_source_count,
                    e.subtree_hkt_count,
                    e.subtree_node_count,
                    e.new_token_count_in_scope
                )
            })
            .collect::<Vec<_>>()
            .join("; ");

        format!(
            "Selected-HKT subtree dry-run report (batch {}): {}",
            report.batch_index, summary
        )
    }
    fn compute_state1_prominent_word_df_in_scope(
        &self,
        scope_snapshot: &HktScopeSnapshot,
        mapped_scope: &MappedHktScopeState,
    ) -> usize {
        // C# considers current node words + expected dynamic words.
        // In this Rust stage, "expected_word_ids" is the HKT's expected words set;
        // plus node_word_ids_by_node_id contains current node words.
        // We take max DF among all those word IDs present in baseline legend.

        let mut candidate_word_ids: BTreeSet<i32> = BTreeSet::new();

        candidate_word_ids.extend(scope_snapshot.expected_word_ids.iter().copied());
        for word_ids in scope_snapshot.node_word_ids_by_node_id.values() {
            for word_id in word_ids {
                if *word_id != -1 {
                    candidate_word_ids.insert(*word_id);
                }
            }
        }

        let mut max_df = 0usize;

        for word_id in candidate_word_ids {
            let Some(token) = self.baseline_word_legend.get(&word_id) else {
                continue;
            };

            let df = mapped_scope
                .word_document_frequency_in_scope
                .get(token)
                .copied()
                .unwrap_or(0);

            if df > max_df {
                max_df = df;
            }
        }

        max_df
    }

    fn build_baseline_source_word_sets(&self) -> Result<BTreeMap<i64, BTreeSet<i32>>, SecaError> {
        let baseline = self
            .processed_batches
            .first()
            .ok_or_else(|| SecaError::StateError {
                message: "baseline batch missing for state0 df".to_string(),
            })?;

        let mut baseline_word_id_by_token: BTreeMap<&str, i32> = BTreeMap::new();
        for (word_id, token) in &self.baseline_word_legend {
            baseline_word_id_by_token.insert(token.as_str(), *word_id);
        }

        let mut by_source: BTreeMap<i64, BTreeSet<i32>> = BTreeMap::new();

        for source in &baseline.sources {
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

    fn build_state0_df_in_parent_scope(
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

    fn build_node_state0_sources_by_word_id(
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

    fn build_child_ancestor_context_from_parent_node(
        &self,
        batch: &SourceBatch,
        parent_scope_snapshot: &HktScopeSnapshot,
        parent_mapped_scope: &MappedHktScopeState,
        _parent_hkt_id: i32,
        parent_node_id: i32,
        incoming_ancestor: &AncestorContext,
    ) -> AncestorContext {
        let mut next = incoming_ancestor.clone();

        // Refuge nodes do not contribute ancestor words in the C# logic
        let Some(node_word_ids) = parent_scope_snapshot
            .node_word_ids_by_node_id
            .get(&parent_node_id)
        else {
            return next;
        };

        if node_word_ids.contains(&-1) {
            return next;
        }

        let baseline_source_word_sets = match self.build_baseline_source_word_sets() {
            Ok(sets) => sets,
            Err(_) => return next,
        };

        let node_state0_sources_by_word_id = self.build_node_state0_sources_by_word_id(
            parent_scope_snapshot,
            &baseline_source_word_sets,
        );
        let state0_df_parent_scope = match self
            .build_state0_df_in_parent_scope(parent_scope_snapshot, &baseline_source_word_sets)
        {
            Ok(df) => df,
            Err(_) => return next,
        };

        let mut token_to_word_id: BTreeMap<String, i32> = BTreeMap::new();
        let mut next_synthetic_word_id: i32 = -2;
        for (word_id, token) in &self.baseline_word_legend {
            token_to_word_id.insert(token.clone(), *word_id);
        }

        let mut normalized_tokens_by_source_index: BTreeMap<usize, BTreeSet<String>> =
            BTreeMap::new();
        for source_index in &parent_mapped_scope.scoped_batch_source_indexes {
            let source = match batch.sources.get(*source_index) {
                Some(s) => s,
                None => continue,
            };
            let mut tokens = BTreeSet::new();
            for token in &source.tokens {
                let normalized = token.trim();
                if !normalized.is_empty() {
                    tokens.insert(normalized.to_string());
                }
            }
            normalized_tokens_by_source_index.insert(*source_index, tokens);
        }

        let mut node_state1_sources_by_word_id: BTreeMap<i32, BTreeMap<i32, BTreeSet<i64>>> =
            BTreeMap::new();
        for (node_id, source_indexes) in &parent_mapped_scope.matched_node_source_indexes_by_node_id
        {
            for source_index in source_indexes {
                let source = match batch.sources.get(*source_index) {
                    Some(s) => s,
                    None => continue,
                };
                let internal_source_id = self
                    .source_id_by_url
                    .get(source.source_id.as_str())
                    .copied()
                    .unwrap_or_else(|| SecaEngine::fnv1a_64(source.source_id.as_str()));

                if let Some(tokens) = normalized_tokens_by_source_index.get(source_index) {
                    for token in tokens {
                        let word_id = token_to_word_id.get(token).copied().unwrap_or_else(|| {
                            let assigned = next_synthetic_word_id;
                            next_synthetic_word_id -= 1;
                            token_to_word_id.insert(token.clone(), assigned);
                            assigned
                        });
                        node_state1_sources_by_word_id
                            .entry(*node_id)
                            .or_default()
                            .entry(word_id)
                            .or_default()
                            .insert(internal_source_id);
                    }
                }
            }
        }

        let mut prominent_expected_count = 1usize;
        for (node_id, word_ids) in &parent_scope_snapshot.node_word_ids_by_node_id {
            if word_ids.contains(&-1) {
                continue;
            }
            for word_id in word_ids.iter().copied().filter(|id| *id != -1) {
                let state0_count = node_state0_sources_by_word_id
                    .get(node_id)
                    .and_then(|by_word| by_word.get(&word_id))
                    .map(|sources| sources.len())
                    .unwrap_or(0);
                let state1_count = node_state1_sources_by_word_id
                    .get(node_id)
                    .and_then(|by_word| by_word.get(&word_id))
                    .map(|sources| sources.len())
                    .unwrap_or(0);
                let total = state0_count + state1_count;
                if total > prominent_expected_count {
                    prominent_expected_count = total;
                }
            }
        }

        for word_id in &parent_scope_snapshot.expected_word_ids {
            let state0_count = state0_df_parent_scope.get(word_id).copied().unwrap_or(0);
            let state1_count = self
                .baseline_word_legend
                .get(word_id)
                .and_then(|token| {
                    parent_mapped_scope
                        .word_document_frequency_in_scope
                        .get(token)
                        .copied()
                })
                .unwrap_or(0);
            let total = state0_count + state1_count;
            if total > prominent_expected_count {
                prominent_expected_count = total;
            }
        }

        let prominent_expected_count_f = prominent_expected_count.max(1) as f64;

        for word_id in node_word_ids.iter().copied().filter(|w| *w != -1) {
            let state0_count = node_state0_sources_by_word_id
                .get(&parent_node_id)
                .and_then(|by_word| by_word.get(&word_id))
                .map(|sources| sources.len())
                .unwrap_or(0);
            let state1_count = node_state1_sources_by_word_id
                .get(&parent_node_id)
                .and_then(|by_word| by_word.get(&word_id))
                .map(|sources| sources.len())
                .unwrap_or(0);

            next.ancestor_words
                .insert(word_id, (state0_count, state1_count));

            let ratio = (state0_count + state1_count) as f64 / prominent_expected_count_f;
            if ratio >= self.config.seca_thresholds.alpha || node_word_ids.contains(&word_id) {
                next.ancestor_accepted_words
                    .insert(word_id, (state0_count, state1_count));
                next.ancestor_rejected_words.remove(&word_id);
            } else {
                next.ancestor_rejected_words
                    .insert(word_id, (state0_count, state1_count));
                next.ancestor_accepted_words.remove(&word_id);
            }
        }

        next
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct HktUpdateStage {
    pub(super) updated_expected_word_ids: BTreeSet<i32>,
    pub(super) assigned_expected_words_by_node_id: BTreeMap<i32, BTreeSet<i32>>,
    pub(super) node_state1_source_ids_by_node_id: BTreeMap<i32, BTreeSet<i64>>,
    pub(super) node_state1_sources_by_word_id: BTreeMap<i32, BTreeMap<i32, BTreeSet<i64>>>,
    pub(super) token_by_word_id: BTreeMap<i32, String>,
}

impl HktUpdateStage {
    fn apply_to_snapshot(&self, snapshot: &HktScopeSnapshot) -> HktScopeSnapshot {
        let mut updated = snapshot.clone();
        updated.expected_word_ids = self.updated_expected_word_ids.clone();

        for (node_id, assigned_words) in &self.assigned_expected_words_by_node_id {
            updated
                .node_word_ids_by_node_id
                .entry(*node_id)
                .or_default()
                .extend(assigned_words.iter().copied());
        }

        for (node_id, state1_sources) in &self.node_state1_source_ids_by_node_id {
            updated
                .node_source_ids_by_node_id
                .entry(*node_id)
                .or_default()
                .extend(state1_sources.iter().copied());
        }

        updated
    }
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub(crate) struct WordChangeMetrics {
    pub(crate) word_id: i32,
    pub(crate) node_id: i32,
    pub(crate) number_of_sources_before_new_batch: f64,
    pub(crate) number_of_sources_in_new_batch: f64,
    pub(crate) number_of_sources_after_new_batch: f64,
    pub(crate) number_of_sources_before_new_batch_over_number_of_sources_of_old_promin_word_in_hkt:
        f64,
    pub(crate) number_of_intersected_sources_with_old_promin_word_in_node_over_num_sources_of_old_promin_word_in_node:
        f64,
    pub(crate) number_of_sources_after_new_batch_over_number_of_sources_of_new_promin_word_in_hkt:
        f64,
    pub(crate) number_of_intersected_sources_with_new_promin_word_in_node_over_num_sources_of_new_promin_word_in_node:
        f64,
    pub(crate) old_deviation_fom_alpha_parameter: f64,
    pub(crate) old_deviation_fom_beta_parameter: f64,
    pub(crate) new_deviation_fom_alpha_parameter: f64,
    pub(crate) new_deviation_fom_beta_parameter: f64,
    pub(crate) erros_in_deviation_fom_alpha_parameter: f64,
    pub(crate) erros_in_deviation_fom_beta_parameter: f64,
    pub(crate) precentage_of_sources_in_hkt_in_old_batch: f64,
    pub(crate) precentage_of_sources_in_hkt_in_new_batch: f64,
}

#[derive(Debug, Clone, Default)]
pub(super) struct AncestorContext {
    // All ancestor words we've considered so far (for diagnostics / parity)
    ancestor_words: BTreeMap<i32, (usize, usize)>,
    // Paper-critical sets
    ancestor_accepted_words: BTreeMap<i32, (usize, usize)>,
    ancestor_rejected_words: BTreeMap<i32, (usize, usize)>,
}

impl AncestorContext {
    #[allow(dead_code)]
    pub(super) fn with_sets(
        ancestor_words: BTreeMap<i32, (usize, usize)>,
        ancestor_accepted_words: BTreeMap<i32, (usize, usize)>,
        ancestor_rejected_words: BTreeMap<i32, (usize, usize)>,
    ) -> Self {
        Self {
            ancestor_words,
            ancestor_accepted_words,
            ancestor_rejected_words,
        }
    }
}
