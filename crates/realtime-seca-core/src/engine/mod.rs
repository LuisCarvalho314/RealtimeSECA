use crate::config::SecaConfig;
use crate::error::SecaError;
use crate::tree::{Hkt, HktBuildOutput, HktBuilder, SourceWordRecord};
use crate::types::{
    BaselineHktExport, BaselineHktVerboseExport, BaselineNodeExport, BaselineNodeVerboseExport,
    BaselineTreeExport, BaselineTreeVerboseExport, BatchProcessingResult, BatchWordStatsSummary,
    ClusteringResult, EngineSnapshot, SourceBatch, SourceLegendEntry, UpdateExplanation,
    VerboseSourceRef, VerboseWordRef, WordLegendEntry,
};
use crate::ENGINE_VERSION;
use std::collections::{BTreeMap, BTreeSet};

mod baseline;
pub(crate) mod batch_stats;
pub mod rebuild;
mod scope_mapping;
mod snapshotting;
mod trigger;

#[cfg(test)]
mod tests;
use self::batch_stats::{compute_batch_word_stats, compute_trigger_metrics_from_batch_stats};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SecaEngine {
    config: SecaConfig,
    rebuild_mode: rebuild::RebuildMode,
    last_processed_batch_index: Option<u32>,
    last_update_explanation: Option<UpdateExplanation>,
    has_baseline: bool,
    hkt_build_output: Option<HktBuildOutput>,
    baseline_word_legend: BTreeMap<i32, String>,
    baseline_source_legend: BTreeMap<i64, String>,
    processed_batches: Vec<SourceBatch>,
    last_batch_word_stats_summary: Option<BatchWordStatsSummary>,
    source_id_by_url: BTreeMap<String, i64>,
    url_by_source_id: BTreeMap<i64, String>,
    source_batch_index_by_internal_source_id: BTreeMap<i64, u32>,
    source_ids_by_batch_index: BTreeMap<u32, BTreeSet<i64>>,
    archived_subtrees_by_root_id: BTreeMap<i32, HktBuildOutput>,
    logically_removed_hkts_by_id: BTreeMap<i32, LogicalRemovedHkt>,
    next_hkt_id: i32,
    next_node_id: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LogicalRemovedHkt {
    hkt: Hkt,
    old_parent_node_id: i32,
}

#[derive(Debug, Clone, Default)]
struct SecaLightPruneReport {
    active_source_count: usize,
    pruned_source_count: usize,
    pruned_node_count: usize,
    pruned_hkt_count: usize,
}

impl SecaEngine {
    pub fn new(config: SecaConfig) -> Result<Self, SecaError> {
        if config.memory_mode == crate::config::MemoryMode::SlidingWindow
            && config.max_batches_in_memory.unwrap_or(0) == 0
        {
            return Err(SecaError::InvalidConfiguration {
                message: "SlidingWindow requires max_batches_in_memory > 0".into(),
            });
        }
        let thresholds = &config.seca_thresholds;
        if !(0.0..=1.0).contains(&thresholds.alpha) {
            return Err(SecaError::InvalidConfiguration {
                message: "seca_thresholds.alpha must be in [0, 1]".to_string(),
            });
        }

        if !(0.0..=1.0).contains(&thresholds.beta) {
            return Err(SecaError::InvalidConfiguration {
                message: "seca_thresholds.beta must be in [0, 1]".to_string(),
            });
        }

        let option_thresholds: [(&str, f64); 8] = [
            (
                "seca_thresholds.alpha_option1_threshold",
                thresholds.alpha_option1_threshold,
            ),
            (
                "seca_thresholds.alpha_option2_threshold",
                thresholds.alpha_option2_threshold,
            ),
            (
                "seca_thresholds.alpha_option3_threshold",
                thresholds.alpha_option3_threshold,
            ),
            (
                "seca_thresholds.beta_option1_threshold",
                thresholds.beta_option1_threshold,
            ),
            (
                "seca_thresholds.beta_option2_threshold",
                thresholds.beta_option2_threshold,
            ),
            (
                "seca_thresholds.beta_option3_threshold",
                thresholds.beta_option3_threshold,
            ),
            (
                "seca_thresholds.word_importance_option1_threshold",
                thresholds.word_importance_option1_threshold,
            ),
            (
                "seca_thresholds.word_importance_option2_threshold",
                thresholds.word_importance_option2_threshold,
            ),
        ];

        for (label, value) in option_thresholds {
            if !(0.0..=1.0).contains(&value) {
                return Err(SecaError::InvalidConfiguration {
                    message: format!("{label} must be in [0, 1]"),
                });
            }
        }

        let hkt = &config.hkt_builder;
        if !(0.0..=1.0).contains(&hkt.minimum_threshold_against_max_word_count) {
            return Err(SecaError::InvalidConfiguration {
                message: "hkt_builder.minimum_threshold_against_max_word_count must be in [0, 1]"
                    .to_string(),
            });
        }

        if !(0.0..=1.0).contains(&hkt.similarity_threshold) {
            return Err(SecaError::InvalidConfiguration {
                message: "hkt_builder.similarity_threshold must be in [0, 1]".to_string(),
            });
        }

        Ok(Self {
            config,
            rebuild_mode: rebuild::RebuildMode::FullFromAllBatches,
            last_processed_batch_index: None,
            last_update_explanation: None,
            has_baseline: false,
            hkt_build_output: None,
            baseline_word_legend: std::collections::BTreeMap::new(),
            baseline_source_legend: std::collections::BTreeMap::new(),
            processed_batches: Vec::new(),
            last_batch_word_stats_summary: None,
            source_id_by_url: BTreeMap::new(),
            url_by_source_id: BTreeMap::new(),
            source_batch_index_by_internal_source_id: BTreeMap::new(),
            source_ids_by_batch_index: BTreeMap::new(),
            archived_subtrees_by_root_id: BTreeMap::new(),
            logically_removed_hkts_by_id: BTreeMap::new(),
            next_hkt_id: 1,
            next_node_id: 1,
        })
    }
    pub fn set_rebuild_mode(&mut self, rebuild_mode: rebuild::RebuildMode) {
        self.rebuild_mode = rebuild_mode;
    }
    pub fn rebuild_mode(&self) -> rebuild::RebuildMode {
        self.rebuild_mode
    }
    pub fn config(&self) -> &SecaConfig {
        &self.config
    }

    pub fn detect_clusters_direct(&self) -> Result<ClusteringResult, SecaError> {
        if !self.has_baseline {
            return Err(SecaError::StateError {
                message: "baseline must be built before clustering".to_string(),
            });
        }

        Ok(ClusteringResult {
            cluster_count: 0,
            notes: vec!["direct clustering stub".to_string()],
        })
    }

    pub fn detect_clusters_indirect(&self) -> Result<ClusteringResult, SecaError> {
        if !self.has_baseline {
            return Err(SecaError::StateError {
                message: "baseline must be built before clustering".to_string(),
            });
        }

        Ok(ClusteringResult {
            cluster_count: 0,
            notes: vec!["indirect clustering stub".to_string()],
        })
    }

    /// Commit an update only when every stage succeeds.
    pub fn process_batch(
        &mut self,
        batch: SourceBatch,
    ) -> Result<BatchProcessingResult, SecaError> {
        let mut candidate = self.clone();
        let result = candidate.process_batch_inner(batch)?;
        *self = candidate;
        Ok(result)
    }

    fn process_batch_inner(
        &mut self,
        mut batch: SourceBatch,
    ) -> Result<BatchProcessingResult, SecaError> {
        if !self.has_baseline || self.hkt_build_output.is_none() {
            return Err(SecaError::StateError {
                message: "baseline tree has not been built yet".to_string(),
            });
        }

        let expected_next_batch_index = self
            .last_processed_batch_index
            .map(|index| index.saturating_add(1))
            .unwrap_or(batch.batch_index);

        if batch.batch_index <= self.last_processed_batch_index.unwrap_or(0) {
            return Err(SecaError::InvalidConfiguration {
                message: format!(
                    "batch_index {} must be greater than last processed batch index {}",
                    batch.batch_index,
                    self.last_processed_batch_index.unwrap_or(0)
                ),
            });
        }

        if batch.batch_index != expected_next_batch_index {
            return Err(SecaError::InvalidConfiguration {
                message: format!(
                    "batch_index {} is out of sequence; expected {}",
                    batch.batch_index, expected_next_batch_index
                ),
            });
        }

        // A source contributes once, even when found in several searches/runs.
        let mut seen: BTreeSet<String> = self
            .processed_batches
            .iter()
            .flat_map(|b| b.sources.iter().map(|s| s.source_id.clone()))
            .collect();
        batch
            .sources
            .retain(|source| seen.insert(source.source_id.clone()));
        if batch
            .sources
            .iter()
            .any(|s| s.source_id.is_empty() || s.batch_index != batch.batch_index)
        {
            return Err(SecaError::StateError {
                message: "invalid source ID or batch index".into(),
            });
        }
        self.register_batch_sources(&batch);
        let known: BTreeSet<String> = self.baseline_word_legend.values().cloned().collect();
        let batch_stats = compute_batch_word_stats(&batch, &self.baseline_word_legend);
        let new_tokens: BTreeSet<String> = batch
            .sources
            .iter()
            .flat_map(|s| s.tokens.iter())
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty() && !known.contains(t))
            .collect();
        let mut next_word_id = self.baseline_word_legend.keys().max().copied().unwrap_or(0) + 1;
        for token in new_tokens {
            self.baseline_word_legend.insert(next_word_id, token);
            next_word_id += 1;
        }

        let sources_processed = batch.sources.len();

        // HKT-local equations, rather than batch-level diagnostics, drive updates.
        let mut notes = vec![
            "Incremental SECA batch update executed".to_string(),
            "Top-down HKT significance and selective reconstruction enabled".to_string(),
        ];

        notes.push(format!(
            "Batch word stats: unique={}, known={}, new={}, max_df={}, sources={}",
            batch_stats.unique_words_in_batch,
            batch_stats.known_words_in_batch,
            batch_stats.new_words_in_batch,
            batch_stats.max_word_document_frequency,
            batch_stats.total_sources_in_batch
        ));

        let mut trigger_plan = self.evaluate_seca_trigger_plan_for_batch(&batch)?;
        debug_assert_eq!(trigger_plan.batch_index, batch.batch_index);

        notes.extend(trigger_plan.notes.clone());

        let reconstruction_triggered = trigger_plan.any_reconstruction_triggered;

        // Store batch according to memory mode
        let mut retained_batch = batch.clone();
        for source in &mut retained_batch.sources {
            source.text = None;
            source.metadata = None;
            source.timestamp_unix_ms = None;
        }
        self.processed_batches.push(retained_batch);

        // Stage 4A scaffold metrics at batch-level (diagnostic only for now).
        // These do NOT drive reconstruction decisions yet; recursive HKT scope trigger plan remains authoritative.
        let batch_trigger_metrics =
            compute_trigger_metrics_from_batch_stats(&batch_stats, &self.config.seca_thresholds);

        notes.push(format!(
            "Batch trigger metrics (scaffold): alpha_est={:.4}, beta_est={:.4}, alpha_err={:.4}, beta_err={:.4}, word_importance_error={:.4}",
            batch_trigger_metrics.alpha_estimate,
            batch_trigger_metrics.beta_estimate,
            batch_trigger_metrics.alpha_error,
            batch_trigger_metrics.beta_error,
            batch_trigger_metrics.word_importance_error,
        ));

        if batch_trigger_metrics.should_reconstruct {
            for reason in &batch_trigger_metrics.trigger_reasons {
                notes.push(format!(
                    "Batch trigger metrics reason (scaffold): {}",
                    reason
                ));
            }
        } else {
            notes.push(
                "Batch trigger metrics reason (scaffold): no thresholds exceeded".to_string(),
            );
        }

        self.last_processed_batch_index = Some(batch.batch_index);

        self.last_batch_word_stats_summary = Some(BatchWordStatsSummary {
            unique_words_in_batch: batch_stats.unique_words_in_batch,
            known_words_in_batch: batch_stats.known_words_in_batch,
            new_words_in_batch: batch_stats.new_words_in_batch,
            max_word_document_frequency: batch_stats.max_word_document_frequency,
            total_sources_in_batch: batch_stats.total_sources_in_batch,
        });

        if reconstruction_triggered {
            notes.push(format!(
                "SECA trigger action plan: HKT-local reconstruction requested for {:?}",
                trigger_plan.reconstruct_hkt_ids
            ));
        } else {
            notes.push(
                "SECA trigger action plan: no HKT-local reconstruction requested".to_string(),
            );
        }

        let mut reason_codes = vec![
            "INCREMENTAL_BATCH_PROCESSED".to_string(),
            "BATCH_WORD_STATS_COMPUTED".to_string(),
            "SECA_SCOPE_MAPPING_COMPLETED".to_string(),
            "SECA_SCOPE_METRICS_COMPUTED".to_string(),
            "SECA_TRIGGER_EVALUATED".to_string(),
        ];
        let mut seca_light_pruning_applied = false;
        let mut sources_forgotten = 0;

        if reconstruction_triggered {
            reason_codes.push("SECA_RECONSTRUCTION_TRIGGERED".to_string());
            reason_codes.push("SECA_REBUILD_ACTION_REQUESTED".to_string());
        } else {
            reason_codes.push("SECA_RECONSTRUCTION_SKIPPED".to_string());
        }

        if reconstruction_triggered {
            let selected_rebuild_plan =
                self.build_selected_hkt_rebuild_plan_from_trigger_plan(&batch, &trigger_plan)?;
            notes.push(self.format_selected_hkt_rebuild_plan_note(&selected_rebuild_plan));

            if self.rebuild_mode == crate::engine::rebuild::RebuildMode::SubtreeTargeted {
                let dry_run = self.build_selected_hkt_subtree_dry_run_report(
                    &selected_rebuild_plan,
                    &trigger_plan,
                )?;
                notes.push(self.format_selected_hkt_subtree_dry_run_report_note(&dry_run));
            }

            if self.rebuild_mode == crate::engine::rebuild::RebuildMode::SubtreeTargeted {
                let replacements =
                    self.rebuild_selected_hkts_from_trigger_plan(&batch, &trigger_plan)?;
                for diagnostic in &mut trigger_plan.diagnostics {
                    if let Some(id) = replacements.get(&diagnostic.hkt_id) {
                        diagnostic.output_hkt_id = Some(*id);
                    }
                }
                notes.push(
                    "Reconstruction action: selected-HKT subtree rebuild completed".to_string(),
                );
            } else {
                self.execute_rebuild_action_for_trigger_plan(&trigger_plan.reconstruct_hkt_ids)?;
                // Backward-compatible note for existing tests
                notes.push(
                    "Reconstruction action: full rebuild from stored batches completed".to_string(),
                );
            }

            if self.rebuild_mode != crate::engine::rebuild::RebuildMode::SubtreeTargeted {
                // Transitional note for non-subtree modes (keeps visibility of the selected-HKT intent)
                notes.push(
                    "Reconstruction action: selected-HKT rebuild requested (currently full rebuild fallback) completed"
                        .to_string(),
                );
            }

            let report =
                self.build_selected_hkt_execution_report(&batch, &selected_rebuild_plan)?;
            notes.push(self.format_selected_hkt_execution_report_note(&report));

            match self.rebuild_mode {
                crate::engine::rebuild::RebuildMode::FullFromAllBatches => {
                    reason_codes.push("SECA_FULL_REBUILD_EXECUTED".to_string());
                }
                crate::engine::rebuild::RebuildMode::HybridFullOnRootTrigger => {
                    reason_codes.push("SECA_HYBRID_REBUILD_FALLBACK_EXECUTED".to_string());
                }
                crate::engine::rebuild::RebuildMode::SubtreeTargeted => {
                    reason_codes.push("SECA_SELECTED_REBUILD_EXECUTED".to_string());
                }
            }
        } else {
            // Backward-compatible note for existing tests
            notes.push("Reconstruction action: no rebuild performed".to_string());
        }

        if let Some(prune_report) = self.apply_seca_light_pruning_if_enabled(batch.batch_index)? {
            notes.push(format!(
                "SECA-Light pruning: active_sources={}, pruned_sources={}, pruned_nodes={}, pruned_hkts={}",
                prune_report.active_source_count,
                prune_report.pruned_source_count,
                prune_report.pruned_node_count,
                prune_report.pruned_hkt_count
            ));
            sources_forgotten = prune_report.pruned_source_count;
            seca_light_pruning_applied = true;
        }

        if seca_light_pruning_applied {
            reason_codes.push("SECA_LIGHT_PRUNING_APPLIED".to_string());
        }

        self.last_update_explanation = Some(UpdateExplanation {
            summary: format!(
                "Processed batch {} (trigger evaluated: reconstruction_triggered={})",
                batch.batch_index, reconstruction_triggered
            ),
            reason_codes,
        });

        // Never join stale IDs to scopes introduced by a full rebuild/pruning.
        for diagnostic in &mut trigger_plan.diagnostics {
            if self.rebuild_mode != rebuild::RebuildMode::SubtreeTargeted
                && reconstruction_triggered
            {
                diagnostic.output_hkt_id = None;
            } else if !diagnostic.output_hkt_id.is_some_and(|id| {
                self.hkt_build_output
                    .as_ref()
                    .is_some_and(|tree| tree.hkts_by_id.contains_key(&id))
            }) {
                diagnostic.output_hkt_id = None;
            }
        }
        let hkts_inspected = trigger_plan.diagnostics.len();
        Ok(BatchProcessingResult {
            display_diagnostics: self.display_diagnostics(Some(&batch))?,
            decision_diagnostics: trigger_plan.diagnostics,
            batch_index: batch.batch_index,
            sources_processed,
            sources_forgotten,
            active_source_count: self.baseline_source_legend.len(),
            hkts_inspected,
            reconstructed_hkt_ids: trigger_plan.reconstruct_hkt_ids.clone(),
            reconstruction_triggered,
            notes,
        })
    }
    pub fn stored_batch_count(&self) -> usize {
        self.processed_batches.len()
    }

    pub fn last_batch_word_stats_summary(&self) -> Option<&BatchWordStatsSummary> {
        self.last_batch_word_stats_summary.as_ref()
    }
    fn fnv1a_64(input: &str) -> i64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut hash = FNV_OFFSET;
        for b in input.as_bytes() {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash as i64
    }

    fn stable_source_id(&mut self, url: &str) -> i64 {
        if let Some(existing) = self.source_id_by_url.get(url) {
            return *existing;
        }

        let mut attempt = 0usize;
        loop {
            let candidate = if attempt == 0 {
                Self::fnv1a_64(url)
            } else {
                Self::fnv1a_64(&format!("{url}#{attempt}"))
            };

            match self.url_by_source_id.get(&candidate) {
                None => {
                    self.source_id_by_url.insert(url.to_string(), candidate);
                    self.url_by_source_id.insert(candidate, url.to_string());
                    return candidate;
                }
                Some(existing_url) if existing_url == url => {
                    self.source_id_by_url.insert(url.to_string(), candidate);
                    return candidate;
                }
                Some(_) => {
                    attempt += 1;
                }
            }
        }
    }

    fn register_source_for_batch(&mut self, external_source_id: &str, batch_index: u32) -> i64 {
        let internal_source_id = self.stable_source_id(external_source_id);
        self.baseline_source_legend
            .insert(internal_source_id, external_source_id.to_string());
        self.source_batch_index_by_internal_source_id
            .insert(internal_source_id, batch_index);
        self.source_ids_by_batch_index
            .entry(batch_index)
            .or_default()
            .insert(internal_source_id);
        internal_source_id
    }

    fn register_batch_sources(&mut self, batch: &SourceBatch) {
        for source in &batch.sources {
            self.register_source_for_batch(source.source_id.as_str(), batch.batch_index);
        }
    }

    fn apply_seca_light_pruning_if_enabled(
        &mut self,
        current_batch_index: u32,
    ) -> Result<Option<SecaLightPruneReport>, SecaError> {
        if self.config.memory_mode != crate::config::MemoryMode::SlidingWindow {
            return Ok(None);
        }

        let Some(gamma) = self.config.max_batches_in_memory else {
            return Ok(None);
        };

        let gamma_usize = usize::try_from(gamma).map_err(|_| SecaError::StateError {
            message: "max_batches_in_memory conversion overflow".to_string(),
        })?;

        if gamma_usize == 0 {
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
            self.source_batch_index_by_internal_source_id
                .remove(source_id);
            if let Some(external_source_id) = self.url_by_source_id.remove(source_id) {
                self.source_id_by_url.remove(external_source_id.as_str());
            }
            self.baseline_source_legend.remove(source_id);
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
                    node.number_of_sources_for_display = Some(node.source_ids.len());
                    tree.nodes_by_id.insert(node.node_id, node.clone());
                }
            }
        };
        if let Some(tree) = self.hkt_build_output.as_mut() {
            prune_tree(tree);
        }
        // Historical trees belong in output snapshots, not active model memory.
        self.archived_subtrees_by_root_id.clear();
        self.logically_removed_hkts_by_id.clear();
        Ok(Some(SecaLightPruneReport {
            active_source_count: active_source_ids.len(),
            pruned_source_count: pruned_source_ids.len(),
            pruned_node_count: 0,
            pruned_hkt_count: 0,
        }))
    }

    fn compute_active_sources_for_window(&self, window_start_batch_index: u32) -> BTreeSet<i64> {
        let mut active_source_ids = BTreeSet::new();
        for (batch_index, source_ids) in &self.source_ids_by_batch_index {
            if *batch_index >= window_start_batch_index {
                active_source_ids.extend(source_ids.iter().copied());
            }
        }
        active_source_ids
    }

    fn rebuild_source_registration_from_processed_batches(&mut self) {
        self.baseline_source_legend.clear();
        self.source_id_by_url.clear();
        self.url_by_source_id.clear();
        self.source_batch_index_by_internal_source_id.clear();
        self.source_ids_by_batch_index.clear();

        let all_batches = self.processed_batches.clone();
        for batch in all_batches {
            self.register_batch_sources(&batch);
        }
    }
}
