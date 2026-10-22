use crate::config::SecaConfig;
use crate::error::SecaError;
use crate::tree::{HktBuildOutput, HktBuilder, SourceWordRecord};
use crate::types::{
    BaselineHktExport, BaselineHktVerboseExport, BaselineNodeExport, BaselineNodeVerboseExport,
    BaselineTreeExport, BaselineTreeVerboseExport, BatchProcessingResult, BatchWordStatsSummary,
    EngineSnapshot, SourceBatch, SourceLegendEntry, UpdateExplanation, VerboseSourceRef,
    VerboseWordRef, WordLegendEntry,
};
use crate::ENGINE_VERSION;
use std::collections::{BTreeMap, BTreeSet};

mod baseline;
pub(crate) mod batch_stats;
mod context;
mod diagnostics;
mod evidence;
mod forgetting;
mod input;
mod metrics;
mod rebuild;
mod scope_mapping;
mod snapshotting;
mod sources;
mod trigger;
mod validation;

#[cfg(test)]
mod tests;
use self::batch_stats::compute_batch_word_stats;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SecaEngine {
    config: SecaConfig,
    last_processed_batch_index: Option<u32>,
    last_update_explanation: Option<UpdateExplanation>,
    hkt_build_output: Option<HktBuildOutput>,
    baseline_word_legend: BTreeMap<i32, String>,
    baseline_source_legend: BTreeMap<i64, String>,
    processed_batches: Vec<SourceBatch>,
    last_batch_word_stats_summary: Option<BatchWordStatsSummary>,
    source_id_by_url: BTreeMap<String, i64>,
    source_ids_by_batch_index: BTreeMap<u32, BTreeSet<i64>>,
    next_hkt_id: i32,
    next_node_id: i32,
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

        for (label, value) in [
            ("alpha_error_threshold", thresholds.alpha_error_threshold),
            ("beta_error_threshold", thresholds.beta_error_threshold),
            (
                "word_importance_error_threshold",
                thresholds.word_importance_error_threshold,
            ),
        ] {
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
            last_processed_batch_index: None,
            last_update_explanation: None,
            hkt_build_output: None,
            baseline_word_legend: std::collections::BTreeMap::new(),
            baseline_source_legend: std::collections::BTreeMap::new(),
            processed_batches: Vec::new(),
            last_batch_word_stats_summary: None,
            source_id_by_url: BTreeMap::new(),
            source_ids_by_batch_index: BTreeMap::new(),
            next_hkt_id: 1,
            next_node_id: 1,
        })
    }

    pub fn config(&self) -> &SecaConfig {
        &self.config
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
        batch: SourceBatch,
    ) -> Result<BatchProcessingResult, SecaError> {
        let batch = self.prepare_batch(batch)?;
        let mut plan = self.evaluate_seca_trigger_plan_for_batch(&batch)?;
        self.retain_batch(&batch);
        let replacements = self.rebuild_selected_hkts_from_trigger_plan(&batch, &plan)?;
        for diagnostic in &mut plan.diagnostics {
            if let Some(id) = replacements.get(&diagnostic.hkt_id) {
                diagnostic.output_hkt_id = Some(*id);
            }
        }
        let sources_forgotten = self
            .apply_seca_light_pruning_if_enabled(batch.batch_index)?
            .unwrap_or(0);
        self.last_update_explanation = Some(UpdateExplanation {
            summary: format!("Processed batch {}", batch.batch_index),
            reason_codes: vec![
                "SECA_TRIGGER_EVALUATED".into(),
                if !plan.targets.is_empty() {
                    "SECA_RECONSTRUCTION_TRIGGERED".into()
                } else {
                    "SECA_RECONSTRUCTION_SKIPPED".into()
                },
            ],
        });
        Ok(BatchProcessingResult {
            display_diagnostics: self.display_diagnostics(Some(&batch))?,
            hkts_inspected: plan.diagnostics.len(),
            decision_diagnostics: plan.diagnostics,
            batch_index: batch.batch_index,
            sources_processed: batch.sources.len(),
            sources_forgotten,
            active_source_count: self.baseline_source_legend.len(),
            reconstructed_hkt_ids: plan.targets.keys().copied().collect(),
            reconstruction_triggered: !plan.targets.is_empty(),
            notes: Vec::new(),
        })
    }
    pub fn stored_batch_count(&self) -> usize {
        self.processed_batches.len()
    }

    pub fn last_batch_word_stats_summary(&self) -> Option<&BatchWordStatsSummary> {
        self.last_batch_word_stats_summary.as_ref()
    }
}
