use crate::config::SecaConfig;
use crate::error::SecaError;
use crate::tree::{HktBuildOutput, HktBuilder, SourceWordRecord};
use crate::types::{
    BaselineHktExport, BaselineHktVerboseExport, BaselineNodeExport, BaselineNodeVerboseExport,
    BaselineTreeExport, BaselineTreeVerboseExport, BatchProcessingResult, BatchWordStatsSummary,
    ClusteringResult, EngineSnapshot, SourceBatch, SourceLegendEntry, UpdateExplanation,
    VerboseSourceRef, VerboseWordRef, WordLegendEntry,
};
use crate::ENGINE_VERSION;
use std::collections::{BTreeMap, BTreeSet, HashMap};

mod baseline;
pub(crate) mod batch_stats;
mod rebuild;
mod scope_mapping;
mod snapshotting;
mod trigger;

#[cfg(test)]
mod tests;
use self::batch_stats::{compute_batch_word_stats, compute_trigger_metrics_from_batch_stats};

#[derive(Debug, Clone)]
pub struct SecaEngine {
    config: SecaConfig,
    last_processed_batch_index: Option<u32>,
    last_update_explanation: Option<UpdateExplanation>,
    has_baseline: bool,
    hkt_build_output: Option<HktBuildOutput>,
    baseline_word_legend: BTreeMap<i32, String>,
    baseline_source_legend: BTreeMap<i32, String>,
    processed_batches: Vec<SourceBatch>,
    last_batch_word_stats_summary: Option<BatchWordStatsSummary>,
}

impl SecaEngine {
    pub fn new(config: SecaConfig) -> Result<Self, SecaError> {
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
            has_baseline: false,
            hkt_build_output: None,
            baseline_word_legend: std::collections::BTreeMap::new(),
            baseline_source_legend: std::collections::BTreeMap::new(),
            processed_batches: Vec::new(),
            last_batch_word_stats_summary: None,
        })
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

    pub fn process_batch(
        &mut self,
        batch: SourceBatch,
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

        let sources_processed = batch.sources.len();

        // Placeholder behavior for now: no significance/reconstruction yet.
        let mut notes = vec![
            "Incremental process_batch skeleton executed".to_string(),
            "SECA significance/reconstruction logic not implemented yet".to_string(),
        ];

        let batch_stats = compute_batch_word_stats(&batch, &self.baseline_word_legend);

        notes.push(format!(
            "Batch word stats: unique={}, known={}, new={}, max_df={}, sources={}",
            batch_stats.unique_words_in_batch,
            batch_stats.known_words_in_batch,
            batch_stats.new_words_in_batch,
            batch_stats.max_word_document_frequency,
            batch_stats.total_sources_in_batch
        ));

        let trigger_plan = self.evaluate_seca_trigger_plan_for_batch(&batch)?;
        debug_assert_eq!(trigger_plan.batch_index, batch.batch_index);

        notes.extend(trigger_plan.notes.clone());

        let reconstruction_triggered = trigger_plan.any_reconstruction_triggered;

        // Store batch according to memory mode
        self.processed_batches.push(batch.clone());

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

        match self.config.memory_mode {
            crate::config::MemoryMode::Full => {
                notes.push(format!(
                    "Memory mode: Full (stored batches: {})",
                    self.processed_batches.len()
                ));
            }
            crate::config::MemoryMode::SlidingWindow => {
                if let Some(max_batches) = self.config.max_batches_in_memory {
                    let max_batches_usize =
                        usize::try_from(max_batches).map_err(|_| SecaError::StateError {
                            message: "max_batches_in_memory conversion overflow".to_string(),
                        })?;

                    if max_batches_usize == 0 {
                        return Err(SecaError::InvalidConfiguration {
                            message: "max_batches_in_memory must be > 0 when using SlidingWindow"
                                .to_string(),
                        });
                    }

                    if self.processed_batches.len() > max_batches_usize {
                        let excess = self.processed_batches.len() - max_batches_usize;
                        self.processed_batches.drain(0..excess);
                    }

                    notes.push(format!(
                        "Memory mode: SlidingWindow (stored batches: {}, max: {})",
                        self.processed_batches.len(),
                        max_batches
                    ));
                } else {
                    notes.push(
                    "Memory mode: SlidingWindow (max_batches_in_memory not set; no trimming applied)"
                        .to_string(),
                );
                }
            }
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
            self.rebuild_from_stored_batches()?;
            notes.push(
                "Reconstruction action: full rebuild from stored batches completed".to_string(),
            );
        } else {
            notes.push("Reconstruction action: no rebuild performed".to_string());
        }

        let mut reason_codes = vec![
            "INCREMENTAL_BATCH_PROCESSED".to_string(),
            "BATCH_WORD_STATS_COMPUTED".to_string(),
            "SECA_SCOPE_MAPPING_COMPLETED".to_string(),
            "SECA_SCOPE_METRICS_COMPUTED".to_string(),
            "SECA_TRIGGER_EVALUATED".to_string(),
        ];

        if reconstruction_triggered {
            reason_codes.push("SECA_RECONSTRUCTION_TRIGGERED".to_string());
            reason_codes.push("SECA_FULL_REBUILD_EXECUTED".to_string());
        } else {
            reason_codes.push("SECA_RECONSTRUCTION_SKIPPED".to_string());
        }
        self.last_update_explanation = Some(UpdateExplanation {
            summary: format!(
                "Processed batch {} (trigger evaluated: reconstruction_triggered={})",
                batch.batch_index, reconstruction_triggered
            ),
            reason_codes,
        });

        Ok(BatchProcessingResult {
            batch_index: batch.batch_index,
            sources_processed,
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
}
