use super::*;
use std::collections::HashMap;

impl SecaEngine {
    pub fn build_baseline_tree(
        &mut self,
        mut batch: SourceBatch,
    ) -> Result<BatchProcessingResult, SecaError> {
        if self.hkt_build_output.is_some() {
            return Err(SecaError::StateError {
                message: "baseline already exists".into(),
            });
        }
        if batch
            .sources
            .iter()
            .any(|s| s.source_id.is_empty() || s.batch_index != batch.batch_index)
        {
            return Err(SecaError::StateError {
                message: "invalid source ID or batch index".into(),
            });
        }
        let mut seen = BTreeSet::new();
        batch
            .sources
            .retain(|source| seen.insert(source.source_id.clone()));
        let mut candidate = self.clone();
        let result = candidate.build_baseline_inner(batch)?;
        *self = candidate;
        Ok(result)
    }

    /// Phase-1 baseline build using HKT construction semantics.
    /// This converts tokenized sources into source-word records and runs the HKT builder.
    fn build_baseline_inner(
        &mut self,
        baseline_batch: SourceBatch,
    ) -> Result<BatchProcessingResult, SecaError> {
        if baseline_batch.sources.is_empty() {
            return Err(SecaError::StateError {
                message: "baseline batch cannot be empty".to_string(),
            });
        }

        self.baseline_word_legend.clear();
        self.baseline_source_legend.clear();
        self.source_id_by_url.clear();
        self.source_ids_by_batch_index.clear();

        let baseline_conversion = convert_batch_to_source_word_records(self, &baseline_batch)?;
        let source_word_records = baseline_conversion.source_word_records;
        if source_word_records.is_empty() {
            return Err(SecaError::StateError {
                message: "baseline batch produced no source-word records".to_string(),
            });
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

        let hkt_build_output = hkt_builder.build_full_tree(source_word_records, false)?;

        self.baseline_word_legend = baseline_conversion.word_legend;

        let hkt_count = hkt_build_output.hkts_by_id.len();
        let node_count = hkt_build_output.nodes_by_id.len();

        self.next_hkt_id = hkt_build_output
            .hkts_by_id
            .keys()
            .max()
            .map(|id| id + 1)
            .unwrap_or(1);
        self.next_node_id = hkt_build_output
            .nodes_by_id
            .keys()
            .max()
            .map(|id| id + 1)
            .unwrap_or(1);

        self.hkt_build_output = Some(hkt_build_output);
        self.processed_batches.clear();
        self.retain_batch(&baseline_batch);
        self.last_update_explanation = Some(UpdateExplanation {
            summary: format!(
                "Baseline HKT tree initialized ({} HKTs, {} nodes)",
                hkt_count, node_count
            ),
            reason_codes: vec![
                "BASELINE_INITIALIZED".to_string(),
                "HKT_BUILDER_PHASE1".to_string(),
            ],
        });

        Ok(BatchProcessingResult {
            decision_diagnostics: Vec::new(),
            display_diagnostics: self.display_diagnostics(None)?,
            batch_index: baseline_batch.batch_index,
            sources_processed: baseline_batch.sources.len(),
            active_source_count: self.baseline_source_legend.len(),
            sources_forgotten: 0,
            hkts_inspected: 0,
            reconstructed_hkt_ids: Vec::new(),
            reconstruction_triggered: false,
            notes: vec![
                "Baseline build completed with phase-1 HKT builder".to_string(),
                format!("HKTs: {}", hkt_count),
                format!("Nodes: {}", node_count),
            ],
        })
    }
}

#[derive(Debug)]
struct BaselineConversionOutput {
    source_word_records: Vec<SourceWordRecord>,
    word_legend: std::collections::BTreeMap<i32, String>,
}

/// Converts tokenized sources to `SourceWordRecord` rows and computes `word_number_of_sources`,
/// matching the shape expected by the HKT builder.
fn convert_batch_to_source_word_records(
    engine: &mut SecaEngine,
    batch: &SourceBatch,
) -> Result<BaselineConversionOutput, SecaError> {
    let mut source_word_records: Vec<SourceWordRecord> = Vec::new();
    let mut source_word_id_counter: i32 = 1;
    let mut word_id_by_token: HashMap<String, i32> = HashMap::new();
    let mut next_word_id: i32 = 1;

    let mut word_legend: std::collections::BTreeMap<i32, String> =
        std::collections::BTreeMap::new();

    for source in &batch.sources {
        let internal_source_id =
            engine.register_source_for_batch(source.source_id.as_str(), batch.batch_index);

        for token in &source.tokens {
            let normalized_token = token.trim();
            if normalized_token.is_empty() {
                continue;
            }

            let assigned_word_id =
                if let Some(existing_word_id) = word_id_by_token.get(normalized_token) {
                    *existing_word_id
                } else {
                    let new_word_id = next_word_id;
                    word_id_by_token.insert(normalized_token.to_string(), new_word_id);
                    word_legend.insert(new_word_id, normalized_token.to_string());
                    next_word_id += 1;
                    new_word_id
                };

            source_word_records.push(SourceWordRecord {
                source_word_id: source_word_id_counter,
                source_id: internal_source_id,
                word_id: assigned_word_id,
                word: Some(normalized_token.to_string()),
                word_number_of_sources: 0, // populated below
            });
            source_word_id_counter += 1;
        }
    }

    crate::tree::hkt_builder::update_document_frequencies(&mut source_word_records);

    Ok(BaselineConversionOutput {
        source_word_records,
        word_legend,
    })
}
