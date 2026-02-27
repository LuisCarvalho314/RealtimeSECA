use super::*;
use std::collections::HashMap;

impl SecaEngine {
    /// Phase-1 baseline build using HKT construction semantics.
    /// This converts tokenized sources into source-word records and runs the HKT builder.
    pub fn build_baseline_tree(
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
        self.url_by_source_id.clear();
        self.source_batch_index_by_internal_source_id.clear();
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
        self.baseline_source_legend = baseline_conversion.source_legend;

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
        self.has_baseline = true;
        self.processed_batches.clear();
        self.processed_batches.push(baseline_batch.clone());
        self.last_processed_batch_index = Some(baseline_batch.batch_index);
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
            batch_index: baseline_batch.batch_index,
            sources_processed: baseline_batch.sources.len(),
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
    source_legend: std::collections::BTreeMap<i64, String>,
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
    let mut source_legend: std::collections::BTreeMap<i64, String> =
        std::collections::BTreeMap::new();

    // Count number of distinct sources containing each token (not raw frequency).
    let mut source_ids_by_word_id: HashMap<i32, std::collections::BTreeSet<i64>> = HashMap::new();

    for source in &batch.sources {
        let internal_source_id =
            engine.register_source_for_batch(source.source_id.as_str(), batch.batch_index);

        source_legend.insert(internal_source_id, source.source_id.clone());

        let mut source_local_word_ids: std::collections::BTreeSet<i32> =
            std::collections::BTreeSet::new();

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

            source_local_word_ids.insert(assigned_word_id);
        }

        for word_id in source_local_word_ids {
            source_ids_by_word_id
                .entry(word_id)
                .or_default()
                .insert(internal_source_id);
        }
    }

    for record in &mut source_word_records {
        let count = source_ids_by_word_id
            .get(&record.word_id)
            .map(|source_ids| source_ids.len())
            .unwrap_or(0);
        record.word_number_of_sources = count;
    }

    source_word_records.sort_by(|left_record, right_record| {
        right_record
            .word_number_of_sources
            .cmp(&left_record.word_number_of_sources)
            .then_with(|| left_record.source_word_id.cmp(&right_record.source_word_id))
    });

    Ok(BaselineConversionOutput {
        source_word_records,
        word_legend,
        source_legend,
    })
}
