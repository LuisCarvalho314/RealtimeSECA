use super::*;
impl SecaEngine {
    pub(super) fn prepare_batch(
        &mut self,
        mut batch: SourceBatch,
    ) -> Result<SourceBatch, SecaError> {
        if self.hkt_build_output.is_none() {
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

        if batch
            .sources
            .iter()
            .any(|s| s.source_id.is_empty() || s.batch_index != batch.batch_index)
        {
            return Err(SecaError::StateError {
                message: "invalid source ID or batch index".into(),
            });
        }
        // A source contributes once, even when found in several searches/runs.
        let mut seen = BTreeSet::new();
        batch.sources.retain(|source| {
            !self.source_id_by_url.contains_key(&source.source_id)
                && seen.insert(source.source_id.clone())
        });
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

        self.last_batch_word_stats_summary = Some(BatchWordStatsSummary {
            unique_words_in_batch: batch_stats.unique_words_in_batch,
            known_words_in_batch: batch_stats.known_words_in_batch,
            new_words_in_batch: batch_stats.new_words_in_batch,
            max_word_document_frequency: batch_stats.max_word_document_frequency,
            total_sources_in_batch: batch_stats.total_sources_in_batch,
        });
        Ok(batch)
    }
    pub(super) fn retain_batch(&mut self, batch: &SourceBatch) {
        let mut retained = batch.clone();
        for source in &mut retained.sources {
            source.text = None;
            source.metadata = None;
            source.timestamp_unix_ms = None;
        }
        self.processed_batches.push(retained);
        self.last_processed_batch_index = Some(batch.batch_index);
    }
}
