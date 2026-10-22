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
        let known: BTreeSet<&str> = self
            .baseline_word_legend
            .values()
            .map(String::as_str)
            .collect();
        let (summary, frequencies) = compute_batch_word_stats(&batch, &self.baseline_word_legend);
        let new_tokens: Vec<_> = frequencies
            .into_keys()
            .filter(|token| !known.contains(token.as_str()))
            .collect();
        let mut next_word_id = self.baseline_word_legend.keys().max().copied().unwrap_or(0) + 1;
        for token in new_tokens {
            self.baseline_word_legend.insert(next_word_id, token);
            next_word_id += 1;
        }

        self.last_batch_word_stats_summary = Some(summary);
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

pub(crate) fn compute_batch_word_stats(
    batch: &SourceBatch,
    baseline_word_legend: &std::collections::BTreeMap<i32, String>,
) -> (BatchWordStatsSummary, BTreeMap<String, usize>) {
    let baseline_vocab: BTreeSet<&str> =
        baseline_word_legend.values().map(|s| s.as_str()).collect();

    let mut word_to_source_count: BTreeMap<String, usize> = BTreeMap::new();

    for source in &batch.sources {
        let mut unique_tokens_in_source: BTreeSet<&str> = BTreeSet::new();

        for token in &source.tokens {
            let normalized = token.trim();
            if normalized.is_empty() {
                continue;
            }
            unique_tokens_in_source.insert(normalized);
        }

        for token in unique_tokens_in_source {
            *word_to_source_count.entry(token.to_string()).or_insert(0) += 1;
        }
    }

    let unique_words_in_batch = word_to_source_count.len();
    let known_words_in_batch = word_to_source_count
        .keys()
        .filter(|word| baseline_vocab.contains(word.as_str()))
        .count();
    let new_words_in_batch = unique_words_in_batch.saturating_sub(known_words_in_batch);
    let max_word_document_frequency = word_to_source_count.values().copied().max().unwrap_or(0);

    (
        BatchWordStatsSummary {
            unique_words_in_batch,
            known_words_in_batch,
            new_words_in_batch,
            max_word_document_frequency,
            total_sources_in_batch: batch.sources.len(),
        },
        word_to_source_count,
    )
}
