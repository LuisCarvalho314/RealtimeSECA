use super::*;
use std::collections::{BTreeMap, BTreeSet};
#[derive(Debug, Clone)]
pub(crate) struct ComputedBatchWordStats {
    pub(crate) unique_words_in_batch: usize,
    pub(crate) known_words_in_batch: usize,
    pub(crate) new_words_in_batch: usize,
    pub(crate) max_word_document_frequency: usize,
    pub(crate) total_sources_in_batch: usize,
    #[allow(dead_code)]
    pub(crate) word_document_frequency: BTreeMap<String, usize>,
}

pub(crate) fn compute_batch_word_stats(
    batch: &SourceBatch,
    baseline_word_legend: &std::collections::BTreeMap<i32, String>,
) -> ComputedBatchWordStats {
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

    ComputedBatchWordStats {
        unique_words_in_batch,
        known_words_in_batch,
        new_words_in_batch,
        max_word_document_frequency,
        total_sources_in_batch: batch.sources.len(),
        word_document_frequency: word_to_source_count,
    }
}
