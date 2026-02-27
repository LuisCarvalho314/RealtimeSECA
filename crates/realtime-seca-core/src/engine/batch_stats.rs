use super::*;
use crate::config::SecaThresholdConfig;
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

#[derive(Debug, Clone)]
pub(crate) struct ComputedTriggerMetrics {
    pub(crate) alpha_estimate: f64,
    pub(crate) beta_estimate: f64,
    pub(crate) alpha_error: f64,
    pub(crate) beta_error: f64,
    pub(crate) word_importance_error: f64,
    pub(crate) should_reconstruct: bool,
    pub(crate) trigger_reasons: Vec<String>,
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

pub(crate) fn compute_trigger_metrics_from_batch_stats(
    stats: &ComputedBatchWordStats,
    thresholds: &SecaThresholdConfig,
) -> ComputedTriggerMetrics {
    let unique_words = stats.unique_words_in_batch as f64;
    let known_words = stats.known_words_in_batch as f64;
    let new_words = stats.new_words_in_batch as f64;
    let total_sources = stats.total_sources_in_batch as f64;
    let max_df = stats.max_word_document_frequency as f64;

    // Placeholder-but-useful metrics (deterministic, bounded, interpretable).
    // These are NOT yet the final paper metrics, but they establish the interface.
    let alpha_estimate = if unique_words > 0.0 {
        known_words / unique_words
    } else {
        0.0
    };

    let beta_estimate = if unique_words > 0.0 {
        new_words / unique_words
    } else {
        0.0
    };

    // "Error" placeholders: distance from configured targets.
    let alpha_error = (alpha_estimate - thresholds.alpha).abs();
    let beta_error = (beta_estimate - thresholds.beta).abs();

    // A simple concentration proxy: if one word dominates many sources, this rises.
    // Range approx [0,1] when total_sources > 0.
    let dominance_ratio = if total_sources > 0.0 {
        max_df / total_sources
    } else {
        0.0
    };
    let word_importance_error = dominance_ratio;

    let mut trigger_reasons = Vec::new();

    if alpha_error > thresholds.alpha_error_threshold {
        trigger_reasons.push(format!(
            "alpha_error {:.4} > threshold {:.4}",
            alpha_error, thresholds.alpha_error_threshold
        ));
    }

    if beta_error > thresholds.beta_error_threshold {
        trigger_reasons.push(format!(
            "beta_error {:.4} > threshold {:.4}",
            beta_error, thresholds.beta_error_threshold
        ));
    }

    if word_importance_error > thresholds.word_importance_error_threshold {
        trigger_reasons.push(format!(
            "word_importance_error {:.4} > threshold {:.4}",
            word_importance_error, thresholds.word_importance_error_threshold
        ));
    }

    let should_reconstruct = !trigger_reasons.is_empty();

    ComputedTriggerMetrics {
        alpha_estimate,
        beta_estimate,
        alpha_error,
        beta_error,
        word_importance_error,
        should_reconstruct,
        trigger_reasons,
    }
}
