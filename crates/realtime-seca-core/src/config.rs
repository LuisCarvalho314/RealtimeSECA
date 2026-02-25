use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemoryMode {
    Full,
    SlidingWindow,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HktBuilderConfig {
    /// Equivalent to "minimum_threshold_against_max_word_count" in the C# HKT builder.
    pub minimum_threshold_against_max_word_count: f64,
    /// Equivalent to "similarity_threshold" in the C# HKT builder.
    pub similarity_threshold: f64,
    /// Minimum number of sources in a node to create a child branch HKT.
    pub minimum_number_of_sources_to_create_branch_for_node: usize,
}

impl Default for HktBuilderConfig {
    fn default() -> Self {
        Self {
            minimum_threshold_against_max_word_count: 0.5,
            similarity_threshold: 0.5,
            minimum_number_of_sources_to_create_branch_for_node: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SecaThresholdConfig {
    /// Paper-level alpha parameter (SECA update/reconstruction stage).
    pub alpha: f64,
    /// Paper-level beta parameter (SECA update/reconstruction stage).
    pub beta: f64,
    pub alpha_error_threshold: f64,
    pub beta_error_threshold: f64,
    pub word_importance_error_threshold: f64,
}

impl Default for SecaThresholdConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.5,
            alpha_error_threshold: 0.1,
            beta_error_threshold: 0.1,
            word_importance_error_threshold: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SecaConfig {
    pub hkt_builder: HktBuilderConfig,
    pub seca_thresholds: SecaThresholdConfig,
    pub memory_mode: MemoryMode,
    pub max_batches_in_memory: Option<u32>,
}

impl Default for SecaConfig {
    fn default() -> Self {
        Self {
            hkt_builder: HktBuilderConfig::default(),
            seca_thresholds: SecaThresholdConfig::default(),
            memory_mode: MemoryMode::Full,
            max_batches_in_memory: None,
        }
    }
}
