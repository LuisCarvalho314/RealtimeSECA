use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemoryMode {
    Full,
    SlidingWindow,
}

pub enum IncrementalStrategy {
    FullRebuildOnly,
    Hybrid,
    TargetedSubtree,
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
    pub alpha_option1_threshold: f64,
    pub alpha_option2_threshold: f64,
    pub alpha_option3_threshold: f64,
    pub beta_option1_threshold: f64,
    pub beta_option2_threshold: f64,
    pub beta_option3_threshold: f64,
    pub word_importance_option1_threshold: f64,
    pub word_importance_option2_threshold: f64,
    pub selected_alpha_option: AlphaErrorOption,
    pub selected_beta_option: BetaErrorOption,
    pub selected_word_importance_option: WordImportanceErrorOption,
}

impl Default for SecaThresholdConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.5,
            alpha_error_threshold: 0.1,
            beta_error_threshold: 0.1,
            word_importance_error_threshold: 0.1,
            alpha_option1_threshold: 0.1,
            alpha_option2_threshold: 0.2,
            alpha_option3_threshold: 0.3,
            beta_option1_threshold: 0.13,
            beta_option2_threshold: 0.2,
            beta_option3_threshold: 0.2,
            word_importance_option1_threshold: 0.3,
            word_importance_option2_threshold: 0.2,
            selected_alpha_option: AlphaErrorOption::Option1,
            selected_beta_option: BetaErrorOption::Option1,
            selected_word_importance_option: WordImportanceErrorOption::Option1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SecaConfig {
    pub hkt_builder: HktBuilderConfig,
    pub seca_thresholds: SecaThresholdConfig,
    pub memory_mode: MemoryMode,
    pub max_batches_in_memory: Option<u32>,
    pub trigger_policy_mode: TriggerPolicyMode,
}

impl Default for SecaConfig {
    fn default() -> Self {
        Self {
            hkt_builder: HktBuilderConfig::default(),
            seca_thresholds: SecaThresholdConfig::default(),
            memory_mode: MemoryMode::Full,
            max_batches_in_memory: None,
            trigger_policy_mode: TriggerPolicyMode::PaperDiagnosticScaffold,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]

pub enum TriggerPolicyMode {
    Placeholder,
    PaperDiagnosticScaffold,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AlphaErrorOption {
    Option1,
    Option2,
    Option3,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BetaErrorOption {
    Option1,
    Option2,
    Option3,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WordImportanceErrorOption {
    Option1,
    Option2,
}
