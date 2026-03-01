use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

use crate::config::SecaConfig;
use crate::tree::{Hkt, HktBuildOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRecord {
    pub source_id: String,
    pub batch_index: u32,
    pub tokens: Vec<String>,
    pub text: Option<String>,
    pub timestamp_unix_ms: Option<i64>,
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceBatch {
    pub batch_index: u32,
    pub sources: Vec<SourceRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BatchProcessingResult {
    pub batch_index: u32,
    pub sources_processed: usize,
    pub reconstruction_triggered: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClusteringResult {
    pub cluster_count: usize,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UpdateExplanation {
    pub summary: String,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSnapshot {
    pub schema_version: u32,
    pub engine_version: String,
    pub config: SecaConfig,
    #[serde(default)]
    pub has_baseline: bool,
    pub last_processed_batch_index: Option<u32>,
    #[serde(default)]
    pub hkt_build_output: Option<HktBuildOutput>,
    #[serde(default)]
    pub baseline_word_legend: BTreeMap<i32, String>,
    #[serde(default)]
    pub baseline_source_legend: BTreeMap<i64, String>,
    #[serde(default)]
    pub processed_batches: Vec<SourceBatch>,
    #[serde(default)]
    pub last_batch_word_stats_summary: Option<BatchWordStatsSummary>,
    #[serde(default)]
    pub source_id_by_url: BTreeMap<String, i64>,
    #[serde(default)]
    pub url_by_source_id: BTreeMap<i64, String>,
    #[serde(default)]
    pub source_batch_index_by_internal_source_id: BTreeMap<i64, u32>,
    #[serde(default)]
    pub source_ids_by_batch_index: BTreeMap<u32, BTreeSet<i64>>,
    #[serde(default)]
    pub archived_subtrees_by_root_id: BTreeMap<i32, HktBuildOutput>,
    #[serde(default)]
    pub logically_removed_hkts_by_id: BTreeMap<i32, LogicalRemovedHktSnapshot>,
    #[serde(default)]
    pub node_diagnostics_by_id: BTreeMap<i32, BaselineNodeDiagnosticsVerboseExport>,
    #[serde(default = "default_i32_one")]
    pub next_hkt_id: i32,
    #[serde(default = "default_i32_one")]
    pub next_node_id: i32,
}

fn default_i32_one() -> i32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalRemovedHktSnapshot {
    pub hkt: Hkt,
    pub old_parent_node_id: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineTreeExport {
    pub hkts: Vec<BaselineHktExport>,
    pub nodes: Vec<BaselineNodeExport>,
    #[serde(default)]
    pub logically_removed_hkts: Vec<BaselineHktExport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineHktExport {
    pub hkt_id: i32,
    pub parent_node_id: i32,
    pub expected_words: Vec<i32>,
    pub node_ids: Vec<i32>,
    pub is_state1: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineNodeExport {
    pub node_id: i32,
    pub hkt_id: i32,
    pub word_ids: Vec<i32>,
    pub source_ids: Vec<i64>,
    pub top_words: Vec<i32>,
    pub is_refuge_node: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineTreeVerboseExport {
    pub hkts: Vec<BaselineHktVerboseExport>,
    pub nodes: Vec<BaselineNodeVerboseExport>,
    pub word_legend: Vec<WordLegendEntry>,
    pub source_legend: Vec<SourceLegendEntry>,
    #[serde(default)]
    pub logically_removed_hkts: Vec<BaselineHktVerboseExport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineHktVerboseExport {
    pub hkt_id: i32,
    pub parent_node_id: i32,
    pub expected_words: Vec<VerboseWordRef>,
    pub all_node_words_union: Vec<VerboseWordRef>,
    pub node_ids: Vec<i32>,
    pub is_state1: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineNodeVerboseExport {
    pub node_id: i32,
    pub hkt_id: i32,
    pub words: Vec<VerboseWordRef>,
    pub sources: Vec<VerboseSourceRef>,
    pub top_words: Vec<VerboseWordRef>,
    pub is_refuge_node: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<BaselineNodeDiagnosticsVerboseExport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineNodeDiagnosticsVerboseExport {
    pub hkt_id: i32,
    pub scoped_source_count: usize,
    pub mapped_source_count: usize,
    pub should_reconstruct: bool,
    #[serde(default)]
    pub trigger_reasons: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha_error: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beta_error: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub word_importance_error: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paper_alpha_error: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paper_beta_error: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paper_word_importance_error: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerboseWordRef {
    pub word_id: i32,
    pub token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerboseSourceRef {
    pub internal_source_id: i64,
    pub external_source_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordLegendEntry {
    pub word_id: i32,
    pub token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLegendEntry {
    pub internal_source_id: i64,
    pub external_source_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BatchWordStatsSummary {
    pub unique_words_in_batch: usize,
    pub known_words_in_batch: usize,
    pub new_words_in_batch: usize,
    pub max_word_document_frequency: usize,
    pub total_sources_in_batch: usize,
}
