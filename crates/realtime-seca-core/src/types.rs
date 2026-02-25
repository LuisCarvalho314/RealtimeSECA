use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::SecaConfig;

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
    pub last_processed_batch_index: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineTreeExport {
    pub hkts: Vec<BaselineHktExport>,
    pub nodes: Vec<BaselineNodeExport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineHktExport {
    pub hkt_id: i32,
    pub parent_node_id: i32,
    pub expected_words: Vec<i32>,
    pub node_ids: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineNodeExport {
    pub node_id: i32,
    pub hkt_id: i32,
    pub word_ids: Vec<i32>,
    pub source_ids: Vec<i32>,
    pub top_words: Vec<i32>,
    pub is_refuge_node: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineTreeVerboseExport {
    pub hkts: Vec<BaselineHktVerboseExport>,
    pub nodes: Vec<BaselineNodeVerboseExport>,
    pub word_legend: Vec<WordLegendEntry>,
    pub source_legend: Vec<SourceLegendEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineHktVerboseExport {
    pub hkt_id: i32,
    pub parent_node_id: i32,
    pub expected_words: Vec<VerboseWordRef>,
    pub all_node_words_union: Vec<VerboseWordRef>,
    pub node_ids: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineNodeVerboseExport {
    pub node_id: i32,
    pub hkt_id: i32,
    pub words: Vec<VerboseWordRef>,
    pub sources: Vec<VerboseSourceRef>,
    pub top_words: Vec<VerboseWordRef>,
    pub is_refuge_node: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerboseWordRef {
    pub word_id: i32,
    pub token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerboseSourceRef {
    pub internal_source_id: i32,
    pub external_source_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordLegendEntry {
    pub word_id: i32,
    pub token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLegendEntry {
    pub internal_source_id: i32,
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
