use super::super::*;
use crate::config::SecaConfig;

use crate::types::{SourceBatch, SourceRecord};

pub(super) fn make_source(source_id: &str, batch_index: u32, tokens: &[&str]) -> SourceRecord {
    SourceRecord {
        source_id: source_id.to_string(),
        batch_index,
        tokens: tokens.iter().map(|token| (*token).to_string()).collect(),
        text: None,
        timestamp_unix_ms: None,
        metadata: None,
    }
}
pub(super) fn make_batch(batch_index: u32, rows: &[(&str, &[&str])]) -> SourceBatch {
    SourceBatch {
        batch_index,
        sources: rows
            .iter()
            .map(|(source_id, tokens)| make_source(source_id, batch_index, tokens))
            .collect(),
    }
}
pub(super) fn build_small_baseline_engine() -> SecaEngine {
    let mut engine = SecaEngine::new(SecaConfig::default()).unwrap();

    let baseline_batch = make_batch(
        0,
        &[
            ("s1", &["a", "b", "c"]),
            ("s2", &["a", "b"]),
            ("s3", &["a", "d"]),
        ],
    );

    let result = engine.build_baseline_tree(baseline_batch).unwrap();
    assert_eq!(result.batch_index, 0);
    engine
}
pub(super) fn root_hkt_id(engine: &SecaEngine) -> i32 {
    engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .hkts_by_id
        .values()
        .find(|hkt| hkt.parent_node_id == 0)
        .map(|hkt| hkt.hkt_id)
        .expect("root HKT should exist")
}
