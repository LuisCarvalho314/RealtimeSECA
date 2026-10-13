use super::super::*;
use crate::config::SecaConfig;

use crate::types::SourceBatch;

use super::fixtures::*;
use std::collections::BTreeSet;
#[test]
fn baseline_build_runs_hkt_builder() {
    let mut engine = SecaEngine::new(SecaConfig::default()).unwrap();

    let batch = make_batch(0, &[("s1", &["a", "b", "c"]), ("s2", &["a", "b"])]);

    let result = engine.build_baseline_tree(batch).unwrap();

    assert_eq!(result.batch_index, 0);
    assert!(!result.reconstruction_triggered);
    assert!(engine.explain_last_update().is_some());
}
#[test]
fn baseline_nodes_populate_word_source_ids() {
    let mut engine = SecaEngine::new(SecaConfig::default()).unwrap();
    let batch = make_batch(0, &[("s1", &["a", "b"]), ("s2", &["a"]), ("s3", &["a"])]);
    engine.build_baseline_tree(batch).unwrap();

    let word_id_a = engine
        .baseline_word_legend
        .iter()
        .find(|(_, token)| token.as_str() == "a")
        .map(|(id, _)| *id)
        .unwrap();

    let expected_sources: BTreeSet<i64> = engine
        .baseline_source_legend
        .iter()
        .filter(|(_, url)| url.as_str() == "s1" || url.as_str() == "s2" || url.as_str() == "s3")
        .map(|(id, _)| *id)
        .collect();

    let node = engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .nodes_by_id
        .values()
        .find(|node| node.word_ids.contains(&word_id_a))
        .unwrap();

    let sources_for_a = node.word_source_ids.get(&word_id_a).unwrap();
    assert_eq!(
        sources_for_a, &expected_sources,
        "expected word_source_ids to track baseline sources for the word"
    );
}
#[test]
fn baseline_prominent_word_selection_respects_insertion_order_on_ties() {
    let mut engine = SecaEngine::new(SecaConfig::default()).unwrap();

    // "b" appears before "a" in insertion order; both have the same document frequency.
    let batch = make_batch(0, &[("s1", &["b"]), ("s2", &["a"])]);
    engine.build_baseline_tree(batch).unwrap();

    let root_hkt_id = root_hkt_id(&engine);
    let root_hkt = engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .hkts_by_id
        .get(&root_hkt_id)
        .unwrap();

    let word_id_b = engine
        .baseline_word_legend
        .iter()
        .find(|(_, token)| token.as_str() == "b")
        .map(|(id, _)| *id)
        .unwrap();

    let first_node = root_hkt.nodes.first().expect("root HKT should have a node");
    assert!(
        first_node.word_ids.contains(&word_id_b),
        "expected first node to use the earliest-inserted tied word as prominent"
    );
}
#[test]
fn baseline_refuge_branch_prominent_word_respects_insertion_order_on_ties() {
    let mut config = SecaConfig::default();
    config.hkt_builder.minimum_threshold_against_max_word_count = 0.6;
    let mut engine = SecaEngine::new(config).unwrap();

    // "a" is dominant; "b" and "c" appear once and become refuge sources.
    let batch = make_batch(
        0,
        &[
            ("s1", &["a"]),
            ("s2", &["a"]),
            ("s3", &["a"]),
            ("s4", &["a"]),
            ("s5", &["b"]),
            ("s6", &["c"]),
        ],
    );
    engine.build_baseline_tree(batch).unwrap();

    let root_hkt_id = root_hkt_id(&engine);
    let root_hkt = engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .hkts_by_id
        .get(&root_hkt_id)
        .unwrap();

    let refuge_node = root_hkt
        .nodes
        .iter()
        .find(|node| node.word_ids.contains(&-1))
        .expect("expected a refuge node in the root HKT");

    let refuge_child_hkt = engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .hkts_by_id
        .values()
        .find(|hkt| hkt.parent_node_id == refuge_node.node_id)
        .expect("expected a child HKT for the refuge node");

    let word_id_b = engine
        .baseline_word_legend
        .iter()
        .find(|(_, token)| token.as_str() == "b")
        .map(|(id, _)| *id)
        .unwrap();

    let first_node = refuge_child_hkt
        .nodes
        .first()
        .expect("refuge child HKT should have a node");
    assert!(
        first_node.word_ids.contains(&word_id_b),
        "expected refuge child HKT to select earliest-inserted tied word as prominent"
    );
}
#[test]
fn baseline_build_rejects_empty_batch() {
    let mut engine = SecaEngine::new(SecaConfig::default()).unwrap();
    let empty_batch = SourceBatch {
        batch_index: 0,
        sources: Vec::new(),
    };

    let error = engine.build_baseline_tree(empty_batch).unwrap_err();
    match error {
        SecaError::StateError { message } => {
            assert!(message.contains("baseline batch cannot be empty"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}
#[test]
fn baseline_hkts_are_state0() {
    let engine = build_small_baseline_engine();

    let hkts = engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .hkts_by_id
        .values();
    for hkt in hkts {
        assert!(
            !hkt.is_state1,
            "baseline HKT {} should have is_state1 = false",
            hkt.hkt_id
        );
    }
}
