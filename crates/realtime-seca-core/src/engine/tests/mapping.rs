use super::super::*;
use crate::config::SecaConfig;

use super::fixtures::*;
use std::collections::BTreeSet;
#[test]
fn snapshot_hkt_scope_root_contains_expected_structure() {
    let mut engine = SecaEngine::new(SecaConfig::default()).unwrap();

    let baseline = make_batch(
        0,
        &[
            ("s1", &["earthquake", "damage", "city"]),
            ("s2", &["earthquake", "rescue", "city"]),
            ("s3", &["storm", "damage", "coast"]),
        ],
    );

    engine.build_baseline_tree(baseline).unwrap();

    let snapshot = engine.snapshot_hkt_scope(root_hkt_id(&engine)).unwrap();

    assert_eq!(snapshot.parent_node_id, 0);

    // Root should have at least one non-refuge node
    assert!(
        !snapshot.non_refuge_node_ids.is_empty(),
        "expected at least one non-refuge node in root scope"
    );

    // Node maps should contain all referenced node IDs
    for node_id in &snapshot.non_refuge_node_ids {
        assert!(snapshot.node_word_ids_by_node_id.contains_key(node_id));
        assert!(snapshot.node_source_ids_by_node_id.contains_key(node_id));
    }

    if let Some(refuge_node_id) = snapshot.refuge_node_id {
        assert!(snapshot
            .node_word_ids_by_node_id
            .contains_key(&refuge_node_id));
        assert!(snapshot
            .node_source_ids_by_node_id
            .contains_key(&refuge_node_id));
    }
}
#[test]
fn snapshot_hkt_scope_returns_sorted_non_refuge_node_ids() {
    let engine = build_small_baseline_engine();
    let snapshot = engine.snapshot_hkt_scope(root_hkt_id(&engine)).unwrap();

    let mut sorted = snapshot.non_refuge_node_ids.clone();
    sorted.sort_unstable();
    assert_eq!(snapshot.non_refuge_node_ids, sorted);
}

#[test]
fn map_batch_into_hkt_scope_uses_document_frequency_per_source_not_raw_repetition() {
    let mut engine = SecaEngine::new(SecaConfig::default()).unwrap();

    let baseline = make_batch(0, &[("s1", &["a", "b"]), ("s2", &["a", "c"])]);

    engine.build_baseline_tree(baseline).unwrap();
    let snapshot = engine.snapshot_hkt_scope(root_hkt_id(&engine)).unwrap();

    let batch = make_batch(1, &[("n1", &["a", "a", "a", "x"]), ("n2", &["a"])]);

    let scoped_indexes = BTreeSet::from([0_usize, 1_usize]);
    let mapped = engine
        .map_batch_into_hkt_scope(&batch, &scoped_indexes, &snapshot)
        .unwrap();

    // "a" appears in both sources => DF = 2, not 4
    assert_eq!(
        mapped.word_document_frequency_in_scope.get("a"),
        Some(&2usize)
    );
    assert_eq!(
        mapped.word_document_frequency_in_scope.get("x"),
        Some(&1usize)
    );
}
