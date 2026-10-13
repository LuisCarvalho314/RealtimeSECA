use super::fixtures::*;
#[test]
fn process_batch_advances_last_processed_batch_index_via_snapshot() {
    let mut engine = build_small_baseline_engine();

    let batch = make_batch(1, &[("n1", &["a"])]);

    let _ = engine.process_batch(batch).unwrap();

    let snapshot = engine.snapshot().unwrap();
    assert_eq!(snapshot.last_processed_batch_index, Some(1));
}

#[test]
fn restoration_rejects_stale_node_cache() {
    let engine = build_small_baseline_engine();
    let mut snapshot = engine.snapshot().unwrap();
    let tree = snapshot
        .state
        .as_mut()
        .unwrap()
        .hkt_build_output
        .as_mut()
        .unwrap();
    tree.nodes_by_id
        .values_mut()
        .next()
        .unwrap()
        .source_ids
        .clear();
    assert!(crate::SecaEngine::load_snapshot(snapshot).is_err());
}

#[test]
fn restoration_rejects_inconsistent_source_provenance() {
    let engine = build_small_baseline_engine();
    let mut snapshot = engine.snapshot().unwrap();
    snapshot
        .state
        .as_mut()
        .unwrap()
        .source_ids_by_batch_index
        .clear();
    assert!(crate::SecaEngine::load_snapshot(snapshot).is_err());
}
