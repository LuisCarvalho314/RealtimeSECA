use super::super::*;
use crate::config::{MemoryMode, SecaConfig};

use super::fixtures::*;
use std::collections::BTreeSet;
#[test]
fn process_batch_in_full_mode_keeps_all_batches() {
    let mut config = SecaConfig::default();
    config.memory_mode = MemoryMode::Full;
    config.seca_thresholds.word_importance_error_threshold = 1.0; // avoid rebuild noise in this test

    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["b"])]))
        .unwrap();

    assert_eq!(engine.stored_batch_count(), 1); // baseline stored

    engine
        .process_batch(make_batch(1, &[("n1", &["x"])]))
        .unwrap();
    engine
        .process_batch(make_batch(2, &[("n2", &["y"])]))
        .unwrap();
    engine
        .process_batch(make_batch(3, &[("n3", &["z"])]))
        .unwrap();

    assert_eq!(engine.stored_batch_count(), 4); // baseline + 3 incrementals
}
#[test]
fn process_batch_in_sliding_window_trims_old_batches() {
    let mut config = SecaConfig::default();
    config.memory_mode = MemoryMode::SlidingWindow;
    config.max_batches_in_memory = Some(2);
    config.seca_thresholds.word_importance_error_threshold = 1.0; // avoid rebuild noise

    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["b"])]))
        .unwrap();

    assert_eq!(engine.stored_batch_count(), 1);

    engine
        .process_batch(make_batch(1, &[("n1", &["x"])]))
        .unwrap();
    assert_eq!(engine.stored_batch_count(), 2);

    engine
        .process_batch(make_batch(2, &[("n2", &["y"])]))
        .unwrap();
    assert_eq!(engine.stored_batch_count(), 2); // trimmed to max

    engine
        .process_batch(make_batch(3, &[("n3", &["z"])]))
        .unwrap();
    assert_eq!(engine.stored_batch_count(), 2); // still trimmed
}
#[test]
fn sliding_window_with_zero_max_batches_is_rejected_at_construction() {
    let mut config = SecaConfig::default();
    config.memory_mode = MemoryMode::SlidingWindow;
    config.max_batches_in_memory = Some(0);
    assert!(SecaEngine::new(config)
        .unwrap_err()
        .to_string()
        .contains("max_batches_in_memory > 0"));
}
#[test]
fn sliding_window_rebuild_uses_only_retained_batches() {
    let mut config = SecaConfig::default();
    config.memory_mode = MemoryMode::SlidingWindow;
    config.max_batches_in_memory = Some(2);
    config.seca_thresholds.word_importance_error_threshold = 0.0; // force rebuilds

    let mut engine = SecaEngine::new(config).unwrap();

    // Baseline includes "baseline_only"
    engine
        .build_baseline_tree(make_batch(0, &[("s1", &["baseline_only", "a"])]))
        .unwrap();

    // batch1 includes "batch1_only"
    engine
        .process_batch(make_batch(1, &[("n1", &["batch1_only"])]))
        .unwrap();

    assert_eq!(engine.stored_batch_count(), 2);

    // batch2 includes "batch2_only"; push then trim => should retain only 2 batches
    engine
        .process_batch(make_batch(2, &[("n2", &["batch2_only"])]))
        .unwrap();

    assert_eq!(engine.stored_batch_count(), 2);

    let exported = engine.export_baseline_tree_verbose().unwrap();
    let tokens: BTreeSet<String> = exported
        .word_legend
        .iter()
        .filter_map(|entry| entry.token.clone())
        .collect();

    // Current implementation trims WHOLE batches, then rebuilds from retained batches.
    assert!(tokens.contains("batch1_only") || tokens.contains("batch2_only"));
}
#[test]
fn seca_light_preserves_dead_nodes_and_hkts_until_reconstruction() {
    let mut config = SecaConfig::default();
    config.memory_mode = MemoryMode::SlidingWindow;
    config.max_batches_in_memory = Some(2);
    config.seca_thresholds.word_importance_error_threshold = 1.0;

    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(
            0,
            &[("base1", &["a", "b"]), ("base2", &["a", "c"])],
        ))
        .unwrap();
    engine
        .process_batch(make_batch(1, &[("fresh", &["a"])]))
        .unwrap();

    let child_hkt_id = engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .hkts_by_id
        .values()
        .find(|hkt| hkt.parent_node_id != 0)
        .map(|hkt| hkt.hkt_id)
        .expect("expected child HKT for pruning test");

    {
        let output = engine.hkt_build_output.as_mut().unwrap();
        let child = output.hkts_by_id.get_mut(&child_hkt_id).unwrap();
        for node in &mut child.nodes {
            node.source_ids.clear();
            node.word_source_ids.clear();
            if let Some(global_node) = output.nodes_by_id.get_mut(&node.node_id) {
                global_node.source_ids.clear();
                global_node.word_source_ids.clear();
            }
        }
    }

    let before_hkt_count = engine.hkt_build_output.as_ref().unwrap().hkts_by_id.len();
    let before_node_count = engine.hkt_build_output.as_ref().unwrap().nodes_by_id.len();

    let report = engine
        .apply_seca_light_pruning_if_enabled(1)
        .unwrap()
        .expect("expected SECA-Light pruning report");
    assert_eq!(report, 0);

    let after_output = engine.hkt_build_output.as_ref().unwrap();
    assert_eq!(after_output.hkts_by_id.len(), before_hkt_count);
    assert_eq!(after_output.nodes_by_id.len(), before_node_count);
    assert!(after_output.hkts_by_id.contains_key(&child_hkt_id));
}
#[test]
fn seca_light_sliding_window_requires_gamma() {
    let mut config = SecaConfig::default();
    config.memory_mode = MemoryMode::SlidingWindow;
    config.max_batches_in_memory = None;
    assert!(SecaEngine::new(config).is_err());
}
#[test]
fn full_mode_does_not_apply_seca_light_pruning() {
    let mut config = SecaConfig::default();
    config.memory_mode = MemoryMode::Full;
    config.seca_thresholds.word_importance_error_threshold = 1.0;

    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(0, &[("s0", &["a"])]))
        .unwrap();
    let s0_internal_id = SecaEngine::fnv1a_64("s0");

    engine
        .process_batch(make_batch(1, &[("s1", &["b"])]))
        .unwrap();
    engine
        .process_batch(make_batch(2, &[("s2", &["c"])]))
        .unwrap();

    assert!(engine.baseline_source_legend.contains_key(&s0_internal_id));
    let explanation = engine.explain_last_update().expect("missing explanation");
    assert!(!explanation
        .reason_codes
        .contains(&"SECA_LIGHT_PRUNING_APPLIED".to_string()));
}
