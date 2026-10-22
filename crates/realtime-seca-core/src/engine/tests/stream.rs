use super::super::*;
use crate::config::{MemoryMode, SecaConfig};
use crate::engine::input::compute_batch_word_stats;

use super::fixtures::*;
use crate::tree::Hkt;
use crate::types::SourceBatch;
use std::collections::BTreeSet;
#[test]
fn update_report_preserves_actual_scope_diagnostics_and_rebuilt_root_lineage() {
    let config = SecaConfig::default();
    let mut engine = SecaEngine::new(config).unwrap();
    let baseline = engine
        .build_baseline_tree(make_batch(0, &[("s1", &["a", "b"]), ("s2", &["a", "c"])]))
        .unwrap();
    assert!(baseline.decision_diagnostics.is_empty()); // No update on initialization.
    let batch = make_batch(1, &[("n1", &["a", "x"]), ("n2", &["y"])]);
    let id = root_hkt_id(&engine);
    let scope = engine.snapshot_hkt_scope(id).unwrap();
    let mapped = engine
        .map_batch_into_hkt_scope(&batch, &(0..2).collect(), &scope)
        .unwrap();
    let stage = engine
        .compute_update_stage_for_scope(
            &batch,
            &scope,
            &mapped,
            &crate::engine::trigger::AncestorContext::default(),
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();
    let changes = engine
        .compute_word_change_metrics_for_scope(
            &scope,
            &mapped,
            &stage,
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();
    let decision = engine
        .evaluate_scope_trigger_decision(&scope, &mapped, &stage, &changes)
        .unwrap();
    let result = engine.process_batch(batch).unwrap();
    let d = result
        .decision_diagnostics
        .iter()
        .find(|d| d.hkt_id == id)
        .unwrap();
    assert_eq!(d.scoped_source_count, 2);
    assert_eq!(d.mapped_source_count, 1);
    assert_eq!(d.paper_alpha_error, decision.paper_alpha_error);
    assert_eq!(d.paper_beta_error, decision.paper_beta_error);
    assert_eq!(
        d.paper_word_importance_error,
        decision.paper_word_importance_error
    );
    assert_eq!(d.should_reconstruct, decision.should_reconstruct);
    assert_eq!(d.trigger_reasons, decision.trigger_reasons);
    let tree = engine.export_baseline_tree_verbose().unwrap();
    assert!(tree.hkts.iter().any(|h| Some(h.hkt_id) == d.output_hkt_id));
    if d.should_reconstruct {
        assert_ne!(d.output_hkt_id, Some(id));
    }
    let restored: crate::types::BatchProcessingResult =
        serde_json::from_str(&serde_json::to_string(&result).unwrap()).unwrap();
    assert_eq!(restored.decision_diagnostics, result.decision_diagnostics);
    let mut legacy = serde_json::to_value(&result).unwrap();
    legacy
        .as_object_mut()
        .unwrap()
        .remove("decision_diagnostics");
    assert!(
        serde_json::from_value::<crate::types::BatchProcessingResult>(legacy)
            .unwrap()
            .decision_diagnostics
            .is_empty()
    );
}
#[test]
fn display_sweep_is_observational_and_covers_final_ids_after_rebuild_and_pruning() {
    let mut config = SecaConfig::default();
    config.memory_mode = MemoryMode::SlidingWindow;
    config.max_batches_in_memory = Some(2);
    let mut engine = SecaEngine::new(config).unwrap();
    let baseline = engine
        .build_baseline_tree(make_batch(0, &[("s1", &["a", "b"]), ("s2", &["a", "c"])]))
        .unwrap();
    assert!(baseline
        .display_diagnostics
        .iter()
        .all(|d| d.paper_alpha_error.is_none()
            && d.unavailable_reason.as_deref() == Some("baseline_no_comparison")));
    for index in 1..=3 {
        let batch = make_batch(index, &[("new", &["a", "x"]), ("other", &["y"])]);
        // Unique IDs each update.
        let mut batch = batch;
        for source in &mut batch.sources {
            source.source_id.push_str(&index.to_string());
        }
        let report = engine.process_batch(batch.clone()).unwrap();
        let before = serde_json::to_value(&engine).unwrap();
        let display = engine.display_diagnostics(Some(&batch)).unwrap();
        assert_eq!(before, serde_json::to_value(&engine).unwrap());
        assert_eq!(display, report.display_diagnostics);
        let ids: BTreeSet<_> = engine
            .export_baseline_tree_verbose()
            .unwrap()
            .hkts
            .iter()
            .map(|h| h.hkt_id)
            .collect();
        assert_eq!(ids, display.iter().map(|d| d.hkt_id).collect());
        assert_eq!(report.hkts_inspected, report.decision_diagnostics.len());
        // Running the sweep cannot change the next actual decision/rebuild.
        let mut untouched = engine.clone();
        engine.display_diagnostics(Some(&batch)).unwrap();
        let next = make_batch(index + 1, &[("probe", &["a", "z"])]);
        let a = engine.clone().process_batch(next.clone()).unwrap();
        let b = untouched.process_batch(next).unwrap();
        assert_eq!(a.decision_diagnostics, b.decision_diagnostics);
        assert_eq!(a.reconstructed_hkt_ids, b.reconstructed_hkt_ids);
    }
}
#[test]
fn rebuilt_display_scopes_have_final_ids_and_decisions_keep_old_provenance() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha_error_threshold = 0.0;
    config.seca_thresholds.beta_error_threshold = 0.0;
    config.seca_thresholds.word_importance_error_threshold = 0.0;
    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(0, &[("a", &["a", "b"]), ("b", &["a", "c"])]))
        .unwrap();
    let old_root = root_hkt_id(&engine);
    let report = engine
        .process_batch(make_batch(1, &[("x", &["x"]), ("y", &["x", "y"])]))
        .unwrap();
    assert!(report.reconstruction_triggered);
    let decision = report
        .decision_diagnostics
        .iter()
        .find(|d| d.hkt_id == old_root)
        .unwrap();
    assert!(decision.should_reconstruct);
    assert_ne!(decision.output_hkt_id, Some(old_root));
    assert!(report
        .display_diagnostics
        .iter()
        .any(|d| Some(d.hkt_id) == decision.output_hkt_id));
    assert!(!report
        .display_diagnostics
        .iter()
        .any(|d| d.hkt_id == old_root));
    let ids: BTreeSet<_> = engine
        .export_baseline_tree_verbose()
        .unwrap()
        .hkts
        .iter()
        .map(|h| h.hkt_id)
        .collect();
    assert_eq!(
        ids,
        report
            .display_diagnostics
            .iter()
            .map(|d| d.hkt_id)
            .collect()
    );
}
#[test]
fn display_empty_scope_is_unavailable_and_equivalent_option1_agrees() {
    let config = SecaConfig::default();
    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(0, &[("s", &["a"])]))
        .unwrap();
    let batch = make_batch(1, &[("n", &["a"])]);
    let result = engine.process_batch(batch.clone()).unwrap();
    let decision = &result.decision_diagnostics[0];
    let display = &result.display_diagnostics[0];
    assert_eq!(display.paper_alpha_error, decision.paper_alpha_error);
    assert_eq!(display.paper_beta_error, decision.paper_beta_error);
    assert_eq!(
        display.paper_word_importance_error,
        decision.paper_word_importance_error
    );
    let tree = engine.hkt_build_output.as_mut().unwrap();
    for node in tree.nodes_by_id.values_mut() {
        node.source_ids.clear();
        node.word_source_ids.clear();
    }
    for hkt in tree.hkts_by_id.values_mut() {
        for node in &mut hkt.nodes {
            node.source_ids.clear();
            node.word_source_ids.clear();
        }
    }
    let empty = make_batch(2, &[]);
    let display = engine.display_diagnostics(Some(&empty)).unwrap();
    assert!(display.iter().all(|d| d.paper_alpha_error.is_none()
        && d.paper_beta_error.is_none()
        && d.paper_word_importance_error.is_none()
        && d.unavailable_reason.is_some()));
}
#[test]
fn three_decisions_and_260_display_scopes_preserve_the_trigger_plan() {
    use crate::tree::Node;
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha_error_threshold = 1.0;
    config.seca_thresholds.beta_error_threshold = 1.0;
    config.seca_thresholds.word_importance_error_threshold = 1.0;
    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(0, &[("old-a", &["a"]), ("old-b", &["b"])]))
        .unwrap();
    let a = *engine
        .baseline_word_legend
        .iter()
        .find(|(_, w)| w.as_str() == "a")
        .unwrap()
        .0;
    let b = *engine
        .baseline_word_legend
        .iter()
        .find(|(_, w)| w.as_str() == "b")
        .unwrap()
        .0;
    let sa = engine.source_id_by_url["old-a"];
    let sb = engine.source_id_by_url["old-b"];
    let tree = engine.hkt_build_output.as_mut().unwrap();
    tree.hkts_by_id.clear();
    tree.nodes_by_id.clear();
    let mut root = Hkt::new(1, 0, BTreeSet::new(), false);
    let mut node_id = 1;
    for hkt_id in 2..=260 {
        let word = if hkt_id <= 3 { a } else { b };
        let source = if hkt_id <= 3 { sa } else { sb };
        let parent_node_id = if hkt_id == 3 {
            2
        } else {
            let mut node = Node::new(node_id, 1);
            node.word_ids.insert(word);
            node.source_ids.insert(source);
            node.word_source_ids.insert(word, [source].into());
            tree.nodes_by_id.insert(node_id, node.clone());
            root.nodes.push(node);
            node_id
        };
        node_id += 1;
        let mut child = Hkt::new(hkt_id, parent_node_id, BTreeSet::new(), false);
        let mut node = Node::new(node_id, hkt_id);
        node.word_ids.insert(word);
        node.source_ids.insert(source);
        node.word_source_ids.insert(word, [source].into());
        tree.nodes_by_id.insert(node_id, node.clone());
        child.nodes.push(node);
        tree.hkts_by_id.insert(hkt_id, child);
        node_id += 1;
    }
    tree.hkts_by_id.insert(1, root);
    engine.next_hkt_id = 261;
    engine.next_node_id = node_id;
    let batch = make_batch(1, &[("incoming-a", &["a"])]);
    let plan = engine
        .clone()
        .evaluate_seca_trigger_plan_for_batch(&batch)
        .unwrap();
    assert_eq!(plan.diagnostics.len(), 3);
    let before = serde_json::to_value(&engine).unwrap();
    let pre_display = engine.display_diagnostics(Some(&batch)).unwrap();
    assert_eq!(before, serde_json::to_value(&engine).unwrap());
    let report = engine.process_batch(batch).unwrap();
    assert_eq!(report.decision_diagnostics, plan.diagnostics);
    assert_eq!(
        report.reconstructed_hkt_ids,
        plan.targets.keys().copied().collect::<Vec<_>>()
    );
    assert!(!report.reconstruction_triggered);
    assert_eq!(report.display_diagnostics.len(), 260);
    assert_eq!(pre_display.len(), 260);
    assert_eq!(
        report
            .display_diagnostics
            .iter()
            .filter(|d| d.paper_alpha_error.is_some())
            .count(),
        260
    );
    println!("CONTROLLED COVERAGE: HKTs=260 decision=3 display=260 alpha=260 beta=260 WI=260");
}
#[test]
fn realistic_display_coverage_is_independent_of_sparse_trigger_traversal() {
    let batch: SourceBatch =
        serde_json::from_str(include_str!("../../../tests/data/large_batch.json")).unwrap();
    let mut config = SecaConfig::default();
    // Use production branch admission rather than the unit-test default (1).
    config
        .hkt_builder
        .minimum_number_of_sources_to_create_branch_for_node = 10;
    config.hkt_builder.minimum_threshold_against_max_word_count = 0.7;
    config.seca_thresholds.alpha = 0.7;
    let mut engine = SecaEngine::new(config).unwrap();
    let baseline = engine.build_baseline_tree(batch.clone()).unwrap();
    let mut incoming = batch.clone();
    incoming.batch_index = batch.batch_index + 1;
    incoming.sources.truncate(3);
    for source in &mut incoming.sources {
        source.batch_index = incoming.batch_index;
        source.source_id.push_str("-incoming");
    }
    let report = engine.process_batch(incoming).unwrap();
    let ids: BTreeSet<_> = engine
        .export_baseline_tree_verbose()
        .unwrap()
        .hkts
        .iter()
        .map(|h| h.hkt_id)
        .collect();
    assert_eq!(
        ids,
        report
            .display_diagnostics
            .iter()
            .map(|d| d.hkt_id)
            .collect()
    );
    let available = report
        .display_diagnostics
        .iter()
        .filter(|d| d.paper_alpha_error.is_some())
        .count();
    println!(
        "COVERAGE baseline={} final={} decision={} display={} alpha={} beta={} WI={}",
        baseline.display_diagnostics.len(),
        ids.len(),
        report.decision_diagnostics.len(),
        report.display_diagnostics.len(),
        available,
        report
            .display_diagnostics
            .iter()
            .filter(|d| d.paper_beta_error.is_some())
            .count(),
        report
            .display_diagnostics
            .iter()
            .filter(|d| d.paper_word_importance_error.is_some())
            .count()
    );
    assert!(available > report.decision_diagnostics.len());
}
#[test]
fn new_rejects_alpha_out_of_range() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha = 1.01;

    let error = SecaEngine::new(config).unwrap_err();
    match error {
        SecaError::InvalidConfiguration { message } => {
            assert!(message.contains("seca_thresholds.alpha"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}
#[test]
fn new_rejects_beta_out_of_range() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.beta = -0.01;

    let error = SecaEngine::new(config).unwrap_err();
    match error {
        SecaError::InvalidConfiguration { message } => {
            assert!(message.contains("seca_thresholds.beta"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}
#[test]
fn new_rejects_hkt_threshold_out_of_range() {
    let mut config = SecaConfig::default();
    config.hkt_builder.minimum_threshold_against_max_word_count = 1.5;

    let error = SecaEngine::new(config).unwrap_err();
    match error {
        SecaError::InvalidConfiguration { message } => {
            assert!(message.contains("minimum_threshold_against_max_word_count"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}
#[test]
fn new_rejects_hkt_similarity_out_of_range() {
    let mut config = SecaConfig::default();
    config.hkt_builder.similarity_threshold = -0.1;

    let error = SecaEngine::new(config).unwrap_err();
    match error {
        SecaError::InvalidConfiguration { message } => {
            assert!(message.contains("similarity_threshold"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}
#[test]
fn compute_batch_word_stats_uses_document_frequency_not_raw_frequency() {
    let baseline_engine = build_small_baseline_engine();

    let batch = make_batch(
        1,
        &[
            ("n1", &["a", "a", "a", "x"]), // duplicate "a" should count once for DF
            ("n2", &["a", "b", "x"]),
            ("n3", &["b"]),
        ],
    );

    let (stats, frequencies) =
        compute_batch_word_stats(&batch, &baseline_engine.baseline_word_legend);

    // unique words = {a,b,x}
    assert_eq!(stats.unique_words_in_batch, 3);

    // known = a,b (x is new)
    assert_eq!(stats.known_words_in_batch, 2);
    assert_eq!(stats.new_words_in_batch, 1);

    // document frequency:
    // a -> 2 sources (n1,n2), b -> 2 sources (n2,n3), x -> 2 sources (n1,n2)
    assert_eq!(stats.max_word_document_frequency, 2);
    assert_eq!(stats.total_sources_in_batch, 3);

    assert_eq!(frequencies.get("a"), Some(&2));
    assert_eq!(frequencies.get("b"), Some(&2));
    assert_eq!(frequencies.get("x"), Some(&2));
}
#[test]
fn process_batch_stores_last_batch_word_stats_summary() {
    let mut engine = build_small_baseline_engine();

    let batch = make_batch(
        1,
        &[
            ("n1", &["a", "new1"]),
            ("n2", &["b", "new2"]),
            ("n3", &["a", "b"]),
        ],
    );

    let result = engine.process_batch(batch).unwrap();
    assert_eq!(result.batch_index, 1);

    let summary = engine
        .last_batch_word_stats_summary()
        .expect("expected batch stats summary");

    assert_eq!(summary.total_sources_in_batch, 3);
    assert_eq!(summary.unique_words_in_batch, 4); // a,b,new1,new2
    assert_eq!(summary.known_words_in_batch, 2); // a,b
    assert_eq!(summary.new_words_in_batch, 2); // new1,new2
    assert_eq!(summary.max_word_document_frequency, 2); // a or b appears in 2 sources
}
#[test]
fn evaluate_trigger_plan_runs_after_baseline_and_is_deterministic() {
    let mut engine = build_small_baseline_engine();

    let batch = make_batch(
        1,
        &[
            ("n1", &["a", "x"]),
            ("n2", &["b", "y"]),
            ("n3", &["a", "b"]),
        ],
    );

    let mut engine_clone = engine.clone();
    let plan_one = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();
    let plan_two = engine_clone
        .evaluate_seca_trigger_plan_for_batch(&batch)
        .unwrap();

    assert_eq!(!plan_one.targets.is_empty(), !plan_two.targets.is_empty());
    assert_eq!(
        plan_one.targets.keys().copied().collect::<Vec<_>>(),
        plan_two.targets.keys().copied().collect::<Vec<_>>()
    );
}
#[test]
fn recursive_trigger_plan_is_deterministic_for_same_input() {
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

    let batch = make_batch(
        1,
        &[
            ("n1", &["earthquake", "aftershock", "city"]),
            ("n2", &["rescue", "teams", "deployed"]),
        ],
    );

    let mut engine_clone = engine.clone();
    let plan_one = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();
    let plan_two = engine_clone
        .evaluate_seca_trigger_plan_for_batch(&batch)
        .unwrap();

    assert_eq!(!plan_one.targets.is_empty(), !plan_two.targets.is_empty());
    assert_eq!(
        plan_one.targets.keys().copied().collect::<Vec<_>>(),
        plan_two.targets.keys().copied().collect::<Vec<_>>()
    );
}
#[test]
fn strict_thresholds_trigger_recursive_plan_and_emit_reasons() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.word_importance_error_threshold = 0.0;

    let mut engine = SecaEngine::new(config).unwrap();

    let baseline = make_batch(0, &[("s1", &["a", "b"]), ("s2", &["a", "c"])]);

    engine.build_baseline_tree(baseline).unwrap();

    let batch = make_batch(1, &[("n1", &["new_token"])]);

    let plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();

    assert!(!plan.targets.is_empty());
    assert!(
        !plan.targets.keys().copied().collect::<Vec<_>>().is_empty(),
        "expected at least one HKT to be marked for reconstruction"
    );
}
#[test]
fn process_batch_requires_baseline_first() {
    let mut engine = SecaEngine::new(SecaConfig::default()).unwrap();

    let batch = make_batch(1, &[("n1", &["a"])]);

    let error = engine.process_batch(batch).unwrap_err();
    match error {
        SecaError::StateError { message } => {
            assert!(message.contains("baseline tree has not been built yet"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}
#[test]
fn process_batch_rejects_out_of_sequence_batch_index() {
    let mut engine = build_small_baseline_engine();

    // baseline is batch 0, next must be 1
    let batch = make_batch(2, &[("n1", &["a"])]);

    let error = engine.process_batch(batch).unwrap_err();
    match error {
        SecaError::InvalidConfiguration { message } => {
            assert!(message.contains("out of sequence"));
            assert!(message.contains("expected 1"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}
#[test]
fn strict_word_importance_threshold_triggers_reconstruction() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.word_importance_error_threshold = 0.0; // any new-token ratio > 0 triggers

    let mut engine = SecaEngine::new(config).unwrap();
    let baseline_batch = make_batch(0, &[("s1", &["a", "b"]), ("s2", &["a", "c"])]);
    engine.build_baseline_tree(baseline_batch).unwrap();

    let batch = make_batch(1, &[("n1", &["brand_new_token"]), ("n2", &["a"])]);

    let result = engine.process_batch(batch).unwrap();
    assert!(result.reconstruction_triggered);

    let explanation = engine.explain_last_update().unwrap();
    assert!(explanation
        .reason_codes
        .contains(&"SECA_RECONSTRUCTION_TRIGGERED".to_string()));
    assert!(explanation
        .reason_codes
        .contains(&"SECA_RECONSTRUCTION_TRIGGERED".to_string()));
}
#[test]
fn process_batch_trigger_rebuilds_tree_from_stored_batches() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.word_importance_error_threshold = 0.0;

    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(
            0,
            &[
                ("s1", &["earthquake", "damage", "city"]),
                ("s2", &["earthquake", "rescue", "city"]),
                ("s3", &["storm", "damage", "coast"]),
            ],
        ))
        .unwrap();

    let before =
        serde_json::to_string_pretty(&engine.export_baseline_tree_verbose().unwrap()).unwrap();

    let result = engine
        .process_batch(make_batch(
            1,
            &[
                ("n1", &["volcano", "ash", "plume"]),
                ("n2", &["evacuation", "zone", "ash"]),
            ],
        ))
        .unwrap();
    assert!(result.reconstruction_triggered);

    let after_tree = engine.export_baseline_tree_verbose().unwrap();
    let after = serde_json::to_string_pretty(&after_tree).unwrap();

    assert_ne!(before, after, "tree should change after triggered rebuild");

    let explanation = engine.explain_last_update().unwrap();
    assert!(explanation
        .reason_codes
        .iter()
        .any(|c| c == "SECA_RECONSTRUCTION_TRIGGERED"));

    let all_tokens: BTreeSet<String> = after_tree
        .word_legend
        .iter()
        .filter_map(|entry| entry.token.clone())
        .collect();

    assert!(all_tokens.contains("volcano") || all_tokens.contains("ash"));
}
#[test]
fn process_batch_rebuild_path_is_deterministic_for_same_sequence() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.word_importance_error_threshold = 0.0;

    let mut engine_a = SecaEngine::new(config.clone()).unwrap();
    engine_a
        .build_baseline_tree(make_batch(
            0,
            &[
                ("s1", &["earthquake", "damage", "city"]),
                ("s2", &["earthquake", "rescue", "city"]),
                ("s3", &["storm", "damage", "coast"]),
            ],
        ))
        .unwrap();
    engine_a
        .process_batch(make_batch(
            1,
            &[
                ("n1", &["earthquake", "aftershock", "city"]),
                ("n2", &["rescue", "teams", "deployed"]),
            ],
        ))
        .unwrap();
    let final_a =
        serde_json::to_string_pretty(&engine_a.export_baseline_tree_verbose().unwrap()).unwrap();

    let mut engine_b = SecaEngine::new(config).unwrap();
    engine_b
        .build_baseline_tree(make_batch(
            0,
            &[
                ("s1", &["earthquake", "damage", "city"]),
                ("s2", &["earthquake", "rescue", "city"]),
                ("s3", &["storm", "damage", "coast"]),
            ],
        ))
        .unwrap();
    engine_b
        .process_batch(make_batch(
            1,
            &[
                ("n1", &["earthquake", "aftershock", "city"]),
                ("n2", &["rescue", "teams", "deployed"]),
            ],
        ))
        .unwrap();
    let final_b =
        serde_json::to_string_pretty(&engine_b.export_baseline_tree_verbose().unwrap()).unwrap();

    assert_eq!(
        final_a, final_b,
        "final rebuilt tree should be deterministic"
    );
}
#[test]
fn export_baseline_tree_errors_before_baseline() {
    let engine = SecaEngine::new(SecaConfig::default()).unwrap();

    let error = engine.export_baseline_tree().unwrap_err();
    match error {
        SecaError::StateError { message } => {
            assert!(message.contains("baseline tree has not been built yet"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}
#[test]
fn export_baseline_tree_verbose_errors_before_baseline() {
    let engine = SecaEngine::new(SecaConfig::default()).unwrap();

    let error = engine.export_baseline_tree_verbose().unwrap_err();
    match error {
        SecaError::StateError { message } => {
            assert!(message.contains("baseline tree has not been built yet"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}
#[test]
fn export_baseline_tree_verbose_contains_legends_after_baseline() {
    let engine = build_small_baseline_engine();

    let exported = engine.export_baseline_tree_verbose().unwrap();
    assert!(!exported.word_legend.is_empty());
    assert!(!exported.source_legend.is_empty());

    let tokens: BTreeSet<String> = exported
        .word_legend
        .iter()
        .filter_map(|entry| entry.token.clone())
        .collect();

    assert!(tokens.contains("a"));
    assert!(tokens.contains("b"));
}
#[test]
fn export_baseline_tree_is_sorted_by_ids() {
    let engine = build_small_baseline_engine();
    let export = engine.export_baseline_tree().unwrap();

    let mut hkt_ids: Vec<i32> = export.hkts.iter().map(|h| h.hkt_id).collect();
    let mut node_ids: Vec<i32> = export.nodes.iter().map(|n| n.node_id).collect();

    let mut sorted_hkt_ids = hkt_ids.clone();
    let mut sorted_node_ids = node_ids.clone();

    sorted_hkt_ids.sort_unstable();
    sorted_node_ids.sort_unstable();

    assert_eq!(hkt_ids, sorted_hkt_ids);
    assert_eq!(node_ids, sorted_node_ids);

    // avoid "unused mut" if optimizer/lint gets clever
    hkt_ids.clear();
    node_ids.clear();
}
