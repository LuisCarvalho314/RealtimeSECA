//! Characterization fixtures for main 42c07c0; known defects are documented in
//! docs/parity-context-metrics.md. These assert observable behavior, not correctness.
use realtime_seca_core::{
    BatchProcessingResult, SecaConfig, SecaEngine, SourceBatch, SourceRecord,
};

fn config(alpha: f64, beta: f64) -> SecaConfig {
    let mut c = SecaConfig::default();
    c.hkt_builder.minimum_threshold_against_max_word_count = alpha;
    c.hkt_builder.similarity_threshold = beta;
    c.hkt_builder
        .minimum_number_of_sources_to_create_branch_for_node = 1000;
    c.seca_thresholds.alpha = alpha;
    c.seca_thresholds.beta = beta;
    c.seca_thresholds.alpha_option1_threshold = 1.0;
    c.seca_thresholds.beta_option1_threshold = 1.0;
    c.seca_thresholds.word_importance_option1_threshold = 1.0;
    c
}
fn batch(index: u32, rows: &[&[&str]]) -> SourceBatch {
    SourceBatch {
        batch_index: index,
        sources: rows
            .iter()
            .enumerate()
            .map(|(i, words)| SourceRecord {
                source_id: format!("{index}-{i}"),
                batch_index: index,
                tokens: words.iter().map(|s| (*s).to_string()).collect(),
                text: None,
                timestamp_unix_ms: None,
                metadata: None,
            })
            .collect(),
    }
}
fn engine(c: SecaConfig, rows: &[&[&str]]) -> SecaEngine {
    let mut e = SecaEngine::new(c).unwrap();
    e.build_baseline_tree(batch(1, rows)).unwrap();
    e
}
fn root(e: &SecaEngine) -> i32 {
    e.export_baseline_tree()
        .unwrap()
        .hkts
        .iter()
        .find(|h| h.parent_node_id == 0)
        .unwrap()
        .hkt_id
}
// Main exposes decision measurements through formatted notes only (four decimals).
fn metric(r: &BatchProcessingResult, hkt: i32, name: &str) -> f64 {
    let prefix = format!("HKT {hkt} metrics (paper diagnostic): {name}=");
    r.notes
        .iter()
        .find_map(|n| n.strip_prefix(&prefix))
        .unwrap()
        .parse()
        .unwrap()
}
fn errors(r: &BatchProcessingResult, hkt: i32, expected: [f64; 3]) {
    for (name, value) in [
        "alpha_error_eq6_scaffold",
        "beta_error_eq9_scaffold",
        "wi_error_eq12",
    ]
    .into_iter()
    .zip(expected)
    {
        assert!(
            (metric(r, hkt, name) - value).abs() < 1e-9,
            "{name}: {:?}",
            r.notes
        );
    }
}
fn expected_count(e: &SecaEngine, hkt: i32) -> usize {
    e.export_baseline_tree()
        .unwrap()
        .hkts
        .iter()
        .find(|h| h.hkt_id == hkt)
        .unwrap()
        .expected_words
        .len()
}

#[test]
fn unchanged_single_word_counts_each_document_once_and_merges_sources() {
    let mut e = engine(config(0.5, 0.5), &[&["a", "a"]]);
    let h = root(&e);
    let r = e.process_batch(batch(2, &[&["a", " a ", ""]])).unwrap();
    errors(&r, h, [0.0, 0.0, 0.0]);
    assert!(!r.reconstruction_triggered);
    assert_eq!(
        e.export_baseline_tree().unwrap().nodes[0].source_ids.len(),
        2
    );
}

#[test]
fn unseen_word_below_alpha_does_not_enter_denominator() {
    let mut e = engine(config(0.7, 0.5), &[&["a"], &["a"], &["a"], &["a"]]);
    let h = root(&e);
    let r = e.process_batch(batch(2, &[&["a", "x"]])).unwrap();
    errors(&r, h, [0.0, 0.0, 0.0]);
    assert_eq!(expected_count(&e, h), 0);
    assert!(!r.reconstruction_triggered);
}

#[test]
fn alpha_pass_beta_fail_keeps_candidate_in_importance_denominator() {
    let mut e = engine(config(0.5, 0.5), &[&["a"], &["a"]]);
    let h = root(&e);
    let r = e.process_batch(batch(2, &[&["a", "x"], &["x"]])).unwrap();
    // D(a)=3, D(x)=2, incoming candidate/node-prominent overlap=1/3.
    errors(&r, h, [0.0, 0.0, 0.4]);
    assert_eq!(expected_count(&e, h), 1);
    assert!(!r.reconstruction_triggered);
}

#[test]
fn exact_alpha_and_beta_admission_pollutes_state0_error_vocabulary() {
    let mut e = engine(config(0.5, 0.5), &[&["a"], &["a"]]);
    let h = root(&e);
    let r = e
        .process_batch(batch(2, &[&["a", "x"], &["a", "x"]]))
        .unwrap();
    // Paper W0={a}: WI=1-4/6=1/3. Main includes newly assigned x in W0.
    errors(&r, h, [0.0, 0.0, 0.0]);
    assert_eq!(expected_count(&e, h), 1);
    assert_eq!(e.export_baseline_tree().unwrap().nodes[0].word_ids.len(), 1);
}

#[test]
fn construction_expected_word_is_counted_twice_even_without_vocab_change() {
    let mut e = engine(config(0.5, 0.5), &[&["a", "b"], &["a", "b"]]);
    let h = root(&e);
    let r = e.process_batch(batch(2, &[&["a", "b"]])).unwrap();
    // b remains in expected_words despite already describing the node: denominator 9.
    errors(&r, h, [0.0, 0.0, 0.3333]);
    assert!(!r.reconstruction_triggered);
}

#[test]
fn old_word_losing_strength_and_eligibility_has_independent_errors() {
    let mut e = engine(config(0.5, 0.5), &[&["a", "b"], &["a", "b"]]);
    let h = root(&e);
    let r = e
        .process_batch(batch(2, &[&["a"], &["a"], &["a"], &["a"]]))
        .unwrap();
    errors(&r, h, [0.0833, 0.0833, 0.0]);
}

#[test]
fn unassigned_expected_word_is_excluded_from_old_vocab_on_next_batch() {
    let mut e = engine(config(0.5, 0.9), &[&["a"], &["a"]]);
    let h = root(&e);
    e.process_batch(batch(2, &[&["a", "x"], &["x"]])).unwrap();
    let r = e.process_batch(batch(3, &[&["a", "x"], &["x"]])).unwrap();
    // x has a synthetic negative ID, absent from retained token legend: D0(x)=0.
    errors(&r, h, [0.0, 0.0, 0.3333]);
    assert_eq!(expected_count(&e, h), 1);
    assert_eq!(
        e.last_batch_word_stats_summary()
            .unwrap()
            .new_words_in_batch,
        1
    );
}

#[test]
fn error_threshold_is_strict_after_rounding() {
    for (threshold, trigger) in [(0.0833, false), (0.08329, true)] {
        let mut c = config(0.5, 0.5);
        c.seca_thresholds.alpha_option1_threshold = threshold;
        let mut e = engine(c, &[&["a", "b"], &["a", "b"]]);
        let h = root(&e);
        let r = e
            .process_batch(batch(2, &[&["a"], &["a"], &["a"], &["a"]]))
            .unwrap();
        errors(&r, h, [0.0833, 0.0833, 0.0]);
        assert_eq!(r.reconstruction_triggered, trigger);
    }
}

#[test]
fn importance_threshold_equality_and_just_below() {
    for (threshold, trigger) in [(0.4, false), (0.3999, true)] {
        let mut c = config(0.5, 0.5);
        c.seca_thresholds.word_importance_option1_threshold = threshold;
        let mut e = engine(c, &[&["a"], &["a"]]);
        let h = root(&e);
        let r = e.process_batch(batch(2, &[&["a", "x"], &["x"]])).unwrap();
        errors(&r, h, [0.0, 0.0, 0.4]);
        assert_eq!(r.reconstruction_triggered, trigger);
    }
}

#[test]
fn parent_child_scopes_only_receive_parent_matched_sources() {
    let mut c = config(0.7, 0.5);
    c.hkt_builder
        .minimum_number_of_sources_to_create_branch_for_node = 1;
    let mut e = engine(c, &[&["a", "b"], &["a", "b"], &["a"], &["a"]]);
    let t = e.export_baseline_tree().unwrap();
    let h = root(&e);
    let child = t
        .hkts
        .iter()
        .find(|x| x.parent_node_id != 0)
        .unwrap()
        .hkt_id;
    let r = e.process_batch(batch(2, &[&["a", "b"], &["x"]])).unwrap();
    assert!(r
        .notes
        .iter()
        .any(|n| n.starts_with(&format!("HKT {h}: scoped_sources=2,"))));
    assert!(r
        .notes
        .iter()
        .any(|n| n.starts_with(&format!("HKT {child}: scoped_sources=1,"))));
    errors(&r, h, [0.0, 0.0, 0.0]);
    errors(&r, child, [0.0, 0.0, 0.0]);
}

#[test]
fn historical_candidate_occurrences_count_for_alpha_but_not_beta_admission() {
    let mut e = engine(
        config(0.6, 0.5),
        &[&["a", "x"], &["a", "x"], &["a"], &["a"]],
    );
    let h = root(&e);
    let r = e
        .process_batch(batch(2, &[&["a", "x"], &["a", "x"]]))
        .unwrap();
    // Full-history beta would be 4/6 >= .5; admission uses incoming 2/6.
    errors(&r, h, [0.0, 0.0, 0.4]);
    assert_eq!(expected_count(&e, h), 1);
    assert_eq!(
        e.export_baseline_tree()
            .unwrap()
            .nodes
            .iter()
            .filter(|n| !n.is_refuge_node)
            .next()
            .unwrap()
            .word_ids
            .len(),
        1
    );
}

#[test]
fn candidate_shared_across_two_nodes_overwrites_measurement_by_node_order() {
    let mut e = engine(config(0.5, 0.5), &[&["a"], &["a"], &["b"], &["b"]]);
    let h = root(&e);
    let r = e
        .process_batch(batch(
            2,
            &[&["a", "x"], &["a", "x"], &["b", "x"], &["b", "x"]],
        ))
        .unwrap();
    // Baseline b is also expected: denominator a4+b4+b4+x4=16.
    // Assigned x has node-local numerator 2, and is incorrectly treated as old.
    errors(&r, h, [0.0, 0.0, 0.375]);
    assert!(!r.reconstruction_triggered);
}

#[test]
fn beta_error_threshold_equality_and_just_below() {
    for (threshold, trigger) in [(0.0833, false), (0.08329, true)] {
        let mut c = config(0.5, 0.5);
        c.seca_thresholds.beta_option1_threshold = threshold;
        let mut e = engine(c, &[&["a", "b"], &["a", "b"]]);
        let h = root(&e);
        let r = e
            .process_batch(batch(2, &[&["a"], &["a"], &["a"], &["a"]]))
            .unwrap();
        errors(&r, h, [0.0833, 0.0833, 0.0]);
        assert_eq!(r.reconstruction_triggered, trigger);
    }
}

#[test]
fn refuge_routes_unmatched_sources_but_does_not_commit_their_membership() {
    let mut e = engine(config(0.7, 0.5), &[&["a"], &["a"], &["a"], &["x"]]);
    let h = root(&e);
    let before = e
        .export_baseline_tree()
        .unwrap()
        .nodes
        .into_iter()
        .find(|n| n.is_refuge_node)
        .unwrap();
    let r = e.process_batch(batch(2, &[&["x"]])).unwrap();
    errors(&r, h, [0.0, 0.0, 0.0]);
    let after = e
        .export_baseline_tree()
        .unwrap()
        .nodes
        .into_iter()
        .find(|n| n.node_id == before.node_id)
        .unwrap();
    assert_eq!(after.source_ids, before.source_ids);
    assert!(!r.reconstruction_triggered);
}

#[test]
fn newly_assigned_word_masks_paper_importance_reconstruction() {
    let mut c = config(0.5, 0.5);
    c.seca_thresholds.word_importance_option1_threshold = 0.2;
    let mut e = engine(c, &[&["a"], &["a"]]);
    let h = root(&e);
    let r = e
        .process_batch(batch(2, &[&["a", "x"], &["a", "x"]]))
        .unwrap();
    errors(&r, h, [0.0, 0.0, 0.0]);
    assert!(!r.reconstruction_triggered); // Paper/C# WI=1/3 > .2.
}

#[test]
fn duplicate_expected_descriptor_causes_spurious_reconstruction() {
    let mut c = config(0.5, 0.5);
    c.seca_thresholds.word_importance_option1_threshold = 0.3;
    let mut e = engine(c, &[&["a", "b"], &["a", "b"]]);
    let h = root(&e);
    let r = e.process_batch(batch(2, &[&["a", "b"]])).unwrap();
    errors(&r, h, [0.0, 0.0, 0.3333]);
    assert!(r.reconstruction_triggered); // Paper/C# unchanged vocabulary gives zero.
}

#[test]
fn empty_incoming_scope_still_reports_errors_from_duplicate_expected_word() {
    let mut e = engine(config(0.5, 0.5), &[&["a", "b"], &["a", "b"]]);
    let h = root(&e);
    let r = e.process_batch(batch(2, &[])).unwrap();
    errors(&r, h, [0.0, 0.0, 0.3333]);
    assert!(!r.reconstruction_triggered);
}

#[test]
fn invalid_sequence_fails_before_changing_context() {
    let mut e = engine(config(0.5, 0.5), &[&["a"]]);
    let before = serde_json::to_value(e.export_baseline_tree_verbose().unwrap()).unwrap();
    assert!(e.process_batch(batch(3, &[&["a", "x"]])).is_err());
    assert_eq!(
        serde_json::to_value(e.export_baseline_tree_verbose().unwrap()).unwrap(),
        before
    );
    let r = e.process_batch(batch(2, &[&["a"]])).unwrap();
    errors(&r, root(&e), [0.0, 0.0, 0.0]);
}

#[test]
fn candidate_beta_uses_prominent_sources_while_error_uses_full_node_union() {
    let mut e = engine(
        config(0.4, 0.4),
        &[&["a", "b"], &["a", "b"], &["a"], &["b"]],
    );
    let h = root(&e);
    let r = e
        .process_batch(batch(2, &[&["b", "x"], &["b", "x"]]))
        .unwrap();
    // Node union is 6, prominent b has 5 sources. Incoming x passes 2/5 >= .4,
    // but x's error eligibility is 2/6 < .4. W0 incorrectly includes x.
    // b is also still expected: importance denominator a3+b5+b5+x2=15.
    errors(&r, h, [0.0, 0.0222, 0.3333]);
    assert!(!r.reconstruction_triggered);
}
