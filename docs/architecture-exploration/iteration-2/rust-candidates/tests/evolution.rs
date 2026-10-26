//! Independent lifecycle witnesses: expectations are derived from set arithmetic.
use seca_research_candidates::{
    evolution::{Batch, Engine, Source, StreamInput},
    Input, Row, Scope,
};
use std::collections::BTreeSet;

fn request(docs: &[(i64, &[i32])]) -> StreamInput {
    StreamInput {
        construction: Input {
            records: docs
                .iter()
                .flat_map(|(source, words)| {
                    words.iter().map(move |word| Row {
                        source: *source,
                        word: *word,
                    })
                })
                .collect(),
            alpha: 0.7,
            beta: 0.5,
            branch_minimum: 1,
        },
        initial_index: 0,
        batches: vec![],
        gamma: None,
        alpha_error_threshold: 1.0,
        beta_error_threshold: 1.0,
        importance_error_threshold: 1.0,
    }
}
fn batch(index: u64, docs: &[(i64, &[i32])]) -> Batch {
    Batch {
        index,
        sources: docs
            .iter()
            .map(|(id, words)| Source {
                id: *id,
                words: words.to_vec(),
            })
            .collect(),
    }
}
fn scope(engine: &Engine) -> &Scope {
    engine.tree.as_ref().unwrap()
}
fn ids(s: &Scope) -> BTreeSet<i32> {
    std::iter::once(s.hkt_id)
        .chain(
            s.nodes
                .iter()
                .filter_map(|n| n.child.as_deref())
                .flat_map(ids),
        )
        .collect()
}
fn snapshot(engine: &Engine) -> serde_json::Value {
    serde_json::to_value(engine).unwrap()
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}

#[test]
fn overlapping_mapping_is_not_a_partition() {
    let mut r = request(&[(1, &[1]), (2, &[1]), (3, &[2]), (4, &[2])]);
    r.construction.alpha = 1.0;
    r.construction.beta = 1.0;
    r.construction.branch_minimum = 99;
    let mut engine = Engine::new(&r).unwrap();
    let report = engine.process(&batch(1, &[(5, &[1, 2])])).unwrap();
    assert!(report.replacements.is_empty());
    for n in &scope(&engine).nodes {
        assert!(n.sources.contains(&5));
        for sources in n.word_sources.values() {
            assert!(sources.contains(&5));
        }
    }
    assert_eq!(scope(&engine).nodes.len(), 2);
}

#[test]
fn candidate_history_counts_retained_sources_without_promoting_descriptors() {
    let mut r = request(&[(1, &[1]), (2, &[1])]);
    r.construction.alpha = 0.6;
    r.construction.beta = 0.9;
    let mut engine = Engine::new(&r).unwrap();
    for (index, a, x, wi) in [(1, 3, 4, 2.0 / 5.0), (2, 5, 6, 0.5), (3, 7, 8, 6.0 / 11.0)] {
        let report = engine
            .process(&batch(index, &[(a, &[1, 2]), (x, &[2])]))
            .unwrap();
        close(report.decisions[0].paper_word_importance_error, wi);
        assert_eq!(scope(&engine).expected_words, BTreeSet::from([2]));
        assert_eq!(scope(&engine).nodes[0].words, BTreeSet::from([1]));
        assert!(!scope(&engine).nodes[0].word_sources.contains_key(&2));
    }
}

#[test]
fn parent_trigger_suppresses_descendant_decisions_and_recycles_no_ids() {
    let mut r = request(&[(1, &[1, 2]), (2, &[1, 2]), (3, &[1]), (4, &[1])]);
    r.importance_error_threshold = 0.1;
    let mut engine = Engine::new(&r).unwrap();
    let old_ids = ids(scope(&engine));
    assert!(old_ids.len() > 1);
    let old_max = *old_ids.last().unwrap();
    let report = engine
        .process(&batch(1, &[(5, &[3]), (6, &[3]), (7, &[3]), (8, &[3])]))
        .unwrap();
    assert_eq!(report.decisions.len(), 1);
    assert!(report.decisions[0].should_reconstruct);
    assert_eq!(report.replacements.len(), 1);
    assert!(ids(scope(&engine)).iter().all(|id| *id > old_max));
    let max_after = *ids(scope(&engine)).last().unwrap();
    let report = engine
        .process(&batch(2, &[(9, &[4]), (10, &[4]), (11, &[4]), (12, &[4])]))
        .unwrap();
    assert_eq!(report.decisions.len(), 1);
    assert!(report.decisions[0].should_reconstruct);
    assert!(ids(scope(&engine)).iter().all(|id| *id > max_after));
}

#[test]
fn disjoint_children_reconstruct_while_unaffected_sibling_is_identical() {
    let mut r = request(&[
        (1, &[1, 2]),
        (2, &[1, 2]),
        (3, &[1]),
        (4, &[1]),
        (5, &[3, 4]),
        (6, &[3, 4]),
        (7, &[3]),
        (8, &[3]),
        (9, &[7, 8]),
        (10, &[7, 8]),
        (11, &[7]),
        (12, &[7]),
    ]);
    r.importance_error_threshold = 0.1;
    let mut engine = Engine::new(&r).unwrap();
    let root_id = scope(&engine).hkt_id;
    let untouched = scope(&engine)
        .nodes
        .iter()
        .find(|n| n.words.contains(&7))
        .unwrap()
        .clone();
    let target_ids: BTreeSet<_> = scope(&engine)
        .nodes
        .iter()
        .filter(|n| !n.words.contains(&7))
        .map(|n| n.child.as_ref().unwrap().hkt_id)
        .collect();
    let report = engine
        .process(&batch(
            1,
            &[
                (13, &[1, 5]),
                (14, &[1, 5]),
                (15, &[1, 5]),
                (16, &[3, 6]),
                (17, &[3, 6]),
                (18, &[3, 6]),
            ],
        ))
        .unwrap();
    assert_eq!(scope(&engine).hkt_id, root_id);
    assert_eq!(
        report.replacements.keys().copied().collect::<BTreeSet<_>>(),
        target_ids
    );
    assert_eq!(
        scope(&engine)
            .nodes
            .iter()
            .find(|n| n.node_id == untouched.node_id)
            .unwrap(),
        &untouched
    );
    assert!(!report.decisions[0].should_reconstruct);
    assert_eq!(
        report
            .decisions
            .iter()
            .filter(|d| d.should_reconstruct)
            .count(),
        2
    );
}

#[test]
fn empty_batches_expire_evidence_but_keep_learned_topology_and_allow_readmission() {
    let mut r = request(&[(1, &[1]), (2, &[1])]);
    r.gamma = Some(2);
    let mut engine = Engine::new(&r).unwrap();
    let node_id = scope(&engine).nodes[0].node_id;
    assert!(engine.process(&batch(1, &[])).unwrap().decisions.is_empty());
    assert!(engine.process(&batch(2, &[])).unwrap().decisions.is_empty());
    assert_eq!(scope(&engine).nodes[0].node_id, node_id);
    assert_eq!(scope(&engine).nodes[0].words, BTreeSet::from([1]));
    assert!(scope(&engine).nodes[0].sources.is_empty());
    engine.process(&batch(3, &[(1, &[1])])).unwrap();
    assert_eq!(scope(&engine).nodes[0].sources, BTreeSet::from([1]));
    assert_eq!(scope(&engine).nodes[0].node_id, node_id);
}

#[test]
fn full_memory_keeps_sources_across_empty_updates() {
    let r = request(&[(1, &[1]), (2, &[1])]);
    let mut engine = Engine::new(&r).unwrap();
    for index in 1..=4 {
        engine.process(&batch(index, &[])).unwrap();
    }
    assert_eq!(scope(&engine).nodes[0].sources, BTreeSet::from([1, 2]));
}

#[test]
fn validation_precedes_deduplication_and_failures_preserve_complete_state() {
    let r = request(&[(1, &[1]), (2, &[1])]);
    let mut engine = Engine::new(&r).unwrap();
    let before = snapshot(&engine);
    for invalid in [
        batch(2, &[(3, &[1])]),
        batch(1, &[(3, &[1]), (3, &[0])]),
        batch(1, &[(1, &[-1])]),
    ] {
        assert!(engine.process(&invalid).is_err());
        assert_eq!(snapshot(&engine), before);
    }
    engine.process(&batch(1, &[(3, &[1]), (3, &[2])])).unwrap();
    assert_eq!(scope(&engine).nodes[0].sources, BTreeSet::from([1, 2, 3]));
    let state = snapshot(&engine);
    assert!(!state["vocabulary"]
        .as_array()
        .unwrap()
        .contains(&serde_json::json!(2)));
}

#[test]
fn json_restart_preserves_subsequent_decisions_allocators_and_state() {
    let mut r = request(&[(1, &[1, 2]), (2, &[1, 2]), (3, &[1])]);
    r.importance_error_threshold = 0.2;
    r.gamma = Some(2);
    let mut continuous = Engine::new(&r).unwrap();
    let mut restarted = continuous.clone();
    for b in [
        batch(1, &[(4, &[1, 3]), (5, &[3])]),
        batch(2, &[(6, &[4]), (7, &[4])]),
        batch(3, &[]),
        batch(4, &[(1, &[1, 4])]),
    ] {
        let expected = continuous.process(&b).unwrap();
        restarted = serde_json::from_value(snapshot(&restarted)).unwrap();
        let actual = restarted.process(&b).unwrap();
        assert_eq!(
            serde_json::to_value(expected).unwrap(),
            serde_json::to_value(actual).unwrap()
        );
        assert_eq!(snapshot(&continuous), snapshot(&restarted));
    }
}

#[test]
fn descriptor_errors_use_contextual_node_union_and_distinct_word_counts() {
    let mut r = request(&[(1, &[1]), (2, &[1, 2]), (3, &[2])]);
    r.construction.alpha = 0.6;
    let mut engine = Engine::new(&r).unwrap();
    assert_eq!(scope(&engine).nodes.len(), 1);
    let report = engine.process(&batch(1, &[(4, &[1]), (5, &[1])])).unwrap();
    let decision = &report.decisions[0];
    // Updated DF=(4,2), prominent=4 and node union=5. Each error averages two words.
    close(decision.paper_alpha_error, 0.05);
    close(decision.paper_beta_error, 0.05);
    close(decision.paper_word_importance_error, 0.0);
}

#[test]
fn importance_trigger_is_strict_and_unassigned_expectations_do_not_join_w0() {
    let mut r = request(&[(1, &[1]), (2, &[1])]);
    r.importance_error_threshold = 0.6;
    let mut engine = Engine::new(&r).unwrap();
    let first = engine
        .process(&batch(1, &[(3, &[2]), (4, &[2]), (5, &[2])]))
        .unwrap();
    close(first.decisions[0].paper_word_importance_error, 3.0 / 5.0);
    assert!(!first.decisions[0].should_reconstruct);
    assert_eq!(scope(&engine).expected_words, BTreeSet::from([2]));
    let second = engine.process(&batch(2, &[(6, &[2])])).unwrap();
    close(second.decisions[0].paper_word_importance_error, 2.0 / 3.0);
    assert!(second.decisions[0].should_reconstruct);
}

#[test]
fn unsupported_empty_root_failure_rolls_back_new_vocabulary_registration() {
    let r = request(&[]);
    let mut engine = Engine::new(&r).unwrap();
    let before = snapshot(&engine);
    let error = engine.process(&batch(1, &[(1, &[1])])).unwrap_err();
    assert!(error.contains("empty-baseline evolution unsupported"));
    // This failure occurs after candidate vocabulary registration, before tree traversal.
    assert_eq!(snapshot(&engine), before);
}

#[test]
fn expiry_prunes_every_recursive_membership_and_preserves_topology_vocabulary_allocators() {
    fn assert_empty(scope: &Scope) {
        for node in &scope.nodes {
            assert!(node.sources.is_empty());
            assert!(node.word_sources.is_empty());
            if let Some(child) = node.child.as_deref() {
                assert_empty(child);
            }
        }
    }
    let mut r = request(&[(1, &[1, 2]), (2, &[1, 2]), (3, &[1]), (4, &[1])]);
    r.gamma = Some(1);
    let mut engine = Engine::new(&r).unwrap();
    let previous = snapshot(&engine);
    let topology_ids = ids(scope(&engine));
    assert!(topology_ids.len() > 1);
    engine.process(&batch(1, &[])).unwrap();
    assert_eq!(ids(scope(&engine)), topology_ids);
    assert_empty(scope(&engine));
    let current = snapshot(&engine);
    for field in ["vocabulary", "next_hkt", "next_node"] {
        assert_eq!(current[field], previous[field]);
    }
    assert_eq!(current["evidence"].as_array().unwrap().len(), 1);
    assert!(current["evidence"][0]["sources"]
        .as_array()
        .unwrap()
        .is_empty());
}
