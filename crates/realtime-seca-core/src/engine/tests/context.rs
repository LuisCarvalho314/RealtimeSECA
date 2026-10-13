use super::super::*;
use crate::config::SecaConfig;

use crate::types::{SourceBatch, SourceRecord};

use super::fixtures::*;
use std::collections::{BTreeMap, BTreeSet};
#[test]
fn update_stage_includes_new_tokens_in_expected_words_count() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha = 0.0; // accept all state1 tokens into expected words
    config.seca_thresholds.beta = 0.0; // accept expected words into nodes

    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["a"])]))
        .unwrap();

    let root_hkt_id = engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .hkts_by_id
        .values()
        .find(|hkt| hkt.parent_node_id == 0)
        .unwrap()
        .hkt_id;

    let scope_snapshot = engine.snapshot_hkt_scope(root_hkt_id).unwrap();
    let mut current_word_ids: BTreeSet<i32> = BTreeSet::new();
    for word_ids in scope_snapshot.node_word_ids_by_node_id.values() {
        for word_id in word_ids {
            if *word_id != -1 {
                current_word_ids.insert(*word_id);
            }
        }
    }
    let current_tokens: BTreeSet<String> = current_word_ids
        .iter()
        .filter_map(|word_id| engine.baseline_word_legend.get(word_id))
        .cloned()
        .collect();

    let batch = make_batch(1, &[("n1", &["a", "new_x"]), ("n2", &["new_y"])]);
    let mut batch_tokens = BTreeSet::new();
    for source in &batch.sources {
        for token in &source.tokens {
            let normalized = token.trim();
            if normalized.is_empty() {
                continue;
            }
            batch_tokens.insert(normalized.to_string());
        }
    }

    let expected_state1_count: usize = batch_tokens.difference(&current_tokens).count();

    let all_source_indexes: BTreeSet<usize> = (0..batch.sources.len()).collect();
    let mapped_scope = engine
        .map_batch_into_hkt_scope(&batch, &all_source_indexes, &scope_snapshot)
        .unwrap();
    let update_stage = engine
        .compute_update_stage_for_scope(
            &batch,
            &scope_snapshot,
            &mapped_scope,
            &crate::engine::trigger::AncestorContext::default(),
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();

    assert_eq!(
        update_stage.updated_expected_word_ids.len(),
        expected_state1_count,
        "expected update-stage expected_words_state1 count to include new tokens"
    );

    let baseline_word_ids: BTreeSet<i32> = engine.baseline_word_legend.keys().copied().collect();
    let baseline_tokens: BTreeSet<String> = engine.baseline_word_legend.values().cloned().collect();
    let new_token_count = batch_tokens.difference(&baseline_tokens).count();
    let synthetic_expected_count = update_stage
        .updated_expected_word_ids
        .difference(&baseline_word_ids)
        .count();
    assert_eq!(
        synthetic_expected_count, new_token_count,
        "expected synthetic expected-word ids for each new token"
    );

    let assigned_synthetic = update_stage
        .assigned_expected_words_by_node_id
        .values()
        .any(|word_ids| word_ids.iter().any(|id| !baseline_word_ids.contains(id)));
    assert!(
        assigned_synthetic,
        "expected at least one synthetic expected word to be assigned to a node"
    );
}
#[test]
fn update_stage_resets_expected_words_below_alpha_threshold() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha = 0.6;
    config.seca_thresholds.beta = 0.0;

    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["a"])]))
        .unwrap();

    let root_hkt_id = engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .hkts_by_id
        .values()
        .find(|hkt| hkt.parent_node_id == 0)
        .unwrap()
        .hkt_id;

    let scope_snapshot = engine.snapshot_hkt_scope(root_hkt_id).unwrap();
    let batch = make_batch(1, &[("n1", &["a", "new_x"])]);
    let all_source_indexes: BTreeSet<usize> = (0..batch.sources.len()).collect();
    let mapped_scope = engine
        .map_batch_into_hkt_scope(&batch, &all_source_indexes, &scope_snapshot)
        .unwrap();
    let update_stage = engine
        .compute_update_stage_for_scope(
            &batch,
            &scope_snapshot,
            &mapped_scope,
            &crate::engine::trigger::AncestorContext::default(),
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();

    let baseline_word_ids: BTreeSet<i32> = engine.baseline_word_legend.keys().copied().collect();
    let synthetic_expected_count = update_stage
        .updated_expected_word_ids
        .difference(&baseline_word_ids)
        .count();
    assert_eq!(
        synthetic_expected_count, 0,
        "expected low-frequency new tokens to be pruned by reset pass"
    );
}
#[test]
fn update_stage_excludes_ancestor_accepted_words_from_expected_set() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha = 0.0;
    config.seca_thresholds.beta = 0.0;

    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["a"])]))
        .unwrap();

    let root_hkt_id = engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .hkts_by_id
        .values()
        .find(|hkt| hkt.parent_node_id == 0)
        .unwrap()
        .hkt_id;

    let scope_snapshot = engine.snapshot_hkt_scope(root_hkt_id).unwrap();

    let batch = make_batch(1, &[("n1", &["new_x"])]);
    let all_source_indexes: BTreeSet<usize> = (0..batch.sources.len()).collect();
    let mapped_scope = engine
        .map_batch_into_hkt_scope(&batch, &all_source_indexes, &scope_snapshot)
        .unwrap();

    let mut ancestor_accepted = BTreeMap::new();
    ancestor_accepted.insert(-2_i32, (1_usize, 0_usize));
    let ancestor = crate::engine::trigger::AncestorContext::with_sets(
        BTreeMap::new(),
        ancestor_accepted,
        BTreeMap::new(),
    );

    let update_stage = engine
        .compute_update_stage_for_scope(
            &batch,
            &scope_snapshot,
            &mapped_scope,
            &ancestor,
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();

    let baseline_word_ids: BTreeSet<i32> = engine.baseline_word_legend.keys().copied().collect();
    let synthetic_expected_count = update_stage
        .updated_expected_word_ids
        .difference(&baseline_word_ids)
        .count();
    assert_eq!(
        synthetic_expected_count, 0,
        "expected ancestor-accepted words to be excluded from expected-word discovery"
    );
}
#[test]
fn update_stage_uses_state0_prominent_word_for_alpha_ratio() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha = 0.6;
    config.seca_thresholds.beta = 0.0;

    let mut engine = SecaEngine::new(config).unwrap();

    let mut baseline_sources = Vec::new();
    for idx in 0..10 {
        baseline_sources.push(SourceRecord {
            source_id: format!("s{idx}"),
            batch_index: 0,
            tokens: vec!["a".to_string()],
            text: None,
            timestamp_unix_ms: None,
            metadata: None,
        });
    }
    let baseline_batch = SourceBatch {
        batch_index: 0,
        sources: baseline_sources,
    };

    engine.build_baseline_tree(baseline_batch).unwrap();

    let root_hkt_id = engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .hkts_by_id
        .values()
        .find(|hkt| hkt.parent_node_id == 0)
        .unwrap()
        .hkt_id;

    let scope_snapshot = engine.snapshot_hkt_scope(root_hkt_id).unwrap();

    let mut batch_sources = Vec::new();
    for idx in 0..5 {
        batch_sources.push(SourceRecord {
            source_id: format!("n{idx}"),
            batch_index: 1,
            tokens: vec!["new_x".to_string()],
            text: None,
            timestamp_unix_ms: None,
            metadata: None,
        });
    }
    let batch = SourceBatch {
        batch_index: 1,
        sources: batch_sources,
    };

    let all_source_indexes: BTreeSet<usize> = (0..batch.sources.len()).collect();
    let mapped_scope = engine
        .map_batch_into_hkt_scope(&batch, &all_source_indexes, &scope_snapshot)
        .unwrap();

    let update_stage = engine
        .compute_update_stage_for_scope(
            &batch,
            &scope_snapshot,
            &mapped_scope,
            &crate::engine::trigger::AncestorContext::default(),
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();

    let baseline_word_ids: BTreeSet<i32> = engine.baseline_word_legend.keys().copied().collect();
    let synthetic_expected_count = update_stage
        .updated_expected_word_ids
        .difference(&baseline_word_ids)
        .count();

    assert_eq!(
        synthetic_expected_count, 0,
        "expected new tokens to be excluded because state0 prominent count dominates"
    );
}
#[test]
fn update_stage_includes_state0_dominant_expected_word() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha = 0.7;
    config.seca_thresholds.beta = 0.0;

    let mut engine = SecaEngine::new(config).unwrap();

    let mut baseline_sources = Vec::new();
    for idx in 0..10 {
        baseline_sources.push(SourceRecord {
            source_id: format!("s{idx}"),
            batch_index: 0,
            tokens: vec!["a".to_string()],
            text: None,
            timestamp_unix_ms: None,
            metadata: None,
        });
    }
    for idx in 10..17 {
        baseline_sources.push(SourceRecord {
            source_id: format!("s{idx}"),
            batch_index: 0,
            tokens: vec!["legacy".to_string()],
            text: None,
            timestamp_unix_ms: None,
            metadata: None,
        });
    }

    let baseline_batch = SourceBatch {
        batch_index: 0,
        sources: baseline_sources,
    };
    engine.build_baseline_tree(baseline_batch).unwrap();

    let root_hkt_id = engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .hkts_by_id
        .values()
        .find(|hkt| hkt.parent_node_id == 0)
        .unwrap()
        .hkt_id;

    let mut scope_snapshot = engine.snapshot_hkt_scope(root_hkt_id).unwrap();

    let legacy_word_id = engine
        .baseline_word_legend
        .iter()
        .find(|(_, token)| token.as_str() == "legacy")
        .map(|(id, _)| *id)
        .unwrap();

    for word_ids in scope_snapshot.node_word_ids_by_node_id.values_mut() {
        word_ids.remove(&legacy_word_id);
    }

    let batch = make_batch(1, &[("n1", &["a", "legacy"])]);
    let all_source_indexes: BTreeSet<usize> = (0..batch.sources.len()).collect();
    let mapped_scope = engine
        .map_batch_into_hkt_scope(&batch, &all_source_indexes, &scope_snapshot)
        .unwrap();

    let update_stage = engine
        .compute_update_stage_for_scope(
            &batch,
            &scope_snapshot,
            &mapped_scope,
            &crate::engine::trigger::AncestorContext::default(),
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();

    let baseline_word_ids: BTreeSet<i32> = engine.baseline_word_legend.keys().copied().collect();
    let synthetic_expected_count = update_stage
        .updated_expected_word_ids
        .difference(&baseline_word_ids)
        .count();

    assert_eq!(
        synthetic_expected_count, 0,
        "expected legacy to be included as a known baseline word, not synthetic"
    );

    assert!(
        update_stage
            .updated_expected_word_ids
            .contains(&legacy_word_id),
        "expected legacy to be admitted via state0+state1 / state0_prominent ratio"
    );
}
#[test]
fn update_stage_ancestor_rejected_words_without_state1_token_are_pruned_by_reset() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha = 0.7;
    config.seca_thresholds.beta = 0.0;

    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["a"])]))
        .unwrap();

    let root_hkt_id = engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .hkts_by_id
        .values()
        .find(|hkt| hkt.parent_node_id == 0)
        .unwrap()
        .hkt_id;

    let scope_snapshot = engine.snapshot_hkt_scope(root_hkt_id).unwrap();
    let batch = make_batch(1, &[("n1", &["a"])]);
    let all_source_indexes: BTreeSet<usize> = (0..batch.sources.len()).collect();
    let mapped_scope = engine
        .map_batch_into_hkt_scope(&batch, &all_source_indexes, &scope_snapshot)
        .unwrap();

    // new token is not in batch, but should be admitted via ancestor rejected counts
    let mut ancestor_rejected = BTreeMap::new();
    ancestor_rejected.insert(-2_i32, (1_usize, 1_usize)); // ratio 2 / prominent(2) = 1.0
    let ancestor = crate::engine::trigger::AncestorContext::with_sets(
        BTreeMap::new(),
        BTreeMap::new(),
        ancestor_rejected,
    );

    let update_stage = engine
        .compute_update_stage_for_scope(
            &batch,
            &scope_snapshot,
            &mapped_scope,
            &ancestor,
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();

    assert!(
        !update_stage.updated_expected_word_ids.contains(&-2),
        "expected ancestor-rejected word without state1 token to be pruned by reset"
    );
}
#[test]
fn update_stage_state0_parent_scope_uses_parent_node_sources_only() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha = 0.7;
    config.seca_thresholds.beta = 0.0;

    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(
            0,
            &[
                ("s1", &["a", "legacy"]),
                ("s2", &["a", "legacy"]),
                ("s3", &["a", "legacy"]),
                ("s4", &["a"]),
            ],
        ))
        .unwrap();

    let root_hkt_id = engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .hkts_by_id
        .values()
        .find(|hkt| hkt.parent_node_id == 0)
        .unwrap()
        .hkt_id;

    let mut scope_snapshot = engine.snapshot_hkt_scope(root_hkt_id).unwrap();

    // Force parent scope to only use sources s1,s2 (so legacy count=2, prominent count=2).
    for source_ids in scope_snapshot.node_source_ids_by_node_id.values_mut() {
        source_ids
            .retain(|id| *id == SecaEngine::fnv1a_64("s1") || *id == SecaEngine::fnv1a_64("s2"));
    }

    let legacy_word_id = engine
        .baseline_word_legend
        .iter()
        .find(|(_, token)| token.as_str() == "legacy")
        .map(|(id, _)| *id)
        .unwrap();
    for word_ids in scope_snapshot.node_word_ids_by_node_id.values_mut() {
        word_ids.remove(&legacy_word_id);
    }

    let batch = make_batch(1, &[("n1", &["a", "legacy"])]);
    let all_source_indexes: BTreeSet<usize> = (0..batch.sources.len()).collect();
    let mapped_scope = engine
        .map_batch_into_hkt_scope(&batch, &all_source_indexes, &scope_snapshot)
        .unwrap();

    let update_stage = engine
        .compute_update_stage_for_scope(
            &batch,
            &scope_snapshot,
            &mapped_scope,
            &crate::engine::trigger::AncestorContext::default(),
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();

    assert!(
        update_stage
            .updated_expected_word_ids
            .contains(&legacy_word_id),
        "expected legacy to be admitted using parent-scope state0 counts"
    );
}
#[test]
fn update_stage_beta_eligibility_uses_prominent_word_sources() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha = 0.0; // ensure new_x becomes expected
    config.seca_thresholds.beta = 0.5;

    let mut engine = SecaEngine::new(config).unwrap();

    let mut baseline_sources = Vec::new();
    for idx in 0..10 {
        baseline_sources.push(SourceRecord {
            source_id: format!("s{idx}"),
            batch_index: 0,
            tokens: vec!["a".to_string()],
            text: None,
            timestamp_unix_ms: None,
            metadata: None,
        });
    }
    let baseline_batch = SourceBatch {
        batch_index: 0,
        sources: baseline_sources,
    };

    engine.build_baseline_tree(baseline_batch).unwrap();

    let root_hkt_id = engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .hkts_by_id
        .values()
        .find(|hkt| hkt.parent_node_id == 0)
        .unwrap()
        .hkt_id;

    let scope_snapshot = engine.snapshot_hkt_scope(root_hkt_id).unwrap();

    let mut batch_sources = Vec::new();
    for idx in 0..2 {
        batch_sources.push(SourceRecord {
            source_id: format!("n{idx}"),
            batch_index: 1,
            tokens: vec!["a".to_string(), "new_x".to_string()],
            text: None,
            timestamp_unix_ms: None,
            metadata: None,
        });
    }
    let batch = SourceBatch {
        batch_index: 1,
        sources: batch_sources,
    };

    let all_source_indexes: BTreeSet<usize> = (0..batch.sources.len()).collect();
    let mapped_scope = engine
        .map_batch_into_hkt_scope(&batch, &all_source_indexes, &scope_snapshot)
        .unwrap();

    let update_stage = engine
        .compute_update_stage_for_scope(
            &batch,
            &scope_snapshot,
            &mapped_scope,
            &crate::engine::trigger::AncestorContext::default(),
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();

    let baseline_word_ids: BTreeSet<i32> = engine.baseline_word_legend.keys().copied().collect();
    let synthetic_word_ids: Vec<i32> = update_stage
        .updated_expected_word_ids
        .difference(&baseline_word_ids)
        .copied()
        .collect();

    assert_eq!(
        synthetic_word_ids.len(),
        1,
        "expected exactly one synthetic expected word for new_x"
    );

    let assigned_synthetic = update_stage
        .assigned_expected_words_by_node_id
        .values()
        .any(|word_ids| word_ids.contains(&synthetic_word_ids[0]));

    assert!(
        !assigned_synthetic,
        "expected new_x to fail beta eligibility when using prominent-word denominator"
    );
}
#[test]
fn update_stage_assigns_expected_word_when_beta_satisfied() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha = 0.0; // admit new_x as expected
    config.seca_thresholds.beta = 0.5;

    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["a"])]))
        .unwrap();

    let root_hkt_id = root_hkt_id(&engine);
    let scope_snapshot = engine.snapshot_hkt_scope(root_hkt_id).unwrap();

    let batch = make_batch(1, &[("n1", &["a", "new_x"]), ("n2", &["a", "new_x"])]);
    let all_source_indexes: BTreeSet<usize> = (0..batch.sources.len()).collect();
    let mapped_scope = engine
        .map_batch_into_hkt_scope(&batch, &all_source_indexes, &scope_snapshot)
        .unwrap();

    let update_stage = engine
        .compute_update_stage_for_scope(
            &batch,
            &scope_snapshot,
            &mapped_scope,
            &crate::engine::trigger::AncestorContext::default(),
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();

    let baseline_word_ids: BTreeSet<i32> = engine.baseline_word_legend.keys().copied().collect();
    let synthetic_word_ids: Vec<i32> = update_stage
        .updated_expected_word_ids
        .difference(&baseline_word_ids)
        .copied()
        .collect();
    assert_eq!(
        synthetic_word_ids.len(),
        1,
        "expected exactly one synthetic expected word for new_x"
    );
    let synthetic_word_id = synthetic_word_ids[0];

    let assigned = update_stage
        .assigned_expected_words_by_node_id
        .values()
        .any(|word_ids| word_ids.contains(&synthetic_word_id));

    assert!(
        assigned,
        "expected new_x to be assigned when eligibility meets beta threshold"
    );
}
#[test]
fn accepted_hkt_retains_candidates_without_changing_old_node_vocabulary() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha = 0.0;
    config.seca_thresholds.alpha_error_threshold = 1.0;
    config.seca_thresholds.beta_error_threshold = 1.0;
    config.seca_thresholds.word_importance_error_threshold = 1.0;
    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["a"])]))
        .unwrap();
    let before = engine.export_baseline_tree().unwrap().nodes[0]
        .word_ids
        .clone();
    assert!(
        !engine
            .process_batch(make_batch(1, &[("s_new", &["a", "new_x"])]))
            .unwrap()
            .reconstruction_triggered
    );
    let output = engine.export_baseline_tree().unwrap();
    assert_eq!(output.nodes[0].word_ids, before);
    let candidate_id = engine
        .baseline_word_legend
        .iter()
        .find(|(_, t)| *t == "new_x")
        .unwrap()
        .0;
    assert!(output.hkts[0].expected_words.contains(candidate_id));
    assert!(!output.nodes[0].word_ids.contains(candidate_id));
}
#[test]
fn word_change_metrics_match_csharp_for_simple_node() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha = 0.0;
    config.seca_thresholds.beta = 0.0;

    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(
            0,
            &[("s1", &["a"]), ("s2", &["a"]), ("s3", &["a"])],
        ))
        .unwrap();

    let root_hkt_id = root_hkt_id(&engine);
    let scope_snapshot = engine.snapshot_hkt_scope(root_hkt_id).unwrap();
    let batch = make_batch(1, &[("n1", &["a", "new_x"]), ("n2", &["a", "new_x"])]);
    let all_source_indexes: BTreeSet<usize> = (0..batch.sources.len()).collect();
    let mapped_scope = engine
        .map_batch_into_hkt_scope(&batch, &all_source_indexes, &scope_snapshot)
        .unwrap();
    let update_stage = engine
        .compute_update_stage_for_scope(
            &batch,
            &scope_snapshot,
            &mapped_scope,
            &crate::engine::trigger::AncestorContext::default(),
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();

    let baseline_word_ids: BTreeSet<i32> = engine.baseline_word_legend.keys().copied().collect();
    let synthetic_word_ids: Vec<i32> = update_stage
        .updated_expected_word_ids
        .difference(&baseline_word_ids)
        .copied()
        .collect();
    assert_eq!(
        synthetic_word_ids.len(),
        1,
        "expected exactly one synthetic expected word"
    );
    let synthetic_word_id = synthetic_word_ids[0];

    let assigned_synthetic = update_stage
        .assigned_expected_words_by_node_id
        .values()
        .any(|word_ids| word_ids.contains(&synthetic_word_id));
    assert!(
        assigned_synthetic,
        "expected synthetic expected word to be assigned to a node"
    );

    let metrics = engine
        .compute_word_change_metrics_for_scope(
            &scope_snapshot,
            &mapped_scope,
            &update_stage,
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();

    let word_id_a = engine
        .baseline_word_legend
        .iter()
        .find(|(_, token)| token.as_str() == "a")
        .map(|(id, _)| *id)
        .unwrap();

    let metrics_a = metrics.get(&word_id_a).unwrap();

    assert!((metrics_a.importance - (5.0 / 7.0)).abs() < 1e-6);
    assert!((metrics_a.strength - 1.0).abs() < 1e-6);

    let metrics_new = metrics.get(&synthetic_word_id).unwrap();
    assert!((metrics_new.importance - (2.0 / 7.0)).abs() < 1e-6);
    assert!((metrics_new.strength - 0.4).abs() < 1e-6);
    assert!((metrics_new.eligibility - 0.4).abs() < 1e-6);
}
#[test]
fn unassigned_candidate_has_scope_strength_and_no_node_eligibility() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha = 0.0; // admit expected words
    config.seca_thresholds.beta = 1.0; // prevent assignment to nodes

    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["a"])]))
        .unwrap();

    let root_hkt_id = root_hkt_id(&engine);
    let scope_snapshot = engine.snapshot_hkt_scope(root_hkt_id).unwrap();
    let batch = make_batch(1, &[("n1", &["a", "new_x"]), ("n2", &["new_x"])]);
    let all_source_indexes: BTreeSet<usize> = (0..batch.sources.len()).collect();
    let mapped_scope = engine
        .map_batch_into_hkt_scope(&batch, &all_source_indexes, &scope_snapshot)
        .unwrap();
    let update_stage = engine
        .compute_update_stage_for_scope(
            &batch,
            &scope_snapshot,
            &mapped_scope,
            &crate::engine::trigger::AncestorContext::default(),
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();

    let baseline_word_ids: BTreeSet<i32> = engine.baseline_word_legend.keys().copied().collect();
    let synthetic_word_ids: Vec<i32> = update_stage
        .updated_expected_word_ids
        .difference(&baseline_word_ids)
        .copied()
        .collect();
    assert_eq!(
        synthetic_word_ids.len(),
        1,
        "expected exactly one synthetic expected word"
    );
    let synthetic_word_id = synthetic_word_ids[0];

    let assigned_synthetic = update_stage
        .assigned_expected_words_by_node_id
        .values()
        .any(|word_ids| word_ids.contains(&synthetic_word_id));
    assert!(
        !assigned_synthetic,
        "expected synthetic word to remain unassigned when beta is strict"
    );

    let metrics = engine
        .compute_word_change_metrics_for_scope(
            &scope_snapshot,
            &mapped_scope,
            &update_stage,
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();
    let candidate = metrics.get(&synthetic_word_id).unwrap();
    assert!((candidate.strength - 2.0 / 3.0).abs() < 1e-6);
    assert!((candidate.importance - 2.0 / 5.0).abs() < 1e-6);
    assert_eq!(candidate.eligibility, 0.0);
}
#[test]
fn word_change_metrics_expected_only_word_includes_state0_parent_scope_counts() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha = 0.0;
    config.seca_thresholds.beta = 1.0; // keep expected word unassigned

    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(
            0,
            &[("s1", &["a", "legacy"]), ("s2", &["a", "legacy"])],
        ))
        .unwrap();

    let root_hkt_id = root_hkt_id(&engine);
    let mut scope_snapshot = engine.snapshot_hkt_scope(root_hkt_id).unwrap();

    let legacy_word_id = engine
        .baseline_word_legend
        .iter()
        .find(|(_, token)| token.as_str() == "legacy")
        .map(|(id, _)| *id)
        .unwrap();

    for word_ids in scope_snapshot.node_word_ids_by_node_id.values_mut() {
        word_ids.remove(&legacy_word_id);
    }
    for word_source_ids in scope_snapshot.node_word_source_ids_by_node_id.values_mut() {
        word_source_ids.remove(&legacy_word_id);
    }

    let batch = make_batch(1, &[("n1", &["a", "legacy"])]);
    let all_source_indexes: BTreeSet<usize> = (0..batch.sources.len()).collect();
    let mapped_scope = engine
        .map_batch_into_hkt_scope(&batch, &all_source_indexes, &scope_snapshot)
        .unwrap();
    let update_stage = engine
        .compute_update_stage_for_scope(
            &batch,
            &scope_snapshot,
            &mapped_scope,
            &crate::engine::trigger::AncestorContext::default(),
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();

    let assigned_legacy = update_stage
        .assigned_expected_words_by_node_id
        .values()
        .any(|word_ids| word_ids.contains(&legacy_word_id));
    assert!(
        !assigned_legacy,
        "expected legacy to remain unassigned when beta is strict"
    );

    let metrics = engine
        .compute_word_change_metrics_for_scope(
            &scope_snapshot,
            &mapped_scope,
            &update_stage,
            &engine.build_baseline_source_word_sets().unwrap(),
        )
        .unwrap();

    let metrics_legacy = metrics.get(&legacy_word_id).unwrap();

    assert!((metrics_legacy.importance - 0.5).abs() < 1e-6);
}
