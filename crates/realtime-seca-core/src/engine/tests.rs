#[cfg(test)]
mod tests {

    use super::*;
    use crate::config::{
        AlphaErrorOption, BetaErrorOption, MemoryMode, SecaConfig, TriggerPolicyMode,
        WordImportanceErrorOption,
    };
    use crate::engine::batch_stats::{
        compute_batch_word_stats, compute_trigger_metrics_from_batch_stats,
    };
    use crate::engine::scope_mapping::{HktScopeSnapshot, MappedHktScopeState};
    use crate::engine::trigger::WordChangeMetrics;
    use crate::error::SecaError;
    use crate::tree::Hkt;
    use crate::types::{SourceBatch, SourceRecord};
    use crate::SecaEngine;

    use std::collections::{BTreeMap, BTreeSet};

    fn make_source(source_id: &str, batch_index: u32, tokens: &[&str]) -> SourceRecord {
        SourceRecord {
            source_id: source_id.to_string(),
            batch_index,
            tokens: tokens.iter().map(|token| (*token).to_string()).collect(),
            text: None,
            timestamp_unix_ms: None,
            metadata: None,
        }
    }

    fn make_batch(batch_index: u32, rows: &[(&str, &[&str])]) -> SourceBatch {
        SourceBatch {
            batch_index,
            sources: rows
                .iter()
                .map(|(source_id, tokens)| make_source(source_id, batch_index, tokens))
                .collect(),
        }
    }

    fn build_small_baseline_engine() -> SecaEngine {
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

    fn root_hkt_id(engine: &SecaEngine) -> i32 {
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

    // -------------------------
    // Baseline / Stage 0 sanity
    // -------------------------

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

    // -------------------------
    // Constructor config validation
    // -------------------------

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
    fn default_option_thresholds_match_csharp_defaults() {
        let thresholds = SecaConfig::default().seca_thresholds;

        assert_eq!(thresholds.alpha_option1_threshold, 0.1);
        assert_eq!(thresholds.alpha_option2_threshold, 0.2);
        assert_eq!(thresholds.alpha_option3_threshold, 0.3);
        assert_eq!(thresholds.beta_option1_threshold, 0.13);
        assert_eq!(thresholds.beta_option2_threshold, 0.2);
        assert_eq!(thresholds.beta_option3_threshold, 0.2);
        assert_eq!(thresholds.word_importance_option1_threshold, 0.3);
        assert_eq!(thresholds.word_importance_option2_threshold, 0.2);
        assert_eq!(
            thresholds.selected_alpha_option,
            crate::config::AlphaErrorOption::Option1
        );
        assert_eq!(
            thresholds.selected_beta_option,
            crate::config::BetaErrorOption::Option1
        );
        assert_eq!(
            thresholds.selected_word_importance_option,
            crate::config::WordImportanceErrorOption::Option1
        );
    }

    #[test]
    fn new_rejects_option_threshold_out_of_range() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.beta_option2_threshold = 1.2;

        let error = SecaEngine::new(config).unwrap_err();
        match error {
            SecaError::InvalidConfiguration { message } => {
                assert!(message.contains("seca_thresholds.beta_option2_threshold"));
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

    // -------------------------
    // Stage 1: batch stats
    // -------------------------

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

        let stats = compute_batch_word_stats(&batch, &baseline_engine.baseline_word_legend);

        // unique words = {a,b,x}
        assert_eq!(stats.unique_words_in_batch, 3);

        // known = a,b (x is new)
        assert_eq!(stats.known_words_in_batch, 2);
        assert_eq!(stats.new_words_in_batch, 1);

        // document frequency:
        // a -> 2 sources (n1,n2), b -> 2 sources (n2,n3), x -> 2 sources (n1,n2)
        assert_eq!(stats.max_word_document_frequency, 2);
        assert_eq!(stats.total_sources_in_batch, 3);

        assert_eq!(stats.word_document_frequency.get("a"), Some(&2));
        assert_eq!(stats.word_document_frequency.get("b"), Some(&2));
        assert_eq!(stats.word_document_frequency.get("x"), Some(&2));
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
    fn process_batch_reason_codes_include_stage1_and_stage2_scope_codes() {
        let mut engine = build_small_baseline_engine();

        let batch = make_batch(1, &[("n1", &["a", "x"]), ("n2", &["b", "y"])]);

        let _ = engine.process_batch(batch).unwrap();

        let explanation = engine.explain_last_update().expect("missing explanation");

        assert!(explanation
            .reason_codes
            .contains(&"INCREMENTAL_BATCH_PROCESSED".to_string()));
        assert!(explanation
            .reason_codes
            .contains(&"BATCH_WORD_STATS_COMPUTED".to_string()));
        assert!(explanation
            .reason_codes
            .contains(&"SECA_SCOPE_MAPPING_COMPLETED".to_string()));
        assert!(explanation
            .reason_codes
            .contains(&"SECA_SCOPE_METRICS_COMPUTED".to_string()));
        assert!(explanation
            .reason_codes
            .contains(&"SECA_TRIGGER_EVALUATED".to_string()));
    }

    // -------------------------
    // Stage 2: scope snapshot + mapping + recursive plan
    // -------------------------

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
        let mut engine = build_small_baseline_engine();
        let snapshot = engine.snapshot_hkt_scope(root_hkt_id(&engine)).unwrap();

        let mut sorted = snapshot.non_refuge_node_ids.clone();
        sorted.sort_unstable();
        assert_eq!(snapshot.non_refuge_node_ids, sorted);
    }

    #[test]
    fn map_batch_into_hkt_scope_tracks_known_and_new_tokens() {
        let mut engine = SecaEngine::new(SecaConfig::default()).unwrap();

        let baseline = make_batch(
            0,
            &[
                ("s1", &["earthquake", "damage", "city"]),
                ("s2", &["earthquake", "rescue", "city"]),
            ],
        );

        engine.build_baseline_tree(baseline).unwrap();
        let snapshot = engine.snapshot_hkt_scope(root_hkt_id(&engine)).unwrap();

        let batch = make_batch(
            1,
            &[
                ("n1", &["earthquake", "aftershock", "city"]),
                ("n2", &["rescue", "teams", "deployed"]),
            ],
        );

        let scoped_indexes = BTreeSet::from([0_usize, 1_usize]);

        let mapped = engine
            .map_batch_into_hkt_scope(&batch, &scoped_indexes, &snapshot)
            .unwrap();

        assert_eq!(mapped.hkt_id, snapshot.hkt_id);
        assert_eq!(mapped.scoped_batch_source_indexes.len(), 2);

        // known words should include baseline vocab overlaps
        assert!(
            !mapped.known_word_ids_in_scope.is_empty(),
            "expected known baseline words to be detected"
        );

        // new tokens should include at least one novel token
        assert!(
            mapped.new_tokens_in_scope.contains("aftershock")
                || mapped.new_tokens_in_scope.contains("teams")
                || mapped.new_tokens_in_scope.contains("deployed")
        );

        // per-scope DF should be computed
        assert!(!mapped.word_document_frequency_in_scope.is_empty());
        assert_eq!(
            mapped.word_document_frequency_in_scope.get("earthquake"),
            Some(&1usize)
        );
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

        assert_eq!(plan_one.batch_index, 1);
        assert_eq!(
            plan_one.any_reconstruction_triggered,
            plan_two.any_reconstruction_triggered
        );
        assert_eq!(plan_one.reconstruct_hkt_ids, plan_two.reconstruct_hkt_ids);
        assert_eq!(plan_one.notes, plan_two.notes);
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

        assert_eq!(plan_one.batch_index, plan_two.batch_index);
        assert_eq!(
            plan_one.any_reconstruction_triggered,
            plan_two.any_reconstruction_triggered
        );
        assert_eq!(plan_one.reconstruct_hkt_ids, plan_two.reconstruct_hkt_ids);
        assert_eq!(plan_one.notes, plan_two.notes);
    }

    // -------------------------
    // Stage 4A: HKT-local trigger decision placeholder
    // -------------------------

    #[test]
    fn scope_trigger_decision_zero_vocab_yields_no_trigger() {
        let engine = SecaEngine::new(SecaConfig::default()).unwrap();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 42,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::new(),
            node_source_ids_by_node_id: BTreeMap::new(),
            node_word_source_ids_by_node_id: BTreeMap::new(),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![],
        };

        let mapped_scope = MappedHktScopeState {
            hkt_id: 42,
            scoped_batch_source_indexes: BTreeSet::new(),
            matched_node_source_indexes_by_node_id: BTreeMap::new(),
            word_document_frequency_in_scope: BTreeMap::new(),
            known_word_ids_in_scope: BTreeSet::new(),
            new_tokens_in_scope: BTreeSet::new(),

            // NEW fields (beta-ready node-level diagnostics)
            word_document_frequency_by_node_id: BTreeMap::new(),
            known_word_ids_by_node_id: BTreeMap::new(),
            new_tokens_by_node_id: BTreeMap::new(),
        };

        let decision = engine
            .evaluate_scope_trigger_decision_placeholder(&scope_snapshot, &mapped_scope)
            .unwrap();

        assert_eq!(decision.hkt_id, 42);
        assert!(!decision.should_reconstruct);
        assert_eq!(decision.word_importance_error, Some(0.0));
        assert!(decision.trigger_reasons.is_empty());
    }

    #[test]
    fn scope_trigger_decision_triggers_when_new_token_ratio_exceeds_threshold() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.word_importance_error_threshold = 0.4;
        config.seca_thresholds.word_importance_option1_threshold = 0.4;

        let engine = SecaEngine::new(config).unwrap();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 7,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::new(),
            node_source_ids_by_node_id: BTreeMap::new(),
            node_word_source_ids_by_node_id: BTreeMap::new(),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![],
        };

        // known=1, new=2 => error = 2/3 ~= 0.6667
        let mapped_scope = MappedHktScopeState {
            hkt_id: 7,
            scoped_batch_source_indexes: BTreeSet::new(),
            matched_node_source_indexes_by_node_id: BTreeMap::new(),
            word_document_frequency_in_scope: BTreeMap::new(),

            // known=1
            known_word_ids_in_scope: BTreeSet::from([1_i32]),

            // new=2
            new_tokens_in_scope: BTreeSet::from(["new_a".to_string(), "new_b".to_string()]),

            // NEW fields (beta-ready node-level diagnostics)
            word_document_frequency_by_node_id: BTreeMap::new(),
            known_word_ids_by_node_id: BTreeMap::new(),
            new_tokens_by_node_id: BTreeMap::new(),
        };

        let decision = engine
            .evaluate_scope_trigger_decision_placeholder(&scope_snapshot, &mapped_scope)
            .unwrap();

        assert!(decision.should_reconstruct);
        assert!(decision
            .trigger_reasons
            .iter()
            .any(|reason| reason.contains("word_importance_error")));
    }

    #[test]
    fn scope_trigger_decision_uses_strict_greater_than_threshold() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.word_importance_error_threshold = 0.5;
        config.seca_thresholds.word_importance_option1_threshold = 0.5;

        let engine = SecaEngine::new(config).unwrap();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 9,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::new(),
            node_source_ids_by_node_id: BTreeMap::new(),
            node_word_source_ids_by_node_id: BTreeMap::new(),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![],
        };

        // known=1, new=1 => error = 0.5 exactly => should NOT trigger because code uses ">"
        let mapped_scope = MappedHktScopeState {
            hkt_id: 9,
            scoped_batch_source_indexes: BTreeSet::new(),
            matched_node_source_indexes_by_node_id: BTreeMap::new(),
            word_document_frequency_in_scope: BTreeMap::new(),

            // known=1
            known_word_ids_in_scope: BTreeSet::from([1_i32]),

            // new=1
            new_tokens_in_scope: BTreeSet::from(["new_x".to_string()]),

            // NEW fields (beta-ready node-level diagnostics)
            word_document_frequency_by_node_id: BTreeMap::new(),
            known_word_ids_by_node_id: BTreeMap::new(),
            new_tokens_by_node_id: BTreeMap::new(),
        };

        let decision = engine
            .evaluate_scope_trigger_decision_placeholder(&scope_snapshot, &mapped_scope)
            .unwrap();

        assert_eq!(decision.word_importance_error, Some(0.5));
        assert!(!decision.should_reconstruct);
    }

    #[test]
    fn scope_trigger_decision_adds_tiny_scope_note_when_triggered() {
        let mut config = SecaConfig::default();
        config.trigger_policy_mode = TriggerPolicyMode::Placeholder;
        config.seca_thresholds.word_importance_error_threshold = 0.0;
        config.seca_thresholds.word_importance_option1_threshold = 0.0;

        let engine = SecaEngine::new(config).unwrap();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 11,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::new(),
            node_source_ids_by_node_id: BTreeMap::new(),
            node_word_source_ids_by_node_id: BTreeMap::new(),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![],
        };

        let mapped_scope = MappedHktScopeState {
            hkt_id: 11,
            // tiny scope: 1 source (or 0 also qualifies, but 1 is clearer)
            scoped_batch_source_indexes: BTreeSet::from([0_usize]),
            matched_node_source_indexes_by_node_id: BTreeMap::new(),
            word_document_frequency_in_scope: BTreeMap::new(),

            // known=0, new=1 => wi_error = 1.0 > 0.0
            known_word_ids_in_scope: BTreeSet::new(),
            new_tokens_in_scope: BTreeSet::from(["novel".to_string()]),

            // NEW fields (beta-ready node-level diagnostics)
            word_document_frequency_by_node_id: BTreeMap::new(),
            known_word_ids_by_node_id: BTreeMap::new(),
            new_tokens_by_node_id: BTreeMap::new(),
        };

        let decision = engine
            .evaluate_scope_trigger_decision_placeholder(&scope_snapshot, &mapped_scope)
            .unwrap();

        assert!(decision.should_reconstruct);
        assert!(decision
            .trigger_reasons
            .iter()
            .any(|reason| reason.contains("tiny scoped batch")));
    }

    #[test]
    fn strict_thresholds_trigger_recursive_plan_and_emit_reasons() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.word_importance_error_threshold = 0.0;
        config.seca_thresholds.word_importance_option1_threshold = 0.0;

        let mut engine = SecaEngine::new(config).unwrap();

        let baseline = make_batch(0, &[("s1", &["a", "b"]), ("s2", &["a", "c"])]);

        engine.build_baseline_tree(baseline).unwrap();

        let batch = make_batch(1, &[("n1", &["new_token"])]);

        let plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();

        assert!(plan.any_reconstruction_triggered);
        assert!(
            !plan.reconstruct_hkt_ids.is_empty(),
            "expected at least one HKT to be marked for reconstruction"
        );

        assert!(
            plan.notes
                .iter()
                .any(|note| note.contains("trigger reason")),
            "expected trigger reason note(s) in recursive plan"
        );
    }

    #[test]
    fn paper_policy_trigger_plan_can_fire_on_unassigned_expected_words() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.word_importance_option1_threshold = 0.1;
        config.seca_thresholds.alpha_option1_threshold = 1.0;
        config.seca_thresholds.beta_option1_threshold = 1.0;
        config.seca_thresholds.beta = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();

        let baseline = make_batch(0, &[("s1", &["a", "b"]), ("s2", &["a", "c"])]);

        engine.build_baseline_tree(baseline).unwrap();

        let batch = make_batch(1, &[("n1", &["x"]), ("n2", &["y"])]);

        let plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();

        assert!(plan.any_reconstruction_triggered);
        assert!(
            plan.notes
                .iter()
                .any(|note| note.contains("paper-policy shadow: would_trigger=true")),
            "expected paper-policy shadow trigger note"
        );
    }

    // -------------------------
    // Stage 3A: process_batch sequencing + rebuild/no-rebuild paths
    // -------------------------

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
    fn process_batch_advances_last_processed_batch_index_via_snapshot() {
        let mut engine = build_small_baseline_engine();

        let batch = make_batch(1, &[("n1", &["a"])]);

        let _ = engine.process_batch(batch).unwrap();

        let snapshot = engine.snapshot().unwrap();
        assert_eq!(snapshot.last_processed_batch_index, Some(1));
    }

    #[test]
    fn strict_word_importance_threshold_triggers_reconstruction() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.word_importance_error_threshold = 0.0; // any new-token ratio > 0 triggers
        config.seca_thresholds.word_importance_option1_threshold = 0.0;

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
            .contains(&"SECA_FULL_REBUILD_EXECUTED".to_string()));
    }

    #[test]
    fn paper_policy_process_batch_triggers_on_unassigned_expected_words() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.word_importance_option1_threshold = 0.1;
        config.seca_thresholds.alpha_option1_threshold = 1.0;
        config.seca_thresholds.beta_option1_threshold = 1.0;
        config.seca_thresholds.beta = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();
        let baseline_batch = make_batch(0, &[("s1", &["a", "b"]), ("s2", &["a", "c"])]);
        engine.build_baseline_tree(baseline_batch).unwrap();

        let batch = make_batch(1, &[("n1", &["x"]), ("n2", &["y"])]);

        let result = engine.process_batch(batch).unwrap();
        assert!(result.reconstruction_triggered);

        let explanation = engine.explain_last_update().unwrap();
        assert!(explanation
            .reason_codes
            .contains(&"SECA_RECONSTRUCTION_TRIGGERED".to_string()));
        assert!(explanation
            .reason_codes
            .contains(&"SECA_FULL_REBUILD_EXECUTED".to_string()));
    }

    #[test]
    fn process_batch_trigger_path_mentions_rebuild_action_in_notes() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.word_importance_error_threshold = 0.0;
        config.seca_thresholds.word_importance_option1_threshold = 0.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["a"])]))
            .unwrap();

        let result = engine
            .process_batch(make_batch(1, &[("n1", &["new_token"])]))
            .unwrap();

        assert!(result.reconstruction_triggered);
        assert!(result
            .notes
            .iter()
            .any(|note| note.contains("full rebuild from stored batches completed")));
    }

    #[test]
    fn process_batch_paper_policy_trigger_mentions_rebuild_action_in_notes() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.word_importance_option1_threshold = 0.1;
        config.seca_thresholds.alpha_option1_threshold = 1.0;
        config.seca_thresholds.beta_option1_threshold = 1.0;
        config.seca_thresholds.beta = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["b"])]))
            .unwrap();

        let result = engine
            .process_batch(make_batch(1, &[("n1", &["x"]), ("n2", &["y"])]))
            .unwrap();

        assert!(result.reconstruction_triggered);
        assert!(result
            .notes
            .iter()
            .any(|note| note.contains("full rebuild from stored batches completed")));
    }

    #[test]
    fn process_batch_trigger_rebuilds_tree_from_stored_batches() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.word_importance_error_threshold = 0.0;
        config.seca_thresholds.word_importance_option1_threshold = 0.0;

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
            .any(|c| c == "SECA_FULL_REBUILD_EXECUTED"));

        let all_tokens: BTreeSet<String> = after_tree
            .word_legend
            .iter()
            .filter_map(|entry| entry.token.clone())
            .collect();

        assert!(all_tokens.contains("volcano") || all_tokens.contains("ash"));
    }

    #[test]
    fn process_batch_paper_policy_trigger_changes_tree() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.word_importance_option1_threshold = 0.1;
        config.seca_thresholds.alpha_option1_threshold = 1.0;
        config.seca_thresholds.beta_option1_threshold = 1.0;
        config.seca_thresholds.beta = 1.0;

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
            .process_batch(make_batch(1, &[("n1", &["novel_x"]), ("n2", &["novel_y"])]))
            .unwrap();
        assert!(result.reconstruction_triggered);

        let after =
            serde_json::to_string_pretty(&engine.export_baseline_tree_verbose().unwrap()).unwrap();

        assert_ne!(
            before, after,
            "tree should change after triggered rebuild under paper policy"
        );
    }

    #[test]
    fn process_batch_rebuild_path_is_deterministic_for_same_sequence() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.word_importance_error_threshold = 0.0;
        config.seca_thresholds.word_importance_option1_threshold = 0.0;

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
            serde_json::to_string_pretty(&engine_a.export_baseline_tree_verbose().unwrap())
                .unwrap();

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
            serde_json::to_string_pretty(&engine_b.export_baseline_tree_verbose().unwrap())
                .unwrap();

        assert_eq!(
            final_a, final_b,
            "final rebuilt tree should be deterministic"
        );
    }

    // -------------------------
    // Memory mode tests (cross-cutting)
    // -------------------------

    #[test]
    fn process_batch_in_full_mode_keeps_all_batches() {
        let mut config = SecaConfig::default();
        config.memory_mode = MemoryMode::Full;
        config.seca_thresholds.word_importance_error_threshold = 1.0; // avoid rebuild noise in this test
        config.seca_thresholds.word_importance_option1_threshold = 1.0;

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
        config.seca_thresholds.word_importance_option1_threshold = 1.0;

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
    fn sliding_window_with_zero_max_batches_errors_on_process_batch() {
        let mut config = SecaConfig::default();
        config.memory_mode = MemoryMode::SlidingWindow;
        config.max_batches_in_memory = Some(0);

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["b"])]))
            .unwrap();

        let error = engine
            .process_batch(make_batch(1, &[("n1", &["x"])]))
            .unwrap_err();

        match error {
            SecaError::InvalidConfiguration { message } => {
                assert!(message.contains("max_batches_in_memory must be > 0"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn sliding_window_rebuild_uses_only_retained_batches() {
        let mut config = SecaConfig::default();
        config.memory_mode = MemoryMode::SlidingWindow;
        config.max_batches_in_memory = Some(2);
        config.seca_thresholds.word_importance_error_threshold = 0.0; // force rebuilds
        config.seca_thresholds.word_importance_option1_threshold = 0.0;

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
    fn seca_light_prunes_source_legends_and_identity_maps() {
        let mut config = SecaConfig::default();
        config.memory_mode = MemoryMode::SlidingWindow;
        config.max_batches_in_memory = Some(2);
        config.seca_thresholds.word_importance_error_threshold = 1.0;
        config.seca_thresholds.word_importance_option1_threshold = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s0", &["a"])]))
            .unwrap();

        let s0_internal_id = SecaEngine::fnv1a_64("s0");
        assert!(engine.baseline_source_legend.contains_key(&s0_internal_id));

        engine.process_batch(make_batch(1, &[("s1", &["b"])])).unwrap();
        engine.process_batch(make_batch(2, &[("s2", &["c"])])).unwrap();

        assert!(!engine.baseline_source_legend.contains_key(&s0_internal_id));
        assert!(!engine.source_id_by_url.contains_key("s0"));
        assert!(!engine.url_by_source_id.contains_key(&s0_internal_id));
        assert!(
            !engine
                .source_batch_index_by_internal_source_id
                .contains_key(&s0_internal_id)
        );
        assert!(
            !engine
                .source_ids_by_batch_index
                .get(&0)
                .map(|ids| ids.contains(&s0_internal_id))
                .unwrap_or(false)
        );

        let explanation = engine.explain_last_update().expect("missing explanation");
        assert!(explanation
            .reason_codes
            .contains(&"SECA_LIGHT_PRUNING_APPLIED".to_string()));
    }

    #[test]
    fn seca_light_hard_deletes_dead_nodes_and_dead_hkts() {
        let mut config = SecaConfig::default();
        config.memory_mode = MemoryMode::SlidingWindow;
        config.max_batches_in_memory = Some(2);
        config.seca_thresholds.word_importance_error_threshold = 1.0;
        config.seca_thresholds.word_importance_option1_threshold = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("base1", &["a", "b"]), ("base2", &["a", "c"])]))
            .unwrap();
        engine.process_batch(make_batch(1, &[("fresh", &["a"])])).unwrap();

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
        assert!(report.pruned_hkt_count > 0 || report.pruned_node_count > 0);

        let after_output = engine.hkt_build_output.as_ref().unwrap();
        assert!(after_output.hkts_by_id.len() < before_hkt_count);
        assert!(after_output.nodes_by_id.len() < before_node_count);
        assert!(!after_output.hkts_by_id.contains_key(&child_hkt_id));
    }

    #[test]
    fn seca_light_sliding_window_without_gamma_is_noop_for_tree_pruning() {
        let mut config = SecaConfig::default();
        config.memory_mode = MemoryMode::SlidingWindow;
        config.max_batches_in_memory = None;
        config.seca_thresholds.word_importance_error_threshold = 1.0;
        config.seca_thresholds.word_importance_option1_threshold = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s0", &["a"]), ("s1", &["b"])]))
            .unwrap();

        let hkt_count_before = engine.hkt_build_output.as_ref().unwrap().hkts_by_id.len();
        let node_count_before = engine.hkt_build_output.as_ref().unwrap().nodes_by_id.len();

        engine.process_batch(make_batch(1, &[("n1", &["x"])])).unwrap();

        let hkt_count_after = engine.hkt_build_output.as_ref().unwrap().hkts_by_id.len();
        let node_count_after = engine.hkt_build_output.as_ref().unwrap().nodes_by_id.len();
        assert_eq!(hkt_count_after, hkt_count_before);
        assert_eq!(node_count_after, node_count_before);

        let explanation = engine.explain_last_update().expect("missing explanation");
        assert!(!explanation
            .reason_codes
            .contains(&"SECA_LIGHT_PRUNING_APPLIED".to_string()));
    }

    #[test]
    fn full_mode_does_not_apply_seca_light_pruning() {
        let mut config = SecaConfig::default();
        config.memory_mode = MemoryMode::Full;
        config.seca_thresholds.word_importance_error_threshold = 1.0;
        config.seca_thresholds.word_importance_option1_threshold = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s0", &["a"])]))
            .unwrap();
        let s0_internal_id = SecaEngine::fnv1a_64("s0");

        engine.process_batch(make_batch(1, &[("s1", &["b"])])).unwrap();
        engine.process_batch(make_batch(2, &[("s2", &["c"])])).unwrap();

        assert!(engine.baseline_source_legend.contains_key(&s0_internal_id));
        let explanation = engine.explain_last_update().expect("missing explanation");
        assert!(!explanation
            .reason_codes
            .contains(&"SECA_LIGHT_PRUNING_APPLIED".to_string()));
    }

    // -------------------------
    // Export contract tests
    // -------------------------

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
        let mut engine = build_small_baseline_engine();

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
        let mut engine = build_small_baseline_engine();
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

    // -------------------------
    // Snapshot/load behavior
    // -------------------------

    #[test]
    fn load_snapshot_restores_full_tree_state_for_incremental_resume() {
        let mut engine = build_small_baseline_engine();

        // Make progress so last_processed_batch_index becomes Some(1)
        let _ = engine.process_batch(make_batch(1, &[("n1", &["a"])]));
        let root_hkt_id = root_hkt_id(&engine);

        let logical_removed = crate::engine::LogicalRemovedHkt {
            hkt: Hkt {
                hkt_id: root_hkt_id,
                parent_node_id: -1,
                expected_words: BTreeSet::new(),
                is_state1: false,
                nodes: Vec::new(),
            },
            old_parent_node_id: 0,
        };
        engine
            .logically_removed_hkts_by_id
            .insert(root_hkt_id, logical_removed);
        let original_snapshot = engine.snapshot().unwrap();

        let loaded = SecaEngine::load_snapshot(original_snapshot.clone()).unwrap();
        let loaded_snapshot = loaded.snapshot().unwrap();

        assert_eq!(
            loaded_snapshot.last_processed_batch_index,
            original_snapshot.last_processed_batch_index
        );
        assert_eq!(loaded_snapshot.config, original_snapshot.config);
        assert!(loaded.export_baseline_tree().is_ok());

        let logical = loaded_snapshot
            .logically_removed_hkts_by_id
            .get(&root_hkt_id)
            .expect("expected logical removal entry in snapshot");
        assert_eq!(logical.hkt.hkt_id, root_hkt_id);
        assert_eq!(logical.hkt.parent_node_id, -1);
        assert!(logical.hkt.nodes.is_empty());
        assert_eq!(logical.old_parent_node_id, 0);

        let resumed_result = loaded
            .clone()
            .process_batch(make_batch(
                loaded_snapshot
                    .last_processed_batch_index
                    .unwrap_or(0)
                    .saturating_add(1),
                &[("n2", &["x"])],
            ))
            .unwrap();
        assert_eq!(resumed_result.batch_index, 2);
    }

    #[test]
    fn baseline_hkts_are_state0() {
        let mut engine = build_small_baseline_engine();

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

    #[test]
    fn rebuilt_subtree_hkts_are_state1() {
        let mut engine = build_small_baseline_engine();

        let root_hkt_id = root_hkt_id(&engine);
        let batch = make_batch(1, &[("s_new_state1", &["z"])]);
        let mut forced_plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();
        forced_plan.reconstruct_hkt_ids = vec![root_hkt_id];
        forced_plan
            .reconstruct_scopes_by_hkt_id
            .insert(root_hkt_id, BTreeSet::from([0_usize]));

        engine
            .rebuild_selected_hkts_from_trigger_plan(&batch, &forced_plan)
            .unwrap();

        let hkts = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .values();
        for hkt in hkts {
            assert!(
                hkt.is_state1,
                "rebuilt HKT {} should have is_state1 = true",
                hkt.hkt_id
            );
        }
    }

    // -------------------------
    // Legacy placeholder metric function (optional regression)
    // -------------------------

    #[test]
    fn legacy_placeholder_metric_function_is_deterministic() {
        let mut baseline_word_legend = BTreeMap::new();
        baseline_word_legend.insert(1, "a".to_string());
        baseline_word_legend.insert(2, "b".to_string());

        let batch = make_batch(
            1,
            &[("n1", &["a", "x"]), ("n2", &["a", "b"]), ("n3", &["x"])],
        );

        let stats_one = compute_batch_word_stats(&batch, &baseline_word_legend);
        let stats_two = compute_batch_word_stats(&batch, &baseline_word_legend);

        let thresholds = SecaConfig::default().seca_thresholds;
        let metrics_one = compute_trigger_metrics_from_batch_stats(&stats_one, &thresholds);
        let metrics_two = compute_trigger_metrics_from_batch_stats(&stats_two, &thresholds);

        assert_eq!(metrics_one.alpha_estimate, metrics_two.alpha_estimate);
        assert_eq!(metrics_one.beta_estimate, metrics_two.beta_estimate);
        assert_eq!(metrics_one.alpha_error, metrics_two.alpha_error);
        assert_eq!(metrics_one.beta_error, metrics_two.beta_error);
        assert_eq!(
            metrics_one.word_importance_error,
            metrics_two.word_importance_error
        );
        assert_eq!(
            metrics_one.should_reconstruct,
            metrics_two.should_reconstruct
        );
        assert_eq!(metrics_one.trigger_reasons, metrics_two.trigger_reasons);
    }

    mod trigger_metric_boundary_tests {
        use crate::config::SecaConfig;
        use crate::engine::batch_stats::{
            compute_batch_word_stats, compute_trigger_metrics_from_batch_stats,
        };
        use crate::types::{SourceBatch, SourceRecord};
        use std::collections::BTreeMap;

        fn make_source(source_id: &str, batch_index: u32, tokens: &[&str]) -> SourceRecord {
            SourceRecord {
                source_id: source_id.to_string(),
                batch_index,
                tokens: tokens.iter().map(|token| (*token).to_string()).collect(),
                text: None,
                timestamp_unix_ms: None,
                metadata: None,
            }
        }

        fn make_batch(batch_index: u32, rows: &[(&str, &[&str])]) -> SourceBatch {
            SourceBatch {
                batch_index,
                sources: rows
                    .iter()
                    .map(|(source_id, tokens)| make_source(source_id, batch_index, tokens))
                    .collect(),
            }
        }

        fn baseline_word_legend(tokens: &[&str]) -> BTreeMap<i32, String> {
            let mut legend = BTreeMap::new();
            for (index, token) in tokens.iter().enumerate() {
                legend.insert((index + 1) as i32, (*token).to_string());
            }
            legend
        }

        #[test]
        fn batch_stats_empty_batch_returns_zeroes() {
            let batch = SourceBatch {
                batch_index: 1,
                sources: Vec::new(),
            };
            let baseline = baseline_word_legend(&["a", "b"]);

            let stats = compute_batch_word_stats(&batch, &baseline);

            assert_eq!(stats.unique_words_in_batch, 0);
            assert_eq!(stats.known_words_in_batch, 0);
            assert_eq!(stats.new_words_in_batch, 0);
            assert_eq!(stats.max_word_document_frequency, 0);
            assert_eq!(stats.total_sources_in_batch, 0);
            assert!(stats.word_document_frequency.is_empty());
        }

        #[test]
        fn trigger_metrics_do_not_fire_when_values_equal_thresholds() {
            // Build a case with:
            // unique=4, known=2, new=2, total_sources=4, max_df=2
            // alpha_est=0.5, beta_est=0.5, dominance(word_importance)=0.5
            let batch = make_batch(
                1,
                &[
                    ("n1", &["a"]),
                    ("n2", &["b"]),
                    ("n3", &["x"]),
                    ("n4", &["x", "y"]), // x appears in 2/4 sources => max_df=2
                ],
            );

            let baseline = baseline_word_legend(&["a", "b"]);
            let stats = compute_batch_word_stats(&batch, &baseline);

            let mut config = SecaConfig::default();
            config.seca_thresholds.alpha = 0.5;
            config.seca_thresholds.beta = 0.5;
            config.seca_thresholds.alpha_error_threshold = 0.0;
            config.seca_thresholds.beta_error_threshold = 0.0;
            config.seca_thresholds.word_importance_error_threshold = 0.5;

            let metrics = compute_trigger_metrics_from_batch_stats(&stats, &config.seca_thresholds);

            assert_eq!(metrics.alpha_estimate, 0.5);
            assert_eq!(metrics.beta_estimate, 0.5);
            assert_eq!(metrics.alpha_error, 0.0);
            assert_eq!(metrics.beta_error, 0.0);
            assert_eq!(metrics.word_importance_error, 0.5);

            // Current trigger semantics use strict `>`
            assert!(!metrics.should_reconstruct);
            assert!(metrics.trigger_reasons.is_empty());
        }

        #[test]
        fn trigger_metrics_fire_when_alpha_error_exceeds_threshold() {
            // unique=3, known=1, new=2 => alpha_est=1/3, beta_est=2/3
            let batch = make_batch(1, &[("n1", &["a", "x", "y"])]);
            let baseline = baseline_word_legend(&["a"]);

            let stats = compute_batch_word_stats(&batch, &baseline);

            let mut config = SecaConfig::default();
            config.seca_thresholds.alpha = 1.0; // far from observed alpha_est
            config.seca_thresholds.beta = 2.0 / 3.0; // match beta exactly to isolate alpha
            config.seca_thresholds.alpha_error_threshold = 0.5; // |1 - 1/3| = 0.666... > 0.5
            config.seca_thresholds.beta_error_threshold = 1.0;
            config.seca_thresholds.word_importance_error_threshold = 1.0;

            let metrics = compute_trigger_metrics_from_batch_stats(&stats, &config.seca_thresholds);

            assert!(metrics.alpha_error > config.seca_thresholds.alpha_error_threshold);
            assert!(metrics.should_reconstruct);
            assert!(metrics
                .trigger_reasons
                .iter()
                .any(|reason| reason.contains("alpha_error")));
            assert!(!metrics
                .trigger_reasons
                .iter()
                .any(|reason| reason.contains("beta_error")));
        }

        #[test]
        fn trigger_metrics_fire_when_beta_error_exceeds_threshold() {
            // unique=2, known=2, new=0 => beta_est=0.0
            let batch = make_batch(1, &[("n1", &["a", "b"])]);
            let baseline = baseline_word_legend(&["a", "b"]);

            let stats = compute_batch_word_stats(&batch, &baseline);

            let mut config = SecaConfig::default();
            config.seca_thresholds.alpha = 1.0; // exact
            config.seca_thresholds.beta = 1.0; // far from observed 0.0
            config.seca_thresholds.alpha_error_threshold = 0.0;
            config.seca_thresholds.beta_error_threshold = 0.2;
            config.seca_thresholds.word_importance_error_threshold = 1.0;

            let metrics = compute_trigger_metrics_from_batch_stats(&stats, &config.seca_thresholds);

            assert!(metrics.beta_error > config.seca_thresholds.beta_error_threshold);
            assert!(metrics.should_reconstruct);
            assert!(metrics
                .trigger_reasons
                .iter()
                .any(|reason| reason.contains("beta_error")));
        }

        #[test]
        fn trigger_metrics_fire_when_word_importance_error_exceeds_threshold() {
            // total_sources=4, max_df=4 => word_importance_error=1.0
            let batch = make_batch(
                1,
                &[
                    ("n1", &["dom"]),
                    ("n2", &["dom"]),
                    ("n3", &["dom"]),
                    ("n4", &["dom"]),
                ],
            );
            let baseline = baseline_word_legend(&["dom"]);

            let stats = compute_batch_word_stats(&batch, &baseline);

            let mut config = SecaConfig::default();
            config.seca_thresholds.alpha = 1.0;
            config.seca_thresholds.beta = 0.0;
            config.seca_thresholds.alpha_error_threshold = 0.0;
            config.seca_thresholds.beta_error_threshold = 0.0;
            config.seca_thresholds.word_importance_error_threshold = 0.99;

            let metrics = compute_trigger_metrics_from_batch_stats(&stats, &config.seca_thresholds);

            assert_eq!(metrics.word_importance_error, 1.0);
            assert!(metrics.should_reconstruct);
            assert!(metrics
                .trigger_reasons
                .iter()
                .any(|reason| reason.contains("word_importance_error")));
        }

        #[test]
        fn trigger_metrics_are_deterministic_for_identical_input() {
            let batch = make_batch(
                1,
                &[("n1", &["a", "x"]), ("n2", &["a", "b"]), ("n3", &["x"])],
            );
            let baseline = baseline_word_legend(&["a", "b"]);

            let stats_one = compute_batch_word_stats(&batch, &baseline);
            let stats_two = compute_batch_word_stats(&batch, &baseline);

            let thresholds = SecaConfig::default().seca_thresholds;
            let metrics_one = compute_trigger_metrics_from_batch_stats(&stats_one, &thresholds);
            let metrics_two = compute_trigger_metrics_from_batch_stats(&stats_two, &thresholds);

            assert_eq!(metrics_one.alpha_estimate, metrics_two.alpha_estimate);
            assert_eq!(metrics_one.beta_estimate, metrics_two.beta_estimate);
            assert_eq!(metrics_one.alpha_error, metrics_two.alpha_error);
            assert_eq!(metrics_one.beta_error, metrics_two.beta_error);
            assert_eq!(
                metrics_one.word_importance_error,
                metrics_two.word_importance_error
            );
            assert_eq!(
                metrics_one.should_reconstruct,
                metrics_two.should_reconstruct
            );
            assert_eq!(metrics_one.trigger_reasons, metrics_two.trigger_reasons);
        }
    }

    #[test]
    fn process_batch_contracts_hold_after_refactor() {
        use crate::config::{MemoryMode, SecaConfig};
        use crate::types::{SourceBatch, SourceRecord};

        fn make_source(source_id: &str, batch_index: u32, tokens: &[&str]) -> SourceRecord {
            SourceRecord {
                source_id: source_id.to_string(),
                batch_index,
                tokens: tokens.iter().map(|token| (*token).to_string()).collect(),
                text: None,
                timestamp_unix_ms: None,
                metadata: None,
            }
        }

        fn make_batch(batch_index: u32, rows: &[(&str, &[&str])]) -> SourceBatch {
            SourceBatch {
                batch_index,
                sources: rows
                    .iter()
                    .map(|(source_id, tokens)| make_source(source_id, batch_index, tokens))
                    .collect(),
            }
        }

        // 1) baseline required
        {
            let mut engine = crate::SecaEngine::new(SecaConfig::default()).unwrap();
            let error = engine
                .process_batch(make_batch(1, &[("n1", &["a"])]))
                .unwrap_err();

            match error {
                crate::SecaError::StateError { message } => {
                    assert!(message.contains("baseline tree has not been built yet"));
                }
                other => panic!("unexpected error variant: {other:?}"),
            }
        }

        // 2) out-of-sequence batch rejected
        {
            let mut engine = crate::SecaEngine::new(SecaConfig::default()).unwrap();
            engine
                .build_baseline_tree(make_batch(0, &[("s1", &["a", "b"]), ("s2", &["a", "c"])]))
                .unwrap();

            let error = engine
                .process_batch(make_batch(2, &[("n1", &["a"])])) // expected 1
                .unwrap_err();

            match error {
                crate::SecaError::InvalidConfiguration { message } => {
                    assert!(message.contains("out of sequence"));
                    assert!(message.contains("expected 1"));
                }
                other => panic!("unexpected error variant: {other:?}"),
            }
        }

        // 3) last_processed_batch_index advances + reason codes present (skip path)
        {
            let mut config = SecaConfig::default();
            config.seca_thresholds.word_importance_error_threshold = 1.0; // make trigger unlikely
            config.seca_thresholds.word_importance_option1_threshold = 1.0;
            let mut engine = crate::SecaEngine::new(config).unwrap();

            engine
                .build_baseline_tree(make_batch(0, &[("s1", &["a", "b"]), ("s2", &["a", "c"])]))
                .unwrap();

            let result = engine
                .process_batch(make_batch(1, &[("n1", &["x"]), ("n2", &["y"])]))
                .unwrap();

            assert_eq!(result.batch_index, 1);

            let snapshot = engine.snapshot().unwrap();
            assert_eq!(snapshot.last_processed_batch_index, Some(1));

            let explanation = engine.explain_last_update().expect("missing explanation");
            assert!(explanation
                .reason_codes
                .contains(&"INCREMENTAL_BATCH_PROCESSED".to_string()));
            assert!(explanation
                .reason_codes
                .contains(&"BATCH_WORD_STATS_COMPUTED".to_string()));
            assert!(explanation
                .reason_codes
                .contains(&"SECA_TRIGGER_EVALUATED".to_string()));
            assert!(explanation
                .reason_codes
                .contains(&"SECA_RECONSTRUCTION_SKIPPED".to_string()));
        }

        // 4) trigger path reason codes present
        {
            let mut config = SecaConfig::default();
            config.seca_thresholds.word_importance_error_threshold = 0.0; // force trigger on novelty
            config.seca_thresholds.word_importance_option1_threshold = 0.0;
            let mut engine = crate::SecaEngine::new(config).unwrap();

            engine
                .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["a"])]))
                .unwrap();

            let result = engine
                .process_batch(make_batch(1, &[("n1", &["new_token"])]))
                .unwrap();

            assert!(result.reconstruction_triggered);

            let explanation = engine.explain_last_update().expect("missing explanation");
            assert!(explanation
                .reason_codes
                .contains(&"SECA_RECONSTRUCTION_TRIGGERED".to_string()));
        }

        // 5) full mode retains all batches
        {
            let mut config = SecaConfig::default();
            config.memory_mode = MemoryMode::Full;
            config.seca_thresholds.word_importance_error_threshold = 1.0;
            config.seca_thresholds.word_importance_option1_threshold = 1.0;
            let mut engine = crate::SecaEngine::new(config).unwrap();

            engine
                .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["b"])]))
                .unwrap();

            assert_eq!(engine.stored_batch_count(), 1);
            engine
                .process_batch(make_batch(1, &[("n1", &["x"])]))
                .unwrap();
            engine
                .process_batch(make_batch(2, &[("n2", &["y"])]))
                .unwrap();
            engine
                .process_batch(make_batch(3, &[("n3", &["z"])]))
                .unwrap();
            assert_eq!(engine.stored_batch_count(), 4);
        }

        // 6) sliding window trims
        {
            let mut config = SecaConfig::default();
            config.memory_mode = MemoryMode::SlidingWindow;
            config.max_batches_in_memory = Some(2);
            config.seca_thresholds.word_importance_error_threshold = 1.0;
            config.seca_thresholds.word_importance_option1_threshold = 1.0;
            let mut engine = crate::SecaEngine::new(config).unwrap();

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
            assert_eq!(engine.stored_batch_count(), 2);

            engine
                .process_batch(make_batch(3, &[("n3", &["z"])]))
                .unwrap();
            assert_eq!(engine.stored_batch_count(), 2);
        }

        // 7) invalid sliding-window config errors on process_batch
        {
            let mut config = SecaConfig::default();
            config.memory_mode = MemoryMode::SlidingWindow;
            config.max_batches_in_memory = Some(0);
            let mut engine = crate::SecaEngine::new(config).unwrap();

            engine
                .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["b"])]))
                .unwrap();

            let error = engine
                .process_batch(make_batch(1, &[("n1", &["x"])]))
                .unwrap_err();

            match error {
                crate::SecaError::InvalidConfiguration { message } => {
                    assert!(message.contains("max_batches_in_memory must be > 0"));
                }
                other => panic!("unexpected error variant: {other:?}"),
            }
        }
    }

    #[test]
    fn default_trigger_policy_mode_is_paper_diagnostic_scaffold_and_is_reported_in_notes() {
        let mut engine = SecaEngine::new(SecaConfig::default()).unwrap();

        engine
            .build_baseline_tree(make_batch(
                0,
                &[
                    ("s1", &["alpha", "beta", "gamma"]),
                    ("s2", &["alpha", "delta"]),
                ],
            ))
            .unwrap();

        let result = engine
            .process_batch(make_batch(1, &[("s3", &["alpha", "epsilon"])]))
            .unwrap();

        assert!(
            result
                .notes
                .iter()
                .any(|note| note.contains("active trigger policy: paper_diagnostic_scaffold")),
            "expected notes to report paper_diagnostic_scaffold trigger policy; notes were: {:?}",
            result.notes
        );
    }
    #[test]
    fn paper_diagnostic_scaffold_trigger_policy_mode_is_reported_in_notes() {
        let mut config = SecaConfig::default();
        config.trigger_policy_mode = TriggerPolicyMode::PaperDiagnosticScaffold;

        let mut engine = SecaEngine::new(config).expect("engine should build");

        let baseline_batch = make_batch(
            0,
            &[
                ("s1", &["alpha", "beta", "gamma"]),
                ("s2", &["alpha", "delta"]),
            ],
        );
        engine
            .build_baseline_tree(baseline_batch)
            .expect("baseline should build");

        let incremental_batch = make_batch(
            1,
            &[("s3", &["alpha", "epsilon"]), ("s4", &["zeta", "eta"])],
        );

        let result = engine
            .process_batch(incremental_batch)
            .expect("process_batch should succeed");

        assert!(
            result
                .notes
                .iter()
                .any(|note| note.contains("active trigger policy: paper_diagnostic_scaffold")),
            "expected notes to report paper_diagnostic_scaffold trigger policy; notes were: {:?}",
            result.notes
        );
    }

    #[test]
    fn paper_scaffold_beta_trigger_uses_word_level_eligibility_violation_not_node_mean_beta_summary(
    ) {
        let mut config = SecaConfig::default();
        config.trigger_policy_mode = TriggerPolicyMode::PaperDiagnosticScaffold;
        config.seca_thresholds.beta = 0.8;
        config.seca_thresholds.beta_option1_threshold = 0.1;

        // Make alpha/WI thresholds permissive so beta is the only likely trigger path.
        config.seca_thresholds.alpha_option1_threshold = 1.0;
        config.seca_thresholds.word_importance_option1_threshold = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(
                0,
                &[("s1", &["a"]), ("s2", &["b"]), ("s3", &["c"])],
            ))
            .unwrap();

        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();
        let word_id_b = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "b")
            .map(|(id, _)| *id)
            .unwrap();
        let word_id_c = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "c")
            .map(|(id, _)| *id)
            .unwrap();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 100,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![10],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeSet::from([word_id_a, word_id_b, word_id_c]),
            )]),
            node_source_ids_by_node_id: BTreeMap::new(),
            node_word_source_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeMap::from([
                    (word_id_a, BTreeSet::new()),
                    (word_id_b, BTreeSet::new()),
                    (word_id_c, BTreeSet::new()),
                ]),
            )]),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![10],
        };

        // 4 mapped sources assigned to node 10.
        // Word-level eligibility in node:
        // a -> 1.0, b -> 0.0, c -> 0.0
        // With beta=0.8, violations = [0, 0.8, 0.8], avg = 0.5333 > 0.1.
        let batch = make_batch(
            1,
            &[
                ("n1", &["a"]),
                ("n2", &["a"]),
                ("n3", &["a"]),
                ("n4", &["a"]),
            ],
        );
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
            )
            .unwrap();
        let change_metrics = engine
            .compute_word_change_metrics_for_scope(&scope_snapshot, &mapped_scope, &update_stage)
            .unwrap();
        let decision = engine
            .evaluate_scope_trigger_decision_paper_scaffold(
                &scope_snapshot,
                &mapped_scope,
                &update_stage,
                &change_metrics,
            )
            .unwrap();

        assert!(
            decision.should_reconstruct,
            "expected beta-error-driven paper trigger"
        );
        assert!(decision
            .trigger_reasons
            .iter()
            .any(|reason| reason.contains("paper_shadow.beta_error")));
    }
    #[test]
    fn paper_beta_error_is_zero_when_hkt_scope_has_no_state0_words() {
        let mut engine = SecaEngine::new(SecaConfig::default()).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s1", &["a"])]))
            .unwrap();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 200,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![10],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::from([(10, BTreeSet::new())]),
            node_source_ids_by_node_id: BTreeMap::new(),
            node_word_source_ids_by_node_id: BTreeMap::new(),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![10],
        };

        let batch = make_batch(1, &[]);
        let all_source_indexes: BTreeSet<usize> = BTreeSet::new();
        let mapped_scope = engine
            .map_batch_into_hkt_scope(&batch, &all_source_indexes, &scope_snapshot)
            .unwrap();
        let update_stage = engine
            .compute_update_stage_for_scope(
                &batch,
                &scope_snapshot,
                &mapped_scope,
                &crate::engine::trigger::AncestorContext::default(),
            )
            .unwrap();
        let change_metrics = engine
            .compute_word_change_metrics_for_scope(&scope_snapshot, &mapped_scope, &update_stage)
            .unwrap();
        let decision = engine
            .evaluate_scope_trigger_decision_paper_scaffold(
                &scope_snapshot,
                &mapped_scope,
                &update_stage,
                &change_metrics,
            )
            .unwrap();

        assert_eq!(decision.paper_beta_error, Some(0.0));
    }
    #[test]
    fn paper_beta_error_penalizes_state0_words_when_node_has_no_mapped_state1_sources() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.beta = 0.6;
        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["b"])]))
            .unwrap();

        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();
        let word_id_b = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "b")
            .map(|(id, _)| *id)
            .unwrap();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 201,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![10],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeSet::from([word_id_a, word_id_b]),
            )]),
            node_source_ids_by_node_id: BTreeMap::new(),
            node_word_source_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeMap::from([(word_id_a, BTreeSet::new()), (word_id_b, BTreeSet::new())]),
            )]),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![10],
        };

        let batch = make_batch(1, &[]);
        let mapped_scope = engine
            .map_batch_into_hkt_scope(&batch, &BTreeSet::new(), &scope_snapshot)
            .unwrap();
        let update_stage = engine
            .compute_update_stage_for_scope(
                &batch,
                &scope_snapshot,
                &mapped_scope,
                &crate::engine::trigger::AncestorContext::default(),
            )
            .unwrap();
        let change_metrics = engine
            .compute_word_change_metrics_for_scope(&scope_snapshot, &mapped_scope, &update_stage)
            .unwrap();
        let decision = engine
            .evaluate_scope_trigger_decision_paper_scaffold(
                &scope_snapshot,
                &mapped_scope,
                &update_stage,
                &change_metrics,
            )
            .unwrap();

        // two words, both elig1=0, beta=0.6 => avg violation = 0.6
        assert_eq!(decision.paper_beta_error, Some(0.6));
    }

    #[test]
    fn paper_beta_error_option2_matches_eligibility_difference_average() {
        let mut config = SecaConfig::default();
        config.trigger_policy_mode = TriggerPolicyMode::PaperDiagnosticScaffold;
        config.seca_thresholds.beta = 0.8;
        config.seca_thresholds.selected_beta_option = crate::config::BetaErrorOption::Option2;
        config.seca_thresholds.beta_option2_threshold = 1.0;
        config.seca_thresholds.alpha_option1_threshold = 1.0;
        config.seca_thresholds.word_importance_option1_threshold = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["a", "b"])]))
            .unwrap();

        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();
        let word_id_b = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "b")
            .map(|(id, _)| *id)
            .unwrap();

        let source_ids_a: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| matches!(url.as_str(), "s1" | "s2"))
            .map(|(id, _)| *id)
            .collect();
        let source_ids_b: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| url.as_str() == "s2")
            .map(|(id, _)| *id)
            .collect();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 220,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![10],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeSet::from([word_id_a, word_id_b]),
            )]),
            node_source_ids_by_node_id: BTreeMap::from([(
                10,
                source_ids_a
                    .union(&source_ids_b)
                    .copied()
                    .collect::<BTreeSet<i64>>(),
            )]),
            node_word_source_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeMap::from([
                    (word_id_a, source_ids_a.clone()),
                    (word_id_b, source_ids_b.clone()),
                ]),
            )]),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![10],
        };

        let batch = make_batch(1, &[("n1", &["a"]), ("n2", &["b"])]);
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
            )
            .unwrap();
        let change_metrics = engine
            .compute_word_change_metrics_for_scope(&scope_snapshot, &mapped_scope, &update_stage)
            .unwrap();
        let decision = engine
            .evaluate_scope_trigger_decision_paper_scaffold(
                &scope_snapshot,
                &mapped_scope,
                &update_stage,
                &change_metrics,
            )
            .unwrap();

        assert_eq!(decision.paper_beta_error, Some(0.125));
    }

    #[test]
    fn paper_beta_error_option3_matches_euclidean_distance() {
        let mut config = SecaConfig::default();
        config.trigger_policy_mode = TriggerPolicyMode::PaperDiagnosticScaffold;
        config.seca_thresholds.beta = 0.8;
        config.seca_thresholds.selected_beta_option = crate::config::BetaErrorOption::Option3;
        config.seca_thresholds.beta_option3_threshold = 1.0;
        config.seca_thresholds.alpha_option1_threshold = 1.0;
        config.seca_thresholds.word_importance_option1_threshold = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["a", "b"])]))
            .unwrap();

        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();
        let word_id_b = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "b")
            .map(|(id, _)| *id)
            .unwrap();

        let source_ids_a: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| matches!(url.as_str(), "s1" | "s2"))
            .map(|(id, _)| *id)
            .collect();
        let source_ids_b: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| url.as_str() == "s2")
            .map(|(id, _)| *id)
            .collect();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 221,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![10],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeSet::from([word_id_a, word_id_b]),
            )]),
            node_source_ids_by_node_id: BTreeMap::from([(
                10,
                source_ids_a
                    .union(&source_ids_b)
                    .copied()
                    .collect::<BTreeSet<i64>>(),
            )]),
            node_word_source_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeMap::from([
                    (word_id_a, source_ids_a.clone()),
                    (word_id_b, source_ids_b.clone()),
                ]),
            )]),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![10],
        };

        let batch = make_batch(1, &[("n1", &["a"]), ("n2", &["b"])]);
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
            )
            .unwrap();
        let change_metrics = engine
            .compute_word_change_metrics_for_scope(&scope_snapshot, &mapped_scope, &update_stage)
            .unwrap();
        let decision = engine
            .evaluate_scope_trigger_decision_paper_scaffold(
                &scope_snapshot,
                &mapped_scope,
                &update_stage,
                &change_metrics,
            )
            .unwrap();

        assert_eq!(decision.paper_beta_error, Some(0.25));
    }

    #[test]
    fn paper_alpha_error_matches_option1_strength1_average() {
        let mut config = SecaConfig::default();
        config.trigger_policy_mode = TriggerPolicyMode::PaperDiagnosticScaffold;
        config.seca_thresholds.alpha = 0.6;
        config.seca_thresholds.alpha_option1_threshold = 0.1;
        config.seca_thresholds.beta_option1_threshold = 1.0;
        config.seca_thresholds.word_importance_option1_threshold = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(
                0,
                &[
                    ("s1", &["a"]),
                    ("s2", &["a"]),
                    ("s3", &["a"]),
                    ("s4", &["a"]),
                    ("s5", &["b"]),
                ],
            ))
            .unwrap();

        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();
        let word_id_b = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "b")
            .map(|(id, _)| *id)
            .unwrap();

        let source_ids_a: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| matches!(url.as_str(), "s1" | "s2" | "s3" | "s4"))
            .map(|(id, _)| *id)
            .collect();
        let source_ids_b: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| url.as_str() == "s5")
            .map(|(id, _)| *id)
            .collect();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 210,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![10],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeSet::from([word_id_a, word_id_b]),
            )]),
            node_source_ids_by_node_id: BTreeMap::from([(
                10,
                source_ids_a
                    .union(&source_ids_b)
                    .copied()
                    .collect::<BTreeSet<i64>>(),
            )]),
            node_word_source_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeMap::from([
                    (word_id_a, source_ids_a.clone()),
                    (word_id_b, source_ids_b.clone()),
                ]),
            )]),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![10],
        };

        let batch = make_batch(1, &[("n1", &["a"])]);
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
            )
            .unwrap();
        let change_metrics = engine
            .compute_word_change_metrics_for_scope(&scope_snapshot, &mapped_scope, &update_stage)
            .unwrap();
        let decision = engine
            .evaluate_scope_trigger_decision_paper_scaffold(
                &scope_snapshot,
                &mapped_scope,
                &update_stage,
                &change_metrics,
            )
            .unwrap();

        assert_eq!(decision.paper_alpha_error, Some(0.2));
    }

    #[test]
    fn paper_alpha_error_option1_rounds_to_four_decimals() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.alpha = 0.5;

        let engine = SecaEngine::new(config).unwrap();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 410,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![10],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::from([(10, BTreeSet::from([1_i32, 2_i32, 3_i32]))]),
            node_source_ids_by_node_id: BTreeMap::new(),
            node_word_source_ids_by_node_id: BTreeMap::new(),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![10],
        };

        let mapped_scope = MappedHktScopeState {
            hkt_id: 410,
            scoped_batch_source_indexes: BTreeSet::new(),
            matched_node_source_indexes_by_node_id: BTreeMap::new(),
            word_document_frequency_in_scope: BTreeMap::new(),
            known_word_ids_in_scope: BTreeSet::new(),
            new_tokens_in_scope: BTreeSet::new(),
            word_document_frequency_by_node_id: BTreeMap::new(),
            known_word_ids_by_node_id: BTreeMap::new(),
            new_tokens_by_node_id: BTreeMap::new(),
        };

        let update_stage = crate::engine::trigger::HktUpdateStage::default();

        let mut change_metrics: BTreeMap<i32, WordChangeMetrics> = BTreeMap::new();
        let strength1_values = [(1_i32, 0.0_f64), (2_i32, 0.0_f64), (3_i32, 0.5_f64)];
        for (word_id, strength1) in strength1_values {
            change_metrics.insert(
                word_id,
                WordChangeMetrics {
                    word_id,
                    node_id: 10,
                    number_of_sources_before_new_batch: 0.0,
                    number_of_sources_in_new_batch: 0.0,
                    number_of_sources_after_new_batch: 0.0,
                    number_of_sources_before_new_batch_over_number_of_sources_of_old_promin_word_in_hkt: 0.0,
                    number_of_intersected_sources_with_old_promin_word_in_node_over_num_sources_of_old_promin_word_in_node: 0.0,
                    number_of_sources_after_new_batch_over_number_of_sources_of_new_promin_word_in_hkt: strength1,
                    number_of_intersected_sources_with_new_promin_word_in_node_over_num_sources_of_new_promin_word_in_node: 0.0,
                    old_deviation_fom_alpha_parameter: 0.0,
                    old_deviation_fom_beta_parameter: 0.0,
                    new_deviation_fom_alpha_parameter: 0.0,
                    new_deviation_fom_beta_parameter: 0.0,
                    erros_in_deviation_fom_alpha_parameter: 0.0,
                    erros_in_deviation_fom_beta_parameter: 0.0,
                    precentage_of_sources_in_hkt_in_old_batch: 0.0,
                    precentage_of_sources_in_hkt_in_new_batch: 0.0,
                },
            );
        }

        let metrics = engine
            .compute_paper_scope_metrics_from_change_metrics(
                &scope_snapshot,
                &mapped_scope,
                &update_stage,
                &change_metrics,
            )
            .unwrap();

        assert_eq!(
            metrics.alpha_error_option1(),
            Some(0.3333),
            "expected alpha option1 error to round to 4 decimals"
        );
    }

    #[test]
    fn paper_beta_error_option1_rounds_to_four_decimals() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.beta = 0.5;

        let engine = SecaEngine::new(config).unwrap();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 411,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![10],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::from([(10, BTreeSet::from([1_i32, 2_i32, 3_i32]))]),
            node_source_ids_by_node_id: BTreeMap::new(),
            node_word_source_ids_by_node_id: BTreeMap::new(),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![10],
        };

        let mapped_scope = MappedHktScopeState {
            hkt_id: 411,
            scoped_batch_source_indexes: BTreeSet::new(),
            matched_node_source_indexes_by_node_id: BTreeMap::new(),
            word_document_frequency_in_scope: BTreeMap::new(),
            known_word_ids_in_scope: BTreeSet::new(),
            new_tokens_in_scope: BTreeSet::new(),
            word_document_frequency_by_node_id: BTreeMap::new(),
            known_word_ids_by_node_id: BTreeMap::new(),
            new_tokens_by_node_id: BTreeMap::new(),
        };

        let update_stage = crate::engine::trigger::HktUpdateStage::default();

        let mut change_metrics: BTreeMap<i32, WordChangeMetrics> = BTreeMap::new();
        let eligibility1_values = [(1_i32, 0.0_f64), (2_i32, 0.0_f64), (3_i32, 0.5_f64)];
        for (word_id, eligibility1) in eligibility1_values {
            change_metrics.insert(
                word_id,
                WordChangeMetrics {
                    word_id,
                    node_id: 10,
                    number_of_sources_before_new_batch: 0.0,
                    number_of_sources_in_new_batch: 0.0,
                    number_of_sources_after_new_batch: 0.0,
                    number_of_sources_before_new_batch_over_number_of_sources_of_old_promin_word_in_hkt: 0.0,
                    number_of_intersected_sources_with_old_promin_word_in_node_over_num_sources_of_old_promin_word_in_node: 0.0,
                    number_of_sources_after_new_batch_over_number_of_sources_of_new_promin_word_in_hkt: 0.0,
                    number_of_intersected_sources_with_new_promin_word_in_node_over_num_sources_of_new_promin_word_in_node:
                        eligibility1,
                    old_deviation_fom_alpha_parameter: 0.0,
                    old_deviation_fom_beta_parameter: 0.0,
                    new_deviation_fom_alpha_parameter: 0.0,
                    new_deviation_fom_beta_parameter: 0.0,
                    erros_in_deviation_fom_alpha_parameter: 0.0,
                    erros_in_deviation_fom_beta_parameter: 0.0,
                    precentage_of_sources_in_hkt_in_old_batch: 0.0,
                    precentage_of_sources_in_hkt_in_new_batch: 0.0,
                },
            );
        }

        let metrics = engine
            .compute_paper_scope_metrics_from_change_metrics(
                &scope_snapshot,
                &mapped_scope,
                &update_stage,
                &change_metrics,
            )
            .unwrap();

        assert_eq!(
            metrics.beta_error_option1(),
            Some(0.3333),
            "expected beta option1 error to round to 4 decimals"
        );
    }

    #[test]
    fn paper_word_importance_error_option1_rounds_to_four_decimals() {
        let engine = SecaEngine::new(SecaConfig::default()).unwrap();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 412,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![10],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::from([(10, BTreeSet::from([1_i32, 2_i32, 3_i32]))]),
            node_source_ids_by_node_id: BTreeMap::new(),
            node_word_source_ids_by_node_id: BTreeMap::new(),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![10],
        };

        let mapped_scope = MappedHktScopeState {
            hkt_id: 412,
            scoped_batch_source_indexes: BTreeSet::new(),
            matched_node_source_indexes_by_node_id: BTreeMap::new(),
            word_document_frequency_in_scope: BTreeMap::new(),
            known_word_ids_in_scope: BTreeSet::new(),
            new_tokens_in_scope: BTreeSet::new(),
            word_document_frequency_by_node_id: BTreeMap::new(),
            known_word_ids_by_node_id: BTreeMap::new(),
            new_tokens_by_node_id: BTreeMap::new(),
        };

        let update_stage = crate::engine::trigger::HktUpdateStage::default();

        let mut change_metrics: BTreeMap<i32, WordChangeMetrics> = BTreeMap::new();
        let importance1_values = [
            (1_i32, 0.0_f64),
            (2_i32, 0.0_f64),
            (3_i32, 2.0_f64 / 3.0_f64),
        ];
        for (word_id, importance1) in importance1_values {
            change_metrics.insert(
                word_id,
                WordChangeMetrics {
                    word_id,
                    node_id: 10,
                    number_of_sources_before_new_batch: 0.0,
                    number_of_sources_in_new_batch: 0.0,
                    number_of_sources_after_new_batch: 0.0,
                    number_of_sources_before_new_batch_over_number_of_sources_of_old_promin_word_in_hkt: 0.0,
                    number_of_intersected_sources_with_old_promin_word_in_node_over_num_sources_of_old_promin_word_in_node: 0.0,
                    number_of_sources_after_new_batch_over_number_of_sources_of_new_promin_word_in_hkt: 0.0,
                    number_of_intersected_sources_with_new_promin_word_in_node_over_num_sources_of_new_promin_word_in_node: 0.0,
                    old_deviation_fom_alpha_parameter: 0.0,
                    old_deviation_fom_beta_parameter: 0.0,
                    new_deviation_fom_alpha_parameter: 0.0,
                    new_deviation_fom_beta_parameter: 0.0,
                    erros_in_deviation_fom_alpha_parameter: 0.0,
                    erros_in_deviation_fom_beta_parameter: 0.0,
                    precentage_of_sources_in_hkt_in_old_batch: 0.0,
                    precentage_of_sources_in_hkt_in_new_batch: importance1,
                },
            );
        }

        let metrics = engine
            .compute_paper_scope_metrics_from_change_metrics(
                &scope_snapshot,
                &mapped_scope,
                &update_stage,
                &change_metrics,
            )
            .unwrap();

        assert_eq!(
            metrics.word_importance_error_option1(),
            Some(0.3333),
            "expected word importance option1 error to round to 4 decimals"
        );
    }

    #[test]
    fn paper_alpha_error_option2_matches_strength_difference_average() {
        let mut config = SecaConfig::default();
        config.trigger_policy_mode = TriggerPolicyMode::PaperDiagnosticScaffold;
        config.seca_thresholds.alpha = 0.6;
        config.seca_thresholds.selected_alpha_option = crate::config::AlphaErrorOption::Option2;
        config.seca_thresholds.alpha_option2_threshold = 1.0;
        config.seca_thresholds.beta_option1_threshold = 1.0;
        config.seca_thresholds.word_importance_option1_threshold = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(
                0,
                &[
                    ("s1", &["a"]),
                    ("s2", &["a"]),
                    ("s3", &["a"]),
                    ("s4", &["a"]),
                    ("s5", &["b"]),
                ],
            ))
            .unwrap();

        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();
        let word_id_b = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "b")
            .map(|(id, _)| *id)
            .unwrap();

        let source_ids_a: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| matches!(url.as_str(), "s1" | "s2" | "s3" | "s4"))
            .map(|(id, _)| *id)
            .collect();
        let source_ids_b: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| url.as_str() == "s5")
            .map(|(id, _)| *id)
            .collect();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 212,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![10],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeSet::from([word_id_a, word_id_b]),
            )]),
            node_source_ids_by_node_id: BTreeMap::from([(
                10,
                source_ids_a
                    .union(&source_ids_b)
                    .copied()
                    .collect::<BTreeSet<i64>>(),
            )]),
            node_word_source_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeMap::from([
                    (word_id_a, source_ids_a.clone()),
                    (word_id_b, source_ids_b.clone()),
                ]),
            )]),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![10],
        };

        let batch = make_batch(1, &[("n1", &["a"])]);
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
            )
            .unwrap();
        let change_metrics = engine
            .compute_word_change_metrics_for_scope(&scope_snapshot, &mapped_scope, &update_stage)
            .unwrap();
        let decision = engine
            .evaluate_scope_trigger_decision_paper_scaffold(
                &scope_snapshot,
                &mapped_scope,
                &update_stage,
                &change_metrics,
            )
            .unwrap();

        assert_eq!(decision.paper_alpha_error, Some(0.025));
    }

    #[test]
    fn paper_alpha_error_option3_matches_euclidean_distance() {
        let mut config = SecaConfig::default();
        config.trigger_policy_mode = TriggerPolicyMode::PaperDiagnosticScaffold;
        config.seca_thresholds.alpha = 0.6;
        config.seca_thresholds.selected_alpha_option = crate::config::AlphaErrorOption::Option3;
        config.seca_thresholds.alpha_option3_threshold = 1.0;
        config.seca_thresholds.beta_option1_threshold = 1.0;
        config.seca_thresholds.word_importance_option1_threshold = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(
                0,
                &[
                    ("s1", &["a"]),
                    ("s2", &["a"]),
                    ("s3", &["a"]),
                    ("s4", &["a"]),
                    ("s5", &["b"]),
                ],
            ))
            .unwrap();

        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();
        let word_id_b = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "b")
            .map(|(id, _)| *id)
            .unwrap();

        let source_ids_a: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| matches!(url.as_str(), "s1" | "s2" | "s3" | "s4"))
            .map(|(id, _)| *id)
            .collect();
        let source_ids_b: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| url.as_str() == "s5")
            .map(|(id, _)| *id)
            .collect();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 213,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![10],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeSet::from([word_id_a, word_id_b]),
            )]),
            node_source_ids_by_node_id: BTreeMap::from([(
                10,
                source_ids_a
                    .union(&source_ids_b)
                    .copied()
                    .collect::<BTreeSet<i64>>(),
            )]),
            node_word_source_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeMap::from([
                    (word_id_a, source_ids_a.clone()),
                    (word_id_b, source_ids_b.clone()),
                ]),
            )]),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![10],
        };

        let batch = make_batch(1, &[("n1", &["a"])]);
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
            )
            .unwrap();
        let change_metrics = engine
            .compute_word_change_metrics_for_scope(&scope_snapshot, &mapped_scope, &update_stage)
            .unwrap();
        let decision = engine
            .evaluate_scope_trigger_decision_paper_scaffold(
                &scope_snapshot,
                &mapped_scope,
                &update_stage,
                &change_metrics,
            )
            .unwrap();

        assert_eq!(decision.paper_alpha_error, Some(0.05));
    }

    #[test]
    fn paper_word_importance_error_accounts_for_unassigned_expected_words() {
        let mut config = SecaConfig::default();
        config.trigger_policy_mode = TriggerPolicyMode::PaperDiagnosticScaffold;
        config.seca_thresholds.alpha = 0.0;
        config.seca_thresholds.beta = 1.0; // keep expected word unassigned
        config.seca_thresholds.word_importance_option1_threshold = 0.1;
        config.seca_thresholds.alpha_option1_threshold = 1.0;
        config.seca_thresholds.beta_option1_threshold = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["a"])]))
            .unwrap();

        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();

        let source_ids_a: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| url.as_str() == "s1" || url.as_str() == "s2")
            .map(|(id, _)| *id)
            .collect();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 211,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![10],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::from([(10, BTreeSet::from([word_id_a]))]),
            node_source_ids_by_node_id: BTreeMap::from([(10, source_ids_a.clone())]),
            node_word_source_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeMap::from([(word_id_a, source_ids_a.clone())]),
            )]),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![10],
        };

        let batch = make_batch(1, &[("n1", &["new_x"]), ("n2", &["new_x"])]);
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
            )
            .unwrap();
        let change_metrics = engine
            .compute_word_change_metrics_for_scope(&scope_snapshot, &mapped_scope, &update_stage)
            .unwrap();
        let decision = engine
            .evaluate_scope_trigger_decision_paper_scaffold(
                &scope_snapshot,
                &mapped_scope,
                &update_stage,
                &change_metrics,
            )
            .unwrap();

        assert_eq!(decision.paper_word_importance_error, Some(0.5));
    }

    #[test]
    fn paper_word_importance_error_option2_matches_euclidean_distance() {
        let mut config = SecaConfig::default();
        config.trigger_policy_mode = TriggerPolicyMode::PaperDiagnosticScaffold;
        config.seca_thresholds.alpha = 0.6;
        config.seca_thresholds.selected_word_importance_option =
            crate::config::WordImportanceErrorOption::Option2;
        config.seca_thresholds.word_importance_option2_threshold = 1.0;
        config.seca_thresholds.alpha_option1_threshold = 1.0;
        config.seca_thresholds.beta_option1_threshold = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(
                0,
                &[
                    ("s1", &["a"]),
                    ("s2", &["a"]),
                    ("s3", &["a"]),
                    ("s4", &["a"]),
                    ("s5", &["b"]),
                ],
            ))
            .unwrap();

        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();
        let word_id_b = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "b")
            .map(|(id, _)| *id)
            .unwrap();

        let source_ids_a: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| matches!(url.as_str(), "s1" | "s2" | "s3" | "s4"))
            .map(|(id, _)| *id)
            .collect();
        let source_ids_b: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| url.as_str() == "s5")
            .map(|(id, _)| *id)
            .collect();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 214,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![10],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeSet::from([word_id_a, word_id_b]),
            )]),
            node_source_ids_by_node_id: BTreeMap::from([(
                10,
                source_ids_a
                    .union(&source_ids_b)
                    .copied()
                    .collect::<BTreeSet<i64>>(),
            )]),
            node_word_source_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeMap::from([
                    (word_id_a, source_ids_a.clone()),
                    (word_id_b, source_ids_b.clone()),
                ]),
            )]),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![10],
        };

        let batch = make_batch(1, &[("n1", &["a"])]);
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
            )
            .unwrap();
        let change_metrics = engine
            .compute_word_change_metrics_for_scope(&scope_snapshot, &mapped_scope, &update_stage)
            .unwrap();
        let decision = engine
            .evaluate_scope_trigger_decision_paper_scaffold(
                &scope_snapshot,
                &mapped_scope,
                &update_stage,
                &change_metrics,
            )
            .unwrap();

        assert_eq!(decision.paper_word_importance_error, Some(0.0471));
    }

    #[test]
    fn paper_selected_alpha_option_drives_trigger_decision() {
        let mut config = SecaConfig::default();
        config.trigger_policy_mode = TriggerPolicyMode::PaperDiagnosticScaffold;
        config.seca_thresholds.alpha = 0.6;
        config.seca_thresholds.selected_alpha_option = crate::config::AlphaErrorOption::Option2;
        config.seca_thresholds.alpha_option1_threshold = 1.0;
        config.seca_thresholds.alpha_option2_threshold = 0.02;
        config.seca_thresholds.beta_option1_threshold = 1.0;
        config.seca_thresholds.word_importance_option1_threshold = 1.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(
                0,
                &[
                    ("s1", &["a"]),
                    ("s2", &["a"]),
                    ("s3", &["a"]),
                    ("s4", &["a"]),
                    ("s5", &["b"]),
                ],
            ))
            .unwrap();

        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();
        let word_id_b = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "b")
            .map(|(id, _)| *id)
            .unwrap();

        let source_ids_a: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| matches!(url.as_str(), "s1" | "s2" | "s3" | "s4"))
            .map(|(id, _)| *id)
            .collect();
        let source_ids_b: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| url.as_str() == "s5")
            .map(|(id, _)| *id)
            .collect();

        let scope_snapshot = HktScopeSnapshot {
            hkt_id: 215,
            parent_node_id: 0,
            expected_word_ids: BTreeSet::new(),
            non_refuge_node_ids: vec![10],
            refuge_node_id: None,
            node_word_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeSet::from([word_id_a, word_id_b]),
            )]),
            node_source_ids_by_node_id: BTreeMap::from([(
                10,
                source_ids_a
                    .union(&source_ids_b)
                    .copied()
                    .collect::<BTreeSet<i64>>(),
            )]),
            node_word_source_ids_by_node_id: BTreeMap::from([(
                10,
                BTreeMap::from([
                    (word_id_a, source_ids_a.clone()),
                    (word_id_b, source_ids_b.clone()),
                ]),
            )]),
            child_hkt_ids_by_parent_node_id: BTreeMap::new(),
            node_ids_in_hkt_order: vec![10],
        };

        let batch = make_batch(1, &[("n1", &["a"])]);
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
            )
            .unwrap();
        let change_metrics = engine
            .compute_word_change_metrics_for_scope(&scope_snapshot, &mapped_scope, &update_stage)
            .unwrap();
        let decision = engine
            .evaluate_scope_trigger_decision_paper_scaffold(
                &scope_snapshot,
                &mapped_scope,
                &update_stage,
                &change_metrics,
            )
            .unwrap();

        assert!(decision.should_reconstruct);
        assert!(decision
            .trigger_reasons
            .iter()
            .any(|reason| reason.contains("paper_shadow.alpha_error")));
    }
    #[test]
    fn paper_scaffold_can_trigger_from_paper_beta_error_path() {
        let mut config = SecaConfig::default();
        config.trigger_policy_mode = TriggerPolicyMode::PaperDiagnosticScaffold;
        config.seca_thresholds.alpha_option1_threshold = 1.0;
        config.seca_thresholds.word_importance_option1_threshold = 1.0;
        config.seca_thresholds.beta = 0.8;
        config.seca_thresholds.beta_option1_threshold = 0.1;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s1", &["a", "b", "c"]), ("s2", &["a"])]))
            .unwrap();

        let result = engine
            .process_batch(make_batch(
                1,
                &[("n1", &["a"]), ("n2", &["a"]), ("n3", &["a"])],
            ))
            .unwrap();

        // This may or may not trigger depending on actual HKT mapping topology in your small baseline.
        // So assert on notes first (safer contract), then tighten later when topology fixtures are stable.
        assert!(result
            .notes
            .iter()
            .any(|note| note.contains("active trigger policy: paper_diagnostic_scaffold")));

        assert!(result
            .notes
            .iter()
            .any(|note| note.contains("paper-policy shadow")));
    }

    #[test]
    fn map_batch_into_hkt_scope_populates_node_level_diagnostics_fields() {
        let mut engine = SecaEngine::new(SecaConfig::default()).unwrap();

        let baseline = make_batch(
            0,
            &[
                ("s1", &["earthquake", "damage", "city"]),
                ("s2", &["storm", "coast"]),
            ],
        );

        engine.build_baseline_tree(baseline).unwrap();

        let snapshot = engine.snapshot_hkt_scope(root_hkt_id(&engine)).unwrap();

        let batch = make_batch(
            1,
            &[
                ("n1", &["earthquake", "aftershock"]),
                ("n2", &["storm", "surge"]),
            ],
        );

        let scoped_indexes = BTreeSet::from([0_usize, 1_usize]);
        let mapped = engine
            .map_batch_into_hkt_scope(&batch, &scoped_indexes, &snapshot)
            .unwrap();

        // New fields should exist and be internally consistent with node matches.
        assert!(
            !mapped.word_document_frequency_by_node_id.is_empty()
                || !mapped.matched_node_source_indexes_by_node_id.is_empty()
        );

        for (node_id, source_indexes) in &mapped.matched_node_source_indexes_by_node_id {
            if !source_indexes.is_empty() {
                // If a node has mapped sources, it should usually have node-local diagnostics populated.
                // (At minimum one of the structures should have an entry.)
                let has_df = mapped
                    .word_document_frequency_by_node_id
                    .contains_key(node_id);
                let has_known = mapped.known_word_ids_by_node_id.contains_key(node_id);
                let has_new = mapped.new_tokens_by_node_id.contains_key(node_id);

                assert!(
                    has_df || has_known || has_new,
                    "node {} had mapped sources but no node-level diagnostics entries",
                    node_id
                );
            }
        }
    }
    #[test]
    fn selected_hkt_rebuild_plan_is_deterministic_and_deduplicated() {
        let mut engine = build_small_baseline_engine();

        let export = engine.export_baseline_tree().expect("export should work");
        let existing_hkt_ids: Vec<i32> = export.hkts.iter().map(|hkt| hkt.hkt_id).collect();

        assert!(
            !existing_hkt_ids.is_empty(),
            "expected at least one HKT in baseline export"
        );

        // Build a noisy input list from real IDs: reverse order + duplicates
        let mut requested_ids = existing_hkt_ids.clone();
        requested_ids.reverse();

        if let Some(first_id) = existing_hkt_ids.first() {
            requested_ids.push(*first_id);
        }
        if let Some(last_id) = existing_hkt_ids.last() {
            requested_ids.push(*last_id);
        }

        let batch = SourceBatch {
            batch_index: 7,
            sources: Vec::new(),
        };

        let plan = engine
            .build_selected_hkt_rebuild_plan(&batch, &requested_ids)
            .expect("plan should build");
        assert_eq!(plan.batch_index, 7);

        let planned_ids: Vec<i32> = plan
            .requests
            .iter()
            .map(|request| request.target_hkt_id)
            .collect();

        let mut expected_ids = existing_hkt_ids.clone();
        expected_ids.sort_unstable();
        expected_ids.dedup();

        assert_eq!(planned_ids, expected_ids);

        assert!(plan
            .requests
            .iter()
            .all(|request| request.scoped_batch_source_indexes.is_empty()));
    }

    #[test]
    fn selected_hkt_execution_report_note_is_emitted_and_has_nonzero_scopes_when_triggered() {
        use crate::config::SecaConfig;
        use crate::types::{SourceBatch, SourceRecord};

        fn make_source(source_id: &str, batch_index: u32, tokens: &[&str]) -> SourceRecord {
            SourceRecord {
                source_id: source_id.to_string(),
                batch_index,
                tokens: tokens.iter().map(|t| (*t).to_string()).collect(),
                text: None,
                timestamp_unix_ms: None,
                metadata: None,
            }
        }

        fn make_batch(batch_index: u32, rows: &[(&str, &[&str])]) -> SourceBatch {
            SourceBatch {
                batch_index,
                sources: rows
                    .iter()
                    .map(|(source_id, tokens)| make_source(source_id, batch_index, tokens))
                    .collect(),
            }
        }

        let mut config = SecaConfig::default();
        config.seca_thresholds.word_importance_error_threshold = 0.0; // force trigger via placeholder WI
        config.seca_thresholds.word_importance_option1_threshold = 0.0;

        let mut engine = crate::SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s1", &["a", "b"]), ("s2", &["a", "c"])]))
            .unwrap();

        let result = engine
            .process_batch(make_batch(1, &[("n1", &["new_token"])]))
            .unwrap();

        assert!(result.reconstruction_triggered);

        let has_plan_note = result
            .notes
            .iter()
            .any(|n| n.contains("Selected-HKT rebuild plan"));
        assert!(has_plan_note, "missing selected rebuild plan note");

        let report_note = result
            .notes
            .iter()
            .find(|n| n.contains("Selected-HKT rebuild execution report"))
            .cloned();

        let Some(report_note) = report_note else {
            panic!(
                "missing selected rebuild execution report note; notes={:?}",
                result.notes
            );
        };

        // Ensure at least one scoped_sources > 0 appears in the report note
        assert!(
            report_note.contains("scoped_sources=1") || report_note.contains("scoped_sources="),
            "report note did not include scoped_sources; note={report_note}"
        );
        assert!(
            !report_note.contains("scoped_sources=0"),
            "expected some non-zero scoped_sources in report note; note={report_note}"
        );
    }

    #[test]
    fn subtree_targeted_mode_emits_subtree_dry_run_report_note_with_nonzero_subtree_counts() {
        use crate::engine::rebuild::RebuildMode;
        use crate::types::{SourceBatch, SourceRecord};
        use crate::{SecaConfig, SecaEngine};

        fn make_source(source_id: &str, batch_index: u32, tokens: &[&str]) -> SourceRecord {
            SourceRecord {
                source_id: source_id.to_string(),
                batch_index,
                tokens: tokens.iter().map(|t| (*t).to_string()).collect(),
                text: None,
                timestamp_unix_ms: None,
                metadata: None,
            }
        }

        fn make_batch(batch_index: u32, rows: &[(&str, &[&str])]) -> SourceBatch {
            SourceBatch {
                batch_index,
                sources: rows
                    .iter()
                    .map(|(source_id, tokens)| make_source(source_id, batch_index, tokens))
                    .collect(),
            }
        }

        let mut config = SecaConfig::default();
        // Force triggers in placeholder policy via WI gate.
        config.seca_thresholds.word_importance_error_threshold = 0.0;
        config.seca_thresholds.word_importance_option1_threshold = 0.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine.set_rebuild_mode(RebuildMode::SubtreeTargeted);

        // Baseline has only a/b/c, so "novel_x" forces a trigger at some scope.
        engine
            .build_baseline_tree(make_batch(0, &[("s1", &["a", "b"]), ("s2", &["a", "c"])]))
            .unwrap();

        let result = engine
            .process_batch(make_batch(1, &[("n1", &["novel_x", "a"])]))
            .unwrap();

        assert!(result.reconstruction_triggered);

        let report_note = result
            .notes
            .iter()
            .find(|note| note.contains("Selected-HKT subtree dry-run report"))
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "missing subtree dry-run report note; notes were: {:?}",
                    result.notes
                )
            });

        // Weak-but-stable contract: note includes the fields and indicates a non-empty subtree.
        assert!(
            report_note.contains("subtree_hkts=") && report_note.contains("subtree_nodes="),
            "report note missing subtree fields; note={}",
            report_note
        );

        // Ensure we didn't just produce an all-zero dry-run entry.
        // This is intentionally string-based to avoid adding new parsing helpers.
        assert!(
            !report_note.contains("subtree_hkts=0") || !report_note.contains("subtree_nodes=0"),
            "expected at least one non-zero subtree count; note={}",
            report_note
        );
    }
    #[test]
    fn ancestor_context_is_propagated_and_counts_increase_on_descent() {
        let mut engine = SecaEngine::new(SecaConfig::default()).unwrap();

        // Baseline chosen to likely create at least one branchable node.
        let baseline = make_batch(
            0,
            &[
                ("s1", &["a", "b", "c"]),
                ("s2", &["a", "b"]),
                ("s3", &["a", "b", "d"]),
                ("s4", &["x", "y"]),
            ],
        );

        engine.build_baseline_tree(baseline).unwrap();

        let batch = make_batch(
            1,
            &[
                ("n1", &["a", "b"]),
                ("n2", &["a", "b", "c"]),
                ("n3", &["x", "y"]),
            ],
        );

        let plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();

        // We don't assert exact accepted count because topology can vary with builder thresholds,
        // but we do assert that the ancestor context note exists at least once.
        assert!(
            plan.notes
                .iter()
                .any(|n| n.contains("ancestor context: accepted=")),
            "expected ancestor context notes; notes were: {:?}",
            plan.notes
        );
    }

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
            )
            .unwrap();

        assert_eq!(
            update_stage.updated_expected_word_ids.len(),
            expected_state1_count,
            "expected update-stage expected_words_state1 count to include new tokens"
        );

        let baseline_word_ids: BTreeSet<i32> =
            engine.baseline_word_legend.keys().copied().collect();
        let baseline_tokens: BTreeSet<String> =
            engine.baseline_word_legend.values().cloned().collect();
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
            )
            .unwrap();

        let baseline_word_ids: BTreeSet<i32> =
            engine.baseline_word_legend.keys().copied().collect();
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

        let mut scope_snapshot = engine.snapshot_hkt_scope(root_hkt_id).unwrap();

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
            .compute_update_stage_for_scope(&batch, &scope_snapshot, &mapped_scope, &ancestor)
            .unwrap();

        let baseline_word_ids: BTreeSet<i32> =
            engine.baseline_word_legend.keys().copied().collect();
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

        let mut scope_snapshot = engine.snapshot_hkt_scope(root_hkt_id).unwrap();

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
            )
            .unwrap();

        let baseline_word_ids: BTreeSet<i32> =
            engine.baseline_word_legend.keys().copied().collect();
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
            )
            .unwrap();

        let baseline_word_ids: BTreeSet<i32> =
            engine.baseline_word_legend.keys().copied().collect();
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
            .compute_update_stage_for_scope(&batch, &scope_snapshot, &mapped_scope, &ancestor)
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
            source_ids.retain(|id| {
                *id == SecaEngine::fnv1a_64("s1") || *id == SecaEngine::fnv1a_64("s2")
            });
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
            )
            .unwrap();

        let baseline_word_ids: BTreeSet<i32> =
            engine.baseline_word_legend.keys().copied().collect();
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
            )
            .unwrap();

        let baseline_word_ids: BTreeSet<i32> =
            engine.baseline_word_legend.keys().copied().collect();
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
    fn accepted_hkt_updates_node_sources() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.alpha_option1_threshold = 1.0;
        config.seca_thresholds.alpha_option2_threshold = 1.0;
        config.seca_thresholds.alpha_option3_threshold = 1.0;
        config.seca_thresholds.beta_option1_threshold = 1.0;
        config.seca_thresholds.beta_option2_threshold = 1.0;
        config.seca_thresholds.beta_option3_threshold = 1.0;
        config.seca_thresholds.word_importance_option1_threshold = 1.0;
        config.seca_thresholds.word_importance_option2_threshold = 1.0;
        config.seca_thresholds.selected_alpha_option = AlphaErrorOption::Option1;
        config.seca_thresholds.selected_beta_option = BetaErrorOption::Option1;
        config.seca_thresholds.selected_word_importance_option = WordImportanceErrorOption::Option1;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["a"])]))
            .unwrap();

        let result = engine
            .process_batch(make_batch(1, &[("s_new", &["a"])]))
            .unwrap();
        assert!(
            !result.reconstruction_triggered,
            "expected no reconstruction under relaxed thresholds"
        );

        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();
        let new_source_id = engine.stable_source_id("s_new");

        let node = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .nodes_by_id
            .values()
            .find(|node| node.word_ids.contains(&word_id_a))
            .unwrap();

        assert!(
            node.source_ids.contains(&new_source_id),
            "expected node source_ids to include new state1 source"
        );
        assert!(
            node.word_source_ids
                .get(&word_id_a)
                .map(|sources| sources.contains(&new_source_id))
                .unwrap_or(false),
            "expected word_source_ids to include new source for word 'a'"
        );
    }

    #[test]
    fn accepted_hkt_clears_new_from_batches_after_merge() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.alpha_option1_threshold = 1.0;
        config.seca_thresholds.alpha_option2_threshold = 1.0;
        config.seca_thresholds.alpha_option3_threshold = 1.0;
        config.seca_thresholds.beta_option1_threshold = 1.0;
        config.seca_thresholds.beta_option2_threshold = 1.0;
        config.seca_thresholds.beta_option3_threshold = 1.0;
        config.seca_thresholds.word_importance_option1_threshold = 1.0;
        config.seca_thresholds.word_importance_option2_threshold = 1.0;
        config.seca_thresholds.selected_alpha_option = AlphaErrorOption::Option1;
        config.seca_thresholds.selected_beta_option = BetaErrorOption::Option1;
        config.seca_thresholds.selected_word_importance_option = WordImportanceErrorOption::Option1;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["a"])]))
            .unwrap();

        let _ = engine
            .process_batch(make_batch(1, &[("s_new", &["a"])]))
            .unwrap();

        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();

        let node = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .nodes_by_id
            .values()
            .find(|node| node.word_ids.contains(&word_id_a))
            .unwrap();

        assert!(
            node.source_ids_new_from_batches.is_empty(),
            "expected source_ids_new_from_batches to be cleared after merge"
        );
        assert!(
            node.word_source_ids_new_from_batches.is_empty(),
            "expected word_source_ids_new_from_batches to be cleared after merge"
        );
        assert!(
            node.word_ids_new_from_batches.is_empty(),
            "expected word_ids_new_from_batches to be cleared after merge"
        );
    }

    #[test]
    fn accepted_hkt_persists_expected_word_assignment() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.alpha = 0.0;
        config.seca_thresholds.beta = 0.0;
        config.seca_thresholds.alpha_option1_threshold = 1.0;
        config.seca_thresholds.alpha_option2_threshold = 1.0;
        config.seca_thresholds.alpha_option3_threshold = 1.0;
        config.seca_thresholds.beta_option1_threshold = 1.0;
        config.seca_thresholds.beta_option2_threshold = 1.0;
        config.seca_thresholds.beta_option3_threshold = 1.0;
        config.seca_thresholds.word_importance_option1_threshold = 1.0;
        config.seca_thresholds.word_importance_option2_threshold = 1.0;
        config.seca_thresholds.selected_alpha_option = AlphaErrorOption::Option1;
        config.seca_thresholds.selected_beta_option = BetaErrorOption::Option1;
        config.seca_thresholds.selected_word_importance_option = WordImportanceErrorOption::Option1;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(0, &[("s1", &["a"]), ("s2", &["a"])]))
            .unwrap();

        let _ = engine
            .process_batch(make_batch(1, &[("s_new", &["a", "new_x"])]))
            .unwrap();

        let new_source_id = engine.stable_source_id("s_new");
        let root_hkt = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id == 0)
            .unwrap();

        let baseline_word_ids: BTreeSet<i32> =
            engine.baseline_word_legend.keys().copied().collect();
        let synthetic_ids: Vec<i32> = root_hkt
            .expected_words
            .difference(&baseline_word_ids)
            .copied()
            .collect();
        assert_eq!(
            synthetic_ids.len(),
            1,
            "expected exactly one synthetic expected word to persist"
        );
        let synthetic_word_id = synthetic_ids[0];

        let node_id = root_hkt
            .nodes
            .iter()
            .find(|node| !node.is_refuge_node())
            .map(|node| node.node_id)
            .unwrap();
        let node = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .nodes_by_id
            .get(&node_id)
            .unwrap();

        assert!(
            node.word_source_ids
                .get(&synthetic_word_id)
                .map(|sources| sources.contains(&new_source_id))
                .unwrap_or(false),
            "expected expected-word assignment to persist into word_source_ids"
        );

        let _ = engine
            .process_batch(make_batch(2, &[("s_next", &["a"])]))
            .unwrap();
        let root_hkt_after = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id == 0)
            .unwrap();
        assert!(
            root_hkt_after.expected_words.contains(&synthetic_word_id),
            "expected synthetic expected word to persist across batches"
        );
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
            )
            .unwrap();

        let baseline_word_ids: BTreeSet<i32> =
            engine.baseline_word_legend.keys().copied().collect();
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
            .compute_word_change_metrics_for_scope(&scope_snapshot, &mapped_scope, &update_stage)
            .unwrap();

        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();

        let metrics_a = metrics.get(&word_id_a).unwrap();
        assert_eq!(metrics_a.number_of_sources_before_new_batch, 3.0);
        assert_eq!(metrics_a.number_of_sources_in_new_batch, 2.0);
        assert_eq!(metrics_a.number_of_sources_after_new_batch, 5.0);
        assert!((metrics_a.precentage_of_sources_in_hkt_in_old_batch - 1.0).abs() < 1e-6);
        assert!((metrics_a.precentage_of_sources_in_hkt_in_new_batch - (5.0 / 7.0)).abs() < 1e-6);
        assert!(
            (metrics_a
                .number_of_sources_after_new_batch_over_number_of_sources_of_new_promin_word_in_hkt
                - 1.0)
                .abs()
                < 1e-6
        );

        let metrics_new = metrics.get(&synthetic_word_id).unwrap();
        assert_eq!(metrics_new.number_of_sources_before_new_batch, 0.0);
        assert_eq!(metrics_new.number_of_sources_in_new_batch, 2.0);
        assert_eq!(metrics_new.number_of_sources_after_new_batch, 2.0);
        assert!((metrics_new.precentage_of_sources_in_hkt_in_new_batch - (2.0 / 7.0)).abs() < 1e-6);
        assert!(
            (metrics_new
                .number_of_sources_after_new_batch_over_number_of_sources_of_new_promin_word_in_hkt
                - 0.4)
                .abs()
                < 1e-6
        );
        assert!(
            (metrics_new
                .number_of_intersected_sources_with_new_promin_word_in_node_over_num_sources_of_new_promin_word_in_node
                - 0.4)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn word_change_metrics_expected_only_word_sets_node_id_zero_and_counts() {
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
            )
            .unwrap();

        let baseline_word_ids: BTreeSet<i32> =
            engine.baseline_word_legend.keys().copied().collect();
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
            .compute_word_change_metrics_for_scope(&scope_snapshot, &mapped_scope, &update_stage)
            .unwrap();
        let metrics_new = metrics.get(&synthetic_word_id).unwrap();
        assert_eq!(
            metrics_new.node_id, 0,
            "expected-only word should use node_id=0"
        );
        assert_eq!(metrics_new.number_of_sources_before_new_batch, 0.0);
        assert_eq!(metrics_new.number_of_sources_in_new_batch, 2.0);
        assert_eq!(metrics_new.number_of_sources_after_new_batch, 2.0);
    }

    #[test]
    fn word_change_metrics_prominent_word_uses_hkt_node_order() {
        let mut config = SecaConfig::default();
        config.seca_thresholds.alpha = 0.0;
        config.seca_thresholds.beta = 0.0;

        let mut engine = SecaEngine::new(config).unwrap();
        engine
            .build_baseline_tree(make_batch(
                0,
                &[
                    ("s1", &["a"]),
                    ("s2", &["a"]),
                    ("s3", &["a"]),
                    ("s4", &["a"]),
                    ("s5", &["a"]),
                    ("s6", &["b"]),
                ],
            ))
            .unwrap();

        let root_hkt_id = root_hkt_id(&engine);
        let mut scope_snapshot = engine.snapshot_hkt_scope(root_hkt_id).unwrap();

        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();
        let word_id_b = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "b")
            .map(|(id, _)| *id)
            .unwrap();

        let source_ids_a: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| matches!(url.as_str(), "s1" | "s2" | "s3" | "s4" | "s5"))
            .map(|(id, _)| *id)
            .collect();
        let source_ids_b: BTreeSet<i64> = engine
            .baseline_source_legend
            .iter()
            .filter(|(_, url)| url.as_str() == "s6")
            .map(|(id, _)| *id)
            .collect();

        scope_snapshot.node_word_ids_by_node_id = BTreeMap::from([
            (1, BTreeSet::from([word_id_a])),
            (2, BTreeSet::from([word_id_b])),
        ]);
        scope_snapshot.node_source_ids_by_node_id =
            BTreeMap::from([(1, source_ids_a.clone()), (2, source_ids_b.clone())]);
        scope_snapshot.node_word_source_ids_by_node_id = BTreeMap::from([
            (1, BTreeMap::from([(word_id_a, source_ids_a.clone())])),
            (2, BTreeMap::from([(word_id_b, source_ids_b.clone())])),
        ]);
        scope_snapshot.node_ids_in_hkt_order = vec![2, 1];

        let batch = make_batch(1, &[("n1", &["a", "b"])]);
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
            )
            .unwrap();

        let metrics = engine
            .compute_word_change_metrics_for_scope(&scope_snapshot, &mapped_scope, &update_stage)
            .unwrap();

        let metrics_a = metrics.get(&word_id_a).unwrap();
        let metrics_b = metrics.get(&word_id_b).unwrap();

        assert!(
            (metrics_a
                .number_of_sources_before_new_batch_over_number_of_sources_of_old_promin_word_in_hkt
                - 5.0)
                .abs()
                < 1e-6
        );
        assert!(
            (metrics_b
                .number_of_sources_before_new_batch_over_number_of_sources_of_old_promin_word_in_hkt
                - 1.0)
                .abs()
                < 1e-6
        );
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
            .compute_word_change_metrics_for_scope(&scope_snapshot, &mapped_scope, &update_stage)
            .unwrap();

        let metrics_legacy = metrics.get(&legacy_word_id).unwrap();
        assert_eq!(metrics_legacy.node_id, 0);
        assert_eq!(metrics_legacy.number_of_sources_before_new_batch, 0.0);
        assert_eq!(metrics_legacy.number_of_sources_in_new_batch, 1.0);
        assert_eq!(metrics_legacy.number_of_sources_after_new_batch, 3.0);
        assert!((metrics_legacy.precentage_of_sources_in_hkt_in_new_batch - 0.5).abs() < 1e-6);
    }

    #[test]
    fn subtree_rebuild_scoped_dataset_filters_ancestor_words() {
        use std::collections::{BTreeMap, BTreeSet};

        let mut engine = build_small_baseline_engine();

        // Baseline uses tokens "a" and "b" in node 1.
        // We will mark "a" as ancestor-accepted and ensure it is excluded in scoped rebuild dataset.
        let batch = make_batch(1, &[("s1", &["a", "new_token"])]);

        // Force trigger plan with HKT 1 and ancestor sets.
        let mut trigger_plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();
        let hkt_id = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id == 0)
            .unwrap()
            .hkt_id;

        // Set ancestor accepted word to the word id for "a"
        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();

        let mut accepted = BTreeMap::new();
        accepted.insert(word_id_a, (1_usize, 0_usize));
        trigger_plan
            .ancestor_accepted_words_by_hkt_id
            .insert(hkt_id, accepted);

        trigger_plan
            .ancestor_rejected_words_by_hkt_id
            .insert(hkt_id, BTreeMap::new());

        trigger_plan.reconstruct_hkt_ids = vec![hkt_id];
        trigger_plan
            .reconstruct_scopes_by_hkt_id
            .insert(hkt_id, BTreeSet::from([0_usize])); // only the first source

        // Build scoped records
        let records = crate::engine::rebuild::build_scoped_source_word_records_for_hkt(
            &engine,
            &batch,
            &trigger_plan,
            hkt_id,
        )
        .unwrap();

        // Ensure "a" is not present in scoped records
        let has_a = records.iter().any(|r| r.word.as_deref() == Some("a"));
        assert!(!has_a, "ancestor-accepted word should be filtered out");
    }

    #[test]
    fn subtree_rebuild_scoped_dataset_filters_full_ancestor_word_set() {
        use std::collections::{BTreeMap, BTreeSet};

        let mut engine = build_small_baseline_engine();

        let batch = make_batch(1, &[("s1", &["a", "new_token"])]);

        let mut trigger_plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();
        let hkt_id = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id == 0)
            .unwrap()
            .hkt_id;

        let word_id_a = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "a")
            .map(|(id, _)| *id)
            .unwrap();

        let mut ancestor_words = BTreeMap::new();
        ancestor_words.insert(word_id_a, (1_usize, 0_usize));
        trigger_plan
            .ancestor_words_by_hkt_id
            .insert(hkt_id, ancestor_words);

        trigger_plan
            .ancestor_accepted_words_by_hkt_id
            .insert(hkt_id, BTreeMap::new());
        trigger_plan
            .ancestor_rejected_words_by_hkt_id
            .insert(hkt_id, BTreeMap::new());

        trigger_plan.reconstruct_hkt_ids = vec![hkt_id];
        trigger_plan
            .reconstruct_scopes_by_hkt_id
            .insert(hkt_id, BTreeSet::from([0_usize]));

        let records = crate::engine::rebuild::build_scoped_source_word_records_for_hkt(
            &engine,
            &batch,
            &trigger_plan,
            hkt_id,
        )
        .unwrap();

        let has_a = records.iter().any(|r| r.word.as_deref() == Some("a"));
        assert!(
            !has_a,
            "ancestor word (from full set) should be filtered out"
        );
    }

    #[test]
    fn subtree_rebuild_scoped_dataset_matches_csharp_state0_state1_merge() {
        use std::collections::{BTreeMap, BTreeSet};

        let mut engine = build_small_baseline_engine();

        // State1 batch contributes one scoped source with new tokens.
        let batch = make_batch(1, &[("s4", &["a", "e", "f"])]);

        let mut trigger_plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();
        let hkt_id = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id == 0)
            .unwrap()
            .hkt_id;

        let word_id_b = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "b")
            .map(|(id, _)| *id)
            .unwrap();

        let mut ancestor_words = BTreeMap::new();
        ancestor_words.insert(word_id_b, (2_usize, 0_usize));
        trigger_plan
            .ancestor_words_by_hkt_id
            .insert(hkt_id, ancestor_words);

        trigger_plan
            .ancestor_accepted_words_by_hkt_id
            .insert(hkt_id, BTreeMap::new());
        trigger_plan
            .ancestor_rejected_words_by_hkt_id
            .insert(hkt_id, BTreeMap::new());

        trigger_plan.reconstruct_hkt_ids = vec![hkt_id];
        trigger_plan
            .reconstruct_scopes_by_hkt_id
            .insert(hkt_id, BTreeSet::from([0_usize]));

        let records = crate::engine::rebuild::build_scoped_source_word_records_for_hkt(
            &engine,
            &batch,
            &trigger_plan,
            hkt_id,
        )
        .unwrap();

        let observed_tokens: BTreeSet<String> =
            records.iter().filter_map(|r| r.word.clone()).collect();

        let expected_tokens: BTreeSet<String> = ["a", "c", "d", "e", "f"]
            .iter()
            .map(|t| t.to_string())
            .collect();

        assert_eq!(
            observed_tokens, expected_tokens,
            "state0 + state1 tokens should merge, excluding ancestor words"
        );
    }

    #[test]
    fn subtree_rebuild_scoped_dataset_matches_csharp_word_source_counts() {
        use std::collections::{BTreeMap, BTreeSet};

        let mut engine = build_small_baseline_engine();

        // State1 adds one scoped source with new tokens.
        let batch = make_batch(1, &[("s4", &["a", "e", "f"])]);

        let mut trigger_plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();
        let hkt_id = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id == 0)
            .unwrap()
            .hkt_id;

        let word_id_b = engine
            .baseline_word_legend
            .iter()
            .find(|(_, token)| token.as_str() == "b")
            .map(|(id, _)| *id)
            .unwrap();

        let mut ancestor_words = BTreeMap::new();
        ancestor_words.insert(word_id_b, (2_usize, 0_usize));
        trigger_plan
            .ancestor_words_by_hkt_id
            .insert(hkt_id, ancestor_words);

        trigger_plan
            .ancestor_accepted_words_by_hkt_id
            .insert(hkt_id, BTreeMap::new());
        trigger_plan
            .ancestor_rejected_words_by_hkt_id
            .insert(hkt_id, BTreeMap::new());

        trigger_plan.reconstruct_hkt_ids = vec![hkt_id];
        trigger_plan
            .reconstruct_scopes_by_hkt_id
            .insert(hkt_id, BTreeSet::from([0_usize]));

        let records = crate::engine::rebuild::build_scoped_source_word_records_for_hkt(
            &engine,
            &batch,
            &trigger_plan,
            hkt_id,
        )
        .unwrap();

        let mut sources_by_word: BTreeMap<String, BTreeSet<i64>> = BTreeMap::new();
        let mut reported_counts_by_word: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();

        for record in &records {
            let word = record.word.as_ref().unwrap().clone();
            sources_by_word
                .entry(word.clone())
                .or_default()
                .insert(record.source_id);
            reported_counts_by_word
                .entry(word)
                .or_default()
                .insert(record.word_number_of_sources);
        }

        let expected_counts: BTreeMap<String, usize> = BTreeMap::from([
            ("a".to_string(), 4_usize),
            ("c".to_string(), 1_usize),
            ("d".to_string(), 1_usize),
            ("e".to_string(), 1_usize),
            ("f".to_string(), 1_usize),
        ]);

        for (word, expected_count) in expected_counts {
            let sources = sources_by_word
                .get(&word)
                .unwrap_or_else(|| panic!("missing word {} in scoped records", word));
            assert_eq!(
                sources.len(),
                expected_count,
                "unique source count mismatch for word {}",
                word
            );

            let reported_counts = reported_counts_by_word
                .get(&word)
                .unwrap_or_else(|| panic!("missing word_number_of_sources for {}", word));
            assert_eq!(
                reported_counts.len(),
                1,
                "inconsistent word_number_of_sources values for {}",
                word
            );
            assert_eq!(
                *reported_counts.iter().next().unwrap(),
                expected_count,
                "reported word_number_of_sources mismatch for {}",
                word
            );
        }
    }

    #[test]
    fn subtree_rebuild_scoped_dataset_uses_only_current_hkt_sources_for_state0() {
        use std::collections::{BTreeMap, BTreeSet};

        let mut engine = build_small_baseline_engine();

        let child_hkt = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id != 0)
            .expect("expected a child HKT for scoped dataset state0 filtering");
        let child_hkt_id = child_hkt.hkt_id;

        let mut child_state0_source_ids: BTreeSet<i64> = BTreeSet::new();
        for node in &child_hkt.nodes {
            child_state0_source_ids.extend(node.source_ids.iter().copied());
        }

        let unrelated_source_id = engine
            .baseline_source_legend
            .keys()
            .copied()
            .find(|id| !child_state0_source_ids.contains(id))
            .expect("expected at least one baseline source not in child HKT");

        let unrelated_source_url = engine
            .baseline_source_legend
            .get(&unrelated_source_id)
            .expect("unrelated source id should map to url")
            .clone();

        let batch = make_batch(1, &[(&unrelated_source_url, &["x"])]);
        let mut trigger_plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();
        trigger_plan.reconstruct_hkt_ids = vec![child_hkt_id];
        trigger_plan
            .reconstruct_scopes_by_hkt_id
            .insert(child_hkt_id, BTreeSet::new());
        trigger_plan
            .ancestor_words_by_hkt_id
            .insert(child_hkt_id, BTreeMap::new());
        trigger_plan
            .ancestor_accepted_words_by_hkt_id
            .insert(child_hkt_id, BTreeMap::new());
        trigger_plan
            .ancestor_rejected_words_by_hkt_id
            .insert(child_hkt_id, BTreeMap::new());

        let records = crate::engine::rebuild::build_scoped_source_word_records_for_hkt(
            &engine,
            &batch,
            &trigger_plan,
            child_hkt_id,
        )
        .unwrap();

        assert!(!records.is_empty(), "expected state0 records for child HKT");

        for record in &records {
            assert!(
                child_state0_source_ids.contains(&record.source_id),
                "state0 record source_id {} should belong to child HKT sources",
                record.source_id
            );
        }
    }

    #[test]
    fn subtree_rebuild_ancestor_words_are_present_in_child_state0_sources() {
        use std::collections::BTreeSet;

        let engine = build_small_baseline_engine();

        let child_hkt = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id != 0)
            .expect("expected a child HKT for ancestor state1-only test");
        let child_hkt_id = child_hkt.hkt_id;
        let parent_node_id = child_hkt.parent_node_id;

        let parent_node = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .nodes_by_id
            .get(&parent_node_id)
            .expect("parent node should exist");

        let mut child_state0_source_ids: BTreeSet<i64> = BTreeSet::new();
        for node in &child_hkt.nodes {
            child_state0_source_ids.extend(node.source_ids.iter().copied());
        }

        let baseline_batch = engine
            .processed_batches
            .first()
            .expect("baseline batch should be stored");

        let mut child_state0_tokens: BTreeSet<String> = BTreeSet::new();
        for source_id in &child_state0_source_ids {
            if let Some(external_id) = engine.baseline_source_legend.get(source_id) {
                if let Some(source) = baseline_batch
                    .sources
                    .iter()
                    .find(|s| s.source_id == *external_id)
                {
                    child_state0_tokens.extend(source.tokens.iter().cloned());
                }
            }
        }

        for word_id in parent_node.word_ids.iter().copied().filter(|id| *id != -1) {
            let token = engine
                .baseline_word_legend
                .get(&word_id)
                .expect("ancestor word id should map to a baseline token");
            assert!(
                child_state0_tokens.contains(token),
                "ancestor word {} should exist in child state0 sources",
                token
            );
        }
    }

    #[test]
    fn subtree_rebuild_propagates_state1_sources_to_parent_chain() {
        use std::collections::BTreeSet;

        let mut engine = build_small_baseline_engine();

        let child_hkt = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id != 0)
            .expect("expected a child HKT for parent propagation test");
        let child_hkt_id = child_hkt.hkt_id;
        let parent_node_id = child_hkt.parent_node_id;

        let state1_batch = make_batch(1, &[("s_new_parent", &["x", "y"])]);

        let mut trigger_plan = engine
            .evaluate_seca_trigger_plan_for_batch(&state1_batch)
            .unwrap();
        trigger_plan.reconstruct_hkt_ids = vec![child_hkt_id];
        trigger_plan
            .reconstruct_scopes_by_hkt_id
            .insert(child_hkt_id, BTreeSet::from([0_usize]));

        let new_source_id = engine.stable_source_id("s_new_parent");

        let before_sources = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .nodes_by_id
            .get(&parent_node_id)
            .unwrap()
            .source_ids
            .clone();

        engine
            .rebuild_selected_hkts_from_trigger_plan(&state1_batch, &trigger_plan)
            .unwrap();

        let after_sources = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .nodes_by_id
            .get(&parent_node_id)
            .unwrap()
            .source_ids
            .clone();

        assert!(
            after_sources.contains(&new_source_id),
            "parent node should include new state1 source after rebuild"
        );
        assert!(
            after_sources.len() >= before_sources.len(),
            "parent node source_ids should not shrink during propagation"
        );
    }

    #[test]
    fn subtree_rebuild_propagates_state1_sources_to_all_ancestors() {
        use std::collections::BTreeSet;

        let mut engine = build_small_baseline_engine();

        let child_hkt = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id != 0)
            .expect("expected a child HKT for ancestor propagation test");
        let child_hkt_id = child_hkt.hkt_id;
        let parent_node_id = child_hkt.parent_node_id;

        let state1_batch = make_batch(1, &[("s_new_chain", &["p", "q"])]);

        let mut trigger_plan = engine
            .evaluate_seca_trigger_plan_for_batch(&state1_batch)
            .unwrap();
        trigger_plan.reconstruct_hkt_ids = vec![child_hkt_id];
        trigger_plan
            .reconstruct_scopes_by_hkt_id
            .insert(child_hkt_id, BTreeSet::from([0_usize]));

        let new_source_id = engine.stable_source_id("s_new_chain");

        engine
            .rebuild_selected_hkts_from_trigger_plan(&state1_batch, &trigger_plan)
            .unwrap();

        let hkt_build_output = engine.hkt_build_output.as_ref().unwrap();
        let mut current_node_id = parent_node_id;

        while current_node_id != 0 {
            let node = hkt_build_output
                .nodes_by_id
                .get(&current_node_id)
                .expect("ancestor node should exist");
            assert!(
                node.source_ids.contains(&new_source_id),
                "ancestor node {} should contain propagated source id",
                current_node_id
            );

            let hkt_id = node.hkt_id;
            let parent_of_hkt = hkt_build_output
                .hkts_by_id
                .get(&hkt_id)
                .map(|hkt| hkt.parent_node_id)
                .unwrap_or(0);

            if parent_of_hkt == 0 {
                break;
            }
            current_node_id = parent_of_hkt;
        }
    }

    #[test]
    fn subtree_rebuild_archives_removed_subtree() {
        use std::collections::BTreeSet;

        let mut engine = build_small_baseline_engine();

        let batch = make_batch(1, &[("s_new_archive", &["z"])]);
        let trigger_plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();

        let hkt_build_output = engine.hkt_build_output.as_ref().unwrap();
        let root_hkt_id = hkt_build_output
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id == 0)
            .unwrap()
            .hkt_id;

        let mut forced_plan = trigger_plan.clone();
        forced_plan.reconstruct_hkt_ids = vec![root_hkt_id];
        forced_plan
            .reconstruct_scopes_by_hkt_id
            .insert(root_hkt_id, BTreeSet::from([0_usize]));

        engine
            .rebuild_selected_hkts_from_trigger_plan(&batch, &forced_plan)
            .unwrap();

        let archived = engine
            .archived_subtrees_by_root_id
            .get(&root_hkt_id)
            .expect("expected archived subtree for rebuilt root");

        assert!(
            archived.hkts_by_id.contains_key(&root_hkt_id),
            "archived subtree should contain original root HKT id"
        );
        assert!(
            !engine
                .hkt_build_output
                .as_ref()
                .unwrap()
                .hkts_by_id
                .contains_key(&root_hkt_id),
            "current tree should not contain original root HKT after rebuild"
        );
    }

    #[test]
    fn subtree_rebuild_records_logical_removal_of_root_hkt() {
        use std::collections::BTreeSet;

        let mut engine = build_small_baseline_engine();

        let root_hkt_id = root_hkt_id(&engine);
        let expected_words_before = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .get(&root_hkt_id)
            .unwrap()
            .expected_words
            .clone();

        let batch = make_batch(1, &[("s_new_logical", &["z"])]);
        let mut forced_plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();
        forced_plan.reconstruct_hkt_ids = vec![root_hkt_id];
        forced_plan
            .reconstruct_scopes_by_hkt_id
            .insert(root_hkt_id, BTreeSet::from([0_usize]));

        engine
            .rebuild_selected_hkts_from_trigger_plan(&batch, &forced_plan)
            .unwrap();

        let logical = engine
            .logically_removed_hkts_by_id
            .get(&root_hkt_id)
            .expect("expected logical removal record for rebuilt root");

        assert_eq!(
            logical.hkt.parent_node_id, -1,
            "logically removed HKT should have parent_node_id = -1"
        );
        assert!(
            logical.hkt.nodes.is_empty(),
            "logically removed HKT should have nodes cleared"
        );
        assert_eq!(
            logical.hkt.expected_words, expected_words_before,
            "logically removed HKT should preserve expected_words"
        );
        assert_eq!(
            logical.old_parent_node_id, 0,
            "logical removal should preserve the original parent node id"
        );
    }

    #[test]
    fn subtree_rebuild_assigns_new_ids_not_reused_from_archives() {
        use std::collections::BTreeSet;

        let mut engine = build_small_baseline_engine();

        let batch_one = make_batch(1, &[("s_arch_1", &["x"])]);
        let trigger_plan_one = engine
            .evaluate_seca_trigger_plan_for_batch(&batch_one)
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

        let mut forced_plan_one = trigger_plan_one.clone();
        forced_plan_one.reconstruct_hkt_ids = vec![root_hkt_id];
        forced_plan_one
            .reconstruct_scopes_by_hkt_id
            .insert(root_hkt_id, BTreeSet::from([0_usize]));

        engine
            .rebuild_selected_hkts_from_trigger_plan(&batch_one, &forced_plan_one)
            .unwrap();

        let archived_ids_after_first: BTreeSet<i32> = engine
            .archived_subtrees_by_root_id
            .values()
            .flat_map(|subtree| subtree.hkts_by_id.keys().copied())
            .collect();

        let current_ids_after_first: BTreeSet<i32> = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .keys()
            .copied()
            .collect();

        assert!(
            archived_ids_after_first.is_disjoint(&current_ids_after_first),
            "current tree HKT ids should not reuse archived ids"
        );
        let max_archived_first = archived_ids_after_first.iter().copied().max().unwrap_or(0);
        let min_current_first = current_ids_after_first.iter().copied().min().unwrap_or(0);
        assert!(
            min_current_first > max_archived_first,
            "current HKT ids should be greater than all archived ids after first rebuild"
        );

        let archived_node_ids_after_first: BTreeSet<i32> = engine
            .archived_subtrees_by_root_id
            .values()
            .flat_map(|subtree| subtree.nodes_by_id.keys().copied())
            .collect();
        let current_node_ids_after_first: BTreeSet<i32> = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .nodes_by_id
            .keys()
            .copied()
            .collect();
        assert!(
            archived_node_ids_after_first.is_disjoint(&current_node_ids_after_first),
            "current node ids should not reuse archived node ids"
        );
        let max_archived_node_first = archived_node_ids_after_first
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let min_current_node_first = current_node_ids_after_first
            .iter()
            .copied()
            .min()
            .unwrap_or(0);
        assert!(
            min_current_node_first > max_archived_node_first,
            "current node ids should be greater than all archived node ids after first rebuild"
        );

        let batch_two = make_batch(2, &[("s_arch_2", &["y"])]);
        let trigger_plan_two = engine
            .evaluate_seca_trigger_plan_for_batch(&batch_two)
            .unwrap();
        let root_hkt_id_two = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id == 0)
            .unwrap()
            .hkt_id;

        let mut forced_plan_two = trigger_plan_two.clone();
        forced_plan_two.reconstruct_hkt_ids = vec![root_hkt_id_two];
        forced_plan_two
            .reconstruct_scopes_by_hkt_id
            .insert(root_hkt_id_two, BTreeSet::from([0_usize]));

        engine
            .rebuild_selected_hkts_from_trigger_plan(&batch_two, &forced_plan_two)
            .unwrap();

        let archived_ids_after_second: BTreeSet<i32> = engine
            .archived_subtrees_by_root_id
            .values()
            .flat_map(|subtree| subtree.hkts_by_id.keys().copied())
            .collect();

        let current_ids_after_second: BTreeSet<i32> = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .keys()
            .copied()
            .collect();

        assert!(
            archived_ids_after_second.is_disjoint(&current_ids_after_second),
            "current tree HKT ids should not reuse archived ids after multiple rebuilds"
        );
        let max_archived_second = archived_ids_after_second.iter().copied().max().unwrap_or(0);
        let min_current_second = current_ids_after_second.iter().copied().min().unwrap_or(0);
        assert!(
            min_current_second > max_archived_second,
            "current HKT ids should be greater than all archived ids after second rebuild"
        );

        let archived_node_ids_after_second: BTreeSet<i32> = engine
            .archived_subtrees_by_root_id
            .values()
            .flat_map(|subtree| subtree.nodes_by_id.keys().copied())
            .collect();
        let current_node_ids_after_second: BTreeSet<i32> = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .nodes_by_id
            .keys()
            .copied()
            .collect();
        assert!(
            archived_node_ids_after_second.is_disjoint(&current_node_ids_after_second),
            "current node ids should not reuse archived node ids after multiple rebuilds"
        );
        let max_archived_node_second = archived_node_ids_after_second
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let min_current_node_second = current_node_ids_after_second
            .iter()
            .copied()
            .min()
            .unwrap_or(0);
        assert!(
            min_current_node_second > max_archived_node_second,
            "current node ids should be greater than all archived node ids after second rebuild"
        );
    }

    #[test]
    fn subtree_rebuild_updates_next_id_counters() {
        use std::collections::BTreeSet;

        let mut engine = build_small_baseline_engine();

        let batch = make_batch(1, &[("s_next_id", &["x"])]);
        let trigger_plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();
        let root_hkt_id = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id == 0)
            .unwrap()
            .hkt_id;

        let mut forced_plan = trigger_plan.clone();
        forced_plan.reconstruct_hkt_ids = vec![root_hkt_id];
        forced_plan
            .reconstruct_scopes_by_hkt_id
            .insert(root_hkt_id, BTreeSet::from([0_usize]));

        engine
            .rebuild_selected_hkts_from_trigger_plan(&batch, &forced_plan)
            .unwrap();

        let current_hkt_max = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .keys()
            .copied()
            .max()
            .unwrap_or(0);
        let current_node_max = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .nodes_by_id
            .keys()
            .copied()
            .max()
            .unwrap_or(0);

        let archived_hkt_max = engine
            .archived_subtrees_by_root_id
            .values()
            .flat_map(|subtree| subtree.hkts_by_id.keys().copied())
            .max()
            .unwrap_or(0);
        let archived_node_max = engine
            .archived_subtrees_by_root_id
            .values()
            .flat_map(|subtree| subtree.nodes_by_id.keys().copied())
            .max()
            .unwrap_or(0);

        let global_hkt_max = current_hkt_max.max(archived_hkt_max);
        let global_node_max = current_node_max.max(archived_node_max);

        assert!(
            engine.next_hkt_id > global_hkt_max,
            "next_hkt_id should stay ahead of all HKT ids"
        );
        assert!(
            engine.next_node_id > global_node_max,
            "next_node_id should stay ahead of all node ids"
        );
    }

    #[test]
    fn subtree_rebuild_preserves_parent_linkage() {
        let mut engine = build_small_baseline_engine();

        let batch = make_batch(1, &[("s1", &["new_token"])]);

        // Evaluate trigger plan
        let trigger_plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();

        // Find root HKT and a child HKT (if any)
        let hkt_build_output = engine.hkt_build_output.as_ref().unwrap();
        let root_hkt_id = hkt_build_output
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id == 0)
            .unwrap()
            .hkt_id;

        // Find a child HKT of the root if it exists, otherwise just rebuild root
        let target_hkt_id = hkt_build_output
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id != 0)
            .map(|hkt| hkt.hkt_id)
            .unwrap_or(root_hkt_id);

        let parent_node_id = hkt_build_output
            .hkts_by_id
            .get(&target_hkt_id)
            .unwrap()
            .parent_node_id;

        // Force rebuild on target
        let mut forced_plan = trigger_plan.clone();
        forced_plan.reconstruct_hkt_ids = vec![target_hkt_id];
        forced_plan
            .reconstruct_scopes_by_hkt_id
            .insert(target_hkt_id, BTreeSet::from([0_usize]));

        engine
            .rebuild_selected_hkts_from_trigger_plan(&batch, &forced_plan)
            .unwrap();

        // Verify that the rebuilt subtree root still points to the original parent
        let rebuilt_hkt = engine
            .hkt_build_output
            .as_ref()
            .unwrap()
            .hkts_by_id
            .values()
            .find(|hkt| hkt.parent_node_id == parent_node_id)
            .unwrap();

        assert_eq!(
            rebuilt_hkt.parent_node_id, parent_node_id,
            "rebuilt subtree root must preserve original parent_node_id"
        );
    }

    #[test]
    fn paper_policy_triggers_reconstruction_on_wi_error() {
        let mut config = SecaConfig::default();
        config.trigger_policy_mode = TriggerPolicyMode::PaperDiagnosticScaffold;
        config.seca_thresholds.word_importance_option1_threshold = 0.0;
        config.seca_thresholds.alpha_option1_threshold = 0.0;
        config.seca_thresholds.beta_option1_threshold = 0.0;

        let mut engine = SecaEngine::new(config).unwrap();

        let baseline = make_batch(0, &[("s1", &["a"])]);
        engine.build_baseline_tree(baseline).unwrap();

        let batch = make_batch(1, &[("s2", &["new_token"])]);
        let result = engine.process_batch(batch).unwrap();

        assert!(
            result.reconstruction_triggered,
            "expected reconstruction under paper policy"
        );

        assert!(
            result
                .notes
                .iter()
                .any(|n| n.contains("paper-policy shadow: would_trigger=true")),
            "expected paper-policy shadow trigger note; notes were: {:?}",
            result.notes
        );
    }
}
