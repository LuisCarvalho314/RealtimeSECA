use super::fixtures::*;
use std::collections::{BTreeMap, BTreeSet};
#[test]
fn rebuilt_subtree_hkts_are_state1() {
    let mut engine = build_small_baseline_engine();

    let root_hkt_id = root_hkt_id(&engine);
    let batch = make_batch(1, &[("s_new_state1", &["z"])]);
    let mut forced_plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();
    forced_plan.targets.retain(|id, _| *id == root_hkt_id);
    forced_plan.targets.entry(root_hkt_id).or_default();
    forced_plan
        .targets
        .entry(root_hkt_id)
        .or_default()
        .source_indexes = BTreeSet::from([0_usize]);

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
#[test]
fn subtree_rebuild_scoped_dataset_filters_ancestor_words() {
    use std::collections::BTreeSet;

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

    let accepted = BTreeSet::from([word_id_a]);
    trigger_plan
        .targets
        .entry(hkt_id)
        .or_default()
        .excluded_words = accepted;

    trigger_plan.targets.retain(|id, _| *id == hkt_id);
    trigger_plan.targets.entry(hkt_id).or_default();
    trigger_plan
        .targets
        .entry(hkt_id)
        .or_default()
        .source_indexes = BTreeSet::from([0_usize]); // only the first source

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
    use std::collections::BTreeSet;

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

    let mut ancestor_words = BTreeSet::new();
    ancestor_words.insert(word_id_a);
    trigger_plan
        .targets
        .entry(hkt_id)
        .or_default()
        .excluded_words = ancestor_words;

    trigger_plan.targets.retain(|id, _| *id == hkt_id);
    trigger_plan.targets.entry(hkt_id).or_default();
    trigger_plan
        .targets
        .entry(hkt_id)
        .or_default()
        .source_indexes = BTreeSet::from([0_usize]);

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
    use std::collections::BTreeSet;

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

    let mut ancestor_words = BTreeSet::new();
    ancestor_words.insert(word_id_b);
    trigger_plan
        .targets
        .entry(hkt_id)
        .or_default()
        .excluded_words = ancestor_words;

    trigger_plan.targets.retain(|id, _| *id == hkt_id);
    trigger_plan.targets.entry(hkt_id).or_default();
    trigger_plan
        .targets
        .entry(hkt_id)
        .or_default()
        .source_indexes = BTreeSet::from([0_usize]);

    let records = crate::engine::rebuild::build_scoped_source_word_records_for_hkt(
        &engine,
        &batch,
        &trigger_plan,
        hkt_id,
    )
    .unwrap();

    let observed_tokens: BTreeSet<String> = records.iter().filter_map(|r| r.word.clone()).collect();

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
    use std::collections::BTreeSet;

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

    let mut ancestor_words = BTreeSet::new();
    ancestor_words.insert(word_id_b);
    trigger_plan
        .targets
        .entry(hkt_id)
        .or_default()
        .excluded_words = ancestor_words;

    trigger_plan.targets.retain(|id, _| *id == hkt_id);
    trigger_plan.targets.entry(hkt_id).or_default();
    trigger_plan
        .targets
        .entry(hkt_id)
        .or_default()
        .source_indexes = BTreeSet::from([0_usize]);

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
    use std::collections::BTreeSet;

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
    trigger_plan.targets.retain(|id, _| *id == child_hkt_id);
    trigger_plan.targets.entry(child_hkt_id).or_default();
    trigger_plan
        .targets
        .entry(child_hkt_id)
        .or_default()
        .source_indexes = BTreeSet::new();
    trigger_plan
        .targets
        .entry(child_hkt_id)
        .or_default()
        .excluded_words = BTreeSet::new();

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
    let _child_hkt_id = child_hkt.hkt_id;
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
    trigger_plan.targets.retain(|id, _| *id == child_hkt_id);
    trigger_plan.targets.entry(child_hkt_id).or_default();
    trigger_plan
        .targets
        .entry(child_hkt_id)
        .or_default()
        .source_indexes = BTreeSet::from([0_usize]);

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
    trigger_plan.targets.retain(|id, _| *id == child_hkt_id);
    trigger_plan.targets.entry(child_hkt_id).or_default();
    trigger_plan
        .targets
        .entry(child_hkt_id)
        .or_default()
        .source_indexes = BTreeSet::from([0_usize]);

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
    forced_plan.targets.retain(|id, _| *id == target_hkt_id);
    forced_plan.targets.entry(target_hkt_id).or_default();
    forced_plan
        .targets
        .entry(target_hkt_id)
        .or_default()
        .source_indexes = BTreeSet::from([0_usize]);

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
fn ancestor_propagation_updates_owned_nodes_and_lookup_cache_together() {
    let mut engine = build_small_baseline_engine();
    let node_id = *engine
        .hkt_build_output
        .as_ref()
        .unwrap()
        .nodes_by_id
        .keys()
        .next()
        .unwrap();
    engine
        .propagate_sources_to_parent_chain(node_id, &[12345].into())
        .unwrap();
    let tree = engine.hkt_build_output.as_ref().unwrap();
    let cached = &tree.nodes_by_id[&node_id];
    let owned = tree.hkts_by_id[&cached.hkt_id]
        .nodes
        .iter()
        .find(|n| n.node_id == node_id)
        .unwrap();
    assert_eq!(owned, cached);
    assert!(owned.source_ids.contains(&12345));
}

#[test]
fn attachment_exclusions_accumulate_through_three_levels() {
    let mut config = crate::config::SecaConfig::default();
    config.hkt_builder.minimum_threshold_against_max_word_count = 0.7;
    let mut engine = crate::engine::SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(
            0,
            &[
                ("s1", &["a", "b", "c"]),
                ("s2", &["a", "b", "c"]),
                ("s3", &["a", "b"]),
                ("s4", &["a", "b"]),
                ("s5", &["a"]),
                ("s6", &["a"]),
                ("s7", &["a"]),
                ("s8", &["a"]),
            ],
        ))
        .unwrap();
    let mut id = root_hkt_id(&engine);
    let mut excluded = BTreeSet::new();
    for token in ["a", "b"] {
        let scope = engine.snapshot_hkt_scope(id).unwrap();
        let word = *engine
            .baseline_word_legend
            .iter()
            .find(|(_, t)| t.as_str() == token)
            .unwrap()
            .0;
        let node = *scope
            .node_word_ids_by_node_id
            .iter()
            .find(|(_, words)| words.contains(&word))
            .unwrap()
            .0;
        excluded = engine.build_child_ancestor_context_from_parent_node(&scope, node, &excluded);
        assert!(excluded.contains(&word));
        id = scope.child_hkt_ids_by_parent_node_id[&node];
    }
    assert_eq!(excluded.len(), 2);
    let plan = crate::engine::trigger::RecursiveTriggerPlan {
        targets: BTreeMap::from([(
            id,
            crate::engine::trigger::ReconstructionTarget {
                source_indexes: BTreeSet::new(),
                excluded_words: excluded,
            },
        )]),
        ..Default::default()
    };
    let records = crate::engine::rebuild::build_scoped_source_word_records_for_hkt(
        &engine,
        &make_batch(1, &[]),
        &plan,
        id,
    )
    .unwrap();
    assert!(!records.is_empty());
    assert!(records.iter().all(|r| r.word.as_deref() == Some("c")));
}

#[test]
fn reconstruction_failure_after_parent_acceptance_does_not_commit() {
    let mut config = crate::config::SecaConfig::default();
    config.hkt_builder.minimum_threshold_against_max_word_count = 0.7;
    config.seca_thresholds.alpha = 0.7;
    config.seca_thresholds.alpha_error_threshold = 0.11;
    config.seca_thresholds.beta_error_threshold = 0.2;
    config.seca_thresholds.word_importance_error_threshold = 0.2;
    let mut engine = crate::engine::SecaEngine::new(config).unwrap();
    let rows: Vec<_> = (0..20)
        .map(|i| {
            let topic = if i < 10 { "a" } else { "d" };
            let words = match i % 10 {
                0..=3 => vec![topic, if topic == "a" { "b" } else { "e" }],
                4..=7 => vec![topic, if topic == "a" { "c" } else { "f" }],
                _ => vec![topic],
            };
            (format!("s{i}"), words)
        })
        .collect();
    let refs: Vec<_> = rows
        .iter()
        .map(|(id, words)| (id.as_str(), words.as_slice()))
        .collect();
    engine.build_baseline_tree(make_batch(0, &refs)).unwrap();
    let rows: Vec<_> = (0..20)
        .map(|i| {
            let topic = if i < 10 { "a" } else { "d" };
            (
                format!("n{i}"),
                if i % 10 < 3 {
                    vec![topic, if topic == "a" { "x" } else { "y" }]
                } else {
                    vec![topic]
                },
            )
        })
        .collect();
    let refs: Vec<_> = rows
        .iter()
        .map(|(id, words)| (id.as_str(), words.as_slice()))
        .collect();
    let batch = make_batch(1, &refs);
    let mut candidate = engine.clone();
    let prepared = candidate.prepare_batch(batch.clone()).unwrap();
    let plan = candidate
        .evaluate_seca_trigger_plan_for_batch(&prepared)
        .unwrap();
    assert!(!plan.diagnostics[0].should_reconstruct);
    assert_eq!(plan.targets.len(), 2, "both disjoint children must trigger");
    assert!(!plan.targets.contains_key(&root_hkt_id(&engine)));

    // Force the builder boundary to fail, after vocabulary admission and parent mutation.
    engine.config.hkt_builder.similarity_threshold = 2.0;
    let before = serde_json::to_value(&engine).unwrap();
    let error = engine.process_batch(batch.clone()).unwrap_err();
    assert!(error.to_string().contains("similarity_threshold"));
    assert_eq!(serde_json::to_value(&engine).unwrap(), before);
    engine.config.hkt_builder.similarity_threshold = 0.5;
    let mut restarted = crate::engine::SecaEngine::load_snapshot(
        serde_json::from_value(serde_json::to_value(engine.snapshot().unwrap()).unwrap()).unwrap(),
    )
    .unwrap();
    let live = engine.process_batch(batch.clone()).unwrap();
    let continued = restarted.process_batch(batch).unwrap();
    assert_eq!(live.reconstructed_hkt_ids.len(), 2);
    assert!(live
        .reconstructed_hkt_ids
        .windows(2)
        .all(|pair| pair[0] < pair[1]));
    assert_eq!(
        serde_json::to_value(live).unwrap(),
        serde_json::to_value(continued).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&engine).unwrap(),
        serde_json::to_value(restarted).unwrap()
    );
}
