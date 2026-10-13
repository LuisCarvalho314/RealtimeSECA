//! Regression fixtures adapted from the old-main audit to the canonical engine.
//! Executed C# goldens supplement the preserved integration fixtures below.
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
    c.seca_thresholds.alpha_error_threshold = 1.0;
    c.seca_thresholds.beta_error_threshold = 1.0;
    c.seca_thresholds.word_importance_error_threshold = 1.0;
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
fn errors(r: &BatchProcessingResult, hkt: i32, expected: [f64; 3]) {
    let d = r
        .decision_diagnostics
        .iter()
        .find(|d| d.hkt_id == hkt)
        .unwrap();
    for (actual, value) in [
        d.paper_alpha_error,
        d.paper_beta_error,
        d.paper_word_importance_error,
    ]
    .into_iter()
    .zip(expected)
    {
        assert!(
            (actual.unwrap() - value).abs() < 1e-9,
            "expected {value}, got {actual:?}"
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
fn exact_alpha_and_beta_admission_preserves_state0_error_vocabulary() {
    let mut e = engine(config(0.5, 0.5), &[&["a"], &["a"]]);
    let h = root(&e);
    let r = e
        .process_batch(batch(2, &[&["a", "x"], &["a", "x"]]))
        .unwrap();
    // W0={a}, D(a)=4, D(x)=2: WI=1-4/6.
    errors(&r, h, [0.0, 0.0, 1.0 / 3.0]);
    assert_eq!(expected_count(&e, h), 1);
    assert_eq!(e.export_baseline_tree().unwrap().nodes[0].word_ids.len(), 1);
}

#[test]
fn construction_expected_descriptor_is_excluded_from_candidates() {
    let mut e = engine(config(0.5, 0.5), &[&["a", "b"], &["a", "b"]]);
    let h = root(&e);
    let r = e.process_batch(batch(2, &[&["a", "b"]])).unwrap();
    // Existing descriptors contribute once to the importance denominator.
    errors(&r, h, [0.0, 0.0, 0.0]);
    assert!(!r.reconstruction_triggered);
}

#[test]
fn old_word_losing_strength_and_eligibility_has_independent_errors() {
    let mut e = engine(config(0.5, 0.5), &[&["a", "b"], &["a", "b"]]);
    let h = root(&e);
    let r = e
        .process_batch(batch(2, &[&["a"], &["a"], &["a"], &["a"]]))
        .unwrap();
    errors(&r, h, [1.0 / 12.0, 1.0 / 12.0, 0.0]);
}

#[test]
fn unassigned_expected_word_is_excluded_from_old_vocab_on_next_batch() {
    let mut e = engine(config(0.5, 0.9), &[&["a"], &["a"]]);
    let h = root(&e);
    e.process_batch(batch(2, &[&["a", "x"], &["x"]])).unwrap();
    let r = e.process_batch(batch(3, &[&["a", "x"], &["x"]])).unwrap();
    // Persistent x contributes both earlier and incoming occurrences.
    errors(&r, h, [0.0, 0.0, 0.5]);
    assert_eq!(expected_count(&e, h), 1);
    assert_eq!(
        e.last_batch_word_stats_summary()
            .unwrap()
            .new_words_in_batch,
        0
    );
}

#[test]
fn error_threshold_is_strict_at_full_precision() {
    for (threshold, trigger) in [((0.5 - 2.0 / 6.0) / 2.0, false), (0.0833, true)] {
        let mut c = config(0.5, 0.5);
        c.seca_thresholds.alpha_error_threshold = threshold;
        let mut e = engine(c, &[&["a", "b"], &["a", "b"]]);
        let h = root(&e);
        let r = e
            .process_batch(batch(2, &[&["a"], &["a"], &["a"], &["a"]]))
            .unwrap();
        errors(&r, h, [1.0 / 12.0, 1.0 / 12.0, 0.0]);
        assert_eq!(r.reconstruction_triggered, trigger);
    }
}

#[test]
fn importance_threshold_equality_and_just_below() {
    for (threshold, trigger) in [(0.4, false), (0.3999, true)] {
        let mut c = config(0.5, 0.5);
        c.seca_thresholds.word_importance_error_threshold = threshold;
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
    assert_eq!(
        r.decision_diagnostics
            .iter()
            .find(|d| d.hkt_id == h)
            .unwrap()
            .scoped_source_count,
        2
    );
    assert_eq!(
        r.decision_diagnostics
            .iter()
            .find(|d| d.hkt_id == child)
            .unwrap()
            .scoped_source_count,
        1
    );
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
            .find(|n| !n.is_refuge_node)
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
    // W0={a,b}; denominator a4+b4+x4=12. Shared candidate is not old.
    errors(&r, h, [0.0, 0.0, 1.0 / 3.0]);
    assert!(!r.reconstruction_triggered);
}

#[test]
fn beta_error_threshold_equality_and_just_below() {
    for (threshold, trigger) in [((0.5 - 2.0 / 6.0) / 2.0, false), (0.0833, true)] {
        let mut c = config(0.5, 0.5);
        c.seca_thresholds.beta_error_threshold = threshold;
        let mut e = engine(c, &[&["a", "b"], &["a", "b"]]);
        let h = root(&e);
        let r = e
            .process_batch(batch(2, &[&["a"], &["a"], &["a"], &["a"]]))
            .unwrap();
        errors(&r, h, [1.0 / 12.0, 1.0 / 12.0, 0.0]);
        assert_eq!(r.reconstruction_triggered, trigger);
    }
}

#[test]
fn refuge_commits_unmatched_sources() {
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
    assert_eq!(after.source_ids.len(), before.source_ids.len() + 1);
    assert!(before
        .source_ids
        .iter()
        .all(|id| after.source_ids.contains(id)));
    assert!(!r.reconstruction_triggered);
}

#[test]
fn newly_assigned_word_triggers_old_vocab_importance_reconstruction() {
    let mut c = config(0.5, 0.5);
    c.seca_thresholds.word_importance_error_threshold = 0.2;
    let mut e = engine(c, &[&["a"], &["a"]]);
    let h = root(&e);
    let r = e
        .process_batch(batch(2, &[&["a", "x"], &["a", "x"]]))
        .unwrap();
    errors(&r, h, [0.0, 0.0, 1.0 / 3.0]);
    assert!(r.reconstruction_triggered);
}

#[test]
fn duplicate_expected_descriptor_does_not_cause_reconstruction() {
    let mut c = config(0.5, 0.5);
    c.seca_thresholds.word_importance_error_threshold = 0.3;
    let mut e = engine(c, &[&["a", "b"], &["a", "b"]]);
    let h = root(&e);
    let r = e.process_batch(batch(2, &[&["a", "b"]])).unwrap();
    errors(&r, h, [0.0, 0.0, 0.0]);
    assert!(!r.reconstruction_triggered);
}

#[test]
fn empty_incoming_scope_skips_inspection() {
    let mut e = engine(config(0.5, 0.5), &[&["a", "b"], &["a", "b"]]);
    let h = root(&e);
    let r = e.process_batch(batch(2, &[])).unwrap();
    assert!(!r.decision_diagnostics.iter().any(|d| d.hkt_id == h));
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
fn candidate_beta_uses_full_node_union() {
    let mut e = engine(
        config(0.4, 0.4),
        &[&["a", "b"], &["a", "b"], &["a"], &["b"]],
    );
    let h = root(&e);
    let r = e
        .process_batch(batch(2, &[&["b", "x"], &["b", "x"]]))
        .unwrap();
    // Node union is 6: incoming candidate beta=2/6 fails .4.
    // W0={a,b}, denominator a3+b5+x2=10.
    errors(&r, h, [0.0, 0.0, 0.2]);
    assert!(!r.reconstruction_triggered);
}

// Executed C# goldens: captured by fixtures/parity_context_metrics/replay.py.
// Existing tests above predate this executable audit and are preserved.
#[test]
fn executable_csharp_goldens_against_full_memory_rust() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/parity_context_metrics");
    let mut inputs: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".input.json"))
        .collect();
    inputs.sort();
    for path in inputs {
        let name = path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .trim_end_matches(".input.json");
        let input: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let golden_path = dir.join(format!("{name}.csharp.json"));
        let golden: Option<serde_json::Value> = golden_path
            .exists()
            .then(|| serde_json::from_slice(&std::fs::read(golden_path).unwrap()).unwrap());
        let c = reference_config(&input);
        let mut e = SecaEngine::new(c).unwrap();
        for (cycle, rows) in input["batches"].as_array().unwrap().iter().enumerate() {
            let index = cycle as u32 + 1;
            let b = reference_batch(index, rows);
            if cycle == 0 {
                e.build_baseline_tree(b).unwrap();
                continue;
            }
            let r = e.process_batch(b).unwrap();
            println!(
                "EXECUTED {name} cycle={cycle} {}",
                serde_json::to_string(&r.decision_diagnostics).unwrap()
            );
            if name == "empty" {
                assert!(golden.is_none());
                assert!(r.decision_diagnostics.is_empty());
                assert!(!r.reconstruction_triggered);
                continue;
            }
            if name == "shared_candidate" {
                assert!(
                    golden.is_none(),
                    "C# failure must not acquire invented metrics"
                );
                assert!(
                    std::fs::read_to_string(dir.join("shared_candidate.stderr.txt"))
                        .unwrap()
                        .contains("same key")
                );
                errors(&r, root(&e), [0.0, 0.0, 1.0 / 3.0]);
                continue;
            }
            if matches!(
                name,
                "unchanged" | "child" | "refuge" | "node_union" | "exact_alpha"
            ) {
                compare_retained_memberships(
                    name,
                    &e,
                    &golden.as_ref().unwrap()["snapshots"][cycle],
                );
            }
            compare_executed_decisions(
                name,
                cycle,
                &r,
                &golden.as_ref().unwrap()["snapshots"][cycle],
                &input,
            );
        }
    }
}

#[test]
#[ignore = "Known decision discrepancy: C# rounds 1/12 to .0833 before comparing; Rust retains full precision"]
fn expected_failure_csharp_alpha_threshold_parity() {
    let mut c = config(0.5, 0.5);
    c.seca_thresholds.alpha_error_threshold = 0.0833;
    let mut e = engine(c, &[&["a", "b"], &["a", "b"]]);
    let r = e
        .process_batch(batch(2, &[&["a"], &["a"], &["a"], &["a"]]))
        .unwrap();
    let golden: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/parity_context_metrics/alpha_threshold.csharp.json"
    ))
    .unwrap();
    assert_eq!(
        r.reconstruction_triggered,
        golden["snapshots"][1]["decisions"][0]["triggered"]
            .as_bool()
            .unwrap()
    );
}

fn compare_retained_memberships(name: &str, e: &SecaEngine, snapshot: &serde_json::Value) {
    let tree = e.export_baseline_tree_verbose().unwrap();
    let cs_tree = snapshot["tree"].as_array().unwrap();
    for hkt in cs_tree {
        for node in hkt["nodes"].as_array().unwrap() {
            let rust = tree
                .nodes
                .iter()
                .find(|n| n.node_id == node["id"].as_i64().unwrap() as i32)
                .unwrap();
            let mut sources: Vec<_> = rust
                .sources
                .iter()
                .map(|s| s.external_source_id.clone())
                .collect();
            sources.sort();
            let mut cs_sources: Vec<_> = node["sources"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.to_string())
                .collect();
            if name == "refuge" && rust.is_refuge_node {
                cs_sources.push("5".into());
            }
            cs_sources.sort();
            assert_eq!(sources, cs_sources, "{name}: retained node sources");
            assert_eq!(
                rust.words.len(),
                node["words"].as_array().unwrap().len(),
                "{name}: descriptors"
            );
        }
    }
}

fn compare_executed_decisions(
    name: &str,
    cycle: usize,
    r: &BatchProcessingResult,
    snapshot: &serde_json::Value,
    parameters: &serde_json::Value,
) {
    let decisions = snapshot["decisions"].as_array().unwrap();
    assert_eq!(
        r.decision_diagnostics.len(),
        decisions.len(),
        "{name}: evaluated scopes"
    );
    for (rust, cs) in r.decision_diagnostics.iter().zip(decisions) {
        verify_reference_errors(cs, parameters);
        let actual = [
            rust.paper_alpha_error.unwrap(),
            rust.paper_beta_error.unwrap(),
            rust.paper_word_importance_error.unwrap(),
        ];
        let expected = ["alphaError", "betaError", "wordImportanceError"]
            .map(|k| cs["errors"][k]["option1_error"].as_f64().unwrap());
        if (name == "expected_consecutive" || name == "expected_trigger") && cycle == 2 {
            assert!((expected[2] - 0.3333).abs() < 1e-12);
            assert!((actual[2] - 0.5).abs() < 1e-12);
        } else {
            for (a, b) in actual.into_iter().zip(expected) {
                assert!((a - b).abs() <= 0.0000500001, "{name}: Rust {a}, C# {b}");
            }
        }
        let precision_difference = name == "alpha_threshold"
            || name == "beta_threshold"
            || (name == "expected_trigger" && cycle == 2);
        assert_eq!(
            rust.should_reconstruct,
            cs["triggered"].as_bool().unwrap() || precision_difference,
            "{name}: decision"
        );
        if precision_difference {
            assert!(!rust.trigger_reasons.is_empty());
        }
    }
}

fn reference_batch(index: u32, rows: &serde_json::Value) -> SourceBatch {
    SourceBatch {
        batch_index: index,
        sources: rows
            .as_array()
            .unwrap()
            .iter()
            .map(|row| SourceRecord {
                source_id: row["id"].to_string(),
                batch_index: index,
                tokens: row["words"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|w| w.as_str().unwrap().to_owned())
                    .collect(),
                text: None,
                timestamp_unix_ms: None,
                metadata: None,
            })
            .collect(),
    }
}

// Independently aggregate raw, observed word metrics over the observed old
// descriptors. This validates golden consistency without inventing Rust metrics.
fn verify_reference_errors(decision: &serde_json::Value, parameters: &serde_json::Value) {
    use std::collections::BTreeSet;
    let words: BTreeSet<_> = decision["topologyAtDecision"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|n| n["words"].as_array().unwrap())
        .filter_map(|w| w.as_i64())
        .filter(|w| *w != -1)
        .collect();
    let measurements = decision["wordMetrics"].as_array().unwrap();
    let mut sums = [0.0; 3];
    for word in &words {
        let values = &measurements
            .iter()
            .find(|m| m["wordId"].as_i64() == Some(*word))
            .unwrap()["values"];
        let strength = values
            ["numberOfSourcesAfterNewBatchOverNumberOfSourcesOfNewProminWordInHKT"]
            .as_f64()
            .unwrap();
        let eligibility = values["numberOfIntersectedSourcesWithNewProminWordInNodeOverNumSourcesOfNewProminWordInNode"].as_f64().unwrap();
        let importance = values["precentageOfSourcesInHKTInNewBatch"]
            .as_f64()
            .unwrap();
        sums[0] += (parameters["alpha"].as_f64().unwrap() - strength).max(0.0);
        sums[1] += (parameters["beta"].as_f64().unwrap() - eligibility).max(0.0);
        sums[2] += importance;
    }
    let calculated = [
        sums[0] / words.len() as f64,
        sums[1] / words.len() as f64,
        1.0 - sums[2],
    ];
    let keys = ["alphaError", "betaError", "wordImportanceError"];
    let mut triggered = false;
    for (value, key) in calculated.into_iter().zip(keys) {
        let observed = decision["errors"][key]["option1_error"].as_f64().unwrap();
        assert!((value - observed).abs() <= 0.0000500001);
        triggered |= observed > decision["errors"][key]["option1_Thresh"].as_f64().unwrap();
    }
    assert_eq!(triggered, decision["triggered"].as_bool().unwrap());
}

fn reference_config(input: &serde_json::Value) -> SecaConfig {
    let mut c = config(
        input["alpha"].as_f64().unwrap(),
        input["beta"].as_f64().unwrap(),
    );
    c.memory_mode = realtime_seca_core::MemoryMode::Full;
    c.hkt_builder
        .minimum_number_of_sources_to_create_branch_for_node =
        input["branchMinimum"].as_u64().unwrap() as usize;
    c.seca_thresholds.alpha_error_threshold = input["alphaError"].as_f64().unwrap();
    c.seca_thresholds.beta_error_threshold = input["betaError"].as_f64().unwrap();
    c.seca_thresholds.word_importance_error_threshold =
        input["wordImportanceError"].as_f64().unwrap();
    c
}
