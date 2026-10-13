//! Characterization of executable behavior, not approval of unresolved semantics.
use realtime_seca_core::{SecaConfig, SecaEngine, SourceBatch, SourceRecord};
use serde_json::{json, Value};
use std::{collections::BTreeSet, path::PathBuf};

fn directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scientific_context")
}

fn fixture_config(input: &Value) -> SecaConfig {
    let mut config = SecaConfig::default();
    config.hkt_builder.minimum_threshold_against_max_word_count = input["alpha"].as_f64().unwrap();
    config.hkt_builder.similarity_threshold = input["beta"].as_f64().unwrap();
    config
        .hkt_builder
        .minimum_number_of_sources_to_create_branch_for_node = 1000;
    config.seca_thresholds.alpha = input["alpha"].as_f64().unwrap();
    config.seca_thresholds.beta = input["beta"].as_f64().unwrap();
    config.seca_thresholds.alpha_error_threshold = 1.0;
    config.seca_thresholds.beta_error_threshold = 1.0;
    config.seca_thresholds.word_importance_error_threshold =
        input["wordImportanceError"].as_f64().unwrap();
    config
}

fn fixture_batch(i: usize, rows: &Value) -> SourceBatch {
    SourceBatch {
        batch_index: i as u32 + 1,
        sources: rows
            .as_array()
            .unwrap()
            .iter()
            .map(|row| SourceRecord {
                source_id: row["id"].to_string(),
                batch_index: i as u32 + 1,
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

#[test]
fn characterize_sequences_without_changing_production_contract() {
    for name in [
        "shared_asymmetric",
        "expected_history",
        "expected_trigger",
        "historical_beta",
        "historical_beta_trigger",
    ] {
        let input: Value = serde_json::from_str(
            &std::fs::read_to_string(directory().join(format!("{name}.input.json"))).unwrap(),
        )
        .unwrap();
        let config = fixture_config(&input);
        let mut engine = SecaEngine::new(config).unwrap();
        let mut observed = Vec::new();
        for (i, rows) in input["batches"].as_array().unwrap().iter().enumerate() {
            let batch = fixture_batch(i, rows);
            if i == 0 {
                engine.build_baseline_tree(batch).unwrap();
            } else {
                let report = engine.process_batch(batch).unwrap();
                let expected_wi: &[f64] = match name {
                    "shared_asymmetric" => &[1.0 / 3.0, 7.0 / 19.0],
                    "expected_history" => &[2.0 / 5.0, 1.0 / 2.0, 6.0 / 11.0, 4.0 / 7.0],
                    "expected_trigger" => &[2.0 / 5.0, 1.0 / 2.0],
                    "historical_beta" => &[2.0 / 5.0, 5.0 / 12.0, 3.0 / 7.0, 7.0 / 16.0],
                    "historical_beta_trigger" => &[2.0 / 5.0, 0.0, 0.0, 0.0],
                    _ => unreachable!(),
                };
                assert_eq!(report.decision_diagnostics.len(), 1);
                let diagnostic = &report.decision_diagnostics[0];
                assert_eq!(diagnostic.paper_alpha_error, Some(0.0));
                assert_eq!(diagnostic.paper_beta_error, Some(0.0));
                assert!(
                    (diagnostic.paper_word_importance_error.unwrap() - expected_wi[i - 1]).abs()
                        < 1e-12
                );
                observed.push(json!({"cycle": i+1,"triggered":report.reconstruction_triggered,"diagnostics":report.decision_diagnostics,"tree":engine.export_baseline_tree_verbose().unwrap()}));
            }
        }
        let actual = json!(observed);
        if std::env::var_os("CAPTURE_SCIENTIFIC_CONTEXT").is_some() {
            std::fs::write(
                directory().join(format!("{name}.rust.json")),
                serde_json::to_string_pretty(&actual).unwrap() + "\n",
            )
            .unwrap();
        } else {
            let expected: Value = serde_json::from_str(
                &std::fs::read_to_string(directory().join(format!("{name}.rust.json"))).unwrap(),
            )
            .unwrap();
            assert_eq!(actual, expected, "{name}");
        }
    }
}

#[test]
fn asymmetric_source_sets_separate_word_and_incidence_contracts() {
    let z1: BTreeSet<_> = (1..=6).collect();
    let z2: BTreeSet<_> = [4, 5, 6, 7].into_iter().collect();
    let word: BTreeSet<_> = [1, 4, 7].into_iter().collect();
    let left = word.intersection(&z1).count();
    let right = word.intersection(&z2).count();
    assert_eq!((left, right, word.len()), (2, 2, 3));
    // Shared source 4 contributes once to HKT DF, twice to descriptor incidences.
    let union: BTreeSet<_> = word
        .intersection(&z1)
        .chain(word.intersection(&z2))
        .copied()
        .collect();
    assert_eq!(union.len(), 3);
    let importance_union = 3.0 / (6.0 + 4.0 + 3.0);
    let importance_last_node = 2.0 / (6.0 + 4.0 + 2.0 + 2.0);
    let wi_union: f64 = 1.0 - (6.0 + 4.0) / 13.0 - importance_union;
    let wi_last = 1.0 - (6.0 + 4.0) / 14.0 - importance_last_node;
    assert!(wi_union.abs() < 1e-12);
    assert!(wi_last > 0.1);
    // Worst-node deficit .6-2/6 differs from mean-node and last-node deficits.
    let worst = 0.6 - 2.0 / 6.0;
    let mean = ((0.6 - 2.0 / 6.0) + (0.6 - 2.0 / 4.0)) / 2.0;
    assert!(worst / 3.0 > 0.07 && mean / 3.0 < 0.07);
}

#[test]
fn retained_beta_passes_every_later_cycle_where_incoming_beta_fails() {
    for added in 2..=5 {
        let node_count = 4 + added;
        let historical_x = 2 + added;
        let incoming_x = if added == 2 { 2 } else { 1 };
        assert!(historical_x as f64 / node_count as f64 >= 0.5);
        assert!((incoming_x as f64 / node_count as f64) < 0.5);
    }
}

#[test]
fn vocabulary_and_persistent_assignment_alternatives_change_future_decisions() {
    // Expected-history cycle3: a4,x4; descriptor W0={a} versus {a,x}.
    let descriptor_wi = 1.0 - 4.0 / 8.0;
    let expectation_wi = 1.0 - (4.0 + 4.0) / 8.0;
    assert!(descriptor_wi > 0.4 && expectation_wi <= 0.4);
    // Historical-beta: successful persistent assignment makes x old next cycle.
    for (a, x) in [(7.0, 5.0), (8.0, 6.0), (9.0, 7.0)] {
        let incoming_only_wi = 1.0 - a / (a + x);
        let persisted_assignment_wi: f64 = 1.0 - (a + x) / (a + x);
        assert!(incoming_only_wi > 0.41);
        assert!(persisted_assignment_wi.abs() < 1e-12);
    }
    // Scoped versus adopted-only DF in a sibling containing unadopted x.
    let scoped_strength = 3.0 / 3.0;
    let adopted_strength = 1.0 / 3.0;
    assert!(scoped_strength >= 0.5);
    assert!((0.5 - adopted_strength) / 3.0 > 0.04);
}
