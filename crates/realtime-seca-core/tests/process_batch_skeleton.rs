use realtime_seca_core::{
    HktBuilderConfig, MemoryMode, SecaConfig, SecaEngine, SecaThresholdConfig, SourceBatch,
    SourceRecord,
};

fn base_config() -> SecaConfig {
    SecaConfig {
        hkt_builder: HktBuilderConfig {
            minimum_threshold_against_max_word_count: 0.5,
            similarity_threshold: 0.5,
            minimum_number_of_sources_to_create_branch_for_node: 1,
        },
        seca_thresholds: SecaThresholdConfig {
            alpha: 0.5,
            beta: 0.5,
            alpha_error_threshold: 0.1,
            beta_error_threshold: 0.1,
            word_importance_error_threshold: 0.1,
        },
        memory_mode: MemoryMode::Full,
        max_batches_in_memory: None,
    }
}

fn make_batch(batch_index: u32, rows: &[(&str, &[&str])]) -> SourceBatch {
    SourceBatch {
        batch_index,
        sources: rows
            .iter()
            .map(|(source_id, tokens)| SourceRecord {
                source_id: (*source_id).to_string(),
                batch_index,
                tokens: tokens.iter().map(|t| (*t).to_string()).collect(),
                text: None,
                timestamp_unix_ms: None,
                metadata: None,
            })
            .collect(),
    }
}

fn baseline_batch() -> SourceBatch {
    make_batch(
        0,
        &[
            ("s1", &["earthquake", "damage", "city"]),
            ("s2", &["earthquake", "rescue", "city"]),
            ("s3", &["storm", "damage", "coast"]),
        ],
    )
}

fn batch_one() -> SourceBatch {
    make_batch(
        1,
        &[
            ("s4", &["earthquake", "aftershock", "city"]),
            ("s5", &["rescue", "teams", "deployed"]),
        ],
    )
}

fn batch_two() -> SourceBatch {
    make_batch(
        2,
        &[
            ("s6", &["storm", "coast", "warning"]),
            ("s7", &["power", "outage", "city"]),
        ],
    )
}

fn batch_with_new_terms() -> SourceBatch {
    make_batch(
        1,
        &[
            ("s4", &["volcano", "ash", "plume"]),
            ("s5", &["evacuation", "zone", "ash"]),
        ],
    )
}

fn batch_three_new_terms() -> SourceBatch {
    make_batch(
        1,
        &[
            ("s4", &["volcano", "ash", "plume"]),
            ("s5", &["evacuation", "zone", "ash"]),
        ],
    )
}

#[test]
fn process_batch_requires_baseline_first() {
    let mut engine = SecaEngine::new(base_config()).unwrap();

    let error = engine.process_batch(batch_one()).unwrap_err();
    let message = error.to_string().to_lowercase();

    assert!(
        message.contains("baseline"),
        "expected baseline-related error, got: {message}"
    );
}

#[test]
fn process_batch_advances_last_processed_batch_index() {
    let mut engine = SecaEngine::new(base_config()).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    let result = engine.process_batch(batch_one()).unwrap();

    assert_eq!(result.batch_index, 1);
    assert_eq!(
        engine.snapshot().unwrap().last_processed_batch_index,
        Some(1)
    );

    let explanation = engine
        .explain_last_update()
        .expect("explanation should exist");
    assert!(explanation
        .reason_codes
        .iter()
        .any(|code| code == "INCREMENTAL_BATCH_PROCESSED"));
}

#[test]
fn process_batch_rejects_out_of_sequence_batch_index() {
    let mut engine = SecaEngine::new(base_config()).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    let error = engine.process_batch(batch_two()).unwrap_err();
    let message = error.to_string().to_lowercase();

    assert!(
        message.contains("out of sequence") || message.contains("expected"),
        "expected sequence error, got: {message}"
    );
}

#[test]
fn process_batch_in_full_mode_keeps_all_batches() {
    let mut engine = SecaEngine::new(base_config()).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    engine.process_batch(batch_one()).unwrap();
    engine.process_batch(batch_two()).unwrap();

    // baseline + batch1 + batch2
    assert_eq!(engine.stored_batch_count(), 3);
}

#[test]
fn process_batch_in_sliding_window_trims_old_batches() {
    let mut config = base_config();
    config.memory_mode = MemoryMode::SlidingWindow;
    config.max_batches_in_memory = Some(2);

    let mut engine = SecaEngine::new(config).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    // After baseline: [0]
    assert_eq!(engine.stored_batch_count(), 1);

    // After batch 1: [0,1]
    engine.process_batch(batch_one()).unwrap();
    assert_eq!(engine.stored_batch_count(), 2);

    // After batch 2: should trim to 2 (likely [1,2])
    engine.process_batch(batch_two()).unwrap();
    assert_eq!(engine.stored_batch_count(), 2);
}

#[test]
fn process_batch_sliding_window_zero_max_batches_errors() {
    let mut config = base_config();
    config.memory_mode = MemoryMode::SlidingWindow;
    config.max_batches_in_memory = Some(0);

    let mut engine = SecaEngine::new(config).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    let error = engine.process_batch(batch_one()).unwrap_err();
    let message = error.to_string().to_lowercase();

    assert!(
        message.contains("max_batches_in_memory") || message.contains("must be > 0"),
        "expected invalid sliding window configuration error, got: {message}"
    );
}

#[test]
fn process_batch_computes_word_stats_summary() {
    let mut engine = SecaEngine::new(base_config()).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    let result = engine.process_batch(batch_one()).unwrap();
    assert_eq!(result.batch_index, 1);

    let stats = engine
        .last_batch_word_stats_summary()
        .expect("stats summary should be present");

    assert_eq!(stats.total_sources_in_batch, 2);
    assert!(stats.unique_words_in_batch >= 1);
    assert!(stats.max_word_document_frequency >= 1);

    let explanation = engine.explain_last_update().unwrap();
    assert!(
        explanation
            .reason_codes
            .iter()
            .any(|code| code == "BATCH_WORD_STATS_COMPUTED"),
        "expected BATCH_WORD_STATS_COMPUTED reason code"
    );
}

#[test]
fn process_batch_detects_new_words_relative_to_baseline() {
    let mut engine = SecaEngine::new(base_config()).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    engine.process_batch(batch_with_new_terms()).unwrap();

    let stats = engine.last_batch_word_stats_summary().unwrap();
    assert!(stats.new_words_in_batch >= 1);
    assert!(stats.unique_words_in_batch >= stats.new_words_in_batch);
}

#[test]
fn process_batch_evaluates_trigger_and_sets_reason_code() {
    let mut engine = SecaEngine::new(base_config()).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    let _result = engine.process_batch(batch_one()).unwrap();

    let explanation = engine.explain_last_update().unwrap();
    assert!(
        explanation
            .reason_codes
            .iter()
            .any(|code| code == "SECA_TRIGGER_EVALUATED"),
        "expected SECA_TRIGGER_EVALUATED reason code"
    );

    assert!(
        explanation.reason_codes.iter().any(|code| {
            code == "SECA_RECONSTRUCTION_TRIGGERED" || code == "SECA_RECONSTRUCTION_SKIPPED"
        }),
        "expected reconstruction trigger decision reason code"
    );
}

#[test]
fn process_batch_includes_stage2_scope_reason_codes() {
    let mut engine = SecaEngine::new(base_config()).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    let _result = engine.process_batch(batch_one()).unwrap();

    let explanation = engine.explain_last_update().unwrap();

    assert!(
        explanation
            .reason_codes
            .iter()
            .any(|code| code == "SECA_SCOPE_MAPPING_COMPLETED"),
        "expected SECA_SCOPE_MAPPING_COMPLETED reason code"
    );

    assert!(
        explanation
            .reason_codes
            .iter()
            .any(|code| code == "SECA_SCOPE_METRICS_COMPUTED"),
        "expected SECA_SCOPE_METRICS_COMPUTED reason code"
    );
}

#[test]
fn process_batch_notes_include_recursive_trigger_plan_messages() {
    let mut engine = SecaEngine::new(base_config()).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    let result = engine.process_batch(batch_one()).unwrap();

    assert!(
        result
            .notes
            .iter()
            .any(|note| { note.contains("SECA recursive trigger evaluation started") }),
        "expected recursive trigger plan start note"
    );

    assert!(
        result
            .notes
            .iter()
            .any(|note| note.contains("SECA trigger plan:")),
        "expected trigger plan summary note"
    );
}

#[test]
fn process_batch_can_skip_reconstruction_with_relaxed_thresholds() {
    let mut config = base_config();
    config.seca_thresholds.alpha = 1.0;
    config.seca_thresholds.beta = 0.0;
    config.seca_thresholds.alpha_error_threshold = 1.0;
    config.seca_thresholds.beta_error_threshold = 1.0;
    config.seca_thresholds.word_importance_error_threshold = 1.0;

    let mut engine = SecaEngine::new(config).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    let result = engine.process_batch(batch_one()).unwrap();
    assert!(
        !result.reconstruction_triggered,
        "expected no trigger under very relaxed thresholds"
    );

    let explanation = engine.explain_last_update().unwrap();
    assert!(explanation
        .reason_codes
        .iter()
        .any(|code| code == "SECA_RECONSTRUCTION_SKIPPED"));
}

#[test]
fn process_batch_can_trigger_reconstruction_with_strict_thresholds() {
    let mut config = base_config();
    config.seca_thresholds.alpha = 0.0;
    config.seca_thresholds.beta = 0.0;
    config.seca_thresholds.alpha_error_threshold = 0.0;
    config.seca_thresholds.beta_error_threshold = 0.0;
    config.seca_thresholds.word_importance_error_threshold = 0.0;

    let mut engine = SecaEngine::new(config).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    let result = engine.process_batch(batch_one()).unwrap();
    assert!(
        result.reconstruction_triggered,
        "expected trigger under strict thresholds"
    );

    let explanation = engine.explain_last_update().unwrap();
    assert!(explanation
        .reason_codes
        .iter()
        .any(|code| code == "SECA_RECONSTRUCTION_TRIGGERED"));
}

#[test]
fn process_batch_no_trigger_keeps_tree_unchanged() {
    let mut config = base_config();
    config.seca_thresholds.alpha = 1.0;
    config.seca_thresholds.beta = 0.0;
    config.seca_thresholds.alpha_error_threshold = 1.0;
    config.seca_thresholds.beta_error_threshold = 1.0;
    config.seca_thresholds.word_importance_error_threshold = 1.0;

    let mut engine = SecaEngine::new(config).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    let before =
        serde_json::to_string_pretty(&engine.export_baseline_tree_verbose().unwrap()).unwrap();

    let result = engine.process_batch(batch_one()).unwrap();
    assert!(!result.reconstruction_triggered);

    let after =
        serde_json::to_string_pretty(&engine.export_baseline_tree_verbose().unwrap()).unwrap();

    assert_eq!(
        before, after,
        "tree should remain unchanged when no rebuild is triggered"
    );
}

#[test]
fn process_batch_trigger_rebuilds_tree_from_stored_batches() {
    let mut config = base_config();
    config.seca_thresholds.alpha = 0.0;
    config.seca_thresholds.beta = 0.0;
    config.seca_thresholds.alpha_error_threshold = 0.0;
    config.seca_thresholds.beta_error_threshold = 0.0;
    config.seca_thresholds.word_importance_error_threshold = 0.0;

    let mut engine = SecaEngine::new(config).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    let before =
        serde_json::to_string_pretty(&engine.export_baseline_tree_verbose().unwrap()).unwrap();

    let result = engine.process_batch(batch_three_new_terms()).unwrap();
    assert!(result.reconstruction_triggered);

    let after_tree = engine.export_baseline_tree_verbose().unwrap();
    let after = serde_json::to_string_pretty(&after_tree).unwrap();

    assert_ne!(before, after, "tree should change after triggered rebuild");

    let explanation = engine.explain_last_update().unwrap();
    assert!(explanation
        .reason_codes
        .iter()
        .any(|c| c == "SECA_FULL_REBUILD_EXECUTED"));

    // Optional stronger assertion: new vocabulary should appear in legends/union words
    let all_tokens: std::collections::BTreeSet<String> = after_tree
        .word_legend
        .iter()
        .filter_map(|entry| entry.token.clone())
        .collect();

    assert!(all_tokens.contains("volcano") || all_tokens.contains("ash"));
}

#[test]
fn process_batch_trigger_path_notes_mention_rebuild_completed() {
    let mut config = base_config();
    config.seca_thresholds.alpha = 0.0;
    config.seca_thresholds.beta = 0.0;
    config.seca_thresholds.alpha_error_threshold = 0.0;
    config.seca_thresholds.beta_error_threshold = 0.0;
    config.seca_thresholds.word_importance_error_threshold = 0.0;

    let mut engine = SecaEngine::new(config).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    let result = engine.process_batch(batch_three_new_terms()).unwrap();

    assert!(result.reconstruction_triggered);
    assert!(
        result
            .notes
            .iter()
            .any(|note| note.contains("full rebuild from stored batches completed")),
        "expected rebuild completion note in trigger path"
    );
}

#[test]
fn process_batch_no_trigger_path_notes_mention_no_rebuild() {
    let mut config = base_config();
    config.seca_thresholds.alpha = 1.0;
    config.seca_thresholds.beta = 0.0;
    config.seca_thresholds.alpha_error_threshold = 1.0;
    config.seca_thresholds.beta_error_threshold = 1.0;
    config.seca_thresholds.word_importance_error_threshold = 1.0;

    let mut engine = SecaEngine::new(config).unwrap();
    engine.build_baseline_tree(baseline_batch()).unwrap();

    let result = engine.process_batch(batch_one()).unwrap();

    assert!(!result.reconstruction_triggered);
    assert!(
        result
            .notes
            .iter()
            .any(|note| note.contains("no rebuild performed")),
        "expected no-rebuild note in non-trigger path"
    );
}

#[test]
fn process_batch_rebuild_path_is_deterministic_for_same_sequence() {
    let mut config = base_config();
    config.seca_thresholds.alpha = 0.0;
    config.seca_thresholds.beta = 0.0;
    config.seca_thresholds.alpha_error_threshold = 0.0;
    config.seca_thresholds.beta_error_threshold = 0.0;
    config.seca_thresholds.word_importance_error_threshold = 0.0;

    let mut engine_a = SecaEngine::new(config.clone()).unwrap();
    engine_a.build_baseline_tree(baseline_batch()).unwrap();
    engine_a.process_batch(batch_one()).unwrap();
    let final_a =
        serde_json::to_string_pretty(&engine_a.export_baseline_tree_verbose().unwrap()).unwrap();

    let mut engine_b = SecaEngine::new(config).unwrap();
    engine_b.build_baseline_tree(baseline_batch()).unwrap();
    engine_b.process_batch(batch_one()).unwrap();
    let final_b =
        serde_json::to_string_pretty(&engine_b.export_baseline_tree_verbose().unwrap()).unwrap();

    assert_eq!(
        final_a, final_b,
        "final rebuilt tree should be deterministic"
    );
}
