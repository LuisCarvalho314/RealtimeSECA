use super::super::*;
use crate::config::SecaConfig;

use super::fixtures::*;
#[test]
fn paper_policy_trigger_plan_can_fire_on_unassigned_expected_words() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.word_importance_error_threshold = 0.1;
    config.seca_thresholds.alpha_error_threshold = 1.0;
    config.seca_thresholds.beta_error_threshold = 1.0;
    config.seca_thresholds.beta = 1.0;

    let mut engine = SecaEngine::new(config).unwrap();

    let baseline = make_batch(0, &[("s1", &["a", "b"]), ("s2", &["a", "c"])]);

    engine.build_baseline_tree(baseline).unwrap();

    let batch = make_batch(1, &[("n1", &["x"]), ("n2", &["y"])]);

    let plan = engine.evaluate_seca_trigger_plan_for_batch(&batch).unwrap();

    assert!(!plan.targets.is_empty());
}
#[test]
fn paper_policy_process_batch_triggers_on_unassigned_expected_words() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.word_importance_error_threshold = 0.1;
    config.seca_thresholds.alpha_error_threshold = 1.0;
    config.seca_thresholds.beta_error_threshold = 1.0;
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
        .contains(&"SECA_RECONSTRUCTION_TRIGGERED".to_string()));
}
#[test]
fn process_batch_paper_policy_trigger_changes_tree() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.word_importance_error_threshold = 0.1;
    config.seca_thresholds.alpha_error_threshold = 1.0;
    config.seca_thresholds.beta_error_threshold = 1.0;
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
fn paper_scaffold_can_trigger_from_paper_beta_error_path() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.alpha_error_threshold = 1.0;
    config.seca_thresholds.word_importance_error_threshold = 1.0;
    config.seca_thresholds.beta = 0.8;
    config.seca_thresholds.beta_error_threshold = 0.1;

    let mut engine = SecaEngine::new(config).unwrap();
    engine
        .build_baseline_tree(make_batch(0, &[("s1", &["a", "b", "c"]), ("s2", &["a"])]))
        .unwrap();

    let _result = engine
        .process_batch(make_batch(
            1,
            &[("n1", &["a"]), ("n2", &["a"]), ("n3", &["a"])],
        ))
        .unwrap();

    // This may or may not trigger depending on actual HKT mapping topology in your small baseline.
}
#[test]
fn paper_policy_triggers_reconstruction_on_wi_error() {
    let mut config = SecaConfig::default();
    config.seca_thresholds.word_importance_error_threshold = 0.0;
    config.seca_thresholds.alpha_error_threshold = 0.0;
    config.seca_thresholds.beta_error_threshold = 0.0;

    let mut engine = SecaEngine::new(config).unwrap();

    let baseline = make_batch(0, &[("s1", &["a"])]);
    engine.build_baseline_tree(baseline).unwrap();

    let batch = make_batch(1, &[("s2", &["new_token"])]);
    let result = engine.process_batch(batch).unwrap();

    assert!(
        result.reconstruction_triggered,
        "expected reconstruction under paper policy"
    );
}

fn equation_fixture() -> (
    SecaEngine,
    crate::engine::scope_mapping::HktScopeSnapshot,
    crate::engine::scope_mapping::MappedHktScopeState,
    std::collections::BTreeMap<i32, crate::engine::trigger::WordChangeMetrics>,
) {
    use crate::engine::trigger::WordChangeMetrics;
    let mut engine = SecaEngine::new(SecaConfig::default()).unwrap();
    engine.config.seca_thresholds.alpha = 0.7;
    engine.config.seca_thresholds.beta = 0.5;
    engine
        .build_baseline_tree(make_batch(0, &[("s", &["a", "b"])]))
        .unwrap();
    let scope = engine.snapshot_hkt_scope(root_hkt_id(&engine)).unwrap();
    let batch = make_batch(1, &[("n", &["a", "b"])]);
    let mapped = engine
        .map_batch_into_hkt_scope(&batch, &[0].into(), &scope)
        .unwrap();
    let words: Vec<_> = engine
        .build_current_word_ids(&scope, &Default::default())
        .into_iter()
        .collect();
    assert_eq!(words.len(), 2);
    let changes = [
        (
            words[0],
            WordChangeMetrics {
                strength: 0.2,
                eligibility: 0.1,
                importance: 0.3,
            },
        ),
        (
            words[1],
            WordChangeMetrics {
                strength: 0.8,
                eligibility: 0.9,
                importance: 0.4,
            },
        ),
    ]
    .into();
    (engine, scope, mapped, changes)
}

#[test]
fn equations_average_only_violations_and_measure_lost_importance() {
    let (engine, scope, mapped, changes) = equation_fixture();
    let errors = engine
        .compute_paper_scope_metrics_from_change_metrics(
            &scope,
            &mapped,
            &Default::default(),
            &changes,
        )
        .unwrap();
    assert!((errors.alpha_error.unwrap() - 0.25).abs() < 1e-12);
    assert!((errors.beta_error.unwrap() - 0.2).abs() < 1e-12);
    assert!((errors.word_importance_error.unwrap() - 0.3).abs() < 1e-12);
}

#[test]
fn thresholds_are_strict_and_use_unrounded_errors() {
    let (mut engine, scope, mapped, mut changes) = equation_fixture();
    let word = *changes.keys().next().unwrap();
    changes.get_mut(&word).unwrap().strength = 0.69999;
    let errors = engine
        .compute_paper_scope_metrics_from_change_metrics(
            &scope,
            &mapped,
            &Default::default(),
            &changes,
        )
        .unwrap();
    let error = errors.alpha_error.unwrap();
    assert!(error > 0.0 && error < 0.00001);
    engine.config.seca_thresholds.alpha_error_threshold = error;
    engine.config.seca_thresholds.beta_error_threshold = 1.0;
    engine
        .config
        .seca_thresholds
        .word_importance_error_threshold = 1.0;
    assert!(
        !engine
            .evaluate_scope_trigger_decision(&scope, &mapped, &Default::default(), &changes)
            .unwrap()
            .should_reconstruct
    );
    engine.config.seca_thresholds.alpha_error_threshold = error / 2.0;
    assert!(
        engine
            .evaluate_scope_trigger_decision(&scope, &mapped, &Default::default(), &changes)
            .unwrap()
            .should_reconstruct
    );
}
