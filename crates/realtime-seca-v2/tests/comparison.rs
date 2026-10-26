//! V1 is comparison evidence; mathematical expectations live in mathematics.rs.
use realtime_seca_core::{
    HktBuilderConfig, MemoryMode, SecaConfig, SecaEngine, SecaThresholdConfig, SourceBatch,
    SourceRecord,
};
use realtime_seca_v2::{metrics::Errors, Config, Model, Source};
use std::collections::BTreeSet;

fn pair(alpha: f64, beta: f64, window: Option<u32>) -> (SecaEngine, Model) {
    let v1 = SecaEngine::new(SecaConfig {
        hkt_builder: HktBuilderConfig {
            minimum_threshold_against_max_word_count: alpha,
            similarity_threshold: beta,
            minimum_number_of_sources_to_create_branch_for_node: 99,
        },
        seca_thresholds: SecaThresholdConfig {
            alpha,
            beta,
            alpha_error_threshold: 1.,
            beta_error_threshold: 1.,
            word_importance_error_threshold: 1.,
        },
        memory_mode: if window.is_some() {
            MemoryMode::SlidingWindow
        } else {
            MemoryMode::Full
        },
        max_batches_in_memory: window,
    })
    .unwrap();
    let v2 = Model::new(Config {
        alpha,
        beta,
        error_limits: Errors {
            alpha: 1.,
            beta: 1.,
            importance: 1.,
        },
        minimum_branch_sources: 99,
        window: window.map(u64::from),
    })
    .unwrap();
    (v1, v2)
}
fn update(v1: &mut SecaEngine, v2: &mut Model, batch: u32, rows: &[(usize, &str)]) {
    let sources: Vec<_> = rows
        .iter()
        .map(|(id, w)| SourceRecord {
            source_id: id.to_string(),
            batch_index: batch,
            tokens: w.split_whitespace().map(str::to_owned).collect(),
            text: None,
            timestamp_unix_ms: None,
            metadata: None,
        })
        .collect();
    v2.update(
        batch.into(),
        sources
            .iter()
            .map(|s| Source {
                id: s.source_id.clone(),
                words: s.tokens.iter().cloned().collect(),
            })
            .collect(),
    )
    .unwrap();
    let b = SourceBatch {
        batch_index: batch,
        sources,
    };
    if batch == 0 {
        v1.build_baseline_tree(b).unwrap();
    } else {
        v1.process_batch(b).unwrap();
    }
}
fn v1_words(v1: &SecaEngine) -> Vec<BTreeSet<String>> {
    let tree = v1.export_baseline_tree_verbose().unwrap();
    tree.nodes
        .iter()
        .filter(|n| !n.is_refuge_node)
        .map(|n| {
            n.words
                .iter()
                .map(|w| w.token.clone().expect("resolved word"))
                .collect()
        })
        .collect()
}
#[test]
fn intentional_frequency_order_correction() {
    let (mut a, mut b) = pair(0.5, 1., None);
    update(
        &mut a,
        &mut b,
        0,
        &[
            (1, "a b"),
            (2, "a b"),
            (3, "b"),
            (4, "z"),
            (5, "z"),
            (6, "z"),
            (7, "z"),
        ],
    );
    assert_eq!(
        v1_words(&a).len(),
        2,
        "V1 vocabulary-order merge characterization changed"
    );
    assert_eq!(
        b.root().nodes.len(),
        3,
        "V2 frequency-first independent expectation"
    );
}
#[test]
fn deferred_candidates_do_not_change_descriptors() {
    let (mut a, mut b) = pair(0.6, 0.5, None);
    update(
        &mut a,
        &mut b,
        0,
        &[(1, "a x"), (2, "a x"), (3, "a"), (4, "a")],
    );
    update(&mut a, &mut b, 1, &[(5, "a x"), (6, "a x")]);
    assert!(!v1_words(&a)[0].contains("x"));
    assert!(!b.root().nodes[0].words.contains("x"));
    update(&mut a, &mut b, 2, &[(7, "a x")]);
    assert!(!v1_words(&a)[0].contains("x"));
    assert!(!b.root().nodes[0].words.contains("x"));
}
#[test]
fn identical_window_and_duplicate_identity_lifecycle() {
    let (mut a, mut b) = pair(0.7, 0.5, Some(2));
    for (batch, rows) in [
        vec![(1, "a"), (1, "ignored")],
        vec![(2, "a")],
        vec![],
        vec![(1, "a")],
    ]
    .iter()
    .enumerate()
    {
        update(&mut a, &mut b, batch as u32, rows);
        let tree = a.export_baseline_tree_verbose().unwrap();
        let retained: BTreeSet<_> = tree
            .source_legend
            .iter()
            .map(|s| s.external_source_id.clone())
            .collect();
        assert_eq!(retained, b.retained_sources());
        assert_eq!(
            v1_words(&a),
            b.root()
                .nodes
                .iter()
                .filter(|n| !n.words.is_empty())
                .map(|n| n.words.clone())
                .collect::<Vec<_>>()
        );
    }
}
