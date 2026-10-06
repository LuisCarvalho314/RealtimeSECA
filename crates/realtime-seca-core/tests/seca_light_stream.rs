use realtime_seca_core::engine::rebuild::RebuildMode;
use realtime_seca_core::{MemoryMode, SecaConfig, SecaEngine, SourceBatch, SourceRecord};
use serde_json::{json, Value};
fn engine(gamma: u32) -> SecaEngine {
    let mut c = SecaConfig::default();
    c.memory_mode = MemoryMode::SlidingWindow;
    c.max_batches_in_memory = Some(gamma);
    c.hkt_builder.minimum_threshold_against_max_word_count = 0.7;
    c.seca_thresholds.alpha = 0.7;
    c.seca_thresholds.alpha_option1_threshold = 0.11;
    c.seca_thresholds.beta_option1_threshold = 0.2;
    c.seca_thresholds.word_importance_option1_threshold = 0.2;
    let mut e = SecaEngine::new(c).unwrap();
    e.set_rebuild_mode(RebuildMode::SubtreeTargeted);
    e
}
fn batch(index: u32, rows: Vec<(String, Vec<&str>)>) -> SourceBatch {
    SourceBatch {
        batch_index: index,
        sources: rows
            .into_iter()
            .map(|(id, words)| SourceRecord {
                source_id: id,
                batch_index: index,
                tokens: words.into_iter().map(String::from).collect(),
                text: Some("article text".into()),
                timestamp_unix_ms: Some(0),
                metadata: Some(json!({"article": true})),
            })
            .collect(),
    }
}
fn one(index: u32, word: &str) -> SourceBatch {
    batch(index, vec![(format!("s{index}"), vec![word])])
}
fn tree(e: &SecaEngine) -> Value {
    serde_json::to_value(e.export_baseline_tree_verbose().unwrap()).unwrap()
}
fn state(e: &SecaEngine) -> Value {
    serde_json::to_value(e.snapshot().unwrap()).unwrap()
}
fn topology(e: &SecaEngine) -> Value {
    let mut t = tree(e);
    t.as_object_mut().unwrap().remove("source_legend");
    for node in t["nodes"].as_array_mut().unwrap() {
        node.as_object_mut().unwrap().remove("sources");
    }
    t
}
fn restored(e: &SecaEngine) -> SecaEngine {
    SecaEngine::load_snapshot(serde_json::from_value(state(e)).unwrap()).unwrap()
}
#[test]
fn persisted_second_batch_retains_structure_and_source_union() {
    let mut e = engine(3);
    e.build_baseline_tree(one(1, "a")).unwrap();
    let first = topology(&e);
    e = restored(&e);
    let result = e.process_batch(one(2, "a")).unwrap();
    assert!(!result.reconstruction_triggered);
    assert_eq!(topology(&e), first);
    assert_eq!(tree(&e)["source_legend"].as_array().unwrap().len(), 2);
    assert!(result
        .notes
        .iter()
        .any(|n| n.contains("alpha_error_eq6_scaffold=0.0000")));
    assert!(result
        .notes
        .iter()
        .any(|n| n.contains("beta_error_eq9_scaffold=0.0000")));
}
#[test]
fn gamma_three_bounds_all_source_memberships_as_batches_grow() {
    let mut e = engine(3);
    e.build_baseline_tree(one(1, "a")).unwrap();
    let first = topology(&e);
    for n in 2..=30 {
        assert!(
            !e.process_batch(one(n, "a"))
                .unwrap()
                .reconstruction_triggered
        );
        assert_eq!(topology(&e), first);
        let s = state(&e);
        let t = tree(&e);
        assert!(s["state"]["processed_batches"].as_array().unwrap().len() <= 3);
        assert!(t["source_legend"].as_array().unwrap().len() <= 3);
        for node in t["nodes"].as_array().unwrap() {
            assert!(node["sources"].as_array().unwrap().len() <= 3);
        }
        if n == 4 {
            assert_eq!(s["state"]["processed_batches"][0]["batch_index"], 2);
            assert!(!s["state"]["baseline_source_legend"]
                .as_object()
                .unwrap()
                .values()
                .any(|id| id == "s1"));
            let forgotten = SecaEngine::load_snapshot(serde_json::from_value(s).unwrap()).unwrap();
            assert_eq!(tree(&forgotten), t);
        }
    }
}
#[test]
fn restart_matches_uninterrupted_updates_and_reconstruction() {
    let mut e = engine(3);
    e.build_baseline_tree(one(1, "a")).unwrap();
    for n in 2..=4 {
        e.process_batch(one(n, "a")).unwrap();
    }
    let mut r = restored(&e);
    let next = batch(
        5,
        vec![
            ("new1".into(), vec!["x"]),
            ("new2".into(), vec!["x"]),
            ("new3".into(), vec!["x"]),
        ],
    );
    assert_eq!(
        serde_json::to_value(e.process_batch(next.clone()).unwrap()).unwrap(),
        serde_json::to_value(r.process_batch(next).unwrap()).unwrap()
    );
    assert_eq!(state(&e), state(&r));
}
#[test]
fn empty_batches_forget_sources_but_preserve_structure() {
    let mut e = engine(1);
    e.build_baseline_tree(one(1, "a")).unwrap();
    let first = topology(&e);
    for n in 2..=5 {
        assert!(
            !e.process_batch(batch(n, vec![]))
                .unwrap()
                .reconstruction_triggered
        );
        assert_eq!(topology(&e), first);
        assert_eq!(tree(&e)["source_legend"], json!([]));
        e = restored(&e);
    }
}
#[test]
fn failed_update_leaves_valid_state_identical() {
    let mut e = engine(3);
    e.build_baseline_tree(one(1, "a")).unwrap();
    let before = state(&e);
    assert!(e.process_batch(one(3, "x")).is_err());
    assert_eq!(state(&e), before);
}
#[test]
fn duplicate_sources_are_counted_once() {
    let mut e = engine(3);
    e.build_baseline_tree(one(1, "a")).unwrap();
    let b = batch(
        2,
        vec![
            ("s1".into(), vec!["a"]),
            ("s2".into(), vec!["a"]),
            ("s2".into(), vec!["a"]),
        ],
    );
    assert_eq!(e.process_batch(b).unwrap().sources_processed, 1);
    assert_eq!(tree(&e)["nodes"][0]["sources"].as_array().unwrap().len(), 2);
}
#[test]
fn change_reconstructs_only_target_child_and_preserves_sibling() {
    let mut e = engine(3);
    let rows = (0..20)
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
    e.build_baseline_tree(batch(1, rows)).unwrap();
    let before = tree(&e);
    let root = before["hkts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|h| h["parent_node_id"] == 0)
        .unwrap();
    let d_node = before["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| {
            n["words"]
                .as_array()
                .unwrap()
                .iter()
                .any(|w| w["token"] == "d")
        })
        .unwrap();
    let sibling = before["hkts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|h| h["parent_node_id"] == d_node["node_id"])
        .unwrap();
    let incoming = (0..10)
        .map(|i| {
            (
                format!("n{i}"),
                if i < 3 { vec!["a", "x"] } else { vec!["a"] },
            )
        })
        .collect();
    let result = e.process_batch(batch(2, incoming)).unwrap();
    assert!(result.reconstruction_triggered, "{:?}", result.notes);
    let after = tree(&e);
    assert!(after["hkts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|h| h["hkt_id"] == root["hkt_id"]));
    assert!(after["hkts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|h| h == sibling));
    assert!(after["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|n| n == d_node));
    assert!(result
        .notes
        .iter()
        .any(|n| n.contains("Selected-HKT rebuild plan") && n.contains("1 target(s)")));
}
#[test]
fn reconstruction_shrinks_tree_after_forgetting_and_history_remains_readable() {
    let mut e = engine(1);
    e.build_baseline_tree(batch(
        1,
        vec![("a1".into(), vec!["a"]), ("b1".into(), vec!["b"])],
    ))
    .unwrap();
    let old = tree(&e);
    e.process_batch(one(2, "a")).unwrap();
    assert_eq!(
        tree(&e)["nodes"].as_array().unwrap().len(),
        old["nodes"].as_array().unwrap().len()
    );
    assert!(
        e.process_batch(one(3, "a"))
            .unwrap()
            .reconstruction_triggered
    );
    assert!(tree(&e)["nodes"].as_array().unwrap().len() < old["nodes"].as_array().unwrap().len());
    assert_eq!(old["nodes"].as_array().unwrap().len(), 2);
}
#[test]
fn old_metadata_only_schema_is_rejected() {
    let mut e = engine(3);
    e.build_baseline_tree(one(1, "a")).unwrap();
    let mut s = e.snapshot().unwrap();
    s.schema_version = 2;
    assert!(SecaEngine::load_snapshot(s).is_err());
}
