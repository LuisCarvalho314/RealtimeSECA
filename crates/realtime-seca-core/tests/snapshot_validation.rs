//! Snapshot corruption and recovery at the public persistence boundary.
use realtime_seca_core::{MemoryMode, SecaConfig, SecaEngine, SourceBatch, SourceRecord};
use serde_json::{json, Value};

fn batch(index: u32, rows: &[(&str, &[&str])]) -> SourceBatch {
    SourceBatch {
        batch_index: index,
        sources: rows
            .iter()
            .map(|(id, tokens)| SourceRecord {
                source_id: (*id).into(),
                batch_index: index,
                tokens: tokens.iter().map(|s| (*s).into()).collect(),
                text: Some("discard me".into()),
                timestamp_unix_ms: Some(42),
                metadata: Some(json!({"discard": true})),
            })
            .collect(),
    }
}
fn engine(gamma: u32) -> SecaEngine {
    let mut config = SecaConfig {
        memory_mode: MemoryMode::SlidingWindow,
        max_batches_in_memory: Some(gamma),
        ..SecaConfig::default()
    };
    config.hkt_builder.minimum_threshold_against_max_word_count = 0.7;
    config.seca_thresholds.alpha = 0.7;
    config.seca_thresholds.alpha_error_threshold = 0.11;
    config.seca_thresholds.beta_error_threshold = 0.2;
    config.seca_thresholds.word_importance_error_threshold = 0.2;
    let mut e = SecaEngine::new(config).unwrap();
    e.build_baseline_tree(batch(1, &[("original", &["a", "b"])]))
        .unwrap();
    e
}
fn snapshot(e: &SecaEngine) -> Value {
    serde_json::to_value(e.snapshot().unwrap()).unwrap()
}
fn load(value: Value) -> Result<SecaEngine, realtime_seca_core::SecaError> {
    // A real JSON byte round trip, rather than cloning the engine in memory.
    let bytes = serde_json::to_vec(&value).unwrap();
    SecaEngine::load_snapshot(serde_json::from_slice(&bytes).unwrap())
}
fn corrupt_and_reject(change: impl FnOnce(&mut Value)) {
    let e = engine(3);
    let clean = snapshot(&e);
    assert!(load(clean.clone()).is_ok());
    let mut corrupted = clean.clone();
    change(&mut corrupted);
    assert!(load(corrupted).is_err(), "corrupted snapshot was accepted");
    assert_eq!(snapshot(&e), clean);
}
fn edit_node(s: &mut Value, change: impl Fn(&mut Value)) {
    let tree = &mut s["state"]["hkt_build_output"];
    let key = tree["nodes_by_id"]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap()
        .clone();
    change(&mut tree["nodes_by_id"][&key]);
    for hkt in tree["hkts_by_id"].as_object_mut().unwrap().values_mut() {
        for node in hkt["nodes"].as_array_mut().unwrap() {
            if node["node_id"].as_i64().unwrap().to_string() == key {
                change(node);
            }
        }
    }
}
macro_rules! rejection {
    ($name:ident, $change:expr) => {
        #[test]
        fn $name() {
            corrupt_and_reject($change);
        }
    };
}
rejection!(snapshot_rejects_invalid_descriptor_word, |s| {
    edit_node(s, |n| n["word_ids"] = json!([2147483647]));
});
rejection!(snapshot_rejects_invalid_expected_word, |s| {
    s["state"]["hkt_build_output"]["hkts_by_id"]
        .as_object_mut()
        .unwrap()
        .values_mut()
        .next()
        .unwrap()["expected_words"] = json!([2147483647]);
});
rejection!(snapshot_rejects_invalid_top_word, |s| {
    edit_node(s, |n| n["top_words"] = json!([2147483647]));
});
rejection!(snapshot_rejects_invalid_membership_word, |s| {
    edit_node(s, |n| {
        let sources = n["source_ids"].clone();
        n["word_source_ids"] = json!({"2147483647": sources});
    });
});
rejection!(snapshot_rejects_expired_retained_evidence, |s| {
    // All envelope and index relationships still agree, but batch 1 is outside gamma=3.
    s["last_processed_batch_index"] = json!(5);
    s["state"]["last_processed_batch_index"] = json!(5);
    s["state"]["processed_batches"]
        .as_array_mut()
        .unwrap()
        .push(json!({"batch_index": 5, "sources": []}));
});

#[test]
fn valid_sliding_window_snapshots_include_boundary_and_saturate_at_zero() {
    for gamma in [1, 3, u32::MAX] {
        let mut e = engine(gamma);
        for index in 2..=5 {
            e.process_batch(batch(index, &[("next", &["a"])])).unwrap();
            let saved = snapshot(&e);
            assert_eq!(snapshot(&load(saved.clone()).unwrap()), saved);
        }
    }
}

#[test]
fn full_memory_snapshot_retains_old_evidence_without_gamma_validation() {
    let mut config = engine(1).config().clone();
    config.memory_mode = MemoryMode::Full;
    let mut e = SecaEngine::new(config).unwrap();
    e.build_baseline_tree(batch(1, &[("original", &["a", "b"])]))
        .unwrap();
    for index in 2..=5 {
        e.process_batch(batch(index, &[])).unwrap();
    }
    let saved = snapshot(&e);
    assert_eq!(saved["state"]["processed_batches"][0]["batch_index"], 1);
    assert_eq!(
        saved["state"]["source_id_by_url"]
            .as_object()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(snapshot(&load(saved.clone()).unwrap()), saved);
}

#[test]
fn empty_updates_advance_boundary_and_clear_all_expired_indexes() {
    let mut e = engine(3);
    for index in 2..=4 {
        e.process_batch(batch(index, &[])).unwrap();
        let saved = snapshot(&e);
        let sources = saved["state"]["baseline_source_legend"]
            .as_object()
            .unwrap();
        assert_eq!(sources.len(), usize::from(index < 4));
        if index == 4 {
            assert_eq!(saved["state"]["processed_batches"][0]["batch_index"], 2);
            for field in ["source_id_by_url", "source_ids_by_batch_index"] {
                assert!(saved["state"][field].as_object().unwrap().is_empty());
            }
        }
        e = load(saved).unwrap();
    }
}

#[test]
fn round_trip_followed_by_update_matches_live_engine() {
    let mut live = engine(3);
    live.process_batch(batch(2, &[("second", &["a"])])).unwrap();
    let mut restored = load(snapshot(&live)).unwrap();
    for incoming in [
        batch(3, &[("third", &["new"])]),
        batch(4, &[]),
        batch(5, &[("original", &["a", "b"])]),
    ] {
        assert_eq!(
            serde_json::to_value(live.process_batch(incoming.clone()).unwrap()).unwrap(),
            serde_json::to_value(restored.process_batch(incoming).unwrap()).unwrap()
        );
        assert_eq!(snapshot(&live), snapshot(&restored));
        restored = load(snapshot(&restored)).unwrap();
    }
}

#[test]
fn refuge_sentinel_is_only_valid_in_node_descriptors() {
    for field in ["expected_words", "top_words", "word_source_ids"] {
        corrupt_and_reject(|s| match field {
            "expected_words" => {
                s["state"]["hkt_build_output"]["hkts_by_id"]
                    .as_object_mut()
                    .unwrap()
                    .values_mut()
                    .next()
                    .unwrap()[field] = json!([-1]);
            }
            "top_words" => edit_node(s, |n| n[field] = json!([-1])),
            _ => edit_node(s, |n| n[field] = json!({"-1": n["source_ids"].clone()})),
        });
    }
    let mut saved = snapshot(&engine(3));
    edit_node(&mut saved, |n| {
        n["word_ids"] = json!([-1]);
        n["top_words"] = json!([]);
        n["word_source_ids"] = json!({});
    });
    assert!(load(saved).is_ok());
}

#[test]
fn expired_empty_batch_and_detached_provenance_are_rejected() {
    let mut e = engine(1);
    e.process_batch(batch(2, &[])).unwrap();
    let saved = snapshot(&e);
    let mut expired = saved.clone();
    expired["state"]["processed_batches"]
        .as_array_mut()
        .unwrap()
        .insert(0, json!({"batch_index": 1, "sources": []}));
    assert!(load(expired).is_err());
    let mut provenance = saved.clone();
    provenance["state"]["source_ids_by_batch_index"]["1"] = json!([123]);
    assert!(load(provenance).is_err());
    let mut identity = saved.clone();
    identity["state"]["source_id_by_url"]["expired"] = json!(123);
    identity["state"]["baseline_source_legend"]["123"] = json!("expired");
    assert!(load(identity).is_err());
    assert_eq!(snapshot(&e), saved);
}

#[test]
fn retained_batches_must_be_ordered_and_not_in_the_future() {
    let mut e = engine(3);
    e.process_batch(batch(2, &[])).unwrap();
    let saved = snapshot(&e);
    let mut unordered = saved.clone();
    unordered["state"]["processed_batches"]
        .as_array_mut()
        .unwrap()
        .insert(0, json!({"batch_index": 2, "sources": []}));
    assert!(load(unordered).is_err());
    let mut future = saved.clone();
    future["state"]["processed_batches"]
        .as_array_mut()
        .unwrap()
        .insert(0, json!({"batch_index": 3, "sources": []}));
    assert!(load(future).is_err());
    assert_eq!(snapshot(&e), saved);
}
