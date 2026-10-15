//! Persistence-boundary regressions recovered from the engineering audit.
use realtime_seca_core::{MemoryMode, SecaConfig, SecaEngine, SourceBatch, SourceRecord};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

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
fn ids(value: &Value) -> BTreeSet<i64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}

// Independent checks over serialized public state, including references omitted
// by restore validation. Candidate word memberships need not be descriptors.
fn assert_source_indexes(s: &Value, gamma: u32) -> BTreeSet<i64> {
    let state = &s["state"];
    let legend = state["baseline_source_legend"].as_object().unwrap();
    let inverse = state["source_id_by_url"].as_object().unwrap();
    assert_eq!(legend.len(), inverse.len());
    let mut provenance = BTreeMap::<String, BTreeSet<i64>>::new();
    let mut seen = BTreeSet::new();
    let current = s["last_processed_batch_index"].as_u64().unwrap();
    for b in state["processed_batches"].as_array().unwrap() {
        let index = b["batch_index"].as_u64().unwrap();
        assert!(index >= current.saturating_sub(u64::from(gamma - 1)));
        for source in b["sources"].as_array().unwrap() {
            assert_eq!(source["batch_index"], b["batch_index"]);
            let external = source["source_id"].as_str().unwrap();
            let internal = inverse[external].as_i64().unwrap();
            assert!(seen.insert(internal));
            assert_eq!(legend[&internal.to_string()], external);
            provenance
                .entry(index.to_string())
                .or_default()
                .insert(internal);
            for field in ["text", "metadata", "timestamp_unix_ms"] {
                assert!(source[field].is_null());
            }
        }
    }
    assert_eq!(seen.len(), legend.len());
    let actual: BTreeMap<_, _> = state["source_ids_by_batch_index"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, value)| (key.clone(), ids(value)))
        .collect();
    assert_eq!(actual, provenance);
    seen
}

fn assert_topology(state: &Value, seen: &BTreeSet<i64>) {
    let words = state["baseline_word_legend"].as_object().unwrap();
    let check_word = |v: &Value| {
        let id = v.as_i64().unwrap();
        assert!(
            id == -1 || (id > 0 && words.contains_key(&id.to_string())),
            "unknown word {id}"
        );
    };
    let tree = &state["hkt_build_output"];
    let hkts = tree["hkts_by_id"].as_object().unwrap();
    let cache = tree["nodes_by_id"].as_object().unwrap();
    let roots: Vec<_> = hkts.values().filter(|h| h["parent_node_id"] == 0).collect();
    assert_eq!(roots.len(), 1);
    let mut queue = vec![roots[0]];
    let mut reached = BTreeSet::new();
    let mut nodes = BTreeSet::new();
    while let Some(hkt) = queue.pop() {
        let hkt_id = hkt["hkt_id"].as_i64().unwrap();
        assert!(reached.insert(hkt_id));
        assert_eq!(hkts[&hkt_id.to_string()], *hkt);
        assert!(hkt_id < state["next_hkt_id"].as_i64().unwrap());
        for word in hkt["expected_words"].as_array().unwrap() {
            check_word(word);
        }
        for node in hkt["nodes"].as_array().unwrap() {
            let node_id = node["node_id"].as_i64().unwrap();
            assert!(nodes.insert(node_id));
            assert_eq!(node["hkt_id"], hkt["hkt_id"]);
            assert_eq!(cache[&node_id.to_string()], *node);
            assert!(node_id < state["next_node_id"].as_i64().unwrap());
            let members = ids(&node["source_ids"]);
            assert!(members.is_subset(seen));
            for field in ["word_ids", "top_words"] {
                for word in node[field].as_array().unwrap() {
                    check_word(word);
                }
            }
            for (word, sources) in node["word_source_ids"].as_object().unwrap() {
                check_word(&json!(word.parse::<i64>().unwrap()));
                assert!(ids(sources).is_subset(&members));
                assert!(!ids(sources).is_empty());
            }
            assert!(node["source_ids_new_from_batches"]
                .as_array()
                .unwrap()
                .is_empty());
            assert!(node["word_source_ids_new_from_batches"]
                .as_object()
                .unwrap()
                .is_empty());
            queue.extend(
                hkts.values()
                    .filter(|h| h["parent_node_id"] == node["node_id"]),
            );
        }
    }
    assert_eq!(reached.len(), hkts.len());
    assert_eq!(nodes.len(), cache.len());
}

fn assert_invariants(e: &SecaEngine, gamma: u32) {
    let s = snapshot(e);
    let seen = assert_source_indexes(&s, gamma);
    assert_topology(&s["state"], &seen);
    let export = e.export_baseline_tree_verbose().unwrap();
    for word in export.hkts.iter().flat_map(|h| &h.expected_words).chain(
        export
            .nodes
            .iter()
            .flat_map(|n| n.words.iter().chain(&n.top_words)),
    ) {
        assert!(word.token.is_some());
    }
    assert_eq!(snapshot(&load(s.clone()).unwrap()), s);
}

#[test]
fn expired_identity_readmission_and_first_duplicate_wins_across_restart() {
    for gamma in [1, 3] {
        let mut e = engine(gamma);
        assert_eq!(
            e.process_batch(batch(
                2,
                &[
                    ("original", &["ignored"]),
                    ("second", &["a"]),
                    ("second", &["discarded"])
                ]
            ))
            .unwrap()
            .sources_processed,
            1
        );
        assert_invariants(&e, gamma);
        assert!(!snapshot(&e)["state"]["baseline_word_legend"]
            .as_object()
            .unwrap()
            .values()
            .any(|v| v == "ignored" || v == "discarded"));
        for index in 3..=gamma + 2 {
            e.process_batch(batch(index, &[])).unwrap();
            assert_invariants(&e, gamma);
        }
        assert!(snapshot(&e)["state"]["source_id_by_url"]
            .get("original")
            .is_none());
        e = load(snapshot(&e)).unwrap();
        assert_eq!(
            e.process_batch(batch(gamma + 3, &[("original", &["readmitted"])]))
                .unwrap()
                .sources_processed,
            1
        );
        assert_invariants(&e, gamma);
        assert!(snapshot(&e)["state"]["baseline_word_legend"]
            .as_object()
            .unwrap()
            .values()
            .any(|v| v == "readmitted"));
    }
}

#[test]
fn repeated_reconstruction_preserves_vocabulary_and_never_recycles_ids() {
    for gamma in [1, 3] {
        let mut e = engine(gamma);
        let mut restarted = load(snapshot(&e)).unwrap();
        let mut old_words = snapshot(&e)["state"]["baseline_word_legend"].clone();
        let mut allocated_hkts = BTreeSet::new();
        let mut allocated_nodes = BTreeSet::new();
        let mut reconstructions = 0;
        for index in 2..=10 {
            let before = snapshot(&e);
            let word = format!("novel{index}");
            let names: Vec<_> = (0..8).map(|i| format!("{index}:{i}")).collect();
            let tokens = [word.as_str()];
            let rows: Vec<_> = names
                .iter()
                .map(|id| (id.as_str(), tokens.as_slice()))
                .collect();
            let incoming = if index % 3 == 0 {
                batch(index, &[])
            } else {
                batch(index, &rows)
            };
            let result = e.process_batch(incoming.clone()).unwrap();
            assert_eq!(
                serde_json::to_value(&result).unwrap(),
                serde_json::to_value(restarted.process_batch(incoming).unwrap()).unwrap()
            );
            assert_eq!(snapshot(&e), snapshot(&restarted));
            reconstructions += usize::from(result.reconstruction_triggered);
            let after = snapshot(&e);
            for (map, counter, allocated) in [
                ("hkts_by_id", "next_hkt_id", &mut allocated_hkts),
                ("nodes_by_id", "next_node_id", &mut allocated_nodes),
            ] {
                let previous = before["state"]["hkt_build_output"][map]
                    .as_object()
                    .unwrap();
                allocated.extend(previous.keys().map(|s| s.parse::<i64>().unwrap()));
                for key in after["state"]["hkt_build_output"][map]
                    .as_object()
                    .unwrap()
                    .keys()
                {
                    if !previous.contains_key(key) {
                        let id = key.parse::<i64>().unwrap();
                        assert!(!allocated.contains(&id));
                        assert!(id >= before["state"][counter].as_i64().unwrap());
                    }
                }
                assert!(after["state"][counter].as_i64() >= before["state"][counter].as_i64());
            }
            for (id, token) in old_words.as_object().unwrap() {
                assert_eq!(after["state"]["baseline_word_legend"][id], *token);
            }
            old_words = after["state"]["baseline_word_legend"].clone();
            assert_invariants(&e, gamma);
            restarted = load(after).unwrap();
        }
        assert!(reconstructions >= 2);
        assert!(old_words.as_object().unwrap().len() > gamma as usize);
    }
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
rejection!(snapshot_rejects_missing_cache_entry, |s| {
    s["state"]["hkt_build_output"]["nodes_by_id"]
        .as_object_mut()
        .unwrap()
        .clear();
});
rejection!(snapshot_rejects_orphan_hkt, |s| {
    s["state"]["hkt_build_output"]["hkts_by_id"]
        .as_object_mut()
        .unwrap()
        .values_mut()
        .next()
        .unwrap()["parent_node_id"] = json!(999);
});
#[test]
fn snapshot_rejects_broken_child_attachment_with_root_preserved() {
    let mut s = snapshot(&engine(3));
    let tree = &mut s["state"]["hkt_build_output"];
    let parent = tree["nodes_by_id"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()["node_id"]
        .clone();
    // A reachable empty child isolates attachment validation from node/cache validation.
    tree["hkts_by_id"]["2"] = json!({
        "hkt_id": 2, "parent_node_id": parent, "expected_words": [],
        "is_state1": false, "nodes": []
    });
    s["state"]["next_hkt_id"] = json!(3);
    assert!(load(s.clone()).is_ok());
    s["state"]["hkt_build_output"]["hkts_by_id"]["2"]["parent_node_id"] = json!(999);
    assert!(load(s).is_err());
}
rejection!(snapshot_rejects_membership_outside_node_sources, |s| {
    edit_node(s, |n| n["source_ids"] = json!([]));
});
rejection!(snapshot_rejects_unretained_node_source, |s| {
    edit_node(s, |n| {
        n["source_ids"].as_array_mut().unwrap().push(json!(123))
    });
});
rejection!(snapshot_rejects_second_root, |s| {
    let hkts = s["state"]["hkt_build_output"]["hkts_by_id"]
        .as_object_mut()
        .unwrap();
    let mut extra = hkts.values().next().unwrap().clone();
    extra["hkt_id"] = json!(2);
    extra["nodes"] = json!([]);
    hkts.insert("2".into(), extra);
    s["state"]["next_hkt_id"] = json!(3);
});
rejection!(snapshot_rejects_inconsistent_inverse_identity, |s| {
    s["state"]["source_id_by_url"]["original"] = json!(123);
});
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
rejection!(snapshot_rejects_hkt_allocator_at_live_id, |s| {
    s["state"]["next_hkt_id"] = json!(1);
});
rejection!(snapshot_rejects_node_allocator_at_live_id, |s| {
    s["state"]["next_node_id"] = json!(1);
});
rejection!(snapshot_rejects_future_schema, |s| {
    s["schema_version"] = json!(5);
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
fn failed_initialization_overwrite_and_invalid_identity_leave_complete_state_unchanged() {
    let mut e = engine(3);
    let before = snapshot(&e);
    assert!(e
        .build_baseline_tree(batch(2, &[("replacement", &["new"])]))
        .is_err());
    assert_eq!(snapshot(&e), before);
    assert!(e
        .process_batch(batch(2, &[("valid", &["new"]), ("", &["bad"])]))
        .is_err());
    assert_eq!(snapshot(&e), before);
    assert_eq!(
        e.process_batch(batch(2, &[("valid", &["new"])]))
            .unwrap()
            .sources_processed,
        1
    );
    assert_invariants(&e, 3);
}

rejection!(snapshot_rejects_future_batch, |s| {
    s["state"]["processed_batches"][0]["batch_index"] = json!(2);
    s["state"]["processed_batches"][0]["sources"][0]["batch_index"] = json!(2);
});
rejection!(snapshot_rejects_duplicate_empty_batch, |s| {
    let batches = s["state"]["processed_batches"].as_array_mut().unwrap();
    batches.push(json!({"batch_index": 1, "sources": []}));
});
rejection!(snapshot_rejects_extra_provenance, |s| {
    s["state"]["source_ids_by_batch_index"]["0"] = json!([123]);
});
rejection!(snapshot_rejects_empty_identity, |s| {
    let state = &mut s["state"];
    let id = state["source_id_by_url"]["original"].clone();
    state["source_id_by_url"]
        .as_object_mut()
        .unwrap()
        .remove("original");
    state["source_id_by_url"][""] = id.clone();
    state["baseline_source_legend"][id.as_i64().unwrap().to_string()] = json!("");
    state["processed_batches"][0]["sources"][0]["source_id"] = json!("");
});

#[test]
fn restoration_retains_inclusive_gamma_boundary_and_full_history() {
    let mut e = engine(3);
    e.process_batch(batch(2, &[])).unwrap();
    e.process_batch(batch(3, &[])).unwrap();
    assert!(load(snapshot(&e)).is_ok());
    assert!(snapshot(&e)["state"]["source_id_by_url"]
        .get("original")
        .is_some());
    e.process_batch(batch(4, &[])).unwrap();
    assert!(load(snapshot(&e)).is_ok());
    assert!(snapshot(&e)["state"]["source_id_by_url"]
        .get("original")
        .is_none());

    let mut full = snapshot(&engine(3));
    full["config"]["memory_mode"] = json!("Full");
    full["state"]["config"]["memory_mode"] = json!("Full");
    full["last_processed_batch_index"] = json!(5);
    full["state"]["last_processed_batch_index"] = json!(5);
    for index in 2..=5 {
        full["state"]["processed_batches"]
            .as_array_mut()
            .unwrap()
            .push(json!({"batch_index": index, "sources": []}));
    }
    assert!(load(full).is_ok());
}

rejection!(snapshot_rejects_missing_empty_batch, |s| {
    s["last_processed_batch_index"] = json!(3);
    s["state"]["last_processed_batch_index"] = json!(3);
    s["state"]["processed_batches"]
        .as_array_mut()
        .unwrap()
        .push(json!({"batch_index": 3, "sources": []}));
});
