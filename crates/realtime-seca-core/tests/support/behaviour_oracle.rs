//! Public-API observer shared by read-only tests and a stdout-only capture example.
use realtime_seca_core::{BatchProcessingResult, SecaConfig, SecaEngine, SourceBatch};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub const INPUT: &str = include_str!("../fixtures/behaviour_oracle/stream.json");

#[derive(Deserialize)]
pub struct Stream {
    pub configs: BTreeMap<String, SecaConfig>,
    pub batches: Vec<SourceBatch>,
}

pub fn snapshot(engine: &SecaEngine) -> Value {
    serde_json::to_value(engine.snapshot().unwrap()).unwrap()
}

pub fn restore(saved: &Value) -> SecaEngine {
    let bytes = serde_json::to_vec(saved).unwrap();
    SecaEngine::load_snapshot(serde_json::from_slice(&bytes).unwrap()).unwrap()
}

// Preserve node order and every attachment; replace only allocation IDs with
// rooted paths, word IDs with tokens, and hashed source IDs with external IDs.
fn hierarchy(engine: &SecaEngine) -> (Value, BTreeMap<i32, String>) {
    let tree = serde_json::to_value(engine.export_baseline_tree_verbose().unwrap()).unwrap();
    fn scope(tree: &Value, id: i32, path: String, paths: &mut BTreeMap<i32, String>) -> Value {
        assert!(paths.insert(id, path.clone()).is_none());
        let h = tree["hkts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|h| h["hkt_id"] == id)
            .unwrap();
        let words = |refs: &Value| -> Vec<Value> {
            refs.as_array()
                .unwrap()
                .iter()
                .map(|w| {
                    if w["word_id"] == -1 {
                        json!({"refuge": true})
                    } else {
                        assert!(w["token"].is_string());
                        w["token"].clone()
                    }
                })
                .collect()
        };
        let nodes: Vec<_> = h["node_ids"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, nid)| {
                let n = tree["nodes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|n| n["node_id"] == *nid)
                    .unwrap();
                let mut sources: Vec<_> = n["sources"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| s["external_source_id"].as_str().unwrap())
                    .collect();
                sources.sort();
                let children: Vec<_> = tree["hkts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|h| h["parent_node_id"] == *nid)
                    .collect();
                assert!(children.len() <= 1);
                let child = children.first().map(|c| {
                    scope(
                        tree,
                        c["hkt_id"].as_i64().unwrap() as i32,
                        format!("{path}/{i}"),
                        paths,
                    )
                });
                json!({"words": words(&n["words"]), "top_words": words(&n["top_words"]),
                "sources": sources, "refuge": n["is_refuge_node"], "child": child})
            })
            .collect();
        json!({"path": path, "expected": words(&h["expected_words"]), "nodes": nodes})
    }
    let roots: Vec<_> = tree["hkts"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|h| h["parent_node_id"] == 0)
        .collect();
    assert_eq!(roots.len(), 1);
    let mut paths = BTreeMap::new();
    let result = scope(
        &tree,
        roots[0]["hkt_id"].as_i64().unwrap() as i32,
        "root".into(),
        &mut paths,
    );
    assert_eq!(paths.len(), tree["hkts"].as_array().unwrap().len());
    (result, paths)
}

// Contextual memberships include candidates that are absent from descriptors.
fn word_memberships(saved: &Value, paths: &BTreeMap<i32, String>) -> Value {
    let state = &saved["state"];
    let mut scopes = BTreeMap::new();
    for hkt in state["hkt_build_output"]["hkts_by_id"]
        .as_object()
        .unwrap()
        .values()
    {
        let nodes: Vec<_> = hkt["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|node| {
                let mut words = BTreeMap::new();
                for (word, ids) in node["word_source_ids"].as_object().unwrap() {
                    let token = state["baseline_word_legend"][word].as_str().unwrap();
                    let mut sources: Vec<_> = ids
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|id| {
                            state["baseline_source_legend"][id.as_i64().unwrap().to_string()]
                                .as_str()
                                .unwrap()
                        })
                        .collect();
                    sources.sort();
                    words.insert(token, sources);
                }
                words
            })
            .collect();
        scopes.insert(
            paths[&(hkt["hkt_id"].as_i64().unwrap() as i32)].clone(),
            nodes,
        );
    }
    json!(scopes)
}

pub fn observe(
    engine: &SecaEngine,
    report: &BatchProcessingResult,
    previous_paths: &BTreeMap<i32, String>,
) -> (Value, Value, BTreeMap<i32, String>) {
    let saved = snapshot(engine);
    let (topology, paths) = hierarchy(engine);
    let decisions: Vec<_> = report
        .decision_diagnostics
        .iter()
        .map(|d| {
            json!({
                "scope": previous_paths[&d.hkt_id],
                "output_scope": d.output_hkt_id.map(|id| paths[&id].clone()),
                "scoped_sources": d.scoped_source_count, "mapped_sources": d.mapped_source_count,
                "reconstruct": d.should_reconstruct, "reasons": d.trigger_reasons,
                "alpha": d.paper_alpha_error, "beta": d.paper_beta_error,
                "wi": d.paper_word_importance_error
            })
        })
        .collect();
    // Strict output deliberately retains IDs, report text and all public fields.
    // Snapshot internals are checked for failure isolation/round trips, not frozen
    // as a required in-memory representation for future implementations.
    let strict = json!({
        "report": report,
        "tree": engine.export_baseline_tree().unwrap(),
        "verbose_tree": engine.export_baseline_tree_verbose().unwrap(),
        "explanation": engine.explain_last_update(),
        "word_stats": engine.last_batch_word_stats_summary(),
        "stored_batches": engine.stored_batch_count(),
        "schema": saved["schema_version"], "batch_position": saved["last_processed_batch_index"],
        "vocabulary": saved["state"]["baseline_word_legend"],
        "identities": saved["state"]["source_id_by_url"],
        "provenance": saved["state"]["source_ids_by_batch_index"],
        "retained_batches": saved["state"]["processed_batches"]
    });
    let structural = json!({"topology": topology, "decisions": decisions,
        "word_memberships": word_memberships(&saved, &paths),
        "reconstructed_scopes": report.reconstructed_hkt_ids.iter().map(|id| &previous_paths[id]).collect::<Vec<_>>(),
        "sources_processed": report.sources_processed, "sources_forgotten": report.sources_forgotten,
        "active_sources": report.active_source_count, "reconstruction_triggered": report.reconstruction_triggered});
    (strict, structural, paths)
}

// Reject a valid prefix followed by malformed input, then continue normally.
// Compare the entire snapshot, including allocators and transient indexes.
fn reject_without_mutation(engine: &mut SecaEngine, incoming: &SourceBatch) -> Value {
    let before = snapshot(engine);
    let mut invalid = incoming.clone();
    invalid.sources.push(realtime_seca_core::SourceRecord {
        source_id: String::new(),
        batch_index: incoming.batch_index,
        tokens: vec!["must-not-leak".into()],
        text: None,
        timestamp_unix_ms: None,
        metadata: None,
    });
    let error = engine.process_batch(invalid).unwrap_err();
    assert_eq!(snapshot(engine), before, "rejected update mutated state");
    json!({"error": error.to_string(), "state_unchanged": true})
}

pub fn capture() -> Value {
    let stream: Stream = serde_json::from_str(INPUT).unwrap();
    let mut runs = serde_json::Map::new();
    for (name, config) in stream.configs {
        let mut engine = SecaEngine::new(config).unwrap();
        let mut steps = Vec::new();
        let mut previous_paths = BTreeMap::new();
        let mut checkpoints = serde_json::Map::new();
        for (i, batch) in stream.batches.iter().enumerate() {
            let rejection = if i == 2 {
                reject_without_mutation(&mut engine, batch)
            } else {
                Value::Null
            };
            let report = if i == 0 {
                engine.build_baseline_tree(batch.clone()).unwrap()
            } else {
                engine.process_batch(batch.clone()).unwrap()
            };
            let (strict, structural, paths) = observe(&engine, &report, &previous_paths);
            steps.push(json!({"strict": strict, "structural": structural, "rejection": rejection}));
            previous_paths = paths;
            let saved = snapshot(&engine);
            assert_eq!(snapshot(&restore(&saved)), saved);
            if i == 1 || i == 5 {
                checkpoints.insert(batch.batch_index.to_string(), saved);
            }
        }
        runs.insert(
            name,
            json!({"steps": steps, "schema4_checkpoints": checkpoints}),
        );
    }
    Value::Object(runs)
}

pub fn paths(engine: &SecaEngine) -> BTreeMap<i32, String> {
    hierarchy(engine).1
}
