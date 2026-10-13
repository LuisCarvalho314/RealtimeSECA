//! Scientific observations of the unchanged production engine; no alternate builder.
#![allow(clippy::field_reassign_with_default)]

use realtime_seca_core::{MemoryMode, SecaConfig, SecaEngine, SourceBatch, SourceRecord};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::PathBuf};

fn directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scientific_hkt")
}

fn cases() -> Vec<PathBuf> {
    let mut paths: Vec<_> = fs::read_dir(directory())
        .unwrap()
        .map(|p| p.unwrap().path())
        .filter(|p| p.is_dir())
        .collect();
    paths.sort();
    paths
}

fn topology(engine: &SecaEngine) -> (Value, BTreeMap<i32, String>) {
    let tree = engine.export_baseline_tree_verbose().unwrap();
    fn scope(
        tree: &realtime_seca_core::BaselineTreeVerboseExport,
        id: i32,
        path: String,
        paths: &mut BTreeMap<i32, String>,
    ) -> Value {
        assert!(paths.insert(id, path.clone()).is_none());
        let h = tree.hkts.iter().find(|h| h.hkt_id == id).unwrap();
        let mut expected: Vec<_> = h
            .expected_words
            .iter()
            .map(|w| w.token.clone().unwrap())
            .collect();
        expected.sort();
        let nodes: Vec<_> = h
            .node_ids
            .iter()
            .enumerate()
            .map(|(i, nid)| {
                let n = tree.nodes.iter().find(|n| n.node_id == *nid).unwrap();
                let mut words: Vec<_> = n
                    .words
                    .iter()
                    .map(|w| {
                        if w.word_id == -1 {
                            "<>".to_owned()
                        } else {
                            w.token.clone().unwrap()
                        }
                    })
                    .collect();
                words.sort();
                let mut sources: Vec<_> = n
                    .sources
                    .iter()
                    .map(|s| s.external_source_id.clone())
                    .collect();
                sources.sort();
                let children: Vec<_> = tree
                    .hkts
                    .iter()
                    .filter(|h| h.parent_node_id == *nid)
                    .collect();
                assert!(children.len() <= 1);
                let child = children
                    .first()
                    .map(|c| scope(tree, c.hkt_id, format!("{path}/{i}"), paths));
                json!({"words":words,"sources":sources,"refuge":n.is_refuge_node,"child":child})
            })
            .collect();
        json!({"path":path,"expected":expected,"nodes":nodes})
    }
    let roots: Vec<_> = tree.hkts.iter().filter(|h| h.parent_node_id == 0).collect();
    assert_eq!(roots.len(), 1);
    let mut paths = BTreeMap::new();
    let result = scope(&tree, roots[0].hkt_id, "root".into(), &mut paths);
    assert_eq!(paths.len(), tree.hkts.len());
    (result, paths)
}

fn run(input: &Value) -> Value {
    let mut config = SecaConfig::default();
    config.memory_mode = MemoryMode::Full;
    config.hkt_builder.minimum_threshold_against_max_word_count = input["alpha"].as_f64().unwrap();
    config.hkt_builder.similarity_threshold = input["beta"].as_f64().unwrap();
    config
        .hkt_builder
        .minimum_number_of_sources_to_create_branch_for_node =
        input["branchMinimum"].as_u64().unwrap() as usize;
    config.seca_thresholds.alpha = input["alpha"].as_f64().unwrap();
    config.seca_thresholds.beta = input["beta"].as_f64().unwrap();
    config.seca_thresholds.alpha_error_threshold = input["alphaError"].as_f64().unwrap();
    config.seca_thresholds.beta_error_threshold = input["betaError"].as_f64().unwrap();
    config.seca_thresholds.word_importance_error_threshold =
        input["wordImportanceError"].as_f64().unwrap();
    let mut engine = SecaEngine::new(config).unwrap();
    let mut snapshots = Vec::new();
    let mut previous_paths = BTreeMap::new();
    for (i, rows) in input["batches"].as_array().unwrap().iter().enumerate() {
        let index = i as u32 + 1;
        let batch = SourceBatch {
            batch_index: index,
            sources: rows
                .as_array()
                .unwrap()
                .iter()
                .map(|r| SourceRecord {
                    source_id: r["id"].to_string(),
                    batch_index: index,
                    tokens: r["words"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|w| w.as_str().unwrap().into())
                        .collect(),
                    text: None,
                    timestamp_unix_ms: None,
                    metadata: None,
                })
                .collect(),
        };
        let mut decisions = Vec::new();
        if i == 0 {
            engine.build_baseline_tree(batch).unwrap();
        } else {
            let result = engine.process_batch(batch).unwrap();
            for d in result.decision_diagnostics {
                decisions.push(
                    json!({"path":previous_paths[&d.hkt_id],"triggered":d.should_reconstruct}),
                );
            }
            decisions.sort_by_key(|d| d["path"].as_str().unwrap().to_owned());
        }
        let (topology, paths) = topology(&engine);
        previous_paths = paths;
        // Validate the public resumable state after each cycle as well as topology.
        SecaEngine::load_snapshot(engine.snapshot().unwrap()).unwrap();
        snapshots.push(json!({"topology":topology,"decisions":decisions}));
    }
    json!(snapshots)
}

fn without_expected(mut value: Value) -> Value {
    fn strip(v: &mut Value) {
        match v {
            Value::Object(map) => {
                map.remove("expected");
                for v in map.values_mut() {
                    strip(v);
                }
            }
            Value::Array(a) => {
                for v in a {
                    strip(v);
                }
            }
            _ => {}
        }
    }
    strip(&mut value);
    value
}

#[test]
fn scientific_construction_observations() {
    let record = std::env::var_os("SECA_RECORD_SCIENTIFIC_HKT").is_some();
    for path in cases() {
        let read = |name: &str| -> Value {
            serde_json::from_slice(&fs::read(path.join(name)).unwrap()).unwrap()
        };
        let actual = run(&read("input.json"));
        let reference = read("csharp.normalized.json");
        let comparison: Vec<_> = actual.as_array().unwrap().iter()
            .zip(reference.as_array().unwrap()).map(|(r,c)| json!({
                "topology_equal": without_expected(r["topology"].clone()) == without_expected(c["topology"].clone()),
                "decisions_equal": r["decisions"] == c["decisions"]
            })).collect();
        assert_eq!(
            actual.as_array().unwrap().len(),
            reference.as_array().unwrap().len()
        );
        if record {
            for (name, value) in [
                ("rust.observed.json", actual.clone()),
                ("comparison.json", json!(comparison)),
            ] {
                fs::write(
                    path.join(name),
                    serde_json::to_string_pretty(&value).unwrap() + "\n",
                )
                .unwrap();
            }
        }
        assert_eq!(actual, read("rust.observed.json"), "{}", path.display());
        assert_eq!(
            json!(comparison),
            read("comparison.json"),
            "{}",
            path.display()
        );
    }
}

#[test]
fn minimum_boundary_is_observed_in_baseline_and_reconstruction() {
    for phase in ["baseline", "rebuilt"] {
        for count in [2, 3, 4] {
            let path = directory().join(format!("boundary_{count}_{phase}"));
            let read = |name: &str| -> Value {
                serde_json::from_slice(&fs::read(path.join(name)).unwrap()).unwrap()
            };
            let cycle = usize::from(phase == "rebuilt");
            for (name, sufficient) in [
                ("csharp.normalized.json", count >= 3),
                ("rust.observed.json", count > 3),
            ] {
                let snapshots = read(name);
                let nodes = snapshots[cycle]["topology"]["nodes"].as_array().unwrap();
                let a = nodes.iter().find(|n| n["words"] == json!(["a"])).unwrap();
                assert_eq!(!a["child"].is_null(), sufficient, "{phase}/{count}/{name}");
            }
            if phase == "rebuilt" {
                for name in ["csharp.normalized.json", "rust.observed.json"] {
                    assert!(read(name)[cycle]["decisions"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|d| d["triggered"] == true));
                }
            }
        }
    }
}

#[test]
fn recursive_scope_difference_and_rebuild_entry_exception_are_explicit() {
    fn read(case: &str, language: &str) -> Value {
        serde_json::from_slice(
            &fs::read(directory().join(case).join(format!("{language}.json"))).unwrap(),
        )
        .unwrap()
    }
    fn descendants(scope: &Value) -> usize {
        scope["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| {
                if n["child"].is_null() {
                    0
                } else {
                    1 + descendants(&n["child"])
                }
            })
            .sum()
    }
    assert_eq!(
        descendants(&read("overlap_recursive", "csharp.normalized")[0]["topology"]),
        1
    );
    assert_eq!(
        descendants(&read("overlap_recursive", "rust.observed")[0]["topology"]),
        3
    );
    for language in ["csharp.normalized", "rust.observed"] {
        let snapshots = read("overlap_child_rebuilt", language);
        assert_eq!(
            snapshots[1]["decisions"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|d| d["triggered"] == true)
                .map(|d| d["path"].clone())
                .collect::<Vec<_>>(),
            vec![json!("root/0")]
        );
        assert_eq!(descendants(&snapshots[1]["topology"]), 3);
        assert_eq!(
            snapshots[0]["topology"]["nodes"][1],
            snapshots[1]["topology"]["nodes"][1]
        );
    }
    let raw = read("overlap_child_rebuilt", "csharp");
    let before = raw["snapshots"][0]["tree"].as_array().unwrap();
    let after = raw["snapshots"][1]["tree"].as_array().unwrap();
    let root = before.iter().find(|h| h["parentNodeId"] == 0).unwrap();
    // The independent d sibling retains the same raw node object, including its ID.
    let d = &root["nodes"][1];
    let next_root = after.iter().find(|h| h["parentNodeId"] == 0).unwrap();
    assert_eq!(d, &next_root["nodes"][1]);
}

#[test]
fn permutation_measurements_preserve_source_identity() {
    fn shape(scope: &Value, sort_siblings: bool) -> Value {
        let mut nodes: Vec<_> = scope["nodes"].as_array().unwrap().iter().map(|n| {
            json!({"words": n["words"], "sources": n["sources"],
                "child": if n["child"].is_null() { Value::Null } else { shape(&n["child"], sort_siblings) }})
        }).collect();
        if sort_siblings {
            nodes.sort_by_key(Value::to_string);
        }
        json!(nodes)
    }
    let mut measurements = Vec::new();
    for path in cases() {
        let name = path.file_name().unwrap().to_str().unwrap();
        let reversed = directory().join(format!("{name}_reversed"));
        if !reversed.exists() {
            continue;
        }
        for language in ["csharp.normalized", "rust.observed"] {
            let read = |folder: &std::path::Path| -> Value {
                serde_json::from_slice(&fs::read(folder.join(format!("{language}.json"))).unwrap())
                    .unwrap()
            };
            let a = read(&path);
            let b = read(&reversed);
            let equality = |sort| -> Vec<_> {
                a.as_array()
                    .unwrap()
                    .iter()
                    .zip(b.as_array().unwrap())
                    .map(|(x, y)| shape(&x["topology"], sort) == shape(&y["topology"], sort))
                    .collect()
            };
            measurements.push(json!({"case": name, "language": language,
                "ordered_equal": equality(false), "unordered_equal": equality(true)}));
        }
    }
    let recorded: Value =
        serde_json::from_slice(&fs::read(directory().join("permutation-comparison.json")).unwrap())
            .unwrap();
    assert_eq!(json!(measurements), recorded);
}
