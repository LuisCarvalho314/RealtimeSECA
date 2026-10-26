use realtime_seca_v2::{metrics::Errors, Config, Hkt, Model, Source};
use serde_json::{json, Value};
use std::path::PathBuf;

fn fixture_directory(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../realtime-seca-core/tests/fixtures")
        .join("parity_hkt_reconstruction")
        .join(name)
}

fn read_json(path: PathBuf) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

// Canonical tree: no IDs, expected-word caches,
// decision paths, or sibling ordering.
fn canonical_nodes(mut nodes: Vec<Value>) -> Value {
    nodes.sort_by_key(|n| n.to_string());
    json!({ "nodes": nodes })
}

fn normalize_csharp(hkt: &Value) -> Value {
    let nodes = hkt["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| {
            let child = if n["child"].is_null() {
                Value::Null
            } else {
                normalize_csharp(&n["child"])
            };

            json!({
                "words": n["words"],
                "sources": n["sources"],
                "child": child,
            })
        })
        .collect();

    canonical_nodes(nodes)
}

fn normalize_v2(hkt: &Hkt) -> Value {
    let nodes = hkt
        .nodes
        .iter()
        .map(|n| {
            let words: Vec<String> = if n.words.is_empty() {
                vec!["<>".into()]
            } else {
                n.words.iter().cloned().collect()
            };

            let child = match n.child.as_deref() {
                Some(child) => normalize_v2(child),
                None => Value::Null,
            };

            json!({
                "words": words,
                "sources": n.sources,
                "child": child,
            })
        })
        .collect();

    canonical_nodes(nodes)
}

fn compare_fixture(name: &str) {
    let dir = fixture_directory(name);

    let input = read_json(dir.join("input.json"));
    let reference = read_json(dir.join("csharp.normalized.json"));

    let config = Config {
        alpha: input["alpha"].as_f64().unwrap(),
        beta: input["beta"].as_f64().unwrap(),
        minimum_branch_sources: input["branchMinimum"].as_u64().unwrap() as usize,
        error_limits: Errors {
            alpha: input["alphaError"].as_f64().unwrap(),
            beta: input["betaError"].as_f64().unwrap(),
            importance: input["wordImportanceError"].as_f64().unwrap(),
        },
        window: None,
    };

    let mut model = Model::new(config).unwrap();

    let batches = input["batches"].as_array().unwrap();
    let snapshots = reference.as_array().unwrap();

    assert_eq!(batches.len(), snapshots.len());

    for (cycle, batch) in batches.iter().enumerate() {
        let documents: Vec<Source> = batch
            .as_array()
            .unwrap()
            .iter()
            .map(|doc| Source {
                id: doc["id"].as_u64().unwrap().to_string(),
                words: doc["words"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|w| w.as_str().unwrap().to_owned())
                    .collect(),
            })
            .collect();

        model.update(cycle as u64, documents).unwrap();

        let actual = normalize_v2(model.root());
        let expected = normalize_csharp(&snapshots[cycle]["topology"]);

        assert_eq!(
            actual,
            expected,
            "Tree mismatch: fixture={name}, cycle={}",
            cycle + 1,
        );

        println!("{name}, cycle {}: MATCH", cycle + 1);
    }
}

#[test]
fn csharp_child_tree_parity() {
    compare_fixture("child");
}

#[test]
fn csharp_child_reconstruction_tree_parity() {
    compare_fixture("child_reconstruct");
}

#[test]
fn csharp_root_reconstruction_tree_parity() {
    compare_fixture("root_reconstruct");
}

#[test]
#[ignore = "C# counts parent sources; v2 counts residual-bearing sources"]
fn csharp_branch_boundary_tree_parity() {
    compare_fixture("branch_boundary");
}

#[test]
fn csharp_overlap_recursive_tree_parity() {
    compare_fixture("overlap_recursive");
}

#[test]
fn csharp_unmatched_tree_parity() {
    compare_fixture("unmatched");
}

#[test]
#[ignore = "C# uses a different equal-overlap tie rule"]
fn csharp_collision_true_tie_tree_parity() {
    compare_fixture("collision_true_tie");
}

#[test]
fn csharp_one_prominent_tree_parity() {
    compare_fixture("one_prominent");
}

#[test]
fn csharp_merge_many_tree_parity() {
    compare_fixture("merge_many");
}

#[test]
fn csharp_separate_tree_parity() {
    compare_fixture("separate");
}

#[test]
fn csharp_frequency_order_tree_parity() {
    compare_fixture("frequency_vs_vocabulary");
}

#[test]
#[ignore = "C# child refuge membership exceeds parent membership"]
fn csharp_refuge_tree_parity() {
    compare_fixture("refuge");
}
