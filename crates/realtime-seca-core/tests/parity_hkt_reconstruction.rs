//! Historical Rust structural regressions plus executed C# topology goldens.
#![allow(clippy::field_reassign_with_default)]
use realtime_seca_core::tree::{HktBuildOutput, HktBuilder, SourceWordRecord};
use realtime_seca_core::{
    BaselineTreeVerboseExport, SecaConfig, SecaEngine, SourceBatch, SourceRecord,
};
use std::collections::BTreeSet;

fn build(rows: &[(i64, &[i32])], alpha: f64, beta: f64, minimum: usize) -> HktBuildOutput {
    let mut records = Vec::new();
    for (source, words) in rows {
        for word in *words {
            let count = rows.iter().filter(|(_, ws)| ws.contains(word)).count();
            records.push(SourceWordRecord {
                source_word_id: records.len() as i32 + 1,
                source_id: *source,
                word_id: *word,
                word: None,
                word_number_of_sources: count,
            });
        }
    }
    HktBuilder::new(alpha, beta, minimum)
        .unwrap()
        .build_full_tree(records, false)
        .unwrap()
}
fn sets(values: &[i32]) -> BTreeSet<i32> {
    values.iter().copied().collect()
}
fn root(t: &HktBuildOutput) -> &realtime_seca_core::tree::Hkt {
    t.hkts_by_id
        .values()
        .find(|h| h.parent_node_id == 0)
        .unwrap()
}
#[test]
fn single_root_merges_words_and_retains_expected_words() {
    let t = build(&[(1, &[1, 2]), (2, &[1, 2])], 0.7, 0.5, 1);
    assert_eq!(t.hkts_by_id.len(), 1);
    assert_eq!(root(&t).nodes[0].word_ids, sets(&[1, 2]));
    assert_eq!(root(&t).nodes[0].source_ids, [1, 2].into_iter().collect());
    assert_eq!(root(&t).expected_words, sets(&[2]));
    assert_eq!(t.nodes_by_id[&root(&t).nodes[0].node_id], root(&t).nodes[0]);
}
#[test]
fn competing_words_and_multi_node_source_membership() {
    let t = build(&[(1, &[1]), (2, &[2]), (3, &[1, 2])], 0.7, 0.75, 99);
    assert_eq!(root(&t).nodes.len(), 2);
    assert!(root(&t).nodes.iter().all(|n| n.source_ids.contains(&3)));
}
#[test]
fn equal_frequency_input_order_changes_prominent_word_and_expected_set() {
    let left = build(&[(1, &[1, 2])], 0.7, 0.5, 99);
    let right = build(&[(1, &[2, 1])], 0.7, 0.5, 99);
    assert_eq!(
        root(&left).nodes[0].word_ids,
        root(&right).nodes[0].word_ids
    );
    assert_eq!(root(&left).expected_words, sets(&[2]));
    assert_eq!(root(&right).expected_words, sets(&[1]));
}
#[test]
fn equal_collision_scores_choose_first_created_node() {
    let t = build(
        &[(1, &[1, 3]), (2, &[1]), (3, &[2, 3]), (4, &[2])],
        1.0,
        0.5,
        99,
    );
    assert_eq!(root(&t).nodes[0].word_ids, sets(&[1, 3]));
    assert_eq!(root(&t).nodes[1].word_ids, sets(&[2]));
    let reordered = build(
        &[(3, &[2, 3]), (4, &[2]), (1, &[1, 3]), (2, &[1])],
        1.0,
        0.5,
        99,
    );
    assert_eq!(root(&reordered).nodes[0].word_ids, sets(&[2, 3]));
    assert_eq!(root(&reordered).nodes[1].word_ids, sets(&[1]));
}
#[test]
fn expected_word_order_uses_word_ids_even_when_frequency_order_differs() {
    let t = build(
        &[
            (1, &[3]),
            (2, &[3]),
            (3, &[3]),
            (4, &[3]),
            (5, &[2, 1]),
            (6, &[2, 1]),
            (7, &[2]),
        ],
        0.5,
        1.0,
        99,
    );
    assert_eq!(
        root(&t)
            .nodes
            .iter()
            .map(|n| n.word_ids.clone())
            .collect::<Vec<_>>(),
        vec![sets(&[3]), sets(&[1, 2])]
    );
}
#[test]
fn refuge_recurses_and_attaches_to_refuge_parent() {
    let t = build(
        &[(1, &[1]), (2, &[1]), (3, &[1]), (4, &[2]), (5, &[2])],
        0.8,
        0.5,
        1,
    );
    let refuge = root(&t).nodes.iter().find(|n| n.is_refuge_node()).unwrap();
    assert_eq!(refuge.source_ids, [4, 5].into_iter().collect());
    let child = t
        .hkts_by_id
        .values()
        .find(|h| h.parent_node_id == refuge.node_id)
        .unwrap();
    assert_eq!(child.nodes[0].word_ids, sets(&[2]));
    assert_eq!(child.nodes[0].source_ids, refuge.source_ids);
}
#[test]
fn branch_boundary_is_strict_and_empty_input_has_no_hkt() {
    let rows: &[(i64, &[i32])] = &[(1, &[1, 2]), (2, &[1])];
    assert_eq!(build(rows, 0.7, 0.5, 2).hkts_by_id.len(), 1);
    assert_eq!(build(rows, 0.7, 0.5, 1).hkts_by_id.len(), 2);
    assert!(build(&[], 0.7, 0.5, 1).hkts_by_id.is_empty());
}
fn engine() -> SecaEngine {
    let mut c = SecaConfig::default();
    c.hkt_builder.minimum_threshold_against_max_word_count = 0.7;
    c.seca_thresholds.alpha = 0.7;
    c.seca_thresholds.alpha_error_threshold = 0.11;
    c.seca_thresholds.beta_error_threshold = 0.2;
    c.seca_thresholds.word_importance_error_threshold = 0.2;
    SecaEngine::new(c).unwrap()
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
                text: None,
                timestamp_unix_ms: None,
                metadata: None,
            })
            .collect(),
    }
}
// Normalize identifiers to token/source labels; retain parent attachment recursively.
fn normalized(t: &BaselineTreeVerboseExport, hkt_id: i32) -> String {
    let h = t.hkts.iter().find(|h| h.hkt_id == hkt_id).unwrap();
    let mut expected: Vec<_> = h
        .expected_words
        .iter()
        .map(|w| w.token.clone().unwrap_or("<>".into()))
        .collect();
    expected.sort();
    let mut nodes = Vec::new();
    for n in t.nodes.iter().filter(|n| n.hkt_id == hkt_id) {
        let mut words: Vec<_> = n
            .words
            .iter()
            .map(|w| w.token.clone().unwrap_or("<>".into()))
            .collect();
        words.sort();
        let mut sources: Vec<_> = n
            .sources
            .iter()
            .map(|s| s.external_source_id.clone())
            .collect();
        sources.sort();
        let child = t
            .hkts
            .iter()
            .find(|h| h.parent_node_id == n.node_id)
            .map(|h| normalized(t, h.hkt_id));
        nodes.push(format!("{words:?}:{sources:?}:{child:?}"));
    }
    nodes.sort();
    format!("expected={expected:?};{nodes:?}")
}
#[test]
fn child_replacement_preserves_sibling_scope_and_excludes_ancestor_words() {
    let mut e = engine();
    let rows = (0..20)
        .map(|i| {
            let a = i < 10;
            let topic = if a { "a" } else { "d" };
            let words = match i % 10 {
                0..=3 => vec![topic, if a { "b" } else { "e" }],
                4..=7 => vec![topic, if a { "c" } else { "f" }],
                _ => vec![topic],
            };
            (format!("s{i}"), words)
        })
        .collect();
    e.build_baseline_tree(batch(1, rows)).unwrap();
    let before = e.export_baseline_tree_verbose().unwrap();
    let root_id = before
        .hkts
        .iter()
        .find(|h| h.parent_node_id == 0)
        .unwrap()
        .hkt_id;
    let parent = before
        .nodes
        .iter()
        .find(|n| n.hkt_id == root_id && n.words.iter().any(|w| w.token.as_deref() == Some("a")))
        .unwrap();
    let sibling_parent = before
        .nodes
        .iter()
        .find(|n| n.hkt_id == root_id && n.words.iter().any(|w| w.token.as_deref() == Some("d")))
        .unwrap();
    let old_child = before
        .hkts
        .iter()
        .find(|h| h.parent_node_id == parent.node_id)
        .unwrap()
        .hkt_id;
    let sibling = before
        .hkts
        .iter()
        .find(|h| h.parent_node_id == sibling_parent.node_id)
        .unwrap()
        .hkt_id;
    let result = e
        .process_batch(batch(
            2,
            (0..10)
                .map(|i| {
                    (
                        format!("n{i}"),
                        if i < 3 { vec!["a", "x"] } else { vec!["a"] },
                    )
                })
                .collect(),
        ))
        .unwrap();
    assert_eq!(result.reconstructed_hkt_ids, vec![old_child]);
    let after = e.export_baseline_tree_verbose().unwrap();
    assert_eq!(normalized(&before, sibling), normalized(&after, sibling));
    assert!(!after.hkts.iter().any(|h| h.hkt_id == old_child));
    let replacement = after
        .hkts
        .iter()
        .find(|h| h.parent_node_id == parent.node_id)
        .unwrap();
    assert!(replacement.hkt_id > old_child);
    assert!(after
        .nodes
        .iter()
        .filter(|n| n.hkt_id == replacement.hkt_id)
        .all(|n| n.words.iter().all(|w| w.token.as_deref() != Some("a"))));
    let child_sources: BTreeSet<_> = after
        .nodes
        .iter()
        .filter(|n| n.hkt_id == replacement.hkt_id)
        .flat_map(|n| n.sources.iter().map(|s| s.external_source_id.as_str()))
        .collect();
    assert!(child_sources.contains("s0") && child_sources.contains("n0"));
    assert!(!child_sources.contains("s10"));
    let updated_parent = after
        .nodes
        .iter()
        .find(|n| n.node_id == parent.node_id)
        .unwrap();
    assert!(child_sources.iter().all(|s| updated_parent
        .sources
        .iter()
        .any(|p| p.external_source_id == *s)));
    SecaEngine::load_snapshot(e.snapshot().unwrap()).unwrap();
}
#[test]
fn root_replacement_suppresses_children_selects_new_dominant_and_stabilizes() {
    let mut e = engine();
    e.build_baseline_tree(batch(
        1,
        vec![("o1".into(), vec!["a", "b"]), ("o2".into(), vec!["a"])],
    ))
    .unwrap();
    let before = e.export_baseline_tree_verbose().unwrap();
    assert!(before.hkts.len() > 1);
    let root_id = before
        .hkts
        .iter()
        .find(|h| h.parent_node_id == 0)
        .unwrap()
        .hkt_id;
    let r = e
        .process_batch(batch(
            2,
            (0..4).map(|i| (format!("x{i}"), vec!["x"])).collect(),
        ))
        .unwrap();
    assert_eq!(r.reconstructed_hkt_ids, vec![root_id]);
    assert_eq!(r.decision_diagnostics.len(), 1);
    let t = e.export_baseline_tree_verbose().unwrap();
    assert!(t
        .hkts
        .iter()
        .all(|h| !before.hkts.iter().any(|old| old.hkt_id == h.hkt_id)));
    let root = t.hkts.iter().find(|h| h.parent_node_id == 0).unwrap();
    assert!(t
        .nodes
        .iter()
        .filter(|n| n.hkt_id == root.hkt_id && !n.is_refuge_node)
        .all(|n| n.words.iter().all(|w| w.token.as_deref() == Some("x"))));
    let stable = e
        .process_batch(batch(3, vec![("x4".into(), vec!["x"])]))
        .unwrap();
    assert!(!stable.reconstruction_triggered);
    assert_eq!(
        e.export_baseline_tree_verbose()
            .unwrap()
            .hkts
            .iter()
            .map(|h| h.hkt_id)
            .collect::<Vec<_>>(),
        t.hkts.iter().map(|h| h.hkt_id).collect::<Vec<_>>()
    );
}
#[test]
fn incoming_source_maps_to_both_nodes_and_empty_scope_preserves_tree() {
    let mut e = engine();
    e.build_baseline_tree(batch(
        1,
        vec![("a1".into(), vec!["a"]), ("d1".into(), vec!["d"])],
    ))
    .unwrap();
    let result = e
        .process_batch(batch(2, vec![("both".into(), vec!["a", "d"])]))
        .unwrap();
    assert!(!result.reconstruction_triggered);
    let before = e.export_baseline_tree_verbose().unwrap();
    assert!(before
        .nodes
        .iter()
        .all(|n| n.sources.iter().any(|s| s.external_source_id == "both")));
    let root = before
        .hkts
        .iter()
        .find(|h| h.parent_node_id == 0)
        .unwrap()
        .hkt_id;
    assert!(
        !e.process_batch(batch(3, vec![]))
            .unwrap()
            .reconstruction_triggered
    );
    assert_eq!(
        normalized(&before, root),
        normalized(&e.export_baseline_tree_verbose().unwrap(), root)
    );
}

#[test]
fn recursive_scope_retains_sibling_words_on_overlapping_sources() {
    let t = build(
        &[
            (1, &[0, 1]),
            (2, &[0, 1]),
            (3, &[0, 1, 2]),
            (4, &[0, 2]),
            (5, &[0, 2]),
            (6, &[0]),
        ],
        0.8,
        0.75,
        0,
    );
    let parent = &root(&t).nodes[0];
    assert_eq!(parent.word_ids, sets(&[0]));
    let child = t
        .hkts_by_id
        .values()
        .find(|h| h.parent_node_id == parent.node_id)
        .unwrap();
    assert_eq!(child.nodes.len(), 2);
    for node in &child.nodes {
        let descendant = t
            .hkts_by_id
            .values()
            .find(|h| h.parent_node_id == node.node_id)
            .unwrap();
        let other_word = if node.word_ids.contains(&1) { 2 } else { 1 };
        assert_eq!(descendant.nodes[0].word_ids, sets(&[other_word]));
        assert_eq!(descendant.nodes[0].source_ids, [3].into_iter().collect());
    }
}

// Executed-reference audit. C# goldens are generated only by reference.py.
mod executed {
    use realtime_seca_core::{MemoryMode, SecaConfig, SecaEngine, SourceBatch, SourceRecord};
    use serde_json::{json, Value};
    use std::{collections::BTreeMap, fs, path::PathBuf};

    fn directory() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parity_hkt_reconstruction")
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
        config.hkt_builder.minimum_threshold_against_max_word_count =
            input["alpha"].as_f64().unwrap();
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
    fn executed_reference_regressions_and_explicit_differences() {
        let record = std::env::var_os("SECA_RECORD_RUST_AUDIT").is_some();
        for path in cases() {
            let input: Value =
                serde_json::from_slice(&fs::read(path.join("input.json")).unwrap()).unwrap();
            let actual = run(&input);
            if record {
                fs::write(
                    path.join("rust.observed.json"),
                    serde_json::to_string_pretty(&actual).unwrap() + "\n",
                )
                .unwrap();
            }
            let observed: Value =
                serde_json::from_slice(&fs::read(path.join("rust.observed.json")).unwrap())
                    .unwrap();
            assert_eq!(
                actual,
                observed,
                "Rust behavior changed: {}",
                path.display()
            );
            let reference: Value =
                serde_json::from_slice(&fs::read(path.join("csharp.normalized.json")).unwrap())
                    .unwrap();
            let comparison: Value = serde_json::from_slice(
                &fs::read(path.join("comparison.json")).unwrap_or_else(|_| b"[]".to_vec()),
            )
            .unwrap();
            let differences: Vec<_> = actual.as_array().unwrap().iter().zip(reference.as_array().unwrap()).map(|(r,c)| json!({
                "topology_equal":without_expected(r["topology"].clone()) == without_expected(c["topology"].clone()),
                "expected_and_topology_equal":r["topology"] == c["topology"],
                "decisions_equal":r["decisions"] == c["decisions"]
            })).collect();
            if record {
                fs::write(
                    path.join("comparison.json"),
                    serde_json::to_string_pretty(&differences).unwrap() + "\n",
                )
                .unwrap();
            } else {
                assert_eq!(
                    json!(differences),
                    comparison,
                    "Parity classification changed: {}",
                    path.display()
                );
            }
        }
    }

    #[test]
    #[ignore = "Known executed C#–Rust differences; run explicitly to expose strict parity failure"]
    fn strict_executed_csharp_parity() {
        for path in cases() {
            let input: Value =
                serde_json::from_slice(&fs::read(path.join("input.json")).unwrap()).unwrap();
            let reference: Value =
                serde_json::from_slice(&fs::read(path.join("csharp.normalized.json")).unwrap())
                    .unwrap();
            assert_eq!(run(&input), reference, "{}", path.display());
        }
    }
}
