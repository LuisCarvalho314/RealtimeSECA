use realtime_seca_core::{
    HktBuilderConfig, SecaEngine, SourceBatch,
};
use realtime_seca_core::config::{MemoryMode, SecaConfig, SecaThresholdConfig, TriggerPolicyMode};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

fn test_config() -> SecaConfig {
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
        trigger_policy_mode: TriggerPolicyMode::Placeholder,
    }
}

fn load_large_batch() -> SourceBatch {
    let path = Path::new("tests/data/large_batch.json");
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&contents)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

fn build_engine_for_batch(batch: SourceBatch) -> SecaEngine {
    let mut engine = SecaEngine::new(test_config()).expect("engine should be created");
    engine
        .build_baseline_tree(batch)
        .expect("baseline build should succeed");
    engine
}

fn validate_verbose_tree_invariants(
    tree: &realtime_seca_core::BaselineTreeVerboseExport,
) -> Result<(), String> {
    if tree.hkts.is_empty() {
        return Err("tree.hkts is empty".to_string());
    }

    let hkt_ids: BTreeSet<i32> = tree.hkts.iter().map(|hkt| hkt.hkt_id).collect();
    if hkt_ids.len() != tree.hkts.len() {
        return Err("duplicate HKT IDs detected".to_string());
    }

    let node_ids: BTreeSet<i32> = tree.nodes.iter().map(|node| node.node_id).collect();
    if node_ids.len() != tree.nodes.len() {
        return Err("duplicate node IDs detected".to_string());
    }

    let nodes_by_id: BTreeMap<i32, &realtime_seca_core::BaselineNodeVerboseExport> =
        tree.nodes.iter().map(|node| (node.node_id, node)).collect();

    let root_hkts: Vec<_> = tree
        .hkts
        .iter()
        .filter(|hkt| hkt.parent_node_id == 0)
        .collect();
    if root_hkts.is_empty() {
        return Err("no root HKT found".to_string());
    }

    for hkt in &tree.hkts {
        if hkt.parent_node_id != 0 && !node_ids.contains(&hkt.parent_node_id) {
            return Err(format!(
                "HKT {} parent_node_id {} does not reference an existing node",
                hkt.hkt_id, hkt.parent_node_id
            ));
        }

        if hkt.node_ids.is_empty() {
            return Err(format!("HKT {} has no node_ids", hkt.hkt_id));
        }

        for node_id in &hkt.node_ids {
            let Some(node) = nodes_by_id.get(node_id) else {
                return Err(format!(
                    "HKT {} references missing node {}",
                    hkt.hkt_id, node_id
                ));
            };

            if node.hkt_id != hkt.hkt_id {
                return Err(format!(
                    "Node {} belongs to HKT {} but is listed under HKT {}",
                    node.node_id, node.hkt_id, hkt.hkt_id
                ));
            }
        }
    }

    for node in &tree.nodes {
        if node.words.is_empty() {
            return Err(format!("Node {} has no words", node.node_id));
        }

        if node.sources.is_empty() {
            return Err(format!("Node {} has no sources", node.node_id));
        }

        if node.is_refuge_node {
            let has_refuge_word = node.words.iter().any(|word| word.word_id == -1);
            if !has_refuge_word {
                return Err(format!(
                    "Node {} is marked refuge but has no -1 word marker",
                    node.node_id
                ));
            }
        }
    }

    Ok(())
}

#[test]
fn large_dataset_baseline_build_and_verbose_export_succeeds() {
    let batch = load_large_batch();
    let engine = build_engine_for_batch(batch);

    let tree = engine
        .export_baseline_tree_verbose()
        .expect("verbose export should succeed");

    validate_verbose_tree_invariants(&tree).expect("tree invariants should hold");

    assert!(
        !tree.word_legend.is_empty(),
        "word legend should not be empty for a non-empty dataset"
    );
    assert!(
        !tree.source_legend.is_empty(),
        "source legend should not be empty for a non-empty dataset"
    );
}

#[test]
fn large_dataset_baseline_build_is_deterministic() {
    let batch_a = load_large_batch();
    let batch_b = load_large_batch();

    let engine_a = build_engine_for_batch(batch_a);
    let engine_b = build_engine_for_batch(batch_b);

    let tree_a = engine_a
        .export_baseline_tree_verbose()
        .expect("tree A export should succeed");
    let tree_b = engine_b
        .export_baseline_tree_verbose()
        .expect("tree B export should succeed");

    let json_a = serde_json::to_string_pretty(&tree_a).expect("serialize tree A");
    let json_b = serde_json::to_string_pretty(&tree_b).expect("serialize tree B");

    assert_eq!(
        json_a, json_b,
        "verbose export should be deterministic for identical input/config"
    );
}

#[test]
fn large_dataset_has_at_least_one_root_hkt_and_one_node() {
    let batch = load_large_batch();
    let engine = build_engine_for_batch(batch);

    let tree = engine
        .export_baseline_tree_verbose()
        .expect("verbose export should succeed");

    let root_count = tree
        .hkts
        .iter()
        .filter(|hkt| hkt.parent_node_id == 0)
        .count();
    assert!(root_count >= 1, "expected at least one root HKT");

    assert!(!tree.nodes.is_empty(), "expected at least one node");
}
