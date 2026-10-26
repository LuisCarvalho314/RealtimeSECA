//! Read-only construction adapter, not an alternative implementation.
use realtime_seca_core::tree::{hkt_builder::HktBuilder, models::SourceWordRecord};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    let v: Value = serde_json::from_str(&input).unwrap();
    let rows = v["records"].as_array().unwrap();
    let total_start = std::time::Instant::now();
    let mut counts: BTreeMap<i32, BTreeSet<i64>> = BTreeMap::new();
    for r in rows {
        counts
            .entry(r["word"].as_i64().unwrap() as i32)
            .or_default()
            .insert(r["source"].as_i64().unwrap());
    }
    let records = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let w = r["word"].as_i64().unwrap() as i32;
            SourceWordRecord {
                source_word_id: i as i32 + 1,
                source_id: r["source"].as_i64().unwrap(),
                word_id: w,
                word: None,
                word_number_of_sources: counts[&w].len(),
            }
        })
        .collect();
    let builder = HktBuilder::new(
        v["alpha"].as_f64().unwrap(),
        v["beta"].as_f64().unwrap(),
        v["branchMinimum"].as_u64().unwrap() as usize,
    )
    .unwrap();
    let t = std::time::Instant::now();
    let tree = builder.build_full_tree(records, false).unwrap();
    let elapsed = t.elapsed().as_secs_f64();
    fn scope(tree: &realtime_seca_core::tree::hkt_builder::HktBuildOutput, parent: i32) -> Value {
        let Some(hkt) = tree
            .hkts_by_id
            .values()
            .find(|h| h.parent_node_id == parent)
        else {
            return json!([]);
        };
        Value::Array(
            hkt.nodes
                .iter()
                .map(|n| {
                    json!({"words":n.word_ids,
            "sources":n.source_ids,"word_sources":n.word_source_ids,
            "child":scope(tree,n.node_id)})
                })
                .collect(),
        )
    }
    let result = scope(&tree, 0);
    let total = total_start.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({"tree":result,"seconds":elapsed,"total_seconds":total})
    );
}
