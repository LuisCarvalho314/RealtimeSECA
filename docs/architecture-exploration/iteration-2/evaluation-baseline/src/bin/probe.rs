#[path = "../allocation.rs"]
mod allocation;
#[global_allocator]
static ALLOCATOR: allocation::Counting = allocation::Counting;
use realtime_seca_core::tree::{hkt_builder::HktBuilder, models::SourceWordRecord};
use seca_research_candidates::{build, Input};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
};
fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let input: Input = serde_json::from_value(v).unwrap();
    let mode = std::env::args().nth(1).unwrap();
    let before = allocation::start();
    let owned: Box<dyn std::any::Any> = if mode == "production" {
        let mut counts: BTreeMap<i32, BTreeSet<i64>> = BTreeMap::new();
        for r in &input.records {
            counts.entry(r.word).or_default().insert(r.source);
        }
        let rows = input
            .records
            .iter()
            .enumerate()
            .map(|(i, r)| SourceWordRecord {
                source_word_id: i as i32 + 1,
                source_id: r.source,
                word_id: r.word,
                word: None,
                word_number_of_sources: counts[&r.word].len(),
            })
            .collect();
        Box::new(
            HktBuilder::new(input.alpha, input.beta, input.branch_minimum)
                .unwrap()
                .build_full_tree(rows, false)
                .unwrap(),
        )
    } else {
        Box::new(build(&input, mode == "masks").unwrap())
    };
    let result = allocation::finish(before);
    std::hint::black_box(&owned);
    println!("{result}");
}
