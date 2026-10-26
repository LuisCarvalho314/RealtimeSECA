//! Production-only lifecycle adapter; normalization is explicit in emitted views.
use realtime_seca_core::{
    config::{MemoryMode, SecaConfig},
    engine::SecaEngine,
    types::SourceBatch,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn batch(index: u64, sources: &Value) -> SourceBatch {
    serde_json::from_value(json!({"batch_index":index,"sources":sources.as_array().unwrap().iter().map(|s|
        json!({"source_id":s["id"].as_i64().unwrap().to_string(),"batch_index":index,
          "tokens":s["words"].as_array().unwrap().iter().map(|w|format!("{:010}",w.as_i64().unwrap())).collect::<Vec<_>>(),
          "text":null,"timestamp_unix_ms":null,"metadata":null})).collect::<Vec<_>>()})).unwrap()
}
fn normalize(engine: &SecaEngine) -> Value {
    let state = serde_json::to_value(engine).unwrap();
    let words: BTreeMap<i64, i64> = state["baseline_word_legend"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.parse().unwrap(), v.as_str().unwrap().parse().unwrap()))
        .chain([(-1, -1)])
        .collect();
    let sources: BTreeMap<i64, i64> = state["baseline_source_legend"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.parse().unwrap(), v.as_str().unwrap().parse().unwrap()))
        .collect();
    let ws = |v: &Value| {
        let mut xs: Vec<_> = v
            .as_array()
            .unwrap()
            .iter()
            .map(|x| words[&x.as_i64().unwrap()])
            .collect();
        xs.sort();
        json!(xs)
    };
    let ss = |v: &Value| {
        let mut xs: Vec<_> = v
            .as_array()
            .unwrap()
            .iter()
            .map(|x| sources[&x.as_i64().unwrap()])
            .collect();
        xs.sort();
        json!(xs)
    };
    fn scope(
        state: &Value,
        parent: i64,
        words: &dyn Fn(&Value) -> Value,
        sources: &dyn Fn(&Value) -> Value,
        map: &BTreeMap<i64, i64>,
    ) -> Value {
        let Some(h) = state["hkt_build_output"]["hkts_by_id"]
            .as_object()
            .unwrap()
            .values()
            .find(|h| h["parent_node_id"] == parent)
        else {
            return Value::Null;
        };
        json!({"hkt_id":h["hkt_id"],"parent_node_id":h["parent_node_id"],"expected_words":words(&h["expected_words"]),
          "nodes":h["nodes"].as_array().unwrap().iter().map(|n| {
            let memberships:serde_json::Map<String,Value>=n["word_source_ids"].as_object().unwrap().iter().map(|(w,ss)|
                (map[&w.parse::<i64>().unwrap()].to_string(),sources(ss))).collect();
            json!({"node_id":n["node_id"],"hkt_id":n["hkt_id"],"words":words(&n["word_ids"]),"sources":sources(&n["source_ids"]),
              "word_sources":memberships,"top_words":words(&n["top_words"]),"child":scope(state,n["node_id"].as_i64().unwrap(),words,sources,map)})
          }).collect::<Vec<_>>()})
    }
    scope(&state, 0, &ws, &ss, &words)
}
pub fn run(v: &Value) -> Value {
    let c = &v["construction"];
    let mut config = SecaConfig::default();
    config.hkt_builder.minimum_threshold_against_max_word_count = c["alpha"].as_f64().unwrap();
    config.hkt_builder.similarity_threshold = c["beta"].as_f64().unwrap();
    config
        .hkt_builder
        .minimum_number_of_sources_to_create_branch_for_node =
        c["branchMinimum"].as_u64().unwrap() as usize;
    config.seca_thresholds.alpha = c["alpha"].as_f64().unwrap();
    config.seca_thresholds.beta = c["beta"].as_f64().unwrap();
    config.seca_thresholds.alpha_error_threshold = v["alpha_error_threshold"].as_f64().unwrap();
    config.seca_thresholds.beta_error_threshold = v["beta_error_threshold"].as_f64().unwrap();
    config.seca_thresholds.word_importance_error_threshold =
        v["importance_error_threshold"].as_f64().unwrap();
    if let Some(gamma) = v["gamma"].as_u64() {
        config.memory_mode = MemoryMode::SlidingWindow;
        config.max_batches_in_memory = Some(gamma as u32);
    }
    let mut grouped: Vec<Value> = Vec::new();
    for r in c["records"].as_array().unwrap() {
        if let Some(s) = grouped.iter_mut().find(|s| s["id"] == r["source"]) {
            s["words"].as_array_mut().unwrap().push(r["word"].clone());
        } else {
            grouped.push(json!({"id":r["source"],"words":[r["word"]]}));
        }
    }
    let mut engine = SecaEngine::new(config).unwrap();
    let start = std::time::Instant::now();
    engine
        .build_baseline_tree(batch(
            v["initial_index"].as_u64().unwrap_or(0),
            &json!(grouped),
        ))
        .unwrap();
    let mut states = vec![normalize(&engine)];
    let mut updates = Vec::new();
    let mut times = Vec::new();
    for b in v["batches"].as_array().unwrap() {
        let previous = serde_json::to_value(&engine).unwrap();
        let incoming = batch(b["index"].as_u64().unwrap(), &b["sources"]);
        let update_start = std::time::Instant::now();
        let update = engine.process_batch(incoming);
        times.push(update_start.elapsed().as_secs_f64());
        match update{
          Ok(report)=>updates.push(json!({"ok":true,"report":report})),
          Err(error)=>updates.push(json!({"ok":false,"error":error.to_string(),"unchanged":serde_json::to_value(&engine).unwrap()==previous})),
        }
        // Exercise production snapshot validation and restart at every successful boundary.
        engine = SecaEngine::load_snapshot(
            serde_json::from_str(&serde_json::to_string(&engine.snapshot().unwrap()).unwrap())
                .unwrap(),
        )
        .unwrap();
        states.push(normalize(&engine));
    }
    json!({"states":states,"updates":updates,"seconds":start.elapsed().as_secs_f64(),"update_seconds":times})
}
