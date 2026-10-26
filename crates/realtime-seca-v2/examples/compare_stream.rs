//! Research file adapter. Neither algorithm depends on the other's types.
use realtime_seca_core::{SecaConfig, SecaEngine, SourceBatch};
use realtime_seca_v2::{metrics::Errors, Config, Hkt, Model, Source};
use serde_json::{json, Value};
use std::{collections::BTreeMap, error::Error, fs, time::Instant};

fn export(model: &Model) -> Value {
    fn visit(
        h: &Hkt,
        parent: usize,
        hkts: &mut Vec<Value>,
        nodes: &mut Vec<Value>,
        word_ids: &BTreeMap<String, usize>,
        source_ids: &BTreeMap<String, usize>,
    ) {
        let hid = hkts.len() + 1;
        hkts.push(json!({"hkt_id":hid,"parent_node_id":parent,"node_ids":[]}));
        let mut ids = Vec::new();
        for node in &h.nodes {
            let nid = nodes.len() + 1;
            ids.push(nid);
            let words: Vec<_> = if node.words.is_empty() {
                vec![json!({"word_id":-1,"token":"<REFUGE>"})]
            } else {
                node.words
                    .iter()
                    .map(|w| json!({"word_id":word_ids[w],"token":w}))
                    .collect()
            };
            let sources: Vec<_> = node
                .sources
                .iter()
                .map(|s| json!({"internal_source_id":source_ids[s],"external_source_id":s}))
                .collect();
            nodes.push(json!({"node_id":nid,"hkt_id":hid,"words":words,"sources":sources,"is_refuge_node":node.words.is_empty()}));
            if let Some(child) = &node.child {
                visit(child, nid, hkts, nodes, word_ids, source_ids);
            }
        }
        hkts[hid - 1]["node_ids"] = json!(ids);
    }
    fn vocabulary(h: &Hkt, words: &mut std::collections::BTreeSet<String>) {
        for n in &h.nodes {
            words.extend(n.words.iter().cloned());
            if let Some(c) = &n.child {
                vocabulary(c, words);
            }
        }
    }
    let mut words = Default::default();
    vocabulary(model.root(), &mut words);
    let word_ids: BTreeMap<_, _> = words
        .into_iter()
        .enumerate()
        .map(|(i, w)| (w, i + 1))
        .collect();
    let source_ids: BTreeMap<_, _> = model
        .retained_sources()
        .into_iter()
        .enumerate()
        .map(|(i, s)| (s, i + 1))
        .collect();
    let (mut hkts, mut nodes) = (Vec::new(), Vec::new());
    visit(
        model.root(),
        0,
        &mut hkts,
        &mut nodes,
        &word_ids,
        &source_ids,
    );
    json!({"hkts":hkts,"nodes":nodes,
        "word_legend":word_ids.iter().map(|(w,i)|json!({"word_id":i,"token":w})).collect::<Vec<_>>(),
        "source_legend":source_ids.iter().map(|(s,i)|json!({"internal_source_id":i,"external_source_id":s})).collect::<Vec<_>>()})
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        return Err("compare_stream v1|v2 CONFIG OUTPUT BATCH...".into());
    }
    let c: SecaConfig = serde_json::from_slice(&fs::read(&args[2])?)?;
    fs::create_dir(&args[3])?;
    let mut v1 = if args[1] == "v1" {
        Some(SecaEngine::new(c.clone())?)
    } else {
        None
    };
    if args[1] != "v1" && args[1] != "v2" {
        return Err("unknown engine".into());
    }
    let mut v2 = Model::new(Config {
        alpha: c.seca_thresholds.alpha,
        beta: c.seca_thresholds.beta,
        error_limits: Errors {
            alpha: c.seca_thresholds.alpha_error_threshold,
            beta: c.seca_thresholds.beta_error_threshold,
            importance: c.seca_thresholds.word_importance_error_threshold,
        },
        minimum_branch_sources: c
            .hkt_builder
            .minimum_number_of_sources_to_create_branch_for_node,
        window: if c.memory_mode == realtime_seca_core::MemoryMode::SlidingWindow {
            c.max_batches_in_memory.map(u64::from)
        } else {
            None
        },
    })?;
    if c.hkt_builder.minimum_threshold_against_max_word_count != c.seca_thresholds.alpha
        || c.hkt_builder.similarity_threshold != c.seca_thresholds.beta
    {
        return Err("comparison requires identical construction/evolution alpha and beta".into());
    }
    for input in &args[4..] {
        let b: SourceBatch = serde_json::from_slice(&fs::read(input)?)?;
        let started = Instant::now();
        let (tree, decisions) = if let Some(engine) = &mut v1 {
            let report = if b.batch_index == 0 {
                engine.build_baseline_tree(b.clone())?
            } else {
                engine.process_batch(b.clone())?
            };
            (
                serde_json::to_value(engine.export_baseline_tree_verbose()?)?,
                serde_json::to_value(report)?,
            )
        } else {
            let report = v2.update(
                b.batch_index.into(),
                b.sources
                    .iter()
                    .map(|s| Source {
                        id: s.source_id.clone(),
                        words: s.tokens.iter().cloned().collect(),
                    })
                    .collect(),
            )?;
            // Verify continuation every cycle, outside timed algorithm measurement below.
            (export(&v2), serde_json::to_value(report)?)
        };
        let seconds = started.elapsed().as_secs_f64();
        if v1.is_none() {
            let restored: Model = serde_json::from_slice(&serde_json::to_vec(&v2)?)?;
            assert_eq!(restored, v2);
        }
        fs::write(
            format!("{}/cycle-{:02}.json", args[3], b.batch_index),
            serde_json::to_vec(&json!({"tree":tree,"decisions":decisions,"seconds":seconds}))?,
        )?;
        println!("{} {} {:.6}", args[1], b.batch_index, seconds);
    }
    Ok(())
}
