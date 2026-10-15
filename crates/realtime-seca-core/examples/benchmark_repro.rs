//! External benchmark adapter: no algorithm changes or evaluation clustering.
use realtime_seca_core::{MemoryMode, SecaConfig, SecaEngine, SourceBatch, SourceRecord};
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeSet, error::Error, fs, io::Write};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Input {
    alpha: f64,
    beta: f64,
    branch_minimum: usize,
    alpha_error: f64,
    beta_error: f64,
    word_importance_error: f64,
    batches: Vec<Vec<Document>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    id: u32,
    words: Vec<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args().nth(1).ok_or("expected input JSON path")?;
    let input: Input = serde_json::from_slice(&fs::read(path)?)?;
    let mut ids = BTreeSet::new();
    if input.batches.is_empty() {
        return Err("no batches".into());
    }
    for batch in &input.batches {
        if batch.is_empty() {
            return Err("empty batch is excluded by common reference contract".into());
        }
        for doc in batch {
            let words: BTreeSet<_> = doc.words.iter().collect();
            if doc.id == 0
                || !ids.insert(doc.id)
                || doc.words.is_empty()
                || words.len() != doc.words.len()
                || doc.words.iter().any(|w| w.is_empty() || w.trim() != w)
            {
                return Err("invalid identity or noncanonical tokens".into());
            }
        }
    }
    let mut config = SecaConfig {
        memory_mode: MemoryMode::Full,
        ..SecaConfig::default()
    };
    config.hkt_builder.minimum_threshold_against_max_word_count = input.alpha;
    config.hkt_builder.similarity_threshold = input.beta;
    config
        .hkt_builder
        .minimum_number_of_sources_to_create_branch_for_node = input.branch_minimum;
    config.seca_thresholds.alpha = input.alpha;
    config.seca_thresholds.beta = input.beta;
    config.seca_thresholds.alpha_error_threshold = input.alpha_error;
    config.seca_thresholds.beta_error_threshold = input.beta_error;
    config.seca_thresholds.word_importance_error_threshold = input.word_importance_error;
    let mut engine = SecaEngine::new(config)?;
    let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
    for (cycle, docs) in input.batches.into_iter().enumerate() {
        let index = u32::try_from(cycle + 1)?;
        let batch = SourceBatch {
            batch_index: index,
            sources: docs
                .into_iter()
                .map(|d| SourceRecord {
                    source_id: d.id.to_string(),
                    batch_index: index,
                    tokens: d.words,
                    text: None,
                    timestamp_unix_ms: None,
                    metadata: None,
                })
                .collect(),
        };
        let report = if cycle == 0 {
            engine.build_baseline_tree(batch)?;
            None
        } else {
            Some(engine.process_batch(batch)?)
        };
        // Flush completed cycles so an interrupted experiment has explicit progress.
        serde_json::to_writer(
            &mut stdout,
            &json!({"cycle":cycle,
            "tree":engine.export_baseline_tree_verbose()?, "report":report}),
        )?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}
