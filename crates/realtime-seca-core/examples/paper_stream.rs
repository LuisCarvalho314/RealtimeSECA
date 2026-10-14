//! Experimental file boundary over the canonical public engine API.
use realtime_seca_core::{EngineSnapshot, SecaConfig, SecaEngine, SourceBatch};
use std::{error::Error, fs, io::Write, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("timeline-many") {
        return Err("expected timeline-many INPUTS --config CONFIG --out-dir OUTPUT".into());
    }
    let (mut inputs, mut config_path, mut output, mut state) = (Vec::new(), None, None, None);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => config_path = Some(PathBuf::from(args.next().ok_or("missing config")?)),
            "--out-dir" => output = Some(PathBuf::from(args.next().ok_or("missing output")?)),
            "--state-in" => state = Some(PathBuf::from(args.next().ok_or("missing state")?)),
            flag if flag.starts_with('-') => return Err(format!("unknown flag {flag}").into()),
            _ => inputs.push(PathBuf::from(arg)),
        }
    }
    if inputs.is_empty() {
        return Err("no batches".into());
    }
    let config: SecaConfig = serde_json::from_slice(&fs::read(config_path.ok_or("no config")?)?)?;
    let output = output.ok_or("no output")?;
    let (mut engine, last) = if let Some(path) = state {
        let snapshot: EngineSnapshot = serde_json::from_slice(&fs::read(path)?)?;
        let last = snapshot.last_processed_batch_index;
        let engine = SecaEngine::load_snapshot(snapshot)?;
        if engine.config() != &config {
            return Err("configuration differs from snapshot".into());
        }
        (engine, last)
    } else {
        (SecaEngine::new(config)?, None)
    };
    // Refuse existing output rather than overwrite scientific evidence.
    fs::create_dir(&output)?;
    for (index, path) in inputs.iter().enumerate() {
        let batch: SourceBatch = serde_json::from_slice(&fs::read(path)?)?;
        if batch.batch_index as usize != index {
            return Err("batch sequence differs from manifest".into());
        }
        if last.is_some_and(|last| batch.batch_index <= last) {
            continue;
        }
        let started = Instant::now();
        let result = if index == 0 {
            engine.build_baseline_tree(batch)?
        } else {
            engine.process_batch(batch)?
        };
        fs::write(
            output.join(format!("tree_batch_{index:04}.json")),
            serde_json::to_vec(&engine.export_baseline_tree_verbose()?)?,
        )?;
        fs::write(
            output.join(format!("snapshot-{index:04}.json")),
            serde_json::to_vec(&engine.snapshot()?)?,
        )?;
        fs::write(
            output.join(format!("report-{index:04}.json")),
            serde_json::to_vec(&result)?,
        )?;
        println!(
            "cycle={} elapsed_seconds={:.3}",
            index + 1,
            started.elapsed().as_secs_f64()
        );
        std::io::stdout().flush()?;
    }
    Ok(())
}
