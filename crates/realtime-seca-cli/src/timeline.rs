use crate::ingestion::{read_config_json, read_source_batch_json};
use crate::presentation::print_usage;
use realtime_seca_core::{SecaConfig, SecaEngine, SourceBatch};
use serde::Serialize;
use std::{fs, path::Path};
#[derive(Debug, Serialize)]
struct TimelineManifest {
    total_batches: usize,
    sources_total: usize,
    chunk_count: usize,
    chunk_size_effective: usize,
    files: Vec<String>,
}

pub(crate) fn run_timeline_command(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if arguments.len() < 3 {
        print_usage(arguments);
        return Ok(());
    }

    let input_path = arguments[2].clone();
    let mut out_dir: Option<String> = None;
    let mut config_path: Option<String> = None;
    let mut chunk_count: Option<usize> = None;
    let mut chunk_size: Option<usize> = None;
    let mut clean_out_dir = false;

    let mut index = 3;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--out-dir" => {
                let Some(value) = arguments.get(index + 1) else {
                    return Err("missing value for --out-dir".into());
                };
                out_dir = Some(value.clone());
                index += 2;
            }
            "--config" => {
                let Some(value) = arguments.get(index + 1) else {
                    return Err("missing value for --config".into());
                };
                config_path = Some(value.clone());
                index += 2;
            }
            "--chunk-count" => {
                let Some(value) = arguments.get(index + 1) else {
                    return Err("missing value for --chunk-count".into());
                };
                chunk_count = Some(value.parse::<usize>()?);
                index += 2;
            }
            "--chunk-size" => {
                let Some(value) = arguments.get(index + 1) else {
                    return Err("missing value for --chunk-size".into());
                };
                chunk_size = Some(value.parse::<usize>()?);
                index += 2;
            }
            "--clean-out-dir" => {
                clean_out_dir = true;
                index += 1;
            }
            unknown_flag => return Err(format!("unknown argument: {unknown_flag}").into()),
        }
    }

    if chunk_count.is_some() && chunk_size.is_some() {
        return Err("use either --chunk-count or --chunk-size, not both".into());
    }

    let out_dir = out_dir.ok_or_else(|| "--out-dir is required".to_string())?;
    let batch = read_source_batch_json(&input_path)?;
    if batch.sources.is_empty() {
        return Err("input batch has no sources".into());
    }

    let total_sources = batch.sources.len();
    let resolved_chunk_count = match (chunk_count, chunk_size) {
        (Some(count), None) => count,
        (None, Some(size)) => {
            if size == 0 {
                return Err("--chunk-size must be > 0".into());
            }
            total_sources.div_ceil(size)
        }
        (None, None) => 8,
        (Some(_), Some(_)) => unreachable!(),
    };
    if resolved_chunk_count == 0 {
        return Err("--chunk-count must be > 0".into());
    }

    let chunks = split_batch_into_sequential_chunks(batch, resolved_chunk_count)?;
    let chunk_size_effective = chunks.first().map(|chunk| chunk.sources.len()).unwrap_or(0);

    let output_dir = Path::new(&out_dir);
    if clean_out_dir && output_dir.exists() {
        fs::remove_dir_all(output_dir)?;
    }
    fs::create_dir_all(output_dir)?;

    let config = if let Some(path) = config_path {
        read_config_json(path)?
    } else {
        SecaConfig::default()
    };

    let mut engine = SecaEngine::new(config)?;
    let manifest = write_timeline(
        &mut engine,
        chunks,
        output_dir,
        total_sources,
        chunk_size_effective,
    )?;
    let manifest_path = output_dir.join("timeline_manifest.json");

    println!("Timeline build completed");
    println!("Output directory: {}", output_dir.display());
    println!("Manifest written to {}", manifest_path.display());
    println!("Batches exported: {}", manifest.total_batches);

    Ok(())
}

pub(crate) fn split_batch_into_sequential_chunks(
    batch: SourceBatch,
    chunk_count: usize,
) -> Result<Vec<SourceBatch>, Box<dyn std::error::Error>> {
    if chunk_count == 0 {
        return Err("chunk_count must be > 0".into());
    }
    if batch.sources.is_empty() {
        return Err("cannot split an empty batch".into());
    }

    let total = batch.sources.len();
    let base = total / chunk_count;
    let remainder = total % chunk_count;
    let mut sources = batch.sources.into_iter();
    let mut chunks = Vec::with_capacity(chunk_count);

    for chunk_index in 0..chunk_count {
        let length = base + usize::from(chunk_index < remainder);
        let chunk_sources = sources
            .by_ref()
            .take(length)
            .map(|mut source| {
                source.batch_index = chunk_index as u32;
                source
            })
            .collect();

        chunks.push(SourceBatch {
            batch_index: chunk_index as u32,
            sources: chunk_sources,
        });
    }

    Ok(chunks)
}

pub(crate) fn run_timeline_many_command(
    arguments: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut inputs: Vec<String> = Vec::new();
    let mut out_dir: Option<String> = None;
    let mut config_path: Option<String> = None;
    let mut clean_out_dir = false;
    let mut index = 2;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--out-dir" | "--config" => {
                let flag = &arguments[index];
                let value = arguments
                    .get(index + 1)
                    .ok_or_else(|| format!("missing value for {flag}"))?;
                if flag == "--out-dir" {
                    out_dir = Some(value.clone());
                } else {
                    config_path = Some(value.clone());
                }
                index += 2;
            }
            "--clean-out-dir" => {
                clean_out_dir = true;
                index += 1;
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown argument: {value}").into())
            }
            value => {
                inputs.push(value.to_string());
                index += 1;
            }
        }
    }
    if inputs.is_empty() {
        return Err("timeline-many requires at least one input batch".into());
    }
    let out_dir = out_dir.ok_or("--out-dir is required")?;
    let mut batches: Vec<SourceBatch> = Vec::with_capacity(inputs.len());
    let mut total_sources = 0;
    for (position, input) in inputs.iter().enumerate() {
        let mut batch = read_source_batch_json(input)?;
        if batch.batch_index as usize != position {
            return Err(format!(
                "batch_index {} is out of sequence; expected {position}",
                batch.batch_index
            )
            .into());
        }
        if batch.sources.is_empty() {
            return Err(format!("input batch {position} has no sources").into());
        }
        for source in &mut batch.sources {
            if source.batch_index != batch.batch_index {
                return Err("source batch_index does not match its input batch".into());
            }
            // from-csv restarts synthetic row IDs for every file. Scope those IDs
            // to the day/batch to avoid merging distinct articles across CSV files.
            // Caller-supplied non-synthetic identifiers retain their identity.
            if source.source_id.strip_prefix("row_").is_some_and(|suffix| {
                !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit())
            }) {
                source.source_id = format!("batch_{}:{}", batch.batch_index, source.source_id);
            }
        }
        total_sources += batch.sources.len();
        batches.push(batch);
    }
    let chunk_size_effective = batches[0].sources.len();
    let total_batches = batches.len();
    let config = if let Some(path) = config_path {
        read_config_json(path)?
    } else {
        SecaConfig::default()
    };
    let mut engine = SecaEngine::new(config)?;
    let output_dir = Path::new(&out_dir);
    if clean_out_dir && output_dir.exists() {
        fs::remove_dir_all(output_dir)?;
    }
    fs::create_dir_all(output_dir)?;
    write_timeline(
        &mut engine,
        batches,
        output_dir,
        total_sources,
        chunk_size_effective,
    )?;
    println!("Timeline build completed: {total_batches} daily batches, {total_sources} sources");
    Ok(())
}
/// Execute already prepared batches; input validation stays with each command.
fn write_timeline(
    engine: &mut SecaEngine,
    batches: Vec<SourceBatch>,
    output_dir: &Path,
    sources_total: usize,
    chunk_size_effective: usize,
) -> Result<TimelineManifest, Box<dyn std::error::Error>> {
    let total_batches = batches.len();
    let mut files = Vec::with_capacity(total_batches);
    for (index, batch) in batches.into_iter().enumerate() {
        if index == 0 {
            engine.build_baseline_tree(batch)?;
        } else {
            engine.process_batch(batch)?;
        }
        let name = format!("tree_batch_{index:04}.json");
        let tree = engine.export_baseline_tree_verbose()?;
        fs::write(output_dir.join(&name), serde_json::to_string_pretty(&tree)?)?;
        files.push(name);
    }
    let manifest = TimelineManifest {
        total_batches,
        sources_total,
        chunk_count: total_batches,
        chunk_size_effective,
        files,
    };
    fs::write(
        output_dir.join("timeline_manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use realtime_seca_core::SourceRecord;

    fn temporary_root() -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("risklive-seca-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_batch(root: &Path, index: u32, count: usize) -> String {
        let sources: Vec<SourceRecord> = (0..count)
            .map(|row| SourceRecord {
                source_id: format!("row_{:06}", row + 1),
                batch_index: index,
                tokens: vec!["fixture".to_string(), "risk".to_string()],
                text: Some("fixture risk".to_string()),
                timestamp_unix_ms: None,
                metadata: None,
            })
            .collect();
        let batch = SourceBatch {
            batch_index: index,
            sources,
        };
        let path = root.join(format!("input-{index}.json"));
        fs::write(&path, serde_json::to_string(&batch).unwrap()).unwrap();
        path.to_string_lossy().to_string()
    }

    #[test]
    fn preserves_unequal_daily_batches_and_scopes_csv_row_ids() {
        let root = temporary_root();
        let first = write_batch(&root, 0, 1);
        let second = write_batch(&root, 1, 2);
        let out = root.join("output");
        let args = vec![
            "cli".into(),
            "timeline-many".into(),
            first,
            second,
            "--out-dir".into(),
            out.to_string_lossy().into_owned(),
        ];
        run_timeline_many_command(&args).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out.join("timeline_manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["total_batches"], 2);
        assert_eq!(manifest["sources_total"], 3);
        let first_tree: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out.join("tree_batch_0000.json")).unwrap())
                .unwrap();
        let second_tree: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out.join("tree_batch_0001.json")).unwrap())
                .unwrap();
        assert_eq!(first_tree["source_legend"].as_array().unwrap().len(), 1);
        assert_eq!(second_tree["source_legend"].as_array().unwrap().len(), 3);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_out_of_order_input_before_creating_output() {
        let root = temporary_root();
        let bad = write_batch(&root, 1, 1);
        let out = root.join("output");
        let args = vec![
            "cli".into(),
            "timeline-many".into(),
            bad,
            "--out-dir".into(),
            out.to_string_lossy().into_owned(),
        ];
        assert!(run_timeline_many_command(&args)
            .unwrap_err()
            .to_string()
            .contains("out of sequence"));
        assert!(!out.exists());
        fs::remove_dir_all(root).unwrap();
    }
}
