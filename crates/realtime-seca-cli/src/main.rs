use axum::{extract::Extension, response::IntoResponse, routing::get, Json, Router};
use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};
use realtime_seca_core::{
    types::BaselineTreeVerboseExport, EngineSnapshot, SecaConfig, SecaEngine, SourceBatch,
    SourceRecord,
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use stop_words::{get as get_stop_words, LANGUAGE};
use tokio::runtime::Runtime;

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = env::args().collect();

    if arguments.len() < 2 {
        print_usage(&arguments);
        return Ok(());
    }

    match arguments[1].as_str() {
        "baseline" => run_baseline_command(&arguments)?,
        "timeline" => run_timeline_command(&arguments)?,
        "incremental" => run_incremental_command(&arguments)?,
        "from-csv" => run_from_csv_command(&arguments)?,
        "serve" => run_serve_command(&arguments)?,
        _ => print_usage(&arguments),
    }

    Ok(())
}

fn run_baseline_command(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if arguments.len() < 3 {
        print_usage(arguments);
        return Ok(());
    }

    let input_path = arguments[2].clone();

    let mut config_path: Option<String> = None;
    let mut snapshot_output_path: Option<String> = None;
    let mut tree_output_path: Option<String> = None;
    let mut verbose_tree_output_path: Option<String> = None;
    let mut print_tree = false;

    let mut index = 3;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--config" => {
                let Some(value) = arguments.get(index + 1) else {
                    return Err("missing value for --config".into());
                };
                config_path = Some(value.clone());
                index += 2;
            }
            "--snapshot-out" => {
                let Some(value) = arguments.get(index + 1) else {
                    return Err("missing value for --snapshot-out".into());
                };
                snapshot_output_path = Some(value.clone());
                index += 2;
            }
            "--dump-tree" => {
                let Some(value) = arguments.get(index + 1) else {
                    return Err("missing value for --dump-tree".into());
                };
                tree_output_path = Some(value.clone());
                index += 2;
            }
            "--dump-tree-verbose" => {
                let Some(value) = arguments.get(index + 1) else {
                    return Err("missing value for --dump-tree-verbose".into());
                };
                verbose_tree_output_path = Some(value.clone());
                index += 2;
            }
            "--print-tree" => {
                print_tree = true;
                index += 1;
            }
            unknown_flag => {
                return Err(format!("unknown argument: {unknown_flag}").into());
            }
        }
    }

    let batch = read_source_batch_json(&input_path)?;
    let config = if let Some(path) = config_path {
        read_config_json(path)?
    } else {
        SecaConfig::default()
    };

    let mut engine = SecaEngine::new(config)?;
    let result = engine.build_baseline_tree(batch)?;
    let snapshot = engine.snapshot()?;

    println!("Baseline build completed");
    println!("Batch index: {}", result.batch_index);
    println!("Sources processed: {}", result.sources_processed);

    if let Some(explanation) = engine.explain_last_update() {
        println!("Explanation: {}", explanation.summary);
    }

    let snapshot_json = serde_json::to_string_pretty(&snapshot)?;
    if let Some(path) = snapshot_output_path {
        fs::write(&path, snapshot_json)?;
        println!("Snapshot written to {}", path);
    }

    if let Some(path) = tree_output_path {
        let tree_export = engine.export_baseline_tree()?;
        let tree_json = serde_json::to_string_pretty(&tree_export)?;
        fs::write(&path, tree_json)?;
        println!("Baseline tree written to {}", path);
    }

    if let Some(path) = verbose_tree_output_path {
        let verbose_tree_export = engine.export_baseline_tree_verbose()?;
        let verbose_tree_json = serde_json::to_string_pretty(&verbose_tree_export)?;
        fs::write(&path, verbose_tree_json)?;
        println!("Verbose baseline tree written to {}", path);
    }

    if print_tree {
        let verbose_tree = engine.export_baseline_tree_verbose()?;
        print_verbose_tree(&verbose_tree);
    }

    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct TimelineManifest {
    batching_mode: String,
    total_batches: usize,
    days: Vec<String>,
    sources_total: usize,
    sources_batched: usize,
    sources_skipped_invalid_timestamp: usize,
    files: Vec<String>,
}

fn run_incremental_command(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if arguments.len() < 3 {
        print_usage(arguments);
        return Ok(());
    }

    let input_path = arguments[2].clone();
    let mut out_dir: Option<String> = None;
    let mut config_path: Option<String> = None;
    let mut snapshot_in_path: Option<String> = None;
    let mut snapshot_out_path: Option<String> = None;
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
            "--snapshot-in" => {
                let Some(value) = arguments.get(index + 1) else {
                    return Err("missing value for --snapshot-in".into());
                };
                snapshot_in_path = Some(value.clone());
                index += 2;
            }
            "--snapshot-out" => {
                let Some(value) = arguments.get(index + 1) else {
                    return Err("missing value for --snapshot-out".into());
                };
                snapshot_out_path = Some(value.clone());
                index += 2;
            }
            "--clean-out-dir" => {
                clean_out_dir = true;
                index += 1;
            }
            unknown_flag => return Err(format!("unknown argument: {unknown_flag}").into()),
        }
    }

    let out_dir = out_dir.ok_or_else(|| "--out-dir is required".to_string())?;
    let snapshot_out_path =
        snapshot_out_path.ok_or_else(|| "--snapshot-out is required".to_string())?;
    let batch = read_source_batch_json(&input_path)?;
    if batch.sources.is_empty() {
        return Err("input batch has no sources".into());
    }

    let config_from_arg = if let Some(path) = config_path {
        Some(read_config_json(path)?)
    } else {
        None
    };

    let snapshot_in = snapshot_in_path
        .as_ref()
        .map(PathBuf::from)
        .filter(|path| path.exists());

    let snapshot_from_disk = if let Some(path) = snapshot_in {
        Some(read_snapshot_json(path)?)
    } else {
        None
    };

    if let (Some(config), Some(snapshot)) = (config_from_arg.as_ref(), snapshot_from_disk.as_ref()) {
        if &snapshot.config != config {
            return Err(
                "snapshot config does not match --config; rebuild variant state or use matching config"
                    .into(),
            );
        }
    }

    let mut engine = if let Some(snapshot) = snapshot_from_disk {
        SecaEngine::load_snapshot(snapshot)?
    } else {
        let config = config_from_arg.unwrap_or_else(SecaConfig::default);
        SecaEngine::new(config)?
    };

    let result = if engine.snapshot()?.last_processed_batch_index.is_some() {
        engine.process_batch(batch.clone())?
    } else {
        engine.build_baseline_tree(batch.clone())?
    };

    let output_dir = Path::new(&out_dir);
    if clean_out_dir && output_dir.exists() {
        fs::remove_dir_all(output_dir)?;
    }
    fs::create_dir_all(output_dir)?;

    let file_name = format!("tree_batch_{:04}.json", result.batch_index);
    let file_path = output_dir.join(&file_name);
    let verbose_tree_export = engine.export_baseline_tree_verbose()?;
    let verbose_tree_json = serde_json::to_string_pretty(&verbose_tree_export)?;
    fs::write(&file_path, verbose_tree_json)?;

    let (_, day_labels, skipped_invalid_timestamp) = split_batch_into_daily_chunks_utc(&batch)?;
    let manifest_path = output_dir.join("timeline_manifest.json");
    let mut manifest = if manifest_path.exists() {
        read_timeline_manifest_json(&manifest_path)?
    } else {
        TimelineManifest {
            batching_mode: "daily_utc_incremental".to_string(),
            total_batches: 0,
            days: Vec::new(),
            sources_total: 0,
            sources_batched: 0,
            sources_skipped_invalid_timestamp: 0,
            files: Vec::new(),
        }
    };

    manifest.batching_mode = "daily_utc_incremental".to_string();
    manifest.total_batches = manifest.total_batches.saturating_add(1);
    for day in day_labels {
        if !manifest.days.contains(&day) {
            manifest.days.push(day);
        }
    }
    manifest.sources_total = manifest.sources_total.saturating_add(batch.sources.len());
    manifest.sources_batched = manifest.sources_batched.saturating_add(result.sources_processed);
    manifest.sources_skipped_invalid_timestamp = manifest
        .sources_skipped_invalid_timestamp
        .saturating_add(skipped_invalid_timestamp);
    manifest.files.push(file_name);
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;

    let snapshot = engine.snapshot()?;
    let snapshot_json = serde_json::to_string_pretty(&snapshot)?;
    fs::write(&snapshot_out_path, snapshot_json)?;

    println!("Incremental build completed");
    println!("Batch index processed: {}", result.batch_index);
    println!("Sources processed: {}", result.sources_processed);
    println!("Output directory: {}", output_dir.display());
    println!("Manifest written to {}", manifest_path.display());
    println!("Snapshot written to {}", snapshot_out_path);

    Ok(())
}

fn run_timeline_command(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if arguments.len() < 3 {
        print_usage(arguments);
        return Ok(());
    }

    let input_path = arguments[2].clone();
    let mut out_dir: Option<String> = None;
    let mut config_path: Option<String> = None;
    let mut chunk_count_flag_seen = false;
    let mut chunk_size_flag_seen = false;
    let mut use_risklive_news_data = false;
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
                let _ = value.parse::<usize>()?;
                chunk_count_flag_seen = true;
                index += 2;
            }
            "--chunk-size" => {
                let Some(value) = arguments.get(index + 1) else {
                    return Err("missing value for --chunk-size".into());
                };
                let _ = value.parse::<usize>()?;
                chunk_size_flag_seen = true;
                index += 2;
            }
            "--clean-out-dir" => {
                clean_out_dir = true;
                index += 1;
            }
            "--use-risklive-news-data" => {
                use_risklive_news_data = true;
                index += 1;
            }
            unknown_flag => return Err(format!("unknown argument: {unknown_flag}").into()),
        }
    }

    if chunk_count_flag_seen || chunk_size_flag_seen {
        return Err(
            "timeline now batches by UTC day; --chunk-count/--chunk-size are no longer supported"
                .into(),
        );
    }

    let out_dir = out_dir.ok_or_else(|| "--out-dir is required".to_string())?;
    let batch = if use_risklive_news_data {
        load_risklive_news_data_batch()?
    } else {
        read_source_batch_json(&input_path)?
    };
    if batch.sources.is_empty() {
        return Err("input batch has no sources".into());
    }

    let total_sources = batch.sources.len();
    let (chunks, days, skipped_invalid_timestamp) = split_batch_into_daily_chunks_utc(&batch)?;
    if chunks.is_empty() {
        return Err("no valid timestamped sources found for daily UTC batching".into());
    }
    let sources_batched = chunks.iter().map(|chunk| chunk.sources.len()).sum::<usize>();

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
    let mut files: Vec<String> = Vec::with_capacity(chunks.len());

    for (idx, chunk) in chunks.into_iter().enumerate() {
        if idx == 0 {
            engine.build_baseline_tree(chunk)?;
        } else {
            engine.process_batch(chunk)?;
        }

        let file_name = format!("tree_batch_{idx:04}.json");
        let file_path = output_dir.join(&file_name);
        let verbose_tree_export = engine.export_baseline_tree_verbose()?;
        let verbose_tree_json = serde_json::to_string_pretty(&verbose_tree_export)?;
        fs::write(&file_path, verbose_tree_json)?;
        files.push(file_name);
    }

    let manifest = TimelineManifest {
        batching_mode: "daily_utc".to_string(),
        total_batches: files.len(),
        days,
        sources_total: total_sources,
        sources_batched,
        sources_skipped_invalid_timestamp: skipped_invalid_timestamp,
        files,
    };
    let manifest_path = output_dir.join("timeline_manifest.json");
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;

    println!("Timeline build completed");
    println!("Output directory: {}", output_dir.display());
    println!("Manifest written to {}", manifest_path.display());
    println!("Batches exported: {}", manifest.total_batches);
    println!(
        "Sources batched: {} (skipped invalid timestamp: {})",
        manifest.sources_batched, manifest.sources_skipped_invalid_timestamp
    );
    if use_risklive_news_data {
        println!("Input mode: risklive news CSVs (results/data + results/backup_data)");
    }

    Ok(())
}

fn load_risklive_news_data_batch() -> Result<SourceBatch, Box<dyn std::error::Error>> {
    let repo_root = find_risklive_repo_root()
        .ok_or("unable to locate RiskLive repo root containing results/data or results/backup_data")?;

    let csv_sources: [(&str, PathBuf); 2] = [
        ("data", repo_root.join("results").join("data").join("news_data.csv")),
        (
            "backup",
            repo_root
                .join("results")
                .join("backup_data")
                .join("news_data.csv"),
        ),
    ];

    let mut merged_sources: Vec<SourceRecord> = Vec::new();
    for (prefix, csv_path) in csv_sources {
        if !csv_path.exists() {
            continue;
        }
        let batch = convert_csv_to_source_batch(csv_path, 0, 1)?;
        for mut source in batch.sources {
            source.source_id = format!("{prefix}::{}", source.source_id);
            merged_sources.push(source);
        }
    }

    if merged_sources.is_empty() {
        return Err("no sources loaded from results/data/news_data.csv or results/backup_data/news_data.csv".into());
    }

    Ok(SourceBatch {
        batch_index: 0,
        sources: merged_sources,
    })
}

fn find_risklive_repo_root() -> Option<PathBuf> {
    let start = env::current_dir().ok()?;
    for ancestor in start.ancestors() {
        let has_data = ancestor
            .join("results")
            .join("data")
            .join("news_data.csv")
            .exists();
        let has_backup = ancestor
            .join("results")
            .join("backup_data")
            .join("news_data.csv")
            .exists();
        if has_data || has_backup {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

fn split_batch_into_daily_chunks_utc(
    batch: &SourceBatch,
) -> Result<(Vec<SourceBatch>, Vec<String>, usize), Box<dyn std::error::Error>> {
    if batch.sources.is_empty() {
        return Ok((Vec::new(), Vec::new(), 0));
    }

    const MILLIS_PER_DAY: i64 = 86_400_000;
    let mut grouped_sources_by_day: BTreeMap<i64, Vec<(SourceRecord, i64)>> = BTreeMap::new();
    let mut skipped_invalid_timestamp = 0usize;

    for source in &batch.sources {
        let Some(timestamp_unix_ms) = extract_source_timestamp_unix_ms(source) else {
            skipped_invalid_timestamp += 1;
            continue;
        };
        let utc_day_index = timestamp_unix_ms.div_euclid(MILLIS_PER_DAY);
        grouped_sources_by_day
            .entry(utc_day_index)
            .or_default()
            .push((
                SourceRecord {
                    source_id: source.source_id.clone(),
                    batch_index: 0,
                    tokens: source.tokens.clone(),
                    text: source.text.clone(),
                    timestamp_unix_ms: Some(timestamp_unix_ms),
                    metadata: source.metadata.clone(),
                },
                timestamp_unix_ms,
            ));
    }

    let mut chunks: Vec<SourceBatch> = Vec::with_capacity(grouped_sources_by_day.len());
    let mut days: Vec<String> = Vec::with_capacity(grouped_sources_by_day.len());

    for (batch_index, (utc_day_index, mut day_sources)) in grouped_sources_by_day.into_iter().enumerate() {
        let day_start_ms = utc_day_index.saturating_mul(MILLIS_PER_DAY);
        let day_label = Utc
            .timestamp_millis_opt(day_start_ms)
            .single()
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| format!("utc-day-{utc_day_index}"));

        day_sources.sort_by(|(source_a, ts_a), (source_b, ts_b)| {
            ts_a.cmp(ts_b)
                .then_with(|| source_a.source_id.cmp(&source_b.source_id))
        });

        let rebatched_sources = day_sources
            .into_iter()
            .map(|(mut source, timestamp_unix_ms)| {
                source.batch_index = batch_index as u32;
                source.timestamp_unix_ms = Some(timestamp_unix_ms);
                source
            })
            .collect::<Vec<_>>();

        chunks.push(SourceBatch {
            batch_index: batch_index as u32,
            sources: rebatched_sources,
        });
        days.push(day_label);
    }

    Ok((chunks, days, skipped_invalid_timestamp))
}

fn run_from_csv_command(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if arguments.len() < 4 {
        eprintln!("Usage:");
        eprintln!(
            "  {} from-csv <input.csv> <output.json> [--batch-index <n>] [--min-tokens <n>]",
            arguments[0]
        );
        return Ok(());
    }

    let input_csv_path = arguments[2].clone();
    let output_json_path = arguments[3].clone();

    let mut batch_index: u32 = 0;
    let mut min_tokens: usize = 1;

    let mut index = 4;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--batch-index" => {
                let Some(value) = arguments.get(index + 1) else {
                    return Err("missing value for --batch-index".into());
                };
                batch_index = value.parse::<u32>()?;
                index += 2;
            }
            "--min-tokens" => {
                let Some(value) = arguments.get(index + 1) else {
                    return Err("missing value for --min-tokens".into());
                };
                min_tokens = value.parse::<usize>()?;
                index += 2;
            }
            unknown_flag => return Err(format!("unknown argument: {unknown_flag}").into()),
        }
    }

    let batch = convert_csv_to_source_batch(&input_csv_path, batch_index, min_tokens)?;
    let json = serde_json::to_string_pretty(&batch)?;
    fs::write(&output_json_path, json)?;
    println!(
        "Wrote {} sources to {}",
        batch.sources.len(),
        output_json_path
    );

    Ok(())
}

fn run_serve_command(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if arguments.len() < 3 {
        eprintln!("Usage:");
        eprintln!(
            "  {} serve --tree-json <path> [--port <port>]",
            arguments[0]
        );
        return Ok(());
    }

    let mut tree_json_path: Option<String> = None;
    let mut port: u16 = 8000;
    let mut index = 2;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--tree-json" => {
                if let Some(value) = arguments.get(index + 1) {
                    tree_json_path = Some(value.clone());
                    index += 2;
                } else {
                    return Err("missing value for --tree-json".into());
                }
            }
            "--port" => {
                if let Some(value) = arguments.get(index + 1) {
                    port = value.parse::<u16>()?;
                    index += 2;
                } else {
                    return Err("missing value for --port".into());
                }
            }
            unknown_flag => {
                return Err(format!("unknown argument: {unknown_flag}").into());
            }
        }
    }

    let tree_json_path =
        tree_json_path.ok_or_else(|| "--tree-json is required for serve".to_string())?;
    let tree = load_tree_json(Path::new(&tree_json_path))?;
    let tree = Arc::new(tree);

    let runtime = Runtime::new()?;
    runtime.block_on(async move { serve_tree(tree, port).await })?;

    Ok(())
}

fn load_tree_json(path: &Path) -> Result<BaselineTreeVerboseExport, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let tree = serde_json::from_str::<BaselineTreeVerboseExport>(&contents)?;
    Ok(tree)
}

async fn serve_tree(
    tree: Arc<BaselineTreeVerboseExport>,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let app = Router::new()
        .route("/seca-tree", get(handle_seca_tree))
        .layer(Extension(tree));

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    let addr = listener.local_addr()?;
    println!("SECA tree server listening at http://{addr}/seca-tree");
    axum::serve(listener, app).await?;
    Ok(())
}

struct TreeResponse(Arc<BaselineTreeVerboseExport>);

impl serde::Serialize for TreeResponse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

async fn handle_seca_tree(
    Extension(tree): Extension<Arc<BaselineTreeVerboseExport>>,
) -> impl IntoResponse {
    Json(TreeResponse(tree.clone()))
}

fn convert_csv_to_source_batch(
    input_csv_path: impl AsRef<Path>,
    batch_index: u32,
    min_tokens: usize,
) -> Result<SourceBatch, Box<dyn std::error::Error>> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(input_csv_path)?;

    let tokenizer = Regex::new(r"[A-Za-z0-9_]+")?;

    let stop_words: std::collections::BTreeSet<&'static str> =
        get_stop_words(LANGUAGE::English).iter().copied().collect();

    let headers = reader.headers()?.clone();

    let mut sources: Vec<SourceRecord> = Vec::new();

    for (row_index, result) in reader.records().enumerate() {
        let record = result?;
        let row_number = row_index + 1;

        let get = |column_name: &str| -> String {
            if let Some(position) = headers.iter().position(|header| header == column_name) {
                record.get(position).unwrap_or("").to_string()
            } else {
                String::new()
            }
        };

        let title = get("Title");
        let short_summary = get("ShortSummary");
        let description = get("Description");

        let text = normalize_whitespace(&format!("{title} {short_summary} {description}"));
        if text.is_empty() {
            continue;
        }

        let mut tokens: Vec<String> = tokenizer
            .find_iter(&text)
            .map(|m| m.as_str().to_lowercase())
            .filter(|token| !stop_words.contains(token.as_str()))
            .collect();

        // Optional: deduplicate consecutive duplicates
        // tokens.dedup();

        if tokens.len() < min_tokens {
            continue;
        }

        let timestamp_raw = get("Timestamp");
        let api_timestamp_raw = get("API_Timestamp");

        let mut metadata = serde_json::Map::new();
        for column_name in [
            "URL",
            "Query",
            "NewsCategory",
            "AlertFlag",
            "Relevance",
            "Timestamp",
            "API_Timestamp",
        ] {
            let value = get(column_name);
            if !value.trim().is_empty() {
                metadata.insert(column_name.to_string(), serde_json::Value::String(value));
            }
        }

        let source = SourceRecord {
            source_id: format!("row_{row_number:06}"),
            batch_index,
            tokens: std::mem::take(&mut tokens),
            text: Some(text),
            timestamp_unix_ms: parse_timestamp_to_unix_ms(&timestamp_raw)
                .or_else(|| parse_timestamp_to_unix_ms(&api_timestamp_raw)),
            metadata: if metadata.is_empty() {
                None
            } else {
                Some(serde_json::Value::Object(metadata))
            },
        };

        sources.push(source);
    }

    Ok(SourceBatch {
        batch_index,
        sources,
    })
}

fn normalize_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn parse_timestamp_to_unix_ms(value: &str) -> Option<i64> {
    parse_timestamp_string_to_unix_ms(value)
}

fn extract_source_timestamp_unix_ms(source: &SourceRecord) -> Option<i64> {
    if let Some(ts) = source.timestamp_unix_ms {
        return Some(ts);
    }

    let metadata = source.metadata.as_ref()?.as_object()?;
    for key in ["Timestamp", "API_Timestamp", "timestamp", "api_timestamp"] {
        if let Some(raw) = metadata.get(key).and_then(|value| value.as_str()) {
            if let Some(ts) = parse_timestamp_string_to_unix_ms(raw) {
                return Some(ts);
            }
        }
    }
    None
}

fn parse_timestamp_string_to_unix_ms(value: &str) -> Option<i64> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Ok(dt) = DateTime::parse_from_rfc3339(trimmed) {
        return Some(dt.timestamp_millis());
    }
    if let Ok(dt) = DateTime::parse_from_str(trimmed, "%Y-%m-%d %H:%M:%S%.f%:z") {
        return Some(dt.timestamp_millis());
    }
    if let Ok(dt) = DateTime::parse_from_str(trimmed, "%Y-%m-%d %H:%M:%S%:z") {
        return Some(dt.timestamp_millis());
    }
    if let Ok(naive_dt) = NaiveDateTime::parse_from_str(trimmed, "%Y-%m-%d %H:%M:%S%.f") {
        return Some(Utc.from_utc_datetime(&naive_dt).timestamp_millis());
    }
    if let Ok(naive_dt) = NaiveDateTime::parse_from_str(trimmed, "%Y-%m-%d %H:%M:%S") {
        return Some(Utc.from_utc_datetime(&naive_dt).timestamp_millis());
    }
    if let Ok(naive_date) = NaiveDate::parse_from_str(trimmed, "%Y-%m-%d") {
        if let Some(naive_dt) = naive_date.and_hms_opt(0, 0, 0) {
            return Some(Utc.from_utc_datetime(&naive_dt).timestamp_millis());
        }
    }
    None
}

fn read_source_batch_json(
    path: impl AsRef<Path>,
) -> Result<SourceBatch, Box<dyn std::error::Error>> {
    let file_contents = fs::read_to_string(path)?;
    let batch: SourceBatch = serde_json::from_str(&file_contents)?;
    Ok(batch)
}

fn read_config_json(path: impl AsRef<Path>) -> Result<SecaConfig, Box<dyn std::error::Error>> {
    let file_contents = fs::read_to_string(path)?;
    let config: SecaConfig = serde_json::from_str(&file_contents)?;
    Ok(config)
}

fn read_snapshot_json(path: impl AsRef<Path>) -> Result<EngineSnapshot, Box<dyn std::error::Error>> {
    let file_contents = fs::read_to_string(path)?;
    let snapshot: EngineSnapshot = serde_json::from_str(&file_contents)?;
    Ok(snapshot)
}

fn read_timeline_manifest_json(
    path: impl AsRef<Path>,
) -> Result<TimelineManifest, Box<dyn std::error::Error>> {
    let file_contents = fs::read_to_string(path)?;
    let manifest: TimelineManifest = serde_json::from_str(&file_contents)?;
    Ok(manifest)
}

fn print_usage(arguments: &[String]) {
    let executable_name = arguments
        .first()
        .map(String::as_str)
        .unwrap_or("realtime-seca-cli");
    eprintln!("Usage:");
    eprintln!("  {executable_name} baseline <input_batch.json> [--config <config.json>] [--snapshot-out <snapshot.json>] [--dump-tree <tree.json>] [--dump-tree-verbose <tree_verbose.json>] [--print-tree]");
    eprintln!("  {executable_name} timeline <input_batch.json> --out-dir <dir> [--config <config.json>] [--clean-out-dir] [--use-risklive-news-data]");
    eprintln!("  {executable_name} incremental <input_batch.json> --out-dir <dir> --snapshot-out <snapshot.json> [--snapshot-in <snapshot.json>] [--config <config.json>] [--clean-out-dir]");
    eprintln!("  {executable_name} from-csv <input.csv> <output.json> [--batch-index <n>] [--min-tokens <n>]");
}

// Reuse your existing pretty-printer helpers here:
fn print_verbose_tree(tree: &realtime_seca_core::BaselineTreeVerboseExport) {
    use std::collections::{BTreeMap, BTreeSet};

    let hkts_by_id: BTreeMap<i32, &realtime_seca_core::BaselineHktVerboseExport> =
        tree.hkts.iter().map(|hkt| (hkt.hkt_id, hkt)).collect();

    let nodes_by_id: BTreeMap<i32, &realtime_seca_core::BaselineNodeVerboseExport> =
        tree.nodes.iter().map(|node| (node.node_id, node)).collect();

    let mut child_hkt_ids_by_parent_node_id: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    for hkt in &tree.hkts {
        if hkt.parent_node_id != 0 {
            child_hkt_ids_by_parent_node_id
                .entry(hkt.parent_node_id)
                .or_default()
                .push(hkt.hkt_id);
        }
    }

    let root_hkt_ids: Vec<i32> = tree
        .hkts
        .iter()
        .filter(|h| h.parent_node_id == 0)
        .map(|h| h.hkt_id)
        .collect();
    let mut visited_hkts = BTreeSet::new();

    for root_hkt_id in root_hkt_ids {
        print_hkt_recursive(
            root_hkt_id,
            0,
            &hkts_by_id,
            &nodes_by_id,
            &child_hkt_ids_by_parent_node_id,
            &mut visited_hkts,
        );
    }
}

fn print_hkt_recursive(
    hkt_id: i32,
    indent_level: usize,
    hkts_by_id: &std::collections::BTreeMap<i32, &realtime_seca_core::BaselineHktVerboseExport>,
    nodes_by_id: &std::collections::BTreeMap<i32, &realtime_seca_core::BaselineNodeVerboseExport>,
    child_hkt_ids_by_parent_node_id: &std::collections::BTreeMap<i32, Vec<i32>>,
    visited_hkts: &mut std::collections::BTreeSet<i32>,
) {
    if !visited_hkts.insert(hkt_id) {
        println!("{}HKT {} (already visited)", indent(indent_level), hkt_id);
        return;
    }

    let Some(hkt) = hkts_by_id.get(&hkt_id) else {
        println!("{}HKT {} (missing)", indent(indent_level), hkt_id);
        return;
    };

    if hkt.parent_node_id == 0 {
        println!("{}HKT {} (root)", indent(indent_level), hkt.hkt_id);
    } else {
        println!(
            "{}HKT {} (parent_node_id={})",
            indent(indent_level),
            hkt.hkt_id,
            hkt.parent_node_id
        );
    }

    if !hkt.all_node_words_union.is_empty() {
        let union_words = hkt
            .all_node_words_union
            .iter()
            .map(|w| w.token.clone().unwrap_or_else(|| format!("#{}", w.word_id)))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "{}all_node_words_union=[{}]",
            indent(indent_level + 1),
            union_words
        );
    }

    if !hkt.expected_words.is_empty() {
        let expected_words = hkt
            .expected_words
            .iter()
            .map(|w| w.token.clone().unwrap_or_else(|| format!("#{}", w.word_id)))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "{}expected_words_excluding_seed=[{}]",
            indent(indent_level + 1),
            expected_words
        );
    }

    for node_id in &hkt.node_ids {
        let Some(node) = nodes_by_id.get(node_id) else {
            continue;
        };

        let words = node
            .words
            .iter()
            .map(|w| w.token.clone().unwrap_or_else(|| format!("#{}", w.word_id)))
            .collect::<Vec<_>>()
            .join(", ");

        let sources = node
            .sources
            .iter()
            .map(|s| s.external_source_id.clone())
            .collect::<Vec<_>>()
            .join(", ");

        if node.is_refuge_node {
            println!(
                "{}Node {} [REFUGE] words=[{}] sources=[{}]",
                indent(indent_level + 1),
                node.node_id,
                words,
                sources
            );
        } else {
            println!(
                "{}Node {} words=[{}] sources=[{}]",
                indent(indent_level + 1),
                node.node_id,
                words,
                sources
            );
        }

        if !node.top_words.is_empty() {
            let top_words = node
                .top_words
                .iter()
                .map(|w| w.token.clone().unwrap_or_else(|| format!("#{}", w.word_id)))
                .collect::<Vec<_>>()
                .join(", ");
            println!("{}top_words=[{}]", indent(indent_level + 2), top_words);
        }

        if let Some(child_hkts) = child_hkt_ids_by_parent_node_id.get(&node.node_id) {
            for child_hkt_id in child_hkts {
                print_hkt_recursive(
                    *child_hkt_id,
                    indent_level + 2,
                    hkts_by_id,
                    nodes_by_id,
                    child_hkt_ids_by_parent_node_id,
                    visited_hkts,
                );
            }
        }
    }
}

fn indent(level: usize) -> String {
    "  ".repeat(level)
}
