use crate::ingestion::{read_config_json, read_source_batch_json};
use crate::presentation::{print_usage, print_verbose_tree};
use realtime_seca_core::{SecaConfig, SecaEngine};
use std::fs;
pub(crate) fn run_baseline_command(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
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

    println!("Baseline build completed");
    println!("Batch index: {}", result.batch_index);
    println!("Sources processed: {}", result.sources_processed);

    if let Some(explanation) = engine.explain_last_update() {
        println!("Explanation: {}", explanation.summary);
    }

    if let Some(path) = snapshot_output_path {
        let snapshot_json = serde_json::to_string_pretty(&engine.snapshot()?)?;
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
