use realtime_seca_core::{SecaConfig, SecaEngine, SourceBatch};
use std::{fs, path::Path};

/// One process boundary for the existing incremental SECA engine. Outputs are
/// staged by the caller; the state file is the final atomic write.
pub(crate) fn run_update_command(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use realtime_seca_core::config::{
        AlphaErrorOption, BetaErrorOption, TriggerPolicyMode, WordImportanceErrorOption,
    };
    use std::io::Write;
    let input = arguments.get(2).ok_or("update requires input batch")?;
    let mut flags = std::collections::BTreeMap::new();
    let mut i = 3;
    while i < arguments.len() {
        let key = &arguments[i];
        if ![
            "--config",
            "--state-in",
            "--state-out",
            "--dump-tree-verbose",
        ]
        .contains(&key.as_str())
        {
            return Err(format!("unknown update argument: {key}").into());
        }
        flags.insert(
            key.as_str(),
            arguments.get(i + 1).ok_or("missing flag value")?.as_str(),
        );
        i += 2;
    }
    let config = serde_json::from_slice::<SecaConfig>(&fs::read(
        flags.get("--config").ok_or("--config required")?,
    )?)?;
    let t = &config.seca_thresholds;
    if config.memory_mode != realtime_seca_core::MemoryMode::SlidingWindow
        || config.trigger_policy_mode != TriggerPolicyMode::PaperDiagnosticScaffold
        || t.selected_alpha_option != AlphaErrorOption::Option1
        || t.selected_beta_option != BetaErrorOption::Option1
        || t.selected_word_importance_option != WordImportanceErrorOption::Option1
        || config.hkt_builder.minimum_threshold_against_max_word_count != t.alpha
        || config.hkt_builder.similarity_threshold != t.beta
    {
        return Err("update requires SECA-Light, paper equation options (Option1), and matching builder/update alpha and beta".into());
    }
    let batch: SourceBatch = serde_json::from_slice(&fs::read(input)?)?;
    let mut engine = if let Some(path) = flags.get("--state-in") {
        let snapshot = serde_json::from_slice(&fs::read(path)?)?;
        let engine = SecaEngine::load_snapshot(snapshot)?;
        if engine.config() != &config {
            return Err("model configuration changed; explicit replay/migration required".into());
        }
        engine
    } else {
        SecaEngine::new(config)?
    };
    engine.set_rebuild_mode(realtime_seca_core::engine::rebuild::RebuildMode::SubtreeTargeted);
    let result = if flags.contains_key("--state-in") {
        engine.process_batch(batch)?
    } else {
        engine.build_baseline_tree(batch)?
    };
    let tree_path = flags
        .get("--dump-tree-verbose")
        .ok_or("--dump-tree-verbose required")?;
    let state_path = Path::new(flags.get("--state-out").ok_or("--state-out required")?);
    fs::write(
        tree_path,
        serde_json::to_vec(&engine.export_baseline_tree_verbose()?)?,
    )?;
    let temporary = state_path.with_extension(format!("tmp-{}", std::process::id()));
    let write_result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&serde_json::to_vec(&engine.snapshot()?)?)?;
        file.sync_all()?;
        fs::rename(&temporary, state_path)?;
        fs::File::open(state_path.parent().unwrap_or(Path::new(".")))?.sync_all()?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    write_result?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
