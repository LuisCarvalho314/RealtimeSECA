mod update;
use update::run_update_command;

mod commands;
use commands::*;
mod timeline;
use timeline::*;
mod ingestion;
use ingestion::*;
mod presentation;
use presentation::*;
mod server;
use server::*;
use std::env;

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
        "update" => run_update_command(&arguments)?,
        "baseline" => run_baseline_command(&arguments)?,
        "timeline" => run_timeline_command(&arguments)?,
        "timeline-many" => run_timeline_many_command(&arguments)?,
        "from-csv" => run_from_csv_command(&arguments)?,
        "serve" => run_serve_command(&arguments)?,
        _ => print_usage(&arguments),
    }

    Ok(())
}

// Reuse your existing pretty-printer helpers here:

// RiskLive compatibility extension: preserve each pre-grouped UTC-day batch.
// The pinned core engine and existing timeline/from-csv implementations are unchanged.
