use realtime_seca_core::{MemoryMode, SecaConfig, SecaEngine, SourceBatch};
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("seca-csv-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
    fn convert(&self, name: &str, index: u32, rows: &str) -> SourceBatch {
        let csv = self.path(&format!("{name}.csv"));
        let json = self.path(&format!("{name}.json"));
        fs::write(&csv, format!("Title,URL,Timestamp,API_Timestamp\n{rows}")).unwrap();
        cli(&[
            "from-csv",
            csv.to_str().unwrap(),
            json.to_str().unwrap(),
            "--batch-index",
            &index.to_string(),
        ]);
        serde_json::from_slice(&fs::read(json).unwrap()).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn cli(args: &[&str]) -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_realtime-seca-cli"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap_or(Value::Null)
}
fn config() -> SecaConfig {
    SecaConfig {
        memory_mode: MemoryMode::SlidingWindow,
        max_batches_in_memory: Some(2),
        ..SecaConfig::default()
    }
}
fn update(f: &Fixture, name: &str, state_in: Option<&Path>) -> Value {
    let input = f.path(&format!("{name}.json"));
    let config = f.path("config.json");
    let state = f.path("state.json");
    let tree = f.path("tree.json");
    let mut args = vec![
        "update",
        input.to_str().unwrap(),
        "--config",
        config.to_str().unwrap(),
        "--state-out",
        state.to_str().unwrap(),
        "--dump-tree-verbose",
        tree.to_str().unwrap(),
    ];
    if let Some(path) = state_in {
        args.extend(["--state-in", path.to_str().unwrap()]);
    }
    cli(&args)
}

#[test]
fn sequential_conversions_survive_restart_and_preserve_retained_deduplication() {
    let f = Fixture::new();
    fs::write(
        f.path("config.json"),
        serde_json::to_vec(&config()).unwrap(),
    )
    .unwrap();
    let first = f.convert("day0", 0,
        "market report, https://example.org/a ,1970-01-01 00:00:00.123456+00:00,\nmarket duplicate,https://example.org/a,,\nmarket anonymous,,,\n");
    let result = update(&f, "day0", None);
    assert_eq!(result["active_source_count"], 2);
    let second = f.convert("day1", 1,
        "market report,https://example.org/b,,\nmarket repeat,https://example.org/a,,\nmarket anonymous,,,\n");
    assert_eq!(first.sources[0].source_id, "url:https://example.org/a");
    assert_eq!(first.sources[0].source_id, first.sources[1].source_id);
    assert_eq!(first.sources[0].source_id, second.sources[1].source_id);
    assert!(first.sources[2].source_id.starts_with("csv-row:"));
    assert_ne!(first.sources[2].source_id, second.sources[2].source_id);
    assert_eq!(first.sources[0].timestamp_unix_ms, Some(123));
    let roundtrip: SourceBatch =
        serde_json::from_slice(&serde_json::to_vec(&first).unwrap()).unwrap();
    assert_eq!(roundtrip.sources[2].source_id, first.sources[2].source_id);
    let state = f.path("state.json");
    let baseline_state = f.path("baseline-state.json");
    fs::copy(&state, &baseline_state).unwrap();
    let result = update(&f, "day1", Some(&state));
    assert_eq!(result["sources_processed"], 2);
    assert_eq!(result["active_source_count"], 4);
    let updated_state = fs::read(&state).unwrap();
    let updated_tree = fs::read(f.path("tree.json")).unwrap();
    // Replay the saved batch from the same baseline in another CLI process.
    assert_eq!(
        update(&f, "day1", Some(&baseline_state))["sources_processed"],
        2
    );
    assert_eq!(fs::read(&state).unwrap(), updated_state);
    assert_eq!(fs::read(f.path("tree.json")).unwrap(), updated_tree);
    let restored =
        SecaEngine::load_snapshot(serde_json::from_slice(&fs::read(&state).unwrap()).unwrap())
            .unwrap();
    assert_eq!(
        restored
            .export_baseline_tree_verbose()
            .unwrap()
            .source_legend
            .len(),
        4
    );
    let retained_ids: std::collections::BTreeSet<_> = restored
        .export_baseline_tree_verbose()
        .unwrap()
        .source_legend
        .into_iter()
        .map(|entry| entry.external_source_id)
        .collect();
    let expected_ids = first
        .sources
        .iter()
        .chain(&second.sources)
        .map(|source| source.source_id.clone())
        .collect();
    assert_eq!(retained_ids, expected_ids);
    // Advance the cutoff, then re-admit the now forgotten URL: no lifetime ledger.
    f.convert("day2", 2, "market new,https://example.org/c,,\n");
    assert_eq!(update(&f, "day2", Some(&state))["active_source_count"], 3);
    f.convert("day3", 3, "market return,https://example.org/a,,\n");
    assert_eq!(update(&f, "day3", Some(&state))["sources_processed"], 1);
}

#[test]
fn independent_default_index_conversions_have_distinct_fallbacks() {
    let f = Fixture::new();
    let a = f.convert("a", 0, "market anonymous,,,\nmarket other,,,\n");
    let b = f.convert("b", 0, "market anonymous,   ,,\n");
    assert_ne!(a.sources[0].source_id, a.sources[1].source_id);
    assert_ne!(a.sources[0].source_id, b.sources[0].source_id);
    let reconverted = f.convert("a", 0, "market anonymous,,,\nmarket other,,,\n");
    assert_ne!(a.sources[0].source_id, reconverted.sources[0].source_id);
    fs::write(f.path("no-url.csv"), "Title\nmarket anonymous\n").unwrap();
    cli(&[
        "from-csv",
        f.path("no-url.csv").to_str().unwrap(),
        f.path("no-url.json").to_str().unwrap(),
    ]);
    let missing: SourceBatch =
        serde_json::from_slice(&fs::read(f.path("no-url.json")).unwrap()).unwrap();
    assert!(missing.sources[0].source_id.starts_with("csv-row:"));
    assert_ne!(missing.sources[0].source_id, a.sources[0].source_id);
}

#[test]
fn timeline_many_preserves_new_namespaces() {
    let f = Fixture::new();
    f.convert(
        "a",
        0,
        "market article,https://example.org/a,,\nmarket anonymous,,,\n",
    );
    f.convert("b", 1, "market repeat,https://example.org/a,,\nmarket article,https://example.org/b,,\nmarket anonymous,,,\n");
    let a = f.path("a.json");
    let b = f.path("b.json");
    let out = f.path("timeline");
    cli(&[
        "timeline-many",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--out-dir",
        out.to_str().unwrap(),
    ]);
    let tree: Value =
        serde_json::from_slice(&fs::read(out.join("tree_batch_0001.json")).unwrap()).unwrap();
    assert_eq!(tree["source_legend"].as_array().unwrap().len(), 4);
}

#[test]
fn malformed_and_local_csv_timestamps_remain_unset() {
    let f = Fixture::new();
    let batch = f.convert("timestamps", 0,
        "market invalid,,nonsense,2026-02-16 15:24:56\nmarket missing,,,\nmarket fallback,,bad,1970-01-01T01:00:00+01:00\n");
    assert_eq!(batch.sources[0].timestamp_unix_ms, None);
    assert_eq!(batch.sources[1].timestamp_unix_ms, None);
    assert_eq!(batch.sources[2].timestamp_unix_ms, Some(0));
    let metadata = batch.sources[0].metadata.as_ref().unwrap();
    assert_eq!(metadata["Timestamp"], "nonsense");
    assert_eq!(metadata["API_Timestamp"], "2026-02-16 15:24:56");
    let valid = f.convert("valid", 0,
        "market utc,,1970-01-01T00:00:00.123Z,1970-01-02T00:00:00Z\nmarket offset,,1970-01-01T01:00:00+01:00,\nmarket invalid date,,2026-02-30T00:00:00Z,\nmarket unknown offset,,1970-01-01T00:00:00-00:00,\n");
    assert_eq!(valid.sources[0].timestamp_unix_ms, Some(123));
    assert_eq!(valid.sources[1].timestamp_unix_ms, Some(0));
    assert_eq!(valid.sources[2].timestamp_unix_ms, None);
    assert_eq!(valid.sources[3].timestamp_unix_ms, None);
    assert_eq!(
        valid.sources[0].metadata.as_ref().unwrap()["Timestamp"],
        "1970-01-01T00:00:00.123Z"
    );
}
