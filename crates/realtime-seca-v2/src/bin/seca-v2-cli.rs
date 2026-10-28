use realtime_seca_v2::{Config, Decision, Hkt, Model, Source};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const ENGINE: &str = "realtime-seca-v2";
const STATE_VERSION: u32 = 1;
const REPORT_VERSION: u32 = 1;
const USAGE: &str = "usage: seca-v2-cli update --batch BATCH.json --config CONFIG.json --state STATE.json\n       seca-v2-cli show-last --state STATE.json";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Batch {
    batch: u64,
    sources: Vec<Source>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LastResult {
    batch: u64,
    decisions: Vec<Decision>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    engine: String,
    version: u32,
    model: Model,
    last_result: LastResult,
}

#[derive(Serialize)]
struct Report<'a> {
    schema_version: u32,
    batch: u64,
    next_batch: u64,
    retained_source_count: usize,
    root: &'a Hkt,
    decisions: &'a [Decision],
}

impl State {
    fn report(&self) -> Report<'_> {
        Report {
            schema_version: REPORT_VERSION,
            batch: self.last_result.batch,
            next_batch: self.model.next_batch(),
            retained_source_count: self.model.retained_sources().len(),
            root: self.model.root(),
            decisions: &self.last_result.decisions,
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.engine != ENGINE {
            return Err(format!("unsupported state engine: {}", self.engine));
        }
        if self.version != STATE_VERSION {
            return Err(format!("unsupported state version: {}", self.version));
        }
        if self.last_result.batch.checked_add(1) != Some(self.model.next_batch()) {
            return Err("last result does not match model batch".into());
        }
        Ok(())
    }
}

fn load_state(path: &Path) -> Result<State, String> {
    let bytes = fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let state: State = serde_json::from_slice(&bytes)
        .map_err(|e| format!("invalid state {}: {e}", path.display()))?;
    state.validate()?;
    Ok(state)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("invalid JSON {}: {e}", path.display()))
}

fn replace_state(path: &Path, temporary: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut created = false;
    let mut write = || -> io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(temporary)?;
        created = true;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(temporary, path)?;
        let directory = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        File::open(directory)?.sync_all()?;
        Ok(())
    };
    let result = write();
    if result.is_err() && created {
        let _ = fs::remove_file(temporary);
    }
    result.map_err(|e| format!("writing state {}: {e}", path.display()))
}

fn commit_state(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let name = path
        .file_name()
        .ok_or_else(|| "state path must name a file".to_string())?;
    let temporary = path.with_file_name(format!(
        ".{}.{}.{}.tmp",
        name.to_string_lossy(),
        std::process::id(),
        nanos
    ));
    replace_state(path, &temporary, bytes)
}

fn update(batch_path: &Path, config_path: &Path, state_path: &Path) -> Result<Vec<u8>, String> {
    let batch: Batch = read_json(batch_path)?;
    let config: Config = read_json(config_path)?;
    let new_model = Model::new(config.clone())?;
    let mut model = match fs::read(state_path) {
        Ok(bytes) => {
            let state: State = serde_json::from_slice(&bytes)
                .map_err(|e| format!("invalid state {}: {e}", state_path.display()))?;
            state.validate()?;
            if state.model.config() != &config {
                return Err("configuration does not match persisted model".into());
            }
            state.model
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => new_model,
        Err(e) => return Err(format!("reading {}: {e}", state_path.display())),
    };
    let decisions = model.update(batch.batch, batch.sources)?;
    let state = State {
        engine: ENGINE.into(),
        version: STATE_VERSION,
        model,
        last_result: LastResult {
            batch: batch.batch,
            decisions,
        },
    };
    let report = serde_json::to_vec(&state.report()).map_err(|e| e.to_string())?;
    let bytes = serde_json::to_vec(&state).map_err(|e| e.to_string())?;
    serde_json::from_slice::<State>(&bytes)
        .map_err(|e| format!("serialized state failed validation: {e}"))?
        .validate()?;
    commit_state(state_path, &bytes)?;
    Ok(report)
}

fn option(args: &[String], flag: &str) -> Result<PathBuf, String> {
    let mut values = args.chunks_exact(2);
    let mut found = None;
    for pair in &mut values {
        if pair[0] == flag {
            if found.replace(PathBuf::from(&pair[1])).is_some() {
                return Err(format!("duplicate option {flag}"));
            }
        } else if !["--batch", "--config", "--state"].contains(&pair[0].as_str()) {
            return Err(format!("unknown option {}", pair[0]));
        }
    }
    if !values.remainder().is_empty() {
        return Err("each option needs a path".into());
    }
    found.ok_or_else(|| format!("missing {flag}"))
}

fn run(args: &[String]) -> Result<Vec<u8>, String> {
    let Some((command, options)) = args.split_first() else {
        return Err(USAGE.into());
    };
    match command.as_str() {
        "update" if options.len() == 6 => update(
            &option(options, "--batch")?,
            &option(options, "--config")?,
            &option(options, "--state")?,
        ),
        "show-last" if options.len() == 2 => {
            let state = load_state(&option(options, "--state")?)?;
            serde_json::to_vec(&state.report()).map_err(|e| e.to_string())
        }
        _ => Err(USAGE.into()),
    }
}

fn main() {
    match run(&env::args().skip(1).collect::<Vec<_>>()) {
        Ok(report) => {
            let mut stdout = io::stdout().lock();
            if let Err(e) = stdout
                .write_all(&report)
                .and_then(|_| stdout.write_all(b"\n"))
            {
                eprintln!("writing report: {e}; recover it with show-last");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

    struct Fixture {
        dir: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = env::temp_dir().join(format!(
                "seca-v2-cli-{}-{}",
                std::process::id(),
                NEXT_DIR.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&dir).unwrap();
            Self { dir }
        }

        fn path(&self, name: &str) -> PathBuf {
            self.dir.join(name)
        }

        fn write(&self, name: &str, value: Value) -> PathBuf {
            let path = self.path(name);
            fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            path
        }

        fn config(&self, window: Option<u64>) -> PathBuf {
            self.write(
                "config.json",
                json!({"alpha": 0.7, "beta": 0.5,
                    "error_limits": {"alpha": 0.1, "beta": 0.2, "importance": 0.3},
                    "minimum_branch_sources": 2, "window": window}),
            )
        }

        fn batch(&self, index: u64, sources: Value) -> PathBuf {
            self.write("batch.json", json!({"batch": index, "sources": sources}))
        }

        fn update(&self, index: u64, sources: Value) -> Result<Value, String> {
            let batch = self.batch(index, sources);
            let bytes = super::update(&batch, &self.path("config.json"), &self.path("state.json"))?;
            Ok(serde_json::from_slice(&bytes).unwrap())
        }

        fn state(&self) -> State {
            load_state(&self.path("state.json")).unwrap()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.dir).unwrap();
        }
    }

    #[test]
    fn initial_and_consecutive_updates_recover_last_report() {
        let f = Fixture::new();
        f.config(None);
        let first = f
            .update(0, json!([{"id": "a", "words": ["b", "a", "a"]}]))
            .unwrap();
        assert_eq!(first["batch"], 0);
        assert_eq!(first["next_batch"], 1);
        assert_eq!(first["retained_source_count"], 1);
        assert!(first["root"]["nodes"].is_array());
        assert_eq!(first["decisions"], json!([]));
        let second = f
            .update(1, json!([{"id": "b", "words": ["b", "c"]}]))
            .unwrap();
        assert_eq!(second["next_batch"], 2);
        let recovered = run(&[
            "show-last".into(),
            "--state".into(),
            f.path("state.json").to_string_lossy().into(),
        ])
        .unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&recovered).unwrap(), second);
    }

    #[test]
    fn restart_matches_uninterrupted_and_exports_are_deterministic() {
        let f = Fixture::new();
        f.config(None);
        let sources = json!([{"id": "a", "words": ["z", "a", "z"]}]);
        let first = f.update(0, sources.clone()).unwrap();
        let serialized = fs::read(f.path("state.json")).unwrap();
        let second = f
            .update(1, json!([{"id": "b", "words": ["a", "b"]}]))
            .unwrap();
        let mut uninterrupted = Model::new(f.state().model.config().clone()).unwrap();
        uninterrupted
            .update(
                0,
                vec![Source {
                    id: "a".into(),
                    words: ["a".into(), "z".into()].into(),
                }],
            )
            .unwrap();
        let decisions = uninterrupted
            .update(
                1,
                vec![Source {
                    id: "b".into(),
                    words: ["a".into(), "b".into()].into(),
                }],
            )
            .unwrap();
        assert_eq!(f.state().model, uninterrupted);
        assert_eq!(f.state().last_result.decisions, decisions);
        assert_eq!(
            second["root"],
            serde_json::to_value(uninterrupted.root()).unwrap()
        );
        let g = Fixture::new();
        g.config(None);
        assert_eq!(g.update(0, sources).unwrap(), first);
        assert_eq!(fs::read(g.path("state.json")).unwrap(), serialized);
    }

    #[test]
    fn empty_batches_advance_gamma_forgetting() {
        let f = Fixture::new();
        f.config(Some(2));
        f.update(0, json!([{"id": "a", "words": ["a"]}])).unwrap();
        assert_eq!(f.update(1, json!([])).unwrap()["retained_source_count"], 1);
        let report = f.update(2, json!([])).unwrap();
        assert_eq!(report["retained_source_count"], 0);
        assert_eq!(report["next_batch"], 3);
        assert_eq!(f.state().model.retained_sources().len(), 0);
    }

    #[test]
    fn duplicate_identity_keeps_first_source_while_retained() {
        let f = Fixture::new();
        f.config(None);
        f.update(0, json!([{"id": "a", "words": ["first"]}]))
            .unwrap();
        f.update(1, json!([{"id": "a", "words": ["second"]}]))
            .unwrap();
        let state = serde_json::to_value(&f.state().model).unwrap();
        assert_eq!(state["retained"]["a"]["words"], json!(["first"]));
    }

    #[test]
    fn malformed_inputs_and_sequence_preserve_state() {
        let f = Fixture::new();
        f.config(None);
        f.update(0, json!([])).unwrap();
        let original = fs::read(f.path("state.json")).unwrap();
        for sources in [
            json!([{"id": " ", "words": ["a"]}]),
            json!([{"id": "a", "words": [""]}]),
            json!([{"id": "a", "words": [" "]}]),
            json!([{"id": "a", "words": "a"}]),
        ] {
            assert!(f.update(1, sources).is_err());
            assert_eq!(fs::read(f.path("state.json")).unwrap(), original);
        }
        assert!(f.update(2, json!([])).is_err());
        assert_eq!(fs::read(f.path("state.json")).unwrap(), original);
    }

    #[test]
    fn mismatched_or_invalid_config_preserves_state() {
        let f = Fixture::new();
        f.config(None);
        f.update(0, json!([])).unwrap();
        let original = fs::read(f.path("state.json")).unwrap();
        f.config(Some(3));
        assert!(f
            .update(1, json!([]))
            .unwrap_err()
            .contains("configuration"));
        f.write(
            "config.json",
            json!({"alpha": 2.0, "beta": 0.5,
            "error_limits": {"alpha": 0.1, "beta": 0.2, "importance": 0.3},
            "minimum_branch_sources": 2, "window": null}),
        );
        assert!(f.update(1, json!([])).is_err());
        assert_eq!(fs::read(f.path("state.json")).unwrap(), original);
    }

    #[test]
    fn corrupt_incompatible_and_inconsistent_state_are_rejected() {
        let f = Fixture::new();
        f.config(None);
        f.update(0, json!([])).unwrap();
        let valid: Value =
            serde_json::from_slice(&fs::read(f.path("state.json")).unwrap()).unwrap();
        let mut unsupported = valid.clone();
        unsupported["version"] = json!(99);
        fs::write(
            f.path("state.json"),
            serde_json::to_vec(&unsupported).unwrap(),
        )
        .unwrap();
        assert!(f.update(1, json!([])).unwrap_err().contains("version"));
        let mut wrong_engine = valid.clone();
        wrong_engine["engine"] = json!("realtime-seca-v1");
        fs::write(
            f.path("state.json"),
            serde_json::to_vec(&wrong_engine).unwrap(),
        )
        .unwrap();
        assert!(f.update(1, json!([])).is_err());
        let mut invalid_model = valid.clone();
        invalid_model["model"]["config"]["alpha"] = json!(2.0);
        fs::write(
            f.path("state.json"),
            serde_json::to_vec(&invalid_model).unwrap(),
        )
        .unwrap();
        assert!(f.update(1, json!([])).is_err());
        let mut wrong_batch = valid;
        wrong_batch["last_result"]["batch"] = json!(8);
        fs::write(
            f.path("state.json"),
            serde_json::to_vec(&wrong_batch).unwrap(),
        )
        .unwrap();
        assert!(f.update(1, json!([])).is_err());
        fs::write(f.path("state.json"), b"not JSON").unwrap();
        assert!(f.update(1, json!([])).is_err());
    }

    #[test]
    fn failed_replacement_keeps_previous_state() {
        let f = Fixture::new();
        f.config(None);
        f.update(0, json!([])).unwrap();
        let path = f.path("state.json");
        let original = fs::read(&path).unwrap();
        let temporary = f.path("occupied");
        fs::create_dir(&temporary).unwrap();
        assert!(replace_state(&path, &temporary, b"new state").is_err());
        assert_eq!(fs::read(path).unwrap(), original);
    }
}
