//! Immutable observations from d7e6e65, independent of the current implementation.
#[path = "support/behaviour_oracle.rs"]
mod oracle;
use serde_json::Value;

const EXPECTED: &str = include_str!("fixtures/behaviour_oracle/expected.json");
fn expected() -> Value {
    serde_json::from_str(EXPECTED).unwrap()
}

// Report the first differing JSON path instead of dumping a whole stream.
fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Object(a), Value::Object(e)) => {
            assert_eq!(
                a.keys().collect::<Vec<_>>(),
                e.keys().collect::<Vec<_>>(),
                "keys at {path}"
            );
            for (key, value) in a {
                compare(value, &e[key], &format!("{path}/{key}"));
            }
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len(), "length at {path}");
            for (i, (a, e)) in a.iter().zip(e).enumerate() {
                compare(a, e, &format!("{path}/{i}"));
            }
        }
        _ => assert_eq!(actual, expected, "behaviour changed at {path}"),
    }
}

#[test]
fn strict_public_contract_matches_frozen_baseline() {
    let actual = oracle::capture();
    let expected = expected();
    assert_eq!(
        actual.as_object().unwrap().keys().collect::<Vec<_>>(),
        expected.as_object().unwrap().keys().collect::<Vec<_>>()
    );
    for (mode, run) in actual.as_object().unwrap() {
        let steps = run["steps"].as_array().unwrap();
        assert_eq!(
            steps.len(),
            expected[mode]["steps"].as_array().unwrap().len()
        );
        for (i, step) in steps.iter().enumerate() {
            for field in ["strict", "rejection"] {
                compare(
                    &step[field],
                    &expected[mode]["steps"][i][field],
                    &format!("{mode}/{i}/{field}"),
                );
            }
        }
    }
}

#[test]
fn scientific_structure_matches_frozen_baseline() {
    let actual = oracle::capture();
    let expected = expected();
    for (mode, run) in actual.as_object().unwrap() {
        for (i, step) in run["steps"].as_array().unwrap().iter().enumerate() {
            compare(
                &step["structural"],
                &expected[mode]["steps"][i]["structural"],
                &format!("{mode}/{i}/structural"),
            );
        }
    }
}

#[test]
fn archived_schema4_snapshots_continue_like_frozen_uninterrupted_run() {
    let stream: oracle::Stream = serde_json::from_str(oracle::INPUT).unwrap();
    let expected = expected();
    for (mode, run) in expected.as_object().unwrap() {
        for (index, saved) in run["schema4_checkpoints"].as_object().unwrap() {
            let index: usize = index.parse().unwrap();
            assert_eq!(saved["schema_version"], 4);
            let mut engine = oracle::restore(saved);
            compare(
                &oracle::snapshot(&engine),
                saved,
                &format!("{mode}/restore/{index}"),
            );
            let mut previous_paths = oracle::paths(&engine);
            for (i, batch) in stream.batches.iter().enumerate().skip(index) {
                let report = engine.process_batch(batch.clone()).unwrap();
                let (strict, structural, paths) =
                    oracle::observe(&engine, &report, &previous_paths);
                compare(
                    &strict,
                    &run["steps"][i]["strict"],
                    &format!("{mode}/resume/{index}/{i}/strict"),
                );
                compare(
                    &structural,
                    &run["steps"][i]["structural"],
                    &format!("{mode}/resume/{index}/{i}/structural"),
                );
                previous_paths = paths;
                // Repeated real JSON restoration through forgetting and reconstruction.
                engine = oracle::restore(&oracle::snapshot(&engine));
            }
        }
    }
}
