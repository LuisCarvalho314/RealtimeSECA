//! Print observations only. This program never opens an expected-output file.
#[allow(dead_code)]
#[path = "../tests/support/behaviour_oracle.rs"]
mod oracle;
fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&oracle::capture()).unwrap()
    );
}
