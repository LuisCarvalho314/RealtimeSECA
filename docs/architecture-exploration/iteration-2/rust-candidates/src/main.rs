use seca_research_candidates::{build, structural, Input};
use std::io::Read;
fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    if serde_json::from_str::<serde_json::Value>(&text)
        .unwrap()
        .get("construction")
        .is_some()
    {
        let request = serde_json::from_str(&text).unwrap();
        let start = std::time::Instant::now();
        let mut result = seca_research_candidates::evolution::run(&request).unwrap();
        result["seconds"] = serde_json::json!(start.elapsed().as_secs_f64());
        println!("{result}");
        return;
    }
    let input: Input = serde_json::from_str(&text).unwrap();
    let start = std::time::Instant::now();
    let full = build(&input, std::env::args().any(|a| a == "--masks")).unwrap();
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::json!({"tree":structural(&full),"full":full,
        "seconds":seconds,"total_seconds":seconds})
    );
}
