use seca_research_candidates::evolution::{Engine, StreamInput};
use std::io::Read;
fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let request: StreamInput = serde_json::from_value(v).unwrap();
    let mut engine = Engine::new(&request).unwrap();
    let mut states = vec![engine.tree.clone()];
    let mut updates = Vec::new();
    let mut times = Vec::new();
    for batch in &request.batches {
        let before = serde_json::to_value(&engine).unwrap();
        let start = std::time::Instant::now();
        let update = engine.process(batch);
        times.push(start.elapsed().as_secs_f64());
        updates.push(match update {Ok(u)=>serde_json::json!({"ok":true,"decisions":u.decisions,"replacements":u.replacements}),
                Err(e)=>serde_json::json!({"ok":false,"error":e,"unchanged":before==serde_json::to_value(&engine).unwrap()})});
        states.push(engine.tree.clone());
        engine = serde_json::from_value(serde_json::to_value(&engine).unwrap()).unwrap();
    }
    println!(
        "{}",
        serde_json::json!({"states":states,"updates":updates,"update_seconds":times})
    );
}
