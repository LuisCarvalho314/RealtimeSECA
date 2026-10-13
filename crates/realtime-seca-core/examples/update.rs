// Exercise the exact production update command without the HTTP/tokenizer dependencies.
#[path = "../../realtime-seca-cli/src/update.rs"]
mod update;
fn main() {
    if let Err(error) = update::run_update_command(&std::env::args().collect::<Vec<_>>()) {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}
