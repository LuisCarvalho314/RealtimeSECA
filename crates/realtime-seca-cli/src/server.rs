use axum::{extract::Extension, response::IntoResponse, routing::get, Json, Router};
use realtime_seca_core::BaselineTreeVerboseExport;
use std::{fs, path::Path, sync::Arc};
use tokio::runtime::Runtime;
struct TreeResponse(Arc<BaselineTreeVerboseExport>);

impl serde::Serialize for TreeResponse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

pub(crate) fn run_serve_command(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if arguments.len() < 3 {
        eprintln!("Usage:");
        eprintln!(
            "  {} serve --tree-json <path> [--port <port>]",
            arguments[0]
        );
        return Ok(());
    }

    let mut tree_json_path: Option<String> = None;
    let mut port: u16 = 8000;
    let mut index = 2;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--tree-json" => {
                if let Some(value) = arguments.get(index + 1) {
                    tree_json_path = Some(value.clone());
                    index += 2;
                } else {
                    return Err("missing value for --tree-json".into());
                }
            }
            "--port" => {
                if let Some(value) = arguments.get(index + 1) {
                    port = value.parse::<u16>()?;
                    index += 2;
                } else {
                    return Err("missing value for --port".into());
                }
            }
            unknown_flag => {
                return Err(format!("unknown argument: {unknown_flag}").into());
            }
        }
    }

    let tree_json_path =
        tree_json_path.ok_or_else(|| "--tree-json is required for serve".to_string())?;
    let tree = load_tree_json(Path::new(&tree_json_path))?;
    let tree = Arc::new(tree);

    let runtime = Runtime::new()?;
    runtime.block_on(async move { serve_tree(tree, port).await })?;

    Ok(())
}

pub(crate) fn load_tree_json(
    path: &Path,
) -> Result<BaselineTreeVerboseExport, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let tree = serde_json::from_str::<BaselineTreeVerboseExport>(&contents)?;
    Ok(tree)
}

pub(crate) async fn serve_tree(
    tree: Arc<BaselineTreeVerboseExport>,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let app = Router::new()
        .route("/seca-tree", get(handle_seca_tree))
        .layer(Extension(tree));

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    let addr = listener.local_addr()?;
    println!("SECA tree server listening at http://{addr}/seca-tree");
    axum::serve(listener, app).await?;
    Ok(())
}

pub(crate) async fn handle_seca_tree(
    Extension(tree): Extension<Arc<BaselineTreeVerboseExport>>,
) -> impl IntoResponse {
    Json(TreeResponse(tree.clone()))
}
