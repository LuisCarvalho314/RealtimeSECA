import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional
import json

import pandas as pd
import plotly.express as px
import streamlit as st
import subprocess

try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False


st.set_page_config(page_title="Newsmap", layout="wide")

st.title("Newsmap")
st.caption("SECA tree explorer with batch playback, treemap drill-down, and diagnostics.")


def _load_sample_data() -> pd.DataFrame:
    data = [
        {
            "category": "World",
            "topic": "Geopolitics",
            "source": "Wire",
            "title": "Ceasefire talks resume in region",
            "url": "https://example.com/world-1",
            "date": dt.date(2026, 2, 24),
            "value": 38,
        },
        {
            "category": "World",
            "topic": "Elections",
            "source": "Daily",
            "title": "National election debate highlights economy",
            "url": "https://example.com/world-2",
            "date": dt.date(2026, 2, 25),
            "value": 24,
        },
        {
            "category": "Business",
            "topic": "Markets",
            "source": "Ticker",
            "title": "Equities edge higher as rates stabilize",
            "url": "https://example.com/biz-1",
            "date": dt.date(2026, 2, 26),
            "value": 31,
        },
        {
            "category": "Business",
            "topic": "Earnings",
            "source": "Ticker",
            "title": "Tech firm posts record quarterly results",
            "url": "https://example.com/biz-2",
            "date": dt.date(2026, 2, 26),
            "value": 18,
        },
        {
            "category": "Technology",
            "topic": "AI",
            "source": "Lab",
            "title": "New model improves long-context reasoning",
            "url": "https://example.com/tech-1",
            "date": dt.date(2026, 2, 25),
            "value": 44,
        },
        {
            "category": "Technology",
            "topic": "Security",
            "source": "Lab",
            "title": "Researchers disclose large-scale phishing campaign",
            "url": "https://example.com/tech-2",
            "date": dt.date(2026, 2, 24),
            "value": 22,
        },
        {
            "category": "Science",
            "topic": "Health",
            "source": "Journal",
            "title": "Study links sleep quality to cognitive resilience",
            "url": "https://example.com/sci-1",
            "date": dt.date(2026, 2, 23),
            "value": 16,
        },
        {
            "category": "Sports",
            "topic": "Basketball",
            "source": "Desk",
            "title": "Playoff race tightens after overtime win",
            "url": "https://example.com/sports-1",
            "date": dt.date(2026, 2, 26),
            "value": 20,
        },
        {
            "category": "Culture",
            "topic": "Film",
            "source": "Review",
            "title": "Festival lineup blends indie and blockbuster",
            "url": "https://example.com/culture-1",
            "date": dt.date(2026, 2, 22),
            "value": 12,
        },
    ]
    df = pd.DataFrame(data)
    return df


def _extract_tokens(word_refs: List[Dict]) -> List[str]:
    tokens: list[str] = []
    if not word_refs:
        return tokens
    for entry in word_refs:
        if isinstance(entry, dict):
            token = entry.get("token")
            if token:
                tokens.append(token)
    return tokens


def _node_tokens(node: Dict) -> List[str]:
    tokens = _extract_tokens(node.get("top_words", []))
    if tokens:
        return tokens
    return _extract_tokens(node.get("words", []))


def _hkt_keyword(hkt: Optional[Dict]) -> str:
    if not hkt:
        return "(unknown HKT)"
    keywords = _extract_tokens(hkt.get("expected_words", []))
    if not keywords:
        keywords = _extract_tokens(hkt.get("all_node_words_union", []))
    if keywords:
        return keywords[0]
    return f"HKT {hkt.get('hkt_id', '?')}"


def _format_node_path(parts: list[str]) -> str:
    if not parts:
        return "(empty path)"
    return " ▶ ".join(parts)


def _build_node_path(node: Dict, nodes_by_id: Dict[int, Dict], hkts_by_id: Dict[int, Dict]) -> list[str]:
    parts: list[str] = []
    current_node = node
    visited_nodes: set[int] = set()

    while current_node:
        node_tokens = _node_tokens(current_node)
        if node_tokens:
            label_text = " ".join(node_tokens[:4])
        elif current_node.get("is_refuge_node"):
            label_text = "<refuge node>"
        else:
            label_text = "(unnamed node)"
        parts.append(label_text)

        hkt = hkts_by_id.get(current_node.get("hkt_id"))
        parent_node_id = hkt.get("parent_node_id") if hkt else 0
        if parent_node_id in (0, None) or parent_node_id in visited_nodes:
            break
        visited_nodes.add(parent_node_id)
        current_node = nodes_by_id.get(parent_node_id)

    return list(reversed(parts))


def _node_label(node: Dict) -> str:
    node_tokens = _node_tokens(node)
    if node_tokens:
        return " ".join(node_tokens[:4])
    if node.get("is_refuge_node"):
        return "<refuge node>"
    return "(unnamed node)"


def _root_hkt_keyword_for_node(node: Dict, nodes_by_id: Dict[int, Dict], hkts_by_id: Dict[int, Dict]) -> str:
    current_node = node
    visited_nodes: set[int] = set()
    last_hkt: Optional[Dict] = None

    while current_node:
        node_id = current_node.get("node_id")
        if node_id in visited_nodes:
            break
        visited_nodes.add(node_id)

        hkt = hkts_by_id.get(current_node.get("hkt_id"))
        if hkt:
            last_hkt = hkt
        parent_node_id = hkt.get("parent_node_id") if hkt else 0
        if parent_node_id in (0, None):
            break
        current_node = nodes_by_id.get(parent_node_id)

    return _hkt_keyword(last_hkt)


def build_tree_view_model(tree: dict) -> tuple[pd.DataFrame, dict]:
    nodes = tree.get("nodes", [])
    hkts = tree.get("hkts", [])
    hkts_by_id = {hkt.get("hkt_id"): hkt for hkt in hkts}
    nodes_by_id = {node.get("node_id"): node for node in nodes}
    max_sources = max((len(node.get("sources", [])) for node in nodes), default=1)

    node_rows: list[dict] = []
    skipped_nodes = 0

    for node in nodes:
        node_id = node.get("node_id")
        if node_id is None:
            skipped_nodes += 1
            continue

        hkt = hkts_by_id.get(node.get("hkt_id"))
        parent_node_id = hkt.get("parent_node_id") if hkt else 0
        if parent_node_id not in (0, None) and parent_node_id not in nodes_by_id:
            skipped_nodes += 1
            continue

        # Compute depth from explicit node parent chain and reject cycles.
        depth = 0
        current_parent_id = parent_node_id
        visited_nodes = {node_id}
        has_cycle = False
        while current_parent_id not in (0, None):
            if current_parent_id in visited_nodes:
                has_cycle = True
                break
            visited_nodes.add(current_parent_id)
            depth += 1
            parent_node = nodes_by_id.get(current_parent_id)
            if parent_node is None:
                has_cycle = True
                break
            parent_hkt = hkts_by_id.get(parent_node.get("hkt_id"))
            current_parent_id = parent_hkt.get("parent_node_id") if parent_hkt else 0

        if has_cycle:
            skipped_nodes += 1
            continue

        node_label = _node_label(node)
        hkt_keyword = _root_hkt_keyword_for_node(node, nodes_by_id, hkts_by_id)
        node_tokens = _node_tokens(node)
        token_display = ", ".join(node_tokens[:10])
        sources = node.get("sources", [])
        value = max(1, len(sources))
        strength_pct = value / max_sources if max_sources else 0.0

        node_rows.append(
            {
                "id": f"node:{node_id}",
                "parent": "" if parent_node_id in (0, None) else f"node:{parent_node_id}",
                "label": node_label,
                "node_id": node_id,
                "hkt_id": node.get("hkt_id"),
                "depth": depth,
                "source_count": len(sources),
                "category": hkt_keyword,
                "topic": node_label,
                "title": node_label,
                "source": node_label,
                "url": "",
                "date": dt.date.today(),
                "value": value,
                "strength_pct": strength_pct,
                "source_ids": ", ".join(
                    str(src.get("external_source_id")) for src in sources
                ),
                "tokens": token_display,
                "is_refuge_node": bool(node.get("is_refuge_node")),
            }
        )

    summary = {
        "total_hkts": len(hkts),
        "total_nodes": len(nodes),
        "rendered_nodes": len(node_rows),
        "skipped_nodes": skipped_nodes,
    }

    if not node_rows:
        empty_df = pd.DataFrame(
            columns=[
                "id",
                "parent",
                "label",
                "node_id",
                "hkt_id",
                "depth",
                "source_count",
                "category",
                "topic",
                "title",
                "source",
                "url",
                "date",
                "value",
                "strength_pct",
                "source_ids",
                "tokens",
            ]
        )
        return empty_df, summary

    return pd.DataFrame(node_rows), summary


def tree_to_newsmap_df(tree: dict) -> pd.DataFrame:
    df, _ = build_tree_view_model(tree)
    return df


def build_config_payload(
    min_threshold: float,
    similarity: float,
    min_sources: int,
    alpha: float,
    beta: float,
    alpha_err: float,
    beta_err: float,
    wi_err: float,
    memory_mode: str,
    max_batches: Optional[int],
    alpha_opt1: float,
    alpha_opt2: float,
    alpha_opt3: float,
    beta_opt1: float,
    beta_opt2: float,
    beta_opt3: float,
    wi_opt1: float,
    wi_opt2: float,
    selected_alpha: str,
    selected_beta: str,
    selected_wi: str,
    trigger_policy: str,
) -> dict:
    return {
        "hkt_builder": {
            "minimum_threshold_against_max_word_count": min_threshold,
            "similarity_threshold": similarity,
            "minimum_number_of_sources_to_create_branch_for_node": min_sources,
        },
        "seca_thresholds": {
            "alpha": alpha,
            "beta": beta,
            "alpha_error_threshold": alpha_err,
            "beta_error_threshold": beta_err,
            "word_importance_error_threshold": wi_err,
            "alpha_option1_threshold": alpha_opt1,
            "alpha_option2_threshold": alpha_opt2,
            "alpha_option3_threshold": alpha_opt3,
            "beta_option1_threshold": beta_opt1,
            "beta_option2_threshold": beta_opt2,
            "beta_option3_threshold": beta_opt3,
            "word_importance_option1_threshold": wi_opt1,
            "word_importance_option2_threshold": wi_opt2,
            "selected_alpha_option": selected_alpha,
            "selected_beta_option": selected_beta,
            "selected_word_importance_option": selected_wi,
        },
        "trigger_policy_mode": trigger_policy,
        "memory_mode": memory_mode,
        "max_batches_in_memory": max_batches,
    }


def run_baseline_command(
    batch_path: str, verbose_path: str, config_path: Optional[str]
) -> str:
    cmd = [
        "cargo",
        "run",
        "-p",
        "realtime-seca-cli",
        "--",
        "baseline",
        batch_path,
        "--dump-tree-verbose",
        verbose_path,
    ]
    if config_path:
        cmd += ["--config", config_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout + result.stderr


def run_timeline_command(
    batch_path: str,
    output_dir: str,
    chunk_count: int,
    config_path: Optional[str],
    clean_out_dir: bool,
) -> str:
    cmd = [
        "cargo",
        "run",
        "-p",
        "realtime-seca-cli",
        "--",
        "timeline",
        batch_path,
        "--out-dir",
        output_dir,
        "--chunk-count",
        str(chunk_count),
    ]
    if config_path:
        cmd += ["--config", config_path]
    if clean_out_dir:
        cmd += ["--clean-out-dir"]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout + result.stderr


def load_tree_from_file(path: Path) -> bool:
    if not path.exists():
        return False
    st.session_state.seca_tree = json.loads(path.read_text())
    st.session_state.seca_tree_error = None
    st.session_state.seca_tree_loaded = True
    tree_df, validation = build_tree_view_model(st.session_state.seca_tree)
    st.session_state.seca_tree_df = tree_df
    st.session_state.tree_validation = validation
    st.session_state.timeline_manifest = None
    st.session_state.timeline_trees = None
    st.session_state.timeline_selected_index = 0
    st.session_state.timeline_error = None
    st.session_state.timeline_output_dir = None
    st.session_state.current_timeline_batch_index = None
    return True


def load_timeline_from_output_dir(output_dir: Path) -> bool:
    manifest_path = output_dir / "timeline_manifest.json"
    if not manifest_path.exists():
        return False

    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", [])
    if not files:
        return False

    trees = []
    for file_name in files:
        tree_path = output_dir / file_name
        if not tree_path.exists():
            return False
        trees.append(json.loads(tree_path.read_text()))

    st.session_state.timeline_manifest = manifest
    st.session_state.timeline_trees = trees
    st.session_state.timeline_selected_index = len(trees) - 1
    st.session_state.current_timeline_batch_index = len(trees) - 1
    st.session_state.timeline_error = None
    st.session_state.timeline_output_dir = str(output_dir)
    st.session_state.seca_tree = trees[-1]
    tree_df, validation = build_tree_view_model(trees[-1])
    st.session_state.seca_tree_df = tree_df
    st.session_state.tree_validation = validation
    st.session_state.seca_tree_error = None
    st.session_state.seca_tree_loaded = True
    return True


def _parse_dates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def _load_from_upload(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    else:
        st.error("Unsupported file type. Use CSV or JSON.")
        return None
    return df


def _load_large_batch() -> Optional[pd.DataFrame]:
    base = Path(__file__).resolve().parents[2]
    path = base / "crates" / "realtime-seca-core" / "tests" / "data" / "large_batch.json"
    if not path.exists():
        return None
    try:
        raw = pd.read_json(path)
    except ValueError:
        # Fallback for JSON lines
        raw = pd.read_json(path, lines=True)

    # Handle SECA batch schema: { batch_index, sources: [...] }
    if isinstance(raw, pd.DataFrame) and "sources" in raw.columns and len(raw) == 1:
        sources = raw.loc[0, "sources"]
        df = pd.DataFrame(sources)
    elif isinstance(raw, pd.DataFrame) and "source_id" in raw.columns:
        df = raw
    else:
        return None

    def title_from_row(row):
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()[:140]
        tokens = row.get("tokens") or []
        if isinstance(tokens, list) and tokens:
            return " ".join(tokens[:24])
        return "(untitled)"

    def date_from_row(row):
        ts = row.get("timestamp_unix_ms")
        if isinstance(ts, (int, float)) and ts > 0:
            return dt.date.fromtimestamp(ts / 1000.0)
        return dt.date.today()

    def get_meta(row, key, default):
        meta = row.get("metadata")
        if isinstance(meta, dict) and key in meta:
            return meta[key]
        if isinstance(meta, dict):
            upper_key = key.upper()
            if upper_key in meta:
                return meta[upper_key]
        return default

    df = df.copy()
    df["category"] = df.apply(lambda r: get_meta(r, "category", "Batch"), axis=1)
    df["topic"] = df.apply(lambda r: get_meta(r, "topic", "Source"), axis=1)
    df["source"] = df.get("source_id", "(unknown)")
    df["title"] = df.apply(title_from_row, axis=1)
    df["url"] = df.apply(lambda r: get_meta(r, "url", ""), axis=1)
    df["date"] = df.apply(date_from_row, axis=1)
    df["value"] = df["tokens"].apply(lambda t: len(t) if isinstance(t, list) else 1)
    return df


def load_source_catalog_from_batch(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (ValueError, OSError):
        return {}

    sources = payload.get("sources", []) if isinstance(payload, dict) else []
    catalog: Dict[str, Dict[str, str]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("source_id", "")).strip()
        if not source_id:
            continue
        text = source.get("text")
        tokens = source.get("tokens") or []
        title = text.strip()[:120] if isinstance(text, str) and text.strip() else " ".join(tokens[:16]).strip()
        meta = source.get("metadata")
        url = ""
        if isinstance(meta, dict):
            url = str(meta.get("url", "")).strip()
            if not url:
                url = str(meta.get("URL", "")).strip()
        catalog[source_id] = {"title": title or source_id, "url": url}
    return catalog


def build_source_assignments(tree: dict, source_catalog: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    nodes = tree.get("nodes", [])
    hkts = tree.get("hkts", [])
    hkts_by_id = {hkt.get("hkt_id"): hkt for hkt in hkts}
    nodes_by_id = {node.get("node_id"): node for node in nodes}

    rows: Dict[str, Dict[str, str]] = {}
    for node in nodes:
        hkt_label = _root_hkt_keyword_for_node(node, nodes_by_id, hkts_by_id)
        node_id = int(node.get("node_id", -1))
        node_label = _node_label(node)
        for source in node.get("sources", []):
            if not isinstance(source, dict):
                continue
            source_id = str(source.get("external_source_id", "")).strip()
            if not source_id:
                continue
            source_meta = source_catalog.get(source_id, {})
            title = source_meta.get("title") or source_id
            url = source_meta.get("url", "")
            key = f"{source_id}::{node_id}"
            rows[key] = {
                "source_id": source_id,
                "hkt": hkt_label,
                "node_id": node_id,
                "node_label": node_label,
                "title": title,
                "url": url,
            }

    if not rows:
        return pd.DataFrame(columns=["source_id", "hkt", "node_id", "node_label", "title", "url"])
    return pd.DataFrame(rows.values())


def build_hkt_source_treemap(assignments: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame(columns=["id", "parent", "label", "value", "url", "hkt"])

    deduped = (
        assignments.sort_values(["hkt", "title"])
        .drop_duplicates(subset=["source_id", "hkt"], keep="first")
        .copy()
    )
    deduped["value"] = 1

    hkt_rows = (
        deduped.groupby("hkt", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "hkt_value"})
    )
    hkt_rows["id"] = hkt_rows["hkt"].map(lambda h: f"hkt:{h}")
    hkt_rows["parent"] = ""
    hkt_rows["label"] = hkt_rows["hkt"]
    hkt_rows["value"] = hkt_rows["hkt_value"]
    hkt_rows["url"] = ""

    leaf_rows = deduped.copy()
    leaf_rows["id"] = leaf_rows.apply(
        lambda r: f"src:{r['hkt']}::{r['source_id']}", axis=1
    )
    leaf_rows["parent"] = leaf_rows["hkt"].map(lambda h: f"hkt:{h}")
    leaf_rows["label"] = leaf_rows["title"].astype(str).str.slice(0, 120)

    return pd.concat(
        [
            hkt_rows[["id", "parent", "label", "value", "url", "hkt"]],
            leaf_rows[["id", "parent", "label", "value", "url", "hkt"]],
        ],
        ignore_index=True,
    )


def filter_structure_df(
    tree_df: pd.DataFrame,
    active_hkt: str,
    active_node_id: Optional[int],
) -> pd.DataFrame:
    if tree_df.empty:
        return tree_df

    scoped = tree_df.copy()
    if active_hkt and active_hkt != "All HKTs":
        scoped = scoped[scoped["category"] == active_hkt]
    if active_node_id is None:
        return scoped

    id_to_parent = {
        int(row["node_id"]): (None if row["parent"] == "" else int(str(row["parent"]).split(":")[-1]))
        for _, row in scoped.iterrows()
    }

    keep: set[int] = set()
    stack = [int(active_node_id)]
    while stack:
        current = stack.pop()
        if current in keep:
            continue
        keep.add(current)
        for node_id, parent_id in id_to_parent.items():
            if parent_id == current:
                stack.append(node_id)

    return scoped[scoped["node_id"].isin(keep)]


def filter_source_assignments(
    assignments: pd.DataFrame,
    active_hkt: str,
    active_node_id: Optional[int],
) -> pd.DataFrame:
    if assignments.empty:
        return assignments
    scoped = assignments.copy()
    if active_hkt and active_hkt != "All HKTs":
        scoped = scoped[scoped["hkt"] == active_hkt]
    if active_node_id is not None:
        scoped = scoped[scoped["node_id"] == int(active_node_id)]
    return scoped


def render_source_review_panel(
    scoped_sources: pd.DataFrame,
    key_prefix: str,
    require_open_node: bool = False,
) -> None:
    st.subheader("Source Review")
    if require_open_node:
        st.info("Open a node in Structure view to show relevant sources.")
        return
    if scoped_sources.empty:
        st.info("No sources in current scope.")
        return

    search = st.text_input("Search sources", value="", key=f"{key_prefix}_search")
    url_only = st.checkbox("URL only", value=True, key=f"{key_prefix}_url_only")
    sort_mode = st.selectbox(
        "Sort by",
        ["Title", "Source ID"],
        index=0,
        key=f"{key_prefix}_sort",
    )
    max_items = st.slider(
        "Max items",
        min_value=10,
        max_value=200,
        value=40,
        step=10,
        key=f"{key_prefix}_max_items",
    )

    view = scoped_sources.copy()
    if search.strip():
        pattern = search.strip()
        view = view[
            view["title"].fillna("").astype(str).str.contains(pattern, case=False, na=False)
            | view["source_id"].fillna("").astype(str).str.contains(pattern, case=False, na=False)
        ]
    if url_only:
        view = view[view["url"].fillna("").astype(str).str.len() > 0]

    view = view.drop_duplicates(subset=["source_id", "hkt"], keep="first")
    if sort_mode == "Title":
        view = view.sort_values(["title", "source_id"], ascending=[True, True])
    else:
        view = view.sort_values(["source_id", "title"], ascending=[True, True])
    view = view.head(max_items)

    st.caption(f"Showing {len(view)} sources")
    for _, row in view.iterrows():
        title = str(row.get("title", "")).strip() or str(row.get("source_id", "source"))
        source_id = str(row.get("source_id", ""))
        hkt = str(row.get("hkt", ""))
        url = str(row.get("url", "")).strip()
        label = f"{title[:78]} ({source_id}) [{hkt}]"
        if url:
            st.link_button(label, url, use_container_width=True)
        else:
            st.button(label, disabled=True, use_container_width=True)


def _resolve_clicked_node_from_plotly_event(point: dict, fig) -> tuple[Optional[int], Optional[str]]:
    clicked_node_id: Optional[int] = None
    clicked_hkt_label: Optional[str] = None

    point_id = point.get("id")
    if isinstance(point_id, str) and point_id.startswith("node:"):
        try:
            clicked_node_id = int(point_id.split(":", 1)[1])
        except ValueError:
            clicked_node_id = None

    if clicked_node_id is None:
        point_number = point.get("pointNumber")
        if isinstance(point_number, int) and fig.data:
            trace = fig.data[0]
            ids = getattr(trace, "ids", None)
            customdata = getattr(trace, "customdata", None)
            if ids is not None and 0 <= point_number < len(ids):
                pid = ids[point_number]
                if isinstance(pid, str) and pid.startswith("node:"):
                    try:
                        clicked_node_id = int(pid.split(":", 1)[1])
                    except ValueError:
                        clicked_node_id = None
            if customdata is not None and 0 <= point_number < len(customdata):
                row = customdata[point_number]
                if isinstance(row, (list, tuple)) and len(row) >= 3:
                    if clicked_node_id is None:
                        try:
                            clicked_node_id = int(row[0])
                        except (TypeError, ValueError):
                            clicked_node_id = None
                    clicked_hkt_label = str(row[2]) if row[2] is not None else None

    if clicked_node_id is None:
        custom_data = point.get("customdata", [])
        if isinstance(custom_data, (list, tuple)) and len(custom_data) >= 3:
            try:
                clicked_node_id = int(custom_data[0])
            except (TypeError, ValueError):
                clicked_node_id = None
            clicked_hkt_label = str(custom_data[2]) if custom_data[2] is not None else None

    return clicked_node_id, clicked_hkt_label


def render_tree_explorer_panel(tree_df: pd.DataFrame) -> None:
    if tree_df.empty or not {"id", "parent", "label", "source_count", "depth"}.issubset(tree_df.columns):
        st.info("Tree explorer is available for loaded SECA trees.")
        return

    with st.expander("Tree Explorer", expanded=True):
        search_value = st.text_input("Search label/tokens", value="", key="tree_explorer_search")
        min_sources = st.number_input(
            "Minimum sources",
            min_value=0,
            value=1,
            step=1,
            key="tree_explorer_min_sources",
        )
        max_depth = int(tree_df["depth"].max()) if not tree_df.empty else 0
        depth_limit = st.slider(
            "Max depth",
            min_value=0,
            max_value=max_depth,
            value=min(max_depth, 4),
            step=1,
            key="tree_explorer_depth_limit",
        )

        view = tree_df.copy()
        view = view[view["source_count"] >= min_sources]
        view = view[view["depth"] <= depth_limit]
        if search_value.strip():
            pattern = search_value.strip()
            label_series = view["label"].fillna("").astype(str)
            token_series = view["tokens"].fillna("").astype(str)
            view = view[
                label_series.str.contains(pattern, case=False, na=False)
                | token_series.str.contains(pattern, case=False, na=False)
            ]

        st.caption(f"Explorer nodes: {len(view)}")
        st.dataframe(
            view[["node_id", "hkt_id", "depth", "label", "source_count", "tokens"]]
            .sort_values(["depth", "source_count"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True,
        )

        rows_by_id = {row["id"]: row for _, row in view.iterrows()}
        children_by_parent: dict[str, list[str]] = {}
        for row in rows_by_id.values():
            parent_id = row["parent"] or ""
            children_by_parent.setdefault(parent_id, []).append(row["id"])

        for parent_id in children_by_parent:
            children_by_parent[parent_id].sort(
                key=lambda child_id: (
                    int(rows_by_id[child_id]["depth"]),
                    -int(rows_by_id[child_id]["source_count"]),
                )
            )

        render_limit = 250
        rendered = 0

        def render_node(node_id: str) -> None:
            nonlocal rendered
            if rendered >= render_limit:
                return
            row = rows_by_id.get(node_id)
            if row is None:
                return
            rendered += 1
            header = (
                f"{row['label']} · sources {int(row['source_count'])} · depth {int(row['depth'])}"
            )
            with st.expander(header, expanded=False):
                st.caption(
                    f"node_id={int(row['node_id'])}, hkt_id={int(row['hkt_id'])}, "
                    f"category={row.get('category', '')}"
                )
                if row.get("tokens"):
                    st.text(f"tokens: {row['tokens']}")
                for child_id in children_by_parent.get(node_id, []):
                    render_node(child_id)

        root_ids = children_by_parent.get("", [])
        for root_id in root_ids:
            render_node(root_id)

        if rendered >= render_limit:
            st.warning(f"Explorer truncated to {render_limit} nodes for readability.")


if "seca_tree" not in st.session_state:
    st.session_state.seca_tree = None
if "seca_tree_error" not in st.session_state:
    st.session_state.seca_tree_error = None
if "seca_tree_loaded" not in st.session_state:
    st.session_state.seca_tree_loaded = False
if "seca_tree_df" not in st.session_state:
    st.session_state.seca_tree_df = None
if "baseline_config_path" not in st.session_state:
    st.session_state.baseline_config_path = "seca_config.json"
if "timeline_manifest" not in st.session_state:
    st.session_state.timeline_manifest = None
if "timeline_trees" not in st.session_state:
    st.session_state.timeline_trees = None
if "timeline_selected_index" not in st.session_state:
    st.session_state.timeline_selected_index = 0
if "timeline_error" not in st.session_state:
    st.session_state.timeline_error = None
if "timeline_output_dir" not in st.session_state:
    st.session_state.timeline_output_dir = None
if "current_timeline_batch_index" not in st.session_state:
    st.session_state.current_timeline_batch_index = None
if "tree_validation" not in st.session_state:
    st.session_state.tree_validation = None
if "source_catalog" not in st.session_state:
    st.session_state.source_catalog = {}
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Structure"
if "previous_view_mode" not in st.session_state:
    st.session_state.previous_view_mode = "Structure"
if "source_view_enabled" not in st.session_state:
    st.session_state.source_view_enabled = False
if "structure_view_state" not in st.session_state:
    st.session_state.structure_view_state = {"active_hkt": "All HKTs", "active_node_id": None}
if "last_structure_view_state" not in st.session_state:
    st.session_state.last_structure_view_state = {"active_hkt": "All HKTs", "active_node_id": None}
if "sources_view_state" not in st.session_state:
    st.session_state.sources_view_state = {"seed_hkt": "All HKTs", "seed_node_id": None}

with st.sidebar:
    st.caption("Controls")

    config_submitted = False
    timeline_submitted = False

    with st.expander("SECA Config", expanded=False):
        with st.form("config_form"):
            st.caption("Use Basic for routine runs. Open Advanced only for option-level diagnostics.")
            st.caption("Basic settings")
            col_a, col_b = st.columns(2)
            with col_a:
                min_threshold = st.number_input(
                    "Min threshold vs max word count",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    format="%.2f",
                )
                similarity = st.number_input(
                    "Similarity threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    format="%.2f",
                )
                min_sources = st.number_input(
                    "Min sources to create branch",
                    min_value=1,
                    value=20,
                    step=1,
                )
            with col_b:
                alpha = st.number_input("Alpha", min_value=0.0, max_value=1.0, value=0.7, format="%.2f")
                beta = st.number_input("Beta", min_value=0.0, max_value=1.0, value=0.5, format="%.2f")
                wi_err = st.number_input(
                    "Word importance error threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    format="%.2f",
                )

            alpha_err = st.number_input(
                "Alpha error threshold", min_value=0.0, max_value=1.0, value=0.1, format="%.2f"
            )
            beta_err = st.number_input(
                "Beta error threshold", min_value=0.0, max_value=1.0, value=0.2, format="%.2f"
            )

            memory_mode = st.selectbox("Memory mode", ("Full", "SlidingWindow"))
            max_batches = st.text_input(
                "Max batches in memory (blank for null)", value="", help="Only used for SlidingWindow."
            )
            trigger_policy = st.selectbox(
                "Trigger policy mode",
                ["Placeholder", "PaperDiagnosticScaffold"],
                index=1,
                help="PaperDiagnosticScaffold matches newsmap diagnostics.",
            )

            with st.expander("Advanced options", expanded=False):
                st.caption("Only change these if you need option-level trigger tuning.")
                adv_a, adv_b = st.columns(2)
                with adv_a:
                    alpha_opt1 = st.number_input(
                        "Alpha option1 threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.1,
                        format="%.2f",
                    )
                    alpha_opt2 = st.number_input(
                        "Alpha option2 threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.2,
                        format="%.2f",
                    )
                    alpha_opt3 = st.number_input(
                        "Alpha option3 threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.3,
                        format="%.2f",
                    )
                    selected_alpha = st.selectbox(
                        "Selected alpha option", ["Option1", "Option2", "Option3"]
                    )

                with adv_b:
                    beta_opt1 = st.number_input(
                        "Beta option1 threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.13,
                        format="%.2f",
                    )
                    beta_opt2 = st.number_input(
                        "Beta option2 threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.2,
                        format="%.2f",
                    )
                    beta_opt3 = st.number_input(
                        "Beta option3 threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.2,
                        format="%.2f",
                    )
                    selected_beta = st.selectbox(
                        "Selected beta option", ["Option1", "Option2", "Option3"]
                    )

                wi_opt1 = st.number_input(
                    "Word importance option1 threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    format="%.2f",
                )
                wi_opt2 = st.number_input(
                    "Word importance option2 threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    format="%.2f",
                )
                selected_wi = st.selectbox(
                    "Selected word importance option", ["Option1", "Option2"]
                )

            config_output_path = st.text_input(
                "Config file path",
                value="seca_config.json",
                help="Written JSON that we later pass to `--config`.",
            )
            config_submitted = st.form_submit_button("Write config file")

    if config_submitted:
        try:
            max_batches_val = None
            if max_batches.strip():
                max_batches_val = int(max_batches)
            payload = build_config_payload(
                min_threshold,
                similarity,
                min_sources,
                alpha,
                beta,
                alpha_err,
                beta_err,
                wi_err,
                memory_mode,
                max_batches_val,
                alpha_opt1,
                alpha_opt2,
                alpha_opt3,
                beta_opt1,
                beta_opt2,
                beta_opt3,
                wi_opt1,
                wi_opt2,
                selected_alpha,
                selected_beta,
                selected_wi,
                trigger_policy,
            )
            Path(config_output_path).write_text(json.dumps(payload, indent=2))
            st.success(f"Config written to {config_output_path}")
            st.session_state.baseline_config_path = config_output_path
        except ValueError:
            st.error("Max batches must be an integer or blank.")

    with st.expander("Timeline Playback", expanded=True):
        with st.form("timeline_form"):
            timeline_batch_input = st.text_input(
                "Timeline batch JSON",
                value="crates/realtime-seca-core/tests/data/large_batch.json",
                help="SourceBatch JSON input used for baseline + incremental batches.",
            )
            timeline_output_dir = st.text_input(
                "Timeline output directory",
                value="seca_timeline",
                help="Writes tree_batch_*.json and timeline_manifest.json.",
            )
            timeline_chunk_count = st.number_input(
                "Chunk count",
                min_value=1,
                value=8,
                step=1,
                help="Number of sequential batches generated from the input sources.",
            )
            timeline_config_path = st.text_input(
                "Timeline config JSON",
                value=st.session_state.get("baseline_config_path", "seca_config.json"),
                help="Optional SecaConfig JSON path passed to `--config`.",
            )
            timeline_clean = st.checkbox(
                "Clean output directory before run",
                value=True,
            )
            timeline_submitted = st.form_submit_button("Run timeline and load")

    if timeline_submitted:
        try:
            with st.spinner("Running `cargo run -p realtime-seca-cli -- timeline`..."):
                output = run_timeline_command(
                    timeline_batch_input,
                    timeline_output_dir,
                    int(timeline_chunk_count),
                    timeline_config_path or None,
                    timeline_clean,
                )
            st.success("Timeline build completed.")
            st.code(output.strip() or "cargo run completed successfully.")
            st.session_state.source_catalog = load_source_catalog_from_batch(Path(timeline_batch_input))
            if load_timeline_from_output_dir(Path(timeline_output_dir)):
                st.success("Loaded timeline trees for slider visualization.")
            else:
                st.session_state.timeline_error = (
                    f"Timeline output not found or invalid in '{timeline_output_dir}'."
                )
                st.error(st.session_state.timeline_error)
        except subprocess.CalledProcessError as exc:
            st.session_state.timeline_error = "Timeline run failed; check console output."
            st.error(st.session_state.timeline_error)
            st.code(exc.stderr or exc.stdout or "unknown error")

if st.session_state.timeline_trees:
    total_batches = len(st.session_state.timeline_trees)
    max_batch_index = total_batches - 1
    default_batch_index = min(
        st.session_state.timeline_selected_index,
        max_batch_index,
    )
    batch_options = list(range(0, max_batch_index + 1))
    selected_batch_index = st.select_slider(
        "Processed batch (slider)",
        options=batch_options,
        value=default_batch_index,
    )
    st.session_state.timeline_selected_index = selected_batch_index
    st.session_state.current_timeline_batch_index = selected_batch_index

    current_tree = st.session_state.timeline_trees[selected_batch_index]
    st.session_state.seca_tree = current_tree
    tree_df, validation = build_tree_view_model(current_tree)
    st.session_state.seca_tree_df = tree_df
    st.session_state.tree_validation = validation

    manifest = st.session_state.timeline_manifest or {}
    files = manifest.get("files", [])
    current_file = files[selected_batch_index] if selected_batch_index < len(files) else ""
    node_count = len(current_tree.get("nodes", []))
    hkt_count = len(current_tree.get("hkts", []))
    source_count = len(current_tree.get("source_legend", []))
    st.caption(
        f"Timeline view: batch {selected_batch_index}/{max_batch_index}"
        + (f" • file: {current_file}" if current_file else "")
        + f" • hkts: {hkt_count} • nodes: {node_count} • sources: {source_count}"
    )
elif st.session_state.timeline_error:
    st.warning(st.session_state.timeline_error)
else:
    st.info("No timeline loaded yet. Use 'Run timeline and load' in the sidebar.")

tree_df = st.session_state.seca_tree_df
if tree_df is None:
    st.info("Run timeline and load to visualize the SECA tree.")
    st.stop()
data = tree_df.copy()

if st.session_state.get("tree_validation"):
    validation = st.session_state.tree_validation
    st.caption(
        "Tree validation: "
        f"HKTs={validation.get('total_hkts', 0)}, "
        f"nodes={validation.get('total_nodes', 0)}, "
        f"rendered={validation.get('rendered_nodes', 0)}, "
        f"skipped={validation.get('skipped_nodes', 0)}"
    )
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("HKTs", int(validation.get("total_hkts", 0)))
    metric_col2.metric("Nodes", int(validation.get("total_nodes", 0)))
    metric_col3.metric("Rendered", int(validation.get("rendered_nodes", 0)))
    metric_col4.metric("Skipped", int(validation.get("skipped_nodes", 0)))

expected_columns = {"category", "topic", "title", "source", "url", "date", "value"}
missing = expected_columns - set(data.columns)
if missing:
    st.warning(
        "Missing columns: " + ", ".join(sorted(missing)) +
        ". Treemap will still render but features may be limited."
    )

# Normalize core fields
if "category" not in data.columns:
    data["category"] = "Uncategorized"
if "topic" not in data.columns:
    data["topic"] = "Misc"
if "title" not in data.columns:
    data["title"] = "(untitled)"
if "source" not in data.columns:
    data["source"] = "(unknown)"
if "url" not in data.columns:
    data["url"] = ""
if "date" in data.columns:
    data = _parse_dates(data, "date")
else:
    data["date"] = dt.date.today()
if "value" not in data.columns:
    data["value"] = 1

filtered = data.copy()

if filtered.empty:
    st.warning("No results after filters.")
    st.stop()

filtered["weight"] = pd.to_numeric(filtered["value"], errors="coerce").fillna(1)

is_tree_dataset = {"id", "parent", "label", "depth"}.issubset(set(data.columns))
source_assignments = pd.DataFrame()
if st.session_state.get("seca_tree") and st.session_state.get("source_catalog"):
    source_assignments = build_source_assignments(
        st.session_state.seca_tree, st.session_state.source_catalog
    )

if is_tree_dataset:
    all_hkts = sorted(filtered["category"].dropna().unique().tolist())
    structure_state = st.session_state.structure_view_state
    active_hkt_default = structure_state.get("active_hkt", "All HKTs")
    if active_hkt_default != "All HKTs" and active_hkt_default not in all_hkts:
        active_hkt_default = "All HKTs"

    mode_col1, mode_col2, mode_col3 = st.columns([2, 2, 2])
    with mode_col1:
        st.caption("Mode: Structure")
    with mode_col2:
        source_view_enabled = st.toggle(
            "Source View",
            value=st.session_state.source_view_enabled,
            help="Show source panel next to treemap when enabled.",
        )
    hkt_options = ["All HKTs"] + all_hkts
    with mode_col3:
        active_hkt = st.selectbox(
            "Opened HKT",
            hkt_options,
            index=hkt_options.index(active_hkt_default),
        )
    if st.button("Reset opened node", use_container_width=True):
        st.session_state.structure_view_state = {
            "active_hkt": active_hkt,
            "active_node_id": None,
        }
        st.rerun()

    node_candidates = filter_structure_df(filtered, active_hkt, None)
    node_options = ["All nodes"] + [
        f"{int(row['node_id'])}: {row['label']}" for _, row in node_candidates.sort_values(["depth", "node_id"]).iterrows()
    ]
    previous_node_id = structure_state.get("active_node_id")
    default_node_label = "All nodes"
    if previous_node_id is not None:
        matching = [opt for opt in node_options if opt.startswith(f"{int(previous_node_id)}:")]
        if matching:
            default_node_label = matching[0]
    active_node_label = st.selectbox(
        "Opened Node",
        node_options,
        index=node_options.index(default_node_label) if default_node_label in node_options else 0,
        help="Uses the last clicked node; you can also set it manually.",
    )
    if not HAS_PLOTLY_EVENTS:
        st.caption("Install `streamlit-plotly-events` for reliable treemap click-to-open behavior.")
    st.caption(
        f"Current open context: HKT={st.session_state.structure_view_state.get('active_hkt', 'All HKTs')} "
        f"node={st.session_state.structure_view_state.get('active_node_id')}"
    )
    active_node_id = None if active_node_label == "All nodes" else int(active_node_label.split(":", 1)[0])

    st.session_state.structure_view_state = {
        "active_hkt": active_hkt,
        "active_node_id": active_node_id,
    }

    view_mode = "Structure"
    st.session_state.view_mode = "Structure"
    st.session_state.source_view_enabled = source_view_enabled
else:
    view_mode = "Structure"
    source_view_enabled = st.session_state.source_view_enabled
    active_hkt = "All HKTs"
    active_node_id = None

structure_filtered = filtered
if is_tree_dataset:
    # Keep the treemap navigable: filter by HKT only, not by opened node subtree.
    structure_filtered = filter_structure_df(filtered, active_hkt, None)
    if structure_filtered.empty:
        structure_filtered = filtered

scope_hkt = active_hkt
scope_node_id = active_node_id

scoped_sources = filter_source_assignments(source_assignments, scope_hkt, None)

# Align source panel with the currently visible node scope (opened node subtree when present).
if is_tree_dataset and not scoped_sources.empty:
    # Source panel follows currently opened node subtree when a node is open.
    scope_tree_df = filter_structure_df(filtered, scope_hkt, scope_node_id)
    if "node_id" in scope_tree_df.columns:
        visible_node_ids = set(scope_tree_df["node_id"].dropna().astype(int).tolist())
        scoped_sources = scoped_sources[scoped_sources["node_id"].isin(visible_node_ids)]

# Source review reflects the currently opened node.
if is_tree_dataset and source_view_enabled and active_node_id is None:
    scoped_sources = scoped_sources.iloc[0:0]

st.subheader("Treemap")
if source_view_enabled:
    main_col, source_col = st.columns([3, 2])
else:
    main_col = st.container()
    source_col = None

with main_col:
    if is_tree_dataset:
        max_strength = structure_filtered["strength_pct"].max() if "strength_pct" in structure_filtered.columns else 1.0
        fig = px.treemap(
            structure_filtered,
            names="label",
            ids="id",
            parents="parent",
            values="weight",
            custom_data=["node_id", "hkt_id", "category"],
            color="strength_pct" if "strength_pct" in structure_filtered.columns else "depth",
            color_continuous_scale="RdYlGn",
            range_color=(0, max(1.0, float(max_strength))),
            hover_data={
                "node_id": True,
                "hkt_id": True,
                "depth": True,
                "source_count": True,
                "tokens": True,
                "strength_pct": ":.2f",
            },
            height=720,
        )
        fig.update_traces(textinfo="label+value")
    else:
        path_level_columns = sorted(
            [col for col in data.columns if col.startswith("path_level_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        path_columns = ["category"]
        path_columns.extend(path_level_columns)
        if not path_level_columns:
            path_columns.append("topic")
        fig = px.treemap(
            filtered,
            path=path_columns,
            values="weight",
            color="strength_pct" if "strength_pct" in filtered.columns else "category",
            color_continuous_scale="RdYlGn",
            range_color=(0, 1),
            hover_data={
                "date": True,
                "url": True,
                "weight": True,
                "strength_pct": ":.2f",
                "tokens": True,
            },
            height=720,
        )
        fig.update_traces(textinfo="label+value")
    fig.update_traces(root_color="lightgrey")
    fig.update_layout(
        margin=dict(t=30, l=10, r=10, b=10),
        uniformtext=dict(minsize=10, mode="hide"),
        clickmode="event+select",
    )
    clicked_points = []
    if is_tree_dataset and view_mode == "Structure" and HAS_PLOTLY_EVENTS:
        clicked_points = plotly_events(
            fig,
            click_event=True,
            select_event=False,
            hover_event=False,
            key="main_treemap_chart_events",
            override_height=720,
        )
    else:
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="main_treemap_chart",
        )

    if is_tree_dataset and view_mode == "Structure" and clicked_points:
        point = clicked_points[-1]
        clicked_node_id, clicked_hkt_label = _resolve_clicked_node_from_plotly_event(point, fig)

        if clicked_node_id is None and point.get("label"):
            label = str(point.get("label"))
            match = structure_filtered[structure_filtered["label"] == label]
            if len(match) == 1:
                clicked_node_id = int(match.iloc[0]["node_id"])
                clicked_hkt_label = str(match.iloc[0]["category"])

        if clicked_node_id is not None:
            if clicked_hkt_label is None and "category" in structure_filtered.columns:
                matches = structure_filtered[structure_filtered["node_id"] == clicked_node_id]
                if not matches.empty:
                    clicked_hkt_label = matches.iloc[0]["category"]
            next_hkt = str(clicked_hkt_label) if clicked_hkt_label else "All HKTs"
            current_state = st.session_state.structure_view_state or {}
            if (
                current_state.get("active_node_id") != clicked_node_id
                or current_state.get("active_hkt") != next_hkt
            ):
                st.session_state.structure_view_state = {
                    "active_hkt": next_hkt,
                    "active_node_id": clicked_node_id,
                }
                st.rerun()

if source_col is not None:
    with source_col:
        missing_open_node = (
            is_tree_dataset
            and view_mode == "Structure"
            and source_view_enabled
            and active_node_id is None
        )
        render_source_review_panel(
            scoped_sources,
            key_prefix=f"{view_mode.lower()}_sources",
            require_open_node=missing_open_node,
        )

st.divider()
st.subheader("Data Table")
table_columns = ["category", "topic", "source", "title", "date", "url", "weight"]
sort_columns = ["weight"]
sort_ascending = [False]
if is_tree_dataset:
    table_columns = ["node_id", "hkt_id", "depth", "label", "source_count", "tokens", "weight"]
    sort_columns = ["source_count", "depth"]
    sort_ascending = [False, True]
    st.dataframe(
        structure_filtered[table_columns].sort_values(
            sort_columns, ascending=sort_ascending
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.dataframe(
        filtered[table_columns].sort_values(
            sort_columns, ascending=sort_ascending
        ),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Tip: Provide your own CSV/JSON with columns: category, topic, source, title, url, date, value."
)

if st.session_state.seca_tree_df is not None and is_tree_dataset:
    render_tree_explorer_panel(st.session_state.seca_tree_df)

if st.session_state.seca_tree:
    with st.expander("Raw tree JSON", expanded=False):
        st.json(st.session_state.seca_tree)
elif st.session_state.seca_tree_error:
    st.warning(st.session_state.seca_tree_error)
