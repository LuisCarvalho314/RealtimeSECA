pub(crate) fn print_usage(arguments: &[String]) {
    let executable_name = arguments
        .first()
        .map(String::as_str)
        .unwrap_or("realtime-seca-cli");
    eprintln!("Usage:");
    eprintln!("  {executable_name} baseline <input_batch.json> [--config <config.json>] [--snapshot-out <snapshot.json>] [--dump-tree <tree.json>] [--dump-tree-verbose <tree_verbose.json>] [--print-tree]");
    eprintln!("  {executable_name} timeline <input_batch.json> --out-dir <dir> [--chunk-count <n> | --chunk-size <n>] [--config <config.json>] [--clean-out-dir]");
    eprintln!("  {executable_name} update <batch.json> --config <config.json> [--state-in <state.json>] --state-out <state.json> --dump-tree-verbose <tree.json>");
    eprintln!("  {executable_name} timeline-many <batch0.json> [batch1.json ...] --out-dir <dir> [--config <config.json>]");
    eprintln!("  {executable_name} from-csv <input.csv> <output.json> [--batch-index <n>] [--min-tokens <n>]");
}

pub(crate) fn print_verbose_tree(tree: &realtime_seca_core::BaselineTreeVerboseExport) {
    use std::collections::{BTreeMap, BTreeSet};

    let hkts_by_id: BTreeMap<i32, &realtime_seca_core::BaselineHktVerboseExport> =
        tree.hkts.iter().map(|hkt| (hkt.hkt_id, hkt)).collect();

    let nodes_by_id: BTreeMap<i32, &realtime_seca_core::BaselineNodeVerboseExport> =
        tree.nodes.iter().map(|node| (node.node_id, node)).collect();

    let mut child_hkt_ids_by_parent_node_id: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    for hkt in &tree.hkts {
        if hkt.parent_node_id != 0 {
            child_hkt_ids_by_parent_node_id
                .entry(hkt.parent_node_id)
                .or_default()
                .push(hkt.hkt_id);
        }
    }

    let root_hkt_ids: Vec<i32> = tree
        .hkts
        .iter()
        .filter(|h| h.parent_node_id == 0)
        .map(|h| h.hkt_id)
        .collect();
    let mut visited_hkts = BTreeSet::new();

    for root_hkt_id in root_hkt_ids {
        print_hkt_recursive(
            root_hkt_id,
            0,
            &hkts_by_id,
            &nodes_by_id,
            &child_hkt_ids_by_parent_node_id,
            &mut visited_hkts,
        );
    }
}

pub(crate) fn print_hkt_recursive(
    hkt_id: i32,
    indent_level: usize,
    hkts_by_id: &std::collections::BTreeMap<i32, &realtime_seca_core::BaselineHktVerboseExport>,
    nodes_by_id: &std::collections::BTreeMap<i32, &realtime_seca_core::BaselineNodeVerboseExport>,
    child_hkt_ids_by_parent_node_id: &std::collections::BTreeMap<i32, Vec<i32>>,
    visited_hkts: &mut std::collections::BTreeSet<i32>,
) {
    if !visited_hkts.insert(hkt_id) {
        println!("{}HKT {} (already visited)", indent(indent_level), hkt_id);
        return;
    }

    let Some(hkt) = hkts_by_id.get(&hkt_id) else {
        println!("{}HKT {} (missing)", indent(indent_level), hkt_id);
        return;
    };

    if hkt.parent_node_id == 0 {
        println!("{}HKT {} (root)", indent(indent_level), hkt.hkt_id);
    } else {
        println!(
            "{}HKT {} (parent_node_id={})",
            indent(indent_level),
            hkt.hkt_id,
            hkt.parent_node_id
        );
    }

    if !hkt.all_node_words_union.is_empty() {
        let union_words = hkt
            .all_node_words_union
            .iter()
            .map(|w| w.token.clone().unwrap_or_else(|| format!("#{}", w.word_id)))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "{}all_node_words_union=[{}]",
            indent(indent_level + 1),
            union_words
        );
    }

    if !hkt.expected_words.is_empty() {
        let expected_words = hkt
            .expected_words
            .iter()
            .map(|w| w.token.clone().unwrap_or_else(|| format!("#{}", w.word_id)))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "{}expected_words_excluding_seed=[{}]",
            indent(indent_level + 1),
            expected_words
        );
    }

    for node_id in &hkt.node_ids {
        let Some(node) = nodes_by_id.get(node_id) else {
            continue;
        };

        let words = node
            .words
            .iter()
            .map(|w| w.token.clone().unwrap_or_else(|| format!("#{}", w.word_id)))
            .collect::<Vec<_>>()
            .join(", ");

        let sources = node
            .sources
            .iter()
            .map(|s| s.external_source_id.clone())
            .collect::<Vec<_>>()
            .join(", ");

        if node.is_refuge_node {
            println!(
                "{}Node {} [REFUGE] words=[{}] sources=[{}]",
                indent(indent_level + 1),
                node.node_id,
                words,
                sources
            );
        } else {
            println!(
                "{}Node {} words=[{}] sources=[{}]",
                indent(indent_level + 1),
                node.node_id,
                words,
                sources
            );
        }

        if !node.top_words.is_empty() {
            let top_words = node
                .top_words
                .iter()
                .map(|w| w.token.clone().unwrap_or_else(|| format!("#{}", w.word_id)))
                .collect::<Vec<_>>()
                .join(", ");
            println!("{}top_words=[{}]", indent(indent_level + 2), top_words);
        }

        if let Some(child_hkts) = child_hkt_ids_by_parent_node_id.get(&node.node_id) {
            for child_hkt_id in child_hkts {
                print_hkt_recursive(
                    *child_hkt_id,
                    indent_level + 2,
                    hkts_by_id,
                    nodes_by_id,
                    child_hkt_ids_by_parent_node_id,
                    visited_hkts,
                );
            }
        }
    }
}

pub(crate) fn indent(level: usize) -> String {
    "  ".repeat(level)
}
