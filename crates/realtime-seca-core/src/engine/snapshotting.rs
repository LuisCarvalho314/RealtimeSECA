use super::*;

impl SecaEngine {
    pub fn snapshot(&self) -> Result<EngineSnapshot, SecaError> {
        Ok(EngineSnapshot {
            schema_version: 3,
            engine_version: ENGINE_VERSION.to_string(),
            config: self.config.clone(),
            has_baseline: self.has_baseline,
            last_processed_batch_index: self.last_processed_batch_index,
            hkt_build_output: self.hkt_build_output.clone(),
            baseline_word_legend: self.baseline_word_legend.clone(),
            baseline_source_legend: self.baseline_source_legend.clone(),
            processed_batches: self.processed_batches.clone(),
            last_batch_word_stats_summary: self.last_batch_word_stats_summary.clone(),
            source_id_by_url: self.source_id_by_url.clone(),
            url_by_source_id: self.url_by_source_id.clone(),
            source_batch_index_by_internal_source_id: self.source_batch_index_by_internal_source_id.clone(),
            source_ids_by_batch_index: self.source_ids_by_batch_index.clone(),
            archived_subtrees_by_root_id: self.archived_subtrees_by_root_id.clone(),
            logically_removed_hkts_by_id: self
                .logically_removed_hkts_by_id
                .iter()
                .map(|(hkt_id, entry)| {
                    (
                        *hkt_id,
                        crate::types::LogicalRemovedHktSnapshot {
                            hkt: entry.hkt.clone(),
                            old_parent_node_id: entry.old_parent_node_id,
                        },
                    )
                })
                .collect(),
            node_diagnostics_by_id: self.node_diagnostics_by_id.clone(),
            next_hkt_id: self.next_hkt_id,
            next_node_id: self.next_node_id,
        })
    }

    pub fn load_snapshot(snapshot: EngineSnapshot) -> Result<Self, SecaError> {
        let mut engine = Self::new(snapshot.config)?;
        engine.last_processed_batch_index = snapshot.last_processed_batch_index;
        engine.has_baseline = snapshot.has_baseline;
        engine.last_update_explanation = None;
        engine.hkt_build_output = snapshot.hkt_build_output;
        engine.baseline_word_legend = snapshot.baseline_word_legend;
        engine.baseline_source_legend = snapshot.baseline_source_legend;
        engine.processed_batches = snapshot.processed_batches;
        engine.last_batch_word_stats_summary = snapshot.last_batch_word_stats_summary;
        engine.source_id_by_url = snapshot.source_id_by_url;
        engine.url_by_source_id = snapshot.url_by_source_id;
        engine.source_batch_index_by_internal_source_id = snapshot.source_batch_index_by_internal_source_id;
        engine.source_ids_by_batch_index = snapshot.source_ids_by_batch_index;
        engine.archived_subtrees_by_root_id = snapshot.archived_subtrees_by_root_id;
        engine.logically_removed_hkts_by_id = snapshot
            .logically_removed_hkts_by_id
            .into_iter()
            .map(|(hkt_id, entry)| {
                (
                    hkt_id,
                    crate::engine::LogicalRemovedHkt {
                        hkt: entry.hkt,
                        old_parent_node_id: entry.old_parent_node_id,
                    },
                )
            })
            .collect();
        engine.node_diagnostics_by_id = snapshot.node_diagnostics_by_id;
        engine.next_hkt_id = snapshot.next_hkt_id;
        engine.next_node_id = snapshot.next_node_id;

        if engine.has_baseline && engine.hkt_build_output.is_none() {
            return Err(SecaError::StateError {
                message: "snapshot indicates baseline exists but tree state is missing".to_string(),
            });
        }

        Ok(engine)
    }

    pub fn explain_last_update(&self) -> Option<&UpdateExplanation> {
        self.last_update_explanation.as_ref()
    }

    pub fn export_baseline_tree(&self) -> Result<BaselineTreeExport, SecaError> {
        let Some(hkt_build_output) = &self.hkt_build_output else {
            return Err(SecaError::StateError {
                message: "baseline tree has not been built yet".to_string(),
            });
        };

        let mut hkts: Vec<BaselineHktExport> = hkt_build_output
            .hkts_by_id
            .values()
            .map(|hkt| BaselineHktExport {
                hkt_id: hkt.hkt_id,
                parent_node_id: hkt.parent_node_id,
                expected_words: hkt.expected_words.iter().copied().collect(),
                node_ids: hkt.nodes.iter().map(|node| node.node_id).collect(),
                is_state1: hkt.is_state1,
            })
            .collect();

        let mut nodes: Vec<BaselineNodeExport> = hkt_build_output
            .nodes_by_id
            .values()
            .map(|node| BaselineNodeExport {
                node_id: node.node_id,
                hkt_id: node.hkt_id,
                word_ids: node.word_ids.iter().copied().collect(),
                source_ids: node.source_ids.iter().copied().collect(),
                top_words: node.top_words.iter().copied().collect(),
                is_refuge_node: node.is_refuge_node(),
            })
            .collect();

        hkts.sort_by_key(|hkt| hkt.hkt_id);
        nodes.sort_by_key(|node| node.node_id);

        let mut logically_removed_hkts: Vec<BaselineHktExport> = self
            .logically_removed_hkts_by_id
            .values()
            .map(|entry| BaselineHktExport {
                hkt_id: entry.hkt.hkt_id,
                parent_node_id: entry.hkt.parent_node_id,
                expected_words: entry.hkt.expected_words.iter().copied().collect(),
                node_ids: entry.hkt.nodes.iter().map(|node| node.node_id).collect(),
                is_state1: entry.hkt.is_state1,
            })
            .collect();

        logically_removed_hkts.sort_by_key(|hkt| hkt.hkt_id);

        Ok(BaselineTreeExport {
            hkts,
            nodes,
            logically_removed_hkts,
        })
    }

    pub fn export_baseline_tree_verbose(&self) -> Result<BaselineTreeVerboseExport, SecaError> {
        let Some(hkt_build_output) = &self.hkt_build_output else {
            return Err(SecaError::StateError {
                message: "baseline tree has not been built yet".to_string(),
            });
        };

        let map_word_ref = |word_id: i32, legend: &std::collections::BTreeMap<i32, String>| {
            let token = if word_id == -1 {
                Some("<REFUGE>".to_string())
            } else {
                legend.get(&word_id).cloned()
            };
            VerboseWordRef { word_id, token }
        };

        let map_source_ref =
            |internal_source_id: i64, legend: &std::collections::BTreeMap<i64, String>| {
                VerboseSourceRef {
                    internal_source_id,
                    external_source_id: legend
                        .get(&internal_source_id)
                        .cloned()
                        .unwrap_or_else(|| "<UNKNOWN_SOURCE>".to_string()),
                }
            };

        let mut hkts: Vec<BaselineHktVerboseExport> = hkt_build_output
            .hkts_by_id
            .values()
            .map(|hkt| {
                let mut union_word_ids = std::collections::BTreeSet::new();
                for node in &hkt.nodes {
                    union_word_ids.extend(node.word_ids.iter().copied());
                }

                BaselineHktVerboseExport {
                    hkt_id: hkt.hkt_id,
                    parent_node_id: hkt.parent_node_id,
                    expected_words: hkt
                        .expected_words
                        .iter()
                        .copied()
                        .map(|word_id| map_word_ref(word_id, &self.baseline_word_legend))
                        .collect(),
                    all_node_words_union: union_word_ids
                        .into_iter()
                        .map(|word_id| map_word_ref(word_id, &self.baseline_word_legend))
                        .collect(),
                    node_ids: hkt.nodes.iter().map(|node| node.node_id).collect(),
                    is_state1: hkt.is_state1,
                }
            })
            .collect();

        let mut nodes: Vec<BaselineNodeVerboseExport> = hkt_build_output
            .nodes_by_id
            .values()
            .map(|node| BaselineNodeVerboseExport {
                node_id: node.node_id,
                hkt_id: node.hkt_id,
                words: node
                    .word_ids
                    .iter()
                    .copied()
                    .map(|word_id| map_word_ref(word_id, &self.baseline_word_legend))
                    .collect(),
                sources: node
                    .source_ids
                    .iter()
                    .copied()
                    .map(|source_id| map_source_ref(source_id, &self.baseline_source_legend))
                    .collect(),
                top_words: node
                    .top_words
                    .iter()
                    .copied()
                    .map(|word_id| map_word_ref(word_id, &self.baseline_word_legend))
                    .collect(),
                is_refuge_node: node.is_refuge_node(),
                diagnostics: self.node_diagnostics_by_id.get(&node.node_id).cloned(),
            })
            .collect();

        hkts.sort_by_key(|hkt| hkt.hkt_id);
        nodes.sort_by_key(|node| node.node_id);

        let word_legend: Vec<WordLegendEntry> = self
            .baseline_word_legend
            .iter()
            .map(|(word_id, token)| WordLegendEntry {
                word_id: *word_id,
                token: Some(token.clone()),
            })
            .collect();

        let source_legend: Vec<SourceLegendEntry> = self
            .baseline_source_legend
            .iter()
            .map(
                |(internal_source_id, external_source_id)| SourceLegendEntry {
                    internal_source_id: *internal_source_id,
                    external_source_id: external_source_id.clone(),
                },
            )
            .collect();

        let mut logically_removed_hkts: Vec<BaselineHktVerboseExport> = self
            .logically_removed_hkts_by_id
            .values()
            .map(|entry| {
                let mut union_word_ids = std::collections::BTreeSet::new();
                for node in &entry.hkt.nodes {
                    union_word_ids.extend(node.word_ids.iter().copied());
                }

                BaselineHktVerboseExport {
                    hkt_id: entry.hkt.hkt_id,
                    parent_node_id: entry.hkt.parent_node_id,
                    expected_words: entry
                        .hkt
                        .expected_words
                        .iter()
                        .copied()
                        .map(|word_id| map_word_ref(word_id, &self.baseline_word_legend))
                        .collect(),
                    all_node_words_union: union_word_ids
                        .into_iter()
                        .map(|word_id| map_word_ref(word_id, &self.baseline_word_legend))
                        .collect(),
                    node_ids: entry.hkt.nodes.iter().map(|node| node.node_id).collect(),
                    is_state1: entry.hkt.is_state1,
                }
            })
            .collect();

        logically_removed_hkts.sort_by_key(|hkt| hkt.hkt_id);

        Ok(BaselineTreeVerboseExport {
            hkts,
            nodes,
            word_legend,
            source_legend,
            logically_removed_hkts,
        })
    }
}
