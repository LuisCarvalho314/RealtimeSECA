use crate::error::SecaError;
use crate::tree::models::{Hkt, Node, SourceWordRecord};
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Debug, Clone)]
pub struct HktBuildOutput {
    pub hkts_by_id: BTreeMap<i32, Hkt>,
    pub nodes_by_id: BTreeMap<i32, Node>,
}

/// Builder for the *HKT construction* phase reflected in the provided C# code.
/// This is intentionally separate from the later SECA real-time update/rebuild pipeline.
#[derive(Debug, Clone)]
pub struct HktBuilder {
    pub minimum_threshold_against_max_word_count: f64,
    pub similarity_threshold: f64,
    pub minimum_number_of_sources_to_create_branch_for_node: usize,
}

impl HktBuilder {
    pub fn new(
        minimum_threshold_against_max_word_count: f64,
        similarity_threshold: f64,
        minimum_number_of_sources_to_create_branch_for_node: usize,
    ) -> Result<Self, SecaError> {
        if !(0.0..=1.0).contains(&minimum_threshold_against_max_word_count) {
            return Err(SecaError::InvalidConfiguration {
                message: "minimum_threshold_against_max_word_count must be in [0,1]".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&similarity_threshold) {
            return Err(SecaError::InvalidConfiguration {
                message: "similarity_threshold must be in [0,1]".to_string(),
            });
        }

        Ok(Self {
            minimum_threshold_against_max_word_count,
            similarity_threshold,
            minimum_number_of_sources_to_create_branch_for_node,
        })
    }

    /// Entry point matching the C# "step 3 + step 5" pattern:
    /// - create first HKT
    /// - recursively create branches
    pub fn build_full_tree(
        &self,
        source_word_records_sorted: Vec<SourceWordRecord>,
    ) -> Result<HktBuildOutput, SecaError> {
        let mut state = BuilderState::default();

        let root_input = records_to_map_by_source_word_id(source_word_records_sorted);
        let root_hkt = self.create_hkt(&mut state, root_input, 0)?;

        if let Some(root_hkt) = root_hkt {
            let root_hkt_id = root_hkt.hkt_id;
            let root_hkt_clone_for_recursion = root_hkt.clone();
            state.hkts_by_id.insert(root_hkt.hkt_id, root_hkt);

            // For recursion, use the original root scope sorted by word count (same conceptual input as C# mainWordDS).
            let root_scope_records = state
                .hkt_input_scopes
                .get(&root_hkt_id)
                .cloned()
                .unwrap_or_default();

            self.create_branches(
                &mut state,
                &root_hkt_clone_for_recursion,
                &root_scope_records,
                0,
            )?;
        }

        Ok(HktBuildOutput {
            hkts_by_id: state.hkts_by_id,
            nodes_by_id: state.nodes_by_id,
        })
    }

    fn create_hkt(
        &self,
        state: &mut BuilderState,
        mut source_word_map: BTreeMap<i32, SourceWordRecord>,
        parent_node_id: i32,
    ) -> Result<Option<Hkt>, SecaError> {
        let expected_words = self.find_expected_words_general(&source_word_map);

        if expected_words.is_empty() {
            return Ok(None);
        }

        let hkt_id = state.next_hkt_id();
        let mut hkt = Hkt::new(hkt_id, parent_node_id, expected_words.clone());

        // Preserve the input scope for branch generation (analogous to C# mainWordDS / sourceWordDS usage).
        state
            .hkt_input_scopes
            .insert(hkt_id, source_word_map.values().cloned().collect());

        // Create first node from the first (highest word count) entry.
        let first_node_id = state.next_node_id();
        let first_node =
            self.create_node_from_first_word(first_node_id, hkt_id, &source_word_map)?;
        hkt.nodes.push(first_node.clone());
        state.nodes_by_id.insert(first_node.node_id, first_node);

        let first_word_id = first_entry(&source_word_map)
            .ok_or_else(|| SecaError::StateError {
                message: "source_word_map unexpectedly empty while creating first node".to_string(),
            })?
            .word_id;

        self.remove_word_and_its_corresponding_sources(first_word_id, &mut source_word_map);
        hkt.expected_words.remove(&first_word_id);

        // Iterate expected words in deterministic ascending order (C# HashSet iteration was nondeterministic).
        let expected_word_ids: Vec<i32> = hkt.expected_words.iter().copied().collect();

        for expected_word_id in expected_word_ids {
            let sources_of_expected_word = source_ids_for_word(&source_word_map, expected_word_id);

            if sources_of_expected_word.is_empty() {
                // If already removed due to prior mutation, skip.
                continue;
            }

            let best_collided_node_index =
                self.find_best_collided_node_index(&hkt.nodes, &sources_of_expected_word);

            if let Some(node_index) = best_collided_node_index {
                hkt.nodes[node_index].word_ids.insert(expected_word_id);
                hkt.nodes[node_index]
                    .source_ids
                    .extend(sources_of_expected_word.iter().copied());

                let collided_node_id = hkt.nodes[node_index].node_id;
                if let Some(global_node) = state.nodes_by_id.get_mut(&collided_node_id) {
                    global_node.word_ids.insert(expected_word_id);
                    global_node
                        .source_ids
                        .extend(sources_of_expected_word.iter().copied());
                }

                self.remove_word_and_its_corresponding_sources(
                    expected_word_id,
                    &mut source_word_map,
                );
            } else {
                let new_node_id = state.next_node_id();
                let new_node = self.create_node_for_specific_word(
                    new_node_id,
                    hkt_id,
                    expected_word_id,
                    &source_word_map,
                )?;
                hkt.nodes.push(new_node.clone());
                state.nodes_by_id.insert(new_node.node_id, new_node);

                self.remove_word_and_its_corresponding_sources(
                    expected_word_id,
                    &mut source_word_map,
                );
            }
        }

        // Refuge node for leftover sources not mapped to any node.
        let refugee_sources = find_refugee_sources(&source_word_map, &hkt.nodes);
        if !refugee_sources.is_empty() {
            let refuge_node_id = state.next_node_id();
            let refuge_node =
                self.create_node_for_refuge_sources(refuge_node_id, hkt_id, &refugee_sources);
            hkt.nodes.push(refuge_node.clone());
            state.nodes_by_id.insert(refuge_node.node_id, refuge_node);
        }

        Ok(Some(hkt))
    }

    fn create_branches(
        &self,
        state: &mut BuilderState,
        hkt: &Hkt,
        source_word_scope: &[SourceWordRecord],
        current_hkt_level: usize,
    ) -> Result<(), SecaError> {
        let _ = current_hkt_level; // kept for parity/future instrumentation

        for node in &hkt.nodes {
            if node.source_ids.len() <= self.minimum_number_of_sources_to_create_branch_for_node {
                continue;
            }

            let mut temp_source_word_records: Vec<SourceWordRecord> = Vec::new();

            if !node.is_refuge_node() {
                // C# logic:
                // include source-word rows whose source is in node.sourceIds
                // but whose word is NOT already inside node.wordIds
                for record in source_word_scope {
                    if !node.word_ids.contains(&record.word_id)
                        && node.source_ids.contains(&record.source_id)
                    {
                        temp_source_word_records.push(record.clone());
                    }
                }
            } else {
                // Refuge node: include all rows for sources in this node
                for record in source_word_scope {
                    if node.source_ids.contains(&record.source_id) {
                        temp_source_word_records.push(record.clone());
                    }
                }
            }

            self.update_word_number_of_sources(&mut temp_source_word_records);

            // Equivalent to C#:
            // mainWordDS = tempSourceWordDS.OrderByDescending(wordNoOfSources)
            let mut sorted_scope = temp_source_word_records;
            sort_source_word_records_desc(sorted_scope.as_mut_slice());

            if sorted_scope.is_empty() {
                continue;
            }

            // "topWords" helper (C# addNodeTopWords)
            if let Some(global_node) = state.nodes_by_id.get_mut(&node.node_id) {
                self.add_node_top_words(global_node, &sorted_scope);
            }

            let child_input_map = records_to_map_by_source_word_id(sorted_scope.clone());
            let child_hkt = self.create_hkt(state, child_input_map, node.node_id)?;

            if let Some(child_hkt) = child_hkt {
                let child_hkt_clone = child_hkt.clone();
                let child_hkt_id = child_hkt.hkt_id;
                state.hkts_by_id.insert(child_hkt.hkt_id, child_hkt);

                if !sorted_scope.is_empty() {
                    let recurse_scope = state
                        .hkt_input_scopes
                        .get(&child_hkt_id)
                        .cloned()
                        .unwrap_or_else(|| sorted_scope.clone());

                    self.create_branches(
                        state,
                        &child_hkt_clone,
                        &recurse_scope,
                        current_hkt_level + 1,
                    )?;
                }
            }
        }

        Ok(())
    }

    fn find_expected_words_general(
        &self,
        source_word_map: &BTreeMap<i32, SourceWordRecord>,
    ) -> BTreeSet<i32> {
        let mut expected_words = BTreeSet::new();

        let Some(first_record) = first_entry(source_word_map) else {
            return expected_words;
        };

        let maximum_word_count = first_record.word_number_of_sources;
        if maximum_word_count == 0 {
            return expected_words;
        }

        // Mirrors C# loop over sorted rows; stops at first ratio below threshold.
        for record in source_word_map.values() {
            let ratio = record.word_number_of_sources as f64 / maximum_word_count as f64;
            if ratio >= self.minimum_threshold_against_max_word_count {
                expected_words.insert(record.word_id);
            } else {
                break;
            }
        }

        expected_words
    }

    fn create_node_from_first_word(
        &self,
        new_node_id: i32,
        hkt_id: i32,
        source_word_map: &BTreeMap<i32, SourceWordRecord>,
    ) -> Result<Node, SecaError> {
        let first_record = first_entry(source_word_map).ok_or_else(|| SecaError::StateError {
            message: "cannot create node from empty source_word_map".to_string(),
        })?;

        self.create_node_for_specific_word(
            new_node_id,
            hkt_id,
            first_record.word_id,
            source_word_map,
        )
    }

    fn create_node_for_specific_word(
        &self,
        new_node_id: i32,
        hkt_id: i32,
        word_id: i32,
        source_word_map: &BTreeMap<i32, SourceWordRecord>,
    ) -> Result<Node, SecaError> {
        let mut node = Node::new(new_node_id, hkt_id);
        node.word_ids.insert(word_id);

        for record in source_word_map.values() {
            if record.word_id == word_id {
                node.source_ids.insert(record.source_id);
            }
        }

        if node.source_ids.is_empty() {
            return Err(SecaError::StateError {
                message: format!(
                    "word_id {} had no sources in scope while creating node",
                    word_id
                ),
            });
        }

        Ok(node)
    }

    fn create_node_for_refuge_sources(
        &self,
        new_node_id: i32,
        hkt_id: i32,
        refuge_sources: &BTreeSet<i32>,
    ) -> Node {
        let mut node = Node::new(new_node_id, hkt_id);
        node.word_ids.insert(-1); // C# refuge marker
        node.source_ids.extend(refuge_sources.iter().copied());
        node
    }

    fn remove_word_and_its_corresponding_sources(
        &self,
        word_id: i32,
        source_word_map: &mut BTreeMap<i32, SourceWordRecord>,
    ) {
        let keys_to_remove: Vec<i32> = source_word_map
            .iter()
            .filter_map(|(source_word_id, record)| {
                (record.word_id == word_id).then_some(*source_word_id)
            })
            .collect();

        for key in keys_to_remove {
            source_word_map.remove(&key);
        }
    }

    fn find_best_collided_node_index(
        &self,
        nodes: &[Node],
        sources_of_expected_word: &BTreeSet<i32>,
    ) -> Option<usize> {
        let mut best_index: Option<usize> = None;
        let mut best_score: f64 = f64::MIN;

        for (index, previous_node) in nodes.iter().enumerate() {
            let intersection_count = previous_node
                .source_ids
                .intersection(sources_of_expected_word)
                .count();

            // C# uses the current node source count as the denominator (not union count).
            let union_like_count = previous_node.source_ids.len();

            if union_like_count == 0 {
                continue;
            }

            let similarity = intersection_count as f64 / union_like_count as f64;

            if similarity >= self.similarity_threshold && similarity > best_score {
                best_score = similarity;
                best_index = Some(index);
            }
        }

        best_index
    }

    fn add_node_top_words(&self, node: &mut Node, sorted_scope: &[SourceWordRecord]) {
        if node.is_refuge_node() {
            return;
        }

        // Start with words already in the node
        for word_id in node.word_ids.iter().copied() {
            node.top_words.insert(word_id);
        }

        // Then fill up to 10 using sorted scope
        for record in sorted_scope {
            if node.top_words.len() >= 10 {
                break;
            }
            node.top_words.insert(record.word_id);
        }
    }

    fn update_word_number_of_sources(&self, records: &mut [SourceWordRecord]) {
        use std::collections::{BTreeSet, HashMap};

        let mut source_ids_by_word_id: HashMap<i32, BTreeSet<i32>> = HashMap::new();

        for record in records.iter() {
            source_ids_by_word_id
                .entry(record.word_id)
                .or_default()
                .insert(record.source_id);
        }

        for record in records.iter_mut() {
            record.word_number_of_sources = source_ids_by_word_id
                .get(&record.word_id)
                .map(|source_ids| source_ids.len())
                .unwrap_or(0);
        }
    }
}

#[derive(Debug, Default, Clone)]
struct BuilderState {
    hkts_by_id: BTreeMap<i32, Hkt>,
    nodes_by_id: BTreeMap<i32, Node>,
    /// Stores the source-word scope used to construct each HKT for later branch expansion.
    hkt_input_scopes: BTreeMap<i32, Vec<SourceWordRecord>>,
}

impl BuilderState {
    fn next_hkt_id(&self) -> i32 {
        self.hkts_by_id.len() as i32 + 1
    }

    fn next_node_id(&self) -> i32 {
        self.nodes_by_id.len() as i32 + 1
    }
}

fn records_to_map_by_source_word_id(
    records: Vec<SourceWordRecord>,
) -> BTreeMap<i32, SourceWordRecord> {
    let mut map = BTreeMap::new();
    for record in records {
        map.insert(record.source_word_id, record);
    }
    map
}

fn sort_source_word_records_desc(records: &mut [SourceWordRecord]) {
    // C# sorts descending by wordNoOfSources. We add stable tie-breakers for determinism.
    records.sort_by(|left_record, right_record| {
        right_record
            .word_number_of_sources
            .cmp(&left_record.word_number_of_sources)
            .then_with(|| left_record.word_id.cmp(&right_record.word_id))
            .then_with(|| left_record.source_id.cmp(&right_record.source_id))
            .then_with(|| left_record.source_word_id.cmp(&right_record.source_word_id))
    });
}

fn first_entry(source_word_map: &BTreeMap<i32, SourceWordRecord>) -> Option<&SourceWordRecord> {
    source_word_map.values().next()
}

fn source_ids_for_word(
    source_word_map: &BTreeMap<i32, SourceWordRecord>,
    word_id: i32,
) -> BTreeSet<i32> {
    source_word_map
        .values()
        .filter_map(|record| (record.word_id == word_id).then_some(record.source_id))
        .collect()
}

fn find_refugee_sources(
    source_word_map: &BTreeMap<i32, SourceWordRecord>,
    nodes: &[Node],
) -> BTreeSet<i32> {
    let all_remaining_sources: BTreeSet<i32> = source_word_map
        .values()
        .map(|record| record.source_id)
        .collect();

    let node_sources: BTreeSet<i32> = nodes
        .iter()
        .flat_map(|node| node.source_ids.iter().copied())
        .collect();

    all_remaining_sources
        .difference(&node_sources)
        .copied()
        .collect()
}
