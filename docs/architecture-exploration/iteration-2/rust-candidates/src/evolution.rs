//! Independent normalized-input evolution experiment. Public token/identity adapters
//! and schema-4 validation are deliberately outside this research engine.
use super::*;
mod context;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Source {
    pub id: i64,
    pub words: Vec<i32>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Batch {
    pub index: u64,
    pub sources: Vec<Source>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StreamInput {
    pub construction: Input,
    pub initial_index: u64,
    pub batches: Vec<Batch>,
    pub gamma: Option<u64>,
    pub alpha_error_threshold: f64,
    pub beta_error_threshold: f64,
    pub importance_error_threshold: f64,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Decision {
    pub hkt_id: i32,
    pub paper_alpha_error: f64,
    pub paper_beta_error: f64,
    pub paper_word_importance_error: f64,
    pub should_reconstruct: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Update {
    pub decisions: Vec<Decision>,
    pub replacements: BTreeMap<i32, i32>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Engine {
    config: Input,
    pub tree: Option<Scope>,
    evidence: Vec<Batch>,
    vocabulary: BTreeSet<i32>,
    current: u64,
    gamma: Option<u64>,
    thresholds: [f64; 3],
    next_hkt: i32,
    next_node: i32,
}
struct Target {
    incoming: Sources,
    excluded: BTreeSet<i32>,
}

impl Engine {
    pub fn new(request: &StreamInput) -> Result<Self, String> {
        if request.gamma == Some(0)
            || [
                request.alpha_error_threshold,
                request.beta_error_threshold,
                request.importance_error_threshold,
            ]
            .iter()
            .any(|t| !(0.0..=1.0).contains(t))
        {
            return Err("invalid retention or error threshold".into());
        }
        let tree = build(&request.construction, false)?;
        let mut sources: Vec<Source> = Vec::new();
        for row in &request.construction.records {
            let index = sources
                .iter()
                .position(|s| s.id == row.source)
                .unwrap_or_else(|| {
                    sources.push(Source {
                        id: row.source,
                        words: Vec::new(),
                    });
                    sources.len() - 1
                });
            if !sources[index].words.contains(&row.word) {
                sources[index].words.push(row.word);
            }
        }
        let vocabulary = sources
            .iter()
            .flat_map(|s| s.words.iter().copied())
            .collect();
        let (hkt, node) = tree.as_ref().map_or((0, 0), max_ids);
        let mut config = request.construction.clone();
        config.records.clear();
        Ok(Self {
            config,
            tree,
            evidence: vec![Batch {
                index: request.initial_index,
                sources,
            }],
            vocabulary,
            current: request.initial_index,
            gamma: request.gamma,
            thresholds: [
                request.alpha_error_threshold,
                request.beta_error_threshold,
                request.importance_error_threshold,
            ],
            next_hkt: hkt,
            next_node: node,
        })
    }
    /// A candidate clone provides atomic rollback. It is retained until an alternative
    /// transaction protocol is independently shown to satisfy the same guarantee.
    pub fn process(&mut self, batch: &Batch) -> Result<Update, String> {
        let mut candidate = self.clone();
        let update = candidate.process_inner(batch)?;
        *self = candidate;
        Ok(update)
    }
    fn process_inner(&mut self, batch: &Batch) -> Result<Update, String> {
        if self.current.checked_add(1) != Some(batch.index)
            || batch
                .sources
                .iter()
                .any(|s| s.words.iter().any(|w| *w <= 0))
        {
            return Err("invalid sequence or normalized word ID".into());
        }
        let retained: BTreeMap<i64, BTreeSet<i32>> = self
            .evidence
            .iter()
            .flat_map(|b| &b.sources)
            .map(|s| (s.id, s.words.iter().copied().collect()))
            .collect();
        let mut seen: Sources = retained.keys().copied().collect();
        let incoming: Vec<Source> = batch
            .sources
            .iter()
            .filter(|s| seen.insert(s.id))
            .cloned()
            .collect();
        for source in &incoming {
            self.vocabulary.extend(&source.words);
        }
        let mut targets = BTreeMap::new();
        let mut decisions = Vec::new();
        if let Some(tree) = &mut self.tree {
            visit(
                tree,
                &incoming,
                None,
                &BTreeSet::new(),
                &retained,
                &self.config,
                self.thresholds,
                &mut decisions,
                &mut targets,
            );
        } else if !incoming.is_empty() {
            return Err("baseline has no root; empty-baseline evolution unsupported".into());
        }
        self.evidence.push(Batch {
            index: batch.index,
            sources: incoming,
        });
        let replacements = self.reconstruct(targets)?;
        self.current = batch.index;
        if let Some(gamma) = self.gamma {
            let cutoff = self.current.saturating_sub(gamma - 1);
            self.evidence.retain(|b| b.index >= cutoff);
            let live: Sources = self
                .evidence
                .iter()
                .flat_map(|b| &b.sources)
                .map(|s| s.id)
                .collect();
            if let Some(tree) = &mut self.tree {
                forget(tree, &live);
            }
        }
        Ok(Update {
            decisions,
            replacements,
        })
    }
    fn reconstruct(
        &mut self,
        targets: BTreeMap<i32, Target>,
    ) -> Result<BTreeMap<i32, i32>, String> {
        let mut replacements = BTreeMap::new();
        for (id, target) in targets {
            let scope = find_scope(self.tree.as_ref().unwrap(), id).ok_or("missing target")?;
            let parent = scope.parent_node_id;
            let members: Sources = scope
                .nodes
                .iter()
                .flat_map(|n| n.sources.iter().copied())
                .chain(target.incoming)
                .collect();
            let rows: Vec<Row> = self
                .evidence
                .iter()
                .flat_map(|b| &b.sources)
                .filter(|s| members.contains(&s.id))
                .flat_map(|s| {
                    s.words
                        .iter()
                        .filter(|w| !target.excluded.contains(w))
                        .map(|word| Row {
                            source: s.id,
                            word: *word,
                        })
                })
                .collect();
            if rows.is_empty() {
                continue;
            }
            let propagated: Sources = rows.iter().map(|r| r.source).collect();
            let mut ids = Ids {
                hkt: self.next_hkt,
                node: self.next_node,
            };
            let replacement = construct(&relation(&rows), parent, &self.config, false, &mut ids)
                .ok_or("nonempty reconstruction produced no root")?;
            replacements.insert(id, replacement.hkt_id);
            replace(self.tree.as_mut().unwrap(), id, replacement)?;
            if parent != 0 {
                propagate(self.tree.as_mut().unwrap(), parent, &propagated);
            }
            self.next_hkt = ids.hkt;
            self.next_node = ids.node;
        }
        Ok(replacements)
    }
}

fn max_ids(scope: &Scope) -> (i32, i32) {
    scope.nodes.iter().fold((scope.hkt_id, 0), |(h, n), node| {
        let (ch, cn) = node.child.as_deref().map_or((0, 0), max_ids);
        (h.max(ch), n.max(node.node_id).max(cn))
    })
}
#[allow(clippy::too_many_arguments)]
fn visit(
    scope: &mut Scope,
    incoming: &[Source],
    parent: Option<&Sources>,
    ancestor: &BTreeSet<i32>,
    retained: &BTreeMap<i64, BTreeSet<i32>>,
    config: &Input,
    thresholds: [f64; 3],
    decisions: &mut Vec<Decision>,
    targets: &mut BTreeMap<i32, Target>,
) {
    if incoming.is_empty() {
        return;
    }
    let context::ScopeUpdate {
        mapped,
        additions,
        expected,
        errors,
    } = context::evaluate(scope, incoming, parent, ancestor, retained, config);
    let trigger = errors.iter().zip(thresholds).any(|(e, t)| *e > t);
    decisions.push(Decision {
        hkt_id: scope.hkt_id,
        paper_alpha_error: errors[0],
        paper_beta_error: errors[1],
        paper_word_importance_error: errors[2],
        should_reconstruct: trigger,
    });
    if trigger {
        targets.insert(
            scope.hkt_id,
            Target {
                incoming: incoming.iter().map(|s| s.id).collect(),
                excluded: ancestor.clone(),
            },
        );
        return;
    }
    scope.expected_words = expected;
    let mut child_order: Vec<_> = scope
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| n.child.as_ref().map(|child| (child.hkt_id, i)))
        .collect();
    child_order.sort_unstable();
    for (_, i) in child_order {
        let node = &mut scope.nodes[i];
        if let Some(child) = node.child.as_deref_mut() {
            let child_incoming: Vec<_> = incoming
                .iter()
                .filter(|s| mapped[i].contains(&s.id))
                .cloned()
                .collect();
            let child_ancestor = ancestor
                .union(&node.words)
                .copied()
                .filter(|w| *w != -1)
                .collect();
            visit(
                child,
                &child_incoming,
                Some(&node.sources),
                &child_ancestor,
                retained,
                config,
                thresholds,
                decisions,
                targets,
            );
        }
    }
    for (i, node) in scope.nodes.iter_mut().enumerate() {
        node.sources.extend(&mapped[i]);
        for word in &node.words {
            if let Some(added) = additions[i].get(word) {
                node.word_sources.entry(*word).or_default().extend(added);
            }
        }
    }
}

fn find_scope(scope: &Scope, id: i32) -> Option<&Scope> {
    if scope.hkt_id == id {
        return Some(scope);
    }
    scope
        .nodes
        .iter()
        .filter_map(|n| n.child.as_deref())
        .find_map(|c| find_scope(c, id))
}
fn replace(scope: &mut Scope, id: i32, replacement: Scope) -> Result<(), String> {
    if scope.hkt_id == id {
        *scope = replacement;
        return Ok(());
    }
    for node in &mut scope.nodes {
        if let Some(child) = node.child.as_deref_mut() {
            if find_scope(child, id).is_some() {
                return replace(child, id, replacement);
            }
        }
    }
    Err("missing replacement attachment".into())
}
fn propagate(scope: &mut Scope, parent: i32, sources: &Sources) -> bool {
    for node in &mut scope.nodes {
        if node.node_id == parent
            || node
                .child
                .as_deref_mut()
                .is_some_and(|c| propagate(c, parent, sources))
        {
            node.sources.extend(sources);
            return true;
        }
    }
    false
}
fn forget(scope: &mut Scope, live: &Sources) {
    for node in &mut scope.nodes {
        node.sources.retain(|s| live.contains(s));
        node.word_sources.retain(|_, ss| {
            ss.retain(|s| live.contains(s));
            !ss.is_empty()
        });
        if let Some(child) = node.child.as_deref_mut() {
            forget(child, live);
        }
    }
}

pub fn run(request: &StreamInput) -> Result<serde_json::Value, String> {
    let mut engine = Engine::new(request)?;
    let mut states = vec![engine.tree.clone()];
    let mut updates = Vec::new();
    for batch in &request.batches {
        let before = serde_json::to_value(&engine).unwrap();
        match engine.process(batch) {
            Ok(update)=>updates.push(serde_json::json!({"ok":true,"decisions":update.decisions,"replacements":update.replacements})),
            Err(error)=>updates.push(serde_json::json!({"ok":false,"error":error,"unchanged":before==serde_json::to_value(&engine).unwrap()})),
        }
        states.push(engine.tree.clone());
        engine = serde_json::from_value(serde_json::to_value(&engine).unwrap()).unwrap();
    }
    Ok(serde_json::json!({"states":states,"updates":updates}))
}
