use crate::residual_scope;
use crate::{
    metrics::{self, Frequencies},
    Config, Decision, Evidence, Hkt, Node, Sources, Words,
};
use std::collections::BTreeMap;

type Store = BTreeMap<String, Evidence>;

fn frequencies(scope: &Sources, evidence: &Store) -> Frequencies {
    let mut df = Frequencies::new();
    for id in scope {
        for word in &evidence[id].words {
            df.entry(word.clone()).or_default().insert(id.clone());
        }
    }
    df
}

pub(crate) fn construct(
    scope: &Sources,
    excluded: &Words,
    evidence: &Store,
    config: &Config,
) -> Hkt {
    let df = frequencies(scope, evidence);
    let admitted = metrics::candidates(&df, &Words::new(), excluded, config.alpha);
    let mut ranked: Vec<_> = admitted.into_iter().collect();
    // The papers specify frequency ranking but no equal-frequency order. Use
    // exact UTF-8 token bytes as the portable tie key at every HKT scope.
    ranked.sort_by(|a, b| {
        df[b]
            .len()
            .cmp(&df[a].len())
            .then_with(|| a.as_bytes().cmp(b.as_bytes()))
    });
    let mut hkt = Hkt::default();
    for word in ranked {
        let sources = &df[&word];
        let mut best: Option<(usize, usize, usize)> = None;
        for (i, node) in hkt.nodes.iter().enumerate() {
            let overlap = sources.intersection(&node.sources).count();
            let total = node.sources.len();
            if metrics::ratio(overlap, total) >= config.beta
                && best.is_none_or(|(_, n, d)| {
                    (overlap as u128) * (d as u128) > (n as u128) * (total as u128)
                })
            {
                best = Some((i, overlap, total));
            }
        }
        if let Some((index, _, _)) = best {
            hkt.nodes[index].words.insert(word);
            hkt.nodes[index].sources.extend(sources.iter().cloned());
        } else {
            hkt.nodes.push(Node {
                words: [word].into(),
                sources: sources.clone(),
                child: None,
            });
        }
    }
    let adopted: Sources = hkt
        .nodes
        .iter()
        .flat_map(|n| n.sources.iter().cloned())
        .collect();
    let refuge: Sources = scope.difference(&adopted).cloned().collect();
    if !refuge.is_empty() {
        hkt.nodes.push(Node {
            sources: refuge,
            ..Node::default()
        });
    }
    for node in &mut hkt.nodes {
        // Only the current node and its actual ancestors are excluded.
        let mut child_excluded = excluded.clone();
        child_excluded.extend(node.words.iter().cloned());
        let child_scope = residual_scope(&node.sources, &child_excluded, evidence);

        if child_scope.len() >= config.minimum_branch_sources {
            node.child = Some(Box::new(construct(
                &child_scope,
                &child_excluded,
                evidence,
                config,
            )));
        }
    }
    hkt
}

fn map_incoming(hkt: &mut Hkt, incoming: &Sources, evidence: &Store) {
    let mut unmatched = Sources::new();
    for id in incoming {
        let mut matched = false;
        for node in hkt.nodes.iter_mut().filter(|n| !n.words.is_empty()) {
            if !node.words.is_disjoint(&evidence[id].words) {
                node.sources.insert(id.clone());
                matched = true;
            }
        }
        if !matched {
            unmatched.insert(id.clone());
        }
    }
    if let Some(refuge) = hkt.nodes.iter_mut().find(|n| n.words.is_empty()) {
        refuge.sources.extend(unmatched);
    }
}

// Direct recursive replacement; a trigger returns before visiting old children.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evolve(
    hkt: &mut Hkt,
    scope: &Sources,
    incoming: &Sources,
    excluded: &Words,
    evidence: &Store,
    config: &Config,
    path: &mut Vec<usize>,
    decisions: &mut Vec<Decision>,
) {
    if incoming.is_empty() {
        return;
    }
    map_incoming(hkt, incoming, evidence);
    let df = frequencies(scope, evidence);
    let candidates = metrics::candidates(&df, &metrics::vocabulary(hkt), excluded, config.alpha);
    let errors = metrics::errors(hkt, &df, &candidates, excluded, config.alpha, config.beta);
    let limits = config.error_limits;
    let reconstructed = errors.alpha > limits.alpha
        || errors.beta > limits.beta
        || errors.importance > limits.importance;
    decisions.push(Decision {
        path: path.clone(),
        errors,
        reconstructed,
    });
    if reconstructed {
        *hkt = construct(scope, excluded, evidence, config);
        return;
    }

    for (index, node) in hkt.nodes.iter_mut().enumerate() {
        if let Some(child) = &mut node.child {
            let mut child_excluded = excluded.clone();
            child_excluded.extend(node.words.iter().cloned());
            let child_scope = residual_scope(&node.sources, &child_excluded, evidence);

            let child_incoming: Sources = child_scope.intersection(incoming).cloned().collect();

            path.push(index);
            evolve(
                child,
                &child_scope,
                &child_incoming,
                &child_excluded,
                evidence,
                config,
                path,
                decisions,
            );
            path.pop();
        }
    }
}

pub(crate) fn forget(hkt: &mut Hkt, retained: &Sources) {
    for node in &mut hkt.nodes {
        node.sources.retain(|id| retained.contains(id));
        if let Some(child) = &mut node.child {
            forget(child, retained);
        }
    }
}
