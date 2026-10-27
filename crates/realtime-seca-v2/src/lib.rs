//! Independent recursive SECA. Scientific contract: docs/v2/specification.md.
pub mod metrics;
mod recursive;

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub type Words = BTreeSet<String>;
pub type Sources = BTreeSet<String>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub alpha: f64,
    pub beta: f64,
    pub error_limits: metrics::Errors,
    /// Minimum sources containing residual words, inclusive.
    pub minimum_branch_sources: usize,
    /// None is full SECA; Some(gamma) is SECA-Light.
    pub window: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            alpha: 0.7,
            beta: 0.5,
            error_limits: metrics::Errors {
                alpha: 0.1,
                beta: 0.2,
                importance: 0.3,
            },
            minimum_branch_sources: 2,
            window: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Hkt {
    pub nodes: Vec<Node>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// Empty for refuge; no synthetic token is needed.
    pub words: Words,
    pub sources: Sources,
    pub child: Option<Box<Hkt>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Source {
    pub id: String,
    pub words: Words,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Evidence {
    words: Words,
    batch: u64,
}

fn residual_scope(
    parent: &Sources,
    excluded: &Words,
    evidence: &BTreeMap<String, Evidence>,
) -> Sources {
    parent
        .iter()
        .filter(|id| {
            evidence
                .get(*id)
                .is_some_and(|source| source.words.iter().any(|w| !excluded.contains(w)))
        })
        .cloned()
        .collect()
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Model {
    config: Config,
    root: Hkt,
    retained: BTreeMap<String, Evidence>,
    next_batch: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    /// Node indices from the root; empty means root HKT.
    pub path: Vec<usize>,
    pub errors: metrics::Errors,
    pub reconstructed: bool,
}

impl Model {
    pub fn new(config: Config) -> Result<Self, String> {
        if [
            config.alpha,
            config.beta,
            config.error_limits.alpha,
            config.error_limits.beta,
            config.error_limits.importance,
        ]
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || config.window == Some(0)
            || config.minimum_branch_sources == 0
        {
            return Err(
                "parameters must be finite probabilities; window and branch minimum positive"
                    .into(),
            );
        }
        Ok(Self {
            config,
            root: Hkt::default(),
            retained: BTreeMap::new(),
            next_batch: 0,
        })
    }

    pub fn root(&self) -> &Hkt {
        &self.root
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn next_batch(&self) -> u64 {
        self.next_batch
    }
    pub fn retained_sources(&self) -> Sources {
        self.retained.keys().cloned().collect()
    }

    /// Validates the complete input before mutation. First identity wins.
    pub fn update(&mut self, batch: u64, sources: Vec<Source>) -> Result<Vec<Decision>, String> {
        if batch != self.next_batch || batch == u64::MAX {
            return Err("batches must be consecutive from zero and leave a successor".into());
        }
        if sources
            .iter()
            .any(|s| s.id.trim().is_empty() || s.words.iter().any(|w| w.trim().is_empty()))
        {
            return Err("source identities and tokens must be nonempty".into());
        }
        let mut incoming = Sources::new();
        for source in sources {
            if let std::collections::btree_map::Entry::Vacant(entry) =
                self.retained.entry(source.id)
            {
                incoming.insert(entry.key().clone());
                entry.insert(Evidence {
                    words: source.words,
                    batch,
                });
            }
        }
        let scope = self.retained_sources();
        let mut decisions = Vec::new();
        if batch == 0 {
            self.root = recursive::construct(&scope, &Words::new(), &self.retained, &self.config);
        } else {
            recursive::evolve(
                &mut self.root,
                &scope,
                &incoming,
                &Words::new(),
                &self.retained,
                &self.config,
                &mut Vec::new(),
                &mut decisions,
            );
        }
        if let Some(gamma) = self.config.window {
            let cutoff = batch.saturating_sub(gamma - 1);
            self.retained.retain(|_, e| e.batch >= cutoff);
            let retained = self.retained_sources();
            recursive::forget(&mut self.root, &retained);
        }
        self.next_batch += 1;
        Ok(decisions)
    }
}

impl<'de> Deserialize<'de> for Model {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct State {
            config: Config,
            root: Hkt,
            retained: BTreeMap<String, Evidence>,
            next_batch: u64,
        }
        let state = State::deserialize(deserializer)?;
        let mut model = Self::new(state.config).map_err(serde::de::Error::custom)?;
        model.root = state.root;
        model.retained = state.retained;
        model.next_batch = state.next_batch;
        model.validate().map_err(serde::de::Error::custom)?;
        Ok(model)
    }
}

impl Model {
    fn validate(&self) -> Result<(), String> {
        let cutoff = self
            .config
            .window
            .map(|g| self.next_batch.saturating_sub(g));
        if self.retained.iter().any(|(id, e)| {
            id.trim().is_empty()
                || e.batch >= self.next_batch
                || cutoff.is_some_and(|c| e.batch < c)
                || e.words.iter().any(|w| w.trim().is_empty())
        }) {
            return Err("invalid retained evidence".into());
        }
        fn tree(
            h: &Hkt,
            scope: &Sources,
            excluded: &Words,
            evidence: &BTreeMap<String, Evidence>,
        ) -> bool {
            h.nodes.iter().filter(|n| n.words.is_empty()).count() <= 1
                && h.nodes.iter().all(|n| {
                    n.sources.is_subset(scope)
                        && n.words.is_disjoint(excluded)
                        && n.words.iter().all(|w| !w.trim().is_empty())
                        && n.child.as_ref().is_none_or(|child| {
                            let mut child_excluded = excluded.clone();
                            child_excluded.extend(n.words.iter().cloned());
                            let child_scope = residual_scope(&n.sources, &child_excluded, evidence);
                            tree(child, &child_scope, &child_excluded, evidence)
                        })
                })
        }
        if !tree(
            &self.root,
            &self.retained_sources(),
            &Words::new(),
            &self.retained,
        ) || (self.next_batch == 0 && !self.root.nodes.is_empty())
        {
            return Err("invalid hierarchy memberships".into());
        }

        Ok(())
    }
}
