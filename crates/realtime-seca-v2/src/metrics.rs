//! Pure source-set measurements; no retained indexes or descriptor mutation.
use crate::{Hkt, Sources, Words};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub type Frequencies = BTreeMap<String, Sources>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Errors {
    pub alpha: f64,
    pub beta: f64,
    pub importance: f64,
}

pub fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

pub fn vocabulary(hkt: &Hkt) -> Words {
    hkt.nodes
        .iter()
        .flat_map(|n| n.words.iter().cloned())
        .collect()
}

pub fn eligibility(word_sources: &Sources, node_sources: &Sources) -> f64 {
    ratio(
        word_sources.intersection(node_sources).count(),
        node_sources.len(),
    )
}

pub fn candidates(df: &Frequencies, old: &Words, excluded: &Words, alpha: f64) -> Words {
    let prominent = df
        .iter()
        .filter(|(w, _)| !excluded.contains(*w) || old.contains(*w))
        .map(|(_, s)| s.len())
        .max()
        .unwrap_or(0);
    df.iter()
        .filter(|(w, s)| {
            !old.contains(*w)
                && !excluded.contains(*w)
                && !s.is_empty()
                && ratio(s.len(), prominent) >= alpha
        })
        .map(|(w, _)| w.clone())
        .collect()
}

/// Old descriptors must still be present in hkt; memberships already include mapping.
pub fn errors(
    hkt: &Hkt,
    df: &Frequencies,
    new: &Words,
    excluded: &Words,
    alpha: f64,
    beta: f64,
) -> Errors {
    let old = vocabulary(hkt);
    let prominent = df
        .iter()
        .filter(|(w, _)| !excluded.contains(*w) || old.contains(*w))
        .map(|(_, s)| s.len())
        .max()
        .unwrap_or(0);
    let empty = Sources::new();
    let mut result = Errors::default();
    for word in &old {
        let sources = df.get(word).unwrap_or(&empty);
        result.alpha += (alpha - ratio(sources.len(), prominent)).max(0.0);
        result.beta += hkt
            .nodes
            .iter()
            .filter(|n| n.words.contains(word))
            .map(|n| (beta - eligibility(sources, &n.sources)).max(0.0))
            .fold(0.0, f64::max);
    }
    if !old.is_empty() {
        result.alpha /= old.len() as f64;
        result.beta /= old.len() as f64;
    }
    let count = |words: &Words| {
        words
            .iter()
            .map(|w| df.get(w).map_or(0, Sources::len))
            .sum::<usize>()
    };
    result.importance = ratio(count(new), count(&old) + count(new));
    result
}
