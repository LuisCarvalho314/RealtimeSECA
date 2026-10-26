use super::*;

type WordSources = BTreeMap<i32, Sources>;
type Counts = BTreeMap<i32, usize>;

pub(super) struct ScopeUpdate {
    pub mapped: Vec<Sources>,
    pub additions: Vec<WordSources>,
    pub expected: BTreeSet<i32>,
    pub errors: [f64; 3],
}

pub(super) fn evaluate(
    scope: &Scope,
    incoming: &[Source],
    parent: Option<&Sources>,
    ancestor: &BTreeSet<i32>,
    retained: &BTreeMap<i64, BTreeSet<i32>>,
    config: &Input,
) -> ScopeUpdate {
    let incoming_words: BTreeMap<i64, BTreeSet<i32>> = incoming
        .iter()
        .map(|s| (s.id, s.words.iter().copied().collect()))
        .collect();
    let added_df = frequencies(incoming_words.values());
    let prior_df = frequencies(
        retained
            .iter()
            .filter(|(id, _)| parent.is_none_or(|p| p.contains(id)))
            .map(|(_, w)| w),
    );
    let current: BTreeSet<i32> = scope
        .nodes
        .iter()
        .flat_map(|n| &n.words)
        .copied()
        .filter(|w| *w != -1)
        .collect();
    let mapped = map_sources(scope, incoming);
    let additions: Vec<WordSources> = mapped
        .iter()
        .map(|ids| {
            let mut words = WordSources::new();
            for id in ids {
                for word in &incoming_words[id] {
                    words.entry(*word).or_default().insert(*id);
                }
            }
            words
        })
        .collect();
    let expected = admit_candidates(
        scope,
        ancestor,
        retained,
        &prior_df,
        &added_df,
        &current,
        &additions,
        config.alpha,
    );
    let measurements = measure_words(
        scope,
        &mapped,
        &additions,
        &expected,
        &prior_df,
        &added_df,
        config.beta,
    );
    let errors = reduce_errors(&current, &measurements, config);
    ScopeUpdate {
        mapped,
        additions,
        expected,
        errors,
    }
}

fn map_sources(scope: &Scope, incoming: &[Source]) -> Vec<Sources> {
    let mut mapped = vec![Sources::new(); scope.nodes.len()];
    for source in incoming {
        let mut matched = false;
        for (i, node) in scope
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.words.contains(&-1))
        {
            if source.words.iter().any(|w| node.words.contains(w)) {
                mapped[i].insert(source.id);
                matched = true;
            }
        }
        if !matched {
            if let Some(i) = scope.nodes.iter().position(|n| n.words.contains(&-1)) {
                mapped[i].insert(source.id);
            }
        }
    }
    mapped
}

#[allow(clippy::too_many_arguments)]
fn admit_candidates(
    scope: &Scope,
    ancestor: &BTreeSet<i32>,
    retained: &BTreeMap<i64, BTreeSet<i32>>,
    prior_df: &Counts,
    added_df: &Counts,
    current: &BTreeSet<i32>,
    additions: &[WordSources],
    alpha: f64,
) -> BTreeSet<i32> {
    let count = |word: &i32| total(word, prior_df, added_df);
    let prominent = prior_df
        .keys()
        .chain(added_df.keys())
        .filter(|w| !ancestor.contains(w))
        .map(count)
        .max()
        .unwrap_or(0)
        .max(1);
    let mut expected = scope.expected_words.clone();
    for word in added_df.keys() {
        if !current.contains(word) && count(word) as f64 / prominent as f64 >= alpha {
            expected.insert(*word);
        }
    }
    expected.retain(|w| !current.contains(w) && !ancestor.contains(w));
    let mut prominent = expected.iter().map(count).max().unwrap_or(0).max(1);
    for (i, node) in scope
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| !n.words.contains(&-1))
    {
        for word in &node.words {
            let old = node
                .sources
                .iter()
                .filter(|id| retained.get(id).is_some_and(|ws| ws.contains(word)))
                .count();
            prominent = prominent.max(old + additions[i].get(word).map_or(0, Sources::len));
        }
    }
    expected.retain(|w| count(w) as f64 / prominent as f64 >= alpha);
    expected
}

fn measure_words(
    scope: &Scope,
    mapped: &[Sources],
    additions: &[WordSources],
    expected: &BTreeSet<i32>,
    prior_df: &Counts,
    added_df: &Counts,
    beta: f64,
) -> BTreeMap<i32, [f64; 3]> {
    let mut counts = Vec::new();
    for (i, node) in scope
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| !n.words.contains(&-1))
    {
        for word in &node.words {
            counts.push(
                node.word_sources.get(word).map_or(0, Sources::len)
                    + additions[i].get(word).map_or(0, Sources::len),
            );
        }
    }
    counts.extend(expected.iter().map(|w| total(w, prior_df, added_df)));
    let denominator = counts.iter().sum::<usize>().max(1) as f64;
    let prominent = counts.iter().copied().max().unwrap_or(0).max(1) as f64;
    let mut measurements = BTreeMap::new();
    for (i, node) in scope
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| !n.words.contains(&-1))
    {
        let members: Sources = node.sources.union(&mapped[i]).copied().collect();
        let mut measured = node.words.clone();
        for word in expected {
            let n = additions[i].get(word).map_or(0, Sources::len);
            if n > 0 && n as f64 / members.len().max(1) as f64 >= beta {
                measured.insert(*word);
            }
        }
        for word in measured {
            let mut words = node.word_sources.get(&word).cloned().unwrap_or_default();
            if let Some(added) = additions[i].get(&word) {
                words.extend(added);
            }
            measurements.insert(
                word,
                [
                    words.len() as f64 / prominent,
                    words.intersection(&members).count() as f64 / members.len().max(1) as f64,
                    words.len() as f64 / denominator,
                ],
            );
        }
    }
    for word in expected {
        measurements.entry(*word).or_insert([
            total(word, prior_df, added_df) as f64 / prominent,
            0.0,
            total(word, prior_df, added_df) as f64 / denominator,
        ]);
    }
    measurements
}

fn reduce_errors(
    current: &BTreeSet<i32>,
    measurements: &BTreeMap<i32, [f64; 3]>,
    config: &Input,
) -> [f64; 3] {
    let mut errors = [0.0; 3];
    for word in current {
        let m = measurements.get(word).copied().unwrap_or([0.0; 3]);
        errors[0] += (config.alpha - m[0]).max(0.0);
        errors[1] += (config.beta - m[1]).max(0.0);
        errors[2] += m[2];
    }
    errors[0] /= current.len().max(1) as f64;
    errors[1] /= current.len().max(1) as f64;
    errors[2] = (1.0 - errors[2]).clamp(0.0, 1.0);
    errors
}

fn total(word: &i32, prior: &Counts, added: &Counts) -> usize {
    prior.get(word).copied().unwrap_or(0) + added.get(word).copied().unwrap_or(0)
}
fn frequencies<'a>(sources: impl Iterator<Item = &'a BTreeSet<i32>>) -> Counts {
    let mut result = Counts::new();
    for words in sources {
        for w in words {
            *result.entry(*w).or_default() += 1;
        }
    }
    result
}
