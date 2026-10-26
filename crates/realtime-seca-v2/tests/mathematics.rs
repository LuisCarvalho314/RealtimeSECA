use realtime_seca_v2::{
    metrics::{self, Errors, Frequencies},
    Config, Hkt, Model, Node, Source, Sources, Words,
};
fn words(s: &str) -> Words {
    s.split_whitespace().map(str::to_owned).collect()
}
fn source(id: usize, s: &str) -> Source {
    Source {
        id: id.to_string(),
        words: words(s),
    }
}
fn sources(ids: &[usize]) -> Sources {
    ids.iter().map(ToString::to_string).collect()
}
fn config() -> Config {
    Config {
        error_limits: Errors {
            alpha: 1.,
            beta: 1.,
            importance: 1.,
        },
        minimum_branch_sources: 99,
        ..Config::default()
    }
}
fn near(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}
fn node(w: &str, ids: &[usize]) -> Node {
    Node {
        words: words(w),
        sources: sources(ids),
        child: None,
    }
}

#[test]
fn unique_df_shared_words_and_worst_beta() {
    let h = Hkt {
        nodes: vec![node("a x", &[1, 2, 3, 4, 5, 6]), node("b x", &[4, 5, 6, 7])],
    };
    let df: Frequencies = [
        ("a".into(), sources(&[1, 2, 3, 4, 5, 6])),
        ("b".into(), sources(&[4, 5, 6, 7])),
        ("x".into(), sources(&[1, 4, 7])),
    ]
    .into();
    let e = metrics::errors(&h, &df, &Words::new(), &Words::new(), 0.45, 0.6);
    near(e.alpha, 0.);
    near(e.beta, 4. / 45.);
    near(e.importance, 0.);
    let mut reversed = h;
    reversed.nodes.reverse();
    assert_eq!(
        e,
        metrics::errors(&reversed, &df, &Words::new(), &Words::new(), 0.45, 0.6)
    );
    near(
        metrics::eligibility(&df["x"], &sources(&[1, 2, 3, 4, 5, 6])),
        1. / 3.,
    );
}

#[test]
fn independently_calculated_three_errors_and_boundary() {
    let h = Hkt {
        nodes: vec![node("a b", &[1, 2, 3, 4])],
    };
    let df: Frequencies = [
        ("a".into(), sources(&[1, 2, 3, 4])),
        ("b".into(), sources(&[1])),
        ("x".into(), sources(&[2, 3, 4])),
    ]
    .into();
    let c = metrics::candidates(&df, &words("a b"), &Words::new(), 0.7);
    assert_eq!(c, words("x"));
    let e = metrics::errors(&h, &df, &c, &Words::new(), 0.7, 0.5);
    near(e.alpha, 0.225);
    near(e.beta, 0.125);
    near(e.importance, 3. / 8.);
    let empty = metrics::errors(
        &Hkt::default(),
        &Frequencies::new(),
        &Words::new(),
        &Words::new(),
        0.7,
        0.5,
    );
    assert_eq!(empty, Errors::default());
    near(metrics::ratio(0, 0), 0.);
}

#[test]
fn accepted_candidate_remains_provisional() {
    let mut m = Model::new(Config {
        alpha: 0.6,
        beta: 0.5,
        ..config()
    })
    .unwrap();
    m.update(
        0,
        vec![
            source(1, "a x"),
            source(2, "a x"),
            source(3, "a"),
            source(4, "a"),
        ],
    )
    .unwrap();
    assert_eq!(m.root().nodes[0].words, words("a"));
    let d = m
        .update(1, vec![source(5, "a x"), source(6, "a x")])
        .unwrap();
    near(d[0].errors.importance, 0.4);
    assert_eq!(m.root().nodes[0].words, words("a"));
    near(
        m.update(2, vec![source(7, "a x")]).unwrap()[0]
            .errors
            .importance,
        5.0 / 12.0,
    );
}

#[test]
fn expectations_recomputed_without_becoming_old_words() {
    let mut m = Model::new(Config {
        alpha: 0.6,
        beta: 0.9,
        ..config()
    })
    .unwrap();
    m.update(0, vec![source(1, "a"), source(2, "a")]).unwrap();
    for (batch, expected) in [(1, 2. / 5.), (2, 0.5), (3, 6. / 11.)] {
        let d = m
            .update(
                batch,
                vec![
                    source(batch as usize * 2 + 1, "a x"),
                    source(batch as usize * 2 + 2, "x"),
                ],
            )
            .unwrap();
        near(d[0].errors.importance, expected);
        assert_eq!(m.root().nodes[0].words, words("a"));
    }
}

#[test]
fn frequency_order_collision_ties_and_permutation() {
    let stream = vec![
        source(1, "a"),
        source(2, "b"),
        source(3, "a c"),
        source(4, "b c"),
    ];
    let c = Config {
        alpha: 1.,
        beta: 0.5,
        ..config()
    };
    let mut a = Model::new(c.clone()).unwrap();
    let mut b = Model::new(c).unwrap();
    a.update(0, stream.clone()).unwrap();
    b.update(0, stream.into_iter().rev().collect()).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.root().nodes[0].words, words("a c"));
    assert_eq!(a.root().nodes[0].sources, sources(&[1, 3, 4]));
    let mut m = Model::new(Config {
        alpha: 0.5,
        beta: 1.,
        ..config()
    })
    .unwrap();
    m.update(
        0,
        vec![
            source(1, "a b"),
            source(2, "a b"),
            source(3, "b"),
            source(4, "z"),
            source(5, "z"),
            source(6, "z"),
            source(7, "z"),
        ],
    )
    .unwrap();
    assert_eq!(
        m.root()
            .nodes
            .iter()
            .map(|n| n.words.clone())
            .collect::<Vec<_>>(),
        vec![words("z"), words("b"), words("a")]
    );
}

#[test]
fn residual_branch_boundary_and_refuge() {
    for (minimum, child) in [(1, true), (2, false)] {
        let mut m = Model::new(Config {
            minimum_branch_sources: minimum,
            ..config()
        })
        .unwrap();
        m.update(
            0,
            vec![
                source(1, "a b"),
                source(2, "a"),
                source(3, "a"),
                source(4, ""),
            ],
        )
        .unwrap();
        assert_eq!(m.root().nodes[0].child.is_some(), child);
        assert_eq!(m.root().nodes.last().unwrap().sources, sources(&[4]));
    }
}

#[test]
fn level_exclusion_prevents_reciprocal_descendants() {
    let mut m = Model::new(Config {
        alpha: 0.8,
        beta: 0.75,
        minimum_branch_sources: 1,
        ..config()
    })
    .unwrap();
    m.update(
        0,
        ["a b", "a b", "a b c", "a b c", "a c", "a c", "a"]
            .iter()
            .enumerate()
            .map(|(i, w)| source(i + 1, w))
            .collect(),
    )
    .unwrap();
    let child = m.root().nodes[0].child.as_ref().unwrap();
    assert_eq!(child.nodes.len(), 2);
    assert_eq!(child.nodes[0].words, words("b"));
    assert_eq!(child.nodes[1].words, words("c"));
    assert_eq!(child.nodes[0].sources, sources(&[1, 2, 3, 4]));
    assert_eq!(child.nodes[1].sources, sources(&[3, 4, 5, 6]));
    assert!(child.nodes.iter().all(|n| n.child.is_none()));
}

#[test]
fn selective_replacement_preserves_sibling_and_suppresses_old_children() {
    let mut c = config();
    c.minimum_branch_sources = 1;
    c.alpha = 0.8;
    c.beta = 0.75;
    c.error_limits.importance = 0.2;
    let mut m = Model::new(c).unwrap();
    let mut baseline: Vec<_> = ["a b", "a b", "a b c", "a b c", "a c", "a c", "a"]
        .iter()
        .enumerate()
        .map(|(i, w)| source(i + 1, w))
        .collect();
    baseline.extend((8..15).map(|i| source(i, "d")));
    m.update(0, baseline).unwrap();
    let sibling = m
        .root()
        .nodes
        .iter()
        .find(|n| n.words.contains("d"))
        .unwrap()
        .clone();
    let mut incoming: Vec<_> = (15..19).map(|i| source(i, "a x")).collect();
    incoming.extend((19..26).map(|i| source(i, "a")));
    let d = m.update(1, incoming).unwrap();
    assert!(!d[0].reconstructed);
    assert_eq!(d.iter().filter(|d| d.reconstructed).count(), 1);
    assert_eq!(d.iter().find(|d| d.reconstructed).unwrap().path.len(), 1);
    assert_eq!(
        *m.root()
            .nodes
            .iter()
            .find(|n| n.words.contains("d"))
            .unwrap(),
        sibling
    );
    assert!(d.iter().all(|d| d.path.len() <= 1));
}

#[test]
fn forgetting_empty_updates_reappearance_and_restart() {
    let mut m = Model::new(Config {
        window: Some(2),
        ..config()
    })
    .unwrap();
    m.update(0, vec![source(1, "a"), source(1, "ignored")])
        .unwrap();
    m.update(1, vec![source(2, "a")]).unwrap();
    assert!(m.update(2, vec![]).unwrap().is_empty());
    assert_eq!(m.retained_sources(), sources(&[2]));
    assert_eq!(m.root().nodes[0].words, words("a"));
    let mut restored: Model = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
    let batch = vec![source(1, "a x")];
    assert_eq!(m.update(3, batch.clone()), restored.update(3, batch));
    assert_eq!(m, restored);
    assert_eq!(m.retained_sources(), sources(&[1]));
    assert_eq!(m.root().nodes[0].sources, sources(&[1]));
}

#[test]
fn invalid_input_and_restore_do_not_create_partial_state() {
    let mut m = Model::new(config()).unwrap();
    let original = m.clone();
    assert!(m.update(1, vec![]).is_err());
    assert_eq!(m, original);
    assert!(m
        .update(
            0,
            vec![
                source(1, "a"),
                Source {
                    id: "".into(),
                    words: words("b")
                }
            ]
        )
        .is_err());
    assert_eq!(m, original);
    m.update(0, vec![source(1, "a")]).unwrap();
    let mut json = serde_json::to_value(&m).unwrap();
    json["root"]["nodes"][0]["sources"] = serde_json::json!(["missing"]);
    assert!(serde_json::from_value::<Model>(json).is_err());
    assert!(Model::new(Config {
        alpha: f64::NAN,
        ..config()
    })
    .is_err());
}

#[test]
fn root_replacement_suppresses_deep_inspection_and_reuses_construction() {
    let c = Config {
        alpha: 0.7,
        beta: 0.8,
        minimum_branch_sources: 1,
        error_limits: Errors {
            alpha: 1.,
            beta: 1.,
            importance: 0.1,
        },
        window: None,
    };
    let baseline: Vec<_> = (1..=10)
        .map(|id| {
            source(
                id,
                if id <= 2 {
                    "a b c"
                } else if id <= 5 {
                    "a b"
                } else {
                    "a"
                },
            )
        })
        .collect();
    let mut m = Model::new(c.clone()).unwrap();
    m.update(0, baseline.clone()).unwrap();
    assert!(m.root().nodes[0].child.as_ref().unwrap().nodes[0]
        .child
        .is_some());
    let incoming: Vec<_> = (11..=30).map(|id| source(id, "x")).collect();
    let d = m.update(1, incoming.clone()).unwrap();
    assert_eq!(d.len(), 1);
    assert!(d[0].path.is_empty());
    assert!(d[0].reconstructed);
    let mut fresh = Model::new(c).unwrap();
    fresh
        .update(0, baseline.into_iter().chain(incoming).collect())
        .unwrap();
    assert_eq!(m.root(), fresh.root());
}

#[test]
fn threshold_equality_accepts_and_adjacent_lower_limit_reconstructs() {
    for (limit, trigger) in [(0.5, false), (0.5 - f64::EPSILON, true)] {
        let mut c = config();
        c.error_limits.importance = limit;
        let mut m = Model::new(c).unwrap();
        m.update(0, vec![source(1, "a")]).unwrap();
        let d = m.update(1, vec![source(2, "x")]).unwrap();
        near(d[0].errors.importance, 0.5);
        assert_eq!(d[0].reconstructed, trigger);
    }
}

#[test]
fn accepted_parent_preserves_child_without_promotion() {
    let mut m = Model::new(Config {
        alpha: 0.6,
        beta: 0.5,
        minimum_branch_sources: 1,
        ..config()
    })
    .unwrap();
    m.update(
        0,
        vec![
            source(1, "a x"),
            source(2, "a x"),
            source(3, "a"),
            source(4, "a"),
        ],
    )
    .unwrap();
    assert_eq!(
        m.root().nodes[0].child.as_ref().unwrap().nodes[0].words,
        words("x")
    );
    let decisions = m
        .update(1, vec![source(5, "a x"), source(6, "a x")])
        .unwrap();
    assert!(decisions.iter().all(|d| !d.reconstructed));
    assert_eq!(m.root().nodes[0].words, words("a"));
    let child = m.root().nodes[0].child.as_ref().unwrap();
    assert_eq!(child.nodes[0].words, words("x"));
    assert_eq!(child.nodes[0].sources, sources(&[1, 2, 5, 6]));
    let restored: Model = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
    assert_eq!(restored, m);
}

#[test]
fn trace_small_hierarchy() {
    let mut model = Model::new(Config::default()).unwrap();

    model
        .update(
            0,
            vec![
                source(1, "nuclear reactor"),
                source(2, "nuclear waste"),
                source(3, "nuclear reactor"),
                source(4, "robotics underwater"),
            ],
        )
        .unwrap();

    println!("BASELINE:");
    println!("{:#?}", model.root());

    let decisions = model.update(1, vec![source(5, "nuclear reactor")]).unwrap();

    println!("DECISIONS:");
    println!("{:#?}", decisions);

    println!("UPDATED TREE:");
    println!("{:#?}", model.root());

    assert!(decisions[0].reconstructed);
}

#[test]
fn accepted_candidates_remain_provisional() {
    let mut m = Model::new(Config {
        alpha: 0.6,
        beta: 0.5,
        minimum_branch_sources: 1,
        error_limits: Errors {
            alpha: 1.0,
            beta: 1.0,
            importance: 1.0,
        },
        window: None,
    })
    .unwrap();

    m.update(
        0,
        vec![
            source(1, "a x"),
            source(2, "a x"),
            source(3, "a"),
            source(4, "a"),
        ],
    )
    .unwrap();

    assert_eq!(m.root().nodes[0].words, words("a"));

    let d = m
        .update(1, vec![source(5, "a x"), source(6, "a x")])
        .unwrap();

    assert!(!d[0].reconstructed);
    near(d[0].errors.importance, 0.4);

    // x is eligible at the parent but is not promoted.
    assert_eq!(m.root().nodes[0].words, words("a"));

    // The child retains x and receives matching sources.
    let child = m.root().nodes[0].child.as_ref().unwrap();
    assert_eq!(child.nodes[0].words, words("x"));
    assert_eq!(child.nodes[0].sources, sources(&[1, 2, 5, 6]));

    // x remains a candidate at the parent next cycle.
    let d = m.update(2, vec![source(7, "a x")]).unwrap();
    near(d[0].errors.importance, 5.0 / 12.0);
}

#[test]
fn child_reconstruction_matches_fresh_construction() {
    let mut c = config();
    c.alpha = 0.8;
    c.beta = 0.75;
    c.minimum_branch_sources = 1;
    c.error_limits.importance = 0.2;

    // Two independent contexts: a and d.
    let mut baseline: Vec<_> = ["a b", "a b", "a b c", "a b c", "a c", "a c", "a"]
        .iter()
        .enumerate()
        .map(|(i, w)| source(i + 1, w))
        .collect();

    baseline.extend((8..15).map(|i| source(i, "d")));

    // New evidence introduces x under the a context.
    let mut incoming: Vec<_> = (15..19).map(|i| source(i, "a x")).collect();
    incoming.extend((19..26).map(|i| source(i, "a")));

    let all_sources: Vec<Source> = baseline.iter().chain(incoming.iter()).cloned().collect();

    // Approach 1: incremental SECA, triggering one child.
    let mut evolving = Model::new(c.clone()).unwrap();
    evolving.update(0, baseline).unwrap();

    let decisions = evolving.update(1, incoming).unwrap();

    let root = evolving.root();
    let a_index = root
        .nodes
        .iter()
        .position(|n| n.words.contains("a"))
        .unwrap();

    assert!(!decisions[0].reconstructed);
    assert_eq!(decisions.iter().filter(|d| d.reconstructed).count(), 1);
    assert!(decisions
        .iter()
        .any(|d| { d.reconstructed && d.path == vec![a_index] }));

    let a_node = &root.nodes[a_index];
    let actual_child = a_node.child.as_deref().unwrap();

    // A child cannot reuse descriptors from its containing HKT.
    let excluded: Words = root
        .nodes
        .iter()
        .flat_map(|n| n.words.iter().cloned())
        .collect();

    // Independently construct the effective child dataset.
    let child_sources: Vec<Source> = all_sources
        .into_iter()
        .filter(|s| a_node.sources.contains(&s.id))
        .filter_map(|mut s| {
            s.words.retain(|w| !excluded.contains(w));

            (!s.words.is_empty()).then_some(s)
        })
        .collect();

    assert_eq!(child_sources.len(), 10);

    // Approach 2: build a new model from exactly that scope.
    let mut fresh = Model::new(c).unwrap();
    fresh.update(0, child_sources).unwrap();

    // Both routes must produce the same entire subtree.
    assert_eq!(actual_child, fresh.root());

    // Independent expected topology, not just agreement
    // between two calls to the same construction code.
    assert_eq!(
        actual_child,
        &Hkt {
            nodes: vec![
                node("b", &[1, 2, 3, 4]),
                node("c", &[3, 4, 5, 6]),
                node("x", &[15, 16, 17, 18]),
            ],
        }
    );
}

#[test]
fn child_only_receives_sources_with_residual_words() {
    let mut c = config();
    c.alpha = 0.7;
    c.beta = 0.5;
    c.minimum_branch_sources = 1;

    let mut model = Model::new(c).unwrap();

    model
        .update(
            0,
            vec![
                source(1, "a b"),
                source(2, "a b"),
                source(3, "a"),
                source(4, "a"),
            ],
        )
        .unwrap();

    // Baseline construction.
    let parent = &model.root().nodes[0];
    assert_eq!(parent.sources, sources(&[1, 2, 3, 4]));

    let child = parent.child.as_ref().unwrap();
    assert_eq!(child.nodes.len(), 1);
    assert_eq!(child.nodes[0].words, words("b"));
    assert_eq!(child.nodes[0].sources, sources(&[1, 2]));

    // Source 5 has no residual vocabulary.
    // Source 6 has residual vocabulary ("z"),
    // but does not match the existing child descriptor.
    model
        .update(1, vec![source(5, "a"), source(6, "a z")])
        .unwrap();

    let parent = &model.root().nodes[0];

    // Both sources match the parent descriptor "a".
    assert_eq!(parent.sources, sources(&[1, 2, 3, 4, 5, 6]));

    let child = parent.child.as_ref().unwrap();

    // Accepted updates do not create new child nodes.
    assert_eq!(child.nodes.len(), 1);
    assert_eq!(child.nodes[0].words, words("b"));
    assert_eq!(child.nodes[0].sources, sources(&[1, 2]));

    // Neither source enters a child node.
    assert!(child
        .nodes
        .iter()
        .all(|n| { !n.sources.contains("5") && !n.sources.contains("6") }));

    // Unmatched evidence remains available for future learning.
    assert!(model.retained_sources().contains("6"));

    // Serialization preserves the model.
    let json = serde_json::to_string(&model).unwrap();
    let restored: Model = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, model);
}

#[test]
fn unmatched_evidence_triggers_later_reconstruction() {
    let mut c = config();
    c.alpha = 0.7;
    c.beta = 0.5;
    c.minimum_branch_sources = 99;
    c.error_limits.importance = 0.4;

    let mut model = Model::new(c).unwrap();

    // Initial tree contains only a.
    model
        .update(0, vec![source(1, "a"), source(2, "a")])
        .unwrap();

    assert_eq!(
        model.root(),
        &Hkt {
            nodes: vec![node("a", &[1, 2])],
        }
    );

    // x appears once. It is below alpha and must
    // not create a new refuge or descriptor.
    let decisions = model.update(1, vec![source(3, "x")]).unwrap();

    assert_eq!(decisions.len(), 1);
    assert!(!decisions[0].reconstructed);
    near(decisions[0].errors.importance, 0.0);

    // The learned tree remains unchanged.
    assert_eq!(
        model.root(),
        &Hkt {
            nodes: vec![node("a", &[1, 2])],
        }
    );

    // But the unmatched source is retained.
    assert_eq!(model.retained_sources(), sources(&[1, 2, 3]));

    // A model with unmapped evidence must also
    // survive serialization and restoration.
    let json = serde_json::to_string(&model).unwrap();
    let restored: Model = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, model);

    // A second x arrives.
    // Now D(a)=2 and D(x)=2.
    // WI error = 2/(2+2) = 0.5 > 0.4.
    let decisions = model.update(2, vec![source(4, "x")]).unwrap();

    assert_eq!(decisions.len(), 1);
    assert!(decisions[0].reconstructed);
    near(decisions[0].errors.importance, 0.5);

    // Reconstruction must include BOTH x sources,
    // including source 3 which was previously unmapped.
    assert_eq!(
        model.root(),
        &Hkt {
            nodes: vec![node("a", &[1, 2]), node("x", &[3, 4]),],
        }
    );
}
