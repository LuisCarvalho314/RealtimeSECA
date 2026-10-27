use realtime_seca_v2::{metrics::Errors, Config, Model, Source, Words};
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn config() -> Config {
    Config {
        alpha: 0.7,
        beta: 0.5,
        minimum_branch_sources: 99,
        error_limits: Errors {
            alpha: 1.0,
            beta: 1.0,
            importance: 1.0,
        },
        window: None,
    }
}

fn news_g_child_sources() -> Vec<Source> {
    let mut documents: BTreeMap<String, Words> = BTreeMap::new();
    for line in include_str!("fixtures/news_g_friday_child.tsv").lines() {
        let mut fields = line.split('\t');
        let source_id = fields.next().unwrap().to_owned();
        let _original_csharp_word_id = fields.next().unwrap();
        let token = fields.next().unwrap().to_owned();
        assert!(fields.next().is_none());
        documents.entry(source_id).or_default().insert(token);
    }
    assert_eq!(documents.len(), 18);
    documents
        .into_iter()
        .map(|(id, words)| Source { id, words })
        .collect()
}

fn root_nodes(model: &Model) -> Vec<Value> {
    model
        .root()
        .nodes
        .iter()
        .map(|node| json!({ "words": node.words, "sources": node.sources }))
        .collect()
}

#[test]
fn news_g_friday_equal_frequency_order_is_deterministic() {
    let sources = news_g_child_sources();
    let mut first = Model::new(config()).unwrap();
    first.update(0, sources.clone()).unwrap();

    // All five eligible words have document frequency four. Lexical UTF-8
    // ordering makes cyber precede deal, then the later sale candidate fails
    // beta against their expanded source union.
    let expected = vec![
        json!({"words": ["cyber", "deal", "monday"], "sources": ["14", "33", "553", "583", "598", "656"]}),
        json!({"words": ["sale"], "sources": ["14", "33", "482", "648"]}),
        json!({"words": ["thanksgiving"], "sources": ["193", "216", "482", "583"]}),
        json!({"words": [], "sources": ["105", "248", "249", "323", "512", "59", "595", "711"]}),
    ];
    assert_eq!(root_nodes(&first), expected);

    let mut reversed = Model::new(config()).unwrap();
    reversed
        .update(0, sources.into_iter().rev().collect())
        .unwrap();
    assert_eq!(root_nodes(&reversed), expected);
}

#[test]
fn equal_frequency_tie_uses_utf8_bytes() {
    let lower_utf8 = "\u{e000}";
    let higher_utf8 = "\u{10000}";
    assert!(lower_utf8.as_bytes() < higher_utf8.as_bytes());

    let mut model = Model::new(config()).unwrap();
    model
        .update(
            0,
            vec![
                Source {
                    id: "2".into(),
                    words: [higher_utf8.into()].into(),
                },
                Source {
                    id: "1".into(),
                    words: [lower_utf8.into()].into(),
                },
            ],
        )
        .unwrap();
    assert_eq!(model.root().nodes[0].words, [lower_utf8.into()].into());
    assert_eq!(model.root().nodes[1].words, [higher_utf8.into()].into());
}
