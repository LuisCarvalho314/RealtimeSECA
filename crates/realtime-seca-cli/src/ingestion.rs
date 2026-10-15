use realtime_seca_core::{SecaConfig, SourceBatch, SourceRecord};
use regex::Regex;
use std::{fs, path::Path};
use stop_words::{get as get_stop_words, Language};
pub(crate) fn run_from_csv_command(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if arguments.len() < 4 {
        eprintln!("Usage:");
        eprintln!(
            "  {} from-csv <input.csv> <output.json> [--batch-index <n>] [--min-tokens <n>]",
            arguments[0]
        );
        return Ok(());
    }

    let input_csv_path = arguments[2].clone();
    let output_json_path = arguments[3].clone();

    let mut batch_index: u32 = 0;
    let mut min_tokens: usize = 1;

    let mut index = 4;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--batch-index" => {
                let Some(value) = arguments.get(index + 1) else {
                    return Err("missing value for --batch-index".into());
                };
                batch_index = value.parse::<u32>()?;
                index += 2;
            }
            "--min-tokens" => {
                let Some(value) = arguments.get(index + 1) else {
                    return Err("missing value for --min-tokens".into());
                };
                min_tokens = value.parse::<usize>()?;
                index += 2;
            }
            unknown_flag => return Err(format!("unknown argument: {unknown_flag}").into()),
        }
    }

    let batch = convert_csv_to_source_batch(&input_csv_path, batch_index, min_tokens)?;
    let json = serde_json::to_string_pretty(&batch)?;
    fs::write(&output_json_path, json)?;
    println!(
        "Wrote {} sources to {}",
        batch.sources.len(),
        output_json_path
    );

    Ok(())
}

pub(crate) fn convert_csv_to_source_batch(
    input_csv_path: impl AsRef<Path>,
    batch_index: u32,
    min_tokens: usize,
) -> Result<SourceBatch, Box<dyn std::error::Error>> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(input_csv_path)?;

    let tokenizer = Regex::new(r"[A-Za-z0-9_]+")?;

    let stop_words: std::collections::BTreeSet<&'static str> =
        get_stop_words(Language::English).iter().copied().collect();

    let headers = reader.headers()?.clone();

    let mut sources: Vec<SourceRecord> = Vec::new();

    for result in reader.records() {
        let record = result?;

        let get = |column_name: &str| -> String {
            if let Some(position) = headers.iter().position(|header| header == column_name) {
                record.get(position).unwrap_or("").to_string()
            } else {
                String::new()
            }
        };

        let title = get("Title");
        let short_summary = get("ShortSummary");
        let description = get("Description");

        let text = normalize_whitespace(&format!("{title} {short_summary} {description}"));
        if text.is_empty() {
            continue;
        }

        let mut tokens: Vec<String> = tokenizer
            .find_iter(&text)
            .map(|m| m.as_str().to_lowercase())
            .filter(|token| !stop_words.contains(token.as_str()))
            .collect();

        // Optional: deduplicate consecutive duplicates
        // tokens.dedup();

        if tokens.len() < min_tokens {
            continue;
        }

        let timestamp_raw = get("Timestamp");
        let api_timestamp_raw = get("API_Timestamp");

        let mut metadata = serde_json::Map::new();
        for column_name in [
            "ArticleId",
            "ArticleID",
            "article_id",
            "URL",
            "Query",
            "NewsCategory",
            "AlertFlag",
            "Relevance",
            "Timestamp",
            "API_Timestamp",
        ] {
            let value = get(column_name);
            if !value.trim().is_empty() {
                metadata.insert(column_name.to_string(), serde_json::Value::String(value));
            }
        }

        let source = SourceRecord {
            source_id: csv_source_identity(&get, &title, &short_summary, &description),
            batch_index,
            tokens: std::mem::take(&mut tokens),
            text: Some(text),
            timestamp_unix_ms: parse_timestamp_to_unix_ms(&timestamp_raw)
                .or_else(|| parse_timestamp_to_unix_ms(&api_timestamp_raw)),
            metadata: if metadata.is_empty() {
                None
            } else {
                Some(serde_json::Value::Object(metadata))
            },
        };

        sources.push(source);
    }

    Ok(SourceBatch {
        batch_index,
        sources,
    })
}

// Length-delimited JSON avoids delimiter and hash collisions in anonymous identities.
fn csv_source_identity(
    get: &impl Fn(&str) -> String,
    title: &str,
    summary: &str,
    description: &str,
) -> String {
    for column in ["ArticleId", "ArticleID", "article_id"] {
        let id = get(column);
        if !id.trim().is_empty() {
            return format!("article:{}", id.trim());
        }
    }
    let raw_url = get("URL");
    if let Ok(mut url) = url::Url::parse(raw_url.trim()) {
        if matches!(url.scheme(), "http" | "https") && url.host_str().is_some() {
            url.set_fragment(None);
            return format!("url:{url}");
        }
    }
    let fields = [title, summary, description].map(normalize_whitespace);
    format!(
        "csv-content:{}",
        serde_json::to_string(&fields).expect("string serialization")
    )
}

pub(crate) fn normalize_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) fn parse_timestamp_to_unix_ms(value: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(value.trim())
        .ok()
        .map(|timestamp| timestamp.timestamp_millis())
}

pub(crate) fn read_source_batch_json(
    path: impl AsRef<Path>,
) -> Result<SourceBatch, Box<dyn std::error::Error>> {
    let file_contents = fs::read_to_string(path)?;
    let batch: SourceBatch = serde_json::from_str(&file_contents)?;
    Ok(batch)
}

pub(crate) fn read_config_json(
    path: impl AsRef<Path>,
) -> Result<SecaConfig, Box<dyn std::error::Error>> {
    let file_contents = fs::read_to_string(path)?;
    let config: SecaConfig = serde_json::from_str(&file_contents)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convert(csv: &str, index: u32) -> SourceBatch {
        let path =
            std::env::temp_dir().join(format!("seca-ingestion-{}-{index}.csv", std::process::id()));
        fs::write(&path, csv).unwrap();
        let batch = convert_csv_to_source_batch(&path, index, 1).unwrap();
        fs::remove_file(path).unwrap();
        batch
    }

    #[test]
    fn identities_survive_files_reordering_and_url_aliases() {
        let a = convert("Title,URL\nmarket news,HTTPS://Example.org:443/a#section\nother news,https://example.org/b\n", 0);
        let b = convert(
            "Title,URL\nother news,https://example.org/b\nmarket news,https://example.org/a\n",
            1,
        );
        assert_ne!(a.sources[0].source_id, b.sources[0].source_id);
        assert_eq!(a.sources[0].source_id, b.sources[1].source_id);
        assert_eq!(a.sources[1].source_id, b.sources[0].source_id);
        assert_eq!(
            a.sources[0].metadata.as_ref().unwrap()["URL"],
            "HTTPS://Example.org:443/a#section"
        );
    }

    #[test]
    fn url_normalization_preserves_meaningful_distinctions() {
        let identity = |url: &str| {
            csv_source_identity(
                &|column| {
                    if column == "URL" {
                        url.to_string()
                    } else {
                        String::new()
                    }
                },
                "news",
                "",
                "",
            )
        };
        assert_eq!(
            identity(" https://EXAMPLE.org:443 "),
            identity("https://example.org/")
        );
        assert_eq!(
            identity("https://example.org/a/../b#fragment"),
            identity("https://example.org/b")
        );
        for other in [
            "http://example.org/b",
            "https://example.org/B",
            "https://example.org/b?query=1",
        ] {
            assert_ne!(identity("https://example.org/b"), identity(other));
        }
        assert_ne!(
            identity("https://example.org/b?a=1&b=2"),
            identity("https://example.org/b?b=2&a=1")
        );
    }

    #[test]
    fn anonymous_identity_is_deterministic_and_ignores_query_provenance() {
        let a = convert("Title,Query\nmarket news,first\nother news,first\n", 2);
        let b = convert("Title,Query\nother news,second\nmarket news,second\n", 3);
        assert_ne!(a.sources[0].source_id, b.sources[0].source_id);
        assert_eq!(a.sources[0].source_id, b.sources[1].source_id);
    }

    #[test]
    fn explicit_article_id_wins_and_is_preserved() {
        let a = convert(
            "ArticleId,Title,URL\narticle-42,market news,https://example.org/a\n",
            4,
        );
        let b = convert(
            "ArticleId,Title,URL\narticle-42,edited news,https://example.org/b\n",
            5,
        );
        assert_eq!(a.sources[0].source_id, b.sources[0].source_id);
        assert_eq!(
            a.sources[0].metadata.as_ref().unwrap()["ArticleId"],
            "article-42"
        );
    }

    #[test]
    fn converted_batches_deduplicate_only_the_same_article() {
        let first = convert("Title,URL\nmarket news,https://example.org/a\n", 10);
        let second = convert("Title,URL\nmarket news,https://example.org/b\n", 11);
        let repeated = convert("Title,URL\nmarket news,https://EXAMPLE.org:443/a#new\n", 12);
        let mut engine = realtime_seca_core::SecaEngine::new(SecaConfig::default()).unwrap();
        engine.build_baseline_tree(first).unwrap();
        let snapshot = serde_json::to_vec(&engine.snapshot().unwrap()).unwrap();
        let mut engine = realtime_seca_core::SecaEngine::load_snapshot(
            serde_json::from_slice(&snapshot).unwrap(),
        )
        .unwrap();
        assert_eq!(engine.process_batch(second).unwrap().sources_processed, 1);
        assert_eq!(engine.process_batch(repeated).unwrap().sources_processed, 0);
    }

    #[test]
    fn timestamps_require_explicit_offsets() {
        assert_eq!(parse_timestamp_to_unix_ms("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(
            parse_timestamp_to_unix_ms("1970-01-01T01:00:00+01:00"),
            Some(0)
        );
        assert_eq!(
            parse_timestamp_to_unix_ms("1969-12-31T19:00:00-05:00"),
            Some(0)
        );
        assert_eq!(
            parse_timestamp_to_unix_ms("1970-01-01T00:00:00.123Z"),
            Some(123)
        );
        for invalid in [
            "",
            " ",
            "bad",
            "2026-02-30T00:00:00Z",
            "2026-01-01T12:00:00",
        ] {
            assert_eq!(parse_timestamp_to_unix_ms(invalid), None);
        }
        let b = convert(
            "Title,Timestamp,API_Timestamp\nmarket news,bad,1970-01-01T00:00:00Z\nother news,,\n",
            6,
        );
        assert_eq!(b.sources[0].timestamp_unix_ms, Some(0));
        assert_eq!(b.sources[1].timestamp_unix_ms, None);
    }
}
