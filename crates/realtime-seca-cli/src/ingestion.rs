use realtime_seca_core::{SecaConfig, SourceBatch, SourceRecord};
use regex::Regex;
use std::{fs, path::Path};
use stop_words::{get as get_stop_words, LANGUAGE};
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
        get_stop_words(LANGUAGE::English).iter().copied().collect();

    let headers = reader.headers()?.clone();
    // A conversion scope avoids collisions even when separate files use the
    // default batch index. Persist the converted JSON to replay URL-less rows
    // with the same identity; reconversion deliberately creates new identities.
    let conversion_scope = uuid::Uuid::new_v4();

    let mut sources: Vec<SourceRecord> = Vec::new();

    for (row_index, result) in reader.records().enumerate() {
        let record = result?;
        let row_number = row_index + 1;

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
            source_id: {
                let url = get("URL");
                let url = url.trim();
                if url.is_empty() {
                    format!("csv-row:{conversion_scope}:{batch_index}:{row_number}")
                } else {
                    format!("url:{url}")
                }
            },
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

pub(crate) fn normalize_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) fn parse_timestamp_to_unix_ms(value: &str) -> Option<i64> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    // CSV Timestamp uses a space separator and an explicit numeric offset.
    // Also accept RFC3339. API_Timestamp without an offset is ambiguous and
    // must remain unset, as must malformed or missing timestamps.
    chrono::DateTime::parse_from_rfc3339(trimmed)
        .or_else(|_| chrono::DateTime::parse_from_str(trimmed, "%Y-%m-%d %H:%M:%S%.f%:z"))
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
    use super::parse_timestamp_to_unix_ms;

    #[test]
    fn parses_only_timestamps_with_explicit_offsets() {
        for value in [
            "1970-01-01 00:00:00.123456+00:00",
            "1970-01-01T00:00:00.123456Z",
            "1970-01-01 01:00:00.123456+01:00",
            " 1970-01-01T00:00:00.123Z ",
        ] {
            assert_eq!(parse_timestamp_to_unix_ms(value), Some(123), "{value}");
        }
        assert_eq!(
            parse_timestamp_to_unix_ms("1969-12-31T23:59:59Z"),
            Some(-1000)
        );
        for value in [
            "",
            " ",
            "bad",
            "2026-02-16 15:24:56",
            "2026-02-30T00:00:00Z",
            "2026-01-01T25:00:00Z",
        ] {
            assert_eq!(parse_timestamp_to_unix_ms(value), None, "{value}");
        }
    }
}
