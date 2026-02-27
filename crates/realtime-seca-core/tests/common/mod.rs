use realtime_seca_core::{SourceBatch, SourceRecord};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

pub fn load_large_batch_fixture() -> SourceBatch {
    let path = Path::new("tests/data/large_batch.json");
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&contents)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

pub fn split_batch_into_chunks(batch: &SourceBatch, chunk_count: usize) -> Vec<SourceBatch> {
    assert!(chunk_count > 0, "chunk_count must be > 0");
    assert!(
        !batch.sources.is_empty(),
        "cannot split empty source batch into chunks"
    );

    let total = batch.sources.len();
    let base = total / chunk_count;
    let remainder = total % chunk_count;

    let mut chunks = Vec::with_capacity(chunk_count);
    let mut start = 0usize;
    for chunk_index in 0..chunk_count {
        let chunk_len = base + usize::from(chunk_index < remainder);
        let end = start + chunk_len;
        let sources = batch.sources[start..end]
            .iter()
            .map(|source| SourceRecord {
                source_id: source.source_id.clone(),
                batch_index: chunk_index as u32,
                tokens: source.tokens.clone(),
                text: source.text.clone(),
                timestamp_unix_ms: source.timestamp_unix_ms,
                metadata: source.metadata.clone(),
            })
            .collect::<Vec<_>>();

        chunks.push(SourceBatch {
            batch_index: chunk_index as u32,
            sources,
        });
        start = end;
    }

    assert_eq!(
        start, total,
        "split consumed an unexpected number of sources"
    );
    assert_eq!(chunks.len(), chunk_count, "unexpected number of chunks");

    let total_after: usize = chunks.iter().map(|chunk| chunk.sources.len()).sum();
    assert_eq!(
        total_after, total,
        "chunking dropped or duplicated sources by count"
    );
    let before_ids: BTreeSet<&str> = batch
        .sources
        .iter()
        .map(|source| source.source_id.as_str())
        .collect();
    let after_ids: BTreeSet<&str> = chunks
        .iter()
        .flat_map(|chunk| chunk.sources.iter().map(|source| source.source_id.as_str()))
        .collect();
    assert_eq!(
        before_ids, after_ids,
        "chunking changed source identity set"
    );

    chunks
}

pub fn chunked_large_batches_8() -> Vec<SourceBatch> {
    let batch = load_large_batch_fixture();
    let chunks = split_batch_into_chunks(&batch, 8);
    assert_eq!(chunks.len(), 8, "expected exactly 8 chunks");
    chunks
}
