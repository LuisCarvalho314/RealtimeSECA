//! Deterministic public-API benchmark; CSV on stdout, summary on stderr.
use realtime_seca_core::{MemoryMode, SecaConfig, SecaEngine, SourceBatch, SourceRecord};
use std::{error::Error, hint::black_box, time::Instant};

struct Options {
    batches: u32,
    sources: usize,
    vocabulary: usize,
    overlap: f64,
    novel_every: u32,
    gamma: u32,
    seed: u64,
    threshold: f64,
    branch_min: usize,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut o = Self {
            batches: 30,
            sources: 100,
            vocabulary: 256,
            overlap: 0.95,
            novel_every: 0,
            gamma: 3,
            seed: 42,
            threshold: 0.1,
            branch_min: 32,
        };
        let mut args = std::env::args().skip(1);
        while let Some(key) = args.next() {
            if key == "--help" {
                println!(
                    "--batches 30|90 --sources N --vocabulary N --overlap 0..1 \
                    --novel-every N (0 disables) --gamma N (0 means Full) \
                    --branch-min N --seed N --threshold 0..1 (all three error thresholds)"
                );
                std::process::exit(0);
            }
            let value = args.next().ok_or("missing option value")?;
            match key.as_str() {
                "--batches" => o.batches = value.parse()?,
                "--sources" => o.sources = value.parse()?,
                "--vocabulary" => o.vocabulary = value.parse()?,
                "--overlap" => o.overlap = value.parse()?,
                "--novel-every" => o.novel_every = value.parse()?,
                "--gamma" => o.gamma = value.parse()?,
                "--seed" => o.seed = value.parse()?,
                "--threshold" => o.threshold = value.parse()?,
                "--branch-min" => o.branch_min = value.parse()?,
                _ => return Err(format!("unknown option: {key}").into()),
            }
        }
        if o.batches == 0
            || o.sources == 0
            || o.vocabulary < 16
            || o.vocabulary > 1_000_000
            || !(0.0..=1.0).contains(&o.overlap)
            || !(0.0..=1.0).contains(&o.threshold)
        {
            return Err(
                "require positive batches/sources, vocabulary 16..1000000, overlap/threshold 0..1"
                    .into(),
            );
        }
        Ok(o)
    }
}

// SplitMix64: fixed algorithm, including seed zero; no external RNG dependency.
fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn batch(o: &Options, index: u32, rng: &mut u64) -> SourceBatch {
    let pool = o.vocabulary / 4;
    let shift = (pool as f64 * (1.0 - o.overlap)).round() as u64;
    let offset = u64::from(index) * shift;
    let novel = o.novel_every != 0 && index != 0 && index.is_multiple_of(o.novel_every);
    let sources = (0..o.sources)
        .map(|s| {
            let topic = s % 4;
            let mut tokens = vec!["news".into(), "report".into()];
            let label = if novel {
                format!("new_topic_{index}_{topic}")
            } else {
                format!("topic_{topic}")
            };
            tokens.push(label);
            for _ in 0..20 {
                let word = offset + random(rng) % pool as u64;
                tokens.push(format!("sector_{topic}_word_{word}"));
            }
            tokens.sort();
            tokens.dedup();
            SourceRecord {
                source_id: format!("news/{index}/{s}"),
                batch_index: index,
                tokens,
                text: None,
                timestamp_unix_ms: None,
                metadata: None,
            }
        })
        .collect();
    SourceBatch {
        batch_index: index,
        sources,
    }
}

fn memory_kib() -> (String, String) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let read = |key: &str| {
        status
            .lines()
            .find_map(|line| {
                line.strip_prefix(key)?
                    .split_whitespace()
                    .next()
                    .map(str::to_owned)
            })
            .unwrap_or_default()
    };
    (read("VmRSS:"), read("VmHWM:"))
}

fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn quantile(values: &[f64], fraction: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(fraction * sorted.len() as f64).ceil().max(1.0) as usize - 1]
}

fn main() -> Result<(), Box<dyn Error>> {
    let o = Options::parse()?;
    let mut config = SecaConfig {
        memory_mode: if o.gamma == 0 {
            MemoryMode::Full
        } else {
            MemoryMode::SlidingWindow
        },
        max_batches_in_memory: (o.gamma != 0).then_some(o.gamma),
        ..SecaConfig::default()
    };
    config
        .hkt_builder
        .minimum_number_of_sources_to_create_branch_for_node = o.branch_min;
    config.seca_thresholds.alpha_error_threshold = o.threshold;
    config.seca_thresholds.beta_error_threshold = o.threshold;
    config.seca_thresholds.word_importance_error_threshold = o.threshold;
    eprintln!("batches={},sources={},vocabulary={},overlap={},novel_every={},gamma={},seed={},threshold={},branch_min={}",
        o.batches, o.sources, o.vocabulary, o.overlap, o.novel_every, o.gamma, o.seed, o.threshold, o.branch_min);
    let mut engine = SecaEngine::new(config)?;
    let mut rng = o.seed;
    let baseline = batch(&o, 0, &mut rng);
    let start = Instant::now();
    engine.build_baseline_tree(baseline)?;
    eprintln!("baseline_ms={:.3}", milliseconds(start));
    let mut latencies = Vec::new();
    let mut serialization = Vec::new();
    let mut reconstructions = 0;
    println!("batch,process_ms,diagnostics_ms,clone_ms,snapshot_ms,json_ms,snapshot_bytes,active_sources,vocabulary,hkts,nodes,reconstructed,targets,rss_kib,hwm_kib");
    for index in 1..=o.batches {
        let incoming = batch(&o, index, &mut rng);
        let diagnostic_input = incoming.clone();
        let start = Instant::now();
        let report = engine.process_batch(incoming)?;
        let process_ms = milliseconds(start);
        latencies.push(process_ms);
        reconstructions += usize::from(report.reconstruction_triggered);
        let (rss, hwm) = memory_kib(); // Before extra measurement copies/exports.
        let start = Instant::now();
        let diagnostics = black_box(engine.display_diagnostics(Some(&diagnostic_input))?);
        let diagnostics_ms = milliseconds(start);
        drop(diagnostics);
        let start = Instant::now();
        let copy = black_box(engine.clone());
        let clone_ms = milliseconds(start);
        drop(copy);
        let start = Instant::now();
        let snapshot = engine.snapshot()?;
        let snapshot_ms = milliseconds(start);
        let start = Instant::now();
        let bytes = serde_json::to_vec(black_box(&snapshot))?;
        let json_ms = milliseconds(start);
        serialization.push(json_ms);
        let tree = engine.export_baseline_tree_verbose()?;
        println!("{index},{process_ms:.6},{diagnostics_ms:.6},{clone_ms:.6},{snapshot_ms:.6},{json_ms:.6},{},{},{},{},{},{},{},{rss},{hwm}",
            bytes.len(), report.active_source_count, tree.word_legend.len(), tree.hkts.len(),
            tree.nodes.len(), report.reconstruction_triggered, report.reconstructed_hkt_ids.len());
    }
    eprintln!("process_ms median={:.3} p95={:.3} max={:.3}; json_ms median={:.3} p95={:.3} max={:.3}; reconstruction_batches={}/{}",
        quantile(&latencies, 0.5), quantile(&latencies, 0.95), quantile(&latencies, 1.0),
        quantile(&serialization, 0.5), quantile(&serialization, 0.95), quantile(&serialization, 1.0),
        reconstructions, o.batches);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> Options {
        Options {
            batches: 30,
            sources: 8,
            vocabulary: 32,
            overlap: 1.0,
            novel_every: 2,
            gamma: 3,
            seed: 0,
            threshold: 1.0,
            branch_min: 32,
        }
    }

    #[test]
    fn generator_is_reproducible_and_has_unique_batch_identities() {
        let o = options();
        let mut a = o.seed;
        let mut b = o.seed;
        for index in 0..4 {
            let first = batch(&o, index, &mut a);
            let second = batch(&o, index, &mut b);
            assert_eq!(
                serde_json::to_vec(&first).unwrap(),
                serde_json::to_vec(&second).unwrap()
            );
            let ids: std::collections::BTreeSet<_> =
                first.sources.iter().map(|s| &s.source_id).collect();
            assert_eq!(ids.len(), o.sources);
            assert!(first.sources.iter().all(|s| s.batch_index == index));
            if index == 2 {
                assert!(first.sources[0]
                    .tokens
                    .contains(&"new_topic_2_0".to_owned()));
            }
        }
    }

    #[test]
    fn reported_quantiles_use_nearest_rank() {
        assert_eq!(quantile(&[4.0, 1.0, 3.0, 2.0], 0.5), 2.0);
        assert_eq!(quantile(&[4.0, 1.0, 3.0, 2.0], 0.95), 4.0);
    }
}
