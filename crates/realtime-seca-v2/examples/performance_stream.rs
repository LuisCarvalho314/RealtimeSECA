//! Algorithm-only research benchmark; input conversion and observations are untimed.
use realtime_seca_core::{SecaConfig, SecaEngine, SourceBatch};
use realtime_seca_v2::{metrics::Errors, Config, Hkt, Model, Source};
use serde_json::json;
use std::{collections::BTreeSet, env, error::Error, fs, time::Instant};

fn rss_kib() -> Option<usize> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes the pointed-to rusage on success.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    let bytes = unsafe { usage.assume_init() }.ru_maxrss as usize;
    Some(if cfg!(target_os = "macos") {
        bytes / 1024
    } else {
        bytes
    })
}

fn cpu_seconds() -> Option<f64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes the pointed-to rusage on success.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1_000_000.0;
    Some(seconds(usage.ru_utime) + seconds(usage.ru_stime))
}

fn shape(h: &Hkt, depth: usize, counts: &mut [usize; 4]) {
    counts[0] += 1;
    counts[3] = counts[3].max(depth);
    for n in &h.nodes {
        counts[1] += 1;
        counts[2] += usize::from(n.words.is_empty());
        if let Some(child) = &n.child {
            shape(child, depth + 1, counts);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || !["v1", "v2"].contains(&args[1].as_str()) {
        return Err("usage: performance_stream v1|v2 CONFIG BATCH...".into());
    }
    let c: SecaConfig = serde_json::from_slice(&fs::read(&args[2])?)?;
    let mut batches = Vec::new();
    for p in &args[3..] {
        batches.push(serde_json::from_slice::<SourceBatch>(&fs::read(p)?)?);
    }
    let inputs: Vec<Vec<Source>> = batches
        .iter()
        .map(|b| {
            b.sources
                .iter()
                .map(|s| Source {
                    id: s.source_id.clone(),
                    words: s.tokens.iter().cloned().collect::<BTreeSet<_>>(),
                })
                .collect()
        })
        .collect();
    let mut v1 = SecaEngine::new(c.clone())?;
    let mut v2 = Model::new(Config {
        alpha: c.seca_thresholds.alpha,
        beta: c.seca_thresholds.beta,
        error_limits: Errors {
            alpha: c.seca_thresholds.alpha_error_threshold,
            beta: c.seca_thresholds.beta_error_threshold,
            importance: c.seca_thresholds.word_importance_error_threshold,
        },
        minimum_branch_sources: c
            .hkt_builder
            .minimum_number_of_sources_to_create_branch_for_node,
        window: if c.memory_mode == realtime_seca_core::MemoryMode::SlidingWindow {
            c.max_batches_in_memory.map(u64::from)
        } else {
            None
        },
    })?;
    let mut processed = 0;
    let stream_start = Instant::now();
    for (b, input) in batches.into_iter().zip(inputs) {
        let batch_index = b.batch_index;
        let batch_size = b.sources.len();
        let cpu_before = cpu_seconds();
        let start = Instant::now();
        let (inspected, reconstructed, hkts, nodes, refuge, depth, memberships, retained, elapsed) =
            if args[1] == "v1" {
                let report = if batch_index == 0 {
                    v1.build_baseline_tree(b)?
                } else {
                    v1.process_batch(b)?
                };
                let elapsed = start.elapsed().as_secs_f64();
                let tree = v1.export_baseline_tree()?;
                let parents: std::collections::BTreeMap<_, _> =
                    tree.nodes.iter().map(|n| (n.node_id, n.hkt_id)).collect();
                let hparents: std::collections::BTreeMap<_, _> = tree
                    .hkts
                    .iter()
                    .map(|h| (h.hkt_id, h.parent_node_id))
                    .collect();
                let max_depth = tree
                    .hkts
                    .iter()
                    .map(|h| {
                        let mut d = 1;
                        let mut p = h.parent_node_id;
                        while p != 0 {
                            d += 1;
                            p = *hparents.get(parents.get(&p).unwrap()).unwrap();
                        }
                        d
                    })
                    .max()
                    .unwrap_or(0);
                processed += batch_size;
                println!(
                    "{}",
                    json!({"engine":"v1","batch":batch_index,"documents":processed,
                    "algorithm_seconds":elapsed,"inspected_hkts":report.hkts_inspected,
                    "stream_wall_seconds":stream_start.elapsed().as_secs_f64(),
                    "reconstructed_hkts":report.reconstructed_hkt_ids.len(),"retained_documents":report.active_source_count,
                    "hkts":tree.hkts.len(),"nodes":tree.nodes.len(),"refuge_nodes":tree.nodes.iter().filter(|n|n.is_refuge_node).count(),
                    "max_depth":max_depth,"source_memberships":tree.nodes.iter().map(|n|n.source_ids.len()).sum::<usize>(),
                    "rss_after_batch_kib":rss_kib(),
                    "algorithm_cpu_seconds":cpu_before.zip(cpu_seconds()).map(|(a,z)|z-a)})
                );
                continue;
            } else {
                let decisions = v2.update(batch_index.into(), input)?;
                let elapsed = start.elapsed().as_secs_f64();
                let mut counts = [0; 4];
                shape(v2.root(), 1, &mut counts);
                let memberships = count_memberships(v2.root());
                (
                    decisions.len(),
                    decisions.iter().filter(|d| d.reconstructed).count(),
                    counts[0],
                    counts[1],
                    counts[2],
                    counts[3],
                    memberships,
                    v2.retained_sources().len(),
                    elapsed,
                )
            };
        processed += batch_size;
        println!(
            "{}",
            json!({"engine":"v2","batch":batch_index,"documents":processed,
            "algorithm_seconds":elapsed,"inspected_hkts":inspected,"reconstructed_hkts":reconstructed,
            "stream_wall_seconds":stream_start.elapsed().as_secs_f64(),
            "retained_documents":retained,"hkts":hkts,"nodes":nodes,"refuge_nodes":refuge,
            "max_depth":depth,"source_memberships":memberships,"rss_after_batch_kib":rss_kib(),
            "algorithm_cpu_seconds":cpu_before.zip(cpu_seconds()).map(|(a,z)|z-a)})
        );
    }
    Ok(())
}

fn count_memberships(h: &Hkt) -> usize {
    h.nodes
        .iter()
        .map(|n| n.sources.len() + n.child.as_ref().map_or(0, |c| count_memberships(c)))
        .sum()
}
