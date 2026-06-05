#[path = "../../../src/distance/mod.rs"]
pub mod distance;
#[path = "../../../src/storage/polarvec_mmap.rs"]
pub mod polarvec_mmap_for_bench;
#[path = "../../../src/storage/pq_mmap.rs"]
pub mod pq_mmap_for_bench;

pub mod storage {
    pub use crate::polarvec_mmap_for_bench as polarvec_mmap;
    pub use crate::pq_mmap_for_bench as pq_mmap;
}

use crate::distance::{compute_distance_f32, DistanceMetric};
use crate::storage::polarvec_mmap::{PolarVecIndex, DEFAULT_BITS, DEFAULT_OVERSAMPLE};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Debug)]
struct Config {
    n: usize,
    dim: usize,
    nq: usize,
    k: usize,
    oversample: usize,
    bits: Vec<usize>,
    metrics: Vec<DistanceMetric>,
    recall_queries: usize,
    warmup: usize,
    normalize_cosine: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n: 100_000,
            dim: 128,
            nq: 100,
            k: 10,
            oversample: DEFAULT_OVERSAMPLE,
            bits: vec![DEFAULT_BITS],
            metrics: vec![
                DistanceMetric::InnerProduct,
                DistanceMetric::L2Squared,
                DistanceMetric::Cosine,
            ],
            recall_queries: 25,
            warmup: 5,
            normalize_cosine: true,
        }
    }
}

fn main() {
    let cfg = parse_args();
    println!(
        "polarvec_bench n={} dim={} nq={} k={} oversample={} bits={:?} metrics={:?} recall_queries={} warmup={} normalize_cosine={}",
        cfg.n, cfg.dim, cfg.nq, cfg.k, cfg.oversample, cfg.bits, cfg.metrics, cfg.recall_queries, cfg.warmup, cfg.normalize_cosine
    );

    let base_data = random_data(cfg.n, cfg.dim, 0x1234_5678);
    let base_queries = random_data(cfg.nq, cfg.dim, 0x8765_4321);

    for &metric in &cfg.metrics {
        let mut data = base_data.clone();
        let mut queries = base_queries.clone();
        if metric == DistanceMetric::Cosine && cfg.normalize_cosine {
            normalize_rows(&mut data, cfg.dim);
            normalize_rows(&mut queries, cfg.dim);
        }

        let recall_queries = cfg.recall_queries.min(cfg.nq);
        let exact_start = Instant::now();
        let exact: Vec<Vec<u32>> = (0..recall_queries)
            .map(|qi| {
                exact_topk(
                    &data,
                    cfg.dim,
                    &queries[qi * cfg.dim..(qi + 1) * cfg.dim],
                    cfg.k,
                    metric,
                )
            })
            .collect();
        let exact_time = exact_start.elapsed();

        for &bits in &cfg.bits {
            let build_start = Instant::now();
            let idx = PolarVecIndex::build_for_metric(&data, cfg.n, cfg.dim, bits, metric);
            let build_time = build_start.elapsed();
            let index_bytes = saved_index_bytes(&idx, metric, bits);

            for qi in 0..cfg.warmup.min(cfg.nq) {
                let q = &queries[qi * cfg.dim..(qi + 1) * cfg.dim];
                let _ = idx.search(q, cfg.k, &data, metric, cfg.oversample);
            }

            let search_start = Instant::now();
            let mut approx_for_recall = Vec::with_capacity(recall_queries);
            for qi in 0..cfg.nq {
                let q = &queries[qi * cfg.dim..(qi + 1) * cfg.dim];
                let (ids, _) = idx.search(q, cfg.k, &data, metric, cfg.oversample);
                if qi < recall_queries {
                    approx_for_recall.push(ids);
                }
            }
            let search_time = search_start.elapsed();

            let recall = if recall_queries == 0 {
                0.0
            } else {
                exact
                    .iter()
                    .zip(approx_for_recall.iter())
                    .map(|(ex, ap)| recall_at_k(ex, ap))
                    .sum::<f32>()
                    / recall_queries as f32
            };

            println!(
                "metric={} bits={} build_ms={:.3} search_us_per_query={:.3} recall_at_{}={:.4} exact_ms_for_recall={:.3} index_bytes={} bytes_per_vector={:.2}",
                metric.name(),
                bits,
                millis(build_time),
                micros_per_query(search_time, cfg.nq),
                cfg.k,
                recall,
                millis(exact_time),
                index_bytes,
                index_bytes as f64 / cfg.n as f64
            );
        }
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let args: Vec<String> = env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        let key = args[i].as_str();
        let val = args.get(i + 1).map(|s| s.as_str());
        match (key, val) {
            ("--n", Some(v)) => cfg.n = parse_usize(key, v),
            ("--dim", Some(v)) => cfg.dim = parse_usize(key, v),
            ("--nq", Some(v)) => cfg.nq = parse_usize(key, v),
            ("--k", Some(v)) => cfg.k = parse_usize(key, v),
            ("--oversample", Some(v)) => cfg.oversample = parse_usize(key, v),
            ("--recall-queries", Some(v)) => cfg.recall_queries = parse_usize(key, v),
            ("--warmup", Some(v)) => cfg.warmup = parse_usize(key, v),
            ("--normalize-cosine", Some(v)) => cfg.normalize_cosine = parse_bool(key, v),
            ("--bits", Some(v)) => cfg.bits = parse_usize_list(key, v),
            ("--metrics", Some(v)) => cfg.metrics = parse_metrics(v),
            ("--help", _) | ("-h", _) => {
                print_help_and_exit();
            }
            _ => {
                eprintln!("unknown or missing argument: {key}");
                print_help_and_exit();
            }
        }
        i += 2;
    }
    cfg
}

fn print_help_and_exit() -> ! {
    eprintln!(
        "Usage: cargo run --release --manifest-path benchmarks/polarvec_bench/Cargo.toml -- [--n 100000] [--dim 128] [--nq 100] [--k 10] [--oversample 20] [--bits 4,3] [--metrics ip,l2,cosine] [--recall-queries 25] [--warmup 5] [--normalize-cosine true]"
    );
    std::process::exit(2);
}

fn parse_usize(name: &str, value: &str) -> usize {
    value.parse().unwrap_or_else(|_| {
        eprintln!("{name} expects an unsigned integer, got {value}");
        std::process::exit(2);
    })
}

fn parse_usize_list(name: &str, value: &str) -> Vec<usize> {
    let parsed: Vec<usize> = value
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| parse_usize(name, s))
        .collect();
    if parsed.is_empty() {
        eprintln!("{name} must contain at least one value");
        std::process::exit(2);
    }
    parsed
}

fn parse_bool(name: &str, value: &str) -> bool {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" => true,
        "0" | "false" | "no" | "n" => false,
        _ => {
            eprintln!("{name} expects true/false, got {value}");
            std::process::exit(2);
        }
    }
}

fn parse_metrics(value: &str) -> Vec<DistanceMetric> {
    let parsed: Vec<DistanceMetric> = value
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            DistanceMetric::from_str(s).unwrap_or_else(|| {
                eprintln!("unknown metric: {s}");
                std::process::exit(2);
            })
        })
        .collect();
    if parsed.is_empty() {
        eprintln!("--metrics must contain at least one metric");
        std::process::exit(2);
    }
    parsed
}

fn random_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n * dim)
        .map(|_| rng.gen_range(-1.0f32..1.0f32))
        .collect()
}

fn normalize_rows(data: &mut [f32], dim: usize) {
    for row in data.chunks_exact_mut(dim) {
        let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
        for v in row {
            *v /= norm;
        }
    }
}

fn exact_topk(
    data: &[f32],
    dim: usize,
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
) -> Vec<u32> {
    let mut scores: Vec<(f32, u32)> = data
        .chunks_exact(dim)
        .enumerate()
        .map(|(i, v)| (compute_distance_f32(query, v, metric), i as u32))
        .collect();

    if metric.is_ascending() {
        scores.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    } else {
        scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    }
    scores.truncate(k);
    scores.into_iter().map(|(_, idx)| idx).collect()
}

fn recall_at_k(exact: &[u32], approx: &[u32]) -> f32 {
    let hits = exact.iter().filter(|id| approx.contains(id)).count();
    hits as f32 / exact.len().max(1) as f32
}

fn saved_index_bytes(idx: &PolarVecIndex, metric: DistanceMetric, bits: usize) -> u64 {
    let mut path = env::temp_dir();
    path.push(format!(
        "lynse_polarvec_bench_{}_{}_{}.bin",
        std::process::id(),
        metric.name(),
        bits
    ));
    let path: PathBuf = path;
    idx.save(&path).expect("save PolarVec index for size check");
    let bytes = std::fs::metadata(&path).expect("stat PolarVec index").len();
    let _ = std::fs::remove_file(path);
    bytes
}

fn millis(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000.0
}

fn micros_per_query(d: Duration, nq: usize) -> f64 {
    d.as_secs_f64() * 1_000_000.0 / nq.max(1) as f64
}
