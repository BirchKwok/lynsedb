//! Approximate flat brute-force search with configurable distance precision.
//!
//! Inspired by:
//! - **ADSampling** (Gao et al.): partial-dimension distance estimates for pruning.
//! - **ANSMET** (ISCA'25): partial-dimension lower/upper bounds for early termination.
//! - **Two-phase coarse→fine**: cheap prefix scan → exact f32 re-score on a shortlist.
//!
//! Coarse pass reuses the fused top-k scan in `flat_mmap` (SIMD batch-8 for IP) on
//! leading dimensions only; `eps` controls output rounding and shortlist size.

use crate::distance::{self, simd, DistanceMetric};
use std::cmp::Ordering;

/// Configuration for approximate distance search on flat mmap data.
#[derive(Debug, Clone, Copy)]
pub struct ApproxSearchConfig {
    pub eps: f32,
}

impl ApproxSearchConfig {
    pub fn new(eps: f32) -> Self {
        Self {
            eps: eps.max(1e-8),
        }
    }
}

/// Round a distance to the nearest multiple of `eps`.
#[inline]
pub fn round_to_eps(value: f32, eps: f32) -> f32 {
    if !value.is_finite() || eps <= 0.0 {
        return value;
    }
    (value / eps).round() * eps
}

/// Leading dimensions scanned in the coarse pass (prefix SIMD; ~75% bandwidth at d=96).
fn coarse_dims(full_dim: usize, eps: f32) -> usize {
    let log = (-eps.log10()).clamp(2.0, 8.0);
    let target = (52.0 + 4.0 * log).round() as usize;
    target.max(64).min(96).min(full_dim)
}

/// Shortlist size for the exact re-score phase.
fn candidate_pool_size(k: usize, n: usize, eps: f32) -> usize {
    let log = (-eps.log10() as f64).clamp(2.0, 8.0);
    let from_k = (k as f64 * (35.0 + 12.0 * log)).round() as usize;
    let from_n = ((n as f64).powf(0.40) * (10.0 + 2.0 * log)).round() as usize;
    let mut pool = from_k.max(from_n).max(k + 128);
    if n >= 100_000 {
        pool = pool.max(8000);
    }
    pool.min(n).min(16384)
}

fn exact_distance(query: &[f32], cand: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::InnerProduct => simd::inner_product_f32(query, cand),
        DistanceMetric::L2Squared => simd::l2_squared_f32(query, cand),
        DistanceMetric::Cosine => simd::cosine_distance_f32(query, cand),
        _ => distance::compute_distance_f32(query, cand, metric),
    }
}

/// Two-phase approximate flat search: prefix-dimension coarse ranking + exact re-score.
pub fn approx_flat_search(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    metric: DistanceMetric,
    config: ApproxSearchConfig,
) -> (Vec<u32>, Vec<f32>) {
    if n == 0 || k == 0 {
        return (vec![], vec![]);
    }
    let k = k.min(n);
    let eps = config.eps;
    let d_coarse = coarse_dims(dim, eps);
    let pool = candidate_pool_size(k, n, eps);

    let shortlist = super::flat_mmap::approx_coarse_shortlist(
        query, candidates, dim, d_coarse, pool, n, metric,
    );

    refine_shortlist(query, candidates, dim, k, metric, eps, shortlist)
}

fn refine_shortlist(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    metric: DistanceMetric,
    eps: f32,
    shortlist: Vec<u32>,
) -> (Vec<u32>, Vec<f32>) {
    let ascending = metric.is_ascending();
    let mut scored: Vec<(f32, u32)> = shortlist
        .into_iter()
        .map(|idx| {
            let start = idx as usize * dim;
            let cand = unsafe { candidates.get_unchecked(start..start + dim) };
            let dist = round_to_eps(exact_distance(query, cand, metric), eps);
            (dist, idx)
        })
        .collect();

    let take = k.min(scored.len());
    if ascending {
        scored.select_nth_unstable_by(take - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
        });
        scored[..take].sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    } else {
        scored.select_nth_unstable_by(take - 1, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal)
        });
        scored[..take].sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    }

    let top = &scored[..take];
    let indices = top.iter().map(|&(_, idx)| idx).collect();
    let dists = top.iter().map(|&(d, _)| d).collect();
    (indices, dists)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_to_eps() {
        assert!((round_to_eps(1.234567, 1e-4) - 1.2346).abs() < 1e-6);
        assert!((round_to_eps(1.23454, 1e-4) - 1.2345).abs() < 1e-6);
    }

    fn test_rng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) as f32 / u32::MAX as f32
        }
    }

    #[test]
    fn test_approx_recall_small() {
        let dim = 128;
        let n = 8000;
        let mut data = vec![0.0f32; n * dim];
        let mut query = vec![0.0f32; dim];
        let mut rng = test_rng(42);
        for v in data.iter_mut() {
            *v = rng();
        }
        for v in query.iter_mut() {
            *v = rng();
        }

        let exact = |metric: DistanceMetric| {
            let mut pairs: Vec<(f32, u32)> = (0..n as u32)
                .map(|idx| {
                    let s = idx as usize * dim;
                    let cand = &data[s..s + dim];
                    (exact_distance(&query, cand, metric), idx)
                })
                .collect();
            if metric.is_ascending() {
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            } else {
                pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            }
            pairs.into_iter().take(10).map(|(_, idx)| idx).collect::<Vec<_>>()
        };

        for metric in [
            DistanceMetric::InnerProduct,
            DistanceMetric::L2Squared,
            DistanceMetric::Cosine,
        ] {
            let gt: std::collections::HashSet<u32> = exact(metric).into_iter().collect();
            let (ids, _) = approx_flat_search(
                &query,
                &data,
                dim,
                10,
                n,
                metric,
                ApproxSearchConfig::new(1e-4),
            );
            let recall = ids.iter().filter(|id| gt.contains(id)).count() as f32 / 10.0;
            assert!(
                recall >= 0.8,
                "metric={:?} recall={recall} ids={ids:?} gt={gt:?}",
                metric
            );
        }
    }
}
