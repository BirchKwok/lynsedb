//! Approximate flat brute-force search with configurable distance precision.
//!
//! Inspired by:
//! - **ADSampling** (Gao et al.): partial-dimension distance estimates for pruning.
//! - **ANSMET** (ISCA'25): partial-dimension lower/upper bounds for early termination.
//! - **Hybrid coarse→fine search**: raw f32 aggregate or partial-dimension
//!   shortlist followed by exact f32 re-score on the shortlist.
//!
//! The approximate path never quantizes vectors. `eps` controls output rounding
//! while ranking is still performed on raw f32 distances. Approximate flat
//! search is only implemented for IP, L2, and Cosine; Hamming/Jaccard use the
//! exact binary-distance path even if a caller passes `approx=True`.

use crate::distance::{self, simd, DistanceMetric};
use rayon::prelude::*;
use std::cmp::Ordering;

/// Configuration for approximate distance search on flat mmap data.
#[derive(Debug, Clone, Copy)]
pub struct ApproxSearchConfig {
    pub eps: f32,
}

impl ApproxSearchConfig {
    pub fn new(eps: f32) -> Self {
        Self {
            eps: normalize_eps(eps),
        }
    }
}

const DEFAULT_EPS: f32 = 1e-4;
const MIN_EPS: f32 = 1e-8;
pub(super) const APPROX_BOUND_BLOCK_LEN: usize = 16;
pub(super) const APPROX_INIT_ROWS: usize = 65_536;

/// Exact f32 norm sidecar for safe early termination.
///
/// This is not vector quantization: values are not bucketed, compressed, or used
/// as a replacement distance. They are f32 suffix norms derived from the original
/// vectors and only provide upper/lower bounds for pruning.
pub(super) struct ApproxBounds {
    pub block_len: usize,
    pub block_count: usize,
    pub suffix_norm2: Vec<f32>,
    pub total_norm2: Vec<f32>,
}

impl ApproxBounds {
    #[allow(dead_code)]
    pub(super) fn build(candidates: &[f32], dim: usize, block_len: usize) -> Self {
        let block_len = block_len.max(1);
        let n = if dim == 0 { 0 } else { candidates.len() / dim };
        let block_count = dim.div_ceil(block_len).max(1);
        let mut suffix_norm2 = vec![0.0f32; n * block_count];
        let mut total_norm2 = vec![0.0f32; n];

        suffix_norm2
            .par_chunks_mut(block_count)
            .zip(total_norm2.par_iter_mut())
            .zip(candidates.par_chunks(dim))
            .for_each(|((tails, total), cand)| {
                let mut suffix = 0.0f32;
                for block in (0..block_count).rev() {
                    tails[block] = suffix;
                    let start = block * block_len;
                    let end = (start + block_len).min(dim);
                    let mut norm = 0.0f32;
                    for &value in &cand[start..end] {
                        norm += value * value;
                    }
                    suffix += norm;
                }
                *total = suffix;
            });

        Self {
            block_len,
            block_count,
            suffix_norm2,
            total_norm2,
        }
    }

    #[inline]
    pub(super) fn is_compatible(&self, dim: usize, block_len: usize, n: usize) -> bool {
        self.block_len == block_len
            && self.block_count == dim.div_ceil(block_len.max(1)).max(1)
            && self.total_norm2.len() == n
            && self.suffix_norm2.len() == n * self.block_count
    }
}

/// Normalize user-provided distance precision.
///
/// `eps` is allowed to be very large, but it must be finite and positive. This
/// keeps `round_to_eps` from producing NaN for inputs such as `inf`.
#[inline]
pub fn normalize_eps(eps: f32) -> f32 {
    if eps.is_finite() && eps > 0.0 {
        eps.max(MIN_EPS)
    } else {
        DEFAULT_EPS
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

fn exact_distance(query: &[f32], cand: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::InnerProduct => simd::inner_product_f32(query, cand),
        DistanceMetric::L2Squared => simd::l2_squared_f32(query, cand),
        DistanceMetric::Cosine => simd::cosine_distance_f32(query, cand),
        _ => distance::compute_distance_f32(query, cand, metric),
    }
}

/// Approximate flat search entry used by tests and fallback callers.
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

    approx_flat_search_with_bounds(query, candidates, dim, k, n, metric, config, None)
}

pub(super) fn approx_flat_search_with_bounds(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    metric: DistanceMetric,
    config: ApproxSearchConfig,
    bounds: Option<&ApproxBounds>,
) -> (Vec<u32>, Vec<f32>) {
    if n == 0 || k == 0 {
        return (vec![], vec![]);
    }
    let k = k.min(n);
    let eps = config.eps;

    if !metric.supports_flat_approx() {
        return distance::top_k_search(query, candidates, dim, k, metric);
    }

    if matches!(
        metric,
        DistanceMetric::InnerProduct | DistanceMetric::L2Squared | DistanceMetric::Cosine
    ) && dim > APPROX_BOUND_BLOCK_LEN
        && n > APPROX_INIT_ROWS
    {
        let (indices, dists) =
            super::flat_mmap::approx_hybrid_search(query, candidates, dim, k, n, metric, eps);
        let dists = dists
            .into_iter()
            .map(|dist| round_to_eps(dist, eps))
            .collect();
        return (indices, dists);
    }

    if let Some(bounds) = bounds {
        if matches!(
            metric,
            DistanceMetric::InnerProduct | DistanceMetric::L2Squared | DistanceMetric::Cosine
        ) && bounds.is_compatible(dim, APPROX_BOUND_BLOCK_LEN, n)
            && dim > APPROX_BOUND_BLOCK_LEN
            && n > APPROX_INIT_ROWS
        {
            let (indices, dists) = super::flat_mmap::approx_bounded_search(
                query, candidates, dim, k, n, metric, bounds,
            );
            let dists = dists
                .into_iter()
                .map(|dist| round_to_eps(dist, eps))
                .collect();
            return (indices, dists);
        }
    }

    let all_rows = (0..n as u32).collect();
    refine_shortlist(query, candidates, dim, k, metric, eps, all_rows)
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
        .filter_map(|idx| {
            let start = idx as usize * dim;
            let cand = candidates.get(start..start + dim)?;
            let dist = round_to_eps(exact_distance(query, cand, metric), eps);
            Some((dist, idx))
        })
        .collect();

    let take = k.min(scored.len());
    if take == 0 {
        return (Vec::new(), Vec::new());
    }
    if ascending {
        scored.select_nth_unstable_by(take - 1, |a, b| cmp_ascending(*a, *b));
        scored[..take].sort_unstable_by(|a, b| cmp_ascending(*a, *b));
    } else {
        scored.select_nth_unstable_by(take - 1, |a, b| cmp_descending(*a, *b));
        scored[..take].sort_unstable_by(|a, b| cmp_descending(*a, *b));
    }

    let top = &scored[..take];
    let indices = top.iter().map(|&(_, idx)| idx).collect();
    let dists = top.iter().map(|&(d, _)| d).collect();
    (indices, dists)
}

#[inline]
fn cmp_ascending(a: (f32, u32), b: (f32, u32)) -> Ordering {
    a.0.partial_cmp(&b.0)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.1.cmp(&b.1))
}

#[inline]
fn cmp_descending(a: (f32, u32), b: (f32, u32)) -> Ordering {
    b.0.partial_cmp(&a.0)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.1.cmp(&b.1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_to_eps() {
        assert!((round_to_eps(1.234567, 1e-4) - 1.2346).abs() < 1e-6);
        assert!((round_to_eps(1.23454, 1e-4) - 1.2345).abs() < 1e-6);
        assert_eq!(round_to_eps(f32::INFINITY, 1e-4), f32::INFINITY);
    }

    #[test]
    fn test_normalize_eps_rejects_non_finite_values() {
        assert_eq!(ApproxSearchConfig::new(f32::INFINITY).eps, 1e-4);
        assert_eq!(ApproxSearchConfig::new(f32::NAN).eps, 1e-4);
        assert_eq!(ApproxSearchConfig::new(-1.0).eps, 1e-4);
        assert_eq!(ApproxSearchConfig::new(0.0).eps, 1e-4);
        assert_eq!(ApproxSearchConfig::new(1e-12).eps, 1e-8);
    }

    #[test]
    fn test_binary_metrics_ignore_approx_and_eps_rounding() {
        let dim = 4;
        let query = vec![1.0, 0.0, 1.0, 0.0];
        let data = vec![
            1.0, 0.0, 1.0, 0.0, //
            1.0, 1.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, 1.0,
        ];
        let cases = [
            (DistanceMetric::Hamming, 2.0f32),
            (DistanceMetric::Jaccard, 0.5f32),
        ];

        for (metric, eps) in cases {
            let exact = distance::top_k_search(&query, &data, dim, 3, metric);
            let approx = approx_flat_search(
                &query,
                &data,
                dim,
                3,
                3,
                metric,
                ApproxSearchConfig::new(eps),
            );
            let rounded: Vec<f32> = exact
                .1
                .iter()
                .map(|&dist| round_to_eps(dist, eps))
                .collect();

            assert_eq!(approx.0, exact.0);
            assert_eq!(approx.1, exact.1);
            assert!(
                approx
                    .1
                    .iter()
                    .zip(rounded.iter())
                    .any(|(actual, rounded)| (actual - rounded).abs() > 1e-6),
                "metric={metric:?} unexpectedly matched eps-rounded distances"
            );
        }
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
            pairs
                .into_iter()
                .take(10)
                .map(|(_, idx)| idx)
                .collect::<Vec<_>>()
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
                recall >= 0.97,
                "metric={:?} recall={recall} ids={ids:?} gt={gt:?}",
                metric
            );
        }
    }
}
