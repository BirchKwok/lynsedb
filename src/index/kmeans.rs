//! Shared IVF KMeans training and assignment helpers.
//!
//! IVF uses L2 Voronoi partitions for routing; the search metric is still
//! applied later when scoring candidates.

use crate::distance::{compute_distance_f32, DistanceMetric};
use rayon::prelude::*;
use std::collections::HashMap;

const INIT_SAMPLE_MIN: usize = 2_048;
const INIT_SAMPLE_MAX: usize = 10_000;
const INIT_SAMPLE_PER_CENTROID: usize = 32;

pub(crate) struct KMeansResult {
    pub centroids: Vec<f32>,
    pub assignments: Vec<usize>,
    pub n_centroids: usize,
}

/// Fast deterministic PRNG (PCG-style LCG, 64-bit state).
struct FastRng(u64);

impl FastRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Fisher-Yates partial shuffle: pick `count` random indices from 0..n.
    fn sample_indices(&mut self, n: usize, count: usize) -> Vec<usize> {
        let count = count.min(n);
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..count {
            let j = i + ((self.next_f64() * (n - i) as f64) as usize).min(n - i - 1);
            indices.swap(i, j);
        }
        indices.truncate(count);
        indices
    }
}

#[inline]
fn adaptive_init_sample_size(n: usize, k: usize) -> usize {
    n.min((k.saturating_mul(INIT_SAMPLE_PER_CENTROID)).clamp(INIT_SAMPLE_MIN, INIT_SAMPLE_MAX))
}

pub(crate) fn train_l2(
    data: &[f32],
    n_vectors: usize,
    dim: usize,
    requested_centroids: usize,
    max_iter: usize,
) -> KMeansResult {
    let n_centroids = requested_centroids.min(n_vectors);
    if n_vectors == 0 || n_centroids == 0 || dim == 0 {
        return KMeansResult {
            centroids: Vec::new(),
            assignments: Vec::new(),
            n_centroids: 0,
        };
    }

    let data = &data[..n_vectors * dim];
    let mut centroids = kmeans_pp_init_l2(data, n_vectors, dim, n_centroids);
    let mut assignments = vec![usize::MAX; n_vectors];

    for _ in 0..max_iter {
        let new_assignments = assign_l2(data, &centroids, dim, n_centroids);
        let changed = new_assignments
            .iter()
            .zip(&assignments)
            .any(|(a, b)| a != b);
        assignments = new_assignments;

        let (sums, counts) = accumulate_centroid_sums(data, &assignments, dim, n_centroids);
        let (max_c, max_count) = counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, count)| count)
            .map(|(c, &count)| (c, count))
            .unwrap_or((0, 0));

        for c in 0..n_centroids {
            let c_off = c * dim;
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for d in 0..dim {
                    centroids[c_off + d] = sums[c_off + d] * inv;
                }
            } else if max_count > 1 {
                let src_off = max_c * dim;
                for d in 0..dim {
                    centroids[c_off + d] = centroids[src_off + d] * (1.0 + 1e-4 * d as f32);
                }
            }
        }

        if !changed {
            break;
        }
    }

    let assignments = assign_l2(data, &centroids, dim, n_centroids);
    KMeansResult {
        centroids,
        assignments,
        n_centroids,
    }
}

fn kmeans_pp_init_l2(data: &[f32], n_vectors: usize, dim: usize, k: usize) -> Vec<f32> {
    let mut rng = FastRng::new(42);
    let sample_n = adaptive_init_sample_size(n_vectors, k);
    let sample_indices = if sample_n >= n_vectors {
        (0..n_vectors).collect::<Vec<_>>()
    } else {
        rng.sample_indices(n_vectors, sample_n)
    };

    let mut sample_data = vec![0.0f32; sample_indices.len() * dim];
    for (si, &orig_i) in sample_indices.iter().enumerate() {
        sample_data[si * dim..(si + 1) * dim]
            .copy_from_slice(&data[orig_i * dim..(orig_i + 1) * dim]);
    }
    let sample_n = sample_indices.len();

    let mut centroids = vec![0.0f32; k * dim];
    let first = (rng.next_f64() * sample_n as f64) as usize % sample_n;
    centroids[..dim].copy_from_slice(&sample_data[first * dim..(first + 1) * dim]);

    let mut min_dists = vec![f32::MAX; sample_n];
    for c in 1..k {
        {
            let prev = c - 1;
            let centroid = &centroids[prev * dim..(prev + 1) * dim];
            min_dists.par_iter_mut().enumerate().for_each(|(i, min_d)| {
                let vec = &sample_data[i * dim..(i + 1) * dim];
                let d = compute_distance_f32(vec, centroid, DistanceMetric::L2Squared);
                if d < *min_d {
                    *min_d = d;
                }
            });
        }

        let best = min_dists
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        centroids[c * dim..(c + 1) * dim]
            .copy_from_slice(&sample_data[best * dim..(best + 1) * dim]);
    }

    centroids
}

pub(crate) fn centroid_half_norms(centroids: &[f32], dim: usize, n_centroids: usize) -> Vec<f32> {
    (0..n_centroids)
        .map(|c| {
            let centroid = &centroids[c * dim..(c + 1) * dim];
            centroid.iter().map(|&v| v * v).sum::<f32>() * 0.5
        })
        .collect()
}

pub(crate) fn nearest_l2_centroid(
    vector: &[f32],
    centroids: &[f32],
    half_norms: &[f32],
    dim: usize,
    n_centroids: usize,
) -> usize {
    let mut best_c = 0usize;
    let mut best_val = f32::MAX;
    for c in 0..n_centroids {
        let centroid = &centroids[c * dim..(c + 1) * dim];
        let dot = compute_distance_f32(vector, centroid, DistanceMetric::InnerProduct);
        let val = half_norms[c] - dot;
        if val < best_val {
            best_val = val;
            best_c = c;
        }
    }
    best_c
}

pub(crate) fn assign_l2(
    data: &[f32],
    centroids: &[f32],
    dim: usize,
    n_centroids: usize,
) -> Vec<usize> {
    let n_vectors = data.len() / dim;
    let half_norms = centroid_half_norms(centroids, dim, n_centroids);
    (0..n_vectors)
        .into_par_iter()
        .map(|i| {
            let vector = &data[i * dim..(i + 1) * dim];
            nearest_l2_centroid(vector, centroids, &half_norms, dim, n_centroids)
        })
        .collect()
}

fn accumulate_centroid_sums(
    data: &[f32],
    assignments: &[usize],
    dim: usize,
    n_centroids: usize,
) -> (Vec<f32>, Vec<u32>) {
    let n_vectors = assignments.len();
    if n_vectors < 8_192 {
        let mut sums = vec![0.0f32; n_centroids * dim];
        let mut counts = vec![0u32; n_centroids];
        for i in 0..n_vectors {
            let c = assignments[i];
            counts[c] += 1;
            let x_off = i * dim;
            let c_off = c * dim;
            for d in 0..dim {
                sums[c_off + d] += data[x_off + d];
            }
        }
        return (sums, counts);
    }

    (0..n_vectors)
        .into_par_iter()
        .fold(
            || (vec![0.0f32; n_centroids * dim], vec![0u32; n_centroids]),
            |mut acc, i| {
                let c = assignments[i];
                acc.1[c] += 1;
                let x_off = i * dim;
                let c_off = c * dim;
                for d in 0..dim {
                    acc.0[c_off + d] += data[x_off + d];
                }
                acc
            },
        )
        .reduce(
            || (vec![0.0f32; n_centroids * dim], vec![0u32; n_centroids]),
            |mut a, b| {
                for (dst, src) in a.0.iter_mut().zip(b.0) {
                    *dst += src;
                }
                for (dst, src) in a.1.iter_mut().zip(b.1) {
                    *dst += src;
                }
                a
            },
        )
}

pub(crate) fn inverted_lists_from_assignments(
    assignments: &[usize],
    n_centroids: usize,
) -> HashMap<usize, Vec<usize>> {
    let mut counts = vec![0usize; n_centroids];
    for &c in assignments {
        counts[c] += 1;
    }

    let mut lists: Vec<Vec<usize>> = counts
        .iter()
        .map(|&count| Vec::with_capacity(count))
        .collect();
    for (idx, &c) in assignments.iter().enumerate() {
        lists[c].push(idx);
    }

    lists
        .into_iter()
        .enumerate()
        .filter_map(|(cluster, ids)| {
            if ids.is_empty() {
                None
            } else {
                Some((cluster, ids))
            }
        })
        .collect()
}
