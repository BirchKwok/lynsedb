//! Distance computation module with SIMD acceleration.
//!
//! Supports: Inner Product, L2 Squared, Cosine, Hamming, Jaccard.
//! Designed for high-dimensional vectors (700-3500 dims) at billion scale.
//!
//! Top-k search uses streaming heap selection (O(n log k), O(k) memory)
//! with chunked parallel for large datasets.

pub mod simd;

use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Distance metric types matching the Python API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DistanceMetric {
    InnerProduct,
    L2Squared,
    Cosine,
    Hamming,
    Jaccard,
}

impl DistanceMetric {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ip" | "inner_product" | "inner" | "dot" => Some(Self::InnerProduct),
            "l2" | "l2sq" | "l2_squared" | "euclidean" => Some(Self::L2Squared),
            "cosine" | "cos" | "cosine_distance" => Some(Self::Cosine),
            "hamming" => Some(Self::Hamming),
            "jaccard" => Some(Self::Jaccard),
            _ => None,
        }
    }

    /// Whether lower distance means more similar (true for L2, Cosine, Hamming, Jaccard)
    /// For IP, higher is more similar.
    pub fn is_ascending(&self) -> bool {
        match self {
            Self::InnerProduct => false, // higher IP = more similar
            _ => true,                   // lower distance = more similar
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::InnerProduct => "ip",
            Self::L2Squared => "l2",
            Self::Cosine => "cosine",
            Self::Hamming => "hamming",
            Self::Jaccard => "jaccard",
        }
    }
}

/// Compute distance between two f32 vectors.
#[inline]
pub fn compute_distance_f32(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    match metric {
        DistanceMetric::InnerProduct => simd::inner_product_f32(a, b),
        DistanceMetric::L2Squared => simd::l2_squared_f32(a, b),
        DistanceMetric::Cosine => simd::cosine_distance_f32(a, b),
        DistanceMetric::Hamming => simd::hamming_f32(a, b),
        DistanceMetric::Jaccard => simd::jaccard_f32(a, b),
    }
}

// ─── f32-Ord wrapper for BinaryHeap ─────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
struct FloatOrd(f32);
impl Eq for FloatOrd {}
impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

// ─── Top-k search ───────────────────────────────────────────────────────────

/// Compute distances from a single query to all candidates, return top-k.
///
/// Uses streaming heap selection — O(n log k) time, O(k) memory.
/// For large datasets, uses chunked parallel with per-thread heaps + merge.
pub fn top_k_search(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    metric: DistanceMetric,
) -> (Vec<u32>, Vec<f32>) {
    let n_candidates = candidates.len() / dim;
    let k = k.min(n_candidates);
    if n_candidates == 0 || k == 0 {
        return (vec![], vec![]);
    }

    let ascending = metric.is_ascending();

    if n_candidates >= 16384 {
        top_k_parallel(query, candidates, dim, k, metric, ascending)
    } else {
        top_k_streaming(query, candidates, dim, k, metric, ascending)
    }
}

/// Sequential streaming: compute distance + heap select in one pass.
/// Zero intermediate allocation — only O(k) heap memory.
fn top_k_streaming(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    metric: DistanceMetric,
    ascending: bool,
) -> (Vec<u32>, Vec<f32>) {
    let n = candidates.len() / dim;
    // Normalize all distances: for descending (IP), negate so max-heap evicts correctly.
    // max-heap always evicts the "largest key"; for ascending that's the worst (correct).
    // For descending, negate so largest negated = smallest original = worst (correct).
    let mut heap: BinaryHeap<(FloatOrd, u32)> = BinaryHeap::with_capacity(k + 1);

    for i in 0..n {
        let raw = compute_distance_f32(query, &candidates[i * dim..(i + 1) * dim], metric);
        let key = if ascending { raw } else { -raw };

        if heap.len() < k {
            heap.push((FloatOrd(key), i as u32));
        } else if key < heap.peek().unwrap().0 .0 {
            // Replace worst element (O(log k) sift-down via PeekMut drop)
            *heap.peek_mut().unwrap() = (FloatOrd(key), i as u32);
        }
    }

    heap_to_sorted(heap, k, ascending)
}

/// Parallel chunked: each thread maintains a local top-k heap, then merge.
/// Memory: O(k × num_threads). No intermediate Vec<f32> allocation.
fn top_k_parallel(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    metric: DistanceMetric,
    ascending: bool,
) -> (Vec<u32>, Vec<f32>) {
    let n = candidates.len() / dim;
    let chunk_vectors = (n / rayon::current_num_threads()).max(512);
    let chunk_floats = chunk_vectors * dim;

    // Each chunk produces a local top-k heap
    let local_results: Vec<Vec<(FloatOrd, u32)>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, batch)| {
            let base = (chunk_idx * chunk_vectors) as u32;
            let n_batch = batch.len() / dim;
            let mut heap: BinaryHeap<(FloatOrd, u32)> = BinaryHeap::with_capacity(k + 1);

            for i in 0..n_batch {
                let raw = compute_distance_f32(query, &batch[i * dim..(i + 1) * dim], metric);
                let key = if ascending { raw } else { -raw };
                let idx = base + i as u32;

                if heap.len() < k {
                    heap.push((FloatOrd(key), idx));
                } else if key < heap.peek().unwrap().0 .0 {
                    *heap.peek_mut().unwrap() = (FloatOrd(key), idx);
                }
            }

            heap.into_vec()
        })
        .collect();

    // Merge local heaps into global top-k
    let mut global: BinaryHeap<(FloatOrd, u32)> = BinaryHeap::with_capacity(k + 1);
    for local in local_results {
        for entry in local {
            if global.len() < k {
                global.push(entry);
            } else if entry.0 .0 < global.peek().unwrap().0 .0 {
                *global.peek_mut().unwrap() = entry;
            }
        }
    }

    heap_to_sorted(global, k, ascending)
}

/// Extract sorted results from a max-heap.
fn heap_to_sorted(
    heap: BinaryHeap<(FloatOrd, u32)>,
    k: usize,
    ascending: bool,
) -> (Vec<u32>, Vec<f32>) {
    // into_sorted_vec returns ascending order by FloatOrd
    let sorted = heap.into_sorted_vec();
    let mut indices = Vec::with_capacity(k);
    let mut dists = Vec::with_capacity(k);

    for (FloatOrd(key), idx) in sorted {
        indices.push(idx);
        dists.push(if ascending { key } else { -key });
    }
    (indices, dists)
}

/// Batch top-k search: multiple queries against same candidate set.
pub fn batch_top_k_search(
    queries: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    metric: DistanceMetric,
) -> (Vec<Vec<u32>>, Vec<Vec<f32>>) {
    let n_queries = queries.len() / dim;

    let results: Vec<(Vec<u32>, Vec<f32>)> = (0..n_queries)
        .into_par_iter()
        .map(|q| {
            let q_start = q * dim;
            let q_end = q_start + dim;
            top_k_search(&queries[q_start..q_end], candidates, dim, k, metric)
        })
        .collect();

    let mut all_ids = Vec::with_capacity(n_queries);
    let mut all_dists = Vec::with_capacity(n_queries);
    for (ids, dists) in results {
        all_ids.push(ids);
        all_dists.push(dists);
    }
    (all_ids, all_dists)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_ip() {
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        // 3 candidates, dim=4
        let candidates = vec![
            1.0, 0.0, 0.0, 0.0, // exactly query → IP=1.0
            0.5, 0.5, 0.0, 0.0, // IP=0.5
            0.0, 1.0, 0.0, 0.0, // IP=0.0
        ];
        let (ids, dists) = top_k_search(&query, &candidates, 4, 2, DistanceMetric::InnerProduct);
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], 0); // highest IP
        assert!((dists[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_l2() {
        let query = vec![0.0f32, 0.0, 0.0];
        let candidates = vec![
            1.0, 0.0, 0.0, // dist=1.0
            0.1, 0.0, 0.0, // dist=0.01
            2.0, 0.0, 0.0, // dist=4.0
        ];
        let (ids, dists) = top_k_search(&query, &candidates, 3, 2, DistanceMetric::L2Squared);
        assert_eq!(ids[0], 1); // smallest L2
    }

    #[test]
    fn test_batch_search() {
        let queries = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let candidates = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let (all_ids, all_dists) =
            batch_top_k_search(&queries, &candidates, 3, 1, DistanceMetric::L2Squared);
        assert_eq!(all_ids.len(), 2);
    }
}
