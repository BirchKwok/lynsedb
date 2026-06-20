//! Distance computation module with SIMD acceleration.
//!
//! Supports embedding, numeric, geospatial, profile, distribution, and packed
//! binary distances.
//! Designed for high-dimensional vectors (700-3500 dims) at billion scale.
//!
//! Top-k search strategy:
//!   1. Batch compute all distances (cache-friendly, SIMD-pipelined)
//!   2. Quickselect (introselect) for O(n) partial sort — matches numpy.argpartition
//!   3. Parallel distance computation via rayon for large datasets (>8k vectors)

pub mod simd;

use rayon::prelude::*;
use std::cmp::Ordering;

/// Distance metric types matching the Python API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DistanceMetric {
    InnerProduct,
    L2Squared,
    Cosine,
    Hamming,
    Jaccard,
    Manhattan,
    Haversine,
    Correlation,
    Hellinger,
    Wasserstein,
    Dice,
    Tanimoto,
    JensenShannon,
    Chebyshev,
    Canberra,
    BrayCurtis,
}

impl DistanceMetric {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ip" | "inner_product" | "inner" | "dot" => Some(Self::InnerProduct),
            "l2" | "l2sq" | "l2_squared" | "euclidean" => Some(Self::L2Squared),
            "cosine" | "cos" | "cosine_distance" => Some(Self::Cosine),
            "hamming" => Some(Self::Hamming),
            "jaccard" => Some(Self::Jaccard),
            "l1" | "manhattan" | "cityblock" => Some(Self::Manhattan),
            "haversine" | "haversine_m" | "haversine-m" | "geo" => Some(Self::Haversine),
            "correlation" | "pearson" => Some(Self::Correlation),
            "hellinger" => Some(Self::Hellinger),
            "wasserstein" | "wasserstein1d" | "wasserstein_1d" | "wasserstein-1d" | "emd" => {
                Some(Self::Wasserstein)
            }
            "dice" | "sorensen" | "sorensen_dice" | "sorensen-dice" => Some(Self::Dice),
            "tanimoto" => Some(Self::Tanimoto),
            "jensen_shannon" | "jensen-shannon" | "jensenshannon" | "js" => {
                Some(Self::JensenShannon)
            }
            "chebyshev" | "chebychev" | "linf" | "l_inf" | "l-infinity" => Some(Self::Chebyshev),
            "canberra" => Some(Self::Canberra),
            "bray_curtis" | "bray-curtis" | "braycurtis" => Some(Self::BrayCurtis),
            _ => None,
        }
    }

    /// Parse the metric token embedded in an index mode such as
    /// `HNSW-CORRELATION` or `FLAT-TANIMOTO-BINARY`.
    pub fn from_index_mode(mode: &str) -> Option<Self> {
        let upper = mode.to_ascii_uppercase();
        let tokens: Vec<&str> = upper.split('-').collect();
        let has = |value: &str| tokens.contains(&value);

        if has("JENSENSHANNON") || has("JS") || (has("JENSEN") && has("SHANNON")) {
            Some(Self::JensenShannon)
        } else if has("CHEBYSHEV") || has("CHEBYCHEV") || has("LINF") {
            Some(Self::Chebyshev)
        } else if has("CANBERRA") {
            Some(Self::Canberra)
        } else if has("BRAYCURTIS") || (has("BRAY") && has("CURTIS")) {
            Some(Self::BrayCurtis)
        } else if has("TANIMOTO") {
            Some(Self::Tanimoto)
        } else if has("JACCARD") {
            Some(Self::Jaccard)
        } else if has("HAMMING") {
            Some(Self::Hamming)
        } else if has("DICE") || has("SORENSEN") {
            Some(Self::Dice)
        } else if has("HAVERSINE") || has("GEO") {
            Some(Self::Haversine)
        } else if has("CORRELATION") || has("PEARSON") {
            Some(Self::Correlation)
        } else if has("HELLINGER") {
            Some(Self::Hellinger)
        } else if has("WASSERSTEIN") || has("WASSERSTEIN1D") || has("EMD") {
            Some(Self::Wasserstein)
        } else if has("L1") || has("MANHATTAN") || has("CITYBLOCK") {
            Some(Self::Manhattan)
        } else if has("L2") || has("L2SQ") {
            Some(Self::L2Squared)
        } else if has("COS") || has("COSINE") {
            Some(Self::Cosine)
        } else if has("IP") {
            Some(Self::InnerProduct)
        } else {
            None
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
            Self::Manhattan => "l1",
            Self::Haversine => "haversine",
            Self::Correlation => "correlation",
            Self::Hellinger => "hellinger",
            Self::Wasserstein => "wasserstein",
            Self::Dice => "dice",
            Self::Tanimoto => "tanimoto",
            Self::JensenShannon => "jensen_shannon",
            Self::Chebyshev => "chebyshev",
            Self::Canberra => "canberra",
            Self::BrayCurtis => "bray_curtis",
        }
    }

    pub fn flat_index_mode(&self) -> &'static str {
        match self {
            Self::InnerProduct => "FLAT-IP",
            Self::L2Squared => "FLAT-L2",
            Self::Cosine => "FLAT-COS",
            Self::Hamming => "FLAT-HAMMING-BINARY",
            Self::Jaccard => "FLAT-JACCARD-BINARY",
            Self::Manhattan => "FLAT-L1",
            Self::Haversine => "FLAT-HAVERSINE",
            Self::Correlation => "FLAT-CORRELATION",
            Self::Hellinger => "FLAT-HELLINGER",
            Self::Wasserstein => "FLAT-WASSERSTEIN",
            Self::Dice => "FLAT-DICE-BINARY",
            Self::Tanimoto => "FLAT-TANIMOTO-BINARY",
            Self::JensenShannon => "FLAT-JENSEN-SHANNON",
            Self::Chebyshev => "FLAT-CHEBYSHEV",
            Self::Canberra => "FLAT-CANBERRA",
            Self::BrayCurtis => "FLAT-BRAY-CURTIS",
        }
    }

    /// Metrics evaluated on packed one-bit rows in the flat-search hot path.
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            Self::Hamming | Self::Jaccard | Self::Dice | Self::Tanimoto
        )
    }

    /// Validate fixed dimensional requirements imposed by a metric.
    pub fn accepts_dimension(&self, dimension: usize) -> bool {
        match self {
            Self::Haversine => dimension == 2,
            _ => dimension > 0,
        }
    }

    /// Whether `search(..., approx=True)` has a metric-specific implementation.
    pub fn supports_flat_approx(&self) -> bool {
        matches!(self, Self::InnerProduct | Self::L2Squared | Self::Cosine)
    }
}

/// Compute distance between two f32 vectors.
#[inline(always)]
pub fn compute_distance_f32(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    match metric {
        DistanceMetric::InnerProduct => simd::inner_product_f32(a, b),
        DistanceMetric::L2Squared => simd::l2_squared_f32(a, b),
        DistanceMetric::Cosine => simd::cosine_distance_f32(a, b),
        DistanceMetric::Hamming => simd::hamming_f32(a, b),
        DistanceMetric::Jaccard => simd::jaccard_f32(a, b),
        DistanceMetric::Manhattan => simd::manhattan_f32(a, b),
        DistanceMetric::Haversine => simd::haversine_meters_f32(a, b),
        DistanceMetric::Correlation => simd::correlation_distance_f32(a, b),
        DistanceMetric::Hellinger => simd::hellinger_distance_f32(a, b),
        DistanceMetric::Wasserstein => simd::wasserstein_1d_f32(a, b),
        DistanceMetric::Dice => simd::dice_distance_f32(a, b),
        DistanceMetric::Tanimoto => simd::jaccard_f32(a, b),
        DistanceMetric::JensenShannon => simd::jensen_shannon_distance_f32(a, b),
        DistanceMetric::Chebyshev => simd::chebyshev_f32(a, b),
        DistanceMetric::Canberra => simd::canberra_f32(a, b),
        DistanceMetric::BrayCurtis => simd::bray_curtis_f32(a, b),
    }
}

/// Compute distance between an f32 query and an f16-encoded candidate row.
#[inline(always)]
pub fn compute_distance_f16(query: &[f32], candidate_bits: &[u16], metric: DistanceMetric) -> f32 {
    debug_assert_eq!(query.len(), candidate_bits.len());
    match metric {
        DistanceMetric::InnerProduct => simd::inner_product_f16(query, candidate_bits),
        DistanceMetric::L2Squared => simd::l2_squared_f16(query, candidate_bits),
        DistanceMetric::Cosine => simd::cosine_distance_f16(query, candidate_bits),
        DistanceMetric::Hamming => simd::hamming_f16(query, candidate_bits),
        DistanceMetric::Jaccard => simd::jaccard_f16(query, candidate_bits),
        DistanceMetric::Manhattan => simd::manhattan_f16(query, candidate_bits),
        DistanceMetric::Haversine => simd::haversine_meters_f16(query, candidate_bits),
        DistanceMetric::Correlation => simd::correlation_distance_f16(query, candidate_bits),
        DistanceMetric::Hellinger => simd::hellinger_distance_f16(query, candidate_bits),
        DistanceMetric::Wasserstein => simd::wasserstein_1d_f16(query, candidate_bits),
        DistanceMetric::Dice => simd::dice_distance_f16(query, candidate_bits),
        DistanceMetric::Tanimoto => simd::jaccard_f16(query, candidate_bits),
        DistanceMetric::JensenShannon => simd::jensen_shannon_distance_f16(query, candidate_bits),
        DistanceMetric::Chebyshev => simd::chebyshev_f16(query, candidate_bits),
        DistanceMetric::Canberra => simd::canberra_f16(query, candidate_bits),
        DistanceMetric::BrayCurtis => simd::bray_curtis_f16(query, candidate_bits),
    }
}

// ─── Batch distance computation ─────────────────────────────────────────────

/// Compute distances from one query to all candidates, writing into a pre-allocated buffer.
/// This is the hot inner loop — kept separate for cache-friendly batch access.
#[inline]
fn batch_distances(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    metric: DistanceMetric,
    out: &mut [f32],
) {
    let n = candidates.len() / dim;
    debug_assert!(out.len() >= n);
    for i in 0..n {
        unsafe {
            let start = i * dim;
            let cand = candidates.get_unchecked(start..start + dim);
            *out.get_unchecked_mut(i) = compute_distance_f32(query, cand, metric);
        }
    }
}

/// Parallel batch distance computation for large datasets.
/// Splits candidates into per-thread chunks and computes distances in parallel.
#[inline]
fn batch_distances_parallel(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    metric: DistanceMetric,
    out: &mut [f32],
) {
    let n = candidates.len() / dim;
    let n_threads = rayon::current_num_threads();
    let chunk_vecs = (n / n_threads).max(256);
    let chunk_floats = chunk_vecs * dim;

    // Parallel iterate over aligned (candidate_chunk, output_chunk) pairs
    candidates
        .chunks(chunk_floats)
        .zip(out[..n].chunks_mut(chunk_vecs))
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|(cand_chunk, out_chunk)| {
            let n_in_chunk = cand_chunk.len() / dim;
            for i in 0..n_in_chunk {
                unsafe {
                    let start = i * dim;
                    let cand = cand_chunk.get_unchecked(start..start + dim);
                    *out_chunk.get_unchecked_mut(i) = compute_distance_f32(query, cand, metric);
                }
            }
        });
}

// ─── Quickselect for top-k ───────────────────────────────────────────────────

/// Partition-based top-k selection: O(n) average.
/// After return, elements [0..k) are the k "best" (smallest for ascending,
/// largest for descending), but NOT necessarily sorted within [0..k).
pub(crate) fn quickselect_k_pub(arr: &mut [(f32, u32)], k: usize, ascending: bool) {
    quickselect_k(arr, k, ascending);
}

fn quickselect_k(arr: &mut [(f32, u32)], k: usize, ascending: bool) {
    let n = arr.len();
    if n <= k || k == 0 {
        return;
    }
    // We want the (k-1)-th element (0-indexed) to be in its final sorted position,
    // with all elements [0..k) <= arr[k-1] and all elements [k..n) >= arr[k-1].
    let target = k - 1;
    let mut lo = 0usize;
    let mut hi = n - 1; // inclusive

    while lo < hi {
        // Median-of-3 pivot when range is large enough
        if hi - lo >= 2 {
            let mid = lo + (hi - lo) / 2;
            if cmp_pair(arr[lo], arr[mid], ascending) == Ordering::Greater {
                arr.swap(lo, mid);
            }
            if cmp_pair(arr[lo], arr[hi], ascending) == Ordering::Greater {
                arr.swap(lo, hi);
            }
            if cmp_pair(arr[mid], arr[hi], ascending) == Ordering::Greater {
                arr.swap(mid, hi);
            }
            arr.swap(mid, hi);
        }

        // Lomuto partition with pivot at arr[hi]
        let pivot = arr[hi];
        let mut store = lo;
        for j in lo..hi {
            if cmp_pair(arr[j], pivot, ascending) != Ordering::Greater {
                arr.swap(store, j);
                store += 1;
            }
        }
        arr.swap(store, hi);

        // Pivot is now at `store`
        if store == target {
            return;
        } else if store < target {
            lo = store + 1;
        } else {
            // store > 0 guaranteed since store > target >= 0 implies store >= 1
            hi = store - 1;
        }
    }
}

/// Compare two (distance, index) pairs for quickselect ordering.
#[inline(always)]
fn cmp_pair(a: (f32, u32), b: (f32, u32), ascending: bool) -> Ordering {
    if ascending {
        a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
    } else {
        b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal)
    }
}

// ─── Top-k search ───────────────────────────────────────────────────────────

/// Threshold for switching from sequential to parallel distance computation.
const PARALLEL_THRESHOLD: usize = 8192;

/// Compute distances from a single query to all candidates, return top-k.
///
/// Strategy: batch compute all distances, then quickselect for O(n) top-k.
/// This matches Python's simsimd + argpartition approach.
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

    // Phase 1: Batch compute all distances
    let mut distances = vec![0.0f32; n_candidates];
    if n_candidates >= PARALLEL_THRESHOLD {
        batch_distances_parallel(query, candidates, dim, metric, &mut distances);
    } else {
        batch_distances(query, candidates, dim, metric, &mut distances);
    }

    // Phase 2: Build (distance, index) pairs
    let mut pairs: Vec<(f32, u32)> = distances
        .iter()
        .enumerate()
        .map(|(i, &d)| (d, i as u32))
        .collect();

    // Phase 3: Quickselect to get top-k in O(n) average
    quickselect_k(&mut pairs, k, ascending);

    // Phase 4: Sort only the top-k results (O(k log k))
    let top_k = &mut pairs[..k];
    if ascending {
        top_k.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    } else {
        top_k.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    }

    let mut indices = Vec::with_capacity(k);
    let mut dists = Vec::with_capacity(k);
    for &(d, idx) in top_k.iter() {
        indices.push(idx);
        dists.push(d);
    }

    (indices, dists)
}

/// Streaming top-k with heap — used for incremental/chunked search in engine.
/// O(n log k) time, O(k) memory. Useful when data arrives in chunks.
pub fn top_k_heap_merge(
    existing_ids: &[u64],
    existing_dists: &[f32],
    new_chunk_dists: &[f32],
    new_chunk_ids: &[u64],
    k: usize,
    ascending: bool,
) -> (Vec<u64>, Vec<f32>) {
    let total = existing_ids.len() + new_chunk_ids.len();
    let k = k.min(total);

    let mut pairs: Vec<(f32, u64)> = Vec::with_capacity(total);
    for i in 0..existing_ids.len() {
        pairs.push((existing_dists[i], existing_ids[i]));
    }
    for i in 0..new_chunk_ids.len() {
        pairs.push((new_chunk_dists[i], new_chunk_ids[i]));
    }

    // Quickselect
    let mut fpairs: Vec<(f32, u32)> = pairs
        .iter()
        .enumerate()
        .map(|(i, &(d, _))| (d, i as u32))
        .collect();
    quickselect_k(&mut fpairs, k, ascending);

    let top_k = &mut fpairs[..k];
    if ascending {
        top_k.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    } else {
        top_k.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    }

    let mut ids = Vec::with_capacity(k);
    let mut dists = Vec::with_capacity(k);
    for &(d, idx) in top_k.iter() {
        let (_, orig_id) = pairs[idx as usize];
        ids.push(orig_id);
        dists.push(d);
    }
    (ids, dists)
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
        let (ids, _dists) = top_k_search(&query, &candidates, 3, 2, DistanceMetric::L2Squared);
        assert_eq!(ids[0], 1); // smallest L2
    }

    #[test]
    fn test_top_k_larger() {
        // Test with moderate dataset (fast in debug mode)
        let dim = 16;
        let n = 1_000;
        let mut candidates = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                candidates[i * dim + j] = (i * dim + j) as f32 * 0.001;
            }
        }
        // Use L2 with query = first candidate → self-match has distance 0 (smallest)
        let query: Vec<f32> = candidates[..dim].to_vec();
        let (ids, dists) = top_k_search(&query, &candidates, dim, 5, DistanceMetric::L2Squared);
        assert_eq!(ids.len(), 5);
        assert_eq!(ids[0], 0); // self-match = L2 distance 0
        assert!(dists[0] < 1e-6);
        // Results should be sorted ascending by distance
        for w in dists.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn test_quickselect_basic() {
        let mut pairs: Vec<(f32, u32)> = vec![(5.0, 0), (1.0, 1), (3.0, 2), (2.0, 3), (4.0, 4)];
        quickselect_k(&mut pairs, 2, true);
        // After quickselect, the 2 smallest should be in [0..2]
        let top2: Vec<f32> = pairs[..2].iter().map(|p| p.0).collect();
        assert!(top2.contains(&1.0));
        assert!(top2.contains(&2.0));
    }

    #[test]
    fn test_batch_search() {
        let queries = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let candidates = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let (all_ids, _all_dists) =
            batch_top_k_search(&queries, &candidates, 3, 1, DistanceMetric::L2Squared);
        assert_eq!(all_ids.len(), 2);
    }

    #[test]
    fn test_top_k_empty_and_zero_k_return_empty() {
        let query = vec![1.0f32, 2.0];
        let empty_candidates: Vec<f32> = Vec::new();

        let (ids, dists) = top_k_search(&query, &empty_candidates, 2, 5, DistanceMetric::L2Squared);
        assert!(ids.is_empty());
        assert!(dists.is_empty());

        let candidates = vec![1.0f32, 2.0, 3.0, 4.0];
        let (ids, dists) = top_k_search(&query, &candidates, 2, 0, DistanceMetric::L2Squared);
        assert!(ids.is_empty());
        assert!(dists.is_empty());
    }

    #[test]
    fn test_top_k_clamps_k_to_candidate_count() {
        let query = vec![0.0f32, 0.0];
        let candidates = vec![
            2.0, 0.0, // dist=4
            1.0, 0.0, // dist=1
        ];

        let (ids, dists) = top_k_search(&query, &candidates, 2, 10, DistanceMetric::L2Squared);

        assert_eq!(ids, vec![1, 0]);
        assert_eq!(dists.len(), 2);
        assert!(dists[0] <= dists[1]);
    }

    #[test]
    fn test_top_k_binary_metrics_are_sorted_by_lower_distance() {
        let query = vec![1.0f32, 0.0, 1.0, 0.0];
        let candidates = vec![
            1.0, 0.0, 1.0, 0.0, // hamming=0, jaccard=0
            1.0, 1.0, 1.0, 0.0, // hamming=1, jaccard=1/3
            0.0, 1.0, 0.0, 1.0, // hamming=4, jaccard=1
        ];

        let (hamming_ids, hamming_dists) =
            top_k_search(&query, &candidates, 4, 3, DistanceMetric::Hamming);
        assert_eq!(hamming_ids, vec![0, 1, 2]);
        assert_eq!(hamming_dists, vec![0.0, 1.0, 4.0]);

        let (jaccard_ids, jaccard_dists) =
            top_k_search(&query, &candidates, 4, 3, DistanceMetric::Jaccard);
        assert_eq!(jaccard_ids, vec![0, 1, 2]);
        assert!((jaccard_dists[0] - 0.0).abs() < 1e-6);
        assert!((jaccard_dists[1] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((jaccard_dists[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quickselect_handles_zero_k_and_descending() {
        let original: Vec<(f32, u32)> = vec![(1.0, 0), (5.0, 1), (3.0, 2), (2.0, 3)];
        let mut zero_k = original.clone();
        quickselect_k(&mut zero_k, 0, true);
        assert_eq!(zero_k, original);

        let mut descending = original;
        quickselect_k(&mut descending, 2, false);
        let top2: Vec<f32> = descending[..2].iter().map(|p| p.0).collect();
        assert!(top2.contains(&5.0));
        assert!(top2.contains(&3.0));
    }

    #[test]
    fn test_heap_merge_handles_empty_existing_and_descending_scores() {
        let (ids, dists) = top_k_heap_merge(&[], &[], &[0.2, 0.9, 0.4], &[20, 90, 40], 2, false);

        assert_eq!(ids, vec![90, 40]);
        assert_eq!(dists, vec![0.9, 0.4]);
    }

    #[test]
    fn test_distance_metric_aliases_and_capabilities() {
        assert_eq!(
            DistanceMetric::from_str("DOT"),
            Some(DistanceMetric::InnerProduct)
        );
        assert_eq!(
            DistanceMetric::from_str("euclidean"),
            Some(DistanceMetric::L2Squared)
        );
        assert_eq!(
            DistanceMetric::from_str("cosine_distance"),
            Some(DistanceMetric::Cosine)
        );
        assert_eq!(DistanceMetric::from_str("unknown"), None);
        assert_eq!(
            DistanceMetric::from_str("cityblock"),
            Some(DistanceMetric::Manhattan)
        );
        assert_eq!(
            DistanceMetric::from_str("pearson"),
            Some(DistanceMetric::Correlation)
        );
        assert_eq!(
            DistanceMetric::from_str("emd"),
            Some(DistanceMetric::Wasserstein)
        );
        assert_eq!(
            DistanceMetric::from_str("js"),
            Some(DistanceMetric::JensenShannon)
        );
        assert_eq!(
            DistanceMetric::from_str("linf"),
            Some(DistanceMetric::Chebyshev)
        );
        assert_eq!(
            DistanceMetric::from_str("bray-curtis"),
            Some(DistanceMetric::BrayCurtis)
        );
        assert_eq!(
            DistanceMetric::from_index_mode("FLAT-CANBERRA"),
            Some(DistanceMetric::Canberra)
        );
        assert_eq!(
            DistanceMetric::from_index_mode("FLAT-TANIMOTO-BINARY"),
            Some(DistanceMetric::Tanimoto)
        );
        assert_eq!(DistanceMetric::from_index_mode("FLAT-BOGUS"), None);

        assert!(DistanceMetric::Cosine.supports_flat_approx());
        assert!(!DistanceMetric::Hamming.supports_flat_approx());
        assert!(DistanceMetric::Dice.is_binary());
        assert!(DistanceMetric::Haversine.accepts_dimension(2));
        assert!(!DistanceMetric::Haversine.accepts_dimension(3));
    }
}
