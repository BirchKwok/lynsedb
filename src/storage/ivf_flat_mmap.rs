//! IVF_FLAT index on top of mmap'd flat storage.
//!
//! Architecture (mirrors Lance's approach):
//! 1. **Build**: KMeans clustering → partition vectors → reorder contiguously
//! 2. **Search**: query→centroid distances → select top-nprobe → brute-force within partitions
//!
//! Files on disk:
//! - `{base}.ivf_data.bin`  — reordered f32 vectors (partitions contiguous)
//! - `{base}.ivf_meta.bin`  — centroids + partition offsets + original ID mapping
//!
//! Target: < 0.5ms for 1M×128 IP top-10 with nprobe=10, 256 partitions.

use crate::distance::{self, DistanceMetric};
use crate::storage::dtype::VectorDtype;
use crate::storage::flat_mmap::FlatMmap;
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// IVF_FLAT index with mmap-backed partition storage.
pub struct IvfFlatMmap {
    /// Underlying flat storage for the reordered data.
    flat: FlatMmap,
    dim: usize,
    n_vectors: usize,
    n_partitions: usize,
    /// Centroids: n_partitions × dim (contiguous f32).
    centroids: Vec<f32>,
    /// partition_offsets[i] = start vector index of partition i in the reordered data.
    /// Length: n_partitions + 1 (last element = n_vectors).
    partition_offsets: Vec<usize>,
    /// Mapping from reordered index → original vector ID.
    original_ids: Vec<u32>,
    /// High-variance centroid dimensions used for fast Inner Product routing.
    routing_dims: Vec<usize>,
    /// Path base for metadata file.
    meta_path: PathBuf,
}

const IP_ROUTING_DIMS: usize = 16;
const IP_ROUTING_MIN_DIM: usize = 64;
const IP_ROUTING_MIN_CENTROIDS: usize = 64;
const IP_ASSIGN_SHORTLIST: usize = 24;
const IP_QUERY_SHORTLIST_CAP: usize = 96;

impl IvfFlatMmap {
    /// Build an IVF_FLAT index from raw f32 data.
    ///
    /// - `data_path`: path for the reordered data file
    /// - `data`: flat f32 array (n_vectors × dim)
    /// - `dim`: vector dimension
    /// - `n_partitions`: number of IVF partitions (e.g., 256)
    /// - `n_iters`: KMeans iterations (e.g., 20)
    /// - `metric`: distance metric used for search (KMeans uses matching metric)
    pub fn build(
        data_path: &Path,
        data: &[f32],
        dim: usize,
        n_partitions: usize,
        n_iters: usize,
        metric: DistanceMetric,
    ) -> std::io::Result<Self> {
        let n_vectors = data.len() / dim;
        assert!(
            n_vectors >= n_partitions,
            "need at least n_partitions vectors"
        );

        // Step 1: KMeans clustering (metric-aware)
        let centroids = kmeans(data, dim, n_partitions, n_iters, metric);
        let routing_dims = select_routing_dims(&centroids, dim, n_partitions);

        // Step 2: Assign each vector to nearest centroid (metric-aware)
        let assignments = assign_partitions(data, &centroids, dim, n_partitions, metric);

        // Step 3: Compute partition sizes and offsets
        let mut partition_sizes = vec![0usize; n_partitions];
        for &p in &assignments {
            partition_sizes[p as usize] += 1;
        }
        let mut partition_offsets = vec![0usize; n_partitions + 1];
        for i in 0..n_partitions {
            partition_offsets[i + 1] = partition_offsets[i] + partition_sizes[i];
        }

        // Step 4: Reorder vectors by partition (contiguous storage)
        let mut reordered = vec![0.0f32; n_vectors * dim];
        let mut original_ids = vec![0u32; n_vectors];
        let mut write_pos = partition_offsets.clone(); // current write position per partition

        for vec_idx in 0..n_vectors {
            let p = assignments[vec_idx] as usize;
            let dst_idx = write_pos[p];
            write_pos[p] += 1;

            // Copy vector
            let src = &data[vec_idx * dim..(vec_idx + 1) * dim];
            let dst = &mut reordered[dst_idx * dim..(dst_idx + 1) * dim];
            dst.copy_from_slice(src);
            original_ids[dst_idx] = vec_idx as u32;
        }

        // Step 5: Write reordered data to flat file
        let mut flat = FlatMmap::open(data_path, dim, VectorDtype::F32)?;
        flat.write(&reordered)?;

        // Step 6: Save metadata
        let meta_path = data_path.with_extension("ivf_meta.bin");
        save_metadata(
            &meta_path,
            dim,
            n_vectors,
            n_partitions,
            &centroids,
            &partition_offsets,
            &original_ids,
        )?;

        Ok(Self {
            flat,
            dim,
            n_vectors,
            n_partitions,
            centroids,
            partition_offsets,
            original_ids,
            routing_dims,
            meta_path,
        })
    }

    /// Open an existing IVF_FLAT index.
    pub fn open(data_path: &Path, dim: usize) -> std::io::Result<Self> {
        let flat = FlatMmap::open(data_path, dim, VectorDtype::F32)?;
        let meta_path = data_path.with_extension("ivf_meta.bin");

        let (dim_loaded, n_vectors, n_partitions, centroids, partition_offsets, original_ids) =
            load_metadata(&meta_path)?;
        let routing_dims = select_routing_dims(&centroids, dim_loaded, n_partitions);

        assert_eq!(dim, dim_loaded, "dimension mismatch");

        Ok(Self {
            flat,
            dim,
            n_vectors,
            n_partitions,
            centroids,
            partition_offsets,
            original_ids,
            routing_dims,
            meta_path,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn n_partitions(&self) -> usize {
        self.n_partitions
    }

    /// Suggested nprobe for ≥95% recall@10.
    ///
    /// Heuristic based on SIFT1M benchmarks:
    /// - k=256, nprobe=10 → 97.2%
    /// - k=512, nprobe=20 → 98.4%
    /// - k=1024, nprobe=40 → 99.0%
    ///
    /// Rule: ~8% of partitions, clamped to [10, n_partitions].
    #[inline]
    pub fn suggested_nprobe(&self) -> usize {
        let np = (self.n_partitions / 13).max(10);
        np.min(self.n_partitions)
    }

    /// IVF_FLAT search: scan only the nprobe nearest partitions.
    ///
    /// Returns (original_indices, distances) sorted by relevance.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        metric: DistanceMetric,
    ) -> (Vec<u32>, Vec<f32>) {
        if self.n_vectors == 0 || k == 0 {
            return (vec![], vec![]);
        }
        let k = k.min(self.n_vectors);
        let nprobe = nprobe.min(self.n_partitions);
        let ascending = metric.is_ascending();

        // Phase 1: Find nprobe nearest centroids
        let nearest_partitions = find_nearest_centroids(
            query,
            &self.centroids,
            self.dim,
            self.n_partitions,
            nprobe,
            metric,
            &self.routing_dims,
        );

        // Phase 2: Brute-force search within selected partitions
        let all_data = self.flat.as_slice();
        let dim = self.dim;

        let mut best: Vec<(f32, u32)> = Vec::new(); // (dist, original_id)

        for &part_id in &nearest_partitions {
            let start = self.partition_offsets[part_id];
            let end = self.partition_offsets[part_id + 1];
            let n_in_part = end - start;
            if n_in_part == 0 {
                continue;
            }

            let part_data = &all_data[start * dim..end * dim];

            // Compute distances for this partition
            for i in 0..n_in_part {
                let dist = unsafe {
                    let s = i * dim;
                    let cand = part_data.get_unchecked(s..s + dim);
                    distance::compute_distance_f32(query, cand, metric)
                };
                let orig_id = self.original_ids[start + i];
                best.push((dist, orig_id));
            }
        }

        if best.is_empty() {
            return (vec![], vec![]);
        }

        // Top-k selection via select_nth_unstable_by
        let k = k.min(best.len());
        if ascending {
            best.select_nth_unstable_by(k - 1, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            best.select_nth_unstable_by(k - 1, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let top = &mut best[..k];
        if ascending {
            top.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            top.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        }

        let ids = top.iter().map(|&(_, id)| id).collect();
        let dists = top.iter().map(|&(d, _)| d).collect();
        (ids, dists)
    }
}

// ─── KMeans (metric-aware + fast KMeans++ init) ─────────────────────────────

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
fn should_use_ip_routing(metric: DistanceMetric, dim: usize, k: usize) -> bool {
    metric == DistanceMetric::InnerProduct
        && dim >= IP_ROUTING_MIN_DIM
        && k >= IP_ROUTING_MIN_CENTROIDS
}

fn select_routing_dims(centroids: &[f32], dim: usize, k: usize) -> Vec<usize> {
    if k == 0 || dim == 0 || !should_use_ip_routing(DistanceMetric::InnerProduct, dim, k) {
        return Vec::new();
    }

    let mut sums = vec![0.0f32; dim];
    let mut sq_sums = vec![0.0f32; dim];
    for c in 0..k {
        let off = c * dim;
        for d in 0..dim {
            let v = centroids[off + d];
            sums[d] += v;
            sq_sums[d] += v * v;
        }
    }

    let inv_k = 1.0 / k as f32;
    let keep = IP_ROUTING_DIMS.min(dim);
    let mut dims: Vec<(f32, usize)> = (0..dim)
        .map(|d| {
            let mean = sums[d] * inv_k;
            let variance = sq_sums[d] * inv_k - mean * mean;
            (variance, d)
        })
        .collect();

    dims.select_nth_unstable_by(keep - 1, |a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut selected: Vec<usize> = dims[..keep].iter().map(|&(_, d)| d).collect();
    selected.sort_unstable();
    selected
}

#[inline]
fn coarse_ip_score(vec: &[f32], centroid: &[f32], routing_dims: &[usize]) -> f32 {
    let mut score = 0.0f32;
    for &d in routing_dims {
        score += vec[d] * centroid[d];
    }
    score
}

#[inline]
fn shortlist_insert(best: &mut [(f32, u32)], len: &mut usize, score: f32, id: u32) {
    if *len < best.len() {
        best[*len] = (score, id);
        *len += 1;
        return;
    }

    let mut worst_idx = 0usize;
    let mut worst_score = best[0].0;
    for i in 1..best.len() {
        if best[i].0 < worst_score {
            worst_score = best[i].0;
            worst_idx = i;
        }
    }
    if score > worst_score {
        best[worst_idx] = (score, id);
    }
}

fn assign_ip_shortlist(
    data: &[f32],
    centroids: &[f32],
    dim: usize,
    k: usize,
    routing_dims: &[usize],
    shortlist: usize,
) -> Vec<u32> {
    let shortlist = shortlist.min(k).max(1);
    (0..data.len() / dim)
        .into_par_iter()
        .map(|i| {
            let vec = &data[i * dim..(i + 1) * dim];
            let mut candidates = vec![(f32::NEG_INFINITY, 0u32); shortlist];
            let mut cand_len = 0usize;

            for c in 0..k {
                let centroid = &centroids[c * dim..(c + 1) * dim];
                let coarse = coarse_ip_score(vec, centroid, routing_dims);
                shortlist_insert(&mut candidates, &mut cand_len, coarse, c as u32);
            }

            let mut best_c = 0u32;
            let mut best_dist = f32::MIN;
            for &(_, c_idx) in &candidates[..cand_len] {
                let centroid = &centroids[c_idx as usize * dim..(c_idx as usize + 1) * dim];
                let d = distance::compute_distance_f32(vec, centroid, DistanceMetric::InnerProduct);
                if d > best_dist {
                    best_dist = d;
                    best_c = c_idx;
                }
            }
            best_c
        })
        .collect()
}

/// MiniBatchKMeans with fast KMeans++ initialization.
///
/// Memory-efficient: only O(batch_size × dim) per iteration, no N-size allocations.
/// Prevents OOM on very large datasets while maintaining good clustering quality.
///
/// Optimizations:
/// 1. Sampled KMeans++ init (subsample ~50K points, incremental min_dist)
/// 2. L2 norm decomposition: ||x-c||² = ||c||² - 2·dot(x,c), saves 33% FLOPs
/// 3. Mini-batch incremental centroid update (constant memory per iteration)
/// 4. Parallel batch assignment via rayon
fn kmeans(data: &[f32], dim: usize, k: usize, n_iters: usize, metric: DistanceMetric) -> Vec<f32> {
    let n = data.len() / dim;
    let ascending = metric.is_ascending();

    // KMeans++ on subsample for fast init
    let mut centroids = kmeans_pp_init_sampled(data, dim, k, metric);

    // ── Phase 1: Fixed subsample iterations ──
    // Sample ONCE, reuse for all iterations to avoid inter-iteration centroid oscillation.
    // Memory: batch_size × dim × 4 bytes (~24MB at 50K × 128).
    let batch_size = n.min(50_000);
    let batch_data = if batch_size < n {
        let mut rng = FastRng::new(123);
        let indices = rng.sample_indices(n, batch_size);
        let mut buf = vec![0.0f32; batch_size * dim];
        for (bi, &idx) in indices.iter().enumerate() {
            buf[bi * dim..(bi + 1) * dim].copy_from_slice(&data[idx * dim..(idx + 1) * dim]);
        }
        buf
    } else {
        data.to_vec()
    };
    let use_ip_routing = should_use_ip_routing(metric, dim, k);

    // Subsample iterations: fast convergence on fixed subset
    let subsample_iters = if batch_size < n {
        n_iters.saturating_sub(2)
    } else {
        n_iters
    };
    for _iter in 0..subsample_iters {
        let assignments = if use_ip_routing {
            let routing_dims = select_routing_dims(&centroids, dim, k);
            assign_ip_shortlist(
                &batch_data,
                &centroids,
                dim,
                k,
                &routing_dims,
                IP_ASSIGN_SHORTLIST,
            )
        } else if metric == DistanceMetric::L2Squared {
            assign_l2_decomposed(&batch_data, &centroids, dim, k)
        } else {
            assign_partitions_inner(&batch_data, &centroids, dim, k, metric, ascending)
        };

        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0u32; k];
        for bi in 0..batch_size {
            let c = assignments[bi] as usize;
            counts[c] += 1;
            let x_off = bi * dim;
            let c_off = c * dim;
            for d in 0..dim {
                sums[c_off + d] += batch_data[x_off + d];
            }
        }
        // Find the largest cluster for dead centroid re-initialization
        let mut max_c = 0usize;
        let mut max_count = 0u32;
        for c in 0..k {
            if counts[c] > max_count {
                max_count = counts[c];
                max_c = c;
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                let c_off = c * dim;
                for d in 0..dim {
                    centroids[c_off + d] = sums[c_off + d] * inv;
                }
            } else if max_count > 1 {
                // Re-initialize dead centroid: perturb the largest cluster's centroid
                let c_off = c * dim;
                let src_off = max_c * dim;
                for d in 0..dim {
                    centroids[c_off + d] = centroids[src_off + d] * (1.0 + 1e-4 * (d as f32));
                }
            }
        }
    }

    // ── Phase 2: Chunked full-data refinement (OOM-safe) ──
    // Process ALL data in chunks to refine centroids to true means.
    // Memory per chunk: chunk_size × 4 bytes for assignments only.
    let refine_iters = if batch_size < n {
        2usize.min(n_iters)
    } else {
        0
    };
    let chunk_size = 100_000usize;

    for _refine in 0..refine_iters {
        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0u32; k];
        let routing_dims = if use_ip_routing {
            select_routing_dims(&centroids, dim, k)
        } else {
            Vec::new()
        };

        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);
            let chunk_data = &data[chunk_start * dim..chunk_end * dim];

            let chunk_assignments = if use_ip_routing {
                assign_ip_shortlist(
                    chunk_data,
                    &centroids,
                    dim,
                    k,
                    &routing_dims,
                    IP_ASSIGN_SHORTLIST,
                )
            } else if metric == DistanceMetric::L2Squared {
                assign_l2_decomposed(chunk_data, &centroids, dim, k)
            } else {
                assign_partitions_inner(chunk_data, &centroids, dim, k, metric, ascending)
            };

            for (i, &c_idx) in chunk_assignments.iter().enumerate() {
                let c = c_idx as usize;
                counts[c] += 1;
                let x_off = i * dim;
                let c_off = c * dim;
                let mut d = 0;
                while d + 3 < dim {
                    sums[c_off + d] += chunk_data[x_off + d];
                    sums[c_off + d + 1] += chunk_data[x_off + d + 1];
                    sums[c_off + d + 2] += chunk_data[x_off + d + 2];
                    sums[c_off + d + 3] += chunk_data[x_off + d + 3];
                    d += 4;
                }
                while d < dim {
                    sums[c_off + d] += chunk_data[x_off + d];
                    d += 1;
                }
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                let c_off = c * dim;
                for d in 0..dim {
                    centroids[c_off + d] = sums[c_off + d] * inv;
                }
            }
        }
    }

    centroids
}

/// Fast L2 assignment using norm decomposition + SIMD dot product.
///
/// ||x-c||² = ||x||² + ||c||² - 2·dot(x,c)
/// Since ||x||² is constant per vector, argmin only needs:
///   argmin_j { ||c_j||²/2 - dot(x, c_j) }
///
/// Uses SIMD inner product (2 ops/elem) instead of full L2 (3 ops/elem).
fn assign_l2_decomposed(data: &[f32], centroids: &[f32], dim: usize, k: usize) -> Vec<u32> {
    let n = data.len() / dim;

    // Precompute half centroid norms: ||c_j||² / 2
    let half_c_norms: Vec<f32> = (0..k)
        .map(|j| {
            let c = &centroids[j * dim..(j + 1) * dim];
            c.iter().map(|&v| v * v).sum::<f32>() * 0.5
        })
        .collect();

    (0..n)
        .into_par_iter()
        .map(|i| {
            let x = &data[i * dim..(i + 1) * dim];
            let mut best_c = 0u32;
            let mut best_val = f32::MAX;

            for j in 0..k {
                let c = &centroids[j * dim..(j + 1) * dim];
                // SIMD dot product — hand-tuned NEON, 2 ops/elem
                let dot = distance::compute_distance_f32(x, c, DistanceMetric::InnerProduct);
                let val = half_c_norms[j] - dot;
                if val < best_val {
                    best_val = val;
                    best_c = j as u32;
                }
            }
            best_c
        })
        .collect()
}

/// Fast KMeans++ initialization with two key optimizations:
/// 1. Run on a subsample (default 50K) — reduces data passes
/// 2. Incremental min_dist tracking — O(k×n_sample) instead of O(k²×n_sample)
fn kmeans_pp_init_sampled(data: &[f32], dim: usize, k: usize, metric: DistanceMetric) -> Vec<f32> {
    let n = data.len() / dim;
    let ascending = metric.is_ascending();
    let mut rng = FastRng::new(42);

    // Subsample for init (cap at 50K or full data if smaller)
    let sample_n = n.min(50_000);
    let sample_indices = if sample_n >= n {
        (0..n).collect::<Vec<_>>()
    } else {
        rng.sample_indices(n, sample_n)
    };

    // Gather sample data contiguously for cache efficiency
    let mut sample_data = vec![0.0f32; sample_indices.len() * dim];
    for (si, &orig_i) in sample_indices.iter().enumerate() {
        sample_data[si * dim..(si + 1) * dim]
            .copy_from_slice(&data[orig_i * dim..(orig_i + 1) * dim]);
    }
    let sn = sample_indices.len();

    let mut centroids = vec![0.0f32; k * dim];

    // Pick first centroid randomly from sample
    let first = (rng.next_f64() * sn as f64) as usize % sn;
    centroids[..dim].copy_from_slice(&sample_data[first * dim..(first + 1) * dim]);

    // min_dist[i] = distance from sample point i to NEAREST existing centroid
    // Initialize with distance to first centroid
    let mut min_dist: Vec<f32> = (0..sn)
        .into_par_iter()
        .map(|i| {
            let vec = &sample_data[i * dim..(i + 1) * dim];
            let d = distance::compute_distance_f32(vec, &centroids[..dim], metric);
            if ascending {
                d
            } else {
                1.0 - d
            } // convert to "farness" for IP
        })
        .collect();

    for ci in 1..k {
        // Weighted random selection (proportional to dist²)
        let total: f64 = min_dist
            .iter()
            .map(|&d| {
                let w = d.max(0.0) as f64;
                w * w
            })
            .sum();

        let chosen = if total <= 0.0 {
            // Fallback: random
            (rng.next_f64() * sn as f64) as usize % sn
        } else {
            let threshold = rng.next_f64() * total;
            let mut cumulative = 0.0;
            let mut pick = sn - 1;
            for i in 0..sn {
                let w = min_dist[i].max(0.0) as f64;
                cumulative += w * w;
                if cumulative >= threshold {
                    pick = i;
                    break;
                }
            }
            pick
        };

        centroids[ci * dim..(ci + 1) * dim]
            .copy_from_slice(&sample_data[chosen * dim..(chosen + 1) * dim]);

        // Incrementally update min_dist: only check new centroid (O(sn×dim))
        let new_cent = &centroids[ci * dim..(ci + 1) * dim];
        let new_dists: Vec<f32> = (0..sn)
            .into_par_iter()
            .map(|i| {
                let vec = &sample_data[i * dim..(i + 1) * dim];
                let d = distance::compute_distance_f32(vec, new_cent, metric);
                if ascending {
                    d
                } else {
                    1.0 - d
                }
            })
            .collect();

        // Update min_dist = min(old_min, new_dist) — element-wise
        for i in 0..sn {
            if new_dists[i] < min_dist[i] {
                min_dist[i] = new_dists[i];
            }
        }
    }

    centroids
}

/// Assign each vector to its nearest centroid (metric-aware, parallel).
fn assign_partitions(
    data: &[f32],
    centroids: &[f32],
    dim: usize,
    n_partitions: usize,
    metric: DistanceMetric,
) -> Vec<u32> {
    if should_use_ip_routing(metric, dim, n_partitions) {
        let routing_dims = select_routing_dims(centroids, dim, n_partitions);
        return assign_ip_shortlist(
            data,
            centroids,
            dim,
            n_partitions,
            &routing_dims,
            IP_ASSIGN_SHORTLIST,
        );
    }
    if metric == DistanceMetric::L2Squared {
        return assign_l2_decomposed(data, centroids, dim, n_partitions);
    }
    let ascending = metric.is_ascending();
    assign_partitions_inner(data, centroids, dim, n_partitions, metric, ascending)
}

fn assign_partitions_inner(
    data: &[f32],
    centroids: &[f32],
    dim: usize,
    n_partitions: usize,
    metric: DistanceMetric,
    ascending: bool,
) -> Vec<u32> {
    let n = data.len() / dim;
    (0..n)
        .into_par_iter()
        .map(|i| {
            let vec = &data[i * dim..(i + 1) * dim];
            let mut best_c = 0u32;
            let mut best_dist = if ascending { f32::MAX } else { f32::MIN };
            for c in 0..n_partitions {
                let centroid = &centroids[c * dim..(c + 1) * dim];
                let d = distance::compute_distance_f32(vec, centroid, metric);
                if ascending {
                    if d < best_dist {
                        best_dist = d;
                        best_c = c as u32;
                    }
                } else {
                    if d > best_dist {
                        best_dist = d;
                        best_c = c as u32;
                    }
                }
            }
            best_c
        })
        .collect()
}

/// Find nprobe nearest centroids to query.
fn find_nearest_centroids(
    query: &[f32],
    centroids: &[f32],
    dim: usize,
    n_partitions: usize,
    nprobe: usize,
    metric: DistanceMetric,
    routing_dims: &[usize],
) -> Vec<usize> {
    if nprobe >= n_partitions {
        return (0..n_partitions).collect();
    }

    if should_use_ip_routing(metric, dim, n_partitions) && !routing_dims.is_empty() {
        let shortlist = (nprobe * 3).clamp(IP_ASSIGN_SHORTLIST, IP_QUERY_SHORTLIST_CAP);
        let shortlist = shortlist.min(n_partitions);
        let mut coarse_best = vec![(f32::NEG_INFINITY, 0u32); shortlist];
        let mut coarse_len = 0usize;

        for c in 0..n_partitions {
            let centroid = &centroids[c * dim..(c + 1) * dim];
            let coarse = coarse_ip_score(query, centroid, routing_dims);
            shortlist_insert(&mut coarse_best, &mut coarse_len, coarse, c as u32);
        }

        let mut dists: Vec<(f32, usize)> = coarse_best[..coarse_len]
            .iter()
            .map(|&(_, c)| {
                let centroid = &centroids[c as usize * dim..(c as usize + 1) * dim];
                (
                    distance::compute_distance_f32(query, centroid, metric),
                    c as usize,
                )
            })
            .collect();

        dists.select_nth_unstable_by(nprobe - 1, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        return dists[..nprobe].iter().map(|&(_, c)| c).collect();
    }

    let mut dists: Vec<(f32, usize)> = (0..n_partitions)
        .map(|c| {
            let centroid = &centroids[c * dim..(c + 1) * dim];
            let d = distance::compute_distance_f32(query, centroid, metric);
            (d, c)
        })
        .collect();

    // Partial sort to find top-nprobe
    let ascending = metric.is_ascending();
    if ascending {
        dists.select_nth_unstable_by(nprobe - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        dists.select_nth_unstable_by(nprobe - 1, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    dists[..nprobe].iter().map(|&(_, c)| c).collect()
}

// ─── Metadata serialization ─────────────────────────────────────────────────

fn save_metadata(
    path: &Path,
    dim: usize,
    n_vectors: usize,
    n_partitions: usize,
    centroids: &[f32],
    partition_offsets: &[usize],
    original_ids: &[u32],
) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    // Header
    file.write_all(&(dim as u64).to_le_bytes())?;
    file.write_all(&(n_vectors as u64).to_le_bytes())?;
    file.write_all(&(n_partitions as u64).to_le_bytes())?;

    // Centroids: n_partitions × dim × f32
    let centroid_bytes =
        unsafe { std::slice::from_raw_parts(centroids.as_ptr() as *const u8, centroids.len() * 4) };
    file.write_all(centroid_bytes)?;

    // Partition offsets: (n_partitions + 1) × u64
    for &off in partition_offsets {
        file.write_all(&(off as u64).to_le_bytes())?;
    }

    // Original IDs: n_vectors × u32
    let id_bytes = unsafe {
        std::slice::from_raw_parts(original_ids.as_ptr() as *const u8, original_ids.len() * 4)
    };
    file.write_all(id_bytes)?;

    file.flush()?;
    Ok(())
}

fn load_metadata(
    path: &Path,
) -> std::io::Result<(usize, usize, usize, Vec<f32>, Vec<usize>, Vec<u32>)> {
    let mut file = File::open(path)?;
    let mut buf8 = [0u8; 8];

    // Header
    file.read_exact(&mut buf8)?;
    let dim = u64::from_le_bytes(buf8) as usize;
    file.read_exact(&mut buf8)?;
    let n_vectors = u64::from_le_bytes(buf8) as usize;
    file.read_exact(&mut buf8)?;
    let n_partitions = u64::from_le_bytes(buf8) as usize;

    // Centroids
    let centroid_count = n_partitions * dim;
    let mut centroid_bytes = vec![0u8; centroid_count * 4];
    file.read_exact(&mut centroid_bytes)?;
    let centroids: Vec<f32> = centroid_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Partition offsets
    let mut partition_offsets = Vec::with_capacity(n_partitions + 1);
    for _ in 0..=n_partitions {
        file.read_exact(&mut buf8)?;
        partition_offsets.push(u64::from_le_bytes(buf8) as usize);
    }

    // Original IDs
    let mut id_bytes = vec![0u8; n_vectors * 4];
    file.read_exact(&mut id_bytes)?;
    let original_ids: Vec<u32> = id_bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok((
        dim,
        n_vectors,
        n_partitions,
        centroids,
        partition_offsets,
        original_ids,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    fn env_usize(key: &str, default: usize) -> usize {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default)
    }

    fn normalize_in_place(v: &mut [f32]) {
        let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x /= norm;
        }
    }

    fn generate_clustered_unit_vectors(n: usize, dim: usize, k: usize) -> Vec<f32> {
        let mut rng = FastRng::new(7);
        let mut centers = vec![0.0f32; k * dim];
        for c in 0..k {
            let center = &mut centers[c * dim..(c + 1) * dim];
            for x in center.iter_mut() {
                *x = (rng.next_f64() as f32) * 2.0 - 1.0;
            }
            normalize_in_place(center);
        }

        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            let cluster = i % k;
            let center = &centers[cluster * dim..(cluster + 1) * dim];
            let vec = &mut data[i * dim..(i + 1) * dim];
            for d in 0..dim {
                let noise = ((rng.next_f64() as f32) * 2.0 - 1.0) * 0.03;
                vec[d] = center[d] + noise;
            }
            normalize_in_place(vec);
        }
        data
    }

    #[test]
    #[ignore]
    fn bench_minibatch_kmeans_baseline() {
        let n = env_usize("LYNSE_BENCH_N", 100_000);
        let dim = env_usize("LYNSE_BENCH_DIM", 128);
        let k = env_usize("LYNSE_BENCH_K", 256);
        let n_iters = env_usize("LYNSE_BENCH_ITERS", 10);
        let n_queries = env_usize("LYNSE_BENCH_QUERIES", 10_000).min(n.max(1));
        let nprobe = env_usize("LYNSE_BENCH_NPROBE", 16).min(k.max(1));
        let metric = DistanceMetric::InnerProduct;

        let data = generate_clustered_unit_vectors(n, dim, k);

        // Warm up SIMD and rayon thread pool before timing.
        let warm_centroids = kmeans(&data, dim, k, 2, metric);
        let warm_routing_dims = select_routing_dims(&warm_centroids, dim, k);
        let warm_assignments =
            assign_partitions(&data[..n_queries * dim], &warm_centroids, dim, k, metric);
        black_box(warm_assignments);
        for i in 0..n_queries.min(128) {
            let query = &data[i * dim..(i + 1) * dim];
            black_box(find_nearest_centroids(
                query,
                &warm_centroids,
                dim,
                k,
                nprobe,
                metric,
                &warm_routing_dims,
            ));
        }

        let train_start = Instant::now();
        let centroids = kmeans(&data, dim, k, n_iters, metric);
        let routing_dims = select_routing_dims(&centroids, dim, k);
        let train_elapsed = train_start.elapsed();

        let assign_start = Instant::now();
        let assignments = assign_partitions(&data, &centroids, dim, k, metric);
        let assign_elapsed = assign_start.elapsed();
        black_box(&assignments);

        let query_start = Instant::now();
        for i in 0..n_queries {
            let query = &data[i * dim..(i + 1) * dim];
            black_box(find_nearest_centroids(
                query,
                &centroids,
                dim,
                k,
                nprobe,
                metric,
                &routing_dims,
            ));
        }
        let query_elapsed = query_start.elapsed();

        let train_ms = train_elapsed.as_secs_f64() * 1_000.0;
        let assign_ms = assign_elapsed.as_secs_f64() * 1_000.0;
        let query_ms = query_elapsed.as_secs_f64() * 1_000.0;

        let train_throughput = n as f64 / train_elapsed.as_secs_f64().max(1e-9);
        let assign_throughput = n as f64 / assign_elapsed.as_secs_f64().max(1e-9);
        let query_qps = n_queries as f64 / query_elapsed.as_secs_f64().max(1e-9);
        let query_us = query_elapsed.as_secs_f64() * 1e6 / n_queries.max(1) as f64;

        eprintln!(
            "MINIBATCH_KMEANS_BASELINE n={} dim={} k={} iters={} nprobe={} train_ms={:.2} train_vec_per_s={:.0} assign_ms={:.2} assign_vec_per_s={:.0} query_ms={:.2} query_qps={:.0} query_us_per_query={:.2}",
            n,
            dim,
            k,
            n_iters,
            nprobe,
            train_ms,
            train_throughput,
            assign_ms,
            assign_throughput,
            query_ms,
            query_qps,
            query_us
        );
    }

    #[test]
    fn test_ivf_flat_build_and_search() {
        let tmp = tempfile::TempDir::new().unwrap();
        let data_path = tmp.path().join("vectors.bin");
        let dim = 4;
        let n_partitions = 3;

        // 12 vectors in 4D
        let data: Vec<f32> = vec![
            // Cluster near (1,0,0,0)
            1.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.1, 0.0,
            // Cluster near (0,1,0,0)
            0.0, 1.0, 0.1, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.1, 0.8, 0.1, 0.0,
            // Cluster near (0,0,1,0)
            0.0, 0.0, 1.0, 0.1, 0.0, 0.1, 0.9, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.8, 0.1,
        ];

        let idx = IvfFlatMmap::build(
            &data_path,
            &data,
            dim,
            n_partitions,
            10,
            DistanceMetric::InnerProduct,
        )
        .unwrap();
        assert_eq!(idx.len(), 12);

        // Query near cluster 0: should find vectors 0-3
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let (ids, _dists) = idx.search(&query, 3, 1, DistanceMetric::InnerProduct);
        assert_eq!(ids.len(), 3);
        // Top result should be from the first cluster (original ids 0-3)
        assert!(
            ids[0] <= 3,
            "top result should be in cluster 0, got id={}",
            ids[0]
        );
    }

    #[test]
    fn test_ivf_flat_reopen() {
        let tmp = tempfile::TempDir::new().unwrap();
        let data_path = tmp.path().join("vectors.bin");
        let dim = 2;

        let data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];

        let _idx =
            IvfFlatMmap::build(&data_path, &data, dim, 2, 5, DistanceMetric::InnerProduct).unwrap();

        // Reopen
        let idx2 = IvfFlatMmap::open(&data_path, dim).unwrap();
        assert_eq!(idx2.len(), 4);
        assert_eq!(idx2.n_partitions(), 2);

        let (ids, _) = idx2.search(&[1.0, 0.0], 1, 2, DistanceMetric::InnerProduct);
        assert_eq!(ids[0], 0); // should find original vector 0
    }

    #[test]
    fn test_ivf_flat_recall() {
        // Test that nprobe=all gives same results as brute force
        let tmp = tempfile::TempDir::new().unwrap();
        let data_path = tmp.path().join("vectors.bin");
        let bf_path = tmp.path().join("bf.bin");
        let dim = 8;
        let n = 1000;
        let n_partitions = 10;

        // Random data
        let mut rng = 42u64;
        let mut data = vec![0.0f32; n * dim];
        for v in data.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((rng >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        }

        let idx = IvfFlatMmap::build(
            &data_path,
            &data,
            dim,
            n_partitions,
            10,
            DistanceMetric::InnerProduct,
        )
        .unwrap();

        // Brute force
        let mut bf = FlatMmap::open(&bf_path, dim, VectorDtype::F32).unwrap();
        bf.write(&data).unwrap();

        let query: Vec<f32> = data[0..dim].to_vec();

        // IVF with nprobe=all should match brute force
        let (ivf_ids, _) = idx.search(&query, 5, n_partitions, DistanceMetric::InnerProduct);
        let (bf_ids, _) = bf.search(&query, 5, DistanceMetric::InnerProduct, false, None);

        // Top-1 should always match
        assert_eq!(ivf_ids[0], bf_ids[0], "IVF top-1 should match brute force");
    }
}
