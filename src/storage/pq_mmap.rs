//! Product Quantization (PQ) index with Asymmetric Distance Computation (ADC).
//!
//! Architecture:
//! 1. **Build**: K-means on subsampled data per subspace → store codebooks
//! 2. **Encode**: assign each vector to nearest centroid per subspace → u8 codes
//! 3. **Search**: build ADC lookup table → scan codes with M additions per vector
//! 4. **Re-rank**: exact f32 distance on top-N candidates → final top-k
//!
//! Default: M=16 subspaces, K=256 clusters (1 byte per subspace).
//! ADC LUT: 16 × 256 × 4B = 16KB — fits entirely in L1 cache.
//! ADC scan throughput: ~2ms/1M vectors (parallel, arm64).

use crate::distance::simd;
use crate::distance::DistanceMetric;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const PQ_MAGIC: u32 = 0x5051_4D4D; // "PQMM"
const PQ_VERSION: u32 = 1;
/// Number of clusters per subspace (must be ≤ 256 to fit in u8 code).
pub const DEFAULT_N_CLUSTERS: usize = 256;
/// Maximum number of training vectors (subsampled for speed).
const TRAINING_SUBSAMPLE: usize = 50_000;
/// K-means iterations per subspace.
const KMEANS_ITERS: usize = 15;
/// Default oversample factor for two-pass re-ranking.
pub const DEFAULT_OVERSAMPLE: usize = 32;

// ─── PQ Index ────────────────────────────────────────────────────────────────

/// Product Quantization index.
///
/// Encodes each vector as `n_subspaces` bytes, one per sub-space cluster.
/// Enables:
/// - `dim / n_subspaces × 4` compression ratio vs f32
/// - ADC search: O(n_subspaces) per vector (LUT lookups, no SIMD needed)
/// - Two-pass: ADC approximate → exact f32 re-rank for high recall
pub struct PQIndex {
    /// Number of subspaces (M)
    n_subspaces: usize,
    /// Number of clusters per subspace (K = 256)
    n_clusters: usize,
    /// Dimension of each subspace (= dim / n_subspaces)
    subspace_size: usize,
    /// Codebooks: \[n_subspaces × n_clusters × subspace_size\] f32 flat array.
    codebooks: Vec<f32>,
    /// Encoded vectors: \[n_vectors × n_subspaces\] u8 flat array.
    codes: Vec<u8>,
    n_vectors: usize,
    dim: usize,
}

impl PQIndex {
    /// Build a PQ index from raw f32 data.
    ///
    /// - `n_subspaces`: number of subspaces; must divide `dim` evenly.
    pub fn build(data: &[f32], n_vectors: usize, dim: usize, n_subspaces: usize) -> Self {
        assert_eq!(dim % n_subspaces, 0, "dim must be divisible by n_subspaces");
        assert!(n_vectors > 0, "need at least one vector");

        let subspace_size = dim / n_subspaces;
        let n_clusters = DEFAULT_N_CLUSTERS.min(n_vectors);

        // Subsample for training (uniform stride, capped at TRAINING_SUBSAMPLE)
        let train_n = n_vectors.min(TRAINING_SUBSAMPLE);
        let train_data: Vec<f32> = if train_n < n_vectors {
            let stride = (n_vectors / train_n).max(1);
            let mut sub = Vec::with_capacity(train_n * dim);
            let mut i = 0usize;
            while sub.len() / dim < train_n && i < n_vectors {
                sub.extend_from_slice(&data[i * dim..(i + 1) * dim]);
                i += stride;
            }
            sub
        } else {
            data[..n_vectors * dim].to_vec()
        };

        let actual_train_n = train_data.len() / dim;

        // Train one K-means per subspace, run in parallel across subspaces
        let codebook_parts: Vec<Vec<f32>> = (0..n_subspaces)
            .into_par_iter()
            .map(|m| {
                let start_col = m * subspace_size;
                kmeans_subspace(
                    &train_data,
                    actual_train_n,
                    dim,
                    start_col,
                    subspace_size,
                    n_clusters,
                    m as u64, // per-subspace seed for reproducibility
                )
            })
            .collect();

        // Flatten codebooks: [n_subspaces × n_clusters × subspace_size]
        let codebooks: Vec<f32> = codebook_parts.into_iter().flatten().collect();

        // Encode all vectors (parallel across vectors)
        let codes = encode_vectors(
            data,
            n_vectors,
            dim,
            &codebooks,
            n_subspaces,
            n_clusters,
            subspace_size,
        );

        PQIndex {
            n_subspaces,
            n_clusters,
            subspace_size,
            codebooks,
            codes,
            n_vectors,
            dim,
        }
    }

    /// ADC (Asymmetric Distance Computation) search.
    ///
    /// Pass 1: build M×K LUT → scan n_vectors codes with M additions each.
    /// Pass 2: exact f32 re-score on top `k * oversample` candidates.
    ///
    /// Returns `(indices, distances)` of the top-k results.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        f32_data: &[f32],
        metric: DistanceMetric,
        oversample: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        if self.n_vectors == 0 || k == 0 {
            return (vec![], vec![]);
        }
        let k = k.min(self.n_vectors);
        let n_candidates = (k * oversample).min(self.n_vectors);

        // Build ADC lookup table: [n_subspaces × n_clusters] f32
        let lut = build_lut(
            query,
            &self.codebooks,
            self.n_subspaces,
            self.n_clusters,
            self.subspace_size,
            metric,
        );

        // ADC scan: find top-N candidate indices
        let candidate_indices = adc_scan_topn(
            &self.codes,
            self.n_vectors,
            self.n_subspaces,
            self.n_clusters,
            &lut,
            n_candidates,
            metric.is_ascending(),
        );

        // Pass 2: exact f32 re-score
        rescore_exact(&candidate_indices, query, f32_data, self.dim, k, metric)
    }

    /// Number of indexed vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    /// Save the PQ index to `path`.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        w.write_all(&PQ_MAGIC.to_le_bytes())?;
        w.write_all(&PQ_VERSION.to_le_bytes())?;
        w.write_all(&(self.n_subspaces as u32).to_le_bytes())?;
        w.write_all(&(self.n_clusters as u32).to_le_bytes())?;
        w.write_all(&(self.subspace_size as u32).to_le_bytes())?;
        w.write_all(&(self.n_vectors as u64).to_le_bytes())?;
        w.write_all(&(self.dim as u32).to_le_bytes())?;

        // Codebooks (f32 LE)
        let cb_bytes: Vec<u8> = self
            .codebooks
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        w.write_all(&cb_bytes)?;

        // Codes (u8, raw)
        w.write_all(&self.codes)?;

        w.flush()?;
        Ok(())
    }

    /// Load a PQ index from `path`.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);

        let magic = read_u32le(&mut r)?;
        if magic != PQ_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid PQ magic bytes",
            ));
        }
        let _version = read_u32le(&mut r)?;
        let n_subspaces = read_u32le(&mut r)? as usize;
        let n_clusters = read_u32le(&mut r)? as usize;
        let subspace_size = read_u32le(&mut r)? as usize;
        let n_vectors = read_u64le(&mut r)? as usize;
        let dim = read_u32le(&mut r)? as usize;

        let cb_len = n_subspaces * n_clusters * subspace_size;
        let mut codebooks = vec![0.0f32; cb_len];
        for v in codebooks.iter_mut() {
            let bytes = read_4bytes(&mut r)?;
            *v = f32::from_le_bytes(bytes);
        }

        let mut codes = vec![0u8; n_vectors * n_subspaces];
        r.read_exact(&mut codes)?;

        Ok(PQIndex {
            n_subspaces,
            n_clusters,
            subspace_size,
            codebooks,
            codes,
            n_vectors,
            dim,
        })
    }
}

// ─── ADC Lookup Table ─────────────────────────────────────────────────────────

/// Build the ADC lookup table for a query.
///
/// `lut[m * K + k]` = distance(query_subspace_m, codebook\[m\]\[k\]).
/// For IP: inner product. For L2/Cosine: L2².
fn build_lut(
    query: &[f32],
    codebooks: &[f32],
    n_subspaces: usize,
    n_clusters: usize,
    subspace_size: usize,
    metric: DistanceMetric,
) -> Vec<f32> {
    let mut lut = vec![0.0f32; n_subspaces * n_clusters];
    for m in 0..n_subspaces {
        let q_sub = &query[m * subspace_size..(m + 1) * subspace_size];
        let cb_base = m * n_clusters * subspace_size;
        for c in 0..n_clusters {
            let cb = &codebooks[cb_base + c * subspace_size..cb_base + (c + 1) * subspace_size];
            lut[m * n_clusters + c] = match metric {
                DistanceMetric::InnerProduct => simd::inner_product_f32(q_sub, cb),
                // For L2 and Cosine: use L2² in each subspace (standard PQ)
                _ => simd::l2_squared_f32(q_sub, cb),
            };
        }
    }
    lut
}

// ─── K-means ──────────────────────────────────────────────────────────────────

/// K-means on a single subspace column of `data`.
///
/// Returns codebook: `\[n_clusters × subspace_size\]` f32 flat array.
fn kmeans_subspace(
    data: &[f32],
    n_vectors: usize,
    stride: usize,
    start_col: usize,
    ss: usize, // subspace_size
    k: usize,  // n_clusters
    seed: u64,
) -> Vec<f32> {
    let k = k.min(n_vectors);

    // Extract subspace column data contiguously for cache-friendly access
    let mut sub = vec![0.0f32; n_vectors * ss];
    for i in 0..n_vectors {
        sub[i * ss..(i + 1) * ss]
            .copy_from_slice(&data[i * stride + start_col..i * stride + start_col + ss]);
    }

    // Random initialization (fast, good enough for PQ with many iterations)
    let mut centroids = random_init_centroids(&sub, n_vectors, ss, k, seed);
    let mut assignments = vec![0u32; n_vectors];

    for _iter in 0..KMEANS_ITERS {
        // --- Assign ---
        let mut changed = false;
        for i in 0..n_vectors {
            let v = &sub[i * ss..(i + 1) * ss];
            let mut best_c = 0u32;
            let mut best_d = f32::MAX;
            for c in 0..k {
                let d = l2sq_slice(v, &centroids[c * ss..(c + 1) * ss]);
                if d < best_d {
                    best_d = d;
                    best_c = c as u32;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // --- Update centroids ---
        let mut counts = vec![0u32; k];
        let mut new_centroids = vec![0.0f32; k * ss];
        for i in 0..n_vectors {
            let c = assignments[i] as usize;
            counts[c] += 1;
            for d in 0..ss {
                new_centroids[c * ss + d] += sub[i * ss + d];
            }
        }

        // Find the most populated cluster (for re-init of empty ones)
        let max_cluster = counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &cnt)| cnt)
            .map(|(i, _)| i)
            .unwrap_or(0);

        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for d in 0..ss {
                    new_centroids[c * ss + d] *= inv;
                }
            } else {
                // Re-init empty cluster from the largest cluster with small perturbation
                let src = max_cluster * ss;
                let dst = c * ss;
                for d in 0..ss {
                    new_centroids[dst + d] =
                        new_centroids[src + d] * (1.0 + 0.01 * ((d % 2) as f32 - 0.5));
                }
            }
        }
        centroids = new_centroids;
    }

    centroids
}

/// Randomly select k distinct vectors as initial centroids.
fn random_init_centroids(data: &[f32], n: usize, ss: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut centroids = vec![0.0f32; k * ss];
    let mut chosen = std::collections::HashSet::with_capacity(k);
    let mut c = 0;
    let mut attempts = 0;
    while c < k && attempts < k * 10 {
        let idx = rng.gen_range(0..n);
        if chosen.insert(idx) {
            centroids[c * ss..(c + 1) * ss].copy_from_slice(&data[idx * ss..(idx + 1) * ss]);
            c += 1;
        }
        attempts += 1;
    }
    // If we couldn't find enough unique vectors, fill remaining with small perturbations
    while c < k {
        let src = (c - 1) * ss;
        let dst = c * ss;
        for d in 0..ss {
            centroids[dst + d] = centroids[src + d] * (1.0 + 0.001 * d as f32);
        }
        c += 1;
    }
    centroids
}

/// Encode all vectors using the trained codebooks (parallel across vectors).
fn encode_vectors(
    data: &[f32],
    n_vectors: usize,
    dim: usize,
    codebooks: &[f32],
    n_subspaces: usize,
    n_clusters: usize,
    subspace_size: usize,
) -> Vec<u8> {
    let mut codes = vec![0u8; n_vectors * n_subspaces];
    codes
        .par_chunks_mut(n_subspaces)
        .enumerate()
        .for_each(|(i, code_row)| {
            for m in 0..n_subspaces {
                let v_sub = &data[i * dim + m * subspace_size..i * dim + (m + 1) * subspace_size];
                let cb_base = m * n_clusters * subspace_size;
                let mut best_c = 0u8;
                let mut best_d = f32::MAX;
                for c in 0..n_clusters {
                    let cb =
                        &codebooks[cb_base + c * subspace_size..cb_base + (c + 1) * subspace_size];
                    let d = l2sq_slice(v_sub, cb);
                    if d < best_d {
                        best_d = d;
                        best_c = c as u8;
                    }
                }
                code_row[m] = best_c;
            }
        });
    codes
}

// ─── ADC Scan ─────────────────────────────────────────────────────────────────

/// Compact entry for heap-based top-k tracking.
#[derive(Clone, Copy)]
struct HeapEntry {
    score: f32,
    idx: u32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Parallel ADC scan: find top-`n_candidates` indices using the LUT.
///
/// - `ascending=true` (L2/Cosine): keep vectors with **smallest** ADC scores.
/// - `ascending=false` (IP): keep vectors with **largest** ADC scores.
fn adc_scan_topn(
    codes: &[u8],
    n_vectors: usize,
    n_subspaces: usize,
    n_clusters: usize,
    lut: &[f32],
    n_candidates: usize,
    ascending: bool,
) -> Vec<u32> {
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_vecs = (n_vectors / n_threads).max(256);

    let chunk_results: Vec<Vec<u32>> = codes
        .par_chunks(chunk_vecs * n_subspaces)
        .enumerate()
        .map(|(chunk_idx, code_chunk)| {
            let n_in_chunk = code_chunk.len() / n_subspaces;
            let base_idx = chunk_idx * chunk_vecs;

            if ascending {
                // Keep smallest N: MAX-heap (evict largest when full)
                let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(n_candidates + 1);
                for i in 0..n_in_chunk {
                    let code =
                        unsafe { code_chunk.get_unchecked(i * n_subspaces..(i + 1) * n_subspaces) };
                    let mut score = 0.0f32;
                    for m in 0..n_subspaces {
                        score += unsafe { *lut.get_unchecked(m * n_clusters + code[m] as usize) };
                    }
                    let entry = HeapEntry {
                        score,
                        idx: (base_idx + i) as u32,
                    };
                    if heap.len() < n_candidates {
                        heap.push(entry);
                    } else if let Some(&top) = heap.peek() {
                        if score < top.score {
                            heap.pop();
                            heap.push(entry);
                        }
                    }
                }
                heap.into_iter().map(|e| e.idx).collect()
            } else {
                // Keep largest N: MIN-heap (evict smallest when full)
                let mut heap: BinaryHeap<std::cmp::Reverse<HeapEntry>> =
                    BinaryHeap::with_capacity(n_candidates + 1);
                for i in 0..n_in_chunk {
                    let code =
                        unsafe { code_chunk.get_unchecked(i * n_subspaces..(i + 1) * n_subspaces) };
                    let mut score = 0.0f32;
                    for m in 0..n_subspaces {
                        score += unsafe { *lut.get_unchecked(m * n_clusters + code[m] as usize) };
                    }
                    let entry = HeapEntry {
                        score,
                        idx: (base_idx + i) as u32,
                    };
                    if heap.len() < n_candidates {
                        heap.push(std::cmp::Reverse(entry));
                    } else if let Some(&std::cmp::Reverse(top)) = heap.peek() {
                        if score > top.score {
                            heap.pop();
                            heap.push(std::cmp::Reverse(entry));
                        }
                    }
                }
                heap.into_iter().map(|std::cmp::Reverse(e)| e.idx).collect()
            }
        })
        .collect();

    // Merge per-thread results, sort globally, take top-N
    let mut all_entries: Vec<(f32, u32)> = chunk_results
        .into_iter()
        .flatten()
        .map(|idx| {
            // Re-compute score for sorting (cheap: M additions)
            let code = &codes[idx as usize * n_subspaces..(idx as usize + 1) * n_subspaces];
            let score: f32 = (0..n_subspaces)
                .map(|m| unsafe { *lut.get_unchecked(m * n_clusters + code[m] as usize) })
                .sum();
            (score, idx)
        })
        .collect();

    if ascending {
        all_entries.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    } else {
        all_entries.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    }
    all_entries.truncate(n_candidates);
    all_entries.iter().map(|e| e.1).collect()
}

// ─── Exact f32 Re-scoring ─────────────────────────────────────────────────────

/// Re-score candidate indices with exact f32 distances and return top-k.
pub fn rescore_exact(
    candidates: &[u32],
    query: &[f32],
    f32_data: &[f32],
    dim: usize,
    k: usize,
    metric: DistanceMetric,
) -> (Vec<u32>, Vec<f32>) {
    let ascending = metric.is_ascending();
    let mut exact: Vec<(f32, u32)> = candidates
        .iter()
        .map(|&idx| {
            let base = idx as usize * dim;
            let cand = unsafe { f32_data.get_unchecked(base..base + dim) };
            let dist = match metric {
                DistanceMetric::InnerProduct => simd::inner_product_f32(query, cand),
                DistanceMetric::L2Squared => simd::l2_squared_f32(query, cand),
                DistanceMetric::Cosine => simd::cosine_distance_f32(query, cand),
                _ => crate::distance::compute_distance_f32(query, cand, metric),
            };
            (dist, idx)
        })
        .collect();

    if ascending {
        exact.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    } else {
        exact.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    }
    exact.truncate(k);

    let indices = exact.iter().map(|e| e.1).collect();
    let dists = exact.iter().map(|e| e.0).collect();
    (indices, dists)
}

// ─── Distance helpers ─────────────────────────────────────────────────────────

/// Scalar L2² distance for small subspace vectors.
#[inline(always)]
fn l2sq_slice(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ─── I/O helpers ──────────────────────────────────────────────────────────────

fn read_u32le<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64le<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_4bytes<R: Read>(r: &mut R) -> std::io::Result<[u8; 4]> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

// ─── Parse helpers ────────────────────────────────────────────────────────────

/// Extract n_subspaces from an index type string like "FLAT-IP-PQ8" or "FLAT-IP-PQ16".
/// Falls back to a sensible divisor of `dim` if not specified.
pub fn parse_n_subspaces(index_type_upper: &str, dim: usize) -> usize {
    // Try to extract numeric suffix from "PQ<N>"
    if let Some(pos) = index_type_upper.find("PQ") {
        let rest = &index_type_upper[pos + 2..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            if let Ok(n) = digits.parse::<usize>() {
                if n > 0 && dim % n == 0 {
                    return n;
                }
            }
        }
    }
    // Auto-select: prefer 16, then 8, 32, 4, 12, 24, 6, 2
    for &candidate in &[16usize, 8, 32, 4, 12, 24, 6, 2, 1] {
        if dim % candidate == 0 {
            return candidate;
        }
    }
    1
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn random_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
    }

    #[test]
    fn test_pq_build_search_ip() {
        let n = 1000;
        let dim = 32;
        let data = random_data(n, dim, 1);
        let pq = PQIndex::build(&data, n, dim, 8);
        assert_eq!(pq.len(), n);

        let query = random_data(1, dim, 99);
        let (ids, dists) = pq.search(&query, 10, &data, DistanceMetric::InnerProduct, 32);
        assert_eq!(ids.len(), 10);
        // Highest IP should be in top-10
        let brute_best = (0..n)
            .map(|i| simd::inner_product_f32(&query, &data[i * dim..(i + 1) * dim]))
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        assert!(ids.contains(&(brute_best as u32)));
    }

    #[test]
    fn test_pq_build_search_l2() {
        let n = 500;
        let dim = 16;
        let data = random_data(n, dim, 2);
        let pq = PQIndex::build(&data, n, dim, 4);

        let query = random_data(1, dim, 88);
        let (ids, _dists) = pq.search(&query, 5, &data, DistanceMetric::L2Squared, 32);
        assert_eq!(ids.len(), 5);

        // Brute-force best
        let brute_best = (0..n)
            .map(|i| simd::l2_squared_f32(&query, &data[i * dim..(i + 1) * dim]))
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        assert!(ids.contains(&(brute_best as u32)));
    }

    #[test]
    fn test_pq_save_load() {
        let n = 200;
        let dim = 16;
        let data = random_data(n, dim, 3);
        let pq = PQIndex::build(&data, n, dim, 4);

        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("pq_index.bin");
        pq.save(&path).unwrap();

        let pq2 = PQIndex::load(&path).unwrap();
        assert_eq!(pq2.len(), n);
        assert_eq!(pq2.n_subspaces, pq.n_subspaces);
        assert_eq!(pq2.codes, pq.codes);
    }

    #[test]
    fn test_parse_n_subspaces() {
        assert_eq!(parse_n_subspaces("FLAT-IP-PQ8", 128), 8);
        assert_eq!(parse_n_subspaces("FLAT-IP-PQ16", 128), 16);
        assert_eq!(parse_n_subspaces("FLAT-IP-PQ", 128), 16); // default
        assert_eq!(parse_n_subspaces("FLAT-L2-PQ", 32), 16); // 32 % 16 == 0
    }
}
