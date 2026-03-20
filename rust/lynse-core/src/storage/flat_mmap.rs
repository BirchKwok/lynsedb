//! Ultra-fast flat vector storage with persistent mmap.
//!
//! Design: single contiguous binary file of raw f32 LE values.
//! - **Write**: append raw bytes (no metadata overhead)
//! - **Read**: mmap the file once → `&[f32]` zero-copy slice
//! - **Search**: parallel SIMD distances on mmap'd slice + quickselect top-k
//!
//! Target: < 0.5ms for 1M×128 brute-force IP search (data hot in page cache).

use crate::distance::{self, DistanceMetric};
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// A flat mmap-backed vector store optimized for brute-force search.
///
/// Holds a persistent mmap handle — data stays in page cache across queries.
/// Uses fused parallel distance + per-thread heap merge — zero large buffer allocation.
pub struct FlatMmap {
    path: PathBuf,
    dim: usize,
    n_vectors: usize,
    /// Persistent mmap — kept alive for the lifetime of this struct.
    mmap: Option<Mmap>,
}

impl FlatMmap {
    /// Create or open a flat vector file.
    ///
    /// - `path`: path to the raw f32 binary file
    /// - `dim`: vector dimension (must match the data in the file)
    pub fn open(path: &Path, dim: usize) -> std::io::Result<Self> {
        if !path.exists() {
            // Create empty file
            File::create(path)?;
        }

        let file = File::open(path)?;
        let file_len = file.metadata()?.len() as usize;
        let n_floats = file_len / 4;
        let n_vectors = if dim > 0 { n_floats / dim } else { 0 };

        let mmap = if n_vectors > 0 {
            Some(unsafe { Mmap::map(&file)? })
        } else {
            None
        };

        Ok(Self {
            path: path.to_path_buf(),
            dim,
            n_vectors,
            mmap,
        })
    }

    /// Number of vectors stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    /// Vector dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the raw f32 slice from mmap (zero-copy).
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        match &self.mmap {
            Some(m) => {
                let ptr = m.as_ptr() as *const f32;
                unsafe { std::slice::from_raw_parts(ptr, self.n_vectors * self.dim) }
            }
            None => &[],
        }
    }

    /// Write vectors to the file (append mode). Re-mmaps after write.
    pub fn write(&mut self, data: &[f32]) -> std::io::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        assert_eq!(data.len() % self.dim, 0, "data length must be multiple of dim");

        // Drop existing mmap before writing (avoid mmap/write conflict)
        self.mmap = None;

        // Append raw bytes
        let mut file = OpenOptions::new().create(true).append(true).open(&self.path)?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        file.write_all(bytes)?;
        file.flush()?;
        drop(file);

        // Re-mmap
        let file = File::open(&self.path)?;
        let file_len = file.metadata()?.len() as usize;
        self.n_vectors = file_len / 4 / self.dim;
        self.mmap = Some(unsafe { Mmap::map(&file)? });

        Ok(())
    }

    /// Brute-force top-k search on mmap'd data.
    ///
    /// Returns (indices, distances) sorted by relevance.
    /// Uses **fused parallel distance + per-thread heap merge**:
    /// - Each rayon thread computes distances AND maintains a size-k BinaryHeap
    /// - No large dist_buf or index_buf — only k entries per thread
    /// - Final merge of per-thread heaps → sorted top-k
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
    ) -> (Vec<u32>, Vec<f32>) {
        let n = self.n_vectors;
        if n == 0 || k == 0 {
            return (vec![], vec![]);
        }
        let k = k.min(n);
        let dim = self.dim;
        let ascending = metric.is_ascending();
        let candidates = self.as_slice();

        // Fused parallel distance + top-k heap
        fused_parallel_topk(query, candidates, dim, k, metric, ascending)
    }
}

// ─── Fused parallel distance + top-k (zero large buffer) ────────────────────

/// Ordered node for BinaryHeap (max-heap by default).
/// For ascending metrics (L2): we want a max-heap so we can evict the largest.
/// For descending metrics (IP): we want a min-heap so we can evict the smallest.
#[derive(Clone, Copy)]
struct HeapNode {
    dist: f32,
    idx: u32,
}

impl PartialEq for HeapNode {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for HeapNode {}

impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}

impl HeapNode {
    #[inline]
    fn cmp_asc(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
    #[inline]
    fn cmp_desc(&self, other: &Self) -> Ordering {
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}

/// Fused parallel distance computation + per-thread top-k heap merge.
///
/// Each rayon thread:
///   1. Iterates its chunk of candidates
///   2. Computes distance via SIMD
///   3. Maintains a BinaryHeap of size k (evicts worst)
/// Then merges all per-thread heaps into final sorted top-k.
///
/// Memory: O(k * n_threads) — tiny compared to O(n) buffers.
fn fused_parallel_topk(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    metric: DistanceMetric,
    ascending: bool,
) -> (Vec<u32>, Vec<f32>) {
    let n = candidates.len() / dim;

    // Sequential path for small datasets (avoid rayon overhead)
    if n < 4096 {
        return fused_sequential_topk(query, candidates, dim, k, metric, ascending, 0);
    }

    // Dispatch to typed helper to use correct heap type
    if ascending {
        fused_parallel_topk_asc(query, candidates, dim, k, n, metric)
    } else {
        fused_parallel_topk_desc(query, candidates, dim, k, n, metric)
    }
}

/// Ascending (L2): MAX-heap keeps largest dist on top → evict when new < largest → keeps k smallest.
#[inline(never)]
fn fused_parallel_topk_asc(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    metric: DistanceMetric,
) -> (Vec<u32>, Vec<f32>) {
    let n_threads = rayon::current_num_threads();
    let chunk_vecs = (n / n_threads).max(512);
    let chunk_floats = chunk_vecs * dim;

    let chunk_results: Vec<Vec<HeapNode>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let n_in_chunk = cand_chunk.len() / dim;
            let base_idx = chunk_idx * chunk_vecs;
            // MAX-heap: BinaryHeap<HeapNode> — largest dist on top
            let mut heap: BinaryHeap<HeapNode> = BinaryHeap::with_capacity(k + 1);

            for i in 0..n_in_chunk {
                let dist = unsafe {
                    let start = i * dim;
                    let cand = cand_chunk.get_unchecked(start..start + dim);
                    distance::compute_distance_f32(query, cand, metric)
                };
                let node = HeapNode { dist, idx: (base_idx + i) as u32 };

                if heap.len() < k {
                    heap.push(node);
                } else if let Some(&top) = heap.peek() {
                    if dist < top.dist {
                        heap.pop();
                        heap.push(node);
                    }
                }
            }
            heap.into_vec()
        })
        .collect();

    // Merge with MAX-heap
    let mut merged: BinaryHeap<HeapNode> = BinaryHeap::with_capacity(k + 1);
    for chunk_top in chunk_results {
        for node in chunk_top {
            if merged.len() < k {
                merged.push(node);
            } else if let Some(&top) = merged.peek() {
                if node.dist < top.dist {
                    merged.pop();
                    merged.push(node);
                }
            }
        }
    }

    let mut result: Vec<HeapNode> = merged.into_vec();
    result.sort_unstable_by(|a, b| a.cmp_asc(b));
    let indices = result.iter().map(|n| n.idx).collect();
    let dists = result.iter().map(|n| n.dist).collect();
    (indices, dists)
}

/// Descending (IP): MIN-heap keeps smallest dist on top → evict when new > smallest → keeps k largest.
#[inline(never)]
fn fused_parallel_topk_desc(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    metric: DistanceMetric,
) -> (Vec<u32>, Vec<f32>) {
    let n_threads = rayon::current_num_threads();
    let chunk_vecs = (n / n_threads).max(512);
    let chunk_floats = chunk_vecs * dim;

    let chunk_results: Vec<Vec<HeapNode>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let n_in_chunk = cand_chunk.len() / dim;
            let base_idx = chunk_idx * chunk_vecs;
            // MIN-heap via Reverse: smallest dist on top
            let mut heap: BinaryHeap<std::cmp::Reverse<HeapNode>> = BinaryHeap::with_capacity(k + 1);

            for i in 0..n_in_chunk {
                let dist = unsafe {
                    let start = i * dim;
                    let cand = cand_chunk.get_unchecked(start..start + dim);
                    distance::compute_distance_f32(query, cand, metric)
                };
                let node = HeapNode { dist, idx: (base_idx + i) as u32 };

                if heap.len() < k {
                    heap.push(std::cmp::Reverse(node));
                } else if let Some(&std::cmp::Reverse(top)) = heap.peek() {
                    if dist > top.dist {
                        heap.pop();
                        heap.push(std::cmp::Reverse(node));
                    }
                }
            }
            heap.into_iter().map(|std::cmp::Reverse(n)| n).collect()
        })
        .collect();

    // Merge with MIN-heap
    let mut merged: BinaryHeap<std::cmp::Reverse<HeapNode>> = BinaryHeap::with_capacity(k + 1);
    for chunk_top in chunk_results {
        for node in chunk_top {
            if merged.len() < k {
                merged.push(std::cmp::Reverse(node));
            } else if let Some(&std::cmp::Reverse(top)) = merged.peek() {
                if node.dist > top.dist {
                    merged.pop();
                    merged.push(std::cmp::Reverse(node));
                }
            }
        }
    }

    let mut result: Vec<HeapNode> = merged.into_iter().map(|std::cmp::Reverse(n)| n).collect();
    result.sort_unstable_by(|a, b| a.cmp_desc(b));
    let indices = result.iter().map(|n| n.idx).collect();
    let dists = result.iter().map(|n| n.dist).collect();
    (indices, dists)
}

/// Sequential fused distance + top-k for small datasets or single-partition search.
fn fused_sequential_topk(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    metric: DistanceMetric,
    ascending: bool,
    base_idx: usize,
) -> (Vec<u32>, Vec<f32>) {
    if ascending {
        fused_sequential_topk_asc(query, candidates, dim, k, metric, base_idx)
    } else {
        fused_sequential_topk_desc(query, candidates, dim, k, metric, base_idx)
    }
}

fn fused_sequential_topk_asc(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    metric: DistanceMetric,
    base_idx: usize,
) -> (Vec<u32>, Vec<f32>) {
    let n = candidates.len() / dim;
    let mut heap: BinaryHeap<HeapNode> = BinaryHeap::with_capacity(k + 1);

    for i in 0..n {
        let dist = unsafe {
            let start = i * dim;
            let cand = candidates.get_unchecked(start..start + dim);
            distance::compute_distance_f32(query, cand, metric)
        };
        let node = HeapNode { dist, idx: (base_idx + i) as u32 };

        if heap.len() < k {
            heap.push(node);
        } else if let Some(&top) = heap.peek() {
            if dist < top.dist {
                heap.pop();
                heap.push(node);
            }
        }
    }

    let mut result: Vec<HeapNode> = heap.into_vec();
    result.sort_unstable_by(|a, b| a.cmp_asc(b));
    let indices = result.iter().map(|n| n.idx).collect();
    let dists = result.iter().map(|n| n.dist).collect();
    (indices, dists)
}

fn fused_sequential_topk_desc(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    metric: DistanceMetric,
    base_idx: usize,
) -> (Vec<u32>, Vec<f32>) {
    let n = candidates.len() / dim;
    let mut heap: BinaryHeap<std::cmp::Reverse<HeapNode>> = BinaryHeap::with_capacity(k + 1);

    for i in 0..n {
        let dist = unsafe {
            let start = i * dim;
            let cand = candidates.get_unchecked(start..start + dim);
            distance::compute_distance_f32(query, cand, metric)
        };
        let node = HeapNode { dist, idx: (base_idx + i) as u32 };

        if heap.len() < k {
            heap.push(std::cmp::Reverse(node));
        } else if let Some(&std::cmp::Reverse(top)) = heap.peek() {
            if dist > top.dist {
                heap.pop();
                heap.push(std::cmp::Reverse(node));
            }
        }
    }

    let mut result: Vec<HeapNode> = heap.into_iter().map(|std::cmp::Reverse(n)| n).collect();
    result.sort_unstable_by(|a, b| a.cmp_desc(b));
    let indices = result.iter().map(|n| n.idx).collect();
    let dists = result.iter().map(|n| n.dist).collect();
    (indices, dists)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_mmap_write_search() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("vectors.bin");
        let dim = 4;

        let mut store = FlatMmap::open(&path, dim).unwrap();
        assert_eq!(store.len(), 0);

        // Write 5 vectors
        let data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // vec 0: unit x
            0.0, 1.0, 0.0, 0.0, // vec 1: unit y
            0.5, 0.5, 0.0, 0.0, // vec 2: mixed
            0.0, 0.0, 1.0, 0.0, // vec 3: unit z
            0.0, 0.0, 0.0, 1.0, // vec 4: unit w
        ];
        store.write(&data).unwrap();
        assert_eq!(store.len(), 5);

        // Search: query = unit x, IP metric, k=2
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let (ids, dists) = store.search(&query, 2, DistanceMetric::InnerProduct);
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], 0); // exact match (IP=1.0)
        assert!((dists[0] - 1.0).abs() < 1e-6);

        // Search: L2, query = origin, k=2 → closest should be vec 0 and vec 1 (tied at 1.0)
        let query_origin = vec![0.0f32, 0.0, 0.0, 0.0];
        let (ids_l2, dists_l2) = store.search(&query_origin, 1, DistanceMetric::L2Squared);
        // All unit vectors have L2=1.0 to origin, mixed has L2=0.5
        assert_eq!(ids_l2[0], 2); // vec 2 is closest (0.5^2+0.5^2=0.5)
    }

    #[test]
    fn test_flat_mmap_append() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("vectors.bin");
        let dim = 2;

        let mut store = FlatMmap::open(&path, dim).unwrap();

        // Write batch 1
        store.write(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(store.len(), 2);

        // Append batch 2
        store.write(&[5.0, 6.0]).unwrap();
        assert_eq!(store.len(), 3);

        let slice = store.as_slice();
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_flat_mmap_reopen() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("vectors.bin");
        let dim = 3;

        // Write data
        {
            let mut store = FlatMmap::open(&path, dim).unwrap();
            store.write(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        }

        // Reopen and verify
        let mut store = FlatMmap::open(&path, dim).unwrap();
        assert_eq!(store.len(), 2);
        let (ids, _) = store.search(&[1.0, 2.0, 3.0], 1, DistanceMetric::L2Squared);
        assert_eq!(ids[0], 0);
    }
}
