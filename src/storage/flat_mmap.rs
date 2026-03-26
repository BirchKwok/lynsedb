//! Ultra-fast flat vector storage with persistent mmap.
//!
//! Design: single contiguous binary file of raw f32 LE values.
//! - **Write**: append raw bytes (no metadata overhead)
//! - **Read**: mmap the file once → `&[f32]` zero-copy slice + madvise hints
//! - **Search**: monomorphized parallel SIMD + sorted-array top-k
//!
//! Optimizations for < 3ms on 1M×128:
//! 1. madvise(SEQUENTIAL) on mmap open — OS prefetches pages ahead
//! 2. Monomorphized distance fn via macro — no per-vector match overhead
//! 3. Software prefetch — hide memory latency for next vector
//! 4. Sorted array top-k — less overhead than BinaryHeap for small k

use crate::distance::{self, DistanceMetric};
use crate::distance::simd;
use memmap2::Mmap;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use parking_lot::RwLock;

// ─── Software prefetch ──────────────────────────────────────────────────────

#[inline(always)]
unsafe fn prefetch_read_data(ptr: *const u8) {
    #[cfg(target_arch = "aarch64")]
    {
        // PRFM PLDL1STRM — prefetch into L1 data cache, streaming (no cache pollution)
        std::arch::asm!("prfm pldl1strm, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
}

// ─── madvise helper ─────────────────────────────────────────────────────────

/// Apply madvise hints to a mmap for sequential scan performance.
#[cfg(unix)]
fn madvise_sequential(mmap: &Mmap) {
    unsafe {
        let ptr = mmap.as_ptr() as *mut libc::c_void;
        let len = mmap.len();
        // MADV_SEQUENTIAL: expect sequential access → aggressive read-ahead
        libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
        // MADV_WILLNEED: pre-fault pages into page cache
        libc::madvise(ptr, len, libc::MADV_WILLNEED);
        // Request huge pages (2MB) for TLB miss reduction on large mmaps.
        // MADV_HUGEPAGE (Linux) / VM_FLAGS_SUPERPAGE_SIZE_2MB (macOS) — best-effort.
        #[cfg(target_os = "linux")]
        {
            const MADV_HUGEPAGE: libc::c_int = 14;
            libc::madvise(ptr, len, MADV_HUGEPAGE);
        }
        #[cfg(target_os = "macos")]
        {
            // macOS: madvise with MADV_FREE_REUSABLE hint for large allocations
            // The kernel may use superpages automatically for large aligned regions
            // We help by touching pages sequentially (WILLNEED already does this)
        }
    }
}

#[cfg(not(unix))]
fn madvise_sequential(_mmap: &Mmap) {}

/// A flat mmap-backed vector store optimized for brute-force search.
///
/// Holds a persistent mmap handle — data stays in page cache across queries.
/// Uses fused parallel distance + per-thread sorted-array top-k — zero large buffer.
/// Number of SQ8 candidates per final result (higher = better recall).
const SQ8_OVERSAMPLE: usize = 20;

pub struct FlatMmap {
    path: PathBuf,
    dim: usize,
    n_vectors: usize,
    /// Persistent mmap — kept alive for the lifetime of this struct.
    mmap: Option<Mmap>,
    /// Lazily-initialized SQ8 quantized data for bandwidth-efficient search.
    sq8: RwLock<Option<SQ8Data>>,
}

impl FlatMmap {
    /// Create or open a flat vector file.
    ///
    /// Applies madvise(SEQUENTIAL | WILLNEED) for optimal OS prefetching.
    pub fn open(path: &Path, dim: usize) -> std::io::Result<Self> {
        if !path.exists() {
            File::create(path)?;
        }

        let file = File::open(path)?;
        let file_len = file.metadata()?.len() as usize;
        let n_floats = file_len / 4;
        let n_vectors = if dim > 0 { n_floats / dim } else { 0 };

        let mmap = if n_vectors > 0 {
            let m = unsafe { Mmap::map(&file)? };
            madvise_sequential(&m);
            // Pin pages in physical memory to avoid soft page faults during search
            #[cfg(unix)]
            unsafe {
                libc::mlock(m.as_ptr() as *const libc::c_void, m.len());
            }
            Some(m)
        } else {
            None
        };

        Ok(Self {
            path: path.to_path_buf(),
            dim,
            n_vectors,
            mmap,
            sq8: RwLock::new(None),
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

        // Re-mmap with madvise hints
        let file = File::open(&self.path)?;
        let file_len = file.metadata()?.len() as usize;
        self.n_vectors = file_len / 4 / self.dim;
        let m = unsafe { Mmap::map(&file)? };
        madvise_sequential(&m);
        self.mmap = Some(m);

        // Invalidate SQ8 cache (data changed)
        *self.sq8.write() = None;

        Ok(())
    }

    /// Ensure SQ8 quantized data is available (lazy init).
    fn ensure_sq8(&self) {
        // Fast path: already initialized
        if self.sq8.read().is_some() {
            return;
        }
        // Slow path: build SQ8 quantization
        let mut guard = self.sq8.write();
        if guard.is_none() {
            let candidates = self.as_slice();
            *guard = Some(SQ8Data::from_f32_parallel(candidates, self.dim));
        }
    }

    /// Brute-force top-k search on mmap'd data.
    ///
    /// When `use_sq8` is true (user specified SQ8 index mode), uses two-pass acceleration:
    /// 1. Scan quantized u8 data (4x less bandwidth) → top-N candidates
    /// 2. Re-score candidates with exact f32 → top-k
    ///
    /// When `use_sq8` is false (default F32 mode), uses full f32 SIMD scan.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
        use_sq8: bool,
    ) -> (Vec<u32>, Vec<f32>) {
        let n = self.n_vectors;
        if n == 0 || k == 0 {
            return (vec![], vec![]);
        }
        let k = k.min(n);
        let dim = self.dim;
        let candidates = self.as_slice();

        // SQ8 two-pass path (only when user explicitly requested SQ8)
        if use_sq8 && matches!(metric,
            DistanceMetric::InnerProduct | DistanceMetric::L2Squared | DistanceMetric::Cosine)
        {
            self.ensure_sq8();
            let sq8_guard = self.sq8.read();
            if let Some(ref sq8) = *sq8_guard {
                return sq8_two_pass_search(query, candidates, sq8, dim, k, n, metric);
            }
        }

        // Full f32 scan
        match metric {
            DistanceMetric::InnerProduct =>
                fused_topk_parallel::<false>(query, candidates, dim, k, n, simd::inner_product_f32),
            DistanceMetric::L2Squared =>
                fused_topk_parallel::<true>(query, candidates, dim, k, n, simd::l2_squared_f32),
            DistanceMetric::Cosine =>
                fused_topk_parallel::<true>(query, candidates, dim, k, n, simd::cosine_distance_f32),
            _ =>
                fused_topk_parallel::<true>(query, candidates, dim, k, n,
                    |a, b| distance::compute_distance_f32(a, b, metric)),
        }
    }
}

// ─── Optimized fused parallel search ────────────────────────────────────────

/// Compact (dist, idx) pair for top-k tracking.
#[derive(Clone, Copy)]
struct TopKEntry {
    dist: f32,
    idx: u32,
}

impl PartialEq for TopKEntry {
    fn eq(&self, other: &Self) -> bool { self.dist == other.dist }
}
impl Eq for TopKEntry {}
impl PartialOrd for TopKEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for TopKEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}

/// Insert an entry into a sorted top-k array.
#[inline(always)]
fn topk_insert<const ASC: bool>(
    top: &mut Vec<TopKEntry>,
    k: usize,
    entry: TopKEntry,
    threshold: &mut f32,
    filled: &mut bool,
) {
    if !*filled {
        top.push(entry);
        if top.len() == k {
            if ASC {
                top.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
            } else {
                top.sort_unstable_by(|a, b| b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal));
            }
            *threshold = top[k - 1].dist;
            *filled = true;
        }
    } else {
        top[k - 1] = entry;
        let mut j = k - 1;
        if ASC {
            while j > 0 && top[j].dist < top[j - 1].dist {
                top.swap(j, j - 1);
                j -= 1;
            }
        } else {
            while j > 0 && top[j].dist > top[j - 1].dist {
                top.swap(j, j - 1);
                j -= 1;
            }
        }
        *threshold = top[k - 1].dist;
    }
}

/// Check if a distance passes the threshold.
#[inline(always)]
fn passes_threshold<const ASC: bool>(dist: f32, threshold: f32) -> bool {
    if ASC { dist < threshold } else { dist > threshold }
}

/// Fused parallel distance + sorted-array top-k.
///
/// `ASC=true`: ascending (L2/Cosine) — keep k smallest distances.
/// `ASC=false`: descending (IP) — keep k largest distances.
#[inline(never)]
fn fused_topk_parallel<const ASC: bool>(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    dist_fn: impl Fn(&[f32], &[f32]) -> f32 + Send + Sync,
) -> (Vec<u32>, Vec<f32>) {
    if n < 4096 {
        return fused_topk_seq::<ASC>(query, candidates, dim, k, 0, &dist_fn);
    }

    let n_threads = rayon::current_num_threads();
    let chunk_vecs = (n / n_threads).max(512);
    let chunk_floats = chunk_vecs * dim;

    let chunk_results: Vec<Vec<TopKEntry>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let n_in_chunk = cand_chunk.len() / dim;
            let base_idx = chunk_idx * chunk_vecs;

            let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
            let mut threshold: f32 = if ASC { f32::INFINITY } else { f32::NEG_INFINITY };
            let mut filled = false;

            let mut ptr = cand_chunk.as_ptr();
            let pairs = n_in_chunk / 2;
            let dim2 = dim * 2;
            for i in 0..pairs {
                let cand0 = unsafe { std::slice::from_raw_parts(ptr, dim) };
                let cand1 = unsafe { std::slice::from_raw_parts(ptr.wrapping_add(dim), dim) };
                let dist0 = dist_fn(query, cand0);
                let dist1 = dist_fn(query, cand1);

                let idx0 = (base_idx + i * 2) as u32;
                if !filled || passes_threshold::<ASC>(dist0, threshold) {
                    topk_insert::<ASC>(&mut top, k, TopKEntry { dist: dist0, idx: idx0 }, &mut threshold, &mut filled);
                }
                if !filled || passes_threshold::<ASC>(dist1, threshold) {
                    topk_insert::<ASC>(&mut top, k, TopKEntry { dist: dist1, idx: idx0 + 1 }, &mut threshold, &mut filled);
                }
                ptr = unsafe { ptr.add(dim2) };
            }
            // Handle odd remainder
            if n_in_chunk % 2 == 1 {
                let cand = unsafe { std::slice::from_raw_parts(ptr, dim) };
                let dist = dist_fn(query, cand);
                if !filled || passes_threshold::<ASC>(dist, threshold) {
                    topk_insert::<ASC>(&mut top, k, TopKEntry { dist, idx: (base_idx + n_in_chunk - 1) as u32 }, &mut threshold, &mut filled);
                }
            }

            if !filled && !top.is_empty() {
                if ASC {
                    top.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
                } else {
                    top.sort_unstable_by(|a, b| b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal));
                }
            }
            top
        })
        .collect();

    merge_topk_results::<ASC>(&chunk_results, k)
}

/// Sequential fallback for small datasets (< 4096 vectors).
fn fused_topk_seq<const ASC: bool>(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    base_idx: usize,
    dist_fn: &(impl Fn(&[f32], &[f32]) -> f32),
) -> (Vec<u32>, Vec<f32>) {
    let n = candidates.len() / dim;
    let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
    let mut threshold: f32 = if ASC { f32::INFINITY } else { f32::NEG_INFINITY };
    let mut filled = false;

    for i in 0..n {
        let cand = unsafe {
            let start = i * dim;
            candidates.get_unchecked(start..start + dim)
        };
        let dist = dist_fn(query, cand);

        if !filled || passes_threshold::<ASC>(dist, threshold) {
            topk_insert::<ASC>(&mut top, k, TopKEntry { dist, idx: (base_idx + i) as u32 }, &mut threshold, &mut filled);
        }
    }

    if !filled && !top.is_empty() {
        if ASC {
            top.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        } else {
            top.sort_unstable_by(|a, b| b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal));
        }
    }

    let indices = top.iter().map(|e| e.idx).collect();
    let dists = top.iter().map(|e| e.dist).collect();
    (indices, dists)
}

/// Merge per-thread sorted results into final top-k.
fn merge_topk_results<const ASC: bool>(
    chunk_results: &[Vec<TopKEntry>],
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut merged: Vec<TopKEntry> = Vec::with_capacity(k);
    let mut threshold: f32 = if ASC { f32::INFINITY } else { f32::NEG_INFINITY };
    let mut filled = false;

    for chunk_top in chunk_results {
        for &entry in chunk_top {
            if !filled || passes_threshold::<ASC>(entry.dist, threshold) {
                topk_insert::<ASC>(&mut merged, k, entry, &mut threshold, &mut filled);
            }
        }
    }

    if !filled && !merged.is_empty() {
        if ASC {
            merged.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        } else {
            merged.sort_unstable_by(|a, b| b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal));
        }
    }

    let indices = merged.iter().map(|e| e.idx).collect();
    let dists = merged.iter().map(|e| e.dist).collect();
    (indices, dists)
}

// ─── SQ8 Scalar Quantization for bandwidth-efficient search ─────────────────
//
// Quantizes f32 vectors to u8 (4x compression).  Per-dimension min/max scaling.
// Two-pass search: u8 approximate scan → f32 re-score on candidates.

/// SQ8 quantized data: per-dimension affine transform f32 → u8.
struct SQ8Data {
    /// Quantized vectors: n_vectors × dim, row-major.
    data: Vec<u8>,
    /// Per-dimension minimum (dim values).
    mins: Vec<f32>,
    /// Per-dimension scale: 255.0 / (max - min).  Zero if range is zero.
    scales: Vec<f32>,
}

impl SQ8Data {
    /// Build SQ8 quantization from f32 data.  Uses rayon for parallel quantization.
    fn from_f32_parallel(candidates: &[f32], dim: usize) -> Self {
        let n = candidates.len() / dim;

        // Pass 1: compute per-dimension min/max
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];
        for v in 0..n {
            let base = v * dim;
            for d in 0..dim {
                let val = unsafe { *candidates.get_unchecked(base + d) };
                if val < mins[d] { mins[d] = val; }
                if val > maxs[d] { maxs[d] = val; }
            }
        }

        // Compute scales
        let mut scales = vec![0.0f32; dim];
        for d in 0..dim {
            let range = maxs[d] - mins[d];
            if range > 1e-30 {
                scales[d] = 255.0 / range;
            }
        }

        // Pass 2: quantize (parallel by chunks)
        let chunk_size = 8192;
        let mut data = vec![0u8; n * dim];
        data.par_chunks_mut(chunk_size * dim)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let base_vec = chunk_idx * chunk_size;
                let n_in_chunk = out_chunk.len() / dim;
                for i in 0..n_in_chunk {
                    let src_base = (base_vec + i) * dim;
                    let dst_base = i * dim;
                    for d in 0..dim {
                        let val = unsafe { *candidates.get_unchecked(src_base + d) };
                        let q = ((val - mins[d]) * scales[d]).round();
                        out_chunk[dst_base + d] = q.clamp(0.0, 255.0) as u8;
                    }
                }
            });

        SQ8Data { data, mins, scales }
    }

    /// Quantize a query vector using the stored parameters.
    #[inline]
    fn quantize_query(&self, query: &[f32]) -> Vec<u8> {
        let dim = self.mins.len();
        let mut q = vec![0u8; dim];
        for d in 0..dim {
            let v = ((query[d] - self.mins[d]) * self.scales[d]).round();
            q[d] = v.clamp(0.0, 255.0) as u8;
        }
        q
    }
}

// ─── SQ8 NEON-accelerated u8 distance functions ────────────────────────────

/// Approximate dot product of two u8 vectors.  Higher = more similar (for IP).
#[inline(always)]
fn sq8_dot_product(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    { return sq8_dot_neon(a, b); }

    #[cfg(target_arch = "x86_64")]
    { return sq8_dot_scalar(a, b); }

    #[allow(unreachable_code)]
    sq8_dot_scalar(a, b)
}

/// Approximate L2² of two u8 vectors.  Lower = more similar.
#[inline(always)]
fn sq8_l2sq(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    { return sq8_l2sq_neon(a, b); }

    #[cfg(target_arch = "x86_64")]
    { return sq8_l2sq_scalar(a, b); }

    #[allow(unreachable_code)]
    sq8_l2sq_scalar(a, b)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn sq8_dot_neon(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::aarch64::*;
    let dim = a.len();
    let chunks16 = dim / 16;
    let remainder = dim % 16;
    unsafe {
        let mut acc0 = vdupq_n_u32(0);
        let mut acc1 = vdupq_n_u32(0);
        for i in 0..chunks16 {
            let base = i * 16;
            let va = vld1q_u8(a.as_ptr().add(base));
            let vb = vld1q_u8(b.as_ptr().add(base));
            let prod_lo = vmull_u8(vget_low_u8(va), vget_low_u8(vb));
            let prod_hi = vmull_u8(vget_high_u8(va), vget_high_u8(vb));
            acc0 = vpadalq_u16(acc0, prod_lo);
            acc1 = vpadalq_u16(acc1, prod_hi);
        }
        acc0 = vaddq_u32(acc0, acc1);
        let mut sum = vaddvq_u32(acc0);
        let base = chunks16 * 16;
        for i in 0..remainder {
            sum += (a[base + i] as u32) * (b[base + i] as u32);
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn sq8_l2sq_neon(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::aarch64::*;
    let dim = a.len();
    let chunks16 = dim / 16;
    let remainder = dim % 16;
    unsafe {
        let mut acc0 = vdupq_n_u32(0);
        let mut acc1 = vdupq_n_u32(0);
        for i in 0..chunks16 {
            let base = i * 16;
            let va = vld1q_u8(a.as_ptr().add(base));
            let vb = vld1q_u8(b.as_ptr().add(base));
            let diff = vabdq_u8(va, vb);
            let sq_lo = vmull_u8(vget_low_u8(diff), vget_low_u8(diff));
            let sq_hi = vmull_u8(vget_high_u8(diff), vget_high_u8(diff));
            acc0 = vpadalq_u16(acc0, sq_lo);
            acc1 = vpadalq_u16(acc1, sq_hi);
        }
        acc0 = vaddq_u32(acc0, acc1);
        let mut sum = vaddvq_u32(acc0);
        let base = chunks16 * 16;
        for i in 0..remainder {
            let d = (a[base + i] as i32) - (b[base + i] as i32);
            sum += (d * d) as u32;
        }
        sum
    }
}

#[inline]
fn sq8_dot_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x as u32) * (y as u32)).sum()
}

#[inline]
fn sq8_l2sq_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| {
        let d = (x as i32) - (y as i32);
        (d * d) as u32
    }).sum()
}

// ─── SQ8 two-pass search ───────────────────────────────────────────────────

/// Two-pass search: SQ8 approximate scan → f32 re-score.
fn sq8_two_pass_search(
    query: &[f32],
    candidates: &[f32],
    sq8: &SQ8Data,
    dim: usize,
    k: usize,
    n: usize,
    metric: DistanceMetric,
) -> (Vec<u32>, Vec<f32>) {
    let query_u8 = sq8.quantize_query(query);

    // Pass 1: fast u8 approximate scan → top-N candidate indices
    // Cosine needs higher oversample because L2 SQ8 ≈ cosine is weaker correlation
    let candidate_indices = match metric {
        DistanceMetric::InnerProduct => {
            let n_cand = (k * SQ8_OVERSAMPLE).max(200).min(n);
            sq8_scan_topn::<false>(&query_u8, &sq8.data, dim, n_cand, n, sq8_dot_product)
        }
        DistanceMetric::Cosine => {
            // L2 SQ8 ≈ cosine for similar-norm vectors; use higher oversample for safety
            let n_cand = (k * SQ8_OVERSAMPLE * 5).max(500).min(n);
            sq8_scan_topn::<true>(&query_u8, &sq8.data, dim, n_cand, n, sq8_l2sq)
        }
        _ => {
            let n_cand = (k * SQ8_OVERSAMPLE).max(200).min(n);
            sq8_scan_topn::<true>(&query_u8, &sq8.data, dim, n_cand, n, sq8_l2sq)
        }
    };

    // Pass 2: exact f32 re-score on candidates
    let mut exact: Vec<TopKEntry> = candidate_indices.iter().map(|&idx| {
        let base = (idx as usize) * dim;
        let cand = unsafe { candidates.get_unchecked(base..base + dim) };
        let dist = match metric {
            DistanceMetric::InnerProduct => simd::inner_product_f32(query, cand),
            DistanceMetric::L2Squared => simd::l2_squared_f32(query, cand),
            DistanceMetric::Cosine => simd::cosine_distance_f32(query, cand),
            _ => distance::compute_distance_f32(query, cand, metric),
        };
        TopKEntry { dist, idx }
    }).collect();

    // Sort by relevance and take top-k
    let ascending = metric.is_ascending();
    if ascending {
        exact.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
    } else {
        exact.sort_unstable_by(|a, b| b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal));
    }
    exact.truncate(k);

    let indices = exact.iter().map(|e| e.idx).collect();
    let dists = exact.iter().map(|e| e.dist).collect();
    (indices, dists)
}

/// Parallel SQ8 approximate scan.  Returns indices of top-N candidates.
///
/// Uses `impl Fn` for monomorphization (NEON inlined) + BinaryHeap for O(log k) inserts.
fn sq8_scan_topn<const ASC: bool>(
    query_u8: &[u8],
    data_u8: &[u8],
    dim: usize,
    n_candidates: usize,
    n: usize,
    dist_fn: impl Fn(&[u8], &[u8]) -> u32 + Send + Sync,
) -> Vec<u32> {
    use std::collections::BinaryHeap;

    let n_threads = rayon::current_num_threads();
    let chunk_vecs = (n / n_threads).max(512);
    let chunk_bytes = chunk_vecs * dim;

    let chunk_results: Vec<Vec<TopKEntry>> = data_u8
        .par_chunks(chunk_bytes)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let n_in_chunk = cand_chunk.len() / dim;
            let base_idx = chunk_idx * chunk_vecs;

            if ASC {
                // Keep k smallest: MAX-heap (largest on top, evict when new < top)
                let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(n_candidates + 1);
                for i in 0..n_in_chunk {
                    if i + 4 < n_in_chunk {
                        unsafe { prefetch_read_data(cand_chunk.as_ptr().add((i + 4) * dim)); }
                    }
                    let cand = unsafe { cand_chunk.get_unchecked(i * dim..(i + 1) * dim) };
                    let dist = dist_fn(query_u8, cand) as f32;
                    let entry = TopKEntry { dist, idx: (base_idx + i) as u32 };
                    if heap.len() < n_candidates {
                        heap.push(entry);
                    } else if let Some(&top) = heap.peek() {
                        if dist < top.dist {
                            heap.pop();
                            heap.push(entry);
                        }
                    }
                }
                heap.into_vec()
            } else {
                // Keep k largest: MIN-heap (smallest on top, evict when new > top)
                let mut heap: BinaryHeap<std::cmp::Reverse<TopKEntry>> =
                    BinaryHeap::with_capacity(n_candidates + 1);
                for i in 0..n_in_chunk {
                    if i + 4 < n_in_chunk {
                        unsafe { prefetch_read_data(cand_chunk.as_ptr().add((i + 4) * dim)); }
                    }
                    let cand = unsafe { cand_chunk.get_unchecked(i * dim..(i + 1) * dim) };
                    let dist = dist_fn(query_u8, cand) as f32;
                    let entry = TopKEntry { dist, idx: (base_idx + i) as u32 };
                    if heap.len() < n_candidates {
                        heap.push(std::cmp::Reverse(entry));
                    } else if let Some(&std::cmp::Reverse(top)) = heap.peek() {
                        if dist > top.dist {
                            heap.pop();
                            heap.push(std::cmp::Reverse(entry));
                        }
                    }
                }
                heap.into_iter().map(|std::cmp::Reverse(e)| e).collect()
            }
        })
        .collect();

    // Merge: collect all per-thread results, sort, take top-N
    let mut all: Vec<TopKEntry> = chunk_results.into_iter().flatten().collect();
    if ASC {
        all.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
    } else {
        all.sort_unstable_by(|a, b| b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal));
    }
    all.truncate(n_candidates);
    all.iter().map(|e| e.idx).collect()
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
        let (ids, dists) = store.search(&query, 2, DistanceMetric::InnerProduct, false);
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], 0); // exact match (IP=1.0)
        assert!((dists[0] - 1.0).abs() < 1e-6);

        // Search: L2, query = origin, k=2 → closest should be vec 0 and vec 1 (tied at 1.0)
        let query_origin = vec![0.0f32, 0.0, 0.0, 0.0];
        let (ids_l2, dists_l2) = store.search(&query_origin, 1, DistanceMetric::L2Squared, false);
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
        let (ids, _) = store.search(&[1.0, 2.0, 3.0], 1, DistanceMetric::L2Squared, false);
        assert_eq!(ids[0], 0);
    }
}
