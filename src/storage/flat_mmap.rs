//! Ultra-fast flat vector storage with persistent mmap.
//!
//! Design: single contiguous binary file of raw f32 LE values.
//! - **Write**: append raw bytes (no metadata overhead)
//! - **Read**: mmap the file once → `&[f32]` zero-copy slice + cheap madvise hints
//! - **Search**: monomorphized parallel SIMD + sorted-array top-k
//!
//! Optimizations for < 3ms on 1M×128:
//! 1. mmap open is cheap; searches fault in only the pages they touch
//! 2. Monomorphized distance fn via macro — no per-vector match overhead
//! 3. Software prefetch — hide memory latency for next vector
//! 4. Sorted array top-k — less overhead than BinaryHeap for small k

use crate::distance::simd;
use crate::distance::{self, DistanceMetric};
use crate::storage::dtype::{decode_f16_bytes_to_f32, encode_f32_slice_as_le_bytes, VectorDtype};
use half::f16;
use memmap2::Mmap;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

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

/// Software prefetch distance (in vectors) for sequential f32 scans.
const PREFETCH_AHEAD_VECTORS: usize = 8;

// ─── madvise helper ─────────────────────────────────────────────────────────

/// Apply cheap madvise hints to a mmap.
///
/// Do not eager-prefetch or pin the whole vector file here: approximate search
/// may only touch a tiny shortlist, and making open cost O(file size) pushes an
/// avoidable latency spike onto the first query.
#[cfg(unix)]
fn madvise_sequential(mmap: &Mmap) {
    unsafe {
        let ptr = mmap.as_ptr() as *mut libc::c_void;
        let len = mmap.len();
        // MADV_SEQUENTIAL: expect sequential access → aggressive read-ahead
        libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
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
const APPROX_IP_ORDER_PREBUILD_POOL: usize = 20_000;
const APPROX_IP_ORDER_MAGIC: &[u8; 8] = b"LIPORD2\0";
const APPROX_NORMS_MAGIC: &[u8; 8] = b"LNRM2\0\0\0";
const APPROX_IP_ORDER_VECTOR_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;

pub struct FlatMmap {
    path: PathBuf,
    dim: usize,
    dtype: VectorDtype,
    n_vectors: usize,
    /// Persistent mmap — kept alive for the lifetime of this struct.
    mmap: Option<Mmap>,
    /// Lazily-initialized SQ8 quantized data for bandwidth-efficient search.
    sq8: RwLock<Option<SQ8Data>>,
    /// Lazily packed one-bit rows for Hamming/Jaccard/Tanimoto/Dice scans.
    /// This adds only 1 bit per stored dimension and avoids touching the f32
    /// mmap after the first binary query.
    binary: RwLock<Option<BinaryData>>,
    /// Lazily-initialized raw-f32 row aggregate order for fast IP approximation.
    approx_ip_order: RwLock<Option<ApproxIpOrder>>,
    /// Raw-f32 row norms for exact-algebra L2/Cosine approximate mode.
    approx_norms: RwLock<Option<ApproxNorms>>,
}

struct ApproxIpOrder {
    n_vectors: usize,
    pool: usize,
    sum_desc: Vec<TopKEntry>,
    sum_desc_vectors: Vec<f32>,
    sum_asc: Vec<TopKEntry>,
    sum_asc_vectors: Vec<f32>,
}

struct ApproxNorms {
    n_vectors: usize,
    norm2: Vec<f32>,
    inv_norm: Vec<f32>,
}

struct BinaryData {
    words_per_vector: usize,
    data: Vec<u64>,
}

impl BinaryData {
    fn from_f32_parallel(values: &[f32], dim: usize) -> Self {
        let n = if dim == 0 { 0 } else { values.len() / dim };
        let words_per_vector = dim.div_ceil(64);
        let mut data = vec![0u64; n * words_per_vector];
        data.par_chunks_mut(words_per_vector)
            .enumerate()
            .for_each(|(row, words)| {
                let source = &values[row * dim..(row + 1) * dim];
                pack_binary_row_f32(source, words);
            });
        Self {
            words_per_vector,
            data,
        }
    }

    fn from_f16_parallel(values: &[u16], dim: usize) -> Self {
        let n = if dim == 0 { 0 } else { values.len() / dim };
        let words_per_vector = dim.div_ceil(64);
        let mut data = vec![0u64; n * words_per_vector];
        data.par_chunks_mut(words_per_vector)
            .enumerate()
            .for_each(|(row, words)| {
                let source = &values[row * dim..(row + 1) * dim];
                for (index, &bits) in source.iter().enumerate() {
                    if f16::from_bits(bits).to_f32() > 0.5 {
                        words[index / 64] |= 1u64 << (index % 64);
                    }
                }
            });
        Self {
            words_per_vector,
            data,
        }
    }
}

impl FlatMmap {
    /// Create or open a flat vector file.
    ///
    /// Applies cheap mmap hints without eager-prefaulting the whole file.
    pub fn open(path: &Path, dim: usize, dtype: VectorDtype) -> std::io::Result<Self> {
        if !path.exists() {
            File::create(path)?;
        }

        let file = File::open(path)?;
        let file_len = file.metadata()?.len() as usize;
        let n_values = file_len / dtype.byte_width();
        let n_vectors = if dim > 0 { n_values / dim } else { 0 };

        let mmap = if n_vectors > 0 {
            let m = unsafe { Mmap::map(&file)? };
            madvise_sequential(&m);
            Some(m)
        } else {
            None
        };

        let approx_ip_order =
            load_approx_ip_order(path, dim, n_vectors, file_len as u64).unwrap_or(None);
        let approx_norms = load_approx_norms(path, dim, n_vectors, file_len as u64).unwrap_or(None);

        Ok(Self {
            path: path.to_path_buf(),
            dim,
            dtype,
            n_vectors,
            mmap,
            sq8: RwLock::new(None),
            binary: RwLock::new(None),
            approx_ip_order: RwLock::new(approx_ip_order),
            approx_norms: RwLock::new(approx_norms),
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

    #[inline]
    pub fn dtype(&self) -> VectorDtype {
        self.dtype
    }

    #[inline]
    fn vector_file_len(&self) -> u64 {
        (self.n_vectors * self.dim * self.dtype.byte_width()) as u64
    }

    /// Get the raw f32 slice from mmap (zero-copy).
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        assert_eq!(
            self.dtype,
            VectorDtype::F32,
            "as_slice() is only zero-copy for float32 vector stores"
        );
        match &self.mmap {
            Some(m) => {
                let ptr = m.as_ptr() as *const f32;
                unsafe { std::slice::from_raw_parts(ptr, self.n_vectors * self.dim) }
            }
            None => &[],
        }
    }

    #[inline]
    pub fn as_f16_bits_slice(&self) -> &[u16] {
        assert_eq!(
            self.dtype,
            VectorDtype::F16,
            "as_f16_bits_slice() is only valid for float16 vector stores"
        );
        match &self.mmap {
            Some(m) => {
                let ptr = m.as_ptr() as *const u16;
                unsafe { std::slice::from_raw_parts(ptr, self.n_vectors * self.dim) }
            }
            None => &[],
        }
    }

    pub fn as_f32_cow(&self) -> Cow<'_, [f32]> {
        match (self.dtype, &self.mmap) {
            (VectorDtype::F32, _) => Cow::Borrowed(self.as_slice()),
            (VectorDtype::F16, Some(m)) => Cow::Owned(decode_f16_bytes_to_f32(m)),
            (VectorDtype::F16, None) => Cow::Borrowed(&[]),
        }
    }

    /// Write vectors to the file (append mode). Re-mmaps after write.
    pub fn write(&mut self, data: &[f32]) -> std::io::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        assert_eq!(
            data.len() % self.dim,
            0,
            "data length must be multiple of dim"
        );
        let old_n_vectors = self.n_vectors;
        let old_order = self.approx_ip_order.write().take();
        let old_norms = self.approx_norms.write().take();

        // Drop existing mmap before writing (avoid mmap/write conflict)
        self.mmap = None;

        // Append raw bytes
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let bytes = encode_f32_slice_as_le_bytes(data, self.dtype);
        file.write_all(&bytes)?;
        file.flush()?;
        drop(file);

        // Re-mmap with madvise hints
        let file = File::open(&self.path)?;
        let file_len = file.metadata()?.len() as usize;
        self.n_vectors = file_len / self.dtype.byte_width() / self.dim;
        let m = unsafe { Mmap::map(&file)? };
        madvise_sequential(&m);
        self.mmap = Some(m);

        // Invalidate SQ8 cache (data changed)
        *self.sq8.write() = None;
        *self.binary.write() = None;
        let next_order = update_approx_ip_order_after_append(
            old_order,
            data,
            self.dim,
            old_n_vectors,
            self.n_vectors,
        );
        if let Some(ref order) = next_order {
            let _ = save_approx_ip_order(&self.path, self.dim, self.vector_file_len(), order);
        } else {
            let _ = fs::remove_file(approx_ip_order_path(&self.path));
        }
        *self.approx_ip_order.write() = next_order;

        let next_norms = update_approx_norms_after_append(
            old_norms,
            data,
            self.dim,
            old_n_vectors,
            self.n_vectors,
        );
        if let Some(ref norms) = next_norms {
            let _ = save_approx_norms(&self.path, self.dim, self.vector_file_len(), norms);
        } else {
            let _ = fs::remove_file(approx_norms_path(&self.path));
        }
        *self.approx_norms.write() = next_norms;

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
            let candidates = self.as_f32_cow();
            *guard = Some(SQ8Data::from_f32_parallel(&candidates, self.dim));
        }
    }

    fn ensure_binary(&self) {
        if self.binary.read().is_some() {
            return;
        }
        let mut guard = self.binary.write();
        if guard.is_none() {
            *guard = Some(match self.dtype {
                VectorDtype::F32 => BinaryData::from_f32_parallel(self.as_slice(), self.dim),
                VectorDtype::F16 => {
                    BinaryData::from_f16_parallel(self.as_f16_bits_slice(), self.dim)
                }
            });
        }
    }

    /// Ensure row-sum order is available for raw-f32 approximate IP search.
    fn ensure_approx_ip_order(&self, pool: usize) {
        if self
            .approx_ip_order
            .read()
            .as_ref()
            .is_some_and(|order| order.n_vectors == self.n_vectors && order.pool >= pool)
        {
            return;
        }

        let mut guard = self.approx_ip_order.write();
        if guard
            .as_ref()
            .is_some_and(|order| order.n_vectors == self.n_vectors && order.pool >= pool)
        {
            return;
        }

        let pool = pool.min(self.n_vectors);
        let candidates = self.as_f32_cow();
        let (sum_desc, sum_desc_vectors, sum_asc, sum_asc_vectors) =
            build_approx_ip_row_order(&candidates, self.dim, 0, self.n_vectors, pool);
        let order = ApproxIpOrder {
            n_vectors: self.n_vectors,
            pool,
            sum_desc,
            sum_desc_vectors,
            sum_asc,
            sum_asc_vectors,
        };
        let _ = save_approx_ip_order(&self.path, self.dim, self.vector_file_len(), &order);
        *guard = Some(order);
    }

    /// Ensure raw-f32 norm cache is available for exact-algebra L2/Cosine approx search.
    fn ensure_approx_norms(&self) {
        if self
            .approx_norms
            .read()
            .as_ref()
            .is_some_and(|norms| approx_norms_compatible(norms, self.n_vectors))
        {
            return;
        }

        let mut guard = self.approx_norms.write();
        if guard
            .as_ref()
            .is_some_and(|norms| approx_norms_compatible(norms, self.n_vectors))
        {
            return;
        }

        let candidates = self.as_f32_cow();
        let norms = build_approx_norms(&candidates, self.dim, self.n_vectors);
        let _ = save_approx_norms(&self.path, self.dim, self.vector_file_len(), &norms);
        *guard = Some(norms);
    }

    /// Filtered brute-force top-k search on mmap'd data.
    ///
    /// Two strategies based on selectivity:
    /// - **Few matches** (≤ 50K): Direct random access — O(matches) work, no full scan.
    /// - **Many matches** (> 50K): Parallel scan with bitset — same SIMD parallelism
    ///   as unfiltered path, skipping non-matching vectors via O(1) bitset check.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
        subset_indices: &[u64],
    ) -> (Vec<u32>, Vec<f32>) {
        let n = self.n_vectors;
        if n == 0 || k == 0 || subset_indices.is_empty() {
            return (vec![], vec![]);
        }
        let k = k.min(subset_indices.len());
        let dim = self.dim;
        if metric.is_binary() {
            self.ensure_binary();
            let query = pack_binary_query(query);
            if let Some(binary) = self.binary.read().as_ref() {
                return packed_binary_search_filtered(&query, binary, k, n, metric, subset_indices);
            }
        }
        if self.dtype == VectorDtype::F16 {
            return search_filtered_f16(
                query,
                self.as_f16_bits_slice(),
                dim,
                k,
                n,
                metric,
                subset_indices,
            );
        }
        let candidates_cow;
        let candidates: &[f32] = if self.dtype == VectorDtype::F32 {
            self.as_slice()
        } else {
            candidates_cow = self.as_f32_cow();
            &candidates_cow
        };

        // Threshold: direct access for few matches, parallel scan for many
        const DIRECT_ACCESS_LIMIT: usize = 50_000;

        if subset_indices.len() <= DIRECT_ACCESS_LIMIT {
            // Strategy 1: Direct random access — O(matches) not O(n)
            return match metric {
                DistanceMetric::InnerProduct => direct_access_topk::<false>(
                    query,
                    candidates,
                    dim,
                    k,
                    n,
                    subset_indices,
                    simd::inner_product_f32,
                ),
                DistanceMetric::L2Squared => direct_access_topk::<true>(
                    query,
                    candidates,
                    dim,
                    k,
                    n,
                    subset_indices,
                    simd::l2_squared_f32,
                ),
                DistanceMetric::Cosine => direct_access_topk::<true>(
                    query,
                    candidates,
                    dim,
                    k,
                    n,
                    subset_indices,
                    simd::cosine_distance_f32,
                ),
                DistanceMetric::Manhattan => direct_access_topk::<true>(
                    query,
                    candidates,
                    dim,
                    k,
                    n,
                    subset_indices,
                    simd::manhattan_f32,
                ),
                DistanceMetric::Haversine => direct_access_topk::<true>(
                    query,
                    candidates,
                    dim,
                    k,
                    n,
                    subset_indices,
                    simd::haversine_meters_f32,
                ),
                DistanceMetric::Correlation => direct_access_topk::<true>(
                    query,
                    candidates,
                    dim,
                    k,
                    n,
                    subset_indices,
                    simd::correlation_distance_f32,
                ),
                DistanceMetric::Hellinger => direct_access_topk::<true>(
                    query,
                    candidates,
                    dim,
                    k,
                    n,
                    subset_indices,
                    simd::hellinger_distance_f32,
                ),
                DistanceMetric::Wasserstein => direct_access_topk::<true>(
                    query,
                    candidates,
                    dim,
                    k,
                    n,
                    subset_indices,
                    simd::wasserstein_1d_f32,
                ),
                _ => direct_access_topk::<true>(
                    query,
                    candidates,
                    dim,
                    k,
                    n,
                    subset_indices,
                    |a, b| distance::compute_distance_f32(a, b, metric),
                ),
            };
        }

        // Strategy 2: Parallel scan with bitset — for high-selectivity filters
        let max_id = subset_indices.iter().copied().max().unwrap_or(0) as usize;
        let mut bitset = vec![0u64; (max_id / 64) + 1];
        for &id in subset_indices {
            let idx = id as usize;
            if idx < n {
                bitset[idx / 64] |= 1u64 << (idx % 64);
            }
        }

        match metric {
            DistanceMetric::InnerProduct => fused_topk_parallel_filtered::<false>(
                query,
                candidates,
                dim,
                k,
                n,
                &bitset,
                max_id,
                simd::inner_product_f32,
            ),
            DistanceMetric::L2Squared => fused_topk_parallel_filtered::<true>(
                query,
                candidates,
                dim,
                k,
                n,
                &bitset,
                max_id,
                simd::l2_squared_f32,
            ),
            DistanceMetric::Cosine => fused_topk_parallel_filtered::<true>(
                query,
                candidates,
                dim,
                k,
                n,
                &bitset,
                max_id,
                simd::cosine_distance_f32,
            ),
            DistanceMetric::Manhattan => fused_topk_parallel_filtered::<true>(
                query,
                candidates,
                dim,
                k,
                n,
                &bitset,
                max_id,
                simd::manhattan_f32,
            ),
            DistanceMetric::Haversine => fused_topk_parallel_filtered::<true>(
                query,
                candidates,
                dim,
                k,
                n,
                &bitset,
                max_id,
                simd::haversine_meters_f32,
            ),
            DistanceMetric::Correlation => fused_topk_parallel_filtered::<true>(
                query,
                candidates,
                dim,
                k,
                n,
                &bitset,
                max_id,
                simd::correlation_distance_f32,
            ),
            DistanceMetric::Hellinger => fused_topk_parallel_filtered::<true>(
                query,
                candidates,
                dim,
                k,
                n,
                &bitset,
                max_id,
                simd::hellinger_distance_f32,
            ),
            DistanceMetric::Wasserstein => fused_topk_parallel_filtered::<true>(
                query,
                candidates,
                dim,
                k,
                n,
                &bitset,
                max_id,
                simd::wasserstein_1d_f32,
            ),
            _ => fused_topk_parallel_filtered::<true>(
                query,
                candidates,
                dim,
                k,
                n,
                &bitset,
                max_id,
                |a, b| distance::compute_distance_f32(a, b, metric),
            ),
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
        approx: Option<super::approx_search::ApproxSearchConfig>,
    ) -> (Vec<u32>, Vec<f32>) {
        let n = self.n_vectors;
        if n == 0 || k == 0 {
            return (vec![], vec![]);
        }
        let k = k.min(n);
        let dim = self.dim;

        if metric.is_binary() {
            self.ensure_binary();
            let query = pack_binary_query(query);
            if let Some(binary) = self.binary.read().as_ref() {
                return packed_binary_search(&query, binary, k, n, metric);
            }
        }

        // Approximate two-phase search on the original f32 data. This path
        // intentionally does not use SQ8/PQ/RaBitQ quantized representations.
        if let Some(config) = approx.filter(|_| metric.supports_flat_approx()) {
            let candidates = self.as_f32_cow();
            if metric == DistanceMetric::InnerProduct
                && dim > super::approx_search::APPROX_BOUND_BLOCK_LEN
                && n > super::approx_search::APPROX_INIT_ROWS
            {
                let pool = approx_ip_order_pool_size(k, n, config.eps);
                self.ensure_approx_ip_order(pool);
                if let Some(order) = self.approx_ip_order.read().as_ref() {
                    return approx_ip_order_search(
                        query,
                        &candidates,
                        dim,
                        k,
                        n,
                        config.eps,
                        order,
                    );
                }
            }
            if matches!(metric, DistanceMetric::L2Squared | DistanceMetric::Cosine) {
                self.ensure_approx_norms();
                if let Some(norms) = self.approx_norms.read().as_ref() {
                    if let Some(result) = approx_norm_cached_search(
                        query,
                        &candidates,
                        norms,
                        dim,
                        k,
                        n,
                        metric,
                        config.eps,
                    ) {
                        return result;
                    }
                }
            }
            return super::approx_search::approx_flat_search(
                query,
                &candidates,
                dim,
                k,
                n,
                metric,
                config,
            );
        }

        // SQ8 two-pass path (only when user explicitly requested SQ8)
        if use_sq8
            && matches!(
                metric,
                DistanceMetric::InnerProduct | DistanceMetric::L2Squared | DistanceMetric::Cosine
            )
        {
            let candidates = self.as_f32_cow();
            self.ensure_sq8();
            let sq8_guard = self.sq8.read();
            if let Some(ref sq8) = *sq8_guard {
                return sq8_two_pass_search(query, &candidates, sq8, dim, k, n, metric);
            }
        }

        if self.dtype == VectorDtype::F16 {
            return exact_flat_search_f16(query, self.as_f16_bits_slice(), dim, k, n, metric);
        }

        // Full f32 scan
        let candidates = self.as_slice();
        exact_flat_search(query, candidates, dim, k, n, metric)
    }
}

fn exact_flat_search(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    metric: DistanceMetric,
) -> (Vec<u32>, Vec<f32>) {
    match metric {
        DistanceMetric::InnerProduct => fused_topk_ip_parallel(query, candidates, dim, k, n),
        DistanceMetric::L2Squared => {
            fused_topk_parallel::<true>(query, candidates, dim, k, n, simd::l2_squared_f32)
        }
        DistanceMetric::Cosine => {
            fused_topk_parallel::<true>(query, candidates, dim, k, n, simd::cosine_distance_f32)
        }
        DistanceMetric::Manhattan => {
            fused_topk_parallel::<true>(query, candidates, dim, k, n, simd::manhattan_f32)
        }
        DistanceMetric::Haversine => {
            fused_topk_parallel::<true>(query, candidates, dim, k, n, simd::haversine_meters_f32)
        }
        DistanceMetric::Correlation => fused_topk_parallel::<true>(
            query,
            candidates,
            dim,
            k,
            n,
            simd::correlation_distance_f32,
        ),
        DistanceMetric::Hellinger => {
            fused_topk_parallel::<true>(query, candidates, dim, k, n, simd::hellinger_distance_f32)
        }
        DistanceMetric::Wasserstein => {
            fused_topk_parallel::<true>(query, candidates, dim, k, n, simd::wasserstein_1d_f32)
        }
        _ => fused_topk_parallel::<true>(query, candidates, dim, k, n, |a, b| {
            distance::compute_distance_f32(a, b, metric)
        }),
    }
}

pub(crate) fn batch_exact_flat_search_f32(
    queries: &[f32],
    n_queries: usize,
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    metric: DistanceMetric,
) -> (Vec<Vec<u32>>, Vec<Vec<f32>>) {
    let results: Vec<(Vec<u32>, Vec<f32>)> = (0..n_queries)
        .into_par_iter()
        .map(|q| {
            let start = q * dim;
            let end = start + dim;
            exact_flat_search(&queries[start..end], candidates, dim, k, n, metric)
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

fn exact_flat_search_f16(
    query: &[f32],
    candidates: &[u16],
    dim: usize,
    k: usize,
    n: usize,
    metric: DistanceMetric,
) -> (Vec<u32>, Vec<f32>) {
    match metric {
        DistanceMetric::InnerProduct => {
            fused_topk_parallel_f16::<false>(query, candidates, dim, k, n, simd::inner_product_f16)
        }
        DistanceMetric::L2Squared => {
            fused_topk_parallel_f16::<true>(query, candidates, dim, k, n, simd::l2_squared_f16)
        }
        DistanceMetric::Cosine => {
            fused_topk_parallel_f16::<true>(query, candidates, dim, k, n, simd::cosine_distance_f16)
        }
        _ => fused_topk_parallel_f16::<true>(query, candidates, dim, k, n, |a, b| {
            distance::compute_distance_f16(a, b, metric)
        }),
    }
}

#[inline]
fn pack_binary_row_f32(source: &[f32], words: &mut [u64]) {
    for (index, &value) in source.iter().enumerate() {
        if value > 0.5 {
            words[index / 64] |= 1u64 << (index % 64);
        }
    }
}

fn pack_binary_query(query: &[f32]) -> Vec<u64> {
    let mut words = vec![0u64; query.len().div_ceil(64)];
    pack_binary_row_f32(query, &mut words);
    words
}

#[inline(always)]
fn packed_hamming(a: &[u64], b: &[u64]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum::<u32>() as f32
}

#[inline(always)]
fn packed_jaccard(a: &[u64], b: &[u64]) -> f32 {
    let mut intersection = 0u32;
    let mut union = 0u32;
    for (&x, &y) in a.iter().zip(b) {
        intersection += (x & y).count_ones();
        union += (x | y).count_ones();
    }
    if union == 0 {
        0.0
    } else {
        1.0 - intersection as f32 / union as f32
    }
}

#[inline(always)]
fn packed_dice(a: &[u64], b: &[u64]) -> f32 {
    let mut intersection = 0u32;
    let mut count = 0u32;
    for (&x, &y) in a.iter().zip(b) {
        intersection += (x & y).count_ones();
        count += x.count_ones() + y.count_ones();
    }
    if count == 0 {
        0.0
    } else {
        1.0 - (2 * intersection) as f32 / count as f32
    }
}

fn packed_distance_fn(metric: DistanceMetric) -> fn(&[u64], &[u64]) -> f32 {
    match metric {
        DistanceMetric::Hamming => packed_hamming,
        DistanceMetric::Jaccard | DistanceMetric::Tanimoto => packed_jaccard,
        DistanceMetric::Dice => packed_dice,
        _ => unreachable!("non-binary metric passed to packed search"),
    }
}

fn packed_binary_search(
    query: &[u64],
    binary: &BinaryData,
    k: usize,
    n: usize,
    metric: DistanceMetric,
) -> (Vec<u32>, Vec<f32>) {
    let words = binary.words_per_vector;
    let dist_fn = packed_distance_fn(metric);
    if n < 4096 {
        let mut top = Vec::with_capacity(k);
        let mut threshold = f32::INFINITY;
        let mut filled = false;
        for (index, row) in binary.data.chunks_exact(words).take(n).enumerate() {
            let dist = dist_fn(query, row);
            if !filled || dist < threshold {
                topk_insert::<true>(
                    &mut top,
                    k,
                    TopKEntry {
                        dist,
                        idx: index as u32,
                    },
                    &mut threshold,
                    &mut filled,
                );
            }
        }
        let ids = top.iter().map(|entry| entry.idx).collect();
        let distances = top.iter().map(|entry| entry.dist).collect();
        return (ids, distances);
    }

    let n_threads = rayon::current_num_threads().max(1);
    let chunk_rows = (n / n_threads).max(1024);
    let chunk_words = chunk_rows * words;
    let chunks: Vec<Vec<TopKEntry>> = binary
        .data
        .par_chunks(chunk_words)
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let base = chunk_index * chunk_rows;
            let mut top = Vec::with_capacity(k);
            let mut threshold = f32::INFINITY;
            let mut filled = false;
            for (local, row) in chunk.chunks_exact(words).enumerate() {
                let dist = dist_fn(query, row);
                if !filled || dist < threshold {
                    topk_insert::<true>(
                        &mut top,
                        k,
                        TopKEntry {
                            dist,
                            idx: (base + local) as u32,
                        },
                        &mut threshold,
                        &mut filled,
                    );
                }
            }
            top
        })
        .collect();
    merge_topk_results::<true>(&chunks, k)
}

fn packed_binary_search_filtered(
    query: &[u64],
    binary: &BinaryData,
    k: usize,
    n: usize,
    metric: DistanceMetric,
    subset: &[u64],
) -> (Vec<u32>, Vec<f32>) {
    let words = binary.words_per_vector;
    let dist_fn = packed_distance_fn(metric);
    let mut top = Vec::with_capacity(k);
    let mut threshold = f32::INFINITY;
    let mut filled = false;
    for &row_id in subset {
        let index = row_id as usize;
        if index >= n {
            continue;
        }
        let start = index * words;
        let row = &binary.data[start..start + words];
        let dist = dist_fn(query, row);
        if !filled || dist < threshold {
            topk_insert::<true>(
                &mut top,
                k,
                TopKEntry {
                    dist,
                    idx: index as u32,
                },
                &mut threshold,
                &mut filled,
            );
        }
    }
    let ids = top.iter().map(|entry| entry.idx).collect();
    let distances = top.iter().map(|entry| entry.dist).collect();
    (ids, distances)
}

// ─── Optimized fused parallel search ────────────────────────────────────────

/// Compact (dist, idx) pair for top-k tracking.
#[derive(Clone, Copy)]
struct TopKEntry {
    dist: f32,
    idx: u32,
}

impl PartialEq for TopKEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for TopKEntry {}
impl PartialOrd for TopKEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TopKEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
    }
}

#[inline]
fn row_sum_f32(values: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { row_sum_neon(values) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { row_sum_avx2(values) };
        }
    }

    #[allow(unreachable_code)]
    values.iter().copied().sum()
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn row_sum_neon(values: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = values.len();
    let chunks16 = n / 16;
    let remainder = n % 16;
    let ptr = values.as_ptr();

    let mut sum;
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        for i in 0..chunks16 {
            let base = i * 16;
            acc0 = vaddq_f32(acc0, vld1q_f32(ptr.add(base)));
            acc1 = vaddq_f32(acc1, vld1q_f32(ptr.add(base + 4)));
            acc2 = vaddq_f32(acc2, vld1q_f32(ptr.add(base + 8)));
            acc3 = vaddq_f32(acc3, vld1q_f32(ptr.add(base + 12)));
        }

        acc0 = vaddq_f32(acc0, acc1);
        acc2 = vaddq_f32(acc2, acc3);
        acc0 = vaddq_f32(acc0, acc2);
        sum = vaddvq_f32(acc0);
    }

    let base = chunks16 * 16;
    for i in 0..remainder {
        sum += values[base + i];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn row_sum_avx2(values: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = values.len();
    let chunks32 = n / 32;
    let remainder = n % 32;
    let ptr = values.as_ptr();

    let mut lanes = [0.0f32; 8];
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        for i in 0..chunks32 {
            let base = i * 32;
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(ptr.add(base)));
            acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(ptr.add(base + 8)));
            acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(ptr.add(base + 16)));
            acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(ptr.add(base + 24)));
        }
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc0);
    }

    let mut sum = lanes.into_iter().sum::<f32>();
    let base = chunks32 * 32;
    for i in 0..remainder {
        sum += values[base + i];
    }
    sum
}

fn build_approx_ip_row_order(
    candidates: &[f32],
    dim: usize,
    base_idx: usize,
    n: usize,
    pool: usize,
) -> (Vec<TopKEntry>, Vec<f32>, Vec<TopKEntry>, Vec<f32>) {
    let pool = pool.min(n);
    if pool == 0 {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let n_threads = rayon::current_num_threads().max(1);
    let chunk_vecs = n.div_ceil(n_threads).max(pool.saturating_mul(4)).max(1024);
    let chunk_floats = chunk_vecs * dim;
    let local: Vec<(Vec<TopKEntry>, Vec<TopKEntry>)> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let local_n = (chunk.len() / dim).min(n.saturating_sub(chunk_idx * chunk_vecs));
            let mut scored = Vec::with_capacity(local_n);
            for (idx, row) in chunk.chunks_exact(dim).take(local_n).enumerate() {
                let sum = row_sum_f32(row);
                if sum.is_finite() {
                    scored.push(TopKEntry {
                        dist: sum,
                        idx: (base_idx + chunk_idx * chunk_vecs + idx) as u32,
                    });
                }
            }

            let local_pool = pool.min(scored.len());
            let mut desc = scored.clone();
            truncate_entries_by(&mut desc, local_pool, cmp_entry_desc);
            truncate_entries_by(&mut scored, local_pool, cmp_entry_asc);
            (desc, scored)
        })
        .collect();

    let mut desc: Vec<TopKEntry> = local
        .iter()
        .flat_map(|(entries, _)| entries.iter().copied())
        .collect();
    let mut asc: Vec<TopKEntry> = local
        .into_iter()
        .flat_map(|(_, entries)| entries.into_iter())
        .collect();
    truncate_entries_by(&mut desc, pool, cmp_entry_desc);
    truncate_entries_by(&mut asc, pool, cmp_entry_asc);

    let should_cache_vectors = pool.saturating_mul(2).saturating_mul(dim).saturating_mul(4)
        <= APPROX_IP_ORDER_VECTOR_CACHE_MAX_BYTES;
    let desc_vectors = if should_cache_vectors {
        gather_cached_vectors(candidates, dim, base_idx, &desc)
    } else {
        Vec::new()
    };
    let asc_vectors = if should_cache_vectors {
        gather_cached_vectors(candidates, dim, base_idx, &asc)
    } else {
        Vec::new()
    };
    (desc, desc_vectors, asc, asc_vectors)
}

fn truncate_entries_by(
    entries: &mut Vec<TopKEntry>,
    limit: usize,
    cmp: fn(&TopKEntry, &TopKEntry) -> Ordering,
) {
    if limit == 0 {
        entries.clear();
        return;
    }
    if entries.len() > limit {
        entries.select_nth_unstable_by(limit - 1, cmp);
        entries.truncate(limit);
    }
    entries.sort_unstable_by(cmp);
}

fn gather_cached_vectors(
    candidates: &[f32],
    dim: usize,
    base_idx: usize,
    entries: &[TopKEntry],
) -> Vec<f32> {
    let mut vectors = Vec::with_capacity(entries.len() * dim);
    for entry in entries {
        let Some(local_idx) = (entry.idx as usize).checked_sub(base_idx) else {
            continue;
        };
        let start = local_idx * dim;
        let end = start + dim;
        if end <= candidates.len() {
            vectors.extend_from_slice(&candidates[start..end]);
        }
    }
    if vectors.len() == entries.len() * dim {
        vectors
    } else {
        Vec::new()
    }
}

fn build_approx_norms(candidates: &[f32], dim: usize, n: usize) -> ApproxNorms {
    if dim == 0 || n == 0 {
        return ApproxNorms {
            n_vectors: n,
            norm2: Vec::new(),
            inv_norm: Vec::new(),
        };
    }

    let norms: Vec<(f32, f32)> = candidates
        .par_chunks(dim)
        .take(n)
        .map(|row| {
            let norm2 = simd::inner_product_f32(row, row);
            let inv = if norm2 > 1e-30 {
                1.0 / norm2.sqrt()
            } else {
                0.0
            };
            (norm2, inv)
        })
        .collect();

    let mut norm2 = Vec::with_capacity(norms.len());
    let mut inv_norm = Vec::with_capacity(norms.len());
    for (n2, inv) in norms {
        norm2.push(n2);
        inv_norm.push(inv);
    }

    ApproxNorms {
        n_vectors: n,
        norm2,
        inv_norm,
    }
}

fn approx_norms_compatible(norms: &ApproxNorms, n: usize) -> bool {
    norms.n_vectors == n && norms.norm2.len() == n && norms.inv_norm.len() == n
}

fn update_approx_norms_after_append(
    old_norms: Option<ApproxNorms>,
    appended: &[f32],
    dim: usize,
    old_n: usize,
    new_n: usize,
) -> Option<ApproxNorms> {
    let appended_n = if dim == 0 { 0 } else { appended.len() / dim };
    if appended_n == 0 {
        return old_norms;
    }

    let appended_norms = build_approx_norms(appended, dim, appended_n);
    if let Some(mut norms) = old_norms.filter(|norms| approx_norms_compatible(norms, old_n)) {
        norms.norm2.extend(appended_norms.norm2);
        norms.inv_norm.extend(appended_norms.inv_norm);
        norms.n_vectors = new_n;
        Some(norms)
    } else if old_n == 0 {
        Some(ApproxNorms {
            n_vectors: new_n,
            norm2: appended_norms.norm2,
            inv_norm: appended_norms.inv_norm,
        })
    } else {
        None
    }
}

fn update_approx_ip_order_after_append(
    old_order: Option<ApproxIpOrder>,
    appended: &[f32],
    dim: usize,
    old_n: usize,
    new_n: usize,
) -> Option<ApproxIpOrder> {
    let appended_n = appended.len() / dim;
    if appended_n == 0 {
        return old_order;
    }

    let pool = old_order
        .as_ref()
        .map(|order| order.pool)
        .unwrap_or_else(|| APPROX_IP_ORDER_PREBUILD_POOL.min(new_n));

    if let Some(mut order) = old_order.filter(|order| order.n_vectors == old_n) {
        let append_pool = pool.min(appended_n);
        let (append_desc, append_desc_vectors, append_asc, append_asc_vectors) =
            build_approx_ip_row_order(appended, dim, old_n, appended_n, append_pool);

        let (sum_desc, sum_desc_vectors) = merge_cached_entries(
            order.sum_desc,
            order.sum_desc_vectors,
            append_desc,
            append_desc_vectors,
            dim,
            pool.min(new_n),
            cmp_entry_desc,
        );
        order.sum_desc = sum_desc;
        order.sum_desc_vectors = sum_desc_vectors;

        let (sum_asc, sum_asc_vectors) = merge_cached_entries(
            order.sum_asc,
            order.sum_asc_vectors,
            append_asc,
            append_asc_vectors,
            dim,
            pool.min(new_n),
            cmp_entry_asc,
        );
        order.sum_asc = sum_asc;
        order.sum_asc_vectors = sum_asc_vectors;

        order.n_vectors = new_n;
        order.pool = pool.min(new_n);
        Some(order)
    } else if old_n == 0 {
        let (sum_desc, sum_desc_vectors, sum_asc, sum_asc_vectors) =
            build_approx_ip_row_order(appended, dim, 0, appended_n, pool);
        Some(ApproxIpOrder {
            n_vectors: new_n,
            pool: pool.min(new_n),
            sum_desc,
            sum_desc_vectors,
            sum_asc,
            sum_asc_vectors,
        })
    } else {
        None
    }
}

fn merge_cached_entries(
    old_entries: Vec<TopKEntry>,
    old_vectors: Vec<f32>,
    new_entries: Vec<TopKEntry>,
    new_vectors: Vec<f32>,
    dim: usize,
    pool: usize,
    cmp: fn(&TopKEntry, &TopKEntry) -> Ordering,
) -> (Vec<TopKEntry>, Vec<f32>) {
    let old_has_vectors = cached_vectors_valid(&old_entries, &old_vectors, dim);
    let new_has_vectors = cached_vectors_valid(&new_entries, &new_vectors, dim);
    let mut positions: Vec<(TopKEntry, usize, bool)> =
        Vec::with_capacity(old_entries.len() + new_entries.len());
    positions.extend(
        old_entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| (*entry, idx, false)),
    );
    positions.extend(
        new_entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| (*entry, idx, true)),
    );
    positions.sort_unstable_by(|a, b| cmp(&a.0, &b.0));
    positions.truncate(pool);

    let mut entries = Vec::with_capacity(positions.len());
    let mut vectors = if old_has_vectors && new_has_vectors {
        Vec::with_capacity(positions.len() * dim)
    } else {
        Vec::new()
    };
    for (entry, idx, is_new) in positions {
        entries.push(entry);
        if old_has_vectors && new_has_vectors {
            let source = if is_new { &new_vectors } else { &old_vectors };
            let start = idx * dim;
            vectors.extend_from_slice(&source[start..start + dim]);
        }
    }
    (entries, vectors)
}

fn cached_vectors_valid(entries: &[TopKEntry], vectors: &[f32], dim: usize) -> bool {
    !entries.is_empty() && vectors.len() == entries.len() * dim
}

fn cached_vectors_cover(entries_len: usize, vectors: &[f32], dim: usize) -> bool {
    entries_len > 0 && vectors.len() >= entries_len * dim
}

fn cmp_entry_desc(a: &TopKEntry, b: &TopKEntry) -> Ordering {
    b.dist
        .partial_cmp(&a.dist)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.idx.cmp(&b.idx))
}

fn cmp_entry_asc(a: &TopKEntry, b: &TopKEntry) -> Ordering {
    a.dist
        .partial_cmp(&b.dist)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.idx.cmp(&b.idx))
}

fn approx_ip_order_path(path: &Path) -> PathBuf {
    let mut sidecar = path.as_os_str().to_os_string();
    sidecar.push(".iporder");
    PathBuf::from(sidecar)
}

fn approx_norms_path(path: &Path) -> PathBuf {
    let mut sidecar = path.as_os_str().to_os_string();
    sidecar.push(".norms");
    PathBuf::from(sidecar)
}

fn save_approx_ip_order(
    vector_path: &Path,
    dim: usize,
    vector_file_len: u64,
    order: &ApproxIpOrder,
) -> std::io::Result<()> {
    let path = approx_ip_order_path(vector_path);
    let mut file = BufWriter::new(File::create(path)?);
    file.write_all(APPROX_IP_ORDER_MAGIC)?;
    file.write_all(&(dim as u64).to_le_bytes())?;
    file.write_all(&(order.n_vectors as u64).to_le_bytes())?;
    file.write_all(&vector_file_len.to_le_bytes())?;
    file.write_all(&(order.pool as u64).to_le_bytes())?;
    write_entries(&mut file, &order.sum_desc, &order.sum_desc_vectors, dim)?;
    write_entries(&mut file, &order.sum_asc, &order.sum_asc_vectors, dim)?;
    file.flush()
}

fn write_entries<W: Write>(
    file: &mut W,
    entries: &[TopKEntry],
    vectors: &[f32],
    dim: usize,
) -> std::io::Result<()> {
    let has_vectors = cached_vectors_valid(entries, vectors, dim);
    file.write_all(&(entries.len() as u64).to_le_bytes())?;
    file.write_all(&(has_vectors as u64).to_le_bytes())?;
    for (i, entry) in entries.iter().enumerate() {
        file.write_all(&entry.dist.to_le_bytes())?;
        file.write_all(&entry.idx.to_le_bytes())?;
        if has_vectors {
            let start = i * dim;
            write_f32_slice_le(file, &vectors[start..start + dim])?;
        }
    }
    Ok(())
}

fn write_f32_slice_le<W: Write>(file: &mut W, values: &[f32]) -> std::io::Result<()> {
    #[cfg(target_endian = "little")]
    {
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
        };
        file.write_all(bytes)
    }

    #[cfg(not(target_endian = "little"))]
    {
        for value in values {
            file.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }
}

fn save_approx_norms(
    vector_path: &Path,
    dim: usize,
    vector_file_len: u64,
    norms: &ApproxNorms,
) -> std::io::Result<()> {
    let path = approx_norms_path(vector_path);
    let mut file = BufWriter::new(File::create(path)?);
    file.write_all(APPROX_NORMS_MAGIC)?;
    file.write_all(&(dim as u64).to_le_bytes())?;
    file.write_all(&(norms.n_vectors as u64).to_le_bytes())?;
    file.write_all(&vector_file_len.to_le_bytes())?;
    write_f32_slice_le(&mut file, &norms.norm2)?;
    write_f32_slice_le(&mut file, &norms.inv_norm)?;
    file.flush()
}

fn load_approx_norms(
    vector_path: &Path,
    dim: usize,
    n_vectors: usize,
    vector_file_len: u64,
) -> std::io::Result<Option<ApproxNorms>> {
    let path = approx_norms_path(vector_path);
    if !path.exists() {
        return Ok(None);
    }

    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let expected_len = APPROX_NORMS_MAGIC.len() + 8 * 3 + n_vectors * 4 * 2;
    if bytes.len() != expected_len {
        return Ok(None);
    }

    let mut pos = 0usize;
    if read_bytes(&bytes, &mut pos, APPROX_NORMS_MAGIC.len()) != Some(APPROX_NORMS_MAGIC) {
        return Ok(None);
    }

    let stored_dim = read_u64(&bytes, &mut pos).unwrap_or_default() as usize;
    let stored_n = read_u64(&bytes, &mut pos).unwrap_or_default() as usize;
    let stored_file_len = read_u64(&bytes, &mut pos).unwrap_or_default();
    if stored_dim != dim || stored_n != n_vectors || stored_file_len != vector_file_len {
        return Ok(None);
    }

    let Some(norm2) = read_f32_vec_le(&bytes, &mut pos, n_vectors) else {
        return Ok(None);
    };
    let Some(inv_norm) = read_f32_vec_le(&bytes, &mut pos, n_vectors) else {
        return Ok(None);
    };
    if norm2.iter().any(|value| !value.is_finite())
        || inv_norm.iter().any(|value| !value.is_finite())
    {
        return Ok(None);
    }

    Ok(Some(ApproxNorms {
        n_vectors,
        norm2,
        inv_norm,
    }))
}

fn read_f32_vec_le(bytes: &[u8], pos: &mut usize, len: usize) -> Option<Vec<f32>> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        let value_bytes = read_bytes(bytes, pos, 4)?;
        values.push(f32::from_le_bytes(value_bytes.try_into().ok()?));
    }
    Some(values)
}

fn load_approx_ip_order(
    vector_path: &Path,
    dim: usize,
    n_vectors: usize,
    vector_file_len: u64,
) -> std::io::Result<Option<ApproxIpOrder>> {
    let path = approx_ip_order_path(vector_path);
    if !path.exists() {
        return Ok(None);
    }

    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let mut pos = 0usize;
    if read_bytes(&bytes, &mut pos, APPROX_IP_ORDER_MAGIC.len()) != Some(APPROX_IP_ORDER_MAGIC) {
        return Ok(None);
    }

    let stored_dim = read_u64(&bytes, &mut pos).unwrap_or_default() as usize;
    let stored_n = read_u64(&bytes, &mut pos).unwrap_or_default() as usize;
    let stored_file_len = read_u64(&bytes, &mut pos).unwrap_or_default();
    let pool = read_u64(&bytes, &mut pos).unwrap_or_default() as usize;
    if stored_dim != dim
        || stored_n != n_vectors
        || stored_file_len != vector_file_len
        || pool == 0
        || pool > n_vectors
    {
        return Ok(None);
    }

    let (sum_desc, sum_desc_vectors) = read_entries(&bytes, &mut pos, n_vectors, dim)?;
    let (sum_asc, sum_asc_vectors) = read_entries(&bytes, &mut pos, n_vectors, dim)?;
    if sum_desc.len() > pool || sum_asc.len() > pool {
        return Ok(None);
    }
    if n_vectors > 0 && (sum_desc.is_empty() || sum_asc.is_empty()) {
        return Ok(None);
    }

    Ok(Some(ApproxIpOrder {
        n_vectors,
        pool,
        sum_desc,
        sum_desc_vectors,
        sum_asc,
        sum_asc_vectors,
    }))
}

fn read_entries(
    bytes: &[u8],
    pos: &mut usize,
    n_vectors: usize,
    dim: usize,
) -> std::io::Result<(Vec<TopKEntry>, Vec<f32>)> {
    let len = read_u64(bytes, pos).unwrap_or_default() as usize;
    let has_vectors = read_u64(bytes, pos).unwrap_or_default() != 0;
    if len > n_vectors {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut entries = Vec::with_capacity(len);
    let mut vectors = if has_vectors {
        Vec::with_capacity(len * dim)
    } else {
        Vec::new()
    };
    for _ in 0..len {
        let Some(dist_bytes) = read_bytes(bytes, pos, 4) else {
            return Ok((Vec::new(), Vec::new()));
        };
        let Some(idx_bytes) = read_bytes(bytes, pos, 4) else {
            return Ok((Vec::new(), Vec::new()));
        };
        let dist = f32::from_le_bytes(dist_bytes.try_into().unwrap());
        let idx = u32::from_le_bytes(idx_bytes.try_into().unwrap());
        if idx as usize >= n_vectors || !dist.is_finite() {
            return Ok((Vec::new(), Vec::new()));
        }
        entries.push(TopKEntry { dist, idx });
        if has_vectors {
            for _ in 0..dim {
                let Some(value_bytes) = read_bytes(bytes, pos, 4) else {
                    return Ok((Vec::new(), Vec::new()));
                };
                let value = f32::from_le_bytes(value_bytes.try_into().unwrap());
                if !value.is_finite() {
                    return Ok((Vec::new(), Vec::new()));
                }
                vectors.push(value);
            }
        }
    }
    Ok((entries, vectors))
}

fn read_u64(bytes: &[u8], pos: &mut usize) -> Option<u64> {
    let value = read_bytes(bytes, pos, 8)?;
    Some(u64::from_le_bytes(value.try_into().ok()?))
}

fn read_bytes<'a>(bytes: &'a [u8], pos: &mut usize, len: usize) -> Option<&'a [u8]> {
    let end = pos.checked_add(len)?;
    let slice = bytes.get(*pos..end)?;
    *pos = end;
    Some(slice)
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
    if ASC {
        dist < threshold
    } else {
        dist > threshold
    }
}

/// Scan one parallel chunk for IP top-k using batch-8 dot products + prefetch.
fn ip_scan_chunk_topk(
    query: &[f32],
    cand_chunk: &[f32],
    dim: usize,
    k: usize,
    base_idx: usize,
) -> Vec<TopKEntry> {
    let n_in_chunk = cand_chunk.len() / dim;
    let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
    let mut threshold = f32::NEG_INFINITY;
    let mut filled = false;

    let blocks8 = n_in_chunk / 8;
    for block in 0..blocks8 {
        if block + 1 < blocks8 {
            unsafe {
                prefetch_read_data(cand_chunk.as_ptr().add((block + 1) * 8 * dim).cast());
            }
        }
        let base = block * 8 * dim;
        let v0 = unsafe { cand_chunk.get_unchecked(base..base + dim) };
        let v1 = unsafe { cand_chunk.get_unchecked(base + dim..base + 2 * dim) };
        let v2 = unsafe { cand_chunk.get_unchecked(base + 2 * dim..base + 3 * dim) };
        let v3 = unsafe { cand_chunk.get_unchecked(base + 3 * dim..base + 4 * dim) };
        let v4 = unsafe { cand_chunk.get_unchecked(base + 4 * dim..base + 5 * dim) };
        let v5 = unsafe { cand_chunk.get_unchecked(base + 5 * dim..base + 6 * dim) };
        let v6 = unsafe { cand_chunk.get_unchecked(base + 6 * dim..base + 7 * dim) };
        let v7 = unsafe { cand_chunk.get_unchecked(base + 7 * dim..base + 8 * dim) };
        let dists = simd::inner_product_batch8_f32(query, v0, v1, v2, v3, v4, v5, v6, v7);
        for (j, &dist) in dists.iter().enumerate() {
            if !filled || dist > threshold {
                topk_insert::<false>(
                    &mut top,
                    k,
                    TopKEntry {
                        dist,
                        idx: (base_idx + block * 8 + j) as u32,
                    },
                    &mut threshold,
                    &mut filled,
                );
            }
        }
    }

    let rem_start = blocks8 * 8;
    for i in rem_start..n_in_chunk {
        if i + PREFETCH_AHEAD_VECTORS < n_in_chunk {
            unsafe {
                prefetch_read_data(
                    cand_chunk
                        .as_ptr()
                        .add((i + PREFETCH_AHEAD_VECTORS) * dim)
                        .cast(),
                );
            }
        }
        let cand = unsafe { cand_chunk.get_unchecked(i * dim..(i + 1) * dim) };
        let dist = simd::inner_product_f32(query, cand);
        if !filled || dist > threshold {
            topk_insert::<false>(
                &mut top,
                k,
                TopKEntry {
                    dist,
                    idx: (base_idx + i) as u32,
                },
                &mut threshold,
                &mut filled,
            );
        }
    }

    if !filled && !top.is_empty() {
        top.sort_unstable_by(|a, b| b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal));
    }
    top
}

fn first_local_row(
    base_idx: usize,
    n_in_chunk: usize,
    row_stride: usize,
    row_offset: usize,
) -> usize {
    if n_in_chunk == 0 {
        return 0;
    }
    let base_mod = base_idx % row_stride;
    if base_mod <= row_offset {
        row_offset - base_mod
    } else {
        row_stride - (base_mod - row_offset)
    }
}

fn contiguous_exact_search(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    metric: DistanceMetric,
    start: usize,
    len: usize,
) -> (Vec<u32>, Vec<f32>) {
    let start_f = start * dim;
    let end_f = start_f + len * dim;
    let slice = unsafe { candidates.get_unchecked(start_f..end_f) };
    match metric {
        DistanceMetric::InnerProduct => contiguous_ip_exact_topk(query, slice, dim, k, len, start),
        DistanceMetric::L2Squared => {
            contiguous_exact_topk::<true>(query, slice, dim, k, len, start, simd::l2_squared_f32)
        }
        DistanceMetric::Cosine => contiguous_exact_topk::<true>(
            query,
            slice,
            dim,
            k,
            len,
            start,
            simd::cosine_distance_f32,
        ),
        _ => contiguous_exact_topk::<true>(query, slice, dim, k, len, start, |a, b| {
            distance::compute_distance_f32(a, b, metric)
        }),
    }
}

fn contiguous_ip_exact_topk(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    base_offset: usize,
) -> (Vec<u32>, Vec<f32>) {
    if n < 4096 {
        return fused_topk_seq::<false>(
            query,
            candidates,
            dim,
            k,
            base_offset,
            &simd::inner_product_f32,
        );
    }

    let n_threads = rayon::current_num_threads();
    let chunk_vecs = (n / n_threads).max(512);
    let chunk_floats = chunk_vecs * dim;
    let chunk_results: Vec<Vec<TopKEntry>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            ip_scan_chunk_topk(
                query,
                cand_chunk,
                dim,
                k,
                base_offset + chunk_idx * chunk_vecs,
            )
        })
        .collect();
    let result = merge_topk_results::<false>(&chunk_results, k);
    if result.0.len() < k {
        contiguous_exact_search(
            query,
            candidates,
            dim,
            k,
            DistanceMetric::InnerProduct,
            0,
            n,
        )
    } else {
        result
    }
}

fn contiguous_exact_topk<const ASC: bool>(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    base_offset: usize,
    dist_fn: impl Fn(&[f32], &[f32]) -> f32 + Send + Sync,
) -> (Vec<u32>, Vec<f32>) {
    if n < 4096 {
        return fused_topk_seq::<ASC>(query, candidates, dim, k, base_offset, &dist_fn);
    }

    let n_threads = rayon::current_num_threads();
    let chunk_vecs = (n / n_threads).max(512);
    let chunk_floats = chunk_vecs * dim;
    let chunk_results: Vec<Vec<TopKEntry>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let n_in_chunk = cand_chunk.len() / dim;
            let base_idx = base_offset + chunk_idx * chunk_vecs;
            let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
            let mut threshold: f32 = if ASC {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
            let mut filled = false;

            let mut ptr = cand_chunk.as_ptr();
            let pairs = n_in_chunk / 2;
            let dim2 = dim * 2;
            for i in 0..pairs {
                if i + 4 < pairs {
                    unsafe {
                        prefetch_read_data(ptr.add((i + 4) * dim2).cast());
                    }
                }
                let cand0 = unsafe { std::slice::from_raw_parts(ptr, dim) };
                let cand1 = unsafe { std::slice::from_raw_parts(ptr.wrapping_add(dim), dim) };
                let dist0 = dist_fn(query, cand0);
                let dist1 = dist_fn(query, cand1);
                let idx0 = (base_idx + i * 2) as u32;
                if !filled || passes_threshold::<ASC>(dist0, threshold) {
                    topk_insert::<ASC>(
                        &mut top,
                        k,
                        TopKEntry {
                            dist: dist0,
                            idx: idx0,
                        },
                        &mut threshold,
                        &mut filled,
                    );
                }
                if !filled || passes_threshold::<ASC>(dist1, threshold) {
                    topk_insert::<ASC>(
                        &mut top,
                        k,
                        TopKEntry {
                            dist: dist1,
                            idx: idx0 + 1,
                        },
                        &mut threshold,
                        &mut filled,
                    );
                }
                ptr = unsafe { ptr.add(dim2) };
            }
            if n_in_chunk % 2 == 1 {
                let cand = unsafe { std::slice::from_raw_parts(ptr, dim) };
                let dist = dist_fn(query, cand);
                if !filled || passes_threshold::<ASC>(dist, threshold) {
                    topk_insert::<ASC>(
                        &mut top,
                        k,
                        TopKEntry {
                            dist,
                            idx: (base_idx + n_in_chunk - 1) as u32,
                        },
                        &mut threshold,
                        &mut filled,
                    );
                }
            }

            if !filled && !top.is_empty() {
                if ASC {
                    top.sort_unstable_by(|a, b| {
                        a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal)
                    });
                } else {
                    top.sort_unstable_by(|a, b| {
                        b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal)
                    });
                }
            }
            top
        })
        .collect();

    merge_topk_results::<ASC>(&chunk_results, k)
}

#[allow(dead_code)]
fn approx_global_shortlist_search(
    query: &[f32],
    candidates: &[f32],
    norms: &ApproxNorms,
    dim: usize,
    k: usize,
    n: usize,
    metric: DistanceMetric,
    eps: f32,
) -> Option<(Vec<u32>, Vec<f32>)> {
    if !approx_norms_compatible(norms, n) {
        return None;
    }
    let sample_dims = approx_global_sample_dims(metric, dim, eps);
    let pool = approx_global_pool_size(metric, k, n, eps);
    if sample_dims >= dim || pool >= n || pool < k {
        return None;
    }

    let subset_ids = match metric {
        DistanceMetric::L2Squared => {
            coarse_l2_global_shortlist(query, candidates, dim, sample_dims, pool, n)
        }
        DistanceMetric::Cosine => {
            let q_norm2 = simd::inner_product_f32(query, query);
            if q_norm2 <= 1e-30 {
                return None;
            }
            coarse_cosine_global_shortlist(
                query,
                candidates,
                &norms.inv_norm,
                dim,
                sample_dims,
                pool,
                n,
                1.0 / q_norm2.sqrt(),
            )
        }
        _ => return None,
    };
    if subset_ids.len() < k {
        return None;
    }

    let result = match metric {
        DistanceMetric::L2Squared => {
            let q_norm2 = simd::inner_product_f32(query, query);
            direct_access_l2_norm_cached_topk(
                query,
                candidates,
                &norms.norm2,
                dim,
                k,
                n,
                &subset_ids,
                q_norm2,
            )
        }
        DistanceMetric::Cosine => {
            let q_norm2 = simd::inner_product_f32(query, query);
            let q_inv = if q_norm2 > 1e-30 {
                1.0 / q_norm2.sqrt()
            } else {
                0.0
            };
            if q_inv <= 0.0 {
                return None;
            }
            direct_access_cosine_norm_cached_topk(
                query,
                candidates,
                &norms.inv_norm,
                dim,
                k,
                n,
                &subset_ids,
                q_inv,
            )
        }
        _ => return None,
    };

    let (ids, dists) = result;
    let dists = dists
        .into_iter()
        .map(|dist| super::approx_search::round_to_eps(dist.max(0.0), eps))
        .collect();
    Some((ids, dists))
}

#[allow(dead_code)]
fn coarse_l2_global_shortlist(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    sample_dims: usize,
    pool: usize,
    n: usize,
) -> Vec<u64> {
    let q = unsafe { query.get_unchecked(..sample_dims) };
    let mut scores = vec![
        TopKEntry {
            dist: f32::INFINITY,
            idx: 0,
        };
        n
    ];
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_vecs = n.div_ceil(n_threads).max(4096);

    scores
        .par_chunks_mut(chunk_vecs)
        .enumerate()
        .for_each(|(chunk_idx, score_chunk)| {
            let base_idx = chunk_idx * chunk_vecs;
            let n_in_chunk = score_chunk.len().min(n.saturating_sub(base_idx));
            let start_f = base_idx * dim;
            let cand_chunk =
                unsafe { candidates.get_unchecked(start_f..start_f + n_in_chunk * dim) };

            let groups8 = n_in_chunk / 8;
            for group in 0..groups8 {
                if group + 1 < groups8 {
                    unsafe {
                        prefetch_read_data(cand_chunk.as_ptr().add((group + 1) * 8 * dim).cast());
                    }
                }
                let row = group * 8;
                let base = row * dim;
                let v0 = unsafe { cand_chunk.get_unchecked(base..base + sample_dims) };
                let v1 = unsafe { cand_chunk.get_unchecked(base + dim..base + dim + sample_dims) };
                let v2 = unsafe {
                    cand_chunk.get_unchecked(base + 2 * dim..base + 2 * dim + sample_dims)
                };
                let v3 = unsafe {
                    cand_chunk.get_unchecked(base + 3 * dim..base + 3 * dim + sample_dims)
                };
                let v4 = unsafe {
                    cand_chunk.get_unchecked(base + 4 * dim..base + 4 * dim + sample_dims)
                };
                let v5 = unsafe {
                    cand_chunk.get_unchecked(base + 5 * dim..base + 5 * dim + sample_dims)
                };
                let v6 = unsafe {
                    cand_chunk.get_unchecked(base + 6 * dim..base + 6 * dim + sample_dims)
                };
                let v7 = unsafe {
                    cand_chunk.get_unchecked(base + 7 * dim..base + 7 * dim + sample_dims)
                };
                let dists = simd::l2_squared_batch8_f32(q, v0, v1, v2, v3, v4, v5, v6, v7);
                for (offset, &dist) in dists.iter().enumerate() {
                    score_chunk[row + offset] = TopKEntry {
                        dist,
                        idx: (base_idx + row + offset) as u32,
                    };
                }
            }

            for local in groups8 * 8..n_in_chunk {
                let cand =
                    unsafe { cand_chunk.get_unchecked(local * dim..local * dim + sample_dims) };
                score_chunk[local] = TopKEntry {
                    dist: simd::l2_squared_f32(q, cand),
                    idx: (base_idx + local) as u32,
                };
            }
        });

    truncate_entries_by(&mut scores, pool.min(n), cmp_entry_asc);
    let mut ids: Vec<u64> = scores.into_iter().map(|entry| entry.idx as u64).collect();
    ids.sort_unstable();
    ids
}

#[allow(dead_code)]
fn coarse_cosine_global_shortlist(
    query: &[f32],
    candidates: &[f32],
    inv_norm: &[f32],
    dim: usize,
    sample_dims: usize,
    pool: usize,
    n: usize,
    q_inv: f32,
) -> Vec<u64> {
    let q = unsafe { query.get_unchecked(..sample_dims) };
    let mut scores = vec![
        TopKEntry {
            dist: f32::INFINITY,
            idx: 0,
        };
        n
    ];
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_vecs = n.div_ceil(n_threads).max(4096);

    scores
        .par_chunks_mut(chunk_vecs)
        .enumerate()
        .for_each(|(chunk_idx, score_chunk)| {
            let base_idx = chunk_idx * chunk_vecs;
            let n_in_chunk = score_chunk.len().min(n.saturating_sub(base_idx));
            let start_f = base_idx * dim;
            let cand_chunk =
                unsafe { candidates.get_unchecked(start_f..start_f + n_in_chunk * dim) };
            let inv_chunk = unsafe { inv_norm.get_unchecked(base_idx..base_idx + n_in_chunk) };

            let groups8 = n_in_chunk / 8;
            for group in 0..groups8 {
                if group + 1 < groups8 {
                    unsafe {
                        prefetch_read_data(cand_chunk.as_ptr().add((group + 1) * 8 * dim).cast());
                    }
                }
                let row = group * 8;
                let base = row * dim;
                let v0 = unsafe { cand_chunk.get_unchecked(base..base + sample_dims) };
                let v1 = unsafe { cand_chunk.get_unchecked(base + dim..base + dim + sample_dims) };
                let v2 = unsafe {
                    cand_chunk.get_unchecked(base + 2 * dim..base + 2 * dim + sample_dims)
                };
                let v3 = unsafe {
                    cand_chunk.get_unchecked(base + 3 * dim..base + 3 * dim + sample_dims)
                };
                let v4 = unsafe {
                    cand_chunk.get_unchecked(base + 4 * dim..base + 4 * dim + sample_dims)
                };
                let v5 = unsafe {
                    cand_chunk.get_unchecked(base + 5 * dim..base + 5 * dim + sample_dims)
                };
                let v6 = unsafe {
                    cand_chunk.get_unchecked(base + 6 * dim..base + 6 * dim + sample_dims)
                };
                let v7 = unsafe {
                    cand_chunk.get_unchecked(base + 7 * dim..base + 7 * dim + sample_dims)
                };
                let dots = simd::inner_product_batch8_f32(q, v0, v1, v2, v3, v4, v5, v6, v7);
                for (offset, &dot) in dots.iter().enumerate() {
                    let local = row + offset;
                    let row_inv = unsafe { *inv_chunk.get_unchecked(local) };
                    let dist = if row_inv > 0.0 {
                        -dot * q_inv * row_inv
                    } else {
                        f32::INFINITY
                    };
                    score_chunk[local] = TopKEntry {
                        dist,
                        idx: (base_idx + local) as u32,
                    };
                }
            }

            for local in groups8 * 8..n_in_chunk {
                let cand =
                    unsafe { cand_chunk.get_unchecked(local * dim..local * dim + sample_dims) };
                let row_inv = unsafe { *inv_chunk.get_unchecked(local) };
                let dist = if row_inv > 0.0 {
                    -simd::inner_product_f32(q, cand) * q_inv * row_inv
                } else {
                    f32::INFINITY
                };
                score_chunk[local] = TopKEntry {
                    dist,
                    idx: (base_idx + local) as u32,
                };
            }
        });

    truncate_entries_by(&mut scores, pool.min(n), cmp_entry_asc);
    let mut ids: Vec<u64> = scores.into_iter().map(|entry| entry.idx as u64).collect();
    ids.sort_unstable();
    ids
}

#[inline]
#[allow(dead_code)]
fn approx_global_sample_dims(metric: DistanceMetric, dim: usize, eps: f32) -> usize {
    let (num, den) = match metric {
        DistanceMetric::L2Squared => {
            if eps <= 1e-4 {
                (1usize, 2usize)
            } else if eps <= 1e-3 {
                (3, 8)
            } else {
                (1, 4)
            }
        }
        DistanceMetric::Cosine => {
            if eps <= 1e-4 {
                (5usize, 8usize)
            } else if eps <= 1e-3 {
                (1, 2)
            } else {
                (3, 8)
            }
        }
        _ => (1, 1),
    };
    let sample = (dim.saturating_mul(num) / den).max(32).min(dim);
    ((sample / 16) * 16).max(16).min(dim)
}

#[inline]
#[allow(dead_code)]
fn approx_global_pool_size(metric: DistanceMetric, k: usize, n: usize, eps: f32) -> usize {
    let pool = match metric {
        DistanceMetric::L2Squared => {
            if eps <= 1e-4 {
                (n / 20).max(k.saturating_mul(5_000))
            } else if eps <= 1e-3 {
                (n / 40).max(k.saturating_mul(2_500))
            } else {
                (n / 80).max(k.saturating_mul(1_250))
            }
        }
        DistanceMetric::Cosine => {
            if eps <= 1e-4 {
                (n.saturating_mul(15) / 100).max(k.saturating_mul(15_000))
            } else if eps <= 1e-3 {
                (n / 12).max(k.saturating_mul(8_000))
            } else {
                (n / 20).max(k.saturating_mul(5_000))
            }
        }
        _ => n,
    };
    pool.max(k + 128).min(n)
}

#[allow(dead_code)]
fn direct_access_l2_norm_cached_topk(
    query: &[f32],
    candidates: &[f32],
    norm2: &[f32],
    dim: usize,
    k: usize,
    n_vectors: usize,
    subset_ids: &[u64],
    q_norm2: f32,
) -> (Vec<u32>, Vec<f32>) {
    let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
    let mut threshold = f32::INFINITY;
    let mut filled = false;

    for &id in subset_ids {
        let idx = id as usize;
        if idx >= n_vectors {
            continue;
        }
        let start = idx * dim;
        let cand = unsafe { candidates.get_unchecked(start..start + dim) };
        let dist = q_norm2 + unsafe { *norm2.get_unchecked(idx) }
            - 2.0 * simd::inner_product_f32(query, cand);
        if !filled || dist < threshold {
            topk_insert::<true>(
                &mut top,
                k,
                TopKEntry {
                    dist,
                    idx: idx as u32,
                },
                &mut threshold,
                &mut filled,
            );
        }
    }

    if !filled && !top.is_empty() {
        top.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
    }
    let indices = top.iter().map(|entry| entry.idx).collect();
    let dists = top.iter().map(|entry| entry.dist).collect();
    (indices, dists)
}

#[allow(dead_code)]
fn direct_access_cosine_norm_cached_topk(
    query: &[f32],
    candidates: &[f32],
    inv_norm: &[f32],
    dim: usize,
    k: usize,
    n_vectors: usize,
    subset_ids: &[u64],
    q_inv: f32,
) -> (Vec<u32>, Vec<f32>) {
    let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
    let mut threshold = f32::INFINITY;
    let mut filled = false;

    for &id in subset_ids {
        let idx = id as usize;
        if idx >= n_vectors {
            continue;
        }
        let row_inv = unsafe { *inv_norm.get_unchecked(idx) };
        let dist = if row_inv <= 0.0 {
            1.0
        } else {
            let start = idx * dim;
            let cand = unsafe { candidates.get_unchecked(start..start + dim) };
            1.0 - simd::inner_product_f32(query, cand) * q_inv * row_inv
        };
        if !filled || dist < threshold {
            topk_insert::<true>(
                &mut top,
                k,
                TopKEntry {
                    dist,
                    idx: idx as u32,
                },
                &mut threshold,
                &mut filled,
            );
        }
    }

    if !filled && !top.is_empty() {
        top.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
    }
    let indices = top.iter().map(|entry| entry.idx).collect();
    let dists = top.iter().map(|entry| entry.dist).collect();
    (indices, dists)
}

fn approx_norm_cached_search(
    query: &[f32],
    candidates: &[f32],
    norms: &ApproxNorms,
    dim: usize,
    k: usize,
    n: usize,
    metric: DistanceMetric,
    eps: f32,
) -> Option<(Vec<u32>, Vec<f32>)> {
    if !approx_norms_compatible(norms, n) {
        return None;
    }

    let result = match metric {
        DistanceMetric::L2Squared => {
            let q_norm2 = simd::inner_product_f32(query, query);
            norm_cached_l2_topk(query, candidates, &norms.norm2, dim, k, n, q_norm2)
        }
        DistanceMetric::Cosine => {
            let q_norm2 = simd::inner_product_f32(query, query);
            let q_inv = if q_norm2 > 1e-30 {
                1.0 / q_norm2.sqrt()
            } else {
                0.0
            };
            norm_cached_cosine_topk(query, candidates, &norms.inv_norm, dim, k, n, q_inv)
        }
        _ => return None,
    };

    let (ids, dists) = result;
    let dists = dists
        .into_iter()
        .map(|dist| {
            let dist = if metric == DistanceMetric::L2Squared {
                dist.max(0.0)
            } else {
                dist
            };
            super::approx_search::round_to_eps(dist, eps)
        })
        .collect();
    Some((ids, dists))
}

fn norm_cached_l2_topk(
    query: &[f32],
    candidates: &[f32],
    norm2: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    q_norm2: f32,
) -> (Vec<u32>, Vec<f32>) {
    if n < 4096 {
        return norm_cached_l2_topk_seq(query, candidates, norm2, dim, k, n, 0, q_norm2);
    }

    let n_threads = rayon::current_num_threads().max(1);
    let chunk_vecs = n.div_ceil(n_threads).max(512);
    let chunk_floats = chunk_vecs * dim;
    let chunk_results: Vec<Vec<TopKEntry>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let base_idx = chunk_idx * chunk_vecs;
            let n_in_chunk = (cand_chunk.len() / dim).min(n.saturating_sub(base_idx));
            let norm_chunk = unsafe { norm2.get_unchecked(base_idx..base_idx + n_in_chunk) };
            let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
            let mut threshold = f32::INFINITY;
            let mut filled = false;

            let groups8 = n_in_chunk / 8;
            for group in 0..groups8 {
                if group + 1 < groups8 {
                    unsafe {
                        prefetch_read_data(cand_chunk.as_ptr().add((group + 1) * 8 * dim).cast());
                    }
                }
                let row = group * 8;
                let base = row * dim;
                let v0 = unsafe { cand_chunk.get_unchecked(base..base + dim) };
                let v1 = unsafe { cand_chunk.get_unchecked(base + dim..base + 2 * dim) };
                let v2 = unsafe { cand_chunk.get_unchecked(base + 2 * dim..base + 3 * dim) };
                let v3 = unsafe { cand_chunk.get_unchecked(base + 3 * dim..base + 4 * dim) };
                let v4 = unsafe { cand_chunk.get_unchecked(base + 4 * dim..base + 5 * dim) };
                let v5 = unsafe { cand_chunk.get_unchecked(base + 5 * dim..base + 6 * dim) };
                let v6 = unsafe { cand_chunk.get_unchecked(base + 6 * dim..base + 7 * dim) };
                let v7 = unsafe { cand_chunk.get_unchecked(base + 7 * dim..base + 8 * dim) };
                let dots = simd::inner_product_batch8_f32(query, v0, v1, v2, v3, v4, v5, v6, v7);
                for offset in 0..8 {
                    let local = row + offset;
                    let dist =
                        q_norm2 + unsafe { *norm_chunk.get_unchecked(local) } - 2.0 * dots[offset];
                    if !filled || dist < threshold {
                        topk_insert::<true>(
                            &mut top,
                            k,
                            TopKEntry {
                                dist,
                                idx: (base_idx + local) as u32,
                            },
                            &mut threshold,
                            &mut filled,
                        );
                    }
                }
            }

            for local in groups8 * 8..n_in_chunk {
                let idx = base_idx + local;
                if local + PREFETCH_AHEAD_VECTORS < n_in_chunk {
                    unsafe {
                        prefetch_read_data(
                            cand_chunk
                                .as_ptr()
                                .add((local + PREFETCH_AHEAD_VECTORS) * dim)
                                .cast(),
                        );
                    }
                }
                let cand = unsafe { cand_chunk.get_unchecked(local * dim..(local + 1) * dim) };
                let dot = simd::inner_product_f32(query, cand);
                let dist = q_norm2 + unsafe { *norm_chunk.get_unchecked(local) } - 2.0 * dot;
                if !filled || dist < threshold {
                    topk_insert::<true>(
                        &mut top,
                        k,
                        TopKEntry {
                            dist,
                            idx: idx as u32,
                        },
                        &mut threshold,
                        &mut filled,
                    );
                }
            }

            if !filled && !top.is_empty() {
                top.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
            }
            top
        })
        .collect();

    merge_topk_results::<true>(&chunk_results, k)
}

fn norm_cached_l2_topk_seq(
    query: &[f32],
    candidates: &[f32],
    norm2: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    base_offset: usize,
    q_norm2: f32,
) -> (Vec<u32>, Vec<f32>) {
    let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
    let mut threshold = f32::INFINITY;
    let mut filled = false;

    for local in 0..n {
        let idx = base_offset + local;
        let cand = unsafe { candidates.get_unchecked(local * dim..(local + 1) * dim) };
        let dot = simd::inner_product_f32(query, cand);
        let dist = q_norm2 + unsafe { *norm2.get_unchecked(idx) } - 2.0 * dot;
        if !filled || dist < threshold {
            topk_insert::<true>(
                &mut top,
                k,
                TopKEntry {
                    dist,
                    idx: idx as u32,
                },
                &mut threshold,
                &mut filled,
            );
        }
    }

    if !filled && !top.is_empty() {
        top.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
    }

    let indices = top.iter().map(|e| e.idx).collect();
    let dists = top.iter().map(|e| e.dist).collect();
    (indices, dists)
}

fn norm_cached_cosine_topk(
    query: &[f32],
    candidates: &[f32],
    inv_norm: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    q_inv: f32,
) -> (Vec<u32>, Vec<f32>) {
    if q_inv <= 0.0 {
        let ids: Vec<u32> = (0..k.min(n) as u32).collect();
        return (ids, vec![1.0; k.min(n)]);
    }
    if n < 4096 {
        return norm_cached_cosine_topk_seq(query, candidates, inv_norm, dim, k, n, 0, q_inv);
    }

    let n_threads = rayon::current_num_threads().max(1);
    let chunk_vecs = n.div_ceil(n_threads).max(512);
    let chunk_floats = chunk_vecs * dim;
    let chunk_results: Vec<Vec<TopKEntry>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let base_idx = chunk_idx * chunk_vecs;
            let n_in_chunk = (cand_chunk.len() / dim).min(n.saturating_sub(base_idx));
            let inv_chunk = unsafe { inv_norm.get_unchecked(base_idx..base_idx + n_in_chunk) };
            let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
            let mut threshold = f32::INFINITY;
            let mut filled = false;

            let groups8 = n_in_chunk / 8;
            for group in 0..groups8 {
                if group + 1 < groups8 {
                    unsafe {
                        prefetch_read_data(cand_chunk.as_ptr().add((group + 1) * 8 * dim).cast());
                    }
                }
                let row = group * 8;
                let base = row * dim;
                let v0 = unsafe { cand_chunk.get_unchecked(base..base + dim) };
                let v1 = unsafe { cand_chunk.get_unchecked(base + dim..base + 2 * dim) };
                let v2 = unsafe { cand_chunk.get_unchecked(base + 2 * dim..base + 3 * dim) };
                let v3 = unsafe { cand_chunk.get_unchecked(base + 3 * dim..base + 4 * dim) };
                let v4 = unsafe { cand_chunk.get_unchecked(base + 4 * dim..base + 5 * dim) };
                let v5 = unsafe { cand_chunk.get_unchecked(base + 5 * dim..base + 6 * dim) };
                let v6 = unsafe { cand_chunk.get_unchecked(base + 6 * dim..base + 7 * dim) };
                let v7 = unsafe { cand_chunk.get_unchecked(base + 7 * dim..base + 8 * dim) };
                let dots = simd::inner_product_batch8_f32(query, v0, v1, v2, v3, v4, v5, v6, v7);
                for offset in 0..8 {
                    let local = row + offset;
                    let row_inv = unsafe { *inv_chunk.get_unchecked(local) };
                    let dist = if row_inv <= 0.0 {
                        1.0
                    } else {
                        1.0 - dots[offset] * q_inv * row_inv
                    };
                    if !filled || dist < threshold {
                        topk_insert::<true>(
                            &mut top,
                            k,
                            TopKEntry {
                                dist,
                                idx: (base_idx + local) as u32,
                            },
                            &mut threshold,
                            &mut filled,
                        );
                    }
                }
            }

            for local in groups8 * 8..n_in_chunk {
                let idx = base_idx + local;
                if local + PREFETCH_AHEAD_VECTORS < n_in_chunk {
                    unsafe {
                        prefetch_read_data(
                            cand_chunk
                                .as_ptr()
                                .add((local + PREFETCH_AHEAD_VECTORS) * dim)
                                .cast(),
                        );
                    }
                }
                let cand = unsafe { cand_chunk.get_unchecked(local * dim..(local + 1) * dim) };
                let row_inv = unsafe { *inv_chunk.get_unchecked(local) };
                let dist = if row_inv <= 0.0 {
                    1.0
                } else {
                    1.0 - simd::inner_product_f32(query, cand) * q_inv * row_inv
                };
                if !filled || dist < threshold {
                    topk_insert::<true>(
                        &mut top,
                        k,
                        TopKEntry {
                            dist,
                            idx: idx as u32,
                        },
                        &mut threshold,
                        &mut filled,
                    );
                }
            }

            if !filled && !top.is_empty() {
                top.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
            }
            top
        })
        .collect();

    merge_topk_results::<true>(&chunk_results, k)
}

fn norm_cached_cosine_topk_seq(
    query: &[f32],
    candidates: &[f32],
    inv_norm: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    base_offset: usize,
    q_inv: f32,
) -> (Vec<u32>, Vec<f32>) {
    let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
    let mut threshold = f32::INFINITY;
    let mut filled = false;

    for local in 0..n {
        let idx = base_offset + local;
        let cand = unsafe { candidates.get_unchecked(local * dim..(local + 1) * dim) };
        let row_inv = unsafe { *inv_norm.get_unchecked(idx) };
        let dist = if row_inv <= 0.0 {
            1.0
        } else {
            1.0 - simd::inner_product_f32(query, cand) * q_inv * row_inv
        };
        if !filled || dist < threshold {
            topk_insert::<true>(
                &mut top,
                k,
                TopKEntry {
                    dist,
                    idx: idx as u32,
                },
                &mut threshold,
                &mut filled,
            );
        }
    }

    if !filled && !top.is_empty() {
        top.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
    }

    let indices = top.iter().map(|e| e.idx).collect();
    let dists = top.iter().map(|e| e.dist).collect();
    (indices, dists)
}

/// Raw-f32 hybrid top-k for `approx=True`.
///
/// The coarse phase scores a deterministic set of contiguous dimension blocks
/// from the original vectors, then the fine phase re-scores only the shortlist
/// with the full exact f32 distance.
pub(super) fn approx_hybrid_search(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    metric: DistanceMetric,
    eps: f32,
) -> (Vec<u32>, Vec<f32>) {
    let sample_dims = approx_hybrid_sample_dims(dim, eps);
    let pool = approx_hybrid_pool_size(k, n, eps);
    if sample_dims >= dim || pool >= n {
        return contiguous_exact_search(query, candidates, dim, k, metric, 0, n);
    }

    if metric == DistanceMetric::InnerProduct {
        return approx_hybrid_ip_adaptive(query, candidates, dim, k, n, eps);
    }

    let shortlist = match metric {
        DistanceMetric::L2Squared => {
            coarse_shortlist_l2_adaptive(query, candidates, dim, pool, n, eps)
        }
        DistanceMetric::Cosine => {
            coarse_shortlist_cosine_adaptive(query, candidates, dim, pool, n, eps)
        }
        _ => return contiguous_exact_search(query, candidates, dim, k, metric, 0, n),
    };

    if shortlist.len() < k {
        return contiguous_exact_search(query, candidates, dim, k, metric, 0, n);
    }

    let subset_ids: Vec<u64> = shortlist.iter().map(|&idx| idx as u64).collect();
    match metric {
        DistanceMetric::InnerProduct => direct_access_topk::<false>(
            query,
            candidates,
            dim,
            k,
            n,
            &subset_ids,
            simd::inner_product_f32,
        ),
        DistanceMetric::L2Squared => direct_access_topk::<true>(
            query,
            candidates,
            dim,
            k,
            n,
            &subset_ids,
            simd::l2_squared_f32,
        ),
        DistanceMetric::Cosine => direct_access_topk::<true>(
            query,
            candidates,
            dim,
            k,
            n,
            &subset_ids,
            simd::cosine_distance_f32,
        ),
        _ => contiguous_exact_search(query, candidates, dim, k, metric, 0, n),
    }
}

#[allow(dead_code)]
fn early_exit_l2_topk(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
) -> (Vec<u32>, Vec<f32>) {
    let initial = loosen_min_threshold(initial_threshold(
        query,
        candidates,
        dim,
        k,
        n,
        DistanceMetric::L2Squared,
    ));
    let block_len = 32usize;
    let block_count = dim.div_ceil(block_len);
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_vecs = (n / n_threads).max(512);
    let chunk_floats = chunk_vecs * dim;

    let chunk_results: Vec<Vec<TopKEntry>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let n_in_chunk = cand_chunk.len() / dim;
            let base_idx = chunk_idx * chunk_vecs;
            let mut top = Vec::with_capacity(k);
            let mut threshold = initial;
            let mut filled = false;

            for local in 0..n_in_chunk {
                if local + PREFETCH_AHEAD_VECTORS < n_in_chunk {
                    unsafe {
                        prefetch_read_data(
                            cand_chunk
                                .as_ptr()
                                .add((local + PREFETCH_AHEAD_VECTORS) * dim)
                                .cast(),
                        );
                    }
                }

                let cand = unsafe { cand_chunk.get_unchecked(local * dim..(local + 1) * dim) };
                let mut dist = 0.0f32;
                let mut pruned = false;
                for block in 0..block_count {
                    let start = block * block_len;
                    let end = (start + block_len).min(dim);
                    dist += simd::l2_squared_f32(&query[start..end], &cand[start..end]);
                    if block + 1 < block_count && dist > threshold {
                        pruned = true;
                        break;
                    }
                }

                if !pruned && dist <= threshold {
                    topk_insert::<true>(
                        &mut top,
                        k,
                        TopKEntry {
                            dist,
                            idx: (base_idx + local) as u32,
                        },
                        &mut threshold,
                        &mut filled,
                    );
                }
            }

            if !filled && !top.is_empty() {
                top.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
            }
            top
        })
        .collect();

    let result = merge_topk_results::<true>(&chunk_results, k);
    if result.0.len() < k {
        contiguous_exact_search(query, candidates, dim, k, DistanceMetric::L2Squared, 0, n)
    } else {
        result
    }
}

fn coarse_shortlist_l2_adaptive(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    pool: usize,
    n: usize,
    eps: f32,
) -> Vec<u32> {
    let sample_dims = approx_hybrid_l2_sample_dims(dim, eps);
    if sample_dims >= dim {
        return (0..n as u32).collect();
    }
    let blocks = dominant_query_block(query, dim, sample_dims);
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_vecs = n.div_ceil(n_threads).max(4096);
    let chunk_floats = chunk_vecs * dim;
    let pool = pool.min(n);

    let chunk_results: Vec<Vec<TopKEntry>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let base_idx = chunk_idx * chunk_vecs;
            let n_in_chunk = (cand_chunk.len() / dim).min(n.saturating_sub(base_idx));
            let mut scored = vec![
                TopKEntry {
                    dist: f32::INFINITY,
                    idx: 0,
                };
                n_in_chunk
            ];

            if let [block] = blocks.as_slice() {
                let start = block.start;
                let end = start + block.len;
                let q = &query[start..end];
                let groups8 = n_in_chunk / 8;
                for group in 0..groups8 {
                    if group + 1 < groups8 {
                        unsafe {
                            prefetch_read_data(
                                cand_chunk
                                    .as_ptr()
                                    .add((group + 1) * 8 * dim + start)
                                    .cast(),
                            );
                        }
                    }
                    let row = group * 8;
                    let base = row * dim + start;
                    let v0 = unsafe { cand_chunk.get_unchecked(base..base + block.len) };
                    let v1 =
                        unsafe { cand_chunk.get_unchecked(base + dim..base + dim + block.len) };
                    let v2 = unsafe {
                        cand_chunk.get_unchecked(base + 2 * dim..base + 2 * dim + block.len)
                    };
                    let v3 = unsafe {
                        cand_chunk.get_unchecked(base + 3 * dim..base + 3 * dim + block.len)
                    };
                    let v4 = unsafe {
                        cand_chunk.get_unchecked(base + 4 * dim..base + 4 * dim + block.len)
                    };
                    let v5 = unsafe {
                        cand_chunk.get_unchecked(base + 5 * dim..base + 5 * dim + block.len)
                    };
                    let v6 = unsafe {
                        cand_chunk.get_unchecked(base + 6 * dim..base + 6 * dim + block.len)
                    };
                    let v7 = unsafe {
                        cand_chunk.get_unchecked(base + 7 * dim..base + 7 * dim + block.len)
                    };
                    let dists = simd::l2_squared_batch8_f32(q, v0, v1, v2, v3, v4, v5, v6, v7);
                    for (offset, &dist) in dists.iter().enumerate() {
                        scored[row + offset] = TopKEntry {
                            dist,
                            idx: (base_idx + row + offset) as u32,
                        };
                    }
                }

                for local in groups8 * 8..n_in_chunk {
                    let cand =
                        unsafe { cand_chunk.get_unchecked(local * dim + start..local * dim + end) };
                    let dist = simd::l2_squared_f32(q, cand);
                    scored[local] = TopKEntry {
                        dist,
                        idx: (base_idx + local) as u32,
                    };
                }
            } else {
                for local in 0..n_in_chunk {
                    if local + PREFETCH_AHEAD_VECTORS < n_in_chunk {
                        unsafe {
                            prefetch_read_data(
                                cand_chunk
                                    .as_ptr()
                                    .add((local + PREFETCH_AHEAD_VECTORS) * dim)
                                    .cast(),
                            );
                        }
                    }
                    let cand = unsafe { cand_chunk.get_unchecked(local * dim..(local + 1) * dim) };
                    let mut dist = 0.0f32;
                    for block in &blocks {
                        let end = block.start + block.len;
                        dist +=
                            simd::l2_squared_f32(&query[block.start..end], &cand[block.start..end]);
                    }
                    scored[local] = TopKEntry {
                        dist,
                        idx: (base_idx + local) as u32,
                    };
                }
            }

            let local_limit = pool.min(scored.len());
            truncate_entries_by(&mut scored, local_limit, cmp_entry_asc);
            scored
        })
        .collect();

    let mut merged: Vec<TopKEntry> = chunk_results.into_iter().flatten().collect();
    truncate_entries_by(&mut merged, pool, cmp_entry_asc);
    merged.into_iter().map(|entry| entry.idx).collect()
}

fn coarse_shortlist_cosine_adaptive(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    pool: usize,
    n: usize,
    eps: f32,
) -> Vec<u32> {
    let sample_dims = approx_hybrid_cosine_sample_dims(dim, eps);
    if sample_dims >= dim {
        return (0..n as u32).collect();
    }
    let blocks = dominant_query_block(query, dim, sample_dims);
    if block_norm2(query, &blocks) <= 1e-30 {
        return Vec::new();
    }
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_vecs = n.div_ceil(n_threads).max(4096);
    let chunk_floats = chunk_vecs * dim;
    let pool = pool.min(n);

    let chunk_results: Vec<Vec<TopKEntry>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let base_idx = chunk_idx * chunk_vecs;
            let n_in_chunk = (cand_chunk.len() / dim).min(n.saturating_sub(base_idx));
            let mut scored = vec![
                TopKEntry {
                    dist: f32::NEG_INFINITY,
                    idx: 0,
                };
                n_in_chunk
            ];

            if let [block] = blocks.as_slice() {
                let start = block.start;
                let end = start + block.len;
                let q = &query[start..end];
                let groups8 = n_in_chunk / 8;
                for group in 0..groups8 {
                    if group + 1 < groups8 {
                        unsafe {
                            prefetch_read_data(
                                cand_chunk
                                    .as_ptr()
                                    .add((group + 1) * 8 * dim + start)
                                    .cast(),
                            );
                        }
                    }
                    let row = group * 8;
                    let base = row * dim + start;
                    let v0 = unsafe { cand_chunk.get_unchecked(base..base + block.len) };
                    let v1 =
                        unsafe { cand_chunk.get_unchecked(base + dim..base + dim + block.len) };
                    let v2 = unsafe {
                        cand_chunk.get_unchecked(base + 2 * dim..base + 2 * dim + block.len)
                    };
                    let v3 = unsafe {
                        cand_chunk.get_unchecked(base + 3 * dim..base + 3 * dim + block.len)
                    };
                    let v4 = unsafe {
                        cand_chunk.get_unchecked(base + 4 * dim..base + 4 * dim + block.len)
                    };
                    let v5 = unsafe {
                        cand_chunk.get_unchecked(base + 5 * dim..base + 5 * dim + block.len)
                    };
                    let v6 = unsafe {
                        cand_chunk.get_unchecked(base + 6 * dim..base + 6 * dim + block.len)
                    };
                    let v7 = unsafe {
                        cand_chunk.get_unchecked(base + 7 * dim..base + 7 * dim + block.len)
                    };
                    let dots = simd::inner_product_batch8_f32(q, v0, v1, v2, v3, v4, v5, v6, v7);
                    for (offset, &score) in dots.iter().enumerate() {
                        scored[row + offset] = TopKEntry {
                            dist: score,
                            idx: (base_idx + row + offset) as u32,
                        };
                    }
                }

                for local in groups8 * 8..n_in_chunk {
                    let cand =
                        unsafe { cand_chunk.get_unchecked(local * dim + start..local * dim + end) };
                    let score = simd::inner_product_f32(q, cand);
                    scored[local] = TopKEntry {
                        dist: score,
                        idx: (base_idx + local) as u32,
                    };
                }
            } else {
                for local in 0..n_in_chunk {
                    if local + PREFETCH_AHEAD_VECTORS < n_in_chunk {
                        unsafe {
                            prefetch_read_data(
                                cand_chunk
                                    .as_ptr()
                                    .add((local + PREFETCH_AHEAD_VECTORS) * dim)
                                    .cast(),
                            );
                        }
                    }
                    let cand = unsafe { cand_chunk.get_unchecked(local * dim..(local + 1) * dim) };
                    let score = block_inner_product(query, cand, &blocks);
                    scored[local] = TopKEntry {
                        dist: score,
                        idx: (base_idx + local) as u32,
                    };
                }
            }

            let local_limit = pool.min(scored.len());
            truncate_entries_by(&mut scored, local_limit, cmp_entry_desc);
            scored
        })
        .collect();

    let mut merged: Vec<TopKEntry> = chunk_results.into_iter().flatten().collect();
    truncate_entries_by(&mut merged, pool, cmp_entry_desc);
    merged.into_iter().map(|entry| entry.idx).collect()
}

fn approx_ip_order_search(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    eps: f32,
    order: &ApproxIpOrder,
) -> (Vec<u32>, Vec<f32>) {
    let pool = approx_ip_order_pool_size(k, n, eps);
    let (source, source_vectors) = if query.iter().all(|&value| value >= 0.0) {
        (&order.sum_desc, &order.sum_desc_vectors)
    } else if query.iter().all(|&value| value <= 0.0) {
        (&order.sum_asc, &order.sum_asc_vectors)
    } else {
        return approx_hybrid_ip_adaptive(query, candidates, dim, k, n, eps);
    };

    let take = pool.min(source.len());
    if cached_vectors_cover(take, source_vectors, dim) {
        return cached_ip_order_topk(query, &source[..take], source_vectors, dim, k, eps);
    }

    let mut subset_ids: Vec<u64> = source
        .iter()
        .take(take)
        .map(|entry| entry.idx as u64)
        .collect();
    subset_ids.sort_unstable();
    let (ids, dists) = direct_access_topk::<false>(
        query,
        candidates,
        dim,
        k,
        n,
        &subset_ids,
        simd::inner_product_f32,
    );
    let dists = dists
        .into_iter()
        .map(|dist| super::approx_search::round_to_eps(dist, eps))
        .collect();
    (ids, dists)
}

fn cached_ip_order_topk(
    query: &[f32],
    entries: &[TopKEntry],
    vectors: &[f32],
    dim: usize,
    k: usize,
    eps: f32,
) -> (Vec<u32>, Vec<f32>) {
    let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
    let mut threshold = f32::NEG_INFINITY;
    let mut filled = false;

    for (i, entry) in entries.iter().enumerate() {
        let start = i * dim;
        let dist = simd::inner_product_f32(query, &vectors[start..start + dim]);
        if !filled || passes_threshold::<false>(dist, threshold) {
            topk_insert::<false>(
                &mut top,
                k,
                TopKEntry {
                    dist,
                    idx: entry.idx,
                },
                &mut threshold,
                &mut filled,
            );
        }
    }

    if !filled && !top.is_empty() {
        top.sort_unstable_by(|a, b| b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal));
    }

    let ids = top.iter().map(|entry| entry.idx).collect();
    let dists = top
        .iter()
        .map(|entry| super::approx_search::round_to_eps(entry.dist, eps))
        .collect();
    (ids, dists)
}

#[inline]
fn approx_ip_order_pool_size(k: usize, n: usize, eps: f32) -> usize {
    let per_k = if eps <= 1e-6 {
        1000usize
    } else if eps <= 1e-5 {
        750
    } else if eps <= 1e-4 {
        500
    } else if eps <= 1e-3 {
        250
    } else if eps <= 1e-2 {
        120
    } else {
        60
    };
    let n_floor = if eps <= 1e-6 {
        n / 100
    } else if eps <= 1e-5 {
        n / 133
    } else if eps <= 1e-4 {
        n / 200
    } else if eps <= 1e-3 {
        n / 400
    } else if eps <= 1e-2 {
        n / 800
    } else {
        n / 1600
    };
    k.saturating_mul(per_k)
        .max(n_floor)
        .max(k + 128)
        .min(n)
        .min(20_000.max(k.min(n)))
}

fn approx_hybrid_ip_adaptive(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    eps: f32,
) -> (Vec<u32>, Vec<f32>) {
    let sample_dims = approx_hybrid_ip_sample_dims(dim, eps);
    let pool = approx_hybrid_ip_pool_size(k, n, eps);
    if sample_dims >= dim || pool >= n {
        return contiguous_exact_search(
            query,
            candidates,
            dim,
            k,
            DistanceMetric::InnerProduct,
            0,
            n,
        );
    }

    let blocks = sampled_dim_blocks(dim, sample_dims);
    let mut scores = vec![
        TopKEntry {
            dist: f32::NEG_INFINITY,
            idx: 0,
        };
        n
    ];
    let n_threads = rayon::current_num_threads();
    let chunk_vecs = (n / n_threads).max(4096);

    scores
        .par_chunks_mut(chunk_vecs)
        .enumerate()
        .for_each(|(chunk_idx, score_chunk)| {
            let base_idx = chunk_idx * chunk_vecs;
            let n_in_chunk = score_chunk.len();
            let start_f = base_idx * dim;
            let end_f = start_f + n_in_chunk * dim;
            let cand_chunk = unsafe { candidates.get_unchecked(start_f..end_f) };

            let blocks8 = n_in_chunk / 8;
            for row_block in 0..blocks8 {
                if row_block + 1 < blocks8 {
                    unsafe {
                        prefetch_read_data(
                            cand_chunk.as_ptr().add((row_block + 1) * 8 * dim).cast(),
                        );
                    }
                }

                let row_base = row_block * 8 * dim;
                let mut acc = [0.0f32; 8];
                for block in &blocks {
                    let start = block.start;
                    let end = block.start + block.len;
                    let q = &query[start..end];
                    let v0 = unsafe { cand_chunk.get_unchecked(row_base + start..row_base + end) };
                    let v1 = unsafe {
                        cand_chunk.get_unchecked(row_base + dim + start..row_base + dim + end)
                    };
                    let v2 = unsafe {
                        cand_chunk
                            .get_unchecked(row_base + 2 * dim + start..row_base + 2 * dim + end)
                    };
                    let v3 = unsafe {
                        cand_chunk
                            .get_unchecked(row_base + 3 * dim + start..row_base + 3 * dim + end)
                    };
                    let v4 = unsafe {
                        cand_chunk
                            .get_unchecked(row_base + 4 * dim + start..row_base + 4 * dim + end)
                    };
                    let v5 = unsafe {
                        cand_chunk
                            .get_unchecked(row_base + 5 * dim + start..row_base + 5 * dim + end)
                    };
                    let v6 = unsafe {
                        cand_chunk
                            .get_unchecked(row_base + 6 * dim + start..row_base + 6 * dim + end)
                    };
                    let v7 = unsafe {
                        cand_chunk
                            .get_unchecked(row_base + 7 * dim + start..row_base + 7 * dim + end)
                    };
                    let partial = simd::inner_product_batch8_f32(q, v0, v1, v2, v3, v4, v5, v6, v7);
                    for lane in 0..8 {
                        acc[lane] += partial[lane];
                    }
                }

                for lane in 0..8 {
                    let local = row_block * 8 + lane;
                    score_chunk[local] = TopKEntry {
                        dist: acc[lane],
                        idx: (base_idx + local) as u32,
                    };
                }
            }

            for local in blocks8 * 8..n_in_chunk {
                let cand = unsafe { cand_chunk.get_unchecked(local * dim..(local + 1) * dim) };
                let mut score = 0.0f32;
                for block in &blocks {
                    let end = block.start + block.len;
                    score +=
                        simd::inner_product_f32(&query[block.start..end], &cand[block.start..end]);
                }
                score_chunk[local] = TopKEntry {
                    dist: score,
                    idx: (base_idx + local) as u32,
                };
            }
        });

    let pool = pool.min(scores.len());
    if scores.len() > pool {
        scores.select_nth_unstable_by(pool, |a, b| {
            b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal)
        });
        scores.truncate(pool);
    }

    let subset_ids: Vec<u64> = scores.iter().map(|entry| entry.idx as u64).collect();
    direct_access_topk::<false>(
        query,
        candidates,
        dim,
        k,
        n,
        &subset_ids,
        simd::inner_product_f32,
    )
}

#[inline]
fn approx_hybrid_ip_sample_dims(dim: usize, eps: f32) -> usize {
    if dim <= 32 {
        return dim;
    }
    let ratio = if eps <= 1e-6 {
        0.75
    } else if eps <= 1e-5 {
        0.625
    } else if eps <= 1e-4 {
        0.5
    } else if eps <= 1e-3 {
        0.375
    } else {
        0.25
    };
    ((dim as f32 * ratio).round() as usize).max(32).min(dim)
}

#[inline]
fn approx_hybrid_ip_pool_size(k: usize, n: usize, eps: f32) -> usize {
    let per_k = if eps <= 1e-6 {
        8000usize
    } else if eps <= 1e-5 {
        6000
    } else if eps <= 1e-4 {
        5000
    } else if eps <= 1e-3 {
        3000
    } else if eps <= 1e-2 {
        2000
    } else {
        1000
    };
    let n_floor = if eps <= 1e-6 {
        n / 20
    } else if eps <= 1e-5 {
        n / 20
    } else if eps <= 1e-4 {
        n / 20
    } else if eps <= 1e-3 {
        n / 40
    } else if eps <= 1e-2 {
        n / 80
    } else {
        n / 160
    };
    k.saturating_mul(per_k)
        .max(n_floor)
        .max(k + 128)
        .min(n)
        .min(100_000)
}

#[inline]
fn approx_hybrid_l2_sample_dims(dim: usize, eps: f32) -> usize {
    if dim <= 32 {
        return dim;
    }
    let ratio = if eps <= 1e-6 {
        0.75
    } else if eps <= 1e-5 {
        0.625
    } else if eps <= 1e-4 {
        0.5
    } else if eps <= 1e-3 {
        0.375
    } else {
        0.25
    };
    ((dim as f32 * ratio).round() as usize).max(32).min(dim)
}

#[inline]
fn approx_hybrid_cosine_sample_dims(dim: usize, eps: f32) -> usize {
    if dim <= 32 {
        return dim;
    }
    let ratio = if eps <= 1e-6 {
        0.75
    } else if eps <= 1e-5 {
        0.625
    } else if eps <= 1e-4 {
        0.5
    } else if eps <= 1e-3 {
        0.375
    } else {
        0.25
    };
    ((dim as f32 * ratio).round() as usize).max(32).min(dim)
}

#[inline]
fn approx_hybrid_sample_dims(dim: usize, eps: f32) -> usize {
    if dim <= 32 {
        return dim;
    }
    let ratio = if eps <= 1e-6 {
        0.875
    } else if eps <= 1e-5 {
        0.75
    } else if eps <= 1e-4 {
        0.625
    } else if eps <= 1e-3 {
        0.5
    } else if eps <= 1e-2 {
        0.375
    } else {
        0.25
    };
    ((dim as f32 * ratio).round() as usize).max(32).min(dim)
}

#[inline]
fn approx_hybrid_pool_size(k: usize, n: usize, eps: f32) -> usize {
    let per_k = if eps <= 1e-6 {
        4000usize
    } else if eps <= 1e-5 {
        3000
    } else if eps <= 1e-4 {
        2000
    } else if eps <= 1e-3 {
        1000
    } else if eps <= 1e-2 {
        500
    } else {
        250
    };
    let n_floor = if eps <= 1e-6 {
        n / 50
    } else if eps <= 1e-5 {
        n / 50
    } else if eps <= 1e-4 {
        n / 50
    } else if eps <= 1e-3 {
        n / 100
    } else if eps <= 1e-2 {
        n / 200
    } else {
        n / 400
    };
    k.saturating_mul(per_k)
        .max(n_floor)
        .max(k + 128)
        .min(n)
        .min(100_000)
}

#[inline]
fn block_norm2(values: &[f32], blocks: &[DimBlock]) -> f32 {
    let mut sum = 0.0f32;
    for block in blocks {
        let end = block.start + block.len;
        sum += simd::inner_product_f32(&values[block.start..end], &values[block.start..end]);
    }
    sum
}

/// Bounded exact top-k for `approx=True`.
///
/// This scans all rows but can skip later f32 blocks once a mathematically safe
/// bound proves the row cannot beat the current top-k threshold.
#[allow(dead_code)]
pub(super) fn approx_bounded_search(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    metric: DistanceMetric,
    bounds: &super::approx_search::ApproxBounds,
) -> (Vec<u32>, Vec<f32>) {
    match metric {
        DistanceMetric::InnerProduct => bounded_ip_topk(query, candidates, dim, k, n, bounds),
        DistanceMetric::L2Squared => bounded_l2_topk(query, candidates, dim, k, n, bounds),
        DistanceMetric::Cosine => bounded_cosine_topk(query, candidates, dim, k, n, bounds),
        _ => contiguous_exact_search(query, candidates, dim, k, metric, 0, n),
    }
}

fn initial_threshold(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    metric: DistanceMetric,
) -> f32 {
    let init_rows = n
        .min(super::approx_search::APPROX_INIT_ROWS.max(k.saturating_mul(64)))
        .max(k);
    let (_, dists) = contiguous_exact_search(query, candidates, dim, k, metric, 0, init_rows);
    dists.last().copied().unwrap_or_else(|| {
        if metric.is_ascending() {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        }
    })
}

#[inline]
fn loosen_min_threshold(threshold: f32) -> f32 {
    threshold + threshold.abs().mul_add(1e-6, 1e-5)
}

#[inline]
fn loosen_max_threshold(threshold: f32) -> f32 {
    threshold - threshold.abs().mul_add(1e-6, 1e-5)
}

fn query_suffix_norms(query: &[f32], bounds: &super::approx_search::ApproxBounds) -> Vec<f32> {
    let mut tails = vec![0.0f32; bounds.block_count];
    let mut suffix = 0.0f32;
    for block in (0..bounds.block_count).rev() {
        tails[block] = suffix;
        let start = block * bounds.block_len;
        let end = (start + bounds.block_len).min(query.len());
        let mut norm = 0.0f32;
        for &value in &query[start..end] {
            norm += value * value;
        }
        suffix += norm;
    }
    tails
}

fn bounded_ip_topk(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    bounds: &super::approx_search::ApproxBounds,
) -> (Vec<u32>, Vec<f32>) {
    let initial = loosen_max_threshold(initial_threshold(
        query,
        candidates,
        dim,
        k,
        n,
        DistanceMetric::InnerProduct,
    ));
    let q_tails = query_suffix_norms(query, bounds);
    let block_len = bounds.block_len;
    let block_count = bounds.block_count;
    let n_threads = rayon::current_num_threads();
    let chunk_vecs = (n / n_threads).max(512);
    let chunk_floats = chunk_vecs * dim;

    let chunk_results: Vec<Vec<TopKEntry>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let n_in_chunk = cand_chunk.len() / dim;
            let base_idx = chunk_idx * chunk_vecs;
            let mut top = Vec::with_capacity(k);
            let mut threshold = initial;
            let mut filled = false;

            for local in 0..n_in_chunk {
                let global = base_idx + local;
                let cand = unsafe { cand_chunk.get_unchecked(local * dim..(local + 1) * dim) };
                let tails =
                    &bounds.suffix_norm2[global * block_count..global * block_count + block_count];

                let mut dot = 0.0f32;
                let mut pruned = false;
                for block in 0..block_count {
                    let start = block * block_len;
                    let end = (start + block_len).min(dim);
                    dot += simd::inner_product_f32(&query[start..end], &cand[start..end]);

                    if block + 1 < block_count && dot < threshold {
                        let gap = threshold - dot;
                        if gap * gap > q_tails[block] * tails[block] {
                            pruned = true;
                            break;
                        }
                    }
                }

                if !pruned && dot >= threshold {
                    topk_insert::<false>(
                        &mut top,
                        k,
                        TopKEntry {
                            dist: dot,
                            idx: global as u32,
                        },
                        &mut threshold,
                        &mut filled,
                    );
                }
            }

            if !filled && !top.is_empty() {
                top.sort_unstable_by(|a, b| b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal));
            }
            top
        })
        .collect();

    merge_topk_results::<false>(&chunk_results, k)
}

fn bounded_l2_topk(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    bounds: &super::approx_search::ApproxBounds,
) -> (Vec<u32>, Vec<f32>) {
    let initial = loosen_min_threshold(initial_threshold(
        query,
        candidates,
        dim,
        k,
        n,
        DistanceMetric::L2Squared,
    ));
    let block_len = bounds.block_len;
    let block_count = bounds.block_count;
    let n_threads = rayon::current_num_threads();
    let chunk_vecs = (n / n_threads).max(512);
    let chunk_floats = chunk_vecs * dim;

    let chunk_results: Vec<Vec<TopKEntry>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let n_in_chunk = cand_chunk.len() / dim;
            let base_idx = chunk_idx * chunk_vecs;
            let mut top = Vec::with_capacity(k);
            let mut threshold = initial;
            let mut filled = false;

            for local in 0..n_in_chunk {
                let global = base_idx + local;
                let cand = unsafe { cand_chunk.get_unchecked(local * dim..(local + 1) * dim) };

                let mut dist = 0.0f32;
                let mut pruned = false;
                for block in 0..block_count {
                    let start = block * block_len;
                    let end = (start + block_len).min(dim);
                    dist += simd::l2_squared_f32(&query[start..end], &cand[start..end]);
                    if block + 1 < block_count && dist > threshold {
                        pruned = true;
                        break;
                    }
                }

                if !pruned && dist <= threshold {
                    topk_insert::<true>(
                        &mut top,
                        k,
                        TopKEntry {
                            dist,
                            idx: global as u32,
                        },
                        &mut threshold,
                        &mut filled,
                    );
                }
            }

            if !filled && !top.is_empty() {
                top.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
            }
            top
        })
        .collect();

    let result = merge_topk_results::<true>(&chunk_results, k);
    if result.0.len() < k {
        contiguous_exact_search(query, candidates, dim, k, DistanceMetric::L2Squared, 0, n)
    } else {
        result
    }
}

fn bounded_cosine_topk(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    bounds: &super::approx_search::ApproxBounds,
) -> (Vec<u32>, Vec<f32>) {
    let q_norm2 = query.iter().map(|v| v * v).sum::<f32>();
    if q_norm2 <= 1e-30 {
        return contiguous_exact_search(query, candidates, dim, k, DistanceMetric::Cosine, 0, n);
    }

    let initial_dist = initial_threshold(query, candidates, dim, k, n, DistanceMetric::Cosine);
    let initial_sim = loosen_max_threshold(1.0 - initial_dist);
    let q_tails = query_suffix_norms(query, bounds);
    let q_norm = q_norm2.sqrt();
    let block_len = bounds.block_len;
    let block_count = bounds.block_count;
    let n_threads = rayon::current_num_threads();
    let chunk_vecs = (n / n_threads).max(512);
    let chunk_floats = chunk_vecs * dim;

    let chunk_results: Vec<Vec<TopKEntry>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let n_in_chunk = cand_chunk.len() / dim;
            let base_idx = chunk_idx * chunk_vecs;
            let mut top = Vec::with_capacity(k);
            let mut threshold = initial_sim;
            let mut filled = false;

            for local in 0..n_in_chunk {
                let global = base_idx + local;
                let cand = unsafe { cand_chunk.get_unchecked(local * dim..(local + 1) * dim) };
                let x_norm2 = bounds.total_norm2[global];
                if x_norm2 <= 1e-30 {
                    continue;
                }
                let denom = q_norm * x_norm2.sqrt();
                let tails =
                    &bounds.suffix_norm2[global * block_count..global * block_count + block_count];

                let mut dot = 0.0f32;
                let mut pruned = false;
                for block in 0..block_count {
                    let start = block * block_len;
                    let end = (start + block_len).min(dim);
                    dot += simd::inner_product_f32(&query[start..end], &cand[start..end]);

                    if block + 1 < block_count {
                        let needed = threshold * denom - dot;
                        if needed > 0.0 && needed * needed > q_tails[block] * tails[block] {
                            pruned = true;
                            break;
                        }
                    }
                }

                if !pruned {
                    let sim = dot / denom;
                    if sim >= threshold {
                        topk_insert::<false>(
                            &mut top,
                            k,
                            TopKEntry {
                                dist: sim,
                                idx: global as u32,
                            },
                            &mut threshold,
                            &mut filled,
                        );
                    }
                }
            }

            if !filled && !top.is_empty() {
                top.sort_unstable_by(|a, b| b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal));
            }
            top
        })
        .collect();

    let (ids, sims) = merge_topk_results::<false>(&chunk_results, k);
    if ids.len() < k {
        return contiguous_exact_search(query, candidates, dim, k, DistanceMetric::Cosine, 0, n);
    }
    let dists = sims.into_iter().map(|sim| 1.0 - sim).collect();
    (ids, dists)
}

/// Block-partial coarse top-k for approximate search.
///
/// The coarse pass reads original f32 dimensions in several contiguous blocks
/// across the vector instead of taking only a prefix. This follows the
/// partial-distance idea used by ADSampling while keeping memory access much
/// friendlier than fully scattered dimension sampling.
#[allow(dead_code)]
fn approx_coarse_search(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    d_coarse: usize,
    k: usize,
    n: usize,
    metric: DistanceMetric,
    skip_rows: Option<(usize, usize)>,
) -> (Vec<u32>, Vec<f32>) {
    let d = d_coarse.min(dim);
    if d == 0 || dim == 0 || n == 0 || k == 0 {
        return (Vec::new(), Vec::new());
    }
    let blocks = sampled_dim_blocks(dim, d);
    let sampled = blocks.iter().map(|b| b.len).sum::<usize>().max(1);
    let scale = dim as f32 / sampled as f32;

    match metric {
        DistanceMetric::InnerProduct => {
            sampled_coarse_topk::<false>(query, candidates, dim, k, n, skip_rows, |q, c| {
                block_inner_product(q, c, &blocks) * scale
            })
        }
        DistanceMetric::L2Squared => {
            sampled_coarse_topk::<true>(query, candidates, dim, k, n, skip_rows, |q, c| {
                block_l2_squared(q, c, &blocks) * scale
            })
        }
        DistanceMetric::Cosine => {
            sampled_coarse_topk::<true>(query, candidates, dim, k, n, skip_rows, |q, c| {
                block_cosine_distance(q, c, &blocks)
            })
        }
        DistanceMetric::Hamming => {
            sampled_coarse_topk::<true>(query, candidates, dim, k, n, skip_rows, |q, c| {
                block_hamming(q, c, &blocks) * scale
            })
        }
        DistanceMetric::Jaccard => {
            sampled_coarse_topk::<true>(query, candidates, dim, k, n, skip_rows, |q, c| {
                block_jaccard(q, c, &blocks)
            })
        }
        _ => sampled_coarse_topk::<true>(query, candidates, dim, k, n, skip_rows, |q, c| {
            distance::compute_distance_f32(q, c, metric)
        }),
    }
}

#[derive(Clone, Copy)]
struct DimBlock {
    start: usize,
    len: usize,
}

fn sampled_dim_blocks(dim: usize, sample: usize) -> Vec<DimBlock> {
    let sample = sample.min(dim);
    if sample == 0 {
        return Vec::new();
    }
    if sample == dim {
        return vec![DimBlock { start: 0, len: dim }];
    }

    let block_count = sample.min(3).max(1);
    if block_count <= 1 {
        return vec![DimBlock {
            start: (dim - sample) / 2,
            len: sample,
        }];
    }

    let mut blocks = Vec::with_capacity(block_count);
    let mut remaining = sample;
    for i in 0..block_count {
        let slots_left = block_count - i;
        let len = remaining.div_ceil(slots_left);
        remaining -= len;
        let max_start = dim.saturating_sub(len);
        let start = (i * max_start + (block_count - 1) / 2) / (block_count - 1);
        blocks.push(DimBlock { start, len });
    }
    blocks
}

fn dominant_query_block(query: &[f32], dim: usize, sample: usize) -> Vec<DimBlock> {
    let sample = sample.min(dim);
    if sample == 0 {
        return Vec::new();
    }
    if sample == dim {
        return vec![DimBlock { start: 0, len: dim }];
    }

    let max_start = dim - sample;
    let candidates = [0usize, max_start / 3, (max_start * 2) / 3, max_start];
    let mut best_start = candidates[0];
    let mut best_norm = f32::NEG_INFINITY;
    for &start in &candidates {
        let end = start + sample;
        let norm = simd::inner_product_f32(&query[start..end], &query[start..end]);
        if norm > best_norm {
            best_norm = norm;
            best_start = start;
        }
    }
    vec![DimBlock {
        start: best_start,
        len: sample,
    }]
}

#[cfg(test)]
fn sampled_dim_indices(dim: usize, sample: usize) -> Vec<usize> {
    let sample = sample.min(dim);
    if sample == 0 {
        return Vec::new();
    }
    if sample == dim {
        return (0..dim).collect();
    }

    let mut dims = Vec::with_capacity(sample);
    let mut seen = vec![false; dim];

    if sample == 1 {
        let idx = dim / 2;
        dims.push(idx);
        return dims;
    }

    // Evenly cover [0, dim - 1], including both edges. This keeps the scan
    // deterministic and protects against prefix-only blind spots.
    for i in 0..sample {
        let idx = (i * (dim - 1) + (sample - 1) / 2) / (sample - 1);
        if !seen[idx] {
            seen[idx] = true;
            dims.push(idx);
        }
    }

    // Fill any duplicate gaps with a coprime stride permutation.
    let stride = coprime_stride(dim);
    let mut idx = dim / 2;
    while dims.len() < sample {
        if !seen[idx] {
            seen[idx] = true;
            dims.push(idx);
        }
        idx = (idx + stride) % dim;
    }

    dims.sort_unstable();
    dims
}

#[cfg(test)]
fn coprime_stride(dim: usize) -> usize {
    if dim <= 2 {
        return 1;
    }
    let mut stride = dim - 1;
    while stride > 1 {
        if gcd(stride, dim) == 1 {
            return stride;
        }
        stride -= 1;
    }
    1
}

#[cfg(test)]
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

#[inline]
fn block_inner_product(query: &[f32], cand: &[f32], blocks: &[DimBlock]) -> f32 {
    let mut sum = 0.0f32;
    for block in blocks {
        let end = block.start + block.len;
        for j in block.start..end {
            sum += query[j] * cand[j];
        }
    }
    sum
}

#[inline]
fn block_l2_squared(query: &[f32], cand: &[f32], blocks: &[DimBlock]) -> f32 {
    let mut sum = 0.0f32;
    for block in blocks {
        let end = block.start + block.len;
        for j in block.start..end {
            let diff = query[j] - cand[j];
            sum += diff * diff;
        }
    }
    sum
}

#[inline]
fn block_cosine_distance(query: &[f32], cand: &[f32], blocks: &[DimBlock]) -> f32 {
    let mut dot = 0.0f32;
    let mut q_norm = 0.0f32;
    let mut c_norm = 0.0f32;
    for block in blocks {
        let end = block.start + block.len;
        for j in block.start..end {
            let q = query[j];
            let c = cand[j];
            dot += q * c;
            q_norm += q * q;
            c_norm += c * c;
        }
    }
    let denom = (q_norm * c_norm).sqrt();
    if denom < 1e-30 {
        1.0
    } else {
        1.0 - dot / denom
    }
}

#[inline]
fn block_hamming(query: &[f32], cand: &[f32], blocks: &[DimBlock]) -> f32 {
    let mut count = 0u32;
    for block in blocks {
        let end = block.start + block.len;
        for j in block.start..end {
            if (query[j] > 0.5) != (cand[j] > 0.5) {
                count += 1;
            }
        }
    }
    count as f32
}

#[inline]
fn block_jaccard(query: &[f32], cand: &[f32], blocks: &[DimBlock]) -> f32 {
    let mut intersection = 0u32;
    let mut union = 0u32;
    for block in blocks {
        let end = block.start + block.len;
        for j in block.start..end {
            let q = query[j] > 0.5;
            let c = cand[j] > 0.5;
            if q || c {
                union += 1;
                if q && c {
                    intersection += 1;
                }
            }
        }
    }
    if union == 0 {
        0.0
    } else {
        1.0 - intersection as f32 / union as f32
    }
}

fn sampled_coarse_topk<const ASC: bool>(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    skip_rows: Option<(usize, usize)>,
    score_fn: impl Fn(&[f32], &[f32]) -> f32 + Send + Sync,
) -> (Vec<u32>, Vec<f32>) {
    if k == 0 {
        return (Vec::new(), Vec::new());
    }

    let n_threads = rayon::current_num_threads();
    let chunk_vecs = (n / n_threads).max(512);
    let chunk_floats = chunk_vecs * dim;
    let k = k.min(n);

    let chunk_results: Vec<Vec<TopKEntry>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let n_in_chunk = cand_chunk.len() / dim;
            let base_idx = chunk_idx * chunk_vecs;

            let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
            let mut threshold: f32 = if ASC {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
            let mut filled = false;

            let scan_row =
                |i: usize, top: &mut Vec<TopKEntry>, threshold: &mut f32, filled: &mut bool| {
                    let global_idx = base_idx + i;
                    if i + PREFETCH_AHEAD_VECTORS < n_in_chunk {
                        unsafe {
                            prefetch_read_data(
                                cand_chunk
                                    .as_ptr()
                                    .add((i + PREFETCH_AHEAD_VECTORS) * dim)
                                    .cast(),
                            );
                        }
                    }
                    let cand = unsafe { cand_chunk.get_unchecked(i * dim..(i + 1) * dim) };
                    let score = normalize_coarse_score::<ASC>(score_fn(query, cand));
                    if !*filled || passes_threshold::<ASC>(score, *threshold) {
                        topk_insert::<ASC>(
                            top,
                            k,
                            TopKEntry {
                                dist: score,
                                idx: global_idx as u32,
                            },
                            threshold,
                            filled,
                        );
                    }
                };

            if let Some((stride, offset)) = skip_rows {
                for residue in 0..stride {
                    if residue == offset {
                        continue;
                    }
                    let mut i = first_local_row(base_idx, n_in_chunk, stride, residue);
                    while i < n_in_chunk {
                        scan_row(i, &mut top, &mut threshold, &mut filled);
                        i += stride;
                    }
                }
            } else {
                for i in 0..n_in_chunk {
                    scan_row(i, &mut top, &mut threshold, &mut filled);
                }
            }

            if !filled && !top.is_empty() {
                if ASC {
                    top.sort_unstable_by(|a, b| {
                        a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal)
                    });
                } else {
                    top.sort_unstable_by(|a, b| {
                        b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal)
                    });
                }
            }
            top
        })
        .collect();

    merge_topk_results::<ASC>(&chunk_results, k)
}

#[inline]
fn normalize_coarse_score<const ASC: bool>(score: f32) -> f32 {
    if score.is_finite() {
        score
    } else if ASC {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    }
}

/// IP-specific fused parallel top-k: batch-8 dot products + software prefetch.
#[inline(never)]
fn fused_topk_ip_parallel(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
) -> (Vec<u32>, Vec<f32>) {
    if n < 4096 {
        return fused_topk_seq::<false>(query, candidates, dim, k, 0, &simd::inner_product_f32);
    }

    let n_threads = rayon::current_num_threads();
    let chunk_vecs = (n / n_threads).max(512);
    let chunk_floats = chunk_vecs * dim;

    let chunk_results: Vec<Vec<TopKEntry>> = candidates
        .par_chunks(chunk_floats)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            ip_scan_chunk_topk(query, cand_chunk, dim, k, chunk_idx * chunk_vecs)
        })
        .collect();

    merge_topk_results::<false>(&chunk_results, k)
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
            let mut threshold: f32 = if ASC {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
            let mut filled = false;

            let mut ptr = cand_chunk.as_ptr();
            let pairs = n_in_chunk / 2;
            let dim2 = dim * 2;
            for i in 0..pairs {
                if i + 4 < pairs {
                    unsafe {
                        prefetch_read_data(ptr.add((i + 4) * dim2).cast());
                    }
                }
                let cand0 = unsafe { std::slice::from_raw_parts(ptr, dim) };
                let cand1 = unsafe { std::slice::from_raw_parts(ptr.wrapping_add(dim), dim) };
                let dist0 = dist_fn(query, cand0);
                let dist1 = dist_fn(query, cand1);

                let idx0 = (base_idx + i * 2) as u32;
                if !filled || passes_threshold::<ASC>(dist0, threshold) {
                    topk_insert::<ASC>(
                        &mut top,
                        k,
                        TopKEntry {
                            dist: dist0,
                            idx: idx0,
                        },
                        &mut threshold,
                        &mut filled,
                    );
                }
                if !filled || passes_threshold::<ASC>(dist1, threshold) {
                    topk_insert::<ASC>(
                        &mut top,
                        k,
                        TopKEntry {
                            dist: dist1,
                            idx: idx0 + 1,
                        },
                        &mut threshold,
                        &mut filled,
                    );
                }
                ptr = unsafe { ptr.add(dim2) };
            }
            // Handle odd remainder
            if n_in_chunk % 2 == 1 {
                let cand = unsafe { std::slice::from_raw_parts(ptr, dim) };
                let dist = dist_fn(query, cand);
                if !filled || passes_threshold::<ASC>(dist, threshold) {
                    topk_insert::<ASC>(
                        &mut top,
                        k,
                        TopKEntry {
                            dist,
                            idx: (base_idx + n_in_chunk - 1) as u32,
                        },
                        &mut threshold,
                        &mut filled,
                    );
                }
            }

            if !filled && !top.is_empty() {
                if ASC {
                    top.sort_unstable_by(|a, b| {
                        a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal)
                    });
                } else {
                    top.sort_unstable_by(|a, b| {
                        b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal)
                    });
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
    dist_fn: &impl Fn(&[f32], &[f32]) -> f32,
) -> (Vec<u32>, Vec<f32>) {
    let n = candidates.len() / dim;
    let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
    let mut threshold: f32 = if ASC {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
    let mut filled = false;

    for i in 0..n {
        if i + PREFETCH_AHEAD_VECTORS < n {
            unsafe {
                prefetch_read_data(
                    candidates
                        .as_ptr()
                        .add((i + PREFETCH_AHEAD_VECTORS) * dim)
                        .cast(),
                );
            }
        }
        let cand = unsafe {
            let start = i * dim;
            candidates.get_unchecked(start..start + dim)
        };
        let dist = dist_fn(query, cand);

        if !filled || passes_threshold::<ASC>(dist, threshold) {
            topk_insert::<ASC>(
                &mut top,
                k,
                TopKEntry {
                    dist,
                    idx: (base_idx + i) as u32,
                },
                &mut threshold,
                &mut filled,
            );
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

#[inline(never)]
fn fused_topk_parallel_f16<const ASC: bool>(
    query: &[f32],
    candidates: &[u16],
    dim: usize,
    k: usize,
    n: usize,
    dist_fn: impl Fn(&[f32], &[u16]) -> f32 + Send + Sync,
) -> (Vec<u32>, Vec<f32>) {
    if n < 4096 {
        return fused_topk_seq_f16::<ASC>(query, candidates, dim, k, 0, &dist_fn);
    }

    let n_threads = rayon::current_num_threads();
    let chunk_vecs = (n / n_threads).max(512);
    let chunk_values = chunk_vecs * dim;

    let chunk_results: Vec<Vec<TopKEntry>> = candidates
        .par_chunks(chunk_values)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let n_in_chunk = cand_chunk.len() / dim;
            let base_idx = chunk_idx * chunk_vecs;

            let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
            let mut threshold: f32 = if ASC {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
            let mut filled = false;

            for i in 0..n_in_chunk {
                if i + PREFETCH_AHEAD_VECTORS < n_in_chunk {
                    unsafe {
                        prefetch_read_data(
                            cand_chunk
                                .as_ptr()
                                .add((i + PREFETCH_AHEAD_VECTORS) * dim)
                                .cast(),
                        );
                    }
                }
                let cand = unsafe { cand_chunk.get_unchecked(i * dim..(i + 1) * dim) };
                let dist = dist_fn(query, cand);

                if !filled || passes_threshold::<ASC>(dist, threshold) {
                    topk_insert::<ASC>(
                        &mut top,
                        k,
                        TopKEntry {
                            dist,
                            idx: (base_idx + i) as u32,
                        },
                        &mut threshold,
                        &mut filled,
                    );
                }
            }

            if !filled && !top.is_empty() {
                if ASC {
                    top.sort_unstable_by(|a, b| {
                        a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal)
                    });
                } else {
                    top.sort_unstable_by(|a, b| {
                        b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal)
                    });
                }
            }
            top
        })
        .collect();

    merge_topk_results::<ASC>(&chunk_results, k)
}

fn fused_topk_seq_f16<const ASC: bool>(
    query: &[f32],
    candidates: &[u16],
    dim: usize,
    k: usize,
    base_idx: usize,
    dist_fn: &impl Fn(&[f32], &[u16]) -> f32,
) -> (Vec<u32>, Vec<f32>) {
    let n = candidates.len() / dim;
    let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
    let mut threshold: f32 = if ASC {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
    let mut filled = false;

    for i in 0..n {
        if i + PREFETCH_AHEAD_VECTORS < n {
            unsafe {
                prefetch_read_data(
                    candidates
                        .as_ptr()
                        .add((i + PREFETCH_AHEAD_VECTORS) * dim)
                        .cast(),
                );
            }
        }
        let cand = unsafe { candidates.get_unchecked(i * dim..(i + 1) * dim) };
        let dist = dist_fn(query, cand);

        if !filled || passes_threshold::<ASC>(dist, threshold) {
            topk_insert::<ASC>(
                &mut top,
                k,
                TopKEntry {
                    dist,
                    idx: (base_idx + i) as u32,
                },
                &mut threshold,
                &mut filled,
            );
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
    let mut threshold: f32 = if ASC {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
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

// ─── Filtered search helpers ─────────────────────────────────────────────────

/// Direct random access top-k for low-selectivity filters.
///
/// Instead of scanning ALL vectors, reads only the matching vectors by index.
/// O(matches) work — ideal when matches << n_vectors.
#[inline(never)]
fn direct_access_topk<const ASC: bool>(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n_vectors: usize,
    subset_ids: &[u64],
    dist_fn: impl Fn(&[f32], &[f32]) -> f32,
) -> (Vec<u32>, Vec<f32>) {
    let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
    let mut threshold: f32 = if ASC {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
    let mut filled = false;

    for &id in subset_ids {
        let idx = id as usize;
        if idx >= n_vectors {
            continue;
        }
        let start = idx * dim;
        let cand = unsafe { candidates.get_unchecked(start..start + dim) };
        let dist = dist_fn(query, cand);

        if !filled || passes_threshold::<ASC>(dist, threshold) {
            topk_insert::<ASC>(
                &mut top,
                k,
                TopKEntry {
                    dist,
                    idx: idx as u32,
                },
                &mut threshold,
                &mut filled,
            );
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

fn direct_access_topk_f16<const ASC: bool>(
    query: &[f32],
    candidates: &[u16],
    dim: usize,
    k: usize,
    n_vectors: usize,
    subset_ids: &[u64],
    dist_fn: impl Fn(&[f32], &[u16]) -> f32,
) -> (Vec<u32>, Vec<f32>) {
    let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
    let mut threshold: f32 = if ASC {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
    let mut filled = false;

    for &id in subset_ids {
        let idx = id as usize;
        if idx >= n_vectors {
            continue;
        }
        let start = idx * dim;
        let cand = unsafe { candidates.get_unchecked(start..start + dim) };
        let dist = dist_fn(query, cand);

        if !filled || passes_threshold::<ASC>(dist, threshold) {
            topk_insert::<ASC>(
                &mut top,
                k,
                TopKEntry {
                    dist,
                    idx: idx as u32,
                },
                &mut threshold,
                &mut filled,
            );
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

fn search_filtered_f16(
    query: &[f32],
    candidates: &[u16],
    dim: usize,
    k: usize,
    n: usize,
    metric: DistanceMetric,
    subset_indices: &[u64],
) -> (Vec<u32>, Vec<f32>) {
    const DIRECT_ACCESS_LIMIT: usize = 50_000;

    if subset_indices.len() <= DIRECT_ACCESS_LIMIT {
        return match metric {
            DistanceMetric::InnerProduct => direct_access_topk_f16::<false>(
                query,
                candidates,
                dim,
                k,
                n,
                subset_indices,
                simd::inner_product_f16,
            ),
            DistanceMetric::L2Squared => direct_access_topk_f16::<true>(
                query,
                candidates,
                dim,
                k,
                n,
                subset_indices,
                simd::l2_squared_f16,
            ),
            DistanceMetric::Cosine => direct_access_topk_f16::<true>(
                query,
                candidates,
                dim,
                k,
                n,
                subset_indices,
                simd::cosine_distance_f16,
            ),
            _ => direct_access_topk_f16::<true>(
                query,
                candidates,
                dim,
                k,
                n,
                subset_indices,
                |a, b| distance::compute_distance_f16(a, b, metric),
            ),
        };
    }

    let max_id = subset_indices.iter().copied().max().unwrap_or(0) as usize;
    let mut bitset = vec![0u64; (max_id / 64) + 1];
    for &id in subset_indices {
        let idx = id as usize;
        if idx < n {
            bitset[idx / 64] |= 1u64 << (idx % 64);
        }
    }

    match metric {
        DistanceMetric::InnerProduct => fused_topk_parallel_filtered_f16::<false>(
            query,
            candidates,
            dim,
            k,
            n,
            &bitset,
            max_id,
            simd::inner_product_f16,
        ),
        DistanceMetric::L2Squared => fused_topk_parallel_filtered_f16::<true>(
            query,
            candidates,
            dim,
            k,
            n,
            &bitset,
            max_id,
            simd::l2_squared_f16,
        ),
        DistanceMetric::Cosine => fused_topk_parallel_filtered_f16::<true>(
            query,
            candidates,
            dim,
            k,
            n,
            &bitset,
            max_id,
            simd::cosine_distance_f16,
        ),
        _ => fused_topk_parallel_filtered_f16::<true>(
            query,
            candidates,
            dim,
            k,
            n,
            &bitset,
            max_id,
            |a, b| distance::compute_distance_f16(a, b, metric),
        ),
    }
}

/// Parallel scan with bitset filtering for high-selectivity filters.
///
/// Same structure as `fused_topk_parallel` but skips non-matching vectors
/// via O(1) bitset lookup. Retains full SIMD parallelism.
#[inline(never)]
fn fused_topk_parallel_filtered<const ASC: bool>(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    bitset: &[u64],
    max_id: usize,
    dist_fn: impl Fn(&[f32], &[f32]) -> f32 + Send + Sync,
) -> (Vec<u32>, Vec<f32>) {
    if n < 4096 {
        // Sequential fallback for small datasets
        let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
        let mut threshold: f32 = if ASC {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };
        let mut filled = false;

        for i in 0..n {
            if i > max_id || (bitset[i / 64] & (1u64 << (i % 64))) == 0 {
                continue;
            }
            let start = i * dim;
            let cand = unsafe { candidates.get_unchecked(start..start + dim) };
            let dist = dist_fn(query, cand);
            if !filled || passes_threshold::<ASC>(dist, threshold) {
                topk_insert::<ASC>(
                    &mut top,
                    k,
                    TopKEntry {
                        dist,
                        idx: i as u32,
                    },
                    &mut threshold,
                    &mut filled,
                );
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
        return (indices, dists);
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
            let mut threshold: f32 = if ASC {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
            let mut filled = false;

            for i in 0..n_in_chunk {
                let global_id = base_idx + i;
                // Bitset check: skip non-matching vectors
                if global_id > max_id || (bitset[global_id / 64] & (1u64 << (global_id % 64))) == 0
                {
                    continue;
                }

                let start = i * dim;
                let cand = unsafe { cand_chunk.get_unchecked(start..start + dim) };
                let dist = dist_fn(query, cand);

                if !filled || passes_threshold::<ASC>(dist, threshold) {
                    topk_insert::<ASC>(
                        &mut top,
                        k,
                        TopKEntry {
                            dist,
                            idx: global_id as u32,
                        },
                        &mut threshold,
                        &mut filled,
                    );
                }
            }

            if !filled && !top.is_empty() {
                if ASC {
                    top.sort_unstable_by(|a, b| {
                        a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal)
                    });
                } else {
                    top.sort_unstable_by(|a, b| {
                        b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal)
                    });
                }
            }
            top
        })
        .collect();

    merge_topk_results::<ASC>(&chunk_results, k)
}

#[inline(never)]
fn fused_topk_parallel_filtered_f16<const ASC: bool>(
    query: &[f32],
    candidates: &[u16],
    dim: usize,
    k: usize,
    n: usize,
    bitset: &[u64],
    max_id: usize,
    dist_fn: impl Fn(&[f32], &[u16]) -> f32 + Send + Sync,
) -> (Vec<u32>, Vec<f32>) {
    if n < 4096 {
        let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
        let mut threshold: f32 = if ASC {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };
        let mut filled = false;

        for i in 0..n {
            if i > max_id || (bitset[i / 64] & (1u64 << (i % 64))) == 0 {
                continue;
            }
            let cand = unsafe { candidates.get_unchecked(i * dim..(i + 1) * dim) };
            let dist = dist_fn(query, cand);
            if !filled || passes_threshold::<ASC>(dist, threshold) {
                topk_insert::<ASC>(
                    &mut top,
                    k,
                    TopKEntry {
                        dist,
                        idx: i as u32,
                    },
                    &mut threshold,
                    &mut filled,
                );
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
        return (indices, dists);
    }

    let n_threads = rayon::current_num_threads();
    let chunk_vecs = (n / n_threads).max(512);
    let chunk_values = chunk_vecs * dim;

    let chunk_results: Vec<Vec<TopKEntry>> = candidates
        .par_chunks(chunk_values)
        .enumerate()
        .map(|(chunk_idx, cand_chunk)| {
            let n_in_chunk = cand_chunk.len() / dim;
            let base_idx = chunk_idx * chunk_vecs;

            let mut top: Vec<TopKEntry> = Vec::with_capacity(k);
            let mut threshold: f32 = if ASC {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
            let mut filled = false;

            for i in 0..n_in_chunk {
                let global_id = base_idx + i;
                if global_id > max_id || (bitset[global_id / 64] & (1u64 << (global_id % 64))) == 0
                {
                    continue;
                }

                let cand = unsafe { cand_chunk.get_unchecked(i * dim..(i + 1) * dim) };
                let dist = dist_fn(query, cand);

                if !filled || passes_threshold::<ASC>(dist, threshold) {
                    topk_insert::<ASC>(
                        &mut top,
                        k,
                        TopKEntry {
                            dist,
                            idx: global_id as u32,
                        },
                        &mut threshold,
                        &mut filled,
                    );
                }
            }

            if !filled && !top.is_empty() {
                if ASC {
                    top.sort_unstable_by(|a, b| {
                        a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal)
                    });
                } else {
                    top.sort_unstable_by(|a, b| {
                        b.dist.partial_cmp(&a.dist).unwrap_or(Ordering::Equal)
                    });
                }
            }
            top
        })
        .collect();

    merge_topk_results::<ASC>(&chunk_results, k)
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
                if val < mins[d] {
                    mins[d] = val;
                }
                if val > maxs[d] {
                    maxs[d] = val;
                }
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
    {
        return sq8_dot_neon(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    {
        return sq8_dot_scalar(a, b);
    }

    #[allow(unreachable_code)]
    sq8_dot_scalar(a, b)
}

/// Approximate L2² of two u8 vectors.  Lower = more similar.
#[inline(always)]
fn sq8_l2sq(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        return sq8_l2sq_neon(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    {
        return sq8_l2sq_scalar(a, b);
    }

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
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as u32) * (y as u32))
        .sum()
}

#[inline]
fn sq8_l2sq_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = (x as i32) - (y as i32);
            (d * d) as u32
        })
        .sum()
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
    let mut exact: Vec<TopKEntry> = candidate_indices
        .iter()
        .map(|&idx| {
            let base = (idx as usize) * dim;
            let cand = unsafe { candidates.get_unchecked(base..base + dim) };
            let dist = match metric {
                DistanceMetric::InnerProduct => simd::inner_product_f32(query, cand),
                DistanceMetric::L2Squared => simd::l2_squared_f32(query, cand),
                DistanceMetric::Cosine => simd::cosine_distance_f32(query, cand),
                _ => distance::compute_distance_f32(query, cand, metric),
            };
            TopKEntry { dist, idx }
        })
        .collect();

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
                        unsafe {
                            prefetch_read_data(cand_chunk.as_ptr().add((i + 4) * dim));
                        }
                    }
                    let cand = unsafe { cand_chunk.get_unchecked(i * dim..(i + 1) * dim) };
                    let dist = dist_fn(query_u8, cand) as f32;
                    let entry = TopKEntry {
                        dist,
                        idx: (base_idx + i) as u32,
                    };
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
                        unsafe {
                            prefetch_read_data(cand_chunk.as_ptr().add((i + 4) * dim));
                        }
                    }
                    let cand = unsafe { cand_chunk.get_unchecked(i * dim..(i + 1) * dim) };
                    let dist = dist_fn(query_u8, cand) as f32;
                    let entry = TopKEntry {
                        dist,
                        idx: (base_idx + i) as u32,
                    };
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

        let mut store = FlatMmap::open(&path, dim, VectorDtype::F32).unwrap();
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
        let (ids, dists) = store.search(&query, 2, DistanceMetric::InnerProduct, false, None);
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], 0); // exact match (IP=1.0)
        assert!((dists[0] - 1.0).abs() < 1e-6);

        // Search: L2, query = origin, k=2 → closest should be vec 0 and vec 1 (tied at 1.0)
        let query_origin = vec![0.0f32, 0.0, 0.0, 0.0];
        let (ids_l2, _dists_l2) =
            store.search(&query_origin, 1, DistanceMetric::L2Squared, false, None);
        // All unit vectors have L2=1.0 to origin, mixed has L2=0.5
        assert_eq!(ids_l2[0], 2); // vec 2 is closest (0.5^2+0.5^2=0.5)
    }

    #[test]
    fn test_flat_mmap_append() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("vectors.bin");
        let dim = 2;

        let mut store = FlatMmap::open(&path, dim, VectorDtype::F32).unwrap();

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
            let mut store = FlatMmap::open(&path, dim, VectorDtype::F32).unwrap();
            store.write(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        }

        // Reopen and verify
        let store = FlatMmap::open(&path, dim, VectorDtype::F32).unwrap();
        assert_eq!(store.len(), 2);
        let (ids, _) = store.search(&[1.0, 2.0, 3.0], 1, DistanceMetric::L2Squared, false, None);
        assert_eq!(ids[0], 0);
    }

    #[test]
    fn test_sampled_dims_cover_tail_not_only_prefix() {
        let dims = sampled_dim_indices(128, 68);
        assert_eq!(dims.len(), 68);
        assert_eq!(dims[0], 0);
        assert_eq!(*dims.last().unwrap(), 127);
        assert!(dims.iter().any(|&d| d >= 96));
    }

    #[test]
    fn test_eps_controls_approx_work_budget_monotonically() {
        let k = 10;
        let n = 1_000_000;
        let dim = 128;

        let tight = approx_ip_order_pool_size(k, n, 1e-6);
        let four_digits = approx_ip_order_pool_size(k, n, 1e-4);
        let loose = approx_ip_order_pool_size(k, n, 1e-2);
        assert!(tight > four_digits);
        assert!(four_digits > loose);

        let tight_dims = approx_hybrid_sample_dims(dim, 1e-6);
        let four_digit_dims = approx_hybrid_sample_dims(dim, 1e-4);
        let loose_dims = approx_hybrid_sample_dims(dim, 1e-2);
        assert!(tight_dims > four_digit_dims);
        assert!(four_digit_dims > loose_dims);

        let tight_pool = approx_hybrid_pool_size(k, n, 1e-6);
        let four_digit_pool = approx_hybrid_pool_size(k, n, 1e-4);
        let loose_pool = approx_hybrid_pool_size(k, n, 1e-2);
        assert!(tight_pool >= four_digit_pool);
        assert!(four_digit_pool > loose_pool);
    }

    #[test]
    fn test_approx_search_finds_tail_only_l2_match_beyond_pool() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("vectors.bin");
        let dim = 128;
        let n = 4096;
        let true_idx = 3000u32;

        let mut data = vec![0.0f32; n * dim];
        let mut query = vec![0.0f32; dim];
        for j in 96..dim {
            query[j] = 1.0;
            data[true_idx as usize * dim + j] = 1.0;
        }

        let mut store = FlatMmap::open(&path, dim, VectorDtype::F32).unwrap();
        store.write(&data).unwrap();

        let (ids, dists) = store.search(
            &query,
            1,
            DistanceMetric::L2Squared,
            false,
            Some(crate::storage::approx_search::ApproxSearchConfig::new(1e-4)),
        );

        assert_eq!(ids, vec![true_idx]);
        assert_eq!(dists, vec![0.0]);
    }

    #[test]
    fn test_approx_search_finds_tail_only_ip_match_beyond_pool() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("vectors.bin");
        let dim = 128;
        let n = 4096;
        let true_idx = 3500u32;

        let mut data = vec![0.0f32; n * dim];
        let mut query = vec![0.0f32; dim];
        for j in 96..dim {
            query[j] = 1.0;
            data[true_idx as usize * dim + j] = 1.0;
        }

        let mut store = FlatMmap::open(&path, dim, VectorDtype::F32).unwrap();
        store.write(&data).unwrap();

        let (ids, dists) = store.search(
            &query,
            1,
            DistanceMetric::InnerProduct,
            false,
            Some(crate::storage::approx_search::ApproxSearchConfig::new(1e-4)),
        );

        assert_eq!(ids, vec![true_idx]);
        assert_eq!(dists, vec![32.0]);
    }

    #[test]
    fn test_approx_ip_order_handles_signs_and_append_invalidation() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("vectors.bin");
        let dim = 32;
        let n = 70_000;
        let positive_idx = 12_345u32;
        let negative_idx = 54_321u32;

        let mut data = vec![0.0f32; n * dim];
        for j in 0..dim {
            data[positive_idx as usize * dim + j] = 1.0;
            data[negative_idx as usize * dim + j] = -2.0;
        }

        let mut store = FlatMmap::open(&path, dim, VectorDtype::F32).unwrap();
        store.write(&data).unwrap();
        let config = Some(crate::storage::approx_search::ApproxSearchConfig::new(1e-4));

        let positive_query = vec![1.0f32; dim];
        let (ids, dists) = store.search(
            &positive_query,
            1,
            DistanceMetric::InnerProduct,
            false,
            config,
        );
        assert_eq!(ids, vec![positive_idx]);
        assert_eq!(dists, vec![32.0]);

        let negative_query = vec![-1.0f32; dim];
        let (ids, dists) = store.search(
            &negative_query,
            1,
            DistanceMetric::InnerProduct,
            false,
            config,
        );
        assert_eq!(ids, vec![negative_idx]);
        assert_eq!(dists, vec![64.0]);

        let appended_idx = n as u32;
        store.write(&vec![2.0f32; dim]).unwrap();
        let (ids, dists) = store.search(
            &positive_query,
            1,
            DistanceMetric::InnerProduct,
            false,
            config,
        );
        assert_eq!(ids, vec![appended_idx]);
        assert_eq!(dists, vec![64.0]);
    }

    #[test]
    fn test_approx_ip_order_persists_and_rejects_stale_sidecar() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("vectors.bin");
        let dim = 32;
        let n = 70_000;
        let top_idx = 12_345u32;

        let mut data = vec![0.0f32; n * dim];
        for j in 0..dim {
            data[top_idx as usize * dim + j] = 1.0;
        }

        let mut store = FlatMmap::open(&path, dim, VectorDtype::F32).unwrap();
        store.write(&data).unwrap();
        let query = vec![1.0f32; dim];
        let config = Some(crate::storage::approx_search::ApproxSearchConfig::new(1e-4));

        let (ids, _) = store.search(&query, 1, DistanceMetric::InnerProduct, false, config);
        assert_eq!(ids, vec![top_idx]);

        let sidecar_path = approx_ip_order_path(&path);
        assert!(sidecar_path.exists());

        drop(store);
        let reopened = FlatMmap::open(&path, dim, VectorDtype::F32).unwrap();
        assert!(reopened.approx_ip_order.read().is_some());
        let (ids, _) = reopened.search(&query, 1, DistanceMetric::InnerProduct, false, config);
        assert_eq!(ids, vec![top_idx]);

        drop(reopened);
        let appended_idx = n as u32;
        let appended = vec![2.0f32; dim];
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        let bytes = unsafe {
            std::slice::from_raw_parts(appended.as_ptr() as *const u8, appended.len() * 4)
        };
        file.write_all(bytes).unwrap();
        file.flush().unwrap();

        let reopened_stale = FlatMmap::open(&path, dim, VectorDtype::F32).unwrap();
        assert!(reopened_stale.approx_ip_order.read().is_none());
        let (ids, dists) =
            reopened_stale.search(&query, 1, DistanceMetric::InnerProduct, false, config);
        assert_eq!(ids, vec![appended_idx]);
        assert_eq!(dists, vec![64.0]);
        assert!(reopened_stale.approx_ip_order.read().is_some());
    }

    #[test]
    fn test_approx_norm_cache_matches_exact_and_rejects_stale_sidecar() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("vectors.bin");
        let dim = 32;
        let n = 96;
        let query_idx = 17usize;

        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                let mixed = (i * 131 + j * 17 + (i * j * 7) % 19) % 997;
                let value = mixed as f32 / 997.0 - 0.5 + i as f32 * 1e-5;
                data[i * dim + j] = value;
            }
        }
        for j in 0..dim {
            data[5 * dim + j] = 0.0;
        }
        let query = data[query_idx * dim..(query_idx + 1) * dim].to_vec();

        let mut store = FlatMmap::open(&path, dim, VectorDtype::F32).unwrap();
        store.write(&data).unwrap();
        let sidecar_path = approx_norms_path(&path);
        assert!(sidecar_path.exists());
        assert!(store.approx_norms.read().is_some());

        let config = Some(crate::storage::approx_search::ApproxSearchConfig::new(1e-8));
        let exact_l2 = store.search(&query, 5, DistanceMetric::L2Squared, false, None);
        let approx_l2 = store.search(&query, 5, DistanceMetric::L2Squared, false, config);
        assert_eq!(approx_l2.0, exact_l2.0);
        for (actual, expected) in approx_l2.1.iter().zip(exact_l2.1.iter()) {
            assert!((actual - expected).abs() <= 1e-4);
        }

        let exact_cos = store.search(&query, 5, DistanceMetric::Cosine, false, None);
        let approx_cos = store.search(&query, 5, DistanceMetric::Cosine, false, config);
        assert_eq!(approx_cos.0, exact_cos.0);
        for (actual, expected) in approx_cos.1.iter().zip(exact_cos.1.iter()) {
            assert!((actual - expected).abs() <= 1e-4);
        }

        drop(store);
        let reopened = FlatMmap::open(&path, dim, VectorDtype::F32).unwrap();
        assert!(reopened.approx_norms.read().is_some());
        drop(reopened);

        let appended = vec![3.0f32; dim];
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        let bytes = unsafe {
            std::slice::from_raw_parts(appended.as_ptr() as *const u8, appended.len() * 4)
        };
        file.write_all(bytes).unwrap();
        file.flush().unwrap();

        let reopened_stale = FlatMmap::open(&path, dim, VectorDtype::F32).unwrap();
        assert!(reopened_stale.approx_norms.read().is_none());
        let _ = reopened_stale.search(&query, 5, DistanceMetric::Cosine, false, config);
        assert!(reopened_stale.approx_norms.read().is_some());
    }

    #[test]
    fn packed_binary_cache_matches_reference_metrics() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("vectors.bin");
        let dim = 130;
        let mut rows = vec![0.0f32; dim * 3];
        for index in [0usize, 1, 64, 129] {
            rows[index] = 1.0;
            rows[dim + index] = 1.0;
        }
        rows[dim + 5] = 1.0;
        for index in [2usize, 3, 65] {
            rows[dim * 2 + index] = 1.0;
        }
        let query = rows[..dim].to_vec();

        let mut store = FlatMmap::open(&path, dim, VectorDtype::F32).unwrap();
        store.write(&rows).unwrap();
        assert!(store.binary.read().is_none());

        for metric in [
            DistanceMetric::Hamming,
            DistanceMetric::Jaccard,
            DistanceMetric::Tanimoto,
            DistanceMetric::Dice,
        ] {
            let packed = store.search(&query, 3, metric, false, None);
            let reference = crate::distance::top_k_search(&query, &rows, dim, 3, metric);
            assert_eq!(packed.0, reference.0, "{metric:?}");
            for (actual, expected) in packed.1.iter().zip(reference.1) {
                assert!((actual - expected).abs() < 1e-6, "{metric:?}");
            }
        }
        let packed_bytes = store.binary.read().as_ref().unwrap().data.len() * 8;
        assert_eq!(packed_bytes, 3 * dim.div_ceil(64) * 8);
        assert!(packed_bytes < rows.len() * std::mem::size_of::<f32>() / 8);
    }
}
