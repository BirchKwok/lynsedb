//! RaBitQ index: Random Binary Quantization with Randomized Hadamard Transform.
//!
//! Algorithm (based on RaBitQ 2024):
//! 1. **Build**: generate random sign vector → apply Randomized Hadamard Transform (RHT)
//!    to each vector → store 1 bit per dimension + original norm.
//! 2. **Search**: rotate query with same RHT → build per-byte asymmetric LUT →
//!    compute IP estimate for each binary code → re-rank top-N with exact f32.
//!
//! Key properties:
//! - **32x compression**: dim f32 values → dim/8 bytes
//! - **O(dim/8) per vector** in binary scan (popcount/byte-LUT)
//! - **Asymmetric estimation**: query in f32, database in binary → better recall than
//!   symmetric binary search
//! - **Two-pass**: binary approx scan → exact f32 re-rank for high recall
//!
//! Randomized Hadamard Transform:
//! 1. Pad vector to `padded_dim = next_power_of_two(dim)` (zero-pad)
//! 2. Apply random signs: `x[i] *= sign_i` where `sign_i ∈ {±1}`
//! 3. Apply Fast Walsh-Hadamard Transform (FWHT): O(D log D)
//! The RHT is equivalent to multiplication by a random orthogonal matrix
//! with O(D log D) application complexity instead of O(D²).

use crate::distance::DistanceMetric;
use crate::storage::pq_mmap::rescore_exact;
use rayon::prelude::*;
use rand::{SeedableRng, Rng};
use rand::rngs::SmallRng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const RABITQ_MAGIC: u32 = 0x5242_5451; // "RBTQ"
const RABITQ_VERSION: u32 = 1;
/// Default oversample for two-pass re-ranking.
/// The binary scan over 1M vectors dominates latency (~2.8ms, fixed cost);
/// increasing candidates from 640→2000 adds only ~0.5ms re-ranking overhead
/// while substantially improving recall from ~82% toward ≥90%.
pub const DEFAULT_OVERSAMPLE: usize = 200;

// ─── RaBitQ Index ─────────────────────────────────────────────────────────────

/// RaBitQ index: Randomized Hadamard binary quantization.
///
/// Stores each vector as `padded_dim / 8` bytes of binary codes plus its L2 norm.
/// Search uses a per-byte asymmetric lookup table to estimate inner products
/// without decompressing the binary codes.
pub struct RaBitQIndex {
    /// Original vector dimension.
    dim: usize,
    /// Padded dimension (= next power of two ≥ dim).
    padded_dim: usize,
    /// Number of code bytes per vector (= padded_dim / 8).
    code_bytes: usize,
    /// Random sign vector packed as u64 words: bit=0 → sign=+1, bit=1 → sign=−1.
    /// Length: padded_dim / 64 (rounded up).
    sign_words: Vec<u64>,
    /// Binary codes: \[n_vectors × code_bytes\] u8 flat array.
    codes: Vec<u8>,
    /// Original L2 norms: \[n_vectors\] f32.
    norms: Vec<f32>,
    n_vectors: usize,
}

impl RaBitQIndex {
    /// Build a RaBitQ index from raw f32 data.
    pub fn build(data: &[f32], n_vectors: usize, dim: usize) -> Self {
        assert!(n_vectors > 0, "need at least one vector");
        assert!(dim > 0, "dimension must be positive");

        let padded_dim = dim.next_power_of_two();
        let code_bytes = padded_dim / 8;
        let sign_words_len = (padded_dim + 63) / 64;

        // Generate random sign vector from a fixed seed for reproducibility
        let sign_words = generate_sign_words(sign_words_len, 42);

        // Encode all vectors in parallel
        let n_total = n_vectors;
        let mut codes = vec![0u8; n_total * code_bytes];
        let mut norms = vec![0.0f32; n_total];

        // Use a temporary buffer per thread; process in parallel chunks
        let chunk_size = 4096;
        let n_chunks = (n_total + chunk_size - 1) / chunk_size;

        // Parallel encoding: each chunk processes chunk_size vectors
        let results: Vec<(usize, Vec<u8>, Vec<f32>)> = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(n_total);
                let n_in_chunk = end - start;

                let mut chunk_codes = vec![0u8; n_in_chunk * code_bytes];
                let mut chunk_norms = vec![0.0f32; n_in_chunk];

                let mut buf = vec![0.0f32; padded_dim];

                for i in 0..n_in_chunk {
                    let vec_idx = start + i;
                    let src = &data[vec_idx * dim..(vec_idx + 1) * dim];

                    // Compute L2 norm of original vector
                    let norm = src.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    chunk_norms[i] = norm;

                    // Pad to padded_dim (zero-pad extra dimensions)
                    buf[..dim].copy_from_slice(src);
                    for j in dim..padded_dim {
                        buf[j] = 0.0;
                    }

                    // Apply random signs: buf[i] *= sign_i
                    apply_signs(&mut buf, &sign_words, padded_dim);

                    // Apply Fast Walsh-Hadamard Transform
                    fwht(&mut buf[..padded_dim]);

                    // Quantize to bits: bit_i = (buf[i] >= 0)
                    let dst = &mut chunk_codes[i * code_bytes..(i + 1) * code_bytes];
                    for byte_idx in 0..code_bytes {
                        let mut byte_val = 0u8;
                        for bit in 0..8 {
                            let d = byte_idx * 8 + bit;
                            if d < padded_dim && buf[d] >= 0.0 {
                                byte_val |= 1 << bit;
                            }
                        }
                        dst[byte_idx] = byte_val;
                    }
                }
                (start, chunk_codes, chunk_norms)
            })
            .collect();

        // Merge results
        for (start, chunk_codes, chunk_norms) in results {
            let end = (start + chunk_size).min(n_total);
            let n_in = end - start;
            codes[start * code_bytes..end * code_bytes].copy_from_slice(&chunk_codes[..n_in * code_bytes]);
            norms[start..end].copy_from_slice(&chunk_norms[..n_in]);
        }

        RaBitQIndex {
            dim,
            padded_dim,
            code_bytes,
            sign_words,
            codes,
            norms,
            n_vectors,
        }
    }

    /// Binary ADC search with exact f32 re-rank.
    ///
    /// Pass 1: rotate query → build per-byte LUT → scan binary codes for top-N.
    /// Pass 2: exact f32 re-score on top-N candidates → return top-k.
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

        // Rotate query: apply signs + FWHT
        let mut q_rot = vec![0.0f32; self.padded_dim];
        q_rot[..self.dim].copy_from_slice(query);
        // zero-pad already done (Vec initialized to 0)
        apply_signs(&mut q_rot, &self.sign_words, self.padded_dim);
        fwht(&mut q_rot[..self.padded_dim]);

        // Precompute: total sum of rotated query (used in asymmetric estimate)
        let total_q: f32 = q_rot.iter().sum();

        // Build per-byte lookup table: lut[b * 256 + v] = sum of q_rot[b*8..b*8+8] at set bits
        let lut = build_byte_lut(&q_rot, self.padded_dim);

        // Pass 1: binary ADC scan → top-N candidate indices
        let candidates = binary_scan_topn(
            &self.codes,
            &self.norms,
            self.n_vectors,
            self.code_bytes,
            &lut,
            total_q,
            self.padded_dim,
            n_candidates,
            metric,
        );

        // Pass 2: exact f32 re-score
        rescore_exact(&candidates, query, f32_data, self.dim, k, metric)
    }

    /// Number of indexed vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    /// Save the RaBitQ index to `path`.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        w.write_all(&RABITQ_MAGIC.to_le_bytes())?;
        w.write_all(&RABITQ_VERSION.to_le_bytes())?;
        w.write_all(&(self.dim as u32).to_le_bytes())?;
        w.write_all(&(self.padded_dim as u32).to_le_bytes())?;
        w.write_all(&(self.n_vectors as u64).to_le_bytes())?;

        // Sign words (u64 LE)
        w.write_all(&(self.sign_words.len() as u32).to_le_bytes())?;
        for &word in &self.sign_words {
            w.write_all(&word.to_le_bytes())?;
        }

        // Binary codes (raw u8)
        w.write_all(&self.codes)?;

        // Norms (f32 LE)
        for &n in &self.norms {
            w.write_all(&n.to_le_bytes())?;
        }

        w.flush()?;
        Ok(())
    }

    /// Load a RaBitQ index from `path`.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);

        let magic = read_u32le(&mut r)?;
        if magic != RABITQ_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid RaBitQ magic bytes",
            ));
        }
        let _version = read_u32le(&mut r)?;
        let dim         = read_u32le(&mut r)? as usize;
        let padded_dim  = read_u32le(&mut r)? as usize;
        let n_vectors   = read_u64le(&mut r)? as usize;
        let code_bytes  = padded_dim / 8;

        let n_sign_words = read_u32le(&mut r)? as usize;
        let mut sign_words = vec![0u64; n_sign_words];
        for word in sign_words.iter_mut() {
            let bytes = read_8bytes(&mut r)?;
            *word = u64::from_le_bytes(bytes);
        }

        let mut codes = vec![0u8; n_vectors * code_bytes];
        r.read_exact(&mut codes)?;

        let mut norms = vec![0.0f32; n_vectors];
        for n in norms.iter_mut() {
            let bytes = read_4bytes(&mut r)?;
            *n = f32::from_le_bytes(bytes);
        }

        Ok(RaBitQIndex { dim, padded_dim, code_bytes, sign_words, codes, norms, n_vectors })
    }
}

// ─── Randomized Hadamard Transform helpers ────────────────────────────────────

/// Generate a random sign vector packed as u64 words.
/// bit=0 → sign=+1, bit=1 → sign=−1.
fn generate_sign_words(n_words: usize, seed: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n_words).map(|_| rng.gen::<u64>()).collect()
}

/// Apply element-wise random signs to `buf`.
/// `sign_words[i/64] bit (i%64)`: 0 → +1 (no-op), 1 → -1 (negate).
#[inline]
fn apply_signs(buf: &mut [f32], sign_words: &[u64], padded_dim: usize) {
    for i in 0..padded_dim {
        let word_idx = i / 64;
        let bit_idx = i % 64;
        if word_idx < sign_words.len() && (sign_words[word_idx] >> bit_idx) & 1 == 1 {
            buf[i] = -buf[i];
        }
    }
}

/// In-place Fast Walsh-Hadamard Transform (unnormalized).
///
/// Requires `data.len()` to be a power of two.
/// Complexity: O(n log n).
fn fwht(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two(), "FWHT requires power-of-two length");
    let mut h = 1usize;
    while h < n {
        let step = h * 2;
        let mut i = 0;
        while i < n {
            for j in 0..h {
                let x = data[i + j];
                let y = data[i + j + h];
                data[i + j]     = x + y;
                data[i + j + h] = x - y;
            }
            i += step;
        }
        h = step;
    }
}

// ─── Per-byte Asymmetric LUT ──────────────────────────────────────────────────

/// Build the per-byte asymmetric lookup table for the rotated query.
///
/// `lut[b * 256 + v]` = sum of `q_rot[b*8 .. b*8+8]` at the bit positions set in `v`.
///
/// For a database code byte `c`, this gives: `lut[b * 256 + c]` = partial IP.
/// Total asymmetric estimate: `2 * Σ_b lut[b][code_b] − total_q`
fn build_byte_lut(q_rot: &[f32], padded_dim: usize) -> Vec<f32> {
    let n_bytes = padded_dim / 8;
    let mut lut = vec![0.0f32; n_bytes * 256];
    for b in 0..n_bytes {
        let base = b * 8;
        let lut_base = b * 256;
        for v in 0u8..=255 {
            let mut s = 0.0f32;
            let mut bits = v;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                if base + bit < padded_dim {
                    s += q_rot[base + bit];
                }
                bits &= bits - 1; // clear lowest set bit
            }
            lut[lut_base + v as usize] = s;
        }
    }
    lut
}

// ─── Binary ADC scan ──────────────────────────────────────────────────────────

/// Compact heap entry for binary scan.
#[derive(Clone, Copy)]
struct BinEntry {
    score: f32,
    idx: u32,
}

impl PartialEq for BinEntry {
    fn eq(&self, other: &Self) -> bool { self.score == other.score }
}
impl Eq for BinEntry {}
impl PartialOrd for BinEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for BinEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal)
    }
}

/// Parallel binary ADC scan using the per-byte LUT.
///
/// Asymmetric IP estimate for vector `i`:
/// ```text
/// ip_est = (2 * Σ_b lut[b][code_i_b] − total_q) * norms[i]
/// ```
///
/// For IP (descending): maximize `ip_est`.
/// For L2/Cosine (ascending): minimize `−ip_est`.
fn binary_scan_topn(
    codes: &[u8],
    norms: &[f32],
    n_vectors: usize,
    code_bytes: usize,
    lut: &[f32],
    total_q: f32,
    padded_dim: usize,
    n_candidates: usize,
    metric: DistanceMetric,
) -> Vec<u32> {
    // For IP: want largest ip_est → descending, ascending=false
    // For L2/Cosine: want closest = smallest L2 → negate ip_est → ascending=true
    let ascending = metric.is_ascending();

    let n_threads = rayon::current_num_threads().max(1);
    let chunk_vecs = (n_vectors / n_threads).max(256);

    let chunk_results: Vec<Vec<u32>> = (0..n_vectors)
        .into_par_iter()
        .chunks(chunk_vecs)
        .enumerate()
        .map(|(chunk_idx, range_chunk)| {
            let base_idx = chunk_idx * chunk_vecs;
            let n_in = range_chunk.len();

            if ascending {
                // Keep smallest N: MAX-heap
                let mut heap: BinaryHeap<BinEntry> = BinaryHeap::with_capacity(n_candidates + 1);
                for i in 0..n_in {
                    let vi = base_idx + i;
                    let score = compute_binary_score(codes, norms, vi, code_bytes, lut, total_q, padded_dim, true);
                    let entry = BinEntry { score, idx: vi as u32 };
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
                // Keep largest N: MIN-heap
                let mut heap: BinaryHeap<std::cmp::Reverse<BinEntry>> =
                    BinaryHeap::with_capacity(n_candidates + 1);
                for i in 0..n_in {
                    let vi = base_idx + i;
                    let score = compute_binary_score(codes, norms, vi, code_bytes, lut, total_q, padded_dim, false);
                    let entry = BinEntry { score, idx: vi as u32 };
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
    let mut all: Vec<(f32, u32)> = chunk_results
        .into_iter()
        .flatten()
        .map(|idx| {
            let score = compute_binary_score(codes, norms, idx as usize, code_bytes, lut, total_q, padded_dim, ascending);
            (score, idx)
        })
        .collect();

    if ascending {
        all.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    } else {
        all.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    }
    all.truncate(n_candidates);
    all.iter().map(|e| e.1).collect()
}

/// Compute the binary asymmetric score for vector `vi`.
///
/// - **IP** (`ascending=false`):  `ip_raw × norm_db` — maximize.
/// - **L2/Cosine** (`ascending=true`): `norm_db² − 2 × norm_db × ip_raw / d` — minimize.
///
/// Derivation: unnormalised FWHT scales norms by `sqrt(d)`, so:
///   `d × ‖q−x‖² = ‖q_rot − x_rot‖² ≈ const + norm_db² × d − 2 × norm_db × ip_raw`
/// Dividing by `d` and dropping the constant per-query term:
///   `score = norm_db² − 2 × norm_db × ip_raw / d`
/// This correctly balances the norm and IP terms for non-unit-norm vectors (e.g. SIFT).
#[inline(always)]
fn compute_binary_score(
    codes: &[u8],
    norms: &[f32],
    vi: usize,
    code_bytes: usize,
    lut: &[f32],
    total_q: f32,
    padded_dim: usize,
    ascending: bool,
) -> f32 {
    let code = unsafe { codes.get_unchecked(vi * code_bytes..(vi + 1) * code_bytes) };
    let mut sum_set: f32 = 0.0;
    for b in 0..code_bytes {
        sum_set += unsafe { *lut.get_unchecked(b * 256 + code[b] as usize) };
    }
    // ip_raw = Σ_i q_rot_i · sign(h_x_i)  (= 2·sum_set − total_q)
    let ip_raw = 2.0 * sum_set - total_q;
    let norm_db = unsafe { *norms.get_unchecked(vi) };
    if ascending {
        // L2 / Cosine: score ∝ norm_db² − 2·norm_db·ip_raw / d  (‖q‖² is constant)
        norm_db * norm_db - 2.0 * ip_raw * norm_db / padded_dim as f32
    } else {
        // IP: rank by ⟨q,x⟩_est  (1/d scale is constant → omit)
        ip_raw * norm_db
    }
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

fn read_8bytes<R: Read>(r: &mut R) -> std::io::Result<[u8; 8]> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::simd;

    fn random_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
    }

    #[test]
    fn test_fwht_roundtrip() {
        // FWHT is its own inverse (up to a scale factor of n)
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = data.clone();
        fwht(&mut data);
        fwht(&mut data);
        let n = original.len() as f32;
        for (a, b) in original.iter().zip(data.iter()) {
            assert!((a * n - b).abs() < 1e-4, "{} vs {}", a * n, b);
        }
    }

    #[test]
    fn test_byte_lut_correctness() {
        let q_rot = vec![0.5f32, -0.3, 1.2, -0.1, 0.8, 0.0, -0.5, 0.2];
        let lut = build_byte_lut(&q_rot, 8);
        // lut[0 * 256 + 0b00000001] should equal q_rot[0] = 0.5
        assert!((lut[1] - 0.5f32).abs() < 1e-6);
        // lut[0 * 256 + 0b00000011] should equal q_rot[0] + q_rot[1] = 0.5 - 0.3 = 0.2
        assert!((lut[3] - 0.2f32).abs() < 1e-5);
    }

    #[test]
    fn test_rabitq_build_search_ip() {
        let n = 1000;
        let dim = 64;
        let data = random_data(n, dim, 10);
        let idx = RaBitQIndex::build(&data, n, dim);
        assert_eq!(idx.len(), n);

        let query = random_data(1, dim, 99);
        let (ids, _dists) = idx.search(&query, 10, &data, DistanceMetric::InnerProduct, 40);
        assert_eq!(ids.len(), 10);

        // The exact best should appear in top-10 with high probability
        let brute_best = (0..n)
            .map(|i| simd::inner_product_f32(&query, &data[i * dim..(i + 1) * dim]))
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            ids.contains(&(brute_best as u32)),
            "exact best {} not in top-10: {:?}", brute_best, ids
        );
    }

    #[test]
    fn test_rabitq_build_search_l2() {
        let n = 500;
        let dim = 32;
        let data = random_data(n, dim, 20);
        let idx = RaBitQIndex::build(&data, n, dim);

        let query = random_data(1, dim, 88);
        let (ids, _) = idx.search(&query, 5, &data, DistanceMetric::L2Squared, 40);
        assert_eq!(ids.len(), 5);

        let brute_best = (0..n)
            .map(|i| simd::l2_squared_f32(&query, &data[i * dim..(i + 1) * dim]))
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        assert!(ids.contains(&(brute_best as u32)));
    }

    #[test]
    fn test_rabitq_non_power_of_two_dim() {
        let n = 200;
        let dim = 100; // not power of 2
        let data = random_data(n, dim, 7);
        let idx = RaBitQIndex::build(&data, n, dim);
        assert_eq!(idx.padded_dim, 128); // next power of 2

        let query = random_data(1, dim, 55);
        let (ids, _) = idx.search(&query, 5, &data, DistanceMetric::InnerProduct, 40);
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn test_rabitq_save_load() {
        let n = 200;
        let dim = 32;
        let data = random_data(n, dim, 42);
        let idx = RaBitQIndex::build(&data, n, dim);

        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("rabitq_index.bin");
        idx.save(&path).unwrap();

        let idx2 = RaBitQIndex::load(&path).unwrap();
        assert_eq!(idx2.len(), n);
        assert_eq!(idx2.dim, dim);
        assert_eq!(idx2.padded_dim, idx.padded_dim);
        assert_eq!(idx2.codes, idx.codes);
    }
}
