//! PolarVec index: training-free multi-bit quantization with Randomized Hadamard Transform.
//!
//! Algorithm:
//! 1. **Build**: Apply Randomized Hadamard Transform (RHT) to each vector (same as RaBitQ
//!    for isotropy) → compute per-dimension min/max of the rotated data → uniformly quantize
//!    each dimension to `b` bits (default b=4, 16 levels) → pack codes into bytes.
//! 2. **Search**: Rotate query with same RHT → build per-dimension LUT (dim × 2^b entries,
//!    ≤8 KB for typical dims, fits in L1) → scan packed codes with LUT lookup for coarse
//!    score → re-rank top-N with exact f32.
//!
//! Key properties:
//! - **Training-free**: no k-means, no codebook training — only per-dim min/max from data
//! - **~3-4 bits/dim**: b=4 → 8× compression vs f32; b=3 → 10.7× compression
//! - **Isotropic quantization**: RHT decorrelates dimensions → uniform quantization is near-optimal
//! - **LUT-based scan**: dim × 2^b per-dim lookup table fits in L1 cache
//! - **Two-pass**: LUT approx scan → exact f32 re-rank for high recall
//!
//! Comparison:
//! - vs **RaBitQ** (1 bit/dim): 3-4× more bits → much more accurate coarse distance estimates
//! - vs **SQ8** (8 bits/dim): 2-3× less memory with comparable recall after re-rank
//! - vs **PQ**: no training, no codebook storage, supports dynamic insertion
//!
//! Randomized Hadamard Transform (shared logic with RaBitQ):
//! 1. Pad vector to `padded_dim = next_power_of_two(dim)` (zero-pad)
//! 2. Apply random signs: `x[i] *= sign_i` (sign_i ∈ {±1}, fixed seed=42)
//! 3. Apply Fast Walsh-Hadamard Transform (FWHT): O(D log D)
//!
//! Binary format (polarvec_index.bin):
//! v1: `[magic u32][version=1 u32][dim u32][padded_dim u32][bits u32][n_vectors u64]`
//!     `[n_sign_words u32][sign_words ×u64][dim_mins ×f32][dim_scales ×f32]`
//!     `[codes n×bytes_per_vec u8][norms n×f32]`
//! v2: same as v1 + `[residual_sq n×f32]`  (||v_rot − vhat_rot||² per vector)
//! v3: same as v2 + `[residual_signs n×u32]`
//! v4: same as v3 + `[inv_v_hat_norms n×f32]`
//! v5: `[magic][version=5][dim][padded_dim][bits][n_vectors][flags u32]`
//!     `[n_sign_words][sign_words][dim_mins][dim_scales][codes]`
//!     plus `[residual_sq]` when `flags & 1 != 0`
//!     plus `[inv_v_hat_norms]` when `flags & 2 != 0`.
//!
//! L2 distance correction (v2+/v5 flag): coarse score = LUT + residual_sq[i]
//!   LUT ~= ||q_rot - vhat_rot||²,  residual_sq[i] = ||e_rot||² = ||v_rot - vhat_rot||²
//!   Together they approximate ||q_rot − v_rot||² minus the cross-term,
//!   which has near-zero mean (RHT isotropy) → better candidate ranking.

use crate::distance::DistanceMetric;
use crate::storage::pq_mmap::rescore_exact;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const POLARVEC_MAGIC: u32 = 0x504C_5651; // "PLVQ"
const POLARVEC_VERSION: u32 = 5;
const FLAG_RESIDUAL_SQ: u32 = 1;
const FLAG_INV_V_HAT_NORMS: u32 = 2;
/// Default oversample multiplier for two-pass re-ranking.
pub const DEFAULT_OVERSAMPLE: usize = 20;
/// Default bits per dimension.
pub const DEFAULT_BITS: usize = 4;
/// Seed for the RHT sign vector (same as RaBitQ for consistency).
const SIGN_SEED: u64 = 42;

// ─── PolarVec Index ────────────────────────────────────────────────────────────

/// PolarVec index: training-free multi-bit quantization after Randomized Hadamard Transform.
///
/// Each vector is stored as `bytes_per_vec = ceil(padded_dim × bits / 8)` packed bytes
/// plus its L2 norm. Distance estimation uses a per-dimension LUT of size `dim × 2^bits`
/// (≤8KB for dim=128, bits=4) that fits in L1 cache.
pub struct PolarVecIndex {
    /// Original vector dimension.
    dim: usize,
    /// Padded dimension (= next power of two ≥ dim).
    padded_dim: usize,
    /// Bits per dimension (2, 3, 4, or 8). Default: 4.
    bits: usize,
    /// Packed bytes per vector = ceil(padded_dim * bits / 8).
    bytes_per_vec: usize,
    /// Quantization levels = 2^bits.
    n_levels: usize,
    /// Random sign vector for RHT (packed as u64 words). Length: ceil(padded_dim / 64).
    sign_words: Vec<u64>,
    /// Per-dimension minimum of rotated data: `[padded_dim]` f32.
    dim_mins: Vec<f32>,
    /// Per-dimension quantization scale: `(max - min) / (n_levels - 1)`, or 1.0 if constant.
    dim_scales: Vec<f32>,
    /// Packed b-bit codes: `[n_vectors × bytes_per_vec]` u8 flat array.
    codes: Vec<u8>,
    /// Per-vector quantization residual squared norm in rotated space:
    /// `residual_sq[i] = ||v_rot_i − v̂_rot_i||²`.
    /// Added to the LUT score for L2 metric to correct for quantization bias:
    /// `LUT[i] + residual_sq[i] ≈ ||q_rot − v_rot_i||² − cross_term`.
    /// Empty for IP/Cosine indices and for v1 files (no correction applied).
    residual_sq: Vec<f32>,
    /// Reciprocal of ||v_hat_rot|| for cosine coarse scoring. Empty for IP/L2
    /// v5 indices; older files may populate it for compatibility.
    inv_v_hat_norms: Vec<f32>,
    n_vectors: usize,
}

impl PolarVecIndex {
    /// Build a PolarVec index from raw f32 data.
    ///
    /// `bits` controls compression: 4 (default, 8× vs f32), 3 (10.7×), 8 (4× like SQ8).
    pub fn build(data: &[f32], n_vectors: usize, dim: usize, bits: usize) -> Self {
        Self::build_for_metric(data, n_vectors, dim, bits, DistanceMetric::L2Squared)
    }

    /// Build a PolarVec index with metric-aware auxiliary storage.
    ///
    /// IP/Cosine do not need per-vector correction arrays because the final exact
    /// re-rank receives the original f32 data. L2 keeps one residual-squared
    /// value per vector because it improves coarse candidate ordering cheaply.
    pub fn build_for_metric(
        data: &[f32],
        n_vectors: usize,
        dim: usize,
        bits: usize,
        metric: DistanceMetric,
    ) -> Self {
        assert!(n_vectors > 0, "need at least one vector");
        assert!(dim > 0, "dimension must be positive");
        assert!(bits >= 1 && bits <= 8, "bits must be in [1, 8]");

        let padded_dim = dim.next_power_of_two();
        let n_levels = 1usize << bits;
        let bytes_per_vec = compute_bytes_per_vec(padded_dim, bits);
        let sign_words_len = (padded_dim + 63) / 64;
        let sign_words = generate_sign_words(sign_words_len, SIGN_SEED);
        let store_residual_sq = metric == DistanceMetric::L2Squared;
        let store_inv_v_hat_norms = metric == DistanceMetric::Cosine;

        // Pass 1: compute per-dim min/max using a subsample (up to 50K vectors)
        let n_sample = n_vectors.min(50_000);
        let stride = if n_vectors <= 50_000 {
            1
        } else {
            n_vectors / n_sample
        };
        let stat_chunk = 1024usize;
        let n_stat_chunks = (n_sample + stat_chunk - 1) / stat_chunk;

        let local_stats: Vec<(Vec<f32>, Vec<f32>)> = (0..n_stat_chunks)
            .into_par_iter()
            .map(|chunk| {
                let start = chunk * stat_chunk;
                let end = (start + stat_chunk).min(n_sample);

                let mut local_mins = vec![f32::INFINITY; padded_dim];
                let mut local_maxs = vec![f32::NEG_INFINITY; padded_dim];
                let mut buf = vec![0.0f32; padded_dim];

                for si in start..end {
                    let vi = si * stride;
                    let src = &data[vi * dim..(vi + 1) * dim];
                    buf[..dim].copy_from_slice(src);
                    for j in dim..padded_dim {
                        buf[j] = 0.0;
                    }
                    apply_signs(&mut buf, &sign_words, padded_dim);
                    fwht(&mut buf[..padded_dim]);
                    for d in 0..padded_dim {
                        if buf[d] < local_mins[d] {
                            local_mins[d] = buf[d];
                        }
                        if buf[d] > local_maxs[d] {
                            local_maxs[d] = buf[d];
                        }
                    }
                }
                (local_mins, local_maxs)
            })
            .collect();

        // Merge per-thread stats
        let mut dim_mins_raw = vec![f32::INFINITY; padded_dim];
        let mut dim_maxs_raw = vec![f32::NEG_INFINITY; padded_dim];
        for (lmin, lmax) in local_stats {
            for d in 0..padded_dim {
                if lmin[d] < dim_mins_raw[d] {
                    dim_mins_raw[d] = lmin[d];
                }
                if lmax[d] > dim_maxs_raw[d] {
                    dim_maxs_raw[d] = lmax[d];
                }
            }
        }

        // Compute quantization parameters with a small margin to reduce clipping
        let margin = 0.05f32;
        let mut dim_mins = vec![0.0f32; padded_dim];
        let mut dim_scales = vec![1.0f32; padded_dim];
        for d in 0..padded_dim {
            let raw_min = dim_mins_raw[d];
            let raw_max = dim_maxs_raw[d];
            let range = raw_max - raw_min;
            if range > 1e-8 {
                dim_mins[d] = raw_min - margin * range;
                dim_scales[d] = range * (1.0 + 2.0 * margin) / (n_levels - 1) as f32;
            } else {
                dim_mins[d] = raw_min;
                dim_scales[d] = 1.0;
            }
        }

        // Pass 2: encode all vectors in parallel
        let enc_chunk = 4096usize;

        let mut codes = vec![0u8; n_vectors * bytes_per_vec];
        let mut residual_sq = if store_residual_sq {
            vec![0.0f32; n_vectors]
        } else {
            Vec::new()
        };
        let mut inv_v_hat_norms = if store_inv_v_hat_norms {
            vec![0.0f32; n_vectors]
        } else {
            Vec::new()
        };

        encode_all_vectors_in_place(
            data,
            n_vectors,
            dim,
            padded_dim,
            bits,
            n_levels,
            bytes_per_vec,
            &sign_words,
            &dim_mins,
            &dim_scales,
            enc_chunk,
            &mut codes,
            if store_residual_sq {
                Some(&mut residual_sq)
            } else {
                None
            },
            if store_inv_v_hat_norms {
                Some(&mut inv_v_hat_norms)
            } else {
                None
            },
        );

        PolarVecIndex {
            dim,
            padded_dim,
            bits,
            bytes_per_vec,
            n_levels,
            sign_words,
            dim_mins,
            dim_scales,
            codes,
            residual_sq,
            inv_v_hat_norms,
            n_vectors,
        }
    }

    /// LUT-based two-pass search.
    ///
    /// Pass 1: rotate query → build per-dim LUT (dim × 2^b, ≤8 KB) → parallel scan → top-N.
    /// Pass 2: exact f32 re-score on top-N candidates → top-k.
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

        // Rotate query
        let mut q_rot = vec![0.0f32; self.padded_dim];
        if metric == DistanceMetric::Cosine {
            let q_norm = query.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
            for d in 0..self.dim {
                q_rot[d] = query[d] / q_norm;
            }
        } else {
            q_rot[..self.dim].copy_from_slice(query);
        }
        apply_signs(&mut q_rot, &self.sign_words, self.padded_dim);
        fwht(&mut q_rot[..self.padded_dim]);

        // Build per-dimension LUT (padded_dim × n_levels floats)
        let lut = build_lut(
            &q_rot,
            self.padded_dim,
            self.bits,
            self.n_levels,
            &self.dim_mins,
            &self.dim_scales,
            metric,
        );

        // For 4-bit codes: build byte-combined LUT (bytes_per_vec × 256).
        // Each entry covers 2 nibbles → halves the number of LUT lookups per vector.
        let byte_lut_4bit: Vec<f32> = if self.bits == 4 {
            build_byte_lut_4bit(&lut, self.padded_dim, self.bytes_per_vec)
        } else {
            Vec::new()
        };

        // Pass 1: parallel LUT coarse scan
        let ascending = metric.is_ascending();
        let correction = if metric == DistanceMetric::L2Squared {
            &self.residual_sq[..]
        } else {
            &[][..]
        };
        let multiplier = if metric == DistanceMetric::Cosine {
            &self.inv_v_hat_norms[..]
        } else {
            &[][..]
        };

        let candidates: Vec<u32> = scan_topn(
            &self.codes,
            self.n_vectors,
            self.bytes_per_vec,
            self.bits,
            self.n_levels,
            &lut,
            &byte_lut_4bit,
            n_candidates,
            ascending,
            correction,
            multiplier,
        )
        .into_iter()
        .map(|(_, idx)| idx)
        .collect();

        // Pass 2: exact f32 re-rank
        rescore_exact(&candidates, query, f32_data, self.dim, k, metric)
    }

    /// Number of indexed vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    /// Save the PolarVec index to `path`.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        w.write_all(&POLARVEC_MAGIC.to_le_bytes())?;
        w.write_all(&POLARVEC_VERSION.to_le_bytes())?;
        w.write_all(&(self.dim as u32).to_le_bytes())?;
        w.write_all(&(self.padded_dim as u32).to_le_bytes())?;
        w.write_all(&(self.bits as u32).to_le_bytes())?;
        w.write_all(&(self.n_vectors as u64).to_le_bytes())?;
        let mut flags = 0u32;
        if !self.residual_sq.is_empty() {
            flags |= FLAG_RESIDUAL_SQ;
        }
        if !self.inv_v_hat_norms.is_empty() {
            flags |= FLAG_INV_V_HAT_NORMS;
        }
        w.write_all(&flags.to_le_bytes())?;

        // Sign words
        w.write_all(&(self.sign_words.len() as u32).to_le_bytes())?;
        for &word in &self.sign_words {
            w.write_all(&word.to_le_bytes())?;
        }

        // Quantization parameters (padded_dim × 2 f32 arrays)
        for &v in &self.dim_mins {
            w.write_all(&v.to_le_bytes())?;
        }
        for &v in &self.dim_scales {
            w.write_all(&v.to_le_bytes())?;
        }

        // Codes (raw u8)
        w.write_all(&self.codes)?;

        // v5 optional: residual squared norms (f32 LE)
        for &r in &self.residual_sq {
            w.write_all(&r.to_le_bytes())?;
        }
        for &v in &self.inv_v_hat_norms {
            w.write_all(&v.to_le_bytes())?;
        }

        w.flush()?;
        Ok(())
    }

    /// Load a PolarVec index from `path`.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);

        let magic = read_u32le(&mut r)?;
        if magic != POLARVEC_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid PolarVec magic bytes",
            ));
        }
        let version = read_u32le(&mut r)?;
        if !(1..=POLARVEC_VERSION).contains(&version) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported PolarVec version: {}", version),
            ));
        }
        let dim = read_u32le(&mut r)? as usize;
        let padded_dim = read_u32le(&mut r)? as usize;
        let bits = read_u32le(&mut r)? as usize;
        let n_vectors = read_u64le(&mut r)? as usize;
        let n_levels = 1usize << bits;
        let bytes_per_vec = compute_bytes_per_vec(padded_dim, bits);
        let flags = if version >= 5 { read_u32le(&mut r)? } else { 0 };

        let n_sign_words = read_u32le(&mut r)? as usize;
        let mut sign_words = vec![0u64; n_sign_words];
        for word in sign_words.iter_mut() {
            let bytes = read_8bytes(&mut r)?;
            *word = u64::from_le_bytes(bytes);
        }

        let mut dim_mins = vec![0.0f32; padded_dim];
        for v in dim_mins.iter_mut() {
            let b = read_4bytes(&mut r)?;
            *v = f32::from_le_bytes(b);
        }
        let mut dim_scales = vec![0.0f32; padded_dim];
        for v in dim_scales.iter_mut() {
            let b = read_4bytes(&mut r)?;
            *v = f32::from_le_bytes(b);
        }

        let mut codes = vec![0u8; n_vectors * bytes_per_vec];
        r.read_exact(&mut codes)?;

        // v1-v4 stored norms even though PolarVec no longer needs them.
        if version <= 4 {
            for _ in 0..n_vectors {
                let _ = read_4bytes(&mut r)?;
            }
        }

        let has_residual_sq = if version >= 5 {
            flags & FLAG_RESIDUAL_SQ != 0
        } else {
            version >= 2
        };
        let residual_sq = if has_residual_sq {
            let mut rs = vec![0.0f32; n_vectors];
            for r_val in rs.iter_mut() {
                let b = read_4bytes(&mut r)?;
                *r_val = f32::from_le_bytes(b);
            }
            rs
        } else {
            Vec::new()
        };
        // v3/v4 stored residual sign sketches; read and discard for compatibility.
        if (3..=4).contains(&version) {
            for _ in 0..n_vectors {
                let _ = read_u32le(&mut r)?;
            }
        }
        let has_inv_v_hat_norms = if version >= 5 {
            flags & FLAG_INV_V_HAT_NORMS != 0
        } else {
            version == 4
        };
        let inv_v_hat_norms = if has_inv_v_hat_norms {
            let mut inv = vec![0.0f32; n_vectors];
            for v in inv.iter_mut() {
                let b = read_4bytes(&mut r)?;
                *v = f32::from_le_bytes(b);
            }
            inv
        } else if version < 4 {
            recompute_inv_v_hat_norms(
                &codes,
                n_vectors,
                bytes_per_vec,
                padded_dim,
                bits,
                &dim_mins,
                &dim_scales,
            )
        } else {
            Vec::new()
        };

        Ok(PolarVecIndex {
            dim,
            padded_dim,
            bits,
            bytes_per_vec,
            n_levels,
            sign_words,
            dim_mins,
            dim_scales,
            codes,
            residual_sq,
            inv_v_hat_norms,
            n_vectors,
        })
    }
}

fn encode_all_vectors_in_place(
    data: &[f32],
    n_vectors: usize,
    dim: usize,
    padded_dim: usize,
    bits: usize,
    n_levels: usize,
    bytes_per_vec: usize,
    sign_words: &[u64],
    dim_mins: &[f32],
    dim_scales: &[f32],
    enc_chunk: usize,
    codes: &mut [u8],
    residual_sq: Option<&mut [f32]>,
    inv_v_hat_norms: Option<&mut [f32]>,
) {
    debug_assert_eq!(codes.len(), n_vectors * bytes_per_vec);
    let chunk_bytes = enc_chunk * bytes_per_vec;

    match (residual_sq, inv_v_hat_norms) {
        (Some(residual_sq), Some(inv_v_hat_norms)) => {
            debug_assert_eq!(residual_sq.len(), n_vectors);
            debug_assert_eq!(inv_v_hat_norms.len(), n_vectors);
            codes
                .par_chunks_mut(chunk_bytes)
                .zip(residual_sq.par_chunks_mut(enc_chunk))
                .zip(inv_v_hat_norms.par_chunks_mut(enc_chunk))
                .enumerate()
                .for_each(|(chunk, ((code_chunk, residual_chunk), inv_chunk))| {
                    encode_vector_chunk_in_place(
                        data,
                        dim,
                        padded_dim,
                        bits,
                        n_levels,
                        bytes_per_vec,
                        sign_words,
                        dim_mins,
                        dim_scales,
                        chunk * enc_chunk,
                        code_chunk,
                        Some(residual_chunk),
                        Some(inv_chunk),
                    );
                });
        }
        (Some(residual_sq), None) => {
            debug_assert_eq!(residual_sq.len(), n_vectors);
            codes
                .par_chunks_mut(chunk_bytes)
                .zip(residual_sq.par_chunks_mut(enc_chunk))
                .enumerate()
                .for_each(|(chunk, (code_chunk, residual_chunk))| {
                    encode_vector_chunk_in_place(
                        data,
                        dim,
                        padded_dim,
                        bits,
                        n_levels,
                        bytes_per_vec,
                        sign_words,
                        dim_mins,
                        dim_scales,
                        chunk * enc_chunk,
                        code_chunk,
                        Some(residual_chunk),
                        None,
                    );
                });
        }
        (None, Some(inv_v_hat_norms)) => {
            debug_assert_eq!(inv_v_hat_norms.len(), n_vectors);
            codes
                .par_chunks_mut(chunk_bytes)
                .zip(inv_v_hat_norms.par_chunks_mut(enc_chunk))
                .enumerate()
                .for_each(|(chunk, (code_chunk, inv_chunk))| {
                    encode_vector_chunk_in_place(
                        data,
                        dim,
                        padded_dim,
                        bits,
                        n_levels,
                        bytes_per_vec,
                        sign_words,
                        dim_mins,
                        dim_scales,
                        chunk * enc_chunk,
                        code_chunk,
                        None,
                        Some(inv_chunk),
                    );
                });
        }
        (None, None) => {
            codes
                .par_chunks_mut(chunk_bytes)
                .enumerate()
                .for_each(|(chunk, code_chunk)| {
                    encode_vector_chunk_in_place(
                        data,
                        dim,
                        padded_dim,
                        bits,
                        n_levels,
                        bytes_per_vec,
                        sign_words,
                        dim_mins,
                        dim_scales,
                        chunk * enc_chunk,
                        code_chunk,
                        None,
                        None,
                    );
                });
        }
    }
}

fn encode_vector_chunk_in_place(
    data: &[f32],
    dim: usize,
    padded_dim: usize,
    bits: usize,
    n_levels: usize,
    bytes_per_vec: usize,
    sign_words: &[u64],
    dim_mins: &[f32],
    dim_scales: &[f32],
    start: usize,
    code_chunk: &mut [u8],
    mut residual_sq: Option<&mut [f32]>,
    mut inv_v_hat_norms: Option<&mut [f32]>,
) {
    let n_in = code_chunk.len() / bytes_per_vec;
    debug_assert_eq!(code_chunk.len(), n_in * bytes_per_vec);
    debug_assert!(residual_sq
        .as_ref()
        .map(|values| values.len() == n_in)
        .unwrap_or(true));
    debug_assert!(inv_v_hat_norms
        .as_ref()
        .map(|values| values.len() == n_in)
        .unwrap_or(true));

    let n_levels_m1 = (n_levels - 1) as f32;
    let store_residual_sq = residual_sq.is_some();
    let store_inv_v_hat_norms = inv_v_hat_norms.is_some();
    let mut buf = vec![0.0f32; padded_dim];

    for i in 0..n_in {
        let vi = start + i;
        let src = &data[vi * dim..(vi + 1) * dim];

        buf[..dim].copy_from_slice(src);
        for v in &mut buf[dim..padded_dim] {
            *v = 0.0;
        }
        apply_signs(&mut buf, sign_words, padded_dim);
        fwht(&mut buf[..padded_dim]);

        let code_base = i * bytes_per_vec;
        let code_row = &mut code_chunk[code_base..code_base + bytes_per_vec];
        let mut res_sq = 0.0f32;
        let mut v_hat_sq = 0.0f32;

        for d in 0..padded_dim {
            let val = unsafe { *buf.get_unchecked(d) };
            let raw_code = (val - unsafe { *dim_mins.get_unchecked(d) })
                / unsafe { *dim_scales.get_unchecked(d) };
            let code = raw_code.round().max(0.0).min(n_levels_m1) as u8;
            pack_code_at_row(code_row, d, code, bits);

            if store_residual_sq || store_inv_v_hat_norms {
                let v_hat_d = unsafe { *dim_mins.get_unchecked(d) }
                    + code as f32 * unsafe { *dim_scales.get_unchecked(d) };
                if store_residual_sq {
                    let diff = val - v_hat_d;
                    res_sq += diff * diff;
                }
                if store_inv_v_hat_norms {
                    v_hat_sq += v_hat_d * v_hat_d;
                }
            }
        }

        if let Some(values) = residual_sq.as_deref_mut() {
            values[i] = res_sq;
        }
        if let Some(values) = inv_v_hat_norms.as_deref_mut() {
            values[i] = 1.0 / v_hat_sq.sqrt().max(1e-12);
        }
    }
}

// ─── Bit packing helpers ──────────────────────────────────────────────────────

/// Compute packed bytes per vector: ceil(padded_dim × bits / 8).
#[inline]
pub fn compute_bytes_per_vec(padded_dim: usize, bits: usize) -> usize {
    (padded_dim * bits + 7) / 8
}

/// Pack a b-bit `code` into `codes` at vector base `vec_base`, dimension index `d`.
///
/// Bits are packed sequentially: dimension d occupies bits [d×bits, (d+1)×bits) in the
/// byte array starting at `vec_base`. Spans two bytes when necessary.
#[inline]
fn pack_code_at(codes: &mut [u8], vec_base: usize, d: usize, code: u8, bits: usize) {
    let bit_start = d * bits;
    let byte_idx = vec_base + bit_start / 8;
    let bit_off = bit_start % 8;
    let available = 8 - bit_off;

    if bits <= available {
        let mask = ((1u16 << bits) - 1) as u8;
        let shifted_mask = mask << bit_off;
        codes[byte_idx] = (codes[byte_idx] & !shifted_mask) | ((code & mask) << bit_off);
    } else {
        // Code spans two bytes
        let lo_bits = available;
        let hi_bits = bits - lo_bits;
        // Lower bits go into [bit_off, 7] of byte_idx
        let lo_mask: u8 = !((1u8 << bit_off).wrapping_sub(1)) & 0xff;
        let lo_keep: u8 = codes[byte_idx] & !lo_mask;
        codes[byte_idx] = lo_keep | (code << bit_off);
        // Upper bits go into [0, hi_bits-1] of byte_idx+1
        let hi_mask = (1u8 << hi_bits) - 1;
        codes[byte_idx + 1] = (codes[byte_idx + 1] & !hi_mask) | (code >> lo_bits);
    }
}

#[inline(always)]
fn pack_code_at_row(row: &mut [u8], d: usize, code: u8, bits: usize) {
    match bits {
        4 => {
            let byte_idx = d >> 1;
            let code = code & 0x0f;
            let byte = unsafe { row.get_unchecked_mut(byte_idx) };
            if d & 1 == 0 {
                *byte = (*byte & 0xf0) | code;
            } else {
                *byte = (*byte & 0x0f) | (code << 4);
            }
        }
        8 => unsafe {
            *row.get_unchecked_mut(d) = code;
        },
        _ => pack_code_at(row, 0, d, code, bits),
    }
}

/// Unpack the b-bit code for dimension `d` from vector at `vec_base` in `codes`.
#[inline(always)]
fn unpack_code_at(codes: &[u8], vec_base: usize, d: usize, bits: usize) -> usize {
    let bit_start = d * bits;
    let byte_idx = vec_base + bit_start / 8;
    let bit_off = bit_start % 8;
    let available = 8 - bit_off;
    let mask = (1usize << bits) - 1;

    if bits <= available {
        (unsafe { *codes.get_unchecked(byte_idx) } as usize >> bit_off) & mask
    } else {
        let lo = unsafe { *codes.get_unchecked(byte_idx) } as usize >> bit_off;
        let hi = unsafe { *codes.get_unchecked(byte_idx + 1) } as usize;
        (lo | (hi << available)) & mask
    }
}

fn recompute_inv_v_hat_norms(
    codes: &[u8],
    n_vectors: usize,
    bytes_per_vec: usize,
    padded_dim: usize,
    bits: usize,
    dim_mins: &[f32],
    dim_scales: &[f32],
) -> Vec<f32> {
    (0..n_vectors)
        .map(|vi| {
            let vec_base = vi * bytes_per_vec;
            let mut v_hat_sq = 0.0f32;
            for d in 0..padded_dim {
                let code = unpack_code_at(codes, vec_base, d, bits);
                let v_hat_d = dim_mins[d] + code as f32 * dim_scales[d];
                v_hat_sq += v_hat_d * v_hat_d;
            }
            1.0 / v_hat_sq.sqrt().max(1e-12)
        })
        .collect()
}

// ─── Randomized Hadamard Transform helpers ────────────────────────────────────

fn generate_sign_words(n_words: usize, seed: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n_words).map(|_| rng.gen::<u64>()).collect()
}

#[inline]
fn apply_signs(buf: &mut [f32], sign_words: &[u64], n: usize) {
    for (word_idx, chunk) in buf[..n].chunks_mut(64).enumerate() {
        let mut word = sign_words.get(word_idx).copied().unwrap_or(0);
        for v in chunk {
            if word & 1 == 1 {
                *v = -*v;
            }
            word >>= 1;
        }
    }
}

fn fwht(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1usize;
    while h < n {
        let step = h * 2;
        let mut i = 0;
        while i < n {
            for j in 0..h {
                let x = data[i + j];
                let y = data[i + j + h];
                data[i + j] = x + y;
                data[i + j + h] = x - y;
            }
            i += step;
        }
        h = step;
    }
}

// ─── LUT construction ─────────────────────────────────────────────────────────

/// Build the per-dimension lookup table for the rotated query.
///
/// `lut[d × n_levels + c]` = contribution of code `c` at dimension `d` to the total score.
///
/// - **IP** (ascending=false):  `lut[d][c] = q_rot[d] × dequant(c, d)` → maximize
/// - **L2** (ascending=true):   `lut[d][c] = (q_rot[d] − dequant(c, d))²` → minimize
/// - **Cosine** (ascending=true): `lut[d][c] = −q_rot[d] × dequant(c, d)` → minimize
fn build_lut(
    q_rot: &[f32],
    padded_dim: usize,
    _bits: usize,
    n_levels: usize,
    dim_mins: &[f32],
    dim_scales: &[f32],
    metric: DistanceMetric,
) -> Vec<f32> {
    let mut lut = vec![0.0f32; padded_dim * n_levels];

    for d in 0..padded_dim {
        let q_d = q_rot[d];
        let min_d = dim_mins[d];
        let scale_d = dim_scales[d];
        let base = d * n_levels;

        match metric {
            DistanceMetric::InnerProduct => {
                for c in 0..n_levels {
                    let v_d = min_d + c as f32 * scale_d;
                    lut[base + c] = q_d * v_d;
                }
            }
            DistanceMetric::L2Squared => {
                for c in 0..n_levels {
                    let diff = q_d - (min_d + c as f32 * scale_d);
                    lut[base + c] = diff * diff;
                }
            }
            DistanceMetric::Cosine => {
                // Use negative IP as proxy; exact cosine handled by re-rank
                for c in 0..n_levels {
                    let v_d = min_d + c as f32 * scale_d;
                    lut[base + c] = -(q_d * v_d);
                }
            }
            _ => {
                for c in 0..n_levels {
                    let diff = q_d - (min_d + c as f32 * scale_d);
                    lut[base + c] = diff * diff;
                }
            }
        }
    }
    lut
}

// ─── LUT scan ─────────────────────────────────────────────────────────────────

/// Build a byte-combined LUT for bits=4.
///
/// `byte_lut[b × 256 + byte_val]` = score contribution of byte `b` (which stores
/// two 4-bit codes: low nibble → dim `2b`, high nibble → dim `2b+1`).
/// Size: `bytes_per_vec × 256` f32 entries (64 × 256 = 16,384 entries = 64 KB for dim=128).
/// Halves the number of LUT accesses per vector vs the per-nibble path.
#[inline]
fn build_byte_lut_4bit(lut: &[f32], padded_dim: usize, bytes_per_vec: usize) -> Vec<f32> {
    let mut byte_lut = vec![0.0f32; bytes_per_vec * 256];
    for b in 0..bytes_per_vec {
        let even_dim = b * 2;
        let odd_dim = even_dim + 1;
        let base_even = even_dim * 16; // lut[d_even * 16 ..]
        let base_odd = odd_dim * 16; // lut[d_odd  * 16 ..]
        let out_base = b * 256;
        for v in 0..256usize {
            let lo = v & 0x0f;
            let hi = v >> 4;
            let mut score = unsafe { *lut.get_unchecked(base_even + lo) };
            if odd_dim < padded_dim {
                score += unsafe { *lut.get_unchecked(base_odd + hi) };
            }
            byte_lut[out_base + v] = score;
        }
    }
    byte_lut
}

/// Compute the LUT-based score for one packed code row.
///
/// - bits=4 with non-empty `byte_lut`: byte-level lookups (fastest path).
/// - bits=4 fallback / bits=8: per-code lookups.
/// - other bits: general bit-unpack path.
#[inline(always)]
fn compute_lut_score_row(
    code_row: &[u8],
    bits: usize,
    lut: &[f32],
    n_levels: usize,
    byte_lut: &[f32],
) -> f32 {
    let mut score = 0.0f32;

    if bits == 4 && !byte_lut.is_empty() {
        // Byte-combined fast path: 64 lookups instead of 128, with 4-accumulator
        // unrolling to break the serial FP-add dependency chain.
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let full = (code_row.len() / 4) * 4;
        let mut b = 0usize;
        while b < full {
            let v0 = unsafe { *code_row.get_unchecked(b) } as usize;
            let v1 = unsafe { *code_row.get_unchecked(b + 1) } as usize;
            let v2 = unsafe { *code_row.get_unchecked(b + 2) } as usize;
            let v3 = unsafe { *code_row.get_unchecked(b + 3) } as usize;
            s0 += unsafe { *byte_lut.get_unchecked(b * 256 + v0) };
            s1 += unsafe { *byte_lut.get_unchecked((b + 1) * 256 + v1) };
            s2 += unsafe { *byte_lut.get_unchecked((b + 2) * 256 + v2) };
            s3 += unsafe { *byte_lut.get_unchecked((b + 3) * 256 + v3) };
            b += 4;
        }
        while b < code_row.len() {
            let v = unsafe { *code_row.get_unchecked(b) } as usize;
            s0 += unsafe { *byte_lut.get_unchecked(b * 256 + v) };
            b += 1;
        }
        score = s0 + s1 + s2 + s3;
    } else {
        let padded_dim = lut.len() / n_levels;
        match bits {
            4 => {
                for d in 0..padded_dim {
                    let byte = unsafe { *code_row.get_unchecked(d / 2) };
                    let code = if d & 1 == 0 {
                        (byte & 0x0f) as usize
                    } else {
                        (byte >> 4) as usize
                    };
                    score += unsafe { *lut.get_unchecked(d * n_levels + code) };
                }
            }
            8 => {
                // 4-accumulator unrolling: breaks the serial FP-add dependency chain,
                // allowing the CPU to execute 4 independent load+add streams in parallel.
                let mut s0 = 0.0f32;
                let mut s1 = 0.0f32;
                let mut s2 = 0.0f32;
                let mut s3 = 0.0f32;
                let full = (code_row.len() / 4) * 4;
                let mut b = 0usize;
                while b < full {
                    let c0 = unsafe { *code_row.get_unchecked(b) } as usize;
                    let c1 = unsafe { *code_row.get_unchecked(b + 1) } as usize;
                    let c2 = unsafe { *code_row.get_unchecked(b + 2) } as usize;
                    let c3 = unsafe { *code_row.get_unchecked(b + 3) } as usize;
                    s0 += unsafe { *lut.get_unchecked(b * n_levels + c0) };
                    s1 += unsafe { *lut.get_unchecked((b + 1) * n_levels + c1) };
                    s2 += unsafe { *lut.get_unchecked((b + 2) * n_levels + c2) };
                    s3 += unsafe { *lut.get_unchecked((b + 3) * n_levels + c3) };
                    b += 4;
                }
                while b < code_row.len() {
                    let code = unsafe { *code_row.get_unchecked(b) } as usize;
                    s0 += unsafe { *lut.get_unchecked(b * n_levels + code) };
                    b += 1;
                }
                score = s0 + s1 + s2 + s3;
            }
            3 => {
                // Process 8 dims per 3-byte group: pack 3 bytes into a u32, extract 8×3-bit codes
                // with simple bit-shifts (no branches, no generic unpack_code_at overhead).
                // 4 independent accumulators to hide FP-add latency.
                debug_assert_eq!(
                    padded_dim % 8,
                    0,
                    "padded_dim must be multiple of 8 for 3-bit"
                );
                let mut s0 = 0.0f32;
                let mut s1 = 0.0f32;
                let mut s2 = 0.0f32;
                let mut s3 = 0.0f32;
                let n_groups = padded_dim / 8;
                for g in 0..n_groups {
                    let byte_base = g * 3;
                    let b0 = unsafe { *code_row.get_unchecked(byte_base) } as u32;
                    let b1 = unsafe { *code_row.get_unchecked(byte_base + 1) } as u32;
                    let b2 = unsafe { *code_row.get_unchecked(byte_base + 2) } as u32;
                    let w = b0 | (b1 << 8) | (b2 << 16);
                    let base_d = g * 8;
                    let c0 = (w & 0x7) as usize;
                    let c1 = ((w >> 3) & 0x7) as usize;
                    let c2 = ((w >> 6) & 0x7) as usize;
                    let c3 = ((w >> 9) & 0x7) as usize;
                    let c4 = ((w >> 12) & 0x7) as usize;
                    let c5 = ((w >> 15) & 0x7) as usize;
                    let c6 = ((w >> 18) & 0x7) as usize;
                    let c7 = ((w >> 21) & 0x7) as usize;
                    s0 += unsafe { *lut.get_unchecked((base_d) * n_levels + c0) };
                    s1 += unsafe { *lut.get_unchecked((base_d + 1) * n_levels + c1) };
                    s2 += unsafe { *lut.get_unchecked((base_d + 2) * n_levels + c2) };
                    s3 += unsafe { *lut.get_unchecked((base_d + 3) * n_levels + c3) };
                    s0 += unsafe { *lut.get_unchecked((base_d + 4) * n_levels + c4) };
                    s1 += unsafe { *lut.get_unchecked((base_d + 5) * n_levels + c5) };
                    s2 += unsafe { *lut.get_unchecked((base_d + 6) * n_levels + c6) };
                    s3 += unsafe { *lut.get_unchecked((base_d + 7) * n_levels + c7) };
                }
                score = s0 + s1 + s2 + s3;
            }
            _ => {
                for d in 0..padded_dim {
                    let code = unpack_code_at(code_row, 0, d, bits);
                    score += unsafe { *lut.get_unchecked(d * n_levels + code) };
                }
            }
        }
    }
    score
}

/// Compact heap entry for the LUT scan.
#[derive(Clone, Copy)]
struct ScanEntry {
    score: f32,
    idx: u32,
}
impl PartialEq for ScanEntry {
    fn eq(&self, o: &Self) -> bool {
        self.score == o.score
    }
}
impl Eq for ScanEntry {}
impl PartialOrd for ScanEntry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for ScanEntry {
    fn cmp(&self, o: &Self) -> Ordering {
        self.score.partial_cmp(&o.score).unwrap_or(Ordering::Equal)
    }
}

/// Parallel LUT scan that returns the top-N candidate indices.
///
/// `ascending=true`:  keep N smallest scores (L2 / Cosine, lower = more similar).
/// `ascending=false`: keep N largest scores  (IP, higher = more similar).
/// `byte_lut`: non-empty only for bits=4; enables the fast byte-combined scan path.
/// `correction_scores`: optional per-vector additive correction (pass empty slice to disable).
/// For L2, pass `residual_sq` so that `score = LUT + ||e_rot||²` better approximates true distance.
fn scan_topn(
    codes: &[u8],
    n_vectors: usize,
    bytes_per_vec: usize,
    bits: usize,
    n_levels: usize,
    lut: &[f32],
    byte_lut: &[f32],
    n_candidates: usize,
    ascending: bool,
    correction_scores: &[f32],
    multiplier_scores: &[f32],
) -> Vec<(f32, u32)> {
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_vecs = (n_vectors / n_threads).max(256);
    let n_chunks = (n_vectors + chunk_vecs - 1) / chunk_vecs;

    // Each thread emits (score, idx) pairs from its local heap – scores are
    // preserved so the merge phase can sort without re-running compute_lut_score.
    let chunk_results: Vec<Vec<(f32, u32)>> = (0..n_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * chunk_vecs;
            let end = (start + chunk_vecs).min(n_vectors);
            let code_start = start * bytes_per_vec;
            let code_end = end * bytes_per_vec;
            let code_rows = codes[code_start..code_end].chunks_exact(bytes_per_vec);

            let has_correction = !correction_scores.is_empty();
            let has_multiplier = !multiplier_scores.is_empty();
            if ascending {
                let mut heap: BinaryHeap<ScanEntry> = BinaryHeap::with_capacity(n_candidates + 1);
                for (offset, code_row) in code_rows.enumerate() {
                    let vi = start + offset;
                    let mut score = compute_lut_score_row(code_row, bits, lut, n_levels, byte_lut);
                    if has_correction {
                        score += unsafe { *correction_scores.get_unchecked(vi) };
                    }
                    if has_multiplier {
                        score *= unsafe { *multiplier_scores.get_unchecked(vi) };
                    }
                    let entry = ScanEntry {
                        score,
                        idx: vi as u32,
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
                heap.into_iter().map(|e| (e.score, e.idx)).collect()
            } else {
                let mut heap: BinaryHeap<std::cmp::Reverse<ScanEntry>> =
                    BinaryHeap::with_capacity(n_candidates + 1);
                for (offset, code_row) in code_rows.enumerate() {
                    let vi = start + offset;
                    let mut score = compute_lut_score_row(code_row, bits, lut, n_levels, byte_lut);
                    if has_correction {
                        score += unsafe { *correction_scores.get_unchecked(vi) };
                    }
                    if has_multiplier {
                        score *= unsafe { *multiplier_scores.get_unchecked(vi) };
                    }
                    let entry = ScanEntry {
                        score,
                        idx: vi as u32,
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
                heap.into_iter()
                    .map(|std::cmp::Reverse(e)| (e.score, e.idx))
                    .collect()
            }
        })
        .collect();

    // Merge: flatten stored (score, idx) pairs, sort, take top-N – no LUT recompute.
    let mut all: Vec<(f32, u32)> = chunk_results.into_iter().flatten().collect();
    if ascending {
        all.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    } else {
        all.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    }
    all.truncate(n_candidates);
    all
}

// ─── Parse helper ─────────────────────────────────────────────────────────────

/// Extract `bits` from an index type string like "FLAT-IP-POLARVEC4" or "FLAT-L2-POLARVEC3".
/// Falls back to `default_bits` if no numeric suffix is present or suffix is out of range.
pub fn parse_bits(index_type_upper: &str, default_bits: usize) -> usize {
    if let Some(pos) = index_type_upper.find("POLARVEC") {
        let rest = &index_type_upper[pos + 8..]; // "POLARVEC" is 8 chars
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            if let Ok(n) = digits.parse::<usize>() {
                if n >= 1 && n <= 8 {
                    return n;
                }
            }
        }
    }
    default_bits
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
    use crate::distance::{compute_distance_f32, simd};

    fn random_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
    }

    fn normalize_rows(data: &mut [f32], dim: usize) {
        for row in data.chunks_exact_mut(dim) {
            let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
            for v in row.iter_mut() {
                *v /= norm;
            }
        }
    }

    fn recall_at_k(exact: &[u32], approx: &[u32]) -> f32 {
        let hits = exact.iter().filter(|id| approx.contains(id)).count();
        hits as f32 / exact.len().max(1) as f32
    }

    fn exact_topk(
        data: &[f32],
        dim: usize,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
    ) -> Vec<u32> {
        let mut scores: Vec<(f32, u32)> = (0..data.len() / dim)
            .map(|i| {
                (
                    compute_distance_f32(query, &data[i * dim..(i + 1) * dim], metric),
                    i as u32,
                )
            })
            .collect();
        if metric.is_ascending() {
            scores.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        } else {
            scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        }
        scores.into_iter().take(k).map(|(_, idx)| idx).collect()
    }

    #[test]
    fn test_pack_unpack_bits4() {
        let padded_dim = 8;
        let bits = 4;
        let bytes_per_vec = compute_bytes_per_vec(padded_dim, bits);
        let mut codes = vec![0u8; bytes_per_vec];
        let test_codes = [0u8, 5, 3, 15, 1, 7, 9, 12];
        for d in 0..padded_dim {
            pack_code_at(&mut codes, 0, d, test_codes[d], bits);
        }
        for d in 0..padded_dim {
            let c = unpack_code_at(&codes, 0, d, bits);
            assert_eq!(c, test_codes[d] as usize, "d={}", d);
        }
    }

    #[test]
    fn test_pack_unpack_bits3() {
        let padded_dim = 8; // 8 × 3 = 24 bits = 3 bytes
        let bits = 3;
        let bytes_per_vec = compute_bytes_per_vec(padded_dim, bits);
        assert_eq!(bytes_per_vec, 3);
        let mut codes = vec![0u8; bytes_per_vec];
        let test_codes = [0u8, 1, 2, 3, 4, 5, 6, 7];
        for d in 0..padded_dim {
            pack_code_at(&mut codes, 0, d, test_codes[d], bits);
        }
        for d in 0..padded_dim {
            let c = unpack_code_at(&codes, 0, d, bits);
            assert_eq!(c, test_codes[d] as usize, "d={}", d);
        }
    }

    #[test]
    fn test_polarvec_build_search_ip() {
        let n = 1000;
        let dim = 64;
        let data = random_data(n, dim, 10);
        let idx = PolarVecIndex::build(&data, n, dim, 4);
        assert_eq!(idx.len(), n);

        let query = random_data(1, dim, 99);
        let (ids, _dists) = idx.search(&query, 10, &data, DistanceMetric::InnerProduct, 40);
        assert_eq!(ids.len(), 10);

        // Exact best should appear in top-10 with high probability
        let brute_best = (0..n)
            .map(|i| simd::inner_product_f32(&query, &data[i * dim..(i + 1) * dim]))
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            ids.contains(&(brute_best as u32)),
            "exact best {} not in top-10: {:?}",
            brute_best,
            ids
        );
    }

    #[test]
    fn test_polarvec_build_search_l2() {
        let n = 500;
        let dim = 32;
        let data = random_data(n, dim, 20);
        let idx = PolarVecIndex::build(&data, n, dim, 4);

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
    fn test_polarvec_bits3() {
        let n = 300;
        let dim = 64;
        let data = random_data(n, dim, 33);
        let idx = PolarVecIndex::build(&data, n, dim, 3);
        assert_eq!(idx.bytes_per_vec, (64 * 3 + 7) / 8); // = 24 bytes

        let query = random_data(1, dim, 77);
        let (ids, _) = idx.search(&query, 5, &data, DistanceMetric::InnerProduct, 40);
        assert_eq!(ids.len(), 5);

        let brute_best = (0..n)
            .map(|i| simd::inner_product_f32(&query, &data[i * dim..(i + 1) * dim]))
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        assert!(ids.contains(&(brute_best as u32)));
    }

    #[test]
    fn test_polarvec_bits8() {
        let n = 200;
        let dim = 32;
        let data = random_data(n, dim, 11);
        let idx = PolarVecIndex::build(&data, n, dim, 8);
        assert_eq!(idx.bytes_per_vec, 32); // padded_dim=32, 1 byte/dim

        let query = random_data(1, dim, 55);
        let (ids, _) = idx.search(&query, 5, &data, DistanceMetric::L2Squared, 10);
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn test_polarvec_non_power_of_two_dim() {
        let n = 200;
        let dim = 100; // not power of 2 → padded to 128
        let data = random_data(n, dim, 7);
        let idx = PolarVecIndex::build(&data, n, dim, 4);
        assert_eq!(idx.padded_dim, 128);

        let query = random_data(1, dim, 55);
        let (ids, _) = idx.search(&query, 5, &data, DistanceMetric::InnerProduct, 40);
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn test_polarvec_dim_one_bits4() {
        let n = 128;
        let dim = 1;
        let data = random_data(n, dim, 17);
        let idx = PolarVecIndex::build(&data, n, dim, 4);
        assert_eq!(idx.padded_dim, 1);
        assert_eq!(idx.bytes_per_vec, 1);

        let query = random_data(1, dim, 71);
        let (ids, _) = idx.search(&query, 5, &data, DistanceMetric::InnerProduct, 40);
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn test_polarvec_save_load() {
        let n = 200;
        let dim = 32;
        let bits = 4;
        let data = random_data(n, dim, 42);
        let idx = PolarVecIndex::build(&data, n, dim, bits);

        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("polarvec_index.bin");
        idx.save(&path).unwrap();

        let idx2 = PolarVecIndex::load(&path).unwrap();
        assert_eq!(idx2.len(), n);
        assert_eq!(idx2.dim, dim);
        assert_eq!(idx2.bits, bits);
        assert_eq!(idx2.padded_dim, idx.padded_dim);
        assert_eq!(idx2.codes, idx.codes);
        assert_eq!(idx2.bytes_per_vec, idx.bytes_per_vec);
        assert_eq!(idx2.residual_sq, idx.residual_sq);
        assert_eq!(idx2.inv_v_hat_norms, idx.inv_v_hat_norms);
    }

    #[test]
    fn test_polarvec_metric_aware_aux_storage() {
        let n = 200;
        let dim = 32;
        let bits = 4;
        let data = random_data(n, dim, 43);

        let idx_ip =
            PolarVecIndex::build_for_metric(&data, n, dim, bits, DistanceMetric::InnerProduct);
        assert!(idx_ip.residual_sq.is_empty());
        assert!(idx_ip.inv_v_hat_norms.is_empty());

        let idx_l2 =
            PolarVecIndex::build_for_metric(&data, n, dim, bits, DistanceMetric::L2Squared);
        assert_eq!(idx_l2.residual_sq.len(), n);
        assert!(idx_l2.inv_v_hat_norms.is_empty());

        let idx_cos = PolarVecIndex::build_for_metric(&data, n, dim, bits, DistanceMetric::Cosine);
        assert!(idx_cos.residual_sq.is_empty());
        assert_eq!(idx_cos.inv_v_hat_norms.len(), n);
    }

    #[test]
    fn test_parse_bits() {
        assert_eq!(parse_bits("FLAT-IP-POLARVEC", 4), 4);
        assert_eq!(parse_bits("FLAT-IP-POLARVEC3", 4), 3);
        assert_eq!(parse_bits("FLAT-L2-POLARVEC4", 4), 4);
        assert_eq!(parse_bits("FLAT-COS-POLARVEC8", 4), 8);
        assert_eq!(parse_bits("FLAT-IP-POLARVEC0", 4), 4); // 0 out of range → default
        assert_eq!(parse_bits("FLAT-IP-POLARVEC9", 4), 4); // 9 out of range → default
    }

    #[test]
    #[ignore]
    fn bench_polarvec_vs_flat_speed() {
        use std::time::Instant;

        let k = 10;
        let nq = 200;
        let dim = 128;

        for &n in &[20_000usize, 100_000, 500_000] {
            for &bits in &[4usize, 3] {
                let oversample = DEFAULT_OVERSAMPLE; // 20
                let data = random_data(n, dim, 1);
                let queries = random_data(nq, dim, 2);

                let idx_ip = PolarVecIndex::build_for_metric(
                    &data,
                    n,
                    dim,
                    bits,
                    DistanceMetric::InnerProduct,
                );
                let idx_l2 =
                    PolarVecIndex::build_for_metric(&data, n, dim, bits, DistanceMetric::L2Squared);

                // --- flat brute-force (IP) ---
                let t0 = Instant::now();
                for qi in 0..nq {
                    let q = &queries[qi * dim..(qi + 1) * dim];
                    let _ = exact_topk(&data, dim, q, k, DistanceMetric::InnerProduct);
                }
                let flat_us = t0.elapsed().as_micros() as f64 / nq as f64;

                // --- PolarVec IP ---
                let t0 = Instant::now();
                for qi in 0..nq {
                    let q = &queries[qi * dim..(qi + 1) * dim];
                    let _ = idx_ip.search(q, k, &data, DistanceMetric::InnerProduct, oversample);
                }
                let pv_ip_us = t0.elapsed().as_micros() as f64 / nq as f64;

                // --- PolarVec L2 ---
                let t0 = Instant::now();
                for qi in 0..nq {
                    let q = &queries[qi * dim..(qi + 1) * dim];
                    let _ = idx_l2.search(q, k, &data, DistanceMetric::L2Squared, oversample);
                }
                let pv_l2_us = t0.elapsed().as_micros() as f64 / nq as f64;

                // --- PolarVec Cosine ---
                let mut data_cos = data.clone();
                normalize_rows(&mut data_cos, dim);
                let mut queries_cos = queries.clone();
                normalize_rows(&mut queries_cos, dim);
                let idx_cos = PolarVecIndex::build_for_metric(
                    &data_cos,
                    n,
                    dim,
                    bits,
                    DistanceMetric::Cosine,
                );

                let t0 = Instant::now();
                for qi in 0..nq {
                    let q = &queries_cos[qi * dim..(qi + 1) * dim];
                    let _ = idx_cos.search(q, k, &data_cos, DistanceMetric::Cosine, oversample);
                }
                let pv_cos_us = t0.elapsed().as_micros() as f64 / nq as f64;

                // --- recall (IP, bits) ---
                let mut recall_ip = 0.0f32;
                let mut recall_cos = 0.0f32;
                let nq_r = nq.min(50);
                for qi in 0..nq_r {
                    let q = &queries[qi * dim..(qi + 1) * dim];
                    let exact = exact_topk(&data, dim, q, k, DistanceMetric::InnerProduct);
                    let approx = idx_ip
                        .search(q, k, &data, DistanceMetric::InnerProduct, oversample)
                        .0;
                    recall_ip += recall_at_k(&exact, &approx);
                }
                for qi in 0..nq_r {
                    let q = &queries_cos[qi * dim..(qi + 1) * dim];
                    let exact = exact_topk(&data_cos, dim, q, k, DistanceMetric::Cosine);
                    let approx = idx_cos
                        .search(q, k, &data_cos, DistanceMetric::Cosine, oversample)
                        .0;
                    recall_cos += recall_at_k(&exact, &approx);
                }

                println!(
                    "n={:>7} bits={} oversample={:2} | flat {:.1}µs | PV-IP {:.1}µs ({:.1}x) recall@{}={:.3} | PV-L2 {:.1}µs ({:.1}x) | PV-Cos {:.1}µs ({:.1}x) recall@{}={:.3}",
                    n, bits, oversample,
                    flat_us,
                    pv_ip_us,  flat_us / pv_ip_us,  k, recall_ip  / nq_r as f32,
                    pv_l2_us,  flat_us / pv_l2_us,
                    pv_cos_us, flat_us / pv_cos_us, k, recall_cos / nq_r as f32,
                );
            }
        }
    }
}
