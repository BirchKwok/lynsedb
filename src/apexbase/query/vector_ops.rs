//! Vector distance operations for ANN / TopK queries.
//!
//! Supports:
//!   - L2 (Euclidean) distance
//!   - Cosine distance / similarity
//!   - Inner product (dot product)
//!   - L1 (Manhattan) distance
//!   - L∞ (Chebyshev) distance
//!
//! Vectors are stored as **Binary** columns (raw little-endian f32 bytes).
//! A `dim`-dimensional vector occupies exactly `dim * 4` bytes.
//!
//! The batch compute functions use Rayon for row-level parallelism; the
//! per-vector kernels are written to enable LLVM AVX-2 auto-vectorisation
//! (8×f32 per register) at opt-level=3.

use std::borrow::Cow;
use std::io;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BinaryArray, Float32Array, Float64Array, StringArray};
use rayon::prelude::*;

const MAX_BLOCKED_BATCH_QUERIES: usize = 32;

// ─────────────────────────────────────────────────────────────────────────────
// Fast reciprocal-sqrt: NEON FRSQRTE + 1 Newton-Raphson step on AArch64,
// plain 1/sqrt elsewhere.  Caller must ensure x > 0.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn rsqrt_positive_f32(x: f32) -> f32 {
    // Scalar NEON FRSQRTE + one Newton-Raphson step.
    // Stays entirely in FP/SIMD s-registers — no GPR↔SIMD lane crossing.
    // Empirically faster than FRSQRTE-alone: LLVM pipelines the NR chain
    // better alongside the surrounding FMA accumulation loop.
    unsafe {
        use std::arch::aarch64::*;
        let e = vrsqrtes_f32(x);
        let step = vrsqrtss_f32(x * e, e); // (3 - x·e²) / 2
        e * step
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn rsqrt_positive_f32(x: f32) -> f32 {
    // SSE RSQRTSS + one Newton-Raphson step (~12→23-bit precision).
    // Matches the NEON FRSQRTE approach on AArch64.
    // Always available on x86_64 — no runtime detection needed.
    unsafe {
        use std::arch::x86_64::*;
        let v = _mm_set_ss(x);
        let est = _mm_rsqrt_ss(v);
        let e = _mm_cvtss_f32(est);
        // Newton-Raphson: e' = e * (3 - x·e²) / 2
        e * (1.5 - 0.5 * x * e * e)
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
fn rsqrt_positive_f32(x: f32) -> f32 {
    1.0 / x.sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Read little-endian f32 bytes as a slice when aligned, otherwise copy safely.
#[inline(always)]
fn bytes_to_f32(bytes: &[u8]) -> Option<Cow<'_, [f32]>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    let len = bytes.len() / 4;
    if bytes.as_ptr().align_offset(std::mem::align_of::<f32>()) == 0 {
        Some(Cow::Borrowed(unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len)
        }))
    } else {
        Some(Cow::Owned(
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ))
    }
}

/// Parse query-vector bytes (either raw f32 LE or raw f64 LE) to a `Vec<f32>`.
pub fn bytes_to_query_vec_f32(bytes: &[u8]) -> Option<Vec<f32>> {
    bytes_to_f32(bytes).map(|floats| floats.into_owned())
}

// ─────────────────────────────────────────────────────────────────────────────
// Core scalar kernels  (LLVM will auto-vectorise with opt-level=3)
// ─────────────────────────────────────────────────────────────────────────────

/// L2 squared distance (Σ(aᵢ−bᵢ)²).  No sqrt → safe for ordering.
///
/// Uses 4 independent accumulators (16-way unroll) to hide FMA latency
/// (~4 cycles on NEON, ~5 on AVX2). LLVM maps each accumulator to a
/// separate SIMD register, sustaining near-peak multiply-add throughput.
#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    unsafe { l2_squared_neon(a, b) }
}

#[inline(always)]
#[cfg(not(target_arch = "aarch64"))]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { l2_squared_avx2(a, b) };
        }
    }
    let n = a.len().min(b.len());
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let c = n / 16;
    for i in 0..c {
        let o = i * 16;
        let (d0, d1, d2, d3) = (
            a[o] - b[o],
            a[o + 1] - b[o + 1],
            a[o + 2] - b[o + 2],
            a[o + 3] - b[o + 3],
        );
        let (d4, d5, d6, d7) = (
            a[o + 4] - b[o + 4],
            a[o + 5] - b[o + 5],
            a[o + 6] - b[o + 6],
            a[o + 7] - b[o + 7],
        );
        let (d8, d9, d10, d11) = (
            a[o + 8] - b[o + 8],
            a[o + 9] - b[o + 9],
            a[o + 10] - b[o + 10],
            a[o + 11] - b[o + 11],
        );
        let (d12, d13, d14, d15) = (
            a[o + 12] - b[o + 12],
            a[o + 13] - b[o + 13],
            a[o + 14] - b[o + 14],
            a[o + 15] - b[o + 15],
        );
        s0 += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        s1 += d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
        s2 += d8 * d8 + d9 * d9 + d10 * d10 + d11 * d11;
        s3 += d12 * d12 + d13 * d13 + d14 * d14 + d15 * d15;
    }
    let mut s = s0 + s1 + s2 + s3;
    for i in (c * 16)..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// L2 (Euclidean) distance √(Σ(aᵢ−bᵢ)²).
#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b).sqrt()
}

/// L1 (Manhattan) distance Σ|aᵢ−bᵢ|.
///
/// 4 independent accumulators hide the abs+add latency chain.
/// NEON maps abs to `FABS`, AVX2 uses a sign-mask `VANDNPS`.
#[inline(always)]
pub fn l1_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { l1_distance_avx2(a, b) };
        }
    }
    let n = a.len().min(b.len());
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let c = n / 16;
    for i in 0..c {
        let o = i * 16;
        s0 += (a[o] - b[o]).abs()
            + (a[o + 1] - b[o + 1]).abs()
            + (a[o + 2] - b[o + 2]).abs()
            + (a[o + 3] - b[o + 3]).abs();
        s1 += (a[o + 4] - b[o + 4]).abs()
            + (a[o + 5] - b[o + 5]).abs()
            + (a[o + 6] - b[o + 6]).abs()
            + (a[o + 7] - b[o + 7]).abs();
        s2 += (a[o + 8] - b[o + 8]).abs()
            + (a[o + 9] - b[o + 9]).abs()
            + (a[o + 10] - b[o + 10]).abs()
            + (a[o + 11] - b[o + 11]).abs();
        s3 += (a[o + 12] - b[o + 12]).abs()
            + (a[o + 13] - b[o + 13]).abs()
            + (a[o + 14] - b[o + 14]).abs()
            + (a[o + 15] - b[o + 15]).abs();
    }
    let mut s = s0 + s1 + s2 + s3;
    for i in (c * 16)..n {
        s += (a[i] - b[i]).abs();
    }
    s
}

/// L∞ (Chebyshev) distance max|aᵢ−bᵢ|.
///
/// 4 independent max lanes allow SIMD vectorization (NEON `FMAX`, AVX2 `VMAXPS`).
#[inline(always)]
pub fn linf_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { linf_distance_avx2(a, b) };
        }
    }
    let n = a.len().min(b.len());
    let mut m0 = 0.0f32;
    let mut m1 = 0.0f32;
    let mut m2 = 0.0f32;
    let mut m3 = 0.0f32;
    let c = n / 16;
    for i in 0..c {
        let o = i * 16;
        m0 = m0
            .max((a[o] - b[o]).abs())
            .max((a[o + 1] - b[o + 1]).abs())
            .max((a[o + 2] - b[o + 2]).abs())
            .max((a[o + 3] - b[o + 3]).abs());
        m1 = m1
            .max((a[o + 4] - b[o + 4]).abs())
            .max((a[o + 5] - b[o + 5]).abs())
            .max((a[o + 6] - b[o + 6]).abs())
            .max((a[o + 7] - b[o + 7]).abs());
        m2 = m2
            .max((a[o + 8] - b[o + 8]).abs())
            .max((a[o + 9] - b[o + 9]).abs())
            .max((a[o + 10] - b[o + 10]).abs())
            .max((a[o + 11] - b[o + 11]).abs());
        m3 = m3
            .max((a[o + 12] - b[o + 12]).abs())
            .max((a[o + 13] - b[o + 13]).abs())
            .max((a[o + 14] - b[o + 14]).abs())
            .max((a[o + 15] - b[o + 15]).abs());
    }
    let mut m = m0.max(m1).max(m2).max(m3);
    for i in (c * 16)..n {
        m = m.max((a[i] - b[i]).abs());
    }
    m
}

/// Dot product Σ aᵢ·bᵢ.
///
/// 4 independent accumulators hide FMA latency and enable
/// out-of-order execution across 4 NEON/AVX2 multiply-add pipelines.
#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    unsafe { inner_product_neon(a, b) }
}

#[inline(always)]
#[cfg(not(target_arch = "aarch64"))]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { inner_product_avx2(a, b) };
        }
    }
    let n = a.len().min(b.len());
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let c = n / 16;
    for i in 0..c {
        let o = i * 16;
        s0 += a[o] * b[o] + a[o + 1] * b[o + 1] + a[o + 2] * b[o + 2] + a[o + 3] * b[o + 3];
        s1 += a[o + 4] * b[o + 4] + a[o + 5] * b[o + 5] + a[o + 6] * b[o + 6] + a[o + 7] * b[o + 7];
        s2 += a[o + 8] * b[o + 8]
            + a[o + 9] * b[o + 9]
            + a[o + 10] * b[o + 10]
            + a[o + 11] * b[o + 11];
        s3 += a[o + 12] * b[o + 12]
            + a[o + 13] * b[o + 13]
            + a[o + 14] * b[o + 14]
            + a[o + 15] * b[o + 15];
    }
    let mut s = s0 + s1 + s2 + s3;
    for i in (c * 16)..n {
        s += a[i] * b[i];
    }
    s
}

/// Squared L2 norm (Σ aᵢ²) for a single vector.
///
/// Uses the same 16-way unrolled accumulation style as the other hot kernels so
/// batch cosine can reuse a row norm across multiple queries without falling
/// back to iterator-heavy code.
#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub fn l2_norm_squared(a: &[f32]) -> f32 {
    unsafe { l2_norm_squared_neon(a) }
}

#[inline(always)]
#[cfg(not(target_arch = "aarch64"))]
pub fn l2_norm_squared(a: &[f32]) -> f32 {
    let n = a.len();
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let c = n / 16;
    for i in 0..c {
        let o = i * 16;
        s0 += a[o] * a[o] + a[o + 1] * a[o + 1] + a[o + 2] * a[o + 2] + a[o + 3] * a[o + 3];
        s1 += a[o + 4] * a[o + 4] + a[o + 5] * a[o + 5] + a[o + 6] * a[o + 6] + a[o + 7] * a[o + 7];
        s2 += a[o + 8] * a[o + 8]
            + a[o + 9] * a[o + 9]
            + a[o + 10] * a[o + 10]
            + a[o + 11] * a[o + 11];
        s3 += a[o + 12] * a[o + 12]
            + a[o + 13] * a[o + 13]
            + a[o + 14] * a[o + 14]
            + a[o + 15] * a[o + 15];
    }
    let mut s = s0 + s1 + s2 + s3;
    for &x in &a[c * 16..] {
        s += x * x;
    }
    s
}

/// Compute dot(row, query_j) for every query while streaming the row only once.
///
/// This is used by blocked batch cosine search where the row norm is already
/// shared across queries. Compared with calling `inner_product(row, query)` N
/// times, this keeps the hot row in registers/L1 and avoids rereading it for
/// each query.
#[inline(always)]
fn batch_inner_products(
    row: &[f32],
    queries: &[&[f32]],
    dots: &mut [f32; MAX_BLOCKED_BATCH_QUERIES],
) {
    let n_queries = queries.len();
    debug_assert!(n_queries <= MAX_BLOCKED_BATCH_QUERIES);
    dots[..n_queries].fill(0.0);

    let n = row.len();
    let c = n / 16;
    for i in 0..c {
        let o = i * 16;
        let r0 = unsafe { *row.get_unchecked(o) };
        let r1 = unsafe { *row.get_unchecked(o + 1) };
        let r2 = unsafe { *row.get_unchecked(o + 2) };
        let r3 = unsafe { *row.get_unchecked(o + 3) };
        let r4 = unsafe { *row.get_unchecked(o + 4) };
        let r5 = unsafe { *row.get_unchecked(o + 5) };
        let r6 = unsafe { *row.get_unchecked(o + 6) };
        let r7 = unsafe { *row.get_unchecked(o + 7) };
        let r8 = unsafe { *row.get_unchecked(o + 8) };
        let r9 = unsafe { *row.get_unchecked(o + 9) };
        let r10 = unsafe { *row.get_unchecked(o + 10) };
        let r11 = unsafe { *row.get_unchecked(o + 11) };
        let r12 = unsafe { *row.get_unchecked(o + 12) };
        let r13 = unsafe { *row.get_unchecked(o + 13) };
        let r14 = unsafe { *row.get_unchecked(o + 14) };
        let r15 = unsafe { *row.get_unchecked(o + 15) };

        let mut qi = 0usize;
        while qi + 4 <= n_queries {
            let q0 = unsafe { *queries.get_unchecked(qi) };
            let q1 = unsafe { *queries.get_unchecked(qi + 1) };
            let q2 = unsafe { *queries.get_unchecked(qi + 2) };
            let q3 = unsafe { *queries.get_unchecked(qi + 3) };

            dots[qi] += r0 * unsafe { *q0.get_unchecked(o) }
                + r1 * unsafe { *q0.get_unchecked(o + 1) }
                + r2 * unsafe { *q0.get_unchecked(o + 2) }
                + r3 * unsafe { *q0.get_unchecked(o + 3) }
                + r4 * unsafe { *q0.get_unchecked(o + 4) }
                + r5 * unsafe { *q0.get_unchecked(o + 5) }
                + r6 * unsafe { *q0.get_unchecked(o + 6) }
                + r7 * unsafe { *q0.get_unchecked(o + 7) }
                + r8 * unsafe { *q0.get_unchecked(o + 8) }
                + r9 * unsafe { *q0.get_unchecked(o + 9) }
                + r10 * unsafe { *q0.get_unchecked(o + 10) }
                + r11 * unsafe { *q0.get_unchecked(o + 11) }
                + r12 * unsafe { *q0.get_unchecked(o + 12) }
                + r13 * unsafe { *q0.get_unchecked(o + 13) }
                + r14 * unsafe { *q0.get_unchecked(o + 14) }
                + r15 * unsafe { *q0.get_unchecked(o + 15) };
            dots[qi + 1] += r0 * unsafe { *q1.get_unchecked(o) }
                + r1 * unsafe { *q1.get_unchecked(o + 1) }
                + r2 * unsafe { *q1.get_unchecked(o + 2) }
                + r3 * unsafe { *q1.get_unchecked(o + 3) }
                + r4 * unsafe { *q1.get_unchecked(o + 4) }
                + r5 * unsafe { *q1.get_unchecked(o + 5) }
                + r6 * unsafe { *q1.get_unchecked(o + 6) }
                + r7 * unsafe { *q1.get_unchecked(o + 7) }
                + r8 * unsafe { *q1.get_unchecked(o + 8) }
                + r9 * unsafe { *q1.get_unchecked(o + 9) }
                + r10 * unsafe { *q1.get_unchecked(o + 10) }
                + r11 * unsafe { *q1.get_unchecked(o + 11) }
                + r12 * unsafe { *q1.get_unchecked(o + 12) }
                + r13 * unsafe { *q1.get_unchecked(o + 13) }
                + r14 * unsafe { *q1.get_unchecked(o + 14) }
                + r15 * unsafe { *q1.get_unchecked(o + 15) };
            dots[qi + 2] += r0 * unsafe { *q2.get_unchecked(o) }
                + r1 * unsafe { *q2.get_unchecked(o + 1) }
                + r2 * unsafe { *q2.get_unchecked(o + 2) }
                + r3 * unsafe { *q2.get_unchecked(o + 3) }
                + r4 * unsafe { *q2.get_unchecked(o + 4) }
                + r5 * unsafe { *q2.get_unchecked(o + 5) }
                + r6 * unsafe { *q2.get_unchecked(o + 6) }
                + r7 * unsafe { *q2.get_unchecked(o + 7) }
                + r8 * unsafe { *q2.get_unchecked(o + 8) }
                + r9 * unsafe { *q2.get_unchecked(o + 9) }
                + r10 * unsafe { *q2.get_unchecked(o + 10) }
                + r11 * unsafe { *q2.get_unchecked(o + 11) }
                + r12 * unsafe { *q2.get_unchecked(o + 12) }
                + r13 * unsafe { *q2.get_unchecked(o + 13) }
                + r14 * unsafe { *q2.get_unchecked(o + 14) }
                + r15 * unsafe { *q2.get_unchecked(o + 15) };
            dots[qi + 3] += r0 * unsafe { *q3.get_unchecked(o) }
                + r1 * unsafe { *q3.get_unchecked(o + 1) }
                + r2 * unsafe { *q3.get_unchecked(o + 2) }
                + r3 * unsafe { *q3.get_unchecked(o + 3) }
                + r4 * unsafe { *q3.get_unchecked(o + 4) }
                + r5 * unsafe { *q3.get_unchecked(o + 5) }
                + r6 * unsafe { *q3.get_unchecked(o + 6) }
                + r7 * unsafe { *q3.get_unchecked(o + 7) }
                + r8 * unsafe { *q3.get_unchecked(o + 8) }
                + r9 * unsafe { *q3.get_unchecked(o + 9) }
                + r10 * unsafe { *q3.get_unchecked(o + 10) }
                + r11 * unsafe { *q3.get_unchecked(o + 11) }
                + r12 * unsafe { *q3.get_unchecked(o + 12) }
                + r13 * unsafe { *q3.get_unchecked(o + 13) }
                + r14 * unsafe { *q3.get_unchecked(o + 14) }
                + r15 * unsafe { *q3.get_unchecked(o + 15) };
            qi += 4;
        }

        while qi < n_queries {
            let q = unsafe { *queries.get_unchecked(qi) };
            dots[qi] += r0 * unsafe { *q.get_unchecked(o) }
                + r1 * unsafe { *q.get_unchecked(o + 1) }
                + r2 * unsafe { *q.get_unchecked(o + 2) }
                + r3 * unsafe { *q.get_unchecked(o + 3) }
                + r4 * unsafe { *q.get_unchecked(o + 4) }
                + r5 * unsafe { *q.get_unchecked(o + 5) }
                + r6 * unsafe { *q.get_unchecked(o + 6) }
                + r7 * unsafe { *q.get_unchecked(o + 7) }
                + r8 * unsafe { *q.get_unchecked(o + 8) }
                + r9 * unsafe { *q.get_unchecked(o + 9) }
                + r10 * unsafe { *q.get_unchecked(o + 10) }
                + r11 * unsafe { *q.get_unchecked(o + 11) }
                + r12 * unsafe { *q.get_unchecked(o + 12) }
                + r13 * unsafe { *q.get_unchecked(o + 13) }
                + r14 * unsafe { *q.get_unchecked(o + 14) }
                + r15 * unsafe { *q.get_unchecked(o + 15) };
            qi += 1;
        }
    }

    for i in (c * 16)..n {
        let rv = unsafe { *row.get_unchecked(i) };
        for (qi, q) in queries.iter().enumerate() {
            dots[qi] += rv * unsafe { *q.get_unchecked(i) };
        }
    }
}

#[inline(always)]
fn batch_l1_distances(
    row: &[f32],
    queries: &[&[f32]],
    distances: &mut [f32; MAX_BLOCKED_BATCH_QUERIES],
) {
    let n_queries = queries.len();
    debug_assert!(n_queries <= MAX_BLOCKED_BATCH_QUERIES);
    distances[..n_queries].fill(0.0);
    let n = row.len();
    let c = n / 16;

    let mut qi = 0usize;
    while qi + 4 <= n_queries {
        let q0 = unsafe { *queries.get_unchecked(qi) };
        let q1 = unsafe { *queries.get_unchecked(qi + 1) };
        let q2 = unsafe { *queries.get_unchecked(qi + 2) };
        let q3 = unsafe { *queries.get_unchecked(qi + 3) };

        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;

        for i in 0..c {
            let o = i * 16;
            let r0 = unsafe { *row.get_unchecked(o) };
            let r1 = unsafe { *row.get_unchecked(o + 1) };
            let r2 = unsafe { *row.get_unchecked(o + 2) };
            let r3 = unsafe { *row.get_unchecked(o + 3) };
            let r4 = unsafe { *row.get_unchecked(o + 4) };
            let r5 = unsafe { *row.get_unchecked(o + 5) };
            let r6 = unsafe { *row.get_unchecked(o + 6) };
            let r7 = unsafe { *row.get_unchecked(o + 7) };
            let r8 = unsafe { *row.get_unchecked(o + 8) };
            let r9 = unsafe { *row.get_unchecked(o + 9) };
            let r10 = unsafe { *row.get_unchecked(o + 10) };
            let r11 = unsafe { *row.get_unchecked(o + 11) };
            let r12 = unsafe { *row.get_unchecked(o + 12) };
            let r13 = unsafe { *row.get_unchecked(o + 13) };
            let r14 = unsafe { *row.get_unchecked(o + 14) };
            let r15 = unsafe { *row.get_unchecked(o + 15) };

            s0 += (r0 - unsafe { *q0.get_unchecked(o) }).abs()
                + (r1 - unsafe { *q0.get_unchecked(o + 1) }).abs()
                + (r2 - unsafe { *q0.get_unchecked(o + 2) }).abs()
                + (r3 - unsafe { *q0.get_unchecked(o + 3) }).abs()
                + (r4 - unsafe { *q0.get_unchecked(o + 4) }).abs()
                + (r5 - unsafe { *q0.get_unchecked(o + 5) }).abs()
                + (r6 - unsafe { *q0.get_unchecked(o + 6) }).abs()
                + (r7 - unsafe { *q0.get_unchecked(o + 7) }).abs()
                + (r8 - unsafe { *q0.get_unchecked(o + 8) }).abs()
                + (r9 - unsafe { *q0.get_unchecked(o + 9) }).abs()
                + (r10 - unsafe { *q0.get_unchecked(o + 10) }).abs()
                + (r11 - unsafe { *q0.get_unchecked(o + 11) }).abs()
                + (r12 - unsafe { *q0.get_unchecked(o + 12) }).abs()
                + (r13 - unsafe { *q0.get_unchecked(o + 13) }).abs()
                + (r14 - unsafe { *q0.get_unchecked(o + 14) }).abs()
                + (r15 - unsafe { *q0.get_unchecked(o + 15) }).abs();
            s1 += (r0 - unsafe { *q1.get_unchecked(o) }).abs()
                + (r1 - unsafe { *q1.get_unchecked(o + 1) }).abs()
                + (r2 - unsafe { *q1.get_unchecked(o + 2) }).abs()
                + (r3 - unsafe { *q1.get_unchecked(o + 3) }).abs()
                + (r4 - unsafe { *q1.get_unchecked(o + 4) }).abs()
                + (r5 - unsafe { *q1.get_unchecked(o + 5) }).abs()
                + (r6 - unsafe { *q1.get_unchecked(o + 6) }).abs()
                + (r7 - unsafe { *q1.get_unchecked(o + 7) }).abs()
                + (r8 - unsafe { *q1.get_unchecked(o + 8) }).abs()
                + (r9 - unsafe { *q1.get_unchecked(o + 9) }).abs()
                + (r10 - unsafe { *q1.get_unchecked(o + 10) }).abs()
                + (r11 - unsafe { *q1.get_unchecked(o + 11) }).abs()
                + (r12 - unsafe { *q1.get_unchecked(o + 12) }).abs()
                + (r13 - unsafe { *q1.get_unchecked(o + 13) }).abs()
                + (r14 - unsafe { *q1.get_unchecked(o + 14) }).abs()
                + (r15 - unsafe { *q1.get_unchecked(o + 15) }).abs();
            s2 += (r0 - unsafe { *q2.get_unchecked(o) }).abs()
                + (r1 - unsafe { *q2.get_unchecked(o + 1) }).abs()
                + (r2 - unsafe { *q2.get_unchecked(o + 2) }).abs()
                + (r3 - unsafe { *q2.get_unchecked(o + 3) }).abs()
                + (r4 - unsafe { *q2.get_unchecked(o + 4) }).abs()
                + (r5 - unsafe { *q2.get_unchecked(o + 5) }).abs()
                + (r6 - unsafe { *q2.get_unchecked(o + 6) }).abs()
                + (r7 - unsafe { *q2.get_unchecked(o + 7) }).abs()
                + (r8 - unsafe { *q2.get_unchecked(o + 8) }).abs()
                + (r9 - unsafe { *q2.get_unchecked(o + 9) }).abs()
                + (r10 - unsafe { *q2.get_unchecked(o + 10) }).abs()
                + (r11 - unsafe { *q2.get_unchecked(o + 11) }).abs()
                + (r12 - unsafe { *q2.get_unchecked(o + 12) }).abs()
                + (r13 - unsafe { *q2.get_unchecked(o + 13) }).abs()
                + (r14 - unsafe { *q2.get_unchecked(o + 14) }).abs()
                + (r15 - unsafe { *q2.get_unchecked(o + 15) }).abs();
            s3 += (r0 - unsafe { *q3.get_unchecked(o) }).abs()
                + (r1 - unsafe { *q3.get_unchecked(o + 1) }).abs()
                + (r2 - unsafe { *q3.get_unchecked(o + 2) }).abs()
                + (r3 - unsafe { *q3.get_unchecked(o + 3) }).abs()
                + (r4 - unsafe { *q3.get_unchecked(o + 4) }).abs()
                + (r5 - unsafe { *q3.get_unchecked(o + 5) }).abs()
                + (r6 - unsafe { *q3.get_unchecked(o + 6) }).abs()
                + (r7 - unsafe { *q3.get_unchecked(o + 7) }).abs()
                + (r8 - unsafe { *q3.get_unchecked(o + 8) }).abs()
                + (r9 - unsafe { *q3.get_unchecked(o + 9) }).abs()
                + (r10 - unsafe { *q3.get_unchecked(o + 10) }).abs()
                + (r11 - unsafe { *q3.get_unchecked(o + 11) }).abs()
                + (r12 - unsafe { *q3.get_unchecked(o + 12) }).abs()
                + (r13 - unsafe { *q3.get_unchecked(o + 13) }).abs()
                + (r14 - unsafe { *q3.get_unchecked(o + 14) }).abs()
                + (r15 - unsafe { *q3.get_unchecked(o + 15) }).abs();
        }

        for i in (c * 16)..n {
            let rv = unsafe { *row.get_unchecked(i) };
            s0 += (rv - unsafe { *q0.get_unchecked(i) }).abs();
            s1 += (rv - unsafe { *q1.get_unchecked(i) }).abs();
            s2 += (rv - unsafe { *q2.get_unchecked(i) }).abs();
            s3 += (rv - unsafe { *q3.get_unchecked(i) }).abs();
        }

        distances[qi] = s0;
        distances[qi + 1] = s1;
        distances[qi + 2] = s2;
        distances[qi + 3] = s3;
        qi += 4;
    }

    while qi < n_queries {
        distances[qi] = l1_distance(row, unsafe { *queries.get_unchecked(qi) });
        qi += 1;
    }
}

#[inline(always)]
fn batch_linf_distances(
    row: &[f32],
    queries: &[&[f32]],
    distances: &mut [f32; MAX_BLOCKED_BATCH_QUERIES],
) {
    let n_queries = queries.len();
    debug_assert!(n_queries <= MAX_BLOCKED_BATCH_QUERIES);
    distances[..n_queries].fill(0.0);
    let n = row.len();
    let c = n / 16;

    let mut qi = 0usize;
    while qi + 4 <= n_queries {
        let q0 = unsafe { *queries.get_unchecked(qi) };
        let q1 = unsafe { *queries.get_unchecked(qi + 1) };
        let q2 = unsafe { *queries.get_unchecked(qi + 2) };
        let q3 = unsafe { *queries.get_unchecked(qi + 3) };

        let mut m0 = 0.0f32;
        let mut m1 = 0.0f32;
        let mut m2 = 0.0f32;
        let mut m3 = 0.0f32;

        for i in 0..c {
            let o = i * 16;
            let r0 = unsafe { *row.get_unchecked(o) };
            let r1 = unsafe { *row.get_unchecked(o + 1) };
            let r2 = unsafe { *row.get_unchecked(o + 2) };
            let r3 = unsafe { *row.get_unchecked(o + 3) };
            let r4 = unsafe { *row.get_unchecked(o + 4) };
            let r5 = unsafe { *row.get_unchecked(o + 5) };
            let r6 = unsafe { *row.get_unchecked(o + 6) };
            let r7 = unsafe { *row.get_unchecked(o + 7) };
            let r8 = unsafe { *row.get_unchecked(o + 8) };
            let r9 = unsafe { *row.get_unchecked(o + 9) };
            let r10 = unsafe { *row.get_unchecked(o + 10) };
            let r11 = unsafe { *row.get_unchecked(o + 11) };
            let r12 = unsafe { *row.get_unchecked(o + 12) };
            let r13 = unsafe { *row.get_unchecked(o + 13) };
            let r14 = unsafe { *row.get_unchecked(o + 14) };
            let r15 = unsafe { *row.get_unchecked(o + 15) };

            m0 = m0
                .max((r0 - unsafe { *q0.get_unchecked(o) }).abs())
                .max((r1 - unsafe { *q0.get_unchecked(o + 1) }).abs())
                .max((r2 - unsafe { *q0.get_unchecked(o + 2) }).abs())
                .max((r3 - unsafe { *q0.get_unchecked(o + 3) }).abs())
                .max((r4 - unsafe { *q0.get_unchecked(o + 4) }).abs())
                .max((r5 - unsafe { *q0.get_unchecked(o + 5) }).abs())
                .max((r6 - unsafe { *q0.get_unchecked(o + 6) }).abs())
                .max((r7 - unsafe { *q0.get_unchecked(o + 7) }).abs())
                .max((r8 - unsafe { *q0.get_unchecked(o + 8) }).abs())
                .max((r9 - unsafe { *q0.get_unchecked(o + 9) }).abs())
                .max((r10 - unsafe { *q0.get_unchecked(o + 10) }).abs())
                .max((r11 - unsafe { *q0.get_unchecked(o + 11) }).abs())
                .max((r12 - unsafe { *q0.get_unchecked(o + 12) }).abs())
                .max((r13 - unsafe { *q0.get_unchecked(o + 13) }).abs())
                .max((r14 - unsafe { *q0.get_unchecked(o + 14) }).abs())
                .max((r15 - unsafe { *q0.get_unchecked(o + 15) }).abs());
            m1 = m1
                .max((r0 - unsafe { *q1.get_unchecked(o) }).abs())
                .max((r1 - unsafe { *q1.get_unchecked(o + 1) }).abs())
                .max((r2 - unsafe { *q1.get_unchecked(o + 2) }).abs())
                .max((r3 - unsafe { *q1.get_unchecked(o + 3) }).abs())
                .max((r4 - unsafe { *q1.get_unchecked(o + 4) }).abs())
                .max((r5 - unsafe { *q1.get_unchecked(o + 5) }).abs())
                .max((r6 - unsafe { *q1.get_unchecked(o + 6) }).abs())
                .max((r7 - unsafe { *q1.get_unchecked(o + 7) }).abs())
                .max((r8 - unsafe { *q1.get_unchecked(o + 8) }).abs())
                .max((r9 - unsafe { *q1.get_unchecked(o + 9) }).abs())
                .max((r10 - unsafe { *q1.get_unchecked(o + 10) }).abs())
                .max((r11 - unsafe { *q1.get_unchecked(o + 11) }).abs())
                .max((r12 - unsafe { *q1.get_unchecked(o + 12) }).abs())
                .max((r13 - unsafe { *q1.get_unchecked(o + 13) }).abs())
                .max((r14 - unsafe { *q1.get_unchecked(o + 14) }).abs())
                .max((r15 - unsafe { *q1.get_unchecked(o + 15) }).abs());
            m2 = m2
                .max((r0 - unsafe { *q2.get_unchecked(o) }).abs())
                .max((r1 - unsafe { *q2.get_unchecked(o + 1) }).abs())
                .max((r2 - unsafe { *q2.get_unchecked(o + 2) }).abs())
                .max((r3 - unsafe { *q2.get_unchecked(o + 3) }).abs())
                .max((r4 - unsafe { *q2.get_unchecked(o + 4) }).abs())
                .max((r5 - unsafe { *q2.get_unchecked(o + 5) }).abs())
                .max((r6 - unsafe { *q2.get_unchecked(o + 6) }).abs())
                .max((r7 - unsafe { *q2.get_unchecked(o + 7) }).abs())
                .max((r8 - unsafe { *q2.get_unchecked(o + 8) }).abs())
                .max((r9 - unsafe { *q2.get_unchecked(o + 9) }).abs())
                .max((r10 - unsafe { *q2.get_unchecked(o + 10) }).abs())
                .max((r11 - unsafe { *q2.get_unchecked(o + 11) }).abs())
                .max((r12 - unsafe { *q2.get_unchecked(o + 12) }).abs())
                .max((r13 - unsafe { *q2.get_unchecked(o + 13) }).abs())
                .max((r14 - unsafe { *q2.get_unchecked(o + 14) }).abs())
                .max((r15 - unsafe { *q2.get_unchecked(o + 15) }).abs());
            m3 = m3
                .max((r0 - unsafe { *q3.get_unchecked(o) }).abs())
                .max((r1 - unsafe { *q3.get_unchecked(o + 1) }).abs())
                .max((r2 - unsafe { *q3.get_unchecked(o + 2) }).abs())
                .max((r3 - unsafe { *q3.get_unchecked(o + 3) }).abs())
                .max((r4 - unsafe { *q3.get_unchecked(o + 4) }).abs())
                .max((r5 - unsafe { *q3.get_unchecked(o + 5) }).abs())
                .max((r6 - unsafe { *q3.get_unchecked(o + 6) }).abs())
                .max((r7 - unsafe { *q3.get_unchecked(o + 7) }).abs())
                .max((r8 - unsafe { *q3.get_unchecked(o + 8) }).abs())
                .max((r9 - unsafe { *q3.get_unchecked(o + 9) }).abs())
                .max((r10 - unsafe { *q3.get_unchecked(o + 10) }).abs())
                .max((r11 - unsafe { *q3.get_unchecked(o + 11) }).abs())
                .max((r12 - unsafe { *q3.get_unchecked(o + 12) }).abs())
                .max((r13 - unsafe { *q3.get_unchecked(o + 13) }).abs())
                .max((r14 - unsafe { *q3.get_unchecked(o + 14) }).abs())
                .max((r15 - unsafe { *q3.get_unchecked(o + 15) }).abs());
        }

        for i in (c * 16)..n {
            let rv = unsafe { *row.get_unchecked(i) };
            m0 = m0.max((rv - unsafe { *q0.get_unchecked(i) }).abs());
            m1 = m1.max((rv - unsafe { *q1.get_unchecked(i) }).abs());
            m2 = m2.max((rv - unsafe { *q2.get_unchecked(i) }).abs());
            m3 = m3.max((rv - unsafe { *q3.get_unchecked(i) }).abs());
        }

        distances[qi] = m0;
        distances[qi + 1] = m1;
        distances[qi + 2] = m2;
        distances[qi + 3] = m3;
        qi += 4;
    }

    while qi < n_queries {
        distances[qi] = linf_distance(row, unsafe { *queries.get_unchecked(qi) });
        qi += 1;
    }
}

/// Cosine similarity cos(a,b) = dot(a,b) / (‖a‖·‖b‖).
/// Returns 0.0 if either vector is zero.
#[inline(always)]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = inner_product(a, b);
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Cosine similarity with **pre-computed reciprocal query norm** `nb_recip = 1/‖b‖`.
///
/// Fuses dot-product and ‖a‖² into a single 16-way pass with 8 independent
/// accumulators (4 for dot, 4 for norm²). On NEON this maps to 8 `VFMLA`
/// chains executing in parallel; on AVX2 to 8 `VFMADD231PS` chains.
/// The final step uses `rsqrt_positive_f32` (NEON FRSQRTE+NR) instead of
/// `sqrt + div`, saving ~11 cycles per call.
/// `nb_recip` is pre-computed once per query by `DistanceComputer::new`.
#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub fn cosine_similarity_fused(a: &[f32], b: &[f32], nb_recip: f32) -> f32 {
    unsafe { cosine_fused_neon(a, b, nb_recip) }
}

#[inline(always)]
#[cfg(not(target_arch = "aarch64"))]
pub fn cosine_similarity_fused(a: &[f32], b: &[f32], nb_recip: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { cosine_fused_avx2(a, b, nb_recip) };
        }
    }
    let n = a.len().min(b.len());
    let mut dot0 = 0.0f32;
    let mut na0 = 0.0f32;
    let mut dot1 = 0.0f32;
    let mut na1 = 0.0f32;
    let mut dot2 = 0.0f32;
    let mut na2 = 0.0f32;
    let mut dot3 = 0.0f32;
    let mut na3 = 0.0f32;
    let c = n / 16;
    for i in 0..c {
        let o = i * 16;
        dot0 += a[o] * b[o] + a[o + 1] * b[o + 1] + a[o + 2] * b[o + 2] + a[o + 3] * b[o + 3];
        na0 += a[o] * a[o] + a[o + 1] * a[o + 1] + a[o + 2] * a[o + 2] + a[o + 3] * a[o + 3];
        dot1 +=
            a[o + 4] * b[o + 4] + a[o + 5] * b[o + 5] + a[o + 6] * b[o + 6] + a[o + 7] * b[o + 7];
        na1 +=
            a[o + 4] * a[o + 4] + a[o + 5] * a[o + 5] + a[o + 6] * a[o + 6] + a[o + 7] * a[o + 7];
        dot2 += a[o + 8] * b[o + 8]
            + a[o + 9] * b[o + 9]
            + a[o + 10] * b[o + 10]
            + a[o + 11] * b[o + 11];
        na2 += a[o + 8] * a[o + 8]
            + a[o + 9] * a[o + 9]
            + a[o + 10] * a[o + 10]
            + a[o + 11] * a[o + 11];
        dot3 += a[o + 12] * b[o + 12]
            + a[o + 13] * b[o + 13]
            + a[o + 14] * b[o + 14]
            + a[o + 15] * b[o + 15];
        na3 += a[o + 12] * a[o + 12]
            + a[o + 13] * a[o + 13]
            + a[o + 14] * a[o + 14]
            + a[o + 15] * a[o + 15];
    }
    let mut dot = dot0 + dot1 + dot2 + dot3;
    let mut na_sq = na0 + na1 + na2 + na3;
    for i in (c * 16)..n {
        dot += a[i] * b[i];
        na_sq += a[i] * a[i];
    }
    if na_sq == 0.0 || nb_recip == 0.0 {
        return 0.0;
    }
    dot * rsqrt_positive_f32(na_sq) * nb_recip
}

/// Cosine distance = 1 − cosine_similarity.
#[inline(always)]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

// ─────────────────────────────────────────────────────────────────────────────
// x86_64 AVX2+FMA f32 distance kernels (explicit SIMD for distribution builds)
// ─────────────────────────────────────────────────────────────────────────────
// Distribution wheels are built with baseline SSE2 target-cpu, so LLVM cannot
// auto-vectorise the scalar kernels above to AVX2. These explicit AVX2+FMA
// implementations are selected at runtime via is_x86_feature_detected!, giving
// full 256-bit SIMD throughput on Windows/Linux x86_64 with AVX2 support
// (Intel Haswell+, AMD Excavator+).

// ─────────────────────────────────────────────────────────────────────────────
// aarch64 NEON f32 distance kernels (Apple Silicon / ARM servers)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len().min(b.len());
    let chunks = n / 16;
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
    );
    for i in 0..chunks {
        let o = i * 16;
        let a0 = vld1q_f32(a.as_ptr().add(o));
        let a1 = vld1q_f32(a.as_ptr().add(o + 4));
        let a2 = vld1q_f32(a.as_ptr().add(o + 8));
        let a3 = vld1q_f32(a.as_ptr().add(o + 12));
        let b0 = vld1q_f32(b.as_ptr().add(o));
        let b1 = vld1q_f32(b.as_ptr().add(o + 4));
        let b2 = vld1q_f32(b.as_ptr().add(o + 8));
        let b3 = vld1q_f32(b.as_ptr().add(o + 12));
        let d0 = vsubq_f32(a0, b0);
        let d1 = vsubq_f32(a1, b1);
        let d2 = vsubq_f32(a2, b2);
        let d3 = vsubq_f32(a3, b3);
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
        acc2 = vfmaq_f32(acc2, d2, d2);
        acc3 = vfmaq_f32(acc3, d3, d3);
    }
    let pair0 = vaddq_f32(acc0, acc1);
    let pair1 = vaddq_f32(acc2, acc3);
    let mut s = vaddvq_f32(vaddq_f32(pair0, pair1));
    for i in (chunks * 16)..n {
        let d = *a.get_unchecked(i) - *b.get_unchecked(i);
        s += d * d;
    }
    s
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn inner_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len().min(b.len());
    let chunks = n / 16;
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
    );
    for i in 0..chunks {
        let o = i * 16;
        let a0 = vld1q_f32(a.as_ptr().add(o));
        let a1 = vld1q_f32(a.as_ptr().add(o + 4));
        let a2 = vld1q_f32(a.as_ptr().add(o + 8));
        let a3 = vld1q_f32(a.as_ptr().add(o + 12));
        let b0 = vld1q_f32(b.as_ptr().add(o));
        let b1 = vld1q_f32(b.as_ptr().add(o + 4));
        let b2 = vld1q_f32(b.as_ptr().add(o + 8));
        let b3 = vld1q_f32(b.as_ptr().add(o + 12));
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
        acc2 = vfmaq_f32(acc2, a2, b2);
        acc3 = vfmaq_f32(acc3, a3, b3);
    }
    let pair0 = vaddq_f32(acc0, acc1);
    let pair1 = vaddq_f32(acc2, acc3);
    let mut s = vaddvq_f32(vaddq_f32(pair0, pair1));
    for i in (chunks * 16)..n {
        s += *a.get_unchecked(i) * *b.get_unchecked(i);
    }
    s
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn l2_norm_squared_neon(a: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let chunks = n / 16;
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
    );
    for i in 0..chunks {
        let o = i * 16;
        let a0 = vld1q_f32(a.as_ptr().add(o));
        let a1 = vld1q_f32(a.as_ptr().add(o + 4));
        let a2 = vld1q_f32(a.as_ptr().add(o + 8));
        let a3 = vld1q_f32(a.as_ptr().add(o + 12));
        acc0 = vfmaq_f32(acc0, a0, a0);
        acc1 = vfmaq_f32(acc1, a1, a1);
        acc2 = vfmaq_f32(acc2, a2, a2);
        acc3 = vfmaq_f32(acc3, a3, a3);
    }
    let pair0 = vaddq_f32(acc0, acc1);
    let pair1 = vaddq_f32(acc2, acc3);
    let mut s = vaddvq_f32(vaddq_f32(pair0, pair1));
    for i in (chunks * 16)..n {
        let v = *a.get_unchecked(i);
        s += v * v;
    }
    s
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn cosine_fused_neon(a: &[f32], b: &[f32], nb_recip: f32) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len().min(b.len());
    let chunks = n / 16;
    let (mut dot0, mut dot1, mut dot2, mut dot3) = (
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
    );
    let (mut na0, mut na1, mut na2, mut na3) = (
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
    );
    for i in 0..chunks {
        let o = i * 16;
        let a0 = vld1q_f32(a.as_ptr().add(o));
        let a1 = vld1q_f32(a.as_ptr().add(o + 4));
        let a2 = vld1q_f32(a.as_ptr().add(o + 8));
        let a3 = vld1q_f32(a.as_ptr().add(o + 12));
        let b0 = vld1q_f32(b.as_ptr().add(o));
        let b1 = vld1q_f32(b.as_ptr().add(o + 4));
        let b2 = vld1q_f32(b.as_ptr().add(o + 8));
        let b3 = vld1q_f32(b.as_ptr().add(o + 12));
        dot0 = vfmaq_f32(dot0, a0, b0);
        dot1 = vfmaq_f32(dot1, a1, b1);
        dot2 = vfmaq_f32(dot2, a2, b2);
        dot3 = vfmaq_f32(dot3, a3, b3);
        na0 = vfmaq_f32(na0, a0, a0);
        na1 = vfmaq_f32(na1, a1, a1);
        na2 = vfmaq_f32(na2, a2, a2);
        na3 = vfmaq_f32(na3, a3, a3);
    }
    let dot_pair0 = vaddq_f32(dot0, dot1);
    let dot_pair1 = vaddq_f32(dot2, dot3);
    let norm_pair0 = vaddq_f32(na0, na1);
    let norm_pair1 = vaddq_f32(na2, na3);
    let mut dot = vaddvq_f32(vaddq_f32(dot_pair0, dot_pair1));
    let mut na_sq = vaddvq_f32(vaddq_f32(norm_pair0, norm_pair1));
    for i in (chunks * 16)..n {
        let ai = *a.get_unchecked(i);
        let bi = *b.get_unchecked(i);
        dot += ai * bi;
        na_sq += ai * ai;
    }
    if na_sq == 0.0 || nb_recip == 0.0 {
        return 0.0;
    }
    dot * rsqrt_positive_f32(na_sq) * nb_recip
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let chunks = n / 16;
    let (mut acc0, mut acc1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    for i in 0..chunks {
        let o = i * 16;
        let a0 = _mm256_loadu_ps(a.as_ptr().add(o));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(o + 8));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(o));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(o + 8));
        let d0 = _mm256_sub_ps(a0, b0);
        let d1 = _mm256_sub_ps(a1, b1);
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    }
    let mut s = hsum256(_mm256_add_ps(acc0, acc1));
    for i in (chunks * 16)..n {
        let d = *a.get_unchecked(i) - *b.get_unchecked(i);
        s += d * d;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn inner_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let chunks = n / 16;
    let (mut acc0, mut acc1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    for i in 0..chunks {
        let o = i * 16;
        let a0 = _mm256_loadu_ps(a.as_ptr().add(o));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(o + 8));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(o));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(o + 8));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
    }
    let mut s = hsum256(_mm256_add_ps(acc0, acc1));
    for i in (chunks * 16)..n {
        s += *a.get_unchecked(i) * *b.get_unchecked(i);
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cosine_fused_avx2(a: &[f32], b: &[f32], nb_recip: f32) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let chunks = n / 16;
    let (mut dot0, mut dot1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut na0, mut na1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    for i in 0..chunks {
        let o = i * 16;
        let a0 = _mm256_loadu_ps(a.as_ptr().add(o));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(o + 8));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(o));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(o + 8));
        dot0 = _mm256_fmadd_ps(a0, b0, dot0);
        dot1 = _mm256_fmadd_ps(a1, b1, dot1);
        na0 = _mm256_fmadd_ps(a0, a0, na0);
        na1 = _mm256_fmadd_ps(a1, a1, na1);
    }
    let mut dot = hsum256(_mm256_add_ps(dot0, dot1));
    let mut na_sq = hsum256(_mm256_add_ps(na0, na1));
    for i in (chunks * 16)..n {
        let ai = *a.get_unchecked(i);
        let bi = *b.get_unchecked(i);
        dot += ai * bi;
        na_sq += ai * ai;
    }
    if na_sq == 0.0 || nb_recip == 0.0 {
        return 0.0;
    }
    dot * rsqrt_positive_f32(na_sq) * nb_recip
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l1_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let chunks = n / 16;
    let sign_mask = _mm256_set1_ps(-0.0);
    let (mut acc0, mut acc1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    for i in 0..chunks {
        let o = i * 16;
        let a0 = _mm256_loadu_ps(a.as_ptr().add(o));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(o + 8));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(o));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(o + 8));
        acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a0, b0)));
        acc1 = _mm256_add_ps(acc1, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a1, b1)));
    }
    let mut s = hsum256(_mm256_add_ps(acc0, acc1));
    for i in (chunks * 16)..n {
        s += (*a.get_unchecked(i) - *b.get_unchecked(i)).abs();
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn linf_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let chunks = n / 16;
    let sign_mask = _mm256_set1_ps(-0.0);
    let (mut acc0, mut acc1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    for i in 0..chunks {
        let o = i * 16;
        let a0 = _mm256_loadu_ps(a.as_ptr().add(o));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(o + 8));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(o));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(o + 8));
        acc0 = _mm256_max_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a0, b0)));
        acc1 = _mm256_max_ps(acc1, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a1, b1)));
    }
    let m = _mm256_max_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(m, 1);
    let lo = _mm256_castps256_ps128(m);
    let s = _mm_max_ps(lo, hi);
    let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    let mut mv = _mm_cvtss_f32(_mm_max_ss(s, _mm_shuffle_ps(s, s, 1)));
    for i in (chunks * 16)..n {
        mv = mv.max((*a.get_unchecked(i) - *b.get_unchecked(i)).abs());
    }
    mv
}

// ─────────────────────────────────────────────────────────────────────────────
// DistanceComputer: precomputes per-query state to avoid N recomputations
// ─────────────────────────────────────────────────────────────────────────────

/// Holds the query vector and any precomputed values (e.g. query norm for cosine).
///
/// Construct once per query; call `compute(row)` N times.
/// This avoids recomputing the query norm O(n) times in batch/TopK scans.
pub struct DistanceComputer {
    pub metric: DistanceMetric,
    pub query: Vec<f32>,
    /// Pre-computed ‖query‖², reused by cosine/L2 batch fast paths.
    query_norm_sq: f32,
    /// Pre-computed 1/‖query‖ for cosine metrics (0.0 for others).
    query_norm_recip: f32,
}

impl DistanceComputer {
    pub fn new(metric: DistanceMetric, query: Vec<f32>) -> Self {
        let query_norm_sq = query.iter().map(|x| x * x).sum::<f32>();
        let query_norm_recip = if query_norm_sq > 0.0
            && matches!(
                metric,
                DistanceMetric::CosineSimilarity | DistanceMetric::CosineDistance
            ) {
            1.0 / query_norm_sq.sqrt()
        } else {
            0.0
        };
        Self {
            metric,
            query,
            query_norm_sq,
            query_norm_recip,
        }
    }

    #[inline(always)]
    pub fn compute(&self, a: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::L2 => l2_distance(a, &self.query),
            DistanceMetric::L2Squared => l2_squared(a, &self.query),
            DistanceMetric::L1 => l1_distance(a, &self.query),
            DistanceMetric::LInf => linf_distance(a, &self.query),
            DistanceMetric::InnerProduct => inner_product(a, &self.query),
            DistanceMetric::NegInnerProduct => -inner_product(a, &self.query),
            DistanceMetric::CosineSimilarity => {
                cosine_similarity_fused(a, &self.query, self.query_norm_recip)
            }
            DistanceMetric::CosineDistance => {
                1.0 - cosine_similarity_fused(a, &self.query, self.query_norm_recip)
            }
        }
    }

    #[inline(always)]
    pub fn compute_topk_ordering(&self, a: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::L2 => l2_squared(a, &self.query),
            DistanceMetric::InnerProduct | DistanceMetric::NegInnerProduct => {
                -inner_product(a, &self.query)
            }
            _ => self.compute(a),
        }
    }

    #[inline(always)]
    pub fn finalize_topk_distance(&self, score: f32) -> f32 {
        match self.metric {
            DistanceMetric::L2 => score.sqrt(),
            _ => score,
        }
    }

    #[inline(always)]
    pub fn compute_cosine_topk_ordering_from_dot(&self, dot: f32, row_norm_recip: f32) -> f32 {
        let sim = if row_norm_recip == 0.0 || self.query_norm_recip == 0.0 {
            0.0
        } else {
            dot * row_norm_recip * self.query_norm_recip
        };
        match self.metric {
            DistanceMetric::CosineSimilarity => sim,
            DistanceMetric::CosineDistance => 1.0 - sim,
            _ => unreachable!("cosine helper only valid for cosine metrics"),
        }
    }

    #[inline(always)]
    pub fn compute_dot_topk_ordering_from_dot(&self, dot: f32) -> f32 {
        match self.metric {
            DistanceMetric::InnerProduct => -dot,
            DistanceMetric::NegInnerProduct => -dot,
            _ => unreachable!("dot helper only valid for inner-product metrics"),
        }
    }

    #[inline(always)]
    pub fn compute_l2_topk_ordering_from_dot(&self, dot: f32, row_norm_sq: f32) -> f32 {
        let sq = row_norm_sq + self.query_norm_sq - (2.0 * dot);
        if sq <= 0.0 {
            0.0
        } else {
            sq
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch (Arrow array) distance computation
// ─────────────────────────────────────────────────────────────────────────────

/// Distance metric selector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    L2,
    L2Squared,
    L1,
    LInf,
    InnerProduct,
    NegInnerProduct,
    CosineSimilarity,
    CosineDistance,
}

impl DistanceMetric {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "L2" | "EUCLIDEAN" | "ARRAY_DISTANCE" | "L2_DISTANCE" => Some(Self::L2),
            "L2_SQUARED" | "SQUARED_L2" => Some(Self::L2Squared),
            "L1" | "MANHATTAN" | "L1_DISTANCE" => Some(Self::L1),
            "LINF" | "CHEBYSHEV" | "LINF_DISTANCE" => Some(Self::LInf),
            "DOT" | "INNER_PRODUCT" | "ARRAY_INNER_PRODUCT" => Some(Self::InnerProduct),
            "NEG_INNER_PRODUCT" | "NEGATIVE_INNER_PRODUCT" | "ARRAY_NEGATIVE_INNER_PRODUCT" => {
                Some(Self::NegInnerProduct)
            }
            "COSINE" | "COSINE_SIMILARITY" | "ARRAY_COSINE_SIMILARITY" => {
                Some(Self::CosineSimilarity)
            }
            "COSINE_DISTANCE" | "ARRAY_COSINE_DISTANCE" => Some(Self::CosineDistance),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::L2 => l2_distance(a, b),
            Self::L2Squared => l2_squared(a, b),
            Self::L1 => l1_distance(a, b),
            Self::LInf => linf_distance(a, b),
            Self::InnerProduct => inner_product(a, b),
            Self::NegInnerProduct => -inner_product(a, b),
            Self::CosineSimilarity => {
                let nb_sq: f32 = b.iter().map(|x| x * x).sum();
                if nb_sq == 0.0 {
                    0.0
                } else {
                    cosine_similarity_fused(a, b, 1.0 / nb_sq.sqrt())
                }
            }
            Self::CosineDistance => {
                let nb_sq: f32 = b.iter().map(|x| x * x).sum();
                if nb_sq == 0.0 {
                    1.0
                } else {
                    1.0 - cosine_similarity_fused(a, b, 1.0 / nb_sq.sqrt())
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extract query vector from various Arrow array types
// ─────────────────────────────────────────────────────────────────────────────

/// Try to extract a `Vec<f32>` query vector from an Arrow array.
///
/// Accepts:
/// - `BinaryArray`  (raw f32 LE bytes, first non-null row)
/// - `Float32Array` (first non-null row — rare, used for inter-column distance)  
/// - `Float64Array` (first non-null row, coerced to f32)
/// - `StringArray`  (JSON array literal "[1.0, 2.0, 3.0]", first non-null row)
pub fn extract_query_vector(arr: &dyn Array) -> io::Result<Vec<f32>> {
    if let Some(ba) = arr.as_any().downcast_ref::<BinaryArray>() {
        for i in 0..ba.len() {
            if !ba.is_null(i) {
                let bytes = ba.value(i);
                if bytes.len() % 4 == 0 && !bytes.is_empty() {
                    let floats = bytes_to_f32(bytes).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid binary vector")
                    })?;
                    return Ok(floats.to_vec());
                }
            }
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Empty or invalid binary vector",
        ));
    }

    if let Some(fa) = arr.as_any().downcast_ref::<Float32Array>() {
        for i in 0..fa.len() {
            if !fa.is_null(i) {
                // Whole column is the query vector (unusual — per-row mode)
                let vals: Vec<f32> = (0..fa.len())
                    .filter(|&j| !fa.is_null(j))
                    .map(|j| fa.value(j))
                    .collect();
                return Ok(vals);
            }
        }
    }

    if let Some(da) = arr.as_any().downcast_ref::<Float64Array>() {
        let vals: Vec<f32> = (0..da.len())
            .filter(|&j| !da.is_null(j))
            .map(|j| da.value(j) as f32)
            .collect();
        if !vals.is_empty() {
            return Ok(vals);
        }
    }

    if let Some(sa) = arr.as_any().downcast_ref::<StringArray>() {
        for i in 0..sa.len() {
            if !sa.is_null(i) {
                let s = sa.value(i).trim();
                // Try JSON array "[1.0, 2.0, …]"
                if s.starts_with('[') && s.ends_with(']') {
                    let inner = &s[1..s.len() - 1];
                    let parsed: Result<Vec<f32>, _> = inner
                        .split(',')
                        .map(|tok| tok.trim().parse::<f32>())
                        .collect();
                    if let Ok(v) = parsed {
                        if !v.is_empty() {
                            return Ok(v);
                        }
                    }
                }
            }
        }
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "Cannot extract query vector from argument",
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch distance computation (Arrow in → Arrow out)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `metric(row_vec, query)` for every row in a Binary column.
///
/// Returns a `Float64Array` of the same length with NULLs for NULL/invalid rows.
pub fn batch_distance(
    col: &dyn Array,
    query: &[f32],
    metric: DistanceMetric,
) -> io::Result<ArrayRef> {
    // ── Binary column path (primary: raw f32 bytes) ──────────────────────────
    if let Some(ba) = col.as_any().downcast_ref::<BinaryArray>() {
        let dim = query.len();
        let expected_bytes = dim * 4;
        let distances: Vec<Option<f64>> = (0..ba.len())
            .into_par_iter()
            .map(|i| {
                if ba.is_null(i) {
                    return None;
                }
                let bytes = ba.value(i);
                if bytes.len() != expected_bytes {
                    return None;
                }
                let vec = bytes_to_f32(bytes)?;
                Some(metric.compute(vec.as_ref(), query) as f64)
            })
            .collect();
        return Ok(Arc::new(Float64Array::from(distances)) as ArrayRef);
    }

    // ── FixedSizeList<Float32> column path (Float16List decoded → f32) ────────
    if let Some(fsl) = col
        .as_any()
        .downcast_ref::<arrow::array::FixedSizeListArray>()
    {
        use arrow::array::Float32Array;
        let dim = query.len() as i32;
        if fsl.value_length() != dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "array_distance: FixedSizeList dim {} != query dim {}",
                    fsl.value_length(),
                    dim
                ),
            ));
        }
        let distances: Vec<Option<f64>> = (0..fsl.len())
            .into_par_iter()
            .map(|i| {
                if fsl.is_null(i) {
                    return None;
                }
                let vals = fsl.value(i);
                let f32arr = vals.as_any().downcast_ref::<Float32Array>()?;
                let slice: Vec<f32> = (0..f32arr.len()).map(|j| f32arr.value(j)).collect();
                Some(metric.compute(&slice, query) as f64)
            })
            .collect();
        return Ok(Arc::new(Float64Array::from(distances)) as ArrayRef);
    }

    // ── String column path ("[1.0, 2.0, …]" per row) ─────────────────────────
    if let Some(sa) = col.as_any().downcast_ref::<StringArray>() {
        let distances: Vec<Option<f64>> = (0..sa.len())
            .into_par_iter()
            .map(|i| {
                if sa.is_null(i) {
                    return None;
                }
                let s = sa.value(i).trim();
                if !(s.starts_with('[') && s.ends_with(']')) {
                    return None;
                }
                let inner = &s[1..s.len() - 1];
                let parsed: Result<Vec<f32>, _> =
                    inner.split(',').map(|t| t.trim().parse::<f32>()).collect();
                let vec = parsed.ok()?;
                if vec.len() != query.len() {
                    return None;
                }
                Some(metric.compute(&vec, query) as f64)
            })
            .collect();
        return Ok(Arc::new(Float64Array::from(distances)) as ArrayRef);
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "array_distance: first argument must be a Binary (vector), FixedSizeList, or String column",
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// TopK helper: partial sort with a binary max-heap
// ─────────────────────────────────────────────────────────────────────────────

/// Given a `Float64Array` of distances (some possibly NULL),
/// return the indices of the `k` smallest non-null values in ascending order.
///
/// This is O(n log k) — much faster than full sort for small k.
pub fn topk_indices_asc(distances: &Float64Array, k: usize) -> Vec<usize> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    if k == 0 {
        return vec![];
    }

    // Max-heap of (dist_bits, idx) — we keep the k smallest
    #[derive(Copy, Clone)]
    struct Entry(u64, usize); // (distance bits for ordering, original row idx)

    impl PartialEq for Entry {
        fn eq(&self, o: &Self) -> bool {
            self.0 == o.0
        }
    }
    impl Eq for Entry {}
    impl PartialOrd for Entry {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Entry {
        fn cmp(&self, o: &Self) -> Ordering {
            self.0.cmp(&o.0)
        }
    }

    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k + 1);

    for i in 0..distances.len() {
        if distances.is_null(i) {
            continue;
        }
        let d = distances.value(i);
        let bits = d.to_bits();
        if heap.len() < k {
            heap.push(Entry(bits, i));
        } else if let Some(&Entry(top_bits, _)) = heap.peek() {
            if bits < top_bits {
                heap.pop();
                heap.push(Entry(bits, i));
            }
        }
    }

    let mut result: Vec<(f64, usize)> = heap
        .into_iter()
        .map(|Entry(b, i)| (f64::from_bits(b), i))
        .collect();
    result.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    result.into_iter().map(|(_, i)| i).collect()
}

/// Given a `Float64Array` of distances (some possibly NULL),
/// return the indices of the `k` *largest* non-null values in descending order.
/// (Used for cosine_similarity / inner_product ORDER BY … DESC LIMIT k.)
pub fn topk_indices_desc(distances: &Float64Array, k: usize) -> Vec<usize> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    if k == 0 {
        return vec![];
    }

    #[derive(Copy, Clone)]
    struct Entry(u64, usize); // negated bits to turn max-heap into min-heap

    impl PartialEq for Entry {
        fn eq(&self, o: &Self) -> bool {
            self.0 == o.0
        }
    }
    impl Eq for Entry {}
    impl PartialOrd for Entry {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Entry {
        fn cmp(&self, o: &Self) -> Ordering {
            self.0.cmp(&o.0)
        }
    }

    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k + 1);

    for i in 0..distances.len() {
        if distances.is_null(i) {
            continue;
        }
        let d = distances.value(i);
        // negate to turn max into min ordering in the heap
        let neg_bits = (-d).to_bits();
        if heap.len() < k {
            heap.push(Entry(neg_bits, i));
        } else if let Some(&Entry(top_bits, _)) = heap.peek() {
            if neg_bits < top_bits {
                heap.pop();
                heap.push(Entry(neg_bits, i));
            }
        }
    }

    let mut result: Vec<(f64, usize)> = heap
        .into_iter()
        .map(|Entry(b, i)| (-f64::from_bits(b), i))
        .collect();
    result.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    result.into_iter().map(|(_, i)| i).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused TopK: single-pass compute + heap, O(n log k)
// ─────────────────────────────────────────────────────────────────────────────

/// Single-pass TopK directly on a `BinaryArray` of stored vectors.
///
/// Fuses SIMD distance computation with max-heap maintenance in **one**
/// sequential pass — no intermediate `Float64Array` allocation for all rows.
/// O(n log k) time, O(k) extra space.
///
/// Returns `Vec<(row_index, f32_distance)>` sorted **ascending** (nearest first).
pub fn topk_heap_direct(
    col: &BinaryArray,
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
) -> Vec<(usize, f32)> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    if k == 0 || col.len() == 0 || query.is_empty() {
        return vec![];
    }
    let k_capped = k.min(col.len());
    let expected_bytes = query.len() * 4;
    let computer = DistanceComputer::new(metric, query.to_vec());
    let signed_topk_ordering = matches!(
        metric,
        DistanceMetric::InnerProduct | DistanceMetric::NegInnerProduct
    );

    // Max-heap of (ordering_key, f32_bits, row_idx).
    #[derive(Copy, Clone)]
    struct Entry(u32, u32, usize);
    impl PartialEq for Entry {
        fn eq(&self, o: &Self) -> bool {
            self.0 == o.0
        }
    }
    impl Eq for Entry {}
    impl PartialOrd for Entry {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Entry {
        fn cmp(&self, o: &Self) -> Ordering {
            self.0.cmp(&o.0)
        }
    }

    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k_capped + 1);

    for i in 0..col.len() {
        if col.is_null(i) {
            continue;
        }
        let bytes = col.value(i);
        if bytes.len() != expected_bytes {
            continue;
        }
        let Some(vec) = bytes_to_f32(bytes) else {
            continue;
        };
        let dist = computer.compute_topk_ordering(vec.as_ref());
        let bits = dist.to_bits();
        let key = if signed_topk_ordering {
            f32_ascending_key_from_bits(bits)
        } else {
            bits
        };
        if heap.len() < k_capped {
            heap.push(Entry(key, bits, i));
        } else if let Some(&Entry(top_key, _, _)) = heap.peek() {
            if key < top_key {
                heap.pop();
                heap.push(Entry(key, bits, i));
            }
        }
    }

    let mut result: Vec<(usize, f32)> = heap
        .into_iter()
        .map(|Entry(_, b, i)| (i, computer.finalize_topk_distance(f32::from_bits(b))))
        .collect();
    result.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel TopK: Rayon fold+reduce, O(n/T log k) per thread
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Copy, Clone)]
struct TopKEntry(u32, u32, usize);

#[inline(always)]
fn f32_ascending_key_from_bits(bits: u32) -> u32 {
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

#[inline(always)]
fn insert_topk_entry(entries: &mut Vec<TopKEntry>, k: usize, key: u32, bits: u32, idx: usize) {
    if k == 0 {
        return;
    }
    if entries.len() == k && key >= entries[entries.len() - 1].0 {
        return;
    }

    entries.push(TopKEntry(key, bits, idx));
    let mut pos = entries.len() - 1;
    while pos > 0 && key < entries[pos - 1].0 {
        entries[pos] = entries[pos - 1];
        pos -= 1;
    }
    entries[pos] = TopKEntry(key, bits, idx);
    if entries.len() > k {
        entries.pop();
    }
}

#[inline(always)]
fn insert_topk_bits(entries: &mut Vec<TopKEntry>, k: usize, bits: u32, idx: usize) {
    insert_topk_entry(entries, k, bits, bits, idx);
}

#[inline(always)]
fn insert_topk_score(entries: &mut Vec<TopKEntry>, k: usize, score: f32, idx: usize) {
    let bits = score.to_bits();
    insert_topk_entry(entries, k, f32_ascending_key_from_bits(bits), bits, idx);
}

#[inline(always)]
fn topk_entries_to_result(
    entries: Vec<TopKEntry>,
    computer: &DistanceComputer,
) -> Vec<(usize, f32)> {
    entries
        .into_iter()
        .map(|TopKEntry(_, bits, idx)| (idx, computer.finalize_topk_distance(f32::from_bits(bits))))
        .collect()
}

/// Parallel TopK on a `BinaryArray` using Rayon.
///
/// Uses `DistanceComputer` which has pre-computed query-norm for cosine metrics.
/// Splits the array into exactly T = num_threads contiguous chunks; each thread
/// maintains one local max-heap of size k, then T heaps are merged.
/// Inner loop uses raw buffer pointers to avoid per-row `col.value(i)` overhead.
///
/// Returns `Vec<(row_index, f32_distance)>` sorted **ascending** (nearest first).
pub fn topk_heap_direct_parallel(
    col: &BinaryArray,
    computer: &DistanceComputer,
    k: usize,
) -> Vec<(usize, f32)> {
    use rayon::prelude::*;

    let n = col.len();
    if k == 0 || n == 0 || computer.query.is_empty() {
        return vec![];
    }
    let k_capped = k.min(n);
    let dim = computer.query.len();
    let expected_bytes = dim * 4;

    // Grab raw buffer slices ONCE — avoid Arc deref + bounds checks inside the hot loop.
    // SAFETY: these references live for the duration of the function, col is immutable.
    let values: &[u8] = col.values().as_slice();
    let offsets: &[i32] = col.offsets().as_ref(); // len = n+1
                                                  // Null bitmap: Option<&[u8]> pointing into Arrow's validity buffer.
    let null_bytes: Option<&[u8]> = col.nulls().map(|nb| nb.buffer().as_slice());
    let signed_topk_ordering = matches!(
        computer.metric,
        DistanceMetric::InnerProduct | DistanceMetric::NegInnerProduct
    );

    // Capture raw pointers as usize so they are Send across Rayon threads.
    // SAFETY: values, offsets, null_bytes are all derived from `col` which is
    //         immutable and outlives the parallel section below.
    let val_ptr = values.as_ptr() as usize;
    let val_len = values.len();
    let off_ptr = offsets.as_ptr() as usize;
    let null_ptr: Option<(usize, usize)> = null_bytes.map(|nb| (nb.as_ptr() as usize, nb.len()));

    // Split into exactly T chunks (one per Rayon worker thread).
    let t = rayon::current_num_threads().max(1);
    let chunk_size = (n + t - 1) / t;

    let per_chunk: Vec<Vec<TopKEntry>> = (0..t)
        .into_par_iter()
        .map(|tid| {
            let start = tid * chunk_size;
            if start >= n {
                return vec![];
            }
            let end = (start + chunk_size).min(n);

            // SAFETY: pointers are valid for `col`'s lifetime (enforced by borrow above).
            let values = unsafe { std::slice::from_raw_parts(val_ptr as *const u8, val_len) };
            let offsets = unsafe { std::slice::from_raw_parts(off_ptr as *const i32, n + 1) };

            let mut entries: Vec<TopKEntry> = Vec::with_capacity(k_capped);

            for i in start..end {
                // Null check via raw bitmap (bit i).
                if let Some((nb_ptr, nb_len)) = null_ptr {
                    let byte_idx = i / 8;
                    if byte_idx < nb_len {
                        // SAFETY: byte_idx < nb_len.
                        let byte = unsafe { *(nb_ptr as *const u8).add(byte_idx) };
                        if (byte >> (i & 7)) & 1 == 0 {
                            continue;
                        }
                    }
                }

                // SAFETY: offsets has n+1 elements; i and i+1 are within bounds.
                let off_s = unsafe { *offsets.get_unchecked(i) } as usize;
                let off_e = unsafe { *offsets.get_unchecked(i + 1) } as usize;
                if off_e - off_s != expected_bytes {
                    continue;
                }

                // SAFETY: off_s..off_e is within values (enforced by BinaryArray invariants).
                let bytes = unsafe { values.get_unchecked(off_s..off_e) };
                let Some(vec) = bytes_to_f32(bytes) else {
                    continue;
                };

                let score = computer.compute_topk_ordering(vec.as_ref());
                if signed_topk_ordering {
                    insert_topk_score(&mut entries, k_capped, score, i);
                } else {
                    insert_topk_bits(&mut entries, k_capped, score.to_bits(), i);
                }
            }
            entries
        })
        .collect();

    // Merge T small top-k lists into final top-k (T is small, e.g. 8-16).
    let mut final_entries: Vec<TopKEntry> = Vec::with_capacity(k_capped);
    for chunk in per_chunk {
        for TopKEntry(key, bits, idx) in chunk {
            insert_topk_entry(&mut final_entries, k_capped, key, bits, idx);
        }
    }
    topk_entries_to_result(final_entries, computer)
}

/// Parallel TopK on a `FixedSizeListArray<Float32>` using Rayon.
///
/// Compared to the `BinaryArray` version, the inner loop is simpler:
/// the float data is stored contiguously with stride = `dim`, so there
/// are no per-row offset lookups.  This eliminates one pointer dereference
/// and the `off_e - off_s != expected_bytes` size-check per row.
///
/// Returns `Vec<(row_index, f32_distance)>` sorted **ascending** (nearest first).
pub fn topk_heap_direct_parallel_fixed(
    col: &arrow::array::FixedSizeListArray,
    computer: &DistanceComputer,
    k: usize,
) -> Vec<(usize, f32)> {
    let n = col.len();
    if k == 0 || n == 0 || computer.query.is_empty() {
        return vec![];
    }
    let dim = computer.query.len();

    // The values child is a flat Float32Array with n*dim elements.
    let values_child = col.values();
    let float_arr = values_child
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .expect("FixedSizeList child must be Float32Array");
    let floats: &[f32] = float_arr.values().as_ref();
    topk_heap_on_floats(floats, n, dim, computer, k)
}

/// Core parallel TopK on a raw contiguous `&[f32]` (no Arrow, no allocation).
/// `floats` has length `n_rows * dim`. Zero-copy when called with an mmap slice.
/// Returns `Vec<(row_index, f32_distance)>` sorted ascending (nearest first).
#[inline(always)]
fn topk_heap_on_floats_impl(
    floats: &[f32],
    n_rows: usize,
    dim: usize,
    computer: &DistanceComputer,
    k: usize,
) -> Vec<(usize, f32)> {
    use rayon::prelude::*;

    if k == 0 || n_rows == 0 || dim == 0 {
        return vec![];
    }
    let k_capped = k.min(n_rows);
    let signed_topk_ordering = matches!(
        computer.metric,
        DistanceMetric::InnerProduct | DistanceMetric::NegInnerProduct
    );

    let t = rayon::current_num_threads().max(1);
    let chunk_size = (n_rows + t - 1) / t;

    if signed_topk_ordering {
        let query = computer.query.as_slice();
        let per_chunk: Vec<Vec<TopKEntry>> = (0..t)
            .into_par_iter()
            .map(|tid| {
                let start = tid * chunk_size;
                if start >= n_rows {
                    return vec![];
                }
                let end = (start + chunk_size).min(n_rows);
                let mut entries: Vec<TopKEntry> = Vec::with_capacity(k_capped);
                for i in start..end {
                    let off = i * dim;
                    if off + dim > floats.len() {
                        break;
                    }
                    let vec = unsafe { floats.get_unchecked(off..off + dim) };
                    insert_topk_score(&mut entries, k_capped, -inner_product(vec, query), i);
                }
                entries
            })
            .collect();

        let mut final_entries: Vec<TopKEntry> = Vec::with_capacity(k_capped);
        for chunk in per_chunk {
            for TopKEntry(key, bits, idx) in chunk {
                insert_topk_entry(&mut final_entries, k_capped, key, bits, idx);
            }
        }
        return topk_entries_to_result(final_entries, computer);
    }

    let per_chunk: Vec<Vec<TopKEntry>> = (0..t)
        .into_par_iter()
        .map(|tid| {
            let start = tid * chunk_size;
            if start >= n_rows {
                return vec![];
            }
            let end = (start + chunk_size).min(n_rows);
            let mut entries: Vec<TopKEntry> = Vec::with_capacity(k_capped);
            for i in start..end {
                let off = i * dim;
                if off + dim > floats.len() {
                    break;
                }
                let vec = unsafe { floats.get_unchecked(off..off + dim) };
                let score = computer.compute_topk_ordering(vec);
                insert_topk_bits(&mut entries, k_capped, score.to_bits(), i);
            }
            entries
        })
        .collect();

    let mut final_entries: Vec<TopKEntry> = Vec::with_capacity(k_capped);
    for chunk in per_chunk {
        for TopKEntry(key, bits, idx) in chunk {
            insert_topk_entry(&mut final_entries, k_capped, key, bits, idx);
        }
    }
    topk_entries_to_result(final_entries, computer)
}

pub fn topk_heap_on_floats(
    floats: &[f32],
    n_rows: usize,
    dim: usize,
    computer: &DistanceComputer,
    k: usize,
) -> Vec<(usize, f32)> {
    if k == 0 || n_rows == 0 || dim == 0 || computer.query.is_empty() {
        return vec![];
    }
    topk_heap_on_floats_impl(floats, n_rows, dim, computer, k)
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch TopK: N queries in parallel, sequential inner scan per query
// ─────────────────────────────────────────────────────────────────────────────

/// Single-query sequential TopK on a raw contiguous `&[f32]` (no Rayon overhead).
/// Used by `batch_topk_on_floats` where outer parallelism is over N queries.
/// Returns `Vec<(row_index, f32_distance)>` sorted ascending (nearest first).
#[inline(always)]
fn topk_sequential_on_floats_impl(
    floats: &[f32],
    n_rows: usize,
    dim: usize,
    computer: &DistanceComputer,
    k: usize,
) -> Vec<(usize, f32)> {
    let mut entries: Vec<TopKEntry> = Vec::with_capacity(k);
    let signed_topk_ordering = matches!(
        computer.metric,
        DistanceMetric::InnerProduct | DistanceMetric::NegInnerProduct
    );
    if signed_topk_ordering {
        let query = computer.query.as_slice();
        for i in 0..n_rows {
            let off = i * dim;
            if off + dim > floats.len() {
                break;
            }
            let vec = unsafe { floats.get_unchecked(off..off + dim) };
            insert_topk_score(&mut entries, k, -inner_product(vec, query), i);
        }
        return topk_entries_to_result(entries, computer);
    }
    for i in 0..n_rows {
        let off = i * dim;
        if off + dim > floats.len() {
            break;
        }
        let vec = unsafe { floats.get_unchecked(off..off + dim) };
        let score = computer.compute_topk_ordering(vec);
        insert_topk_bits(&mut entries, k, score.to_bits(), i);
    }
    topk_entries_to_result(entries, computer)
}

fn topk_sequential_on_floats(
    floats: &[f32],
    n_rows: usize,
    dim: usize,
    computer: &DistanceComputer,
    k: usize,
) -> Vec<(usize, f32)> {
    topk_sequential_on_floats_impl(floats, n_rows, dim, computer, k)
}

/// Row-blocked multi-query TopK on a raw contiguous `&[f32]`.
///
/// Each Rayon worker scans a disjoint row chunk once and maintains one local
/// heap per query. This avoids reading the full vector table once per query and
/// works well for small/medium batch sizes where memory traffic dominates.
fn batch_topk_blocked_on_floats(
    floats: &[f32],
    n_rows: usize,
    dim: usize,
    computers: &[DistanceComputer],
    k: usize,
) -> Vec<Vec<(usize, f32)>> {
    use rayon::prelude::*;

    let n_queries = computers.len();
    debug_assert!(n_queries <= MAX_BLOCKED_BATCH_QUERIES);
    if n_queries == 0 || k == 0 || n_rows == 0 || dim == 0 {
        return vec![vec![]; n_queries];
    }
    let k_capped = k.min(n_rows);
    let metric = computers[0].metric;
    let use_shared_dot_kernel = matches!(
        metric,
        DistanceMetric::CosineSimilarity
            | DistanceMetric::CosineDistance
            | DistanceMetric::InnerProduct
            | DistanceMetric::NegInnerProduct
            | DistanceMetric::L2
            | DistanceMetric::L2Squared
            | DistanceMetric::L1
            | DistanceMetric::LInf
    );

    if !use_shared_dot_kernel {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(Copy, Clone)]
        struct Entry(u32, usize);
        impl PartialEq for Entry {
            fn eq(&self, o: &Self) -> bool {
                self.0 == o.0
            }
        }
        impl Eq for Entry {}
        impl PartialOrd for Entry {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Entry {
            fn cmp(&self, o: &Self) -> Ordering {
                self.0.cmp(&o.0)
            }
        }

        #[inline(always)]
        fn push_entry(heap: &mut BinaryHeap<Entry>, k_capped: usize, score: f32, idx: usize) {
            let bits = score.to_bits();
            if heap.len() < k_capped {
                heap.push(Entry(bits, idx));
            } else if let Some(&Entry(top, _)) = heap.peek() {
                if bits < top {
                    heap.pop();
                    heap.push(Entry(bits, idx));
                }
            }
        }

        let t = rayon::current_num_threads().max(1);
        let chunk_size = (n_rows + t - 1) / t;

        let per_chunk: Vec<Vec<Vec<(usize, f32)>>> = (0..t)
            .into_par_iter()
            .map(|tid| {
                let start = tid * chunk_size;
                if start >= n_rows {
                    return vec![vec![]; n_queries];
                }
                let end = (start + chunk_size).min(n_rows);
                let mut heaps: Vec<BinaryHeap<Entry>> = (0..n_queries)
                    .map(|_| BinaryHeap::with_capacity(k_capped + 1))
                    .collect();

                for i in start..end {
                    let off = i * dim;
                    if off + dim > floats.len() {
                        break;
                    }
                    let row = unsafe { floats.get_unchecked(off..off + dim) };
                    for (heap, computer) in heaps.iter_mut().zip(computers.iter()) {
                        push_entry(heap, k_capped, computer.compute_topk_ordering(row), i);
                    }
                }

                heaps
                    .into_iter()
                    .map(|heap| {
                        heap.into_iter()
                            .map(|Entry(bits, idx)| (idx, f32::from_bits(bits)))
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let mut final_heaps: Vec<BinaryHeap<Entry>> = (0..n_queries)
            .map(|_| BinaryHeap::with_capacity(k_capped + 1))
            .collect();
        for chunk in per_chunk {
            for (qi, entries) in chunk.into_iter().enumerate() {
                let final_heap = &mut final_heaps[qi];
                for (idx, score) in entries {
                    push_entry(final_heap, k_capped, score, idx);
                }
            }
        }

        return computers
            .iter()
            .zip(final_heaps.into_iter())
            .map(|(computer, heap)| {
                let mut result: Vec<(usize, f32)> = heap
                    .into_iter()
                    .map(|Entry(bits, idx)| {
                        (idx, computer.finalize_topk_distance(f32::from_bits(bits)))
                    })
                    .collect();
                result.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                result
            })
            .collect();
    }

    let t = rayon::current_num_threads().max(1);
    let chunk_size = (n_rows + t - 1) / t;
    debug_assert!(computers
        .iter()
        .all(|computer| computer.metric == computers[0].metric));
    let cosine_metric = matches!(
        metric,
        DistanceMetric::CosineSimilarity | DistanceMetric::CosineDistance
    );
    let dot_metric = matches!(
        metric,
        DistanceMetric::InnerProduct | DistanceMetric::NegInnerProduct
    );
    let l2_metric = matches!(metric, DistanceMetric::L2 | DistanceMetric::L2Squared);
    let l1_metric = matches!(metric, DistanceMetric::L1);
    let linf_metric = matches!(metric, DistanceMetric::LInf);
    let query_slices: Vec<&[f32]> = computers
        .iter()
        .map(|computer| computer.query.as_slice())
        .collect();

    let per_chunk: Vec<Vec<Vec<TopKEntry>>> = (0..t)
        .into_par_iter()
        .map(|tid| {
            let start = tid * chunk_size;
            if start >= n_rows {
                return vec![vec![]; n_queries];
            }
            let end = (start + chunk_size).min(n_rows);
            let mut entries_per_query: Vec<Vec<TopKEntry>> = (0..n_queries)
                .map(|_| Vec::with_capacity(k_capped))
                .collect();
            let mut metric_scores = [0.0f32; MAX_BLOCKED_BATCH_QUERIES];

            for i in start..end {
                let off = i * dim;
                if off + dim > floats.len() {
                    break;
                }
                let row = unsafe { floats.get_unchecked(off..off + dim) };
                let row_norm_sq = if cosine_metric || l2_metric {
                    l2_norm_squared(row)
                } else {
                    0.0
                };
                let row_norm_recip = if cosine_metric && row_norm_sq > 0.0 {
                    rsqrt_positive_f32(row_norm_sq)
                } else {
                    0.0
                };
                if cosine_metric || dot_metric || l2_metric {
                    batch_inner_products(row, &query_slices, &mut metric_scores);
                } else if l1_metric {
                    batch_l1_distances(row, &query_slices, &mut metric_scores);
                } else if linf_metric {
                    batch_linf_distances(row, &query_slices, &mut metric_scores);
                } else {
                    unreachable!();
                }
                for (qi, (entries, computer)) in entries_per_query
                    .iter_mut()
                    .zip(computers.iter())
                    .enumerate()
                {
                    let score = if cosine_metric {
                        computer.compute_cosine_topk_ordering_from_dot(
                            metric_scores[qi],
                            row_norm_recip,
                        )
                    } else if dot_metric {
                        computer.compute_dot_topk_ordering_from_dot(metric_scores[qi])
                    } else if l2_metric {
                        computer.compute_l2_topk_ordering_from_dot(metric_scores[qi], row_norm_sq)
                    } else if l1_metric || linf_metric {
                        metric_scores[qi]
                    } else {
                        unreachable!()
                    };
                    if dot_metric {
                        insert_topk_score(entries, k_capped, score, i);
                    } else {
                        insert_topk_bits(entries, k_capped, score.to_bits(), i);
                    }
                }
            }
            entries_per_query
        })
        .collect();

    let mut final_entries: Vec<Vec<TopKEntry>> = (0..n_queries)
        .map(|_| Vec::with_capacity(k_capped))
        .collect();
    for chunk in per_chunk {
        for (qi, entries) in chunk.into_iter().enumerate() {
            let final_query_entries = &mut final_entries[qi];
            for TopKEntry(key, bits, idx) in entries {
                insert_topk_entry(final_query_entries, k_capped, key, bits, idx);
            }
        }
    }

    computers
        .iter()
        .zip(final_entries.into_iter())
        .map(|(computer, entries)| topk_entries_to_result(entries, computer))
        .collect()
}

/// Adaptive batch TopK on a raw contiguous `&[f32]` database.
///
/// Chooses the best parallelism strategy based on N vs num_threads:
/// - **N < num_threads**: sequential over queries, each using the full Rayon pool
///   for row-level parallelism (same as N individual `topk_heap_on_floats` calls).
/// - **N ≥ num_threads**: outer Rayon over queries, sequential inner per query;
///   avoids nested Rayon contention and scales well when N >> threads.
///
/// `floats`:  `n_rows × dim`, row-major — the stored database vectors.
/// `queries`: `n_queries × dim`, row-major — the query vectors.
///
/// Returns `Vec<Vec<(row_index, f32_distance)>>` of length `n_queries`,
/// each inner Vec sorted ascending (nearest first) and capped at `k`.
pub fn batch_topk_on_floats(
    floats: &[f32],
    n_rows: usize,
    dim: usize,
    queries: &[f32],
    n_queries: usize,
    k: usize,
    metric: DistanceMetric,
) -> Vec<Vec<(usize, f32)>> {
    use rayon::prelude::*;

    if k == 0 || n_rows == 0 || dim == 0 || n_queries == 0 {
        return vec![vec![]; n_queries];
    }
    let k_capped = k.min(n_rows);
    let t = rayon::current_num_threads().max(1);
    let computers: Vec<DistanceComputer> = (0..n_queries)
        .map(|qi| {
            let start = qi * dim;
            let end = start + dim;
            DistanceComputer::new(metric, queries[start..end].to_vec())
        })
        .collect();

    if n_queries > 1 && n_queries <= MAX_BLOCKED_BATCH_QUERIES {
        return batch_topk_blocked_on_floats(floats, n_rows, dim, &computers, k_capped);
    }

    if n_queries < t {
        // N < threads: inner-parallel per query (full Rayon pool on each query's rows).
        // Same perf as N individual topk_heap_on_floats calls, but all within one
        // py.allow_threads scope → saves N-1 Python→Rust transitions + N-1 _id reads.
        computers
            .iter()
            .map(|computer| topk_heap_on_floats(floats, n_rows, dim, computer, k_capped))
            .collect()
    } else {
        // N >= threads: outer-parallel over queries, sequential inner per query.
        // Each Rayon thread handles N/T queries sequentially — no nested contention.
        computers
            .par_iter()
            .map(|computer| topk_sequential_on_floats(floats, n_rows, dim, computer, k_capped))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Native f16 TopK: decode f16→f32 per-element during distance computation.
// Halves memory bandwidth vs a fully-decoded f32 buffer (50MB vs 100MB).
// ─────────────────────────────────────────────────────────────────────────────

/// IEEE-754 half-precision → f32 (branchless; valid for normal numbers).
#[inline(always)]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let b = bits as u32;
    f32::from_bits(
        ((b & 0x8000) << 16) | (((b >> 10) & 0x1F).wrapping_add(112) << 23) | ((b & 0x03FF) << 13),
    )
}
#[inline(always)]
fn f16u(d: &[u8], i: usize) -> f32 {
    f16_bits_to_f32(u16::from_le_bytes([d[i * 2], d[i * 2 + 1]]))
}

// ── aarch64 NEON + fp16 (hardware FCVTL via inline asm, stable Rust) ────────
/// Load 8 packed f16 LE bytes and return two float32x4_t halves.
/// Uses FCVTL/FCVTL2 ARM instructions (ARMv8.2-A fp16, always on Apple Silicon).
/// No stdarch_neon_f16 unstable feature needed — conversion done via inline asm.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn f16x8_to_f32pair(
    ptr: *const u8,
) -> (
    std::arch::aarch64::float32x4_t,
    std::arch::aarch64::float32x4_t,
) {
    use std::arch::aarch64::*;
    let src: uint16x8_t = vld1q_u16(ptr as *const u16);
    let lo: float32x4_t;
    let hi: float32x4_t;
    std::arch::asm!(
        "fcvtl  {lo}.4s, {src}.4h",
        "fcvtl2 {hi}.4s, {src}.8h",
        src = in(vreg) src,
        lo  = out(vreg) lo,
        hi  = out(vreg) hi,
        options(nostack, pure, nomem)
    );
    (lo, hi)
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn l2sq_f16_neon(d: &[u8], q: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let (dim, n) = (q.len(), q.len() / 8);
    let (mut a, mut b) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    for i in 0..n {
        let (x0, x1) = f16x8_to_f32pair(d.as_ptr().add(i * 16));
        let (q0, q1) = (
            vld1q_f32(q.as_ptr().add(i * 8)),
            vld1q_f32(q.as_ptr().add(i * 8 + 4)),
        );
        let (d0, d1) = (vsubq_f32(q0, x0), vsubq_f32(q1, x1));
        a = vfmaq_f32(a, d0, d0);
        b = vfmaq_f32(b, d1, d1);
    }
    let mut s = vaddvq_f32(vaddq_f32(a, b));
    for i in n * 8..dim {
        let e = q[i] - f16u(d, i);
        s += e * e;
    }
    s
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn dot_f16_neon(d: &[u8], q: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let (dim, n) = (q.len(), q.len() / 8);
    let (mut a, mut b) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    for i in 0..n {
        let (x0, x1) = f16x8_to_f32pair(d.as_ptr().add(i * 16));
        let (q0, q1) = (
            vld1q_f32(q.as_ptr().add(i * 8)),
            vld1q_f32(q.as_ptr().add(i * 8 + 4)),
        );
        a = vfmaq_f32(a, x0, q0);
        b = vfmaq_f32(b, x1, q1);
    }
    let mut s = vaddvq_f32(vaddq_f32(a, b));
    for i in n * 8..dim {
        s += f16u(d, i) * q[i];
    }
    s
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn cosine_f16_neon(d: &[u8], q: &[f32], qnr: f32) -> f32 {
    use std::arch::aarch64::*;
    let (dim, n) = (q.len(), q.len() / 8);
    let (mut da, mut db, mut na, mut nb) = (
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
    );
    for i in 0..n {
        let (x0, x1) = f16x8_to_f32pair(d.as_ptr().add(i * 16));
        let (q0, q1) = (
            vld1q_f32(q.as_ptr().add(i * 8)),
            vld1q_f32(q.as_ptr().add(i * 8 + 4)),
        );
        da = vfmaq_f32(da, x0, q0);
        db = vfmaq_f32(db, x1, q1);
        na = vfmaq_f32(na, x0, x0);
        nb = vfmaq_f32(nb, x1, x1);
    }
    let (mut dot, mut ns) = (vaddvq_f32(vaddq_f32(da, db)), vaddvq_f32(vaddq_f32(na, nb)));
    for i in n * 8..dim {
        let x = f16u(d, i);
        dot += x * q[i];
        ns += x * x;
    }
    if ns == 0.0 || qnr == 0.0 {
        return 0.0;
    }
    dot * rsqrt_positive_f32(ns) * qnr
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn l1_f16_neon(d: &[u8], q: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let (dim, n) = (q.len(), q.len() / 8);
    let (mut a, mut b) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    for i in 0..n {
        let (x0, x1) = f16x8_to_f32pair(d.as_ptr().add(i * 16));
        let (q0, q1) = (
            vld1q_f32(q.as_ptr().add(i * 8)),
            vld1q_f32(q.as_ptr().add(i * 8 + 4)),
        );
        a = vaddq_f32(a, vabsq_f32(vsubq_f32(q0, x0)));
        b = vaddq_f32(b, vabsq_f32(vsubq_f32(q1, x1)));
    }
    let mut s = vaddvq_f32(vaddq_f32(a, b));
    for i in n * 8..dim {
        s += (q[i] - f16u(d, i)).abs();
    }
    s
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn linf_f16_neon(d: &[u8], q: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let (dim, n) = (q.len(), q.len() / 8);
    let (mut a, mut b) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    for i in 0..n {
        let (x0, x1) = f16x8_to_f32pair(d.as_ptr().add(i * 16));
        let (q0, q1) = (
            vld1q_f32(q.as_ptr().add(i * 8)),
            vld1q_f32(q.as_ptr().add(i * 8 + 4)),
        );
        a = vmaxq_f32(a, vabsq_f32(vsubq_f32(q0, x0)));
        b = vmaxq_f32(b, vabsq_f32(vsubq_f32(q1, x1)));
    }
    let mut m = vmaxvq_f32(vmaxq_f32(a, b));
    for i in n * 8..dim {
        m = m.max((q[i] - f16u(d, i)).abs());
    }
    m
}

// ── x86_64 AVX2 + F16C + FMA ─────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn l2sq_f16_avx(d: &[u8], q: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let (dim, n) = (q.len(), q.len() / 16);
    let (mut a, mut b) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    for i in 0..n {
        let p = d.as_ptr().add(i * 32) as *const __m128i;
        let (x0, x1) = (
            _mm256_cvtph_ps(_mm_loadu_si128(p)),
            _mm256_cvtph_ps(_mm_loadu_si128(p.add(1))),
        );
        let (q0, q1) = (
            _mm256_loadu_ps(q.as_ptr().add(i * 16)),
            _mm256_loadu_ps(q.as_ptr().add(i * 16 + 8)),
        );
        let (d0, d1) = (_mm256_sub_ps(q0, x0), _mm256_sub_ps(q1, x1));
        a = _mm256_fmadd_ps(d0, d0, a);
        b = _mm256_fmadd_ps(d1, d1, b);
    }
    let mut s = hsum256(_mm256_add_ps(a, b));
    for i in n * 16..dim {
        let e = q[i] - f16u(d, i);
        s += e * e;
    }
    s
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn dot_f16_avx(d: &[u8], q: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let (dim, n) = (q.len(), q.len() / 16);
    let (mut a, mut b) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    for i in 0..n {
        let p = d.as_ptr().add(i * 32) as *const __m128i;
        let (x0, x1) = (
            _mm256_cvtph_ps(_mm_loadu_si128(p)),
            _mm256_cvtph_ps(_mm_loadu_si128(p.add(1))),
        );
        let (q0, q1) = (
            _mm256_loadu_ps(q.as_ptr().add(i * 16)),
            _mm256_loadu_ps(q.as_ptr().add(i * 16 + 8)),
        );
        a = _mm256_fmadd_ps(x0, q0, a);
        b = _mm256_fmadd_ps(x1, q1, b);
    }
    let mut s = hsum256(_mm256_add_ps(a, b));
    for i in n * 16..dim {
        s += f16u(d, i) * q[i];
    }
    s
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn cosine_f16_avx(d: &[u8], q: &[f32], qnr: f32) -> f32 {
    use std::arch::x86_64::*;
    let (dim, n) = (q.len(), q.len() / 16);
    let (mut da, mut db, mut na, mut nb) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );
    for i in 0..n {
        let p = d.as_ptr().add(i * 32) as *const __m128i;
        let (x0, x1) = (
            _mm256_cvtph_ps(_mm_loadu_si128(p)),
            _mm256_cvtph_ps(_mm_loadu_si128(p.add(1))),
        );
        let (q0, q1) = (
            _mm256_loadu_ps(q.as_ptr().add(i * 16)),
            _mm256_loadu_ps(q.as_ptr().add(i * 16 + 8)),
        );
        da = _mm256_fmadd_ps(x0, q0, da);
        db = _mm256_fmadd_ps(x1, q1, db);
        na = _mm256_fmadd_ps(x0, x0, na);
        nb = _mm256_fmadd_ps(x1, x1, nb);
    }
    let (mut dot, mut ns) = (
        hsum256(_mm256_add_ps(da, db)),
        hsum256(_mm256_add_ps(na, nb)),
    );
    for i in n * 16..dim {
        let x = f16u(d, i);
        dot += x * q[i];
        ns += x * x;
    }
    if ns == 0.0 || qnr == 0.0 {
        return 0.0;
    }
    dot * rsqrt_positive_f32(ns) * qnr
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn l1_f16_avx(d: &[u8], q: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let (dim, n) = (q.len(), q.len() / 16);
    let sm = _mm256_set1_ps(-0.0);
    let (mut a, mut b) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    for i in 0..n {
        let p = d.as_ptr().add(i * 32) as *const __m128i;
        let (x0, x1) = (
            _mm256_cvtph_ps(_mm_loadu_si128(p)),
            _mm256_cvtph_ps(_mm_loadu_si128(p.add(1))),
        );
        let (q0, q1) = (
            _mm256_loadu_ps(q.as_ptr().add(i * 16)),
            _mm256_loadu_ps(q.as_ptr().add(i * 16 + 8)),
        );
        a = _mm256_add_ps(a, _mm256_andnot_ps(sm, _mm256_sub_ps(q0, x0)));
        b = _mm256_add_ps(b, _mm256_andnot_ps(sm, _mm256_sub_ps(q1, x1)));
    }
    let mut s = hsum256(_mm256_add_ps(a, b));
    for i in n * 16..dim {
        s += (q[i] - f16u(d, i)).abs();
    }
    s
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn linf_f16_avx(d: &[u8], q: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let (dim, n) = (q.len(), q.len() / 16);
    let sm = _mm256_set1_ps(-0.0);
    let (mut a, mut b) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    for i in 0..n {
        let p = d.as_ptr().add(i * 32) as *const __m128i;
        let (x0, x1) = (
            _mm256_cvtph_ps(_mm_loadu_si128(p)),
            _mm256_cvtph_ps(_mm_loadu_si128(p.add(1))),
        );
        let (q0, q1) = (
            _mm256_loadu_ps(q.as_ptr().add(i * 16)),
            _mm256_loadu_ps(q.as_ptr().add(i * 16 + 8)),
        );
        a = _mm256_max_ps(a, _mm256_andnot_ps(sm, _mm256_sub_ps(q0, x0)));
        b = _mm256_max_ps(b, _mm256_andnot_ps(sm, _mm256_sub_ps(q1, x1)));
    }
    let m = _mm256_max_ps(a, b);
    let hi = _mm256_extractf128_ps(m, 1);
    let lo = _mm256_castps256_ps128(m);
    let s = _mm_max_ps(lo, hi);
    let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    let mut mv = _mm_cvtss_f32(_mm_max_ss(s, _mm_shuffle_ps(s, s, 1)));
    for i in n * 16..dim {
        mv = mv.max((q[i] - f16u(d, i)).abs());
    }
    mv
}

// ── Scalar fallbacks (all targets; also used as aarch64/non-fp16 fallback) ────
fn l2sq_f16_scalar(d: &[u8], q: &[f32]) -> f32 {
    let (mut s0, mut s1, mut s2, mut s3) = (0f32, 0f32, 0f32, 0f32);
    let c = q.len() / 4;
    for i in 0..c {
        let o = i * 4;
        let (d0, d1, d2, d3) = (
            q[o] - f16u(d, o),
            q[o + 1] - f16u(d, o + 1),
            q[o + 2] - f16u(d, o + 2),
            q[o + 3] - f16u(d, o + 3),
        );
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    let mut s = s0 + s1 + s2 + s3;
    for i in c * 4..q.len() {
        let e = q[i] - f16u(d, i);
        s += e * e;
    }
    s
}
fn dot_f16_scalar(d: &[u8], q: &[f32]) -> f32 {
    let (mut s0, mut s1, mut s2, mut s3) = (0f32, 0f32, 0f32, 0f32);
    let c = q.len() / 4;
    for i in 0..c {
        let o = i * 4;
        s0 += f16u(d, o) * q[o];
        s1 += f16u(d, o + 1) * q[o + 1];
        s2 += f16u(d, o + 2) * q[o + 2];
        s3 += f16u(d, o + 3) * q[o + 3];
    }
    let mut s = s0 + s1 + s2 + s3;
    for i in c * 4..q.len() {
        s += f16u(d, i) * q[i];
    }
    s
}
fn cosine_f16_scalar(d: &[u8], q: &[f32], qnr: f32) -> f32 {
    let (mut dot, mut ns) = (0f32, 0f32);
    for i in 0..q.len() {
        let x = f16u(d, i);
        dot += x * q[i];
        ns += x * x;
    }
    if ns == 0.0 || qnr == 0.0 {
        return 0.0;
    }
    dot * rsqrt_positive_f32(ns) * qnr
}
fn l1_f16_scalar(d: &[u8], q: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..q.len() {
        s += (q[i] - f16u(d, i)).abs();
    }
    s
}
fn linf_f16_scalar(d: &[u8], q: &[f32]) -> f32 {
    let mut m = 0f32;
    for i in 0..q.len() {
        m = m.max((q[i] - f16u(d, i)).abs());
    }
    m
}

// ── Runtime dispatcher ────────────────────────────────────────────────────────
#[inline(always)]
fn compute_f16_row_topk_ordering(row: &[u8], c: &DistanceComputer) -> f32 {
    let (q, qnr) = (&c.query, c.query_norm_recip);
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("fp16") {
        return match c.metric {
            DistanceMetric::L2Squared => unsafe { l2sq_f16_neon(row, q) },
            DistanceMetric::L2 => unsafe { l2sq_f16_neon(row, q) },
            DistanceMetric::InnerProduct => -unsafe { dot_f16_neon(row, q) },
            DistanceMetric::NegInnerProduct => -unsafe { dot_f16_neon(row, q) },
            DistanceMetric::CosineSimilarity => unsafe { cosine_f16_neon(row, q, qnr) },
            DistanceMetric::CosineDistance => 1.0 - unsafe { cosine_f16_neon(row, q, qnr) },
            DistanceMetric::L1 => unsafe { l1_f16_neon(row, q) },
            DistanceMetric::LInf => unsafe { linf_f16_neon(row, q) },
        };
    }
    #[cfg(target_arch = "aarch64")]
    return match c.metric {
        DistanceMetric::L2Squared => l2sq_f16_scalar(row, q),
        DistanceMetric::L2 => l2sq_f16_scalar(row, q),
        DistanceMetric::InnerProduct => -dot_f16_scalar(row, q),
        DistanceMetric::NegInnerProduct => -dot_f16_scalar(row, q),
        DistanceMetric::CosineSimilarity => cosine_f16_scalar(row, q, qnr),
        DistanceMetric::CosineDistance => 1.0 - cosine_f16_scalar(row, q, qnr),
        DistanceMetric::L1 => l1_f16_scalar(row, q),
        DistanceMetric::LInf => linf_f16_scalar(row, q),
    };
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx2") {
            return match c.metric {
                DistanceMetric::L2Squared => unsafe { l2sq_f16_avx(row, q) },
                DistanceMetric::L2 => unsafe { l2sq_f16_avx(row, q) },
                DistanceMetric::InnerProduct => -unsafe { dot_f16_avx(row, q) },
                DistanceMetric::NegInnerProduct => -unsafe { dot_f16_avx(row, q) },
                DistanceMetric::CosineSimilarity => unsafe { cosine_f16_avx(row, q, qnr) },
                DistanceMetric::CosineDistance => 1.0 - unsafe { cosine_f16_avx(row, q, qnr) },
                DistanceMetric::L1 => unsafe { l1_f16_avx(row, q) },
                DistanceMetric::LInf => unsafe { linf_f16_avx(row, q) },
            };
        }
        match c.metric {
            DistanceMetric::L2Squared => l2sq_f16_scalar(row, q),
            DistanceMetric::L2 => l2sq_f16_scalar(row, q),
            DistanceMetric::InnerProduct => -dot_f16_scalar(row, q),
            DistanceMetric::NegInnerProduct => -dot_f16_scalar(row, q),
            DistanceMetric::CosineSimilarity => cosine_f16_scalar(row, q, qnr),
            DistanceMetric::CosineDistance => 1.0 - cosine_f16_scalar(row, q, qnr),
            DistanceMetric::L1 => l1_f16_scalar(row, q),
            DistanceMetric::LInf => linf_f16_scalar(row, q),
        }
    }
}

/// Parallel TopK on raw f16 LE bytes (n_rows × dim × 2 bytes), no pre-decode.
/// Uses SIMD f16 distance kernels; no intermediate row_buf write.
/// Returns `Vec<(row_index, f32_distance)>` sorted ascending (nearest first).
pub fn topk_heap_on_f16_bytes(
    f16_bytes: &[u8],
    n_rows: usize,
    dim: usize,
    computer: &DistanceComputer,
    k: usize,
) -> Vec<(usize, f32)> {
    use rayon::prelude::*;

    if k == 0 || n_rows == 0 || dim == 0 || computer.query.is_empty() {
        return vec![];
    }
    let k_capped = k.min(n_rows);
    let row_bytes = dim * 2;
    let signed_topk_ordering = matches!(
        computer.metric,
        DistanceMetric::InnerProduct | DistanceMetric::NegInnerProduct
    );

    let bytes_ptr = f16_bytes.as_ptr() as usize;
    let bytes_len = f16_bytes.len();
    let t = rayon::current_num_threads().max(1);
    let chunk = (n_rows + t - 1) / t;

    let per_chunk: Vec<Vec<TopKEntry>> = (0..t)
        .into_par_iter()
        .map(|tid| {
            let start = tid * chunk;
            if start >= n_rows {
                return vec![];
            }
            let end = (start + chunk).min(n_rows);
            let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr as *const u8, bytes_len) };
            let mut entries: Vec<TopKEntry> = Vec::with_capacity(k_capped);
            for i in start..end {
                let off = i * row_bytes;
                if off + row_bytes > bytes_len {
                    break;
                }
                let score = compute_f16_row_topk_ordering(&bytes[off..off + row_bytes], computer);
                if signed_topk_ordering {
                    insert_topk_score(&mut entries, k_capped, score, i);
                } else {
                    insert_topk_bits(&mut entries, k_capped, score.to_bits(), i);
                }
            }
            entries
        })
        .collect();

    let mut final_entries: Vec<TopKEntry> = Vec::with_capacity(k_capped);
    for chunk in per_chunk {
        for TopKEntry(key, bits, idx) in chunk {
            insert_topk_entry(&mut final_entries, k_capped, key, bits, idx);
        }
    }
    topk_entries_to_result(final_entries, computer)
}

/// Single-query sequential TopK on raw f16 LE bytes.
/// Used by `batch_topk_on_f16_bytes` for outer-parallel-over-queries strategy.
fn topk_sequential_on_f16_bytes(
    f16_bytes: &[u8],
    n_rows: usize,
    dim: usize,
    computer: &DistanceComputer,
    k: usize,
) -> Vec<(usize, f32)> {
    let row_bytes = dim * 2;
    let bytes_len = f16_bytes.len();
    let mut entries: Vec<TopKEntry> = Vec::with_capacity(k);
    let signed_topk_ordering = matches!(
        computer.metric,
        DistanceMetric::InnerProduct | DistanceMetric::NegInnerProduct
    );
    for i in 0..n_rows {
        let off = i * row_bytes;
        if off + row_bytes > bytes_len {
            break;
        }
        let score = compute_f16_row_topk_ordering(&f16_bytes[off..off + row_bytes], computer);
        if signed_topk_ordering {
            insert_topk_score(&mut entries, k, score, i);
        } else {
            insert_topk_bits(&mut entries, k, score.to_bits(), i);
        }
    }
    topk_entries_to_result(entries, computer)
}

/// Adaptive batch TopK on raw f16 LE bytes — N queries in one call.
/// Mirrors `batch_topk_on_floats` but decodes f16→f32 per row during computation.
/// `f16_bytes`: n_rows × dim × 2, row-major.
/// `queries`:   n_queries × dim, f32 row-major.
/// Returns `Vec<Vec<(row_index, f32_distance)>>` length n_queries, each sorted asc.
pub fn batch_topk_on_f16_bytes(
    f16_bytes: &[u8],
    n_rows: usize,
    dim: usize,
    queries: &[f32],
    n_queries: usize,
    k: usize,
    metric: DistanceMetric,
) -> Vec<Vec<(usize, f32)>> {
    use rayon::prelude::*;

    if k == 0 || n_rows == 0 || dim == 0 || n_queries == 0 {
        return vec![vec![]; n_queries];
    }
    let k_capped = k.min(n_rows);
    let t = rayon::current_num_threads().max(1);
    let computers: Vec<DistanceComputer> = (0..n_queries)
        .map(|qi| {
            let start = qi * dim;
            let end = start + dim;
            DistanceComputer::new(metric, queries[start..end].to_vec())
        })
        .collect();

    if n_queries < t {
        computers
            .iter()
            .map(|computer| topk_heap_on_f16_bytes(f16_bytes, n_rows, dim, computer, k_capped))
            .collect()
    } else {
        computers
            .par_iter()
            .map(|computer| {
                topk_sequential_on_f16_bytes(f16_bytes, n_rows, dim, computer, k_capped)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Python-side vector encoding helpers (used by bindings.rs)
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a slice of f32 values as raw LE bytes.
pub fn encode_f32_vec(floats: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(floats.len() * 4);
    for &f in floats {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Decode raw LE bytes to a Vec<f32>.
pub fn decode_f32_vec(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    let len = bytes.len() / 4;
    let mut out = Vec::with_capacity(len);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 0.0, 0.0];
        assert!((l2_distance(&a, &b) - 1.0).abs() < 1e-6);

        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let expected = ((3.0f32 * 3.0 + 3.0 * 3.0 + 3.0 * 3.0) as f32).sqrt();
        assert!((l2_distance(&a, &b) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        assert!((inner_product(&a, &b) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_l1_distance() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        assert!((l1_distance(&a, &b) - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_encode_decode() {
        let v = vec![1.0f32, 2.0, 3.0];
        let encoded = encode_f32_vec(&v);
        let decoded = decode_f32_vec(&encoded).unwrap();
        assert_eq!(decoded, v);
    }

    #[test]
    fn test_topk_indices() {
        let dists = Float64Array::from(vec![Some(5.0), Some(1.0), Some(3.0), Some(2.0), None]);
        let idx = topk_indices_asc(&dists, 3);
        assert_eq!(idx, vec![1, 3, 2]); // distances 1.0, 2.0, 3.0
    }

    #[test]
    fn test_batch_distance() {
        // Build a binary column with two 3-dim f32 vectors
        let v1 = encode_f32_vec(&[1.0, 0.0, 0.0]);
        let v2 = encode_f32_vec(&[0.0, 1.0, 0.0]);
        let col: BinaryArray = vec![Some(v1.as_slice()), Some(v2.as_slice())].into();
        let query = vec![1.0f32, 0.0, 0.0];

        let out = batch_distance(&col, &query, DistanceMetric::L2).unwrap();
        let fa = out.as_any().downcast_ref::<Float64Array>().unwrap();
        assert!((fa.value(0)).abs() < 1e-5); // distance to itself
        assert!((fa.value(1) - 1.4142135).abs() < 1e-4); // √2
    }
}
