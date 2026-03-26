//! SIMD-accelerated distance computations.
//!
//! Optimized for high-dimensional vectors (700-3500 dims).
//! Uses platform-specific SIMD intrinsics with fallback to scalar code.

/// Inner product (dot product) of two f32 vectors.
/// Higher value = more similar.
#[inline(always)]
pub fn inner_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { inner_product_avx2_fma(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return inner_product_neon(a, b);
    }

    #[allow(unreachable_code)]
    inner_product_scalar(a, b)
}

/// Squared Euclidean distance (L2²).
/// Lower value = more similar.
#[inline(always)]
pub fn l2_squared_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { l2_squared_avx2_fma(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return l2_squared_neon(a, b);
    }

    #[allow(unreachable_code)]
    l2_squared_scalar(a, b)
}

/// Cosine distance = 1 - cosine_similarity.
/// Lower value = more similar.
#[inline(always)]
pub fn cosine_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { cosine_distance_avx2_fma(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return cosine_distance_neon(a, b);
    }

    #[allow(unreachable_code)]
    cosine_distance_scalar(a, b)
}

/// Hamming distance for float vectors (thresholded to binary).
#[inline]
pub fn hamming_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut count = 0u32;
    for i in 0..a.len() {
        let ab = a[i] > 0.5;
        let bb = b[i] > 0.5;
        if ab != bb {
            count += 1;
        }
    }
    count as f32
}

/// Jaccard distance for float vectors (thresholded to binary).
#[inline]
pub fn jaccard_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut intersection = 0u32;
    let mut union = 0u32;
    for i in 0..a.len() {
        let ab = a[i] > 0.5;
        let bb = b[i] > 0.5;
        if ab || bb {
            union += 1;
            if ab && bb {
                intersection += 1;
            }
        }
    }
    if union == 0 {
        0.0
    } else {
        1.0 - (intersection as f32 / union as f32)
    }
}

/// Hamming distance for packed u8 binary vectors.
#[inline]
pub fn hamming_u8(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

// ─── Scalar fallbacks ────────────────────────────────────────────────────────

#[inline]
fn inner_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    // Process in chunks of 8 for better pipelining
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;

    for i in 0..chunks {
        let base = i * 8;
        let mut local_sum = 0.0f64;
        local_sum += (a[base] as f64) * (b[base] as f64);
        local_sum += (a[base + 1] as f64) * (b[base + 1] as f64);
        local_sum += (a[base + 2] as f64) * (b[base + 2] as f64);
        local_sum += (a[base + 3] as f64) * (b[base + 3] as f64);
        local_sum += (a[base + 4] as f64) * (b[base + 4] as f64);
        local_sum += (a[base + 5] as f64) * (b[base + 5] as f64);
        local_sum += (a[base + 6] as f64) * (b[base + 6] as f64);
        local_sum += (a[base + 7] as f64) * (b[base + 7] as f64);
        sum += local_sum;
    }

    let base = chunks * 8;
    for i in 0..remainder {
        sum += (a[base + i] as f64) * (b[base + i] as f64);
    }

    sum as f32
}

#[inline]
fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;

    for i in 0..chunks {
        let base = i * 8;
        let mut local_sum = 0.0f64;
        for j in 0..8 {
            let diff = (a[base + j] - b[base + j]) as f64;
            local_sum += diff * diff;
        }
        sum += local_sum;
    }

    let base = chunks * 8;
    for i in 0..remainder {
        let diff = (a[base + i] - b[base + i]) as f64;
        sum += diff * diff;
    }

    sum as f32
}

#[inline]
fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-30 {
        1.0
    } else {
        1.0 - (dot / denom) as f32
    }
}

// ─── x86_64 AVX2+FMA implementations ────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn inner_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let double_chunks = chunks / 2;
    let single_remaining = chunks % 2;

    // Process 16 elements at a time (2x unrolled)
    for i in 0..double_chunks {
        let base = i * 16;
        let va0 = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(base));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        let va1 = _mm256_loadu_ps(a.as_ptr().add(base + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(base + 8));
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
    }

    // Process remaining 8-element chunks
    if single_remaining > 0 {
        let base = double_chunks * 16;
        let va = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb = _mm256_loadu_ps(b.as_ptr().add(base));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
    }

    // Combine accumulators
    acc0 = _mm256_add_ps(acc0, acc1);

    // Horizontal sum
    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut sum = _mm_cvtss_f32(result);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        sum += a[base + i] * b[base + i];
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_squared_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let double_chunks = chunks / 2;
    let single_remaining = chunks % 2;

    for i in 0..double_chunks {
        let base = i * 16;
        let va0 = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(base));
        let diff0 = _mm256_sub_ps(va0, vb0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        let va1 = _mm256_loadu_ps(a.as_ptr().add(base + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(base + 8));
        let diff1 = _mm256_sub_ps(va1, vb1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
    }

    if single_remaining > 0 {
        let base = double_chunks * 16;
        let va = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb = _mm256_loadu_ps(b.as_ptr().add(base));
        let diff = _mm256_sub_ps(va, vb);
        acc0 = _mm256_fmadd_ps(diff, diff, acc0);
    }

    acc0 = _mm256_add_ps(acc0, acc1);

    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut sum = _mm_cvtss_f32(result);

    let base = chunks * 8;
    for i in 0..remainder {
        let diff = a[base + i] - b[base + i];
        sum += diff * diff;
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cosine_distance_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut dot_acc = _mm256_setzero_ps();
    let mut norm_a_acc = _mm256_setzero_ps();
    let mut norm_b_acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb = _mm256_loadu_ps(b.as_ptr().add(base));

        dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
        norm_a_acc = _mm256_fmadd_ps(va, va, norm_a_acc);
        norm_b_acc = _mm256_fmadd_ps(vb, vb, norm_b_acc);
    }

    // Horizontal sums
    let hsum = |acc: __m256| -> f32 {
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    };

    let mut dot = hsum(dot_acc);
    let mut norm_a = hsum(norm_a_acc);
    let mut norm_b = hsum(norm_b_acc);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        dot += a[base + i] * b[base + i];
        norm_a += a[base + i] * a[base + i];
        norm_b += b[base + i] * b[base + i];
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-30 {
        1.0
    } else {
        1.0 - dot / denom
    }
}

// ─── aarch64 NEON implementations ───────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn inner_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks16 = n / 16;
    let remainder = n % 16;

    let mut sum;
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let ap = a.as_ptr();
        let bp = b.as_ptr();

        // 4x unrolled: 16 floats per iteration.
        // LDNP (non-temporal) for candidate b avoids L2/L3 cache pollution — critical
        // for multi-core performance (25% improvement at 10 threads on Apple Silicon).
        // Query a uses regular loads to stay in L1 cache across iterations.
        for i in 0..chunks16 {
            let base = i * 16;
            let va0 = vld1q_f32(ap.add(base));
            let va1 = vld1q_f32(ap.add(base + 4));
            let va2 = vld1q_f32(ap.add(base + 8));
            let va3 = vld1q_f32(ap.add(base + 12));

            let vb0: float32x4_t;
            let vb1: float32x4_t;
            let vb2: float32x4_t;
            let vb3: float32x4_t;
            core::arch::asm!(
                "ldnp {0:q}, {1:q}, [{4}]",
                "ldnp {2:q}, {3:q}, [{4}, #32]",
                out(vreg) vb0,
                out(vreg) vb1,
                out(vreg) vb2,
                out(vreg) vb3,
                in(reg) bp.add(base),
                options(nostack, preserves_flags),
            );

            acc0 = vfmaq_f32(acc0, va0, vb0);
            acc1 = vfmaq_f32(acc1, va1, vb1);
            acc2 = vfmaq_f32(acc2, va2, vb2);
            acc3 = vfmaq_f32(acc3, va3, vb3);
        }

        // Combine 4 accumulators
        acc0 = vaddq_f32(acc0, acc1);
        acc2 = vaddq_f32(acc2, acc3);
        acc0 = vaddq_f32(acc0, acc2);
        sum = vaddvq_f32(acc0);
    }

    // Handle remainder
    let base = chunks16 * 16;
    for i in 0..remainder {
        sum += a[base + i] * b[base + i];
    }

    sum
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks16 = n / 16;
    let remainder = n % 16;

    let mut sum;
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        // 4x unrolled: 16 floats per iteration
        for i in 0..chunks16 {
            let base = i * 16;
            let va0 = vld1q_f32(a.as_ptr().add(base));
            let vb0 = vld1q_f32(b.as_ptr().add(base));
            let diff0 = vsubq_f32(va0, vb0);
            acc0 = vfmaq_f32(acc0, diff0, diff0);

            let va1 = vld1q_f32(a.as_ptr().add(base + 4));
            let vb1 = vld1q_f32(b.as_ptr().add(base + 4));
            let diff1 = vsubq_f32(va1, vb1);
            acc1 = vfmaq_f32(acc1, diff1, diff1);

            let va2 = vld1q_f32(a.as_ptr().add(base + 8));
            let vb2 = vld1q_f32(b.as_ptr().add(base + 8));
            let diff2 = vsubq_f32(va2, vb2);
            acc2 = vfmaq_f32(acc2, diff2, diff2);

            let va3 = vld1q_f32(a.as_ptr().add(base + 12));
            let vb3 = vld1q_f32(b.as_ptr().add(base + 12));
            let diff3 = vsubq_f32(va3, vb3);
            acc3 = vfmaq_f32(acc3, diff3, diff3);
        }

        acc0 = vaddq_f32(acc0, acc1);
        acc2 = vaddq_f32(acc2, acc3);
        acc0 = vaddq_f32(acc0, acc2);
        sum = vaddvq_f32(acc0);
    }

    let base = chunks16 * 16;
    for i in 0..remainder {
        let diff = a[base + i] - b[base + i];
        sum += diff * diff;
    }

    sum
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn cosine_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks16 = n / 16;
    let remainder = n % 16;

    let (mut dot, mut norm_a, mut norm_b);
    unsafe {
        let mut dot0 = vdupq_n_f32(0.0);
        let mut dot1 = vdupq_n_f32(0.0);
        let mut na0 = vdupq_n_f32(0.0);
        let mut na1 = vdupq_n_f32(0.0);
        let mut nb0 = vdupq_n_f32(0.0);
        let mut nb1 = vdupq_n_f32(0.0);

        // 4x unrolled (2 accumulators × 2 loads per iteration = 16 floats)
        for i in 0..chunks16 {
            let base = i * 16;
            let va0 = vld1q_f32(a.as_ptr().add(base));
            let vb0 = vld1q_f32(b.as_ptr().add(base));
            dot0 = vfmaq_f32(dot0, va0, vb0);
            na0 = vfmaq_f32(na0, va0, va0);
            nb0 = vfmaq_f32(nb0, vb0, vb0);

            let va1 = vld1q_f32(a.as_ptr().add(base + 4));
            let vb1 = vld1q_f32(b.as_ptr().add(base + 4));
            dot1 = vfmaq_f32(dot1, va1, vb1);
            na1 = vfmaq_f32(na1, va1, va1);
            nb1 = vfmaq_f32(nb1, vb1, vb1);

            let va2 = vld1q_f32(a.as_ptr().add(base + 8));
            let vb2 = vld1q_f32(b.as_ptr().add(base + 8));
            dot0 = vfmaq_f32(dot0, va2, vb2);
            na0 = vfmaq_f32(na0, va2, va2);
            nb0 = vfmaq_f32(nb0, vb2, vb2);

            let va3 = vld1q_f32(a.as_ptr().add(base + 12));
            let vb3 = vld1q_f32(b.as_ptr().add(base + 12));
            dot1 = vfmaq_f32(dot1, va3, vb3);
            na1 = vfmaq_f32(na1, va3, va3);
            nb1 = vfmaq_f32(nb1, vb3, vb3);
        }

        dot0 = vaddq_f32(dot0, dot1);
        na0 = vaddq_f32(na0, na1);
        nb0 = vaddq_f32(nb0, nb1);
        dot = vaddvq_f32(dot0);
        norm_a = vaddvq_f32(na0);
        norm_b = vaddvq_f32(nb0);
    }

    let base = chunks16 * 16;
    for i in 0..remainder {
        dot += a[base + i] * b[base + i];
        norm_a += a[base + i] * a[base + i];
        norm_b += b[base + i] * b[base + i];
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-30 {
        1.0
    } else {
        1.0 - dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inner_product() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![4.0f32, 3.0, 2.0, 1.0];
        let result = inner_product_f32(&a, &b);
        assert!((result - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_squared() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let result = l2_squared_f32(&a, &b);
        assert!((result - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        let result = cosine_distance_f32(&a, &b);
        assert!(result.abs() < 1e-5); // same vector → distance 0
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let result = cosine_distance_f32(&a, &b);
        assert!((result - 1.0).abs() < 1e-5); // orthogonal → distance 1
    }

    #[test]
    fn test_high_dim_ip() {
        // Test with typical embedding dimension
        let dim = 768;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((dim - i) as f32) * 0.001).collect();
        let result = inner_product_f32(&a, &b);
        let expected = inner_product_scalar(&a, &b);
        assert!((result - expected).abs() < 1e-2);
    }
}
