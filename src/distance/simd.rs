//! SIMD-accelerated distance computations.
//!
//! Optimized for high-dimensional vectors (700-3500 dims).
//! Uses platform-specific SIMD intrinsics with fallback to scalar code.

use half::f16;

const JENSEN_SHANNON_STABLE_DIVERGENCE: f32 = 1e-5;

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

/// Batch inner product: one query against four candidate vectors.
///
/// Shares query loads across four dot products — critical for bandwidth-bound
/// brute-force scans (e.g. 1M × 128 flat search).
#[inline(always)]
pub fn inner_product_batch4_f32(
    query: &[f32],
    v0: &[f32],
    v1: &[f32],
    v2: &[f32],
    v3: &[f32],
) -> [f32; 4] {
    debug_assert_eq!(query.len(), v0.len());
    debug_assert_eq!(query.len(), v1.len());
    debug_assert_eq!(query.len(), v2.len());
    debug_assert_eq!(query.len(), v3.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { inner_product_batch4_avx2_fma(query, v0, v1, v2, v3) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return inner_product_batch4_neon(query, v0, v1, v2, v3);
    }

    #[allow(unreachable_code)]
    inner_product_batch4_scalar(query, v0, v1, v2, v3)
}

/// Batch inner product: one query against eight candidate vectors.
#[inline(always)]
pub fn inner_product_batch8_f32(
    query: &[f32],
    v0: &[f32],
    v1: &[f32],
    v2: &[f32],
    v3: &[f32],
    v4: &[f32],
    v5: &[f32],
    v6: &[f32],
    v7: &[f32],
) -> [f32; 8] {
    debug_assert_eq!(query.len(), v0.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { inner_product_batch8_avx2_fma(query, v0, v1, v2, v3, v4, v5, v6, v7) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return inner_product_batch8_neon(query, v0, v1, v2, v3, v4, v5, v6, v7);
    }

    #[allow(unreachable_code)]
    inner_product_batch8_scalar(query, v0, v1, v2, v3, v4, v5, v6, v7)
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

/// Batch L2 squared: one query against eight candidate vectors.
#[inline(always)]
pub fn l2_squared_batch8_f32(
    query: &[f32],
    v0: &[f32],
    v1: &[f32],
    v2: &[f32],
    v3: &[f32],
    v4: &[f32],
    v5: &[f32],
    v6: &[f32],
    v7: &[f32],
) -> [f32; 8] {
    debug_assert_eq!(query.len(), v0.len());

    #[cfg(target_arch = "aarch64")]
    {
        return l2_squared_batch8_neon(query, v0, v1, v2, v3, v4, v5, v6, v7);
    }

    #[allow(unreachable_code)]
    [
        l2_squared_f32(query, v0),
        l2_squared_f32(query, v1),
        l2_squared_f32(query, v2),
        l2_squared_f32(query, v3),
        l2_squared_f32(query, v4),
        l2_squared_f32(query, v5),
        l2_squared_f32(query, v6),
        l2_squared_f32(query, v7),
    ]
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

/// Manhattan (L1/city-block) distance.
#[inline(always)]
pub fn manhattan_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { manhattan_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return manhattan_neon(a, b);
    }

    #[allow(unreachable_code)]
    manhattan_scalar(a, b)
}

/// Jensen-Shannon distance between non-negative vectors. Inputs are normalized
/// to probability mass internally and natural logarithms are used.
#[inline]
pub fn jensen_shannon_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    for (&av, &bv) in a.iter().zip(b) {
        if !av.is_finite() || !bv.is_finite() || av < 0.0 || bv < 0.0 {
            return f32::INFINITY;
        }
        sum_a += av as f64;
        sum_b += bv as f64;
    }
    if sum_a == 0.0 || sum_b == 0.0 {
        return if sum_a == sum_b {
            0.0
        } else {
            std::f32::consts::LN_2.sqrt()
        };
    }

    let inv_a = (1.0 / sum_a) as f32;
    let inv_b = (1.0 / sum_b) as f32;
    // Preserve the full input contract for pathological subnormal or enormous
    // total mass. Normal probability/count vectors stay on the SIMD path.
    if !inv_a.is_finite() || !inv_b.is_finite() || inv_a == 0.0 || inv_b == 0.0 {
        return jensen_shannon_scalar_f64(a, b, sum_a, sum_b);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let distance = unsafe { jensen_shannon_avx2_fma(a, b, inv_a, inv_b) };
            return refine_small_jensen_shannon(a, b, sum_a, sum_b, distance);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let distance = jensen_shannon_neon(a, b, inv_a, inv_b);
        return refine_small_jensen_shannon(a, b, sum_a, sum_b, distance);
    }

    #[allow(unreachable_code)]
    refine_small_jensen_shannon(
        a,
        b,
        sum_a,
        sum_b,
        jensen_shannon_scalar_f32(a, b, inv_a, inv_b),
    )
}

/// Return `(inverse_mass, sum(p * ln(p)))` for a non-negative row. These two
/// f32 values are sufficient to reuse the row in the entropy-form
/// Jensen-Shannon kernel. Invalid rows return `(NaN, +inf)` and zero-mass rows
/// return `(0, 0)`.
#[inline]
pub fn probability_row_stats_f32(row: &[f32]) -> (f32, f32) {
    let mut sum = 0.0f64;
    for &value in row {
        if !value.is_finite() || value < 0.0 {
            return (f32::NAN, f32::INFINITY);
        }
        sum += value as f64;
    }
    if sum == 0.0 {
        return (0.0, 0.0);
    }
    let inv_mass = (1.0 / sum) as f32;
    if !inv_mass.is_finite() || inv_mass == 0.0 {
        let entropy = row
            .iter()
            .filter(|&&value| value > 0.0)
            .map(|&value| {
                let p = value as f64 / sum;
                p * p.ln()
            })
            .sum::<f64>() as f32;
        return (inv_mass, entropy);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return (inv_mass, unsafe {
                probability_entropy_avx2_fma(row, inv_mass)
            });
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return (inv_mass, probability_entropy_neon(row, inv_mass));
    }

    #[allow(unreachable_code)]
    (inv_mass, probability_entropy_scalar(row, inv_mass))
}

/// Jensen-Shannon distance using a normalized positive-mass query and cached
/// candidate row statistics. This entropy form evaluates one logarithm per
/// dimension instead of two and avoids recomputing either row's mass.
#[inline(always)]
pub fn jensen_shannon_precomputed_f32(
    normalized_query: &[f32],
    candidate: &[f32],
    query_entropy: f32,
    candidate_inv_mass: f32,
    candidate_entropy: f32,
) -> f32 {
    debug_assert_eq!(normalized_query.len(), candidate.len());
    if candidate_inv_mass == 0.0 {
        return std::f32::consts::LN_2.sqrt();
    }
    if !candidate_entropy.is_finite() {
        return f32::INFINITY;
    }
    if !candidate_inv_mass.is_finite() {
        return jensen_shannon_distance_f32(normalized_query, candidate);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe {
                jensen_shannon_precomputed_avx2_fma(
                    normalized_query,
                    candidate,
                    query_entropy,
                    candidate_inv_mass,
                    candidate_entropy,
                )
            };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return jensen_shannon_precomputed_neon(
            normalized_query,
            candidate,
            query_entropy,
            candidate_inv_mass,
            candidate_entropy,
        );
    }

    #[allow(unreachable_code)]
    jensen_shannon_precomputed_scalar(
        normalized_query,
        candidate,
        query_entropy,
        candidate_inv_mass,
        candidate_entropy,
    )
}

/// Two-candidate entropy-form kernel. Loading the normalized query once and
/// interleaving two independent logarithm chains improves instruction-level
/// parallelism in exhaustive scans.
#[inline(always)]
pub fn jensen_shannon_precomputed_batch2_f32(
    normalized_query: &[f32],
    candidate0: &[f32],
    candidate1: &[f32],
    query_entropy: f32,
    stats0: (f32, f32),
    stats1: (f32, f32),
) -> [f32; 2] {
    let divergence = jensen_shannon_precomputed_batch2_divergence_f32(
        normalized_query,
        candidate0,
        candidate1,
        query_entropy,
        stats0,
        stats1,
    );
    [divergence[0].sqrt(), divergence[1].sqrt()]
}

/// Squared Jensen-Shannon distances for top-k ranking. Avoiding a square root
/// for every scanned row is safe because square root is monotonic; callers
/// convert only the final results back to public distances.
#[inline(always)]
pub fn jensen_shannon_precomputed_batch2_divergence_f32(
    normalized_query: &[f32],
    candidate0: &[f32],
    candidate1: &[f32],
    query_entropy: f32,
    stats0: (f32, f32),
    stats1: (f32, f32),
) -> [f32; 2] {
    if stats0.0 <= 0.0
        || stats1.0 <= 0.0
        || !stats0.0.is_finite()
        || !stats1.0.is_finite()
        || !stats0.1.is_finite()
        || !stats1.1.is_finite()
    {
        let distance0 = jensen_shannon_precomputed_f32(
            normalized_query,
            candidate0,
            query_entropy,
            stats0.0,
            stats0.1,
        );
        let distance1 = jensen_shannon_precomputed_f32(
            normalized_query,
            candidate1,
            query_entropy,
            stats1.0,
            stats1.1,
        );
        return [distance0 * distance0, distance1 * distance1];
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe {
                jensen_shannon_precomputed_batch2_divergence_avx2_fma(
                    normalized_query,
                    candidate0,
                    candidate1,
                    query_entropy,
                    stats0,
                    stats1,
                )
            };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return jensen_shannon_precomputed_batch2_divergence_neon(
            normalized_query,
            candidate0,
            candidate1,
            query_entropy,
            stats0,
            stats1,
        );
    }

    #[allow(unreachable_code)]
    {
        let distance0 = jensen_shannon_precomputed_scalar(
            normalized_query,
            candidate0,
            query_entropy,
            stats0.0,
            stats0.1,
        );
        let distance1 = jensen_shannon_precomputed_scalar(
            normalized_query,
            candidate1,
            query_entropy,
            stats1.0,
            stats1.1,
        );
        [distance0 * distance0, distance1 * distance1]
    }
}

#[inline(always)]
fn jensen_shannon_normalized_query_f32(
    normalized_query: &[f32],
    candidate: &[f32],
    candidate_inv_mass: f32,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let distance = unsafe {
                jensen_shannon_avx2_fma(normalized_query, candidate, 1.0, candidate_inv_mass)
            };
            return if distance * distance <= JENSEN_SHANNON_STABLE_DIVERGENCE {
                jensen_shannon_distance_f32(normalized_query, candidate)
            } else {
                distance
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let distance = jensen_shannon_neon(normalized_query, candidate, 1.0, candidate_inv_mass);
        return if distance * distance <= JENSEN_SHANNON_STABLE_DIVERGENCE {
            jensen_shannon_distance_f32(normalized_query, candidate)
        } else {
            distance
        };
    }
    #[allow(unreachable_code)]
    {
        let distance =
            jensen_shannon_scalar_f32(normalized_query, candidate, 1.0, candidate_inv_mass);
        if distance * distance <= JENSEN_SHANNON_STABLE_DIVERGENCE {
            jensen_shannon_distance_f32(normalized_query, candidate)
        } else {
            distance
        }
    }
}

/// Chebyshev (L-infinity) distance.
#[inline(always)]
pub fn chebyshev_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { chebyshev_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return chebyshev_neon(a, b);
    }

    #[allow(unreachable_code)]
    chebyshev_scalar(a, b)
}

/// Canberra distance. Dimensions where both values are zero contribute zero.
#[inline(always)]
pub fn canberra_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { canberra_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return canberra_neon(a, b);
    }

    #[allow(unreachable_code)]
    canberra_scalar(a, b)
}

/// Bray-Curtis distance: `sum(|a-b|) / sum(|a+b|)`.
#[inline(always)]
pub fn bray_curtis_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { bray_curtis_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return bray_curtis_neon(a, b);
    }

    #[allow(unreachable_code)]
    bray_curtis_scalar(a, b)
}

/// Great-circle distance in meters for `[longitude_degrees, latitude_degrees]`.
/// Uses the IUGG mean Earth radius (6,371,008.8 meters).
#[inline]
pub fn haversine_meters_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != 2 || b.len() != 2 {
        return f32::INFINITY;
    }
    const EARTH_MEAN_RADIUS_M: f64 = 6_371_008.8;
    let lon1 = (a[0] as f64).to_radians();
    let lat1 = (a[1] as f64).to_radians();
    let lon2 = (b[0] as f64).to_radians();
    let lat2 = (b[1] as f64).to_radians();
    if !lon1.is_finite()
        || !lat1.is_finite()
        || !lon2.is_finite()
        || !lat2.is_finite()
        || a[1].abs() > 90.0
        || b[1].abs() > 90.0
    {
        return f32::INFINITY;
    }
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let sin_lat = (dlat * 0.5).sin();
    let sin_lon = (dlon * 0.5).sin();
    let h = (sin_lat * sin_lat + lat1.cos() * lat2.cos() * sin_lon * sin_lon).clamp(0.0, 1.0);
    (2.0 * EARTH_MEAN_RADIUS_M * h.sqrt().asin()) as f32
}

/// Pearson correlation distance (`1 - r`). Constant rows use a finite policy:
/// two identical rows have distance 0; otherwise their distance is 1.
#[inline]
pub fn correlation_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f64;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_aa = 0.0f64;
    let mut sum_bb = 0.0f64;
    let mut sum_ab = 0.0f64;
    for (&av, &bv) in a.iter().zip(b) {
        let av = av as f64;
        let bv = bv as f64;
        sum_a += av;
        sum_b += bv;
        sum_aa += av * av;
        sum_bb += bv * bv;
        sum_ab += av * bv;
    }
    let var_a = (sum_aa - sum_a * sum_a / n).max(0.0);
    let var_b = (sum_bb - sum_b * sum_b / n).max(0.0);
    let denom = (var_a * var_b).sqrt();
    if denom <= f64::EPSILON {
        return if a == b { 0.0 } else { 1.0 };
    }
    let covariance = sum_ab - sum_a * sum_b / n;
    (1.0 - (covariance / denom).clamp(-1.0, 1.0)) as f32
}

/// Hellinger distance between non-negative vectors. Inputs are normalized to
/// probability mass internally, so counts and probabilities are both accepted.
#[inline]
pub fn hellinger_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut coefficient_raw = 0.0f64;
    for (&av, &bv) in a.iter().zip(b) {
        if !av.is_finite() || !bv.is_finite() || av < 0.0 || bv < 0.0 {
            return f32::INFINITY;
        }
        sum_a += av as f64;
        sum_b += bv as f64;
        coefficient_raw += ((av as f64) * (bv as f64)).sqrt();
    }
    if sum_a == 0.0 || sum_b == 0.0 {
        return if sum_a == sum_b { 0.0 } else { 1.0 };
    }
    let coefficient = coefficient_raw / (sum_a * sum_b).sqrt();
    (1.0 - coefficient.clamp(0.0, 1.0)).sqrt() as f32
}

/// Wasserstein-1 distance for equal-width ordered bins. Inputs are normalized
/// to probability mass; the result is expressed in bin-width units.
#[inline]
pub fn wasserstein_1d_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    for (&av, &bv) in a.iter().zip(b) {
        if !av.is_finite() || !bv.is_finite() || av < 0.0 || bv < 0.0 {
            return f32::INFINITY;
        }
        sum_a += av as f64;
        sum_b += bv as f64;
    }
    if sum_a == 0.0 || sum_b == 0.0 {
        return if sum_a == sum_b { 0.0 } else { f32::INFINITY };
    }
    let inv_a = 1.0 / sum_a;
    let inv_b = 1.0 / sum_b;
    let mut cdf_delta = 0.0f64;
    let mut distance = 0.0f64;
    // The final CDF delta is always zero, so only the first n-1 intervals
    // contribute for unit-spaced bins.
    for i in 0..a.len().saturating_sub(1) {
        cdf_delta += a[i] as f64 * inv_a - b[i] as f64 * inv_b;
        distance += cdf_delta.abs();
    }
    distance as f32
}

/// Sørensen-Dice distance for float rows thresholded at 0.5.
#[inline]
pub fn dice_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut intersection = 0u32;
    let mut count_a = 0u32;
    let mut count_b = 0u32;
    for (&av, &bv) in a.iter().zip(b) {
        let abit = av > 0.5;
        let bbit = bv > 0.5;
        count_a += abit as u32;
        count_b += bbit as u32;
        intersection += (abit && bbit) as u32;
    }
    let total = count_a + count_b;
    if total == 0 {
        0.0
    } else {
        1.0 - (2 * intersection) as f32 / total as f32
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

// ─── Float16 candidate distance helpers ─────────────────────────────────────

/// Inner product between an f32 query and an f16-encoded candidate row.
#[inline(always)]
pub fn inner_product_f16(query: &[f32], candidate_bits: &[u16]) -> f32 {
    debug_assert_eq!(query.len(), candidate_bits.len());
    let mut sum = 0.0f32;
    for i in 0..query.len() {
        sum += query[i] * f16::from_bits(candidate_bits[i]).to_f32();
    }
    sum
}

/// Squared L2 distance between an f32 query and an f16-encoded candidate row.
#[inline(always)]
pub fn l2_squared_f16(query: &[f32], candidate_bits: &[u16]) -> f32 {
    debug_assert_eq!(query.len(), candidate_bits.len());
    let mut sum = 0.0f32;
    for i in 0..query.len() {
        let cand = f16::from_bits(candidate_bits[i]).to_f32();
        let diff = query[i] - cand;
        sum += diff * diff;
    }
    sum
}

/// Cosine distance between an f32 query and an f16-encoded candidate row.
#[inline(always)]
pub fn cosine_distance_f16(query: &[f32], candidate_bits: &[u16]) -> f32 {
    debug_assert_eq!(query.len(), candidate_bits.len());
    let mut dot = 0.0f32;
    let mut norm_q = 0.0f32;
    let mut norm_c = 0.0f32;
    for i in 0..query.len() {
        let q = query[i];
        let c = f16::from_bits(candidate_bits[i]).to_f32();
        dot += q * c;
        norm_q += q * q;
        norm_c += c * c;
    }
    if norm_q == 0.0 || norm_c == 0.0 {
        1.0
    } else {
        1.0 - dot / (norm_q.sqrt() * norm_c.sqrt())
    }
}

#[inline]
pub fn hamming_f16(query: &[f32], candidate_bits: &[u16]) -> f32 {
    debug_assert_eq!(query.len(), candidate_bits.len());
    let mut count = 0u32;
    for i in 0..query.len() {
        let qb = query[i] > 0.5;
        let cb = f16::from_bits(candidate_bits[i]).to_f32() > 0.5;
        if qb != cb {
            count += 1;
        }
    }
    count as f32
}

#[inline]
pub fn jaccard_f16(query: &[f32], candidate_bits: &[u16]) -> f32 {
    debug_assert_eq!(query.len(), candidate_bits.len());
    let mut intersection = 0u32;
    let mut union = 0u32;
    for i in 0..query.len() {
        let qb = query[i] > 0.5;
        let cb = f16::from_bits(candidate_bits[i]).to_f32() > 0.5;
        if qb || cb {
            union += 1;
            if qb && cb {
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

#[inline]
pub fn manhattan_f16(query: &[f32], candidate_bits: &[u16]) -> f32 {
    debug_assert_eq!(query.len(), candidate_bits.len());
    query
        .iter()
        .zip(candidate_bits)
        .map(|(&q, &c)| (q - f16::from_bits(c).to_f32()).abs())
        .sum()
}

#[inline]
pub fn jensen_shannon_distance_f16(query: &[f32], candidate_bits: &[u16]) -> f32 {
    debug_assert_eq!(query.len(), candidate_bits.len());
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    for (&av, &bits) in query.iter().zip(candidate_bits) {
        let bv = f16::from_bits(bits).to_f32();
        if !av.is_finite() || !bv.is_finite() || av < 0.0 || bv < 0.0 {
            return f32::INFINITY;
        }
        sum_a += av as f64;
        sum_b += bv as f64;
    }
    if sum_a == 0.0 || sum_b == 0.0 {
        return if sum_a == sum_b {
            0.0
        } else {
            std::f32::consts::LN_2.sqrt()
        };
    }
    let mut divergence = 0.0f64;
    for (&av, &bits) in query.iter().zip(candidate_bits) {
        let p = av as f64 / sum_a;
        let q = f16::from_bits(bits).to_f32() as f64 / sum_b;
        let m = 0.5 * (p + q);
        if p > 0.0 {
            divergence += 0.5 * p * (p / m).ln();
        }
        if q > 0.0 {
            divergence += 0.5 * q * (q / m).ln();
        }
    }
    divergence.max(0.0).sqrt() as f32
}

#[inline]
pub fn chebyshev_f16(query: &[f32], candidate_bits: &[u16]) -> f32 {
    debug_assert_eq!(query.len(), candidate_bits.len());
    query
        .iter()
        .zip(candidate_bits)
        .map(|(&q, &c)| (q - f16::from_bits(c).to_f32()).abs())
        .fold(0.0, f32::max)
}

#[inline]
pub fn canberra_f16(query: &[f32], candidate_bits: &[u16]) -> f32 {
    debug_assert_eq!(query.len(), candidate_bits.len());
    query
        .iter()
        .zip(candidate_bits)
        .map(|(&q, &c)| {
            let c = f16::from_bits(c).to_f32();
            let denominator = q.abs() + c.abs();
            if denominator == 0.0 {
                0.0
            } else {
                (q - c).abs() / denominator
            }
        })
        .sum()
}

#[inline]
pub fn bray_curtis_f16(query: &[f32], candidate_bits: &[u16]) -> f32 {
    debug_assert_eq!(query.len(), candidate_bits.len());
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for (&q, &bits) in query.iter().zip(candidate_bits) {
        let c = f16::from_bits(bits).to_f32();
        numerator += (q - c).abs();
        denominator += (q + c).abs();
    }
    if denominator == 0.0 {
        if numerator == 0.0 {
            0.0
        } else {
            f32::INFINITY
        }
    } else {
        numerator / denominator
    }
}

#[inline]
pub fn haversine_meters_f16(query: &[f32], candidate_bits: &[u16]) -> f32 {
    if candidate_bits.len() != 2 {
        return f32::INFINITY;
    }
    let candidate = [
        f16::from_bits(candidate_bits[0]).to_f32(),
        f16::from_bits(candidate_bits[1]).to_f32(),
    ];
    haversine_meters_f32(query, &candidate)
}

#[inline]
pub fn correlation_distance_f16(query: &[f32], candidate_bits: &[u16]) -> f32 {
    debug_assert_eq!(query.len(), candidate_bits.len());
    if query.is_empty() {
        return 0.0;
    }
    let n = query.len() as f64;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_aa = 0.0f64;
    let mut sum_bb = 0.0f64;
    let mut sum_ab = 0.0f64;
    let mut identical = true;
    for (&av, &bits) in query.iter().zip(candidate_bits) {
        let bv32 = f16::from_bits(bits).to_f32();
        identical &= av == bv32;
        let av = av as f64;
        let bv = bv32 as f64;
        sum_a += av;
        sum_b += bv;
        sum_aa += av * av;
        sum_bb += bv * bv;
        sum_ab += av * bv;
    }
    let var_a = (sum_aa - sum_a * sum_a / n).max(0.0);
    let var_b = (sum_bb - sum_b * sum_b / n).max(0.0);
    let denom = (var_a * var_b).sqrt();
    if denom <= f64::EPSILON {
        return if identical { 0.0 } else { 1.0 };
    }
    (1.0 - ((sum_ab - sum_a * sum_b / n) / denom).clamp(-1.0, 1.0)) as f32
}

#[inline]
pub fn hellinger_distance_f16(query: &[f32], candidate_bits: &[u16]) -> f32 {
    debug_assert_eq!(query.len(), candidate_bits.len());
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut coefficient_raw = 0.0f64;
    for (&av, &bits) in query.iter().zip(candidate_bits) {
        let bv = f16::from_bits(bits).to_f32();
        if !av.is_finite() || !bv.is_finite() || av < 0.0 || bv < 0.0 {
            return f32::INFINITY;
        }
        sum_a += av as f64;
        sum_b += bv as f64;
        coefficient_raw += ((av as f64) * (bv as f64)).sqrt();
    }
    if sum_a == 0.0 || sum_b == 0.0 {
        return if sum_a == sum_b { 0.0 } else { 1.0 };
    }
    let coefficient = coefficient_raw / (sum_a * sum_b).sqrt();
    (1.0 - coefficient.clamp(0.0, 1.0)).sqrt() as f32
}

#[inline]
pub fn wasserstein_1d_f16(query: &[f32], candidate_bits: &[u16]) -> f32 {
    debug_assert_eq!(query.len(), candidate_bits.len());
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    for (&av, &bits) in query.iter().zip(candidate_bits) {
        let bv = f16::from_bits(bits).to_f32();
        if !av.is_finite() || !bv.is_finite() || av < 0.0 || bv < 0.0 {
            return f32::INFINITY;
        }
        sum_a += av as f64;
        sum_b += bv as f64;
    }
    if sum_a == 0.0 || sum_b == 0.0 {
        return if sum_a == sum_b { 0.0 } else { f32::INFINITY };
    }
    let mut cdf_delta = 0.0f64;
    let mut distance = 0.0f64;
    for i in 0..query.len().saturating_sub(1) {
        let bv = f16::from_bits(candidate_bits[i]).to_f32();
        cdf_delta += query[i] as f64 / sum_a - bv as f64 / sum_b;
        distance += cdf_delta.abs();
    }
    distance as f32
}

#[inline]
pub fn dice_distance_f16(query: &[f32], candidate_bits: &[u16]) -> f32 {
    debug_assert_eq!(query.len(), candidate_bits.len());
    let mut intersection = 0u32;
    let mut count_a = 0u32;
    let mut count_b = 0u32;
    for (&q, &bits) in query.iter().zip(candidate_bits) {
        let a = q > 0.5;
        let b = f16::from_bits(bits).to_f32() > 0.5;
        count_a += a as u32;
        count_b += b as u32;
        intersection += (a && b) as u32;
    }
    let total = count_a + count_b;
    if total == 0 {
        0.0
    } else {
        1.0 - (2 * intersection) as f32 / total as f32
    }
}

// ─── Scalar fallbacks ────────────────────────────────────────────────────────

#[inline]
fn manhattan_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).sum()
}

#[inline]
fn jensen_shannon_scalar_f32(a: &[f32], b: &[f32], inv_a: f32, inv_b: f32) -> f32 {
    let mut divergence = 0.0f32;
    for (&av, &bv) in a.iter().zip(b) {
        let p = av * inv_a;
        let q = bv * inv_b;
        let m = 0.5 * (p + q);
        if p > 0.0 {
            divergence += 0.5 * p * (p / m).ln();
        }
        if q > 0.0 {
            divergence += 0.5 * q * (q / m).ln();
        }
    }
    divergence.max(0.0).sqrt()
}

#[inline]
fn refine_small_jensen_shannon(a: &[f32], b: &[f32], sum_a: f64, sum_b: f64, distance: f32) -> f32 {
    if distance * distance <= JENSEN_SHANNON_STABLE_DIVERGENCE && a != b {
        jensen_shannon_scalar_f64(a, b, sum_a, sum_b)
    } else {
        distance
    }
}

#[inline]
fn probability_entropy_scalar(row: &[f32], inv_mass: f32) -> f32 {
    row.iter()
        .map(|&value| value * inv_mass)
        .filter(|&p| p > 0.0)
        .map(|p| p * p.ln())
        .sum()
}

#[inline]
fn jensen_shannon_precomputed_scalar(
    normalized_query: &[f32],
    candidate: &[f32],
    query_entropy: f32,
    candidate_inv_mass: f32,
    candidate_entropy: f32,
) -> f32 {
    let mut mixture_entropy_term = 0.0f32;
    for (&p, &value) in normalized_query.iter().zip(candidate) {
        let sum = p + value * candidate_inv_mass;
        if sum > 0.0 {
            mixture_entropy_term += sum * sum.ln();
        }
    }
    let divergence = (std::f32::consts::LN_2
        + 0.5 * (query_entropy + candidate_entropy - mixture_entropy_term))
        .max(0.0);
    if divergence <= JENSEN_SHANNON_STABLE_DIVERGENCE {
        jensen_shannon_normalized_query_f32(normalized_query, candidate, candidate_inv_mass)
    } else {
        divergence.sqrt()
    }
}

#[cold]
fn jensen_shannon_scalar_f64(a: &[f32], b: &[f32], sum_a: f64, sum_b: f64) -> f32 {
    let inv_a = 1.0 / sum_a;
    let inv_b = 1.0 / sum_b;
    let mut divergence = 0.0f64;
    for (&av, &bv) in a.iter().zip(b) {
        let p = av as f64 * inv_a;
        let q = bv as f64 * inv_b;
        let m = 0.5 * (p + q);
        if p > 0.0 {
            divergence += 0.5 * p * (p / m).ln();
        }
        if q > 0.0 {
            divergence += 0.5 * q * (q / m).ln();
        }
    }
    divergence.max(0.0).sqrt() as f32
}

#[inline]
fn chebyshev_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[inline]
fn canberra_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let denominator = x.abs() + y.abs();
            if denominator == 0.0 {
                0.0
            } else {
                (x - y).abs() / denominator
            }
        })
        .sum()
}

#[inline]
fn bray_curtis_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        numerator += (x - y).abs();
        denominator += (x + y).abs();
    }
    if denominator == 0.0 {
        if numerator == 0.0 {
            0.0
        } else {
            f32::INFINITY
        }
    } else {
        numerator / denominator
    }
}

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
fn inner_product_batch4_scalar(
    query: &[f32],
    v0: &[f32],
    v1: &[f32],
    v2: &[f32],
    v3: &[f32],
) -> [f32; 4] {
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut s3 = 0.0f64;
    for i in 0..query.len() {
        let q = query[i] as f64;
        s0 += q * (v0[i] as f64);
        s1 += q * (v1[i] as f64);
        s2 += q * (v2[i] as f64);
        s3 += q * (v3[i] as f64);
    }
    [s0 as f32, s1 as f32, s2 as f32, s3 as f32]
}

#[inline]
fn inner_product_batch8_scalar(
    query: &[f32],
    v0: &[f32],
    v1: &[f32],
    v2: &[f32],
    v3: &[f32],
    v4: &[f32],
    v5: &[f32],
    v6: &[f32],
    v7: &[f32],
) -> [f32; 8] {
    let b4a = inner_product_batch4_scalar(query, v0, v1, v2, v3);
    let b4b = inner_product_batch4_scalar(query, v4, v5, v6, v7);
    [
        b4a[0], b4a[1], b4a[2], b4a[3], b4b[0], b4b[1], b4b[2], b4b[3],
    ]
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
unsafe fn inner_product_batch4_avx2_fma(
    query: &[f32],
    v0: &[f32],
    v1: &[f32],
    v2: &[f32],
    v3: &[f32],
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let n = query.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * 8;
        let qv = _mm256_loadu_ps(query.as_ptr().add(base));
        acc0 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(v0.as_ptr().add(base)), acc0);
        acc1 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(v1.as_ptr().add(base)), acc1);
        acc2 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(v2.as_ptr().add(base)), acc2);
        acc3 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(v3.as_ptr().add(base)), acc3);
    }

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

    let mut out = [hsum(acc0), hsum(acc1), hsum(acc2), hsum(acc3)];
    let base = chunks * 8;
    for i in 0..remainder {
        let q = query[base + i];
        out[0] += q * v0[base + i];
        out[1] += q * v1[base + i];
        out[2] += q * v2[base + i];
        out[3] += q * v3[base + i];
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn inner_product_batch8_avx2_fma(
    query: &[f32],
    v0: &[f32],
    v1: &[f32],
    v2: &[f32],
    v3: &[f32],
    v4: &[f32],
    v5: &[f32],
    v6: &[f32],
    v7: &[f32],
) -> [f32; 8] {
    use std::arch::x86_64::*;

    let n = query.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * 8;
        let qv = _mm256_loadu_ps(query.as_ptr().add(base));
        acc0 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(v0.as_ptr().add(base)), acc0);
        acc1 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(v1.as_ptr().add(base)), acc1);
        acc2 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(v2.as_ptr().add(base)), acc2);
        acc3 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(v3.as_ptr().add(base)), acc3);
        acc4 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(v4.as_ptr().add(base)), acc4);
        acc5 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(v5.as_ptr().add(base)), acc5);
        acc6 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(v6.as_ptr().add(base)), acc6);
        acc7 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(v7.as_ptr().add(base)), acc7);
    }

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

    let mut out = [
        hsum(acc0),
        hsum(acc1),
        hsum(acc2),
        hsum(acc3),
        hsum(acc4),
        hsum(acc5),
        hsum(acc6),
        hsum(acc7),
    ];
    let base = chunks * 8;
    for i in 0..remainder {
        let q = query[base + i];
        out[0] += q * v0[base + i];
        out[1] += q * v1[base + i];
        out[2] += q * v2[base + i];
        out[3] += q * v3[base + i];
        out[4] += q * v4[base + i];
        out[5] += q * v5[base + i];
        out[6] += q * v6[base + i];
        out[7] += q * v7[base + i];
    }
    out
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
fn inner_product_batch4_neon(
    query: &[f32],
    v0: &[f32],
    v1: &[f32],
    v2: &[f32],
    v3: &[f32],
) -> [f32; 4] {
    use std::arch::aarch64::*;

    let n = query.len();
    let chunks16 = n / 16;
    let remainder = n % 16;

    let mut out;
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let qp = query.as_ptr();
        let p0 = v0.as_ptr();
        let p1 = v1.as_ptr();
        let p2 = v2.as_ptr();
        let p3 = v3.as_ptr();

        for i in 0..chunks16 {
            let base = i * 16;
            let qa0 = vld1q_f32(qp.add(base));
            let qa1 = vld1q_f32(qp.add(base + 4));
            let qa2 = vld1q_f32(qp.add(base + 8));
            let qa3 = vld1q_f32(qp.add(base + 12));

            macro_rules! accumulate {
                ($acc:ident, $bp:ident) => {{
                    let (vb0, vb1, vb2, vb3): (
                        float32x4_t,
                        float32x4_t,
                        float32x4_t,
                        float32x4_t,
                    );
                    core::arch::asm!(
                        "ldnp {0:q}, {1:q}, [{4}]",
                        "ldnp {2:q}, {3:q}, [{4}, #32]",
                        out(vreg) vb0,
                        out(vreg) vb1,
                        out(vreg) vb2,
                        out(vreg) vb3,
                        in(reg) $bp.add(base),
                        options(nostack, preserves_flags),
                    );
                    $acc = vfmaq_f32($acc, qa0, vb0);
                    $acc = vfmaq_f32($acc, qa1, vb1);
                    $acc = vfmaq_f32($acc, qa2, vb2);
                    $acc = vfmaq_f32($acc, qa3, vb3);
                }};
            }

            accumulate!(acc0, p0);
            accumulate!(acc1, p1);
            accumulate!(acc2, p2);
            accumulate!(acc3, p3);
        }

        out = [
            vaddvq_f32(acc0),
            vaddvq_f32(acc1),
            vaddvq_f32(acc2),
            vaddvq_f32(acc3),
        ];
    }

    let base = chunks16 * 16;
    for i in 0..remainder {
        let q = query[base + i];
        out[0] += q * v0[base + i];
        out[1] += q * v1[base + i];
        out[2] += q * v2[base + i];
        out[3] += q * v3[base + i];
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn inner_product_batch8_neon(
    query: &[f32],
    v0: &[f32],
    v1: &[f32],
    v2: &[f32],
    v3: &[f32],
    v4: &[f32],
    v5: &[f32],
    v6: &[f32],
    v7: &[f32],
) -> [f32; 8] {
    use std::arch::aarch64::*;

    let n = query.len();
    let chunks16 = n / 16;
    let remainder = n % 16;

    let mut out;
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut acc4 = vdupq_n_f32(0.0);
        let mut acc5 = vdupq_n_f32(0.0);
        let mut acc6 = vdupq_n_f32(0.0);
        let mut acc7 = vdupq_n_f32(0.0);

        let qp = query.as_ptr();
        let p0 = v0.as_ptr();
        let p1 = v1.as_ptr();
        let p2 = v2.as_ptr();
        let p3 = v3.as_ptr();
        let p4 = v4.as_ptr();
        let p5 = v5.as_ptr();
        let p6 = v6.as_ptr();
        let p7 = v7.as_ptr();

        for i in 0..chunks16 {
            let base = i * 16;
            let qa0 = vld1q_f32(qp.add(base));
            let qa1 = vld1q_f32(qp.add(base + 4));
            let qa2 = vld1q_f32(qp.add(base + 8));
            let qa3 = vld1q_f32(qp.add(base + 12));

            macro_rules! accumulate8 {
                ($acc:ident, $bp:ident) => {{
                    let (vb0, vb1, vb2, vb3): (
                        float32x4_t,
                        float32x4_t,
                        float32x4_t,
                        float32x4_t,
                    );
                    core::arch::asm!(
                        "ldnp {0:q}, {1:q}, [{4}]",
                        "ldnp {2:q}, {3:q}, [{4}, #32]",
                        out(vreg) vb0,
                        out(vreg) vb1,
                        out(vreg) vb2,
                        out(vreg) vb3,
                        in(reg) $bp.add(base),
                        options(nostack, preserves_flags),
                    );
                    $acc = vfmaq_f32($acc, qa0, vb0);
                    $acc = vfmaq_f32($acc, qa1, vb1);
                    $acc = vfmaq_f32($acc, qa2, vb2);
                    $acc = vfmaq_f32($acc, qa3, vb3);
                }};
            }

            accumulate8!(acc0, p0);
            accumulate8!(acc1, p1);
            accumulate8!(acc2, p2);
            accumulate8!(acc3, p3);
            accumulate8!(acc4, p4);
            accumulate8!(acc5, p5);
            accumulate8!(acc6, p6);
            accumulate8!(acc7, p7);
        }

        out = [
            vaddvq_f32(acc0),
            vaddvq_f32(acc1),
            vaddvq_f32(acc2),
            vaddvq_f32(acc3),
            vaddvq_f32(acc4),
            vaddvq_f32(acc5),
            vaddvq_f32(acc6),
            vaddvq_f32(acc7),
        ];
    }

    let base = chunks16 * 16;
    for i in 0..remainder {
        let q = query[base + i];
        out[0] += q * v0[base + i];
        out[1] += q * v1[base + i];
        out[2] += q * v2[base + i];
        out[3] += q * v3[base + i];
        out[4] += q * v4[base + i];
        out[5] += q * v5[base + i];
        out[6] += q * v6[base + i];
        out[7] += q * v7[base + i];
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn l2_squared_batch8_neon(
    query: &[f32],
    v0: &[f32],
    v1: &[f32],
    v2: &[f32],
    v3: &[f32],
    v4: &[f32],
    v5: &[f32],
    v6: &[f32],
    v7: &[f32],
) -> [f32; 8] {
    use std::arch::aarch64::*;

    let n = query.len();
    let chunks16 = n / 16;
    let remainder = n % 16;

    let mut out;
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut acc4 = vdupq_n_f32(0.0);
        let mut acc5 = vdupq_n_f32(0.0);
        let mut acc6 = vdupq_n_f32(0.0);
        let mut acc7 = vdupq_n_f32(0.0);

        let qp = query.as_ptr();
        let p0 = v0.as_ptr();
        let p1 = v1.as_ptr();
        let p2 = v2.as_ptr();
        let p3 = v3.as_ptr();
        let p4 = v4.as_ptr();
        let p5 = v5.as_ptr();
        let p6 = v6.as_ptr();
        let p7 = v7.as_ptr();

        for i in 0..chunks16 {
            let base = i * 16;
            let qa0 = vld1q_f32(qp.add(base));
            let qa1 = vld1q_f32(qp.add(base + 4));
            let qa2 = vld1q_f32(qp.add(base + 8));
            let qa3 = vld1q_f32(qp.add(base + 12));

            macro_rules! accumulate_l2 {
                ($acc:ident, $bp:ident) => {{
                    let vb0 = vld1q_f32($bp.add(base));
                    let vb1 = vld1q_f32($bp.add(base + 4));
                    let vb2 = vld1q_f32($bp.add(base + 8));
                    let vb3 = vld1q_f32($bp.add(base + 12));
                    let d0 = vsubq_f32(qa0, vb0);
                    let d1 = vsubq_f32(qa1, vb1);
                    let d2 = vsubq_f32(qa2, vb2);
                    let d3 = vsubq_f32(qa3, vb3);
                    $acc = vfmaq_f32($acc, d0, d0);
                    $acc = vfmaq_f32($acc, d1, d1);
                    $acc = vfmaq_f32($acc, d2, d2);
                    $acc = vfmaq_f32($acc, d3, d3);
                }};
            }

            accumulate_l2!(acc0, p0);
            accumulate_l2!(acc1, p1);
            accumulate_l2!(acc2, p2);
            accumulate_l2!(acc3, p3);
            accumulate_l2!(acc4, p4);
            accumulate_l2!(acc5, p5);
            accumulate_l2!(acc6, p6);
            accumulate_l2!(acc7, p7);
        }

        out = [
            vaddvq_f32(acc0),
            vaddvq_f32(acc1),
            vaddvq_f32(acc2),
            vaddvq_f32(acc3),
            vaddvq_f32(acc4),
            vaddvq_f32(acc5),
            vaddvq_f32(acc6),
            vaddvq_f32(acc7),
        ];
    }

    let base = chunks16 * 16;
    for i in 0..remainder {
        let q = query[base + i];
        let d0 = q - v0[base + i];
        let d1 = q - v1[base + i];
        let d2 = q - v2[base + i];
        let d3 = q - v3[base + i];
        let d4 = q - v4[base + i];
        let d5 = q - v5[base + i];
        let d6 = q - v6[base + i];
        let d7 = q - v7[base + i];
        out[0] += d0 * d0;
        out[1] += d1 * d1;
        out[2] += d2 * d2;
        out[3] += d3 * d3;
        out[4] += d4 * d4;
        out[5] += d5 * d5;
        out[6] += d6 * d6;
        out[7] += d7 * d7;
    }
    out
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn manhattan_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let chunks = a.len() / 8;
    let mut lanes = [0.0f32; 8];
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let sign_mask = _mm256_set1_ps(-0.0);
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            let diff = _mm256_sub_ps(va, vb);
            acc = _mm256_add_ps(acc, _mm256_andnot_ps(sign_mask, diff));
        }
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    }
    let mut sum: f32 = lanes.into_iter().sum();
    for i in chunks * 8..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum
}

/// Eight-lane natural logarithm adapted from the Cephes minimax polynomial.
/// The maximum error over positive normal f32 values is a few ULPs, while the
/// Jensen-Shannon hot loop only needs the compact ratio domain `(0, 2]`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn fast_ln_avx2_fma(mut x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let bits = _mm256_castps_si256(x);
    let exponent_bits = _mm256_srli_epi32(bits, 23);
    let mantissa_bits = _mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi32(0x007f_ffff)),
        _mm256_set1_epi32(0x3f00_0000),
    );
    x = _mm256_castsi256_ps(mantissa_bits);

    let mut exponent = _mm256_cvtepi32_ps(_mm256_sub_epi32(exponent_bits, _mm256_set1_epi32(0x7f)));
    exponent = _mm256_add_ps(exponent, _mm256_set1_ps(1.0));

    let mask = _mm256_cmp_ps(
        x,
        _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2),
        _CMP_LT_OQ,
    );
    let tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, _mm256_set1_ps(1.0));
    exponent = _mm256_sub_ps(exponent, _mm256_and_ps(_mm256_set1_ps(1.0), mask));
    x = _mm256_add_ps(x, tmp);

    let z = _mm256_mul_ps(x, x);
    let mut y = _mm256_set1_ps(7.037_683_6E-2);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-1.151_461E-1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.167_699_84E-1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-1.242_014_1E-1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.424_932_3E-1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-1.666_805_7E-1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(2.000_071_4E-1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-2.499_999_4E-1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(3.333_333E-1));
    y = _mm256_mul_ps(_mm256_mul_ps(y, x), z);
    y = _mm256_fmadd_ps(exponent, _mm256_set1_ps(-2.121_944_4E-4), y);
    y = _mm256_fnmadd_ps(z, _mm256_set1_ps(0.5), y);
    x = _mm256_add_ps(x, y);
    _mm256_fmadd_ps(exponent, _mm256_set1_ps(0.693_359_4), x)
}

/// Six-coefficient variant used by the entropy cache. Its lower instruction
/// count is valuable in exhaustive scans; near-zero divergences are recomputed
/// by the full stable kernel before returning.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn fast_ln_compact_avx2_fma(mut x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let bits = _mm256_castps_si256(x);
    let exponent_bits = _mm256_srli_epi32(bits, 23);
    let mantissa_bits = _mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi32(0x007f_ffff)),
        _mm256_set1_epi32(0x3f00_0000),
    );
    x = _mm256_castsi256_ps(mantissa_bits);
    let mut exponent = _mm256_cvtepi32_ps(_mm256_sub_epi32(exponent_bits, _mm256_set1_epi32(0x7f)));
    exponent = _mm256_add_ps(exponent, _mm256_set1_ps(1.0));
    let mask = _mm256_cmp_ps(
        x,
        _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2),
        _CMP_LT_OQ,
    );
    let tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, _mm256_set1_ps(1.0));
    exponent = _mm256_sub_ps(exponent, _mm256_and_ps(_mm256_set1_ps(1.0), mask));
    x = _mm256_add_ps(x, tmp);

    let z = _mm256_mul_ps(x, x);
    let mut y = _mm256_set1_ps(-1.242_014_1E-1);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.424_932_3E-1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-1.666_805_7E-1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(2.000_071_4E-1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-2.499_999_4E-1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(3.333_333E-1));
    y = _mm256_mul_ps(_mm256_mul_ps(y, x), z);
    y = _mm256_fmadd_ps(exponent, _mm256_set1_ps(-2.121_944_4E-4), y);
    y = _mm256_fnmadd_ps(z, _mm256_set1_ps(0.5), y);
    x = _mm256_add_ps(x, y);
    _mm256_fmadd_ps(exponent, _mm256_set1_ps(0.693_359_4), x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn jensen_shannon_avx2_fma(a: &[f32], b: &[f32], inv_a: f32, inv_b: f32) -> f32 {
    use std::arch::x86_64::*;

    let chunks = a.len() / 8;
    let mut lanes = [0.0f32; 8];
    let mut acc = _mm256_setzero_ps();
    let inv_a = _mm256_set1_ps(inv_a);
    let inv_b = _mm256_set1_ps(inv_b);
    let half = _mm256_set1_ps(0.5);
    let min_normal = _mm256_set1_ps(f32::MIN_POSITIVE);
    for i in 0..chunks {
        let offset = i * 8;
        let p = _mm256_mul_ps(_mm256_loadu_ps(a.as_ptr().add(offset)), inv_a);
        let q = _mm256_mul_ps(_mm256_loadu_ps(b.as_ptr().add(offset)), inv_b);
        let m = _mm256_mul_ps(_mm256_add_ps(p, q), half);
        let safe_m = _mm256_max_ps(m, min_normal);
        let log_p = fast_ln_avx2_fma(_mm256_div_ps(_mm256_max_ps(p, min_normal), safe_m));
        let log_q = fast_ln_avx2_fma(_mm256_div_ps(_mm256_max_ps(q, min_normal), safe_m));
        let terms = _mm256_add_ps(_mm256_mul_ps(p, log_p), _mm256_mul_ps(q, log_q));
        acc = _mm256_fmadd_ps(terms, half, acc);
    }
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut divergence: f32 = lanes.into_iter().sum();
    for i in chunks * 8..a.len() {
        let p = a[i] * _mm_cvtss_f32(_mm256_castps256_ps128(inv_a));
        let q = b[i] * _mm_cvtss_f32(_mm256_castps256_ps128(inv_b));
        let m = 0.5 * (p + q);
        if p > 0.0 {
            divergence += 0.5 * p * (p / m).ln();
        }
        if q > 0.0 {
            divergence += 0.5 * q * (q / m).ln();
        }
    }
    divergence.max(0.0).sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn probability_entropy_avx2_fma(row: &[f32], inv_mass: f32) -> f32 {
    use std::arch::x86_64::*;

    let chunks = row.len() / 8;
    let mut lanes = [0.0f32; 8];
    let mut acc = _mm256_setzero_ps();
    let inv_mass_vec = _mm256_set1_ps(inv_mass);
    let min_normal = _mm256_set1_ps(f32::MIN_POSITIVE);
    for i in 0..chunks {
        let p = _mm256_mul_ps(_mm256_loadu_ps(row.as_ptr().add(i * 8)), inv_mass_vec);
        let log_p = fast_ln_compact_avx2_fma(_mm256_max_ps(p, min_normal));
        acc = _mm256_fmadd_ps(p, log_p, acc);
    }
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut entropy: f32 = lanes.into_iter().sum();
    for &value in &row[chunks * 8..] {
        let p = value * inv_mass;
        if p > 0.0 {
            entropy += p * p.ln();
        }
    }
    entropy
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn jensen_shannon_precomputed_avx2_fma(
    normalized_query: &[f32],
    candidate: &[f32],
    query_entropy: f32,
    candidate_inv_mass: f32,
    candidate_entropy: f32,
) -> f32 {
    use std::arch::x86_64::*;

    let chunks = normalized_query.len() / 8;
    let mut lanes = [0.0f32; 8];
    let mut acc = _mm256_setzero_ps();
    let inv_mass = _mm256_set1_ps(candidate_inv_mass);
    let min_normal = _mm256_set1_ps(f32::MIN_POSITIVE);
    for i in 0..chunks {
        let offset = i * 8;
        let p = _mm256_loadu_ps(normalized_query.as_ptr().add(offset));
        let q = _mm256_mul_ps(_mm256_loadu_ps(candidate.as_ptr().add(offset)), inv_mass);
        let sum = _mm256_add_ps(p, q);
        let log_sum = fast_ln_compact_avx2_fma(_mm256_max_ps(sum, min_normal));
        acc = _mm256_fmadd_ps(sum, log_sum, acc);
    }
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut mixture_entropy_term: f32 = lanes.into_iter().sum();
    for i in chunks * 8..normalized_query.len() {
        let sum = normalized_query[i] + candidate[i] * candidate_inv_mass;
        if sum > 0.0 {
            mixture_entropy_term += sum * sum.ln();
        }
    }
    let divergence = (std::f32::consts::LN_2
        + 0.5 * (query_entropy + candidate_entropy - mixture_entropy_term))
        .max(0.0);
    if divergence <= JENSEN_SHANNON_STABLE_DIVERGENCE {
        jensen_shannon_normalized_query_f32(normalized_query, candidate, candidate_inv_mass)
    } else {
        divergence.sqrt()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn jensen_shannon_precomputed_batch2_divergence_avx2_fma(
    normalized_query: &[f32],
    candidate0: &[f32],
    candidate1: &[f32],
    query_entropy: f32,
    stats0: (f32, f32),
    stats1: (f32, f32),
) -> [f32; 2] {
    use std::arch::x86_64::*;

    let chunks = normalized_query.len() / 8;
    let mut lanes0 = [0.0f32; 8];
    let mut lanes1 = [0.0f32; 8];
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let inv0 = _mm256_set1_ps(stats0.0);
    let inv1 = _mm256_set1_ps(stats1.0);
    let min_normal = _mm256_set1_ps(f32::MIN_POSITIVE);
    for i in 0..chunks {
        let offset = i * 8;
        let p = _mm256_loadu_ps(normalized_query.as_ptr().add(offset));
        let q0 = _mm256_mul_ps(_mm256_loadu_ps(candidate0.as_ptr().add(offset)), inv0);
        let q1 = _mm256_mul_ps(_mm256_loadu_ps(candidate1.as_ptr().add(offset)), inv1);
        let sum0 = _mm256_add_ps(p, q0);
        let sum1 = _mm256_add_ps(p, q1);
        let log0 = fast_ln_compact_avx2_fma(_mm256_max_ps(sum0, min_normal));
        let log1 = fast_ln_compact_avx2_fma(_mm256_max_ps(sum1, min_normal));
        acc0 = _mm256_fmadd_ps(sum0, log0, acc0);
        acc1 = _mm256_fmadd_ps(sum1, log1, acc1);
    }
    _mm256_storeu_ps(lanes0.as_mut_ptr(), acc0);
    _mm256_storeu_ps(lanes1.as_mut_ptr(), acc1);
    let mut mixture0: f32 = lanes0.into_iter().sum();
    let mut mixture1: f32 = lanes1.into_iter().sum();
    for i in chunks * 8..normalized_query.len() {
        let p = normalized_query[i];
        let sum0 = p + candidate0[i] * stats0.0;
        let sum1 = p + candidate1[i] * stats1.0;
        if sum0 > 0.0 {
            mixture0 += sum0 * sum0.ln();
        }
        if sum1 > 0.0 {
            mixture1 += sum1 * sum1.ln();
        }
    }
    let divergence0 =
        (std::f32::consts::LN_2 + 0.5 * (query_entropy + stats0.1 - mixture0)).max(0.0);
    let divergence1 =
        (std::f32::consts::LN_2 + 0.5 * (query_entropy + stats1.1 - mixture1)).max(0.0);
    [
        if divergence0 <= JENSEN_SHANNON_STABLE_DIVERGENCE {
            let distance =
                jensen_shannon_normalized_query_f32(normalized_query, candidate0, stats0.0);
            distance * distance
        } else {
            divergence0
        },
        if divergence1 <= JENSEN_SHANNON_STABLE_DIVERGENCE {
            let distance =
                jensen_shannon_normalized_query_f32(normalized_query, candidate1, stats1.0);
            distance * distance
        } else {
            divergence1
        },
    ]
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn manhattan_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let chunks = a.len() / 4;
    let mut sum;
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            acc = vaddq_f32(acc, vabdq_f32(va, vb));
        }
        sum = vaddvq_f32(acc);
    }
    for i in chunks * 4..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fast_ln_neon(mut x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    unsafe {
        let bits = vreinterpretq_u32_f32(x);
        let exponent_bits = vshrq_n_u32(bits, 23);
        let mantissa_bits = vorrq_u32(
            vandq_u32(bits, vdupq_n_u32(0x007f_ffff)),
            vdupq_n_u32(0x3f00_0000),
        );
        x = vreinterpretq_f32_u32(mantissa_bits);

        let mut exponent = vcvtq_f32_s32(vsubq_s32(
            vreinterpretq_s32_u32(exponent_bits),
            vdupq_n_s32(0x7f),
        ));
        exponent = vaddq_f32(exponent, vdupq_n_f32(1.0));

        let mask = vcltq_f32(x, vdupq_n_f32(std::f32::consts::FRAC_1_SQRT_2));
        let tmp = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), mask));
        x = vsubq_f32(x, vdupq_n_f32(1.0));
        exponent = vsubq_f32(
            exponent,
            vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vdupq_n_f32(1.0)), mask)),
        );
        x = vaddq_f32(x, tmp);

        let z = vmulq_f32(x, x);
        let mut y = vdupq_n_f32(7.037_683_6E-2);
        y = vfmaq_f32(vdupq_n_f32(-1.151_461E-1), y, x);
        y = vfmaq_f32(vdupq_n_f32(1.167_699_84E-1), y, x);
        y = vfmaq_f32(vdupq_n_f32(-1.242_014_1E-1), y, x);
        y = vfmaq_f32(vdupq_n_f32(1.424_932_3E-1), y, x);
        y = vfmaq_f32(vdupq_n_f32(-1.666_805_7E-1), y, x);
        y = vfmaq_f32(vdupq_n_f32(2.000_071_4E-1), y, x);
        y = vfmaq_f32(vdupq_n_f32(-2.499_999_4E-1), y, x);
        y = vfmaq_f32(vdupq_n_f32(3.333_333E-1), y, x);
        y = vmulq_f32(vmulq_f32(y, x), z);
        y = vfmaq_n_f32(y, exponent, -2.121_944_4E-4);
        y = vfmsq_n_f32(y, z, 0.5);
        x = vaddq_f32(x, y);
        vfmaq_n_f32(x, exponent, 0.693_359_4)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fast_ln_compact_neon(mut x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    unsafe {
        let bits = vreinterpretq_u32_f32(x);
        let exponent_bits = vshrq_n_u32(bits, 23);
        let mantissa_bits = vorrq_u32(
            vandq_u32(bits, vdupq_n_u32(0x007f_ffff)),
            vdupq_n_u32(0x3f00_0000),
        );
        x = vreinterpretq_f32_u32(mantissa_bits);
        let mut exponent = vcvtq_f32_s32(vsubq_s32(
            vreinterpretq_s32_u32(exponent_bits),
            vdupq_n_s32(0x7f),
        ));
        exponent = vaddq_f32(exponent, vdupq_n_f32(1.0));
        let mask = vcltq_f32(x, vdupq_n_f32(std::f32::consts::FRAC_1_SQRT_2));
        let tmp = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), mask));
        x = vsubq_f32(x, vdupq_n_f32(1.0));
        exponent = vsubq_f32(
            exponent,
            vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vdupq_n_f32(1.0)), mask)),
        );
        x = vaddq_f32(x, tmp);

        let z = vmulq_f32(x, x);
        let mut y = vdupq_n_f32(-1.242_014_1E-1);
        y = vfmaq_f32(vdupq_n_f32(1.424_932_3E-1), y, x);
        y = vfmaq_f32(vdupq_n_f32(-1.666_805_7E-1), y, x);
        y = vfmaq_f32(vdupq_n_f32(2.000_071_4E-1), y, x);
        y = vfmaq_f32(vdupq_n_f32(-2.499_999_4E-1), y, x);
        y = vfmaq_f32(vdupq_n_f32(3.333_333E-1), y, x);
        y = vmulq_f32(vmulq_f32(y, x), z);
        y = vfmaq_n_f32(y, exponent, -2.121_944_4E-4);
        y = vfmsq_n_f32(y, z, 0.5);
        x = vaddq_f32(x, y);
        vfmaq_n_f32(x, exponent, 0.693_359_4)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn jensen_shannon_neon(a: &[f32], b: &[f32], inv_a: f32, inv_b: f32) -> f32 {
    use std::arch::aarch64::*;

    let chunks = a.len() / 4;
    let mut divergence;
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let inv_a_vec = vdupq_n_f32(inv_a);
        let inv_b_vec = vdupq_n_f32(inv_b);
        let half = vdupq_n_f32(0.5);
        let min_normal = vdupq_n_f32(f32::MIN_POSITIVE);
        for i in 0..chunks {
            let offset = i * 4;
            let p = vmulq_f32(vld1q_f32(a.as_ptr().add(offset)), inv_a_vec);
            let q = vmulq_f32(vld1q_f32(b.as_ptr().add(offset)), inv_b_vec);
            let m = vmulq_f32(vaddq_f32(p, q), half);
            let safe_m = vmaxq_f32(m, min_normal);
            let log_p = fast_ln_neon(vdivq_f32(vmaxq_f32(p, min_normal), safe_m));
            let log_q = fast_ln_neon(vdivq_f32(vmaxq_f32(q, min_normal), safe_m));
            let terms = vaddq_f32(vmulq_f32(p, log_p), vmulq_f32(q, log_q));
            acc = vfmaq_f32(acc, terms, half);
        }
        divergence = vaddvq_f32(acc);
    }
    for i in chunks * 4..a.len() {
        let p = a[i] * inv_a;
        let q = b[i] * inv_b;
        let m = 0.5 * (p + q);
        if p > 0.0 {
            divergence += 0.5 * p * (p / m).ln();
        }
        if q > 0.0 {
            divergence += 0.5 * q * (q / m).ln();
        }
    }
    divergence.max(0.0).sqrt()
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn probability_entropy_neon(row: &[f32], inv_mass: f32) -> f32 {
    use std::arch::aarch64::*;

    let chunks = row.len() / 4;
    let mut entropy;
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let inv_mass_vec = vdupq_n_f32(inv_mass);
        let min_normal = vdupq_n_f32(f32::MIN_POSITIVE);
        for i in 0..chunks {
            let p = vmulq_f32(vld1q_f32(row.as_ptr().add(i * 4)), inv_mass_vec);
            let log_p = fast_ln_compact_neon(vmaxq_f32(p, min_normal));
            acc = vfmaq_f32(acc, p, log_p);
        }
        entropy = vaddvq_f32(acc);
    }
    for &value in &row[chunks * 4..] {
        let p = value * inv_mass;
        if p > 0.0 {
            entropy += p * p.ln();
        }
    }
    entropy
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn jensen_shannon_precomputed_neon(
    normalized_query: &[f32],
    candidate: &[f32],
    query_entropy: f32,
    candidate_inv_mass: f32,
    candidate_entropy: f32,
) -> f32 {
    use std::arch::aarch64::*;

    let chunks = normalized_query.len() / 4;
    let mut mixture_entropy_term;
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let inv_mass = vdupq_n_f32(candidate_inv_mass);
        let min_normal = vdupq_n_f32(f32::MIN_POSITIVE);
        for i in 0..chunks {
            let offset = i * 4;
            let p = vld1q_f32(normalized_query.as_ptr().add(offset));
            let q = vmulq_f32(vld1q_f32(candidate.as_ptr().add(offset)), inv_mass);
            let sum = vaddq_f32(p, q);
            let log_sum = fast_ln_compact_neon(vmaxq_f32(sum, min_normal));
            acc = vfmaq_f32(acc, sum, log_sum);
        }
        mixture_entropy_term = vaddvq_f32(acc);
    }
    for i in chunks * 4..normalized_query.len() {
        let sum = normalized_query[i] + candidate[i] * candidate_inv_mass;
        if sum > 0.0 {
            mixture_entropy_term += sum * sum.ln();
        }
    }
    let divergence = (std::f32::consts::LN_2
        + 0.5 * (query_entropy + candidate_entropy - mixture_entropy_term))
        .max(0.0);
    if divergence <= JENSEN_SHANNON_STABLE_DIVERGENCE {
        jensen_shannon_normalized_query_f32(normalized_query, candidate, candidate_inv_mass)
    } else {
        divergence.sqrt()
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn jensen_shannon_precomputed_batch2_divergence_neon(
    normalized_query: &[f32],
    candidate0: &[f32],
    candidate1: &[f32],
    query_entropy: f32,
    stats0: (f32, f32),
    stats1: (f32, f32),
) -> [f32; 2] {
    use std::arch::aarch64::*;

    let chunks = normalized_query.len() / 4;
    let (mut mixture0, mut mixture1);
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let inv0 = vdupq_n_f32(stats0.0);
        let inv1 = vdupq_n_f32(stats1.0);
        let min_normal = vdupq_n_f32(f32::MIN_POSITIVE);
        for i in 0..chunks {
            let offset = i * 4;
            let p = vld1q_f32(normalized_query.as_ptr().add(offset));
            let q0 = vmulq_f32(vld1q_f32(candidate0.as_ptr().add(offset)), inv0);
            let q1 = vmulq_f32(vld1q_f32(candidate1.as_ptr().add(offset)), inv1);
            let sum0 = vaddq_f32(p, q0);
            let sum1 = vaddq_f32(p, q1);
            let log0 = fast_ln_compact_neon(vmaxq_f32(sum0, min_normal));
            let log1 = fast_ln_compact_neon(vmaxq_f32(sum1, min_normal));
            acc0 = vfmaq_f32(acc0, sum0, log0);
            acc1 = vfmaq_f32(acc1, sum1, log1);
        }
        mixture0 = vaddvq_f32(acc0);
        mixture1 = vaddvq_f32(acc1);
    }
    for i in chunks * 4..normalized_query.len() {
        let p = normalized_query[i];
        let sum0 = p + candidate0[i] * stats0.0;
        let sum1 = p + candidate1[i] * stats1.0;
        if sum0 > 0.0 {
            mixture0 += sum0 * sum0.ln();
        }
        if sum1 > 0.0 {
            mixture1 += sum1 * sum1.ln();
        }
    }
    let divergence0 =
        (std::f32::consts::LN_2 + 0.5 * (query_entropy + stats0.1 - mixture0)).max(0.0);
    let divergence1 =
        (std::f32::consts::LN_2 + 0.5 * (query_entropy + stats1.1 - mixture1)).max(0.0);
    [
        if divergence0 <= JENSEN_SHANNON_STABLE_DIVERGENCE {
            let distance =
                jensen_shannon_normalized_query_f32(normalized_query, candidate0, stats0.0);
            distance * distance
        } else {
            divergence0
        },
        if divergence1 <= JENSEN_SHANNON_STABLE_DIVERGENCE {
            let distance =
                jensen_shannon_normalized_query_f32(normalized_query, candidate1, stats1.0);
            distance * distance
        } else {
            divergence1
        },
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn chebyshev_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let chunks = a.len() / 8;
    let mut lanes = [0.0f32; 8];
    let mut acc = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm256_andnot_ps(sign_mask, _mm256_sub_ps(va, vb));
        acc = _mm256_max_ps(acc, diff);
    }
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut maximum = lanes.into_iter().fold(0.0, f32::max);
    for i in chunks * 8..a.len() {
        maximum = maximum.max((a[i] - b[i]).abs());
    }
    maximum
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn chebyshev_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let chunks = a.len() / 4;
    let mut maximum;
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            acc = vmaxq_f32(acc, vabdq_f32(va, vb));
        }
        maximum = vmaxvq_f32(acc);
    }
    for i in chunks * 4..a.len() {
        maximum = maximum.max((a[i] - b[i]).abs());
    }
    maximum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn canberra_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let chunks = a.len() / 8;
    let mut lanes = [0.0f32; 8];
    let mut acc = _mm256_setzero_ps();
    let zero = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let numerator = _mm256_andnot_ps(sign_mask, _mm256_sub_ps(va, vb));
        let abs_a = _mm256_andnot_ps(sign_mask, va);
        let abs_b = _mm256_andnot_ps(sign_mask, vb);
        let denominator = _mm256_add_ps(abs_a, abs_b);
        let nonzero = _mm256_cmp_ps(denominator, zero, _CMP_NEQ_OQ);
        let quotient = _mm256_div_ps(numerator, denominator);
        acc = _mm256_add_ps(acc, _mm256_and_ps(nonzero, quotient));
    }
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum: f32 = lanes.into_iter().sum();
    for i in chunks * 8..a.len() {
        let denominator = a[i].abs() + b[i].abs();
        if denominator != 0.0 {
            sum += (a[i] - b[i]).abs() / denominator;
        }
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn canberra_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let chunks = a.len() / 4;
    let mut sum;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let mut acc = zero;
        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            let denominator = vaddq_f32(vabsq_f32(va), vabsq_f32(vb));
            let quotient = vdivq_f32(vabdq_f32(va, vb), denominator);
            acc = vaddq_f32(acc, vbslq_f32(vcgtq_f32(denominator, zero), quotient, zero));
        }
        sum = vaddvq_f32(acc);
    }
    for i in chunks * 4..a.len() {
        let denominator = a[i].abs() + b[i].abs();
        if denominator != 0.0 {
            sum += (a[i] - b[i]).abs() / denominator;
        }
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bray_curtis_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let chunks = a.len() / 8;
    let mut numerator_lanes = [0.0f32; 8];
    let mut denominator_lanes = [0.0f32; 8];
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut numerator_acc = _mm256_setzero_ps();
    let mut denominator_acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        numerator_acc = _mm256_add_ps(
            numerator_acc,
            _mm256_andnot_ps(sign_mask, _mm256_sub_ps(va, vb)),
        );
        denominator_acc = _mm256_add_ps(
            denominator_acc,
            _mm256_andnot_ps(sign_mask, _mm256_add_ps(va, vb)),
        );
    }
    _mm256_storeu_ps(numerator_lanes.as_mut_ptr(), numerator_acc);
    _mm256_storeu_ps(denominator_lanes.as_mut_ptr(), denominator_acc);
    let mut numerator: f32 = numerator_lanes.into_iter().sum();
    let mut denominator: f32 = denominator_lanes.into_iter().sum();
    for i in chunks * 8..a.len() {
        numerator += (a[i] - b[i]).abs();
        denominator += (a[i] + b[i]).abs();
    }
    if denominator == 0.0 {
        if numerator == 0.0 {
            0.0
        } else {
            f32::INFINITY
        }
    } else {
        numerator / denominator
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn bray_curtis_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let chunks = a.len() / 4;
    let (mut numerator, mut denominator);
    unsafe {
        let mut numerator_acc = vdupq_n_f32(0.0);
        let mut denominator_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            numerator_acc = vaddq_f32(numerator_acc, vabdq_f32(va, vb));
            denominator_acc = vaddq_f32(denominator_acc, vabsq_f32(vaddq_f32(va, vb)));
        }
        numerator = vaddvq_f32(numerator_acc);
        denominator = vaddvq_f32(denominator_acc);
    }
    for i in chunks * 4..a.len() {
        numerator += (a[i] - b[i]).abs();
        denominator += (a[i] + b[i]).abs();
    }
    if denominator == 0.0 {
        if numerator == 0.0 {
            0.0
        } else {
            f32::INFINITY
        }
    } else {
        numerator / denominator
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
    fn test_l2_squared_batch8_matches_scalar() {
        let dim = 128;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let rows: Vec<Vec<f32>> = (0..8)
            .map(|row| {
                (0..dim)
                    .map(|i| ((row * 13 + i * 7) as f32 % 31.0 - 15.0) * 0.02)
                    .collect()
            })
            .collect();

        let batch = l2_squared_batch8_f32(
            &query, &rows[0], &rows[1], &rows[2], &rows[3], &rows[4], &rows[5], &rows[6], &rows[7],
        );
        for i in 0..8 {
            let expected = l2_squared_f32(&query, &rows[i]);
            assert!((batch[i] - expected).abs() < 1e-4);
        }
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
    fn test_domain_distances() {
        assert!((manhattan_f32(&[1.0, 2.0], &[4.0, 0.0]) - 5.0).abs() < 1e-6);
        assert!(correlation_distance_f32(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]) < 1e-6);
        assert!((correlation_distance_f32(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]) - 2.0).abs() < 1e-6);
        assert!(hellinger_distance_f32(&[1.0, 0.0], &[1.0, 0.0]) < 1e-6);
        assert!((hellinger_distance_f32(&[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-6);
        assert!((wasserstein_1d_f32(&[1.0, 0.0, 0.0], &[0.0, 0.0, 1.0]) - 2.0).abs() < 1e-6);
        assert!((dice_distance_f32(&[1.0, 1.0, 0.0], &[1.0, 0.0, 1.0]) - 0.5).abs() < 1e-6);
        assert!(
            (jensen_shannon_distance_f32(&[1.0, 0.0], &[0.0, 1.0]) - std::f32::consts::LN_2.sqrt())
                .abs()
                < 1e-6
        );
        assert!((chebyshev_f32(&[1.0, 2.0, 3.0], &[4.0, 0.0, 3.0]) - 3.0).abs() < 1e-6);
        assert!((canberra_f32(&[1.0, 0.0, 3.0], &[2.0, 0.0, 1.0]) - 5.0 / 6.0).abs() < 1e-6);
        assert!((bray_curtis_f32(&[1.0, 2.0], &[2.0, 4.0]) - 1.0 / 3.0).abs() < 1e-6);
        assert!(jensen_shannon_distance_f32(&[-1.0, 2.0], &[1.0, 2.0]).is_infinite());
    }

    #[test]
    fn test_haversine_uses_geojson_order_and_meters() {
        // Shanghai to Beijing is about 1,068 km on a mean-radius sphere.
        let shanghai = [121.4737, 31.2304];
        let beijing = [116.4074, 39.9042];
        let meters = haversine_meters_f32(&shanghai, &beijing);
        assert!((meters - 1_067_000.0).abs() < 10_000.0, "{meters}");
        assert_eq!(haversine_meters_f32(&shanghai, &shanghai), 0.0);
    }

    #[test]
    fn test_jensen_shannon_simd_matches_f64_reference() {
        fn reference(a: &[f32], b: &[f32]) -> f32 {
            let sum_a: f64 = a.iter().map(|&x| x as f64).sum();
            let sum_b: f64 = b.iter().map(|&x| x as f64).sum();
            let mut divergence = 0.0f64;
            for (&av, &bv) in a.iter().zip(b) {
                let p = av as f64 / sum_a;
                let q = bv as f64 / sum_b;
                let m = 0.5 * (p + q);
                if p > 0.0 {
                    divergence += 0.5 * p * (p / m).ln();
                }
                if q > 0.0 {
                    divergence += 0.5 * q * (q / m).ln();
                }
            }
            divergence.max(0.0).sqrt() as f32
        }

        let mut state = 0x1234_5678u32;
        for dim in [1, 3, 4, 7, 8, 16, 127, 128, 257] {
            for round in 0..20 {
                let mut a = Vec::with_capacity(dim);
                let mut b = Vec::with_capacity(dim);
                for i in 0..dim {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let av = ((state >> 8) as f32 + 1.0) / 16_777_216.0;
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let bv = ((state >> 8) as f32 + 1.0) / 16_777_216.0;
                    a.push(if (i + round) % 11 == 0 { 0.0 } else { av });
                    b.push(if (i + round) % 13 == 0 { 0.0 } else { bv });
                }
                // Keep one-dimensional zero patterns from becoming zero-mass.
                if a.iter().all(|&x| x == 0.0) {
                    a[0] = 1.0;
                }
                if b.iter().all(|&x| x == 0.0) {
                    b[0] = 1.0;
                }
                let actual = jensen_shannon_distance_f32(&a, &b);
                let expected = reference(&a, &b);
                assert!(
                    (actual - expected).abs() <= 2e-5,
                    "dim={dim}, actual={actual}, expected={expected}"
                );
                assert!((actual - jensen_shannon_distance_f32(&b, &a)).abs() <= 2e-5);

                let (inv_a, entropy_a) = probability_row_stats_f32(&a);
                let (inv_b, entropy_b) = probability_row_stats_f32(&b);
                let normalized_a: Vec<f32> = a.iter().map(|&value| value * inv_a).collect();
                let cached =
                    jensen_shannon_precomputed_f32(&normalized_a, &b, entropy_a, inv_b, entropy_b);
                assert!(
                    (cached - expected).abs() <= 3e-5,
                    "cached dim={dim}, actual={cached}, expected={expected}"
                );
                let batch = jensen_shannon_precomputed_batch2_f32(
                    &normalized_a,
                    &b,
                    &a,
                    entropy_a,
                    (inv_b, entropy_b),
                    (inv_a, entropy_a),
                );
                assert!((batch[0] - expected).abs() <= 3e-5);
                assert!(batch[1] <= 1e-6, "self distance={}", batch[1]);
            }
        }

        let near_a: Vec<f32> = (1..=128).map(|value| value as f32).collect();
        let near_b: Vec<f32> = near_a
            .iter()
            .enumerate()
            .map(|(index, &value)| value * (1.0 + (index as f32 % 3.0 - 1.0) * 1e-4))
            .collect();
        let (near_inv_a, near_entropy_a) = probability_row_stats_f32(&near_a);
        let (near_inv_b, near_entropy_b) = probability_row_stats_f32(&near_b);
        let normalized_near_a: Vec<f32> = near_a.iter().map(|&value| value * near_inv_a).collect();
        let near_actual = jensen_shannon_precomputed_f32(
            &normalized_near_a,
            &near_b,
            near_entropy_a,
            near_inv_b,
            near_entropy_b,
        );
        let near_expected = reference(&near_a, &near_b);
        assert!(
            (near_actual - near_expected).abs() <= 2e-5,
            "near actual={near_actual}, expected={near_expected}"
        );

        let tiny = f32::from_bits(1);
        assert!(
            (jensen_shannon_distance_f32(&[tiny, 0.0], &[0.0, tiny])
                - std::f32::consts::LN_2.sqrt())
            .abs()
                <= 1e-6
        );
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

    #[test]
    fn test_inner_product_batch4() {
        let dim = 128;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let v0: Vec<f32> = (0..dim).map(|i| ((i + 1) as f32) * 0.01).collect();
        let v1: Vec<f32> = (0..dim).map(|i| ((i + 2) as f32) * 0.01).collect();
        let v2: Vec<f32> = (0..dim).map(|i| ((i + 3) as f32) * 0.01).collect();
        let v3: Vec<f32> = (0..dim).map(|i| ((i + 4) as f32) * 0.01).collect();
        let batch = inner_product_batch4_f32(&query, &v0, &v1, &v2, &v3);
        assert!((batch[0] - inner_product_f32(&query, &v0)).abs() < 1e-3);
        assert!((batch[1] - inner_product_f32(&query, &v1)).abs() < 1e-3);
        assert!((batch[2] - inner_product_f32(&query, &v2)).abs() < 1e-3);
        assert!((batch[3] - inner_product_f32(&query, &v3)).abs() < 1e-3);
    }
}
