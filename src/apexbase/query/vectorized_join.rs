// Vectorized JOIN utilities and filter optimization
// This module provides helper functions for vectorized operations

use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array};

/// Zone map for column statistics
#[derive(Debug, Clone)]
pub struct ZoneMapStats {
    pub min: Option<i64>,
    pub max: Option<i64>,
    pub null_count: usize,
}

impl ZoneMapStats {
    pub fn from_int64_array(arr: &Int64Array) -> Self {
        if arr.is_empty() {
            return Self {
                min: None,
                max: None,
                null_count: 0,
            };
        }

        let values = arr.values();
        let mut min = i64::MAX;
        let mut max = i64::MIN;

        for &v in values {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }

        Self {
            min: Some(min),
            max: Some(max),
            null_count: arr.null_count(),
        }
    }

    pub fn from_float64_array(arr: &Float64Array) -> Self {
        if arr.is_empty() {
            return Self {
                min: None,
                max: None,
                null_count: 0,
            };
        }

        let values = arr.values();
        let mut min = f64::MAX;
        let mut max = f64::MIN;

        for &v in values {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }

        Self {
            min: Some(min as i64), // Convert f64 to i64 for comparison
            max: Some(max as i64),
            null_count: arr.null_count(),
        }
    }

    /// Check if the zone can possibly match a predicate
    pub fn can_match_predicate(&self, predicate: i64, is_greater: bool) -> bool {
        match (self.min, self.max, is_greater) {
            (Some(min), Some(max), true) => max > predicate, // v > predicate, need max > predicate
            (Some(min), Some(max), false) => min < predicate, // v < predicate, need min < predicate
            _ => true,                                       // Unknown range, assume can match
        }
    }
}

/// Optimized filter evaluation with early exit
pub mod filter_optimize {
    use arrow::array::{Array, BooleanArray};

    /// Check if a boolean array is all false (no matches)
    #[inline]
    pub fn is_all_false(mask: &BooleanArray) -> bool {
        if mask.null_count() == mask.len() {
            return false; // All null = no matches
        }

        // Fast path: check first few values
        let check_len = mask.len().min(64);
        for i in 0..check_len {
            if !mask.is_null(i) && mask.value(i) {
                return false;
            }
        }

        // Full check if needed
        for i in check_len..mask.len() {
            if !mask.is_null(i) && mask.value(i) {
                return false;
            }
        }

        true
    }

    /// Check if a boolean array is all true (select all)
    #[inline]
    pub fn is_all_true(mask: &BooleanArray) -> bool {
        // Empty array is not "all true"
        if mask.len() == 0 {
            return false;
        }

        // If there are nulls, we can't say it's all true
        if mask.null_count() > 0 {
            return false;
        }

        // Check if any value is false
        for i in 0..mask.len() {
            if !mask.value(i) {
                return false;
            }
        }

        true
    }

    /// Quick null check for arrays
    #[inline]
    pub fn has_any_null(mask: &BooleanArray) -> bool {
        mask.null_count() > 0
    }
}

/// SIMD-optimized batch filter for Int64 arrays
/// Returns indices where values match the predicate
/// is_greater: true for > or >=, false for < or <=
/// is_equal: true for >= or <= (inclusive), false for > or < (exclusive)
pub fn filter_int64_batch(
    values: &[i64],
    predicate: i64,
    is_greater: bool,
    is_equal: bool,
) -> Vec<usize> {
    let mut result = Vec::with_capacity(values.len() / 2);

    if is_greater {
        if is_equal {
            // >=
            for (i, &v) in values.iter().enumerate() {
                if v >= predicate {
                    result.push(i);
                }
            }
        } else {
            // >
            for (i, &v) in values.iter().enumerate() {
                if v > predicate {
                    result.push(i);
                }
            }
        }
    } else {
        if is_equal {
            // <=
            for (i, &v) in values.iter().enumerate() {
                if v <= predicate {
                    result.push(i);
                }
            }
        } else {
            // <
            for (i, &v) in values.iter().enumerate() {
                if v < predicate {
                    result.push(i);
                }
            }
        }
    }

    result
}

/// SIMD-optimized batch filter for Float64 arrays
/// is_greater: true for > or >=, false for < or <=
/// is_equal: true for >= or <= (inclusive), false for > or < (exclusive)
pub fn filter_float64_batch(
    values: &[f64],
    predicate: f64,
    is_greater: bool,
    is_equal: bool,
) -> Vec<usize> {
    let mut result = Vec::with_capacity(values.len() / 2);

    if is_greater {
        if is_equal {
            // >=
            for (i, &v) in values.iter().enumerate() {
                if v >= predicate {
                    result.push(i);
                }
            }
        } else {
            // >
            for (i, &v) in values.iter().enumerate() {
                if v > predicate {
                    result.push(i);
                }
            }
        }
    } else {
        if is_equal {
            // <=
            for (i, &v) in values.iter().enumerate() {
                if v <= predicate {
                    result.push(i);
                }
            }
        } else {
            // <
            for (i, &v) in values.iter().enumerate() {
                if v < predicate {
                    result.push(i);
                }
            }
        }
    }

    result
}

/// Batch count of matching values (faster than filtering for count-only queries)
pub fn count_matching_int64(values: &[i64], predicate: i64, op: &str) -> usize {
    match op {
        ">" => values.iter().filter(|&&v| v > predicate).count(),
        "<" => values.iter().filter(|&&v| v < predicate).count(),
        ">=" => values.iter().filter(|&&v| v >= predicate).count(),
        "<=" => values.iter().filter(|&&v| v <= predicate).count(),
        "=" | "==" => values.iter().filter(|&&v| v == predicate).count(),
        "!=" => values.iter().filter(|&&v| v != predicate).count(),
        _ => 0,
    }
}

/// Batch count of matching float values
pub fn count_matching_float64(values: &[f64], predicate: f64, op: &str) -> usize {
    match op {
        ">" => values.iter().filter(|&&v| v > predicate).count(),
        "<" => values.iter().filter(|&&v| v < predicate).count(),
        ">=" => values.iter().filter(|&&v| v >= predicate).count(),
        "<=" => values.iter().filter(|&&v| v <= predicate).count(),
        "=" | "==" => values
            .iter()
            .filter(|&&v| (v - predicate).abs() < f64::EPSILON)
            .count(),
        "!=" => values
            .iter()
            .filter(|&&v| (v - predicate).abs() >= f64::EPSILON)
            .count(),
        _ => 0,
    }
}

/// Check if all values in array match a condition (for early exit optimization)
pub fn all_values_match_int64(values: &[i64], predicate: i64, op: &str) -> Option<bool> {
    if values.is_empty() {
        return None;
    }

    match op {
        ">" => {
            for &v in values.iter().take(64) {
                if !(v > predicate) {
                    return Some(false);
                }
            }
            for &v in &values[64..] {
                if !(v > predicate) {
                    return Some(false);
                }
            }
            Some(true)
        }
        "<" => {
            for &v in values.iter().take(64) {
                if !(v < predicate) {
                    return Some(false);
                }
            }
            for &v in &values[64..] {
                if !(v < predicate) {
                    return Some(false);
                }
            }
            Some(true)
        }
        ">=" => {
            for &v in values.iter().take(64) {
                if !(v >= predicate) {
                    return Some(false);
                }
            }
            for &v in &values[64..] {
                if !(v >= predicate) {
                    return Some(false);
                }
            }
            Some(true)
        }
        "<=" => {
            for &v in values.iter().take(64) {
                if !(v <= predicate) {
                    return Some(false);
                }
            }
            for &v in &values[64..] {
                if !(v <= predicate) {
                    return Some(false);
                }
            }
            Some(true)
        }
        _ => None,
    }
}

/// Check if any value matches (for early exit)
pub fn any_value_matches_int64(values: &[i64], predicate: i64, op: &str) -> Option<bool> {
    if values.is_empty() {
        return Some(false);
    }

    match op {
        ">" => {
            for &v in values.iter().take(64) {
                if v > predicate {
                    return Some(true);
                }
            }
            for &v in &values[64..] {
                if v > predicate {
                    return Some(true);
                }
            }
            Some(false)
        }
        "<" => {
            for &v in values.iter().take(64) {
                if v < predicate {
                    return Some(true);
                }
            }
            for &v in &values[64..] {
                if v < predicate {
                    return Some(true);
                }
            }
            Some(false)
        }
        ">=" => {
            for &v in values.iter().take(64) {
                if v >= predicate {
                    return Some(true);
                }
            }
            for &v in &values[64..] {
                if v >= predicate {
                    return Some(true);
                }
            }
            Some(false)
        }
        "<=" => {
            for &v in values.iter().take(64) {
                if v <= predicate {
                    return Some(true);
                }
            }
            for &v in &values[64..] {
                if v <= predicate {
                    return Some(true);
                }
            }
            Some(false)
        }
        _ => None,
    }
}

/// Re-export filter_optimize for external use
pub use filter_optimize::{has_any_null, is_all_false, is_all_true};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zone_map_int64() {
        let arr = Int64Array::from(vec![1, 2, 3, 4, 5]);
        let stats = ZoneMapStats::from_int64_array(&arr);
        assert_eq!(stats.min, Some(1));
        assert_eq!(stats.max, Some(5));
        assert_eq!(stats.null_count, 0);
    }

    #[test]
    fn test_filter_optimize_all_false() {
        let mask = BooleanArray::from(vec![false, false, false]);
        assert!(is_all_false(&mask));
        assert!(!is_all_true(&mask));
    }

    #[test]
    fn test_filter_optimize_all_true() {
        let mask = BooleanArray::from(vec![true, true, true]);
        assert!(!is_all_false(&mask));
        assert!(is_all_true(&mask));
    }

    #[test]
    fn test_count_matching() {
        let values = vec![1i64, 2, 3, 4, 5];
        assert_eq!(count_matching_int64(&values, 3, ">"), 2); // 4, 5
        assert_eq!(count_matching_int64(&values, 3, "<"), 2); // 1, 2
        assert_eq!(count_matching_int64(&values, 3, ">="), 3); // 3, 4, 5
    }
}
