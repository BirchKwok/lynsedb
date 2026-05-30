//! Vectorized Execution Engine for GROUP BY
//!
//! This module implements DuckDB-style vectorized execution:
//! - Process data in small batches (vectors) of 2048 rows
//! - Stream data through pipeline instead of loading all at once
//! - Use efficient hash aggregation with pre-computed hashes
//!
//! Key optimizations:
//! 1. Cache-friendly batch processing
//! 2. Pre-computed hash values for grouping
//! 3. SIMD-friendly aggregation loops
//! 4. Minimal memory allocations

use ahash::{AHashMap, AHasher};
use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray, UInt64Array,
};
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::hash::{Hash, Hasher};
use std::io;
use std::sync::Arc;

/// Vector size for batch processing (DuckDB uses 2048)
pub const VECTOR_SIZE: usize = 2048;

/// Pre-computed hash for a group key
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GroupHash(u64);

impl GroupHash {
    #[inline(always)]
    pub fn from_i64(val: i64) -> Self {
        // OPTIMIZATION: Use direct bit pattern as hash for integers
        // This avoids AHasher creation overhead for integer keys
        GroupHash(val as u64)
    }

    #[inline(always)]
    pub fn from_u32(val: u32) -> Self {
        // For dictionary indices, use direct value as hash (perfect hash)
        GroupHash(val as u64)
    }

    #[inline(always)]
    pub fn from_str(val: &str) -> Self {
        let mut hasher = AHasher::default();
        val.hash(&mut hasher);
        GroupHash(hasher.finish())
    }

    #[inline(always)]
    pub fn combine(self, other: GroupHash) -> Self {
        // Combine two hashes using XOR and rotation
        GroupHash(self.0.rotate_left(5) ^ other.0)
    }
}

/// Aggregate state for a single group
#[derive(Clone)]
pub struct AggregateState {
    pub count: i64,
    pub sum_int: i64,
    pub sum_float: f64,
    pub min_int: Option<i64>,
    pub max_int: Option<i64>,
    pub min_float: Option<f64>,
    pub max_float: Option<f64>,
    pub first_row_idx: usize,
}

impl AggregateState {
    #[inline(always)]
    pub fn new(first_row_idx: usize) -> Self {
        Self {
            count: 0,
            sum_int: 0,
            sum_float: 0.0,
            min_int: None,
            max_int: None,
            min_float: None,
            max_float: None,
            first_row_idx,
        }
    }

    #[inline(always)]
    pub fn update_int(&mut self, val: i64) {
        self.count += 1;
        self.sum_int = self.sum_int.wrapping_add(val);
        self.min_int = Some(self.min_int.map_or(val, |m| m.min(val)));
        self.max_int = Some(self.max_int.map_or(val, |m| m.max(val)));
    }

    #[inline(always)]
    pub fn update_float(&mut self, val: f64) {
        self.count += 1;
        self.sum_float += val;
        self.min_float = Some(self.min_float.map_or(val, |m| m.min(val)));
        self.max_float = Some(self.max_float.map_or(val, |m| m.max(val)));
    }

    #[inline(always)]
    pub fn update_count(&mut self) {
        self.count += 1;
    }

    /// Merge another state into this one
    #[inline(always)]
    pub fn merge(&mut self, other: &AggregateState) {
        self.count += other.count;
        self.sum_int = self.sum_int.wrapping_add(other.sum_int);
        self.sum_float += other.sum_float;
        if let Some(other_min) = other.min_int {
            self.min_int = Some(self.min_int.map_or(other_min, |m| m.min(other_min)));
        }
        if let Some(other_max) = other.max_int {
            self.max_int = Some(self.max_int.map_or(other_max, |m| m.max(other_max)));
        }
        if let Some(other_min) = other.min_float {
            self.min_float = Some(self.min_float.map_or(other_min, |m| m.min(other_min)));
        }
        if let Some(other_max) = other.max_float {
            self.max_float = Some(self.max_float.map_or(other_max, |m| m.max(other_max)));
        }
    }
}

/// Vectorized hash aggregation table
/// Uses a two-level structure for efficiency:
/// 1. Hash table mapping GroupHash -> group_id
/// 2. Flat vector of AggregateState indexed by group_id
pub struct VectorizedHashAgg {
    /// Hash -> group_id mapping
    hash_table: AHashMap<GroupHash, u32>,
    /// Aggregate states indexed by group_id
    states: Vec<AggregateState>,
    /// Group keys (for result building)
    group_keys_int: Vec<i64>,
    group_keys_str: Vec<String>,
}

/// Fast counting aggregation for low-cardinality integer keys
/// Uses direct array indexing instead of hash table (O(1) vs O(1) but lower constant)
/// Optimized for GROUP BY with small integer ranges (e.g., category IDs 0-1000)
pub struct DirectCountAgg {
    /// Direct count array indexed by (value - min_val)
    counts: Vec<i64>,
    /// Sum values for SUM/AVG
    sums_int: Vec<i64>,
    sums_float: Vec<f64>,
    /// Min value offset
    min_val: i64,
    /// Track which indices have data
    has_data: Vec<bool>,
}

impl DirectCountAgg {
    /// Create with known value range [min_val, max_val]
    pub fn new(min_val: i64, max_val: i64) -> Self {
        let range = (max_val - min_val + 1) as usize;
        Self {
            counts: vec![0; range],
            sums_int: vec![0; range],
            sums_float: vec![0.0; range],
            min_val,
            has_data: vec![false; range],
        }
    }

    #[inline(always)]
    pub fn update_count(&mut self, key: i64) {
        let idx = (key - self.min_val) as usize;
        if idx < self.counts.len() {
            unsafe {
                *self.counts.get_unchecked_mut(idx) += 1;
                *self.has_data.get_unchecked_mut(idx) = true;
            }
        }
    }

    #[inline(always)]
    pub fn update_int(&mut self, key: i64, val: i64) {
        let idx = (key - self.min_val) as usize;
        if idx < self.counts.len() {
            unsafe {
                *self.counts.get_unchecked_mut(idx) += 1;
                *self.sums_int.get_unchecked_mut(idx) += val;
                *self.has_data.get_unchecked_mut(idx) = true;
            }
        }
    }

    #[inline(always)]
    pub fn update_float(&mut self, key: i64, val: f64) {
        let idx = (key - self.min_val) as usize;
        if idx < self.counts.len() {
            unsafe {
                *self.counts.get_unchecked_mut(idx) += 1;
                *self.sums_float.get_unchecked_mut(idx) += val;
                *self.has_data.get_unchecked_mut(idx) = true;
            }
        }
    }

    /// Convert to vectors of (key, count, sum_int, sum_float)
    pub fn to_results(&self) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<f64>) {
        let mut keys = Vec::new();
        let mut counts = Vec::new();
        let mut sums_int = Vec::new();
        let mut sums_float = Vec::new();

        for (i, &has) in self.has_data.iter().enumerate() {
            if has {
                keys.push(self.min_val + i as i64);
                counts.push(self.counts[i]);
                sums_int.push(self.sums_int[i]);
                sums_float.push(self.sums_float[i]);
            }
        }

        (keys, counts, sums_int, sums_float)
    }

    /// Check if direct counting is beneficial for given min/max range
    /// Returns true if range is small enough to fit in L2 cache
    #[inline]
    pub fn is_beneficial(min_val: i64, max_val: i64) -> bool {
        let range = max_val.saturating_sub(min_val) + 1;
        // Use direct counting if range fits in ~256KB (L2 cache friendly)
        // Each entry uses ~25 bytes (count + sum_int + sum_float + has_data)
        range <= 10000 && range > 0
    }
}

impl VectorizedHashAgg {
    pub fn new(is_int_key: bool, estimated_groups: usize) -> Self {
        // OPTIMIZATION: Pre-allocate with 2x estimated capacity to reduce rehashing
        let capacity = (estimated_groups * 2).max(64);
        Self {
            hash_table: AHashMap::with_capacity_and_hasher(capacity, Default::default()),
            states: Vec::with_capacity(estimated_groups),
            group_keys_int: if is_int_key {
                Vec::with_capacity(estimated_groups)
            } else {
                Vec::new()
            },
            group_keys_str: if !is_int_key {
                Vec::with_capacity(estimated_groups)
            } else {
                Vec::new()
            },
        }
    }

    /// Get or create a group, returns the group_id
    #[inline(always)]
    pub fn get_or_create_group_int(&mut self, key: i64, row_idx: usize) -> u32 {
        let hash = GroupHash::from_i64(key);
        if let Some(&group_id) = self.hash_table.get(&hash) {
            group_id
        } else {
            let group_id = self.states.len() as u32;
            self.hash_table.insert(hash, group_id);
            self.states.push(AggregateState::new(row_idx));
            self.group_keys_int.push(key);
            group_id
        }
    }

    /// Get or create a group for string key
    #[inline(always)]
    pub fn get_or_create_group_str(&mut self, key: &str, row_idx: usize) -> u32 {
        let hash = GroupHash::from_str(key);
        if let Some(&group_id) = self.hash_table.get(&hash) {
            group_id
        } else {
            let group_id = self.states.len() as u32;
            self.hash_table.insert(hash, group_id);
            self.states.push(AggregateState::new(row_idx));
            self.group_keys_str.push(key.to_string());
            group_id
        }
    }

    /// Get or create a group for dictionary index (perfect hash)
    #[inline(always)]
    pub fn get_or_create_group_dict(
        &mut self,
        dict_idx: u32,
        key_str: &str,
        row_idx: usize,
    ) -> u32 {
        let hash = GroupHash::from_u32(dict_idx);
        if let Some(&group_id) = self.hash_table.get(&hash) {
            group_id
        } else {
            let group_id = self.states.len() as u32;
            self.hash_table.insert(hash, group_id);
            self.states.push(AggregateState::new(row_idx));
            self.group_keys_str.push(key_str.to_string());
            group_id
        }
    }

    /// Update aggregate state for a group
    #[inline(always)]
    pub fn update_int(&mut self, group_id: u32, val: i64) {
        unsafe {
            self.states
                .get_unchecked_mut(group_id as usize)
                .update_int(val);
        }
    }

    #[inline(always)]
    pub fn update_float(&mut self, group_id: u32, val: f64) {
        unsafe {
            self.states
                .get_unchecked_mut(group_id as usize)
                .update_float(val);
        }
    }

    #[inline(always)]
    pub fn update_count(&mut self, group_id: u32) {
        unsafe {
            self.states
                .get_unchecked_mut(group_id as usize)
                .update_count();
        }
    }

    /// Get number of groups
    pub fn num_groups(&self) -> usize {
        self.states.len()
    }

    /// Get aggregate states
    pub fn states(&self) -> &[AggregateState] {
        &self.states
    }

    /// Get group keys (int)
    pub fn group_keys_int(&self) -> &[i64] {
        &self.group_keys_int
    }

    /// Get group keys (string)
    pub fn group_keys_str(&self) -> &[String] {
        &self.group_keys_str
    }
}

/// Process a vector (batch) of rows for GROUP BY aggregation
/// This is the core vectorized execution function
pub fn process_vector_group_by(
    hash_agg: &mut VectorizedHashAgg,
    // Group column data
    group_col_int: Option<&[i64]>,
    group_col_str: Option<&StringArray>,
    group_col_dict_indices: Option<&[u32]>,
    group_col_dict_values: Option<&[&str]>,
    // Aggregate column data
    agg_col_int: Option<&[i64]>,
    agg_col_float: Option<&[f64]>,
    // Row range in the vector
    start_row: usize,
    end_row: usize,
    // Whether to only count (COUNT(*))
    count_only: bool,
) {
    // FAST PATH 1: Dictionary-encoded string column (perfect hash)
    if let (Some(dict_indices), Some(dict_values)) = (group_col_dict_indices, group_col_dict_values)
    {
        if count_only {
            for row_idx in start_row..end_row {
                let dict_idx = unsafe { *dict_indices.get_unchecked(row_idx) };
                if dict_idx == 0 {
                    continue;
                } // NULL
                let key_str = unsafe { *dict_values.get_unchecked((dict_idx - 1) as usize) };
                let group_id = hash_agg.get_or_create_group_dict(dict_idx, key_str, row_idx);
                hash_agg.update_count(group_id);
            }
        } else if let Some(vals) = agg_col_int {
            for row_idx in start_row..end_row {
                let dict_idx = unsafe { *dict_indices.get_unchecked(row_idx) };
                if dict_idx == 0 {
                    continue;
                }
                let key_str = unsafe { *dict_values.get_unchecked((dict_idx - 1) as usize) };
                let group_id = hash_agg.get_or_create_group_dict(dict_idx, key_str, row_idx);
                let val = unsafe { *vals.get_unchecked(row_idx) };
                hash_agg.update_int(group_id, val);
            }
        } else if let Some(vals) = agg_col_float {
            for row_idx in start_row..end_row {
                let dict_idx = unsafe { *dict_indices.get_unchecked(row_idx) };
                if dict_idx == 0 {
                    continue;
                }
                let key_str = unsafe { *dict_values.get_unchecked((dict_idx - 1) as usize) };
                let group_id = hash_agg.get_or_create_group_dict(dict_idx, key_str, row_idx);
                let val = unsafe { *vals.get_unchecked(row_idx) };
                hash_agg.update_float(group_id, val);
            }
        }
        return;
    }

    // FAST PATH 2: Integer group column
    if let Some(group_vals) = group_col_int {
        if count_only {
            for row_idx in start_row..end_row {
                let key = unsafe { *group_vals.get_unchecked(row_idx) };
                let group_id = hash_agg.get_or_create_group_int(key, row_idx);
                hash_agg.update_count(group_id);
            }
        } else if let Some(vals) = agg_col_int {
            for row_idx in start_row..end_row {
                let key = unsafe { *group_vals.get_unchecked(row_idx) };
                let group_id = hash_agg.get_or_create_group_int(key, row_idx);
                let val = unsafe { *vals.get_unchecked(row_idx) };
                hash_agg.update_int(group_id, val);
            }
        } else if let Some(vals) = agg_col_float {
            for row_idx in start_row..end_row {
                let key = unsafe { *group_vals.get_unchecked(row_idx) };
                let group_id = hash_agg.get_or_create_group_int(key, row_idx);
                let val = unsafe { *vals.get_unchecked(row_idx) };
                hash_agg.update_float(group_id, val);
            }
        }
        return;
    }

    // FAST PATH 3: Regular string column
    if let Some(str_arr) = group_col_str {
        if count_only {
            for row_idx in start_row..end_row {
                if str_arr.is_null(row_idx) {
                    continue;
                }
                let key = str_arr.value(row_idx);
                let group_id = hash_agg.get_or_create_group_str(key, row_idx);
                hash_agg.update_count(group_id);
            }
        } else if let Some(vals) = agg_col_int {
            for row_idx in start_row..end_row {
                if str_arr.is_null(row_idx) {
                    continue;
                }
                let key = str_arr.value(row_idx);
                let group_id = hash_agg.get_or_create_group_str(key, row_idx);
                let val = unsafe { *vals.get_unchecked(row_idx) };
                hash_agg.update_int(group_id, val);
            }
        } else if let Some(vals) = agg_col_float {
            for row_idx in start_row..end_row {
                if str_arr.is_null(row_idx) {
                    continue;
                }
                let key = str_arr.value(row_idx);
                let group_id = hash_agg.get_or_create_group_str(key, row_idx);
                let val = unsafe { *vals.get_unchecked(row_idx) };
                hash_agg.update_float(group_id, val);
            }
        }
    }
}

/// Fast counting aggregation for dictionary-encoded string GROUP BY
/// Uses dictionary index as direct array index (O(1) lookup)
pub struct DictCountAgg {
    /// Count per dictionary index
    counts: Vec<i64>,
    /// Sum values per dictionary index (for int aggregates)
    sums_int: Vec<i64>,
    /// Sum values per dictionary index (for float aggregates)
    sums_float: Vec<f64>,
    /// Dictionary size
    dict_size: usize,
}

impl DictCountAgg {
    pub fn new(dict_size: usize) -> Self {
        Self {
            counts: vec![0; dict_size + 1], // +1 for null handling
            sums_int: vec![0; dict_size + 1],
            sums_float: vec![0.0; dict_size + 1],
            dict_size,
        }
    }

    #[inline(always)]
    pub fn update_count(&mut self, dict_idx: u32) {
        let idx = dict_idx as usize;
        if idx <= self.dict_size {
            unsafe {
                *self.counts.get_unchecked_mut(idx) += 1;
            }
        }
    }

    #[inline(always)]
    pub fn update_int(&mut self, dict_idx: u32, val: i64) {
        let idx = dict_idx as usize;
        if idx <= self.dict_size {
            unsafe {
                *self.counts.get_unchecked_mut(idx) += 1;
                *self.sums_int.get_unchecked_mut(idx) += val;
            }
        }
    }

    #[inline(always)]
    pub fn update_float(&mut self, dict_idx: u32, val: f64) {
        let idx = dict_idx as usize;
        if idx <= self.dict_size {
            unsafe {
                *self.counts.get_unchecked_mut(idx) += 1;
                *self.sums_float.get_unchecked_mut(idx) += val;
            }
        }
    }

    /// Get results as (indices with data, counts, sums_int, sums_float)
    pub fn to_results(&self) -> (Vec<u32>, Vec<i64>, Vec<i64>, Vec<f64>) {
        let mut indices = Vec::new();
        let mut counts = Vec::new();
        let mut sums_int = Vec::new();
        let mut sums_float = Vec::new();

        for i in 1..=self.dict_size {
            // Skip 0 (NULL)
            if self.counts[i] > 0 {
                indices.push(i as u32);
                counts.push(self.counts[i]);
                sums_int.push(self.sums_int[i]);
                sums_float.push(self.sums_float[i]);
            }
        }

        (indices, counts, sums_int, sums_float)
    }
}

/// Fast GROUP BY for low-cardinality integer keys using direct counting
/// Returns None if not applicable (key range too large or not integer column)
pub fn try_direct_count_group_by(
    batch: &RecordBatch,
    group_col_name: &str,
    agg_col_name: Option<&str>,
) -> Option<(Vec<i64>, Vec<i64>, Vec<i64>, Vec<f64>)> {
    // Get group column - must be Int64
    let group_col = batch.column_by_name(group_col_name)?;
    let int_arr = group_col.as_any().downcast_ref::<Int64Array>()?;

    if int_arr.is_empty() {
        return Some((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }

    // Compute min/max to check if direct counting is beneficial
    let values = int_arr.values();
    let mut min_val = i64::MAX;
    let mut max_val = i64::MIN;

    // OPTIMIZATION: Use pointer-based min/max scan
    let ptr = values.as_ptr();
    let len = values.len();
    for i in 0..len {
        let v = unsafe { *ptr.add(i) };
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    // Check if direct counting is beneficial
    if !DirectCountAgg::is_beneficial(min_val, max_val) {
        return None;
    }

    let mut agg = DirectCountAgg::new(min_val, max_val);

    // Get aggregate column if specified
    let agg_col = agg_col_name.and_then(|name| batch.column_by_name(name));
    let agg_col_int: Option<&[i64]> = agg_col.as_ref().and_then(|c| {
        c.as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| a.values().as_ref())
    });
    let agg_col_float: Option<&[f64]> = agg_col.as_ref().and_then(|c| {
        c.as_any()
            .downcast_ref::<Float64Array>()
            .map(|a| a.values().as_ref())
    });

    // Process all rows with direct counting
    if let Some(vals) = agg_col_int {
        for i in 0..len {
            let key = unsafe { *ptr.add(i) };
            let val = unsafe { *vals.as_ptr().add(i) };
            agg.update_int(key, val);
        }
    } else if let Some(vals) = agg_col_float {
        for i in 0..len {
            let key = unsafe { *ptr.add(i) };
            let val = unsafe { *vals.as_ptr().add(i) };
            agg.update_float(key, val);
        }
    } else {
        // COUNT only
        for i in 0..len {
            let key = unsafe { *ptr.add(i) };
            agg.update_count(key);
        }
    }

    Some(agg.to_results())
}

/// Execute vectorized GROUP BY on a RecordBatch
/// Processes data in VECTOR_SIZE batches for cache efficiency
pub fn execute_vectorized_group_by(
    batch: &RecordBatch,
    group_col_name: &str,
    agg_col_name: Option<&str>,
    _has_int_agg: bool,
) -> io::Result<VectorizedHashAgg> {
    let num_rows = batch.num_rows();
    let estimated_groups = (num_rows / 100).max(16).min(10000);

    // Get group column
    let group_col = batch
        .column_by_name(group_col_name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Group column not found"))?;

    // Get aggregate column if specified
    let agg_col = agg_col_name.and_then(|name| batch.column_by_name(name));

    // Extract aggregate column data first (needed for all paths)
    let agg_col_int: Option<&[i64]> = agg_col.and_then(|c| {
        c.as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| a.values().as_ref())
    });
    let agg_col_float: Option<&[f64]> = agg_col.and_then(|c| {
        c.as_any()
            .downcast_ref::<Float64Array>()
            .map(|a| a.values().as_ref())
    });
    let count_only = agg_col_int.is_none() && agg_col_float.is_none();

    // Determine group column type and extract data
    let group_col_int: Option<&[i64]>;
    let group_col_str: Option<&StringArray>;
    let mut group_col_dict_indices: Option<Vec<u32>> = None;
    let mut group_col_dict_values: Option<Vec<&str>> = None;
    let is_int_key: bool;

    // Try DictionaryArray first
    use arrow::array::DictionaryArray;
    use arrow::datatypes::UInt32Type;

    if let Some(dict_arr) = group_col
        .as_any()
        .downcast_ref::<DictionaryArray<UInt32Type>>()
    {
        let keys = dict_arr.keys();
        let values = dict_arr.values();
        if let Some(str_values) = values.as_any().downcast_ref::<StringArray>() {
            // Extract dictionary indices
            let indices: Vec<u32> = (0..num_rows)
                .map(|i| {
                    if keys.is_null(i) {
                        0u32
                    } else {
                        keys.value(i) + 1
                    }
                })
                .collect();
            let dict_vals: Vec<&str> = (0..str_values.len()).map(|i| str_values.value(i)).collect();
            group_col_dict_indices = Some(indices);
            group_col_dict_values = Some(dict_vals);
            group_col_int = None;
            group_col_str = None;
            is_int_key = false;
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Unsupported dictionary value type",
            ));
        }
    } else if let Some(int_arr) = group_col.as_any().downcast_ref::<Int64Array>() {
        group_col_int = Some(int_arr.values());
        group_col_str = None;
        is_int_key = true;
    } else if let Some(str_arr) = group_col.as_any().downcast_ref::<StringArray>() {
        group_col_int = None;
        group_col_str = Some(str_arr);
        is_int_key = false;
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Unsupported group column type",
        ));
    }

    // Create hash aggregation table
    let mut hash_agg = VectorizedHashAgg::new(is_int_key, estimated_groups);

    // Process in vectors (batches) for cache efficiency
    let dict_indices_ref = group_col_dict_indices.as_deref();
    let dict_values_ref: Option<Vec<&str>> = group_col_dict_values;
    let dict_values_slice: Option<&[&str]> = dict_values_ref.as_deref();

    for batch_start in (0..num_rows).step_by(VECTOR_SIZE) {
        let batch_end = (batch_start + VECTOR_SIZE).min(num_rows);

        process_vector_group_by(
            &mut hash_agg,
            group_col_int,
            group_col_str,
            dict_indices_ref,
            dict_values_slice,
            agg_col_int,
            agg_col_float,
            batch_start,
            batch_end,
            count_only,
        );
    }

    Ok(hash_agg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_hash() {
        let h1 = GroupHash::from_i64(42);
        let h2 = GroupHash::from_i64(42);
        let h3 = GroupHash::from_i64(43);

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_aggregate_state() {
        let mut state = AggregateState::new(0);
        state.update_int(10);
        state.update_int(20);
        state.update_int(5);

        assert_eq!(state.count, 3);
        assert_eq!(state.sum_int, 35);
        assert_eq!(state.min_int, Some(5));
        assert_eq!(state.max_int, Some(20));
    }
}
