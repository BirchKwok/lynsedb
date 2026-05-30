//! Multi-Column Vectorized Execution Engine for GROUP BY
//!
//! This module extends the vectorized execution to support multi-column GROUP BY
//! with efficient hash aggregation and parallel processing.

use ahash::{AHashMap, AHashSet, AHasher};
use arrow::array::{
    Array, ArrayRef, BooleanArray, DictionaryArray, Float64Array, Int64Array, StringArray,
};
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema, UInt32Type};
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;
use std::hash::{Hash, Hasher};
use std::io;
use std::sync::Arc;

use crate::query::vectorized::{AggregateState, GroupHash, VECTOR_SIZE};

/// Typed column reference for multi-column grouping
#[derive(Clone, Copy)]
pub enum TypedColumnRef<'a> {
    Int64(&'a Int64Array),
    Float64(&'a Float64Array),
    String(&'a StringArray),
    Bool(&'a BooleanArray),
    DictString(&'a DictionaryArray<UInt32Type>, &'a StringArray),
}

impl<'a> TypedColumnRef<'a> {
    /// Compute hash for a row in this column
    #[inline(always)]
    pub fn hash_row(&self, row_idx: usize, hasher: &mut AHasher) {
        match self {
            TypedColumnRef::Int64(arr) => {
                if !arr.is_null(row_idx) {
                    hasher.write_i64(arr.value(row_idx));
                } else {
                    hasher.write_u8(0xFF);
                }
            }
            TypedColumnRef::Float64(arr) => {
                if !arr.is_null(row_idx) {
                    hasher.write_u64(arr.value(row_idx).to_bits());
                } else {
                    hasher.write_u8(0xFF);
                }
            }
            TypedColumnRef::String(arr) => {
                if !arr.is_null(row_idx) {
                    hasher.write(arr.value(row_idx).as_bytes());
                } else {
                    hasher.write_u8(0xFF);
                }
            }
            TypedColumnRef::Bool(arr) => {
                if !arr.is_null(row_idx) {
                    hasher.write_u8(arr.value(row_idx) as u8);
                } else {
                    hasher.write_u8(0xFF);
                }
            }
            TypedColumnRef::DictString(dict_arr, values_arr) => {
                let keys = dict_arr.keys();
                if !keys.is_null(row_idx) {
                    let idx = keys.value(row_idx) as usize;
                    if idx < values_arr.len() {
                        hasher.write(values_arr.value(idx).as_bytes());
                    }
                } else {
                    hasher.write_u8(0xFF);
                }
            }
        }
    }

    /// Check if value at row is null
    #[inline(always)]
    pub fn is_null(&self, row_idx: usize) -> bool {
        match self {
            TypedColumnRef::Int64(arr) => arr.is_null(row_idx),
            TypedColumnRef::Float64(arr) => arr.is_null(row_idx),
            TypedColumnRef::String(arr) => arr.is_null(row_idx),
            TypedColumnRef::Bool(arr) => arr.is_null(row_idx),
            TypedColumnRef::DictString(dict_arr, _) => dict_arr.keys().is_null(row_idx),
        }
    }
}

/// Multi-column group key using pre-computed hash
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultiColumnGroupKey(u64);

impl MultiColumnGroupKey {
    #[inline(always)]
    pub fn from_columns(cols: &[TypedColumnRef<'_>], row_idx: usize) -> Self {
        let mut hasher = AHasher::default();
        for col in cols {
            col.hash_row(row_idx, &mut hasher);
        }
        MultiColumnGroupKey(hasher.finish())
    }
}

/// Multi-column hash aggregation table
pub struct MultiColumnHashAgg {
    hash_table: AHashMap<MultiColumnGroupKey, u32>,
    states: Vec<AggregateState>,
    first_row_indices: Vec<usize>,
}

impl MultiColumnHashAgg {
    pub fn new(estimated_groups: usize) -> Self {
        let capacity = (estimated_groups * 2).max(64);
        Self {
            hash_table: AHashMap::with_capacity_and_hasher(capacity, Default::default()),
            states: Vec::with_capacity(estimated_groups),
            first_row_indices: Vec::with_capacity(estimated_groups),
        }
    }

    #[inline(always)]
    pub fn get_or_create_group(&mut self, key: MultiColumnGroupKey, row_idx: usize) -> u32 {
        if let Some(&group_id) = self.hash_table.get(&key) {
            group_id
        } else {
            let group_id = self.states.len() as u32;
            self.hash_table.insert(key, group_id);
            self.states.push(AggregateState::new(row_idx));
            self.first_row_indices.push(row_idx);
            group_id
        }
    }

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

    pub fn num_groups(&self) -> usize {
        self.states.len()
    }

    pub fn states(&self) -> &[AggregateState] {
        &self.states
    }

    pub fn first_row_indices(&self) -> &[usize] {
        &self.first_row_indices
    }
}

/// Process a batch of rows for multi-column GROUP BY
pub fn process_multi_column_group_by(
    hash_agg: &mut MultiColumnHashAgg,
    group_cols: &[TypedColumnRef<'_>],
    agg_col_int: Option<&Int64Array>,
    agg_col_float: Option<&Float64Array>,
    start_row: usize,
    end_row: usize,
    count_only: bool,
) {
    if count_only {
        for row_idx in start_row..end_row {
            let key = MultiColumnGroupKey::from_columns(group_cols, row_idx);
            let group_id = hash_agg.get_or_create_group(key, row_idx);
            hash_agg.update_count(group_id);
        }
    } else if let Some(int_arr) = agg_col_int {
        for row_idx in start_row..end_row {
            let key = MultiColumnGroupKey::from_columns(group_cols, row_idx);
            let group_id = hash_agg.get_or_create_group(key, row_idx);
            if !int_arr.is_null(row_idx) {
                hash_agg.update_int(group_id, int_arr.value(row_idx));
            }
        }
    } else if let Some(float_arr) = agg_col_float {
        for row_idx in start_row..end_row {
            let key = MultiColumnGroupKey::from_columns(group_cols, row_idx);
            let group_id = hash_agg.get_or_create_group(key, row_idx);
            if !float_arr.is_null(row_idx) {
                hash_agg.update_float(group_id, float_arr.value(row_idx));
            }
        }
    }
}

/// Execute multi-column GROUP BY with parallel partition aggregation
pub fn execute_multi_column_group_by(
    batch: &RecordBatch,
    group_col_names: &[String],
    agg_col_name: Option<&str>,
) -> io::Result<MultiColumnHashAgg> {
    let num_rows = batch.num_rows();
    let estimated_groups = (num_rows / 10).max(16).min(10000);

    // Extract and type group columns
    let mut typed_group_cols: Vec<TypedColumnRef<'_>> = Vec::with_capacity(group_col_names.len());
    for col_name in group_col_names {
        let col = batch.column_by_name(col_name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Group column '{}' not found", col_name),
            )
        })?;

        let typed_col = if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
            TypedColumnRef::Int64(arr)
        } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
            TypedColumnRef::Float64(arr)
        } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
            TypedColumnRef::String(arr)
        } else if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
            TypedColumnRef::Bool(arr)
        } else if let Some(dict_arr) = col.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
            let values = dict_arr.values();
            if let Some(str_arr) = values.as_any().downcast_ref::<StringArray>() {
                TypedColumnRef::DictString(dict_arr, str_arr)
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Unsupported dictionary value type",
                ));
            }
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Unsupported group column type for column '{}'", col_name),
            ));
        };
        typed_group_cols.push(typed_col);
    }

    // Get aggregate column
    let agg_col_int: Option<&Int64Array> = agg_col_name.and_then(|name| {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
    });
    let agg_col_float: Option<&Float64Array> = agg_col_name.and_then(|name| {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
    });
    let count_only = agg_col_int.is_none() && agg_col_float.is_none();

    // Use parallel partitioning for large datasets
    if num_rows > 50_000 {
        let num_partitions = rayon::current_num_threads().max(4);
        let partition_size = (num_rows + num_partitions - 1) / num_partitions;

        // Process partitions in parallel
        let partition_results: Vec<MultiColumnHashAgg> = (0..num_partitions)
            .into_par_iter()
            .map(|p| {
                let start = p * partition_size;
                let end = ((p + 1) * partition_size).min(num_rows);
                let mut local_agg = MultiColumnHashAgg::new(estimated_groups / num_partitions + 1);

                for batch_start in (start..end).step_by(VECTOR_SIZE) {
                    let batch_end = (batch_start + VECTOR_SIZE).min(end);
                    process_multi_column_group_by(
                        &mut local_agg,
                        &typed_group_cols,
                        agg_col_int,
                        agg_col_float,
                        batch_start,
                        batch_end,
                        count_only,
                    );
                }
                local_agg
            })
            .collect();

        // Merge partition results
        let mut final_agg = MultiColumnHashAgg::new(estimated_groups);
        for local_agg in partition_results {
            for (key, local_group_id) in local_agg.hash_table {
                let local_state = &local_agg.states[local_group_id as usize];

                if let Some(&global_group_id) = final_agg.hash_table.get(&key) {
                    // Merge into existing group
                    final_agg.states[global_group_id as usize].merge(local_state);
                } else {
                    // Create new group
                    let new_group_id = final_agg.states.len() as u32;
                    final_agg.hash_table.insert(key, new_group_id);
                    final_agg.states.push(local_state.clone());
                    final_agg
                        .first_row_indices
                        .push(local_agg.first_row_indices[local_group_id as usize]);
                }
            }
        }

        Ok(final_agg)
    } else {
        // Sequential processing for small datasets
        let mut hash_agg = MultiColumnHashAgg::new(estimated_groups);

        for batch_start in (0..num_rows).step_by(VECTOR_SIZE) {
            let batch_end = (batch_start + VECTOR_SIZE).min(num_rows);
            process_multi_column_group_by(
                &mut hash_agg,
                &typed_group_cols,
                agg_col_int,
                agg_col_float,
                batch_start,
                batch_end,
                count_only,
            );
        }

        Ok(hash_agg)
    }
}

/// Build result RecordBatch from multi-column aggregation
pub fn build_multi_column_result(
    hash_agg: &MultiColumnHashAgg,
    batch: &RecordBatch,
    group_col_names: &[String],
    agg_func: Option<crate::query::AggregateFunc>,
    agg_col_name: Option<&str>,
) -> io::Result<RecordBatch> {
    use crate::query::AggregateFunc;

    let num_groups = hash_agg.num_groups();
    let first_indices = hash_agg.first_row_indices();
    let states = hash_agg.states();

    let mut result_fields: Vec<Field> = Vec::new();
    let mut result_arrays: Vec<ArrayRef> = Vec::new();

    // Add group columns (take first value from each group)
    for col_name in group_col_names {
        if let Some(src_col) = batch.column_by_name(col_name) {
            let indices_arr = arrow::array::UInt32Array::from(
                first_indices.iter().map(|&i| i as u32).collect::<Vec<_>>(),
            );
            let taken = arrow::compute::take(src_col.as_ref(), &indices_arr, None)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            result_fields.push(Field::new(col_name, taken.data_type().clone(), true));
            result_arrays.push(taken);
        }
    }

    // Add aggregate column
    if let Some(func) = agg_func {
        let func_name = match func {
            AggregateFunc::Count => "COUNT",
            AggregateFunc::Sum => "SUM",
            AggregateFunc::Avg => "AVG",
            AggregateFunc::Min => "MIN",
            AggregateFunc::Max => "MAX",
        };

        let output_name = agg_col_name
            .map(|c| format!("{}({})", func_name, c))
            .unwrap_or_else(|| format!("{}(*)", func_name));

        match func {
            AggregateFunc::Count => {
                let counts: Vec<i64> = states.iter().map(|s| s.count).collect();
                result_fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                result_arrays.push(Arc::new(Int64Array::from(counts)));
            }
            AggregateFunc::Sum => {
                // Determine if we should use int or float sum based on aggregate column type
                let is_int_agg = agg_col_name
                    .and_then(|name| {
                        batch
                            .column_by_name(name)
                            .map(|c| c.as_any().downcast_ref::<Int64Array>().is_some())
                    })
                    .unwrap_or(false);

                if is_int_agg {
                    let sums: Vec<i64> = states.iter().map(|s| s.sum_int).collect();
                    result_fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                    result_arrays.push(Arc::new(Int64Array::from(sums)));
                } else {
                    let sums: Vec<f64> = states.iter().map(|s| s.sum_float).collect();
                    result_fields.push(Field::new(&output_name, ArrowDataType::Float64, false));
                    result_arrays.push(Arc::new(Float64Array::from(sums)));
                }
            }
            AggregateFunc::Avg => {
                let avgs: Vec<Option<f64>> = states
                    .iter()
                    .map(|s| {
                        if s.count > 0 {
                            Some(s.sum_float / s.count as f64)
                        } else {
                            None
                        }
                    })
                    .collect();
                result_fields.push(Field::new(&output_name, ArrowDataType::Float64, true));
                result_arrays.push(Arc::new(Float64Array::from(avgs)));
            }
            AggregateFunc::Min => {
                let is_int_agg = agg_col_name
                    .and_then(|name| {
                        batch
                            .column_by_name(name)
                            .map(|c| c.as_any().downcast_ref::<Int64Array>().is_some())
                    })
                    .unwrap_or(false);

                if is_int_agg {
                    let mins: Vec<Option<i64>> = states.iter().map(|s| s.min_int).collect();
                    result_fields.push(Field::new(&output_name, ArrowDataType::Int64, true));
                    result_arrays.push(Arc::new(Int64Array::from(mins)));
                } else {
                    let mins: Vec<Option<f64>> = states.iter().map(|s| s.min_float).collect();
                    result_fields.push(Field::new(&output_name, ArrowDataType::Float64, true));
                    result_arrays.push(Arc::new(Float64Array::from(mins)));
                }
            }
            AggregateFunc::Max => {
                let is_int_agg = agg_col_name
                    .and_then(|name| {
                        batch
                            .column_by_name(name)
                            .map(|c| c.as_any().downcast_ref::<Int64Array>().is_some())
                    })
                    .unwrap_or(false);

                if is_int_agg {
                    let maxs: Vec<Option<i64>> = states.iter().map(|s| s.max_int).collect();
                    result_fields.push(Field::new(&output_name, ArrowDataType::Int64, true));
                    result_arrays.push(Arc::new(Int64Array::from(maxs)));
                } else {
                    let maxs: Vec<Option<f64>> = states.iter().map(|s| s.max_float).collect();
                    result_fields.push(Field::new(&output_name, ArrowDataType::Float64, true));
                    result_arrays.push(Arc::new(Float64Array::from(maxs)));
                }
            }
        }
    }

    let schema = Arc::new(Schema::new(result_fields));
    RecordBatch::try_new(schema, result_arrays)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
}
