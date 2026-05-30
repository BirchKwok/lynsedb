//! Delta Merger - Applies deltas to base columnar data during reads
//!
//! Merges DeltaStore (updates + deletes) with base columnar data to produce
//! a consistent view without modifying the base file.

use std::collections::HashMap;
use std::io;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Float64Array, Float64Builder, Int64Array,
    Int64Builder, StringArray, StringBuilder, UInt64Array, UInt64Builder,
};
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use super::update_log::{DeleteBitmap, DeltaStore};
use crate::data::Value;

// ============================================================================
// Delta Merger
// ============================================================================

/// Merges delta changes (updates + deletes) with base Arrow RecordBatch
///
/// Read path:
/// 1. Start with base RecordBatch from columnar file
/// 2. Filter out deleted rows (via DeleteBitmap)
/// 3. Overlay updated cells (latest value from DeltaStore)
/// 4. Return merged RecordBatch
pub struct DeltaMerger;

impl DeltaMerger {
    /// Apply deletes and updates to a base RecordBatch
    ///
    /// # Arguments
    /// * `base` - The base RecordBatch from columnar storage
    /// * `delta` - The DeltaStore with pending changes
    /// * `base_row_ids` - Row IDs corresponding to each row in the batch
    ///
    /// # Returns
    /// A new RecordBatch with deletes filtered out and updates applied
    pub fn merge(
        base: &RecordBatch,
        delta: &DeltaStore,
        base_row_ids: &[u64],
    ) -> io::Result<RecordBatch> {
        if delta.is_empty() {
            return Ok(base.clone());
        }

        let num_rows = base.num_rows();
        if num_rows == 0 || base_row_ids.len() != num_rows {
            return Ok(base.clone());
        }

        // Step 1: Build a boolean mask for non-deleted rows
        let keep_mask: Vec<bool> = base_row_ids
            .iter()
            .map(|id| !delta.is_deleted(*id))
            .collect();

        let kept_count = keep_mask.iter().filter(|&&k| k).count();
        if kept_count == 0 {
            // All rows deleted - return empty batch with same schema
            return Ok(RecordBatch::new_empty(base.schema()));
        }

        // Step 2: Build mapping from kept position → original position
        let kept_positions: Vec<usize> = keep_mask
            .iter()
            .enumerate()
            .filter(|(_, &k)| k)
            .map(|(i, _)| i)
            .collect();

        // Step 3: For each column, apply deletes and overlay updates
        let schema = base.schema();
        let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());

        for (col_idx, field) in schema.fields().iter().enumerate() {
            let col_name = field.name();
            let base_col = base.column(col_idx);

            let new_col = Self::merge_column(
                base_col,
                col_name,
                field.data_type(),
                &kept_positions,
                base_row_ids,
                delta,
            )?;

            new_columns.push(new_col);
        }

        RecordBatch::try_new(schema, new_columns)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Apply deletes only (no updates) - faster path
    pub fn apply_deletes(
        base: &RecordBatch,
        delete_bitmap: &DeleteBitmap,
        base_row_ids: &[u64],
    ) -> io::Result<RecordBatch> {
        if delete_bitmap.is_empty() {
            return Ok(base.clone());
        }

        let num_rows = base.num_rows();
        let keep_indices: Vec<u32> = (0..num_rows as u32)
            .filter(|&i| !delete_bitmap.is_deleted(base_row_ids[i as usize]))
            .collect();

        if keep_indices.len() == num_rows {
            return Ok(base.clone());
        }

        let indices = arrow::array::UInt32Array::from(keep_indices);
        let new_columns: Vec<ArrayRef> = base
            .columns()
            .iter()
            .map(|col| arrow::compute::take(col, &indices, None).unwrap())
            .collect();

        RecordBatch::try_new(base.schema(), new_columns)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    // ========================================================================
    // Internal: per-column merge
    // ========================================================================

    fn merge_column(
        base_col: &ArrayRef,
        col_name: &str,
        data_type: &ArrowDataType,
        kept_positions: &[usize],
        base_row_ids: &[u64],
        delta: &DeltaStore,
    ) -> io::Result<ArrayRef> {
        match data_type {
            ArrowDataType::Int64 => {
                Self::merge_int64(base_col, col_name, kept_positions, base_row_ids, delta)
            }
            ArrowDataType::UInt64 => {
                Self::merge_uint64(base_col, col_name, kept_positions, base_row_ids, delta)
            }
            ArrowDataType::Float64 => {
                Self::merge_float64(base_col, col_name, kept_positions, base_row_ids, delta)
            }
            ArrowDataType::Utf8 => {
                Self::merge_string(base_col, col_name, kept_positions, base_row_ids, delta)
            }
            ArrowDataType::Boolean => {
                Self::merge_bool(base_col, col_name, kept_positions, base_row_ids, delta)
            }
            _ => {
                // For unsupported types, just take the kept rows without updates
                let indices = arrow::array::UInt32Array::from(
                    kept_positions.iter().map(|&p| p as u32).collect::<Vec<_>>(),
                );
                arrow::compute::take(base_col, &indices, None)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
            }
        }
    }

    fn merge_int64(
        base_col: &ArrayRef,
        col_name: &str,
        kept_positions: &[usize],
        base_row_ids: &[u64],
        delta: &DeltaStore,
    ) -> io::Result<ArrayRef> {
        let arr = base_col.as_any().downcast_ref::<Int64Array>().unwrap();
        let mut builder = Int64Builder::with_capacity(kept_positions.len());
        for &pos in kept_positions {
            let row_id = base_row_ids[pos];
            if let Some(updated) = delta.get_updated_value(row_id, col_name) {
                match updated.as_i64() {
                    Some(v) => builder.append_value(v),
                    None => builder.append_null(),
                }
            } else if arr.is_null(pos) {
                builder.append_null();
            } else {
                builder.append_value(arr.value(pos));
            }
        }
        Ok(Arc::new(builder.finish()))
    }

    fn merge_uint64(
        base_col: &ArrayRef,
        col_name: &str,
        kept_positions: &[usize],
        base_row_ids: &[u64],
        delta: &DeltaStore,
    ) -> io::Result<ArrayRef> {
        let arr = base_col.as_any().downcast_ref::<UInt64Array>().unwrap();
        let mut builder = UInt64Builder::with_capacity(kept_positions.len());
        for &pos in kept_positions {
            let row_id = base_row_ids[pos];
            if let Some(updated) = delta.get_updated_value(row_id, col_name) {
                match updated {
                    Value::UInt64(v) => builder.append_value(*v),
                    Value::Int64(v) => builder.append_value(*v as u64),
                    _ => builder.append_null(),
                }
            } else if arr.is_null(pos) {
                builder.append_null();
            } else {
                builder.append_value(arr.value(pos));
            }
        }
        Ok(Arc::new(builder.finish()))
    }

    fn merge_float64(
        base_col: &ArrayRef,
        col_name: &str,
        kept_positions: &[usize],
        base_row_ids: &[u64],
        delta: &DeltaStore,
    ) -> io::Result<ArrayRef> {
        let arr = base_col.as_any().downcast_ref::<Float64Array>().unwrap();
        let mut builder = Float64Builder::with_capacity(kept_positions.len());
        for &pos in kept_positions {
            let row_id = base_row_ids[pos];
            if let Some(updated) = delta.get_updated_value(row_id, col_name) {
                match updated.as_f64() {
                    Some(v) => builder.append_value(v),
                    None => builder.append_null(),
                }
            } else if arr.is_null(pos) {
                builder.append_null();
            } else {
                builder.append_value(arr.value(pos));
            }
        }
        Ok(Arc::new(builder.finish()))
    }

    fn merge_string(
        base_col: &ArrayRef,
        col_name: &str,
        kept_positions: &[usize],
        base_row_ids: &[u64],
        delta: &DeltaStore,
    ) -> io::Result<ArrayRef> {
        let arr = base_col.as_any().downcast_ref::<StringArray>().unwrap();
        let mut builder =
            StringBuilder::with_capacity(kept_positions.len(), kept_positions.len() * 32);
        for &pos in kept_positions {
            let row_id = base_row_ids[pos];
            if let Some(updated) = delta.get_updated_value(row_id, col_name) {
                match updated.as_str() {
                    Some(s) => builder.append_value(s),
                    None => builder.append_null(),
                }
            } else if arr.is_null(pos) {
                builder.append_null();
            } else {
                builder.append_value(arr.value(pos));
            }
        }
        Ok(Arc::new(builder.finish()))
    }

    fn merge_bool(
        base_col: &ArrayRef,
        col_name: &str,
        kept_positions: &[usize],
        base_row_ids: &[u64],
        delta: &DeltaStore,
    ) -> io::Result<ArrayRef> {
        let arr = base_col.as_any().downcast_ref::<BooleanArray>().unwrap();
        let mut builder = BooleanBuilder::with_capacity(kept_positions.len());
        for &pos in kept_positions {
            let row_id = base_row_ids[pos];
            if let Some(updated) = delta.get_updated_value(row_id, col_name) {
                match updated.as_bool() {
                    Some(b) => builder.append_value(b),
                    None => builder.append_null(),
                }
            } else if arr.is_null(pos) {
                builder.append_null();
            } else {
                builder.append_value(arr.value(pos));
            }
        }
        Ok(Arc::new(builder.finish()))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Field;

    fn make_test_batch() -> (RecordBatch, Vec<u64>) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("_id", ArrowDataType::UInt64, false),
            Field::new("name", ArrowDataType::Utf8, true),
            Field::new("age", ArrowDataType::Int64, true),
        ]));

        let ids = Arc::new(UInt64Array::from(vec![0, 1, 2, 3, 4]));
        let names = Arc::new(StringArray::from(vec![
            "alice", "bob", "carol", "dave", "eve",
        ]));
        let ages = Arc::new(Int64Array::from(vec![25, 30, 35, 40, 45]));

        let batch = RecordBatch::try_new(schema, vec![ids, names, ages]).unwrap();
        let row_ids = vec![0, 1, 2, 3, 4];
        (batch, row_ids)
    }

    #[test]
    fn test_merge_with_deletes() {
        let (batch, row_ids) = make_test_batch();
        let dir = tempfile::tempdir().unwrap();
        let mut delta = DeltaStore::new(&dir.path().join("test.apex"));

        delta.delete_row(1); // delete bob
        delta.delete_row(3); // delete dave

        let merged = DeltaMerger::merge(&batch, &delta, &row_ids).unwrap();
        assert_eq!(merged.num_rows(), 3);

        let names = merged
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.value(0), "alice");
        assert_eq!(names.value(1), "carol");
        assert_eq!(names.value(2), "eve");
    }

    #[test]
    fn test_merge_with_updates() {
        let (batch, row_ids) = make_test_batch();
        let dir = tempfile::tempdir().unwrap();
        let mut delta = DeltaStore::new(&dir.path().join("test.apex"));

        delta.update_cell(0, "name", Value::String("alice_updated".into()));
        delta.update_cell(2, "age", Value::Int64(99));

        let merged = DeltaMerger::merge(&batch, &delta, &row_ids).unwrap();
        assert_eq!(merged.num_rows(), 5);

        let names = merged
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.value(0), "alice_updated");
        assert_eq!(names.value(1), "bob");

        let ages = merged
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ages.value(2), 99);
    }

    #[test]
    fn test_merge_deletes_and_updates() {
        let (batch, row_ids) = make_test_batch();
        let dir = tempfile::tempdir().unwrap();
        let mut delta = DeltaStore::new(&dir.path().join("test.apex"));

        delta.delete_row(1);
        delta.update_cell(4, "age", Value::Int64(100));

        let merged = DeltaMerger::merge(&batch, &delta, &row_ids).unwrap();
        assert_eq!(merged.num_rows(), 4); // bob removed

        let ages = merged
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        // eve is at position 3 (after bob removed: alice, carol, dave, eve)
        assert_eq!(ages.value(3), 100);
    }

    #[test]
    fn test_merge_empty_delta() {
        let (batch, row_ids) = make_test_batch();
        let dir = tempfile::tempdir().unwrap();
        let delta = DeltaStore::new(&dir.path().join("test.apex"));

        let merged = DeltaMerger::merge(&batch, &delta, &row_ids).unwrap();
        assert_eq!(merged.num_rows(), 5);
    }
}
