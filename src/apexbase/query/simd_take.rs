//! Fast take operations for ORDER BY

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array};
use arrow::buffer::Buffer;
use std::sync::Arc;

/// Fast take for Int64
pub fn fast_take_i64(arr: &Int64Array, indices: &[u32]) -> Int64Array {
    let values = arr.values();
    let n = indices.len();
    let mut result: Vec<i64> = Vec::with_capacity(n);

    if let Some(nulls) = arr.nulls() {
        let mut null_builder: Vec<bool> = Vec::with_capacity(n);

        const BLOCK: usize = 4096;
        let mut i = 0;
        while i < n {
            let end = (i + BLOCK).min(n);

            for j in i..end {
                let idx = indices[j] as usize;
                result.push(values[idx]);
                null_builder.push(!nulls.is_null(idx));
            }

            i = end;
        }

        Int64Array::new(
            Buffer::from_vec(result).into(),
            Some(arrow::buffer::NullBuffer::from(null_builder)),
        )
    } else {
        const BLOCK: usize = 4096;
        let mut i = 0;
        while i < n {
            let end = (i + BLOCK).min(n);
            for j in i..end {
                let idx = indices[j] as usize;
                result.push(values[idx]);
            }
            i = end;
        }

        Int64Array::from(result)
    }
}

/// Fast take for Float64
pub fn fast_take_f64(arr: &Float64Array, indices: &[u32]) -> Float64Array {
    let values = arr.values();
    let n = indices.len();
    let mut result: Vec<f64> = Vec::with_capacity(n);

    if let Some(nulls) = arr.nulls() {
        let mut null_builder: Vec<bool> = Vec::with_capacity(n);

        const BLOCK: usize = 4096;
        let mut i = 0;
        while i < n {
            let end = (i + BLOCK).min(n);
            for j in i..end {
                let idx = indices[j] as usize;
                result.push(values[idx]);
                null_builder.push(!nulls.is_null(idx));
            }
            i = end;
        }

        Float64Array::new(
            Buffer::from_vec(result).into(),
            Some(arrow::buffer::NullBuffer::from(null_builder)),
        )
    } else {
        const BLOCK: usize = 4096;
        let mut i = 0;
        while i < n {
            let end = (i + BLOCK).min(n);
            for j in i..end {
                let idx = indices[j] as usize;
                result.push(values[idx]);
            }
            i = end;
        }

        Float64Array::from(result)
    }
}

/// Optimized take dispatch
pub fn optimized_take(col: &ArrayRef, indices: &arrow::array::UInt32Array) -> ArrayRef {
    let indices_slice: Vec<u32> = indices.values().iter().copied().collect();

    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        return Arc::new(fast_take_i64(arr, &indices_slice));
    }

    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        return Arc::new(fast_take_f64(arr, &indices_slice));
    }

    match arrow::compute::take(col, indices, None) {
        Ok(result) => result,
        Err(_) => col.clone(),
    }
}
