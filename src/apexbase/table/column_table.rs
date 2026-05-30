//! Column type definitions for ApexBase storage
//!
//! This module provides core type definitions:
//! - BitVec: Efficient boolean/null bitmap storage
//! - TypedColumn: Type-specific column storage
//! - ColumnSchema: Column metadata with fast lookup

use crate::data::{DataType, Value};
use crate::table::arrow_column::ArrowStringColumn;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bit vector for efficient boolean storage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BitVec {
    data: Vec<u64>,
    len: usize,
}

impl BitVec {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            len: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity((capacity + 63) / 64),
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, value: bool) {
        let word_idx = self.len / 64;
        let bit_idx = self.len % 64;

        if word_idx >= self.data.len() {
            self.data.push(0);
        }

        if value {
            self.data[word_idx] |= 1u64 << bit_idx;
        }
        self.len += 1;
    }

    #[inline]
    pub fn get(&self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        (self.data[word_idx] >> bit_idx) & 1 == 1
    }

    #[inline]
    pub fn set(&mut self, index: usize, value: bool) {
        let word_idx = index / 64;
        let bit_idx = index % 64;

        // Extend if needed
        while word_idx >= self.data.len() {
            self.data.push(0);
        }
        if index >= self.len {
            self.len = index + 1;
        }

        if value {
            self.data[word_idx] |= 1u64 << bit_idx;
        } else {
            self.data[word_idx] &= !(1u64 << bit_idx);
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Count set bits (for counting non-deleted rows)
    pub fn count_ones(&self) -> usize {
        self.data.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Check if all bits are false (no nulls) - O(words) not O(bits)
    #[inline]
    pub fn all_false(&self) -> bool {
        self.data.iter().all(|&w| w == 0)
    }

    /// Get raw u64 data for direct Arrow buffer conversion
    #[inline]
    pub fn raw_data(&self) -> &[u64] {
        &self.data
    }

    /// Extend with n false values - fast batch operation
    #[inline]
    pub fn extend_false(&mut self, count: usize) {
        // Calculate new length
        let new_len = self.len + count;
        let required_words = (new_len + 63) / 64;

        // Extend data vec if needed (new words are already 0)
        self.data.resize(required_words, 0);
        self.len = new_len;
    }

    /// Slice BitVec for delta extraction [start, end)
    pub fn slice(&self, start: usize, end: usize) -> Self {
        let mut result = Self::with_capacity(end - start);
        for i in start..end {
            result.push(self.get(i));
        }
        result
    }

    /// Extend from another BitVec
    pub fn extend(&mut self, other: &BitVec) {
        for i in 0..other.len() {
            self.push(other.get(i));
        }
    }

    /// Extend with n true values - fast batch operation
    #[inline]
    pub fn extend_true(&mut self, count: usize) {
        let start_idx = self.len;
        let new_len = self.len + count;
        let required_words = (new_len + 63) / 64;

        // Ensure capacity
        self.data.resize(required_words, 0);

        // Set bits from start_idx to new_len
        for i in start_idx..new_len {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            self.data[word_idx] |= 1u64 << bit_idx;
        }
        self.len = new_len;
    }

    /// Batch extend from a boolean slice - optimized for SIMD
    #[inline]
    pub fn extend_from_bools(&mut self, values: &[bool]) {
        let count = values.len();
        if count == 0 {
            return;
        }

        let new_len = self.len + count;
        let required_words = (new_len + 63) / 64;

        // Ensure capacity
        self.data.resize(required_words, 0);

        // Process in chunks of 64 for better performance
        let mut idx = 0;
        let base_bit = self.len;

        // Fast path: process full words at once
        while idx + 64 <= count {
            let mut word = 0u64;
            for bit in 0..64 {
                if values[idx + bit] {
                    word |= 1u64 << bit;
                }
            }
            let word_idx = (base_bit + idx) / 64;
            let bit_offset = (base_bit + idx) % 64;

            if bit_offset == 0 {
                self.data[word_idx] = word;
            } else {
                // Handle cross-word boundary
                self.data[word_idx] |= word << bit_offset;
                if word_idx + 1 < self.data.len() {
                    self.data[word_idx + 1] |= word >> (64 - bit_offset);
                }
            }
            idx += 64;
        }

        // Handle remaining bits
        while idx < count {
            let bit_idx = base_bit + idx;
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            if values[idx] {
                self.data[word_idx] |= 1u64 << bit_pos;
            }
            idx += 1;
        }

        self.len = new_len;
    }
}

/// Type-specific column storage for maximum performance
///
/// String columns use Arrow-native contiguous buffer storage for 50-70% memory savings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypedColumn {
    Int64 {
        data: Vec<i64>,
        nulls: BitVec,
    },
    Float64 {
        data: Vec<f64>,
        nulls: BitVec,
    },
    /// Arrow-native string storage with contiguous buffers
    /// Memory layout: offsets (4 bytes/row) + data buffer (no per-string allocation)
    String(ArrowStringColumn),
    Bool {
        data: BitVec,
        nulls: BitVec,
    },
    /// For mixed/unknown types, fall back to Value
    Mixed {
        data: Vec<Value>,
        nulls: BitVec,
    },
}

impl TypedColumn {
    pub fn new(dtype: DataType) -> Self {
        match dtype {
            DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                TypedColumn::Int64 {
                    data: Vec::new(),
                    nulls: BitVec::new(),
                }
            }
            DataType::Float64 | DataType::Float32 => TypedColumn::Float64 {
                data: Vec::new(),
                nulls: BitVec::new(),
            },
            DataType::String => TypedColumn::String(ArrowStringColumn::new()),
            DataType::Bool => TypedColumn::Bool {
                data: BitVec::new(),
                nulls: BitVec::new(),
            },
            _ => TypedColumn::Mixed {
                data: Vec::new(),
                nulls: BitVec::new(),
            },
        }
    }

    pub fn with_capacity(dtype: DataType, capacity: usize) -> Self {
        match dtype {
            DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                TypedColumn::Int64 {
                    data: Vec::with_capacity(capacity),
                    nulls: BitVec::with_capacity(capacity),
                }
            }
            DataType::Float64 | DataType::Float32 => TypedColumn::Float64 {
                data: Vec::with_capacity(capacity),
                nulls: BitVec::with_capacity(capacity),
            },
            DataType::String => TypedColumn::String(ArrowStringColumn::with_capacity(capacity, 32)),
            DataType::Bool => TypedColumn::Bool {
                data: BitVec::with_capacity(capacity),
                nulls: BitVec::with_capacity(capacity),
            },
            _ => TypedColumn::Mixed {
                data: Vec::with_capacity(capacity),
                nulls: BitVec::with_capacity(capacity),
            },
        }
    }

    /// Push a value to the column - O(1) amortized
    #[inline]
    pub fn push(&mut self, value: &Value) {
        match (self, value) {
            (TypedColumn::Int64 { data, nulls }, Value::Int64(v)) => {
                data.push(*v);
                nulls.push(false);
            }
            (TypedColumn::Int64 { data, nulls }, Value::Int32(v)) => {
                data.push(*v as i64);
                nulls.push(false);
            }
            (TypedColumn::Int64 { data, nulls }, Value::Null) => {
                data.push(0);
                nulls.push(true);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Float64(v)) => {
                data.push(*v);
                nulls.push(false);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Float32(v)) => {
                data.push(*v as f64);
                nulls.push(false);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Int64(v)) => {
                data.push(*v as f64);
                nulls.push(false);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Null) => {
                data.push(0.0);
                nulls.push(true);
            }
            (TypedColumn::String(col), Value::String(v)) => {
                col.push(v);
            }
            (TypedColumn::String(col), Value::Null) => {
                col.push_null();
            }
            (TypedColumn::Bool { data, nulls }, Value::Bool(v)) => {
                data.push(*v);
                nulls.push(false);
            }
            (TypedColumn::Bool { data, nulls }, Value::Null) => {
                data.push(false);
                nulls.push(true);
            }
            (TypedColumn::Mixed { data, nulls }, v) => {
                nulls.push(v.is_null());
                data.push(v.clone());
            }
            // Type mismatch - convert to mixed or store as null
            (col, _value) => {
                col.push_null();
                // Log warning in debug mode
                #[cfg(debug_assertions)]
                eprintln!("Warning: Type mismatch when pushing {:?}", _value);
            }
        }
    }

    /// Push a null value
    #[inline]
    pub fn push_null(&mut self) {
        match self {
            TypedColumn::Int64 { data, nulls } => {
                data.push(0);
                nulls.push(true);
            }
            TypedColumn::Float64 { data, nulls } => {
                data.push(0.0);
                nulls.push(true);
            }
            TypedColumn::String(col) => {
                col.push_null();
            }
            TypedColumn::Bool { data, nulls } => {
                data.push(false);
                nulls.push(true);
            }
            TypedColumn::Mixed { data, nulls } => {
                data.push(Value::Null);
                nulls.push(true);
            }
        }
    }

    /// Push `count` NULLs efficiently (batch)
    #[inline]
    pub fn push_null_n(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        match self {
            TypedColumn::Int64 { data, nulls } => {
                let new_len = data.len() + count;
                data.resize(new_len, 0);
                nulls.extend_true(count);
            }
            TypedColumn::Float64 { data, nulls } => {
                let new_len = data.len() + count;
                data.resize(new_len, 0.0);
                nulls.extend_true(count);
            }
            TypedColumn::String(col) => {
                col.push_null_n(count);
            }
            TypedColumn::Bool { data, nulls } => {
                data.extend_false(count);
                nulls.extend_true(count);
            }
            TypedColumn::Mixed { data, nulls } => {
                let new_len = data.len() + count;
                data.resize(new_len, Value::Null);
                nulls.extend_true(count);
            }
        }
    }

    #[inline]
    pub fn set_null_fast(&mut self, index: usize) {
        match self {
            TypedColumn::Int64 { nulls, .. } => {
                nulls.set(index, true);
            }
            TypedColumn::Float64 { nulls, .. } => {
                nulls.set(index, true);
            }
            TypedColumn::String(col) => {
                col.set_null(index);
            }
            TypedColumn::Bool { nulls, .. } => {
                nulls.set(index, true);
            }
            TypedColumn::Mixed { data, nulls } => {
                if index < data.len() {
                    data[index] = Value::Null;
                }
                nulls.set(index, true);
            }
        }
    }

    /// Get value at index
    #[inline]
    pub fn get(&self, index: usize) -> Option<Value> {
        match self {
            TypedColumn::Int64 { data, nulls } => {
                if index >= data.len() || nulls.get(index) {
                    None
                } else {
                    Some(Value::Int64(data[index]))
                }
            }
            TypedColumn::Float64 { data, nulls } => {
                if index >= data.len() || nulls.get(index) {
                    None
                } else {
                    Some(Value::Float64(data[index]))
                }
            }
            TypedColumn::String(col) => Some(col.get_value(index)),
            TypedColumn::Bool { data, nulls } => {
                if index >= data.len() || nulls.get(index) {
                    None
                } else {
                    Some(Value::Bool(data.get(index)))
                }
            }
            TypedColumn::Mixed { data, nulls } => {
                if index >= data.len() || nulls.get(index) {
                    None
                } else {
                    Some(data[index].clone())
                }
            }
        }
    }

    #[inline]
    pub fn get_str(&self, index: usize) -> Option<&str> {
        match self {
            TypedColumn::String(col) => col.get(index),
            _ => None,
        }
    }

    /// Set value at index
    #[inline]
    pub fn set(&mut self, index: usize, value: &Value) {
        match (self, value) {
            (TypedColumn::Int64 { data, nulls }, Value::Int64(v)) => {
                if index < data.len() {
                    data[index] = *v;
                    nulls.set(index, false);
                }
            }
            (TypedColumn::Int64 { data, nulls }, Value::Null) => {
                if index < data.len() {
                    nulls.set(index, true);
                }
            }
            (TypedColumn::Float64 { data, nulls }, Value::Float64(v)) => {
                if index < data.len() {
                    data[index] = *v;
                    nulls.set(index, false);
                }
            }
            (TypedColumn::Float64 { data, nulls }, Value::Null) => {
                if index < data.len() {
                    nulls.set(index, true);
                }
            }
            (TypedColumn::String(col), Value::String(v)) => {
                col.set(index, v);
            }
            (TypedColumn::String(col), Value::Null) => {
                col.set_null(index);
            }
            (TypedColumn::Bool { data, nulls }, Value::Bool(v)) => {
                if index < data.len() {
                    data.set(index, *v);
                    nulls.set(index, false);
                }
            }
            (TypedColumn::Bool { data, nulls }, Value::Null) => {
                if index < data.len() {
                    nulls.set(index, true);
                }
            }
            (TypedColumn::Mixed { data, nulls }, v) => {
                if index < data.len() {
                    data[index] = v.clone();
                    nulls.set(index, v.is_null());
                }
            }
            _ => {}
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TypedColumn::Int64 { data, .. } => data.len(),
            TypedColumn::Float64 { data, .. } => data.len(),
            TypedColumn::String(col) => col.len(),
            TypedColumn::Bool { data, .. } => data.len(),
            TypedColumn::Mixed { data, .. } => data.len(),
        }
    }

    pub fn is_null(&self, index: usize) -> bool {
        match self {
            TypedColumn::Int64 { nulls, .. } => nulls.get(index),
            TypedColumn::Float64 { nulls, .. } => nulls.get(index),
            TypedColumn::String(col) => col.is_null(index),
            TypedColumn::Bool { nulls, .. } => nulls.get(index),
            TypedColumn::Mixed { nulls, .. } => nulls.get(index),
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            TypedColumn::Int64 { .. } => DataType::Int64,
            TypedColumn::Float64 { .. } => DataType::Float64,
            TypedColumn::String(_) => DataType::String,
            TypedColumn::Bool { .. } => DataType::Bool,
            TypedColumn::Mixed { .. } => DataType::Json, // fallback
        }
    }

    /// Slice column for delta extraction [start, end)
    pub fn slice(&self, start: usize, end: usize) -> Self {
        match self {
            TypedColumn::Int64 { data, nulls } => TypedColumn::Int64 {
                data: data[start..end].to_vec(),
                nulls: nulls.slice(start, end),
            },
            TypedColumn::Float64 { data, nulls } => TypedColumn::Float64 {
                data: data[start..end].to_vec(),
                nulls: nulls.slice(start, end),
            },
            TypedColumn::String(col) => TypedColumn::String(col.slice(start, end)),
            TypedColumn::Bool { data, nulls } => TypedColumn::Bool {
                data: data.slice(start, end),
                nulls: nulls.slice(start, end),
            },
            TypedColumn::Mixed { data, nulls } => TypedColumn::Mixed {
                data: data[start..end].to_vec(),
                nulls: nulls.slice(start, end),
            },
        }
    }

    /// Append another TypedColumn to this one (for delta loading)
    pub fn append(&mut self, other: Self) {
        match (self, other) {
            (
                TypedColumn::Int64 { data, nulls },
                TypedColumn::Int64 {
                    data: other_data,
                    nulls: other_nulls,
                },
            ) => {
                data.extend(other_data);
                nulls.extend(&other_nulls);
            }
            (
                TypedColumn::Float64 { data, nulls },
                TypedColumn::Float64 {
                    data: other_data,
                    nulls: other_nulls,
                },
            ) => {
                data.extend(other_data);
                nulls.extend(&other_nulls);
            }
            (TypedColumn::String(col), TypedColumn::String(other_col)) => {
                col.append(&other_col);
            }
            (
                TypedColumn::Bool { data, nulls },
                TypedColumn::Bool {
                    data: other_data,
                    nulls: other_nulls,
                },
            ) => {
                data.extend(&other_data);
                nulls.extend(&other_nulls);
            }
            (
                TypedColumn::Mixed { data, nulls },
                TypedColumn::Mixed {
                    data: other_data,
                    nulls: other_nulls,
                },
            ) => {
                data.extend(other_data);
                nulls.extend(&other_nulls);
            }
            _ => {} // Type mismatch - ignore
        }
    }
}

/// Column schema with fast lookup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnSchema {
    /// Column definitions: (name, data_type)
    pub columns: Vec<(String, DataType)>,
    /// Name to index mapping for O(1) lookup
    pub name_to_index: HashMap<String, usize>,
}

impl ColumnSchema {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            name_to_index: HashMap::new(),
        }
    }

    pub fn add_column(&mut self, name: &str, dtype: DataType) -> usize {
        if let Some(&idx) = self.name_to_index.get(name) {
            return idx;
        }
        let idx = self.columns.len();
        self.columns.push((name.to_string(), dtype));
        self.name_to_index.insert(name.to_string(), idx);
        idx
    }

    pub fn get_index(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }

    pub fn get_type(&self, index: usize) -> Option<DataType> {
        self.columns.get(index).map(|(_, t)| *t)
    }

    pub fn column_names(&self) -> Vec<String> {
        self.columns.iter().map(|(n, _)| n.clone()).collect()
    }

    pub fn len(&self) -> usize {
        self.columns.len()
    }
}

impl Default for ColumnSchema {
    fn default() -> Self {
        Self::new()
    }
}
