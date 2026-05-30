//! Arrow-native column storage for memory-efficient string handling
//!
//! This module provides Arrow-native storage for TypedColumn that significantly
//! reduces memory usage for string columns by using contiguous buffers instead
//! of individual String heap allocations.
//!
//! Memory savings:
//! - Standard Vec<String>: 24 bytes per String (ptr+len+cap) + heap allocation + string data
//! - Arrow StringArray: 4 bytes offset per string + contiguous data buffer
//! - Expected savings: 50-70% for string-heavy workloads

use crate::data::{DataType, Value};
use crate::table::column_table::BitVec;
use arrow::array::{ArrayRef, BooleanBuilder, Float64Builder, GenericByteBuilder, Int64Builder};
use arrow::array::{DictionaryArray, Int32Array, StringArray};
use arrow::buffer::{Buffer, MutableBuffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::GenericStringType;
use arrow::datatypes::Int32Type;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

#[derive(Debug)]
enum CowBytes {
    Mutable(MutableBuffer),
    Frozen(Buffer),
}

impl Default for CowBytes {
    fn default() -> Self {
        Self::new_mutable()
    }
}

impl Clone for CowBytes {
    fn clone(&self) -> Self {
        match self {
            Self::Mutable(b) => Self::Mutable(MutableBuffer::from(b.as_slice().to_vec())),
            Self::Frozen(b) => Self::Frozen(b.clone()),
        }
    }
}

impl CowBytes {
    #[inline]
    fn new_mutable() -> Self {
        Self::Mutable(MutableBuffer::new(0))
    }

    #[inline]
    fn with_capacity(capacity: usize) -> Self {
        Self::Mutable(MutableBuffer::new(capacity))
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Mutable(b) => b.len(),
            Self::Frozen(b) => b.len(),
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        match self {
            Self::Mutable(b) => b.capacity(),
            Self::Frozen(b) => b.capacity(),
        }
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mutable(b) => b.as_slice(),
            Self::Frozen(b) => b.as_slice(),
        }
    }

    #[inline]
    fn push_bytes(&mut self, bytes: &[u8]) {
        match self {
            Self::Mutable(b) => b.extend_from_slice(bytes),
            Self::Frozen(buf) => {
                let owned = std::mem::take(buf);
                match owned.into_mutable() {
                    Ok(mut mb) => {
                        mb.extend_from_slice(bytes);
                        *self = Self::Mutable(mb);
                    }
                    Err(orig) => {
                        let mut mb = MutableBuffer::new(orig.len() + bytes.len());
                        mb.extend_from_slice(orig.as_slice());
                        mb.extend_from_slice(bytes);
                        *self = Self::Mutable(mb);
                    }
                }
            }
        }
    }

    #[inline]
    fn freeze(&mut self) {
        if let Self::Mutable(mb) = self {
            let owned = std::mem::take(mb);
            *self = Self::Frozen(Buffer::from(owned));
        }
    }

    #[inline]
    fn as_buffer_cloned(&self) -> Buffer {
        match self {
            Self::Mutable(mb) => Buffer::from(mb.as_slice()),
            Self::Frozen(b) => b.clone(),
        }
    }

    #[inline]
    fn reserve(&mut self, additional: usize) {
        match self {
            Self::Mutable(mb) => mb.reserve(additional),
            Self::Frozen(buf) => {
                let owned = std::mem::take(buf);
                match owned.into_mutable() {
                    Ok(mut mb) => {
                        mb.reserve(additional);
                        *self = Self::Mutable(mb);
                    }
                    Err(orig) => {
                        let mut mb = MutableBuffer::new(orig.len() + additional);
                        mb.extend_from_slice(orig.as_slice());
                        mb.reserve(additional);
                        *self = Self::Mutable(mb);
                    }
                }
            }
        }
    }

    #[inline]
    fn replace_range(&mut self, start: usize, end: usize, replacement: &[u8]) {
        let mut vec = self.as_slice().to_vec();
        vec.splice(start..end, replacement.iter().copied());
        *self = Self::Mutable(MutableBuffer::from(vec));
    }
}

/// Arrow-native string column using contiguous buffer storage
///
/// Memory layout:
/// - offsets: Vec<i32> - 4 bytes per row (start offset into data buffer)
/// - data: Vec<u8> - contiguous string bytes
/// - nulls: BitVec - 1 bit per row
///
/// This is ~80% more memory efficient than Vec<String> for typical workloads
#[derive(Debug, Clone)]
pub struct ArrowStringColumn {
    /// Offsets into the data buffer (length = row_count + 1)
    offsets: Vec<i32>,
    /// Contiguous string data
    data: CowBytes,
    /// Dictionary encoding keys (row -> dict id). Only used when dictionary mode is enabled.
    dict_keys: Vec<u32>,
    /// Dictionary offsets (length = unique_count + 1). Only used when dictionary mode is enabled.
    dict_offsets: Vec<i32>,
    /// Dictionary string data (unique strings only). Only used when dictionary mode is enabled.
    dict_data: CowBytes,
    /// Hash index for dictionary lookup (hash -> dict id). Not serialized.
    #[allow(dead_code)]
    dict_index: HashMap<u64, Vec<u32>>,
    /// Whether dictionary mode is enabled
    dict_enabled: bool,
    /// Sampling counters for adaptive switching
    sample_hashes: Vec<u64>,
    sample_len: usize,
    /// Null bitmap
    nulls: BitVec,
    /// Number of rows
    len: usize,
}

impl ArrowStringColumn {
    pub fn new() -> Self {
        Self {
            offsets: vec![0], // Initial offset
            data: CowBytes::new_mutable(),
            dict_keys: Vec::new(),
            dict_offsets: vec![0],
            dict_data: CowBytes::new_mutable(),
            dict_index: HashMap::new(),
            dict_enabled: false,
            sample_hashes: Vec::new(),
            sample_len: 0,
            nulls: BitVec::new(),
            len: 0,
        }
    }

    pub fn with_capacity(capacity: usize, avg_string_len: usize) -> Self {
        let mut offsets = Vec::with_capacity(capacity + 1);
        offsets.push(0);
        Self {
            offsets,
            data: CowBytes::with_capacity(capacity * avg_string_len),
            dict_keys: Vec::with_capacity(capacity),
            dict_offsets: vec![0],
            dict_data: CowBytes::new_mutable(),
            dict_index: HashMap::new(),
            dict_enabled: false,
            sample_hashes: Vec::new(),
            sample_len: 0,
            nulls: BitVec::with_capacity(capacity),
            len: 0,
        }
    }

    pub fn freeze_for_arrow_export(&mut self) {
        self.data.freeze();
        self.dict_data.freeze();
    }

    pub fn is_dictionary_enabled(&self) -> bool {
        self.dict_enabled
    }

    #[inline]
    fn hash_str(s: &str) -> u64 {
        use ahash::AHasher;
        use std::hash::Hasher;
        let mut hasher = AHasher::default();
        s.hash(&mut hasher);
        hasher.finish()
    }

    #[inline]
    fn maybe_sample_and_switch(&mut self, value: &str) {
        const SAMPLE_TARGET: usize = 4096;
        const UNIQUE_RATIO_THRESHOLD: f64 = 0.85;
        if self.dict_enabled {
            return;
        }
        if self.sample_len >= SAMPLE_TARGET {
            return;
        }
        let h = Self::hash_str(value);
        self.sample_hashes.push(h);
        self.sample_len += 1;
        if self.sample_len == SAMPLE_TARGET {
            let mut uniq = self.sample_hashes.clone();
            uniq.sort_unstable();
            uniq.dedup();
            let unique_ratio = (uniq.len() as f64) / (SAMPLE_TARGET as f64);
            if unique_ratio < UNIQUE_RATIO_THRESHOLD {
                self.enable_dictionary_mode();
            }
            self.sample_hashes.clear();
        }
    }

    fn enable_dictionary_mode(&mut self) {
        if self.dict_enabled {
            return;
        }
        let prev_len = self.len;
        let prev_nulls = self.nulls.clone();

        let prev_offsets = std::mem::take(&mut self.offsets);
        let prev_data = std::mem::take(&mut self.data);

        self.dict_enabled = true;
        self.dict_keys = Vec::with_capacity(prev_len);
        self.dict_offsets = vec![0];
        self.dict_data = CowBytes::new_mutable();
        self.dict_index.clear();

        for i in 0..prev_len {
            if prev_nulls.get(i) {
                self.dict_keys.push(0);
                continue;
            }
            let start = prev_offsets[i] as usize;
            let end = prev_offsets[i + 1] as usize;
            let s = unsafe { std::str::from_utf8_unchecked(&prev_data.as_slice()[start..end]) };
            let key = self.intern_dict(s);
            self.dict_keys.push(key);
        }

        self.offsets = vec![0];
        self.data = CowBytes::new_mutable();
        self.nulls = prev_nulls;
        self.len = prev_len;
    }

    #[inline]
    fn intern_dict(&mut self, value: &str) -> u32 {
        let h = Self::hash_str(value);
        if let Some(candidates) = self.dict_index.get(&h) {
            for &id in candidates {
                let id_usize = id as usize;
                let start = self.dict_offsets[id_usize] as usize;
                let end = self.dict_offsets[id_usize + 1] as usize;
                let existing = unsafe {
                    std::str::from_utf8_unchecked(&self.dict_data.as_slice()[start..end])
                };
                if existing == value {
                    return id;
                }
            }
        }

        let id = (self.dict_offsets.len() - 1) as u32;
        self.dict_data.push_bytes(value.as_bytes());
        self.dict_offsets.push(self.dict_data.len() as i32);
        self.dict_index.entry(h).or_insert_with(Vec::new).push(id);
        id
    }

    /// Push a string value
    #[inline]
    pub fn push(&mut self, value: &str) {
        self.maybe_sample_and_switch(value);
        if self.dict_enabled {
            let key = self.intern_dict(value);
            self.dict_keys.push(key);
            self.nulls.push(false);
            self.len += 1;
            return;
        }
        self.data.push_bytes(value.as_bytes());
        self.offsets.push(self.data.len() as i32);
        self.nulls.push(false);
        self.len += 1;
    }

    /// Push a null value
    #[inline]
    pub fn push_null(&mut self) {
        if self.dict_enabled {
            self.dict_keys.push(0);
            self.nulls.push(true);
            self.len += 1;
            return;
        }
        // Offset stays the same (empty string)
        self.offsets.push(self.data.len() as i32);
        self.nulls.push(true);
        self.len += 1;
    }

    /// Push null values
    #[inline]
    pub fn push_null_n(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        if self.dict_enabled {
            self.dict_keys.reserve(count);
            self.dict_keys.extend(std::iter::repeat(0u32).take(count));
            self.nulls.extend_true(count);
            self.len += count;
            return;
        }

        let last = *self.offsets.last().unwrap_or(&0);
        self.offsets.reserve(count);
        self.offsets.extend(std::iter::repeat(last).take(count));
        self.nulls.extend_true(count);
        self.len += count;
    }

    /// Push from Value
    #[inline]
    pub fn push_value(&mut self, value: &Value) {
        match value {
            Value::String(s) => self.push(s),
            Value::Null => self.push_null(),
            _ => {
                // Convert other types to string
                let s = value.to_string_value();
                self.push(&s);
            }
        }
    }

    /// Get string at index (returns None for null or out of bounds)
    #[inline]
    pub fn get(&self, index: usize) -> Option<&str> {
        if index >= self.len || self.nulls.get(index) {
            return None;
        }
        if self.dict_enabled {
            let key = self.dict_keys.get(index).copied().unwrap_or(0) as usize;
            if key + 1 >= self.dict_offsets.len() {
                return None;
            }
            let start = self.dict_offsets[key] as usize;
            let end = self.dict_offsets[key + 1] as usize;
            return Some(unsafe {
                std::str::from_utf8_unchecked(&self.dict_data.as_slice()[start..end])
            });
        }
        let start = self.offsets[index] as usize;
        let end = self.offsets[index + 1] as usize;
        // SAFETY: We only store valid UTF-8 strings
        Some(unsafe { std::str::from_utf8_unchecked(&self.data.as_slice()[start..end]) })
    }

    /// Get string at index as Value
    #[inline]
    pub fn get_value(&self, index: usize) -> Value {
        match self.get(index) {
            Some(s) => Value::String(s.to_string()),
            None => Value::Null,
        }
    }

    /// Check if value at index is null
    #[inline]
    pub fn is_null(&self, index: usize) -> bool {
        self.nulls.get(index)
    }

    /// Number of rows
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Reserve capacity
    pub fn reserve(&mut self, additional: usize) {
        if self.dict_enabled {
            self.dict_keys.reserve(additional);
            return;
        }
        self.offsets.reserve(additional);
        // Estimate 32 bytes per string on average
        self.data.reserve(additional * 32);
    }

    /// Set value at index (for updates)
    pub fn set(&mut self, index: usize, value: &str) {
        if index >= self.len {
            return;
        }

        if self.dict_enabled {
            let key = self.intern_dict(value);
            if index < self.dict_keys.len() {
                self.dict_keys[index] = key;
            }
            self.nulls.set(index, false);
            return;
        }

        let old_start = self.offsets[index] as usize;
        let old_end = self.offsets[index + 1] as usize;
        let old_len = old_end - old_start;
        let new_len = value.len();

        if new_len == old_len {
            // Same length - in-place update
            // For simplicity and safety across Mutable/Frozen, rebuild this small range
            self.data
                .replace_range(old_start, old_end, value.as_bytes());
        } else {
            // Different length - need to rebuild (expensive but rare)
            let len_diff = new_len as i32 - old_len as i32;

            // Update data buffer
            self.data
                .replace_range(old_start, old_end, value.as_bytes());

            // Update all subsequent offsets
            for i in (index + 1)..=self.len {
                self.offsets[i] = (self.offsets[i] as i32 + len_diff) as i32;
            }
        }

        self.nulls.set(index, false);
    }

    /// Set null at index
    pub fn set_null(&mut self, index: usize) {
        if index < self.len {
            self.nulls.set(index, true);
        }
    }

    /// Slice column for delta extraction [start, end)
    pub fn slice(&self, start: usize, end: usize) -> Self {
        let end = end.min(self.len);
        if start >= end {
            return Self::new();
        }

        if self.dict_enabled {
            let mut out = Self::with_capacity(end - start, 0);
            out.dict_enabled = true;
            out.dict_offsets = self.dict_offsets.clone();
            out.dict_data = self.dict_data.clone();
            out.dict_keys = self.dict_keys[start..end].to_vec();
            out.nulls = self.nulls.slice(start, end);
            out.len = end - start;
            out.rebuild_dict_index();
            return out;
        }

        let data_start = self.offsets[start] as usize;
        let data_end = self.offsets[end] as usize;

        let mut new_offsets = Vec::with_capacity(end - start + 1);
        new_offsets.push(0);
        for i in start..end {
            let len = self.offsets[i + 1] - self.offsets[i];
            new_offsets.push(new_offsets.last().unwrap() + len);
        }

        Self {
            offsets: new_offsets,
            data: CowBytes::Frozen(Buffer::from(
                self.data.as_slice()[data_start..data_end].to_vec(),
            )),
            dict_keys: Vec::new(),
            dict_offsets: vec![0],
            dict_data: CowBytes::new_mutable(),
            dict_index: HashMap::new(),
            dict_enabled: false,
            sample_hashes: Vec::new(),
            sample_len: 0,
            nulls: self.nulls.slice(start, end),
            len: end - start,
        }
    }

    /// Append another ArrowStringColumn
    pub fn append(&mut self, other: &Self) {
        if self.dict_enabled || other.dict_enabled {
            if !self.dict_enabled {
                self.enable_dictionary_mode();
            }
            for i in 0..other.len {
                if other.nulls.get(i) {
                    self.push_null();
                } else if let Some(s) = other.get(i) {
                    self.push(s);
                } else {
                    self.push_null();
                }
            }
            return;
        }
        let base_offset = *self.offsets.last().unwrap_or(&0);

        // Append offsets (skip first which is 0)
        for i in 1..=other.len {
            self.offsets.push(base_offset + other.offsets[i]);
        }

        // Append data
        self.data.push_bytes(other.data.as_slice());

        // Append nulls
        self.nulls.extend(&other.nulls);

        self.len += other.len;
    }

    /// Convert to Arrow StringArray
    pub fn to_arrow_array(&self) -> ArrayRef {
        if self.dict_enabled {
            let values = self.build_dictionary_values();
            let keys = self.build_dictionary_keys(&self.dict_keys);
            if let Ok(dict) = DictionaryArray::<Int32Type>::try_new(keys, values) {
                return Arc::new(dict);
            }

            // Fallback: build Utf8 array (should be rare)
            let mut builder = GenericByteBuilder::<GenericStringType<i32>>::with_capacity(
                self.len,
                self.len * 32,
            );
            for i in 0..self.len {
                if self.nulls.get(i) {
                    builder.append_null();
                } else if let Some(s) = self.get(i) {
                    builder.append_value(s);
                } else {
                    builder.append_null();
                }
            }
            return Arc::new(builder.finish());
        }

        // B: build from offsets + contiguous data buffers
        let offsets: Vec<i32> = self.offsets.clone();
        let value_buf = self.data.as_buffer_cloned();
        let valid = if self.nulls.all_false() {
            None
        } else {
            Some(
                (0..self.len)
                    .map(|i| !self.nulls.get(i))
                    .collect::<Vec<bool>>(),
            )
        };

        Self::build_utf8_array_from_buffers(offsets, value_buf, valid)
    }

    /// Convert to Arrow StringArray for specific indices
    pub fn to_arrow_array_indexed(&self, indices: &[usize]) -> ArrayRef {
        if self.dict_enabled {
            let values = self.build_dictionary_values();
            let mut keys = Vec::with_capacity(indices.len());
            let mut valid = Vec::with_capacity(indices.len());

            for &idx in indices {
                if idx >= self.len || self.nulls.get(idx) {
                    keys.push(0);
                    valid.push(false);
                } else {
                    keys.push(*self.dict_keys.get(idx).unwrap_or(&0) as i32);
                    valid.push(true);
                }
            }

            let key_array =
                Int32Array::new(ScalarBuffer::from(keys), Some(NullBuffer::from(valid)));
            if let Ok(dict) = DictionaryArray::<Int32Type>::try_new(key_array, values) {
                return Arc::new(dict);
            }

            // Fallback: build Utf8 array (should be rare)
            let mut builder = GenericByteBuilder::<GenericStringType<i32>>::with_capacity(
                indices.len(),
                indices.len() * 32,
            );
            for &idx in indices {
                if idx >= self.len || self.nulls.get(idx) {
                    builder.append_null();
                } else if let Some(s) = self.get(idx) {
                    builder.append_value(s);
                } else {
                    builder.append_null();
                }
            }
            return Arc::new(builder.finish());
        }

        // B: gather path - build compact buffers in one pass
        let mut total_bytes = 0usize;
        for &idx in indices {
            if idx < self.len && !self.nulls.get(idx) {
                total_bytes += (self.offsets[idx + 1] - self.offsets[idx]) as usize;
            }
        }

        let mut value_buffer = Vec::with_capacity(total_bytes);
        let mut offsets = Vec::with_capacity(indices.len() + 1);
        offsets.push(0i32);
        let mut valid = Vec::with_capacity(indices.len());

        for &idx in indices {
            if idx >= self.len || self.nulls.get(idx) {
                valid.push(false);
                offsets.push(value_buffer.len() as i32);
                continue;
            }
            let start = self.offsets[idx] as usize;
            let end = self.offsets[idx + 1] as usize;
            value_buffer.extend_from_slice(&self.data.as_slice()[start..end]);
            valid.push(true);
            offsets.push(value_buffer.len() as i32);
        }

        Self::build_utf8_array_from_buffers(offsets, Buffer::from(value_buffer), Some(valid))
    }

    fn build_dictionary_values(&self) -> ArrayRef {
        // Dictionary values never contain nulls
        let offsets: Vec<i32> = self.dict_offsets.clone();
        let values = self.dict_data.as_buffer_cloned();
        Self::build_utf8_array_from_buffers(offsets, values, None)
    }

    fn build_dictionary_keys(&self, keys: &[u32]) -> Int32Array {
        let mut valid = Vec::with_capacity(self.len);
        for i in 0..self.len {
            valid.push(!self.nulls.get(i));
        }
        let keys_i32: Vec<i32> = keys.iter().map(|&k| k as i32).collect();
        Int32Array::new(ScalarBuffer::from(keys_i32), Some(NullBuffer::from(valid)))
    }

    #[inline]
    fn build_utf8_array_from_buffers(
        offsets: Vec<i32>,
        values: Buffer,
        valid: Option<Vec<bool>>,
    ) -> ArrayRef {
        let offset_buffer = OffsetBuffer::new(offsets.into());
        let null_buf = valid.map(NullBuffer::from);
        Arc::new(unsafe { StringArray::new_unchecked(offset_buffer, values, null_buf) })
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        if self.dict_enabled {
            return self.dict_keys.capacity() * 4
                + self.dict_offsets.capacity() * 4
                + self.dict_data.capacity()
                + (self.len + 7) / 8;
        }
        self.offsets.capacity() * 4 + self.data.capacity() + (self.len + 7) / 8
    }

    /// Compare memory usage with equivalent Vec<String>
    pub fn memory_savings_vs_vec_string(&self) -> (usize, usize) {
        let arrow_mem = self.memory_usage();
        let vec_string_mem = self.len * 24
            + if self.dict_enabled {
                self.dict_data.len()
            } else {
                self.data.len()
            };
        (arrow_mem, vec_string_mem)
    }

    fn rebuild_dict_index(&mut self) {
        if !self.dict_enabled {
            return;
        }
        self.dict_index.clear();
        let unique = self.dict_offsets.len().saturating_sub(1);
        for id in 0..unique {
            let start = self.dict_offsets[id] as usize;
            let end = self.dict_offsets[id + 1] as usize;
            let s =
                unsafe { std::str::from_utf8_unchecked(&self.dict_data.as_slice()[start..end]) };
            let h = Self::hash_str(s);
            self.dict_index
                .entry(h)
                .or_insert_with(Vec::new)
                .push(id as u32);
        }
    }

    /// Get raw nulls reference
    pub fn nulls(&self) -> &BitVec {
        &self.nulls
    }

    /// Extend from iterator of strings
    pub fn extend_from_strings<'a, I>(&mut self, iter: I)
    where
        I: Iterator<Item = Option<&'a str>>,
    {
        for opt_s in iter {
            match opt_s {
                Some(s) => self.push(s),
                None => self.push_null(),
            }
        }
    }

    /// Batch push from Vec<String> (for migration from old format)
    pub fn extend_from_vec_string(&mut self, strings: &[String], nulls: &BitVec) {
        self.reserve(strings.len());
        for (i, s) in strings.iter().enumerate() {
            if nulls.get(i) {
                self.push_null();
            } else {
                self.push(s);
            }
        }
    }
}

/// Custom serialization for ArrowStringColumn
impl Serialize for ArrowStringColumn {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("ArrowStringColumn", 9)?;
        state.serialize_field("offsets", &self.offsets)?;
        state.serialize_field("data", &self.data.as_slice())?;
        state.serialize_field("dict_keys", &self.dict_keys)?;
        state.serialize_field("dict_offsets", &self.dict_offsets)?;
        state.serialize_field("dict_data", &self.dict_data.as_slice())?;
        state.serialize_field("dict_enabled", &self.dict_enabled)?;
        state.serialize_field("nulls", &self.nulls)?;
        state.serialize_field("len", &self.len)?;
        state.serialize_field("sample_len", &self.sample_len)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for ArrowStringColumn {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ArrowStringColumnData {
            offsets: Vec<i32>,
            data: Vec<u8>,
            #[serde(default)]
            dict_keys: Vec<u32>,
            #[serde(default)]
            dict_offsets: Vec<i32>,
            #[serde(default)]
            dict_data: Vec<u8>,
            #[serde(default)]
            dict_enabled: bool,
            nulls: BitVec,
            len: usize,
            #[serde(default)]
            sample_len: usize,
        }

        let data = ArrowStringColumnData::deserialize(deserializer)?;
        let mut out = Self {
            offsets: data.offsets,
            data: CowBytes::Frozen(Buffer::from(data.data)),
            nulls: data.nulls,
            len: data.len,
            dict_keys: data.dict_keys,
            dict_offsets: if data.dict_offsets.is_empty() {
                vec![0]
            } else {
                data.dict_offsets
            },
            dict_data: CowBytes::Frozen(Buffer::from(data.dict_data)),
            dict_index: HashMap::new(),
            dict_enabled: data.dict_enabled,
            sample_hashes: Vec::new(),
            sample_len: data.sample_len,
        };
        if out.dict_enabled {
            out.rebuild_dict_index();
        }
        Ok(out)
    }
}

/// Arrow-native TypedColumn enum with optimized string storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArrowTypedColumn {
    Int64 {
        data: Vec<i64>,
        nulls: BitVec,
    },
    Float64 {
        data: Vec<f64>,
        nulls: BitVec,
    },
    /// Arrow-native string storage with contiguous buffers
    String(ArrowStringColumn),
    Bool {
        data: BitVec,
        nulls: BitVec,
    },
    Mixed {
        data: Vec<Value>,
        nulls: BitVec,
    },
}

impl ArrowTypedColumn {
    pub fn new(dtype: DataType) -> Self {
        match dtype {
            DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                ArrowTypedColumn::Int64 {
                    data: Vec::new(),
                    nulls: BitVec::new(),
                }
            }
            DataType::Float64 | DataType::Float32 => ArrowTypedColumn::Float64 {
                data: Vec::new(),
                nulls: BitVec::new(),
            },
            DataType::String => ArrowTypedColumn::String(ArrowStringColumn::new()),
            DataType::Bool => ArrowTypedColumn::Bool {
                data: BitVec::new(),
                nulls: BitVec::new(),
            },
            _ => ArrowTypedColumn::Mixed {
                data: Vec::new(),
                nulls: BitVec::new(),
            },
        }
    }

    pub fn with_capacity(dtype: DataType, capacity: usize) -> Self {
        match dtype {
            DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                ArrowTypedColumn::Int64 {
                    data: Vec::with_capacity(capacity),
                    nulls: BitVec::with_capacity(capacity),
                }
            }
            DataType::Float64 | DataType::Float32 => ArrowTypedColumn::Float64 {
                data: Vec::with_capacity(capacity),
                nulls: BitVec::with_capacity(capacity),
            },
            DataType::String => {
                ArrowTypedColumn::String(ArrowStringColumn::with_capacity(capacity, 32))
            }
            DataType::Bool => ArrowTypedColumn::Bool {
                data: BitVec::with_capacity(capacity),
                nulls: BitVec::with_capacity(capacity),
            },
            _ => ArrowTypedColumn::Mixed {
                data: Vec::with_capacity(capacity),
                nulls: BitVec::with_capacity(capacity),
            },
        }
    }

    /// Push a value to the column
    #[inline]
    pub fn push(&mut self, value: &Value) {
        match (self, value) {
            (ArrowTypedColumn::Int64 { data, nulls }, Value::Int64(v)) => {
                data.push(*v);
                nulls.push(false);
            }
            (ArrowTypedColumn::Int64 { data, nulls }, Value::Int32(v)) => {
                data.push(*v as i64);
                nulls.push(false);
            }
            (ArrowTypedColumn::Int64 { data, nulls }, Value::Null) => {
                data.push(0);
                nulls.push(true);
            }
            (ArrowTypedColumn::Float64 { data, nulls }, Value::Float64(v)) => {
                data.push(*v);
                nulls.push(false);
            }
            (ArrowTypedColumn::Float64 { data, nulls }, Value::Float32(v)) => {
                data.push(*v as f64);
                nulls.push(false);
            }
            (ArrowTypedColumn::Float64 { data, nulls }, Value::Int64(v)) => {
                data.push(*v as f64);
                nulls.push(false);
            }
            (ArrowTypedColumn::Float64 { data, nulls }, Value::Null) => {
                data.push(0.0);
                nulls.push(true);
            }
            (ArrowTypedColumn::String(col), Value::String(v)) => {
                col.push(v);
            }
            (ArrowTypedColumn::String(col), Value::Null) => {
                col.push_null();
            }
            (ArrowTypedColumn::Bool { data, nulls }, Value::Bool(v)) => {
                data.push(*v);
                nulls.push(false);
            }
            (ArrowTypedColumn::Bool { data, nulls }, Value::Null) => {
                data.push(false);
                nulls.push(true);
            }
            (ArrowTypedColumn::Mixed { data, nulls }, v) => {
                nulls.push(v.is_null());
                data.push(v.clone());
            }
            (col, _value) => {
                col.push_null();
            }
        }
    }

    /// Push a null value
    #[inline]
    pub fn push_null(&mut self) {
        match self {
            ArrowTypedColumn::Int64 { data, nulls } => {
                data.push(0);
                nulls.push(true);
            }
            ArrowTypedColumn::Float64 { data, nulls } => {
                data.push(0.0);
                nulls.push(true);
            }
            ArrowTypedColumn::String(col) => {
                col.push_null();
            }
            ArrowTypedColumn::Bool { data, nulls } => {
                data.push(false);
                nulls.push(true);
            }
            ArrowTypedColumn::Mixed { data, nulls } => {
                data.push(Value::Null);
                nulls.push(true);
            }
        }
    }

    /// Get value at index
    #[inline]
    pub fn get(&self, index: usize) -> Option<Value> {
        match self {
            ArrowTypedColumn::Int64 { data, nulls } => {
                if index >= data.len() || nulls.get(index) {
                    Some(Value::Null)
                } else {
                    Some(Value::Int64(data[index]))
                }
            }
            ArrowTypedColumn::Float64 { data, nulls } => {
                if index >= data.len() || nulls.get(index) {
                    Some(Value::Null)
                } else {
                    Some(Value::Float64(data[index]))
                }
            }
            ArrowTypedColumn::String(col) => Some(col.get_value(index)),
            ArrowTypedColumn::Bool { data, nulls } => {
                if index >= data.len() || nulls.get(index) {
                    Some(Value::Null)
                } else {
                    Some(Value::Bool(data.get(index)))
                }
            }
            ArrowTypedColumn::Mixed { data, nulls } => {
                if index >= data.len() {
                    None
                } else if nulls.get(index) {
                    Some(Value::Null)
                } else {
                    Some(data[index].clone())
                }
            }
        }
    }

    /// Get string reference (for string columns only) - zero-copy
    #[inline]
    pub fn get_str(&self, index: usize) -> Option<&str> {
        match self {
            ArrowTypedColumn::String(col) => col.get(index),
            _ => None,
        }
    }

    /// Set value at index
    #[inline]
    pub fn set(&mut self, index: usize, value: &Value) {
        match (self, value) {
            (ArrowTypedColumn::Int64 { data, nulls }, Value::Int64(v)) => {
                if index < data.len() {
                    data[index] = *v;
                    nulls.set(index, false);
                }
            }
            (ArrowTypedColumn::Int64 { data, nulls }, Value::Null) => {
                if index < data.len() {
                    nulls.set(index, true);
                }
            }
            (ArrowTypedColumn::Float64 { data, nulls }, Value::Float64(v)) => {
                if index < data.len() {
                    data[index] = *v;
                    nulls.set(index, false);
                }
            }
            (ArrowTypedColumn::Float64 { data, nulls }, Value::Null) => {
                if index < data.len() {
                    nulls.set(index, true);
                }
            }
            (ArrowTypedColumn::String(col), Value::String(v)) => {
                col.set(index, v);
            }
            (ArrowTypedColumn::String(col), Value::Null) => {
                col.set_null(index);
            }
            (ArrowTypedColumn::Bool { data, nulls }, Value::Bool(v)) => {
                if index < data.len() {
                    data.set(index, *v);
                    nulls.set(index, false);
                }
            }
            (ArrowTypedColumn::Bool { data, nulls }, Value::Null) => {
                if index < data.len() {
                    nulls.set(index, true);
                }
            }
            (ArrowTypedColumn::Mixed { data, nulls }, v) => {
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
            ArrowTypedColumn::Int64 { data, .. } => data.len(),
            ArrowTypedColumn::Float64 { data, .. } => data.len(),
            ArrowTypedColumn::String(col) => col.len(),
            ArrowTypedColumn::Bool { data, .. } => data.len(),
            ArrowTypedColumn::Mixed { data, .. } => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_null(&self, index: usize) -> bool {
        match self {
            ArrowTypedColumn::Int64 { nulls, .. } => nulls.get(index),
            ArrowTypedColumn::Float64 { nulls, .. } => nulls.get(index),
            ArrowTypedColumn::String(col) => col.is_null(index),
            ArrowTypedColumn::Bool { nulls, .. } => nulls.get(index),
            ArrowTypedColumn::Mixed { nulls, .. } => nulls.get(index),
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            ArrowTypedColumn::Int64 { .. } => DataType::Int64,
            ArrowTypedColumn::Float64 { .. } => DataType::Float64,
            ArrowTypedColumn::String(_) => DataType::String,
            ArrowTypedColumn::Bool { .. } => DataType::Bool,
            ArrowTypedColumn::Mixed { .. } => DataType::Json,
        }
    }

    /// Reserve capacity
    pub fn reserve(&mut self, additional: usize) {
        match self {
            ArrowTypedColumn::Int64 { data, .. } => data.reserve(additional),
            ArrowTypedColumn::Float64 { data, .. } => data.reserve(additional),
            ArrowTypedColumn::String(col) => col.reserve(additional),
            ArrowTypedColumn::Bool { .. } => {}
            ArrowTypedColumn::Mixed { data, .. } => data.reserve(additional),
        }
    }

    /// Slice column for delta extraction [start, end)
    pub fn slice(&self, start: usize, end: usize) -> Self {
        match self {
            ArrowTypedColumn::Int64 { data, nulls } => ArrowTypedColumn::Int64 {
                data: data[start..end].to_vec(),
                nulls: nulls.slice(start, end),
            },
            ArrowTypedColumn::Float64 { data, nulls } => ArrowTypedColumn::Float64 {
                data: data[start..end].to_vec(),
                nulls: nulls.slice(start, end),
            },
            ArrowTypedColumn::String(col) => ArrowTypedColumn::String(col.slice(start, end)),
            ArrowTypedColumn::Bool { data, nulls } => ArrowTypedColumn::Bool {
                data: data.slice(start, end),
                nulls: nulls.slice(start, end),
            },
            ArrowTypedColumn::Mixed { data, nulls } => ArrowTypedColumn::Mixed {
                data: data[start..end].to_vec(),
                nulls: nulls.slice(start, end),
            },
        }
    }

    /// Append another column
    pub fn append(&mut self, other: Self) {
        match (self, other) {
            (
                ArrowTypedColumn::Int64 { data, nulls },
                ArrowTypedColumn::Int64 {
                    data: od,
                    nulls: on,
                },
            ) => {
                data.extend(od);
                nulls.extend(&on);
            }
            (
                ArrowTypedColumn::Float64 { data, nulls },
                ArrowTypedColumn::Float64 {
                    data: od,
                    nulls: on,
                },
            ) => {
                data.extend(od);
                nulls.extend(&on);
            }
            (ArrowTypedColumn::String(col), ArrowTypedColumn::String(other_col)) => {
                col.append(&other_col);
            }
            (
                ArrowTypedColumn::Bool { data, nulls },
                ArrowTypedColumn::Bool {
                    data: od,
                    nulls: on,
                },
            ) => {
                data.extend(&od);
                nulls.extend(&on);
            }
            (
                ArrowTypedColumn::Mixed { data, nulls },
                ArrowTypedColumn::Mixed {
                    data: od,
                    nulls: on,
                },
            ) => {
                data.extend(od);
                nulls.extend(&on);
            }
            _ => {}
        }
    }

    /// Convert to Arrow ArrayRef
    pub fn to_arrow_array(&self) -> ArrayRef {
        match self {
            ArrowTypedColumn::Int64 { data, nulls } => {
                let mut builder = Int64Builder::with_capacity(data.len());
                for (i, &v) in data.iter().enumerate() {
                    if nulls.get(i) {
                        builder.append_null();
                    } else {
                        builder.append_value(v);
                    }
                }
                Arc::new(builder.finish())
            }
            ArrowTypedColumn::Float64 { data, nulls } => {
                let mut builder = Float64Builder::with_capacity(data.len());
                for (i, &v) in data.iter().enumerate() {
                    if nulls.get(i) {
                        builder.append_null();
                    } else {
                        builder.append_value(v);
                    }
                }
                Arc::new(builder.finish())
            }
            ArrowTypedColumn::String(col) => col.to_arrow_array(),
            ArrowTypedColumn::Bool { data, nulls } => {
                let mut builder = BooleanBuilder::with_capacity(data.len());
                for i in 0..data.len() {
                    if nulls.get(i) {
                        builder.append_null();
                    } else {
                        builder.append_value(data.get(i));
                    }
                }
                Arc::new(builder.finish())
            }
            ArrowTypedColumn::Mixed { data, nulls } => {
                // Convert to string array
                let mut builder = GenericByteBuilder::<GenericStringType<i32>>::with_capacity(
                    data.len(),
                    data.len() * 32,
                );
                for (i, v) in data.iter().enumerate() {
                    if nulls.get(i) {
                        builder.append_null();
                    } else {
                        builder.append_value(v.to_string_value());
                    }
                }
                Arc::new(builder.finish())
            }
        }
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        match self {
            ArrowTypedColumn::Int64 { data, nulls } => data.capacity() * 8 + (nulls.len() + 7) / 8,
            ArrowTypedColumn::Float64 { data, nulls } => {
                data.capacity() * 8 + (nulls.len() + 7) / 8
            }
            ArrowTypedColumn::String(col) => col.memory_usage(),
            ArrowTypedColumn::Bool { data, nulls } => (data.len() + 7) / 8 + (nulls.len() + 7) / 8,
            ArrowTypedColumn::Mixed { data, nulls } => {
                // Rough estimate: 64 bytes per Value on average
                data.capacity() * 64 + (nulls.len() + 7) / 8
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arrow_string_column() {
        let mut col = ArrowStringColumn::new();
        col.push("hello");
        col.push("world");
        col.push_null();
        col.push("test");

        assert_eq!(col.len(), 4);
        assert_eq!(col.get(0), Some("hello"));
        assert_eq!(col.get(1), Some("world"));
        assert_eq!(col.get(2), None);
        assert_eq!(col.get(3), Some("test"));
        assert!(col.is_null(2));
    }

    #[test]
    fn test_memory_savings() {
        let mut col = ArrowStringColumn::new();
        for i in 0..10000 {
            col.push(&format!("test string number {}", i));
        }

        let (arrow_mem, vec_mem) = col.memory_savings_vs_vec_string();
        assert!(arrow_mem < vec_mem, "Arrow should use less memory");
        println!(
            "Arrow: {} bytes, Vec<String>: {} bytes, Savings: {:.1}%",
            arrow_mem,
            vec_mem,
            (1.0 - arrow_mem as f64 / vec_mem as f64) * 100.0
        );
    }
}
