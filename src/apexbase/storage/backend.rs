//! Storage Backend Bridge
//!
//! This module bridges OnDemandStorage with ColumnTable, enabling:
//! - Lazy loading: only load data when needed
//! - Column projection: only load requested columns
//! Memory-efficient persistence using the V4 format

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use parking_lot::RwLock;

use arrow::record_batch::RecordBatch;

use crate::data::{DataType, Value};
use crate::storage::on_demand::{
    ColumnData, ColumnType, ColumnValue, CompressionType, OnDemandStorage,
};
use crate::table::arrow_column::ArrowStringColumn;
use crate::table::column_table::{BitVec, TypedColumn};

// ============================================================================
// Global Dict Cache — persists across backend opens for GROUP BY acceleration
// ============================================================================

/// Global string dictionary cache keyed by (file_path, column_name).
/// Invalidated by file modification time.
static GLOBAL_DICT_CACHE: once_cell::sync::Lazy<
    RwLock<HashMap<(PathBuf, String), (SystemTime, Arc<(Vec<String>, Vec<u16>)>)>>,
> = once_cell::sync::Lazy::new(|| RwLock::new(HashMap::new()));

type FirstStringRowIdCache = ahash::AHashMap<Box<str>, u64>;

/// Get or build a global dict cache entry. Returns Arc to avoid cloning 1M+ entries.
pub fn get_global_dict_cache(
    path: &Path,
    col_name: &str,
    storage: &OnDemandStorage,
) -> io::Result<Option<Arc<(Vec<String>, Vec<u16>)>>> {
    let mtime = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);
    let key = (path.to_path_buf(), col_name.to_string());

    // Check cache
    {
        let cache = GLOBAL_DICT_CACHE.read();
        if let Some((cached_mtime, data)) = cache.get(&key) {
            if *cached_mtime >= mtime {
                return Ok(Some(Arc::clone(data)));
            }
        }
    }

    // Build and cache
    if let Some((dict_strings, group_ids)) = storage.build_string_dict_cache(col_name)? {
        let data = Arc::new((dict_strings, group_ids));
        let mut cache = GLOBAL_DICT_CACHE.write();
        cache.insert(key, (mtime, Arc::clone(&data)));
        Ok(Some(data))
    } else {
        Ok(None)
    }
}

/// Invalidate global dict cache for a file path
pub fn invalidate_global_dict_cache(path: &Path) {
    let mut cache = GLOBAL_DICT_CACHE.write();
    cache.retain(|(p, _), _| p != path);
}

// ============================================================================
// Type Conversions
// ============================================================================

/// Convert DataType to OnDemand ColumnType
pub fn datatype_to_column_type(dt: &DataType) -> ColumnType {
    match dt {
        DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => ColumnType::Int64,
        DataType::Float64 | DataType::Float32 => ColumnType::Float64,
        DataType::String => ColumnType::String,
        DataType::Bool => ColumnType::Bool,
        DataType::Binary => ColumnType::Binary,
        DataType::Timestamp => ColumnType::Timestamp,
        DataType::Date => ColumnType::Date,
        _ => ColumnType::String, // Fallback for complex types
    }
}

/// Convert OnDemand ColumnType to DataType
pub fn column_type_to_datatype(ct: ColumnType) -> DataType {
    match ct {
        ColumnType::Int64
        | ColumnType::Int32
        | ColumnType::Int16
        | ColumnType::Int8
        | ColumnType::UInt64
        | ColumnType::UInt32
        | ColumnType::UInt16
        | ColumnType::UInt8 => DataType::Int64,
        ColumnType::Float64 | ColumnType::Float32 => DataType::Float64,
        ColumnType::String | ColumnType::StringDict => DataType::String,
        ColumnType::Bool => DataType::Bool,
        ColumnType::Binary => DataType::Binary,
        ColumnType::FixedList | ColumnType::Float16List => DataType::Binary,
        ColumnType::Timestamp => DataType::Timestamp,
        ColumnType::Date => DataType::Date,
        ColumnType::Null => DataType::String,
    }
}

/// Convert TypedColumn to OnDemand ColumnData
pub fn typed_column_to_column_data(col: &TypedColumn) -> ColumnData {
    match col {
        TypedColumn::Int64 { data, .. } => {
            let mut cd = ColumnData::new(ColumnType::Int64);
            cd.extend_i64(data);
            cd
        }
        TypedColumn::Float64 { data, .. } => {
            let mut cd = ColumnData::new(ColumnType::Float64);
            cd.extend_f64(data);
            cd
        }
        TypedColumn::String(arrow_col) => {
            let mut cd = ColumnData::new(ColumnType::String);
            for i in 0..arrow_col.len() {
                if let Some(s) = arrow_col.get(i) {
                    cd.push_string(&s);
                } else {
                    cd.push_string("");
                }
            }
            cd
        }
        TypedColumn::Bool { data, .. } => {
            let mut cd = ColumnData::new(ColumnType::Bool);
            for i in 0..data.len() {
                cd.push_bool(data.get(i));
            }
            cd
        }
        TypedColumn::Mixed { data, .. } => {
            // Serialize mixed as JSON strings
            let mut cd = ColumnData::new(ColumnType::String);
            for v in data {
                let s = match v {
                    Value::String(s) => s.clone(),
                    Value::Int64(i) => i.to_string(),
                    Value::Float64(f) => f.to_string(),
                    Value::Bool(b) => b.to_string(),
                    _ => serde_json::to_string(v).unwrap_or_default(),
                };
                cd.push_string(&s);
            }
            cd
        }
    }
}

/// Convert OnDemand ColumnData to TypedColumn
pub fn column_data_to_typed_column(cd: &ColumnData, _dtype: DataType) -> TypedColumn {
    match cd {
        ColumnData::Int64(data) => {
            let mut nulls = BitVec::new();
            nulls.extend_false(data.len());
            TypedColumn::Int64 {
                data: data.clone(),
                nulls,
            }
        }
        ColumnData::Float64(data) => {
            let mut nulls = BitVec::new();
            nulls.extend_false(data.len());
            TypedColumn::Float64 {
                data: data.clone(),
                nulls,
            }
        }
        ColumnData::Bool { data, len } => {
            let mut bit_data = BitVec::new();
            let mut nulls = BitVec::new();
            for i in 0..*len {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                let val = if byte_idx < data.len() {
                    (data[byte_idx] >> bit_idx) & 1 == 1
                } else {
                    false
                };
                bit_data.push(val);
                nulls.push(false);
            }
            TypedColumn::Bool {
                data: bit_data,
                nulls,
            }
        }
        ColumnData::String { offsets, data } => {
            let mut arrow_col = ArrowStringColumn::new();
            let count = offsets.len().saturating_sub(1);
            for i in 0..count {
                let start = offsets[i] as usize;
                let end = offsets[i + 1] as usize;
                if let Ok(s) = std::str::from_utf8(&data[start..end]) {
                    arrow_col.push(s);
                } else {
                    arrow_col.push_null();
                }
            }
            TypedColumn::String(arrow_col)
        }
        ColumnData::Binary { offsets, data } => {
            // Convert binary to Mixed with Binary values
            let mut values = Vec::new();
            let mut nulls = BitVec::new();
            let count = offsets.len().saturating_sub(1);
            for i in 0..count {
                let start = offsets[i] as usize;
                let end = offsets[i + 1] as usize;
                values.push(Value::Binary(data[start..end].to_vec()));
                nulls.push(false);
            }
            TypedColumn::Mixed {
                data: values,
                nulls,
            }
        }
        ColumnData::StringDict {
            indices,
            dict_offsets,
            dict_data,
        } => {
            // Convert dictionary-encoded string to regular String column
            let mut arrow_col = ArrowStringColumn::new();
            for &idx in indices {
                if idx == 0 {
                    arrow_col.push_null();
                } else {
                    let dict_idx = (idx - 1) as usize;
                    if dict_idx + 1 < dict_offsets.len() {
                        let start = dict_offsets[dict_idx] as usize;
                        let end = dict_offsets[dict_idx + 1] as usize;
                        if let Ok(s) = std::str::from_utf8(&dict_data[start..end]) {
                            arrow_col.push(s);
                        } else {
                            arrow_col.push_null();
                        }
                    } else {
                        arrow_col.push_null();
                    }
                }
            }
            TypedColumn::String(arrow_col)
        }
        ColumnData::FixedList { data, dim } => {
            // Represent as Mixed<Value::Binary> for legacy TypedColumn
            let mut values = Vec::new();
            let mut nulls = BitVec::new();
            let dim_usize = *dim as usize;
            let row_count = if dim_usize == 0 {
                0
            } else {
                data.len() / (dim_usize * 4)
            };
            for i in 0..row_count {
                let start = i * dim_usize * 4;
                let end = start + dim_usize * 4;
                values.push(Value::Binary(data[start..end].to_vec()));
                nulls.push(false);
            }
            TypedColumn::Mixed {
                data: values,
                nulls,
            }
        }
        ColumnData::Float16List { data, dim } => {
            let mut values = Vec::new();
            let mut nulls = BitVec::new();
            let dim_usize = *dim as usize;
            let row_count = if dim_usize == 0 {
                0
            } else {
                data.len() / (dim_usize * 2)
            };
            for i in 0..row_count {
                let start = i * dim_usize * 2;
                let end = start + dim_usize * 2;
                values.push(Value::Binary(data[start..end].to_vec()));
                nulls.push(false);
            }
            TypedColumn::Mixed {
                data: values,
                nulls,
            }
        }
    }
}

/// Convert ColumnData::Float16List to (ArrowDataType, ArrayRef) - decodes f16->f32
fn float16list_to_arrow_pair(
    data: &[u8],
    dim: u32,
) -> (
    arrow::datatypes::DataType,
    std::sync::Arc<dyn arrow::array::Array>,
) {
    let values = crate::storage::on_demand::f16_bytes_to_f32_values(data);
    fixedlist_values_to_arrow_pair(values, dim)
}

/// Convert ColumnData::FixedList to (ArrowDataType, ArrayRef)
fn fixedlist_to_arrow_pair(
    data: &[u8],
    dim: u32,
) -> (
    arrow::datatypes::DataType,
    std::sync::Arc<dyn arrow::array::Array>,
) {
    let values = crate::storage::on_demand::f32_le_bytes_to_values(data);
    fixedlist_values_to_arrow_pair(values, dim)
}

fn fixedlist_values_to_arrow_pair(
    values: Vec<f32>,
    dim: u32,
) -> (
    arrow::datatypes::DataType,
    std::sync::Arc<dyn arrow::array::Array>,
) {
    use arrow::array::{FixedSizeListArray, Float32Array};
    use arrow::datatypes::{DataType as ArrowDataType, Field};
    let dim_usize = dim as usize;
    let row_count = if dim_usize == 0 {
        0
    } else {
        values.len() / dim_usize
    };
    let float_arr = Float32Array::from(
        values
            .into_iter()
            .take(row_count * dim_usize)
            .collect::<Vec<_>>(),
    );
    let item_field = std::sync::Arc::new(Field::new("item", ArrowDataType::Float32, false));
    let list_dt = ArrowDataType::FixedSizeList(item_field.clone(), dim_usize as i32);
    let arr = FixedSizeListArray::new(
        item_field,
        dim_usize as i32,
        std::sync::Arc::new(float_arr),
        None,
    );
    (
        list_dt,
        std::sync::Arc::new(arr) as std::sync::Arc<dyn arrow::array::Array>,
    )
}

// ============================================================================
// TableStorageBackend - Lazy Loading Storage Backend
// ============================================================================

/// Metadata for a lazy-loaded table
#[derive(Debug, Clone)]
pub struct TableMetadata {
    pub name: String,
    pub row_count: u64,
    pub schema: Vec<(String, DataType)>,
}

/// Storage backend with lazy loading support
///
/// This backend uses OnDemandStorage for persistence and supports:
/// - Lazy loading: data is only loaded when requested
/// - Column projection: only load specific columns
/// - Memory release: unload columns when not needed
/// - Configurable durability levels for ACID guarantees
pub struct TableStorageBackend {
    path: PathBuf,
    pub(crate) storage: OnDemandStorage,
    /// Cached column data (column_name -> TypedColumn)
    /// Only loaded columns are in cache
    cached_columns: RwLock<HashMap<String, TypedColumn>>,
    /// Schema mapping (column_name -> DataType)
    schema: RwLock<Vec<(String, DataType)>>,
    /// Cached row count
    row_count: RwLock<u64>,
    /// Whether data has been modified (needs save)
    dirty: RwLock<bool>,
    /// Cached string dictionary indices for GROUP BY acceleration
    /// col_name -> (dict_strings, group_ids)
    dict_cache: RwLock<HashMap<String, (Vec<String>, Vec<u16>)>>,
    /// High-cardinality string equality accelerator for `WHERE col = 'x' LIMIT 1`.
    /// Stores the first active row id for each distinct string value.
    first_string_row_id_cache: RwLock<HashMap<String, Arc<FirstStringRowIdCache>>>,
}

impl TableStorageBackend {
    /// Helper to build Self from storage (reduces code duplication)
    #[inline]
    fn from_storage_with_row_count(path: &Path, storage: OnDemandStorage, row_count: u64) -> Self {
        let storage_schema = storage.get_schema();
        let schema: Vec<(String, DataType)> = storage_schema
            .into_iter()
            .map(|(name, ct)| (name, column_type_to_datatype(ct)))
            .collect();
        Self {
            path: path.to_path_buf(),
            storage,
            cached_columns: RwLock::new(HashMap::new()),
            schema: RwLock::new(schema),
            row_count: RwLock::new(row_count),
            dirty: RwLock::new(false),
            dict_cache: RwLock::new(HashMap::new()),
            first_string_row_id_cache: RwLock::new(HashMap::new()),
        }
    }

    #[inline]
    fn from_storage(path: &Path, storage: OnDemandStorage) -> Self {
        let row_count = storage.row_count();
        Self::from_storage_with_row_count(path, storage, row_count)
    }

    pub fn create(path: &Path) -> io::Result<Self> {
        Self::create_with_durability(path, super::DurabilityLevel::Fast)
    }

    pub fn create_with_durability(
        path: &Path,
        durability: super::DurabilityLevel,
    ) -> io::Result<Self> {
        Self::create_with_schema_and_durability(path, durability, &[])
    }

    pub fn create_with_schema_and_durability(
        path: &Path,
        durability: super::DurabilityLevel,
        schema_cols: &[(String, ColumnType)],
    ) -> io::Result<Self> {
        let storage =
            OnDemandStorage::create_with_schema_and_durability(path, durability, schema_cols)?;
        let schema: Vec<(String, DataType)> = schema_cols
            .iter()
            .map(|(name, ct)| (name.clone(), column_type_to_datatype(*ct)))
            .collect();
        Ok(Self {
            path: path.to_path_buf(),
            storage,
            cached_columns: RwLock::new(HashMap::new()),
            schema: RwLock::new(schema),
            row_count: RwLock::new(0),
            dirty: RwLock::new(false),
            dict_cache: RwLock::new(HashMap::new()),
            first_string_row_id_cache: RwLock::new(HashMap::new()),
        })
    }

    pub fn open(path: &Path) -> io::Result<Self> {
        Self::open_with_durability(path, super::DurabilityLevel::Fast)
    }

    pub fn open_with_durability(
        path: &Path,
        durability: super::DurabilityLevel,
    ) -> io::Result<Self> {
        let storage = OnDemandStorage::open_with_durability(path, durability)?;
        Ok(Self::from_storage(path, storage))
    }

    /// Open for reading only using a pre-opened File and known file_len.
    /// Saves 2 syscalls vs open(): skips internal File::open + DeltaStore stat.
    pub fn open_with_file(path: &Path, file: std::fs::File, file_len: u64) -> io::Result<Self> {
        let storage = OnDemandStorage::open_for_read_with_file(path, file, file_len)?;
        Ok(Self::from_storage(path, storage))
    }

    pub fn open_or_create(path: &Path) -> io::Result<Self> {
        Self::open_or_create_with_durability(path, super::DurabilityLevel::Fast)
    }

    pub fn open_or_create_with_durability(
        path: &Path,
        durability: super::DurabilityLevel,
    ) -> io::Result<Self> {
        if path.exists() {
            Self::open_with_durability(path, durability)
        } else {
            Self::create_with_durability(path, durability)
        }
    }

    pub fn open_for_write(path: &Path) -> io::Result<Self> {
        Self::open_for_write_with_durability(path, super::DurabilityLevel::Fast)
    }

    pub fn open_for_write_with_durability(
        path: &Path,
        durability: super::DurabilityLevel,
    ) -> io::Result<Self> {
        let storage = OnDemandStorage::open_for_write_with_durability(path, durability)?;
        Ok(Self::from_storage(path, storage))
    }

    /// Open for compaction only — loads metadata, NOT column data.
    /// The streaming compact reads columns one at a time from mmap.
    pub fn open_for_compact(path: &Path) -> io::Result<Self> {
        let storage =
            OnDemandStorage::open_for_insert_with_durability(path, super::DurabilityLevel::Fast)?;
        Ok(Self::from_storage(path, storage))
    }

    pub fn open_for_insert(path: &Path) -> io::Result<Self> {
        Self::open_for_insert_with_durability(path, super::DurabilityLevel::Fast)
    }

    pub fn open_for_insert_with_durability(
        path: &Path,
        durability: super::DurabilityLevel,
    ) -> io::Result<Self> {
        let storage = OnDemandStorage::open_for_insert_with_durability(path, durability)?;
        let row_count = storage.base_row_count();
        Ok(Self::from_storage_with_row_count(path, storage, row_count))
    }

    /// Open for DELETE operations — mmap only, does NOT load column data into memory.
    /// Used by execute_delete to avoid the expensive load_all_columns_into_memory() + save_v4()
    /// cycle. Deletion vectors are updated in-place via save_delete_only().
    pub fn open_for_delete(path: &Path) -> io::Result<Self> {
        // Light-weight open: skips tmp file cleanup, DeltaStore::load, and WAL init.
        // None of these are needed for the in-place delete fast path (~50µs savings).
        // Falls back to Self::open() if file::open or metadata fails.
        if let Ok(file) = std::fs::File::open(path) {
            if let Ok(meta) = file.metadata() {
                return Self::open_with_file(path, file, meta.len());
            }
        }
        Self::open(path)
    }

    pub fn open_for_schema_change(path: &Path) -> io::Result<Self> {
        Self::open_for_schema_change_with_durability(path, super::DurabilityLevel::Fast)
    }

    pub fn open_for_schema_change_with_durability(
        path: &Path,
        durability: super::DurabilityLevel,
    ) -> io::Result<Self> {
        let storage = OnDemandStorage::open_for_schema_change_with_durability(path, durability)?;
        Ok(Self::from_storage(path, storage))
    }

    /// Insert rows to delta file (memory efficient - doesn't load existing column data)
    /// Auto-compacts when delta exceeds threshold
    pub fn insert_rows_to_delta(&self, rows: &[HashMap<String, Value>]) -> io::Result<Vec<u64>> {
        let converted: Vec<HashMap<String, ColumnValue>> = rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|(k, v)| {
                        let cv = match v {
                            Value::Int64(i) => ColumnValue::Int64(*i),
                            Value::Float64(f) => ColumnValue::Float64(*f),
                            Value::String(s) => ColumnValue::String(s.clone()),
                            Value::Bool(b) => ColumnValue::Bool(*b),
                            Value::Binary(b) => ColumnValue::Binary(b.clone()),
                            Value::FixedList(b) => ColumnValue::FixedList(b.clone()),
                            _ => ColumnValue::Null,
                        };
                        (k.clone(), cv)
                    })
                    .collect()
            })
            .collect();

        self.insert_column_rows_to_delta(&converted)
    }

    pub fn insert_column_rows_to_delta(
        &self,
        rows: &[HashMap<String, ColumnValue>],
    ) -> io::Result<Vec<u64>> {
        let ids = self.storage.insert_rows_to_delta(rows)?;

        // Auto-compact if delta file is too large (> 10MB or > 100K rows)
        const DELTA_SIZE_THRESHOLD: u64 = 10 * 1024 * 1024; // 10MB
        const DELTA_ROWS_THRESHOLD: usize = 100_000;

        if ids.len() >= DELTA_ROWS_THRESHOLD {
            *self.dirty.write() = true;
            return Ok(ids);
        }

        if ids.len() > 1 {
            let delta_path = Self::delta_path(&self.path);
            let delta_size = std::fs::metadata(&delta_path).map(|m| m.len()).unwrap_or(0);
            if delta_size > DELTA_SIZE_THRESHOLD {
                *self.dirty.write() = true;
            }
        }

        Ok(ids)
    }

    /// Get delta file path
    fn delta_path(base_path: &Path) -> std::path::PathBuf {
        let mut delta = base_path.to_path_buf();
        let name = delta.file_name().unwrap_or_default().to_string_lossy();
        delta.set_file_name(format!("{}.delta", name));
        delta
    }

    /// Check if delta file exists
    pub fn has_delta(&self) -> bool {
        self.storage.has_delta()
    }

    /// Compact delta into base file
    pub fn compact(&self) -> io::Result<()> {
        self.storage.compact()
    }

    /// Check if compaction is needed
    pub fn needs_compaction(&self) -> bool {
        self.storage.needs_compaction()
    }

    /// Get metadata without loading data
    pub fn metadata(&self) -> TableMetadata {
        TableMetadata {
            name: self
                .path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            row_count: *self.row_count.read(),
            schema: self.schema.read().clone(),
        }
    }

    /// Get row count
    pub fn row_count(&self) -> u64 {
        *self.row_count.read()
    }

    /// Get schema
    pub fn get_schema(&self) -> Vec<(String, DataType)> {
        self.schema.read().clone()
    }

    /// Get column names
    pub fn column_names(&self) -> Vec<String> {
        let mut names = vec!["_id".to_string()];
        names.extend(self.schema.read().iter().map(|(n, _)| n.clone()));
        names
    }

    /// Acquire global read lock for thread-safe concurrent reads.
    /// Multiple readers can hold the lock simultaneously.
    #[inline]
    pub fn read_lock(&self) -> parking_lot::RwLockReadGuard<()> {
        self.storage.read_lock()
    }

    /// Acquire global write lock for thread-safe writes.
    /// Only one writer can hold the lock; readers are blocked while held.
    #[inline]
    pub fn write_lock(&self) -> parking_lot::RwLockWriteGuard<()> {
        self.storage.write_lock()
    }

    // ========================================================================
    // Lazy Loading APIs
    // ========================================================================

    /// Load specific columns into cache (lazy load)
    /// Only loads columns that are not already cached
    pub fn load_columns(&self, column_names: &[&str]) -> io::Result<()> {
        let cached = self.cached_columns.read();
        let to_load: Vec<&str> = column_names
            .iter()
            .filter(|&name| !cached.contains_key(*name))
            .copied()
            .collect();
        drop(cached);

        if to_load.is_empty() {
            return Ok(());
        }

        // Read columns from storage
        let col_data = self.storage.read_columns(Some(&to_load), 0, None)?;

        // Convert and cache
        let schema = self.schema.read();
        let mut cached = self.cached_columns.write();

        for (name, data) in col_data {
            let dtype = schema
                .iter()
                .find(|(n, _)| n == &name)
                .map(|(_, dt)| dt.clone())
                .unwrap_or(DataType::String);

            let typed_col = column_data_to_typed_column(&data, dtype);
            cached.insert(name, typed_col);
        }

        Ok(())
    }

    /// Load all columns into cache
    pub fn load_all_columns(&self) -> io::Result<()> {
        let names: Vec<String> = self.schema.read().iter().map(|(n, _)| n.clone()).collect();
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        self.load_columns(&refs)
    }

    /// Get a cached column (returns None if not loaded)
    pub fn get_cached_column(&self, name: &str) -> Option<TypedColumn> {
        self.cached_columns.read().get(name).cloned()
    }

    /// Get column, loading if necessary
    pub fn get_column(&self, name: &str) -> io::Result<Option<TypedColumn>> {
        // Check cache first
        if let Some(col) = self.cached_columns.read().get(name).cloned() {
            return Ok(Some(col));
        }

        // Load from storage
        self.load_columns(&[name])?;
        Ok(self.cached_columns.read().get(name).cloned())
    }

    /// Release cached columns to free memory
    pub fn release_columns(&self, column_names: &[&str]) {
        let mut cached = self.cached_columns.write();
        for name in column_names {
            cached.remove(*name);
        }
    }

    /// Release all cached columns
    pub fn release_all_columns(&self) {
        self.cached_columns.write().clear();
    }

    /// Get memory usage of cached columns (approximate)
    pub fn cached_memory_bytes(&self) -> usize {
        let cached = self.cached_columns.read();
        let mut total = 0;
        for (_, col) in cached.iter() {
            total += match col {
                TypedColumn::Int64 { data, .. } => data.len() * 8,
                TypedColumn::Float64 { data, .. } => data.len() * 8,
                TypedColumn::String(arrow_col) => arrow_col.len() * 32, // Approximate: 32 bytes per string
                TypedColumn::Bool { data, .. } => data.len() / 8 + 1,
                TypedColumn::Mixed { data, .. } => data.len() * 64, // Approximate
            };
        }
        total
    }

    #[inline]
    fn invalidate_read_caches(&self) {
        self.cached_columns.write().clear();
        self.dict_cache.write().clear();
        self.first_string_row_id_cache.write().clear();
        invalidate_global_dict_cache(self.path());
    }

    fn get_or_build_first_string_row_id_cache(
        &self,
        col_name: &str,
    ) -> io::Result<Option<Arc<FirstStringRowIdCache>>> {
        {
            let cache = self.first_string_row_id_cache.read();
            if let Some(cached) = cache.get(col_name) {
                return Ok(Some(Arc::clone(cached)));
            }
        }

        let built = match self.storage.build_first_string_row_id_cache(col_name)? {
            Some(cache) => Arc::new(cache),
            None => return Ok(None),
        };

        let mut caches = self.first_string_row_id_cache.write();
        if let Some(existing) = caches.get(col_name) {
            return Ok(Some(Arc::clone(existing)));
        }
        caches.insert(col_name.to_string(), Arc::clone(&built));
        Ok(Some(built))
    }

    #[inline]
    pub fn first_row_id_for_string_eq(
        &self,
        col_name: &str,
        value: &str,
    ) -> io::Result<Option<u64>> {
        Ok(self
            .get_or_build_first_string_row_id_cache(col_name)?
            .and_then(|cache| cache.get(value).copied()))
    }

    // ========================================================================
    // Write APIs
    // ========================================================================

    /// Insert rows (updates cache and marks dirty)
    /// Optimized with parallel conversion for large batches
    pub fn insert_rows(&self, rows: &[HashMap<String, Value>]) -> io::Result<Vec<u64>> {
        use crate::storage::on_demand::ColumnValue;
        use rayon::prelude::*;

        if rows.is_empty() {
            return Ok(Vec::new());
        }

        // Convert to ColumnValue format - use parallel for large batches
        let converted: Vec<HashMap<String, ColumnValue>> = if rows.len() > 1000 {
            rows.par_iter()
                .map(|row| {
                    row.iter()
                        .map(|(k, v)| {
                            let cv = match v {
                                Value::Int64(i) => ColumnValue::Int64(*i),
                                Value::Int32(i) => ColumnValue::Int64(*i as i64),
                                Value::Float64(f) => ColumnValue::Float64(*f),
                                Value::Float32(f) => ColumnValue::Float64(*f as f64),
                                Value::String(s) => ColumnValue::String(s.clone()),
                                Value::Bool(b) => ColumnValue::Bool(*b),
                                Value::Binary(b) => ColumnValue::Binary(b.clone()),
                                Value::FixedList(b) => ColumnValue::FixedList(b.clone()),
                                Value::Null => ColumnValue::Null,
                                _ => ColumnValue::String(
                                    serde_json::to_string(v).unwrap_or_default(),
                                ),
                            };
                            (k.clone(), cv)
                        })
                        .collect()
                })
                .collect()
        } else {
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|(k, v)| {
                            let cv = match v {
                                Value::Int64(i) => ColumnValue::Int64(*i),
                                Value::Int32(i) => ColumnValue::Int64(*i as i64),
                                Value::Float64(f) => ColumnValue::Float64(*f),
                                Value::Float32(f) => ColumnValue::Float64(*f as f64),
                                Value::String(s) => ColumnValue::String(s.clone()),
                                Value::Bool(b) => ColumnValue::Bool(*b),
                                Value::Binary(b) => ColumnValue::Binary(b.clone()),
                                Value::FixedList(b) => ColumnValue::FixedList(b.clone()),
                                Value::Null => ColumnValue::Null,
                                _ => ColumnValue::String(
                                    serde_json::to_string(v).unwrap_or_default(),
                                ),
                            };
                            (k.clone(), cv)
                        })
                        .collect()
                })
                .collect()
        };

        // Insert into storage
        let ids = self.storage.insert_rows(&converted)?;

        // Update schema if new columns (only check first row for perf)
        {
            let mut schema = self.schema.write();
            if let Some(row) = rows.first() {
                for (k, v) in row {
                    if k != "_id" && !schema.iter().any(|(n, _)| n == k) {
                        schema.push((k.clone(), v.data_type()));
                    }
                }
            }
        }

        // Update row count
        *self.row_count.write() += rows.len() as u64;

        // Invalidate cache (data changed)
        self.invalidate_read_caches();
        *self.dirty.write() = true;

        Ok(ids)
    }

    /// Insert typed columns directly - bypasses row-by-row conversion
    /// Much faster for bulk inserts with homogeneous columnar data
    pub fn insert_typed(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        // Delegate to storage
        let ids = self.storage.insert_typed(
            int_columns.clone(),
            float_columns.clone(),
            string_columns.clone(),
            binary_columns.clone(),
            bool_columns.clone(),
        )?;

        // Update schema if new columns
        {
            let mut schema = self.schema.write();
            for name in int_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Int64));
                }
            }
            for name in float_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Float64));
                }
            }
            for name in string_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::String));
                }
            }
            for name in binary_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Binary));
                }
            }
            for name in bool_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Bool));
                }
            }
        }

        // Update row count
        *self.row_count.write() += ids.len() as u64;

        // Invalidate cache (data changed)
        self.invalidate_read_caches();
        *self.dirty.write() = true;

        Ok(ids)
    }

    /// Insert typed columns with null tracking - supports NULL values
    pub fn insert_typed_with_nulls(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
        null_positions: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        // Delegate to storage with null tracking
        let ids = self.storage.insert_typed_with_nulls(
            int_columns.clone(),
            float_columns.clone(),
            string_columns.clone(),
            binary_columns.clone(),
            bool_columns.clone(),
            null_positions,
        )?;

        // Update schema if new columns
        {
            let mut schema = self.schema.write();
            for name in int_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Int64));
                }
            }
            for name in float_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Float64));
                }
            }
            for name in string_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::String));
                }
            }
            for name in binary_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Binary));
                }
            }
            for name in bool_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Bool));
                }
            }
        }

        // Update row count
        *self.row_count.write() += ids.len() as u64;

        // Invalidate cache (data changed)
        self.invalidate_read_caches();
        *self.dirty.write() = true;

        Ok(ids)
    }

    /// Insert typed columns with null tracking — full version supporting FixedList columns
    pub fn insert_typed_with_nulls_full(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        fixedlist_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
        null_positions: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        let ids = self.storage.insert_typed_with_nulls_full(
            int_columns.clone(),
            float_columns.clone(),
            string_columns.clone(),
            binary_columns.clone(),
            fixedlist_columns.clone(),
            bool_columns.clone(),
            null_positions,
        )?;

        {
            let mut schema = self.schema.write();
            for name in int_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Int64));
                }
            }
            for name in float_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Float64));
                }
            }
            for name in string_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::String));
                }
            }
            for name in binary_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Binary));
                }
            }
            for name in fixedlist_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Binary));
                }
            }
            for name in bool_columns.keys() {
                if !schema.iter().any(|(n, _)| n == name) {
                    schema.push((name.clone(), crate::data::DataType::Bool));
                }
            }
        }

        *self.row_count.write() += ids.len() as u64;
        self.invalidate_read_caches();
        *self.dirty.write() = true;
        Ok(ids)
    }

    /// Save changes to disk
    pub fn save(&self) -> io::Result<()> {
        if self.storage.pending_v4_in_memory_rows() > 0
            && self.storage.spill_pending_v4_rows_to_delta()?
        {
            if self.has_pending_deltas() {
                self.storage.save_delta_store()?;
            }
            self.invalidate_read_caches();
            *self.dirty.write() = false;
            return Ok(());
        }
        self.storage.save()?;
        *self.dirty.write() = false;
        Ok(())
    }

    /// Force a full base-file rewrite instead of using append-only spill fast paths.
    pub fn save_full(&self) -> io::Result<()> {
        self.storage.save_full()?;
        *self.dirty.write() = false;
        Ok(())
    }

    /// Save after deletion-only operations.
    /// For uncompressed V4 files: O(num_RGs) in-place deletion vector writes instead of
    /// a full O(all_data) rewrite. For compressed files, falls back to full rewrite.
    pub fn save_delete_only(&self) -> io::Result<()> {
        self.storage.save_delete_only()?;
        *self.dirty.write() = false;
        Ok(())
    }

    // ========================================================================
    // DeltaStore methods (Phase 4.5)
    // ========================================================================

    /// Record cell-level updates for a row in the delta store.
    /// This avoids delete+insert for UPDATE operations.
    pub fn delta_update_row(
        &self,
        row_id: u64,
        values: &std::collections::HashMap<String, crate::data::Value>,
    ) {
        self.storage.delta_update_row(row_id, values);
        self.invalidate_read_caches();
    }

    /// Record a row deletion in DeltaStore without rewriting the base file.
    pub fn delta_delete_row(&self, row_id: u64) -> io::Result<bool> {
        let result = self.storage.delta_delete_row(row_id)?;
        if result {
            self.invalidate_read_caches();
            *self.dirty.write() = true;
        }
        Ok(result)
    }

    /// Mark an unflushed V4 memtable row deleted without creating DeltaStore work.
    pub fn delete_pending_v4_in_memory_row(&self, row_id: u64) -> bool {
        let result = self.storage.delete_pending_v4_in_memory_row(row_id);
        if result {
            self.invalidate_read_caches();
            *self.dirty.write() = true;
        }
        result
    }

    /// Record a full-row replacement in DeltaStore for an existing row.
    pub fn delta_update_existing_row(
        &self,
        row_id: u64,
        values: &std::collections::HashMap<String, crate::data::Value>,
    ) -> io::Result<bool> {
        let result = self.storage.delta_update_existing_row(row_id, values)?;
        if result {
            self.invalidate_read_caches();
            *self.dirty.write() = true;
        }
        Ok(result)
    }

    /// Batch update multiple rows in a single lock acquisition.
    pub fn delta_batch_update_rows(&self, batch: &[(u64, &str, crate::data::Value)]) {
        self.storage.delta_batch_update_rows(batch);
        self.invalidate_read_caches();
    }

    /// Scan a numeric column for rows in [low, high] and return matching row IDs.
    pub fn scan_numeric_range_with_ids(
        &self,
        col_name: &str,
        low: f64,
        high: f64,
    ) -> io::Result<Option<Vec<u64>>> {
        self.storage
            .scan_numeric_range_with_ids(col_name, low, high)
    }

    /// Single-pass scan+write: find WHERE column matches and overwrite SET column in-place.
    /// Returns Some(count) if successful, None if fallback to DeltaStore needed.
    pub fn scan_and_update_inplace(
        &self,
        where_col: &str,
        low: f64,
        high: f64,
        set_col: &str,
        new_value_bytes: &[u8; 8],
    ) -> io::Result<Option<i64>> {
        self.storage
            .scan_and_update_inplace(where_col, low, high, set_col, new_value_bytes)
    }

    /// O(1) in-place overwrite for `UPDATE ... SET numeric_col = literal WHERE _id = N`.
    /// Returns Some((count, physically_written)) when the fast path handled the
    /// statement, or None to fall back.
    pub fn update_by_id_inplace(
        &self,
        id: u64,
        set_col: &str,
        new_value_bytes: &[u8; 8],
    ) -> io::Result<Option<(i64, bool)>> {
        let result = self
            .storage
            .update_by_id_inplace(id, set_col, new_value_bytes)?;
        if matches!(result, Some((_, true))) {
            self.storage.mark_sync_pending();
        }
        Ok(result)
    }

    /// O(1) existence check for a V4 `_id` using row-group id/deletion sections.
    pub fn row_id_active_rcix(&self, id: u64) -> io::Result<Option<bool>> {
        self.storage.row_id_active_rcix(id)
    }

    /// Save the delta store to disk.
    pub fn save_delta_store(&self) -> io::Result<()> {
        self.storage.save_delta_store()
    }

    /// Check if the delta store has pending changes.
    pub fn has_pending_deltas(&self) -> bool {
        self.storage.has_pending_deltas()
    }

    /// Check whether pending DeltaStore updates touch a specific column.
    pub fn pending_delta_updates_column(&self, column_name: &str) -> bool {
        self.storage.delta_updates_column(column_name)
    }

    /// Count pending DeltaStore deletes.
    pub fn pending_delta_delete_count(&self) -> usize {
        self.storage.delta_delete_count()
    }

    /// Return row IDs whose pending DeltaStore update sets `column_name` to `value`.
    pub fn pending_delta_string_update_matches(&self, column_name: &str, value: &str) -> Vec<u64> {
        self.storage
            .delta_rows_with_string_update(column_name, value)
    }

    /// Return row IDs from the committed append-only `.delta` file where a string column matches.
    pub fn delta_string_match_ids(&self, column_name: &str, value: &str) -> io::Result<Vec<u64>> {
        self.storage.delta_string_match_ids(column_name, value)
    }

    /// Get the file path for this backend
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Check if this backend is in V4 mmap-only mode (no in-memory column data).
    /// When true, all fast paths that read from in-memory columns should be skipped.
    pub fn is_mmap_only(&self) -> bool {
        self.storage.is_v4_format() && !self.storage.has_v4_in_memory_data()
    }

    /// Check if delta compaction is needed.
    pub fn needs_delta_compaction(&self) -> bool {
        self.storage.needs_delta_compaction()
    }

    /// Compact deltas into the base file (merge updates, apply deletes, rewrite).
    pub fn compact_deltas(&self) -> io::Result<()> {
        self.storage.compact_deltas()?;
        self.invalidate_read_caches();
        Ok(())
    }

    /// Explicitly sync data to disk (fsync)
    ///
    /// This ensures all buffered data is written to persistent storage.
    /// For Safe/Max durability levels, save() automatically calls fsync.
    /// For Fast durability, use this method when you need explicit durability.
    pub fn sync(&self) -> io::Result<()> {
        self.storage.sync()
    }

    /// Get the current durability level
    pub fn durability(&self) -> super::DurabilityLevel {
        self.storage.durability()
    }

    /// Set auto-flush thresholds
    ///
    /// When either threshold is exceeded during writes, data is automatically
    /// written to file. Set to 0 to disable the respective threshold.
    pub fn set_auto_flush(&self, rows: u64, bytes: u64) {
        self.storage.set_auto_flush(rows, bytes);
    }

    /// Get current auto-flush configuration
    pub fn get_auto_flush(&self) -> (u64, u64) {
        self.storage.get_auto_flush()
    }

    /// Estimate current in-memory data size in bytes
    pub fn estimate_memory_bytes(&self) -> u64 {
        self.storage.estimate_memory_bytes()
    }

    /// Get the current compression type for this table.
    pub fn compression(&self) -> CompressionType {
        self.storage.compression()
    }

    /// Set compression type. Only effective on empty tables (row_count == 0).
    /// Returns Ok(true) if applied, Ok(false) if table is non-empty (no-op).
    pub fn set_compression(&self, comp: CompressionType) -> io::Result<bool> {
        self.storage.set_compression(comp)
    }

    /// Check if there are unsaved changes
    pub fn is_dirty(&self) -> bool {
        *self.dirty.read()
    }

    /// Flush and close - releases mmap and file handle
    /// IMPORTANT: On Windows, this must be called before temp directory cleanup
    pub fn close(&self) -> io::Result<()> {
        if self.is_dirty() {
            self.save()?;
        }
        // Release mmap and file handle (critical for Windows)
        self.storage.close()
    }

    // ========================================================================
    // Delete/Update APIs
    // ========================================================================

    /// Delete a row by ID (soft delete)
    pub fn delete(&self, id: u64) -> bool {
        let result = self.storage.delete(id);
        if result {
            self.invalidate_read_caches();
            *self.dirty.write() = true;
        }
        result
    }

    /// Delete multiple rows by IDs (soft delete)
    pub fn delete_batch(&self, ids: &[u64]) -> bool {
        let result = self.storage.delete_batch(ids);
        if result {
            self.invalidate_read_caches();
            *self.dirty.write() = true;
        }
        result
    }

    /// Check if a row exists and is not deleted
    pub fn exists(&self, id: u64) -> bool {
        self.storage.exists(id)
    }

    /// Get active (non-deleted) row count
    pub fn active_row_count(&self) -> u64 {
        self.storage.active_row_count()
    }

    /// Get the next `_id` that will be assigned to a new row.
    pub fn next_id_value(&self) -> u64 {
        self.storage.next_id_value()
    }

    /// Fast path: Get base table row count only (no delta scan)
    /// Use this for COUNT(*) without WHERE clause - O(1) lock-free read
    pub fn base_row_count(&self) -> u64 {
        self.storage.base_row_count()
    }

    /// Rows buffered in the V4 in-memory append area and not yet persisted.
    pub fn pending_v4_in_memory_rows(&self) -> usize {
        self.storage.pending_v4_in_memory_rows()
    }

    /// Replace a row (delete + insert new)
    pub fn replace(&self, id: u64, data: &HashMap<String, Value>) -> io::Result<bool> {
        use crate::storage::on_demand::ColumnValue;

        // Convert Value to ColumnValue
        let cv_data: HashMap<String, ColumnValue> = data
            .iter()
            .map(|(k, v)| {
                let cv = match v {
                    Value::Int64(i) => ColumnValue::Int64(*i),
                    Value::Int32(i) => ColumnValue::Int64(*i as i64),
                    Value::Float64(f) => ColumnValue::Float64(*f),
                    Value::Float32(f) => ColumnValue::Float64(*f as f64),
                    Value::String(s) => ColumnValue::String(s.clone()),
                    Value::Bool(b) => ColumnValue::Bool(*b),
                    Value::Binary(b) => ColumnValue::Binary(b.clone()),
                    Value::Null => ColumnValue::Null,
                    _ => ColumnValue::String(serde_json::to_string(v).unwrap_or_default()),
                };
                (k.clone(), cv)
            })
            .collect();

        let result = self.storage.replace(id, &cv_data)?;
        if result {
            *self.dirty.write() = true;
            self.invalidate_read_caches();
            // Update row count
            *self.row_count.write() = self.storage.row_count();
        }
        Ok(result)
    }

    // ========================================================================
    // Schema Operations
    // ========================================================================

    /// Add a column to the schema and storage with padding for existing rows
    pub fn add_column(&self, name: &str, dtype: DataType) -> io::Result<()> {
        // Check if column already exists
        {
            let schema = self.schema.read();
            if schema.iter().any(|(n, _)| n == name) {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("Column '{}' already exists", name),
                ));
            }
        }

        // Use the underlying storage's add_column_with_padding for proper data alignment
        self.storage.add_column_with_padding(name, dtype)?;

        // Update our schema cache
        let mut schema = self.schema.write();
        schema.push((name.to_string(), dtype));
        self.invalidate_read_caches();
        *self.dirty.write() = true;
        Ok(())
    }

    /// Drop a column from the schema and storage
    pub fn drop_column(&self, name: &str) -> io::Result<()> {
        // Drop from underlying storage (removes schema, data, nulls, index)
        self.storage.drop_column(name)?;

        // Also update the cached schema
        let mut schema = self.schema.write();
        let pos = schema.iter().position(|(n, _)| n == name);
        if let Some(idx) = pos {
            schema.remove(idx);
        }

        self.invalidate_read_caches();
        *self.dirty.write() = true;
        Ok(())
    }

    /// Rename a column
    pub fn rename_column(&self, old_name: &str, new_name: &str) -> io::Result<()> {
        let mut schema = self.schema.write();
        let mut found = false;
        for (name, _) in schema.iter_mut() {
            if name == old_name {
                *name = new_name.to_string();
                found = true;
                break;
            }
        }
        if !found {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Column '{}' not found", old_name),
            ));
        }
        *self.dirty.write() = true;
        drop(schema);

        // Also update the underlying OnDemandStorage schema so that
        // save() and update_v4_footer_schema() persist the new name.
        self.storage.rename_column_in_schema(old_name, new_name);
        self.invalidate_read_caches();
        Ok(())
    }

    /// List all column names
    pub fn list_columns(&self) -> Vec<String> {
        self.schema.read().iter().map(|(n, _)| n.clone()).collect()
    }

    /// Get column data type
    pub fn get_column_type(&self, name: &str) -> Option<DataType> {
        self.schema
            .read()
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, dt)| dt.clone())
    }

    // ========================================================================
    // True On-Demand Column Projection APIs
    // ========================================================================

    /// Zero-copy parallel TopK for a Binary vector column.
    /// Returns `Some(topk)` sorted ascending, or `None` to fall back to the Arrow path.
    pub fn topk_binary_direct(
        &self,
        col_name: &str,
        computer: &crate::query::vector_ops::DistanceComputer,
        k: usize,
    ) -> io::Result<Option<Vec<(usize, f32)>>> {
        self.storage.topk_binary_direct(col_name, computer, k)
    }

    /// Zero-copy parallel TopK for a FixedList column.
    /// Runs directly on OS mmap — no Arrow construction, no memcpy.
    /// Returns `Some(topk)` sorted ascending, or `None` to fall back to the Arrow path.
    pub fn topk_fixedlist_direct(
        &self,
        col_name: &str,
        computer: &crate::query::vector_ops::DistanceComputer,
        k: usize,
    ) -> io::Result<Option<Vec<(usize, f32)>>> {
        self.storage.topk_fixedlist_direct(col_name, computer, k)
    }

    /// Batch parallel TopK for a FixedList column — N queries in one call.
    /// Loads `scan_buf` once and runs all queries in parallel; much faster than N sequential calls.
    pub fn batch_topk_fixedlist_direct(
        &self,
        col_name: &str,
        queries: &[f32],
        n_queries: usize,
        k: usize,
        metric: crate::query::vector_ops::DistanceMetric,
    ) -> io::Result<Option<Vec<Vec<(usize, f32)>>>> {
        self.storage
            .batch_topk_fixedlist_direct(col_name, queries, n_queries, k, metric)
    }

    /// Batch parallel TopK for a Binary vector column — N queries in one call.
    pub fn batch_topk_binary_direct(
        &self,
        col_name: &str,
        queries: &[f32],
        n_queries: usize,
        k: usize,
        metric: crate::query::vector_ops::DistanceMetric,
    ) -> io::Result<Option<Vec<Vec<(usize, f32)>>>> {
        self.storage
            .batch_topk_binary_direct(col_name, queries, n_queries, k, metric)
    }

    /// Read columns to Arrow with dictionary encoding for low-cardinality string columns.
    /// Use this for GROUP BY queries where DictionaryArray accelerates aggregation.
    pub fn read_columns_to_arrow_dict(
        &self,
        column_names: Option<&[&str]>,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        if let Ok(batch) = self.storage.to_arrow_batch_dict(
            column_names,
            column_names.map(|c| c.contains(&"_id")).unwrap_or(true),
        ) {
            if batch.num_rows() > 0 || batch.num_columns() > 0 {
                return Ok(batch);
            }
        }
        // Fallback to normal path
        self.read_columns_to_arrow(column_names, 0, None)
    }

    /// Fast path for FTS backfill on persisted V4 tables.
    /// Returns None when pending/delta rows require the general Arrow read path.
    pub(crate) fn read_fts_string_columns_mmap(
        &self,
        column_names: &[String],
    ) -> io::Result<Option<(Vec<u32>, Vec<(String, ColumnData)>)>> {
        let base_rows = self.base_row_count();
        let has_delta =
            self.has_delta() || self.row_count() > base_rows || self.active_row_count() > base_rows;
        if has_delta || self.storage.has_v4_in_memory_data() {
            return Ok(None);
        }
        self.storage.read_fts_string_columns_mmap(column_names)
    }

    /// Read specific columns directly to Arrow RecordBatch (TRUE on-demand read)
    ///
    /// This method bypasses ColumnTable and reads only the requested columns
    /// from storage, converting directly to Arrow format.
    ///
    /// Features:
    /// - Column projection: only reads requested columns from disk
    /// - Row range: supports start_row and row_count for partial reads
    /// - Caching: caches full column reads for repeated access
    ///
    /// # Arguments
    /// * `column_names` - Columns to read (None = all columns)
    /// * `start_row` - Starting row index
    /// * `row_count` - Number of rows to read (None = all)
    pub fn read_columns_to_arrow(
        &self,
        column_names: Option<&[&str]>,
        start_row: usize,
        row_count: Option<usize>,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        let base_rows = self.base_row_count();
        let has_delta =
            self.has_delta() || self.row_count() > base_rows || self.active_row_count() > base_rows;
        let use_cache = start_row == 0 && row_count.is_none() && !has_delta;

        // OPTIMIZATION: V4 fast path — build Arrow directly from in-memory or mmap columns
        // Bypasses read_columns→HashMap→get_null_mask→Vec<bool> pipeline entirely
        if use_cache {
            if let Ok(batch) = self.storage.to_arrow_batch(
                column_names,
                column_names.map(|c| c.contains(&"_id")).unwrap_or(true),
            ) {
                if batch.num_rows() > 0 || batch.num_columns() > 0 {
                    return Ok(batch);
                }
            }
        }

        // OPTIMIZATION: V4 fast path for LIMIT reads (start_row=0, row_count=Some)
        // Use to_arrow_batch_with_limit which leverages RCIX for O(1) column seeks —
        // reads only the needed rows instead of loading the full table.
        if start_row == 0 && row_count.is_some() && !has_delta {
            let limit = row_count.unwrap();
            if let Ok(batch) = self.storage.to_arrow_batch_with_limit(
                column_names,
                column_names.map(|c| c.contains(&"_id")).unwrap_or(true),
                limit,
            ) {
                if batch.num_rows() > 0 || batch.num_columns() > 0 {
                    return Ok(batch);
                }
            }
        }

        // Handle SELECT _id ONLY case FIRST (before reading columns)
        if let Some(cols) = column_names {
            if cols.len() == 1 && cols[0] == "_id" {
                // Only _id requested - return batch with just _id column
                let ids = self.storage.read_ids(start_row, row_count)?;
                let fields = vec![Field::new("_id", ArrowDataType::Int64, false)];
                // OPTIMIZATION: Direct transmute from Vec<u64> to Vec<i64>
                let ids_i64: Vec<i64> = unsafe {
                    let mut ids = std::mem::ManuallyDrop::new(ids);
                    Vec::from_raw_parts(ids.as_mut_ptr() as *mut i64, ids.len(), ids.capacity())
                };
                let arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(ids_i64))];
                let schema = Arc::new(Schema::new(fields));
                return arrow::record_batch::RecordBatch::try_new(schema, arrays)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
            }
        }

        let storage_column_names: Option<Vec<&str>> =
            column_names.map(|cols| cols.iter().copied().filter(|name| *name != "_id").collect());
        let storage_column_refs = storage_column_names.as_ref().map(|cols| cols.as_slice());

        // Read columns from storage (only the requested physical columns!)
        let mut col_data = self
            .storage
            .read_columns(storage_column_refs, start_row, row_count)?;

        if col_data.is_empty() {
            // Return empty batch with schema (including _id if requested)
            let schema = self.schema.read();
            let include_id = column_names
                .map(|cols| cols.contains(&"_id"))
                .unwrap_or(true);
            let mut fields: Vec<Field> = Vec::new();

            if include_id {
                fields.push(Field::new("_id", ArrowDataType::Int64, false));
            }

            for (name, dt) in schema.iter() {
                if column_names
                    .map(|cols| cols.contains(&name.as_str()))
                    .unwrap_or(true)
                {
                    let arrow_dt = match dt {
                        DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                            ArrowDataType::Int64
                        }
                        DataType::Float64 | DataType::Float32 => ArrowDataType::Float64,
                        DataType::String => ArrowDataType::Utf8,
                        DataType::Bool => ArrowDataType::Boolean,
                        _ => ArrowDataType::Utf8,
                    };
                    fields.push(Field::new(name, arrow_dt, true));
                }
            }
            let schema = Arc::new(Schema::new(fields));
            return Ok(arrow::record_batch::RecordBatch::new_empty(schema));
        }

        // Build Arrow arrays from ColumnData
        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        // Always include _id as the first column (unless explicitly excluded)
        let include_id = column_names
            .map(|cols| cols.contains(&"_id"))
            .unwrap_or(true);
        let expected_row_count: usize;
        if include_id {
            let ids = self.storage.read_ids(start_row, row_count)?;
            expected_row_count = ids.len();
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            // OPTIMIZATION: Direct transmute from Vec<u64> to Vec<i64> - same memory layout
            let ids_i64: Vec<i64> = unsafe {
                let mut ids = std::mem::ManuallyDrop::new(ids);
                Vec::from_raw_parts(ids.as_mut_ptr() as *mut i64, ids.len(), ids.capacity())
            };
            arrays.push(Arc::new(Int64Array::from(ids_i64)));
        } else {
            // If no _id, get row count from any column
            expected_row_count = col_data.values().next().map(|d| d.len()).unwrap_or(0);
        }

        // Determine column order from schema (or from column_names if specified)
        let col_order: Vec<String> = if let Some(names) = column_names {
            names
                .iter()
                .filter(|&s| *s != "_id") // Skip _id, already handled
                .map(|s| s.to_string())
                .collect()
        } else {
            schema.iter().map(|(n, _)| n.clone()).collect()
        };

        // Get actual start_row and row_count for null mask lookup
        let actual_start = start_row;
        let actual_count = expected_row_count;

        for col_name in &col_order {
            // Get null mask for this column
            let null_mask = self
                .storage
                .get_null_mask(col_name, actual_start, actual_count);
            let has_nulls = null_mask.iter().any(|&is_null| is_null);

            // Use remove() to take ownership and avoid clone
            if let Some(data) = col_data.remove(col_name) {
                let (arrow_dt, array): (ArrowDataType, ArrayRef) = match data {
                    ColumnData::Int64(values) => {
                        // Apply null mask if there are any nulls
                        if has_nulls {
                            let with_nulls: Vec<Option<i64>> = values
                                .into_iter()
                                .enumerate()
                                .map(|(i, v)| {
                                    if i < null_mask.len() && null_mask[i] {
                                        None
                                    } else {
                                        Some(v)
                                    }
                                })
                                .collect();
                            (ArrowDataType::Int64, Arc::new(Int64Array::from(with_nulls)))
                        } else if values.len() < expected_row_count {
                            let mut padded: Vec<Option<i64>> =
                                values.into_iter().map(Some).collect();
                            padded.extend(
                                std::iter::repeat(None).take(expected_row_count - padded.len()),
                            );
                            (ArrowDataType::Int64, Arc::new(Int64Array::from(padded)))
                        } else if values.len() > expected_row_count {
                            let truncated: Vec<i64> =
                                values.into_iter().take(expected_row_count).collect();
                            (ArrowDataType::Int64, Arc::new(Int64Array::from(truncated)))
                        } else {
                            (ArrowDataType::Int64, Arc::new(Int64Array::from(values)))
                        }
                    }
                    ColumnData::Float64(values) => {
                        if has_nulls {
                            let with_nulls: Vec<Option<f64>> = values
                                .into_iter()
                                .enumerate()
                                .map(|(i, v)| {
                                    if i < null_mask.len() && null_mask[i] {
                                        None
                                    } else {
                                        Some(v)
                                    }
                                })
                                .collect();
                            (
                                ArrowDataType::Float64,
                                Arc::new(Float64Array::from(with_nulls)),
                            )
                        } else if values.len() < expected_row_count {
                            let mut padded: Vec<Option<f64>> =
                                values.into_iter().map(Some).collect();
                            padded.extend(
                                std::iter::repeat(None).take(expected_row_count - padded.len()),
                            );
                            (ArrowDataType::Float64, Arc::new(Float64Array::from(padded)))
                        } else if values.len() > expected_row_count {
                            let truncated: Vec<f64> =
                                values.into_iter().take(expected_row_count).collect();
                            (
                                ArrowDataType::Float64,
                                Arc::new(Float64Array::from(truncated)),
                            )
                        } else {
                            (ArrowDataType::Float64, Arc::new(Float64Array::from(values)))
                        }
                    }
                    ColumnData::String {
                        offsets,
                        data: bytes,
                    } => {
                        // Build StringArray with null mask support
                        let count = offsets.len().saturating_sub(1);

                        // Always use the path that supports nulls when has_nulls is true
                        let strings: Vec<Option<String>> = (0..count.min(expected_row_count))
                            .map(|i| {
                                // Check null mask first
                                if has_nulls && i < null_mask.len() && null_mask[i] {
                                    return None;
                                }
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&bytes[start..end])
                                    .ok()
                                    .map(|s| s.to_string())
                            })
                            .collect();

                        if strings.len() < expected_row_count {
                            let mut owned = strings;
                            owned.extend(
                                std::iter::repeat(None).take(expected_row_count - owned.len()),
                            );
                            (ArrowDataType::Utf8, Arc::new(StringArray::from(owned)))
                        } else {
                            (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                        }
                    }
                    ColumnData::Bool { data: packed, len } => {
                        let mut bools: Vec<Option<bool>> = (0..len)
                            .map(|i| {
                                // Check null mask first
                                if has_nulls && i < null_mask.len() && null_mask[i] {
                                    return None;
                                }
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(
                                    byte_idx < packed.len()
                                        && (packed[byte_idx] >> bit_idx) & 1 == 1,
                                )
                            })
                            .collect();
                        if bools.len() < expected_row_count {
                            bools.extend(
                                std::iter::repeat(None).take(expected_row_count - bools.len()),
                            );
                        } else if bools.len() > expected_row_count {
                            bools.truncate(expected_row_count);
                        }
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    }
                    ColumnData::Binary {
                        offsets,
                        data: bytes,
                    } => {
                        use arrow::array::BinaryArray;
                        let count = offsets.len().saturating_sub(1);
                        let binary_data: Vec<Option<&[u8]>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&bytes[start..end] as &[u8])
                            })
                            .collect();
                        if binary_data.len() < expected_row_count {
                            // Need owned data for padding
                            let mut owned: Vec<Option<Vec<u8>>> = binary_data
                                .into_iter()
                                .map(|b| b.map(|s| s.to_vec()))
                                .collect();
                            owned.extend(
                                std::iter::repeat(None).take(expected_row_count - owned.len()),
                            );
                            let refs: Vec<Option<&[u8]>> = owned
                                .iter()
                                .map(|o| o.as_ref().map(|v| v.as_slice()))
                                .collect();
                            (ArrowDataType::Binary, Arc::new(BinaryArray::from(refs)))
                        } else if binary_data.len() > expected_row_count {
                            // Truncate to expected row count
                            let truncated: Vec<Option<&[u8]>> =
                                binary_data.into_iter().take(expected_row_count).collect();
                            (
                                ArrowDataType::Binary,
                                Arc::new(BinaryArray::from(truncated)),
                            )
                        } else {
                            (
                                ArrowDataType::Binary,
                                Arc::new(BinaryArray::from(binary_data)),
                            )
                        }
                    }
                    ColumnData::StringDict {
                        indices,
                        dict_offsets,
                        dict_data,
                    } => {
                        // OPTIMIZATION: Use Arrow DictionaryArray to preserve dictionary encoding
                        // This avoids string decoding and allows executor to use indices directly
                        use arrow::array::{DictionaryArray, UInt32Array};
                        use arrow::datatypes::UInt32Type;

                        // Build dictionary values (unique strings)
                        let dict_count = dict_offsets.len().saturating_sub(1);
                        let dict_strings: Vec<Option<&str>> = (0..dict_count)
                            .map(|i| {
                                let start = dict_offsets[i] as usize;
                                let end = dict_offsets[i + 1] as usize;
                                std::str::from_utf8(&dict_data[start..end]).ok()
                            })
                            .collect();
                        let values = StringArray::from(dict_strings);

                        // Convert indices (0 = NULL, 1+ = dict index)
                        // Arrow DictionaryArray uses 0-based indices, NULL is separate
                        // Truncate or pad indices to match expected_row_count
                        let keys: Vec<Option<u32>> = if indices.len() > expected_row_count {
                            indices
                                .iter()
                                .take(expected_row_count)
                                .map(|&idx| if idx == 0 { None } else { Some(idx - 1) })
                                .collect()
                        } else if indices.len() < expected_row_count {
                            let mut keys: Vec<Option<u32>> = indices
                                .iter()
                                .map(|&idx| if idx == 0 { None } else { Some(idx - 1) })
                                .collect();
                            keys.extend(
                                std::iter::repeat(None).take(expected_row_count - keys.len()),
                            );
                            keys
                        } else {
                            indices
                                .iter()
                                .map(|&idx| if idx == 0 { None } else { Some(idx - 1) })
                                .collect()
                        };
                        let keys_array = UInt32Array::from(keys);

                        // Create DictionaryArray
                        let dict_array =
                            DictionaryArray::<UInt32Type>::try_new(keys_array, Arc::new(values))
                                .map_err(|e| {
                                    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
                                })?;

                        (
                            ArrowDataType::Dictionary(
                                Box::new(ArrowDataType::UInt32),
                                Box::new(ArrowDataType::Utf8),
                            ),
                            Arc::new(dict_array) as ArrayRef,
                        )
                    }
                    ColumnData::FixedList { data, dim } => fixedlist_to_arrow_pair(&data, dim),
                    ColumnData::Float16List { data, dim } => float16list_to_arrow_pair(&data, dim),
                };

                fields.push(Field::new(col_name, arrow_dt, true));
                arrays.push(array);
            }
        }

        let schema = Arc::new(arrow::datatypes::Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Read an Arrow row window without forcing V4 mmap callers to rescan from row 0.
    ///
    /// This is intended for streaming table consumers such as CREATE INDEX. On pure
    /// persisted V4 tables it reads only the requested active-row window; otherwise it
    /// falls back to the general reader, preserving delta/in-memory semantics.
    pub fn read_columns_to_arrow_window(
        &self,
        column_names: Option<&[&str]>,
        start_row: usize,
        row_count: Option<usize>,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        let base_rows = self.base_row_count();
        let has_delta =
            self.has_delta() || self.row_count() > base_rows || self.active_row_count() > base_rows;
        if !has_delta {
            let include_id = column_names
                .map(|cols| cols.contains(&"_id"))
                .unwrap_or(true);
            if let Ok(Some(batch)) = self.storage.to_arrow_batch_mmap_range(
                column_names,
                include_id,
                start_row,
                row_count,
                false,
            ) {
                return Ok(batch);
            }
        }
        self.read_columns_to_arrow(column_names, start_row, row_count)
    }

    /// Read all columns to Arrow (convenience method)
    pub fn read_all_to_arrow(&self) -> io::Result<arrow::record_batch::RecordBatch> {
        self.read_columns_to_arrow(None, 0, None)
    }

    /// Read columns with predicate pushdown to Arrow
    /// Filters rows at storage level before converting to Arrow
    pub fn read_columns_filtered_to_arrow(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_op: &str,
        filter_value: f64,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        let (col_data, matching_indices) = self.storage.read_columns_filtered(
            column_names,
            filter_column,
            filter_op,
            filter_value,
        )?;

        if col_data.is_empty() || matching_indices.is_empty() {
            // Return empty batch with proper schema
            let schema = self.schema.read();
            let include_id = column_names
                .map(|cols| cols.contains(&"_id"))
                .unwrap_or(true);
            let mut fields: Vec<Field> = Vec::new();
            if include_id {
                fields.push(Field::new("_id", ArrowDataType::Int64, false));
            }
            for (name, dt) in schema.iter() {
                if column_names
                    .map(|cols| cols.contains(&name.as_str()))
                    .unwrap_or(true)
                {
                    let arrow_dt = match dt {
                        DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                            ArrowDataType::Int64
                        }
                        DataType::Float64 | DataType::Float32 => ArrowDataType::Float64,
                        DataType::String => ArrowDataType::Utf8,
                        DataType::Bool => ArrowDataType::Boolean,
                        _ => ArrowDataType::Utf8,
                    };
                    fields.push(Field::new(name, arrow_dt, true));
                }
            }
            let schema = Arc::new(Schema::new(fields));
            return Ok(arrow::record_batch::RecordBatch::new_empty(schema));
        }

        // Build Arrow arrays from filtered ColumnData
        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();
        let expected_row_count = matching_indices.len();

        // Include _id column with filtered indices
        let include_id = column_names
            .map(|cols| cols.contains(&"_id"))
            .unwrap_or(true);
        if include_id {
            // OPTIMIZED: Read only the IDs we need instead of all IDs
            let filtered_ids = self.storage.read_ids_by_indices(&matching_indices)?;
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(filtered_ids)));
        }

        // Determine column order
        let col_order: Vec<String> = if let Some(names) = column_names {
            names
                .iter()
                .filter(|&s| *s != "_id")
                .map(|s| s.to_string())
                .collect()
        } else {
            schema.iter().map(|(n, _)| n.clone()).collect()
        };

        for col_name in &col_order {
            if let Some(data) = col_data.get(col_name) {
                let (arrow_dt, array): (ArrowDataType, ArrayRef) = match data {
                    ColumnData::Int64(values) => (
                        ArrowDataType::Int64,
                        Arc::new(Int64Array::from_iter_values(values.iter().copied())),
                    ),
                    ColumnData::Float64(values) => (
                        ArrowDataType::Float64,
                        Arc::new(Float64Array::from_iter_values(values.iter().copied())),
                    ),
                    ColumnData::String {
                        offsets,
                        data: bytes,
                    } => {
                        let count = offsets.len().saturating_sub(1);
                        let strings: Vec<Option<&str>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&bytes[start..end]).ok()
                            })
                            .collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                    ColumnData::Bool { data: packed, len } => {
                        let bools: Vec<Option<bool>> = (0..*len)
                            .map(|i| {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(
                                    byte_idx < packed.len()
                                        && (packed[byte_idx] >> bit_idx) & 1 == 1,
                                )
                            })
                            .collect();
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    }
                    ColumnData::Binary {
                        offsets,
                        data: bytes,
                    } => {
                        use arrow::array::BinaryArray;
                        let count = offsets.len().saturating_sub(1);
                        let binary_data: Vec<Option<&[u8]>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&bytes[start..end] as &[u8])
                            })
                            .collect();
                        (
                            ArrowDataType::Binary,
                            Arc::new(BinaryArray::from(binary_data)),
                        )
                    }
                    ColumnData::StringDict {
                        indices,
                        dict_offsets,
                        dict_data,
                    } => {
                        let strings: Vec<Option<String>> = indices
                            .iter()
                            .map(|&idx| {
                                if idx == 0 {
                                    None
                                } else {
                                    let dict_idx = (idx - 1) as usize;
                                    if dict_idx + 1 < dict_offsets.len() {
                                        let start = dict_offsets[dict_idx] as usize;
                                        let end = dict_offsets[dict_idx + 1] as usize;
                                        std::str::from_utf8(&dict_data[start..end])
                                            .ok()
                                            .map(|s| s.to_string())
                                    } else {
                                        None
                                    }
                                }
                            })
                            .collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                    ColumnData::FixedList { data, dim } => fixedlist_to_arrow_pair(data, *dim),
                    ColumnData::Float16List { data, dim } => float16list_to_arrow_pair(data, *dim),
                };

                fields.push(Field::new(col_name, arrow_dt, true));
                arrays.push(array);
            }
        }

        let schema = Arc::new(arrow::datatypes::Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Read columns for specific row indices to Arrow (for late materialization)
    /// Only reads the specified rows from disk, reducing I/O for filtered queries
    pub fn read_columns_by_indices_to_arrow(
        &self,
        row_indices: &[usize],
        col_refs: Option<&[&str]>,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        if row_indices.is_empty() {
            return self.read_columns_to_arrow(None, 0, Some(0));
        }

        // V4: use mmap indexed extraction when available (faster than full-table read + take)
        if self.storage.is_v4_format() {
            if let Some(batch) = self
                .storage
                .extract_rows_by_indices_to_arrow(row_indices, col_refs)?
            {
                return self.apply_delta_overlay_to_batch(batch);
            }
            // Fallback (extraction returned None) — load full table
            let full_batch = self.read_columns_to_arrow(col_refs, 0, None)?;
            if full_batch.num_rows() == 0 {
                return Ok(full_batch);
            }
            let indices_arr = arrow::array::UInt32Array::from(
                row_indices.iter().map(|&i| i as u32).collect::<Vec<_>>(),
            );
            let taken_columns: Vec<ArrayRef> = full_batch
                .columns()
                .iter()
                .map(|col| {
                    arrow::compute::take(col.as_ref(), &indices_arr, None)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
                })
                .collect::<io::Result<Vec<_>>>()?;
            let batch =
                arrow::record_batch::RecordBatch::try_new(full_batch.schema(), taken_columns)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            return self.apply_delta_overlay_to_batch(batch);
        }

        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        // Include _id column
        // OPTIMIZED: Read only the IDs we need instead of all IDs
        let filtered_ids = self.storage.read_ids_by_indices(row_indices)?;
        fields.push(Field::new("_id", ArrowDataType::Int64, false));
        arrays.push(Arc::new(Int64Array::from(filtered_ids)));

        // Build column filter set if col_refs provided
        let col_set: Option<std::collections::HashSet<&str>> =
            col_refs.map(|refs| refs.iter().copied().collect());

        // Read each column for the specified row indices
        for (col_name, _dt) in schema.iter() {
            if let Some(ref set) = col_set {
                if !set.iter().any(|r| r.eq_ignore_ascii_case(col_name)) {
                    continue;
                }
            }
            let col_data = self.storage.read_column_by_indices(col_name, row_indices)?;

            let (arrow_dt, array): (ArrowDataType, ArrayRef) = match col_data {
                ColumnData::Int64(values) => {
                    (ArrowDataType::Int64, Arc::new(Int64Array::from(values)))
                }
                ColumnData::Float64(values) => {
                    (ArrowDataType::Float64, Arc::new(Float64Array::from(values)))
                }
                ColumnData::String {
                    offsets,
                    data: bytes,
                } => {
                    let count = offsets.len().saturating_sub(1);
                    let strings: Vec<Option<&str>> = (0..count)
                        .map(|i| {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            std::str::from_utf8(&bytes[start..end]).ok()
                        })
                        .collect();
                    (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                }
                ColumnData::Bool { data: packed, len } => {
                    let bools: Vec<Option<bool>> = (0..len)
                        .map(|i| {
                            let byte_idx = i / 8;
                            let bit_idx = i % 8;
                            Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                        })
                        .collect();
                    (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                }
                ColumnData::Binary {
                    offsets,
                    data: bytes,
                } => {
                    use arrow::array::BinaryArray;
                    let count = offsets.len().saturating_sub(1);
                    let binary_data: Vec<Option<&[u8]>> = (0..count)
                        .map(|i| {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            Some(&bytes[start..end] as &[u8])
                        })
                        .collect();
                    (
                        ArrowDataType::Binary,
                        Arc::new(BinaryArray::from(binary_data)),
                    )
                }
                ColumnData::StringDict {
                    indices,
                    dict_offsets,
                    dict_data,
                } => {
                    let strings: Vec<Option<String>> = indices
                        .iter()
                        .map(|&idx| {
                            if idx == 0 {
                                None
                            } else {
                                let dict_idx = (idx - 1) as usize;
                                if dict_idx + 1 < dict_offsets.len() {
                                    let start = dict_offsets[dict_idx] as usize;
                                    let end = dict_offsets[dict_idx + 1] as usize;
                                    std::str::from_utf8(&dict_data[start..end])
                                        .ok()
                                        .map(|s| s.to_string())
                                } else {
                                    None
                                }
                            }
                        })
                        .collect();
                    (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                }
                ColumnData::FixedList { data, dim } => fixedlist_to_arrow_pair(&data, dim),
                ColumnData::Float16List { data, dim } => float16list_to_arrow_pair(&data, dim),
            };

            fields.push(Field::new(col_name, arrow_dt, true));
            arrays.push(array);
        }

        let schema = Arc::new(arrow::datatypes::Schema::new(fields));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        self.apply_delta_overlay_to_batch(batch)
    }

    fn apply_delta_overlay_to_batch(
        &self,
        batch: arrow::record_batch::RecordBatch,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        if batch.num_rows() == 0 || !self.has_pending_deltas() {
            return Ok(batch);
        }

        let Some(row_ids) = Self::row_ids_from_batch(&batch) else {
            return Ok(batch);
        };

        let delta = self.storage.delta_store();
        crate::storage::DeltaMerger::merge(&batch, &delta, &row_ids)
    }

    fn row_ids_from_batch(batch: &arrow::record_batch::RecordBatch) -> Option<Vec<u64>> {
        use arrow::array::Array;

        let id_col = batch.column_by_name("_id")?;
        if let Some(arr) = id_col.as_any().downcast_ref::<arrow::array::Int64Array>() {
            Some((0..arr.len()).map(|i| arr.value(i) as u64).collect())
        } else {
            id_col
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .map(|arr| (0..arr.len()).map(|i| arr.value(i)).collect())
        }
    }

    fn project_record_batch_by_names(
        batch: arrow::record_batch::RecordBatch,
        column_names: Option<&[&str]>,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        let Some(cols) = column_names else {
            return Ok(batch);
        };

        let mut fields = Vec::with_capacity(cols.len());
        let mut arrays = Vec::with_capacity(cols.len());
        for &name in cols {
            if let Ok(idx) = batch.schema().index_of(name) {
                fields.push(batch.schema().field(idx).clone());
                arrays.push(batch.column(idx).clone());
            }
        }
        let schema = Arc::new(arrow::datatypes::Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// OPTIMIZED: Read a single row by ID using O(1) index lookup
    /// Much faster than WHERE _id = X which scans all data
    pub fn read_row_by_id_to_arrow(
        &self,
        id: u64,
    ) -> io::Result<Option<arrow::record_batch::RecordBatch>> {
        use crate::data::DataType;
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        // V4 mmap-only: user-space page cache path — avoids mmap page faults and hash map build.
        // retrieve_rcix reads field values using cached 4KB pages (~50ns/page after warmup).
        if self.storage.is_v4_format() && !self.storage.has_v4_in_memory_data() {
            if let Some(row_vals) = self.storage.retrieve_rcix(id)? {
                // Convert Vec<(col_name, Value)> to single-row Arrow RecordBatch
                use crate::data::Value;
                let mut fields: Vec<Field> = Vec::with_capacity(row_vals.len());
                let mut arrays: Vec<ArrayRef> = Vec::with_capacity(row_vals.len());
                for (col_name, val) in &row_vals {
                    let (dt, arr): (ArrowDataType, ArrayRef) = match val {
                        Value::Int64(v) => {
                            (ArrowDataType::Int64, Arc::new(Int64Array::from(vec![*v])))
                        }
                        Value::Float64(v) => (
                            ArrowDataType::Float64,
                            Arc::new(Float64Array::from(vec![*v])),
                        ),
                        Value::String(s) => (
                            ArrowDataType::Utf8,
                            Arc::new(StringArray::from(vec![s.as_str()])),
                        ),
                        Value::Bool(b) => (
                            ArrowDataType::Boolean,
                            Arc::new(BooleanArray::from(vec![*b])),
                        ),
                        Value::Binary(bytes) => {
                            use arrow::array::BinaryArray;
                            (
                                ArrowDataType::Binary,
                                Arc::new(BinaryArray::from(vec![Some(bytes.as_slice())]))
                                    as ArrayRef,
                            )
                        }
                        Value::Null => (
                            ArrowDataType::Utf8,
                            Arc::new(StringArray::from(vec![None as Option<&str>])),
                        ),
                        _ => (
                            ArrowDataType::Utf8,
                            Arc::new(StringArray::from(vec![None as Option<&str>])),
                        ),
                    };
                    let nullable = matches!(val, Value::Null);
                    fields.push(Field::new(col_name, dt, nullable));
                    arrays.push(arr);
                }
                let schema = Arc::new(Schema::new(fields));
                let batch = arrow::record_batch::RecordBatch::try_new(schema, arrays)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                return Ok(Some(batch));
            }
            return Ok(None);
        }

        let row_data = match self.storage.read_row_by_id(id, None)? {
            Some(data) => data,
            None => return Ok(None),
        };

        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        // Add _id column first
        fields.push(Field::new("_id", ArrowDataType::Int64, false));
        arrays.push(Arc::new(Int64Array::from(vec![id as i64])));

        // Add other columns in schema order - ensure all have length 1
        for (col_name, dt) in schema.iter() {
            let (arrow_dt, array): (ArrowDataType, ArrayRef) = if let Some(col_data) =
                row_data.get(col_name)
            {
                match col_data {
                    ColumnData::Int64(values) if !values.is_empty() => (
                        ArrowDataType::Int64,
                        Arc::new(Int64Array::from(vec![values[0]])),
                    ),
                    ColumnData::Float64(values) if !values.is_empty() => (
                        ArrowDataType::Float64,
                        Arc::new(Float64Array::from(vec![values[0]])),
                    ),
                    ColumnData::String {
                        offsets,
                        data: bytes,
                    } if offsets.len() > 1 => {
                        let start = offsets[0] as usize;
                        let end = offsets[1] as usize;
                        let s = std::str::from_utf8(&bytes[start..end]).ok();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(vec![s])))
                    }
                    ColumnData::Bool { data: packed, len } if *len > 0 => {
                        let val = !packed.is_empty() && (packed[0] & 1) == 1;
                        (
                            ArrowDataType::Boolean,
                            Arc::new(BooleanArray::from(vec![Some(val)])),
                        )
                    }
                    ColumnData::Binary {
                        offsets,
                        data: bytes,
                    } if offsets.len() > 1 => {
                        use arrow::array::BinaryArray;
                        let start = offsets[0] as usize;
                        let end = offsets[1] as usize;
                        (
                            ArrowDataType::Binary,
                            Arc::new(BinaryArray::from(vec![Some(&bytes[start..end] as &[u8])])),
                        )
                    }
                    ColumnData::StringDict {
                        indices,
                        dict_offsets,
                        dict_data,
                    } if !indices.is_empty() => {
                        let idx = indices[0];
                        let s = if idx == 0 {
                            None
                        } else {
                            let dict_idx = (idx - 1) as usize;
                            if dict_idx + 1 < dict_offsets.len() {
                                let start = dict_offsets[dict_idx] as usize;
                                let end = dict_offsets[dict_idx + 1] as usize;
                                std::str::from_utf8(&dict_data[start..end])
                                    .ok()
                                    .map(|s| s.to_string())
                            } else {
                                None
                            }
                        };
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(vec![s])))
                    }
                    // Empty column data - return typed null
                    _ => Self::create_typed_null_array(dt),
                }
            } else {
                // Column not in row_data - return typed null value
                Self::create_typed_null_array(dt)
            };
            fields.push(Field::new(col_name, arrow_dt, true));
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        Ok(Some(batch))
    }

    /// Read a single row by ID with projection, reusing the point-lookup path.
    pub fn read_row_by_id_projected_to_arrow(
        &self,
        id: u64,
        column_names: &[&str],
    ) -> io::Result<Option<arrow::record_batch::RecordBatch>> {
        use crate::data::Value;
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        if column_names.is_empty() {
            let schema = Arc::new(Schema::new(Vec::<Field>::new()));
            return Ok(Some(arrow::record_batch::RecordBatch::new_empty(schema)));
        }

        if column_names.len() == 1 && column_names[0] == "_id" {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "_id",
                ArrowDataType::Int64,
                false,
            )]));
            let batch = arrow::record_batch::RecordBatch::try_new(
                schema,
                vec![Arc::new(Int64Array::from(vec![id as i64]))],
            )
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            return Ok(Some(batch));
        }

        let requested_cols: Vec<&str> = column_names
            .iter()
            .copied()
            .filter(|name| *name != "_id")
            .collect();

        if self.storage.is_v4_format() && !self.storage.has_v4_in_memory_data() {
            if let Some(row_vals) = self.storage.retrieve_rcix_projected(id, column_names)? {
                let schema = self.schema.read();
                let mut fields: Vec<Field> = Vec::with_capacity(row_vals.len());
                let mut arrays: Vec<ArrayRef> = Vec::with_capacity(fields.capacity());

                for (col_name, value) in row_vals {
                    let nullable = col_name != "_id";
                    let (arrow_dt, array): (ArrowDataType, ArrayRef) = match value {
                        Value::Int64(v) => {
                            (ArrowDataType::Int64, Arc::new(Int64Array::from(vec![v])))
                        }
                        Value::Float64(v) => (
                            ArrowDataType::Float64,
                            Arc::new(Float64Array::from(vec![v])),
                        ),
                        Value::String(s) => (
                            ArrowDataType::Utf8,
                            Arc::new(StringArray::from(vec![s.as_str()])),
                        ),
                        Value::Bool(v) => (
                            ArrowDataType::Boolean,
                            Arc::new(BooleanArray::from(vec![v])),
                        ),
                        Value::Binary(bytes) => {
                            use arrow::array::BinaryArray;
                            (
                                ArrowDataType::Binary,
                                Arc::new(BinaryArray::from(vec![Some(bytes.as_slice())])),
                            )
                        }
                        Value::Null => {
                            let dt = schema
                                .iter()
                                .find(|(name, _)| name == &col_name)
                                .map(|(_, dt)| dt.clone())
                                .unwrap_or(crate::data::DataType::String);
                            Self::create_typed_null_array(&dt)
                        }
                        _ => {
                            let dt = schema
                                .iter()
                                .find(|(name, _)| name == &col_name)
                                .map(|(_, dt)| dt.clone())
                                .unwrap_or(crate::data::DataType::String);
                            Self::create_typed_null_array(&dt)
                        }
                    };
                    fields.push(Field::new(&col_name, arrow_dt, nullable));
                    arrays.push(array);
                }

                let schema = Arc::new(Schema::new(fields));
                let batch = arrow::record_batch::RecordBatch::try_new(schema, arrays)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                return Ok(Some(batch));
            }
            return Ok(None);
        }

        let row_data = match self
            .storage
            .read_row_by_id(id, Some(requested_cols.as_slice()))?
        {
            Some(data) => data,
            None => return Ok(None),
        };

        let schema = self.schema.read();
        let include_id = column_names.contains(&"_id");
        let mut fields: Vec<Field> =
            Vec::with_capacity(requested_cols.len() + usize::from(include_id));
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(fields.capacity());

        if include_id {
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(vec![id as i64])));
        }

        for &col_name in &requested_cols {
            let dt = match schema.iter().find(|(name, _)| name == col_name) {
                Some((_, dt)) => dt,
                None => continue,
            };

            let (arrow_dt, array): (ArrowDataType, ArrayRef) = if let Some(col_data) =
                row_data.get(col_name)
            {
                match col_data {
                    ColumnData::Int64(values) if !values.is_empty() => (
                        ArrowDataType::Int64,
                        Arc::new(Int64Array::from(vec![values[0]])),
                    ),
                    ColumnData::Float64(values) if !values.is_empty() => (
                        ArrowDataType::Float64,
                        Arc::new(Float64Array::from(vec![values[0]])),
                    ),
                    ColumnData::String {
                        offsets,
                        data: bytes,
                    } if offsets.len() > 1 => {
                        let start = offsets[0] as usize;
                        let end = offsets[1] as usize;
                        let s = std::str::from_utf8(&bytes[start..end]).ok();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(vec![s])))
                    }
                    ColumnData::Bool { data: packed, len } if *len > 0 => {
                        let val = !packed.is_empty() && (packed[0] & 1) == 1;
                        (
                            ArrowDataType::Boolean,
                            Arc::new(BooleanArray::from(vec![Some(val)])),
                        )
                    }
                    ColumnData::Binary {
                        offsets,
                        data: bytes,
                    } if offsets.len() > 1 => {
                        use arrow::array::BinaryArray;
                        let start = offsets[0] as usize;
                        let end = offsets[1] as usize;
                        (
                            ArrowDataType::Binary,
                            Arc::new(BinaryArray::from(vec![Some(&bytes[start..end] as &[u8])])),
                        )
                    }
                    ColumnData::StringDict {
                        indices,
                        dict_offsets,
                        dict_data,
                    } if !indices.is_empty() => {
                        let idx = indices[0];
                        let s = if idx == 0 {
                            None
                        } else {
                            let dict_idx = (idx - 1) as usize;
                            if dict_idx + 1 < dict_offsets.len() {
                                let start = dict_offsets[dict_idx] as usize;
                                let end = dict_offsets[dict_idx + 1] as usize;
                                std::str::from_utf8(&dict_data[start..end])
                                    .ok()
                                    .map(|v| v.to_string())
                            } else {
                                None
                            }
                        };
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(vec![s])))
                    }
                    _ => Self::create_typed_null_array(dt),
                }
            } else {
                Self::create_typed_null_array(dt)
            };

            fields.push(Field::new(col_name, arrow_dt, true));
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        Ok(Some(batch))
    }

    /// Batch retrieve multiple rows by IDs.
    /// Fast path: retrieve_many_mmap (one footer lock + one mmap slice per RG).
    /// Fallback: per-ID retrieve_rcix loop (for non-RCIX files).
    /// Returns a RecordBatch with all found rows (in original ID order, missing IDs skipped).
    pub fn read_rows_by_ids_to_arrow(
        &self,
        ids: &[u64],
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use crate::data::Value;
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        if ids.is_empty() {
            let schema = Arc::new(Schema::new(Vec::<Field>::new()));
            return Ok(RecordBatch::new_empty(schema));
        }

        // Fast path: one footer lock + one mmap body slice per RG
        if self.storage.is_v4_format() && !self.storage.has_v4_in_memory_data() {
            if let Ok(Some(batch)) = self.storage.retrieve_many_mmap(ids) {
                let batch = self.apply_delta_overlay_to_batch(batch)?;
                if !self.has_delta() || batch.num_rows() == ids.len() {
                    return Ok(batch);
                }

                let found_ids = Self::row_ids_from_batch(&batch)
                    .unwrap_or_default()
                    .into_iter()
                    .collect::<std::collections::HashSet<_>>();
                let missing_ids: Vec<u64> = ids
                    .iter()
                    .copied()
                    .filter(|id| !found_ids.contains(id))
                    .collect();
                if missing_ids.is_empty() {
                    return Ok(batch);
                }

                let delta_batch = self.storage.read_delta_rows_by_ids_to_arrow(&missing_ids)?;
                if delta_batch.num_rows() == 0 {
                    return Ok(batch);
                }
                if batch.num_rows() == 0 {
                    return Ok(delta_batch);
                }

                let schema = batch.schema();
                let refs = vec![&batch, &delta_batch];
                return arrow::compute::concat_batches(&schema, refs)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
            }
        }

        // Fallback: per-ID retrieve_rcix loop (non-RCIX RGs)
        let mut all_rows = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Ok(Some(row_vals)) = self.storage.retrieve_rcix(id) {
                all_rows.push(row_vals);
            }
        }

        if all_rows.is_empty() {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "_id",
                ArrowDataType::Int64,
                false,
            )]));
            return Ok(RecordBatch::new_empty(schema));
        }

        let col_names: Vec<String> = all_rows[0].iter().map(|(name, _)| name.clone()).collect();
        let num_cols = col_names.len();
        let num_rows = all_rows.len();

        let mut fields: Vec<Field> = Vec::with_capacity(num_cols);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(num_cols);

        for col_idx in 0..num_cols {
            let col_name = &col_names[col_idx];
            let (arrow_dt, array): (ArrowDataType, ArrayRef) = {
                let first_val = &all_rows[0][col_idx].1;
                match first_val {
                    Value::Int64(_) => {
                        let mut values = Vec::with_capacity(num_rows);
                        for row in &all_rows {
                            match &row[col_idx].1 {
                                Value::Int64(v) => values.push(*v),
                                _ => values.push(0),
                            }
                        }
                        (ArrowDataType::Int64, Arc::new(Int64Array::from(values)))
                    }
                    Value::Float64(_) => {
                        let mut values = Vec::with_capacity(num_rows);
                        for row in &all_rows {
                            match &row[col_idx].1 {
                                Value::Float64(v) => values.push(*v),
                                _ => values.push(0.0),
                            }
                        }
                        (ArrowDataType::Float64, Arc::new(Float64Array::from(values)))
                    }
                    Value::String(_) => {
                        let mut values = Vec::with_capacity(num_rows);
                        for row in &all_rows {
                            match &row[col_idx].1 {
                                Value::String(s) => values.push(s.as_str()),
                                _ => values.push(""),
                            }
                        }
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(values)))
                    }
                    Value::Bool(_) => {
                        let mut values = Vec::with_capacity(num_rows);
                        for row in &all_rows {
                            match &row[col_idx].1 {
                                Value::Bool(b) => values.push(Some(*b)),
                                _ => values.push(None),
                            }
                        }
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(values)))
                    }
                    Value::Null => (
                        ArrowDataType::Utf8,
                        Arc::new(StringArray::from(vec![""; num_rows])),
                    ),
                    _ => (
                        ArrowDataType::Utf8,
                        Arc::new(StringArray::from(vec![""; num_rows])),
                    ),
                }
            };
            fields.push(Field::new(col_name, arrow_dt, false));
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        self.apply_delta_overlay_to_batch(batch)
    }

    /// Create a typed null array with a single null value
    fn create_typed_null_array(
        dt: &crate::data::DataType,
    ) -> (arrow::datatypes::DataType, arrow::array::ArrayRef) {
        use crate::data::DataType;
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::DataType as ArrowDataType;
        use std::sync::Arc;

        match dt {
            DataType::Int64
            | DataType::Int32
            | DataType::Int16
            | DataType::Int8
            | DataType::UInt64
            | DataType::UInt32
            | DataType::UInt16
            | DataType::UInt8 => (
                ArrowDataType::Int64,
                Arc::new(Int64Array::from(vec![None as Option<i64>])) as ArrayRef,
            ),
            DataType::Float64 | DataType::Float32 => (
                ArrowDataType::Float64,
                Arc::new(Float64Array::from(vec![None as Option<f64>])) as ArrayRef,
            ),
            DataType::Bool => (
                ArrowDataType::Boolean,
                Arc::new(BooleanArray::from(vec![None as Option<bool>])) as ArrayRef,
            ),
            DataType::String | _ => (
                ArrowDataType::Utf8,
                Arc::new(StringArray::from(vec![None as Option<&str>])) as ArrayRef,
            ),
        }
    }

    fn read_string_filter_with_delta_column_updates(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_value: &str,
        base_indices: &[usize],
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::StringArray;
        use arrow::compute::kernels::cmp;
        use std::collections::HashSet;

        let mut batches = Vec::new();
        let mut seen_ids = HashSet::new();

        if !base_indices.is_empty() {
            let batch = self.read_columns_by_indices_to_arrow(base_indices, None)?;
            if let Some(ids) = Self::row_ids_from_batch(&batch) {
                seen_ids.extend(ids);
            }
            if batch.num_rows() > 0 {
                batches.push(batch);
            }
        }

        let mut extra_ids = self.pending_delta_string_update_matches(filter_column, filter_value);
        if self.has_delta() {
            extra_ids.extend(self.delta_string_match_ids(filter_column, filter_value)?);
        }
        extra_ids.sort_unstable();
        extra_ids.dedup();
        extra_ids.retain(|id| !seen_ids.contains(id));
        if !extra_ids.is_empty() {
            let batch = self.read_rows_by_ids_to_arrow(&extra_ids)?;
            if batch.num_rows() > 0 {
                batches.push(batch);
            }
        }

        if batches.is_empty() {
            return self.read_columns_to_arrow(column_names, 0, Some(0));
        }

        let schema = batches[0].schema();
        let refs: Vec<&arrow::record_batch::RecordBatch> = batches.iter().collect();
        let combined = arrow::compute::concat_batches(&schema, refs)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        let Some(col) = combined.column_by_name(filter_column) else {
            return Self::project_record_batch_by_names(combined, column_names);
        };
        let Some(str_arr) = col.as_any().downcast_ref::<StringArray>() else {
            return Self::project_record_batch_by_names(combined, column_names);
        };
        let scalar_arr = StringArray::from(vec![Some(filter_value)]);
        let scalar = arrow::array::Scalar::new(&scalar_arr);
        let mask = cmp::eq(str_arr, &scalar)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let filtered = arrow::compute::filter_record_batch(&combined, &mask)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        Self::project_record_batch_by_names(filtered, column_names)
    }

    /// Read columns with STRING predicate pushdown to Arrow
    /// Filters rows at storage level for string equality (much faster than post-filtering)
    pub fn read_columns_filtered_string_to_arrow(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_value: &str,
        filter_eq: bool,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        // FAST PATH: mmap-native string equality scan + late materialization.
        // Avoids materializing all rows (which causes ~100ms for 1M unique StringDict strings).
        // Works for cold backends — scan_string_filter_mmap loads the footer lazily.
        if filter_eq {
            let scan_res =
                self.storage
                    .scan_string_filter_mmap(filter_column, filter_value, None)?;
            if let Some(indices) = scan_res {
                if self.pending_delta_updates_column(filter_column) || self.has_delta() {
                    return self.read_string_filter_with_delta_column_updates(
                        column_names,
                        filter_column,
                        filter_value,
                        &indices,
                    );
                }
                if indices.is_empty() {
                    return self.read_columns_to_arrow(column_names, 0, Some(0));
                }
                return self.read_columns_by_indices_to_arrow(&indices, column_names);
            }
        }

        let needs_filter_col_for_fallback = column_names
            .map(|cols| !cols.iter().any(|c| c.eq_ignore_ascii_case(filter_column)))
            .unwrap_or(false);
        let fallback_cols: Option<Vec<&str>> = column_names.map(|cols| {
            let mut expanded = cols.to_vec();
            if needs_filter_col_for_fallback {
                expanded.push(filter_column);
            }
            expanded
        });
        let fallback_col_refs = fallback_cols.as_deref().or(column_names);

        // V4 mmap-only: fall back to full batch read + Arrow filter (for neq or non-V4 format)
        if self.storage.is_v4_format() && !self.storage.has_v4_in_memory_data() {
            let full_batch = self.read_columns_to_arrow(fallback_col_refs, 0, None)?;
            if full_batch.num_rows() == 0 {
                return Ok(full_batch);
            }
            if let Some(col) = full_batch.column_by_name(filter_column) {
                if let Some(str_arr) = col.as_any().downcast_ref::<StringArray>() {
                    use arrow::compute::kernels::cmp;
                    let scalar_arr = StringArray::from(vec![Some(filter_value)]);
                    let scalar = arrow::array::Scalar::new(&scalar_arr);
                    let mask = if filter_eq {
                        cmp::eq(str_arr, &scalar)
                    } else {
                        cmp::neq(str_arr, &scalar)
                    }
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                    let filtered = arrow::compute::filter_record_batch(&full_batch, &mask)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                    if needs_filter_col_for_fallback {
                        if let Some(cols) = column_names {
                            let mut fields = Vec::with_capacity(cols.len());
                            let mut arrays = Vec::with_capacity(cols.len());
                            for &name in cols {
                                if let Ok(idx) = filtered.schema().index_of(name) {
                                    fields.push(filtered.schema().field(idx).clone());
                                    arrays.push(filtered.column(idx).clone());
                                }
                            }
                            let schema = Arc::new(Schema::new(fields));
                            return RecordBatch::try_new(schema, arrays).map_err(|e| {
                                io::Error::new(io::ErrorKind::InvalidData, e.to_string())
                            });
                        }
                    }
                    return Ok(filtered);
                }
            }
            return Ok(full_batch);
        }

        let (col_data, matching_indices) = self.storage.read_columns_filtered_string(
            column_names,
            filter_column,
            filter_value,
            filter_eq,
        )?;

        if col_data.is_empty() || matching_indices.is_empty() {
            // Return empty batch with proper schema
            return self.read_columns_to_arrow(column_names, 0, Some(0));
        }

        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        // Include _id column with filtered indices
        let include_id = column_names
            .map(|cols| cols.contains(&"_id"))
            .unwrap_or(true);
        if include_id {
            // OPTIMIZED: Read only the IDs we need instead of all IDs
            let filtered_ids = self.storage.read_ids_by_indices(&matching_indices)?;
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(filtered_ids)));
        }

        // Determine column order
        let col_order: Vec<String> = if let Some(names) = column_names {
            names
                .iter()
                .filter(|&s| *s != "_id")
                .map(|s| s.to_string())
                .collect()
        } else {
            schema.iter().map(|(n, _)| n.clone()).collect()
        };

        for col_name in &col_order {
            if let Some(data) = col_data.get(col_name) {
                let (arrow_dt, array): (ArrowDataType, ArrayRef) = match data {
                    ColumnData::Int64(values) => (
                        ArrowDataType::Int64,
                        Arc::new(Int64Array::from_iter_values(values.iter().copied())),
                    ),
                    ColumnData::Float64(values) => (
                        ArrowDataType::Float64,
                        Arc::new(Float64Array::from_iter_values(values.iter().copied())),
                    ),
                    ColumnData::String {
                        offsets,
                        data: bytes,
                    } => {
                        let count = offsets.len().saturating_sub(1);
                        let strings: Vec<Option<&str>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&bytes[start..end]).ok()
                            })
                            .collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                    ColumnData::Bool { data: packed, len } => {
                        let bools: Vec<Option<bool>> = (0..*len)
                            .map(|i| {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(
                                    byte_idx < packed.len()
                                        && (packed[byte_idx] >> bit_idx) & 1 == 1,
                                )
                            })
                            .collect();
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    }
                    ColumnData::Binary {
                        offsets,
                        data: bytes,
                    } => {
                        use arrow::array::BinaryArray;
                        let count = offsets.len().saturating_sub(1);
                        let binary_data: Vec<Option<&[u8]>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&bytes[start..end] as &[u8])
                            })
                            .collect();
                        (
                            ArrowDataType::Binary,
                            Arc::new(BinaryArray::from(binary_data)),
                        )
                    }
                    ColumnData::StringDict {
                        indices,
                        dict_offsets,
                        dict_data,
                    } => {
                        let strings: Vec<Option<String>> = indices
                            .iter()
                            .map(|&idx| {
                                if idx == 0 {
                                    None
                                } else {
                                    let dict_idx = (idx - 1) as usize;
                                    if dict_idx + 1 < dict_offsets.len() {
                                        let start = dict_offsets[dict_idx] as usize;
                                        let end = dict_offsets[dict_idx + 1] as usize;
                                        std::str::from_utf8(&dict_data[start..end])
                                            .ok()
                                            .map(|s| s.to_string())
                                    } else {
                                        None
                                    }
                                }
                            })
                            .collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                    ColumnData::FixedList { data, dim } => fixedlist_to_arrow_pair(data, *dim),
                    ColumnData::Float16List { data, dim } => float16list_to_arrow_pair(data, *dim),
                };

                fields.push(Field::new(col_name, arrow_dt, true));
                arrays.push(array);
            }
        }

        let schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Read columns with STRING predicate pushdown and LIMIT early termination
    /// Much faster for queries like SELECT * WHERE col = 'value' LIMIT n
    pub fn read_columns_filtered_string_with_limit_to_arrow(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_value: &str,
        filter_eq: bool,
        limit: usize,
        offset: usize,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        if filter_eq && limit == 1 && offset == 0 {
            if let Some(cache) = self.get_or_build_first_string_row_id_cache(filter_column)? {
                if let Some(&row_id) = cache.get(filter_value) {
                    let batch = if let Some(cols) = column_names {
                        self.read_row_by_id_projected_to_arrow(row_id, cols)?
                    } else {
                        self.read_row_by_id_to_arrow(row_id)?
                    };
                    if let Some(batch) = batch {
                        return Ok(batch);
                    }
                    self.first_string_row_id_cache.write().remove(filter_column);
                } else {
                    return self.read_columns_to_arrow(column_names, 0, Some(0));
                }
            }
        }

        if filter_eq
            && limit > 0
            && self.pending_delta_delete_count() == 0
            && !self.pending_delta_updates_column(filter_column)
        {
            let needed = offset.saturating_add(limit);
            if let Some(indices) =
                self.scan_string_filter_mmap(filter_column, filter_value, Some(needed))?
            {
                if indices.len() >= needed || !self.has_delta() {
                    let final_indices: Vec<usize> =
                        indices.into_iter().skip(offset).take(limit).collect();
                    if final_indices.is_empty() {
                        return self.read_columns_to_arrow(column_names, 0, Some(0));
                    }
                    let batch =
                        self.read_columns_by_indices_to_arrow(&final_indices, column_names)?;
                    return Self::project_record_batch_by_names(batch, column_names);
                }
            }
        }

        let (col_data, matching_indices) = self.storage.read_columns_filtered_string_with_limit(
            column_names,
            filter_column,
            filter_value,
            filter_eq,
            limit,
            offset,
        )?;

        if col_data.is_empty() || matching_indices.is_empty() {
            return self.read_columns_to_arrow(column_names, 0, Some(0));
        }

        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        let include_id = column_names
            .map(|cols| cols.contains(&"_id"))
            .unwrap_or(true);
        if include_id {
            // OPTIMIZED: Read only the IDs we need instead of all IDs
            let filtered_ids = self.storage.read_ids_by_indices(&matching_indices)?;
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(filtered_ids)));
        }

        let col_order: Vec<String> = if let Some(names) = column_names {
            names
                .iter()
                .filter(|&s| *s != "_id")
                .map(|s| s.to_string())
                .collect()
        } else {
            schema.iter().map(|(n, _)| n.clone()).collect()
        };

        for col_name in &col_order {
            if let Some(data) = col_data.get(col_name) {
                let (arrow_dt, array): (ArrowDataType, ArrayRef) = match data {
                    ColumnData::Int64(values) => (
                        ArrowDataType::Int64,
                        Arc::new(Int64Array::from_iter_values(values.iter().copied())),
                    ),
                    ColumnData::Float64(values) => (
                        ArrowDataType::Float64,
                        Arc::new(Float64Array::from_iter_values(values.iter().copied())),
                    ),
                    ColumnData::String {
                        offsets,
                        data: bytes,
                    } => {
                        let count = offsets.len().saturating_sub(1);
                        let strings: Vec<Option<&str>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&bytes[start..end]).ok()
                            })
                            .collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                    ColumnData::Bool { data: packed, len } => {
                        let bools: Vec<Option<bool>> = (0..*len)
                            .map(|i| {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(
                                    byte_idx < packed.len()
                                        && (packed[byte_idx] >> bit_idx) & 1 == 1,
                                )
                            })
                            .collect();
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    }
                    ColumnData::Binary {
                        offsets,
                        data: bytes,
                    } => {
                        use arrow::array::BinaryArray;
                        let count = offsets.len().saturating_sub(1);
                        let binary_data: Vec<Option<&[u8]>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&bytes[start..end] as &[u8])
                            })
                            .collect();
                        (
                            ArrowDataType::Binary,
                            Arc::new(BinaryArray::from(binary_data)),
                        )
                    }
                    ColumnData::StringDict {
                        indices,
                        dict_offsets,
                        dict_data,
                    } => {
                        // OPTIMIZED: Use Arrow DictionaryArray to avoid string allocations
                        use arrow::array::{DictionaryArray, UInt32Array};
                        use arrow::datatypes::UInt32Type;

                        // Build dictionary values (unique strings) - use &str references
                        let dict_count = dict_offsets.len().saturating_sub(1);
                        let dict_strings: Vec<Option<&str>> = (0..dict_count)
                            .map(|i| {
                                let start = dict_offsets[i] as usize;
                                let end = dict_offsets[i + 1] as usize;
                                std::str::from_utf8(&dict_data[start..end]).ok()
                            })
                            .collect();
                        let values = StringArray::from(dict_strings);

                        // Convert indices (0 = NULL, 1+ = dict index)
                        let keys: Vec<Option<u32>> = indices
                            .iter()
                            .map(|&idx| if idx == 0 { None } else { Some(idx - 1) })
                            .collect();
                        let keys_array = UInt32Array::from(keys);

                        let dict_array =
                            DictionaryArray::<UInt32Type>::try_new(keys_array, Arc::new(values))
                                .map_err(|e| {
                                    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
                                })?;
                        (
                            ArrowDataType::Dictionary(
                                Box::new(ArrowDataType::UInt32),
                                Box::new(ArrowDataType::Utf8),
                            ),
                            Arc::new(dict_array) as ArrayRef,
                        )
                    }
                    ColumnData::FixedList { data, dim } => fixedlist_to_arrow_pair(data, *dim),
                    ColumnData::Float16List { data, dim } => float16list_to_arrow_pair(data, *dim),
                };
                fields.push(Field::new(col_name, arrow_dt, true));
                arrays.push(array);
            }
        }

        let schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Read columns with numeric RANGE predicate pushdown and LIMIT early termination
    /// Much faster for queries like SELECT * WHERE col BETWEEN low AND high LIMIT n
    pub fn read_columns_filtered_range_with_limit_to_arrow(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        low: f64,
        high: f64,
        limit: usize,
        offset: usize,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        let (col_data, matching_indices) = self.storage.read_columns_filtered_range_with_limit(
            column_names,
            filter_column,
            low,
            high,
            limit,
            offset,
        )?;

        if col_data.is_empty() || matching_indices.is_empty() {
            return self.read_columns_to_arrow(column_names, 0, Some(0));
        }

        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        let include_id = column_names
            .map(|cols| cols.contains(&"_id"))
            .unwrap_or(true);
        if include_id {
            let filtered_ids = self.storage.read_ids_by_indices(&matching_indices)?;
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(filtered_ids)));
        }

        let col_order: Vec<String> = if let Some(names) = column_names {
            names
                .iter()
                .filter(|&s| *s != "_id")
                .map(|s| s.to_string())
                .collect()
        } else {
            schema.iter().map(|(n, _)| n.clone()).collect()
        };

        for col_name in &col_order {
            if let Some(data) = col_data.get(col_name) {
                let (arrow_dt, array): (ArrowDataType, ArrayRef) = match data {
                    ColumnData::Int64(values) => (
                        ArrowDataType::Int64,
                        Arc::new(Int64Array::from_iter_values(values.iter().copied())),
                    ),
                    ColumnData::Float64(values) => (
                        ArrowDataType::Float64,
                        Arc::new(Float64Array::from_iter_values(values.iter().copied())),
                    ),
                    ColumnData::String {
                        offsets,
                        data: bytes,
                    } => {
                        let count = offsets.len().saturating_sub(1);
                        let strings: Vec<Option<&str>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&bytes[start..end]).ok()
                            })
                            .collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                    ColumnData::Bool { data: packed, len } => {
                        let bools: Vec<Option<bool>> = (0..*len)
                            .map(|i| {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(
                                    byte_idx < packed.len()
                                        && (packed[byte_idx] >> bit_idx) & 1 == 1,
                                )
                            })
                            .collect();
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    }
                    ColumnData::Binary {
                        offsets,
                        data: bytes,
                    } => {
                        use arrow::array::BinaryArray;
                        let count = offsets.len().saturating_sub(1);
                        let binary_data: Vec<Option<&[u8]>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&bytes[start..end])
                            })
                            .collect();
                        (
                            ArrowDataType::Binary,
                            Arc::new(BinaryArray::from(binary_data)),
                        )
                    }
                    ColumnData::StringDict {
                        indices,
                        dict_offsets,
                        dict_data,
                    } => {
                        // OPTIMIZED: Use Arrow DictionaryArray to avoid string allocations
                        use arrow::array::{DictionaryArray, UInt32Array};
                        use arrow::datatypes::UInt32Type;

                        // Build dictionary values (unique strings) - use &str references
                        let dict_count = dict_offsets.len().saturating_sub(1);
                        let dict_strings: Vec<Option<&str>> = (0..dict_count)
                            .map(|i| {
                                let start = dict_offsets[i] as usize;
                                let end = dict_offsets[i + 1] as usize;
                                std::str::from_utf8(&dict_data[start..end]).ok()
                            })
                            .collect();
                        let values = StringArray::from(dict_strings);

                        // Convert indices (0 = NULL, 1+ = dict index)
                        let keys: Vec<Option<u32>> = indices
                            .iter()
                            .map(|&idx| if idx == 0 { None } else { Some(idx - 1) })
                            .collect();
                        let keys_array = UInt32Array::from(keys);

                        let dict_array =
                            DictionaryArray::<UInt32Type>::try_new(keys_array, Arc::new(values))
                                .map_err(|e| {
                                    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
                                })?;

                        (
                            ArrowDataType::Dictionary(
                                Box::new(ArrowDataType::UInt32),
                                Box::new(ArrowDataType::Utf8),
                            ),
                            Arc::new(dict_array) as ArrayRef,
                        )
                    }
                    ColumnData::FixedList { data, dim } => fixedlist_to_arrow_pair(data, *dim),
                    ColumnData::Float16List { data, dim } => float16list_to_arrow_pair(data, *dim),
                };
                fields.push(Field::new(col_name, arrow_dt, true));
                arrays.push(array);
            }
        }

        let schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Compute numeric column aggregates directly without Arrow arrays.
    /// Returns (count, sum, min, max) for the specified column.
    pub fn compute_column_stats_mmap(
        &self,
        col_name: &str,
    ) -> io::Result<Option<(u64, f64, f64, f64)>> {
        // mmap path: reads only the requested column via RCIX fast seek
        if let Ok(Some(stats)) = self.storage.compute_column_stats_mmap(col_name) {
            return Ok(Some(stats));
        }
        // in-memory fallback
        self.storage.compute_column_stats_inmemory(col_name)
    }

    /// Read IDs for specific global row indices from mmap.
    pub fn get_ids_for_global_indices_mmap(&self, indices: &[usize]) -> io::Result<Vec<u64>> {
        self.storage.get_ids_for_global_indices_mmap(indices)
    }

    /// Mmap-level string equality scan: find matching row indices without Arrow arrays.
    pub fn scan_string_filter_mmap(
        &self,
        col_name: &str,
        target: &str,
        limit: Option<usize>,
    ) -> io::Result<Option<Vec<usize>>> {
        self.storage
            .scan_string_filter_mmap(col_name, target, limit)
    }

    /// Mmap-level string IN scan: find matching row indices without Arrow arrays.
    pub fn scan_string_in_mmap(
        &self,
        col_name: &str,
        values: &[String],
        limit: Option<usize>,
    ) -> io::Result<Option<Vec<usize>>> {
        self.storage.scan_string_in_mmap(col_name, values, limit)
    }

    /// Mmap-level LIKE pattern scan: find matching row indices without Arrow arrays.
    pub fn scan_like_filter_mmap(
        &self,
        col_name: &str,
        pattern: &str,
        limit: Option<usize>,
    ) -> io::Result<Option<Vec<usize>>> {
        self.storage.scan_like_filter_mmap(col_name, pattern, limit)
    }

    /// Single-pass parallel LIKE scan + row extraction: no separate scan/extract passes.
    pub fn scan_like_and_extract_mmap(
        &self,
        col_name: &str,
        pattern: &str,
        limit: Option<usize>,
    ) -> io::Result<Option<arrow::record_batch::RecordBatch>> {
        self.storage
            .scan_like_and_extract_mmap(col_name, pattern, limit)
    }

    /// Numeric range scan returning matching row IDs directly (not indices).
    /// Uses zone maps to prune Row Groups — O(matching_RGs) not O(all_rows).
    pub fn scan_numeric_range_mmap_with_ids(
        &self,
        col_name: &str,
        low: f64,
        high: f64,
    ) -> io::Result<Option<Vec<u64>>> {
        self.storage
            .scan_numeric_range_mmap_with_ids(col_name, low, high)
    }

    /// Single-pass scan + mark deleted + save for numeric predicates.
    /// Returns `Some(deleted_count)` on success, `None` if fast path unavailable.
    /// Never builds an id_to_idx HashMap — works directly with flat row indices.
    pub fn delete_where_numeric_range_inplace(
        &self,
        col_name: &str,
        low: f64,
        high: f64,
    ) -> io::Result<Option<i64>> {
        let result = self
            .storage
            .delete_where_numeric_range_inplace(col_name, low, high)?;
        if let Some(count) = result {
            *self.dirty.write() = false; // written directly to disk
            if count > 0 {
                self.storage.mark_sync_pending();
            }
        }
        Ok(result)
    }

    /// Delete rows by IDs using mmap binary search — bypasses the id_to_idx HashMap.
    /// Returns `Some(newly_deleted)` on success, `None` if fast path unavailable.
    pub fn delete_ids_inplace_v4(&self, ids: &[u64]) -> io::Result<Option<i64>> {
        let result = self.storage.delete_ids_inplace_v4(ids)?;
        if let Some(count) = result {
            *self.dirty.write() = false;
            if count > 0 {
                self.storage.mark_sync_pending();
            }
        }
        Ok(result)
    }

    /// Mmap-level numeric range scan: find matching row indices without Arrow arrays.
    pub fn scan_numeric_range_mmap(
        &self,
        col_name: &str,
        low: f64,
        high: f64,
        limit: Option<usize>,
    ) -> io::Result<Option<Vec<usize>>> {
        self.storage
            .scan_numeric_range_mmap(col_name, low, high, limit)
    }

    /// Mmap-level numeric IN scan: find rows where col_name IN (v1, v2, ...).
    pub fn scan_numeric_in_mmap(
        &self,
        col_name: &str,
        values: &[i64],
        limit: Option<usize>,
    ) -> io::Result<Option<Vec<usize>>> {
        self.storage.scan_numeric_in_mmap(col_name, values, limit)
    }

    /// Scan multiple predicates in parallel on a single shared mmap (one lock acquisition).
    pub fn scan_multi_predicates_parallel(
        &self,
        predicates: &[crate::storage::on_demand::MmapScanPred],
    ) -> io::Result<Option<Vec<usize>>> {
        self.storage.scan_multi_predicates_parallel(predicates)
    }

    /// Mmap-level boolean equality scan: find matching row indices without Arrow arrays.
    pub fn scan_bool_filter_mmap(
        &self,
        col_name: &str,
        target_value: bool,
        limit: Option<usize>,
    ) -> io::Result<Option<Vec<usize>>> {
        self.storage
            .scan_bool_filter_mmap(col_name, target_value, limit)
    }

    /// Direct mmap top-K scan: finds top-k row indices without materializing the full Arrow column.
    pub fn scan_top_k_indices_mmap(
        &self,
        col_name: &str,
        k: usize,
        descending: bool,
    ) -> io::Result<Option<Vec<(usize, f64)>>> {
        self.storage
            .scan_top_k_indices_mmap(col_name, k, descending)
    }

    /// Get underlying storage for direct access
    pub fn storage(&self) -> &OnDemandStorage {
        &self.storage
    }

    /// Read columns with combined STRING + NUMERIC filter and LIMIT early termination
    /// Optimized for SELECT * WHERE string_col = 'value' AND numeric_col > N LIMIT n
    pub fn read_columns_filtered_string_numeric_with_limit_to_arrow(
        &self,
        column_names: Option<&[&str]>,
        string_column: &str,
        string_value: &str,
        numeric_column: &str,
        numeric_op: &str,
        numeric_value: f64,
        limit: usize,
        offset: usize,
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        let (col_data, matching_indices) = self
            .storage
            .read_columns_filtered_string_numeric_with_limit(
                column_names,
                string_column,
                string_value,
                numeric_column,
                numeric_op,
                numeric_value,
                limit,
                offset,
            )?;

        if col_data.is_empty() || matching_indices.is_empty() {
            return self.read_columns_to_arrow(column_names, 0, Some(0));
        }

        let schema = self.schema.read();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        let include_id = column_names
            .map(|cols| cols.contains(&"_id"))
            .unwrap_or(true);
        if include_id {
            let filtered_ids = self.storage.read_ids_by_indices(&matching_indices)?;
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(filtered_ids)));
        }

        let col_order: Vec<String> = if let Some(names) = column_names {
            names
                .iter()
                .filter(|&s| *s != "_id")
                .map(|s| s.to_string())
                .collect()
        } else {
            schema.iter().map(|(n, _)| n.clone()).collect()
        };

        for col_name in &col_order {
            if let Some(data) = col_data.get(col_name) {
                let (arrow_dt, array): (ArrowDataType, ArrayRef) = match data {
                    ColumnData::Int64(values) => (
                        ArrowDataType::Int64,
                        Arc::new(Int64Array::from_iter_values(values.iter().copied())),
                    ),
                    ColumnData::Float64(values) => (
                        ArrowDataType::Float64,
                        Arc::new(Float64Array::from_iter_values(values.iter().copied())),
                    ),
                    ColumnData::String {
                        offsets,
                        data: bytes,
                    } => {
                        let count = offsets.len().saturating_sub(1);
                        let strings: Vec<Option<&str>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&bytes[start..end]).ok()
                            })
                            .collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    }
                    ColumnData::Bool { data: packed, len } => {
                        let bools: Vec<Option<bool>> = (0..*len)
                            .map(|i| {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(
                                    byte_idx < packed.len()
                                        && (packed[byte_idx] >> bit_idx) & 1 == 1,
                                )
                            })
                            .collect();
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    }
                    ColumnData::Binary {
                        offsets,
                        data: bytes,
                    } => {
                        use arrow::array::BinaryArray;
                        let count = offsets.len().saturating_sub(1);
                        let binary_data: Vec<Option<&[u8]>> = (0..count)
                            .map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&bytes[start..end])
                            })
                            .collect();
                        (
                            ArrowDataType::Binary,
                            Arc::new(BinaryArray::from(binary_data)),
                        )
                    }
                    ColumnData::StringDict {
                        indices,
                        dict_offsets,
                        dict_data,
                    } => {
                        // OPTIMIZED: Use Arrow DictionaryArray
                        use arrow::array::{DictionaryArray, UInt32Array};
                        use arrow::datatypes::UInt32Type;

                        let dict_count = dict_offsets.len().saturating_sub(1);
                        let dict_strings: Vec<Option<&str>> = (0..dict_count)
                            .map(|i| {
                                let start = dict_offsets[i] as usize;
                                let end = dict_offsets[i + 1] as usize;
                                std::str::from_utf8(&dict_data[start..end]).ok()
                            })
                            .collect();
                        let values = StringArray::from(dict_strings);

                        let keys: Vec<Option<u32>> = indices
                            .iter()
                            .map(|&idx| if idx == 0 { None } else { Some(idx - 1) })
                            .collect();
                        let keys_array = UInt32Array::from(keys);

                        let dict_array =
                            DictionaryArray::<UInt32Type>::try_new(keys_array, Arc::new(values))
                                .map_err(|e| {
                                    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
                                })?;

                        (
                            ArrowDataType::Dictionary(
                                Box::new(ArrowDataType::UInt32),
                                Box::new(ArrowDataType::Utf8),
                            ),
                            Arc::new(dict_array) as ArrayRef,
                        )
                    }
                    ColumnData::FixedList { data, dim } => fixedlist_to_arrow_pair(data, *dim),
                    ColumnData::Float16List { data, dim } => float16list_to_arrow_pair(data, *dim),
                };
                fields.push(Field::new(col_name, arrow_dt, true));
                arrays.push(array);
            }
        }

        let schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Get or lazily build cached string dictionary for a column
    pub fn get_or_build_dict_cache(
        &self,
        col_name: &str,
    ) -> io::Result<Option<(Vec<String>, Vec<u16>)>> {
        // Check cache first
        {
            let cache = self.dict_cache.read();
            if let Some(cached) = cache.get(col_name) {
                return Ok(Some(cached.clone()));
            }
        }
        // Build and cache
        if let Some((dict_strings, group_ids)) = self.storage.build_string_dict_cache(col_name)? {
            let result = (dict_strings.clone(), group_ids.clone());
            let mut cache = self.dict_cache.write();
            cache.insert(col_name.to_string(), (dict_strings, group_ids));
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Execute simple aggregation (no GROUP BY, no WHERE) directly on V4 columns
    pub fn execute_simple_agg(
        &self,
        agg_cols: &[&str],
    ) -> io::Result<Option<Vec<(i64, f64, f64, f64, bool)>>> {
        self.storage.execute_simple_agg(agg_cols)
    }

    /// Single-pass filtered string aggregation: scan string column and aggregate
    /// numeric columns in one sequential pass per row group.
    pub fn execute_filtered_string_agg_mmap(
        &self,
        filter_col: &str,
        target: &str,
        agg_cols: &[&str],
    ) -> io::Result<Option<Vec<(i64, f64, f64, f64, bool)>>> {
        self.storage
            .execute_filtered_string_agg_mmap(filter_col, target, agg_cols)
    }

    /// Single-pass filtered numeric aggregation: scan numeric predicate column and
    /// aggregate numeric columns without materializing matching rows.
    pub fn execute_filtered_numeric_agg_mmap(
        &self,
        filter_col: &str,
        low: f64,
        high: f64,
        agg_cols: &[&str],
    ) -> io::Result<Option<Vec<(i64, f64, f64, f64, bool)>>> {
        self.storage
            .execute_filtered_numeric_agg_mmap(filter_col, low, high, agg_cols)
    }

    /// Build cached string dictionary indices for a column
    pub fn build_string_dict_cache(
        &self,
        col_name: &str,
    ) -> io::Result<Option<(Vec<String>, Vec<u16>)>> {
        self.storage.build_string_dict_cache(col_name)
    }

    /// Returns true if the column has any NULL values in the in-memory store.
    pub fn column_has_nulls(&self, col_name: &str) -> bool {
        self.storage.column_has_nulls(col_name)
    }

    /// Fast COUNT(DISTINCT col) for string columns. Null-aware via bitmaps.
    pub fn count_distinct_string(&self, col_name: &str) -> io::Result<Option<i64>> {
        self.storage.count_distinct_string(col_name)
    }

    /// Fast COUNT(DISTINCT col) given a pre-built dict (from global cache). Null-aware via bitmaps.
    pub fn count_distinct_with_dict(
        &self,
        col_name: &str,
        dict_strings: &[String],
        group_ids: &[u16],
    ) -> io::Result<i64> {
        self.storage
            .count_distinct_with_dict(col_name, dict_strings, group_ids)
    }

    /// Fast top-k for ORDER BY (str_col, f64_col) without Arrow string conversion.
    /// Uses global dict cache for warm O(dict_size) rank lookup instead of rebuilding the dict.
    pub fn order_topk_str_float64(
        &self,
        str_col: &str,
        str_asc: bool,
        f64_col: &str,
        f64_asc: bool,
        k: usize,
        offset: usize,
    ) -> io::Result<Option<Vec<usize>>> {
        let dict = match get_global_dict_cache(self.path(), str_col, &self.storage)? {
            Some(d) => d,
            None => return Ok(None),
        };
        self.storage.order_topk_str_float64_with_dict(
            dict.0.as_slice(),
            dict.1.as_slice(),
            str_asc,
            f64_col,
            f64_asc,
            k,
            offset,
        )
    }

    /// Execute GROUP BY + aggregate using pre-built dict cache.
    /// Delegates directly to storage: uses in-memory path when data is loaded,
    /// otherwise uses mmap path reading only the needed agg columns via RCIX.
    pub fn execute_group_agg_cached(
        &self,
        dict_strings: &[String],
        group_ids: &[u16],
        agg_cols: &[(&str, bool)],
    ) -> io::Result<Option<Vec<(String, Vec<(f64, i64)>)>>> {
        self.storage
            .execute_group_agg_cached(dict_strings, group_ids, agg_cols)
    }

    /// Execute 2-column GROUP BY + aggregate using pre-built dict caches.
    pub fn execute_group_agg_2col_cached(
        &self,
        dict1_strings: &[String],
        group_ids1: &[u16],
        dict2_strings: &[String],
        group_ids2: &[u16],
        agg_cols: &[(&str, bool)],
    ) -> io::Result<Option<Vec<((String, String), Vec<(f64, i64)>)>>> {
        self.storage.execute_group_agg_2col_cached(
            dict1_strings,
            group_ids1,
            dict2_strings,
            group_ids2,
            agg_cols,
        )
    }

    /// Execute BETWEEN + GROUP BY using pre-built dict cache.
    /// Delegates directly to storage: uses mmap path reading only the needed columns.
    pub fn execute_between_group_agg_cached(
        &self,
        filter_col: &str,
        lo: f64,
        hi: f64,
        dict_strings: &[String],
        group_ids: &[u16],
        agg_col: Option<&str>,
    ) -> io::Result<Option<Vec<(String, f64, i64)>>> {
        self.storage.execute_between_group_agg_cached(
            filter_col,
            lo,
            hi,
            dict_strings,
            group_ids,
            agg_col,
        )
    }

    /// Execute BETWEEN + GROUP BY + aggregate directly on V4 columns
    pub fn execute_between_group_agg(
        &self,
        filter_col: &str,
        lo: f64,
        hi: f64,
        group_col: &str,
        agg_col: Option<&str>,
    ) -> io::Result<Option<Vec<(String, f64, i64)>>> {
        self.storage
            .execute_between_group_agg(filter_col, lo, hi, group_col, agg_col)
    }

    /// Execute GROUP BY + aggregate directly on V4 columns (no WHERE)
    pub fn execute_group_agg(
        &self,
        group_col: &str,
        agg_cols: &[(&str, bool)],
    ) -> io::Result<Option<Vec<(String, Vec<(f64, i64)>)>>> {
        self.storage.execute_group_agg(group_col, agg_cols)
    }

    /// Execute Complex (Filter+Group+Order) query with single-pass optimization
    /// This is the key optimization for queries like:
    /// SELECT region, SUM(value) FROM table WHERE status = 'active' GROUP BY region ORDER BY total DESC LIMIT 5
    pub fn execute_filter_group_order(
        &self,
        filter_col: &str,
        filter_val: &str,
        group_col: &str,
        agg_col: Option<&str>,
        agg_func: crate::query::AggregateFunc,
        _order_col: &str,
        descending: bool,
        limit: usize,
        offset: usize,
    ) -> io::Result<Option<RecordBatch>> {
        use crate::query::AggregateFunc;
        use arrow::array::{DictionaryArray, Float64Array, Int64Array, StringArray, UInt32Array};
        use arrow::datatypes::UInt32Type;
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        // This optimization requires dictionary-encoded columns for maximum performance
        // Get filter column info
        let schema_guard = self.schema.read();
        let filter_idx = schema_guard.iter().position(|(name, _)| name == filter_col);
        let group_idx = schema_guard.iter().position(|(name, _)| name == group_col);

        if filter_idx.is_none() || group_idx.is_none() {
            return Ok(None);
        }

        // For now, delegate to the OnDemandStorage implementation
        let result = self.storage.execute_filter_group_order(
            filter_col, filter_val, group_col, agg_col, agg_func, descending, limit, offset,
        )?;

        Ok(result)
    }
}

// ============================================================================
// Incremental Storage Backend - Fast Writes with WAL
// ============================================================================

use crate::storage::incremental::IncrementalStorage;
use crate::storage::on_demand::ColumnValue as OnDemandColumnValue;

/// High-performance storage backend with incremental writes
///
/// Uses WAL (Write-Ahead Log) for fast append-only writes:
/// - Writes append to WAL file - O(1) time
/// - Reads merge main file + WAL transparently
/// - Background compaction merges WAL into main file
///
/// This provides significantly faster write performance compared to
/// TableStorageBackend which rewrites the entire file on each save.
pub struct IncrementalStorageBackend {
    storage: IncrementalStorage,
    /// Schema mapping (column_name -> DataType)
    schema: RwLock<Vec<(String, DataType)>>,
    /// Fast lookup for known column names (avoids O(n) scan per insert)
    known_columns: RwLock<HashSet<String>>,
}

impl IncrementalStorageBackend {
    /// Create a new incremental storage
    pub fn create(path: &Path) -> io::Result<Self> {
        let storage = IncrementalStorage::create(path)?;
        Ok(Self {
            storage,
            schema: RwLock::new(Vec::new()),
            known_columns: RwLock::new(HashSet::new()),
        })
    }

    /// Open existing incremental storage
    pub fn open(path: &Path) -> io::Result<Self> {
        let storage = IncrementalStorage::open(path)?;
        let schema: Vec<(String, DataType)> = storage
            .get_schema()
            .into_iter()
            .map(|(name, ct)| (name, column_type_to_datatype(ct)))
            .collect();
        let known: HashSet<String> = schema.iter().map(|(n, _)| n.clone()).collect();

        Ok(Self {
            storage,
            schema: RwLock::new(schema),
            known_columns: RwLock::new(known),
        })
    }

    /// Open or create
    pub fn open_or_create(path: &Path) -> io::Result<Self> {
        if path.exists() {
            Self::open(path)
        } else {
            Self::create(path)
        }
    }

    /// Get row count
    pub fn row_count(&self) -> u64 {
        self.storage.row_count()
    }

    /// Get schema
    pub fn get_schema(&self) -> Vec<(String, DataType)> {
        self.schema.read().clone()
    }

    /// Insert rows - FAST incremental write
    pub fn insert_rows(&self, rows: &[HashMap<String, Value>]) -> io::Result<Vec<u64>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        // Convert Value to ColumnValue
        let converted: Vec<HashMap<String, OnDemandColumnValue>> = rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|(k, v)| {
                        let cv = match v {
                            Value::Int64(i) => OnDemandColumnValue::Int64(*i),
                            Value::Int32(i) => OnDemandColumnValue::Int64(*i as i64),
                            Value::Float64(f) => OnDemandColumnValue::Float64(*f),
                            Value::Float32(f) => OnDemandColumnValue::Float64(*f as f64),
                            Value::String(s) => OnDemandColumnValue::String(s.clone()),
                            Value::Bool(b) => OnDemandColumnValue::Bool(*b),
                            Value::Binary(b) => OnDemandColumnValue::Binary(b.clone()),
                            Value::FixedList(b) => OnDemandColumnValue::FixedList(b.clone()),
                            Value::Null => OnDemandColumnValue::Null,
                            _ => OnDemandColumnValue::String(
                                serde_json::to_string(v).unwrap_or_default(),
                            ),
                        };
                        (k.clone(), cv)
                    })
                    .collect()
            })
            .collect();

        // Insert into storage (fast WAL append)
        let ids = self.storage.insert_rows(&converted)?;

        // Update schema if new columns (O(1) HashSet lookup instead of O(n) scan)
        if let Some(row) = rows.first() {
            let known = self.known_columns.read();
            let has_new = row.keys().any(|k| k != "_id" && !known.contains(k));
            drop(known);
            if has_new {
                let mut schema = self.schema.write();
                let mut known = self.known_columns.write();
                for (k, v) in row {
                    if k != "_id" && !known.contains(k) {
                        schema.push((k.clone(), v.data_type()));
                        known.insert(k.clone());
                    }
                }
            }
        }

        Ok(ids)
    }

    /// Delete row by ID
    pub fn delete(&self, id: u64) -> io::Result<bool> {
        self.storage.delete(id)
    }

    /// Save/compact (merge WAL into main file)
    pub fn save(&self) -> io::Result<()> {
        self.storage.save()
    }

    /// Flush WAL to disk (without compaction)
    pub fn flush(&self) -> io::Result<()> {
        self.storage.flush()
    }

    /// Check if compaction is needed
    pub fn needs_compaction(&self) -> bool {
        self.storage.needs_compaction()
    }

    /// Compact WAL into main file
    pub fn compact(&self) -> io::Result<()> {
        self.storage.compact()
    }

    /// Get WAL record count
    pub fn wal_record_count(&self) -> usize {
        self.storage.wal_record_count()
    }

    /// Close storage
    pub fn close(&self) -> io::Result<()> {
        self.storage.close()
    }
}

// ============================================================================
// Multi-Table Storage Manager
// ============================================================================

/// Manages multiple tables with lazy loading
pub struct StorageManager {
    base_dir: PathBuf,
    /// Table backends (table_name -> backend)
    tables: RwLock<HashMap<String, Arc<TableStorageBackend>>>,
    /// Current table name
    current_table: RwLock<String>,
}

impl StorageManager {
    /// Create or open a storage manager
    pub fn open_or_create(base_path: &Path) -> io::Result<Self> {
        let base_dir = base_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));

        let mut tables = HashMap::new();

        // Check if main file exists
        if base_path.exists() {
            // Load existing table
            let backend = TableStorageBackend::open(base_path)?;
            let name = backend.metadata().name;
            tables.insert(name.clone(), Arc::new(backend));
        }

        Ok(Self {
            base_dir,
            tables: RwLock::new(tables),
            current_table: RwLock::new("default".to_string()),
        })
    }

    /// Get or create a table
    pub fn get_or_create_table(&self, name: &str) -> io::Result<Arc<TableStorageBackend>> {
        // Check if already loaded
        if let Some(backend) = self.tables.read().get(name).cloned() {
            return Ok(backend);
        }

        // Create new table
        let path = self.base_dir.join(format!("{}.apex", name));
        let backend = Arc::new(TableStorageBackend::open_or_create(&path)?);

        self.tables
            .write()
            .insert(name.to_string(), backend.clone());

        Ok(backend)
    }

    /// Get current table
    pub fn current_table(&self) -> io::Result<Arc<TableStorageBackend>> {
        let name = self.current_table.read().clone();
        self.get_or_create_table(&name)
    }

    /// Set current table
    pub fn set_current_table(&self, name: &str) {
        *self.current_table.write() = name.to_string();
    }

    /// List all tables
    pub fn list_tables(&self) -> Vec<String> {
        self.tables.read().keys().cloned().collect()
    }

    /// Save all tables
    pub fn save_all(&self) -> io::Result<()> {
        for (_, backend) in self.tables.read().iter() {
            backend.save()?;
        }
        Ok(())
    }

    /// Release memory from all tables
    pub fn release_all_memory(&self) {
        for (_, backend) in self.tables.read().iter() {
            backend.release_all_columns();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_backend_bool_null() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_bool_null.apex");

        // Create and insert with NULL boolean - mimics Python client flow
        {
            let backend = TableStorageBackend::create(&path).unwrap();

            let mut row1 = HashMap::new();
            row1.insert("id".to_string(), Value::Int64(1));
            row1.insert("flag".to_string(), Value::Bool(true));

            let mut row2 = HashMap::new();
            row2.insert("id".to_string(), Value::Int64(2));
            row2.insert("flag".to_string(), Value::Bool(false));

            let mut row3 = HashMap::new();
            row3.insert("id".to_string(), Value::Int64(3));
            row3.insert("flag".to_string(), Value::Null); // NULL boolean

            let ids = backend.insert_rows(&[row1, row2, row3]).unwrap();
            assert_eq!(ids.len(), 3);

            backend.save().unwrap();
        }

        // Reopen and check null mask
        {
            let backend = TableStorageBackend::open(&path).unwrap();

            // Check null mask via storage
            let null_mask = backend.storage.get_null_mask("flag", 0, 3);
            println!("Null mask via backend: {:?}", null_mask);
            assert_eq!(null_mask, vec![false, false, true], "Row 2 should be NULL");

            // Check via Arrow conversion
            let batch = backend.read_columns_to_arrow(None, 0, None).unwrap();
            println!("Arrow batch schema: {:?}", batch.schema());

            // Find flag column and check nulls
            let flag_idx = batch.schema().index_of("flag").unwrap();
            let flag_col = batch.column(flag_idx);
            println!("Flag column null count: {}", flag_col.null_count());
            assert_eq!(flag_col.null_count(), 1, "Should have 1 null value");
        }
    }

    #[test]
    fn test_backend_create_and_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apex");

        // Create and insert
        {
            let backend = TableStorageBackend::create(&path).unwrap();

            let mut row = HashMap::new();
            row.insert("name".to_string(), Value::String("Alice".to_string()));
            row.insert("age".to_string(), Value::Int64(30));

            let ids = backend.insert_rows(&[row]).unwrap();
            assert_eq!(ids.len(), 1);

            backend.save().unwrap();
        }

        // Reopen and verify (lazy load)
        {
            let backend = TableStorageBackend::open(&path).unwrap();

            // Metadata available without loading data
            assert_eq!(backend.row_count(), 1);

            // Cache should be empty
            assert!(backend.get_cached_column("name").is_none());

            // Load specific column
            backend.load_columns(&["name"]).unwrap();
            assert!(backend.get_cached_column("name").is_some());
        }
    }

    #[test]
    fn test_column_type_conversions() {
        // Test Int64
        let mut col = TypedColumn::Int64 {
            data: vec![1, 2, 3],
            nulls: BitVec::new(),
        };
        if let TypedColumn::Int64 { nulls, .. } = &mut col {
            nulls.extend_false(3);
        }

        let cd = typed_column_to_column_data(&col);
        let back = column_data_to_typed_column(&cd, DataType::Int64);

        if let TypedColumn::Int64 { data, .. } = back {
            assert_eq!(data, vec![1, 2, 3]);
        } else {
            panic!("Expected Int64 column");
        }
    }

    #[test]
    fn test_insert_typed_and_reload() {
        use crate::storage::OnDemandStorage;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test_typed.apex");

        // Save using insert_typed (like save_to_v3 does)
        {
            let storage = OnDemandStorage::create(&path).unwrap();

            let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
            let mut string_cols: HashMap<String, Vec<String>> = HashMap::new();

            int_cols.insert("age".to_string(), vec![30, 25]);
            string_cols.insert(
                "name".to_string(),
                vec!["Alice".to_string(), "Bob".to_string()],
            );

            let ids = storage
                .insert_typed(
                    int_cols,
                    HashMap::new(), // float
                    string_cols,
                    HashMap::new(), // binary
                    HashMap::new(), // bool
                )
                .unwrap();

            assert_eq!(ids.len(), 2);
            assert_eq!(storage.row_count(), 2);

            storage.save().unwrap();
        }

        // Reopen and verify with backend
        {
            let backend = TableStorageBackend::open(&path).unwrap();

            // Check metadata
            let schema = backend.get_schema();
            println!("Schema after reopen: {:?}", schema);
            assert!(!schema.is_empty(), "Schema should not be empty");

            let row_count = backend.row_count();
            println!("Row count after reopen: {}", row_count);
            assert_eq!(row_count, 2, "Should have 2 rows");

            // Load all columns
            backend.load_all_columns().unwrap();

            // Check cached columns
            let name_col = backend.get_cached_column("name");
            println!("Name column: {:?}", name_col.is_some());
            assert!(name_col.is_some(), "Name column should be loaded");

            let age_col = backend.get_cached_column("age");
            println!("Age column: {:?}", age_col.is_some());
            assert!(age_col.is_some(), "Age column should be loaded");
        }
    }

    #[test]
    fn test_read_columns_to_arrow() {
        use crate::storage::OnDemandStorage;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test_arrow.apex");

        // Create test data
        {
            let storage = OnDemandStorage::create(&path).unwrap();

            let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
            let mut float_cols: HashMap<String, Vec<f64>> = HashMap::new();
            let mut string_cols: HashMap<String, Vec<String>> = HashMap::new();

            int_cols.insert("age".to_string(), vec![30, 25, 35]);
            float_cols.insert("score".to_string(), vec![85.5, 90.0, 78.5]);
            string_cols.insert(
                "name".to_string(),
                vec![
                    "Alice".to_string(),
                    "Bob".to_string(),
                    "Charlie".to_string(),
                ],
            );

            storage
                .insert_typed(
                    int_cols,
                    float_cols,
                    string_cols,
                    HashMap::new(),
                    HashMap::new(),
                )
                .unwrap();
            storage.save().unwrap();
        }

        // Test read_columns_to_arrow
        {
            let backend = TableStorageBackend::open(&path).unwrap();

            // Read all columns (age, score, name + auto-generated _id = 4)
            let batch = backend.read_columns_to_arrow(None, 0, None).unwrap();
            assert_eq!(batch.num_rows(), 3);
            assert_eq!(batch.num_columns(), 4);

            // Read specific columns (column projection)
            let batch2 = backend
                .read_columns_to_arrow(Some(&["name", "age"]), 0, None)
                .unwrap();
            assert_eq!(batch2.num_rows(), 3);
            assert_eq!(batch2.num_columns(), 2);

            // Read with row limit
            let batch3 = backend.read_columns_to_arrow(None, 0, Some(2)).unwrap();
            assert_eq!(batch3.num_rows(), 2);

            // Read single column with limit
            let batch4 = backend
                .read_columns_to_arrow(Some(&["name"]), 0, Some(1))
                .unwrap();
            assert_eq!(batch4.num_rows(), 1);
            assert_eq!(batch4.num_columns(), 1);
        }
    }

    #[test]
    fn test_read_columns_to_arrow_internal_id_with_projection_range() {
        use arrow::array::{Int64Array, StringArray};

        let dir = tempdir().unwrap();
        let path = dir.path().join("test_arrow_id_projection.apex");

        let backend = TableStorageBackend::create(&path).unwrap();
        let mut rows = Vec::new();
        for name in ["Alice", "Bob", "Charlie"] {
            let mut row = HashMap::new();
            row.insert("name".to_string(), Value::String(name.to_string()));
            rows.push(row);
        }
        backend.insert_rows(&rows).unwrap();

        let batch = backend
            .read_columns_to_arrow(Some(&["_id", "name"]), 1, Some(1))
            .unwrap();

        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.schema().field(0).name(), "_id");
        assert_eq!(batch.schema().field(1).name(), "name");

        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let names = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(ids.value(0), 2);
        assert_eq!(names.value(0), "Bob");
    }

    #[test]
    fn test_fixedlist_to_arrow_pair_uses_aligned_float_values() {
        use arrow::array::{Array, FixedSizeListArray, Float32Array};

        let raw: Vec<u8> = [1.0f32, 0.0, 0.5, 0.25, 0.75, 1.0]
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let (_dt, arr) = fixedlist_to_arrow_pair(&raw, 3);
        let list = arr.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        let values = list
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();

        assert_eq!(list.len(), 2);
        assert_eq!(list.value_length(), 3);
        assert_eq!(values.value(0), 1.0);
        assert_eq!(values.value(5), 1.0);
    }

    #[test]
    fn test_insert_rows_preserves_fixedlist_for_float16_schema() {
        use arrow::array::{Array, FixedSizeListArray, Float32Array};

        let dir = tempdir().unwrap();
        let path = dir.path().join("test_f16_fixedlist_insert.apex");
        let backend = TableStorageBackend::create(&path).unwrap();
        backend
            .add_column("vec", crate::data::DataType::Float16Vector)
            .unwrap();

        let raw: Vec<u8> = [0.10f32, 0.82, 0.20]
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut row = HashMap::new();
        row.insert("vec".to_string(), Value::FixedList(raw));
        backend.insert_rows(&[row]).unwrap();

        let batch = backend
            .read_columns_to_arrow(Some(&["vec"]), 0, None)
            .unwrap();
        assert_eq!(batch.num_rows(), 1);

        let list = batch
            .column(0)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        let values = list
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();

        assert_eq!(list.len(), 1);
        assert_eq!(list.value_length(), 3);
        assert!((values.value(0) - 0.10).abs() < 2e-3);
    }

    #[test]
    fn test_column_projection_correctness() {
        use crate::storage::OnDemandStorage;
        use arrow::array::{Int64Array, StringArray};

        let dir = tempdir().unwrap();
        let path = dir.path().join("test_proj.apex");

        // Create test data
        {
            let storage = OnDemandStorage::create(&path).unwrap();

            let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
            let mut string_cols: HashMap<String, Vec<String>> = HashMap::new();

            int_cols.insert("id".to_string(), vec![1, 2, 3]);
            int_cols.insert("value".to_string(), vec![100, 200, 300]);
            string_cols.insert(
                "label".to_string(),
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
            );

            storage
                .insert_typed(
                    int_cols,
                    HashMap::new(),
                    string_cols,
                    HashMap::new(),
                    HashMap::new(),
                )
                .unwrap();
            storage.save().unwrap();
        }

        // Verify column projection returns correct data
        {
            let backend = TableStorageBackend::open(&path).unwrap();

            // Read only 'id' and 'label' columns
            let batch = backend
                .read_columns_to_arrow(Some(&["id", "label"]), 0, None)
                .unwrap();

            assert_eq!(batch.num_columns(), 2);

            // Verify column names
            let schema = batch.schema();
            let field_names: Vec<&str> =
                schema.fields().iter().map(|f| f.name().as_str()).collect();
            assert!(field_names.contains(&"id"));
            assert!(field_names.contains(&"label"));
            assert!(!field_names.contains(&"value")); // Should NOT include 'value'
        }
    }

    #[test]
    fn test_row_range_scan() {
        use crate::storage::OnDemandStorage;
        use arrow::array::Int64Array;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test_range.apex");

        // Create test data with 10 rows
        {
            let storage = OnDemandStorage::create(&path).unwrap();

            let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
            int_cols.insert("index".to_string(), (0..10).collect());

            storage
                .insert_typed(
                    int_cols,
                    HashMap::new(),
                    HashMap::new(),
                    HashMap::new(),
                    HashMap::new(),
                )
                .unwrap();
            storage.save().unwrap();
        }

        // Test row range scanning
        {
            let backend = TableStorageBackend::open(&path).unwrap();

            // Read first 3 rows
            let batch1 = backend.read_columns_to_arrow(None, 0, Some(3)).unwrap();
            assert_eq!(batch1.num_rows(), 3);

            // Read middle 4 rows (rows 3-6)
            let batch2 = backend.read_columns_to_arrow(None, 3, Some(4)).unwrap();
            assert_eq!(batch2.num_rows(), 4);

            // Read last 2 rows
            let batch3 = backend.read_columns_to_arrow(None, 8, Some(10)).unwrap();
            assert_eq!(batch3.num_rows(), 2); // Only 2 rows left
        }
    }

    #[test]
    fn test_progressive_schema() {
        use crate::data::Value;
        use arrow::array::StringArray;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test_progressive.apex");

        // Step 1: Create table and insert first row with column 'a'
        {
            let backend = TableStorageBackend::create(&path).unwrap();
            let mut row = HashMap::new();
            row.insert("a".to_string(), Value::Int64(1));
            backend.insert_rows(&[row]).unwrap();
            backend.save().unwrap();

            println!("After first insert:");
            let batch = backend.read_columns_to_arrow(None, 0, None).unwrap();
            println!("  num_rows: {}", batch.num_rows());
            println!(
                "  schema: {:?}",
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .collect::<Vec<_>>()
            );
        }

        // Step 2: Reopen and insert second row with columns 'a' and 'b'
        {
            let backend = TableStorageBackend::open_for_write(&path).unwrap();

            println!("After reopen, before second insert:");

            let mut row = HashMap::new();
            row.insert("a".to_string(), Value::Int64(2));
            row.insert("b".to_string(), Value::String("hello".to_string()));
            backend.insert_rows(&[row]).unwrap();
            backend.save().unwrap();
        }

        // Step 3: Read back and verify
        {
            let backend = TableStorageBackend::open(&path).unwrap();
            let batch = backend.read_columns_to_arrow(None, 0, None).unwrap();

            println!("Final result:");
            println!("  num_rows: {}", batch.num_rows());

            // Check each row
            let a_col = batch.column_by_name("a").unwrap();
            let a_arr = a_col
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .unwrap();

            let b_col = batch.column_by_name("b");

            for i in 0..batch.num_rows() {
                let a_val = a_arr.value(i);
                let b_val = if let Some(col) = b_col {
                    let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
                    arr.value(i).to_string()
                } else {
                    "N/A".to_string()
                };
                println!("  Row {}: a={}, b={}", i, a_val, b_val);
            }

            // Row 0 should be a=1, b=NULL/empty
            assert_eq!(a_arr.value(0), 1);
            if let Some(col) = b_col {
                let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
                assert!(arr.value(0).is_empty(), "Row 0 should have empty b");
            }

            // Row 1 should be a=2, b='hello'
            assert_eq!(a_arr.value(1), 2);
            if let Some(col) = b_col {
                let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
                assert_eq!(arr.value(1), "hello", "Row 1 should have b='hello'");
            }
        }
    }

    #[test]
    fn test_alter_then_insert() {
        use crate::data::{DataType, Value};
        use arrow::array::StringArray;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test_alter_insert.apex");

        // Step 1: Create empty table and add columns via add_column_with_padding
        {
            let backend = TableStorageBackend::create(&path).unwrap();

            // Add columns to empty table (simulates ALTER TABLE ADD COLUMN)
            backend.add_column("name", DataType::String).unwrap();
            backend.add_column("value", DataType::Int64).unwrap();

            println!("After add_column:");
            println!("  Schema: {:?}", backend.list_columns());

            backend.save().unwrap();
        }

        // Step 2: Reopen and insert data
        {
            let backend = TableStorageBackend::open_for_write(&path).unwrap();

            println!("After reopen:");
            println!("  Schema: {:?}", backend.list_columns());

            // Insert a row
            let mut row = HashMap::new();
            row.insert("name".to_string(), Value::String("Test".to_string()));
            row.insert("value".to_string(), Value::Int64(100));

            backend.insert_rows(&[row]).unwrap();
            backend.save().unwrap();
        }

        // Step 3: Read back and verify
        {
            let backend = TableStorageBackend::open(&path).unwrap();
            let batch = backend.read_columns_to_arrow(None, 0, None).unwrap();

            println!("Result batch:");
            println!("  num_rows: {}", batch.num_rows());
            println!("  schema: {:?}", batch.schema());

            // Check name column
            if let Some(name_col) = batch.column_by_name("name") {
                let arr = name_col.as_any().downcast_ref::<StringArray>().unwrap();
                println!("  name[0]: {:?}", arr.value(0));
                assert_eq!(arr.value(0), "Test", "Name should be 'Test'");
            } else {
                panic!("Name column not found");
            }

            // Check value column
            if let Some(value_col) = batch.column_by_name("value") {
                let arr = value_col
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .unwrap();
                println!("  value[0]: {:?}", arr.value(0));
                assert_eq!(arr.value(0), 100, "Value should be 100");
            }
        }
    }
}
