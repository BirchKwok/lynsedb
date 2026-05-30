//! PyO3 bindings - On-demand storage engine
//!
//! This module provides Python bindings that use on-demand storage directly,
//! enabling on-demand reading without loading entire tables into memory.

use crate::data::Value;
use crate::fts::FtsConfig;
use crate::fts::FtsManager;
use crate::query::{ApexExecutor, ApexResult, SqlParser};
use crate::storage::on_demand::ColumnValue;
use crate::storage::{DurabilityLevel, StorageEngine, StorageManager, TableStorageBackend};
use arrow::record_batch::RecordBatch;
use dashmap::DashMap;
use fs2::FileExt;
use parking_lot::RwLock;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Convert Python dict to HashMap<String, Value>
fn dict_to_values(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, Value>> {
    let mut fields = HashMap::with_capacity(dict.len());

    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        if key == "_id" {
            continue;
        }
        let val = py_to_value(&value)?;
        fields.insert(key, val);
    }

    Ok(fields)
}

#[inline]
fn sort_and_dedupe_ids(ids: &[u64]) -> Vec<u64> {
    if ids.len() < 2 {
        return ids.to_vec();
    }

    let mut sorted_ids = ids.to_vec();
    sorted_ids.sort_unstable();
    sorted_ids.dedup();
    sorted_ids
}

#[inline]
fn next_up_f64_binding(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

#[inline]
fn next_down_f64_binding(value: f64) -> f64 {
    -next_up_f64_binding(-value)
}

/// Parse aggregate expressions from SELECT clause: "SELECT COUNT(*) as cnt, AVG(score)"
/// Returns Vec of (function_name, optional_column_name, optional_alias)
fn parse_agg_select(sql: &str) -> Option<Vec<(String, Option<String>, Option<String>)>> {
    let upper = sql.to_ascii_uppercase();
    let select_start = upper.find("SELECT")? + 6;
    let from_pos = upper[select_start..].find(" FROM")?;
    let select_clause = &sql[select_start..select_start + from_pos];
    let mut result = Vec::new();
    for part in select_clause.split(',') {
        let part = part.trim();
        let part_upper = part.to_ascii_uppercase();
        if let Some(lp) = part_upper.find('(') {
            let func_name = part_upper[..lp].trim().to_string();
            if matches!(func_name.as_str(), "COUNT" | "SUM" | "AVG" | "MIN" | "MAX") {
                // Find closing paren
                let after_lp = &part[lp + 1..];
                if let Some(rp) = after_lp.find(')') {
                    let inner = after_lp[..rp].trim();
                    let col = if inner == "*" || inner.is_empty() {
                        None
                    } else {
                        Some(inner.trim_matches('"').to_string())
                    };
                    // Check for alias after closing paren
                    let after_paren = after_lp[rp + 1..].trim();
                    let alias = if after_paren.to_ascii_uppercase().starts_with("AS ") {
                        Some(after_paren[3..].trim().trim_matches('"').to_string())
                    } else if !after_paren.is_empty() && !after_paren.starts_with(',') {
                        Some(after_paren.trim().trim_matches('"').to_string())
                    } else {
                        None
                    };
                    result.push((func_name, col, alias));
                }
            }
        }
    }
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Compute sum/min/max from an Arrow array (Int64 or Float64)
fn agg_array_stats(arr: &dyn arrow::array::Array) -> (f64, f64, f64, bool) {
    use arrow::array::{Float64Array, Int64Array};
    if let Some(ia) = arr.as_any().downcast_ref::<Int64Array>() {
        let sum: i64 = ia.iter().flatten().sum();
        let min = ia.iter().flatten().min().unwrap_or(i64::MAX);
        let max = ia.iter().flatten().max().unwrap_or(i64::MIN);
        (sum as f64, min as f64, max as f64, true)
    } else if let Some(fa) = arr.as_any().downcast_ref::<Float64Array>() {
        let sum: f64 = fa.iter().flatten().sum();
        let min = fa.iter().flatten().fold(f64::INFINITY, f64::min);
        let max = fa.iter().flatten().fold(f64::NEG_INFINITY, f64::max);
        (sum, min, max, false)
    } else {
        (0.0, 0.0, 0.0, false)
    }
}

/// Convert Python value to Value
fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    use pyo3::types::PyBytes;

    if obj.is_none() {
        return Ok(Value::Null);
    }

    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }

    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Int64(i));
    }

    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::Float64(f));
    }

    // Check for bytes BEFORE string (bytes can be extracted as string)
    if obj.is_instance_of::<PyBytes>() {
        if let Ok(bytes) = obj.extract::<Vec<u8>>() {
            return Ok(Value::Binary(bytes));
        }
    }

    // numpy ndarray (1-D float32 or float64) → FixedList (raw LE f32 bytes)
    // Checked BEFORE list/sequence to catch np.ndarray first.
    if obj
        .get_type()
        .name()
        .map(|n| n == "ndarray")
        .unwrap_or(false)
    {
        if let Ok(floats) = obj
            .call_method0("flatten")
            .and_then(|flat| flat.call_method1("astype", ("float32",)))
            .and_then(|f32arr| f32arr.call_method0("tobytes"))
            .and_then(|b| b.extract::<Vec<u8>>())
        {
            if floats.len() % 4 == 0 {
                return Ok(Value::FixedList(floats));
            }
        }
    }

    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }

    // Python list/tuple of numbers → FixedList (raw LE f32 bytes), matching numpy vectors.
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        if let Ok(values) = obj.extract::<Vec<f32>>() {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            return Ok(Value::FixedList(bytes));
        }
    }

    if let Ok(bytes) = obj.extract::<Vec<u8>>() {
        return Ok(Value::Binary(bytes));
    }

    Ok(Value::Null)
}

fn py_to_column_value(obj: &Bound<'_, PyAny>) -> PyResult<ColumnValue> {
    use pyo3::types::PyBytes;

    if obj.is_none() {
        return Ok(ColumnValue::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(ColumnValue::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(ColumnValue::Int64(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(ColumnValue::Float64(f));
    }
    if obj.is_instance_of::<PyBytes>() {
        if let Ok(bytes) = obj.extract::<Vec<u8>>() {
            return Ok(ColumnValue::Binary(bytes));
        }
    }
    if obj
        .get_type()
        .name()
        .map(|n| n == "ndarray")
        .unwrap_or(false)
    {
        if let Ok(floats) = obj
            .call_method0("flatten")
            .and_then(|flat| flat.call_method1("astype", ("float32",)))
            .and_then(|f32arr| f32arr.call_method0("tobytes"))
            .and_then(|b| b.extract::<Vec<u8>>())
        {
            if floats.len() % 4 == 0 {
                return Ok(ColumnValue::FixedList(floats));
            }
        }
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(ColumnValue::String(s));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        if let Ok(values) = obj.extract::<Vec<f32>>() {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            return Ok(ColumnValue::FixedList(bytes));
        }
    }
    Ok(ColumnValue::Null)
}

fn dict_to_column_values(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, ColumnValue>> {
    let mut fields = HashMap::with_capacity(dict.len());

    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        if key == "_id" {
            continue;
        }
        fields.insert(key, py_to_column_value(&value)?);
    }

    Ok(fields)
}

/// Convert ColumnValue to Python object
#[allow(dead_code)]
fn column_value_to_py(py: Python<'_>, val: &ColumnValue) -> PyResult<PyObject> {
    match val {
        ColumnValue::Null => Ok(py.None()),
        ColumnValue::Bool(b) => Ok(b.into_py(py)),
        ColumnValue::Int64(i) => Ok(i.into_py(py)),
        ColumnValue::Float64(f) => Ok(f.into_py(py)),
        ColumnValue::String(s) => Ok(s.into_py(py)),
        ColumnValue::Binary(b) => Ok(b.clone().into_py(py)),
        ColumnValue::FixedList(b) => Ok(b.clone().into_py(py)),
    }
}

/// Convert Value to Python object
fn value_to_py(py: Python<'_>, val: &Value) -> PyResult<PyObject> {
    use pyo3::types::PyBytes;

    match val {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.into_py(py)),
        Value::Int8(i) => Ok((*i as i64).into_py(py)),
        Value::Int16(i) => Ok((*i as i64).into_py(py)),
        Value::Int32(i) => Ok((*i as i64).into_py(py)),
        Value::Int64(i) => Ok(i.into_py(py)),
        Value::UInt8(i) => Ok((*i as i64).into_py(py)),
        Value::UInt16(i) => Ok((*i as i64).into_py(py)),
        Value::UInt32(i) => Ok((*i as i64).into_py(py)),
        Value::UInt64(i) => Ok((*i as i64).into_py(py)),
        Value::Float32(f) => Ok((*f as f64).into_py(py)),
        Value::Float64(f) => Ok(f.into_py(py)),
        Value::String(s) => Ok(s.into_py(py)),
        Value::Binary(b) => Ok(PyBytes::new_bound(py, b).into()),
        Value::FixedList(b) => Ok(PyBytes::new_bound(py, b).into()),
        Value::Json(j) => Ok(j.to_string().into_py(py)),
        Value::Timestamp(t) => Ok(t.into_py(py)),
        Value::Date(d) => Ok(d.into_py(py)),
        Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for v in arr {
                list.append(value_to_py(py, v)?)?;
            }
            Ok(list.into())
        }
    }
}

fn values_to_columns_dict<'py>(
    py: Python<'py>,
    vals: &[(String, Value)],
) -> PyResult<Bound<'py, PyDict>> {
    let columns_dict = PyDict::new_bound(py);
    for (col_name, val) in vals {
        let pyval = value_to_py(py, val)?;
        columns_dict.set_item(col_name.as_str(), PyList::new_bound(py, [pyval]))?;
    }
    Ok(columns_dict)
}

fn mmap_batch_columns_to_pydict<'py>(
    py: Python<'py>,
    batch: crate::storage::on_demand::MmapBatchColumns,
    requested: Option<&[String]>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    use crate::storage::on_demand::MmapBatchColumn;
    use pyo3::types::{PyBytes, PyList};

    let columns_dict = PyDict::new_bound(py);
    if batch.row_count == 0 {
        return Ok(Some(columns_dict));
    }

    let mut columns = batch.columns;
    let emit_column = |name: String, col: MmapBatchColumn| -> PyResult<()> {
        match col {
            MmapBatchColumn::I64(vals) => {
                columns_dict.set_item(name.as_str(), PyList::new_bound(py, vals))?;
            }
            MmapBatchColumn::F64(vals) => {
                columns_dict.set_item(name.as_str(), PyList::new_bound(py, vals))?;
            }
            MmapBatchColumn::Str(vals) => {
                columns_dict.set_item(name.as_str(), PyList::new_bound(py, vals))?;
            }
            MmapBatchColumn::Bool(vals) => {
                columns_dict.set_item(name.as_str(), PyList::new_bound(py, vals))?;
            }
            MmapBatchColumn::Bin(vals) => {
                let list = PyList::empty_bound(py);
                for val in vals {
                    match val {
                        Some(bytes) => list.append(PyBytes::new_bound(py, &bytes))?,
                        None => list.append(py.None())?,
                    }
                }
                columns_dict.set_item(name.as_str(), list)?;
            }
        }
        Ok(())
    };

    if let Some(requested) = requested {
        for requested_col in requested {
            let Some(pos) = columns.iter().position(|(name, _)| name == requested_col) else {
                return Ok(None);
            };
            let (name, col) = columns.swap_remove(pos);
            emit_column(name, col)?;
        }
    } else {
        for (name, col) in columns {
            emit_column(name, col)?;
        }
    }

    Ok(Some(columns_dict))
}

fn projected_values_to_columns_dict<'py>(
    py: Python<'py>,
    vals: &[(String, Value)],
    columns: &[String],
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let columns_dict = PyDict::new_bound(py);
    if vals.len() >= columns.len()
        && columns.iter().enumerate().all(|(idx, requested_col)| {
            vals.get(idx)
                .map(|(col_name, _)| col_name == requested_col)
                .unwrap_or(false)
        })
    {
        for (requested_col, (_, val)) in columns.iter().zip(vals.iter()) {
            let pyval = value_to_py(py, val)?;
            columns_dict.set_item(requested_col.as_str(), PyList::new_bound(py, [pyval]))?;
        }
        return Ok(Some(columns_dict));
    }

    for requested_col in columns {
        let Some((_, val)) = vals.iter().find(|(col_name, _)| col_name == requested_col) else {
            return Ok(None);
        };
        let pyval = value_to_py(py, val)?;
        columns_dict.set_item(requested_col.as_str(), PyList::new_bound(py, [pyval]))?;
    }
    Ok(Some(columns_dict))
}

fn projected_values_to_row_dict<'py>(
    py: Python<'py>,
    vals: &[(String, Value)],
    columns: &[String],
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let row = PyDict::new_bound(py);
    if vals.len() >= columns.len()
        && columns.iter().enumerate().all(|(idx, requested_col)| {
            vals.get(idx)
                .map(|(col_name, _)| col_name == requested_col)
                .unwrap_or(false)
        })
    {
        for (requested_col, (_, val)) in columns.iter().zip(vals.iter()) {
            row.set_item(requested_col.as_str(), value_to_py(py, val)?)?;
        }
        return Ok(Some(row));
    }

    for requested_col in columns {
        let Some((_, val)) = vals.iter().find(|(col_name, _)| col_name == requested_col) else {
            return Ok(None);
        };
        row.set_item(requested_col.as_str(), value_to_py(py, val)?)?;
    }
    Ok(Some(row))
}

#[derive(Clone, Copy)]
struct NumericUpdateCellCache {
    footer_offset: u64,
    null_byte_file_offset: u64,
    null_mask: u8,
    value_file_offset: u64,
}

/// ApexStorage - On-demand columnar storage engine
///
/// This storage engine uses V4 format (.apex) for persistence and supports:
/// - On-demand column reading (only loads requested columns)
/// - On-demand row range reading (only loads requested rows)
/// - Soft delete with deleted bitmap
/// - Full SQL query support via ApexExecutor
/// - Cross-platform file locking for concurrent access safety
///   - Read operations use shared locks (multiple readers allowed)
///   - Write operations use exclusive locks (single writer)
/// - Multi-database support: named databases stored in subdirectories
#[pyclass(name = "ApexStorage")]
pub struct ApexStorageImpl {
    /// Root directory (top-level dir; contains both default tables and named-db subdirs)
    root_dir: PathBuf,
    /// Current database name. "" or "default" means root_dir (backward-compat default).
    /// Named databases (e.g. "analytics") reside at root_dir/analytics/.
    current_database: RwLock<String>,
    /// Current base directory = root_dir (default) or root_dir/db_name (named db).
    /// Updated atomically by use_database_().
    base_dir: RwLock<PathBuf>,
    /// Table paths (table_name -> path) - lazily populated
    table_paths: RwLock<HashMap<String, PathBuf>>,
    /// Whether table_paths has been fully scanned from directory
    tables_scanned: RwLock<bool>,
    /// Cached storage backends per table (table_name -> backend)
    /// Backends are opened once and reused for all operations
    /// Uses DashMap for lock-free concurrent reads
    cached_backends: DashMap<String, Arc<TableStorageBackend>>,
    /// Verified `(table, column) -> ColumnType` entries for numeric `_id` update fast paths.
    update_by_id_numeric_cache: DashMap<String, crate::storage::on_demand::ColumnType>,
    /// Verified `(table, column, id) -> physical cell offsets` entries for repeated numeric updates.
    update_by_id_cell_cache: DashMap<String, NumericUpdateCellCache>,
    /// Exact full-row payloads for repeated idempotent `replace(id, row)` calls.
    replace_exact_row_cache: DashMap<String, HashMap<String, Value>>,
    /// Current table name
    current_table: RwLock<String>,
    /// FTS Manager (optional) — Arc so it can be shared with the global SQL executor registry
    fts_manager: RwLock<Option<Arc<FtsManager>>>,
    /// FTS index field names per table
    fts_index_fields: RwLock<HashMap<String, Vec<String>>>,
    /// Durability level for ACID guarantees
    durability: DurabilityLevel,
    /// Current active transaction ID (None if not in a transaction)
    current_txn_id: RwLock<Option<u64>>,
    /// Auto-flush row threshold (struct-level so it survives backend cache invalidation)
    auto_flush_rows: RwLock<u64>,
    /// Auto-flush byte threshold (struct-level so it survives backend cache invalidation)
    auto_flush_bytes: RwLock<u64>,
    /// Temp directory for temporary tables (root_dir/.apex_tmp/)
    temp_dir: PathBuf,
}

/// Internal Rust-only methods (not exposed to Python)
impl ApexStorageImpl {
    #[inline]
    fn backend_cache_key(table_path: &std::path::Path, table_name: &str) -> String {
        format!("{}\0{}", table_path.to_string_lossy(), table_name)
    }

    #[inline]
    fn insert_backend_cache_key(table_path: &std::path::Path, table_name: &str) -> String {
        format!("{}\0{}\0insert", table_path.to_string_lossy(), table_name)
    }

    #[inline]
    fn replace_row_cache_key(
        table_path: &std::path::Path,
        table_name: &str,
        row_id: u64,
    ) -> String {
        format!(
            "{}\0{}\0replace\0{}",
            table_path.to_string_lossy(),
            table_name,
            row_id
        )
    }

    /// Get the lock file path for a table
    #[inline]
    fn get_lock_path(table_path: &Path) -> PathBuf {
        table_path.with_extension("apex.lock")
    }

    /// Acquire a lock on the table (shared for read, exclusive for write).
    /// Uses retry with exponential backoff (100µs → 200µs → ... → 50ms max total wait).
    /// This avoids spurious "Database is locked" errors under concurrent load.
    fn acquire_lock(table_path: &Path, exclusive: bool) -> io::Result<File> {
        let lock_path = Self::get_lock_path(table_path);
        let lock_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;

        let max_wait = std::time::Duration::from_millis(50);
        let mut backoff = std::time::Duration::from_micros(100);
        let start = std::time::Instant::now();

        loop {
            let result: io::Result<()> = if exclusive {
                lock_file.try_lock_exclusive()
            } else {
                lock_file
                    .try_lock_shared()
                    .map_err(|e| io::Error::new(io::ErrorKind::WouldBlock, e.to_string()))
            };

            match result {
                Ok(()) => return Ok(lock_file),
                Err(_) if start.elapsed() < max_wait => {
                    std::thread::sleep(backoff);
                    backoff = (backoff * 2).min(std::time::Duration::from_millis(5));
                }
                Err(e) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WouldBlock,
                        format!(
                            "Database is locked (waited {}ms): {}",
                            start.elapsed().as_millis(),
                            e
                        ),
                    ));
                }
            }
        }
    }

    #[inline]
    fn acquire_read_lock(table_path: &Path) -> io::Result<File> {
        Self::acquire_lock(table_path, false)
    }

    #[inline]
    fn acquire_write_lock(table_path: &Path) -> io::Result<File> {
        Self::acquire_lock(table_path, true)
    }

    /// Release a lock (unlock and drop the file handle)
    #[inline]
    fn release_lock(lock_file: File) {
        let _ = lock_file.unlock();
        drop(lock_file);
    }

    /// Parse a Python dict {col_name: type_str} into Vec<(String, ColumnType)>
    fn parse_schema_dict(
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<Vec<(String, crate::storage::on_demand::ColumnType)>> {
        use crate::storage::on_demand::ColumnType;
        let mut cols = Vec::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            let col_name: String = key.extract()?;
            let type_str: String = value.extract()?;
            let ct = match type_str.to_lowercase().as_str() {
                "int8" | "i8" => ColumnType::Int8,
                "int16" | "i16" => ColumnType::Int16,
                "int32" | "i32" | "int" => ColumnType::Int32,
                "int64" | "i64" | "integer" => ColumnType::Int64,
                "uint8" | "u8" => ColumnType::UInt8,
                "uint16" | "u16" => ColumnType::UInt16,
                "uint32" | "u32" => ColumnType::UInt32,
                "uint64" | "u64" => ColumnType::UInt64,
                "float32" | "f32" | "float" => ColumnType::Float32,
                "float64" | "f64" | "double" => ColumnType::Float64,
                "bool" | "boolean" => ColumnType::Bool,
                "str" | "string" | "text" | "varchar" => ColumnType::String,
                "bytes" | "binary" => ColumnType::Binary,
                "timestamp" | "datetime" => ColumnType::Timestamp,
                "date" => ColumnType::Date,
                _ => return Err(PyValueError::new_err(format!(
                    "Unknown column type '{}' for column '{}'. Supported: int8, int16, int32, int64, \
                     uint8, uint16, uint32, uint64, float32, float64, bool, string, binary, timestamp, date",
                    type_str, col_name
                ))),
            };
            cols.push((col_name, ct));
        }
        Ok(cols)
    }

    /// Get the path for the current table
    #[inline]
    fn get_current_table_path(&self) -> PyResult<PathBuf> {
        let table_name = self.current_table.read().clone();
        if table_name.is_empty() {
            return Err(PyValueError::new_err(
                "No table selected. Call create_table() or use_table() first.",
            ));
        }
        let paths = self.table_paths.read();
        if let Some(p) = paths.get(&table_name) {
            return Ok(p.clone());
        }
        drop(paths);
        // Lazy: check disk using current base_dir
        let base_dir = self.current_base_dir();
        let p = base_dir.join(format!("{}.apex", table_name));
        if p.exists() {
            self.table_paths
                .write()
                .insert(table_name.clone(), p.clone());
            return Ok(p);
        }
        Err(PyValueError::new_err(format!(
            "Table not found: {}",
            table_name
        )))
    }

    /// Get both table path and name in one lock acquisition (optimization)
    #[inline]
    fn get_current_table_info(&self) -> PyResult<(PathBuf, String)> {
        let table_name = self.current_table.read().clone();
        if table_name.is_empty() {
            return Err(PyValueError::new_err(
                "No table selected. Call create_table() or use_table() first.",
            ));
        }
        let path = {
            let paths = self.table_paths.read();
            paths.get(&table_name).cloned()
        };
        if let Some(p) = path {
            return Ok((p, table_name));
        }
        // Lazy: check disk using current base_dir
        let base_dir = self.current_base_dir();
        let p = base_dir.join(format!("{}.apex", table_name));
        if p.exists() {
            self.table_paths
                .write()
                .insert(table_name.clone(), p.clone());
            return Ok((p, table_name));
        }
        Err(PyValueError::new_err(format!(
            "Table not found: {}",
            table_name
        )))
    }

    /// Resolve the table path for a query-signature fast path.
    #[inline]
    fn resolve_signature_table(
        &self,
        explicit_table: Option<&str>,
        default_table_name: &str,
        default_table_path: &Path,
        base_dir: &Path,
    ) -> (String, PathBuf) {
        let clean_name = match explicit_table {
            Some(name) => name.trim_matches('"').trim_matches('`'),
            None => default_table_name,
        };

        if clean_name.is_empty() {
            return (
                default_table_name.to_string(),
                default_table_path.to_path_buf(),
            );
        }

        if let Some(dot_pos) = clean_name.find('.') {
            let db_name = clean_name[..dot_pos].trim();
            let tbl_name = clean_name[dot_pos + 1..].trim();
            let safe_tbl: String = tbl_name
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || c == '_' || c == '-' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            let safe_tbl = if safe_tbl.len() > 200 {
                &safe_tbl[..200]
            } else {
                &safe_tbl
            };

            let db_dir = if db_name.is_empty() || db_name.eq_ignore_ascii_case("default") {
                self.root_dir.clone()
            } else {
                self.root_dir.join(db_name)
            };
            return (
                clean_name.to_string(),
                db_dir.join(format!("{}.apex", safe_tbl)),
            );
        }

        if clean_name.eq_ignore_ascii_case("default") || clean_name == default_table_name {
            return (clean_name.to_string(), default_table_path.to_path_buf());
        }

        let safe_name: String = clean_name
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let safe_name = if safe_name.len() > 200 {
            &safe_name[..200]
        } else {
            &safe_name
        };
        (
            clean_name.to_string(),
            base_dir.join(format!("{}.apex", safe_name)),
        )
    }

    /// Get or create cached backend for current table
    /// Uses open_for_write to ensure existing data is loaded for write operations
    /// Get backend for INSERT operations - memory efficient!
    /// Uses open_for_insert which doesn't load existing column data.
    /// Data is written to delta file and merged on read.
    fn get_backend_for_insert(&self) -> PyResult<Arc<TableStorageBackend>> {
        let table_name = self.current_table.read().clone();
        let table_path = self.get_current_table_path()?;
        let cache_key = Self::insert_backend_cache_key(&table_path, &table_name);

        // Check if backend is already cached (lock-free read)
        if let Some(entry) = self.cached_backends.get(&cache_key) {
            return Ok(entry.clone());
        }

        // Create new backend with open_for_insert (memory efficient)
        let backend = if table_path.exists() {
            TableStorageBackend::open_for_insert_with_durability(&table_path, self.durability)
                .map_err(|e| PyIOError::new_err(e.to_string()))?
        } else {
            TableStorageBackend::create_with_durability(&table_path, self.durability)
                .map_err(|e| PyIOError::new_err(e.to_string()))?
        };

        let backend = Arc::new(backend);
        self.cached_backends.insert(cache_key, backend.clone());

        Ok(backend)
    }

    /// Get a mmap/read backend suitable for fast overlay writes.
    fn get_backend_for_overlay(
        &self,
        table_path: &Path,
        table_name: &str,
    ) -> PyResult<Arc<TableStorageBackend>> {
        let cache_key = Self::backend_cache_key(table_path, table_name);
        if let Some(entry) = self.cached_backends.get(&cache_key) {
            return Ok(entry.clone());
        }

        if let Ok(backend) = crate::query::get_cached_backend_pub(table_path) {
            self.cached_backends.insert(cache_key, Arc::clone(&backend));
            return Ok(backend);
        }

        let backend = Arc::new(
            TableStorageBackend::open_with_durability(table_path, self.durability)
                .map_err(|e| PyIOError::new_err(e.to_string()))?,
        );
        self.cached_backends.insert(cache_key, Arc::clone(&backend));
        crate::query::executor::cache_backend_pub(table_path, Arc::clone(&backend));
        Ok(backend)
    }

    fn table_has_secondary_indexes(&self, table_path: &Path, table_name: &str) -> bool {
        let base_dir = table_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();
        crate::storage::index::IndexManager::load(table_name, &base_dir)
            .map(|mgr| !mgr.catalog_is_empty())
            .unwrap_or(false)
    }

    #[inline]
    fn py_value_matches_exact(obj: &Bound<'_, PyAny>, stored: &Value) -> PyResult<bool> {
        use pyo3::types::PyBytes;

        if obj.is_none() {
            return Ok(matches!(stored, Value::Null));
        }

        if let Ok(value) = obj.extract::<bool>() {
            return Ok(matches!(stored, Value::Bool(current) if *current == value));
        }

        if let Ok(value) = obj.extract::<i64>() {
            return Ok(matches!(stored, Value::Int64(current) if *current == value));
        }

        if let Ok(value) = obj.extract::<f64>() {
            return Ok(matches!(stored, Value::Float64(current) if *current == value));
        }

        if obj.is_instance_of::<PyBytes>() {
            if let Ok(value) = obj.extract::<Vec<u8>>() {
                return Ok(matches!(stored, Value::Binary(current) if *current == value));
            }
        }

        if obj
            .get_type()
            .name()
            .map(|name| name == "ndarray")
            .unwrap_or(false)
        {
            if let Ok(value) = obj
                .call_method0("flatten")
                .and_then(|flat| flat.call_method1("astype", ("float32",)))
                .and_then(|f32arr| f32arr.call_method0("tobytes"))
                .and_then(|bytes| bytes.extract::<Vec<u8>>())
            {
                if value.len() % 4 == 0 {
                    return Ok(matches!(stored, Value::FixedList(current) if *current == value));
                }
            }
            return Ok(false);
        }

        if let Ok(value) = obj.extract::<String>() {
            return Ok(matches!(stored, Value::String(current) if *current == value));
        }

        if let Ok(value) = obj.extract::<Vec<u8>>() {
            return Ok(matches!(stored, Value::Binary(current) if *current == value));
        }

        Ok(matches!(stored, Value::Null))
    }

    fn py_dict_matches_exact_fields(
        data: &Bound<'_, PyDict>,
        fields: &HashMap<String, Value>,
    ) -> PyResult<bool> {
        let dict_len = data.len();
        if dict_len != fields.len() {
            if dict_len != fields.len() + 1 || data.get_item("_id").ok().flatten().is_none() {
                return Ok(false);
            }
        }

        for (name, stored) in fields {
            let Some(value) = data.get_item(name).ok().flatten() else {
                return Ok(false);
            };
            if !Self::py_value_matches_exact(&value, stored)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn row_matches_exact_py_dict(
        &self,
        backend: &TableStorageBackend,
        row_id: u64,
        data: &Bound<'_, PyDict>,
    ) -> PyResult<Option<bool>> {
        let schema = backend.storage.get_schema();
        if schema.is_empty() {
            return Ok(None);
        }

        let mut field_count = 0usize;
        for (key, _) in data.iter() {
            let key: String = key.extract()?;
            if key == "_id" {
                continue;
            }
            if !schema.iter().any(|(name, _)| name == &key) {
                return Ok(None);
            }
            field_count += 1;
        }
        if field_count != schema.len() {
            return Ok(None);
        }

        {
            let delta = backend.storage.delta_store();
            if delta.is_deleted(row_id) {
                return Ok(Some(false));
            }
            if let Some(updates) = delta.get_row_updates(row_id) {
                if schema.iter().all(|(name, _)| updates.contains_key(name)) {
                    for (name, _) in &schema {
                        let Some(value) = data.get_item(name).ok().flatten() else {
                            return Ok(None);
                        };
                        let Some(record) = updates.get(name) else {
                            return Ok(Some(false));
                        };
                        if !Self::py_value_matches_exact(&value, &record.new_value)? {
                            return Ok(Some(false));
                        }
                    }
                    return Ok(Some(true));
                }
            }
        }

        let mut current_row: HashMap<String, Value> = backend
            .storage
            .retrieve_rcix(row_id)
            .ok()
            .flatten()
            .or_else(|| backend.storage.read_row_by_id_values(row_id).ok().flatten())
            .map(|vals| vals.into_iter().collect())
            .unwrap_or_default();

        if current_row.is_empty() {
            return Ok(Some(false));
        }

        {
            let delta = backend.storage.delta_store();
            if let Some(updates) = delta.get_row_updates(row_id) {
                for (col_name, record) in updates {
                    current_row.insert(col_name.clone(), record.new_value.clone());
                }
            }
        }

        for (name, _) in &schema {
            let Some(value) = data.get_item(name).ok().flatten() else {
                return Ok(None);
            };
            let Some(current) = current_row.get(name) else {
                return Ok(Some(false));
            };
            if !Self::py_value_matches_exact(&value, current)? {
                return Ok(Some(false));
            }
        }

        Ok(Some(true))
    }

    /// Return `Some(true)` when the current stored row is already identical to
    /// the provided full-row payload. Returns `None` when we cannot cheaply
    /// determine equality (for example partial-row replacements).
    fn row_matches_exact_fields(
        &self,
        backend: &TableStorageBackend,
        row_id: u64,
        fields: &HashMap<String, Value>,
    ) -> PyResult<Option<bool>> {
        let schema = backend.storage.get_schema();
        if schema.is_empty()
            || schema.len() != fields.len()
            || schema.iter().any(|(name, _)| !fields.contains_key(name))
        {
            return Ok(None);
        }

        {
            let delta = backend.storage.delta_store();
            if delta.is_deleted(row_id) {
                return Ok(Some(false));
            }
            if let Some(updates) = delta.get_row_updates(row_id) {
                if schema.iter().all(|(name, _)| updates.contains_key(name)) {
                    return Ok(Some(schema.iter().all(|(name, _)| {
                        updates.get(name).map(|record| &record.new_value) == fields.get(name)
                    })));
                }
            }
        }

        let mut current_row: HashMap<String, Value> = backend
            .storage
            .retrieve_rcix(row_id)
            .ok()
            .flatten()
            .or_else(|| backend.storage.read_row_by_id_values(row_id).ok().flatten())
            .map(|vals| vals.into_iter().collect())
            .unwrap_or_default();

        if current_row.is_empty() {
            return Ok(Some(false));
        }

        {
            let delta = backend.storage.delta_store();
            if let Some(updates) = delta.get_row_updates(row_id) {
                for (col_name, record) in updates {
                    current_row.insert(col_name.clone(), record.new_value.clone());
                }
            }
        }

        Ok(Some(schema.iter().all(|(name, _)| {
            current_row.get(name) == fields.get(name)
        })))
    }

    fn persist_pending_overlay_for_table(
        &self,
        table_path: &Path,
        table_name: &str,
    ) -> PyResult<()> {
        let mut backends: Vec<Arc<TableStorageBackend>> = Vec::new();
        let cache_key = Self::backend_cache_key(table_path, table_name);

        if let Some(entry) = self.cached_backends.get(&cache_key) {
            backends.push(Arc::clone(entry.value()));
        }
        if let Some(entry) = self.cached_backends.get(table_name) {
            let backend = Arc::clone(entry.value());
            if !backends.iter().any(|b| Arc::ptr_eq(b, &backend)) {
                backends.push(backend);
            }
        }
        for backend in backends {
            if backend.has_pending_deltas() {
                backend
                    .save_delta_store()
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Get backend for UPDATE/DELETE operations - loads all data into memory.
    /// This is required because save() rewrites the entire file.
    fn get_backend(&self) -> PyResult<Arc<TableStorageBackend>> {
        let table_name = self.current_table.read().clone();
        let table_path = self.get_current_table_path()?;
        let cache_key = Self::backend_cache_key(&table_path, &table_name);

        // Check if backend is already cached (lock-free read)
        if let Some(entry) = self.cached_backends.get(&cache_key) {
            return Ok(entry.clone());
        }

        // Create new backend with durability level and cache it
        // Use open_for_write to ensure existing column data is loaded
        // This is necessary because save() rewrites the entire file from in-memory columns
        let backend = if table_path.exists() {
            TableStorageBackend::open_for_write_with_durability(&table_path, self.durability)
                .map_err(|e| PyIOError::new_err(e.to_string()))?
        } else {
            TableStorageBackend::create_with_durability(&table_path, self.durability)
                .map_err(|e| PyIOError::new_err(e.to_string()))?
        };

        let backend = Arc::new(backend);
        self.cached_backends.insert(cache_key, backend.clone());

        Ok(backend)
    }

    /// Invalidate cached backend for a table (used when table is dropped or modified externally)
    fn invalidate_backend(&self, table_name: &str) {
        self.cached_backends.remove(table_name);
        self.cached_backends
            .remove(&format!("{}_insert", table_name));
        let table_suffix = format!("\0{table_name}");
        let insert_suffix = format!("\0{table_name}\0insert");
        self.cached_backends
            .retain(|key, _| !(key.ends_with(&table_suffix) || key.ends_with(&insert_suffix)));
        let update_cache_marker = format!("\0{table_name}\0");
        let legacy_prefix = format!("{table_name}\0");
        self.update_by_id_numeric_cache.retain(|key, _| {
            !(key.starts_with(&legacy_prefix) || key.contains(&update_cache_marker))
        });
        self.update_by_id_cell_cache.retain(|key, _| {
            !(key.starts_with(&legacy_prefix) || key.contains(&update_cache_marker))
        });
        let replace_cache_marker = format!("\0{table_name}\0replace\0");
        self.replace_exact_row_cache
            .retain(|key, _| !key.contains(&replace_cache_marker));
    }

    /// Return current base directory (root_dir for default db, root_dir/db for named db)
    #[inline]
    fn current_base_dir(&self) -> PathBuf {
        self.base_dir.read().clone()
    }
}

#[pymethods]
impl ApexStorageImpl {
    /// Create or open a storage
    ///
    /// Parameters:
    /// - path: Path to the storage file (will use .apex extension)
    /// - drop_if_exists: If true, delete existing database
    /// - durability: Durability level ('fast', 'safe', or 'max')
    #[new]
    #[pyo3(signature = (path, drop_if_exists = false, durability = "fast"))]
    fn new(path: &str, drop_if_exists: bool, durability: &str) -> PyResult<Self> {
        // Parse durability level
        let durability_level = DurabilityLevel::from_str(durability).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Invalid durability level '{}'. Must be 'fast', 'safe', or 'max'",
                durability
            ))
        })?;
        // Convert to absolute path to avoid issues with relative paths
        let path_obj = PathBuf::from(path);
        let abs_path = if path_obj.is_absolute() {
            path_obj
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(&path_obj)
        };
        let root_dir = abs_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));

        // Handle drop_if_exists
        if drop_if_exists {
            crate::storage::engine::engine().invalidate_dir(&root_dir);
            crate::query::executor::unregister_fts_manager(&root_dir);

            // Remove all .apex files in the directory
            if let Ok(entries) = fs::read_dir(&root_dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().map(|e| e == "apex").unwrap_or(false) {
                        let _ = fs::remove_file(&p);
                    }
                }
            }

            // Also remove FTS indexes
            let fts_dir = root_dir.join("fts_indexes");
            if fts_dir.exists() {
                let _ = fs::remove_dir_all(&fts_dir);
            }
        }

        // No default table - users must explicitly create or use a table
        // Existing .apex files in the directory are discovered lazily via use_table() or list_tables()

        let temp_dir = root_dir.join(".apex_tmp");
        let _ = fs::create_dir_all(&temp_dir);

        Ok(Self {
            root_dir: root_dir.clone(),
            current_database: RwLock::new(String::new()),
            base_dir: RwLock::new(root_dir),
            table_paths: RwLock::new(HashMap::new()),
            tables_scanned: RwLock::new(false),
            cached_backends: DashMap::new(),
            update_by_id_numeric_cache: DashMap::new(),
            update_by_id_cell_cache: DashMap::new(),
            replace_exact_row_cache: DashMap::new(),
            current_table: RwLock::new(String::new()),
            fts_manager: RwLock::new(None::<Arc<FtsManager>>),
            fts_index_fields: RwLock::new(HashMap::new()),
            durability: durability_level,
            current_txn_id: RwLock::new(None),
            auto_flush_rows: RwLock::new(0),
            auto_flush_bytes: RwLock::new(0),
            temp_dir,
        })
    }

    /// Store a single record using StorageEngine
    /// Automatically chooses delta or full write based on conditions
    fn store(&self, py: Python<'_>, data: &Bound<'_, PyDict>) -> PyResult<i64> {
        let fields = dict_to_values(data)?;
        let (table_path, table_name) = self.get_current_table_info()?;
        let durability = self.durability;
        self.persist_pending_overlay_for_table(&table_path, &table_name)?;

        // Skip file lock for 'fast' durability — StorageEngine handles thread safety
        // internally via parking_lot::RwLock. File locks only needed for cross-process safety.
        let lock_file = if durability != DurabilityLevel::Fast {
            Some(
                Self::acquire_write_lock(&table_path)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?,
            )
        } else {
            None
        };

        // Use StorageEngine for smart write routing
        let result = py.allow_threads(|| {
            let engine = crate::storage::engine::engine();
            let ids = engine
                .write(&table_path, &[fields], durability)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            Ok::<i64, PyErr>(ids.first().copied().unwrap_or(0) as i64)
        });

        if let Some(lf) = lock_file {
            Self::release_lock(lf);
        }

        // Invalidate local backend cache (StorageEngine handles its own cache)
        self.invalidate_backend(&table_name);

        let id = result?;

        // Index in FTS if enabled
        self.index_for_fts(id, data)?;

        Ok(id)
    }

    /// Store multiple records using StorageEngine
    /// Automatically chooses delta or full write based on conditions
    fn store_batch(&self, py: Python<'_>, data: &Bound<'_, PyList>) -> PyResult<Vec<i64>> {
        let num_rows = data.len();
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        // Collect all rows
        let mut rows: Vec<HashMap<String, Value>> = Vec::with_capacity(num_rows);
        for item in data.iter() {
            let dict = item.downcast::<PyDict>()?;
            let fields = dict_to_values(dict)?;
            rows.push(fields);
        }

        let (table_path, table_name) = self.get_current_table_info()?;
        let durability = self.durability;
        self.persist_pending_overlay_for_table(&table_path, &table_name)?;

        // Skip file lock for 'fast' durability
        let lock_file = if durability != DurabilityLevel::Fast {
            Some(
                Self::acquire_write_lock(&table_path)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?,
            )
        } else {
            None
        };

        // Use StorageEngine for smart write routing
        let result = py.allow_threads(|| {
            let engine = crate::storage::engine::engine();
            engine
                .write(&table_path, &rows, durability)
                .map_err(|e| PyIOError::new_err(e.to_string()))
        });

        if let Some(lf) = lock_file {
            Self::release_lock(lf);
        }

        // Invalidate local backend cache
        self.invalidate_backend(&table_name);

        let ids = result?;

        // Index in FTS if enabled (batch operation - only if FTS manager exists)
        // OPTIMIZED: Use add_documents_arrow_str (🥈 ~3.3M docs/s, zero-copy &str path)
        {
            let mgr = self.fts_manager.read();
            if mgr.is_some() {
                let table_name = self.current_table.read().clone();
                let index_fields = self.fts_index_fields.read().get(&table_name).cloned();

                if let Some(m) = mgr.as_ref() {
                    if let Ok(engine) = m.get_engine(&table_name) {
                        // Determine which fields to index
                        let fields_to_index: Vec<String> = match &index_fields {
                            Some(fields) => fields.clone(),
                            None => {
                                // Auto-detect string fields from first document
                                let mut auto_fields = Vec::new();
                                if let Some(first_item) = data.iter().next() {
                                    if let Ok(dict) = first_item.downcast::<PyDict>() {
                                        for (key, value) in dict.iter() {
                                            if let Ok(key_str) = key.extract::<String>() {
                                                if key_str != "_id"
                                                    && value.extract::<String>().is_ok()
                                                {
                                                    auto_fields.push(key_str);
                                                }
                                            }
                                        }
                                    }
                                }
                                auto_fields
                            }
                        };

                        if !fields_to_index.is_empty() {
                            let num_docs = ids.len();
                            // Build columnar String data — direct per-field lookup, no per-doc HashMap
                            let mut columns: Vec<(String, Vec<String>)> = fields_to_index
                                .iter()
                                .map(|f| (f.clone(), Vec::with_capacity(num_docs)))
                                .collect();

                            for (i, item) in data.iter().enumerate() {
                                if i >= ids.len() {
                                    break;
                                }
                                if let Ok(dict) = item.downcast::<PyDict>() {
                                    for (field_idx, field_name) in
                                        fields_to_index.iter().enumerate()
                                    {
                                        let value = dict
                                            .get_item(field_name)
                                            .ok()
                                            .flatten()
                                            .and_then(|v| v.extract::<String>().ok())
                                            .unwrap_or_default();
                                        columns[field_idx].1.push(value);
                                    }
                                }
                            }

                            // 🥈 add_documents_arrow_str: zero-copy &str slices, ~3.3M docs/s
                            if !columns.is_empty() && !columns[0].1.is_empty() {
                                let doc_ids_u32: Vec<u32> =
                                    ids.iter().map(|&id| id as u32).collect();
                                let columns_ref: Vec<(String, Vec<&str>)> = columns
                                    .iter()
                                    .map(|(name, vals)| {
                                        (name.clone(), vals.iter().map(|s| s.as_str()).collect())
                                    })
                                    .collect();
                                let _ = py.allow_threads(|| {
                                    engine.add_documents_arrow_str(&doc_ids_u32, columns_ref)
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(ids.into_iter().map(|id| id as i64).collect())
    }

    /// Store columnar data directly - bypasses row-by-row conversion
    /// Much faster for bulk inserts with homogeneous data
    ///
    /// Args:
    ///     columns: Dict[str, list] - column name to list of values
    ///     
    /// Returns:
    ///     List[int] - list of generated IDs
    fn store_one(&self, py: Python<'_>, row: &Bound<'_, PyDict>) -> PyResult<Vec<i64>> {
        if row.is_empty() {
            return Ok(Vec::new());
        }

        let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
        let mut binary_columns_map: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut fixedlist_columns_map: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();
        let mut null_positions: HashMap<String, Vec<bool>> = HashMap::new();

        for (key, value) in row.iter() {
            let col_name: String = key.extract()?;
            if col_name == "_id" {
                continue;
            }

            if value.is_none() {
                string_columns.insert(col_name.clone(), vec![String::new()]);
                null_positions.insert(col_name, vec![true]);
            } else if let Ok(v) = value.extract::<bool>() {
                bool_columns.insert(col_name.clone(), vec![v]);
                null_positions.insert(col_name, vec![false]);
            } else if let Ok(v) = value.extract::<i64>() {
                int_columns.insert(col_name.clone(), vec![v]);
                null_positions.insert(col_name, vec![false]);
            } else if let Ok(v) = value.extract::<f64>() {
                float_columns.insert(col_name.clone(), vec![v]);
                null_positions.insert(col_name, vec![false]);
            } else if let Ok(bytes) = value.extract::<Vec<u8>>() {
                binary_columns_map.insert(col_name.clone(), vec![bytes]);
                null_positions.insert(col_name, vec![false]);
            } else if value
                .get_type()
                .name()
                .map(|n| n == "ndarray")
                .unwrap_or(false)
            {
                if let Ok(bytes) = value
                    .call_method0("flatten")
                    .and_then(|flat| flat.call_method1("astype", ("float32",)))
                    .and_then(|f32arr| f32arr.call_method0("tobytes"))
                    .and_then(|b| b.extract::<Vec<u8>>())
                {
                    fixedlist_columns_map.insert(col_name.clone(), vec![bytes]);
                    null_positions.insert(col_name, vec![false]);
                } else {
                    string_columns.insert(
                        col_name.clone(),
                        vec![value.extract::<String>().unwrap_or_default()],
                    );
                    null_positions.insert(col_name, vec![false]);
                }
            } else {
                string_columns.insert(
                    col_name.clone(),
                    vec![value.extract::<String>().unwrap_or_default()],
                );
                null_positions.insert(col_name, vec![false]);
            }
        }

        if int_columns.is_empty()
            && float_columns.is_empty()
            && string_columns.is_empty()
            && binary_columns_map.is_empty()
            && fixedlist_columns_map.is_empty()
            && bool_columns.is_empty()
        {
            return Ok(Vec::new());
        }

        let (table_path, table_name) = self.get_current_table_info()?;
        let durability = self.durability;
        self.persist_pending_overlay_for_table(&table_path, &table_name)?;
        let result = py.allow_threads(|| {
            crate::storage::engine::engine()
                .write_typed(
                    &table_path,
                    int_columns,
                    float_columns,
                    string_columns,
                    binary_columns_map,
                    fixedlist_columns_map,
                    bool_columns,
                    null_positions,
                    durability,
                )
                .map_err(|e| PyIOError::new_err(e.to_string()))
        })?;

        self.invalidate_backend(&table_name);
        #[cfg(target_os = "windows")]
        crate::storage::engine::engine().invalidate(&table_path);

        Ok(result.into_iter().map(|id| id as i64).collect())
    }

    /// Experimental storage-level memtable append for one schema-stable row.
    ///
    /// The row is immediately visible through this storage instance and is
    /// persisted by flush()/close(). It is intentionally narrow and opt-in from
    /// Python until cross-client visibility semantics are fully settled.
    fn store_one_memtable(
        &self,
        py: Python<'_>,
        row: &Bound<'_, PyDict>,
    ) -> PyResult<Option<Vec<i64>>> {
        if row.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let (table_path, table_name) = self.get_current_table_info()?;
        let base_dir = table_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();

        if let Ok(index_mgr) = crate::storage::index::IndexManager::load(&table_name, &base_dir) {
            if !index_mgr.catalog_is_empty() {
                return Ok(None);
            }
        }

        self.persist_pending_overlay_for_table(&table_path, &table_name)?;
        let backend = self.get_backend_for_insert()?;
        if !backend.storage.is_v4_format() || backend.storage.has_constraints() {
            return Ok(None);
        }

        let schema = backend.storage.get_schema();
        if schema.is_empty() {
            return Ok(None);
        }
        if schema.iter().any(|(_, ty)| {
            matches!(
                ty,
                crate::storage::on_demand::ColumnType::FixedList
                    | crate::storage::on_demand::ColumnType::Float16List
                    | crate::storage::on_demand::ColumnType::Null
            )
        }) {
            return Ok(None);
        }
        let schema_len = schema.len();

        let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
        let mut binary_columns_map: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let fixedlist_columns_map: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();
        let mut null_positions: HashMap<String, Vec<bool>> = HashMap::new();
        let mut field_count = 0usize;

        for (key, value) in row.iter() {
            let col_name: String = key.extract()?;
            if col_name == "_id" {
                continue;
            }
            field_count += 1;
            let Some((_, col_type)) = schema.iter().find(|(name, _)| name == &col_name) else {
                return Ok(None);
            };

            use crate::storage::on_demand::ColumnType;
            match *col_type {
                ColumnType::Bool => {
                    let v = if value.is_none() {
                        false
                    } else if let Ok(v) = value.extract::<bool>() {
                        v
                    } else {
                        return Ok(None);
                    };
                    bool_columns.insert(col_name.clone(), vec![v]);
                    null_positions.insert(col_name, vec![value.is_none()]);
                }
                ColumnType::Int8
                | ColumnType::Int16
                | ColumnType::Int32
                | ColumnType::Int64
                | ColumnType::UInt8
                | ColumnType::UInt16
                | ColumnType::UInt32
                | ColumnType::UInt64
                | ColumnType::Timestamp
                | ColumnType::Date => {
                    let v = if value.is_none() {
                        0
                    } else if let Ok(v) = value.extract::<i64>() {
                        v
                    } else {
                        return Ok(None);
                    };
                    int_columns.insert(col_name.clone(), vec![v]);
                    null_positions.insert(col_name, vec![value.is_none()]);
                }
                ColumnType::Float32 | ColumnType::Float64 => {
                    let v = if value.is_none() {
                        0.0
                    } else if let Ok(v) = value.extract::<f64>() {
                        v
                    } else {
                        match value.extract::<i64>() {
                            Ok(v) => v as f64,
                            Err(_) => return Ok(None),
                        }
                    };
                    float_columns.insert(col_name.clone(), vec![v]);
                    null_positions.insert(col_name, vec![value.is_none()]);
                }
                ColumnType::String | ColumnType::StringDict => {
                    let v = if value.is_none() {
                        String::new()
                    } else if let Ok(v) = value.extract::<String>() {
                        v
                    } else {
                        return Ok(None);
                    };
                    string_columns.insert(col_name.clone(), vec![v]);
                    null_positions.insert(col_name, vec![value.is_none()]);
                }
                ColumnType::Binary => {
                    let v = if value.is_none() {
                        Vec::new()
                    } else if let Ok(v) = value.extract::<Vec<u8>>() {
                        v
                    } else {
                        return Ok(None);
                    };
                    binary_columns_map.insert(col_name.clone(), vec![v]);
                    null_positions.insert(col_name, vec![value.is_none()]);
                }
                ColumnType::FixedList | ColumnType::Float16List | ColumnType::Null => {
                    return Ok(None);
                }
            }
        }

        if field_count != schema_len {
            return Ok(None);
        }

        let result = py.allow_threads(|| {
            backend
                .insert_typed_with_nulls_full(
                    int_columns,
                    float_columns,
                    string_columns,
                    binary_columns_map,
                    fixedlist_columns_map,
                    bool_columns,
                    null_positions,
                )
                .map_err(|e| PyIOError::new_err(e.to_string()))
        })?;

        let cache_key = Self::backend_cache_key(&table_path, &table_name);
        self.cached_backends.insert(cache_key, Arc::clone(&backend));
        crate::query::executor::cache_backend_pub(&table_path, Arc::clone(&backend));
        crate::query::planner::invalidate_table_stats(&table_path.to_string_lossy());

        Ok(Some(result.into_iter().map(|id| id as i64).collect()))
    }

    /// Fast OLTP append for one schema-stable row.
    ///
    /// This intentionally handles only the low-risk case: an existing V4 table,
    /// exact schema match, no constraints, and no secondary indexes. Other
    /// cases fall back to store_one(), which preserves schema evolution,
    /// constraint checks, and index maintenance through the existing path.
    fn store_one_delta(
        &self,
        py: Python<'_>,
        row: &Bound<'_, PyDict>,
    ) -> PyResult<Option<Vec<i64>>> {
        if row.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let fields = dict_to_column_values(row)?;
        if fields.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if fields.values().any(|value| {
            matches!(
                value,
                ColumnValue::Null | ColumnValue::Binary(_) | ColumnValue::FixedList(_)
            )
        }) {
            return Ok(None);
        }

        let (table_path, table_name) = self.get_current_table_info()?;
        let base_dir = table_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();

        // Keep indexed tables on the existing path until index maintenance for
        // delta-only rows is fully covered.
        if let Ok(index_mgr) = crate::storage::index::IndexManager::load(&table_name, &base_dir) {
            if !index_mgr.catalog_is_empty() {
                return Ok(None);
            }
        }

        self.persist_pending_overlay_for_table(&table_path, &table_name)?;
        let backend = self.get_backend_for_insert()?;

        if !backend.storage.is_v4_format() || backend.storage.has_constraints() {
            return Ok(None);
        }

        let schema = backend.storage.get_schema();
        if schema.is_empty() {
            return Ok(None);
        }
        let schema_len = schema.len();
        if schema_len != fields.len()
            || fields
                .keys()
                .any(|name| !schema.iter().any(|(schema_name, _)| schema_name == name))
        {
            return Ok(None);
        }

        let durability = self.durability;
        let lock_file = if durability != DurabilityLevel::Fast {
            Some(
                Self::acquire_write_lock(&table_path)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?,
            )
        } else {
            None
        };

        let result = py.allow_threads(|| {
            backend
                .insert_column_rows_to_delta(&[fields])
                .map_err(|e| PyIOError::new_err(e.to_string()))
        });

        if let Some(lf) = lock_file {
            Self::release_lock(lf);
        }

        let ids = result?;
        // Keep the insert backend warm so repeated OLTP appends don't reopen
        // and rescan the delta file. Read/query caches must still be invalidated.
        self.cached_backends
            .remove(&Self::backend_cache_key(&table_path, &table_name));
        crate::query::executor::invalidate_storage_cache(&table_path);
        crate::query::planner::invalidate_table_stats(&table_path.to_string_lossy());

        Ok(Some(ids.into_iter().map(|id| id as i64).collect()))
    }

    /// Fast OLTP append for multiple schema-stable transaction rows.
    fn store_rows_delta(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyList>,
    ) -> PyResult<Option<Vec<i64>>> {
        if rows.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let mut all_fields = Vec::with_capacity(rows.len());
        for item in rows.iter() {
            let row = item.downcast::<PyDict>()?;
            let fields = dict_to_column_values(row)?;
            if fields.is_empty() {
                return Ok(None);
            }
            if fields.values().any(|value| {
                matches!(
                    value,
                    ColumnValue::Null | ColumnValue::Binary(_) | ColumnValue::FixedList(_)
                )
            }) {
                return Ok(None);
            }
            all_fields.push(fields);
        }

        let (table_path, table_name) = self.get_current_table_info()?;
        let base_dir = table_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();

        if let Ok(index_mgr) = crate::storage::index::IndexManager::load(&table_name, &base_dir) {
            if !index_mgr.catalog_is_empty() {
                return Ok(None);
            }
        }

        self.persist_pending_overlay_for_table(&table_path, &table_name)?;
        let backend = self.get_backend_for_insert()?;

        if !backend.storage.is_v4_format() || backend.storage.has_constraints() {
            return Ok(None);
        }

        let schema = backend.storage.get_schema();
        if schema.is_empty() {
            return Ok(None);
        }
        let schema_len = schema.len();
        for fields in &all_fields {
            if schema_len != fields.len()
                || fields
                    .keys()
                    .any(|name| !schema.iter().any(|(schema_name, _)| schema_name == name))
            {
                return Ok(None);
            }
        }

        let durability = self.durability;
        let lock_file = if durability != DurabilityLevel::Fast {
            Some(
                Self::acquire_write_lock(&table_path)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?,
            )
        } else {
            None
        };

        let result = py.allow_threads(|| {
            backend
                .insert_column_rows_to_delta(&all_fields)
                .map_err(|e| PyIOError::new_err(e.to_string()))
        });

        if let Some(lf) = lock_file {
            Self::release_lock(lf);
        }

        let ids = result?;
        self.cached_backends
            .remove(&Self::backend_cache_key(&table_path, &table_name));
        crate::query::executor::invalidate_storage_cache(&table_path);
        crate::query::planner::invalidate_table_stats(&table_path.to_string_lossy());

        Ok(Some(ids.into_iter().map(|id| id as i64).collect()))
    }

    /// Direct durable single-row append for the same narrow schema-stable OLTP
    /// case as `store_one_delta()`, but with an immediate file sync so callers
    /// do not need a separate `flush()`.
    fn store_one_delta_durable(
        &self,
        py: Python<'_>,
        row: &Bound<'_, PyDict>,
    ) -> PyResult<Option<Vec<i64>>> {
        if row.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let fields = dict_to_column_values(row)?;
        if fields.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if fields.values().any(|value| {
            matches!(
                value,
                ColumnValue::Null | ColumnValue::Binary(_) | ColumnValue::FixedList(_)
            )
        }) {
            return Ok(None);
        }

        let (table_path, table_name) = self.get_current_table_info()?;
        let base_dir = table_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();

        if let Ok(index_mgr) = crate::storage::index::IndexManager::load(&table_name, &base_dir) {
            if !index_mgr.catalog_is_empty() {
                return Ok(None);
            }
        }

        self.persist_pending_overlay_for_table(&table_path, &table_name)?;
        let backend = self.get_backend_for_insert()?;

        if !backend.storage.is_v4_format() || backend.storage.has_constraints() {
            return Ok(None);
        }

        let schema = backend.storage.get_schema();
        if schema.is_empty() {
            return Ok(None);
        }
        let schema_len = schema.len();
        if schema_len != fields.len()
            || fields
                .keys()
                .any(|name| !schema.iter().any(|(schema_name, _)| schema_name == name))
        {
            return Ok(None);
        }

        let result = py.allow_threads(|| -> PyResult<Vec<u64>> {
            let ids = backend
                .insert_column_rows_to_delta(&[fields])
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            backend
                .sync()
                .map_err(|e| PyIOError::new_err(format!("Failed to durable-insert: {}", e)))?;
            Ok(ids)
        });

        let ids = result?;
        self.cached_backends
            .remove(&Self::backend_cache_key(&table_path, &table_name));
        crate::query::executor::invalidate_storage_cache(&table_path);
        crate::query::planner::invalidate_table_stats(&table_path.to_string_lossy());

        Ok(Some(ids.into_iter().map(|id| id as i64).collect()))
    }

    fn store_columnar(&self, py: Python<'_>, columns: &Bound<'_, PyDict>) -> PyResult<Vec<i64>> {
        if columns.is_empty() {
            return Ok(Vec::new());
        }

        // First pass: validate all columns have the same length
        let mut col_lengths: Vec<(String, usize)> = Vec::new();
        for (key, value) in columns.iter() {
            let col_name: String = key.extract()?;
            if col_name == "_id" {
                continue;
            }

            let list = value.downcast::<PyList>().map_err(|_| {
                PyValueError::new_err(format!("Column '{}' must be a list", col_name))
            })?;
            col_lengths.push((col_name, list.len()));
        }

        if col_lengths.is_empty() {
            return Ok(Vec::new());
        }

        // Check all columns have same length
        let first_len = col_lengths[0].1;
        for (name, len) in &col_lengths {
            if *len != first_len {
                return Err(PyValueError::new_err(format!(
                    "All columns must have the same length: '{}' has {} rows, expected {}",
                    name, len, first_len
                )));
            }
        }

        let num_rows = first_len;
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        // Separate columns by type with NULL tracking
        let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
        let mut binary_columns_map: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut fixedlist_columns_map: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();
        let mut null_positions: HashMap<String, Vec<bool>> = HashMap::new();

        for (key, value) in columns.iter() {
            let col_name: String = key.extract()?;
            if col_name == "_id" {
                continue;
            }

            let list = value.downcast::<PyList>().map_err(|_| {
                PyValueError::new_err(format!("Column '{}' must be a list", col_name))
            })?;

            let col_len = list.len();
            if col_len == 0 {
                continue;
            }

            // Detect type from first non-None element
            // NOTE: Check bool before int because in Python bool is a subclass of int
            // NOTE: Check bytes before string because PyBytes can also be extracted as str in some pyo3 versions
            let mut col_type: Option<&str> = None;
            for item in list.iter() {
                if !item.is_none() {
                    if item.extract::<bool>().is_ok()
                        && item.get_type().name().map_or(false, |n| n == "bool")
                    {
                        col_type = Some("bool");
                    } else if item.downcast::<pyo3::types::PyBytes>().is_ok() {
                        col_type = Some("bytes");
                    } else if item.get_type().name().map_or(false, |n| n == "ndarray") {
                        // Always use "fixedlist" for numpy arrays (any dtype).
                        // The fixedlist path calls .astype("float32").tobytes() which produces
                        // f32 bytes.  insert_typed_with_nulls_full then calls
                        // push_float16_list_from_f32() which does the single correct f32→f16
                        // conversion for Float16List columns.  Routing through "float16_vector"
                        // would produce f16 bytes here AND call push_float16_list_from_f32()
                        // again — causing a double conversion and garbled data.
                        col_type = Some("fixedlist");
                    } else if item
                        .downcast::<pyo3::types::PyList>()
                        .ok()
                        .and_then(|seq| seq.get_item(0).ok())
                        .map_or(false, |first| first.extract::<f64>().is_ok())
                    {
                        col_type = Some("fixedlist");
                    } else if item.extract::<i64>().is_ok() {
                        col_type = Some("int");
                    } else if item.extract::<f64>().is_ok() {
                        col_type = Some("float");
                    } else if item.extract::<String>().is_ok() {
                        col_type = Some("string");
                    }
                    break;
                }
            }

            match col_type {
                Some("int") => {
                    let mut vals = Vec::with_capacity(col_len);
                    let mut nulls = Vec::with_capacity(col_len);
                    for item in list.iter() {
                        let is_null = item.is_none();
                        nulls.push(is_null);
                        vals.push(if is_null {
                            0
                        } else {
                            item.extract::<i64>().unwrap_or(0)
                        });
                    }
                    int_columns.insert(col_name.clone(), vals);
                    null_positions.insert(col_name, nulls);
                }
                Some("float") => {
                    let mut vals = Vec::with_capacity(col_len);
                    let mut nulls = Vec::with_capacity(col_len);
                    for item in list.iter() {
                        let is_null = item.is_none();
                        nulls.push(is_null);
                        vals.push(if is_null {
                            0.0
                        } else {
                            item.extract::<f64>().unwrap_or(0.0)
                        });
                    }
                    float_columns.insert(col_name.clone(), vals);
                    null_positions.insert(col_name, nulls);
                }
                Some("bool") => {
                    let mut vals = Vec::with_capacity(col_len);
                    let mut nulls = Vec::with_capacity(col_len);
                    for item in list.iter() {
                        let is_null = item.is_none();
                        nulls.push(is_null);
                        vals.push(if is_null {
                            false
                        } else {
                            item.extract::<bool>().unwrap_or(false)
                        });
                    }
                    bool_columns.insert(col_name.clone(), vals);
                    null_positions.insert(col_name, nulls);
                }
                Some("bytes") => {
                    let mut vals: Vec<Vec<u8>> = Vec::with_capacity(col_len);
                    let mut nulls = Vec::with_capacity(col_len);
                    for item in list.iter() {
                        let is_null = item.is_none();
                        nulls.push(is_null);
                        if is_null {
                            vals.push(Vec::new());
                        } else if let Ok(b) = item.downcast::<pyo3::types::PyBytes>() {
                            vals.push(b.as_bytes().to_vec());
                        } else if let Ok(s) = item.extract::<Vec<u8>>() {
                            vals.push(s);
                        } else {
                            vals.push(Vec::new());
                        }
                    }
                    binary_columns_map.insert(col_name.clone(), vals);
                    null_positions.insert(col_name, nulls);
                }
                Some("fixedlist") => {
                    let mut vals: Vec<Vec<u8>> = Vec::with_capacity(col_len);
                    let mut nulls = Vec::with_capacity(col_len);
                    for item in list.iter() {
                        let is_null = item.is_none();
                        nulls.push(is_null);
                        if is_null {
                            vals.push(Vec::new());
                        } else if let Ok(bytes) = item
                            .call_method0("flatten")
                            .and_then(|flat| flat.call_method1("astype", ("float32",)))
                            .and_then(|f32arr| f32arr.call_method0("tobytes"))
                            .and_then(|b| b.extract::<Vec<u8>>())
                        {
                            vals.push(bytes);
                        } else if let Ok(seq) = item.downcast::<pyo3::types::PyList>() {
                            let mut bytes = Vec::with_capacity(seq.len() * 4);
                            for elem in seq.iter() {
                                let f = elem.extract::<f32>().unwrap_or(0.0);
                                bytes.extend_from_slice(&f.to_le_bytes());
                            }
                            vals.push(bytes);
                        } else {
                            vals.push(Vec::new());
                        }
                    }
                    fixedlist_columns_map.insert(col_name.clone(), vals);
                    null_positions.insert(col_name, nulls);
                }
                Some("float16_vector") => {
                    let mut vals: Vec<Vec<u8>> = Vec::with_capacity(col_len);
                    let mut nulls = Vec::with_capacity(col_len);
                    for item in list.iter() {
                        let is_null = item.is_none();
                        nulls.push(is_null);
                        if is_null {
                            vals.push(Vec::new());
                        } else if let Ok(f32_bytes) = item
                            .call_method0("flatten")
                            .and_then(|flat| flat.call_method1("astype", ("float32",)))
                            .and_then(|f32arr| f32arr.call_method0("tobytes"))
                            .and_then(|b| b.extract::<Vec<u8>>())
                        {
                            let f16_bytes: Vec<u8> = f32_bytes
                                .chunks_exact(4)
                                .flat_map(|c| {
                                    let f = f32::from_le_bytes(c.try_into().unwrap());
                                    crate::storage::on_demand::f32_to_f16(f).to_le_bytes()
                                })
                                .collect();
                            vals.push(f16_bytes);
                        } else {
                            vals.push(Vec::new());
                        }
                    }
                    fixedlist_columns_map.insert(col_name.clone(), vals);
                    null_positions.insert(col_name, nulls);
                }
                Some("string") | None => {
                    let mut vals = Vec::with_capacity(col_len);
                    let mut nulls = Vec::with_capacity(col_len);
                    for item in list.iter() {
                        let is_null = item.is_none();
                        nulls.push(is_null);
                        vals.push(if is_null {
                            String::new()
                        } else {
                            item.extract::<String>().unwrap_or_default()
                        });
                    }
                    string_columns.insert(col_name.clone(), vals);
                    null_positions.insert(col_name, nulls);
                }
                _ => {}
            }
        }

        if num_rows == 0 {
            return Ok(Vec::new());
        }

        let (table_path, table_name) = self.get_current_table_info()?;
        let durability = self.durability;
        self.persist_pending_overlay_for_table(&table_path, &table_name)?;

        // Skip file lock for 'fast' durability
        let lock_file = if durability != DurabilityLevel::Fast {
            Some(
                Self::acquire_write_lock(&table_path)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?,
            )
        } else {
            None
        };

        // Save a copy of string_columns for FTS indexing (before insert_typed consumes it)
        let string_columns_for_fts = string_columns.clone();

        // Use StorageEngine for unified write
        let result = py.allow_threads(|| {
            let engine = crate::storage::engine::engine();
            engine
                .write_typed(
                    &table_path,
                    int_columns,
                    float_columns,
                    string_columns,
                    binary_columns_map,
                    fixedlist_columns_map,
                    bool_columns,
                    null_positions,
                    durability,
                )
                .map_err(|e| PyIOError::new_err(e.to_string()))
        });

        if let Some(lf) = lock_file {
            Self::release_lock(lf);
        }

        // Invalidate local backend cache
        self.invalidate_backend(&table_name);
        // On Windows, engine.insert_cache holds a mmap'd backend after write_typed.
        // Clearing it ensures set_len() in subsequent transaction-commit delete paths succeeds
        // (ERROR_USER_MAPPED_FILE / os error 1224 is triggered when any mmap is open).
        #[cfg(target_os = "windows")]
        crate::storage::engine::engine().invalidate(&table_path);

        let ids = result?;

        // Index in FTS if enabled - OPTIMIZED: Use add_documents_arrow_str (🥈 zero-copy &str path)
        {
            let mgr = self.fts_manager.read();
            if mgr.is_some() {
                let table_name = self.current_table.read().clone();
                let index_fields = self.fts_index_fields.read().get(&table_name).cloned();

                if let Some(m) = mgr.as_ref() {
                    if let Ok(engine) = m.get_engine(&table_name) {
                        // Determine which string fields to index
                        let string_field_names: Vec<String> = match &index_fields {
                            Some(fields) => fields
                                .iter()
                                .cloned()
                                .filter(|f| string_columns_for_fts.contains_key(f))
                                .collect(),
                            None => string_columns_for_fts.keys().cloned().collect(),
                        };

                        if !string_field_names.is_empty() {
                            // Build owned String columns, then convert to &str for zero-copy call
                            let fts_columns: Vec<(String, Vec<String>)> = string_field_names
                                .iter()
                                .filter_map(|f| {
                                    string_columns_for_fts
                                        .get(f)
                                        .map(|v| (f.clone(), v.clone()))
                                })
                                .collect();

                            // 🥈 add_documents_arrow_str: zero-copy &str slices, ~3.3M docs/s
                            if !fts_columns.is_empty() {
                                let doc_ids_u32: Vec<u32> =
                                    ids.iter().map(|&id| id as u32).collect();
                                let columns_ref: Vec<(String, Vec<&str>)> = fts_columns
                                    .iter()
                                    .map(|(name, vals)| {
                                        (name.clone(), vals.iter().map(|s| s.as_str()).collect())
                                    })
                                    .collect();
                                let _ = py.allow_threads(|| {
                                    engine.add_documents_arrow_str(&doc_ids_u32, columns_ref)
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(ids.into_iter().map(|id| id as i64).collect())
    }

    /// Helper to index a document for FTS (single document - uses slower path)
    fn index_for_fts(&self, id: i64, data: &Bound<'_, PyDict>) -> PyResult<()> {
        let table_name = self.current_table.read().clone();
        let mgr = self.fts_manager.read();

        if mgr.is_none() {
            return Ok(());
        }

        // Get index fields config
        let index_fields = self.fts_index_fields.read().get(&table_name).cloned();

        // Build fields map from dict
        let mut fields = HashMap::new();
        for (key, value) in data.iter() {
            let key_str: String = key.extract()?;
            if key_str == "_id" {
                continue;
            }

            // Check if this field should be indexed
            let should_index = match &index_fields {
                Some(idx_fields) => idx_fields.contains(&key_str),
                None => value.extract::<String>().is_ok(), // Index all string fields by default
            };

            if should_index {
                if let Ok(s) = value.extract::<String>() {
                    fields.insert(key_str, s);
                }
            }
        }

        if fields.is_empty() {
            return Ok(());
        }

        // 🥇 Index the document via add_documents_arrow_texts (pre-joined text, zero-copy &str)
        if let Some(m) = mgr.as_ref() {
            if let Ok(engine) = m.get_engine(&table_name) {
                // Pre-join all field values into a single text (fastest path for single doc)
                let joined = fields.values().cloned().collect::<Vec<_>>().join(" ");
                let doc_id = id as u32;
                let _ = engine.add_documents_arrow_texts(&[doc_id], &[joined.as_str()]);
            }
        }

        Ok(())
    }

    /// Delete a record by ID using StorageEngine
    fn delete(&self, id: i64) -> PyResult<bool> {
        let (table_path, table_name) = self.get_current_table_info()?;
        let durability = self.durability;

        if id < 0 {
            return Ok(false);
        }

        let replace_cache_key = Self::replace_row_cache_key(&table_path, &table_name, id as u64);

        if durability == DurabilityLevel::Fast
            && !self.table_has_secondary_indexes(&table_path, &table_name)
        {
            let backend = self.get_backend_for_overlay(&table_path, &table_name)?;
            if !backend.storage.has_constraints() {
                if backend.delete_pending_v4_in_memory_row(id as u64) {
                    self.replace_exact_row_cache.remove(&replace_cache_key);
                    crate::query::executor::cache_backend_pub(&table_path, Arc::clone(&backend));
                    crate::query::planner::invalidate_table_stats(&table_path.to_string_lossy());
                    return Ok(true);
                }

                let result = backend
                    .delta_delete_row(id as u64)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                if result {
                    self.replace_exact_row_cache.remove(&replace_cache_key);
                    crate::query::executor::cache_backend_pub(&table_path, Arc::clone(&backend));
                    crate::query::planner::invalidate_table_stats(&table_path.to_string_lossy());
                }
                return Ok(result);
            }
        }

        // Skip file lock for 'fast' durability
        let lock_file = if durability != DurabilityLevel::Fast {
            Some(
                Self::acquire_write_lock(&table_path)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?,
            )
        } else {
            None
        };

        // Use StorageEngine for unified delete
        let engine = crate::storage::engine::engine();
        let result = engine
            .delete_one(&table_path, id as u64, durability)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        if let Some(lf) = lock_file {
            Self::release_lock(lf);
        }

        // Invalidate local backend cache
        self.invalidate_backend(&table_name);

        Ok(result)
    }

    /// Delete multiple records by IDs using StorageEngine
    fn delete_batch(&self, ids: Vec<i64>) -> PyResult<bool> {
        // Empty list is a successful no-op
        if ids.is_empty() {
            return Ok(true);
        }

        let (table_path, table_name) = self.get_current_table_info()?;
        let durability = self.durability;

        if durability == DurabilityLevel::Fast
            && !self.table_has_secondary_indexes(&table_path, &table_name)
        {
            let backend = self.get_backend_for_overlay(&table_path, &table_name)?;
            if !backend.storage.has_constraints() {
                let mut deleted = 0usize;
                for id in &ids {
                    if *id < 0 {
                        continue;
                    }
                    if backend.delete_pending_v4_in_memory_row(*id as u64) {
                        self.replace_exact_row_cache
                            .remove(&Self::replace_row_cache_key(
                                &table_path,
                                &table_name,
                                *id as u64,
                            ));
                        deleted += 1;
                        continue;
                    }
                    if backend
                        .delta_delete_row(*id as u64)
                        .map_err(|e| PyIOError::new_err(e.to_string()))?
                    {
                        self.replace_exact_row_cache
                            .remove(&Self::replace_row_cache_key(
                                &table_path,
                                &table_name,
                                *id as u64,
                            ));
                        deleted += 1;
                    }
                }
                if deleted > 0 {
                    crate::query::executor::cache_backend_pub(&table_path, Arc::clone(&backend));
                    crate::query::planner::invalidate_table_stats(&table_path.to_string_lossy());
                }
                return Ok(deleted > 0);
            }
        }

        // Skip file lock for 'fast' durability
        let lock_file = if durability != DurabilityLevel::Fast {
            Some(
                Self::acquire_write_lock(&table_path)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?,
            )
        } else {
            None
        };

        // Use StorageEngine for unified delete
        let engine = crate::storage::engine::engine();
        let ids_u64: Vec<u64> = ids.into_iter().map(|id| id as u64).collect();
        let deleted = engine
            .delete(&table_path, &ids_u64, durability)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        if let Some(lf) = lock_file {
            Self::release_lock(lf);
        }

        // Invalidate local backend cache
        self.invalidate_backend(&table_name);

        Ok(deleted > 0)
    }

    /// Delete records matching a WHERE clause
    /// Returns the number of deleted rows
    fn delete_where(&self, where_clause: &str) -> PyResult<i64> {
        let (table_path, table_name) = self.get_current_table_info()?;

        // Build DELETE SQL statement
        let sql = format!("DELETE FROM {} WHERE {}", table_name, where_clause);

        // Execute using ApexExecutor
        let base_dir = self.current_base_dir();
        crate::query::executor::set_query_root_dir(&self.root_dir);
        let exec_result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path);
        crate::query::executor::clear_query_root_dir();
        let result = exec_result.map_err(|e| PyIOError::new_err(e.to_string()))?;

        // Invalidate cached backend since data changed
        self.invalidate_backend(&table_name);
        // Invalidate StorageEngine cache so count_rows() sees updated state
        crate::storage::engine::engine().invalidate(&table_path);

        // Extract scalar result (number of deleted rows)
        match result {
            ApexResult::Scalar(count) => Ok(count),
            _ => Ok(0),
        }
    }

    /// Delete all records (no WHERE clause)
    /// Returns the number of deleted rows
    fn delete_all(&self) -> PyResult<i64> {
        let (table_path, table_name) = self.get_current_table_info()?;

        // Build DELETE SQL statement without WHERE
        let sql = format!("DELETE FROM {}", table_name);

        // Execute using ApexExecutor
        let base_dir = self.current_base_dir();
        crate::query::executor::set_query_root_dir(&self.root_dir);
        let exec_result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path);
        crate::query::executor::clear_query_root_dir();
        let result = exec_result.map_err(|e| PyIOError::new_err(e.to_string()))?;

        // Invalidate cached backend since data changed
        self.invalidate_backend(&table_name);
        // Invalidate StorageEngine cache so count_rows() sees updated state
        crate::storage::engine::engine().invalidate(&table_path);

        // Extract scalar result (number of deleted rows)
        match result {
            ApexResult::Scalar(count) => Ok(count),
            _ => Ok(0),
        }
    }

    /// Execute SQL query.
    ///
    /// Uses `QuerySignature::classify()` for single-point dispatch — no duplicate
    /// pattern matching. Read queries that need high performance should prefer
    /// `_execute_arrow_ffi()` from Python; this method is primarily for writes,
    /// transactions, and point lookups that benefit from `columns_dict` format.
    fn execute(&self, py: Python<'_>, sql: &str) -> PyResult<PyObject> {
        use crate::query::query_signature::{self, QuerySignature};

        let sig = query_signature::classify(sql);
        let is_write = matches!(&sig, QuerySignature::DmlWrite | QuerySignature::Ddl { .. });

        // Single read of current_table — avoids 3x RwLock acquire + String clone
        let table_name = self.current_table.read().clone();
        let base_dir = self.current_base_dir();
        let table_path = if table_name.is_empty() {
            base_dir.clone()
        } else {
            self.table_paths
                .read()
                .get(&table_name)
                .cloned()
                .unwrap_or_else(|| base_dir.join(format!("{}.apex", table_name)))
        };

        // ── ULTRA-FAST PATH: _id point lookup via cached_backends (warm) ──
        // Uses per-instance DashMap — zero PathBuf hashing, bypasses STORAGE_CACHE.
        if let QuerySignature::PointLookup { id, ref table } = &sig {
            let (target_table, target_path) =
                self.resolve_signature_table(table.as_deref(), &table_name, &table_path, &base_dir);
            let maybe_backend = self
                .cached_backends
                .get(&target_table)
                .map(|v| Arc::clone(&v));
            if let Some(backend) = maybe_backend {
                if backend.has_pending_deltas() {
                    // Fall back to the Arrow executor path below so DeltaMerger overlays updates.
                } else {
                    let rcix_result = py.allow_threads(|| backend.storage.retrieve_rcix(*id));
                    if let Ok(Some(vals)) = rcix_result {
                        let out = PyDict::new_bound(py);
                        let columns_dict = PyDict::new_bound(py);
                        for (col_name, val) in &vals {
                            let pyval = value_to_py(py, val)?;
                            columns_dict
                                .set_item(col_name.as_str(), PyList::new_bound(py, [pyval]))?;
                        }
                        out.set_item("columns_dict", columns_dict)?;
                        out.set_item("rows_affected", 0i64)?;
                        return Ok(out.into());
                    }
                }
            }
            if let Ok(backend) = crate::query::get_cached_backend_pub(&target_path) {
                self.cached_backends
                    .insert(target_table.clone(), Arc::clone(&backend));
                if !backend.has_pending_deltas() {
                    let rcix_result = py.allow_threads(|| backend.storage.retrieve_rcix(*id));
                    if let Ok(Some(vals)) = rcix_result {
                        let out = PyDict::new_bound(py);
                        let columns_dict = PyDict::new_bound(py);
                        for (col_name, val) in &vals {
                            let pyval = value_to_py(py, val)?;
                            columns_dict
                                .set_item(col_name.as_str(), PyList::new_bound(py, [pyval]))?;
                        }
                        out.set_item("columns_dict", columns_dict)?;
                        out.set_item("rows_affected", 0i64)?;
                        return Ok(out.into());
                    }
                }
                // Fallback: Arrow batch path
                if let Ok(Some(batch)) = backend.read_row_by_id_to_arrow(*id) {
                    if batch.num_rows() > 0 {
                        let out = PyDict::new_bound(py);
                        let columns_dict = PyDict::new_bound(py);
                        let schema = batch.schema();
                        for col_idx in 0..batch.num_columns() {
                            let col_name = schema.field(col_idx).name();
                            let arr = batch.column(col_idx);
                            let vals_1row: Vec<_> = (0..batch.num_rows())
                                .map(|r| value_to_py(py, &arrow_value_at(arr, r)))
                                .collect::<PyResult<_>>()?;
                            columns_dict.set_item(col_name, PyList::new_bound(py, vals_1row))?;
                        }
                        out.set_item("columns_dict", columns_dict)?;
                        out.set_item("rows_affected", 0i64)?;
                        return Ok(out.into());
                    }
                }
            }
        }

        // ── ULTRA-FAST PATH: projected _id point lookup ──
        // Keep OLTP-style `SELECT col1, col2 ... WHERE _id = N` on the same
        // direct rcix path as SELECT *, avoiding an intermediate Arrow batch.
        if let QuerySignature::ProjectedPointLookup {
            id,
            ref table,
            columns,
        } = &sig
        {
            let (target_table, target_path) =
                self.resolve_signature_table(table.as_deref(), &table_name, &table_path, &base_dir);
            let maybe_backend = self
                .cached_backends
                .get(&target_table)
                .map(|v| Arc::clone(&v))
                .or_else(|| {
                    crate::query::get_cached_backend_pub(&target_path)
                        .ok()
                        .map(|b| {
                            self.cached_backends
                                .insert(target_table.clone(), Arc::clone(&b));
                            b
                        })
                });

            if let Some(backend) = maybe_backend {
                if !backend.has_pending_deltas() {
                    let projected_cols: Vec<&str> = columns.iter().map(String::as_str).collect();
                    let rcix_result = py.allow_threads(|| {
                        backend
                            .storage
                            .retrieve_rcix_projected(*id, &projected_cols)
                    });
                    let vals_result = match rcix_result {
                        Ok(Some(vals)) => Some(vals),
                        _ => py
                            .allow_threads(|| backend.storage.read_row_by_id_values(*id))
                            .ok()
                            .flatten(),
                    };
                    if let Some(vals) = vals_result {
                        let out = PyDict::new_bound(py);
                        if let Some(columns_dict) =
                            projected_values_to_columns_dict(py, &vals, columns)?
                        {
                            out.set_item("columns_dict", columns_dict)?;
                            out.set_item("rows_affected", 0i64)?;
                            return Ok(out.into());
                        }
                    }
                }
            }
        }

        // ── FAST PATH: SELECT * ... WHERE _id IN (...) ──
        if let QuerySignature::IdBatchLookup { ids, ref table } = &sig {
            let (target_table, target_path) =
                self.resolve_signature_table(table.as_deref(), &table_name, &table_path, &base_dir);
            let maybe_backend = self
                .cached_backends
                .get(&target_table)
                .map(|v| Arc::clone(&v))
                .or_else(|| {
                    crate::query::get_cached_backend_pub(&target_path)
                        .ok()
                        .map(|b| {
                            self.cached_backends
                                .insert(target_table.clone(), Arc::clone(&b));
                            b
                        })
                });

            if let Some(backend) = maybe_backend {
                let sorted_ids = sort_and_dedupe_ids(ids);
                let batch_result =
                    py.allow_threads(|| backend.read_rows_by_ids_to_arrow(&sorted_ids));
                if let Ok(batch) = batch_result {
                    let batch = if batch.num_rows() > 0 {
                        batch
                    } else if let Ok(empty) = backend.read_columns_to_arrow(None, 0, Some(0)) {
                        empty
                    } else {
                        batch
                    };
                    let out = PyDict::new_bound(py);
                    let columns_dict = PyDict::new_bound(py);
                    let schema = batch.schema();
                    for col_idx in 0..batch.num_columns() {
                        let col_name = schema.field(col_idx).name();
                        let arr = batch.column(col_idx);
                        let col_list = arrow_col_to_pylist(py, arr)?;
                        columns_dict.set_item(col_name, col_list)?;
                    }
                    out.set_item("columns_dict", columns_dict)?;
                    out.set_item("rows_affected", 0i64)?;
                    return Ok(out.into());
                }
            }
        }

        // ── FAST PATH: SELECT ... WHERE string_col = 'value' LIMIT N [OFFSET M] ──
        if let QuerySignature::StringEqualityFilterLimit {
            ref table,
            column,
            value,
            limit,
            offset,
        } = &sig
        {
            let (target_table, target_path) =
                self.resolve_signature_table(table.as_deref(), &table_name, &table_path, &base_dir);
            let maybe_backend = self
                .cached_backends
                .get(&target_table)
                .map(|v| Arc::clone(&v))
                .or_else(|| {
                    crate::query::get_cached_backend_pub(&target_path)
                        .ok()
                        .map(|b| {
                            self.cached_backends
                                .insert(target_table.clone(), Arc::clone(&b));
                            b
                        })
                });

            if let Some(backend) = maybe_backend {
                let can_use_limit_scan = !backend.has_pending_deltas()
                    || (backend.pending_delta_delete_count() == 0
                        && !backend.pending_delta_updates_column(column));
                if can_use_limit_scan {
                    if *limit == 1 && *offset == 0 {
                        let row_id_result =
                            py.allow_threads(|| backend.first_row_id_for_string_eq(column, value));
                        if let Ok(Some(row_id)) = row_id_result {
                            let vals_result = py
                                .allow_threads(|| backend.storage.retrieve_rcix(row_id))
                                .ok()
                                .flatten()
                                .or_else(|| {
                                    py.allow_threads(|| {
                                        backend.storage.read_row_by_id_values(row_id)
                                    })
                                    .ok()
                                    .flatten()
                                });
                            if let Some(vals) = vals_result {
                                let out = PyDict::new_bound(py);
                                let columns_dict = values_to_columns_dict(py, &vals)?;
                                out.set_item("columns_dict", columns_dict)?;
                                out.set_item("rows_affected", 0i64)?;
                                return Ok(out.into());
                            }
                        }
                    }

                    let batch_result = py.allow_threads(|| {
                        backend.read_columns_filtered_string_with_limit_to_arrow(
                            None, column, value, true, *limit, *offset,
                        )
                    });
                    if let Ok(batch) = batch_result {
                        let out = PyDict::new_bound(py);
                        let columns_dict = PyDict::new_bound(py);
                        let schema = batch.schema();
                        for col_idx in 0..batch.num_columns() {
                            let col_name = schema.field(col_idx).name();
                            let arr = batch.column(col_idx);
                            let col_list = arrow_col_to_pylist(py, arr)?;
                            columns_dict.set_item(col_name, col_list)?;
                        }
                        out.set_item("columns_dict", columns_dict)?;
                        out.set_item("rows_affected", 0i64)?;
                        return Ok(out.into());
                    }
                }
            }
        }

        // ── FAST PATH: projected string equality + LIMIT ──
        if let QuerySignature::ProjectedStringEqualityFilterLimit {
            ref table,
            columns,
            column,
            value,
            limit,
            offset,
        } = &sig
        {
            let (target_table, target_path) =
                self.resolve_signature_table(table.as_deref(), &table_name, &table_path, &base_dir);
            let maybe_backend = self
                .cached_backends
                .get(&target_table)
                .map(|v| Arc::clone(&v))
                .or_else(|| {
                    crate::query::get_cached_backend_pub(&target_path)
                        .ok()
                        .map(|b| {
                            self.cached_backends
                                .insert(target_table.clone(), Arc::clone(&b));
                            b
                        })
                });

            if let Some(backend) = maybe_backend {
                let can_use_limit_scan = !backend.has_pending_deltas()
                    || (backend.pending_delta_delete_count() == 0
                        && !backend.pending_delta_updates_column(column));
                if can_use_limit_scan {
                    if *limit == 1 && *offset == 0 {
                        let row_id_result =
                            py.allow_threads(|| backend.first_row_id_for_string_eq(column, value));
                        if let Ok(Some(row_id)) = row_id_result {
                            let vals_result = py
                                .allow_threads(|| backend.storage.retrieve_rcix(row_id))
                                .ok()
                                .flatten()
                                .or_else(|| {
                                    py.allow_threads(|| {
                                        backend.storage.read_row_by_id_values(row_id)
                                    })
                                    .ok()
                                    .flatten()
                                });
                            if let Some(vals) = vals_result {
                                if let Some(columns_dict) =
                                    projected_values_to_columns_dict(py, &vals, columns)?
                                {
                                    let out = PyDict::new_bound(py);
                                    out.set_item("columns_dict", columns_dict)?;
                                    out.set_item("rows_affected", 0i64)?;
                                    return Ok(out.into());
                                }
                            }
                        }
                    }

                    let col_refs: Vec<&str> = columns.iter().map(String::as_str).collect();
                    let batch_result = py.allow_threads(|| {
                        backend.read_columns_filtered_string_with_limit_to_arrow(
                            Some(col_refs.as_slice()),
                            column,
                            value,
                            true,
                            *limit,
                            *offset,
                        )
                    });
                    if let Ok(batch) = batch_result {
                        let out = PyDict::new_bound(py);
                        let columns_dict = PyDict::new_bound(py);
                        let schema = batch.schema();
                        for col_idx in 0..batch.num_columns() {
                            let col_name = schema.field(col_idx).name();
                            let arr = batch.column(col_idx);
                            let col_list = arrow_col_to_pylist(py, arr)?;
                            columns_dict.set_item(col_name, col_list)?;
                        }
                        out.set_item("columns_dict", columns_dict)?;
                        out.set_item("rows_affected", 0i64)?;
                        return Ok(out.into());
                    }
                }
            }
        }

        // ── FAST PATH: SELECT * ... WHERE numeric_col <op> value LIMIT N [OFFSET M] ──
        if let QuerySignature::NumericRangeFilterLimit {
            ref table,
            column,
            low,
            high,
            limit,
            offset,
        } = &sig
        {
            if self.current_txn_id.read().is_none() {
                let (target_table, target_path) = self.resolve_signature_table(
                    table.as_deref(),
                    &table_name,
                    &table_path,
                    &base_dir,
                );
                let maybe_backend = self
                    .cached_backends
                    .get(&target_table)
                    .map(|v| Arc::clone(&v))
                    .or_else(|| {
                        crate::query::get_cached_backend_pub(&target_path)
                            .ok()
                            .map(|b| {
                                self.cached_backends
                                    .insert(target_table.clone(), Arc::clone(&b));
                                b
                            })
                    });

                if let Some(backend) = maybe_backend {
                    if !backend.has_pending_deltas() && !backend.has_delta() {
                        let needed = (*offset).saturating_add(*limit);
                        let cols_result = py.allow_threads(
                            || -> std::io::Result<
                                Option<crate::storage::on_demand::MmapBatchColumns>,
                            > {
                                let Some(indices) = backend.scan_numeric_range_mmap(
                                    column,
                                    *low,
                                    *high,
                                    Some(needed),
                                )?
                                else {
                                    return Ok(None);
                                };
                                let final_indices: Vec<usize> =
                                    indices.into_iter().skip(*offset).take(*limit).collect();
                                backend
                                    .storage
                                    .extract_rows_by_indices_mmap_columns(&final_indices, None)
                            }
                        );

                        if let Ok(Some(batch_cols)) = cols_result {
                            if let Some(columns_dict) =
                                mmap_batch_columns_to_pydict(py, batch_cols, None)?
                            {
                                let out = PyDict::new_bound(py);
                                out.set_item("columns_dict", columns_dict)?;
                                out.set_item("rows_affected", 0i64)?;
                                return Ok(out.into());
                            }
                        } else if let Ok(Some(final_indices)) =
                            py.allow_threads(|| -> std::io::Result<Option<Vec<usize>>> {
                                let Some(indices) = backend.scan_numeric_range_mmap(
                                    column,
                                    *low,
                                    *high,
                                    Some(needed),
                                )?
                                else {
                                    return Ok(None);
                                };
                                Ok(Some(
                                    indices.into_iter().skip(*offset).take(*limit).collect(),
                                ))
                            })
                        {
                            let batch_result =
                                py.allow_threads(|| -> std::io::Result<Option<RecordBatch>> {
                                    if final_indices.is_empty() {
                                        backend.read_columns_to_arrow(None, 0, Some(0)).map(Some)
                                    } else {
                                        backend
                                            .read_columns_by_indices_to_arrow(&final_indices, None)
                                            .map(Some)
                                    }
                                });

                            if let Ok(Some(batch)) = batch_result {
                                let out = PyDict::new_bound(py);
                                let columns_dict = PyDict::new_bound(py);
                                let schema = batch.schema();
                                for col_idx in 0..batch.num_columns() {
                                    let col_name = schema.field(col_idx).name();
                                    let arr = batch.column(col_idx);
                                    let col_list = arrow_col_to_pylist(py, arr)?;
                                    columns_dict.set_item(col_name, col_list)?;
                                }
                                out.set_item("columns_dict", columns_dict)?;
                                out.set_item("rows_affected", 0i64)?;
                                return Ok(out.into());
                            }
                        }
                    }
                }
            }
        }

        // ── FAST PATH: projected numeric comparison + LIMIT ──
        if let QuerySignature::ProjectedNumericRangeFilterLimit {
            ref table,
            columns,
            column,
            low,
            high,
            limit,
            offset,
        } = &sig
        {
            if self.current_txn_id.read().is_none() {
                let (target_table, target_path) = self.resolve_signature_table(
                    table.as_deref(),
                    &table_name,
                    &table_path,
                    &base_dir,
                );
                let maybe_backend = self
                    .cached_backends
                    .get(&target_table)
                    .map(|v| Arc::clone(&v))
                    .or_else(|| {
                        crate::query::get_cached_backend_pub(&target_path)
                            .ok()
                            .map(|b| {
                                self.cached_backends
                                    .insert(target_table.clone(), Arc::clone(&b));
                                b
                            })
                    });

                if let Some(backend) = maybe_backend {
                    if !backend.has_pending_deltas() && !backend.has_delta() {
                        let needed = (*offset).saturating_add(*limit);
                        let col_refs: Vec<&str> = columns.iter().map(String::as_str).collect();
                        let cols_result = py.allow_threads(
                            || -> std::io::Result<
                                Option<crate::storage::on_demand::MmapBatchColumns>,
                            > {
                                let Some(indices) = backend.scan_numeric_range_mmap(
                                    column,
                                    *low,
                                    *high,
                                    Some(needed),
                                )?
                                else {
                                    return Ok(None);
                                };
                                let final_indices: Vec<usize> =
                                    indices.into_iter().skip(*offset).take(*limit).collect();
                                backend.storage.extract_rows_by_indices_mmap_columns(
                                    &final_indices,
                                    Some(col_refs.as_slice()),
                                )
                            }
                        );

                        if let Ok(Some(batch_cols)) = cols_result {
                            if let Some(columns_dict) =
                                mmap_batch_columns_to_pydict(py, batch_cols, Some(columns))?
                            {
                                let out = PyDict::new_bound(py);
                                out.set_item("columns_dict", columns_dict)?;
                                out.set_item("rows_affected", 0i64)?;
                                return Ok(out.into());
                            }
                        } else if let Ok(Some(final_indices)) =
                            py.allow_threads(|| -> std::io::Result<Option<Vec<usize>>> {
                                let Some(indices) = backend.scan_numeric_range_mmap(
                                    column,
                                    *low,
                                    *high,
                                    Some(needed),
                                )?
                                else {
                                    return Ok(None);
                                };
                                Ok(Some(
                                    indices.into_iter().skip(*offset).take(*limit).collect(),
                                ))
                            })
                        {
                            let batch_result =
                                py.allow_threads(|| -> std::io::Result<Option<RecordBatch>> {
                                    if final_indices.is_empty() {
                                        backend
                                            .read_columns_to_arrow(
                                                Some(col_refs.as_slice()),
                                                0,
                                                Some(0),
                                            )
                                            .map(Some)
                                    } else {
                                        backend
                                            .read_columns_by_indices_to_arrow(
                                                &final_indices,
                                                Some(col_refs.as_slice()),
                                            )
                                            .map(Some)
                                    }
                                });

                            if let Ok(Some(batch)) = batch_result {
                                let out = PyDict::new_bound(py);
                                let columns_dict = PyDict::new_bound(py);
                                let schema = batch.schema();
                                for col_idx in 0..batch.num_columns() {
                                    let col_name = schema.field(col_idx).name();
                                    let arr = batch.column(col_idx);
                                    let col_list = arrow_col_to_pylist(py, arr)?;
                                    columns_dict.set_item(col_name, col_list)?;
                                }
                                out.set_item("columns_dict", columns_dict)?;
                                out.set_item("rows_affected", 0i64)?;
                                return Ok(out.into());
                            }
                        }
                    }
                }
            }
        }

        // ── FAST PATH: SELECT * LIMIT N — pread RCIX, returns columnar dict ──
        if let QuerySignature::SimpleScanLimit {
            limit,
            offset,
            ref table,
        } = &sig
        {
            let (_, target_path) =
                self.resolve_signature_table(table.as_deref(), &table_name, &table_path, &base_dir);
            if let Ok(backend) = crate::query::get_cached_backend_pub(&target_path) {
                if backend.pending_v4_in_memory_rows() == 0 {
                    let limit = *limit;
                    let offset = *offset;
                    let batch_result = py.allow_threads(|| {
                        if offset > 0 {
                            if backend.has_pending_deltas()
                                || backend.has_delta()
                                || backend.active_row_count() != backend.row_count()
                            {
                                Ok(None)
                            } else {
                                let end = offset
                                    .saturating_add(limit)
                                    .min(backend.row_count() as usize);
                                let indices: Vec<usize> = (offset..end).collect();
                                backend
                                    .read_columns_by_indices_to_arrow(&indices, None)
                                    .map(Some)
                            }
                        } else {
                            match backend.storage.get_or_load_footer() {
                                Ok(Some(footer)) => {
                                    let col_indices: Vec<usize> =
                                        (0..footer.schema.column_count()).collect();
                                    backend.storage.to_arrow_batch_pread_rcix(
                                        &col_indices,
                                        true,
                                        limit,
                                    )
                                }
                                _ => Ok(None),
                            }
                        }
                    });
                    if let Ok(Some(batch)) = batch_result {
                        if batch.num_rows() > 0 {
                            let out = PyDict::new_bound(py);
                            let columns_dict = PyDict::new_bound(py);
                            let schema = batch.schema();
                            for col_idx in 0..batch.num_columns() {
                                let col_name = schema.field(col_idx).name();
                                let arr = batch.column(col_idx);
                                let col_list = arrow_col_to_pylist(py, arr)?;
                                columns_dict.set_item(col_name, col_list)?;
                            }
                            out.set_item("columns_dict", columns_dict)?;
                            out.set_item("rows_affected", 0i64)?;
                            return Ok(out.into());
                        }
                    }
                }
            }
        }

        // ── FAST PATH: SELECT col1, col2 FROM table — projected full scan ──
        if let QuerySignature::ProjectedFullScan { ref table, columns } = &sig {
            if self.current_txn_id.read().is_none() {
                let (_, target_path) = self.resolve_signature_table(
                    table.as_deref(),
                    &table_name,
                    &table_path,
                    &base_dir,
                );
                if let Ok(backend) = crate::query::get_cached_backend_pub(&target_path) {
                    if backend.pending_v4_in_memory_rows() == 0 {
                        let col_refs: Vec<&str> = columns.iter().map(String::as_str).collect();
                        let batch_result = py.allow_threads(|| {
                            backend.read_columns_to_arrow(Some(col_refs.as_slice()), 0, None)
                        });
                        if let Ok(batch) = batch_result {
                            if batch.num_rows() > 0 {
                                let out = PyDict::new_bound(py);
                                let columns_dict = PyDict::new_bound(py);
                                let schema = batch.schema();
                                for col_idx in 0..batch.num_columns() {
                                    let col_name = schema.field(col_idx).name();
                                    let arr = batch.column(col_idx);
                                    let col_list = arrow_col_to_pylist(py, arr)?;
                                    columns_dict.set_item(col_name, col_list)?;
                                }
                                out.set_item("columns_dict", columns_dict)?;
                                out.set_item("rows_affected", 0i64)?;
                                return Ok(out.into());
                            }
                        }
                    }
                }
            }
        }

        // ── FAST PATH: SELECT * WHERE col > N LIMIT M — numeric range filter ──
        if let QuerySignature::NumericRangeFilterLimit {
            ref table,
            column,
            low,
            high,
            limit,
            offset,
        } = &sig
        {
            if self.current_txn_id.read().is_none() {
                let (_, target_path) = self.resolve_signature_table(
                    table.as_deref(),
                    &table_name,
                    &table_path,
                    &base_dir,
                );
                if let Ok(backend) = crate::query::get_cached_backend_pub(&target_path) {
                    if !backend.has_pending_deltas()
                        && !backend.has_delta()
                        && backend.pending_v4_in_memory_rows() == 0
                    {
                        let needed = (*offset).saturating_add(*limit);
                        let cols_result = py.allow_threads(
                            || -> std::io::Result<
                                Option<crate::storage::on_demand::MmapBatchColumns>,
                            > {
                                let Some(indices) = backend.scan_numeric_range_mmap(
                                    column,
                                    *low,
                                    *high,
                                    Some(needed),
                                )?
                                else {
                                    return Ok(None);
                                };
                                let final_indices: Vec<usize> =
                                    indices.into_iter().skip(*offset).take(*limit).collect();
                                backend
                                    .storage
                                    .extract_rows_by_indices_mmap_columns(&final_indices, None)
                            }
                        );
                        if let Ok(Some(batch_cols)) = cols_result {
                            if let Some(columns_dict) =
                                mmap_batch_columns_to_pydict(py, batch_cols, None)?
                            {
                                let out = PyDict::new_bound(py);
                                out.set_item("columns_dict", columns_dict)?;
                                out.set_item("rows_affected", 0i64)?;
                                return Ok(out.into());
                            }
                        } else if let Ok(Some(indices)) = py.allow_threads(|| {
                            backend.scan_numeric_range_mmap(column, *low, *high, Some(needed))
                        }) {
                            let final_indices: Vec<usize> =
                                indices.into_iter().skip(*offset).take(*limit).collect();
                            let batch_result = py.allow_threads(|| {
                                if final_indices.is_empty() {
                                    backend.read_columns_to_arrow(None, 0, Some(0))
                                } else {
                                    backend.read_columns_by_indices_to_arrow(&final_indices, None)
                                }
                            });
                            if let Ok(batch) = batch_result {
                                if batch.num_rows() > 0 {
                                    let out = PyDict::new_bound(py);
                                    let columns_dict = PyDict::new_bound(py);
                                    let schema = batch.schema();
                                    for col_idx in 0..batch.num_columns() {
                                        let col_name = schema.field(col_idx).name();
                                        let arr = batch.column(col_idx);
                                        let col_list = arrow_col_to_pylist(py, arr)?;
                                        columns_dict.set_item(col_name, col_list)?;
                                    }
                                    out.set_item("columns_dict", columns_dict)?;
                                    out.set_item("rows_affected", 0i64)?;
                                    return Ok(out.into());
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── FAST PATH: Filtered string equality aggregation (pre-parse) ──
        if let QuerySignature::FilteredStringAgg {
            ref table,
            ref filter_column,
            ref filter_value,
        } = &sig
        {
            if self.current_txn_id.read().is_none() {
                let (_, target_path) = self.resolve_signature_table(
                    table.as_deref(),
                    &table_name,
                    &table_path,
                    &base_dir,
                );
                if let Ok(backend) = crate::query::get_cached_backend_pub(&target_path) {
                    if backend.is_mmap_only()
                        && !backend.has_pending_deltas()
                        && !backend.has_delta()
                        && backend.pending_v4_in_memory_rows() == 0
                    {
                        let filter_col = filter_column.clone();
                        let filter_val = filter_value.clone();
                        // Parse aggregation expressions from SQL
                        if let Some(agg_exprs) = parse_agg_select(sql) {
                            // Collect unique columns needed by the storage fast path.
                            // Add "*" when COUNT(*) / COUNT(1) is present so the storage
                            // layer returns the true match count without an extra scan.
                            let mut unique_cols: Vec<String> = Vec::new();
                            for (func, col, _alias) in &agg_exprs {
                                let is_count_star = func == "COUNT"
                                    && col
                                        .as_ref()
                                        .map(|c| {
                                            c == "*"
                                                || c.chars()
                                                    .next()
                                                    .map(|ch| ch.is_ascii_digit())
                                                    .unwrap_or(false)
                                        })
                                        .unwrap_or(true);
                                if is_count_star {
                                    if !unique_cols.iter().any(|c| c == "*") {
                                        unique_cols.push("*".to_string());
                                    }
                                } else if let Some(c) = col {
                                    if !unique_cols.contains(c) {
                                        unique_cols.push(c.clone());
                                    }
                                }
                            }
                            let col_refs: Vec<&str> =
                                unique_cols.iter().map(|s| s.as_str()).collect();
                            // Single-pass: scan string filter + aggregate in one sequential pass
                            let agg_result = py.allow_threads(|| {
                                backend.execute_filtered_string_agg_mmap(
                                    &filter_col,
                                    &filter_val,
                                    &col_refs,
                                )
                            });
                            if let Ok(Some(stats)) = agg_result {
                                // Build stat lookup: column name -> (count, sum, min, max, is_int)
                                let mut stat_map: std::collections::HashMap<
                                    &str,
                                    (i64, f64, f64, f64, bool),
                                > = std::collections::HashMap::new();
                                for (i, col_name) in col_refs.iter().enumerate() {
                                    if i < stats.len() {
                                        stat_map.insert(col_name, stats[i]);
                                    }
                                }
                                let match_count = stat_map.get("*").map(|s| s.0).unwrap_or(0);

                                let out = PyDict::new_bound(py);
                                let columns_dict = PyDict::new_bound(py);
                                for (func, col, alias) in &agg_exprs {
                                    let output_name = if let Some(a) = alias {
                                        a.clone()
                                    } else if let Some(c) = col {
                                        format!("{}({})", func, c)
                                    } else {
                                        format!("{}(*)", func)
                                    };
                                    match func.as_str() {
                                        "COUNT" => {
                                            let count = if let Some(c) = col {
                                                let is_count_star = c == "*"
                                                    || c.chars()
                                                        .next()
                                                        .map(|ch| ch.is_ascii_digit())
                                                        .unwrap_or(false);
                                                if is_count_star {
                                                    match_count
                                                } else {
                                                    stat_map
                                                        .get(c.as_str())
                                                        .map(|s| s.0)
                                                        .unwrap_or(0)
                                                }
                                            } else {
                                                match_count
                                            };
                                            columns_dict.set_item(
                                                &output_name,
                                                PyList::new_bound(py, &[count]),
                                            )?;
                                        }
                                        "SUM" | "AVG" | "MIN" | "MAX" => {
                                            if let Some(c) = col {
                                                let (count, sum, min_v, max_v, is_int) = stat_map
                                                    .get(c.as_str())
                                                    .copied()
                                                    .unwrap_or((0, 0.0, 0.0, 0.0, false));
                                                match func.as_str() {
                                                    "SUM" => {
                                                        if is_int {
                                                            columns_dict.set_item(
                                                                &output_name,
                                                                PyList::new_bound(
                                                                    py,
                                                                    &[sum as i64],
                                                                ),
                                                            )?;
                                                        } else {
                                                            columns_dict.set_item(
                                                                &output_name,
                                                                PyList::new_bound(py, &[sum]),
                                                            )?;
                                                        }
                                                    }
                                                    "AVG" => {
                                                        let avg = if count > 0 {
                                                            sum / count as f64
                                                        } else {
                                                            0.0
                                                        };
                                                        columns_dict.set_item(
                                                            &output_name,
                                                            PyList::new_bound(py, &[avg]),
                                                        )?;
                                                    }
                                                    "MIN" => {
                                                        if is_int {
                                                            columns_dict.set_item(
                                                                &output_name,
                                                                PyList::new_bound(
                                                                    py,
                                                                    &[min_v as i64],
                                                                ),
                                                            )?;
                                                        } else {
                                                            columns_dict.set_item(
                                                                &output_name,
                                                                PyList::new_bound(py, &[min_v]),
                                                            )?;
                                                        }
                                                    }
                                                    "MAX" => {
                                                        if is_int {
                                                            columns_dict.set_item(
                                                                &output_name,
                                                                PyList::new_bound(
                                                                    py,
                                                                    &[max_v as i64],
                                                                ),
                                                            )?;
                                                        } else {
                                                            columns_dict.set_item(
                                                                &output_name,
                                                                PyList::new_bound(py, &[max_v]),
                                                            )?;
                                                        }
                                                    }
                                                    _ => {}
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                                out.set_item("columns_dict", columns_dict)?;
                                out.set_item("rows_affected", 0i64)?;
                                return Ok(out.into());
                            }
                        }
                    }
                }
            }
        }

        // ── Transaction handling (single uppercase pass) ──
        let is_txn = matches!(&sig, QuerySignature::Transaction);
        let txn_upper = if is_txn {
            sql.trim().to_ascii_uppercase()
        } else {
            String::new()
        };
        let is_begin = is_txn && txn_upper.starts_with("BEGIN");
        let is_commit = is_txn && (txn_upper == "COMMIT" || txn_upper == "COMMIT;");
        let is_rollback = is_txn
            && (txn_upper == "ROLLBACK" || txn_upper == "ROLLBACK;")
            && !txn_upper.starts_with("ROLLBACK TO");
        let is_savepoint = is_txn && txn_upper.starts_with("SAVEPOINT ");
        let is_rollback_to = is_txn && txn_upper.starts_with("ROLLBACK TO");
        let is_release = is_txn && txn_upper.starts_with("RELEASE");

        let current_txn = *self.current_txn_id.read();
        let is_txn_dml = current_txn.is_some() && matches!(&sig, QuerySignature::DmlWrite);
        let is_txn_select = current_txn.is_some()
            && !is_write
            && !is_txn
            && matches!(
                &sig,
                QuerySignature::Complex
                    | QuerySignature::CountStar { .. }
                    | QuerySignature::PointLookup { .. }
                    | QuerySignature::ProjectedPointLookup { .. }
                    | QuerySignature::SimpleScanLimit { .. }
                    | QuerySignature::ProjectedScanLimit { .. }
                    | QuerySignature::IdBatchLookup { .. }
                    | QuerySignature::ProjectedIdBatchLookup { .. }
                    | QuerySignature::FullScan { .. }
                    | QuerySignature::ProjectedFullScan { .. }
                    | QuerySignature::StringEqualityFilter { .. }
                    | QuerySignature::StringEqualityFilterLimit { .. }
                    | QuerySignature::ProjectedStringEqualityFilter { .. }
                    | QuerySignature::ProjectedStringEqualityFilterLimit { .. }
                    | QuerySignature::NumericRangeFilterLimit { .. }
                    | QuerySignature::ProjectedNumericRangeFilterLimit { .. }
                    | QuerySignature::LikeFilter { .. }
                    | QuerySignature::FilteredStringAgg { .. }
                    | QuerySignature::TableFunction
            );

        let sql = sql.to_string();
        crate::query::executor::set_query_root_dir(&self.root_dir);
        crate::query::executor::set_temp_dir(&self.temp_dir);

        // Return enum to avoid per-cell arrow_value_at inside allow_threads
        enum ExecOut {
            Scalar(String, i64), // key, value — for txn commands
            Batch(RecordBatch),  // data result — columnar conversion with GIL
            Empty,               // no result
        }

        let exec_out = py.allow_threads(|| -> PyResult<ExecOut> {
            if is_begin {
                let result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                if let ApexResult::Scalar(txn_id) = &result {
                    return Ok(ExecOut::Scalar("txn_id".to_string(), *txn_id));
                }
                return Ok(ExecOut::Empty);
            }

            if is_commit {
                if let Some(txn_id) = current_txn {
                    let result = ApexExecutor::execute_commit_txn(txn_id, &base_dir, &table_path)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    if let ApexResult::Scalar(n) = &result {
                        return Ok(ExecOut::Scalar("rows_applied".to_string(), *n));
                    }
                }
                return Ok(ExecOut::Empty);
            }

            if is_rollback {
                if let Some(txn_id) = current_txn {
                    ApexExecutor::execute_rollback_txn(txn_id)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                }
                return Ok(ExecOut::Empty);
            }

            if is_savepoint {
                if let Some(txn_id) = current_txn {
                    let name = sql
                        .trim()
                        .strip_prefix("SAVEPOINT ")
                        .or_else(|| sql.trim().strip_prefix("savepoint "))
                        .unwrap_or("")
                        .trim()
                        .trim_end_matches(';')
                        .to_string();
                    let mgr = crate::txn::txn_manager();
                    mgr.with_context(txn_id, |ctx| {
                        ctx.savepoint(&name);
                        Ok(())
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                }
                return Ok(ExecOut::Empty);
            }

            if is_rollback_to {
                if let Some(txn_id) = current_txn {
                    let rest = txn_upper.strip_prefix("ROLLBACK TO").unwrap_or("").trim();
                    let rest = rest
                        .strip_prefix("SAVEPOINT")
                        .unwrap_or(rest)
                        .trim()
                        .trim_end_matches(';');
                    let name_start = txn_upper.find(rest).unwrap_or(0);
                    let name = sql.trim()[name_start..]
                        .trim()
                        .trim_end_matches(';')
                        .to_string();
                    let mgr = crate::txn::txn_manager();
                    mgr.with_context(txn_id, |ctx| ctx.rollback_to_savepoint(&name))
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                }
                return Ok(ExecOut::Empty);
            }

            if is_release {
                if let Some(txn_id) = current_txn {
                    let rest = txn_upper.strip_prefix("RELEASE").unwrap_or("").trim();
                    let rest = rest
                        .strip_prefix("SAVEPOINT")
                        .unwrap_or(rest)
                        .trim()
                        .trim_end_matches(';');
                    let name_start = txn_upper.find(rest).unwrap_or(0);
                    let name = sql.trim()[name_start..]
                        .trim()
                        .trim_end_matches(';')
                        .to_string();
                    let mgr = crate::txn::txn_manager();
                    mgr.with_context(txn_id, |ctx| ctx.release_savepoint(&name))
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                }
                return Ok(ExecOut::Empty);
            }

            if is_txn_dml || is_txn_select {
                let txn_id = current_txn.unwrap();
                let parsed =
                    SqlParser::parse(&sql).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let result = ApexExecutor::execute_in_txn(txn_id, parsed, &base_dir, &table_path)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                if let ApexResult::Scalar(n) = &result {
                    return Ok(ExecOut::Scalar("rows_buffered".to_string(), *n));
                }
                let batch = result
                    .to_record_batch()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                return Ok(ExecOut::Batch(batch));
            }

            // Normal execution (non-transaction writes, DDL, and fallback reads)
            let result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            if let ApexResult::Scalar(n) = &result {
                return Ok(ExecOut::Scalar("rows_affected".to_string(), *n));
            }
            let batch = result
                .to_record_batch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(ExecOut::Batch(batch))
        })?;
        crate::query::executor::clear_temp_dir();
        crate::query::executor::clear_query_root_dir();

        // Update transaction state after execution
        if is_begin {
            if let ExecOut::Scalar(_, txn_id) = &exec_out {
                *self.current_txn_id.write() = Some(*txn_id as u64);
            }
        }
        if is_commit || is_rollback {
            *self.current_txn_id.write() = None;
            if !table_name.is_empty() {
                self.invalidate_backend(&table_name);
            }
        }

        // Build Python result — columnar conversion for batches (single downcast per column)
        let out = PyDict::new_bound(py);
        match exec_out {
            ExecOut::Batch(batch) => {
                let columns_dict = PyDict::new_bound(py);
                let schema = batch.schema();
                for col_idx in 0..batch.num_columns() {
                    let col_name = schema.field(col_idx).name();
                    let arr = batch.column(col_idx);
                    let col_list = arrow_col_to_pylist(py, arr)?;
                    columns_dict.set_item(col_name, col_list)?;
                }
                out.set_item("columns_dict", columns_dict)?;
            }
            ExecOut::Scalar(key, val) => {
                out.set_item("columns", PyList::new_bound(py, [&key]))?;
                let row = PyList::new_bound(py, [val.into_py(py)]);
                out.set_item("rows", PyList::new_bound(py, [row]))?;
            }
            ExecOut::Empty => {
                out.set_item("columns", PyList::empty_bound(py))?;
                out.set_item("rows", PyList::empty_bound(py))?;
            }
        }
        out.set_item("rows_affected", 0)?;

        // Invalidate cached backend AFTER write operations
        if is_write && !table_name.is_empty() {
            self.invalidate_backend(&table_name);
        }

        // After CREATE TABLE, register the new table and set it as current
        if let QuerySignature::Ddl {
            kind: crate::query::query_signature::DdlKind::CreateTable { ref name },
        } = &sig
        {
            let tbl_path = self.current_base_dir().join(format!("{}.apex", name));
            self.table_paths.write().insert(name.clone(), tbl_path);
            *self.current_table.write() = name.clone();
        }

        Ok(out.into())
    }

    /// Execute multiple SQL queries in parallel using Rayon
    /// Returns a list of IPC byte arrays for each query
    fn execute_batch(&self, py: Python<'_>, queries: Vec<String>) -> PyResult<PyObject> {
        use arrow::ipc::writer::StreamWriter;

        let table_path = self
            .get_current_table_path()
            .unwrap_or_else(|_| self.current_base_dir());
        let base_dir = self.current_base_dir();
        let root_dir = self.root_dir.clone();
        let temp_dir = self.temp_dir.clone();

        // Execute queries in parallel using Rayon (releases GIL)
        let ipc_results: Vec<Result<Vec<u8>, String>> = py.allow_threads(|| {
            use rayon::prelude::*;

            queries
                .par_iter()
                .map(|sql| {
                    crate::query::executor::set_query_root_dir(&root_dir);
                    crate::query::executor::set_temp_dir(&temp_dir);

                    // Execute query in Rust thread pool
                    let result = ApexExecutor::execute_with_base_dir(sql, &base_dir, &table_path);
                    let batch = match result {
                        Ok(r) => r.to_record_batch(),
                        Err(e) => return Err(e.to_string()),
                    };
                    let batch = match batch {
                        Ok(b) => b,
                        Err(e) => return Err(e.to_string()),
                    };

                    crate::query::executor::clear_temp_dir();
                    crate::query::executor::clear_query_root_dir();

                    // Serialize to IPC format
                    let estimated_size = batch.get_array_memory_size() + 512;
                    let mut buf = Vec::with_capacity(estimated_size);
                    {
                        let mut writer = StreamWriter::try_new(&mut buf, batch.schema().as_ref())
                            .map_err(|e| format!("IPC writer error: {}", e))?;
                        writer
                            .write(&batch)
                            .map_err(|e| format!("IPC write error: {}", e))?;
                        writer
                            .finish()
                            .map_err(|e| format!("IPC finish error: {}", e))?;
                    }
                    Ok(buf)
                })
                .collect()
        });

        // Build Python list of results
        let empty_slice: &[PyObject] = &[];
        let list = PyList::new_bound(py, empty_slice);

        for result in ipc_results {
            match result {
                Ok(buf) => {
                    let py_bytes = pyo3::types::PyBytes::new_bound(py, &buf);
                    list.append(py_bytes)?;
                }
                Err(e) => return Err(PyRuntimeError::new_err(e)),
            }
        }
        Ok(list.into())
    }

    /// Execute SQL query and return Arrow FFI pointers for zero-copy transfer
    /// Returns (schema_ptr, array_ptr) that can be imported by PyArrow
    fn _execute_arrow_ffi(&self, py: Python<'_>, sql: &str) -> PyResult<(usize, usize)> {
        use crate::query::query_signature::{self, QuerySignature};
        use arrow::array::{Array, StructArray};
        use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};

        let sql = sql.to_string();
        let sig = query_signature::classify(&sql);
        let is_write = matches!(&sig, QuerySignature::DmlWrite | QuerySignature::Ddl { .. });
        let table_name = self.current_table.read().clone();
        let base_dir = self.current_base_dir();
        // Fall back to base_dir when no table selected (e.g. SELECT * FROM read_csv(...)).
        // Table-function queries don't use the default_table_path at all.
        let table_path = self
            .get_current_table_path()
            .unwrap_or_else(|_| base_dir.clone());
        crate::query::executor::set_query_root_dir(&self.root_dir);
        crate::query::executor::set_temp_dir(&self.temp_dir);

        // Execute query in Rust thread pool
        let batch = py.allow_threads(|| -> PyResult<RecordBatch> {
            let result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            result
                .to_record_batch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        crate::query::executor::clear_temp_dir();
        crate::query::executor::clear_query_root_dir();

        if is_write && !table_name.is_empty() {
            self.invalidate_backend(&table_name);
        }

        // Empty result
        if batch.num_rows() == 0 {
            return Ok((0, 0));
        }

        // Convert RecordBatch to StructArray for FFI export
        let struct_array: StructArray = batch.into();
        let array_data = struct_array.to_data();

        // Export to FFI
        let (ffi_array, ffi_schema) = arrow::ffi::to_ffi(&array_data)
            .map_err(|e| PyRuntimeError::new_err(format!("FFI export failed: {}", e)))?;

        // Leak the FFI structs to get stable pointers (caller must free via _free_arrow_ffi)
        let schema_ptr = Box::into_raw(Box::new(ffi_schema)) as usize;
        let array_ptr = Box::into_raw(Box::new(ffi_array)) as usize;

        Ok((schema_ptr, array_ptr))
    }

    /// Single-pass LIKE scan+extract via scan_like_and_extract_mmap, returned as zero-copy
    /// Arrow FFI pointers.  Returns (0, 0) on any error or when the fast path is unavailable
    /// (compressed/non-RCIX files), letting Python fall back to the IPC path.
    ///
    /// Uses `QuerySignature::classify()` — no inline pattern matching.
    fn _execute_like_ffi(&self, py: Python<'_>, sql: &str) -> PyResult<(usize, usize)> {
        use crate::query::query_signature::{self, QuerySignature};
        use arrow::array::{Array, StructArray};
        use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};

        let sig = query_signature::classify(sql);
        let (table, col, pattern) = match sig {
            QuerySignature::LikeFilter {
                table,
                column,
                pattern,
            } => (table, column, pattern),
            _ => return Ok((0, 0)),
        };

        let default_table_name = self.current_table.read().clone();
        let base_dir = self.current_base_dir();
        let default_table_path = if default_table_name.is_empty() {
            base_dir.clone()
        } else {
            self.table_paths
                .read()
                .get(&default_table_name)
                .cloned()
                .unwrap_or_else(|| base_dir.join(format!("{}.apex", default_table_name)))
        };
        let (_, table_path) = self.resolve_signature_table(
            table.as_deref(),
            &default_table_name,
            &default_table_path,
            &base_dir,
        );

        let batch = py.allow_threads(|| -> Option<arrow::record_batch::RecordBatch> {
            let backend = crate::query::get_cached_backend_pub(&table_path).ok()?;
            if backend.pending_v4_in_memory_rows() > 0 {
                return None;
            }
            backend
                .scan_like_and_extract_mmap(&col, &pattern, None)
                .ok()
                .flatten()
        });

        let batch = match batch {
            Some(b) if b.num_rows() > 0 => b,
            _ => return Ok((0, 0)),
        };

        // Export via Arrow C Data Interface (zero-copy)
        let struct_array: StructArray = batch.into();
        let array_data = struct_array.to_data();
        let (ffi_array, ffi_schema) = arrow::ffi::to_ffi(&array_data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("FFI export: {}", e)))?;
        let schema_ptr = Box::into_raw(Box::new(ffi_schema)) as usize;
        let array_ptr = Box::into_raw(Box::new(ffi_array)) as usize;
        Ok((schema_ptr, array_ptr))
    }

    /// Free Arrow FFI pointers allocated by _execute_arrow_ffi or _query_arrow_ffi
    fn _free_arrow_ffi(&self, schema_ptr: usize, array_ptr: usize) -> PyResult<()> {
        use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};

        if schema_ptr != 0 {
            unsafe {
                let _ = Box::from_raw(schema_ptr as *mut FFI_ArrowSchema);
            }
        }
        if array_ptr != 0 {
            unsafe {
                let _ = Box::from_raw(array_ptr as *mut FFI_ArrowArray);
            }
        }
        Ok(())
    }

    /// Execute SQL and return Arrow IPC bytes for efficient transfer.
    ///
    /// Uses `QuerySignature::classify()` — no inline pattern matching.
    /// Primarily used for multi-statement SQL and as a fallback for Arrow FFI failures.
    fn _execute_arrow_ipc(&self, py: Python<'_>, sql: &str) -> PyResult<PyObject> {
        use crate::query::query_signature::{self, QuerySignature};
        use arrow::ipc::writer::StreamWriter;
        use pyo3::types::PyBytes;

        let sig = query_signature::classify(sql);
        let is_write = matches!(&sig, QuerySignature::DmlWrite | QuerySignature::Ddl { .. });
        let is_multi = matches!(&sig, QuerySignature::MultiStatement);

        // Single read of current_table — avoids double RwLock acquire in get_current_table_path()
        let table_name = self.current_table.read().clone();
        let table_path = if table_name.is_empty() {
            self.current_base_dir()
        } else {
            self.table_paths
                .read()
                .get(&table_name)
                .cloned()
                .unwrap_or_else(|| self.current_base_dir().join(format!("{}.apex", table_name)))
        };
        let base_dir = self.current_base_dir();
        crate::query::executor::set_query_root_dir(&self.root_dir);
        crate::query::executor::set_temp_dir(&self.temp_dir);

        // FAST PATH: SELECT * LIMIT N — build Arrow batch directly from V4
        if let QuerySignature::SimpleScanLimit {
            limit,
            offset,
            ref table,
        } = &sig
        {
            let (_, target_path) =
                self.resolve_signature_table(table.as_deref(), &table_name, &table_path, &base_dir);
            if let Ok(backend) = crate::query::get_cached_backend_pub(&target_path) {
                if backend.pending_v4_in_memory_rows() == 0 {
                    let batch_result = if *offset > 0 {
                        if backend.has_pending_deltas()
                            || backend.has_delta()
                            || backend.active_row_count() != backend.row_count()
                        {
                            Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                "simple scan offset fast path unavailable",
                            ))
                        } else {
                            let end = (*offset)
                                .saturating_add(*limit)
                                .min(backend.row_count() as usize);
                            let indices: Vec<usize> = (*offset..end).collect();
                            backend.read_columns_by_indices_to_arrow(&indices, None)
                        }
                    } else {
                        backend
                            .storage
                            .to_arrow_batch_with_limit(None, false, *limit)
                    };
                    if let Ok(batch) = batch_result {
                        if batch.num_rows() > 0 || batch.num_columns() > 0 {
                            let mut buf = Vec::with_capacity(batch.get_array_memory_size() + 256);
                            {
                                let mut writer =
                                    StreamWriter::try_new(&mut buf, batch.schema().as_ref())
                                        .map_err(|e| {
                                            PyRuntimeError::new_err(format!(
                                                "IPC writer error: {}",
                                                e
                                            ))
                                        })?;
                                writer.write(&batch).map_err(|e| {
                                    PyRuntimeError::new_err(format!("IPC write error: {}", e))
                                })?;
                                writer.finish().map_err(|e| {
                                    PyRuntimeError::new_err(format!("IPC finish error: {}", e))
                                })?;
                            }
                            return Ok(PyBytes::new_bound(py, &buf).into());
                        }
                    }
                }
            }
        }

        let sql = sql.to_string();
        let current_txn = *self.current_txn_id.read();

        let (batch, new_txn_id) = if is_multi {
            py.allow_threads(|| -> PyResult<(RecordBatch, Option<u64>)> {
                let stmts = crate::query::sql_parser::SqlParser::parse_multi(&sql)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let (result, final_txn) = ApexExecutor::execute_multi_with_txn(
                    stmts,
                    &base_dir,
                    &table_path,
                    current_txn,
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let batch = result
                    .to_record_batch()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok((batch, final_txn))
            })?
        } else {
            let batch = py.allow_threads(|| -> PyResult<RecordBatch> {
                let result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                result
                    .to_record_batch()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
            (batch, current_txn)
        };

        if is_multi && new_txn_id != current_txn {
            *self.current_txn_id.write() = new_txn_id;
        }

        // Serialize to IPC format
        let estimated_size = batch.get_array_memory_size() + 512;
        let mut buf = Vec::with_capacity(estimated_size);
        {
            let mut writer = StreamWriter::try_new(&mut buf, batch.schema().as_ref())
                .map_err(|e| PyRuntimeError::new_err(format!("IPC writer error: {}", e)))?;
            writer
                .write(&batch)
                .map_err(|e| PyRuntimeError::new_err(format!("IPC write error: {}", e)))?;
            writer
                .finish()
                .map_err(|e| PyRuntimeError::new_err(format!("IPC finish error: {}", e)))?;
        }

        // Invalidate cached backend AFTER write operations
        if (is_write || is_multi) && !table_name.is_empty() {
            self.invalidate_backend(&table_name);
        }

        // After DROP TABLE, remove from table_paths (uses pre-extracted DdlKind — no re-uppercase)
        if let QuerySignature::Ddl {
            kind: crate::query::query_signature::DdlKind::DropTable { ref name },
        } = &sig
        {
            self.table_paths.write().remove(name);
            self.invalidate_backend(name);
            if *self.current_table.read() == *name {
                *self.current_table.write() = String::new();
            }
        }

        // After CREATE TABLE, register the new table (uses pre-extracted DdlKind)
        if let QuerySignature::Ddl {
            kind: crate::query::query_signature::DdlKind::CreateTable { ref name },
        } = &sig
        {
            let tbl_path = self.current_base_dir().join(format!("{}.apex", name));
            self.table_paths.write().insert(name.clone(), tbl_path);
            *self.current_table.write() = name.clone();
        }

        crate::query::executor::clear_temp_dir();
        crate::query::executor::clear_query_root_dir();
        Ok(PyBytes::new_bound(py, &buf).into())
    }

    /// Query with Arrow FFI (zero-copy transfer)
    fn _query_arrow_ffi(
        &self,
        py: Python<'_>,
        where_clause: &str,
        limit: Option<usize>,
    ) -> PyResult<(usize, usize)> {
        use arrow::array::{Array, StructArray};
        use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};

        // Single read of current_table — avoids double RwLock acquire
        let table_name = self.current_table.read().clone();
        if table_name.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No table selected. Call create_table() or use_table() first.",
            ));
        }
        let table_path = self
            .table_paths
            .read()
            .get(&table_name)
            .cloned()
            .unwrap_or_else(|| self.current_base_dir().join(format!("{}.apex", table_name)));
        let base_dir = self.current_base_dir();
        crate::query::executor::set_query_root_dir(&self.root_dir);
        let where_clause = where_clause.to_string();

        // Build SQL from where clause using current table name
        let sql = if let Some(lim) = limit {
            if where_clause == "1=1" || where_clause.is_empty() {
                format!("SELECT * FROM \"{}\" LIMIT {}", table_name, lim)
            } else {
                format!(
                    "SELECT * FROM \"{}\" WHERE {} LIMIT {}",
                    table_name, where_clause, lim
                )
            }
        } else {
            if where_clause == "1=1" || where_clause.is_empty() {
                format!("SELECT * FROM \"{}\"", table_name)
            } else {
                format!("SELECT * FROM \"{}\" WHERE {}", table_name, where_clause)
            }
        };

        // Execute query
        let batch = py.allow_threads(|| -> PyResult<RecordBatch> {
            let result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            result
                .to_record_batch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;

        // Empty result
        if batch.num_rows() == 0 {
            return Ok((0, 0));
        }

        // Convert to StructArray for FFI
        let struct_array: StructArray = batch.into();
        let array_data = struct_array.to_data();

        let (ffi_array, ffi_schema) = arrow::ffi::to_ffi(&array_data)
            .map_err(|e| PyRuntimeError::new_err(format!("FFI export failed: {}", e)))?;

        let schema_ptr = Box::into_raw(Box::new(ffi_schema)) as usize;
        let array_ptr = Box::into_raw(Box::new(ffi_array)) as usize;

        Ok((schema_ptr, array_ptr))
    }

    // ========== Table Management ==========

    /// Use a table
    fn use_table(&self, name: &str) -> PyResult<()> {
        // First check cache
        {
            let paths = self.table_paths.read();
            if paths.contains_key(name) {
                drop(paths);
                *self.current_table.write() = name.to_string();
                return Ok(());
            }
        }

        // Table not in cache - check if it exists on disk (lazy discovery)
        let table_path = self.current_base_dir().join(format!("{}.apex", name));
        if table_path.exists() {
            // Add to cache
            self.table_paths
                .write()
                .insert(name.to_string(), table_path);
            *self.current_table.write() = name.to_string();
            return Ok(());
        }

        Err(PyValueError::new_err(format!("Table not found: {}", name)))
    }

    /// Get current table name
    fn current_table(&self) -> String {
        self.current_table.read().clone()
    }

    /// Create a new table, optionally with a pre-defined schema dict {col_name: type_str}.
    /// Pre-defining schema avoids type inference on the first insert.
    #[pyo3(signature = (name, schema=None))]
    fn create_table(&self, name: &str, schema: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let mut paths = self.table_paths.write();
        if paths.contains_key(name) {
            // Verify the file actually exists on disk (table_paths may be stale after SQL DROP TABLE)
            let existing_path = self.current_base_dir().join(format!("{}.apex", name));
            if existing_path.exists() {
                return Err(PyValueError::new_err(format!(
                    "Table already exists: {}",
                    name
                )));
            }
            // Stale entry — remove it and proceed with creation
            paths.remove(name);
        }

        let table_path = self.current_base_dir().join(format!("{}.apex", name));
        let engine = crate::storage::engine::engine();

        if let Some(schema_dict) = schema {
            let schema_cols = Self::parse_schema_dict(schema_dict)?;
            engine
                .create_table_with_schema(&table_path, self.durability, &schema_cols)
                .map_err(|e| PyIOError::new_err(format!("Failed to create table: {}", e)))?;
        } else {
            engine
                .create_table(&table_path, self.durability)
                .map_err(|e| PyIOError::new_err(format!("Failed to create table: {}", e)))?;
        }

        paths.insert(name.to_string(), table_path);
        drop(paths);

        *self.current_table.write() = name.to_string();
        Ok(())
    }

    /// Drop a table
    fn drop_table(&self, name: &str) -> PyResult<()> {
        // Invalidate cached backend first (releases file lock)
        self.invalidate_backend(name);

        let mut paths = self.table_paths.write();
        if let Some(path) = paths.remove(name) {
            fs::remove_file(&path)
                .map_err(|e| PyIOError::new_err(format!("Failed to delete table file: {}", e)))?;
        } else {
            return Err(PyValueError::new_err(format!("Table not found: {}", name)));
        }
        drop(paths);

        if *self.current_table.read() == name {
            *self.current_table.write() = String::new();
        }
        Ok(())
    }

    /// Register a data file (CSV, JSON, Parquet) as a temporary table.
    ///
    /// The file is parsed once and materialized into a native .apex table stored
    /// in a temp directory. Subsequent queries benefit from mmap zero-copy access,
    /// zone maps, and bloom filters — an order of magnitude faster than repeated
    /// read_csv/read_json/read_parquet calls. The temp table is cleaned up when
    /// the ApexStorage is dropped or the client is closed.
    fn register_temp_table(&self, name: &str, file_path: &str) -> PyResult<()> {
        use crate::query::executor::ApexExecutor;

        let temp_path = self.temp_dir.join(format!("{}.apex", name));
        let _ = fs::create_dir_all(&self.temp_dir);

        if temp_path.exists() {
            return Err(PyValueError::new_err(format!(
                "Temp table '{}' already exists. Use drop_temp_table() first.",
                name
            )));
        }

        let fmt = {
            let lower = file_path.to_lowercase();
            if lower.ends_with(".csv") || lower.ends_with(".tsv") {
                "CSV"
            } else if lower.ends_with(".json")
                || lower.ends_with(".ndjson")
                || lower.ends_with(".jsonl")
            {
                "JSON"
            } else {
                "PARQUET"
            }
        };

        crate::query::executor::set_temp_dir(&self.temp_dir);
        let base_dir = self.current_base_dir();
        let result = ApexExecutor::execute_copy_import(
            &temp_path,
            name,
            file_path,
            fmt,
            &[],
            &base_dir,
            &base_dir,
        );
        crate::query::executor::clear_temp_dir();

        match result {
            Ok(_) => {
                self.table_paths.write().insert(name.to_string(), temp_path);
                Ok(())
            }
            Err(e) => {
                let _ = fs::remove_file(&temp_path);
                Err(PyIOError::new_err(format!(
                    "Failed to register temp table: {}",
                    e
                )))
            }
        }
    }

    /// Drop a previously registered temporary table.
    fn drop_temp_table(&self, name: &str) -> PyResult<()> {
        if let Some(path) = self.table_paths.write().remove(name) {
            let _ = fs::remove_file(&path);
            let _ = fs::remove_file(path.with_extension("apex.wal"));
            crate::storage::engine::engine().invalidate(&path);
        }
        Ok(())
    }

    /// List all tables
    fn list_tables(&self) -> Vec<String> {
        // Scan directory for .apex files to ensure we catch tables created via SQL
        let mut tables = Vec::new();
        let base_dir = self.current_base_dir();
        if let Ok(entries) = fs::read_dir(&base_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension()
                    .and_then(|e| e.to_str())
                    .map(|s| s == "apex")
                    .unwrap_or(false)
                {
                    if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                        tables.push(stem.to_string());
                    }
                }
            }
        }
        tables.sort();
        tables.dedup();
        tables
    }

    // ========== Multi-Database Operations ==========

    /// Switch to a named database (creates its subdirectory if needed).
    /// "default" or "" means the root directory (backward-compatible default).
    #[pyo3(name = "use_database_")]
    fn use_database_(&self, db_name: &str) -> PyResult<()> {
        let new_base_dir = if db_name.is_empty() || db_name.eq_ignore_ascii_case("default") {
            self.root_dir.clone()
        } else {
            let db_dir = self.root_dir.join(db_name);
            fs::create_dir_all(&db_dir).map_err(|e| {
                PyIOError::new_err(format!("Cannot create database '{}': {}", db_name, e))
            })?;
            db_dir
        };

        *self.current_database.write() = db_name.to_string();
        *self.base_dir.write() = new_base_dir;

        // Clear all per-database caches
        self.cached_backends.clear();
        self.update_by_id_numeric_cache.clear();
        self.update_by_id_cell_cache.clear();
        self.replace_exact_row_cache.clear();
        self.table_paths.write().clear();
        *self.tables_scanned.write() = false;
        *self.current_table.write() = String::new();

        Ok(())
    }

    /// Return the current database name ("" / "default" means root/default).
    #[pyo3(name = "current_database_")]
    fn current_database_(&self) -> String {
        self.current_database.read().clone()
    }

    /// List all available databases (named subdirectories of root_dir).
    /// "default" is always included to represent the root-level tables.
    #[pyo3(name = "list_databases_")]
    fn list_databases_(&self) -> Vec<String> {
        let mut dbs = vec!["default".to_string()];
        if let Ok(entries) = fs::read_dir(&self.root_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                        // Skip hidden dirs and internal dirs
                        if !name.starts_with('.') && name != "fts_indexes" {
                            dbs.push(name.to_string());
                        }
                    }
                }
            }
        }
        dbs.sort();
        dbs.dedup();
        dbs
    }

    /// Get row count for current table (excluding deleted rows) using StorageEngine.
    /// LOCK-FREE: active_count is an AtomicU64 — no file lock needed for this metadata read.
    fn row_count(&self) -> PyResult<u64> {
        let table_path = self.get_current_table_path()?;
        // If file doesn't exist (e.g., after drop_if_exists), return 0
        if !table_path.exists() {
            return Ok(0);
        }

        let table_name = self.current_table.read().clone();
        let cache_key = Self::backend_cache_key(&table_path, &table_name);
        if let Some(backend) = self.cached_backends.get(&cache_key) {
            return Ok(backend.active_row_count());
        }

        // No file lock needed — active_count is atomic and always consistent
        let engine = crate::storage::engine::engine();
        let count = engine
            .active_row_count(&table_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        Ok(count)
    }

    /// Alias for row_count (compatibility)
    fn count_rows(&self) -> PyResult<u64> {
        self.row_count()
    }

    /// Whether the current table has V4 rows buffered in memory but not yet
    /// saved as an on-disk row group.
    fn has_pending_memtable_rows(&self) -> PyResult<bool> {
        let table_name = self.current_table.read().clone();
        if table_name.is_empty() {
            return Ok(false);
        }
        let table_path = self.get_current_table_path()?;
        let mut backends: Vec<Arc<TableStorageBackend>> = Vec::new();
        for cache_key in [
            Self::backend_cache_key(&table_path, &table_name),
            Self::insert_backend_cache_key(&table_path, &table_name),
        ] {
            if let Some(entry) = self.cached_backends.get(&cache_key) {
                let backend = Arc::clone(entry.value());
                if !backends.iter().any(|cached| Arc::ptr_eq(cached, &backend)) {
                    backends.push(backend);
                }
            }
        }
        Ok(backends
            .iter()
            .any(|backend| backend.pending_v4_in_memory_rows() > 0))
    }

    /// Whether the current table has any same-client overlay state that should
    /// be flushed before handing control to a write SQL executor that reopens
    /// storage from disk.
    fn has_pending_overlay_writes(&self) -> PyResult<bool> {
        let table_name = self.current_table.read().clone();
        if table_name.is_empty() {
            return Ok(false);
        }

        let table_path = self.get_current_table_path()?;
        let mut backends: Vec<Arc<TableStorageBackend>> = Vec::new();
        for cache_key in [
            Self::backend_cache_key(&table_path, &table_name),
            Self::insert_backend_cache_key(&table_path, &table_name),
        ] {
            if let Some(entry) = self.cached_backends.get(&cache_key) {
                let backend = Arc::clone(entry.value());
                if !backends.iter().any(|cached| Arc::ptr_eq(cached, &backend)) {
                    backends.push(backend);
                }
            }
        }

        Ok(backends.iter().any(|backend| {
            backend.is_dirty()
                || backend.has_pending_deltas()
                || backend.pending_v4_in_memory_rows() > 0
        }))
    }

    /// Resolve a table name to its path (handles temp tables, current dir)
    fn resolve_table_path_for_count(&self, table_name: &str) -> PyResult<PathBuf> {
        let clean = table_name.trim_matches('"').trim_matches('`');
        // Check per-instance path cache first
        {
            let paths = self.table_paths.read();
            if let Some(p) = paths.get(clean) {
                return Ok(p.clone());
            }
        }
        let base_dir = self.current_base_dir();
        // Check temp dir first (temp tables shadow persistent)
        if let Some(temp_dir) = crate::query::executor::get_temp_dir() {
            let temp_path = temp_dir.join(format!("{}.apex", clean));
            if temp_path.exists() {
                let mut paths = self.table_paths.write();
                paths.insert(clean.to_string(), temp_path.clone());
                return Ok(temp_path);
            }
        }
        // Try base_dir
        let p = base_dir.join(format!("{}.apex", clean));
        if p.exists() {
            let mut paths = self.table_paths.write();
            paths.insert(clean.to_string(), p.clone());
            return Ok(p);
        }
        Err(PyValueError::new_err(format!("Table not found: {}", clean)))
    }

    /// Ultra-fast row count for ANY table (resolves path, uses per-instance cache)
    fn fast_row_count_for(&self, table_name: &str) -> PyResult<u64> {
        let table_path = self.resolve_table_path_for_count(table_name)?;
        let cache_key = Self::backend_cache_key(&table_path, table_name);

        // Per-instance backend cache (no stat(), no delta check)
        if let Some(backend) = self.cached_backends.get(&cache_key) {
            let backend = Arc::clone(&backend);
            return Ok(backend.active_row_count());
        }

        // Global cache fallback
        if let Ok(backend) = crate::query::get_cached_backend_pub(&table_path) {
            self.cached_backends
                .insert(cache_key.clone(), Arc::clone(&backend));
            return Ok(backend.active_row_count());
        }

        // Engine fallback
        let engine = crate::storage::engine::engine();
        let count = engine
            .active_row_count(&table_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(count)
    }

    /// Ultra-fast row count for the CURRENT table
    fn fast_row_count(&self) -> PyResult<u64> {
        let table_name = self.current_table.read().clone();
        if table_name.is_empty() {
            return Err(PyValueError::new_err(
                "No table selected. Call create_table() or use_table() first.",
            ));
        }
        self.fast_row_count_for(&table_name)
    }

    /// Save current table
    fn save(&self) -> PyResult<()> {
        // Storage auto-saves on each operation
        Ok(())
    }

    /// Flush changes to disk with fsync
    ///
    /// For 'safe' and 'max' durability levels, save() automatically calls fsync.
    /// For 'fast' durability, call this method explicitly when you need durability guarantees.
    fn flush(&self) -> PyResult<()> {
        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        let mut backends: Vec<Arc<TableStorageBackend>> = Vec::new();
        for cache_key in [
            Self::backend_cache_key(&table_path, &table_name),
            Self::insert_backend_cache_key(&table_path, &table_name),
        ] {
            if let Some(entry) = self.cached_backends.get(&cache_key) {
                let backend = Arc::clone(entry.value());
                if !backends.iter().any(|cached| Arc::ptr_eq(cached, &backend)) {
                    backends.push(backend);
                }
            }
        }

        if backends.is_empty() {
            return Ok(());
        }

        let mut actions: Vec<(Arc<TableStorageBackend>, bool)> = Vec::new();
        for backend in backends {
            let needs_save = backend.is_dirty()
                || backend.has_pending_deltas()
                || backend.pending_v4_in_memory_rows() > 0;
            let needs_sync = backend.storage.sync_pending();
            if needs_save || needs_sync {
                actions.push((backend, needs_save));
            }
        }

        if actions.is_empty() {
            return Ok(());
        }

        let any_needs_save = actions.iter().any(|(_, needs_save)| *needs_save);
        let lock_file = if any_needs_save {
            Some(
                Self::acquire_read_lock(&table_path)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?,
            )
        } else {
            None
        };

        let result: PyResult<()> = (|| {
            for (backend, needs_save) in actions {
                if needs_save {
                    backend
                        .save()
                        .and_then(|_| backend.sync())
                        .map_err(|e| PyIOError::new_err(format!("Failed to flush: {}", e)))?;
                } else {
                    backend
                        .sync()
                        .map_err(|e| PyIOError::new_err(format!("Failed to flush: {}", e)))?;
                }
            }
            Ok(())
        })();

        if let Some(lock_file) = lock_file {
            Self::release_lock(lock_file);
        }
        result?;

        if any_needs_save {
            crate::storage::engine::engine().invalidate(&table_path);
            crate::query::executor::invalidate_storage_cache(&table_path);
            crate::query::planner::invalidate_table_stats(&table_path.to_string_lossy());
            self.invalidate_backend(&table_name);
        }
        Ok(())
    }

    /// Get the current durability level
    fn get_durability(&self) -> String {
        self.durability.as_str().to_string()
    }

    /// Set auto-flush thresholds
    ///
    /// When either threshold is exceeded during writes, data is automatically
    /// written to file. Set to 0 to disable the respective threshold.
    ///
    /// Parameters:
    /// - rows: Auto-flush when pending rows exceed this count (0 = disabled)
    /// - bytes: Auto-flush when estimated memory exceeds this size (0 = disabled)
    #[pyo3(signature = (rows = 0, bytes = 0))]
    fn set_auto_flush(&self, rows: u64, bytes: u64) -> PyResult<()> {
        // Persist at struct level so thresholds survive backend cache invalidation
        *self.auto_flush_rows.write() = rows;
        *self.auto_flush_bytes.write() = bytes;
        // Also apply to cached backend if present
        let table_name = self.current_table.read().clone();
        let table_path = self.get_current_table_path()?;
        let cache_key = Self::backend_cache_key(&table_path, &table_name);
        if let Some(backend) = self.cached_backends.get(&cache_key) {
            backend.set_auto_flush(rows, bytes);
        }
        Ok(())
    }

    /// Get current auto-flush configuration
    ///
    /// Returns a tuple of (rows_threshold, bytes_threshold)
    fn get_auto_flush(&self) -> PyResult<(u64, u64)> {
        Ok((*self.auto_flush_rows.read(), *self.auto_flush_bytes.read()))
    }

    /// Get estimated memory usage in bytes
    fn estimate_memory_bytes(&self) -> PyResult<u64> {
        let table_name = self.current_table.read().clone();
        let table_path = self.get_current_table_path()?;
        let cache_key = Self::backend_cache_key(&table_path, &table_name);
        if let Some(backend) = self.cached_backends.get(&cache_key) {
            let mem = backend.estimate_memory_bytes();
            if mem > 0 {
                return Ok(mem);
            }
        }
        // No in-memory data (flushed to disk): estimate from file size
        if let Ok(meta) = std::fs::metadata(&table_path) {
            return Ok(meta.len());
        }
        Ok(0)
    }

    /// Set compression type for the current table.
    /// Only effective on empty tables (row_count == 0); ignored if table has data.
    /// The setting persists across restarts.
    ///
    /// Args:
    ///     compression: "none", "lz4", or "zstd"
    ///
    /// Returns:
    ///     True if applied, False if table is non-empty (no-op)
    fn set_compression(&self, compression: &str) -> PyResult<bool> {
        use crate::storage::on_demand::{CompressionType, OnDemandStorage};
        let comp = CompressionType::from_str_opt(compression).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Invalid compression type '{}'. Use 'none', 'lz4', or 'zstd'.",
                compression
            ))
        })?;
        let table_path = self.get_current_table_path()?;
        let storage = if table_path.exists() {
            OnDemandStorage::open_with_durability(&table_path, self.durability)
        } else {
            OnDemandStorage::create_with_durability(&table_path, self.durability)
        }
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        storage
            .set_compression(comp)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get the current compression type for the current table.
    ///
    /// Returns:
    ///     "none", "lz4", or "zstd"
    fn get_compression(&self) -> PyResult<String> {
        use crate::storage::on_demand::OnDemandStorage;
        let table_path = self.get_current_table_path()?;
        if table_path.exists() {
            let storage = OnDemandStorage::open_with_durability(&table_path, self.durability)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(storage.compression().as_str().to_string())
        } else {
            Ok("none".to_string())
        }
    }

    /// Close storage
    fn close(&self) -> PyResult<()> {
        crate::query::executor::wait_fts_backfills_for_dir(&self.current_base_dir());

        for entry in self.cached_backends.iter() {
            let backend = entry.value();
            if backend.has_pending_deltas() || backend.pending_v4_in_memory_rows() > 0 {
                let _ = backend.save();
            }
        }

        // Clear per-instance cached backends (releases per-instance references)
        self.cached_backends.clear();
        self.update_by_id_numeric_cache.clear();
        self.update_by_id_cell_cache.clear();
        self.replace_exact_row_cache.clear();

        // Clean up temp tables
        let _ = fs::remove_dir_all(&self.temp_dir);

        // On Windows: release all mmaps so temp directories can be cleaned up.
        // On Unix: mmaps remain valid after atomic rename; keep STORAGE_CACHE alive
        // so the 50ms fast path in get_cached_backend skips stat() calls on next retrieve().
        #[cfg(target_os = "windows")]
        ApexExecutor::invalidate_cache_for_dir(&self.current_base_dir());
        Ok(())
    }

    // ========== Retrieve Operations ==========

    /// Retrieve a single record by ID
    fn retrieve(&self, py: Python<'_>, id: i64) -> PyResult<Option<PyObject>> {
        let table_path = self.get_current_table_path()?;

        if id < 0 {
            return Ok(None);
        }

        // ULTRA-FAST PATH: Direct V4 value read - no file lock, no Arrow, no GIL release
        // Skip allow_threads() for sub-0.1ms operations where GIL overhead dominates
        // Use per-instance cached_backends first: no stat() syscalls (~600µs saved vs get_cached_backend_pub).
        let table_name = self.current_table.read().clone();
        let cache_key = Self::backend_cache_key(&table_path, &table_name);
        let maybe_cached = { self.cached_backends.get(&cache_key).map(|v| Arc::clone(&v)) };
        let backend_opt: Option<Arc<TableStorageBackend>> = if let Some(b) = maybe_cached {
            Some(b)
        } else if let Ok(b) = crate::query::get_cached_backend_pub(&table_path) {
            // Populate per-instance cache so next call is zero-syscall
            self.cached_backends.insert(cache_key, Arc::clone(&b));
            Some(b)
        } else {
            None
        };
        if let Some(backend) = backend_opt {
            let id_u64 = id as u64;
            if id_u64 >= backend.next_id_value() {
                return Ok(None);
            }
            if backend.has_pending_deltas() {
                // Pending UPDATE overlays are applied by the Arrow executor fallback below.
            } else {
                if backend.pending_v4_in_memory_rows() > 0 {
                    let vals_result =
                        py.allow_threads(|| backend.storage.read_row_by_id_values(id_u64));
                    return match vals_result {
                        Ok(Some(vals)) => {
                            let dict = PyDict::new_bound(py);
                            for (k, v) in vals {
                                dict.set_item(k, value_to_py(py, &v)?)?;
                            }
                            Ok(Some(dict.into()))
                        }
                        Ok(None) => Ok(None),
                        Err(_) => Ok(None),
                    };
                }
                // Release GIL for all Rust computation; re-acquire only for PyDict construction.
                // retrieve_rcix: page-cached RCIX read, handles PLAIN/BITPACK/RLE/StringDict.
                let rcix_result = py.allow_threads(|| backend.storage.retrieve_rcix(id_u64));
                if let Ok(Some(vals)) = rcix_result {
                    let dict = PyDict::new_bound(py);
                    for (k, v) in vals {
                        dict.set_item(k, value_to_py(py, &v)?)?;
                    }
                    return Ok(Some(dict.into()));
                }
                // Fallback: may need to (re)create mmap after save_v4 invalidation
                let vals_result =
                    py.allow_threads(|| backend.storage.read_row_by_id_values(id_u64));
                if let Ok(Some(vals)) = vals_result {
                    let dict = PyDict::new_bound(py);
                    for (k, v) in vals {
                        dict.set_item(k, value_to_py(py, &v)?)?;
                    }
                    return Ok(Some(dict.into()));
                }
                // Arrow batch cache path: O(1) index lookup + batch.slice(idx, 1)
                let batch_result = py.allow_threads(|| backend.read_row_by_id_to_arrow(id_u64));
                if let Ok(Some(batch)) = batch_result {
                    if batch.num_rows() > 0 {
                        let dict = PyDict::new_bound(py);
                        let schema = batch.schema();
                        for col_idx in 0..batch.num_columns() {
                            let col_name = schema.field(col_idx).name();
                            let val = arrow_value_at(batch.column(col_idx), 0);
                            dict.set_item(col_name.as_str(), value_to_py(py, &val)?)?;
                        }
                        return Ok(Some(dict.into()));
                    }
                }
            }
        }

        // FALLBACK: File lock + Arrow path for edge cases
        let lock_file =
            Self::acquire_read_lock(&table_path).map_err(|e| PyIOError::new_err(e.to_string()))?;

        let base_dir = self.current_base_dir();
        crate::query::executor::set_query_root_dir(&self.root_dir);
        let table_name = self.current_table.read().clone();

        let result = py.allow_threads(|| -> PyResult<Option<HashMap<String, Value>>> {
            let sql = format!("SELECT * FROM \"{}\" WHERE _id = {}", table_name, id);
            let result = ApexExecutor::execute_with_base_dir(&sql, &base_dir, &table_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let batch = result
                .to_record_batch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            if batch.num_rows() == 0 {
                return Ok(None);
            }

            let mut row_data = HashMap::new();
            for (col_idx, field) in batch.schema().fields().iter().enumerate() {
                let val = arrow_value_at(batch.column(col_idx), 0);
                row_data.insert(field.name().clone(), val);
            }
            Ok(Some(row_data))
        });

        Self::release_lock(lock_file);

        let result = result?;

        match result {
            None => Ok(None),
            Some(row_data) => {
                let dict = PyDict::new_bound(py);
                for (k, v) in row_data {
                    dict.set_item(k, value_to_py(py, &v)?)?;
                }
                Ok(Some(dict.into()))
            }
        }
    }

    /// Retrieve selected columns for one ID without SQL parsing/classification.
    fn retrieve_projected(
        &self,
        py: Python<'_>,
        id: i64,
        columns: Vec<String>,
    ) -> PyResult<Option<PyObject>> {
        if id < 0 || columns.is_empty() {
            return Ok(None);
        }
        let table_name = self.current_table.read().clone();
        if table_name.is_empty() {
            return Err(PyValueError::new_err(
                "No table selected. Call create_table() or use_table() first.",
            ));
        }
        let backend_opt: Option<Arc<TableStorageBackend>> = self
            .cached_backends
            .get(&table_name)
            .map(|v| Arc::clone(&v))
            .or_else(|| {
                let table_path = self.get_current_table_path().ok()?;
                let cache_key = Self::backend_cache_key(&table_path, &table_name);
                self.cached_backends
                    .get(&cache_key)
                    .map(|v| {
                        let backend = Arc::clone(&v);
                        self.cached_backends
                            .insert(table_name.clone(), Arc::clone(&backend));
                        backend
                    })
                    .or_else(|| {
                        crate::query::get_cached_backend_pub(&table_path)
                            .ok()
                            .map(|b| {
                                self.cached_backends
                                    .insert(cache_key.clone(), Arc::clone(&b));
                                self.cached_backends
                                    .insert(table_name.clone(), Arc::clone(&b));
                                b
                            })
                    })
            });

        let Some(backend) = backend_opt else {
            return Ok(None);
        };
        if backend.has_pending_deltas() || backend.pending_v4_in_memory_rows() > 0 {
            return Ok(None);
        }

        let requested_cols: Vec<&str> = columns.iter().map(String::as_str).collect();
        let vals_result = backend
            .storage
            .retrieve_rcix_projected(id as u64, &requested_cols)
            .ok()
            .flatten()
            .or_else(|| {
                backend
                    .storage
                    .read_row_by_id_values(id as u64)
                    .ok()
                    .flatten()
            });
        let Some(vals) = vals_result else {
            return Ok(None);
        };

        let out = PyDict::new_bound(py);
        let Some(columns_dict) = projected_values_to_columns_dict(py, &vals, &columns)? else {
            return Ok(None);
        };
        out.set_item("columns_dict", columns_dict)?;
        out.set_item("rows_affected", 0i64)?;
        Ok(Some(out.into()))
    }

    /// Retrieve the first row matching `column = value` using the lazy equality cache.
    fn retrieve_first_by_string_eq_limit1(
        &self,
        py: Python<'_>,
        column: String,
        value: String,
    ) -> PyResult<Option<PyObject>> {
        if column.is_empty() {
            return Ok(None);
        }
        let (table_path, table_name) = self.get_current_table_info()?;
        let cache_key = Self::backend_cache_key(&table_path, &table_name);
        let backend_opt: Option<Arc<TableStorageBackend>> = self
            .cached_backends
            .get(&cache_key)
            .map(|v| Arc::clone(&v))
            .or_else(|| {
                crate::query::get_cached_backend_pub(&table_path)
                    .ok()
                    .map(|b| {
                        self.cached_backends
                            .insert(cache_key.clone(), Arc::clone(&b));
                        b
                    })
            });

        let Some(backend) = backend_opt else {
            return Ok(None);
        };
        if backend.has_pending_deltas() || backend.pending_v4_in_memory_rows() > 0 {
            return Ok(None);
        }

        let Some(row_id) = backend
            .first_row_id_for_string_eq(&column, &value)
            .ok()
            .flatten()
        else {
            return Ok(None);
        };

        let vals_result = backend
            .storage
            .retrieve_rcix(row_id)
            .ok()
            .flatten()
            .or_else(|| backend.storage.read_row_by_id_values(row_id).ok().flatten());
        let Some(vals) = vals_result else {
            return Ok(None);
        };

        let out = PyDict::new_bound(py);
        let columns_dict = values_to_columns_dict(py, &vals)?;
        out.set_item("columns_dict", columns_dict)?;
        out.set_item("rows_affected", 0i64)?;
        Ok(Some(out.into()))
    }

    /// Retrieve selected columns for the first row matching `column = value`.
    fn retrieve_projected_first_by_string_eq_limit1(
        &self,
        py: Python<'_>,
        filter_column: String,
        value: String,
        columns: Vec<String>,
    ) -> PyResult<Option<PyObject>> {
        if filter_column.is_empty() || columns.is_empty() {
            return Ok(None);
        }
        let (table_path, table_name) = self.get_current_table_info()?;
        let cache_key = Self::backend_cache_key(&table_path, &table_name);
        let backend_opt: Option<Arc<TableStorageBackend>> = self
            .cached_backends
            .get(&cache_key)
            .map(|v| Arc::clone(&v))
            .or_else(|| {
                crate::query::get_cached_backend_pub(&table_path)
                    .ok()
                    .map(|b| {
                        self.cached_backends
                            .insert(cache_key.clone(), Arc::clone(&b));
                        b
                    })
            });

        let Some(backend) = backend_opt else {
            return Ok(None);
        };
        if backend.has_pending_deltas() || backend.pending_v4_in_memory_rows() > 0 {
            return Ok(None);
        }

        let Some(row_id) = backend
            .first_row_id_for_string_eq(&filter_column, &value)
            .ok()
            .flatten()
        else {
            return Ok(None);
        };

        let requested_cols: Vec<&str> = columns.iter().map(String::as_str).collect();
        let vals_result = backend
            .storage
            .retrieve_rcix_projected(row_id, &requested_cols)
            .ok()
            .flatten()
            .or_else(|| backend.storage.read_row_by_id_values(row_id).ok().flatten());
        let Some(vals) = vals_result else {
            return Ok(None);
        };

        let Some(columns_dict) = projected_values_to_columns_dict(py, &vals, &columns)? else {
            return Ok(None);
        };

        let out = PyDict::new_bound(py);
        out.set_item("columns_dict", columns_dict)?;
        out.set_item("rows_affected", 0i64)?;
        Ok(Some(out.into()))
    }

    /// Retrieve selected columns for the first N rows matching `column = value`.
    /// This is the Python hot path for small projected equality+LIMIT SQL, avoiding
    /// full SQL execution and Arrow round-trips through the client layer.
    fn retrieve_projected_by_string_eq_limit(
        &self,
        py: Python<'_>,
        filter_column: String,
        value: String,
        columns: Vec<String>,
        limit: usize,
        offset: usize,
    ) -> PyResult<Option<PyObject>> {
        if filter_column.is_empty() || columns.is_empty() {
            return Ok(None);
        }

        let (table_path, table_name) = self.get_current_table_info()?;
        let cache_key = Self::backend_cache_key(&table_path, &table_name);
        let backend_opt: Option<Arc<TableStorageBackend>> = self
            .cached_backends
            .get(&cache_key)
            .map(|v| Arc::clone(&v))
            .or_else(|| {
                crate::query::get_cached_backend_pub(&table_path)
                    .ok()
                    .map(|b| {
                        self.cached_backends
                            .insert(cache_key.clone(), Arc::clone(&b));
                        b
                    })
            });

        let Some(backend) = backend_opt else {
            return Ok(None);
        };
        if backend.has_pending_deltas() || backend.pending_v4_in_memory_rows() > 0 {
            return Ok(None);
        }

        let col_refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        let needed = offset.saturating_add(limit);
        if limit == 0 {
            let out = PyDict::new_bound(py);
            let columns_dict = PyDict::new_bound(py);
            for col_name in &columns {
                columns_dict.set_item(col_name.as_str(), PyList::empty_bound(py))?;
            }
            out.set_item("columns_dict", columns_dict)?;
            out.set_item("rows_affected", 0i64)?;
            return Ok(Some(out.into()));
        }

        let indices_result = py.allow_threads(|| -> io::Result<Option<Vec<usize>>> {
            let Some(indices) =
                backend.scan_string_filter_mmap(&filter_column, &value, Some(needed))?
            else {
                return Ok(None);
            };
            Ok(Some(indices.into_iter().skip(offset).take(limit).collect()))
        });

        let Some(final_indices) = indices_result.map_err(|e| PyIOError::new_err(e.to_string()))?
        else {
            return Ok(None);
        };

        if final_indices.is_empty() {
            let out = PyDict::new_bound(py);
            let columns_dict = PyDict::new_bound(py);
            for col_name in &columns {
                columns_dict.set_item(col_name.as_str(), PyList::empty_bound(py))?;
            }
            out.set_item("columns_dict", columns_dict)?;
            out.set_item("rows_affected", 0i64)?;
            return Ok(Some(out.into()));
        }

        let cols_result = py.allow_threads(|| {
            backend
                .storage
                .extract_rows_by_indices_mmap_columns(&final_indices, Some(col_refs.as_slice()))
        });
        if let Some(batch_cols) = cols_result.map_err(|e| PyIOError::new_err(e.to_string()))? {
            if let Some(columns_dict) =
                mmap_batch_columns_to_pydict(py, batch_cols, Some(&columns))?
            {
                let out = PyDict::new_bound(py);
                out.set_item("columns_dict", columns_dict)?;
                out.set_item("rows_affected", 0i64)?;
                return Ok(Some(out.into()));
            }
        }

        let batch_result = py.allow_threads(|| -> io::Result<Option<RecordBatch>> {
            if final_indices.is_empty() {
                backend
                    .read_columns_to_arrow(Some(col_refs.as_slice()), 0, Some(0))
                    .map(Some)
            } else {
                backend
                    .read_columns_by_indices_to_arrow(&final_indices, Some(col_refs.as_slice()))
                    .map(Some)
            }
        });

        let Some(batch) = batch_result.map_err(|e| PyIOError::new_err(e.to_string()))? else {
            return Ok(None);
        };

        let out = PyDict::new_bound(py);
        let columns_dict = PyDict::new_bound(py);
        if batch.num_rows() == 0 {
            for col_name in &columns {
                columns_dict.set_item(col_name.as_str(), PyList::empty_bound(py))?;
            }
        } else {
            let schema = batch.schema();
            for col_idx in 0..batch.num_columns() {
                let col_name = schema.field(col_idx).name();
                let arr = batch.column(col_idx);
                let col_list = arrow_col_to_pylist(py, arr)?;
                columns_dict.set_item(col_name, col_list)?;
            }
        }
        out.set_item("columns_dict", columns_dict)?;
        out.set_item("rows_affected", 0i64)?;
        Ok(Some(out.into()))
    }

    /// Retrieve rows for `numeric_col <op> value LIMIT N [OFFSET M]` without
    /// going through SQL parsing. The scan is executed on every call.
    fn retrieve_by_numeric_range_limit(
        &self,
        py: Python<'_>,
        filter_column: String,
        op: String,
        value: f64,
        limit: usize,
        offset: usize,
    ) -> PyResult<Option<PyObject>> {
        if filter_column.is_empty() {
            return Ok(None);
        }

        let (low, high) = match op.as_str() {
            "=" => (value, value),
            ">" => (next_up_f64_binding(value), f64::INFINITY),
            ">=" => (value, f64::INFINITY),
            "<" => (f64::NEG_INFINITY, next_down_f64_binding(value)),
            "<=" => (f64::NEG_INFINITY, value),
            _ => return Ok(None),
        };

        let (table_path, table_name) = self.get_current_table_info()?;
        let cache_key = Self::backend_cache_key(&table_path, &table_name);
        let backend_opt: Option<Arc<TableStorageBackend>> = self
            .cached_backends
            .get(&cache_key)
            .map(|v| Arc::clone(&v))
            .or_else(|| {
                crate::query::get_cached_backend_pub(&table_path)
                    .ok()
                    .map(|b| {
                        self.cached_backends
                            .insert(cache_key.clone(), Arc::clone(&b));
                        b
                    })
            });

        let Some(backend) = backend_opt else {
            return Ok(None);
        };
        if backend.has_pending_deltas()
            || backend.has_delta()
            || backend.pending_v4_in_memory_rows() > 0
        {
            return Ok(None);
        }

        let needed = offset.saturating_add(limit);
        let cols_result = py.allow_threads(
            || -> io::Result<Option<crate::storage::on_demand::MmapBatchColumns>> {
                let Some(mut indices) =
                    backend.scan_numeric_range_mmap(&filter_column, low, high, Some(needed))?
                else {
                    return Ok(None);
                };
                if offset == 0 {
                    indices.truncate(limit);
                    backend
                        .storage
                        .extract_rows_by_indices_mmap_columns(&indices, None)
                } else {
                    let final_indices: Vec<usize> =
                        indices.into_iter().skip(offset).take(limit).collect();
                    backend
                        .storage
                        .extract_rows_by_indices_mmap_columns(&final_indices, None)
                }
            },
        );

        let Some(batch_cols) = cols_result.map_err(|e| PyIOError::new_err(e.to_string()))? else {
            return Ok(None);
        };
        let Some(columns_dict) = mmap_batch_columns_to_pydict(py, batch_cols, None)? else {
            return Ok(None);
        };

        let out = PyDict::new_bound(py);
        out.set_item("columns_dict", columns_dict)?;
        out.set_item("rows_affected", 0i64)?;
        Ok(Some(out.into()))
    }

    /// Execute a simple filtered numeric aggregation without SQL executor
    /// dispatch. The storage layer scans and aggregates on every call.
    fn execute_filtered_numeric_agg(
        &self,
        py: Python<'_>,
        sql: String,
        table: String,
        filter_column: String,
        op: String,
        value: f64,
    ) -> PyResult<Option<PyObject>> {
        if table.is_empty() || filter_column.is_empty() {
            return Ok(None);
        }

        let agg_exprs = match parse_agg_select(&sql) {
            Some(exprs) if !exprs.is_empty() => exprs,
            _ => return Ok(None),
        };

        let (low, high) = match op.as_str() {
            "=" => (value, value),
            ">" => (next_up_f64_binding(value), f64::INFINITY),
            ">=" => (value, f64::INFINITY),
            "<" => (f64::NEG_INFINITY, next_down_f64_binding(value)),
            "<=" => (f64::NEG_INFINITY, value),
            _ => return Ok(None),
        };

        let (default_table_path, default_table_name) = self.get_current_table_info()?;
        let base_dir = self.current_base_dir();
        let target_table = table.trim_matches('"').trim_matches('`').to_string();
        let target_path =
            if target_table.eq_ignore_ascii_case("default") || target_table == default_table_name {
                default_table_path
            } else {
                self.table_paths
                    .read()
                    .get(&target_table)
                    .cloned()
                    .unwrap_or_else(|| base_dir.join(format!("{}.apex", target_table)))
            };

        let cache_key = Self::backend_cache_key(&target_path, &target_table);
        let backend_opt: Option<Arc<TableStorageBackend>> = self
            .cached_backends
            .get(&cache_key)
            .map(|v| Arc::clone(&v))
            .or_else(|| {
                crate::query::get_cached_backend_pub(&target_path)
                    .ok()
                    .map(|b| {
                        self.cached_backends
                            .insert(cache_key.clone(), Arc::clone(&b));
                        b
                    })
            });

        let Some(backend) = backend_opt else {
            return Ok(None);
        };
        if backend.has_pending_deltas()
            || backend.has_delta()
            || backend.pending_v4_in_memory_rows() > 0
        {
            return Ok(None);
        }

        let mut unique_cols: Vec<String> = Vec::new();
        for (func, col, _) in &agg_exprs {
            let is_count_star = func == "COUNT"
                && col
                    .as_ref()
                    .map(|c| {
                        c == "*"
                            || c.chars()
                                .next()
                                .map(|ch| ch.is_ascii_digit())
                                .unwrap_or(false)
                    })
                    .unwrap_or(true);
            if is_count_star {
                if !unique_cols.iter().any(|c| c == "*") {
                    unique_cols.push("*".to_string());
                }
            } else if let Some(c) = col {
                if !unique_cols.contains(c) {
                    unique_cols.push(c.clone());
                }
            }
        }
        if unique_cols.is_empty() {
            return Ok(None);
        }

        let col_refs: Vec<&str> = unique_cols.iter().map(String::as_str).collect();
        let agg_result = py.allow_threads(|| {
            backend.execute_filtered_numeric_agg_mmap(&filter_column, low, high, &col_refs)
        });
        let Some(stats) = agg_result.map_err(|e| PyIOError::new_err(e.to_string()))? else {
            return Ok(None);
        };

        let mut stat_map: std::collections::HashMap<&str, (i64, f64, f64, f64, bool)> =
            std::collections::HashMap::new();
        for (idx, col_name) in col_refs.iter().enumerate() {
            if idx < stats.len() {
                stat_map.insert(col_name, stats[idx]);
            }
        }
        let match_count = stat_map.get("*").map(|s| s.0).unwrap_or(0);

        let out = PyDict::new_bound(py);
        let columns_dict = PyDict::new_bound(py);
        for (func, col, alias) in &agg_exprs {
            let output_name = if let Some(a) = alias {
                a.clone()
            } else if let Some(c) = col {
                format!("{}({})", func, c)
            } else {
                format!("{}(*)", func)
            };

            match func.as_str() {
                "COUNT" => {
                    let count = if let Some(c) = col {
                        let is_count_star = c == "*"
                            || c.chars()
                                .next()
                                .map(|ch| ch.is_ascii_digit())
                                .unwrap_or(false);
                        if is_count_star {
                            match_count
                        } else {
                            stat_map.get(c.as_str()).map(|s| s.0).unwrap_or(0)
                        }
                    } else {
                        match_count
                    };
                    columns_dict.set_item(&output_name, PyList::new_bound(py, &[count]))?;
                }
                "SUM" | "AVG" | "MIN" | "MAX" => {
                    if let Some(c) = col {
                        let (count, sum, min_v, max_v, is_int) = stat_map
                            .get(c.as_str())
                            .copied()
                            .unwrap_or((0, 0.0, 0.0, 0.0, false));
                        match func.as_str() {
                            "SUM" => {
                                if is_int {
                                    columns_dict.set_item(
                                        &output_name,
                                        PyList::new_bound(py, &[sum as i64]),
                                    )?;
                                } else {
                                    columns_dict
                                        .set_item(&output_name, PyList::new_bound(py, &[sum]))?;
                                }
                            }
                            "AVG" => {
                                let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                                columns_dict
                                    .set_item(&output_name, PyList::new_bound(py, &[avg]))?;
                            }
                            "MIN" => {
                                if is_int {
                                    columns_dict.set_item(
                                        &output_name,
                                        PyList::new_bound(py, &[min_v as i64]),
                                    )?;
                                } else {
                                    columns_dict
                                        .set_item(&output_name, PyList::new_bound(py, &[min_v]))?;
                                }
                            }
                            "MAX" => {
                                if is_int {
                                    columns_dict.set_item(
                                        &output_name,
                                        PyList::new_bound(py, &[max_v as i64]),
                                    )?;
                                } else {
                                    columns_dict
                                        .set_item(&output_name, PyList::new_bound(py, &[max_v]))?;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => return Ok(None),
            }
        }
        out.set_item("columns_dict", columns_dict)?;
        out.set_item("rows_affected", 0i64)?;
        Ok(Some(out.into()))
    }

    /// Retrieve selected columns for one ID as a row dict; optimized for SQL
    /// point lookups immediately consumed via ResultView.to_dict().
    fn retrieve_projected_row(
        &self,
        py: Python<'_>,
        id: i64,
        columns: Vec<String>,
    ) -> PyResult<Option<PyObject>> {
        if id < 0 || columns.is_empty() {
            return Ok(None);
        }
        let (table_path, table_name) = self.get_current_table_info()?;
        let cache_key = Self::backend_cache_key(&table_path, &table_name);
        let backend_opt: Option<Arc<TableStorageBackend>> = self
            .cached_backends
            .get(&cache_key)
            .map(|v| Arc::clone(&v))
            .or_else(|| {
                crate::query::get_cached_backend_pub(&table_path)
                    .ok()
                    .map(|b| {
                        self.cached_backends
                            .insert(cache_key.clone(), Arc::clone(&b));
                        b
                    })
            });

        let Some(backend) = backend_opt else {
            return Ok(None);
        };
        if backend.has_pending_deltas() || backend.pending_v4_in_memory_rows() > 0 {
            return Ok(None);
        }

        let requested_cols: Vec<&str> = columns.iter().map(String::as_str).collect();
        let vals_result = backend
            .storage
            .retrieve_rcix_projected(id as u64, &requested_cols)
            .ok()
            .flatten()
            .or_else(|| {
                backend
                    .storage
                    .read_row_by_id_values(id as u64)
                    .ok()
                    .flatten()
            });
        let Some(vals) = vals_result else {
            return Ok(None);
        };

        let Some(row) = projected_values_to_row_dict(py, &vals, &columns)? else {
            return Ok(None);
        };
        Ok(Some(row.into()))
    }

    /// Narrow fast path for `UPDATE table SET numeric_col = literal WHERE _id = N`.
    /// Returns None whenever indexes, constraints, deltas, unsupported column types,
    /// or storage layout require the general SQL executor for correctness.
    fn update_numeric_by_id_inplace(
        &self,
        id: i64,
        column: String,
        value: f64,
    ) -> PyResult<Option<i64>> {
        if id < 0 || column == "_id" {
            return Ok(None);
        }

        let (table_path, table_name) = self.get_current_table_info()?;
        let backend_cache_key = Self::backend_cache_key(&table_path, &table_name);
        let cache_key = format!("{}\0{}", backend_cache_key, column);
        let replace_cache_key = Self::replace_row_cache_key(&table_path, &table_name, id as u64);

        let backend_opt: Option<Arc<TableStorageBackend>> = self
            .cached_backends
            .get(&backend_cache_key)
            .map(|v| Arc::clone(&v))
            .or_else(|| {
                crate::query::get_cached_backend_pub(&table_path)
                    .ok()
                    .map(|b| {
                        self.cached_backends
                            .insert(backend_cache_key.clone(), Arc::clone(&b));
                        b
                    })
            })
            .or_else(|| TableStorageBackend::open(&table_path).ok().map(Arc::new));

        let Some(backend) = backend_opt else {
            return Ok(None);
        };
        if backend.has_pending_deltas() || backend.pending_v4_in_memory_rows() > 0 {
            return Ok(None);
        }

        let col_type = if let Some(entry) = self.update_by_id_numeric_cache.get(&cache_key) {
            *entry.value()
        } else {
            let base_dir = table_path
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .to_path_buf();
            if let Ok(index_mgr) = crate::storage::index::IndexManager::load(&table_name, &base_dir)
            {
                if index_mgr
                    .list_indexes()
                    .iter()
                    .any(|meta| meta.effective_columns().iter().any(|c| *c == column))
                {
                    return Ok(None);
                }
            }

            if backend.storage.has_constraints() {
                return Ok(None);
            }

            let Some((_, col_type)) = backend
                .storage
                .get_schema()
                .into_iter()
                .find(|(name, _)| name == &column)
            else {
                return Ok(None);
            };

            let is_numeric = matches!(
                col_type,
                crate::storage::on_demand::ColumnType::Float64
                    | crate::storage::on_demand::ColumnType::Float32
                    | crate::storage::on_demand::ColumnType::Int64
                    | crate::storage::on_demand::ColumnType::Int32
                    | crate::storage::on_demand::ColumnType::Int16
                    | crate::storage::on_demand::ColumnType::Int8
                    | crate::storage::on_demand::ColumnType::UInt8
                    | crate::storage::on_demand::ColumnType::UInt16
                    | crate::storage::on_demand::ColumnType::UInt32
                    | crate::storage::on_demand::ColumnType::UInt64
                    | crate::storage::on_demand::ColumnType::Timestamp
                    | crate::storage::on_demand::ColumnType::Date
            );
            if !is_numeric {
                return Ok(None);
            }
            self.update_by_id_numeric_cache.insert(cache_key, col_type);
            col_type
        };

        let bytes = match col_type {
            crate::storage::on_demand::ColumnType::Float64
            | crate::storage::on_demand::ColumnType::Float32 => value.to_le_bytes(),
            crate::storage::on_demand::ColumnType::Int64
            | crate::storage::on_demand::ColumnType::Int32
            | crate::storage::on_demand::ColumnType::Int16
            | crate::storage::on_demand::ColumnType::Int8
            | crate::storage::on_demand::ColumnType::UInt8
            | crate::storage::on_demand::ColumnType::UInt16
            | crate::storage::on_demand::ColumnType::UInt32
            | crate::storage::on_demand::ColumnType::UInt64
            | crate::storage::on_demand::ColumnType::Timestamp
            | crate::storage::on_demand::ColumnType::Date => (value as i64).to_le_bytes(),
            _ => return Ok(None),
        };

        let cell_cache_key = format!("{}\0{}\0{}", backend_cache_key, column, id);
        if let Some(entry) = self.update_by_id_cell_cache.get(&cell_cache_key) {
            let cached = *entry.value();
            match backend.storage.update_numeric_cell_cached(
                cached.footer_offset,
                cached.null_byte_file_offset,
                cached.null_mask,
                cached.value_file_offset,
                &bytes,
            ) {
                Ok(Some((n, physically_written))) => {
                    if physically_written {
                        self.replace_exact_row_cache.remove(&replace_cache_key);
                        crate::storage::engine::engine().invalidate(&table_path);
                        crate::query::executor::invalidate_storage_cache(&table_path);
                        crate::query::planner::invalidate_table_stats(
                            &table_path.to_string_lossy(),
                        );
                    }
                    return Ok(Some(n));
                }
                Ok(None) => {
                    self.update_by_id_cell_cache.remove(&cell_cache_key);
                }
                Err(e) => return Err(PyIOError::new_err(e.to_string())),
            }
        }

        if let Ok(Some((footer_offset, null_byte_file_offset, null_mask, value_file_offset))) =
            backend
                .storage
                .locate_numeric_cell_for_update(id as u64, &column)
        {
            if footer_offset != 0 && value_file_offset != 0 {
                let cached = NumericUpdateCellCache {
                    footer_offset,
                    null_byte_file_offset,
                    null_mask,
                    value_file_offset,
                };
                self.update_by_id_cell_cache
                    .insert(cell_cache_key.clone(), cached);
                match backend.storage.update_numeric_cell_cached(
                    cached.footer_offset,
                    cached.null_byte_file_offset,
                    cached.null_mask,
                    cached.value_file_offset,
                    &bytes,
                ) {
                    Ok(Some((n, physically_written))) => {
                        if physically_written {
                            self.replace_exact_row_cache.remove(&replace_cache_key);
                            crate::storage::engine::engine().invalidate(&table_path);
                            crate::query::executor::invalidate_storage_cache(&table_path);
                            crate::query::planner::invalidate_table_stats(
                                &table_path.to_string_lossy(),
                            );
                        }
                        return Ok(Some(n));
                    }
                    Ok(None) => {
                        self.update_by_id_cell_cache.remove(&cell_cache_key);
                    }
                    Err(e) => return Err(PyIOError::new_err(e.to_string())),
                }
            }
        }

        match backend.update_by_id_inplace(id as u64, &column, &bytes) {
            Ok(Some((n, physically_written))) => {
                if physically_written {
                    self.replace_exact_row_cache.remove(&replace_cache_key);
                    crate::storage::engine::engine().invalidate(&table_path);
                    crate::query::executor::invalidate_storage_cache(&table_path);
                    crate::query::planner::invalidate_table_stats(&table_path.to_string_lossy());
                }
                Ok(Some(n))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyIOError::new_err(e.to_string())),
        }
    }

    /// Retrieve selected columns for multiple IDs in table order.
    ///
    /// SQL results for `_id IN (...)` are expected to be de-duplicated and
    /// returned in storage order when no ORDER BY is present; for Apex row IDs
    /// that means ascending `_id`.
    fn retrieve_many_projected(
        &self,
        py: Python<'_>,
        ids: Vec<i64>,
        columns: Vec<String>,
    ) -> PyResult<Option<PyObject>> {
        if ids.is_empty() || columns.is_empty() {
            return Ok(None);
        }
        let (table_path, table_name) = self.get_current_table_info()?;
        let backend_opt: Option<Arc<TableStorageBackend>> = self
            .cached_backends
            .get(&table_name)
            .map(|v| Arc::clone(&v))
            .or_else(|| {
                crate::query::get_cached_backend_pub(&table_path)
                    .ok()
                    .map(|b| {
                        self.cached_backends
                            .insert(table_name.clone(), Arc::clone(&b));
                        b
                    })
            });

        let Some(backend) = backend_opt else {
            return Ok(None);
        };
        if backend.has_pending_deltas() {
            return Ok(None);
        }

        let mut sorted_ids: Vec<u64> = ids
            .into_iter()
            .filter(|id| *id >= 0)
            .map(|id| id as u64)
            .collect();
        if sorted_ids.is_empty() {
            return Ok(None);
        }
        sorted_ids.sort_unstable();
        sorted_ids.dedup();

        if backend.storage.is_v4_format() && !backend.storage.has_v4_in_memory_data() {
            if let Ok(Some(batch_cols)) = backend.storage.retrieve_many_mmap_columns(&sorted_ids) {
                let row_count = batch_cols.row_count;
                let Some(columns_dict) =
                    mmap_batch_columns_to_pydict(py, batch_cols, Some(&columns))?
                else {
                    return Ok(None);
                };
                let out = PyDict::new_bound(py);
                out.set_item("columns_dict", columns_dict)?;
                out.set_item("rows_affected", row_count as i64)?;
                return Ok(Some(out.into()));
            }
        }

        let mut col_values: Vec<Vec<PyObject>> = (0..columns.len())
            .map(|_| Vec::with_capacity(sorted_ids.len()))
            .collect();
        let mut matched_rows = 0usize;

        for id in sorted_ids {
            let vals_result = backend
                .storage
                .retrieve_rcix(id)
                .ok()
                .flatten()
                .or_else(|| backend.storage.read_row_by_id_values(id).ok().flatten());
            let Some(vals) = vals_result else {
                continue;
            };

            let mut row_values: Vec<PyObject> = Vec::with_capacity(columns.len());
            for requested_col in &columns {
                let Some((_, val)) = vals.iter().find(|(col_name, _)| col_name == requested_col)
                else {
                    return Ok(None);
                };
                row_values.push(value_to_py(py, val)?);
            }
            for (idx, value) in row_values.into_iter().enumerate() {
                col_values[idx].push(value);
            }
            matched_rows += 1;
        }

        let columns_dict = PyDict::new_bound(py);
        for (idx, requested_col) in columns.iter().enumerate() {
            columns_dict.set_item(
                requested_col.as_str(),
                PyList::new_bound(py, &col_values[idx]),
            )?;
        }

        let out = PyDict::new_bound(py);
        out.set_item("columns_dict", columns_dict)?;
        out.set_item("rows_affected", matched_rows as i64)?;
        Ok(Some(out.into()))
    }

    /// Retrieve multiple records by IDs
    /// Uses direct storage access for optimal small-batch performance
    fn retrieve_many(&self, py: Python<'_>, ids: Vec<i64>) -> PyResult<PyObject> {
        use pyo3::types::PyDict;

        if ids.is_empty() {
            let out = PyDict::new_bound(py);
            out.set_item("columns_dict", PyDict::new_bound(py))?;
            out.set_item("rows_affected", 0i64)?;
            return Ok(out.into());
        }

        let (table_path, table_name) = self.get_current_table_info()?;

        // Try to get cached backend for direct storage access
        let maybe_cached = self
            .cached_backends
            .get(&table_name)
            .map(|v| Arc::clone(&v));
        let backend_opt: Option<Arc<TableStorageBackend>> = if let Some(b) = maybe_cached {
            Some(b)
        } else if let Ok(b) = crate::query::get_cached_backend_pub(&table_path) {
            self.cached_backends
                .insert(table_name.clone(), Arc::clone(&b));
            Some(b)
        } else {
            None
        };

        // Use direct storage batch read (one mmap pass per RG, no per-row lock overhead)
        if let Some(backend) = backend_opt {
            let ids_u64: Vec<u64> = ids.iter().map(|&id| id as u64).collect();

            // Fast path: direct mmap-to-Python columns — one footer lock + one mmap slice per RG.
            let batch_cols_opt =
                if backend.storage.is_v4_format() && !backend.storage.has_v4_in_memory_data() {
                    backend
                        .storage
                        .retrieve_many_mmap_columns(&ids_u64)
                        .ok()
                        .flatten()
                } else {
                    None
                };

            if let Some(batch_cols) = batch_cols_opt {
                let row_count = batch_cols.row_count;
                let columns_dict = mmap_batch_columns_to_pydict(py, batch_cols, None)?
                    .unwrap_or_else(|| PyDict::new_bound(py));
                let out = PyDict::new_bound(py);
                out.set_item("columns_dict", columns_dict)?;
                out.set_item("rows_affected", row_count as i64)?;
                return Ok(out.into());
            }

            // Fallback: per-row retrieve_rcix (non-RCIX RGs)
            let mut all_rows: Vec<Vec<(String, Value)>> = Vec::with_capacity(ids_u64.len());
            for &id in &ids_u64 {
                if let Ok(Some(row)) = backend.storage.retrieve_rcix(id) {
                    all_rows.push(row);
                }
            }

            if all_rows.is_empty() {
                let out = PyDict::new_bound(py);
                out.set_item("columns_dict", PyDict::new_bound(py))?;
                out.set_item("rows_affected", 0i64)?;
                return Ok(out.into());
            }

            let num_rows = all_rows.len();
            let col_names: Vec<String> = all_rows[0].iter().map(|(n, _)| n.clone()).collect();
            let num_cols = col_names.len();

            let columns_dict = PyDict::new_bound(py);
            for col_idx in 0..num_cols {
                let col_name = &col_names[col_idx];
                let mut py_list: Vec<PyObject> = Vec::with_capacity(num_rows);
                for row in &all_rows {
                    let val = value_to_py(py, &row[col_idx].1)?;
                    py_list.push(val);
                }
                let py_list_bound = PyList::new_bound(py, &py_list);
                columns_dict.set_item(col_name.as_str(), py_list_bound)?;
            }

            let out = PyDict::new_bound(py);
            out.set_item("columns_dict", columns_dict)?;
            out.set_item("rows_affected", num_rows as i64)?;
            return Ok(out.into());
        }

        // Fallback: empty result
        let out = PyDict::new_bound(py);
        out.set_item("columns_dict", PyDict::new_bound(py))?;
        out.set_item("rows_affected", 0i64)?;
        Ok(out.into())
    }

    /// Retrieve all records
    fn retrieve_all(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let (table_path, table_name) = self.get_current_table_info()?;

        let rows = py.allow_threads(|| -> PyResult<Vec<HashMap<String, Value>>> {
            let sql = format!("SELECT * FROM {}", table_name);
            let sql = sql.as_str();
            let result = ApexExecutor::execute(sql, &table_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let batch = result
                .to_record_batch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let mut rows = Vec::with_capacity(batch.num_rows());
            for row_idx in 0..batch.num_rows() {
                let mut row_data = HashMap::new();
                for (col_idx, field) in batch.schema().fields().iter().enumerate() {
                    let val = arrow_value_at(batch.column(col_idx), row_idx);
                    row_data.insert(field.name().clone(), val);
                }
                rows.push(row_data);
            }

            Ok(rows)
        })?;

        let mut result = Vec::with_capacity(rows.len());
        for row_data in rows {
            let dict = PyDict::new_bound(py);
            for (k, v) in row_data {
                dict.set_item(k, value_to_py(py, &v)?)?;
            }
            result.push(dict.into());
        }

        Ok(result)
    }

    /// Query with WHERE clause
    #[pyo3(signature = (where_clause, limit=None))]
    fn query(
        &self,
        py: Python<'_>,
        where_clause: &str,
        limit: Option<usize>,
    ) -> PyResult<Vec<PyObject>> {
        let (table_path, table_name) = self.get_current_table_info()?;

        let rows = py.allow_threads(|| -> PyResult<Vec<HashMap<String, Value>>> {
            let sql = if let Some(lim) = limit {
                format!(
                    "SELECT * FROM {} WHERE {} LIMIT {}",
                    table_name, where_clause, lim
                )
            } else {
                format!("SELECT * FROM {} WHERE {}", table_name, where_clause)
            };

            let result = ApexExecutor::execute(&sql, &table_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let batch = result
                .to_record_batch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let mut rows = Vec::with_capacity(batch.num_rows());
            for row_idx in 0..batch.num_rows() {
                let mut row_data = HashMap::new();
                for (col_idx, field) in batch.schema().fields().iter().enumerate() {
                    let val = arrow_value_at(batch.column(col_idx), row_idx);
                    row_data.insert(field.name().clone(), val);
                }
                rows.push(row_data);
            }

            Ok(rows)
        })?;

        let mut result = Vec::with_capacity(rows.len());
        for row_data in rows {
            let dict = PyDict::new_bound(py);
            for (k, v) in row_data {
                dict.set_item(k, value_to_py(py, &v)?)?;
            }
            result.push(dict.into());
        }

        Ok(result)
    }

    /// Replace a record by ID using StorageEngine
    fn replace(&self, py: Python<'_>, id: i64, data: &Bound<'_, PyDict>) -> PyResult<bool> {
        if id < 0 {
            return Ok(false);
        }

        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        let durability = self.durability;
        let replace_cache_key = Self::replace_row_cache_key(&table_path, &table_name, id as u64);

        if let Some(entry) = self.replace_exact_row_cache.get(&replace_cache_key) {
            if Self::py_dict_matches_exact_fields(data, entry.value())? {
                return Ok(true);
            }
        }

        if let Ok(backend) = self.get_backend_for_overlay(&table_path, &table_name) {
            if let Some(true) = self.row_matches_exact_py_dict(&backend, id as u64, data)? {
                return Ok(true);
            }
        }

        let fields = dict_to_values(data)?;

        if !fields.is_empty() {
            if let Ok(backend) = self.get_backend_for_overlay(&table_path, &table_name) {
                if let Some(true) = self.row_matches_exact_fields(&backend, id as u64, &fields)? {
                    return Ok(true);
                }
            }
        }

        if durability == DurabilityLevel::Fast
            && !fields.is_empty()
            && !self.table_has_secondary_indexes(&table_path, &table_name)
        {
            let backend = self.get_backend_for_overlay(&table_path, &table_name)?;
            if !backend.storage.has_constraints() {
                let schema = backend.storage.get_schema();
                let schema_cols: std::collections::HashSet<&str> =
                    schema.iter().map(|(name, _)| name.as_str()).collect();
                let schema_supported = schema.iter().all(|(_, ty)| {
                    use crate::storage::on_demand::ColumnType;
                    !matches!(
                        *ty,
                        ColumnType::Binary
                            | ColumnType::FixedList
                            | ColumnType::Float16List
                            | ColumnType::Null
                    )
                });
                let exact_schema = fields.len() == schema_cols.len()
                    && fields
                        .keys()
                        .all(|name| schema_cols.contains(name.as_str()));

                if schema_supported && exact_schema {
                    let result = py.allow_threads(|| {
                        backend
                            .delta_update_existing_row(id as u64, &fields)
                            .map_err(|e| PyIOError::new_err(e.to_string()))
                    })?;
                    if result {
                        self.replace_exact_row_cache
                            .insert(replace_cache_key.clone(), fields.clone());
                        crate::query::executor::cache_backend_pub(
                            &table_path,
                            Arc::clone(&backend),
                        );
                        crate::query::planner::invalidate_table_stats(
                            &table_path.to_string_lossy(),
                        );
                    } else {
                        self.replace_exact_row_cache.remove(&replace_cache_key);
                    }
                    return Ok(result);
                }
            }
        }

        // Acquire exclusive write lock
        let lock_file =
            Self::acquire_write_lock(&table_path).map_err(|e| PyIOError::new_err(e.to_string()))?;

        // Use StorageEngine for unified replace
        let result = py.allow_threads(|| {
            let engine = crate::storage::engine::engine();
            engine
                .replace(&table_path, id as u64, &fields, durability)
                .map_err(|e| PyIOError::new_err(e.to_string()))
        });

        Self::release_lock(lock_file);

        // Invalidate local backend cache
        self.invalidate_backend(&table_name);

        if matches!(&result, Ok(true)) {
            self.replace_exact_row_cache.remove(&replace_cache_key);
        }

        result
    }

    // ========== Schema Operations ==========

    /// Add a column to current table using StorageEngine
    fn add_column(&self, column_name: &str, column_type: &str) -> PyResult<()> {
        let dtype = match column_type.to_lowercase().as_str() {
            "int" | "int64" | "i64" | "integer" => crate::data::DataType::Int64,
            "float" | "float64" | "f64" | "double" => crate::data::DataType::Float64,
            "bool" | "boolean" => crate::data::DataType::Bool,
            "str" | "string" | "text" => crate::data::DataType::String,
            "bytes" | "binary" => crate::data::DataType::Binary,
            "float16_vector" | "float16vector" | "f16_vector" => {
                crate::data::DataType::Float16Vector
            }
            "timestamp" | "datetime" => crate::data::DataType::Timestamp,
            "date" => crate::data::DataType::Date,
            _ => crate::data::DataType::String,
        };

        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        let durability = self.durability;

        // Invalidate local backend cache before operation
        self.invalidate_backend(&table_name);

        // Acquire exclusive write lock
        let lock_file =
            Self::acquire_write_lock(&table_path).map_err(|e| PyIOError::new_err(e.to_string()))?;

        // Use StorageEngine for unified add_column
        let engine = crate::storage::engine::engine();
        let result = engine
            .add_column(&table_path, column_name, dtype, durability)
            .map_err(|e| PyIOError::new_err(e.to_string()));

        Self::release_lock(lock_file);

        // Invalidate local backend cache after operation
        self.invalidate_backend(&table_name);

        result
    }

    /// Drop a column from current table using StorageEngine
    fn drop_column(&self, column_name: &str) -> PyResult<()> {
        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        let durability = self.durability;

        // Invalidate local backend cache before operation
        self.invalidate_backend(&table_name);

        // Acquire exclusive write lock
        let lock_file =
            Self::acquire_write_lock(&table_path).map_err(|e| PyIOError::new_err(e.to_string()))?;

        // Use StorageEngine for unified drop_column
        let engine = crate::storage::engine::engine();
        let result = engine
            .drop_column(&table_path, column_name, durability)
            .map_err(|e| PyIOError::new_err(e.to_string()));

        Self::release_lock(lock_file);

        // Invalidate local backend cache after operation
        self.invalidate_backend(&table_name);

        result
    }

    /// Rename a column using StorageEngine
    fn rename_column(&self, old_name: &str, new_name: &str) -> PyResult<()> {
        let table_path = self.get_current_table_path()?;
        let table_name = self.current_table.read().clone();
        let durability = self.durability;

        // Acquire exclusive write lock
        let lock_file =
            Self::acquire_write_lock(&table_path).map_err(|e| PyIOError::new_err(e.to_string()))?;

        // Use StorageEngine for unified rename_column
        let engine = crate::storage::engine::engine();
        let result = engine
            .rename_column(&table_path, old_name, new_name, durability)
            .map_err(|e| PyIOError::new_err(e.to_string()));

        Self::release_lock(lock_file);

        // Invalidate local backend cache
        self.invalidate_backend(&table_name);

        result
    }

    /// List fields (columns) in current table using StorageEngine
    fn list_fields(&self) -> PyResult<Vec<String>> {
        let table_path = self.get_current_table_path()?;

        // Acquire shared read lock
        let lock_file =
            Self::acquire_read_lock(&table_path).map_err(|e| PyIOError::new_err(e.to_string()))?;

        // Use StorageEngine for unified list_columns
        let engine = crate::storage::engine::engine();
        let columns = engine
            .list_columns(&table_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        Self::release_lock(lock_file);
        Ok(columns)
    }

    /// Get column data type using StorageEngine
    fn get_column_dtype(&self, column_name: &str) -> PyResult<Option<String>> {
        let table_path = self.get_current_table_path()?;

        // Acquire shared read lock
        let lock_file =
            Self::acquire_read_lock(&table_path).map_err(|e| PyIOError::new_err(e.to_string()))?;

        // Use StorageEngine for unified get_column_type
        let engine = crate::storage::engine::engine();
        let dtype = engine
            .get_column_type(&table_path, column_name)
            .map_err(|e| PyIOError::new_err(e.to_string()))?
            .map(|dt| format!("{:?}", dt));

        Self::release_lock(lock_file);
        Ok(dtype)
    }

    // ========== FTS Operations ==========

    /// Initialize FTS for current table
    #[pyo3(name = "_init_fts")]
    #[pyo3(signature = (index_fields=None, lazy_load=false, cache_size=10000))]
    fn init_fts(
        &self,
        index_fields: Option<Vec<String>>,
        lazy_load: bool,
        cache_size: usize,
    ) -> PyResult<()> {
        let table_name = self.current_table.read().clone();

        // Record index field configuration
        if let Some(fields) = index_fields.clone() {
            self.fts_index_fields
                .write()
                .insert(table_name.clone(), fields);
        }

        // Ensure manager exists. SQL DDL may have already built and registered
        // a manager with populated engines; reuse it so Python-side config sync
        // does not replace a freshly backfilled FTS index with an empty one.
        if self.fts_manager.read().is_none() {
            let base_dir = self.current_base_dir();
            let manager = if let Some(existing) = crate::query::executor::get_fts_manager(&base_dir)
            {
                existing
            } else {
                let fts_dir = base_dir.join("fts_indexes");
                let config = FtsConfig {
                    lazy_load,
                    cache_size,
                    ..FtsConfig::default()
                };
                Arc::new(FtsManager::new(&fts_dir, config))
            };
            // Register with the global SQL executor registry (enables MATCH() in PG Wire / Arrow Flight)
            crate::query::executor::register_fts_manager(&base_dir, manager.clone());
            *self.fts_manager.write() = Some(manager);
        } else {
            // Already initialized — ensure global registry is up to date
            let mgr_arc = self.fts_manager.read().clone();
            if let Some(m) = mgr_arc {
                crate::query::executor::register_fts_manager(&self.current_base_dir(), m);
            }
        }

        // Touch/create engine for current table
        let mgr = self.fts_manager.read();
        if let Some(m) = mgr.as_ref() {
            let _ = m.get_engine(&table_name);
        }

        Ok(())
    }

    /// Check if FTS is enabled
    #[pyo3(name = "_is_fts_enabled")]
    fn is_fts_enabled(&self) -> bool {
        self.fts_manager.read().is_some()
    }

    /// Get FTS index fields for current table
    #[pyo3(name = "_get_fts_config")]
    fn get_fts_config(&self) -> Option<Vec<String>> {
        let table_name = self.current_table.read().clone();
        self.fts_index_fields.read().get(&table_name).cloned()
    }

    /// FTS search
    #[pyo3(signature = (query, limit=None))]
    fn search_text(
        &self,
        py: Python<'_>,
        query: &str,
        limit: Option<usize>,
    ) -> PyResult<Vec<(i64, f32)>> {
        let table_name = self.current_table.read().clone();
        let base_dir = self.current_base_dir();
        let mgr = self.fts_manager.read();

        if let Some(m) = mgr.as_ref() {
            let engine = m
                .get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            // Release GIL during search for better concurrency
            let results = py.allow_threads(|| {
                crate::query::executor::wait_fts_backfill(&base_dir, &table_name);
                engine
                    .search_top_n(query, limit.unwrap_or(100))
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
            // Return with score=1.0 for each result (nanofts doesn't return scores directly)
            Ok(results.into_iter().map(|id| (id as i64, 1.0f32)).collect())
        } else {
            Err(PyRuntimeError::new_err("FTS not initialized"))
        }
    }

    /// Remove FTS engine for current table (and optionally delete index files)
    #[pyo3(name = "_fts_remove_engine")]
    #[pyo3(signature = (delete_files=false))]
    fn fts_remove_engine(&self, py: Python<'_>, delete_files: bool) -> PyResult<()> {
        let table_name = self.current_table.read().clone();
        crate::query::executor::wait_fts_backfill(&self.current_base_dir(), &table_name);

        // Remove any cached index field configuration for this table
        self.fts_index_fields.write().remove(&table_name);

        let mut mgr = self.fts_manager.write();
        if let Some(m) = mgr.as_ref() {
            // Release GIL during engine removal (I/O operation)
            py.allow_threads(|| {
                m.remove_engine(&table_name, delete_files)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
        }

        Ok(())
    }

    /// FTS fuzzy search
    #[pyo3(signature = (query, limit=None, _max_distance=None))]
    fn fuzzy_search_text(
        &self,
        py: Python<'_>,
        query: &str,
        limit: Option<usize>,
        _max_distance: Option<u8>,
    ) -> PyResult<Vec<(i64, f32)>> {
        let table_name = self.current_table.read().clone();
        let mgr = self.fts_manager.read();

        if let Some(m) = mgr.as_ref() {
            let engine = m
                .get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            // Release GIL during fuzzy search for better concurrency
            let ids: Vec<u64> = py.allow_threads(|| -> PyResult<Vec<u64>> {
                let result = engine
                    .fuzzy_search(query, limit.unwrap_or(100))
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                // Convert result handle to Vec<u64>
                Ok(result
                    .page(0, limit.unwrap_or(100))
                    .into_iter()
                    .map(|id| id as u64)
                    .collect())
            })?;
            Ok(ids.into_iter().map(|id| (id as i64, 1.0f32)).collect())
        } else {
            Err(PyRuntimeError::new_err("FTS not initialized"))
        }
    }

    /// Search and retrieve records
    #[pyo3(signature = (query, limit=None))]
    fn search_and_retrieve(
        &self,
        py: Python<'_>,
        query: &str,
        limit: Option<usize>,
    ) -> PyResult<PyObject> {
        let results = self.search_text(py, query, limit)?;
        let ids: Vec<i64> = results.into_iter().map(|(id, _)| id).collect();
        self.retrieve_many(py, ids)
    }

    /// Index a document for FTS
    #[pyo3(name = "_fts_index")]
    fn fts_index(&self, py: Python<'_>, id: i64, text: &str) -> PyResult<()> {
        let table_name = self.current_table.read().clone();
        let mgr = self.fts_manager.read();

        if let Some(m) = mgr.as_ref() {
            let engine = m
                .get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let mut fields = HashMap::new();
            fields.insert("content".to_string(), text.to_string());
            // Release GIL during indexing operation
            py.allow_threads(|| {
                engine
                    .add_document(id as u64, fields)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
        }

        Ok(())
    }

    /// Remove a document from FTS index
    #[pyo3(name = "_fts_remove")]
    fn fts_remove(&self, py: Python<'_>, id: i64) -> PyResult<()> {
        let table_name = self.current_table.read().clone();
        let mgr = self.fts_manager.read();

        if let Some(m) = mgr.as_ref() {
            let engine = m
                .get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            // Release GIL during remove operation
            py.allow_threads(|| {
                engine
                    .remove_document(id as u64)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
        }

        Ok(())
    }

    /// Flush FTS index
    #[pyo3(name = "_fts_flush")]
    fn fts_flush(&self, py: Python<'_>) -> PyResult<()> {
        let table_name = self.current_table.read().clone();
        let base_dir = self.current_base_dir();
        let mgr = self.fts_manager.read();

        if let Some(m) = mgr.as_ref() {
            let engine = m
                .get_engine(&table_name)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            // Release GIL during flush (I/O operation)
            py.allow_threads(|| {
                crate::query::executor::wait_fts_backfill(&base_dir, &table_name);
                engine
                    .flush()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
        }

        Ok(())
    }

    /// Bulk index columnar string data into FTS (no storage write, GIL released)
    #[pyo3(name = "_fts_index_columns")]
    fn fts_index_columns(
        &self,
        py: Python<'_>,
        ids: Vec<i64>,
        columns: HashMap<String, Vec<String>>,
    ) -> PyResult<usize> {
        if ids.is_empty() || columns.is_empty() {
            return Ok(0);
        }
        let table_name = self.current_table.read().clone();
        let mgr = self.fts_manager.read();
        if let Some(m) = mgr.as_ref() {
            if let Ok(engine) = m.get_engine(&table_name) {
                let count = ids.len();
                let doc_ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
                // Build owned Vec<String> columns then borrow as &str — zero extra copy
                let owned: Vec<(String, Vec<String>)> = columns.into_iter().collect();
                let columns_ref: Vec<(String, Vec<&str>)> = owned
                    .iter()
                    .map(|(name, vals)| (name.clone(), vals.iter().map(|s| s.as_str()).collect()))
                    .collect();
                py.allow_threads(|| {
                    let _ = engine.add_documents_arrow_str(&doc_ids_u32, columns_ref);
                });
                return Ok(count);
            }
        }
        Ok(0)
    }

    /// Heap-based TopK vector distance search — O(n log k), faster than ORDER BY + LIMIT.
    ///
    /// Builds a `SELECT * FROM topk_distance(col, [q], k, 'metric')` SQL and executes
    /// it via Arrow FFI for zero-copy result transfer.
    ///
    /// Parameters:
    /// - col: name of the binary vector column
    /// - query_bytes: raw little-endian float32 bytes of the query vector
    /// - k: number of nearest neighbours to return
    /// - metric: distance metric name ("l2", "cosine", "dot", "l1", "linf", "l2_squared")
    ///
    /// Returns (schema_ptr, array_ptr) for PyArrow import, or (0, 0) for empty result.
    #[pyo3(name = "_topk_distance_ffi")]
    fn topk_distance_ffi(
        &self,
        py: Python<'_>,
        col: &str,
        query_bytes: &[u8],
        k: usize,
        metric: &str,
    ) -> PyResult<(usize, usize)> {
        use crate::query::vector_ops::bytes_to_query_vec_f32;
        use arrow::array::{Array, StructArray};
        use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};

        let query_f32 = bytes_to_query_vec_f32(query_bytes).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "_topk_distance_ffi: query_bytes must be raw little-endian float32 bytes",
            )
        })?;

        let table_path = self
            .get_current_table_path()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Direct path — no SQL string formatting or parsing overhead.
        let col_owned = col.to_string();
        let metric_str = metric.to_string();
        let names = vec!["_id".to_string(), "dist".to_string()];

        let batch = py.allow_threads(|| -> PyResult<RecordBatch> {
            use crate::query::executor::get_cached_backend_pub;
            use crate::query::vector_ops::{
                topk_heap_direct_parallel, DistanceComputer, DistanceMetric,
            };
            use arrow::array::{ArrayRef, BinaryArray, Float64Array, Int64Array};
            use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};

            let metric_enum = DistanceMetric::from_str(&metric_str).ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "_topk_distance_ffi: unknown metric '{}'",
                    metric_str
                ))
            })?;

            let backend = get_cached_backend_pub(&table_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let id_field = Field::new(&names[0], ArrowDataType::Int64, false);
            let dist_field = Field::new(&names[1], ArrowDataType::Float64, false);
            let out_schema = std::sync::Arc::new(Schema::new(vec![id_field, dist_field]));

            let computer = DistanceComputer::new(metric_enum, query_f32.clone());

            // FAST PATH: zero-copy scan on OS mmap — no Arrow batch, no memcpy
            let direct_topk = backend
                .topk_fixedlist_direct(&col_owned, &computer, k)
                .ok()
                .flatten()
                .or_else(|| {
                    backend
                        .topk_binary_direct(&col_owned, &computer, k)
                        .ok()
                        .flatten()
                });
            if let Some(topk) = direct_topk {
                if topk.is_empty() {
                    return RecordBatch::try_new(
                        out_schema,
                        vec![
                            std::sync::Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
                            std::sync::Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef,
                        ],
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()));
                }
                // Read only the _id column (8MB) to map row indices → IDs
                let id_batch = backend
                    .read_columns_to_arrow(Some(&["_id"]), 0, None)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let id_col = id_batch.column_by_name("_id");
                let ids: Vec<i64> = topk
                    .iter()
                    .map(|(row_idx, _)| {
                        id_col
                            .and_then(|a| a.as_any().downcast_ref::<Int64Array>())
                            .map(|a| a.value(*row_idx))
                            .unwrap_or(*row_idx as i64)
                    })
                    .collect();
                let dists: Vec<f64> = topk.iter().map(|(_, d)| *d as f64).collect();
                return RecordBatch::try_new(
                    out_schema,
                    vec![
                        std::sync::Arc::new(Int64Array::from(ids)) as ArrayRef,
                        std::sync::Arc::new(Float64Array::from(dists)) as ArrayRef,
                    ],
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()));
            }

            // FALLBACK: Arrow path for Binary columns / compressed RGs
            let needed: &[&str] = &[&col_owned, "_id"];
            let full_batch = backend
                .read_columns_to_arrow(Some(needed), 0, None)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            if full_batch.num_rows() == 0 {
                return RecordBatch::try_new(
                    out_schema,
                    vec![
                        std::sync::Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
                        std::sync::Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef,
                    ],
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()));
            }

            let bin_col = full_batch.column_by_name(&col_owned).ok_or_else(|| {
                PyRuntimeError::new_err(format!("column '{}' not found", col_owned))
            })?;

            let topk = if let Some(fixed_arr) = bin_col
                .as_any()
                .downcast_ref::<arrow::array::FixedSizeListArray>()
            {
                use crate::query::vector_ops::topk_heap_direct_parallel_fixed;
                topk_heap_direct_parallel_fixed(fixed_arr, &computer, k)
            } else if let Some(bin_arr) = bin_col.as_any().downcast_ref::<BinaryArray>() {
                topk_heap_direct_parallel(bin_arr, &computer, k)
            } else {
                return Err(PyRuntimeError::new_err(format!(
                    "column '{}' is not a vector column",
                    col_owned
                )));
            };

            let id_col = full_batch.column_by_name("_id");
            let ids: Vec<i64> = topk
                .iter()
                .map(|(row_idx, _)| {
                    if let Some(arr) = &id_col {
                        if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
                            return a.value(*row_idx);
                        }
                    }
                    *row_idx as i64
                })
                .collect();
            let dists: Vec<f64> = topk.iter().map(|(_, d)| *d as f64).collect();

            RecordBatch::try_new(
                out_schema,
                vec![
                    std::sync::Arc::new(Int64Array::from(ids)) as ArrayRef,
                    std::sync::Arc::new(Float64Array::from(dists)) as ArrayRef,
                ],
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;

        if batch.num_rows() == 0 {
            return Ok((0, 0));
        }

        let struct_array: StructArray = batch.into();
        let array_data = struct_array.to_data();
        let (ffi_array, ffi_schema) = arrow::ffi::to_ffi(&array_data)
            .map_err(|e| PyRuntimeError::new_err(format!("FFI export failed: {}", e)))?;

        let schema_ptr = Box::into_raw(Box::new(ffi_schema)) as usize;
        let array_ptr = Box::into_raw(Box::new(ffi_array)) as usize;
        Ok((schema_ptr, array_ptr))
    }

    /// Batch TopK vector distance search — N queries in a single Rust call.
    ///
    /// Much faster than N sequential `_topk_distance_ffi` calls because:
    /// - `scan_buf` (the mmap→heap float cache) is loaded once regardless of N.
    /// - All N queries run in parallel via Rayon (outer parallelism over queries,
    ///   sequential inner scan per query — no nested Rayon contention).
    /// - The `_id` column is read only once.
    ///
    /// Parameters:
    /// - col:          name of the vector column (FixedList or Binary)
    /// - queries_bytes: raw little-endian float32 bytes of all N query vectors,
    ///                  row-major, shape (n_queries, dim)
    /// - n_queries:    number of query vectors (N)
    /// - k:            number of nearest neighbours per query
    /// - metric:       distance metric ("l2", "cosine", "dot", …)
    ///
    /// Returns raw little-endian float64 bytes of shape (N, K, 2) where
    ///   result[i * K * 2 + j * 2 + 0]  = _id  of the j-th neighbour for query i
    ///   result[i * K * 2 + j * 2 + 1]  = dist of the j-th neighbour for query i
    ///
    /// Python side: `np.frombuffer(result, dtype=np.float64).reshape(n_queries, k, 2)`
    /// Rows with fewer than k neighbours are padded with (-1, inf).
    #[pyo3(name = "_batch_topk_ffi")]
    fn batch_topk_ffi(
        &self,
        py: Python<'_>,
        col: &str,
        queries_bytes: &[u8],
        n_queries: usize,
        k: usize,
        metric: &str,
    ) -> PyResult<PyObject> {
        use crate::query::executor::get_cached_backend_pub;
        use crate::query::vector_ops::DistanceMetric;
        use arrow::array::Int64Array;
        use pyo3::types::PyBytes;

        if n_queries == 0 || k == 0 {
            let empty: Vec<u8> = vec![];
            return Ok(PyBytes::new_bound(py, &empty).into());
        }
        if queries_bytes.len() % 4 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "_batch_topk_ffi: queries_bytes length must be a multiple of 4",
            ));
        }
        let total_floats = queries_bytes.len() / 4;
        if total_floats % n_queries != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "_batch_topk_ffi: queries_bytes length must be divisible by n_queries",
            ));
        }
        let dim = total_floats / n_queries;

        // Parse raw LE f32 bytes into Vec<f32>
        let queries_f32: Vec<f32> = queries_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let table_path = self
            .get_current_table_path()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let col_owned = col.to_string();
        let metric_str = metric.to_string();
        let n_q = n_queries;

        let (all_results, ids_map) =
            py.allow_threads(|| -> PyResult<(Vec<Vec<(usize, f32)>>, Vec<i64>)> {
                let metric_enum = DistanceMetric::from_str(&metric_str).ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "_batch_topk_ffi: unknown metric '{}'",
                        metric_str
                    ))
                })?;

                let backend = get_cached_backend_pub(&table_path)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                // FAST PATH: mmap direct scan (FixedList → Binary fallback)
                let batch_results = backend
                    .batch_topk_fixedlist_direct(&col_owned, &queries_f32, n_q, k, metric_enum)
                    .ok()
                    .flatten()
                    .or_else(|| {
                        backend
                            .batch_topk_binary_direct(&col_owned, &queries_f32, n_q, k, metric_enum)
                            .ok()
                            .flatten()
                    });

                let all_results: Vec<Vec<(usize, f32)>> = if let Some(r) = batch_results {
                    r
                } else {
                    // FALLBACK: load Arrow batch, run batch topk on FixedSizeListArray / BinaryArray
                    use crate::query::vector_ops::{
                        topk_heap_direct_parallel, topk_heap_direct_parallel_fixed,
                        DistanceComputer,
                    };
                    use arrow::array::{BinaryArray, FixedSizeListArray};

                    let needed: &[&str] = &[&col_owned, "_id"];
                    let full_batch = backend
                        .read_columns_to_arrow(Some(needed), 0, None)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                    if full_batch.num_rows() == 0 {
                        return Ok((vec![vec![]; n_q], vec![]));
                    }

                    let bin_col = full_batch.column_by_name(&col_owned).ok_or_else(|| {
                        PyRuntimeError::new_err(format!("column '{}' not found", col_owned))
                    })?;

                    // Run N queries sequentially (Arrow fallback — uncommon path)
                    let mut results = Vec::with_capacity(n_q);
                    for qi in 0..n_q {
                        let q = queries_f32[qi * dim..(qi + 1) * dim].to_vec();
                        let computer = DistanceComputer::new(metric_enum, q);
                        let topk = if let Some(fixed_arr) =
                            bin_col.as_any().downcast_ref::<FixedSizeListArray>()
                        {
                            topk_heap_direct_parallel_fixed(fixed_arr, &computer, k)
                        } else if let Some(bin_arr) = bin_col.as_any().downcast_ref::<BinaryArray>()
                        {
                            topk_heap_direct_parallel(bin_arr, &computer, k)
                        } else {
                            return Err(PyRuntimeError::new_err(format!(
                                "column '{}' is not a vector column",
                                col_owned
                            )));
                        };
                        results.push(topk);
                    }

                    let id_col = full_batch.column_by_name("_id");
                    let n_rows = full_batch.num_rows();
                    let ids: Vec<i64> = (0..n_rows)
                        .map(|i| {
                            id_col
                                .and_then(|a| a.as_any().downcast_ref::<Int64Array>())
                                .map(|a| a.value(i))
                                .unwrap_or(i as i64)
                        })
                        .collect();
                    return Ok((results, ids));
                };

                // Read _id column once to map row_idx → _id
                let id_batch = backend
                    .read_columns_to_arrow(Some(&["_id"]), 0, None)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let n_rows = id_batch.num_rows();
                let id_col = id_batch.column_by_name("_id");
                let ids: Vec<i64> = (0..n_rows)
                    .map(|i| {
                        id_col
                            .and_then(|a| a.as_any().downcast_ref::<Int64Array>())
                            .map(|a| a.value(i))
                            .unwrap_or(i as i64)
                    })
                    .collect();

                Ok((all_results, ids))
            })?;

        // Encode results as flat f64 bytes: (N × K × 2), row-major
        // [i, j, 0] = id (as f64), [i, j, 1] = dist (as f64)
        // Pad with (-1.0, f64::INFINITY) when fewer than k neighbours found.
        let out_len = n_queries * k * 2;
        let mut out: Vec<u8> = Vec::with_capacity(out_len * 8);
        for qi in 0..n_queries {
            let row = if qi < all_results.len() {
                &all_results[qi]
            } else {
                &[][..]
            };
            for j in 0..k {
                let (id_f64, dist_f64) = if j < row.len() {
                    let (row_idx, dist) = row[j];
                    let id = if row_idx < ids_map.len() {
                        ids_map[row_idx]
                    } else {
                        row_idx as i64
                    };
                    (id as f64, dist as f64)
                } else {
                    (-1.0f64, f64::INFINITY)
                };
                out.extend_from_slice(&id_f64.to_le_bytes());
                out.extend_from_slice(&dist_f64.to_le_bytes());
            }
        }

        Ok(PyBytes::new_bound(py, &out).into())
    }

    /// Get FTS stats
    fn get_fts_stats(&self) -> PyResult<Option<(usize, usize)>> {
        let table_name = self.current_table.read().clone();
        let mgr = self.fts_manager.read();

        if let Some(m) = mgr.as_ref() {
            if let Ok(engine) = m.get_engine(&table_name) {
                let stats = engine.stats();
                let doc_count = stats.get("doc_count").copied().unwrap_or(0) as usize;
                let term_count = stats.get("term_count").copied().unwrap_or(0) as usize;
                Ok(Some((doc_count, term_count)))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
}

/// Batch-convert an Arrow column to a Python list. Single downcast per column.
/// Much faster than calling arrow_value_at() per element (no per-row dispatch/downcast/Value alloc).
fn arrow_col_to_pylist(py: Python<'_>, arr: &arrow::array::ArrayRef) -> PyResult<PyObject> {
    use arrow::array::*;
    use arrow::datatypes::DataType as ArrowDT;
    use pyo3::types::PyList;

    fn sampled_unique_count_i64(values: &[i64], n: usize) -> usize {
        use ahash::AHashSet;

        let sample = n.min(512);
        let step = (n / sample.max(1)).max(1);
        let mut seen = AHashSet::with_capacity(sample);
        let mut idx = 0usize;
        while idx < n && seen.len() <= sample / 2 {
            seen.insert(values[idx]);
            idx = idx.saturating_add(step);
        }
        seen.len()
    }

    fn sampled_unique_count_f64(values: &[f64], n: usize) -> usize {
        use ahash::AHashSet;

        let sample = n.min(512);
        let step = (n / sample.max(1)).max(1);
        let mut seen = AHashSet::with_capacity(sample);
        let mut idx = 0usize;
        while idx < n && seen.len() <= sample / 2 {
            seen.insert(values[idx].to_bits());
            idx = idx.saturating_add(step);
        }
        seen.len()
    }

    fn should_cache_repeated_numeric(sample_unique: usize, n: usize) -> bool {
        let sample = n.min(512);
        sample >= 64 && sample_unique.saturating_mul(4) <= sample
    }

    fn should_cache_repeated_strings(a: &StringArray, n: usize) -> bool {
        use ahash::AHashSet;

        let sample = n.min(256);
        if sample < 32 {
            return false;
        }
        let step = (n / sample).max(1);
        let mut seen: AHashSet<&str> = AHashSet::with_capacity(sample);
        let mut idx = 0usize;
        while idx < n && seen.len() <= sample / 2 {
            seen.insert(a.value(idx));
            idx = idx.saturating_add(step);
        }
        seen.len().saturating_mul(3) <= sample
    }

    let n = arr.len();
    match arr.data_type() {
        ArrowDT::Int64 => {
            let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();
            let has_nulls = a.null_count() > 0;
            if !has_nulls {
                let values = a.values();
                if should_cache_repeated_numeric(sampled_unique_count_i64(values, n), n) {
                    use pyo3::ffi;

                    let mut cache: ahash::AHashMap<i64, pyo3::PyObject> =
                        ahash::AHashMap::with_capacity(64);
                    let list_obj = unsafe {
                        let list_ptr = ffi::PyList_New(n as ffi::Py_ssize_t);
                        if list_ptr.is_null() {
                            return Err(pyo3::PyErr::fetch(py));
                        }
                        for i in 0..n {
                            let value = values[i];
                            let py_obj = match cache.get(&value) {
                                Some(obj) => obj.clone_ref(py),
                                None => {
                                    let obj = value.into_py(py);
                                    cache.insert(value, obj.clone_ref(py));
                                    obj
                                }
                            };
                            ffi::PyList_SET_ITEM(list_ptr, i as ffi::Py_ssize_t, py_obj.into_ptr());
                        }
                        pyo3::PyObject::from_owned_ptr(py, list_ptr)
                    };
                    Ok(list_obj.into())
                } else {
                    use pyo3::ffi;

                    let list_obj = unsafe {
                        let list_ptr = ffi::PyList_New(n as ffi::Py_ssize_t);
                        if list_ptr.is_null() {
                            return Err(pyo3::PyErr::fetch(py));
                        }
                        for (i, value) in values.iter().take(n).enumerate() {
                            let item = ffi::PyLong_FromLongLong(*value);
                            if item.is_null() {
                                ffi::Py_DECREF(list_ptr);
                                return Err(pyo3::PyErr::fetch(py));
                            }
                            ffi::PyList_SET_ITEM(list_ptr, i as ffi::Py_ssize_t, item);
                        }
                        pyo3::PyObject::from_owned_ptr(py, list_ptr)
                    };
                    Ok(list_obj.into())
                }
            } else {
                let list = PyList::empty_bound(py);
                for i in 0..n {
                    if a.is_null(i) {
                        list.append(py.None())?;
                    } else {
                        list.append(a.value(i))?;
                    }
                }
                Ok(list.into())
            }
        }
        ArrowDT::Float64 => {
            let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
            let has_nulls = a.null_count() > 0;
            if !has_nulls {
                let values = a.values();
                if should_cache_repeated_numeric(sampled_unique_count_f64(values, n), n) {
                    use pyo3::ffi;

                    let mut cache: ahash::AHashMap<u64, pyo3::PyObject> =
                        ahash::AHashMap::with_capacity(64);
                    let list_obj = unsafe {
                        let list_ptr = ffi::PyList_New(n as ffi::Py_ssize_t);
                        if list_ptr.is_null() {
                            return Err(pyo3::PyErr::fetch(py));
                        }
                        for i in 0..n {
                            let value = values[i];
                            let key = value.to_bits();
                            let py_obj = match cache.get(&key) {
                                Some(obj) => obj.clone_ref(py),
                                None => {
                                    let obj = value.into_py(py);
                                    cache.insert(key, obj.clone_ref(py));
                                    obj
                                }
                            };
                            ffi::PyList_SET_ITEM(list_ptr, i as ffi::Py_ssize_t, py_obj.into_ptr());
                        }
                        pyo3::PyObject::from_owned_ptr(py, list_ptr)
                    };
                    Ok(list_obj.into())
                } else {
                    use pyo3::ffi;

                    let list_obj = unsafe {
                        let list_ptr = ffi::PyList_New(n as ffi::Py_ssize_t);
                        if list_ptr.is_null() {
                            return Err(pyo3::PyErr::fetch(py));
                        }
                        for (i, value) in values.iter().take(n).enumerate() {
                            let item = ffi::PyFloat_FromDouble(*value);
                            if item.is_null() {
                                ffi::Py_DECREF(list_ptr);
                                return Err(pyo3::PyErr::fetch(py));
                            }
                            ffi::PyList_SET_ITEM(list_ptr, i as ffi::Py_ssize_t, item);
                        }
                        pyo3::PyObject::from_owned_ptr(py, list_ptr)
                    };
                    Ok(list_obj.into())
                }
            } else {
                let list = PyList::empty_bound(py);
                for i in 0..n {
                    if a.is_null(i) {
                        list.append(py.None())?;
                    } else {
                        list.append(a.value(i))?;
                    }
                }
                Ok(list.into())
            }
        }
        ArrowDT::Utf8 => {
            let a = arr.as_any().downcast_ref::<StringArray>().unwrap();
            if a.null_count() == 0 {
                if should_cache_repeated_strings(a, n) {
                    // Low-cardinality string columns benefit from interning and
                    // pre-sized list construction. High-cardinality columns like
                    // `name` are faster with the direct iterator path below.
                    use pyo3::ffi;

                    let mut cache: std::collections::HashMap<&str, pyo3::PyObject> =
                        std::collections::HashMap::with_capacity(32);
                    let list_obj = unsafe {
                        let list_ptr = ffi::PyList_New(n as ffi::Py_ssize_t);
                        if list_ptr.is_null() {
                            return Err(pyo3::PyErr::fetch(py));
                        }
                        for i in 0..n {
                            let s = a.value(i);
                            let py_obj: pyo3::PyObject = match cache.get(s) {
                                Some(o) => o.clone_ref(py),
                                None => {
                                    let o: pyo3::PyObject = s.into_py(py);
                                    cache.insert(s, o.clone_ref(py));
                                    o
                                }
                            };
                            ffi::PyList_SET_ITEM(list_ptr, i as ffi::Py_ssize_t, py_obj.into_ptr());
                        }
                        pyo3::PyObject::from_owned_ptr(py, list_ptr)
                    };
                    Ok(list_obj.into())
                } else {
                    use pyo3::ffi;
                    use std::ffi::c_char;

                    let list_obj = unsafe {
                        let list_ptr = ffi::PyList_New(n as ffi::Py_ssize_t);
                        if list_ptr.is_null() {
                            return Err(pyo3::PyErr::fetch(py));
                        }
                        for i in 0..n {
                            let s = a.value(i);
                            let item = ffi::PyUnicode_FromStringAndSize(
                                s.as_ptr() as *const c_char,
                                s.len() as ffi::Py_ssize_t,
                            );
                            if item.is_null() {
                                ffi::Py_DECREF(list_ptr);
                                return Err(pyo3::PyErr::fetch(py));
                            }
                            ffi::PyList_SET_ITEM(list_ptr, i as ffi::Py_ssize_t, item);
                        }
                        pyo3::PyObject::from_owned_ptr(py, list_ptr)
                    };
                    Ok(list_obj.into())
                }
            } else {
                let list = PyList::empty_bound(py);
                for i in 0..n {
                    if a.is_null(i) {
                        list.append(py.None())?;
                    } else {
                        let s = a.value(i);
                        if s == "\x00__NULL__\x00" {
                            list.append(py.None())?;
                        } else {
                            list.append(s)?;
                        }
                    }
                }
                Ok(list.into())
            }
        }
        ArrowDT::Boolean => {
            let a = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
            let list = PyList::empty_bound(py);
            for i in 0..n {
                if a.is_null(i) {
                    list.append(py.None())?;
                } else {
                    list.append(a.value(i))?;
                }
            }
            Ok(list.into())
        }
        _ => {
            // Fallback: per-element generic path
            let list = PyList::empty_bound(py);
            for i in 0..n {
                list.append(value_to_py(py, &arrow_value_at(arr, i))?)?;
            }
            Ok(list.into())
        }
    }
}

/// Extract a Value from an Arrow array at a given index
fn arrow_value_at(array: &arrow::array::ArrayRef, idx: usize) -> Value {
    use arrow::array::*;
    use arrow::datatypes::DataType as ArrowDataType;

    if array.is_null(idx) {
        return Value::Null;
    }

    match array.data_type() {
        ArrowDataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            Value::Int64(arr.value(idx))
        }
        ArrowDataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            Value::Int64(arr.value(idx) as i64)
        }
        ArrowDataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            Value::Float64(arr.value(idx))
        }
        ArrowDataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            let s = arr.value(idx);
            // Check for NULL marker
            if s == "\x00__NULL__\x00" {
                Value::Null
            } else {
                Value::String(s.to_string())
            }
        }
        ArrowDataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            Value::Bool(arr.value(idx))
        }
        ArrowDataType::UInt64 => {
            let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            Value::UInt64(arr.value(idx))
        }
        ArrowDataType::Binary => {
            let arr = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            Value::Binary(arr.value(idx).to_vec())
        }
        ArrowDataType::Dictionary(_, _) => {
            // Handle DictionaryArray<UInt32Type> with Utf8 values
            use arrow::datatypes::UInt32Type;
            if let Some(dict_arr) = array.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
                if dict_arr.is_null(idx) {
                    Value::Null
                } else {
                    let key = dict_arr.keys().value(idx) as usize;
                    let values = dict_arr.values();
                    if let Some(str_values) = values.as_any().downcast_ref::<StringArray>() {
                        if key < str_values.len() {
                            let s = str_values.value(key);
                            if s == "\x00__NULL__\x00" {
                                Value::Null
                            } else {
                                Value::String(s.to_string())
                            }
                        } else {
                            Value::Null
                        }
                    } else {
                        Value::Null
                    }
                }
            } else {
                Value::Null
            }
        }
        _ => Value::Null,
    }
}
