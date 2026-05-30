//! Native Query Executor
//!
//! This module provides a pure Arrow-based query execution engine that operates
//! directly on OnDemandStorage without requiring ColumnTable.
//!
//! Architecture:
//! - Reads columns on-demand from storage
//! - Performs all filtering/projection/aggregation using Arrow compute kernels
//! - Returns Arrow RecordBatch directly (zero-copy to Python)

use ahash::AHashMap;
use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray, UInt64Array,
};
use arrow::compute::kernels::cmp;
use arrow::compute::kernels::numeric as arith;
use arrow::compute::{self, SortOptions};
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use dashmap::DashMap;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::query::jit::{
    simd_max_i64, simd_min_i64, simd_sum_f64, simd_sum_i64, ExprJIT, FilterFnI64,
};
use crate::query::planner::{
    get_table_stats, invalidate_table_stats, ExecutionStrategy, QueryPlanner,
};
use crate::query::sql_parser::BinaryOperator;
use crate::query::sql_parser::FromItem;
use crate::query::{
    AggregateFunc, JoinClause, JoinType, SelectColumn, SelectStatement, SqlExpr, SqlParser,
    SqlStatement, UnionStatement,
};

/// Zone Map optimization result for filter pruning
#[derive(PartialEq, Eq, Clone, Copy)]
enum ZoneMapResult {
    NoMatch,  // Filter definitely won't match any rows
    MayMatch, // Filter might match some rows
}
use crate::data::{DataType, Value};
use crate::storage::TableStorageBackend;
use ahash::AHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

// ============================================================================
// Global SQL parse cache — avoids re-tokenizing/parsing the same SQL across cold iterations
// ============================================================================
static SQL_PARSE_CACHE: Lazy<RwLock<AHashMap<String, Vec<SqlStatement>>>> =
    Lazy::new(|| RwLock::new(AHashMap::new()));

// ============================================================================
// Thread-local root directory for multi-database cross-db table resolution
// Set by Python bindings before calling execute_with_base_dir when a named
// database is active. Allows resolve_table_path to locate db.table references.
// ============================================================================
thread_local! {
    static QUERY_ROOT_DIR: std::cell::RefCell<Option<std::path::PathBuf>> =
        std::cell::RefCell::new(None);
}

// ============================================================================
// Thread-local temp directory for CREATE TEMP TABLE storage
// Set by embedded layer before calling execute_with_base_dir.
// ============================================================================
thread_local! {
    static TEMP_DIR: std::cell::RefCell<Option<std::path::PathBuf>> =
        std::cell::RefCell::new(None);
}

// ============================================================================
// Thread-local session variables — SET VARIABLE / RESET VARIABLE / $varname
// ============================================================================
thread_local! {
    static SESSION_VARS: std::cell::RefCell<AHashMap<String, crate::data::Value>> =
        std::cell::RefCell::new(AHashMap::new());
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct ViewCatalog {
    views: HashMap<String, SelectStatement>,
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

/// Store a session variable (SET VARIABLE name = value).
pub fn set_session_variable(name: &str, value: crate::data::Value) {
    SESSION_VARS.with(|m| m.borrow_mut().insert(name.to_lowercase(), value));
}

/// Remove a session variable (RESET VARIABLE name).
pub fn reset_session_variable(name: &str) {
    SESSION_VARS.with(|m| m.borrow_mut().remove(&name.to_lowercase()));
}

/// Retrieve a session variable by name.
pub fn get_session_variable(name: &str) -> Option<crate::data::Value> {
    SESSION_VARS.with(|m| m.borrow().get(&name.to_lowercase()).cloned())
}

/// Set the root directory for the current thread's query context.
/// Call this before execute_with_base_dir when using named databases.
pub fn set_query_root_dir(root_dir: &Path) {
    QUERY_ROOT_DIR.with(|r| *r.borrow_mut() = Some(root_dir.to_path_buf()));
}

/// Clear the root directory from the current thread's query context.
pub fn clear_query_root_dir() {
    QUERY_ROOT_DIR.with(|r| *r.borrow_mut() = None);
}

/// Get the root directory for the current thread's query context.
pub fn get_query_root_dir() -> Option<std::path::PathBuf> {
    QUERY_ROOT_DIR.with(|r| r.borrow().clone())
}

/// Set the temp directory for the current thread's query context.
pub fn set_temp_dir(dir: &Path) {
    TEMP_DIR.with(|r| *r.borrow_mut() = Some(dir.to_path_buf()));
}

/// Clear the temp directory from the current thread's query context.
pub fn clear_temp_dir() {
    TEMP_DIR.with(|r| *r.borrow_mut() = None);
}

/// Get the temp directory for the current thread's query context.
pub fn get_temp_dir() -> Option<std::path::PathBuf> {
    TEMP_DIR.with(|r| r.borrow().clone())
}

// ============================================================================
// Helper functions to reduce code duplication
// ============================================================================

/// Create an InvalidInput error with message
#[inline]
fn err_input(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg.into())
}

/// Create an InvalidData error with message  
#[inline]
fn err_data(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Create an Unsupported error with message
#[inline]
fn err_unsupported(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, msg.into())
}

/// Create a NotFound error with message
#[inline]
fn err_not_found(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::NotFound, msg.into())
}

/// Helper to apply a unary function on numeric arrays
#[inline]
fn map_numeric_unary<F1, F2>(
    arr: &ArrayRef,
    batch_rows: usize,
    int_fn: F1,
    float_fn: F2,
    func_name: &str,
) -> io::Result<ArrayRef>
where
    F1: Fn(i64) -> i64,
    F2: Fn(f64) -> f64,
{
    if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
        let result: Vec<Option<i64>> = (0..batch_rows)
            .map(|i| {
                if int_arr.is_null(i) {
                    None
                } else {
                    Some(int_fn(int_arr.value(i)))
                }
            })
            .collect();
        Ok(Arc::new(Int64Array::from(result)))
    } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
        let result: Vec<Option<f64>> = (0..batch_rows)
            .map(|i| {
                if float_arr.is_null(i) {
                    None
                } else {
                    Some(float_fn(float_arr.value(i)))
                }
            })
            .collect();
        Ok(Arc::new(Float64Array::from(result)))
    } else {
        Err(err_data(format!("{} requires numeric argument", func_name)))
    }
}

/// Helper to apply a unary string function
#[inline]
fn map_string_unary<F>(
    arr: &ArrayRef,
    batch_rows: usize,
    f: F,
    func_name: &str,
) -> io::Result<ArrayRef>
where
    F: Fn(&str) -> String,
{
    if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
        let result: Vec<Option<String>> = (0..batch_rows)
            .map(|i| {
                if str_arr.is_null(i) {
                    None
                } else {
                    Some(f(str_arr.value(i)))
                }
            })
            .collect();
        Ok(Arc::new(StringArray::from(
            result.iter().map(|s| s.as_deref()).collect::<Vec<_>>(),
        )))
    } else {
        Err(err_data(format!("{} requires string argument", func_name)))
    }
}

/// Helper to apply a unary string function returning &str (no allocation)
#[inline]
fn map_string_unary_ref<'a, F>(
    arr: &'a ArrayRef,
    batch_rows: usize,
    f: F,
    func_name: &str,
) -> io::Result<ArrayRef>
where
    F: Fn(&'a str) -> &'a str,
{
    if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
        let result: Vec<Option<&str>> = (0..batch_rows)
            .map(|i| {
                if str_arr.is_null(i) {
                    None
                } else {
                    Some(f(str_arr.value(i)))
                }
            })
            .collect();
        Ok(Arc::new(StringArray::from(result)))
    } else {
        Err(err_data(format!("{} requires string argument", func_name)))
    }
}

/// Helper to apply a string-to-int function
#[inline]
fn map_string_to_int<F>(
    arr: &ArrayRef,
    batch_rows: usize,
    f: F,
    func_name: &str,
) -> io::Result<ArrayRef>
where
    F: Fn(&str) -> i64,
{
    if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
        let result: Vec<Option<i64>> = (0..batch_rows)
            .map(|i| {
                if str_arr.is_null(i) {
                    None
                } else {
                    Some(f(str_arr.value(i)))
                }
            })
            .collect();
        Ok(Arc::new(Int64Array::from(result)))
    } else {
        Err(err_data(format!("{} requires string argument", func_name)))
    }
}

/// Helper to apply an int-to-string function
#[inline]
fn map_int_to_string<F>(
    arr: &ArrayRef,
    batch_rows: usize,
    f: F,
    func_name: &str,
) -> io::Result<ArrayRef>
where
    F: Fn(i64) -> Option<String>,
{
    if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
        let result: Vec<Option<String>> = (0..batch_rows)
            .map(|i| {
                if int_arr.is_null(i) {
                    None
                } else {
                    f(int_arr.value(i))
                }
            })
            .collect();
        Ok(Arc::new(StringArray::from(
            result.iter().map(|s| s.as_deref()).collect::<Vec<_>>(),
        )))
    } else {
        Err(err_data(format!("{} requires int argument", func_name)))
    }
}

// Global storage cache to avoid repeated open() calls which load all IDs
// Key: canonical path, Value: (backend, last_modified_time, last_access_time)
// Uses LRU eviction when cache exceeds MAX_CACHE_ENTRIES
const MAX_CACHE_ENTRIES: usize = 64; // Limit cache to 64 tables

type CacheEntry = (
    Arc<TableStorageBackend>,
    std::time::SystemTime,
    Arc<AtomicU64>,
);
// DashMap provides fine-grained locking - concurrent reads don't block each other
static STORAGE_CACHE: Lazy<DashMap<PathBuf, CacheEntry>> = Lazy::new(DashMap::new);

/// Returns current time as nanoseconds since UNIX_EPOCH (fits in u64 until year 2554)
#[inline(always)]
fn now_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Returns nanoseconds elapsed since a stored timestamp
#[inline(always)]
fn nanos_elapsed(stored: u64) -> u64 {
    now_nanos().saturating_sub(stored)
}

// ============================================================================
// Per-table Write Locks — serializes concurrent writes to the same table
// ============================================================================
// Two-layer locking for concurrent access safety:
// Layer 1: parking_lot::Mutex (~10-20ns uncontended) — same-process threads
// Layer 2: fs2 flock on cached File handle (~0.5μs) — cross-process safety
//
// The File handle is opened once and cached, so repeated writes only pay
// the flock() syscall cost, not open()+flock().
struct TableLock {
    mutex: parking_lot::Mutex<()>,
    file: Option<std::fs::File>,
}

static TABLE_WRITE_LOCKS: Lazy<RwLock<AHashMap<PathBuf, Arc<TableLock>>>> =
    Lazy::new(|| RwLock::new(AHashMap::with_capacity(32)));

/// Get or create a per-table lock entry (Mutex + cached fs2 file handle).
fn get_table_lock(table_path: &Path) -> Arc<TableLock> {
    // Fast path: read-lock the map
    {
        let locks = TABLE_WRITE_LOCKS.read();
        if let Some(lock) = locks.get(table_path) {
            return lock.clone();
        }
    }
    // Slow path: create lock + open sidecar .lock file once
    let lock_path = {
        let mut p = table_path.to_path_buf();
        let name = p
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        p.set_file_name(format!("{}.lock", name));
        p
    };
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(&lock_path)
        .ok();
    let entry = Arc::new(TableLock {
        mutex: parking_lot::Mutex::new(()),
        file,
    });
    let mut locks = TABLE_WRITE_LOCKS.write();
    locks
        .entry(table_path.to_path_buf())
        .or_insert_with(|| entry.clone());
    entry
}

/// Execute a write operation with per-table locking.
/// Acquires both in-process Mutex and cross-process fs2 flock.
#[inline]
fn with_table_write_lock<F, R>(table_path: &Path, f: F) -> io::Result<R>
where
    F: FnOnce() -> io::Result<R>,
{
    let lock = get_table_lock(table_path);
    // Layer 1: in-process serialization
    let _guard = lock.mutex.lock();
    // Layer 2: cross-process serialization (best-effort, ~0.5μs)
    if let Some(ref file) = lock.file {
        use fs2::FileExt;
        let _ = file.lock_exclusive();
    }
    let result = f();
    // Release cross-process lock immediately
    if let Some(ref file) = lock.file {
        use fs2::FileExt;
        let _ = file.unlock();
    }
    result
}

/// Evict least recently used entries from cache if over limit
fn evict_lru_cache_entries() {
    if STORAGE_CACHE.len() <= MAX_CACHE_ENTRIES {
        return;
    }
    let entries_to_remove = STORAGE_CACHE.len() - MAX_CACHE_ENTRIES + 1;
    let mut access_times: Vec<(PathBuf, u64)> = STORAGE_CACHE
        .iter()
        .map(|entry| {
            let (path, (_, _, access)) = entry.pair();
            (path.clone(), access.load(Ordering::Relaxed))
        })
        .collect();
    access_times.sort_by_key(|(_, t)| *t);
    for (path, _) in access_times.into_iter().take(entries_to_remove) {
        STORAGE_CACHE.remove(&path);
    }
}

// ============================================================================
// Global Index Manager Cache
// ============================================================================
// Key: base_dir path, Value: table_name -> IndexManager
// Lazily loaded from disk catalog on first access per table
static INDEX_CACHE: Lazy<
    RwLock<AHashMap<PathBuf, Arc<parking_lot::Mutex<crate::storage::index::IndexManager>>>>,
> = Lazy::new(|| RwLock::new(AHashMap::with_capacity(32)));

/// Get or create an IndexManager for a table. Returns None if base_dir is not available.
/// The key is base_dir/table_name to uniquely identify each table's index manager.
fn get_index_manager(
    base_dir: &Path,
    table_name: &str,
) -> Arc<parking_lot::Mutex<crate::storage::index::IndexManager>> {
    use crate::storage::index::IndexManager;
    let cache_key = base_dir.join(table_name);

    // Fast path: check read lock
    {
        let cache = INDEX_CACHE.read();
        if let Some(mgr) = cache.get(&cache_key) {
            return mgr.clone();
        }
    }

    // Slow path: create and cache
    let mgr = IndexManager::load(table_name, base_dir)
        .unwrap_or_else(|_| IndexManager::new(table_name, base_dir));
    let mgr = Arc::new(parking_lot::Mutex::new(mgr));

    let mut cache = INDEX_CACHE.write();
    cache.entry(cache_key).or_insert_with(|| mgr.clone());
    mgr
}

/// Invalidate index cache for a specific table
#[allow(dead_code)]
fn invalidate_index_cache(base_dir: &Path, table_name: &str) {
    let cache_key = base_dir.join(table_name);
    INDEX_CACHE.write().remove(&cache_key);
}

/// Invalidate all index cache entries under a directory
fn invalidate_index_cache_dir(dir: &Path) {
    INDEX_CACHE.write().retain(|path, _| !path.starts_with(dir));
}

// ============================================================================
// Global FTS Manager Cache
// ============================================================================
// Key: base_dir path (one FtsManager per database directory)
// FtsManager internally manages one FtsEngine per table
static FTS_MANAGER_CACHE: Lazy<RwLock<AHashMap<PathBuf, Arc<crate::fts::FtsManager>>>> =
    Lazy::new(|| RwLock::new(AHashMap::with_capacity(8)));
static FTS_BACKFILL_TASKS: Lazy<RwLock<AHashMap<(PathBuf, String), std::thread::JoinHandle<()>>>> =
    Lazy::new(|| RwLock::new(AHashMap::with_capacity(8)));

/// Return the FtsManager for a base_dir if one has been registered.
pub fn get_fts_manager(base_dir: &Path) -> Option<Arc<crate::fts::FtsManager>> {
    FTS_MANAGER_CACHE.read().get(base_dir).cloned()
}

/// Register (or replace) the FtsManager for a base_dir.
/// Called by Python `_init_fts()` and by the `CREATE FTS INDEX` DDL handler.
pub fn register_fts_manager(base_dir: &Path, manager: Arc<crate::fts::FtsManager>) {
    FTS_MANAGER_CACHE
        .write()
        .insert(base_dir.to_path_buf(), manager);
}

pub fn register_fts_backfill_task(
    base_dir: &Path,
    table_name: &str,
    handle: std::thread::JoinHandle<()>,
) {
    let previous = FTS_BACKFILL_TASKS
        .write()
        .insert((base_dir.to_path_buf(), table_name.to_string()), handle);
    if let Some(previous) = previous {
        let _ = previous.join();
    }
}

pub fn wait_fts_backfill(base_dir: &Path, table_name: &str) {
    let handle = FTS_BACKFILL_TASKS
        .write()
        .remove(&(base_dir.to_path_buf(), table_name.to_string()));
    if let Some(handle) = handle {
        let _ = handle.join();
    }
}

pub fn wait_fts_backfills_for_dir(base_dir: &Path) {
    let handles: Vec<_> = {
        let mut tasks = FTS_BACKFILL_TASKS.write();
        let keys: Vec<_> = tasks
            .keys()
            .filter(|(dir, _)| dir == base_dir)
            .cloned()
            .collect();
        keys.into_iter()
            .filter_map(|key| tasks.remove(&key))
            .collect()
    };
    for handle in handles {
        let _ = handle.join();
    }
}

/// Remove the FTS manager registered for a base_dir.
///
/// Used when a database directory is recreated in-process, so a later
/// CREATE FTS INDEX builds fresh engines instead of reusing stale in-memory
/// indexes whose files were deleted.
pub fn unregister_fts_manager(base_dir: &Path) {
    wait_fts_backfills_for_dir(base_dir);
    FTS_MANAGER_CACHE.write().remove(base_dir);
}

/// Get or lazily create a FtsManager for a base_dir.
/// Creates the manager with default config; actual per-table engine is created on demand.
fn get_or_create_fts_manager(base_dir: &Path) -> Arc<crate::fts::FtsManager> {
    if let Some(mgr) = FTS_MANAGER_CACHE.read().get(base_dir).cloned() {
        return mgr;
    }
    let fts_dir = base_dir.join("fts_indexes");
    let mgr = Arc::new(crate::fts::FtsManager::new(
        &fts_dir,
        crate::fts::FtsConfig::default(),
    ));
    FTS_MANAGER_CACHE
        .write()
        .entry(base_dir.to_path_buf())
        .or_insert_with(|| mgr.clone());
    mgr
}

/// Derive (base_dir, table_name) from a storage_path like /data/users.apex
fn base_dir_and_table(storage_path: &Path) -> (PathBuf, String) {
    base_dir_and_table_pub(storage_path)
}

/// Public (crate-visible) version for use in submodules.
pub(crate) fn base_dir_and_table_pub(storage_path: &Path) -> (PathBuf, String) {
    let base_dir = storage_path
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let table_name = storage_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "default".to_string());
    (base_dir, table_name)
}

/// Zone Map (min-max index) for a column
/// Used to skip filtering when conditions can't match
#[derive(Clone, Debug)]
struct ZoneMap {
    min_int: Option<i64>,
    max_int: Option<i64>,
    min_float: Option<f64>,
    max_float: Option<f64>,
    has_nulls: bool,
}

impl ZoneMap {
    fn from_int64_array(arr: &Int64Array) -> Self {
        let mut min_val: Option<i64> = None;
        let mut max_val: Option<i64> = None;
        let mut has_nulls = false;

        for i in 0..arr.len() {
            if arr.is_null(i) {
                has_nulls = true;
            } else {
                let v = arr.value(i);
                min_val = Some(min_val.map_or(v, |m| m.min(v)));
                max_val = Some(max_val.map_or(v, |m| m.max(v)));
            }
        }

        Self {
            min_int: min_val,
            max_int: max_val,
            min_float: None,
            max_float: None,
            has_nulls,
        }
    }

    fn from_float64_array(arr: &Float64Array) -> Self {
        let mut min_val: Option<f64> = None;
        let mut max_val: Option<f64> = None;
        let mut has_nulls = false;

        for i in 0..arr.len() {
            if arr.is_null(i) {
                has_nulls = true;
            } else {
                let v = arr.value(i);
                min_val = Some(min_val.map_or(v, |m| m.min(v)));
                max_val = Some(max_val.map_or(v, |m| m.max(v)));
            }
        }

        Self {
            min_int: None,
            max_int: None,
            min_float: min_val,
            max_float: max_val,
            has_nulls,
        }
    }

    /// Check if a comparison can potentially match any rows
    /// Returns true if the filter might match, false if it definitely won't match
    #[inline]
    fn can_match(&self, op: &BinaryOperator, literal: &Value) -> bool {
        match literal {
            Value::Int64(v) => self.can_match_int(*v, op),
            Value::Float64(v) => self.can_match_float(*v, op),
            _ => true, // Can't optimize, assume might match
        }
    }

    #[inline]
    fn can_match_int(&self, v: i64, op: &BinaryOperator) -> bool {
        let (min, max) = match (self.min_int, self.max_int) {
            (Some(min), Some(max)) => (min, max),
            _ => return true, // No stats, assume might match
        };

        match op {
            BinaryOperator::Eq => v >= min && v <= max,
            BinaryOperator::NotEq => true, // Can't optimize !=
            BinaryOperator::Lt => min < v,
            BinaryOperator::Le => min <= v,
            BinaryOperator::Gt => max > v,
            BinaryOperator::Ge => max >= v,
            _ => true,
        }
    }

    #[inline]
    fn can_match_float(&self, v: f64, op: &BinaryOperator) -> bool {
        let (min, max) = match (self.min_float, self.max_float) {
            (Some(min), Some(max)) => (min, max),
            _ => {
                // Try int stats for float comparison
                if let (Some(min), Some(max)) = (self.min_int, self.max_int) {
                    (min as f64, max as f64)
                } else {
                    return true;
                }
            }
        };

        match op {
            BinaryOperator::Eq => v >= min && v <= max,
            BinaryOperator::NotEq => true,
            BinaryOperator::Lt => min < v,
            BinaryOperator::Le => min <= v,
            BinaryOperator::Gt => max > v,
            BinaryOperator::Ge => max >= v,
            _ => true,
        }
    }
}

/// Invalidate the storage cache for a specific path
/// CRITICAL: Must be called before any write operation to release mmap on Windows
#[inline]
pub fn invalidate_storage_cache(path: &Path) {
    STORAGE_CACHE.remove(path);
}

#[inline]
fn storage_effective_modified(path: &Path) -> std::time::SystemTime {
    let modified = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    let mut delta_path = path.to_path_buf();
    let name = delta_path.file_name().unwrap_or_default().to_string_lossy();
    delta_path.set_file_name(format!("{}.delta", name));

    let delta_modified = std::fs::metadata(delta_path)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    if delta_modified > modified {
        delta_modified
    } else {
        modified
    }
}

#[inline]
fn refresh_storage_cache_signature(path: &Path) {
    if let Some(mut entry) = STORAGE_CACHE.get_mut(path) {
        let value = entry.value_mut();
        value.1 = storage_effective_modified(path);
        value.2.store(now_nanos(), Ordering::Relaxed);
    }
}

/// Register an already-open backend in the SQL executor cache.
///
/// Used by storage-level memtable appends: SQL reads should see the same warm
/// backend instead of reopening the mmap-only file and missing pending rows.
#[inline]
pub fn cache_backend_pub(path: &Path, backend: Arc<TableStorageBackend>) {
    if STORAGE_CACHE.len() >= MAX_CACHE_ENTRIES {
        evict_lru_cache_entries();
    }

    let modified = storage_effective_modified(path);

    STORAGE_CACHE.insert(
        path.to_path_buf(),
        (backend, modified, Arc::new(AtomicU64::new(now_nanos()))),
    );
}

/// Invalidate all storage cache entries under a directory
/// CRITICAL: Must be called when closing a client to release all mmaps on Windows
#[inline]
pub fn invalidate_storage_cache_dir(dir: &Path) {
    // Use path directly - avoid expensive canonicalize
    // DashMap::retain is available in dashmap 5.5+
    STORAGE_CACHE.retain(|path, _| !path.starts_with(dir));
    // Also invalidate index caches
    invalidate_index_cache_dir(dir);
}

/// Public wrapper for get_cached_backend (used by Python bindings for fast point lookups)
#[inline]
pub fn get_cached_backend_pub(path: &Path) -> io::Result<Arc<TableStorageBackend>> {
    get_cached_backend(path)
}

/// Get or open a cached storage backend.
/// Read paths merge pending `.delta` rows on demand; they must not compact,
/// because a small transaction append would otherwise become a full-table rewrite.
#[inline]
fn get_cached_backend(path: &Path) -> io::Result<Arc<TableStorageBackend>> {
    let cache_key = path.to_path_buf();
    let delta_path = {
        let mut dp = cache_key.clone();
        let name = dp.file_name().unwrap_or_default().to_string_lossy();
        dp.set_file_name(format!("{}.delta", name));
        dp
    };
    let delta_meta_initial = std::fs::metadata(&delta_path).ok();

    // FASTEST PATH: if backend was validated within last 500ms, skip ALL stat() syscalls.
    // Uses AtomicU64 for last_access so no write-lock is needed on the warm hit path.
    // DashMap::get is lock-free for reads
    if delta_meta_initial.is_none() {
        if let Some(entry) = STORAGE_CACHE.get(&cache_key) {
            let pair = entry.pair();
            let value = pair.1;
            let last_access = &value.2;
            if nanos_elapsed(last_access.load(Ordering::Relaxed)) < 500_000_000 {
                let backend = Arc::clone(&value.0);
                // Refresh last_access atomically — no write lock needed
                last_access.store(now_nanos(), Ordering::Relaxed);
                return Ok(backend);
            }
        }
    }

    // Open the file once — gives us both existence check and metadata without a separate stat()
    let file = std::fs::File::open(path)?;
    let metadata = file.metadata()?;
    let file_len = metadata.len();
    let modified = metadata
        .modified()
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    // Check delta metadata (also gets modified time if exists)
    let effective_modified = match delta_meta_initial {
        Some(delta_meta) => {
            let delta_modified = delta_meta
                .modified()
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            if delta_modified > modified {
                delta_modified
            } else {
                modified
            }
        }
        None => modified,
    };

    // Try read from cache first. The effective mtime includes the append-only
    // delta file so committed txn inserts invalidate stale cached backends.
    if let Some(entry) = STORAGE_CACHE.get(&cache_key) {
        let pair = entry.pair();
        let value = pair.1;
        let cached_time = value.1;
        let last_access = &value.2;
        if cached_time >= effective_modified {
            let backend = Arc::clone(&value.0);
            last_access.store(now_nanos(), Ordering::Relaxed);
            return Ok(backend);
        }
    }

    // Open backend using the already-opened file (saves another File::open).
    // Pending `.delta` data is read lazily by storage-level scan/extract paths.
    let backend = Arc::new(
        match TableStorageBackend::open_with_file(path, file, file_len) {
            Ok(b) => b,
            Err(_) => {
                // Transient footer corruption during concurrent write —
                // return stale cached backend if available.
                if let Some(entry) = STORAGE_CACHE.get(&cache_key) {
                    let pair = entry.pair();
                    pair.1 .2.store(now_nanos(), Ordering::Relaxed);
                    return Ok(Arc::clone(&pair.1 .0));
                }
                // No cached backend — retry after a brief sleep (write should finish quickly)
                std::thread::sleep(std::time::Duration::from_millis(1));
                let file2 = std::fs::File::open(path)?;
                let meta2 = file2.metadata()?;
                TableStorageBackend::open_with_file(path, file2, meta2.len())?
            }
        },
    );

    // LRU eviction before insert - check cache size and evict if needed
    if STORAGE_CACHE.len() >= MAX_CACHE_ENTRIES {
        evict_lru_cache_entries();
    }

    STORAGE_CACHE.insert(
        cache_key,
        (
            Arc::clone(&backend),
            effective_modified,
            Arc::new(AtomicU64::new(now_nanos())),
        ),
    );

    Ok(backend)
}

/// Native Query Executor
///
/// Executes SQL queries directly on storage using Arrow compute kernels.
pub struct ApexExecutor;

/// Query execution result
pub enum ApexResult {
    /// Query returned data rows
    Data(RecordBatch),
    /// Query returned empty result
    Empty(Arc<Schema>),
    /// Query returned a scalar (COUNT, etc.)
    Scalar(i64),
}

impl ApexResult {
    pub fn to_record_batch(self) -> io::Result<RecordBatch> {
        match self {
            ApexResult::Data(batch) => Ok(batch),
            ApexResult::Empty(schema) => Ok(RecordBatch::new_empty(schema)),
            ApexResult::Scalar(val) => {
                let schema = Arc::new(Schema::new(vec![Field::new(
                    "result",
                    ArrowDataType::Int64,
                    false,
                )]));
                let array: ArrayRef = Arc::new(Int64Array::from(vec![val]));
                RecordBatch::try_new(schema, vec![array]).map_err(|e| err_data(e.to_string()))
            }
        }
    }

    pub fn num_rows(&self) -> usize {
        match self {
            ApexResult::Data(batch) => batch.num_rows(),
            ApexResult::Empty(_) => 0,
            ApexResult::Scalar(_) => 1,
        }
    }
}

impl ApexExecutor {
    /// Invalidate the storage cache for a specific path
    pub fn invalidate_cache_for_path(path: &Path) {
        invalidate_storage_cache(path);
    }

    /// Invalidate all storage cache entries under a directory
    pub fn invalidate_cache_for_dir(dir: &Path) {
        invalidate_storage_cache_dir(dir);
    }

    /// Derive temp table path from table name using the thread-local TEMP_DIR.
    fn temp_table_path(table_name: &str) -> PathBuf {
        let dir = TEMP_DIR
            .with(|r| r.borrow().clone())
            .unwrap_or_else(|| PathBuf::from(".apex_tmp"));
        let safe_name: String = table_name
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let truncated = if safe_name.len() > 200 {
            &safe_name[..200]
        } else {
            &safe_name
        };
        dir.join(format!("{}.apex", truncated))
    }

    /// Helper to get column refs from statement's required columns
    #[inline]
    fn get_col_refs(stmt: &SelectStatement) -> Option<Vec<String>> {
        stmt.required_columns().filter(|cols| !cols.is_empty())
    }

    #[inline]
    fn project_batch_by_names(
        batch: &RecordBatch,
        columns: &[String],
    ) -> io::Result<Option<RecordBatch>> {
        let schema = batch.schema();
        let mut fields = Vec::with_capacity(columns.len());
        let mut arrays = Vec::with_capacity(columns.len());
        for column in columns {
            let idx = match schema.index_of(column) {
                Ok(idx) => idx,
                Err(_) => return Ok(None),
            };
            fields.push(schema.field(idx).as_ref().clone());
            arrays.push(batch.column(idx).clone());
        }
        let projected = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
            .map_err(|e| err_data(e.to_string()))?;
        Ok(Some(projected))
    }

    fn view_catalog_path(base_dir: &Path) -> PathBuf {
        base_dir.join(".apex_views.json")
    }

    fn read_view_catalog(base_dir: &Path) -> AHashMap<String, SelectStatement> {
        let path = Self::view_catalog_path(base_dir);
        let content = match std::fs::read_to_string(&path) {
            Ok(content) => content,
            Err(_) => return AHashMap::new(),
        };
        let parsed: ViewCatalog = serde_json::from_str(&content).unwrap_or_default();
        parsed
            .views
            .into_iter()
            .map(|(name, stmt)| (name.to_lowercase(), stmt))
            .collect()
    }

    fn write_view_catalog(
        base_dir: &Path,
        views: &AHashMap<String, SelectStatement>,
    ) -> io::Result<()> {
        let path = Self::view_catalog_path(base_dir);
        let mut sorted = std::collections::BTreeMap::new();
        for (name, stmt) in views {
            sorted.insert(name.clone(), stmt.clone());
        }
        let catalog = ViewCatalog {
            views: sorted.into_iter().collect(),
        };
        let json = serde_json::to_string(&catalog)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        std::fs::write(path, json)
    }

    fn persist_view(base_dir: &Path, name: &str, stmt: &SelectStatement) -> io::Result<()> {
        let mut catalog = Self::read_view_catalog(base_dir);
        catalog.insert(name.to_lowercase(), stmt.clone());
        Self::write_view_catalog(base_dir, &catalog)
    }

    fn remove_persisted_view(base_dir: &Path, name: &str) -> io::Result<()> {
        let mut catalog = Self::read_view_catalog(base_dir);
        catalog.remove(&name.to_lowercase());
        if catalog.is_empty() {
            let path = Self::view_catalog_path(base_dir);
            let _ = std::fs::remove_file(path);
            return Ok(());
        }
        Self::write_view_catalog(base_dir, &catalog)
    }

    fn load_all_views(
        base_dir: &Path,
        session_views: &AHashMap<String, SelectStatement>,
    ) -> AHashMap<String, SelectStatement> {
        let mut merged = Self::read_view_catalog(base_dir);
        for (name, stmt) in session_views {
            merged.insert(name.to_lowercase(), stmt.clone());
        }
        merged
    }

    fn rewrite_statement_views(
        stmt: SqlStatement,
        views: &AHashMap<String, SelectStatement>,
    ) -> SqlStatement {
        match stmt {
            SqlStatement::Select(select) => {
                SqlStatement::Select(Self::rewrite_select_views(select, views))
            }
            SqlStatement::Union(mut union) => {
                union.left = Box::new(Self::rewrite_statement_views(*union.left, views));
                union.right = Box::new(Self::rewrite_statement_views(*union.right, views));
                SqlStatement::Union(union)
            }
            SqlStatement::Cte {
                name,
                column_aliases,
                body,
                main,
                recursive,
            } => SqlStatement::Cte {
                name,
                column_aliases,
                body: Box::new(Self::rewrite_statement_views(*body, views)),
                main: Box::new(Self::rewrite_statement_views(*main, views)),
                recursive,
            },
            SqlStatement::InsertSelect {
                table,
                columns,
                query,
            } => SqlStatement::InsertSelect {
                table,
                columns,
                query: Box::new(Self::rewrite_statement_views(*query, views)),
            },
            SqlStatement::CreateTableAs {
                table,
                query,
                if_not_exists,
                temp,
            } => SqlStatement::CreateTableAs {
                table,
                query: Box::new(Self::rewrite_statement_views(*query, views)),
                if_not_exists,
                temp,
            },
            SqlStatement::Explain { stmt, analyze } => SqlStatement::Explain {
                stmt: Box::new(Self::rewrite_statement_views(*stmt, views)),
                analyze,
            },
            SqlStatement::CreateView { name, stmt } => SqlStatement::CreateView {
                name,
                stmt: Self::rewrite_select_views(stmt, views),
            },
            other => other,
        }
    }

    fn rewrite_from_item_views(
        from: FromItem,
        views: &AHashMap<String, SelectStatement>,
    ) -> FromItem {
        match from {
            FromItem::Table { table, alias } => {
                let table_name = table.trim_matches('"').to_lowercase();
                if let Some(view_stmt) = views.get(&table_name) {
                    let alias_name = alias.clone().unwrap_or_else(|| table.clone());
                    FromItem::Subquery {
                        stmt: Box::new(SqlStatement::Select(Self::rewrite_select_views(
                            view_stmt.clone(),
                            views,
                        ))),
                        alias: alias_name,
                    }
                } else {
                    FromItem::Table { table, alias }
                }
            }
            FromItem::Subquery { stmt, alias } => FromItem::Subquery {
                stmt: Box::new(Self::rewrite_statement_views(*stmt, views)),
                alias,
            },
            other => other,
        }
    }

    fn rewrite_expr_views(expr: SqlExpr, views: &AHashMap<String, SelectStatement>) -> SqlExpr {
        match expr {
            SqlExpr::BinaryOp { left, op, right } => SqlExpr::BinaryOp {
                left: Box::new(Self::rewrite_expr_views(*left, views)),
                op,
                right: Box::new(Self::rewrite_expr_views(*right, views)),
            },
            SqlExpr::UnaryOp { op, expr } => SqlExpr::UnaryOp {
                op,
                expr: Box::new(Self::rewrite_expr_views(*expr, views)),
            },
            SqlExpr::InSubquery {
                column,
                stmt,
                negated,
            } => SqlExpr::InSubquery {
                column,
                stmt: Box::new(Self::rewrite_select_views(*stmt, views)),
                negated,
            },
            SqlExpr::ExistsSubquery { stmt } => SqlExpr::ExistsSubquery {
                stmt: Box::new(Self::rewrite_select_views(*stmt, views)),
            },
            SqlExpr::ScalarSubquery { stmt } => SqlExpr::ScalarSubquery {
                stmt: Box::new(Self::rewrite_select_views(*stmt, views)),
            },
            SqlExpr::Case {
                when_then,
                else_expr,
            } => SqlExpr::Case {
                when_then: when_then
                    .into_iter()
                    .map(|(when, then)| {
                        (
                            Self::rewrite_expr_views(when, views),
                            Self::rewrite_expr_views(then, views),
                        )
                    })
                    .collect(),
                else_expr: else_expr.map(|expr| Box::new(Self::rewrite_expr_views(*expr, views))),
            },
            SqlExpr::Between {
                column,
                low,
                high,
                negated,
            } => SqlExpr::Between {
                column,
                low: Box::new(Self::rewrite_expr_views(*low, views)),
                high: Box::new(Self::rewrite_expr_views(*high, views)),
                negated,
            },
            SqlExpr::Function { name, args } => SqlExpr::Function {
                name,
                args: args
                    .into_iter()
                    .map(|arg| Self::rewrite_expr_views(arg, views))
                    .collect(),
            },
            SqlExpr::Cast { expr, data_type } => SqlExpr::Cast {
                expr: Box::new(Self::rewrite_expr_views(*expr, views)),
                data_type,
            },
            SqlExpr::Paren(expr) => {
                SqlExpr::Paren(Box::new(Self::rewrite_expr_views(*expr, views)))
            }
            SqlExpr::ArrayIndex { array, index } => SqlExpr::ArrayIndex {
                array: Box::new(Self::rewrite_expr_views(*array, views)),
                index: Box::new(Self::rewrite_expr_views(*index, views)),
            },
            SqlExpr::ExplodeRename { inner, names } => SqlExpr::ExplodeRename {
                inner: Box::new(Self::rewrite_expr_views(*inner, views)),
                names,
            },
            other => other,
        }
    }

    /// Execute a SQL query on storage (single table)
    pub fn execute(sql: &str, storage_path: &Path) -> io::Result<ApexResult> {
        let base_dir = storage_path.parent().unwrap_or(storage_path);
        Self::execute_with_base_dir(sql, base_dir, storage_path)
    }

    /// Execute a SQL query with multi-table support (for JOINs)
    ///
    /// Uses `QuerySignature::classify()` for single-point pre-parse dispatch.
    /// Fast paths (COUNT*, point lookup) are executed here without SQL parsing.
    /// Everything else falls through to the SQL parser + executor pipeline.
    pub fn execute_with_base_dir(
        sql: &str,
        base_dir: &Path,
        default_table_path: &Path,
    ) -> io::Result<ApexResult> {
        use crate::query::query_signature::{self, QuerySignature};

        let sig = query_signature::classify(sql);

        // ── Pre-parse fast paths (bypass SQL parser entirely) ──
        match &sig {
            QuerySignature::CountStar { table } => {
                let table_path = Self::resolve_table_path(table, base_dir, default_table_path);
                if let Ok(backend) = get_cached_backend(&table_path) {
                    let count = backend.active_row_count() as i64;
                    let schema = Arc::new(Schema::new(vec![Field::new(
                        "COUNT(*)",
                        ArrowDataType::Int64,
                        false,
                    )]));
                    let array: ArrayRef = Arc::new(Int64Array::from(vec![count]));
                    let batch = RecordBatch::try_new(schema, vec![array])
                        .map_err(|e| err_data(e.to_string()))?;
                    return Ok(ApexResult::Data(batch));
                }
                // Fall through to full parse if backend open fails
            }
            QuerySignature::PointLookup { id, ref table } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    if backend.has_pending_deltas() || backend.has_delta() {
                        // Fall through so the general executor applies DeltaMerger overlays
                        // and append-only transaction delta rows.
                    } else {
                        if backend.storage.is_v4_format()
                            && !backend.storage.has_v4_in_memory_data()
                        {
                            if let Ok(Some(vals)) = backend.storage.retrieve_rcix(*id) {
                                use crate::data::Value as V;
                                let mut fields = Vec::with_capacity(vals.len());
                                let mut arrays: Vec<ArrayRef> = Vec::with_capacity(vals.len());
                                for (col_name, val) in &vals {
                                    let nullable = matches!(val, V::Null);
                                    let (dt, arr): (ArrowDataType, ArrayRef) = match val {
                                        V::Int64(v) => (
                                            ArrowDataType::Int64,
                                            Arc::new(Int64Array::from(vec![*v])),
                                        ),
                                        V::Float64(v) => (
                                            ArrowDataType::Float64,
                                            Arc::new(arrow::array::Float64Array::from(vec![*v])),
                                        ),
                                        V::String(s) => (
                                            ArrowDataType::Utf8,
                                            Arc::new(arrow::array::StringArray::from(vec![
                                                s.as_str()
                                            ])),
                                        ),
                                        V::Bool(b) => (
                                            ArrowDataType::Boolean,
                                            Arc::new(arrow::array::BooleanArray::from(vec![*b])),
                                        ),
                                        _ => (
                                            ArrowDataType::Utf8,
                                            Arc::new(arrow::array::StringArray::from(vec![
                                                None as Option<&str>,
                                            ])),
                                        ),
                                    };
                                    fields.push(Field::new(col_name, dt, nullable));
                                    arrays.push(arr);
                                }
                                let schema = Arc::new(Schema::new(fields));
                                if let Ok(batch) = RecordBatch::try_new(schema, arrays) {
                                    return Ok(ApexResult::Data(batch));
                                }
                            }
                        }
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::ProjectedPointLookup {
                id,
                ref table,
                columns,
            } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    if backend.has_pending_deltas() || backend.has_delta() {
                        // Fall through so the general executor applies DeltaMerger overlays
                        // and append-only transaction delta rows.
                    } else {
                        if let Ok(Some(batch)) = backend.read_row_by_id_to_arrow(*id) {
                            if let Some(projected) = Self::project_batch_by_names(&batch, columns)?
                            {
                                return Ok(ApexResult::Data(projected));
                            }
                        }
                        let col_refs: Vec<&str> = columns.iter().map(String::as_str).collect();
                        if let Ok(empty) =
                            backend.read_columns_to_arrow(Some(col_refs.as_slice()), 0, Some(0))
                        {
                            return Ok(ApexResult::Data(empty));
                        }
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::IdBatchLookup { ids, ref table } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    let sorted_ids = sort_and_dedupe_ids(ids);
                    if let Ok(batch) = backend.read_rows_by_ids_to_arrow(&sorted_ids) {
                        if batch.num_rows() > 0 {
                            return Ok(ApexResult::Data(batch));
                        }
                        if let Ok(empty) = backend.read_columns_to_arrow(None, 0, Some(0)) {
                            return Ok(ApexResult::Data(empty));
                        }
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::ProjectedIdBatchLookup {
                ids,
                ref table,
                columns,
            } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    let sorted_ids = sort_and_dedupe_ids(ids);
                    if let Ok(batch) = backend.read_rows_by_ids_to_arrow(&sorted_ids) {
                        if batch.num_rows() > 0 {
                            if let Some(projected) = Self::project_batch_by_names(&batch, columns)?
                            {
                                return Ok(ApexResult::Data(projected));
                            }
                        }
                        let col_refs: Vec<&str> = columns.iter().map(String::as_str).collect();
                        if let Ok(empty) =
                            backend.read_columns_to_arrow(Some(col_refs.as_slice()), 0, Some(0))
                        {
                            return Ok(ApexResult::Data(empty));
                        }
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::FullScan { ref table } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    if let Ok(batch) = backend.read_columns_to_arrow(None, 0, None) {
                        return Ok(ApexResult::Data(batch));
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::ProjectedFullScan { ref table, columns } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    let col_refs: Vec<&str> = columns.iter().map(String::as_str).collect();
                    if let Ok(batch) =
                        backend.read_columns_to_arrow(Some(col_refs.as_slice()), 0, None)
                    {
                        return Ok(ApexResult::Data(batch));
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::SimpleScanLimit {
                limit,
                offset,
                ref table,
            } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    if backend.pending_v4_in_memory_rows() == 0 {
                        if *offset == 0 {
                            if let Ok(batch) = backend.read_columns_to_arrow(None, 0, Some(*limit))
                            {
                                return Ok(ApexResult::Data(batch));
                            }
                        } else if !backend.has_pending_deltas()
                            && !backend.has_delta()
                            && backend.active_row_count() == backend.row_count()
                        {
                            let end = (*offset)
                                .saturating_add(*limit)
                                .min(backend.row_count() as usize);
                            let indices: Vec<usize> = (*offset..end).collect();
                            if let Ok(batch) =
                                backend.read_columns_by_indices_to_arrow(&indices, None)
                            {
                                return Ok(ApexResult::Data(batch));
                            }
                        }
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::ProjectedScanLimit {
                limit,
                offset,
                ref table,
                columns,
            } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    if backend.pending_v4_in_memory_rows() == 0 {
                        let col_refs: Vec<&str> = columns.iter().map(String::as_str).collect();
                        if let Ok(batch) = backend.read_columns_to_arrow(
                            Some(col_refs.as_slice()),
                            *offset,
                            Some(*limit),
                        ) {
                            return Ok(ApexResult::Data(batch));
                        }
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::NumericRangeFilterLimit {
                ref table,
                column,
                low,
                high,
                limit,
                offset,
            } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    if !backend.has_pending_deltas()
                        && !backend.has_delta()
                        && backend.pending_v4_in_memory_rows() == 0
                    {
                        let needed = (*offset).saturating_add(*limit);
                        if let Ok(Some(indices)) =
                            backend.scan_numeric_range_mmap(column, *low, *high, Some(needed))
                        {
                            let final_indices: Vec<usize> =
                                indices.into_iter().skip(*offset).take(*limit).collect();
                            let batch = if final_indices.is_empty() {
                                backend.read_columns_to_arrow(None, 0, Some(0))
                            } else {
                                backend.read_columns_by_indices_to_arrow(&final_indices, None)
                            };
                            if let Ok(batch) = batch {
                                return Ok(ApexResult::Data(batch));
                            }
                        }
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::ProjectedNumericRangeFilterLimit {
                ref table,
                columns,
                column,
                low,
                high,
                limit,
                offset,
            } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    if !backend.has_pending_deltas()
                        && !backend.has_delta()
                        && backend.pending_v4_in_memory_rows() == 0
                    {
                        let needed = (*offset).saturating_add(*limit);
                        let col_refs: Vec<&str> = columns.iter().map(String::as_str).collect();
                        if let Ok(Some(indices)) =
                            backend.scan_numeric_range_mmap(column, *low, *high, Some(needed))
                        {
                            let final_indices: Vec<usize> =
                                indices.into_iter().skip(*offset).take(*limit).collect();
                            let batch = if final_indices.is_empty() {
                                backend.read_columns_to_arrow(Some(col_refs.as_slice()), 0, Some(0))
                            } else {
                                backend.read_columns_by_indices_to_arrow(
                                    &final_indices,
                                    Some(col_refs.as_slice()),
                                )
                            };
                            if let Ok(batch) = batch {
                                return Ok(ApexResult::Data(batch));
                            }
                        }
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::StringEqualityFilter {
                ref table,
                column,
                value,
            } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    if backend.pending_v4_in_memory_rows() == 0 && !backend.has_pending_deltas() {
                        if let Ok(batch) =
                            backend.read_columns_filtered_string_to_arrow(None, column, value, true)
                        {
                            return Ok(ApexResult::Data(batch));
                        }
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::NumericEqualityFilter {
                ref table,
                column,
                value,
            } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    if backend.pending_v4_in_memory_rows() == 0
                        && !backend.has_pending_deltas()
                        && !backend.has_delta()
                        && backend.is_mmap_only()
                    {
                        let low = *value as f64;
                        if let Ok(Some(indices)) =
                            backend.scan_numeric_range_mmap(column, low, low, None)
                        {
                            if indices.is_empty() {
                                let schema = backend.read_columns_to_arrow(None, 0, Some(0))?;
                                return Ok(ApexResult::Empty(schema.schema().clone().into()));
                            }
                            if let Ok(batch) =
                                backend.read_columns_by_indices_to_arrow(&indices, None)
                            {
                                return Ok(ApexResult::Data(batch));
                            }
                        }
                    }
                }
            }
            QuerySignature::NumericInFilter {
                ref table,
                column,
                values,
            } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    if backend.pending_v4_in_memory_rows() == 0
                        && !backend.has_pending_deltas()
                        && !backend.has_delta()
                        && backend.is_mmap_only()
                    {
                        if let Ok(Some(indices)) =
                            backend.scan_numeric_in_mmap(column, values, None)
                        {
                            if indices.is_empty() {
                                let schema = backend.read_columns_to_arrow(None, 0, Some(0))?;
                                return Ok(ApexResult::Empty(schema.schema().clone().into()));
                            }
                            if let Ok(batch) =
                                backend.read_columns_by_indices_to_arrow(&indices, None)
                            {
                                return Ok(ApexResult::Data(batch));
                            }
                        }
                    }
                }
            }
            QuerySignature::StringEqualityFilterLimit {
                ref table,
                column,
                value,
                limit,
                offset,
            } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    if backend.pending_v4_in_memory_rows() == 0 && !backend.has_pending_deltas() {
                        let batch = if backend.has_delta() {
                            backend
                                .read_columns_filtered_string_to_arrow(None, column, value, true)
                                .map(|full| {
                                    let offset = (*offset).min(full.num_rows());
                                    let len = (*limit).min(full.num_rows().saturating_sub(offset));
                                    full.slice(offset, len)
                                })
                        } else {
                            backend.read_columns_filtered_string_with_limit_to_arrow(
                                None, column, value, true, *limit, *offset,
                            )
                        };
                        if let Ok(batch) = batch {
                            return Ok(ApexResult::Data(batch));
                        }
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::ProjectedStringEqualityFilter {
                ref table,
                columns,
                column,
                value,
            } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    if backend.pending_v4_in_memory_rows() == 0 && !backend.has_pending_deltas() {
                        let col_refs: Vec<&str> = columns.iter().map(String::as_str).collect();
                        if let Ok(batch) = backend.read_columns_filtered_string_to_arrow(
                            Some(col_refs.as_slice()),
                            column,
                            value,
                            true,
                        ) {
                            return Ok(ApexResult::Data(batch));
                        }
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::ProjectedStringEqualityFilterLimit {
                ref table,
                columns,
                column,
                value,
                limit,
                offset,
            } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    if backend.pending_v4_in_memory_rows() == 0 && !backend.has_pending_deltas() {
                        let col_refs: Vec<&str> = columns.iter().map(String::as_str).collect();
                        let batch = if backend.has_delta() {
                            backend
                                .read_columns_filtered_string_to_arrow(
                                    Some(col_refs.as_slice()),
                                    column,
                                    value,
                                    true,
                                )
                                .map(|full| {
                                    let offset = (*offset).min(full.num_rows());
                                    let len = (*limit).min(full.num_rows().saturating_sub(offset));
                                    full.slice(offset, len)
                                })
                        } else {
                            backend.read_columns_filtered_string_with_limit_to_arrow(
                                Some(col_refs.as_slice()),
                                column,
                                value,
                                true,
                                *limit,
                                *offset,
                            )
                        };
                        if let Ok(batch) = batch {
                            return Ok(ApexResult::Data(batch));
                        }
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::LikeFilter {
                ref table,
                column,
                pattern,
            } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    if backend.pending_v4_in_memory_rows() == 0 {
                        if let Ok(Some(batch)) =
                            backend.scan_like_and_extract_mmap(column, pattern, None)
                        {
                            return Ok(ApexResult::Data(batch));
                        }
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::FilteredStringAgg {
                ref table,
                ref filter_column,
                ref filter_value,
            } => {
                let table_path = table
                    .as_ref()
                    .map(|tname| Self::resolve_table_path(tname, base_dir, default_table_path))
                    .unwrap_or_else(|| default_table_path.to_path_buf());
                if let Ok(backend) = get_cached_backend(&table_path) {
                    if !backend.has_pending_deltas()
                        && !backend.has_delta()
                        && backend.pending_v4_in_memory_rows() == 0
                    {
                        // Fall through to full parse — the executor select() has
                        // try_fast_filtered_string_agg which handles this optimally
                    }
                }
                // Fall through to full parse if fast path unavailable
            }
            QuerySignature::DmlWrite => {
                // PRE-PARSE FAST PATH: simple DELETE / UPDATE with single numeric WHERE.
                // Bypasses SqlParser::parse_multi (~200µs) for the common OLTP shape.
                let s = sql.trim().trim_end_matches(';');
                if s.len() <= 300 && !s.is_empty() {
                    let first = s.as_bytes()[0];
                    if first == b'D' || first == b'd' {
                        // DELETE FROM <table> WHERE <col> <op> <num>
                        let su = s.to_ascii_uppercase();
                        if su.starts_with("DELETE FROM ") {
                            let after_df = &s["DELETE FROM ".len()..];
                            let after_df_u = &su["DELETE FROM ".len()..];
                            if let Some(where_pos) = after_df_u.find(" WHERE ") {
                                let table_raw = after_df[..where_pos].trim();
                                let after_where = after_df[where_pos + 7..].trim();
                                let after_where_u = after_where.to_ascii_uppercase();
                                if !after_where_u.contains(" AND ")
                                    && !after_where_u.contains(" OR ")
                                    && !after_where_u.contains("NOT ")
                                    && !after_where_u.contains(" IN ")
                                    && !table_raw.contains(' ')
                                {
                                    if let Some(expr) =
                                        Self::try_parse_delete_numeric_where(after_where)
                                    {
                                        let table_path = Self::resolve_table_path(
                                            table_raw,
                                            base_dir,
                                            default_table_path,
                                        );
                                        if table_path.exists() {
                                            return with_table_write_lock(&table_path, || {
                                                Self::execute_delete(&table_path, Some(&expr))
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    } else if first == b'U' || first == b'u' {
                        // UPDATE <table> SET <col> = <num> WHERE <col> <op> <num>
                        let su = s.to_ascii_uppercase();
                        if su.starts_with("UPDATE ") {
                            let after_up = &s["UPDATE ".len()..];
                            let after_up_u = &su["UPDATE ".len()..];
                            if let Some(set_pos) = after_up_u.find(" SET ") {
                                let table_raw = after_up[..set_pos].trim();
                                if !table_raw.is_empty() && !table_raw.contains(' ') {
                                    let after_set = &after_up[set_pos + 5..];
                                    let after_set_u = &after_up_u[set_pos + 5..];
                                    if let Some(where_pos) = after_set_u.find(" WHERE ") {
                                        let set_part = after_set[..where_pos].trim();
                                        let where_part = after_set[where_pos + 7..].trim();
                                        let where_part_u = after_set_u[where_pos + 7..]
                                            .trim()
                                            .to_ascii_uppercase();
                                        if !where_part_u.contains(" AND ")
                                            && !where_part_u.contains(" OR ")
                                            && !where_part_u.contains("NOT ")
                                            && !where_part_u.contains(" IN ")
                                        {
                                            if let Some(eq_pos) = set_part.find('=') {
                                                let set_col =
                                                    set_part[..eq_pos].trim().trim_matches('"');
                                                let set_val_str = set_part[eq_pos + 1..].trim();
                                                if !set_col.is_empty()
                                                    && !set_col.contains(' ')
                                                    && set_col != "_id"
                                                {
                                                    let set_val: Option<Value> = if set_val_str
                                                        .contains('.')
                                                        || set_val_str.contains('e')
                                                        || set_val_str.contains('E')
                                                    {
                                                        set_val_str
                                                            .parse::<f64>()
                                                            .ok()
                                                            .map(Value::Float64)
                                                    } else {
                                                        set_val_str
                                                            .parse::<i64>()
                                                            .ok()
                                                            .map(Value::Int64)
                                                    };
                                                    if let Some(ref sv) = set_val {
                                                        if let Some(expr) =
                                                            Self::try_parse_delete_numeric_where(
                                                                where_part,
                                                            )
                                                        {
                                                            let table_path =
                                                                Self::resolve_table_path(
                                                                    table_raw,
                                                                    base_dir,
                                                                    default_table_path,
                                                                );
                                                            if table_path.exists() {
                                                                let assignments = vec![(
                                                                    set_col.to_string(),
                                                                    SqlExpr::Literal(sv.clone()),
                                                                )];
                                                                return with_table_write_lock(
                                                                    &table_path,
                                                                    || {
                                                                        Self::execute_update(
                                                                            &table_path,
                                                                            &assignments,
                                                                            Some(&expr),
                                                                        )
                                                                    },
                                                                );
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Fall through to full parse for complex DML or unsupported shorthand.
            }
            QuerySignature::MultiStatement => {
                // Multi-statement: parse all at once and execute
                let stmts = SqlParser::parse_multi(sql).map_err(|e| err_input(e.to_string()))?;
                return Self::execute_parsed_multi_statements(stmts, base_dir, default_table_path);
            }
            _ => {
                // All other signatures fall through to full parse below
            }
        }

        // ── Full parse + execute pipeline ──
        // Parse as multi-statement unconditionally to avoid relying on string heuristics.
        let stmts = {
            // Fast path: check parse cache first (read lock, no allocation on hit)
            let cached = SQL_PARSE_CACHE.read().get(sql).cloned();
            if let Some(stmts) = cached {
                stmts
            } else {
                let stmts = SqlParser::parse_multi(sql).map_err(|e| err_input(e.to_string()))?;
                // Only cache read-only statements (SELECT) to avoid stale DDL/DML
                let is_select_only = stmts
                    .iter()
                    .all(|s| matches!(s, SqlStatement::Select(_) | SqlStatement::Union(_)));
                if is_select_only {
                    let mut cache = SQL_PARSE_CACHE.write();
                    if cache.len() < 1024 {
                        cache.insert(sql.to_string(), stmts.clone());
                    }
                }
                stmts
            }
        };

        if stmts.len() > 1
            || matches!(
                stmts.first(),
                Some(SqlStatement::CreateView { .. } | SqlStatement::DropView { .. })
            )
        {
            return Self::execute_parsed_multi_statements(stmts, base_dir, default_table_path);
        }

        let stmt = stmts
            .into_iter()
            .next()
            .ok_or_else(|| err_input("No statement to execute"))?;

        Self::execute_parsed_multi(stmt, base_dir, default_table_path)
    }

    /// Execute a parsed SQL statement (single table)
    pub fn execute_parsed(stmt: SqlStatement, storage_path: &Path) -> io::Result<ApexResult> {
        match stmt {
            SqlStatement::Select(select) => Self::execute_select(select, storage_path),
            SqlStatement::Union(union) => Self::execute_union(union, storage_path, storage_path),
            SqlStatement::Insert {
                values, columns, ..
            } => with_table_write_lock(storage_path, || {
                Self::execute_insert(storage_path, columns.as_deref(), &values)
            }),
            SqlStatement::InsertOnConflict {
                values,
                columns,
                conflict_columns,
                do_update,
                ..
            } => with_table_write_lock(storage_path, || {
                Self::execute_insert_on_conflict(
                    storage_path,
                    columns.as_deref(),
                    &values,
                    &conflict_columns,
                    do_update.as_deref(),
                )
            }),
            SqlStatement::InsertSelect { columns, query, .. } => {
                with_table_write_lock(storage_path, || {
                    Self::execute_insert_select(
                        storage_path,
                        columns.as_deref(),
                        *query,
                        storage_path.parent().unwrap_or(Path::new(".")),
                        storage_path,
                    )
                })
            }
            SqlStatement::Delete { where_clause, .. } => {
                with_table_write_lock(storage_path, || {
                    Self::execute_delete(storage_path, where_clause.as_ref())
                })
            }
            SqlStatement::Update {
                assignments,
                where_clause,
                ..
            } => with_table_write_lock(storage_path, || {
                Self::execute_update(storage_path, &assignments, where_clause.as_ref())
            }),
            SqlStatement::TruncateTable { .. } => {
                with_table_write_lock(storage_path, || Self::execute_truncate(storage_path))
            }
            SqlStatement::Explain { stmt, analyze } => Self::execute_explain(
                *stmt,
                analyze,
                storage_path.parent().unwrap_or(Path::new(".")),
                storage_path,
            ),
            SqlStatement::Cte {
                name,
                column_aliases,
                body,
                main,
                recursive,
            } => Self::execute_cte(
                &name,
                &column_aliases,
                *body,
                *main,
                recursive,
                storage_path.parent().unwrap_or(Path::new(".")),
                storage_path,
            ),
            SqlStatement::BeginTransaction { read_only } => Self::execute_begin(read_only),
            SqlStatement::Commit => Err(err_input(
                "COMMIT requires txn_id context - use execute_commit_txn()",
            )),
            SqlStatement::Rollback => Err(err_input(
                "ROLLBACK requires txn_id context - use execute_rollback_txn()",
            )),
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "DDL statements require base_dir context - use execute_with_base_dir()",
            )),
        }
    }

    /// Execute a parsed SQL statement with multi-table support
    pub fn execute_parsed_multi(
        stmt: SqlStatement,
        base_dir: &Path,
        default_table_path: &Path,
    ) -> io::Result<ApexResult> {
        let persisted_views = Self::read_view_catalog(base_dir);
        let stmt = if persisted_views.is_empty() {
            stmt
        } else {
            Self::rewrite_statement_views(stmt, &persisted_views)
        };
        match stmt {
            SqlStatement::Select(select) => {
                if select.joins.is_empty() {
                    // Resolve the actual table path from FROM clause for non-join queries
                    let actual_path =
                        Self::resolve_from_table_path(&select, base_dir, default_table_path);
                    Self::execute_select_with_base_dir(
                        select,
                        &actual_path,
                        base_dir,
                        default_table_path,
                    )
                } else {
                    Self::execute_select_with_joins(select, base_dir, default_table_path)
                }
            }
            SqlStatement::Union(union) => Self::execute_union(union, base_dir, default_table_path),
            // DDL Statements — acquire per-table write lock for concurrency safety
            SqlStatement::CreateTable {
                table,
                columns,
                if_not_exists,
                temp,
            } => {
                let table_path = if temp {
                    Self::temp_table_path(&table)
                } else {
                    Self::resolve_table_path(&table, base_dir, default_table_path)
                };
                with_table_write_lock(&table_path, || {
                    Self::execute_create_table(&table_path, &table, &columns, if_not_exists)
                })
            }
            SqlStatement::DropTable { table, if_exists } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_drop_table(&table_path, &table, if_exists)
                })
            }
            SqlStatement::AlterTable { table, operation } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_alter_table(&table_path, &table, &operation)
                })
            }
            SqlStatement::TruncateTable { table } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || Self::execute_truncate(&table_path))
            }
            // DML Statements — acquire per-table write lock for concurrency safety
            SqlStatement::Insert {
                table,
                columns,
                values,
            } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_insert(&table_path, columns.as_deref(), &values)
                })
            }
            SqlStatement::InsertOnConflict {
                table,
                columns,
                values,
                conflict_columns,
                do_update,
            } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_insert_on_conflict(
                        &table_path,
                        columns.as_deref(),
                        &values,
                        &conflict_columns,
                        do_update.as_deref(),
                    )
                })
            }
            SqlStatement::AnalyzeTable { table } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                Self::execute_analyze(&table_path, &table)
            }
            SqlStatement::CopyToParquet { table, file_path } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                Self::execute_copy_to_parquet(&table_path, &table, &file_path)
            }
            SqlStatement::CopyExport {
                table,
                file_path,
                format,
                options,
            } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                Self::execute_copy_export(&table_path, &table, &file_path, &format, &options)
            }
            SqlStatement::CopyFromParquet { table, file_path } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_copy_from_parquet(
                        &table_path,
                        &table,
                        &file_path,
                        base_dir.as_ref(),
                        default_table_path.as_ref(),
                    )
                })
            }
            SqlStatement::InsertSelect {
                table,
                columns,
                query,
            } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_insert_select(
                        &table_path,
                        columns.as_deref(),
                        *query,
                        base_dir,
                        default_table_path,
                    )
                })
            }
            SqlStatement::CreateTableAs {
                table,
                query,
                if_not_exists,
                temp,
            } => {
                let table_path = if temp {
                    Self::temp_table_path(&table)
                } else {
                    Self::resolve_table_path(&table, base_dir, default_table_path)
                };
                with_table_write_lock(&table_path, || {
                    Self::execute_create_table_as(
                        base_dir,
                        default_table_path,
                        &table,
                        *query,
                        if_not_exists,
                    )
                })
            }
            SqlStatement::Delete {
                table,
                where_clause,
            } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_delete(&table_path, where_clause.as_ref())
                })
            }
            SqlStatement::Update {
                table,
                assignments,
                where_clause,
            } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_update(&table_path, &assignments, where_clause.as_ref())
                })
            }
            // Index Statements
            SqlStatement::CreateIndex {
                name,
                table,
                columns,
                unique,
                index_type,
                if_not_exists,
            } => Self::execute_create_index(
                base_dir,
                default_table_path,
                &name,
                &table,
                &columns,
                unique,
                index_type.as_deref(),
                if_not_exists,
            ),
            SqlStatement::DropIndex {
                name,
                table,
                if_exists,
            } => Self::execute_drop_index(base_dir, &name, &table, if_exists),
            // EXPLAIN
            SqlStatement::Explain { stmt, analyze } => {
                Self::execute_explain(*stmt, analyze, base_dir, default_table_path)
            }
            // CTE
            SqlStatement::Cte {
                name,
                column_aliases,
                body,
                main,
                recursive,
            } => Self::execute_cte(
                &name,
                &column_aliases,
                *body,
                *main,
                recursive,
                base_dir,
                default_table_path,
            ),
            // Transaction Statements
            SqlStatement::BeginTransaction { read_only } => Self::execute_begin(read_only),
            SqlStatement::Commit => Err(err_input(
                "COMMIT requires txn_id context - use execute_commit_txn()",
            )),
            SqlStatement::Rollback => Err(err_input(
                "ROLLBACK requires txn_id context - use execute_rollback_txn()",
            )),
            SqlStatement::Reindex { table } => {
                Self::execute_reindex(base_dir, default_table_path, &table)
            }
            SqlStatement::Pragma { name, arg } => {
                Self::execute_pragma(base_dir, default_table_path, &name, arg.as_deref())
            }
            // FTS DDL Statements
            SqlStatement::CreateFtsIndex {
                table,
                fields,
                lazy_load,
                cache_size,
            } => Self::execute_create_fts_index(
                base_dir,
                &table,
                fields.as_deref(),
                lazy_load,
                cache_size,
            ),
            SqlStatement::DropFtsIndex { table } => Self::execute_drop_fts_index(base_dir, &table),
            SqlStatement::AlterFtsIndexDisable { table } => {
                Self::execute_alter_fts_index_disable(base_dir, &table)
            }
            SqlStatement::AlterFtsIndexEnable { table } => {
                Self::execute_alter_fts_index_enable(base_dir, &table)
            }
            SqlStatement::ShowFtsIndexes => Self::execute_show_fts_indexes(base_dir),
            SqlStatement::SetVariable { name, value } => {
                set_session_variable(&name, value);
                Ok(ApexResult::Scalar(0))
            }
            SqlStatement::ResetVariable { name } => {
                reset_session_variable(&name);
                Ok(ApexResult::Scalar(0))
            }
            SqlStatement::CopyImport {
                table,
                file_path,
                format,
                options,
            } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                with_table_write_lock(&table_path, || {
                    Self::execute_copy_import(
                        &table_path,
                        &table,
                        &file_path,
                        &format,
                        &options,
                        base_dir,
                        default_table_path,
                    )
                })
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Statement type not supported",
            )),
        }
    }

    /// Execute multiple SQL statements separated by semicolons.
    /// Currently used for temporary VIEW support within a single execute() call.
    fn execute_parsed_multi_statements(
        stmts: Vec<SqlStatement>,
        base_dir: &Path,
        default_table_path: &Path,
    ) -> io::Result<ApexResult> {
        let (result, _txn_id) =
            Self::execute_multi_with_txn(stmts, base_dir, default_table_path, None)?;
        Ok(result)
    }

    /// Execute multiple SQL statements with full transaction support.
    /// Handles BEGIN/COMMIT/ROLLBACK within the statement sequence, tracks txn_id,
    /// invalidates storage cache between write operations, and routes DML inside
    /// active transactions through execute_in_txn.
    /// Returns (last_result, final_txn_id) where final_txn_id reflects the
    /// transaction state after all statements have been executed.
    pub fn execute_multi_with_txn(
        stmts: Vec<SqlStatement>,
        base_dir: &Path,
        default_table_path: &Path,
        initial_txn_id: Option<u64>,
    ) -> io::Result<(ApexResult, Option<u64>)> {
        let mut views: AHashMap<String, SelectStatement> = AHashMap::new();
        let mut last_result: Option<ApexResult> = None;
        let mut current_txn: Option<u64> = initial_txn_id;

        for stmt in stmts {
            // Determine if this is a write operation (for cache invalidation)
            let is_write = matches!(
                &stmt,
                SqlStatement::Insert { .. }
                    | SqlStatement::InsertOnConflict { .. }
                    | SqlStatement::InsertSelect { .. }
                    | SqlStatement::Delete { .. }
                    | SqlStatement::Update { .. }
                    | SqlStatement::TruncateTable { .. }
                    | SqlStatement::AlterTable { .. }
                    | SqlStatement::CreateTable { .. }
                    | SqlStatement::DropTable { .. }
                    | SqlStatement::CreateIndex { .. }
                    | SqlStatement::DropIndex { .. }
                    | SqlStatement::Reindex { .. }
            );

            match stmt {
                // Transaction commands
                SqlStatement::BeginTransaction { read_only } => {
                    let result = Self::execute_begin(read_only)?;
                    if let ApexResult::Scalar(txn_id) = &result {
                        current_txn = Some(*txn_id as u64);
                    }
                    last_result = Some(result);
                }
                SqlStatement::Commit => {
                    if let Some(txn_id) = current_txn {
                        let result =
                            Self::execute_commit_txn(txn_id, base_dir, default_table_path)?;
                        // Invalidate cache after commit to ensure fresh data
                        invalidate_storage_cache(default_table_path);
                        crate::storage::engine::engine().invalidate(default_table_path);
                        current_txn = None;
                        last_result = Some(result);
                    } else {
                        return Err(err_input("COMMIT without active transaction"));
                    }
                }
                SqlStatement::Rollback => {
                    if let Some(txn_id) = current_txn {
                        let result = Self::execute_rollback_txn(txn_id)?;
                        current_txn = None;
                        last_result = Some(result);
                    } else {
                        return Err(err_input("ROLLBACK without active transaction"));
                    }
                }
                SqlStatement::Savepoint { name } => {
                    if let Some(txn_id) = current_txn {
                        let mgr = crate::txn::txn_manager();
                        mgr.with_context(txn_id, |ctx| {
                            ctx.savepoint(&name);
                            Ok(())
                        })?;
                        last_result = Some(ApexResult::Scalar(0));
                    } else {
                        return Err(err_input("SAVEPOINT without active transaction"));
                    }
                }
                SqlStatement::RollbackToSavepoint { name } => {
                    if let Some(txn_id) = current_txn {
                        let mgr = crate::txn::txn_manager();
                        mgr.with_context(txn_id, |ctx| ctx.rollback_to_savepoint(&name))?;
                        last_result = Some(ApexResult::Scalar(0));
                    } else {
                        return Err(err_input(
                            "ROLLBACK TO SAVEPOINT without active transaction",
                        ));
                    }
                }
                SqlStatement::ReleaseSavepoint { name } => {
                    if let Some(txn_id) = current_txn {
                        let mgr = crate::txn::txn_manager();
                        mgr.with_context(txn_id, |ctx| ctx.release_savepoint(&name))?;
                        last_result = Some(ApexResult::Scalar(0));
                    } else {
                        return Err(err_input("RELEASE SAVEPOINT without active transaction"));
                    }
                }
                // View management
                SqlStatement::CreateView { name, stmt } => {
                    let view_name = name.trim_matches('"').to_string();
                    if view_name.eq_ignore_ascii_case("default") {
                        return Err(err_input("View name conflicts with default table"));
                    }

                    // Disallow conflict with existing table file
                    let table_path =
                        Self::resolve_table_path(&view_name, base_dir, default_table_path);
                    if table_path.exists() {
                        return Err(err_input("View name conflicts with existing table"));
                    }

                    let persisted = Self::read_view_catalog(base_dir);
                    let rewritten_stmt = if persisted.is_empty() {
                        stmt
                    } else {
                        Self::rewrite_select_views(stmt, &persisted)
                    };
                    Self::persist_view(base_dir, &view_name, &rewritten_stmt)?;
                    views.insert(view_name.to_lowercase(), rewritten_stmt);
                    last_result = Some(ApexResult::Scalar(0));
                }
                SqlStatement::DropView { name } => {
                    let view_name = name.trim_matches('"');
                    views.remove(&view_name.to_lowercase());
                    Self::remove_persisted_view(base_dir, view_name)?;
                    last_result = Some(ApexResult::Scalar(0));
                }
                // SELECT with view rewriting
                SqlStatement::Select(mut select) => {
                    let merged_views = Self::load_all_views(base_dir, &views);
                    select = Self::rewrite_select_views(select, &merged_views);
                    if let Some(txn_id) = current_txn {
                        last_result = Some(Self::execute_in_txn(
                            txn_id,
                            SqlStatement::Select(select),
                            base_dir,
                            default_table_path,
                        )?);
                    } else {
                        last_result = Some(Self::execute_parsed_multi(
                            SqlStatement::Select(select),
                            base_dir,
                            default_table_path,
                        )?);
                    }
                }
                SqlStatement::Union(union) => {
                    let merged_views = Self::load_all_views(base_dir, &views);
                    let union = match Self::rewrite_statement_views(
                        SqlStatement::Union(union),
                        &merged_views,
                    ) {
                        SqlStatement::Union(rewritten) => rewritten,
                        _ => unreachable!(),
                    };
                    last_result = Some(Self::execute_union(union, base_dir, default_table_path)?);
                }
                // DML/DDL statements - route through txn if active
                other => {
                    let merged_views = Self::load_all_views(base_dir, &views);
                    let other = if merged_views.is_empty() {
                        other
                    } else {
                        Self::rewrite_statement_views(other, &merged_views)
                    };
                    if let Some(txn_id) = current_txn {
                        // Inside transaction: buffer DML through execute_in_txn
                        last_result = Some(Self::execute_in_txn(
                            txn_id,
                            other,
                            base_dir,
                            default_table_path,
                        )?);
                    } else {
                        // Outside transaction: execute directly
                        last_result = Some(Self::execute_parsed_multi(
                            other,
                            base_dir,
                            default_table_path,
                        )?);
                        // Invalidate cache after write operations to ensure next statement sees fresh data
                        if is_write {
                            invalidate_storage_cache(default_table_path);
                            crate::storage::engine::engine().invalidate(default_table_path);
                        }
                    }
                }
            }
        }

        let result = last_result.ok_or_else(|| err_input("No query to execute"))?;
        Ok((result, current_txn))
    }

    fn rewrite_select_views(
        mut select: SelectStatement,
        views: &AHashMap<String, SelectStatement>,
    ) -> SelectStatement {
        select.from = select
            .from
            .take()
            .map(|from| Self::rewrite_from_item_views(from, views));
        select.joins = select
            .joins
            .into_iter()
            .map(|join| JoinClause {
                join_type: join.join_type,
                right: Self::rewrite_from_item_views(join.right, views),
                on: Self::rewrite_expr_views(join.on, views),
            })
            .collect();
        select.where_clause = select
            .where_clause
            .take()
            .map(|expr| Self::rewrite_expr_views(expr, views));
        select.group_by_exprs = select
            .group_by_exprs
            .into_iter()
            .map(|expr| expr.map(|inner| Self::rewrite_expr_views(inner, views)))
            .collect();
        select.having = select
            .having
            .take()
            .map(|expr| Self::rewrite_expr_views(expr, views));
        select.order_by = select
            .order_by
            .into_iter()
            .map(|mut clause| {
                clause.expr = clause
                    .expr
                    .take()
                    .map(|expr| Self::rewrite_expr_views(expr, views));
                clause
            })
            .collect();
        select.columns = select
            .columns
            .into_iter()
            .map(|column| match column {
                SelectColumn::Expression { expr, alias } => SelectColumn::Expression {
                    expr: Self::rewrite_expr_views(expr, views),
                    alias,
                },
                other => other,
            })
            .collect();
        select
    }

    /// Parse "col=N", "col>N", "col>=N", "col<N", "col<=N", "col BETWEEN A AND B" into a SqlExpr.
    /// Used by the DELETE pre-parse fast path to skip SqlParser::parse_multi entirely.
    fn try_parse_delete_numeric_where(where_str: &str) -> Option<SqlExpr> {
        let upper = where_str.to_ascii_uppercase();

        // BETWEEN: "col BETWEEN a AND b"
        if let Some(bet_pos) = upper.find(" BETWEEN ") {
            let col = where_str[..bet_pos].trim().trim_matches('"').to_string();
            let rest = &where_str[bet_pos + 9..];
            if let Some(and_pos) = rest.to_ascii_uppercase().find(" AND ") {
                let low: f64 = rest[..and_pos].trim().parse().ok()?;
                let high: f64 = rest[and_pos + 5..].trim().parse().ok()?;
                return Some(SqlExpr::Between {
                    column: col,
                    low: Box::new(SqlExpr::Literal(Value::Float64(low))),
                    high: Box::new(SqlExpr::Literal(Value::Float64(high))),
                    negated: false,
                });
            }
        }

        // Binary operators — check two-char ops before single-char to avoid prefix ambiguity
        let (col_s, op, val_s) = if let Some(pos) = where_str.find(">=") {
            (&where_str[..pos], BinaryOperator::Ge, &where_str[pos + 2..])
        } else if let Some(pos) = where_str.find("<=") {
            (&where_str[..pos], BinaryOperator::Le, &where_str[pos + 2..])
        } else if let Some(pos) = where_str.find('>') {
            (&where_str[..pos], BinaryOperator::Gt, &where_str[pos + 1..])
        } else if let Some(pos) = where_str.find('<') {
            (&where_str[..pos], BinaryOperator::Lt, &where_str[pos + 1..])
        } else if let Some(pos) = where_str.find('=') {
            (&where_str[..pos], BinaryOperator::Eq, &where_str[pos + 1..])
        } else {
            return None;
        };
        let col = col_s.trim().trim_matches('"');
        if col.is_empty() || col.contains(' ') || col.contains('(') {
            return None;
        }
        // Column name must be a valid identifier (starts with letter or _), not a numeric literal
        if !col
            .chars()
            .next()
            .map_or(false, |c| c.is_alphabetic() || c == '_')
        {
            return None;
        }
        let val: f64 = val_s.trim().parse().ok()?;
        Some(SqlExpr::BinaryOp {
            left: Box::new(SqlExpr::Column(col.to_string())),
            op,
            right: Box::new(SqlExpr::Literal(Value::Float64(val))),
        })
    }

    /// Execute SELECT statement (resolves FROM table path relative to storage_path's directory)
    fn execute_select(stmt: SelectStatement, storage_path: &Path) -> io::Result<ApexResult> {
        let base_dir = storage_path.parent().unwrap_or(storage_path);
        // Resolve the actual table path from the FROM clause so UNION/subquery sides use the right table
        let actual_path = if let Some(FromItem::Table { ref table, .. }) = stmt.from {
            Self::resolve_table_path(table, base_dir, storage_path)
        } else {
            storage_path.to_path_buf()
        };
        Self::execute_select_with_base_dir(stmt, &actual_path, base_dir, storage_path)
    }
}

// Split impl blocks for ApexExecutor methods
include!("select.rs");
include!("joins.rs");
include!("expressions.rs");
include!("aggregation.rs");
include!("window.rs");
include!("ddl.rs");
include!("dml.rs");

#[cfg(test)]
mod tests;
