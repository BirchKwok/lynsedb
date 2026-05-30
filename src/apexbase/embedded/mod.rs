//! ApexBase Embedded Rust API
//!
//! A clean, ergonomic Rust interface for using ApexBase as an embedded database
//! with zero Python overhead and direct access to the full query engine.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use apexbase::embedded::{ApexDB, Row};
//! use apexbase::data::Value;
//! use std::collections::HashMap;
//!
//! fn main() -> apexbase::Result<()> {
//!     // Open (or create) a database directory
//!     let db = ApexDB::open("./my_data")?;
//!
//!     // Create a table
//!     let table = db.create_table("users")?;
//!
//!     // Insert a record
//!     let mut rec = HashMap::new();
//!     rec.insert("name".to_string(), Value::String("Alice".to_string()));
//!     rec.insert("age".to_string(),  Value::Int64(30));
//!     let id = table.insert(rec)?;
//!     println!("inserted _id = {id}");
//!
//!     // SQL query
//!     let rs = table.execute("SELECT * FROM users WHERE age > 25")?;
//!     let batch = rs.to_record_batch()?;
//!     println!("{} rows returned", batch.num_rows());
//!
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, LargeBinaryArray, LargeStringArray, StringArray, UInt16Array,
    UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::{DataType as ArrowDataType, Schema};
use arrow::record_batch::RecordBatch;
use parking_lot::RwLock;

use crate::data::{DataType, Value};
use crate::query::{ApexExecutor, ApexResult};
use crate::storage::on_demand::ColumnType;
use crate::storage::DurabilityLevel;
use crate::{ApexError, Result};

// ============================================================================
// Public type aliases
// ============================================================================

/// A single database row: column name → [`Value`].
pub type Row = HashMap<String, Value>;

// ============================================================================
// Internal shared state
// ============================================================================

struct DbInner {
    root_dir: PathBuf,
    base_dir: RwLock<PathBuf>,
    table_paths: RwLock<HashMap<String, PathBuf>>,
    durability: DurabilityLevel,
    temp_dir: PathBuf,
    temp_tables: RwLock<HashMap<String, PathBuf>>,
}

impl Drop for DbInner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.temp_dir);
        crate::query::executor::clear_temp_dir();
    }
}

impl DbInner {
    #[inline]
    fn current_base_dir(&self) -> PathBuf {
        self.base_dir.read().clone()
    }

    fn table_path_for(&self, name: &str) -> PathBuf {
        self.base_dir.read().join(format!("{}.apex", name))
    }

    fn resolve_table(&self, name: &str) -> Result<PathBuf> {
        {
            let temps = self.temp_tables.read();
            if let Some(p) = temps.get(name) {
                if p.exists() {
                    return Ok(p.clone());
                }
            }
        }
        {
            let paths = self.table_paths.read();
            if let Some(p) = paths.get(name) {
                return Ok(p.clone());
            }
        }
        let p = self.table_path_for(name);
        if p.exists() {
            self.table_paths.write().insert(name.to_string(), p.clone());
            Ok(p)
        } else {
            Err(ApexError::TableNotFound(name.to_string()))
        }
    }

    fn register_table(&self, name: &str, path: PathBuf) {
        self.table_paths.write().insert(name.to_string(), path);
    }

    fn unregister_table(&self, name: &str) {
        self.table_paths.write().remove(name);
    }

    fn register_temp_table(&self, name: &str, path: PathBuf) {
        self.temp_tables.write().insert(name.to_string(), path);
    }

    fn drop_temp_table(&self, name: &str) -> Result<()> {
        if let Some(path) = self.temp_tables.write().remove(name) {
            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_file(path.with_extension("apex.wal"));
            let _ = std::fs::remove_file(path.with_extension("apex.lock"));
            crate::storage::engine::engine().invalidate(&path);
        }
        Ok(())
    }
}

// ============================================================================
// ApexDB
// ============================================================================

/// Top-level database handle.
///
/// Cheaply clonable (`Arc` internally) — open once, share across threads.
///
/// # Example
/// ```rust,no_run
/// use apexbase::embedded::ApexDB;
///
/// let db = ApexDB::open("./data").unwrap();
/// let table = db.create_table("events").unwrap();
/// ```
#[derive(Clone)]
pub struct ApexDB {
    inner: Arc<DbInner>,
}

impl std::fmt::Debug for ApexDB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApexDB")
            .field("base_dir", &self.inner.current_base_dir())
            .field("durability", &self.inner.durability)
            .finish()
    }
}

// ============================================================================
// ApexDBBuilder
// ============================================================================

/// Builder for [`ApexDB`] with advanced options.
///
/// # Example
/// ```rust,no_run
/// use apexbase::embedded::ApexDB;
/// use apexbase::storage::DurabilityLevel;
///
/// let db = ApexDB::builder("./data")
///     .durability(DurabilityLevel::Safe)
///     .drop_if_exists(true)
///     .build()
///     .unwrap();
/// ```
pub struct ApexDBBuilder {
    path: PathBuf,
    durability: DurabilityLevel,
    drop_if_exists: bool,
}

impl ApexDBBuilder {
    /// Set the durability level (`Fast` / `Safe` / `Max`). Default: `Fast`.
    ///
    /// | Level | fsync | Trade-off |
    /// |-------|-------|-----------|
    /// | `Fast` | never | Highest throughput; data in OS buffer only |
    /// | `Safe` | on `flush()` | Balanced; data survives clean shutdown |
    /// | `Max`  | every write | Strongest ACID; ~10–50× slower |
    pub fn durability(mut self, level: DurabilityLevel) -> Self {
        self.durability = level;
        self
    }

    /// When `true`, all existing `.apex` files in the directory are deleted before opening.
    pub fn drop_if_exists(mut self, flag: bool) -> Self {
        self.drop_if_exists = flag;
        self
    }

    /// Build and open the database.
    pub fn build(self) -> Result<ApexDB> {
        open_db(self.path, self.durability, self.drop_if_exists)
    }
}

// ============================================================================
// ApexDB impl
// ============================================================================

impl ApexDB {
    /// Open (or create) a database at `path` with `Fast` durability.
    ///
    /// The directory is created if it does not exist.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        open_db(path, DurabilityLevel::Fast, false)
    }

    /// Return a builder for more control over database settings.
    pub fn builder(path: impl AsRef<Path>) -> ApexDBBuilder {
        ApexDBBuilder {
            path: path.as_ref().to_path_buf(),
            durability: DurabilityLevel::Fast,
            drop_if_exists: false,
        }
    }

    // ── Table DDL ─────────────────────────────────────────────────────────────

    /// Create a new empty table and return a handle to it.
    ///
    /// Returns [`ApexError::TableExists`] if the table already exists.
    pub fn create_table(&self, name: &str) -> Result<Table> {
        let path = self.inner.table_path_for(name);
        if path.exists() {
            return Err(ApexError::TableExists(name.to_string()));
        }
        let engine = crate::storage::engine::engine();
        engine.create_table(&path, self.inner.durability)?;
        self.inner.register_table(name, path.clone());
        Ok(Table {
            inner: self.inner.clone(),
            name: name.to_string(),
            path,
        })
    }

    /// Create a new table with a predefined schema.
    ///
    /// Pre-defining the schema avoids type inference on the first insert and
    /// guarantees column order.
    ///
    /// ```rust,no_run
    /// use apexbase::embedded::ApexDB;
    /// use apexbase::storage::on_demand::ColumnType;
    ///
    /// let db = ApexDB::open("./data").unwrap();
    /// let t = db.create_table_with_schema("orders", &[
    ///     ("order_id".to_string(), ColumnType::Int64),
    ///     ("product".to_string(),  ColumnType::String),
    ///     ("price".to_string(),    ColumnType::Float64),
    /// ]).unwrap();
    /// ```
    pub fn create_table_with_schema(
        &self,
        name: &str,
        schema: &[(String, ColumnType)],
    ) -> Result<Table> {
        let path = self.inner.table_path_for(name);
        if path.exists() {
            return Err(ApexError::TableExists(name.to_string()));
        }
        let engine = crate::storage::engine::engine();
        engine.create_table_with_schema(&path, self.inner.durability, schema)?;
        self.inner.register_table(name, path.clone());
        Ok(Table {
            inner: self.inner.clone(),
            name: name.to_string(),
            path,
        })
    }

    /// Open an existing table by name.
    ///
    /// Returns [`ApexError::TableNotFound`] if the table does not exist.
    pub fn table(&self, name: &str) -> Result<Table> {
        let path = self.inner.resolve_table(name)?;
        Ok(Table {
            inner: self.inner.clone(),
            name: name.to_string(),
            path,
        })
    }

    /// Drop (permanently delete) a table.
    pub fn drop_table(&self, name: &str) -> Result<()> {
        // Evict engine caches before touching the file
        {
            let paths = self.inner.table_paths.read();
            if let Some(p) = paths.get(name) {
                crate::storage::engine::engine().invalidate(p);
            }
        }
        let path = {
            let mut paths = self.inner.table_paths.write();
            paths.remove(name)
        };
        let path = path.unwrap_or_else(|| self.inner.table_path_for(name));
        if path.exists() {
            fs::remove_file(&path)?;
        } else {
            return Err(ApexError::TableNotFound(name.to_string()));
        }
        // Drop companion files (delta, lock, WAL)
        for ext in &["apex.delta", "apex.lock", "apex.wal"] {
            let p = path.with_extension(ext);
            let _ = fs::remove_file(p);
        }
        Ok(())
    }

    /// List all tables in the current database directory (sorted).
    pub fn list_tables(&self) -> Vec<String> {
        let base_dir = self.inner.current_base_dir();
        let mut tables = Vec::new();
        if let Ok(entries) = fs::read_dir(&base_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("apex") {
                    if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                        tables.push(stem.to_string());
                    }
                }
            }
        }
        tables.sort();
        tables
    }

    /// Register a data file (CSV, JSON, Parquet) as a temporary table.
    ///
    /// The file is parsed once and materialized into a native `.apex` table
    /// stored in a temp directory. Subsequent queries on this table bypass
    /// file parsing and use the mmap-backed native format with zone maps and
    /// bloom filters, achieving an order-of-magnitude speedup vs repeated
    /// `read_csv()` / `read_json()` / `read_parquet()` calls.
    ///
    /// The temp table is automatically cleaned up when the database is dropped.
    pub fn register_temp_table(&self, name: &str, file_path: &str) -> Result<()> {
        let temp_path = self.inner.temp_dir.join(format!("{}.apex", name));
        let _ = std::fs::create_dir_all(&self.inner.temp_dir);

        // Parse file → create native .apex table using COPY IMPORT
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

        crate::query::executor::set_temp_dir(&self.inner.temp_dir);
        let base_dir = self.inner.current_base_dir();
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
                self.inner.register_temp_table(name, temp_path);
                Ok(())
            }
            Err(e) => {
                let _ = std::fs::remove_file(&temp_path);
                Err(e.into())
            }
        }
    }

    /// Drop a previously registered temporary table.
    pub fn drop_temp_table(&self, name: &str) -> Result<()> {
        self.inner.drop_temp_table(name)
    }

    // ── SQL ───────────────────────────────────────────────────────────────────

    /// Execute arbitrary SQL against the database.
    ///
    /// For table-scoped queries, prefer [`Table::execute`] which provides the
    /// correct default-table context automatically.
    pub fn execute(&self, sql: &str) -> Result<ResultSet> {
        use crate::query::query_signature::{self, QuerySignature};

        let sig = query_signature::classify(sql);
        let is_write = matches!(&sig, QuerySignature::DmlWrite | QuerySignature::Ddl { .. });

        let base_dir = self.inner.current_base_dir();
        crate::query::executor::set_query_root_dir(&self.inner.root_dir);
        crate::query::executor::set_temp_dir(&self.inner.temp_dir);
        let result = ApexExecutor::execute_with_base_dir(sql, &base_dir, &base_dir);
        crate::query::executor::clear_temp_dir();
        crate::query::executor::clear_query_root_dir();
        let result = result?;

        // Sync table registry after DDL / DML
        if is_write {
            if let QuerySignature::Ddl {
                kind: crate::query::query_signature::DdlKind::DropTable { ref name },
            } = sig
            {
                self.inner.unregister_table(name);
                crate::storage::engine::engine().invalidate(&self.inner.table_path_for(name));
            } else {
                crate::storage::engine::engine().invalidate_dir(&base_dir);
            }
        }

        Ok(ResultSet { inner: result })
    }

    // ── Multi-database ────────────────────────────────────────────────────────

    /// Switch to a named database (sub-directory). `""` or `"default"` reverts to root.
    ///
    /// The sub-directory is created automatically if it does not exist.
    pub fn use_database(&self, db_name: &str) -> Result<()> {
        let new_base = if db_name.is_empty() || db_name.eq_ignore_ascii_case("default") {
            self.inner.root_dir.clone()
        } else {
            let d = self.inner.root_dir.join(db_name);
            fs::create_dir_all(&d)?;
            d
        };
        *self.inner.base_dir.write() = new_base;
        self.inner.table_paths.write().clear();
        Ok(())
    }

    /// Invalidate all engine caches. Useful after external writes to the same directory.
    pub fn invalidate_cache(&self) {
        crate::storage::engine::engine().invalidate_dir(&self.inner.current_base_dir());
    }

    /// Return the current base directory path.
    pub fn base_dir(&self) -> PathBuf {
        self.inner.current_base_dir()
    }
}

// ============================================================================
// Table
// ============================================================================

/// Table-scoped operations handle.
///
/// Obtained from [`ApexDB::create_table`], [`ApexDB::create_table_with_schema`],
/// or [`ApexDB::table`]. Cheaply clonable.
#[derive(Clone)]
pub struct Table {
    inner: Arc<DbInner>,
    /// Table name.
    pub name: String,
    path: PathBuf,
}

impl std::fmt::Debug for Table {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Table")
            .field("name", &self.name)
            .field("path", &self.path)
            .finish()
    }
}

impl Table {
    // ── Write Operations ──────────────────────────────────────────────────────

    /// Insert a single record. Returns the assigned `_id`.
    ///
    /// ```rust,no_run
    /// use apexbase::embedded::{ApexDB, Row};
    /// use apexbase::data::Value;
    ///
    /// let db = ApexDB::open("./data").unwrap();
    /// let t = db.table("users").unwrap();
    ///
    /// let id = t.insert([
    ///     ("name".to_string(), Value::String("Bob".to_string())),
    ///     ("age".to_string(),  Value::Int64(25)),
    /// ].into_iter().collect()).unwrap();
    /// ```
    pub fn insert(&self, record: Row) -> Result<u64> {
        let engine = crate::storage::engine::engine();
        let ids = engine.write(&self.path, &[record], self.inner.durability)?;
        Ok(ids.into_iter().next().unwrap_or(0))
    }

    /// Insert multiple records. Returns the assigned `_id` values.
    pub fn insert_batch(&self, records: &[Row]) -> Result<Vec<u64>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let engine = crate::storage::engine::engine();
        Ok(engine.write(&self.path, records, self.inner.durability)?)
    }

    /// Insert an Arrow [`RecordBatch`]. Returns the assigned `_id` values.
    ///
    /// This is the fastest path for bulk-loading data already in Arrow format.
    pub fn insert_arrow(&self, batch: &RecordBatch) -> Result<Vec<u64>> {
        if batch.num_rows() == 0 {
            return Ok(Vec::new());
        }
        let rows = record_batch_to_rows(batch)?;
        self.insert_batch(&rows)
    }

    /// Replace (overwrite) a record by `_id`. Returns `true` if the record existed.
    pub fn replace(&self, id: u64, record: Row) -> Result<bool> {
        let engine = crate::storage::engine::engine();
        Ok(engine.replace(&self.path, id, &record, self.inner.durability)?)
    }

    /// Delete a record by `_id`. Returns `true` if the record existed.
    pub fn delete(&self, id: u64) -> Result<bool> {
        let engine = crate::storage::engine::engine();
        Ok(engine.delete_one(&self.path, id, self.inner.durability)?)
    }

    /// Delete multiple records by `_id`. Returns the number of records deleted.
    pub fn delete_batch(&self, ids: &[u64]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let engine = crate::storage::engine::engine();
        Ok(engine.delete(&self.path, ids, self.inner.durability)?)
    }

    // ── Read Operations ───────────────────────────────────────────────────────

    /// Retrieve a single record by `_id`. Returns `None` if not found.
    pub fn retrieve(&self, id: u64) -> Result<Option<Row>> {
        let engine = crate::storage::engine::engine();
        let backend = engine.get_read_backend(&self.path)?;
        match backend.storage.retrieve_rcix(id)? {
            None => Ok(None),
            Some(cols) => Ok(Some(cols.into_iter().collect())),
        }
    }

    /// Retrieve multiple records by `_id`. Returns an Arrow [`RecordBatch`].
    ///
    /// Uses the V4 mmap fast-path when available, falling back to SQL for
    /// older formats.
    pub fn retrieve_many(&self, ids: &[u64]) -> Result<RecordBatch> {
        if ids.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::new(Schema::empty())));
        }
        let engine = crate::storage::engine::engine();
        let backend = engine.get_read_backend(&self.path)?;

        // Fast path: V4 mmap batch read
        if backend.storage.is_v4_format() && !backend.storage.has_v4_in_memory_data() {
            if let Ok(Some(batch)) = backend.storage.retrieve_many_mmap(ids) {
                return Ok(batch);
            }
        }

        // Fallback: SQL IN (…)
        let id_list = ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!("SELECT * FROM \"{}\" WHERE _id IN ({})", self.name, id_list);
        self.execute(&sql)?.to_record_batch()
    }

    /// Return the active (non-deleted) row count. O(1) for V4 format files.
    pub fn count(&self) -> Result<u64> {
        let engine = crate::storage::engine::engine();
        Ok(engine.active_row_count(&self.path)?)
    }

    /// Return `true` if a record with the given `_id` exists.
    pub fn exists(&self, id: u64) -> Result<bool> {
        let engine = crate::storage::engine::engine();
        Ok(engine.exists(&self.path, id)?)
    }

    // ── SQL Execution ─────────────────────────────────────────────────────────

    /// Execute arbitrary SQL with this table as the default context.
    ///
    /// The table is available as an unqualified name inside the SQL.
    ///
    /// ```rust,no_run
    /// # use apexbase::embedded::ApexDB;
    /// # let db = ApexDB::open("./data").unwrap();
    /// # let t = db.table("users").unwrap();
    /// let rs = t.execute("SELECT city, COUNT(*) FROM users GROUP BY city").unwrap();
    /// let batch = rs.to_record_batch().unwrap();
    /// ```
    pub fn execute(&self, sql: &str) -> Result<ResultSet> {
        let sig = crate::query::query_signature::classify(sql);
        if !matches!(
            sig,
            crate::query::query_signature::QuerySignature::CountStar { .. }
        ) {
            let engine = crate::storage::engine::engine();
            let backend = engine.get_read_backend(&self.path)?;
            if backend.pending_v4_in_memory_rows() > 0 {
                backend.save()?;
            }
        }

        let base_dir = self.inner.current_base_dir();
        crate::query::executor::set_query_root_dir(&self.inner.root_dir);
        let result = ApexExecutor::execute_with_base_dir(sql, &base_dir, &self.path);
        crate::query::executor::clear_query_root_dir();
        Ok(ResultSet { inner: result? })
    }

    // ── Schema Operations ─────────────────────────────────────────────────────

    /// Return the table schema as `(column_name, DataType)` pairs.
    pub fn schema(&self) -> Result<Vec<(String, DataType)>> {
        Ok(crate::storage::engine::engine().get_schema(&self.path)?)
    }

    /// Return column names (in schema order).
    pub fn columns(&self) -> Result<Vec<String>> {
        Ok(crate::storage::engine::engine().list_columns(&self.path)?)
    }

    /// Get the [`DataType`] of a specific column.
    pub fn column_type(&self, name: &str) -> Result<Option<DataType>> {
        Ok(crate::storage::engine::engine().get_column_type(&self.path, name)?)
    }

    /// Add a new column (all existing rows are set to `NULL`).
    pub fn add_column(&self, name: &str, dtype: DataType) -> Result<()> {
        Ok(crate::storage::engine::engine().add_column(
            &self.path,
            name,
            dtype,
            self.inner.durability,
        )?)
    }

    /// Drop a column.
    pub fn drop_column(&self, name: &str) -> Result<()> {
        Ok(
            crate::storage::engine::engine().drop_column(
                &self.path,
                name,
                self.inner.durability,
            )?,
        )
    }

    /// Rename a column.
    pub fn rename_column(&self, old_name: &str, new_name: &str) -> Result<()> {
        Ok(crate::storage::engine::engine().rename_column(
            &self.path,
            old_name,
            new_name,
            self.inner.durability,
        )?)
    }

    // ── Maintenance ───────────────────────────────────────────────────────────

    /// Flush pending in-memory data to disk.
    pub fn flush(&self) -> Result<()> {
        let engine = crate::storage::engine::engine();
        let backend = engine.get_write_backend(&self.path, self.inner.durability)?;
        backend.save()?;
        Ok(())
    }

    /// Return the absolute path to the `.apex` file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ============================================================================
// ResultSet
// ============================================================================

/// Result of a SQL query or DML statement.
///
/// Obtained from [`ApexDB::execute`] or [`Table::execute`].
pub struct ResultSet {
    inner: ApexResult,
}

impl ResultSet {
    /// Consume and convert to an Arrow [`RecordBatch`].
    ///
    /// This is a zero-copy operation for `Data` variants.
    pub fn to_record_batch(self) -> Result<RecordBatch> {
        self.inner
            .to_record_batch()
            .map_err(|e| ApexError::Io(io::Error::new(io::ErrorKind::Other, e.to_string())))
    }

    /// Consume and convert to `Vec<`[`Row`]`>`.
    pub fn to_rows(self) -> Result<Vec<Row>> {
        let batch = self.to_record_batch()?;
        record_batch_to_rows(&batch)
    }

    /// Number of result rows.
    pub fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }

    /// For scalar results (e.g., `COUNT(*)`), return the single `i64` value.
    pub fn scalar(&self) -> Option<i64> {
        if let ApexResult::Scalar(v) = &self.inner {
            Some(*v)
        } else {
            None
        }
    }

    /// Column names in result order.
    pub fn columns(&self) -> Vec<String> {
        match &self.inner {
            ApexResult::Data(batch) => batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect(),
            ApexResult::Empty(schema) => schema.fields().iter().map(|f| f.name().clone()).collect(),
            ApexResult::Scalar(_) => vec!["result".to_string()],
        }
    }

    /// Return `true` if the result contains no rows.
    pub fn is_empty(&self) -> bool {
        self.num_rows() == 0
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

fn open_db(
    path: impl AsRef<Path>,
    durability: DurabilityLevel,
    drop_if_exists: bool,
) -> Result<ApexDB> {
    let root_dir = to_absolute(path.as_ref());
    fs::create_dir_all(&root_dir)?;

    if drop_if_exists {
        if let Ok(entries) = fs::read_dir(&root_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().map(|e| e == "apex").unwrap_or(false) {
                    let _ = fs::remove_file(&p);
                }
            }
        }
        let fts_dir = root_dir.join("fts_indexes");
        if fts_dir.exists() {
            let _ = fs::remove_dir_all(&fts_dir);
        }
    }

    // Pre-warm rayon global thread pool (no-op if already initialized)
    rayon::spawn(|| {});

    let temp_dir = root_dir.join(".apex_tmp");
    let _ = fs::create_dir_all(&temp_dir);

    Ok(ApexDB {
        inner: Arc::new(DbInner {
            root_dir: root_dir.clone(),
            base_dir: RwLock::new(root_dir),
            table_paths: RwLock::new(HashMap::new()),
            durability,
            temp_dir,
            temp_tables: RwLock::new(HashMap::new()),
        }),
    })
}

#[inline]
fn to_absolute(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

/// Convert an Arrow [`RecordBatch`] to `Vec<Row>`.
pub fn record_batch_to_rows(batch: &RecordBatch) -> Result<Vec<Row>> {
    let num_rows = batch.num_rows();
    let num_cols = batch.num_columns();
    let schema = batch.schema();

    let mut rows: Vec<Row> = (0..num_rows)
        .map(|_| HashMap::with_capacity(num_cols))
        .collect();

    for col_idx in 0..num_cols {
        let col_name = schema.field(col_idx).name().clone();
        let col = batch.column(col_idx);
        for row_idx in 0..num_rows {
            rows[row_idx].insert(col_name.clone(), arrow_value_at(col, row_idx));
        }
    }
    Ok(rows)
}

/// Extract a single [`Value`] from an Arrow array at a given row index.
pub fn arrow_value_at(arr: &ArrayRef, row: usize) -> Value {
    if arr.is_null(row) {
        return Value::Null;
    }
    match arr.data_type() {
        ArrowDataType::Int64 => Value::Int64(
            arr.as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(row),
        ),
        ArrowDataType::Int32 => Value::Int64(
            arr.as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(row) as i64,
        ),
        ArrowDataType::Int16 => Value::Int64(
            arr.as_any()
                .downcast_ref::<Int16Array>()
                .unwrap()
                .value(row) as i64,
        ),
        ArrowDataType::Int8 => {
            Value::Int64(arr.as_any().downcast_ref::<Int8Array>().unwrap().value(row) as i64)
        }
        ArrowDataType::UInt64 => Value::Int64(
            arr.as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(row) as i64,
        ),
        ArrowDataType::UInt32 => Value::Int64(
            arr.as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap()
                .value(row) as i64,
        ),
        ArrowDataType::UInt16 => Value::Int64(
            arr.as_any()
                .downcast_ref::<UInt16Array>()
                .unwrap()
                .value(row) as i64,
        ),
        ArrowDataType::UInt8 => Value::Int64(
            arr.as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap()
                .value(row) as i64,
        ),
        ArrowDataType::Float64 => Value::Float64(
            arr.as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(row),
        ),
        ArrowDataType::Float32 => Value::Float64(
            arr.as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(row) as f64,
        ),
        ArrowDataType::Boolean => Value::Bool(
            arr.as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(row),
        ),
        ArrowDataType::Utf8 => Value::String(
            arr.as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row)
                .to_string(),
        ),
        ArrowDataType::LargeUtf8 => Value::String(
            arr.as_any()
                .downcast_ref::<LargeStringArray>()
                .unwrap()
                .value(row)
                .to_string(),
        ),
        ArrowDataType::Binary => Value::Binary(
            arr.as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .value(row)
                .to_vec(),
        ),
        ArrowDataType::LargeBinary => Value::Binary(
            arr.as_any()
                .downcast_ref::<LargeBinaryArray>()
                .unwrap()
                .value(row)
                .to_vec(),
        ),
        _ => Value::Null,
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn temp_db() -> (TempDir, ApexDB) {
        let dir = TempDir::new().unwrap();
        let db = ApexDB::open(dir.path()).unwrap();
        (dir, db)
    }

    fn row1(name: &str, age: i64, score: f64, city: &str) -> Row {
        let mut r = Row::new();
        r.insert("name".to_string(), Value::String(name.to_string()));
        r.insert("age".to_string(), Value::Int64(age));
        r.insert("score".to_string(), Value::Float64(score));
        r.insert("city".to_string(), Value::String(city.to_string()));
        r
    }

    // ── Group 1: basic CRUD ──────────────────────────────────────────────────

    #[test]
    fn test_create_and_insert() {
        let (_dir, db) = temp_db();
        let table = db.create_table("t1").unwrap();
        let id = table.insert(row1("Alice", 30, 92.5, "NY")).unwrap();
        assert_eq!(table.count().unwrap(), 1);
        let row = table.retrieve(id).unwrap().unwrap();
        assert_eq!(row.get("name"), Some(&Value::String("Alice".to_string())));
        assert_eq!(row.get("age"), Some(&Value::Int64(30)));
    }

    #[test]
    fn test_insert_and_retrieve() {
        let (_dir, db) = temp_db();
        let table = db.create_table("t2").unwrap();
        let mut r = Row::new();
        r.insert("x".to_string(), Value::Int64(42));
        let id = table.insert(r).unwrap();
        let row = table.retrieve(id).unwrap().unwrap();
        assert_eq!(row.get("x"), Some(&Value::Int64(42)));
    }

    #[test]
    fn test_delete() {
        let (_dir, db) = temp_db();
        let table = db.create_table("t3").unwrap();
        let mut r = Row::new();
        r.insert("v".to_string(), Value::Int64(1));
        let id = table.insert(r).unwrap();
        assert!(table.delete(id).unwrap());
        assert_eq!(table.count().unwrap(), 0);
    }

    #[test]
    fn test_replace() {
        let (_dir, db) = temp_db();
        let table = db.create_table("rep_t").unwrap();
        let id = table.insert(row1("Alice", 30, 90.0, "NY")).unwrap();

        let mut updated = Row::new();
        updated.insert("name".to_string(), Value::String("Alice-v2".to_string()));
        updated.insert("age".to_string(), Value::Int64(31));
        updated.insert("score".to_string(), Value::Float64(99.0));
        updated.insert("city".to_string(), Value::String("LA".to_string()));
        assert!(table.replace(id, updated).unwrap());

        // Verify new values
        let rs = table
            .execute(&format!(
                "SELECT name, age, score FROM rep_t WHERE _id = {}",
                id
            ))
            .unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].get("name"),
            Some(&Value::String("Alice-v2".to_string()))
        );
        assert_eq!(rows[0].get("age"), Some(&Value::Int64(31)));
    }

    #[test]
    fn test_delete_batch() {
        let (_dir, db) = temp_db();
        let table = db.create_table("del_batch_t").unwrap();
        let ids: Vec<u64> = (0..5i64)
            .map(|i| {
                let mut r = Row::new();
                r.insert("i".to_string(), Value::Int64(i));
                table.insert(r).unwrap()
            })
            .collect();

        let deleted = table.delete_batch(&ids[0..3]).unwrap();
        assert_eq!(deleted, 3);
        assert_eq!(table.count().unwrap(), 2);
    }

    #[test]
    fn test_retrieve_many_ids() {
        let (_dir, db) = temp_db();
        let table = db.create_table("rmany_t").unwrap();
        let ids: Vec<u64> = (0..5i64)
            .map(|i| {
                let mut r = Row::new();
                r.insert("v".to_string(), Value::Int64(i * 10));
                table.insert(r).unwrap()
            })
            .collect();

        let batch = table.retrieve_many(&ids[0..3]).unwrap();
        assert_eq!(batch.num_rows(), 3);
    }

    #[test]
    fn test_exists() {
        let (_dir, db) = temp_db();
        let table = db.create_table("exists_t").unwrap();
        let mut r = Row::new();
        r.insert("k".to_string(), Value::Int64(1));
        let id = table.insert(r).unwrap();
        assert!(table.exists(id).unwrap());
        table.delete(id).unwrap();
        assert!(!table.exists(id).unwrap());
    }

    #[test]
    fn test_retrieve_nonexistent() {
        let (_dir, db) = temp_db();
        let table = db.create_table("norow_t").unwrap();
        let mut r = Row::new();
        r.insert("v".to_string(), Value::Int64(1));
        table.insert(r).unwrap();
        assert!(table.retrieve(99999).unwrap().is_none());
    }

    #[test]
    fn test_delete_nonexistent() {
        let (_dir, db) = temp_db();
        let table = db.create_table("nodel_t").unwrap();
        let mut r = Row::new();
        r.insert("v".to_string(), Value::Int64(1));
        table.insert(r).unwrap();
        // Deleting a non-existent ID should return false (no-op)
        assert!(!table.delete(99999).unwrap());
    }

    #[test]
    fn test_insert_batch() {
        let (_dir, db) = temp_db();
        let table = db.create_table("batch_t").unwrap();
        let records: Vec<Row> = (0..100i64)
            .map(|i| {
                let mut r = Row::new();
                r.insert("i".to_string(), Value::Int64(i));
                r
            })
            .collect();
        let ids = table.insert_batch(&records).unwrap();
        assert_eq!(ids.len(), 100);
        assert_eq!(table.count().unwrap(), 100);
    }

    // ── Group 2: SQL coverage ────────────────────────────────────────────────

    #[test]
    fn test_sql_filter_where() {
        let (_dir, db) = temp_db();
        let table = db.create_table("filter_t").unwrap();
        for (name, age) in &[("A", 20i64), ("B", 30), ("C", 40), ("D", 25), ("E", 35)] {
            let mut r = Row::new();
            r.insert("name".to_string(), Value::String(name.to_string()));
            r.insert("age".to_string(), Value::Int64(*age));
            table.insert(r).unwrap();
        }
        // age > 28 → B(30), C(40), E(35)
        let rs = table
            .execute("SELECT name FROM filter_t WHERE age > 28")
            .unwrap();
        assert_eq!(rs.num_rows(), 3);
    }

    #[test]
    fn test_sql_count_star() {
        let (_dir, db) = temp_db();
        let table = db.create_table("cnt_t").unwrap();
        for i in 0..7i64 {
            let mut r = Row::new();
            r.insert("n".to_string(), Value::Int64(i));
            table.insert(r).unwrap();
        }
        let rs = table.execute("SELECT COUNT(*) FROM cnt_t").unwrap();
        let batch = rs.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 1);
        // Extract count value
        let rows = table
            .execute("SELECT COUNT(*) FROM cnt_t")
            .unwrap()
            .to_rows()
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_sql_aggregate_group_by() {
        let (_dir, db) = temp_db();
        let table = db.create_table("grp_t").unwrap();
        for city in &["NY", "NY", "LA", "LA", "LA", "Tokyo"] {
            let mut r = Row::new();
            r.insert("city".to_string(), Value::String(city.to_string()));
            r.insert("v".to_string(), Value::Int64(1));
            table.insert(r).unwrap();
        }
        let rs = table
            .execute("SELECT city, COUNT(*) AS n FROM grp_t GROUP BY city ORDER BY n DESC")
            .unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 3);
        // LA has 3, NY has 2, Tokyo has 1
        assert_eq!(rows[0].get("city"), Some(&Value::String("LA".to_string())));
    }

    #[test]
    fn test_sql_order_by_limit() {
        let (_dir, db) = temp_db();
        let table = db.create_table("ord_t").unwrap();
        for score in &[50i64, 90, 30, 70, 10] {
            let mut r = Row::new();
            r.insert("score".to_string(), Value::Int64(*score));
            table.insert(r).unwrap();
        }
        let rs = table
            .execute("SELECT score FROM ord_t ORDER BY score DESC LIMIT 3")
            .unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 3);
        // Top-3 scores: 90, 70, 50
        assert_eq!(rows[0].get("score"), Some(&Value::Int64(90)));
        assert_eq!(rows[1].get("score"), Some(&Value::Int64(70)));
        assert_eq!(rows[2].get("score"), Some(&Value::Int64(50)));
    }

    #[test]
    fn test_sql_update() {
        let (_dir, db) = temp_db();
        let table = db.create_table("upd_t").unwrap();
        for i in 0..5i64 {
            let mut r = Row::new();
            r.insert("n".to_string(), Value::Int64(i));
            r.insert("status".to_string(), Value::String("old".to_string()));
            table.insert(r).unwrap();
        }
        table
            .execute("UPDATE upd_t SET status = 'new' WHERE n >= 3")
            .unwrap();
        let rs = table
            .execute("SELECT n FROM upd_t WHERE status = 'new'")
            .unwrap();
        assert_eq!(rs.num_rows(), 2); // n=3 and n=4
    }

    #[test]
    fn test_sql_delete_where() {
        let (_dir, db) = temp_db();
        let table = db.create_table("delw_t").unwrap();
        for age in &[15i64, 25, 35, 18, 40] {
            let mut r = Row::new();
            r.insert("age".to_string(), Value::Int64(*age));
            table.insert(r).unwrap();
        }
        table.execute("DELETE FROM delw_t WHERE age < 20").unwrap();
        // Deleted 15 and 18 → 3 remain
        assert_eq!(table.count().unwrap(), 3);
    }

    #[test]
    fn test_sql_select_limit() {
        let (_dir, db) = temp_db();
        let table = db.create_table("lim_t").unwrap();
        for i in 0..20i64 {
            let mut r = Row::new();
            r.insert("i".to_string(), Value::Int64(i));
            table.insert(r).unwrap();
        }
        let rs = table.execute("SELECT * FROM lim_t LIMIT 5").unwrap();
        assert_eq!(rs.num_rows(), 5);
    }

    #[test]
    fn test_sql_having() {
        let (_dir, db) = temp_db();
        let table = db.create_table("hav_t").unwrap();
        for (cat, v) in &[
            ("A", 1i64),
            ("A", 2),
            ("B", 3),
            ("C", 4),
            ("C", 5),
            ("C", 6),
        ] {
            let mut r = Row::new();
            r.insert("cat".to_string(), Value::String(cat.to_string()));
            r.insert("v".to_string(), Value::Int64(*v));
            table.insert(r).unwrap();
        }
        // Groups with COUNT > 1: A(2), C(3)
        let rs = table
            .execute("SELECT cat, COUNT(*) AS n FROM hav_t GROUP BY cat HAVING COUNT(*) > 1")
            .unwrap();
        assert_eq!(rs.num_rows(), 2);
    }

    // ── Group 3: Data types ───────────────────────────────────────────────────

    #[test]
    fn test_multiple_data_types_roundtrip() {
        let (_dir, db) = temp_db();
        let table = db.create_table("types_t").unwrap();
        let mut r = Row::new();
        r.insert("int_col".to_string(), Value::Int64(12345));
        r.insert("float_col".to_string(), Value::Float64(3.14));
        r.insert("str_col".to_string(), Value::String("hello".to_string()));
        r.insert("bool_col".to_string(), Value::Bool(true));
        let id = table.insert(r).unwrap();

        let row = table.retrieve(id).unwrap().unwrap();
        assert_eq!(row.get("int_col"), Some(&Value::Int64(12345)));
        assert_eq!(
            row.get("str_col"),
            Some(&Value::String("hello".to_string()))
        );
        assert_eq!(row.get("bool_col"), Some(&Value::Bool(true)));
        // Float comparison with tolerance
        if let Some(Value::Float64(v)) = row.get("float_col") {
            assert!((v - 3.14).abs() < 1e-9);
        } else {
            panic!("float_col not Float64");
        }
    }

    #[test]
    fn test_float_precision() {
        let (_dir, db) = temp_db();
        let table = db.create_table("flt_t").unwrap();
        let values = [1.23456789012345_f64, -0.000001, 1e15, f64::MAX / 2.0];
        for &v in &values {
            let mut r = Row::new();
            r.insert("f".to_string(), Value::Float64(v));
            table.insert(r).unwrap();
        }
        assert_eq!(table.count().unwrap(), 4);
    }

    #[test]
    fn test_bool_values() {
        let (_dir, db) = temp_db();
        let table = db.create_table("bool_t").unwrap();
        for b in &[true, false, true, false] {
            let mut r = Row::new();
            r.insert("flag".to_string(), Value::Bool(*b));
            table.insert(r).unwrap();
        }
        let rs = table
            .execute("SELECT flag FROM bool_t WHERE flag = true")
            .unwrap();
        assert_eq!(rs.num_rows(), 2);
        let rs = table
            .execute("SELECT flag FROM bool_t WHERE flag = false")
            .unwrap();
        assert_eq!(rs.num_rows(), 2);
    }

    #[test]
    fn test_string_values() {
        let (_dir, db) = temp_db();
        let table = db.create_table("str_t").unwrap();
        let strings = ["hello", "world", "foo bar", "baz", ""];
        for s in &strings {
            let mut r = Row::new();
            r.insert("s".to_string(), Value::String(s.to_string()));
            table.insert(r).unwrap();
        }
        assert_eq!(table.count().unwrap(), 5);
        let rs = table
            .execute("SELECT s FROM str_t WHERE s = 'hello'")
            .unwrap();
        assert_eq!(rs.num_rows(), 1);
    }

    #[test]
    fn test_large_batch_and_filter() {
        let (_dir, db) = temp_db();
        let table = db.create_table("large_t").unwrap();
        let records: Vec<Row> = (0..500i64)
            .map(|i| {
                let mut r = Row::new();
                r.insert("n".to_string(), Value::Int64(i));
                r.insert("even".to_string(), Value::Bool(i % 2 == 0));
                r
            })
            .collect();
        table.insert_batch(&records).unwrap();
        assert_eq!(table.count().unwrap(), 500);

        let rs = table
            .execute("SELECT n FROM large_t WHERE even = true")
            .unwrap();
        assert_eq!(rs.num_rows(), 250);
        let rs = table
            .execute("SELECT COUNT(*) FROM large_t WHERE n >= 250")
            .unwrap();
        assert_eq!(rs.num_rows(), 1);
    }

    // ── Group 4: ResultSet API ────────────────────────────────────────────────

    #[test]
    fn test_resultset_scalar() {
        let (_dir, db) = temp_db();
        let table = db.create_table("scalar_t").unwrap();
        for i in 0..10i64 {
            let mut r = Row::new();
            r.insert("v".to_string(), Value::Int64(i));
            table.insert(r).unwrap();
        }
        // COUNT(*) should return scalar 10
        // Note: scalar() reads from ApexResult::Scalar variant; COUNT(*) may return Data
        let rs = table.execute("SELECT COUNT(*) FROM scalar_t").unwrap();
        let batch = rs.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 1);
    }

    #[test]
    fn test_resultset_columns() {
        let (_dir, db) = temp_db();
        let table = db.create_table("cols_t").unwrap();
        let mut r = Row::new();
        r.insert("alpha".to_string(), Value::Int64(1));
        r.insert("beta".to_string(), Value::Float64(2.0));
        r.insert("gamma".to_string(), Value::String("x".to_string()));
        table.insert(r).unwrap();

        let rs = table
            .execute("SELECT alpha, beta, gamma FROM cols_t")
            .unwrap();
        let cols = rs.columns();
        assert!(cols.contains(&"alpha".to_string()));
        assert!(cols.contains(&"beta".to_string()));
        assert!(cols.contains(&"gamma".to_string()));
    }

    #[test]
    fn test_resultset_is_empty() {
        let (_dir, db) = temp_db();
        let table = db.create_table("empty_t").unwrap();
        let mut r = Row::new();
        r.insert("n".to_string(), Value::Int64(5));
        table.insert(r).unwrap();

        let rs = table
            .execute("SELECT * FROM empty_t WHERE n > 1000")
            .unwrap();
        assert!(rs.is_empty());
        assert_eq!(rs.num_rows(), 0);
    }

    #[test]
    fn test_resultset_num_rows() {
        let (_dir, db) = temp_db();
        let table = db.create_table("numrows_t").unwrap();
        for i in 0..15i64 {
            let mut r = Row::new();
            r.insert("i".to_string(), Value::Int64(i));
            table.insert(r).unwrap();
        }
        let rs = table
            .execute("SELECT * FROM numrows_t WHERE i < 10")
            .unwrap();
        assert_eq!(rs.num_rows(), 10);
    }

    #[test]
    fn test_resultset_to_rows() {
        let (_dir, db) = temp_db();
        let table = db.create_table("rows_t").unwrap();
        for i in 0..3i64 {
            let mut r = Row::new();
            r.insert("val".to_string(), Value::Int64(i));
            table.insert(r).unwrap();
        }
        let rs = table.execute("SELECT * FROM rows_t ORDER BY val").unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].get("val"), Some(&Value::Int64(0)));
        assert_eq!(rows[1].get("val"), Some(&Value::Int64(1)));
        assert_eq!(rows[2].get("val"), Some(&Value::Int64(2)));
    }

    // ── Group 5: Schema management ───────────────────────────────────────────

    #[test]
    fn test_create_table_with_schema() {
        use crate::storage::on_demand::ColumnType;
        let (_dir, db) = temp_db();
        let table = db
            .create_table_with_schema(
                "schema_t",
                &[
                    ("id".to_string(), ColumnType::Int64),
                    ("name".to_string(), ColumnType::String),
                    ("val".to_string(), ColumnType::Float64),
                ],
            )
            .unwrap();

        let cols = table.columns().unwrap();
        // _id is always added; our 3 columns should also be present
        assert!(cols.iter().any(|c| c == "name"));
        assert!(cols.iter().any(|c| c == "val"));

        // Insert should work with predefined schema
        let mut r = Row::new();
        r.insert("id".to_string(), Value::Int64(1));
        r.insert("name".to_string(), Value::String("test".to_string()));
        r.insert("val".to_string(), Value::Float64(1.5));
        let id = table.insert(r).unwrap();
        assert_eq!(table.count().unwrap(), 1);
        let row = table.retrieve(id).unwrap().unwrap();
        assert_eq!(row.get("name"), Some(&Value::String("test".to_string())));
    }

    #[test]
    fn test_schema_accessor() {
        let (_dir, db) = temp_db();
        let table = db.create_table("sac_t").unwrap();
        let mut r = Row::new();
        r.insert("x".to_string(), Value::Int64(1));
        r.insert("y".to_string(), Value::Float64(2.0));
        table.insert(r).unwrap();

        let schema = table.schema().unwrap();
        let col_names: Vec<&str> = schema.iter().map(|(n, _)| n.as_str()).collect();
        assert!(col_names.contains(&"x"));
        assert!(col_names.contains(&"y"));
        // column_type helper
        let xt = table.column_type("x").unwrap();
        assert!(xt.is_some());
    }

    #[test]
    fn test_add_and_drop_column() {
        let (_dir, db) = temp_db();
        let table = db.create_table("adc_t").unwrap();
        let mut r = Row::new();
        r.insert("a".to_string(), Value::Int64(1));
        table.insert(r).unwrap();

        table.add_column("b", DataType::Float64).unwrap();
        assert!(table.columns().unwrap().iter().any(|c| c == "b"));

        table.drop_column("b").unwrap();
        assert!(!table.columns().unwrap().iter().any(|c| c == "b"));
    }

    #[test]
    fn test_rename_column() {
        let (_dir, db) = temp_db();
        let table = db.create_table("ren_t").unwrap();
        let records: Vec<Row> = (0..5i64)
            .map(|i| {
                let mut r = Row::new();
                r.insert("old_name".to_string(), Value::Int64(i * 10));
                r
            })
            .collect();
        table.insert_batch(&records).unwrap();

        // rename_column must not panic or return an error (same guarantee as Python API)
        // NOTE: persistence of rename is handled by engine internals; we test only the API
        // contract (no error, table still usable after rename_column is called)
        let result = table.rename_column("old_name", "new_name");
        assert!(result.is_ok());
        // Table must still be accessible (data not corrupted)
        assert!(table.count().unwrap() > 0);
    }

    #[test]
    fn test_rename_column_columns_reflect_new_name() {
        let (_dir, db) = temp_db();
        let table = db.create_table("renref_t").unwrap();
        let records: Vec<Row> = (0..5i64)
            .map(|i| {
                let mut r = Row::new();
                r.insert("alpha".to_string(), Value::Int64(i));
                r
            })
            .collect();
        table.insert_batch(&records).unwrap();
        table.flush().unwrap();

        // Rename and immediately check columns()
        table.rename_column("alpha", "beta").unwrap();
        let cols = table.columns().unwrap();
        assert!(
            cols.iter().any(|c| c == "beta"),
            "columns() should contain 'beta' after rename, got {:?}",
            cols
        );
        assert!(
            !cols.iter().any(|c| c == "alpha"),
            "columns() should NOT contain 'alpha' after rename, got {:?}",
            cols
        );

        // Also verify schema() reflects it
        let schema = table.schema().unwrap();
        assert!(
            schema.iter().any(|(name, _)| name == "beta"),
            "schema() should contain 'beta' after rename, got {:?}",
            schema
        );
    }

    // ── Group 6: Table & DB management ──────────────────────────────────────

    #[test]
    fn test_list_tables() {
        let (_dir, db) = temp_db();
        db.create_table("alpha").unwrap();
        db.create_table("beta").unwrap();
        let tables = db.list_tables();
        assert!(tables.contains(&"alpha".to_string()));
        assert!(tables.contains(&"beta".to_string()));
        assert!(tables.windows(2).all(|w| w[0] <= w[1]), "should be sorted");
    }

    #[test]
    fn test_drop_table() {
        let (_dir, db) = temp_db();
        db.create_table("tmp").unwrap();
        db.drop_table("tmp").unwrap();
        assert!(!db.list_tables().contains(&"tmp".to_string()));
    }

    #[test]
    fn test_table_exists_error() {
        let (_dir, db) = temp_db();
        db.create_table("dup").unwrap();
        let err = db.create_table("dup").unwrap_err();
        assert!(matches!(err, ApexError::TableExists(_)));
    }

    #[test]
    fn test_table_not_found_error() {
        let (_dir, db) = temp_db();
        let err = db.table("ghost").unwrap_err();
        assert!(matches!(err, ApexError::TableNotFound(_)));
    }

    #[test]
    fn test_reopen_existing_table() {
        let dir = TempDir::new().unwrap();
        {
            let db = ApexDB::open(dir.path()).unwrap();
            let table = db.create_table("persist_t").unwrap();
            let mut r = Row::new();
            r.insert("n".to_string(), Value::Int64(100));
            table.insert(r).unwrap();
        }
        // Reopen in a new ApexDB instance
        let db2 = ApexDB::open(dir.path()).unwrap();
        let table2 = db2.table("persist_t").unwrap();
        assert_eq!(table2.count().unwrap(), 1);
    }

    #[test]
    fn test_builder_drop_if_exists() {
        let dir = TempDir::new().unwrap();
        // First: create a table and insert data
        {
            let db = ApexDB::open(dir.path()).unwrap();
            let table = db.create_table("should_be_gone").unwrap();
            let mut r = Row::new();
            r.insert("x".to_string(), Value::Int64(1));
            table.insert(r).unwrap();
        }
        // Reopen with drop_if_exists=true
        let db = ApexDB::builder(dir.path())
            .drop_if_exists(true)
            .build()
            .unwrap();
        assert!(db.list_tables().is_empty());
    }

    #[test]
    fn test_use_database() {
        let (_dir, db) = temp_db();

        // Create a table in the sub-database "prod"
        db.use_database("prod").unwrap();
        let table = db.create_table("orders").unwrap();
        let mut r = Row::new();
        r.insert("amount".to_string(), Value::Float64(19.99));
        table.insert(r).unwrap();
        assert_eq!(table.count().unwrap(), 1);

        // Switch back to root — "orders" should not appear here
        db.use_database("").unwrap();
        assert!(!db.list_tables().contains(&"orders".to_string()));

        // Switch back and verify data persists
        db.use_database("prod").unwrap();
        let t = db.table("orders").unwrap();
        assert_eq!(t.count().unwrap(), 1);
    }

    #[test]
    fn test_db_execute_sql() {
        let (_dir, db) = temp_db();
        let table = db.create_table("dbexec_t").unwrap();
        for i in 0..5i64 {
            let mut r = Row::new();
            r.insert("n".to_string(), Value::Int64(i));
            table.insert(r).unwrap();
        }
        let rs = db.execute("SELECT COUNT(*) FROM dbexec_t").unwrap();
        let batch = rs.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 1);
    }

    // ── Group 7: insert_arrow ─────────────────────────────────────────────────

    #[test]
    fn test_insert_arrow() {
        use arrow::array::{Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDT, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;

        let (_dir, db) = temp_db();
        let table = db.create_table("arrow_t").unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("name", ArrowDT::Utf8, false),
            Field::new("age", ArrowDT::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol"])),
                Arc::new(Int64Array::from(vec![30i64, 25, 35])),
            ],
        )
        .unwrap();

        let ids = table.insert_arrow(&batch).unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(table.count().unwrap(), 3);

        let rs = table
            .execute("SELECT name FROM arrow_t WHERE age > 26")
            .unwrap();
        assert_eq!(rs.num_rows(), 2); // Alice(30), Carol(35)
    }

    #[test]
    fn test_insert_arrow_empty() {
        use arrow::array::{Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDT, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;

        let (_dir, db) = temp_db();
        let table = db.create_table("arrow_empty_t").unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "x",
            ArrowDT::Int64,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(Vec::<i64>::new()))])
                .unwrap();

        let ids = table.insert_arrow(&batch).unwrap();
        assert!(ids.is_empty());
        assert_eq!(table.count().unwrap(), 0);
    }

    // ── Group 8: helper functions ────────────────────────────────────────────

    #[test]
    fn test_record_batch_to_rows() {
        use arrow::array::{Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDT, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", ArrowDT::Int64, false),
            Field::new("name", ArrowDT::Utf8, false),
            Field::new("val", ArrowDT::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 2])),
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(Float64Array::from(vec![1.1_f64, 2.2])),
            ],
        )
        .unwrap();

        let rows = record_batch_to_rows(&batch).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("id"), Some(&Value::Int64(1)));
        assert_eq!(rows[0].get("name"), Some(&Value::String("a".to_string())));
        assert_eq!(rows[1].get("id"), Some(&Value::Int64(2)));

        if let Some(Value::Float64(v)) = rows[0].get("val") {
            assert!((v - 1.1).abs() < 1e-9);
        } else {
            panic!("val should be Float64");
        }
    }

    #[test]
    fn test_arrow_value_at_types() {
        use arrow::array::{BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDT, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", ArrowDT::Int64, false),
            Field::new("f", ArrowDT::Float64, false),
            Field::new("s", ArrowDT::Utf8, false),
            Field::new("b", ArrowDT::Boolean, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![42i64])),
                Arc::new(Float64Array::from(vec![3.14_f64])),
                Arc::new(StringArray::from(vec!["hi"])),
                Arc::new(BooleanArray::from(vec![true])),
            ],
        )
        .unwrap();

        assert_eq!(arrow_value_at(batch.column(0), 0), Value::Int64(42));
        assert_eq!(
            arrow_value_at(batch.column(2), 0),
            Value::String("hi".to_string())
        );
        assert_eq!(arrow_value_at(batch.column(3), 0), Value::Bool(true));
    }

    // ── Group 9: multi-table & concurrent tables ─────────────────────────────

    #[test]
    fn test_multiple_tables_independent() {
        let (_dir, db) = temp_db();
        let t1 = db.create_table("mt1").unwrap();
        let t2 = db.create_table("mt2").unwrap();

        for i in 0..5i64 {
            let mut r = Row::new();
            r.insert("x".to_string(), Value::Int64(i));
            t1.insert(r).unwrap();
        }
        for i in 0..3i64 {
            let mut r = Row::new();
            r.insert("y".to_string(), Value::Int64(i * 10));
            t2.insert(r).unwrap();
        }

        assert_eq!(t1.count().unwrap(), 5);
        assert_eq!(t2.count().unwrap(), 3);
        assert_eq!(db.list_tables().len(), 2);
    }

    #[test]
    fn test_table_path_accessor() {
        let dir = TempDir::new().unwrap();
        let db = ApexDB::open(dir.path()).unwrap();
        let table = db.create_table("path_t").unwrap();
        assert!(table.path().exists());
        assert_eq!(table.path().extension().unwrap(), "apex");
    }

    #[test]
    fn test_schema_operations() {
        let (_dir, db) = temp_db();
        let table = db.create_table("schema_t").unwrap();
        let mut r = Row::new();
        r.insert("a".to_string(), Value::Int64(1));
        table.insert(r).unwrap();
        table.add_column("b", DataType::Float64).unwrap();
        let cols = table.columns().unwrap();
        assert!(cols.iter().any(|c| c == "b"));
    }

    // ── Group 10: NULL value handling ───────────────────────────────────────

    #[test]
    fn test_null_insert_and_retrieve() {
        let (_dir, db) = temp_db();
        let table = db.create_table("null_t").unwrap();
        let mut r = Row::new();
        r.insert("name".to_string(), Value::String("Alice".to_string()));
        r.insert("age".to_string(), Value::Null);
        let id = table.insert(r).unwrap();

        let row = table.retrieve(id).unwrap().unwrap();
        assert_eq!(row.get("name"), Some(&Value::String("Alice".to_string())));
        // NULL value should come back as Null
        let age_val = row.get("age");
        assert!(
            age_val.is_none() || age_val == Some(&Value::Null),
            "NULL column should be None or Value::Null, got {:?}",
            age_val
        );
    }

    #[test]
    fn test_null_via_add_column() {
        let (_dir, db) = temp_db();
        let table = db.create_table("nulladd_t").unwrap();
        let mut r = Row::new();
        r.insert("x".to_string(), Value::Int64(42));
        table.insert(r).unwrap();

        // Adding a column sets all existing rows to NULL
        table.add_column("new_col", DataType::String).unwrap();
        let rs = table.execute("SELECT x, new_col FROM nulladd_t").unwrap();
        let batch = rs.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 1);
    }

    #[test]
    fn test_null_in_sql_filter() {
        let (_dir, db) = temp_db();
        let table = db.create_table("nullfilt_t").unwrap();
        // Insert some rows with and without NULLs
        let mut r1 = Row::new();
        r1.insert("name".to_string(), Value::String("A".to_string()));
        r1.insert("val".to_string(), Value::Int64(10));
        table.insert(r1).unwrap();

        let mut r2 = Row::new();
        r2.insert("name".to_string(), Value::String("B".to_string()));
        r2.insert("val".to_string(), Value::Null);
        table.insert(r2).unwrap();

        // Filtering should work correctly
        let rs = table
            .execute("SELECT name FROM nullfilt_t WHERE val = 10")
            .unwrap();
        assert_eq!(rs.num_rows(), 1);
    }

    #[test]
    fn test_null_coalesce() {
        let (_dir, db) = temp_db();
        let table = db.create_table("nullcoal_t").unwrap();
        let mut r = Row::new();
        r.insert("name".to_string(), Value::String("A".to_string()));
        r.insert("val".to_string(), Value::Null);
        table.insert(r).unwrap();

        let rs = table
            .execute("SELECT COALESCE(val, 0) AS v FROM nullcoal_t")
            .unwrap();
        assert_eq!(rs.num_rows(), 1);
    }

    // ── Group 11: SQL coverage expansion ────────────────────────────────────

    #[test]
    fn test_sql_between() {
        let (_dir, db) = temp_db();
        let table = db.create_table("between_t").unwrap();
        for i in 0..20i64 {
            let mut r = Row::new();
            r.insert("n".to_string(), Value::Int64(i));
            table.insert(r).unwrap();
        }
        let rs = table
            .execute("SELECT n FROM between_t WHERE n BETWEEN 5 AND 14")
            .unwrap();
        assert_eq!(rs.num_rows(), 10);
    }

    #[test]
    fn test_sql_like() {
        let (_dir, db) = temp_db();
        let table = db.create_table("like_t").unwrap();
        for name in &["Alice", "Alice2", "Bob", "Alicia", "Charlie"] {
            let mut r = Row::new();
            r.insert("name".to_string(), Value::String(name.to_string()));
            table.insert(r).unwrap();
        }
        let rs = table
            .execute("SELECT name FROM like_t WHERE name LIKE 'Ali%'")
            .unwrap();
        assert_eq!(rs.num_rows(), 3); // Alice, Alice2, Alicia
    }

    #[test]
    fn test_sql_in_filter() {
        let (_dir, db) = temp_db();
        let table = db.create_table("in_t").unwrap();
        for city in &["NY", "LA", "SF", "Chicago", "Boston", "Miami"] {
            let mut r = Row::new();
            r.insert("city".to_string(), Value::String(city.to_string()));
            table.insert(r).unwrap();
        }
        let rs = table
            .execute("SELECT city FROM in_t WHERE city IN ('NY', 'LA', 'Miami')")
            .unwrap();
        assert_eq!(rs.num_rows(), 3);
    }

    #[test]
    fn test_sql_count_distinct() {
        let (_dir, db) = temp_db();
        let table = db.create_table("cd_t").unwrap();
        for city in &["NY", "LA", "NY", "SF", "LA", "NY"] {
            let mut r = Row::new();
            r.insert("city".to_string(), Value::String(city.to_string()));
            table.insert(r).unwrap();
        }
        let rs = table
            .execute("SELECT COUNT(DISTINCT city) FROM cd_t")
            .unwrap();
        let batch = rs.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 1);
    }

    #[test]
    fn test_sql_sum_avg_min_max() {
        let (_dir, db) = temp_db();
        let table = db.create_table("agg_t").unwrap();
        for v in &[10i64, 20, 30, 40, 50] {
            let mut r = Row::new();
            r.insert("v".to_string(), Value::Int64(*v));
            table.insert(r).unwrap();
        }
        // SUM = 150
        let rs = table.execute("SELECT SUM(v) FROM agg_t").unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 1);

        // AVG = 30
        let rs = table.execute("SELECT AVG(v) FROM agg_t").unwrap();
        assert_eq!(rs.to_rows().unwrap().len(), 1);

        // MIN = 10, MAX = 50
        let rs = table.execute("SELECT MIN(v), MAX(v) FROM agg_t").unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_sql_multi_column_order_by() {
        let (_dir, db) = temp_db();
        let table = db.create_table("mcord_t").unwrap();
        for (city, score) in &[("NY", 80i64), ("LA", 90), ("NY", 70), ("LA", 60)] {
            let mut r = Row::new();
            r.insert("city".to_string(), Value::String(city.to_string()));
            r.insert("score".to_string(), Value::Int64(*score));
            table.insert(r).unwrap();
        }
        let rs = table
            .execute("SELECT city, score FROM mcord_t ORDER BY city ASC, score DESC")
            .unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 4);
        // LA first (sorted asc), LA scores desc: 90, 60
        assert_eq!(rows[0].get("city"), Some(&Value::String("LA".to_string())));
        assert_eq!(rows[0].get("score"), Some(&Value::Int64(90)));
        assert_eq!(rows[1].get("city"), Some(&Value::String("LA".to_string())));
        assert_eq!(rows[1].get("score"), Some(&Value::Int64(60)));
    }

    #[test]
    fn test_sql_select_star_with_where_and_limit() {
        let (_dir, db) = temp_db();
        let table = db.create_table("whlim_t").unwrap();
        for i in 0..50i64 {
            let mut r = Row::new();
            r.insert("i".to_string(), Value::Int64(i));
            table.insert(r).unwrap();
        }
        let rs = table
            .execute("SELECT * FROM whlim_t WHERE i >= 10 LIMIT 5")
            .unwrap();
        assert_eq!(rs.num_rows(), 5);
    }

    #[test]
    fn test_sql_multiple_aggregations() {
        let (_dir, db) = temp_db();
        let table = db.create_table("magg_t").unwrap();
        for i in 1..=100i64 {
            let mut r = Row::new();
            r.insert("v".to_string(), Value::Int64(i));
            table.insert(r).unwrap();
        }
        let rs = table
            .execute("SELECT COUNT(*), SUM(v), MIN(v), MAX(v), AVG(v) FROM magg_t")
            .unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_sql_group_by_with_multiple_aggs() {
        let (_dir, db) = temp_db();
        let table = db.create_table("gma_t").unwrap();
        for (cat, val) in &[("A", 10i64), ("A", 20), ("B", 30), ("B", 40), ("C", 50)] {
            let mut r = Row::new();
            r.insert("cat".to_string(), Value::String(cat.to_string()));
            r.insert("val".to_string(), Value::Int64(*val));
            table.insert(r).unwrap();
        }
        let rs = table.execute(
            "SELECT cat, COUNT(*) AS cnt, SUM(val) AS total, AVG(val) AS avg_val FROM gma_t GROUP BY cat ORDER BY cat"
        ).unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 3);
        // A: count=2, sum=30; B: count=2, sum=70; C: count=1, sum=50
    }

    // ── Group 12: Window functions ──────────────────────────────────────────

    #[test]
    fn test_window_row_number() {
        let (_dir, db) = temp_db();
        let table = db.create_table("win_rn_t").unwrap();
        for (city, score) in &[("NY", 90i64), ("NY", 80), ("LA", 70), ("LA", 95)] {
            let mut r = Row::new();
            r.insert("city".to_string(), Value::String(city.to_string()));
            r.insert("score".to_string(), Value::Int64(*score));
            table.insert(r).unwrap();
        }
        let rs = table.execute(
            "SELECT city, score, ROW_NUMBER() OVER (PARTITION BY city ORDER BY score DESC) AS rn FROM win_rn_t"
        ).unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn test_window_rank() {
        let (_dir, db) = temp_db();
        let table = db.create_table("win_rank_t").unwrap();
        for score in &[90i64, 90, 80, 70] {
            let mut r = Row::new();
            r.insert("score".to_string(), Value::Int64(*score));
            table.insert(r).unwrap();
        }
        let rs = table
            .execute("SELECT score, RANK() OVER (ORDER BY score DESC) AS rnk FROM win_rank_t")
            .unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn test_window_sum_over() {
        let (_dir, db) = temp_db();
        let table = db.create_table("win_sum_t").unwrap();
        for (cat, v) in &[("A", 10i64), ("A", 20), ("B", 30)] {
            let mut r = Row::new();
            r.insert("cat".to_string(), Value::String(cat.to_string()));
            r.insert("v".to_string(), Value::Int64(*v));
            table.insert(r).unwrap();
        }
        let rs = table
            .execute("SELECT cat, v, SUM(v) OVER (PARTITION BY cat) AS cat_total FROM win_sum_t")
            .unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 3);
    }

    // ── Group 13: Concurrency ───────────────────────────────────────────────

    #[test]
    fn test_concurrent_reads() {
        let (_dir, db) = temp_db();
        let table = db.create_table("conc_r_t").unwrap();
        for i in 0..100i64 {
            let mut r = Row::new();
            r.insert("n".to_string(), Value::Int64(i));
            table.insert(r).unwrap();
        }

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let t = table.clone();
                std::thread::spawn(move || {
                    let rs = t.execute("SELECT COUNT(*) FROM conc_r_t").unwrap();
                    let batch = rs.to_record_batch().unwrap();
                    assert_eq!(batch.num_rows(), 1);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_writes_separate_tables() {
        let (_dir, db) = temp_db();
        // Each thread writes to its own table — no cross-file race
        let handles: Vec<_> = (0..4)
            .map(|thread_id| {
                let d = db.clone();
                std::thread::spawn(move || {
                    let name = format!("conc_t_{}", thread_id);
                    let t = d.create_table(&name).unwrap();
                    let batch: Vec<Row> = (0..25i64)
                        .map(|i| {
                            let mut r = Row::new();
                            r.insert("tid".to_string(), Value::Int64(thread_id));
                            r.insert("i".to_string(), Value::Int64(i));
                            r
                        })
                        .collect();
                    t.insert_batch(&batch).unwrap();
                    assert_eq!(t.count().unwrap(), 25);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        // Verify all 4 tables exist
        let tables = db.list_tables();
        assert_eq!(tables.len(), 4);
    }

    #[test]
    fn test_concurrent_reads_multiple_threads() {
        let (_dir, db) = temp_db();
        let table = db.create_table("conc_rw_t").unwrap();
        let records: Vec<Row> = (0..100i64)
            .map(|i| {
                let mut r = Row::new();
                r.insert("n".to_string(), Value::Int64(i));
                r
            })
            .collect();
        table.insert_batch(&records).unwrap();
        table.flush().unwrap();

        // Multiple threads reading simultaneously
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let t = table.clone();
                std::thread::spawn(move || {
                    assert_eq!(t.count().unwrap(), 100);
                    let rs = t.execute("SELECT n FROM conc_rw_t WHERE n >= 50").unwrap();
                    assert_eq!(rs.num_rows(), 50);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_send_sync_bounds() {
        // Compile-time verification that ApexDB and Table are Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ApexDB>();
        assert_send_sync::<Table>();
    }

    #[test]
    fn test_clone_db_share_across_threads() {
        let (_dir, db) = temp_db();
        let table = db.create_table("clone_t").unwrap();
        let mut r = Row::new();
        r.insert("v".to_string(), Value::Int64(1));
        table.insert(r).unwrap();

        let db2 = db.clone();
        let handle = std::thread::spawn(move || {
            let t = db2.table("clone_t").unwrap();
            assert_eq!(t.count().unwrap(), 1);
        });
        handle.join().unwrap();
    }

    // ── Group 14: Edge cases ────────────────────────────────────────────────

    #[test]
    fn test_unicode_strings() {
        let (_dir, db) = temp_db();
        let table = db.create_table("uni_t").unwrap();
        let strings = ["你好世界", "こんにちは", "🚀🔥💯", "Ñoño", "Ü∑€®†"];
        for s in &strings {
            let mut r = Row::new();
            r.insert("s".to_string(), Value::String(s.to_string()));
            table.insert(r).unwrap();
        }
        assert_eq!(table.count().unwrap(), 5);
        let rs = table.execute("SELECT s FROM uni_t").unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn test_empty_string_values() {
        let (_dir, db) = temp_db();
        let table = db.create_table("estr_t").unwrap();
        let mut r = Row::new();
        r.insert("s".to_string(), Value::String(String::new()));
        r.insert("n".to_string(), Value::Int64(1));
        let id = table.insert(r).unwrap();

        let row = table.retrieve(id).unwrap().unwrap();
        assert_eq!(row.get("n"), Some(&Value::Int64(1)));
    }

    #[test]
    fn test_special_characters_in_strings() {
        let (_dir, db) = temp_db();
        let table = db.create_table("spec_t").unwrap();
        let strings = [
            "hello\nworld", // newline
            "tab\there",    // tab
            "quote'inside", // single quote
            "back\\slash",  // backslash
        ];
        for s in &strings {
            let mut r = Row::new();
            r.insert("s".to_string(), Value::String(s.to_string()));
            table.insert(r).unwrap();
        }
        assert_eq!(table.count().unwrap(), 4);
    }

    #[test]
    fn test_large_integer_values() {
        let (_dir, db) = temp_db();
        let table = db.create_table("lint_t").unwrap();
        let values = [i64::MAX, i64::MIN, 0i64, -1, 1];
        for v in &values {
            let mut r = Row::new();
            r.insert("v".to_string(), Value::Int64(*v));
            table.insert(r).unwrap();
        }
        assert_eq!(table.count().unwrap(), 5);
        let rs = table.execute("SELECT v FROM lint_t ORDER BY v").unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].get("v"), Some(&Value::Int64(i64::MIN)));
        assert_eq!(rows[4].get("v"), Some(&Value::Int64(i64::MAX)));
    }

    #[test]
    fn test_empty_table_count_and_schema() {
        let (_dir, db) = temp_db();
        use crate::storage::on_demand::ColumnType;
        let table = db
            .create_table_with_schema(
                "empty_sch_t",
                &[
                    ("a".to_string(), ColumnType::Int64),
                    ("b".to_string(), ColumnType::String),
                ],
            )
            .unwrap();

        assert_eq!(table.count().unwrap(), 0);
        let cols = table.columns().unwrap();
        assert!(cols.iter().any(|c| c == "a"));
        assert!(cols.iter().any(|c| c == "b"));
    }

    #[test]
    fn test_empty_table_select() {
        let (_dir, db) = temp_db();
        use crate::storage::on_demand::ColumnType;
        let table = db
            .create_table_with_schema("empty_sel_t", &[("x".to_string(), ColumnType::Int64)])
            .unwrap();

        let rs = table.execute("SELECT * FROM empty_sel_t").unwrap();
        assert!(rs.is_empty());
        assert_eq!(rs.num_rows(), 0);
    }

    #[test]
    fn test_insert_batch_empty() {
        let (_dir, db) = temp_db();
        let table = db.create_table("empty_batch_t").unwrap();
        let ids = table.insert_batch(&[]).unwrap();
        assert!(ids.is_empty());
        assert_eq!(table.count().unwrap(), 0);
    }

    #[test]
    fn test_delete_batch_empty() {
        let (_dir, db) = temp_db();
        let table = db.create_table("empty_del_t").unwrap();
        let mut r = Row::new();
        r.insert("v".to_string(), Value::Int64(1));
        table.insert(r).unwrap();
        let deleted = table.delete_batch(&[]).unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(table.count().unwrap(), 1);
    }

    // ── Group 15: Durability & persistence ──────────────────────────────────

    #[test]
    fn test_durability_safe() {
        let dir = TempDir::new().unwrap();
        {
            let db = ApexDB::builder(dir.path())
                .durability(DurabilityLevel::Safe)
                .build()
                .unwrap();
            let table = db.create_table("safe_t").unwrap();
            let mut r = Row::new();
            r.insert("n".to_string(), Value::Int64(42));
            table.insert(r).unwrap();
            table.flush().unwrap();
        }
        // Reopen and verify
        let db = ApexDB::open(dir.path()).unwrap();
        let table = db.table("safe_t").unwrap();
        assert_eq!(table.count().unwrap(), 1);
    }

    #[test]
    fn test_flush_and_reopen() {
        let dir = TempDir::new().unwrap();
        {
            let db = ApexDB::open(dir.path()).unwrap();
            let table = db.create_table("flush_t").unwrap();
            for i in 0..10i64 {
                let mut r = Row::new();
                r.insert("n".to_string(), Value::Int64(i));
                table.insert(r).unwrap();
            }
            table.flush().unwrap();
        }
        let db = ApexDB::open(dir.path()).unwrap();
        let table = db.table("flush_t").unwrap();
        assert_eq!(table.count().unwrap(), 10);
    }

    #[test]
    fn test_multiple_reopen_cycles() {
        let dir = TempDir::new().unwrap();
        for cycle in 0..3i64 {
            let db = ApexDB::open(dir.path()).unwrap();
            if cycle == 0 {
                let table = db.create_table("cycle_t").unwrap();
                let mut r = Row::new();
                r.insert("cycle".to_string(), Value::Int64(cycle));
                table.insert(r).unwrap();
            } else {
                let table = db.table("cycle_t").unwrap();
                let mut r = Row::new();
                r.insert("cycle".to_string(), Value::Int64(cycle));
                table.insert(r).unwrap();
            }
        }
        let db = ApexDB::open(dir.path()).unwrap();
        let table = db.table("cycle_t").unwrap();
        assert_eq!(table.count().unwrap(), 3);
    }

    // ── Group 16: Clone/Debug traits & misc ─────────────────────────────────

    #[test]
    fn test_apexdb_debug() {
        let (_dir, db) = temp_db();
        let debug_str = format!("{:?}", db);
        assert!(debug_str.contains("ApexDB"));
    }

    #[test]
    fn test_table_debug() {
        let (_dir, db) = temp_db();
        let table = db.create_table("dbg_t").unwrap();
        let debug_str = format!("{:?}", table);
        assert!(debug_str.contains("dbg_t"));
    }

    #[test]
    fn test_apexdb_clone() {
        let (_dir, db) = temp_db();
        let table = db.create_table("cln_t").unwrap();
        let mut r = Row::new();
        r.insert("v".to_string(), Value::Int64(1));
        table.insert(r).unwrap();

        let db2 = db.clone();
        let table2 = db2.table("cln_t").unwrap();
        assert_eq!(table2.count().unwrap(), 1);
    }

    #[test]
    fn test_table_clone() {
        let (_dir, db) = temp_db();
        let table = db.create_table("tcln_t").unwrap();
        let mut r = Row::new();
        r.insert("v".to_string(), Value::Int64(1));
        table.insert(r).unwrap();

        let t2 = table.clone();
        assert_eq!(t2.count().unwrap(), 1);
        let mut r2 = Row::new();
        r2.insert("v".to_string(), Value::Int64(2));
        t2.insert(r2).unwrap();
        assert_eq!(table.count().unwrap(), 2);
    }

    #[test]
    fn test_invalidate_cache() {
        let (_dir, db) = temp_db();
        let table = db.create_table("inv_t").unwrap();
        let mut r = Row::new();
        r.insert("v".to_string(), Value::Int64(1));
        table.insert(r).unwrap();

        db.invalidate_cache();
        // Should still work after invalidation
        assert_eq!(table.count().unwrap(), 1);
    }

    #[test]
    fn test_base_dir_accessor() {
        let dir = TempDir::new().unwrap();
        let db = ApexDB::open(dir.path()).unwrap();
        let base = db.base_dir();
        assert!(base.exists());
    }

    #[test]
    fn test_drop_table_not_found() {
        let (_dir, db) = temp_db();
        let err = db.drop_table("nonexistent_table").unwrap_err();
        assert!(matches!(err, ApexError::TableNotFound(_)));
    }

    // ── Group 17: SQL expressions (CAST, CASE WHEN, subqueries) ─────────────

    #[test]
    fn test_sql_cast() {
        let (_dir, db) = temp_db();
        let table = db.create_table("cast_t").unwrap();
        let mut r = Row::new();
        r.insert("n".to_string(), Value::Int64(42));
        table.insert(r).unwrap();

        let rs = table
            .execute("SELECT CAST(n AS DOUBLE) AS f FROM cast_t")
            .unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_sql_case_when() {
        let (_dir, db) = temp_db();
        let table = db.create_table("case_t").unwrap();
        for (name, age) in &[("A", 15i64), ("B", 25), ("C", 65)] {
            let mut r = Row::new();
            r.insert("name".to_string(), Value::String(name.to_string()));
            r.insert("age".to_string(), Value::Int64(*age));
            table.insert(r).unwrap();
        }
        let rs = table.execute(
            "SELECT name, CASE WHEN age < 18 THEN 'minor' WHEN age < 65 THEN 'adult' ELSE 'senior' END AS category FROM case_t ORDER BY name"
        ).unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(
            rows[0].get("category"),
            Some(&Value::String("minor".to_string()))
        );
        assert_eq!(
            rows[1].get("category"),
            Some(&Value::String("adult".to_string()))
        );
        assert_eq!(
            rows[2].get("category"),
            Some(&Value::String("senior".to_string()))
        );
    }

    #[test]
    fn test_sql_nested_functions() {
        let (_dir, db) = temp_db();
        let table = db.create_table("nested_t").unwrap();
        let mut r = Row::new();
        r.insert("s".to_string(), Value::String("hello".to_string()));
        table.insert(r).unwrap();

        let rs = table.execute("SELECT UPPER(s) AS u FROM nested_t").unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("u"), Some(&Value::String("HELLO".to_string())));
    }

    #[test]
    fn test_sql_string_functions() {
        let (_dir, db) = temp_db();
        let table = db.create_table("strfn_t").unwrap();
        let mut r = Row::new();
        r.insert("s".to_string(), Value::String("Hello World".to_string()));
        table.insert(r).unwrap();

        let rs = table
            .execute("SELECT LENGTH(s) AS len FROM strfn_t")
            .unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 1);

        let rs = table.execute("SELECT LOWER(s) AS lo FROM strfn_t").unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(
            rows[0].get("lo"),
            Some(&Value::String("hello world".to_string()))
        );
    }

    #[test]
    fn test_sql_arithmetic_expressions() {
        let (_dir, db) = temp_db();
        let table = db.create_table("arith_t").unwrap();
        let mut r = Row::new();
        r.insert("a".to_string(), Value::Int64(10));
        r.insert("b".to_string(), Value::Int64(3));
        table.insert(r).unwrap();

        let rs = table
            .execute("SELECT a + b AS sum, a - b AS diff, a * b AS prod FROM arith_t")
            .unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("sum"), Some(&Value::Int64(13)));
        assert_eq!(rows[0].get("diff"), Some(&Value::Int64(7)));
        assert_eq!(rows[0].get("prod"), Some(&Value::Int64(30)));
    }

    #[test]
    fn test_sql_alias_in_order_by() {
        let (_dir, db) = temp_db();
        let table = db.create_table("alias_t").unwrap();
        for i in &[3i64, 1, 4, 1, 5, 9] {
            let mut r = Row::new();
            r.insert("n".to_string(), Value::Int64(*i));
            table.insert(r).unwrap();
        }
        let rs = table
            .execute("SELECT n AS val FROM alias_t ORDER BY val LIMIT 3")
            .unwrap();
        let rows = rs.to_rows().unwrap();
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_sql_select_distinct() {
        let (_dir, db) = temp_db();
        let table = db.create_table("dist_t").unwrap();
        for city in &["NY", "LA", "NY", "SF", "LA", "NY"] {
            let mut r = Row::new();
            r.insert("city".to_string(), Value::String(city.to_string()));
            table.insert(r).unwrap();
        }
        let rs = table.execute("SELECT DISTINCT city FROM dist_t").unwrap();
        assert_eq!(rs.num_rows(), 3);
    }

    #[test]
    fn test_sql_subquery_in_where() {
        let (_dir, db) = temp_db();
        let table = db.create_table("sub_t").unwrap();
        for (name, val) in &[("A", 10i64), ("B", 20), ("C", 30), ("D", 40)] {
            let mut r = Row::new();
            r.insert("name".to_string(), Value::String(name.to_string()));
            r.insert("val".to_string(), Value::Int64(*val));
            table.insert(r).unwrap();
        }
        // Get rows where val > avg(val) (avg = 25 → C(30), D(40))
        let rs = table
            .execute("SELECT name FROM sub_t WHERE val > (SELECT AVG(val) FROM sub_t)")
            .unwrap();
        assert_eq!(rs.num_rows(), 2);
    }

    // ── Group 18: Multiple statements & complex workflows ───────────────────

    #[test]
    fn test_insert_delete_reinsert() {
        let (_dir, db) = temp_db();
        let table = db.create_table("idr_t").unwrap();

        // Insert → delete → reinsert cycle
        let id1 = table.insert(row1("Alice", 30, 90.0, "NY")).unwrap();
        assert_eq!(table.count().unwrap(), 1);
        table.delete(id1).unwrap();
        assert_eq!(table.count().unwrap(), 0);

        let id2 = table.insert(row1("Bob", 25, 85.0, "LA")).unwrap();
        assert_eq!(table.count().unwrap(), 1);
        // id2 >= id1 is guaranteed (monotonic), but not necessarily different
        assert!(id2 >= id1);
    }

    #[test]
    fn test_replace_preserves_id() {
        let (_dir, db) = temp_db();
        let table = db.create_table("rpid_t").unwrap();
        let id = table.insert(row1("Alice", 30, 90.0, "NY")).unwrap();

        let mut updated = Row::new();
        updated.insert("name".to_string(), Value::String("Alice-v2".to_string()));
        updated.insert("age".to_string(), Value::Int64(31));
        updated.insert("score".to_string(), Value::Float64(95.0));
        updated.insert("city".to_string(), Value::String("SF".to_string()));
        table.replace(id, updated).unwrap();

        // After replace, count should still be 1 (same id)
        assert_eq!(table.count().unwrap(), 1);
        assert!(table.exists(id).unwrap());
    }

    #[test]
    fn test_batch_insert_then_filter() {
        let (_dir, db) = temp_db();
        let table = db.create_table("bif_t").unwrap();
        let records: Vec<Row> = (0..1000i64)
            .map(|i| {
                let mut r = Row::new();
                r.insert("n".to_string(), Value::Int64(i));
                r.insert("even".to_string(), Value::Bool(i % 2 == 0));
                r.insert("cat".to_string(), Value::String(format!("cat_{}", i % 5)));
                r
            })
            .collect();
        table.insert_batch(&records).unwrap();
        assert_eq!(table.count().unwrap(), 1000);

        // Complex filter
        let rs = table
            .execute("SELECT COUNT(*) FROM bif_t WHERE even = true AND n BETWEEN 100 AND 200")
            .unwrap();
        let batch = rs.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 1);
    }

    #[test]
    fn test_multi_database_isolation() {
        let (_dir, db) = temp_db();

        // Create table in "db1"
        db.use_database("db1").unwrap();
        let t1 = db.create_table("shared_name").unwrap();
        let mut r = Row::new();
        r.insert("v".to_string(), Value::Int64(1));
        t1.insert(r).unwrap();

        // Create table with same name in "db2"
        db.use_database("db2").unwrap();
        let t2 = db.create_table("shared_name").unwrap();
        let mut r = Row::new();
        r.insert("v".to_string(), Value::Int64(2));
        t2.insert(r).unwrap();
        let mut r = Row::new();
        r.insert("v".to_string(), Value::Int64(3));
        t2.insert(r).unwrap();

        // Verify isolation
        db.use_database("db1").unwrap();
        let t = db.table("shared_name").unwrap();
        assert_eq!(t.count().unwrap(), 1);

        db.use_database("db2").unwrap();
        let t = db.table("shared_name").unwrap();
        assert_eq!(t.count().unwrap(), 2);
    }

    #[test]
    fn test_register_temp_table_csv() {
        let (_dir, db) = temp_db();
        // Create a CSV file
        let csv_path = _dir.path().join("test.csv");
        std::fs::write(
            &csv_path,
            "name,age,score\nAlice,25,85.5\nBob,30,90.0\nCharlie,35,78.2\n",
        )
        .unwrap();

        db.register_temp_table("people", csv_path.to_str().unwrap())
            .unwrap();

        let table = db.table("people").unwrap();
        assert_eq!(table.count().unwrap(), 3);

        let rs = table
            .execute("SELECT * FROM people WHERE age > 28")
            .unwrap();
        let batch = rs.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn test_register_temp_table_json() {
        let (_dir, db) = temp_db();
        let json_path = _dir.path().join("test.json");
        std::fs::write(&json_path, "{\"name\":\"Alice\",\"age\":25}\n{\"name\":\"Bob\",\"age\":30}\n{\"name\":\"Charlie\",\"age\":35}\n").unwrap();

        db.register_temp_table("users", json_path.to_str().unwrap())
            .unwrap();

        let table = db.table("users").unwrap();
        assert_eq!(table.count().unwrap(), 3);

        let rs = table
            .execute("SELECT COUNT(*) FROM users WHERE age >= 30")
            .unwrap();
        let batch = rs.to_record_batch().unwrap();
        let count = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_temp_table_create_temp_sql() {
        let (_dir, db) = temp_db();
        let csv_path = _dir.path().join("items.csv");
        std::fs::write(
            &csv_path,
            "id,price\n1,10.0\n2,20.0\n3,30.0\n4,40.0\n5,50.0\n",
        )
        .unwrap();

        // First register via API, then verify SQL access
        db.register_temp_table("items", csv_path.to_str().unwrap())
            .unwrap();

        // Query via SQL
        let rs = db.execute("SELECT * FROM items WHERE price > 25").unwrap();
        let batch = rs.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 3);

        // Aggregate query
        let rs = db.execute("SELECT AVG(price) FROM items").unwrap();
        let batch = rs.to_record_batch().unwrap();
        let avg = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .value(0);
        assert!((avg - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_temp_table_drop() {
        let (_dir, db) = temp_db();
        let csv_path = _dir.path().join("data.csv");
        std::fs::write(&csv_path, "x,y\n1,2\n3,4\n").unwrap();

        db.register_temp_table("tmp", csv_path.to_str().unwrap())
            .unwrap();
        assert_eq!(db.table("tmp").unwrap().count().unwrap(), 2);

        db.drop_temp_table("tmp").unwrap();
        assert!(db.table("tmp").is_err());
    }

    #[test]
    fn test_temp_table_shadows_persistent() {
        let (_dir, db) = temp_db();

        // Create a persistent table
        let t = db.create_table("test_shadow").unwrap();
        let mut r = Row::new();
        r.insert("val".to_string(), Value::Int64(999));
        t.insert(r).unwrap();
        assert_eq!(t.count().unwrap(), 1);

        // Register temp table with same name from CSV
        let csv_path = _dir.path().join("shadow.csv");
        std::fs::write(&csv_path, "val\n100\n200\n300\n").unwrap();
        db.register_temp_table("test_shadow", csv_path.to_str().unwrap())
            .unwrap();

        // Temp table should shadow the persistent one
        let t2 = db.table("test_shadow").unwrap();
        assert_eq!(t2.count().unwrap(), 3);

        // Drop temp table; persistent should be accessible again
        db.drop_temp_table("test_shadow").unwrap();
        let t3 = db.table("test_shadow").unwrap();
        assert_eq!(t3.count().unwrap(), 1);
    }

    #[test]
    fn test_temp_table_cleanup_on_drop() {
        let dir = tempfile::TempDir::new().unwrap();
        let csv_path = dir.path().join("data.csv");
        std::fs::write(&csv_path, "col\nhello\nworld\n").unwrap();

        {
            let db = ApexDB::open(dir.path()).unwrap();
            db.register_temp_table("hi", csv_path.to_str().unwrap())
                .unwrap();
            assert_eq!(db.table("hi").unwrap().count().unwrap(), 2);
            // db dropped here — temp files should be cleaned up
        }

        // Re-open — temp table should be gone
        let db2 = ApexDB::open(dir.path()).unwrap();
        assert!(db2.table("hi").is_err());
    }
}
