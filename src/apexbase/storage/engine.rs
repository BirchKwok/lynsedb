//! StorageEngine - Unified storage interface
//!
//! This module provides a single entry point for all storage operations,
//! handling complexity like caching, delta writes, and data merging internally.
//!
//! # Architecture
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │            Python Bindings                   │
//! │  store() / retrieve() / execute() / ...     │
//! └─────────────────┬───────────────────────────┘
//!                   │ Simple API
//!                   ▼
//! ┌─────────────────────────────────────────────┐
//! │           StorageEngine                      │
//! │  - Unified cache management                  │
//! │  - Automatic write routing (full/delta)     │
//! │  - Automatic read merging (base+delta)      │
//! └─────────────────┬───────────────────────────┘
//!                   │
//!                   ▼
//! ┌─────────────────────────────────────────────┐
//! │     Low-level storage (OnDemandStorage)     │
//! └─────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime};

use ahash::AHashMap;
use arrow::record_batch::RecordBatch;
use once_cell::sync::Lazy;
use parking_lot::RwLock;

use super::backend::TableStorageBackend;
use super::on_demand::ColumnType;
use super::DurabilityLevel;
use crate::data::Value;
use crate::query::ApexExecutor;

// ============================================================================
// Configuration
// ============================================================================

/// Maximum number of cached table backends
const MAX_CACHE_ENTRIES: usize = 64;

/// Delta file size threshold for auto-compaction (10MB)
const DELTA_COMPACT_SIZE: u64 = 10 * 1024 * 1024;

/// Delta row count threshold for auto-compaction
const DELTA_COMPACT_ROWS: usize = 100_000;

// ============================================================================
// Cache Entry
// ============================================================================

/// Cached table entry with metadata for LRU eviction
struct CacheEntry {
    /// The storage backend
    backend: Arc<TableStorageBackend>,
    /// File modification time when cached
    modified_time: SystemTime,
    /// Last access time for LRU eviction
    last_access: Instant,
    /// Whether there's a pending delta file
    has_delta: bool,
}

/// Lightweight schema cache entry (for fast should_use_delta checks)
struct SchemaCache {
    /// Column names in schema order
    columns: std::collections::HashSet<String>,
    /// Row count (0 means empty table, use full write)
    row_count: u64,
    /// File modification time when cached
    modified_time: SystemTime,
    /// Whether this is a V4 format file (cached to avoid repeated header reads)
    is_v4: bool,
}

// ============================================================================
// Global Engine Instance
// ============================================================================

/// Global storage engine instance (singleton pattern)
static ENGINE: Lazy<StorageEngine> = Lazy::new(StorageEngine::new);

// ============================================================================
// StorageEngine
// ============================================================================

/// Unified storage engine that handles all read/write operations
///
/// # Features
/// - **Unified caching**: Single cache for all table backends
/// - **Automatic write routing**: Chooses full write or delta write based on conditions
/// - **Automatic read merging**: Transparently merges base and delta data
/// - **Simple API**: Hides all complexity from callers
pub struct StorageEngine {
    /// Unified cache for all table backends
    cache: RwLock<AHashMap<PathBuf, CacheEntry>>,
    /// Lightweight schema cache for fast should_use_delta checks
    schema_cache: RwLock<AHashMap<PathBuf, SchemaCache>>,
    /// Cache for insert-mode backends (for delta writes) - avoids repeated file I/O
    insert_cache: RwLock<AHashMap<PathBuf, Arc<TableStorageBackend>>>,
}

impl StorageEngine {
    /// Create a new storage engine
    fn new() -> Self {
        Self {
            cache: RwLock::new(AHashMap::with_capacity(MAX_CACHE_ENTRIES)),
            schema_cache: RwLock::new(AHashMap::with_capacity(MAX_CACHE_ENTRIES * 2)),
            insert_cache: RwLock::new(AHashMap::with_capacity(MAX_CACHE_ENTRIES)),
        }
    }

    /// Get the global engine instance
    pub fn global() -> &'static StorageEngine {
        &ENGINE
    }

    // ========================================================================
    // Cache Management
    // ========================================================================

    /// Evict least recently used entries if cache is full
    fn evict_lru_if_needed(cache: &mut AHashMap<PathBuf, CacheEntry>) {
        while cache.len() >= MAX_CACHE_ENTRIES {
            // Find LRU entry
            let lru_key = cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(k, _)| k.clone());

            if let Some(key) = lru_key {
                cache.remove(&key);
            } else {
                break;
            }
        }
    }

    /// Check if delta file exists for a table
    fn has_delta_file(table_path: &Path) -> bool {
        let delta_path = Self::delta_path(table_path);
        delta_path.exists()
    }

    /// Get delta file path for a table
    fn delta_path(table_path: &Path) -> PathBuf {
        let mut delta = table_path.to_path_buf();
        let name = delta.file_name().unwrap_or_default().to_string_lossy();
        delta.set_file_name(format!("{}.delta", name));
        delta
    }

    /// Get file modification time
    fn get_modified_time(path: &Path) -> SystemTime {
        std::fs::metadata(path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH)
    }

    /// Invalidate cache for a specific table
    pub fn invalidate(&self, table_path: &Path) {
        // Use path directly without canonicalize for speed (already absolute in most cases)
        self.cache.write().remove(table_path);
        self.insert_cache.write().remove(table_path);
        self.schema_cache.write().remove(table_path);

        // Also invalidate executor cache
        ApexExecutor::invalidate_cache_for_path(table_path);
    }

    /// Invalidate read/query caches after an append while keeping the insert backend warm.
    ///
    /// The insert backend owns the just-updated footer/next-id state, so dropping it after
    /// every small append defeats the write fast path. Read caches still need to be cleared
    /// so subsequent queries see the newly appended row group.
    #[inline]
    fn invalidate_after_append(&self, table_path: &Path) {
        self.cache.write().remove(table_path);
        self.schema_cache.write().remove(table_path);
        ApexExecutor::invalidate_cache_for_path(table_path);
    }

    /// Invalidate all caches under a directory
    pub fn invalidate_dir(&self, dir: &Path) {
        // Use path directly without canonicalize for speed
        self.cache.write().retain(|path, _| !path.starts_with(dir));
        self.schema_cache
            .write()
            .retain(|path, _| !path.starts_with(dir));
        self.insert_cache
            .write()
            .retain(|path, _| !path.starts_with(dir));

        // Also invalidate executor cache
        ApexExecutor::invalidate_cache_for_dir(dir);
    }

    // ========================================================================
    // Backend Access
    // ========================================================================

    /// Get or create a backend for writing (loads existing data)
    ///
    /// This method:
    /// 1. Checks cache for existing backend
    /// 2. If cached and fresh, returns it
    /// 3. If delta exists, compacts it first
    /// 4. Opens fresh backend and caches it
    pub fn get_write_backend(
        &self,
        table_path: &Path,
        durability: DurabilityLevel,
    ) -> io::Result<Arc<TableStorageBackend>> {
        // Use path directly - avoid expensive canonicalize
        let cache_key = table_path.to_path_buf();
        let has_delta = Self::has_delta_file(table_path);
        let modified = Self::get_modified_time(table_path);

        // Check cache first (only if no delta pending)
        if !has_delta {
            let mut cache = self.cache.write();
            if let Some(entry) = cache.get_mut(&cache_key) {
                if entry.modified_time >= modified && !entry.has_delta {
                    entry.last_access = Instant::now();
                    return Ok(entry.backend.clone());
                }
            }
        }

        // If delta exists, compact it first using lightweight open (metadata only)
        if has_delta {
            let storage = TableStorageBackend::open_for_compact(table_path)?;
            storage.compact()?;
            // Invalidate after compaction
            self.cache.write().remove(&cache_key);
            self.schema_cache.write().remove(&cache_key);
        }

        // Open fresh backend
        let backend = if table_path.exists() {
            TableStorageBackend::open_for_write_with_durability(table_path, durability)?
        } else {
            TableStorageBackend::create_with_durability(table_path, durability)?
        };

        let backend = Arc::new(backend);
        let new_modified = Self::get_modified_time(table_path);

        // Cache the backend and update schema cache
        {
            let mut cache = self.cache.write();
            Self::evict_lru_if_needed(&mut cache);
            cache.insert(
                cache_key.clone(),
                CacheEntry {
                    backend: backend.clone(),
                    modified_time: new_modified,
                    last_access: Instant::now(),
                    has_delta: false,
                },
            );
        }

        // Update schema cache
        {
            let schema_cols: std::collections::HashSet<String> = backend
                .get_schema()
                .into_iter()
                .map(|(name, _)| name)
                .collect();
            let row_count = backend.row_count();
            let mut schema_cache = self.schema_cache.write();
            schema_cache.insert(
                cache_key,
                SchemaCache {
                    columns: schema_cols,
                    row_count,
                    modified_time: new_modified,
                    is_v4: false,
                },
            );
        }

        Ok(backend)
    }

    /// Get backend for read-only operations (may use cached version)
    pub fn get_read_backend(&self, table_path: &Path) -> io::Result<Arc<TableStorageBackend>> {
        // Use path directly - avoid expensive canonicalize
        let cache_key = table_path.to_path_buf();
        let has_delta = Self::has_delta_file(table_path);
        let modified = Self::get_modified_time(table_path);

        // If delta exists, compact first for consistent reads (metadata-only open)
        if has_delta {
            let storage = TableStorageBackend::open_for_compact(table_path)?;
            storage.compact()?;
            self.cache.write().remove(&cache_key);
            self.schema_cache.write().remove(&cache_key);
        }

        // Check cache
        {
            let mut cache = self.cache.write();
            if let Some(entry) = cache.get_mut(&cache_key) {
                if entry.modified_time >= modified {
                    entry.last_access = Instant::now();
                    return Ok(entry.backend.clone());
                }
            }
        }

        // Open fresh backend (read-only mode)
        let backend = Arc::new(TableStorageBackend::open(table_path)?);
        let new_modified = Self::get_modified_time(table_path);

        // Cache it
        {
            let mut cache = self.cache.write();
            Self::evict_lru_if_needed(&mut cache);
            cache.insert(
                cache_key.clone(),
                CacheEntry {
                    backend: backend.clone(),
                    modified_time: new_modified,
                    last_access: Instant::now(),
                    has_delta: false,
                },
            );
        }

        // Update schema cache
        {
            let schema_cols: std::collections::HashSet<String> = backend
                .get_schema()
                .into_iter()
                .map(|(name, _)| name)
                .collect();
            let row_count = backend.row_count();
            let mut schema_cache = self.schema_cache.write();
            schema_cache.insert(
                cache_key,
                SchemaCache {
                    columns: schema_cols,
                    row_count,
                    modified_time: new_modified,
                    is_v4: false,
                },
            );
        }

        Ok(backend)
    }

    /// Get or create a cached insert backend (for delta writes)
    /// This avoids repeated file I/O when doing many small writes
    fn get_insert_backend(
        &self,
        table_path: &Path,
        durability: DurabilityLevel,
    ) -> io::Result<Arc<TableStorageBackend>> {
        let cache_key = table_path.to_path_buf();

        // Check insert cache first
        {
            let cache = self.insert_cache.read();
            if let Some(backend) = cache.get(&cache_key) {
                return Ok(backend.clone());
            }
        }

        // Open new insert backend
        let backend = Arc::new(TableStorageBackend::open_for_insert_with_durability(
            table_path, durability,
        )?);

        // Cache it
        {
            let mut cache = self.insert_cache.write();
            // Evict if too many entries
            if cache.len() >= MAX_CACHE_ENTRIES {
                if let Some(key) = cache.keys().next().cloned() {
                    cache.remove(&key);
                }
            }
            cache.insert(cache_key, backend.clone());
        }

        Ok(backend)
    }

    // ========================================================================
    // Write Operations
    // ========================================================================

    /// Write rows to a table with smart routing
    ///
    /// Automatically chooses the optimal write strategy:
    /// - **Delta write**: When table exists with data AND columns match exactly
    /// - **Full write**: For new tables, schema evolution, or partial columns
    pub fn write(
        &self,
        table_path: &Path,
        rows: &[HashMap<String, Value>],
        durability: DurabilityLevel,
    ) -> io::Result<Vec<u64>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        // Determine write strategy and V4 status in one pass (avoids double is_v4_file)
        let (use_delta, is_v4) = self.classify_write(table_path, rows);

        let ids = if use_delta {
            // Delta write: memory efficient, schema unchanged
            // Use cached insert backend to avoid repeated file I/O
            let backend = self.get_insert_backend(table_path, durability)?;
            let ids = backend.insert_rows_to_delta(rows)?;

            // For delta writes, only invalidate backend cache (not schema cache)
            // Schema doesn't change, so schema cache remains valid
            self.cache.write().remove(table_path);
            ApexExecutor::invalidate_cache_for_path(table_path);

            ids
        } else {
            // Full write: for new tables, schema evolution, or partial columns
            // V4 files: use insert backend (metadata-only) since save() uses
            // append_row_group. open_for_write loads all data which is empty
            // for V4 mmap-only, causing data loss.
            let backend = if is_v4 {
                self.get_insert_backend(table_path, durability)?
            } else {
                self.get_write_backend(table_path, durability)?
            };
            let ids = backend.insert_rows(rows)?;
            backend.save()?;

            self.invalidate(table_path);

            ids
        };

        // Notify indexes about the new rows (keeps indexes in sync for Python store() API)
        ApexExecutor::notify_indexes_after_write(table_path, &ids);

        Ok(ids)
    }

    /// Check if a file is V4 format by reading the header version field.
    #[inline]
    fn is_v4_file(table_path: &Path) -> bool {
        // Header: 8-byte magic + 4-byte version
        let mut buf = [0u8; 12];
        if let Ok(mut f) = std::fs::File::open(table_path) {
            use std::io::Read;
            if f.read_exact(&mut buf).is_ok() {
                let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
                return version >= 4;
            }
        }
        false
    }

    /// Classify write operation: returns (use_delta, is_v4)
    ///
    /// OPTIMIZED: Combines should_use_delta + is_v4_file into a single pass.
    /// Uses schema cache (with is_v4 field) to avoid repeated file header reads.
    #[inline]
    fn classify_write(&self, table_path: &Path, rows: &[HashMap<String, Value>]) -> (bool, bool) {
        if rows.is_empty() {
            return (false, false);
        }

        // Single metadata call for existence, size, and modified time
        let meta = match std::fs::metadata(table_path) {
            Ok(m) => m,
            Err(_) => return (false, false), // File doesn't exist
        };

        // Fast path: check file size (empty files should use full write)
        if meta.len() < 256 {
            return (false, false);
        }

        let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        let cache_key = table_path.to_path_buf();

        // Try schema cache first (FAST PATH — avoids file I/O entirely)
        {
            let schema_cache = self.schema_cache.read();
            if let Some(cached) = schema_cache.get(&cache_key) {
                if cached.modified_time >= modified {
                    // V4 files always use full write (append_row_group)
                    if cached.is_v4 {
                        return (false, true);
                    }
                    // Non-V4 with data: check schema match for delta
                    if cached.row_count > 0 {
                        let use_delta = rows.iter().all(|row| {
                            let data_cols: std::collections::HashSet<_> =
                                row.keys().cloned().collect();
                            cached.columns == data_cols
                        });
                        return (use_delta, false);
                    }
                    return (false, false);
                }
            }
        }

        // Cache miss — read V4 status from file header (single file open)
        let is_v4 = Self::is_v4_file(table_path);

        // V4 files: always full write, update cache with is_v4 flag
        if is_v4 {
            // Lightweight cache update (just V4 flag, no full schema load)
            let mut schema_cache = self.schema_cache.write();
            schema_cache
                .entry(cache_key)
                .or_insert_with(|| SchemaCache {
                    columns: std::collections::HashSet::new(),
                    row_count: 0,
                    modified_time: modified,
                    is_v4: true,
                })
                .is_v4 = true;
            return (false, true);
        }

        // Non-V4 cache miss: open backend for schema check (SLOW PATH)
        let backend = match TableStorageBackend::open(table_path) {
            Ok(b) => b,
            Err(_) => return (false, false),
        };

        let row_count = backend.row_count();
        if row_count == 0 {
            return (false, false);
        }

        let schema_cols: std::collections::HashSet<_> = backend
            .get_schema()
            .into_iter()
            .map(|(name, _)| name)
            .collect();

        // Update schema cache
        {
            let mut schema_cache = self.schema_cache.write();
            schema_cache.insert(
                cache_key,
                SchemaCache {
                    columns: schema_cols.clone(),
                    row_count,
                    modified_time: modified,
                    is_v4: false,
                },
            );
        }

        let use_delta = rows.iter().all(|row| {
            let data_cols: std::collections::HashSet<_> = row.keys().cloned().collect();
            schema_cols == data_cols
        });
        (use_delta, false)
    }

    /// Write a single row to a table
    pub fn write_one(
        &self,
        table_path: &Path,
        row: HashMap<String, Value>,
        durability: DurabilityLevel,
    ) -> io::Result<u64> {
        let ids = self.write(table_path, &[row], durability)?;
        Ok(ids.into_iter().next().unwrap_or(0))
    }

    // ========================================================================
    // Read Operations
    // ========================================================================

    /// Execute a SQL query and return results
    ///
    /// This method automatically:
    /// 1. Compacts delta files if needed
    /// 2. Executes the query
    ///
    /// # Arguments
    /// * `sql` - SQL query string
    /// * `base_dir` - Base directory for table resolution
    /// * `default_table_path` - Default table path for unqualified table names
    pub fn query(
        &self,
        sql: &str,
        base_dir: &Path,
        default_table_path: &Path,
    ) -> io::Result<RecordBatch> {
        // Compact delta if exists
        if Self::has_delta_file(default_table_path) {
            let storage = TableStorageBackend::open_for_write(default_table_path)?;
            storage.compact()?;
            self.invalidate(default_table_path);
        }

        // Execute query
        let result = ApexExecutor::execute_with_base_dir(sql, base_dir, default_table_path)?;
        result.to_record_batch()
    }

    /// Check if a record with given ID exists
    pub fn exists(&self, table_path: &Path, id: u64) -> io::Result<bool> {
        let backend = self.get_read_backend(table_path)?;
        Ok(backend.exists(id))
    }

    /// Get row count for a table
    pub fn row_count(&self, table_path: &Path) -> io::Result<u64> {
        let backend = self.get_read_backend(table_path)?;
        Ok(backend.row_count())
    }

    /// Retrieve a single record by ID
    pub fn retrieve(
        &self,
        table_path: &Path,
        base_dir: &Path,
        table_name: &str,
        id: u64,
    ) -> io::Result<Option<RecordBatch>> {
        // Use SQL query which handles delta merging
        let sql = format!("SELECT * FROM \"{}\" WHERE _id = {}", table_name, id);
        let batch = self.query(&sql, base_dir, table_path)?;

        if batch.num_rows() == 0 {
            Ok(None)
        } else {
            Ok(Some(batch))
        }
    }

    // ========================================================================
    // Delete Operations
    // ========================================================================

    /// Delete records matching a filter
    pub fn delete(
        &self,
        table_path: &Path,
        ids: &[u64],
        durability: DurabilityLevel,
    ) -> io::Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }

        // Invalidate before delete
        self.invalidate(table_path);

        let backend = self.get_write_backend(table_path, durability)?;

        let mut deleted = 0;
        for &id in ids {
            if backend.delete(id) {
                deleted += 1;
            }
        }

        backend.save()?;

        // Invalidate after delete
        self.invalidate(table_path);

        Ok(deleted)
    }

    // ========================================================================
    // Schema Operations
    // ========================================================================

    /// Get schema for a table
    pub fn get_schema(
        &self,
        table_path: &Path,
    ) -> io::Result<Vec<(String, crate::data::DataType)>> {
        let backend = self.get_read_backend(table_path)?;
        Ok(backend.get_schema())
    }

    /// Create a new table, optionally with a pre-defined schema.
    /// Pre-defining schema avoids schema inference on the first insert.
    pub fn create_table(&self, table_path: &Path, durability: DurabilityLevel) -> io::Result<()> {
        let _backend = TableStorageBackend::create_with_durability(table_path, durability)?;
        Ok(())
    }

    /// Create a new table with a pre-defined schema.
    /// Columns and null vectors are pre-allocated with correct types so
    /// insert_typed() hits the fast path immediately on the first insert.
    pub fn create_table_with_schema(
        &self,
        table_path: &Path,
        durability: DurabilityLevel,
        schema_cols: &[(String, ColumnType)],
    ) -> io::Result<()> {
        let _backend = TableStorageBackend::create_with_schema_and_durability(
            table_path,
            durability,
            schema_cols,
        )?;
        Ok(())
    }

    /// Delete a single record by ID
    pub fn delete_one(
        &self,
        table_path: &Path,
        id: u64,
        durability: DurabilityLevel,
    ) -> io::Result<bool> {
        self.invalidate(table_path);
        let backend = self.get_write_backend(table_path, durability)?;
        let result = backend.delete(id);
        backend.save()?;
        self.invalidate(table_path);
        Ok(result)
    }

    /// Replace a record by ID
    pub fn replace(
        &self,
        table_path: &Path,
        id: u64,
        fields: &HashMap<String, Value>,
        durability: DurabilityLevel,
    ) -> io::Result<bool> {
        self.invalidate(table_path);
        let backend = self.get_write_backend(table_path, durability)?;
        let result = backend.replace(id, fields)?;
        backend.save()?;
        self.invalidate(table_path);
        Ok(result)
    }

    /// Get active row count (excluding deleted)
    pub fn active_row_count(&self, table_path: &Path) -> io::Result<u64> {
        let backend = self.get_read_backend(table_path)?;
        Ok(backend.active_row_count())
    }

    /// Fast path: Get base table row count only (no delta scan)
    /// Use this for COUNT(*) without WHERE clause - O(1) lock-free read
    pub fn base_row_count(&self, table_path: &Path) -> io::Result<u64> {
        let backend = self.get_read_backend(table_path)?;
        Ok(backend.base_row_count())
    }

    // ========================================================================
    // Schema Modification Operations
    // ========================================================================

    /// Add a column to the table
    pub fn add_column(
        &self,
        table_path: &Path,
        column_name: &str,
        dtype: crate::data::DataType,
        durability: DurabilityLevel,
    ) -> io::Result<()> {
        self.invalidate(table_path);
        let backend = self.get_write_backend(table_path, durability)?;
        backend.add_column(column_name, dtype)?;
        backend.save()?;
        self.invalidate(table_path);
        Ok(())
    }

    /// Drop a column from the table
    pub fn drop_column(
        &self,
        table_path: &Path,
        column_name: &str,
        durability: DurabilityLevel,
    ) -> io::Result<()> {
        self.invalidate(table_path);
        let backend = self.get_write_backend(table_path, durability)?;
        backend.drop_column(column_name)?;
        backend.save()?;
        self.invalidate(table_path);
        Ok(())
    }

    /// Rename a column
    pub fn rename_column(
        &self,
        table_path: &Path,
        old_name: &str,
        new_name: &str,
        durability: DurabilityLevel,
    ) -> io::Result<()> {
        self.invalidate(table_path);
        if Self::is_v4_file(table_path) {
            // V4: modify schema in-memory then update footer only (no data reload)
            let backend = self.get_insert_backend(table_path, durability)?;
            backend.rename_column(old_name, new_name)?;
            backend.storage.update_v4_footer_schema()?;
            self.invalidate(table_path);
            return Ok(());
        }
        let backend = self.get_write_backend(table_path, durability)?;
        backend.rename_column(old_name, new_name)?;
        backend.save()?;
        self.invalidate(table_path);
        Ok(())
    }

    /// List all columns
    pub fn list_columns(&self, table_path: &Path) -> io::Result<Vec<String>> {
        let backend = self.get_read_backend(table_path)?;
        Ok(backend.list_columns())
    }

    /// Get column type
    pub fn get_column_type(
        &self,
        table_path: &Path,
        column_name: &str,
    ) -> io::Result<Option<crate::data::DataType>> {
        let backend = self.get_read_backend(table_path)?;
        Ok(backend.get_column_type(column_name))
    }

    /// Write typed columns (for store_columnar)
    /// OPTIMIZED: Uses V4 append_row_group for small inserts into existing tables
    /// to avoid rewriting the entire file (50x+ speedup for incremental inserts)
    pub fn write_typed(
        &self,
        table_path: &Path,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        fixedlist_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
        null_positions: HashMap<String, Vec<bool>>,
        durability: DurabilityLevel,
    ) -> io::Result<Vec<u64>> {
        use crate::storage::on_demand::{ColumnData, ColumnType};

        // Determine row count from first non-empty column
        let row_count = int_columns
            .values()
            .next()
            .map(|v| v.len())
            .or_else(|| float_columns.values().next().map(|v| v.len()))
            .or_else(|| string_columns.values().next().map(|v| v.len()))
            .or_else(|| bool_columns.values().next().map(|v| v.len()))
            .or_else(|| binary_columns.values().next().map(|v| v.len()))
            .or_else(|| fixedlist_columns.values().next().map(|v| v.len()))
            .unwrap_or(0);

        // FAST PATH: V4 append for existing tables with matching schema
        // Check if file exists, is V4, and schema matches
        if row_count > 0 && table_path.exists() {
            if let Ok(meta) = std::fs::metadata(table_path) {
                if meta.len() >= 256 {
                    // Reuse the insert backend so repeated small appends avoid reopening
                    // and reparsing the V4 footer on every call.
                    if let Ok(backend) = self.get_insert_backend(table_path, durability) {
                        let storage = &backend.storage;
                        let schema = storage.get_schema();
                        let header = storage.header_info();
                        let is_v4 = header.0 > 0; // footer_offset > 0 means V4

                        if is_v4 && !schema.is_empty() {
                            // Check column match
                            let schema_cols: std::collections::HashSet<String> =
                                schema.iter().map(|(name, _)| name.clone()).collect();
                            let mut data_cols = std::collections::HashSet::new();
                            for k in int_columns.keys() {
                                data_cols.insert(k.clone());
                            }
                            for k in float_columns.keys() {
                                data_cols.insert(k.clone());
                            }
                            for k in string_columns.keys() {
                                data_cols.insert(k.clone());
                            }
                            for k in bool_columns.keys() {
                                data_cols.insert(k.clone());
                            }
                            for k in binary_columns.keys() {
                                data_cols.insert(k.clone());
                            }
                            for k in fixedlist_columns.keys() {
                                data_cols.insert(k.clone());
                            }

                            if schema_cols == data_cols {
                                // Build ColumnData + null bitmaps in schema order
                                let mut new_columns: Vec<ColumnData> =
                                    Vec::with_capacity(schema.len());
                                let mut new_nulls: Vec<Vec<u8>> = Vec::with_capacity(schema.len());

                                // Allocate IDs
                                let start_id = storage.next_id_value();
                                let ids: Vec<u64> =
                                    (start_id..start_id + row_count as u64).collect();

                                for (col_name, col_type) in &schema {
                                    // Build null bitmap
                                    let null_bitmap =
                                        if let Some(null_vec) = null_positions.get(col_name) {
                                            let mut bitmap = vec![0u8; (row_count + 7) / 8];
                                            for (i, &is_null) in null_vec.iter().enumerate() {
                                                if is_null {
                                                    bitmap[i / 8] |= 1 << (i % 8);
                                                }
                                            }
                                            if bitmap.iter().any(|&b| b != 0) {
                                                bitmap
                                            } else {
                                                Vec::new()
                                            }
                                        } else {
                                            Vec::new()
                                        };
                                    new_nulls.push(null_bitmap);

                                    match col_type {
                                        ColumnType::Int64
                                        | ColumnType::Int32
                                        | ColumnType::Int16
                                        | ColumnType::Int8
                                        | ColumnType::UInt8
                                        | ColumnType::UInt16
                                        | ColumnType::UInt32
                                        | ColumnType::UInt64
                                        | ColumnType::Timestamp
                                        | ColumnType::Date => {
                                            let vals = int_columns
                                                .get(col_name)
                                                .cloned()
                                                .unwrap_or_else(|| vec![0; row_count]);
                                            new_columns.push(ColumnData::Int64(vals));
                                        }
                                        ColumnType::Float64 | ColumnType::Float32 => {
                                            let vals = float_columns
                                                .get(col_name)
                                                .cloned()
                                                .unwrap_or_else(|| vec![0.0; row_count]);
                                            new_columns.push(ColumnData::Float64(vals));
                                        }
                                        ColumnType::String
                                        | ColumnType::StringDict
                                        | ColumnType::Null => {
                                            if let Some(vals) = string_columns.get(col_name) {
                                                let mut offsets =
                                                    Vec::with_capacity(vals.len() + 1);
                                                let mut data = Vec::new();
                                                offsets.push(0u32);
                                                for s in vals {
                                                    data.extend_from_slice(s.as_bytes());
                                                    offsets.push(data.len() as u32);
                                                }
                                                new_columns
                                                    .push(ColumnData::String { offsets, data });
                                            } else {
                                                let offsets = vec![0u32; row_count + 1];
                                                new_columns.push(ColumnData::String {
                                                    offsets,
                                                    data: Vec::new(),
                                                });
                                            }
                                        }
                                        ColumnType::Bool => {
                                            if let Some(vals) = bool_columns.get(col_name) {
                                                let byte_count = (vals.len() + 7) / 8;
                                                let mut packed = vec![0u8; byte_count];
                                                for (i, &v) in vals.iter().enumerate() {
                                                    if v {
                                                        packed[i / 8] |= 1 << (i % 8);
                                                    }
                                                }
                                                new_columns.push(ColumnData::Bool {
                                                    data: packed,
                                                    len: vals.len(),
                                                });
                                            } else {
                                                new_columns.push(ColumnData::Bool {
                                                    data: vec![0u8; (row_count + 7) / 8],
                                                    len: row_count,
                                                });
                                            }
                                        }
                                        ColumnType::Binary => {
                                            if let Some(vals) = binary_columns.get(col_name) {
                                                let mut offsets =
                                                    Vec::with_capacity(vals.len() + 1);
                                                let mut data = Vec::new();
                                                offsets.push(0u32);
                                                for b in vals {
                                                    data.extend_from_slice(b);
                                                    offsets.push(data.len() as u32);
                                                }
                                                new_columns
                                                    .push(ColumnData::Binary { offsets, data });
                                            } else {
                                                let offsets = vec![0u32; row_count + 1];
                                                new_columns.push(ColumnData::Binary {
                                                    offsets,
                                                    data: Vec::new(),
                                                });
                                            }
                                        }
                                        ColumnType::FixedList => {
                                            if let Some(vals) = fixedlist_columns.get(col_name) {
                                                let dim = vals
                                                    .iter()
                                                    .find(|b| !b.is_empty())
                                                    .map(|b| b.len() / 4)
                                                    .unwrap_or(0)
                                                    as u32;
                                                let mut data: Vec<u8> = Vec::with_capacity(
                                                    row_count * dim as usize * 4,
                                                );
                                                for b in vals {
                                                    data.extend_from_slice(b);
                                                }
                                                new_columns
                                                    .push(ColumnData::FixedList { data, dim });
                                            } else {
                                                new_columns.push(ColumnData::FixedList {
                                                    data: Vec::new(),
                                                    dim: 0,
                                                });
                                            }
                                        }
                                        ColumnType::Float16List => {
                                            if let Some(vals) = fixedlist_columns.get(col_name) {
                                                let dim = vals
                                                    .iter()
                                                    .find(|b| !b.is_empty())
                                                    .map(|b| b.len() / 4)
                                                    .unwrap_or(0)
                                                    as u32;
                                                let mut data: Vec<u8> = Vec::with_capacity(
                                                    row_count * dim as usize * 2,
                                                );
                                                for b in vals {
                                                    for chunk in b.chunks_exact(4) {
                                                        let f = f32::from_le_bytes(
                                                            chunk.try_into().unwrap(),
                                                        );
                                                        let h =
                                                            crate::storage::on_demand::f32_to_f16(
                                                                f,
                                                            );
                                                        data.extend_from_slice(&h.to_le_bytes());
                                                    }
                                                }
                                                new_columns
                                                    .push(ColumnData::Float16List { data, dim });
                                            } else {
                                                new_columns.push(ColumnData::Float16List {
                                                    data: Vec::new(),
                                                    dim: 0,
                                                });
                                            }
                                        }
                                    }
                                }

                                // Append row group and return
                                match storage.append_row_group(&ids, &new_columns, &new_nulls) {
                                    Ok(()) => {
                                        self.invalidate_after_append(table_path);
                                        ApexExecutor::notify_indexes_after_write(table_path, &ids);
                                        return Ok(ids);
                                    }
                                    Err(_) => {
                                        // Fall through to full write
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // SLOW PATH: Full write (for new tables, schema changes, etc.)
        self.invalidate(table_path);
        let backend = self.get_write_backend(table_path, durability)?;
        let ids = backend.insert_typed_with_nulls_full(
            int_columns,
            float_columns,
            string_columns,
            binary_columns,
            fixedlist_columns,
            bool_columns,
            null_positions,
        )?;
        backend.save()?;
        self.invalidate(table_path);
        ApexExecutor::notify_indexes_after_write(table_path, &ids);
        Ok(ids)
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Get the global storage engine
pub fn engine() -> &'static StorageEngine {
    StorageEngine::global()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_engine_write_read() {
        let dir = tempdir().unwrap();
        let table_path = dir.path().join("test.apex");

        let engine = StorageEngine::global();

        // Write some rows
        let mut row1 = HashMap::new();
        row1.insert("name".to_string(), Value::String("Alice".to_string()));
        row1.insert("age".to_string(), Value::Int64(30));

        let mut row2 = HashMap::new();
        row2.insert("name".to_string(), Value::String("Bob".to_string()));
        row2.insert("age".to_string(), Value::Int64(25));

        let ids = engine
            .write(&table_path, &[row1, row2], DurabilityLevel::Fast)
            .unwrap();
        assert_eq!(ids.len(), 2);

        // Check row count
        let count = engine.row_count(&table_path).unwrap();
        assert_eq!(count, 2);

        // Check exists — use actual returned IDs (not hardcoded 1/2)
        // because StorageEngine::global() is shared and next_id may not start at 1
        assert!(engine.exists(&table_path, ids[0]).unwrap());
        assert!(engine.exists(&table_path, ids[1]).unwrap());
        assert!(!engine.exists(&table_path, 999).unwrap());
    }
}
