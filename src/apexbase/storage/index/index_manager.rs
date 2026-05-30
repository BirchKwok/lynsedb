//! Index Manager - Manages all indexes for a table
//!
//! Handles index lifecycle (create, drop, rebuild) and query optimization
//! by selecting the best index for a given query predicate.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::btree::{BTreeIndex, IndexKey};
use super::hash_index::HashIndex;
use crate::data::{DataType, Value};

// ============================================================================
// Index Type
// ============================================================================

/// Type of index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    /// B-Tree index: good for range queries and ordered access
    BTree,
    /// Hash index: optimal for equality lookups
    Hash,
}

impl IndexType {
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "btree" | "b-tree" | "b_tree" => Some(IndexType::BTree),
            "hash" => Some(IndexType::Hash),
            _ => None,
        }
    }
}

// ============================================================================
// Index Metadata
// ============================================================================

/// Metadata for a single index (persisted in catalog)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMeta {
    /// Index name (unique within a table)
    pub name: String,
    /// Column name the index is built on (first column for composite)
    pub column_name: String,
    /// Index type
    pub index_type: IndexType,
    /// Whether it enforces uniqueness
    pub unique: bool,
    /// Column data type (type of first column for composite)
    pub data_type: DataType,
    /// Creation timestamp
    pub created_at: i64,
    /// All columns in the index (for composite indexes).
    /// Empty means single-column (backward compat with old serialized catalogs).
    #[serde(default)]
    pub columns: Vec<String>,
}

impl IndexMeta {
    /// Get the effective column list (handles backward compat)
    pub fn effective_columns(&self) -> Vec<&str> {
        if self.columns.is_empty() {
            vec![&self.column_name]
        } else {
            self.columns.iter().map(|s| s.as_str()).collect()
        }
    }

    /// Whether this is a composite (multi-column) index
    pub fn is_composite(&self) -> bool {
        self.columns.len() > 1
    }
}

/// Build a composite IndexKey from multiple column values.
/// For single-column, returns the value's key directly.
/// For multi-column, concatenates as "val1\0val2\0val3" string key.
fn composite_key(columns: &[String], values: &HashMap<String, Value>) -> Option<IndexKey> {
    if columns.len() == 1 {
        values.get(&columns[0]).map(|v| IndexKey::from_value(v))
    } else {
        let mut parts: Vec<String> = Vec::with_capacity(columns.len());
        for col in columns {
            match values.get(col) {
                Some(v) => parts.push(v.to_string()),
                None => return None, // Missing column value
            }
        }
        Some(IndexKey::from_value(&Value::String(parts.join("\0"))))
    }
}

// ============================================================================
// Index Instance (runtime)
// ============================================================================

/// A runtime index instance (either BTree or Hash)
enum IndexInstance {
    BTree(BTreeIndex),
    Hash(HashIndex),
}

impl IndexInstance {
    fn insert(&mut self, key: IndexKey, row_id: u64) -> io::Result<()> {
        match self {
            IndexInstance::BTree(idx) => idx.insert(key, row_id),
            IndexInstance::Hash(idx) => idx.insert(key, row_id),
        }
    }

    fn remove(&mut self, key: &IndexKey, row_id: u64) -> bool {
        match self {
            IndexInstance::BTree(idx) => idx.remove(key, row_id),
            IndexInstance::Hash(idx) => idx.remove(key, row_id),
        }
    }

    fn get(&self, key: &IndexKey) -> Option<&[u64]> {
        match self {
            IndexInstance::BTree(idx) => idx.get(key),
            IndexInstance::Hash(idx) => idx.get(key),
        }
    }

    fn save(&mut self) -> io::Result<()> {
        match self {
            IndexInstance::BTree(idx) => idx.save(),
            IndexInstance::Hash(idx) => idx.save(),
        }
    }

    fn clear(&mut self) {
        match self {
            IndexInstance::BTree(idx) => idx.clear(),
            IndexInstance::Hash(idx) => idx.clear(),
        }
    }

    fn len(&self) -> u64 {
        match self {
            IndexInstance::BTree(idx) => idx.len(),
            IndexInstance::Hash(idx) => idx.len(),
        }
    }
}

// ============================================================================
// Query Hint (what the planner tells us)
// ============================================================================

/// A predicate hint from the query planner for index selection
#[derive(Debug, Clone)]
pub enum PredicateHint {
    /// Equality: col = value
    Eq(Value),
    /// Range: col BETWEEN low AND high
    Range { low: Value, high: Value },
    /// Greater than: col > value
    Gt(Value),
    /// Greater than or equal: col >= value
    Gte(Value),
    /// Less than: col < value
    Lt(Value),
    /// Less than or equal: col <= value
    Lte(Value),
    /// IN list: col IN (v1, v2, ...)
    In(Vec<Value>),
}

/// Result of an index lookup
#[derive(Debug)]
pub struct IndexLookupResult {
    /// Row IDs that match the predicate
    pub row_ids: Vec<u64>,
    /// Whether this is an exact result (no further filtering needed)
    pub exact: bool,
}

// ============================================================================
// Index Manager
// ============================================================================

/// Manages all indexes for a single table
///
/// Responsibilities:
/// - Create / drop / rebuild indexes
/// - Route queries to the best index
/// - Keep indexes in sync with data changes
/// - Persist index catalog
pub struct IndexManager {
    /// Table name
    table_name: String,
    /// Base directory for index files
    base_dir: PathBuf,
    /// Index catalog: index_name → metadata
    catalog: HashMap<String, IndexMeta>,
    /// Runtime index instances: index_name → instance
    instances: HashMap<String, IndexInstance>,
    /// Column → index name mapping for fast lookup
    column_index_map: HashMap<String, Vec<String>>,
    /// Whether catalog has been modified
    dirty: bool,
}

impl IndexManager {
    /// Create a new index manager for a table
    pub fn new(table_name: &str, base_dir: &Path) -> Self {
        let idx_dir = base_dir.join("indexes");
        Self {
            table_name: table_name.to_string(),
            base_dir: idx_dir,
            catalog: HashMap::new(),
            instances: HashMap::new(),
            column_index_map: HashMap::new(),
            dirty: false,
        }
    }

    /// Load existing index catalog from disk
    pub fn load(table_name: &str, base_dir: &Path) -> io::Result<Self> {
        let idx_dir = base_dir.join("indexes");
        let catalog_path = idx_dir.join(format!("{}.idxcat", table_name));

        if !catalog_path.exists() {
            return Ok(Self::new(table_name, base_dir));
        }

        let data = std::fs::read(&catalog_path)?;
        let catalog: HashMap<String, IndexMeta> = bincode::deserialize(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        let mut mgr = Self {
            table_name: table_name.to_string(),
            base_dir: idx_dir,
            catalog: catalog.clone(),
            instances: HashMap::new(),
            column_index_map: HashMap::new(),
            dirty: false,
        };

        // Build column→index mapping
        for (name, meta) in &catalog {
            mgr.column_index_map
                .entry(meta.column_name.clone())
                .or_insert_with(Vec::new)
                .push(name.clone());
        }

        // Lazily load index instances (only when needed)
        Ok(mgr)
    }

    /// Table name
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// List all indexes
    pub fn list_indexes(&self) -> Vec<&IndexMeta> {
        self.catalog.values().collect()
    }

    /// Get index metadata by name
    pub fn get_index_meta(&self, name: &str) -> Option<&IndexMeta> {
        self.catalog.get(name)
    }

    /// Check if a column has any index
    pub fn has_index_on(&self, column_name: &str) -> bool {
        self.column_index_map.contains_key(column_name)
    }

    /// Returns true when this table has no indexes at all — used as a fast CBO bypass.
    #[inline]
    pub fn catalog_is_empty(&self) -> bool {
        self.catalog.is_empty()
    }

    // ========================================================================
    // Index Lifecycle
    // ========================================================================

    /// Create a new single-column index
    pub fn create_index(
        &mut self,
        name: &str,
        column_name: &str,
        index_type: IndexType,
        unique: bool,
        data_type: DataType,
    ) -> io::Result<()> {
        self.create_index_multi(
            name,
            &[column_name.to_string()],
            index_type,
            unique,
            data_type,
        )
    }

    /// Create a new index (supports single or multi-column)
    pub fn create_index_multi(
        &mut self,
        name: &str,
        columns: &[String],
        index_type: IndexType,
        unique: bool,
        data_type: DataType,
    ) -> io::Result<()> {
        if columns.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Index requires at least one column",
            ));
        }
        if self.catalog.contains_key(name) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("Index '{}' already exists", name),
            ));
        }

        // Create index directory if needed
        std::fs::create_dir_all(&self.base_dir)?;

        let first_col = &columns[0];
        let meta = IndexMeta {
            name: name.to_string(),
            column_name: first_col.clone(),
            index_type,
            unique,
            data_type,
            created_at: chrono::Utc::now().timestamp(),
            columns: columns.to_vec(),
        };

        // Create the runtime instance
        let index_path = self.index_file_path(name, index_type);
        let instance = match index_type {
            IndexType::BTree => {
                IndexInstance::BTree(BTreeIndex::with_path(first_col, unique, index_path))
            }
            IndexType::Hash => {
                IndexInstance::Hash(HashIndex::with_path(first_col, unique, index_path))
            }
        };

        // Register: map each column to this index
        for col in columns {
            self.column_index_map
                .entry(col.clone())
                .or_insert_with(Vec::new)
                .push(name.to_string());
        }
        self.catalog.insert(name.to_string(), meta);
        self.instances.insert(name.to_string(), instance);
        self.dirty = true;

        Ok(())
    }

    /// Drop an index
    pub fn drop_index(&mut self, name: &str) -> io::Result<()> {
        let meta = self.catalog.remove(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Index '{}' not found", name),
            )
        })?;

        // Remove from column mapping
        if let Some(names) = self.column_index_map.get_mut(&meta.column_name) {
            names.retain(|n| n != name);
            if names.is_empty() {
                self.column_index_map.remove(&meta.column_name);
            }
        }

        // Remove runtime instance
        self.instances.remove(name);

        // Remove index file
        let index_path = self.index_file_path(name, meta.index_type);
        let _ = std::fs::remove_file(&index_path);

        self.dirty = true;
        Ok(())
    }

    // ========================================================================
    // Data Maintenance (keep indexes in sync)
    // ========================================================================

    /// Notify that a row was inserted
    pub fn on_insert(
        &mut self,
        row_id: u64,
        column_values: &HashMap<String, Value>,
    ) -> io::Result<()> {
        // Collect unique index names that need updating
        let mut seen_indexes: std::collections::HashSet<String> = std::collections::HashSet::new();
        for col_name in column_values.keys() {
            if let Some(index_names) = self.column_index_map.get(col_name).cloned() {
                for idx_name in index_names {
                    seen_indexes.insert(idx_name);
                }
            }
        }
        for idx_name in &seen_indexes {
            let cols = self
                .catalog
                .get(idx_name)
                .map(|m| {
                    m.effective_columns()
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if let Some(key) = composite_key(&cols, column_values) {
                let instance = self.ensure_loaded(idx_name)?;
                instance.insert(key, row_id)?;
            }
        }
        Ok(())
    }

    /// Notify that a row was deleted
    pub fn on_delete(&mut self, row_id: u64, column_values: &HashMap<String, Value>) {
        let mut seen_indexes: std::collections::HashSet<String> = std::collections::HashSet::new();
        for col_name in column_values.keys() {
            if let Some(index_names) = self.column_index_map.get(col_name).cloned() {
                for idx_name in index_names {
                    seen_indexes.insert(idx_name);
                }
            }
        }
        for idx_name in &seen_indexes {
            let cols = self
                .catalog
                .get(idx_name)
                .map(|m| {
                    m.effective_columns()
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if let Some(key) = composite_key(&cols, column_values) {
                if let Some(instance) = self.instances.get_mut(idx_name.as_str()) {
                    instance.remove(&key, row_id);
                }
            }
        }
    }

    /// Notify that a row was updated
    pub fn on_update(
        &mut self,
        row_id: u64,
        old_values: &HashMap<String, Value>,
        new_values: &HashMap<String, Value>,
    ) -> io::Result<()> {
        // Remove old entries, insert new entries
        for (col_name, old_val) in old_values {
            if let Some(new_val) = new_values.get(col_name) {
                if old_val != new_val {
                    if let Some(index_names) = self.column_index_map.get(col_name).cloned() {
                        for idx_name in &index_names {
                            if let Some(instance) = self.instances.get_mut(idx_name) {
                                let old_key = IndexKey::from_value(old_val);
                                instance.remove(&old_key, row_id);
                                let new_key = IndexKey::from_value(new_val);
                                instance.insert(new_key, row_id)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // ========================================================================
    // Query Optimization
    // ========================================================================

    /// Try to use an index for a predicate on a column
    /// Returns row IDs if an index can satisfy the predicate, None otherwise
    pub fn lookup(
        &mut self,
        column_name: &str,
        predicate: &PredicateHint,
    ) -> io::Result<Option<IndexLookupResult>> {
        let index_names = match self.column_index_map.get(column_name) {
            Some(names) => names.clone(),
            None => return Ok(None),
        };

        // Find the best index for this predicate
        let best_idx_name = self.select_best_index(&index_names, predicate);
        if best_idx_name.is_none() {
            return Ok(None);
        }
        let idx_name = best_idx_name.unwrap();

        let instance = self.ensure_loaded(&idx_name)?;

        let row_ids = match predicate {
            PredicateHint::Eq(val) => {
                let key = IndexKey::from_value(val);
                instance
                    .get(&key)
                    .map(|ids| ids.to_vec())
                    .unwrap_or_default()
            }
            PredicateHint::Range { low, high } => {
                match instance {
                    IndexInstance::BTree(bt) => {
                        let low_key = IndexKey::from_value(low);
                        let high_key = IndexKey::from_value(high);
                        bt.range_inclusive(&low_key, &high_key)
                    }
                    IndexInstance::Hash(_) => return Ok(None), // Hash can't do range
                }
            }
            PredicateHint::Gt(val) => match instance {
                IndexInstance::BTree(bt) => {
                    let key = IndexKey::from_value(val);
                    bt.greater_than(&key)
                }
                IndexInstance::Hash(_) => return Ok(None),
            },
            PredicateHint::Gte(val) => match instance {
                IndexInstance::BTree(bt) => {
                    let key = IndexKey::from_value(val);
                    bt.greater_than_or_equal(&key)
                }
                IndexInstance::Hash(_) => return Ok(None),
            },
            PredicateHint::Lt(val) => match instance {
                IndexInstance::BTree(bt) => {
                    let key = IndexKey::from_value(val);
                    bt.less_than(&key)
                }
                IndexInstance::Hash(_) => return Ok(None),
            },
            PredicateHint::Lte(val) => match instance {
                IndexInstance::BTree(bt) => {
                    let key = IndexKey::from_value(val);
                    bt.less_than_or_equal(&key)
                }
                IndexInstance::Hash(_) => return Ok(None),
            },
            PredicateHint::In(vals) => {
                let mut result = Vec::new();
                for val in vals {
                    let key = IndexKey::from_value(val);
                    if let Some(ids) = instance.get(&key) {
                        result.extend_from_slice(ids);
                    }
                }
                result
            }
        };

        Ok(Some(IndexLookupResult {
            row_ids,
            exact: true,
        }))
    }

    // ========================================================================
    // Persistence
    // ========================================================================

    /// Save catalog and all dirty indexes to disk
    pub fn save(&mut self) -> io::Result<()> {
        if self.dirty {
            std::fs::create_dir_all(&self.base_dir)?;
            let catalog_path = self.base_dir.join(format!("{}.idxcat", self.table_name));
            let data = bincode::serialize(&self.catalog)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            std::fs::write(&catalog_path, &data)?;
            self.dirty = false;
        }

        // Save all dirty index instances
        for instance in self.instances.values_mut() {
            instance.save()?;
        }
        Ok(())
    }

    /// Rebuild all indexes from scratch (used after compaction)
    pub fn rebuild_all(&mut self) {
        for instance in self.instances.values_mut() {
            instance.clear();
        }
    }

    // ========================================================================
    // Internal Helpers
    // ========================================================================

    /// Get the file path for an index
    fn index_file_path(&self, name: &str, index_type: IndexType) -> PathBuf {
        let ext = match index_type {
            IndexType::BTree => "btidx",
            IndexType::Hash => "hashidx",
        };
        self.base_dir
            .join(format!("{}_{}.{}", self.table_name, name, ext))
    }

    /// Ensure an index instance is loaded into memory
    fn ensure_loaded(&mut self, name: &str) -> io::Result<&mut IndexInstance> {
        if !self.instances.contains_key(name) {
            let meta = self
                .catalog
                .get(name)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("Index '{}' not found", name),
                    )
                })?
                .clone();
            let path = self.index_file_path(name, meta.index_type);
            let instance = if path.exists() {
                match meta.index_type {
                    IndexType::BTree => IndexInstance::BTree(BTreeIndex::load(&path)?),
                    IndexType::Hash => IndexInstance::Hash(HashIndex::load(&path)?),
                }
            } else {
                match meta.index_type {
                    IndexType::BTree => IndexInstance::BTree(BTreeIndex::with_path(
                        &meta.column_name,
                        meta.unique,
                        path,
                    )),
                    IndexType::Hash => IndexInstance::Hash(HashIndex::with_path(
                        &meta.column_name,
                        meta.unique,
                        path,
                    )),
                }
            };
            self.instances.insert(name.to_string(), instance);
        }
        Ok(self.instances.get_mut(name).unwrap())
    }

    /// Select the best index for a predicate
    fn select_best_index(
        &self,
        index_names: &[String],
        predicate: &PredicateHint,
    ) -> Option<String> {
        match predicate {
            PredicateHint::Eq(_) | PredicateHint::In(_) => {
                // Prefer hash index for equality, fall back to btree
                for name in index_names {
                    if let Some(meta) = self.catalog.get(name) {
                        if meta.index_type == IndexType::Hash {
                            return Some(name.clone());
                        }
                    }
                }
                // Fall back to any available index
                index_names.first().cloned()
            }
            PredicateHint::Range { .. }
            | PredicateHint::Gt(_)
            | PredicateHint::Gte(_)
            | PredicateHint::Lt(_)
            | PredicateHint::Lte(_) => {
                // Only BTree supports range queries
                for name in index_names {
                    if let Some(meta) = self.catalog.get(name) {
                        if meta.index_type == IndexType::BTree {
                            return Some(name.clone());
                        }
                    }
                }
                None
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_manager_create_and_lookup() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = IndexManager::new("test_table", dir.path());

        // Create a hash index on _id
        mgr.create_index("idx_id", "_id", IndexType::Hash, true, DataType::UInt64)
            .unwrap();

        // Insert some data
        let mut row = HashMap::new();
        row.insert("_id".to_string(), Value::UInt64(1));
        mgr.on_insert(0, &row).unwrap();

        row.insert("_id".to_string(), Value::UInt64(2));
        mgr.on_insert(1, &row).unwrap();

        // Lookup
        let result = mgr
            .lookup("_id", &PredicateHint::Eq(Value::UInt64(1)))
            .unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().row_ids, vec![0]);
    }

    #[test]
    fn test_index_manager_btree_range() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = IndexManager::new("test_table", dir.path());

        mgr.create_index("idx_age", "age", IndexType::BTree, false, DataType::Int64)
            .unwrap();

        for i in 0..100 {
            let mut row = HashMap::new();
            row.insert("age".to_string(), Value::Int64(i));
            mgr.on_insert(i as u64, &row).unwrap();
        }

        let result = mgr
            .lookup(
                "age",
                &PredicateHint::Range {
                    low: Value::Int64(10),
                    high: Value::Int64(20),
                },
            )
            .unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().row_ids.len(), 11);
    }

    #[test]
    fn test_index_manager_persistence() {
        let dir = tempfile::tempdir().unwrap();

        {
            let mut mgr = IndexManager::new("test_table", dir.path());
            mgr.create_index(
                "idx_name",
                "name",
                IndexType::BTree,
                false,
                DataType::String,
            )
            .unwrap();

            let mut row = HashMap::new();
            row.insert("name".to_string(), Value::String("alice".into()));
            mgr.on_insert(0, &row).unwrap();

            mgr.save().unwrap();
        }

        // Reload
        let mut mgr = IndexManager::load("test_table", dir.path()).unwrap();
        assert!(mgr.has_index_on("name"));

        let result = mgr
            .lookup("name", &PredicateHint::Eq(Value::String("alice".into())))
            .unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().row_ids, vec![0]);
    }

    #[test]
    fn test_index_manager_drop() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = IndexManager::new("test_table", dir.path());
        mgr.create_index("idx_x", "x", IndexType::Hash, false, DataType::Int64)
            .unwrap();
        assert!(mgr.has_index_on("x"));

        mgr.drop_index("idx_x").unwrap();
        assert!(!mgr.has_index_on("x"));
    }
}
