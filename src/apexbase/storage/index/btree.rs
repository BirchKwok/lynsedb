//! Disk-persistent B-Tree index for range queries and point lookups
//!
//! Supports ordered key lookups in O(log N) time.
//! Keys are stored sorted, enabling efficient range scans.
//! The index maps column values → row IDs (positions in the columnar file).

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::data::Value;

// ============================================================================
// B-Tree Index Entry
// ============================================================================

/// A comparable key extracted from a Value for B-Tree ordering
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexKey {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    /// Float stored as ordered bits for correct Ord implementation
    Float(u64),
    Str(String),
    Bytes(Vec<u8>),
}

impl IndexKey {
    /// Convert a Value to an IndexKey
    pub fn from_value(val: &Value) -> Self {
        match val {
            Value::Null => IndexKey::Null,
            Value::Bool(b) => IndexKey::Bool(*b),
            Value::Int8(v) => IndexKey::Int(*v as i64),
            Value::Int16(v) => IndexKey::Int(*v as i64),
            Value::Int32(v) => IndexKey::Int(*v as i64),
            Value::Int64(v) => IndexKey::Int(*v),
            Value::UInt8(v) => IndexKey::UInt(*v as u64),
            Value::UInt16(v) => IndexKey::UInt(*v as u64),
            Value::UInt32(v) => IndexKey::UInt(*v as u64),
            Value::UInt64(v) => IndexKey::UInt(*v),
            Value::Float32(f) => IndexKey::Float((*f as f64).to_bits()),
            Value::Float64(f) => IndexKey::Float(f.to_bits()),
            Value::String(s) => IndexKey::Str(s.clone()),
            Value::Binary(b) => IndexKey::Bytes(b.clone()),
            Value::Timestamp(t) => IndexKey::Int(*t),
            Value::Date(d) => IndexKey::Int(*d as i64),
            Value::Json(j) => IndexKey::Str(j.to_string()),
            Value::Array(_) => IndexKey::Null,
            Value::FixedList(_) => IndexKey::Null,
        }
    }
}

impl PartialOrd for IndexKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self, other) {
            (IndexKey::Null, IndexKey::Null) => Ordering::Equal,
            (IndexKey::Null, _) => Ordering::Less,
            (_, IndexKey::Null) => Ordering::Greater,
            (IndexKey::Bool(a), IndexKey::Bool(b)) => a.cmp(b),
            (IndexKey::Int(a), IndexKey::Int(b)) => a.cmp(b),
            (IndexKey::UInt(a), IndexKey::UInt(b)) => a.cmp(b),
            (IndexKey::Int(a), IndexKey::UInt(b)) => {
                if *a < 0 {
                    Ordering::Less
                } else {
                    (*a as u64).cmp(b)
                }
            }
            (IndexKey::UInt(a), IndexKey::Int(b)) => {
                if *b < 0 {
                    Ordering::Greater
                } else {
                    a.cmp(&(*b as u64))
                }
            }
            (IndexKey::Float(a), IndexKey::Float(b)) => {
                let fa = f64::from_bits(*a);
                let fb = f64::from_bits(*b);
                fa.partial_cmp(&fb).unwrap_or(Ordering::Equal)
            }
            (IndexKey::Str(a), IndexKey::Str(b)) => a.cmp(b),
            (IndexKey::Bytes(a), IndexKey::Bytes(b)) => a.cmp(b),
            // Cross-type: use discriminant index for ordering
            _ => {
                fn disc_index(k: &IndexKey) -> u8 {
                    match k {
                        IndexKey::Null => 0,
                        IndexKey::Bool(_) => 1,
                        IndexKey::Int(_) => 2,
                        IndexKey::UInt(_) => 3,
                        IndexKey::Float(_) => 4,
                        IndexKey::Str(_) => 5,
                        IndexKey::Bytes(_) => 6,
                    }
                }
                disc_index(self).cmp(&disc_index(other))
            }
        }
    }
}

// Make IndexKey usable as HashMap key
impl std::hash::Hash for IndexKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            IndexKey::Null => {}
            IndexKey::Bool(b) => b.hash(state),
            IndexKey::Int(v) => v.hash(state),
            IndexKey::UInt(v) => v.hash(state),
            IndexKey::Float(v) => v.hash(state),
            IndexKey::Str(s) => s.hash(state),
            IndexKey::Bytes(b) => b.hash(state),
        }
    }
}

// ============================================================================
// B-Tree Index
// ============================================================================

/// Persistent B-Tree index mapping column values to row IDs
///
/// Supports:
/// - Point lookup: O(log N)
/// - Range scan: O(log N + K) where K is result size
/// - Ordered iteration
/// - Duplicate keys (multiple rows per key)
#[derive(Debug, Serialize, Deserialize)]
pub struct BTreeIndex {
    /// Column name this index is built on
    column_name: String,
    /// Whether this is a unique index
    unique: bool,
    /// The B-Tree mapping: key → list of row IDs
    tree: BTreeMap<IndexKey, Vec<u64>>,
    /// Total number of entries
    entry_count: u64,
    /// Persistence path (None = memory-only)
    #[serde(skip)]
    path: Option<PathBuf>,
    /// Whether the index has been modified since last save
    #[serde(skip)]
    dirty: bool,
}

impl BTreeIndex {
    /// Create a new empty B-Tree index
    pub fn new(column_name: &str, unique: bool) -> Self {
        Self {
            column_name: column_name.to_string(),
            unique,
            tree: BTreeMap::new(),
            entry_count: 0,
            path: None,
            dirty: false,
        }
    }

    /// Create a new B-Tree index with a persistence path
    pub fn with_path(column_name: &str, unique: bool, path: PathBuf) -> Self {
        Self {
            column_name: column_name.to_string(),
            unique,
            tree: BTreeMap::new(),
            entry_count: 0,
            path: Some(path),
            dirty: false,
        }
    }

    /// Column name this index covers
    pub fn column_name(&self) -> &str {
        &self.column_name
    }

    /// Whether this is a unique index
    pub fn is_unique(&self) -> bool {
        self.unique
    }

    /// Number of indexed entries
    pub fn len(&self) -> u64 {
        self.entry_count
    }

    /// Whether the index is empty
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Number of distinct keys
    pub fn distinct_keys(&self) -> usize {
        self.tree.len()
    }

    // ========================================================================
    // Insert / Delete
    // ========================================================================

    /// Insert a key-rowid pair into the index
    pub fn insert(&mut self, key: IndexKey, row_id: u64) -> io::Result<()> {
        if self.unique {
            let entry = self.tree.entry(key);
            match entry {
                std::collections::btree_map::Entry::Occupied(e) => {
                    if !e.get().is_empty() {
                        return Err(io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            format!("Duplicate key in unique index '{}'", self.column_name),
                        ));
                    }
                    e.into_mut().push(row_id);
                }
                std::collections::btree_map::Entry::Vacant(e) => {
                    e.insert(vec![row_id]);
                }
            }
        } else {
            self.tree.entry(key).or_insert_with(Vec::new).push(row_id);
        }
        self.entry_count += 1;
        self.dirty = true;
        Ok(())
    }

    /// Insert a Value-rowid pair (convenience method)
    pub fn insert_value(&mut self, value: &Value, row_id: u64) -> io::Result<()> {
        self.insert(IndexKey::from_value(value), row_id)
    }

    /// Remove a key-rowid pair from the index
    pub fn remove(&mut self, key: &IndexKey, row_id: u64) -> bool {
        if let Some(ids) = self.tree.get_mut(key) {
            if let Some(pos) = ids.iter().position(|&id| id == row_id) {
                ids.swap_remove(pos);
                self.entry_count -= 1;
                self.dirty = true;
                if ids.is_empty() {
                    self.tree.remove(key);
                }
                return true;
            }
        }
        false
    }

    /// Remove by Value (convenience method)
    pub fn remove_value(&mut self, value: &Value, row_id: u64) -> bool {
        let key = IndexKey::from_value(value);
        self.remove(&key, row_id)
    }

    /// Bulk insert for building index from existing data
    pub fn bulk_insert(&mut self, pairs: Vec<(IndexKey, u64)>) {
        for (key, row_id) in pairs {
            self.tree.entry(key).or_insert_with(Vec::new).push(row_id);
            self.entry_count += 1;
        }
        self.dirty = true;
    }

    // ========================================================================
    // Lookup
    // ========================================================================

    /// Point lookup: get all row IDs for an exact key
    pub fn get(&self, key: &IndexKey) -> Option<&[u64]> {
        self.tree.get(key).map(|v| v.as_slice())
    }

    /// Point lookup by Value
    pub fn get_value(&self, value: &Value) -> Option<&[u64]> {
        let key = IndexKey::from_value(value);
        self.get(&key)
    }

    /// Range scan: get all row IDs for keys in [start, end]
    pub fn range_inclusive(&self, start: &IndexKey, end: &IndexKey) -> Vec<u64> {
        let mut result = Vec::new();
        for (_, ids) in self.tree.range(start.clone()..=end.clone()) {
            result.extend_from_slice(ids);
        }
        result
    }

    /// Range scan: get all row IDs for keys in [start, end)
    pub fn range(&self, start: &IndexKey, end: &IndexKey) -> Vec<u64> {
        let mut result = Vec::new();
        for (_, ids) in self.tree.range(start.clone()..end.clone()) {
            result.extend_from_slice(ids);
        }
        result
    }

    /// Get all row IDs where key > value
    pub fn greater_than(&self, key: &IndexKey) -> Vec<u64> {
        let mut result = Vec::new();
        for (k, ids) in self.tree.range(key.clone()..) {
            if k > key {
                result.extend_from_slice(ids);
            }
        }
        result
    }

    /// Get all row IDs where key >= value
    pub fn greater_than_or_equal(&self, key: &IndexKey) -> Vec<u64> {
        let mut result = Vec::new();
        for (_, ids) in self.tree.range(key.clone()..) {
            result.extend_from_slice(ids);
        }
        result
    }

    /// Get all row IDs where key < value
    pub fn less_than(&self, key: &IndexKey) -> Vec<u64> {
        let mut result = Vec::new();
        for (_, ids) in self.tree.range(..key.clone()) {
            result.extend_from_slice(ids);
        }
        result
    }

    /// Get all row IDs where key <= value
    pub fn less_than_or_equal(&self, key: &IndexKey) -> Vec<u64> {
        let mut result = Vec::new();
        for (_, ids) in self.tree.range(..=key.clone()) {
            result.extend_from_slice(ids);
        }
        result
    }

    // ========================================================================
    // Persistence
    // ========================================================================

    /// Save index to disk
    pub fn save(&mut self) -> io::Result<()> {
        if !self.dirty {
            return Ok(());
        }
        if let Some(path) = &self.path {
            let data = bincode::serialize(self)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            std::fs::write(path, &data)?;
            self.dirty = false;
        }
        Ok(())
    }

    /// Load index from disk
    pub fn load(path: &Path) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        let mut index: Self = bincode::deserialize(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        index.path = Some(path.to_path_buf());
        index.dirty = false;
        Ok(index)
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.tree.clear();
        self.entry_count = 0;
        self.dirty = true;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_btree_insert_and_lookup() {
        let mut idx = BTreeIndex::new("age", false);
        idx.insert(IndexKey::Int(25), 0).unwrap();
        idx.insert(IndexKey::Int(30), 1).unwrap();
        idx.insert(IndexKey::Int(25), 2).unwrap();

        assert_eq!(idx.get(&IndexKey::Int(25)), Some(&[0u64, 2][..]));
        assert_eq!(idx.get(&IndexKey::Int(30)), Some(&[1u64][..]));
        assert_eq!(idx.get(&IndexKey::Int(99)), None);
        assert_eq!(idx.len(), 3);
        assert_eq!(idx.distinct_keys(), 2);
    }

    #[test]
    fn test_btree_unique_constraint() {
        let mut idx = BTreeIndex::new("email", true);
        idx.insert(IndexKey::Str("a@b.com".into()), 0).unwrap();
        let result = idx.insert(IndexKey::Str("a@b.com".into()), 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_btree_range_scan() {
        let mut idx = BTreeIndex::new("score", false);
        for i in 0..100 {
            idx.insert(IndexKey::Int(i), i as u64).unwrap();
        }
        let result = idx.range_inclusive(&IndexKey::Int(10), &IndexKey::Int(20));
        assert_eq!(result.len(), 11); // 10..=20
    }

    #[test]
    fn test_btree_remove() {
        let mut idx = BTreeIndex::new("name", false);
        idx.insert(IndexKey::Str("alice".into()), 0).unwrap();
        idx.insert(IndexKey::Str("alice".into()), 1).unwrap();
        assert!(idx.remove(&IndexKey::Str("alice".into()), 0));
        assert_eq!(idx.get(&IndexKey::Str("alice".into())), Some(&[1u64][..]));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_index_key_ordering() {
        assert!(IndexKey::Null < IndexKey::Int(0));
        assert!(IndexKey::Int(-1) < IndexKey::Int(0));
        assert!(IndexKey::Int(0) < IndexKey::Int(1));
        assert!(IndexKey::Str("a".into()) < IndexKey::Str("b".into()));
    }

    #[test]
    fn test_btree_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.btidx");

        let mut idx = BTreeIndex::with_path("col", false, path.clone());
        idx.insert(IndexKey::Int(42), 7).unwrap();
        idx.save().unwrap();

        let loaded = BTreeIndex::load(&path).unwrap();
        assert_eq!(loaded.get(&IndexKey::Int(42)), Some(&[7u64][..]));
        assert_eq!(loaded.len(), 1);
    }
}
