//! In-memory Hash index for O(1) point lookups
//!
//! Optimized for equality queries (`WHERE col = value`).
//! Uses AHash for fast non-cryptographic hashing.
//! Memory-resident with optional persistence.

use std::io;
use std::path::{Path, PathBuf};

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use super::btree::IndexKey;

// ============================================================================
// Hash Index
// ============================================================================

/// In-memory hash index for O(1) equality lookups
///
/// Best for:
/// - Primary key lookups (`WHERE _id = X`)
/// - Equality filters on high-cardinality columns
/// - Foreign key joins
///
/// Not suitable for:
/// - Range queries (use BTreeIndex instead)
/// - Low-cardinality columns (use bloom filter instead)
#[derive(Debug)]
pub struct HashIndex {
    /// Column name this index covers
    column_name: String,
    /// Whether this is a unique index
    unique: bool,
    /// The hash map: key → row IDs
    map: AHashMap<IndexKey, Vec<u64>>,
    /// Total entry count
    entry_count: u64,
    /// Persistence path
    path: Option<PathBuf>,
    /// Whether index has been modified
    dirty: bool,
}

impl HashIndex {
    /// Create a new empty hash index
    pub fn new(column_name: &str, unique: bool) -> Self {
        Self {
            column_name: column_name.to_string(),
            unique,
            map: AHashMap::new(),
            entry_count: 0,
            path: None,
            dirty: false,
        }
    }

    /// Create with persistence path
    pub fn with_path(column_name: &str, unique: bool, path: PathBuf) -> Self {
        Self {
            column_name: column_name.to_string(),
            unique,
            map: AHashMap::new(),
            entry_count: 0,
            path: Some(path),
            dirty: false,
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(column_name: &str, unique: bool, capacity: usize) -> Self {
        Self {
            column_name: column_name.to_string(),
            unique,
            map: AHashMap::with_capacity(capacity),
            entry_count: 0,
            path: None,
            dirty: false,
        }
    }

    /// Column name
    pub fn column_name(&self) -> &str {
        &self.column_name
    }

    /// Whether unique
    pub fn is_unique(&self) -> bool {
        self.unique
    }

    /// Number of entries
    pub fn len(&self) -> u64 {
        self.entry_count
    }

    /// Whether empty
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Number of distinct keys
    pub fn distinct_keys(&self) -> usize {
        self.map.len()
    }

    // ========================================================================
    // Insert / Delete
    // ========================================================================

    /// Insert a key-rowid pair
    pub fn insert(&mut self, key: IndexKey, row_id: u64) -> io::Result<()> {
        if self.unique {
            let entry = self.map.entry(key);
            match entry {
                std::collections::hash_map::Entry::Occupied(e) => {
                    if !e.get().is_empty() {
                        return Err(io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            format!("Duplicate key in unique hash index '{}'", self.column_name),
                        ));
                    }
                    e.into_mut().push(row_id);
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(vec![row_id]);
                }
            }
        } else {
            self.map.entry(key).or_insert_with(Vec::new).push(row_id);
        }
        self.entry_count += 1;
        self.dirty = true;
        Ok(())
    }

    /// Remove a key-rowid pair
    pub fn remove(&mut self, key: &IndexKey, row_id: u64) -> bool {
        if let Some(ids) = self.map.get_mut(key) {
            if let Some(pos) = ids.iter().position(|&id| id == row_id) {
                ids.swap_remove(pos);
                self.entry_count -= 1;
                self.dirty = true;
                if ids.is_empty() {
                    self.map.remove(key);
                }
                return true;
            }
        }
        false
    }

    /// Bulk insert for building index from existing data
    pub fn bulk_insert(&mut self, pairs: Vec<(IndexKey, u64)>) {
        self.map.reserve(pairs.len());
        for (key, row_id) in pairs {
            self.map.entry(key).or_insert_with(Vec::new).push(row_id);
            self.entry_count += 1;
        }
        self.dirty = true;
    }

    // ========================================================================
    // Lookup
    // ========================================================================

    /// Point lookup: O(1) average case
    pub fn get(&self, key: &IndexKey) -> Option<&[u64]> {
        self.map.get(key).map(|v| v.as_slice())
    }

    /// Check if a key exists
    pub fn contains_key(&self, key: &IndexKey) -> bool {
        self.map.contains_key(key)
    }

    /// Get all keys (for iteration)
    pub fn keys(&self) -> impl Iterator<Item = &IndexKey> {
        self.map.keys()
    }

    // ========================================================================
    // Persistence
    // ========================================================================

    /// Save index to disk using bincode
    pub fn save(&mut self) -> io::Result<()> {
        if !self.dirty {
            return Ok(());
        }
        if let Some(path) = &self.path {
            // Serialize: column_name, unique, entries
            let serializable = HashIndexSer {
                column_name: self.column_name.clone(),
                unique: self.unique,
                entries: self
                    .map
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
                entry_count: self.entry_count,
            };
            let data = bincode::serialize(&serializable)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            std::fs::write(path, &data)?;
            self.dirty = false;
        }
        Ok(())
    }

    /// Load index from disk
    pub fn load(path: &Path) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        let ser: HashIndexSer = bincode::deserialize(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let mut map = AHashMap::with_capacity(ser.entries.len());
        for (k, v) in ser.entries {
            map.insert(k, v);
        }
        Ok(Self {
            column_name: ser.column_name,
            unique: ser.unique,
            map,
            entry_count: ser.entry_count,
            path: Some(path.to_path_buf()),
            dirty: false,
        })
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.map.clear();
        self.entry_count = 0;
        self.dirty = true;
    }
}

/// Serializable form of HashIndex
#[derive(Serialize, Deserialize)]
struct HashIndexSer {
    column_name: String,
    unique: bool,
    entries: Vec<(IndexKey, Vec<u64>)>,
    entry_count: u64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_insert_and_lookup() {
        let mut idx = HashIndex::new("_id", true);
        idx.insert(IndexKey::UInt(1), 0).unwrap();
        idx.insert(IndexKey::UInt(2), 1).unwrap();
        idx.insert(IndexKey::UInt(3), 2).unwrap();

        assert_eq!(idx.get(&IndexKey::UInt(1)), Some(&[0u64][..]));
        assert_eq!(idx.get(&IndexKey::UInt(2)), Some(&[1u64][..]));
        assert_eq!(idx.get(&IndexKey::UInt(99)), None);
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn test_hash_unique_constraint() {
        let mut idx = HashIndex::new("email", true);
        idx.insert(IndexKey::Str("a@b.com".into()), 0).unwrap();
        assert!(idx.insert(IndexKey::Str("a@b.com".into()), 1).is_err());
    }

    #[test]
    fn test_hash_non_unique() {
        let mut idx = HashIndex::new("status", false);
        idx.insert(IndexKey::Str("active".into()), 0).unwrap();
        idx.insert(IndexKey::Str("active".into()), 1).unwrap();
        assert_eq!(idx.get(&IndexKey::Str("active".into())).unwrap().len(), 2);
    }

    #[test]
    fn test_hash_remove() {
        let mut idx = HashIndex::new("col", false);
        idx.insert(IndexKey::Int(10), 0).unwrap();
        idx.insert(IndexKey::Int(10), 1).unwrap();
        assert!(idx.remove(&IndexKey::Int(10), 0));
        assert_eq!(idx.get(&IndexKey::Int(10)), Some(&[1u64][..]));
    }

    #[test]
    fn test_hash_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.hashidx");

        let mut idx = HashIndex::with_path("col", false, path.clone());
        idx.insert(IndexKey::Str("hello".into()), 42).unwrap();
        idx.save().unwrap();

        let loaded = HashIndex::load(&path).unwrap();
        assert_eq!(
            loaded.get(&IndexKey::Str("hello".into())),
            Some(&[42u64][..])
        );
    }
}
