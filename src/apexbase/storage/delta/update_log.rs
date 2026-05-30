//! Delta Store - Row-level update/delete tracking
//!
//! Provides append-only logs for updates and a bitmap for deletes,
//! eliminating the need to rewrite the entire columnar file on mutations.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::data::Value;

// ============================================================================
// Delete Bitmap
// ============================================================================

/// Bitmap tracking deleted row IDs
///
/// Uses a sorted Vec<u64> for space efficiency on sparse deletes,
/// and a HashSet for fast lookup on dense deletes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteBitmap {
    /// Deleted row IDs (sorted)
    deleted: Vec<u64>,
    /// Cached set for O(1) lookup (rebuilt on load)
    #[serde(skip)]
    lookup: Option<std::collections::HashSet<u64>>,
}

impl DeleteBitmap {
    /// Create empty bitmap
    pub fn new() -> Self {
        Self {
            deleted: Vec::new(),
            lookup: Some(std::collections::HashSet::new()),
        }
    }

    /// Mark a row as deleted
    pub fn delete(&mut self, row_id: u64) {
        if let Some(ref mut set) = self.lookup {
            if set.insert(row_id) {
                // Maintain sorted order via binary search insert
                match self.deleted.binary_search(&row_id) {
                    Ok(_) => {} // already present
                    Err(pos) => self.deleted.insert(pos, row_id),
                }
            }
        } else {
            match self.deleted.binary_search(&row_id) {
                Ok(_) => {}
                Err(pos) => self.deleted.insert(pos, row_id),
            }
            self.rebuild_lookup();
        }
    }

    /// Check if a row is deleted: O(1) with lookup set
    pub fn is_deleted(&self, row_id: u64) -> bool {
        if let Some(ref set) = self.lookup {
            set.contains(&row_id)
        } else {
            self.deleted.binary_search(&row_id).is_ok()
        }
    }

    /// Number of deleted rows
    pub fn count(&self) -> usize {
        self.deleted.len()
    }

    /// Whether empty
    pub fn is_empty(&self) -> bool {
        self.deleted.is_empty()
    }

    /// Get all deleted row IDs
    pub fn deleted_ids(&self) -> &[u64] {
        &self.deleted
    }

    /// Clear all deletes
    pub fn clear(&mut self) {
        self.deleted.clear();
        self.lookup = Some(std::collections::HashSet::new());
    }

    /// Undelete a row
    pub fn undelete(&mut self, row_id: u64) {
        if let Ok(pos) = self.deleted.binary_search(&row_id) {
            self.deleted.remove(pos);
        }
        if let Some(ref mut set) = self.lookup {
            set.remove(&row_id);
        }
    }

    /// Rebuild the lookup set from sorted vec
    fn rebuild_lookup(&mut self) {
        self.lookup = Some(self.deleted.iter().copied().collect());
    }

    /// Ensure lookup set is built (call after deserialization)
    pub fn ensure_lookup(&mut self) {
        if self.lookup.is_none() {
            self.rebuild_lookup();
        }
    }
}

impl Default for DeleteBitmap {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Delta Record
// ============================================================================

/// A single update record: which row, which column, what new value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaRecord {
    /// Row ID being updated
    pub row_id: u64,
    /// Column name
    pub column_name: String,
    /// New value
    pub new_value: Value,
    /// Transaction ID that created this delta (for MVCC)
    pub txn_id: u64,
    /// Timestamp of the update
    pub timestamp: i64,
}

// ============================================================================
// Delta Store
// ============================================================================

/// Manages all pending updates and deletes for a table
///
/// Write path:
/// 1. UPDATE → append DeltaRecord to update_log
/// 2. DELETE → set bit in delete_bitmap
/// 3. Periodically flush to .delta file
///
/// Read path:
/// 1. Read base columnar data
/// 2. Apply delete bitmap (skip deleted rows)
/// 3. Overlay update_log (latest value wins)
pub struct DeltaStore {
    /// Path for persistence
    path: PathBuf,
    /// Delete bitmap
    delete_bitmap: DeleteBitmap,
    /// Update log: row_id → (column_name → latest DeltaRecord)
    /// Collapsed to latest value per (row, col) for fast reads
    updates: HashMap<u64, HashMap<String, DeltaRecord>>,
    /// Sequential log for WAL persistence
    log: Vec<DeltaRecord>,
    /// Whether store has been modified since last save
    dirty: bool,
    /// Next transaction ID counter (local, pre-MVCC)
    next_txn_id: u64,
}

impl DeltaStore {
    /// Create a new empty delta store
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            delete_bitmap: DeleteBitmap::new(),
            updates: HashMap::new(),
            log: Vec::new(),
            dirty: false,
            next_txn_id: 1,
        }
    }

    /// Load delta store from disk
    pub fn load(path: &Path) -> io::Result<Self> {
        let delta_path = Self::file_path(path);
        if !delta_path.exists() {
            return Ok(Self::new(path));
        }

        let data = std::fs::read(&delta_path)?;
        let persisted: DeltaStorePersisted = bincode::deserialize(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        let mut store = Self {
            path: path.to_path_buf(),
            delete_bitmap: persisted.delete_bitmap,
            updates: HashMap::new(),
            log: persisted.log.clone(),
            dirty: false,
            next_txn_id: persisted.next_txn_id,
        };

        store.delete_bitmap.ensure_lookup();

        // Rebuild updates map from log
        for record in &persisted.log {
            store
                .updates
                .entry(record.row_id)
                .or_insert_with(HashMap::new)
                .insert(record.column_name.clone(), record.clone());
        }

        Ok(store)
    }

    /// File path for the delta store
    fn file_path(base_path: &Path) -> PathBuf {
        let mut p = base_path.to_path_buf();
        let name = p.file_name().unwrap_or_default().to_string_lossy();
        p.set_file_name(format!("{}.deltastore", name));
        p
    }

    // ========================================================================
    // Write Operations
    // ========================================================================

    /// Record a row deletion
    pub fn delete_row(&mut self, row_id: u64) {
        self.delete_bitmap.delete(row_id);
        // Also remove any pending updates for this row
        self.updates.remove(&row_id);
        self.dirty = true;
    }

    /// Record a cell update
    pub fn update_cell(&mut self, row_id: u64, column_name: &str, new_value: Value) {
        let txn_id = self.next_txn_id;
        self.next_txn_id += 1;

        let record = DeltaRecord {
            row_id,
            column_name: column_name.to_string(),
            new_value,
            txn_id,
            timestamp: 0,
        };

        self.updates
            .entry(row_id)
            .or_insert_with(HashMap::new)
            .insert(column_name.to_string(), record);
        self.dirty = true;
    }

    /// Record a full row update
    pub fn update_row(&mut self, row_id: u64, values: &HashMap<String, Value>) {
        for (col, val) in values {
            self.update_cell(row_id, col, val.clone());
        }
    }

    /// Batch update multiple rows in a single call (avoids repeated lock acquisitions).
    /// All updates share one txn_id block for efficiency.
    pub fn batch_update_rows(&mut self, batch: &[(u64, &str, Value)]) {
        for (row_id, column_name, new_value) in batch {
            let txn_id = self.next_txn_id;
            self.next_txn_id += 1;
            let record = DeltaRecord {
                row_id: *row_id,
                column_name: column_name.to_string(),
                new_value: new_value.clone(),
                txn_id,
                timestamp: 0,
            };
            self.updates
                .entry(*row_id)
                .or_insert_with(HashMap::new)
                .insert(column_name.to_string(), record);
        }
        if !batch.is_empty() {
            self.dirty = true;
        }
    }

    // ========================================================================
    // Read Operations
    // ========================================================================

    /// Check if a row is deleted
    pub fn is_deleted(&self, row_id: u64) -> bool {
        self.delete_bitmap.is_deleted(row_id)
    }

    /// Get the latest value for a cell, if updated
    pub fn get_updated_value(&self, row_id: u64, column_name: &str) -> Option<&Value> {
        self.updates
            .get(&row_id)
            .and_then(|cols| cols.get(column_name))
            .map(|r| &r.new_value)
    }

    /// Get all updated columns for a row
    pub fn get_row_updates(&self, row_id: u64) -> Option<&HashMap<String, DeltaRecord>> {
        self.updates.get(&row_id)
    }

    /// Check if a row has any pending updates
    pub fn has_updates(&self, row_id: u64) -> bool {
        self.updates.contains_key(&row_id)
    }

    /// Number of pending deletes
    pub fn delete_count(&self) -> usize {
        self.delete_bitmap.count()
    }

    /// Number of pending update records
    pub fn update_count(&self) -> usize {
        self.updates.values().map(|m| m.len()).sum()
    }

    /// Whether the delta store is empty (no pending changes)
    pub fn is_empty(&self) -> bool {
        self.delete_bitmap.is_empty() && self.updates.is_empty()
    }

    /// Get delete bitmap reference
    pub fn delete_bitmap(&self) -> &DeleteBitmap {
        &self.delete_bitmap
    }

    /// Get all updates (for merge/compaction)
    pub fn all_updates(&self) -> &HashMap<u64, HashMap<String, DeltaRecord>> {
        &self.updates
    }

    /// Return true when any pending update touches the given column.
    pub fn updates_column(&self, column_name: &str) -> bool {
        self.updates.values().any(|cols| {
            cols.keys()
                .any(|name| name.eq_ignore_ascii_case(column_name))
        })
    }

    /// Return row IDs whose latest pending update sets `column_name` to `value`.
    pub fn rows_with_string_update(&self, column_name: &str, value: &str) -> Vec<u64> {
        self.updates
            .iter()
            .filter_map(|(row_id, cols)| {
                cols.iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(column_name))
                    .and_then(|(_, record)| match &record.new_value {
                        Value::String(s) if s == value => Some(*row_id),
                        _ => None,
                    })
            })
            .collect()
    }

    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    // ========================================================================
    // Persistence
    // ========================================================================

    /// Save delta store to disk
    pub fn save(&mut self) -> io::Result<()> {
        if !self.dirty {
            return Ok(());
        }
        let delta_path = Self::file_path(&self.path);

        // Create parent directory if needed
        if let Some(parent) = delta_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Persist only the COLLAPSED updates (one record per unique (row, col) pair).
        // This prevents unbounded log growth across repeated UPDATE calls on the same rows.
        let collapsed_log: Vec<DeltaRecord> = self
            .updates
            .values()
            .flat_map(|col_map| col_map.values().cloned())
            .collect();

        let persisted = DeltaStorePersisted {
            delete_bitmap: self.delete_bitmap.clone(),
            log: collapsed_log,
            next_txn_id: self.next_txn_id,
        };

        let data = bincode::serialize(&persisted)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        // Atomic write: write to .tmp then rename to prevent corruption on crash
        let tmp_path = delta_path.with_extension("deltastore.tmp");
        std::fs::write(&tmp_path, &data)?;
        std::fs::rename(&tmp_path, &delta_path)?;
        self.dirty = false;
        Ok(())
    }

    /// Clear all deltas (after compaction into base file)
    pub fn clear(&mut self) {
        self.delete_bitmap.clear();
        self.updates.clear();
        self.log.clear();
        self.next_txn_id = 1;
        self.dirty = true;
    }

    /// Remove the delta file from disk
    pub fn remove_file(&self) -> io::Result<()> {
        let delta_path = Self::file_path(&self.path);
        if delta_path.exists() {
            std::fs::remove_file(&delta_path)?;
        }
        Ok(())
    }

    /// Check if compaction is needed (heuristic)
    pub fn needs_compaction(&self, base_row_count: u64) -> bool {
        let delta_count = (self.delete_count() + self.update_count()) as u64;
        // Compact when deltas exceed 20% of base rows or absolute threshold
        delta_count > 10_000 || (base_row_count > 0 && delta_count * 5 > base_row_count)
    }
}

/// Serializable form of DeltaStore
#[derive(Serialize, Deserialize)]
struct DeltaStorePersisted {
    delete_bitmap: DeleteBitmap,
    log: Vec<DeltaRecord>,
    next_txn_id: u64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delete_bitmap() {
        let mut bm = DeleteBitmap::new();
        bm.delete(5);
        bm.delete(10);
        bm.delete(3);
        assert!(bm.is_deleted(5));
        assert!(bm.is_deleted(10));
        assert!(!bm.is_deleted(7));
        assert_eq!(bm.count(), 3);
        assert_eq!(bm.deleted_ids(), &[3, 5, 10]);
    }

    #[test]
    fn test_delete_bitmap_undelete() {
        let mut bm = DeleteBitmap::new();
        bm.delete(5);
        assert!(bm.is_deleted(5));
        bm.undelete(5);
        assert!(!bm.is_deleted(5));
    }

    #[test]
    fn test_delta_store_update() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.apex");
        let mut store = DeltaStore::new(&path);

        store.update_cell(0, "name", Value::String("alice_updated".into()));
        store.update_cell(0, "age", Value::Int64(30));

        assert_eq!(
            store.get_updated_value(0, "name"),
            Some(&Value::String("alice_updated".into()))
        );
        assert_eq!(store.get_updated_value(0, "age"), Some(&Value::Int64(30)));
        assert_eq!(store.get_updated_value(1, "name"), None);
    }

    #[test]
    fn test_delta_store_delete() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.apex");
        let mut store = DeltaStore::new(&path);

        store.update_cell(5, "x", Value::Int64(42));
        store.delete_row(5);

        assert!(store.is_deleted(5));
        // Updates should be cleared for deleted rows
        assert!(!store.has_updates(5));
    }

    #[test]
    fn test_delta_store_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.apex");

        {
            let mut store = DeltaStore::new(&path);
            store.delete_row(3);
            store.update_cell(7, "score", Value::Float64(99.5));
            store.save().unwrap();
        }

        let store = DeltaStore::load(&path).unwrap();
        assert!(store.is_deleted(3));
        assert_eq!(
            store.get_updated_value(7, "score"),
            Some(&Value::Float64(99.5))
        );
    }

    #[test]
    fn test_needs_compaction() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.apex");
        let mut store = DeltaStore::new(&path);

        // Small delta on large base: no compaction
        store.delete_row(1);
        assert!(!store.needs_compaction(10000));

        // Large delta: needs compaction
        for i in 0..10001 {
            store.delete_row(i);
        }
        assert!(store.needs_compaction(100000));
    }
}
