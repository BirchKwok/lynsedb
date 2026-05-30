//! Version Store - Manages row version chains for MVCC
//!
//! Each row can have multiple versions, forming a chain from newest to oldest.
//! Visibility is determined by comparing version timestamps with the reader's snapshot.

use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::data::Value;

// ============================================================================
// Timestamps
// ============================================================================

/// Global monotonically increasing timestamp generator
static GLOBAL_TS: AtomicU64 = AtomicU64::new(1);

/// Get the next global timestamp
pub fn next_timestamp() -> u64 {
    GLOBAL_TS.fetch_add(1, Ordering::SeqCst)
}

/// Get the current timestamp without incrementing
pub fn current_timestamp() -> u64 {
    GLOBAL_TS.load(Ordering::SeqCst)
}

/// Set the timestamp counter (used on recovery)
pub fn set_timestamp(ts: u64) {
    GLOBAL_TS.store(ts, Ordering::SeqCst);
}

/// Special timestamp: transaction is still active (not yet committed)
pub const TS_ACTIVE: u64 = u64::MAX;

/// Special timestamp: version has no end (still the latest)
pub const TS_INF: u64 = u64::MAX - 1;

// ============================================================================
// Row Version
// ============================================================================

/// A single version of a row
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowVersion {
    /// Transaction ID that created this version
    pub begin_ts: u64,
    /// Transaction ID that invalidated this version (TS_INF if still valid)
    pub end_ts: u64,
    /// The row data (column_name → value)
    pub data: HashMap<String, Value>,
    /// Whether this version represents a deletion
    pub is_delete: bool,
}

impl RowVersion {
    /// Create a new version for an insert
    pub fn new_insert(begin_ts: u64, data: HashMap<String, Value>) -> Self {
        Self {
            begin_ts,
            end_ts: TS_INF,
            data,
            is_delete: false,
        }
    }

    /// Create a new version for an update
    pub fn new_update(begin_ts: u64, data: HashMap<String, Value>) -> Self {
        Self {
            begin_ts,
            end_ts: TS_INF,
            data,
            is_delete: false,
        }
    }

    /// Create a tombstone version for a delete
    pub fn new_delete(begin_ts: u64) -> Self {
        Self {
            begin_ts,
            end_ts: TS_INF,
            data: HashMap::new(),
            is_delete: true,
        }
    }

    /// Check if this version is visible to a snapshot at the given timestamp
    pub fn is_visible_at(&self, snapshot_ts: u64) -> bool {
        self.begin_ts <= snapshot_ts && snapshot_ts < self.end_ts
    }

    /// Mark this version as ended (superseded by a newer version)
    pub fn set_end(&mut self, end_ts: u64) {
        self.end_ts = end_ts;
    }

    /// Whether this version is still the latest (not yet superseded)
    pub fn is_latest(&self) -> bool {
        self.end_ts == TS_INF
    }
}

// ============================================================================
// Version Chain
// ============================================================================

/// A chain of versions for a single row, ordered newest-first
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionChain {
    /// Row ID
    pub row_id: u64,
    /// Versions ordered from newest to oldest
    pub versions: Vec<RowVersion>,
}

impl VersionChain {
    /// Create a new chain with an initial version
    pub fn new(row_id: u64, version: RowVersion) -> Self {
        Self {
            row_id,
            versions: vec![version],
        }
    }

    /// Add a new version (must be newer than all existing versions)
    pub fn add_version(&mut self, version: RowVersion) {
        // Set end_ts on the previous latest version
        if let Some(prev) = self.versions.first_mut() {
            if prev.is_latest() {
                prev.set_end(version.begin_ts);
            }
        }
        self.versions.insert(0, version); // newest first
    }

    /// Get the version visible at a given timestamp
    pub fn get_visible(&self, snapshot_ts: u64) -> Option<&RowVersion> {
        for v in &self.versions {
            if v.is_visible_at(snapshot_ts) {
                return if v.is_delete { None } else { Some(v) };
            }
        }
        None
    }

    /// Get the latest version (regardless of visibility)
    pub fn latest(&self) -> Option<&RowVersion> {
        self.versions.first()
    }

    /// Check if the row is deleted at a given timestamp
    pub fn is_deleted_at(&self, snapshot_ts: u64) -> bool {
        for v in &self.versions {
            if v.is_visible_at(snapshot_ts) {
                return v.is_delete;
            }
        }
        true // no visible version = effectively deleted
    }

    /// Remove versions older than the given timestamp (GC)
    /// Keeps at least one version for correctness
    pub fn gc(&mut self, oldest_active_ts: u64) -> usize {
        if self.versions.len() <= 1 {
            return 0;
        }
        let original_len = self.versions.len();
        // Keep versions that might still be visible to any active snapshot
        // A version is removable if end_ts < oldest_active_ts
        self.versions
            .retain(|v| v.is_latest() || v.end_ts > oldest_active_ts);
        // Always keep at least the latest version
        if self.versions.is_empty() {
            // This shouldn't happen due to is_latest() check, but be safe
        }
        original_len - self.versions.len()
    }

    /// Number of versions
    pub fn version_count(&self) -> usize {
        self.versions.len()
    }
}

// ============================================================================
// Version Store
// ============================================================================

/// Central store for all row versions across a table
///
/// Thread-safe: uses RwLock for concurrent access
pub struct VersionStore {
    /// Row ID → VersionChain
    chains: RwLock<HashMap<u64, VersionChain>>,
    /// Total version count (for GC heuristics)
    total_versions: AtomicU64,
}

impl VersionStore {
    /// Create a new empty version store
    pub fn new() -> Self {
        Self {
            chains: RwLock::new(HashMap::new()),
            total_versions: AtomicU64::new(0),
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            chains: RwLock::new(HashMap::with_capacity(capacity)),
            total_versions: AtomicU64::new(0),
        }
    }

    /// Insert a new row (creates initial version)
    pub fn insert(&self, row_id: u64, begin_ts: u64, data: HashMap<String, Value>) {
        let version = RowVersion::new_insert(begin_ts, data);
        let chain = VersionChain::new(row_id, version);
        self.chains.write().insert(row_id, chain);
        self.total_versions.fetch_add(1, Ordering::Relaxed);
    }

    /// Update a row (adds new version to chain)
    pub fn update(
        &self,
        row_id: u64,
        begin_ts: u64,
        data: HashMap<String, Value>,
    ) -> io::Result<()> {
        let mut chains = self.chains.write();
        let chain = chains.get_mut(&row_id).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Row {} not found", row_id))
        })?;
        let version = RowVersion::new_update(begin_ts, data);
        chain.add_version(version);
        self.total_versions.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Delete a row (adds tombstone version)
    pub fn delete(&self, row_id: u64, begin_ts: u64) -> io::Result<()> {
        let mut chains = self.chains.write();
        let chain = chains.get_mut(&row_id).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Row {} not found", row_id))
        })?;
        let version = RowVersion::new_delete(begin_ts);
        chain.add_version(version);
        self.total_versions.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Read a row at a specific snapshot timestamp
    pub fn read(&self, row_id: u64, snapshot_ts: u64) -> Option<HashMap<String, Value>> {
        let chains = self.chains.read();
        chains
            .get(&row_id)
            .and_then(|chain| chain.get_visible(snapshot_ts))
            .map(|v| v.data.clone())
    }

    /// Check if a row exists at a specific snapshot
    pub fn exists(&self, row_id: u64, snapshot_ts: u64) -> bool {
        let chains = self.chains.read();
        chains
            .get(&row_id)
            .map(|chain| !chain.is_deleted_at(snapshot_ts))
            .unwrap_or(false)
    }

    /// Get the latest version data for a row (for write operations)
    pub fn read_latest(&self, row_id: u64) -> Option<HashMap<String, Value>> {
        let chains = self.chains.read();
        chains
            .get(&row_id)
            .and_then(|chain| chain.latest())
            .filter(|v| !v.is_delete)
            .map(|v| v.data.clone())
    }

    /// Run garbage collection, removing versions older than oldest_active_ts
    pub fn gc(&self, oldest_active_ts: u64) -> usize {
        let mut chains = self.chains.write();
        let mut removed = 0;
        for chain in chains.values_mut() {
            removed += chain.gc(oldest_active_ts);
        }
        // Remove chains that only have tombstones older than oldest_active_ts
        chains.retain(|_, chain| {
            if let Some(latest) = chain.latest() {
                !(latest.is_delete && latest.end_ts < oldest_active_ts)
            } else {
                false
            }
        });
        self.total_versions
            .fetch_sub(removed as u64, Ordering::Relaxed);
        removed
    }

    /// Total number of version records
    pub fn total_versions(&self) -> u64 {
        self.total_versions.load(Ordering::Relaxed)
    }

    /// Number of tracked rows
    pub fn row_count(&self) -> usize {
        self.chains.read().len()
    }

    /// Get read access to the chains map (for snapshot isolation visibility checks)
    pub fn chains_ref(&self) -> parking_lot::RwLockReadGuard<'_, HashMap<u64, VersionChain>> {
        self.chains.read()
    }

    /// Whether GC should run (heuristic)
    pub fn needs_gc(&self) -> bool {
        let total = self.total_versions();
        let rows = self.row_count() as u64;
        // GC when average version count per row > 3
        rows > 0 && total > rows * 3
    }
}

impl Default for VersionStore {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(name: &str, age: i64) -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), Value::String(name.to_string()));
        m.insert("age".to_string(), Value::Int64(age));
        m
    }

    #[test]
    fn test_insert_and_read() {
        let store = VersionStore::new();
        let ts = next_timestamp();
        store.insert(1, ts, make_row("alice", 25));

        let data = store.read(1, ts).unwrap();
        assert_eq!(data.get("name").unwrap().as_str(), Some("alice"));
    }

    #[test]
    fn test_update_versioning() {
        let store = VersionStore::new();
        let ts1 = next_timestamp();
        store.insert(1, ts1, make_row("alice", 25));

        let ts2 = next_timestamp();
        store.update(1, ts2, make_row("alice", 30)).unwrap();

        // Old snapshot sees old data
        let old = store.read(1, ts1).unwrap();
        assert_eq!(old.get("age").unwrap().as_i64(), Some(25));

        // New snapshot sees new data
        let new = store.read(1, ts2).unwrap();
        assert_eq!(new.get("age").unwrap().as_i64(), Some(30));
    }

    #[test]
    fn test_delete() {
        let store = VersionStore::new();
        let ts1 = next_timestamp();
        store.insert(1, ts1, make_row("alice", 25));

        let ts2 = next_timestamp();
        store.delete(1, ts2).unwrap();

        assert!(store.exists(1, ts1));
        assert!(!store.exists(1, ts2));
        assert!(store.read(1, ts2).is_none());
    }

    #[test]
    fn test_gc() {
        let store = VersionStore::new();
        let ts1 = next_timestamp();
        store.insert(1, ts1, make_row("alice", 25));

        let ts2 = next_timestamp();
        store.update(1, ts2, make_row("alice", 30)).unwrap();

        let ts3 = next_timestamp();
        store.update(1, ts3, make_row("alice", 35)).unwrap();

        assert_eq!(store.total_versions(), 3);

        // GC versions older than ts3
        let removed = store.gc(ts3);
        assert!(removed > 0);
    }

    #[test]
    fn test_version_chain_visibility() {
        let mut chain = VersionChain::new(1, RowVersion::new_insert(10, make_row("v1", 1)));
        chain.add_version(RowVersion::new_update(20, make_row("v2", 2)));
        chain.add_version(RowVersion::new_update(30, make_row("v3", 3)));

        assert_eq!(
            chain
                .get_visible(10)
                .unwrap()
                .data
                .get("name")
                .unwrap()
                .as_str(),
            Some("v1")
        );
        assert_eq!(
            chain
                .get_visible(15)
                .unwrap()
                .data
                .get("name")
                .unwrap()
                .as_str(),
            Some("v1")
        );
        assert_eq!(
            chain
                .get_visible(20)
                .unwrap()
                .data
                .get("name")
                .unwrap()
                .as_str(),
            Some("v2")
        );
        assert_eq!(
            chain
                .get_visible(30)
                .unwrap()
                .data
                .get("name")
                .unwrap()
                .as_str(),
            Some("v3")
        );
    }
}
