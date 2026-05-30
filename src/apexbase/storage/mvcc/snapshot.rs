//! Snapshot Manager - Tracks active snapshots for MVCC visibility
//!
//! Each transaction/query gets a snapshot that defines which row versions
//! are visible. The manager tracks all active snapshots to determine
//! the oldest active timestamp for garbage collection.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use super::version_store::{current_timestamp, next_timestamp};

// ============================================================================
// Snapshot ID
// ============================================================================

/// Unique identifier for a snapshot
pub type SnapshotId = u64;

/// Global snapshot ID generator
static NEXT_SNAPSHOT_ID: AtomicU64 = AtomicU64::new(1);

fn next_snapshot_id() -> SnapshotId {
    NEXT_SNAPSHOT_ID.fetch_add(1, Ordering::SeqCst)
}

// ============================================================================
// Snapshot
// ============================================================================

/// A point-in-time snapshot for consistent reads
///
/// All reads within a snapshot see data as of `read_ts`.
/// The snapshot is automatically released when dropped (via SnapshotManager).
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Unique snapshot ID
    pub id: SnapshotId,
    /// The timestamp at which this snapshot was taken
    pub read_ts: u64,
    /// Whether this snapshot is for a read-only query or a read-write transaction
    pub read_only: bool,
}

impl Snapshot {
    /// Check if a version with given begin_ts and end_ts is visible
    #[inline]
    pub fn is_visible(&self, begin_ts: u64, end_ts: u64) -> bool {
        begin_ts <= self.read_ts && self.read_ts < end_ts
    }
}

// ============================================================================
// Snapshot Manager
// ============================================================================

/// Manages active snapshots across all concurrent transactions/queries
///
/// Thread-safe: all operations are guarded by RwLock.
///
/// Key responsibilities:
/// - Create new snapshots at the current timestamp
/// - Track all active snapshots
/// - Provide the oldest active timestamp for GC decisions
pub struct SnapshotManager {
    /// Active snapshots: snapshot_id → Snapshot
    active: RwLock<BTreeMap<SnapshotId, Snapshot>>,
    /// Cached oldest active timestamp (optimization)
    oldest_active_ts: AtomicU64,
}

impl SnapshotManager {
    /// Create a new snapshot manager
    pub fn new() -> Self {
        Self {
            active: RwLock::new(BTreeMap::new()),
            oldest_active_ts: AtomicU64::new(u64::MAX),
        }
    }

    /// Create a new read-only snapshot at the current timestamp
    pub fn create_snapshot(&self) -> Snapshot {
        let snap = Snapshot {
            id: next_snapshot_id(),
            read_ts: current_timestamp(),
            read_only: true,
        };
        self.register(snap.clone());
        snap
    }

    /// Create a new read-write snapshot (for transactions)
    pub fn create_rw_snapshot(&self) -> Snapshot {
        let ts = next_timestamp();
        let snap = Snapshot {
            id: next_snapshot_id(),
            read_ts: ts,
            read_only: false,
        };
        self.register(snap.clone());
        snap
    }

    /// Register a snapshot
    fn register(&self, snap: Snapshot) {
        let ts = snap.read_ts;
        self.active.write().insert(snap.id, snap);
        // Update oldest if this is older
        self.oldest_active_ts.fetch_min(ts, Ordering::SeqCst);
    }

    /// Release (drop) a snapshot
    pub fn release(&self, snapshot_id: SnapshotId) {
        let mut active = self.active.write();
        active.remove(&snapshot_id);
        // Recalculate oldest active timestamp
        let oldest = active.values().map(|s| s.read_ts).min().unwrap_or(u64::MAX);
        self.oldest_active_ts.store(oldest, Ordering::SeqCst);
    }

    /// Get the oldest active snapshot timestamp
    /// This is used by GC to determine which versions can be safely removed
    pub fn oldest_active_timestamp(&self) -> u64 {
        self.oldest_active_ts.load(Ordering::SeqCst)
    }

    /// Number of active snapshots
    pub fn active_count(&self) -> usize {
        self.active.read().len()
    }

    /// Check if there are any active snapshots
    pub fn has_active_snapshots(&self) -> bool {
        !self.active.read().is_empty()
    }

    /// Get a snapshot by ID (for debugging/monitoring)
    pub fn get_snapshot(&self, id: SnapshotId) -> Option<Snapshot> {
        self.active.read().get(&id).cloned()
    }
}

impl Default for SnapshotManager {
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

    #[test]
    fn test_snapshot_creation() {
        let mgr = SnapshotManager::new();
        let snap1 = mgr.create_snapshot();
        let snap2 = mgr.create_snapshot();

        assert_ne!(snap1.id, snap2.id);
        assert!(snap2.read_ts >= snap1.read_ts);
        assert_eq!(mgr.active_count(), 2);
    }

    #[test]
    fn test_snapshot_release() {
        let mgr = SnapshotManager::new();
        let snap1 = mgr.create_snapshot();
        let snap2 = mgr.create_snapshot();

        mgr.release(snap1.id);
        assert_eq!(mgr.active_count(), 1);

        mgr.release(snap2.id);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_oldest_active_timestamp() {
        let mgr = SnapshotManager::new();

        // No snapshots: oldest is MAX
        assert_eq!(mgr.oldest_active_timestamp(), u64::MAX);

        let snap1 = mgr.create_snapshot();
        let ts1 = snap1.read_ts;

        let _snap2 = mgr.create_snapshot();

        // Oldest should be snap1's timestamp
        assert_eq!(mgr.oldest_active_timestamp(), ts1);

        // Release snap1, oldest should update
        mgr.release(snap1.id);
        assert!(mgr.oldest_active_timestamp() >= ts1);
    }

    #[test]
    fn test_snapshot_visibility() {
        let snap = Snapshot {
            id: 1,
            read_ts: 100,
            read_only: true,
        };

        // Version [50, INF) visible at ts=100
        assert!(snap.is_visible(50, u64::MAX - 1));
        // Version [50, 80) NOT visible at ts=100
        assert!(!snap.is_visible(50, 80));
        // Version [150, INF) NOT visible at ts=100
        assert!(!snap.is_visible(150, u64::MAX - 1));
    }
}
