//! Transaction Manager - Lifecycle management for transactions
//!
//! Coordinates transaction creation, commit, and abort across the system.
//! Integrates with MVCC (VersionStore + SnapshotManager) and ConflictDetector.

use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use parking_lot::RwLock;

use super::conflict::{ConflictDetector, ConflictResult};
use super::context::{TxnContext, TxnWrite};
use crate::storage::mvcc::gc::GarbageCollector;
use crate::storage::mvcc::snapshot::{Snapshot, SnapshotManager};
use crate::storage::mvcc::version_store::{next_timestamp, VersionStore};

// ============================================================================
// Global TxnManager Singleton
// ============================================================================

static TXN_MANAGER: Lazy<TxnManager> = Lazy::new(TxnManager::new_standalone);

/// Get the global transaction manager
pub fn txn_manager() -> &'static TxnManager {
    &TXN_MANAGER
}

// ============================================================================
// Transaction ID
// ============================================================================

/// Unique transaction identifier
pub type TxnId = u64;

/// Global transaction ID generator
static NEXT_TXN_ID: AtomicU64 = AtomicU64::new(1);

fn next_txn_id() -> TxnId {
    NEXT_TXN_ID.fetch_add(1, Ordering::SeqCst)
}

// ============================================================================
// Transaction Status
// ============================================================================

/// Current status of a transaction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnStatus {
    /// Transaction is active (reads/writes in progress)
    Active,
    /// Transaction is in validation phase (checking for conflicts)
    Validating,
    /// Transaction has been committed
    Committed,
    /// Transaction has been aborted
    Aborted,
}

// ============================================================================
// Active Transaction Entry
// ============================================================================

/// Default transaction timeout (30 seconds)
const DEFAULT_TXN_TIMEOUT_SECS: u64 = 30;

/// Tracking entry for an active transaction
struct ActiveTxn {
    context: TxnContext,
    snapshot: Snapshot,
    status: TxnStatus,
    started_at: Instant,
    /// Last activity timestamp — refreshed on every with_context call.
    /// Timeout is based on idle time (elapsed since last activity), not total age.
    last_activity: Instant,
}

// ============================================================================
// Transaction Manager
// ============================================================================

/// Central transaction manager
///
/// Coordinates the lifecycle of all transactions:
/// 1. BEGIN: Create TxnContext + Snapshot
/// 2. READ/WRITE: Track in TxnContext
/// 3. COMMIT: Validate (OCC) → Apply writes → Release snapshot
/// 4. ROLLBACK: Discard writes → Release snapshot
///
/// Thread-safe: all operations are guarded by locks.
pub struct TxnManager {
    /// Active transactions: txn_id → ActiveTxn
    active_txns: RwLock<HashMap<TxnId, ActiveTxn>>,
    /// Snapshot manager (shared with query engine)
    snapshot_manager: Arc<SnapshotManager>,
    /// Conflict detector
    conflict_detector: ConflictDetector,
    /// Per-table version stores for MVCC visibility
    version_stores: RwLock<HashMap<String, Arc<VersionStore>>>,
    /// Garbage collector for old MVCC versions
    gc: GarbageCollector,
    /// Total committed transactions (for monitoring)
    total_committed: AtomicU64,
    /// Total aborted transactions (for monitoring)
    total_aborted: AtomicU64,
    /// Transaction timeout duration
    txn_timeout: Duration,
}

impl TxnManager {
    /// Create a new transaction manager
    pub fn new(snapshot_manager: Arc<SnapshotManager>) -> Self {
        Self {
            active_txns: RwLock::new(HashMap::new()),
            snapshot_manager,
            conflict_detector: ConflictDetector::new(),
            version_stores: RwLock::new(HashMap::new()),
            gc: GarbageCollector::new(),
            total_committed: AtomicU64::new(0),
            total_aborted: AtomicU64::new(0),
            txn_timeout: Duration::from_secs(DEFAULT_TXN_TIMEOUT_SECS),
        }
    }

    /// Create a new transaction manager with a fresh snapshot manager
    pub fn new_standalone() -> Self {
        Self::new(Arc::new(SnapshotManager::new()))
    }

    /// Get or create a VersionStore for a table
    pub fn get_version_store(&self, table: &str) -> Arc<VersionStore> {
        {
            let stores = self.version_stores.read();
            if let Some(store) = stores.get(table) {
                return Arc::clone(store);
            }
        }
        let mut stores = self.version_stores.write();
        stores
            .entry(table.to_string())
            .or_insert_with(|| Arc::new(VersionStore::new()))
            .clone()
    }

    /// Get reference to the snapshot manager
    pub fn snapshot_manager(&self) -> &Arc<SnapshotManager> {
        &self.snapshot_manager
    }

    /// Get reference to the conflict detector
    pub fn conflict_detector(&self) -> &ConflictDetector {
        &self.conflict_detector
    }

    // ========================================================================
    // Transaction Lifecycle
    // ========================================================================

    /// BEGIN TRANSACTION - Create a new read-write transaction
    pub fn begin(&self) -> TxnId {
        let txn_id = next_txn_id();
        let snapshot = self.snapshot_manager.create_rw_snapshot();
        let context = TxnContext::new(txn_id, snapshot.read_ts, false);

        let now = Instant::now();
        self.active_txns.write().insert(
            txn_id,
            ActiveTxn {
                context,
                snapshot: snapshot.clone(),
                status: TxnStatus::Active,
                started_at: now,
                last_activity: now,
            },
        );

        // Opportunistically abort timed-out transactions
        self.abort_timed_out_txns();

        txn_id
    }

    /// BEGIN READ ONLY - Create a new read-only transaction
    pub fn begin_read_only(&self) -> TxnId {
        let txn_id = next_txn_id();
        let snapshot = self.snapshot_manager.create_snapshot();
        let context = TxnContext::new(txn_id, snapshot.read_ts, true);

        let now = Instant::now();
        self.active_txns.write().insert(
            txn_id,
            ActiveTxn {
                context,
                snapshot: snapshot.clone(),
                status: TxnStatus::Active,
                started_at: now,
                last_activity: now,
            },
        );

        txn_id
    }

    /// COMMIT - Validate and commit a transaction
    ///
    /// OCC validation protocol:
    /// 1. Acquire write intent locks (early conflict detection)
    /// 2. Validate read set (has anything we read been modified?)
    /// 3. Validate write set (has any row we want to write been written?)
    /// 4. If valid: apply writes, record commit, release snapshot
    /// 5. If conflict: abort transaction
    pub fn commit(&self, txn_id: TxnId) -> io::Result<()> {
        self.commit_with_writes(txn_id).map(|_| ())
    }

    /// COMMIT and return the committed write set to the caller.
    ///
    /// This lets the query executor apply buffered writes to storage without
    /// cloning the write set before validation.
    pub fn commit_with_writes(&self, txn_id: TxnId) -> io::Result<Vec<TxnWrite>> {
        let mut txns = self.active_txns.write();
        let txn = txns.get_mut(&txn_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Transaction {} not found", txn_id),
            )
        })?;

        if txn.status != TxnStatus::Active {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Transaction {} is not active (status: {:?})",
                    txn_id, txn.status
                ),
            ));
        }

        txn.status = TxnStatus::Validating;

        // For read-only transactions, just release
        if txn.context.is_read_only() || !txn.context.has_writes() {
            txn.status = TxnStatus::Committed;
            txn.context.set_finished();
            let snapshot_id = txn.snapshot.id;
            txns.remove(&txn_id);
            drop(txns);
            self.snapshot_manager.release(snapshot_id);
            self.total_committed.fetch_add(1, Ordering::Relaxed);
            return Ok(Vec::new());
        }

        // OCC Validation
        let validation_result = self.conflict_detector.validate(&txn.context);
        if !validation_result.is_ok() {
            // Conflict detected → abort
            txn.status = TxnStatus::Aborted;
            txn.context.set_finished();
            self.conflict_detector.record_abort(&txn.context);
            let snapshot_id = txn.snapshot.id;
            txns.remove(&txn_id);
            drop(txns);
            self.snapshot_manager.release(snapshot_id);
            self.total_aborted.fetch_add(1, Ordering::Relaxed);
            return validation_result.to_io_result().map(|_| Vec::new());
        }

        // Commit successful
        let commit_ts = next_timestamp();
        self.conflict_detector
            .record_commit(&txn.context, commit_ts);

        // Extract write set for VersionStore recording and executor storage apply.
        let writes = txn.context.take_write_set();

        txn.status = TxnStatus::Committed;
        txn.context.set_finished();
        let snapshot_id = txn.snapshot.id;
        txns.remove(&txn_id);
        drop(txns);

        // Record committed writes in per-table VersionStores for MVCC visibility
        for write in &writes {
            match write {
                TxnWrite::Insert {
                    table,
                    row_id,
                    data,
                    ..
                } => {
                    let store = self.get_version_store(table);
                    store.insert(*row_id, commit_ts, data.clone());
                }
                TxnWrite::Delete {
                    table,
                    row_id,
                    old_data,
                    ..
                } => {
                    let store = self.get_version_store(table);
                    // Ensure base version exists so older snapshots can see the row
                    if store.read_latest(*row_id).is_none() && !old_data.is_empty() {
                        store.insert(*row_id, 1, old_data.clone()); // begin_ts=1: existed from start
                    }
                    let _ = store.delete(*row_id, commit_ts);
                }
                TxnWrite::Update {
                    table,
                    row_id,
                    old_data,
                    new_data,
                    ..
                } => {
                    let store = self.get_version_store(table);
                    // Ensure base version exists so older snapshots can see old data
                    if store.read_latest(*row_id).is_none() && !old_data.is_empty() {
                        store.insert(*row_id, 1, old_data.clone()); // begin_ts=1: existed from start
                    }
                    let _ = store.update(*row_id, commit_ts, new_data.clone());
                }
            }
        }

        self.snapshot_manager.release(snapshot_id);
        self.total_committed.fetch_add(1, Ordering::Relaxed);

        // Advance watermark periodically
        let oldest = self.snapshot_manager.oldest_active_timestamp();
        if oldest != u64::MAX {
            self.conflict_detector.advance_watermark(oldest);
        }

        // Auto-trigger GC for old MVCC versions
        for store in self.version_stores.read().values() {
            self.gc.maybe_run(store, &self.snapshot_manager);
        }

        Ok(writes)
    }

    /// ROLLBACK - Abort a transaction and discard all writes
    pub fn rollback(&self, txn_id: TxnId) -> io::Result<()> {
        let mut txns = self.active_txns.write();
        let txn = txns.get_mut(&txn_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Transaction {} not found", txn_id),
            )
        })?;

        txn.status = TxnStatus::Aborted;
        txn.context.set_finished();
        self.conflict_detector.record_abort(&txn.context);

        let snapshot_id = txn.snapshot.id;
        txns.remove(&txn_id);
        drop(txns);

        self.snapshot_manager.release(snapshot_id);
        self.total_aborted.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    // ========================================================================
    // Context Access (for read/write operations within a transaction)
    // ========================================================================

    /// Get the snapshot timestamp for a transaction (for MVCC visibility checks)
    pub fn get_snapshot_ts(&self, txn_id: TxnId) -> io::Result<u64> {
        let txns = self.active_txns.read();
        let txn = txns.get(&txn_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Transaction {} not found", txn_id),
            )
        })?;
        Ok(txn.snapshot.read_ts)
    }

    /// Check if a row version is visible to a transaction's snapshot
    pub fn is_visible(&self, txn_id: TxnId, table: &str, row_id: u64) -> io::Result<bool> {
        let snapshot_ts = self.get_snapshot_ts(txn_id)?;
        let store = self.get_version_store(table);
        Ok(store.exists(row_id, snapshot_ts))
    }

    /// Read a row's visible version for a transaction
    pub fn read_versioned(
        &self,
        txn_id: TxnId,
        table: &str,
        row_id: u64,
    ) -> io::Result<Option<HashMap<String, crate::data::Value>>> {
        let snapshot_ts = self.get_snapshot_ts(txn_id)?;
        let store = self.get_version_store(table);
        Ok(store.read(row_id, snapshot_ts))
    }

    /// Execute a closure with mutable access to the transaction context.
    /// Checks for transaction timeout before executing.
    pub fn with_context<F, R>(&self, txn_id: TxnId, f: F) -> io::Result<R>
    where
        F: FnOnce(&mut TxnContext) -> io::Result<R>,
    {
        let mut txns = self.active_txns.write();
        let txn = txns.get_mut(&txn_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Transaction {} not found", txn_id),
            )
        })?;
        if txn.status != TxnStatus::Active {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Transaction is not active",
            ));
        }
        // Check idle timeout (time since last activity, not total age)
        if txn.last_activity.elapsed() > self.txn_timeout {
            txn.status = TxnStatus::Aborted;
            txn.context.set_finished();
            self.conflict_detector.record_abort(&txn.context);
            let snapshot_id = txn.snapshot.id;
            txns.remove(&txn_id);
            drop(txns);
            self.snapshot_manager.release(snapshot_id);
            self.total_aborted.fetch_add(1, Ordering::Relaxed);
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "Transaction {} timed out (idle > {:?})",
                    txn_id, self.txn_timeout
                ),
            ));
        }
        // Refresh last activity timestamp
        txn.last_activity = Instant::now();
        f(&mut txn.context)
    }

    // ========================================================================
    // Monitoring
    // ========================================================================

    /// Number of active transactions
    pub fn active_count(&self) -> usize {
        self.active_txns.read().len()
    }

    /// Total committed transactions
    pub fn total_committed(&self) -> u64 {
        self.total_committed.load(Ordering::Relaxed)
    }

    /// Total aborted transactions
    pub fn total_aborted(&self) -> u64 {
        self.total_aborted.load(Ordering::Relaxed)
    }

    /// Check if a transaction is active
    pub fn is_active(&self, txn_id: TxnId) -> bool {
        self.active_txns
            .read()
            .get(&txn_id)
            .map(|t| t.status == TxnStatus::Active)
            .unwrap_or(false)
    }

    /// Check if a specific transaction has timed out
    pub fn is_timed_out(&self, txn_id: TxnId) -> bool {
        self.active_txns
            .read()
            .get(&txn_id)
            .map(|t| t.last_activity.elapsed() > self.txn_timeout)
            .unwrap_or(false)
    }

    /// Abort all transactions that have exceeded the timeout.
    /// Called opportunistically from begin() to prevent leaked snapshots.
    fn abort_timed_out_txns(&self) {
        let mut txns = self.active_txns.write();
        let timed_out: Vec<TxnId> = txns
            .iter()
            .filter(|(_, t)| {
                t.status == TxnStatus::Active && t.last_activity.elapsed() > self.txn_timeout
            })
            .map(|(id, _)| *id)
            .collect();
        for txn_id in timed_out {
            if let Some(txn) = txns.get_mut(&txn_id) {
                txn.status = TxnStatus::Aborted;
                txn.context.set_finished();
                self.conflict_detector.record_abort(&txn.context);
                let snapshot_id = txn.snapshot.id;
                // Can't call snapshot_manager.release while holding txns write lock
                // because release only needs snapshot_manager's own lock.
                // But since snapshot_manager is separate, this is fine.
                self.snapshot_manager.release(snapshot_id);
                self.total_aborted.fetch_add(1, Ordering::Relaxed);
            }
            txns.remove(&txn_id);
        }
    }

    /// Get the current transaction timeout duration
    pub fn timeout(&self) -> Duration {
        self.txn_timeout
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Value;

    fn make_row(name: &str) -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), Value::String(name.to_string()));
        m
    }

    #[test]
    fn test_begin_commit() {
        let mgr = TxnManager::new_standalone();
        let txn_id = mgr.begin();

        assert!(mgr.is_active(txn_id));
        assert_eq!(mgr.active_count(), 1);

        mgr.commit(txn_id).unwrap();
        assert!(!mgr.is_active(txn_id));
        assert_eq!(mgr.active_count(), 0);
        assert_eq!(mgr.total_committed(), 1);
    }

    #[test]
    fn test_begin_rollback() {
        let mgr = TxnManager::new_standalone();
        let txn_id = mgr.begin();

        mgr.with_context(txn_id, |ctx| {
            ctx.buffer_insert("users", 0, make_row("alice"))
        })
        .unwrap();

        mgr.rollback(txn_id).unwrap();
        assert_eq!(mgr.total_aborted(), 1);
    }

    #[test]
    fn test_read_only_commit() {
        let mgr = TxnManager::new_standalone();
        let txn_id = mgr.begin_read_only();

        mgr.with_context(txn_id, |ctx| {
            ctx.record_read("users", 0, 50);
            Ok(())
        })
        .unwrap();

        mgr.commit(txn_id).unwrap();
        assert_eq!(mgr.total_committed(), 1);
    }

    #[test]
    fn test_write_write_conflict_detected() {
        let mgr = TxnManager::new_standalone();

        // Both txns start before either commits (concurrent scenario)
        let txn1 = mgr.begin();
        let txn2 = mgr.begin();

        // Txn 1: write row 0
        mgr.with_context(txn1, |ctx| ctx.buffer_insert("users", 0, make_row("alice")))
            .unwrap();

        // Txn 2: also write row 0
        mgr.with_context(txn2, |ctx| {
            ctx.buffer_update("users", 0, make_row("alice"), make_row("alice2"))
        })
        .unwrap();

        // Txn 1 commits first (succeeds)
        mgr.commit(txn1).unwrap();

        // Txn 2 tries to commit — should detect write-write conflict
        let result = mgr.commit(txn2);
        assert!(result.is_err());
        assert_eq!(mgr.total_aborted(), 1);
    }

    #[test]
    fn test_concurrent_non_conflicting_commits() {
        let mgr = TxnManager::new_standalone();

        let txn1 = mgr.begin();
        let txn2 = mgr.begin();

        // Write to different rows
        mgr.with_context(txn1, |ctx| ctx.buffer_insert("users", 0, make_row("alice")))
            .unwrap();

        mgr.with_context(txn2, |ctx| ctx.buffer_insert("users", 1, make_row("bob")))
            .unwrap();

        mgr.commit(txn1).unwrap();
        mgr.commit(txn2).unwrap();
        assert_eq!(mgr.total_committed(), 2);
    }
}
