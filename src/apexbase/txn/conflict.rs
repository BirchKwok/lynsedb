//! Conflict Detector - OCC (Optimistic Concurrency Control) validation
//!
//! At commit time, validates that a transaction's read set hasn't been
//! modified by other committed transactions since the snapshot was taken.
//! Uses first-committer-wins strategy for write-write conflicts.

use std::collections::{HashMap, HashSet};
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use super::context::{TxnContext, TxnWrite};
use super::manager::TxnId;

// ============================================================================
// Conflict Result
// ============================================================================

/// Result of conflict validation
#[derive(Debug, Clone)]
pub enum ConflictResult {
    /// No conflicts detected - safe to commit
    NoConflict,
    /// Read-write conflict: another txn modified a row we read
    ReadWriteConflict {
        table: String,
        row_id: u64,
        conflicting_txn: TxnId,
    },
    /// Write-write conflict: another txn wrote to the same row
    WriteWriteConflict {
        table: String,
        row_id: u64,
        conflicting_txn: TxnId,
    },
}

impl ConflictResult {
    /// Whether the result indicates no conflict
    pub fn is_ok(&self) -> bool {
        matches!(self, ConflictResult::NoConflict)
    }

    /// Convert to io::Result (for error propagation)
    pub fn to_io_result(&self) -> io::Result<()> {
        match self {
            ConflictResult::NoConflict => Ok(()),
            ConflictResult::ReadWriteConflict {
                table,
                row_id,
                conflicting_txn,
            } => Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!(
                    "Read-write conflict: row {} in table '{}' was modified by txn {}",
                    row_id, table, conflicting_txn
                ),
            )),
            ConflictResult::WriteWriteConflict {
                table,
                row_id,
                conflicting_txn,
            } => Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!(
                    "Write-write conflict: row {} in table '{}' was also written by txn {}",
                    row_id, table, conflicting_txn
                ),
            )),
        }
    }
}

// ============================================================================
// Committed Write Record
// ============================================================================

/// Record of a committed write (kept until safe to discard)
#[derive(Debug, Clone)]
struct CommittedWrite {
    /// Transaction that committed this write
    txn_id: TxnId,
    /// Commit timestamp
    commit_ts: u64,
    /// Table name
    table: String,
    /// Row ID
    row_id: u64,
}

// ============================================================================
// Conflict Detector
// ============================================================================

/// OCC-based conflict detector
///
/// Maintains a log of recently committed writes. At validation time,
/// checks if any row in the transaction's read/write set was modified
/// by another transaction after our snapshot timestamp.
///
/// Strategy: **First-Committer-Wins**
/// - If two transactions write the same row, the first to commit succeeds
/// - The second transaction must abort and retry
pub struct ConflictDetector {
    /// Recently committed writes: (table, row_id) → CommittedWrite
    committed_writes: RwLock<Vec<CommittedWrite>>,
    /// Active write locks: (table, row_id) → txn_id
    /// Used to detect write-write conflicts early (optional optimization)
    active_writes: RwLock<HashMap<(String, u64), TxnId>>,
    /// Watermark: oldest snapshot ts still active (for cleanup)
    watermark: AtomicU64,
}

impl ConflictDetector {
    /// Create a new conflict detector
    pub fn new() -> Self {
        Self {
            committed_writes: RwLock::new(Vec::new()),
            active_writes: RwLock::new(HashMap::new()),
            watermark: AtomicU64::new(0),
        }
    }

    /// Validate a transaction at commit time (OCC validation phase)
    ///
    /// Checks:
    /// 1. Read set validation: did any row we read get modified after our snapshot?
    /// 2. Write set validation: did any row we want to write get written by another txn?
    pub fn validate(&self, ctx: &TxnContext) -> ConflictResult {
        let committed = self.committed_writes.read();
        let snapshot_ts = ctx.snapshot_ts();

        // Check read set: any row we read was modified after our snapshot?
        for read in ctx.read_set() {
            for cw in committed.iter() {
                if cw.table == read.table
                    && cw.row_id == read.row_id
                    && cw.commit_ts > snapshot_ts
                    && cw.txn_id != ctx.txn_id()
                {
                    return ConflictResult::ReadWriteConflict {
                        table: read.table.clone(),
                        row_id: read.row_id,
                        conflicting_txn: cw.txn_id,
                    };
                }
            }
        }

        // Check write set: any row we want to write was also written after our snapshot?
        for write_key in ctx.write_keys() {
            for cw in committed.iter() {
                if cw.table == write_key.0
                    && cw.row_id == write_key.1
                    && cw.commit_ts > snapshot_ts
                    && cw.txn_id != ctx.txn_id()
                {
                    return ConflictResult::WriteWriteConflict {
                        table: write_key.0.clone(),
                        row_id: write_key.1,
                        conflicting_txn: cw.txn_id,
                    };
                }
            }
        }

        ConflictResult::NoConflict
    }

    /// Record that a transaction has committed its writes
    pub fn record_commit(&self, ctx: &TxnContext, commit_ts: u64) {
        let mut committed = self.committed_writes.write();
        for write in ctx.write_set() {
            committed.push(CommittedWrite {
                txn_id: ctx.txn_id(),
                commit_ts,
                table: write.table().to_string(),
                row_id: write.row_id(),
            });
        }

        // Release active write locks
        let mut active = self.active_writes.write();
        for key in ctx.write_keys() {
            if active.get(key) == Some(&ctx.txn_id()) {
                active.remove(key);
            }
        }
    }

    /// Record that a transaction has aborted
    pub fn record_abort(&self, ctx: &TxnContext) {
        // Release active write locks
        let mut active = self.active_writes.write();
        for key in ctx.write_keys() {
            if active.get(key) == Some(&ctx.txn_id()) {
                active.remove(key);
            }
        }
    }

    /// Try to acquire write intent locks (optional early conflict detection)
    /// Returns Err if another transaction already has a write lock on the same row
    pub fn acquire_write_intent(&self, ctx: &TxnContext) -> ConflictResult {
        let active = self.active_writes.read();
        for key in ctx.write_keys() {
            if let Some(&other_txn) = active.get(key) {
                if other_txn != ctx.txn_id() {
                    return ConflictResult::WriteWriteConflict {
                        table: key.0.clone(),
                        row_id: key.1,
                        conflicting_txn: other_txn,
                    };
                }
            }
        }
        drop(active);

        // Acquire locks
        let mut active = self.active_writes.write();
        for key in ctx.write_keys() {
            active.insert(key.clone(), ctx.txn_id());
        }
        ConflictResult::NoConflict
    }

    /// Update watermark and clean up old committed writes
    pub fn advance_watermark(&self, new_watermark: u64) {
        self.watermark.store(new_watermark, Ordering::SeqCst);
        let mut committed = self.committed_writes.write();
        committed.retain(|cw| cw.commit_ts >= new_watermark);
    }

    /// Number of tracked committed writes (for monitoring)
    pub fn committed_write_count(&self) -> usize {
        self.committed_writes.read().len()
    }
}

impl Default for ConflictDetector {
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
    use crate::data::Value;

    fn make_row(name: &str) -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), Value::String(name.to_string()));
        m
    }

    #[test]
    fn test_no_conflict() {
        let detector = ConflictDetector::new();

        let mut ctx1 = TxnContext::new(1, 100, false);
        ctx1.buffer_insert("users", 0, make_row("alice")).unwrap();

        assert!(detector.validate(&ctx1).is_ok());
    }

    #[test]
    fn test_write_write_conflict() {
        let detector = ConflictDetector::new();

        // Txn 1 commits a write to row 0
        let mut ctx1 = TxnContext::new(1, 100, false);
        ctx1.buffer_insert("users", 0, make_row("alice")).unwrap();
        detector.record_commit(&ctx1, 110);

        // Txn 2 (snapshot at 100) tries to write to the same row
        let mut ctx2 = TxnContext::new(2, 100, false);
        ctx2.buffer_update("users", 0, make_row("alice"), make_row("alice2"))
            .unwrap();

        let result = detector.validate(&ctx2);
        assert!(!result.is_ok());
        match result {
            ConflictResult::WriteWriteConflict { row_id, .. } => assert_eq!(row_id, 0),
            _ => panic!("Expected WriteWriteConflict"),
        }
    }

    #[test]
    fn test_read_write_conflict() {
        let detector = ConflictDetector::new();

        // Txn 1 commits a write to row 5
        let mut ctx1 = TxnContext::new(1, 100, false);
        ctx1.buffer_update("users", 5, make_row("old"), make_row("new"))
            .unwrap();
        detector.record_commit(&ctx1, 110);

        // Txn 2 (snapshot at 100) read row 5 and now tries to commit
        let mut ctx2 = TxnContext::new(2, 100, false);
        ctx2.record_read("users", 5, 50);
        ctx2.buffer_insert("users", 99, make_row("unrelated"))
            .unwrap();

        let result = detector.validate(&ctx2);
        assert!(!result.is_ok());
    }

    #[test]
    fn test_no_conflict_different_rows() {
        let detector = ConflictDetector::new();

        let mut ctx1 = TxnContext::new(1, 100, false);
        ctx1.buffer_insert("users", 0, make_row("alice")).unwrap();
        detector.record_commit(&ctx1, 110);

        // Txn 2 writes to a different row — no conflict
        let mut ctx2 = TxnContext::new(2, 100, false);
        ctx2.buffer_insert("users", 1, make_row("bob")).unwrap();

        assert!(detector.validate(&ctx2).is_ok());
    }

    #[test]
    fn test_watermark_cleanup() {
        let detector = ConflictDetector::new();

        let mut ctx1 = TxnContext::new(1, 100, false);
        ctx1.buffer_insert("users", 0, make_row("alice")).unwrap();
        detector.record_commit(&ctx1, 110);

        assert_eq!(detector.committed_write_count(), 1);

        // Advance watermark past the commit
        detector.advance_watermark(120);
        assert_eq!(detector.committed_write_count(), 0);
    }
}
