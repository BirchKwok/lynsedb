//! Transaction Context - Per-transaction state tracking
//!
//! Tracks all reads and writes within a transaction for conflict detection
//! and rollback support.

use std::collections::{HashMap, HashSet};
use std::io;

use crate::data::Value;

use super::manager::TxnId;

// ============================================================================
// Write Operation Types
// ============================================================================

/// A buffered write operation within a transaction
#[derive(Debug, Clone)]
pub enum TxnWrite {
    /// Insert a new row
    Insert {
        table: String,
        row_id: u64,
        data: HashMap<String, Value>,
    },
    /// Update an existing row
    Update {
        table: String,
        row_id: u64,
        old_data: HashMap<String, Value>,
        new_data: HashMap<String, Value>,
    },
    /// Delete a row
    Delete {
        table: String,
        row_id: u64,
        old_data: HashMap<String, Value>,
    },
}

impl TxnWrite {
    /// Get the table name
    pub fn table(&self) -> &str {
        match self {
            TxnWrite::Insert { table, .. } => table,
            TxnWrite::Update { table, .. } => table,
            TxnWrite::Delete { table, .. } => table,
        }
    }

    /// Get the row ID
    pub fn row_id(&self) -> u64 {
        match self {
            TxnWrite::Insert { row_id, .. } => *row_id,
            TxnWrite::Update { row_id, .. } => *row_id,
            TxnWrite::Delete { row_id, .. } => *row_id,
        }
    }
}

// ============================================================================
// Read Record
// ============================================================================

/// A record of a read operation (for OCC validation)
#[derive(Debug, Clone)]
pub struct ReadRecord {
    /// Table that was read
    pub table: String,
    /// Row ID that was read
    pub row_id: u64,
    /// Version timestamp that was read (for staleness detection)
    pub version_ts: u64,
}

// ============================================================================
// Transaction Context
// ============================================================================

/// Per-transaction context holding all read/write state
///
/// Used by the conflict detector to validate at commit time,
/// and as an undo log for rollback.
pub struct TxnContext {
    /// Transaction ID
    txn_id: TxnId,
    /// Snapshot timestamp (reads see data as of this time)
    snapshot_ts: u64,
    /// Read set: all rows read during this transaction
    read_set: Vec<ReadRecord>,
    /// Write set: all pending writes (applied on commit)
    write_set: Vec<TxnWrite>,
    /// Set of (table, row_id) pairs written (for fast conflict check)
    write_keys: HashSet<(String, u64)>,
    /// Pending insert count per table (for monotonic row-id reservation)
    pending_insert_counts: HashMap<String, u64>,
    /// First storage row ID observed for transactional inserts per table.
    insert_base_ids: HashMap<String, u64>,
    /// Tables touched (for table-level lock tracking)
    tables_touched: HashSet<String>,
    /// Whether the transaction is read-only
    read_only: bool,
    /// Whether the transaction has been committed or aborted
    finished: bool,
    /// Savepoints: (name, write_set_len_at_savepoint).
    ///
    /// write_keys are rebuilt only on rollback, keeping the common successful
    /// statement path O(1) even when a transaction already has many writes.
    savepoints: Vec<(String, usize)>,
}

impl TxnContext {
    /// Create a new transaction context
    pub fn new(txn_id: TxnId, snapshot_ts: u64, read_only: bool) -> Self {
        Self {
            txn_id,
            snapshot_ts,
            read_set: Vec::new(),
            write_set: Vec::new(),
            write_keys: HashSet::new(),
            pending_insert_counts: HashMap::new(),
            insert_base_ids: HashMap::new(),
            tables_touched: HashSet::new(),
            read_only,
            finished: false,
            savepoints: Vec::new(),
        }
    }

    /// Transaction ID
    pub fn txn_id(&self) -> TxnId {
        self.txn_id
    }

    /// Snapshot timestamp
    pub fn snapshot_ts(&self) -> u64 {
        self.snapshot_ts
    }

    /// Whether read-only
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Whether finished (committed or aborted)
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Mark as finished
    pub fn set_finished(&mut self) {
        self.finished = true;
    }

    // ========================================================================
    // Read tracking
    // ========================================================================

    /// Record a read operation
    pub fn record_read(&mut self, table: &str, row_id: u64, version_ts: u64) {
        self.read_set.push(ReadRecord {
            table: table.to_string(),
            row_id,
            version_ts,
        });
        self.tables_touched.insert(table.to_string());
    }

    /// Get the read set
    pub fn read_set(&self) -> &[ReadRecord] {
        &self.read_set
    }

    // ========================================================================
    // Write buffering
    // ========================================================================

    /// Buffer an insert operation
    pub fn buffer_insert(
        &mut self,
        table: &str,
        row_id: u64,
        data: HashMap<String, Value>,
    ) -> io::Result<()> {
        if self.read_only {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Cannot write in a read-only transaction",
            ));
        }
        let key = (table.to_string(), row_id);
        if self.write_keys.contains(&key) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "Row {} in table '{}' already written in this transaction",
                    row_id, table
                ),
            ));
        }
        self.write_keys.insert(key);
        *self
            .pending_insert_counts
            .entry(table.to_string())
            .or_insert(0) += 1;
        self.tables_touched.insert(table.to_string());
        self.write_set.push(TxnWrite::Insert {
            table: table.to_string(),
            row_id,
            data,
        });
        Ok(())
    }

    /// Buffer an update operation
    pub fn buffer_update(
        &mut self,
        table: &str,
        row_id: u64,
        old_data: HashMap<String, Value>,
        new_data: HashMap<String, Value>,
    ) -> io::Result<()> {
        if self.read_only {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Cannot write in a read-only transaction",
            ));
        }
        let key = (table.to_string(), row_id);
        self.write_keys.insert(key);
        self.tables_touched.insert(table.to_string());
        self.write_set.push(TxnWrite::Update {
            table: table.to_string(),
            row_id,
            old_data,
            new_data,
        });
        Ok(())
    }

    /// Buffer a delete operation
    pub fn buffer_delete(
        &mut self,
        table: &str,
        row_id: u64,
        old_data: HashMap<String, Value>,
    ) -> io::Result<()> {
        if self.read_only {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Cannot write in a read-only transaction",
            ));
        }
        let key = (table.to_string(), row_id);
        self.write_keys.insert(key);
        self.tables_touched.insert(table.to_string());
        self.write_set.push(TxnWrite::Delete {
            table: table.to_string(),
            row_id,
            old_data,
        });
        Ok(())
    }

    /// Get the write set
    pub fn write_set(&self) -> &[TxnWrite] {
        &self.write_set
    }

    /// Get the write keys (for conflict detection)
    pub fn write_keys(&self) -> &HashSet<(String, u64)> {
        &self.write_keys
    }

    /// Number of buffered inserts for a table.
    pub fn pending_insert_count(&self, table: &str) -> u64 {
        self.pending_insert_counts.get(table).copied().unwrap_or(0)
    }

    /// First storage row ID observed for inserts into this table in this transaction.
    pub fn insert_base_id(&self, table: &str) -> Option<u64> {
        self.insert_base_ids.get(table).copied()
    }

    /// Remember the initial storage row ID for a table's transactional inserts.
    pub fn remember_insert_base_id(&mut self, table: &str, base_id: u64) {
        self.insert_base_ids
            .entry(table.to_string())
            .or_insert(base_id);
    }

    /// Get tables touched
    pub fn tables_touched(&self) -> &HashSet<String> {
        &self.tables_touched
    }

    /// Number of pending writes
    pub fn write_count(&self) -> usize {
        self.write_set.len()
    }

    /// Whether there are any pending writes
    pub fn has_writes(&self) -> bool {
        !self.write_set.is_empty()
    }

    // ========================================================================
    // Undo (for rollback)
    // ========================================================================

    /// Get the undo operations (reverse of write set)
    /// Used during ROLLBACK to undo all buffered writes
    pub fn undo_operations(&self) -> Vec<TxnWrite> {
        self.write_set.iter().rev().cloned().collect()
    }

    /// Take buffered writes when a transaction commits.
    pub fn take_write_set(&mut self) -> Vec<TxnWrite> {
        std::mem::take(&mut self.write_set)
    }

    // ========================================================================
    // Savepoints
    // ========================================================================

    /// Create a savepoint at the current write-set position
    pub fn savepoint(&mut self, name: &str) {
        self.savepoints
            .push((name.to_string(), self.write_set.len()));
    }

    /// Rollback to a named savepoint — truncate write_set and restore write_keys
    pub fn rollback_to_savepoint(&mut self, name: &str) -> io::Result<()> {
        if let Some(pos) = self.savepoints.iter().rposition(|(n, _)| n == name) {
            let ws_len = self.savepoints[pos].1;
            self.write_set.truncate(ws_len);
            self.rebuild_write_indexes();
            // Remove this savepoint and all later ones
            self.savepoints.truncate(pos);
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Savepoint '{}' not found", name),
            ))
        }
    }

    /// Release a named savepoint (just removes it, keeps writes)
    pub fn release_savepoint(&mut self, name: &str) -> io::Result<()> {
        if let Some(pos) = self.savepoints.iter().rposition(|(n, _)| n == name) {
            self.savepoints.remove(pos);
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Savepoint '{}' not found", name),
            ))
        }
    }

    /// Clear all state (after commit or abort)
    pub fn clear(&mut self) {
        self.read_set.clear();
        self.write_set.clear();
        self.write_keys.clear();
        self.pending_insert_counts.clear();
        self.insert_base_ids.clear();
        self.tables_touched.clear();
        self.savepoints.clear();
        self.finished = true;
    }

    fn rebuild_write_indexes(&mut self) {
        self.write_keys.clear();
        self.pending_insert_counts.clear();
        for write in &self.write_set {
            let table = write.table().to_string();
            self.write_keys.insert((table.clone(), write.row_id()));
            if matches!(write, TxnWrite::Insert { .. }) {
                *self.pending_insert_counts.entry(table).or_insert(0) += 1;
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

    fn make_row(name: &str) -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), Value::String(name.to_string()));
        m
    }

    #[test]
    fn test_read_only_transaction() {
        let mut ctx = TxnContext::new(1, 100, true);
        ctx.record_read("users", 0, 50);
        assert_eq!(ctx.read_set().len(), 1);
        assert!(ctx.buffer_insert("users", 1, make_row("alice")).is_err());
    }

    #[test]
    fn test_write_buffering() {
        let mut ctx = TxnContext::new(1, 100, false);
        ctx.buffer_insert("users", 0, make_row("alice")).unwrap();
        ctx.buffer_update("users", 1, make_row("bob"), make_row("bob2"))
            .unwrap();
        ctx.buffer_delete("users", 2, make_row("carol")).unwrap();

        assert_eq!(ctx.write_count(), 3);
        assert!(ctx.has_writes());
        assert!(ctx.write_keys().contains(&("users".to_string(), 0)));
    }

    #[test]
    fn test_duplicate_insert_blocked() {
        let mut ctx = TxnContext::new(1, 100, false);
        ctx.buffer_insert("users", 0, make_row("alice")).unwrap();
        assert!(ctx.buffer_insert("users", 0, make_row("alice2")).is_err());
    }

    #[test]
    fn test_undo_operations() {
        let mut ctx = TxnContext::new(1, 100, false);
        ctx.buffer_insert("users", 0, make_row("alice")).unwrap();
        ctx.buffer_insert("users", 1, make_row("bob")).unwrap();

        let undos = ctx.undo_operations();
        assert_eq!(undos.len(), 2);
        // Undo is in reverse order
        assert_eq!(undos[0].row_id(), 1);
        assert_eq!(undos[1].row_id(), 0);
    }

    #[test]
    fn test_savepoint_rebuilds_write_indexes_on_rollback() {
        let mut ctx = TxnContext::new(1, 100, false);
        ctx.buffer_insert("users", 0, make_row("alice")).unwrap();
        ctx.savepoint("sp1");
        ctx.buffer_insert("users", 1, make_row("bob")).unwrap();

        ctx.rollback_to_savepoint("sp1").unwrap();

        assert_eq!(ctx.write_count(), 1);
        assert_eq!(ctx.pending_insert_count("users"), 1);
        assert!(ctx.write_keys().contains(&("users".to_string(), 0)));
        assert!(!ctx.write_keys().contains(&("users".to_string(), 1)));
    }
}
