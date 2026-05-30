//! Storage module - On-Demand Columnar Storage (V4)
//!
//! This module provides the core columnar storage format for ApexBase.
//! The V4 format supports on-demand column/row reading without loading
//! the entire dataset into memory.
//!
//! Incremental writes use WAL (Write-Ahead Log) for fast append-only writes.

pub mod backend;
pub mod bloom;
pub mod concurrent;
pub mod delta;
pub mod engine;
pub mod incremental;
pub mod index;
pub mod mvcc;
pub mod on_demand;

/// First user-visible row ID. ApexBase uses 1-based `_id` values.
pub const FIRST_ROW_ID: u64 = 1;

// ============================================================================
// Durability Level - Controls fsync behavior for ACID guarantees
// ============================================================================

/// Durability level for write operations
///
/// Controls how aggressively data is synced to disk:
/// - `Fast`: No fsync - fastest but data may be lost on crash
/// - `Safe`: fsync on flush() - balanced performance and durability  
/// - `Max`: fsync on every write - strongest ACID guarantee but slower
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DurabilityLevel {
    /// Highest performance, no fsync. Data written to OS buffer only.
    /// Suitable for batch import, reconstructible data, performance-critical scenarios.
    /// Risk: Data loss possible on system crash before OS flushes buffers.
    #[default]
    Fast,

    /// Balanced mode. fsync called on explicit flush() calls.
    /// Suitable for most production environments.
    /// Risk: Data loss possible only for writes between last flush and crash.
    Safe,

    /// Strongest ACID guarantee. fsync on every write operation.
    /// Suitable for financial, orders, and critical data scenarios.
    /// Performance: ~10-50x slower than Fast mode due to disk latency.
    Max,
}

impl DurabilityLevel {
    /// Parse from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "fast" => Some(DurabilityLevel::Fast),
            "safe" => Some(DurabilityLevel::Safe),
            "max" => Some(DurabilityLevel::Max),
            _ => None,
        }
    }

    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            DurabilityLevel::Fast => "fast",
            DurabilityLevel::Safe => "safe",
            DurabilityLevel::Max => "max",
        }
    }
}

// Re-export all public types from on_demand
pub use on_demand::{
    ColumnData,
    ColumnDef,
    ColumnIndexEntry,
    // Data types
    ColumnType,
    ColumnValue,
    // Compression
    CompressionType,
    FileSchema,
    OnDemandHeader,
    OnDemandSchema,
    // Storage engine
    OnDemandStorage,
};

// Re-export backend types
pub use backend::{
    column_data_to_typed_column, column_type_to_datatype, datatype_to_column_type,
    typed_column_to_column_data, IncrementalStorageBackend, StorageManager, TableMetadata,
    TableStorageBackend,
};

// Re-export incremental storage types
pub use incremental::{ConcurrentWalWriter, IncrementalStorage, WalReader, WalRecord, WalWriter};

// Re-export bloom filter types
pub use bloom::{ColumnBloomIndex, RowGroupBloomFilter, BLOOM_FP_RATE, BLOOM_ROW_GROUP_SIZE};

// Re-export storage engine
pub use engine::{engine, StorageEngine};

// Re-export index types
pub use index::{BTreeIndex, HashIndex, IndexManager, IndexMeta, IndexType};

// Re-export delta store types
pub use delta::{DeleteBitmap, DeltaMerger, DeltaRecord, DeltaStore};

// Re-export MVCC types
pub use mvcc::{
    GarbageCollector, RowVersion, Snapshot, SnapshotManager, VersionChain, VersionStore,
};

// Re-export concurrent access types
pub use concurrent::{
    global_stats, BackendStats, ReadGuard, StorageSnapshot, StorageStats, WriteGuard,
};

// Type alias for backward compatibility
pub type ColumnarStorage = OnDemandStorage;
