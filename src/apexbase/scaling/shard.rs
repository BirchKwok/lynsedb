//! Shard Manager - Manages shard lifecycle and metadata
//!
//! Each shard is an independent storage unit containing a subset of table data.
//! The shard manager handles creating, splitting, merging, and migrating shards.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::node::NodeId;
use super::partition::{PartitionKey, PartitionStrategy};

// ============================================================================
// Shard ID
// ============================================================================

/// Unique shard identifier
pub type ShardId = u32;

/// Global shard ID generator
static NEXT_SHARD_ID: AtomicU32 = AtomicU32::new(1);

fn next_shard_id() -> ShardId {
    NEXT_SHARD_ID.fetch_add(1, Ordering::SeqCst)
}

// ============================================================================
// Shard Status
// ============================================================================

/// Current status of a shard
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardStatus {
    /// Shard is active and serving reads/writes
    Active,
    /// Shard is being created (data loading)
    Creating,
    /// Shard is being split into two smaller shards
    Splitting,
    /// Shard is being merged with another shard
    Merging,
    /// Shard is being migrated to another node
    Migrating,
    /// Shard is read-only (e.g., during migration source)
    ReadOnly,
    /// Shard is offline/unavailable
    Offline,
}

// ============================================================================
// Shard Metadata
// ============================================================================

/// Metadata for a single shard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMeta {
    /// Shard ID
    pub id: ShardId,
    /// Table name this shard belongs to
    pub table_name: String,
    /// Node hosting this shard
    pub node_id: NodeId,
    /// Current status
    pub status: ShardStatus,
    /// Shard data directory/path
    pub data_path: PathBuf,
    /// Approximate row count
    pub row_count: u64,
    /// Approximate data size in bytes
    pub data_size: u64,
    /// Partition key range: (min_key, max_key) - None for hash partitioning
    pub key_range: Option<(PartitionKey, PartitionKey)>,
    /// Creation timestamp
    pub created_at: i64,
    /// Last modified timestamp
    pub modified_at: i64,
    /// Replica shard IDs (for fault tolerance)
    pub replicas: Vec<ShardId>,
}

impl ShardMeta {
    /// Create metadata for a new shard
    pub fn new(table_name: &str, node_id: NodeId, data_path: PathBuf) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: next_shard_id(),
            table_name: table_name.to_string(),
            node_id,
            status: ShardStatus::Creating,
            data_path,
            row_count: 0,
            data_size: 0,
            key_range: None,
            created_at: now,
            modified_at: now,
            replicas: Vec::new(),
        }
    }

    /// Whether the shard is available for reads
    pub fn is_readable(&self) -> bool {
        matches!(self.status, ShardStatus::Active | ShardStatus::ReadOnly)
    }

    /// Whether the shard is available for writes
    pub fn is_writable(&self) -> bool {
        matches!(self.status, ShardStatus::Active)
    }
}

// ============================================================================
// Shard Manager
// ============================================================================

/// Manages all shards for a database instance
///
/// Responsibilities:
/// - Track shard metadata and topology
/// - Create/split/merge shards
/// - Coordinate shard migrations between nodes
/// - Persist shard catalog
pub struct ShardManager {
    /// All shard metadata: shard_id → ShardMeta
    shards: RwLock<HashMap<ShardId, ShardMeta>>,
    /// Table → shard IDs mapping
    table_shards: RwLock<HashMap<String, Vec<ShardId>>>,
    /// Base directory for shard data
    base_dir: PathBuf,
    /// Local node ID
    local_node_id: NodeId,
    /// Whether catalog has been modified
    dirty: RwLock<bool>,
}

impl ShardManager {
    /// Create a new shard manager
    pub fn new(base_dir: &Path, local_node_id: NodeId) -> Self {
        Self {
            shards: RwLock::new(HashMap::new()),
            table_shards: RwLock::new(HashMap::new()),
            base_dir: base_dir.to_path_buf(),
            local_node_id,
            dirty: RwLock::new(false),
        }
    }

    /// Load shard catalog from disk
    pub fn load(base_dir: &Path, local_node_id: NodeId) -> io::Result<Self> {
        let catalog_path = base_dir.join("shard_catalog.bin");
        if !catalog_path.exists() {
            return Ok(Self::new(base_dir, local_node_id));
        }

        let data = std::fs::read(&catalog_path)?;
        let catalog: ShardCatalog = bincode::deserialize(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        let mut table_shards: HashMap<String, Vec<ShardId>> = HashMap::new();
        for (id, meta) in &catalog.shards {
            table_shards
                .entry(meta.table_name.clone())
                .or_insert_with(Vec::new)
                .push(*id);
        }

        Ok(Self {
            shards: RwLock::new(catalog.shards),
            table_shards: RwLock::new(table_shards),
            base_dir: base_dir.to_path_buf(),
            local_node_id,
            dirty: RwLock::new(false),
        })
    }

    // ========================================================================
    // Shard Lifecycle
    // ========================================================================

    /// Create a new shard for a table
    pub fn create_shard(&self, table_name: &str) -> io::Result<ShardMeta> {
        let shard_dir = self.base_dir.join("shards");
        std::fs::create_dir_all(&shard_dir)?;

        let mut meta = ShardMeta::new(
            table_name,
            self.local_node_id,
            shard_dir.join(format!("shard_{}", next_shard_id())),
        );
        meta.status = ShardStatus::Active;

        let shard_id = meta.id;
        self.shards.write().insert(shard_id, meta.clone());
        self.table_shards
            .write()
            .entry(table_name.to_string())
            .or_insert_with(Vec::new)
            .push(shard_id);
        *self.dirty.write() = true;

        Ok(meta)
    }

    /// Get shard metadata
    pub fn get_shard(&self, shard_id: ShardId) -> Option<ShardMeta> {
        self.shards.read().get(&shard_id).cloned()
    }

    /// Get all shards for a table
    pub fn get_table_shards(&self, table_name: &str) -> Vec<ShardMeta> {
        let table_shards = self.table_shards.read();
        let shards = self.shards.read();
        table_shards
            .get(table_name)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| shards.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get writable shards for a table
    pub fn get_writable_shards(&self, table_name: &str) -> Vec<ShardMeta> {
        self.get_table_shards(table_name)
            .into_iter()
            .filter(|s| s.is_writable())
            .collect()
    }

    /// Update shard status
    pub fn set_shard_status(&self, shard_id: ShardId, status: ShardStatus) -> io::Result<()> {
        let mut shards = self.shards.write();
        let shard = shards.get_mut(&shard_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Shard {} not found", shard_id),
            )
        })?;
        shard.status = status;
        shard.modified_at = chrono::Utc::now().timestamp();
        *self.dirty.write() = true;
        Ok(())
    }

    /// Update shard statistics
    pub fn update_shard_stats(&self, shard_id: ShardId, row_count: u64, data_size: u64) {
        let mut shards = self.shards.write();
        if let Some(shard) = shards.get_mut(&shard_id) {
            shard.row_count = row_count;
            shard.data_size = data_size;
            shard.modified_at = chrono::Utc::now().timestamp();
        }
    }

    /// Remove a shard
    pub fn remove_shard(&self, shard_id: ShardId) -> io::Result<ShardMeta> {
        let meta = self.shards.write().remove(&shard_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Shard {} not found", shard_id),
            )
        })?;

        if let Some(ids) = self.table_shards.write().get_mut(&meta.table_name) {
            ids.retain(|&id| id != shard_id);
        }
        *self.dirty.write() = true;
        Ok(meta)
    }

    // ========================================================================
    // Split / Merge (framework stubs for future implementation)
    // ========================================================================

    /// Plan a shard split (returns the split point)
    ///
    /// Splitting is needed when a shard grows too large.
    /// The split point is chosen to balance data evenly.
    pub fn plan_split(&self, shard_id: ShardId) -> io::Result<SplitPlan> {
        let shards = self.shards.read();
        let shard = shards.get(&shard_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Shard {} not found", shard_id),
            )
        })?;

        Ok(SplitPlan {
            source_shard: shard_id,
            table_name: shard.table_name.clone(),
            estimated_rows_per_half: shard.row_count / 2,
        })
    }

    /// Plan a shard merge
    pub fn plan_merge(&self, shard_a: ShardId, shard_b: ShardId) -> io::Result<MergePlan> {
        let shards = self.shards.read();
        let a = shards.get(&shard_a).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Shard {} not found", shard_a),
            )
        })?;
        let b = shards.get(&shard_b).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Shard {} not found", shard_b),
            )
        })?;

        if a.table_name != b.table_name {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot merge shards from different tables",
            ));
        }

        Ok(MergePlan {
            source_shards: (shard_a, shard_b),
            table_name: a.table_name.clone(),
            estimated_total_rows: a.row_count + b.row_count,
        })
    }

    // ========================================================================
    // Rebalancing
    // ========================================================================

    /// Check if any shards need rebalancing
    pub fn needs_rebalance(&self, table_name: &str) -> bool {
        let shards = self.get_table_shards(table_name);
        if shards.len() < 2 {
            return false;
        }

        let avg_rows = shards.iter().map(|s| s.row_count).sum::<u64>() / shards.len() as u64;
        // Rebalance if any shard has >2x the average
        shards
            .iter()
            .any(|s| s.row_count > avg_rows * 2 || (avg_rows > 100 && s.row_count < avg_rows / 3))
    }

    /// Get shard count for a table
    pub fn shard_count(&self, table_name: &str) -> usize {
        self.table_shards
            .read()
            .get(table_name)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Total shard count
    pub fn total_shards(&self) -> usize {
        self.shards.read().len()
    }

    // ========================================================================
    // Persistence
    // ========================================================================

    /// Save shard catalog to disk
    pub fn save(&self) -> io::Result<()> {
        if !*self.dirty.read() {
            return Ok(());
        }

        std::fs::create_dir_all(&self.base_dir)?;
        let catalog_path = self.base_dir.join("shard_catalog.bin");
        let catalog = ShardCatalog {
            shards: self.shards.read().clone(),
        };
        let data = bincode::serialize(&catalog)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        std::fs::write(&catalog_path, &data)?;
        *self.dirty.write() = false;
        Ok(())
    }
}

/// Serializable shard catalog
#[derive(Serialize, Deserialize)]
struct ShardCatalog {
    shards: HashMap<ShardId, ShardMeta>,
}

/// Plan for splitting a shard
#[derive(Debug)]
pub struct SplitPlan {
    pub source_shard: ShardId,
    pub table_name: String,
    pub estimated_rows_per_half: u64,
}

/// Plan for merging two shards
#[derive(Debug)]
pub struct MergePlan {
    pub source_shards: (ShardId, ShardId),
    pub table_name: String,
    pub estimated_total_rows: u64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_shard() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ShardManager::new(dir.path(), 1);
        let shard = mgr.create_shard("users").unwrap();

        assert_eq!(shard.table_name, "users");
        assert_eq!(shard.status, ShardStatus::Active);
        assert_eq!(mgr.shard_count("users"), 1);
    }

    #[test]
    fn test_get_table_shards() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ShardManager::new(dir.path(), 1);
        mgr.create_shard("users").unwrap();
        mgr.create_shard("users").unwrap();
        mgr.create_shard("orders").unwrap();

        assert_eq!(mgr.get_table_shards("users").len(), 2);
        assert_eq!(mgr.get_table_shards("orders").len(), 1);
    }

    #[test]
    fn test_shard_status_update() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ShardManager::new(dir.path(), 1);
        let shard = mgr.create_shard("users").unwrap();

        mgr.set_shard_status(shard.id, ShardStatus::ReadOnly)
            .unwrap();
        let updated = mgr.get_shard(shard.id).unwrap();
        assert_eq!(updated.status, ShardStatus::ReadOnly);
        assert!(updated.is_readable());
        assert!(!updated.is_writable());
    }

    #[test]
    fn test_remove_shard() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = ShardManager::new(dir.path(), 1);
        let shard = mgr.create_shard("users").unwrap();

        mgr.remove_shard(shard.id).unwrap();
        assert_eq!(mgr.shard_count("users"), 0);
    }

    #[test]
    fn test_shard_persistence() {
        let dir = tempfile::tempdir().unwrap();

        {
            let mgr = ShardManager::new(dir.path(), 1);
            mgr.create_shard("users").unwrap();
            mgr.save().unwrap();
        }

        let mgr = ShardManager::load(dir.path(), 1).unwrap();
        assert_eq!(mgr.shard_count("users"), 1);
    }
}
