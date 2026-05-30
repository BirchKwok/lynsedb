//! Partition Strategy - Determines how data is distributed across shards
//!
//! Provides multiple partitioning strategies:
//! - Hash partitioning: uniform distribution via consistent hashing
//! - Range partitioning: ordered ranges for range-scan friendly layouts
//! - Custom partitioning: user-defined routing logic

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use ahash::AHasher;
use serde::{Deserialize, Serialize};

use super::shard::ShardId;
use crate::data::Value;

// ============================================================================
// Partition Key
// ============================================================================

/// A partition key extracted from a row for shard routing
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PartitionKey {
    Int(i64),
    UInt(u64),
    Str(String),
    Bytes(Vec<u8>),
    Composite(Vec<PartitionKey>),
}

impl PartitionKey {
    /// Extract partition key from a Value
    pub fn from_value(val: &Value) -> Self {
        match val {
            Value::Int8(v) => PartitionKey::Int(*v as i64),
            Value::Int16(v) => PartitionKey::Int(*v as i64),
            Value::Int32(v) => PartitionKey::Int(*v as i64),
            Value::Int64(v) => PartitionKey::Int(*v),
            Value::UInt8(v) => PartitionKey::UInt(*v as u64),
            Value::UInt16(v) => PartitionKey::UInt(*v as u64),
            Value::UInt32(v) => PartitionKey::UInt(*v as u64),
            Value::UInt64(v) => PartitionKey::UInt(*v),
            Value::String(s) => PartitionKey::Str(s.clone()),
            Value::Binary(b) => PartitionKey::Bytes(b.clone()),
            _ => PartitionKey::Str(val.to_string_value()),
        }
    }

    /// Compute hash of the key (for hash partitioning)
    pub fn hash_value(&self) -> u64 {
        let mut hasher = AHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

// ============================================================================
// Partition Strategy Trait
// ============================================================================

/// Trait for partition strategy implementations
pub trait PartitionStrategy: Send + Sync {
    /// Given a partition key and total shard count, return the target shard
    fn route(&self, key: &PartitionKey, shard_count: u32) -> ShardId;

    /// Name of this strategy (for serialization)
    fn name(&self) -> &str;

    /// Get all shards that might contain data for a range query
    /// Returns None if all shards must be scanned
    fn route_range(
        &self,
        _low: &PartitionKey,
        _high: &PartitionKey,
        shard_count: u32,
    ) -> Option<Vec<ShardId>> {
        None // Default: must scan all shards
    }
}

// ============================================================================
// Hash Partitioner
// ============================================================================

/// Hash-based partitioning for uniform distribution
///
/// Uses consistent hashing to minimize data movement during resharding.
/// Each key is hashed and mapped to a shard via modulo.
///
/// Properties:
/// - Uniform distribution (good for most workloads)
/// - No ordering guarantee
/// - Range queries must scatter to all shards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashPartitioner {
    /// Number of virtual nodes per physical shard (for consistent hashing)
    virtual_nodes: u32,
    /// Hash ring: virtual_node_hash → shard_id
    #[serde(skip)]
    ring: Option<BTreeMap<u64, ShardId>>,
}

impl HashPartitioner {
    /// Create a new hash partitioner
    pub fn new() -> Self {
        Self {
            virtual_nodes: 150,
            ring: None,
        }
    }

    /// Create with custom virtual node count
    pub fn with_virtual_nodes(virtual_nodes: u32) -> Self {
        Self {
            virtual_nodes,
            ring: None,
        }
    }

    /// Build the consistent hash ring for the given shard count
    fn build_ring(&mut self, shard_count: u32) {
        let mut ring = BTreeMap::new();
        for shard_id in 0..shard_count {
            for vn in 0..self.virtual_nodes {
                let mut hasher = AHasher::default();
                (shard_id, vn).hash(&mut hasher);
                let hash = hasher.finish();
                ring.insert(hash, shard_id);
            }
        }
        self.ring = Some(ring);
    }

    /// Lookup on the consistent hash ring
    fn ring_lookup(&self, hash: u64) -> ShardId {
        if let Some(ring) = &self.ring {
            // Find the first node on the ring with hash >= key_hash
            ring.range(hash..)
                .next()
                .or_else(|| ring.iter().next()) // Wrap around
                .map(|(_, &shard_id)| shard_id)
                .unwrap_or(0)
        } else {
            0
        }
    }
}

impl Default for HashPartitioner {
    fn default() -> Self {
        Self::new()
    }
}

impl PartitionStrategy for HashPartitioner {
    fn route(&self, key: &PartitionKey, shard_count: u32) -> ShardId {
        if shard_count <= 1 {
            return 0;
        }
        if self.ring.is_some() {
            self.ring_lookup(key.hash_value())
        } else {
            // Fallback: simple modulo
            (key.hash_value() % shard_count as u64) as u32
        }
    }

    fn name(&self) -> &str {
        "hash"
    }
}

// ============================================================================
// Range Partitioner
// ============================================================================

/// Range-based partitioning for ordered data
///
/// Data is split into contiguous ranges, each assigned to a shard.
/// Excellent for range queries but can lead to hotspots.
///
/// Properties:
/// - Range queries can be routed to subset of shards
/// - Ordered access patterns are efficient
/// - May need rebalancing if data is skewed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangePartitioner {
    /// Boundary keys between shards (sorted)
    /// If there are N shards, there are N-1 boundaries.
    /// Shard 0: [MIN, boundaries[0])
    /// Shard 1: [boundaries[0], boundaries[1])
    /// ...
    /// Shard N-1: [boundaries[N-2], MAX)
    boundaries: Vec<PartitionKey>,
}

impl RangePartitioner {
    /// Create with explicit boundaries
    pub fn new(boundaries: Vec<PartitionKey>) -> Self {
        Self { boundaries }
    }

    /// Create with evenly spaced integer boundaries
    pub fn for_int_range(min: i64, max: i64, shard_count: u32) -> Self {
        if shard_count <= 1 {
            return Self {
                boundaries: Vec::new(),
            };
        }
        let range = max - min;
        let step = range / shard_count as i64;
        let boundaries: Vec<PartitionKey> = (1..shard_count)
            .map(|i| PartitionKey::Int(min + step * i as i64))
            .collect();
        Self { boundaries }
    }

    /// Get boundaries
    pub fn boundaries(&self) -> &[PartitionKey] {
        &self.boundaries
    }

    /// Update boundaries (for rebalancing)
    pub fn set_boundaries(&mut self, boundaries: Vec<PartitionKey>) {
        self.boundaries = boundaries;
    }
}

impl PartitionStrategy for RangePartitioner {
    fn route(&self, key: &PartitionKey, _shard_count: u32) -> ShardId {
        // Find the first boundary > key
        for (i, boundary) in self.boundaries.iter().enumerate() {
            if key_less_than(key, boundary) {
                return i as ShardId;
            }
        }
        self.boundaries.len() as ShardId // Last shard
    }

    fn name(&self) -> &str {
        "range"
    }

    fn route_range(
        &self,
        low: &PartitionKey,
        high: &PartitionKey,
        shard_count: u32,
    ) -> Option<Vec<ShardId>> {
        let start_shard = self.route(low, shard_count);
        let end_shard = self.route(high, shard_count);
        Some((start_shard..=end_shard).collect())
    }
}

/// Compare two partition keys
fn key_less_than(a: &PartitionKey, b: &PartitionKey) -> bool {
    match (a, b) {
        (PartitionKey::Int(a), PartitionKey::Int(b)) => a < b,
        (PartitionKey::UInt(a), PartitionKey::UInt(b)) => a < b,
        (PartitionKey::Str(a), PartitionKey::Str(b)) => a < b,
        (PartitionKey::Bytes(a), PartitionKey::Bytes(b)) => a < b,
        _ => false,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_partitioner_uniform() {
        let p = HashPartitioner::new();
        let shard_count = 4;
        let mut counts = vec![0u32; shard_count as usize];
        for i in 0..10000 {
            let key = PartitionKey::Int(i);
            let shard = p.route(&key, shard_count);
            counts[shard as usize] += 1;
        }
        // Each shard should have roughly 2500 ± tolerance
        for &c in &counts {
            assert!(
                c > 1500 && c < 3500,
                "Unbalanced distribution: {:?}",
                counts
            );
        }
    }

    #[test]
    fn test_hash_partitioner_single_shard() {
        let p = HashPartitioner::new();
        assert_eq!(p.route(&PartitionKey::Int(42), 1), 0);
    }

    #[test]
    fn test_range_partitioner() {
        let p = RangePartitioner::for_int_range(0, 100, 4);
        // Boundaries at 25, 50, 75
        assert_eq!(p.route(&PartitionKey::Int(10), 4), 0);
        assert_eq!(p.route(&PartitionKey::Int(30), 4), 1);
        assert_eq!(p.route(&PartitionKey::Int(60), 4), 2);
        assert_eq!(p.route(&PartitionKey::Int(90), 4), 3);
    }

    #[test]
    fn test_range_partitioner_route_range() {
        let p = RangePartitioner::for_int_range(0, 100, 4);
        let shards = p.route_range(&PartitionKey::Int(10), &PartitionKey::Int(60), 4);
        assert!(shards.is_some());
        let shards = shards.unwrap();
        assert!(shards.contains(&0));
        assert!(shards.contains(&1));
        assert!(shards.contains(&2));
    }

    #[test]
    fn test_partition_key_from_value() {
        assert_eq!(
            PartitionKey::from_value(&Value::Int64(42)),
            PartitionKey::Int(42)
        );
        assert_eq!(
            PartitionKey::from_value(&Value::String("hello".into())),
            PartitionKey::Str("hello".into())
        );
    }
}
