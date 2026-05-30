//! Shard Router - Routes queries and writes to the correct shard(s)
//!
//! Handles:
//! - Single-shard routing for point queries
//! - Scatter-gather for cross-shard queries
//! - Write routing based on partition key

use std::collections::HashMap;
use std::io;
use std::sync::Arc;

use super::partition::{PartitionKey, PartitionStrategy};
use super::shard::{ShardId, ShardManager, ShardMeta};

// ============================================================================
// Routing Decision
// ============================================================================

/// Result of routing analysis
#[derive(Debug, Clone)]
pub enum RoutingDecision {
    /// Query targets a single shard
    SingleShard { shard_id: ShardId },
    /// Query targets a subset of shards
    MultiShard { shard_ids: Vec<ShardId> },
    /// Query must scatter to all shards (full scan)
    AllShards { shard_ids: Vec<ShardId> },
    /// Table is not sharded (single-node mode)
    Local,
}

impl RoutingDecision {
    /// Get all shard IDs involved
    pub fn shard_ids(&self) -> Vec<ShardId> {
        match self {
            RoutingDecision::SingleShard { shard_id } => vec![*shard_id],
            RoutingDecision::MultiShard { shard_ids } => shard_ids.clone(),
            RoutingDecision::AllShards { shard_ids } => shard_ids.clone(),
            RoutingDecision::Local => Vec::new(),
        }
    }

    /// Whether this is a single-shard operation
    pub fn is_single_shard(&self) -> bool {
        matches!(
            self,
            RoutingDecision::SingleShard { .. } | RoutingDecision::Local
        )
    }

    /// Whether this requires scatter-gather
    pub fn is_scatter_gather(&self) -> bool {
        matches!(
            self,
            RoutingDecision::MultiShard { .. } | RoutingDecision::AllShards { .. }
        )
    }
}

// ============================================================================
// Shard Router
// ============================================================================

/// Routes queries and writes to the appropriate shard(s)
///
/// Uses the partition strategy to determine which shard(s) should handle
/// a given operation. Supports:
/// - Point queries (single shard via partition key)
/// - Range queries (subset of shards for range partitioning)
/// - Full scans (all shards)
pub struct ShardRouter {
    /// Shard manager reference
    shard_manager: Arc<ShardManager>,
    /// Table → partition strategy mapping
    strategies: HashMap<String, Box<dyn PartitionStrategy>>,
    /// Table → partition column mapping
    partition_columns: HashMap<String, String>,
}

impl ShardRouter {
    /// Create a new router
    pub fn new(shard_manager: Arc<ShardManager>) -> Self {
        Self {
            shard_manager,
            strategies: HashMap::new(),
            partition_columns: HashMap::new(),
        }
    }

    /// Register a partition strategy for a table
    pub fn register_strategy(
        &mut self,
        table_name: &str,
        partition_column: &str,
        strategy: Box<dyn PartitionStrategy>,
    ) {
        self.strategies.insert(table_name.to_string(), strategy);
        self.partition_columns
            .insert(table_name.to_string(), partition_column.to_string());
    }

    /// Get the partition column for a table
    pub fn partition_column(&self, table_name: &str) -> Option<&str> {
        self.partition_columns.get(table_name).map(|s| s.as_str())
    }

    // ========================================================================
    // Write Routing
    // ========================================================================

    /// Route a write operation to the correct shard
    pub fn route_write(&self, table_name: &str, partition_key: &PartitionKey) -> RoutingDecision {
        let shards = self.shard_manager.get_writable_shards(table_name);
        if shards.is_empty() {
            return RoutingDecision::Local;
        }

        if let Some(strategy) = self.strategies.get(table_name) {
            let shard_count = shards.len() as u32;
            let shard_idx = strategy.route(partition_key, shard_count) as usize;
            if shard_idx < shards.len() {
                RoutingDecision::SingleShard {
                    shard_id: shards[shard_idx].id,
                }
            } else {
                RoutingDecision::SingleShard {
                    shard_id: shards[0].id,
                }
            }
        } else {
            // No strategy = single shard mode
            RoutingDecision::SingleShard {
                shard_id: shards[0].id,
            }
        }
    }

    // ========================================================================
    // Read Routing
    // ========================================================================

    /// Route a point query to the correct shard
    pub fn route_point_query(
        &self,
        table_name: &str,
        partition_key: &PartitionKey,
    ) -> RoutingDecision {
        let shards = self.shard_manager.get_table_shards(table_name);
        let readable: Vec<_> = shards.into_iter().filter(|s| s.is_readable()).collect();

        if readable.is_empty() {
            return RoutingDecision::Local;
        }

        if let Some(strategy) = self.strategies.get(table_name) {
            let shard_count = readable.len() as u32;
            let shard_idx = strategy.route(partition_key, shard_count) as usize;
            if shard_idx < readable.len() {
                RoutingDecision::SingleShard {
                    shard_id: readable[shard_idx].id,
                }
            } else {
                RoutingDecision::SingleShard {
                    shard_id: readable[0].id,
                }
            }
        } else {
            RoutingDecision::SingleShard {
                shard_id: readable[0].id,
            }
        }
    }

    /// Route a range query
    pub fn route_range_query(
        &self,
        table_name: &str,
        low: &PartitionKey,
        high: &PartitionKey,
    ) -> RoutingDecision {
        let shards = self.shard_manager.get_table_shards(table_name);
        let readable: Vec<_> = shards.into_iter().filter(|s| s.is_readable()).collect();

        if readable.is_empty() {
            return RoutingDecision::Local;
        }

        if let Some(strategy) = self.strategies.get(table_name) {
            let shard_count = readable.len() as u32;
            if let Some(target_shards) = strategy.route_range(low, high, shard_count) {
                let shard_ids: Vec<ShardId> = target_shards
                    .into_iter()
                    .filter_map(|idx| readable.get(idx as usize).map(|s| s.id))
                    .collect();
                if shard_ids.len() == 1 {
                    RoutingDecision::SingleShard {
                        shard_id: shard_ids[0],
                    }
                } else {
                    RoutingDecision::MultiShard { shard_ids }
                }
            } else {
                // Strategy can't prune → all shards
                RoutingDecision::AllShards {
                    shard_ids: readable.iter().map(|s| s.id).collect(),
                }
            }
        } else {
            RoutingDecision::AllShards {
                shard_ids: readable.iter().map(|s| s.id).collect(),
            }
        }
    }

    /// Route a full scan query (scatter to all shards)
    pub fn route_full_scan(&self, table_name: &str) -> RoutingDecision {
        let shards = self.shard_manager.get_table_shards(table_name);
        let readable: Vec<_> = shards.into_iter().filter(|s| s.is_readable()).collect();

        if readable.is_empty() {
            return RoutingDecision::Local;
        }

        RoutingDecision::AllShards {
            shard_ids: readable.iter().map(|s| s.id).collect(),
        }
    }

    /// Check if a table is sharded
    pub fn is_sharded(&self, table_name: &str) -> bool {
        self.shard_manager.shard_count(table_name) > 1
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::partition::HashPartitioner;
    use super::*;

    #[test]
    fn test_local_routing() {
        let dir = tempfile::tempdir().unwrap();
        let shard_mgr = Arc::new(ShardManager::new(dir.path(), 1));
        let router = ShardRouter::new(shard_mgr);

        let decision = router.route_write("users", &PartitionKey::Int(1));
        assert!(matches!(decision, RoutingDecision::Local));
    }

    #[test]
    fn test_single_shard_routing() {
        let dir = tempfile::tempdir().unwrap();
        let shard_mgr = Arc::new(ShardManager::new(dir.path(), 1));
        shard_mgr.create_shard("users").unwrap();

        let mut router = ShardRouter::new(shard_mgr);
        router.register_strategy("users", "_id", Box::new(HashPartitioner::new()));

        let decision = router.route_write("users", &PartitionKey::Int(42));
        assert!(decision.is_single_shard());
    }

    #[test]
    fn test_routing_decision_helpers() {
        let single = RoutingDecision::SingleShard { shard_id: 1 };
        assert!(single.is_single_shard());
        assert!(!single.is_scatter_gather());

        let multi = RoutingDecision::MultiShard {
            shard_ids: vec![1, 2, 3],
        };
        assert!(!multi.is_single_shard());
        assert!(multi.is_scatter_gather());
        assert_eq!(multi.shard_ids().len(), 3);

        let local = RoutingDecision::Local;
        assert!(local.is_single_shard());
        assert!(local.shard_ids().is_empty());
    }
}
