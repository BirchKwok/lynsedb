//! Node Manager - Tracks cluster membership and node health
//!
//! In a distributed deployment, manages the set of nodes that form the cluster.
//! For single-node (embedded) mode, this simply tracks the local node.
//!
//! Designed to be extended with:
//! - Network-based health checking
//! - Leader election (Raft/Paxos)
//! - Automatic failover

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

// ============================================================================
// Node ID
// ============================================================================

/// Unique node identifier
pub type NodeId = u32;

/// Global node ID generator
static NEXT_NODE_ID: AtomicU32 = AtomicU32::new(1);

fn next_node_id() -> NodeId {
    NEXT_NODE_ID.fetch_add(1, Ordering::SeqCst)
}

// ============================================================================
// Node Status
// ============================================================================

/// Current status of a node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is up and serving requests
    Active,
    /// Node is joining the cluster
    Joining,
    /// Node is leaving the cluster (draining)
    Leaving,
    /// Node is suspected down (missed heartbeats)
    Suspect,
    /// Node is confirmed down
    Down,
}

// ============================================================================
// Node Info
// ============================================================================

/// Information about a single node in the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Unique node ID
    pub id: NodeId,
    /// Human-readable name
    pub name: String,
    /// Network address (host:port) — empty for embedded mode
    pub address: String,
    /// Current status
    pub status: NodeStatus,
    /// Data directory on this node
    pub data_dir: String,
    /// Number of shards hosted on this node
    pub shard_count: u32,
    /// Approximate total data size (bytes)
    pub data_size: u64,
    /// Last heartbeat time (epoch millis)
    pub last_heartbeat: i64,
    /// Node join time
    pub joined_at: i64,
    /// Node capabilities/tags (e.g., "gpu", "ssd", "region:us-east")
    pub tags: Vec<String>,
}

impl NodeInfo {
    /// Create info for the local node
    pub fn local(name: &str, data_dir: &str) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: next_node_id(),
            name: name.to_string(),
            address: String::new(),
            status: NodeStatus::Active,
            data_dir: data_dir.to_string(),
            shard_count: 0,
            data_size: 0,
            last_heartbeat: now,
            joined_at: now,
            tags: vec!["local".to_string()],
        }
    }

    /// Create info for a remote node
    pub fn remote(name: &str, address: &str) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: next_node_id(),
            name: name.to_string(),
            address: address.to_string(),
            status: NodeStatus::Joining,
            data_dir: String::new(),
            shard_count: 0,
            data_size: 0,
            last_heartbeat: now,
            joined_at: now,
            tags: Vec::new(),
        }
    }

    /// Whether the node is available for operations
    pub fn is_available(&self) -> bool {
        matches!(self.status, NodeStatus::Active)
    }
}

// ============================================================================
// Node Manager
// ============================================================================

/// Manages cluster membership and node health
///
/// For embedded/single-node mode, simply tracks the local node.
/// For distributed mode, provides the framework for:
/// - Node discovery and registration
/// - Heartbeat monitoring
/// - Failure detection
/// - Leader election (future)
pub struct NodeManager {
    /// All known nodes: node_id → NodeInfo
    nodes: RwLock<HashMap<NodeId, NodeInfo>>,
    /// Local node ID
    local_node_id: NodeId,
    /// Heartbeat timeout (if no heartbeat within this duration, mark suspect)
    heartbeat_timeout: Duration,
    /// Whether running in cluster mode
    cluster_mode: bool,
}

impl NodeManager {
    /// Create a node manager for single-node (embedded) mode
    pub fn new_local(name: &str, data_dir: &str) -> Self {
        let local_node = NodeInfo::local(name, data_dir);
        let local_id = local_node.id;
        let mut nodes = HashMap::new();
        nodes.insert(local_id, local_node);

        Self {
            nodes: RwLock::new(nodes),
            local_node_id: local_id,
            heartbeat_timeout: Duration::from_secs(30),
            cluster_mode: false,
        }
    }

    /// Create a node manager for cluster mode
    pub fn new_cluster(local_name: &str, local_data_dir: &str) -> Self {
        let mut mgr = Self::new_local(local_name, local_data_dir);
        mgr.cluster_mode = true;
        mgr
    }

    /// Local node ID
    pub fn local_node_id(&self) -> NodeId {
        self.local_node_id
    }

    /// Whether in cluster mode
    pub fn is_cluster_mode(&self) -> bool {
        self.cluster_mode
    }

    // ========================================================================
    // Node Registration
    // ========================================================================

    /// Register a new node in the cluster
    pub fn register_node(&self, info: NodeInfo) -> NodeId {
        let id = info.id;
        self.nodes.write().insert(id, info);
        id
    }

    /// Remove a node from the cluster
    pub fn remove_node(&self, node_id: NodeId) -> Option<NodeInfo> {
        if node_id == self.local_node_id {
            return None; // Can't remove self
        }
        self.nodes.write().remove(&node_id)
    }

    /// Get node info
    pub fn get_node(&self, node_id: NodeId) -> Option<NodeInfo> {
        self.nodes.read().get(&node_id).cloned()
    }

    /// Get all active nodes
    pub fn active_nodes(&self) -> Vec<NodeInfo> {
        self.nodes
            .read()
            .values()
            .filter(|n| n.is_available())
            .cloned()
            .collect()
    }

    /// Get all nodes
    pub fn all_nodes(&self) -> Vec<NodeInfo> {
        self.nodes.read().values().cloned().collect()
    }

    /// Number of active nodes
    pub fn active_count(&self) -> usize {
        self.nodes
            .read()
            .values()
            .filter(|n| n.is_available())
            .count()
    }

    /// Total node count
    pub fn total_count(&self) -> usize {
        self.nodes.read().len()
    }

    // ========================================================================
    // Health Monitoring
    // ========================================================================

    /// Record a heartbeat from a node
    pub fn record_heartbeat(&self, node_id: NodeId) {
        let mut nodes = self.nodes.write();
        if let Some(node) = nodes.get_mut(&node_id) {
            node.last_heartbeat = chrono::Utc::now().timestamp();
            if node.status == NodeStatus::Suspect {
                node.status = NodeStatus::Active;
            }
        }
    }

    /// Update node status
    pub fn set_node_status(&self, node_id: NodeId, status: NodeStatus) {
        let mut nodes = self.nodes.write();
        if let Some(node) = nodes.get_mut(&node_id) {
            node.status = status;
        }
    }

    /// Check for nodes that have missed heartbeats
    pub fn check_health(&self) -> Vec<NodeId> {
        let now = chrono::Utc::now().timestamp();
        let timeout_secs = self.heartbeat_timeout.as_secs() as i64;
        let mut suspect_nodes = Vec::new();

        let mut nodes = self.nodes.write();
        for (id, node) in nodes.iter_mut() {
            if *id == self.local_node_id {
                continue; // Skip self
            }
            if node.status == NodeStatus::Active && (now - node.last_heartbeat) > timeout_secs {
                node.status = NodeStatus::Suspect;
                suspect_nodes.push(*id);
            }
        }

        suspect_nodes
    }

    /// Update node statistics
    pub fn update_stats(&self, node_id: NodeId, shard_count: u32, data_size: u64) {
        let mut nodes = self.nodes.write();
        if let Some(node) = nodes.get_mut(&node_id) {
            node.shard_count = shard_count;
            node.data_size = data_size;
        }
    }

    // ========================================================================
    // Node Selection (for shard placement)
    // ========================================================================

    /// Select the best node for placing a new shard
    /// Strategy: least-loaded node
    pub fn select_node_for_shard(&self) -> Option<NodeId> {
        let nodes = self.nodes.read();
        nodes
            .values()
            .filter(|n| n.is_available())
            .min_by_key(|n| n.shard_count)
            .map(|n| n.id)
    }

    /// Select N nodes for replica placement (excluding the primary node)
    pub fn select_replica_nodes(&self, primary_node: NodeId, count: usize) -> Vec<NodeId> {
        let nodes = self.nodes.read();
        let mut candidates: Vec<_> = nodes
            .values()
            .filter(|n| n.is_available() && n.id != primary_node)
            .collect();
        candidates.sort_by_key(|n| n.shard_count);
        candidates.iter().take(count).map(|n| n.id).collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_node_manager() {
        let mgr = NodeManager::new_local("node1", "/data");
        assert_eq!(mgr.total_count(), 1);
        assert_eq!(mgr.active_count(), 1);
        assert!(!mgr.is_cluster_mode());
    }

    #[test]
    fn test_register_remote_node() {
        let mgr = NodeManager::new_cluster("node1", "/data");
        let remote = NodeInfo::remote("node2", "192.168.1.2:9090");
        let remote_id = mgr.register_node(remote);

        assert_eq!(mgr.total_count(), 2);
        let info = mgr.get_node(remote_id).unwrap();
        assert_eq!(info.name, "node2");
    }

    #[test]
    fn test_select_node_for_shard() {
        let mgr = NodeManager::new_cluster("node1", "/data");
        let node_id = mgr.select_node_for_shard();
        assert!(node_id.is_some());
    }

    #[test]
    fn test_cannot_remove_self() {
        let mgr = NodeManager::new_local("node1", "/data");
        let local_id = mgr.local_node_id();
        assert!(mgr.remove_node(local_id).is_none());
    }

    #[test]
    fn test_health_check() {
        let mgr = NodeManager::new_cluster("node1", "/data");

        // Add a remote node with old heartbeat
        let mut remote = NodeInfo::remote("node2", "192.168.1.2:9090");
        remote.status = NodeStatus::Active;
        remote.last_heartbeat = chrono::Utc::now().timestamp() - 60; // 60 seconds ago
        let remote_id = remote.id;
        mgr.register_node(remote);

        // Short timeout for test
        // Note: default timeout is 30s, so a 60s-ago heartbeat should be suspect
        let suspect = mgr.check_health();
        assert!(suspect.contains(&remote_id));
    }
}
