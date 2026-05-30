//! Horizontal Scaling Framework
//!
//! Provides abstractions for distributing data across multiple nodes/shards.
//! Designed as a pluggable framework that can be implemented with different
//! backends (local filesystem, network, cloud storage).
//!
//! Architecture:
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                    ShardRouter                            │
//! │  - Routes queries/writes to correct shard                │
//! │  - Handles cross-shard queries (scatter-gather)          │
//! │  - Manages shard topology changes                        │
//! ├──────────────────────────────────────────────────────────┤
//! │  ShardManager                                            │
//! │  - Manages shard lifecycle (create/split/merge/migrate)  │
//! │  - Monitors shard health and rebalancing                 │
//! │  - Coordinates shard metadata                            │
//! ├──────────────────────────────────────────────────────────┤
//! │  PartitionStrategy                                       │
//! │  - Hash partitioning (uniform distribution)              │
//! │  - Range partitioning (ordered data)                     │
//! │  - Custom partitioning (user-defined)                    │
//! ├──────────────────────────────────────────────────────────┤
//! │  NodeManager                                             │
//! │  - Tracks cluster membership                             │
//! │  - Health checking                                       │
//! │  - Leader election (future)                              │
//! └──────────────────────────────────────────────────────────┘
//! ```

pub mod node;
pub mod partition;
pub mod router;
pub mod shard;

pub use node::{NodeId, NodeInfo, NodeManager, NodeStatus};
pub use partition::{HashPartitioner, PartitionKey, PartitionStrategy, RangePartitioner};
pub use router::{RoutingDecision, ShardRouter};
pub use shard::{ShardId, ShardManager, ShardMeta, ShardStatus};
