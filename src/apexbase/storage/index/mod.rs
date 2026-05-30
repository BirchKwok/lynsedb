//! Index subsystem for HTAP hybrid database
//!
//! Provides B-Tree and Hash indexes for fast point lookups (OLTP path)
//! while maintaining compatibility with columnar scan (OLAP path).
//!
//! Architecture:
//! ```text
//! ┌──────────────────────────────────────────┐
//! │           IndexManager                    │
//! │  - Manages all indexes for a table       │
//! │  - Auto-selects best index for query     │
//! │  - Persists index metadata               │
//! ├──────────────┬───────────────────────────┤
//! │  BTreeIndex  │     HashIndex             │
//! │  - Range     │  - Point lookup O(1)      │
//! │    queries   │  - Equality only          │
//! │  - O(log N)  │  - Memory-resident        │
//! └──────────────┴───────────────────────────┘
//! ```

pub mod btree;
pub mod hash_index;
pub mod index_manager;

pub use btree::BTreeIndex;
pub use hash_index::HashIndex;
pub use index_manager::{IndexManager, IndexMeta, IndexType};
