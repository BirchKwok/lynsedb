//! Transaction Manager - ACID transaction support for HTAP
//!
//! Provides BEGIN / COMMIT / ROLLBACK semantics with OCC (Optimistic Concurrency Control).
//!
//! Architecture:
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │              TxnManager                           │
//! │  - Creates and tracks active transactions        │
//! │  - Assigns monotonic transaction IDs             │
//! │  - Coordinates commit/abort                      │
//! ├──────────────────────────────────────────────────┤
//! │  TxnContext                                      │
//! │  - Per-transaction read/write sets               │
//! │  - Buffered writes (applied on commit)           │
//! │  - Undo log for rollback                         │
//! ├──────────────────────────────────────────────────┤
//! │  ConflictDetector (OCC)                          │
//! │  - Validates read set at commit time             │
//! │  - Detects write-write conflicts                 │
//! │  - First-committer-wins strategy                 │
//! └──────────────────────────────────────────────────┘
//! ```

pub mod conflict;
pub mod context;
pub mod manager;

pub use conflict::{ConflictDetector, ConflictResult};
pub use context::TxnContext;
pub use manager::{txn_manager, TxnId, TxnManager, TxnStatus};
