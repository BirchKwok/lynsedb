//! ApexBase Core Storage Engine
//!
//! A high-performance embedded database storage engine implemented in Rust.
//! Provides Python bindings via PyO3 for seamless integration.

pub mod data;
pub mod embedded;
pub mod fts;
pub mod query;
pub mod scaling;
pub mod storage;
pub mod table;
pub mod txn;

// Re-export main types
pub use data::{DataType, Row, Value};
pub use query::{ApexExecutor, ApexResult};
pub use storage::{ColumnType, ColumnValue, ColumnarStorage, FileSchema};
pub use table::TableCatalog;

// Re-export embedded API for Rust users
pub use embedded::{ApexDB, ResultSet, Table};

/// Storage engine error type
#[derive(Debug, thiserror::Error)]
pub enum ApexError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Table not found: {0}")]
    TableNotFound(String),

    #[error("Table already exists: {0}")]
    TableExists(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Column already exists: {0}")]
    ColumnExists(String),

    #[error("Row not found: {0}")]
    RowNotFound(u64),

    #[error("Invalid data type: {0}")]
    InvalidDataType(String),

    #[error("Query parse error: {0}")]
    QueryParseError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Checksum mismatch")]
    ChecksumMismatch,

    #[error("Invalid file format")]
    InvalidFileFormat,

    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },

    #[error("Cannot drop default table")]
    CannotDropDefaultTable,

    #[error("Cannot modify _id column")]
    CannotModifyIdColumn,
}

pub type Result<T> = std::result::Result<T, ApexError>;
