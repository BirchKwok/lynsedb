//! Table management module
//!
//! Provides table catalog and Arrow-based column storage.

pub mod arrow_column;
mod catalog;
pub mod column_table;
mod schema;

pub use arrow_column::{ArrowStringColumn, ArrowTypedColumn};
pub use catalog::{TableCatalog, TableEntry};
pub use column_table::{BitVec, ColumnSchema, TypedColumn};
pub use schema::Schema;
