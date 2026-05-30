//! Query parsing and execution
//!
//! This module provides SQL:2023 compliant query parsing and execution.

pub(crate) mod executor;
mod expr_compiler;
mod filter;
pub mod jit;
pub mod multi_column;
pub mod planner;
pub mod query_signature;
pub mod scheduler;
pub mod simd_take;
pub(crate) mod sql_parser;
pub mod vector_ops;
pub mod vectorized;
pub mod vectorized_join;

pub use executor::{
    get_cached_backend_pub, get_session_variable, reset_session_variable, set_session_variable,
    ApexExecutor, ApexResult,
};
pub use expr_compiler::sql_expr_to_filter;
pub use filter::{CompareOp, Filter, LikeMatcher, RegexpMatcher};
pub use query_signature::QuerySignature;
pub use sql_parser::{
    AggregateFunc,
    AlterTableOp,
    // DDL types
    ColumnDef,
    FromItem,
    JoinClause,
    JoinType,
    OrderByClause,
    SelectColumn,
    SelectStatement,
    SetOpType,
    SqlExpr,
    SqlParser,
    SqlStatement,
    UnionStatement,
};
