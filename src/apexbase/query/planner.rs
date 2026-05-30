//! Query Planner - Routes queries to OLTP or OLAP execution paths
//!
//! Analyzes SQL queries and selects the optimal execution strategy:
//! - **OLTP path**: Index-based point lookups, single-row mutations
//! - **OLAP path**: Vectorized columnar scans with SIMD/JIT
//!
//! Architecture:
//! ```text
//! ┌─────────────────┐
//! │   SQL Query      │
//! └────────┬────────┘
//!          │
//! ┌────────▼────────┐
//! │  QueryPlanner    │
//! │  - Analyze AST   │
//! │  - Check indexes │
//! │  - Estimate cost │
//! └────────┬────────┘
//!          │
//!   ┌──────┴──────────┐
//!   │                 │
//!   ▼                 ▼
//! ┌──────────┐  ┌──────────────┐
//! │ OLTP     │  │  OLAP        │
//! │ Executor │  │  Executor    │
//! │ (index)  │  │  (vectorized)│
//! └──────────┘  └──────────────┘
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use once_cell::sync::Lazy;
use parking_lot::RwLock;

use crate::data::Value;
use crate::query::sql_parser::BinaryOperator;
use crate::query::{AggregateFunc, SelectColumn, SelectStatement, SqlExpr, SqlStatement};
use crate::storage::index::IndexManager;

// ============================================================================
// Table Statistics Cache (for CBO)
// ============================================================================

/// Per-column statistics collected by ANALYZE
#[derive(Debug, Clone)]
pub struct ColumnStats {
    /// Number of distinct values
    pub ndv: u64,
    /// Number of null values
    pub null_count: u64,
    /// Min value (as string for universal comparison)
    pub min_value: String,
    /// Max value (as string for universal comparison)
    pub max_value: String,
}

/// Per-table statistics collected by ANALYZE
#[derive(Debug, Clone)]
pub struct TableStats {
    /// Total row count
    pub row_count: u64,
    /// Per-column statistics: column_name → stats
    pub columns: HashMap<String, ColumnStats>,
    /// Timestamp when stats were collected (epoch millis)
    pub collected_at: u64,
}

/// Global stats cache: table_path → TableStats
static STATS_CACHE: Lazy<RwLock<HashMap<String, TableStats>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Store ANALYZE results into the stats cache
pub fn store_table_stats(table_key: &str, stats: TableStats) {
    STATS_CACHE.write().insert(table_key.to_string(), stats);
}

/// Retrieve cached stats for a table
pub fn get_table_stats(table_key: &str) -> Option<TableStats> {
    STATS_CACHE.read().get(table_key).cloned()
}

/// Invalidate stats for a table (e.g., after DML)
pub fn invalidate_table_stats(table_key: &str) {
    STATS_CACHE.write().remove(table_key);
}

// ============================================================================
// Cost Model
// ============================================================================

/// Cost of different operations (relative units)
const COST_SEQ_SCAN_PER_ROW: f64 = 1.0;
const COST_INDEX_LOOKUP: f64 = 4.0;
const COST_INDEX_SCAN_PER_ROW: f64 = 1.5;
const COST_HASH_BUILD_PER_ROW: f64 = 2.0;
const COST_HASH_PROBE_PER_ROW: f64 = 0.5;
const COST_SORT_PER_ROW_LOG: f64 = 0.1;

/// Estimated cost of an execution plan
#[derive(Debug, Clone)]
pub struct PlanCost {
    /// Total estimated cost (lower is better)
    pub total: f64,
    /// Estimated output rows
    pub output_rows: f64,
}

impl PlanCost {
    fn seq_scan(row_count: f64) -> Self {
        Self {
            total: row_count * COST_SEQ_SCAN_PER_ROW,
            output_rows: row_count,
        }
    }

    fn index_scan(row_count: f64, selectivity: f64) -> Self {
        let output = row_count * selectivity;
        Self {
            total: COST_INDEX_LOOKUP + output * COST_INDEX_SCAN_PER_ROW,
            output_rows: output,
        }
    }

    fn hash_join(left: &PlanCost, right: &PlanCost) -> Self {
        let build = left.output_rows * COST_HASH_BUILD_PER_ROW;
        let probe = right.output_rows * COST_HASH_PROBE_PER_ROW;
        Self {
            total: left.total + right.total + build + probe,
            output_rows: left.output_rows.min(right.output_rows),
        }
    }
}

// ============================================================================
// Selectivity Estimator
// ============================================================================

impl QueryPlanner {
    /// Estimate selectivity of a WHERE expression using table stats
    pub fn estimate_selectivity(expr: &SqlExpr, stats: &TableStats) -> f64 {
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                match op {
                    BinaryOperator::Eq => {
                        if let SqlExpr::Column(col) = left.as_ref() {
                            if let Some(cs) = stats.columns.get(col) {
                                if cs.ndv > 0 {
                                    return 1.0 / cs.ndv as f64;
                                }
                            }
                        }
                        0.01 // default for equality
                    }
                    BinaryOperator::Gt
                    | BinaryOperator::Ge
                    | BinaryOperator::Lt
                    | BinaryOperator::Le => {
                        // Range: assume uniform distribution → 1/3
                        0.33
                    }
                    BinaryOperator::And => {
                        let s1 = Self::estimate_selectivity(left, stats);
                        let s2 = Self::estimate_selectivity(right, stats);
                        s1 * s2
                    }
                    BinaryOperator::Or => {
                        let s1 = Self::estimate_selectivity(left, stats);
                        let s2 = Self::estimate_selectivity(right, stats);
                        (s1 + s2 - s1 * s2).min(1.0)
                    }
                    BinaryOperator::NotEq => {
                        if let SqlExpr::Column(col) = left.as_ref() {
                            if let Some(cs) = stats.columns.get(col) {
                                if cs.ndv > 0 {
                                    return 1.0 - 1.0 / cs.ndv as f64;
                                }
                            }
                        }
                        0.99
                    }
                    _ => 0.5,
                }
            }
            SqlExpr::Between { column, .. } => {
                // Assume 10-20% selectivity for BETWEEN
                0.15
            }
            SqlExpr::In { column, values, .. } => {
                if let Some(cs) = stats.columns.get(column) {
                    if cs.ndv > 0 {
                        return (values.len() as f64 / cs.ndv as f64).min(1.0);
                    }
                }
                (values.len() as f64 * 0.01).min(0.5)
            }
            SqlExpr::Like { .. } => 0.1,
            SqlExpr::IsNull { negated, .. } => {
                if *negated {
                    0.95
                } else {
                    0.05
                }
            }
            SqlExpr::UnaryOp {
                op: crate::query::sql_parser::UnaryOperator::Not,
                expr,
            } => 1.0 - Self::estimate_selectivity(expr, stats),
            _ => 0.5,
        }
    }

    /// Determine whether to use an index or full scan based on cost
    pub fn should_use_index(col: &str, selectivity: f64, row_count: u64) -> bool {
        let scan_cost = PlanCost::seq_scan(row_count as f64);
        let index_cost = PlanCost::index_scan(row_count as f64, selectivity);
        index_cost.total < scan_cost.total
    }

    /// Optimize join order for a list of tables with estimated row counts
    /// Returns tables sorted by ascending row count (smallest table first = build side)
    pub fn optimize_join_order(tables: &[(String, u64)]) -> Vec<String> {
        let mut sorted = tables.to_vec();
        sorted.sort_by_key(|(_, rows)| *rows);
        sorted.into_iter().map(|(name, _)| name).collect()
    }
}

// ============================================================================
// Execution Strategy
// ============================================================================

/// The chosen execution strategy for a query
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionStrategy {
    /// Use OLTP path: index-based lookups
    OltpIndexLookup {
        /// Column to use for index lookup
        column: String,
        /// Type of lookup
        lookup_type: IndexLookupType,
    },
    /// Use OLTP path: primary key lookup (_id = X)
    OltpPrimaryKey {
        /// The _id value to look up
        id_value: i64,
    },
    /// Use OLAP path: full vectorized columnar scan
    OlapFullScan,
    /// Use OLAP path: vectorized scan with filter pushdown
    OlapFilteredScan,
    /// Use OLAP path: aggregation query
    OlapAggregation,
    /// Direct write (INSERT/UPDATE/DELETE)
    DirectWrite,
    /// DDL operation (CREATE/ALTER/DROP TABLE)
    Ddl,
}

/// Type of index lookup for OLTP path
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexLookupType {
    /// Exact equality: col = value
    Equality,
    /// Range: col BETWEEN low AND high
    Range,
    /// IN list: col IN (v1, v2, ...)
    InList,
}

// ============================================================================
// Query Characteristics
// ============================================================================

/// Analyzed characteristics of a query
#[derive(Debug, Clone)]
pub struct QueryCharacteristics {
    /// Whether the query has aggregation functions
    pub has_aggregation: bool,
    /// Whether the query has GROUP BY
    pub has_group_by: bool,
    /// Whether the query has ORDER BY
    pub has_order_by: bool,
    /// Whether the query has JOIN
    pub has_join: bool,
    /// Whether the query has subqueries
    pub has_subquery: bool,
    /// Whether the query has LIMIT
    pub has_limit: bool,
    /// Whether the query filters on _id (primary key)
    pub filters_on_pk: bool,
    /// Columns used in WHERE clause equality conditions
    pub equality_filter_columns: Vec<String>,
    /// Columns used in WHERE clause range conditions
    pub range_filter_columns: Vec<String>,
    /// Estimated selectivity (0.0 = no rows, 1.0 = all rows)
    pub estimated_selectivity: f64,
    /// Whether this is a write operation
    pub is_write: bool,
    /// Whether this is a DDL operation
    pub is_ddl: bool,
}

impl Default for QueryCharacteristics {
    fn default() -> Self {
        Self {
            has_aggregation: false,
            has_group_by: false,
            has_order_by: false,
            has_join: false,
            has_subquery: false,
            has_limit: false,
            filters_on_pk: false,
            equality_filter_columns: Vec::new(),
            range_filter_columns: Vec::new(),
            estimated_selectivity: 1.0,
            is_write: false,
            is_ddl: false,
        }
    }
}

// ============================================================================
// Query Planner
// ============================================================================

/// Query planner that analyzes SQL and selects execution strategy
pub struct QueryPlanner;

impl QueryPlanner {
    /// Analyze a parsed SQL statement and determine the best execution strategy
    pub fn plan(stmt: &SqlStatement, index_manager: Option<&IndexManager>) -> ExecutionStrategy {
        match stmt {
            SqlStatement::Select(select) => Self::plan_select(select, index_manager),
            SqlStatement::Insert { .. } => ExecutionStrategy::DirectWrite,
            SqlStatement::Update { .. } => ExecutionStrategy::DirectWrite,
            SqlStatement::Delete { .. } => ExecutionStrategy::DirectWrite,
            SqlStatement::CreateTable { .. }
            | SqlStatement::DropTable { .. }
            | SqlStatement::AlterTable { .. }
            | SqlStatement::TruncateTable { .. } => ExecutionStrategy::Ddl,
            _ => ExecutionStrategy::OlapFullScan,
        }
    }

    /// Plan a SELECT query
    fn plan_select(
        select: &SelectStatement,
        index_manager: Option<&IndexManager>,
    ) -> ExecutionStrategy {
        let chars = Self::analyze_select(select);

        // DDL/write check
        if chars.is_write {
            return ExecutionStrategy::DirectWrite;
        }
        if chars.is_ddl {
            return ExecutionStrategy::Ddl;
        }

        // Aggregation queries always go OLAP
        if chars.has_aggregation || chars.has_group_by {
            return ExecutionStrategy::OlapAggregation;
        }

        // JOIN queries always go OLAP
        if chars.has_join || chars.has_subquery {
            return ExecutionStrategy::OlapFullScan;
        }

        // Primary key lookup: _id = X
        if chars.filters_on_pk {
            if let Some(id) = Self::extract_pk_value(&select.where_clause) {
                return ExecutionStrategy::OltpPrimaryKey { id_value: id };
            }
        }

        // Check if we have an index for any equality filter column
        if let Some(idx_mgr) = index_manager {
            for col in &chars.equality_filter_columns {
                if idx_mgr.has_index_on(col) {
                    return ExecutionStrategy::OltpIndexLookup {
                        column: col.clone(),
                        lookup_type: IndexLookupType::Equality,
                    };
                }
            }
            for col in &chars.range_filter_columns {
                if idx_mgr.has_index_on(col) {
                    return ExecutionStrategy::OltpIndexLookup {
                        column: col.clone(),
                        lookup_type: IndexLookupType::Range,
                    };
                }
            }
        }

        // High selectivity with LIMIT → OLAP with filter pushdown
        if chars.estimated_selectivity < 0.1 || chars.has_limit {
            return ExecutionStrategy::OlapFilteredScan;
        }

        // Default: OLAP full scan
        ExecutionStrategy::OlapFullScan
    }

    /// Plan with CBO: use table stats for cost-based index/scan decisions
    pub fn plan_with_stats(
        stmt: &SqlStatement,
        index_manager: Option<&IndexManager>,
        table_key: &str,
    ) -> ExecutionStrategy {
        let stats = get_table_stats(table_key);
        match stmt {
            SqlStatement::Select(select) => {
                Self::plan_select_with_stats(select, index_manager, stats.as_ref())
            }
            _ => Self::plan(stmt, index_manager),
        }
    }

    /// Plan SELECT with CBO — takes &SelectStatement directly, avoiding a clone at the call site.
    pub fn plan_select_pub(
        select: &SelectStatement,
        index_manager: Option<&IndexManager>,
        table_key: &str,
    ) -> ExecutionStrategy {
        let stats = get_table_stats(table_key);
        Self::plan_select_with_stats(select, index_manager, stats.as_ref())
    }

    /// Plan SELECT with cost-based optimization using ANALYZE stats
    fn plan_select_with_stats(
        select: &SelectStatement,
        index_manager: Option<&IndexManager>,
        stats: Option<&TableStats>,
    ) -> ExecutionStrategy {
        let chars = Self::analyze_select(select);

        if chars.is_write {
            return ExecutionStrategy::DirectWrite;
        }
        if chars.is_ddl {
            return ExecutionStrategy::Ddl;
        }
        if chars.has_aggregation || chars.has_group_by {
            return ExecutionStrategy::OlapAggregation;
        }
        if chars.has_join || chars.has_subquery {
            return ExecutionStrategy::OlapFullScan;
        }

        // Primary key lookup
        if chars.filters_on_pk {
            if let Some(id) = Self::extract_pk_value(&select.where_clause) {
                return ExecutionStrategy::OltpPrimaryKey { id_value: id };
            }
        }

        // CBO: Use stats to decide index vs scan
        if let (Some(idx_mgr), Some(where_expr)) = (index_manager, &select.where_clause) {
            let row_count = stats.map(|s| s.row_count).unwrap_or(10000);

            // Estimate selectivity using stats if available
            let selectivity = if let Some(s) = stats {
                Self::estimate_selectivity(where_expr, s)
            } else {
                chars.estimated_selectivity
            };

            // Check each indexed column — use cost model to decide
            for col in chars
                .equality_filter_columns
                .iter()
                .chain(chars.range_filter_columns.iter())
            {
                if idx_mgr.has_index_on(col) {
                    let col_selectivity = if let Some(s) = stats {
                        if let Some(cs) = s.columns.get(col) {
                            if cs.ndv > 0 {
                                1.0 / cs.ndv as f64
                            } else {
                                selectivity
                            }
                        } else {
                            selectivity
                        }
                    } else {
                        selectivity
                    };

                    if Self::should_use_index(col, col_selectivity, row_count) {
                        let lookup_type = if chars.equality_filter_columns.contains(col) {
                            IndexLookupType::Equality
                        } else {
                            IndexLookupType::Range
                        };
                        return ExecutionStrategy::OltpIndexLookup {
                            column: col.clone(),
                            lookup_type,
                        };
                    }
                }
            }
        }

        if chars.estimated_selectivity < 0.1 || chars.has_limit {
            return ExecutionStrategy::OlapFilteredScan;
        }

        ExecutionStrategy::OlapFullScan
    }

    /// Analyze a SELECT statement to extract characteristics
    fn analyze_select(select: &SelectStatement) -> QueryCharacteristics {
        let mut chars = QueryCharacteristics::default();

        // Check for aggregation in select columns
        for col in &select.columns {
            match col {
                SelectColumn::Aggregate { .. } => {
                    chars.has_aggregation = true;
                }
                _ => {}
            }
        }

        // Check GROUP BY
        if !select.group_by.is_empty() {
            chars.has_group_by = true;
        }

        // Check ORDER BY
        if !select.order_by.is_empty() {
            chars.has_order_by = true;
        }

        // Check LIMIT
        if select.limit.is_some() {
            chars.has_limit = true;
        }

        // Check JOINs
        if !select.joins.is_empty() {
            chars.has_join = true;
        }

        // Analyze WHERE clause
        if let Some(ref where_expr) = select.where_clause {
            Self::analyze_where(where_expr, &mut chars);
        }

        chars
    }

    /// Analyze WHERE clause for index-friendly patterns
    fn analyze_where(expr: &SqlExpr, chars: &mut QueryCharacteristics) {
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                match op {
                    BinaryOperator::Eq => {
                        if let SqlExpr::Column(col) = left.as_ref() {
                            if col == "_id" {
                                chars.filters_on_pk = true;
                            }
                            chars.equality_filter_columns.push(col.clone());
                            chars.estimated_selectivity *= 0.01; // Very selective
                        }
                    }
                    BinaryOperator::Gt
                    | BinaryOperator::Ge
                    | BinaryOperator::Lt
                    | BinaryOperator::Le => {
                        if let SqlExpr::Column(col) = left.as_ref() {
                            chars.range_filter_columns.push(col.clone());
                            chars.estimated_selectivity *= 0.3; // Moderately selective
                        }
                    }
                    BinaryOperator::And => {
                        Self::analyze_where(left, chars);
                        Self::analyze_where(right, chars);
                    }
                    BinaryOperator::Or => {
                        Self::analyze_where(left, chars);
                        Self::analyze_where(right, chars);
                        chars.estimated_selectivity = (chars.estimated_selectivity * 2.0).min(1.0);
                    }
                    _ => {}
                }
            }
            SqlExpr::Between {
                column, low, high, ..
            } => {
                chars.range_filter_columns.push(column.clone());
                chars.estimated_selectivity *= 0.2;
            }
            SqlExpr::In { column, values, .. } => {
                chars.equality_filter_columns.push(column.clone());
                chars.estimated_selectivity *= (values.len() as f64 * 0.01).min(0.5);
            }
            _ => {}
        }
    }

    /// Extract primary key value from WHERE _id = X
    fn extract_pk_value(where_clause: &Option<SqlExpr>) -> Option<i64> {
        if let Some(SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        }) = where_clause
        {
            if let SqlExpr::Column(col) = left.as_ref() {
                if col == "_id" {
                    if let SqlExpr::Literal(val) = right.as_ref() {
                        return val.as_i64();
                    }
                }
            }
        }
        None
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_display() {
        let strategy = ExecutionStrategy::OltpPrimaryKey { id_value: 42 };
        assert_eq!(strategy, ExecutionStrategy::OltpPrimaryKey { id_value: 42 });
    }

    #[test]
    fn test_query_characteristics_default() {
        let chars = QueryCharacteristics::default();
        assert!(!chars.has_aggregation);
        assert!(!chars.has_group_by);
        assert_eq!(chars.estimated_selectivity, 1.0);
    }
}
