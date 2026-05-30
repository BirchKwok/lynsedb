//! Query Signature Classifier
//!
//! Single-point-of-truth for SQL query classification. All layers (Python client,
//! PyO3 bindings, executor) share this classification — no duplicate pattern matching.
//!
//! The classifier operates on raw SQL text (~2-5µs) and produces a `QuerySignature`
//! enum that determines the optimal execution path. This replaces the previous
//! architecture where 83 fast-path checks were scattered across 4 layers.

/// DDL sub-kind — pre-extracted table name avoids re-uppercasing in bindings.
#[derive(Debug, Clone, PartialEq)]
pub enum DdlKind {
    CreateTable { name: String },
    DropTable { name: String },
    Other,
}

/// Query signature — lightweight classification of a SQL statement.
///
/// Produced ONCE by `classify()`, then consumed by all layers without re-parsing.
/// Variants are ordered from most-specific (cheapest to execute) to least-specific.
#[derive(Debug, Clone, PartialEq)]
pub enum QuerySignature {
    /// `SELECT COUNT(*) FROM <table>` — O(1) metadata read
    CountStar { table: String },

    /// `SELECT * ... WHERE _id = N` — O(1) point lookup by primary key
    PointLookup { id: u64, table: Option<String> },

    /// `SELECT col1, col2 ... WHERE _id = N` — projected point lookup by primary key
    ProjectedPointLookup {
        id: u64,
        table: Option<String>,
        columns: Vec<String>,
    },

    /// `SELECT * ... WHERE _id IN (...)` — batch point lookup by primary key
    IdBatchLookup {
        ids: Vec<u64>,
        table: Option<String>,
    },

    /// `SELECT col1, col2 ... WHERE _id IN (...)` — projected batch primary-key lookup
    ProjectedIdBatchLookup {
        ids: Vec<u64>,
        table: Option<String>,
        columns: Vec<String>,
    },

    /// `SELECT * FROM <table>` — full table scan without parser/planner overhead
    FullScan { table: Option<String> },

    /// `SELECT col1, col2 FROM <table>` — projected full scan without parser/planner overhead
    ProjectedFullScan {
        table: Option<String>,
        columns: Vec<String>,
    },

    /// `SELECT * FROM <table> LIMIT N [OFFSET M]` — sequential scan with early termination
    SimpleScanLimit {
        limit: usize,
        offset: usize,
        table: Option<String>,
    },

    /// `SELECT col1, col2 FROM <table> LIMIT N [OFFSET M]` — projected sequential scan
    ProjectedScanLimit {
        limit: usize,
        offset: usize,
        table: Option<String>,
        columns: Vec<String>,
    },

    /// `SELECT * FROM <table> WHERE col = 'val'` — mmap string equality scan
    StringEqualityFilter {
        table: Option<String>,
        column: String,
        value: String,
    },

    /// `SELECT * FROM <table> WHERE numeric_col = N` — mmap numeric equality scan
    NumericEqualityFilter {
        table: Option<String>,
        column: String,
        value: i64,
    },

    /// `SELECT * FROM <table> WHERE numeric_col IN (N1, N2, ...)` — mmap numeric IN scan
    NumericInFilter {
        table: Option<String>,
        column: String,
        values: Vec<i64>,
    },

    /// `SELECT * FROM <table> WHERE col = 'val' LIMIT N [OFFSET M]` — early-terminating string equality scan
    StringEqualityFilterLimit {
        table: Option<String>,
        column: String,
        value: String,
        limit: usize,
        offset: usize,
    },

    /// `SELECT col1, col2 FROM <table> WHERE filter_col = 'val'` — projected string equality scan
    ProjectedStringEqualityFilter {
        table: Option<String>,
        columns: Vec<String>,
        column: String,
        value: String,
    },

    /// `SELECT col1, col2 FROM <table> WHERE filter_col = 'val' LIMIT N [OFFSET M]` — projected early-terminating string equality scan
    ProjectedStringEqualityFilterLimit {
        table: Option<String>,
        columns: Vec<String>,
        column: String,
        value: String,
        limit: usize,
        offset: usize,
    },

    /// `SELECT * FROM <table> WHERE numeric_col <op> N LIMIT M [OFFSET K]`
    NumericRangeFilterLimit {
        table: Option<String>,
        column: String,
        low: f64,
        high: f64,
        limit: usize,
        offset: usize,
    },

    /// `SELECT col1, col2 FROM <table> WHERE numeric_col <op> N LIMIT M [OFFSET K]`
    ProjectedNumericRangeFilterLimit {
        table: Option<String>,
        columns: Vec<String>,
        column: String,
        low: f64,
        high: f64,
        limit: usize,
        offset: usize,
    },

    /// `SELECT * FROM <table> WHERE col LIKE 'pattern'` — mmap LIKE scan
    LikeFilter {
        table: Option<String>,
        column: String,
        pattern: String,
    },

    /// `SELECT ... FROM read_csv(...) / read_parquet(...) / read_json(...)` — table function
    TableFunction,

    /// DuckDB-style direct file reading: `SELECT * FROM 'file.parquet' / 'file.csv' / 'file.json'`
    DirectFileRead,

    /// DDL: CREATE TABLE, DROP TABLE, ALTER TABLE, CREATE INDEX, etc.
    Ddl { kind: DdlKind },

    /// DML write: INSERT, UPDATE, DELETE, TRUNCATE, COPY IMPORT
    DmlWrite,

    /// Transaction control: BEGIN, COMMIT, ROLLBACK, SAVEPOINT, RELEASE
    Transaction,

    /// Multi-statement SQL (contains ';' separator)
    MultiStatement,

    /// SET / RESET variable
    SessionCommand,

    /// EXPLAIN / ANALYZE
    Explain,

    /// CTE: WITH ... AS (...)
    Cte,

    /// `SELECT COUNT(*), AVG(col), ... FROM <table> WHERE str_col = 'val'` — filtered string equality aggregation
    FilteredStringAgg {
        table: Option<String>,
        filter_column: String,
        filter_value: String,
    },

    /// Everything else — full parse + plan + execute
    Complex,
}

impl QuerySignature {
    /// Returns true if this signature represents a read-only query.
    #[inline]
    pub fn is_read_only(&self) -> bool {
        matches!(
            self,
            QuerySignature::CountStar { .. }
                | QuerySignature::PointLookup { .. }
                | QuerySignature::ProjectedPointLookup { .. }
                | QuerySignature::IdBatchLookup { .. }
                | QuerySignature::ProjectedIdBatchLookup { .. }
                | QuerySignature::FullScan { .. }
                | QuerySignature::ProjectedFullScan { .. }
                | QuerySignature::SimpleScanLimit { .. }
                | QuerySignature::ProjectedScanLimit { .. }
                | QuerySignature::StringEqualityFilter { .. }
                | QuerySignature::StringEqualityFilterLimit { .. }
                | QuerySignature::NumericEqualityFilter { .. }
                | QuerySignature::NumericInFilter { .. }
                | QuerySignature::ProjectedStringEqualityFilter { .. }
                | QuerySignature::ProjectedStringEqualityFilterLimit { .. }
                | QuerySignature::NumericRangeFilterLimit { .. }
                | QuerySignature::ProjectedNumericRangeFilterLimit { .. }
                | QuerySignature::LikeFilter { .. }
                | QuerySignature::FilteredStringAgg { .. }
                | QuerySignature::TableFunction
                | QuerySignature::DirectFileRead
                | QuerySignature::Explain
                | QuerySignature::Cte
                | QuerySignature::Complex
        )
    }

    /// Returns true if this is a write operation that needs locking.
    #[inline]
    pub fn needs_write_lock(&self) -> bool {
        matches!(self, QuerySignature::DmlWrite)
    }

    /// Returns true if the Python layer needs to hold its threading lock.
    #[inline]
    pub fn needs_python_lock(&self) -> bool {
        matches!(
            self,
            QuerySignature::DmlWrite
                | QuerySignature::Ddl { .. }
                | QuerySignature::Transaction
                | QuerySignature::MultiStatement
        )
    }

    /// Returns true if this signature can bypass SQL parsing entirely.
    #[inline]
    pub fn can_skip_parse(&self) -> bool {
        matches!(
            self,
            QuerySignature::CountStar { .. }
                | QuerySignature::PointLookup { .. }
                | QuerySignature::ProjectedPointLookup { .. }
                | QuerySignature::IdBatchLookup { .. }
                | QuerySignature::ProjectedIdBatchLookup { .. }
                | QuerySignature::FullScan { .. }
                | QuerySignature::ProjectedFullScan { .. }
                | QuerySignature::SimpleScanLimit { .. }
                | QuerySignature::ProjectedScanLimit { .. }
                | QuerySignature::StringEqualityFilter { .. }
                | QuerySignature::StringEqualityFilterLimit { .. }
                | QuerySignature::NumericEqualityFilter { .. }
                | QuerySignature::NumericInFilter { .. }
                | QuerySignature::ProjectedStringEqualityFilter { .. }
                | QuerySignature::ProjectedStringEqualityFilterLimit { .. }
                | QuerySignature::NumericRangeFilterLimit { .. }
                | QuerySignature::ProjectedNumericRangeFilterLimit { .. }
                | QuerySignature::LikeFilter { .. }
                | QuerySignature::FilteredStringAgg { .. }
        )
    }
}

/// Classify a SQL string into a `QuerySignature`.
///
/// This is the SINGLE point where SQL text pattern matching happens.
/// Cost: ~2-5µs (one uppercase pass + a handful of prefix/contains checks).
///
/// The function takes a pre-computed uppercase SQL to avoid redundant allocations
/// when the caller already has it.
pub fn classify(sql: &str) -> QuerySignature {
    let s = sql.trim();
    // We work on a bounded prefix for safety. 4 KiB comfortably covers common
    // `IN (...)` lookup lists from benchmarks while keeping classification cheap.
    let upper_buf: String;
    let su = if s.len() <= 4096 {
        upper_buf = s.to_ascii_uppercase();
        &upper_buf
    } else {
        // For very long SQL, only uppercase the first 4 KiB for classification
        upper_buf = s[..4096].to_ascii_uppercase();
        &upper_buf
    };

    // ── Multi-statement detection (must be first — overrides everything) ──
    {
        let trimmed = s.trim_end_matches(';').trim();
        if trimmed.contains(';') {
            return QuerySignature::MultiStatement;
        }
    }

    // ── Transaction commands ──
    if su.starts_with("BEGIN")
        || su == "COMMIT"
        || su == "COMMIT;"
        || su == "ROLLBACK"
        || su == "ROLLBACK;"
        || su.starts_with("SAVEPOINT ")
        || su.starts_with("ROLLBACK TO")
        || su.starts_with("RELEASE")
    {
        return QuerySignature::Transaction;
    }

    // ── Session commands ──
    if su.starts_with("SET ") || su.starts_with("RESET ") {
        return QuerySignature::SessionCommand;
    }

    // ── DDL ──
    if su.starts_with("CREATE ") || su.starts_with("DROP ") || su.starts_with("ALTER ") {
        return QuerySignature::Ddl {
            kind: extract_ddl_kind(s, su),
        };
    }

    // ── DML writes ──
    if su.starts_with("INSERT")
        || su.starts_with("UPDATE")
        || su.starts_with("DELETE")
        || su.starts_with("TRUNCATE")
    {
        // Special case: DELETE can still use pre-parse fast path inside executor
        return QuerySignature::DmlWrite;
    }

    // ── COPY (can be read or write) ──
    if su.starts_with("COPY ") {
        // COPY ... FROM is a write, COPY ... TO is a read
        return QuerySignature::DmlWrite;
    }

    // ── EXPLAIN ──
    if su.starts_with("EXPLAIN") {
        return QuerySignature::Explain;
    }

    // ── CTE ──
    if su.starts_with("WITH ") {
        return QuerySignature::Cte;
    }

    // ── REINDEX / PRAGMA ──
    if su.starts_with("REINDEX") || su.starts_with("PRAGMA") {
        return QuerySignature::Ddl {
            kind: DdlKind::Other,
        };
    }

    // ── SHOW / FTS DDL ──
    if su.starts_with("SHOW ") {
        return QuerySignature::Ddl {
            kind: DdlKind::Other,
        };
    }

    // ── SELECT queries — classify further ──
    if !su.starts_with("SELECT") {
        return QuerySignature::Complex;
    }

    if contains_unquoted_keyword(s, "UNION")
        || contains_unquoted_keyword(s, "INTERSECT")
        || contains_unquoted_keyword(s, "EXCEPT")
    {
        return QuerySignature::Complex;
    }

    // Guard flags for modifier keywords
    let has_where = su.contains("WHERE");
    let has_group = su.contains("GROUP");
    let has_having = su.contains("HAVING");
    let has_join = su.contains("JOIN");
    let has_order = contains_unquoted_keyword(s, "ORDER BY");
    let has_limit = su.contains("LIMIT");
    let has_distinct = su.contains("DISTINCT");

    // ── Table function: FROM READ_CSV / READ_PARQUET / READ_JSON ──
    if su.contains("FROM READ_CSV(")
        || su.contains("FROM READ_PARQUET(")
        || su.contains("FROM READ_JSON(")
    {
        return QuerySignature::TableFunction;
    }

    // ── DuckDB-style direct file reading: SELECT * FROM 'file.parquet' / 'file.csv' / 'file.json' ──
    if let Some(from_pos) = su.find("FROM '") {
        let after_from = &su[(from_pos + "FROM '".len())..];
        if let Some(end_quote) = after_from.find('\'') {
            let file = &after_from[..end_quote];
            if file.ends_with(".PARQUET")
                || file.ends_with(".CSV")
                || file.ends_with(".TSV")
                || file.ends_with(".JSON")
                || file.ends_with(".NDJSON")
                || file.ends_with(".JSONL")
                || file.ends_with(".CSV.GZ")
                || file.ends_with(".CSV.GZIP")
            {
                return QuerySignature::DirectFileRead;
            }
        }
    }

    // ── COUNT(*) — no WHERE/GROUP/HAVING/JOIN/DISTINCT ──
    if su.starts_with("SELECT COUNT(*) FROM ")
        && !has_where
        && !has_group
        && !has_having
        && !has_join
        && !has_distinct
    {
        let after_from = su["SELECT COUNT(*) FROM ".len()..].trim();
        let tname = after_from.trim_end_matches(';').trim();
        if !tname.is_empty() && !tname.contains(' ') {
            return QuerySignature::CountStar {
                table: tname.to_lowercase(),
            };
        }
    }

    let is_exact_star_select = has_exact_star_projection(s, su);
    let simple_projection = extract_simple_projection_columns(s, su);

    if let Some(columns) = simple_projection.clone() {
        if !has_limit && !has_order && !has_group && !has_join {
            if let Some(id) = extract_simple_id_equality(s, su) {
                let table = extract_from_table(s, su);
                return QuerySignature::ProjectedPointLookup { id, table, columns };
            }
        }

        if !has_limit && !has_order && !has_group && !has_join {
            if let Some(ids) = extract_simple_id_in_list(s, su) {
                let table = extract_from_table(s, su);
                return QuerySignature::ProjectedIdBatchLookup {
                    ids,
                    table,
                    columns,
                };
            }
        }

        if has_limit && !has_where && !has_order && !has_group && !has_join {
            if let Some((limit, offset)) = extract_limit_offset_from_upper(su) {
                return QuerySignature::ProjectedScanLimit {
                    limit,
                    offset,
                    table: extract_from_table(s, su),
                    columns,
                };
            }
        }

        if has_where
            && !has_order
            && !has_group
            && !has_join
            && !su.contains("BETWEEN")
            && !su.contains(" IN ")
            && !su.contains('>')
            && !su.contains('<')
            && !su.contains(" LIKE ")
            && s.contains('\'')
        {
            if has_limit {
                if let Some((column, value, limit, offset)) =
                    extract_string_equality_with_limit(s, su)
                {
                    return QuerySignature::ProjectedStringEqualityFilterLimit {
                        table: extract_from_table(s, su),
                        columns,
                        column,
                        value,
                        limit,
                        offset,
                    };
                }
            }
            if let Some((column, value)) = extract_string_equality(s, su) {
                return QuerySignature::ProjectedStringEqualityFilter {
                    table: extract_from_table(s, su),
                    columns,
                    column,
                    value,
                };
            }
        }

        if has_where
            && has_limit
            && !has_order
            && !has_group
            && !has_join
            && !su.contains("BETWEEN")
            && !su.contains(" IN ")
            && !su.contains(" LIKE ")
            && !su.contains(" AND ")
            && !su.contains(" OR ")
            && !s.contains('\'')
        {
            if let Some((column, low, high, limit, offset)) =
                extract_numeric_range_with_limit(s, su)
            {
                return QuerySignature::ProjectedNumericRangeFilterLimit {
                    table: extract_from_table(s, su),
                    columns,
                    column,
                    low,
                    high,
                    limit,
                    offset,
                };
            }
        }

        if !has_where && !has_order && !has_group && !has_join && !has_limit && !has_distinct {
            let table = extract_from_table(s, su);
            return QuerySignature::ProjectedFullScan { table, columns };
        }
    }

    // ── Point lookup: SELECT * ... WHERE _ID = N ──
    if is_exact_star_select && !has_limit && !has_order && !has_group && !has_join {
        if let Some(id) = extract_simple_id_equality(s, su) {
            let table = extract_from_table(s, su);
            return QuerySignature::PointLookup { id, table };
        }
    }

    // ── Batch point lookup: SELECT * ... WHERE _ID IN (...) ──
    if is_exact_star_select && !has_limit && !has_order && !has_group && !has_join {
        if let Some(ids) = extract_simple_id_in_list(s, su) {
            let table = extract_from_table(s, su);
            return QuerySignature::IdBatchLookup { ids, table };
        }
    }

    // ── Simple scan: SELECT * ... LIMIT N (no WHERE/ORDER/GROUP/JOIN) ──
    if is_exact_star_select && has_limit && !has_where && !has_order && !has_group && !has_join {
        if let Some((limit, offset)) = extract_limit_offset_from_upper(su) {
            return QuerySignature::SimpleScanLimit {
                limit,
                offset,
                table: extract_from_table(s, su),
            };
        }
    }

    // ── String equality: SELECT * ... WHERE col = 'val' (no LIMIT/ORDER/GROUP/JOIN/BETWEEN/IN) ──
    if is_exact_star_select
        && has_where
        && !has_order
        && !has_group
        && !has_join
        && !su.contains("BETWEEN")
        && !su.contains(" IN ")
        && !su.contains('>')
        && !su.contains('<')
        && !su.contains(" LIKE ")
        && s.contains('\'')
    {
        if has_limit {
            if let Some((col, val, limit, offset)) = extract_string_equality_with_limit(s, su) {
                return QuerySignature::StringEqualityFilterLimit {
                    table: extract_from_table(s, su),
                    column: col,
                    value: val,
                    limit,
                    offset,
                };
            }
        }
        if let Some((col, val)) = extract_string_equality(s, su) {
            return QuerySignature::StringEqualityFilter {
                table: extract_from_table(s, su),
                column: col,
                value: val,
            };
        }
    }

    // ── Numeric equality: SELECT * ... WHERE col = N (simple, no AND/OR/IN/LIMIT) ──
    if is_exact_star_select
        && has_where
        && !has_order
        && !has_group
        && !has_join
        && !su.contains("BETWEEN")
        && !su.contains(" IN ")
        && !su.contains(" LIKE ")
        && !su.contains(" AND ")
        && !su.contains(" OR ")
        && !s.contains('\'')
        && !su.contains('>')
        && !su.contains('<')
    {
        if let Some((col, value)) = extract_numeric_equality(s, su) {
            return QuerySignature::NumericEqualityFilter {
                table: extract_from_table(s, su),
                column: col,
                value,
            };
        }
    }

    // ── Numeric IN: SELECT * ... WHERE col IN (N1, N2, ...) ──
    if is_exact_star_select
        && has_where
        && su.contains(" IN ")
        && !has_order
        && !has_group
        && !has_join
        && !su.contains("BETWEEN")
        && !su.contains(" LIKE ")
        && !su.contains(" AND ")
        && !su.contains(" OR ")
        && !s.contains('\'')
    {
        if let Some((col, values)) = extract_numeric_in_list(s, su) {
            return QuerySignature::NumericInFilter {
                table: extract_from_table(s, su),
                column: col,
                values,
            };
        }
    }

    // ── Numeric comparison + LIMIT: SELECT * ... WHERE col > 1 LIMIT N ──
    if is_exact_star_select
        && has_where
        && has_limit
        && !has_order
        && !has_group
        && !has_join
        && !su.contains("BETWEEN")
        && !su.contains(" IN ")
        && !su.contains(" LIKE ")
        && !su.contains(" AND ")
        && !su.contains(" OR ")
        && !s.contains('\'')
    {
        if let Some((column, low, high, limit, offset)) = extract_numeric_range_with_limit(s, su) {
            return QuerySignature::NumericRangeFilterLimit {
                table: extract_from_table(s, su),
                column,
                low,
                high,
                limit,
                offset,
            };
        }
    }

    // ── LIKE filter: SELECT * ... WHERE col LIKE 'pattern' (simple, no AND/OR/NOT) ──
    if is_exact_star_select
        && su.contains(" LIKE ")
        && has_where
        && !su.contains("NOT LIKE")
        && !has_limit
        && !has_order
        && !has_group
        && !has_join
        && !su.contains(" AND ")
        && !su.contains(" OR ")
        && s.contains('\'')
    {
        if let Some((col, pattern)) = extract_like_pattern(s, su) {
            return QuerySignature::LikeFilter {
                table: extract_from_table(s, su),
                column: col,
                pattern,
            };
        }
    }

    // ── Full scan: SELECT * FROM <table> (no WHERE/LIMIT/ORDER/GROUP/JOIN/DISTINCT) ──
    if is_exact_star_select
        && !has_where
        && !has_order
        && !has_group
        && !has_join
        && !has_limit
        && !has_distinct
    {
        let table = extract_from_table(s, su);
        return QuerySignature::FullScan { table };
    }

    // ── Filtered string equality aggregation: SELECT COUNT(*), AVG(col) ... FROM t WHERE str_col = 'val' ──
    if has_where
        && !has_group
        && !has_order
        && !has_limit
        && !has_join
        && !has_distinct
        && !has_having
    {
        let has_agg = su.contains("COUNT(")
            || su.contains("AVG(")
            || su.contains("SUM(")
            || su.contains("MIN(")
            || su.contains("MAX(");
        if has_agg {
            if let Some((col, val)) = extract_string_equality(s, su) {
                let table = extract_from_table(s, su);
                return QuerySignature::FilteredStringAgg {
                    table,
                    filter_column: col,
                    filter_value: val,
                };
            }
        }
    }

    QuerySignature::Complex
}

/// Extract the integer ID from a simple `WHERE _id = N` clause.
/// Returns None when the WHERE clause contains anything beyond the equality.
fn extract_simple_id_equality(sql: &str, su: &str) -> Option<u64> {
    let where_pos = su.find("WHERE")?;
    let after_where = sql[where_pos + 5..].trim().trim_end_matches(';').trim();
    let after_where_upper = after_where.to_ascii_uppercase();

    let id_prefix = if after_where_upper.starts_with("_ID =") {
        "_ID ="
    } else if after_where_upper.starts_with("_ID=") {
        "_ID="
    } else if after_where_upper.starts_with("\"_ID\" =") {
        "\"_ID\" ="
    } else if after_where_upper.starts_with("\"_ID\"=") {
        "\"_ID\"="
    } else {
        return None;
    };

    let rhs = after_where[id_prefix.len()..].trim_start();
    let num_end = rhs.find(|c: char| !c.is_ascii_digit()).unwrap_or(rhs.len());
    if num_end == 0 {
        return None;
    }
    let rest = rhs[num_end..].trim();
    if !rest.is_empty() {
        return None;
    }
    rhs[..num_end].parse::<u64>().ok()
}

/// Extract IDs from a simple `WHERE _id IN (...)` clause.
/// Returns None when the WHERE clause contains anything beyond the IN list.
fn extract_simple_id_in_list(sql: &str, su: &str) -> Option<Vec<u64>> {
    let where_pos = su.find("WHERE")?;
    let after_where = sql[where_pos + 5..].trim().trim_end_matches(';').trim();
    let after_where_upper = after_where.to_ascii_uppercase();

    let id_prefix = if after_where_upper.starts_with("_ID IN") {
        "_ID IN"
    } else if after_where_upper.starts_with("\"_ID\" IN") {
        "\"_ID\" IN"
    } else {
        return None;
    };

    let rhs = after_where[id_prefix.len()..].trim_start();
    if !rhs.starts_with('(') {
        return None;
    }
    let end_pos = rhs.find(')')?;
    let list = rhs[1..end_pos].trim();
    let rest = rhs[end_pos + 1..].trim();
    if !rest.is_empty() || list.is_empty() {
        return None;
    }

    let mut ids = Vec::new();
    for part in list.split(',') {
        let id = part.trim().parse::<u64>().ok()?;
        ids.push(id);
    }
    if ids.is_empty() {
        return None;
    }
    Some(ids)
}

/// Extract LIMIT value from uppercased SQL.
fn extract_limit_from_upper(su: &str) -> Option<usize> {
    let after_limit = su.rsplit("LIMIT").next()?;
    after_limit
        .trim()
        .trim_end_matches(';')
        .parse::<usize>()
        .ok()
}

/// Extract LIMIT/OFFSET values from uppercased SQL.
fn extract_limit_offset_from_upper(su: &str) -> Option<(usize, usize)> {
    let after_limit = su.rsplit("LIMIT").next()?;
    parse_limit_offset_clause(after_limit.trim().trim_end_matches(';'))
}

/// Extract a simple comma-separated projection list containing only plain column references.
/// Rejects `*`, expressions, aliases, and mixed `table.*` forms.
fn extract_simple_projection_columns(sql: &str, su: &str) -> Option<Vec<String>> {
    if !su.starts_with("SELECT") {
        return None;
    }
    let from_pos = su.find(" FROM ")?;
    let projection = sql["SELECT".len()..from_pos].trim();
    if projection.is_empty() || projection == "*" {
        return None;
    }

    let mut columns = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for raw_part in projection.split(',') {
        let raw = raw_part.trim();
        if raw.is_empty() {
            return None;
        }
        let raw_upper = raw.to_ascii_uppercase();
        if raw == "*"
            || raw.ends_with(".*")
            || raw.contains('(')
            || raw.contains(')')
            || raw.contains('+')
            || raw.contains('-')
            || raw.contains('/')
            || raw.contains('\'')
            || raw_upper.contains(" AS ")
            || raw.chars().any(|c| c.is_whitespace())
        {
            return None;
        }

        let normalized = raw
            .rsplit('.')
            .next()?
            .trim_matches(|c| c == '"' || c == '`');
        if normalized.is_empty() || normalized == "*" {
            return None;
        }
        if !seen.insert(normalized.to_string()) {
            return None;
        }
        columns.push(normalized.to_string());
    }

    if columns.is_empty() {
        None
    } else {
        Some(columns)
    }
}

/// Returns true only for exact `SELECT * FROM ...` projections.
/// Rejects forms like `SELECT *, _id ...`, `SELECT _id, * ...`, `SELECT t.* ...`.
fn has_exact_star_projection(sql: &str, su: &str) -> bool {
    if !su.starts_with("SELECT") {
        return false;
    }
    let from_pos = match su.find(" FROM ") {
        Some(pos) => pos,
        None => return false,
    };
    let projection = sql["SELECT".len()..from_pos].trim();
    projection == "*"
}

/// Extract (column, value) from `WHERE col = 'val'` in original-case SQL,
/// guided by the uppercased version for keyword positions.
fn extract_string_equality(sql: &str, su: &str) -> Option<(String, String)> {
    let where_pos = su.find("WHERE")?;
    let after_where = sql[where_pos + 5..].trim().trim_end_matches(';');
    parse_string_equality_clause(after_where)
}

/// Extract (column, value, limit, offset) from
/// `WHERE col = 'val' LIMIT N [OFFSET M]`.
fn extract_string_equality_with_limit(
    sql: &str,
    su: &str,
) -> Option<(String, String, usize, usize)> {
    let where_pos = su.find("WHERE")?;
    let limit_pos = su.rfind("LIMIT")?;
    if limit_pos <= where_pos {
        return None;
    }

    let where_clause = sql[where_pos + 5..limit_pos].trim();
    let (column, value) = parse_string_equality_clause(where_clause)?;
    let after_limit = sql[limit_pos + "LIMIT".len()..]
        .trim()
        .trim_end_matches(';')
        .trim();
    let (limit, offset) = parse_limit_offset_clause(after_limit)?;
    Some((column, value, limit, offset))
}

/// Extract (column, inclusive-low, inclusive-high, limit, offset) from
/// `WHERE numeric_col <op> numeric_literal LIMIT N [OFFSET M]`.
fn extract_numeric_range_with_limit(
    sql: &str,
    su: &str,
) -> Option<(String, f64, f64, usize, usize)> {
    let where_pos = su.find("WHERE")?;
    let limit_pos = su.rfind("LIMIT")?;
    if limit_pos <= where_pos {
        return None;
    }

    let where_clause = sql[where_pos + 5..limit_pos].trim();
    let (column, low, high) = parse_numeric_comparison_clause(where_clause)?;
    let after_limit = sql[limit_pos + "LIMIT".len()..]
        .trim()
        .trim_end_matches(';')
        .trim();
    let (limit, offset) = parse_limit_offset_clause(after_limit)?;
    Some((column, low, high, limit, offset))
}

fn parse_string_equality_clause(clause: &str) -> Option<(String, String)> {
    let after_where = clause.trim();
    let eq_pos = after_where.find('=')?;
    let col = after_where[..eq_pos].trim().trim_matches('"').to_string();
    if col.contains(' ') || col.contains('(') {
        return None;
    }
    let rhs = after_where[eq_pos + 1..].trim();
    if !rhs.starts_with('\'') {
        return None;
    }
    let val_end = rhs[1..].find('\'')?;
    let val = rhs[1..1 + val_end].to_string();
    let rest = rhs[1 + val_end + 1..].trim();
    if !rest.is_empty() {
        return None;
    }
    Some((col, val))
}

fn parse_numeric_comparison_clause(clause: &str) -> Option<(String, f64, f64)> {
    let clause = clause.trim();
    let (op, op_pos) = [">=", "<=", ">", "<", "="]
        .iter()
        .find_map(|op| clause.find(op).map(|pos| (*op, pos)))?;
    let col = trim_column_ident(clause[..op_pos].trim());
    if col.is_empty() || col.contains(' ') || col.contains('(') {
        return None;
    }
    let rhs = clause[op_pos + op.len()..].trim();
    let value = rhs.parse::<f64>().ok()?;
    let (low, high) = match op {
        "=" => (value, value),
        ">" => (next_up_f64(value), f64::INFINITY),
        ">=" => (value, f64::INFINITY),
        "<" => (f64::NEG_INFINITY, next_down_f64(value)),
        "<=" => (f64::NEG_INFINITY, value),
        _ => return None,
    };
    Some((col, low, high))
}

fn trim_column_ident(name: &str) -> String {
    let name = name.trim();
    if name.starts_with('`') && name.ends_with('`') && name.len() >= 2 {
        return name[1..name.len() - 1].to_string();
    }
    name.trim_matches('"').to_string()
}

/// Extract `WHERE col = N` when the clause is a single numeric equality.
fn extract_numeric_equality(sql: &str, su: &str) -> Option<(String, i64)> {
    let where_pos = su.find("WHERE")?;
    let after_where = sql[where_pos + 5..].trim().trim_end_matches(';').trim();
    let after_upper = after_where.to_ascii_uppercase();
    if after_upper.contains(" AND ") || after_upper.contains(" OR ") {
        return None;
    }
    let (column, low, high) = parse_numeric_comparison_clause(after_where)?;
    if (low - high).abs() > f64::EPSILON {
        return None;
    }
    if !low.is_finite() || low.fract().abs() > f64::EPSILON {
        return None;
    }
    Some((column, low as i64))
}

/// Extract `WHERE col IN (N1, N2, ...)` for integer lists.
fn extract_numeric_in_list(sql: &str, su: &str) -> Option<(String, Vec<i64>)> {
    let where_pos = su.find("WHERE")?;
    let after_where = sql[where_pos + 5..].trim().trim_end_matches(';').trim();
    let after_upper = after_where.to_ascii_uppercase();
    if after_upper.contains(" AND ") || after_upper.contains(" OR ") {
        return None;
    }
    let in_pos = after_upper.find(" IN ")?;
    let col_part = after_where[..in_pos].trim();
    let column = trim_column_ident(col_part);
    if column.is_empty() {
        return None;
    }
    let rhs = after_where[in_pos + 4..].trim();
    if !rhs.starts_with('(') {
        return None;
    }
    let end_pos = rhs.find(')')?;
    let list = rhs[1..end_pos].trim();
    let rest = rhs[end_pos + 1..].trim();
    if !rest.is_empty() || list.is_empty() {
        return None;
    }
    let mut values = Vec::new();
    for part in list.split(',') {
        let token = part.trim();
        let value = token.parse::<i64>().ok()?;
        values.push(value);
    }
    if values.is_empty() {
        return None;
    }
    Some((column, values))
}

#[inline]
fn next_up_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

#[inline]
fn next_down_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

fn parse_limit_offset_clause(clause: &str) -> Option<(usize, usize)> {
    let mut parts = clause.split_whitespace();
    let limit = parts.next()?.parse::<usize>().ok()?;
    let offset = match parts.next() {
        None => 0,
        Some(keyword) if keyword.eq_ignore_ascii_case("OFFSET") => {
            let parsed = parts.next()?.parse::<usize>().ok()?;
            if parts.next().is_some() {
                return None;
            }
            parsed
        }
        Some(_) => return None,
    };
    Some((limit, offset))
}

/// Extract table name from `FROM <table>` clause using original-case SQL + uppercased version.
fn extract_from_table(sql: &str, su: &str) -> Option<String> {
    let fp = su.find(" FROM ")?;
    let after_from = sql[fp + 6..].trim_start();
    let tn_end = after_from
        .find(|c: char| c.is_whitespace() || c == ';')
        .unwrap_or(after_from.len());
    let tname = after_from[..tn_end].trim_matches('"').to_lowercase();
    if tname.is_empty() {
        None
    } else {
        Some(tname)
    }
}

/// Extract DDL sub-kind from original-case SQL + uppercased version.
/// Extracts table name for CREATE TABLE / DROP TABLE; returns Other for everything else.
fn extract_ddl_kind(sql: &str, su: &str) -> DdlKind {
    if su.starts_with("CREATE TABLE")
        || su.starts_with("CREATE TEMP TABLE")
        || su.starts_with("CREATE TEMPORARY TABLE")
    {
        // Locate TABLE keyword in uppercased version to find offset in original SQL
        let table_pos = su.find("TABLE").unwrap();
        let rest = &sql[table_pos + "TABLE".len()..].trim_start();
        // Skip "IF NOT EXISTS"
        let rest = if rest.len() >= 13 && rest[..13].eq_ignore_ascii_case("IF NOT EXISTS") {
            rest[13..].trim_start()
        } else {
            rest
        };
        if let Some(name) = rest
            .split(|c: char| c.is_whitespace() || c == '(' || c == ';')
            .next()
        {
            let tbl = name
                .trim_matches(|c: char| c == '"' || c == '\'' || c == '`')
                .to_lowercase();
            if !tbl.is_empty() {
                return DdlKind::CreateTable { name: tbl };
            }
        }
    } else if su.starts_with("DROP TABLE") {
        let rest = &sql["DROP TABLE".len()..].trim_start();
        // Skip "IF EXISTS"
        let rest = if rest.len() >= 9 && rest[..9].eq_ignore_ascii_case("IF EXISTS") {
            rest[9..].trim_start()
        } else {
            rest
        };
        if let Some(name) = rest.split(|c: char| c.is_whitespace() || c == ';').next() {
            let tbl = name
                .trim_matches(|c: char| c == '"' || c == '\'' || c == '`')
                .to_lowercase();
            if !tbl.is_empty() {
                return DdlKind::DropTable { name: tbl };
            }
        }
    }
    DdlKind::Other
}

/// Extract (column, pattern) from `WHERE col LIKE 'pattern'`.
fn extract_like_pattern(sql: &str, su: &str) -> Option<(String, String)> {
    let where_pos = su.find("WHERE")?;
    let after_where = sql[where_pos + 5..].trim().trim_end_matches(';');
    let after_where_upper = after_where.to_uppercase();
    let like_pos = after_where_upper.find(" LIKE ")?;
    let col = after_where[..like_pos].trim().trim_matches('"').to_string();
    if col.contains(' ') || col.contains('(') {
        return None;
    }
    let rhs = after_where[like_pos + 6..].trim();
    if !rhs.starts_with('\'') {
        return None;
    }
    let val_end = rhs[1..].find('\'')?;
    let pattern = rhs[1..1 + val_end].to_string();
    let rest = rhs[1 + val_end + 1..].trim();
    if !rest.is_empty() {
        return None;
    }
    Some((col, pattern))
}

fn contains_unquoted_keyword(sql: &str, keyword: &str) -> bool {
    let bytes = sql.as_bytes();
    let keyword_bytes = keyword.as_bytes();
    let kw_len = keyword_bytes.len();
    let mut i = 0usize;
    let mut in_single_quote = false;
    let mut in_backtick = false;

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'`' && !in_single_quote {
            in_backtick = !in_backtick;
            i += 1;
            continue;
        }
        if b == b'\'' && !in_backtick {
            if in_single_quote && i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                i += 2;
                continue;
            }
            in_single_quote = !in_single_quote;
            i += 1;
            continue;
        }

        if !in_single_quote
            && !in_backtick
            && i + kw_len <= bytes.len()
            && bytes[i..i + kw_len].eq_ignore_ascii_case(keyword_bytes)
        {
            let prev_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let next_ok = i + kw_len == bytes.len() || !is_ident_byte(bytes[i + kw_len]);
            if prev_ok && next_ok {
                return true;
            }
        }

        i += 1;
    }

    false
}

#[inline]
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_star() {
        assert_eq!(
            classify("SELECT COUNT(*) FROM users"),
            QuerySignature::CountStar {
                table: "users".to_string()
            }
        );
        // With WHERE — should be Complex
        assert_eq!(
            classify("SELECT COUNT(*) FROM users WHERE age > 10"),
            QuerySignature::Complex
        );
    }

    #[test]
    fn test_point_lookup() {
        assert_eq!(
            classify("SELECT * FROM t WHERE _id = 42"),
            QuerySignature::PointLookup {
                id: 42,
                table: Some("t".to_string())
            }
        );
        assert_eq!(
            classify("SELECT * FROM t WHERE _id=100"),
            QuerySignature::PointLookup {
                id: 100,
                table: Some("t".to_string())
            }
        );
        assert_eq!(
            classify("SELECT name FROM t WHERE _id = 42"),
            QuerySignature::ProjectedPointLookup {
                id: 42,
                table: Some("t".to_string()),
                columns: vec!["name".to_string()],
            }
        );
        assert_eq!(
            classify("SELECT * FROM t WHERE _id = 42 AND age = 1"),
            QuerySignature::Complex
        );
        assert_eq!(
            classify("SELECT name, age FROM t WHERE _id = 42"),
            QuerySignature::ProjectedPointLookup {
                id: 42,
                table: Some("t".to_string()),
                columns: vec!["name".to_string(), "age".to_string()],
            }
        );
    }

    #[test]
    fn test_id_batch_lookup() {
        assert_eq!(
            classify("SELECT * FROM t WHERE _id IN (1, 5, 9)"),
            QuerySignature::IdBatchLookup {
                ids: vec![1, 5, 9],
                table: Some("t".to_string()),
            }
        );
        assert_eq!(
            classify("SELECT * FROM t WHERE _id IN (1, 5, 9) AND age > 1"),
            QuerySignature::Complex
        );
        assert_eq!(
            classify("SELECT name FROM t WHERE _id IN (1, 5, 9)"),
            QuerySignature::ProjectedIdBatchLookup {
                ids: vec![1, 5, 9],
                table: Some("t".to_string()),
                columns: vec!["name".to_string()],
            }
        );
    }

    #[test]
    fn test_simple_scan_limit() {
        assert_eq!(
            classify("SELECT * FROM t LIMIT 100"),
            QuerySignature::SimpleScanLimit {
                limit: 100,
                offset: 0,
                table: Some("t".to_string()),
            }
        );
        // With WHERE — not a simple scan
        assert_eq!(
            classify("SELECT * FROM t WHERE x > 1 LIMIT 100"),
            QuerySignature::NumericRangeFilterLimit {
                limit: 100,
                offset: 0,
                table: Some("t".to_string()),
                column: "x".to_string(),
                low: next_up_f64(1.0),
                high: f64::INFINITY,
            }
        );
        assert_eq!(
            classify("SELECT name, age FROM t LIMIT 100"),
            QuerySignature::ProjectedScanLimit {
                limit: 100,
                offset: 0,
                table: Some("t".to_string()),
                columns: vec!["name".to_string(), "age".to_string()],
            }
        );
        assert_eq!(
            classify("SELECT name, age FROM t LIMIT 100 OFFSET 20"),
            QuerySignature::ProjectedScanLimit {
                limit: 100,
                offset: 20,
                table: Some("t".to_string()),
                columns: vec!["name".to_string(), "age".to_string()],
            }
        );
    }

    #[test]
    fn test_numeric_range_filter_limit() {
        assert_eq!(
            classify("SELECT * FROM t WHERE age > 30 LIMIT 100 OFFSET 5"),
            QuerySignature::NumericRangeFilterLimit {
                table: Some("t".to_string()),
                column: "age".to_string(),
                low: next_up_f64(30.0),
                high: f64::INFINITY,
                limit: 100,
                offset: 5,
            }
        );
        assert_eq!(
            classify("SELECT name FROM t WHERE score <= 50.5 LIMIT 10"),
            QuerySignature::ProjectedNumericRangeFilterLimit {
                table: Some("t".to_string()),
                columns: vec!["name".to_string()],
                column: "score".to_string(),
                low: f64::NEG_INFINITY,
                high: 50.5,
                limit: 10,
                offset: 0,
            }
        );
    }

    #[test]
    fn test_full_scan() {
        assert_eq!(
            classify("SELECT * FROM t"),
            QuerySignature::FullScan {
                table: Some("t".to_string())
            }
        );
        assert_eq!(classify("SELECT *, _id FROM t"), QuerySignature::Complex);
        assert_eq!(
            classify("SELECT name, age FROM t"),
            QuerySignature::ProjectedFullScan {
                table: Some("t".to_string()),
                columns: vec!["name".to_string(), "age".to_string()],
            }
        );
        assert_eq!(
            classify("SELECT name FROM t UNION ALL SELECT name FROM t"),
            QuerySignature::Complex
        );
    }

    #[test]
    fn test_string_equality() {
        assert_eq!(
            classify("SELECT * FROM t WHERE city = 'NYC'"),
            QuerySignature::StringEqualityFilter {
                table: Some("t".to_string()),
                column: "city".to_string(),
                value: "NYC".to_string(),
            }
        );
        assert_eq!(
            classify("SELECT * FROM t WHERE city = 'NYC' AND age = 20"),
            QuerySignature::Complex
        );
        assert_eq!(
            classify("SELECT name FROM t WHERE city = 'NYC'"),
            QuerySignature::ProjectedStringEqualityFilter {
                table: Some("t".to_string()),
                columns: vec!["name".to_string()],
                column: "city".to_string(),
                value: "NYC".to_string(),
            }
        );
        assert_eq!(
            classify("SELECT * FROM t WHERE city = 'NYC' LIMIT 1"),
            QuerySignature::StringEqualityFilterLimit {
                table: Some("t".to_string()),
                column: "city".to_string(),
                value: "NYC".to_string(),
                limit: 1,
                offset: 0,
            }
        );
        assert_eq!(
            classify("SELECT name FROM t WHERE city = 'NYC' LIMIT 5 OFFSET 2"),
            QuerySignature::ProjectedStringEqualityFilterLimit {
                table: Some("t".to_string()),
                columns: vec!["name".to_string()],
                column: "city".to_string(),
                value: "NYC".to_string(),
                limit: 5,
                offset: 2,
            }
        );
    }

    #[test]
    fn test_numeric_equality_and_in() {
        assert_eq!(
            classify("SELECT * FROM users WHERE `order` = 1"),
            QuerySignature::NumericEqualityFilter {
                table: Some("users".to_string()),
                column: "order".to_string(),
                value: 1,
            }
        );
        assert_eq!(
            classify("SELECT * FROM users WHERE `order` IN (1, 2)"),
            QuerySignature::NumericInFilter {
                table: Some("users".to_string()),
                column: "order".to_string(),
                values: vec![1, 2],
            }
        );
        assert_eq!(
            classify("SELECT * FROM users WHERE `order` = 1 OR `order` = 2"),
            QuerySignature::Complex
        );
    }

    #[test]
    fn test_like_filter() {
        assert_eq!(
            classify("SELECT * FROM t WHERE name LIKE '%smith%'"),
            QuerySignature::LikeFilter {
                table: Some("t".to_string()),
                column: "name".to_string(),
                pattern: "%smith%".to_string(),
            }
        );
    }

    #[test]
    fn test_table_function() {
        assert_eq!(
            classify("SELECT * FROM read_csv('/tmp/data.csv')"),
            QuerySignature::TableFunction
        );
    }

    #[test]
    fn test_ddl() {
        assert_eq!(
            classify("CREATE TABLE t (id INT)"),
            QuerySignature::Ddl {
                kind: DdlKind::CreateTable {
                    name: "t".to_string()
                }
            }
        );
        assert_eq!(
            classify("CREATE TEMP TABLE t (id INT)"),
            QuerySignature::Ddl {
                kind: DdlKind::CreateTable {
                    name: "t".to_string()
                }
            }
        );
        assert_eq!(
            classify("CREATE TEMPORARY TABLE t (id INT)"),
            QuerySignature::Ddl {
                kind: DdlKind::CreateTable {
                    name: "t".to_string()
                }
            }
        );
        assert_eq!(
            classify("DROP TABLE t"),
            QuerySignature::Ddl {
                kind: DdlKind::DropTable {
                    name: "t".to_string()
                }
            }
        );
        assert!(matches!(
            classify("ALTER TABLE t ADD COLUMN x INT"),
            QuerySignature::Ddl {
                kind: DdlKind::Other
            }
        ));
    }

    #[test]
    fn test_dml_write() {
        assert_eq!(
            classify("INSERT INTO t VALUES (1)"),
            QuerySignature::DmlWrite
        );
        assert_eq!(
            classify("DELETE FROM t WHERE id = 1"),
            QuerySignature::DmlWrite
        );
        assert_eq!(classify("UPDATE t SET x = 1"), QuerySignature::DmlWrite);
    }

    #[test]
    fn test_transaction() {
        assert_eq!(classify("BEGIN"), QuerySignature::Transaction);
        assert_eq!(classify("COMMIT"), QuerySignature::Transaction);
        assert_eq!(classify("ROLLBACK"), QuerySignature::Transaction);
        assert_eq!(classify("SAVEPOINT sp1"), QuerySignature::Transaction);
    }

    #[test]
    fn test_multi_statement() {
        assert_eq!(
            classify("INSERT INTO t VALUES (1); SELECT * FROM t"),
            QuerySignature::MultiStatement
        );
    }

    #[test]
    fn test_cte() {
        assert_eq!(
            classify("WITH cte AS (SELECT 1) SELECT * FROM cte"),
            QuerySignature::Cte
        );
    }

    #[test]
    fn test_complex() {
        assert_eq!(
            classify("SELECT a, b FROM t WHERE x > 1 ORDER BY a LIMIT 10"),
            QuerySignature::Complex
        );
        assert_eq!(
            classify("SELECT * FROM t JOIN u ON t.id = u.id"),
            QuerySignature::Complex
        );
    }

    #[test]
    fn test_session_command() {
        assert_eq!(classify("SET x = 1"), QuerySignature::SessionCommand);
        assert_eq!(classify("RESET x"), QuerySignature::SessionCommand);
    }
}
