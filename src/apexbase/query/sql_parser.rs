//! SQL:2023 Parser for ApexBase
//!
//! Supports standard SQL SELECT statements with:
//! - SELECT columns or SELECT *
//! - FROM table
//! - WHERE conditions (with LIKE, IN, AND, OR, NOT, comparison operators)
//! - ORDER BY column [ASC|DESC]
//! - LIMIT n [OFFSET m]
//! - DISTINCT
//! - Column aliases (AS)
//! - Aggregate functions (COUNT, SUM, AVG, MIN, MAX)
//! - GROUP BY / HAVING

use crate::data::DataType;
use crate::data::Value;
use crate::ApexError;
use serde::{Deserialize, Serialize};

/// Column-level constraint kinds
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColumnConstraintKind {
    NotNull,
    PrimaryKey,
    Unique,
    Default(Value),
    Check(String),
    ForeignKey {
        ref_table: String,
        ref_column: String,
    },
    Autoincrement,
}

/// Column definition for CREATE TABLE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub constraints: Vec<ColumnConstraintKind>,
}

/// ALTER TABLE operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlterTableOp {
    AddColumn { name: String, data_type: DataType },
    DropColumn { name: String },
    RenameColumn { old_name: String, new_name: String },
}

/// SQL Statement types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SqlStatement {
    Select(SelectStatement),
    Union(UnionStatement),
    CreateView {
        name: String,
        stmt: SelectStatement,
    },
    DropView {
        name: String,
    },
    // DDL Statements
    CreateTable {
        table: String,
        columns: Vec<ColumnDef>,
        if_not_exists: bool,
        temp: bool,
    },
    DropTable {
        table: String,
        if_exists: bool,
    },
    AlterTable {
        table: String,
        operation: AlterTableOp,
    },
    TruncateTable {
        table: String,
    },
    // Index Statements
    CreateIndex {
        name: String,
        table: String,
        columns: Vec<String>,
        unique: bool,
        index_type: Option<String>,
        if_not_exists: bool,
    },
    DropIndex {
        name: String,
        table: String,
        if_exists: bool,
    },
    // DML Statements
    Insert {
        table: String,
        columns: Option<Vec<String>>,
        values: Vec<Vec<Value>>,
    },
    InsertOnConflict {
        table: String,
        columns: Option<Vec<String>>,
        values: Vec<Vec<Value>>,
        conflict_columns: Vec<String>,
        do_update: Option<Vec<(String, SqlExpr)>>,
    },
    InsertSelect {
        table: String,
        columns: Option<Vec<String>>,
        query: Box<SqlStatement>,
    },
    CreateTableAs {
        table: String,
        query: Box<SqlStatement>,
        if_not_exists: bool,
        temp: bool,
    },
    Explain {
        stmt: Box<SqlStatement>,
        analyze: bool,
    },
    Delete {
        table: String,
        where_clause: Option<SqlExpr>,
    },
    Update {
        table: String,
        assignments: Vec<(String, SqlExpr)>,
        where_clause: Option<SqlExpr>,
    },
    // CTE wrapper
    Cte {
        name: String,
        column_aliases: Vec<String>,
        body: Box<SqlStatement>,
        main: Box<SqlStatement>,
        recursive: bool,
    },
    // Transaction Statements
    BeginTransaction {
        read_only: bool,
    },
    Commit,
    Rollback,
    Savepoint {
        name: String,
    },
    ReleaseSavepoint {
        name: String,
    },
    RollbackToSavepoint {
        name: String,
    },
    AnalyzeTable {
        table: String,
    },
    CopyToParquet {
        table: String,
        file_path: String,
    },
    CopyExport {
        table: String,
        file_path: String,
        format: String,
        options: Vec<(String, String)>,
    },
    CopyFromParquet {
        table: String,
        file_path: String,
    },
    CopyImport {
        table: String,
        file_path: String,
        format: String,
        options: Vec<(String, String)>,
    },
    SetVariable {
        name: String,
        value: Value,
    },
    ResetVariable {
        name: String,
    },
    Reindex {
        table: String,
    },
    Pragma {
        name: String,
        arg: Option<String>,
    },
    // FTS DDL Statements
    CreateFtsIndex {
        table: String,
        fields: Option<Vec<String>>,
        lazy_load: bool,
        cache_size: usize,
    },
    DropFtsIndex {
        table: String,
    },
    AlterFtsIndexDisable {
        table: String,
    },
    AlterFtsIndexEnable {
        table: String,
    },
    ShowFtsIndexes,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SetOpType {
    Union,
    Intersect,
    Except,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnionStatement {
    pub left: Box<SqlStatement>,
    pub right: Box<SqlStatement>,
    pub all: bool,
    pub set_op: SetOpType,
    pub order_by: Vec<OrderByClause>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FromItem {
    Table {
        table: String,
        alias: Option<String>,
    },
    Subquery {
        stmt: Box<SqlStatement>,
        alias: String,
    },
    TableFunction {
        func: String,
        file: String,
        options: Vec<(String, String)>,
        alias: Option<String>,
    },
    /// `TOPK_DISTANCE(col, [q1,q2,...], k, 'metric')` — heap-based vector TopK.
    TopkDistance {
        col: String,
        query: Vec<f64>,
        k: usize,
        metric: String,
        alias: Option<String>,
    },
    /// DuckDB-style direct file reading: `SELECT * FROM 'path/file.parquet'`
    DirectFile { file: String, alias: Option<String> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinClause {
    pub join_type: JoinType,
    pub right: FromItem,
    pub on: SqlExpr,
}

/// SELECT statement structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectStatement {
    pub distinct: bool,
    pub distinct_on: Option<Vec<String>>,
    pub columns: Vec<SelectColumn>,
    pub from: Option<FromItem>,
    pub joins: Vec<JoinClause>,
    pub where_clause: Option<SqlExpr>,
    pub group_by: Vec<String>,
    pub group_by_exprs: Vec<Option<SqlExpr>>,
    pub having: Option<SqlExpr>,
    pub order_by: Vec<OrderByClause>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Column selection in SELECT clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectColumn {
    /// SELECT *
    All,
    /// SELECT * EXCLUDE (col1, col2, ...)
    AllExclude(Vec<String>),
    /// SELECT * REPLACE (expr AS col, ...)
    AllReplace(Vec<(SqlExpr, String)>),
    /// SELECT COLUMNS('regex')
    Columns(String),
    /// SELECT column_name
    Column(String),
    /// SELECT column_name AS alias
    ColumnAlias { column: String, alias: String },
    /// SELECT COUNT(*), SUM(col), etc.
    Aggregate {
        func: AggregateFunc,
        column: Option<String>,
        distinct: bool,
        alias: Option<String>,
    },
    /// SELECT expression AS alias
    Expression {
        expr: SqlExpr,
        alias: Option<String>,
    },
    /// SELECT row_number() OVER (PARTITION BY ... ORDER BY ...) AS alias
    /// Also supports LAG(col, offset, default) and LEAD(col, offset, default)
    WindowFunction {
        name: String,
        args: Vec<String>, // Function arguments (column, offset, default)
        partition_by: Vec<String>,
        order_by: Vec<OrderByClause>,
        alias: Option<String>,
    },
}

/// Aggregate functions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregateFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

impl std::fmt::Display for AggregateFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregateFunc::Count => write!(f, "COUNT"),
            AggregateFunc::Sum => write!(f, "SUM"),
            AggregateFunc::Avg => write!(f, "AVG"),
            AggregateFunc::Min => write!(f, "MIN"),
            AggregateFunc::Max => write!(f, "MAX"),
        }
    }
}

/// ORDER BY clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderByClause {
    pub column: String,
    pub descending: bool,
    pub nulls_first: Option<bool>, // SQL:2023 NULLS FIRST/LAST
    /// Optional SQL expression (for ORDER BY func_call(...) syntax).
    /// When set, the executor will evaluate this expression if `column` is
    /// not found as a pre-existing column in the result batch.
    pub expr: Option<SqlExpr>,
}

/// SQL Expression (for WHERE, HAVING, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SqlExpr {
    /// Column reference
    Column(String),
    /// Literal value
    Literal(Value),
    /// Binary operation: expr op expr
    BinaryOp {
        left: Box<SqlExpr>,
        op: BinaryOperator,
        right: Box<SqlExpr>,
    },
    /// Unary operation: NOT expr
    UnaryOp {
        op: UnaryOperator,
        expr: Box<SqlExpr>,
    },
    /// LIKE pattern matching
    Like {
        column: String,
        pattern: String,
        negated: bool,
    },
    /// REGEXP pattern matching
    Regexp {
        column: String,
        pattern: String,
        negated: bool,
    },
    /// IN list: column IN (v1, v2, ...)
    In {
        column: String,
        values: Vec<Value>,
        negated: bool,
    },
    /// IN subquery: column IN (SELECT ...)
    InSubquery {
        column: String,
        stmt: Box<SelectStatement>,
        negated: bool,
    },
    /// EXISTS subquery: EXISTS (SELECT ...)
    ExistsSubquery { stmt: Box<SelectStatement> },
    /// Scalar subquery: (SELECT ...)
    ScalarSubquery { stmt: Box<SelectStatement> },
    /// CASE WHEN <cond> THEN <expr> [WHEN <cond> THEN <expr>]* [ELSE <expr>] END
    Case {
        when_then: Vec<(SqlExpr, SqlExpr)>,
        else_expr: Option<Box<SqlExpr>>,
    },
    /// BETWEEN: column BETWEEN low AND high
    Between {
        column: String,
        low: Box<SqlExpr>,
        high: Box<SqlExpr>,
        negated: bool,
    },
    /// IS NULL / IS NOT NULL
    IsNull { column: String, negated: bool },
    /// FTS: MATCH('query') or FUZZY_MATCH('query') in WHERE clause
    FtsMatch { query: String, fuzzy: bool },
    /// Function call
    Function { name: String, args: Vec<SqlExpr> },
    /// Session variable reference: $varname
    Variable(String),
    /// CAST(expr AS TYPE)
    Cast {
        expr: Box<SqlExpr>,
        data_type: DataType,
    },
    /// Parenthesized expression
    Paren(Box<SqlExpr>),
    /// Array index: expr[index]
    ArrayIndex {
        array: Box<SqlExpr>,
        index: Box<SqlExpr>,
    },
    /// Array literal: [1.0, 2.0, 3.0]  (used for vector distance queries)
    ArrayLiteral(Vec<f64>),
    /// topk_distance(col, [q], k, 'metric') — whole-column TopK used inside explode_rename
    TopkDistance {
        col: String,
        query: Vec<f64>,
        k: usize,
        metric: String,
    },
    /// explode_rename(topk_distance_expr, "name1", "name2") — expands TopK pairs into k rows
    ExplodeRename {
        inner: Box<SqlExpr>,
        names: Vec<String>,
    },
}

/// Binary operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BinaryOperator {
    // Comparison
    Eq,    // =
    NotEq, // != or <>
    Lt,    // <
    Le,    // <=
    Gt,    // >
    Ge,    // >=
    // Logical
    And,
    Or,
    // Arithmetic (for expressions)
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// Unary operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UnaryOperator {
    Not,
    Minus,
}

// ============================================================================
// Column Extraction for On-Demand Reading
// ============================================================================

impl SelectStatement {
    /// Extract columns needed only for WHERE clause evaluation
    /// Used for late materialization optimization
    pub fn where_columns(&self) -> Vec<String> {
        let mut columns = Vec::new();
        if let Some(ref expr) = self.where_clause {
            Self::extract_columns_from_expr(expr, &mut columns);
        }
        columns.sort();
        columns.dedup();
        columns
    }

    /// Check if this query uses SELECT * (needs all columns)
    pub fn is_select_star(&self) -> bool {
        self.columns.iter().any(|col| {
            matches!(
                col,
                SelectColumn::All
                    | SelectColumn::AllExclude(..)
                    | SelectColumn::AllReplace(..)
                    | SelectColumn::Columns(..)
            )
        })
    }

    /// Check if this is a pure SELECT * (no EXCLUDE/REPLACE/COLUMNS)
    pub fn is_pure_star(&self) -> bool {
        let mut has_all = false;
        for col in &self.columns {
            match col {
                SelectColumn::All => has_all = true,
                SelectColumn::AllExclude(..)
                | SelectColumn::AllReplace(..)
                | SelectColumn::Columns(..) => return false,
                _ => {}
            }
        }
        has_all
    }

    /// Extract all column names required by this SELECT statement
    /// Returns None if SELECT * is used (meaning all columns needed)
    pub fn required_columns(&self) -> Option<Vec<String>> {
        let mut columns = Vec::new();
        let mut has_star = false;
        let mut has_explicit_id = false;

        // Extract from SELECT clause
        for col in &self.columns {
            match col {
                SelectColumn::All
                | SelectColumn::AllExclude(..)
                | SelectColumn::AllReplace(..)
                | SelectColumn::Columns(..) => {
                    has_star = true;
                }
                SelectColumn::Column(name) => {
                    // Strip table prefix if present (e.g., "default._id" -> "_id")
                    let actual_name = if let Some(dot_pos) = name.rfind('.') {
                        &name[dot_pos + 1..]
                    } else {
                        name.as_str()
                    };
                    if actual_name == "_id" {
                        has_explicit_id = true;
                    } else {
                        columns.push(actual_name.to_string());
                    }
                }
                SelectColumn::ColumnAlias { column, .. } => {
                    if column == "_id" {
                        has_explicit_id = true;
                    } else {
                        columns.push(column.clone());
                    }
                }
                SelectColumn::Aggregate { column, .. } => {
                    if let Some(col) = column {
                        if col == "_id" {
                            has_explicit_id = true;
                        } else if col != "*"
                            && !col
                                .chars()
                                .next()
                                .map(|c| c.is_ascii_digit())
                                .unwrap_or(false)
                        {
                            // Skip constants like "1", "2" and "*" - only add real column names
                            columns.push(col.clone());
                        }
                    }
                }
                SelectColumn::Expression { expr, .. } => {
                    Self::extract_columns_from_expr(expr, &mut columns);
                }
                SelectColumn::WindowFunction {
                    args,
                    partition_by,
                    order_by,
                    ..
                } => {
                    // Add columns from args (for LAG/LEAD)
                    for arg in args {
                        if !arg.starts_with("Int") && !arg.starts_with("Float") && arg != "_id" {
                            columns.push(arg.clone());
                        }
                    }
                    for col in partition_by {
                        if col != "_id" {
                            columns.push(col.clone());
                        }
                    }
                    for ob in order_by {
                        if ob.column != "_id" {
                            columns.push(ob.column.clone());
                        }
                    }
                }
            }
        }

        // Extract from WHERE clause
        if let Some(ref expr) = self.where_clause {
            Self::extract_columns_from_expr(expr, &mut columns);
        }

        // Extract from ORDER BY
        for ob in &self.order_by {
            if let Some(ref expr) = ob.expr {
                // Expression ORDER BY (e.g. ORDER BY array_distance(col, [...])):
                // extract all column references from the expression.
                Self::extract_columns_from_expr(expr, &mut columns);
            } else if ob.column != "_id" {
                columns.push(ob.column.clone());
            }
        }

        // Extract from GROUP BY (including expression column refs)
        for (i, col) in self.group_by.iter().enumerate() {
            if let Some(Some(expr)) = self.group_by_exprs.get(i) {
                Self::extract_columns_from_expr(expr, &mut columns);
            } else if col != "_id" {
                columns.push(col.clone());
            }
        }

        // Extract from HAVING
        if let Some(ref expr) = self.having {
            Self::extract_columns_from_expr(expr, &mut columns);
        }

        if has_star {
            None // SELECT * means all columns
        } else {
            // Include _id if explicitly requested
            if has_explicit_id {
                columns.push("_id".to_string());
            }
            // Deduplicate
            columns.sort();
            columns.dedup();
            // If no columns needed (e.g., COUNT(*)), return None to get all columns
            // so the batch has correct row count for aggregation
            if columns.is_empty() {
                None
            } else {
                Some(columns)
            }
        }
    }

    fn extract_columns_from_expr(expr: &SqlExpr, columns: &mut Vec<String>) {
        match expr {
            SqlExpr::Column(name) => {
                // Strip table prefix if present (e.g., "o.user_id" -> "user_id")
                let actual_name = if let Some(dot_pos) = name.rfind('.') {
                    &name[dot_pos + 1..]
                } else {
                    name.as_str()
                };
                columns.push(actual_name.to_string());
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                Self::extract_columns_from_expr(left, columns);
                Self::extract_columns_from_expr(right, columns);
            }
            SqlExpr::UnaryOp { expr, .. } => {
                Self::extract_columns_from_expr(expr, columns);
            }
            SqlExpr::FtsMatch { .. } => {} // FTS — no column to extract
            SqlExpr::Like { column, .. }
            | SqlExpr::Regexp { column, .. }
            | SqlExpr::In { column, .. }
            | SqlExpr::Between { column, .. }
            | SqlExpr::IsNull { column, .. }
            | SqlExpr::InSubquery { column, .. } => {
                // Strip table prefix if present
                let actual_name = if let Some(dot_pos) = column.rfind('.') {
                    &column[dot_pos + 1..]
                } else {
                    column.as_str()
                };
                columns.push(actual_name.to_string());
            }
            SqlExpr::Case {
                when_then,
                else_expr,
            } => {
                for (cond, then_expr) in when_then {
                    Self::extract_columns_from_expr(cond, columns);
                    Self::extract_columns_from_expr(then_expr, columns);
                }
                if let Some(else_e) = else_expr {
                    Self::extract_columns_from_expr(else_e, columns);
                }
            }
            SqlExpr::Function { args, .. } => {
                for arg in args {
                    Self::extract_columns_from_expr(arg, columns);
                }
            }
            SqlExpr::Cast { expr, .. } => {
                Self::extract_columns_from_expr(expr, columns);
            }
            SqlExpr::Paren(inner) => {
                Self::extract_columns_from_expr(inner, columns);
            }
            SqlExpr::ExistsSubquery { stmt } | SqlExpr::ScalarSubquery { stmt } => {
                // For correlated subqueries, extract outer column references from WHERE clause
                // These are columns like "u.user_id" or "outer_table.col" that reference the outer query
                if let Some(ref where_clause) = stmt.where_clause {
                    Self::extract_outer_refs_from_subquery(where_clause, columns);
                }
            }
            SqlExpr::InSubquery { column, stmt, .. } => {
                // The column being compared (e.g., "user_id" in "user_id IN (SELECT ...)")
                if column != "_id" {
                    columns.push(column.clone());
                }
                // Also extract outer references from subquery WHERE clause
                if let Some(ref where_clause) = stmt.where_clause {
                    Self::extract_outer_refs_from_subquery(where_clause, columns);
                }
            }
            SqlExpr::ArrayIndex { array, index } => {
                Self::extract_columns_from_expr(array, columns);
                Self::extract_columns_from_expr(index, columns);
            }
            _ => {}
        }
    }

    /// Extract outer column references from subquery expressions
    /// These are qualified column names like "u.col" or "table.col" that reference outer tables
    fn extract_outer_refs_from_subquery(expr: &SqlExpr, columns: &mut Vec<String>) {
        match expr {
            SqlExpr::Column(name) => {
                let clean_name = name.trim_matches('"');
                // Check for qualified names like "u.user_id" or "users.user_id"
                if let Some(dot_pos) = clean_name.find('.') {
                    let col_part = &clean_name[dot_pos + 1..];
                    if col_part != "_id" && !columns.contains(&col_part.to_string()) {
                        columns.push(col_part.to_string());
                    }
                }
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                Self::extract_outer_refs_from_subquery(left, columns);
                Self::extract_outer_refs_from_subquery(right, columns);
            }
            SqlExpr::UnaryOp { expr: inner, .. } | SqlExpr::Paren(inner) => {
                Self::extract_outer_refs_from_subquery(inner, columns);
            }
            _ => {}
        }
    }
}

/// SQL Parser
pub struct SqlParser {
    sql_raw: String,
    sql_chars: Option<Vec<char>>,
    tokens: Vec<SpannedToken>,
    pos: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct SpannedToken {
    token: Token,
    start: usize,
    end: usize,
}

/// Token types for SQL lexer
#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Keywords
    Select,
    From,
    Where,
    And,
    Or,
    Not,
    As,
    Distinct,
    Order,
    By,
    Asc,
    Desc,
    Limit,
    Offset,
    Nulls,
    First,
    Last,
    Like,
    In,
    Between,
    Is,
    Null,
    Group,
    Having,
    Count,
    Sum,
    Avg,
    Min,
    Max,
    True,
    False,
    Regexp,
    Over,
    Partition,
    Join,
    Left,
    Right,
    Full,
    Inner,
    Outer,
    Cross,
    On,
    Union,
    Intersect,
    Except,
    All,
    Exists,
    Cast,
    Case,
    When,
    Then,
    Else,
    End,
    Create,
    Drop,
    View,
    // DDL keywords
    Table,
    Alter,
    Add,
    Column,
    Rename,
    To,
    If,
    Truncate,
    // Index keywords
    Index,
    Unique,
    Using,
    // DML keywords
    Insert,
    Into,
    Values,
    Delete,
    Update,
    Set,
    // Variable keywords (token for $name)
    Variable(String),
    // Transaction keywords
    Begin,
    Commit,
    Rollback,
    Transaction,
    Read,
    Exclude,
    Replace,
    ColumnsKw,
    // CTE / EXPLAIN keywords
    With,
    Explain,
    Recursive,
    // Symbols
    Star,      // *
    Comma,     // ,
    Dot,       // .
    LParen,    // (
    RParen,    // )
    Semicolon, // ;
    Eq,        // =
    NotEq,     // != or <>
    Lt,        // <
    Le,        // <=
    Gt,        // >
    Ge,        // >=
    Plus,      // +
    Minus,     // -
    Slash,     // /
    Percent,   // %
    LBracket,  // [
    RBracket,  // ]
    // Literals
    Identifier(String),
    StringLit(String),
    IntLit(i64),
    FloatLit(f64),
    // End
    Eof,
}

impl SqlParser {
    /// Parse a SQL statement
    pub fn parse(sql: &str) -> Result<SqlStatement, ApexError> {
        let tokens = Self::tokenize(sql)?;
        let mut parser = SqlParser {
            sql_raw: sql.to_owned(),
            sql_chars: None,
            tokens,
            pos: 0,
        };
        let stmt = parser.parse_statement()?;
        // Skip trailing semicolons
        while matches!(parser.current(), Token::Semicolon) {
            parser.advance();
        }
        // Ensure all tokens are consumed
        if !matches!(parser.current(), Token::Eof) {
            let (start, _) = parser.current_span();
            let mut msg = format!("Unexpected token {:?} after statement", parser.current());
            if let Some(kw) = parser.keyword_suggestion() {
                msg = format!("{} (did you mean {}?)", msg, kw);
            }
            return Err(parser.syntax_error(start, msg));
        }
        Ok(stmt)
    }

    /// Parse multiple SQL statements separated by semicolons.
    pub fn parse_multi(sql: &str) -> Result<Vec<SqlStatement>, ApexError> {
        let tokens = Self::tokenize(sql)?;
        let mut parser = SqlParser {
            sql_raw: sql.to_owned(),
            sql_chars: None,
            tokens,
            pos: 0,
        };
        parser.parse_statements()
    }

    /// Parse a standalone SQL expression (same grammar as WHERE/HAVING).
    ///
    /// This is used to unify the non-SQL query language with SQL semantics.
    pub fn parse_expression(expr: &str) -> Result<SqlExpr, ApexError> {
        let tokens = Self::tokenize(expr)?;
        let mut parser = SqlParser {
            sql_raw: expr.to_owned(),
            sql_chars: None,
            tokens,
            pos: 0,
        };
        let e = parser.parse_expr()?;

        if !matches!(parser.current(), Token::Eof) {
            let (start, _) = parser.current_span();
            return Err(parser.syntax_error(
                start,
                format!(
                    "Unexpected token {:?} after end of expression",
                    parser.current()
                ),
            ));
        }

        Ok(e)
    }

    /// Ensure sql_chars is populated (lazy — only for error reporting / CHECK extraction).
    fn ensure_chars(&mut self) {
        if self.sql_chars.is_none() {
            self.sql_chars = Some(self.sql_raw.chars().collect());
        }
    }

    /// Get sql_chars reference, building lazily if needed.
    fn chars(&mut self) -> &[char] {
        self.ensure_chars();
        self.sql_chars.as_ref().unwrap()
    }

    /// Tokenize SQL string — returns token list.
    /// Uses byte-level scanning for cache-friendly 4x smaller working set vs Vec<char>.
    /// SQL is ASCII; `'`/`"` delimiters can never appear inside a multi-byte UTF-8 sequence.
    fn tokenize(sql: &str) -> Result<Vec<SpannedToken>, ApexError> {
        let mut tokens: Vec<SpannedToken> = Vec::with_capacity(sql.len() / 4 + 8);
        let bytes = sql.as_bytes(); // no allocation: reference into existing str
        let len = bytes.len();
        let mut i = 0;

        while i < len {
            let c = bytes[i];

            // Skip whitespace
            if c.is_ascii_whitespace() {
                i += 1;
                continue;
            }

            // -- line comment: skip to end of line
            if c == b'-' && i + 1 < len && bytes[i + 1] == b'-' {
                i += 2;
                while i < len && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }

            // /* block comment */: skip to closing */
            if c == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
                i += 2;
                loop {
                    if i + 1 >= len {
                        return Err(ApexError::QueryParseError(
                            "Unterminated block comment /* ... */".to_string(),
                        ));
                    }
                    if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        i += 2;
                        break;
                    }
                    i += 1;
                }
                continue;
            }

            // Single character tokens
            match c {
                b'*' => {
                    tokens.push(SpannedToken {
                        token: Token::Star,
                        start: i,
                        end: i + 1,
                    });
                    i += 1;
                    continue;
                }
                b',' => {
                    tokens.push(SpannedToken {
                        token: Token::Comma,
                        start: i,
                        end: i + 1,
                    });
                    i += 1;
                    continue;
                }
                b'.' => {
                    tokens.push(SpannedToken {
                        token: Token::Dot,
                        start: i,
                        end: i + 1,
                    });
                    i += 1;
                    continue;
                }
                b'(' => {
                    tokens.push(SpannedToken {
                        token: Token::LParen,
                        start: i,
                        end: i + 1,
                    });
                    i += 1;
                    continue;
                }
                b')' => {
                    tokens.push(SpannedToken {
                        token: Token::RParen,
                        start: i,
                        end: i + 1,
                    });
                    i += 1;
                    continue;
                }
                b';' => {
                    tokens.push(SpannedToken {
                        token: Token::Semicolon,
                        start: i,
                        end: i + 1,
                    });
                    i += 1;
                    continue;
                }
                b'+' => {
                    tokens.push(SpannedToken {
                        token: Token::Plus,
                        start: i,
                        end: i + 1,
                    });
                    i += 1;
                    continue;
                }
                b'-' => {
                    tokens.push(SpannedToken {
                        token: Token::Minus,
                        start: i,
                        end: i + 1,
                    });
                    i += 1;
                    continue;
                }
                b'/' => {
                    tokens.push(SpannedToken {
                        token: Token::Slash,
                        start: i,
                        end: i + 1,
                    });
                    i += 1;
                    continue;
                }
                b'%' => {
                    tokens.push(SpannedToken {
                        token: Token::Percent,
                        start: i,
                        end: i + 1,
                    });
                    i += 1;
                    continue;
                }
                b'[' => {
                    tokens.push(SpannedToken {
                        token: Token::LBracket,
                        start: i,
                        end: i + 1,
                    });
                    i += 1;
                    continue;
                }
                b']' => {
                    tokens.push(SpannedToken {
                        token: Token::RBracket,
                        start: i,
                        end: i + 1,
                    });
                    i += 1;
                    continue;
                }
                _ => {}
            }

            // Multi-character operators
            if c == b'=' {
                tokens.push(SpannedToken {
                    token: Token::Eq,
                    start: i,
                    end: i + 1,
                });
                i += 1;
                continue;
            }

            // Double-quoted identifier: "identifier"
            if c == b'"' {
                let start0 = i;
                i += 1;
                let start = i;
                while i < len && bytes[i] != b'"' {
                    i += 1;
                }
                if i >= len {
                    return Err(ApexError::QueryParseError(format!(
                        "Syntax error at byte {}: Unterminated double-quoted identifier",
                        start0
                    )));
                }
                let ident = sql[start..i].to_string();
                i += 1;
                tokens.push(SpannedToken {
                    token: Token::Identifier(ident),
                    start: start0,
                    end: i,
                });
                continue;
            }
            // Backtick-quoted identifier: `identifier` (Hive/MySQL style)
            if c == b'`' {
                let start0 = i;
                i += 1;
                let start = i;
                while i < len && bytes[i] != b'`' {
                    i += 1;
                }
                if i >= len {
                    return Err(ApexError::QueryParseError(format!(
                        "Syntax error at byte {}: Unterminated backtick-quoted identifier",
                        start0
                    )));
                }
                let ident = sql[start..i].to_string();
                i += 1;
                tokens.push(SpannedToken {
                    token: Token::Identifier(ident),
                    start: start0,
                    end: i,
                });
                continue;
            }
            if c == b'\'' {
                let start0 = i;
                i += 1;
                let start = i;
                while i < len && bytes[i] != b'\'' {
                    i += 1;
                }
                if i >= len {
                    return Err(ApexError::QueryParseError(format!(
                        "Syntax error at byte {}: Unterminated string literal",
                        start0
                    )));
                }
                let s = sql[start..i].to_string();
                i += 1;
                tokens.push(SpannedToken {
                    token: Token::StringLit(s),
                    start: start0,
                    end: i,
                });
                continue;
            }
            if c == b'!' && i + 1 < len && bytes[i + 1] == b'=' {
                tokens.push(SpannedToken {
                    token: Token::NotEq,
                    start: i,
                    end: i + 2,
                });
                i += 2;
                continue;
            }
            if c == b'<' {
                if i + 1 < len && bytes[i + 1] == b'=' {
                    tokens.push(SpannedToken {
                        token: Token::Le,
                        start: i,
                        end: i + 2,
                    });
                    i += 2;
                } else if i + 1 < len && bytes[i + 1] == b'>' {
                    tokens.push(SpannedToken {
                        token: Token::NotEq,
                        start: i,
                        end: i + 2,
                    });
                    i += 2;
                } else {
                    tokens.push(SpannedToken {
                        token: Token::Lt,
                        start: i,
                        end: i + 1,
                    });
                    i += 1;
                }
                continue;
            }
            if c == b'>' {
                if i + 1 < len && bytes[i + 1] == b'=' {
                    tokens.push(SpannedToken {
                        token: Token::Ge,
                        start: i,
                        end: i + 2,
                    });
                    i += 2;
                } else {
                    tokens.push(SpannedToken {
                        token: Token::Gt,
                        start: i,
                        end: i + 1,
                    });
                    i += 1;
                }
                continue;
            }

            // Numbers
            if c.is_ascii_digit() || (c == b'.' && i + 1 < len && bytes[i + 1].is_ascii_digit()) {
                let start = i;
                let mut has_dot = c == b'.';
                i += 1;
                while i < len && (bytes[i].is_ascii_digit() || (!has_dot && bytes[i] == b'.')) {
                    if bytes[i] == b'.' {
                        has_dot = true;
                    }
                    i += 1;
                }
                let num_str = &sql[start..i];
                if has_dot {
                    let f: f64 = num_str.parse().map_err(|_| {
                        ApexError::QueryParseError(format!(
                            "Syntax error at byte {}: Invalid number: {}",
                            start, num_str
                        ))
                    })?;
                    tokens.push(SpannedToken {
                        token: Token::FloatLit(f),
                        start,
                        end: i,
                    });
                } else {
                    let n: i64 = num_str.parse().map_err(|_| {
                        ApexError::QueryParseError(format!(
                            "Syntax error at byte {}: Invalid number: {}",
                            start, num_str
                        ))
                    })?;
                    tokens.push(SpannedToken {
                        token: Token::IntLit(n),
                        start,
                        end: i,
                    });
                }
                continue;
            }

            // Identifiers and keywords
            if c.is_ascii_alphabetic() || c == b'_' {
                let start = i;
                i += 1;
                while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let word = &sql[start..i];
                // Stack-allocated uppercase buffer: avoids heap alloc for keyword matching.
                let mut upper_buf = [0u8; 32];
                let kw_len = word.len().min(32);
                for (j, b) in word.bytes().enumerate().take(kw_len) {
                    upper_buf[j] = b.to_ascii_uppercase();
                }
                let upper = &upper_buf[..kw_len];
                let token = match upper {
                    b"SELECT" => Token::Select,
                    b"FROM" => Token::From,
                    b"WHERE" => Token::Where,
                    b"AND" => Token::And,
                    b"OR" => Token::Or,
                    b"NOT" => Token::Not,
                    b"AS" => Token::As,
                    b"DISTINCT" => Token::Distinct,
                    b"ORDER" => Token::Order,
                    b"BY" => Token::By,
                    b"ASC" => Token::Asc,
                    b"DESC" => Token::Desc,
                    b"LIMIT" => Token::Limit,
                    b"OFFSET" => Token::Offset,
                    b"NULLS" => Token::Nulls,
                    b"FIRST" => Token::First,
                    b"LAST" => Token::Last,
                    b"LIKE" => Token::Like,
                    b"IN" => Token::In,
                    b"BETWEEN" => Token::Between,
                    b"IS" => Token::Is,
                    b"NULL" => Token::Null,
                    b"GROUP" => Token::Group,
                    b"HAVING" => Token::Having,
                    b"COUNT" => Token::Count,
                    b"SUM" => Token::Sum,
                    b"AVG" => Token::Avg,
                    b"MIN" => Token::Min,
                    b"MAX" => Token::Max,
                    b"TRUE" => Token::True,
                    b"FALSE" => Token::False,
                    b"REGEXP" => Token::Regexp,
                    b"OVER" => Token::Over,
                    b"PARTITION" => Token::Partition,
                    b"JOIN" => Token::Join,
                    b"LEFT" => Token::Left,
                    b"RIGHT" => Token::Right,
                    b"FULL" => Token::Full,
                    b"INNER" => Token::Inner,
                    b"OUTER" => Token::Outer,
                    b"CROSS" => Token::Cross,
                    b"ON" => Token::On,
                    b"UNION" => Token::Union,
                    b"INTERSECT" => Token::Intersect,
                    b"EXCEPT" => Token::Except,
                    b"ALL" => Token::All,
                    b"EXISTS" => Token::Exists,
                    b"CAST" => Token::Cast,
                    b"CASE" => Token::Case,
                    b"WHEN" => Token::When,
                    b"THEN" => Token::Then,
                    b"ELSE" => Token::Else,
                    b"END" => Token::End,
                    b"CREATE" => Token::Create,
                    b"DROP" => Token::Drop,
                    b"VIEW" => Token::View,
                    // DDL keywords
                    b"TABLE" => Token::Table,
                    b"ALTER" => Token::Alter,
                    b"ADD" => Token::Add,
                    b"COLUMN" => Token::Column,
                    b"RENAME" => Token::Rename,
                    b"TO" => Token::To,
                    b"IF" => Token::If,
                    b"TRUNCATE" => Token::Truncate,
                    // Index keywords
                    b"INDEX" => Token::Index,
                    b"UNIQUE" => Token::Unique,
                    b"USING" => Token::Using,
                    // DML keywords
                    b"INSERT" => Token::Insert,
                    b"INTO" => Token::Into,
                    b"VALUES" => Token::Values,
                    b"DELETE" => Token::Delete,
                    b"UPDATE" => Token::Update,
                    b"SET" => Token::Set,
                    // Transaction keywords
                    b"BEGIN" => Token::Begin,
                    b"COMMIT" => Token::Commit,
                    b"ROLLBACK" => Token::Rollback,
                    b"TRANSACTION" => Token::Transaction,
                    b"READ" => Token::Read,
                    // CTE / EXPLAIN keywords
                    b"WITH" => Token::With,
                    b"EXPLAIN" => Token::Explain,
                    b"RECURSIVE" => Token::Recursive,
                    b"EXCLUDE" => Token::Exclude,
                    b"REPLACE" => Token::Replace,
                    b"COLUMNS" => Token::ColumnsKw,
                    _ => Token::Identifier(word.to_string()),
                };
                tokens.push(SpannedToken {
                    token,
                    start,
                    end: i,
                });
                continue;
            }

            // $variable_name — session variable reference
            if c == b'$'
                && i + 1 < len
                && (bytes[i + 1].is_ascii_alphabetic() || bytes[i + 1] == b'_')
            {
                let start = i;
                i += 1; // skip '$'
                let var_start = i;
                while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let name = sql[var_start..i].to_string();
                tokens.push(SpannedToken {
                    token: Token::Variable(name),
                    start,
                    end: i,
                });
                continue;
            }

            return Err(ApexError::QueryParseError(format!(
                "Syntax error at byte {}: Unexpected character: {}",
                i, c as char
            )));
        }

        tokens.push(SpannedToken {
            token: Token::Eof,
            start: len,
            end: len,
        });
        Ok(tokens)
    }

    fn current(&self) -> &Token {
        &self.tokens[self.pos].token
    }

    fn current_span(&self) -> (usize, usize) {
        let t = &self.tokens[self.pos];
        (t.start, t.end)
    }

    fn parse_statements(&mut self) -> Result<Vec<SqlStatement>, ApexError> {
        let mut out = Vec::new();
        while !matches!(self.current(), Token::Eof) {
            while matches!(self.current(), Token::Semicolon) {
                self.advance();
            }
            if matches!(self.current(), Token::Eof) {
                break;
            }
            let stmt = self.parse_statement()?;
            out.push(stmt);
            while matches!(self.current(), Token::Semicolon) {
                self.advance();
            }
        }
        Ok(out)
    }

    fn format_near(&mut self, at: usize) -> String {
        self.ensure_chars();
        let chars = self.sql_chars.as_ref().unwrap();
        if chars.is_empty() {
            return String::new();
        }
        let start = at.saturating_sub(16);
        let end = (at + 16).min(chars.len());
        let snippet: String = chars[start..end].iter().collect();
        snippet.replace('\n', " ")
    }

    fn line_col(&mut self, at: usize) -> (usize, usize) {
        self.ensure_chars();
        let chars = self.sql_chars.as_ref().unwrap();
        // 1-based line/col
        let mut line = 1usize;
        let mut col = 1usize;
        let end = at.min(chars.len());
        for ch in chars.iter().take(end) {
            if *ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        (line, col)
    }

    fn syntax_error(&mut self, at: usize, msg: String) -> ApexError {
        let near = self.format_near(at);
        let (line, col) = self.line_col(at);
        ApexError::QueryParseError(format!(
            "Syntax error at {}:{} (pos {}): {} (near: {})",
            line, col, at, msg, near
        ))
    }

    fn keyword_suggestion(&self) -> Option<String> {
        match self.current().clone() {
            Token::Identifier(s) => {
                let u = s.to_uppercase();
                // Keep list small and stable; used only for human-friendly hints.
                const KWS: [&str; 74] = [
                    "SELECT",
                    "FROM",
                    "WHERE",
                    "AND",
                    "OR",
                    "NOT",
                    "AS",
                    "DISTINCT",
                    "ORDER",
                    "BY",
                    "ASC",
                    "DESC",
                    "LIMIT",
                    "OFFSET",
                    "NULLS",
                    "FIRST",
                    "LAST",
                    "LIKE",
                    "IN",
                    "BETWEEN",
                    "IS",
                    "NULL",
                    "GROUP",
                    "HAVING",
                    "COUNT",
                    "SUM",
                    "AVG",
                    "MIN",
                    "MAX",
                    "TRUE",
                    "FALSE",
                    "REGEXP",
                    "OVER",
                    "PARTITION",
                    "JOIN",
                    "LEFT",
                    "RIGHT",
                    "FULL",
                    "INNER",
                    "OUTER",
                    "ON",
                    "UNION",
                    "ALL",
                    "EXISTS",
                    "CASE",
                    "WHEN",
                    "THEN",
                    "ELSE",
                    "END",
                    // DDL keywords
                    "TABLE",
                    "ALTER",
                    "ADD",
                    "COLUMN",
                    "RENAME",
                    "TRUNCATE",
                    // Index keywords
                    "INDEX",
                    "UNIQUE",
                    "USING",
                    // DML keywords
                    "INSERT",
                    "INTO",
                    "VALUES",
                    "DELETE",
                    "UPDATE",
                    "SET",
                    // Transaction keywords
                    "BEGIN",
                    "COMMIT",
                    "ROLLBACK",
                    "TRANSACTION",
                    "READ",
                    "WITH",
                    "EXPLAIN",
                    "RECURSIVE",
                    "EXCLUDE",
                    "REPLACE",
                ];

                // Fast path for common "plural" / extra trailing char typos: FROMs, WHEREs, LIKEs, LIMITs
                for kw in KWS {
                    if u.len() == kw.len() + 1 && u.starts_with(kw) {
                        return Some(kw.to_string());
                    }
                    if u.ends_with('S') && &u[..u.len() - 1] == kw {
                        return Some(kw.to_string());
                    }
                }

                // Fuzzy match: allow small edit distance (e.g., SELECTE -> SELECT)
                let mut best: Option<(&str, usize)> = None;
                for kw in KWS {
                    let dist = Self::edit_distance(&u, kw);
                    if dist <= 2 {
                        match best {
                            None => best = Some((kw, dist)),
                            Some((_, best_dist)) if dist < best_dist => best = Some((kw, dist)),
                            _ => {}
                        }
                    }
                }
                best.map(|(kw, _)| kw.to_string())
            }
            _ => None,
        }
    }

    /// Stricter check than keyword_suggestion: only returns true if the
    /// identifier is very likely a misspelled keyword (edit distance <= 1).
    /// Used to prevent misspelled keywords from being consumed as table aliases.
    fn is_likely_misspelled_keyword(&self, ident: &str) -> bool {
        let u = ident.to_uppercase();
        const KWS: [&str; 15] = [
            "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "ORDER", "GROUP",
            "HAVING", "LIMIT", "OFFSET", "UNION", "INSERT", "DELETE",
        ];
        for kw in KWS {
            // Length must be similar (differ by at most 1)
            let len_diff = (u.len() as isize - kw.len() as isize).unsigned_abs();
            if len_diff <= 1 && Self::edit_distance(&u, kw) <= 1 {
                return true;
            }
        }
        false
    }

    fn edit_distance(a: &str, b: &str) -> usize {
        // Classic DP Levenshtein distance. Inputs are short keywords; performance is irrelevant.
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        let n = a.len();
        let m = b.len();

        if n == 0 {
            return m;
        }
        if m == 0 {
            return n;
        }

        let mut dp = vec![vec![0usize; m + 1]; n + 1];
        for i in 0..=n {
            dp[i][0] = i;
        }
        for j in 0..=m {
            dp[0][j] = j;
        }

        for i in 1..=n {
            for j in 1..=m {
                let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }
        dp[n][m]
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos].token;
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, expected: Token) -> Result<(), ApexError> {
        if std::mem::discriminant(self.current()) == std::mem::discriminant(&expected) {
            self.advance();
            Ok(())
        } else {
            let (start, _) = self.current_span();
            let mut msg = format!("Expected {:?}, got {:?}", expected, self.current());
            if let Some(kw) = self.keyword_suggestion() {
                msg = format!("{} (did you mean {}?)", msg, kw);
            }
            Err(self.syntax_error(start, msg))
        }
    }

    fn parse_statement(&mut self) -> Result<SqlStatement, ApexError> {
        match self.current() {
            Token::Explain => {
                self.advance();
                let analyze = if matches!(self.current(), Token::Identifier(ref s) if s.to_uppercase() == "ANALYZE")
                {
                    self.advance();
                    true
                } else {
                    false
                };
                let inner = self.parse_statement()?;
                Ok(SqlStatement::Explain {
                    stmt: Box::new(inner),
                    analyze,
                })
            }
            Token::With => {
                // CTE: WITH name AS (SELECT ...) [, name2 AS (SELECT ...)] SELECT ...
                self.advance();
                let recursive = if matches!(self.current(), Token::Recursive) {
                    self.advance();
                    true
                } else {
                    false
                };
                let mut ctes: Vec<(String, Vec<String>, SqlStatement)> = Vec::new();
                loop {
                    let cte_name = self.parse_identifier()?;
                    // Optional column alias list: cte_name(col1, col2, ...)
                    let mut col_aliases = Vec::new();
                    if matches!(self.current(), Token::LParen) {
                        self.advance();
                        loop {
                            col_aliases.push(self.parse_identifier()?);
                            if matches!(self.current(), Token::Comma) {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        self.expect(Token::RParen)?;
                    }
                    self.expect(Token::As)?;
                    self.expect(Token::LParen)?;
                    let cte_stmt = self.parse_statement()?;
                    self.expect(Token::RParen)?;
                    ctes.push((cte_name, col_aliases, cte_stmt));
                    if matches!(self.current(), Token::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                // Parse the main statement (SELECT, INSERT, etc.)
                let mut main_stmt = self.parse_statement()?;
                // Attach CTEs to the main statement by wrapping in CTE variant
                // We store CTEs by converting to temp-table materialization at execution time
                // For now, return the main statement with CTEs stored
                // We'll use a simple approach: wrap in a new variant or pass through
                // Since CTE execution is handled in executor, we need a way to carry the CTE definitions.
                // Add a CTE wrapper approach:
                for (cte_name, col_aliases, cte_body) in ctes.into_iter().rev() {
                    main_stmt = SqlStatement::Cte {
                        name: cte_name,
                        column_aliases: col_aliases,
                        body: Box::new(cte_body),
                        main: Box::new(main_stmt),
                        recursive,
                    };
                }
                Ok(main_stmt)
            }
            Token::Select => {
                // Parse the first SELECT part without consuming ORDER/LIMIT/OFFSET.
                // Those trailing clauses belong to UNION result if a UNION follows.
                let mut stmt = SqlStatement::Select(self.parse_select_part()?);

                // UNION / INTERSECT / EXCEPT chain
                while matches!(
                    self.current(),
                    Token::Union | Token::Intersect | Token::Except
                ) {
                    let set_op = match self.current() {
                        Token::Union => SetOpType::Union,
                        Token::Intersect => SetOpType::Intersect,
                        Token::Except => SetOpType::Except,
                        _ => unreachable!(),
                    };
                    self.advance();
                    let all = if matches!(self.current(), Token::All) {
                        self.advance();
                        true
                    } else {
                        false
                    };

                    if !matches!(self.current(), Token::Select) {
                        let (start, _) = self.current_span();
                        return Err(self.syntax_error(
                            start,
                            "Expected SELECT after set operation".to_string(),
                        ));
                    }
                    let right = SqlStatement::Select(self.parse_select_part()?);

                    stmt = SqlStatement::Union(UnionStatement {
                        left: Box::new(stmt),
                        right: Box::new(right),
                        all,
                        set_op,
                        order_by: Vec::new(),
                        limit: None,
                        offset: None,
                    });
                }

                // Trailing clauses apply to the final result (SELECT or UNION)
                let order_by = if matches!(self.current(), Token::Order) {
                    self.advance();
                    self.expect(Token::By)?;
                    self.parse_order_by()?
                } else {
                    Vec::new()
                };

                let limit = if matches!(self.current(), Token::Limit) {
                    self.advance();
                    if let Token::IntLit(n) = self.current().clone() {
                        self.advance();
                        Some(n as usize)
                    } else {
                        return Err(ApexError::QueryParseError(
                            "Expected number after LIMIT".to_string(),
                        ));
                    }
                } else {
                    None
                };

                let offset = if matches!(self.current(), Token::Offset) {
                    self.advance();
                    if let Token::IntLit(n) = self.current().clone() {
                        self.advance();
                        Some(n as usize)
                    } else {
                        return Err(ApexError::QueryParseError(
                            "Expected number after OFFSET".to_string(),
                        ));
                    }
                } else {
                    None
                };

                if !order_by.is_empty() || limit.is_some() || offset.is_some() {
                    match stmt {
                        SqlStatement::Union(mut u) => {
                            u.order_by = order_by;
                            u.limit = limit;
                            u.offset = offset;
                            stmt = SqlStatement::Union(u);
                        }
                        SqlStatement::Select(mut s) => {
                            s.order_by = order_by;
                            s.limit = limit;
                            s.offset = offset;
                            stmt = SqlStatement::Select(s);
                        }
                        _ => {}
                    }
                }

                Ok(stmt)
            }
            Token::Create => {
                self.advance();
                // CREATE UNIQUE INDEX ...
                let unique = if matches!(self.current(), Token::Unique) {
                    self.advance();
                    true
                } else {
                    false
                };
                // CREATE [TEMP | TEMPORARY] TABLE ...
                let temp = if let Token::Identifier(ref kw) = self.current() {
                    let upper = kw.to_uppercase();
                    if upper == "TEMP" || upper == "TEMPORARY" {
                        self.advance();
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                match self.current() {
                    Token::View if !unique => {
                        self.advance();
                        let name = self.parse_identifier()?;
                        self.expect(Token::As)?;
                        if !matches!(self.current(), Token::Select) {
                            let (start, _) = self.current_span();
                            return Err(
                                self.syntax_error(start, "Expected SELECT after AS".to_string())
                            );
                        }
                        let stmt = self.parse_select_internal(true)?;
                        Ok(SqlStatement::CreateView { name, stmt })
                    }
                    Token::Table if !unique => {
                        self.advance();
                        // Check for TEMP / TEMPORARY (after TABLE — backward compat)
                        let temp = temp
                            || if let Token::Identifier(ref kw) = self.current() {
                                let upper = kw.to_uppercase();
                                if upper == "TEMP" || upper == "TEMPORARY" {
                                    self.advance();
                                    true
                                } else {
                                    false
                                }
                            } else {
                                false
                            };
                        // Check for IF NOT EXISTS
                        let if_not_exists = self.parse_if_not_exists()?;
                        let table = self.parse_table_name()?;
                        // CTAS: CREATE TABLE name AS SELECT ...
                        if matches!(self.current(), Token::As) {
                            self.advance();
                            let query = self.parse_statement()?;
                            return Ok(SqlStatement::CreateTableAs {
                                table,
                                query: Box::new(query),
                                if_not_exists,
                                temp,
                            });
                        }
                        // Column definitions are optional: CREATE TABLE t OR CREATE TABLE t (col INT)
                        let columns = if matches!(self.current(), Token::LParen) {
                            self.advance();
                            let cols = self.parse_column_defs()?;
                            self.expect(Token::RParen)?;
                            cols
                        } else {
                            Vec::new()
                        };
                        Ok(SqlStatement::CreateTable {
                            table,
                            columns,
                            if_not_exists,
                            temp,
                        })
                    }
                    Token::Identifier(ref fts_kw) if fts_kw.to_uppercase() == "FTS" && !unique => {
                        // CREATE FTS INDEX ON table [(col1, col2)] [WITH (opt=val, ...)]
                        self.advance(); // consume FTS
                                        // Expect INDEX keyword (as identifier or Token::Index)
                        match self.current().clone() {
                            Token::Index => {
                                self.advance();
                            }
                            Token::Identifier(ref s) if s.to_uppercase() == "INDEX" => {
                                self.advance();
                            }
                            other => {
                                return Err(ApexError::QueryParseError(format!(
                                    "Expected INDEX after CREATE FTS, got {:?}",
                                    other
                                )))
                            }
                        }
                        self.expect(Token::On)?;
                        let table = self.parse_identifier()?;
                        // Optional column list
                        let fields = if matches!(self.current(), Token::LParen) {
                            self.advance();
                            let cols = self.parse_identifier_list()?;
                            self.expect(Token::RParen)?;
                            Some(cols)
                        } else {
                            None
                        };
                        // Optional WITH (key=value, ...)
                        let mut lazy_load = false;
                        let mut cache_size: usize = 10000;
                        if matches!(self.current(), Token::Identifier(ref s) if s.to_uppercase() == "WITH")
                        {
                            self.advance();
                            self.expect(Token::LParen)?;
                            loop {
                                let key = self.parse_identifier()?.to_lowercase();
                                self.expect(Token::Eq)?;
                                let val = match self.current().clone() {
                                    Token::Identifier(v) => {
                                        self.advance();
                                        v
                                    }
                                    Token::IntLit(n) => {
                                        self.advance();
                                        n.to_string()
                                    }
                                    Token::True => {
                                        self.advance();
                                        "true".to_string()
                                    }
                                    Token::False => {
                                        self.advance();
                                        "false".to_string()
                                    }
                                    other => {
                                        return Err(ApexError::QueryParseError(format!(
                                            "Expected value in WITH options, got {:?}",
                                            other
                                        )))
                                    }
                                };
                                match key.as_str() {
                                    "lazy_load" => {
                                        lazy_load =
                                            matches!(val.to_lowercase().as_str(), "true" | "1")
                                    }
                                    "cache_size" => cache_size = val.parse().unwrap_or(10000),
                                    _ => {}
                                }
                                if !matches!(self.current(), Token::Comma) {
                                    break;
                                }
                                self.advance();
                            }
                            self.expect(Token::RParen)?;
                        }
                        return Ok(SqlStatement::CreateFtsIndex {
                            table,
                            fields,
                            lazy_load,
                            cache_size,
                        });
                    }
                    Token::Index => {
                        // CREATE [UNIQUE] INDEX [IF NOT EXISTS] name ON table (col1[, col2, ...]) [USING HASH|BTREE]
                        self.advance();
                        let if_not_exists = self.parse_if_not_exists()?;
                        let name = self.parse_identifier()?;
                        self.expect(Token::On)?;
                        let table = self.parse_identifier()?;
                        self.expect(Token::LParen)?;
                        let columns = self.parse_identifier_list()?;
                        self.expect(Token::RParen)?;
                        let index_type = if matches!(self.current(), Token::Using) {
                            self.advance();
                            let t = self.parse_identifier()?;
                            Some(t.to_uppercase())
                        } else {
                            None
                        };
                        Ok(SqlStatement::CreateIndex {
                            name,
                            table,
                            columns,
                            unique,
                            index_type,
                            if_not_exists,
                        })
                    }
                    _ => {
                        let (start, _) = self.current_span();
                        if unique {
                            Err(self.syntax_error(
                                start,
                                "Expected INDEX after CREATE UNIQUE".to_string(),
                            ))
                        } else {
                            Err(self.syntax_error(
                                start,
                                "Expected TABLE, VIEW, or INDEX after CREATE".to_string(),
                            ))
                        }
                    }
                }
            }
            Token::Drop => {
                self.advance();
                match self.current() {
                    Token::View => {
                        self.advance();
                        let name = self.parse_identifier()?;
                        Ok(SqlStatement::DropView { name })
                    }
                    Token::Table => {
                        self.advance();
                        // Check for IF EXISTS
                        let if_exists = self.parse_if_exists()?;
                        let table = self.parse_table_name()?;
                        Ok(SqlStatement::DropTable { table, if_exists })
                    }
                    Token::Identifier(ref fts_kw) if fts_kw.to_uppercase() == "FTS" => {
                        // DROP FTS INDEX ON table
                        self.advance(); // consume FTS
                        match self.current().clone() {
                            Token::Index => {
                                self.advance();
                            }
                            Token::Identifier(ref s) if s.to_uppercase() == "INDEX" => {
                                self.advance();
                            }
                            other => {
                                return Err(ApexError::QueryParseError(format!(
                                    "Expected INDEX after DROP FTS, got {:?}",
                                    other
                                )))
                            }
                        }
                        self.expect(Token::On)?;
                        let table = self.parse_identifier()?;
                        Ok(SqlStatement::DropFtsIndex { table })
                    }
                    Token::Index => {
                        // DROP INDEX [IF EXISTS] name ON table
                        self.advance();
                        let if_exists = self.parse_if_exists()?;
                        let name = self.parse_identifier()?;
                        self.expect(Token::On)?;
                        let table = self.parse_identifier()?;
                        Ok(SqlStatement::DropIndex {
                            name,
                            table,
                            if_exists,
                        })
                    }
                    _ => {
                        let (start, _) = self.current_span();
                        Err(self.syntax_error(
                            start,
                            "Expected TABLE, VIEW, or INDEX after DROP".to_string(),
                        ))
                    }
                }
            }
            Token::Alter => {
                self.advance();
                // Detect ALTER FTS INDEX ON table DISABLE
                if matches!(self.current(), Token::Identifier(ref s) if s.to_uppercase() == "FTS") {
                    self.advance(); // consume FTS
                    match self.current().clone() {
                        Token::Index => {
                            self.advance();
                        }
                        Token::Identifier(ref s) if s.to_uppercase() == "INDEX" => {
                            self.advance();
                        }
                        other => {
                            return Err(ApexError::QueryParseError(format!(
                                "Expected INDEX after ALTER FTS, got {:?}",
                                other
                            )))
                        }
                    }
                    self.expect(Token::On)?;
                    let table = self.parse_identifier()?;
                    // Expect ENABLE or DISABLE
                    match self.current().clone() {
                        Token::Identifier(ref s) if s.to_uppercase() == "DISABLE" => {
                            self.advance();
                            return Ok(SqlStatement::AlterFtsIndexDisable { table });
                        }
                        Token::Identifier(ref s) if s.to_uppercase() == "ENABLE" => {
                            self.advance();
                            return Ok(SqlStatement::AlterFtsIndexEnable { table });
                        }
                        other => {
                            return Err(ApexError::QueryParseError(format!(
                            "Expected ENABLE or DISABLE after ALTER FTS INDEX ON table, got {:?}",
                            other
                        )))
                        }
                    }
                }
                self.expect(Token::Table)?;
                let table = self.parse_table_name()?;
                let operation = self.parse_alter_operation()?;
                Ok(SqlStatement::AlterTable { table, operation })
            }
            Token::Truncate => {
                self.advance();
                self.expect(Token::Table)?;
                let table = self.parse_table_name()?;
                Ok(SqlStatement::TruncateTable { table })
            }
            Token::Insert => {
                self.advance();
                self.expect(Token::Into)?;
                let table = self.parse_table_name()?;
                // Optional column list
                let columns = if matches!(self.current(), Token::LParen) {
                    // Peek ahead: could be column list followed by VALUES/SELECT,
                    // or could be VALUES directly
                    self.advance();
                    let cols = self.parse_identifier_list()?;
                    self.expect(Token::RParen)?;
                    Some(cols)
                } else {
                    None
                };
                // INSERT ... SELECT or INSERT ... VALUES
                if matches!(self.current(), Token::Select) || matches!(self.current(), Token::With)
                {
                    let query = self.parse_statement()?;
                    Ok(SqlStatement::InsertSelect {
                        table,
                        columns,
                        query: Box::new(query),
                    })
                } else {
                    self.expect(Token::Values)?;
                    let values = self.parse_values_list()?;
                    // Check for ON CONFLICT clause (UPSERT)
                    if matches!(self.current(), Token::On) {
                        self.advance();
                        // Expect CONFLICT as identifier
                        let kw = self.parse_identifier()?;
                        if kw.to_uppercase() != "CONFLICT" {
                            let (start, _) = self.current_span();
                            return Err(
                                self.syntax_error(start, "Expected CONFLICT after ON".to_string())
                            );
                        }
                        // Parse conflict target: (col1, col2, ...)
                        self.expect(Token::LParen)?;
                        let conflict_columns = self.parse_identifier_list()?;
                        self.expect(Token::RParen)?;
                        // Expect DO
                        let do_kw = self.parse_identifier()?;
                        if do_kw.to_uppercase() != "DO" {
                            let (start, _) = self.current_span();
                            return Err(self.syntax_error(
                                start,
                                "Expected DO after conflict target".to_string(),
                            ));
                        }
                        // DO UPDATE SET ... or DO NOTHING
                        let do_update = if matches!(self.current(), Token::Update) {
                            self.advance();
                            self.expect(Token::Set)?;
                            let assignments = self.parse_assignments()?;
                            Some(assignments)
                        } else {
                            let action_kw = self.parse_identifier()?;
                            match action_kw.to_uppercase().as_str() {
                                "NOTHING" => None,
                                _ => {
                                    let (start, _) = self.current_span();
                                    return Err(self.syntax_error(
                                        start,
                                        "Expected UPDATE or NOTHING after DO".to_string(),
                                    ));
                                }
                            }
                        };
                        Ok(SqlStatement::InsertOnConflict {
                            table,
                            columns,
                            values,
                            conflict_columns,
                            do_update,
                        })
                    } else {
                        Ok(SqlStatement::Insert {
                            table,
                            columns,
                            values,
                        })
                    }
                }
            }
            Token::Delete => {
                self.advance();
                self.expect(Token::From)?;
                let table = self.parse_table_name()?;
                let where_clause = if matches!(self.current(), Token::Where) {
                    self.advance();
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                Ok(SqlStatement::Delete {
                    table,
                    where_clause,
                })
            }
            Token::Update => {
                self.advance();
                let table = self.parse_table_name()?;
                self.expect(Token::Set)?;
                let assignments = self.parse_assignments()?;
                let where_clause = if matches!(self.current(), Token::Where) {
                    self.advance();
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                Ok(SqlStatement::Update {
                    table,
                    assignments,
                    where_clause,
                })
            }
            Token::Set => {
                // SET VARIABLE name = value
                self.advance();
                let kw = self.parse_identifier()?;
                if kw.to_uppercase() != "VARIABLE" {
                    let (start, _) = self.current_span();
                    return Err(self.syntax_error(start, "Expected VARIABLE after SET".to_string()));
                }
                let name = self.parse_identifier()?;
                self.expect(Token::Eq)?;
                let value = self.parse_literal_value()?;
                Ok(SqlStatement::SetVariable { name, value })
            }
            Token::Begin => {
                // BEGIN [TRANSACTION] [READ ONLY]
                self.advance();
                if matches!(self.current(), Token::Transaction) {
                    self.advance();
                }
                let read_only = if matches!(self.current(), Token::Read) {
                    self.advance();
                    // Expect "ONLY" as identifier
                    let word = self.parse_identifier()?;
                    if word.to_uppercase() != "ONLY" {
                        let (start, _) = self.current_span();
                        return Err(
                            self.syntax_error(start, "Expected ONLY after READ".to_string())
                        );
                    }
                    true
                } else {
                    false
                };
                Ok(SqlStatement::BeginTransaction { read_only })
            }
            Token::Commit => {
                self.advance();
                Ok(SqlStatement::Commit)
            }
            Token::Rollback => {
                self.advance();
                // Check for ROLLBACK TO [SAVEPOINT] name
                if matches!(self.current(), Token::Identifier(ref s) if s.to_uppercase() == "TO") {
                    self.advance();
                    // Optional SAVEPOINT keyword
                    if matches!(self.current(), Token::Identifier(ref s) if s.to_uppercase() == "SAVEPOINT")
                    {
                        self.advance();
                    }
                    let name = self.parse_identifier()?;
                    Ok(SqlStatement::RollbackToSavepoint { name })
                } else {
                    Ok(SqlStatement::Rollback)
                }
            }
            _ => {
                // Check for SAVEPOINT / RELEASE as identifier-based keywords
                if let Token::Identifier(ref s) = self.current().clone() {
                    let upper = s.to_uppercase();
                    match upper.as_str() {
                        "SAVEPOINT" => {
                            self.advance();
                            let name = self.parse_identifier()?;
                            return Ok(SqlStatement::Savepoint { name });
                        }
                        "ANALYZE" => {
                            self.advance();
                            // Optional TABLE keyword
                            if matches!(self.current(), Token::Table) {
                                self.advance();
                            }
                            let table = self.parse_identifier()?;
                            return Ok(SqlStatement::AnalyzeTable { table });
                        }
                        "COPY" => {
                            self.advance();
                            let table = self.parse_identifier()?;
                            // Expect TO or FROM (may be keywords or identifiers)
                            let direction = match self.current().clone() {
                                Token::From => {
                                    self.advance();
                                    "FROM".to_string()
                                }
                                Token::To => {
                                    self.advance();
                                    "TO".to_string()
                                }
                                Token::Identifier(ref kw) => {
                                    let u = kw.to_uppercase();
                                    self.advance();
                                    u
                                }
                                _ => {
                                    return Err(ApexError::QueryParseError(
                                        "Expected TO or FROM after COPY table".to_string(),
                                    ))
                                }
                            };
                            // file path as string literal
                            let file_path = if let Token::StringLit(ref s) = self.current().clone()
                            {
                                let p = s.clone();
                                self.advance();
                                p
                            } else {
                                return Err(ApexError::QueryParseError(
                                    "Expected file path string after COPY table TO/FROM"
                                        .to_string(),
                                ));
                            };
                            match direction.as_str() {
                                "TO" => {
                                    let (format, options) = self.parse_copy_options(&file_path)?;
                                    if format == "PARQUET" && options.is_empty() {
                                        return Ok(SqlStatement::CopyToParquet {
                                            table,
                                            file_path,
                                        });
                                    }
                                    return Ok(SqlStatement::CopyExport {
                                        table,
                                        file_path,
                                        format,
                                        options,
                                    });
                                }
                                "FROM" => {
                                    let (format, options) = self.parse_copy_options(&file_path)?;
                                    if format == "PARQUET" && options.is_empty() {
                                        return Ok(SqlStatement::CopyFromParquet {
                                            table,
                                            file_path,
                                        });
                                    }
                                    return Ok(SqlStatement::CopyImport {
                                        table,
                                        file_path,
                                        format,
                                        options,
                                    });
                                }
                                _ => {
                                    return Err(ApexError::QueryParseError(
                                        "Expected TO or FROM in COPY statement".to_string(),
                                    ))
                                }
                            }
                        }
                        "RELEASE" => {
                            self.advance();
                            // Optional SAVEPOINT keyword
                            if matches!(self.current(), Token::Identifier(ref s2) if s2.to_uppercase() == "SAVEPOINT")
                            {
                                self.advance();
                            }
                            let name = self.parse_identifier()?;
                            return Ok(SqlStatement::ReleaseSavepoint { name });
                        }
                        "RESET" => {
                            self.advance();
                            let kw = self.parse_identifier()?;
                            if kw.to_uppercase() != "VARIABLE" {
                                let (start, _) = self.current_span();
                                return Err(self.syntax_error(
                                    start,
                                    "Expected VARIABLE after RESET".to_string(),
                                ));
                            }
                            let name = self.parse_identifier()?;
                            return Ok(SqlStatement::ResetVariable { name });
                        }
                        "SHOW" => {
                            self.advance();
                            // SHOW FTS INDEXES
                            match self.current().clone() {
                                Token::Identifier(ref s) if s.to_uppercase() == "FTS" => {
                                    self.advance();
                                    // consume INDEXES or INDEX
                                    if matches!(self.current(), Token::Index)
                                        || matches!(self.current(), Token::Identifier(ref s2) if s2.to_uppercase() == "INDEXES" || s2.to_uppercase() == "INDEX")
                                    {
                                        self.advance();
                                    }
                                    return Ok(SqlStatement::ShowFtsIndexes);
                                }
                                other => {
                                    return Err(ApexError::QueryParseError(format!(
                                        "Expected FTS after SHOW, got {:?}",
                                        other
                                    )))
                                }
                            }
                        }
                        "REINDEX" => {
                            self.advance();
                            // Optional TABLE keyword
                            if matches!(self.current(), Token::Table) {
                                self.advance();
                            }
                            let table = self.parse_identifier()?;
                            return Ok(SqlStatement::Reindex { table });
                        }
                        "PRAGMA" => {
                            self.advance();
                            let name = self.parse_identifier()?;
                            // Optional argument: PRAGMA name(arg) or PRAGMA name = arg
                            let arg = if matches!(self.current(), Token::LParen) {
                                self.advance();
                                let a = self.parse_identifier()?;
                                self.expect(Token::RParen)?;
                                Some(a)
                            } else if matches!(self.current(), Token::Eq) {
                                self.advance();
                                if let Token::Identifier(ref v) = self.current().clone() {
                                    let a = v.clone();
                                    self.advance();
                                    Some(a)
                                } else if let Token::StringLit(ref v) = self.current().clone() {
                                    let a = v.clone();
                                    self.advance();
                                    Some(a)
                                } else if let Token::IntLit(v) = self.current().clone() {
                                    let a = v.to_string();
                                    self.advance();
                                    Some(a)
                                } else if let Token::FloatLit(v) = self.current().clone() {
                                    let a = v.to_string();
                                    self.advance();
                                    Some(a)
                                } else {
                                    None
                                }
                            } else {
                                None
                            };
                            return Ok(SqlStatement::Pragma { name, arg });
                        }
                        _ => {}
                    }
                }
                let (start, _) = self.current_span();
                let mut msg = "Expected SQL statement".to_string();
                if let Some(kw) = self.keyword_suggestion() {
                    msg = format!("{} (did you mean {}?)", msg, kw);
                }
                Err(self.syntax_error(start, msg))
            }
        }
    }

    // Parse a SELECT used as a UNION operand: does not consume ORDER BY / LIMIT / OFFSET.
    pub fn parse_select_statement(&mut self) -> Result<SelectStatement, ApexError> {
        self.parse_select_internal(false)
    }

    fn parse_select_part(&mut self) -> Result<SelectStatement, ApexError> {
        self.parse_select_internal(false)
    }

    #[allow(dead_code)]
    fn parse_select(&mut self) -> Result<SelectStatement, ApexError> {
        self.parse_select_internal(true)
    }

    fn parse_select_internal(&mut self, parse_tail: bool) -> Result<SelectStatement, ApexError> {
        self.expect(Token::Select)?;

        // DISTINCT [ON (...)]
        let (distinct, distinct_on) = if matches!(self.current(), Token::Distinct) {
            self.advance();
            let on = if matches!(self.current(), Token::On) {
                self.advance();
                self.expect(Token::LParen)?;
                let cols = self.parse_column_list()?;
                self.expect(Token::RParen)?;
                Some(cols)
            } else {
                None
            };
            (true, on)
        } else {
            (false, None)
        };

        // Columns
        let columns = self.parse_select_columns()?;

        // FROM (optional for simple queries)
        let from = if matches!(self.current(), Token::From) {
            self.advance();
            match self.current().clone() {
                Token::StringLit(file) => {
                    self.advance();
                    let alias = if matches!(self.current(), Token::As) {
                        self.advance();
                        Some(self.parse_identifier()?)
                    } else if let Token::Identifier(a) = self.current().clone() {
                        if a.len() >= 4 && self.is_likely_misspelled_keyword(&a) {
                            None
                        } else {
                            self.advance();
                            Some(a)
                        }
                    } else {
                        None
                    };
                    Some(FromItem::DirectFile { file, alias })
                }
                Token::Identifier(table) => {
                    self.advance();
                    let upper = table.to_uppercase();
                    // TOPK_DISTANCE(col, [vec], k, 'metric') — heap-based vector TopK table function
                    if upper == "TOPK_DISTANCE" && matches!(self.current(), Token::LParen) {
                        self.advance(); // consume '('
                        let col = self.parse_identifier()?;
                        self.expect(Token::Comma)?;
                        let query_expr = self.parse_expr()?;
                        let query = match query_expr {
                            SqlExpr::ArrayLiteral(v) => v,
                            _ => return Err(ApexError::QueryParseError(
                                "topk_distance: second argument must be an array literal [f1, f2, ...]".to_string(),
                            )),
                        };
                        self.expect(Token::Comma)?;
                        let k = match self.current().clone() {
                            Token::IntLit(n) => {
                                self.advance();
                                n as usize
                            }
                            other => {
                                return Err(ApexError::QueryParseError(format!(
                                    "topk_distance: third argument must be integer k, got {:?}",
                                    other
                                )))
                            }
                        };
                        self.expect(Token::Comma)?;
                        let metric = match self.current().clone() {
                            Token::StringLit(s) => {
                                self.advance();
                                s
                            }
                            other => {
                                return Err(ApexError::QueryParseError(format!(
                                "topk_distance: fourth argument must be a metric string, got {:?}",
                                other
                            )))
                            }
                        };
                        self.expect(Token::RParen)?;
                        let alias = if matches!(self.current(), Token::As) {
                            self.advance();
                            Some(self.parse_identifier()?)
                        } else if let Token::Identifier(a) = self.current().clone() {
                            if a.len() >= 4 && self.is_likely_misspelled_keyword(&a) {
                                None
                            } else {
                                self.advance();
                                Some(a)
                            }
                        } else {
                            None
                        };
                        Some(FromItem::TopkDistance {
                            col,
                            query,
                            k,
                            metric,
                            alias,
                        })
                    // Check for table function: read_csv(...), read_parquet(...), read_json(...)
                    } else if matches!(upper.as_str(), "READ_CSV" | "READ_PARQUET" | "READ_JSON")
                        && matches!(self.current(), Token::LParen)
                    {
                        self.advance(); // consume '('
                        let file = match self.current().clone() {
                            Token::StringLit(s) => {
                                self.advance();
                                s
                            }
                            other => {
                                return Err(ApexError::QueryParseError(format!(
                                    "Expected file path string in {}(), got {:?}",
                                    table, other
                                )))
                            }
                        };
                        let mut options: Vec<(String, String)> = Vec::new();
                        while matches!(self.current(), Token::Comma) {
                            self.advance();
                            let key = self.parse_identifier()?.to_lowercase();
                            self.expect(Token::Eq)?;
                            let val = match self.current().clone() {
                                Token::StringLit(s) => {
                                    self.advance();
                                    s
                                }
                                Token::Identifier(s) => {
                                    self.advance();
                                    s
                                }
                                Token::IntLit(n) => {
                                    self.advance();
                                    n.to_string()
                                }
                                Token::FloatLit(f) => {
                                    self.advance();
                                    f.to_string()
                                }
                                Token::True => {
                                    self.advance();
                                    "true".to_string()
                                }
                                Token::False => {
                                    self.advance();
                                    "false".to_string()
                                }
                                other => {
                                    return Err(ApexError::QueryParseError(format!(
                                        "Expected option value, got {:?}",
                                        other
                                    )))
                                }
                            };
                            options.push((key, val));
                        }
                        self.expect(Token::RParen)?;
                        let alias = if matches!(self.current(), Token::As) {
                            self.advance();
                            Some(self.parse_identifier()?)
                        } else if let Token::Identifier(a) = self.current().clone() {
                            if a.len() >= 4 && self.is_likely_misspelled_keyword(&a) {
                                None
                            } else {
                                self.advance();
                                Some(a)
                            }
                        } else {
                            None
                        };
                        Some(FromItem::TableFunction {
                            func: upper,
                            file,
                            options,
                            alias,
                        })
                    } else {
                        // Check for qualified db.table syntax
                        let table = if matches!(self.current(), Token::Dot) {
                            self.advance(); // consume '.'
                            if let Token::Identifier(tbl) = self.current().clone() {
                                self.advance();
                                format!("{}.{}", table, tbl)
                            } else {
                                table
                            }
                        } else {
                            table
                        };
                        // Only consume an identifier as alias if it doesn't look like
                        // a misspelled keyword (e.g., "joinn" should NOT be an alias).
                        let alias = if let Token::Identifier(a) = self.current().clone() {
                            if a.len() >= 4 && self.is_likely_misspelled_keyword(&a) {
                                None
                            } else {
                                self.advance();
                                Some(a)
                            }
                        } else {
                            None
                        };
                        Some(FromItem::Table { table, alias })
                    }
                }
                Token::LParen => {
                    self.advance();

                    // Derived table: FROM (SELECT ... [UNION/INTERSECT/EXCEPT SELECT ...]) alias
                    if !matches!(self.current(), Token::Select) {
                        let (start, _) = self.current_span();
                        return Err(
                            self.syntax_error(start, "Expected SELECT after FROM (".to_string())
                        );
                    }
                    let sub = self.parse_select_part()?;
                    let mut sub_sql = SqlStatement::Select(sub);
                    // Allow set-operation chains inside the derived table
                    while matches!(
                        self.current(),
                        Token::Union | Token::Intersect | Token::Except
                    ) {
                        let set_op = match self.current() {
                            Token::Union => SetOpType::Union,
                            Token::Intersect => SetOpType::Intersect,
                            Token::Except => SetOpType::Except,
                            _ => unreachable!(),
                        };
                        self.advance();
                        let all = if matches!(self.current(), Token::All) {
                            self.advance();
                            true
                        } else {
                            false
                        };
                        if !matches!(self.current(), Token::Select) {
                            let (start, _) = self.current_span();
                            return Err(self.syntax_error(
                                start,
                                "Expected SELECT after set operation".to_string(),
                            ));
                        }
                        let right = SqlStatement::Select(self.parse_select_part()?);
                        sub_sql = SqlStatement::Union(UnionStatement {
                            left: Box::new(sub_sql),
                            right: Box::new(right),
                            all,
                            set_op,
                            order_by: Vec::new(),
                            limit: None,
                            offset: None,
                        });
                    }
                    self.expect(Token::RParen)?;

                    let alias = if matches!(self.current(), Token::As) {
                        self.advance();
                        self.parse_identifier()?
                    } else if let Token::Identifier(a) = self.current().clone() {
                        self.advance();
                        a
                    } else {
                        return Err(ApexError::QueryParseError(
                            "Derived table in FROM requires an alias".to_string(),
                        ));
                    };

                    Some(FromItem::Subquery {
                        stmt: Box::new(sub_sql),
                        alias,
                    })
                }
                _ => {
                    return Err(ApexError::QueryParseError(
                        "Expected table name after FROM".to_string(),
                    ));
                }
            }
        } else {
            None
        };

        // JOIN clauses
        let mut joins: Vec<JoinClause> = Vec::new();
        loop {
            let join_type = if matches!(self.current(), Token::Join) {
                JoinType::Inner
            } else if matches!(self.current(), Token::Left) {
                self.advance();
                if matches!(self.current(), Token::Outer) {
                    self.advance();
                }
                self.expect(Token::Join)?;
                JoinType::Left
            } else if matches!(self.current(), Token::Right) {
                self.advance();
                if matches!(self.current(), Token::Outer) {
                    self.advance();
                }
                self.expect(Token::Join)?;
                JoinType::Right
            } else if matches!(self.current(), Token::Full) {
                self.advance();
                if matches!(self.current(), Token::Outer) {
                    self.advance();
                }
                self.expect(Token::Join)?;
                JoinType::Full
            } else if matches!(self.current(), Token::Inner) {
                self.advance();
                self.expect(Token::Join)?;
                JoinType::Inner
            } else if matches!(self.current(), Token::Cross) {
                self.advance();
                self.expect(Token::Join)?;
                JoinType::Cross
            } else {
                break;
            };

            if matches!(self.current(), Token::Join) {
                self.advance();
            }

            let right = match self.current().clone() {
                Token::StringLit(file) => {
                    self.advance();
                    let alias = if matches!(self.current(), Token::As) {
                        self.advance();
                        Some(self.parse_identifier()?)
                    } else if let Token::Identifier(a) = self.current().clone() {
                        if a.len() >= 4 && self.is_likely_misspelled_keyword(&a) {
                            None
                        } else {
                            self.advance();
                            Some(a)
                        }
                    } else {
                        None
                    };
                    FromItem::DirectFile { file, alias }
                }
                Token::LParen => {
                    // JOIN (SELECT ...) alias
                    self.advance(); // consume '('
                    if !matches!(self.current(), Token::Select) {
                        return Err(ApexError::QueryParseError(
                            "Expected SELECT inside JOIN subquery parentheses".to_string(),
                        ));
                    }
                    let sub = self.parse_select_internal(true)?;
                    self.expect(Token::RParen)?;
                    let alias = if matches!(self.current(), Token::As) {
                        self.advance();
                        self.parse_identifier()?
                    } else if let Token::Identifier(a) = self.current().clone() {
                        self.advance();
                        a
                    } else {
                        return Err(ApexError::QueryParseError(
                            "JOIN subquery requires an alias".to_string(),
                        ));
                    };
                    FromItem::Subquery {
                        stmt: Box::new(crate::query::SqlStatement::Select(sub)),
                        alias,
                    }
                }
                Token::Identifier(table) => {
                    self.advance();
                    let upper = table.to_uppercase();
                    // Check for table function: read_csv(...), read_parquet(...), read_json(...)
                    if matches!(upper.as_str(), "READ_CSV" | "READ_PARQUET" | "READ_JSON")
                        && matches!(self.current(), Token::LParen)
                    {
                        self.advance(); // consume '('
                        let file = match self.current().clone() {
                            Token::StringLit(s) => {
                                self.advance();
                                s
                            }
                            other => {
                                return Err(ApexError::QueryParseError(format!(
                                    "Expected file path string in {}(), got {:?}",
                                    table, other
                                )))
                            }
                        };
                        let mut options: Vec<(String, String)> = Vec::new();
                        while matches!(self.current(), Token::Comma) {
                            self.advance();
                            let key = self.parse_identifier()?.to_lowercase();
                            self.expect(Token::Eq)?;
                            let val = match self.current().clone() {
                                Token::StringLit(s) => {
                                    self.advance();
                                    s
                                }
                                Token::Identifier(s) => {
                                    self.advance();
                                    s
                                }
                                Token::IntLit(n) => {
                                    self.advance();
                                    n.to_string()
                                }
                                Token::FloatLit(f) => {
                                    self.advance();
                                    f.to_string()
                                }
                                Token::True => {
                                    self.advance();
                                    "true".to_string()
                                }
                                Token::False => {
                                    self.advance();
                                    "false".to_string()
                                }
                                other => {
                                    return Err(ApexError::QueryParseError(format!(
                                        "Expected option value, got {:?}",
                                        other
                                    )))
                                }
                            };
                            options.push((key, val));
                        }
                        self.expect(Token::RParen)?;
                        let alias = if matches!(self.current(), Token::As) {
                            self.advance();
                            Some(self.parse_identifier()?)
                        } else if let Token::Identifier(a) = self.current().clone() {
                            if a.len() >= 4 && self.is_likely_misspelled_keyword(&a) {
                                None
                            } else {
                                self.advance();
                                Some(a)
                            }
                        } else {
                            None
                        };
                        FromItem::TableFunction {
                            func: upper,
                            file,
                            options,
                            alias,
                        }
                    } else {
                        // Regular table — check for qualified db.table syntax
                        let table = if matches!(self.current(), Token::Dot) {
                            self.advance();
                            if let Token::Identifier(tbl) = self.current().clone() {
                                self.advance();
                                format!("{}.{}", table, tbl)
                            } else {
                                table
                            }
                        } else {
                            table
                        };
                        let alias = if let Token::Identifier(a) = self.current().clone() {
                            self.advance();
                            Some(a)
                        } else {
                            None
                        };
                        FromItem::Table { table, alias }
                    }
                }
                _ => {
                    return Err(ApexError::QueryParseError(
                        "Expected table name after JOIN".to_string(),
                    ));
                }
            };

            let on = if join_type == JoinType::Cross {
                // CROSS JOIN has no ON clause — use a dummy true expression
                SqlExpr::Literal(Value::Bool(true))
            } else {
                self.expect(Token::On)?;
                self.parse_expr()?
            };
            joins.push(JoinClause {
                join_type,
                right,
                on,
            });
        }

        // WHERE
        let where_clause = if matches!(self.current(), Token::Where) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        // GROUP BY (supports expressions like YEAR(date), city)
        let (group_by, group_by_exprs) = if matches!(self.current(), Token::Group) {
            self.advance();
            self.expect(Token::By)?;
            self.parse_group_by_list()?
        } else {
            (Vec::new(), Vec::new())
        };

        // HAVING
        let having = if matches!(self.current(), Token::Having) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        // ORDER BY / LIMIT / OFFSET
        let order_by = if parse_tail && matches!(self.current(), Token::Order) {
            self.advance();
            self.expect(Token::By)?;
            self.parse_order_by()?
        } else {
            Vec::new()
        };

        let limit = if parse_tail && matches!(self.current(), Token::Limit) {
            self.advance();
            if let Token::IntLit(n) = self.current().clone() {
                self.advance();
                Some(n as usize)
            } else {
                return Err(ApexError::QueryParseError(
                    "Expected number after LIMIT".to_string(),
                ));
            }
        } else {
            None
        };

        let offset = if parse_tail && matches!(self.current(), Token::Offset) {
            self.advance();
            if let Token::IntLit(n) = self.current().clone() {
                self.advance();
                Some(n as usize)
            } else {
                return Err(ApexError::QueryParseError(
                    "Expected number after OFFSET".to_string(),
                ));
            }
        } else {
            None
        };

        Ok(SelectStatement {
            distinct,
            distinct_on,
            columns,
            from,
            joins,
            where_clause,
            group_by,
            group_by_exprs,
            having,
            order_by,
            limit,
            offset,
        })
    }

    fn parse_alias_identifier(&mut self) -> Option<String> {
        let alias = match self.current() {
            Token::Identifier(name) => {
                let n = name.clone();
                self.advance();
                Some(n)
            }
            // Allow keyword aliases, e.g. COUNT(1) count, FIRST_VALUE(...) first
            Token::Count => {
                self.advance();
                Some("count".to_string())
            }
            Token::Sum => {
                self.advance();
                Some("sum".to_string())
            }
            Token::Avg => {
                self.advance();
                Some("avg".to_string())
            }
            Token::Min => {
                self.advance();
                Some("min".to_string())
            }
            Token::Max => {
                self.advance();
                Some("max".to_string())
            }
            Token::First => {
                self.advance();
                Some("first".to_string())
            }
            Token::Last => {
                self.advance();
                Some("last".to_string())
            }
            Token::Order => {
                self.advance();
                Some("order".to_string())
            }
            Token::Group => {
                self.advance();
                Some("group".to_string())
            }
            Token::By => {
                self.advance();
                Some("by".to_string())
            }
            Token::Asc => {
                self.advance();
                Some("asc".to_string())
            }
            Token::Desc => {
                self.advance();
                Some("desc".to_string())
            }
            Token::Nulls => {
                self.advance();
                Some("nulls".to_string())
            }
            Token::Is => {
                self.advance();
                Some("is".to_string())
            }
            Token::In => {
                self.advance();
                Some("in".to_string())
            }
            Token::Not => {
                self.advance();
                Some("not".to_string())
            }
            Token::Null => {
                self.advance();
                Some("null".to_string())
            }
            _ => None,
        };
        alias
    }

    /// Parse a column reference, supporting qualified names like t.col.
    ///
    /// We preserve the full qualified name (e.g. "t._id"). Execution may
    /// normalize this as needed.
    fn parse_column_ref(&mut self) -> Result<String, ApexError> {
        let mut full = if let Token::Identifier(n) = self.current().clone() {
            self.advance();
            n
        } else {
            return Err(ApexError::QueryParseError(
                "Expected column identifier".to_string(),
            ));
        };

        while matches!(self.current(), Token::Dot) {
            self.advance();
            if let Token::Identifier(n) = self.current().clone() {
                self.advance();
                full.push('.');
                full.push_str(&n);
            } else {
                return Err(ApexError::QueryParseError(
                    "Expected identifier after '.'".to_string(),
                ));
            }
        }

        Ok(full)
    }

    fn parse_select_columns(&mut self) -> Result<Vec<SelectColumn>, ApexError> {
        let mut columns = Vec::new();

        loop {
            // SELECT *, SELECT * EXCLUDE (...), SELECT * REPLACE (...)
            if matches!(self.current(), Token::Star) {
                self.advance();
                if matches!(self.current(), Token::Exclude) {
                    self.advance();
                    self.expect(Token::LParen)?;
                    let exclude_cols = self.parse_column_list()?;
                    self.expect(Token::RParen)?;
                    columns.push(SelectColumn::AllExclude(exclude_cols));
                } else if matches!(self.current(), Token::Replace) {
                    self.advance();
                    self.expect(Token::LParen)?;
                    let mut replacements = Vec::new();
                    loop {
                        let expr = self.parse_expr()?;
                        self.expect(Token::As)?;
                        let col = self.parse_identifier()?;
                        replacements.push((expr, col));
                        if !matches!(self.current(), Token::Comma) {
                            break;
                        }
                        self.advance();
                    }
                    self.expect(Token::RParen)?;
                    columns.push(SelectColumn::AllReplace(replacements));
                } else {
                    columns.push(SelectColumn::All);
                }
            }
            // Aggregate functions
            else if matches!(
                self.current(),
                Token::Count | Token::Sum | Token::Avg | Token::Min | Token::Max
            ) {
                let func = match self.current() {
                    Token::Count => AggregateFunc::Count,
                    Token::Sum => AggregateFunc::Sum,
                    Token::Avg => AggregateFunc::Avg,
                    Token::Min => AggregateFunc::Min,
                    Token::Max => AggregateFunc::Max,
                    _ => unreachable!(),
                };
                self.advance();
                self.expect(Token::LParen)?;

                let distinct = if matches!(self.current(), Token::Distinct) {
                    self.advance();
                    true
                } else {
                    false
                };

                let column = if matches!(self.current(), Token::Star) {
                    if distinct {
                        let (start, _) = self.current_span();
                        return Err(self.syntax_error(
                            start,
                            "COUNT(DISTINCT *) is not supported".to_string(),
                        ));
                    }
                    self.advance();
                    None
                } else if matches!(self.current(), Token::Identifier(_)) {
                    if distinct && func != AggregateFunc::Count {
                        let (start, _) = self.current_span();
                        return Err(self.syntax_error(
                            start,
                            "DISTINCT is only supported for COUNT".to_string(),
                        ));
                    }
                    Some(self.parse_column_ref()?)
                } else if func == AggregateFunc::Count
                    && matches!(
                        self.current(),
                        Token::IntLit(_)
                            | Token::FloatLit(_)
                            | Token::StringLit(_)
                            | Token::True
                            | Token::False
                            | Token::Null
                    )
                {
                    // COUNT(1) / COUNT(constant) are commonly used and semantically equivalent to COUNT(*)
                    // for our execution engine.
                    let arg = match self.current().clone() {
                        Token::IntLit(n) => n.to_string(),
                        Token::FloatLit(f) => f.to_string(),
                        Token::StringLit(s) => format!("'{}'", s),
                        Token::True => "true".to_string(),
                        Token::False => "false".to_string(),
                        Token::Null => "null".to_string(),
                        _ => "1".to_string(),
                    };
                    self.advance();
                    Some(arg)
                } else {
                    None
                };

                self.expect(Token::RParen)?;

                // Check if this is a window function (has OVER clause)
                if matches!(self.current(), Token::Over) {
                    // Convert aggregate to window function
                    let func_name = format!("{}", func);
                    let args: Vec<String> = column.clone().into_iter().collect();

                    self.advance(); // consume OVER
                    self.expect(Token::LParen)?;

                    let mut partition_by = Vec::new();
                    if matches!(self.current(), Token::Partition) {
                        self.advance();
                        self.expect(Token::By)?;
                        partition_by = self.parse_column_list()?;
                    }

                    let order_by = if matches!(self.current(), Token::Order) {
                        self.advance();
                        self.expect(Token::By)?;
                        self.parse_order_by()?
                    } else {
                        Vec::new()
                    };

                    self.expect(Token::RParen)?;

                    let alias = if matches!(self.current(), Token::As) {
                        self.advance();
                        self.parse_alias_identifier()
                    } else {
                        self.parse_alias_identifier()
                    };

                    columns.push(SelectColumn::WindowFunction {
                        name: func_name,
                        args,
                        partition_by,
                        order_by,
                        alias,
                    });
                } else {
                    let alias = if matches!(self.current(), Token::As) {
                        self.advance();
                        self.parse_alias_identifier()
                    } else {
                        // Allow implicit aliases: MIN(x) min_x
                        // Also allow keyword aliases like COUNT(1) count
                        self.parse_alias_identifier()
                    };

                    columns.push(SelectColumn::Aggregate {
                        func,
                        column,
                        distinct,
                        alias,
                    });
                }
            }
            // COLUMNS('regex') — DuckDB-style column selection by pattern
            else if matches!(self.current(), Token::ColumnsKw) {
                self.advance();
                self.expect(Token::LParen)?;
                let pattern = if let Token::StringLit(s) = self.current().clone() {
                    self.advance();
                    s
                } else {
                    let (start, _) = self.current_span();
                    return Err(self.syntax_error(
                        start,
                        "COLUMNS requires a string literal pattern".to_string(),
                    ));
                };
                self.expect(Token::RParen)?;
                columns.push(SelectColumn::Columns(pattern));
            }
            // Column or window function name
            else if matches!(self.current(), Token::Identifier(_)) {
                let name = self.parse_column_ref()?;

                // Only window function supported: row_number() OVER (...)
                if matches!(self.current(), Token::LParen) {
                    // Parse function call.
                    // If it is followed by OVER, treat it as a window function.
                    // Otherwise, treat it as a scalar expression in the SELECT list.
                    let func_expr = self.parse_function_call_from_name(name.clone())?;

                    if matches!(self.current(), Token::Over) {
                        // Window function: func(args...) OVER (PARTITION BY ... ORDER BY ...)
                        // Extract args from the function expression
                        let args = if let SqlExpr::Function {
                            args: func_args, ..
                        } = &func_expr
                        {
                            func_args
                                .iter()
                                .filter_map(|a| {
                                    if let SqlExpr::Column(c) = a {
                                        Some(c.clone())
                                    } else if let SqlExpr::Literal(v) = a {
                                        Some(format!("{:?}", v))
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        } else {
                            Vec::new()
                        };

                        self.advance();
                        self.expect(Token::LParen)?;

                        let mut partition_by = Vec::new();
                        if matches!(self.current(), Token::Partition) {
                            self.advance();
                            self.expect(Token::By)?;
                            partition_by = self.parse_column_list()?;
                        }

                        let order_by = if matches!(self.current(), Token::Order) {
                            self.advance();
                            self.expect(Token::By)?;
                            self.parse_order_by()?
                        } else {
                            Vec::new()
                        };

                        self.expect(Token::RParen)?;

                        let alias = if matches!(self.current(), Token::As) {
                            self.advance();
                            self.parse_alias_identifier()
                        } else {
                            self.parse_alias_identifier()
                        };

                        columns.push(SelectColumn::WindowFunction {
                            name,
                            args,
                            partition_by,
                            order_by,
                            alias,
                        });
                    } else {
                        let alias = if matches!(self.current(), Token::As) {
                            self.advance();
                            self.parse_alias_identifier()
                        } else {
                            self.parse_alias_identifier()
                        };

                        columns.push(SelectColumn::Expression {
                            expr: func_expr,
                            alias,
                        });
                    }
                } else if matches!(
                    self.current(),
                    Token::Plus | Token::Minus | Token::Star | Token::Slash | Token::Percent
                ) {
                    // Column reference followed by arithmetic operator → parse as expression
                    // Build left side as SqlExpr::Column, then continue with binary op parsing
                    let mut left = SqlExpr::Column(name);
                    // Parse add/sub level (which internally parses mul/div)
                    loop {
                        let op = match self.current() {
                            Token::Plus => Some(BinaryOperator::Add),
                            Token::Minus => Some(BinaryOperator::Sub),
                            Token::Star => Some(BinaryOperator::Mul),
                            Token::Slash => Some(BinaryOperator::Div),
                            Token::Percent => Some(BinaryOperator::Mod),
                            _ => None,
                        };
                        if let Some(op) = op {
                            self.advance();
                            let right = self.parse_unary()?;
                            left = SqlExpr::BinaryOp {
                                left: Box::new(left),
                                op,
                                right: Box::new(right),
                            };
                        } else {
                            break;
                        }
                    }
                    let alias = if matches!(self.current(), Token::As) {
                        self.advance();
                        self.parse_alias_identifier()
                    } else {
                        self.parse_alias_identifier()
                    };
                    columns.push(SelectColumn::Expression { expr: left, alias });
                } else {
                    // Regular column with optional alias
                    if matches!(self.current(), Token::As) {
                        self.advance();
                        if let Some(alias) = self.parse_alias_identifier() {
                            columns.push(SelectColumn::ColumnAlias {
                                column: name,
                                alias,
                            });
                        } else {
                            return Err(ApexError::QueryParseError(
                                "Expected alias after AS".to_string(),
                            ));
                        }
                    } else {
                        // Allow implicit aliases: col alias
                        if let Some(alias) = self.parse_alias_identifier() {
                            columns.push(SelectColumn::ColumnAlias {
                                column: name,
                                alias,
                            });
                        } else {
                            columns.push(SelectColumn::Column(name));
                        }
                    }
                }
            }
            // Handle keywords that can also be function names: LEFT, RIGHT, IF, TRUNCATE
            // Also REPLACE, EXCLUDE (when not preceded by *)
            else if matches!(
                self.current(),
                Token::Left
                    | Token::Right
                    | Token::If
                    | Token::Truncate
                    | Token::Replace
                    | Token::Exclude
                    | Token::ColumnsKw
            ) {
                let name = match self.current() {
                    Token::Left => "LEFT",
                    Token::Right => "RIGHT",
                    Token::If => "IF",
                    Token::Truncate => "TRUNCATE",
                    Token::Replace => "REPLACE",
                    Token::Exclude => "EXCLUDE",
                    Token::ColumnsKw => "COLUMNS",
                    _ => unreachable!(),
                }
                .to_string();
                self.advance();
                if matches!(self.current(), Token::LParen) {
                    let func_expr = self.parse_function_call_from_name(name)?;
                    let alias = if matches!(self.current(), Token::As) {
                        self.advance();
                        self.parse_alias_identifier()
                    } else {
                        self.parse_alias_identifier()
                    };
                    columns.push(SelectColumn::Expression {
                        expr: func_expr,
                        alias,
                    });
                } else {
                    let (start, _) = self.current_span();
                    return Err(
                        self.syntax_error(start, format!("Expected '(' after function name"))
                    );
                }
            } else {
                // Fallback: allow expression/literal select items like `SELECT 1`.
                // This is commonly used in EXISTS subqueries.
                if matches!(
                    self.current(),
                    Token::IntLit(_)
                        | Token::FloatLit(_)
                        | Token::StringLit(_)
                        | Token::True
                        | Token::False
                        | Token::Null
                        | Token::LParen
                        | Token::Exists
                        | Token::Cast
                        | Token::Case
                        | Token::Not
                        | Token::Minus
                ) {
                    let expr = self.parse_expr()?;
                    let alias = if matches!(self.current(), Token::As) {
                        self.advance();
                        self.parse_alias_identifier()
                    } else {
                        self.parse_alias_identifier()
                    };
                    columns.push(SelectColumn::Expression { expr, alias });
                } else {
                    break;
                }
            }

            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        if columns.is_empty() {
            let (start, _) = self.current_span();
            return Err(self.syntax_error(start, "Expected column list after SELECT".to_string()));
        }

        Ok(columns)
    }

    /// Parse GROUP BY list: supports both simple columns and expressions like YEAR(date)
    fn parse_group_by_list(&mut self) -> Result<(Vec<String>, Vec<Option<SqlExpr>>), ApexError> {
        let mut names = Vec::new();
        let mut exprs = Vec::new();

        loop {
            // Check if this is a function call: IDENT '(' ...
            let is_func = matches!(self.current(), Token::Identifier(_)) && {
                let next_pos = self.pos + 1;
                next_pos < self.tokens.len() && matches!(self.tokens[next_pos].token, Token::LParen)
            };

            if is_func {
                // Parse as expression (function call like YEAR(date))
                let expr = self.parse_expr()?;
                let display = Self::expr_to_display_string(&expr);
                names.push(display);
                exprs.push(Some(expr));
            } else if matches!(self.current(), Token::Identifier(_)) {
                let name = self.parse_column_ref()?;
                names.push(name);
                exprs.push(None);
            } else {
                return Err(ApexError::QueryParseError(
                    "Expected column name or expression in GROUP BY".to_string(),
                ));
            }

            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok((names, exprs))
    }

    /// Convert a SqlExpr to a display string for use as column name
    fn expr_to_display_string(expr: &SqlExpr) -> String {
        match expr {
            SqlExpr::Column(name) => name.clone(),
            SqlExpr::Function { name, args } => {
                let arg_strs: Vec<String> = args.iter().map(Self::expr_to_display_string).collect();
                format!("{}({})", name.to_uppercase(), arg_strs.join(", "))
            }
            SqlExpr::BinaryOp { left, op, right } => {
                let op_str = match op {
                    crate::query::sql_parser::BinaryOperator::Add => "+",
                    crate::query::sql_parser::BinaryOperator::Sub => "-",
                    crate::query::sql_parser::BinaryOperator::Mul => "*",
                    crate::query::sql_parser::BinaryOperator::Div => "/",
                    crate::query::sql_parser::BinaryOperator::Mod => "%",
                    _ => "?",
                };
                format!(
                    "{} {} {}",
                    Self::expr_to_display_string(left),
                    op_str,
                    Self::expr_to_display_string(right)
                )
            }
            SqlExpr::Cast { expr, data_type } => {
                format!(
                    "CAST({} AS {:?})",
                    Self::expr_to_display_string(expr),
                    data_type
                )
            }
            SqlExpr::Literal(val) => format!("{:?}", val),
            SqlExpr::Paren(inner) => format!("({})", Self::expr_to_display_string(inner)),
            _ => format!("{:?}", expr),
        }
    }

    fn parse_column_list(&mut self) -> Result<Vec<String>, ApexError> {
        let mut columns = Vec::new();

        loop {
            if matches!(self.current(), Token::Identifier(_)) {
                let name = self.parse_column_ref()?;
                columns.push(name);
            } else {
                return Err(ApexError::QueryParseError(
                    "Expected column name".to_string(),
                ));
            }

            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(columns)
    }

    fn parse_order_by(&mut self) -> Result<Vec<OrderByClause>, ApexError> {
        let mut clauses = Vec::new();

        loop {
            // Accept: identifier, or aggregate function token (SUM/COUNT/AVG/MIN/MAX) followed by (...)
            let is_agg_func = matches!(
                self.current(),
                Token::Sum | Token::Count | Token::Avg | Token::Min | Token::Max
            );
            let is_ident = matches!(self.current(), Token::Identifier(_));
            let is_int_lit = matches!(self.current(), Token::IntLit(_));
            let is_expr_start = matches!(
                self.current(),
                Token::LBracket | Token::LParen | Token::Minus | Token::FloatLit(_)
            );

            if !is_ident && !is_agg_func && !is_int_lit && !is_expr_start {
                break;
            }

            let column = if is_agg_func {
                // Parse SUM(col) / COUNT(*) / etc. as a string for the ORDER BY clause
                let func_name = match self.current() {
                    Token::Sum => "SUM",
                    Token::Count => "COUNT",
                    Token::Avg => "AVG",
                    Token::Min => "MIN",
                    Token::Max => "MAX",
                    _ => unreachable!(),
                }
                .to_string();
                self.advance();
                // Consume parenthesised argument(s) as opaque text
                let mut depth = 0usize;
                let mut arg = String::new();
                if matches!(self.current(), Token::LParen) {
                    self.advance();
                    depth = 1;
                    while depth > 0 {
                        match self.current() {
                            Token::LParen => {
                                depth += 1;
                                arg.push('(');
                                self.advance();
                            }
                            Token::RParen => {
                                depth -= 1;
                                if depth > 0 {
                                    arg.push(')');
                                }
                                self.advance();
                            }
                            Token::Star => {
                                arg.push('*');
                                self.advance();
                            }
                            Token::Identifier(s) => {
                                arg.push_str(s);
                                self.advance();
                            }
                            Token::Comma => {
                                arg.push(',');
                                self.advance();
                            }
                            _ => {
                                self.advance();
                            }
                        }
                    }
                }
                format!("{}({})", func_name, arg.trim())
            } else if is_int_lit {
                // Positional ORDER BY like ORDER BY 1
                if let Token::IntLit(n) = self.current() {
                    let s = n.to_string();
                    self.advance();
                    s
                } else {
                    break;
                }
            } else if is_expr_start {
                // Expression that starts with '[', '(', '-', or a float literal
                let expr = self.parse_expr()?;
                let display = Self::expr_to_display_string(&expr);
                let descending = if matches!(self.current(), Token::Desc) {
                    self.advance();
                    true
                } else if matches!(self.current(), Token::Asc) {
                    self.advance();
                    false
                } else {
                    false
                };
                let nulls_first = if matches!(self.current(), Token::Nulls) {
                    self.advance();
                    if matches!(self.current(), Token::First) {
                        self.advance();
                        Some(true)
                    } else if matches!(self.current(), Token::Last) {
                        self.advance();
                        Some(false)
                    } else {
                        None
                    }
                } else {
                    None
                };
                clauses.push(OrderByClause {
                    column: display.clone(),
                    descending,
                    nulls_first,
                    expr: Some(expr),
                });
                if matches!(self.current(), Token::Comma) {
                    self.advance();
                } else {
                    break;
                }
                continue;
            } else {
                // Identifier — check if it's followed by '(' (function call expression)
                let saved_pos = self.pos;
                let col = self.parse_column_ref()?;
                if matches!(self.current(), Token::LParen) {
                    // Restore and re-parse as full expression
                    self.pos = saved_pos;
                    let expr = self.parse_expr()?;
                    let display = Self::expr_to_display_string(&expr);
                    let descending = if matches!(self.current(), Token::Desc) {
                        self.advance();
                        true
                    } else if matches!(self.current(), Token::Asc) {
                        self.advance();
                        false
                    } else {
                        false
                    };
                    let nulls_first = if matches!(self.current(), Token::Nulls) {
                        self.advance();
                        if matches!(self.current(), Token::First) {
                            self.advance();
                            Some(true)
                        } else if matches!(self.current(), Token::Last) {
                            self.advance();
                            Some(false)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    clauses.push(OrderByClause {
                        column: display.clone(),
                        descending,
                        nulls_first,
                        expr: Some(expr),
                    });
                    if matches!(self.current(), Token::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                    continue;
                }
                col
            };

            let descending = if matches!(self.current(), Token::Desc) {
                self.advance();
                true
            } else if matches!(self.current(), Token::Asc) {
                self.advance();
                false
            } else {
                false
            };

            // SQL:2023 NULLS FIRST/LAST
            let nulls_first = if matches!(self.current(), Token::Nulls) {
                self.advance();
                if matches!(self.current(), Token::First) {
                    self.advance();
                    Some(true)
                } else if matches!(self.current(), Token::Last) {
                    self.advance();
                    Some(false)
                } else {
                    None
                }
            } else {
                None
            };

            clauses.push(OrderByClause {
                column,
                descending,
                nulls_first,
                expr: None,
            });

            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(clauses)
    }

    fn parse_expr(&mut self) -> Result<SqlExpr, ApexError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<SqlExpr, ApexError> {
        let mut left = self.parse_and()?;
        while matches!(self.current(), Token::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = SqlExpr::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::Or,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<SqlExpr, ApexError> {
        match self.current() {
            Token::Minus => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(SqlExpr::UnaryOp {
                    op: UnaryOperator::Minus,
                    expr: Box::new(expr),
                })
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_literal_value(&mut self) -> Result<Value, ApexError> {
        match self.current().clone() {
            Token::StringLit(s) => {
                self.advance();
                Ok(Value::String(s))
            }
            Token::IntLit(n) => {
                self.advance();
                Ok(Value::Int64(n))
            }
            Token::FloatLit(f) => {
                self.advance();
                Ok(Value::Float64(f))
            }
            Token::True => {
                self.advance();
                Ok(Value::Bool(true))
            }
            Token::False => {
                self.advance();
                Ok(Value::Bool(false))
            }
            Token::Null => {
                self.advance();
                Ok(Value::Null)
            }
            Token::Minus => {
                self.advance();
                match self.current().clone() {
                    Token::IntLit(n) => {
                        self.advance();
                        Ok(Value::Int64(-n))
                    }
                    Token::FloatLit(f) => {
                        self.advance();
                        Ok(Value::Float64(-f))
                    }
                    other => Err(ApexError::QueryParseError(format!(
                        "Expected number after '-', got {:?}",
                        other
                    ))),
                }
            }
            // Array literal [f1, f2, ...] → little-endian float32 binary (vector)
            Token::LBracket => {
                self.advance();
                let mut floats: Vec<f32> = Vec::new();
                loop {
                    if matches!(self.current(), Token::RBracket) {
                        break;
                    }
                    let sign = if matches!(self.current(), Token::Minus) {
                        self.advance();
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    let v = match self.current().clone() {
                        Token::FloatLit(f) => {
                            self.advance();
                            f as f32
                        }
                        Token::IntLit(n) => {
                            self.advance();
                            n as f32
                        }
                        other => {
                            return Err(ApexError::QueryParseError(format!(
                                "Expected number in array literal, got {:?}",
                                other
                            )))
                        }
                    };
                    floats.push(sign * v);
                    if matches!(self.current(), Token::Comma) {
                        self.advance();
                    }
                }
                self.expect(Token::RBracket)?;
                let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
                Ok(Value::Binary(bytes))
            }
            other => Err(ApexError::QueryParseError(format!(
                "Expected literal value, got {:?}",
                other
            ))),
        }
    }

    fn parse_function_call_from_name(&mut self, name: String) -> Result<SqlExpr, ApexError> {
        self.expect(Token::LParen)?;
        let upper = name.to_uppercase();

        // topk_distance(col, [q1,q2,...], k, 'metric')
        if upper == "TOPK_DISTANCE" {
            let col = self.parse_identifier()?;
            self.expect(Token::Comma)?;
            let query = match self.parse_expr()? {
                SqlExpr::ArrayLiteral(v) => v,
                _ => {
                    return Err(ApexError::QueryParseError(
                        "topk_distance: second argument must be an array literal [f1, f2, ...]"
                            .to_string(),
                    ))
                }
            };
            self.expect(Token::Comma)?;
            let k = match self.current().clone() {
                Token::IntLit(n) => {
                    self.advance();
                    n as usize
                }
                other => {
                    return Err(ApexError::QueryParseError(format!(
                        "topk_distance: third argument must be integer k, got {:?}",
                        other
                    )))
                }
            };
            self.expect(Token::Comma)?;
            let metric = match self.current().clone() {
                Token::StringLit(s) => {
                    self.advance();
                    s
                }
                other => {
                    return Err(ApexError::QueryParseError(format!(
                        "topk_distance: fourth argument must be a metric string, got {:?}",
                        other
                    )))
                }
            };
            self.expect(Token::RParen)?;
            return Ok(SqlExpr::TopkDistance {
                col,
                query,
                k,
                metric,
            });
        }

        // explode_rename(topk_expr, "name1", "name2", ...)
        if upper == "EXPLODE_RENAME" {
            let inner = self.parse_expr()?;
            let mut names: Vec<String> = Vec::new();
            while matches!(self.current(), Token::Comma) {
                self.advance();
                match self.current().clone() {
                    Token::StringLit(s) => {
                        self.advance();
                        names.push(s);
                    }
                    other => {
                        return Err(ApexError::QueryParseError(format!(
                            "explode_rename: column names must be string literals, got {:?}",
                            other
                        )))
                    }
                }
            }
            if names.len() < 2 {
                return Err(ApexError::QueryParseError(
                    "explode_rename: requires at least 2 column name arguments".to_string(),
                ));
            }
            self.expect(Token::RParen)?;
            return Ok(SqlExpr::ExplodeRename {
                inner: Box::new(inner),
                names,
            });
        }

        let mut args = Vec::new();
        // Special-case COUNT(*) in expression contexts (e.g. HAVING COUNT(*) > 1).
        // In SELECT list we have separate aggregate parsing that already handles COUNT(*),
        // but expressions go through this generic function-call parser.
        if matches!(self.current(), Token::Star) && name.eq_ignore_ascii_case("count") {
            self.advance();
            self.expect(Token::RParen)?;
            return Ok(SqlExpr::Function { name, args });
        }

        if !matches!(self.current(), Token::RParen) {
            loop {
                args.push(self.parse_expr()?);
                if matches!(self.current(), Token::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(Token::RParen)?;
        let mut result = SqlExpr::Function { name, args };

        // Check for array indexing: func(...)[index]
        result = self.parse_array_index(result)?;
        Ok(result)
    }

    /// Parse array index suffix: expr[index]
    fn parse_array_index(&mut self, mut expr: SqlExpr) -> Result<SqlExpr, ApexError> {
        while matches!(self.current(), Token::LBracket) {
            self.advance(); // consume [
            let index = self.parse_expr()?;
            self.expect(Token::RBracket)?;
            expr = SqlExpr::ArrayIndex {
                array: Box::new(expr),
                index: Box::new(index),
            };
        }
        Ok(expr)
    }

    fn parse_and(&mut self) -> Result<SqlExpr, ApexError> {
        let mut left = self.parse_not()?;
        while matches!(self.current(), Token::And) {
            self.advance();
            let right = self.parse_not()?;
            left = SqlExpr::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::And,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<SqlExpr, ApexError> {
        if matches!(self.current(), Token::Not) {
            self.advance();
            let expr = self.parse_not()?;
            return Ok(SqlExpr::UnaryOp {
                op: UnaryOperator::Not,
                expr: Box::new(expr),
            });
        }
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<SqlExpr, ApexError> {
        let left = self.parse_add_sub()?;

        // Special forms only supported when left is a column
        let left_col = if let SqlExpr::Column(ref c) = left {
            Some(c.clone())
        } else {
            None
        };

        // Support infix negation: <col> NOT LIKE/IN/BETWEEN/REGEXP ...
        // Note: Unary NOT is already handled in parse_not(). If we see NOT here,
        // it must be part of one of the supported infix forms.
        let mut negated = false;
        if matches!(self.current(), Token::Not) {
            self.advance();
            negated = true;
        }

        if matches!(self.current(), Token::Like) {
            let column = left_col.ok_or_else(|| {
                ApexError::QueryParseError("LIKE requires column on left side".to_string())
            })?;
            self.advance();
            let pattern = match self.current().clone() {
                Token::StringLit(s) => {
                    self.advance();
                    s
                }
                Token::Identifier(s) => {
                    // Support double-quoted patterns like LIKE "foo%" which tokenize as Identifier.
                    self.advance();
                    s
                }
                _ => {
                    return Err(ApexError::QueryParseError(
                        "LIKE pattern must be a string literal".to_string(),
                    ))
                }
            };
            return Ok(SqlExpr::Like {
                column,
                pattern,
                negated,
            });
        }

        if matches!(self.current(), Token::Regexp) {
            let column = left_col.ok_or_else(|| {
                ApexError::QueryParseError("REGEXP requires column on left side".to_string())
            })?;
            self.advance();
            let pattern = match self.current().clone() {
                Token::StringLit(s) => {
                    self.advance();
                    s
                }
                Token::Identifier(s) => {
                    // Support double-quoted patterns like REGEXP "test*" which tokenize as Identifier.
                    self.advance();
                    s
                }
                _ => {
                    return Err(ApexError::QueryParseError(
                        "REGEXP pattern must be a string literal".to_string(),
                    ))
                }
            };
            return Ok(SqlExpr::Regexp {
                column,
                pattern,
                negated,
            });
        }

        if matches!(self.current(), Token::In) {
            let column = left_col.ok_or_else(|| {
                ApexError::QueryParseError("IN requires column on left side".to_string())
            })?;
            self.advance();
            self.expect(Token::LParen)?;
            if matches!(self.current(), Token::Select) {
                let sub = self.parse_select_internal(true)?;
                self.expect(Token::RParen)?;
                return Ok(SqlExpr::InSubquery {
                    column,
                    stmt: Box::new(sub),
                    negated,
                });
            }
            let mut values = Vec::new();
            loop {
                match self.current() {
                    Token::StringLit(_)
                    | Token::IntLit(_)
                    | Token::FloatLit(_)
                    | Token::True
                    | Token::False
                    | Token::Null => {
                        values.push(self.parse_literal_value()?);
                    }
                    _ => break,
                }
                if matches!(self.current(), Token::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect(Token::RParen)?;
            return Ok(SqlExpr::In {
                column,
                values,
                negated,
            });
        }

        if matches!(self.current(), Token::Between) {
            let column = left_col.ok_or_else(|| {
                ApexError::QueryParseError("BETWEEN requires column on left side".to_string())
            })?;
            self.advance();
            let low = Box::new(self.parse_add_sub()?);
            self.expect(Token::And)?;
            let high = Box::new(self.parse_add_sub()?);
            return Ok(SqlExpr::Between {
                column,
                low,
                high,
                negated,
            });
        }

        if matches!(self.current(), Token::Is) {
            let column = left_col.ok_or_else(|| {
                ApexError::QueryParseError("IS NULL requires column on left side".to_string())
            })?;
            self.advance();
            let negated = if matches!(self.current(), Token::Not) {
                self.advance();
                true
            } else {
                false
            };
            self.expect(Token::Null)?;
            return Ok(SqlExpr::IsNull { column, negated });
        }

        if negated {
            let (start, _) = self.current_span();
            return Err(self.syntax_error(
                start,
                "Expected LIKE/IN/BETWEEN/REGEXP after NOT".to_string(),
            ));
        }

        let op = match self.current() {
            Token::Eq => Some(BinaryOperator::Eq),
            Token::NotEq => Some(BinaryOperator::NotEq),
            Token::Lt => Some(BinaryOperator::Lt),
            Token::Le => Some(BinaryOperator::Le),
            Token::Gt => Some(BinaryOperator::Gt),
            Token::Ge => Some(BinaryOperator::Ge),
            _ => None,
        };
        if let Some(op) = op {
            self.advance();
            let right = self.parse_add_sub()?;
            return Ok(SqlExpr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            });
        }

        Ok(left)
    }

    fn parse_add_sub(&mut self) -> Result<SqlExpr, ApexError> {
        let mut left = self.parse_mul_div()?;
        loop {
            let op = match self.current() {
                Token::Plus => Some(BinaryOperator::Add),
                Token::Minus => Some(BinaryOperator::Sub),
                _ => None,
            };
            if let Some(op) = op {
                self.advance();
                let right = self.parse_mul_div()?;
                left = SqlExpr::BinaryOp {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_mul_div(&mut self) -> Result<SqlExpr, ApexError> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.current() {
                Token::Star => Some(BinaryOperator::Mul),
                Token::Slash => Some(BinaryOperator::Div),
                Token::Percent => Some(BinaryOperator::Mod),
                _ => None,
            };
            if let Some(op) = op {
                self.advance();
                let right = self.parse_unary()?;
                left = SqlExpr::BinaryOp {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_primary(&mut self) -> Result<SqlExpr, ApexError> {
        match self.current().clone() {
            Token::LParen => {
                self.advance();
                if matches!(self.current(), Token::Select) {
                    let sub = self.parse_select_internal(true)?;
                    self.expect(Token::RParen)?;
                    Ok(SqlExpr::ScalarSubquery {
                        stmt: Box::new(sub),
                    })
                } else {
                    let expr = self.parse_expr()?;
                    self.expect(Token::RParen)?;
                    Ok(SqlExpr::Paren(Box::new(expr)))
                }
            }
            Token::Exists => {
                self.advance();
                self.expect(Token::LParen)?;
                if !matches!(self.current(), Token::Select) {
                    return Err(ApexError::QueryParseError(
                        "EXISTS requires a SELECT subquery".to_string(),
                    ));
                }
                let sub = self.parse_select_internal(true)?;
                self.expect(Token::RParen)?;
                Ok(SqlExpr::ExistsSubquery {
                    stmt: Box::new(sub),
                })
            }
            Token::Cast => {
                self.advance();
                self.expect(Token::LParen)?;
                let expr = self.parse_expr()?;
                self.expect(Token::As)?;
                let ty = match self.current().clone() {
                    Token::Identifier(t) => {
                        self.advance();
                        t
                    }
                    other => {
                        return Err(ApexError::QueryParseError(format!(
                            "Expected type name after AS in CAST(), got {:?}",
                            other
                        )))
                    }
                };

                // Be conservative but tolerant here: some callers rely on `CAST(expr AS TYPE) AS alias`.
                // In rare cases we may see `AS` immediately after the type token (alias),
                // so avoid failing with a confusing "Expected RParen, got As" error.
                if matches!(self.current(), Token::RParen) {
                    self.advance();
                } else if !matches!(self.current(), Token::As) {
                    self.expect(Token::RParen)?;
                }
                Ok(SqlExpr::Cast {
                    expr: Box::new(expr),
                    data_type: DataType::from_sql_type(&ty),
                })
            }
            Token::Case => {
                self.advance();

                let mut when_then: Vec<(SqlExpr, SqlExpr)> = Vec::new();
                let mut else_expr: Option<Box<SqlExpr>> = None;

                if !matches!(self.current(), Token::When) {
                    let (start, _) = self.current_span();
                    return Err(self.syntax_error(start, "CASE must start with WHEN".to_string()));
                }

                while matches!(self.current(), Token::When) {
                    self.advance();
                    let cond = self.parse_expr()?;
                    self.expect(Token::Then)?;
                    let val = self.parse_expr()?;
                    when_then.push((cond, val));
                }

                if matches!(self.current(), Token::Else) {
                    self.advance();
                    let v = self.parse_expr()?;
                    else_expr = Some(Box::new(v));
                }

                self.expect(Token::End)?;

                Ok(SqlExpr::Case {
                    when_then,
                    else_expr,
                })
            }
            Token::StringLit(s) => {
                self.advance();
                Ok(SqlExpr::Literal(Value::String(s)))
            }
            Token::IntLit(n) => {
                self.advance();
                Ok(SqlExpr::Literal(Value::Int64(n)))
            }
            Token::FloatLit(f) => {
                self.advance();
                Ok(SqlExpr::Literal(Value::Float64(f)))
            }
            Token::True => {
                self.advance();
                Ok(SqlExpr::Literal(Value::Bool(true)))
            }
            Token::False => {
                self.advance();
                Ok(SqlExpr::Literal(Value::Bool(false)))
            }
            Token::Null => {
                self.advance();
                Ok(SqlExpr::Literal(Value::Null))
            }
            Token::LBracket => {
                // Array literal: [1.0, 2.0, 3.0]
                self.advance(); // consume '['
                let mut values: Vec<f64> = Vec::new();
                while !matches!(self.current(), Token::RBracket | Token::Eof) {
                    let neg = if matches!(self.current(), Token::Minus) {
                        self.advance();
                        true
                    } else {
                        false
                    };
                    let v = match self.current().clone() {
                        Token::FloatLit(f) => {
                            self.advance();
                            f
                        }
                        Token::IntLit(n) => {
                            self.advance();
                            n as f64
                        }
                        other => {
                            return Err(ApexError::QueryParseError(format!(
                                "Array literal must contain numbers, got {:?}",
                                other
                            )))
                        }
                    };
                    values.push(if neg { -v } else { v });
                    if matches!(self.current(), Token::Comma) {
                        self.advance();
                    }
                }
                self.expect(Token::RBracket)?;
                Ok(SqlExpr::ArrayLiteral(values))
            }
            Token::Identifier(name) => {
                self.advance();

                // Intercept MATCH('text') and FUZZY_MATCH('text') before generic function call
                if matches!(self.current(), Token::LParen) {
                    let upper = name.to_uppercase();
                    if upper == "MATCH" || upper == "FUZZY_MATCH" {
                        let fuzzy = upper == "FUZZY_MATCH";
                        self.advance(); // consume '('
                        let query = match self.current().clone() {
                            Token::StringLit(s) => {
                                self.advance();
                                s
                            }
                            other => {
                                return Err(ApexError::QueryParseError(format!(
                                    "MATCH() requires a string literal, got {:?}",
                                    other
                                )))
                            }
                        };
                        self.expect(Token::RParen)?;
                        return Ok(SqlExpr::FtsMatch { query, fuzzy });
                    }
                    return self.parse_function_call_from_name(name);
                }

                let mut full = name;
                while matches!(self.current(), Token::Dot) {
                    self.advance();
                    if let Token::Identifier(n) = self.current().clone() {
                        self.advance();
                        full.push('.');
                        full.push_str(&n);
                    } else {
                        return Err(ApexError::QueryParseError(
                            "Expected identifier after '.'".to_string(),
                        ));
                    }
                }

                Ok(SqlExpr::Column(full))
            }
            Token::Variable(name) => {
                let name = name.clone();
                self.advance();
                Ok(SqlExpr::Variable(name))
            }
            Token::Count => {
                self.advance();
                self.parse_function_call_from_name("count".to_string())
            }
            Token::Sum => {
                self.advance();
                self.parse_function_call_from_name("sum".to_string())
            }
            Token::Avg => {
                self.advance();
                self.parse_function_call_from_name("avg".to_string())
            }
            Token::Min => {
                self.advance();
                self.parse_function_call_from_name("min".to_string())
            }
            Token::Max => {
                self.advance();
                self.parse_function_call_from_name("max".to_string())
            }
            _ => Err(ApexError::QueryParseError(format!(
                "Unexpected token in expression: {:?}",
                self.current()
            ))),
        }
    }

    // ========== DDL Helper Methods ==========

    /// Parse an identifier (table name, column name, etc.)
    fn parse_identifier(&mut self) -> Result<String, ApexError> {
        match self.current().clone() {
            Token::Identifier(s) => {
                self.advance();
                Ok(s)
            }
            _ => {
                let (start, _) = self.current_span();
                Err(self.syntax_error(start, "Expected identifier".to_string()))
            }
        }
    }

    /// Parse a (possibly qualified) table name: `table` or `database.table`.
    fn parse_table_name(&mut self) -> Result<String, ApexError> {
        let name = self.parse_identifier()?;
        if matches!(self.current(), Token::Dot) {
            self.advance(); // consume '.'
            let tbl = self.parse_identifier()?;
            Ok(format!("{}.{}", name, tbl))
        } else {
            Ok(name)
        }
    }

    /// Parse IF NOT EXISTS clause
    fn parse_if_not_exists(&mut self) -> Result<bool, ApexError> {
        if matches!(self.current(), Token::If) {
            self.advance();
            self.expect(Token::Not)?;
            self.expect(Token::Exists)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Parse IF EXISTS clause
    fn parse_if_exists(&mut self) -> Result<bool, ApexError> {
        if matches!(self.current(), Token::If) {
            self.advance();
            self.expect(Token::Exists)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Parse column definitions for CREATE TABLE
    /// Supports: name TYPE [NOT NULL] [PRIMARY KEY] [UNIQUE] [DEFAULT value]
    fn parse_column_defs(&mut self) -> Result<Vec<ColumnDef>, ApexError> {
        let mut columns = Vec::new();

        loop {
            let name = self.parse_identifier()?;
            let data_type = self.parse_data_type()?;
            let constraints = self.parse_column_constraints()?;
            columns.push(ColumnDef {
                name,
                data_type,
                constraints,
            });

            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(columns)
    }

    /// Parse column constraints after the data type.
    /// Uses context-sensitive keyword detection: PRIMARY, KEY, DEFAULT are identifiers
    /// in general SQL but treated as constraint keywords here.
    fn parse_column_constraints(&mut self) -> Result<Vec<ColumnConstraintKind>, ApexError> {
        let mut constraints = Vec::new();
        loop {
            match self.current() {
                Token::Not => {
                    self.advance();
                    // NOT NULL
                    if matches!(self.current(), Token::Null) {
                        self.advance();
                        constraints.push(ColumnConstraintKind::NotNull);
                    } else {
                        let (start, _) = self.current_span();
                        return Err(self.syntax_error(start, "Expected NULL after NOT".to_string()));
                    }
                }
                Token::Identifier(s) if s.to_uppercase() == "PRIMARY" => {
                    self.advance();
                    // PRIMARY KEY
                    if matches!(self.current(), Token::Identifier(ref k) if k.to_uppercase() == "KEY")
                    {
                        self.advance();
                        constraints.push(ColumnConstraintKind::PrimaryKey);
                        // PK implies NOT NULL
                        if !constraints.contains(&ColumnConstraintKind::NotNull) {
                            constraints.push(ColumnConstraintKind::NotNull);
                        }
                    } else {
                        let (start, _) = self.current_span();
                        return Err(
                            self.syntax_error(start, "Expected KEY after PRIMARY".to_string())
                        );
                    }
                }
                Token::Unique => {
                    self.advance();
                    constraints.push(ColumnConstraintKind::Unique);
                }
                Token::Identifier(s) if s.to_uppercase() == "CHECK" => {
                    self.advance();
                    // CHECK ( expression ) — capture raw SQL text between parens
                    if !matches!(self.current(), Token::LParen) {
                        let (start, _) = self.current_span();
                        return Err(
                            self.syntax_error(start, "Expected '(' after CHECK".to_string())
                        );
                    }
                    let (paren_start, _) = self.current_span();
                    self.advance(); // consume '('
                    let (expr_start, _) = self.current_span();
                    // Skip tokens until matching ')' (handle nested parens)
                    let mut depth = 1u32;
                    while depth > 0 {
                        match self.current() {
                            Token::LParen => {
                                depth += 1;
                                self.advance();
                            }
                            Token::RParen => {
                                depth -= 1;
                                if depth > 0 {
                                    self.advance();
                                }
                            }
                            Token::Eof => {
                                return Err(self.syntax_error(
                                    paren_start,
                                    "Unclosed CHECK expression".to_string(),
                                ));
                            }
                            _ => {
                                self.advance();
                            }
                        }
                    }
                    let (expr_end, _) = self.current_span();
                    // Extract the SQL text between '(' and ')' from sql_chars
                    self.ensure_chars();
                    let check_sql: String = self.sql_chars.as_ref().unwrap()[expr_start..expr_end]
                        .iter()
                        .collect::<String>()
                        .trim()
                        .to_string();
                    self.advance(); // consume final ')'
                    constraints.push(ColumnConstraintKind::Check(check_sql));
                }
                Token::Identifier(s) if s.to_uppercase() == "REFERENCES" => {
                    self.advance();
                    // REFERENCES table_name(column_name)
                    let ref_table = self.parse_identifier()?;
                    if !matches!(self.current(), Token::LParen) {
                        let (start, _) = self.current_span();
                        return Err(self.syntax_error(
                            start,
                            "Expected '(' after REFERENCES table".to_string(),
                        ));
                    }
                    self.advance(); // consume '('
                    let ref_column = self.parse_identifier()?;
                    if !matches!(self.current(), Token::RParen) {
                        let (start, _) = self.current_span();
                        return Err(self.syntax_error(
                            start,
                            "Expected ')' after REFERENCES column".to_string(),
                        ));
                    }
                    self.advance(); // consume ')'
                    constraints.push(ColumnConstraintKind::ForeignKey {
                        ref_table,
                        ref_column,
                    });
                }
                Token::Identifier(s) if s.to_uppercase() == "DEFAULT" => {
                    self.advance();
                    // Parse default value (literal)
                    let val = match self.current().clone() {
                        Token::IntLit(n) => {
                            self.advance();
                            Value::Int64(n)
                        }
                        Token::FloatLit(f) => {
                            self.advance();
                            Value::Float64(f)
                        }
                        Token::StringLit(s) => {
                            self.advance();
                            Value::String(s)
                        }
                        Token::True => {
                            self.advance();
                            Value::Bool(true)
                        }
                        Token::False => {
                            self.advance();
                            Value::Bool(false)
                        }
                        Token::Null => {
                            self.advance();
                            Value::Null
                        }
                        _ => {
                            let (start, _) = self.current_span();
                            return Err(self.syntax_error(
                                start,
                                "Expected literal value after DEFAULT".to_string(),
                            ));
                        }
                    };
                    constraints.push(ColumnConstraintKind::Default(val));
                }
                Token::Identifier(s)
                    if s.to_uppercase() == "AUTOINCREMENT"
                        || s.to_uppercase() == "AUTO_INCREMENT" =>
                {
                    self.advance();
                    constraints.push(ColumnConstraintKind::Autoincrement);
                }
                _ => break,
            }
        }
        Ok(constraints)
    }

    /// Parse data type for column definition
    fn parse_data_type(&mut self) -> Result<DataType, ApexError> {
        match self.current().clone() {
            Token::Identifier(s) => {
                self.advance();
                match s.to_uppercase().as_str() {
                    "TINYINT" | "INT1" => Ok(DataType::Int8),
                    "SMALLINT" | "INT2" => Ok(DataType::Int16),
                    "INT" | "INT4" | "INTEGER" => Ok(DataType::Int32),
                    "BIGINT" | "INT64" => Ok(DataType::Int64),
                    "UTINYINT" => Ok(DataType::UInt8),
                    "USMALLINT" => Ok(DataType::UInt16),
                    "UINTEGER" => Ok(DataType::UInt32),
                    "UBIGINT" => Ok(DataType::UInt64),
                    "FLOAT" | "FLOAT32" => Ok(DataType::Float32),
                    "FLOAT64" | "DOUBLE" | "REAL" => Ok(DataType::Float64),
                    "STRING" | "TEXT" | "VARCHAR" => Ok(DataType::String),
                    "BOOL" | "BOOLEAN" => Ok(DataType::Bool),
                    "BYTES" | "BLOB" | "BINARY" | "VARBINARY" | "BYTEA" => Ok(DataType::Binary),
                    "JSON" => Ok(DataType::Json),
                    "DECIMAL" | "NUMERIC" => {
                        // Skip optional (precision, scale) parameters
                        if matches!(self.current(), Token::LParen) {
                            self.advance(); // skip '('
                                            // precision
                            if matches!(self.current(), Token::IntLit(_)) {
                                self.advance();
                            }
                            // optional comma + scale
                            if matches!(self.current(), Token::Comma) {
                                self.advance();
                                if matches!(self.current(), Token::IntLit(_)) {
                                    self.advance();
                                }
                            }
                            self.expect(Token::RParen)?;
                        }
                        Ok(DataType::String) // Store DECIMAL as String for now (preserves precision)
                    }
                    "TIMESTAMP" | "DATETIME" => Ok(DataType::Timestamp),
                    "DATE" => Ok(DataType::Date),
                    "ARRAY" => Ok(DataType::Array),
                    "FLOAT16_VECTOR" | "FLOAT16VECTOR" | "F16_VECTOR" => {
                        Ok(DataType::Float16Vector)
                    }
                    "FLOAT_VECTOR" | "FLOATVECTOR" | "VECTOR" => Ok(DataType::Binary),
                    _ => {
                        let (start, _) = self.current_span();
                        Err(self.syntax_error(start, format!("Unknown data type: {}", s)))
                    }
                }
            }
            _ => {
                let (start, _) = self.current_span();
                Err(self.syntax_error(start, "Expected data type".to_string()))
            }
        }
    }

    /// Parse ALTER TABLE operation
    fn parse_alter_operation(&mut self) -> Result<AlterTableOp, ApexError> {
        match self.current() {
            Token::Add => {
                self.advance();
                // Optional COLUMN keyword
                if matches!(self.current(), Token::Column) {
                    self.advance();
                }
                let name = self.parse_identifier()?;
                let data_type = self.parse_data_type()?;
                Ok(AlterTableOp::AddColumn { name, data_type })
            }
            Token::Drop => {
                self.advance();
                // Optional COLUMN keyword
                if matches!(self.current(), Token::Column) {
                    self.advance();
                }
                let name = self.parse_identifier()?;
                Ok(AlterTableOp::DropColumn { name })
            }
            Token::Rename => {
                self.advance();
                // Optional COLUMN keyword
                if matches!(self.current(), Token::Column) {
                    self.advance();
                }
                let old_name = self.parse_identifier()?;
                self.expect(Token::To)?;
                let new_name = self.parse_identifier()?;
                Ok(AlterTableOp::RenameColumn { old_name, new_name })
            }
            _ => {
                let (start, _) = self.current_span();
                Err(self.syntax_error(start, "Expected ADD, DROP, or RENAME".to_string()))
            }
        }
    }

    fn parse_copy_options(
        &mut self,
        file_path: &str,
    ) -> Result<(String, Vec<(String, String)>), ApexError> {
        let mut format: Option<String> = None;
        let mut options: Vec<(String, String)> = Vec::new();
        if matches!(self.current(), Token::LParen) {
            self.advance();
            loop {
                let key = self.parse_identifier()?.to_uppercase();
                if key == "FORMAT" {
                    let fmt_val = self.parse_identifier()?.to_uppercase();
                    format = Some(fmt_val);
                } else {
                    let val = match self.current().clone() {
                        Token::StringLit(s) => {
                            self.advance();
                            s
                        }
                        Token::Identifier(s) => {
                            self.advance();
                            s
                        }
                        Token::True => {
                            self.advance();
                            "true".to_string()
                        }
                        Token::False => {
                            self.advance();
                            "false".to_string()
                        }
                        Token::IntLit(n) => {
                            self.advance();
                            n.to_string()
                        }
                        _ => "true".to_string(),
                    };
                    options.push((key.to_lowercase(), val));
                }
                if !matches!(self.current(), Token::Comma) {
                    break;
                }
                self.advance();
            }
            self.expect(Token::RParen)?;
        }

        let format = format.unwrap_or_else(|| {
            let lower = file_path.to_lowercase();
            if lower.ends_with(".csv") || lower.ends_with(".tsv") {
                "CSV".to_string()
            } else if lower.ends_with(".json")
                || lower.ends_with(".ndjson")
                || lower.ends_with(".jsonl")
            {
                "JSON".to_string()
            } else {
                "PARQUET".to_string()
            }
        });
        Ok((format, options))
    }

    /// Parse comma-separated list of identifiers
    fn parse_identifier_list(&mut self) -> Result<Vec<String>, ApexError> {
        let mut list = Vec::new();

        loop {
            list.push(self.parse_identifier()?);

            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(list)
    }

    /// Parse VALUES clause for INSERT
    fn parse_values_list(&mut self) -> Result<Vec<Vec<Value>>, ApexError> {
        let mut rows = Vec::new();

        loop {
            self.expect(Token::LParen)?;
            let mut row = Vec::new();

            loop {
                let value = self.parse_literal_value()?;
                row.push(value);

                if matches!(self.current(), Token::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }

            self.expect(Token::RParen)?;
            rows.push(row);

            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(rows)
    }

    /// Parse SET clause for UPDATE (column = value pairs)
    fn parse_assignments(&mut self) -> Result<Vec<(String, SqlExpr)>, ApexError> {
        let mut assignments = Vec::new();

        loop {
            let column = self.parse_identifier()?;
            self.expect(Token::Eq)?;
            let value = self.parse_expr()?;
            assignments.push((column, value));

            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(assignments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_select() {
        let sql = "SELECT * FROM users";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt else {
            panic!("expected select");
        };
        assert!(!s.distinct);
        assert_eq!(s.columns.len(), 1);
        assert!(matches!(s.columns[0], SelectColumn::All));
        assert!(matches!(
            s.from,
            Some(FromItem::Table {
                table,
                alias: None
            }) if table == "users"
        ));
    }

    #[test]
    fn test_select_with_where() {
        let sql = "SELECT name, age FROM users WHERE age > 18 AND name LIKE 'John%'";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt else {
            panic!("expected select");
        };
        assert_eq!(s.columns.len(), 2);
        assert!(s.where_clause.is_some());
    }

    #[test]
    fn test_select_with_order_limit() {
        let sql = "SELECT * FROM users ORDER BY age DESC LIMIT 10 OFFSET 5";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt else {
            panic!("expected select");
        };
        assert_eq!(s.order_by.len(), 1);
        assert!(s.order_by[0].descending);
        assert_eq!(s.limit, Some(10));
        assert_eq!(s.offset, Some(5));
    }

    #[test]
    fn test_select_qualified_id() {
        let sql = "SELECT default._id, name FROM default ORDER BY default._id";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt else {
            panic!("expected select");
        };
        assert_eq!(s.columns.len(), 2);
        match &s.columns[0] {
            SelectColumn::Column(c) => assert_eq!(c, "default._id"),
            other => panic!("unexpected column: {:?}", other),
        }
        match &s.columns[1] {
            SelectColumn::Column(c) => assert_eq!(c, "name"),
            other => panic!("unexpected column: {:?}", other),
        }
        assert_eq!(s.order_by.len(), 1);
        assert_eq!(s.order_by[0].column, "default._id");
    }

    #[test]
    fn test_select_quoted_id() {
        let sql = "SELECT \"_id\", name FROM default ORDER BY \"_id\"";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt else {
            panic!("expected select");
        };
        assert_eq!(s.columns.len(), 2);
        match &s.columns[0] {
            SelectColumn::Column(c) => assert_eq!(c, "_id"),
            other => panic!("unexpected column: {:?}", other),
        }
        assert_eq!(s.order_by.len(), 1);
        assert_eq!(s.order_by[0].column, "_id");
    }

    #[test]
    fn test_syntax_error_missing_select_list() {
        let err = SqlParser::parse("SELECT FROM t").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Syntax error"));
        assert!(msg.contains("Expected column list"));
    }

    #[test]
    fn test_syntax_error_unterminated_string() {
        let err = SqlParser::parse("SELECT * FROM t WHERE name = 'abc").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Unterminated string literal"));
        assert!(msg.contains("Syntax error"));
    }

    #[test]
    fn test_syntax_error_unexpected_character() {
        let err = SqlParser::parse("SELECT * FROM t WHERE a = @").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Unexpected character"));
        assert!(msg.contains("Syntax error"));
    }

    #[test]
    fn test_syntax_error_misspelled_keywords_like_froms() {
        let sql = "select * froms default wheres title likes 'Python%' limits 10";
        let err = SqlParser::parse(sql).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Syntax error"));
        assert!(
            msg.contains("did you mean FROM")
                || msg.contains("did you mean WHERE")
                || msg.contains("did you mean LIKE")
                || msg.contains("did you mean LIMIT")
        );
    }

    #[test]
    fn test_syntax_error_misspelled_select_keyword() {
        let sql = "selecte * from default";
        let err = SqlParser::parse(sql).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Syntax error"));
        assert!(msg.contains("did you mean SELECT"));
    }

    #[test]
    fn test_syntax_error_misspelled_select_keyword_selects() {
        let sql = "selects max(_id) from default";
        let err = SqlParser::parse(sql).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Syntax error"));
        assert!(msg.contains("did you mean SELECT"));
    }

    #[test]
    fn test_syntax_error_misspelled_join_keyword() {
        let sql = "select * from t1 joinn t2 on t1.id = t2.id";
        let err = SqlParser::parse(sql).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Syntax error"));
        assert!(msg.contains("did you mean JOIN"));
    }

    // ====== CTE column alias parsing ======

    #[test]
    fn test_cte_without_column_aliases() {
        let sql = "WITH cte AS (SELECT 1) SELECT * FROM cte";
        let stmt = SqlParser::parse(sql).unwrap();
        if let SqlStatement::Cte {
            name,
            column_aliases,
            recursive,
            ..
        } = stmt
        {
            assert_eq!(name, "cte");
            assert!(column_aliases.is_empty());
            assert!(!recursive);
        } else {
            panic!("Expected CTE statement");
        }
    }

    #[test]
    fn test_cte_with_single_column_alias() {
        let sql = "WITH cte(x) AS (SELECT 1) SELECT x FROM cte";
        let stmt = SqlParser::parse(sql).unwrap();
        if let SqlStatement::Cte {
            name,
            column_aliases,
            ..
        } = stmt
        {
            assert_eq!(name, "cte");
            assert_eq!(column_aliases, vec!["x"]);
        } else {
            panic!("Expected CTE statement");
        }
    }

    #[test]
    fn test_cte_with_multiple_column_aliases() {
        let sql = "WITH RECURSIVE fact(n, val) AS (SELECT 1, 1 UNION ALL SELECT n+1, val*(n+1) FROM fact WHERE n < 5) SELECT n, val FROM fact";
        let stmt = SqlParser::parse(sql).unwrap();
        if let SqlStatement::Cte {
            name,
            column_aliases,
            recursive,
            ..
        } = stmt
        {
            assert_eq!(name, "fact");
            assert_eq!(column_aliases, vec!["n", "val"]);
            assert!(recursive);
        } else {
            panic!("Expected CTE statement");
        }
    }

    #[test]
    fn test_cte_recursive_three_columns() {
        let sql = "WITH RECURSIVE fib(n, a, b) AS (SELECT 1, 0, 1 UNION ALL SELECT n+1, b, a+b FROM fib WHERE n < 10) SELECT n, b FROM fib";
        let stmt = SqlParser::parse(sql).unwrap();
        if let SqlStatement::Cte {
            column_aliases,
            recursive,
            ..
        } = stmt
        {
            assert_eq!(column_aliases, vec!["n", "a", "b"]);
            assert!(recursive);
        } else {
            panic!("Expected CTE statement");
        }
    }

    // ====== CHECK constraint parsing ======

    #[test]
    fn test_check_constraint_parsed() {
        let sql = "CREATE TABLE t (age INTEGER CHECK(age > 0))";
        let stmt = SqlParser::parse(sql).unwrap();
        if let SqlStatement::CreateTable { columns, .. } = stmt {
            assert_eq!(columns.len(), 1);
            let has_check = columns[0]
                .constraints
                .iter()
                .any(|c| matches!(c, ColumnConstraintKind::Check(_)));
            assert!(has_check, "CHECK constraint not parsed");
        } else {
            panic!("Expected CreateTable");
        }
    }

    #[test]
    fn test_check_constraint_expression_captured() {
        let sql = "CREATE TABLE t (val INTEGER CHECK(val >= 0 AND val < 1000))";
        let stmt = SqlParser::parse(sql).unwrap();
        if let SqlStatement::CreateTable { columns, .. } = stmt {
            let check = columns[0].constraints.iter().find_map(|c| {
                if let ColumnConstraintKind::Check(s) = c {
                    Some(s.clone())
                } else {
                    None
                }
            });
            assert!(check.is_some());
            let expr = check.unwrap();
            assert!(expr.contains("val") && expr.contains("0") && expr.contains("1000"));
        } else {
            panic!("Expected CreateTable");
        }
    }

    // ====== FOREIGN KEY constraint parsing ======

    #[test]
    fn test_fk_constraint_parsed() {
        let sql = "CREATE TABLE child (id INTEGER, parent_id INTEGER REFERENCES parent(id))";
        let stmt = SqlParser::parse(sql).unwrap();
        if let SqlStatement::CreateTable { columns, .. } = stmt {
            assert_eq!(columns.len(), 2);
            let fk = columns[1].constraints.iter().find_map(|c| {
                if let ColumnConstraintKind::ForeignKey {
                    ref_table,
                    ref_column,
                } = c
                {
                    Some((ref_table.clone(), ref_column.clone()))
                } else {
                    None
                }
            });
            assert_eq!(fk, Some(("parent".to_string(), "id".to_string())));
        } else {
            panic!("Expected CreateTable");
        }
    }

    #[test]
    fn test_fk_and_check_together() {
        let sql = "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER CHECK(val > 0) REFERENCES other(val))";
        let stmt = SqlParser::parse(sql).unwrap();
        if let SqlStatement::CreateTable { columns, .. } = stmt {
            assert_eq!(columns.len(), 2);
            let has_pk = columns[0]
                .constraints
                .iter()
                .any(|c| matches!(c, ColumnConstraintKind::PrimaryKey));
            assert!(has_pk);
            let has_check = columns[1]
                .constraints
                .iter()
                .any(|c| matches!(c, ColumnConstraintKind::Check(_)));
            let has_fk = columns[1]
                .constraints
                .iter()
                .any(|c| matches!(c, ColumnConstraintKind::ForeignKey { .. }));
            assert!(has_check);
            assert!(has_fk);
        } else {
            panic!("Expected CreateTable");
        }
    }

    #[test]
    fn test_copy_to_csv_parsed_as_export() {
        let stmt = SqlParser::parse("COPY t TO 'out.csv'").unwrap();
        match stmt {
            SqlStatement::CopyExport {
                table,
                file_path,
                format,
                options,
            } => {
                assert_eq!(table, "t");
                assert_eq!(file_path, "out.csv");
                assert_eq!(format, "CSV");
                assert!(options.is_empty());
            }
            other => panic!("Expected CopyExport, got {:?}", other),
        }
    }

    #[test]
    fn test_copy_to_json_with_options() {
        let stmt = SqlParser::parse("COPY t TO 'out.jsonl' (FORMAT JSON, HEADER false)").unwrap();
        match stmt {
            SqlStatement::CopyExport {
                format, options, ..
            } => {
                assert_eq!(format, "JSON");
                assert!(options.iter().any(|(k, v)| k == "header" && v == "false"));
            }
            other => panic!("Expected CopyExport, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_extended_numeric_and_json_types() {
        let stmt = SqlParser::parse(
            "CREATE TABLE t (a TINYINT, b SMALLINT, c UINTEGER, d FLOAT32, e JSON, f BLOB)",
        )
        .unwrap();
        if let SqlStatement::CreateTable { columns, .. } = stmt {
            assert_eq!(columns[0].data_type, DataType::Int8);
            assert_eq!(columns[1].data_type, DataType::Int16);
            assert_eq!(columns[2].data_type, DataType::UInt32);
            assert_eq!(columns[3].data_type, DataType::Float32);
            assert_eq!(columns[4].data_type, DataType::Json);
            assert_eq!(columns[5].data_type, DataType::Binary);
        } else {
            panic!("Expected CreateTable");
        }
    }

    // ====== CREATE TEMP / TEMPORARY TABLE parsing ======

    #[test]
    fn test_create_temp_table() {
        let stmt = SqlParser::parse("CREATE TEMP TABLE t (a INTEGER)").unwrap();
        if let SqlStatement::CreateTable {
            table,
            temp,
            columns,
            ..
        } = stmt
        {
            assert_eq!(table, "t");
            assert!(temp);
            assert_eq!(columns.len(), 1);
        } else {
            panic!("Expected CreateTable");
        }
    }

    #[test]
    fn test_create_temporary_table() {
        let stmt = SqlParser::parse("CREATE TEMPORARY TABLE t (a INTEGER)").unwrap();
        if let SqlStatement::CreateTable {
            table,
            temp,
            columns,
            ..
        } = stmt
        {
            assert_eq!(table, "t");
            assert!(temp);
            assert_eq!(columns.len(), 1);
        } else {
            panic!("Expected CreateTable");
        }
    }

    #[test]
    fn test_create_temp_table_as() {
        let stmt = SqlParser::parse("CREATE TEMP TABLE t AS SELECT * FROM src").unwrap();
        if let SqlStatement::CreateTableAs { table, temp, .. } = stmt {
            assert_eq!(table, "t");
            assert!(temp);
        } else {
            panic!("Expected CreateTableAs");
        }
    }

    #[test]
    fn test_create_temp_table_if_not_exists() {
        let stmt = SqlParser::parse("CREATE TEMP TABLE IF NOT EXISTS t (a INTEGER)").unwrap();
        if let SqlStatement::CreateTable {
            table,
            temp,
            if_not_exists,
            ..
        } = stmt
        {
            assert_eq!(table, "t");
            assert!(temp);
            assert!(if_not_exists);
        } else {
            panic!("Expected CreateTable");
        }
    }

    // ====== Backtick-quoted identifier parsing ======

    #[test]
    fn test_backtick_quoted_select() {
        let sql = "SELECT `order`, `group`, `select` FROM t";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt else {
            panic!("expected select");
        };
        assert_eq!(s.columns.len(), 3);
        match &s.columns[0] {
            SelectColumn::Column(c) => assert_eq!(c, "order"),
            other => panic!("unexpected column: {:?}", other),
        }
        match &s.columns[1] {
            SelectColumn::Column(c) => assert_eq!(c, "group"),
            other => panic!("unexpected column: {:?}", other),
        }
        match &s.columns[2] {
            SelectColumn::Column(c) => assert_eq!(c, "select"),
            other => panic!("unexpected column: {:?}", other),
        }
    }

    #[test]
    fn test_backtick_quoted_where() {
        let sql = "SELECT * FROM t WHERE `order` > 10";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt else {
            panic!("expected select");
        };
        assert!(s.where_clause.is_some());
    }

    #[test]
    fn test_backtick_quoted_order_by() {
        let sql = "SELECT `order` FROM t ORDER BY `order` DESC";
        let stmt = SqlParser::parse(sql).unwrap();
        let SqlStatement::Select(s) = stmt else {
            panic!("expected select");
        };
        assert_eq!(s.order_by.len(), 1);
        assert_eq!(s.order_by[0].column, "order");
        assert!(s.order_by[0].descending);
    }

    #[test]
    fn test_backtick_unterminated_error() {
        let err = SqlParser::parse("SELECT `order FROM t").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Unterminated backtick-quoted identifier"));
    }
}
