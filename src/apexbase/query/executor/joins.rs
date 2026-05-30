// JOIN execution: hash join, full outer join, cross join

impl ApexExecutor {
    /// Execute SELECT statement with JOINs
    fn execute_select_with_joins(stmt: SelectStatement, base_dir: &Path, default_table_path: &Path) -> io::Result<ApexResult> {
        // CBO: Reorder INNER JOINs by ascending right-table size (star join optimization).
        // Only when ALL joins are INNER and each ON condition references only the FROM table
        // and its own right table (no cross-JOIN column dependencies).
        let joins = Self::maybe_reorder_joins(&stmt.joins, base_dir, default_table_path);

        // Get the left (base) table - supports both Table and Subquery (VIEW)
        let mut result_batch = match &stmt.from {
            Some(FromItem::Table { table, .. }) => {
                let left_path = Self::resolve_table_path(table, base_dir, default_table_path);
                let left_backend = get_cached_backend(&left_path)?;
                left_backend.read_columns_to_arrow(None, 0, None)?
            }
            Some(FromItem::Subquery { stmt: sub_stmt, .. }) => {
                match sub_stmt.as_ref() {
                    crate::query::SqlStatement::Select(sel) => {
                        let sub_path = Self::resolve_from_table_path(sel, base_dir, default_table_path);
                        Self::execute_select_with_base_dir(sel.clone(), &sub_path, base_dir, default_table_path)?.to_record_batch()?
                    }
                    crate::query::SqlStatement::Union(u) => {
                        Self::execute_union(u.clone(), base_dir, default_table_path)?.to_record_batch()?
                    }
                    _ => return Err(io::Error::new(io::ErrorKind::InvalidInput, "Subquery must be SELECT or set operation")),
                }
            }
            Some(FromItem::TableFunction { func, file, options, .. }) => {
                Self::read_table_function(func, file, options)?
            }
            Some(FromItem::TopkDistance { col, query, k, metric, .. }) => {
                Self::execute_topk_distance(default_table_path, col, query, *k, metric)?
            }
            Some(FromItem::DirectFile { file, .. }) => Self::read_direct_file(file)?,
            None => {
                let left_backend = get_cached_backend(default_table_path)?;
                left_backend.read_columns_to_arrow(None, 0, None)?
            }
        };

        // Process each JOIN clause - supports both Table and Subquery (VIEW)
        for join_clause in &joins {
            let right_batch = match &join_clause.right {
                FromItem::Table { table, .. } => {
                    let right_path = Self::resolve_table_path(table, base_dir, default_table_path);
                    let right_backend = get_cached_backend(&right_path)?;
                    right_backend.read_columns_to_arrow(None, 0, None)?
                }
                FromItem::Subquery { stmt: sub_stmt, .. } => {
                    match sub_stmt.as_ref() {
                        crate::query::SqlStatement::Select(sel) => {
                            let sub_path = Self::resolve_from_table_path(sel, base_dir, default_table_path);
                            Self::execute_select_with_base_dir(sel.clone(), &sub_path, base_dir, default_table_path)?.to_record_batch()?
                        }
                        crate::query::SqlStatement::Union(u) => {
                            Self::execute_union(u.clone(), base_dir, default_table_path)?.to_record_batch()?
                        }
                        _ => return Err(io::Error::new(io::ErrorKind::InvalidInput, "Subquery must be SELECT or set operation")),
                    }
                }
                FromItem::TableFunction { func, file, options, .. } => {
                    Self::read_table_function(func, file, options)?
                }
                FromItem::TopkDistance { col, query, k, metric, .. } => {
                    Self::execute_topk_distance(default_table_path, col, query, *k, metric)?
                }
                FromItem::DirectFile { file, .. } => Self::read_direct_file(file)?,
            };

            // CROSS JOIN has no ON clause — use cartesian product directly
            if join_clause.join_type == JoinType::Cross {
                result_batch = Self::cross_join(&result_batch, &right_batch)?;
            } else {
                // Extract join keys from ON condition (supports AND with extra filter)
                let (left_key, right_key, left_key_qualifier, right_key_qualifier, extra_filter) =
                    Self::extract_join_keys_with_filter(&join_clause.on)?;
                let right_alias = match &join_clause.right {
                    FromItem::Table { alias, .. }
                    | FromItem::DirectFile { alias, .. }
                    | FromItem::TableFunction { alias, .. } => alias.clone(),
                    _ => None,
                };

                // Perform the join (passing right alias for self-join column naming)
                result_batch = Self::hash_join_aliased(
                    &result_batch,
                    &right_batch,
                    &left_key,
                    &right_key,
                    &join_clause.join_type,
                    right_alias.as_deref(),
                    left_key_qualifier.as_deref(),
                    right_key_qualifier.as_deref(),
                )?;

                // Apply extra ON predicates (non-equality conditions from AND)
                if let Some(filter) = extra_filter {
                    result_batch = Self::apply_filter_with_storage(&result_batch, &filter, default_table_path)?;
                }
            }
        }

        if result_batch.num_rows() == 0 {
            return Ok(ApexResult::Empty(result_batch.schema()));
        }

        // Apply WHERE filter (with storage path for subquery support)
        let filtered = if let Some(ref where_clause) = stmt.where_clause {
            Self::apply_filter_with_storage(&result_batch, where_clause, default_table_path)?
        } else {
            result_batch
        };

        if filtered.num_rows() == 0 {
            return Ok(ApexResult::Empty(filtered.schema()));
        }

        // Check for aggregation
        let has_aggregation = stmt.columns.iter().any(|col| matches!(col, SelectColumn::Aggregate { .. }));

        if has_aggregation && stmt.group_by.is_empty() {
            return Self::execute_aggregation(&filtered, &stmt);
        }

        // Handle GROUP BY with aggregation
        if has_aggregation && !stmt.group_by.is_empty() {
            return Self::execute_group_by(&filtered, &stmt);
        }

        // Apply ORDER BY
        let sorted = if !stmt.order_by.is_empty() {
            Self::apply_order_by(&filtered, &stmt.order_by)?
        } else {
            filtered
        };

        // Apply LIMIT/OFFSET
        let limited = Self::apply_limit_offset(&sorted, stmt.limit, stmt.offset)?;

        // Apply projection - pass default_table_path for scalar subqueries
        let projected = Self::apply_projection_with_storage(&limited, &stmt.columns, Some(default_table_path))?;

        // Apply DISTINCT if specified
        let result = if stmt.distinct {
            Self::deduplicate_batch(&projected)?
        } else {
            projected
        };

        Ok(ApexResult::Data(result))
    }

    /// Resolve table path from FROM clause.
    /// Delegates to resolve_table_path so that db.table qualified names are handled uniformly.
    fn resolve_from_table_path(stmt: &SelectStatement, base_dir: &Path, default_table_path: &Path) -> std::path::PathBuf {
        if let Some(FromItem::Table { table, .. }) = &stmt.from {
            let table_name = table.trim_matches('"');
            // Check if table matches the default table's file stem (unqualified fast path)
            if !table_name.contains('.') {
                if let Some(stem) = default_table_path.file_stem() {
                    if stem.to_string_lossy() == table_name {
                        return default_table_path.to_path_buf();
                    }
                }
            }
            return Self::resolve_table_path(table_name, base_dir, default_table_path);
        }
        // No FROM clause - use default_table_path
        default_table_path.to_path_buf()
    }

    /// Resolve table path from table name.
    /// Supports qualified `database.table` syntax for cross-database queries.
    /// The root directory (parent of all databases) is retrieved from the
    /// thread-local QUERY_ROOT_DIR set by Python bindings before execution.
    fn resolve_table_path(table_name: &str, base_dir: &Path, default_table_path: &Path) -> std::path::PathBuf {
        let clean_name = table_name.trim_matches('"').trim_matches('`');

        // Check temp dir first: temp tables shadow persistent tables
        let safe_check: String = clean_name.chars()
            .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
            .collect();
        let truncated_check = if safe_check.len() > 200 { &safe_check[..200] } else { &safe_check };
        if let Some(temp_dir) = crate::query::executor::get_temp_dir() {
            let temp_path = temp_dir.join(format!("{}.apex", truncated_check));
            if temp_path.exists() {
                return temp_path;
            }
        }

        // Handle qualified db.table syntax
        if let Some(dot_pos) = clean_name.find('.') {
            let db_name = clean_name[..dot_pos].trim();
            let tbl_name = clean_name[dot_pos + 1..].trim();
            let safe_tbl: String = tbl_name.chars()
                .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
                .collect();
            let safe_tbl = if safe_tbl.len() > 200 { &safe_tbl[..200] } else { &safe_tbl };

            // Determine root_dir: from thread-local if available, else base_dir.parent()
            let root_dir = crate::query::executor::get_query_root_dir()
                .unwrap_or_else(|| base_dir.parent().unwrap_or(base_dir).to_path_buf());

            let db_dir = if db_name.is_empty() || db_name.eq_ignore_ascii_case("default") {
                root_dir
            } else {
                root_dir.join(db_name)
            };
            return db_dir.join(format!("{}.apex", safe_tbl));
        }

        if clean_name == "default" {
            if default_table_path.is_dir() {
                base_dir.join("default.apex")
            } else {
                default_table_path.to_path_buf()
            }
        } else {
            let safe_name: String = clean_name.chars()
                .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
                .collect();
            let truncated_name = if safe_name.len() > 200 { &safe_name[..200] } else { &safe_name };
            base_dir.join(format!("{}.apex", truncated_name))
        }
    }

    /// Resolve the table path for a point-lookup query by extracting the FROM clause table name.
    /// Used by the QuerySignature::PointLookup pre-parse fast path.
    fn resolve_point_lookup_table_path(sql: &str, base_dir: &Path, default_table_path: &Path) -> std::path::PathBuf {
        let su = sql.trim().to_ascii_uppercase();
        if let Some(fp) = su.find(" FROM ") {
            let after_from = su[fp + 6..].trim_start();
            let tn_end = after_from.find(|c: char| c == ' ' || c == '\t' || c == '\n' || c == ';')
                .unwrap_or(after_from.len());
            let tname = after_from[..tn_end].trim_matches('"').to_lowercase();
            if !tname.is_empty() {
                let default_stem = default_table_path.file_stem()
                    .and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
                if tname == default_stem {
                    return default_table_path.to_path_buf();
                }
                return base_dir.join(format!("{}.apex", tname));
            }
        }
        default_table_path.to_path_buf()
    }

    /// CBO: Reorder INNER JOIN clauses by ascending right-table row count.
    /// Only applies when ALL joins are INNER and each ON condition is a simple
    /// equality (star join pattern — no cross-JOIN column dependencies).
    /// Returns the original order unchanged if reordering is unsafe.
    fn maybe_reorder_joins(
        joins: &[JoinClause],
        base_dir: &Path,
        default_table_path: &Path,
    ) -> Vec<JoinClause> {
        // Need 2+ joins to benefit from reordering
        if joins.len() < 2 {
            return joins.to_vec();
        }
        // Only reorder if ALL joins are INNER (LEFT/RIGHT/FULL/CROSS are order-dependent)
        if !joins.iter().all(|j| j.join_type == JoinType::Inner) {
            return joins.to_vec();
        }
        // Only reorder if all ON conditions are simple equalities (safe to reorder)
        for j in joins {
            if Self::extract_join_keys(&j.on).is_err() {
                return joins.to_vec();
            }
        }
        // Collect right-table row counts
        let mut indexed: Vec<(usize, u64)> = Vec::with_capacity(joins.len());
        for (i, j) in joins.iter().enumerate() {
            let row_count = match &j.right {
                FromItem::Table { table, .. } => {
                    let path = Self::resolve_table_path(table, base_dir, default_table_path);
                    get_cached_backend(&path)
                        .map(|b| b.active_row_count())
                        .unwrap_or(u64::MAX)
                }
                _ => u64::MAX, // Subqueries: unknown size, keep original position
            };
            indexed.push((i, row_count));
        }
        // Sort by ascending row count (smallest tables joined first)
        indexed.sort_by_key(|&(_, rows)| rows);
        let reordered: Vec<JoinClause> = indexed.iter().map(|&(i, _)| joins[i].clone()).collect();
        
        // Validate column dependencies: each JOIN's ON clause left key must reference
        // a column available from the FROM table or a previously-joined right table.
        // If reordering breaks this (e.g., ON a.col = p.col where 'a' hasn't been joined yet),
        // fall back to the original order.
        let mut available_tables: Vec<String> = Vec::new();
        // Add FROM table alias/name
        // (We don't have the FROM here, but we can detect cross-table references in ON clauses)
        for (idx, j) in reordered.iter().enumerate() {
            if let Ok((left_key_full, _)) = Self::extract_join_keys_qualified(&j.on) {
                // Check if the left key has a table qualifier that matches a right table
                // that hasn't been joined yet in the reordered sequence
                if let Some(dot_pos) = left_key_full.rfind('.') {
                    let table_prefix = &left_key_full[..dot_pos];
                    // Check if this table prefix matches any RIGHT table that comes AFTER this join
                    let is_forward_ref = reordered[idx + 1..].iter().any(|future_j| {
                        match &future_j.right {
                            FromItem::Table { table, alias, .. } => {
                                alias.as_deref().unwrap_or(table) == table_prefix
                            }
                            _ => false,
                        }
                    });
                    if is_forward_ref {
                        // Reordering broke column dependency — fall back to original order
                        return joins.to_vec();
                    }
                }
            }
            // Track joined tables
            if let FromItem::Table { table, alias, .. } = &j.right {
                available_tables.push(alias.clone().unwrap_or_else(|| table.clone()));
            }
        }
        
        reordered
    }

    /// Extract join keys preserving table qualifiers (e.g., "a.project_id" not just "project_id")
    fn extract_join_keys_qualified(on_expr: &SqlExpr) -> io::Result<(String, String)> {
        use crate::query::sql_parser::BinaryOperator;
        match on_expr {
            SqlExpr::BinaryOp { left, op: BinaryOperator::Eq, right } => {
                let left_col = match left.as_ref() {
                    SqlExpr::Column(name) => name.clone(),
                    _ => return Err(io::Error::new(io::ErrorKind::Unsupported, "JOIN key must be a column reference")),
                };
                let right_col = match right.as_ref() {
                    SqlExpr::Column(name) => name.clone(),
                    _ => return Err(io::Error::new(io::ErrorKind::Unsupported, "JOIN key must be a column reference")),
                };
                Ok((left_col, right_col))
            }
            _ => Err(io::Error::new(io::ErrorKind::Unsupported, "JOIN ON clause must be a simple equality")),
        }
    }

    /// Extract join keys from ON condition (expects simple equality: left.col = right.col)
    fn extract_join_keys(on_expr: &SqlExpr) -> io::Result<(String, String)> {
        let (left_col, right_col, _, _) = Self::extract_join_key_refs(on_expr)?;
        Ok((left_col, right_col))
    }

    /// Extract join key references from ON condition, preserving optional table qualifiers.
    fn extract_join_key_refs(on_expr: &SqlExpr) -> io::Result<(String, String, Option<String>, Option<String>)> {
        use crate::query::sql_parser::BinaryOperator;
        match on_expr {
            SqlExpr::BinaryOp { left, op: BinaryOperator::Eq, right } => {
                let (left_col, left_qualifier) = Self::extract_column_reference(left)?;
                let (right_col, right_qualifier) = Self::extract_column_reference(right)?;
                Ok((left_col, right_col, left_qualifier, right_qualifier))
            }
            _ => Err(io::Error::new(io::ErrorKind::Unsupported, "JOIN ON clause must be a simple equality")),
        }
    }

    /// Extract join keys from ON condition, returning an optional extra filter for non-equality predicates.
    /// Supports compound AND: ON a.id=b.id AND a.x<b.x → key=(id,id), filter=a.x<b.x
    fn extract_join_keys_with_filter(
        on_expr: &SqlExpr,
    ) -> io::Result<(String, String, Option<String>, Option<String>, Option<SqlExpr>)> {
        use crate::query::sql_parser::BinaryOperator;
        match on_expr {
            SqlExpr::BinaryOp { .. } if matches!(on_expr, SqlExpr::BinaryOp { op: BinaryOperator::Eq, .. }) => {
                let (left_col, right_col, left_qualifier, right_qualifier) =
                    Self::extract_join_key_refs(on_expr)?;
                Ok((left_col, right_col, left_qualifier, right_qualifier, None))
            }
            SqlExpr::BinaryOp { left, op: BinaryOperator::And, right } => {
                // Try left as equality predicate, right as extra filter
                if let Ok((lk, rk, lq, rq)) = Self::extract_join_key_refs(left) {
                    return Ok((lk, rk, lq, rq, Some(*right.clone())));
                }
                // Try right as equality predicate, left as extra filter
                if let Ok((lk, rk, lq, rq)) = Self::extract_join_key_refs(right) {
                    return Ok((lk, rk, lq, rq, Some(*left.clone())));
                }
                Err(io::Error::new(io::ErrorKind::Unsupported, "No equijoin condition found in AND ON clause"))
            }
            _ => Err(io::Error::new(io::ErrorKind::Unsupported, "JOIN ON clause must be a simple equality (e.g., a.id = b.id)")),
        }
    }

    /// Wrapper around hash_join that supports an optional right-table alias for column naming.
    /// When alias is provided, duplicate columns are named '{alias}.{col}' instead of '{col}_right'.
    fn hash_join_aliased(
        left: &RecordBatch,
        right: &RecordBatch,
        left_key: &str,
        right_key: &str,
        join_type: &JoinType,
        right_alias: Option<&str>,
        left_key_qualifier: Option<&str>,
        right_key_qualifier: Option<&str>,
    ) -> io::Result<RecordBatch> {
        let preserve_qualified_left_key =
            *join_type == JoinType::Right && left_key_qualifier.is_some();
        let preserve_qualified_right_key =
            matches!(join_type, JoinType::Right | JoinType::Full)
                && right_key_qualifier.is_some();
        if right_alias.is_none() && !preserve_qualified_left_key && !preserve_qualified_right_key {
            return Self::hash_join(left, right, left_key, right_key, join_type);
        }
        let prepared_left = if preserve_qualified_left_key {
            Some(Self::append_qualified_join_key(left, left_key, left_key_qualifier)?)
        } else {
            None
        };
        // Prepare right-side columns before joining:
        // - preserve alias-qualified conflicting columns for self-joins
        // - for RIGHT/FULL joins, keep a qualified copy of the right join key so
        //   explicit projections like `r.id` stay correct for unmatched right rows
        let prepared_right = Self::prepare_right_join_columns(
            right,
            left,
            right_key,
            right_alias,
            if preserve_qualified_right_key {
                right_key_qualifier
            } else {
                None
            },
        )?;
        let left_batch = prepared_left.as_ref().unwrap_or(left);
        Self::hash_join(left_batch, &prepared_right, left_key, right_key, join_type)
    }

    /// Prepare right-side columns for join output.
    ///
    /// - Conflicting non-key columns can be exposed as `{alias}.{col}` for self-joins.
    /// - RIGHT/FULL joins can retain a qualified copy of the right join key (for
    ///   projections like `table.id` on unmatched right rows).
    fn prepare_right_join_columns(
        right: &RecordBatch,
        left: &RecordBatch,
        join_key: &str,
        conflict_alias: Option<&str>,
        qualified_join_key: Option<&str>,
    ) -> io::Result<RecordBatch> {
        let left_name_vec: Vec<String> = left.schema().fields()
            .iter().map(|f| f.name().clone()).collect();
        let left_names: std::collections::HashSet<&str> = left_name_vec.iter().map(|s| s.as_str()).collect();
        let mut fields: Vec<arrow::datatypes::Field> = Vec::new();
        let mut arrays: Vec<arrow::array::ArrayRef> = Vec::new();
        for (i, field) in right.schema().fields().iter().enumerate() {
            let name = field.name();
            let new_name = if name != join_key && left_names.contains(name.as_str()) {
                if let Some(alias) = conflict_alias {
                    format!("{}.{}", alias, name)
                } else {
                    format!("{}_right", name)
                }
            } else {
                name.clone()
            };
            fields.push(arrow::datatypes::Field::new(&new_name, field.data_type().clone(), field.is_nullable()));
            arrays.push(right.column(i).clone());

            if name == join_key {
                if let Some(qualifier) = qualified_join_key {
                    let qualified_name = format!("{}.{}", qualifier, name);
                    if qualified_name != *name
                        && !fields.iter().any(|f| f.name() == &qualified_name)
                    {
                        fields.push(arrow::datatypes::Field::new(
                            &qualified_name,
                            field.data_type().clone(),
                            field.is_nullable(),
                        ));
                        arrays.push(right.column(i).clone());
                    }
                }
            }
        }
        let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(fields));
        RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Append a qualified copy of the join key without renaming any existing columns.
    ///
    /// RIGHT JOIN needs this for the original left key because the swap-to-LEFT path
    /// would otherwise drop it from the result schema.
    fn append_qualified_join_key(
        batch: &RecordBatch,
        join_key: &str,
        qualified_join_key: Option<&str>,
    ) -> io::Result<RecordBatch> {
        let Some(qualifier) = qualified_join_key else {
            return Ok(batch.clone());
        };
        let qualified_name = format!("{}.{}", qualifier, join_key);
        if qualified_name == join_key || batch.column_by_name(&qualified_name).is_some() {
            return Ok(batch.clone());
        }

        let join_key_idx = batch.schema().fields().iter()
            .position(|field| field.name() == join_key)
            .ok_or_else(|| io::Error::new(
                io::ErrorKind::NotFound,
                format!("Join key '{}' not found", join_key),
            ))?;

        let mut fields: Vec<arrow::datatypes::Field> = Vec::with_capacity(batch.num_columns() + 1);
        let mut arrays: Vec<arrow::array::ArrayRef> = Vec::with_capacity(batch.num_columns() + 1);
        for (i, field) in batch.schema().fields().iter().enumerate() {
            fields.push(field.as_ref().clone());
            arrays.push(batch.column(i).clone());
            if i == join_key_idx {
                fields.push(arrow::datatypes::Field::new(
                    &qualified_name,
                    field.data_type().clone(),
                    field.is_nullable(),
                ));
                arrays.push(batch.column(i).clone());
            }
        }

        let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(fields));
        RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Extract column reference from expression (handles table.column and column).
    fn extract_column_reference(expr: &SqlExpr) -> io::Result<(String, Option<String>)> {
        match expr {
            SqlExpr::Column(name) => {
                let clean_name = name.trim_matches('"');
                if let Some(dot_pos) = clean_name.rfind('.') {
                    Ok((
                        clean_name[dot_pos + 1..].to_string(),
                        Some(clean_name[..dot_pos].to_string()),
                    ))
                } else {
                    Ok((clean_name.to_string(), None))
                }
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "JOIN key must be a column reference",
            )),
        }
    }

    /// Convert expression to column name string (for display/field naming)
    fn expr_to_column_name(expr: &SqlExpr) -> String {
        match expr {
            SqlExpr::Column(name) => {
                // Handle table.column format - take the column part
                if let Some(dot_pos) = name.rfind('.') {
                    name[dot_pos + 1..].trim_matches('"').to_string()
                } else {
                    name.trim_matches('"').to_string()
                }
            }
            _ => "group".to_string(),
        }
    }

    /// Perform hash join between two RecordBatches
    fn hash_join(
        left: &RecordBatch,
        right: &RecordBatch,
        left_key: &str,
        right_key: &str,
        join_type: &JoinType,
    ) -> io::Result<RecordBatch> {
        // RIGHT JOIN → LEFT JOIN with swapped tables, then reorder columns
        if *join_type == JoinType::Right {
            let swapped = Self::hash_join(right, left, right_key, left_key, &JoinType::Left)?;
            return Self::reorder_join_columns(&swapped, left, right, right_key);
        }

        // FULL OUTER JOIN → LEFT JOIN + append unmatched right rows
        if *join_type == JoinType::Full {
            return Self::full_outer_join(left, right, left_key, right_key);
        }

        // CROSS JOIN → cartesian product
        if *join_type == JoinType::Cross {
            return Self::cross_join(left, right);
        }

        // Get key columns
        let left_key_col = left.column_by_name(left_key)
            .ok_or_else(|| io::Error::new(
                io::ErrorKind::NotFound,
                format!("Left join key '{}' not found", left_key),
            ))?;
        let right_key_col = right.column_by_name(right_key)
            .ok_or_else(|| io::Error::new(
                io::ErrorKind::NotFound,
                format!("Right join key '{}' not found", right_key),
            ))?;

        // For INNER JOIN, swap tables if right is larger (build from smaller table)
        let should_swap = !matches!(join_type, JoinType::Left) && right.num_rows() > left.num_rows() * 2;
        
        let (build_batch, probe_batch, build_key_col, probe_key_col, build_key, probe_key, swapped) = if should_swap {
            (left, right, left_key_col, right_key_col, left_key, right_key, true)
        } else {
            (right, left, right_key_col, left_key_col, right_key, left_key, false)
        };

        // Build hash table from build side (smaller table for INNER JOIN)
        let build_rows = build_batch.num_rows();
        
        // For Int64 keys, use optimized direct hash building
        let hash_table: AHashMap<u64, Vec<usize>> = if let Some(build_int_arr) = build_key_col.as_any().downcast_ref::<Int64Array>() {
            let mut table: AHashMap<u64, Vec<usize>> = AHashMap::with_capacity(build_rows);
            for i in 0..build_rows {
                if !build_int_arr.is_null(i) {
                    let val = build_int_arr.value(i);
                    let hash = {
                        let mut h = AHasher::default();
                        val.hash(&mut h);
                        h.finish()
                    };
                    table.entry(hash).or_insert_with(|| Vec::with_capacity(2)).push(i);
                }
            }
            table
        } else {
            let mut table: AHashMap<u64, Vec<usize>> = AHashMap::with_capacity(build_rows);
            for i in 0..build_rows {
                let hash = Self::hash_array_value_fast(build_key_col, i);
                table.entry(hash).or_insert_with(|| Vec::with_capacity(4)).push(i);
            }
            table
        };
        
        let right_rows = right.num_rows();

        // Probe phase - use probe_batch (which may be swapped)
        let probe_rows = probe_batch.num_rows();
        let is_left_join = matches!(join_type, JoinType::Left);
        
        // Check if key columns are Int64 (most common case) - can skip equality check
        let is_int64_key = probe_key_col.as_any().downcast_ref::<Int64Array>().is_some() 
            && build_key_col.as_any().downcast_ref::<Int64Array>().is_some();

        // For INNER JOIN with Int64 keys, use optimized path without Option overhead
        let is_inner_join_fast_path = is_int64_key && (!is_left_join || swapped);
        
        // Collect probe_idx -> build_idx mappings
        let (probe_indices, build_indices_u32, build_indices_opt): (Vec<u32>, Vec<u32>, Vec<Option<u32>>) = 
        if is_inner_join_fast_path {
            // Ultra-fast INNER JOIN path with direct u32 vectors - no Option overhead
            let probe_int_arr = probe_key_col.as_any().downcast_ref::<Int64Array>().unwrap();
            
            let est_matches = probe_rows;
            let mut probe_idx_vec: Vec<u32> = Vec::with_capacity(est_matches);
            let mut build_idx_vec: Vec<u32> = Vec::with_capacity(est_matches);
            
            for probe_idx in 0..probe_rows {
                let probe_val = probe_int_arr.value(probe_idx);
                let probe_hash = {
                    let mut h = AHasher::default();
                    probe_val.hash(&mut h);
                    h.finish()
                };
                
                if let Some(build_matches) = hash_table.get(&probe_hash) {
                    for &build_idx in build_matches {
                        probe_idx_vec.push(probe_idx as u32);
                        build_idx_vec.push(build_idx as u32);
                    }
                }
            }
            (probe_idx_vec, build_idx_vec, Vec::new())
        } else if is_int64_key {
            // LEFT JOIN with Int64 keys - needs Option for NULL handling
            let probe_int_arr = probe_key_col.as_any().downcast_ref::<Int64Array>().unwrap();
            
            let est_matches = probe_rows;
            let mut probe_idx_vec: Vec<u32> = Vec::with_capacity(est_matches);
            let mut build_idx_vec: Vec<Option<u32>> = Vec::with_capacity(est_matches);
            
            for probe_idx in 0..probe_rows {
                let probe_val = probe_int_arr.value(probe_idx);
                let probe_hash = {
                    let mut h = AHasher::default();
                    probe_val.hash(&mut h);
                    h.finish()
                };
                
                if let Some(build_matches) = hash_table.get(&probe_hash) {
                    for &build_idx in build_matches {
                        probe_idx_vec.push(probe_idx as u32);
                        build_idx_vec.push(Some(build_idx as u32));
                    }
                } else {
                    probe_idx_vec.push(probe_idx as u32);
                    build_idx_vec.push(None);
                }
            }
            (probe_idx_vec, Vec::new(), build_idx_vec)
        } else {
            let mut probe_idx_vec: Vec<u32> = Vec::with_capacity(probe_rows);
            let mut build_idx_vec: Vec<Option<u32>> = Vec::with_capacity(probe_rows);
            
            if is_left_join && !swapped {
                for probe_idx in 0..probe_rows {
                    let probe_hash = Self::hash_array_value_fast(probe_key_col, probe_idx);
                    let mut found_match = false;
                    
                    if let Some(build_matches) = hash_table.get(&probe_hash) {
                        for &build_idx in build_matches {
                            if Self::arrays_equal_at(probe_key_col, probe_idx, build_key_col, build_idx) {
                                probe_idx_vec.push(probe_idx as u32);
                                build_idx_vec.push(Some(build_idx as u32));
                                found_match = true;
                            }
                        }
                    }
                    
                    if !found_match {
                        probe_idx_vec.push(probe_idx as u32);
                        build_idx_vec.push(None);
                    }
                }
            } else {
                for probe_idx in 0..probe_rows {
                    let probe_hash = Self::hash_array_value_fast(probe_key_col, probe_idx);
                    
                    if let Some(build_matches) = hash_table.get(&probe_hash) {
                        for &build_idx in build_matches {
                            if Self::arrays_equal_at(probe_key_col, probe_idx, build_key_col, build_idx) {
                                probe_idx_vec.push(probe_idx as u32);
                                build_idx_vec.push(Some(build_idx as u32));
                            }
                        }
                    }
                }
            }
            (probe_idx_vec, Vec::new(), build_idx_vec)
        };
        
        // Convert back to left/right indices based on whether we swapped
        // For INNER JOIN fast path, use direct u32 indices
        let (left_indices, right_indices_u32, right_indices_opt): (Vec<u32>, Vec<u32>, Vec<Option<u32>>) = 
        if is_inner_join_fast_path {
            if swapped {
                (build_indices_u32.clone(), probe_indices.clone(), Vec::new())
            } else {
                (probe_indices, build_indices_u32, Vec::new())
            }
        } else if swapped {
            (build_indices_opt.iter().map(|x| x.unwrap_or(0)).collect(), 
             Vec::new(),
             probe_indices.iter().map(|x| Some(*x)).collect())
        } else {
            (probe_indices, Vec::new(), build_indices_opt)
        };
        
        let left_rows = left.num_rows();

        // Build result schema (combine left and right, avoiding duplicate key column)
        let mut fields: Vec<Field> = left.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
        for field in right.schema().fields() {
            if field.name() != right_key {
                let new_name = if fields.iter().any(|f| f.name() == field.name()) {
                    format!("{}_{}", field.name(), "right")
                } else {
                    field.name().clone()
                };
                fields.push(Field::new(&new_name, field.data_type().clone(), true));
            }
        }
        let result_schema = Arc::new(Schema::new(fields));

        // Build result columns - pre-allocate for all columns
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(left.num_columns() + right.num_columns());

        // Take from left - avoid clone by creating array directly
        let left_indices_array = arrow::array::UInt32Array::from(left_indices);
        for col in left.columns() {
            let taken = compute::take(col.as_ref(), &left_indices_array, None)
                .map_err(|e| err_data( e.to_string()))?;
            columns.push(taken);
        }

        // Take from right (excluding join key to avoid duplication)
        // Use direct u32 indices for INNER JOIN fast path (no Option overhead)
        if is_inner_join_fast_path {
            let right_indices_array = arrow::array::UInt32Array::from(right_indices_u32);
            for (col_idx, field) in right.schema().fields().iter().enumerate() {
                if field.name() != right_key {
                    let right_col = right.column(col_idx);
                    let taken = compute::take(right_col.as_ref(), &right_indices_array, None)
                        .map_err(|e| err_data( e.to_string()))?;
                    columns.push(taken);
                }
            }
        } else {
            // LEFT JOIN path - handle nulls with Option indices
            for (col_idx, field) in right.schema().fields().iter().enumerate() {
                if field.name() != right_key {
                    let right_col = right.column(col_idx);
                    let taken = Self::take_with_nulls(right_col, &right_indices_opt)?;
                    columns.push(taken);
                }
            }
        }

        RecordBatch::try_new(result_schema, columns)
            .map_err(|e| err_data( e.to_string()))
    }

    /// Reorder columns from a swapped LEFT JOIN back to original left→right order.
    /// Used by RIGHT JOIN: we did LEFT JOIN(right, left) so result has right cols first.
    /// We need to put left cols first, right cols second (excluding join key from right).
    fn reorder_join_columns(
        swapped_result: &RecordBatch,
        original_left: &RecordBatch,
        original_right: &RecordBatch,
        right_key: &str,
    ) -> io::Result<RecordBatch> {
        // swapped_result has: [right_columns..., left_columns_minus_join_key...]
        // We want:            [left_columns...(nullable), right_columns_minus_join_key...]
        let left_ncols = original_left.num_columns();
        let right_ncols = original_right.num_columns();
        // In swapped result: first right_ncols are from original_right, rest are from original_left (minus key)
        let right_cols_in_result = right_ncols;
        let left_cols_in_result = swapped_result.num_columns() - right_cols_in_result;

        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        // Left columns come from the tail of swapped_result (but they are nullable — RIGHT JOIN preserves all right rows)
        for i in 0..left_cols_in_result {
            let col_idx = right_cols_in_result + i;
            fields.push(swapped_result.schema().field(col_idx).as_ref().clone());
            arrays.push(swapped_result.column(col_idx).clone());
        }

        // Right columns (excluding join key) come from the head of swapped_result
        let swapped_schema = swapped_result.schema();
        for i in 0..right_cols_in_result {
            let field = swapped_schema.field(i);
            if field.name() == right_key { continue; }
            let new_name = if fields.iter().any(|f| f.name() == field.name()) {
                format!("{}_right", field.name())
            } else {
                field.name().clone()
            };
            fields.push(Field::new(&new_name, field.data_type().clone(), true));
            arrays.push(swapped_result.column(i).clone());
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays)
            .map_err(|e| err_data(e.to_string()))
    }

    /// FULL OUTER JOIN: LEFT JOIN + append unmatched right rows with NULL left columns
    fn full_outer_join(
        left: &RecordBatch,
        right: &RecordBatch,
        left_key: &str,
        right_key: &str,
    ) -> io::Result<RecordBatch> {
        // Step 1: Do LEFT JOIN to get all left rows + matched right rows
        let left_result = Self::hash_join(left, right, left_key, right_key, &JoinType::Left)?;

        // Step 2: Find unmatched right rows
        let left_key_col = left.column_by_name(left_key)
            .ok_or_else(|| err_data(format!("Left join key '{}' not found", left_key)))?;
        let right_key_col = right.column_by_name(right_key)
            .ok_or_else(|| err_data(format!("Right join key '{}' not found", right_key)))?;

        // Build hash set of left keys
        let mut left_key_hashes: ahash::AHashSet<u64> = ahash::AHashSet::with_capacity(left.num_rows());
        for i in 0..left.num_rows() {
            left_key_hashes.insert(Self::hash_array_value_fast(left_key_col, i));
        }

        // Find right rows whose keys don't exist in left
        let mut unmatched_right_indices: Vec<u32> = Vec::new();
        for i in 0..right.num_rows() {
            let hash = Self::hash_array_value_fast(right_key_col, i);
            if !left_key_hashes.contains(&hash) {
                unmatched_right_indices.push(i as u32);
            } else {
                // Hash match doesn't guarantee value match — verify
                let mut found = false;
                for j in 0..left.num_rows() {
                    if Self::arrays_equal_at(right_key_col, i, left_key_col, j) {
                        found = true;
                        break;
                    }
                }
                if !found {
                    unmatched_right_indices.push(i as u32);
                }
            }
        }

        if unmatched_right_indices.is_empty() {
            return Ok(left_result);
        }

        // Step 3: Build rows for unmatched right: NULL left cols + right values
        // Make all fields nullable for FULL OUTER JOIN (left side can be NULL for unmatched right rows)
        let nullable_fields: Vec<Field> = left_result.schema().fields().iter()
            .map(|f| Field::new(f.name(), f.data_type().clone(), true))
            .collect();
        let nullable_schema = Arc::new(Schema::new(nullable_fields));

        // Re-wrap left_result columns with nullable schema
        let left_result = RecordBatch::try_new(nullable_schema.clone(), left_result.columns().to_vec())
            .map_err(|e| err_data(e.to_string()))?;

        let left_ncols = left.num_columns();
        let n_unmatched = unmatched_right_indices.len();
        let right_take = arrow::array::UInt32Array::from(unmatched_right_indices);

        let mut extra_columns: Vec<ArrayRef> = Vec::with_capacity(nullable_schema.fields().len());

        // Left columns → all NULL
        for i in 0..left_ncols {
            let dt = nullable_schema.field(i).data_type();
            extra_columns.push(arrow::array::new_null_array(dt, n_unmatched));
        }

        // Right columns (excluding join key) → take from right batch
        for (col_idx, field) in right.schema().fields().iter().enumerate() {
            if field.name() == right_key { continue; }
            let taken = compute::take(right.column(col_idx).as_ref(), &right_take, None)
                .map_err(|e| err_data(e.to_string()))?;
            extra_columns.push(taken);
        }

        let extra_batch = RecordBatch::try_new(nullable_schema.clone(), extra_columns)
            .map_err(|e| err_data(e.to_string()))?;

        // Step 4: Concat left_result + extra_batch
        arrow::compute::concat_batches(&nullable_schema, &[left_result, extra_batch])
            .map_err(|e| err_data(e.to_string()))
    }

    /// CROSS JOIN: Cartesian product of two tables
    fn cross_join(left: &RecordBatch, right: &RecordBatch) -> io::Result<RecordBatch> {
        let left_rows = left.num_rows();
        let right_rows = right.num_rows();
        let total = left_rows * right_rows;

        if total == 0 {
            // Build empty schema
            let mut fields: Vec<Field> = left.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
            for field in right.schema().fields() {
                let new_name = if fields.iter().any(|f| f.name() == field.name()) {
                    format!("{}_right", field.name())
                } else {
                    field.name().clone()
                };
                fields.push(Field::new(&new_name, field.data_type().clone(), true));
            }
            let schema = Arc::new(Schema::new(fields));
            let empty_cols: Vec<ArrayRef> = schema.fields().iter().map(|f| arrow::array::new_null_array(f.data_type(), 0) as ArrayRef).collect();
            return RecordBatch::try_new(schema, empty_cols)
                .map_err(|e| err_data(e.to_string()));
        }

        // Build index arrays: left[0,0,...,1,1,...] right[0,1,2,...,0,1,2,...]
        let mut left_idx: Vec<u32> = Vec::with_capacity(total);
        let mut right_idx: Vec<u32> = Vec::with_capacity(total);
        for l in 0..left_rows {
            for r in 0..right_rows {
                left_idx.push(l as u32);
                right_idx.push(r as u32);
            }
        }

        let left_take = arrow::array::UInt32Array::from(left_idx);
        let right_take = arrow::array::UInt32Array::from(right_idx);

        let mut fields: Vec<Field> = left.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(left.num_columns() + right.num_columns());

        for col in left.columns() {
            let taken = compute::take(col.as_ref(), &left_take, None)
                .map_err(|e| err_data(e.to_string()))?;
            columns.push(taken);
        }

        for field in right.schema().fields() {
            let new_name = if fields.iter().any(|f| f.name() == field.name()) {
                format!("{}_right", field.name())
            } else {
                field.name().clone()
            };
            fields.push(Field::new(&new_name, field.data_type().clone(), true));
        }
        for col in right.columns() {
            let taken = compute::take(col.as_ref(), &right_take, None)
                .map_err(|e| err_data(e.to_string()))?;
            columns.push(taken);
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, columns)
            .map_err(|e| err_data(e.to_string()))
    }

    /// Hash a value at given index in an array (legacy, uses DefaultHasher)
    fn hash_array_value(array: &ArrayRef, idx: usize) -> u64 {
        Self::hash_array_value_fast(array, idx)
    }

    /// Fast hash using ahash (2-3x faster than DefaultHasher)
    #[inline]
    fn hash_array_value_fast(array: &ArrayRef, idx: usize) -> u64 {
        let mut hasher = AHasher::default();
        
        if array.is_null(idx) {
            0u64.hash(&mut hasher);
        } else if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
            arr.value(idx).hash(&mut hasher);
        } else if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
            arr.value(idx).hash(&mut hasher);
        } else if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
            arr.value(idx).to_bits().hash(&mut hasher);
        } else if let Some(arr) = array.as_any().downcast_ref::<BooleanArray>() {
            arr.value(idx).hash(&mut hasher);
        } else {
            idx.hash(&mut hasher);
        }
        
        hasher.finish()
    }

    /// Strip table alias prefix from column name (e.g., "o.user_id" -> "user_id")
    fn strip_table_prefix(col_name: &str) -> &str {
        if let Some(dot_pos) = col_name.find('.') {
            &col_name[dot_pos + 1..]
        } else {
            col_name
        }
    }
    
    /// Get column from batch, stripping table prefix if needed
    fn get_column_by_name<'a>(batch: &'a RecordBatch, col_name: &str) -> Option<&'a ArrayRef> {
        let clean_name = col_name.trim_matches('"');
        // Try exact match first
        if let Some(col) = batch.column_by_name(clean_name) {
            return Some(col);
        }
        // Try without table prefix (e.g., "o.user_id" -> "user_id")
        let stripped = Self::strip_table_prefix(clean_name);
        batch.column_by_name(stripped)
    }

    /// Check if two array values are equal at given indices
    fn arrays_equal_at(left: &ArrayRef, left_idx: usize, right: &ArrayRef, right_idx: usize) -> bool {
        if left.is_null(left_idx) && right.is_null(right_idx) {
            return true;
        }
        if left.is_null(left_idx) || right.is_null(right_idx) {
            return false;
        }

        if let (Some(l), Some(r)) = (
            left.as_any().downcast_ref::<Int64Array>(),
            right.as_any().downcast_ref::<Int64Array>(),
        ) {
            return l.value(left_idx) == r.value(right_idx);
        }
        if let (Some(l), Some(r)) = (
            left.as_any().downcast_ref::<StringArray>(),
            right.as_any().downcast_ref::<StringArray>(),
        ) {
            return l.value(left_idx) == r.value(right_idx);
        }
        if let (Some(l), Some(r)) = (
            left.as_any().downcast_ref::<Float64Array>(),
            right.as_any().downcast_ref::<Float64Array>(),
        ) {
            return (l.value(left_idx) - r.value(right_idx)).abs() < f64::EPSILON;
        }
        
        false
    }

    /// Take values from array with optional null indices (for LEFT JOIN)
    fn take_with_nulls(array: &ArrayRef, indices: &[Option<u32>]) -> io::Result<ArrayRef> {
        use arrow::array::*;
        
        let len = indices.len();
        
        if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
            let values: Vec<Option<i64>> = indices.iter().map(|opt_idx| {
                opt_idx.map(|idx| arr.value(idx as usize))
            }).collect();
            return Ok(Arc::new(Int64Array::from(values)));
        }
        
        if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
            let values: Vec<Option<f64>> = indices.iter().map(|opt_idx| {
                opt_idx.map(|idx| arr.value(idx as usize))
            }).collect();
            return Ok(Arc::new(Float64Array::from(values)));
        }
        
        if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
            let values: Vec<Option<&str>> = indices.iter().map(|opt_idx| {
                opt_idx.map(|idx| arr.value(idx as usize))
            }).collect();
            return Ok(Arc::new(StringArray::from(values)));
        }
        
        if let Some(arr) = array.as_any().downcast_ref::<BooleanArray>() {
            let values: Vec<Option<bool>> = indices.iter().map(|opt_idx| {
                opt_idx.map(|idx| arr.value(idx as usize))
            }).collect();
            return Ok(Arc::new(BooleanArray::from(values)));
        }

        // Fallback: create null array
        Ok(arrow::array::new_null_array(array.data_type(), len))
    }

}
