// SELECT execution: fast paths, index scans, late materialization.

/// Represents a single scannable leaf predicate from an OR decomposition.
enum OrLeafPredicate {
    StringEq(String, String),       // (col, value)
    NumericRange(String, f64, f64), // (col, low, high) — covers =, >, >=, <, <=, BETWEEN
    NumericIn(String, Vec<i64>),    // (col, values)
    StringIn(String, Vec<String>),  // (col, values)
}

impl ApexExecutor {
    /// Execute SELECT statement with base_dir for proper subquery table resolution
    fn execute_select_with_base_dir(
        mut stmt: SelectStatement,
        storage_path: &Path,
        base_dir: &Path,
        default_table_path: &Path,
    ) -> io::Result<ApexResult> {
        // Resolve MATCH()/FUZZY_MATCH() predicates to _id IN (...) before anything else
        if let Some(ref wc) = stmt.where_clause {
            if Self::expr_has_fts_match(wc) {
                let (_, table_name) = crate::query::executor::base_dir_and_table_pub(storage_path);
                let resolved = Self::resolve_fts_in_expr(
                    stmt.where_clause.take().unwrap(),
                    base_dir,
                    &table_name,
                )?;
                stmt.where_clause = Some(resolved);
            }
        }

        // FAST PATH: explode_rename(topk_distance(col,[q],k,'m'), "name1", "name2") FROM table
        // Single-pass O(n log k) topk that generates k rows with 2 user-named columns.
        if let Some((col, query, k, metric, names)) = Self::detect_topk_explode(&stmt) {
            let result = Self::execute_topk_explode(storage_path, col, query, k, metric, names)?;
            return Ok(ApexResult::Data(result));
        }

        // FAST PATH: Pure COUNT(*) without WHERE/GROUP BY - O(1) from metadata
        // Skip for TableFunction sources (read_csv/read_parquet/read_json) — no stored backend.
        // Also skip for TopkDistance — it has different row semantics.
        let from_is_table_fn = matches!(
            &stmt.from,
            Some(FromItem::TableFunction { .. })
                | Some(FromItem::TopkDistance { .. })
                | Some(FromItem::DirectFile { .. })
        );
        if !from_is_table_fn && Self::is_pure_count_star(&stmt) {
            if !storage_path.exists() {
                let tbl = storage_path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy();
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Table '{}' does not exist", tbl),
                ));
            }
            let backend = get_cached_backend(storage_path)?;
            let count = backend.active_row_count() as i64;

            let output_name =
                if let Some(SelectColumn::Aggregate { alias, .. }) = stmt.columns.first() {
                    alias.clone().unwrap_or_else(|| "COUNT(*)".to_string())
                } else {
                    "COUNT(*)".to_string()
                };
            let schema = Arc::new(Schema::new(vec![Field::new(
                &output_name,
                ArrowDataType::Int64,
                false,
            )]));
            let array: ArrayRef = Arc::new(Int64Array::from(vec![count]));
            let batch =
                RecordBatch::try_new(schema, vec![array]).map_err(|e| err_data(e.to_string()))?;
            return Ok(ApexResult::Data(batch));
        }

        // Check for derived table (FROM subquery) - resolve table path from subquery's FROM clause
        let batch = match &stmt.from {
            Some(FromItem::TopkDistance {
                col,
                query,
                k,
                metric,
                ..
            }) => Self::execute_topk_distance(storage_path, col, query, *k, metric)?,
            Some(FromItem::TableFunction {
                func,
                file,
                options,
                ..
            }) => {
                if let Some(count_batch) = Self::try_fast_json_count_table_function(&stmt, func, file)? {
                    return Ok(ApexResult::Data(count_batch));
                }
                let mut opts = options.clone();
                if let Some(ref wc) = stmt.where_clause {
                    if let Some(pushdown) = Self::try_extract_filter_for_pushdown(wc) {
                        opts.push(("filter".to_string(), pushdown));
                    }
                }
                Self::read_table_function(func, file, &opts)?
            },
            Some(FromItem::DirectFile { file, .. }) => Self::read_direct_file(file)?,
            Some(FromItem::Subquery { stmt: sub_stmt, .. }) => match sub_stmt.as_ref() {
                crate::query::SqlStatement::Select(sel) => {
                    let sub_path = Self::resolve_from_table_path(sel, base_dir, default_table_path);
                    Self::execute_select_with_base_dir(
                        sel.clone(),
                        &sub_path,
                        base_dir,
                        default_table_path,
                    )?
                    .to_record_batch()?
                }
                crate::query::SqlStatement::Union(u) => {
                    Self::execute_union(u.clone(), base_dir, default_table_path)?
                        .to_record_batch()?
                }
                _ => return Err(err_input("Subquery must be SELECT or set operation")),
            },
            None => {
                // No FROM clause (e.g., SELECT 1, 1) — create a single-row virtual batch
                let schema = Arc::new(Schema::new(vec![Field::new(
                    "_dummy",
                    ArrowDataType::Int64,
                    false,
                )]));
                RecordBatch::try_new(
                    schema,
                    vec![Arc::new(Int64Array::from(vec![0i64])) as ArrayRef],
                )
                .map_err(|e| err_data(e.to_string()))?
            }
            Some(FromItem::Table { .. }) => {
                // Normal table - read from storage
                if !storage_path.exists() {
                    let tbl = storage_path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy();
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("Table '{}' does not exist", tbl),
                    ));
                } else {
                    let backend = get_cached_backend(storage_path)?;

                    // Check if any SELECT column contains a scalar subquery
                    // Scalar subqueries may reference arbitrary columns, so read all
                    let has_scalar_subquery = stmt.columns.iter().any(|col| {
                        if let SelectColumn::Expression { expr, .. } = col {
                            Self::expr_contains_scalar_subquery(expr)
                        } else {
                            false
                        }
                    });

                    if backend.pending_v4_in_memory_rows() > 0 {
                        let col_refs = if has_scalar_subquery
                            || stmt.where_clause.is_some()
                            || backend.has_pending_deltas()
                            || backend.has_delta()
                        {
                            None
                        } else {
                            Self::get_col_refs(&stmt)
                        };
                        backend.read_columns_to_arrow(
                            col_refs
                                .as_ref()
                                .map(|v| v.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                                .as_deref(),
                            0,
                            None,
                        )?
                    } else if has_scalar_subquery {
                        let col_refs = Self::get_col_refs(&stmt);
                        backend.read_columns_to_arrow(
                            col_refs
                                .as_ref()
                                .map(|v| v.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                                .as_deref(),
                            0,
                            None,
                        )?
                    } else {
                        // Check conditions for late materialization optimization
                        let has_aggregation_check = stmt.columns.iter().any(|col| {
                            matches!(col, SelectColumn::Aggregate { .. })
                                || matches!(col, SelectColumn::Expression { expr, .. } if Self::expr_contains_aggregate(expr))
                        });

                        // FAST PATH: Direct aggregation for simple numeric aggregates
                        // Compute COUNT/SUM/AVG/MIN/MAX directly from V4 columns (mmap or in-memory)
                        if has_aggregation_check
                            && stmt.where_clause.is_none()
                            && stmt.group_by.is_empty()
                            && stmt.joins.is_empty()
                            && !backend.has_pending_deltas()
                            && !backend.has_delta()
                        {
                            if let Some(result) = Self::try_mmap_aggregation(&backend, &stmt)? {
                                return Ok(result);
                            }
                        }

                        // FAST PATH: Filtered aggregation with string equality
                        // SELECT COUNT(*), AVG(col), MAX(col) FROM table WHERE str_col = 'val'
                        if has_aggregation_check
                            && stmt.where_clause.is_some()
                            && stmt.group_by.is_empty()
                            && stmt.joins.is_empty()
                            && stmt.limit.is_none()
                            && stmt.order_by.is_empty()
                            && !backend.has_pending_deltas()
                            && !backend.has_delta()
                        {
                            if let Some(result) =
                                Self::try_fast_filtered_string_agg(&backend, &stmt)?
                            {
                                return Ok(result);
                            }
                            if let Some(result) =
                                Self::try_fast_filtered_numeric_agg(&backend, &stmt)?
                            {
                                return Ok(result);
                            }
                        }

                        // Correctness fallback for filtered aggregates when delta-backed
                        // rows are present. The generic full-batch path can miss string
                        // overlay state, but the string-filter reader already merges it.
                        if has_aggregation_check
                            && stmt.where_clause.is_some()
                            && stmt.group_by.is_empty()
                            && stmt.joins.is_empty()
                            && stmt.limit.is_none()
                            && stmt.order_by.is_empty()
                        {
                            if let Some(filtered) =
                                Self::try_fast_string_filter_no_limit(&backend, &stmt)?
                            {
                                return Self::execute_aggregation(&filtered, &stmt);
                            }
                        }

                        // Late Materialization for WHERE: with WHERE (no ORDER BY)
                        // Works for both SELECT * and projected column queries.
                        let where_cols = stmt.where_columns();
                        let has_window_func = stmt
                            .columns
                            .iter()
                            .any(|col| matches!(col, SelectColumn::WindowFunction { .. }));
                        let can_late_materialize_where = stmt.where_clause.is_some()
                            && stmt.order_by.is_empty()
                            && stmt.group_by.is_empty()
                            && !has_aggregation_check
                            && !has_window_func
                            && !where_cols.is_empty();

                        // Late Materialization for ORDER BY: SELECT * with ORDER BY + LIMIT (no WHERE)
                        let order_cols: Vec<String> = stmt
                            .order_by
                            .iter()
                            .map(|o| o.column.trim_matches('"').to_string())
                            .collect();
                        let can_late_materialize_order = stmt.is_select_star()
                            && stmt.where_clause.is_none()
                            && !stmt.order_by.is_empty()
                            && stmt.limit.is_some()
                            && stmt.group_by.is_empty()
                            && !has_aggregation_check;

                        // EARLY FAST PATH: _id = X point lookup — skip CBO entirely
                        if !has_scalar_subquery
                            && stmt.group_by.is_empty()
                            && !has_aggregation_check
                            && !backend.has_pending_deltas()
                            && !backend.has_delta()
                        {
                            if let Some(ref where_clause) = stmt.where_clause {
                                if let Some(id) = Self::extract_id_equality_filter(where_clause) {
                                    if let Some(row_batch) = backend.read_row_by_id_to_arrow(id)? {
                                        let projected = Self::apply_projection_with_storage(
                                            &row_batch,
                                            &stmt.columns,
                                            Some(storage_path),
                                        )?;
                                        return Ok(ApexResult::Data(projected));
                                    }
                                }
                            }
                        }

                        // CBO: Use plan_select_pub() to decide execution strategy.
                        // Skip expensive index checks when CBO recommends full scan or aggregation.
                        // Also skip CBO entirely for: (a) no WHERE clause, (b) table has no indexes.
                        let cbo_skip_index = if stmt.where_clause.is_none() {
                            true
                        } else {
                            let (bd, tname) = base_dir_and_table(storage_path);
                            let idx_mgr_arc = get_index_manager(&bd, &tname);
                            let idx_mgr = idx_mgr_arc.lock();
                            // Fast exit: if table has no indexes, CBO can only say full/filtered scan
                            if idx_mgr.catalog_is_empty() {
                                true
                            } else {
                                let table_key = storage_path.to_string_lossy();
                                let cbo_strategy = QueryPlanner::plan_select_pub(
                                    &stmt,
                                    Some(&*idx_mgr),
                                    &table_key,
                                );
                                matches!(
                                    cbo_strategy,
                                    ExecutionStrategy::OlapFullScan
                                        | ExecutionStrategy::OlapAggregation
                                        | ExecutionStrategy::OlapFilteredScan
                                )
                            }
                        };

                        // FAST PATH INDEX: Check if WHERE clause can use a secondary index
                        // (skipped when CBO says full scan/aggregation is cheaper)
                        if !cbo_skip_index {
                            if let Some(ref where_clause) = stmt.where_clause {
                                if let Some(result) = Self::try_index_accelerated_read(
                                    &backend,
                                    &stmt,
                                    where_clause,
                                    base_dir,
                                    storage_path,
                                )? {
                                    return Ok(result);
                                }
                            }
                        }

                        // FAST PATH 0: Check for _id = X pattern (O(1) lookup)
                        if let Some(where_clause) = &stmt.where_clause {
                            if let Some(id) = Self::extract_id_equality_filter(where_clause) {
                                if !backend.has_pending_deltas() && !backend.has_delta() {
                                    if let Some(batch) = backend.read_row_by_id_to_arrow(id)? {
                                        batch
                                    } else {
                                        // Not in memory — fall through to general mmap → Arrow → WHERE filter path
                                        let batch = backend.read_columns_to_arrow(None, 0, None)?;
                                        if batch.num_rows() == 0 {
                                            backend.read_columns_to_arrow(None, 0, Some(0))?
                                        } else {
                                            // Apply WHERE filter on the mmap-read batch
                                            let filtered = Self::apply_filter_with_storage(
                                                &batch,
                                                where_clause,
                                                storage_path,
                                            )?;
                                            if filtered.num_rows() == 0 {
                                                return Ok(ApexResult::Empty(filtered.schema()));
                                            }
                                            return Ok(ApexResult::Data(
                                                Self::apply_projection_with_storage(
                                                    &filtered,
                                                    &stmt.columns,
                                                    Some(storage_path),
                                                )?,
                                            ));
                                        }
                                    }
                                } else {
                                    // Pending DeltaStore updates must be merged through the full scan path.
                                    let batch = backend.read_columns_to_arrow(None, 0, None)?;
                                    if batch.num_rows() == 0 {
                                        backend.read_columns_to_arrow(None, 0, Some(0))?
                                    } else {
                                        let filtered = Self::apply_filter_with_storage(
                                            &batch,
                                            where_clause,
                                            storage_path,
                                        )?;
                                        if filtered.num_rows() == 0 {
                                            return Ok(ApexResult::Empty(filtered.schema()));
                                        }
                                        return Ok(ApexResult::Data(
                                            Self::apply_projection_with_storage(
                                                &filtered,
                                                &stmt.columns,
                                                Some(storage_path),
                                            )?,
                                        ));
                                    }
                                }
                            } else if let Some(result) =
                                Self::try_fast_filter_group_order(&backend, &stmt)?
                            {
                                // FAST PATH for Complex (Filter+Group+Order) - biggest optimization
                                return Ok(result);
                            } else if can_late_materialize_where {
                                // FAST PATH 1: Try dictionary-based filter for simple string equality (with LIMIT)
                                if let Some(result) = Self::try_fast_string_filter(&backend, &stmt)?
                                {
                                    result
                                // FAST PATH 1b: String equality without LIMIT - storage-level scan
                                } else if let Some(result) =
                                    Self::try_fast_string_filter_no_limit(&backend, &stmt)?
                                {
                                    if !stmt.is_pure_star() {
                                        let projected = Self::apply_projection_with_storage(
                                            &result,
                                            &stmt.columns,
                                            Some(storage_path),
                                        )?;
                                        return Ok(ApexResult::Data(projected));
                                    }
                                    return Ok(ApexResult::Data(result));
                                // FAST PATH 1c: LIKE pattern scan (prefix/suffix/contains)
                                } else if let Some(result) =
                                    Self::try_fast_like_filter(&backend, &stmt)?
                                {
                                    if !stmt.is_pure_star() {
                                        let projected = Self::apply_projection_with_storage(
                                            &result,
                                            &stmt.columns,
                                            Some(storage_path),
                                        )?;
                                        return Ok(ApexResult::Data(projected));
                                    }
                                    return Ok(ApexResult::Data(result));
                                // FAST PATH 2: Try numeric range filter for BETWEEN
                                } else if let Some(result) =
                                    Self::try_fast_numeric_range_filter(&backend, &stmt)?
                                {
                                    result
                                // FAST PATH 3: Try combined string + numeric filter for multi-condition
                                } else if let Some(result) =
                                    Self::try_fast_multi_condition_filter(&backend, &stmt)?
                                {
                                    result
                                // FAST PATH 4: Mmap multi-condition AND on two different numeric columns
                                } else if let Some(result) = Self::try_fast_mmap_multi_condition(
                                    &backend,
                                    &stmt,
                                    storage_path,
                                )? {
                                    return Ok(result);
                                // FAST PATH 5: Mmap IN filter on string column
                                } else if let Some(result) =
                                    Self::try_fast_mmap_in_filter(&backend, &stmt, storage_path)?
                                {
                                    return Ok(result);
                                // FAST PATH 5b: Mmap IN / OR-of-equalities on numeric column
                                } else if let Some(result) =
                                    Self::try_fast_mmap_numeric_in_filter(
                                        &backend,
                                        &stmt,
                                        storage_path,
                                    )?
                                {
                                    return Ok(result);
                                } else if backend.is_mmap_only()
                                    && !backend.has_pending_deltas()
                                    && !backend.has_delta()
                                {
                                    // MMAP FAST PATH: byte-level scan + point lookups
                                    if let Some(where_clause) = &stmt.where_clause {
                                        let _limit_with_off =
                                            stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
                                        let (matching_indices, prefer_index_materialization) =
                                            if let Some((col, val)) =
                                                Self::extract_string_equality(where_clause)
                                            {
                                                (
                                                    backend.scan_string_filter_mmap(
                                                        &col,
                                                        &val,
                                                        _limit_with_off,
                                                    )?,
                                                    false,
                                                )
                                            } else if let Some((col, low, high)) =
                                                Self::extract_between_range(where_clause)
                                            {
                                                (
                                                    backend.scan_numeric_range_mmap(
                                                        &col,
                                                        low,
                                                        high,
                                                        _limit_with_off,
                                                    )?,
                                                    false,
                                                )
                                            } else if let Some((col, low, high)) =
                                                Self::extract_two_sided_same_col_range(where_clause)
                                            {
                                                // col >= N AND col <= M — logically equivalent to BETWEEN
                                                (
                                                    backend.scan_numeric_range_mmap(
                                                        &col,
                                                        low,
                                                        high,
                                                        _limit_with_off,
                                                    )?,
                                                    false,
                                                )
                                            } else if let Some((col, low, high)) =
                                                Self::extract_single_comparison_as_range(
                                                    where_clause,
                                                )
                                            {
                                                (
                                                    backend.scan_numeric_range_mmap(
                                                        &col,
                                                        low,
                                                        high,
                                                        _limit_with_off,
                                                    )?,
                                                    false,
                                                )
                                            } else if let Some((col, values)) =
                                                Self::extract_in_string_filter(where_clause)
                                            {
                                                (
                                                    backend.scan_string_in_mmap(
                                                        &col,
                                                        &values,
                                                        _limit_with_off,
                                                    )?,
                                                    true,
                                                )
                                            } else if let Some((col, nums)) =
                                                Self::extract_in_numeric_filter(where_clause)
                                                    .or_else(|| {
                                                        Self::extract_or_numeric_equalities(
                                                            where_clause,
                                                        )
                                                    })
                                            {
                                                // Numeric IN or OR-of-equalities: single-pass mmap scan
                                                (
                                                    backend.scan_numeric_in_mmap(
                                                        &col,
                                                        &nums,
                                                        _limit_with_off,
                                                    )?,
                                                    false,
                                                )
                                            } else if let Some(leaves) =
                                                Self::extract_or_leaf_predicates(where_clause)
                                            {
                                                // General OR decomposition: scan each leaf, union indices
                                                match Self::scan_or_leaves_mmap(
                                                    &backend,
                                                    &leaves,
                                                    _limit_with_off,
                                                )? {
                                                    Some(v) if !v.is_empty() => (Some(v), true),
                                                    Some(_) => (None, true), // empty result
                                                    None => (None, true),
                                                }
                                            } else {
                                                (None, false)
                                            };
                                        if let Some(indices) = matching_indices {
                                            if indices.is_empty() {
                                                return Ok(ApexResult::Empty(Arc::new(
                                                    Schema::empty(),
                                                )));
                                            }
                                            let batch = if prefer_index_materialization {
                                                Self::read_matching_rows_by_indices(
                                                    &backend, &stmt, &indices,
                                                )?
                                            } else {
                                                Self::read_matching_rows_adaptive(
                                                    &backend, &stmt, &indices,
                                                )?
                                            };

                                            // Apply ORDER BY with LIMIT if needed
                                            if !stmt.order_by.is_empty() {
                                                let k = stmt
                                                    .limit
                                                    .map(|l| l + stmt.offset.unwrap_or(0));
                                                let sort_batch = Self::augment_batch_for_order_by(
                                                    &batch,
                                                    &stmt.columns,
                                                    &stmt.order_by,
                                                )?;
                                                let sorted = Self::apply_order_by_topk(
                                                    &sort_batch,
                                                    &stmt.order_by,
                                                    k,
                                                )?;
                                                let limited = Self::apply_limit_offset(
                                                    &sorted,
                                                    stmt.limit,
                                                    stmt.offset,
                                                )?;
                                                let projected =
                                                    Self::apply_projection_with_storage(
                                                        &limited,
                                                        &stmt.columns,
                                                        Some(storage_path),
                                                    )?;
                                                return Ok(ApexResult::Data(projected));
                                            }

                                            if !stmt.is_pure_star() {
                                                let projected =
                                                    Self::apply_projection_with_storage(
                                                        &batch,
                                                        &stmt.columns,
                                                        Some(storage_path),
                                                    )?;
                                                return Ok(ApexResult::Data(projected));
                                            }
                                            return Ok(ApexResult::Data(batch));
                                        }
                                    }
                                    let filtered = Self::execute_with_late_materialization(
                                        &backend,
                                        &stmt,
                                        storage_path,
                                    )?;
                                    if filtered.num_rows() == 0 {
                                        return Ok(ApexResult::Empty(filtered.schema()));
                                    }
                                    if !stmt.is_pure_star() {
                                        let projected = Self::apply_projection_with_storage(
                                            &filtered,
                                            &stmt.columns,
                                            Some(storage_path),
                                        )?;
                                        return Ok(ApexResult::Data(projected));
                                    }
                                    return Ok(ApexResult::Data(filtered));
                                } else {
                                    // Late materialization for WHERE path
                                    // Return directly to avoid applying WHERE filter twice
                                    let filtered = Self::execute_with_late_materialization(
                                        &backend,
                                        &stmt,
                                        storage_path,
                                    )?;
                                    if filtered.num_rows() == 0 {
                                        return Ok(ApexResult::Empty(filtered.schema()));
                                    }
                                    if !stmt.is_pure_star() {
                                        let projected = Self::apply_projection_with_storage(
                                            &filtered,
                                            &stmt.columns,
                                            Some(storage_path),
                                        )?;
                                        return Ok(ApexResult::Data(projected));
                                    }
                                    return Ok(ApexResult::Data(filtered));
                                }
                            } else if !stmt.group_by.is_empty() {
                                // GROUP BY with WHERE: use dict-encoded path for faster string aggregation
                                let col_refs = Self::get_col_refs(&stmt);
                                backend.read_columns_to_arrow_dict(
                                    col_refs
                                        .as_ref()
                                        .map(|v| v.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                                        .as_deref(),
                                )?
                            } else if let Some(batch) =
                                Self::try_numeric_predicate_pushdown(&backend, &stmt)?
                            {
                                batch
                            } else {
                                let col_refs = Self::get_col_refs(&stmt);
                                backend.read_columns_to_arrow(
                                    col_refs
                                        .as_ref()
                                        .map(|v| v.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                                        .as_deref(),
                                    0,
                                    None,
                                )?
                            }
                        } else if can_late_materialize_where {
                            // FAST PATH 1: Try dictionary-based filter for simple string equality
                            if let Some(result) = Self::try_fast_string_filter(&backend, &stmt)? {
                                result
                            // FAST PATH 1b: No-LIMIT string filter (uses mmap scan + late materialization)
                            } else if let Some(result) =
                                Self::try_fast_string_filter_no_limit(&backend, &stmt)?
                            {
                                result
                            // FAST PATH 1c: LIKE pattern scan (prefix/suffix/contains)
                            } else if let Some(result) =
                                Self::try_fast_like_filter(&backend, &stmt)?
                            {
                                result
                            // FAST PATH 2: Try numeric range filter for BETWEEN
                            } else if let Some(result) =
                                Self::try_fast_numeric_range_filter(&backend, &stmt)?
                            {
                                result
                            // FAST PATH 3: Try combined string + numeric filter for multi-condition
                            } else if let Some(result) =
                                Self::try_fast_multi_condition_filter(&backend, &stmt)?
                            {
                                result
                            } else {
                                // Late materialization for SELECT * WHERE path
                                Self::execute_with_late_materialization(
                                    &backend,
                                    &stmt,
                                    storage_path,
                                )?
                            }
                        } else if stmt.where_clause.is_some() && stmt.limit.is_none() {
                            // FAST PATH: String filter without LIMIT (uses dictionary scan)
                            if let Some(result) =
                                Self::try_fast_string_filter_no_limit(&backend, &stmt)?
                            {
                                result
                            // FAST PATH: LIKE pattern scan (prefix/suffix/contains)
                            } else if let Some(result) =
                                Self::try_fast_like_filter(&backend, &stmt)?
                            {
                                result
                            } else if let Some(batch) =
                                Self::try_numeric_predicate_pushdown(&backend, &stmt)?
                            {
                                batch
                            } else {
                                let col_refs = Self::get_col_refs(&stmt);
                                backend.read_columns_to_arrow(
                                    col_refs
                                        .as_ref()
                                        .map(|v| v.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                                        .as_deref(),
                                    0,
                                    None,
                                )?
                            }
                        } else if can_late_materialize_order {
                            // Late materialization for ORDER BY + LIMIT path
                            Self::execute_with_order_late_materialization(&backend, &stmt)?
                        } else {
                            let col_refs = Self::get_col_refs(&stmt);
                            let col_refs_vec: Option<Vec<&str>> = col_refs
                                .as_ref()
                                .map(|v| v.iter().map(|s| s.as_str()).collect());
                            let can_pushdown_limit = stmt.where_clause.is_none()
                                && stmt.order_by.is_empty()
                                && stmt.group_by.is_empty()
                                && !has_aggregation_check;

                            // Note: V4 fast agg disabled - Arrow clone+SIMD outperforms due to cache warming

                            if !stmt.group_by.is_empty() {
                                // V4 FAST PATH: Cached GROUP BY
                                if let Some(result) =
                                    Self::try_fast_cached_group_by(&backend, &stmt)?
                                {
                                    return Ok(result);
                                }
                                // Fallback: dict-encoded Arrow path
                                backend.read_columns_to_arrow_dict(col_refs_vec.as_deref())?
                            } else {
                                let _row_limit = if can_pushdown_limit {
                                    stmt.limit.map(|l| l + stmt.offset.unwrap_or(0))
                                } else {
                                    None
                                };
                                backend.read_columns_to_arrow(
                                    col_refs_vec.as_deref(),
                                    0,
                                    _row_limit,
                                )?
                            }
                        }
                    }
                }
            }
        };

        // Determine row limit for early termination
        let row_limit = stmt.limit;

        // Check for aggregation BEFORE checking empty batch
        // Aggregations like COUNT(*) should return 0 for empty tables
        // Also check for aggregates inside expressions (e.g., CASE WHEN SUM(x) > 100 ...)
        let has_aggregation = stmt.columns.iter().any(|col| match col {
            SelectColumn::Aggregate { .. } => true,
            SelectColumn::Expression { expr, .. } => Self::expr_contains_aggregate(expr),
            _ => false,
        });

        if batch.num_rows() == 0 {
            // For aggregations on empty tables, still execute aggregation (COUNT(*) returns 0)
            if has_aggregation && stmt.group_by.is_empty() {
                return Self::execute_aggregation(&batch, &stmt);
            }
            return Ok(ApexResult::Empty(batch.schema()));
        }

        // Apply WHERE filter (with storage path for subquery support)
        let filtered = if let Some(ref where_clause) = stmt.where_clause {
            Self::apply_filter_with_storage(&batch, where_clause, storage_path)?
        } else {
            batch
        };

        if filtered.num_rows() == 0 {
            // For aggregations on filtered empty result, still execute aggregation
            if has_aggregation && stmt.group_by.is_empty() {
                return Self::execute_aggregation(&filtered, &stmt);
            }
            return Ok(ApexResult::Empty(filtered.schema()));
        }

        // Check for window functions
        let has_window = stmt
            .columns
            .iter()
            .any(|col| matches!(col, SelectColumn::WindowFunction { .. }));
        if has_window {
            return Self::execute_window_function(&filtered, &stmt);
        }

        if has_aggregation && stmt.group_by.is_empty() {
            // Simple aggregation without GROUP BY
            return Self::execute_aggregation(&filtered, &stmt);
        }

        // Handle GROUP BY (also triggered by HAVING even without SELECT aggregates)
        if !stmt.group_by.is_empty() && (has_aggregation || stmt.having.is_some()) {
            return Self::execute_group_by(&filtered, &stmt);
        }

        // For DISTINCT: sort without top-k limit, project, deduplicate, then limit
        // For DISTINCT ON: sort, deduplicate by ON columns, project, then limit/offset
        // For non-DISTINCT: apply top-k sort + limit, then project
        let result = if stmt.distinct {
            let sorted = if !stmt.order_by.is_empty() {
                Self::apply_order_by(&filtered, &stmt.order_by)?
            } else {
                filtered
            };
            if let Some(ref on_cols) = stmt.distinct_on {
                // DISTINCT ON: deduplicate by ON columns, then project + limit/offset
                let deduped = Self::deduplicate_batch_on(&sorted, on_cols)?;
                let projected = Self::apply_projection_with_storage(&deduped, &stmt.columns, Some(storage_path))?;
                Self::apply_limit_offset(&projected, stmt.limit, stmt.offset)?
            } else {
                // Regular DISTINCT: project, deduplicate all columns, then limit
                let projected = Self::apply_projection_with_storage(&sorted, &stmt.columns, Some(storage_path))?;
                let deduped = Self::deduplicate_batch(&projected)?;
                Self::apply_limit_offset(&deduped, stmt.limit, stmt.offset)?
            }
        } else {
            // Apply ORDER BY with LIMIT optimization (top-k heap sort)
            let limited = if !stmt.order_by.is_empty() {
                let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
                // Pre-evaluate any SELECT expression aliases referenced in ORDER BY
                // (e.g. SELECT array_distance(vec,[…]) AS dist … ORDER BY dist)
                let sort_batch =
                    Self::augment_batch_for_order_by(&filtered, &stmt.columns, &stmt.order_by)?;
                let sorted = Self::apply_order_by_topk(&sort_batch, &stmt.order_by, k)?;
                Self::apply_limit_offset(&sorted, stmt.limit, stmt.offset)?
            } else {
                Self::apply_limit_offset(&filtered, stmt.limit, stmt.offset)?
            };
            Self::apply_projection_with_storage(&limited, &stmt.columns, Some(storage_path))?
        };

        Ok(ApexResult::Data(result))
    }

    /// Try to use a secondary index to accelerate a SELECT query.
    /// Returns Some(result) if an index was used, None to fall through to scan paths.
    /// Only used for simple equality WHERE clauses on indexed columns (no GROUP BY/aggregation).
    fn try_index_accelerated_read(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
        where_clause: &SqlExpr,
        base_dir: &Path,
        storage_path: &Path,
    ) -> io::Result<Option<ApexResult>> {
        use crate::query::sql_parser::BinaryOperator;
        use crate::storage::index::index_manager::PredicateHint;

        // Skip index path for mmap-only: per-row reads would each do a full mmap scan
        if backend.is_mmap_only() {
            return Ok(None);
        }

        // Only use index for simple queries: no GROUP BY, no aggregation, no JOIN
        if !stmt.group_by.is_empty() || !stmt.joins.is_empty() {
            return Ok(None);
        }
        let has_agg = stmt
            .columns
            .iter()
            .any(|c| matches!(c, SelectColumn::Aggregate { .. }));
        if has_agg {
            return Ok(None);
        }

        // Extract predicate(s): single or AND-combined predicates
        // Flatten AND chains into individual (col_name, hint) pairs
        let mut predicates: Vec<(String, PredicateHint)> = Vec::new();
        Self::extract_index_predicates(where_clause, &mut predicates);

        if predicates.is_empty() {
            return Ok(None);
        }

        // Check which predicates have indexes available
        let (bd, tname) = base_dir_and_table(storage_path);
        let idx_mgr_arc = get_index_manager(&bd, &tname);
        let mut idx_mgr = idx_mgr_arc.lock();

        // Filter to predicates that have indexes
        let indexed_preds: Vec<(String, PredicateHint)> = predicates
            .into_iter()
            .filter(|(col, _)| idx_mgr.has_index_on(col))
            .collect();

        if indexed_preds.is_empty() {
            return Ok(None);
        }

        // CBO: Pre-estimate selectivity using ANALYZE stats before expensive index lookup
        let table_key = storage_path.to_string_lossy();
        if let Some(stats) = get_table_stats(&table_key) {
            let selectivity = QueryPlanner::estimate_selectivity(where_clause, &stats);
            if stats.row_count > 0
                && !QueryPlanner::should_use_index("", selectivity, stats.row_count)
            {
                return Ok(None); // CBO says full scan is cheaper, skip index lookup
            }
        }

        // Look up each indexed predicate and intersect row ID sets
        let mut row_ids: Option<Vec<u64>> = None;
        for (col_name, hint) in &indexed_preds {
            let lookup_result = idx_mgr.lookup(col_name, hint)?;
            match lookup_result {
                Some(r) => {
                    row_ids = Some(match row_ids {
                        None => r.row_ids,
                        Some(existing) => {
                            // Intersect: keep only IDs in both sets
                            let set: std::collections::HashSet<u64> =
                                r.row_ids.into_iter().collect();
                            existing.into_iter().filter(|id| set.contains(id)).collect()
                        }
                    });
                }
                None => {
                    // Index couldn't satisfy this predicate, skip it
                    // (still use other index results if available)
                }
            }
        }

        let row_ids = match row_ids {
            Some(ids) => ids,
            None => return Ok(None),
        };

        if row_ids.is_empty() {
            let empty = backend.read_columns_to_arrow(None, 0, Some(0))?;
            return Ok(Some(ApexResult::Empty(empty.schema())));
        }

        // Read matching rows by their _ids
        // CBO: use ANALYZE stats to decide index vs full scan cost
        let total_rows = backend.active_row_count();
        let selectivity = if total_rows > 0 {
            row_ids.len() as f64 / total_rows as f64
        } else {
            1.0
        };
        if !QueryPlanner::should_use_index("", selectivity, total_rows as u64) {
            return Ok(None); // Cost model says full scan is cheaper
        }

        // Covering index (index-only scan): if all SELECT columns are covered by
        // the index (_id + indexed columns), build result directly without reading
        // the base table — avoids expensive per-row table lookups.
        if let Some(covered) = Self::try_index_only_scan(stmt, &indexed_preds, &row_ids)? {
            return Ok(Some(covered));
        }

        // Read matching rows one by one and concat
        let mut batches: Vec<RecordBatch> = Vec::with_capacity(row_ids.len().min(1024));
        for &rid in &row_ids {
            if let Some(batch) = backend.read_row_by_id_to_arrow(rid)? {
                batches.push(batch);
            }
        }

        if batches.is_empty() {
            let empty = backend.read_columns_to_arrow(None, 0, Some(0))?;
            return Ok(Some(ApexResult::Empty(empty.schema())));
        }

        // Concat all batches into one
        let schema = batches[0].schema();
        let combined = arrow::compute::concat_batches(&schema, &batches)
            .map_err(|e| err_data(e.to_string()))?;

        // Apply ORDER BY if present
        let sorted = if !stmt.order_by.is_empty() {
            Self::apply_order_by(&combined, &stmt.order_by)?
        } else {
            combined
        };

        // Apply OFFSET + LIMIT
        let result = {
            let offset = stmt.offset.unwrap_or(0);
            let total = sorted.num_rows();
            if offset >= total {
                sorted.slice(0, 0)
            } else if let Some(limit) = stmt.limit {
                let end = (offset + limit).min(total);
                sorted.slice(offset, end - offset)
            } else if offset > 0 {
                sorted.slice(offset, total - offset)
            } else {
                sorted
            }
        };

        // Apply column projection if not pure SELECT *
        if !stmt.is_pure_star() {
            let projected = Self::apply_projection(&result, &stmt.columns)?;
            if projected.num_rows() == 0 {
                return Ok(Some(ApexResult::Empty(projected.schema())));
            }
            return Ok(Some(ApexResult::Data(projected)));
        }

        if result.num_rows() == 0 {
            return Ok(Some(ApexResult::Empty(result.schema())));
        }
        Ok(Some(ApexResult::Data(result)))
    }

    /// Extract index-usable predicates from an expression (flattens AND chains).
    /// Each predicate is (column_name, PredicateHint).
    fn extract_index_predicates(
        expr: &SqlExpr,
        out: &mut Vec<(String, crate::storage::index::index_manager::PredicateHint)>,
    ) {
        use crate::query::sql_parser::BinaryOperator;
        use crate::storage::index::index_manager::PredicateHint;
        match expr {
            // AND chain: recurse into both sides
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::And,
                right,
            } => {
                Self::extract_index_predicates(left, out);
                Self::extract_index_predicates(right, out);
            }
            // col OP literal or literal OP col
            SqlExpr::BinaryOp { left, op, right } => {
                if let SqlExpr::Column(col) = left.as_ref() {
                    if col != "_id" {
                        if let Some(val) = Self::expr_to_value(right) {
                            let h = match op {
                                BinaryOperator::Eq => Some(PredicateHint::Eq(val)),
                                BinaryOperator::Gt => Some(PredicateHint::Gt(val)),
                                BinaryOperator::Ge => Some(PredicateHint::Gte(val)),
                                BinaryOperator::Lt => Some(PredicateHint::Lt(val)),
                                BinaryOperator::Le => Some(PredicateHint::Lte(val)),
                                _ => None,
                            };
                            if let Some(hint) = h {
                                out.push((col.clone(), hint));
                            }
                        }
                    }
                } else if let SqlExpr::Column(col) = right.as_ref() {
                    if col != "_id" {
                        if let Some(val) = Self::expr_to_value(left) {
                            let h = match op {
                                BinaryOperator::Eq => Some(PredicateHint::Eq(val)),
                                BinaryOperator::Gt => Some(PredicateHint::Lt(val)),
                                BinaryOperator::Ge => Some(PredicateHint::Lte(val)),
                                BinaryOperator::Lt => Some(PredicateHint::Gt(val)),
                                BinaryOperator::Le => Some(PredicateHint::Gte(val)),
                                _ => None,
                            };
                            if let Some(hint) = h {
                                out.push((col.clone(), hint));
                            }
                        }
                    }
                }
            }
            SqlExpr::Between {
                column,
                low,
                high,
                negated,
            } => {
                if !negated && column != "_id" {
                    if let (Some(low_val), Some(high_val)) =
                        (Self::expr_to_value(low), Self::expr_to_value(high))
                    {
                        out.push((
                            column.clone(),
                            PredicateHint::Range {
                                low: low_val,
                                high: high_val,
                            },
                        ));
                    }
                }
            }
            SqlExpr::In {
                column,
                values,
                negated,
            } => {
                if !negated && column != "_id" {
                    out.push((column.clone(), PredicateHint::In(values.clone())));
                }
            }
            _ => {}
        }
    }

    /// Covering index (index-only scan): build result directly from index data
    /// when all SELECT columns are covered by {_id, indexed_columns}.
    /// For equality predicates, the column value is known from the predicate itself.
    /// Returns None if the query needs columns not available from the index.
    fn try_index_only_scan(
        stmt: &SelectStatement,
        indexed_preds: &[(String, crate::storage::index::index_manager::PredicateHint)],
        row_ids: &[u64],
    ) -> io::Result<Option<ApexResult>> {
        use std::collections::HashMap;
        // Only for non-* queries (SELECT * needs all columns)
        if stmt.is_select_star() {
            return Ok(None);
        }
        // Only for simple equality predicates (we know the exact value)
        // Collect indexed column names and their known values
        use crate::storage::index::index_manager::PredicateHint;
        let mut known_values: HashMap<String, Value> = HashMap::new();
        for (col, hint) in indexed_preds {
            match hint {
                PredicateHint::Eq(val) => {
                    known_values.insert(col.clone(), val.clone());
                }
                PredicateHint::In(_) if row_ids.len() <= 1 => {
                    return Ok(None);
                }
                _ => {
                    return Ok(None);
                } // Range predicates: values vary per row
            }
        }
        if known_values.is_empty() {
            return Ok(None);
        }

        // Check if all SELECT columns are covered by {_id} ∪ {indexed columns}
        let mut need_id = false;
        let mut need_cols: Vec<String> = Vec::new();
        for col in &stmt.columns {
            match col {
                SelectColumn::Column(name) => {
                    let clean = name.trim_matches('"');
                    if clean == "_id" {
                        need_id = true;
                    } else if known_values.contains_key(clean) {
                        need_cols.push(clean.to_string());
                    } else {
                        return Ok(None); // Need a column not in index → can't cover
                    }
                }
                SelectColumn::ColumnAlias { column, .. } => {
                    let clean = column.trim_matches('"');
                    if clean == "_id" {
                        need_id = true;
                    } else if known_values.contains_key(clean) {
                        need_cols.push(clean.to_string());
                    } else {
                        return Ok(None);
                    }
                }
                SelectColumn::All => {
                    return Ok(None);
                }
                _ => {
                    return Ok(None);
                } // Aggregate, expression, etc.
            }
        }

        // Build Arrow RecordBatch directly from index data
        let n = row_ids.len();
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        if need_id {
            fields.push(Field::new("_id", arrow::datatypes::DataType::Int64, false));
            let id_arr: Int64Array = row_ids.iter().map(|&id| id as i64).collect();
            arrays.push(Arc::new(id_arr) as ArrayRef);
        }

        for col_name in &need_cols {
            let val = &known_values[col_name];
            match val {
                Value::Int64(v) => {
                    fields.push(Field::new(
                        col_name,
                        arrow::datatypes::DataType::Int64,
                        true,
                    ));
                    let arr = Int64Array::from(vec![*v; n]);
                    arrays.push(Arc::new(arr) as ArrayRef);
                }
                Value::Float64(f) => {
                    fields.push(Field::new(
                        col_name,
                        arrow::datatypes::DataType::Float64,
                        true,
                    ));
                    let arr = Float64Array::from(vec![*f; n]);
                    arrays.push(Arc::new(arr) as ArrayRef);
                }
                Value::String(s) => {
                    fields.push(Field::new(col_name, arrow::datatypes::DataType::Utf8, true));
                    let arr = StringArray::from(vec![s.as_str(); n]);
                    arrays.push(Arc::new(arr) as ArrayRef);
                }
                Value::Bool(b) => {
                    fields.push(Field::new(
                        col_name,
                        arrow::datatypes::DataType::Boolean,
                        true,
                    ));
                    let arr = BooleanArray::from(vec![*b; n]);
                    arrays.push(Arc::new(arr) as ArrayRef);
                }
                _ => {
                    return Ok(None);
                } // Unsupported value type for index-only scan
            }
        }

        if fields.is_empty() {
            return Ok(None);
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays).map_err(|e| err_data(e.to_string()))?;

        // Apply ORDER BY if present
        let sorted = if !stmt.order_by.is_empty() {
            Self::apply_order_by(&batch, &stmt.order_by)?
        } else {
            batch
        };

        // Apply OFFSET + LIMIT
        let result = Self::apply_limit_offset(&sorted, stmt.limit, stmt.offset)?;

        if result.num_rows() == 0 {
            return Ok(Some(ApexResult::Empty(result.schema())));
        }
        Ok(Some(ApexResult::Data(result)))
    }

    /// Convert a SqlExpr literal to a Value (for index lookup)
    fn expr_to_value(expr: &SqlExpr) -> Option<Value> {
        match expr {
            SqlExpr::Literal(v) => Some(v.clone()),
            _ => None,
        }
    }

    /// Fast path for simple string equality filters on dictionary-encoded columns
    /// Uses storage-level early termination for LIMIT queries when limit is Some
    /// Supports column projection pushdown (not limited to SELECT *)
    fn try_fast_string_filter(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        // Must have LIMIT for early termination benefit
        if stmt.limit.is_none() {
            return Ok(None);
        }
        Self::try_fast_string_filter_impl(backend, stmt, stmt.limit)
    }

    /// Fast path for string equality filters WITHOUT LIMIT
    fn try_fast_string_filter_no_limit(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        Self::try_fast_string_filter_impl(backend, stmt, None)
    }

    /// Fast path for LIKE filters: adaptive strategy based on selectivity.
    /// Uses full table scan + Arrow filter for high-selectivity queries,
    /// and index extraction for low-selectivity queries.
    fn try_fast_like_filter(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        // Skip if there are pending deltas (use slower but accurate path)
        if backend.has_pending_deltas() || backend.is_mmap_only() {
            return Ok(None);
        }

        // Must have LIMIT for early termination benefit
        if stmt.limit.is_none() {
            return Ok(None);
        }

        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };
        let (col_name, pattern) = match Self::extract_like_pattern(where_clause) {
            Some(v) => v,
            None => return Ok(None),
        };

        // Fast path: single-pass parallel scan+extract (V4 mmap, any selectivity).
        // Avoids materializing non-matching rows — only builds Arrow arrays for hits.
        // Returns None for compressed/non-RCIX files → falls through to old paths.
        let limit_with_offset = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
        if let Some(batch) =
            backend.scan_like_and_extract_mmap(&col_name, &pattern, limit_with_offset)?
        {
            let offset = stmt.offset.unwrap_or(0);
            let result = if offset > 0 {
                let n = batch.num_rows().saturating_sub(offset);
                batch.slice(offset, n)
            } else {
                batch
            };
            if let Some(lim) = stmt.limit {
                let n = result.num_rows().min(lim);
                return Ok(Some(result.slice(0, n)));
            }
            return Ok(Some(result));
        }

        // Fallback: index-based extraction (compressed/non-RCIX files)
        let mut indices =
            match backend.scan_like_filter_mmap(&col_name, &pattern, limit_with_offset)? {
                Some(v) => v,
                None => return Ok(None),
            };

        if indices.is_empty() {
            let empty = Self::read_matching_rows_adaptive(backend, stmt, &indices)?;
            return Ok(Some(empty));
        }

        let offset = stmt.offset.unwrap_or(0);
        if offset > 0 {
            if offset >= indices.len() {
                let empty = Self::read_matching_rows_adaptive(backend, stmt, &[])?;
                return Ok(Some(empty));
            }
            indices = indices[offset..].to_vec();
        }
        if let Some(lim) = stmt.limit {
            indices.truncate(lim);
        }

        Ok(Some(Self::read_matching_rows_adaptive(
            backend, stmt, &indices,
        )?))
    }

    /// Unified implementation for string equality filter fast path
    fn try_fast_string_filter_impl(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
        limit: Option<usize>,
    ) -> io::Result<Option<RecordBatch>> {
        if backend.pending_v4_in_memory_rows() > 0 || backend.has_pending_deltas() {
            return Ok(None);
        }

        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };

        let (col_name, filter_value) = match Self::extract_string_equality(where_clause) {
            Some(v) => v,
            None => return Ok(None),
        };

        // Column projection pushdown
        let projected_cols: Option<Vec<String>> = if stmt.is_select_star() {
            None
        } else {
            Some(stmt.required_columns().unwrap_or_default())
        };
        let col_refs: Option<Vec<&str>> = projected_cols
            .as_ref()
            .map(|cols| cols.iter().map(|s| s.as_str()).collect());

        let result = if let Some(lim) = limit {
            if backend.is_mmap_only()
                && stmt.offset.unwrap_or(0) == 0
                && !backend.has_delta()
                && !backend.has_pending_deltas()
            {
                let indices =
                    match backend.scan_string_filter_mmap(&col_name, &filter_value, Some(lim))? {
                        Some(v) => v,
                        None => return Ok(None),
                    };
                return Ok(Some(Self::read_matching_rows_adaptive(
                    backend, stmt, &indices,
                )?));
            }
            if backend.pending_delta_updates_column(&col_name) || backend.has_delta() {
                let full = backend.read_columns_filtered_string_to_arrow(
                    col_refs.as_deref(),
                    &col_name,
                    &filter_value,
                    true,
                )?;
                let offset = stmt.offset.unwrap_or(0).min(full.num_rows());
                let len = lim.min(full.num_rows().saturating_sub(offset));
                full.slice(offset, len)
            } else {
                backend.read_columns_filtered_string_with_limit_to_arrow(
                    col_refs.as_deref(),
                    &col_name,
                    &filter_value,
                    true,
                    lim,
                    stmt.offset.unwrap_or(0),
                )?
            }
        } else {
            backend.read_columns_filtered_string_to_arrow(
                col_refs.as_deref(),
                &col_name,
                &filter_value,
                true,
            )?
        };

        Ok(Some(result))
    }

    /// Fast path for numeric range filters (BETWEEN)
    /// Uses streaming scan with early termination for LIMIT queries
    /// Supports column projection pushdown (not limited to SELECT *)
    fn try_fast_numeric_range_filter(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        use crate::query::sql_parser::BinaryOperator;

        if backend.has_pending_deltas() || backend.has_delta() {
            return Ok(None);
        }

        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };

        if backend.is_mmap_only() {
            if stmt.limit.is_none() {
                return Ok(None);
            }
            let (col, lo, hi) = match Self::extract_any_numeric_range(where_clause) {
                Some(v) => v,
                None => return Ok(None),
            };
            let limit_with_off = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
            let indices = match backend.scan_numeric_range_mmap(&col, lo, hi, limit_with_off)? {
                Some(v) => v,
                None => return Ok(None),
            };
            if indices.is_empty() {
                let schema = backend.read_columns_to_arrow(None, 0, Some(0))?;
                return Ok(Some(schema));
            }
            let batch = Self::read_matching_rows_adaptive(backend, stmt, &indices)?;
            return Ok(Some(batch));
        }

        // The storage-level range reader is a LIMIT-oriented fast path.
        if stmt.limit.is_none() {
            return Ok(None);
        }

        // Extract BETWEEN pattern: col BETWEEN low AND high
        let (col_name, low, high) = match where_clause {
            SqlExpr::Between {
                column,
                low,
                high,
                negated,
            } => {
                if *negated {
                    return Ok(None);
                }
                let low_val = Self::extract_numeric_value(low)?;
                let high_val = Self::extract_numeric_value(high)?;
                (column.trim_matches('"').to_string(), low_val, high_val)
            }
            // Also handle col >= low AND col <= high pattern
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::And,
                right,
            } => {
                let (col1, op1, val1) = match Self::extract_comparison(left) {
                    Ok(v) => v,
                    Err(_) => return Ok(None),
                };
                let (col2, op2, val2) = match Self::extract_comparison(right) {
                    Ok(v) => v,
                    Err(_) => return Ok(None),
                };

                if col1 != col2 {
                    return Ok(None);
                }

                // Determine low and high from the operators
                let (low, high) = match (op1, op2) {
                    (BinaryOperator::Ge, BinaryOperator::Le) => (val1, val2),
                    (BinaryOperator::Le, BinaryOperator::Ge) => (val2, val1),
                    (BinaryOperator::Gt, BinaryOperator::Lt) => (val1, val2),
                    (BinaryOperator::Lt, BinaryOperator::Gt) => (val2, val1),
                    _ => return Ok(None),
                };
                (col1, low, high)
            }
            _ => return Ok(None),
        };

        let projected_cols: Option<Vec<String>> = if stmt.is_select_star() {
            None
        } else {
            Some(stmt.required_columns().unwrap_or_default())
        };
        let col_refs: Option<Vec<&str>> = projected_cols
            .as_ref()
            .map(|cols| cols.iter().map(|s| s.as_str()).collect());

        let limit = stmt.limit.unwrap_or(100);
        let offset = stmt.offset.unwrap_or(0);

        // Use storage-level numeric range filter with early termination
        let result = backend.read_columns_filtered_range_with_limit_to_arrow(
            col_refs.as_deref(),
            &col_name,
            low,
            high,
            limit,
            offset,
        )?;

        Ok(Some(result))
    }

    /// Helper to extract numeric value from SqlExpr
    fn extract_numeric_value(expr: &SqlExpr) -> io::Result<f64> {
        match expr {
            SqlExpr::Literal(Value::Int64(n)) => Ok(*n as f64),
            SqlExpr::Literal(Value::Int32(n)) => Ok(*n as f64),
            SqlExpr::Literal(Value::Float64(n)) => Ok(*n),
            SqlExpr::Literal(Value::Float32(n)) => Ok(*n as f64),
            _ => Err(err_input("not a number")),
        }
    }

    /// Helper to extract comparison from binary op
    fn extract_comparison(
        expr: &SqlExpr,
    ) -> io::Result<(String, crate::query::sql_parser::BinaryOperator, f64)> {
        use crate::query::sql_parser::BinaryOperator;
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                match (left.as_ref(), right.as_ref()) {
                    (SqlExpr::Column(col), lit) => {
                        let val = Self::extract_numeric_value(lit)?;
                        Ok((col.trim_matches('"').to_string(), op.clone(), val))
                    }
                    (lit, SqlExpr::Column(col)) => {
                        let val = Self::extract_numeric_value(lit)?;
                        // Flip the operator
                        let flipped_op = match op {
                            BinaryOperator::Gt => BinaryOperator::Lt,
                            BinaryOperator::Lt => BinaryOperator::Gt,
                            BinaryOperator::Ge => BinaryOperator::Le,
                            BinaryOperator::Le => BinaryOperator::Ge,
                            _ => return Err(err_input("unsupported op")),
                        };
                        Ok((col.trim_matches('"').to_string(), flipped_op, val))
                    }
                    _ => Err(err_input("not a comparison")),
                }
            }
            _ => Err(err_input("not a binary op")),
        }
    }

    /// Fast path for multi-condition WHERE with string equality AND numeric comparison
    /// Handles: SELECT * WHERE string_col = 'value' AND numeric_col > N LIMIT n
    fn try_fast_multi_condition_filter(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        use crate::query::sql_parser::BinaryOperator;

        if backend.has_pending_deltas() || backend.is_mmap_only() {
            return Ok(None);
        }

        // Must have LIMIT
        if stmt.limit.is_none() {
            return Ok(None);
        }

        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };

        // Must be AND of two conditions
        let (left_cond, right_cond) = match where_clause {
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::And,
                right,
            } => (left.as_ref(), right.as_ref()),
            _ => return Ok(None),
        };

        // Try to extract string equality and numeric comparison from either order
        let (str_col, str_val, num_col, num_op, num_val) =
            if let (Some((sc, sv)), Some((nc, no, nv))) = (
                Self::extract_string_equality(left_cond),
                Self::extract_numeric_comparison(right_cond),
            ) {
                (sc, sv, nc, no, nv)
            } else if let (Some((sc, sv)), Some((nc, no, nv))) = (
                Self::extract_string_equality(right_cond),
                Self::extract_numeric_comparison(left_cond),
            ) {
                (sc, sv, nc, no, nv)
            } else {
                return Ok(None);
            };

        let limit = stmt.limit.unwrap_or(100);
        let offset = stmt.offset.unwrap_or(0);

        // Use storage-level combined filter
        let result = backend.read_columns_filtered_string_numeric_with_limit_to_arrow(
            None, // All columns (SELECT *)
            &str_col, &str_val, &num_col, &num_op, num_val, limit, offset,
        )?;

        Ok(Some(result))
    }

    /// FAST PATH for Complex (Filter+Group+Order) queries
    /// Optimized for: SELECT group_col, AGG(agg_col) FROM table WHERE filter_col = 'value' GROUP BY group_col ORDER BY agg DESC LIMIT n
    /// Uses single-pass execution with direct dictionary indexing for maximum performance
    fn try_fast_filter_group_order(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<ApexResult>> {
        if backend.has_pending_deltas() || backend.has_delta() {
            return Ok(None);
        }

        use crate::query::sql_parser::BinaryOperator;
        use crate::query::AggregateFunc;

        // Check pattern: must have WHERE, GROUP BY, ORDER BY, and LIMIT
        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };

        if stmt.group_by.is_empty() || stmt.order_by.is_empty() || stmt.limit.is_none() {
            return Ok(None);
        }

        // Support: string equality (col = 'val') OR BETWEEN (col BETWEEN low AND high)
        enum FilterType<'a> {
            StringEq(String, &'a str),
            Between(String, f64, f64),
        }

        let filter = match where_clause {
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::Eq,
                right,
            } => match (left.as_ref(), right.as_ref()) {
                (SqlExpr::Column(col), SqlExpr::Literal(Value::String(val))) => {
                    FilterType::StringEq(col.trim_matches('"').to_string(), val.as_str())
                }
                (SqlExpr::Literal(Value::String(val)), SqlExpr::Column(col)) => {
                    FilterType::StringEq(col.trim_matches('"').to_string(), val.as_str())
                }
                _ => return Ok(None),
            },
            SqlExpr::Between {
                column,
                low,
                high,
                negated,
            } if !negated => {
                let low_val = Self::extract_numeric_value(low).ok();
                let high_val = Self::extract_numeric_value(high).ok();
                if let (Some(lo), Some(hi)) = (low_val, high_val) {
                    FilterType::Between(column.trim_matches('"').to_string(), lo, hi)
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        };

        // Must have exactly one GROUP BY column (string)
        if stmt.group_by.len() != 1 {
            return Ok(None);
        }
        let group_col = stmt.group_by[0].trim_matches('"');

        // Must have exactly one ORDER BY clause
        if stmt.order_by.len() != 1 {
            return Ok(None);
        }
        let order_clause = &stmt.order_by[0];
        let order_col = order_clause.column.trim_matches('"');
        let descending = order_clause.descending;

        // Check if we have exactly one aggregate column
        let mut agg_func = None;
        let mut agg_col = None;
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, .. } = col {
                agg_func = Some(func.clone());
                agg_col = column.as_deref();
            }
        }

        let agg_func = match agg_func {
            Some(f) => f,
            None => return Ok(None),
        };

        // Support SUM, COUNT, and AVG
        if !matches!(
            agg_func,
            AggregateFunc::Sum | AggregateFunc::Count | AggregateFunc::Avg
        ) {
            return Ok(None);
        }

        // Check HAVING clause - must be simple
        if let Some(having) = &stmt.having {
            match having {
                SqlExpr::BinaryOp {
                    left,
                    op: BinaryOperator::Gt,
                    right,
                } => match (left.as_ref(), right.as_ref()) {
                    (SqlExpr::Column(_col), SqlExpr::Literal(Value::Int64(_val))) => {}
                    _ => return Ok(None),
                },
                _ => return Ok(None),
            }
        }

        let limit = stmt.limit.unwrap_or(100);
        let offset = stmt.offset.unwrap_or(0);

        // For string equality filter, use existing storage-level path
        match &filter {
            FilterType::StringEq(filter_col, filter_val) => {
                // Only SUM/COUNT for the storage-level string eq path
                if !matches!(agg_func, AggregateFunc::Sum | AggregateFunc::Count) {
                    return Ok(None);
                }
                match backend.execute_filter_group_order(
                    filter_col, filter_val, group_col, agg_col, agg_func, order_col, descending,
                    limit, offset,
                ) {
                    Ok(Some(result)) => Ok(Some(ApexResult::Data(result))),
                    Ok(None) => Ok(None),
                    Err(e) => Err(e),
                }
            }
            FilterType::Between(filter_col, lo, hi) => {
                let raw = if let Some(dict_arc) = crate::storage::backend::get_global_dict_cache(
                    backend.path(),
                    group_col,
                    &backend.storage,
                )? {
                    backend.execute_between_group_agg_cached(
                        filter_col,
                        *lo,
                        *hi,
                        &dict_arc.0,
                        &dict_arc.1,
                        agg_col,
                    )?
                } else {
                    backend
                        .storage
                        .execute_between_group_agg(filter_col, *lo, *hi, group_col, agg_col)?
                };

                let raw = match raw {
                    Some(r) if !r.is_empty() => r,
                    _ => return Ok(None),
                };

                // Compute final aggregated values
                let mut results: Vec<(String, f64)> = raw
                    .iter()
                    .map(|(k, sum, count)| {
                        let val = match agg_func {
                            AggregateFunc::Sum => *sum,
                            AggregateFunc::Count => *count as f64,
                            AggregateFunc::Avg => {
                                if *count > 0 {
                                    *sum / *count as f64
                                } else {
                                    0.0
                                }
                            }
                            _ => *sum,
                        };
                        (k.clone(), val)
                    })
                    .collect();

                // Sort
                if descending {
                    results
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    results
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                }
                let results: Vec<_> = results.into_iter().skip(offset).take(limit).collect();

                if results.is_empty() {
                    return Ok(None);
                }

                // Build Arrow result
                let group_values: Vec<&str> = results.iter().map(|(k, _)| k.as_str()).collect();
                let agg_values: Vec<f64> = results.iter().map(|(_, v)| *v).collect();

                let group_col_name = group_col.to_string();
                let agg_col_name = stmt
                    .columns
                    .iter()
                    .find_map(|c| {
                        if let SelectColumn::Aggregate { alias, .. } = c {
                            alias.clone().or_else(|| Some(order_col.to_string()))
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| order_col.to_string());

                let schema = Arc::new(Schema::new(vec![
                    Field::new(&group_col_name, ArrowDataType::Utf8, false),
                    Field::new(&agg_col_name, ArrowDataType::Float64, false),
                ]));
                let arrays: Vec<ArrayRef> = vec![
                    Arc::new(StringArray::from(group_values)),
                    Arc::new(Float64Array::from(agg_values)),
                ];
                let result =
                    RecordBatch::try_new(schema, arrays).map_err(|e| err_data(e.to_string()))?;

                Ok(Some(ApexResult::Data(result)))
            }
        }
    }

    /// V4 FAST PATH for GROUP BY queries without WHERE
    /// Handles: SELECT group_col, AGG1(col1), AGG2(col2) FROM table GROUP BY group_col
    fn try_fast_v4_group_by(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<ApexResult>> {
        if backend.pending_v4_in_memory_rows() > 0
            || backend.has_pending_deltas()
            || backend.has_delta()
        {
            return Ok(None);
        }

        use crate::query::AggregateFunc;

        // Must be single GROUP BY column, no WHERE, no ORDER BY
        if stmt.group_by.len() != 1 || stmt.where_clause.is_some() || !stmt.order_by.is_empty() {
            return Ok(None);
        }

        let group_col = stmt.group_by[0].trim_matches('"');

        // Extract aggregate columns: (col_name_or_"*", is_count_star, func, alias)
        let mut agg_info: Vec<(&str, bool, AggregateFunc, Option<String>)> = Vec::new();

        for col in &stmt.columns {
            match col {
                SelectColumn::Aggregate {
                    func,
                    column,
                    alias,
                    ..
                } => {
                    let is_count_star = matches!(func, AggregateFunc::Count) && column.is_none();
                    let col_name = column.as_deref().unwrap_or("*");
                    agg_info.push((col_name, is_count_star, func.clone(), alias.clone()));
                }
                SelectColumn::Column(name) => {
                    if name.trim_matches('"') == group_col {
                        continue;
                    }
                    return Ok(None);
                }
                SelectColumn::ColumnAlias { column, .. } => {
                    if column.trim_matches('"') == group_col {
                        continue;
                    }
                    return Ok(None);
                }
                _ => return Ok(None),
            }
        }

        if agg_info.is_empty() {
            return Ok(None);
        }

        // Build agg_cols for storage call
        let agg_cols: Vec<(&str, bool)> = agg_info
            .iter()
            .map(|(col, is_count, _, _)| (*col, *is_count))
            .collect();

        let raw = match backend.execute_group_agg(group_col, &agg_cols)? {
            Some(r) if !r.is_empty() => r,
            _ => return Ok(None),
        };

        // Build result: group_col + one column per aggregate
        let num_groups = raw.len();
        let group_values: Vec<&str> = raw.iter().map(|(k, _)| k.as_str()).collect();

        let mut fields: Vec<Field> = vec![Field::new(group_col, ArrowDataType::Utf8, false)];
        let mut arrays: Vec<ArrayRef> = vec![Arc::new(StringArray::from(group_values))];

        for (ai, (_, _, func, alias)) in agg_info.iter().enumerate() {
            let col_name = alias.as_deref().unwrap_or(match func {
                AggregateFunc::Count => "COUNT(*)",
                AggregateFunc::Avg => "AVG",
                AggregateFunc::Sum => "SUM",
                AggregateFunc::Min => "MIN",
                AggregateFunc::Max => "MAX",
            });

            let values: Vec<f64> = raw
                .iter()
                .map(|(_, aggs)| {
                    let (sum, count) = aggs[ai];
                    match func {
                        AggregateFunc::Count => count as f64,
                        AggregateFunc::Avg => {
                            if count > 0 {
                                sum / count as f64
                            } else {
                                0.0
                            }
                        }
                        AggregateFunc::Sum => sum,
                        _ => sum,
                    }
                })
                .collect();

            // Use Int64 for COUNT, Float64 for others
            if matches!(func, AggregateFunc::Count) {
                let int_values: Vec<i64> = values.iter().map(|v| *v as i64).collect();
                fields.push(Field::new(col_name, ArrowDataType::Int64, false));
                arrays.push(Arc::new(Int64Array::from(int_values)));
            } else {
                fields.push(Field::new(col_name, ArrowDataType::Float64, false));
                arrays.push(Arc::new(Float64Array::from(values)));
            }
        }

        // Apply HAVING if present
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays).map_err(|e| err_data(e.to_string()))?;

        let mut result = if let Some(having) = &stmt.having {
            let mask = Self::evaluate_predicate(&batch, having)?;
            let filtered = arrow::compute::filter_record_batch(&batch, &mask)
                .map_err(|e| err_data(e.to_string()))?;
            if filtered.num_rows() == 0 {
                return Ok(Some(ApexResult::Empty(filtered.schema())));
            }
            filtered
        } else {
            batch
        };

        // Apply ORDER BY with aggregate expression resolver
        if !stmt.order_by.is_empty() {
            let resolved_ob = Self::resolve_order_by_cols(&stmt.columns, &stmt.order_by);
            let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
            result = Self::apply_order_by_topk(&result, &resolved_ob, k)?;
        }

        // Apply LIMIT + OFFSET
        if stmt.limit.is_some() || stmt.offset.is_some() {
            result = Self::apply_limit_offset(&result, stmt.limit, stmt.offset)?;
        }

        Ok(Some(ApexResult::Data(result)))
    }

    /// Extract a simple comparison filter from a WHERE clause for pushdown
    /// into file readers (CSV/JSON/Parquet). Returns "col>val" style string.
    fn try_extract_filter_for_pushdown(expr: &SqlExpr) -> Option<String> {
        use crate::query::sql_parser::BinaryOperator;
        if let SqlExpr::BinaryOp { left, op, right } = expr {
            if let (SqlExpr::Column(col), SqlExpr::Literal(val)) = (left.as_ref(), right.as_ref()) {
                let col_name = col.trim_matches('"');
                let col_name = if let Some(d) = col_name.rfind('.') { &col_name[d + 1..] } else { col_name };
                let op_str = match op {
                    BinaryOperator::Gt => ">",
                    BinaryOperator::Lt => "<",
                    BinaryOperator::Ge => ">=",
                    BinaryOperator::Le => "<=",
                    BinaryOperator::Eq => "=",
                    BinaryOperator::NotEq => "!=",
                    _ => return None,
                };
                let val_str = match val {
                    crate::data::Value::Int64(v) => v.to_string(),
                    crate::data::Value::Int32(v) => v.to_string(),
                    crate::data::Value::Float64(v) => v.to_string(),
                    crate::data::Value::Float32(v) => v.to_string(),
                    crate::data::Value::String(v) => format!("'{}'", v),
                    _ => return None,
                };
                return Some(format!("{}{}{}", col_name, op_str, val_str));
            }
        }
        None
    }

    fn count_star_output_name_for_table_fn(stmt: &SelectStatement) -> Option<String> {
        if stmt.columns.len() != 1
            || !stmt.group_by.is_empty()
            || stmt.having.is_some()
            || !stmt.joins.is_empty()
            || !stmt.order_by.is_empty()
            || stmt.limit.is_some()
            || stmt.offset.is_some()
        {
            return None;
        }

        match &stmt.columns[0] {
            SelectColumn::Aggregate {
                func,
                column,
                distinct,
                alias,
            } if matches!(func, AggregateFunc::Count) && !distinct => {
                let column_ok = column
                    .as_ref()
                    .map(|c| {
                        c == "*"
                            || c.chars()
                                .next()
                                .map(|ch| ch.is_ascii_digit())
                                .unwrap_or(false)
                    })
                    .unwrap_or(true);
                if column_ok {
                    Some(alias.clone().unwrap_or_else(|| "COUNT(*)".to_string()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn try_fast_json_count_table_function(
        stmt: &SelectStatement,
        func: &str,
        file: &str,
    ) -> io::Result<Option<RecordBatch>> {
        if !func.eq_ignore_ascii_case("READ_JSON") {
            return Ok(None);
        }
        let Some(output_name) = Self::count_star_output_name_for_table_fn(stmt) else {
            return Ok(None);
        };
        let Some(count) = Self::try_fast_json_count(file, stmt.where_clause.as_ref())? else {
            return Ok(None);
        };

        let schema = Arc::new(Schema::new(vec![Field::new(
            &output_name,
            ArrowDataType::Int64,
            false,
        )]));
        let array: ArrayRef = Arc::new(Int64Array::from(vec![count]));
        RecordBatch::try_new(schema, vec![array])
            .map(Some)
            .map_err(|e| err_data(e.to_string()))
    }

    /// V4 FAST PATH: Simple aggregation (no GROUP BY, no WHERE)
    /// Handles: SELECT COUNT(*), AVG(col), SUM(col), MIN(col), MAX(col) FROM table
    fn try_fast_simple_agg(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<ApexResult>> {
        if backend.has_pending_deltas() || backend.is_mmap_only() {
            return Ok(None);
        }

        use crate::query::AggregateFunc;

        // Collect unique column names needed for aggregation
        let mut unique_cols: Vec<String> = Vec::new();
        for col in &stmt.columns {
            if let SelectColumn::Aggregate {
                func,
                column,
                distinct,
                ..
            } = col
            {
                if *distinct {
                    return Ok(None);
                } // DISTINCT needs full scan
                let name = column.as_deref().unwrap_or("*");
                if name == "_id" {
                    return Ok(None);
                } // _id stored separately
                   // COUNT(col) and AVG(col) must exclude NULLs: check zone maps first.
                   // If the column has no NULLs, the storage fast path is safe.
                let is_star_or_const = name == "*"
                    || name
                        .chars()
                        .next()
                        .map(|c| c.is_ascii_digit())
                        .unwrap_or(false);
                if !is_star_or_const {
                    if matches!(func, AggregateFunc::Count | AggregateFunc::Avg) {
                        if backend.column_has_nulls(name) {
                            return Ok(None);
                        }
                    }
                }
                if !unique_cols.contains(&name.to_string()) {
                    unique_cols.push(name.to_string());
                }
            } else {
                return Ok(None); // Non-aggregate column present
            }
        }
        if unique_cols.is_empty() {
            return Ok(None);
        }

        let col_refs: Vec<&str> = unique_cols.iter().map(|s| s.as_str()).collect();
        let raw = match backend.execute_simple_agg(&col_refs)? {
            Some(r) => r,
            None => return Ok(None),
        };

        // Build result
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        for col in &stmt.columns {
            if let SelectColumn::Aggregate {
                func,
                column,
                alias,
                ..
            } = col
            {
                let col_name = column.as_deref().unwrap_or("*");
                let fn_name = match func {
                    AggregateFunc::Count => "COUNT",
                    AggregateFunc::Sum => "SUM",
                    AggregateFunc::Avg => "AVG",
                    AggregateFunc::Min => "MIN",
                    AggregateFunc::Max => "MAX",
                };
                let output_name = alias.clone().unwrap_or_else(|| {
                    if let Some(c) = column {
                        format!("{}({})", fn_name, c)
                    } else {
                        format!("{}(*)", fn_name)
                    }
                });

                let idx = unique_cols.iter().position(|s| s == col_name).unwrap_or(0);
                let (count, sum, min_v, max_v, is_int) = raw[idx];

                match func {
                    AggregateFunc::Count => {
                        fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                        arrays.push(Arc::new(Int64Array::from(vec![count])));
                    }
                    AggregateFunc::Sum => {
                        if is_int {
                            fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                            arrays.push(Arc::new(Int64Array::from(vec![sum as i64])));
                        } else {
                            fields.push(Field::new(&output_name, ArrowDataType::Float64, false));
                            arrays.push(Arc::new(Float64Array::from(vec![sum])));
                        }
                    }
                    AggregateFunc::Avg => {
                        let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                        fields.push(Field::new(&output_name, ArrowDataType::Float64, false));
                        arrays.push(Arc::new(Float64Array::from(vec![avg])));
                    }
                    AggregateFunc::Min => {
                        if is_int {
                            fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                            arrays.push(Arc::new(Int64Array::from(vec![min_v as i64])));
                        } else {
                            fields.push(Field::new(&output_name, ArrowDataType::Float64, false));
                            arrays.push(Arc::new(Float64Array::from(vec![min_v])));
                        }
                    }
                    AggregateFunc::Max => {
                        if is_int {
                            fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                            arrays.push(Arc::new(Int64Array::from(vec![max_v as i64])));
                        } else {
                            fields.push(Field::new(&output_name, ArrowDataType::Float64, false));
                            arrays.push(Arc::new(Float64Array::from(vec![max_v])));
                        }
                    }
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays).map_err(|e| err_data(e.to_string()))?;
        Ok(Some(ApexResult::Data(batch)))
    }

    /// V4 FAST PATH: Filtered aggregation with string equality WHERE
    /// Handles: SELECT COUNT(*), AVG(col), MAX(col) FROM table WHERE str_col = 'val'
    /// Scans string column for matching indices, then computes aggregates directly.
    fn try_fast_filtered_string_agg(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<ApexResult>> {
        use crate::query::AggregateFunc;

        if backend.pending_v4_in_memory_rows() > 0 || backend.has_pending_deltas() || backend.has_delta()
        {
            return Ok(None);
        }

        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };

        let (filter_col, filter_val) = match Self::extract_string_equality(where_clause) {
            Some(v) => v,
            None => return Ok(None),
        };

        // Collect unique aggregation columns. Add "*" when COUNT(*)/COUNT(1)
        // is present so the storage path returns the real match count.
        let mut unique_cols: Vec<String> = Vec::new();
        for col in &stmt.columns {
            if let SelectColumn::Aggregate {
                func,
                column,
                distinct,
                ..
            } = col
            {
                if *distinct {
                    return Ok(None);
                }
                if let Some(col_name) = column {
                    let is_count_star = matches!(func, AggregateFunc::Count)
                        && (col_name.as_str() == "*"
                            || col_name
                                .chars()
                                .next()
                                .map(|c| c.is_ascii_digit())
                                .unwrap_or(false));
                    if is_count_star {
                        if !unique_cols.iter().any(|c| c == "*") {
                            unique_cols.push("*".to_string());
                        }
                    } else if !unique_cols.contains(col_name) {
                        unique_cols.push(col_name.clone());
                    }
                } else if !matches!(func, AggregateFunc::Count) {
                    return Ok(None);
                } else if !unique_cols.iter().any(|c| c == "*") {
                    unique_cols.push("*".to_string());
                }
            } else {
                return Ok(None);
            }
        }

        let col_refs: Vec<&str> = unique_cols.iter().map(|s| s.as_str()).collect();

        // Single-pass: scan string filter + aggregate in one sequential pass
        use std::collections::HashMap;
        let agg_results = match backend.execute_filtered_string_agg_mmap(
            &filter_col,
            &filter_val,
            &col_refs,
        )? {
            Some(r) => r,
            None => return Ok(None),
        };

        // Build stat lookup: column name -> (count, sum, min, max, is_int)
        let mut stat_map: HashMap<&str, (i64, f64, f64, f64, bool)> = HashMap::new();
        for (i, &col_name) in col_refs.iter().enumerate() {
            if i < agg_results.len() {
                stat_map.insert(col_name, agg_results[i]);
            }
        }

        let match_count = stat_map.get("*").map(|s| s.0).unwrap_or(0);

        // Build result
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        for col in &stmt.columns {
            if let SelectColumn::Aggregate {
                func,
                column,
                alias,
                ..
            } = col
            {
                let fn_name = match func {
                    AggregateFunc::Count => "COUNT",
                    AggregateFunc::Sum => "SUM",
                    AggregateFunc::Avg => "AVG",
                    AggregateFunc::Min => "MIN",
                    AggregateFunc::Max => "MAX",
                };
                let output_name = alias.clone().unwrap_or_else(|| {
                    if let Some(c) = column {
                        format!("{}({})", fn_name, c)
                    } else {
                        format!("{}(*)", fn_name)
                    }
                });

                match func {
                    AggregateFunc::Count => {
                        let count = if let Some(col_name) = column {
                            let is_count_star = col_name.as_str() == "*"
                                || col_name
                                    .chars()
                                    .next()
                                    .map(|c| c.is_ascii_digit())
                                    .unwrap_or(false);
                            if is_count_star {
                                match_count
                            } else {
                                stat_map.get(col_name.as_str()).map(|s| s.0).unwrap_or(0)
                            }
                        } else {
                            match_count
                        };
                        fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                        arrays.push(Arc::new(Int64Array::from(vec![count])));
                    }
                    AggregateFunc::Sum
                    | AggregateFunc::Avg
                    | AggregateFunc::Min
                    | AggregateFunc::Max => {
                        let col_name = column.as_ref().unwrap();
                        let (count, sum, min_v, max_v, is_int) = stat_map
                            .get(col_name.as_str())
                            .copied()
                            .unwrap_or((0, 0.0, 0.0, 0.0, false));

                        match func {
                            AggregateFunc::Sum => {
                                if is_int {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Int64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Int64Array::from(vec![sum as i64])));
                                } else {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Float64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Float64Array::from(vec![sum])));
                                }
                            }
                            AggregateFunc::Avg => {
                                let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                                fields.push(Field::new(&output_name, ArrowDataType::Float64, true));
                                arrays.push(Arc::new(Float64Array::from(vec![avg])));
                            }
                            AggregateFunc::Min => {
                                if count == 0 {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Float64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Float64Array::from(vec![None::<f64>])));
                                } else if is_int {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Int64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Int64Array::from(vec![min_v as i64])));
                                } else {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Float64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Float64Array::from(vec![min_v])));
                                }
                            }
                            AggregateFunc::Max => {
                                if count == 0 {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Float64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Float64Array::from(vec![None::<f64>])));
                                } else if is_int {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Int64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Int64Array::from(vec![max_v as i64])));
                                } else {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Float64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Float64Array::from(vec![max_v])));
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
        }

        if fields.is_empty() {
            return Ok(None);
        }
        let schema = Arc::new(Schema::new(fields));
        let result = RecordBatch::try_new(schema, arrays).map_err(|e| err_data(e.to_string()))?;
        Ok(Some(ApexResult::Data(result)))
    }

    /// V4 FAST PATH: Filtered aggregation with a numeric WHERE predicate.
    /// Handles: SELECT COUNT(*), AVG(col), MAX(col) FROM table WHERE num_col > value
    fn try_fast_filtered_numeric_agg(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<ApexResult>> {
        use crate::query::AggregateFunc;

        if backend.pending_v4_in_memory_rows() > 0 || backend.has_pending_deltas() || backend.has_delta()
        {
            return Ok(None);
        }

        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };
        let (filter_col, low, high) = match Self::extract_any_numeric_range(where_clause) {
            Some(v) => v,
            None => return Ok(None),
        };

        let mut unique_cols: Vec<String> = Vec::new();
        for col in &stmt.columns {
            if let SelectColumn::Aggregate {
                func,
                column,
                distinct,
                ..
            } = col
            {
                if *distinct {
                    return Ok(None);
                }
                if let Some(col_name) = column {
                    let is_count_star = matches!(func, AggregateFunc::Count)
                        && (col_name.as_str() == "*"
                            || col_name
                                .chars()
                                .next()
                                .map(|c| c.is_ascii_digit())
                                .unwrap_or(false));
                    if is_count_star {
                        if !unique_cols.iter().any(|c| c == "*") {
                            unique_cols.push("*".to_string());
                        }
                    } else if !unique_cols.contains(col_name) {
                        unique_cols.push(col_name.clone());
                    }
                } else if !matches!(func, AggregateFunc::Count) {
                    return Ok(None);
                } else if !unique_cols.iter().any(|c| c == "*") {
                    unique_cols.push("*".to_string());
                }
            } else {
                return Ok(None);
            }
        }
        if unique_cols.is_empty() {
            return Ok(None);
        }

        let col_refs: Vec<&str> = unique_cols.iter().map(|s| s.as_str()).collect();
        let agg_results =
            match backend.execute_filtered_numeric_agg_mmap(&filter_col, low, high, &col_refs)? {
                Some(r) => r,
                None => return Ok(None),
        };

        use std::collections::HashMap;
        let mut stat_map: HashMap<&str, (i64, f64, f64, f64, bool)> = HashMap::new();
        for (i, &col_name) in col_refs.iter().enumerate() {
            if i < agg_results.len() {
                stat_map.insert(col_name, agg_results[i]);
            }
        }
        let match_count = stat_map.get("*").map(|s| s.0).unwrap_or(0);

        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        for col in &stmt.columns {
            if let SelectColumn::Aggregate {
                func,
                column,
                alias,
                ..
            } = col
            {
                let fn_name = match func {
                    AggregateFunc::Count => "COUNT",
                    AggregateFunc::Sum => "SUM",
                    AggregateFunc::Avg => "AVG",
                    AggregateFunc::Min => "MIN",
                    AggregateFunc::Max => "MAX",
                };
                let output_name = alias.clone().unwrap_or_else(|| {
                    if let Some(c) = column {
                        format!("{}({})", fn_name, c)
                    } else {
                        format!("{}(*)", fn_name)
                    }
                });

                match func {
                    AggregateFunc::Count => {
                        let count = if let Some(col_name) = column {
                            let is_count_star = col_name.as_str() == "*"
                                || col_name
                                    .chars()
                                    .next()
                                    .map(|c| c.is_ascii_digit())
                                    .unwrap_or(false);
                            if is_count_star {
                                match_count
                            } else {
                                stat_map.get(col_name.as_str()).map(|s| s.0).unwrap_or(0)
                            }
                        } else {
                            match_count
                        };
                        fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                        arrays.push(Arc::new(Int64Array::from(vec![count])));
                    }
                    AggregateFunc::Sum
                    | AggregateFunc::Avg
                    | AggregateFunc::Min
                    | AggregateFunc::Max => {
                        let col_name = column.as_ref().unwrap();
                        let (count, sum, min_v, max_v, is_int) = stat_map
                            .get(col_name.as_str())
                            .copied()
                            .unwrap_or((0, 0.0, 0.0, 0.0, false));

                        match func {
                            AggregateFunc::Sum => {
                                if is_int {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Int64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Int64Array::from(vec![sum as i64])));
                                } else {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Float64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Float64Array::from(vec![sum])));
                                }
                            }
                            AggregateFunc::Avg => {
                                let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                                fields.push(Field::new(&output_name, ArrowDataType::Float64, true));
                                arrays.push(Arc::new(Float64Array::from(vec![avg])));
                            }
                            AggregateFunc::Min => {
                                if count == 0 {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Float64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Float64Array::from(vec![None::<f64>])));
                                } else if is_int {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Int64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Int64Array::from(vec![min_v as i64])));
                                } else {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Float64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Float64Array::from(vec![min_v])));
                                }
                            }
                            AggregateFunc::Max => {
                                if count == 0 {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Float64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Float64Array::from(vec![None::<f64>])));
                                } else if is_int {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Int64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Int64Array::from(vec![max_v as i64])));
                                } else {
                                    fields.push(Field::new(
                                        &output_name,
                                        ArrowDataType::Float64,
                                        true,
                                    ));
                                    arrays.push(Arc::new(Float64Array::from(vec![max_v])));
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
        }

        if fields.is_empty() {
            return Ok(None);
        }
        let schema = Arc::new(Schema::new(fields));
        let result = RecordBatch::try_new(schema, arrays).map_err(|e| err_data(e.to_string()))?;
        Ok(Some(ApexResult::Data(result)))
    }

    /// V4 FAST PATH: Cached GROUP BY (builds dict cache on first call, reuses on subsequent calls)
    fn try_fast_cached_group_by(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<ApexResult>> {
        if backend.pending_v4_in_memory_rows() > 0
            || backend.has_pending_deltas()
            || backend.has_delta()
        {
            return Ok(None);
        }

        use crate::query::AggregateFunc;

        // Must be 1 or 2 GROUP BY columns, no WHERE.
        // ORDER BY/LIMIT can be applied after the cached aggregate result is built.
        if stmt.group_by.is_empty() || stmt.group_by.len() > 2 || stmt.where_clause.is_some() {
            return Ok(None);
        }

        // Handle 2-column GROUP BY as a separate fast path
        if stmt.group_by.len() == 2 {
            return Self::try_fast_cached_group_by_2col(backend, stmt);
        }

        // Must be single GROUP BY column, no WHERE
        if stmt.group_by.len() != 1 {
            return Ok(None);
        }

        let group_col = stmt.group_by[0].trim_matches('"');

        // Extract aggregate info
        let mut agg_info: Vec<(&str, bool, AggregateFunc, Option<String>)> = Vec::new();
        for col in &stmt.columns {
            match col {
                SelectColumn::Aggregate {
                    func,
                    column,
                    alias,
                    ..
                } => {
                    let is_count_star = matches!(func, AggregateFunc::Count) && column.is_none();
                    let col_name = column.as_deref().unwrap_or("*");
                    agg_info.push((col_name, is_count_star, func.clone(), alias.clone()));
                }
                SelectColumn::Column(name) => {
                    if name.trim_matches('"') == group_col {
                        continue;
                    }
                    return Ok(None);
                }
                SelectColumn::ColumnAlias { column, .. } => {
                    if column.trim_matches('"') == group_col {
                        continue;
                    }
                    return Ok(None);
                }
                _ => return Ok(None),
            }
        }
        if agg_info.is_empty() {
            return Ok(None);
        }

        // Get or build cached dict (global cache — survives backend reopens)
        let dict_arc = match crate::storage::backend::get_global_dict_cache(
            backend.path(),
            group_col,
            &backend.storage,
        )? {
            Some(c) => c,
            None => return Ok(None),
        };
        let (dict_strings, group_ids) = (dict_arc.0.as_slice(), dict_arc.1.as_slice());

        let agg_cols: Vec<(&str, bool)> = agg_info
            .iter()
            .map(|(col, is_count, _, _)| (*col, *is_count))
            .collect();

        let raw = match backend.execute_group_agg_cached(dict_strings, group_ids, &agg_cols)? {
            Some(r) if !r.is_empty() => r,
            _ => return Ok(None),
        };

        // Build result
        let num_groups = raw.len();
        let group_values: Vec<&str> = raw.iter().map(|(k, _)| k.as_str()).collect();

        let mut fields: Vec<Field> = vec![Field::new(group_col, ArrowDataType::Utf8, false)];
        let mut arrays: Vec<ArrayRef> = vec![Arc::new(StringArray::from(group_values))];

        for (ai, (agg_col, _, func, alias)) in agg_info.iter().enumerate() {
            let output_name;
            let col_name = match alias {
                Some(alias) => alias.as_str(),
                None => {
                    output_name = match func {
                        AggregateFunc::Count if *agg_col == "*" => "COUNT(*)".to_string(),
                        AggregateFunc::Count => format!("COUNT({})", agg_col),
                        AggregateFunc::Avg => format!("AVG({})", agg_col),
                        AggregateFunc::Sum => format!("SUM({})", agg_col),
                        AggregateFunc::Min => format!("MIN({})", agg_col),
                        AggregateFunc::Max => format!("MAX({})", agg_col),
                    };
                    output_name.as_str()
                }
            };
            let values: Vec<f64> = raw
                .iter()
                .map(|(_, aggs)| {
                    let (sum, count) = aggs[ai];
                    match func {
                        AggregateFunc::Count => count as f64,
                        AggregateFunc::Avg => {
                            if count > 0 {
                                sum / count as f64
                            } else {
                                0.0
                            }
                        }
                        _ => sum,
                    }
                })
                .collect();
            if matches!(func, AggregateFunc::Count) {
                let int_values: Vec<i64> = values.iter().map(|v| *v as i64).collect();
                fields.push(Field::new(col_name, ArrowDataType::Int64, false));
                arrays.push(Arc::new(Int64Array::from(int_values)));
            } else {
                fields.push(Field::new(col_name, ArrowDataType::Float64, false));
                arrays.push(Arc::new(Float64Array::from(values)));
            }
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays).map_err(|e| err_data(e.to_string()))?;

        // Apply HAVING
        let mut result = if let Some(having) = &stmt.having {
            let mask = Self::evaluate_predicate(&batch, having)?;
            let filtered = arrow::compute::filter_record_batch(&batch, &mask)
                .map_err(|e| err_data(e.to_string()))?;
            if filtered.num_rows() == 0 {
                return Ok(Some(ApexResult::Empty(filtered.schema())));
            }
            filtered
        } else {
            batch
        };

        if !stmt.order_by.is_empty() {
            let resolved_ob = Self::resolve_order_by_cols(&stmt.columns, &stmt.order_by);
            let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
            result = Self::apply_order_by_topk(&result, &resolved_ob, k)?;
        }

        if stmt.limit.is_some() || stmt.offset.is_some() {
            result = Self::apply_limit_offset(&result, stmt.limit, stmt.offset)?;
        }

        Ok(Some(ApexResult::Data(result)))
    }

    /// 2-column GROUP BY fast path using dict caches for both columns.
    fn try_fast_cached_group_by_2col(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<ApexResult>> {
        use crate::query::AggregateFunc;

        let group_col1 = stmt.group_by[0].trim_matches('"');
        let group_col2 = stmt.group_by[1].trim_matches('"');

        // Extract aggregate info (support multiple aggregates)
        let mut agg_info: Vec<(&str, bool, AggregateFunc, Option<String>)> = Vec::new();
        for col in &stmt.columns {
            match col {
                SelectColumn::Aggregate {
                    func,
                    column,
                    alias,
                    ..
                } => {
                    let is_count_star = matches!(func, AggregateFunc::Count) && column.is_none();
                    let col_name = column.as_deref().unwrap_or("*");
                    agg_info.push((col_name, is_count_star, func.clone(), alias.clone()));
                }
                SelectColumn::Column(name) => {
                    let n = name.trim_matches('"');
                    if n == group_col1 || n == group_col2 {
                        continue;
                    }
                    return Ok(None);
                }
                SelectColumn::ColumnAlias { column, .. } => {
                    let n = column.trim_matches('"');
                    if n == group_col1 || n == group_col2 {
                        continue;
                    }
                    return Ok(None);
                }
                _ => return Ok(None),
            }
        }
        if agg_info.is_empty() {
            return Ok(None);
        }

        // Get dict caches for both group columns
        let dict1_arc = match crate::storage::backend::get_global_dict_cache(
            backend.path(),
            group_col1,
            &backend.storage,
        )? {
            Some(c) => c,
            None => return Ok(None),
        };
        let dict2_arc = match crate::storage::backend::get_global_dict_cache(
            backend.path(),
            group_col2,
            &backend.storage,
        )? {
            Some(c) => c,
            None => return Ok(None),
        };

        let (dict1_strings, group_ids1) = (dict1_arc.0.as_slice(), dict1_arc.1.as_slice());
        let (dict2_strings, group_ids2) = (dict2_arc.0.as_slice(), dict2_arc.1.as_slice());

        let agg_cols: Vec<(&str, bool)> = agg_info
            .iter()
            .map(|(col, is_count, _, _)| (*col, *is_count))
            .collect();

        let raw = match backend.execute_group_agg_2col_cached(
            dict1_strings,
            group_ids1,
            dict2_strings,
            group_ids2,
            &agg_cols,
        )? {
            Some(r) if !r.is_empty() => r,
            _ => return Ok(None),
        };

        // Build result RecordBatch
        let col1_vals: Vec<&str> = raw.iter().map(|((k1, _), _)| k1.as_str()).collect();
        let col2_vals: Vec<&str> = raw.iter().map(|((_, k2), _)| k2.as_str()).collect();

        let mut fields: Vec<Field> = vec![
            Field::new(group_col1, ArrowDataType::Utf8, false),
            Field::new(group_col2, ArrowDataType::Utf8, false),
        ];
        let mut arrays: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(col1_vals)),
            Arc::new(StringArray::from(col2_vals)),
        ];

        for (ai, (_, _, func, alias)) in agg_info.iter().enumerate() {
            let col_name = alias.as_deref().unwrap_or(match func {
                AggregateFunc::Count => "COUNT(*)",
                AggregateFunc::Avg => "AVG",
                AggregateFunc::Sum => "SUM",
                AggregateFunc::Min => "MIN",
                AggregateFunc::Max => "MAX",
            });
            match func {
                AggregateFunc::Count => {
                    let vals: Vec<i64> = raw.iter().map(|(_, aggs)| aggs[ai].1).collect();
                    fields.push(Field::new(col_name, ArrowDataType::Int64, false));
                    arrays.push(Arc::new(Int64Array::from(vals)));
                }
                AggregateFunc::Avg => {
                    let vals: Vec<f64> = raw
                        .iter()
                        .map(|(_, aggs)| {
                            let (sum, cnt) = aggs[ai];
                            if cnt > 0 {
                                sum / cnt as f64
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    fields.push(Field::new(col_name, ArrowDataType::Float64, false));
                    arrays.push(Arc::new(Float64Array::from(vals)));
                }
                AggregateFunc::Sum | AggregateFunc::Min | AggregateFunc::Max => {
                    let vals: Vec<f64> = raw.iter().map(|(_, aggs)| aggs[ai].0).collect();
                    fields.push(Field::new(col_name, ArrowDataType::Float64, false));
                    arrays.push(Arc::new(Float64Array::from(vals)));
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));
        let mut batch =
            RecordBatch::try_new(schema, arrays).map_err(|e| err_data(e.to_string()))?;

        // Apply HAVING
        if let Some(having) = &stmt.having {
            let mask = Self::evaluate_predicate(&batch, having)?;
            batch = arrow::compute::filter_record_batch(&batch, &mask)
                .map_err(|e| err_data(e.to_string()))?;
        }

        if batch.num_rows() == 0 {
            return Ok(Some(ApexResult::Empty(batch.schema())));
        }
        Ok(Some(ApexResult::Data(batch)))
    }

    /// Helper to extract LIKE pattern: col LIKE 'pattern' (non-negated only)
    fn extract_like_pattern(expr: &SqlExpr) -> Option<(String, String)> {
        match expr {
            SqlExpr::Like {
                column,
                pattern,
                negated,
            } if !negated => Some((column.trim_matches('"').to_string(), pattern.clone())),
            _ => None,
        }
    }

    /// Helper to extract string equality: col = 'value'
    fn extract_string_equality(expr: &SqlExpr) -> Option<(String, String)> {
        use crate::query::sql_parser::BinaryOperator;
        match expr {
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::Eq,
                right,
            } => match (left.as_ref(), right.as_ref()) {
                (SqlExpr::Column(col), SqlExpr::Literal(Value::String(val)))
                | (SqlExpr::Literal(Value::String(val)), SqlExpr::Column(col)) => {
                    Some((col.trim_matches('"').to_string(), val.clone()))
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Helper to extract BETWEEN range: col BETWEEN low AND high
    fn extract_between_range(expr: &SqlExpr) -> Option<(String, f64, f64)> {
        match expr {
            SqlExpr::Between {
                column,
                low,
                high,
                negated,
            } => {
                if *negated {
                    return None;
                }
                let col = column.trim_matches('"').to_string();
                let low_val = Self::extract_numeric_value(low).ok()?;
                let high_val = Self::extract_numeric_value(high).ok()?;
                Some((col, low_val, high_val))
            }
            _ => None,
        }
    }

    /// Convert a single-sided numeric comparison to an inclusive range for scan_numeric_range_mmap.
    /// col > N  → (col, next_f64(N), MAX)   exclusive lower bound via next representable f64
    /// col >= N → (col, N, MAX)
    /// col < N  → (col, MIN, prev_f64(N))   exclusive upper bound via prev representable f64
    /// col <= N → (col, MIN, N)
    fn extract_single_comparison_as_range(expr: &SqlExpr) -> Option<(String, f64, f64)> {
        use crate::query::sql_parser::BinaryOperator;
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                // col OP literal  OR  literal OP col (reversed)
                let (col, effective_op, val) = match (left.as_ref(), right.as_ref()) {
                    (SqlExpr::Column(c), lit) => {
                        let v = Self::extract_numeric_value(lit).ok()?;
                        (c.trim_matches('"').to_string(), op.clone(), v)
                    }
                    (lit, SqlExpr::Column(c)) => {
                        let v = Self::extract_numeric_value(lit).ok()?;
                        // Flip: N > col → col < N
                        let flipped = match op {
                            BinaryOperator::Gt => BinaryOperator::Lt,
                            BinaryOperator::Ge => BinaryOperator::Le,
                            BinaryOperator::Lt => BinaryOperator::Gt,
                            BinaryOperator::Le => BinaryOperator::Ge,
                            _ => return None,
                        };
                        (c.trim_matches('"').to_string(), flipped, v)
                    }
                    _ => return None,
                };
                // Return next/prev representable f64 for strict inequalities so that
                // scan_numeric_range_mmap (which uses inclusive bounds) is exact.
                let (low, high) = match effective_op {
                    BinaryOperator::Gt => {
                        // col > N: smallest representable value strictly above N
                        let next = if val >= 0.0 {
                            f64::from_bits(val.to_bits() + 1)
                        } else {
                            f64::from_bits(val.to_bits() - 1)
                        };
                        (next, f64::MAX)
                    }
                    BinaryOperator::Ge => (val, f64::MAX),
                    BinaryOperator::Lt => {
                        // col < N: largest representable value strictly below N
                        let prev = if val > 0.0 {
                            f64::from_bits(val.to_bits() - 1)
                        } else {
                            f64::from_bits(val.to_bits() + 1)
                        };
                        (f64::MIN, prev)
                    }
                    BinaryOperator::Le => (f64::MIN, val),
                    BinaryOperator::Eq => (val, val), // exact match as degenerate range [N, N]
                    _ => return None,
                };
                Some((col, low, high))
            }
            _ => None,
        }
    }

    /// Extract a two-sided AND range on the SAME column: col >= N AND col <= M etc.
    /// Returns (col, inclusive_low, inclusive_high) with strict-inequality adjustment.
    /// Each side is extracted via extract_single_comparison_as_range; the intersection
    /// of the two (lo, hi) intervals gives the final range.
    fn extract_two_sided_same_col_range(expr: &SqlExpr) -> Option<(String, f64, f64)> {
        use crate::query::sql_parser::BinaryOperator;
        match expr {
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::And,
                right,
            } => {
                let (col1, lo1, hi1) = Self::extract_single_comparison_as_range(left.as_ref())?;
                let (col2, lo2, hi2) = Self::extract_single_comparison_as_range(right.as_ref())?;
                if col1 != col2 {
                    return None;
                }
                let combined_low = lo1.max(lo2);
                let combined_high = hi1.min(hi2);
                if combined_low > combined_high {
                    return None;
                }
                Some((col1, combined_low, combined_high))
            }
            _ => None,
        }
    }

    /// Extract any single-column numeric range from an expression.
    /// Handles: BETWEEN, col op N (single comparison including equality).
    fn extract_any_numeric_range(expr: &SqlExpr) -> Option<(String, f64, f64)> {
        if let Some(r) = Self::extract_between_range(expr) {
            return Some(r);
        }
        if let Some(r) = Self::extract_single_comparison_as_range(expr) {
            return Some(r);
        }
        None
    }

    /// Merge-intersect two sorted index slices in O(n+m).
    fn intersect_sorted_indices(a: &[usize], b: &[usize]) -> Vec<usize> {
        let mut result = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Equal => {
                    result.push(a[i]);
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        result
    }

    /// MMAP fast path for AND of two numeric conditions on DIFFERENT columns.
    /// Example: WHERE age > 30 AND score > 50 [LIMIT n]
    /// Strategy: scan each column independently → merge-intersect sorted index sets → scatter read.
    fn try_fast_mmap_multi_condition(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
        storage_path: &Path,
    ) -> io::Result<Option<ApexResult>> {
        use crate::query::sql_parser::BinaryOperator;
        if !backend.is_mmap_only() || backend.has_pending_deltas() || backend.has_delta() {
            return Ok(None);
        }
        // Without LIMIT the result set can be very large; sequential Arrow scan is faster
        // than index intersection + scatter read for high-selectivity filters.
        if stmt.limit.is_none() {
            return Ok(None);
        }

        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };
        // Must be top-level AND
        let (left_cond, right_cond) = match where_clause {
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::And,
                right,
            } => (left.as_ref(), right.as_ref()),
            _ => return Ok(None),
        };

        // --- Case A: numeric AND numeric (two different columns) ---
        if let (Some((col1, lo1, hi1)), Some((col2, lo2, hi2))) = (
            Self::extract_any_numeric_range(left_cond),
            Self::extract_any_numeric_range(right_cond),
        ) {
            if col1 != col2 {
                let idxs1 = match backend.scan_numeric_range_mmap(&col1, lo1, hi1, None)? {
                    Some(v) => v,
                    None => return Ok(None),
                };
                let idxs2 = match backend.scan_numeric_range_mmap(&col2, lo2, hi2, None)? {
                    Some(v) => v,
                    None => return Ok(None),
                };
                let mut intersected = Self::intersect_sorted_indices(&idxs1, &idxs2);
                let offset = stmt.offset.unwrap_or(0);
                if offset > 0 {
                    if offset >= intersected.len() {
                        return Ok(Some(ApexResult::Empty(Arc::new(Schema::empty()))));
                    }
                    intersected = intersected[offset..].to_vec();
                }
                if let Some(lim) = stmt.limit {
                    intersected.truncate(lim);
                }
                if intersected.is_empty() {
                    return Ok(Some(ApexResult::Empty(Arc::new(Schema::empty()))));
                }
                let batch = Self::read_matching_rows_adaptive(backend, stmt, &intersected)?;
                if !stmt.is_pure_star() {
                    let projected = Self::apply_projection_with_storage(
                        &batch,
                        &stmt.columns,
                        Some(storage_path),
                    )?;
                    return Ok(Some(ApexResult::Data(projected)));
                }
                return Ok(Some(ApexResult::Data(batch)));
            }
        }

        // --- Case B: string equality AND numeric range ---
        // Try both orderings: (str, num) and (num, str)
        let str_num = Self::extract_string_equality(left_cond)
            .and_then(|(sc, sv)| Self::extract_any_numeric_range(right_cond).map(|r| (sc, sv, r)))
            .or_else(|| {
                Self::extract_string_equality(right_cond).and_then(|(sc, sv)| {
                    Self::extract_any_numeric_range(left_cond).map(|r| (sc, sv, r))
                })
            });

        let (str_col, str_val, (num_col, num_lo, num_hi)) = match str_num {
            Some(v) => v,
            None => return Ok(None),
        };

        let str_indices = match backend.scan_string_filter_mmap(&str_col, &str_val, None)? {
            Some(v) => v,
            None => return Ok(None),
        };
        let num_indices = match backend.scan_numeric_range_mmap(&num_col, num_lo, num_hi, None)? {
            Some(v) => v,
            None => return Ok(None),
        };

        let mut intersected = Self::intersect_sorted_indices(&str_indices, &num_indices);

        // Apply offset + limit
        let offset = stmt.offset.unwrap_or(0);
        if offset > 0 {
            if offset >= intersected.len() {
                return Ok(Some(ApexResult::Empty(Arc::new(Schema::empty()))));
            }
            intersected = intersected[offset..].to_vec();
        }
        if let Some(lim) = stmt.limit {
            intersected.truncate(lim);
        }

        if intersected.is_empty() {
            return Ok(Some(ApexResult::Empty(Arc::new(Schema::empty()))));
        }

        let batch = Self::read_matching_rows_adaptive(backend, stmt, &intersected)?;
        if !stmt.is_pure_star() {
            let projected =
                Self::apply_projection_with_storage(&batch, &stmt.columns, Some(storage_path))?;
            return Ok(Some(ApexResult::Data(projected)));
        }
        Ok(Some(ApexResult::Data(batch)))
    }

    /// Extract IN list of string values: col IN ('a', 'b', 'c')
    /// Returns (column_name, vec_of_string_values) if all values are strings.
    fn extract_in_string_filter(expr: &SqlExpr) -> Option<(String, Vec<String>)> {
        match expr {
            SqlExpr::In {
                column,
                values,
                negated,
            } => {
                if *negated {
                    return None;
                }
                let col = column.trim_matches('"').to_string();
                let mut strs = Vec::with_capacity(values.len());
                for v in values {
                    match v {
                        Value::String(s) => strs.push(s.clone()),
                        _ => return None,
                    }
                }
                if strs.is_empty() {
                    return None;
                }
                Some((col, strs))
            }
            _ => None,
        }
    }

    /// Extract IN list of numeric (integer) values: col IN (1, 2, 3)
    /// Returns (column_name, vec_of_i64_values) if all values are integers.
    fn extract_in_numeric_filter(expr: &SqlExpr) -> Option<(String, Vec<i64>)> {
        match expr {
            SqlExpr::In {
                column,
                values,
                negated,
            } => {
                if *negated {
                    return None;
                }
                let col = column.trim_matches('"').to_string();
                let mut nums = Vec::with_capacity(values.len());
                for v in values {
                    match v {
                        Value::Int64(n) => nums.push(*n),
                        Value::Int32(n) => nums.push(*n as i64),
                        _ => return None,
                    }
                }
                if nums.is_empty() {
                    return None;
                }
                Some((col, nums))
            }
            _ => None,
        }
    }

    /// Extract OR chain of same-column numeric equalities: col = 1 OR col = 2 OR ...
    /// Returns (column_name, vec_of_i64_values) — equivalent to numeric IN.
    fn extract_or_numeric_equalities(expr: &SqlExpr) -> Option<(String, Vec<i64>)> {
        use crate::query::sql_parser::BinaryOperator;
        let mut values = Vec::new();
        let mut col_name: Option<String> = None;
        Self::collect_or_numeric_equalities(expr, &mut col_name, &mut values)?;
        let col = col_name?;
        if values.len() < 2 {
            return None;
        }
        Some((col, values))
    }

    /// Recursively collect col = N leaves from an OR tree.
    fn collect_or_numeric_equalities(
        expr: &SqlExpr,
        col_name: &mut Option<String>,
        values: &mut Vec<i64>,
    ) -> Option<()> {
        use crate::query::sql_parser::BinaryOperator;
        match expr {
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::Or,
                right,
            } => {
                Self::collect_or_numeric_equalities(left, col_name, values)?;
                Self::collect_or_numeric_equalities(right, col_name, values)?;
                Some(())
            }
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::Eq,
                right,
            } => {
                let (c, v) = match (left.as_ref(), right.as_ref()) {
                    (SqlExpr::Column(c), lit) | (lit, SqlExpr::Column(c)) => {
                        let val = match lit {
                            SqlExpr::Literal(Value::Int64(n)) => *n,
                            SqlExpr::Literal(Value::Int32(n)) => *n as i64,
                            _ => return None,
                        };
                        (c.trim_matches('"').to_string(), val)
                    }
                    _ => return None,
                };
                match col_name {
                    Some(ref existing) => {
                        if *existing != c {
                            return None;
                        }
                    }
                    None => {
                        *col_name = Some(c);
                    }
                }
                values.push(v);
                Some(())
            }
            _ => None,
        }
    }

    /// Decompose an OR tree into leaf predicates that can each be scanned via mmap.
    /// Returns None if any leaf is not a simple scannable predicate.
    fn extract_or_leaf_predicates(expr: &SqlExpr) -> Option<Vec<OrLeafPredicate>> {
        use crate::query::sql_parser::BinaryOperator;
        match expr {
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::Or,
                right,
            } => {
                let mut left_leaves = Self::extract_or_leaf_predicates(left)?;
                let right_leaves = Self::extract_or_leaf_predicates(right)?;
                left_leaves.extend(right_leaves);
                Some(left_leaves)
            }
            _ => {
                // Try to classify this as a single scannable predicate
                if let Some((col, val)) = Self::extract_string_equality(expr) {
                    Some(vec![OrLeafPredicate::StringEq(col, val)])
                } else if let Some((col, low, high)) =
                    Self::extract_single_comparison_as_range(expr)
                {
                    Some(vec![OrLeafPredicate::NumericRange(col, low, high)])
                } else if let Some((col, low, high)) = Self::extract_between_range(expr) {
                    Some(vec![OrLeafPredicate::NumericRange(col, low, high)])
                } else if let Some((col, nums)) = Self::extract_in_numeric_filter(expr) {
                    Some(vec![OrLeafPredicate::NumericIn(col, nums)])
                } else if let Some((col, strs)) = Self::extract_in_string_filter(expr) {
                    Some(vec![OrLeafPredicate::StringIn(col, strs)])
                } else {
                    None
                }
            }
        }
    }

    /// Execute each OR leaf via the appropriate mmap scan, union all index sets.
    /// For 2+ leaves, uses parallel scanning (single mmap lock, rayon dispatch).
    fn scan_or_leaves_mmap(
        backend: &TableStorageBackend,
        leaves: &[OrLeafPredicate],
        limit: Option<usize>,
    ) -> io::Result<Option<Vec<usize>>> {
        use crate::storage::on_demand::MmapScanPred;

        // Try parallel path: convert leaves to MmapScanPred (skip StringIn — rare, needs multi-call)
        let has_string_in = leaves
            .iter()
            .any(|l| matches!(l, OrLeafPredicate::StringIn(..)));
        if leaves.len() >= 2 && !has_string_in {
            let preds: Vec<MmapScanPred> = leaves
                .iter()
                .map(|leaf| match leaf {
                    OrLeafPredicate::StringEq(col, val) => {
                        MmapScanPred::StringEq { col, value: val }
                    }
                    OrLeafPredicate::NumericRange(col, low, high) => MmapScanPred::NumericRange {
                        col,
                        low: *low,
                        high: *high,
                    },
                    OrLeafPredicate::NumericIn(col, nums) => {
                        MmapScanPred::NumericIn { col, values: nums }
                    }
                    OrLeafPredicate::StringIn(..) => unreachable!(),
                })
                .collect();
            if let Some(mut indices) = backend.scan_multi_predicates_parallel(&preds)? {
                if let Some(lim) = limit {
                    indices.truncate(lim);
                }
                return Ok(Some(indices));
            }
            // Parallel path returned None (e.g. compressed data) — fall through to sequential
        }

        // Sequential fallback
        let mut all_indices: Vec<usize> = Vec::new();
        for leaf in leaves {
            let indices = match leaf {
                OrLeafPredicate::StringEq(col, val) => {
                    backend.scan_string_filter_mmap(col, val, None)?
                }
                OrLeafPredicate::NumericRange(col, low, high) => {
                    backend.scan_numeric_range_mmap(col, *low, *high, None)?
                }
                OrLeafPredicate::NumericIn(col, nums) => {
                    backend.scan_numeric_in_mmap(col, nums, None)?
                }
                OrLeafPredicate::StringIn(col, strs) => {
                    backend.scan_string_in_mmap(col, strs, None)?
                }
            };
            if let Some(mut idxs) = indices {
                all_indices.append(&mut idxs);
            }
        }
        if all_indices.is_empty() {
            return Ok(Some(Vec::new()));
        }
        all_indices.sort_unstable();
        all_indices.dedup();
        if let Some(lim) = limit {
            all_indices.truncate(lim);
        }
        Ok(Some(all_indices))
    }

    #[inline]
    fn should_use_scatter_read(total_rows: usize, matched_rows: usize) -> bool {
        matched_rows < 200_000 && matched_rows.saturating_mul(4) < total_rows
    }

    fn take_rows_from_full_batch(
        full_batch: &RecordBatch,
        row_indices: &[usize],
    ) -> io::Result<RecordBatch> {
        use arrow::array::{ArrayRef, UInt32Array};

        let indices_arr =
            UInt32Array::from(row_indices.iter().map(|&i| i as u32).collect::<Vec<_>>());
        let taken_columns: Vec<ArrayRef> = full_batch
            .columns()
            .iter()
            .map(|col| {
                arrow::compute::take(col.as_ref(), &indices_arr, None)
                    .map_err(|e| err_data(e.to_string()))
            })
            .collect::<io::Result<Vec<_>>>()?;
        RecordBatch::try_new(full_batch.schema(), taken_columns)
            .map_err(|e| err_data(e.to_string()))
    }

    fn read_matching_rows_adaptive(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
        row_indices: &[usize],
    ) -> io::Result<RecordBatch> {
        let col_refs = Self::get_col_refs(stmt);
        let col_refs_vec: Option<Vec<&str>> = col_refs
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        if row_indices.is_empty() {
            return backend.read_columns_to_arrow(col_refs_vec.as_deref(), 0, Some(0));
        }

        let total_rows = backend.row_count() as usize;
        if backend.is_mmap_only() && !Self::should_use_scatter_read(total_rows, row_indices.len()) {
            let full_batch = backend.read_columns_to_arrow(col_refs_vec.as_deref(), 0, None)?;
            return Self::take_rows_from_full_batch(&full_batch, row_indices);
        }

        backend.read_columns_by_indices_to_arrow(row_indices, col_refs_vec.as_deref())
    }

    fn read_matching_rows_by_indices(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
        row_indices: &[usize],
    ) -> io::Result<RecordBatch> {
        let col_refs = Self::get_col_refs(stmt);
        let col_refs_vec: Option<Vec<&str>> = col_refs
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        if row_indices.is_empty() {
            return backend.read_columns_to_arrow(col_refs_vec.as_deref(), 0, Some(0));
        }

        backend.read_columns_by_indices_to_arrow(row_indices, col_refs_vec.as_deref())
    }

    /// MMAP fast path for IN filter on string column.
    /// Strategy: scan each IN value independently, merge-union sorted indices, scatter read.
    fn try_fast_mmap_in_filter(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
        storage_path: &Path,
    ) -> io::Result<Option<ApexResult>> {
        if !backend.is_mmap_only() || backend.has_pending_deltas() || backend.has_delta() {
            return Ok(None);
        }

        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };
        let (col, values) = match Self::extract_in_string_filter(where_clause) {
            Some(v) => v,
            None => return Ok(None),
        };

        let mut all_indices = match backend.scan_string_in_mmap(&col, &values, None)? {
            Some(v) => v,
            None => return Ok(None),
        };

        // Apply offset + limit
        let offset = stmt.offset.unwrap_or(0);
        if offset > 0 {
            if offset >= all_indices.len() {
                return Ok(Some(ApexResult::Empty(Arc::new(Schema::empty()))));
            }
            all_indices = all_indices[offset..].to_vec();
        }
        if let Some(lim) = stmt.limit {
            all_indices.truncate(lim);
        }

        if all_indices.is_empty() {
            return Ok(Some(ApexResult::Empty(Arc::new(Schema::empty()))));
        }

        let batch = Self::read_matching_rows_by_indices(backend, stmt, &all_indices)?;
        if !stmt.is_pure_star() {
            let projected =
                Self::apply_projection_with_storage(&batch, &stmt.columns, Some(storage_path))?;
            return Ok(Some(ApexResult::Data(projected)));
        }
        Ok(Some(ApexResult::Data(batch)))
    }

    /// MMAP fast path for numeric IN and OR-of-equalities on the same column.
    fn try_fast_mmap_numeric_in_filter(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
        storage_path: &Path,
    ) -> io::Result<Option<ApexResult>> {
        if !backend.is_mmap_only() || backend.has_pending_deltas() || backend.has_delta() {
            return Ok(None);
        }

        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };

        let (col, nums) = match Self::extract_in_numeric_filter(where_clause)
            .or_else(|| Self::extract_or_numeric_equalities(where_clause))
        {
            Some(v) => v,
            None => return Ok(None),
        };

        let limit_with_off = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
        let mut all_indices = match backend.scan_numeric_in_mmap(&col, &nums, limit_with_off)? {
            Some(v) => v,
            None => return Ok(None),
        };

        let offset = stmt.offset.unwrap_or(0);
        if offset > 0 {
            if offset >= all_indices.len() {
                return Ok(Some(ApexResult::Empty(Arc::new(Schema::empty()))));
            }
            all_indices = all_indices[offset..].to_vec();
        }
        if let Some(lim) = stmt.limit {
            all_indices.truncate(lim);
        }

        if all_indices.is_empty() {
            return Ok(Some(ApexResult::Empty(Arc::new(Schema::empty()))));
        }

        let batch = Self::read_matching_rows_adaptive(backend, stmt, &all_indices)?;
        if !stmt.is_pure_star() {
            let projected =
                Self::apply_projection_with_storage(&batch, &stmt.columns, Some(storage_path))?;
            return Ok(Some(ApexResult::Data(projected)));
        }
        Ok(Some(ApexResult::Data(batch)))
    }

    /// Helper to extract boolean equality: col = true/false
    fn extract_bool_equality(expr: &SqlExpr) -> Option<(String, bool)> {
        use crate::query::sql_parser::BinaryOperator;
        match expr {
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::Eq,
                right,
            } => match (left.as_ref(), right.as_ref()) {
                (SqlExpr::Column(col), SqlExpr::Literal(Value::Bool(val)))
                | (SqlExpr::Literal(Value::Bool(val)), SqlExpr::Column(col)) => {
                    Some((col.trim_matches('"').to_string(), *val))
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Helper to extract numeric comparison: col > N, col >= N, col < N, col <= N
    fn extract_numeric_comparison(expr: &SqlExpr) -> Option<(String, String, f64)> {
        use crate::query::sql_parser::BinaryOperator;
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                let op_str = match op {
                    BinaryOperator::Gt => ">",
                    BinaryOperator::Ge => ">=",
                    BinaryOperator::Lt => "<",
                    BinaryOperator::Le => "<=",
                    BinaryOperator::Eq => "=",
                    _ => return None,
                };

                match (left.as_ref(), right.as_ref()) {
                    (SqlExpr::Column(col), lit) => {
                        if let Ok(val) = Self::extract_numeric_value(lit) {
                            Some((col.trim_matches('"').to_string(), op_str.to_string(), val))
                        } else {
                            None
                        }
                    }
                    (lit, SqlExpr::Column(col)) => {
                        if let Ok(val) = Self::extract_numeric_value(lit) {
                            // Flip operator for reversed order
                            let flipped = match op_str {
                                ">" => "<",
                                ">=" => "<=",
                                "<" => ">",
                                "<=" => ">=",
                                _ => op_str,
                            };
                            Some((col.trim_matches('"').to_string(), flipped.to_string(), val))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Systematic predicate pushdown: extract simple numeric comparison from WHERE
    /// and use storage-level filtered read instead of full table scan.
    /// Handles: col > N, col >= N, col < N, col <= N, col = N, col != N
    /// Returns Some(batch) if pushdown succeeded, None to fall through.
    fn try_numeric_predicate_pushdown(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        if backend.has_pending_deltas() || backend.has_delta() {
            return Ok(None);
        }
        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };

        // --- mmap_only path: scan → indices → scatter read (LIMIT only) ---
        if backend.is_mmap_only() {
            if stmt.limit.is_none() {
                return Ok(None);
            }
            let (col, lo, hi) = match Self::extract_any_numeric_range(where_clause) {
                Some(v) => v,
                None => return Ok(None),
            };
            let limit_with_off = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
            let indices = match backend.scan_numeric_range_mmap(&col, lo, hi, limit_with_off)? {
                Some(v) => v,
                None => return Ok(None),
            };
            if indices.is_empty() {
                let schema = backend.read_columns_to_arrow(None, 0, Some(0))?;
                return Ok(Some(schema));
            }
            let batch = Self::read_matching_rows_adaptive(backend, stmt, &indices)?;
            return Ok(Some(batch));
        }

        // --- in-memory path: storage-level filtered read ---
        let (col_name, op_str, value) = match Self::extract_numeric_comparison(where_clause) {
            Some(v) => v,
            None => return Ok(None),
        };
        // Column projection pushdown
        let col_refs = Self::get_col_refs(stmt);
        let col_refs_vec: Option<Vec<&str>> = col_refs
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let batch = backend.read_columns_filtered_to_arrow(
            col_refs_vec.as_deref(),
            &col_name,
            &op_str,
            value,
        )?;
        Ok(Some(batch))
    }

    /// Execute SELECT * with late materialization optimization
    /// 1. Read only WHERE columns first
    /// 2. Apply filter to get matching row indices
    /// 3. Read remaining columns only for matching rows
    fn execute_with_late_materialization(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
        storage_path: &Path,
    ) -> io::Result<RecordBatch> {
        use arrow::compute;

        let where_clause = stmt.where_clause.as_ref().unwrap();
        let need_count = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));

        // FAST PATH: no LIMIT.
        // Try mmap numeric IN / OR-of-equalities before full sequential read.
        if need_count.is_none() {
            if !backend.has_pending_deltas()
                && !backend.has_delta()
                && backend.is_mmap_only()
            {
                if let Some((col, nums)) = Self::extract_in_numeric_filter(where_clause)
                    .or_else(|| Self::extract_or_numeric_equalities(where_clause))
                {
                    if let Some(indices) = backend.scan_numeric_in_mmap(&col, &nums, None)? {
                        if indices.is_empty() {
                            let schema = backend.read_columns_to_arrow(None, 0, Some(0))?;
                            return Ok(schema);
                        }
                        return Self::read_matching_rows_adaptive(backend, stmt, &indices);
                    }
                }
                if let Some(leaves) = Self::extract_or_leaf_predicates(where_clause) {
                    if let Some(indices) = Self::scan_or_leaves_mmap(backend, &leaves, None)? {
                        if indices.is_empty() {
                            let schema = backend.read_columns_to_arrow(None, 0, Some(0))?;
                            return Ok(schema);
                        }
                        return Self::read_matching_rows_adaptive(backend, stmt, &indices);
                    }
                }
            }

            // Fallback: full sequential read + vectorized Arrow filter
            // Need both SELECT columns and WHERE columns (WHERE is applied on this batch)
            // For SELECT *, required_columns() returns None → read all columns
            let col_refs_vec: Option<Vec<String>> = stmt.required_columns().map(|mut sel_cols| {
                for wc in stmt.where_columns() {
                    if !sel_cols.iter().any(|c| c.eq_ignore_ascii_case(&wc)) {
                        sel_cols.push(wc);
                    }
                }
                sel_cols
            });
            let col_refs_strs: Option<Vec<&str>> = col_refs_vec
                .as_ref()
                .map(|v| v.iter().map(|s| s.as_str()).collect());
            let full_batch = backend.read_columns_to_arrow(col_refs_strs.as_deref(), 0, None)?;
            if full_batch.num_rows() > 0 {
                let mask =
                    Self::evaluate_predicate_with_storage(&full_batch, where_clause, storage_path)?;
                return compute::filter_record_batch(&full_batch, &mask)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
            }
            return Ok(full_batch);
        }

        // Step 1: Read only columns needed for WHERE clause
        let where_cols = stmt.where_columns();
        let where_col_refs: Vec<&str> = where_cols.iter().map(|s| s.as_str()).collect();

        // Also include _id for later row identification
        let mut cols_to_read: Vec<&str> = vec!["_id"];
        cols_to_read.extend(where_col_refs.iter());

        // OPTIMIZATION: Streaming filter evaluation with early termination
        // Read data in chunks and stop once we have enough matches
        let total_rows = backend.row_count() as usize;
        // Adaptive chunk size: smaller for small LIMIT (assume ~50% selectivity)
        let chunk_size: usize = if let Some(need) = need_count {
            // Start with 4x the needed rows, grow if selectivity is low
            (need * 4).max(1000).min(100_000)
        } else {
            50_000
        };

        let limited_indices: Vec<usize> = if let Some(need) = need_count {
            let mut indices = Vec::with_capacity(need);
            let mut start_row: usize = 0;

            while start_row < total_rows && indices.len() < need {
                let rows_to_read = chunk_size.min(total_rows - start_row);
                let filter_batch = backend.read_columns_to_arrow(
                    Some(&cols_to_read),
                    start_row,
                    Some(rows_to_read),
                )?;

                if filter_batch.num_rows() == 0 {
                    break;
                }

                let mask = Self::evaluate_predicate_with_storage(
                    &filter_batch,
                    where_clause,
                    storage_path,
                )?;

                #[cfg(test)]
                {
                    let true_count = mask.iter().filter(|v| *v == Some(true)).count();
                    eprintln!("DEBUG late_mat chunk: mask true_count={}", true_count);
                }

                // Collect matching indices from this chunk
                for (i, v) in mask.iter().enumerate() {
                    if v == Some(true) {
                        indices.push(start_row + i);
                        if indices.len() >= need {
                            break;
                        }
                    }
                }

                start_row += rows_to_read;
            }

            // Apply offset
            if let Some(offset) = stmt.offset {
                indices.into_iter().skip(offset).collect()
            } else {
                indices
            }
        } else {
            // No LIMIT - use streaming chunks to avoid loading all data at once
            let mut all_indices = Vec::new();
            let mut start_row: usize = 0;

            while start_row < total_rows {
                let rows_to_read = chunk_size.min(total_rows - start_row);
                let filter_batch = backend.read_columns_to_arrow(
                    Some(&cols_to_read),
                    start_row,
                    Some(rows_to_read),
                )?;

                if filter_batch.num_rows() == 0 {
                    break;
                }

                let mask = Self::evaluate_predicate_with_storage(
                    &filter_batch,
                    where_clause,
                    storage_path,
                )?;

                // Collect matching indices from this chunk
                for (i, v) in mask.iter().enumerate() {
                    if v == Some(true) {
                        all_indices.push(start_row + i);
                    }
                }

                start_row += rows_to_read;
            }

            all_indices
        };

        if limited_indices.is_empty() {
            return backend.read_columns_to_arrow(None, 0, Some(0));
        }

        // Step 4: Read ALL columns but only for matching row indices.
        // NOTE: limited_indices are positions in the ACTIVE row sequence (deleted rows excluded).
        // For V4 mmap-only backends, read_columns_by_indices_to_arrow delegates to
        // extract_rows_by_indices_to_arrow which uses PHYSICAL row positions — causing
        // wrong results when deletions shift active vs physical positions.
        // For mmap-only: use full active read + Arrow take (active indices match active batch).
        // For in-memory (data loaded): read_columns_by_indices_to_arrow falls back to the
        // same full-read + take path, so physical==active there too.
        if backend.is_mmap_only() {
            use arrow::array::ArrayRef;
            let col_refs = Self::get_col_refs(stmt);
            let col_refs_vec: Option<Vec<&str>> = col_refs
                .as_ref()
                .map(|v| v.iter().map(|s| s.as_str()).collect());
            let full_batch = backend.read_columns_to_arrow(col_refs_vec.as_deref(), 0, None)?;
            let indices_arr = arrow::array::UInt32Array::from(
                limited_indices
                    .iter()
                    .map(|&i| i as u32)
                    .collect::<Vec<_>>(),
            );
            let taken_cols: Vec<ArrayRef> = full_batch
                .columns()
                .iter()
                .map(|col| {
                    arrow::compute::take(col.as_ref(), &indices_arr, None)
                        .map_err(|e| err_data(e.to_string()))
                })
                .collect::<io::Result<Vec<_>>>()?;
            arrow::record_batch::RecordBatch::try_new(full_batch.schema(), taken_cols)
                .map_err(|e| err_data(e.to_string()))
        } else {
            let col_refs = Self::get_col_refs(stmt);
            let col_refs_vec: Option<Vec<&str>> = col_refs
                .as_ref()
                .map(|v| v.iter().map(|s| s.as_str()).collect());
            backend.read_columns_by_indices_to_arrow(&limited_indices, col_refs_vec.as_deref())
        }
    }

    /// Execute SELECT * with ORDER BY + LIMIT late materialization
    /// 1. Read only ORDER BY columns in chunks
    /// 2. Use streaming top-k to find best rows without loading all data
    /// 3. Read all other columns only for those k rows
    fn execute_with_order_late_materialization(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<RecordBatch> {
        let limit = stmt.limit.unwrap_or(0);
        let offset = stmt.offset.unwrap_or(0);
        let k = limit + offset;
        if k == 0 {
            return backend.read_columns_to_arrow(None, 0, Some(0));
        }

        if backend.has_delta() {
            let full_batch = backend.read_columns_to_arrow(None, 0, None)?;
            let sorted = Self::apply_order_by_topk(&full_batch, &stmt.order_by, Some(k))?;
            return Self::apply_limit_offset(&sorted, stmt.limit, stmt.offset);
        }

        // IN-MEMORY FAST PATH: 2-col ORDER BY (string, float64) — skip Arrow string conversion.
        // Uses global dict cache (u16 group_ids) + raw float64 column for typed sort keys.
        if stmt.order_by.len() == 2 && !backend.has_pending_deltas() {
            let o0 = &stmt.order_by[0];
            let o1 = &stmt.order_by[1];
            let c0 = {
                let c = o0.column.trim_matches('"');
                if let Some(p) = c.rfind('.') {
                    &c[p + 1..]
                } else {
                    c
                }
            };
            let c1 = {
                let c = o1.column.trim_matches('"');
                if let Some(p) = c.rfind('.') {
                    &c[p + 1..]
                } else {
                    c
                }
            };
            if let Some(indices) =
                backend.order_topk_str_float64(c0, !o0.descending, c1, !o1.descending, k, offset)?
            {
                if !indices.is_empty() {
                    return backend.read_columns_by_indices_to_arrow(&indices, None);
                }
                return backend.read_columns_to_arrow(None, 0, Some(0));
            }
        }

        // MMAP FAST PATH: single ORDER BY column + mmap-only → direct top-K scan without Arrow
        if backend.is_mmap_only() && stmt.order_by.len() == 1 && !backend.has_pending_deltas() {
            let clause = &stmt.order_by[0];
            let col_name = clause.column.trim_matches('"');
            let actual_col = if let Some(p) = col_name.rfind('.') {
                &col_name[p + 1..]
            } else {
                col_name
            };
            if let Some(heap) = backend.scan_top_k_indices_mmap(actual_col, k, clause.descending)? {
                let final_indices: Vec<usize> =
                    heap.into_iter().skip(offset).map(|(idx, _)| idx).collect();
                if !final_indices.is_empty() {
                    return backend.read_columns_by_indices_to_arrow(&final_indices, None);
                }
                return backend.read_columns_to_arrow(None, 0, Some(0));
            }
        }

        // Step 1: Read only columns needed for ORDER BY
        let order_cols: Vec<&str> = stmt
            .order_by
            .iter()
            .map(|o| {
                let col = o.column.trim_matches('"');
                if let Some(dot_pos) = col.rfind('.') {
                    &col[dot_pos + 1..]
                } else {
                    col
                }
            })
            .collect();

        let sort_batch = backend.read_columns_to_arrow(Some(&order_cols), 0, None)?;
        let num_rows = sort_batch.num_rows();

        if num_rows == 0 {
            return backend.read_columns_to_arrow(None, 0, Some(0));
        }

        let k_actual = k.min(num_rows);

        // Step 2: Find top-k indices using optimized streaming algorithm
        let final_indices: Vec<usize> = if stmt.order_by.len() == 1 && k_actual <= 100 {
            let clause = &stmt.order_by[0];
            let col_name = clause.column.trim_matches('"');
            let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                &col_name[dot_pos + 1..]
            } else {
                col_name
            };

            if let Some(col) = sort_batch.column_by_name(actual_col) {
                // Fast path for Float64 DESC (most common case)
                if let Some(float_arr) = col.as_any().downcast_ref::<Float64Array>() {
                    let descending = clause.descending;

                    // Streaming top-k: maintain sorted list of top k (value, index) pairs
                    let mut top_k: Vec<(f64, usize)> = Vec::with_capacity(k_actual + 1);

                    if descending {
                        // DESC: keep k largest values
                        for i in 0..num_rows {
                            let val = if float_arr.is_null(i) {
                                f64::NEG_INFINITY
                            } else {
                                float_arr.value(i)
                            };

                            if top_k.len() < k_actual {
                                let pos = top_k.partition_point(|(v, _)| *v > val);
                                top_k.insert(pos, (val, i));
                            } else if val > top_k[k_actual - 1].0 {
                                let pos = top_k.partition_point(|(v, _)| *v > val);
                                top_k.insert(pos, (val, i));
                                top_k.pop();
                            }
                        }
                    } else {
                        // ASC: keep k smallest values
                        for i in 0..num_rows {
                            let val = if float_arr.is_null(i) {
                                f64::INFINITY
                            } else {
                                float_arr.value(i)
                            };

                            if top_k.len() < k_actual {
                                let pos = top_k.partition_point(|(v, _)| *v < val);
                                top_k.insert(pos, (val, i));
                            } else if val < top_k[k_actual - 1].0 {
                                let pos = top_k.partition_point(|(v, _)| *v < val);
                                top_k.insert(pos, (val, i));
                                top_k.pop();
                            }
                        }
                    }

                    top_k.into_iter().skip(offset).map(|(_, idx)| idx).collect()
                } else if let Some(int_arr) = col.as_any().downcast_ref::<Int64Array>() {
                    let descending = clause.descending;
                    let mut top_k: Vec<(i64, usize)> = Vec::with_capacity(k_actual + 1);

                    if descending {
                        for i in 0..num_rows {
                            let val = if int_arr.is_null(i) {
                                i64::MIN
                            } else {
                                int_arr.value(i)
                            };

                            if top_k.len() < k_actual {
                                let pos = top_k.partition_point(|(v, _)| *v > val);
                                top_k.insert(pos, (val, i));
                            } else if val > top_k[k_actual - 1].0 {
                                let pos = top_k.partition_point(|(v, _)| *v > val);
                                top_k.insert(pos, (val, i));
                                top_k.pop();
                            }
                        }
                    } else {
                        for i in 0..num_rows {
                            let val = if int_arr.is_null(i) {
                                i64::MAX
                            } else {
                                int_arr.value(i)
                            };

                            if top_k.len() < k_actual {
                                let pos = top_k.partition_point(|(v, _)| *v < val);
                                top_k.insert(pos, (val, i));
                            } else if val < top_k[k_actual - 1].0 {
                                let pos = top_k.partition_point(|(v, _)| *v < val);
                                top_k.insert(pos, (val, i));
                                top_k.pop();
                            }
                        }
                    }

                    top_k.into_iter().skip(offset).map(|(_, idx)| idx).collect()
                } else {
                    Self::compute_topk_indices_generic(
                        &sort_batch,
                        &stmt.order_by,
                        k_actual,
                        stmt.offset,
                    )
                }
            } else {
                Self::compute_topk_indices_generic(
                    &sort_batch,
                    &stmt.order_by,
                    k_actual,
                    stmt.offset,
                )
            }
        } else {
            Self::compute_topk_indices_generic(&sort_batch, &stmt.order_by, k_actual, stmt.offset)
        };

        if final_indices.is_empty() {
            return backend.read_columns_to_arrow(None, 0, Some(0));
        }

        // Step 3: Read ALL columns but only for top-k row indices
        backend.read_columns_by_indices_to_arrow(&final_indices, None)
    }

    /// Pre-evaluate SELECT expression aliases that are referenced by ORDER BY clauses.
    /// e.g. `SELECT array_distance(vec,[...]) AS dist … ORDER BY dist`
    /// The `dist` column doesn't exist in `batch` yet, but we can evaluate the expression
    /// and add it temporarily so the sort has something to work with.
    /// The extra columns are stripped by the subsequent projection step.
    fn augment_batch_for_order_by(
        batch: &RecordBatch,
        select_cols: &[crate::query::SelectColumn],
        order_by: &[crate::query::OrderByClause],
    ) -> io::Result<RecordBatch> {
        use crate::query::SelectColumn;
        use arrow::datatypes::Field;

        // Build alias→expr map from SELECT columns
        let mut alias_map: std::collections::HashMap<String, &crate::query::SqlExpr> =
            std::collections::HashMap::new();
        for sc in select_cols {
            if let SelectColumn::Expression {
                expr,
                alias: Some(alias),
            } = sc
            {
                alias_map.insert(alias.to_lowercase(), expr);
            }
        }

        if alias_map.is_empty() {
            return Ok(batch.clone());
        }

        // Find ORDER BY columns that are aliases not yet in the batch
        let mut extra: Vec<(String, arrow::array::ArrayRef)> = Vec::new();
        for ob in order_by {
            if ob.expr.is_some() {
                continue; // already handled by apply_order_by_topk expression path
            }
            let cn = ob.column.trim_matches('"');
            let cn = cn.rfind('.').map_or(cn, |p| &cn[p + 1..]);
            if batch.column_by_name(cn).is_none() {
                // Try to resolve as alias
                if let Some(expr) = alias_map.get(&cn.to_lowercase()) {
                    let arr = Self::evaluate_expr_to_array(batch, expr)?;
                    extra.push((cn.to_string(), arr));
                }
            }
        }

        if extra.is_empty() {
            return Ok(batch.clone());
        }

        let mut fields: Vec<Field> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| (**f).clone())
            .collect();
        let mut cols = batch.columns().to_vec();
        for (name, arr) in extra {
            fields.push(Field::new(&name, arr.data_type().clone(), true));
            cols.push(arr);
        }
        RecordBatch::try_new(Arc::new(arrow::datatypes::Schema::new(fields)), cols)
            .map_err(|e| err_data(e.to_string()))
    }

    /// Generic top-k computation using partial sort (fallback for complex cases)
    fn compute_topk_indices_generic(
        sort_batch: &RecordBatch,
        order_by: &[crate::query::OrderByClause],
        k: usize,
        offset: Option<usize>,
    ) -> Vec<usize> {
        let num_rows = sort_batch.num_rows();

        // FAST PATH: 2-column (StringArray, Float64Array) — typed comparison, no closure overhead
        if order_by.len() == 2 {
            let col0_name = {
                let c = order_by[0].column.trim_matches('"');
                if let Some(p) = c.rfind('.') {
                    &c[p + 1..]
                } else {
                    c
                }
            };
            let col1_name = {
                let c = order_by[1].column.trim_matches('"');
                if let Some(p) = c.rfind('.') {
                    &c[p + 1..]
                } else {
                    c
                }
            };
            let arr0 = sort_batch.column_by_name(col0_name);
            let arr1 = sort_batch.column_by_name(col1_name);
            if let (Some(a0), Some(a1)) = (arr0, arr1) {
                use arrow::array::Float64Array as FA;
                use arrow::array::StringArray as SA;
                if let (Some(str_arr), Some(flt_arr)) = (
                    a0.as_any().downcast_ref::<SA>(),
                    a1.as_any().downcast_ref::<FA>(),
                ) {
                    use ahash::AHashMap;
                    // Build dict for string col → u16 id (1-based, 0=null)
                    let mut dict: AHashMap<&str, u16> = AHashMap::with_capacity(64);
                    let mut dict_vals: Vec<&str> = Vec::with_capacity(64);
                    let str_ids: Vec<u16> = (0..num_rows)
                        .map(|i| {
                            if str_arr.is_null(i) {
                                return 0u16;
                            }
                            let s = str_arr.value(i);
                            let next = dict_vals.len() as u16 + 1;
                            *dict.entry(s).or_insert_with(|| {
                                dict_vals.push(s);
                                next
                            })
                        })
                        .collect();
                    // Sort dict entries to get alphabetical rank mapping
                    let mut sorted: Vec<(u16, &str)> = dict_vals
                        .iter()
                        .enumerate()
                        .map(|(i, &s)| (i as u16 + 1, s))
                        .collect();
                    sorted.sort_unstable_by_key(|&(_, s)| s);
                    let mut rank_of = vec![0u16; dict_vals.len() + 1];
                    for (rank, &(id, _)) in sorted.iter().enumerate() {
                        rank_of[id as usize] = rank as u16;
                    }
                    let asc0 = !order_by[0].descending;
                    let desc1 = order_by[1].descending;
                    // Pack (str_rank, score_sortable_bits) into (u64, u64) composite key
                    let mut packed: Vec<(u64, u64, usize)> = (0..num_rows)
                        .map(|i| {
                            let sid = str_ids[i] as usize;
                            let sr = if sid == 0 {
                                u16::MAX as u64
                            } else {
                                rank_of[sid] as u64
                            };
                            let sk0 = if asc0 { sr } else { u16::MAX as u64 - sr };
                            let f = if flt_arr.is_null(i) {
                                f64::NEG_INFINITY
                            } else {
                                flt_arr.value(i)
                            };
                            let fb = f.to_bits();
                            let fs = if fb >> 63 == 0 {
                                fb ^ (1u64 << 63)
                            } else {
                                !fb
                            };
                            let sk1 = if desc1 { !fs } else { fs };
                            (sk0, sk1, i)
                        })
                        .collect();
                    if k < num_rows {
                        packed.select_nth_unstable_by_key(k - 1, |&(a, b, _)| (a, b));
                        packed.truncate(k);
                    }
                    packed.sort_unstable_by_key(|&(a, b, _)| (a, b));
                    let off = offset.unwrap_or(0);
                    return packed
                        .into_iter()
                        .skip(off)
                        .map(|(_, _, idx)| idx)
                        .collect();
                }
            }
        }

        let sort_cols: Vec<(ArrayRef, bool)> = order_by
            .iter()
            .filter_map(|clause| {
                let col_name = clause.column.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                sort_batch
                    .column_by_name(actual_col)
                    .map(|col| (col.clone(), clause.descending))
            })
            .collect();

        let compare_rows = |a: usize, b: usize| -> std::cmp::Ordering {
            for (col, descending) in &sort_cols {
                let ord = Self::compare_array_values(col, a, b);
                if ord != std::cmp::Ordering::Equal {
                    return if *descending { ord.reverse() } else { ord };
                }
            }
            std::cmp::Ordering::Equal
        };

        let mut indices: Vec<usize> = (0..num_rows).collect();

        if k < num_rows {
            indices.select_nth_unstable_by(k - 1, |&a, &b| compare_rows(a, b));
            indices.truncate(k);
        }
        indices.sort_by(|&a, &b| compare_rows(a, b));

        if let Some(off) = offset {
            indices.into_iter().skip(off).collect()
        } else {
            indices
        }
    }

    /// Fast path for combined WHERE filter + GROUP BY on dictionary columns
    /// Does filter and aggregation in a single pass without intermediate materialization
    fn try_fast_filter_groupby(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
    ) -> io::Result<Option<RecordBatch>> {
        if backend.has_pending_deltas() || backend.is_mmap_only() {
            return Ok(None);
        }

        use crate::query::AggregateFunc;
        use arrow::array::DictionaryArray;
        use arrow::datatypes::UInt32Type;

        // Only handle simple patterns: WHERE col = 'value' with single-column GROUP BY
        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };

        if stmt.group_by.len() != 1 {
            return Ok(None);
        }

        // Extract filter column and value
        let (filter_col, filter_value) = match where_clause {
            SqlExpr::BinaryOp { left, op, right } => {
                use crate::query::sql_parser::BinaryOperator;
                if *op != BinaryOperator::Eq {
                    return Ok(None);
                }
                match (left.as_ref(), right.as_ref()) {
                    (SqlExpr::Column(col), SqlExpr::Literal(Value::String(val))) => {
                        (col.trim_matches('"').to_string(), val.clone())
                    }
                    (SqlExpr::Literal(Value::String(val)), SqlExpr::Column(col)) => {
                        (col.trim_matches('"').to_string(), val.clone())
                    }
                    _ => return Ok(None),
                }
            }
            _ => return Ok(None),
        };

        let group_col = stmt.group_by[0].trim_matches('"').to_string();

        // Find aggregate column
        let mut agg_col_name: Option<String> = None;
        let mut agg_func: Option<AggregateFunc> = None;
        let mut agg_alias: Option<String> = None;

        for col in &stmt.columns {
            if let SelectColumn::Aggregate {
                func,
                column,
                alias,
                ..
            } = col
            {
                if let Some(col_name) = column {
                    let actual = col_name.trim_matches('"');
                    if actual != "*" {
                        agg_col_name = Some(actual.to_string());
                        agg_func = Some(func.clone());
                        agg_alias = alias.clone();
                    }
                }
                break;
            }
        }

        let agg_col_name = match agg_col_name {
            Some(c) => c,
            None => return Ok(None),
        };

        // Read only needed columns
        let cols_to_read: Vec<&str> = vec![
            filter_col.as_str(),
            group_col.as_str(),
            agg_col_name.as_str(),
        ];
        let batch = backend.read_columns_to_arrow(Some(&cols_to_read), 0, None)?;

        if batch.num_rows() == 0 {
            return Ok(None);
        }

        let num_rows = batch.num_rows();

        // Get filter column as dictionary
        let filter_arr = match batch.column_by_name(&filter_col) {
            Some(c) => c,
            None => return Ok(None),
        };

        let filter_dict = match filter_arr
            .as_any()
            .downcast_ref::<DictionaryArray<UInt32Type>>()
        {
            Some(d) => d,
            None => return Ok(None),
        };

        // Find filter key
        let filter_keys = filter_dict.keys();
        let filter_values = filter_dict.values();
        let filter_str_values = match filter_values.as_any().downcast_ref::<StringArray>() {
            Some(s) => s,
            None => return Ok(None),
        };

        let mut target_filter_key: Option<u32> = None;
        for i in 0..filter_str_values.len() {
            if filter_str_values.value(i) == filter_value {
                target_filter_key = Some(i as u32);
                break;
            }
        }

        let target_filter_key = match target_filter_key {
            Some(k) => k,
            None => {
                // Value not in dictionary - return empty result
                let schema = Arc::new(Schema::new(vec![Field::new(
                    &group_col,
                    ArrowDataType::Utf8,
                    false,
                )]));
                return Ok(Some(RecordBatch::new_empty(schema)));
            }
        };

        // Get group column as dictionary
        let group_arr = match batch.column_by_name(&group_col) {
            Some(c) => c,
            None => return Ok(None),
        };

        let group_dict = match group_arr
            .as_any()
            .downcast_ref::<DictionaryArray<UInt32Type>>()
        {
            Some(d) => d,
            None => return Ok(None),
        };

        let group_keys = group_dict.keys();
        let group_values = group_dict.values();
        let group_str_values = match group_values.as_any().downcast_ref::<StringArray>() {
            Some(s) => s,
            None => return Ok(None),
        };
        let group_dict_size = group_str_values.len() + 1;

        // Get aggregate column
        let agg_arr = match batch.column_by_name(&agg_col_name) {
            Some(c) => c,
            None => return Ok(None),
        };

        let agg_float = agg_arr.as_any().downcast_ref::<Float64Array>();
        let agg_int = agg_arr.as_any().downcast_ref::<Int64Array>();

        // Single-pass: filter + aggregate
        let mut counts: Vec<i64> = vec![0; group_dict_size];
        let mut sums: Vec<f64> = vec![0.0; group_dict_size];

        let filter_key_values = filter_keys.values();
        let group_key_values = group_keys.values();

        if let Some(float_arr) = agg_float {
            if filter_keys.null_count() == 0
                && group_keys.null_count() == 0
                && float_arr.null_count() == 0
            {
                let float_values = float_arr.values();
                for i in 0..num_rows {
                    if unsafe { *filter_key_values.get_unchecked(i) } == target_filter_key {
                        let gk = unsafe { *group_key_values.get_unchecked(i) as usize + 1 };
                        unsafe {
                            *counts.get_unchecked_mut(gk) += 1;
                            *sums.get_unchecked_mut(gk) += *float_values.get_unchecked(i);
                        }
                    }
                }
            } else {
                for i in 0..num_rows {
                    if !filter_keys.is_null(i) && filter_keys.value(i) == target_filter_key {
                        let gk = if group_keys.is_null(i) {
                            0
                        } else {
                            group_keys.value(i) as usize + 1
                        };
                        counts[gk] += 1;
                        if !float_arr.is_null(i) {
                            sums[gk] += float_arr.value(i);
                        }
                    }
                }
            }
        } else if let Some(int_arr) = agg_int {
            for i in 0..num_rows {
                if !filter_keys.is_null(i) && filter_keys.value(i) == target_filter_key {
                    let gk = if group_keys.is_null(i) {
                        0
                    } else {
                        group_keys.value(i) as usize + 1
                    };
                    counts[gk] += 1;
                    if !int_arr.is_null(i) {
                        sums[gk] += int_arr.value(i) as f64;
                    }
                }
            }
        } else {
            return Ok(None);
        }

        // Collect results - pre-allocate with estimated group count
        let estimated_groups = (group_dict_size / 4).max(16);
        let mut result_groups: Vec<&str> = Vec::with_capacity(estimated_groups);
        let mut result_values: Vec<f64> = Vec::with_capacity(estimated_groups);

        for gk in 1..group_dict_size {
            if counts[gk] > 0 {
                result_groups.push(group_str_values.value(gk - 1));
                let value = match agg_func {
                    Some(AggregateFunc::Sum) => sums[gk],
                    Some(AggregateFunc::Avg) => sums[gk] / counts[gk] as f64,
                    Some(AggregateFunc::Count) => counts[gk] as f64,
                    _ => sums[gk],
                };
                result_values.push(value);
            }
        }

        // Build result batch
        let agg_field_name = agg_alias.unwrap_or_else(|| {
            let func_name = match agg_func {
                Some(AggregateFunc::Sum) => "SUM",
                Some(AggregateFunc::Avg) => "AVG",
                Some(AggregateFunc::Count) => "COUNT",
                _ => "AGG",
            };
            format!("{}({})", func_name, agg_col_name)
        });

        let schema = Arc::new(Schema::new(vec![
            Field::new(&group_col, ArrowDataType::Utf8, false),
            Field::new(&agg_field_name, ArrowDataType::Float64, true),
        ]));

        let mut result_batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(result_groups)),
                Arc::new(Float64Array::from(result_values)),
            ],
        )
        .map_err(|e| err_data(e.to_string()))?;

        // Apply ORDER BY if present
        if !stmt.order_by.is_empty() {
            let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
            result_batch = Self::apply_order_by_topk(&result_batch, &stmt.order_by, k)?;
        }

        // Apply LIMIT/OFFSET
        result_batch = Self::apply_limit_offset(&result_batch, stmt.limit, stmt.offset)?;

        Ok(Some(result_batch))
    }

    /// Execute GROUP BY with WHERE using late materialization
    /// 1. Read only WHERE columns first
    /// 2. Filter to get matching row indices
    /// 3. Read GROUP BY + aggregate columns only for matching rows
    fn execute_with_groupby_late_materialization(
        backend: &TableStorageBackend,
        stmt: &SelectStatement,
        storage_path: &Path,
    ) -> io::Result<RecordBatch> {
        use arrow::array::DictionaryArray;
        use arrow::datatypes::UInt32Type;

        // FAST PATH: Try combined filter + GROUP BY on dictionary columns in single pass
        if let Some(result) = Self::try_fast_filter_groupby(backend, stmt)? {
            return Ok(result);
        }

        // Step 1: Read only columns needed for WHERE clause
        let where_cols = stmt.where_columns();
        let where_col_refs: Vec<&str> = where_cols.iter().map(|s| s.as_str()).collect();

        let filter_batch = backend.read_columns_to_arrow(Some(&where_col_refs), 0, None)?;

        if filter_batch.num_rows() == 0 {
            let col_refs = Self::get_col_refs(stmt);
            let col_refs_vec: Option<Vec<&str>> = col_refs
                .as_ref()
                .map(|v| v.iter().map(|s| s.as_str()).collect());
            return backend.read_columns_to_arrow(col_refs_vec.as_deref(), 0, Some(0));
        }

        // Step 2: Apply WHERE filter to get matching row indices
        let where_clause = stmt.where_clause.as_ref().unwrap();
        let mask =
            Self::evaluate_predicate_with_storage(&filter_batch, where_clause, storage_path)?;

        // Collect matching indices
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if v == Some(true) { Some(i) } else { None })
            .collect();

        if indices.is_empty() {
            let col_refs = Self::get_col_refs(stmt);
            let col_refs_vec: Option<Vec<&str>> = col_refs
                .as_ref()
                .map(|v| v.iter().map(|s| s.as_str()).collect());
            return backend.read_columns_to_arrow(col_refs_vec.as_deref(), 0, Some(0));
        }

        // Step 3: Read only required columns (GROUP BY + aggregates) for matching rows
        let required_cols = stmt.required_columns();
        let other_cols: Vec<&str> = if let Some(ref cols) = required_cols {
            cols.iter()
                .filter(|c| !where_cols.contains(c))
                .map(|s| s.as_str())
                .collect()
        } else {
            Vec::new()
        };

        // Read other columns for matching indices only
        if other_cols.is_empty() {
            // All needed columns are in WHERE - just filter the batch
            let indices_array = arrow::array::UInt64Array::from(
                indices.iter().map(|&i| i as u64).collect::<Vec<_>>(),
            );
            let columns: Vec<ArrayRef> = filter_batch
                .columns()
                .iter()
                .map(|col| compute::take(col, &indices_array, None))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| err_data(e.to_string()))?;
            RecordBatch::try_new(filter_batch.schema(), columns)
                .map_err(|e| err_data(e.to_string()))
        } else {
            // Need to read additional columns for matching rows
            let other_batch = backend.read_columns_by_indices_to_arrow(&indices, None)?;

            // Also filter the WHERE columns batch
            let indices_array = arrow::array::UInt64Array::from(
                indices.iter().map(|&i| i as u64).collect::<Vec<_>>(),
            );
            let where_columns: Vec<ArrayRef> = filter_batch
                .columns()
                .iter()
                .map(|col| compute::take(col, &indices_array, None))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| err_data(e.to_string()))?;

            // Merge: use other_batch as base (has _id and other columns)
            // Add WHERE columns that aren't already present
            let mut fields: Vec<Field> = other_batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .collect();
            let mut arrays: Vec<ArrayRef> = other_batch.columns().to_vec();

            for (i, field) in filter_batch.schema().fields().iter().enumerate() {
                if other_batch.column_by_name(field.name()).is_none() {
                    fields.push(field.as_ref().clone());
                    arrays.push(where_columns[i].clone());
                }
            }

            let schema = Arc::new(Schema::new(fields));
            RecordBatch::try_new(schema, arrays).map_err(|e| err_data(e.to_string()))
        }
    }

    // ========== FTS Helper: resolve MATCH()/FUZZY_MATCH() to _id IN (...) ==========

    /// Recursively replace every `FtsMatch { query, fuzzy }` node in an expression with
    /// `In { column: "_id", values: [matching doc ids] }`.  Requires the FtsManager to
    /// be registered for `base_dir` (via `register_fts_manager` or `CREATE FTS INDEX`).
    fn resolve_fts_in_expr(
        expr: SqlExpr,
        base_dir: &Path,
        table_name: &str,
    ) -> io::Result<SqlExpr> {
        match expr {
            SqlExpr::FtsMatch { query, fuzzy } => {
                let mgr = crate::query::executor::get_fts_manager(base_dir)
                    .ok_or_else(|| io::Error::new(
                        io::ErrorKind::Other,
                        format!("FTS not initialised for this database. Run CREATE FTS INDEX ON {} first.", table_name),
                    ))?;
                crate::query::executor::wait_fts_backfill(base_dir, table_name);
                let engine = mgr
                    .get_engine(table_name)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                let ids: Vec<u64> = if fuzzy {
                    engine
                        .fuzzy_search(&query, 1)
                        .map(|r| r.iter().map(|id| id as u64).collect())
                        .unwrap_or_default()
                } else {
                    engine.search_ids(&query).unwrap_or_default()
                };
                if ids.is_empty() {
                    // _id < 0  — guaranteed empty (all valid _id are >= 1)
                    Ok(SqlExpr::BinaryOp {
                        left: Box::new(SqlExpr::Column("_id".to_string())),
                        op: crate::query::sql_parser::BinaryOperator::Lt,
                        right: Box::new(SqlExpr::Literal(crate::data::Value::Int64(0))),
                    })
                } else {
                    let values: Vec<crate::data::Value> = ids
                        .into_iter()
                        .map(|id| crate::data::Value::Int64(id as i64))
                        .collect();
                    Ok(SqlExpr::In {
                        column: "_id".to_string(),
                        values,
                        negated: false,
                    })
                }
            }
            SqlExpr::BinaryOp { left, op, right } => Ok(SqlExpr::BinaryOp {
                left: Box::new(Self::resolve_fts_in_expr(*left, base_dir, table_name)?),
                op,
                right: Box::new(Self::resolve_fts_in_expr(*right, base_dir, table_name)?),
            }),
            SqlExpr::UnaryOp { op, expr } => Ok(SqlExpr::UnaryOp {
                op,
                expr: Box::new(Self::resolve_fts_in_expr(*expr, base_dir, table_name)?),
            }),
            SqlExpr::Paren(inner) => Ok(SqlExpr::Paren(Box::new(Self::resolve_fts_in_expr(
                *inner, base_dir, table_name,
            )?))),
            // All other variants have no nested SqlExpr that could contain FtsMatch
            other => Ok(other),
        }
    }

    /// Detect `SELECT explode_rename(topk_distance(col,[q],k,'m'), "n1","n2") FROM table`.
    /// Returns `(col, query, k, metric, names)` if the pattern matches, else `None`.
    fn detect_topk_explode(
        stmt: &SelectStatement,
    ) -> Option<(&str, &[f64], usize, &str, &[String])> {
        // Must have a real FROM table (not None / subquery / table-function)
        if !matches!(&stmt.from, Some(FromItem::Table { .. })) {
            return None;
        }
        // Exactly one SELECT column that is an Expression wrapping ExplodeRename(TopkDistance)
        if stmt.columns.len() == 1 {
            if let SelectColumn::Expression {
                expr: SqlExpr::ExplodeRename { inner, names },
                ..
            } = &stmt.columns[0]
            {
                if let SqlExpr::TopkDistance {
                    col,
                    query,
                    k,
                    metric,
                } = inner.as_ref()
                {
                    return Some((
                        col.as_str(),
                        query.as_slice(),
                        *k,
                        metric.as_str(),
                        names.as_slice(),
                    ));
                }
            }
        }
        None
    }

    /// Execute `explode_rename(topk_distance(col,[q],k,'m'), "name1","name2")`.
    ///
    /// Returns a RecordBatch with exactly 2 columns:
    /// - `names[0]`: Int64 — the `_id` values of the top-k rows
    /// - `names[1]`: Float64 — the corresponding distances
    ///
    /// The result has k rows, sorted ascending by distance.
    fn execute_topk_explode(
        storage_path: &Path,
        col: &str,
        query: &[f64],
        k: usize,
        metric: &str,
        names: &[String],
    ) -> io::Result<RecordBatch> {
        use crate::query::vector_ops::{topk_heap_direct, DistanceMetric};
        use arrow::array::{BinaryArray, Float64Array, Int64Array};

        if !storage_path.exists() {
            let tbl = storage_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy();
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("topk_distance: table '{}' does not exist", tbl),
            ));
        }

        let metric_enum = DistanceMetric::from_str(metric).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("topk_distance: unknown metric '{}'", metric),
            )
        })?;

        let query_f32: Vec<f32> = query.iter().map(|&x| x as f32).collect();

        let backend = get_cached_backend(storage_path)?;

        // Output schema: names[0]=Int64(_id), names[1]=Float64(dist)
        let id_field = Field::new(&names[0], ArrowDataType::Int64, false);
        let dist_field = Field::new(&names[1], ArrowDataType::Float64, false);
        let out_schema = Arc::new(Schema::new(vec![id_field, dist_field]));

        use crate::query::vector_ops::{
            topk_heap_direct_parallel, topk_heap_direct_parallel_fixed, DistanceComputer,
        };
        let computer = DistanceComputer::new(metric_enum, query_f32);

        // FAST PATH: zero-copy scan directly on OS mmap (no Arrow batch, no memcpy)
        let direct_topk = backend
            .topk_fixedlist_direct(col, &computer, k)
            .ok()
            .flatten()
            .or_else(|| backend.topk_binary_direct(col, &computer, k).ok().flatten());
        if let Some(topk) = direct_topk {
            if topk.is_empty() {
                return RecordBatch::try_new(
                    out_schema,
                    vec![
                        Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
                        Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef,
                    ],
                )
                .map_err(|e| err_data(e.to_string()));
            }
            // Read only the _id column (8MB) instead of all columns (512MB+)
            let id_batch = backend.read_columns_to_arrow(Some(&["_id"]), 0, None)?;
            let id_col = id_batch.column_by_name("_id");
            let ids: Vec<i64> = topk
                .iter()
                .map(|(row_idx, _)| {
                    id_col
                        .and_then(|a| a.as_any().downcast_ref::<arrow::array::Int64Array>())
                        .map(|a| a.value(*row_idx))
                        .unwrap_or(*row_idx as i64)
                })
                .collect();
            let dists: Vec<f64> = topk.iter().map(|(_, d)| *d as f64).collect();
            return RecordBatch::try_new(
                out_schema,
                vec![
                    Arc::new(Int64Array::from(ids)) as ArrayRef,
                    Arc::new(Float64Array::from(dists)) as ArrayRef,
                ],
            )
            .map_err(|e| err_data(e.to_string()));
        }

        // FALLBACK: full Arrow path (Binary columns / compressed RGs)
        let full_batch = backend.read_columns_to_arrow(None, 0, None)?;

        if full_batch.num_rows() == 0 {
            return RecordBatch::try_new(
                out_schema,
                vec![
                    Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
                    Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef,
                ],
            )
            .map_err(|e| err_data(e.to_string()));
        }

        let bin_col = full_batch.column_by_name(col).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("topk_distance: column '{}' not found", col),
            )
        })?;
        let topk = if let Some(fixed_arr) = bin_col
            .as_any()
            .downcast_ref::<arrow::array::FixedSizeListArray>()
        {
            topk_heap_direct_parallel_fixed(fixed_arr, &computer, k)
        } else if let Some(bin_arr) = bin_col.as_any().downcast_ref::<BinaryArray>() {
            topk_heap_direct_parallel(bin_arr, &computer, k)
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("topk_distance: column '{}' is not a vector column", col),
            ));
        };

        let id_col = full_batch.column_by_name("_id");
        let ids: Vec<i64> = topk
            .iter()
            .map(|(row_idx, _)| {
                if let Some(id_arr) = &id_col {
                    if let Some(arr) = id_arr.as_any().downcast_ref::<arrow::array::Int64Array>() {
                        return arr.value(*row_idx);
                    }
                }
                *row_idx as i64
            })
            .collect();
        let dists: Vec<f64> = topk.iter().map(|(_, d)| *d as f64).collect();

        RecordBatch::try_new(
            out_schema,
            vec![
                Arc::new(Int64Array::from(ids)) as ArrayRef,
                Arc::new(Float64Array::from(dists)) as ArrayRef,
            ],
        )
        .map_err(|e| err_data(e.to_string()))
    }

    /// Execute `TOPK_DISTANCE(col, [vec], k, 'metric')` table function.
    ///
    /// Algorithm (O(n log k)):
    /// 1. Read the full RecordBatch from storage.
    /// 2. Locate the binary vector column `col`.
    /// 3. Run `topk_heap_direct` — single-pass fused distance + max-heap.
    /// 4. Gather only the top-k rows via `arrow::compute::take`.
    /// 5. Append a `dist` (Float64) column with the computed distances.
    fn execute_topk_distance(
        storage_path: &Path,
        col: &str,
        query: &[f64],
        k: usize,
        metric: &str,
    ) -> io::Result<RecordBatch> {
        use crate::query::vector_ops::DistanceMetric;
        use arrow::array::{BinaryArray, Float64Array, UInt32Array};
        use arrow::compute;
        use arrow::datatypes::DataType as ArrowDT;

        if !storage_path.exists() {
            let tbl = storage_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy();
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("topk_distance: table '{}' does not exist", tbl),
            ));
        }

        let metric_enum = DistanceMetric::from_str(metric).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("topk_distance: unknown metric '{}'", metric),
            )
        })?;

        let query_f32: Vec<f32> = query.iter().map(|&x| x as f32).collect();

        let backend = get_cached_backend(storage_path)?;
        let full_batch = backend.read_columns_to_arrow(None, 0, None)?;

        // Build the schema with the extra `dist` column
        let mut fields: Vec<Field> = full_batch
            .schema()
            .fields()
            .iter()
            .map(|f| (**f).clone())
            .collect();
        fields.push(Field::new("dist", ArrowDT::Float64, false));
        let out_schema = Arc::new(Schema::new(fields));

        if full_batch.num_rows() == 0 {
            let empty_cols: Vec<ArrayRef> = out_schema
                .fields()
                .iter()
                .map(|f| arrow::array::new_empty_array(f.data_type()))
                .collect();
            return RecordBatch::try_new(out_schema, empty_cols)
                .map_err(|e| err_data(e.to_string()));
        }

        let bin_col = full_batch.column_by_name(col).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("topk_distance: column '{}' not found", col),
            )
        })?;

        // Parallel O(n/T log k) heap — dispatch on Binary vs FixedSizeList
        use crate::query::vector_ops::{
            topk_heap_direct_parallel, topk_heap_direct_parallel_fixed, DistanceComputer,
        };
        let computer = DistanceComputer::new(metric_enum, query_f32);
        let topk = if let Some(fixed_arr) = bin_col
            .as_any()
            .downcast_ref::<arrow::array::FixedSizeListArray>()
        {
            topk_heap_direct_parallel_fixed(fixed_arr, &computer, k)
        } else if let Some(bin_arr) = bin_col.as_any().downcast_ref::<BinaryArray>() {
            topk_heap_direct_parallel(bin_arr, &computer, k)
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("topk_distance: column '{}' is not a vector column", col),
            ));
        };

        if topk.is_empty() {
            let empty_cols: Vec<ArrayRef> = out_schema
                .fields()
                .iter()
                .map(|f| arrow::array::new_empty_array(f.data_type()))
                .collect();
            return RecordBatch::try_new(out_schema, empty_cols)
                .map_err(|e| err_data(e.to_string()));
        }

        // Gather only the top-k rows from the full batch
        let take_indices =
            UInt32Array::from(topk.iter().map(|(i, _)| *i as u32).collect::<Vec<_>>());
        let distances: Vec<f64> = topk.iter().map(|(_, d)| *d as f64).collect();

        let mut new_cols: Vec<ArrayRef> = Vec::with_capacity(full_batch.num_columns() + 1);
        for col_arr in full_batch.columns() {
            let taken = compute::take(col_arr.as_ref(), &take_indices, None)
                .map_err(|e| err_data(e.to_string()))?;
            new_cols.push(taken);
        }
        new_cols.push(Arc::new(Float64Array::from(distances)) as ArrayRef);

        RecordBatch::try_new(out_schema, new_cols).map_err(|e| err_data(e.to_string()))
    }

    /// Return true iff `expr` contains at least one `FtsMatch` node.
    fn expr_has_fts_match(expr: &SqlExpr) -> bool {
        match expr {
            SqlExpr::FtsMatch { .. } => true,
            SqlExpr::BinaryOp { left, right, .. } => {
                Self::expr_has_fts_match(left) || Self::expr_has_fts_match(right)
            }
            SqlExpr::UnaryOp { expr, .. } => Self::expr_has_fts_match(expr),
            SqlExpr::Paren(inner) => Self::expr_has_fts_match(inner),
            _ => false,
        }
    }
}
