// DDL execution: CREATE TABLE, DROP TABLE, ALTER TABLE, TRUNCATE, EXPLAIN, CTE

impl ApexExecutor {
    // ========== DDL Execution Methods ==========

    fn delta_path_for_table(table_path: &Path) -> PathBuf {
        let mut delta = table_path.to_path_buf();
        let name = delta.file_name().unwrap_or_default().to_string_lossy();
        delta.set_file_name(format!("{}.delta", name));
        delta
    }

    fn delta_meta_path_for_delta(delta_path: &Path) -> PathBuf {
        let mut meta = delta_path.to_path_buf();
        let name = meta.file_name().unwrap_or_default().to_string_lossy();
        meta.set_file_name(format!("{}.meta", name));
        meta
    }

    fn deltastore_path_for_table(table_path: &Path) -> PathBuf {
        let mut path = table_path.to_path_buf();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        path.set_file_name(format!("{}.deltastore", name));
        path
    }

    fn remove_table_sidecars(table_path: &Path) {
        let delta_path = Self::delta_path_for_table(table_path);
        let deltastore_path = Self::deltastore_path_for_table(table_path);
        let mut deltastore_tmp = table_path.to_path_buf();
        let name = deltastore_tmp
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        deltastore_tmp.set_file_name(format!("{}.deltastore.tmp", name));

        let _ = std::fs::remove_file(Self::delta_meta_path_for_delta(&delta_path));
        let _ = std::fs::remove_file(&delta_path);
        let _ = std::fs::remove_file(&deltastore_path);
        let _ = std::fs::remove_file(&deltastore_tmp);
    }

    fn materialize_table_sidecars(table_path: &Path) -> io::Result<()> {
        let delta_path = Self::delta_path_for_table(table_path);
        if delta_path.exists() {
            let storage = TableStorageBackend::open_for_compact(table_path)?;
            storage.compact()?;
            invalidate_storage_cache(table_path);
            crate::storage::engine::engine().invalidate(table_path);
        }

        let deltastore_path = Self::deltastore_path_for_table(table_path);
        if deltastore_path.exists() {
            let storage = TableStorageBackend::open_for_write(table_path)?;
            if storage.has_pending_deltas() {
                storage.save_full()?;
                invalidate_storage_cache(table_path);
                crate::storage::engine::engine().invalidate(table_path);
            }
        }

        Ok(())
    }

    /// Execute CREATE TABLE statement
    /// High-performance: O(1) - just creates file header
    fn execute_create_table(
        table_path: &Path,
        table: &str,
        columns: &[crate::query::sql_parser::ColumnDef],
        if_not_exists: bool,
    ) -> io::Result<ApexResult> {
        // Ensure parent directory exists (needed for named databases)
        if let Some(parent) = table_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        if table_path.exists() {
            if if_not_exists {
                // Return success without error
                return Ok(ApexResult::Scalar(0));
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("Table '{}' already exists", table),
                ));
            }
        }
        
        // Create empty storage file with schema
        TableStorageBackend::create(&table_path)?;
        let storage = TableStorageBackend::open_for_write(&table_path)?;
        
        // Add columns to schema (if provided)
        for col_def in columns {
            storage.add_column(&col_def.name, col_def.data_type.clone())?;
        }
        
        // Set constraints on the underlying schema
        {
            use crate::query::sql_parser::ColumnConstraintKind;
            use crate::storage::on_demand::ColumnConstraints;
            for col_def in columns {
                if !col_def.constraints.is_empty() {
                    let default_val = col_def.constraints.iter().find_map(|c| {
                        if let ColumnConstraintKind::Default(v) = c {
                            use crate::storage::on_demand::DefaultValue;
                            Some(match v {
                                Value::Int64(n) => DefaultValue::Int64(*n),
                                Value::Float64(f) => DefaultValue::Float64(*f),
                                Value::String(s) => DefaultValue::String(s.clone()),
                                Value::Bool(b) => DefaultValue::Bool(*b),
                                _ => DefaultValue::Null,
                            })
                        } else { None }
                    });
                    let check_sql = col_def.constraints.iter().find_map(|c| {
                        if let ColumnConstraintKind::Check(sql) = c {
                            Some(sql.clone())
                        } else { None }
                    });
                    let fk = col_def.constraints.iter().find_map(|c| {
                        if let ColumnConstraintKind::ForeignKey { ref_table, ref_column } = c {
                            Some((ref_table.clone(), ref_column.clone()))
                        } else { None }
                    });
                    storage.storage.set_column_constraints(&col_def.name, ColumnConstraints {
                        not_null: col_def.constraints.contains(&ColumnConstraintKind::NotNull),
                        primary_key: col_def.constraints.contains(&ColumnConstraintKind::PrimaryKey),
                        unique: col_def.constraints.contains(&ColumnConstraintKind::Unique),
                        default_value: default_val,
                        check_expr_sql: check_sql,
                        foreign_key: fk,
                        autoincrement: col_def.constraints.contains(&ColumnConstraintKind::Autoincrement),
                    });
                }
            }
        }
        
        storage.save()?;
        
        Ok(ApexResult::Scalar(0))
    }

    /// Execute DROP TABLE statement
    /// High-performance: O(1) - just deletes file
    fn execute_drop_table(table_path: &Path, table: &str, if_exists: bool) -> io::Result<ApexResult> {
        // Invalidate ALL caches to release file handles and mmaps
        invalidate_storage_cache(table_path);
        crate::storage::engine::engine().invalidate(table_path);
        
        if !table_path.exists() {
            if if_exists {
                return Ok(ApexResult::Scalar(0));
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Table '{}' does not exist", table),
                ));
            }
        }
        
        std::fs::remove_file(table_path)?;
        
        // Clean up associated files (WAL, delta, deltastore) in same directory as table
        let parent_dir = table_path.parent().unwrap_or(table_path);
        let file_stem = table_path.file_name().unwrap_or_default().to_string_lossy();
        let cleanup_extensions = [
            format!("{}.wal", file_stem),
            format!("{}.delta", file_stem),
            format!("{}.deltastore", file_stem),
        ];
        for name in &cleanup_extensions {
            let path = parent_dir.join(name);
            if path.exists() {
                let _ = std::fs::remove_file(&path);
            }
        }
        
        Ok(ApexResult::Scalar(0))
    }

    /// Execute ALTER TABLE statement
    fn execute_alter_table(
        table_path: &Path,
        table: &str,
        operation: &crate::query::sql_parser::AlterTableOp,
    ) -> io::Result<ApexResult> {
        use crate::query::sql_parser::AlterTableOp;
        
        if !table_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Table '{}' does not exist", table),
            ));
        }
        
        // Invalidate all caches before write (executor + StorageEngine)
        invalidate_storage_cache(&table_path);
        crate::storage::engine::engine().invalidate(&table_path);
        
        // Schema rewrites must see all committed append-only rows and cell deltas.
        // Keep SELECT fast by avoiding auto-compact there; pay this cost only for DDL.
        Self::materialize_table_sidecars(table_path)?;

        // Note: ALTER TABLE operations need to preserve existing data, so we use open_for_write
        // which loads all column data. For true schema-only operations (like TRUNCATE),
        // we can use open_for_schema_change which only loads metadata.
        let storage = TableStorageBackend::open_for_write(&table_path)?;
        
        match operation {
            AlterTableOp::AddColumn { name, data_type } => {
                storage.add_column(name, data_type.clone())?;
            }
            AlterTableOp::DropColumn { name } => {
                storage.drop_column(name)?;
            }
            AlterTableOp::RenameColumn { old_name, new_name } => {
                storage.rename_column(old_name, new_name)?;
            }
        }
        
        storage.save()?;
        
        // Invalidate all caches after write to ensure subsequent reads get fresh data
        invalidate_storage_cache(&table_path);
        crate::storage::engine::engine().invalidate(&table_path);
        
        Ok(ApexResult::Scalar(0))
    }

    /// Execute TRUNCATE TABLE statement
    /// High-performance: recreates empty file
    fn execute_truncate(storage_path: &Path) -> io::Result<ApexResult> {
        if !storage_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Table does not exist",
            ));
        }
        
        // Invalidate caches before write
        invalidate_storage_cache(storage_path);
        // On Windows, engine insert_cache may hold mmaps that block file truncate (OS error 1224)
        crate::storage::engine::engine().invalidate(storage_path);
        Self::remove_table_sidecars(storage_path);
        
        // OPTIMIZATION: Use open_for_schema_change - only loads metadata, NOT column data
        let old_storage = TableStorageBackend::open_for_schema_change(storage_path)?;
        let schema = old_storage.get_schema();
        drop(old_storage);
        
        // Recreate empty file with same schema
        TableStorageBackend::create(storage_path)?;
        // Use open_for_schema_change for adding columns (schema only)
        let storage = TableStorageBackend::open_for_schema_change(storage_path)?;
        for (name, dtype) in &schema {
            storage.add_column(name, dtype.clone())?;
        }
        storage.save()?;
        Self::remove_table_sidecars(storage_path);
        
        // Invalidate cache after write to ensure subsequent reads get fresh data
        invalidate_storage_cache(storage_path);
        invalidate_table_stats(&storage_path.to_string_lossy());
        crate::storage::engine::engine().invalidate(storage_path);
        
        Ok(ApexResult::Scalar(0))
    }

    // ========== EXPLAIN / CTE / INSERT SELECT ==========

    /// Execute EXPLAIN statement — returns a text description of the query plan
    fn execute_explain(stmt: SqlStatement, analyze: bool, base_dir: &Path, default_table_path: &Path) -> io::Result<ApexResult> {
        let mut plan_lines: Vec<String> = Vec::new();

        // Describe the statement type
        match &stmt {
            SqlStatement::Select(select) => {
                plan_lines.push("Query Plan:".to_string());
                // FROM
                if let Some(ref from) = select.from {
                    match from {
                        FromItem::Table { table, alias } => {
                            let alias_str = alias.as_ref().map(|a| format!(" AS {}", a)).unwrap_or_default();
                            plan_lines.push(format!("  Scan: table={}{}", table, alias_str));
                            let table_path = Self::resolve_table_path(table, base_dir, default_table_path);
                            if table_path.exists() {
                                if let Ok(backend) = get_cached_backend(&table_path) {
                                    let row_count = backend.row_count();
                                    let schema = backend.get_schema();
                                    plan_lines.push(format!("    Rows: ~{}", row_count));
                                    plan_lines.push(format!("    Columns: {}", schema.len()));
                                }
                            }
                        }
                        FromItem::Subquery { alias, .. } => {
                            plan_lines.push(format!("  Scan: subquery AS {}", alias));
                        }
                        FromItem::TableFunction { func, file, alias, .. } => {
                            let alias_str = alias.as_ref().map(|a| format!(" AS {}", a)).unwrap_or_default();
                            plan_lines.push(format!("  Scan: {}('{}'){}", func, file, alias_str));
                        }
                        FromItem::TopkDistance { col, k, metric, alias, .. } => {
                            let alias_str = alias.as_ref().map(|a| format!(" AS {}", a)).unwrap_or_default();
                            plan_lines.push(format!("  Scan: topk_distance({}, k={}, metric='{}'){}", col, k, metric, alias_str));
                        }
                        FromItem::DirectFile { file, alias } => {
                            let alias_str = alias.as_ref().map(|a| format!(" AS {}", a)).unwrap_or_default();
                            plan_lines.push(format!("  DirectFile: '{}'{}", file, alias_str));
                        }
                    }
                }
                // JOINs
                for join in &select.joins {
                    let jt = match join.join_type {
                        JoinType::Inner => "INNER JOIN",
                        JoinType::Left => "LEFT JOIN",
                        JoinType::Right => "RIGHT JOIN",
                        JoinType::Full => "FULL OUTER JOIN",
                        JoinType::Cross => "CROSS JOIN",
                    };
                    let table_name = match &join.right {
                        FromItem::Table { table, .. } => table.clone(),
                        FromItem::Subquery { alias, .. } => format!("(subquery) {}", alias),
                        FromItem::TableFunction { func, file, .. } => format!("{}('{}')", func, file),
                        FromItem::TopkDistance { col, k, metric, .. } => format!("topk_distance({}, k={}, metric='{}')", col, k, metric),
                        FromItem::DirectFile { file, .. } => format!("'{}'", file),
                    };
                    plan_lines.push(format!("  {}: {}", jt, table_name));
                }
                // WHERE
                if select.where_clause.is_some() {
                    plan_lines.push("  Filter: WHERE <predicate>".to_string());
                }
                // GROUP BY
                if !select.group_by.is_empty() {
                    plan_lines.push(format!("  GroupBy: {}", select.group_by.join(", ")));
                }
                // HAVING
                if select.having.is_some() {
                    plan_lines.push("  Filter: HAVING <predicate>".to_string());
                }
                // ORDER BY
                if !select.order_by.is_empty() {
                    let obs: Vec<String> = select.order_by.iter()
                        .map(|o| format!("{} {}", o.column, if o.descending { "DESC" } else { "ASC" }))
                        .collect();
                    plan_lines.push(format!("  Sort: {}", obs.join(", ")));
                }
                // LIMIT/OFFSET
                if let Some(limit) = select.limit {
                    plan_lines.push(format!("  Limit: {}", limit));
                }
                if let Some(offset) = select.offset {
                    plan_lines.push(format!("  Offset: {}", offset));
                }
                // Projection
                let proj: Vec<String> = select.columns.iter().map(|c| match c {
                    SelectColumn::All => "*".to_string(),
                    SelectColumn::AllExclude(exclude) => format!("* EXCLUDE ({})", exclude.join(", ")),
                    SelectColumn::AllReplace(reps) => {
                        let parts: Vec<String> = reps.iter().map(|(_, col)| col.clone()).collect();
                        format!("* REPLACE ({})", parts.join(", "))
                    }
                    SelectColumn::Columns(pattern) => format!("COLUMNS('{}')", pattern),
                    SelectColumn::Column(name) => name.clone(),
                    SelectColumn::ColumnAlias { column, alias } => format!("{} AS {}", column, alias),
                    SelectColumn::Aggregate { func, column, .. } => {
                        format!("{}({})", func, column.as_deref().unwrap_or("*"))
                    }
                    SelectColumn::Expression { alias, .. } => {
                        alias.as_deref().unwrap_or("<expr>").to_string()
                    }
                    SelectColumn::WindowFunction { name, alias, .. } => {
                        alias.as_deref().unwrap_or(name.as_str()).to_string()
                    }
                }).collect();
                plan_lines.push(format!("  Output: {}", proj.join(", ")));
            }
            SqlStatement::Insert { table, values, .. } => {
                plan_lines.push("Query Plan:".to_string());
                plan_lines.push(format!("  Insert: table={}, rows={}", table, values.len()));
            }
            SqlStatement::InsertSelect { table, .. } => {
                plan_lines.push("Query Plan:".to_string());
                plan_lines.push(format!("  InsertSelect: table={}", table));
            }
            SqlStatement::CreateTableAs { table, .. } => {
                plan_lines.push("Query Plan:".to_string());
                plan_lines.push(format!("  CreateTableAs: table={}", table));
            }
            SqlStatement::Update { table, .. } => {
                plan_lines.push("Query Plan:".to_string());
                plan_lines.push(format!("  Update: table={}", table));
            }
            SqlStatement::Delete { table, .. } => {
                plan_lines.push("Query Plan:".to_string());
                plan_lines.push(format!("  Delete: table={}", table));
            }
            SqlStatement::Union(_) => {
                plan_lines.push("Query Plan:".to_string());
                plan_lines.push("  Union".to_string());
            }
            SqlStatement::Cte { name, .. } => {
                plan_lines.push("Query Plan:".to_string());
                plan_lines.push(format!("  CTE: {}", name));
            }
            other => {
                plan_lines.push(format!("Query Plan: {:?}", std::mem::discriminant(other)));
            }
        }

        // EXPLAIN ANALYZE: actually run the query and report timing
        if analyze {
            let start = std::time::Instant::now();
            let result = Self::execute_parsed_multi(stmt, base_dir, default_table_path)?;
            let elapsed = start.elapsed();
            plan_lines.push(format!("  Actual Time: {:.3}ms", elapsed.as_secs_f64() * 1000.0));
            if let Ok(batch) = result.to_record_batch() {
                plan_lines.push(format!("  Actual Rows: {}", batch.num_rows()));
            }
        }

        // Return as a single-column RecordBatch with plan lines
        let plan_text = plan_lines.join("\n");
        let arr = arrow::array::StringArray::from(vec![plan_text.as_str()]);
        let schema = Schema::new(vec![Field::new("plan", ArrowDataType::Utf8, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(arr)])
            .map_err(|e| err_data(e.to_string()))?;
        Ok(ApexResult::Data(batch))
    }

    /// Execute CTE (WITH name AS (...) main_query)
    /// Materializes the CTE body into a temp table and rewrites the main query to reference it.
    /// For recursive CTEs, implements iterative fixpoint: anchor UNION ALL recursive until no new rows.
    fn execute_cte(name: &str, column_aliases: &[String], body: SqlStatement, main: SqlStatement, recursive: bool, base_dir: &Path, default_table_path: &Path) -> io::Result<ApexResult> {
        let temp_path = base_dir.join(format!("__cte_{}_{}.apex", name, std::process::id()));
        
        // Helper: materialize a RecordBatch into a temp table (create or append)
        let materialize_batch = |batch: &RecordBatch, path: &Path, create: bool| -> io::Result<()> {
            if create {
                let backend = TableStorageBackend::create(path)?;
                let schema = batch.schema();
                for field in schema.fields() {
                    let col_type = match field.data_type() {
                        ArrowDataType::Int64 | ArrowDataType::UInt64 => crate::data::DataType::Int64,
                        ArrowDataType::Float64 => crate::data::DataType::Float64,
                        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => crate::data::DataType::String,
                        ArrowDataType::Boolean => crate::data::DataType::Bool,
                        _ => crate::data::DataType::String,
                    };
                    backend.add_column(field.name(), col_type)?;
                }
                if batch.num_rows() > 0 {
                    Self::insert_batch_into_backend(&backend, batch)?;
                }
                backend.save()?;
            } else if batch.num_rows() > 0 {
                let backend = TableStorageBackend::open_for_write(path)?;
                Self::insert_batch_into_backend(&backend, batch)?;
                backend.save()?;
            }
            invalidate_storage_cache(path);
            Ok(())
        };

        if recursive {
            // Recursive CTE: body must be UNION ALL with anchor (left) and recursive part (right)
            let (anchor_stmt, recursive_stmt, _union_all) = match body {
                SqlStatement::Union(ref u) => {
                    ((*u.left).clone(), (*u.right).clone(), u.all)
                }
                _ => {
                    // Not a UNION — treat as non-recursive fallback
                    let cte_result = Self::execute_parsed_multi(body, base_dir, default_table_path)?;
                    let cte_batch = cte_result.to_record_batch()
                        .map_err(|e| err_data(format!("CTE body must return a result set: {}", e)))?;
                    materialize_batch(&cte_batch, &temp_path, true)?;
                    
                    let result = Self::execute_main_with_cte(name, main, base_dir, default_table_path, &temp_path)?;
                    Self::cleanup_temp_table(&temp_path);
                    return result;
                }
            };
            
            // Step 1: Execute anchor query
            let anchor_result = Self::execute_parsed_multi(anchor_stmt, base_dir, default_table_path)?;
            let mut anchor_batch = anchor_result.to_record_batch()
                .map_err(|e| err_data(format!("Recursive CTE anchor must return a result set: {}", e)))?;
            
            // Apply column aliases if provided: WITH RECURSIVE fact(n, val) AS (...)
            if !column_aliases.is_empty() {
                anchor_batch = Self::remap_batch_columns(&anchor_batch, column_aliases)?;
            }
            
            if anchor_batch.num_rows() == 0 {
                // Empty anchor → materialize empty table, run main
                materialize_batch(&anchor_batch, &temp_path, true)?;
                let result = Self::execute_main_with_cte(name, main, base_dir, default_table_path, &temp_path)?;
                Self::cleanup_temp_table(&temp_path);
                return result;
            }
            
            // Record anchor column names (defines CTE schema — uses aliases if provided)
            let anchor_col_names: Vec<String> = anchor_batch.schema().fields()
                .iter().map(|f| f.name().clone()).collect();
            
            // Materialize anchor into temp table
            materialize_batch(&anchor_batch, &temp_path, true)?;
            
            // Step 2: Iterative fixpoint — execute recursive part until no new rows
            // Use a working table to hold last iteration's new rows (same schema as anchor)
            let working_path = base_dir.join(format!("__cte_{}_work_{}.apex", name, std::process::id()));
            materialize_batch(&anchor_batch, &working_path, true)?;
            
            const MAX_ITERATIONS: usize = 1000;
            for _iter in 0..MAX_ITERATIONS {
                // Execute recursive part with CTE name → working table
                let recursive_result = Self::execute_main_with_cte(name, recursive_stmt.clone(), base_dir, default_table_path, &working_path)?;
                let new_batch = match recursive_result {
                    Ok(r) => r.to_record_batch().ok(),
                    Err(_) => None,
                };
                
                let new_batch = match new_batch {
                    Some(b) if b.num_rows() > 0 => b,
                    _ => break, // No new rows — fixpoint reached
                };
                
                // Remap recursive result columns to anchor column names (by position)
                let remapped = Self::remap_batch_columns(&new_batch, &anchor_col_names)?;
                
                // Append remapped rows to the main CTE temp table
                materialize_batch(&remapped, &temp_path, false)?;
                
                // Replace working table with remapped rows for next iteration
                Self::cleanup_temp_table(&working_path);
                materialize_batch(&remapped, &working_path, true)?;
            }
            
            // Clean up working table
            Self::cleanup_temp_table(&working_path);
            
            // Step 3: Execute main query against accumulated CTE table
            let result = Self::execute_main_with_cte(name, main, base_dir, default_table_path, &temp_path)?;
            Self::cleanup_temp_table(&temp_path);
            result
        } else {
            // Non-recursive CTE: materialize body, execute main
            let cte_result = Self::execute_parsed_multi(body, base_dir, default_table_path)?;
            let mut cte_batch = cte_result.to_record_batch()
                .map_err(|e| err_data(format!("CTE body must return a result set: {}", e)))?;
            if !column_aliases.is_empty() {
                cte_batch = Self::remap_batch_columns(&cte_batch, column_aliases)?;
            }
            materialize_batch(&cte_batch, &temp_path, true)?;

            let result = Self::execute_main_with_cte(name, main, base_dir, default_table_path, &temp_path)?;
            Self::cleanup_temp_table(&temp_path);
            result
        }
    }
    
    /// Helper: insert a RecordBatch into a TableStorageBackend
    fn insert_batch_into_backend(backend: &TableStorageBackend, batch: &RecordBatch) -> io::Result<()> {
        let schema = batch.schema();
        let mut rows: Vec<std::collections::HashMap<String, crate::data::Value>> = Vec::with_capacity(batch.num_rows());
        for row_idx in 0..batch.num_rows() {
            let mut row = std::collections::HashMap::new();
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let col = batch.column(col_idx);
                let val = Self::arrow_value_at_col(col, row_idx);
                row.insert(field.name().clone(), val);
            }
            rows.push(row);
        }
        backend.insert_rows(&rows)?;
        Ok(())
    }
    
    /// Helper: execute a statement with CTE name rewritten to reference temp table
    fn execute_main_with_cte(name: &str, main: SqlStatement, base_dir: &Path, default_table_path: &Path, temp_path: &Path) -> io::Result<io::Result<ApexResult>> {
        let temp_table_name = temp_path.file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        
        match main {
            SqlStatement::Cte { name: inner_name, column_aliases: inner_aliases, body: mut inner_body, main: inner_main, recursive: inner_recursive } => {
                // Rewrite outer CTE references inside inner CTE's body (chained CTEs)
                if let SqlStatement::Select(ref mut sel) = *inner_body {
                    Self::rewrite_cte_references_in_select(sel, name, &temp_table_name);
                }
                Ok(Self::execute_cte(&inner_name, &inner_aliases, *inner_body, *inner_main, inner_recursive, base_dir, temp_path))
            }
            SqlStatement::Select(mut select) => {
                Self::rewrite_cte_references_in_select(&mut select, name, &temp_table_name);
                Ok(Self::execute_parsed_multi(SqlStatement::Select(select), base_dir, default_table_path))
            }
            other => Ok(Self::execute_parsed_multi(other, base_dir, default_table_path)),
        }
    }
    
    /// Helper: rewrite FROM/JOIN references to CTE name → temp table name
    fn rewrite_cte_references_in_select(select: &mut SelectStatement, cte_name: &str, temp_table_name: &str) {
        if let Some(ref from) = select.from {
            if let FromItem::Table { table, .. } = from {
                if table.eq_ignore_ascii_case(cte_name) {
                    select.from = Some(FromItem::Table { table: temp_table_name.to_string(), alias: Some(cte_name.to_string()) });
                }
            }
        }
        for join in &mut select.joins {
            if let FromItem::Table { table, alias } = &join.right {
                if table.eq_ignore_ascii_case(cte_name) {
                    join.right = FromItem::Table {
                        table: temp_table_name.to_string(),
                        alias: Some(alias.clone().unwrap_or_else(|| cte_name.to_string())),
                    };
                }
            }
        }
    }
    
    /// Helper: remap a RecordBatch's column names by position to match target names.
    /// Used in recursive CTEs where the recursive part may have different column names than the anchor.
    fn remap_batch_columns(batch: &RecordBatch, target_names: &[String]) -> io::Result<RecordBatch> {
        let num_cols = batch.num_columns().min(target_names.len());
        let batch_schema = batch.schema();
        let mut fields = Vec::with_capacity(num_cols);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(num_cols);
        for i in 0..num_cols {
            let old_field = batch_schema.field(i);
            fields.push(Field::new(&target_names[i], old_field.data_type().clone(), old_field.is_nullable()));
            arrays.push(batch.column(i).clone());
        }
        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays)
            .map_err(|e| err_data(format!("Failed to remap CTE columns: {}", e)))
    }
    
    /// Helper: clean up temp CTE files
    fn cleanup_temp_table(path: &Path) {
        let delta_path = Self::delta_path_for_table(path);
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("apex.wal"));
        let _ = std::fs::remove_file(path.with_extension("apex.lock"));
        let _ = std::fs::remove_file(Self::delta_meta_path_for_delta(&delta_path));
        let _ = std::fs::remove_file(&delta_path);
        let _ = std::fs::remove_file(path.with_extension("apex.deltastore"));
        invalidate_storage_cache(path);
    }

    /// Execute INSERT ... SELECT statement
    fn execute_insert_select(
        target_path: &Path,
        columns: Option<&[String]>,
        query: SqlStatement,
        base_dir: &Path,
        default_table_path: &Path,
    ) -> io::Result<ApexResult> {
        // Step 1: Execute the SELECT query
        let select_result = Self::execute_parsed_multi(query, base_dir, default_table_path)?;
        let select_batch = select_result.to_record_batch()
            .map_err(|e| err_data(format!("INSERT SELECT query must return a result set: {}", e)))?;

        if select_batch.num_rows() == 0 {
            return Ok(ApexResult::Scalar(0));
        }

        // Step 2: Convert Arrow batch rows to Value rows for execute_insert
        let schema = select_batch.schema();
        let num_cols = if let Some(cols) = columns {
            cols.len().min(schema.fields().len())
        } else {
            schema.fields().len()
        };

        let mut values: Vec<Vec<crate::data::Value>> = Vec::with_capacity(select_batch.num_rows());
        for row_idx in 0..select_batch.num_rows() {
            let mut row: Vec<crate::data::Value> = Vec::with_capacity(num_cols);
            for col_idx in 0..num_cols {
                let col = select_batch.column(col_idx);
                let val = Self::arrow_value_at_col(col, row_idx);
                row.push(val);
            }
            values.push(row);
        }

        // Step 3: Determine column names
        let col_names: Option<Vec<String>> = if let Some(cols) = columns {
            Some(cols.to_vec())
        } else {
            // Use column names from the SELECT result, excluding _id
            let names: Vec<String> = schema.fields().iter()
                .take(num_cols)
                .map(|f| f.name().clone())
                .filter(|n| n != "_id")
                .collect();
            if names.len() < num_cols {
                // Had _id column, need to rebuild values without _id
                let id_indices: Vec<usize> = schema.fields().iter()
                    .enumerate()
                    .filter(|(_, f)| f.name() == "_id")
                    .map(|(i, _)| i)
                    .collect();
                if !id_indices.is_empty() {
                    let mut new_values: Vec<Vec<crate::data::Value>> = Vec::with_capacity(values.len());
                    for row in &values {
                        let filtered: Vec<crate::data::Value> = row.iter().enumerate()
                            .filter(|(i, _)| !id_indices.contains(i))
                            .map(|(_, v)| v.clone())
                            .collect();
                        new_values.push(filtered);
                    }
                    return Self::execute_insert(target_path, Some(&names), &new_values);
                }
                Some(names)
            } else {
                Some(names)
            }
        };

        Self::execute_insert(target_path, col_names.as_deref(), &values)
    }

    /// Execute CREATE TABLE ... AS SELECT statement
    fn execute_create_table_as(
        base_dir: &Path,
        default_table_path: &Path,
        table: &str,
        query: SqlStatement,
        if_not_exists: bool,
    ) -> io::Result<ApexResult> {
        let table_path = Self::resolve_table_path(table, base_dir, default_table_path);

        if table_path.exists() {
            if if_not_exists {
                return Ok(ApexResult::Scalar(0));
            }
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("Table '{}' already exists", table),
            ));
        }

        // Step 1: Execute the SELECT query
        let select_result = Self::execute_parsed_multi(query, base_dir, default_table_path)?;
        let select_batch = select_result.to_record_batch()
            .map_err(|e| err_data(format!("CTAS query must return a result set: {}", e)))?;

        // Step 2: Create the table with schema from SELECT result
        let schema = select_batch.schema();
        let backend = TableStorageBackend::create(&table_path)?;
        for field in schema.fields() {
            if field.name() == "_id" { continue; }
            let col_type = match field.data_type() {
                ArrowDataType::Int64 | ArrowDataType::UInt64 => crate::data::DataType::Int64,
                ArrowDataType::Float64 => crate::data::DataType::Float64,
                ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => crate::data::DataType::String,
                ArrowDataType::Boolean => crate::data::DataType::Bool,
                _ => crate::data::DataType::String,
            };
            backend.add_column(field.name(), col_type)?;
        }

        // Step 3: Insert rows if any
        let inserted = select_batch.num_rows();
        if inserted > 0 {
            let mut rows: Vec<std::collections::HashMap<String, crate::data::Value>> = Vec::with_capacity(inserted);
            for row_idx in 0..inserted {
                let mut row = std::collections::HashMap::new();
                for (col_idx, field) in schema.fields().iter().enumerate() {
                    if field.name() == "_id" { continue; }
                    let col = select_batch.column(col_idx);
                    let val = Self::arrow_value_at_col(col, row_idx);
                    row.insert(field.name().clone(), val);
                }
                rows.push(row);
            }
            backend.insert_rows(&rows)?;
        }
        backend.save()?;
        invalidate_storage_cache(&table_path);
        invalidate_table_stats(&table_path.to_string_lossy());

        Ok(ApexResult::Scalar(inserted as i64))
    }

    // ========== FTS DDL Handlers ==========

    /// Path of the FTS config JSON file for a given database directory.
    fn fts_config_path(base_dir: &Path) -> PathBuf {
        base_dir.join("fts_config.json")
    }

    /// Read the fts_config.json as a serde_json::Value (object).  Returns empty object if missing.
    fn read_fts_config(base_dir: &Path) -> serde_json::Value {
        let path = Self::fts_config_path(base_dir);
        if !path.exists() {
            return serde_json::Value::Object(serde_json::Map::new());
        }
        match std::fs::read_to_string(&path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
            Err(_) => serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Write the fts_config.json from a serde_json::Value.
    fn write_fts_config(base_dir: &Path, cfg: &serde_json::Value) {
        let path = Self::fts_config_path(base_dir);
        if let Ok(s) = serde_json::to_string(cfg) {
            let _ = std::fs::write(path, s);
        }
    }

    /// CREATE FTS INDEX ON table [(col1, col2)] [WITH (lazy_load=.., cache_size=..)]
    pub(super) fn execute_create_fts_index(
        base_dir: &Path,
        table: &str,
        fields: Option<&[String]>,
        lazy_load: bool,
        cache_size: usize,
    ) -> io::Result<ApexResult> {
        use crate::fts::{FtsConfig, FtsManager};

        // Update fts_config.json
        let mut cfg = Self::read_fts_config(base_dir);
        let obj = cfg.as_object_mut().ok_or_else(|| err_input("Corrupt fts_config.json"))?;
        let table_cfg = serde_json::json!({
            "enabled": true,
            "index_fields": fields.map(|f| serde_json::json!(f)).unwrap_or(serde_json::Value::Null),
            "config": {
                "lazy_load": lazy_load,
                "cache_size": cache_size
            }
        });
        obj.insert(table.to_string(), table_cfg);
        Self::write_fts_config(base_dir, &cfg);

        // Ensure FtsManager for this base_dir exists and the engine for this table is created
        let fts_cfg = FtsConfig {
            lazy_load,
            cache_size,
            ..FtsConfig::default()
        };
        let fts_dir = base_dir.join("fts_indexes");
        let mgr = {
            let existing = crate::query::executor::get_fts_manager(base_dir);
            match existing {
                Some(m) => m,
                None => {
                    let m = std::sync::Arc::new(FtsManager::new(&fts_dir, fts_cfg));
                    crate::query::executor::register_fts_manager(base_dir, m.clone());
                    m
                }
            }
        };
        // Warm up the engine (creates it if not present)
        let _ = mgr.get_engine(table);

        // Back-fill existing rows into FTS index
        let backfilled = Self::fts_backfill_table(base_dir, table, fields, mgr.clone()).unwrap_or(0);

        let fields_desc = fields
            .map(|f| f.join(", "))
            .unwrap_or_else(|| "all string cols".to_string());
        let msg = format!("FTS index created on '{}' (fields: {}, {} rows indexed)", table, fields_desc, backfilled);

        // Return a one-row status RecordBatch
        use arrow::array::StringArray;
        use arrow::datatypes::{Field, Schema};
        let schema = std::sync::Arc::new(Schema::new(vec![Field::new("status", ArrowDataType::Utf8, false)]));
        let batch = RecordBatch::try_new(schema, vec![std::sync::Arc::new(StringArray::from(vec![msg.as_str()]))])
            .map_err(|e| err_data(e.to_string()))?;
        Ok(ApexResult::Data(batch))
    }

    /// Back-fill existing table rows into an FTS engine.
    /// Reads all rows with their string columns and adds them to the FTS index.
    /// Returns the number of rows indexed.
    fn fts_backfill_table(
        base_dir: &Path,
        table: &str,
        fields: Option<&[String]>,
        mgr: std::sync::Arc<crate::fts::FtsManager>,
    ) -> io::Result<usize> {
        Self::fts_backfill_table_inner(base_dir, table, fields, mgr, true)
    }

    fn fts_backfill_table_sync(
        base_dir: &Path,
        table: &str,
        fields: Option<&[String]>,
        mgr: std::sync::Arc<crate::fts::FtsManager>,
    ) -> io::Result<usize> {
        Self::fts_backfill_table_inner(base_dir, table, fields, mgr, false)
    }

    fn fts_backfill_table_inner(
        base_dir: &Path,
        table: &str,
        fields: Option<&[String]>,
        mgr: std::sync::Arc<crate::fts::FtsManager>,
        allow_async: bool,
    ) -> io::Result<usize> {
        use arrow::array::{Int64Array, StringArray, UInt64Array};
        use arrow::compute;

        let table_path = base_dir.join(format!("{}.apex", table));
        if !table_path.exists() {
            return Ok(0);
        }

        let engine = mgr.get_engine(table).map_err(|e| err_data(e.to_string()))?;

        let storage = crate::storage::TableStorageBackend::open(&table_path)?;
        let schema = storage.get_schema();

        // Determine which string columns to index
        let string_cols: Vec<String> = if let Some(f) = fields {
            f.to_vec()
        } else {
            schema.iter()
                .filter(|(_, dt)| matches!(dt, crate::data::DataType::String))
                .map(|(n, _)| n.clone())
                .collect()
        };

        if string_cols.is_empty() {
            return Ok(0);
        }

        let estimated_count = storage.active_row_count() as usize;
        if allow_async && estimated_count > 100_000 {
            let base_dir = base_dir.to_path_buf();
            let table_name = table.to_string();
            let fields = fields.map(|f| f.to_vec());
            let mgr = mgr.clone();
            let handle = std::thread::spawn({
                let base_dir = base_dir.clone();
                let table_name = table_name.clone();
                move || {
                    let _ = Self::fts_backfill_table_sync(
                        &base_dir,
                        &table_name,
                        fields.as_deref(),
                        mgr,
                    );
                }
            });
            crate::query::executor::register_fts_backfill_task(&base_dir, &table_name, handle);
            return Ok(estimated_count);
        }

        if let Some((doc_ids, column_data)) = storage.read_fts_string_columns_mmap(&string_cols)? {
            if doc_ids.is_empty() || column_data.is_empty() {
                return Ok(0);
            }

            let columns: Vec<(String, Vec<&str>)> = column_data
                .iter()
                .filter_map(|(col_name, data)| {
                    if let crate::storage::on_demand::ColumnData::String { offsets, data } = data {
                        let values = (0..doc_ids.len())
                            .map(|row_idx| {
                                if row_idx + 1 < offsets.len() {
                                    let start = offsets[row_idx] as usize;
                                    let end = offsets[row_idx + 1] as usize;
                                    if end <= data.len() && start <= end {
                                        // Storage writes valid UTF-8; skip repeat validation during FTS build.
                                        unsafe { std::str::from_utf8_unchecked(&data[start..end]) }
                                    } else {
                                        ""
                                    }
                                } else {
                                    ""
                                }
                            })
                            .collect();
                        Some((col_name.clone(), values))
                    } else {
                        None
                    }
                })
                .collect();

            if !columns.is_empty() {
                let count = doc_ids.len();
                engine
                    .add_documents_arrow_str(&doc_ids, columns)
                    .map_err(|e| err_data(e.to_string()))?;
                engine.flush_async().map_err(|e| err_data(e.to_string()))?;
                return Ok(count);
            }
        }

        // Read _id + string columns
        let mut col_names = vec!["_id".to_string()];
        col_names.extend(string_cols.iter().cloned());
        col_names.sort();
        col_names.dedup();
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();

        let batch = storage.read_columns_to_arrow(Some(&col_refs), 0, None)?;
        if batch.num_rows() == 0 {
            return Ok(0);
        }

        // Extract row IDs
        let id_col = match batch.column_by_name("_id") {
            Some(c) => c,
            None => return Ok(0),
        };
        let ids: Vec<u64> = if let Some(arr) = id_col.as_any().downcast_ref::<UInt64Array>() {
            (0..arr.len()).map(|i| arr.value(i)).collect()
        } else if let Some(arr) = id_col.as_any().downcast_ref::<Int64Array>() {
            (0..arr.len()).map(|i| arr.value(i) as u64).collect()
        } else {
            return Ok(0);
        };

        let mut string_arrays: Vec<(String, arrow::array::ArrayRef)> = Vec::new();
        for col_name in &string_cols {
            if let Some(col) = batch.column_by_name(col_name) {
                if col.as_any().downcast_ref::<StringArray>().is_some() {
                    string_arrays.push((col_name.clone(), col.clone()));
                } else if let Ok(casted) = compute::cast(col, &ArrowDataType::Utf8) {
                    if casted.as_any().downcast_ref::<StringArray>().is_some() {
                        string_arrays.push((col_name.clone(), casted));
                    }
                }
            }
        }

        if string_arrays.is_empty() {
            return Ok(0);
        }

        let doc_ids: Vec<u32> = ids.iter().map(|id| *id as u32).collect();
        let columns = string_arrays
            .iter()
            .map(|(col_name, col)| {
                let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
                let values = (0..ids.len())
                    .map(|row_idx| {
                        if row_idx < arr.len() && !arr.is_null(row_idx) {
                            arr.value(row_idx)
                        } else {
                            ""
                        }
                    })
                    .collect();
                (col_name.clone(), values)
            })
            .collect();

        let count = ids.len();
        engine
            .add_documents_arrow_str(&doc_ids, columns)
            .map_err(|e| err_data(e.to_string()))?;
        engine.flush_async().map_err(|e| err_data(e.to_string()))?;
        Ok(count)
    }

    /// DROP FTS INDEX ON table — remove config entry and delete index files
    pub(super) fn execute_drop_fts_index(base_dir: &Path, table: &str) -> io::Result<ApexResult> {
        crate::query::executor::wait_fts_backfill(base_dir, table);

        // Update config
        let mut cfg = Self::read_fts_config(base_dir);
        if let Some(obj) = cfg.as_object_mut() {
            obj.remove(table);
        }
        Self::write_fts_config(base_dir, &cfg);

        // Remove from FTS manager if present
        if let Some(mgr) = crate::query::executor::get_fts_manager(base_dir) {
            let _ = mgr.remove_engine(table, true);
        }

        let msg = format!("FTS index dropped for table '{}'", table);
        use arrow::array::StringArray;
        use arrow::datatypes::{Field, Schema};
        let schema = std::sync::Arc::new(Schema::new(vec![Field::new("status", ArrowDataType::Utf8, false)]));
        let batch = RecordBatch::try_new(schema, vec![std::sync::Arc::new(StringArray::from(vec![msg.as_str()]))])
            .map_err(|e| err_data(e.to_string()))?;
        Ok(ApexResult::Data(batch))
    }

    /// ALTER FTS INDEX ON table ENABLE — mark enabled and back-fill missing rows
    pub(super) fn execute_alter_fts_index_enable(base_dir: &Path, table: &str) -> io::Result<ApexResult> {
        use crate::fts::{FtsConfig, FtsManager};

        // Update fts_config.json: set enabled = true
        let mut cfg = Self::read_fts_config(base_dir);
        let obj = cfg.as_object_mut().ok_or_else(|| err_input("Corrupt fts_config.json"))?;
        if let Some(entry) = obj.get_mut(table) {
            if let Some(map) = entry.as_object_mut() {
                map.insert("enabled".to_string(), serde_json::Value::Bool(true));
            }
        } else {
            obj.insert(table.to_string(), serde_json::json!({
                "enabled": true,
                "index_fields": null,
                "config": { "lazy_load": false, "cache_size": 10000 }
            }));
        }
        Self::write_fts_config(base_dir, &cfg);

        // Re-read to get the per-table settings
        let cfg2 = Self::read_fts_config(base_dir);
        let entry = cfg2.as_object().and_then(|o| o.get(table));
        let lazy_load = entry.and_then(|e| e.get("config")).and_then(|c| c.get("lazy_load")).and_then(|v| v.as_bool()).unwrap_or(false);
        let cache_size = entry.and_then(|e| e.get("config")).and_then(|c| c.get("cache_size")).and_then(|v| v.as_u64()).unwrap_or(10000) as usize;
        let fields: Option<Vec<String>> = entry
            .and_then(|e| e.get("index_fields"))
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|x| x.as_str().map(String::from)).collect());

        // Get or create FTS manager
        let fts_dir = base_dir.join("fts_indexes");
        let fts_cfg = FtsConfig { lazy_load, cache_size, ..FtsConfig::default() };
        let mgr = {
            let existing = crate::query::executor::get_fts_manager(base_dir);
            match existing {
                Some(m) => m,
                None => {
                    let m = std::sync::Arc::new(FtsManager::new(&fts_dir, fts_cfg));
                    crate::query::executor::register_fts_manager(base_dir, m.clone());
                    m
                }
            }
        };

        // Back-fill all rows into FTS (safe to re-index already-indexed rows)
        let backfilled = Self::fts_backfill_table(base_dir, table, fields.as_deref(), mgr.clone()).unwrap_or(0);

        let msg = format!("FTS index enabled for '{}' ({} rows indexed)", table, backfilled);
        use arrow::array::StringArray;
        use arrow::datatypes::{Field, Schema};
        let schema = std::sync::Arc::new(Schema::new(vec![Field::new("status", ArrowDataType::Utf8, false)]));
        let batch = RecordBatch::try_new(schema, vec![std::sync::Arc::new(StringArray::from(vec![msg.as_str()]))])
            .map_err(|e| err_data(e.to_string()))?;
        Ok(ApexResult::Data(batch))
    }

    /// ALTER FTS INDEX ON table DISABLE — mark disabled, keep files
    pub(super) fn execute_alter_fts_index_disable(base_dir: &Path, table: &str) -> io::Result<ApexResult> {
        let mut cfg = Self::read_fts_config(base_dir);
        if let Some(obj) = cfg.as_object_mut() {
            if let Some(entry) = obj.get_mut(table) {
                if let Some(map) = entry.as_object_mut() {
                    map.insert("enabled".to_string(), serde_json::Value::Bool(false));
                }
            }
        }
        Self::write_fts_config(base_dir, &cfg);

        let msg = format!("FTS index disabled for table '{}' (index files kept)", table);
        use arrow::array::StringArray;
        use arrow::datatypes::{Field, Schema};
        let schema = std::sync::Arc::new(Schema::new(vec![Field::new("status", ArrowDataType::Utf8, false)]));
        let batch = RecordBatch::try_new(schema, vec![std::sync::Arc::new(StringArray::from(vec![msg.as_str()]))])
            .map_err(|e| err_data(e.to_string()))?;
        Ok(ApexResult::Data(batch))
    }

    /// SHOW FTS INDEXES — list FTS-configured tables for this database and all named sub-databases
    pub(super) fn execute_show_fts_indexes(base_dir: &Path) -> io::Result<ApexResult> {
        use arrow::array::{BooleanArray, Int64Array, StringArray};
        use arrow::datatypes::{Field, Schema};

        let mut databases: Vec<String> = Vec::new();
        let mut tables: Vec<String> = Vec::new();
        let mut enabled_list: Vec<bool> = Vec::new();
        let mut fields_list: Vec<String> = Vec::new();
        let mut lazy_list: Vec<bool> = Vec::new();
        let mut cache_list: Vec<i64> = Vec::new();

        // Helper closure to add entries from a config object
        let mut add_from_config = |db_name: &str, obj: &serde_json::Map<String, serde_json::Value>| {
            for (tname, entry) in obj {
                let enabled = entry.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
                let fields = entry.get("index_fields")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", "))
                    .unwrap_or_else(|| "(all string cols)".to_string());
                let inner = entry.get("config");
                let lazy = inner.and_then(|c| c.get("lazy_load")).and_then(|v| v.as_bool()).unwrap_or(false);
                let cache = inner.and_then(|c| c.get("cache_size")).and_then(|v| v.as_i64()).unwrap_or(10000);
                databases.push(db_name.to_string());
                tables.push(tname.clone());
                enabled_list.push(enabled);
                fields_list.push(fields);
                lazy_list.push(lazy);
                cache_list.push(cache);
            }
        };

        // Current database (default / root)
        let cfg = Self::read_fts_config(base_dir);
        if let Some(obj) = cfg.as_object() {
            add_from_config("default", obj);
        }

        // Named sub-databases (immediate subdirectories that have fts_config.json)
        if let Ok(entries) = std::fs::read_dir(base_dir) {
            let mut sub_dirs: Vec<(String, std::path::PathBuf)> = entries
                .flatten()
                .filter_map(|e| {
                    let p = e.path();
                    if !p.is_dir() { return None; }
                    let name = p.file_name()?.to_string_lossy().into_owned();
                    if name == "fts_indexes" { return None; }
                    let sub_cfg = p.join("fts_config.json");
                    if sub_cfg.exists() { Some((name, p)) } else { None }
                })
                .collect();
            sub_dirs.sort_by(|a, b| a.0.cmp(&b.0));
            for (db_name, db_path) in sub_dirs {
                let sub_cfg = Self::read_fts_config(&db_path);
                if let Some(obj) = sub_cfg.as_object() {
                    add_from_config(&db_name, obj);
                }
            }
        }

        let schema = std::sync::Arc::new(Schema::new(vec![
            Field::new("database",   ArrowDataType::Utf8,    false),
            Field::new("table",      ArrowDataType::Utf8,    false),
            Field::new("enabled",    ArrowDataType::Boolean, false),
            Field::new("fields",     ArrowDataType::Utf8,    false),
            Field::new("lazy_load",  ArrowDataType::Boolean, false),
            Field::new("cache_size", ArrowDataType::Int64,   false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            std::sync::Arc::new(StringArray::from(databases)),
            std::sync::Arc::new(StringArray::from(tables)),
            std::sync::Arc::new(BooleanArray::from(enabled_list)),
            std::sync::Arc::new(StringArray::from(fields_list)),
            std::sync::Arc::new(BooleanArray::from(lazy_list)),
            std::sync::Arc::new(Int64Array::from(cache_list)),
        ]).map_err(|e| err_data(e.to_string()))?;
        Ok(ApexResult::Data(batch))
    }
}
