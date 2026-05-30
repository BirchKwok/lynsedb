// Transaction, DML (INSERT/DELETE/UPDATE), Index management, COPY, PRAGMA

/// Parsed pushdown filter: column index, op, and comparison value
struct PushdownFilter {
    col_idx: usize,
    op: u8,
    op_eq: bool,
    val_f64: f64,
}

#[derive(Clone)]
struct JsonNumericFilter {
    key: Vec<u8>,
    op: BinaryOperator,
    flipped: bool,
    val_f64: f64,
}

fn parse_pushdown_filter(
    filter_str: &str,
    schema: &arrow::datatypes::Schema,
) -> Option<PushdownFilter> {
    let bytes = filter_str.as_bytes();
    let op_pos = bytes.iter().position(|&b| b == b'>' || b == b'<' || b == b'=' || b == b'!')?;
    let col_name = &filter_str[..op_pos];
    let op = bytes[op_pos];
    let op_eq = op_pos + 1 < bytes.len() && bytes[op_pos + 1] == b'=';
    let val_start = if op_eq { op_pos + 2 } else { op_pos + 1 };
    let val_str = &filter_str[val_start..];
    let col_idx = schema.index_of(col_name).ok()?;
    let val_f64: f64 = if val_str.starts_with('\'') || val_str.starts_with('"') {
        return None;
    } else {
        val_str.parse().ok()?
    };
    Some(PushdownFilter { col_idx, op, op_eq, val_f64 })
}

#[inline]
fn csv_filter_match(field: &[u8], filter: &PushdownFilter) -> bool {
    if field.is_empty() { return false; }
    let val: Option<f64> = std::str::from_utf8(field).ok().and_then(|s| s.parse().ok());
    match val {
        Some(v) => match filter.op {
            b'>' if filter.op_eq => v >= filter.val_f64,
            b'<' if filter.op_eq => v <= filter.val_f64,
            b'>' => v > filter.val_f64,
            b'<' => v < filter.val_f64,
            b'=' => (v - filter.val_f64).abs() < 1e-12,
            b'!' => (v - filter.val_f64).abs() >= 1e-12,
            _ => true,
        },
        None => false,
    }
}

impl ApexExecutor {
    // ========== Transaction Execution Methods ==========

    /// Execute BEGIN [TRANSACTION] [READ ONLY]
    /// Returns the transaction ID as a scalar result
    fn execute_begin(read_only: bool) -> io::Result<ApexResult> {
        let mgr = crate::txn::txn_manager();
        let txn_id = if read_only {
            mgr.begin_read_only()
        } else {
            mgr.begin()
        };
        Ok(ApexResult::Scalar(txn_id as i64))
    }

    /// Execute COMMIT for a specific transaction
    /// Validates OCC conflicts, writes WAL transaction markers, then applies all buffered writes to storage.
    ///
    /// Crash recovery protocol:
    /// 1. Write TxnBegin to each affected table's WAL
    /// 2. Apply buffered writes to storage
    /// 3. Write TxnCommit to each affected table's WAL + flush/sync
    /// 4. On recovery: only replay DML between TxnBegin..TxnCommit pairs
    pub fn execute_commit_txn(
        txn_id: u64,
        base_dir: &Path,
        default_table_path: &Path,
    ) -> io::Result<ApexResult> {
        let mgr = crate::txn::txn_manager();

        // Commit validates OCC conflicts and returns buffered writes without
        // cloning them in the executor first.
        let writes = mgr.commit_with_writes(txn_id).map_err(|e| {
            io::Error::new(io::ErrorKind::Other, format!("Transaction conflict: {}", e))
        })?;

        // Collect affected table paths
        let mut affected_tables: std::collections::HashSet<std::path::PathBuf> =
            std::collections::HashSet::new();
        for write in &writes {
            let table_name = write.table();
            let table_path = Self::resolve_table_path(table_name, base_dir, default_table_path);
            affected_tables.insert(table_path);
        }

        // Phase 1: Write TxnBegin to each affected table's WAL, if that table
        // is using a WAL-backed durability mode. Do not use get_cached_backend()
        // here: that read path compacts pending .delta files and turns small
        // transaction commits into full-table rewrites.
        let mut wal_backends: std::collections::HashMap<std::path::PathBuf, TableStorageBackend> =
            std::collections::HashMap::new();
        for table_path in &affected_tables {
            if let Ok(Some(backend)) = Self::open_txn_wal_backend(table_path) {
                let _ = backend.storage.wal_write_txn_begin(txn_id);
                wal_backends.insert(table_path.clone(), backend);
            }
        }

        // Phase 2 (P0-4): Write buffered DML to WAL with txn_id BEFORE applying to storage
        // This ensures crash recovery can replay committed transactions from WAL.
        for write in &writes {
            use crate::txn::context::TxnWrite;
            let table_name = write.table();
            let table_path = Self::resolve_table_path(table_name, base_dir, default_table_path);
            if let Some(backend) = wal_backends.get(&table_path) {
                match write {
                    TxnWrite::Insert { data, row_id, .. } => {
                        use crate::storage::on_demand::ColumnValue as CV;
                        let wal_data: std::collections::HashMap<String, CV> = data
                            .iter()
                            .map(|(k, v)| {
                                let cv = match v {
                                    Value::Null => CV::Null,
                                    Value::Bool(b) => CV::Bool(*b),
                                    Value::Int64(i) => CV::Int64(*i),
                                    Value::Int32(i) => CV::Int64(*i as i64),
                                    Value::Float64(f) => CV::Float64(*f),
                                    Value::String(s) => CV::String(s.clone()),
                                    Value::UInt64(u) => CV::Int64(*u as i64),
                                    _ => CV::Null,
                                };
                                (k.clone(), cv)
                            })
                            .collect();
                        let _ = backend
                            .storage
                            .wal_write_txn_insert(txn_id, *row_id, wal_data);
                    }
                    TxnWrite::Delete { row_id, .. } => {
                        let _ = backend.storage.wal_write_txn_delete(txn_id, *row_id);
                    }
                    TxnWrite::Update { .. } => {
                        // Updates are applied as delta store changes (not WAL-logged individually)
                    }
                }
            }
        }

        // Phase 3: Apply buffered writes to storage
        let applied = Self::apply_txn_writes(&writes, base_dir, default_table_path);

        // Phase 4: Write TxnCommit to each affected table's WAL (flush + optional sync)
        for table_path in &affected_tables {
            if let Some(backend) = wal_backends.get(table_path) {
                let _ = backend.storage.wal_write_txn_commit(txn_id);
            }
        }

        // Invalidate StorageEngine cache for all affected tables
        let engine = crate::storage::engine::engine();
        for table_path in &affected_tables {
            engine.invalidate(table_path);
        }

        Ok(ApexResult::Scalar(applied))
    }

    fn open_txn_wal_backend(storage_path: &Path) -> io::Result<Option<TableStorageBackend>> {
        let wal_path = {
            let mut p = storage_path.to_path_buf();
            let ext = p
                .extension()
                .map(|e| format!("{}.wal", e.to_string_lossy()))
                .unwrap_or_else(|| "wal".to_string());
            p.set_extension(ext);
            p
        };
        if !wal_path.exists() {
            return Ok(None);
        }

        TableStorageBackend::open_for_insert_with_durability(
            storage_path,
            crate::storage::DurabilityLevel::Safe,
        )
        .map(Some)
    }

    /// Apply buffered writes while coalescing adjacent INSERTs into one storage append.
    fn apply_txn_writes(
        writes: &[crate::txn::context::TxnWrite],
        base_dir: &Path,
        default_table_path: &Path,
    ) -> i64 {
        use crate::txn::context::TxnWrite;

        let mut applied = 0i64;
        let mut idx = 0usize;
        while idx < writes.len() {
            match &writes[idx] {
                TxnWrite::Insert { table, data, .. } => {
                    let columns = Self::txn_insert_columns(data);
                    let mut values_list = Vec::new();
                    let mut row_maps = Vec::new();
                    let mut next_idx = idx;
                    while next_idx < writes.len() {
                        match &writes[next_idx] {
                            TxnWrite::Insert {
                                table: next_table,
                                data: next_data,
                                ..
                            } if next_table == table
                                && Self::txn_insert_columns(next_data) == columns =>
                            {
                                row_maps.push(next_data.clone());
                                values_list.push(
                                    columns
                                        .iter()
                                        .map(|c| next_data.get(c).cloned().unwrap_or(Value::Null))
                                        .collect::<Vec<_>>(),
                                );
                                next_idx += 1;
                            }
                            _ => break,
                        }
                    }

                    let table_path = Self::resolve_table_path(table, base_dir, default_table_path);
                    match Self::try_apply_txn_insert_delta(&table_path, &row_maps) {
                        Ok(Some(count)) => applied += count,
                        Ok(None) => {
                            match Self::execute_insert(&table_path, Some(&columns), &values_list) {
                                Ok(_) => applied += values_list.len() as i64,
                                Err(e) => {
                                    eprintln!("Warning: failed to apply txn insert batch: {}", e)
                                }
                            }
                        }
                        Err(e) => eprintln!("Warning: failed to apply txn insert delta: {}", e),
                    }
                    idx = next_idx;
                }
                _ => {
                    match Self::apply_txn_write(&writes[idx], base_dir, default_table_path) {
                        Ok(count) => applied += count,
                        Err(e) => eprintln!("Warning: failed to apply txn write: {}", e),
                    }
                    idx += 1;
                }
            }
        }
        applied
    }

    fn txn_insert_columns(data: &std::collections::HashMap<String, Value>) -> Vec<String> {
        let mut columns: Vec<String> = data.keys().cloned().collect();
        columns.sort_unstable();
        columns
    }

    /// Fast committed txn INSERT path for simple existing tables.
    ///
    /// The fallback `execute_insert` path is still used for schema evolution,
    /// partial INSERTs, constraints, indexes, or FTS tables.
    fn try_apply_txn_insert_delta(
        storage_path: &Path,
        rows: &[std::collections::HashMap<String, Value>],
    ) -> io::Result<Option<i64>> {
        if rows.is_empty() || !storage_path.exists() {
            return Ok(Some(0));
        }

        let storage = TableStorageBackend::open_for_insert(storage_path)?;
        if storage.storage.has_constraints() {
            return Ok(None);
        }

        let schema = storage.get_schema();
        let schema_cols: std::collections::HashSet<&str> =
            schema.iter().map(|(name, _)| name.as_str()).collect();
        for row in rows {
            if row.len() != schema_cols.len()
                || row
                    .keys()
                    .any(|name| name == "_id" || !schema_cols.contains(name.as_str()))
            {
                return Ok(None);
            }
        }

        let (base_dir, table_name) = base_dir_and_table(storage_path);
        {
            let idx_mgr_arc = get_index_manager(&base_dir, &table_name);
            let idx_mgr = idx_mgr_arc.lock();
            if !idx_mgr.list_indexes().is_empty() {
                return Ok(None);
            }
        }
        if Self::table_fts_enabled(&base_dir, &table_name) {
            return Ok(None);
        }

        let ids = storage.insert_rows_to_delta(rows)?;
        refresh_storage_cache_signature(storage_path);
        invalidate_table_stats(&storage_path.to_string_lossy());
        Ok(Some(ids.len() as i64))
    }

    fn table_fts_enabled(base_dir: &Path, table_name: &str) -> bool {
        let cfg = Self::read_fts_config(base_dir);
        cfg.as_object()
            .and_then(|o| o.get(table_name))
            .and_then(|entry| entry.get("enabled"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    /// Apply a single buffered write from a committed transaction
    fn apply_txn_write(
        write: &crate::txn::context::TxnWrite,
        base_dir: &Path,
        default_table_path: &Path,
    ) -> io::Result<i64> {
        use crate::txn::context::TxnWrite;
        match write {
            TxnWrite::Insert { table, data, .. } => {
                let table_path = Self::resolve_table_path(table, base_dir, default_table_path);
                let columns: Vec<String> = data.keys().cloned().collect();
                let values: Vec<Value> = columns.iter().map(|c| data[c].clone()).collect();
                let values_list = vec![values];
                Self::execute_insert(&table_path, Some(&columns), &values_list)?;
                Ok(1)
            }
            TxnWrite::Delete { table, row_id, .. } => {
                let table_path = Self::resolve_table_path(table, base_dir, default_table_path);
                let where_clause = SqlExpr::BinaryOp {
                    left: Box::new(SqlExpr::Column("_id".to_string())),
                    op: BinaryOperator::Eq,
                    right: Box::new(SqlExpr::Literal(Value::UInt64(*row_id))),
                };
                Self::execute_delete(&table_path, Some(&where_clause))?;
                Ok(1)
            }
            TxnWrite::Update {
                table,
                row_id,
                new_data,
                ..
            } => {
                let table_path = Self::resolve_table_path(table, base_dir, default_table_path);
                if let Some(count) =
                    Self::try_apply_txn_update_by_id_fast(&table_path, *row_id, new_data)?
                {
                    return Ok(count);
                }

                let assignments: Vec<(String, SqlExpr)> = new_data
                    .iter()
                    .map(|(col, val)| (col.clone(), SqlExpr::Literal(val.clone())))
                    .collect();
                let where_clause = SqlExpr::BinaryOp {
                    left: Box::new(SqlExpr::Column("_id".to_string())),
                    op: BinaryOperator::Eq,
                    right: Box::new(SqlExpr::Literal(Value::UInt64(*row_id))),
                };
                Self::execute_update(&table_path, &assignments, Some(&where_clause))?;
                Ok(1)
            }
        }
    }

    fn try_apply_txn_update_by_id_fast(
        storage_path: &Path,
        row_id: u64,
        new_data: &std::collections::HashMap<String, Value>,
    ) -> io::Result<Option<i64>> {
        if new_data.is_empty() || new_data.contains_key("_id") {
            return Ok(None);
        }

        let storage = TableStorageBackend::open_for_insert(storage_path)?;
        if storage.storage.has_constraints() {
            return Ok(None);
        }

        let (base_dir, table_name) = base_dir_and_table(storage_path);
        {
            let idx_mgr_arc = get_index_manager(&base_dir, &table_name);
            let idx_mgr = idx_mgr_arc.lock();
            if !idx_mgr.list_indexes().is_empty() {
                return Ok(None);
            }
        }
        if Self::table_fts_enabled(&base_dir, &table_name) {
            return Ok(None);
        }

        match storage.row_id_active_rcix(row_id)? {
            Some(true) => {}
            Some(false) => return Ok(Some(0)),
            None => return Ok(None),
        }

        if new_data.len() == 1 {
            let (col_name, value) = new_data.iter().next().unwrap();
            let bytes_opt = match value {
                Value::Float64(f) => Some(f.to_le_bytes()),
                Value::Int64(i) => Some(i.to_le_bytes()),
                Value::Int32(i) => Some((*i as i64).to_le_bytes()),
                _ => None,
            };
            if let Some(bytes) = bytes_opt {
                if let Some((count, physically_written)) =
                    storage.update_by_id_inplace(row_id, col_name, &bytes)?
                {
                    if physically_written {
                        invalidate_storage_cache(storage_path);
                        invalidate_table_stats(&storage_path.to_string_lossy());
                    }
                    return Ok(Some(count));
                }
            }
        }

        let batch: Vec<(u64, &str, Value)> = new_data
            .iter()
            .map(|(col, val)| (row_id, col.as_str(), val.clone()))
            .collect();
        storage.delta_batch_update_rows(&batch);
        storage.save_delta_store()?;
        invalidate_storage_cache(storage_path);
        invalidate_table_stats(&storage_path.to_string_lossy());
        Ok(Some(1))
    }

    /// Execute ROLLBACK for a specific transaction
    pub fn execute_rollback_txn(txn_id: u64) -> io::Result<ApexResult> {
        let mgr = crate::txn::txn_manager();
        mgr.rollback(txn_id)?;
        Ok(ApexResult::Scalar(0))
    }

    /// Execute a DML statement within a transaction (buffers writes in TxnContext)
    /// Returns the count of buffered operations.
    ///
    /// Statement-level rollback: an implicit savepoint is created before each DML.
    /// If the statement fails, only that statement's writes are rolled back;
    /// the transaction remains active with prior writes intact.
    pub fn execute_in_txn(
        txn_id: u64,
        stmt: SqlStatement,
        base_dir: &Path,
        default_table_path: &Path,
    ) -> io::Result<ApexResult> {
        let mgr = crate::txn::txn_manager();

        // Statement-level rollback: create implicit savepoint for DML statements
        let is_dml = matches!(
            &stmt,
            SqlStatement::Insert { .. } | SqlStatement::Delete { .. } | SqlStatement::Update { .. }
        );
        let implicit_sp = "__stmt_savepoint__";
        if is_dml {
            let _ = mgr.with_context(txn_id, |ctx| {
                ctx.savepoint(implicit_sp);
                Ok(())
            });
        }

        let result = Self::execute_in_txn_inner(txn_id, stmt, base_dir, default_table_path, mgr);

        // On DML failure, rollback to implicit savepoint (undo this statement only)
        if is_dml {
            if result.is_err() {
                let _ = mgr.with_context(txn_id, |ctx| ctx.rollback_to_savepoint(implicit_sp));
            } else {
                // Success — release the implicit savepoint
                let _ = mgr.with_context(txn_id, |ctx| ctx.release_savepoint(implicit_sp));
            }
        }

        result
    }

    /// Inner implementation of execute_in_txn (separated for statement-level rollback)
    fn execute_in_txn_inner(
        txn_id: u64,
        stmt: SqlStatement,
        base_dir: &Path,
        default_table_path: &Path,
        mgr: &'static crate::txn::manager::TxnManager,
    ) -> io::Result<ApexResult> {
        match stmt {
            SqlStatement::Insert {
                table,
                columns,
                values,
            } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                // Reserve synthetic row IDs from the storage allocator so transactional
                // inserts follow the same 1-based monotonic sequence as committed rows.
                let (existing_inserts, cached_base_id) = mgr.with_context(txn_id, |ctx| {
                    Ok((ctx.pending_insert_count(&table), ctx.insert_base_id(&table)))
                })?;
                let mut storage = None;
                let base_root = if let Some(base_id) = cached_base_id {
                    base_id
                } else {
                    let opened = TableStorageBackend::open_for_insert(&table_path)?;
                    let base_id = opened.next_id_value().max(crate::storage::FIRST_ROW_ID);
                    storage = Some(opened);
                    base_id
                };
                let remember_base_id = cached_base_id.is_none();
                let base_id = base_root + existing_inserts;
                let col_names: Vec<String> = if let Some(cols) = &columns {
                    cols.clone()
                } else {
                    let opened = match storage.take() {
                        Some(opened) => opened,
                        None => TableStorageBackend::open_for_insert(&table_path)?,
                    };
                    let schema = opened.get_schema();
                    schema.iter().map(|(n, _)| n.clone()).collect()
                };
                let mut pending_rows = Vec::with_capacity(values.len());
                for (ri, row_values) in values.iter().enumerate() {
                    let row_id = base_id + ri as u64;
                    let mut data = std::collections::HashMap::new();
                    for (i, val) in row_values.iter().enumerate() {
                        if i < col_names.len() {
                            data.insert(col_names[i].clone(), val.clone());
                        }
                    }
                    pending_rows.push((row_id, data));
                }
                let buffered = pending_rows.len() as i64;
                mgr.with_context(txn_id, |ctx| {
                    if remember_base_id {
                        ctx.remember_insert_base_id(&table, base_root);
                    }
                    for (row_id, data) in pending_rows {
                        ctx.buffer_insert(&table, row_id, data)?;
                    }
                    Ok(())
                })?;
                Ok(ApexResult::Scalar(buffered))
            }
            SqlStatement::Delete {
                table,
                where_clause,
            } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                let storage = TableStorageBackend::open_for_insert(&table_path)?;
                let mut buffered = 0i64;

                if let Some(where_expr) = &where_clause {
                    if let Some(rid) = Self::extract_id_equality_filter(where_expr) {
                        if let Some(batch) = storage.read_row_by_id_to_arrow(rid)? {
                            let old_data = Self::row_value_map_from_batch(&batch, 0);
                            mgr.with_context(txn_id, |ctx| {
                                ctx.buffer_delete(&table, rid, old_data)
                            })?;
                            return Ok(ApexResult::Scalar(1));
                        }
                        return Ok(ApexResult::Scalar(0));
                    }
                }

                // Read ALL columns so we can capture old_data for VersionStore (snapshot isolation)
                let batch = storage.read_columns_to_arrow(None, 0, None)?;
                let col_names: Vec<String> = batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect();

                let filter = if let Some(where_expr) = &where_clause {
                    Self::evaluate_predicate(&batch, where_expr)?
                } else {
                    BooleanArray::from(vec![true; batch.num_rows()])
                };

                for i in 0..filter.len() {
                    if filter.value(i) {
                        if let Some(id_col) = batch.column_by_name("_id") {
                            let rid = if let Some(a) = id_col.as_any().downcast_ref::<UInt64Array>()
                            {
                                Some(a.value(i))
                            } else if let Some(a) = id_col.as_any().downcast_ref::<Int64Array>() {
                                Some(a.value(i) as u64)
                            } else {
                                None
                            };
                            if let Some(rid) = rid {
                                // Capture old row data for VersionStore
                                let mut old_data = std::collections::HashMap::new();
                                for (ci, cn) in col_names.iter().enumerate() {
                                    if cn == "_id" {
                                        continue;
                                    }
                                    old_data.insert(
                                        cn.clone(),
                                        Self::arrow_value_at_col(batch.column(ci), i),
                                    );
                                }
                                mgr.with_context(txn_id, |ctx| {
                                    ctx.buffer_delete(&table, rid, old_data)
                                })?;
                                buffered += 1;
                            }
                        }
                    }
                }
                Ok(ApexResult::Scalar(buffered))
            }
            SqlStatement::Update {
                table,
                assignments,
                where_clause,
            } => {
                let table_path = Self::resolve_table_path(&table, base_dir, default_table_path);
                let storage = TableStorageBackend::open_for_insert(&table_path)?;

                if let Some(where_expr) = &where_clause {
                    if let Some(rid) = Self::extract_id_equality_filter(where_expr) {
                        if let Some(batch) = storage.read_row_by_id_to_arrow(rid)? {
                            let old_data = Self::row_value_map_from_batch(&batch, 0);
                            let mut new_data = std::collections::HashMap::new();
                            for (col, expr) in &assignments {
                                if let SqlExpr::Literal(val) = expr {
                                    new_data.insert(col.clone(), val.clone());
                                }
                            }
                            mgr.with_context(txn_id, |ctx| {
                                ctx.buffer_update(&table, rid, old_data, new_data)
                            })?;
                            return Ok(ApexResult::Scalar(1));
                        }
                        return Ok(ApexResult::Scalar(0));
                    }
                }

                // Read ALL columns for old_data capture (snapshot isolation)
                let batch = storage.read_columns_to_arrow(None, 0, None)?;
                let col_names: Vec<String> = batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect();
                let mut buffered = 0i64;

                let filter = if let Some(where_expr) = &where_clause {
                    Self::evaluate_predicate(&batch, where_expr)?
                } else {
                    BooleanArray::from(vec![true; batch.num_rows()])
                };

                for i in 0..filter.len() {
                    if filter.value(i) {
                        if let Some(id_col) = batch.column_by_name("_id") {
                            let rid = if let Some(a) = id_col.as_any().downcast_ref::<UInt64Array>()
                            {
                                Some(a.value(i))
                            } else if let Some(a) = id_col.as_any().downcast_ref::<Int64Array>() {
                                Some(a.value(i) as u64)
                            } else {
                                None
                            };
                            if let Some(rid) = rid {
                                // Capture old row data for VersionStore
                                let mut old_data = std::collections::HashMap::new();
                                for (ci, cn) in col_names.iter().enumerate() {
                                    if cn == "_id" {
                                        continue;
                                    }
                                    old_data.insert(
                                        cn.clone(),
                                        Self::arrow_value_at_col(batch.column(ci), i),
                                    );
                                }
                                let mut new_data = std::collections::HashMap::new();
                                for (col, expr) in &assignments {
                                    if let SqlExpr::Literal(val) = expr {
                                        new_data.insert(col.clone(), val.clone());
                                    }
                                }
                                mgr.with_context(txn_id, |ctx| {
                                    ctx.buffer_update(&table, rid, old_data, new_data)
                                })?;
                                buffered += 1;
                            }
                        }
                    }
                }
                Ok(ApexResult::Scalar(buffered))
            }
            SqlStatement::Select(ref select) => {
                // Extract table name from SELECT for overlay lookup
                let table_name = match &select.from {
                    Some(crate::query::sql_parser::FromItem::Table { table, .. }) => table.clone(),
                    _ => "default".to_string(),
                };

                // Execute SELECT normally against base storage
                let result = Self::execute_parsed_multi(stmt, base_dir, default_table_path)?;

                // Get transaction snapshot timestamp for MVCC visibility
                let snapshot_ts = mgr.get_snapshot_ts(txn_id)?;
                let version_store = mgr.get_version_store(&table_name);

                // Phase A: Overlay buffered writes from this transaction
                let writes = mgr.with_context(txn_id, |ctx| Ok(ctx.write_set().to_vec()))?;

                // Collect inserts and deletes for this table (own writes)
                let mut inserted_rows: Vec<(u64, &std::collections::HashMap<String, Value>)> =
                    Vec::new();
                let mut deleted_ids: std::collections::HashSet<u64> =
                    std::collections::HashSet::new();
                let mut updated_rows: Vec<(u64, &std::collections::HashMap<String, Value>)> =
                    Vec::new();
                for w in &writes {
                    use crate::txn::context::TxnWrite;
                    match w {
                        TxnWrite::Insert {
                            table,
                            row_id,
                            data,
                            ..
                        } if table == &table_name => {
                            inserted_rows.push((*row_id, data));
                        }
                        TxnWrite::Delete { table, row_id, .. } if table == &table_name => {
                            deleted_ids.insert(*row_id);
                        }
                        TxnWrite::Update {
                            table,
                            row_id,
                            new_data,
                            ..
                        } if table == &table_name => {
                            updated_rows.push((*row_id, new_data));
                        }
                        _ => {}
                    }
                }

                // Check if VersionStore has any entries (Phase B: cross-txn isolation)
                let has_versions = version_store.row_count() > 0;
                let has_own_writes = !inserted_rows.is_empty()
                    || !deleted_ids.is_empty()
                    || !updated_rows.is_empty();

                // If no version history and no own writes, return as-is (fast path)
                if !has_versions && !has_own_writes {
                    return Ok(result);
                }

                // Convert result to record batch and apply overlay
                let batch = result.to_record_batch()?;
                let schema = batch.schema();
                let num_cols = batch.num_columns();
                let col_names: Vec<String> =
                    schema.fields().iter().map(|f| f.name().clone()).collect();

                // Build row-level representation with MVCC snapshot filter + own-write overlay
                let mut rows: Vec<Vec<Value>> = Vec::new();
                for row_idx in 0..batch.num_rows() {
                    // Get row ID
                    let rid = if let Some(id_col) = batch.column_by_name("_id") {
                        if let Some(a) = id_col.as_any().downcast_ref::<UInt64Array>() {
                            Some(a.value(row_idx))
                        } else if let Some(a) = id_col.as_any().downcast_ref::<Int64Array>() {
                            Some(a.value(row_idx) as u64)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    // Check own deletes first
                    if let Some(rid) = rid {
                        if deleted_ids.contains(&rid) {
                            continue; // Deleted by this transaction
                        }
                    }

                    // Phase B: MVCC snapshot isolation — check VersionStore
                    if has_versions {
                        if let Some(rid) = rid {
                            // Check if VersionStore has a version chain for this row
                            let visible = version_store.read(rid, snapshot_ts);
                            let exists_in_vs = version_store.row_count() > 0 && {
                                // Check if this specific row has a chain
                                version_store.exists(rid, u64::MAX - 2) || // exists at any ts
                                version_store.read_latest(rid).is_some() ||
                                version_store.read(rid, u64::MAX - 2).is_some()
                            };

                            if exists_in_vs {
                                // Row has version history — use snapshot visibility
                                if version_store.read(rid, snapshot_ts).is_none() {
                                    // Not visible at snapshot_ts:
                                    // Either inserted after snapshot or deleted before snapshot
                                    // Check if it's an insert after snapshot (begin_ts > snapshot_ts)
                                    let is_deleted = {
                                        let chains = version_store.chains_ref();
                                        chains
                                            .get(&rid)
                                            .map(|c| c.is_deleted_at(snapshot_ts))
                                            .unwrap_or(false)
                                    };
                                    if is_deleted {
                                        // Row was deleted before our snapshot — already gone
                                        continue;
                                    }
                                    // Row was inserted after our snapshot — invisible
                                    continue;
                                }
                                // Visible: check if we should use version data instead of storage data
                                if let Some(versioned_data) = visible {
                                    // Use versioned data (may differ from current storage)
                                    let mut row = Vec::with_capacity(num_cols);
                                    for cn in &col_names {
                                        if cn == "_id" {
                                            row.push(Value::UInt64(rid));
                                        } else if let Some(val) = versioned_data.get(cn) {
                                            row.push(val.clone());
                                        } else {
                                            // Column not in version data, use storage value
                                            let ci =
                                                col_names.iter().position(|n| n == cn).unwrap();
                                            row.push(Self::arrow_value_at_col(
                                                batch.column(ci),
                                                row_idx,
                                            ));
                                        }
                                    }
                                    // Apply own updates on top
                                    for (uid, new_data) in &updated_rows {
                                        if rid == *uid {
                                            for (col, val) in *new_data {
                                                if let Some(ci) =
                                                    col_names.iter().position(|n| n == col)
                                                {
                                                    row[ci] = val.clone();
                                                }
                                            }
                                        }
                                    }
                                    rows.push(row);
                                    continue;
                                }
                            }
                        }
                    }

                    // No version history for this row — use storage data as-is
                    let mut row = Vec::with_capacity(num_cols);
                    for col_idx in 0..num_cols {
                        row.push(Self::arrow_value_at_col(batch.column(col_idx), row_idx));
                    }

                    // Apply own updates
                    if !updated_rows.is_empty() {
                        if let Some(rid) = rid {
                            for (uid, new_data) in &updated_rows {
                                if rid == *uid {
                                    for (col, val) in *new_data {
                                        if let Some(ci) = col_names.iter().position(|n| n == col) {
                                            row[ci] = val.clone();
                                        }
                                    }
                                }
                            }
                        }
                    }

                    rows.push(row);
                }

                // Append buffered inserts (own writes)
                for (row_id, insert_data) in &inserted_rows {
                    let mut row = vec![Value::Null; num_cols];
                    if let Some(ci) = col_names.iter().position(|n| n == "_id") {
                        row[ci] = Value::Int64(*row_id as i64);
                    }
                    for (col, val) in *insert_data {
                        if let Some(ci) = col_names.iter().position(|n| n == col) {
                            row[ci] = val.clone();
                        }
                    }
                    rows.push(row);
                }

                // Convert back to RecordBatch
                Self::rows_to_apex_result(&col_names, &rows, &schema)
            }
            SqlStatement::Commit => Self::execute_commit_txn(txn_id, base_dir, default_table_path),
            SqlStatement::Rollback => Self::execute_rollback_txn(txn_id),
            SqlStatement::Savepoint { name } => {
                mgr.with_context(txn_id, |ctx| {
                    ctx.savepoint(&name);
                    Ok(())
                })?;
                Ok(ApexResult::Scalar(0))
            }
            SqlStatement::RollbackToSavepoint { name } => {
                mgr.with_context(txn_id, |ctx| ctx.rollback_to_savepoint(&name))?;
                Ok(ApexResult::Scalar(0))
            }
            SqlStatement::ReleaseSavepoint { name } => {
                mgr.with_context(txn_id, |ctx| ctx.release_savepoint(&name))?;
                Ok(ApexResult::Scalar(0))
            }
            other => Self::execute_parsed_multi(other, base_dir, default_table_path),
        }
    }

    // ========== Index DDL Execution Methods ==========

    /// Execute CREATE INDEX statement
    /// Builds the index from existing data in the table
    fn execute_create_index(
        base_dir: &Path,
        default_table_path: &Path,
        name: &str,
        table: &str,
        columns: &[String],
        unique: bool,
        index_type_str: Option<&str>,
        if_not_exists: bool,
    ) -> io::Result<ApexResult> {
        use crate::storage::index::IndexType;

        if columns.is_empty() {
            return Err(err_input("CREATE INDEX requires at least one column"));
        }

        let idx_type = match index_type_str {
            Some("HASH") => IndexType::Hash,
            Some("BTREE") | Some("B-TREE") | Some("B_TREE") => IndexType::BTree,
            None => {
                // Default: HASH for equality-heavy columns, BTREE otherwise
                IndexType::Hash
            }
            Some(other) => {
                return Err(err_input(format!(
                    "Unknown index type: {}. Use HASH or BTREE",
                    other
                )))
            }
        };

        let table_path = Self::resolve_table_path(table, base_dir, default_table_path);
        if !table_path.exists() {
            return Err(err_not_found(format!("Table '{}' does not exist", table)));
        }

        // Get the IndexManager for this table
        let idx_mgr_arc = get_index_manager(base_dir, table);
        let mut idx_mgr = idx_mgr_arc.lock();

        // Check if already exists
        if idx_mgr.get_index_meta(name).is_some() {
            if if_not_exists {
                return Ok(ApexResult::Scalar(0));
            }
            return Err(err_input(format!(
                "Index '{}' already exists on table '{}'",
                name, table
            )));
        }

        // Determine column data type from table schema (use first column's type)
        let storage = TableStorageBackend::open(&table_path)?;
        let schema = storage.get_schema();
        let first_col = &columns[0];
        let data_type = schema
            .iter()
            .find(|(n, _)| n == first_col)
            .map(|(_, dt)| dt.clone())
            .ok_or_else(|| {
                err_not_found(format!(
                    "Column '{}' not found in table '{}'",
                    first_col, table
                ))
            })?;

        // Validate all columns exist
        for col in columns {
            if !schema.iter().any(|(n, _)| n == col) {
                return Err(err_not_found(format!(
                    "Column '{}' not found in table '{}'",
                    col, table
                )));
            }
        }

        // Create the index (supports single or multi-column)
        idx_mgr.create_index_multi(name, columns, idx_type, unique, data_type)?;

        // Build index from existing data
        let row_count = storage.row_count();
        if row_count > 0 {
            const INDEX_BUILD_BATCH_ROWS: usize = 65_536;

            // Read _id column and all indexed columns
            let mut col_refs_owned: Vec<String> = vec!["_id".to_string()];
            for c in columns {
                if !col_refs_owned.contains(c) {
                    col_refs_owned.push(c.clone());
                }
            }
            let col_refs: Vec<&str> = col_refs_owned.iter().map(|s| s.as_str()).collect();

            let build_result = (|| -> io::Result<()> {
                let mut start_row = 0usize;
                let total_rows = row_count as usize;

                while start_row < total_rows {
                    let limit = (total_rows - start_row).min(INDEX_BUILD_BATCH_ROWS);
                    let batch = storage.read_columns_to_arrow_window(
                        Some(&col_refs),
                        start_row,
                        Some(limit),
                    )?;

                    if batch.num_rows() == 0 {
                        break;
                    }

                    let id_col = batch
                        .column_by_name("_id")
                        .ok_or_else(|| err_data("_id column not found"))?;

                    let id_arr = id_col.as_any().downcast_ref::<UInt64Array>();
                    let id_arr_i64 = id_col.as_any().downcast_ref::<Int64Array>();

                    for row in 0..batch.num_rows() {
                        let row_id: u64 = if let Some(arr) = id_arr {
                            arr.value(row)
                        } else if let Some(arr) = id_arr_i64 {
                            arr.value(row) as u64
                        } else {
                            (start_row + row) as u64
                        };

                        // Extract values from all indexed columns
                        let mut col_vals = std::collections::HashMap::with_capacity(columns.len());
                        for col in columns {
                            if let Some(data_col) = batch.column_by_name(col) {
                                col_vals.insert(col.clone(), Self::arrow_value_at_col(data_col, row));
                            }
                        }
                        idx_mgr.on_insert(row_id, &col_vals)?;
                    }

                    start_row += batch.num_rows();
                }

                Ok(())
            })();

            if let Err(err) = build_result {
                let _ = idx_mgr.drop_index(name);
                return Err(err);
            }
        }

        // Save index to disk
        idx_mgr.save()?;

        Ok(ApexResult::Scalar(row_count as i64))
    }

    /// Execute DROP INDEX statement
    fn execute_drop_index(
        base_dir: &Path,
        name: &str,
        table: &str,
        if_exists: bool,
    ) -> io::Result<ApexResult> {
        let idx_mgr_arc = get_index_manager(base_dir, table);
        let mut idx_mgr = idx_mgr_arc.lock();

        if idx_mgr.get_index_meta(name).is_none() {
            if if_exists {
                return Ok(ApexResult::Scalar(0));
            }
            return Err(err_not_found(format!(
                "Index '{}' not found on table '{}'",
                name, table
            )));
        }

        idx_mgr.drop_index(name)?;
        idx_mgr.save()?;

        // Invalidate index cache to reload on next access
        invalidate_index_cache(base_dir, table);

        Ok(ApexResult::Scalar(0))
    }

    /// Convert rows of Value back into an ApexResult with a RecordBatch matching the given schema.
    /// Used by P0-6 read-your-writes overlay.
    fn rows_to_apex_result(
        col_names: &[String],
        rows: &[Vec<Value>],
        schema: &Arc<arrow::datatypes::Schema>,
    ) -> io::Result<ApexResult> {
        use arrow::array::{
            ArrayBuilder, BooleanBuilder, Float64Builder, Int64Builder, StringBuilder,
            UInt64Builder,
        };
        use arrow::datatypes::DataType;

        let num_rows = rows.len();
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());

        for (col_idx, field) in schema.fields().iter().enumerate() {
            match field.data_type() {
                DataType::Int64 => {
                    let mut builder = Int64Builder::with_capacity(num_rows);
                    for row in rows {
                        match row.get(col_idx) {
                            Some(Value::Int64(v)) => builder.append_value(*v),
                            Some(Value::Int32(v)) => builder.append_value(*v as i64),
                            Some(Value::UInt64(v)) => builder.append_value(*v as i64),
                            Some(Value::Null) | None => builder.append_null(),
                            _ => builder.append_null(),
                        }
                    }
                    arrays.push(Arc::new(builder.finish()));
                }
                DataType::UInt64 => {
                    let mut builder = UInt64Builder::with_capacity(num_rows);
                    for row in rows {
                        match row.get(col_idx) {
                            Some(Value::UInt64(v)) => builder.append_value(*v),
                            Some(Value::Int64(v)) => builder.append_value(*v as u64),
                            Some(Value::Null) | None => builder.append_null(),
                            _ => builder.append_null(),
                        }
                    }
                    arrays.push(Arc::new(builder.finish()));
                }
                DataType::Float64 => {
                    let mut builder = Float64Builder::with_capacity(num_rows);
                    for row in rows {
                        match row.get(col_idx) {
                            Some(Value::Float64(v)) => builder.append_value(*v),
                            Some(Value::Float32(v)) => builder.append_value(*v as f64),
                            Some(Value::Null) | None => builder.append_null(),
                            _ => builder.append_null(),
                        }
                    }
                    arrays.push(Arc::new(builder.finish()));
                }
                DataType::Utf8 => {
                    let mut builder = StringBuilder::with_capacity(num_rows, num_rows * 16);
                    for row in rows {
                        match row.get(col_idx) {
                            Some(Value::String(v)) => builder.append_value(v),
                            Some(Value::Null) | None => builder.append_null(),
                            _ => builder.append_null(),
                        }
                    }
                    arrays.push(Arc::new(builder.finish()));
                }
                DataType::Boolean => {
                    let mut builder = BooleanBuilder::with_capacity(num_rows);
                    for row in rows {
                        match row.get(col_idx) {
                            Some(Value::Bool(v)) => builder.append_value(*v),
                            Some(Value::Null) | None => builder.append_null(),
                            _ => builder.append_null(),
                        }
                    }
                    arrays.push(Arc::new(builder.finish()));
                }
                _ => {
                    // Fallback: null array
                    let mut builder = StringBuilder::with_capacity(num_rows, 0);
                    for _ in 0..num_rows {
                        builder.append_null();
                    }
                    arrays.push(Arc::new(builder.finish()));
                }
            }
        }

        // Make all fields nullable to avoid Arrow errors when overlay produces nulls
        let nullable_fields: Vec<Field> = schema
            .fields()
            .iter()
            .map(|f| Field::new(f.name(), f.data_type().clone(), true))
            .collect();
        let nullable_schema = Arc::new(Schema::new(nullable_fields));
        let batch = RecordBatch::try_new(nullable_schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        Ok(ApexResult::Data(batch))
    }

    /// Extract a Value from an Arrow ArrayRef at a given row index (for index building)
    fn arrow_value_at_col(col: &ArrayRef, row: usize) -> Value {
        use arrow::array::{BooleanArray, Float32Array};
        if col.is_null(row) {
            return Value::Null;
        }
        if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
            Value::Int64(arr.value(row))
        } else if let Some(arr) = col.as_any().downcast_ref::<UInt64Array>() {
            Value::UInt64(arr.value(row))
        } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
            Value::Float64(arr.value(row))
        } else if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
            Value::Float32(arr.value(row))
        } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
            Value::String(arr.value(row).to_string())
        } else if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
            Value::Bool(arr.value(row))
        } else {
            Value::Null
        }
    }

    fn row_value_map_from_batch(
        batch: &RecordBatch,
        row: usize,
    ) -> std::collections::HashMap<String, Value> {
        let mut data = std::collections::HashMap::new();
        for (ci, field) in batch.schema().fields().iter().enumerate() {
            let name = field.name();
            if name == "_id" {
                continue;
            }
            data.insert(
                name.clone(),
                Self::arrow_value_at_col(batch.column(ci), row),
            );
        }
        data
    }

    // ========== Index Maintenance Helpers ==========

    /// Public API: Notify indexes about rows inserted via non-SQL path (Python store() API).
    /// Called from engine.rs after write()/write_typed() completes.
    /// `storage_path` is the .apex file path, `inserted_ids` are the new row IDs.
    pub fn notify_indexes_after_write(storage_path: &Path, inserted_ids: &[u64]) {
        if inserted_ids.is_empty() {
            return;
        }
        let (base_dir, table_name) = base_dir_and_table(storage_path);
        let idx_mgr_arc = get_index_manager(&base_dir, &table_name);
        let mut idx_mgr = idx_mgr_arc.lock();
        if idx_mgr.list_indexes().is_empty() {
            return;
        }

        let indexed_cols: Vec<String> = idx_mgr
            .list_indexes()
            .iter()
            .map(|m| m.column_name.clone())
            .collect();
        let mut col_names: Vec<String> = vec!["_id".to_string()];
        col_names.extend(indexed_cols.iter().cloned());
        col_names.sort();
        col_names.dedup();

        // Open a read backend to get inserted row data
        if let Ok(backend) = TableStorageBackend::open(storage_path) {
            let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
            if let Ok(batch) = backend.read_columns_to_arrow(Some(&col_refs), 0, None) {
                let id_col = batch.column_by_name("_id");
                let id_set: std::collections::HashSet<u64> = inserted_ids.iter().copied().collect();
                for row in 0..batch.num_rows() {
                    let row_id = if let Some(col) = id_col {
                        if let Some(arr) = col.as_any().downcast_ref::<UInt64Array>() {
                            arr.value(row)
                        } else if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                            arr.value(row) as u64
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    };

                    if !id_set.contains(&row_id) {
                        continue;
                    }

                    let mut col_vals = std::collections::HashMap::new();
                    for col_name in &indexed_cols {
                        if let Some(col) = batch.column_by_name(col_name) {
                            col_vals.insert(col_name.clone(), Self::arrow_value_at_col(col, row));
                        }
                    }
                    let _ = idx_mgr.on_insert(row_id, &col_vals);
                }
                let _ = idx_mgr.save();
            }
        }
    }

    /// Public API: Notify indexes about rows deleted via non-SQL path (Python delete() API).
    pub fn notify_indexes_after_delete(
        storage_path: &Path,
        deleted_ids: &[u64],
        column_values: &[std::collections::HashMap<String, Value>],
    ) {
        if deleted_ids.is_empty() {
            return;
        }
        let (base_dir, table_name) = base_dir_and_table(storage_path);
        let idx_mgr_arc = get_index_manager(&base_dir, &table_name);
        let mut idx_mgr = idx_mgr_arc.lock();
        if idx_mgr.list_indexes().is_empty() {
            return;
        }
        for (i, &row_id) in deleted_ids.iter().enumerate() {
            if i < column_values.len() {
                idx_mgr.on_delete(row_id, &column_values[i]);
            }
        }
        let _ = idx_mgr.save();
    }

    /// Notify indexes that rows were inserted.
    /// Called after SQL INSERT completes. Reads newly inserted rows' indexed columns.
    fn notify_index_insert(
        storage_path: &Path,
        storage: &TableStorageBackend,
        start_row_count: u64,
    ) {
        let (base_dir, table_name) = base_dir_and_table(storage_path);
        let idx_mgr_arc = get_index_manager(&base_dir, &table_name);
        let mut idx_mgr = idx_mgr_arc.lock();
        if idx_mgr.list_indexes().is_empty() {
            return;
        }

        // Read _id + indexed columns for new rows
        let indexed_cols: Vec<String> = idx_mgr
            .list_indexes()
            .iter()
            .map(|m| m.column_name.clone())
            .collect();
        let mut col_names: Vec<String> = vec!["_id".to_string()];
        col_names.extend(indexed_cols.iter().cloned());
        col_names.sort();
        col_names.dedup();

        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
        let new_count = storage.row_count();
        if new_count <= start_row_count {
            return;
        }
        // Read all rows and only process new ones (rows after start_row_count)
        if let Ok(batch) = storage.read_columns_to_arrow(Some(&col_refs), 0, None) {
            let id_col = batch.column_by_name("_id");
            // Process rows from start_row_count onwards
            let start = start_row_count as usize;
            for row in start..batch.num_rows() {
                let row_id = if let Some(col) = id_col {
                    if let Some(arr) = col.as_any().downcast_ref::<UInt64Array>() {
                        arr.value(row)
                    } else if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                        arr.value(row) as u64
                    } else {
                        continue;
                    }
                } else {
                    continue;
                };

                let mut col_vals = std::collections::HashMap::new();
                for col_name in &indexed_cols {
                    if let Some(col) = batch.column_by_name(col_name) {
                        col_vals.insert(col_name.clone(), Self::arrow_value_at_col(col, row));
                    }
                }
                let _ = idx_mgr.on_insert(row_id, &col_vals);
            }
            let _ = idx_mgr.save();
        }
    }

    /// Notify indexes that rows were deleted.
    /// `deleted_ids` contains the _ids of deleted rows with their column values.
    fn notify_index_delete(
        storage_path: &Path,
        deleted_entries: &[(u64, std::collections::HashMap<String, Value>)],
    ) {
        if deleted_entries.is_empty() {
            return;
        }
        let (base_dir, table_name) = base_dir_and_table(storage_path);
        let idx_mgr_arc = get_index_manager(&base_dir, &table_name);
        let mut idx_mgr = idx_mgr_arc.lock();
        if idx_mgr.list_indexes().is_empty() {
            return;
        }
        for (row_id, col_vals) in deleted_entries {
            idx_mgr.on_delete(*row_id, col_vals);
        }
        let _ = idx_mgr.save();
    }

    /// Notify FTS index about newly inserted rows.
    /// Called after SQL INSERT completes. Indexes new rows' string columns in FTS.
    fn notify_fts_insert(storage_path: &Path, storage: &TableStorageBackend, start_row_count: u64) {
        use arrow::array::{Int64Array, StringArray, UInt64Array};

        let (base_dir, table_name) = base_dir_and_table(storage_path);

        // Check if FTS is enabled for this table via config
        let cfg = Self::read_fts_config(&base_dir);
        let table_cfg = cfg.as_object().and_then(|o| o.get(&table_name));
        let enabled = table_cfg
            .and_then(|e| e.get("enabled"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !enabled {
            return;
        }

        let fields: Option<Vec<String>> = table_cfg
            .and_then(|e| e.get("index_fields"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            });

        // Get or create FTS manager for this database
        let mgr_arc = {
            if let Some(m) = get_fts_manager(&base_dir) {
                m
            } else {
                let lazy_load = table_cfg
                    .and_then(|e| e.get("config"))
                    .and_then(|c| c.get("lazy_load"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let cache_size = table_cfg
                    .and_then(|e| e.get("config"))
                    .and_then(|c| c.get("cache_size"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10000) as usize;
                let fts_dir = base_dir.join("fts_indexes");
                let fts_cfg = crate::fts::FtsConfig {
                    lazy_load,
                    cache_size,
                    ..crate::fts::FtsConfig::default()
                };
                let m = std::sync::Arc::new(crate::fts::FtsManager::new(&fts_dir, fts_cfg));
                crate::query::executor::register_fts_manager(&base_dir, m.clone());
                m
            }
        };

        let engine = match mgr_arc.get_engine(&table_name) {
            Ok(e) => e,
            Err(_) => return,
        };
        crate::query::executor::wait_fts_backfill(&base_dir, &table_name);

        // Determine string columns to index
        let schema = storage.get_schema();
        let string_cols: Vec<String> = if let Some(f) = &fields {
            f.clone()
        } else {
            schema
                .iter()
                .filter(|(_, dt)| matches!(dt, crate::data::DataType::String))
                .map(|(n, _)| n.clone())
                .collect()
        };
        if string_cols.is_empty() {
            return;
        }

        let new_count = storage.row_count();
        if new_count <= start_row_count {
            return;
        }

        let mut col_names = vec!["_id".to_string()];
        col_names.extend(string_cols.iter().cloned());
        col_names.sort();
        col_names.dedup();
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();

        let batch = match storage.read_columns_to_arrow(Some(&col_refs), 0, None) {
            Ok(b) => b,
            Err(_) => return,
        };

        let id_col = match batch.column_by_name("_id") {
            Some(c) => c,
            None => return,
        };

        let start = start_row_count as usize;
        let mut ids: Vec<u64> = Vec::new();
        let mut col_data: Vec<Vec<String>> = vec![Vec::new(); string_cols.len()];

        for row in start..batch.num_rows() {
            let rid = if let Some(arr) = id_col.as_any().downcast_ref::<UInt64Array>() {
                arr.value(row)
            } else if let Some(arr) = id_col.as_any().downcast_ref::<Int64Array>() {
                arr.value(row) as u64
            } else {
                continue;
            };
            ids.push(rid);
            for (ci, col_name) in string_cols.iter().enumerate() {
                let val = batch
                    .column_by_name(col_name)
                    .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                    .map(|arr| {
                        if arr.is_null(row) {
                            String::new()
                        } else {
                            arr.value(row).to_string()
                        }
                    })
                    .unwrap_or_default();
                col_data[ci].push(val);
            }
        }

        if ids.is_empty() {
            return;
        }

        let columns: Vec<(String, Vec<String>)> = string_cols.into_iter().zip(col_data).collect();
        let _ = engine.add_documents_columnar(ids, columns);
        let _ = engine.flush();
    }

    /// Notify FTS index about deleted rows.
    /// Called after SQL DELETE completes. Removes deleted doc IDs from FTS index.
    fn notify_fts_delete(
        storage_path: &Path,
        deleted_entries: &[(u64, std::collections::HashMap<String, Value>)],
    ) {
        if deleted_entries.is_empty() {
            return;
        }
        let (base_dir, table_name) = base_dir_and_table(storage_path);

        // Check if FTS is enabled for this table
        let cfg = Self::read_fts_config(&base_dir);
        let enabled = cfg
            .as_object()
            .and_then(|o| o.get(&table_name))
            .and_then(|e| e.get("enabled"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !enabled {
            return;
        }

        // Only sync if manager is already registered (avoid creating one just for a delete)
        let mgr_arc = match get_fts_manager(&base_dir) {
            Some(m) => m,
            None => return,
        };

        let engine = match mgr_arc.get_engine(&table_name) {
            Ok(e) => e,
            Err(_) => return,
        };
        crate::query::executor::wait_fts_backfill(&base_dir, &table_name);

        let ids: Vec<u64> = deleted_entries.iter().map(|(id, _)| *id).collect();
        let _ = engine.remove_documents(&ids);
        let _ = engine.flush();
    }

    // ========== DML Execution Methods ==========

    /// Execute INSERT statement
    fn execute_insert(
        storage_path: &Path,
        columns: Option<&[String]>,
        values: &[Vec<Value>],
    ) -> io::Result<ApexResult> {
        use std::collections::HashMap;

        if !storage_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Table does not exist",
            ));
        }

        // Invalidate cache before write
        invalidate_storage_cache(storage_path);

        // Use open_for_write to load all data (needed for correct column alignment)
        let storage = TableStorageBackend::open_for_write(storage_path)?;

        // Get column names from schema or explicit list
        let col_names: Vec<String> = if let Some(cols) = columns {
            cols.to_vec()
        } else {
            storage
                .get_schema()
                .iter()
                .map(|(n, _)| n.clone())
                .collect()
        };

        // Build a schema type lookup for auto-coercing string→timestamp/date
        let schema_types: std::collections::HashMap<String, crate::storage::on_demand::ColumnType> = {
            let schema = storage.get_schema();
            schema
                .iter()
                .map(|(name, dt)| {
                    (
                        name.clone(),
                        crate::storage::backend::datatype_to_column_type(dt),
                    )
                })
                .collect()
        };

        // Build row-based data with proper NULL handling via Value::Null
        // Auto-coerce string→timestamp/date based on schema type
        let mut rows: Vec<HashMap<String, Value>> = Vec::with_capacity(values.len());
        for row_values in values {
            let mut row = HashMap::new();
            for (i, value) in row_values.iter().enumerate() {
                if i < col_names.len() {
                    let col_name = &col_names[i];
                    let col_schema_type = schema_types.get(col_name).copied();
                    let coerced = match value {
                        Value::String(v) => {
                            match col_schema_type {
                                Some(crate::storage::on_demand::ColumnType::Timestamp) => {
                                    Value::Timestamp(Self::parse_timestamp_string(v))
                                }
                                Some(crate::storage::on_demand::ColumnType::Date) => {
                                    Value::Date(Self::parse_date_string(v) as i32)
                                }
                                Some(crate::storage::on_demand::ColumnType::Binary) => {
                                    // Auto-encode JSON-style float array string '[1.0,2.0,…]' as binary vector
                                    Self::try_parse_vector_string(v)
                                        .map(Value::Binary)
                                        .unwrap_or_else(|| value.clone())
                                }
                                _ => value.clone(),
                            }
                        }
                        _ => value.clone(),
                    };
                    row.insert(col_name.clone(), coerced);
                }
            }
            rows.push(row);
        }

        let rows_inserted = values.len() as i64;

        // Enforce NOT NULL constraints
        if storage.storage.has_constraints() {
            for (row_idx, row_values) in values.iter().enumerate() {
                for (i, value) in row_values.iter().enumerate() {
                    if i < col_names.len() {
                        let col_name = &col_names[i];
                        let cons = storage.storage.get_column_constraints(col_name);
                        if cons.not_null && matches!(value, Value::Null) {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                format!("NOT NULL constraint violated: column '{}' cannot be NULL (row {})", col_name, row_idx + 1),
                            ));
                        }
                    }
                }
                // Check for missing NOT NULL columns — allow if DEFAULT is set
                let schema = storage.get_schema();
                for (schema_col, _) in &schema {
                    if !col_names.iter().any(|c| c == schema_col) {
                        let cons = storage.storage.get_column_constraints(schema_col);
                        if cons.not_null && cons.default_value.is_none() && !cons.autoincrement {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                format!("NOT NULL constraint violated: column '{}' has no value and no DEFAULT", schema_col),
                            ));
                        }
                    }
                }
            }
        }

        // Fill in DEFAULT values for missing columns into row maps
        if storage.storage.has_constraints() {
            let schema = storage.get_schema();
            for (schema_col, _) in &schema {
                if !col_names.iter().any(|c| c == schema_col) {
                    let cons = storage.storage.get_column_constraints(schema_col);
                    if let Some(ref dv) = cons.default_value {
                        use crate::storage::on_demand::DefaultValue;
                        let default_val = match dv {
                            DefaultValue::Int64(v) => Value::Int64(*v),
                            DefaultValue::Float64(v) => Value::Float64(*v),
                            DefaultValue::String(v) => Value::String(v.clone()),
                            DefaultValue::Bool(v) => Value::Bool(*v),
                            DefaultValue::Null => Value::Null,
                        };
                        for row in rows.iter_mut() {
                            row.entry(schema_col.clone())
                                .or_insert_with(|| default_val.clone());
                        }
                    }
                }
            }
        }

        // AUTOINCREMENT: auto-fill sequential values for autoincrement columns
        if storage.storage.has_constraints() {
            let schema = storage.get_schema();
            for (schema_col, _) in &schema {
                let cons = storage.storage.get_column_constraints(schema_col);
                if cons.autoincrement {
                    // Find current max value for this column
                    let col_refs = [schema_col.as_str()];
                    let existing = storage.read_columns_to_arrow(Some(&col_refs), 0, None)?;
                    let mut next_val: i64 = if let Some(col) = existing.column_by_name(schema_col) {
                        if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                            arr.iter().filter_map(|v| v).max().unwrap_or(0) + 1
                        } else {
                            1
                        }
                    } else {
                        1
                    };
                    // Fill in autoincrement values for rows that don't have this column set
                    for row in rows.iter_mut() {
                        if !row.contains_key(schema_col)
                            || matches!(row.get(schema_col), Some(Value::Null))
                        {
                            row.insert(schema_col.clone(), Value::Int64(next_val));
                            next_val += 1;
                        }
                    }
                }
            }
        }

        // Enforce UNIQUE / PRIMARY KEY constraints
        if storage.storage.has_constraints() {
            // Collect columns that need uniqueness checks
            let schema = storage.get_schema();
            let unique_cols: Vec<String> = schema
                .iter()
                .filter(|(name, _)| {
                    let c = storage.storage.get_column_constraints(name);
                    c.unique || c.primary_key
                })
                .map(|(name, _)| name.clone())
                .collect();

            if !unique_cols.is_empty() {
                // Read existing values for uniqueness columns
                let col_refs: Vec<&str> = unique_cols.iter().map(|s| s.as_str()).collect();
                let existing_batch = storage.read_columns_to_arrow(Some(&col_refs), 0, None)?;

                for uc in &unique_cols {
                    let constraint_kind = if storage.storage.get_column_constraints(uc).primary_key
                    {
                        "PRIMARY KEY"
                    } else {
                        "UNIQUE"
                    };

                    // Collect new values for this column from the INSERT
                    let mut new_vals: Vec<Value> = Vec::new();
                    for row_values in values.iter() {
                        for (i, value) in row_values.iter().enumerate() {
                            if i < col_names.len() && col_names[i] == *uc {
                                new_vals.push(value.clone());
                            }
                        }
                    }

                    // Check duplicates within new values themselves
                    {
                        let mut seen = std::collections::HashSet::new();
                        for v in &new_vals {
                            if !matches!(v, Value::Null) {
                                let key = format!("{:?}", v);
                                if !seen.insert(key) {
                                    return Err(io::Error::new(
                                        io::ErrorKind::InvalidInput,
                                        format!("{} constraint violated: duplicate value in column '{}'", constraint_kind, uc),
                                    ));
                                }
                            }
                        }
                    }

                    // Check against existing data
                    if let Some(existing_col) = existing_batch.column_by_name(uc) {
                        use arrow::array::{
                            BooleanArray, Float64Array, Int64Array, StringArray, UInt64Array,
                        };
                        for new_val in &new_vals {
                            if matches!(new_val, Value::Null) {
                                continue;
                            }
                            let len = existing_col.len();
                            for row in 0..len {
                                if existing_col.is_null(row) {
                                    continue;
                                }
                                let matches = match new_val {
                                    Value::Int64(v) => existing_col
                                        .as_any()
                                        .downcast_ref::<Int64Array>()
                                        .map(|a| a.value(row) == *v)
                                        .or_else(|| {
                                            existing_col
                                                .as_any()
                                                .downcast_ref::<UInt64Array>()
                                                .map(|a| a.value(row) as i64 == *v)
                                        })
                                        .unwrap_or(false),
                                    Value::Int32(v) => existing_col
                                        .as_any()
                                        .downcast_ref::<Int64Array>()
                                        .map(|a| a.value(row) == *v as i64)
                                        .unwrap_or(false),
                                    Value::Float64(v) => existing_col
                                        .as_any()
                                        .downcast_ref::<Float64Array>()
                                        .map(|a| (a.value(row) - v).abs() < f64::EPSILON)
                                        .unwrap_or(false),
                                    Value::String(v) => existing_col
                                        .as_any()
                                        .downcast_ref::<StringArray>()
                                        .map(|a| a.value(row) == v.as_str())
                                        .unwrap_or(false),
                                    Value::Bool(v) => existing_col
                                        .as_any()
                                        .downcast_ref::<BooleanArray>()
                                        .map(|a| a.value(row) == *v)
                                        .unwrap_or(false),
                                    _ => false,
                                };
                                if matches {
                                    return Err(io::Error::new(
                                        io::ErrorKind::InvalidInput,
                                        format!("{} constraint violated: duplicate value in column '{}'", constraint_kind, uc),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Enforce CHECK constraints
        if storage.storage.has_constraints() {
            let schema = storage.get_schema();
            for (schema_col, _) in &schema {
                let cons = storage.storage.get_column_constraints(schema_col);
                if let Some(ref check_sql) = cons.check_expr_sql {
                    // Parse the CHECK expression once
                    let check_expr = {
                        // Wrap in a SELECT WHERE to reuse the expression parser
                        let parse_sql = format!("SELECT 1 FROM _dummy WHERE {}", check_sql);
                        match crate::query::sql_parser::SqlParser::parse(&parse_sql) {
                            Ok(crate::query::sql_parser::SqlStatement::Select(sel)) => {
                                sel.where_clause.clone()
                            }
                            _ => None,
                        }
                    };
                    if let Some(ref expr) = check_expr {
                        // Evaluate CHECK for each row
                        for (row_idx, row_values) in values.iter().enumerate() {
                            // Build a 1-row RecordBatch with this row's values
                            let mut fields = Vec::new();
                            let mut arrays: Vec<ArrayRef> = Vec::new();
                            for (i, value) in row_values.iter().enumerate() {
                                if i < col_names.len() {
                                    let cn = &col_names[i];
                                    match value {
                                        Value::Int64(v) => {
                                            fields.push(Field::new(cn, ArrowDataType::Int64, true));
                                            arrays.push(Arc::new(Int64Array::from(vec![*v])));
                                        }
                                        Value::Float64(v) => {
                                            fields.push(Field::new(
                                                cn,
                                                ArrowDataType::Float64,
                                                true,
                                            ));
                                            arrays.push(Arc::new(Float64Array::from(vec![*v])));
                                        }
                                        Value::String(v) => {
                                            fields.push(Field::new(cn, ArrowDataType::Utf8, true));
                                            arrays.push(Arc::new(StringArray::from(vec![
                                                v.as_str()
                                            ])));
                                        }
                                        Value::Bool(v) => {
                                            fields.push(Field::new(
                                                cn,
                                                ArrowDataType::Boolean,
                                                true,
                                            ));
                                            arrays.push(Arc::new(BooleanArray::from(vec![*v])));
                                        }
                                        Value::Null => {
                                            fields.push(Field::new(cn, ArrowDataType::Utf8, true));
                                            arrays.push(Arc::new(StringArray::from(vec![
                                                Option::<&str>::None,
                                            ])));
                                        }
                                        _ => {
                                            fields.push(Field::new(cn, ArrowDataType::Utf8, true));
                                            arrays.push(Arc::new(StringArray::from(vec![
                                                format!("{:?}", value).as_str(),
                                            ])));
                                        }
                                    }
                                }
                            }
                            if !fields.is_empty() {
                                let batch_schema = Arc::new(Schema::new(fields));
                                if let Ok(row_batch) = RecordBatch::try_new(batch_schema, arrays) {
                                    match Self::evaluate_predicate(&row_batch, expr) {
                                        Ok(mask) => {
                                            if mask.len() > 0 && !mask.value(0) {
                                                return Err(io::Error::new(
                                                    io::ErrorKind::InvalidInput,
                                                    format!("CHECK constraint violated: {} (column '{}', row {})", check_sql, schema_col, row_idx + 1),
                                                ));
                                            }
                                        }
                                        Err(_) => {
                                            // If evaluation fails (e.g., column not in batch), treat as violation
                                            // This handles cases where the CHECK references a column not present
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Enforce FOREIGN KEY constraints
        if storage.storage.has_constraints() {
            let (base_dir, _) = base_dir_and_table(storage_path);
            let schema = storage.get_schema();
            for (schema_col, _) in &schema {
                let cons = storage.storage.get_column_constraints(schema_col);
                if let Some((ref ref_table, ref ref_column)) = cons.foreign_key {
                    // Find column index in inserted data
                    let col_idx = col_names.iter().position(|n| n == schema_col);
                    if col_idx.is_none() {
                        continue;
                    }
                    let col_idx = col_idx.unwrap();

                    // Open referenced table
                    let ref_path = base_dir.join(format!("{}.apex", ref_table));
                    if !ref_path.exists() {
                        return Err(io::Error::new(
                            io::ErrorKind::NotFound,
                            format!(
                                "FOREIGN KEY: referenced table '{}' does not exist",
                                ref_table
                            ),
                        ));
                    }
                    let ref_storage = TableStorageBackend::open(&ref_path)?;
                    let ref_batch =
                        ref_storage.read_columns_to_arrow(Some(&[ref_column.as_str()]), 0, None)?;
                    let ref_col_arr = ref_batch.column_by_name(ref_column);

                    // Check each inserted value exists in referenced column
                    for (row_idx, row_values) in values.iter().enumerate() {
                        if col_idx >= row_values.len() {
                            continue;
                        }
                        let val = &row_values[col_idx];
                        if matches!(val, Value::Null) {
                            continue;
                        } // NULL is allowed in FK

                        let found = if let Some(ref_arr) = ref_col_arr {
                            let mut exists = false;
                            for r in 0..ref_arr.len() {
                                if ref_arr.is_null(r) {
                                    continue;
                                }
                                let matches = match val {
                                    Value::Int64(v) => ref_arr
                                        .as_any()
                                        .downcast_ref::<Int64Array>()
                                        .map(|a| a.value(r) == *v)
                                        .unwrap_or(false),
                                    Value::Float64(v) => ref_arr
                                        .as_any()
                                        .downcast_ref::<Float64Array>()
                                        .map(|a| (a.value(r) - v).abs() < f64::EPSILON)
                                        .unwrap_or(false),
                                    Value::String(v) => ref_arr
                                        .as_any()
                                        .downcast_ref::<StringArray>()
                                        .map(|a| a.value(r) == v.as_str())
                                        .unwrap_or(false),
                                    Value::Bool(v) => ref_arr
                                        .as_any()
                                        .downcast_ref::<BooleanArray>()
                                        .map(|a| a.value(r) == *v)
                                        .unwrap_or(false),
                                    _ => false,
                                };
                                if matches {
                                    exists = true;
                                    break;
                                }
                            }
                            exists
                        } else {
                            false
                        };

                        if !found {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                format!("FOREIGN KEY constraint violated: value in column '{}' not found in {}.{} (row {})", schema_col, ref_table, ref_column, row_idx + 1),
                            ));
                        }
                    }
                }
            }
        }

        let storage = if storage.has_delta() {
            storage.compact()?;
            TableStorageBackend::open_for_write(storage_path)?
        } else {
            storage
        };

        // Capture row count before insert for index maintenance
        let pre_insert_count = storage.row_count();

        // Use insert_rows for proper NULL handling (Value::Null → ColumnValue::Null)
        storage.insert_rows(&rows)?;
        storage.save_full()?;

        // Update indexes for newly inserted rows
        Self::notify_index_insert(storage_path, &storage, pre_insert_count);
        // Update FTS index for newly inserted rows
        Self::notify_fts_insert(storage_path, &storage, pre_insert_count);

        // Invalidate cache after write to ensure subsequent reads get fresh data
        invalidate_storage_cache(storage_path);
        invalidate_table_stats(&storage_path.to_string_lossy());

        Ok(ApexResult::Scalar(rows_inserted))
    }

    /// Execute COPY table TO 'file.parquet' — export table data to Parquet file
    fn execute_copy_to_parquet(
        storage_path: &Path,
        table_name: &str,
        file_path: &str,
    ) -> io::Result<ApexResult> {
        if !storage_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Table '{}' does not exist", table_name),
            ));
        }
        let storage = TableStorageBackend::open(storage_path)?;
        let batch = storage.read_columns_to_arrow(None, 0, None)?;
        let schema = batch.schema();

        let file = std::fs::File::create(file_path).map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Cannot create parquet file '{}': {}", file_path, e),
            )
        })?;

        let props = parquet::file::properties::WriterProperties::builder().build();
        let mut writer =
            parquet::arrow::arrow_writer::ArrowWriter::try_new(file, schema.clone(), Some(props))
                .map_err(|e| {
                io::Error::new(io::ErrorKind::Other, format!("Parquet writer error: {}", e))
            })?;

        writer.write(&batch).map_err(|e| {
            io::Error::new(io::ErrorKind::Other, format!("Parquet write error: {}", e))
        })?;
        writer.close().map_err(|e| {
            io::Error::new(io::ErrorKind::Other, format!("Parquet close error: {}", e))
        })?;

        Ok(ApexResult::Scalar(batch.num_rows() as i64))
    }

    fn execute_copy_export(
        storage_path: &Path,
        table_name: &str,
        file_path: &str,
        format: &str,
        options: &[(String, String)],
    ) -> io::Result<ApexResult> {
        use std::io::Write;

        if format.eq_ignore_ascii_case("PARQUET") {
            return Self::execute_copy_to_parquet(storage_path, table_name, file_path);
        }
        if !storage_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Table '{}' does not exist", table_name),
            ));
        }

        let storage = TableStorageBackend::open(storage_path)?;
        let batch = storage.read_columns_to_arrow(None, 0, None)?;
        let schema = batch.schema();

        match format.to_uppercase().as_str() {
            "CSV" | "TSV" => {
                let delimiter = options
                    .iter()
                    .find(|(k, _)| k == "delimiter" || k == "delim" || k == "sep")
                    .and_then(|(_, v)| v.chars().next())
                    .unwrap_or(if format.eq_ignore_ascii_case("TSV") { '\t' } else { ',' });
                let header = options
                    .iter()
                    .find(|(k, _)| k == "header")
                    .map(|(_, v)| !matches!(v.to_lowercase().as_str(), "false" | "0"))
                    .unwrap_or(true);
                let file = std::fs::File::create(file_path)?;
                let mut writer = std::io::BufWriter::new(file);
                if header {
                    let columns: Vec<String> =
                        schema.fields().iter().map(|field| field.name().clone()).collect();
                    writeln!(writer, "{}", columns.join(&delimiter.to_string()))?;
                }
                for row in 0..batch.num_rows() {
                    let mut cells = Vec::with_capacity(batch.num_columns());
                    for col in 0..batch.num_columns() {
                        let value = Self::arrow_value_at_col(batch.column(col), row);
                        let mut cell = value.to_string();
                        if cell.contains(delimiter)
                            || cell.contains('"')
                            || cell.contains('\n')
                            || cell.contains('\r')
                        {
                            cell = format!("\"{}\"", cell.replace('"', "\"\""));
                        }
                        cells.push(cell);
                    }
                    writeln!(writer, "{}", cells.join(&delimiter.to_string()))?;
                }
                writer.flush()?;
                Ok(ApexResult::Scalar(batch.num_rows() as i64))
            }
            "JSON" | "NDJSON" | "JSONL" => {
                let file = std::fs::File::create(file_path)?;
                let mut writer = std::io::BufWriter::new(file);
                for row in 0..batch.num_rows() {
                    let mut obj = serde_json::Map::with_capacity(batch.num_columns());
                    for (col_idx, field) in schema.fields().iter().enumerate() {
                        let value = Self::arrow_value_at_col(batch.column(col_idx), row);
                        obj.insert(field.name().clone(), value.to_json_value());
                    }
                    writeln!(writer, "{}", serde_json::Value::Object(obj))?;
                }
                writer.flush()?;
                Ok(ApexResult::Scalar(batch.num_rows() as i64))
            }
            other => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("Unsupported COPY TO format: {}", other),
            )),
        }
    }

    /// Execute COPY table FROM 'file.parquet' — import data from Parquet file into table
    fn execute_copy_from_parquet(
        storage_path: &Path,
        table_name: &str,
        file_path: &str,
        base_dir: &Path,
        default_table_path: &Path,
    ) -> io::Result<ApexResult> {
        let file = std::fs::File::open(file_path).map_err(|e| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Cannot open parquet file '{}': {}", file_path, e),
            )
        })?;

        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| {
                io::Error::new(io::ErrorKind::Other, format!("Parquet reader error: {}", e))
            })?
            .build()
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("Parquet reader build error: {}", e),
                )
            })?;

        let mut total_rows = 0i64;
        for batch_result in reader {
            let batch = batch_result.map_err(|e| {
                io::Error::new(io::ErrorKind::Other, format!("Parquet read error: {}", e))
            })?;
            let schema = batch.schema();
            let num_rows = batch.num_rows();
            if num_rows == 0 {
                continue;
            }

            // Convert RecordBatch rows to Value vectors for insert
            let col_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
            let mut values: Vec<Vec<Value>> = Vec::with_capacity(num_rows);
            for row_idx in 0..num_rows {
                let mut row: Vec<Value> = Vec::with_capacity(col_names.len());
                for col_idx in 0..col_names.len() {
                    let col = batch.column(col_idx);
                    row.push(Self::arrow_value_at_col(col, row_idx));
                }
                values.push(row);
            }

            // Ensure table exists — create if not
            if !storage_path.exists() {
                let mut col_defs = Vec::new();
                for field in schema.fields() {
                    let type_str = match field.data_type() {
                        arrow::datatypes::DataType::Int64 => "INTEGER",
                        arrow::datatypes::DataType::Float64 => "REAL",
                        arrow::datatypes::DataType::Boolean => "BOOLEAN",
                        arrow::datatypes::DataType::UInt64 => "INTEGER",
                        _ => "TEXT",
                    };
                    col_defs.push(format!("{} {}", field.name(), type_str));
                }
                let create_sql = format!("CREATE TABLE {} ({})", table_name, col_defs.join(", "));
                let create_stmt = SqlParser::parse(&create_sql).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Failed to parse CREATE TABLE: {}", e),
                    )
                })?;
                Self::execute_parsed_multi(create_stmt, base_dir, default_table_path)?;
            }

            Self::execute_insert(storage_path, Some(&col_names), &values)?;
            total_rows += num_rows as i64;
        }

        Ok(ApexResult::Scalar(total_rows))
    }

    /// Execute ANALYZE — collect column statistics (NDV, min, max, null_count, row_count)
    fn execute_analyze(storage_path: &Path, table_name: &str) -> io::Result<ApexResult> {
        if !storage_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Table '{}' does not exist", table_name),
            ));
        }
        let storage = TableStorageBackend::open(storage_path)?;
        let batch = storage.read_columns_to_arrow(None, 0, None)?;
        let schema = batch.schema();
        let num_rows = batch.num_rows();

        let mut col_names: Vec<String> = Vec::new();
        let mut col_types: Vec<String> = Vec::new();
        let mut ndv_vals: Vec<i64> = Vec::new();
        let mut null_counts: Vec<i64> = Vec::new();
        let mut min_vals: Vec<String> = Vec::new();
        let mut max_vals: Vec<String> = Vec::new();
        let mut row_counts: Vec<i64> = Vec::new();
        let mut col_stats_map: std::collections::HashMap<
            String,
            crate::query::planner::ColumnStats,
        > = std::collections::HashMap::new();

        for (col_idx, field) in schema.fields().iter().enumerate() {
            let col = batch.column(col_idx);
            col_names.push(field.name().clone());
            col_types.push(format!("{}", field.data_type()));
            row_counts.push(num_rows as i64);

            // Count nulls
            let nc = (0..num_rows).filter(|&i| col.is_null(i)).count() as i64;
            null_counts.push(nc);

            // Collect distinct values and min/max
            let mut distinct = std::collections::HashSet::new();
            let mut min_s = String::new();
            let mut max_s = String::new();
            let mut first = true;

            for i in 0..num_rows {
                if col.is_null(i) {
                    continue;
                }
                let val_str = Self::arrow_value_at_col(col, i).to_string();
                distinct.insert(val_str.clone());
                if first {
                    min_s = val_str.clone();
                    max_s = val_str;
                    first = false;
                } else {
                    if val_str < min_s {
                        min_s = val_str.clone();
                    }
                    if val_str > max_s {
                        max_s = val_str;
                    }
                }
            }
            ndv_vals.push(distinct.len() as i64);
            min_vals.push(min_s.clone());
            max_vals.push(max_s.clone());

            // Accumulate stats for CBO cache
            col_stats_map.insert(
                field.name().clone(),
                crate::query::planner::ColumnStats {
                    ndv: distinct.len() as u64,
                    null_count: nc as u64,
                    min_value: min_s,
                    max_value: max_s,
                },
            );
        }

        // Store stats in CBO cache for cost-based optimization
        let table_stats = crate::query::planner::TableStats {
            row_count: num_rows as u64,
            columns: col_stats_map,
            collected_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };
        crate::query::planner::store_table_stats(&storage_path.to_string_lossy(), table_stats);

        // Build result as a RecordBatch
        let result_schema = Arc::new(Schema::new(vec![
            Field::new("column_name", arrow::datatypes::DataType::Utf8, false),
            Field::new("column_type", arrow::datatypes::DataType::Utf8, false),
            Field::new("row_count", arrow::datatypes::DataType::Int64, false),
            Field::new("null_count", arrow::datatypes::DataType::Int64, false),
            Field::new("ndv", arrow::datatypes::DataType::Int64, false),
            Field::new("min_value", arrow::datatypes::DataType::Utf8, true),
            Field::new("max_value", arrow::datatypes::DataType::Utf8, true),
        ]));
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(
                col_names.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                col_types.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(row_counts)),
            Arc::new(Int64Array::from(null_counts)),
            Arc::new(Int64Array::from(ndv_vals)),
            Arc::new(StringArray::from(
                min_vals.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                max_vals.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
        ];
        let result_batch = RecordBatch::try_new(result_schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        Ok(ApexResult::Data(result_batch))
    }

    /// Execute REINDEX — rebuild all indexes on a table from scratch
    fn execute_reindex(
        base_dir: &Path,
        default_table_path: &Path,
        table: &str,
    ) -> io::Result<ApexResult> {
        let table_path = Self::resolve_table_path(table, base_dir, default_table_path);
        if !table_path.exists() {
            return Err(err_not_found(format!("Table '{}' does not exist", table)));
        }

        // REINDEX must rebuild from committed values, including txn append
        // sidecars and DeltaStore cell updates.
        Self::materialize_table_sidecars(&table_path)?;

        let idx_mgr_arc = get_index_manager(base_dir, table);
        let mut idx_mgr = idx_mgr_arc.lock();
        let indexes = idx_mgr
            .list_indexes()
            .iter()
            .map(|m| (m.name.clone(), m.column_name.clone()))
            .collect::<Vec<_>>();

        if indexes.is_empty() {
            return Ok(ApexResult::Scalar(0));
        }

        // Clear all index data
        idx_mgr.rebuild_all();

        // Re-read table data and rebuild
        let storage = TableStorageBackend::open(&table_path)?;
        let row_count = storage.row_count();
        if row_count > 0 {
            let mut col_names: Vec<String> = vec!["_id".to_string()];
            for (_, col) in &indexes {
                if !col_names.contains(col) {
                    col_names.push(col.clone());
                }
            }
            let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
            let batch = storage.read_columns_to_arrow(Some(&col_refs), 0, None)?;

            let id_col = batch
                .column_by_name("_id")
                .ok_or_else(|| err_data("_id column not found"))?;
            let id_arr = id_col.as_any().downcast_ref::<UInt64Array>();
            let id_arr_i64 = id_col.as_any().downcast_ref::<Int64Array>();

            for row in 0..batch.num_rows() {
                let row_id: u64 = if let Some(arr) = id_arr {
                    arr.value(row)
                } else if let Some(arr) = id_arr_i64 {
                    arr.value(row) as u64
                } else {
                    row as u64
                };

                for (_, idx_col) in &indexes {
                    if let Some(data_col) = batch.column_by_name(idx_col) {
                        let value = Self::arrow_value_at_col(data_col, row);
                        let mut col_vals = std::collections::HashMap::new();
                        col_vals.insert(idx_col.clone(), value);
                        let _ = idx_mgr.on_insert(row_id, &col_vals);
                    }
                }
            }
        }

        idx_mgr.save()?;
        Ok(ApexResult::Scalar(indexes.len() as i64))
    }

    /// Execute PRAGMA commands
    /// Supported: integrity_check, table_info(table), version, stats
    fn execute_pragma(
        base_dir: &Path,
        default_table_path: &Path,
        name: &str,
        arg: Option<&str>,
    ) -> io::Result<ApexResult> {
        match name.to_lowercase().as_str() {
            "integrity_check" => {
                let table_name = arg.unwrap_or("default");
                let table_path = Self::resolve_table_path(table_name, base_dir, default_table_path);
                Self::pragma_integrity_check(&table_path, table_name)
            }
            "table_info" => {
                let table_name = arg
                    .ok_or_else(|| err_input("PRAGMA table_info requires a table name argument"))?;
                let table_path = Self::resolve_table_path(table_name, base_dir, default_table_path);
                Self::pragma_table_info(&table_path, table_name)
            }
            "version" => {
                let schema = Arc::new(Schema::new(vec![Field::new(
                    "version",
                    arrow::datatypes::DataType::Utf8,
                    false,
                )]));
                let arrays: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec!["ApexBase 1.0"]))];
                let batch = RecordBatch::try_new(schema, arrays)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                Ok(ApexResult::Data(batch))
            }
            "stats" => {
                let table_name = arg.unwrap_or("default");
                let table_path = Self::resolve_table_path(table_name, base_dir, default_table_path);
                let key = table_path.to_string_lossy().to_string();
                if let Some(stats) = crate::query::planner::get_table_stats(&key) {
                    let schema = Arc::new(Schema::new(vec![
                        Field::new("table", arrow::datatypes::DataType::Utf8, false),
                        Field::new("row_count", arrow::datatypes::DataType::Int64, false),
                        Field::new("columns", arrow::datatypes::DataType::Int64, false),
                        Field::new("collected_at", arrow::datatypes::DataType::Int64, false),
                    ]));
                    let arrays: Vec<ArrayRef> = vec![
                        Arc::new(StringArray::from(vec![table_name])),
                        Arc::new(Int64Array::from(vec![stats.row_count as i64])),
                        Arc::new(Int64Array::from(vec![stats.columns.len() as i64])),
                        Arc::new(Int64Array::from(vec![stats.collected_at as i64])),
                    ];
                    let batch = RecordBatch::try_new(schema, arrays)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                    Ok(ApexResult::Data(batch))
                } else {
                    Ok(ApexResult::Scalar(0))
                }
            }
            _ => Err(err_input(format!(
                "Unknown PRAGMA: {}. Supported: integrity_check, table_info, version, stats",
                name
            ))),
        }
    }

    /// PRAGMA integrity_check — verify file structure, header, schema, CRC
    fn pragma_integrity_check(table_path: &Path, table_name: &str) -> io::Result<ApexResult> {
        let mut checks: Vec<String> = Vec::new();
        let mut statuses: Vec<String> = Vec::new();

        // Check 1: File exists
        if !table_path.exists() {
            checks.push("file_exists".to_string());
            statuses.push(format!("FAIL: Table '{}' file not found", table_name));
            return Self::integrity_result(&checks, &statuses);
        }
        checks.push("file_exists".to_string());
        statuses.push("ok".to_string());

        // Check 2: File is readable and has valid header
        let file_meta = std::fs::metadata(table_path)?;
        let file_size = file_meta.len();
        if file_size < 128 {
            checks.push("header_valid".to_string());
            statuses.push(format!(
                "FAIL: File too small ({} bytes, minimum 128)",
                file_size
            ));
            return Self::integrity_result(&checks, &statuses);
        }
        checks.push("header_valid".to_string());
        statuses.push("ok".to_string());

        // Check 3: Can open storage
        match TableStorageBackend::open(table_path) {
            Ok(storage) => {
                checks.push("storage_open".to_string());
                statuses.push("ok".to_string());

                // Check 4: Schema readable
                let schema = storage.get_schema();
                checks.push("schema_valid".to_string());
                statuses.push(format!("ok ({} columns)", schema.len()));

                // Check 5: Row count consistent
                let row_count = storage.row_count();
                checks.push("row_count".to_string());
                statuses.push(format!("ok ({} rows)", row_count));

                // Check 6: Can read data
                match storage.read_columns_to_arrow(None, 0, Some(1)) {
                    Ok(batch) => {
                        checks.push("data_readable".to_string());
                        statuses.push(format!("ok ({} columns in batch)", batch.num_columns()));
                    }
                    Err(e) => {
                        checks.push("data_readable".to_string());
                        statuses.push(format!("FAIL: {}", e));
                    }
                }

                // Check 7: WAL file (if exists)
                let wal_path = table_path.with_extension("apex.wal");
                if wal_path.exists() {
                    match crate::storage::incremental::WalReader::open(&wal_path) {
                        Ok(mut reader) => match reader.read_all() {
                            Ok(records) => {
                                checks.push("wal_valid".to_string());
                                statuses.push(format!("ok ({} records)", records.len()));
                            }
                            Err(e) => {
                                checks.push("wal_valid".to_string());
                                statuses.push(format!("WARN: WAL read error: {}", e));
                            }
                        },
                        Err(e) => {
                            checks.push("wal_valid".to_string());
                            statuses.push(format!("WARN: WAL open error: {}", e));
                        }
                    }
                } else {
                    checks.push("wal_valid".to_string());
                    statuses.push("ok (no WAL file)".to_string());
                }

                // Check 8: Index files (if any)
                let (bd, tn) = base_dir_and_table(table_path);
                let idx_mgr_arc = get_index_manager(&bd, &tn);
                let idx_mgr = idx_mgr_arc.lock();
                let idx_list = idx_mgr.list_indexes();
                checks.push("indexes".to_string());
                statuses.push(format!("ok ({} indexes)", idx_list.len()));
            }
            Err(e) => {
                checks.push("storage_open".to_string());
                statuses.push(format!("FAIL: {}", e));
            }
        }

        Self::integrity_result(&checks, &statuses)
    }

    /// Build integrity check result as RecordBatch
    fn integrity_result(checks: &[String], statuses: &[String]) -> io::Result<ApexResult> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("check", arrow::datatypes::DataType::Utf8, false),
            Field::new("status", arrow::datatypes::DataType::Utf8, false),
        ]));
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(
                checks.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                statuses.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
        ];
        let batch = RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        Ok(ApexResult::Data(batch))
    }

    /// PRAGMA table_info — show table schema
    fn pragma_table_info(table_path: &Path, table_name: &str) -> io::Result<ApexResult> {
        if !table_path.exists() {
            return Err(err_not_found(format!(
                "Table '{}' does not exist",
                table_name
            )));
        }
        let storage = TableStorageBackend::open(table_path)?;
        let schema = storage.get_schema();

        let mut cids: Vec<i64> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        let mut types: Vec<String> = Vec::new();

        for (idx, (col_name, col_type)) in schema.iter().enumerate() {
            cids.push(idx as i64);
            names.push(col_name.clone());
            types.push(format!("{:?}", col_type));
        }

        let result_schema = Arc::new(Schema::new(vec![
            Field::new("cid", arrow::datatypes::DataType::Int64, false),
            Field::new("name", arrow::datatypes::DataType::Utf8, false),
            Field::new("type", arrow::datatypes::DataType::Utf8, false),
        ]));
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(cids)),
            Arc::new(StringArray::from(
                names.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                types.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
        ];
        let batch = RecordBatch::try_new(result_schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        Ok(ApexResult::Data(batch))
    }

    /// Execute INSERT ON CONFLICT (UPSERT)
    /// - DO NOTHING: skip rows that conflict on the specified columns
    /// - DO UPDATE SET ...: update conflicting rows with the specified assignments
    fn execute_insert_on_conflict(
        storage_path: &Path,
        columns: Option<&[String]>,
        values: &[Vec<Value>],
        conflict_columns: &[String],
        do_update: Option<&[(String, SqlExpr)]>,
    ) -> io::Result<ApexResult> {
        use std::collections::HashMap;

        if !storage_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Table does not exist",
            ));
        }

        invalidate_storage_cache(storage_path);
        let storage = TableStorageBackend::open_for_write(storage_path)?;
        let schema = storage.get_schema();

        let col_names: Vec<String> = if let Some(cols) = columns {
            cols.to_vec()
        } else {
            schema.iter().map(|(n, _)| n.clone()).collect()
        };

        // Read existing data for conflict detection
        let conflict_refs: Vec<&str> = conflict_columns.iter().map(|s| s.as_str()).collect();
        let mut read_cols: Vec<&str> = vec!["_id"];
        for c in &conflict_refs {
            if !read_cols.contains(c) {
                read_cols.push(c);
            }
        }
        let existing = storage.read_columns_to_arrow(Some(&read_cols), 0, None)?;

        // Build lookup: conflict key → row index in existing data
        let existing_rows = existing.num_rows();

        // Helper: extract value from Arrow column at row
        let extract_val = |col_name: &str, row: usize| -> Option<Value> {
            existing
                .column_by_name(col_name)
                .map(|c| Self::arrow_value_at_col(c, row))
        };

        // For each input row, check if it conflicts
        let mut inserted = 0i64;
        let mut updated = 0i64;

        for row_values in values {
            // Build this row's conflict key values
            let mut conflict_vals: Vec<Value> = Vec::new();
            for cc in conflict_columns {
                if let Some(idx) = col_names.iter().position(|n| n == cc) {
                    if idx < row_values.len() {
                        conflict_vals.push(row_values[idx].clone());
                    } else {
                        conflict_vals.push(Value::Null);
                    }
                } else {
                    conflict_vals.push(Value::Null);
                }
            }

            // Search for a matching existing row
            let mut conflict_row_id: Option<u64> = None;
            for er in 0..existing_rows {
                let mut all_match = true;
                for (ci, cc) in conflict_columns.iter().enumerate() {
                    let existing_val = extract_val(cc, er);
                    let new_val = &conflict_vals[ci];
                    let matches = match (&existing_val, new_val) {
                        (Some(Value::Int64(a)), Value::Int64(b)) => a == b,
                        (Some(Value::Int32(a)), Value::Int32(b)) => a == b,
                        (Some(Value::Int64(a)), Value::Int32(b)) => *a == *b as i64,
                        (Some(Value::Float64(a)), Value::Float64(b)) => {
                            (a - b).abs() < f64::EPSILON
                        }
                        (Some(Value::String(a)), Value::String(b)) => a == b,
                        (Some(Value::Bool(a)), Value::Bool(b)) => a == b,
                        (Some(Value::UInt64(a)), Value::Int64(b)) => *a as i64 == *b,
                        (Some(Value::Int64(a)), Value::UInt64(b)) => *a == *b as i64,
                        (Some(Value::Null), Value::Null) => false, // NULL != NULL
                        _ => false,
                    };
                    if !matches {
                        all_match = false;
                        break;
                    }
                }
                if all_match {
                    // Found conflict — get row_id
                    if let Some(id_col) = existing.column_by_name("_id") {
                        if let Some(a) = id_col.as_any().downcast_ref::<UInt64Array>() {
                            conflict_row_id = Some(a.value(er));
                        } else if let Some(a) = id_col.as_any().downcast_ref::<Int64Array>() {
                            conflict_row_id = Some(a.value(er) as u64);
                        }
                    }
                    break;
                }
            }

            if let Some(rid) = conflict_row_id {
                // Conflict found
                if let Some(assignments) = do_update {
                    // DO UPDATE SET ...
                    // Build a row data map for expression evaluation
                    let mut row_data: HashMap<String, Value> = HashMap::new();
                    for (i, val) in row_values.iter().enumerate() {
                        if i < col_names.len() {
                            row_data.insert(col_names[i].clone(), val.clone());
                        }
                    }

                    let mut update_assignments: Vec<(String, SqlExpr)> = Vec::new();
                    for (col, expr) in assignments {
                        // Support EXCLUDED.col syntax: replace with the new row's value
                        let resolved_expr = match expr {
                            SqlExpr::Column(ref name)
                                if name.starts_with("EXCLUDED.")
                                    || name.starts_with("excluded.") =>
                            {
                                let src_col = &name[9..]; // skip "EXCLUDED." or "excluded."
                                if let Some(val) = row_data.get(src_col) {
                                    SqlExpr::Literal(val.clone())
                                } else {
                                    expr.clone()
                                }
                            }
                            _ => expr.clone(),
                        };
                        update_assignments.push((col.clone(), resolved_expr));
                    }

                    // Execute UPDATE for this specific row
                    let where_clause = SqlExpr::BinaryOp {
                        left: Box::new(SqlExpr::Column("_id".to_string())),
                        op: BinaryOperator::Eq,
                        right: Box::new(SqlExpr::Literal(Value::UInt64(rid))),
                    };
                    Self::execute_update(storage_path, &update_assignments, Some(&where_clause))?;
                    updated += 1;
                }
                // else DO NOTHING — skip this row
            } else {
                // No conflict — insert the row
                Self::execute_insert(storage_path, Some(&col_names), &[row_values.clone()])?;
                inserted += 1;
            }
        }

        invalidate_storage_cache(storage_path);
        invalidate_table_stats(&storage_path.to_string_lossy());
        Ok(ApexResult::Scalar(inserted + updated))
    }

    /// Execute DELETE statement (soft delete - marks rows as deleted without physical removal)
    fn execute_delete(
        storage_path: &Path,
        where_clause: Option<&SqlExpr>,
    ) -> io::Result<ApexResult> {
        if !storage_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Table does not exist",
            ));
        }

        // Invalidate cache before write
        invalidate_storage_cache(storage_path);

        // Collect indexed column names for this table (for index maintenance)
        // Use effective_columns() to include ALL columns of composite indexes,
        // not just the first column (column_name), so composite_key() can find the full key.
        let indexed_cols: Vec<String> = {
            let (bd, tn) = base_dir_and_table(storage_path);
            let mgr = get_index_manager(&bd, &tn);
            let lock = mgr.lock();
            let mut cols: Vec<String> = lock
                .list_indexes()
                .iter()
                .flat_map(|m| m.effective_columns().into_iter().map(|s| s.to_string()))
                .collect();
            cols.sort();
            cols.dedup();
            cols
        };

        // FOREIGN KEY enforcement: check if any child tables reference this table
        // We need to scan sibling .apex files for FK constraints pointing here
        let (base_dir, this_table_name) = base_dir_and_table(storage_path);
        let fk_children: Vec<(String, String, String)> = {
            // (child_table_path, child_col, ref_col)
            let mut children = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&base_dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().map(|e| e == "apex").unwrap_or(false) && p != storage_path {
                        if let Ok(child_storage) = TableStorageBackend::open(&p) {
                            let child_schema = child_storage.get_schema();
                            for (col_name, _) in &child_schema {
                                let cons = child_storage.storage.get_column_constraints(col_name);
                                if let Some((ref rt, ref rc)) = cons.foreign_key {
                                    if rt == &this_table_name {
                                        children.push((
                                            p.to_string_lossy().to_string(),
                                            col_name.clone(),
                                            rc.clone(),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            children
        };

        // For DELETE without WHERE, delete all rows (soft delete)
        if where_clause.is_none() {
            // FK enforcement: if deleting ALL rows, check child tables have no referencing rows
            if !fk_children.is_empty() {
                for (child_path, child_col, _ref_col) in &fk_children {
                    let child_storage =
                        TableStorageBackend::open(std::path::Path::new(child_path))?;
                    let child_batch = child_storage.read_columns_to_arrow(
                        Some(&[child_col.as_str()]),
                        0,
                        None,
                    )?;
                    if let Some(col_arr) = child_batch.column_by_name(child_col) {
                        // Check if any non-null FK values exist in child
                        for r in 0..col_arr.len() {
                            if !col_arr.is_null(r) {
                                let child_table = std::path::Path::new(child_path)
                                    .file_stem()
                                    .map(|s| s.to_string_lossy().to_string())
                                    .unwrap_or_default();
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidInput,
                                    format!("FOREIGN KEY constraint violated: cannot delete from '{}' — referenced by '{}.{}'", this_table_name, child_table, child_col),
                                ));
                            }
                        }
                    }
                }
            }
            let storage = TableStorageBackend::open_for_delete(storage_path)?;
            let count = storage.active_row_count() as i64;

            // Read _id + indexed columns for index maintenance
            let mut read_cols: Vec<String> = vec!["_id".to_string()];
            read_cols.extend(indexed_cols.iter().cloned());
            read_cols.sort();
            read_cols.dedup();
            let col_refs: Vec<&str> = read_cols.iter().map(|s| s.as_str()).collect();
            let batch = storage.read_columns_to_arrow(Some(&col_refs), 0, None)?;

            let mut deleted_entries: Vec<(u64, std::collections::HashMap<String, Value>)> =
                Vec::new();
            let mut all_rids: Vec<u64> = Vec::new();
            if let Some(id_col) = batch.column_by_name("_id") {
                if let Some(id_arr) = id_col.as_any().downcast_ref::<UInt64Array>() {
                    for i in 0..id_arr.len() {
                        let rid = id_arr.value(i);
                        let mut vals = std::collections::HashMap::new();
                        for cn in &indexed_cols {
                            if let Some(c) = batch.column_by_name(cn) {
                                vals.insert(cn.clone(), Self::arrow_value_at_col(c, i));
                            }
                        }
                        deleted_entries.push((rid, vals));
                        all_rids.push(rid);
                    }
                } else if let Some(id_arr) = id_col.as_any().downcast_ref::<Int64Array>() {
                    for i in 0..id_arr.len() {
                        let rid = id_arr.value(i) as u64;
                        let mut vals = std::collections::HashMap::new();
                        for cn in &indexed_cols {
                            if let Some(c) = batch.column_by_name(cn) {
                                vals.insert(cn.clone(), Self::arrow_value_at_col(c, i));
                            }
                        }
                        deleted_entries.push((rid, vals));
                        all_rids.push(rid);
                    }
                }
            }
            storage.delete_batch(&all_rids);
            storage.save_delete_only()?;
            Self::notify_index_delete(storage_path, &deleted_entries);
            Self::notify_fts_delete(storage_path, &deleted_entries);
            invalidate_storage_cache(storage_path);
            crate::storage::engine::engine().invalidate(storage_path);
            invalidate_table_stats(&storage_path.to_string_lossy());
            return Ok(ApexResult::Scalar(count));
        }

        // Soft delete: use mmap-only open (no full column load) + in-place deletion vector update
        let storage = TableStorageBackend::open_for_delete(storage_path)?;

        // ── Fast scan path: simple numeric/string predicate, no FK, no indexes ──
        // Numeric: delete_where_numeric_range_inplace — single pass, no id_to_idx HashMap.
        // String:  scan_string_filter_mmap → get_ids → delete_batch + save_delete_only.
        if fk_children.is_empty()
            && indexed_cols.is_empty()
            && storage.pending_v4_in_memory_rows() == 0
        {
            if let Some((col, low, high)) =
                Self::extract_numeric_range_from_where(where_clause.unwrap())
            {
                if let Some(deleted) =
                    storage.delete_where_numeric_range_inplace(&col, low, high)?
                {
                    if deleted > 0 {
                        invalidate_storage_cache(storage_path);
                        crate::storage::engine::engine().invalidate(storage_path);
                        invalidate_table_stats(&storage_path.to_string_lossy());
                    }
                    return Ok(ApexResult::Scalar(deleted));
                }
                // Inplace path unavailable (non-PLAIN encoding) — scan for IDs then use
                // ID-based inplace delete (binary search, no 1M-entry id_to_idx HashMap build)
                if let Some(all_rids) = storage.scan_numeric_range_mmap_with_ids(&col, low, high)? {
                    let deleted = all_rids.len() as i64;
                    if deleted > 0 {
                        if storage.delete_ids_inplace_v4(&all_rids)?.is_none() {
                            // Compressed RGs: full in-memory fallback
                            storage.delete_batch(&all_rids);
                            storage.save_delete_only()?;
                        }
                        invalidate_storage_cache(storage_path);
                        crate::storage::engine::engine().invalidate(storage_path);
                        invalidate_table_stats(&storage_path.to_string_lossy());
                    }
                    return Ok(ApexResult::Scalar(deleted));
                }
            } else if let Some((col, val)) = Self::extract_string_equality(where_clause.unwrap()) {
                if let Some(indices) = storage.scan_string_filter_mmap(&col, &val, None)? {
                    let all_rids = storage.get_ids_for_global_indices_mmap(&indices)?;
                    let deleted = all_rids.len() as i64;
                    if deleted > 0 {
                        if storage.delete_ids_inplace_v4(&all_rids)?.is_none() {
                            storage.delete_batch(&all_rids);
                            storage.save_delete_only()?;
                        }
                        invalidate_storage_cache(storage_path);
                        crate::storage::engine::engine().invalidate(storage_path);
                        invalidate_table_stats(&storage_path.to_string_lossy());
                    }
                    return Ok(ApexResult::Scalar(deleted));
                }
            }
        }
        // ── End fast scan path ──

        // Extract column names from WHERE clause + indexed columns + FK-referenced columns
        let mut where_cols: Vec<String> = Vec::new();
        Self::collect_columns_from_expr(where_clause.unwrap(), &mut where_cols);
        where_cols.extend(indexed_cols.iter().cloned());
        // Include columns referenced by child FKs so we can check them
        for (_, _, ref_col) in &fk_children {
            if !where_cols.contains(ref_col) {
                where_cols.push(ref_col.clone());
            }
        }
        where_cols.sort();
        where_cols.dedup();

        // Always include _id for deletion
        if !where_cols.iter().any(|c| c == "_id") {
            where_cols.push("_id".to_string());
        }

        let col_refs: Vec<&str> = where_cols.iter().map(|s| s.as_str()).collect();
        let batch = storage.read_columns_to_arrow(Some(&col_refs), 0, None)?;
        let filter_mask = Self::evaluate_predicate(&batch, where_clause.unwrap())?;

        // FK enforcement: for rows being deleted, check no child references exist
        if !fk_children.is_empty() {
            for (child_path, child_col, ref_col) in &fk_children {
                // Get the parent column values being deleted
                let parent_arr = batch.column_by_name(ref_col);
                if parent_arr.is_none() {
                    continue;
                }
                let parent_arr = parent_arr.unwrap();

                // Load child FK column
                let child_storage = TableStorageBackend::open(std::path::Path::new(child_path))?;
                let child_batch =
                    child_storage.read_columns_to_arrow(Some(&[child_col.as_str()]), 0, None)?;
                let child_arr = child_batch.column_by_name(child_col);
                if child_arr.is_none() {
                    continue;
                }
                let child_arr = child_arr.unwrap();

                for i in 0..filter_mask.len() {
                    if !filter_mask.value(i) || parent_arr.is_null(i) {
                        continue;
                    }
                    let parent_val = Self::arrow_value_at_col(parent_arr, i);
                    // Check if any child row references this value
                    for cr in 0..child_arr.len() {
                        if child_arr.is_null(cr) {
                            continue;
                        }
                        let child_val = Self::arrow_value_at_col(child_arr, cr);
                        if parent_val == child_val {
                            let child_table = std::path::Path::new(child_path)
                                .file_stem()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or_default();
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                format!("FOREIGN KEY constraint violated: cannot delete from '{}' — value referenced by '{}.{}'", this_table_name, child_table, child_col),
                            ));
                        }
                    }
                }
            }
        }

        // Collect matching row IDs and index values for batch deletion
        let mut deleted = 0i64;
        let mut deleted_entries: Vec<(u64, std::collections::HashMap<String, Value>)> = Vec::new();
        let mut all_rids: Vec<u64> = Vec::new();
        if let Some(id_col) = batch.column_by_name("_id") {
            for i in 0..filter_mask.len() {
                if filter_mask.value(i) {
                    let rid = if let Some(id_arr) = id_col.as_any().downcast_ref::<UInt64Array>() {
                        Some(id_arr.value(i))
                    } else if let Some(id_arr) = id_col.as_any().downcast_ref::<Int64Array>() {
                        Some(id_arr.value(i) as u64)
                    } else {
                        None
                    };
                    if let Some(rid) = rid {
                        let mut vals = std::collections::HashMap::new();
                        for cn in &indexed_cols {
                            if let Some(c) = batch.column_by_name(cn) {
                                vals.insert(cn.clone(), Self::arrow_value_at_col(c, i));
                            }
                        }
                        deleted_entries.push((rid, vals));
                        all_rids.push(rid);
                        deleted += 1;
                    }
                }
            }
        }

        if deleted > 0 {
            // Fast path: no FK/index notifications — use mmap binary-search delete
            // (bypasses the id_to_idx HashMap build in delete_batch)
            let used_inplace = if fk_children.is_empty() && indexed_cols.is_empty() {
                storage.delete_ids_inplace_v4(&all_rids)?.is_some()
            } else {
                false
            };
            if !used_inplace {
                storage.delete_batch(&all_rids);
                storage.save_delete_only()?;
            }
            Self::notify_index_delete(storage_path, &deleted_entries);
            Self::notify_fts_delete(storage_path, &deleted_entries);
            invalidate_storage_cache(storage_path);
            crate::storage::engine::engine().invalidate(storage_path);
            invalidate_table_stats(&storage_path.to_string_lossy());
        }

        Ok(ApexResult::Scalar(deleted))
    }

    /// Execute UPDATE statement
    /// Uses DeltaStore for cell-level updates (no delete+insert, no full file rewrite)
    fn execute_update(
        storage_path: &Path,
        assignments: &[(String, SqlExpr)],
        where_clause: Option<&SqlExpr>,
    ) -> io::Result<ApexResult> {
        use std::collections::HashMap as StdHashMap;

        if !storage_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Table does not exist",
            ));
        }

        // Invalidate cache before write
        invalidate_storage_cache(storage_path);

        // Collect indexed column names for index maintenance
        // Use effective_columns() to include ALL columns of composite indexes.
        let indexed_cols: Vec<String> = {
            let (bd, tn) = base_dir_and_table(storage_path);
            let mgr = get_index_manager(&bd, &tn);
            let lock = mgr.lock();
            let mut cols: Vec<String> = lock
                .list_indexes()
                .iter()
                .flat_map(|m| m.effective_columns().into_iter().map(|s| s.to_string()))
                .collect();
            cols.sort();
            cols.dedup();
            cols
        };

        // Use open (mmap-only) — avoids loading ALL columns into memory.
        // DeltaStore updates are applied lazily at read time (DeltaMerger) and
        // merged into the base file via apply_pending_deltas_in_place() on next
        // open_for_write (e.g., DELETE), which is safe because of Fix B.
        let storage = TableStorageBackend::open(storage_path)?;

        // ── Mmap super-fast path ─────────────────────────────────────────────────
        // For: all-literal SET + simple numeric WHERE + no constraints + no indexes.
        // Uses scan_and_update_inplace: single pass per SET column, writes directly to
        // the base .apex file — no DeltaStore serialization, no Arrow batch, no bincode.
        if indexed_cols.is_empty()
            && !storage.storage.has_constraints()
            && storage.pending_v4_in_memory_rows() == 0
        {
            let all_lit = assignments
                .iter()
                .all(|(_, e)| matches!(e, SqlExpr::Literal(_)));
            if all_lit {
                if let Some((where_col, low, high)) =
                    where_clause.and_then(|w| Self::extract_numeric_range_from_where(w))
                {
                    // Safety: if WHERE column has delta-store overrides, skip fast path
                    let where_col_clean = {
                        let ds = storage.storage.delta_store();
                        !ds.all_updates()
                            .values()
                            .any(|m| m.contains_key(&where_col))
                    };
                    if where_col_clean {
                        if where_col == "_id"
                            && low.is_finite()
                            && high.is_finite()
                            && (low - high).abs() < f64::EPSILON
                            && low >= 0.0
                        {
                            let row_id = low as u64;
                            {
                                let delta = storage.storage.delta_store();
                                if delta.is_deleted(row_id) {
                                    return Ok(ApexResult::Scalar(0));
                                }
                            }
                            if let Some(row_exists) = storage.row_id_active_rcix(row_id)? {
                                if !row_exists {
                                    return Ok(ApexResult::Scalar(0));
                                }
                            }
                            if assignments.len() == 1 {
                                let (col_name, expr) = &assignments[0];
                                if col_name != "_id" {
                                    if let SqlExpr::Literal(v) = expr {
                                        let bytes_opt = match v {
                                            Value::Float64(f) => Some(f.to_le_bytes()),
                                            Value::Int64(i) => Some(i.to_le_bytes()),
                                            _ => None,
                                        };
                                        if let Some(bytes) = bytes_opt {
                                            if let Some((n, physically_written)) = storage
                                                .update_by_id_inplace(row_id, col_name, &bytes)?
                                            {
                                                if physically_written {
                                                    invalidate_storage_cache(storage_path);
                                                    invalidate_table_stats(
                                                        &storage_path.to_string_lossy(),
                                                    );
                                                }
                                                return Ok(ApexResult::Scalar(n));
                                            }
                                        }
                                    }
                                }
                            }

                            let mut delta_values: Vec<(u64, &str, Value)> =
                                Vec::with_capacity(assignments.len());
                            let mut can_delta_by_id = true;
                            for (col_name, expr) in assignments {
                                if col_name == "_id" {
                                    can_delta_by_id = false;
                                    break;
                                }
                                if let SqlExpr::Literal(v) = expr {
                                    delta_values.push((row_id, col_name.as_str(), v.clone()));
                                } else {
                                    can_delta_by_id = false;
                                    break;
                                }
                            }
                            if can_delta_by_id {
                                if let Some(row_exists) = storage.row_id_active_rcix(row_id)? {
                                    let updated = if row_exists { 1 } else { 0 };
                                    if updated > 0 {
                                        storage.delta_batch_update_rows(&delta_values);
                                        storage.save_delta_store()?;
                                    }
                                    invalidate_storage_cache(storage_path);
                                    invalidate_table_stats(&storage_path.to_string_lossy());
                                    return Ok(ApexResult::Scalar(updated));
                                }
                            }
                        }

                        // Try in-place write for each SET column
                        let mut all_inplace = true;
                        let mut inplace_count: i64 = 0;
                        for (col_name, expr) in assignments {
                            if let SqlExpr::Literal(v) = expr {
                                let bytes_opt = match v {
                                    Value::Float64(f) => Some(f.to_le_bytes()),
                                    Value::Int64(i) => Some(i.to_le_bytes()),
                                    _ => None,
                                };
                                if let Some(bytes) = bytes_opt {
                                    match storage.scan_and_update_inplace(
                                        &where_col, low, high, col_name, &bytes,
                                    )? {
                                        Some(n) => {
                                            inplace_count = n;
                                        }
                                        None => {
                                            all_inplace = false;
                                            break;
                                        }
                                    }
                                } else {
                                    all_inplace = false;
                                    break;
                                }
                            }
                        }
                        if all_inplace {
                            // In-place write succeeded — invalidate executor cache so next read rebuilds
                            invalidate_storage_cache(storage_path);
                            invalidate_table_stats(&storage_path.to_string_lossy());
                            return Ok(ApexResult::Scalar(inplace_count));
                        }
                    }
                }
            }
        }
        // ── End mmap super-fast path ─────────────────────────────────────────────

        // COLUMN PROJECTION: only read the columns we actually need.
        // For a simple UPDATE SET col=literal WHERE other_col=val, this is just
        // [_id, WHERE-col] instead of all columns — a significant I/O reduction.
        // Also, if DeltaStore has no entries for the projected columns (e.g. previous
        // UPDATEs only changed a different column), DeltaMerger is effectively a no-op.
        let has_constraints = storage.storage.has_constraints();
        let set_col_refs: Vec<String> = assignments
            .iter()
            .flat_map(|(_, expr)| Self::collect_column_refs(expr))
            .collect();
        let unique_pk_cols: Vec<String> = if has_constraints {
            let schema = storage.get_schema();
            schema
                .iter()
                .filter(|(name, _)| {
                    let c = storage.storage.get_column_constraints(name);
                    c.unique || c.primary_key
                })
                .map(|(name, _)| name.clone())
                .collect()
        } else {
            vec![]
        };
        let total_schema_cols = storage.get_schema().len() + 1; // +1 for _id
        let mut needed_cols: Vec<String> = vec!["_id".to_string()];
        if let Some(where_expr) = where_clause {
            for c in Self::collect_column_refs(where_expr) {
                if !needed_cols.iter().any(|x| x == &c) {
                    needed_cols.push(c);
                }
            }
        }
        for c in &set_col_refs {
            if !needed_cols.iter().any(|x| x == c) {
                needed_cols.push(c.clone());
            }
        }
        for c in &indexed_cols {
            if !needed_cols.iter().any(|x| x == c) {
                needed_cols.push(c.clone());
            }
        }
        for c in &unique_pk_cols {
            if !needed_cols.iter().any(|x| x == c) {
                needed_cols.push(c.clone());
            }
        }
        let batch = if needed_cols.len() < total_schema_cols {
            let refs: Vec<&str> = needed_cols.iter().map(|s| s.as_str()).collect();
            storage.read_columns_to_arrow(Some(&refs), 0, None)?
        } else {
            storage.read_columns_to_arrow(None, 0, None)?
        };

        let filter_mask = if let Some(where_expr) = where_clause {
            Self::evaluate_predicate(&batch, where_expr)?
        } else {
            BooleanArray::from(vec![true; batch.num_rows()])
        };

        // Collect row IDs and update data for matching rows.
        // Hoist column_by_name + downcast outside the hot loop to avoid O(N) repeated lookups.
        let mut updates: Vec<(u64, StdHashMap<String, Value>)> = Vec::new();
        let mut old_index_entries: Vec<(u64, StdHashMap<String, Value>)> = Vec::new();
        let mut new_index_entries: Vec<(u64, StdHashMap<String, Value>)> = Vec::new();

        // Pre-extract _id array once (avoids column_by_name + downcast inside hot loop)
        enum IdArray<'a> {
            U64(&'a UInt64Array),
            I64(&'a Int64Array),
        }
        let id_array: Option<IdArray<'_>> = batch.column_by_name("_id").and_then(|c| {
            if let Some(a) = c.as_any().downcast_ref::<UInt64Array>() {
                Some(IdArray::U64(a))
            } else {
                c.as_any().downcast_ref::<Int64Array>().map(IdArray::I64)
            }
        });
        let id_array = match id_array {
            Some(a) => a,
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "_id column not found",
                ))
            }
        };

        // Pre-evaluate assignments that are pure literals (SET col = literal) — avoids
        // per-row evaluate_expr_to_value calls for the common case.
        let all_literals = assignments
            .iter()
            .all(|(_, e)| matches!(e, SqlExpr::Literal(_)));
        let literal_update: Option<StdHashMap<String, Value>> = if all_literals {
            let mut m = StdHashMap::with_capacity(assignments.len());
            for (col_name, expr) in assignments {
                if let SqlExpr::Literal(v) = expr {
                    m.insert(col_name.clone(), v.clone());
                }
            }
            Some(m)
        } else {
            None
        };

        // Pre-extract index columns arrays to avoid repeated column_by_name in loop
        let indexed_col_arrays: Vec<(&str, &ArrayRef)> = if !indexed_cols.is_empty() {
            indexed_cols
                .iter()
                .filter_map(|cn| batch.column_by_name(cn).map(|c| (cn.as_str(), c)))
                .collect()
        } else {
            Vec::new()
        };

        // ── Fast path: all-literal SET + no constraints + no indexes ───────────
        // Avoids building per-row HashMap for each matched row.
        if let Some(ref lit) = literal_update {
            if indexed_col_arrays.is_empty() && !storage.storage.has_constraints() {
                // Collect matching row IDs (no HashMap allocation per row)
                let mut matched_ids: Vec<u64> = Vec::new();
                for i in 0..filter_mask.len() {
                    if filter_mask.value(i) {
                        let id = match &id_array {
                            IdArray::U64(a) => a.value(i),
                            IdArray::I64(a) => a.value(i) as u64,
                        };
                        matched_ids.push(id);
                    }
                }
                let updated = matched_ids.len() as i64;
                if updated > 0 {
                    // Build flat batch directly without per-row HashMap
                    let mut flat_batch: Vec<(u64, &str, Value)> =
                        Vec::with_capacity(matched_ids.len() * lit.len());
                    for id in &matched_ids {
                        for (col, val) in lit {
                            flat_batch.push((*id, col.as_str(), val.clone()));
                        }
                    }
                    storage.delta_batch_update_rows(&flat_batch);
                    storage.save_delta_store()?;
                }
                invalidate_storage_cache(storage_path);
                invalidate_table_stats(&storage_path.to_string_lossy());
                return Ok(ApexResult::Scalar(updated));
            }
        }
        // ── End fast path ───────────────────────────────────────────────────────

        for i in 0..filter_mask.len() {
            if !filter_mask.value(i) {
                continue;
            }

            let id = match &id_array {
                IdArray::U64(a) => a.value(i),
                IdArray::I64(a) => a.value(i) as u64,
            };

            // Evaluate assignment expressions to get new values
            let update_data: StdHashMap<String, Value> = if let Some(ref lit) = literal_update {
                lit.clone()
            } else {
                let mut m = StdHashMap::with_capacity(assignments.len());
                for (col_name, expr) in assignments {
                    let value = Self::evaluate_expr_to_value(&batch, expr, i)?;
                    m.insert(col_name.clone(), value);
                }
                m
            };

            // Index maintenance: collect old and new indexed values
            if !indexed_col_arrays.is_empty() {
                let mut old_vals = StdHashMap::with_capacity(indexed_col_arrays.len());
                let mut new_vals = StdHashMap::with_capacity(indexed_col_arrays.len());
                for (cn, c) in &indexed_col_arrays {
                    let old_val = Self::arrow_value_at_col(c, i);
                    let new_val = update_data
                        .get(*cn)
                        .cloned()
                        .unwrap_or_else(|| old_val.clone());
                    old_vals.insert(cn.to_string(), old_val);
                    new_vals.insert(cn.to_string(), new_val);
                }
                old_index_entries.push((id, old_vals));
                new_index_entries.push((id, new_vals));
            }

            updates.push((id, update_data));
        }

        let updated = updates.len() as i64;

        // Enforce constraints on updated values
        if updated > 0 && storage.storage.has_constraints() {
            for (_row_id, update_data) in &updates {
                for (col_name, value) in update_data {
                    let cons = storage.storage.get_column_constraints(col_name);
                    // NOT NULL check
                    if cons.not_null && matches!(value, Value::Null) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!(
                                "NOT NULL constraint violated: column '{}' cannot be NULL",
                                col_name
                            ),
                        ));
                    }
                }
            }

            // UNIQUE / PK check: ensure new values don't conflict with existing or other updates
            let schema = storage.get_schema();
            let unique_cols: Vec<String> = schema
                .iter()
                .filter(|(name, _)| {
                    let c = storage.storage.get_column_constraints(name);
                    c.unique || c.primary_key
                })
                .map(|(name, _)| name.clone())
                .collect();

            for uc in &unique_cols {
                let constraint_kind = if storage.storage.get_column_constraints(uc).primary_key {
                    "PRIMARY KEY"
                } else {
                    "UNIQUE"
                };

                // Collect new values being set for this unique column
                let updated_row_ids: std::collections::HashSet<u64> = updates
                    .iter()
                    .filter(|(_, data)| data.contains_key(uc))
                    .map(|(id, _)| *id)
                    .collect();
                let new_vals: Vec<&Value> = updates
                    .iter()
                    .filter_map(|(_, data)| data.get(uc))
                    .collect();

                if new_vals.is_empty() {
                    continue;
                }

                // Check duplicates among new values
                {
                    let mut seen = std::collections::HashSet::new();
                    for v in &new_vals {
                        if !matches!(v, Value::Null) {
                            let key = format!("{:?}", v);
                            if !seen.insert(key) {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidInput,
                                    format!(
                                        "{} constraint violated: duplicate value in column '{}'",
                                        constraint_kind, uc
                                    ),
                                ));
                            }
                        }
                    }
                }

                // Check against existing rows (excluding rows being updated)
                if let Some(existing_col) = batch.column_by_name(uc) {
                    use arrow::array::{
                        BooleanArray as BA, Float64Array as F64A, Int64Array as I64A,
                        StringArray as SA,
                    };
                    let id_col = batch.column_by_name("_id");
                    for new_val in &new_vals {
                        if matches!(new_val, Value::Null) {
                            continue;
                        }
                        for row in 0..existing_col.len() {
                            if existing_col.is_null(row) {
                                continue;
                            }
                            // Skip rows that are being updated (they'll have new values)
                            if let Some(idc) = &id_col {
                                let rid = idc
                                    .as_any()
                                    .downcast_ref::<UInt64Array>()
                                    .map(|a| a.value(row))
                                    .or_else(|| {
                                        idc.as_any()
                                            .downcast_ref::<Int64Array>()
                                            .map(|a| a.value(row) as u64)
                                    });
                                if let Some(rid) = rid {
                                    if updated_row_ids.contains(&rid) {
                                        continue;
                                    }
                                }
                            }
                            let matches = match new_val {
                                Value::Int64(v) => existing_col
                                    .as_any()
                                    .downcast_ref::<I64A>()
                                    .map(|a| a.value(row) == *v)
                                    .unwrap_or(false),
                                Value::String(v) => existing_col
                                    .as_any()
                                    .downcast_ref::<SA>()
                                    .map(|a| a.value(row) == v.as_str())
                                    .unwrap_or(false),
                                Value::Float64(v) => existing_col
                                    .as_any()
                                    .downcast_ref::<F64A>()
                                    .map(|a| (a.value(row) - v).abs() < f64::EPSILON)
                                    .unwrap_or(false),
                                Value::Bool(v) => existing_col
                                    .as_any()
                                    .downcast_ref::<BA>()
                                    .map(|a| a.value(row) == *v)
                                    .unwrap_or(false),
                                _ => false,
                            };
                            if matches {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidInput,
                                    format!(
                                        "{} constraint violated: duplicate value in column '{}'",
                                        constraint_kind, uc
                                    ),
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Enforce CHECK constraints on updated values
        if updated > 0 && storage.storage.has_constraints() {
            let schema = storage.get_schema();
            for (schema_col, _) in &schema {
                let cons = storage.storage.get_column_constraints(schema_col);
                if let Some(ref check_sql) = cons.check_expr_sql {
                    let check_expr = {
                        let parse_sql = format!("SELECT 1 FROM _dummy WHERE {}", check_sql);
                        match crate::query::sql_parser::SqlParser::parse(&parse_sql) {
                            Ok(crate::query::sql_parser::SqlStatement::Select(sel)) => {
                                sel.where_clause.clone()
                            }
                            _ => None,
                        }
                    };
                    if let Some(ref expr) = check_expr {
                        for (_row_id, update_data) in &updates {
                            // Build a 1-row batch from the update data
                            let mut fields = Vec::new();
                            let mut arrays: Vec<ArrayRef> = Vec::new();
                            for (cn, value) in update_data {
                                match value {
                                    Value::Int64(v) => {
                                        fields.push(Field::new(cn, ArrowDataType::Int64, true));
                                        arrays.push(Arc::new(Int64Array::from(vec![*v])));
                                    }
                                    Value::Float64(v) => {
                                        fields.push(Field::new(cn, ArrowDataType::Float64, true));
                                        arrays.push(Arc::new(Float64Array::from(vec![*v])));
                                    }
                                    Value::String(v) => {
                                        fields.push(Field::new(cn, ArrowDataType::Utf8, true));
                                        arrays.push(Arc::new(StringArray::from(vec![v.as_str()])));
                                    }
                                    Value::Bool(v) => {
                                        fields.push(Field::new(cn, ArrowDataType::Boolean, true));
                                        arrays.push(Arc::new(BooleanArray::from(vec![*v])));
                                    }
                                    _ => {}
                                }
                            }
                            if !fields.is_empty() {
                                let batch_schema = Arc::new(Schema::new(fields));
                                if let Ok(row_batch) = RecordBatch::try_new(batch_schema, arrays) {
                                    if let Ok(mask) = Self::evaluate_predicate(&row_batch, expr) {
                                        if mask.len() > 0 && !mask.value(0) {
                                            return Err(io::Error::new(
                                                io::ErrorKind::InvalidInput,
                                                format!(
                                                    "CHECK constraint violated: {} (column '{}')",
                                                    check_sql, schema_col
                                                ),
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Enforce FOREIGN KEY constraints on updated values
        if updated > 0 && storage.storage.has_constraints() {
            let (base_dir, _) = base_dir_and_table(storage_path);
            let schema = storage.get_schema();
            for (schema_col, _) in &schema {
                let cons = storage.storage.get_column_constraints(schema_col);
                if let Some((ref ref_table, ref ref_column)) = cons.foreign_key {
                    // Collect updated values for this FK column
                    let fk_vals: Vec<&Value> = updates
                        .iter()
                        .filter_map(|(_, data)| data.get(schema_col))
                        .collect();
                    if fk_vals.is_empty() {
                        continue;
                    }

                    let ref_path = base_dir.join(format!("{}.apex", ref_table));
                    if !ref_path.exists() {
                        return Err(io::Error::new(
                            io::ErrorKind::NotFound,
                            format!(
                                "FOREIGN KEY: referenced table '{}' does not exist",
                                ref_table
                            ),
                        ));
                    }
                    let ref_storage = TableStorageBackend::open(&ref_path)?;
                    let ref_batch =
                        ref_storage.read_columns_to_arrow(Some(&[ref_column.as_str()]), 0, None)?;
                    let ref_col_arr = ref_batch.column_by_name(ref_column);

                    for val in &fk_vals {
                        if matches!(val, Value::Null) {
                            continue;
                        }
                        let found = if let Some(ref_arr) = ref_col_arr {
                            let mut exists = false;
                            for r in 0..ref_arr.len() {
                                if ref_arr.is_null(r) {
                                    continue;
                                }
                                let matches = match val {
                                    Value::Int64(v) => ref_arr
                                        .as_any()
                                        .downcast_ref::<Int64Array>()
                                        .map(|a| a.value(r) == *v)
                                        .unwrap_or(false),
                                    Value::Float64(v) => ref_arr
                                        .as_any()
                                        .downcast_ref::<Float64Array>()
                                        .map(|a| (a.value(r) - v).abs() < f64::EPSILON)
                                        .unwrap_or(false),
                                    Value::String(v) => ref_arr
                                        .as_any()
                                        .downcast_ref::<StringArray>()
                                        .map(|a| a.value(r) == v.as_str())
                                        .unwrap_or(false),
                                    Value::Bool(v) => ref_arr
                                        .as_any()
                                        .downcast_ref::<BooleanArray>()
                                        .map(|a| a.value(r) == *v)
                                        .unwrap_or(false),
                                    _ => false,
                                };
                                if matches {
                                    exists = true;
                                    break;
                                }
                            }
                            exists
                        } else {
                            false
                        };
                        if !found {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                format!("FOREIGN KEY constraint violated: value in column '{}' not found in {}.{}", schema_col, ref_table, ref_column),
                            ));
                        }
                    }
                }
            }
        }

        // Record cell-level updates in DeltaStore in a single batch (one lock acquisition).
        {
            let mut batch: Vec<(u64, &str, Value)> =
                Vec::with_capacity(updates.iter().map(|(_, m)| m.len()).sum());
            for (row_id, update_data) in &updates {
                for (col, val) in update_data {
                    batch.push((*row_id, col.as_str(), val.clone()));
                }
            }
            storage.delta_batch_update_rows(&batch);
        }

        // Persist delta store to disk for crash safety.
        // No compact_deltas() needed: pending deltas are applied at read time via
        // DeltaMerger, and baked into the base file via apply_pending_deltas_in_place()
        // on the next open_for_write (e.g., DELETE). Safe because of Fix B.
        if updated > 0 {
            storage.save_delta_store()?;

            // Index maintenance: update indexed values
            if !indexed_cols.is_empty() {
                Self::notify_index_delete(storage_path, &old_index_entries);
                // Insert new indexed values directly
                let (base_dir, table_name) = base_dir_and_table(storage_path);
                let idx_mgr_arc = get_index_manager(&base_dir, &table_name);
                let mut idx_mgr = idx_mgr_arc.lock();
                if !idx_mgr.list_indexes().is_empty() {
                    for (row_id, col_vals) in &new_index_entries {
                        let _ = idx_mgr.on_insert(*row_id, col_vals);
                    }
                    let _ = idx_mgr.save();
                }
            }
        }

        // Invalidate cache after write
        invalidate_storage_cache(storage_path);
        invalidate_table_stats(&storage_path.to_string_lossy());

        Ok(ApexResult::Scalar(updated))
    }

    /// Try to extract a simple numeric range from a WHERE clause for the mmap fast path.
    /// Returns (col_name, low, high) for patterns like:
    ///   col = N           → (col, N, N)
    ///   col > N           → (col, N+ε, +∞)
    ///   col >= N          → (col, N, +∞)
    ///   col < N           → (col, -∞, N-ε)
    ///   col <= N          → (col, -∞, N)
    ///   col BETWEEN A AND B → (col, A, B)
    fn extract_numeric_range_from_where(expr: &SqlExpr) -> Option<(String, f64, f64)> {
        match expr {
            SqlExpr::Between {
                column,
                low,
                high,
                negated: false,
            } => {
                let lo = Self::literal_to_f64(low)?;
                let hi = Self::literal_to_f64(high)?;
                Some((column.trim_matches('"').to_string(), lo, hi))
            }
            SqlExpr::BinaryOp { left, op, right } => {
                let (col, val, flipped) = if let SqlExpr::Column(c) = left.as_ref() {
                    if let Some(v) = Self::literal_to_f64(right) {
                        (c.trim_matches('"').to_string(), v, false)
                    } else {
                        return None;
                    }
                } else if let SqlExpr::Column(c) = right.as_ref() {
                    if let Some(v) = Self::literal_to_f64(left) {
                        (c.trim_matches('"').to_string(), v, true)
                    } else {
                        return None;
                    }
                } else {
                    return None;
                };
                let inf = f64::INFINITY;
                let ninf = f64::NEG_INFINITY;
                let tiny = 0.5; // integer step for integer columns
                match op {
                    BinaryOperator::Eq => Some((col, val, val)),
                    BinaryOperator::Gt => {
                        if !flipped {
                            Some((col, val + tiny, inf))
                        } else {
                            Some((col, ninf, val - tiny))
                        }
                    }
                    BinaryOperator::Ge => {
                        if !flipped {
                            Some((col, val, inf))
                        } else {
                            Some((col, ninf, val))
                        }
                    }
                    BinaryOperator::Lt => {
                        if !flipped {
                            Some((col, ninf, val - tiny))
                        } else {
                            Some((col, val + tiny, inf))
                        }
                    }
                    BinaryOperator::Le => {
                        if !flipped {
                            Some((col, ninf, val))
                        } else {
                            Some((col, val, inf))
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn literal_to_f64(expr: &SqlExpr) -> Option<f64> {
        match expr {
            SqlExpr::Literal(Value::Int64(v)) => Some(*v as f64),
            SqlExpr::Literal(Value::Float64(v)) => Some(*v),
            _ => None,
        }
    }

    /// Collect all column names referenced in a SqlExpr (used for column projection).
    fn collect_column_refs(expr: &SqlExpr) -> Vec<String> {
        let mut refs = Vec::new();
        Self::collect_column_refs_inner(expr, &mut refs);
        refs
    }

    fn collect_column_refs_inner(expr: &SqlExpr, refs: &mut Vec<String>) {
        match expr {
            SqlExpr::Column(name) => refs.push(name.trim_matches('"').to_string()),
            SqlExpr::BinaryOp { left, right, .. } => {
                Self::collect_column_refs_inner(left, refs);
                Self::collect_column_refs_inner(right, refs);
            }
            SqlExpr::UnaryOp { expr, .. } => Self::collect_column_refs_inner(expr, refs),
            SqlExpr::Like { column, .. }
            | SqlExpr::Regexp { column, .. }
            | SqlExpr::In { column, .. }
            | SqlExpr::InSubquery { column, .. }
            | SqlExpr::IsNull { column, .. } => refs.push(column.trim_matches('"').to_string()),
            SqlExpr::Between {
                column, low, high, ..
            } => {
                refs.push(column.trim_matches('"').to_string());
                Self::collect_column_refs_inner(low, refs);
                Self::collect_column_refs_inner(high, refs);
            }
            SqlExpr::Case {
                when_then,
                else_expr,
            } => {
                for (cond, then) in when_then {
                    Self::collect_column_refs_inner(cond, refs);
                    Self::collect_column_refs_inner(then, refs);
                }
                if let Some(e) = else_expr {
                    Self::collect_column_refs_inner(e, refs);
                }
            }
            SqlExpr::Function { args, .. } => {
                for a in args {
                    Self::collect_column_refs_inner(a, refs);
                }
            }
            SqlExpr::Cast { expr, .. } | SqlExpr::Paren(expr) => {
                Self::collect_column_refs_inner(expr, refs);
            }
            SqlExpr::ArrayIndex { array, index } => {
                Self::collect_column_refs_inner(array, refs);
                Self::collect_column_refs_inner(index, refs);
            }
            _ => {}
        }
    }

    /// Get a value from an Arrow array at a specific row index
    fn get_value_at(array: &ArrayRef, row: usize) -> Option<Value> {
        if array.is_null(row) {
            return Some(Value::Null);
        }
        if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
            Some(Value::Int64(arr.value(row)))
        } else if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
            Some(Value::Float64(arr.value(row)))
        } else if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
            Some(Value::String(arr.value(row).to_string()))
        } else if let Some(arr) = array.as_any().downcast_ref::<BooleanArray>() {
            Some(Value::Bool(arr.value(row)))
        } else if let Some(arr) = array.as_any().downcast_ref::<UInt64Array>() {
            Some(Value::Int64(arr.value(row) as i64))
        } else {
            None
        }
    }

    /// Evaluate an expression to a Value for UPDATE
    fn evaluate_expr_to_value(
        batch: &RecordBatch,
        expr: &SqlExpr,
        row: usize,
    ) -> io::Result<Value> {
        match expr {
            SqlExpr::Literal(v) => Ok(v.clone()),
            SqlExpr::Column(name) => {
                let col_name = name.trim_matches('"');
                if let Some(col) = batch.column_by_name(col_name) {
                    if col.is_null(row) {
                        return Ok(Value::Null);
                    }
                    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                        Ok(Value::Int64(arr.value(row)))
                    } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                        Ok(Value::Float64(arr.value(row)))
                    } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                        Ok(Value::String(arr.value(row).to_string()))
                    } else if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
                        Ok(Value::Bool(arr.value(row)))
                    } else {
                        Ok(Value::Null)
                    }
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Column '{}' not found", col_name),
                    ))
                }
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Complex expressions in UPDATE not yet supported",
            )),
        }
    }

    // =====================================================================
    // Data Import Table Functions: read_csv / read_json / read_parquet
    // =====================================================================

    /// Dispatch to the right reader based on the table function name.
    pub(crate) fn read_table_function(
        func: &str,
        file: &str,
        options: &[(String, String)],
    ) -> io::Result<RecordBatch> {
        match func.to_uppercase().as_str() {
            "READ_CSV" => Self::read_csv_to_batch(file, options),
            "READ_JSON" => Self::read_json_to_batch(file, options),
            "READ_PARQUET" => Self::read_parquet_to_batch(file),
            other => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("Unknown table function: {}", other),
            )),
        }
    }

    /// DuckDB-style direct file reading: auto-detect format from file extension.
    /// Supports: .csv, .tsv, .json, .jsonl, .ndjson, .parquet, .csv.gz, .csv.gzip
    pub(crate) fn read_direct_file(file: &str) -> io::Result<RecordBatch> {
        let lower = file.to_lowercase();
        if lower.ends_with(".csv.gz") || lower.ends_with(".csv.gzip") {
            return Self::read_csv_to_batch(file, &[]);
        }
        if lower.ends_with(".csv") {
            return Self::read_csv_to_batch(file, &[]);
        }
        if lower.ends_with(".tsv") {
            return Self::read_csv_to_batch(
                file,
                &[("delimiter".to_string(), "\t".to_string())],
            );
        }
        if lower.ends_with(".json")
            || lower.ends_with(".jsonl")
            || lower.ends_with(".ndjson")
        {
            return Self::read_json_to_batch(file, &[]);
        }
        if lower.ends_with(".parquet") {
            return Self::read_parquet_to_batch(file);
        }
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "Unsupported file format for '{}'. Supported: .csv, .tsv, .json, .jsonl, .ndjson, .parquet, .csv.gz",
                file
            ),
        ))
    }

    /// Read a CSV / TSV file into an Arrow RecordBatch — parallel reader.
    /// Splits the file into per-core chunks at newline boundaries, parses concurrently.
    fn read_csv_to_batch(path: &str, options: &[(String, String)]) -> io::Result<RecordBatch> {
        use rayon::prelude::*;

        let has_header = options
            .iter()
            .find(|(k, _)| k == "header")
            .map(|(_, v)| !matches!(v.to_lowercase().as_str(), "false" | "0"))
            .unwrap_or(true);

        let delimiter: u8 = options
            .iter()
            .find(|(k, _)| k == "delimiter" || k == "delim" || k == "sep")
            .and_then(|(_, v)| v.chars().next())
            .map(|c| c as u8)
            .unwrap_or(b',');

        let file = std::fs::File::open(path).map_err(|e| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Cannot open CSV file '{}': {}", path, e),
            )
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("mmap error: {}", e)))?;
        let data: &[u8] = &mmap;

        // Fast schema inference: single-pass over first 100 data rows,
        // bypasses Arrow's CSV reader overhead (extra buffering + allocation layer).
        let schema = Arc::new(Self::infer_csv_schema_fast(
            data, has_header, delimiter, 100,
        )?);

        // Find end of header line (SIMD via memchr)
        let header_end = if has_header {
            memchr::memchr(b'\n', data)
                .map(|i| i + 1)
                .unwrap_or(data.len())
        } else {
            0
        };
        let data_section = &data[header_end..];
        if data_section.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::clone(&schema)));
        }

        // Compute split offsets — one chunk per core, aligned to newlines (SIMD search)
        let n_threads = rayon::current_num_threads().min(16).max(1);
        let mut starts: Vec<usize> = vec![0];
        if n_threads > 1 {
            let chunk = (data_section.len() + n_threads - 1) / n_threads;
            for t in 1..n_threads {
                let approx = t * chunk;
                if approx >= data_section.len() {
                    break;
                }
                let nl = memchr::memchr(b'\n', &data_section[approx..])
                    .map(|p| approx + p + 1)
                    .unwrap_or(data_section.len());
                if nl != *starts.last().unwrap() {
                    starts.push(nl);
                }
            }
        }
        starts.push(data_section.len());

        // Raw-pointer wrapper so slices can cross rayon thread boundaries safely.
        // Safe: `data` Vec outlives all spawned tasks (rayon join before function returns).
        struct SendSlice(*const u8, usize);
        unsafe impl Send for SendSlice {}
        unsafe impl Sync for SendSlice {}

        let chunks: Vec<SendSlice> = starts
            .windows(2)
            .map(|w| {
                let s = &data_section[w[0]..w[1]];
                SendSlice(s.as_ptr(), s.len())
            })
            .collect();

        // Parse filter pushdown option: "col>val" or "col<val" style
        let filter_info = options
            .iter()
            .find(|(k, _)| k == "filter")
            .and_then(|(_, v)| parse_pushdown_filter(v, &schema));

        let schema_ref = Arc::clone(&schema);
        let batches: Vec<io::Result<RecordBatch>> = chunks
            .par_iter()
            .map(|ss| {
                let chunk = unsafe { std::slice::from_raw_parts(ss.0, ss.1) };
                if chunk.is_empty() {
                    return Ok(RecordBatch::new_empty(Arc::clone(&schema_ref)));
                }
                Self::parse_csv_chunk_fast(chunk, &schema_ref, delimiter, filter_info.as_ref())
            })
            .collect();

        let all: Vec<RecordBatch> = batches.into_iter().collect::<io::Result<Vec<_>>>()?;
        Self::merge_record_batches(all)
    }

    /// Custom fast CSV chunk parser.
    /// Single-pass row-by-row with direct Vec buffer building — bypasses
    /// Arrow builder API overhead (5+ fn calls per append → 1-2 direct ops).
    fn parse_csv_chunk_fast(
        data: &[u8],
        schema: &arrow::datatypes::Schema,
        delimiter: u8,
        filter_info: Option<&PushdownFilter>,
    ) -> io::Result<RecordBatch> {
        use arrow::array::{BooleanArray, BooleanBuilder};
        use arrow::buffer::{Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
        use arrow::datatypes::DataType;

        let n_cols = schema.fields().len();
        if n_cols == 0 {
            return Ok(RecordBatch::new_empty(Arc::new(schema.clone())));
        }

        // Estimate row count from chunk size — avoids an extra full-scan just for capacity.
        // Assume ≥20 bytes per line (conservative underestimate ⇒ slight overalloc, no resize).
        if data.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::new(schema.clone())));
        }
        let n_rows = (data.len() / 20).max(1);

        // Per-column raw buffers — direct Vec ops, no builder hierarchy.
        // has_null starts false; nulls Vec is only written when a null IS seen.
        enum ColBuf {
            I64 {
                vals: Vec<i64>,
                nulls: Vec<bool>,
                has_null: bool,
            },
            F64 {
                vals: Vec<f64>,
                nulls: Vec<bool>,
                has_null: bool,
            },
            Str {
                bytes: Vec<u8>,
                offsets: Vec<i32>,
            },
            Bool(BooleanBuilder),
        }

        let mut cols: Vec<ColBuf> = schema
            .fields()
            .iter()
            .map(|f| match f.data_type() {
                DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                    ColBuf::I64 {
                        vals: Vec::with_capacity(n_rows),
                        nulls: Vec::new(),
                        has_null: false,
                    }
                }
                DataType::Float64 | DataType::Float32 => ColBuf::F64 {
                    vals: Vec::with_capacity(n_rows),
                    nulls: Vec::new(),
                    has_null: false,
                },
                DataType::Boolean => ColBuf::Bool(BooleanBuilder::with_capacity(n_rows)),
                _ => {
                    let mut offsets = Vec::with_capacity(n_rows + 1);
                    offsets.push(0i32);
                    ColBuf::Str {
                        bytes: Vec::with_capacity(n_rows * 12),
                        offsets,
                    }
                }
            })
            .collect();

        macro_rules! push_field {
            ($c:expr, $f:expr) => {
                match $c {
                    ColBuf::I64 {
                        vals,
                        nulls,
                        has_null,
                    } => {
                        if $f.is_empty() {
                            if !*has_null {
                                *has_null = true;
                                nulls.resize(vals.len(), true);
                            }
                            vals.push(0);
                            nulls.push(false);
                        } else {
                            vals.push(Self::parse_i64_bytes($f));
                            if *has_null {
                                nulls.push(true);
                            }
                        }
                    }
                    ColBuf::F64 {
                        vals,
                        nulls,
                        has_null,
                    } => match fast_float::parse::<f64, _>($f) {
                        Ok(v) => {
                            vals.push(v);
                            if *has_null {
                                nulls.push(true);
                            }
                        }
                        Err(_) => {
                            if !*has_null {
                                *has_null = true;
                                nulls.resize(vals.len(), true);
                            }
                            vals.push(0.0);
                            nulls.push(false);
                        }
                    },
                    ColBuf::Str { bytes, offsets } => {
                        bytes.extend_from_slice($f);
                        offsets.push(bytes.len() as i32);
                    }
                    ColBuf::Bool(b) => match $f {
                        b"true" | b"True" | b"TRUE" | b"1" => b.append_value(true),
                        b"false" | b"False" | b"FALSE" | b"0" => b.append_value(false),
                        _ => b.append_null(),
                    },
                }
            };
        }
        macro_rules! push_null {
            ($c:expr) => {
                match $c {
                    ColBuf::I64 {
                        vals,
                        nulls,
                        has_null,
                    } => {
                        if !*has_null {
                            *has_null = true;
                            nulls.resize(vals.len(), true);
                        }
                        vals.push(0);
                        nulls.push(false);
                    }
                    ColBuf::F64 {
                        vals,
                        nulls,
                        has_null,
                    } => {
                        if !*has_null {
                            *has_null = true;
                            nulls.resize(vals.len(), true);
                        }
                        vals.push(0.0);
                        nulls.push(false);
                    }
                    ColBuf::Str { bytes, offsets } => {
                        offsets.push(bytes.len() as i32);
                    }
                    ColBuf::Bool(b) => b.append_null(),
                }
            };
        }

        // Single forward pass — outer newline search via SIMD memchr_iter.
        // Inner delimiter search also uses SIMD memchr_iter (replaces scalar byte loop).
        let mut line_start = 0usize;
        for nl in memchr::memchr_iter(b'\n', data) {
            let raw = &data[line_start..nl];
            line_start = nl + 1;
            let line = if raw.last() == Some(&b'\r') {
                &raw[..raw.len() - 1]
            } else {
                raw
            };
            if line.is_empty() {
                continue;
            }
            // Pushdown filter: skip row if filter column's value doesn't match
            if let Some(ref fi) = filter_info {
                let fv = Self::get_csv_field(line, fi.col_idx, delimiter);
                if !csv_filter_match(fv, fi) {
                    continue;
                }
            }
            let mut fs = 0usize;
            let mut col = 0usize;
            for i in memchr::memchr_iter(delimiter, line) {
                if col < n_cols {
                    push_field!(&mut cols[col], &line[fs..i]);
                }
                col += 1;
                fs = i + 1;
            }
            if col < n_cols {
                push_field!(&mut cols[col], &line[fs..]);
                col += 1;
            }
            while col < n_cols {
                push_null!(&mut cols[col]);
                col += 1;
            }
        }
        if line_start < data.len() {
            let raw = &data[line_start..];
            let line = if raw.last() == Some(&b'\r') {
                &raw[..raw.len() - 1]
            } else {
                raw
            };
            if !line.is_empty() {
                let mut fs = 0usize;
                let mut col = 0usize;
                for i in memchr::memchr_iter(delimiter, line) {
                    if col < n_cols {
                        push_field!(&mut cols[col], &line[fs..i]);
                    }
                    col += 1;
                    fs = i + 1;
                }
                if col < n_cols {
                    push_field!(&mut cols[col], &line[fs..]);
                    col += 1;
                }
                while col < n_cols {
                    push_null!(&mut cols[col]);
                    col += 1;
                }
            }
        }

        // Materialize Arrow arrays from raw buffers
        use arrow::array::{Float64Array, Int64Array, StringArray};
        let arrays: Vec<arrow::array::ArrayRef> = cols
            .into_iter()
            .map(|c| match c {
                ColBuf::I64 {
                    vals,
                    nulls,
                    has_null,
                } => {
                    let null_buf = if has_null {
                        Some(NullBuffer::from(nulls))
                    } else {
                        None
                    };
                    Arc::new(Int64Array::new(ScalarBuffer::from(vals), null_buf)) as _
                }
                ColBuf::F64 {
                    vals,
                    nulls,
                    has_null,
                } => {
                    let null_buf = if has_null {
                        Some(NullBuffer::from(nulls))
                    } else {
                        None
                    };
                    Arc::new(Float64Array::new(ScalarBuffer::from(vals), null_buf)) as _
                }
                ColBuf::Str { bytes, offsets } => {
                    let ob = OffsetBuffer::new(ScalarBuffer::from(offsets));
                    let vb = Buffer::from_vec(bytes);
                    Arc::new(StringArray::new(ob, vb, None)) as _
                }
                ColBuf::Bool(mut b) => Arc::new(b.finish()) as _,
            })
            .collect();

        RecordBatch::try_new(Arc::new(schema.clone()), arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))
    }

    /// Fast single-pass CSV schema inference.
    /// Reads header + up to `max_rows` data rows directly from mmap bytes.
    /// Avoids Arrow's CSV reader (which does its own buffering + allocation).
    fn infer_csv_schema_fast(
        data: &[u8],
        has_header: bool,
        delimiter: u8,
        max_rows: usize,
    ) -> io::Result<arrow::datatypes::Schema> {
        use arrow::datatypes::{DataType as ArrowDT, Field, Schema};

        if data.is_empty() {
            return Ok(Schema::empty());
        }

        // ── Parse header row ──────────────────────────────────────────────
        let first_nl = memchr::memchr(b'\n', data).unwrap_or(data.len());
        let first_line_raw = &data[..first_nl];
        let first_line = if first_line_raw.last() == Some(&b'\r') {
            &first_line_raw[..first_line_raw.len() - 1]
        } else {
            first_line_raw
        };

        let col_names: Vec<String> = if has_header {
            // Split header by delimiter — field names (UTF-8)
            let mut names = Vec::new();
            let mut fs = 0usize;
            for i in memchr::memchr_iter(delimiter, first_line) {
                let raw = &first_line[fs..i];
                let s = std::str::from_utf8(raw)
                    .unwrap_or("")
                    .trim_matches('"')
                    .to_string();
                names.push(if s.is_empty() {
                    format!("col_{}", names.len())
                } else {
                    s
                });
                fs = i + 1;
            }
            let tail = &first_line[fs..];
            let s = std::str::from_utf8(tail)
                .unwrap_or("")
                .trim_matches('"')
                .to_string();
            names.push(if s.is_empty() {
                format!("col_{}", names.len())
            } else {
                s
            });
            names
        } else {
            // No header: synthesise f0, f1, …
            let n_cols = memchr::memchr_iter(delimiter, first_line).count() + 1;
            (0..n_cols).map(|i| format!("f{}", i)).collect()
        };

        let n_cols = col_names.len();
        if n_cols == 0 {
            return Ok(Schema::empty());
        }

        // ── Sample up to max_rows data lines to infer types ───────────────
        // Type priority: 0=unknown, 1=Int64, 2=Float64, 3=String
        let mut col_types: Vec<u8> = vec![0u8; n_cols];

        let data_start = if has_header { first_nl + 1 } else { 0 };
        let mut rows_seen = 0usize;
        let mut ls = data_start;

        // Iterate lines via SIMD newline search
        for nl in memchr::memchr_iter(b'\n', &data[data_start..]).map(|p| p + data_start) {
            if rows_seen >= max_rows {
                break;
            }
            let raw = &data[ls..nl];
            ls = nl + 1;
            let line = if raw.last() == Some(&b'\r') {
                &raw[..raw.len() - 1]
            } else {
                raw
            };
            if line.is_empty() {
                continue;
            }

            let mut col = 0usize;
            let mut fs = 0usize;
            for i in memchr::memchr_iter(delimiter, line) {
                if col < n_cols {
                    Self::update_col_type(&mut col_types[col], &line[fs..i]);
                }
                col += 1;
                fs = i + 1;
            }
            if col < n_cols {
                Self::update_col_type(&mut col_types[col], &line[fs..]);
            }
            rows_seen += 1;
        }
        // Handle tail line (no trailing newline)
        if ls < data.len() && rows_seen < max_rows {
            let raw = &data[ls..];
            let line = if raw.last() == Some(&b'\r') {
                &raw[..raw.len() - 1]
            } else {
                raw
            };
            if !line.is_empty() {
                let mut col = 0usize;
                let mut fs = 0usize;
                for i in memchr::memchr_iter(delimiter, line) {
                    if col < n_cols {
                        Self::update_col_type(&mut col_types[col], &line[fs..i]);
                    }
                    col += 1;
                    fs = i + 1;
                }
                if col < n_cols {
                    Self::update_col_type(&mut col_types[col], &line[fs..]);
                }
            }
        }

        let fields: Vec<Field> = col_names
            .iter()
            .zip(col_types.iter())
            .map(|(name, &t)| {
                let dt = match t {
                    0 | 1 => ArrowDT::Int64, // unknown or Int64 → Int64
                    2 => ArrowDT::Float64,
                    _ => ArrowDT::Utf8,
                };
                Field::new(name, dt, true)
            })
            .collect();

        Ok(Schema::new(fields))
    }

    /// Probe a single CSV field byte-slice and escalate the column type tracker.
    /// Type codes: 0=unknown(empty), 1=Int64, 2=Float64, 3=String
    #[inline(always)]
    fn update_col_type(col_type: &mut u8, field: &[u8]) {
        if *col_type >= 3 {
            return;
        } // already String — no point checking
        if field.is_empty() {
            return;
        } // null/empty — don't escalate

        // Strip surrounding quotes (common in exported CSVs)
        let f = if field.first() == Some(&b'"') && field.last() == Some(&b'"') && field.len() >= 2 {
            &field[1..field.len() - 1]
        } else {
            field
        };
        if f.is_empty() {
            return;
        }

        // Try Int64 first (cheapest check)
        if *col_type <= 1 {
            let digits = match f.first() {
                Some(&b'-') | Some(&b'+') => &f[1..],
                _ => f,
            };
            if !digits.is_empty() && digits.iter().all(|&b| b >= b'0' && b <= b'9') {
                *col_type = 1; // Int64
                return;
            }
        }
        // Try Float64
        if *col_type <= 2 {
            if fast_float::parse::<f64, _>(f).is_ok() {
                *col_type = 2; // Float64
                return;
            }
        }
        // Must be String
        *col_type = 3;
    }

    #[inline(always)]
    fn parse_i64_bytes(b: &[u8]) -> i64 {
        let (neg, digits) = match b.first() {
            Some(&b'-') => (true, &b[1..]),
            Some(&b'+') => (false, &b[1..]),
            _ => (false, b),
        };
        let mut v = 0i64;
        for &d in digits {
            v = v * 10 + (d.wrapping_sub(b'0')) as i64;
        }
        if neg {
            -v
        } else {
            v
        }
    }

    #[inline(always)]
    fn parse_u64_bytes(b: &[u8]) -> u64 {
        let digits = if b.first() == Some(&b'+') { &b[1..] } else { b };
        let mut v = 0u64;
        for &d in digits {
            v = v * 10 + (d.wrapping_sub(b'0')) as u64;
        }
        v
    }

    /// Return the `col`-th delimiter-separated field from `line` (0-indexed).
    #[inline(always)]
    fn get_csv_field(line: &[u8], col: usize, delimiter: u8) -> &[u8] {
        let mut count = 0usize;
        let mut start = 0usize;
        for (i, &b) in line.iter().enumerate() {
            if b == delimiter {
                if count == col {
                    return &line[start..i];
                }
                count += 1;
                start = i + 1;
            }
        }
        if count == col {
            &line[start..]
        } else {
            b""
        }
    }

    /// Extract one CSV column from `data` with static type dispatch — no per-field enum match.
    fn extract_csv_column(
        data: &[u8],
        col_idx: usize,
        dtype: &arrow::datatypes::DataType,
        delimiter: u8,
        n_rows: usize,
    ) -> io::Result<arrow::array::ArrayRef> {
        use arrow::array::{
            BooleanBuilder, Float64Builder, Int64Builder, StringBuilder, UInt64Builder,
        };
        use arrow::datatypes::DataType;

        // Shared line iterator body
        macro_rules! scan_lines {
            ($callback:expr) => {{
                let mut ls = 0usize;
                for nl in memchr::memchr_iter(b'\n', data) {
                    let raw = &data[ls..nl];
                    ls = nl + 1;
                    let line = if raw.last() == Some(&b'\r') {
                        &raw[..raw.len() - 1]
                    } else {
                        raw
                    };
                    if !line.is_empty() {
                        $callback(Self::get_csv_field(line, col_idx, delimiter));
                    }
                }
                if ls < data.len() {
                    let raw = &data[ls..];
                    let line = if raw.last() == Some(&b'\r') {
                        &raw[..raw.len() - 1]
                    } else {
                        raw
                    };
                    if !line.is_empty() {
                        $callback(Self::get_csv_field(line, col_idx, delimiter));
                    }
                }
            }};
        }

        match dtype {
            DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                let mut b = Int64Builder::with_capacity(n_rows);
                scan_lines!(|f: &[u8]| if f.is_empty() {
                    b.append_null()
                } else {
                    b.append_value(Self::parse_i64_bytes(f))
                });
                Ok(Arc::new(b.finish()) as _)
            }
            DataType::UInt64 | DataType::UInt32 | DataType::UInt16 | DataType::UInt8 => {
                let mut b = UInt64Builder::with_capacity(n_rows);
                scan_lines!(|f: &[u8]| if f.is_empty() {
                    b.append_null()
                } else {
                    b.append_value(Self::parse_u64_bytes(f))
                });
                Ok(Arc::new(b.finish()) as _)
            }
            DataType::Float64 | DataType::Float32 => {
                let mut b = Float64Builder::with_capacity(n_rows);
                scan_lines!(|f: &[u8]| match fast_float::parse::<f64, _>(f) {
                    Ok(v) => b.append_value(v),
                    Err(_) => b.append_null(),
                });
                Ok(Arc::new(b.finish()) as _)
            }
            DataType::Boolean => {
                let mut b = BooleanBuilder::with_capacity(n_rows);
                scan_lines!(|f: &[u8]| match f {
                    b"true" | b"True" | b"TRUE" | b"1" => b.append_value(true),
                    b"false" | b"False" | b"FALSE" | b"0" => b.append_value(false),
                    _ => b.append_null(),
                });
                Ok(Arc::new(b.finish()) as _)
            }
            _ => {
                let mut b = StringBuilder::with_capacity(n_rows, n_rows * 12);
                scan_lines!(|f: &[u8]| {
                    // SAFETY: CSV is text data — valid UTF-8 in practice
                    b.append_value(unsafe { std::str::from_utf8_unchecked(f) });
                });
                Ok(Arc::new(b.finish()) as _)
            }
        }
    }

    /// Read a NDJSON / JSON-lines file into an Arrow RecordBatch.
    fn read_json_to_batch(path: &str, _options: &[(String, String)]) -> io::Result<RecordBatch> {
        use rayon::prelude::*;

        let file = std::fs::File::open(path).map_err(|e| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Cannot open JSON file '{}': {}", path, e),
            )
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("mmap error: {}", e)))?;
        let bytes: &[u8] = &mmap;

        // Trim leading/trailing whitespace via byte scan (no UTF-8 decode overhead)
        let start = bytes
            .iter()
            .position(|&b| !b.is_ascii_whitespace())
            .unwrap_or(bytes.len());
        let end = bytes
            .iter()
            .rposition(|&b| !b.is_ascii_whitespace())
            .map(|i| i + 1)
            .unwrap_or(start);
        let trimmed_bytes = &bytes[start..end];
        if trimmed_bytes.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::new(
                arrow::datatypes::Schema::empty(),
            )));
        }

        // Fast path: pandas "columns" / structured JSON format (starts with '{').
        // Convert mmap bytes to str for serde — safe because JSON is UTF-8.
        if trimmed_bytes.first() == Some(&b'{') {
            let trimmed = unsafe { std::str::from_utf8_unchecked(trimmed_bytes) };
            if let Some(batch) = Self::try_columns_format_fast(trimmed)? {
                return Ok(batch);
            }
            // Try single-value parse (split/index/records format)
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
                return Self::json_value_to_batch(value);
            }
        }

        // Try array-of-records format
        if trimmed_bytes.first() == Some(&b'[') {
            let trimmed = unsafe { std::str::from_utf8_unchecked(trimmed_bytes) };
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
                return Self::json_value_to_batch(value);
            }
        }

        // NDJSON path: parallel chunk parsing.
        // 1. Infer schema from first 100 lines (sequential, fast).
        // 2. Split file into N_threads line-aligned chunks.
        // 3. Parse each chunk in parallel → Vec<RecordBatch>.
        // 4. Merge.
        let n_threads = rayon::current_num_threads().min(16).max(1);

        // Schema inference: read first 100 lines with Arrow (sequential, small)
        let schema = {
            use arrow::json::reader::infer_json_schema_from_seekable;
            use std::io::BufReader;
            // Take first 100 lines for inference
            let mut ls = 0usize;
            let mut rows = 0usize;
            let mut infer_end = trimmed_bytes.len();
            for nl in memchr::memchr_iter(b'\n', trimmed_bytes) {
                ls = nl + 1;
                rows += 1;
                if rows >= 100 {
                    infer_end = nl + 1;
                    break;
                }
            }
            let _ = ls;
            let mut buf = BufReader::new(std::io::Cursor::new(&trimmed_bytes[..infer_end]));
            let (schema, _) =
                infer_json_schema_from_seekable(&mut buf, Some(100)).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::Other,
                        format!("JSON schema inference: {}", e),
                    )
                })?;
            Arc::new(schema)
        };

        // Build line-aligned chunk boundaries (SIMD newline search)
        let mut starts: Vec<usize> = vec![0];
        if n_threads > 1 {
            let chunk = (trimmed_bytes.len() + n_threads - 1) / n_threads;
            for t in 1..n_threads {
                let approx = t * chunk;
                if approx >= trimmed_bytes.len() {
                    break;
                }
                let nl = memchr::memchr(b'\n', &trimmed_bytes[approx..])
                    .map(|p| approx + p + 1)
                    .unwrap_or(trimmed_bytes.len());
                if nl != *starts.last().unwrap() {
                    starts.push(nl);
                }
            }
        }
        starts.push(trimmed_bytes.len());

        // Raw-pointer wrapper for cross-thread mmap slice sharing
        struct SendSlice(*const u8, usize);
        unsafe impl Send for SendSlice {}
        unsafe impl Sync for SendSlice {}

        let chunks: Vec<SendSlice> = starts
            .windows(2)
            .map(|w| {
                let s = &trimmed_bytes[w[0]..w[1]];
                SendSlice(s.as_ptr(), s.len())
            })
            .collect();

        let schema_ref = Arc::clone(&schema);
        let batches: Vec<io::Result<RecordBatch>> = chunks
            .par_iter()
            .map(|ss| {
                use std::io::BufReader;
                let chunk = unsafe { std::slice::from_raw_parts(ss.0, ss.1) };
                if chunk.is_empty() {
                    return Ok(RecordBatch::new_empty(Arc::clone(&schema_ref)));
                }
                let mut buf = BufReader::new(std::io::Cursor::new(chunk));
                let reader = arrow::json::ReaderBuilder::new(Arc::clone(&schema_ref))
                    .build(&mut buf)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                let sub: Vec<RecordBatch> = reader
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                Self::merge_record_batches(sub)
            })
            .collect();

        let all: Vec<RecordBatch> = batches.into_iter().collect::<io::Result<Vec<_>>>()?;
        Self::merge_record_batches(all)
    }

    fn try_fast_json_count(
        path: &str,
        where_clause: Option<&SqlExpr>,
    ) -> io::Result<Option<i64>> {
        use rayon::prelude::*;

        let file = std::fs::File::open(path).map_err(|e| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Cannot open JSON file '{}': {}", path, e),
            )
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("mmap error: {}", e)))?;
        let bytes: &[u8] = &mmap;

        let start = bytes
            .iter()
            .position(|&b| !b.is_ascii_whitespace())
            .unwrap_or(bytes.len());
        let end = bytes
            .iter()
            .rposition(|&b| !b.is_ascii_whitespace())
            .map(|i| i + 1)
            .unwrap_or(start);
        let trimmed = &bytes[start..end];
        if trimmed.is_empty() {
            return Ok(Some(0));
        }
        if !Self::looks_like_ndjson(trimmed) {
            return Ok(None);
        }

        let filter = match where_clause {
            Some(expr) => Some(match Self::extract_json_numeric_filter(expr) {
                Some(f) => f,
                None => return Ok(None),
            }),
            None => None,
        };

        let n_threads = rayon::current_num_threads().min(16).max(1);
        let mut starts: Vec<usize> = vec![0];
        if n_threads > 1 {
            let chunk = (trimmed.len() + n_threads - 1) / n_threads;
            for t in 1..n_threads {
                let approx = t * chunk;
                if approx >= trimmed.len() {
                    break;
                }
                let nl = memchr::memchr(b'\n', &trimmed[approx..])
                    .map(|p| approx + p + 1)
                    .unwrap_or(trimmed.len());
                if nl != *starts.last().unwrap() {
                    starts.push(nl);
                }
            }
        }
        starts.push(trimmed.len());

        struct SendSlice(*const u8, usize);
        unsafe impl Send for SendSlice {}
        unsafe impl Sync for SendSlice {}

        let chunks: Vec<SendSlice> = starts
            .windows(2)
            .map(|w| {
                let s = &trimmed[w[0]..w[1]];
                SendSlice(s.as_ptr(), s.len())
            })
            .collect();

        let count: usize = chunks
            .par_iter()
            .map(|ss| {
                let chunk = unsafe { std::slice::from_raw_parts(ss.0, ss.1) };
                Self::count_json_chunk_rows(chunk, filter.as_ref())
            })
            .sum();
        Ok(Some(count as i64))
    }

    fn looks_like_ndjson(bytes: &[u8]) -> bool {
        let mut checked = 0usize;
        let mut line_start = 0usize;
        for nl in memchr::memchr_iter(b'\n', bytes).take(16) {
            let raw = &bytes[line_start..nl];
            line_start = nl + 1;
            let line = Self::trim_ascii_json_line(raw);
            if line.is_empty() {
                continue;
            }
            if line.first() != Some(&b'{') || line.last() != Some(&b'}') {
                return false;
            }
            checked += 1;
            if checked >= 4 {
                return true;
            }
        }
        if checked == 0 && line_start < bytes.len() {
            let line = Self::trim_ascii_json_line(&bytes[line_start..]);
            checked = usize::from(
                !line.is_empty() && line.first() == Some(&b'{') && line.last() == Some(&b'}'),
            );
        }
        checked > 0 && memchr::memchr(b'\n', bytes).is_some()
    }

    #[inline(always)]
    fn trim_ascii_json_line(mut line: &[u8]) -> &[u8] {
        while line.first().is_some_and(|b| b.is_ascii_whitespace()) {
            line = &line[1..];
        }
        while line.last().is_some_and(|b| b.is_ascii_whitespace()) {
            line = &line[..line.len() - 1];
        }
        line
    }

    fn extract_json_numeric_filter(expr: &SqlExpr) -> Option<JsonNumericFilter> {
        let SqlExpr::BinaryOp { left, op, right } = expr else {
            return None;
        };
        let (col, val, flipped) = if let SqlExpr::Column(c) = left.as_ref() {
            (c.as_str(), Self::literal_to_f64(right)?, false)
        } else if let SqlExpr::Column(c) = right.as_ref() {
            (c.as_str(), Self::literal_to_f64(left)?, true)
        } else {
            return None;
        };
        if !matches!(
            op,
            BinaryOperator::Eq
                | BinaryOperator::NotEq
                | BinaryOperator::Lt
                | BinaryOperator::Le
                | BinaryOperator::Gt
                | BinaryOperator::Ge
        ) {
            return None;
        }
        let col = col.trim_matches('"');
        let col = col.rsplit('.').next().unwrap_or(col);
        let mut key = Vec::with_capacity(col.len() + 2);
        key.push(b'"');
        key.extend_from_slice(col.as_bytes());
        key.push(b'"');
        Some(JsonNumericFilter {
            key,
            op: op.clone(),
            flipped,
            val_f64: val,
        })
    }

    fn count_json_chunk_rows(data: &[u8], filter: Option<&JsonNumericFilter>) -> usize {
        let mut count = 0usize;
        let mut line_start = 0usize;
        for nl in memchr::memchr_iter(b'\n', data) {
            let line = Self::trim_ascii_json_line(&data[line_start..nl]);
            line_start = nl + 1;
            if Self::json_line_matches_filter(line, filter) {
                count += 1;
            }
        }
        if line_start < data.len() {
            let line = Self::trim_ascii_json_line(&data[line_start..]);
            if Self::json_line_matches_filter(line, filter) {
                count += 1;
            }
        }
        count
    }

    #[inline(always)]
    fn json_line_matches_filter(line: &[u8], filter: Option<&JsonNumericFilter>) -> bool {
        if line.is_empty() {
            return false;
        }
        let Some(filter) = filter else {
            return true;
        };
        let Some(value) = Self::json_line_numeric_value(line, &filter.key) else {
            return false;
        };
        let (lhs, rhs) = if filter.flipped {
            (filter.val_f64, value)
        } else {
            (value, filter.val_f64)
        };
        match filter.op {
            BinaryOperator::Eq => lhs == rhs,
            BinaryOperator::NotEq => lhs != rhs,
            BinaryOperator::Lt => lhs < rhs,
            BinaryOperator::Le => lhs <= rhs,
            BinaryOperator::Gt => lhs > rhs,
            BinaryOperator::Ge => lhs >= rhs,
            _ => false,
        }
    }

    fn json_line_numeric_value(line: &[u8], key: &[u8]) -> Option<f64> {
        let mut search_from = 0usize;
        while search_from < line.len() {
            let pos = memchr::memmem::find(&line[search_from..], key)?;
            let mut i = search_from + pos + key.len();
            while i < line.len() && line[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= line.len() || line[i] != b':' {
                search_from += pos + key.len();
                continue;
            }
            i += 1;
            while i < line.len() && line[i].is_ascii_whitespace() {
                i += 1;
            }
            let quoted = i < line.len() && line[i] == b'"';
            if quoted {
                i += 1;
            }
            let mut end = i;
            while end < line.len() {
                let b = line[end];
                if quoted {
                    if b == b'"' {
                        break;
                    }
                } else if b == b',' || b == b'}' || b == b']' || b.is_ascii_whitespace() {
                    break;
                }
                end += 1;
            }
            if end > i {
                return fast_float::parse::<f64, _>(&line[i..end]).ok();
            }
            return None;
        }
        None
    }

    /// Fast reader for pandas "columns" orientation: {"col": {"0": v, "1": v, ...}, ...}
    ///
    /// Strategy:
    ///   1. Parse outer map with `Box<RawValue>` — only 5 allocations (one per column), zero inner.
    ///   2. Peek first raw token of each inner map to detect the Arrow type.
    ///   3. Parse each inner map with typed `HashMap<u64, T>` — no String key allocations.
    ///   4. Sort by u64 key, build Arrow array directly.
    ///
    /// Returns `None` if the content is not "columns" format (caller falls through to slow path).
    fn try_columns_format_fast(content: &str) -> io::Result<Option<RecordBatch>> {
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDT, Field, Schema};
        use serde_json::value::RawValue;

        // Step 1: parse outer map lazily — each column value stays as unparsed raw JSON.
        // std::HashMap is required here; serde_json cannot deserialize into AHashMap<K,V,S>.
        let outer: std::collections::HashMap<String, Box<RawValue>> =
            match serde_json::from_str(content) {
                Ok(m) => m,
                Err(_) => return Ok(None),
            };
        if outer.is_empty() {
            return Ok(None);
        }

        // Confirm "columns" format: each value must start with '{' (it's a nested object)
        let first_raw = outer.values().next().unwrap().get().trim_start();
        if !first_raw.starts_with('{') {
            return Ok(None); // index/split/records format — fall through to slow path
        }

        // Step 2+3+4: per column — detect type, parse typed, sort, build Arrow array
        let col_names: Vec<String> = outer.keys().cloned().collect();
        let mut fields: Vec<Field> = Vec::with_capacity(col_names.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(col_names.len());

        for col in &col_names {
            let raw_col = outer[col].get();

            // Peek at first non-null value's raw bytes to determine type
            let first_token = Self::peek_first_json_value(raw_col);
            let col_type = match first_token {
                Some(t) if t.starts_with('"') => 3u8, // String
                Some("true") | Some("false") => 2u8,  // Bool
                Some(t) if t.contains('.') || t.contains('e') || t.contains('E') => 1u8, // Float
                _ => 0u8,                             // Int (default)
            };

            match col_type {
                0 => {
                    // Integer column: HashMap<u64, Option<i64>> — no String key allocs
                    let map: std::collections::HashMap<u64, Option<i64>> =
                        serde_json::from_str(raw_col).map_err(|e| {
                            io::Error::new(io::ErrorKind::InvalidData, e.to_string())
                        })?;
                    let mut entries: Vec<(u64, Option<i64>)> = map.into_iter().collect();
                    entries.sort_unstable_by_key(|(k, _)| *k);
                    let data: Vec<Option<i64>> = entries.into_iter().map(|(_, v)| v).collect();
                    fields.push(Field::new(col, ArrowDT::Int64, true));
                    arrays.push(Arc::new(Int64Array::from(data)));
                }
                1 => {
                    // Float column
                    let map: std::collections::HashMap<u64, Option<f64>> =
                        serde_json::from_str(raw_col).map_err(|e| {
                            io::Error::new(io::ErrorKind::InvalidData, e.to_string())
                        })?;
                    let mut entries: Vec<(u64, Option<f64>)> = map.into_iter().collect();
                    entries.sort_unstable_by_key(|(k, _)| *k);
                    let data: Vec<Option<f64>> = entries.into_iter().map(|(_, v)| v).collect();
                    fields.push(Field::new(col, ArrowDT::Float64, true));
                    arrays.push(Arc::new(Float64Array::from(data)));
                }
                2 => {
                    // Bool column
                    let map: std::collections::HashMap<u64, Option<bool>> =
                        serde_json::from_str(raw_col).map_err(|e| {
                            io::Error::new(io::ErrorKind::InvalidData, e.to_string())
                        })?;
                    let mut entries: Vec<(u64, Option<bool>)> = map.into_iter().collect();
                    entries.sort_unstable_by_key(|(k, _)| *k);
                    let data: Vec<Option<bool>> = entries.into_iter().map(|(_, v)| v).collect();
                    fields.push(Field::new(col, ArrowDT::Boolean, true));
                    arrays.push(Arc::new(BooleanArray::from(data)));
                }
                _ => {
                    // String column: HashMap<u64, Option<String>> — no String key allocs
                    let map: std::collections::HashMap<u64, Option<String>> =
                        serde_json::from_str(raw_col).map_err(|e| {
                            io::Error::new(io::ErrorKind::InvalidData, e.to_string())
                        })?;
                    let mut entries: Vec<(u64, Option<String>)> = map.into_iter().collect();
                    entries.sort_unstable_by_key(|(k, _)| *k);
                    let data: Vec<Option<String>> = entries.into_iter().map(|(_, v)| v).collect();
                    fields.push(Field::new(col, ArrowDT::Utf8, true));
                    arrays.push(Arc::new(StringArray::from(data)));
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        Ok(Some(batch))
    }

    /// Peek at the first non-null scalar value in a JSON object `{"k": <value>, ...}`.
    /// Returns the raw token bytes (e.g. `"42"`, `"3.14"`, `"true"`, `"\"hello\""`) or None.
    fn peek_first_json_value(obj_json: &str) -> Option<&str> {
        let bytes = obj_json.as_bytes();
        let mut i = 0;
        // Skip opening '{'
        while i < bytes.len() && bytes[i] != b'{' {
            i += 1;
        }
        i += 1;
        // Find first value (skip key, colon, then read value token)
        'outer: loop {
            // skip whitespace / comma
            while i < bytes.len()
                && (bytes[i] == b' '
                    || bytes[i] == b'\n'
                    || bytes[i] == b'\r'
                    || bytes[i] == b'\t'
                    || bytes[i] == b',')
            {
                i += 1;
            }
            if i >= bytes.len() || bytes[i] == b'}' {
                break;
            }
            // skip key string
            if bytes[i] == b'"' {
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                i += 1; // closing quote
            }
            // skip whitespace + colon
            while i < bytes.len()
                && (bytes[i] == b' '
                    || bytes[i] == b'\n'
                    || bytes[i] == b'\r'
                    || bytes[i] == b'\t'
                    || bytes[i] == b':')
            {
                i += 1;
            }
            if i >= bytes.len() {
                break;
            }
            // read value token
            let start = i;
            let end = match bytes[i] {
                b'"' => {
                    i += 1;
                    while i < bytes.len() && bytes[i] != b'"' {
                        if bytes[i] == b'\\' {
                            i += 1;
                        }
                        i += 1;
                    }
                    i + 1
                }
                b'n' => {
                    // null — skip and try next entry
                    i += 4;
                    continue 'outer;
                }
                _ => {
                    while i < bytes.len()
                        && bytes[i] != b','
                        && bytes[i] != b'}'
                        && bytes[i] != b' '
                        && bytes[i] != b'\n'
                    {
                        i += 1;
                    }
                    i
                }
            };
            if start < end && end <= bytes.len() {
                return Some(&obj_json[start..end]);
            }
            break;
        }
        None
    }

    /// Convert a parsed serde_json::Value directly to a RecordBatch.
    /// Supports array-of-records, pandas columns/index/split formats, and single records.
    fn json_value_to_batch(value: serde_json::Value) -> io::Result<RecordBatch> {
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDT, Field, Schema};

        let err = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, msg.to_string());

        match value {
            // ── Array of records ─────────────────────────────────────────────
            serde_json::Value::Array(records) => {
                if records.is_empty() {
                    return Ok(RecordBatch::new_empty(Arc::new(Schema::empty())));
                }
                // Collect column names from first record
                let col_names: Vec<String> = if let serde_json::Value::Object(ref m) = records[0] {
                    m.keys().cloned().collect()
                } else {
                    return Err(err("Expected array of objects"));
                };
                let n = records.len();
                let mut fields = Vec::with_capacity(col_names.len());
                let mut arrays: Vec<ArrayRef> = Vec::with_capacity(col_names.len());
                for col in &col_names {
                    // Detect type from first non-null value
                    let first = records.iter().find_map(|r| {
                        r.get(col)
                            .and_then(|v| if v.is_null() { None } else { Some(v) })
                    });
                    match first {
                        Some(serde_json::Value::Bool(_)) => {
                            let data: Vec<Option<bool>> = records
                                .iter()
                                .map(|r| r.get(col).and_then(|v| v.as_bool()))
                                .collect();
                            fields.push(Field::new(col, ArrowDT::Boolean, true));
                            arrays.push(Arc::new(BooleanArray::from(data)));
                        }
                        Some(serde_json::Value::Number(num))
                            if num.as_i64().is_some()
                                && !num.as_f64().map(|f| f.fract() != 0.0).unwrap_or(false) =>
                        {
                            let data: Vec<Option<i64>> = records
                                .iter()
                                .map(|r| r.get(col).and_then(|v| v.as_i64()))
                                .collect();
                            fields.push(Field::new(col, ArrowDT::Int64, true));
                            arrays.push(Arc::new(Int64Array::from(data)));
                        }
                        Some(serde_json::Value::Number(_)) => {
                            let data: Vec<Option<f64>> = records
                                .iter()
                                .map(|r| r.get(col).and_then(|v| v.as_f64()))
                                .collect();
                            fields.push(Field::new(col, ArrowDT::Float64, true));
                            arrays.push(Arc::new(Float64Array::from(data)));
                        }
                        _ => {
                            let data: Vec<Option<&str>> = records
                                .iter()
                                .map(|r| r.get(col).and_then(|v| v.as_str()))
                                .collect();
                            fields.push(Field::new(col, ArrowDT::Utf8, true));
                            arrays.push(Arc::new(StringArray::from(data)));
                        }
                    }
                    let _ = n;
                }
                let schema = Arc::new(Schema::new(fields));
                RecordBatch::try_new(schema, arrays)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
            }

            serde_json::Value::Object(map) => {
                // ── pandas "split": {"columns":[...], "data":[[...],...]} ──────
                if let (
                    Some(serde_json::Value::Array(cols)),
                    Some(serde_json::Value::Array(data)),
                ) = (map.get("columns").cloned(), map.get("data").cloned())
                {
                    let col_names: Vec<String> = cols
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect();
                    let n = data.len();
                    let ncols = col_names.len();
                    // Transpose: row-major data → column vecs of serde_json::Value
                    let mut cols_data: Vec<Vec<serde_json::Value>> =
                        vec![Vec::with_capacity(n); ncols];
                    for row in &data {
                        if let serde_json::Value::Array(cells) = row {
                            for ci in 0..ncols {
                                cols_data[ci].push(
                                    cells.get(ci).cloned().unwrap_or(serde_json::Value::Null),
                                );
                            }
                        } else {
                            for ci in 0..ncols {
                                cols_data[ci].push(serde_json::Value::Null);
                            }
                        }
                    }
                    let col_refs: Vec<Vec<&serde_json::Value>> =
                        cols_data.iter().map(|c| c.iter().collect()).collect();
                    return Self::column_vecs_to_batch(col_names, col_refs);
                }

                // ── pandas "columns": {"col": {"0": v, ...}} ──────────────────
                let is_columns = !map.is_empty()
                    && map.values().all(|v| {
                        matches!(v, serde_json::Value::Object(inner)
                        if !inner.is_empty() && inner.keys().all(|k| k.parse::<u64>().is_ok()))
                    });
                if is_columns {
                    let first = map.values().next().unwrap();
                    let mut indices: Vec<u64> = if let serde_json::Value::Object(inner) = first {
                        inner.keys().filter_map(|k| k.parse().ok()).collect()
                    } else {
                        vec![]
                    };
                    indices.sort_unstable();

                    let col_names: Vec<String> = map.keys().cloned().collect();
                    let null = serde_json::Value::Null;
                    let col_vecs: Vec<Vec<&serde_json::Value>> = col_names
                        .iter()
                        .map(|col| {
                            if let Some(serde_json::Value::Object(inner)) = map.get(col) {
                                indices
                                    .iter()
                                    .map(|i| inner.get(&i.to_string()).unwrap_or(&null))
                                    .collect()
                            } else {
                                vec![]
                            }
                        })
                        .collect();
                    return Self::column_vecs_to_batch(col_names, col_vecs);
                }

                // ── pandas "index": {"0": {"col": v}, ...} ────────────────────
                let is_index = !map.is_empty()
                    && map.keys().all(|k| k.parse::<u64>().is_ok())
                    && map
                        .values()
                        .all(|v| matches!(v, serde_json::Value::Object(_)));
                if is_index {
                    let mut entries: Vec<(u64, serde_json::Value)> = map
                        .into_iter()
                        .filter_map(|(k, v)| k.parse::<u64>().ok().map(|n| (n, v)))
                        .collect();
                    entries.sort_by_key(|(n, _)| *n);
                    let records: Vec<serde_json::Value> =
                        entries.into_iter().map(|(_, v)| v).collect();
                    return Self::json_value_to_batch(serde_json::Value::Array(records));
                }

                // ── Single record ──────────────────────────────────────────────
                Self::json_value_to_batch(serde_json::Value::Array(vec![
                    serde_json::Value::Object(map),
                ]))
            }
            _ => Err(err("Unsupported top-level JSON type")),
        }
    }

    /// Build a RecordBatch from column-major data (column names + slices of &Value).
    fn column_vecs_to_batch(
        col_names: Vec<String>,
        cols: Vec<Vec<&serde_json::Value>>,
    ) -> io::Result<RecordBatch> {
        use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDT, Field, Schema};

        let mut fields = Vec::with_capacity(col_names.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(col_names.len());

        for (col, values) in col_names.iter().zip(cols.iter()) {
            let first = values.iter().find(|v| !v.is_null()).copied();
            match first {
                Some(serde_json::Value::Bool(_)) => {
                    let data: Vec<Option<bool>> = values.iter().map(|v| v.as_bool()).collect();
                    fields.push(Field::new(col, ArrowDT::Boolean, true));
                    arrays.push(Arc::new(BooleanArray::from(data)));
                }
                Some(serde_json::Value::Number(num))
                    if num.as_i64().is_some()
                        && !num.as_f64().map(|f| f.fract() != 0.0).unwrap_or(false) =>
                {
                    let data: Vec<Option<i64>> = values.iter().map(|v| v.as_i64()).collect();
                    fields.push(Field::new(col, ArrowDT::Int64, true));
                    arrays.push(Arc::new(Int64Array::from(data)));
                }
                Some(serde_json::Value::Number(_)) => {
                    let data: Vec<Option<f64>> = values.iter().map(|v| v.as_f64()).collect();
                    fields.push(Field::new(col, ArrowDT::Float64, true));
                    arrays.push(Arc::new(Float64Array::from(data)));
                }
                _ => {
                    let data: Vec<Option<&str>> = values.iter().map(|v| v.as_str()).collect();
                    fields.push(Field::new(col, ArrowDT::Utf8, true));
                    arrays.push(Arc::new(StringArray::from(data)));
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Normalize arbitrary JSON orientations to NDJSON — used by COPY FROM JSON.
    fn normalize_json_to_ndjson(content: &str) -> io::Result<String> {
        let trimmed = content.trim();
        let value: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => return Ok(content.to_owned()),
        };
        // Re-use the direct converter path, then serialize each row as NDJSON
        // (COPY path only; read_json_to_batch uses json_value_to_batch directly)
        match &value {
            serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                let batch = Self::json_value_to_batch(value)?;
                // Serialize batch rows back to NDJSON for the COPY insert pipeline
                let mut out = String::new();
                let schema = batch.schema();
                for row_i in 0..batch.num_rows() {
                    let mut obj = serde_json::Map::with_capacity(schema.fields().len());
                    for (col_i, field) in schema.fields().iter().enumerate() {
                        let col = batch.column(col_i);
                        let val = Self::arrow_value_at_col(col, row_i);
                        let jval = match val {
                            crate::data::Value::Int64(n) => serde_json::Value::Number(n.into()),
                            crate::data::Value::Int32(n) => {
                                serde_json::Value::Number((n as i64).into())
                            }
                            crate::data::Value::Float64(f) => serde_json::json!(f),
                            crate::data::Value::Float32(f) => serde_json::json!(f as f64),
                            crate::data::Value::String(s) => serde_json::Value::String(s),
                            crate::data::Value::Bool(b) => serde_json::Value::Bool(b),
                            _ => serde_json::Value::Null,
                        };
                        obj.insert(field.name().clone(), jval);
                    }
                    out.push_str(
                        &serde_json::to_string(&obj)
                            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?,
                    );
                    out.push('\n');
                }
                Ok(out)
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unsupported JSON type",
            )),
        }
    }

    /// Read a Parquet file into an Arrow RecordBatch — parallel row-group reader.
    fn read_parquet_to_batch(path: &str) -> io::Result<RecordBatch> {
        use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};
        use rayon::prelude::*;

        let file = std::fs::File::open(path).map_err(|e| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Cannot open Parquet file '{}': {}", path, e),
            )
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("mmap error on '{}': {}", path, e),
            )
        })?;
        let shared = Arc::new(bytes::Bytes::from_owner(mmap));

        // Parse metadata ONCE; all parallel readers share it via clone() (cheap Arc increments).
        // Bytes implements ChunkReader; clone() is O(1) (just increments the Arc refcount).
        let arrow_meta = ArrowReaderMetadata::load(&(*shared).clone(), Default::default())
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        let n_groups = arrow_meta.metadata().num_row_groups();
        let total_rows = arrow_meta.metadata().file_metadata().num_rows() as usize;

        if n_groups <= 1 {
            // Single row group: read columns in parallel for max decompression throughput.
            // Each column reader shares the same mmap bytes (O(1) Arc clone).
            // We cap at min(n_cols, n_threads) to bound metadata-parse overhead.
            let n_threads = rayon::current_num_threads();
            let schema = arrow_meta.schema().clone();
            let n_cols = schema.fields().len();

            if n_cols <= 1 {
                // Trivial case: single column, build directly.
                let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(
                    (*shared).clone(),
                    arrow_meta,
                )
                .with_batch_size(total_rows.max(1))
                .build()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                let batches: Vec<RecordBatch> = reader
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                return Self::merge_record_batches(batches);
            }

            // Parquet schema descriptor (needed for ProjectionMask::leaves).
            let parquet_schema = arrow_meta.parquet_schema().clone();

            // Group columns into at most n_threads buckets.
            let bucket_size = ((n_cols + n_threads - 1) / n_threads).max(1);
            let col_buckets: Vec<Vec<usize>> = (0..n_cols)
                .collect::<Vec<_>>()
                .chunks(bucket_size)
                .map(|c| c.to_vec())
                .collect();

            let bucket_results: Vec<io::Result<RecordBatch>> = col_buckets
                .into_par_iter()
                .map(|col_idxs| {
                    use parquet::arrow::ProjectionMask;
                    // new_with_metadata: reuses pre-parsed metadata (cheap clone — all Arc internals).
                    let b = ParquetRecordBatchReaderBuilder::new_with_metadata(
                        (*shared).clone(),
                        arrow_meta.clone(),
                    );
                    let mask = ProjectionMask::leaves(&parquet_schema, col_idxs);
                    let reader = b
                        .with_batch_size(total_rows.max(1))
                        .with_projection(mask)
                        .build()
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                    let batches: Vec<RecordBatch> = reader
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                    Self::merge_record_batches(batches)
                })
                .collect();

            // Reassemble columns in original order
            let sub_batches: Vec<RecordBatch> =
                bucket_results.into_iter().collect::<io::Result<Vec<_>>>()?;

            // Stitch columns from sub-batches back into one RecordBatch
            let mut all_arrays: Vec<(usize, arrow::array::ArrayRef)> = Vec::with_capacity(n_cols);
            let mut col_written = 0usize;
            for sb in &sub_batches {
                for ci in 0..sb.num_columns() {
                    all_arrays.push((col_written + ci, sb.column(ci).clone()));
                }
                col_written += sb.num_columns();
            }
            all_arrays.sort_by_key(|(i, _)| *i);
            let arrays: Vec<arrow::array::ArrayRef> =
                all_arrays.into_iter().map(|(_, a)| a).collect();
            return RecordBatch::try_new(schema, arrays)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()));
        }

        // Multiple row groups: decode each in parallel, sharing pre-parsed metadata.
        let batches: Vec<io::Result<RecordBatch>> = (0..n_groups)
            .into_par_iter()
            .map(|rg| {
                let b = ParquetRecordBatchReaderBuilder::new_with_metadata(
                    (*shared).clone(),
                    arrow_meta.clone(),
                );

                let rows_in_group = b.metadata().row_group(rg).num_rows() as usize;
                let reader = b
                    .with_row_groups(vec![rg])
                    .with_batch_size(rows_in_group.max(1))
                    .build()
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                let sub: Vec<RecordBatch> = reader
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                Self::merge_record_batches(sub)
            })
            .collect();

        let all: Vec<RecordBatch> = batches.into_iter().collect::<io::Result<Vec<_>>>()?;
        Self::merge_record_batches(all)
    }

    /// Concatenate multiple RecordBatches into one (or return empty if none).
    fn merge_record_batches(batches: Vec<RecordBatch>) -> io::Result<RecordBatch> {
        if batches.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::new(
                arrow::datatypes::Schema::empty(),
            )));
        }
        if batches.len() == 1 {
            return Ok(batches.into_iter().next().unwrap());
        }
        let schema = batches[0].schema();
        let refs: Vec<&RecordBatch> = batches.iter().collect();
        arrow::compute::concat_batches(&schema, refs).map_err(|e| {
            io::Error::new(io::ErrorKind::Other, format!("Concat batches error: {}", e))
        })
    }

    /// Execute COPY table FROM 'file' — auto-detect format (CSV, JSON, PARQUET).
    pub(crate) fn execute_copy_import(
        storage_path: &Path,
        table_name: &str,
        file_path: &str,
        format: &str,
        options: &[(String, String)],
        base_dir: &Path,
        default_table_path: &Path,
    ) -> io::Result<ApexResult> {
        let batch = match format.to_uppercase().as_str() {
            "CSV" | "TSV" => Self::read_csv_to_batch(file_path, options)?,
            "JSON" | "NDJSON" | "JSONL" => Self::read_json_to_batch(file_path, options)?,
            _ => Self::read_parquet_to_batch(file_path)?,
        };

        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(ApexResult::Scalar(0));
        }

        let schema = batch.schema();
        let col_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

        // Ensure table exists — auto-create from schema if needed.
        // NOTE: we call execute_create_table directly (NOT via execute_parsed_multi) because
        // we are already inside a with_table_write_lock; going through execute_parsed_multi
        // would attempt to re-acquire the same lock and deadlock.
        if !storage_path.exists() {
            use crate::data::DataType as ApexDataType;
            use crate::query::sql_parser::ColumnDef;
            let columns: Vec<ColumnDef> = schema
                .fields()
                .iter()
                .map(|field| {
                    let apex_type = match field.data_type() {
                        arrow::datatypes::DataType::Int64
                        | arrow::datatypes::DataType::Int32
                        | arrow::datatypes::DataType::Int16
                        | arrow::datatypes::DataType::Int8
                        | arrow::datatypes::DataType::UInt64
                        | arrow::datatypes::DataType::UInt32 => ApexDataType::Int64,
                        arrow::datatypes::DataType::Float64
                        | arrow::datatypes::DataType::Float32 => ApexDataType::Float64,
                        arrow::datatypes::DataType::Boolean => ApexDataType::Bool,
                        _ => ApexDataType::String,
                    };
                    ColumnDef {
                        name: field.name().clone(),
                        data_type: apex_type,
                        constraints: vec![],
                    }
                })
                .collect();
            Self::execute_create_table(storage_path, table_name, &columns, true)?;
        }

        // Insert row-by-row via execute_insert
        let mut values: Vec<Vec<Value>> = Vec::with_capacity(num_rows);
        for row_idx in 0..num_rows {
            let mut row: Vec<Value> = Vec::with_capacity(col_names.len());
            for col_idx in 0..col_names.len() {
                let col = batch.column(col_idx);
                row.push(Self::arrow_value_at_col(col, row_idx));
            }
            values.push(row);
        }

        Self::execute_insert(storage_path, Some(&col_names), &values)?;
        Ok(ApexResult::Scalar(num_rows as i64))
    }
}
