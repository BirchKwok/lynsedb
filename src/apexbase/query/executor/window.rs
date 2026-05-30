// Window functions, UNION execution

impl ApexExecutor {
    /// Execute UNION / INTERSECT / EXCEPT statement
    fn execute_union(union: UnionStatement, base_dir: &Path, default_table_path: &Path) -> io::Result<ApexResult> {
        use crate::query::SetOpType;

        let left_result = Self::execute_parsed_multi(*union.left, base_dir, default_table_path)?;
        let left_batch = left_result.to_record_batch()?;

        let right_result = Self::execute_parsed_multi(*union.right, base_dir, default_table_path)?;
        let right_batch = right_result.to_record_batch()?;

        if left_batch.num_columns() != right_batch.num_columns() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Set operation requires same number of columns",
            ));
        }

        let mut result = match union.set_op {
            SetOpType::Union => {
                let combined = Self::concat_batches(&left_batch, &right_batch)?;
                if union.all { combined } else { Self::deduplicate_batch(&combined)? }
            }
            SetOpType::Intersect => {
                // Keep rows from left that also appear in right (deduplicated)
                Self::intersect_batches(&left_batch, &right_batch, union.all)?
            }
            SetOpType::Except => {
                // Keep rows from left that do NOT appear in right (deduplicated)
                Self::except_batches(&left_batch, &right_batch, union.all)?
            }
        };

        if !union.order_by.is_empty() {
            result = Self::apply_order_by(&result, &union.order_by)?;
        }
        if union.limit.is_some() || union.offset.is_some() {
            result = Self::apply_limit_offset(&result, union.limit, union.offset)?;
        }

        Ok(ApexResult::Data(result))
    }

    /// INTERSECT: rows in left that also appear in right
    fn intersect_batches(left: &RecordBatch, right: &RecordBatch, all: bool) -> io::Result<RecordBatch> {
        // Build a hash set of right rows
        let right_hashes: std::collections::HashSet<u64> = (0..right.num_rows())
            .map(|i| Self::hash_row(right, i))
            .collect();

        let mut keep: Vec<u32> = Vec::new();
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for i in 0..left.num_rows() {
            let h = Self::hash_row(left, i);
            if right_hashes.contains(&h) {
                if all || seen.insert(h) {
                    keep.push(i as u32);
                }
            }
        }
        Self::take_rows(left, &keep)
    }

    /// EXCEPT: rows in left that do NOT appear in right
    fn except_batches(left: &RecordBatch, right: &RecordBatch, all: bool) -> io::Result<RecordBatch> {
        let right_hashes: std::collections::HashSet<u64> = (0..right.num_rows())
            .map(|i| Self::hash_row(right, i))
            .collect();

        let mut keep: Vec<u32> = Vec::new();
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for i in 0..left.num_rows() {
            let h = Self::hash_row(left, i);
            if !right_hashes.contains(&h) {
                if all || seen.insert(h) {
                    keep.push(i as u32);
                }
            }
        }
        Self::take_rows(left, &keep)
    }

    /// Hash all column values in a single row to a u64 fingerprint
    fn hash_row(batch: &RecordBatch, row: usize) -> u64 {
        use std::hash::Hasher;
        let mut hasher = AHasher::default();
        for col in batch.columns() {
            hasher.write_u64(Self::hash_array_value_fast(col, row));
        }
        hasher.finish()
    }

    /// Take rows by index from a RecordBatch
    fn take_rows(batch: &RecordBatch, indices: &[u32]) -> io::Result<RecordBatch> {
        use arrow::array::UInt32Array;
        let idx_arr = UInt32Array::from(indices.to_vec());
        let cols: Vec<ArrayRef> = batch.columns().iter()
            .map(|col| compute::take(col.as_ref(), &idx_arr, None)
                .map_err(|e| err_data(e.to_string())))
            .collect::<io::Result<Vec<_>>>()?;
        RecordBatch::try_new(batch.schema(), cols)
            .map_err(|e| err_data(e.to_string()))
    }

    /// Concatenate two record batches
    fn concat_batches(left: &RecordBatch, right: &RecordBatch) -> io::Result<RecordBatch> {
        if left.num_rows() == 0 {
            return Ok(right.clone());
        }
        if right.num_rows() == 0 {
            return Ok(left.clone());
        }

        let mut columns: Vec<ArrayRef> = Vec::with_capacity(left.num_columns());
        
        for i in 0..left.num_columns() {
            let left_col = left.column(i);
            let right_col = right.column(i);
            
            let concatenated = compute::concat(&[left_col.as_ref(), right_col.as_ref()])
                .map_err(|e| err_data( e.to_string()))?;
            columns.push(concatenated);
        }

        RecordBatch::try_new(left.schema(), columns)
            .map_err(|e| err_data( e.to_string()))
    }

    /// Deduplicate rows in a record batch (for UNION without ALL)
    /// OPTIMIZATION: Fast path for single-column DISTINCT using dictionary indexing
    fn deduplicate_batch(batch: &RecordBatch) -> io::Result<RecordBatch> {
        use ahash::AHashSet;
        use std::hash::Hasher;
        use arrow::array::DictionaryArray;
        use arrow::datatypes::UInt32Type;
        
        let num_rows = batch.num_rows();
        if num_rows <= 1 {
            return Ok(batch.clone());
        }

        let num_cols = batch.num_columns();
        
        // FAST PATH: Single column DISTINCT - use direct dictionary indexing
        if num_cols == 1 {
            let col = batch.column(0);
            
            // Case 1: DictionaryArray - already has unique values, just get first occurrence of each key
            if let Some(dict_arr) = col.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
                let keys = dict_arr.keys();
                let dict_size = dict_arr.values().len() + 1; // +1 for NULL
                let mut first_occurrence: Vec<Option<u32>> = vec![None; dict_size];
                let mut keep_indices: Vec<u32> = Vec::with_capacity(dict_size);
                
                for row_idx in 0..num_rows {
                    let key = if keys.is_null(row_idx) { 0usize } else { keys.value(row_idx) as usize + 1 };
                    if first_occurrence[key].is_none() {
                        first_occurrence[key] = Some(row_idx as u32);
                        keep_indices.push(row_idx as u32);
                    }
                }
                
                if keep_indices.len() == num_rows {
                    return Ok(batch.clone());
                }
                
                let indices = arrow::array::UInt32Array::from(keep_indices);
                let filtered = compute::take(col.as_ref(), &indices, None)
                    .map_err(|e| err_data( e.to_string()))?;
                return RecordBatch::try_new(batch.schema(), vec![filtered])
                    .map_err(|e| err_data( e.to_string()));
            }
            
            // Case 2: StringArray - build dictionary on the fly for low cardinality
            // REMOVED sampling to stabilize performance
            if let Some(str_arr) = col.as_any().downcast_ref::<StringArray>() {
                // Build dictionary directly without sampling
                let mut dict: AHashMap<&str, u32> = AHashMap::with_capacity(1000);
                let mut keep_indices: Vec<u32> = Vec::with_capacity(1000);
                let mut has_null = false;
                
                for row_idx in 0..num_rows {
                    if str_arr.is_null(row_idx) {
                        if !has_null {
                            has_null = true;
                            keep_indices.push(row_idx as u32);
                        }
                    } else {
                        let s = str_arr.value(row_idx);
                        if !dict.contains_key(s) {
                            dict.insert(s, row_idx as u32);
                            keep_indices.push(row_idx as u32);
                        }
                    }
                }
                
                if keep_indices.len() == num_rows {
                    return Ok(batch.clone());
                }
                
                let indices = arrow::array::UInt32Array::from(keep_indices);
                let filtered = compute::take(col.as_ref(), &indices, None)
                    .map_err(|e| err_data( e.to_string()))?;
                return RecordBatch::try_new(batch.schema(), vec![filtered])
                    .map_err(|e| err_data( e.to_string()));
            }
            
            // Case 3: Int64Array - use direct value dedup
            if let Some(int_arr) = col.as_any().downcast_ref::<Int64Array>() {
                let mut seen: AHashSet<i64> = AHashSet::with_capacity(num_rows.min(10000));
                let mut keep_indices: Vec<u32> = Vec::with_capacity(num_rows.min(10000));
                let mut has_null = false;
                
                for row_idx in 0..num_rows {
                    if int_arr.is_null(row_idx) {
                        if !has_null {
                            has_null = true;
                            keep_indices.push(row_idx as u32);
                        }
                    } else if seen.insert(int_arr.value(row_idx)) {
                        keep_indices.push(row_idx as u32);
                    }
                }
                
                if keep_indices.len() == num_rows {
                    return Ok(batch.clone());
                }
                
                let indices = arrow::array::UInt32Array::from(keep_indices);
                let filtered = compute::take(col.as_ref(), &indices, None)
                    .map_err(|e| err_data( e.to_string()))?;
                return RecordBatch::try_new(batch.schema(), vec![filtered])
                    .map_err(|e| err_data( e.to_string()));
            }
        }
        
        // General path for multi-column deduplication
        // Pre-compute column types for faster dispatch
        enum ColType<'a> {
            Int64(&'a Int64Array),
            Float64(&'a Float64Array),
            String(&'a StringArray, Vec<u64>),  // Pre-computed string hashes
            Bool(&'a BooleanArray),
            Other(&'a ArrayRef),
        }
        
        let typed_cols: Vec<ColType> = batch.columns().iter().map(|col| {
            if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                ColType::Int64(arr)
            } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                ColType::Float64(arr)
            } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                // Pre-compute hashes for strings
                let hashes: Vec<u64> = (0..num_rows).map(|i| {
                    if arr.is_null(i) { 0 } else {
                        let mut h = ahash::AHasher::default();
                        h.write(arr.value(i).as_bytes());
                        h.finish()
                    }
                }).collect();
                ColType::String(arr, hashes)
            } else if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
                ColType::Bool(arr)
            } else {
                ColType::Other(col)
            }
        }).collect();
        
        // Pre-compute all row hashes for parallel deduplication
        let row_hashes: Vec<u64> = (0..num_rows)
            .map(|row_idx| {
                let mut hasher = ahash::AHasher::default();
                for typed_col in &typed_cols {
                    match typed_col {
                        ColType::Int64(arr) => {
                            if arr.is_null(row_idx) {
                                hasher.write_u8(0);
                            } else {
                                hasher.write_u8(1);
                                hasher.write_i64(arr.value(row_idx));
                            }
                        }
                        ColType::Float64(arr) => {
                            if arr.is_null(row_idx) {
                                hasher.write_u8(0);
                            } else {
                                hasher.write_u8(1);
                                hasher.write_u64(arr.value(row_idx).to_bits());
                            }
                        }
                        ColType::String(_arr, hashes) => {
                            hasher.write_u64(hashes[row_idx]);
                        }
                        ColType::Bool(arr) => {
                            if arr.is_null(row_idx) {
                                hasher.write_u8(0);
                            } else {
                                hasher.write_u8(if arr.value(row_idx) { 2 } else { 1 });
                            }
                        }
                        ColType::Other(arr) => {
                            hasher.write_u8(if arr.is_null(row_idx) { 0 } else { 1 });
                            hasher.write_usize(row_idx);
                        }
                    }
                }
                hasher.finish()
            })
            .collect();
        
        // Sequential deduplication using pre-computed hashes
        let mut seen: AHashSet<u64> = AHashSet::with_capacity(num_rows.min(10000));
        let mut keep_indices: Vec<u32> = Vec::with_capacity(num_rows.min(10000));

        for (row_idx, &hash) in row_hashes.iter().enumerate() {
            if seen.insert(hash) {
                keep_indices.push(row_idx as u32);
            }
        }

        if keep_indices.len() == num_rows {
            return Ok(batch.clone());
        }

        // Create filtered batch
        let indices = arrow::array::UInt32Array::from(keep_indices);
        let mut result_columns: Vec<ArrayRef> = Vec::with_capacity(num_cols);
        
        for col in batch.columns() {
            let filtered = compute::take(col.as_ref(), &indices, None)
                .map_err(|e| err_data( e.to_string()))?;
            result_columns.push(filtered);
        }

        RecordBatch::try_new(batch.schema(), result_columns)
            .map_err(|e| err_data( e.to_string()))
    }

    /// DISTINCT ON: keep first row per unique combination of specified columns
    pub(crate) fn deduplicate_batch_on(batch: &RecordBatch, on_columns: &[String]) -> io::Result<RecordBatch> {
        if batch.num_rows() <= 1 || on_columns.is_empty() {
            return Ok(batch.clone());
        }

        // Find column indices for the ON columns
        let col_indices: Vec<usize> = on_columns
            .iter()
            .filter_map(|col_name| {
                batch.schema().index_of(col_name).ok()
            })
            .collect();

        if col_indices.is_empty() {
            return Ok(batch.clone());
        }

        let num_rows = batch.num_rows();
        let mut keep_indices: Vec<u32> = Vec::with_capacity(num_rows.min(10000));
        let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::with_capacity(num_rows.min(10000));

        for row_idx in 0..num_rows {
            let mut key = Vec::with_capacity(col_indices.len() * 16);
            for &col_idx in &col_indices {
                let col = batch.column(col_idx);
                if col.is_null(row_idx) {
                    key.push(0);
                } else {
                    key.push(1);
                    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                        key.extend_from_slice(&arr.value(row_idx).to_le_bytes());
                    } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                        key.extend_from_slice(&arr.value(row_idx).to_bits().to_le_bytes());
                    } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                        key.extend_from_slice(arr.value(row_idx).as_bytes());
                    } else if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
                        key.push(arr.value(row_idx) as u8);
                    } else {
                        // Fallback: use Debug representation
                        key.extend_from_slice(format!("{:?}", col).as_bytes());
                    }
                }
            }
            if seen.insert(key) {
                keep_indices.push(row_idx as u32);
            }
        }

        if keep_indices.len() == num_rows {
            return Ok(batch.clone());
        }

        let indices = arrow::array::UInt32Array::from(keep_indices);
        let mut result_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
        for col in batch.columns() {
            let filtered = compute::take(col.as_ref(), &indices, None)
                .map_err(|e| err_data( e.to_string()))?;
            result_columns.push(filtered);
        }

        RecordBatch::try_new(batch.schema(), result_columns)
            .map_err(|e| err_data( e.to_string()))
    }

    /// Append value signature for deduplication
    fn append_value_signature(sig: &mut Vec<u8>, array: &ArrayRef, idx: usize) {
        if array.is_null(idx) {
            sig.push(0);
            return;
        }
        sig.push(1);

        if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
            sig.extend_from_slice(&arr.value(idx).to_le_bytes());
        } else if let Some(arr) = array.as_any().downcast_ref::<UInt64Array>() {
            sig.extend_from_slice(&arr.value(idx).to_le_bytes());
        } else if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
            sig.extend_from_slice(&arr.value(idx).to_bits().to_le_bytes());
        } else if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
            let s = arr.value(idx);
            sig.extend_from_slice(&(s.len() as u32).to_le_bytes());
            sig.extend_from_slice(s.as_bytes());
        } else if let Some(arr) = array.as_any().downcast_ref::<BooleanArray>() {
            sig.push(if arr.value(idx) { 1 } else { 0 });
        }
    }

    /// Execute window function (ROW_NUMBER, RANK, DENSE_RANK, NTILE, PERCENT_RANK, CUME_DIST, LAG, LEAD, SUM, AVG, etc.)
    fn execute_window_function(batch: &RecordBatch, stmt: &SelectStatement) -> io::Result<ApexResult> {
        // Collect window specs: (func_name, args, partition_by, order_by, output_name)
        let mut window_specs: Vec<(String, Vec<String>, Vec<String>, Vec<crate::query::OrderByClause>, String)> = Vec::new();
        
        let supported = ["ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE", "PERCENT_RANK", "CUME_DIST", "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE", "SUM", "AVG", "COUNT", "MIN", "MAX", "RUNNING_SUM"];
        
        for col in &stmt.columns {
            if let SelectColumn::WindowFunction { name, args, partition_by, order_by, alias } = col {
                if !supported.iter().any(|s| name.eq_ignore_ascii_case(s)) {
                    return Err(err_input(format!("Unsupported window function: {}", name)));
                }
                let out_name = alias.clone().unwrap_or_else(|| name.to_lowercase());
                let upper = name.to_ascii_uppercase();
                window_specs.push((upper, args.clone(), partition_by.clone(), order_by.clone(), out_name));
            }
        }

        if window_specs.is_empty() {
            return Err(err_input("No window function found"));
        }

        let num_rows = batch.num_rows();
        let num_specs = window_specs.len();
        // Per-spec nullable output: int (rank/row_number) or float (sum/avg/lag)
        let mut per_int: Vec<Vec<Option<i64>>> = vec![vec![None; num_rows]; num_specs];
        let mut per_flt: Vec<Vec<Option<f64>>> = vec![vec![None; num_rows]; num_specs];
        let mut use_float: Vec<bool> = vec![false; num_specs];

        for (spec_idx, (func_name, func_args, partition_by, order_by, _)) in window_specs.iter().enumerate() {
            let src_col_name = func_args.get(0).map(|s| s.trim_matches('"').to_string());
            let is_float_src = src_col_name.as_deref().and_then(|cn| batch.column_by_name(cn))
                .map_or(false, |c| c.as_any().downcast_ref::<Float64Array>().is_some());
            use_float[spec_idx] = is_float_src || matches!(func_name.as_str(), "AVG" | "PERCENT_RANK" | "CUME_DIST");

            // Build partition groups
            let mut groups: AHashMap<u64, Vec<usize>> = AHashMap::with_capacity(num_rows / 10 + 1);
            let part_cols: Vec<Option<&ArrayRef>> = partition_by.iter()
                .map(|cn| batch.column_by_name(cn.trim_matches('"')))
                .collect();
            for row_idx in 0..num_rows {
                let mut hasher = AHasher::default();
                for col_opt in &part_cols {
                    if let Some(col) = col_opt {
                        hasher.write_u64(Self::hash_array_value_fast(col, row_idx));
                    }
                }
                let key = if partition_by.is_empty() { 0 } else { hasher.finish() };
                groups.entry(key).or_insert_with(|| Vec::with_capacity(16)).push(row_idx);
            }

            for (_, mut indices) in groups {
                // Sort within partition by ORDER BY
                let order_col: Option<ArrayRef> = if !order_by.is_empty() {
                    let ocn = order_by[0].column.trim_matches('"');
                    let desc = order_by[0].descending;
                    batch.column_by_name(ocn).map(|c| {
                        indices.sort_by(|&a, &b| {
                            let cmp = Self::compare_array_values(c, a, b);
                            if desc { cmp.reverse() } else { cmp }
                        });
                        c.clone()
                    })
                } else { None };

                match func_name.as_str() {
                    "ROW_NUMBER" => {
                        for (pos, &row_idx) in indices.iter().enumerate() {
                            per_int[spec_idx][row_idx] = Some((pos + 1) as i64);
                        }
                    }
                    "RANK" => {
                        let mut rank = 1i64;
                        let mut prev: Option<usize> = None;
                        for (pos, &row_idx) in indices.iter().enumerate() {
                            if let (Some(p), Some(ref col)) = (prev, &order_col) {
                                if Self::compare_array_values(col, p, row_idx) != std::cmp::Ordering::Equal {
                                    rank = (pos + 1) as i64;
                                }
                            }
                            per_int[spec_idx][row_idx] = Some(rank);
                            prev = Some(row_idx);
                        }
                    }
                    "DENSE_RANK" => {
                        let mut rank = 1i64;
                        let mut prev: Option<usize> = None;
                        for &row_idx in &indices {
                            if let (Some(p), Some(ref col)) = (prev, &order_col) {
                                if Self::compare_array_values(col, p, row_idx) != std::cmp::Ordering::Equal {
                                    rank += 1;
                                }
                            }
                            per_int[spec_idx][row_idx] = Some(rank);
                            prev = Some(row_idx);
                        }
                    }
                    "NTILE" => {
                        let n = func_args.get(0).and_then(|s| s.parse::<i64>().ok()).unwrap_or(4);
                        let count = indices.len() as i64;
                        for (pos, &row_idx) in indices.iter().enumerate() {
                            per_int[spec_idx][row_idx] = Some(((pos as i64 * n / count) + 1).min(n));
                        }
                    }
                    "PERCENT_RANK" => {
                        let count = indices.len();
                        let mut rank = 1i64;
                        let mut prev: Option<usize> = None;
                        for (pos, &row_idx) in indices.iter().enumerate() {
                            if let (Some(p), Some(ref col)) = (prev, &order_col) {
                                if Self::compare_array_values(col, p, row_idx) != std::cmp::Ordering::Equal {
                                    rank = (pos + 1) as i64;
                                }
                            }
                            let pct = if count <= 1 { 0.0 } else { (rank - 1) as f64 / (count - 1) as f64 };
                            per_flt[spec_idx][row_idx] = Some(pct);
                            prev = Some(row_idx);
                        }
                    }
                    "CUME_DIST" => {
                        let count = indices.len();
                        let mut rank = 0usize;
                        let mut same = 1usize;
                        let mut prev: Option<usize> = None;
                        for (pos, &row_idx) in indices.iter().enumerate() {
                            if let (Some(p), Some(ref col)) = (prev, &order_col) {
                                if Self::compare_array_values(col, p, row_idx) == std::cmp::Ordering::Equal {
                                    same += 1;
                                } else { rank = pos; same = 1; }
                            }
                            per_flt[spec_idx][row_idx] = Some((rank + same) as f64 / count as f64);
                            prev = Some(row_idx);
                        }
                    }
                    "LAG" => {
                        let offset = func_args.get(1).and_then(|s| s.trim_start_matches("Int64(").trim_end_matches(')').parse().ok()).unwrap_or(1usize);
                        let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                        if let Some(src_col) = batch.column_by_name(col_name) {
                            if let Some(fa) = src_col.as_any().downcast_ref::<Float64Array>() {
                                for (pos, &ri) in indices.iter().enumerate() {
                                    per_flt[spec_idx][ri] = if pos >= offset {
                                        let pr = indices[pos - offset];
                                        if fa.is_null(pr) { None } else { Some(fa.value(pr)) }
                                    } else { None };
                                }
                            } else if let Some(ia) = src_col.as_any().downcast_ref::<Int64Array>() {
                                for (pos, &ri) in indices.iter().enumerate() {
                                    per_int[spec_idx][ri] = if pos >= offset {
                                        let pr = indices[pos - offset];
                                        if ia.is_null(pr) { None } else { Some(ia.value(pr)) }
                                    } else { None };
                                }
                            }
                        }
                    }
                    "LEAD" => {
                        let offset = func_args.get(1).and_then(|s| s.trim_start_matches("Int64(").trim_end_matches(')').parse().ok()).unwrap_or(1usize);
                        let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                        let len = indices.len();
                        if let Some(src_col) = batch.column_by_name(col_name) {
                            if let Some(fa) = src_col.as_any().downcast_ref::<Float64Array>() {
                                for (pos, &ri) in indices.iter().enumerate() {
                                    per_flt[spec_idx][ri] = if pos + offset < len {
                                        let nr = indices[pos + offset];
                                        if fa.is_null(nr) { None } else { Some(fa.value(nr)) }
                                    } else { None };
                                }
                            } else if let Some(ia) = src_col.as_any().downcast_ref::<Int64Array>() {
                                for (pos, &ri) in indices.iter().enumerate() {
                                    per_int[spec_idx][ri] = if pos + offset < len {
                                        let nr = indices[pos + offset];
                                        if ia.is_null(nr) { None } else { Some(ia.value(nr)) }
                                    } else { None };
                                }
                            }
                        }
                    }
                    "FIRST_VALUE" => {
                        let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                        if let Some(src_col) = batch.column_by_name(col_name) {
                            let fr = indices[0];
                            if let Some(fa) = src_col.as_any().downcast_ref::<Float64Array>() {
                                let v = if fa.is_null(fr) { None } else { Some(fa.value(fr)) };
                                for &ri in &indices { per_flt[spec_idx][ri] = v; }
                            } else if let Some(ia) = src_col.as_any().downcast_ref::<Int64Array>() {
                                let v = if ia.is_null(fr) { None } else { Some(ia.value(fr)) };
                                for &ri in &indices { per_int[spec_idx][ri] = v; }
                            }
                        }
                    }
                    "LAST_VALUE" => {
                        let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                        if let Some(src_col) = batch.column_by_name(col_name) {
                            let lr = indices[indices.len() - 1];
                            if let Some(fa) = src_col.as_any().downcast_ref::<Float64Array>() {
                                let v = if fa.is_null(lr) { None } else { Some(fa.value(lr)) };
                                for &ri in &indices { per_flt[spec_idx][ri] = v; }
                            } else if let Some(ia) = src_col.as_any().downcast_ref::<Int64Array>() {
                                let v = if ia.is_null(lr) { None } else { Some(ia.value(lr)) };
                                for &ri in &indices { per_int[spec_idx][ri] = v; }
                            }
                        }
                    }
                    "SUM" => {
                        let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                        if let Some(src_col) = batch.column_by_name(col_name) {
                            if !order_by.is_empty() {
                                // Running (cumulative) sum when ORDER BY is present
                                if let Some(fa) = src_col.as_any().downcast_ref::<Float64Array>() {
                                    let mut running = 0.0f64;
                                    for &ri in &indices {
                                        if !fa.is_null(ri) { running += fa.value(ri); }
                                        per_flt[spec_idx][ri] = Some(running);
                                    }
                                } else if let Some(ia) = src_col.as_any().downcast_ref::<Int64Array>() {
                                    let mut running = 0i64;
                                    for &ri in &indices {
                                        if !ia.is_null(ri) { running += ia.value(ri); }
                                        per_int[spec_idx][ri] = Some(running);
                                    }
                                }
                            } else {
                                // Total partition sum when no ORDER BY
                                if let Some(fa) = src_col.as_any().downcast_ref::<Float64Array>() {
                                    let total: f64 = indices.iter().filter_map(|&i| if fa.is_null(i) { None } else { Some(fa.value(i)) }).sum();
                                    for &ri in &indices { per_flt[spec_idx][ri] = Some(total); }
                                } else if let Some(ia) = src_col.as_any().downcast_ref::<Int64Array>() {
                                    let total: i64 = indices.iter().filter_map(|&i| if ia.is_null(i) { None } else { Some(ia.value(i)) }).sum();
                                    for &ri in &indices { per_int[spec_idx][ri] = Some(total); }
                                }
                            }
                        }
                    }
                    "RUNNING_SUM" => {
                        let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                        if let Some(src_col) = batch.column_by_name(col_name) {
                            if let Some(fa) = src_col.as_any().downcast_ref::<Float64Array>() {
                                let mut running = 0.0f64;
                                for &ri in &indices { if !fa.is_null(ri) { running += fa.value(ri); } per_flt[spec_idx][ri] = Some(running); }
                            } else if let Some(ia) = src_col.as_any().downcast_ref::<Int64Array>() {
                                let mut running = 0i64;
                                for &ri in &indices { if !ia.is_null(ri) { running += ia.value(ri); } per_int[spec_idx][ri] = Some(running); }
                            }
                        }
                    }
                    "AVG" => {
                        let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                        if let Some(src_col) = batch.column_by_name(col_name) {
                            if let Some(fa) = src_col.as_any().downcast_ref::<Float64Array>() {
                                let vals: Vec<f64> = indices.iter().filter_map(|&i| if fa.is_null(i) { None } else { Some(fa.value(i)) }).collect();
                                let avg = if vals.is_empty() { 0.0 } else { vals.iter().sum::<f64>() / vals.len() as f64 };
                                for &ri in &indices { per_flt[spec_idx][ri] = Some(avg); }
                            } else if let Some(ia) = src_col.as_any().downcast_ref::<Int64Array>() {
                                let vals: Vec<i64> = indices.iter().filter_map(|&i| if ia.is_null(i) { None } else { Some(ia.value(i)) }).collect();
                                let avg = if vals.is_empty() { 0.0 } else { vals.iter().sum::<i64>() as f64 / vals.len() as f64 };
                                for &ri in &indices { per_flt[spec_idx][ri] = Some(avg); }
                            }
                        }
                    }
                    "COUNT" => {
                        let cnt = indices.len() as i64;
                        for &ri in &indices { per_int[spec_idx][ri] = Some(cnt); }
                    }
                    "MIN" => {
                        let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                        if let Some(src_col) = batch.column_by_name(col_name) {
                            if let Some(fa) = src_col.as_any().downcast_ref::<Float64Array>() {
                                let mv = indices.iter().filter_map(|&i| if fa.is_null(i) { None } else { Some(fa.value(i)) }).fold(f64::INFINITY, f64::min);
                                let mv = if mv == f64::INFINITY { None } else { Some(mv) };
                                for &ri in &indices { per_flt[spec_idx][ri] = mv; }
                            } else if let Some(ia) = src_col.as_any().downcast_ref::<Int64Array>() {
                                let mv = indices.iter().filter_map(|&i| if ia.is_null(i) { None } else { Some(ia.value(i)) }).min();
                                for &ri in &indices { per_int[spec_idx][ri] = mv; }
                            }
                        }
                    }
                    "MAX" => {
                        let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                        if let Some(src_col) = batch.column_by_name(col_name) {
                            if let Some(fa) = src_col.as_any().downcast_ref::<Float64Array>() {
                                let mv = indices.iter().filter_map(|&i| if fa.is_null(i) { None } else { Some(fa.value(i)) }).fold(f64::NEG_INFINITY, f64::max);
                                let mv = if mv == f64::NEG_INFINITY { None } else { Some(mv) };
                                for &ri in &indices { per_flt[spec_idx][ri] = mv; }
                            } else if let Some(ia) = src_col.as_any().downcast_ref::<Int64Array>() {
                                let mv = indices.iter().filter_map(|&i| if ia.is_null(i) { None } else { Some(ia.value(i)) }).max();
                                for &ri in &indices { per_int[spec_idx][ri] = mv; }
                            }
                        }
                    }
                    "NTH_VALUE" => {
                        let col_name = func_args.get(0).map(|s| s.trim_matches('"')).unwrap_or("");
                        let n = func_args.get(1).and_then(|s| s.trim_start_matches("Int64(").trim_end_matches(')').parse::<usize>().ok()).unwrap_or(1);
                        if let Some(src_col) = batch.column_by_name(col_name) {
                            if let Some(fa) = src_col.as_any().downcast_ref::<Float64Array>() {
                                let v = if n > 0 && n <= indices.len() {
                                    let nr = indices[n-1]; if fa.is_null(nr) { None } else { Some(fa.value(nr)) }
                                } else { None };
                                for &ri in &indices { per_flt[spec_idx][ri] = v; }
                            } else if let Some(ia) = src_col.as_any().downcast_ref::<Int64Array>() {
                                let v = if n > 0 && n <= indices.len() {
                                    let nr = indices[n-1]; if ia.is_null(nr) { None } else { Some(ia.value(nr)) }
                                } else { None };
                                for &ri in &indices { per_int[spec_idx][ri] = v; }
                            }
                        }
                    }
                    _ => {}
                }
            } // end groups loop
        } // end spec loop

        // Build result with original columns + window function result columns
        let mut result_fields: Vec<Field> = Vec::new();
        let mut result_arrays: Vec<ArrayRef> = Vec::new();
        let mut spec_idx = 0usize;

        for col in &stmt.columns {
            match col {
                SelectColumn::Column(name) => {
                    let col_name = name.trim_matches('"');
                    if let Some(arr) = batch.column_by_name(col_name) {
                        result_fields.push(Field::new(col_name, arr.data_type().clone(), true));
                        result_arrays.push(arr.clone());
                    }
                }
                SelectColumn::ColumnAlias { column, alias } => {
                    let col_name = column.trim_matches('"');
                    if let Some(arr) = batch.column_by_name(col_name) {
                        result_fields.push(Field::new(alias, arr.data_type().clone(), true));
                        result_arrays.push(arr.clone());
                    }
                }
                SelectColumn::All => {
                    for (i, field) in batch.schema().fields().iter().enumerate() {
                        result_fields.push(field.as_ref().clone());
                        result_arrays.push(batch.column(i).clone());
                    }
                }
                SelectColumn::WindowFunction { name, alias, .. } => {
                    let out_name = alias.clone().unwrap_or_else(|| name.to_lowercase());
                    if use_float[spec_idx] {
                        result_fields.push(Field::new(&out_name, ArrowDataType::Float64, true));
                        result_arrays.push(Arc::new(Float64Array::from(per_flt[spec_idx].clone())));
                    } else {
                        result_fields.push(Field::new(&out_name, ArrowDataType::Int64, true));
                        result_arrays.push(Arc::new(Int64Array::from(per_int[spec_idx].clone())));
                    }
                    spec_idx += 1;
                }
                _ => {}
            }
        }

        let schema = Arc::new(Schema::new(result_fields));
        let result = RecordBatch::try_new(schema, result_arrays)
            .map_err(|e| err_data(e.to_string()))?;

        Ok(ApexResult::Data(result))
    }

    /// Compare two array values for sorting
    fn compare_array_values(array: &ArrayRef, a: usize, b: usize) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        
        if array.is_null(a) && array.is_null(b) {
            return Ordering::Equal;
        }
        if array.is_null(a) {
            return Ordering::Greater;
        }
        if array.is_null(b) {
            return Ordering::Less;
        }

        if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
            arr.value(a).cmp(&arr.value(b))
        } else if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
            arr.value(a).partial_cmp(&arr.value(b)).unwrap_or(Ordering::Equal)
        } else if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
            arr.value(a).cmp(arr.value(b))
        } else {
            Ordering::Equal
        }
    }

}
