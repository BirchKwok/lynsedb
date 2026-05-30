// Sorting, aggregation, GROUP BY execution

impl ApexExecutor {

    /// Resolve ORDER BY column names against SELECT list aliases.
    /// Maps expressions like `SUM(amount)` to their output column name (e.g. `total`)
    /// so ORDER BY works correctly when aggregate columns are aliased.
    fn resolve_order_by_cols(
        columns: &[crate::query::SelectColumn],
        order_by: &[crate::query::OrderByClause],
    ) -> Vec<crate::query::OrderByClause> {
        use crate::query::{SelectColumn, AggregateFunc};
        order_by.iter().map(|clause| {
            // Try to match against SELECT aggregate columns
            for sel_col in columns {
                if let SelectColumn::Aggregate { func, column, alias, .. } = sel_col {
                    let fn_str = match func {
                        AggregateFunc::Sum   => "SUM",
                        AggregateFunc::Count => "COUNT",
                        AggregateFunc::Avg   => "AVG",
                        AggregateFunc::Min   => "MIN",
                        AggregateFunc::Max   => "MAX",
                    };
                    let col_part = column.as_deref().unwrap_or("*");
                    let default_name = format!("{}({})", fn_str, col_part);
                    if default_name.eq_ignore_ascii_case(&clause.column) {
                        let out_name = if let Some(a) = alias {
                            a.clone()
                        } else {
                            default_name
                        };
                        return crate::query::OrderByClause {
                            column: out_name,
                            descending: clause.descending,
                            nulls_first: clause.nulls_first,
                            expr: None,
                        };
                    }
                }
            }
            clause.clone()
        }).collect()
    }

    /// Apply ORDER BY clause
    /// OPTIMIZATION: Use parallel sort for large datasets (>100K rows) using Rayon
    fn apply_order_by(
        batch: &RecordBatch,
        order_by: &[crate::query::OrderByClause],
    ) -> io::Result<RecordBatch> {
        use arrow::compute::SortColumn;
        use rayon::prelude::*;

        // Pre-evaluate expression ORDER BY columns (e.g. ORDER BY array_distance(...))
        let batch_cow = if order_by.iter().any(|o| {
            o.expr.is_some() && {
                let cn = o.column.trim_matches('"');
                let cn = cn.rfind('.').map_or(cn, |p| &cn[p+1..]);
                batch.column_by_name(cn).is_none()
            }
        }) {
            let mut extra: Vec<(String, arrow::array::ArrayRef)> = Vec::new();
            for o in order_by {
                if let Some(ref expr) = o.expr {
                    let cn = o.column.trim_matches('"');
                    let cn = cn.rfind('.').map_or(cn, |p| &cn[p+1..]);
                    if batch.column_by_name(cn).is_none() {
                        let arr = Self::evaluate_expr_to_array(batch, expr)?;
                        extra.push((o.column.clone(), arr));
                    }
                }
            }
            if extra.is_empty() {
                std::borrow::Cow::Borrowed(batch)
            } else {
                let mut fields: Vec<arrow::datatypes::Field> =
                    batch.schema().fields().iter().map(|f| (**f).clone()).collect();
                let mut cols = batch.columns().to_vec();
                for (name, arr) in extra {
                    fields.push(arrow::datatypes::Field::new(&name, arr.data_type().clone(), true));
                    cols.push(arr);
                }
                let nb = RecordBatch::try_new(Arc::new(arrow::datatypes::Schema::new(fields)), cols)
                    .map_err(|e| err_data(e.to_string()))?;
                std::borrow::Cow::Owned(nb)
            }
        } else {
            std::borrow::Cow::Borrowed(batch)
        };
        let batch: &RecordBatch = &batch_cow;

        let sort_columns: Vec<SortColumn> = order_by
            .iter()
            .filter_map(|clause| {
                let col_name = clause.column.trim_matches('"');
                // Strip table prefix if present (e.g., "u.tier" -> "tier")
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                batch.column_by_name(actual_col).map(|col| SortColumn {
                    values: col.clone(),
                    options: Some(SortOptions {
                        descending: clause.descending,
                        nulls_first: clause.nulls_first.unwrap_or(clause.descending),
                    }),
                })
            })
            .collect();

        if sort_columns.is_empty() {
            return Ok(batch.clone());
        }

        let num_rows = batch.num_rows();
        
        // For large datasets (>50K rows), use parallel sort with Rayon
        let indices = if num_rows > 50_000 && sort_columns.len() == 1 {
            // Single column sort - use parallel sort for better performance
            Self::parallel_sort_indices(batch, order_by)?
        } else {
            // Multi-column or small dataset - use Arrow's lexsort
            compute::lexsort_to_indices(&sort_columns, None)
                .map_err(|e| err_data( e.to_string()))?
        };

        // OPTIMIZATION: Use SIMD-accelerated take for better performance
        let indices_array = &indices;
        let columns: Vec<ArrayRef> = if num_rows > 100_000 {
            use rayon::prelude::*;
            use crate::query::simd_take::optimized_take;
            batch
                .columns()
                .par_iter()
                .map(|col| Arc::new(optimized_take(col, indices_array)) as ArrayRef)
                .collect()
        } else {
            use crate::query::simd_take::optimized_take;
            batch
                .columns()
                .iter()
                .map(|col| Arc::new(optimized_take(col, indices_array)) as ArrayRef)
                .collect()
        };

        RecordBatch::try_new(batch.schema(), columns)
            .map_err(|e| err_data( e.to_string()))
    }
    
    /// Parallel sort for single-column ORDER BY using Rayon
    /// Uses indices with custom comparator for better memory efficiency
    fn parallel_sort_indices(
        batch: &RecordBatch,
        order_by: &[crate::query::OrderByClause],
    ) -> io::Result<arrow::array::UInt32Array> {
        use rayon::prelude::*;
        
        let clause = &order_by[0];
        let col_name = clause.column.trim_matches('"');
        let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
            &col_name[dot_pos + 1..]
        } else {
            col_name
        };
        
        let col = batch.column_by_name(actual_col)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column {} not found", actual_col)))?;
        
        let num_rows = batch.num_rows();
        
        // For Int64 - use fast parallel sort with custom comparator
        if let Some(int_arr) = col.as_any().downcast_ref::<Int64Array>() {
            let descending = clause.descending;
            
            // Check if we can use counting sort (range is limited)
            if !descending {
                let (min_val, max_val) = {
                    let mut min = i64::MAX;
                    let mut max = i64::MIN;
                    for i in 0..num_rows {
                        if !int_arr.is_null(i) {
                            let v = int_arr.value(i);
                            min = min.min(v);
                            max = max.max(v);
                        }
                    }
                    (min, max)
                };
                
                let range = (max_val - min_val + 1) as usize;
                // Use counting sort if range is reasonable (< 5M values)
                if range <= 5_000_000 && range > 0 {
                    return Self::counting_sort_indices(int_arr, min_val, max_val, num_rows);
                }
            }
            
            // Create (value, index) pairs and sort in parallel
            let mut pairs: Vec<(i64, usize)> = (0..num_rows)
                .map(|i| {
                    let val = if int_arr.is_null(i) { 
                        if descending { i64::MIN } else { i64::MAX }
                    } else { 
                        int_arr.value(i) 
                    };
                    (val, i)
                })
                .collect();
            
            // Parallel sort using unstable sort for better performance
            if descending {
                pairs.par_sort_unstable_by(|a, b| b.0.cmp(&a.0));
            } else {
                pairs.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
            }
            
            let sorted_indices: Vec<u32> = pairs.iter().map(|(_, idx)| *idx as u32).collect();
            return Ok(arrow::array::UInt32Array::from(sorted_indices));
        }
        
        // For Float64 - use parallel sort
        if let Some(float_arr) = col.as_any().downcast_ref::<Float64Array>() {
            let descending = clause.descending;
            
            let mut pairs: Vec<(f64, usize)> = (0..num_rows)
                .map(|i| {
                    let val = if float_arr.is_null(i) { 
                        if descending { f64::NEG_INFINITY } else { f64::INFINITY }
                    } else { 
                        float_arr.value(i) 
                    };
                    (val, i)
                })
                .collect();
            
            if descending {
                pairs.par_sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                pairs.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
            
            let sorted_indices: Vec<u32> = pairs.iter().map(|(_, idx)| *idx as u32).collect();
            return Ok(arrow::array::UInt32Array::from(sorted_indices));
        }
        
        // Fallback to Arrow's lexsort for other types
        use arrow::compute::{SortColumn, SortOptions};
        let sort_columns: Vec<_> = order_by
            .iter()
            .filter_map(|clause| {
                let col_name = clause.column.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                batch.column_by_name(actual_col).map(|col| SortColumn {
                    values: col.clone(),
                    options: Some(SortOptions {
                        descending: clause.descending,
                        nulls_first: clause.nulls_first.unwrap_or(clause.descending),
                    }),
                })
            })
            .collect();
        
        compute::lexsort_to_indices(&sort_columns, None)
            .map_err(|e| err_data( e.to_string()))
    }
    
    /// Counting sort for integer arrays with limited range
    /// O(n + range) complexity, much faster than comparison sort for small ranges
    fn counting_sort_indices(
        arr: &Int64Array,
        min_val: i64,
        max_val: i64,
        num_rows: usize,
    ) -> io::Result<arrow::array::UInt32Array> {
        let range = (max_val - min_val + 1) as usize;
        let mut counts: Vec<usize> = vec![0; range + 1]; // +1 for nulls
        
        // Count occurrences
        for i in 0..num_rows {
            if arr.is_null(i) {
                counts[range] += 1;
            } else {
                let idx = (arr.value(i) - min_val) as usize;
                counts[idx] += 1;
            }
        }
        
        // Compute prefix sums for output positions
        let mut positions: Vec<usize> = vec![0; range + 1];
        let mut total = 0;
        for i in 0..range {
            positions[i] = total;
            total += counts[i];
        }
        positions[range] = total; // nulls at the end
        
        // Build result indices
        let mut result: Vec<u32> = vec![0; num_rows];
        for i in 0..num_rows {
            let pos = if arr.is_null(i) {
                positions[range]
            } else {
                let idx = (arr.value(i) - min_val) as usize;
                positions[idx]
            };
            result[pos] = i as u32;
            if arr.is_null(i) {
                positions[range] += 1;
            } else {
                positions[(arr.value(i) - min_val) as usize] += 1;
            }
        }
        
        Ok(arrow::array::UInt32Array::from(result))
    }

    /// Apply ORDER BY with top-k optimization (heap sort for LIMIT queries)
    /// When k is Some, uses partial sort O(n log k) instead of full sort O(n log n)
    /// OPTIMIZATION: Pre-downcast columns once to avoid repeated dynamic dispatch
    fn apply_order_by_topk(
        batch: &RecordBatch,
        order_by: &[crate::query::OrderByClause],
        k: Option<usize>,
    ) -> io::Result<RecordBatch> {
        use std::cmp::Ordering;

        // Pre-evaluate any expression-based ORDER BY columns not yet in the batch.
        // e.g. ORDER BY array_distance(vec, [1,2,3]) — expression not in SELECT output yet.
        let batch = if order_by.iter().any(|o| {
            o.expr.is_some() && {
                let cn = o.column.trim_matches('"');
                let cn = cn.rfind('.').map_or(cn, |p| &cn[p+1..]);
                batch.column_by_name(cn).is_none()
            }
        }) {
            let mut new_cols: Vec<(String, arrow::array::ArrayRef)> = Vec::new();
            for o in order_by {
                if let Some(ref expr) = o.expr {
                    let cn = o.column.trim_matches('"');
                    let cn = cn.rfind('.').map_or(cn, |p| &cn[p+1..]);
                    if batch.column_by_name(cn).is_none() {
                        let arr = Self::evaluate_expr_to_array(batch, expr)?;
                        new_cols.push((o.column.clone(), arr));
                    }
                }
            }
            if new_cols.is_empty() {
                std::borrow::Cow::Borrowed(batch)
            } else {
                let mut fields: Vec<arrow::datatypes::Field> =
                    batch.schema().fields().iter().map(|f| (**f).clone()).collect();
                let mut cols: Vec<arrow::array::ArrayRef> =
                    batch.columns().to_vec();
                for (name, arr) in new_cols {
                    fields.push(arrow::datatypes::Field::new(
                        &name,
                        arr.data_type().clone(),
                        true,
                    ));
                    cols.push(arr);
                }
                let schema = Arc::new(arrow::datatypes::Schema::new(fields));
                let nb = RecordBatch::try_new(schema, cols)
                    .map_err(|e| err_data(e.to_string()))?;
                std::borrow::Cow::Owned(nb)
            }
        } else {
            std::borrow::Cow::Borrowed(batch)
        };
        let batch: &RecordBatch = &batch;

        // If no limit or limit >= rows, use standard sort
        let num_rows = batch.num_rows();
        if k.is_none() || k.unwrap() >= num_rows {
            return Self::apply_order_by(batch, order_by);
        }
        let k = k.unwrap();

        if k == 0 {
            return Ok(RecordBatch::new_empty(batch.schema()));
        }

        // Pre-downcast sort columns for fast comparison (avoid repeated dynamic dispatch)
        enum TypedSortCol<'a> {
            Int64(&'a Int64Array, bool),   // (array, descending)
            Float64(&'a Float64Array, bool),
            String(&'a StringArray, bool),
            Other(&'a ArrayRef, bool),
        }
        
        let typed_sort_cols: Vec<TypedSortCol> = order_by
            .iter()
            .filter_map(|clause| {
                let col_name = clause.column.trim_matches('"');
                // Try exact name first; only strip table prefix if the prefix part is a
                // simple identifier (no parens/brackets) — avoids mis-splitting float literals.
                let actual_col = if batch.column_by_name(col_name).is_some() {
                    col_name
                } else if let Some(dot_pos) = col_name.rfind('.') {
                    let prefix = &col_name[..dot_pos];
                    if prefix.chars().all(|c| c.is_alphanumeric() || c == '_') {
                        &col_name[dot_pos + 1..]
                    } else {
                        col_name
                    }
                } else {
                    col_name
                };
                batch.column_by_name(actual_col).map(|col| {
                    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                        TypedSortCol::Int64(arr, clause.descending)
                    } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                        TypedSortCol::Float64(arr, clause.descending)
                    } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                        TypedSortCol::String(arr, clause.descending)
                    } else {
                        TypedSortCol::Other(col, clause.descending)
                    }
                })
            })
            .collect();

        if typed_sort_cols.is_empty() {
            return Ok(batch.clone());
        }

        // Fast comparison using pre-downcast columns
        let compare_rows = |a: usize, b: usize| -> Ordering {
            for col in &typed_sort_cols {
                let ord = match col {
                    TypedSortCol::Int64(arr, desc) => {
                        let a_null = arr.is_null(a);
                        let b_null = arr.is_null(b);
                        let ord = if a_null && b_null {
                            Ordering::Equal
                        } else if a_null {
                            Ordering::Greater
                        } else if b_null {
                            Ordering::Less
                        } else {
                            arr.value(a).cmp(&arr.value(b))
                        };
                        if *desc { ord.reverse() } else { ord }
                    }
                    TypedSortCol::Float64(arr, desc) => {
                        let a_null = arr.is_null(a);
                        let b_null = arr.is_null(b);
                        let ord = if a_null && b_null {
                            Ordering::Equal
                        } else if a_null {
                            Ordering::Greater
                        } else if b_null {
                            Ordering::Less
                        } else {
                            arr.value(a).partial_cmp(&arr.value(b)).unwrap_or(Ordering::Equal)
                        };
                        if *desc { ord.reverse() } else { ord }
                    }
                    TypedSortCol::String(arr, desc) => {
                        let a_null = arr.is_null(a);
                        let b_null = arr.is_null(b);
                        let ord = if a_null && b_null {
                            Ordering::Equal
                        } else if a_null {
                            Ordering::Greater
                        } else if b_null {
                            Ordering::Less
                        } else {
                            arr.value(a).cmp(arr.value(b))
                        };
                        if *desc { ord.reverse() } else { ord }
                    }
                    TypedSortCol::Other(arr, desc) => {
                        let ord = Self::compare_array_values(arr, a, b);
                        if *desc { ord.reverse() } else { ord }
                    }
                };
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            Ordering::Equal
        };

        // OPTIMIZATION: For small k, use heap-based top-k (O(n log k))
        // For larger k, use partial sort (O(n + k log k))
        let indices: Vec<usize> = if k <= 100 {
            // Heap-based approach for small k - maintains a max-heap of size k
            use std::collections::BinaryHeap;
            
            // Wrapper for reverse comparison (we want min-heap behavior for top-k)
            struct HeapItem(usize);
            
            impl PartialEq for HeapItem {
                fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
            }
            impl Eq for HeapItem {}
            
            // Create comparison closure that captures typed_sort_cols
            let mut heap: BinaryHeap<(std::cmp::Reverse<usize>, usize)> = BinaryHeap::with_capacity(k + 1);
            
            // Simple approach: store (score, index) where score is computed once
            // For numeric DESC sorting, we can use the value directly as score
            if typed_sort_cols.len() == 1 {
                match &typed_sort_cols[0] {
                    TypedSortCol::Float64(arr, true) => {
                        // DESC sort on float - use value as score, keep k largest
                        let mut top_k: Vec<(i64, usize)> = Vec::with_capacity(k);
                        for i in 0..num_rows {
                            let score = if arr.is_null(i) { i64::MIN } else { 
                                (arr.value(i) * 1e10) as i64 // Scale to preserve precision
                            };
                            if top_k.len() < k {
                                top_k.push((score, i));
                                if top_k.len() == k {
                                    top_k.sort_by(|a, b| b.0.cmp(&a.0));
                                }
                            } else if score > top_k[k-1].0 {
                                top_k[k-1] = (score, i);
                                // Bubble up
                                let mut j = k - 1;
                                while j > 0 && top_k[j].0 > top_k[j-1].0 {
                                    top_k.swap(j, j-1);
                                    j -= 1;
                                }
                            }
                        }
                        top_k.iter().map(|(_, i)| *i).collect()
                    }
                    TypedSortCol::Int64(arr, true) => {
                        // DESC sort on int - use value as score, keep k largest
                        let mut top_k: Vec<(i64, usize)> = Vec::with_capacity(k);
                        for i in 0..num_rows {
                            let score = if arr.is_null(i) { i64::MIN } else { arr.value(i) };
                            if top_k.len() < k {
                                top_k.push((score, i));
                                if top_k.len() == k {
                                    top_k.sort_by(|a, b| b.0.cmp(&a.0));
                                }
                            } else if score > top_k[k-1].0 {
                                top_k[k-1] = (score, i);
                                let mut j = k - 1;
                                while j > 0 && top_k[j].0 > top_k[j-1].0 {
                                    top_k.swap(j, j-1);
                                    j -= 1;
                                }
                            }
                        }
                        top_k.iter().map(|(_, i)| *i).collect()
                    }
                    _ => {
                        // Fall back to generic approach
                        let mut all_indices: Vec<usize> = (0..num_rows).collect();
                        all_indices.select_nth_unstable_by(k - 1, |&a, &b| compare_rows(a, b));
                        all_indices.truncate(k);
                        all_indices.sort_by(|&a, &b| compare_rows(a, b));
                        all_indices
                    }
                }
            } else {
                // Multi-column sort - use generic approach
                let mut all_indices: Vec<usize> = (0..num_rows).collect();
                all_indices.select_nth_unstable_by(k - 1, |&a, &b| compare_rows(a, b));
                all_indices.truncate(k);
                all_indices.sort_by(|&a, &b| compare_rows(a, b));
                all_indices
            }
        } else {
            // Larger k - use partial sort (more efficient for k > 100)
            let mut all_indices: Vec<usize> = (0..num_rows).collect();
            if k < num_rows {
                all_indices.select_nth_unstable_by(k - 1, |&a, &b| compare_rows(a, b));
                all_indices.truncate(k);
            }
            all_indices.sort_by(|&a, &b| compare_rows(a, b));
            all_indices
        };

        // Take rows by indices
        let indices_array = arrow::array::UInt64Array::from(
            indices.iter().map(|&i| i as u64).collect::<Vec<_>>()
        );

        let columns: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .map(|col| compute::take(col, &indices_array, None))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| err_data( e.to_string()))?;

        RecordBatch::try_new(batch.schema(), columns)
            .map_err(|e| err_data( e.to_string()))
    }

    /// Apply LIMIT and OFFSET
    fn apply_limit_offset(
        batch: &RecordBatch,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> io::Result<RecordBatch> {
        let start = offset.unwrap_or(0);
        let end = limit
            .map(|l| (start + l).min(batch.num_rows()))
            .unwrap_or(batch.num_rows());

        if start >= batch.num_rows() {
            return Ok(RecordBatch::new_empty(batch.schema()));
        }

        let length = end - start;
        Ok(batch.slice(start, length))
    }

    /// Project columns according to SELECT list
    fn apply_projection(
        batch: &RecordBatch,
        columns: &[SelectColumn],
    ) -> io::Result<RecordBatch> {
        Self::apply_projection_with_storage(batch, columns, None)
    }
    
    /// Project columns according to SELECT list (with optional storage path for scalar subqueries)
    fn apply_projection_with_storage(
        batch: &RecordBatch,
        columns: &[SelectColumn],
        storage_path: Option<&Path>,
    ) -> io::Result<RecordBatch> {
        // Handle simple SELECT * (only * with no other columns)
        if columns.len() == 1 && matches!(columns[0], SelectColumn::All) {
            return Ok(batch.clone());
        }

        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();
        let mut added_columns: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Collect all explicitly named columns to exclude from * expansion
        let mut explicit_cols: std::collections::HashSet<String> = std::collections::HashSet::new();
        for col in columns {
            match col {
                SelectColumn::Column(name) => {
                    let col_name = name.trim_matches('"');
                    let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                        &col_name[dot_pos + 1..]
                    } else {
                        col_name
                    };
                    explicit_cols.insert(actual_col.to_string());
                }
                SelectColumn::ColumnAlias { column, .. } => {
                    let col_name = column.trim_matches('"');
                    explicit_cols.insert(col_name.to_string());
                }
                _ => {}
            }
        }

        // Process columns in order - preserving user-specified order
        for col in columns {
            match col {
                SelectColumn::Column(name) => {
                    let col_name = name.trim_matches('"');
                    // Try full qualified name first (e.g., 'b.name' for self-join),
                    // then fall back to bare name.
                    let bare_col = if let Some(dot_pos) = col_name.rfind('.') {
                        &col_name[dot_pos + 1..]
                    } else {
                        col_name
                    };
                    let (out_name, array) = if let Some(arr) = batch.column_by_name(col_name) {
                        (bare_col, arr)
                    } else if let Some(arr) = batch.column_by_name(bare_col) {
                        (bare_col, arr)
                    } else {
                        continue;
                    };
                    if !added_columns.contains(out_name) {
                        fields.push(Field::new(out_name, array.data_type().clone(), true));
                        arrays.push(array.clone());
                        added_columns.insert(out_name.to_string());
                    }
                }
                SelectColumn::ColumnAlias { column, alias } => {
                    let col_name = column.trim_matches('"');
                    // Try full qualified name first (for self-join disambiguation),
                    // then bare name.
                    let bare_col = if let Some(dot_pos) = col_name.rfind('.') {
                        &col_name[dot_pos + 1..]
                    } else {
                        col_name
                    };
                    let array = batch.column_by_name(col_name)
                        .or_else(|| batch.column_by_name(bare_col));
                    if let Some(array) = array {
                        fields.push(Field::new(alias, array.data_type().clone(), true));
                        arrays.push(array.clone());
                        added_columns.insert(alias.clone());
                    }
                }
                SelectColumn::Expression { expr, alias } => {
                    // Use storage-aware evaluation for expressions that may contain scalar subqueries
                    let array = if let Some(path) = storage_path {
                        Self::evaluate_expr_to_array_with_storage(batch, expr, path)?
                    } else {
                        Self::evaluate_expr_to_array(batch, expr)?
                    };
                    let output_name = alias.clone().unwrap_or_else(|| "expr".to_string());
                    fields.push(Field::new(&output_name, array.data_type().clone(), true));
                    arrays.push(array);
                }
                SelectColumn::All => {
                    // Add all columns from batch EXCEPT:
                    // 1. _id (always skip here - it will be added at its explicit position if requested)
                    // 2. Columns already added (from explicit references before *)
                    for (i, field) in batch.schema().fields().iter().enumerate() {
                        let col_name = field.name();
                        if col_name == "_id" {
                            continue;
                        }
                        if !added_columns.contains(col_name) {
                            fields.push(field.as_ref().clone());
                            arrays.push(batch.column(i).clone());
                            added_columns.insert(col_name.clone());
                        }
                    }
                }
                SelectColumn::AllExclude(exclude) => {
                    for (i, field) in batch.schema().fields().iter().enumerate() {
                        let col_name = field.name();
                        if col_name == "_id" || exclude.iter().any(|e| e == col_name) {
                            continue;
                        }
                        if !added_columns.contains(col_name) {
                            fields.push(field.as_ref().clone());
                            arrays.push(batch.column(i).clone());
                            added_columns.insert(col_name.clone());
                        }
                    }
                }
                SelectColumn::AllReplace(replacements) => {
                    let replace_cols: Vec<&str> = replacements.iter().map(|(_, col)| col.as_str()).collect();
                    for (i, field) in batch.schema().fields().iter().enumerate() {
                        let col_name = field.name();
                        if col_name == "_id" || replace_cols.iter().any(|c| *c == col_name) {
                            continue;
                        }
                        if !added_columns.contains(col_name) {
                            fields.push(field.as_ref().clone());
                            arrays.push(batch.column(i).clone());
                            added_columns.insert(col_name.clone());
                        }
                    }
                    for (expr, alias) in replacements {
                        let array = if let Some(path) = storage_path {
                            Self::evaluate_expr_to_array_with_storage(batch, expr, path)?
                        } else {
                            Self::evaluate_expr_to_array(batch, expr)?
                        };
                        fields.push(Field::new(alias, array.data_type().clone(), true));
                        arrays.push(array);
                    }
                }
                SelectColumn::Columns(pattern) => {
                    let re = regex::Regex::new(pattern).map_err(|e| {
                        io::Error::new(io::ErrorKind::InvalidInput, format!("Invalid regex pattern '{}': {}", pattern, e))
                    })?;
                    for (i, field) in batch.schema().fields().iter().enumerate() {
                        let col_name = field.name();
                        if col_name == "_id" {
                            continue;
                        }
                        if re.is_match(col_name) && !added_columns.contains(col_name) {
                            fields.push(field.as_ref().clone());
                            arrays.push(batch.column(i).clone());
                            added_columns.insert(col_name.clone());
                        }
                    }
                }
                SelectColumn::Aggregate { .. } => {
                    // Aggregates are handled separately
                }
                SelectColumn::WindowFunction { .. } => {
                    // Window functions not yet supported
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays)
            .map_err(|e| err_data( e.to_string()))
    }

    /// Execute aggregation query
    fn execute_aggregation(
        batch: &RecordBatch,
        stmt: &SelectStatement,
    ) -> io::Result<ApexResult> {
        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, distinct, alias } = col {
                let (field, array) = Self::compute_aggregate(batch, func, column, *distinct, alias)?;
                fields.push(field);
                arrays.push(array);
            }
        }

        if fields.is_empty() {
            return Ok(ApexResult::Scalar(batch.num_rows() as i64));
        }

        let schema = Arc::new(Schema::new(fields));
        let result = RecordBatch::try_new(schema, arrays)
            .map_err(|e| err_data( e.to_string()))?;

        Ok(ApexResult::Data(result))
    }

    /// Compute a single aggregate function
    fn compute_aggregate(
        batch: &RecordBatch,
        func: &crate::query::AggregateFunc,
        column: &Option<String>,
        distinct: bool,
        alias: &Option<String>,
    ) -> io::Result<(Field, ArrayRef)> {
        use crate::query::AggregateFunc;
        use ahash::AHashSet;

        let fn_name = match func { AggregateFunc::Count => "COUNT", AggregateFunc::Sum => "SUM", AggregateFunc::Avg => "AVG", AggregateFunc::Min => "MIN", AggregateFunc::Max => "MAX" };
        let output_name = alias.clone().unwrap_or_else(|| if let Some(col) = column { format!("{}({})", fn_name, col) } else { format!("{}(*)", fn_name) });

        match func {
            AggregateFunc::Count => {
                let count = if let Some(col_name) = column {
                    // Treat "*" and numeric constants (like "1") as COUNT(*) - count all rows
                    if col_name == "*" || col_name.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                        batch.num_rows() as i64
                    } else if let Some(array) = batch.column_by_name(col_name) {
                        if distinct {
                            // COUNT(DISTINCT column) - count unique non-null values
                            // OPTIMIZATION: Use AHashSet instead of std HashSet for faster hashing
                            if let Some(int_arr) = array.as_any().downcast_ref::<Int64Array>() {
                                let unique: AHashSet<i64> = int_arr.iter().filter_map(|v| v).collect();
                                unique.len() as i64
                            } else if let Some(str_arr) = array.as_any().downcast_ref::<StringArray>() {
                                let unique: AHashSet<&str> = (0..str_arr.len())
                                    .filter(|&i| !str_arr.is_null(i))
                                    .map(|i| str_arr.value(i))
                                    .collect();
                                unique.len() as i64
                            } else if let Some(float_arr) = array.as_any().downcast_ref::<Float64Array>() {
                                let unique: AHashSet<u64> = float_arr.iter()
                                    .filter_map(|v| v.map(|f| f.to_bits()))
                                    .collect();
                                unique.len() as i64
                            } else {
                                (array.len() - array.null_count()) as i64
                            }
                        } else {
                            (array.len() - array.null_count()) as i64
                        }
                    } else {
                        0
                    }
                } else {
                    batch.num_rows() as i64
                };
                Ok((
                    Field::new(&output_name, ArrowDataType::Int64, false),
                    Arc::new(Int64Array::from(vec![count])),
                ))
            }
            AggregateFunc::Sum => {
                let col_name = column.as_ref()
                    .ok_or_else(|| err_input( "SUM requires column"))?;
                let array = batch.column_by_name(col_name)
                    .ok_or_else(|| err_not_found(format!("Column: {}", col_name)))?;

                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    // SIMD-optimized sum for non-null arrays
                    let sum = if int_array.null_count() == 0 {
                        simd_sum_i64(int_array.values())
                    } else {
                        int_array.iter().filter_map(|v| v).sum()
                    };
                    Ok((
                        Field::new(&output_name, ArrowDataType::Int64, false),
                        Arc::new(Int64Array::from(vec![sum])),
                    ))
                } else if let Some(uint_array) = array.as_any().downcast_ref::<UInt64Array>() {
                    let sum: i64 = uint_array.iter().filter_map(|v| v).map(|v| v as i64).sum();
                    Ok((
                        Field::new(&output_name, ArrowDataType::Int64, false),
                        Arc::new(Int64Array::from(vec![sum])),
                    ))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    // SIMD-optimized sum for non-null arrays
                    let sum = if float_array.null_count() == 0 {
                        simd_sum_f64(float_array.values())
                    } else {
                        float_array.iter().filter_map(|v| v).sum()
                    };
                    Ok((
                        Field::new(&output_name, ArrowDataType::Float64, false),
                        Arc::new(Float64Array::from(vec![sum])),
                    ))
                } else {
                    Err(err_data( "SUM requires numeric column"))
                }
            }
            AggregateFunc::Avg => {
                let col_name = column.as_ref()
                    .ok_or_else(|| err_input( "AVG requires column"))?;
                let array = batch.column_by_name(col_name)
                    .ok_or_else(|| err_not_found(format!("Column: {}", col_name)))?;

                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    // SIMD-optimized AVG for non-null arrays
                    let (sum, count) = if int_array.null_count() == 0 {
                        (simd_sum_i64(int_array.values()), int_array.len())
                    } else {
                        let values: Vec<i64> = int_array.iter().filter_map(|v| v).collect();
                        (values.iter().sum::<i64>(), values.len())
                    };
                    let avg = if count == 0 { 0.0 } else { sum as f64 / count as f64 };
                    Ok((
                        Field::new(&output_name, ArrowDataType::Float64, false),
                        Arc::new(Float64Array::from(vec![avg])),
                    ))
                } else if let Some(uint_array) = array.as_any().downcast_ref::<UInt64Array>() {
                    let values: Vec<u64> = uint_array.iter().filter_map(|v| v).collect();
                    let avg = if values.is_empty() { 0.0 } else { values.iter().sum::<u64>() as f64 / values.len() as f64 };
                    Ok((
                        Field::new(&output_name, ArrowDataType::Float64, false),
                        Arc::new(Float64Array::from(vec![avg])),
                    ))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    // SIMD-optimized AVG for non-null arrays
                    let (sum, count) = if float_array.null_count() == 0 {
                        (simd_sum_f64(float_array.values()), float_array.len())
                    } else {
                        let values: Vec<f64> = float_array.iter().filter_map(|v| v).collect();
                        (values.iter().sum::<f64>(), values.len())
                    };
                    let avg = if count == 0 { 0.0 } else { sum / count as f64 };
                    Ok((
                        Field::new(&output_name, ArrowDataType::Float64, false),
                        Arc::new(Float64Array::from(vec![avg])),
                    ))
                } else {
                    Err(err_data( "AVG requires numeric column"))
                }
            }
            AggregateFunc::Min | AggregateFunc::Max => {
                let is_min = matches!(func, AggregateFunc::Min);
                let fn_name = if is_min { "MIN" } else { "MAX" };
                let col_name = column.as_ref()
                    .ok_or_else(|| err_input( format!("{} requires column", fn_name)))?;
                let array = batch.column_by_name(col_name)
                    .ok_or_else(|| err_not_found(format!("Column: {}", col_name)))?;

                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    let val = if is_min { compute::min(int_array) } else { compute::max(int_array) };
                    Ok((Field::new(&output_name, ArrowDataType::Int64, true), Arc::new(Int64Array::from(vec![val]))))
                } else if let Some(uint_array) = array.as_any().downcast_ref::<UInt64Array>() {
                    let val = if is_min { compute::min(uint_array) } else { compute::max(uint_array) }.map(|v| v as i64);
                    Ok((Field::new(&output_name, ArrowDataType::Int64, true), Arc::new(Int64Array::from(vec![val]))))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    let val = if is_min { compute::min(float_array) } else { compute::max(float_array) };
                    Ok((Field::new(&output_name, ArrowDataType::Float64, true), Arc::new(Float64Array::from(vec![val]))))
                } else {
                    Err(err_data( format!("{} requires numeric column", fn_name)))
                }
            }
        }
    }

    /// Execute GROUP BY aggregation query
    /// OPTIMIZATION: Uses vectorized execution engine for maximum performance
    fn execute_group_by(batch: &RecordBatch, stmt: &SelectStatement) -> io::Result<ApexResult> {
        if stmt.group_by.is_empty() {
            return Err(err_input( "GROUP BY requires at least one column"));
        }

        // Materialize expression-based GROUP BY columns (e.g., YEAR(date), MONTH(ts))
        let batch = Self::materialize_group_by_exprs(batch, stmt)?;

        // Build group keys - strip table prefix if present (e.g., "u.tier" -> "tier")
        let group_cols: Vec<String> = stmt.group_by.iter().map(|s| {
            let trimmed = s.trim_matches('"');
            if let Some(dot_pos) = trimmed.rfind('.') {
                trimmed[dot_pos + 1..].to_string()
            } else {
                trimmed.to_string()
            }
        }).collect();

        // If HAVING references aggregates that are not in the SELECT list (e.g.
        // "SELECT city … HAVING COUNT(*) > 1"), inject them as extra SELECT columns
        // so every GROUP-BY sub-path can materialise the value for filter evaluation.
        // We strip the extra columns from the final result after HAVING is applied.
        let select_col_count = stmt.columns.len();
        let extra_agg_count;
        let owned_stmt: SelectStatement;
        let effective_stmt: &SelectStatement;
        if let Some(having_expr) = &stmt.having {
            let extras = Self::collect_having_extra_aggs(having_expr, &stmt.columns);
            if !extras.is_empty() {
                extra_agg_count = extras.len();
                let mut s = stmt.clone();
                for (func, col) in extras {
                    use crate::query::AggregateFunc;
                    let fn_name = match func {
                        AggregateFunc::Count => "COUNT",
                        AggregateFunc::Sum   => "SUM",
                        AggregateFunc::Avg   => "AVG",
                        AggregateFunc::Min   => "MIN",
                        AggregateFunc::Max   => "MAX",
                    };
                    let alias = format!("{}({})", fn_name, col.as_deref().unwrap_or("*"));
                    s.columns.push(crate::query::SelectColumn::Aggregate {
                        func,
                        column: col,
                        distinct: false,
                        alias: Some(alias),
                    });
                }
                owned_stmt = s;
                effective_stmt = &owned_stmt;
            } else {
                extra_agg_count = 0;
                effective_stmt = stmt;
                owned_stmt = stmt.clone(); // unused but required for lifetime
            }
        } else {
            extra_agg_count = 0;
            effective_stmt = stmt;
            owned_stmt = stmt.clone(); // unused but required for lifetime
        }

        // Check if we can use fast path: only simple aggregates (COUNT, SUM, AVG, MIN, MAX)
        // without DISTINCT, expressions, or HAVING that needs row access
        let can_use_incremental = Self::can_use_incremental_aggregation(effective_stmt);
        
        let mut result = if can_use_incremental {
            // Try vectorized execution for single-column GROUP BY
            if group_cols.len() == 1 {
                if let Ok(r) = Self::execute_group_by_vectorized(&batch, effective_stmt, &group_cols[0]) {
                    r
                } else {
                    Self::execute_group_by_incremental(&batch, effective_stmt, &group_cols)?
                }
            } else {
                Self::execute_group_by_incremental(&batch, effective_stmt, &group_cols)?
            }
        } else {
            // Fall back to full row-index based aggregation for complex cases
            Self::execute_group_by_with_indices(&batch, effective_stmt, &group_cols)?
        };

        // Strip the extra HAVING-only aggregate columns we injected above
        if extra_agg_count > 0 {
            if let ApexResult::Data(ref rb) = result {
                let keep = select_col_count.min(rb.num_columns());
                let new_schema = Arc::new(Schema::new(
                    rb.schema().fields()[..keep].iter().map(|f| f.as_ref().clone()).collect::<Vec<_>>(),
                ));
                let new_arrays: Vec<ArrayRef> = (0..keep).map(|i| rb.column(i).clone()).collect();
                match RecordBatch::try_new(new_schema, new_arrays) {
                    Ok(trimmed) => result = ApexResult::Data(trimmed),
                    Err(e) => return Err(err_data(e.to_string())),
                }
            }
        }

        Ok(result)
    }

    /// Collect aggregate functions referenced in a HAVING expression that are NOT
    /// already present in the SELECT column list.  Returns (AggregateFunc, column) pairs.
    fn collect_having_extra_aggs(
        expr: &crate::query::SqlExpr,
        select_cols: &[crate::query::SelectColumn],
    ) -> Vec<(crate::query::AggregateFunc, Option<String>)> {
        use crate::query::{SqlExpr, AggregateFunc, SelectColumn};

        // Build set of already-present aggregate output names
        let existing: Vec<String> = select_cols.iter().filter_map(|c| {
            if let SelectColumn::Aggregate { func, column, alias, .. } = c {
                let fn_name = match func {
                    AggregateFunc::Count => "COUNT",
                    AggregateFunc::Sum   => "SUM",
                    AggregateFunc::Avg   => "AVG",
                    AggregateFunc::Min   => "MIN",
                    AggregateFunc::Max   => "MAX",
                };
                Some(alias.clone().unwrap_or_else(|| format!("{}({})", fn_name, column.as_deref().unwrap_or("*"))))
            } else { None }
        }).collect();

        let mut found: Vec<(AggregateFunc, Option<String>)> = Vec::new();
        Self::walk_having_expr(expr, &existing, &mut found);
        found
    }

    fn walk_having_expr(
        expr: &crate::query::SqlExpr,
        existing: &[String],
        out: &mut Vec<(crate::query::AggregateFunc, Option<String>)>,
    ) {
        use crate::query::{SqlExpr, AggregateFunc};
        match expr {
            SqlExpr::Function { name, args } => {
                let agg_func = if name.eq_ignore_ascii_case("COUNT") { Some(AggregateFunc::Count) }
                    else if name.eq_ignore_ascii_case("SUM") { Some(AggregateFunc::Sum) }
                    else if name.eq_ignore_ascii_case("AVG") { Some(AggregateFunc::Avg) }
                    else if name.eq_ignore_ascii_case("MIN") { Some(AggregateFunc::Min) }
                    else if name.eq_ignore_ascii_case("MAX") { Some(AggregateFunc::Max) }
                    else { None };
                if let Some(func) = agg_func {
                    // Determine column argument
                    let col: Option<String> = args.first().and_then(|a| match a {
                        SqlExpr::Column(c) if c == "*" => None,
                        SqlExpr::Column(c) => Some(c.clone()),
                        SqlExpr::Literal(crate::data::Value::String(s)) if s == "*" => None,
                        _ => None,
                    });
                    let fn_name = match func {
                        AggregateFunc::Count => "COUNT",
                        AggregateFunc::Sum   => "SUM",
                        AggregateFunc::Avg   => "AVG",
                        AggregateFunc::Min   => "MIN",
                        AggregateFunc::Max   => "MAX",
                    };
                    let key = format!("{}({})", fn_name, col.as_deref().unwrap_or("*"));
                    if !existing.iter().any(|e| e.eq_ignore_ascii_case(&key))
                        && !out.iter().any(|(f, c)| {
                            let fn2 = match f { AggregateFunc::Count=>"COUNT", AggregateFunc::Sum=>"SUM", AggregateFunc::Avg=>"AVG", AggregateFunc::Min=>"MIN", AggregateFunc::Max=>"MAX" };
                            format!("{}({})", fn2, c.as_deref().unwrap_or("*")).eq_ignore_ascii_case(&key)
                        })
                    {
                        out.push((func, col));
                    }
                } else {
                    for arg in args { Self::walk_having_expr(arg, existing, out); }
                }
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                Self::walk_having_expr(left, existing, out);
                Self::walk_having_expr(right, existing, out);
            }
            SqlExpr::UnaryOp { expr, .. } | SqlExpr::Paren(expr) | SqlExpr::Cast { expr, .. } => {
                Self::walk_having_expr(expr, existing, out);
            }
            SqlExpr::Case { when_then, else_expr } => {
                for (c, t) in when_then {
                    Self::walk_having_expr(c, existing, out);
                    Self::walk_having_expr(t, existing, out);
                }
                if let Some(e) = else_expr { Self::walk_having_expr(e, existing, out); }
            }
            _ => {}
        }
    }
    
    /// Materialize expression-based GROUP BY columns into the batch as virtual columns.
    /// For example, GROUP BY YEAR(date) evaluates YEAR(date) for each row and adds a
    /// column named "YEAR(date)" to the batch so the grouping logic can use it.
    fn materialize_group_by_exprs(batch: &RecordBatch, stmt: &SelectStatement) -> io::Result<RecordBatch> {
        let mut fields: Vec<Field> = batch.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
        let mut arrays: Vec<ArrayRef> = (0..batch.num_columns()).map(|i| batch.column(i).clone()).collect();
        let mut added_any = false;

        // 1. Evaluate explicit expression-based GROUP BY columns (e.g., YEAR(date))
        for (i, expr_opt) in stmt.group_by_exprs.iter().enumerate() {
            if let Some(expr) = expr_opt {
                let col_name = &stmt.group_by[i];
                if batch.column_by_name(col_name).is_some() { continue; }
                let result_array = Self::evaluate_expr_to_array(batch, expr)?;
                let dt = result_array.data_type().clone();
                fields.push(Field::new(col_name, dt, true));
                arrays.push(result_array);
                added_any = true;
            }
        }

        // 2. Resolve GROUP BY aliases: if a GROUP BY name is not in the batch but
        //    matches a SELECT alias, evaluate that SELECT expression.
        for gb_name in &stmt.group_by {
            let trimmed = gb_name.trim_matches('"');
            if batch.column_by_name(trimmed).is_some() { continue; }
            // Already added above?
            if fields.iter().any(|f| f.name() == trimmed) { continue; }
            // Look for a SELECT column with this alias
            for sel_col in &stmt.columns {
                match sel_col {
                    SelectColumn::Expression { expr, alias: Some(alias) } if alias == trimmed => {
                        let result_array = Self::evaluate_expr_to_array(batch, expr)?;
                        let dt = result_array.data_type().clone();
                        fields.push(Field::new(trimmed, dt, true));
                        arrays.push(result_array);
                        added_any = true;
                        break;
                    }
                    SelectColumn::Aggregate { alias: Some(alias), .. } if alias == trimmed => {
                        // GROUP BY on an aggregate alias is not meaningful, skip
                        break;
                    }
                    _ => {}
                }
            }
        }

        if !added_any {
            return Ok(batch.clone());
        }
        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }
    
    /// Execute GROUP BY using vectorized execution engine
    /// Processes data in 2048-row batches for cache efficiency
    fn execute_group_by_vectorized(
        batch: &RecordBatch,
        stmt: &SelectStatement,
        group_col_name: &str,
    ) -> io::Result<ApexResult> {
        use crate::query::vectorized::{execute_vectorized_group_by, VectorizedHashAgg};
        use crate::query::AggregateFunc;
        
        // If group column has NULLs, fall back to incremental path which handles null keys
        if let Some(col) = batch.column_by_name(group_col_name) {
            if col.null_count() > 0 {
                return Err(io::Error::new(io::ErrorKind::Unsupported, "null group key"));
            }
        }

        // OPTIMIZATION: For DictionaryArray columns, use direct indexing (much faster)
        if let Some(col) = batch.column_by_name(group_col_name) {
            use arrow::array::DictionaryArray;
            use arrow::datatypes::UInt32Type;
            if let Some(dict_arr) = col.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
                let keys = dict_arr.keys();
                let values = dict_arr.values();
                if let Some(str_values) = values.as_any().downcast_ref::<StringArray>() {
                    let num_rows = batch.num_rows();
                    let dict_size = str_values.len() + 1; // +1 for NULL slot
                    
                    // Check if this is COUNT(*) only - can optimize by streaming directly
                    let is_count_only = stmt.columns.iter().all(|c| {
                        matches!(c, SelectColumn::Aggregate { func: AggregateFunc::Count, column: None, .. })
                    });
                    
                    if is_count_only {
                        // OPTIMIZED: Direct aggregation without building indices Vec
                        let mut counts: Vec<i64> = vec![0; dict_size];
                        
                        for row_idx in 0..num_rows {
                            if !keys.is_null(row_idx) {
                                let group_idx = keys.value(row_idx) as usize + 1;
                                unsafe { *counts.get_unchecked_mut(group_idx) += 1; }
                            }
                        }
                        
                        // Build result directly
                        let active_groups: Vec<usize> = (1..dict_size)
                            .filter(|&i| counts[i] > 0)
                            .collect();
                        
                        let mut result_fields: Vec<Field> = Vec::new();
                        let mut result_arrays: Vec<ArrayRef> = Vec::new();
                        
                        // Add group column
                        let group_col_name_clean = stmt.group_by.first()
                            .map(|s| {
                                let trimmed = s.trim_matches('"');
                                if let Some(dot_pos) = trimmed.rfind('.') {
                                    trimmed[dot_pos + 1..].to_string()
                                } else {
                                    trimmed.to_string()
                                }
                            })
                            .unwrap_or_else(|| "group".to_string());
                        
                        let group_values: Vec<&str> = active_groups.iter()
                            .map(|&i| str_values.value(i - 1))
                            .collect();
                        result_fields.push(Field::new(&group_col_name_clean, ArrowDataType::Utf8, false));
                        result_arrays.push(Arc::new(StringArray::from(group_values)));
                        
                        // Add COUNT(*) column
                        let count_values: Vec<i64> = active_groups.iter().map(|&i| counts[i]).collect();
                        result_fields.push(Field::new("COUNT(*)", ArrowDataType::Int64, false));
                        result_arrays.push(Arc::new(Int64Array::from(count_values)));
                        
                        let schema = Arc::new(Schema::new(result_fields));
                        let result_batch = RecordBatch::try_new(schema, result_arrays)
                            .map_err(|e| err_data( e.to_string()))?;
                        
                        return Ok(ApexResult::Data(result_batch));
                    }
                    
                    // For other aggregates, use the standard path
                    let indices: Vec<u32> = (0..num_rows)
                        .map(|i| {
                            if keys.is_null(i) { 0u32 } else { keys.value(i) + 1 }
                        })
                        .collect();
                    
                    let dict_values: Vec<&str> = (0..str_values.len())
                        .map(|i| str_values.value(i))
                        .collect();
                    
                    return Self::execute_group_by_string_dict(
                        batch, stmt, str_values, &indices, &dict_values, dict_size
                    );
                }
            }
            
            // OPTIMIZATION: For low-cardinality StringArray, build dictionary on the fly
            // REMOVED sampling to stabilize performance - always try dictionary path first
            if let Some(str_arr) = col.as_any().downcast_ref::<StringArray>() {
                let num_rows = batch.num_rows();
                
                // Build dictionary directly without sampling - more stable performance
                let mut dict: AHashMap<&str, u32> = AHashMap::with_capacity(200);
                let mut dict_values: Vec<&str> = Vec::with_capacity(200);
                let mut next_id = 1u32;
                
                // First pass: build dictionary and check cardinality
                let mut indices: Vec<u32> = Vec::with_capacity(num_rows);
                indices.resize(num_rows, 0);
                
                for i in 0..num_rows {
                    if !str_arr.is_null(i) {
                        let s = str_arr.value(i);
                        let id = *dict.entry(s).or_insert_with(|| {
                            let id = next_id;
                            next_id += 1;
                            dict_values.push(s);
                            id
                        });
                        indices[i] = id;
                    }
                }
                
                // Only use dict indexing if cardinality is reasonable (<=1000)
                let dict_size = dict_values.len() + 1;
                if dict_size <= 1000 {
                    return Self::execute_group_by_string_dict(
                        batch, stmt, str_arr, &indices, &dict_values, dict_size
                    );
                }
                // Fall through to hash-based aggregation for high cardinality
            }
        }
        
        // Find aggregate column name and type
        let mut agg_col_name: Option<&str> = None;
        let mut has_int_agg = false;
        
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { column: Some(col_name), .. } = col {
                let actual_col = col_name.trim_matches('"');
                let actual_col = if let Some(dot_pos) = actual_col.rfind('.') {
                    &actual_col[dot_pos + 1..]
                } else {
                    actual_col
                };
                if actual_col != "*" {
                    agg_col_name = Some(actual_col);
                    // Check if it's an int column
                    if let Some(arr) = batch.column_by_name(actual_col) {
                        has_int_agg = arr.as_any().downcast_ref::<Int64Array>().is_some();
                    }
                }
                break;
            }
        }
        
        // Execute vectorized GROUP BY for non-dictionary columns
        let hash_agg = execute_vectorized_group_by(batch, group_col_name, agg_col_name, has_int_agg)?;
        
        // Build result from hash aggregation table
        Self::build_group_by_result_from_vectorized(stmt, group_col_name, &hash_agg, has_int_agg)
    }
    
    /// Build GROUP BY result from vectorized hash aggregation
    fn build_group_by_result_from_vectorized(
        stmt: &SelectStatement,
        group_col_name: &str,
        hash_agg: &crate::query::vectorized::VectorizedHashAgg,
        has_int_agg: bool,
    ) -> io::Result<ApexResult> {
        use crate::query::AggregateFunc;
        
        let num_groups = hash_agg.num_groups();
        if num_groups == 0 {
            // Return empty result
            let schema = Arc::new(Schema::new(vec![
                Field::new(group_col_name, ArrowDataType::Utf8, false),
            ]));
            return Ok(ApexResult::Empty(schema));
        }
        
        let states = hash_agg.states();
        let group_keys_str = hash_agg.group_keys_str();
        let group_keys_int = hash_agg.group_keys_int();
        
        let mut result_fields: Vec<Field> = Vec::new();
        let mut result_arrays: Vec<ArrayRef> = Vec::new();
        
        // Check if group column has an alias in the SELECT clause
        let mut group_col_alias: Option<&str> = None;
        for col in &stmt.columns {
            if let SelectColumn::ColumnAlias { column, alias } = col {
                let col_name = column.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if actual_col == group_col_name {
                    group_col_alias = Some(alias.as_str());
                    break;
                }
            }
        }
        let output_group_name = group_col_alias.unwrap_or(group_col_name);
        
        // Add group column
        if !group_keys_str.is_empty() {
            result_fields.push(Field::new(output_group_name, ArrowDataType::Utf8, false));
            result_arrays.push(Arc::new(StringArray::from(
                group_keys_str.iter().map(|s| s.as_str()).collect::<Vec<_>>()
            )));
        } else {
            result_fields.push(Field::new(output_group_name, ArrowDataType::Int64, false));
            result_arrays.push(Arc::new(Int64Array::from(group_keys_int.to_vec())));
        }
        
        // Add aggregate columns
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, alias, .. } = col {
                let func_name = match func {
                    AggregateFunc::Count => "COUNT",
                    AggregateFunc::Sum => "SUM",
                    AggregateFunc::Avg => "AVG",
                    AggregateFunc::Min => "MIN",
                    AggregateFunc::Max => "MAX",
                };
                let field_name = alias.clone().unwrap_or_else(|| {
                    format!("{}({})", func_name, column.as_deref().unwrap_or("*"))
                });
                
                match func {
                    AggregateFunc::Count => { result_fields.push(Field::new(&field_name, ArrowDataType::Int64, false)); result_arrays.push(Arc::new(Int64Array::from(states.iter().map(|s| s.count).collect::<Vec<_>>()))); }
                    AggregateFunc::Sum => {
                        if has_int_agg { result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true)); result_arrays.push(Arc::new(Int64Array::from(states.iter().map(|s| s.sum_int).collect::<Vec<_>>()))); }
                        else { result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true)); result_arrays.push(Arc::new(Float64Array::from(states.iter().map(|s| s.sum_float).collect::<Vec<_>>()))); }
                    }
                    AggregateFunc::Avg => { result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true)); result_arrays.push(Arc::new(Float64Array::from(states.iter().map(|s| if s.count > 0 { if has_int_agg { s.sum_int as f64 / s.count as f64 } else { s.sum_float / s.count as f64 } } else { 0.0 }).collect::<Vec<_>>()))); }
                    AggregateFunc::Min => {
                        if has_int_agg { result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true)); result_arrays.push(Arc::new(Int64Array::from(states.iter().map(|s| s.min_int).collect::<Vec<_>>()))); }
                        else { result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true)); result_arrays.push(Arc::new(Float64Array::from(states.iter().map(|s| s.min_float).collect::<Vec<_>>()))); }
                    }
                    AggregateFunc::Max => {
                        if has_int_agg { result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true)); result_arrays.push(Arc::new(Int64Array::from(states.iter().map(|s| s.max_int).collect::<Vec<_>>()))); }
                        else { result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true)); result_arrays.push(Arc::new(Float64Array::from(states.iter().map(|s| s.max_float).collect::<Vec<_>>()))); }
                    }
                }
            }
        }
        
        let schema = Arc::new(Schema::new(result_fields));
        let mut result_batch = RecordBatch::try_new(schema, result_arrays)
            .map_err(|e| err_data( e.to_string()))?;
        
        // Apply HAVING clause if present
        if let Some(having_expr) = &stmt.having {
            let mask = Self::evaluate_predicate(&result_batch, having_expr)?;
            result_batch = compute::filter_record_batch(&result_batch, &mask)
                .map_err(|e| err_data( e.to_string()))?;
        }
        
        // Apply ORDER BY if present (resolve aggregate expressions to output column names first)
        if !stmt.order_by.is_empty() {
            let resolved_ob = Self::resolve_order_by_cols(&stmt.columns, &stmt.order_by);
            result_batch = Self::apply_order_by(&result_batch, &resolved_ob)?;
        }
        
        Ok(ApexResult::Data(result_batch))
    }
    
    /// Check if we can use incremental aggregation (no DISTINCT, no complex expressions)
    fn can_use_incremental_aggregation(stmt: &SelectStatement) -> bool {
        for col in &stmt.columns {
            match col {
                SelectColumn::Aggregate { func, column, distinct, .. } => {
                    if *distinct { return false; }
                    // COUNT(col) with specific col needs null-aware path
                    if matches!(func, crate::query::AggregateFunc::Count) {
                        if let Some(c) = column {
                            if c != "*" && c != "1" { return false; }
                        }
                    }
                }
                SelectColumn::Expression { .. } => {
                    return false; // Expressions may need row access
                }
                _ => {}
            }
        }
        // HAVING with aggregates is OK, but complex expressions aren't
        true
    }
    
    /// Ultra-fast GROUP BY for string columns using direct dictionary indexing
    /// Uses cache-friendly sequential aggregation with bounds-check elimination
    fn execute_group_by_string_dict(
        batch: &RecordBatch,
        stmt: &SelectStatement,
        _str_arr: &StringArray,
        indices: &[u32],
        dict_values: &[&str],
        dict_size: usize,
    ) -> io::Result<ApexResult> {
        use crate::query::AggregateFunc;
        
        let num_rows = batch.num_rows();
        
        // Direct-indexed aggregate state - pre-allocated for all possible groups
        let mut counts: Vec<i64> = vec![0; dict_size];
        let mut sums_int: Vec<i64> = vec![0; dict_size];
        let mut sums_float: Vec<f64> = vec![0.0; dict_size];
        let mut mins_int: Vec<Option<i64>> = vec![None; dict_size];
        let mut maxs_int: Vec<Option<i64>> = vec![None; dict_size];
        
        // Find aggregate column
        let mut agg_col_int: Option<&Int64Array> = None;
        let mut agg_col_float: Option<&Float64Array> = None;
        
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { column: Some(col_name), .. } = col {
                let actual_col = col_name.trim_matches('"');
                let actual_col = if let Some(dot_pos) = actual_col.rfind('.') {
                    &actual_col[dot_pos + 1..]
                } else {
                    actual_col
                };
                if actual_col != "*" {
                    if let Some(arr) = batch.column_by_name(actual_col) {
                        if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                            agg_col_int = Some(int_arr);
                        } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                            agg_col_float = Some(float_arr);
                        }
                    }
                }
                break;
            }
        }
        
        // OPTIMIZED AGGREGATION: Single pass with bounds-check elimination
        // Uses unsafe for hot path when no nulls present
        if let Some(int_arr) = agg_col_int {
            if int_arr.null_count() == 0 {
                // Fast path: no nulls - use raw slice access
                let values = int_arr.values();
                for row_idx in 0..num_rows {
                    let group_idx = unsafe { *indices.get_unchecked(row_idx) as usize };
                    if group_idx != 0 {
                        unsafe {
                            *counts.get_unchecked_mut(group_idx) += 1;
                            let val = *values.get_unchecked(row_idx);
                            *sums_int.get_unchecked_mut(group_idx) = 
                                sums_int.get_unchecked(group_idx).wrapping_add(val);
                            let min_slot = mins_int.get_unchecked_mut(group_idx);
                            *min_slot = Some(min_slot.map_or(val, |m| m.min(val)));
                            let max_slot = maxs_int.get_unchecked_mut(group_idx);
                            *max_slot = Some(max_slot.map_or(val, |m| m.max(val)));
                        }
                    }
                }
            } else {
                // Slow path: has nulls
                for row_idx in 0..num_rows {
                    let group_idx = indices[row_idx] as usize;
                    if group_idx == 0 { continue; }
                    counts[group_idx] += 1;
                    if !int_arr.is_null(row_idx) {
                        let val = int_arr.value(row_idx);
                        sums_int[group_idx] = sums_int[group_idx].wrapping_add(val);
                        mins_int[group_idx] = Some(mins_int[group_idx].map_or(val, |m| m.min(val)));
                        maxs_int[group_idx] = Some(maxs_int[group_idx].map_or(val, |m| m.max(val)));
                    }
                }
            }
        } else if let Some(float_arr) = agg_col_float {
            if float_arr.null_count() == 0 {
                let values = float_arr.values();
                for row_idx in 0..num_rows {
                    let group_idx = unsafe { *indices.get_unchecked(row_idx) as usize };
                    if group_idx != 0 {
                        unsafe {
                            *counts.get_unchecked_mut(group_idx) += 1;
                            *sums_float.get_unchecked_mut(group_idx) += *values.get_unchecked(row_idx);
                        }
                    }
                }
            } else {
                for row_idx in 0..num_rows {
                    let group_idx = indices[row_idx] as usize;
                    if group_idx == 0 { continue; }
                    counts[group_idx] += 1;
                    if !float_arr.is_null(row_idx) {
                        sums_float[group_idx] += float_arr.value(row_idx);
                    }
                }
            }
        } else {
            // COUNT(*) only
            for row_idx in 0..num_rows {
                let group_idx = unsafe { *indices.get_unchecked(row_idx) as usize };
                if group_idx != 0 {
                    unsafe { *counts.get_unchecked_mut(group_idx) += 1; }
                }
            }
        }
        
        // Collect non-empty groups (skip index 0 which is NULL)
        let active_groups: Vec<usize> = (1..dict_size)
            .filter(|&i| counts[i] > 0)
            .collect();
        
        // Build result arrays
        let mut result_fields: Vec<Field> = Vec::new();
        let mut result_arrays: Vec<ArrayRef> = Vec::new();
        
        // Add group column (string values from dictionary)
        // OPTIMIZATION: Check if group column has an alias in SELECT clause
        let group_by_col = stmt.group_by.first().map(|s| s.trim_matches('"')).unwrap_or("");
        let group_col_name = stmt.columns.iter()
            .find_map(|col| {
                // Check for ColumnAlias pattern: column AS alias
                if let SelectColumn::ColumnAlias { column, alias } = col {
                    let col_trimmed = column.trim_matches('"');
                    // Match either full name (u.tier) or just column name (tier)
                    if col_trimmed == group_by_col || 
                       (group_by_col.contains('.') && col_trimmed == group_by_col.split('.').next().unwrap_or("")) ||
                       (col_trimmed.contains('.') && col_trimmed.ends_with(&format!(". {}", group_by_col.split('.').last().unwrap_or("")))) {
                        return Some(alias.clone());
                    }
                }
                None
            })
            .unwrap_or_else(|| {
                // No alias found, use column name (stripping table prefix)
                if let Some(dot_pos) = group_by_col.rfind('.') {
                    group_by_col[dot_pos + 1..].to_string()
                } else {
                    group_by_col.to_string()
                }
            });
        
        let group_values: Vec<&str> = active_groups.iter()
            .map(|&i| dict_values[i - 1]) // -1 because dict_values is 0-indexed, indices are 1-indexed
            .collect();
        result_fields.push(Field::new(&group_col_name, ArrowDataType::Utf8, false));
        result_arrays.push(Arc::new(StringArray::from(group_values)));
        
        // Add aggregate columns
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, alias, .. } = col {
                let func_name = match func {
                    AggregateFunc::Count => "COUNT",
                    AggregateFunc::Sum => "SUM",
                    AggregateFunc::Avg => "AVG",
                    AggregateFunc::Min => "MIN",
                    AggregateFunc::Max => "MAX",
                };
                let field_name = alias.clone().unwrap_or_else(|| {
                    format!("{}({})", func_name, column.as_deref().unwrap_or("*"))
                });
                
                match func {
                    AggregateFunc::Count => {
                        let values: Vec<i64> = active_groups.iter().map(|&i| counts[i]).collect();
                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, false));
                        result_arrays.push(Arc::new(Int64Array::from(values)));
                    }
                    AggregateFunc::Sum => {
                        if agg_col_int.is_some() {
                            let values: Vec<i64> = active_groups.iter().map(|&i| sums_int[i]).collect();
                            result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                            result_arrays.push(Arc::new(Int64Array::from(values)));
                        } else {
                            let values: Vec<f64> = active_groups.iter().map(|&i| sums_float[i]).collect();
                            result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                            result_arrays.push(Arc::new(Float64Array::from(values)));
                        }
                    }
                    AggregateFunc::Avg => {
                        let values: Vec<f64> = active_groups.iter().map(|&i| {
                            if counts[i] > 0 {
                                if agg_col_int.is_some() {
                                    sums_int[i] as f64 / counts[i] as f64
                                } else {
                                    sums_float[i] / counts[i] as f64
                                }
                            } else { 0.0 }
                        }).collect();
                        result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                        result_arrays.push(Arc::new(Float64Array::from(values)));
                    }
                    AggregateFunc::Min => {
                        let values: Vec<Option<i64>> = active_groups.iter().map(|&i| mins_int[i]).collect();
                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                        result_arrays.push(Arc::new(Int64Array::from(values)));
                    }
                    AggregateFunc::Max => {
                        let values: Vec<Option<i64>> = active_groups.iter().map(|&i| maxs_int[i]).collect();
                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                        result_arrays.push(Arc::new(Int64Array::from(values)));
                    }
                }
            }
        }
        
        let schema = Arc::new(Schema::new(result_fields));
        let mut result_batch = RecordBatch::try_new(schema, result_arrays)
            .map_err(|e| err_data( e.to_string()))?;
        
        // Apply HAVING clause if present
        if let Some(having_expr) = &stmt.having {
            let mask = Self::evaluate_predicate(&result_batch, having_expr)?;
            result_batch = compute::filter_record_batch(&result_batch, &mask)
                .map_err(|e| err_data( e.to_string()))?;
        }
        
        // Apply ORDER BY if present (resolve aggregate expressions to output column names first)
        if !stmt.order_by.is_empty() {
            let resolved_ob = Self::resolve_order_by_cols(&stmt.columns, &stmt.order_by);
            result_batch = Self::apply_order_by(&result_batch, &resolved_ob)?;
        }
        
        Ok(ApexResult::Data(result_batch))
    }
    
    /// Ultra-fast GROUP BY using direct array indexing for small integer ranges
    /// This avoids hash map overhead entirely - O(1) per row instead of hash lookup
    fn execute_group_by_direct_index(
        batch: &RecordBatch,
        stmt: &SelectStatement,
        group_col: &Int64Array,
        min_val: usize,
        range: usize,
    ) -> io::Result<ApexResult> {
        use crate::query::AggregateFunc;
        
        let num_rows = batch.num_rows();
        
        // Direct-indexed aggregate state: [count, sum_int, sum_float, min_int, max_int, first_row]
        let mut counts: Vec<i64> = vec![0; range];
        let mut sums_int: Vec<i64> = vec![0; range];
        let mut sums_float: Vec<f64> = vec![0.0; range];
        let mut mins_int: Vec<Option<i64>> = vec![None; range];
        let mut maxs_int: Vec<Option<i64>> = vec![None; range];
        let mut first_rows: Vec<usize> = vec![usize::MAX; range];
        // Separate state for NULL-key rows
        let mut null_count: i64 = 0;
        let mut null_sum_int: i64 = 0;
        let mut null_sum_float: f64 = 0.0;
        let mut null_min_int: Option<i64> = None;
        let mut null_max_int: Option<i64> = None;
        let mut null_first_row: usize = usize::MAX;
        
        // Find aggregate column
        let mut agg_col_int: Option<&Int64Array> = None;
        let mut agg_col_float: Option<&Float64Array> = None;
        
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { column: Some(col_name), .. } = col {
                let actual_col = col_name.trim_matches('"');
                let actual_col = if let Some(dot_pos) = actual_col.rfind('.') {
                    &actual_col[dot_pos + 1..]
                } else {
                    actual_col
                };
                if actual_col != "*" {
                    if let Some(arr) = batch.column_by_name(actual_col) {
                        if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                            agg_col_int = Some(int_arr);
                        } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                            agg_col_float = Some(float_arr);
                        }
                    }
                }
                break;
            }
        }
        
        // Single pass aggregation with direct indexing
        for row_idx in 0..num_rows {
            if group_col.is_null(row_idx) {
                // NULL key: track separately
                null_count += 1;
                if null_first_row == usize::MAX { null_first_row = row_idx; }
                if let Some(int_arr) = agg_col_int {
                    if !int_arr.is_null(row_idx) {
                        let val = int_arr.value(row_idx);
                        null_sum_int = null_sum_int.wrapping_add(val);
                        null_min_int = Some(null_min_int.map_or(val, |m: i64| m.min(val)));
                        null_max_int = Some(null_max_int.map_or(val, |m: i64| m.max(val)));
                    }
                }
                if let Some(float_arr) = agg_col_float {
                    if !float_arr.is_null(row_idx) { null_sum_float += float_arr.value(row_idx); }
                }
                continue;
            }
            let group_val = group_col.value(row_idx) as usize - min_val;
            
            counts[group_val] += 1;
            if first_rows[group_val] == usize::MAX {
                first_rows[group_val] = row_idx;
            }
            
            if let Some(int_arr) = agg_col_int {
                if !int_arr.is_null(row_idx) {
                    let val = int_arr.value(row_idx);
                    sums_int[group_val] = sums_int[group_val].wrapping_add(val);
                    mins_int[group_val] = Some(mins_int[group_val].map_or(val, |m| m.min(val)));
                    maxs_int[group_val] = Some(maxs_int[group_val].map_or(val, |m| m.max(val)));
                }
            }
            if let Some(float_arr) = agg_col_float {
                if !float_arr.is_null(row_idx) {
                    sums_float[group_val] += float_arr.value(row_idx);
                }
            }
        }
        
        // Collect non-empty groups
        let active_groups: Vec<usize> = (0..range)
            .filter(|&i| counts[i] > 0)
            .collect();
        
        // Build result arrays
        let mut result_fields: Vec<Field> = Vec::new();
        let mut result_arrays: Vec<ArrayRef> = Vec::new();
        
        // Add group column
        let group_col_name = stmt.group_by.first()
            .map(|s| s.trim_matches('"').to_string())
            .unwrap_or_else(|| "group".to_string());
        
        // Build group key array including optional NULL group
        let has_null_group = null_count > 0;
        let total_groups = active_groups.len() + if has_null_group { 1 } else { 0 };
        let _ = total_groups; // suppress warning
        {
            // Build Int64 array with possible null entry
            let mut key_vals: Vec<Option<i64>> = active_groups.iter()
                .map(|&i| Some((i + min_val) as i64))
                .collect();
            if has_null_group { key_vals.push(None); }
            result_fields.push(Field::new(&group_col_name, ArrowDataType::Int64, true));
            result_arrays.push(Arc::new(Int64Array::from(key_vals)));
        }
        
        // Add aggregate columns
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, alias, .. } = col {
                let field_name = alias.clone().unwrap_or_else(|| format!("{}({})", func.to_string(), column.as_deref().unwrap_or("*")));
                match func {
                    AggregateFunc::Count => {
                        let mut vals: Vec<i64> = active_groups.iter().map(|&i| counts[i]).collect();
                        if has_null_group { vals.push(null_count); }
                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, false));
                        result_arrays.push(Arc::new(Int64Array::from(vals)));
                    }
                    AggregateFunc::Sum => {
                        if agg_col_int.is_some() {
                            let mut vals: Vec<i64> = active_groups.iter().map(|&i| sums_int[i]).collect();
                            if has_null_group { vals.push(null_sum_int); }
                            result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                            result_arrays.push(Arc::new(Int64Array::from(vals)));
                        } else {
                            let mut vals: Vec<f64> = active_groups.iter().map(|&i| sums_float[i]).collect();
                            if has_null_group { vals.push(null_sum_float); }
                            result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                            result_arrays.push(Arc::new(Float64Array::from(vals)));
                        }
                    }
                    AggregateFunc::Avg => {
                        let mut vals: Vec<f64> = active_groups.iter().map(|&i| {
                            if counts[i] > 0 { if agg_col_int.is_some() { sums_int[i] as f64 / counts[i] as f64 } else { sums_float[i] / counts[i] as f64 } } else { 0.0 }
                        }).collect();
                        if has_null_group { vals.push(if null_count > 0 { if agg_col_int.is_some() { null_sum_int as f64 / null_count as f64 } else { null_sum_float / null_count as f64 } } else { 0.0 }); }
                        result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                        result_arrays.push(Arc::new(Float64Array::from(vals)));
                    }
                    AggregateFunc::Min => {
                        let mut vals: Vec<Option<i64>> = active_groups.iter().map(|&i| mins_int[i]).collect();
                        if has_null_group { vals.push(null_min_int); }
                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                        result_arrays.push(Arc::new(Int64Array::from(vals)));
                    }
                    AggregateFunc::Max => {
                        let mut vals: Vec<Option<i64>> = active_groups.iter().map(|&i| maxs_int[i]).collect();
                        if has_null_group { vals.push(null_max_int); }
                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                        result_arrays.push(Arc::new(Int64Array::from(vals)));
                    }
                }
            }
        }
        
        let schema = Arc::new(Schema::new(result_fields));
        let mut result_batch = RecordBatch::try_new(schema, result_arrays).map_err(|e| err_data(e.to_string()))?;
        if let Some(having_expr) = &stmt.having { let mask = Self::evaluate_predicate(&result_batch, having_expr)?; result_batch = compute::filter_record_batch(&result_batch, &mask).map_err(|e| err_data(e.to_string()))?; }
        if !stmt.order_by.is_empty() {
            let resolved_ob = Self::resolve_order_by_cols(&stmt.columns, &stmt.order_by);
            result_batch = Self::apply_order_by(&result_batch, &resolved_ob)?;
        }
        Ok(ApexResult::Data(result_batch))
    }
    
    /// Fast incremental GROUP BY with parallel partitioned aggregation (DuckDB-style)
    /// Key optimizations:
    /// 1. Parallel partition-based aggregation for large datasets
    /// 2. Single-pass hash+aggregate for each partition
    /// 3. Merge partition results at the end
    fn execute_group_by_incremental(
        batch: &RecordBatch,
        stmt: &SelectStatement,
        group_cols: &[String],
    ) -> io::Result<ApexResult> {
        use crate::query::AggregateFunc;
        
        let num_rows = batch.num_rows();
        
        // FAST PATH: Single column GROUP BY on small integer range (e.g., category_id 0-999)
        // Uses direct array indexing instead of hash map - much faster
        if group_cols.len() == 1 {
            if let Some(col) = batch.column_by_name(&group_cols[0]) {
                if let Some(int_arr) = col.as_any().downcast_ref::<Int64Array>() {
                    // Check if values are in a small range for direct indexing
                    let (min_val, max_val) = {
                        let mut min = i64::MAX;
                        let mut max = i64::MIN;
                        for i in 0..num_rows {
                            if !int_arr.is_null(i) {
                                let v = int_arr.value(i);
                                min = min.min(v);
                                max = max.max(v);
                            }
                        }
                        (min, max)
                    };
                    
                    // Use direct indexing if range is reasonable (< 10000 unique values)
                    let range = (max_val - min_val + 1) as usize;
                    if min_val >= 0 && range <= 10000 && range > 0 {
                        return Self::execute_group_by_direct_index(batch, stmt, int_arr, min_val as usize, range);
                    }
                }
            }
        }
        
        let estimated_groups = (num_rows / 10).max(16);
        
        // Incremental aggregate state per group
        #[derive(Clone)]
        struct GroupState {
            first_row: usize,
            count: i64,
            sum_int: i64,
            sum_float: f64,
            min_int: Option<i64>,
            max_int: Option<i64>,
            min_float: Option<f64>,
            max_float: Option<f64>,
        }
        
        impl GroupState {
            #[inline(always)]
            fn new(first_row: usize) -> Self {
                Self {
                    first_row,
                    count: 0,
                    sum_int: 0,
                    sum_float: 0.0,
                    min_int: None,
                    max_int: None,
                    min_float: None,
                    max_float: None,
                }
            }
        }
        
        // Pre-downcast group columns and build runtime dictionaries for strings
        // OPTIMIZATION: Build dictionary (string -> integer ID) for low-cardinality string columns
        // This converts string hashing to integer operations, similar to DuckDB's storage-level dictionary
        enum TypedCol<'a> {
            Int64(&'a Int64Array),
            Float64(&'a Float64Array),
            StringDict(&'a StringArray, Vec<u32>),  // (array, dictionary indices per row)
            Bool(&'a BooleanArray),
            Other(&'a ArrayRef),
        }
        
        // OPTIMIZATION: For single column GROUP BY, use direct dictionary indexing
        // This is much faster than hash-based grouping for low-cardinality columns
        if group_cols.len() == 1 {
            if let Some(col) = batch.column_by_name(&group_cols[0]) {
                // FAST PATH 1: Arrow DictionaryArray - indices already available, no conversion needed!
                use arrow::array::DictionaryArray;
                use arrow::datatypes::UInt32Type;
                if let Some(dict_arr) = col.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
                    let keys = dict_arr.keys();
                    let values = dict_arr.values();
                    if let Some(str_values) = values.as_any().downcast_ref::<StringArray>() {
                        let dict_size = str_values.len() + 1; // +1 for NULL slot
                        
                        // Extract indices directly - no dictionary building needed!
                        let indices: Vec<u32> = (0..num_rows)
                            .map(|i| {
                                if keys.is_null(i) { 0u32 } else { keys.value(i) + 1 } // +1 for NULL at 0
                            })
                            .collect();
                        
                        // Build dict_values from StringArray
                        let dict_values: Vec<&str> = (0..str_values.len())
                            .map(|i| str_values.value(i))
                            .collect();
                        
                        return Self::execute_group_by_string_dict(
                            batch, stmt, str_values, &indices, &dict_values, dict_size
                        );
                    }
                }
                
                // FAST PATH 2: Regular StringArray - build dictionary
                // REMOVED sampling to stabilize performance
                if let Some(str_arr) = col.as_any().downcast_ref::<StringArray>() {
                    // Build dictionary directly without sampling
                    let mut dict: AHashMap<&str, u32> = AHashMap::with_capacity(200);
                    let mut dict_values: Vec<&str> = Vec::with_capacity(200);
                    let mut next_id = 1u32;
                    
                    let mut indices: Vec<u32> = Vec::with_capacity(num_rows);
                    indices.resize(num_rows, 0);
                    
                    for i in 0..num_rows {
                        if !str_arr.is_null(i) {
                            let s = str_arr.value(i);
                            let id = *dict.entry(s).or_insert_with(|| {
                                let id = next_id;
                                next_id += 1;
                                dict_values.push(s);
                                id
                            });
                            indices[i] = id;
                        }
                    }
                    
                    // Only use dict indexing if cardinality is reasonable
                    let dict_size = dict_values.len() + 1;
                    if dict_size <= 1000 {
                        return Self::execute_group_by_string_dict(
                            batch, stmt, str_arr, &indices, &dict_values, dict_size
                        );
                    }
                }
            }
        }
        
        // FAST PATH: 2-column GROUP BY with low-cardinality string columns
        // Uses composite dictionary indexing: (dict1_id * dict2_size + dict2_id) as direct array index
        if group_cols.len() == 2 {
            use arrow::array::DictionaryArray;
            use arrow::datatypes::UInt32Type;
            
            let col1 = batch.column_by_name(&group_cols[0]);
            let col2 = batch.column_by_name(&group_cols[1]);
            
            if let (Some(c1), Some(c2)) = (col1, col2) {
                // Build dictionaries for both columns - handles both StringArray and DictionaryArray
                let build_dict = |col: &ArrayRef, n_rows: usize| -> Option<(Vec<u32>, Vec<String>, usize)> {
                    // Case 1: DictionaryArray - already dictionary encoded!
                    if let Some(dict_arr) = col.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
                        let keys = dict_arr.keys();
                        let values = dict_arr.values();
                        if let Some(str_values) = values.as_any().downcast_ref::<StringArray>() {
                            let dict_size = str_values.len() + 1;
                            if dict_size <= 1000 {
                                let indices: Vec<u32> = (0..n_rows)
                                    .map(|i| {
                                        if keys.is_null(i) { 0u32 } else { keys.value(i) + 1 }
                                    })
                                    .collect();
                                let dict_values: Vec<String> = (0..str_values.len())
                                    .map(|i| str_values.value(i).to_string())
                                    .collect();
                                return Some((indices, dict_values, dict_size));
                            }
                        }
                    }
                    
                    // Case 2: StringArray - build dictionary
                    if let Some(str_arr) = col.as_any().downcast_ref::<StringArray>() {
                        let mut dict: AHashMap<&str, u32> = AHashMap::with_capacity(200);
                        let mut dict_values: Vec<String> = Vec::with_capacity(200);
                        let mut next_id = 1u32;
                        
                        let indices: Vec<u32> = (0..n_rows)
                            .map(|i| {
                                if str_arr.is_null(i) {
                                    0u32
                                } else {
                                    let s = str_arr.value(i);
                                    *dict.entry(s).or_insert_with(|| {
                                        let id = next_id;
                                        next_id += 1;
                                        dict_values.push(s.to_string());
                                        id
                                    })
                                }
                            })
                            .collect();
                        
                        let dict_size = dict_values.len() + 1;
                        if dict_size <= 1000 {
                            return Some((indices, dict_values, dict_size));
                        }
                    }
                    
                    // Case 3: LargeStringArray - build dictionary
                    if let Some(str_arr) = col.as_any().downcast_ref::<arrow::array::LargeStringArray>() {
                        let mut dict: AHashMap<String, u32> = AHashMap::with_capacity(200);
                        let mut dict_values: Vec<String> = Vec::with_capacity(200);
                        let mut next_id = 1u32;
                        
                        let indices: Vec<u32> = (0..n_rows)
                            .map(|i| {
                                if str_arr.is_null(i) {
                                    0u32
                                } else {
                                    let s = str_arr.value(i);
                                    *dict.entry(s.to_string()).or_insert_with(|| {
                                        let id = next_id;
                                        next_id += 1;
                                        dict_values.push(s.to_string());
                                        id
                                    })
                                }
                            })
                            .collect();
                        
                        let dict_size = dict_values.len() + 1;
                        if dict_size <= 1000 {
                            return Some((indices, dict_values, dict_size));
                        }
                    }
                    
                    // Case 4: BinaryArray - build dictionary
                    if let Some(bin_arr) = col.as_any().downcast_ref::<arrow::array::BinaryArray>() {
                        let mut dict: AHashMap<String, u32> = AHashMap::with_capacity(200);
                        let mut dict_values: Vec<String> = Vec::with_capacity(200);
                        let mut next_id = 1u32;
                        
                        let indices: Vec<u32> = (0..n_rows)
                            .map(|i| {
                                if bin_arr.is_null(i) {
                                    0u32
                                } else {
                                    let s = bin_arr.value(i);
                                    let s_str = String::from_utf8_lossy(s);
                                    *dict.entry(s_str.to_string()).or_insert_with(|| {
                                        let id = next_id;
                                        next_id += 1;
                                        dict_values.push(s_str.to_string());
                                        id
                                    })
                                }
                            })
                            .collect();
                        
                        let dict_size = dict_values.len() + 1;
                        if dict_size <= 1000 {
                            return Some((indices, dict_values, dict_size));
                        }
                    }
                    
                    None
                };
                
                if let (Some((indices1, dict1_values, dict1_size)), Some((indices2, dict2_values, dict2_size))) = 
                    (build_dict(c1, num_rows), build_dict(c2, num_rows)) 
                {
                    // Use composite key: (idx1 * dict2_size + idx2) for direct array indexing
                    let total_size = dict1_size * dict2_size;
                    if total_size <= 100_000 {
                        // Find aggregate column - support both Int64 and Float64
                        let mut agg_col_float: Option<&Float64Array> = None;
                        let mut agg_col_int: Option<&Int64Array> = None;
                        for col in &stmt.columns {
                            if let SelectColumn::Aggregate { column: Some(col_name), .. } = col {
                                let actual_col = col_name.trim_matches('"');
                                let actual_col = if let Some(dot_pos) = actual_col.rfind('.') {
                                    &actual_col[dot_pos + 1..]
                                } else {
                                    actual_col
                                };
                                if actual_col != "*" {
                                    if let Some(arr) = batch.column_by_name(actual_col) {
                                        if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                                            agg_col_float = Some(float_arr);
                                        } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                                            agg_col_int = Some(int_arr);
                                        }
                                    }
                                }
                                break;
                            }
                        }
                        
                        // Direct-indexed aggregation - no hash map needed!
                        let mut counts: Vec<i64> = vec![0; total_size];
                        let mut sums_int: Vec<i64> = vec![0; total_size];
                        let mut sums_float: Vec<f64> = vec![0.0; total_size];
                        
                        if let Some(int_arr) = agg_col_int {
                            // Int64 aggregate
                            if int_arr.null_count() == 0 {
                                let values = int_arr.values();
                                for row_idx in 0..num_rows {
                                    let idx1 = unsafe { *indices1.get_unchecked(row_idx) as usize };
                                    let idx2 = unsafe { *indices2.get_unchecked(row_idx) as usize };
                                    if idx1 != 0 && idx2 != 0 {
                                        let composite = idx1 * dict2_size + idx2;
                                        unsafe {
                                            *counts.get_unchecked_mut(composite) += 1;
                                            *sums_int.get_unchecked_mut(composite) += *values.get_unchecked(row_idx);
                                        }
                                    }
                                }
                            } else {
                                for row_idx in 0..num_rows {
                                    let idx1 = indices1[row_idx] as usize;
                                    let idx2 = indices2[row_idx] as usize;
                                    if idx1 == 0 || idx2 == 0 { continue; }
                                    let composite = idx1 * dict2_size + idx2;
                                    counts[composite] += 1;
                                    if !int_arr.is_null(row_idx) {
                                        sums_int[composite] += int_arr.value(row_idx);
                                    }
                                }
                            }
                        } else if let Some(float_arr) = agg_col_float {
                            // Float64 aggregate
                            if float_arr.null_count() == 0 {
                                let values = float_arr.values();
                                for row_idx in 0..num_rows {
                                    let idx1 = unsafe { *indices1.get_unchecked(row_idx) as usize };
                                    let idx2 = unsafe { *indices2.get_unchecked(row_idx) as usize };
                                    if idx1 != 0 && idx2 != 0 {
                                        let composite = idx1 * dict2_size + idx2;
                                        unsafe {
                                            *counts.get_unchecked_mut(composite) += 1;
                                            *sums_float.get_unchecked_mut(composite) += *values.get_unchecked(row_idx);
                                        }
                                    }
                                }
                            } else {
                                for row_idx in 0..num_rows {
                                    let idx1 = indices1[row_idx] as usize;
                                    let idx2 = indices2[row_idx] as usize;
                                    if idx1 == 0 || idx2 == 0 { continue; }
                                    let composite = idx1 * dict2_size + idx2;
                                    counts[composite] += 1;
                                    if !float_arr.is_null(row_idx) {
                                        sums_float[composite] += float_arr.value(row_idx);
                                    }
                                }
                            }
                        } else {
                            // COUNT(*) only
                            for row_idx in 0..num_rows {
                                let idx1 = unsafe { *indices1.get_unchecked(row_idx) as usize };
                                let idx2 = unsafe { *indices2.get_unchecked(row_idx) as usize };
                                if idx1 != 0 && idx2 != 0 {
                                    let composite = idx1 * dict2_size + idx2;
                                    unsafe { *counts.get_unchecked_mut(composite) += 1; }
                                }
                            }
                        }
                        
                        // Collect active groups
                        let mut result_col1: Vec<&str> = Vec::with_capacity(dict1_size * dict2_size / 10);
                        let mut result_col2: Vec<&str> = Vec::with_capacity(dict1_size * dict2_size / 10);
                        let mut result_counts: Vec<i64> = Vec::with_capacity(dict1_size * dict2_size / 10);
                        let mut result_sums_int: Vec<i64> = Vec::with_capacity(dict1_size * dict2_size / 10);
                        let mut result_sums_float: Vec<f64> = Vec::with_capacity(dict1_size * dict2_size / 10);
                        
                        for idx1 in 1..dict1_size {
                            for idx2 in 1..dict2_size {
                                let composite = idx1 * dict2_size + idx2;
                                if counts[composite] > 0 {
                                    result_col1.push(&dict1_values[idx1 - 1]);
                                    result_col2.push(&dict2_values[idx2 - 1]);
                                    result_counts.push(counts[composite]);
                                    result_sums_int.push(sums_int[composite]);
                                    result_sums_float.push(sums_float[composite]);
                                }
                            }
                        }
                        
                        // Build result
                        use crate::query::AggregateFunc;
                        let mut result_fields: Vec<Field> = Vec::new();
                        let mut result_arrays: Vec<ArrayRef> = Vec::new();
                        result_fields.push(Field::new(group_cols[0].trim_matches('"'), ArrowDataType::Utf8, false));
                        result_arrays.push(Arc::new(StringArray::from(result_col1)));
                        result_fields.push(Field::new(group_cols[1].trim_matches('"'), ArrowDataType::Utf8, false));
                        result_arrays.push(Arc::new(StringArray::from(result_col2)));
                        let has_int_agg = agg_col_int.is_some();
                        for col in &stmt.columns {
                            if let SelectColumn::Aggregate { func, column, alias, .. } = col {
                                let fn_name = match func { AggregateFunc::Count => "COUNT", AggregateFunc::Sum => "SUM", AggregateFunc::Avg => "AVG", AggregateFunc::Min => "MIN", AggregateFunc::Max => "MAX" };
                                let field_name = alias.clone().unwrap_or_else(|| format!("{}({})", fn_name, column.as_deref().unwrap_or("*")));
                                match func {
                                    AggregateFunc::Count => { result_fields.push(Field::new(&field_name, ArrowDataType::Int64, false)); result_arrays.push(Arc::new(Int64Array::from(result_counts.clone()))); }
                                    AggregateFunc::Sum => { if has_int_agg { result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true)); result_arrays.push(Arc::new(Int64Array::from(result_sums_int.clone()))); } else { result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true)); result_arrays.push(Arc::new(Float64Array::from(result_sums_float.clone()))); } }
                                    AggregateFunc::Avg => { let avgs: Vec<f64> = if has_int_agg { result_counts.iter().zip(result_sums_int.iter()).map(|(&c, &s)| if c > 0 { s as f64 / c as f64 } else { 0.0 }).collect() } else { result_counts.iter().zip(result_sums_float.iter()).map(|(&c, &s)| if c > 0 { s / c as f64 } else { 0.0 }).collect() }; result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true)); result_arrays.push(Arc::new(Float64Array::from(avgs))); }
                                    _ => {}
                                }
                            }
                        }
                        let schema = Arc::new(Schema::new(result_fields));
                        let mut result_batch = RecordBatch::try_new(schema, result_arrays).map_err(|e| err_data(e.to_string()))?;
                        
                        // Apply HAVING/ORDER BY/LIMIT
                        if let Some(ref having) = stmt.having {
                            let mask = Self::evaluate_predicate(&result_batch, having)?;
                            result_batch = compute::filter_record_batch(&result_batch, &mask)
                                .map_err(|e| err_data( e.to_string()))?;
                        }
                        
                        if !stmt.order_by.is_empty() {
                            let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
                            result_batch = Self::apply_order_by_topk(&result_batch, &stmt.order_by, k)?;
                        }
                        
                        result_batch = Self::apply_limit_offset(&result_batch, stmt.limit, stmt.offset)?;
                        
                        return Ok(ApexResult::Data(result_batch));
                    }
                }
            }
        }
        
        // FAST PATH: String + Int64 2-column GROUP BY (common case: category + numeric id)
        // Uses composite key: (string_dict_id * int_range + int_value_offset) for direct array indexing
        if group_cols.len() == 2 {
            let col1 = batch.column_by_name(&group_cols[0]);
            let col2 = batch.column_by_name(&group_cols[1]);
            
            if let (Some(c1), Some(c2)) = (col1, col2) {
                // Try to build dictionary for string column and get int range for int column
                let string_dict_result: Option<(Vec<u32>, Vec<String>, usize)> = {
                    use arrow::array::DictionaryArray;
                    use arrow::datatypes::UInt32Type;
                    
                    // Case 1: DictionaryArray
                    if let Some(dict_arr) = c1.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
                        let keys = dict_arr.keys();
                        let values = dict_arr.values();
                        if let Some(str_values) = values.as_any().downcast_ref::<StringArray>() {
                            let dict_size = str_values.len() + 1;
                            if dict_size <= 1000 {
                                let indices: Vec<u32> = (0..num_rows)
                                    .map(|i| if keys.is_null(i) { 0u32 } else { keys.value(i) + 1 })
                                    .collect();
                                let dict_values: Vec<String> = (0..str_values.len())
                                    .map(|i| str_values.value(i).to_string())
                                    .collect();
                                Some((indices, dict_values, dict_size))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    // Case 2: StringArray - build dictionary
                    else if let Some(str_arr) = c1.as_any().downcast_ref::<StringArray>() {
                        let mut dict: AHashMap<&str, u32> = AHashMap::with_capacity(200);
                        let mut dict_values: Vec<String> = Vec::with_capacity(200);
                        let mut next_id = 1u32;
                        
                        let indices: Vec<u32> = (0..num_rows)
                            .map(|i| {
                                if str_arr.is_null(i) {
                                    0u32
                                } else {
                                    let s = str_arr.value(i);
                                    *dict.entry(s).or_insert_with(|| {
                                        let id = next_id;
                                        next_id += 1;
                                        dict_values.push(s.to_string());
                                        id
                                    })
                                }
                            })
                            .collect();
                        
                        let dict_size = dict_values.len() + 1;
                        if dict_size <= 1000 {
                            Some((indices, dict_values, dict_size))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };
                
                // Get int column range
                let int_range_result: Option<(Vec<u32>, i64, usize)> = if let Some(int_arr) = c2.as_any().downcast_ref::<Int64Array>() {
                    let (min_val, max_val) = {
                        let mut min = i64::MAX;
                        let mut max = i64::MIN;
                        for i in 0..num_rows {
                            if !int_arr.is_null(i) {
                                let v = int_arr.value(i);
                                min = min.min(v);
                                max = max.max(v);
                            }
                        }
                        (min, max)
                    };
                    
                    let range = (max_val - min_val + 1) as usize;
                    if min_val >= 0 && range <= 1000 && range > 0 {
                        let indices: Vec<u32> = (0..num_rows)
                            .map(|i| {
                                if int_arr.is_null(i) {
                                    0u32
                                } else {
                                    (int_arr.value(i) - min_val + 1) as u32
                                }
                            })
                            .collect();
                        Some((indices, min_val, range))
                    } else {
                        None
                    }
                } else {
                    None
                };
                
                // If both columns can use dictionary indexing
                if let (Some((str_indices, str_values, str_size)), Some((int_indices, int_min, int_range))) = 
                    (string_dict_result, int_range_result) 
                {
                    let total_size = str_size * (int_range + 1);
                    if total_size <= 100_000 {
                        // Find aggregate column
                        let mut agg_col_int: Option<&Int64Array> = None;
                        let mut agg_col_float: Option<&Float64Array> = None;
                        for col in &stmt.columns {
                            if let SelectColumn::Aggregate { column: Some(col_name), .. } = col {
                                let actual_col = col_name.trim_matches('"');
                                let actual_col = if let Some(dot_pos) = actual_col.rfind('.') {
                                    &actual_col[dot_pos + 1..]
                                } else {
                                    actual_col
                                };
                                if actual_col != "*" {
                                    if let Some(arr) = batch.column_by_name(actual_col) {
                                        if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                                            agg_col_float = Some(float_arr);
                                        } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                                            agg_col_int = Some(int_arr);
                                        }
                                    }
                                }
                                break;
                            }
                        }
                        
                        // Direct-indexed aggregation
                        let mut counts: Vec<i64> = vec![0; total_size];
                        let mut sums_int: Vec<i64> = vec![0; total_size];
                        let mut sums_float: Vec<f64> = vec![0.0; total_size];
                        
                        if let Some(int_arr) = agg_col_int {
                            if int_arr.null_count() == 0 {
                                let values = int_arr.values();
                                for row_idx in 0..num_rows {
                                    let str_idx = unsafe { *str_indices.get_unchecked(row_idx) as usize };
                                    let int_idx = unsafe { *int_indices.get_unchecked(row_idx) as usize };
                                    if str_idx != 0 && int_idx != 0 {
                                        let composite = str_idx * (int_range + 1) + int_idx;
                                        unsafe {
                                            *counts.get_unchecked_mut(composite) += 1;
                                            *sums_int.get_unchecked_mut(composite) += *values.get_unchecked(row_idx);
                                        }
                                    }
                                }
                            } else {
                                for row_idx in 0..num_rows {
                                    let str_idx = str_indices[row_idx] as usize;
                                    let int_idx = int_indices[row_idx] as usize;
                                    if str_idx == 0 || int_idx == 0 { continue; }
                                    let composite = str_idx * (int_range + 1) + int_idx;
                                    counts[composite] += 1;
                                    if !int_arr.is_null(row_idx) {
                                        sums_int[composite] += int_arr.value(row_idx);
                                    }
                                }
                            }
                        } else if let Some(float_arr) = agg_col_float {
                            if float_arr.null_count() == 0 {
                                let values = float_arr.values();
                                for row_idx in 0..num_rows {
                                    let str_idx = unsafe { *str_indices.get_unchecked(row_idx) as usize };
                                    let int_idx = unsafe { *int_indices.get_unchecked(row_idx) as usize };
                                    if str_idx != 0 && int_idx != 0 {
                                        let composite = str_idx * (int_range + 1) + int_idx;
                                        unsafe {
                                            *counts.get_unchecked_mut(composite) += 1;
                                            *sums_float.get_unchecked_mut(composite) += *values.get_unchecked(row_idx);
                                        }
                                    }
                                }
                            } else {
                                for row_idx in 0..num_rows {
                                    let str_idx = str_indices[row_idx] as usize;
                                    let int_idx = int_indices[row_idx] as usize;
                                    if str_idx == 0 || int_idx == 0 { continue; }
                                    let composite = str_idx * (int_range + 1) + int_idx;
                                    counts[composite] += 1;
                                    if !float_arr.is_null(row_idx) {
                                        sums_float[composite] += float_arr.value(row_idx);
                                    }
                                }
                            }
                        } else {
                            // COUNT(*) only
                            for row_idx in 0..num_rows {
                                let str_idx = unsafe { *str_indices.get_unchecked(row_idx) as usize };
                                let int_idx = unsafe { *int_indices.get_unchecked(row_idx) as usize };
                                if str_idx != 0 && int_idx != 0 {
                                    let composite = str_idx * (int_range + 1) + int_idx;
                                    unsafe { *counts.get_unchecked_mut(composite) += 1; }
                                }
                            }
                        }
                        
                        // Collect active groups
                        let mut result_col1: Vec<&str> = Vec::with_capacity(total_size / 10);
                        let mut result_col2: Vec<i64> = Vec::with_capacity(total_size / 10);
                        let mut result_counts: Vec<i64> = Vec::with_capacity(total_size / 10);
                        let mut result_sums_int: Vec<i64> = Vec::with_capacity(total_size / 10);
                        let mut result_sums_float: Vec<f64> = Vec::with_capacity(total_size / 10);
                        
                        for str_idx in 1..str_size {
                            for int_offset in 1..=int_range {
                                let composite = str_idx * (int_range + 1) + int_offset;
                                if counts[composite] > 0 {
                                    result_col1.push(&str_values[str_idx - 1]);
                                    result_col2.push(int_min + (int_offset - 1) as i64);
                                    result_counts.push(counts[composite]);
                                    result_sums_int.push(sums_int[composite]);
                                    result_sums_float.push(sums_float[composite]);
                                }
                            }
                        }
                        
                        // Build result
                        use crate::query::AggregateFunc;
                        let mut result_fields: Vec<Field> = Vec::new();
                        let mut result_arrays: Vec<ArrayRef> = Vec::new();
                        
                        result_fields.push(Field::new(group_cols[0].trim_matches('"'), ArrowDataType::Utf8, false));
                        result_arrays.push(Arc::new(StringArray::from(result_col1)));
                        result_fields.push(Field::new(group_cols[1].trim_matches('"'), ArrowDataType::Int64, false));
                        result_arrays.push(Arc::new(Int64Array::from(result_col2)));
                        
                        let has_int_agg = agg_col_int.is_some();
                        
                        for col in &stmt.columns {
                            if let SelectColumn::Aggregate { func, column, alias, .. } = col {
                                let func_name = match func {
                                    AggregateFunc::Count => "COUNT",
                                    AggregateFunc::Sum => "SUM",
                                    AggregateFunc::Avg => "AVG",
                                    AggregateFunc::Min => "MIN",
                                    AggregateFunc::Max => "MAX",
                                };
                                let field_name = alias.clone().unwrap_or_else(|| {
                                    format!("{}({})", func_name, column.as_deref().unwrap_or("*"))
                                });
                                
                                match func {
                                    AggregateFunc::Count => {
                                        result_fields.push(Field::new(&field_name, ArrowDataType::Int64, false));
                                        result_arrays.push(Arc::new(Int64Array::from(result_counts.clone())));
                                    }
                                    AggregateFunc::Sum => {
                                        if has_int_agg {
                                            result_fields.push(Field::new(&field_name, ArrowDataType::Int64, true));
                                            result_arrays.push(Arc::new(Int64Array::from(result_sums_int.clone())));
                                        } else {
                                            result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                                            result_arrays.push(Arc::new(Float64Array::from(result_sums_float.clone())));
                                        }
                                    }
                                    AggregateFunc::Avg => {
                                        let avgs: Vec<f64> = if has_int_agg {
                                            result_counts.iter().zip(result_sums_int.iter())
                                                .map(|(&c, &s)| if c > 0 { s as f64 / c as f64 } else { 0.0 })
                                                .collect()
                                        } else {
                                            result_counts.iter().zip(result_sums_float.iter())
                                                .map(|(&c, &s)| if c > 0 { s / c as f64 } else { 0.0 })
                                                .collect()
                                        };
                                        result_fields.push(Field::new(&field_name, ArrowDataType::Float64, true));
                                        result_arrays.push(Arc::new(Float64Array::from(avgs)));
                                    }
                                    _ => {}
                                }
                            }
                        }
                        
                        let schema = Arc::new(Schema::new(result_fields));
                        let mut result_batch = RecordBatch::try_new(schema, result_arrays)
                            .map_err(|e| err_data( e.to_string()))?;
                        
                        // Apply HAVING/ORDER BY/LIMIT
                        if let Some(ref having) = stmt.having {
                            let mask = Self::evaluate_predicate(&result_batch, having)?;
                            result_batch = compute::filter_record_batch(&result_batch, &mask)
                                .map_err(|e| err_data( e.to_string()))?;
                        }
                        
                        if !stmt.order_by.is_empty() {
                            let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
                            result_batch = Self::apply_order_by_topk(&result_batch, &stmt.order_by, k)?;
                        }
                        
                        result_batch = Self::apply_limit_offset(&result_batch, stmt.limit, stmt.offset)?;
                        
                        return Ok(ApexResult::Data(result_batch));
                    }
                }
            }
        }
        
        // FAST PATH: Multi-column GROUP BY (3+ columns) using vectorized execution
        // This is faster than the general path because it uses pre-typed columns and batch processing
        if group_cols.len() >= 3 {
            use crate::query::multi_column::{execute_multi_column_group_by, build_multi_column_result};
            
            // Extract aggregate function info
            let (agg_func, agg_col_name) = stmt.columns.iter()
                .find_map(|col| {
                    if let SelectColumn::Aggregate { func, column, .. } = col {
                        Some((func.clone(), column.as_deref()))
                    } else {
                        None
                    }
                })
                .unwrap_or((crate::query::AggregateFunc::Count, None));
            
            // Execute optimized multi-column group by
            match execute_multi_column_group_by(batch, group_cols, agg_col_name) {
                Ok(hash_agg) => {
                    let result_batch = build_multi_column_result(
                        &hash_agg, batch, group_cols, Some(agg_func), agg_col_name
                    )?;
                    
                    // Apply HAVING if present
                    let mut result = result_batch;
                    if let Some(having_expr) = &stmt.having {
                        let mask = Self::evaluate_predicate(&result, having_expr)?;
                        result = compute::filter_record_batch(&result, &mask)
                            .map_err(|e| err_data( e.to_string()))?;
                    }
                    
                    // Apply ORDER BY with top-k optimization
                    if !stmt.order_by.is_empty() {
                        let resolved_ob = Self::resolve_order_by_cols(&stmt.columns, &stmt.order_by);
                        let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
                        result = Self::apply_order_by_topk(&result, &resolved_ob, k)?;
                    }
                    
                    // Apply LIMIT + OFFSET
                    if stmt.limit.is_some() || stmt.offset.is_some() {
                        result = Self::apply_limit_offset(&result, stmt.limit, stmt.offset)?;
                    }
                    
                    return Ok(ApexResult::Data(result));
                }
                Err(_) => {
                    // Fall through to general path
                }
            }
        }
        
        let typed_group_cols: Vec<Option<TypedCol>> = group_cols.iter()
            .map(|col_name| {
                batch.column_by_name(col_name).map(|col| {
                    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                        TypedCol::Int64(arr)
                    } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                        // Build runtime dictionary: string -> unique ID
                        // This converts O(string_len) hashing to O(1) integer operations
                        let mut dict: AHashMap<&str, u32> = AHashMap::with_capacity(1000);
                        let mut next_id = 1u32; // 0 reserved for NULL
                        let indices: Vec<u32> = (0..num_rows)
                            .map(|i| {
                                if arr.is_null(i) {
                                    0u32
                                } else {
                                    let s = arr.value(i);
                                    *dict.entry(s).or_insert_with(|| {
                                        let id = next_id;
                                        next_id += 1;
                                        id
                                    })
                                }
                            })
                            .collect();
                        TypedCol::StringDict(arr, indices)
                    } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                        TypedCol::Float64(arr)
                    } else if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
                        TypedCol::Bool(arr)
                    } else {
                        TypedCol::Other(col)
                    }
                })
            })
            .collect();
        
        // Find aggregate columns for incremental updates
        let mut agg_col_int: Option<&Int64Array> = None;
        let mut agg_col_float: Option<&Float64Array> = None;
        
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { column: Some(col_name), .. } = col {
                let actual_col = col_name.trim_matches('"');
                let actual_col = if let Some(dot_pos) = actual_col.rfind('.') {
                    &actual_col[dot_pos + 1..]
                } else {
                    actual_col
                };
                if actual_col != "*" {
                    if let Some(arr) = batch.column_by_name(actual_col) {
                        if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                            agg_col_int = Some(int_arr);
                        } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                            agg_col_float = Some(float_arr);
                        }
                    }
                }
                break;
            }
        }
        
        // Pre-compute all group keys (row_idx -> hash) for fast parallel access
        // OPTIMIZATION: Parallel hash computation for large datasets
        use rayon::prelude::*;
        let group_keys: Vec<u64> = if num_rows > 50_000 {
            // Parallel hash computation
            (0..num_rows)
                .into_par_iter()
                .map(|row_idx| {
                    let mut hasher = AHasher::default();
                    for col_opt in &typed_group_cols {
                        match col_opt {
                            Some(TypedCol::Int64(arr)) => {
                                if !arr.is_null(row_idx) {
                                    hasher.write_i64(arr.value(row_idx));
                                } else {
                                    hasher.write_u8(0);
                                }
                            }
                            Some(TypedCol::StringDict(_arr, indices)) => {
                                hasher.write_u32(indices[row_idx]);
                            }
                            Some(TypedCol::Float64(arr)) => {
                                if !arr.is_null(row_idx) {
                                    hasher.write_u64(arr.value(row_idx).to_bits());
                                } else {
                                    hasher.write_u8(0);
                                }
                            }
                            Some(TypedCol::Bool(arr)) => {
                                if !arr.is_null(row_idx) {
                                    hasher.write_u8(arr.value(row_idx) as u8);
                                } else {
                                    hasher.write_u8(2);
                                }
                            }
                            Some(TypedCol::Other(col)) => {
                                hasher.write_u64(Self::hash_array_value_fast(col, row_idx));
                            }
                            None => {}
                        }
                    }
                    hasher.finish()
                })
                .collect()
        } else {
            // Sequential for small datasets
            (0..num_rows)
                .map(|row_idx| {
                    let mut hasher = AHasher::default();
                    for col_opt in &typed_group_cols {
                        match col_opt {
                            Some(TypedCol::Int64(arr)) => {
                                if !arr.is_null(row_idx) {
                                    hasher.write_i64(arr.value(row_idx));
                                } else {
                                    hasher.write_u8(0);
                                }
                            }
                            Some(TypedCol::StringDict(_arr, indices)) => {
                                hasher.write_u32(indices[row_idx]);
                            }
                            Some(TypedCol::Float64(arr)) => {
                                if !arr.is_null(row_idx) {
                                    hasher.write_u64(arr.value(row_idx).to_bits());
                                } else {
                                    hasher.write_u8(0);
                                }
                            }
                            Some(TypedCol::Bool(arr)) => {
                                if !arr.is_null(row_idx) {
                                    hasher.write_u8(arr.value(row_idx) as u8);
                                } else {
                                    hasher.write_u8(2);
                                }
                            }
                            Some(TypedCol::Other(col)) => {
                                hasher.write_u64(Self::hash_array_value_fast(col, row_idx));
                            }
                            None => {}
                        }
                    }
                    hasher.finish()
                })
                .collect()
        };
        
        // Pre-compute aggregate values for parallel access
        let agg_int_vals: Option<Vec<Option<i64>>> = agg_col_int.map(|arr| {
            (0..num_rows).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) }).collect()
        });
        let agg_float_vals: Option<Vec<Option<f64>>> = agg_col_float.map(|arr| {
            (0..num_rows).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) }).collect()
        });
        
        // Parallel partitioned aggregation for large datasets
        use rayon::prelude::*;
        let use_parallel = num_rows > 50_000;
        
        let groups: AHashMap<u64, GroupState> = if use_parallel {
            let num_partitions = rayon::current_num_threads().max(4);
            let partition_size = (num_rows + num_partitions - 1) / num_partitions;
            
            // Each partition aggregates independently
            let partition_results: Vec<AHashMap<u64, GroupState>> = (0..num_partitions)
                .into_par_iter()
                .map(|p| {
                    let start = p * partition_size;
                    let end = ((p + 1) * partition_size).min(num_rows);
                    let mut local: AHashMap<u64, GroupState> = AHashMap::with_capacity(estimated_groups / num_partitions + 1);
                    
                    for row_idx in start..end {
                        let key = group_keys[row_idx];
                        let state = local.entry(key).or_insert_with(|| GroupState::new(row_idx));
                        state.count += 1;
                        
                        if let Some(ref vals) = agg_int_vals {
                            if let Some(val) = vals[row_idx] {
                                state.sum_int = state.sum_int.wrapping_add(val);
                                state.min_int = Some(state.min_int.map_or(val, |m| m.min(val)));
                                state.max_int = Some(state.max_int.map_or(val, |m| m.max(val)));
                            }
                        }
                        if let Some(ref vals) = agg_float_vals {
                            if let Some(val) = vals[row_idx] {
                                state.sum_float += val;
                                state.min_float = Some(state.min_float.map_or(val, |m| m.min(val)));
                                state.max_float = Some(state.max_float.map_or(val, |m| m.max(val)));
                            }
                        }
                    }
                    local
                })
                .collect();
            
            // Merge partition results
            let mut merged: AHashMap<u64, GroupState> = AHashMap::with_capacity(estimated_groups);
            for local in partition_results {
                for (key, state) in local {
                    merged.entry(key)
                        .and_modify(|e| {
                            e.count += state.count;
                            e.sum_int = e.sum_int.wrapping_add(state.sum_int);
                            e.sum_float += state.sum_float;
                            if let Some(v) = state.min_int {
                                e.min_int = Some(e.min_int.map_or(v, |m| m.min(v)));
                            }
                            if let Some(v) = state.max_int {
                                e.max_int = Some(e.max_int.map_or(v, |m| m.max(v)));
                            }
                            if let Some(v) = state.min_float {
                                e.min_float = Some(e.min_float.map_or(v, |m| m.min(v)));
                            }
                            if let Some(v) = state.max_float {
                                e.max_float = Some(e.max_float.map_or(v, |m| m.max(v)));
                            }
                        })
                        .or_insert(state);
                }
            }
            merged
        } else {
            // Sequential for small datasets
            let mut groups: AHashMap<u64, GroupState> = AHashMap::with_capacity(estimated_groups);
            for row_idx in 0..num_rows {
                let key = group_keys[row_idx];
                let state = groups.entry(key).or_insert_with(|| GroupState::new(row_idx));
                state.count += 1;
                
                if let Some(ref vals) = agg_int_vals {
                    if let Some(val) = vals[row_idx] {
                        state.sum_int = state.sum_int.wrapping_add(val);
                        state.min_int = Some(state.min_int.map_or(val, |m| m.min(val)));
                        state.max_int = Some(state.max_int.map_or(val, |m| m.max(val)));
                    }
                }
                if let Some(ref vals) = agg_float_vals {
                    if let Some(val) = vals[row_idx] {
                        state.sum_float += val;
                        state.min_float = Some(state.min_float.map_or(val, |m| m.min(val)));
                        state.max_float = Some(state.max_float.map_or(val, |m| m.max(val)));
                    }
                }
            }
            groups
        };
        
        // Build result arrays from group states
        let num_groups = groups.len();
        let states: Vec<GroupState> = groups.into_values().collect();
        
        let mut result_fields: Vec<Field> = Vec::new();
        let mut result_arrays: Vec<ArrayRef> = Vec::new();
        
        for col in &stmt.columns {
            match col {
                SelectColumn::Column(name) | SelectColumn::ColumnAlias { column: name, .. } => {
                    let col_name = name.trim_matches('"');
                    let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                        &col_name[dot_pos + 1..]
                    } else {
                        col_name
                    };
                    let output_name = match col {
                        SelectColumn::ColumnAlias { alias, .. } => alias.as_str(),
                        _ => actual_col,
                    };
                    
                    if let Some(src_col) = batch.column_by_name(actual_col) {
                        // Take value from first row of each group
                        let first_indices: Vec<usize> = states.iter().map(|s| s.first_row).collect();
                        let indices_arr = arrow::array::UInt32Array::from(first_indices.iter().map(|&i| i as u32).collect::<Vec<_>>());
                        let taken = compute::take(src_col.as_ref(), &indices_arr, None)
                            .map_err(|e| err_data( e.to_string()))?;
                        result_fields.push(Field::new(output_name, taken.data_type().clone(), true));
                        result_arrays.push(taken);
                    }
                }
                SelectColumn::Aggregate { func, column, alias, .. } => {
                    let fn_name = match func { AggregateFunc::Count => "COUNT", AggregateFunc::Sum => "SUM", AggregateFunc::Avg => "AVG", AggregateFunc::Min => "MIN", AggregateFunc::Max => "MAX" };
                    let output_name = alias.clone().unwrap_or_else(|| if let Some(c) = column { format!("{}({})", fn_name, c) } else { format!("{}(*)", fn_name) });
                    let has_int = agg_col_int.is_some();
                    match func {
                        AggregateFunc::Count => { result_fields.push(Field::new(&output_name, ArrowDataType::Int64, false)); result_arrays.push(Arc::new(Int64Array::from(states.iter().map(|s| s.count).collect::<Vec<_>>()))); }
                        AggregateFunc::Sum => { if has_int { result_fields.push(Field::new(&output_name, ArrowDataType::Int64, false)); result_arrays.push(Arc::new(Int64Array::from(states.iter().map(|s| s.sum_int).collect::<Vec<_>>()))); } else { result_fields.push(Field::new(&output_name, ArrowDataType::Float64, false)); result_arrays.push(Arc::new(Float64Array::from(states.iter().map(|s| s.sum_float).collect::<Vec<_>>()))); } }
                        AggregateFunc::Avg => { let avgs: Vec<Option<f64>> = states.iter().map(|s| if s.count > 0 { Some(if has_int { s.sum_int as f64 / s.count as f64 } else { s.sum_float / s.count as f64 }) } else { None }).collect(); result_fields.push(Field::new(&output_name, ArrowDataType::Float64, true)); result_arrays.push(Arc::new(Float64Array::from(avgs))); }
                        AggregateFunc::Min => { if has_int { result_fields.push(Field::new(&output_name, ArrowDataType::Int64, true)); result_arrays.push(Arc::new(Int64Array::from(states.iter().map(|s| s.min_int).collect::<Vec<_>>()))); } else { result_fields.push(Field::new(&output_name, ArrowDataType::Float64, true)); result_arrays.push(Arc::new(Float64Array::from(states.iter().map(|s| s.min_float).collect::<Vec<_>>()))); } }
                        AggregateFunc::Max => { if has_int { result_fields.push(Field::new(&output_name, ArrowDataType::Int64, true)); result_arrays.push(Arc::new(Int64Array::from(states.iter().map(|s| s.max_int).collect::<Vec<_>>()))); } else { result_fields.push(Field::new(&output_name, ArrowDataType::Float64, true)); result_arrays.push(Arc::new(Float64Array::from(states.iter().map(|s| s.max_float).collect::<Vec<_>>()))); } }
                    }
                }
                _ => {}
            }
        }
        
        if result_fields.is_empty() {
            return Ok(ApexResult::Scalar(num_groups as i64));
        }
        
        let schema = Arc::new(Schema::new(result_fields));
        let mut result = RecordBatch::try_new(schema, result_arrays)
            .map_err(|e| err_data( e.to_string()))?;
        
        // Apply HAVING clause if present
        if let Some(having_expr) = &stmt.having {
            let mask = Self::evaluate_predicate(&result, having_expr)?;
            result = compute::filter_record_batch(&result, &mask)
                .map_err(|e| err_data( e.to_string()))?;
        }
        
        // Apply ORDER BY with top-k optimization if LIMIT is present
        if !stmt.order_by.is_empty() {
            let resolved_ob = Self::resolve_order_by_cols(&stmt.columns, &stmt.order_by);
            let k = stmt.limit.map(|l| l + stmt.offset.unwrap_or(0));
            result = Self::apply_order_by_topk(&result, &resolved_ob, k)?;
        }
        
        // Apply LIMIT + OFFSET
        if stmt.limit.is_some() || stmt.offset.is_some() {
            result = Self::apply_limit_offset(&result, stmt.limit, stmt.offset)?;
        }
        
        Ok(ApexResult::Data(result))
    }
    
    /// Original GROUP BY with full row indices (for complex queries with DISTINCT or expressions)
    fn execute_group_by_with_indices(
        batch: &RecordBatch,
        stmt: &SelectStatement,
        group_cols: &[String],
    ) -> io::Result<ApexResult> {
        // Create groups: key -> row indices (using AHashMap for speed)
        let num_rows = batch.num_rows();
        let estimated_groups = (num_rows / 10).max(16); // Estimate ~10 rows per group
        let mut groups: AHashMap<u64, Vec<usize>> = AHashMap::with_capacity(estimated_groups);
        
        // OPTIMIZATION: Pre-downcast columns to typed arrays for faster access
        // This avoids repeated dynamic dispatch in the hot loop
        enum TypedColumn<'a> {
            Int64(&'a Int64Array),
            Float64(&'a Float64Array),
            String(&'a StringArray),
            Bool(&'a BooleanArray),
            Other(&'a ArrayRef),
        }
        
        let typed_cols: Vec<Option<TypedColumn>> = group_cols.iter()
            .map(|col_name| {
                batch.column_by_name(col_name).map(|col| {
                    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                        TypedColumn::Int64(arr)
                    } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                        TypedColumn::String(arr)
                    } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                        TypedColumn::Float64(arr)
                    } else if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
                        TypedColumn::Bool(arr)
                    } else {
                        TypedColumn::Other(col)
                    }
                })
            })
            .collect();
        
        // Build groups with optimized type-specific hashing
        for row_idx in 0..num_rows {
            let mut hasher = AHasher::default();
            for col_opt in &typed_cols {
                match col_opt {
                    Some(TypedColumn::Int64(arr)) => {
                        if !arr.is_null(row_idx) {
                            hasher.write_i64(arr.value(row_idx));
                        } else {
                            hasher.write_u8(0);
                        }
                    }
                    Some(TypedColumn::String(arr)) => {
                        if !arr.is_null(row_idx) {
                            hasher.write(arr.value(row_idx).as_bytes());
                        } else {
                            hasher.write_u8(0);
                        }
                    }
                    Some(TypedColumn::Float64(arr)) => {
                        if !arr.is_null(row_idx) {
                            hasher.write_u64(arr.value(row_idx).to_bits());
                        } else {
                            hasher.write_u8(0);
                        }
                    }
                    Some(TypedColumn::Bool(arr)) => {
                        if !arr.is_null(row_idx) {
                            hasher.write_u8(arr.value(row_idx) as u8);
                        } else {
                            hasher.write_u8(2);
                        }
                    }
                    Some(TypedColumn::Other(col)) => {
                        hasher.write_u64(Self::hash_array_value_fast(col, row_idx));
                    }
                    None => {}
                }
            }
            let key = hasher.finish();
            groups.entry(key).or_insert_with(|| Vec::with_capacity(16)).push(row_idx);
        }

        // Build result arrays
        let mut result_fields: Vec<Field> = Vec::new();
        let mut result_arrays: Vec<ArrayRef> = Vec::new();
        
        let num_groups = groups.len();
        let group_indices: Vec<Vec<usize>> = groups.into_values().collect();

        for col in &stmt.columns {
            match col {
                SelectColumn::Column(name) => {
                    let col_name = name.trim_matches('"');
                    // Strip table prefix if present (e.g., "u.tier" -> "tier")
                    let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                        &col_name[dot_pos + 1..]
                    } else {
                        col_name
                    };
                    if let Some(src_col) = batch.column_by_name(actual_col) {
                        let (field, array) = Self::take_first_from_groups(src_col, &group_indices, actual_col)?;
                        result_fields.push(field);
                        result_arrays.push(array);
                    }
                }
                SelectColumn::ColumnAlias { column, alias } => {
                    let col_name = column.trim_matches('"');
                    // Strip table prefix if present
                    let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                        &col_name[dot_pos + 1..]
                    } else {
                        col_name
                    };
                    if let Some(src_col) = batch.column_by_name(actual_col) {
                        let (field, array) = Self::take_first_from_groups(src_col, &group_indices, alias)?;
                        result_fields.push(field);
                        result_arrays.push(array);
                    }
                }
                SelectColumn::Aggregate { func, column, distinct, alias } => {
                    let (field, array) = Self::compute_aggregate_for_groups(batch, func, column, alias, &group_indices, *distinct)?;
                    result_fields.push(field);
                    result_arrays.push(array);
                }
                SelectColumn::Expression { expr, alias } => {
                    // For expressions containing aggregates (like CASE WHEN SUM(x) > 100 THEN ...),
                    // we need to evaluate the expression for each group
                    let (field, array) = Self::evaluate_expr_for_groups(batch, expr, alias.as_deref(), &group_indices)?;
                    result_fields.push(field);
                    result_arrays.push(array);
                }
                _ => {}
            }
        }

        if result_fields.is_empty() {
            return Ok(ApexResult::Scalar(num_groups as i64));
        }

        let schema = Arc::new(Schema::new(result_fields));
        let mut result = RecordBatch::try_new(schema, result_arrays)
            .map_err(|e| err_data( e.to_string()))?;

        // Apply HAVING clause if present
        if let Some(having_expr) = &stmt.having {
            let mask = Self::evaluate_predicate(&result, having_expr)?;
            result = compute::filter_record_batch(&result, &mask)
                .map_err(|e| err_data( e.to_string()))?;
        }

        // Apply ORDER BY if present
        if !stmt.order_by.is_empty() {
            let resolved_ob = Self::resolve_order_by_cols(&stmt.columns, &stmt.order_by);
            result = Self::apply_order_by(&result, &resolved_ob)?;
        }

        Ok(ApexResult::Data(result))
    }

    /// Evaluate expression for groups (handles CASE with aggregates)
    fn evaluate_expr_for_groups(
        batch: &RecordBatch,
        expr: &SqlExpr,
        alias: Option<&str>,
        group_indices: &[Vec<usize>],
    ) -> io::Result<(Field, ArrayRef)> {
        let output_name = alias.unwrap_or("expr");
        let num_groups = group_indices.len();
        
        // For CASE expressions with aggregates, we need to:
        // 1. For each group, compute the aggregate values
        // 2. Evaluate the CASE condition using those aggregates
        // 3. Return the CASE result for each group
        
        match expr {
            SqlExpr::Case { when_then, else_expr } => {
                // Evaluate CASE for each group
                let mut string_results: Vec<Option<String>> = Vec::with_capacity(num_groups);
                let mut int_results: Vec<Option<i64>> = Vec::with_capacity(num_groups);
                let mut is_string = false;
                
                // Determine result type from first THEN expression
                if let Some((_, then_expr)) = when_then.first() {
                    if matches!(then_expr, SqlExpr::Literal(Value::String(_))) {
                        is_string = true;
                    }
                }
                
                for indices in group_indices {
                    // Create a sub-batch for this group
                    let group_batch = Self::create_group_batch(batch, indices)?;
                    
                    // For each WHEN clause, check if condition is true for this group
                    let mut matched = false;
                    for (cond_expr, then_expr) in when_then {
                        // Evaluate condition - if it contains aggregates, compute them
                        let cond_result = Self::evaluate_aggregate_condition(&group_batch, cond_expr)?;
                        
                        if cond_result {
                            // Evaluate THEN expression
                            if is_string {
                                if let SqlExpr::Literal(Value::String(s)) = then_expr {
                                    string_results.push(Some(s.clone()));
                                } else {
                                    string_results.push(None);
                                }
                            } else {
                                let then_arr = Self::evaluate_expr_to_array(&group_batch, then_expr)?;
                                if let Some(int_arr) = then_arr.as_any().downcast_ref::<Int64Array>() {
                                    int_results.push(if int_arr.len() > 0 && !int_arr.is_null(0) { Some(int_arr.value(0)) } else { None });
                                } else {
                                    int_results.push(None);
                                }
                            }
                            matched = true;
                            break;
                        }
                    }
                    
                    if !matched {
                        // Use ELSE value
                        if let Some(else_e) = else_expr {
                            if is_string {
                                if let SqlExpr::Literal(Value::String(s)) = else_e.as_ref() {
                                    string_results.push(Some(s.clone()));
                                } else {
                                    string_results.push(None);
                                }
                            } else {
                                let else_arr = Self::evaluate_expr_to_array(&group_batch, else_e)?;
                                if let Some(int_arr) = else_arr.as_any().downcast_ref::<Int64Array>() {
                                    int_results.push(if int_arr.len() > 0 && !int_arr.is_null(0) { Some(int_arr.value(0)) } else { None });
                                } else {
                                    int_results.push(None);
                                }
                            }
                        } else {
                            if is_string {
                                string_results.push(None);
                            } else {
                                int_results.push(None);
                            }
                        }
                    }
                }
                
                if is_string {
                    let array = Arc::new(StringArray::from(string_results)) as ArrayRef;
                    Ok((Field::new(output_name, arrow::datatypes::DataType::Utf8, true), array))
                } else {
                    let array = Arc::new(Int64Array::from(int_results)) as ArrayRef;
                    Ok((Field::new(output_name, arrow::datatypes::DataType::Int64, true), array))
                }
            }
            _ => {
                // For non-CASE expressions, evaluate normally on first row of each group
                let first_indices: Vec<usize> = group_indices.iter().map(|g| g[0]).collect();
                let array = Self::evaluate_expr_to_array(batch, expr)?;
                
                // Take values at first indices
                if let Some(str_arr) = array.as_any().downcast_ref::<StringArray>() {
                    let values: Vec<Option<String>> = first_indices.iter().map(|&i| {
                        if str_arr.is_null(i) { None } else { Some(str_arr.value(i).to_string()) }
                    }).collect();
                    let result = Arc::new(StringArray::from(values)) as ArrayRef;
                    Ok((Field::new(output_name, arrow::datatypes::DataType::Utf8, true), result))
                } else if let Some(int_arr) = array.as_any().downcast_ref::<Int64Array>() {
                    let values: Vec<Option<i64>> = first_indices.iter().map(|&i| {
                        if int_arr.is_null(i) { None } else { Some(int_arr.value(i)) }
                    }).collect();
                    let result = Arc::new(Int64Array::from(values)) as ArrayRef;
                    Ok((Field::new(output_name, arrow::datatypes::DataType::Int64, true), result))
                } else {
                    Err(err_data( "Unsupported expression type"))
                }
            }
        }
    }

    /// Create a sub-batch containing only the specified row indices
    fn create_group_batch(batch: &RecordBatch, indices: &[usize]) -> io::Result<RecordBatch> {
        let indices_array = arrow::array::UInt64Array::from(indices.iter().map(|&i| i as u64).collect::<Vec<_>>());
        compute::take_record_batch(batch, &indices_array)
            .map_err(|e| err_data( e.to_string()))
    }

    /// Evaluate condition that may contain aggregate functions
    fn evaluate_aggregate_condition(batch: &RecordBatch, expr: &SqlExpr) -> io::Result<bool> {
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                // Check if this is a comparison operation
                match op {
                    BinaryOperator::Ge | BinaryOperator::Gt | BinaryOperator::Le | 
                    BinaryOperator::Lt | BinaryOperator::Eq | BinaryOperator::NotEq => {
                        // Evaluate left and right, handling aggregates
                        let left_val = Self::evaluate_aggregate_expr_scalar(batch, left)?;
                        let right_val = Self::evaluate_aggregate_expr_scalar(batch, right)?;
                        
                        match op {
                            BinaryOperator::Ge => Ok(left_val >= right_val),
                            BinaryOperator::Gt => Ok(left_val > right_val),
                            BinaryOperator::Le => Ok(left_val <= right_val),
                            BinaryOperator::Lt => Ok(left_val < right_val),
                            BinaryOperator::Eq => Ok((left_val - right_val).abs() < f64::EPSILON),
                            BinaryOperator::NotEq => Ok((left_val - right_val).abs() >= f64::EPSILON),
                            _ => unreachable!(),
                        }
                    }
                    _ => {
                        // For logical operators, evaluate as predicate
                        let result = Self::evaluate_predicate(batch, expr)?;
                        Ok(result.len() > 0 && result.value(0))
                    }
                }
            }
            _ => {
                // For other expressions, try to evaluate as predicate
                let result = Self::evaluate_predicate(batch, expr)?;
                Ok(result.len() > 0 && result.value(0))
            }
        }
    }

    /// Evaluate expression that may be an aggregate, returning scalar value
    fn evaluate_aggregate_expr_scalar(batch: &RecordBatch, expr: &SqlExpr) -> io::Result<f64> {
        match expr {
            SqlExpr::Function { name, args } => {
                // Check if this is an aggregate function (zero-allocation)
                let func_opt = if name.eq_ignore_ascii_case("SUM") { Some(AggregateFunc::Sum) }
                    else if name.eq_ignore_ascii_case("COUNT") { Some(AggregateFunc::Count) }
                    else if name.eq_ignore_ascii_case("AVG") { Some(AggregateFunc::Avg) }
                    else if name.eq_ignore_ascii_case("MIN") { Some(AggregateFunc::Min) }
                    else if name.eq_ignore_ascii_case("MAX") { Some(AggregateFunc::Max) }
                    else { None };
                if let Some(func) = func_opt {
                    let col_name = if args.is_empty() {
                        "*"
                    } else if let SqlExpr::Column(c) = &args[0] {
                        c.as_str()
                    } else {
                        "*"
                    };
                    // Create group indices covering all rows in the batch
                    let all_indices: Vec<usize> = (0..batch.num_rows()).collect();
                    let (_, result_arr) = Self::compute_aggregate_for_groups(batch, &func, &Some(col_name.to_string()), &None, &[all_indices], false)?;
                    if let Some(int_arr) = result_arr.as_any().downcast_ref::<Int64Array>() {
                        Ok(if int_arr.len() > 0 && !int_arr.is_null(0) { int_arr.value(0) as f64 } else { 0.0 })
                    } else if let Some(float_arr) = result_arr.as_any().downcast_ref::<Float64Array>() {
                        Ok(if float_arr.len() > 0 && !float_arr.is_null(0) { float_arr.value(0) } else { 0.0 })
                    } else {
                        Ok(0.0)
                    }
                } else {
                    let arr = Self::evaluate_expr_to_array(batch, expr)?;
                    Self::extract_scalar_from_array(&arr)
                }
            }
            SqlExpr::Literal(Value::Int64(i)) => Ok(*i as f64),
            SqlExpr::Literal(Value::Float64(f)) => Ok(*f),
            _ => {
                let arr = Self::evaluate_expr_to_array(batch, expr)?;
                Self::extract_scalar_from_array(&arr)
            }
        }
    }

    /// Extract scalar value from array
    fn extract_scalar_from_array(arr: &ArrayRef) -> io::Result<f64> {
        if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
            Ok(if int_arr.len() > 0 && !int_arr.is_null(0) { int_arr.value(0) as f64 } else { 0.0 })
        } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
            Ok(if float_arr.len() > 0 && !float_arr.is_null(0) { float_arr.value(0) } else { 0.0 })
        } else {
            Ok(0.0)
        }
    }

    /// Check if an expression contains a correlated subquery
    fn has_correlated_subquery(expr: &SqlExpr) -> bool {
        match expr {
            SqlExpr::ExistsSubquery { .. } | SqlExpr::InSubquery { .. } | SqlExpr::ScalarSubquery { .. } => true,
            SqlExpr::BinaryOp { left, right, .. } => {
                Self::has_correlated_subquery(left) || Self::has_correlated_subquery(right)
            }
            SqlExpr::UnaryOp { expr, .. } => Self::has_correlated_subquery(expr),
            SqlExpr::Paren(inner) => Self::has_correlated_subquery(inner),
            _ => false,
        }
    }

    fn coerce_numeric_for_comparison(left: ArrayRef, right: ArrayRef) -> io::Result<(ArrayRef, ArrayRef)> {
        use arrow::compute::cast;
        use arrow::datatypes::DataType;

        let l_is_f = left.as_any().downcast_ref::<Float64Array>().is_some();
        let r_is_f = right.as_any().downcast_ref::<Float64Array>().is_some();
        let l_is_i = left.as_any().downcast_ref::<Int64Array>().is_some();
        let r_is_i = right.as_any().downcast_ref::<Int64Array>().is_some();
        let l_is_u = left.as_any().downcast_ref::<UInt64Array>().is_some();
        let r_is_u = right.as_any().downcast_ref::<UInt64Array>().is_some();

        if l_is_f && r_is_i {
            let r2 = cast(&right, &DataType::Float64)
                .map_err(|e| err_data( e.to_string()))?;
            return Ok((left, r2));
        }
        if l_is_i && r_is_f {
            let l2 = cast(&left, &DataType::Float64)
                .map_err(|e| err_data( e.to_string()))?;
            return Ok((l2, right));
        }
        // UInt64 ↔ Int64 coercion (e.g. _id column is UInt64, literals are Int64)
        if l_is_u && r_is_i {
            let l2 = cast(&left, &DataType::Int64)
                .map_err(|e| err_data( e.to_string()))?;
            return Ok((l2, right));
        }
        if l_is_i && r_is_u {
            let r2 = cast(&right, &DataType::Int64)
                .map_err(|e| err_data( e.to_string()))?;
            return Ok((left, r2));
        }

        Ok((left, right))
    }

    /// OPTIMIZED: Extract _id = X pattern for O(1) lookup
    /// Returns Some(id) if WHERE clause is simple `_id = literal` or `literal = _id`
    #[inline]
    fn extract_id_equality_filter(expr: &SqlExpr) -> Option<u64> {
        use crate::query::sql_parser::BinaryOperator;
        
        if let SqlExpr::BinaryOp { left, op, right } = expr {
            if !matches!(op, BinaryOperator::Eq) {
                return None;
            }
            
            // Check _id = literal pattern
            if let SqlExpr::Column(col) = left.as_ref() {
                let col_name = col.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if actual_col == "_id" {
                    if let SqlExpr::Literal(Value::Int64(id)) = right.as_ref() {
                        return Some(*id as u64);
                    }
                }
            }
            
            // Check literal = _id pattern
            if let SqlExpr::Column(col) = right.as_ref() {
                let col_name = col.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if actual_col == "_id" {
                    if let SqlExpr::Literal(Value::Int64(id)) = left.as_ref() {
                        return Some(*id as u64);
                    }
                }
            }
        }
        None
    }

    /// Extract simple string equality filter: column = 'literal' or 'literal' = column
    /// Returns (column_name, literal_value, is_equality) if matches, None otherwise
    #[inline]
    fn extract_simple_string_filter(expr: &SqlExpr) -> Option<(String, String, bool)> {
        use crate::query::sql_parser::BinaryOperator;
        
        if let SqlExpr::BinaryOp { left, op, right } = expr {
            let is_eq = matches!(op, BinaryOperator::Eq);
            let is_neq = matches!(op, BinaryOperator::NotEq);
            
            if !is_eq && !is_neq {
                return None;
            }
            
            // Check column = 'literal' pattern
            if let (SqlExpr::Column(col), SqlExpr::Literal(Value::String(lit))) = (left.as_ref(), right.as_ref()) {
                let col_name = col.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                return Some((actual_col.to_string(), lit.clone(), is_eq));
            }
            
            // Check 'literal' = column pattern
            if let (SqlExpr::Literal(Value::String(lit)), SqlExpr::Column(col)) = (left.as_ref(), right.as_ref()) {
                let col_name = col.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                return Some((actual_col.to_string(), lit.clone(), is_eq));
            }
        }
        
        None
    }

    /// Collect column names from an expression (for determining required columns)
    fn collect_columns_from_expr(expr: &SqlExpr, columns: &mut Vec<String>) {
        match expr {
            SqlExpr::Column(name) => {
                let col_name = name.trim_matches('"');
                // Strip table alias prefix if present
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if !columns.contains(&actual_col.to_string()) {
                    columns.push(actual_col.to_string());
                }
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                Self::collect_columns_from_expr(left, columns);
                Self::collect_columns_from_expr(right, columns);
            }
            SqlExpr::UnaryOp { expr, .. } => {
                Self::collect_columns_from_expr(expr, columns);
            }
            SqlExpr::Paren(inner) => {
                Self::collect_columns_from_expr(inner, columns);
            }
            SqlExpr::Function { args, .. } => {
                for arg in args {
                    Self::collect_columns_from_expr(arg, columns);
                }
            }
            SqlExpr::Case { when_then, else_expr } => {
                for (cond, then_expr) in when_then {
                    Self::collect_columns_from_expr(cond, columns);
                    Self::collect_columns_from_expr(then_expr, columns);
                }
                if let Some(else_e) = else_expr {
                    Self::collect_columns_from_expr(else_e, columns);
                }
            }
            SqlExpr::Cast { expr, .. } => {
                Self::collect_columns_from_expr(expr, columns);
            }
            SqlExpr::Between { column, low, high, .. } => {
                // Handle BETWEEN expression: column BETWEEN low AND high
                let col_name = column.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if !columns.contains(&actual_col.to_string()) {
                    columns.push(actual_col.to_string());
                }
                Self::collect_columns_from_expr(low, columns);
                Self::collect_columns_from_expr(high, columns);
            }
            SqlExpr::Like { column, .. } | SqlExpr::Regexp { column, .. } => {
                // Handle LIKE/REGEXP expressions
                let col_name = column.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if !columns.contains(&actual_col.to_string()) {
                    columns.push(actual_col.to_string());
                }
            }
            SqlExpr::In { column, .. } => {
                // Handle IN expression (values are literals, not column refs)
                let col_name = column.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if !columns.contains(&actual_col.to_string()) {
                    columns.push(actual_col.to_string());
                }
            }
            SqlExpr::IsNull { column, .. } => {
                // Handle IS NULL / IS NOT NULL (negated field handles both)
                let col_name = column.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if !columns.contains(&actual_col.to_string()) {
                    columns.push(actual_col.to_string());
                }
            }
            SqlExpr::ExistsSubquery { stmt } | SqlExpr::ScalarSubquery { stmt } => {
                // For correlated subqueries, we need all outer columns that might be referenced
                // The subquery might reference outer columns like u.user_id
                if let Some(ref where_clause) = stmt.where_clause {
                    Self::collect_columns_from_expr(where_clause, columns);
                }
            }
            SqlExpr::InSubquery { column, stmt, .. } => {
                let col_name = column.trim_matches('"');
                let actual_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                if !columns.contains(&actual_col.to_string()) {
                    columns.push(actual_col.to_string());
                }
                if let Some(ref where_clause) = stmt.where_clause {
                    Self::collect_columns_from_expr(where_clause, columns);
                }
            }
            SqlExpr::ArrayIndex { array, index } => {
                Self::collect_columns_from_expr(array, columns);
                Self::collect_columns_from_expr(index, columns);
            }
            _ => {}
        }
    }

    /// Check if this is a pure COUNT(*) query that can be answered from metadata
    /// Returns true if: SELECT COUNT(*) FROM table (no WHERE, no GROUP BY, no HAVING)
    fn is_pure_count_star(stmt: &SelectStatement) -> bool {
        // Must have exactly one column which is COUNT(*)
        if stmt.columns.len() != 1 {
            return false;
        }
        
        // Must be COUNT(*) aggregate
        let is_count_star = match &stmt.columns[0] {
            SelectColumn::Aggregate { func, column, distinct, .. } => {
                matches!(func, AggregateFunc::Count) 
                    && !distinct 
                    && column.as_ref().map(|c| c == "*" || c.chars().next().map(|ch| ch.is_ascii_digit()).unwrap_or(false)).unwrap_or(true)
            }
            _ => false,
        };
        
        if !is_count_star {
            return false;
        }
        
        // No WHERE clause
        if stmt.where_clause.is_some() {
            return false;
        }
        
        // No GROUP BY
        if !stmt.group_by.is_empty() {
            return false;
        }
        
        // No HAVING
        if stmt.having.is_some() {
            return false;
        }
        
        // No subquery in FROM
        if matches!(stmt.from, Some(FromItem::Subquery { .. })) {
            return false;
        }
        
        // No JOINs
        if !stmt.joins.is_empty() {
            return false;
        }
        
        true
    }

    /// Check if an expression contains an aggregate function (SUM, COUNT, AVG, MIN, MAX)
    fn expr_contains_aggregate(expr: &SqlExpr) -> bool {
        match expr {
            SqlExpr::Function { name, args } => name.eq_ignore_ascii_case("SUM") || name.eq_ignore_ascii_case("COUNT") || name.eq_ignore_ascii_case("AVG") || name.eq_ignore_ascii_case("MIN") || name.eq_ignore_ascii_case("MAX") || args.iter().any(Self::expr_contains_aggregate),
            SqlExpr::Case { when_then, else_expr } => when_then.iter().any(|(c, t)| Self::expr_contains_aggregate(c) || Self::expr_contains_aggregate(t)) || else_expr.as_ref().map_or(false, |e| Self::expr_contains_aggregate(e)),
            SqlExpr::BinaryOp { left, right, .. } => Self::expr_contains_aggregate(left) || Self::expr_contains_aggregate(right),
            SqlExpr::UnaryOp { expr, .. } | SqlExpr::Paren(expr) | SqlExpr::Cast { expr, .. } => Self::expr_contains_aggregate(expr),
            _ => false,
        }
    }
    
    /// Check if an expression contains a scalar subquery
    fn expr_contains_scalar_subquery(expr: &SqlExpr) -> bool {
        match expr {
            SqlExpr::ScalarSubquery { .. } => true,
            SqlExpr::BinaryOp { left, right, .. } => Self::expr_contains_scalar_subquery(left) || Self::expr_contains_scalar_subquery(right),
            SqlExpr::UnaryOp { expr, .. } | SqlExpr::Paren(expr) | SqlExpr::Cast { expr, .. } => Self::expr_contains_scalar_subquery(expr),
            SqlExpr::Case { when_then, else_expr } => when_then.iter().any(|(c, t)| Self::expr_contains_scalar_subquery(c) || Self::expr_contains_scalar_subquery(t)) || else_expr.as_ref().map_or(false, |e| Self::expr_contains_scalar_subquery(e)),
            _ => false,
        }
    }

    /// Take first value from each group
    fn take_first_from_groups(array: &ArrayRef, group_indices: &[Vec<usize>], output_name: &str) -> io::Result<(Field, ArrayRef)> {
        use arrow::datatypes::DataType;
        let first_indices: Vec<usize> = group_indices.iter().map(|g| g[0]).collect();
        match array.data_type() {
            DataType::Int64 => { let src = array.as_any().downcast_ref::<Int64Array>().unwrap(); Ok((Field::new(output_name, DataType::Int64, true), Arc::new(Int64Array::from(first_indices.iter().map(|&i| if src.is_null(i) { None } else { Some(src.value(i)) }).collect::<Vec<_>>())))) }
            DataType::Float64 => { let src = array.as_any().downcast_ref::<Float64Array>().unwrap(); Ok((Field::new(output_name, DataType::Float64, true), Arc::new(Float64Array::from(first_indices.iter().map(|&i| if src.is_null(i) { None } else { Some(src.value(i)) }).collect::<Vec<_>>())))) }
            DataType::Utf8 => { let src = array.as_any().downcast_ref::<StringArray>().unwrap(); Ok((Field::new(output_name, DataType::Utf8, true), Arc::new(StringArray::from(first_indices.iter().map(|&i| if src.is_null(i) { None } else { Some(src.value(i)) }).collect::<Vec<_>>())))) }
            DataType::Boolean => { let src = array.as_any().downcast_ref::<BooleanArray>().unwrap(); Ok((Field::new(output_name, DataType::Boolean, true), Arc::new(BooleanArray::from(first_indices.iter().map(|&i| if src.is_null(i) { None } else { Some(src.value(i)) }).collect::<Vec<_>>())))) }
            _ => Ok((Field::new(output_name, DataType::Int64, true), Arc::new(Int64Array::from(vec![None::<i64>; group_indices.len()]))))
        }
    }

    /// Compute aggregate for each group (parallelized for large datasets)
    fn compute_aggregate_for_groups(
        batch: &RecordBatch,
        func: &crate::query::AggregateFunc,
        column: &Option<String>,
        alias: &Option<String>,
        group_indices: &[Vec<usize>],
        distinct: bool,
    ) -> io::Result<(Field, ArrayRef)> {
        use crate::query::AggregateFunc;
        use ahash::AHashSet;
        use rayon::prelude::*;
        
        let func_name = match func {
            AggregateFunc::Count => "COUNT",
            AggregateFunc::Sum => "SUM",
            AggregateFunc::Avg => "AVG",
            AggregateFunc::Min => "MIN",
            AggregateFunc::Max => "MAX",
        };
        
        let output_name = alias.clone().unwrap_or_else(|| {
            if let Some(col) = column { format!("{}({})", func_name, col) } else { format!("{}(*)", func_name) }
        });

        // Strip table prefix from column name if present (e.g., "o.amount" -> "amount")
        let actual_column: Option<String> = column.as_ref().map(|c| {
            let trimmed = c.trim_matches('"');
            if let Some(dot_pos) = trimmed.rfind('.') {
                trimmed[dot_pos + 1..].to_string()
            } else {
                trimmed.to_string()
            }
        });

        match func {
            AggregateFunc::Count => {
                let use_parallel = group_indices.len() > 100;
                let counts: Vec<i64> = if let Some(col_name) = &actual_column {
                    if col_name == "*" || col_name.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                        if use_parallel {
                            group_indices.par_iter().map(|g| g.len() as i64).collect()
                        } else {
                            group_indices.iter().map(|g| g.len() as i64).collect()
                        }
                    } else if let Some(array) = batch.column_by_name(col_name) {
                        if distinct {
                            // COUNT(DISTINCT column) - count unique values per group
                            if let Some(int_arr) = array.as_any().downcast_ref::<Int64Array>() {
                                if use_parallel {
                                    group_indices.par_iter().map(|g| {
                                        let unique: AHashSet<i64> = g.iter()
                                            .filter(|&&i| !int_arr.is_null(i))
                                            .map(|&i| int_arr.value(i))
                                            .collect();
                                        unique.len() as i64
                                    }).collect()
                                } else {
                                    group_indices.iter().map(|g| {
                                        let unique: AHashSet<i64> = g.iter()
                                            .filter(|&&i| !int_arr.is_null(i))
                                            .map(|&i| int_arr.value(i))
                                            .collect();
                                        unique.len() as i64
                                    }).collect()
                                }
                            } else if let Some(str_arr) = array.as_any().downcast_ref::<StringArray>() {
                                if use_parallel {
                                    group_indices.par_iter().map(|g| {
                                        let unique: AHashSet<&str> = g.iter()
                                            .filter(|&&i| !str_arr.is_null(i))
                                            .map(|&i| str_arr.value(i))
                                            .collect();
                                        unique.len() as i64
                                    }).collect()
                                } else {
                                    group_indices.iter().map(|g| {
                                        let unique: AHashSet<&str> = g.iter()
                                            .filter(|&&i| !str_arr.is_null(i))
                                            .map(|&i| str_arr.value(i))
                                            .collect();
                                        unique.len() as i64
                                    }).collect()
                                }
                            } else {
                                if use_parallel {
                                    group_indices.par_iter().map(|g| g.iter().filter(|&&i| !array.is_null(i)).count() as i64).collect()
                                } else {
                                    group_indices.iter().map(|g| g.iter().filter(|&&i| !array.is_null(i)).count() as i64).collect()
                                }
                            }
                        } else {
                            if use_parallel {
                                group_indices.par_iter().map(|g| g.iter().filter(|&&i| !array.is_null(i)).count() as i64).collect()
                            } else {
                                group_indices.iter().map(|g| g.iter().filter(|&&i| !array.is_null(i)).count() as i64).collect()
                            }
                        }
                    } else {
                        vec![0; group_indices.len()]
                    }
                } else {
                    if use_parallel {
                        group_indices.par_iter().map(|g| g.len() as i64).collect()
                    } else {
                        group_indices.iter().map(|g| g.len() as i64).collect()
                    }
                };
                Ok((Field::new(&output_name, ArrowDataType::Int64, false), Arc::new(Int64Array::from(counts))))
            }
            AggregateFunc::Sum => {
                let col_name = actual_column.as_ref().ok_or_else(|| err_input( "SUM requires column"))?;
                let array = batch.column_by_name(col_name).ok_or_else(|| err_not_found(format!("Column: {}", col_name)))?;
                
                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    // Fast path: direct slice access with loop unrolling
                    let values = int_array.values();
                    let use_parallel = group_indices.len() > 100;
                    
                    let sums: Vec<i64> = if use_parallel {
                        group_indices.par_iter().map(|g| {
                            // Unrolled summation for better instruction pipelining
                            let mut sum0: i64 = 0;
                            let mut sum1: i64 = 0;
                            let mut sum2: i64 = 0;
                            let mut sum3: i64 = 0;
                            let chunks = g.chunks_exact(4);
                            let remainder = chunks.remainder();
                            for chunk in chunks {
                                sum0 = sum0.wrapping_add(values[chunk[0]]);
                                sum1 = sum1.wrapping_add(values[chunk[1]]);
                                sum2 = sum2.wrapping_add(values[chunk[2]]);
                                sum3 = sum3.wrapping_add(values[chunk[3]]);
                            }
                            for &i in remainder {
                                sum0 = sum0.wrapping_add(values[i]);
                            }
                            sum0.wrapping_add(sum1).wrapping_add(sum2).wrapping_add(sum3)
                        }).collect()
                    } else {
                        group_indices.iter().map(|g| {
                            let mut sum: i64 = 0;
                            for &i in g {
                                sum = sum.wrapping_add(values[i]);
                            }
                            sum
                        }).collect()
                    };
                    Ok((Field::new(&output_name, ArrowDataType::Int64, false), Arc::new(Int64Array::from(sums))))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    let values = float_array.values();
                    let use_parallel = group_indices.len() > 100;
                    
                    let sums: Vec<f64> = if use_parallel {
                        group_indices.par_iter().map(|g| {
                            let mut sum: f64 = 0.0;
                            for &i in g {
                                sum += values[i];
                            }
                            sum
                        }).collect()
                    } else {
                        group_indices.iter().map(|g| {
                            let mut sum: f64 = 0.0;
                            for &i in g {
                                sum += values[i];
                            }
                            sum
                        }).collect()
                    };
                    Ok((Field::new(&output_name, ArrowDataType::Float64, false), Arc::new(Float64Array::from(sums))))
                } else {
                    Err(err_data( "SUM requires numeric column"))
                }
            }
            AggregateFunc::Avg => {
                let col_name = actual_column.as_ref().ok_or_else(|| err_input( "AVG requires column"))?;
                let array = batch.column_by_name(col_name).ok_or_else(|| err_not_found(format!("Column: {}", col_name)))?;
                
                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    // Fast path: direct slice access, compute sum and count together
                    let values = int_array.values();
                    let avgs: Vec<f64> = group_indices.iter().map(|g| {
                        if g.is_empty() { return 0.0; }
                        let mut sum: i64 = 0;
                        for &i in g {
                            sum = sum.wrapping_add(values[i]);
                        }
                        sum as f64 / g.len() as f64
                    }).collect();
                    Ok((Field::new(&output_name, ArrowDataType::Float64, false), Arc::new(Float64Array::from(avgs))))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    let values = float_array.values();
                    let avgs: Vec<f64> = group_indices.iter().map(|g| {
                        if g.is_empty() { return 0.0; }
                        let mut sum: f64 = 0.0;
                        for &i in g {
                            sum += values[i];
                        }
                        sum / g.len() as f64
                    }).collect();
                    Ok((Field::new(&output_name, ArrowDataType::Float64, false), Arc::new(Float64Array::from(avgs))))
                } else {
                    Err(err_data( "AVG requires numeric column"))
                }
            }
            AggregateFunc::Min => {
                let col_name = actual_column.as_ref().ok_or_else(|| err_input( "MIN requires column"))?;
                let array = batch.column_by_name(col_name).ok_or_else(|| err_not_found(format!("Column: {}", col_name)))?;
                
                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    let mins: Vec<Option<i64>> = group_indices.iter().map(|g| {
                        g.iter().filter_map(|&i| if int_array.is_null(i) { None } else { Some(int_array.value(i)) }).min()
                    }).collect();
                    Ok((Field::new(&output_name, ArrowDataType::Int64, true), Arc::new(Int64Array::from(mins))))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    let mins: Vec<Option<f64>> = group_indices.iter().map(|g| {
                        g.iter().filter_map(|&i| if float_array.is_null(i) { None } else { Some(float_array.value(i)) }).reduce(f64::min)
                    }).collect();
                    Ok((Field::new(&output_name, ArrowDataType::Float64, true), Arc::new(Float64Array::from(mins))))
                } else {
                    Err(err_data( "MIN requires numeric column"))
                }
            }
            AggregateFunc::Max => {
                let col_name = actual_column.as_ref().ok_or_else(|| err_input( "MAX requires column"))?;
                let array = batch.column_by_name(col_name).ok_or_else(|| err_not_found(format!("Column: {}", col_name)))?;
                
                if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                    let maxs: Vec<Option<i64>> = group_indices.iter().map(|g| {
                        g.iter().filter_map(|&i| if int_array.is_null(i) { None } else { Some(int_array.value(i)) }).max()
                    }).collect();
                    Ok((Field::new(&output_name, ArrowDataType::Int64, true), Arc::new(Int64Array::from(maxs))))
                } else if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                    let maxs: Vec<Option<f64>> = group_indices.iter().map(|g| {
                        g.iter().filter_map(|&i| if float_array.is_null(i) { None } else { Some(float_array.value(i)) }).reduce(f64::max)
                    }).collect();
                    Ok((Field::new(&output_name, ArrowDataType::Float64, true), Arc::new(Float64Array::from(maxs))))
                } else {
                    Err(err_data( "MAX requires numeric column"))
                }
            }
        }
    }

    /// FAST PATH: Compute simple aggregates directly from mmap bytes.
    /// Handles COUNT(*), COUNT(col), SUM(col), AVG(col), MIN(col), MAX(col)
    /// for numeric columns without WHERE/GROUP BY.
    /// Caches stats per column to avoid redundant mmap scans.
    /// Returns None if the query can't be handled by this fast path.
    pub(crate) fn try_mmap_aggregation(
        backend: &crate::storage::TableStorageBackend,
        stmt: &crate::query::SelectStatement,
    ) -> io::Result<Option<ApexResult>> {
        use crate::query::{AggregateFunc, SelectColumn};
        use std::collections::HashMap;
        let active_count = backend.active_row_count() as i64;

        // FAST PATH: COUNT(DISTINCT col) — global dict cache + null-aware count.
        // Warm calls: O(dict_size) ≈ O(10) — dict already cached, just count entries.
        // Cold calls: O(N) to build dict (same as before), but result is cached for next call.
        // Null-aware: uses null bitmaps to exclude NULL rows from the count.
        if stmt.columns.len() == 1 {
            if let SelectColumn::Aggregate { func: AggregateFunc::Count, column: Some(col_name), distinct: true, alias } = &stmt.columns[0] {
                let actual = col_name.trim_matches('"');
                let actual = if let Some(p) = actual.rfind('.') { &actual[p+1..] } else { actual };
                if let Some(dict) = crate::storage::backend::get_global_dict_cache(
                    backend.path(), actual, &backend.storage,
                )? {
                    let (dict_strings, group_ids) = (dict.0.as_slice(), dict.1.as_slice());
                    let count = backend.count_distinct_with_dict(actual, dict_strings, group_ids)?;
                    let out = alias.clone().unwrap_or_else(|| format!("COUNT(DISTINCT {})", actual));
                    let schema = Arc::new(Schema::new(vec![Field::new(&out, ArrowDataType::Int64, false)]));
                    let arr: ArrayRef = Arc::new(Int64Array::from(vec![count]));
                    let batch = RecordBatch::try_new(schema, vec![arr]).map_err(|e| err_data(e.to_string()))?;
                    return Ok(Some(ApexResult::Data(batch)));
                }
                return Ok(None); // non-string col — fall through to slow path
            }
        }

        // Collect unique column names for a single-pass multi-column aggregation
        let mut unique_cols: Vec<String> = Vec::new();
        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, distinct, .. } = col {
                if *distinct { return Ok(None); }
                if let Some(col_name) = column {
                    let is_count_star = matches!(func, AggregateFunc::Count)
                        && (col_name.as_str() == "*"
                            || col_name.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false));
                    if !is_count_star && !unique_cols.contains(col_name) {
                        unique_cols.push(col_name.clone());
                    }
                } else if !matches!(func, AggregateFunc::Count) {
                    return Ok(None);
                }
            } else {
                return Ok(None);
            }
        }

        // Single-pass multi-column aggregation (replaces N separate column scans)
        // execute_simple_agg uses the RCIX streaming path: one mmap pass for all columns
        let col_refs: Vec<&str> = unique_cols.iter().map(|s| s.as_str()).collect();
        // (count: i64, sum: f64, min: f64, max: f64, is_int: bool)
        let stats_vec = match backend.execute_simple_agg(&col_refs)? {
            Some(v) => v,
            None => return Ok(None),
        };
        // Map column name → (count, sum, min, max, is_int)
        let stats_cache: HashMap<&str, (i64, f64, f64, f64, bool)> = unique_cols.iter()
            .enumerate()
            .filter_map(|(i, name)| stats_vec.get(i).map(|&s| (name.as_str(), s)))
            .collect();

        let mut fields: Vec<Field> = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        for col in &stmt.columns {
            if let SelectColumn::Aggregate { func, column, alias, .. } = col {
                let fn_name = match func {
                    AggregateFunc::Count => "COUNT", AggregateFunc::Sum => "SUM",
                    AggregateFunc::Avg => "AVG", AggregateFunc::Min => "MIN", AggregateFunc::Max => "MAX",
                };
                let output_name = alias.clone().unwrap_or_else(|| {
                    if let Some(c) = column { format!("{}({})", fn_name, c) } else { format!("{}(*)", fn_name) }
                });

                match func {
                    AggregateFunc::Count => {
                        let is_star = column.as_deref() == Some("*") || column.is_none()
                            || column.as_ref().map(|c| c.chars().next().map(|ch| ch.is_ascii_digit()).unwrap_or(false)).unwrap_or(false);
                        if is_star {
                            fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                            arrays.push(Arc::new(Int64Array::from(vec![active_count])));
                        } else {
                            let col_name = column.as_ref().unwrap().as_str();
                            let count = stats_cache.get(col_name).map(|s| s.0).unwrap_or(0);
                            fields.push(Field::new(&output_name, ArrowDataType::Int64, false));
                            arrays.push(Arc::new(Int64Array::from(vec![count])));
                        }
                    }
                    AggregateFunc::Sum => {
                        let col_name = column.as_ref().unwrap().as_str();
                        let s = match stats_cache.get(col_name) { Some(s) => s, None => return Ok(None) };
                        if s.4 { // is_int
                            fields.push(Field::new(&output_name, ArrowDataType::Int64, true));
                            arrays.push(Arc::new(Int64Array::from(vec![s.1 as i64])));
                        } else {
                            fields.push(Field::new(&output_name, ArrowDataType::Float64, true));
                            arrays.push(Arc::new(Float64Array::from(vec![s.1])));
                        }
                    }
                    AggregateFunc::Avg => {
                        let col_name = column.as_ref().unwrap().as_str();
                        let s = match stats_cache.get(col_name) { Some(s) => s, None => return Ok(None) };
                        let avg = if s.0 > 0 { s.1 / s.0 as f64 } else { 0.0 };
                        fields.push(Field::new(&output_name, ArrowDataType::Float64, true));
                        arrays.push(Arc::new(Float64Array::from(vec![avg])));
                    }
                    AggregateFunc::Min => {
                        let col_name = column.as_ref().unwrap().as_str();
                        let s = match stats_cache.get(col_name) { Some(s) => s, None => return Ok(None) };
                        fields.push(Field::new(&output_name, ArrowDataType::Float64, true));
                        if s.0 == 0 { arrays.push(Arc::new(Float64Array::from(vec![None as Option<f64>]))); }
                        else {
                            if s.4 { // is_int
                                fields.pop();
                                fields.push(Field::new(&output_name, ArrowDataType::Int64, true));
                                arrays.push(Arc::new(Int64Array::from(vec![s.2 as i64])));
                            } else {
                                arrays.push(Arc::new(Float64Array::from(vec![s.2])));
                            }
                        }
                    }
                    AggregateFunc::Max => {
                        let col_name = column.as_ref().unwrap().as_str();
                        let s = match stats_cache.get(col_name) { Some(s) => s, None => return Ok(None) };
                        fields.push(Field::new(&output_name, ArrowDataType::Float64, true));
                        if s.0 == 0 { arrays.push(Arc::new(Float64Array::from(vec![None as Option<f64>]))); }
                        else {
                            if s.4 { // is_int
                                fields.pop();
                                fields.push(Field::new(&output_name, ArrowDataType::Int64, true));
                                arrays.push(Arc::new(Int64Array::from(vec![s.3 as i64])));
                            } else {
                                arrays.push(Arc::new(Float64Array::from(vec![s.3])));
                            }
                        }
                    }
                }
            }
        }

        if fields.is_empty() { return Ok(None); }
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays)
            .map_err(|e| err_data(e.to_string()))?;
        Ok(Some(ApexResult::Data(batch)))
    }

}
