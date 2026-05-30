// Filter, predicate evaluation, expression evaluation, SQL functions

use crate::query::vectorized_join::{
    count_matching_int64, count_matching_float64,
    filter_int64_batch, filter_float64_batch,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum JsonMutationMode {
    Set,
    Insert,
    Replace,
    Remove,
}

#[derive(Clone)]
enum JsonPathSegment {
    Key(String),
    Index(usize),
}

impl ApexExecutor {
    /// Apply WHERE clause filter using Arrow compute
    /// When storage_path is provided, enables Zone Map optimization and subquery support
    fn apply_filter_impl(batch: &RecordBatch, expr: &SqlExpr, storage_path: Option<&Path>) -> io::Result<RecordBatch> {
        // Zone Map optimization (always try, regardless of storage_path)
        if let Some(result) = Self::try_zone_map_filter(batch, expr) {
            if result == ZoneMapResult::NoMatch {
                return Ok(RecordBatch::new_empty(batch.schema()));
            }
        }

        let mask = if let Some(path) = storage_path {
            Self::evaluate_predicate_with_storage(batch, expr, path)?
        } else {
            Self::evaluate_predicate(batch, expr)?
        };

        // Early exit optimization: check if mask selects all or none
        // This avoids unnecessary filter operation overhead
        let num_rows = batch.num_rows();
        if num_rows > 0 {
            use arrow::array::BooleanArray;

            // OPTIMIZATION: Check if mask is all true (select all) - skip filtering
            if mask.null_count() == 0 {
                // All values are non-null, check if all are true
                let mut all_true = true;
                let check_len = num_rows.min(64); // Sample first 64 rows
                for i in 0..check_len {
                    if !mask.value(i) {
                        all_true = false;
                        break;
                    }
                }
                if all_true {
                    // Verify all rows are true
                    let mut verified = true;
                    for i in check_len..num_rows {
                        if !mask.value(i) {
                            verified = false;
                            break;
                        }
                    }
                    if verified {
                        return Ok(batch.clone());
                    }
                }
            }
        }

        compute::filter_record_batch(batch, &mask)
            .map_err(|e| err_data( e.to_string()))
    }

    #[inline]
    fn apply_filter(batch: &RecordBatch, expr: &SqlExpr) -> io::Result<RecordBatch> {
        Self::apply_filter_impl(batch, expr, None)
    }

    #[inline]
    fn apply_filter_with_storage(batch: &RecordBatch, expr: &SqlExpr, storage_path: &Path) -> io::Result<RecordBatch> {
        Self::apply_filter_impl(batch, expr, Some(storage_path))
    }
    
    
    /// Try to use Zone Maps to skip filtering
    /// Returns Some(NoMatch) if filter definitely won't match, None otherwise
    fn try_zone_map_filter(batch: &RecordBatch, expr: &SqlExpr) -> Option<ZoneMapResult> {
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                // Handle simple column vs literal comparisons
                if let (SqlExpr::Column(col_name), SqlExpr::Literal(lit)) = (left.as_ref(), right.as_ref()) {
                    return Self::check_zone_map_comparison(batch, col_name, op, lit);
                }
                if let (SqlExpr::Literal(lit), SqlExpr::Column(col_name)) = (left.as_ref(), right.as_ref()) {
                    // Flip the operator for literal vs column
                    let flipped_op = match op {
                        BinaryOperator::Lt => BinaryOperator::Gt,
                        BinaryOperator::Le => BinaryOperator::Ge,
                        BinaryOperator::Gt => BinaryOperator::Lt,
                        BinaryOperator::Ge => BinaryOperator::Le,
                        _ => op.clone(),
                    };
                    return Self::check_zone_map_comparison(batch, col_name, &flipped_op, lit);
                }
                
                // Handle AND: if either side is NoMatch, result is NoMatch
                if *op == BinaryOperator::And {
                    let left_result = Self::try_zone_map_filter(batch, left);
                    if left_result == Some(ZoneMapResult::NoMatch) {
                        return Some(ZoneMapResult::NoMatch);
                    }
                    let right_result = Self::try_zone_map_filter(batch, right);
                    if right_result == Some(ZoneMapResult::NoMatch) {
                        return Some(ZoneMapResult::NoMatch);
                    }
                }
                None
            }
            SqlExpr::Between { column, low, high, negated } => {
                // Check if BETWEEN range overlaps with column's value range
                if *negated {
                    return None; // NOT BETWEEN is harder to optimize
                }
                if let (SqlExpr::Literal(low_lit), SqlExpr::Literal(high_lit)) = (low.as_ref(), high.as_ref()) {
                    Self::check_zone_map_between(batch, column, low_lit, high_lit)
                } else {
                    None
                }
            }
            SqlExpr::Paren(inner) => Self::try_zone_map_filter(batch, inner),
            _ => None,
        }
    }
    
    /// Create ZoneMap from a column (helper to avoid duplication)
    #[inline]
    fn create_zone_map(col: &ArrayRef) -> Option<ZoneMap> {
        if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
            Some(ZoneMap::from_int64_array(arr))
        } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
            Some(ZoneMap::from_float64_array(arr))
        } else {
            None
        }
    }

    /// Check Zone Map for a simple comparison
    fn check_zone_map_comparison(batch: &RecordBatch, col_name: &str, op: &BinaryOperator, lit: &Value) -> Option<ZoneMapResult> {
        let col = batch.column_by_name(col_name.trim_matches('"'))?;
        let zone_map = Self::create_zone_map(col)?;
        Some(if zone_map.can_match(op, lit) { ZoneMapResult::MayMatch } else { ZoneMapResult::NoMatch })
    }
    
    /// Check Zone Map for BETWEEN
    fn check_zone_map_between(batch: &RecordBatch, col_name: &str, low: &Value, high: &Value) -> Option<ZoneMapResult> {
        let col = batch.column_by_name(col_name.trim_matches('"'))?;
        let zone_map = Self::create_zone_map(col)?;
        let can_match = zone_map.can_match(&BinaryOperator::Ge, low) && zone_map.can_match(&BinaryOperator::Le, high);
        Some(if can_match { ZoneMapResult::MayMatch } else { ZoneMapResult::NoMatch })
    }

    /// Evaluate a predicate expression to a boolean mask
    fn evaluate_predicate(batch: &RecordBatch, expr: &SqlExpr) -> io::Result<BooleanArray> {
        use crate::query::sql_parser::{BinaryOperator, UnaryOperator};

        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                // Handle logical operators (AND, OR)
                match op {
                    BinaryOperator::And => {
                        // Short-circuit evaluation: if left is all false, don't evaluate right
                        let left_mask = Self::evaluate_predicate(batch, left)?;

                        // Check if left is all false (early exit)
                        if left_mask.null_count() == 0 {
                            let mut all_false = true;
                            for i in 0..left_mask.len().min(64) {
                                if left_mask.value(i) { all_false = false; break; }
                            }
                            if all_false {
                                for i in 64..left_mask.len() {
                                    if left_mask.value(i) { all_false = false; break; }
                                }
                            }
                            if all_false {
                                // Left is all false, AND result is all false
                                return Ok(BooleanArray::from(vec![false; left_mask.len()]));
                            }
                        }

                        let right_mask = Self::evaluate_predicate(batch, right)?;
                        compute::and(&left_mask, &right_mask)
                            .map_err(|e| err_data( e.to_string()))
                    }
                    BinaryOperator::Or => {
                        // Short-circuit evaluation: if left is all true, don't evaluate right
                        let left_mask = Self::evaluate_predicate(batch, left)?;

                        // Check if left is all true (early exit)
                        if left_mask.null_count() == 0 {
                            let mut all_true = true;
                            for i in 0..left_mask.len().min(64) {
                                if !left_mask.value(i) { all_true = false; break; }
                            }
                            if all_true {
                                for i in 64..left_mask.len() {
                                    if !left_mask.value(i) { all_true = false; break; }
                                }
                            }
                            if all_true {
                                // Left is all true, OR result is all true
                                return Ok(BooleanArray::from(vec![true; left_mask.len()]));
                            }
                        }

                        let right_mask = Self::evaluate_predicate(batch, right)?;
                        compute::or(&left_mask, &right_mask)
                            .map_err(|e| err_data( e.to_string()))
                    }
                    // Comparison operators
                    _ => Self::evaluate_comparison(batch, left, op, right)
                }
            }
            SqlExpr::UnaryOp { op, expr } => {
                match op {
                    UnaryOperator::Not => {
                        let inner_mask = Self::evaluate_predicate(batch, expr)?;
                        compute::not(&inner_mask)
                            .map_err(|e| err_data( e.to_string()))
                    }
                    _ => Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "Unsupported unary operator in predicate",
                    ))
                }
            }
            SqlExpr::IsNull { column, negated } => {
                let col_name = column.trim_matches('"');
                let array = Self::get_column_by_name(batch, col_name)
                    .ok_or_else(|| err_not_found(format!("Column: {}", col_name)))?;
                let null_mask = compute::is_null(array)
                    .map_err(|e| err_data( e.to_string()))?;
                if *negated {
                    compute::not(&null_mask)
                        .map_err(|e| err_data( e.to_string()))
                } else {
                    Ok(null_mask)
                }
            }
            SqlExpr::Between { column, low, high, negated } => {
                let col_name = column.trim_matches('"');
                let val = Self::get_column_by_name(batch, col_name)
                    .ok_or_else(|| err_not_found(format!("Column: {}", col_name)))?;
                let low_val = Self::evaluate_expr_to_array(batch, low)?;
                let high_val = Self::evaluate_expr_to_array(batch, high)?;

                let (val_for_cmp, low_for_cmp) = Self::coerce_numeric_for_comparison(val.clone(), low_val)?;
                let (val_for_cmp2, high_for_cmp) = Self::coerce_numeric_for_comparison(val.clone(), high_val)?;
                
                let ge_low = cmp::gt_eq(&val_for_cmp, &low_for_cmp)
                    .map_err(|e| err_data( e.to_string()))?;
                let le_high = cmp::lt_eq(&val_for_cmp2, &high_for_cmp)
                    .map_err(|e| err_data( e.to_string()))?;
                
                let result = compute::and(&ge_low, &le_high)
                    .map_err(|e| err_data( e.to_string()))?;
                
                if *negated {
                    compute::not(&result)
                        .map_err(|e| err_data( e.to_string()))
                } else {
                    Ok(result)
                }
            }
            SqlExpr::In { column, values, negated } => {
                Self::evaluate_in_values(batch, column, values, *negated)
            }
            SqlExpr::Like { column, pattern, negated } => {
                Self::evaluate_like(batch, column, pattern, *negated)
            }
            SqlExpr::Regexp { column, pattern, negated } => {
                Self::evaluate_regexp(batch, column, pattern, *negated)
            }
            SqlExpr::Paren(inner) => {
                Self::evaluate_predicate(batch, inner)
            }
            // Subqueries require storage path - return error for now, will be handled by evaluate_predicate_with_storage
            SqlExpr::InSubquery { .. } | SqlExpr::ExistsSubquery { .. } | SqlExpr::ScalarSubquery { .. } => {
                Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "Subqueries require storage path - use evaluate_predicate_with_storage",
                ))
            }
            _ => {
                Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("Unsupported expression type in predicate: {:?}", expr),
                ))
            }
        }
    }

    /// Evaluate predicate with storage path (for subqueries)
    fn evaluate_predicate_with_storage(batch: &RecordBatch, expr: &SqlExpr, storage_path: &Path) -> io::Result<BooleanArray> {
        use crate::query::sql_parser::{BinaryOperator, UnaryOperator};
        
        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                match op {
                    BinaryOperator::And => {
                        let left_mask = Self::evaluate_predicate_with_storage(batch, left, storage_path)?;
                        let right_mask = Self::evaluate_predicate_with_storage(batch, right, storage_path)?;
                        compute::and(&left_mask, &right_mask)
                            .map_err(|e| err_data( e.to_string()))
                    }
                    BinaryOperator::Or => {
                        let left_mask = Self::evaluate_predicate_with_storage(batch, left, storage_path)?;
                        let right_mask = Self::evaluate_predicate_with_storage(batch, right, storage_path)?;
                        compute::or(&left_mask, &right_mask)
                            .map_err(|e| err_data( e.to_string()))
                    }
                    _ => Self::evaluate_comparison_with_storage(batch, left, op, right, storage_path)
                }
            }
            SqlExpr::UnaryOp { op, expr } => {
                match op {
                    UnaryOperator::Not => {
                        let inner_mask = Self::evaluate_predicate_with_storage(batch, expr, storage_path)?;
                        compute::not(&inner_mask)
                            .map_err(|e| err_data( e.to_string()))
                    }
                    _ => Err(io::Error::new(io::ErrorKind::Unsupported, "Unsupported unary operator"))
                }
            }
            SqlExpr::InSubquery { column, stmt, negated } => {
                Self::evaluate_in_subquery(batch, column, stmt, *negated, storage_path)
            }
            SqlExpr::ExistsSubquery { stmt } => {
                Self::evaluate_exists_subquery(batch, stmt, storage_path)
            }
            SqlExpr::ScalarSubquery { .. } => {
                // Scalar subquery alone in predicate - treat as boolean (non-null = true)
                let val = Self::evaluate_expr_to_array_with_storage(batch, expr, storage_path)?;
                let mut results = Vec::with_capacity(batch.num_rows());
                for i in 0..batch.num_rows() {
                    results.push(!val.is_null(i));
                }
                Ok(BooleanArray::from(results))
            }
            SqlExpr::Paren(inner) => {
                Self::evaluate_predicate_with_storage(batch, inner, storage_path)
            }
            // Delegate non-subquery expressions to regular evaluate_predicate
            _ => Self::evaluate_predicate(batch, expr)
        }
    }

    /// Execute IN subquery (supports correlated subqueries)
    fn evaluate_in_subquery(
        batch: &RecordBatch, 
        column: &str, 
        stmt: &SelectStatement, 
        negated: bool,
        storage_path: &Path
    ) -> io::Result<BooleanArray> {
        // Resolve the subquery's table path from its FROM clause
        let subquery_path = Self::resolve_subquery_table_path(stmt, storage_path)?;
        
        // Check if this is a correlated subquery
        let outer_cols = Self::find_outer_column_refs(stmt, batch);
        
        let col_name = column.trim_matches('"');
        // Strip table alias (e.g., "u.user_id" -> "user_id")
        let lookup_name = if let Some(dot_pos) = col_name.find('.') {
            &col_name[dot_pos + 1..]
        } else {
            col_name
        };
        let main_col = batch.column_by_name(lookup_name)
            .ok_or_else(|| err_not_found(format!("Column: {}", col_name)))?;
        
        if outer_cols.is_empty() {
            // Non-correlated: execute once
            let sub_result = Self::execute_select(stmt.clone(), &subquery_path)?;
            let sub_batch = sub_result.to_record_batch()?;
            
            if sub_batch.num_rows() == 0 {
                return Ok(BooleanArray::from(vec![negated; batch.num_rows()]));
            }
            
            if sub_batch.num_columns() == 0 {
                return Err(err_data( "Subquery must return at least one column"));
            }
            let sub_col = sub_batch.column(0);
            
            // Build hash set of subquery values
            let mut value_set: HashSet<u64> = HashSet::with_capacity(sub_batch.num_rows());
            for i in 0..sub_batch.num_rows() {
                if !sub_col.is_null(i) {
                    value_set.insert(Self::hash_array_value(sub_col, i));
                }
            }
            
            let mut results = Vec::with_capacity(batch.num_rows());
            for i in 0..batch.num_rows() {
                let hash = Self::hash_array_value(main_col, i);
                let found = value_set.contains(&hash);
                results.push(if negated { !found } else { found });
            }
            
            Ok(BooleanArray::from(results))
        } else {
            // Decorrelation: try to convert correlated IN to hash semi-join
            if let Some(result) = Self::try_decorrelate_in(batch, main_col, stmt, &outer_cols, negated, &subquery_path)? {
                return Ok(result);
            }
            
            // Fallback: Correlated, evaluate for each row
            let mut results = Vec::with_capacity(batch.num_rows());
            
            for row_idx in 0..batch.num_rows() {
                let modified_stmt = Self::substitute_outer_refs(stmt, batch, row_idx, &outer_cols);
                
                let sub_result = Self::execute_select(modified_stmt, &subquery_path)?;
                let sub_batch = sub_result.to_record_batch()?;
                
                let found = if sub_batch.num_rows() == 0 || sub_batch.num_columns() == 0 {
                    false
                } else {
                    let sub_col = sub_batch.column(0);
                    let main_hash = Self::hash_array_value(main_col, row_idx);
                    (0..sub_batch.num_rows()).any(|i| !sub_col.is_null(i) && Self::hash_array_value(sub_col, i) == main_hash)
                };
                
                results.push(if negated { !found } else { found });
            }
            
            Ok(BooleanArray::from(results))
        }
    }
    
    /// Try to decorrelate a correlated IN subquery using hash semi-join.
    /// Pattern: col IN (SELECT x FROM t WHERE t.y = outer.y)
    fn try_decorrelate_in(
        batch: &RecordBatch,
        main_col: &ArrayRef,
        stmt: &SelectStatement,
        outer_refs: &[String],
        negated: bool,
        subquery_path: &Path,
    ) -> io::Result<Option<BooleanArray>> {
        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };
        
        let (outer_col, inner_col, remaining_pred) = match Self::extract_correlation_equality(where_clause, outer_refs) {
            Some(v) => v,
            None => return Ok(None),
        };
        
        // Execute decorrelated subquery: SELECT original_select_col, inner_col FROM ... WHERE remaining
        let mut decorrelated = stmt.clone();
        // Keep original SELECT columns and add the inner join column
        let orig_cols = decorrelated.columns.clone();
        let inner_col_clean = if let Some(dot_pos) = inner_col.rfind('.') {
            inner_col[dot_pos + 1..].to_string()
        } else {
            inner_col.clone()
        };
        // Check if inner_col is already in SELECT
        let has_inner = orig_cols.iter().any(|c| {
            if let SelectColumn::Column(name) = c { name == &inner_col_clean } else { false }
        });
        if !has_inner {
            decorrelated.columns.push(SelectColumn::Column(inner_col_clean.clone()));
        }
        decorrelated.where_clause = remaining_pred;
        
        let sub_result = Self::execute_select(decorrelated, subquery_path)?;
        let sub_batch = sub_result.to_record_batch()?;
        
        if sub_batch.num_rows() == 0 {
            return Ok(Some(BooleanArray::from(vec![negated; batch.num_rows()])));
        }
        
        // Build a map: inner_col_value -> set of subquery result values
        let inner_join_col = sub_batch.column_by_name(&inner_col_clean)
            .or_else(|| sub_batch.column(sub_batch.num_columns() - 1).into())
            .ok_or_else(|| err_not_found(format!("Inner join column: {}", inner_col_clean)))?;
        let sub_value_col = sub_batch.column(0);

        // Build hash map: outer_value -> set of IN values
        // OPTIMIZATION: Use AHashMap and AHashSet for faster hashing
        use ahash::AHashMap;
        use ahash::AHashSet;
        let mut join_map: AHashMap<String, AHashSet<u64>> = AHashMap::new();
        for i in 0..sub_batch.num_rows() {
            if !inner_join_col.is_null(i) {
                let join_key = Self::arrow_value_to_string(inner_join_col, i);
                let in_hash = Self::hash_array_value(sub_value_col, i);
                join_map.entry(join_key).or_default().insert(in_hash);
            }
        }
        
        // Resolve outer column
        let outer_col_clean = if let Some(dot_pos) = outer_col.rfind('.') {
            &outer_col[dot_pos + 1..]
        } else {
            &outer_col
        };
        let outer_join_array = batch.column_by_name(outer_col_clean)
            .ok_or_else(|| err_not_found(format!("Outer column: {}", outer_col_clean)))?;
        
        let mut results = Vec::with_capacity(batch.num_rows());
        for i in 0..batch.num_rows() {
            let outer_key = Self::arrow_value_to_string(outer_join_array, i);
            let main_hash = Self::hash_array_value(main_col, i);
            let found = join_map.get(&outer_key)
                .map(|set| set.contains(&main_hash))
                .unwrap_or(false);
            results.push(if negated { !found } else { found });
        }
        
        Ok(Some(BooleanArray::from(results)))
    }

    /// Execute EXISTS subquery (supports correlated subqueries)
    fn evaluate_exists_subquery(
        batch: &RecordBatch, 
        stmt: &SelectStatement, 
        storage_path: &Path
    ) -> io::Result<BooleanArray> {
        // Resolve the subquery's table path from its FROM clause
        let subquery_path = Self::resolve_subquery_table_path(stmt, storage_path)?;
        
        // Check if this is a correlated subquery by looking for outer column references
        let outer_cols = Self::find_outer_column_refs(stmt, batch);
        
        if outer_cols.is_empty() {
            // Non-correlated: execute once
            let sub_result = Self::execute_select(stmt.clone(), &subquery_path)?;
            let sub_batch = sub_result.to_record_batch()?;
            let exists = sub_batch.num_rows() > 0;
            Ok(BooleanArray::from(vec![exists; batch.num_rows()]))
        } else {
            // Decorrelation: try to convert correlated EXISTS to hash semi-join.
            // Pattern: WHERE inner_col = outer_col (simple equality correlation)
            // Instead of executing subquery N times, execute once and hash-join.
            // Use Ok(...) swallowing: if decorrelation fails/has complex predicates, fall through.
            if let Ok(Some(result)) = Self::try_decorrelate_exists(batch, stmt, &outer_cols, &subquery_path) {
                return Ok(result);
            }
            
            // Fallback: Correlated, evaluate for each row
            let mut results = Vec::with_capacity(batch.num_rows());
            
            for row_idx in 0..batch.num_rows() {
                let modified_stmt = Self::substitute_outer_refs(stmt, batch, row_idx, &outer_cols);
                
                let sub_result = Self::execute_select(modified_stmt, &subquery_path)?;
                let sub_batch = sub_result.to_record_batch()?;
                results.push(sub_batch.num_rows() > 0);
            }
            
            Ok(BooleanArray::from(results))
        }
    }
    
    /// Try to decorrelate an EXISTS subquery by converting to hash semi-join.
    /// Detects pattern: WHERE inner_col = outer_col (AND other_predicates)
    /// Executes subquery once with non-correlated predicates, builds hash set of
    /// join column values, then probes for each outer row — O(N+M) vs O(N*M).
    fn try_decorrelate_exists(
        batch: &RecordBatch,
        stmt: &SelectStatement,
        outer_refs: &[String],
        subquery_path: &Path,
    ) -> io::Result<Option<BooleanArray>> {
        use crate::query::sql_parser::BinaryOperator;
        
        // Extract correlation predicate: find "outer.col = inner.col" pattern
        let where_clause = match &stmt.where_clause {
            Some(w) => w,
            None => return Ok(None),
        };
        
        // Try to split WHERE into correlation predicate and remaining predicates
        let (outer_col, inner_col, remaining_pred) = match Self::extract_correlation_equality(where_clause, outer_refs) {
            Some(v) => v,
            None => return Ok(None),
        };
        
        // If remaining_pred still contains outer refs, can't decorrelate safely — fall back.
        if let Some(ref remaining) = remaining_pred {
            let mut remaining_refs = Vec::new();
            Self::collect_outer_refs_from_expr(remaining, &outer_refs.iter().map(|s| {
                if let Some(d) = s.rfind('.') { s[d+1..].to_string() } else { s.clone() }
            }).collect::<Vec<_>>(), "", &mut remaining_refs);
            // Also check with full qualified names
            let mut remaining_refs2 = Vec::new();
            Self::collect_outer_refs_from_expr(remaining, &outer_refs.iter().map(|s| s.clone()).collect::<Vec<_>>(), "", &mut remaining_refs2);
            if !remaining_refs.is_empty() || !remaining_refs2.is_empty() {
                return Ok(None);
            }
        }

        // Build decorrelated subquery: SELECT inner_col FROM ... WHERE remaining_pred
        let mut decorrelated = stmt.clone();
        decorrelated.columns = vec![SelectColumn::Column(inner_col.clone())];
        decorrelated.where_clause = remaining_pred;
        decorrelated.group_by_exprs = decorrelated.group_by_exprs.clone();
        
        // Execute decorrelated subquery once
        let sub_result = Self::execute_select(decorrelated, subquery_path)?;
        let sub_batch = sub_result.to_record_batch()?;
        
        if sub_batch.num_columns() == 0 {
            return Ok(Some(BooleanArray::from(vec![false; batch.num_rows()])));
        }
        
        // Build hash set from inner column values
        // OPTIMIZATION: Use AHashSet for faster string hashing
        let inner_array = sub_batch.column(0);
        use ahash::AHashSet;
        let mut hash_set: AHashSet<String> = AHashSet::new();
        for i in 0..inner_array.len() {
            if !inner_array.is_null(i) {
                let val = Self::arrow_value_to_string(inner_array, i);
                hash_set.insert(val);
            }
        }
        
        // Resolve outer column name (strip table prefix)
        let outer_col_clean = if let Some(dot_pos) = outer_col.rfind('.') {
            &outer_col[dot_pos + 1..]
        } else {
            &outer_col
        };
        
        // Probe outer batch against hash set
        let outer_array = batch.column_by_name(outer_col_clean)
            .ok_or_else(|| err_not_found(format!("Outer column: {}", outer_col_clean)))?;
        
        let mut results = Vec::with_capacity(batch.num_rows());
        for i in 0..batch.num_rows() {
            if outer_array.is_null(i) {
                results.push(false);
            } else {
                let val = Self::arrow_value_to_string(outer_array, i);
                results.push(hash_set.contains(&val));
            }
        }
        
        Ok(Some(BooleanArray::from(results)))
    }
    
    /// Extract correlation equality from a WHERE clause.
    /// Returns (outer_col, inner_col, remaining_predicate) or None.
    /// Handles: outer.col = inner.col AND other_pred
    fn extract_correlation_equality(
        expr: &SqlExpr,
        outer_refs: &[String],
    ) -> Option<(String, String, Option<SqlExpr>)> {
        use crate::query::sql_parser::BinaryOperator;
        
        match expr {
            // Simple equality: outer.col = inner.col
            SqlExpr::BinaryOp { left, op: BinaryOperator::Eq, right } => {
                if let (SqlExpr::Column(a), SqlExpr::Column(b)) = (left.as_ref(), right.as_ref()) {
                    let a_clean = a.trim_matches('"');
                    let b_clean = b.trim_matches('"');
                    if outer_refs.iter().any(|r| r == a_clean) {
                        return Some((a_clean.to_string(), b_clean.to_string(), None));
                    }
                    if outer_refs.iter().any(|r| r == b_clean) {
                        return Some((b_clean.to_string(), a_clean.to_string(), None));
                    }
                }
                None
            }
            // AND chain: try to find correlation equality in one branch
            SqlExpr::BinaryOp { left, op: BinaryOperator::And, right } => {
                // Try left side first
                if let Some((outer_col, inner_col, _)) = Self::extract_correlation_equality(left, outer_refs) {
                    return Some((outer_col, inner_col, Some(*right.clone())));
                }
                // Try right side
                if let Some((outer_col, inner_col, _)) = Self::extract_correlation_equality(right, outer_refs) {
                    return Some((outer_col, inner_col, Some(*left.clone())));
                }
                None
            }
            _ => None,
        }
    }
    
    /// Convert an Arrow array value at index to a string for hash comparison
    fn arrow_value_to_string(array: &ArrayRef, idx: usize) -> String {
        use arrow::array::*;
        if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
            return a.value(idx).to_string();
        }
        if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
            return a.value(idx).to_string();
        }
        if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
            return a.value(idx).to_string();
        }
        if let Some(a) = array.as_any().downcast_ref::<BooleanArray>() {
            return a.value(idx).to_string();
        }
        if let Some(a) = array.as_any().downcast_ref::<UInt64Array>() {
            return a.value(idx).to_string();
        }
        format!("{:?}", array)
    }
    
    /// Resolve subquery's table path from its FROM clause
    fn resolve_subquery_table_path(stmt: &SelectStatement, main_storage_path: &Path) -> io::Result<std::path::PathBuf> {
        // Get table name from subquery's FROM clause
        if let Some(FromItem::Table { table, .. }) = &stmt.from {
            let base_dir = main_storage_path.parent()
                .ok_or_else(|| err_input( "Cannot determine base directory"))?;
            Ok(Self::resolve_table_path(table, base_dir, main_storage_path))
        } else {
            // No FROM or derived table - use main storage path
            Ok(main_storage_path.to_path_buf())
        }
    }
    
    /// Find column references in subquery that refer to outer query columns
    fn find_outer_column_refs(stmt: &SelectStatement, outer_batch: &RecordBatch) -> Vec<String> {
        let mut outer_refs = Vec::with_capacity(4); // Most subqueries have few outer refs
        let outer_cols: Vec<String> = outer_batch.schema().fields().iter()
            .map(|f| f.name().clone())
            .collect();
        
        // Get subquery's table alias to exclude from outer refs
        let subquery_alias = match &stmt.from {
            Some(FromItem::Table { alias, table, .. }) => {
                alias.clone().unwrap_or_else(|| table.clone())
            }
            Some(FromItem::DirectFile { alias, file, .. }) => {
                alias.clone().unwrap_or_else(|| file.clone())
            }
            _ => String::new(),
        };
        
        // Check WHERE clause for outer column references
        if let Some(where_clause) = &stmt.where_clause {
            Self::collect_outer_refs_from_expr(where_clause, &outer_cols, &subquery_alias, &mut outer_refs);
        }
        
        outer_refs
    }
    
    /// Recursively collect outer column references from expression
    fn collect_outer_refs_from_expr(expr: &SqlExpr, outer_cols: &[String], subquery_alias: &str, refs: &mut Vec<String>) {
        match expr {
            SqlExpr::Column(name) => {
                let clean_name = name.trim_matches('"');
                // Check if column has table qualifier like "u.id" or "outer.col"
                if let Some(dot_pos) = clean_name.find('.') {
                    let table_part = &clean_name[..dot_pos];
                    let col_part = &clean_name[dot_pos + 1..];
                    
                    // Skip if the table prefix matches the subquery's table alias
                    // (e.g., "o.user_id" when subquery is "FROM orders o")
                    if !subquery_alias.is_empty() && table_part == subquery_alias {
                        return;
                    }
                    
                    // Check if the column part exists in outer batch columns
                    if outer_cols.iter().any(|c| c == col_part || c.trim_matches('"') == col_part) {
                        if !refs.contains(&clean_name.to_string()) {
                            refs.push(clean_name.to_string());
                        }
                    }
                }
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                Self::collect_outer_refs_from_expr(left, outer_cols, subquery_alias, refs);
                Self::collect_outer_refs_from_expr(right, outer_cols, subquery_alias, refs);
            }
            SqlExpr::UnaryOp { expr: inner, .. } | SqlExpr::Paren(inner) => {
                Self::collect_outer_refs_from_expr(inner, outer_cols, subquery_alias, refs);
            }
            _ => {}
        }
    }
    
    /// Substitute outer column references with actual values for a specific row
    fn substitute_outer_refs(stmt: &SelectStatement, outer_batch: &RecordBatch, row_idx: usize, outer_refs: &[String]) -> SelectStatement {
        let mut new_stmt = stmt.clone();
        
        if let Some(where_clause) = &mut new_stmt.where_clause {
            *where_clause = Self::substitute_expr(where_clause, outer_batch, row_idx, outer_refs);
        }
        
        new_stmt
    }
    
    /// Substitute outer column references in expression with literal values
    fn substitute_expr(expr: &SqlExpr, outer_batch: &RecordBatch, row_idx: usize, outer_refs: &[String]) -> SqlExpr {
        match expr {
            SqlExpr::Column(name) => {
                let clean_name = name.trim_matches('"');
                if outer_refs.iter().any(|r| r == clean_name) {
                    // This is an outer reference - substitute with value
                    if let Some(dot_pos) = clean_name.find('.') {
                        let col_part = &clean_name[dot_pos + 1..];
                        if let Some(col) = outer_batch.column_by_name(col_part) {
                            return Self::array_value_to_literal(col, row_idx);
                        }
                    }
                }
                expr.clone()
            }
            SqlExpr::BinaryOp { left, op, right } => {
                SqlExpr::BinaryOp {
                    left: Box::new(Self::substitute_expr(left, outer_batch, row_idx, outer_refs)),
                    op: op.clone(),
                    right: Box::new(Self::substitute_expr(right, outer_batch, row_idx, outer_refs)),
                }
            }
            SqlExpr::UnaryOp { op, expr: inner } => {
                SqlExpr::UnaryOp {
                    op: op.clone(),
                    expr: Box::new(Self::substitute_expr(inner, outer_batch, row_idx, outer_refs)),
                }
            }
            SqlExpr::Paren(inner) => {
                SqlExpr::Paren(Box::new(Self::substitute_expr(inner, outer_batch, row_idx, outer_refs)))
            }
            _ => expr.clone(),
        }
    }
    
    /// Convert array value at index to literal expression
    fn array_value_to_literal(array: &ArrayRef, idx: usize) -> SqlExpr {
        use crate::data::Value;
        
        if array.is_null(idx) {
            return SqlExpr::Literal(Value::Null);
        }
        
        if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
            SqlExpr::Literal(Value::Int64(arr.value(idx)))
        } else if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
            SqlExpr::Literal(Value::Float64(arr.value(idx)))
        } else if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
            SqlExpr::Literal(Value::String(arr.value(idx).to_string()))
        } else if let Some(arr) = array.as_any().downcast_ref::<BooleanArray>() {
            SqlExpr::Literal(Value::Bool(arr.value(idx)))
        } else {
            SqlExpr::Literal(Value::Null)
        }
    }

    /// Evaluate comparison operator
    fn evaluate_comparison(
        batch: &RecordBatch,
        left: &SqlExpr,
        op: &crate::query::sql_parser::BinaryOperator,
        right: &SqlExpr,
    ) -> io::Result<BooleanArray> {
        use crate::query::sql_parser::BinaryOperator;
        use arrow::array::Datum;
        
        // OPTIMIZATION: Fast path for column vs literal comparisons using scalar ops
        // This avoids broadcasting the literal to a full array
        if let Some(result) = Self::try_scalar_comparison(batch, left, op, right)? {
            return Ok(result);
        }
        
        let left_array = Self::evaluate_expr_to_array(batch, left)?;
        let right_array = Self::evaluate_expr_to_array(batch, right)?;

        let (left_array, right_array) = Self::coerce_numeric_for_comparison(left_array, right_array)?;

        let result = match op {
            BinaryOperator::Eq => cmp::eq(&left_array, &right_array),
            BinaryOperator::NotEq => cmp::neq(&left_array, &right_array),
            BinaryOperator::Lt => cmp::lt(&left_array, &right_array),
            BinaryOperator::Le => cmp::lt_eq(&left_array, &right_array),
            BinaryOperator::Gt => cmp::gt(&left_array, &right_array),
            BinaryOperator::Ge => cmp::gt_eq(&left_array, &right_array),
            _ => return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Unsupported comparison operator: {:?}", op),
            )),
        };

        result.map_err(|e| err_data( e.to_string()))
    }
    
    /// Try to use scalar comparison for column vs literal (faster than array vs array)
    #[inline]
    fn try_scalar_comparison(
        batch: &RecordBatch,
        left: &SqlExpr,
        op: &crate::query::sql_parser::BinaryOperator,
        right: &SqlExpr,
    ) -> io::Result<Option<BooleanArray>> {
        use crate::query::sql_parser::BinaryOperator;
        use arrow::array::Scalar;
        
        // Check for column = literal pattern
        let (col_expr, lit_val, reversed) = match (left, right) {
            (SqlExpr::Column(_), SqlExpr::Literal(v)) => (left, v, false),
            (SqlExpr::Literal(v), SqlExpr::Column(_)) => (right, v, true),
            _ => return Ok(None),
        };
        
        let col_array = Self::evaluate_expr_to_array(batch, col_expr)?;
        
        // FAST PATH: DictionaryArray<UInt32, Utf8> - compare using dictionary indices
        // OPTIMIZATION: Use direct buffer access instead of iterator for maximum speed
        use arrow::array::DictionaryArray;
        use arrow::datatypes::UInt32Type;
        use arrow::buffer::BooleanBuffer;
        if let Some(dict_arr) = col_array.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
            if let Value::String(s) = lit_val {
                let keys = dict_arr.keys();
                let values = dict_arr.values();
                if let Some(str_values) = values.as_any().downcast_ref::<StringArray>() {
                    // Find which dictionary index matches the filter value
                    let mut target_idx: Option<u32> = None;
                    for i in 0..str_values.len() {
                        if str_values.value(i) == s {
                            target_idx = Some(i as u32);
                            break;
                        }
                    }
                    
                    // Ultra-fast integer comparison on dictionary indices using direct buffer access
                    let num_rows = keys.len();
                    let result: BooleanArray = match (op, reversed, target_idx) {
                        (BinaryOperator::Eq, _, Some(idx)) => {
                            // Fast path: no nulls - use raw slice comparison
                            if keys.null_count() == 0 {
                                let key_values = keys.values();
                                let bools: Vec<bool> = key_values.iter().map(|&k| k == idx).collect();
                                BooleanArray::from(bools)
                            } else {
                                // Has nulls - handle them
                                let mut builder = arrow::array::BooleanBuilder::with_capacity(num_rows);
                                for i in 0..num_rows {
                                    if keys.is_null(i) {
                                        builder.append_value(false);
                                    } else {
                                        builder.append_value(keys.value(i) == idx);
                                    }
                                }
                                builder.finish()
                            }
                        }
                        (BinaryOperator::Eq, _, None) => {
                            // Value not in dictionary - no matches
                            BooleanArray::from(vec![false; num_rows])
                        }
                        (BinaryOperator::NotEq, _, Some(idx)) => {
                            if keys.null_count() == 0 {
                                let key_values = keys.values();
                                let bools: Vec<bool> = key_values.iter().map(|&k| k != idx).collect();
                                BooleanArray::from(bools)
                            } else {
                                let mut builder = arrow::array::BooleanBuilder::with_capacity(num_rows);
                                for i in 0..num_rows {
                                    if keys.is_null(i) {
                                        builder.append_value(false);
                                    } else {
                                        builder.append_value(keys.value(i) != idx);
                                    }
                                }
                                builder.finish()
                            }
                        }
                        (BinaryOperator::NotEq, _, None) => {
                            // Value not in dictionary - all non-null match
                            if keys.null_count() == 0 {
                                BooleanArray::from(vec![true; num_rows])
                            } else {
                                let mut builder = arrow::array::BooleanBuilder::with_capacity(num_rows);
                                for i in 0..num_rows {
                                    builder.append_value(!keys.is_null(i));
                                }
                                builder.finish()
                            }
                        }
                        _ => return Ok(None), // Other comparisons fall through
                    };
                    return Ok(Some(result));
                }
            }
        }
        
        // String scalar comparison (regular StringArray)
        if let Some(str_arr) = col_array.as_any().downcast_ref::<StringArray>() {
            if let Value::String(s) = lit_val {
                let scalar = Scalar::new(arrow::array::StringArray::from(vec![s.as_str()]));
                let result = match (op, reversed) {
                    (BinaryOperator::Eq, _) => cmp::eq(str_arr, &scalar),
                    (BinaryOperator::NotEq, _) => cmp::neq(str_arr, &scalar),
                    (BinaryOperator::Lt, false) | (BinaryOperator::Gt, true) => cmp::lt(str_arr, &scalar),
                    (BinaryOperator::Le, false) | (BinaryOperator::Ge, true) => cmp::lt_eq(str_arr, &scalar),
                    (BinaryOperator::Gt, false) | (BinaryOperator::Lt, true) => cmp::gt(str_arr, &scalar),
                    (BinaryOperator::Ge, false) | (BinaryOperator::Le, true) => cmp::gt_eq(str_arr, &scalar),
                    _ => return Ok(None),
                };
                return result.map(Some).map_err(|e| err_data( e.to_string()));
            }
        }
        
        // Int64 scalar comparison
        if let Some(int_arr) = col_array.as_any().downcast_ref::<Int64Array>() {
            let int_val = match lit_val {
                Value::Int64(i) => *i,
                Value::Float64(f) => *f as i64,
                _ => return Ok(None),
            };

            // VECTORIZED OPTIMIZATION: Use batch filtering for simple comparisons
            // This is faster than Arrow compute kernels for most cases
            let num_rows = int_arr.len();
            let op_str = match (op, reversed) {
                (BinaryOperator::Gt, false) => ">",
                (BinaryOperator::Lt, false) => "<",
                (BinaryOperator::Ge, false) => ">=",
                (BinaryOperator::Le, false) => "<=",
                (BinaryOperator::Gt, true) => "<",  // reversed: val > literal means literal < val
                (BinaryOperator::Lt, true) => ">",
                (BinaryOperator::Ge, true) => "<=",
                (BinaryOperator::Le, true) => ">=",
                _ => "",
            };

            // Use batch filtering for simple range comparisons (> < >= <=)
            if !op_str.is_empty() && int_arr.null_count() == 0 {
                let values = int_arr.values();
                let is_equal = op_str == ">=" || op_str == "<=";
                let indices = filter_int64_batch(values, int_val, op_str == ">" || op_str == ">=", is_equal);

                if indices.len() == num_rows {
                    // All match - return all true
                    return Ok(Some(BooleanArray::from(vec![true; num_rows])));
                } else if indices.is_empty() {
                    // None match - return all false
                    return Ok(Some(BooleanArray::from(vec![false; num_rows])));
                }
                // Partial match - build boolean array from indices
                if indices.len() < num_rows / 2 {
                    // Sparse result - use builder
                    let mut builder = arrow::array::BooleanBuilder::with_capacity(num_rows);
                    let mut idx_pos = 0;
                    for i in 0..num_rows {
                        if idx_pos < indices.len() && indices[idx_pos] == i {
                            builder.append_value(true);
                            idx_pos += 1;
                        } else {
                            builder.append_value(false);
                        }
                    }
                    return Ok(Some(builder.finish()));
                }
                // Otherwise fall through to Arrow compute (dense result)
            }

            // JIT optimization for large arrays (>100k rows)
            if num_rows > 100_000 {
                if let Some(result) = Self::try_jit_int_filter(int_arr, op, int_val, reversed) {
                    return Ok(Some(result));
                }
            }

            let scalar = Scalar::new(Int64Array::from(vec![int_val]));
            let result = match (op, reversed) {
                (BinaryOperator::Eq, _) => cmp::eq(int_arr, &scalar),
                (BinaryOperator::NotEq, _) => cmp::neq(int_arr, &scalar),
                (BinaryOperator::Lt, false) | (BinaryOperator::Gt, true) => cmp::lt(int_arr, &scalar),
                (BinaryOperator::Le, false) | (BinaryOperator::Ge, true) => cmp::lt_eq(int_arr, &scalar),
                (BinaryOperator::Gt, false) | (BinaryOperator::Lt, true) => cmp::gt(int_arr, &scalar),
                (BinaryOperator::Ge, false) | (BinaryOperator::Le, true) => cmp::gt_eq(int_arr, &scalar),
                _ => return Ok(None),
            };
            return result.map(Some).map_err(|e| err_data( e.to_string()));
        }
        
        // UInt64 scalar comparison (for _id column)
        if let Some(uint_arr) = col_array.as_any().downcast_ref::<UInt64Array>() {
            let uint_val = match lit_val {
                Value::UInt64(i) => *i,
                Value::Int64(i) => *i as u64,
                Value::UInt32(i) => *i as u64,
                Value::Int32(i) => *i as u64,
                _ => return Ok(None),
            };
            let scalar = Scalar::new(UInt64Array::from(vec![uint_val]));
            let result = match (op, reversed) {
                (BinaryOperator::Eq, _) => cmp::eq(uint_arr, &scalar),
                (BinaryOperator::NotEq, _) => cmp::neq(uint_arr, &scalar),
                (BinaryOperator::Lt, false) | (BinaryOperator::Gt, true) => cmp::lt(uint_arr, &scalar),
                (BinaryOperator::Le, false) | (BinaryOperator::Ge, true) => cmp::lt_eq(uint_arr, &scalar),
                (BinaryOperator::Gt, false) | (BinaryOperator::Lt, true) => cmp::gt(uint_arr, &scalar),
                (BinaryOperator::Ge, false) | (BinaryOperator::Le, true) => cmp::gt_eq(uint_arr, &scalar),
                _ => return Ok(None),
            };
            return result.map(Some).map_err(|e| err_data( e.to_string()));
        }
        
        // Float64 scalar comparison
        if let Some(float_arr) = col_array.as_any().downcast_ref::<Float64Array>() {
            let float_val = match lit_val {
                Value::Float64(f) => *f,
                Value::Int64(i) => *i as f64,
                _ => return Ok(None),
            };

            // VECTORIZED OPTIMIZATION: Use batch filtering for Float64
            let num_rows = float_arr.len();
            let op_str = match (op, reversed) {
                (BinaryOperator::Gt, false) => ">",
                (BinaryOperator::Lt, false) => "<",
                (BinaryOperator::Ge, false) => ">=",
                (BinaryOperator::Le, false) => "<=",
                (BinaryOperator::Gt, true) => "<",
                (BinaryOperator::Lt, true) => ">",
                (BinaryOperator::Ge, true) => "<=",
                (BinaryOperator::Le, true) => ">=",
                _ => "",
            };

            // Use batch filtering for simple range comparisons
            if !op_str.is_empty() && float_arr.null_count() == 0 {
                let values = float_arr.values();
                let is_equal = op_str == ">=" || op_str == "<=";
                let indices = filter_float64_batch(values, float_val, op_str == ">" || op_str == ">=", is_equal);

                if indices.len() == num_rows {
                    return Ok(Some(BooleanArray::from(vec![true; num_rows])));
                } else if indices.is_empty() {
                    return Ok(Some(BooleanArray::from(vec![false; num_rows])));
                }
                if indices.len() < num_rows / 2 {
                    let mut builder = arrow::array::BooleanBuilder::with_capacity(num_rows);
                    let mut idx_pos = 0;
                    for i in 0..num_rows {
                        if idx_pos < indices.len() && indices[idx_pos] == i {
                            builder.append_value(true);
                            idx_pos += 1;
                        } else {
                            builder.append_value(false);
                        }
                    }
                    return Ok(Some(builder.finish()));
                }
            }

            let scalar = Scalar::new(Float64Array::from(vec![float_val]));
            let result = match (op, reversed) {
                (BinaryOperator::Eq, _) => cmp::eq(float_arr, &scalar),
                (BinaryOperator::NotEq, _) => cmp::neq(float_arr, &scalar),
                (BinaryOperator::Lt, false) | (BinaryOperator::Gt, true) => cmp::lt(float_arr, &scalar),
                (BinaryOperator::Le, false) | (BinaryOperator::Ge, true) => cmp::lt_eq(float_arr, &scalar),
                (BinaryOperator::Gt, false) | (BinaryOperator::Lt, true) => cmp::gt(float_arr, &scalar),
                (BinaryOperator::Ge, false) | (BinaryOperator::Le, true) => cmp::gt_eq(float_arr, &scalar),
                _ => return Ok(None),
            };
            return result.map(Some).map_err(|e| err_data( e.to_string()));
        }
        
        Ok(None)
    }

    /// Evaluate comparison with storage path (for scalar subqueries)
    fn evaluate_comparison_with_storage(
        batch: &RecordBatch,
        left: &SqlExpr,
        op: &crate::query::sql_parser::BinaryOperator,
        right: &SqlExpr,
        storage_path: &Path,
    ) -> io::Result<BooleanArray> {
        use crate::query::sql_parser::BinaryOperator;

        if !Self::expr_contains_scalar_subquery(left) && !Self::expr_contains_scalar_subquery(right)
        {
            return Self::evaluate_comparison(batch, left, op, right);
        }

        let left_array = Self::evaluate_expr_to_array_with_storage(batch, left, storage_path)?;
        let right_array = Self::evaluate_expr_to_array_with_storage(batch, right, storage_path)?;

        let (left_array, right_array) = Self::coerce_numeric_for_comparison(left_array, right_array)?;

        let result = match op {
            BinaryOperator::Eq => cmp::eq(&left_array, &right_array),
            BinaryOperator::NotEq => cmp::neq(&left_array, &right_array),
            BinaryOperator::Lt => cmp::lt(&left_array, &right_array),
            BinaryOperator::Le => cmp::lt_eq(&left_array, &right_array),
            BinaryOperator::Gt => cmp::gt(&left_array, &right_array),
            BinaryOperator::Ge => cmp::gt_eq(&left_array, &right_array),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("Unsupported comparison operator: {:?}", op),
                ))
            }
        };

        result.map_err(|e| err_data( e.to_string()))
    }

    /// Try to use JIT compilation for integer filter (for large arrays)
    /// Returns None if JIT compilation fails
    fn try_jit_int_filter(
        int_arr: &Int64Array,
        op: &BinaryOperator,
        lit_val: i64,
        reversed: bool,
    ) -> Option<BooleanArray> {
        // Adjust operator if reversed
        let actual_op = if reversed {
            match op {
                BinaryOperator::Lt => BinaryOperator::Gt,
                BinaryOperator::Le => BinaryOperator::Ge,
                BinaryOperator::Gt => BinaryOperator::Lt,
                BinaryOperator::Ge => BinaryOperator::Le,
                _ => op.clone(),
            }
        } else {
            op.clone()
        };
        
        // Try to compile and execute JIT filter
        let mut jit = ExprJIT::new().ok()?;
        let filter_fn = jit.compile_int_filter(actual_op, lit_val).ok()?;
        
        let num_rows = int_arr.len();
        let mut result_bytes = vec![0u8; num_rows];
        
        // Get raw pointer to i64 data
        let data_ptr = int_arr.values().as_ptr();
        
        // Execute JIT-compiled filter
        unsafe {
            filter_fn(data_ptr, num_rows, result_bytes.as_mut_ptr());
        }
        
        // Convert result bytes to BooleanArray
        let bools: Vec<bool> = result_bytes.iter().map(|&b| b != 0).collect();
        Some(BooleanArray::from(bools))
    }

    /// Evaluate expression to Arrow array
    fn evaluate_expr_to_array(batch: &RecordBatch, expr: &SqlExpr) -> io::Result<ArrayRef> {
        match expr {
            SqlExpr::Column(name) => {
                let col_name = name.trim_matches('"');
                // Try full qualified name first (e.g., "b.name" for self-join disambiguation),
                // then fall back to bare name (e.g., "name" for "a.name" / "t.col").
                let bare_col = if let Some(dot_pos) = col_name.rfind('.') {
                    &col_name[dot_pos + 1..]
                } else {
                    col_name
                };
                batch
                    .column_by_name(col_name)
                    .cloned()
                    .or_else(|| batch.column_by_name(bare_col).cloned())
                    .ok_or_else(|| io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("Column '{}' not found", col_name),
                    ))
            }
            SqlExpr::Literal(val) => {
                Self::value_to_array(val, batch.num_rows())
            }
            SqlExpr::BinaryOp { left, op, right } => {
                Self::evaluate_arithmetic_op(batch, left, op, right)
            }
            SqlExpr::Case { when_then, else_expr } => {
                Self::evaluate_case_expr(batch, when_then, else_expr.as_deref())
            }
            SqlExpr::Function { name, args } => {
                Self::evaluate_function_expr(batch, name, args)
            }
            SqlExpr::Cast { expr: inner, data_type } => {
                Self::evaluate_cast_expr(batch, inner, data_type)
            }
            SqlExpr::Paren(inner) => {
                Self::evaluate_expr_to_array(batch, inner)
            }
            SqlExpr::ArrayIndex { array, index } => {
                Self::evaluate_array_index(batch, array, index)
            }
            SqlExpr::ArrayLiteral(values) => {
                // Encode as a single-value BinaryArray (raw f32 LE bytes), broadcast to all rows
                use arrow::array::BinaryArray;
                let bytes = crate::query::vector_ops::encode_f32_vec(
                    &values.iter().map(|&f| f as f32).collect::<Vec<_>>()
                );
                let n = batch.num_rows().max(1);
                let arr: BinaryArray = (0..n).map(|_| Some(bytes.as_slice())).collect();
                Ok(Arc::new(arr) as ArrayRef)
            }
            SqlExpr::UnaryOp { op, expr: inner } => {
                use crate::query::sql_parser::UnaryOperator;
                match op {
                    UnaryOperator::Minus => {
                        let arr = Self::evaluate_expr_to_array(batch, inner)?;
                        if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                            let negated: Vec<Option<i64>> = (0..int_arr.len())
                                .map(|i| if int_arr.is_null(i) { None } else { Some(-int_arr.value(i)) })
                                .collect();
                            Ok(Arc::new(Int64Array::from(negated)) as ArrayRef)
                        } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                            let negated: Vec<Option<f64>> = (0..float_arr.len())
                                .map(|i| if float_arr.is_null(i) { None } else { Some(-float_arr.value(i)) })
                                .collect();
                            Ok(Arc::new(Float64Array::from(negated)) as ArrayRef)
                        } else {
                            Err(io::Error::new(io::ErrorKind::Unsupported,
                                "Unary minus only supported for numeric types"))
                        }
                    }
                    UnaryOperator::Not => {
                        Err(io::Error::new(io::ErrorKind::Unsupported,
                            "NOT operator not supported as expression"))
                    }
                }
            }
            SqlExpr::Variable(name) => {
                use crate::query::executor::get_session_variable;
                let value = get_session_variable(name).unwrap_or(crate::data::Value::Null);
                Self::value_to_array(&value, batch.num_rows())
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("Unsupported expression type: {:?}", expr),
            )),
        }
    }

    /// Evaluate expression to Arrow array with storage path (for scalar subqueries)
    fn evaluate_expr_to_array_with_storage(batch: &RecordBatch, expr: &SqlExpr, storage_path: &Path) -> io::Result<ArrayRef> {
        match expr {
            SqlExpr::ScalarSubquery { stmt } => {
                Self::evaluate_scalar_subquery(batch, stmt, storage_path)
            }
            SqlExpr::BinaryOp { left, op, right } => {
                // Check if operands contain scalar subqueries
                let left_array = Self::evaluate_expr_to_array_with_storage(batch, left, storage_path)?;
                let right_array = Self::evaluate_expr_to_array_with_storage(batch, right, storage_path)?;
                Self::evaluate_arithmetic_op_arrays(&left_array, &right_array, op)
            }
            // Delegate non-subquery expressions
            _ => Self::evaluate_expr_to_array(batch, expr)
        }
    }

    /// Execute scalar subquery and broadcast result to array
    fn evaluate_scalar_subquery(batch: &RecordBatch, stmt: &SelectStatement, storage_path: &Path) -> io::Result<ArrayRef> {
        // Resolve the subquery's table path from its FROM clause
        let subquery_path = Self::resolve_subquery_table_path(stmt, storage_path)?;
        
        // Check if this is a correlated subquery
        let outer_cols = Self::find_outer_column_refs(stmt, batch);
        
        if outer_cols.is_empty() {
            // Non-correlated: execute once and broadcast to all rows
            let sub_result = Self::execute_select(stmt.clone(), &subquery_path)?;
            let sub_batch = sub_result.to_record_batch()?;
            
            if sub_batch.num_rows() == 0 || sub_batch.num_columns() == 0 {
                return Ok(Arc::new(Int64Array::from(vec![None::<i64>; batch.num_rows()])));
            }
            
            if sub_batch.num_rows() > 1 {
                return Err(err_data( "Scalar subquery returned more than one row"));
            }
            
            let sub_col = sub_batch.column(0);
            return Self::broadcast_scalar_array(sub_col, 0, batch.num_rows());
        }
        
        // Correlated: execute for each row
        let mut results: Vec<Option<i64>> = Vec::with_capacity(batch.num_rows());
        
        for row_idx in 0..batch.num_rows() {
            let modified_stmt = Self::substitute_outer_refs(stmt, batch, row_idx, &outer_cols);
            let sub_result = Self::execute_select(modified_stmt, &subquery_path)?;
            let sub_batch = sub_result.to_record_batch()?;
            
            if sub_batch.num_rows() == 0 || sub_batch.num_columns() == 0 {
                results.push(None);
            } else if sub_batch.num_rows() > 1 {
                return Err(err_data( "Scalar subquery returned more than one row"));
            } else {
                let sub_col = sub_batch.column(0);
                if sub_col.is_null(0) {
                    results.push(None);
                } else if let Some(arr) = sub_col.as_any().downcast_ref::<Int64Array>() {
                    results.push(Some(arr.value(0)));
                } else if let Some(arr) = sub_col.as_any().downcast_ref::<Float64Array>() {
                    results.push(Some(arr.value(0) as i64));
                } else {
                    results.push(None);
                }
            }
        }
        
        return Ok(Arc::new(Int64Array::from(results)));
    }
    
    /// Execute non-correlated scalar subquery (original implementation)
    fn evaluate_scalar_subquery_simple(batch: &RecordBatch, stmt: &SelectStatement, storage_path: &Path) -> io::Result<ArrayRef> {
        let sub_result = Self::execute_select(stmt.clone(), storage_path)?;
        let sub_batch = sub_result.to_record_batch()?;
        
        if sub_batch.num_rows() == 0 || sub_batch.num_columns() == 0 {
            // Return null array
            return Ok(Arc::new(Int64Array::from(vec![None::<i64>; batch.num_rows()])));
        }
        
        if sub_batch.num_rows() > 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData, 
                "Scalar subquery returned more than one row"
            ));
        }
        
        // Get single value and broadcast
        let sub_col = sub_batch.column(0);
        Self::broadcast_scalar_array(sub_col, 0, batch.num_rows())
    }

    /// Broadcast a single value from array to num_rows
    fn broadcast_scalar_array(array: &ArrayRef, idx: usize, num_rows: usize) -> io::Result<ArrayRef> {
        if array.is_null(idx) {
            return Ok(Arc::new(Int64Array::from(vec![None::<i64>; num_rows])));
        }
        
        use arrow::datatypes::DataType;
        Ok(match array.data_type() {
            DataType::Int64 => {
                let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
                Arc::new(Int64Array::from(vec![arr.value(idx); num_rows]))
            }
            DataType::Float64 => {
                let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
                Arc::new(Float64Array::from(vec![arr.value(idx); num_rows]))
            }
            DataType::Utf8 => {
                let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
                Arc::new(StringArray::from(vec![arr.value(idx); num_rows]))
            }
            DataType::Boolean => {
                let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                Arc::new(BooleanArray::from(vec![arr.value(idx); num_rows]))
            }
            _ => {
                // Fallback - try Int64
                Arc::new(Int64Array::from(vec![None::<i64>; num_rows]))
            }
        })
    }

    /// Evaluate arithmetic operation on pre-computed arrays
    fn evaluate_arithmetic_op_arrays(left: &ArrayRef, right: &ArrayRef, op: &crate::query::sql_parser::BinaryOperator) -> io::Result<ArrayRef> {
        use crate::query::sql_parser::BinaryOperator;
        use arrow::compute::kernels::numeric;
        use arrow::datatypes::DataType;

        // Helper: coerce both sides to Float64Array
        fn to_f64(arr: &ArrayRef) -> io::Result<Float64Array> {
            match arr.data_type() {
                DataType::Float64 => Ok(arr.as_any().downcast_ref::<Float64Array>().unwrap().clone()),
                DataType::Int64 => {
                    let ia = arr.as_any().downcast_ref::<Int64Array>().unwrap();
                    Ok(Float64Array::from_iter(ia.iter().map(|v| v.map(|x| x as f64))))
                }
                _ => Err(err_data(format!("Cannot coerce {:?} to Float64", arr.data_type()))),
            }
        }

        let left_is_float = matches!(left.data_type(), DataType::Float64 | DataType::Float32);
        let right_is_float = matches!(right.data_type(), DataType::Float64 | DataType::Float32);

        if left_is_float || right_is_float {
            let l = to_f64(left)?;
            let r = to_f64(right)?;
            let result: ArrayRef = match op {
                BinaryOperator::Add => Arc::new(numeric::add(&l, &r).map_err(|e| err_data(e.to_string()))?),
                BinaryOperator::Sub => Arc::new(numeric::sub(&l, &r).map_err(|e| err_data(e.to_string()))?),
                BinaryOperator::Mul => Arc::new(numeric::mul(&l, &r).map_err(|e| err_data(e.to_string()))?),
                BinaryOperator::Div => Arc::new(numeric::div(&l, &r).map_err(|e| err_data(e.to_string()))?),
                _ => return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("Unsupported arithmetic operator: {:?}", op))),
            };
            return Ok(result);
        }

        let result: ArrayRef = match op {
            BinaryOperator::Add => Arc::new(numeric::add(
                left.as_any().downcast_ref::<Int64Array>().ok_or_else(|| err_data("Expected Int64"))?,
                right.as_any().downcast_ref::<Int64Array>().ok_or_else(|| err_data("Expected Int64"))?
            ).map_err(|e| err_data(e.to_string()))?),
            BinaryOperator::Sub => Arc::new(numeric::sub(
                left.as_any().downcast_ref::<Int64Array>().ok_or_else(|| err_data("Expected Int64"))?,
                right.as_any().downcast_ref::<Int64Array>().ok_or_else(|| err_data("Expected Int64"))?
            ).map_err(|e| err_data(e.to_string()))?),
            BinaryOperator::Mul => Arc::new(numeric::mul(
                left.as_any().downcast_ref::<Int64Array>().ok_or_else(|| err_data("Expected Int64"))?,
                right.as_any().downcast_ref::<Int64Array>().ok_or_else(|| err_data("Expected Int64"))?
            ).map_err(|e| err_data(e.to_string()))?),
            BinaryOperator::Div => Arc::new(numeric::div(
                left.as_any().downcast_ref::<Int64Array>().ok_or_else(|| err_data("Expected Int64"))?,
                right.as_any().downcast_ref::<Int64Array>().ok_or_else(|| err_data("Expected Int64"))?
            ).map_err(|e| err_data(e.to_string()))?),
            _ => return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Unsupported arithmetic operator: {:?}", op),
            )),
        };
        Ok(result)
    }

    /// Convert Value to Arrow array (broadcast to num_rows)
    fn value_to_array(val: &Value, num_rows: usize) -> io::Result<ArrayRef> {
        Ok(match val {
            Value::Int64(i) => Arc::new(Int64Array::from(vec![*i; num_rows])),
            Value::Int32(i) => Arc::new(Int64Array::from(vec![*i as i64; num_rows])),
            Value::Int16(i) => Arc::new(Int64Array::from(vec![*i as i64; num_rows])),
            Value::Int8(i) => Arc::new(Int64Array::from(vec![*i as i64; num_rows])),
            Value::UInt64(i) => Arc::new(Int64Array::from(vec![*i as i64; num_rows])),
            Value::UInt32(i) => Arc::new(Int64Array::from(vec![*i as i64; num_rows])),
            Value::UInt16(i) => Arc::new(Int64Array::from(vec![*i as i64; num_rows])),
            Value::UInt8(i) => Arc::new(Int64Array::from(vec![*i as i64; num_rows])),
            Value::Float64(f) => Arc::new(Float64Array::from(vec![*f; num_rows])),
            Value::Float32(f) => Arc::new(Float64Array::from(vec![*f as f64; num_rows])),
            Value::String(s) => Arc::new(StringArray::from(vec![s.as_str(); num_rows])),
            Value::Bool(b) => Arc::new(BooleanArray::from(vec![*b; num_rows])),
            Value::Null => Arc::new(Int64Array::from(vec![None::<i64>; num_rows])),
            Value::Binary(b) => {
                use arrow::array::BinaryArray;
                Arc::new(BinaryArray::from(vec![Some(b.as_slice()); num_rows]))
            }
            Value::FixedList(b) => {
                use arrow::array::BinaryArray;
                Arc::new(BinaryArray::from(vec![Some(b.as_slice()); num_rows]))
            }
            Value::Json(j) => {
                let s = j.to_string();
                Arc::new(StringArray::from(vec![s.as_str(); num_rows]))
            }
            Value::Timestamp(ts) => {
                use arrow::array::PrimitiveArray;
                use arrow::datatypes::TimestampMicrosecondType;
                use arrow::buffer::ScalarBuffer;
                Arc::new(PrimitiveArray::<TimestampMicrosecondType>::new(
                    ScalarBuffer::from(vec![*ts; num_rows]), None,
                ))
            }
            Value::Date(d) => {
                use arrow::array::PrimitiveArray;
                use arrow::datatypes::Date32Type;
                use arrow::buffer::ScalarBuffer;
                Arc::new(PrimitiveArray::<Date32Type>::new(
                    ScalarBuffer::from(vec![*d; num_rows]), None,
                ))
            }
            Value::Array(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "Array values not supported in expressions",
                ));
            }
        })
    }

    /// Evaluate arithmetic expression
    fn evaluate_arithmetic_op(
        batch: &RecordBatch,
        left: &SqlExpr,
        op: &crate::query::sql_parser::BinaryOperator,
        right: &SqlExpr,
    ) -> io::Result<ArrayRef> {
        use crate::query::sql_parser::BinaryOperator;
        
        let left_array = Self::evaluate_expr_to_array(batch, left)?;
        let right_array = Self::evaluate_expr_to_array(batch, right)?;

        // Try to cast to common numeric type
        let result: ArrayRef = match op {
            BinaryOperator::Add => {
                if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Int64Array>(),
                    right_array.as_any().downcast_ref::<Int64Array>(),
                ) {
                    Arc::new(arith::add(l, r)
                        .map_err(|e| err_data( e.to_string()))?)
                } else if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Float64Array>(),
                    right_array.as_any().downcast_ref::<Float64Array>(),
                ) {
                    Arc::new(arith::add(l, r)
                        .map_err(|e| err_data( e.to_string()))?)
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Cannot add non-numeric types",
                    ));
                }
            }
            BinaryOperator::Sub => {
                if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Int64Array>(),
                    right_array.as_any().downcast_ref::<Int64Array>(),
                ) {
                    Arc::new(arith::sub(l, r)
                        .map_err(|e| err_data( e.to_string()))?)
                } else if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Float64Array>(),
                    right_array.as_any().downcast_ref::<Float64Array>(),
                ) {
                    Arc::new(arith::sub(l, r)
                        .map_err(|e| err_data( e.to_string()))?)
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Cannot subtract non-numeric types",
                    ));
                }
            }
            BinaryOperator::Mul => {
                if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Int64Array>(),
                    right_array.as_any().downcast_ref::<Int64Array>(),
                ) {
                    Arc::new(arith::mul(l, r)
                        .map_err(|e| err_data( e.to_string()))?)
                } else if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Float64Array>(),
                    right_array.as_any().downcast_ref::<Float64Array>(),
                ) {
                    Arc::new(arith::mul(l, r)
                        .map_err(|e| err_data( e.to_string()))?)
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Cannot multiply non-numeric types",
                    ));
                }
            }
            BinaryOperator::Div => {
                if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Int64Array>(),
                    right_array.as_any().downcast_ref::<Int64Array>(),
                ) {
                    Arc::new(arith::div(l, r)
                        .map_err(|e| err_data( e.to_string()))?)
                } else if let (Some(l), Some(r)) = (
                    left_array.as_any().downcast_ref::<Float64Array>(),
                    right_array.as_any().downcast_ref::<Float64Array>(),
                ) {
                    Arc::new(arith::div(l, r)
                        .map_err(|e| err_data( e.to_string()))?)
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Cannot divide non-numeric types",
                    ));
                }
            }
            // Comparison operators - return BooleanArray
            BinaryOperator::Gt | BinaryOperator::Ge | BinaryOperator::Lt | BinaryOperator::Le | BinaryOperator::Eq | BinaryOperator::NotEq => {
                use arrow::compute::kernels::cmp;
                if let (Some(l), Some(r)) = (left_array.as_any().downcast_ref::<Int64Array>(), right_array.as_any().downcast_ref::<Int64Array>()) {
                    match op {
                        BinaryOperator::Gt => Arc::new(cmp::gt(l, r).map_err(|e| err_data( e.to_string()))?),
                        BinaryOperator::Ge => Arc::new(cmp::gt_eq(l, r).map_err(|e| err_data( e.to_string()))?),
                        BinaryOperator::Lt => Arc::new(cmp::lt(l, r).map_err(|e| err_data( e.to_string()))?),
                        BinaryOperator::Le => Arc::new(cmp::lt_eq(l, r).map_err(|e| err_data( e.to_string()))?),
                        BinaryOperator::Eq => Arc::new(cmp::eq(l, r).map_err(|e| err_data( e.to_string()))?),
                        BinaryOperator::NotEq => Arc::new(cmp::neq(l, r).map_err(|e| err_data( e.to_string()))?),
                        _ => unreachable!(),
                    }
                } else if let (Some(l), Some(r)) = (left_array.as_any().downcast_ref::<Float64Array>(), right_array.as_any().downcast_ref::<Float64Array>()) {
                    match op {
                        BinaryOperator::Gt => Arc::new(cmp::gt(l, r).map_err(|e| err_data( e.to_string()))?),
                        BinaryOperator::Ge => Arc::new(cmp::gt_eq(l, r).map_err(|e| err_data( e.to_string()))?),
                        BinaryOperator::Lt => Arc::new(cmp::lt(l, r).map_err(|e| err_data( e.to_string()))?),
                        BinaryOperator::Le => Arc::new(cmp::lt_eq(l, r).map_err(|e| err_data( e.to_string()))?),
                        BinaryOperator::Eq => Arc::new(cmp::eq(l, r).map_err(|e| err_data( e.to_string()))?),
                        BinaryOperator::NotEq => Arc::new(cmp::neq(l, r).map_err(|e| err_data( e.to_string()))?),
                        _ => unreachable!(),
                    }
                } else if let (Some(l), Some(r)) = (left_array.as_any().downcast_ref::<StringArray>(), right_array.as_any().downcast_ref::<StringArray>()) {
                    match op {
                        BinaryOperator::Gt => Arc::new(cmp::gt(l, r).map_err(|e| err_data( e.to_string()))?),
                        BinaryOperator::Ge => Arc::new(cmp::gt_eq(l, r).map_err(|e| err_data( e.to_string()))?),
                        BinaryOperator::Lt => Arc::new(cmp::lt(l, r).map_err(|e| err_data( e.to_string()))?),
                        BinaryOperator::Le => Arc::new(cmp::lt_eq(l, r).map_err(|e| err_data( e.to_string()))?),
                        BinaryOperator::Eq => Arc::new(cmp::eq(l, r).map_err(|e| err_data( e.to_string()))?),
                        BinaryOperator::NotEq => Arc::new(cmp::neq(l, r).map_err(|e| err_data( e.to_string()))?),
                        _ => unreachable!(),
                    }
                } else {
                    return Err(err_data( "Cannot compare incompatible types"));
                }
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("Unsupported arithmetic operator: {:?}", op),
                ));
            }
        };

        Ok(result)
    }

    /// Evaluate CAST expression
    fn evaluate_cast_expr(
        batch: &RecordBatch,
        expr: &SqlExpr,
        target_type: &DataType,
    ) -> io::Result<ArrayRef> {
        let arr = Self::evaluate_expr_to_array(batch, expr)?;
        let num_rows = batch.num_rows();
        
        match target_type {
            DataType::Int64 => {
                if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                    Ok(Arc::new(int_arr.clone()))
                } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                    let result: Vec<Option<i64>> = (0..num_rows).map(|i| {
                        if float_arr.is_null(i) { None } else { Some(float_arr.value(i) as i64) }
                    }).collect();
                    Ok(Arc::new(Int64Array::from(result)))
                } else if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                    let mut result: Vec<Option<i64>> = Vec::with_capacity(num_rows);
                    for i in 0..num_rows {
                        if str_arr.is_null(i) {
                            result.push(None);
                            continue;
                        }
                        let s = str_arr.value(i);
                        match s.parse::<i64>() {
                            Ok(v) => result.push(Some(v)),
                            Err(_) => {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!("Invalid cast to INT64 for value '{}'", s),
                                ));
                            }
                        }
                    }
                    Ok(Arc::new(Int64Array::from(result)))
                } else {
                    Err(err_data( "Cannot cast to INT64"))
                }
            }
            DataType::Float64 => {
                if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                    Ok(Arc::new(float_arr.clone()))
                } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                    let result: Vec<Option<f64>> = (0..num_rows).map(|i| {
                        if int_arr.is_null(i) { None } else { Some(int_arr.value(i) as f64) }
                    }).collect();
                    Ok(Arc::new(Float64Array::from(result)))
                } else if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                    let mut result: Vec<Option<f64>> = Vec::with_capacity(num_rows);
                    for i in 0..num_rows {
                        if str_arr.is_null(i) {
                            result.push(None);
                            continue;
                        }
                        let s = str_arr.value(i);
                        match s.parse::<f64>() {
                            Ok(v) => result.push(Some(v)),
                            Err(_) => {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!("Invalid cast to FLOAT64 for value '{}'", s),
                                ));
                            }
                        }
                    }
                    Ok(Arc::new(Float64Array::from(result)))
                } else {
                    Err(err_data( "Cannot cast to FLOAT64"))
                }
            }
            DataType::String => {
                if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                    Ok(Arc::new(str_arr.clone()))
                } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                    let result: Vec<Option<String>> = (0..num_rows).map(|i| {
                        if int_arr.is_null(i) { None } else { Some(int_arr.value(i).to_string()) }
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                    let result: Vec<Option<String>> = (0..num_rows).map(|i| {
                        if float_arr.is_null(i) { None } else { Some(float_arr.value(i).to_string()) }
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else {
                    Err(err_data( "Cannot cast to STRING"))
                }
            }
            DataType::Bool => {
                if let Some(bool_arr) = arr.as_any().downcast_ref::<BooleanArray>() {
                    Ok(Arc::new(bool_arr.clone()))
                } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                    let result: Vec<Option<bool>> = (0..num_rows).map(|i| {
                        if int_arr.is_null(i) { None } else { Some(int_arr.value(i) != 0) }
                    }).collect();
                    Ok(Arc::new(BooleanArray::from(result)))
                } else if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                    let result: Vec<Option<bool>> = (0..num_rows).map(|i| {
                        if str_arr.is_null(i) { None } 
                        else { 
                            let s = str_arr.value(i).to_lowercase();
                            Some(s == "true" || s == "1" || s == "yes" || s == "t")
                        }
                    }).collect();
                    Ok(Arc::new(BooleanArray::from(result)))
                } else {
                    Err(err_data( "Cannot cast to BOOL"))
                }
            }
            DataType::Int32 => {
                // Treat Int32 same as Int64 for simplicity, return Int64 array
                if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                    Ok(Arc::new(int_arr.clone()))
                } else if let Some(float_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                    let result: Vec<Option<i64>> = (0..num_rows).map(|i| {
                        if float_arr.is_null(i) { None } else { Some(float_arr.value(i) as i64) }
                    }).collect();
                    Ok(Arc::new(Int64Array::from(result)))
                } else if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                    let mut result: Vec<Option<i64>> = Vec::with_capacity(num_rows);
                    for i in 0..num_rows {
                        if str_arr.is_null(i) {
                            result.push(None);
                            continue;
                        }
                        let s = str_arr.value(i);
                        match s.parse::<i64>() {
                            Ok(v) => result.push(Some(v)),
                            Err(_) => {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!("Invalid cast to INT32 for value '{}'", s),
                                ));
                            }
                        }
                    }
                    Ok(Arc::new(Int64Array::from(result)))
                } else {
                    Err(err_data( "Cannot cast to INT32"))
                }
            }
            _ => Err(io::Error::new(io::ErrorKind::Unsupported, format!("CAST to {:?} not supported", target_type))),
        }
    }

    /// Evaluate CASE WHEN expression
    fn evaluate_case_expr(
        batch: &RecordBatch,
        when_then: &[(SqlExpr, SqlExpr)],
        else_expr: Option<&SqlExpr>,
    ) -> io::Result<ArrayRef> {
        let num_rows = batch.num_rows();
        
        // Determine result type from first THEN expression
        let first_then = Self::evaluate_expr_to_array(batch, &when_then[0].1)?;
        let is_string = first_then.as_any().downcast_ref::<StringArray>().is_some();
        
        if is_string {
            // Handle string CASE results
            let mut result: Vec<Option<String>> = if let Some(else_e) = else_expr {
                let else_array = Self::evaluate_expr_to_array(batch, else_e)?;
                if let Some(arr) = else_array.as_any().downcast_ref::<StringArray>() {
                    (0..num_rows).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i).to_string()) }).collect()
                } else {
                    vec![None; num_rows]
                }
            } else {
                vec![None; num_rows]
            };
            
            let mut assigned = vec![false; num_rows];
            
            for (cond_expr, then_expr) in when_then {
                let cond = Self::evaluate_predicate(batch, cond_expr)?;
                let then_array = Self::evaluate_expr_to_array(batch, then_expr)?;
                
                if let Some(then_str) = then_array.as_any().downcast_ref::<StringArray>() {
                    for i in 0..num_rows {
                        if !assigned[i] && cond.value(i) {
                            result[i] = if then_str.is_null(i) { None } else { Some(then_str.value(i).to_string()) };
                            assigned[i] = true;
                        }
                    }
                }
            }
            
            Ok(Arc::new(StringArray::from(result)))
        } else {
            // Handle numeric CASE results (Int64)
            let mut result: Vec<Option<i64>> = if let Some(else_e) = else_expr {
                let else_array = Self::evaluate_expr_to_array(batch, else_e)?;
                if let Some(arr) = else_array.as_any().downcast_ref::<Int64Array>() {
                    (0..num_rows).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) }).collect()
                } else {
                    vec![None; num_rows]
                }
            } else {
                vec![None; num_rows]
            };
            
            let mut assigned = vec![false; num_rows];
            
            for (cond_expr, then_expr) in when_then {
                let cond = Self::evaluate_predicate(batch, cond_expr)?;
                let then_array = Self::evaluate_expr_to_array(batch, then_expr)?;
                
                if let Some(then_int) = then_array.as_any().downcast_ref::<Int64Array>() {
                    for i in 0..num_rows {
                        if !assigned[i] && cond.value(i) {
                            result[i] = if then_int.is_null(i) { None } else { Some(then_int.value(i)) };
                            assigned[i] = true;
                        }
                    }
                }
            }
            
            Ok(Arc::new(Int64Array::from(result)))
        }
    }

    /// Evaluate function expression (COALESCE, etc.)
    fn evaluate_function_expr(
        batch: &RecordBatch,
        name: &str,
        args: &[SqlExpr],
    ) -> io::Result<ArrayRef> {
        // Handle aggregate function references (for HAVING clause) — zero-allocation dispatch
        let agg_upper: &str = if name.eq_ignore_ascii_case("COUNT") { "COUNT" }
            else if name.eq_ignore_ascii_case("SUM") { "SUM" }
            else if name.eq_ignore_ascii_case("AVG") { "AVG" }
            else if name.eq_ignore_ascii_case("MIN") { "MIN" }
            else if name.eq_ignore_ascii_case("MAX") { "MAX" }
            else { "" };
        if !agg_upper.is_empty() {
                // Build possible column names as they might appear in the result batch
                let col_name = if args.is_empty() {
                    format!("{}(*)", agg_upper)
                } else if let Some(SqlExpr::Literal(Value::String(s))) = args.first() {
                    if s == "*" {
                        format!("{}(*)", agg_upper)
                    } else {
                        format!("{}({})", agg_upper, s)
                    }
                } else if let Some(SqlExpr::Column(col)) = args.first() {
                    format!("{}({})", agg_upper, col)
                } else if let Some(SqlExpr::Literal(Value::Int64(n))) = args.first() {
                    format!("{}({})", agg_upper, n)
                } else {
                    format!("{}(*)", agg_upper)
                };
                
                // Try to find the column in the batch by exact name
                if let Some(array) = batch.column_by_name(&col_name) {
                    return Ok(array.clone());
                }
                // Also try lowercase version
                let lower_col = col_name.to_lowercase();
                if let Some(array) = batch.column_by_name(&lower_col) {
                    return Ok(array.clone());
                }
                // Try with just the function name pattern (handles aliased columns)
                let prefix = format!("{}(", agg_upper);
                for field in batch.schema().fields() {
                    if field.name().eq_ignore_ascii_case(&col_name) || field.name().to_uppercase().starts_with(&prefix) {
                        return batch.column_by_name(field.name()).cloned()
                            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("Column not found: {}", col_name)));
                    }
                }
                // If column not found by name pattern, it might be aliased
                // Count how many aggregate-like columns we have (numeric, not group-by columns)
                let mut agg_columns: Vec<(usize, String)> = Vec::new();
                let mut string_columns = 0;
                for (idx, field) in batch.schema().fields().iter().enumerate() {
                    match field.data_type() {
                        arrow::datatypes::DataType::Utf8 => string_columns += 1,
                        arrow::datatypes::DataType::Int64 | arrow::datatypes::DataType::Float64 => {
                            // Numeric column after string columns are likely aggregates
                            if idx >= string_columns && !field.name().contains('(') {
                                agg_columns.push((idx, field.name().clone()));
                            }
                        }
                        _ => {}
                    }
                }
                
                // Try to match based on aggregate type position
                if !agg_columns.is_empty() {
                    let target_idx = match agg_upper {
                        "COUNT" => 0,
                        "SUM" => if agg_columns.len() > 1 { 1 } else { 0 },
                        "AVG" => if agg_columns.len() > 2 { 2 } else { agg_columns.len().saturating_sub(1) },
                        "MIN" | "MAX" => agg_columns.len().saturating_sub(1),
                        _ => 0,
                    };
                    let idx = target_idx.min(agg_columns.len().saturating_sub(1));
                    return Ok(batch.column(agg_columns[idx].0).clone());
                }
                
                return Err(io::Error::new(io::ErrorKind::NotFound, format!("Aggregate column '{}' not found in result", col_name)));
        }
        
        let upper = name.to_uppercase();
        match upper.as_str() {
            "GETVARIABLE" => {
                if args.len() != 1 {
                    return Err(err_input("GETVARIABLE requires exactly one argument"));
                }
                let var_name = match &args[0] {
                    SqlExpr::Literal(crate::data::Value::String(s)) => s.clone(),
                    SqlExpr::Column(s) => s.clone(),
                    other => {
                        let arr = Self::evaluate_expr_to_array(batch, other)?;
                        if let Some(s) = arr.as_any().downcast_ref::<StringArray>() {
                            s.value(0).to_string()
                        } else {
                            return Err(err_input("GETVARIABLE argument must be a string"));
                        }
                    }
                };
                use crate::query::executor::get_session_variable;
                let value = get_session_variable(&var_name).unwrap_or(crate::data::Value::Null);
                return Self::value_to_array(&value, batch.num_rows());
            }
            "COALESCE" => {
                if args.is_empty() {
                    return Err(err_input( "COALESCE requires at least one argument"));
                }
                
                let num_rows = batch.num_rows();
                
                // Determine result type from first non-null argument (skip arrays that are all null)
                let mut result_type: Option<&str> = None;
                let mut arrays: Vec<ArrayRef> = Vec::new();
                for arg in args {
                    let arr = Self::evaluate_expr_to_array(batch, arg)?;
                    // Only set type from arrays that have at least one non-null value
                    let has_non_null = (0..arr.len()).any(|i| !arr.is_null(i));
                    if result_type.is_none() && has_non_null {
                        if arr.as_any().downcast_ref::<StringArray>().is_some() {
                            result_type = Some("string");
                        } else if arr.as_any().downcast_ref::<Int64Array>().is_some() {
                            result_type = Some("int");
                        } else if arr.as_any().downcast_ref::<Float64Array>().is_some() {
                            result_type = Some("float");
                        }
                    }
                    arrays.push(arr);
                }
                
                match result_type.unwrap_or("int") {
                    "string" => {
                        let mut result: Vec<Option<String>> = vec![None; num_rows];
                        let mut assigned = vec![false; num_rows];
                        for arr in &arrays {
                            if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                                for i in 0..num_rows {
                                    if !assigned[i] && !str_arr.is_null(i) {
                                        result[i] = Some(str_arr.value(i).to_string());
                                        assigned[i] = true;
                                    }
                                }
                            }
                        }
                        Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                    }
                    "float" => {
                        let mut result: Vec<Option<f64>> = vec![None; num_rows];
                        let mut assigned = vec![false; num_rows];
                        for arr in &arrays {
                            if let Some(f_arr) = arr.as_any().downcast_ref::<Float64Array>() {
                                for i in 0..num_rows {
                                    if !assigned[i] && !f_arr.is_null(i) {
                                        result[i] = Some(f_arr.value(i));
                                        assigned[i] = true;
                                    }
                                }
                            }
                        }
                        Ok(Arc::new(Float64Array::from(result)))
                    }
                    _ => {
                        let mut result: Vec<Option<i64>> = vec![None; num_rows];
                        let mut assigned = vec![false; num_rows];
                        for arr in &arrays {
                            if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                                for i in 0..num_rows {
                                    if !assigned[i] && !int_arr.is_null(i) {
                                        result[i] = Some(int_arr.value(i));
                                        assigned[i] = true;
                                    }
                                }
                            }
                        }
                        Ok(Arc::new(Int64Array::from(result)))
                    }
                }
            }
            "IFNULL" | "NVL" | "ISNULL" => {
                if args.len() != 2 {
                    return Err(err_input( format!("{} requires exactly 2 arguments", upper)));
                }
                
                let arr1 = Self::evaluate_expr_to_array(batch, &args[0])?;
                let arr2 = Self::evaluate_expr_to_array(batch, &args[1])?;
                let num_rows = batch.num_rows();
                
                // Check if arr1 is all nulls - use arr2's type
                let arr1_all_null = (0..num_rows).all(|i| arr1.is_null(i));
                
                // Try integer types
                if let Some(int2) = arr2.as_any().downcast_ref::<Int64Array>() {
                    if arr1_all_null || arr1.as_any().downcast_ref::<Int64Array>().is_some() {
                        let int1 = arr1.as_any().downcast_ref::<Int64Array>();
                        let result: Vec<Option<i64>> = (0..num_rows).map(|i| {
                            if let Some(i1) = int1 { if !i1.is_null(i) { return Some(i1.value(i)); } }
                            if !int2.is_null(i) { Some(int2.value(i)) } else { None }
                        }).collect();
                        return Ok(Arc::new(Int64Array::from(result)));
                    }
                }
                // Try string types
                if let Some(str2) = arr2.as_any().downcast_ref::<StringArray>() {
                    if arr1_all_null || arr1.as_any().downcast_ref::<StringArray>().is_some() {
                        let str1 = arr1.as_any().downcast_ref::<StringArray>();
                        let result: Vec<Option<&str>> = (0..num_rows).map(|i| {
                            if let Some(s1) = str1 { if !s1.is_null(i) { return Some(s1.value(i)); } }
                            if !str2.is_null(i) { Some(str2.value(i)) } else { None }
                        }).collect();
                        return Ok(Arc::new(StringArray::from(result)));
                    }
                }
                // Try float types
                if let Some(f2) = arr2.as_any().downcast_ref::<Float64Array>() {
                    if arr1_all_null || arr1.as_any().downcast_ref::<Float64Array>().is_some() {
                        let f1 = arr1.as_any().downcast_ref::<Float64Array>();
                        let result: Vec<Option<f64>> = (0..num_rows).map(|i| {
                            if let Some(ff1) = f1 { if !ff1.is_null(i) { return Some(ff1.value(i)); } }
                            if !f2.is_null(i) { Some(f2.value(i)) } else { None }
                        }).collect();
                        return Ok(Arc::new(Float64Array::from(result)));
                    }
                }
                // Default: return arr2 if arr1 is all null
                if arr1_all_null {
                    return Ok(arr2);
                }
                Err(err_data( "IFNULL/NVL argument types must match"))
            }
            "ABS" => {
                if args.len() != 1 { return Err(err_input("ABS requires exactly 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                map_numeric_unary(&arr, batch.num_rows(), |x| x.abs(), |x| x.abs(), "ABS")
            }
            "NULLIF" => {
                if args.len() != 2 { return Err(err_input("NULLIF requires 2 arguments")); }
                let arr1 = Self::evaluate_expr_to_array(batch, &args[0])?;
                let arr2 = Self::evaluate_expr_to_array(batch, &args[1])?;
                let n = batch.num_rows();
                if let (Some(i1), Some(i2)) = (arr1.as_any().downcast_ref::<Int64Array>(), arr2.as_any().downcast_ref::<Int64Array>()) {
                    let result: Vec<Option<i64>> = (0..n).map(|i| if i1.is_null(i) || (!i2.is_null(i) && i1.value(i) == i2.value(i)) { None } else { Some(i1.value(i)) }).collect();
                    Ok(Arc::new(Int64Array::from(result)))
                } else if let (Some(s1), Some(s2)) = (arr1.as_any().downcast_ref::<StringArray>(), arr2.as_any().downcast_ref::<StringArray>()) {
                    let result: Vec<Option<&str>> = (0..n).map(|i| if s1.is_null(i) || (!s2.is_null(i) && s1.value(i) == s2.value(i)) { None } else { Some(s1.value(i)) }).collect();
                    Ok(Arc::new(StringArray::from(result)))
                } else { Err(err_data("NULLIF type mismatch")) }
            }
            "UPPER" | "UCASE" | "LOWER" | "LCASE" => {
                let is_upper = upper == "UPPER" || upper == "UCASE";
                if args.len() != 1 { return Err(err_input(format!("{} requires 1 argument", upper))); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                if (0..arr.len()).all(|i| arr.is_null(i)) { return Ok(Arc::new(StringArray::from(vec![None::<&str>; batch.num_rows()]))); }
                map_string_unary(&arr, batch.num_rows(), |s| if is_upper { s.to_uppercase() } else { s.to_lowercase() }, &upper)
            }
            "LENGTH" | "LEN" | "CHAR_LENGTH" | "CHARACTER_LENGTH" => {
                if args.len() != 1 { return Err(err_input("LENGTH requires 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                map_string_to_int(&arr, batch.num_rows(), |s| s.chars().count() as i64, "LENGTH")
            }
            "TRIM" => {
                if args.len() != 1 { return Err(err_input("TRIM requires exactly 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                map_string_unary(&arr, batch.num_rows(), |s| s.trim().to_string(), "TRIM")
            }
            "CONCAT" => {
                if args.is_empty() {
                    return Ok(Arc::new(StringArray::from(vec![""; batch.num_rows()])));
                }
                let num_rows = batch.num_rows();
                let mut result: Vec<String> = vec![String::new(); num_rows];
                for arg in args {
                    let arr = Self::evaluate_expr_to_array(batch, arg)?;
                    if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
                        for i in 0..num_rows {
                            if !str_arr.is_null(i) {
                                result[i].push_str(str_arr.value(i));
                            }
                        }
                    } else if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() {
                        for i in 0..num_rows {
                            if !int_arr.is_null(i) {
                                result[i].push_str(&int_arr.value(i).to_string());
                            }
                        }
                    }
                }
                Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_str()).collect::<Vec<_>>())))
            }
            "SUBSTR" | "SUBSTRING" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(err_input( "SUBSTR requires 2 or 3 arguments"));
                }
                let str_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let start_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                let len_arr = if args.len() == 3 { Some(Self::evaluate_expr_to_array(batch, &args[2])?) } else { None };
                
                if let (Some(strs), Some(starts)) = (
                    str_arr.as_any().downcast_ref::<StringArray>(),
                    start_arr.as_any().downcast_ref::<Int64Array>(),
                ) {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| {
                        if strs.is_null(i) || starts.is_null(i) { return None; }
                        let s = strs.value(i);
                        let start = (starts.value(i).max(1) - 1) as usize;
                        if start >= s.len() { return Some(String::new()); }
                        let len = if let Some(ref larr) = len_arr {
                            if let Some(la) = larr.as_any().downcast_ref::<Int64Array>() {
                                if la.is_null(i) { s.len() } else { la.value(i).max(0) as usize }
                            } else { s.len() }
                        } else { s.len() };
                        Some(s.chars().skip(start).take(len).collect())
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else {
                    Err(err_data( "SUBSTR type mismatch"))
                }
            }
            "REPLACE" => {
                if args.len() != 3 { return Err(err_input("REPLACE requires 3 arguments")); }
                let str_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let from_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                let to_arr = Self::evaluate_expr_to_array(batch, &args[2])?;
                if let (Some(strs), Some(froms), Some(tos)) = (str_arr.as_any().downcast_ref::<StringArray>(), from_arr.as_any().downcast_ref::<StringArray>(), to_arr.as_any().downcast_ref::<StringArray>()) {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| {
                        if strs.is_null(i) { None } else { Some(strs.value(i).replace(if froms.is_null(i) { "" } else { froms.value(i) }, if tos.is_null(i) { "" } else { tos.value(i) })) }
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else { Err(err_data("REPLACE requires string arguments")) }
            }
            "ROUND" => {
                if args.is_empty() || args.len() > 2 { return Err(err_input("ROUND requires 1-2 arguments")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let dec = if args.len() == 2 { Self::evaluate_expr_to_array(batch, &args[1])?.as_any().downcast_ref::<Int64Array>().map_or(0, |a| if a.len() > 0 && !a.is_null(0) { a.value(0) as i32 } else { 0 }) } else { 0 };
                if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() { return Ok(Arc::new(int_arr.clone())); }
                if let Some(fa) = arr.as_any().downcast_ref::<Float64Array>() {
                    let f = 10f64.powi(dec);
                    let result: Vec<Option<f64>> = (0..batch.num_rows()).map(|i| if fa.is_null(i) { None } else { Some((fa.value(i) * f).round() / f) }).collect();
                    Ok(Arc::new(Float64Array::from(result)))
                } else { Err(err_data("ROUND requires numeric argument")) }
            }
            "FLOOR" | "CEIL" | "CEILING" => {
                let is_floor = upper == "FLOOR";
                if args.len() != 1 { return Err(err_input(format!("{} requires 1 argument", upper))); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                if let Some(int_arr) = arr.as_any().downcast_ref::<Int64Array>() { return Ok(Arc::new(int_arr.clone())); }
                map_numeric_unary(&arr, batch.num_rows(), |x| x, |x| if is_floor { x.floor() } else { x.ceil() }, &upper)
            }
            "MOD" => {
                if args.len() != 2 { return Err(err_input("MOD requires 2 arguments")); }
                let arr1 = Self::evaluate_expr_to_array(batch, &args[0])?;
                let arr2 = Self::evaluate_expr_to_array(batch, &args[1])?;
                if let (Some(i1), Some(i2)) = (arr1.as_any().downcast_ref::<Int64Array>(), arr2.as_any().downcast_ref::<Int64Array>()) {
                    let result: Vec<Option<i64>> = (0..batch.num_rows()).map(|i| if i1.is_null(i) || i2.is_null(i) || i2.value(i) == 0 { None } else { Some(i1.value(i) % i2.value(i)) }).collect();
                    Ok(Arc::new(Int64Array::from(result)))
                } else { Err(err_data("MOD requires integer arguments")) }
            }
            "SQRT" => {
                if args.len() != 1 { return Err(err_input("SQRT requires 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                map_numeric_unary(&arr, batch.num_rows(), |x| if x < 0 { 0 } else { ((x as f64).sqrt()) as i64 }, |x| if x < 0.0 { f64::NAN } else { x.sqrt() }, "SQRT")
            }
            "MID" | "SUBSTR" | "SUBSTRING" => {
                if args.len() < 2 || args.len() > 3 { return Err(err_input("SUBSTR requires 2-3 arguments")); }
                let str_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let start_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                let len_arr = if args.len() == 3 { Some(Self::evaluate_expr_to_array(batch, &args[2])?) } else { None };
                if let (Some(strs), Some(starts)) = (str_arr.as_any().downcast_ref::<StringArray>(), start_arr.as_any().downcast_ref::<Int64Array>()) {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| {
                        if strs.is_null(i) || starts.is_null(i) { return None; }
                        let s = strs.value(i); let start = (starts.value(i).max(1) - 1) as usize;
                        let len = len_arr.as_ref().and_then(|la| la.as_any().downcast_ref::<Int64Array>()).map_or(s.len(), |ia| if ia.is_null(i) { s.len() } else { ia.value(i).max(0) as usize });
                        let chars: Vec<char> = s.chars().collect();
                        Some(if start >= chars.len() { String::new() } else { chars[start..].iter().take(len).collect() })
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else { Err(err_data("SUBSTR requires string and integer arguments")) }
            }
            "NOW" | "CURRENT_TIMESTAMP" => {
                let now_str = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
                Ok(Arc::new(StringArray::from(vec![Some(now_str.as_str()); batch.num_rows()])))
            }
            "RAND" | "RANDOM" => {
                use std::time::{SystemTime, UNIX_EPOCH};
                let seed = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0) as u64;
                let result: Vec<f64> = (0..batch.num_rows()).map(|i| {
                    // Simple LCG random number generator with better distribution
                    let mut state = seed.wrapping_add((i as u64).wrapping_mul(2685821657736338717));
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    state ^= state >> 33;
                    state = state.wrapping_mul(0xff51afd7ed558ccd);
                    state ^= state >> 33;
                    (state as f64) / (u64::MAX as f64)
                }).collect();
                Ok(Arc::new(Float64Array::from(result)))
            }
            // ============== Hive String Functions ==============
            "INSTR" => {
                if args.len() != 2 { return Err(err_input("INSTR requires 2 arguments")); }
                let str_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let substr_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                if let (Some(strs), Some(substrs)) = (str_arr.as_any().downcast_ref::<StringArray>(), substr_arr.as_any().downcast_ref::<StringArray>()) {
                    let result: Vec<Option<i64>> = (0..batch.num_rows()).map(|i| {
                        if strs.is_null(i) || substrs.is_null(i) { None } 
                        else { let s = strs.value(i); let sub = substrs.value(i); s.char_indices().enumerate().find(|(_, (idx, _))| s[*idx..].starts_with(sub)).map_or(Some(0), |(pos, _)| Some((pos + 1) as i64)) }
                    }).collect();
                    Ok(Arc::new(Int64Array::from(result)))
                } else { Err(err_data("INSTR requires string arguments")) }
            }
            "LOCATE" => {
                if args.len() < 2 || args.len() > 3 { return Err(err_input( "LOCATE requires 2 or 3 arguments")); }
                let substr_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let str_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                let start_arr = if args.len() == 3 { Some(Self::evaluate_expr_to_array(batch, &args[2])?) } else { None };
                if let (Some(substrs), Some(strs)) = (substr_arr.as_any().downcast_ref::<StringArray>(), str_arr.as_any().downcast_ref::<StringArray>()) {
                    let result: Vec<Option<i64>> = (0..batch.num_rows()).map(|i| {
                        if strs.is_null(i) || substrs.is_null(i) { return None; }
                        let s = strs.value(i); let sub = substrs.value(i);
                        let start = start_arr.as_ref().and_then(|sa| sa.as_any().downcast_ref::<Int64Array>()).map_or(1, |ia| if ia.is_null(i) { 1 } else { ia.value(i).max(1) });
                        let offset = (start - 1) as usize;
                        if offset >= s.len() { Some(0) } else { Some(s[offset..].find(sub).map_or(0, |p| (p + offset + 1) as i64)) }
                    }).collect();
                    Ok(Arc::new(Int64Array::from(result)))
                } else { Err(err_data( "LOCATE requires string arguments")) }
            }
            "LPAD" | "RPAD" => {
                let is_lpad = upper == "LPAD";
                if args.len() != 3 { return Err(err_input( format!("{} requires 3 arguments", upper))); }
                let str_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let len_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                let pad_arr = Self::evaluate_expr_to_array(batch, &args[2])?;
                if let (Some(strs), Some(lens), Some(pads)) = (str_arr.as_any().downcast_ref::<StringArray>(), len_arr.as_any().downcast_ref::<Int64Array>(), pad_arr.as_any().downcast_ref::<StringArray>()) {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| {
                        if strs.is_null(i) || lens.is_null(i) { return None; }
                        let s = strs.value(i); let tlen = lens.value(i) as usize; let pad = if pads.is_null(i) { " " } else { pads.value(i) };
                        if s.chars().count() >= tlen { Some(s.chars().take(tlen).collect()) }
                        else { 
                            let pc: Vec<char> = pad.chars().collect(); 
                            if pc.is_empty() { Some(s.to_string()) } 
                            else if is_lpad { let mut r = String::new(); for j in 0..(tlen - s.chars().count()) { r.push(pc[j % pc.len()]); } r.push_str(s); Some(r) }
                            else { let mut r = s.to_string(); for j in 0..(tlen - s.chars().count()) { r.push(pc[j % pc.len()]); } Some(r) }
                        }
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else { Err(err_data( format!("{} type error", upper))) }
            }
            "LTRIM" | "RTRIM" => {
                let is_ltrim = upper == "LTRIM";
                if args.len() != 1 { return Err(err_input(format!("{} requires 1 argument", upper))); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                map_string_unary(&arr, batch.num_rows(), |s| if is_ltrim { s.trim_start().to_string() } else { s.trim_end().to_string() }, &upper)
            }
            "REVERSE" => {
                if args.len() != 1 { return Err(err_input("REVERSE requires 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                map_string_unary(&arr, batch.num_rows(), |s| s.chars().rev().collect(), "REVERSE")
            }
            "INITCAP" => {
                if args.len() != 1 { return Err(err_input("INITCAP requires 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                map_string_unary(&arr, batch.num_rows(), |s| {
                    let mut r = String::new(); let mut cap = true;
                    for c in s.chars() { if c.is_whitespace() || !c.is_alphanumeric() { r.push(c); cap = true; } else if cap { r.extend(c.to_uppercase()); cap = false; } else { r.extend(c.to_lowercase()); } }
                    r
                }, "INITCAP")
            }
            "CONCAT_WS" => {
                if args.len() < 2 { return Err(err_input("CONCAT_WS requires 2+ arguments")); }
                let sep = Self::evaluate_expr_to_array(batch, &args[0])?.as_any().downcast_ref::<StringArray>().map_or(String::new(), |sa| if sa.len() > 0 && !sa.is_null(0) { sa.value(0).to_string() } else { String::new() });
                let (num_rows, mut result, mut first) = (batch.num_rows(), vec![String::new(); batch.num_rows()], vec![true; batch.num_rows()]);
                for arg in &args[1..] { if let Ok(arr) = Self::evaluate_expr_to_array(batch, arg) { if let Some(sa) = arr.as_any().downcast_ref::<StringArray>() { for i in 0..num_rows { if !sa.is_null(i) { if !first[i] { result[i].push_str(&sep); } result[i].push_str(sa.value(i)); first[i] = false; } } } } }
                Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_str()).collect::<Vec<_>>())))
            }
            "REPEAT" => {
                if args.len() != 2 { return Err(err_input( "REPEAT requires 2 arguments")); }
                let str_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let n_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                if let (Some(strs), Some(ns)) = (str_arr.as_any().downcast_ref::<StringArray>(), n_arr.as_any().downcast_ref::<Int64Array>()) {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| if strs.is_null(i) || ns.is_null(i) { None } else { Some(strs.value(i).repeat(ns.value(i).max(0) as usize)) }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else { Err(err_data( "REPEAT type error")) }
            }
            "SPACE" => {
                if args.len() != 1 { return Err(err_input("SPACE requires 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                map_int_to_string(&arr, batch.num_rows(), |v| Some(" ".repeat(v.max(0) as usize)), "SPACE")
            }
            "ASCII" => {
                if args.len() != 1 { return Err(err_input("ASCII requires 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                map_string_to_int(&arr, batch.num_rows(), |s| s.chars().next().map_or(0, |c| c as i64), "ASCII")
            }
            "CHR" | "CHAR" => {
                if args.len() != 1 { return Err(err_input("CHR requires 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                map_int_to_string(&arr, batch.num_rows(), |v| char::from_u32(v as u32).map(|c| c.to_string()), "CHR")
            }
            "LEFT" | "RIGHT" => {
                let is_left = upper == "LEFT";
                if args.len() != 2 { return Err(err_input(format!("{} requires 2 arguments", upper))); }
                let str_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let n_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                if let (Some(strs), Some(ns)) = (str_arr.as_any().downcast_ref::<StringArray>(), n_arr.as_any().downcast_ref::<Int64Array>()) {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| {
                        if strs.is_null(i) || ns.is_null(i) { None } 
                        else { let c: Vec<char> = strs.value(i).chars().collect(); let n = ns.value(i).max(0) as usize;
                            Some(if is_left { c.iter().take(n).collect() } else { c[c.len().saturating_sub(n)..].iter().collect() }) }
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else { Err(err_data(format!("{} type error", upper))) }
            }
            "REGEXP_REPLACE" => {
                if args.len() != 3 { return Err(err_input( "REGEXP_REPLACE requires 3 arguments")); }
                let str_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let pat_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                let rep_arr = Self::evaluate_expr_to_array(batch, &args[2])?;
                if let (Some(strs), Some(pats), Some(reps)) = (str_arr.as_any().downcast_ref::<StringArray>(), pat_arr.as_any().downcast_ref::<StringArray>(), rep_arr.as_any().downcast_ref::<StringArray>()) {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| { if strs.is_null(i) { None } else { let s = strs.value(i); let p = if pats.is_null(i) { "" } else { pats.value(i) }; let r = if reps.is_null(i) { "" } else { reps.value(i) }; regex::Regex::new(p).ok().map(|re| re.replace_all(s, r).to_string()).or(Some(s.to_string())) } }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else { Err(err_data( "REGEXP_REPLACE type error")) }
            }
            "REGEXP_EXTRACT" => {
                if args.len() < 2 || args.len() > 3 { return Err(err_input( "REGEXP_EXTRACT requires 2-3 arguments")); }
                let str_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let pat_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                let grp_arr = if args.len() == 3 { Some(Self::evaluate_expr_to_array(batch, &args[2])?) } else { None };
                if let (Some(strs), Some(pats)) = (str_arr.as_any().downcast_ref::<StringArray>(), pat_arr.as_any().downcast_ref::<StringArray>()) {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| { if strs.is_null(i) || pats.is_null(i) { return None; } let gi = grp_arr.as_ref().and_then(|g| g.as_any().downcast_ref::<Int64Array>()).map_or(0, |ia| if ia.is_null(i) { 0 } else { ia.value(i) as usize }); regex::Regex::new(pats.value(i)).ok().and_then(|re| re.captures(strs.value(i)).and_then(|c| c.get(gi).map(|m| m.as_str().to_string()))) }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else { Err(err_data( "REGEXP_EXTRACT type error")) }
            }
            "SPLIT" => {
                if args.len() != 2 { return Err(err_input("SPLIT requires 2 arguments")); }
                let str_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let delim_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                if let (Some(strs), Some(delims)) = (str_arr.as_any().downcast_ref::<StringArray>(), delim_arr.as_any().downcast_ref::<StringArray>()) {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| if strs.is_null(i) || delims.is_null(i) { None } else { Some(strs.value(i).split(delims.value(i)).collect::<Vec<_>>().join("\x00")) }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else { Err(err_data("SPLIT requires string arguments")) }
            }
            // ============== Hive Math Functions ==============
            "POWER" | "POW" => {
                if args.len() != 2 { return Err(err_input( "POWER requires 2 arguments")); }
                let b_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let e_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                let result: Vec<Option<f64>> = (0..batch.num_rows()).map(|i| {
                    let b = b_arr.as_any().downcast_ref::<Int64Array>().filter(|a| !a.is_null(i)).map(|a| a.value(i) as f64).or_else(|| b_arr.as_any().downcast_ref::<Float64Array>().filter(|a| !a.is_null(i)).map(|a| a.value(i)))?;
                    let e = e_arr.as_any().downcast_ref::<Int64Array>().filter(|a| !a.is_null(i)).map(|a| a.value(i) as f64).or_else(|| e_arr.as_any().downcast_ref::<Float64Array>().filter(|a| !a.is_null(i)).map(|a| a.value(i)))?;
                    Some(b.powf(e))
                }).collect();
                Ok(Arc::new(Float64Array::from(result)))
            }
            "EXP" => { Self::unary_float_fn(batch, &args[0], |x| x.exp()) }
            "LN" => { Self::unary_float_fn(batch, &args[0], |x| if x > 0.0 { x.ln() } else { f64::NAN }) }
            "LOG" => {
                if args.len() == 1 { Self::unary_float_fn(batch, &args[0], |x| if x > 0.0 { x.ln() } else { f64::NAN }) }
                else if args.len() == 2 {
                    let b_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                    let x_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                    let result: Vec<Option<f64>> = (0..batch.num_rows()).map(|i| {
                        let b = b_arr.as_any().downcast_ref::<Int64Array>().filter(|a| !a.is_null(i)).map(|a| a.value(i) as f64).or_else(|| b_arr.as_any().downcast_ref::<Float64Array>().filter(|a| !a.is_null(i)).map(|a| a.value(i)))?;
                        let x = x_arr.as_any().downcast_ref::<Int64Array>().filter(|a| !a.is_null(i)).map(|a| a.value(i) as f64).or_else(|| x_arr.as_any().downcast_ref::<Float64Array>().filter(|a| !a.is_null(i)).map(|a| a.value(i)))?;
                        if b <= 0.0 || b == 1.0 || x <= 0.0 { None } else { Some(x.log(b)) }
                    }).collect();
                    Ok(Arc::new(Float64Array::from(result)))
                } else { Err(err_input( "LOG requires 1 or 2 arguments")) }
            }
            "LOG10" => { Self::unary_float_fn(batch, &args[0], |x| if x > 0.0 { x.log10() } else { f64::NAN }) }
            "LOG2" => { Self::unary_float_fn(batch, &args[0], |x| if x > 0.0 { x.log2() } else { f64::NAN }) }
            "SIN" => { Self::unary_float_fn(batch, &args[0], |x| x.sin()) }
            "COS" => { Self::unary_float_fn(batch, &args[0], |x| x.cos()) }
            "TAN" => { Self::unary_float_fn(batch, &args[0], |x| x.tan()) }
            "ASIN" => { Self::unary_float_fn(batch, &args[0], |x| x.asin()) }
            "ACOS" => { Self::unary_float_fn(batch, &args[0], |x| x.acos()) }
            "ATAN" => { Self::unary_float_fn(batch, &args[0], |x| x.atan()) }
            "SIGN" => {
                if args.len() != 1 { return Err(err_input("SIGN requires 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let result: Vec<Option<i64>> = (0..batch.num_rows()).map(|i| {
                    arr.as_any().downcast_ref::<Int64Array>().filter(|a| !a.is_null(i)).map(|a| a.value(i).signum())
                    .or_else(|| arr.as_any().downcast_ref::<Float64Array>().filter(|a| !a.is_null(i)).map(|a| if a.value(i) > 0.0 { 1 } else if a.value(i) < 0.0 { -1 } else { 0 }))
                }).collect();
                Ok(Arc::new(Int64Array::from(result)))
            }
            "GREATEST" | "LEAST" => {
                let is_greatest = upper == "GREATEST";
                if args.is_empty() { return Err(err_input(format!("{} requires at least 1 argument", upper))); }
                let arrays: Vec<ArrayRef> = args.iter().map(|a| Self::evaluate_expr_to_array(batch, a)).collect::<io::Result<_>>()?;
                let result: Vec<Option<f64>> = (0..batch.num_rows()).map(|i| {
                    let mut agg: Option<f64> = None;
                    for arr in &arrays {
                        let v = arr.as_any().downcast_ref::<Int64Array>().filter(|a| !a.is_null(i)).map(|a| a.value(i) as f64).or_else(|| arr.as_any().downcast_ref::<Float64Array>().filter(|a| !a.is_null(i)).map(|a| a.value(i)));
                        if let Some(vv) = v { agg = Some(agg.map_or(vv, |m| if is_greatest { m.max(vv) } else { m.min(vv) })); }
                    }
                    agg
                }).collect();
                Ok(Arc::new(Float64Array::from(result)))
            }
            "TRUNCATE" | "TRUNC" => {
                if args.is_empty() || args.len() > 2 { return Err(err_input( "TRUNCATE requires 1-2 arguments")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let dec = if args.len() == 2 { Self::evaluate_expr_to_array(batch, &args[1])?.as_any().downcast_ref::<Int64Array>().map_or(0, |a| if a.len() > 0 && !a.is_null(0) { a.value(0) as i32 } else { 0 }) } else { 0 };
                if let Some(fa) = arr.as_any().downcast_ref::<Float64Array>() {
                    let f = 10f64.powi(dec);
                    let result: Vec<Option<f64>> = (0..batch.num_rows()).map(|i| if fa.is_null(i) { None } else { Some((fa.value(i) * f).trunc() / f) }).collect();
                    Ok(Arc::new(Float64Array::from(result)))
                } else if let Some(ia) = arr.as_any().downcast_ref::<Int64Array>() { Ok(Arc::new(ia.clone())) }
                else { Err(err_data( "TRUNCATE requires numeric")) }
            }
            "PI" => { Ok(Arc::new(Float64Array::from(vec![Some(std::f64::consts::PI); batch.num_rows()]))) }
            "E" => { Ok(Arc::new(Float64Array::from(vec![Some(std::f64::consts::E); batch.num_rows()]))) }
            // ============== Hive Date Functions ==============
            "YEAR" => { Self::extract_date_part(batch, &args[0], |s| s.get(0..4).and_then(|p| p.parse().ok())) }
            "MONTH" => { Self::extract_date_part(batch, &args[0], |s| s.get(5..7).and_then(|p| p.parse().ok())) }
            "DAY" | "DAYOFMONTH" => { Self::extract_date_part(batch, &args[0], |s| s.get(8..10).and_then(|p| p.parse().ok())) }
            "HOUR" => { Self::extract_date_part(batch, &args[0], |s| s.get(11..13).and_then(|p| p.parse().ok())) }
            "MINUTE" => { Self::extract_date_part(batch, &args[0], |s| s.get(14..16).and_then(|p| p.parse().ok())) }
            "SECOND" => { Self::extract_date_part(batch, &args[0], |s| s.get(17..19).and_then(|p| p.parse().ok())) }
            "CURRENT_DATE" => {
                let ds = chrono::Utc::now().format("%Y-%m-%d").to_string();
                Ok(Arc::new(StringArray::from(vec![Some(ds.as_str()); batch.num_rows()])))
            }
            "DATE_ADD" | "DATE_SUB" => {
                if args.len() != 2 { return Err(err_input( "DATE_ADD/DATE_SUB requires 2 arguments")); }
                let date_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let days_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                let is_sub = upper == "DATE_SUB";
                if let (Some(dates), Some(days)) = (date_arr.as_any().downcast_ref::<StringArray>(), days_arr.as_any().downcast_ref::<Int64Array>()) {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| {
                        if dates.is_null(i) || days.is_null(i) { return None; }
                        let ds = dates.value(i); let d = days.value(i);
                        chrono::NaiveDate::parse_from_str(&ds[0..10.min(ds.len())], "%Y-%m-%d").ok().map(|dt| (dt + chrono::Duration::days(if is_sub { -d } else { d })).format("%Y-%m-%d").to_string())
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else { Err(err_data( "DATE_ADD type error")) }
            }
            "DATEDIFF" => {
                if args.len() != 2 { return Err(err_input( "DATEDIFF requires 2 arguments")); }
                let d1_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let d2_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                if let (Some(d1s), Some(d2s)) = (d1_arr.as_any().downcast_ref::<StringArray>(), d2_arr.as_any().downcast_ref::<StringArray>()) {
                    let result: Vec<Option<i64>> = (0..batch.num_rows()).map(|i| {
                        if d1s.is_null(i) || d2s.is_null(i) { return None; }
                        let s1 = d1s.value(i); let s2 = d2s.value(i);
                        let dt1 = chrono::NaiveDate::parse_from_str(&s1[0..10.min(s1.len())], "%Y-%m-%d").ok()?;
                        let dt2 = chrono::NaiveDate::parse_from_str(&s2[0..10.min(s2.len())], "%Y-%m-%d").ok()?;
                        Some((dt1 - dt2).num_days())
                    }).collect();
                    Ok(Arc::new(Int64Array::from(result)))
                } else { Err(err_data( "DATEDIFF type error")) }
            }
            "DATE_FORMAT" => {
                if args.len() != 2 { return Err(err_input( "DATE_FORMAT requires 2 arguments")); }
                let date_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let fmt_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                if let (Some(dates), Some(fmts)) = (date_arr.as_any().downcast_ref::<StringArray>(), fmt_arr.as_any().downcast_ref::<StringArray>()) {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| {
                        if dates.is_null(i) || fmts.is_null(i) { return None; }
                        let ds = dates.value(i); let fmt = fmts.value(i);
                        chrono::NaiveDate::parse_from_str(&ds[0..10.min(ds.len())], "%Y-%m-%d").ok().map(|dt| dt.format(fmt).to_string())
                    }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else { Err(err_data( "DATE_FORMAT type error")) }
            }
            "TO_DATE" => {
                if args.len() != 1 { return Err(err_input("TO_DATE requires 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                map_string_unary(&arr, batch.num_rows(), |s| s[0..10.min(s.len())].to_string(), "TO_DATE")
            }
            "UNIX_TIMESTAMP" => {
                if args.is_empty() {
                    use std::time::{SystemTime, UNIX_EPOCH};
                    let ts = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
                    Ok(Arc::new(Int64Array::from(vec![Some(ts); batch.num_rows()])))
                } else {
                    let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                    if let Some(sa) = arr.as_any().downcast_ref::<StringArray>() {
                        let result: Vec<Option<i64>> = (0..batch.num_rows()).map(|i| {
                            if sa.is_null(i) { return None; }
                            let s = sa.value(i);
                            chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").ok().or_else(|| chrono::NaiveDate::parse_from_str(&s[0..10.min(s.len())], "%Y-%m-%d").ok().map(|d| d.and_hms_opt(0,0,0).unwrap())).map(|dt| dt.and_utc().timestamp())
                        }).collect();
                        Ok(Arc::new(Int64Array::from(result)))
                    } else { Err(err_data( "UNIX_TIMESTAMP type error")) }
                }
            }
            "FROM_UNIXTIME" => {
                if args.len() != 1 { return Err(err_input("FROM_UNIXTIME requires 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                map_int_to_string(&arr, batch.num_rows(), |v| chrono::DateTime::from_timestamp(v, 0).map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string()), "FROM_UNIXTIME")
            }
            // ============== Hive Conditional Functions ==============
            "IF" => {
                if args.len() != 3 { return Err(err_input("IF requires 3 arguments")); }
                let (cond_arr, then_arr, else_arr) = (Self::evaluate_expr_to_array(batch, &args[0])?, Self::evaluate_expr_to_array(batch, &args[1])?, Self::evaluate_expr_to_array(batch, &args[2])?);
                let get_cond = |i: usize| cond_arr.as_any().downcast_ref::<BooleanArray>().map_or(false, |b| !b.is_null(i) && b.value(i));
                if then_arr.as_any().downcast_ref::<StringArray>().is_some() || else_arr.as_any().downcast_ref::<StringArray>().is_some() {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| { let src = if get_cond(i) { &then_arr } else { &else_arr }; src.as_any().downcast_ref::<StringArray>().and_then(|sa| if sa.is_null(i) { None } else { Some(sa.value(i).to_string()) }) }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else {
                    let result: Vec<Option<i64>> = (0..batch.num_rows()).map(|i| { let src = if get_cond(i) { &then_arr } else { &else_arr }; src.as_any().downcast_ref::<Int64Array>().and_then(|ia| if ia.is_null(i) { None } else { Some(ia.value(i)) }) }).collect();
                    Ok(Arc::new(Int64Array::from(result)))
                }
            }
            "NVL2" => {
                if args.len() != 3 { return Err(err_input("NVL2 requires 3 arguments")); }
                let (chk, nn, nl) = (Self::evaluate_expr_to_array(batch, &args[0])?, Self::evaluate_expr_to_array(batch, &args[1])?, Self::evaluate_expr_to_array(batch, &args[2])?);
                let is_null = |i: usize| chk.is_null(i) || chk.as_any().downcast_ref::<StringArray>().map_or(false, |sa| sa.is_null(i));
                if nn.as_any().downcast_ref::<StringArray>().is_some() || nl.as_any().downcast_ref::<StringArray>().is_some() {
                    let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| { let src = if is_null(i) { &nl } else { &nn }; src.as_any().downcast_ref::<StringArray>().and_then(|sa| if sa.is_null(i) { None } else { Some(sa.value(i).to_string()) }) }).collect();
                    Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
                } else {
                    let result: Vec<Option<i64>> = (0..batch.num_rows()).map(|i| { let src = if chk.is_null(i) { &nl } else { &nn }; src.as_any().downcast_ref::<Int64Array>().and_then(|ia| if ia.is_null(i) { None } else { Some(ia.value(i)) }) }).collect();
                    Ok(Arc::new(Int64Array::from(result)))
                }
            }
            "DECODE" => {
                if args.len() < 3 { return Err(err_input("DECODE requires 3+ arguments")); }
                let expr_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let (has_def, pairs) = ((args.len() - 1) % 2 == 1, (args.len() - 1 - if (args.len() - 1) % 2 == 1 { 1 } else { 0 }) / 2);
                let (search_arrs, result_arrs): (Vec<_>, Vec<_>) = (0..pairs).map(|p| (Self::evaluate_expr_to_array(batch, &args[1 + p * 2]), Self::evaluate_expr_to_array(batch, &args[2 + p * 2]))).map(|(s, r)| (s.ok(), r.ok())).filter(|(s, r)| s.is_some() && r.is_some()).map(|(s, r)| (s.unwrap(), r.unwrap())).unzip();
                let def = if has_def { Self::evaluate_expr_to_array(batch, args.last().unwrap()).ok() } else { None };
                let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| {
                    for (sa, ra) in search_arrs.iter().zip(result_arrs.iter()) {
                        let m = expr_arr.as_any().downcast_ref::<Int64Array>().and_then(|ea| sa.as_any().downcast_ref::<Int64Array>().map(|saa| !ea.is_null(i) && !saa.is_null(i) && ea.value(i) == saa.value(i))).or_else(|| expr_arr.as_any().downcast_ref::<StringArray>().and_then(|ea| sa.as_any().downcast_ref::<StringArray>().map(|saa| !ea.is_null(i) && !saa.is_null(i) && ea.value(i) == saa.value(i)))).unwrap_or(false);
                        if m { return ra.as_any().downcast_ref::<StringArray>().and_then(|r| if r.is_null(i) { None } else { Some(r.value(i).to_string()) }); }
                    }
                    def.as_ref().and_then(|d| d.as_any().downcast_ref::<StringArray>().and_then(|r| if r.is_null(i) { None } else { Some(r.value(i).to_string()) }))
                }).collect();
                Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
            }
            "GROUP_CONCAT" | "LISTAGG" => {
                if args.is_empty() { return Err(err_input("GROUP_CONCAT requires 1+ argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                if let Some(sa) = arr.as_any().downcast_ref::<StringArray>() {
                    let sep = if args.len() > 1 { Self::evaluate_expr_to_array(batch, &args[1])?.as_any().downcast_ref::<StringArray>().map_or(",".to_string(), |s| if s.len() > 0 && !s.is_null(0) { s.value(0).to_string() } else { ",".to_string() }) } else { ",".to_string() };
                    let joined = (0..batch.num_rows()).filter(|&i| !sa.is_null(i)).map(|i| sa.value(i)).collect::<Vec<_>>().join(&sep);
                    Ok(Arc::new(StringArray::from(vec![Some(joined.as_str()); batch.num_rows()])))
                } else { Err(err_data("GROUP_CONCAT requires string")) }
            }
            // ===== JSON Functions =====
            "JSON_EXTRACT" | "JSON_VALUE" => {
                if args.len() != 2 { return Err(err_input("JSON_EXTRACT requires 2 arguments (json, path)")); }
                let json_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let path_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                let n = batch.num_rows();
                let mut result: Vec<Option<String>> = Vec::with_capacity(n);
                let json_sa = json_arr.as_any().downcast_ref::<StringArray>();
                let path_sa = path_arr.as_any().downcast_ref::<StringArray>();
                for i in 0..n {
                    if json_arr.is_null(i) || path_arr.is_null(i) {
                        result.push(None);
                        continue;
                    }
                    let json_str = json_sa.map(|a| a.value(i)).unwrap_or("");
                    let path_str = path_sa.map(|a| a.value(i)).unwrap_or("");
                    result.push(Self::json_extract_path(json_str, path_str));
                }
                Ok(Arc::new(StringArray::from(result)))
            }
            "JSON_TYPE" => {
                if args.len() != 1 { return Err(err_input("JSON_TYPE requires 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let n = batch.num_rows();
                let mut result: Vec<Option<String>> = Vec::with_capacity(n);
                let sa = arr.as_any().downcast_ref::<StringArray>();
                for i in 0..n {
                    if arr.is_null(i) {
                        result.push(Some("NULL".to_string()));
                        continue;
                    }
                    let s = sa.map(|a| a.value(i)).unwrap_or("");
                    let trimmed = s.trim();
                    let jtype = if trimmed.starts_with('{') { "OBJECT" }
                        else if trimmed.starts_with('[') { "ARRAY" }
                        else if trimmed == "true" || trimmed == "false" { "BOOLEAN" }
                        else if trimmed == "null" { "NULL" }
                        else if trimmed.starts_with('"') { "STRING" }
                        else if trimmed.parse::<f64>().is_ok() { if trimmed.contains('.') { "REAL" } else { "INTEGER" } }
                        else { "NULL" };
                    result.push(Some(jtype.to_string()));
                }
                Ok(Arc::new(StringArray::from(result)))
            }
            "JSON_VALID" => {
                if args.len() != 1 { return Err(err_input("JSON_VALID requires 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let n = batch.num_rows();
                let mut result: Vec<i64> = Vec::with_capacity(n);
                let sa = arr.as_any().downcast_ref::<StringArray>();
                for i in 0..n {
                    if arr.is_null(i) { result.push(0); continue; }
                    let s = sa.map(|a| a.value(i)).unwrap_or("");
                    result.push(if Self::is_valid_json(s) { 1 } else { 0 });
                }
                Ok(Arc::new(Int64Array::from(result)))
            }
            "JSON_ARRAY_LENGTH" => {
                if args.len() < 1 { return Err(err_input("JSON_ARRAY_LENGTH requires at least 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let n = batch.num_rows();
                let mut result: Vec<Option<i64>> = Vec::with_capacity(n);
                let sa = arr.as_any().downcast_ref::<StringArray>();
                for i in 0..n {
                    if arr.is_null(i) { result.push(None); continue; }
                    let s = sa.map(|a| a.value(i)).unwrap_or("");
                    // If path arg provided, extract first
                    let json_str = if args.len() > 1 {
                        let path_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                        let path_sa = path_arr.as_any().downcast_ref::<StringArray>();
                        let p = path_sa.map(|a| a.value(i)).unwrap_or("");
                        Self::json_extract_path(s, p).unwrap_or_default()
                    } else {
                        s.to_string()
                    };
                    let trimmed = json_str.trim();
                    if trimmed.starts_with('[') {
                        // Count top-level array elements
                        result.push(Some(Self::json_count_array_elements(trimmed)));
                    } else {
                        result.push(Some(0));
                    }
                }
                Ok(Arc::new(Int64Array::from(result)))
            }
            "JSON_OBJECT" => {
                // JSON_OBJECT('key1', val1, 'key2', val2, ...)
                if args.len() % 2 != 0 { return Err(err_input("JSON_OBJECT requires even number of arguments (key-value pairs)")); }
                let n = batch.num_rows();
                let mut result: Vec<String> = Vec::with_capacity(n);
                let mut key_arrs = Vec::new();
                let mut val_arrs = Vec::new();
                for i in (0..args.len()).step_by(2) {
                    key_arrs.push(Self::evaluate_expr_to_array(batch, &args[i])?);
                    val_arrs.push(Self::evaluate_expr_to_array(batch, &args[i + 1])?);
                }
                for row in 0..n {
                    let mut obj = String::from("{");
                    for (pi, (ka, va)) in key_arrs.iter().zip(val_arrs.iter()).enumerate() {
                        if pi > 0 { obj.push(','); }
                        let key = if let Some(sa) = ka.as_any().downcast_ref::<StringArray>() {
                            if sa.is_null(row) { "null".to_string() } else { format!("\"{}\"", sa.value(row)) }
                        } else { "null".to_string() };
                        let val = Self::json_value_from_array(va, row);
                        obj.push_str(&format!("{}:{}", key, val));
                    }
                    obj.push('}');
                    result.push(obj);
                }
                Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_str()).collect::<Vec<_>>())))
            }
            "JSON_ARRAY" => {
                let n = batch.num_rows();
                let mut result: Vec<String> = Vec::with_capacity(n);
                let mut elem_arrs = Vec::new();
                for arg in args {
                    elem_arrs.push(Self::evaluate_expr_to_array(batch, arg)?);
                }
                for row in 0..n {
                    let mut arr_str = String::from("[");
                    for (ei, ea) in elem_arrs.iter().enumerate() {
                        if ei > 0 { arr_str.push(','); }
                        arr_str.push_str(&Self::json_value_from_array(ea, row));
                    }
                    arr_str.push(']');
                    result.push(arr_str);
                }
                Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_str()).collect::<Vec<_>>())))
            }
            "JSON_SET" | "JSON_INSERT" | "JSON_REPLACE" => {
                if args.len() != 3 {
                    return Err(err_input(format!(
                        "{} requires 3 arguments (json, path, value)",
                        name
                    )));
                }
                let json_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let path_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                let value_arr = Self::evaluate_expr_to_array(batch, &args[2])?;
                let json_sa = json_arr.as_any().downcast_ref::<StringArray>();
                let path_sa = path_arr.as_any().downcast_ref::<StringArray>();
                let mut result: Vec<Option<String>> = Vec::with_capacity(batch.num_rows());
                for row in 0..batch.num_rows() {
                    if json_arr.is_null(row) || path_arr.is_null(row) {
                        result.push(None);
                        continue;
                    }
                    let Some(json_str) = json_sa.map(|a| a.value(row)) else {
                        result.push(None);
                        continue;
                    };
                    let Some(path_str) = path_sa.map(|a| a.value(row)) else {
                        result.push(None);
                        continue;
                    };
                    let value = Self::arrow_value_at_col(&value_arr, row).to_json_value();
                    let updated = match name {
                        "JSON_SET" => Self::json_set_path(json_str, path_str, value),
                        "JSON_INSERT" => Self::json_insert_path(json_str, path_str, value),
                        _ => Self::json_replace_path(json_str, path_str, value),
                    };
                    result.push(updated);
                }
                Ok(Arc::new(StringArray::from(result)))
            }
            "JSON_REMOVE" => {
                if args.len() != 2 {
                    return Err(err_input("JSON_REMOVE requires 2 arguments (json, path)"));
                }
                let json_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                let path_arr = Self::evaluate_expr_to_array(batch, &args[1])?;
                let json_sa = json_arr.as_any().downcast_ref::<StringArray>();
                let path_sa = path_arr.as_any().downcast_ref::<StringArray>();
                let mut result: Vec<Option<String>> = Vec::with_capacity(batch.num_rows());
                for row in 0..batch.num_rows() {
                    if json_arr.is_null(row) || path_arr.is_null(row) {
                        result.push(None);
                        continue;
                    }
                    let Some(json_str) = json_sa.map(|a| a.value(row)) else {
                        result.push(None);
                        continue;
                    };
                    let Some(path_str) = path_sa.map(|a| a.value(row)) else {
                        result.push(None);
                        continue;
                    };
                    result.push(Self::json_remove_path(json_str, path_str));
                }
                Ok(Arc::new(StringArray::from(result)))
            }
            // ===== Vector Distance Functions =====
            // Naming mirrors DuckDB's array functions for drop-in compatibility.
            "ARRAY_DISTANCE"            | "L2_DISTANCE"                   |
            "ARRAY_L2_DISTANCE"         | "EUCLIDEAN_DISTANCE"            => {
                Self::evaluate_vector_distance(batch, args, crate::query::vector_ops::DistanceMetric::L2)
            }
            "L2_SQUARED_DISTANCE"       | "SQUARED_L2"                    => {
                Self::evaluate_vector_distance(batch, args, crate::query::vector_ops::DistanceMetric::L2Squared)
            }
            "COSINE_DISTANCE"           | "ARRAY_COSINE_DISTANCE"         => {
                Self::evaluate_vector_distance(batch, args, crate::query::vector_ops::DistanceMetric::CosineDistance)
            }
            "COSINE_SIMILARITY"         | "ARRAY_COSINE_SIMILARITY"       => {
                Self::evaluate_vector_distance(batch, args, crate::query::vector_ops::DistanceMetric::CosineSimilarity)
            }
            "INNER_PRODUCT"             | "DOT_PRODUCT"                   |
            "ARRAY_INNER_PRODUCT"       => {
                Self::evaluate_vector_distance(batch, args, crate::query::vector_ops::DistanceMetric::InnerProduct)
            }
            "NEGATIVE_INNER_PRODUCT"    | "ARRAY_NEGATIVE_INNER_PRODUCT"  => {
                Self::evaluate_vector_distance(batch, args, crate::query::vector_ops::DistanceMetric::NegInnerProduct)
            }
            "L1_DISTANCE"               | "ARRAY_L1_DISTANCE"             |
            "MANHATTAN_DISTANCE"        => {
                Self::evaluate_vector_distance(batch, args, crate::query::vector_ops::DistanceMetric::L1)
            }
            "LINF_DISTANCE"             | "ARRAY_LINF_DISTANCE"           |
            "CHEBYSHEV_DISTANCE"        => {
                Self::evaluate_vector_distance(batch, args, crate::query::vector_ops::DistanceMetric::LInf)
            }
            // ── Utility ──────────────────────────────────────────────────────────
            "VECTOR_DIM" | "ARRAY_LENGTH" => {
                if args.len() != 1 { return Err(err_input("VECTOR_DIM requires 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                use arrow::array::BinaryArray;
                if let Some(ba) = arr.as_any().downcast_ref::<BinaryArray>() {
                    let result: Vec<Option<i64>> = (0..ba.len())
                        .map(|i| if ba.is_null(i) { None } else { Some((ba.value(i).len() / 4) as i64) })
                        .collect();
                    Ok(Arc::new(Int64Array::from(result)) as ArrayRef)
                } else if let Some(fsl) = arr.as_any().downcast_ref::<arrow::array::FixedSizeListArray>() {
                    let dim = fsl.value_length() as i64;
                    let result: Vec<Option<i64>> = (0..fsl.len())
                        .map(|i| if fsl.is_null(i) { None } else { Some(dim) })
                        .collect();
                    Ok(Arc::new(Int64Array::from(result)) as ArrayRef)
                } else {
                    Err(err_input("VECTOR_DIM requires a binary or fixed-size list vector column"))
                }
            }
            "VECTOR_NORM" | "L2_NORM" => {
                if args.len() != 1 { return Err(err_input("VECTOR_NORM requires 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                use arrow::array::BinaryArray;
                if let Some(ba) = arr.as_any().downcast_ref::<BinaryArray>() {
                    let result: Vec<Option<f64>> = (0..ba.len())
                        .map(|i| {
                            if ba.is_null(i) { return None; }
                            let bytes = ba.value(i);
                            if bytes.len() % 4 != 0 { return None; }
                            let vec = unsafe {
                                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
                            };
                            Some(vec.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt())
                        })
                        .collect();
                    Ok(Arc::new(Float64Array::from(result)) as ArrayRef)
                } else if let Some(fsl) = arr.as_any().downcast_ref::<arrow::array::FixedSizeListArray>() {
                    let values = fsl.values().as_any().downcast_ref::<arrow::array::Float32Array>()
                        .ok_or_else(|| err_input("VECTOR_NORM: FixedSizeList must contain Float32"))?;
                    let dim = fsl.value_length() as usize;
                    let result: Vec<Option<f64>> = (0..fsl.len())
                        .map(|i| {
                            if fsl.is_null(i) { return None; }
                            let offset = i * dim;
                            let sum_sq: f64 = (0..dim).map(|j| { let v = values.value(offset + j) as f64; v * v }).sum();
                            Some(sum_sq.sqrt())
                        })
                        .collect();
                    Ok(Arc::new(Float64Array::from(result)) as ArrayRef)
                } else {
                    Err(err_input("VECTOR_NORM requires a binary or fixed-size list vector column"))
                }
            }
            "VECTOR_TO_STRING" | "ARRAY_TO_STRING" => {
                if args.len() != 1 { return Err(err_input("VECTOR_TO_STRING requires 1 argument")); }
                let arr = Self::evaluate_expr_to_array(batch, &args[0])?;
                use arrow::array::BinaryArray;
                if let Some(ba) = arr.as_any().downcast_ref::<BinaryArray>() {
                    let result: Vec<Option<String>> = (0..ba.len())
                        .map(|i| {
                            if ba.is_null(i) { return None; }
                            let bytes = ba.value(i);
                            if bytes.len() % 4 != 0 { return None; }
                            let floats = crate::query::vector_ops::decode_f32_vec(bytes)?;
                            Some(format!("[{}]", floats.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(",")))
                        })
                        .collect();
                    Ok(Arc::new(StringArray::from(result)) as ArrayRef)
                } else if let Some(fsl) = arr.as_any().downcast_ref::<arrow::array::FixedSizeListArray>() {
                    let values = fsl.values().as_any().downcast_ref::<arrow::array::Float32Array>()
                        .ok_or_else(|| err_input("VECTOR_TO_STRING: FixedSizeList must contain Float32"))?;
                    let dim = fsl.value_length() as usize;
                    let result: Vec<Option<String>> = (0..fsl.len())
                        .map(|i| {
                            if fsl.is_null(i) { return None; }
                            let offset = i * dim;
                            let floats: Vec<String> = (0..dim).map(|j| values.value(offset + j).to_string()).collect();
                            Some(format!("[{}]", floats.join(",")))
                        })
                        .collect();
                    Ok(Arc::new(StringArray::from(result)) as ArrayRef)
                } else {
                    Err(err_input("VECTOR_TO_STRING requires a binary or fixed-size list vector column"))
                }
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("Unsupported function: {}", name),
            )),
        }
    }

    /// Shared implementation for all pairwise vector distance functions.
    fn evaluate_vector_distance(
        batch: &RecordBatch,
        args: &[SqlExpr],
        metric: crate::query::vector_ops::DistanceMetric,
    ) -> io::Result<ArrayRef> {
        if args.len() != 2 {
            return Err(err_input(format!(
                "Vector distance function requires exactly 2 arguments, got {}", args.len()
            )));
        }
        let col_arr = Self::evaluate_expr_to_array(batch, &args[0])?;
        let query_vec = Self::extract_vector_literal(&args[1], batch)?;
        crate::query::vector_ops::batch_distance(&*col_arr, &query_vec, metric)
    }

    /// Extract a query vector (Vec<f32>) from a SQL expression.
    /// Handles: ArrayLiteral, StringLit "[1.0,…]", BinaryArray, Float32Array, Float64Array.
    fn extract_vector_literal(expr: &SqlExpr, batch: &RecordBatch) -> io::Result<Vec<f32>> {
        match expr {
            SqlExpr::ArrayLiteral(values) => {
                Ok(values.iter().map(|&f| f as f32).collect())
            }
            SqlExpr::Literal(crate::data::Value::String(s)) => {
                // "[1.0, 2.0, 3.0]" string literal
                let s = s.trim();
                if s.starts_with('[') && s.ends_with(']') {
                    let inner = &s[1..s.len()-1];
                    inner.split(',')
                        .map(|t| t.trim().parse::<f32>()
                            .map_err(|_| err_input(format!("Invalid float in vector literal: {}", t))))
                        .collect()
                } else {
                    Err(err_input("Vector literal must be a JSON array string like '[1.0,2.0]'"))
                }
            }
            _ => {
                let arr = Self::evaluate_expr_to_array(batch, expr)?;
                crate::query::vector_ops::extract_query_vector(&*arr)
                    .map_err(|e| err_input(e.to_string()))
            }
        }
    }

    // ========== JSON Helper Methods ==========

    /// Extract a value from a JSON string using a simplified JSON path (e.g., "$.key", "$.arr[0]", "$.a.b")
    fn json_extract_path(json_str: &str, path: &str) -> Option<String> {
        let trimmed = json_str.trim();
        // Parse path: strip leading $
        let path_str = path.trim().trim_start_matches('$').trim_start_matches('.');
        if path_str.is_empty() {
            return Some(trimmed.to_string());
        }

        // Split path into segments
        let mut current = trimmed;
        let mut owned = String::new();
        for segment in path_str.split('.') {
            let (key, array_idx) = if let Some(bracket_pos) = segment.find('[') {
                let key_part = &segment[..bracket_pos];
                let idx_str = segment[bracket_pos + 1..].trim_end_matches(']');
                (key_part, idx_str.parse::<usize>().ok())
            } else {
                (segment, None)
            };

            if !key.is_empty() {
                // Navigate into object key
                if let Some(val) = Self::json_get_key(current, key) {
                    owned = val;
                    current = &owned;
                } else {
                    return None;
                }
            }

            if let Some(idx) = array_idx {
                // Navigate into array index
                if let Some(val) = Self::json_get_index(current, idx) {
                    owned = val;
                    current = &owned;
                } else {
                    return None;
                }
            }
        }
        Some(current.to_string())
    }

    fn json_set_path(
        json_str: &str,
        path: &str,
        value: serde_json::Value,
    ) -> Option<String> {
        Self::json_mutate_path(json_str, path, Some(value), JsonMutationMode::Set)
    }

    fn json_insert_path(
        json_str: &str,
        path: &str,
        value: serde_json::Value,
    ) -> Option<String> {
        Self::json_mutate_path(json_str, path, Some(value), JsonMutationMode::Insert)
    }

    fn json_replace_path(
        json_str: &str,
        path: &str,
        value: serde_json::Value,
    ) -> Option<String> {
        Self::json_mutate_path(json_str, path, Some(value), JsonMutationMode::Replace)
    }

    fn json_remove_path(json_str: &str, path: &str) -> Option<String> {
        Self::json_mutate_path(json_str, path, None, JsonMutationMode::Remove)
    }

    fn json_mutate_path(
        json_str: &str,
        path: &str,
        value: Option<serde_json::Value>,
        mode: JsonMutationMode,
    ) -> Option<String> {
        let mut root: serde_json::Value = serde_json::from_str(json_str).ok()?;
        let segments = Self::json_parse_path_segments(path)?;
        if segments.is_empty() {
            return Some(root.to_string());
        }
        let changed = Self::json_apply_mutation(&mut root, &segments, value, mode);
        if changed {
            Some(root.to_string())
        } else {
            Some(json_str.trim().to_string())
        }
    }

    fn json_parse_path_segments(path: &str) -> Option<Vec<JsonPathSegment>> {
        let mut chars = path.trim().chars().peekable();
        if chars.peek() == Some(&'$') {
            chars.next();
        }

        let mut segments = Vec::new();
        while let Some(ch) = chars.peek().copied() {
            match ch {
                '.' => {
                    chars.next();
                    let mut key = String::new();
                    while let Some(c) = chars.peek().copied() {
                        if c == '.' || c == '[' {
                            break;
                        }
                        key.push(c);
                        chars.next();
                    }
                    if key.is_empty() {
                        return None;
                    }
                    segments.push(JsonPathSegment::Key(key));
                }
                '[' => {
                    chars.next();
                    let mut idx = String::new();
                    while let Some(c) = chars.peek().copied() {
                        if c == ']' {
                            break;
                        }
                        idx.push(c);
                        chars.next();
                    }
                    if chars.next() != Some(']') {
                        return None;
                    }
                    segments.push(JsonPathSegment::Index(idx.parse().ok()?));
                }
                _ => {
                    let mut key = String::new();
                    while let Some(c) = chars.peek().copied() {
                        if c == '.' || c == '[' {
                            break;
                        }
                        key.push(c);
                        chars.next();
                    }
                    if key.is_empty() {
                        return None;
                    }
                    segments.push(JsonPathSegment::Key(key));
                }
            }
        }
        Some(segments)
    }

    fn json_apply_mutation(
        current: &mut serde_json::Value,
        segments: &[JsonPathSegment],
        value: Option<serde_json::Value>,
        mode: JsonMutationMode,
    ) -> bool {
        if segments.is_empty() {
            return false;
        }

        let is_last = segments.len() == 1;
        match &segments[0] {
            JsonPathSegment::Key(key) => {
                if !current.is_object() {
                    if mode == JsonMutationMode::Replace || mode == JsonMutationMode::Remove {
                        return false;
                    }
                    *current = serde_json::Value::Object(serde_json::Map::new());
                }
                let Some(map) = current.as_object_mut() else {
                    return false;
                };
                if is_last {
                    return match mode {
                        JsonMutationMode::Set => {
                            map.insert(key.clone(), value.unwrap_or(serde_json::Value::Null));
                            true
                        }
                        JsonMutationMode::Insert => {
                            if map.contains_key(key) {
                                false
                            } else {
                                map.insert(key.clone(), value.unwrap_or(serde_json::Value::Null));
                                true
                            }
                        }
                        JsonMutationMode::Replace => {
                            if let Some(slot) = map.get_mut(key) {
                                *slot = value.unwrap_or(serde_json::Value::Null);
                                true
                            } else {
                                false
                            }
                        }
                        JsonMutationMode::Remove => map.remove(key).is_some(),
                    };
                }

                if !map.contains_key(key) {
                    if mode == JsonMutationMode::Replace || mode == JsonMutationMode::Remove {
                        return false;
                    }
                    map.insert(
                        key.clone(),
                        Self::json_default_container_for(&segments[1]),
                    );
                }
                let Some(child) = map.get_mut(key) else {
                    return false;
                };
                Self::json_apply_mutation(child, &segments[1..], value, mode)
            }
            JsonPathSegment::Index(idx) => {
                if !current.is_array() {
                    if mode == JsonMutationMode::Replace || mode == JsonMutationMode::Remove {
                        return false;
                    }
                    *current = serde_json::Value::Array(Vec::new());
                }
                let Some(arr) = current.as_array_mut() else {
                    return false;
                };
                if is_last {
                    return match mode {
                        JsonMutationMode::Set => {
                            if *idx >= arr.len() {
                                arr.resize(*idx + 1, serde_json::Value::Null);
                            }
                            arr[*idx] = value.unwrap_or(serde_json::Value::Null);
                            true
                        }
                        JsonMutationMode::Insert => {
                            if *idx < arr.len() {
                                false
                            } else {
                                arr.resize(*idx, serde_json::Value::Null);
                                arr.push(value.unwrap_or(serde_json::Value::Null));
                                true
                            }
                        }
                        JsonMutationMode::Replace => {
                            if *idx < arr.len() {
                                arr[*idx] = value.unwrap_or(serde_json::Value::Null);
                                true
                            } else {
                                false
                            }
                        }
                        JsonMutationMode::Remove => {
                            if *idx < arr.len() {
                                arr.remove(*idx);
                                true
                            } else {
                                false
                            }
                        }
                    };
                }

                if *idx >= arr.len() {
                    if mode == JsonMutationMode::Replace || mode == JsonMutationMode::Remove {
                        return false;
                    }
                    arr.resize(*idx + 1, serde_json::Value::Null);
                }
                if arr[*idx].is_null() {
                    arr[*idx] = Self::json_default_container_for(&segments[1]);
                }
                Self::json_apply_mutation(&mut arr[*idx], &segments[1..], value, mode)
            }
        }
    }

    fn json_default_container_for(next: &JsonPathSegment) -> serde_json::Value {
        match next {
            JsonPathSegment::Key(_) => serde_json::Value::Object(serde_json::Map::new()),
            JsonPathSegment::Index(_) => serde_json::Value::Array(Vec::new()),
        }
    }

    /// Get a value from a JSON object by key (simple parser, no external deps)
    fn json_get_key(json: &str, key: &str) -> Option<String> {
        let trimmed = json.trim();
        if !trimmed.starts_with('{') { return None; }
        let bytes = trimmed.as_bytes();
        let mut i = 1; // skip '{'
        let len = bytes.len();

        loop {
            // Skip whitespace
            while i < len && (bytes[i] == b' ' || bytes[i] == b'\n' || bytes[i] == b'\r' || bytes[i] == b'\t') { i += 1; }
            if i >= len || bytes[i] == b'}' { return None; }
            // Parse key
            if bytes[i] != b'"' { return None; }
            i += 1;
            let key_start = i;
            while i < len && bytes[i] != b'"' { if bytes[i] == b'\\' { i += 1; } i += 1; }
            let parsed_key = &trimmed[key_start..i];
            i += 1; // skip closing "
            // Skip whitespace + colon
            while i < len && bytes[i] != b':' { i += 1; }
            i += 1; // skip ':'
            while i < len && (bytes[i] == b' ' || bytes[i] == b'\n' || bytes[i] == b'\r' || bytes[i] == b'\t') { i += 1; }
            // Parse value
            let val_start = i;
            let val_end = Self::json_skip_value(trimmed, i);
            let value = trimmed[val_start..val_end].trim();

            if parsed_key == key {
                // Unquote strings
                if value.starts_with('"') && value.ends_with('"') && value.len() >= 2 {
                    return Some(value[1..value.len()-1].to_string());
                }
                return Some(value.to_string());
            }
            i = val_end;
            while i < len && (bytes[i] == b' ' || bytes[i] == b'\n' || bytes[i] == b'\r' || bytes[i] == b'\t') { i += 1; }
            if i < len && bytes[i] == b',' { i += 1; }
        }
    }

    /// Get an element from a JSON array by index
    fn json_get_index(json: &str, idx: usize) -> Option<String> {
        let trimmed = json.trim();
        if !trimmed.starts_with('[') { return None; }
        let mut i = 1;
        let len = trimmed.len();
        let bytes = trimmed.as_bytes();
        let mut current_idx = 0;

        loop {
            while i < len && (bytes[i] == b' ' || bytes[i] == b'\n' || bytes[i] == b'\r' || bytes[i] == b'\t') { i += 1; }
            if i >= len || bytes[i] == b']' { return None; }
            let val_start = i;
            let val_end = Self::json_skip_value(trimmed, i);
            if current_idx == idx {
                let value = trimmed[val_start..val_end].trim();
                if value.starts_with('"') && value.ends_with('"') && value.len() >= 2 {
                    return Some(value[1..value.len()-1].to_string());
                }
                return Some(value.to_string());
            }
            i = val_end;
            current_idx += 1;
            while i < len && (bytes[i] == b' ' || bytes[i] == b'\n' || bytes[i] == b'\r' || bytes[i] == b'\t') { i += 1; }
            if i < len && bytes[i] == b',' { i += 1; }
        }
    }

    /// Skip a JSON value starting at position i, returns the position after the value
    fn json_skip_value(json: &str, start: usize) -> usize {
        let bytes = json.as_bytes();
        let len = bytes.len();
        if start >= len { return start; }
        match bytes[start] {
            b'"' => {
                let mut i = start + 1;
                while i < len { if bytes[i] == b'\\' { i += 2; } else if bytes[i] == b'"' { return i + 1; } else { i += 1; } }
                len
            }
            b'{' | b'[' => {
                let open = bytes[start];
                let close = if open == b'{' { b'}' } else { b']' };
                let mut depth = 1;
                let mut i = start + 1;
                let mut in_string = false;
                while i < len && depth > 0 {
                    if in_string { if bytes[i] == b'\\' { i += 1; } else if bytes[i] == b'"' { in_string = false; } }
                    else { match bytes[i] { b'"' => in_string = true, b if b == open => depth += 1, b if b == close => depth -= 1, _ => {} } }
                    i += 1;
                }
                i
            }
            _ => {
                // Number, true, false, null
                let mut i = start;
                while i < len && bytes[i] != b',' && bytes[i] != b'}' && bytes[i] != b']' && bytes[i] != b' ' && bytes[i] != b'\n' { i += 1; }
                i
            }
        }
    }

    /// Check if a string is valid JSON
    fn is_valid_json(s: &str) -> bool {
        let trimmed = s.trim();
        if trimmed.is_empty() { return false; }
        match trimmed.as_bytes()[0] {
            b'{' => trimmed.ends_with('}'),
            b'[' => trimmed.ends_with(']'),
            b'"' => trimmed.ends_with('"') && trimmed.len() >= 2,
            _ => trimmed == "true" || trimmed == "false" || trimmed == "null" || trimmed.parse::<f64>().is_ok(),
        }
    }

    /// Count top-level elements in a JSON array string
    fn json_count_array_elements(json: &str) -> i64 {
        let trimmed = json.trim();
        if !trimmed.starts_with('[') { return 0; }
        if trimmed == "[]" { return 0; }
        let mut i = 1;
        let len = trimmed.len();
        let bytes = trimmed.as_bytes();
        let mut count = 0i64;
        loop {
            while i < len && (bytes[i] == b' ' || bytes[i] == b'\n' || bytes[i] == b'\r' || bytes[i] == b'\t') { i += 1; }
            if i >= len || bytes[i] == b']' { break; }
            count += 1;
            i = Self::json_skip_value(trimmed, i);
            while i < len && (bytes[i] == b' ' || bytes[i] == b'\n' || bytes[i] == b'\r' || bytes[i] == b'\t') { i += 1; }
            if i < len && bytes[i] == b',' { i += 1; }
        }
        count
    }

    /// Convert an Arrow array value at a given row to a JSON-encoded string
    fn json_value_from_array(arr: &ArrayRef, row: usize) -> String {
        if arr.is_null(row) { return "null".to_string(); }
        if let Some(sa) = arr.as_any().downcast_ref::<StringArray>() {
            format!("\"{}\"", sa.value(row).replace('\\', "\\\\").replace('"', "\\\""))
        } else if let Some(ia) = arr.as_any().downcast_ref::<Int64Array>() {
            ia.value(row).to_string()
        } else if let Some(fa) = arr.as_any().downcast_ref::<Float64Array>() {
            fa.value(row).to_string()
        } else if let Some(ba) = arr.as_any().downcast_ref::<BooleanArray>() {
            if ba.value(row) { "true".to_string() } else { "false".to_string() }
        } else if let Some(ua) = arr.as_any().downcast_ref::<UInt64Array>() {
            ua.value(row).to_string()
        } else {
            "null".to_string()
        }
    }

    /// Helper for unary float functions (EXP, LN, SIN, COS, etc.)
    fn unary_float_fn<F: Fn(f64) -> f64>(batch: &RecordBatch, arg: &SqlExpr, f: F) -> io::Result<ArrayRef> {
        let arr = Self::evaluate_expr_to_array(batch, arg)?;
        let result: Vec<Option<f64>> = (0..batch.num_rows()).map(|i| {
            arr.as_any().downcast_ref::<Float64Array>().filter(|a| !a.is_null(i)).map(|a| f(a.value(i)))
            .or_else(|| arr.as_any().downcast_ref::<Int64Array>().filter(|a| !a.is_null(i)).map(|a| f(a.value(i) as f64)))
        }).collect();
        Ok(Arc::new(Float64Array::from(result)))
    }

    /// Helper for extracting date parts (YEAR, MONTH, DAY, etc.)
    fn extract_date_part<F: Fn(&str) -> Option<i64>>(batch: &RecordBatch, arg: &SqlExpr, extractor: F) -> io::Result<ArrayRef> {
        let arr = Self::evaluate_expr_to_array(batch, arg)?;
        if let Some(sa) = arr.as_any().downcast_ref::<StringArray>() {
            let result: Vec<Option<i64>> = (0..batch.num_rows()).map(|i| {
                if sa.is_null(i) { None } else { extractor(sa.value(i)) }
            }).collect();
            Ok(Arc::new(Int64Array::from(result)))
        } else {
            Err(err_data( "Date function requires string argument"))
        }
    }

    /// Parse a timestamp string to microseconds since Unix epoch.
    /// Supports: "YYYY-MM-DD HH:MM:SS", "YYYY-MM-DD HH:MM:SS.fff", "YYYY-MM-DD"
    fn parse_timestamp_string(s: &str) -> i64 {
        // Try datetime with fractional seconds
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f") {
            return dt.and_utc().timestamp_micros();
        }
        // Try datetime without fractional seconds
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
            return dt.and_utc().timestamp_micros();
        }
        // Try date only (midnight)
        if s.len() >= 10 {
            if let Ok(d) = chrono::NaiveDate::parse_from_str(&s[..10], "%Y-%m-%d") {
                if let Some(dt) = d.and_hms_opt(0, 0, 0) {
                    return dt.and_utc().timestamp_micros();
                }
            }
        }
        // Try parsing as raw integer (epoch micros)
        s.parse::<i64>().unwrap_or(0)
    }

    /// Parse a JSON-style float array string like `[1.0, 2.0, 3.0]` into
    /// little-endian float32 bytes.  Returns `None` if the string is not a
    /// valid float-array literal.
    pub(crate) fn try_parse_vector_string(s: &str) -> Option<Vec<u8>> {
        let s = s.trim();
        if !s.starts_with('[') || !s.ends_with(']') {
            return None;
        }
        let inner = &s[1..s.len() - 1];
        if inner.trim().is_empty() {
            return Some(Vec::new());
        }
        let mut bytes = Vec::new();
        for part in inner.split(',') {
            let v: f32 = part.trim().parse().ok()?;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Some(bytes)
    }

    /// Convert a SqlExpr back to a SQL string (for CHECK constraint persistence)
    fn sql_expr_to_string(expr: &SqlExpr) -> String {
        use crate::query::sql_parser::{BinaryOperator, UnaryOperator};
        match expr {
            SqlExpr::Column(name) => name.clone(),
            SqlExpr::Literal(v) => match v {
                Value::Int64(n) => n.to_string(),
                Value::Int32(n) => n.to_string(),
                Value::Float64(f) => format!("{}", f),
                Value::Float32(f) => format!("{}", f),
                Value::String(s) => format!("'{}'", s.replace('\'', "''")),
                Value::Bool(b) => if *b { "TRUE".to_string() } else { "FALSE".to_string() },
                Value::Null => "NULL".to_string(),
                _ => format!("{:?}", v),
            },
            SqlExpr::BinaryOp { left, op, right } => {
                let op_str = match op {
                    BinaryOperator::Eq => "=",
                    BinaryOperator::NotEq => "!=",
                    BinaryOperator::Lt => "<",
                    BinaryOperator::Le => "<=",
                    BinaryOperator::Gt => ">",
                    BinaryOperator::Ge => ">=",
                    BinaryOperator::And => "AND",
                    BinaryOperator::Or => "OR",
                    BinaryOperator::Add => "+",
                    BinaryOperator::Sub => "-",
                    BinaryOperator::Mul => "*",
                    BinaryOperator::Div => "/",
                    BinaryOperator::Mod => "%",
                };
                format!("{} {} {}", Self::sql_expr_to_string(left), op_str, Self::sql_expr_to_string(right))
            }
            SqlExpr::UnaryOp { op, expr } => {
                match op {
                    UnaryOperator::Not => format!("NOT {}", Self::sql_expr_to_string(expr)),
                    UnaryOperator::Minus => format!("-{}", Self::sql_expr_to_string(expr)),
                }
            }
            SqlExpr::Paren(inner) => format!("({})", Self::sql_expr_to_string(inner)),
            SqlExpr::Like { column, pattern, negated } => {
                if *negated { format!("{} NOT LIKE '{}'", column, pattern) }
                else { format!("{} LIKE '{}'", column, pattern) }
            }
            SqlExpr::In { column, values, negated } => {
                let vals: Vec<String> = values.iter().map(|v| match v {
                    Value::Int64(n) => n.to_string(),
                    Value::String(s) => format!("'{}'", s),
                    _ => format!("{:?}", v),
                }).collect();
                if *negated { format!("{} NOT IN ({})", column, vals.join(", ")) }
                else { format!("{} IN ({})", column, vals.join(", ")) }
            }
            SqlExpr::IsNull { column, negated } => {
                if *negated { format!("{} IS NOT NULL", column) }
                else { format!("{} IS NULL", column) }
            }
            SqlExpr::Between { column, low, high, negated } => {
                if *negated { format!("{} NOT BETWEEN {} AND {}", column, Self::sql_expr_to_string(low), Self::sql_expr_to_string(high)) }
                else { format!("{} BETWEEN {} AND {}", column, Self::sql_expr_to_string(low), Self::sql_expr_to_string(high)) }
            }
            SqlExpr::Function { name, args } => {
                let arg_strs: Vec<String> = args.iter().map(|a| Self::sql_expr_to_string(a)).collect();
                format!("{}({})", name, arg_strs.join(", "))
            }
            SqlExpr::Cast { expr, data_type } => {
                format!("CAST({} AS {:?})", Self::sql_expr_to_string(expr), data_type)
            }
            _ => format!("{:?}", expr),
        }
    }

    /// Parse a date string to days since Unix epoch.
    /// Supports: "YYYY-MM-DD"
    fn parse_date_string(s: &str) -> i64 {
        if s.len() >= 10 {
            if let Ok(d) = chrono::NaiveDate::parse_from_str(&s[..10], "%Y-%m-%d") {
                let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                return (d - epoch).num_days();
            }
        }
        // Try parsing as raw integer (epoch days)
        s.parse::<i64>().unwrap_or(0)
    }

    /// Evaluate array index expression: array[index]
    fn evaluate_array_index(batch: &RecordBatch, array_expr: &SqlExpr, index_expr: &SqlExpr) -> io::Result<ArrayRef> {
        // The array expression should be a SPLIT function that returns pipe-separated values
        // We'll evaluate the array expr and treat it as a string with delimiter
        let array_arr = Self::evaluate_expr_to_array(batch, array_expr)?;
        let index_arr = Self::evaluate_expr_to_array(batch, index_expr)?;
        
        if let Some(str_arr) = array_arr.as_any().downcast_ref::<StringArray>() {
            let result: Vec<Option<String>> = (0..batch.num_rows()).map(|i| {
                if str_arr.is_null(i) { return None; }
                let s = str_arr.value(i);
                // The array is stored as pipe-separated values from SPLIT function
                let parts: Vec<&str> = s.split('\x00').collect(); // Use null char as internal delimiter
                let idx = index_arr.as_any().downcast_ref::<Int64Array>()
                    .map(|ia| if ia.is_null(i) { 0 } else { ia.value(i) as usize })
                    .unwrap_or(0);
                parts.get(idx).map(|s| s.to_string())
            }).collect();
            Ok(Arc::new(StringArray::from(result.iter().map(|s| s.as_deref()).collect::<Vec<_>>())))
        } else {
            Err(err_data( "Array index requires array expression"))
        }
    }

    /// Evaluate IN expression with Value list.
    /// Fast path for string columns: builds an AHashSet and does a single O(N) scan
    /// instead of K separate vectorized-eq passes (K = number of IN values).
    fn evaluate_in_values(
        batch: &RecordBatch,
        column: &str,
        values: &[Value],
        negated: bool,
    ) -> io::Result<BooleanArray> {
        use ahash::AHashSet;
        use arrow::array::DictionaryArray;
        use arrow::datatypes::UInt32Type;

        let col_name = column.trim_matches('"');
        let target = Self::get_column_by_name(batch, col_name)
            .ok_or_else(|| err_not_found(format!("Column: {}", col_name)))?;
        let num_rows = batch.num_rows();

        // Fast path: all IN values are strings → use AHashSet for O(1) lookup per row
        let all_strings = values.iter().all(|v| matches!(v, Value::String(_)));
        if all_strings {
            let str_set: AHashSet<&str> = values.iter()
                .filter_map(|v| if let Value::String(s) = v { Some(s.as_str()) } else { None })
                .collect();

            let result: BooleanArray = if let Some(sa) = target.as_any().downcast_ref::<StringArray>() {
                sa.iter().map(|opt| opt.map(|s| str_set.contains(s)).unwrap_or(false)).collect()
            } else if let Some(da) = target.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
                let vals = da.values();
                let sv = vals.as_any().downcast_ref::<StringArray>()
                    .ok_or_else(|| err_data("Dictionary values must be strings"))?;
                let keys = da.keys();
                // Check which dictionary values are in the set (only iterate unique dict entries)
                let dict_match: Vec<bool> = (0..sv.len())
                    .map(|k| !sv.is_null(k) && str_set.contains(sv.value(k)))
                    .collect();
                (0..num_rows).map(|i| {
                    if keys.is_null(i) { false }
                    else { dict_match.get(keys.value(i) as usize).copied().unwrap_or(false) }
                }).collect()
            } else {
                // Non-string column: fall back to generic path
                let mut result = BooleanArray::from(vec![false; num_rows]);
                for val in values {
                    let va = Self::value_to_array(val, num_rows)?;
                    result = compute::or(&result, &cmp::eq(target, &va).map_err(|e| err_data(e.to_string()))?)
                        .map_err(|e| err_data(e.to_string()))?;
                }
                result
            };

            return if negated {
                compute::not(&result).map_err(|e| err_data(e.to_string()))
            } else {
                Ok(result)
            };
        }

        // Generic path for numeric IN values (vectorized eq + or)
        let mut result = BooleanArray::from(vec![false; num_rows]);
        for val in values {
            let val_array = Self::value_to_array(val, num_rows)?;
            let eq_mask = cmp::eq(target, &val_array)
                .map_err(|e| err_data(e.to_string()))?;
            result = compute::or(&result, &eq_mask)
                .map_err(|e| err_data(e.to_string()))?;
        }
        if negated {
            compute::not(&result).map_err(|e| err_data(e.to_string()))
        } else {
            Ok(result)
        }
    }

    /// Parse a SQL LIKE pattern into a fast-match closure, avoiding regex for simple cases.
    fn like_pattern_to_matcher(pattern: &str) -> io::Result<Box<dyn Fn(&str) -> bool + Send + Sync>> {
        let bytes = pattern.as_bytes();
        let len = bytes.len();
        // Check for wildcards (_ is single-char wildcard, % is any-length)
        let has_underscore = bytes.iter().any(|&b| b == b'_');
        let leading_pct  = !bytes.is_empty() && bytes[0] == b'%';
        let trailing_pct = !bytes.is_empty() && bytes[len - 1] == b'%';
        // Inner = pattern without leading/trailing %
        let inner_start = if leading_pct { 1 } else { 0 };
        let inner_end   = if trailing_pct && len > 0 { len - 1 } else { len };
        let inner = if inner_start <= inner_end { &pattern[inner_start..inner_end] } else { "" };
        let inner_has_wildcard = inner.contains(['%', '_']);

        if !has_underscore && !leading_pct && trailing_pct && !inner_has_wildcard {
            // "prefix%" — starts_with
            let prefix = inner.to_string();
            Ok(Box::new(move |s: &str| s.starts_with(prefix.as_str())))
        } else if !has_underscore && leading_pct && !trailing_pct && !inner_has_wildcard {
            // "%suffix" — ends_with
            let suffix = inner.to_string();
            Ok(Box::new(move |s: &str| s.ends_with(suffix.as_str())))
        } else if !has_underscore && leading_pct && trailing_pct && !inner_has_wildcard {
            // "%substr%" — contains
            let substr = inner.to_string();
            Ok(Box::new(move |s: &str| s.contains(substr.as_str())))
        } else if !has_underscore && !leading_pct && !trailing_pct && !inner_has_wildcard {
            // Exact match (no wildcards at all)
            let exact = pattern.to_string();
            Ok(Box::new(move |s: &str| s == exact.as_str()))
        } else {
            // Complex pattern: compile regex once
            let regex_pat = Self::like_to_regex(pattern);
            let re = regex::Regex::new(&regex_pat).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
            Ok(Box::new(move |s: &str| re.is_match(s)))
        }
    }

    /// Evaluate LIKE expression.
    /// Uses Arrow's optimized match_like kernel when available, with fallback to custom implementation.
    fn evaluate_like(
        batch: &RecordBatch,
        column: &str,
        pattern: &str,
        negated: bool,
    ) -> io::Result<BooleanArray> {
        use arrow::array::DictionaryArray;
        use arrow::datatypes::UInt32Type;

        let col_name = column.trim_matches('"');
        let array = Self::get_column_by_name(batch, col_name)
            .ok_or_else(|| err_not_found(format!("Column: {}", col_name)))?;

        // Use custom implementation with parallel processing (Arrow's compute::like has API compatibility issues)
        // Custom implementation with parallel processing
        let matcher = Self::like_pattern_to_matcher(pattern)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;

        let result: BooleanArray = if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
            const PAR_THRESHOLD: usize = 8192;
            if string_array.len() >= PAR_THRESHOLD {
                use rayon::prelude::*;
                let bools: Vec<bool> = (0..string_array.len())
                    .into_par_iter()
                    .map(|i| {
                        if string_array.is_null(i) { false }
                        else { matcher(string_array.value(i)) }
                    })
                    .collect();
                BooleanArray::from(bools)
            } else {
                string_array.iter().map(|opt| opt.map(|s| matcher(s)).unwrap_or(false)).collect()
            }
        } else if let Some(dict_array) = array.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
            let values = dict_array.values();
            let str_values = values.as_any().downcast_ref::<StringArray>()
                .ok_or_else(|| err_data("Dictionary values must be strings"))?;
            let keys = dict_array.keys();
            // Apply matcher only to unique dictionary entries, then map by key
            let dict_match: Vec<bool> = (0..str_values.len())
                .map(|k| !str_values.is_null(k) && matcher(str_values.value(k)))
                .collect();
            (0..dict_array.len()).map(|i| {
                if keys.is_null(i) { false }
                else { dict_match.get(keys.value(i) as usize).copied().unwrap_or(false) }
            }).collect()
        } else {
            return Err(err_data("LIKE requires string column"));
        };

        if negated {
            compute::not(&result).map_err(|e| err_data(e.to_string()))
        } else {
            Ok(result)
        }
    }

    /// Evaluate REGEXP expression
    fn evaluate_regexp(
        batch: &RecordBatch,
        column: &str,
        pattern: &str,
        negated: bool,
    ) -> io::Result<BooleanArray> {
        let col_name = column.trim_matches('"');
        let array = Self::get_column_by_name(batch, col_name)
            .ok_or_else(|| err_not_found(format!("Column: {}", col_name)))?;
        
        // Use pattern directly as regex (convert glob-style * to regex .*)
        let regex_pattern = pattern.replace("*", ".*");
        let regex = regex::Regex::new(&regex_pattern)
            .map_err(|e| err_input( e.to_string()))?;
        
        // Handle both StringArray and DictionaryArray
        use arrow::array::DictionaryArray;
        use arrow::datatypes::UInt32Type;
        
        let result: BooleanArray = if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
            string_array
                .iter()
                .map(|opt| opt.map(|s| regex.is_match(s)).unwrap_or(false))
                .collect()
        } else if let Some(dict_array) = array.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
            let values = dict_array.values();
            let str_values = values.as_any().downcast_ref::<StringArray>()
                .ok_or_else(|| err_data( "Dictionary values must be strings"))?;
            let keys = dict_array.keys();
            
            (0..dict_array.len())
                .map(|i| {
                    if keys.is_null(i) {
                        false
                    } else {
                        let key = keys.value(i) as usize;
                        if key < str_values.len() && !str_values.is_null(key) {
                            regex.is_match(str_values.value(key))
                        } else {
                            false
                        }
                    }
                })
                .collect()
        } else {
            return Err(err_data( "REGEXP requires string column"));
        };
        
        if negated {
            compute::not(&result)
                .map_err(|e| err_data( e.to_string()))
        } else {
            Ok(result)
        }
    }

    /// Convert SQL LIKE pattern to regex
    fn like_to_regex(pattern: &str) -> String {
        let mut regex = String::from("^");
        let mut chars = pattern.chars().peekable();
        
        while let Some(c) = chars.next() {
            match c {
                '%' => regex.push_str(".*"),
                '_' => regex.push('.'),
                '\\' => {
                    if let Some(&next) = chars.peek() {
                        if next == '%' || next == '_' {
                            regex.push(chars.next().unwrap());
                            continue;
                        }
                    }
                    regex.push_str("\\\\");
                }
                c if "[](){}|^$.*+?\\".contains(c) => {
                    regex.push('\\');
                    regex.push(c);
                }
                c => regex.push(c),
            }
        }
        
        regex.push('$');
        regex
    }
}
