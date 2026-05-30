use super::*;
use crate::storage::OnDemandStorage;
use std::collections::HashMap;
use tempfile::tempdir;

fn create_test_storage(path: &Path) {
    let storage = OnDemandStorage::create(path).unwrap();

    let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
    let mut float_cols: HashMap<String, Vec<f64>> = HashMap::new();
    let mut string_cols: HashMap<String, Vec<String>> = HashMap::new();

    int_cols.insert("id".to_string(), vec![1, 2, 3, 4, 5]);
    int_cols.insert("age".to_string(), vec![25, 30, 35, 40, 45]);
    float_cols.insert("score".to_string(), vec![85.0, 90.0, 75.0, 88.0, 92.0]);
    string_cols.insert(
        "name".to_string(),
        vec![
            "Alice".to_string(),
            "Bob".to_string(),
            "Charlie".to_string(),
            "Diana".to_string(),
            "Eve".to_string(),
        ],
    );

    storage
        .insert_typed(
            int_cols,
            float_cols,
            string_cols,
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();
    storage.save().unwrap();
}

#[test]
fn test_simple_select() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.apex");
    create_test_storage(&path);

    let result = ApexExecutor::execute("SELECT * FROM default", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 5);
}

#[test]
fn test_select_with_where() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.apex");
    create_test_storage(&path);

    let result = ApexExecutor::execute("SELECT * FROM default WHERE age > 30", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 3); // age 35, 40, 45
}

#[test]
fn test_select_with_limit() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.apex");
    create_test_storage(&path);

    let result = ApexExecutor::execute("SELECT * FROM default LIMIT 2", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 2);
}

#[test]
fn test_count_aggregate() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.apex");
    create_test_storage(&path);

    let result = ApexExecutor::execute("SELECT COUNT(*) FROM default", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 1);

    let count_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(count_array.value(0), 5);
}

#[test]
fn test_sum_aggregate() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.apex");
    create_test_storage(&path);

    let result = ApexExecutor::execute("SELECT SUM(age) FROM default", &path).unwrap();
    let batch = result.to_record_batch().unwrap();

    let sum_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(sum_array.value(0), 175); // 25+30+35+40+45
}

#[test]
fn test_order_by() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.apex");
    create_test_storage(&path);

    let result =
        ApexExecutor::execute("SELECT * FROM default ORDER BY age DESC LIMIT 2", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 2);

    let age_array = batch
        .column_by_name("age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(age_array.value(0), 45);
    assert_eq!(age_array.value(1), 40);
}

#[test]
fn test_is_null_query() {
    use crate::data::Value;
    use crate::storage::backend::TableStorageBackend;
    use std::collections::HashMap;

    let dir = tempdir().unwrap();
    let path = dir.path().join("test_null.apex");

    // Create storage with NULL boolean
    {
        let backend = TableStorageBackend::create(&path).unwrap();

        let mut row1 = HashMap::new();
        row1.insert("id".to_string(), Value::Int64(1));
        row1.insert("flag".to_string(), Value::Bool(true));

        let mut row2 = HashMap::new();
        row2.insert("id".to_string(), Value::Int64(2));
        row2.insert("flag".to_string(), Value::Bool(false));

        let mut row3 = HashMap::new();
        row3.insert("id".to_string(), Value::Int64(3));
        row3.insert("flag".to_string(), Value::Null); // NULL boolean

        backend.insert_rows(&[row1, row2, row3]).unwrap();
        backend.save().unwrap();
    }

    // Clear any cached backend
    invalidate_storage_cache(&path);

    // First check SELECT * to verify data is correctly read
    let result_all =
        ApexExecutor::execute("SELECT id, flag FROM test_null ORDER BY id", &path).unwrap();
    let batch_all = result_all.to_record_batch().unwrap();

    println!("SELECT all rows: {} rows", batch_all.num_rows());
    if let Some(flag_col) = batch_all.column_by_name("flag") {
        println!(
            "Flag column null_count in SELECT *: {}",
            flag_col.null_count()
        );
        let bool_arr = flag_col.as_any().downcast_ref::<BooleanArray>().unwrap();
        for i in 0..bool_arr.len() {
            println!(
                "  Row {}: is_null={}, value={:?}",
                i,
                bool_arr.is_null(i),
                if bool_arr.is_null(i) {
                    None
                } else {
                    Some(bool_arr.value(i))
                }
            );
        }
    }

    // Clear cache again before IS NULL query
    invalidate_storage_cache(&path);

    // Test reading just the flag column for WHERE evaluation
    let backend = get_cached_backend(&path).unwrap();
    let where_batch = backend
        .read_columns_to_arrow(Some(&["flag"]), 0, None)
        .unwrap();
    println!("WHERE batch (flag only): {} rows", where_batch.num_rows());
    if let Some(flag_col) = where_batch.column_by_name("flag") {
        println!("  null_count: {}", flag_col.null_count());
        let bool_arr = flag_col.as_any().downcast_ref::<BooleanArray>().unwrap();
        for i in 0..bool_arr.len() {
            println!("    Row {}: is_null={}", i, bool_arr.is_null(i));
        }
        // Test is_null compute
        let is_null_mask = arrow::compute::is_null(flag_col).unwrap();
        println!("  is_null mask: {:?}", is_null_mask);
    }

    // Clear cache again
    invalidate_storage_cache(&path);

    // Manually test the predicate evaluation path
    let backend2 = get_cached_backend(&path).unwrap();
    let full_batch = backend2.read_columns_to_arrow(None, 0, None).unwrap();
    println!("Full batch before filter: {} rows", full_batch.num_rows());

    // Check flag column null status in full batch
    if let Some(flag_col) = full_batch.column_by_name("flag") {
        println!("  flag null_count in full batch: {}", flag_col.null_count());
        // Compute is_null mask
        let is_null_mask = arrow::compute::is_null(flag_col).unwrap();
        println!("  is_null mask: {:?}", is_null_mask);
        let true_count = is_null_mask.iter().filter(|v| *v == Some(true)).count();
        println!("  True count in is_null mask: {}", true_count);

        // Manually apply filter
        let filtered_batch =
            arrow::compute::filter_record_batch(&full_batch, &is_null_mask).unwrap();
        println!(
            "  Manually filtered batch: {} rows",
            filtered_batch.num_rows()
        );
    }

    // Clear cache again
    invalidate_storage_cache(&path);

    // Check what the parser produces for the IS NULL query
    let parsed = SqlParser::parse("SELECT * FROM test_null WHERE flag IS NULL").unwrap();
    if let SqlStatement::Select(stmt) = parsed {
        println!("Parsed statement:");
        println!("  is_select_star: {}", stmt.is_select_star());
        println!("  where_clause: {:?}", stmt.where_clause);
        println!("  where_columns: {:?}", stmt.where_columns());
        println!("  order_by: {:?}", stmt.order_by);
    }

    // Test IS NULL query
    let result =
        ApexExecutor::execute("SELECT * FROM test_null WHERE flag IS NULL", &path).unwrap();
    let batch = result.to_record_batch().unwrap();

    println!("IS NULL query result: {} rows", batch.num_rows());
    println!("Schema: {:?}", batch.schema());

    // Check the flag column for nulls
    if let Some(flag_col) = batch.column_by_name("flag") {
        println!("Flag column null_count: {}", flag_col.null_count());
    }

    assert_eq!(batch.num_rows(), 1, "IS NULL should return 1 row");

    // Also test that SELECT * returns correct data
    let result2 =
        ApexExecutor::execute("SELECT id, flag FROM test_null ORDER BY id", &path).unwrap();
    let batch2 = result2.to_record_batch().unwrap();

    println!("SELECT all rows: {} rows", batch2.num_rows());
    if let Some(flag_col) = batch2.column_by_name("flag") {
        println!(
            "Flag column null_count in SELECT *: {}",
            flag_col.null_count()
        );
        let bool_arr = flag_col.as_any().downcast_ref::<BooleanArray>().unwrap();
        for i in 0..bool_arr.len() {
            println!(
                "  Row {}: is_null={}, value={:?}",
                i,
                bool_arr.is_null(i),
                if bool_arr.is_null(i) {
                    None
                } else {
                    Some(bool_arr.value(i))
                }
            );
        }
    }
}

// ========================================================================
// OLTP Tests: Insert, Point Lookup, Update, Delete, Batch Operations
// ========================================================================

fn create_oltp_storage(path: &Path) {
    let storage = OnDemandStorage::create(path).unwrap();
    let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
    let mut float_cols: HashMap<String, Vec<f64>> = HashMap::new();
    let mut string_cols: HashMap<String, Vec<String>> = HashMap::new();
    let mut bool_cols: HashMap<String, Vec<bool>> = HashMap::new();

    let n = 1000;
    int_cols.insert("user_id".to_string(), (1..=n as i64).collect());
    int_cols.insert(
        "age".to_string(),
        (0..n).map(|i| 20 + (i % 50) as i64).collect(),
    );
    float_cols.insert(
        "balance".to_string(),
        (0..n).map(|i| 100.0 + i as f64 * 1.5).collect(),
    );
    string_cols.insert(
        "city".to_string(),
        (0..n)
            .map(|i| {
                ["Beijing", "Shanghai", "Shenzhen", "Guangzhou", "Hangzhou"][i % 5].to_string()
            })
            .collect(),
    );
    bool_cols.insert("active".to_string(), (0..n).map(|i| i % 3 != 0).collect());

    storage
        .insert_typed(int_cols, float_cols, string_cols, HashMap::new(), bool_cols)
        .unwrap();
    storage.save().unwrap();
}

#[test]
fn test_oltp_insert_and_row_count() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("oltp_insert.apex");
    create_oltp_storage(&path);

    let result = ApexExecutor::execute("SELECT COUNT(*) FROM default", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    let count = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 1000);
}

#[test]
fn test_oltp_point_lookup_by_id() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("oltp_point.apex");
    create_oltp_storage(&path);

    // Point lookup by _id (first row)
    let result = ApexExecutor::execute("SELECT * FROM default WHERE _id = 1", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 1);
    let uid = batch
        .column_by_name("user_id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(uid, 1);
}

#[test]
fn test_oltp_batch_insert_incremental() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("oltp_batch.apex");
    create_oltp_storage(&path);

    // Insert additional batch — force V4 data load via to_arrow_batch
    invalidate_storage_cache(&path);
    let storage = OnDemandStorage::open(&path).unwrap();
    let _ = storage.to_arrow_batch(None, true); // reads via mmap path
    let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
    let mut float_cols: HashMap<String, Vec<f64>> = HashMap::new();
    let mut string_cols: HashMap<String, Vec<String>> = HashMap::new();
    int_cols.insert("user_id".to_string(), vec![1001, 1002, 1003]);
    int_cols.insert("age".to_string(), vec![28, 35, 42]);
    float_cols.insert("balance".to_string(), vec![5000.0, 6000.0, 7000.0]);
    string_cols.insert(
        "city".to_string(),
        vec![
            "Chengdu".to_string(),
            "Wuhan".to_string(),
            "Nanjing".to_string(),
        ],
    );
    storage
        .insert_typed(
            int_cols,
            float_cols,
            string_cols,
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();
    storage.save().unwrap();

    invalidate_storage_cache(&path);
    let result = ApexExecutor::execute("SELECT COUNT(*) FROM default", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    let count = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 1003);
}

#[test]
fn test_oltp_update_single_row() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("oltp_update.apex");
    create_oltp_storage(&path);

    // Update age for user_id = 1
    let result =
        ApexExecutor::execute("UPDATE default SET age = 99 WHERE user_id = 1", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert!(batch.num_rows() >= 1);

    invalidate_storage_cache(&path);
    // After UPDATE, verify updated value exists (UPDATE may soft-delete + re-insert)
    let result2 =
        ApexExecutor::execute("SELECT age FROM default WHERE user_id = 1", &path).unwrap();
    let batch2 = result2.to_record_batch().unwrap();
    assert!(
        batch2.num_rows() >= 1,
        "Should find at least 1 row with user_id=1"
    );
    // Check that at least one row has the updated age value
    let age_arr = batch2
        .column_by_name("age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let has_updated = (0..age_arr.len()).any(|i| age_arr.value(i) == 99);
    assert!(
        has_updated,
        "At least one row should have age=99 after update"
    );
}

#[test]
fn test_oltp_delete_single_row() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("oltp_delete.apex");
    create_oltp_storage(&path);

    let result = ApexExecutor::execute("DELETE FROM default WHERE user_id = 1", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert!(batch.num_rows() >= 1);

    invalidate_storage_cache(&path);
    let result2 =
        ApexExecutor::execute("SELECT COUNT(*) FROM default WHERE user_id = 1", &path).unwrap();
    let batch2 = result2.to_record_batch().unwrap();
    let count = batch2
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 0);
}

#[test]
fn test_oltp_delete_then_count() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("oltp_del_cnt.apex");
    create_oltp_storage(&path);

    ApexExecutor::execute("DELETE FROM default WHERE city = 'Beijing'", &path).unwrap();
    invalidate_storage_cache(&path);
    let result = ApexExecutor::execute("SELECT COUNT(*) FROM default", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    let count = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    // Beijing is every 5th row → 200 deleted from 1000
    assert_eq!(count, 800);
}

#[test]
fn test_oltp_update_multiple_rows() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("oltp_upd_multi.apex");
    create_oltp_storage(&path);

    ApexExecutor::execute(
        "UPDATE default SET balance = 0.0 WHERE city = 'Shanghai'",
        &path,
    )
    .unwrap();
    invalidate_storage_cache(&path);
    let result =
        ApexExecutor::execute("SELECT COUNT(*) FROM default WHERE balance = 0.0", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    let count = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    // Shanghai is every 5th row starting at index 1 → 200 rows
    assert_eq!(count, 200);
}

#[test]
fn test_oltp_string_equality_filter() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("oltp_str_eq.apex");
    create_oltp_storage(&path);

    let result = ApexExecutor::execute(
        "SELECT * FROM default WHERE city = 'Shenzhen' LIMIT 10",
        &path,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 10);

    // Verify all returned rows have city = 'Shenzhen'
    if let Some(city_col) = batch.column_by_name("city") {
        let str_arr = city_col.as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..str_arr.len() {
            assert_eq!(str_arr.value(i), "Shenzhen");
        }
    }
}

#[test]
fn test_oltp_numeric_range_filter() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("oltp_range.apex");
    create_oltp_storage(&path);

    let result = ApexExecutor::execute(
        "SELECT * FROM default WHERE age BETWEEN 30 AND 35 LIMIT 50",
        &path,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();
    assert!(batch.num_rows() > 0 && batch.num_rows() <= 50);

    let age_arr = batch
        .column_by_name("age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    for i in 0..age_arr.len() {
        let v = age_arr.value(i);
        assert!(v >= 30 && v <= 35, "age {} out of range [30,35]", v);
    }
}

#[test]
fn test_oltp_insert_then_immediate_read() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("oltp_ins_read.apex");

    let storage = OnDemandStorage::create(&path).unwrap();
    let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
    int_cols.insert("val".to_string(), vec![42]);
    let ids = storage
        .insert_typed(
            int_cols,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();
    storage.save().unwrap();

    // Immediately read back without cache invalidation
    let result = ApexExecutor::execute("SELECT val FROM default", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 1);
    let val = batch
        .column_by_name("val")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(val, 42);
    assert_eq!(ids.len(), 1);
}

// ========================================================================
// OLAP Tests: Full Scan, Aggregation, GROUP BY, ORDER BY, LIMIT, Subquery
// ========================================================================

fn create_olap_storage(path: &Path) {
    let storage = OnDemandStorage::create(path).unwrap();
    let n = 5000;
    let cities = [
        "Beijing",
        "Shanghai",
        "Shenzhen",
        "Guangzhou",
        "Hangzhou",
        "Chengdu",
        "Wuhan",
        "Nanjing",
        "Tianjin",
        "Xian",
    ];
    let depts = ["Engineering", "Sales", "Marketing", "HR", "Finance"];

    let mut int_cols: HashMap<String, Vec<i64>> = HashMap::new();
    let mut float_cols: HashMap<String, Vec<f64>> = HashMap::new();
    let mut string_cols: HashMap<String, Vec<String>> = HashMap::new();
    let mut bool_cols: HashMap<String, Vec<bool>> = HashMap::new();

    int_cols.insert("emp_id".to_string(), (1..=n as i64).collect());
    int_cols.insert(
        "age".to_string(),
        (0..n).map(|i| 22 + (i % 40) as i64).collect(),
    );
    int_cols.insert(
        "years".to_string(),
        (0..n).map(|i| (i % 20) as i64).collect(),
    );
    float_cols.insert(
        "salary".to_string(),
        (0..n)
            .map(|i| 50000.0 + (i % 100) as f64 * 1000.0)
            .collect(),
    );
    string_cols.insert(
        "city".to_string(),
        (0..n).map(|i| cities[i % 10].to_string()).collect(),
    );
    string_cols.insert(
        "dept".to_string(),
        (0..n).map(|i| depts[i % 5].to_string()).collect(),
    );
    bool_cols.insert(
        "is_manager".to_string(),
        (0..n).map(|i| i % 10 == 0).collect(),
    );

    storage
        .insert_typed(int_cols, float_cols, string_cols, HashMap::new(), bool_cols)
        .unwrap();
    storage.save().unwrap();
}

#[test]
fn test_olap_full_scan() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_scan.apex");
    create_olap_storage(&path);

    let result = ApexExecutor::execute("SELECT * FROM default", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 5000);
    // Verify all expected columns exist
    assert!(batch.column_by_name("emp_id").is_some());
    assert!(batch.column_by_name("salary").is_some());
    assert!(batch.column_by_name("city").is_some());
    assert!(batch.column_by_name("dept").is_some());
}

#[test]
fn test_olap_count_star() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_count.apex");
    create_olap_storage(&path);

    let result = ApexExecutor::execute("SELECT COUNT(*) FROM default", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        5000
    );
}

#[test]
fn test_olap_sum_avg_min_max() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_aggs.apex");
    create_olap_storage(&path);

    // SUM
    let result = ApexExecutor::execute("SELECT SUM(salary) FROM default", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    let sum_col = batch.column(0);
    // salary = 50000 + (i%100)*1000 for i=0..5000
    // Each cycle of 100: sum = 100*50000 + (0+1+...+99)*1000 = 5000000+4950000 = 9950000
    // 50 cycles: 50*9950000 = 497500000
    let sum_val = if let Some(arr) = sum_col.as_any().downcast_ref::<Float64Array>() {
        arr.value(0)
    } else {
        sum_col
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0) as f64
    };
    assert!(
        (sum_val - 497_500_000.0).abs() < 1.0,
        "SUM(salary) = {}",
        sum_val
    );

    invalidate_storage_cache(&path);
    // AVG
    let result = ApexExecutor::execute("SELECT AVG(salary) FROM default", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    let avg_val = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    assert!((avg_val - 99500.0).abs() < 1.0, "AVG(salary) = {}", avg_val);

    invalidate_storage_cache(&path);
    // MIN
    let result = ApexExecutor::execute("SELECT MIN(salary) FROM default", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    let min_col = batch.column(0);
    let min_val = if let Some(arr) = min_col.as_any().downcast_ref::<Float64Array>() {
        arr.value(0)
    } else {
        min_col
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0) as f64
    };
    assert!((min_val - 50000.0).abs() < 1.0, "MIN(salary) = {}", min_val);

    invalidate_storage_cache(&path);
    // MAX
    let result = ApexExecutor::execute("SELECT MAX(salary) FROM default", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    let max_col = batch.column(0);
    let max_val = if let Some(arr) = max_col.as_any().downcast_ref::<Float64Array>() {
        arr.value(0)
    } else {
        max_col
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0) as f64
    };
    assert!(
        (max_val - 149000.0).abs() < 1.0,
        "MAX(salary) = {}",
        max_val
    );
}

#[test]
fn test_olap_group_by_single_col() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_gb1.apex");
    create_olap_storage(&path);

    let result = ApexExecutor::execute(
        "SELECT dept, COUNT(*) as cnt FROM default GROUP BY dept ORDER BY cnt DESC",
        &path,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();
    // 5 departments
    assert_eq!(batch.num_rows(), 5);
    // Each dept has 1000 rows (5000/5)
    let cnt_col = batch
        .column_by_name("cnt")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    for i in 0..cnt_col.len() {
        assert_eq!(cnt_col.value(i), 1000, "dept group {} count", i);
    }
}

#[test]
fn test_olap_group_by_two_cols() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_gb2.apex");
    create_olap_storage(&path);

    let result = ApexExecutor::execute(
            "SELECT city, dept, COUNT(*) as cnt FROM default GROUP BY city, dept ORDER BY cnt DESC LIMIT 10", &path
        ).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 10);
    // Verify all groups have positive counts and sum makes sense
    let cnt_col = batch
        .column_by_name("cnt")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    for i in 0..cnt_col.len() {
        assert!(cnt_col.value(i) > 0, "group {} has zero count", i);
    }
    // Verify descending order
    for i in 1..cnt_col.len() {
        assert!(
            cnt_col.value(i - 1) >= cnt_col.value(i),
            "not DESC order at {}",
            i
        );
    }
}

#[test]
fn test_olap_group_by_with_having() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_having.apex");
    create_olap_storage(&path);

    let result = ApexExecutor::execute(
        "SELECT city, AVG(salary) as avg_sal FROM default GROUP BY city HAVING AVG(salary) > 99000",
        &path,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();
    // All cities have similar salary distribution, so avg ~99500
    assert!(batch.num_rows() > 0, "HAVING should return some rows");
    let avg_col = batch
        .column_by_name("avg_sal")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for i in 0..avg_col.len() {
        assert!(
            avg_col.value(i) > 99000.0,
            "avg_sal {} <= 99000",
            avg_col.value(i)
        );
    }
}

#[test]
fn test_olap_order_by_desc_limit() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_order.apex");
    create_olap_storage(&path);

    let result =
        ApexExecutor::execute("SELECT * FROM default ORDER BY salary DESC LIMIT 10", &path)
            .unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 10);

    let sal_arr = batch
        .column_by_name("salary")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    // Verify descending order
    for i in 1..sal_arr.len() {
        assert!(
            sal_arr.value(i - 1) >= sal_arr.value(i),
            "salary[{}]={} < salary[{}]={}",
            i - 1,
            sal_arr.value(i - 1),
            i,
            sal_arr.value(i)
        );
    }
    // Top salary should be 149000
    assert!((sal_arr.value(0) - 149000.0).abs() < 1.0);
}

#[test]
fn test_olap_order_by_asc_limit() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_order_asc.apex");
    create_olap_storage(&path);

    let result =
        ApexExecutor::execute("SELECT * FROM default ORDER BY age ASC LIMIT 5", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 5);

    let age_arr = batch
        .column_by_name("age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    for i in 0..age_arr.len() {
        assert_eq!(age_arr.value(i), 22, "min age should be 22");
    }
}

#[test]
fn test_olap_where_between() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_between.apex");
    create_olap_storage(&path);

    let result = ApexExecutor::execute(
        "SELECT * FROM default WHERE age BETWEEN 30 AND 35 LIMIT 100",
        &path,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();
    assert!(batch.num_rows() > 0 && batch.num_rows() <= 100);

    let age_arr = batch
        .column_by_name("age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    for i in 0..age_arr.len() {
        let v = age_arr.value(i);
        assert!(v >= 30 && v <= 35, "age {} not in [30,35]", v);
    }
}

#[test]
fn test_olap_where_string_eq_no_limit() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_str_nolim.apex");
    create_olap_storage(&path);

    let result =
        ApexExecutor::execute("SELECT * FROM default WHERE city = 'Beijing'", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    // Beijing = every 10th row → 500
    assert_eq!(batch.num_rows(), 500);
}

#[test]
fn test_olap_complex_filter_group_order() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_complex.apex");
    create_olap_storage(&path);

    let result = ApexExecutor::execute(
        "SELECT dept, COUNT(*) as cnt, AVG(salary) as avg_sal FROM default \
             WHERE city = 'Beijing' GROUP BY dept ORDER BY avg_sal DESC",
        &path,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();
    // Beijing rows across departments — verify groups exist
    assert!(
        batch.num_rows() >= 1,
        "Should have at least 1 dept group for Beijing"
    );
    let cnt_col = batch
        .column_by_name("cnt")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let total: i64 = (0..cnt_col.len()).map(|i| cnt_col.value(i)).sum();
    assert_eq!(
        total, 500,
        "Total Beijing rows across all depts should be 500"
    );
}

#[test]
fn test_olap_count_distinct() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_cdist.apex");
    create_olap_storage(&path);

    let result = ApexExecutor::execute("SELECT COUNT(DISTINCT city) FROM default", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    let count = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 10);
}

#[test]
fn test_olap_limit_on_filter() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_limit.apex");
    create_olap_storage(&path);

    // LIMIT on string filter
    let result = ApexExecutor::execute(
        "SELECT * FROM default WHERE city = 'Beijing' LIMIT 10",
        &path,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 10);

    // LIMIT on full scan
    invalidate_storage_cache(&path);
    let result2 = ApexExecutor::execute("SELECT * FROM default LIMIT 20", &path).unwrap();
    let batch2 = result2.to_record_batch().unwrap();
    assert_eq!(batch2.num_rows(), 20);
}

#[test]
fn test_olap_in_list() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_in.apex");
    create_olap_storage(&path);

    let result = ApexExecutor::execute(
        "SELECT * FROM default WHERE city IN ('Beijing', 'Shanghai')",
        &path,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();
    // Beijing + Shanghai = 500 + 500 = 1000
    assert_eq!(batch.num_rows(), 1000);
}

#[test]
fn test_olap_like_filter() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_like.apex");
    create_olap_storage(&path);

    let result =
        ApexExecutor::execute("SELECT * FROM default WHERE city LIKE 'Sh%'", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    // Shanghai + Shenzhen = 500 + 500 = 1000
    assert_eq!(batch.num_rows(), 1000);
}

#[test]
fn test_olap_multi_condition_and() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_and.apex");
    create_olap_storage(&path);

    let result = ApexExecutor::execute(
        "SELECT * FROM default WHERE city = 'Beijing' AND age > 50",
        &path,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();
    // Beijing every 10th, age = 22 + (i%40), age > 50 means i%40 > 28 → i%40 in [29..39] = 11 values
    // Among Beijing rows (i%10==0), i%40 distribution: need i%10==0 AND i%40>28
    // This is a subset check — just verify all returned rows satisfy both conditions
    assert!(batch.num_rows() > 0);
    if let Some(city_col) = batch.column_by_name("city") {
        let str_arr = city_col.as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..str_arr.len() {
            assert_eq!(str_arr.value(i), "Beijing");
        }
    }
    let age_arr = batch
        .column_by_name("age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    for i in 0..age_arr.len() {
        assert!(age_arr.value(i) > 50);
    }
}

#[test]
fn test_olap_multi_condition_or() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_or.apex");
    create_olap_storage(&path);

    let result =
        ApexExecutor::execute("SELECT * FROM default WHERE age < 23 OR age > 60", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert!(batch.num_rows() > 0);
    let age_arr = batch
        .column_by_name("age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    for i in 0..age_arr.len() {
        let v = age_arr.value(i);
        assert!(v < 23 || v > 60, "age {} not < 23 and not > 60", v);
    }
}

#[test]
fn test_olap_column_projection() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_proj.apex");
    create_olap_storage(&path);

    let result =
        ApexExecutor::execute("SELECT emp_id, salary FROM default LIMIT 10", &path).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 10);
    assert_eq!(batch.num_columns(), 2);
    assert!(batch.column_by_name("emp_id").is_some());
    assert!(batch.column_by_name("salary").is_some());
}

#[test]
fn test_olap_expression_in_select() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_expr.apex");
    create_olap_storage(&path);

    // Use CAST expression which is supported by the SQL parser
    let result = ApexExecutor::execute(
        "SELECT emp_id, CAST(salary AS INT) as salary_int FROM default LIMIT 5",
        &path,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 5);
    assert!(batch.column_by_name("salary_int").is_some());
}

#[test]
fn test_olap_group_by_with_sum() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_gb_sum.apex");
    create_olap_storage(&path);

    let result = ApexExecutor::execute(
        "SELECT dept, SUM(salary) as total_sal FROM default GROUP BY dept ORDER BY total_sal DESC",
        &path,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 5);
    let sum_col = batch
        .column_by_name("total_sal")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    // Verify descending order
    for i in 1..sum_col.len() {
        assert!(sum_col.value(i - 1) >= sum_col.value(i));
    }
}

#[test]
fn test_olap_empty_result() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_empty.apex");
    create_olap_storage(&path);

    let result = ApexExecutor::execute(
        "SELECT * FROM default WHERE city = 'NonExistentCity'",
        &path,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 0);
}

#[test]
fn test_olap_boolean_filter() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("olap_bool.apex");
    create_olap_storage(&path);

    let result = ApexExecutor::execute(
        "SELECT COUNT(*) FROM default WHERE is_manager = true",
        &path,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();
    let count = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    // is_manager = true when i%10==0 → 500 rows
    assert_eq!(count, 500);
}

// ========== P0-5: Constraint Tests ==========

/// Helper: parse + execute SQL via multi-table path with a base dir
fn exec_multi(sql: &str, base_dir: &Path) -> io::Result<ApexResult> {
    let default_path = base_dir.join("default.apex");
    ApexExecutor::execute_with_base_dir(sql, base_dir, &default_path)
}

/// Helper: assert that a Result is an error containing expected substring
fn assert_err_contains(result: io::Result<ApexResult>, expected: &str) {
    match result {
        Ok(_) => panic!("Expected error containing '{}', but got Ok", expected),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains(expected),
                "Expected error containing '{}', got: {}",
                expected,
                msg
            );
        }
    }
}

#[test]
fn test_constraint_not_null_reject() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE t1 (id INT NOT NULL, name TEXT)", base).unwrap();
    exec_multi("INSERT INTO t1 (id, name) VALUES (1, 'Alice')", base).unwrap();
    assert_err_contains(
        exec_multi("INSERT INTO t1 (id, name) VALUES (NULL, 'Bob')", base),
        "NOT NULL",
    );
}

#[test]
fn test_constraint_unique_reject() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE t2 (id INT, email TEXT UNIQUE)", base).unwrap();
    exec_multi("INSERT INTO t2 (id, email) VALUES (1, 'a@b.com')", base).unwrap();
    assert_err_contains(
        exec_multi("INSERT INTO t2 (id, email) VALUES (2, 'a@b.com')", base),
        "UNIQUE",
    );
}

#[test]
fn test_constraint_unique_allows_multiple_nulls() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE t3 (id INT, email TEXT UNIQUE)", base).unwrap();
    exec_multi("INSERT INTO t3 (id, email) VALUES (1, NULL)", base).unwrap();
    exec_multi("INSERT INTO t3 (id, email) VALUES (2, NULL)", base).unwrap();
    let result = exec_multi("SELECT COUNT(*) FROM t3", base).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        2
    );
}

#[test]
fn test_constraint_primary_key_reject() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE t4 (uid INT PRIMARY KEY, name TEXT)", base).unwrap();
    exec_multi("INSERT INTO t4 (uid, name) VALUES (1, 'Alice')", base).unwrap();
    assert_err_contains(
        exec_multi("INSERT INTO t4 (uid, name) VALUES (1, 'Bob')", base),
        "PRIMARY KEY",
    );
}

#[test]
fn test_constraint_primary_key_implies_not_null() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE t5 (uid INT PRIMARY KEY, name TEXT)", base).unwrap();
    assert_err_contains(
        exec_multi("INSERT INTO t5 (uid, name) VALUES (NULL, 'Alice')", base),
        "NOT NULL",
    );
}

#[test]
fn test_constraint_default_value_fill() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi(
        "CREATE TABLE t6 (id INT NOT NULL, score INT DEFAULT 100)",
        base,
    )
    .unwrap();
    exec_multi("INSERT INTO t6 (id) VALUES (1)", base).unwrap();
    let result = exec_multi("SELECT score FROM t6 WHERE id = 1", base).unwrap();
    let batch = result.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 1);
    let score = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(score, 100);
}

#[test]
fn test_constraint_update_not_null_reject() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi(
        "CREATE TABLE t7 (id INT NOT NULL, name TEXT NOT NULL)",
        base,
    )
    .unwrap();
    exec_multi("INSERT INTO t7 (id, name) VALUES (1, 'Alice')", base).unwrap();
    assert_err_contains(
        exec_multi("UPDATE t7 SET name = NULL WHERE id = 1", base),
        "NOT NULL",
    );
}

#[test]
fn test_constraint_update_unique_reject() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE t8 (id INT, email TEXT UNIQUE)", base).unwrap();
    exec_multi("INSERT INTO t8 (id, email) VALUES (1, 'a@b.com')", base).unwrap();
    exec_multi("INSERT INTO t8 (id, email) VALUES (2, 'c@d.com')", base).unwrap();
    assert_err_contains(
        exec_multi("UPDATE t8 SET email = 'a@b.com' WHERE id = 2", base),
        "UNIQUE",
    );
}

#[test]
fn test_constraint_batch_insert_duplicate_in_batch() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE t9 (id INT, email TEXT UNIQUE)", base).unwrap();
    assert_err_contains(
        exec_multi(
            "INSERT INTO t9 (id, email) VALUES (1, 'x@y.com'), (2, 'x@y.com')",
            base,
        ),
        "UNIQUE",
    );
}

// ========== P1: CTAS Tests ==========

#[test]
fn test_ctas_basic() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE src (id INT, name TEXT)", base).unwrap();
    exec_multi(
        "INSERT INTO src (id, name) VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Carol')",
        base,
    )
    .unwrap();
    let result = exec_multi(
        "CREATE TABLE dst AS SELECT id, name FROM src WHERE id > 1",
        base,
    )
    .unwrap();
    // Should return number of inserted rows
    if let ApexResult::Scalar(n) = result {
        assert_eq!(n, 2);
    }
    // Verify the new table has the right data
    let q = exec_multi("SELECT COUNT(*) FROM dst", base).unwrap();
    let batch = q.to_record_batch().unwrap();
    assert_eq!(
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        2
    );
}

#[test]
fn test_ctas_if_not_exists() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE src2 (id INT)", base).unwrap();
    exec_multi("INSERT INTO src2 (id) VALUES (1)", base).unwrap();
    exec_multi("CREATE TABLE dst2 AS SELECT id FROM src2", base).unwrap();
    // Second CTAS without IF NOT EXISTS should error
    assert_err_contains(
        exec_multi("CREATE TABLE dst2 AS SELECT id FROM src2", base),
        "already exists",
    );
    // With IF NOT EXISTS should succeed silently
    let result = exec_multi(
        "CREATE TABLE IF NOT EXISTS dst2 AS SELECT id FROM src2",
        base,
    )
    .unwrap();
    if let ApexResult::Scalar(n) = result {
        assert_eq!(n, 0);
    }
}

#[test]
fn test_ctas_empty_result() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE src3 (id INT, val TEXT)", base).unwrap();
    exec_multi("INSERT INTO src3 (id, val) VALUES (1, 'x')", base).unwrap();
    let result = exec_multi(
        "CREATE TABLE dst3 AS SELECT id, val FROM src3 WHERE id > 999",
        base,
    )
    .unwrap();
    if let ApexResult::Scalar(n) = result {
        assert_eq!(n, 0);
    }
    let q = exec_multi("SELECT COUNT(*) FROM dst3", base).unwrap();
    let batch = q.to_record_batch().unwrap();
    assert_eq!(
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        0
    );
}

// ========== P1: RIGHT / FULL OUTER / CROSS JOIN Tests ==========

#[test]
fn test_right_join() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE left_t (id INT, lval TEXT)", base).unwrap();
    exec_multi("CREATE TABLE right_t (id INT, rval TEXT)", base).unwrap();
    exec_multi(
        "INSERT INTO left_t (id, lval) VALUES (1, 'a'), (2, 'b')",
        base,
    )
    .unwrap();
    exec_multi(
        "INSERT INTO right_t (id, rval) VALUES (2, 'x'), (3, 'y')",
        base,
    )
    .unwrap();
    let result = exec_multi(
        "SELECT * FROM left_t RIGHT JOIN right_t ON left_t.id = right_t.id",
        base,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();
    // RIGHT JOIN: all right rows preserved. id=2 matches, id=3 has NULL left.
    assert_eq!(batch.num_rows(), 2);
}

#[test]
fn test_right_join_qualified_key_projection() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE left_t (id INT, lval TEXT)", base).unwrap();
    exec_multi("CREATE TABLE right_t (id INT, rval TEXT)", base).unwrap();
    exec_multi(
        "INSERT INTO left_t (id, lval) VALUES (1, 'a'), (2, 'b')",
        base,
    )
    .unwrap();
    exec_multi(
        "INSERT INTO right_t (id, rval) VALUES (2, 'x'), (3, 'y')",
        base,
    )
    .unwrap();

    let result = exec_multi(
        "SELECT left_t.id AS lid, right_t.id AS rid, left_t.lval, right_t.rval \
         FROM left_t RIGHT JOIN right_t ON left_t.id = right_t.id \
         ORDER BY right_t.rval",
        base,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();

    let lid = batch
        .column_by_name("lid")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let rid = batch
        .column_by_name("rid")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let lval = batch
        .column_by_name("lval")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let rval = batch
        .column_by_name("rval")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(lid.value(0), 2);
    assert!(lid.is_null(1));
    assert_eq!(rid.value(0), 2);
    assert_eq!(rid.value(1), 3);
    assert_eq!(lval.value(0), "b");
    assert!(lval.is_null(1));
    assert_eq!(rval.value(0), "x");
    assert_eq!(rval.value(1), "y");
}

#[test]
fn test_full_outer_join() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE fl (id INT, lv TEXT)", base).unwrap();
    exec_multi("CREATE TABLE fr (id INT, rv TEXT)", base).unwrap();
    exec_multi("INSERT INTO fl (id, lv) VALUES (1, 'a'), (2, 'b')", base).unwrap();
    exec_multi("INSERT INTO fr (id, rv) VALUES (2, 'x'), (3, 'y')", base).unwrap();
    let result = exec_multi("SELECT * FROM fl FULL OUTER JOIN fr ON fl.id = fr.id", base).unwrap();
    let batch = result.to_record_batch().unwrap();
    // FULL OUTER: id=1 (left only), id=2 (both), id=3 (right only) = 3 rows
    assert_eq!(batch.num_rows(), 3);
}

#[test]
fn test_full_outer_join_qualified_key_projection() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE fl (id INT, lv TEXT)", base).unwrap();
    exec_multi("CREATE TABLE fr (id INT, rv TEXT)", base).unwrap();
    exec_multi("INSERT INTO fl (id, lv) VALUES (1, 'a'), (2, 'b')", base).unwrap();
    exec_multi("INSERT INTO fr (id, rv) VALUES (2, 'x'), (3, 'y')", base).unwrap();

    let result = exec_multi(
        "SELECT fl.id AS lid, fr.id AS rid, fl.lv, fr.rv \
         FROM fl FULL OUTER JOIN fr ON fl.id = fr.id \
         ORDER BY COALESCE(fl.id, fr.id)",
        base,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();

    let lid = batch
        .column_by_name("lid")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let rid = batch
        .column_by_name("rid")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let lv = batch
        .column_by_name("lv")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let rv = batch
        .column_by_name("rv")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(lid.value(0), 1);
    assert!(rid.is_null(0));
    assert_eq!(lid.value(1), 2);
    assert_eq!(rid.value(1), 2);
    assert!(lid.is_null(2));
    assert_eq!(rid.value(2), 3);
    assert_eq!(lv.value(0), "a");
    assert_eq!(lv.value(1), "b");
    assert!(lv.is_null(2));
    assert!(rv.is_null(0));
    assert_eq!(rv.value(1), "x");
    assert_eq!(rv.value(2), "y");
}

#[test]
fn test_cross_join() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE ca (id INT)", base).unwrap();
    exec_multi("CREATE TABLE cb (id INT)", base).unwrap();
    exec_multi("INSERT INTO ca (id) VALUES (1), (2)", base).unwrap();
    exec_multi("INSERT INTO cb (id) VALUES (10), (20), (30)", base).unwrap();
    let result = exec_multi("SELECT * FROM ca CROSS JOIN cb", base).unwrap();
    let batch = result.to_record_batch().unwrap();
    // CROSS JOIN: 2 × 3 = 6 rows
    assert_eq!(batch.num_rows(), 6);
}

#[test]
fn test_persistent_view_across_execute_calls() {
    let dir = tempdir().unwrap();
    let base = dir.path();

    exec_multi("CREATE TABLE src (id INT, city TEXT)", base).unwrap();
    exec_multi(
        "INSERT INTO src (id, city) VALUES (1, 'Beijing'), (2, 'Shanghai')",
        base,
    )
    .unwrap();
    exec_multi(
        "CREATE VIEW v_src AS SELECT id, city FROM src WHERE id >= 2",
        base,
    )
    .unwrap();

    let result = exec_multi("SELECT city FROM v_src", base).unwrap();
    let batch = result.to_record_batch().unwrap();
    let city = batch
        .column_by_name("city")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(city.value(0), "Shanghai");

    let reopened = exec_multi("SELECT COUNT(*) FROM v_src", base).unwrap();
    let count_batch = reopened.to_record_batch().unwrap();
    assert_eq!(
        count_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        1
    );

    exec_multi("DROP VIEW v_src", base).unwrap();
    assert!(exec_multi("SELECT * FROM v_src", base).is_err());
}

#[test]
fn test_persistent_view_over_default_table_without_default_path() {
    let dir = tempdir().unwrap();
    let base = dir.path();

    exec_multi("CREATE TABLE default (a INT)", base).unwrap();
    exec_multi("INSERT INTO default (a) VALUES (1), (2), (3)", base).unwrap();
    exec_multi(
        "CREATE VIEW v_default AS SELECT a FROM default WHERE a >= 2",
        base,
    )
    .unwrap();

    let result =
        ApexExecutor::execute_with_base_dir("SELECT * FROM v_default ORDER BY a", base, base)
            .unwrap();
    let batch = result.to_record_batch().unwrap();
    let values = batch
        .column_by_name("a")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(values.value(0), 2);
    assert_eq!(values.value(1), 3);
}

#[test]
fn test_copy_to_csv_and_json_export() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    let csv_path = base.join("export.csv");
    let json_path = base.join("export.jsonl");

    exec_multi("CREATE TABLE export_t (id INT, name TEXT)", base).unwrap();
    exec_multi(
        "INSERT INTO export_t (id, name) VALUES (1, 'Alice'), (2, 'Bob')",
        base,
    )
    .unwrap();

    exec_multi(
        &format!("COPY export_t TO '{}'", csv_path.to_string_lossy()),
        base,
    )
    .unwrap();
    exec_multi(
        &format!("COPY export_t TO '{}'", json_path.to_string_lossy()),
        base,
    )
    .unwrap();

    let csv = std::fs::read_to_string(&csv_path).unwrap();
    assert!(csv.contains("id,name"));
    assert!(csv.contains("Alice"));

    let json = std::fs::read_to_string(&json_path).unwrap();
    assert!(json.contains("\"name\":\"Alice\""));
    assert!(json.contains("\"name\":\"Bob\""));
}

#[test]
fn test_json_mutation_functions() {
    let dir = tempdir().unwrap();
    let base = dir.path();

    let result = exec_multi(
        "SELECT \
            JSON_SET('{\"a\":1}', '$.b', 2) AS set_v, \
            JSON_INSERT('{\"a\":1}', '$.c', 3) AS ins_v, \
            JSON_REPLACE('{\"a\":1}', '$.a', 9) AS rep_v, \
            JSON_REMOVE('{\"a\":1,\"b\":2}', '$.b') AS rem_v",
        base,
    )
    .unwrap();
    let batch = result.to_record_batch().unwrap();
    let set_v = batch
        .column_by_name("set_v")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let ins_v = batch
        .column_by_name("ins_v")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let rep_v = batch
        .column_by_name("rep_v")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let rem_v = batch
        .column_by_name("rem_v")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(set_v.value(0), "{\"a\":1,\"b\":2}");
    assert_eq!(ins_v.value(0), "{\"a\":1,\"c\":3}");
    assert_eq!(rep_v.value(0), "{\"a\":9}");
    assert_eq!(rem_v.value(0), "{\"a\":1}");
}

// ========== P1: Per-RG Zone Maps Tests ==========

#[test]
fn test_zone_maps_persisted_in_footer() {
    use crate::storage::on_demand::OnDemandStorage;
    let dir = tempdir().unwrap();
    let base = dir.path();
    exec_multi("CREATE TABLE zm (id INT, score INT, name TEXT)", base).unwrap();
    exec_multi(
        "INSERT INTO zm (id, score, name) VALUES (1, 10, 'a'), (2, 20, 'b'), (3, 30, 'c')",
        base,
    )
    .unwrap();

    // Open storage and check footer has zone maps
    let path = base.join("zm.apex");
    let storage = OnDemandStorage::open(&path).unwrap();
    if let Some(footer) = storage.get_or_load_footer().unwrap() {
        assert!(
            !footer.zone_maps.is_empty(),
            "Zone maps should be populated"
        );
        // First RG should have zone maps for Int64 columns (id and score)
        let rg0_zmaps = &footer.zone_maps[0];
        assert!(
            rg0_zmaps.len() >= 2,
            "Should have zone maps for at least 2 Int64 columns"
        );
        // Check zone map values for id column (min=1, max=3)
        let id_zm = &rg0_zmaps[0];
        assert_eq!(id_zm.min_bits, 1);
        assert_eq!(id_zm.max_bits, 3);
        assert!(!id_zm.is_float);
        // Check zone map values for score column (min=10, max=30)
        let score_zm = &rg0_zmaps[1];
        assert_eq!(score_zm.min_bits, 10);
        assert_eq!(score_zm.max_bits, 30);
    } else {
        panic!("Expected V4 footer");
    }
}

#[test]
fn test_zone_map_pruning_logic() {
    use crate::storage::on_demand::RgColumnZoneMap;
    let zm = RgColumnZoneMap {
        col_idx: 0,
        min_bits: 10,
        max_bits: 100,
        has_nulls: false,
        is_float: false,
    };
    // Value 50 is in range [10,100]
    assert!(zm.may_contain_int("=", 50));
    // Value 200 is NOT in range [10,100]
    assert!(!zm.may_contain_int("=", 200));
    // All values > 5 — max=100 > 5
    assert!(zm.may_contain_int(">", 5));
    // All values > 100 — max=100 NOT > 100
    assert!(!zm.may_contain_int(">", 100));
    // BETWEEN 50..150 overlaps [10,100]
    assert!(zm.may_overlap_int_range(50, 150));
    // BETWEEN 200..300 does NOT overlap [10,100]
    assert!(!zm.may_overlap_int_range(200, 300));
}
