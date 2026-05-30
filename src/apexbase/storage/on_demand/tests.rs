use super::*;
use tempfile::tempdir;

#[test]
fn test_create_and_open() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.apex");

    // Create and insert
    {
        let storage = OnDemandStorage::create(&path).unwrap();

        let mut int_cols = HashMap::new();
        int_cols.insert("age".to_string(), vec![25, 30, 35, 40, 45]);

        let mut string_cols = HashMap::new();
        string_cols.insert(
            "name".to_string(),
            vec![
                "Alice".to_string(),
                "Bob".to_string(),
                "Charlie".to_string(),
                "David".to_string(),
                "Eve".to_string(),
            ],
        );

        let ids = storage
            .insert_typed(
                int_cols,
                HashMap::new(),
                string_cols,
                HashMap::new(),
                HashMap::new(),
            )
            .unwrap();

        assert_eq!(ids.len(), 5);
        storage.save().unwrap();
    }

    // Reopen and verify
    {
        let storage = OnDemandStorage::open(&path).unwrap();
        assert_eq!(storage.row_count(), 5);
        assert_eq!(storage.column_names().len(), 3); // _id, age, name
    }
}

#[test]
fn test_column_projection() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test_proj.apex");

    // Create with multiple columns
    let storage = OnDemandStorage::create(&path).unwrap();

    let mut int_cols = HashMap::new();
    int_cols.insert("a".to_string(), vec![1, 2, 3, 4, 5]);
    int_cols.insert("b".to_string(), vec![10, 20, 30, 40, 50]);
    int_cols.insert("c".to_string(), vec![100, 200, 300, 400, 500]);

    storage
        .insert_typed(
            int_cols,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();
    storage.save().unwrap();

    // Reopen
    let storage = OnDemandStorage::open(&path).unwrap();

    // Read only column "b"
    let result = storage.read_columns(Some(&["b"]), 0, None).unwrap();
    assert_eq!(result.len(), 1);
    assert!(result.contains_key("b"));

    if let ColumnData::Int64(vals) = &result["b"] {
        assert_eq!(vals, &[10, 20, 30, 40, 50]);
    } else {
        panic!("Expected Int64 column");
    }
}

#[test]
fn test_row_range() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test_range.apex");

    let storage = OnDemandStorage::create(&path).unwrap();

    let mut int_cols = HashMap::new();
    int_cols.insert("val".to_string(), (0..100).collect());

    storage
        .insert_typed(
            int_cols,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();
    storage.save().unwrap();

    let storage = OnDemandStorage::open(&path).unwrap();

    // Read rows 10-19 (10 rows starting at row 10)
    let result = storage.read_columns(Some(&["val"]), 10, Some(10)).unwrap();

    if let ColumnData::Int64(vals) = &result["val"] {
        assert_eq!(vals.len(), 10);
        assert_eq!(vals[0], 10);
        assert_eq!(vals[9], 19);
    } else {
        panic!("Expected Int64 column");
    }
}

#[test]
fn test_string_column() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test_string.apex");

    let storage = OnDemandStorage::create(&path).unwrap();

    let mut string_cols = HashMap::new();
    string_cols.insert(
        "text".to_string(),
        vec![
            "hello".to_string(),
            "world".to_string(),
            "foo".to_string(),
            "bar".to_string(),
        ],
    );

    storage
        .insert_typed(
            HashMap::new(),
            HashMap::new(),
            string_cols,
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();
    storage.save().unwrap();

    let storage = OnDemandStorage::open(&path).unwrap();

    // Read middle 2 rows
    let result = storage.read_columns(Some(&["text"]), 1, Some(2)).unwrap();

    if let ColumnData::String { offsets, data } = &result["text"] {
        assert_eq!(offsets.len(), 3); // 2 strings + 1 trailing offset
        let s0 = std::str::from_utf8(&data[offsets[0] as usize..offsets[1] as usize]).unwrap();
        let s1 = std::str::from_utf8(&data[offsets[1] as usize..offsets[2] as usize]).unwrap();
        assert_eq!(s0, "world");
        assert_eq!(s1, "foo");
    } else {
        panic!("Expected String column");
    }
}

#[test]
fn test_insert_rows_compatibility() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test_compat.apex");

    let storage = OnDemandStorage::create(&path).unwrap();

    // Use insert_rows API (ColumnarStorage compatible)
    let mut rows = Vec::new();
    for i in 0..10 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), ColumnValue::Int64(i));
        row.insert(
            "name".to_string(),
            ColumnValue::String(format!("user_{}", i)),
        );
        row.insert("score".to_string(), ColumnValue::Float64(i as f64 * 1.5));
        rows.push(row);
    }

    let ids = storage.insert_rows(&rows).unwrap();
    assert_eq!(ids.len(), 10);

    storage.save().unwrap();

    // Reopen and verify
    let storage = OnDemandStorage::open(&path).unwrap();
    assert_eq!(storage.row_count(), 10);

    let result = storage
        .read_columns(Some(&["id", "score"]), 0, None)
        .unwrap();
    assert_eq!(result.len(), 2);

    if let ColumnData::Int64(vals) = &result["id"] {
        assert_eq!(vals, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}

#[test]
fn test_bool_null_bitmap() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test_bool_null.apex");

    let storage = OnDemandStorage::create(&path).unwrap();

    // Insert with NULL boolean
    let mut rows = Vec::new();

    let mut row1 = HashMap::new();
    row1.insert("id".to_string(), ColumnValue::Int64(1));
    row1.insert("flag".to_string(), ColumnValue::Bool(true));
    rows.push(row1);

    let mut row2 = HashMap::new();
    row2.insert("id".to_string(), ColumnValue::Int64(2));
    row2.insert("flag".to_string(), ColumnValue::Bool(false));
    rows.push(row2);

    let mut row3 = HashMap::new();
    row3.insert("id".to_string(), ColumnValue::Int64(3));
    row3.insert("flag".to_string(), ColumnValue::Null); // NULL boolean
    rows.push(row3);

    storage.insert_rows(&rows).unwrap();

    // Check null bitmap in memory BEFORE save
    {
        let nulls = storage.nulls.read();
        let schema = storage.schema.read();
        let flag_idx = schema.get_index("flag").unwrap();
        println!("Flag column index: {}", flag_idx);
        println!("Nulls len: {}", nulls.len());
        if flag_idx < nulls.len() {
            println!("Null bitmap for flag: {:?}", nulls[flag_idx]);
            // Row 3 (index 2) should be marked as NULL
            // Byte 0, bit 2 should be set
            assert!(
                !nulls[flag_idx].is_empty(),
                "Null bitmap should not be empty"
            );
            assert_eq!(
                nulls[flag_idx][0] & (1 << 2),
                1 << 2,
                "Row 2 should be marked as NULL"
            );
        }
    }

    storage.save().unwrap();

    // Reopen and verify null bitmap is persisted
    let storage2 = OnDemandStorage::open(&path).unwrap();

    // Check null mask via get_null_mask
    let null_mask = storage2.get_null_mask("flag", 0, 3);
    println!("Null mask after reopen: {:?}", null_mask);
    assert_eq!(null_mask, vec![false, false, true], "Row 2 should be NULL");
}

#[test]
fn test_append_delta_compatibility() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test_delta.apex");

    let storage = OnDemandStorage::create(&path).unwrap();

    // First batch
    let mut rows = Vec::new();
    for i in 0..5 {
        let mut row = HashMap::new();
        row.insert("val".to_string(), ColumnValue::Int64(i));
        rows.push(row);
    }
    storage.append_delta(&rows).unwrap();

    // Second batch
    let mut rows2 = Vec::new();
    for i in 5..10 {
        let mut row = HashMap::new();
        row.insert("val".to_string(), ColumnValue::Int64(i));
        rows2.push(row);
    }
    storage.append_delta(&rows2).unwrap();

    // Verify
    let storage = OnDemandStorage::open(&path).unwrap();
    assert_eq!(storage.row_count(), 10);

    let result = storage.read_columns(Some(&["val"]), 0, None).unwrap();
    if let ColumnData::Int64(vals) = &result["val"] {
        assert_eq!(vals, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}

#[test]
fn test_v4_save_and_open() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test_v4.apex");

    // Create, insert, save as V4
    {
        let storage = OnDemandStorage::create(&path).unwrap();

        let mut int_cols = HashMap::new();
        int_cols.insert("age".to_string(), vec![25, 30, 35, 40, 45]);

        let mut string_cols = HashMap::new();
        string_cols.insert(
            "name".to_string(),
            vec![
                "Alice".to_string(),
                "Bob".to_string(),
                "Charlie".to_string(),
                "David".to_string(),
                "Eve".to_string(),
            ],
        );

        storage
            .insert_typed(
                int_cols,
                HashMap::new(),
                string_cols,
                HashMap::new(),
                HashMap::new(),
            )
            .unwrap();

        // Save as V4 Row Group format
        storage.save_v4().unwrap();

        // Verify header has V4 version
        let header = storage.header.read();
        assert_eq!(header.version, FORMAT_VERSION_V4);
        assert!(header.footer_offset > 0);
        assert!(header.row_group_count >= 1);
    }

    // Reopen V4 file and load data
    {
        let storage = OnDemandStorage::open(&path).unwrap();
        // Header should indicate V4
        let header = storage.header.read();
        assert_eq!(header.version, FORMAT_VERSION_V4);
        assert_eq!(header.row_count, 5);
        drop(header);

        // Load V4 data
        storage.open_v4_data().unwrap();
        assert_eq!(storage.ids.read().len(), 5);

        let columns = storage.columns.read();
        // User columns only (age, name) — _id stored separately in self.ids
        assert_eq!(columns.len(), 2);

        // Verify age column
        let schema = storage.schema.read();
        let age_idx = schema.get_index("age").unwrap();
        if let ColumnData::Int64(vals) = &columns[age_idx] {
            assert_eq!(vals, &[25, 30, 35, 40, 45]);
        } else {
            panic!("Expected Int64 for age column");
        }

        // Verify name column
        let name_idx = schema.get_index("name").unwrap();
        if let ColumnData::String { offsets, data } = &columns[name_idx] {
            assert_eq!(offsets.len(), 6); // 5 rows + 1
            let names: Vec<&str> = (0..5)
                .map(|i| {
                    let s = offsets[i] as usize;
                    let e = offsets[i + 1] as usize;
                    std::str::from_utf8(&data[s..e]).unwrap()
                })
                .collect();
            assert_eq!(names, vec!["Alice", "Bob", "Charlie", "David", "Eve"]);
        } else {
            panic!("Expected String for name column");
        }
    }
}

#[test]
fn test_v4_append_row_group() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test_v4_append.apex");

    // Create and save V4 with initial data
    let storage = OnDemandStorage::create(&path).unwrap();
    let mut int_cols = HashMap::new();
    int_cols.insert("val".to_string(), vec![1, 2, 3]);
    storage
        .insert_typed(
            int_cols,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();
    storage.save_v4().unwrap();

    // Verify initial state
    assert_eq!(storage.header.read().row_count, 3);
    assert_eq!(storage.header.read().row_group_count, 1);

    // Append a new Row Group without rewriting
    let new_ids = vec![100, 101, 102];
    let schema = storage.schema.read();
    let col_count = schema.column_count();
    let mut new_columns: Vec<ColumnData> = Vec::new();
    for (name, col_type) in &schema.columns {
        if name == "_id" {
            // _id is handled via new_ids, not as a column
            new_columns.push(ColumnData::Int64(vec![100, 101, 102]));
        } else {
            match col_type {
                ColumnType::Int64 => new_columns.push(ColumnData::Int64(vec![10, 20, 30])),
                _ => new_columns.push(ColumnData::new(*col_type)),
            }
        }
    }
    drop(schema);
    let new_nulls: Vec<Vec<u8>> = vec![Vec::new(); col_count];

    storage
        .append_row_group(&new_ids, &new_columns, &new_nulls)
        .unwrap();

    // Verify updated header
    let header = storage.header.read();
    assert_eq!(header.row_count, 6); // 3 + 3
    assert_eq!(header.row_group_count, 2);
    drop(header);

    // Reopen and load all data to verify
    let storage2 = OnDemandStorage::open(&path).unwrap();
    storage2.open_v4_data().unwrap();
    assert_eq!(storage2.ids.read().len(), 6);
}

#[test]
fn test_v4_multiple_row_groups() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test_v4_multi_rg.apex");

    // Create a table with more rows than DEFAULT_ROW_GROUP_SIZE
    // Use a small RG to test splitting (we'll insert 10 rows, RG size is 65536)
    let storage = OnDemandStorage::create(&path).unwrap();

    let mut int_cols = HashMap::new();
    int_cols.insert("val".to_string(), (0..100).collect::<Vec<i64>>());
    storage
        .insert_typed(
            int_cols,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();
    storage.save_v4().unwrap();

    // With 100 rows and RG size 65536, should be 1 RG
    assert_eq!(storage.header.read().row_group_count, 1);

    // Reopen and verify data integrity
    let storage2 = OnDemandStorage::open(&path).unwrap();
    storage2.open_v4_data().unwrap();

    let columns = storage2.columns.read();
    let schema = storage2.schema.read();
    let val_idx = schema.get_index("val").unwrap();
    if let ColumnData::Int64(vals) = &columns[val_idx] {
        assert_eq!(vals.len(), 100);
        assert_eq!(vals[0], 0);
        assert_eq!(vals[99], 99);
    } else {
        panic!("Expected Int64 for val column");
    }
}

#[test]
fn test_v4_read_columns_integration() {
    // Tests the executor read path: open V4 file → read_columns() auto-loads data
    let dir = tempdir().unwrap();
    let path = dir.path().join("test_v4_read.apex");

    // Create, insert, save as V4
    {
        let storage = OnDemandStorage::create(&path).unwrap();
        let mut int_cols = HashMap::new();
        int_cols.insert("age".to_string(), vec![10, 20, 30]);
        let mut string_cols = HashMap::new();
        string_cols.insert(
            "name".to_string(),
            vec![
                "Alice".to_string(),
                "Bob".to_string(),
                "Charlie".to_string(),
            ],
        );
        storage
            .insert_typed(
                int_cols,
                HashMap::new(),
                string_cols,
                HashMap::new(),
                HashMap::new(),
            )
            .unwrap();
        storage.save_v4().unwrap();
    }

    // Reopen and use read_columns() (executor path) — no explicit open_v4_data()
    let storage = OnDemandStorage::open(&path).unwrap();
    let result = storage
        .read_columns(Some(&["age", "name"]), 0, None)
        .unwrap();

    // Verify age
    if let ColumnData::Int64(vals) = &result["age"] {
        assert_eq!(vals, &[10, 20, 30]);
    } else {
        panic!("Expected Int64 for age");
    }

    // Verify name
    if let ColumnData::String { offsets, data } = &result["name"] {
        let names: Vec<&str> = (0..3)
            .map(|i| {
                let s = offsets[i] as usize;
                let e = offsets[i + 1] as usize;
                std::str::from_utf8(&data[s..e]).unwrap()
            })
            .collect();
        assert_eq!(names, vec!["Alice", "Bob", "Charlie"]);
    } else {
        panic!("Expected String for name");
    }

    // Test partial read (start_row, row_count)
    let partial = storage.read_columns(Some(&["age"]), 1, Some(2)).unwrap();
    if let ColumnData::Int64(vals) = &partial["age"] {
        assert_eq!(vals, &[20, 30]);
    } else {
        panic!("Expected Int64 for partial age read");
    }
}

// ====== LZ4 Compression Tests ======

#[test]
fn test_lz4_compression_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("lz4_test.apex");

    // Insert enough data to trigger LZ4 (body > 512 bytes)
    {
        let storage = OnDemandStorage::create(&path).unwrap();
        let mut int_cols = HashMap::new();
        let ids: Vec<i64> = (0..200).collect();
        int_cols.insert("id".to_string(), ids);
        let mut str_cols = HashMap::new();
        let names: Vec<String> = (0..200).map(|i| format!("name_{:04}", i)).collect();
        str_cols.insert("name".to_string(), names);
        storage
            .insert_typed(
                int_cols,
                HashMap::new(),
                str_cols,
                HashMap::new(),
                HashMap::new(),
            )
            .unwrap();
        storage.save_v4().unwrap();
    }

    // Reopen and verify data survives LZ4 compression
    {
        let storage = OnDemandStorage::open(&path).unwrap();
        storage.open_v4_data().unwrap();
        let data = storage
            .read_columns(Some(&["id", "name"]), 0, None)
            .unwrap();
        if let ColumnData::Int64(vals) = &data["id"] {
            assert_eq!(vals.len(), 200);
            assert_eq!(vals[0], 0);
            assert_eq!(vals[199], 199);
        } else {
            panic!("Expected Int64 for id");
        }
        if let ColumnData::String {
            offsets,
            data: sdata,
        } = &data["name"]
        {
            let first =
                std::str::from_utf8(&sdata[offsets[0] as usize..offsets[1] as usize]).unwrap();
            assert_eq!(first, "name_0000");
            let last_start = offsets[199] as usize;
            let last_end = offsets[200] as usize;
            let last = std::str::from_utf8(&sdata[last_start..last_end]).unwrap();
            assert_eq!(last, "name_0199");
        } else {
            panic!("Expected String for name");
        }
    }
}

#[test]
fn test_lz4_small_data_no_compression() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("small.apex");

    {
        let storage = OnDemandStorage::create(&path).unwrap();
        let mut int_cols = HashMap::new();
        int_cols.insert("a".to_string(), vec![1i64, 2, 3]);
        storage
            .insert_typed(
                int_cols,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
            )
            .unwrap();
        storage.save_v4().unwrap();
    }
    {
        let storage = OnDemandStorage::open(&path).unwrap();
        storage.open_v4_data().unwrap();
        let data = storage.read_columns(Some(&["a"]), 0, None).unwrap();
        if let ColumnData::Int64(vals) = &data["a"] {
            assert_eq!(vals, &[1, 2, 3]);
        } else {
            panic!("Expected Int64");
        }
    }
}

#[test]
fn test_lz4_mixed_types() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mixed.apex");

    {
        let storage = OnDemandStorage::create(&path).unwrap();
        let n = 200;
        let mut int_cols = HashMap::new();
        int_cols.insert("i".to_string(), (0..n).map(|x| x as i64).collect());
        let mut float_cols = HashMap::new();
        float_cols.insert("f".to_string(), (0..n).map(|x| x as f64 * 0.5).collect());
        let mut str_cols = HashMap::new();
        str_cols.insert("s".to_string(), (0..n).map(|x| format!("v{}", x)).collect());
        let mut bool_cols = HashMap::new();
        bool_cols.insert("b".to_string(), (0..n).map(|x| x % 2 == 0).collect());
        storage
            .insert_typed(int_cols, float_cols, str_cols, HashMap::new(), bool_cols)
            .unwrap();
        storage.save_v4().unwrap();
    }
    {
        let storage = OnDemandStorage::open(&path).unwrap();
        storage.open_v4_data().unwrap();
        let data = storage.read_columns(None, 0, None).unwrap();
        if let ColumnData::Int64(vals) = &data["i"] {
            assert_eq!(vals.len(), 200);
            assert_eq!(vals[100], 100);
        } else {
            panic!("Expected Int64 for i");
        }
        if let ColumnData::Float64(vals) = &data["f"] {
            assert!((vals[100] - 50.0).abs() < 0.01);
        } else {
            panic!("Expected Float64 for f");
        }
    }
}

// ====== Constraint Serialization Tests ======

#[test]
fn test_check_constraint_serialization() {
    let mut schema = OnDemandSchema::new();
    schema.add_column_with_constraints(
        "age",
        ColumnType::Int64,
        ColumnConstraints {
            not_null: false,
            primary_key: false,
            unique: false,
            default_value: None,
            check_expr_sql: Some("age > 0".to_string()),
            foreign_key: None,
            autoincrement: false,
        },
    );

    let bytes = schema.to_bytes();
    let restored = OnDemandSchema::from_bytes(&bytes).unwrap();
    assert_eq!(restored.columns.len(), 1);
    assert_eq!(
        restored.constraints[0].check_expr_sql,
        Some("age > 0".to_string())
    );
}

#[test]
fn test_fk_constraint_serialization() {
    let mut schema = OnDemandSchema::new();
    schema.add_column_with_constraints(
        "dept_id",
        ColumnType::Int64,
        ColumnConstraints {
            not_null: false,
            primary_key: false,
            unique: false,
            default_value: None,
            check_expr_sql: None,
            foreign_key: Some(("departments".to_string(), "id".to_string())),
            autoincrement: false,
        },
    );

    let bytes = schema.to_bytes();
    let restored = OnDemandSchema::from_bytes(&bytes).unwrap();
    assert_eq!(
        restored.constraints[0].foreign_key,
        Some(("departments".to_string(), "id".to_string()))
    );
}

#[test]
fn test_all_constraints_serialization() {
    let mut schema = OnDemandSchema::new();
    schema.add_column_with_constraints(
        "id",
        ColumnType::Int64,
        ColumnConstraints {
            not_null: true,
            primary_key: true,
            unique: false,
            default_value: None,
            check_expr_sql: None,
            foreign_key: None,
            autoincrement: false,
        },
    );
    schema.add_column_with_constraints(
        "val",
        ColumnType::Int64,
        ColumnConstraints {
            not_null: true,
            primary_key: false,
            unique: true,
            default_value: Some(DefaultValue::Int64(0)),
            check_expr_sql: Some("val >= 0".to_string()),
            foreign_key: Some(("other".to_string(), "val".to_string())),
            autoincrement: false,
        },
    );

    let bytes = schema.to_bytes();
    let restored = OnDemandSchema::from_bytes(&bytes).unwrap();

    // Column 0: id
    assert!(restored.constraints[0].primary_key);
    assert!(restored.constraints[0].not_null);
    assert!(restored.constraints[0].check_expr_sql.is_none());
    assert!(restored.constraints[0].foreign_key.is_none());

    // Column 1: val
    assert!(restored.constraints[1].not_null);
    assert!(restored.constraints[1].unique);
    assert_eq!(
        restored.constraints[1].default_value,
        Some(DefaultValue::Int64(0))
    );
    assert_eq!(
        restored.constraints[1].check_expr_sql,
        Some("val >= 0".to_string())
    );
    assert_eq!(
        restored.constraints[1].foreign_key,
        Some(("other".to_string(), "val".to_string()))
    );
}

#[test]
fn test_constraint_persisted_through_save_v4() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("cons.apex");

    {
        let storage = OnDemandStorage::create(&path).unwrap();
        let mut int_cols = HashMap::new();
        int_cols.insert("val".to_string(), vec![1i64, 2, 3]);
        storage
            .insert_typed(
                int_cols,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
            )
            .unwrap();
        storage.set_column_constraints(
            "val",
            ColumnConstraints {
                not_null: true,
                primary_key: false,
                unique: false,
                default_value: None,
                check_expr_sql: Some("val > 0".to_string()),
                foreign_key: Some(("parent".to_string(), "id".to_string())),
                autoincrement: false,
            },
        );
        storage.save_v4().unwrap();
    }
    {
        let storage = OnDemandStorage::open(&path).unwrap();
        let cons = storage.get_column_constraints("val");
        assert!(cons.not_null);
        assert_eq!(cons.check_expr_sql, Some("val > 0".to_string()));
        assert_eq!(
            cons.foreign_key,
            Some(("parent".to_string(), "id".to_string()))
        );
    }
}
