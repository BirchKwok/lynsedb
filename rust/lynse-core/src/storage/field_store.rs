//! ApexBase-based structured metadata/field storage.
//!
//! Replaces the Python `FieldsCache` with ApexBase tables for:
//! - Storing per-vector metadata fields
//! - SQL-like queries for field filtering
//! - Batch insert/retrieve operations
//! - Schema inference from first insert

use crate::error::{LynseError, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// ApexBase-backed field store for per-vector metadata.
///
/// Each vector ID maps to a row of arbitrary metadata fields stored in ApexBase.
/// Supports SQL-like filtering used by `search_filter` in the Python API.
pub struct FieldStore {
    /// Path to the ApexBase database directory
    #[allow(dead_code)]
    db_path: PathBuf,
    /// ApexBase database handle
    db: apexbase::embedded::ApexDB,
    /// Table name for fields
    table_name: String,
    /// Auto-increment counter for internal IDs
    next_id: Arc<RwLock<u64>>,
}

impl FieldStore {
    /// Open or create a field store at the given path.
    pub fn new(db_path: &Path, table_name: &str) -> Result<Self> {
        std::fs::create_dir_all(db_path)?;

        let db = apexbase::embedded::ApexDB::open(db_path)
            .map_err(|e| LynseError::ApexBase(format!("Failed to open ApexBase: {}", e)))?;

        // Try to open existing table or create new one
        let _ = db.create_table(table_name);

        // Determine next_id from existing data
        let next_id = {
            let table = db
                .table(table_name)
                .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;
            let count = table.count()
                .map_err(|e| LynseError::ApexBase(format!("Count error: {}", e)))?;
            count as u64
        };

        Ok(Self {
            db_path: db_path.to_path_buf(),
            db,
            table_name: table_name.to_string(),
            next_id: Arc::new(RwLock::new(next_id)),
        })
    }

    /// Store a single record with its fields.
    /// Returns the assigned external ID.
    pub fn store(&self, fields: &HashMap<String, serde_json::Value>) -> Result<u64> {
        let table = self
            .db
            .table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;

        let mut row: HashMap<String, apexbase::data::Value> = HashMap::new();

        // Convert JSON values to ApexBase values
        for (key, value) in fields {
            row.insert(key.clone(), json_to_apex_value(value));
        }

        // Add external_id field
        let ext_id = {
            let mut next = self.next_id.write();
            let id = *next;
            *next += 1;
            id
        };
        row.insert(
            "external_id".to_string(),
            apexbase::data::Value::Int64(ext_id as i64),
        );

        table
            .insert(row)
            .map_err(|e| LynseError::ApexBase(format!("Insert error: {}", e)))?;

        Ok(ext_id)
    }

    /// Batch store multiple records.
    /// Returns the list of assigned external IDs.
    pub fn batch_store(
        &self,
        fields_list: &[HashMap<String, serde_json::Value>],
    ) -> Result<Vec<u64>> {
        let table = self
            .db
            .table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;

        let mut ids = Vec::with_capacity(fields_list.len());

        // Batch insert
        let mut rows: Vec<HashMap<String, apexbase::data::Value>> =
            Vec::with_capacity(fields_list.len());

        for fields in fields_list {
            let mut row: HashMap<String, apexbase::data::Value> = HashMap::new();

            for (key, value) in fields {
                row.insert(key.clone(), json_to_apex_value(value));
            }

            let ext_id = {
                let mut next = self.next_id.write();
                let id = *next;
                *next += 1;
                id
            };
            row.insert(
                "external_id".to_string(),
                apexbase::data::Value::Int64(ext_id as i64),
            );

            ids.push(ext_id);
            rows.push(row);
        }

        // Use batch insert for better performance
        for row in rows {
            table
                .insert(row)
                .map_err(|e| LynseError::ApexBase(format!("Batch insert error: {}", e)))?;
        }

        Ok(ids)
    }

    /// Query fields using a SQL-like filter expression.
    /// Returns matching external IDs.
    pub fn query(&self, filter_expr: &str) -> Result<Vec<u64>> {
        let table = self
            .db
            .table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;

        // Execute SQL query through ApexBase
        let sql = format!(
            "SELECT external_id FROM {} WHERE {}",
            self.table_name, filter_expr
        );

        let result_set = table
            .execute(&sql)
            .map_err(|e| LynseError::ApexBase(format!("Query error: {}", e)))?;

        let rows = result_set
            .to_rows()
            .map_err(|e| LynseError::ApexBase(format!("Result conversion error: {}", e)))?;

        let mut ids = Vec::new();
        for row in rows {
            if let Some(apexbase::data::Value::Int64(id)) = row.get("external_id") {
                ids.push(*id as u64);
            }
        }

        Ok(ids)
    }

    /// Retrieve a single record by external ID.
    pub fn retrieve(&self, external_id: u64) -> Result<HashMap<String, serde_json::Value>> {
        let table = self
            .db
            .table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;

        let sql = format!(
            "SELECT * FROM {} WHERE external_id = {}",
            self.table_name, external_id
        );

        let result_set = table
            .execute(&sql)
            .map_err(|e| LynseError::ApexBase(format!("Query error: {}", e)))?;

        let rows = result_set
            .to_rows()
            .map_err(|e| LynseError::ApexBase(format!("Result conversion error: {}", e)))?;

        if let Some(row) = rows.into_iter().next() {
            let mut fields = HashMap::new();
            for (key, value) in row.iter() {
                if key != "external_id" && key != "_id" {
                    fields.insert(key.clone(), apex_value_to_json(value));
                }
            }
            Ok(fields)
        } else {
            Ok(HashMap::new())
        }
    }

    /// Retrieve multiple records by external IDs.
    pub fn retrieve_many(
        &self,
        external_ids: &[u64],
    ) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        let mut results = Vec::with_capacity(external_ids.len());
        for &id in external_ids {
            results.push(self.retrieve(id)?);
        }
        Ok(results)
    }

    /// List all field names (column names) in the store.
    pub fn list_fields(&self) -> Result<Vec<String>> {
        let table = self
            .db
            .table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;

        let result_set = table
            .execute(&format!("SELECT * FROM {} LIMIT 1", self.table_name))
            .map_err(|e| LynseError::ApexBase(format!("Query error: {}", e)))?;

        let rows = result_set
            .to_rows()
            .map_err(|e| LynseError::ApexBase(format!("Result conversion error: {}", e)))?;

        if let Some(row) = rows.into_iter().next() {
            Ok(row
                .keys()
                .filter(|k| *k != "_id" && *k != "external_id")
                .cloned()
                .collect())
        } else {
            Ok(vec![])
        }
    }

    /// Get total record count.
    pub fn count(&self) -> Result<u64> {
        let table = self
            .db
            .table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;
        let count = table
            .count()
            .map_err(|e| LynseError::ApexBase(format!("Count error: {}", e)))?;
        Ok(count as u64)
    }

    /// Drop the field store table.
    pub fn drop(&self) -> Result<()> {
        self.db
            .drop_table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Drop table error: {}", e)))?;
        Ok(())
    }
}

/// Convert a serde_json::Value to an ApexBase Value.
fn json_to_apex_value(v: &serde_json::Value) -> apexbase::data::Value {
    match v {
        serde_json::Value::Null => apexbase::data::Value::Null,
        serde_json::Value::Bool(b) => apexbase::data::Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                apexbase::data::Value::Int64(i)
            } else if let Some(f) = n.as_f64() {
                apexbase::data::Value::Float64(f)
            } else {
                apexbase::data::Value::String(n.to_string())
            }
        }
        serde_json::Value::String(s) => apexbase::data::Value::String(s.clone()),
        serde_json::Value::Array(arr) => {
            // Store arrays as JSON strings
            apexbase::data::Value::String(serde_json::to_string(arr).unwrap_or_default())
        }
        serde_json::Value::Object(obj) => {
            // Store objects as JSON strings
            apexbase::data::Value::String(serde_json::to_string(obj).unwrap_or_default())
        }
    }
}

/// Convert an ApexBase Value back to serde_json::Value.
fn apex_value_to_json(v: &apexbase::data::Value) -> serde_json::Value {
    match v {
        apexbase::data::Value::Null => serde_json::Value::Null,
        apexbase::data::Value::Bool(b) => serde_json::Value::Bool(*b),
        apexbase::data::Value::Int64(i) => serde_json::json!(*i),
        apexbase::data::Value::Float64(f) => serde_json::json!(*f),
        apexbase::data::Value::String(s) => {
            // Try to parse as JSON first (for arrays/objects stored as strings)
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                if parsed.is_array() || parsed.is_object() {
                    return parsed;
                }
            }
            serde_json::Value::String(s.clone())
        }
        _ => serde_json::Value::String(format!("{:?}", v)),
    }
}
