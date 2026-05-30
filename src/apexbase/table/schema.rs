//! Schema definition for tables

use crate::data::{DataType, Value};
use crate::{ApexError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Column definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnDef {
    /// Column ID (unique within a table)
    pub id: u16,
    /// Column name
    pub name: String,
    /// Data type
    pub data_type: DataType,
    /// Whether the column can contain null values
    pub nullable: bool,
    /// Whether this column is indexed
    pub indexed: bool,
    /// Default value (optional)
    pub default_value: Option<Value>,
    /// Ordinal position in the table
    pub ordinal_position: u16,
}

impl ColumnDef {
    /// Create a new column definition
    pub fn new(id: u16, name: impl Into<String>, data_type: DataType) -> Self {
        Self {
            id,
            name: name.into(),
            data_type,
            nullable: true,
            indexed: false,
            default_value: None,
            ordinal_position: id,
        }
    }

    /// Set nullable flag
    pub fn nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    /// Set indexed flag
    pub fn indexed(mut self, indexed: bool) -> Self {
        self.indexed = indexed;
        self
    }

    /// Set ordinal position
    pub fn position(mut self, pos: u16) -> Self {
        self.ordinal_position = pos;
        self
    }
}

/// Table schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    /// Columns by name
    columns: HashMap<String, ColumnDef>,
    /// Column order
    column_order: Vec<String>,
    /// Next column ID
    next_column_id: u16,
}

impl Schema {
    /// Create a new schema with default _id column
    pub fn new() -> Self {
        let mut schema = Self {
            columns: HashMap::new(),
            column_order: Vec::new(),
            next_column_id: 1,
        };

        // Add default _id column
        let id_col = ColumnDef::new(0, "_id", DataType::Int64)
            .nullable(false)
            .indexed(true);
        schema.columns.insert("_id".to_string(), id_col);
        schema.column_order.push("_id".to_string());

        schema
    }

    /// Add a column
    pub fn add_column(&mut self, name: &str, data_type: DataType) -> Result<()> {
        if self.columns.contains_key(name) {
            return Err(ApexError::ColumnExists(name.to_string()));
        }

        let col =
            ColumnDef::new(self.next_column_id, name, data_type).position(self.next_column_id);

        self.columns.insert(name.to_string(), col);
        self.column_order.push(name.to_string());
        self.next_column_id += 1;

        Ok(())
    }

    /// Remove a column
    pub fn remove_column(&mut self, name: &str) -> Result<()> {
        if name == "_id" {
            return Err(ApexError::CannotModifyIdColumn);
        }

        if !self.columns.contains_key(name) {
            return Err(ApexError::ColumnNotFound(name.to_string()));
        }

        self.columns.remove(name);
        self.column_order.retain(|n| n != name);

        Ok(())
    }

    /// Rename a column
    pub fn rename_column(&mut self, old_name: &str, new_name: &str) -> Result<()> {
        if old_name == "_id" {
            return Err(ApexError::CannotModifyIdColumn);
        }

        if !self.columns.contains_key(old_name) {
            return Err(ApexError::ColumnNotFound(old_name.to_string()));
        }

        if self.columns.contains_key(new_name) {
            return Err(ApexError::ColumnExists(new_name.to_string()));
        }

        if let Some(mut col) = self.columns.remove(old_name) {
            col.name = new_name.to_string();
            self.columns.insert(new_name.to_string(), col);

            // Update column order
            if let Some(pos) = self.column_order.iter().position(|n| n == old_name) {
                self.column_order[pos] = new_name.to_string();
            }
        }

        Ok(())
    }

    /// Get a column definition
    pub fn get_column(&self, name: &str) -> Option<&ColumnDef> {
        self.columns.get(name)
    }

    /// Check if a column exists
    pub fn has_column(&self, name: &str) -> bool {
        self.columns.contains_key(name)
    }

    /// Get all column names in order
    pub fn column_names(&self) -> Vec<String> {
        self.column_order.clone()
    }

    /// Get the number of columns
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Check if empty (only has _id)
    pub fn is_empty(&self) -> bool {
        self.columns.len() <= 1
    }

    /// Iterate over columns
    pub fn columns(&self) -> impl Iterator<Item = &ColumnDef> {
        self.column_order
            .iter()
            .filter_map(|name| self.columns.get(name))
    }

    /// Update schema from data (add missing columns)
    pub fn update_from_data(&mut self, data: &HashMap<String, crate::data::Value>) {
        for (key, value) in data {
            if key != "_id" && !self.columns.contains_key(key) {
                let data_type = value.data_type();
                let _ = self.add_column(key, data_type);
            }
        }
    }
}

impl Default for Schema {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_operations() {
        let mut schema = Schema::new();

        // Should have _id column
        assert!(schema.has_column("_id"));
        assert_eq!(schema.len(), 1);

        // Add columns
        schema.add_column("name", DataType::String).unwrap();
        schema.add_column("age", DataType::Int64).unwrap();

        assert!(schema.has_column("name"));
        assert!(schema.has_column("age"));
        assert_eq!(schema.len(), 3);

        // Get column
        let name_col = schema.get_column("name").unwrap();
        assert_eq!(name_col.data_type, DataType::String);

        // Rename column
        schema.rename_column("name", "full_name").unwrap();
        assert!(!schema.has_column("name"));
        assert!(schema.has_column("full_name"));

        // Remove column
        schema.remove_column("age").unwrap();
        assert!(!schema.has_column("age"));
    }

    #[test]
    fn test_cannot_modify_id() {
        let mut schema = Schema::new();

        assert!(schema.remove_column("_id").is_err());
        assert!(schema.rename_column("_id", "id").is_err());
    }
}
