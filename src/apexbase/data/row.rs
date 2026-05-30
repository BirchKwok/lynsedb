//! Row representation for database records

use super::Value;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A row of data in the database
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Row {
    /// The row ID (primary key)
    pub id: u64,
    /// The data fields
    pub fields: HashMap<String, Value>,
}

impl Row {
    /// Create a new empty row with the given ID
    pub fn new(id: u64) -> Self {
        Self {
            id,
            fields: HashMap::new(),
        }
    }

    /// Create a row from a HashMap
    pub fn from_fields(id: u64, fields: HashMap<String, Value>) -> Self {
        Self { id, fields }
    }

    /// Get a field value by name
    pub fn get(&self, field: &str) -> Option<&Value> {
        self.fields.get(field)
    }

    /// Set a field value
    pub fn set(&mut self, field: impl Into<String>, value: impl Into<Value>) {
        self.fields.insert(field.into(), value.into());
    }

    /// Check if a field exists
    pub fn has_field(&self, field: &str) -> bool {
        self.fields.contains_key(field)
    }

    /// Get all field names
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of fields
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Check if the row has no fields
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        bincode::deserialize(bytes).ok()
    }

    /// Convert to JSON value (for Python interop)
    pub fn to_json(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert("_id".to_string(), serde_json::json!(self.id));
        for (key, value) in &self.fields {
            map.insert(key.clone(), value.to_json_value());
        }
        serde_json::Value::Object(map)
    }

    /// Create from JSON value (for Python interop)
    pub fn from_json(id: u64, json: &serde_json::Value) -> Option<Self> {
        if let serde_json::Value::Object(map) = json {
            let mut fields = HashMap::new();
            for (key, value) in map {
                if key != "_id" {
                    fields.insert(key.clone(), Value::infer_from_python_value(value));
                }
            }
            Some(Self { id, fields })
        } else {
            None
        }
    }

    /// Merge another row into this one (update fields)
    pub fn merge(&mut self, other: &Row) {
        for (key, value) in &other.fields {
            self.fields.insert(key.clone(), value.clone());
        }
    }

    /// Remove a field
    pub fn remove(&mut self, field: &str) -> Option<Value> {
        self.fields.remove(field)
    }

    /// Iterate over fields
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.fields.iter()
    }
}

impl Default for Row {
    fn default() -> Self {
        Self::new(0)
    }
}

impl IntoIterator for Row {
    type Item = (String, Value);
    type IntoIter = std::collections::hash_map::IntoIter<String, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.fields.into_iter()
    }
}

impl<'a> IntoIterator for &'a Row {
    type Item = (&'a String, &'a Value);
    type IntoIter = std::collections::hash_map::Iter<'a, String, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.fields.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_operations() {
        let mut row = Row::new(1);
        row.set("name", "John");
        row.set("age", Value::Int64(30));

        assert_eq!(row.get("name"), Some(&Value::String("John".into())));
        assert_eq!(row.get("age"), Some(&Value::Int64(30)));
        assert!(row.has_field("name"));
        assert!(!row.has_field("email"));
    }

    #[test]
    fn test_row_serialization() {
        let mut row = Row::new(1);
        row.set("name", "John");
        row.set("age", Value::Int64(30));

        let bytes = row.to_bytes();
        let restored = Row::from_bytes(&bytes).unwrap();

        assert_eq!(row, restored);
    }

    #[test]
    fn test_row_json() {
        let mut row = Row::new(1);
        row.set("name", "John");
        row.set("age", Value::Int64(30));

        let json = row.to_json();
        let restored = Row::from_json(1, &json).unwrap();

        assert_eq!(row, restored);
    }
}
