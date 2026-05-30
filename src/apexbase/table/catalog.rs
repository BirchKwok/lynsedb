//! Table catalog management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A table entry in the catalog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableEntry {
    /// Table ID
    pub id: u32,
    /// Table name
    pub name: String,
    /// Row count
    pub row_count: u64,
    /// Creation timestamp
    pub created_at: i64,
    /// Last modified timestamp
    pub modified_at: i64,
    /// Data offset in file
    pub data_offset: u64,
    /// Flags
    pub flags: u32,
}

impl TableEntry {
    /// Create a new table entry
    pub fn new(id: u32, name: &str) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id,
            name: name.to_string(),
            row_count: 0,
            created_at: now,
            modified_at: now,
            data_offset: 0,
            flags: 0,
        }
    }

    /// Update modified timestamp
    pub fn touch(&mut self) {
        self.modified_at = chrono::Utc::now().timestamp();
    }
}

/// Table catalog - manages metadata for all tables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableCatalog {
    /// Table entries by name
    entries: HashMap<String, TableEntry>,
    /// Next table ID
    next_id: u32,
}

impl TableCatalog {
    /// Create a new empty catalog
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            next_id: 1,
        }
    }

    /// Add a table entry
    pub fn add(&mut self, entry: TableEntry) {
        self.entries.insert(entry.name.clone(), entry);
        self.next_id += 1;
    }

    /// Remove a table entry
    pub fn remove(&mut self, name: &str) -> Option<TableEntry> {
        self.entries.remove(name)
    }

    /// Get a table entry
    pub fn get(&self, name: &str) -> Option<&TableEntry> {
        self.entries.get(name)
    }

    /// Get a mutable table entry
    pub fn get_mut(&mut self, name: &str) -> Option<&mut TableEntry> {
        self.entries.get_mut(name)
    }

    /// Check if a table exists
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Get all table names
    pub fn table_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.entries.keys().cloned().collect();
        names.sort();
        names
    }

    /// Get all table entries
    pub fn tables(&self) -> impl Iterator<Item = &TableEntry> {
        self.entries.values()
    }

    /// Get the next table ID
    pub fn next_id(&self) -> u32 {
        self.next_id
    }

    /// Get the number of tables
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for TableCatalog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_operations() {
        let mut catalog = TableCatalog::new();

        let entry1 = TableEntry::new(1, "users");
        let entry2 = TableEntry::new(2, "orders");

        catalog.add(entry1);
        catalog.add(entry2);

        assert!(catalog.contains("users"));
        assert!(catalog.contains("orders"));
        assert!(!catalog.contains("products"));

        let names = catalog.table_names();
        assert_eq!(names, vec!["orders", "users"]);

        catalog.remove("users");
        assert!(!catalog.contains("users"));
    }
}
