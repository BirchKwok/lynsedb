//! ApexBase-based structured metadata/field storage.
//!
//! Replaces the Python `FieldsCache` with ApexBase tables for:
//! - Storing per-vector metadata fields
//! - SQL-like queries for field filtering
//! - Batch insert/retrieve operations
//! - Schema inference from first insert

use crate::error::{LynseError, Result};
use arrow::array::{
    Array, BinaryArray, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, LargeBinaryArray, LargeStringArray, StringArray, UInt16Array,
    UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::DataType as ArrowDataType;
use arrow::record_batch::RecordBatch;
use parking_lot::RwLock;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Binary record size in the .apex_id_map file: [u64 ext_id][u64 apex_id] = 16 bytes.
const MAP_RECORD_SIZE: usize = 16;
/// apex_id value used to mark a deleted external_id (tombstone).
const TOMBSTONE: u64 = u64::MAX;

/// Maximum unique values per field to index (skip high-cardinality fields like unique IDs).
const MAX_INDEX_UNIQUE_VALUES: usize = 100_000;
/// Maximum total entries (key→id mappings) across all indexed fields.
const MAX_INDEX_TOTAL_ENTRIES: usize = 1_000_000;

/// Hash-able discriminated union for field values used as in-memory index keys.
#[derive(Hash, PartialEq, Eq, Clone)]
enum IndexKey {
    Int(i64),
    Float(u64), // f64 bits with sign-bit normalization for stable Eq/Hash
    Str(String),
    Bool(bool),
    Null,
}

impl IndexKey {
    fn from_json(v: &serde_json::Value) -> Self {
        match v {
            serde_json::Value::Null => IndexKey::Null,
            serde_json::Value::Bool(b) => IndexKey::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    IndexKey::Int(i)
                } else if let Some(f) = n.as_f64() {
                    IndexKey::Float(normalize_f64_bits(f))
                } else {
                    IndexKey::Str(n.to_string())
                }
            }
            serde_json::Value::String(s) => IndexKey::Str(s.clone()),
            // Arrays/objects are not indexed
            v => IndexKey::Str(v.to_string()),
        }
    }

    /// Parse a raw value token (from WHERE expression) into an IndexKey.
    fn from_token(s: &str) -> Option<Self> {
        // Double-quoted string
        if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
            return Some(IndexKey::Str(s[1..s.len() - 1].to_string()));
        }
        // Single-quoted string
        if s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2 {
            return Some(IndexKey::Str(s[1..s.len() - 1].to_string()));
        }
        // Boolean / null literals
        match s.to_lowercase().as_str() {
            "true" => return Some(IndexKey::Bool(true)),
            "false" => return Some(IndexKey::Bool(false)),
            "null" => return Some(IndexKey::Null),
            _ => {}
        }
        // Integer
        if let Ok(i) = s.parse::<i64>() {
            return Some(IndexKey::Int(i));
        }
        // Float
        if let Ok(f) = s.parse::<f64>() {
            return Some(IndexKey::Float(normalize_f64_bits(f)));
        }
        None
    }
}

#[inline]
fn normalize_f64_bits(f: f64) -> u64 {
    let bits = f.to_bits();
    if bits >> 63 == 0 {
        bits | (1u64 << 63)
    } else {
        !bits
    }
}

#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone)]
enum RangeKey {
    Number(u64),
    Text(String),
}

impl RangeKey {
    fn from_json(v: &serde_json::Value) -> Option<Self> {
        match v {
            serde_json::Value::Number(n) => n
                .as_f64()
                .filter(|v| v.is_finite())
                .map(|v| RangeKey::Number(normalize_f64_bits(v))),
            serde_json::Value::String(s) => Some(RangeKey::Text(s.clone())),
            _ => None,
        }
    }

    fn from_token(s: &str) -> Option<Self> {
        let token = unquote_token(s);
        if let Ok(v) = token.parse::<f64>() {
            if v.is_finite() {
                return Some(RangeKey::Number(normalize_f64_bits(v)));
            }
        }
        Some(RangeKey::Text(token.to_string()))
    }
}

#[derive(Clone, Copy)]
enum RangeOp {
    Lt,
    Lte,
    Gt,
    Gte,
}

/// Per-field index: uses Vec for dense integer ranges, HashMap for sparse/non-integer.
enum FieldIndexType {
    /// Dense integer range: values are in 0..len, direct index.
    /// Memory: len * sizeof(Vec<u64>) + actual data.
    DenseInt(Vec<Vec<u64>>),
    /// General case: HashMap for arbitrary values.
    Sparse(HashMap<IndexKey, Vec<u64>>),
}

impl FieldIndexType {
    fn get(&self, key: &IndexKey) -> Option<&Vec<u64>> {
        match self {
            FieldIndexType::DenseInt(vec) => {
                if let IndexKey::Int(i) = key {
                    let idx = *i as usize;
                    if idx < vec.len() && !vec[idx].is_empty() {
                        return Some(&vec[idx]);
                    }
                }
                None
            }
            FieldIndexType::Sparse(map) => map.get(key),
        }
    }

    fn remove(
        &mut self,
        field: &str,
        int_ranges: &HashMap<String, (i64, i64, usize)>,
        key: &IndexKey,
        ext_id: u64,
    ) {
        match self {
            FieldIndexType::DenseInt(vec) => {
                if let IndexKey::Int(i) = key {
                    let min = int_ranges.get(field).map(|(m, _, _)| *m).unwrap_or(0);
                    let offset = i - min;
                    if offset >= 0 {
                        let idx = offset as usize;
                        if idx < vec.len() {
                            vec[idx].retain(|&id| id != ext_id);
                        }
                    }
                }
            }
            FieldIndexType::Sparse(map) => {
                if let Some(ids) = map.get_mut(key) {
                    ids.retain(|&id| id != ext_id);
                    if ids.is_empty() {
                        map.remove(key);
                    }
                }
            }
        }
    }
}

/// In-memory equality index with memory limits and dense integer optimization.
/// For integer fields with contiguous range (0..N), uses Vec<Vec<u64>> for O(1) access.
/// For sparse/non-integer fields, uses HashMap.
#[derive(Default)]
struct FieldIndex {
    /// Per-field index (either dense Vec or sparse HashMap).
    index: HashMap<String, FieldIndexType>,
    /// Per-field ordered index for numeric values and lexicographically sortable
    /// strings such as ISO-8601 datetime values.
    range_index: HashMap<String, BTreeMap<RangeKey, Vec<u64>>>,
    /// Per-field array element index for `field CONTAINS value` filters.
    array_index: HashMap<String, HashMap<IndexKey, Vec<u64>>>,
    /// Track integer ranges to detect dense fields.
    int_ranges: HashMap<String, (i64, i64, usize)>, // (min, max, count)
    /// Fields that exceeded cardinality limit (blacklisted).
    blacklisted: std::collections::HashSet<String>,
}

impl FieldIndex {
    /// Try to insert a value→id mapping. Returns false if field is blacklisted.
    fn insert(&mut self, field: String, key: IndexKey, ext_id: u64) -> bool {
        if self.blacklisted.contains(&field) {
            return false;
        }

        // Track integer ranges for dense field detection
        let is_int = matches!(key, IndexKey::Int(_));
        if let IndexKey::Int(i) = &key {
            let entry = self
                .int_ranges
                .entry(field.clone())
                .or_insert((i64::MAX, i64::MIN, 0));
            entry.0 = entry.0.min(*i);
            entry.1 = entry.1.max(*i);
            entry.2 += 1;
        }

        // Check if we should convert an existing sparse index to dense
        if is_int && !self.index.contains_key(&field) {
            if let Some((min, max, count)) = self.int_ranges.get(&field) {
                let range = max - min + 1;
                // Dense if: contiguous range < 2M and we have enough entries
                if range > 0 && range < 2_000_000 && (*count as i64) >= range.min(1000) {
                    // Create DenseInt index now that we know the full range
                    let mut vec = Vec::with_capacity(range as usize);
                    vec.resize_with(range as usize, Vec::new);
                    self.index
                        .insert(field.clone(), FieldIndexType::DenseInt(vec));
                }
            }
        }

        // Get or create the field index (Sparse if not dense enough yet)
        let idx = self
            .index
            .entry(field.clone())
            .or_insert_with(|| FieldIndexType::Sparse(HashMap::new()));

        // Insert based on index type
        match idx {
            FieldIndexType::DenseInt(vec) => {
                if let IndexKey::Int(i) = key {
                    let min = self.int_ranges.get(&field).map(|(m, _, _)| *m).unwrap_or(0);
                    let idx_usize = (i - min) as usize;
                    if idx_usize < vec.len() {
                        vec[idx_usize].push(ext_id);
                        return true;
                    }
                    // Out of bounds but still within limit: grow the Vec
                    if idx_usize < 2_000_000 {
                        vec.resize_with(idx_usize + 1, Vec::new);
                        vec[idx_usize].push(ext_id);
                        if let Some(entry) = self.int_ranges.get_mut(&field) {
                            entry.1 = entry.1.max(min + idx_usize as i64);
                        }
                        return true;
                    }
                    // Exceeds 2M limit: convert to Sparse
                }
                // Non-integer key or exceeds limit: convert DenseInt to Sparse
                let mut new_map = HashMap::new();
                if let FieldIndexType::DenseInt(ref vec) = *idx {
                    let min = self.int_ranges.get(&field).map(|(m, _, _)| *m).unwrap_or(0);
                    for (i, ids) in vec.iter().enumerate() {
                        if !ids.is_empty() {
                            new_map.insert(IndexKey::Int(min + i as i64), ids.clone());
                        }
                    }
                }
                new_map.entry(key).or_default().push(ext_id);
                *idx = FieldIndexType::Sparse(new_map);
                true
            }
            FieldIndexType::Sparse(map) => {
                map.entry(key).or_default().push(ext_id);
                true
            }
        }
    }

    fn insert_value(&mut self, field: &str, value: &serde_json::Value, ext_id: u64) {
        if self.blacklisted.contains(field) {
            return;
        }

        match value {
            serde_json::Value::Array(values) => {
                for value in values {
                    if matches!(
                        value,
                        serde_json::Value::Array(_) | serde_json::Value::Object(_)
                    ) {
                        continue;
                    }
                    self.array_index
                        .entry(field.to_string())
                        .or_default()
                        .entry(IndexKey::from_json(value))
                        .or_default()
                        .push(ext_id);
                }
            }
            serde_json::Value::Object(_) => {}
            _ => {
                self.insert(field.to_string(), IndexKey::from_json(value), ext_id);
                if let Some(key) = RangeKey::from_json(value) {
                    self.range_index
                        .entry(field.to_string())
                        .or_default()
                        .entry(key)
                        .or_default()
                        .push(ext_id);
                }
            }
        }

        self.maybe_blacklist_field(field);
    }

    fn remove_value(&mut self, field: &str, value: &serde_json::Value, ext_id: u64) {
        match value {
            serde_json::Value::Array(values) => {
                if let Some(map) = self.array_index.get_mut(field) {
                    for value in values {
                        if matches!(
                            value,
                            serde_json::Value::Array(_) | serde_json::Value::Object(_)
                        ) {
                            continue;
                        }
                        let key = IndexKey::from_json(value);
                        if let Some(ids) = map.get_mut(&key) {
                            ids.retain(|&id| id != ext_id);
                            if ids.is_empty() {
                                map.remove(&key);
                            }
                        }
                    }
                }
            }
            serde_json::Value::Object(_) => {}
            _ => {
                let key = IndexKey::from_json(value);
                self.remove(field, &key, ext_id);
                if let Some(range_key) = RangeKey::from_json(value) {
                    if let Some(map) = self.range_index.get_mut(field) {
                        if let Some(ids) = map.get_mut(&range_key) {
                            ids.retain(|&id| id != ext_id);
                            if ids.is_empty() {
                                map.remove(&range_key);
                            }
                        }
                    }
                }
            }
        }
    }

    fn get(&self, field: &str, key: &IndexKey) -> Option<&Vec<u64>> {
        let idx = self.index.get(field)?;
        // For dense int, need to shift by min
        if let FieldIndexType::DenseInt(vec) = idx {
            if let IndexKey::Int(i) = key {
                let min = self.int_ranges.get(field).map(|(m, _, _)| *m).unwrap_or(0);
                let idx_usize = (i - min) as usize;
                if idx_usize < vec.len() && !vec[idx_usize].is_empty() {
                    return Some(&vec[idx_usize]);
                }
            }
            return None;
        }
        idx.get(key)
    }

    fn get_contains(&self, field: &str, key: &IndexKey) -> Vec<u64> {
        self.array_index
            .get(field)
            .and_then(|map| map.get(key))
            .cloned()
            .unwrap_or_default()
    }

    fn get_range(&self, field: &str, op: RangeOp, key: &RangeKey) -> Option<Vec<u64>> {
        use std::ops::Bound::{Excluded, Included, Unbounded};

        let map = self.range_index.get(field)?;
        let range = match op {
            RangeOp::Lt => (Unbounded, Excluded(key)),
            RangeOp::Lte => (Unbounded, Included(key)),
            RangeOp::Gt => (Excluded(key), Unbounded),
            RangeOp::Gte => (Included(key), Unbounded),
        };
        let mut ids = Vec::new();
        for value_ids in map.range(range).map(|(_, ids)| ids) {
            ids.extend(value_ids.iter().copied());
        }
        ids.sort_unstable();
        ids.dedup();
        Some(ids)
    }

    fn remove(&mut self, field: &str, key: &IndexKey, ext_id: u64) {
        if let Some(idx) = self.index.get_mut(field) {
            idx.remove(field, &self.int_ranges, key, ext_id);
        }
    }

    fn maybe_blacklist_field(&mut self, field: &str) {
        // DenseInt indexes are already bounded by the 2M range cap and provide O(1)
        // equality lookups for contiguous integer fields (e.g. row counters). Do not
        // blacklist them based on range/cardinality — that would force every query onto
        // ApexBase SQL and regress filtered search performance by ~2x.
        if matches!(self.index.get(field), Some(FieldIndexType::DenseInt(_))) {
            return;
        }

        if self.field_unique_count(field) <= MAX_INDEX_UNIQUE_VALUES {
            return;
        }

        self.index.remove(field);
        self.range_index.remove(field);
        self.array_index.remove(field);
        self.int_ranges.remove(field);
        self.blacklisted.insert(field.to_string());
    }

    fn field_unique_count(&self, field: &str) -> usize {
        let equality_count = match self.index.get(field) {
            Some(FieldIndexType::DenseInt(_)) => self
                .int_ranges
                .get(field)
                .and_then(|(min, max, _)| max.checked_sub(*min))
                .and_then(|range| range.checked_add(1))
                .map(|range| range as usize)
                .unwrap_or(0),
            Some(FieldIndexType::Sparse(map)) => map.len(),
            None => 0,
        };
        let range_count = self.range_index.get(field).map(|map| map.len()).unwrap_or(0);
        let array_count = self.array_index.get(field).map(|map| map.len()).unwrap_or(0);

        equality_count.max(range_count).max(array_count)
    }
}

/// Maximum number of entries to keep in the in-memory field cache.
/// Beyond this limit inserts are stored in ApexBase only and retrieved on demand.
const MAX_CACHE_ENTRIES: usize = 200_000;

/// ApexBase-backed field store for per-vector metadata.
///
/// Each vector ID maps to a row of arbitrary metadata fields stored in ApexBase.
/// Supports SQL-like filtering used by `where` in the Python API.
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
    /// In-memory cache: external_id -> fields (populated during insert, O(1) retrieval)
    cache: RwLock<HashMap<u64, HashMap<String, serde_json::Value>>>,
    /// Maps external_id → ApexBase internal _id for O(1) mmap point lookups.
    /// Indexed directly by external_id (Vec[ext_id] = apex_id, TOMBSTONE = deleted/absent).
    /// Persisted to `map_path`; loaded on open, updated on every write.
    apex_id_map: RwLock<Vec<u64>>,
    /// Path to the persistent .apex_id_map binary log file.
    map_path: PathBuf,
    /// In-memory equality index with automatic high-cardinality field exclusion.
    /// Built during inserts; skips fields with >100k unique values to control memory.
    field_eq_index: RwLock<FieldIndex>,
}

impl FieldStore {
    /// Open or create a field store at the given path.
    pub fn new(db_path: &Path, table_name: &str) -> Result<Self> {
        std::fs::create_dir_all(db_path)?;

        let db = apexbase::embedded::ApexDB::open(db_path)
            .map_err(|e| LynseError::ApexBase(format!("Failed to open ApexBase: {}", e)))?;

        // Try to open existing table or create new one. Seed the reserved
        // external_id column so scans against an empty table have a stable schema.
        let _ = db.create_table_with_schema(
            table_name,
            &[("external_id".to_string(), apexbase::ColumnType::Int64)],
        );

        // Determine next_id from existing data
        let next_id = {
            let table = db
                .table(table_name)
                .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;
            let count = table
                .count()
                .map_err(|e| LynseError::ApexBase(format!("Count error: {}", e)))?;
            count as u64
        };

        let map_path = db_path.join(format!("{}.apex_id_map", table_name));

        // Fast path: load apex_id_map from persisted file
        let apex_id_map = if map_path.exists() {
            let map = Self::load_map_file(&map_path).unwrap_or_default();
            if next_id > 0 && map.iter().any(|&apex_id| apex_id == 0) {
                // Older map files used 0 as the tombstone sentinel, which
                // conflicts with ApexBase's first real row id. Rebuild once
                // from the table and persist using the current sentinel.
                let vec = Self::rebuild_map_from_db(&db, table_name)?;
                let _ = Self::save_map_file(&vec, &map_path);
                vec
            } else {
                map
            }
        } else if next_id > 0 {
            // Slow path (first open after upgrade): rebuild from DB, then save
            let vec = Self::rebuild_map_from_db(&db, table_name)?;
            let _ = Self::save_map_file(&vec, &map_path);
            vec
        } else {
            Vec::new()
        };

        let store = Self {
            db_path: db_path.to_path_buf(),
            db,
            table_name: table_name.to_string(),
            next_id: Arc::new(RwLock::new(next_id)),
            cache: RwLock::new(HashMap::new()),
            apex_id_map: RwLock::new(apex_id_map),
            map_path,
            field_eq_index: RwLock::new(FieldIndex::default()),
        };
        store.rebuild_in_memory_indexes()?;
        Ok(store)
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

        let apex_id = table
            .insert(row)
            .map_err(|e| LynseError::ApexBase(format!("Insert error: {}", e)))?;

        // Update apex_id_map, cache (capped), field_eq_index, and persist
        {
            let mut map = self.apex_id_map.write();
            if ext_id as usize >= map.len() {
                map.resize(ext_id as usize + 1, TOMBSTONE);
            }
            map[ext_id as usize] = apex_id;
        }
        {
            let mut cache = self.cache.write();
            if cache.len() < MAX_CACHE_ENTRIES {
                cache.insert(ext_id, fields.clone());
            }
        }
        {
            let mut idx = self.field_eq_index.write();
            for (field, value) in fields {
                idx.insert_value(field, value, ext_id);
            }
        }
        let _ = self.append_map_records(&[(ext_id, apex_id)]);

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

        // Use native batch insert for much better performance
        let apex_ids = table
            .insert_batch(&rows)
            .map_err(|e| LynseError::ApexBase(format!("Batch insert error: {}", e)))?;

        // Update apex_id_map (sequential push), cache (capped), field_eq_index, and persist
        {
            let mut map = self.apex_id_map.write();
            let mut cache = self.cache.write();
            let mut idx = self.field_eq_index.write();
            for ((&ext_id, &apex_id), flds) in
                ids.iter().zip(apex_ids.iter()).zip(fields_list.iter())
            {
                if ext_id as usize >= map.len() {
                    map.resize(ext_id as usize + 1, TOMBSTONE);
                }
                map[ext_id as usize] = apex_id;
                if cache.len() < MAX_CACHE_ENTRIES {
                    cache.insert(ext_id, flds.clone());
                }
                for (field, value) in flds {
                    idx.insert_value(field, value, ext_id);
                }
            }
        }
        let pairs: Vec<(u64, u64)> = ids.iter().copied().zip(apex_ids.iter().copied()).collect();
        let _ = self.append_map_records(&pairs);

        Ok(ids)
    }

    /// Batch store records with caller-provided external IDs.
    ///
    /// Collection storage uses row offsets as field-store external IDs. This
    /// method keeps metadata aligned even when earlier vector rows had no
    /// metadata and were intentionally skipped in ApexBase.
    pub fn batch_store_at_ids(
        &self,
        external_ids: &[u64],
        fields_list: &[HashMap<String, serde_json::Value>],
    ) -> Result<Vec<u64>> {
        if external_ids.len() != fields_list.len() {
            return Err(LynseError::InvalidArgument(format!(
                "external_ids length ({}) must match fields length ({})",
                external_ids.len(),
                fields_list.len()
            )));
        }

        let table = self
            .db
            .table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;

        let mut ids = Vec::new();
        let mut rows: Vec<HashMap<String, apexbase::data::Value>> = Vec::new();
        let mut stored_fields: Vec<HashMap<String, serde_json::Value>> = Vec::new();

        for (&ext_id, fields) in external_ids.iter().zip(fields_list.iter()) {
            if fields.is_empty() {
                continue;
            }

            let mut row: HashMap<String, apexbase::data::Value> = HashMap::new();
            for (key, value) in fields {
                row.insert(key.clone(), json_to_apex_value(value));
            }
            row.insert(
                "external_id".to_string(),
                apexbase::data::Value::Int64(ext_id as i64),
            );

            ids.push(ext_id);
            rows.push(row);
            stored_fields.push(fields.clone());
        }

        if ids.is_empty() {
            return Ok(Vec::new());
        }

        {
            let max_ext_id = ids.iter().copied().max().unwrap_or(0);
            let mut next = self.next_id.write();
            if *next <= max_ext_id {
                *next = max_ext_id + 1;
            }
        }

        let apex_ids = table
            .insert_batch(&rows)
            .map_err(|e| LynseError::ApexBase(format!("Batch insert error: {}", e)))?;

        {
            let mut map = self.apex_id_map.write();
            let mut cache = self.cache.write();
            let mut idx = self.field_eq_index.write();
            for ((&ext_id, &apex_id), flds) in
                ids.iter().zip(apex_ids.iter()).zip(stored_fields.iter())
            {
                if ext_id as usize >= map.len() {
                    map.resize(ext_id as usize + 1, TOMBSTONE);
                }
                map[ext_id as usize] = apex_id;
                if cache.len() < MAX_CACHE_ENTRIES {
                    cache.insert(ext_id, flds.clone());
                }
                for (field, value) in flds {
                    idx.insert_value(field, value, ext_id);
                }
            }
        }

        let pairs: Vec<(u64, u64)> = ids.iter().copied().zip(apex_ids.iter().copied()).collect();
        self.append_map_records(&pairs)?;

        Ok(ids)
    }

    /// Replace field records for caller-provided external IDs.
    ///
    /// Existing ApexBase rows are deleted and the point-lookup map is moved to
    /// the new rows. Empty replacement maps clear fields for that external ID.
    pub fn replace_fields_at_ids(
        &self,
        external_ids: &[u64],
        fields_list: &[HashMap<String, serde_json::Value>],
    ) -> Result<()> {
        if external_ids.len() != fields_list.len() {
            return Err(LynseError::InvalidArgument(format!(
                "external_ids length ({}) must match fields length ({})",
                external_ids.len(),
                fields_list.len()
            )));
        }
        if external_ids.is_empty() {
            return Ok(());
        }

        let old_fields = self.retrieve_many(external_ids)?;
        let table = self
            .db
            .table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;

        let old_apex_ids = {
            let map = self.apex_id_map.read();
            external_ids
                .iter()
                .filter_map(|&ext_id| {
                    let i = ext_id as usize;
                    if i < map.len() && map[i] != TOMBSTONE {
                        Some(map[i])
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        };

        if !old_apex_ids.is_empty() {
            table
                .delete_batch(&old_apex_ids)
                .map_err(|e| LynseError::ApexBase(format!("Delete batch error: {}", e)))?;
        }

        {
            let mut map = self.apex_id_map.write();
            let mut cache = self.cache.write();
            let mut idx = self.field_eq_index.write();
            for (&ext_id, old) in external_ids.iter().zip(old_fields.iter()) {
                let i = ext_id as usize;
                if i < map.len() {
                    map[i] = TOMBSTONE;
                }
                cache.remove(&ext_id);
                for (field, value) in old {
                    idx.remove_value(field, value, ext_id);
                }
            }
        }

        let tombstones: Vec<(u64, u64)> = external_ids.iter().map(|&id| (id, TOMBSTONE)).collect();
        self.append_map_records(&tombstones)?;
        self.batch_store_at_ids(external_ids, fields_list)?;

        Ok(())
    }

    /// Query fields using a SQL-like filter expression.
    /// Returns matching external IDs.
    /// Fast path: if the expression is a simple AND of equality conditions and the
    /// in-memory field_eq_index is populated, answers in O(1) without touching ApexBase.
    pub fn query(&self, filter_expr: &str) -> Result<Vec<u64>> {
        // Fast path: in-memory equality index
        if let Some(ids) = self.query_from_index(filter_expr) {
            return Ok(ids);
        }

        // Slow path: ApexBase SQL
        let table = self
            .db
            .table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;

        let sql = format!(
            "SELECT _id, external_id FROM {} WHERE {}",
            self.table_name, filter_expr
        );

        let result_set = table
            .execute(&sql)
            .map_err(|e| LynseError::ApexBase(format!("Query error: {}", e)))?;

        let batch = result_set
            .to_record_batch()
            .map_err(|e| LynseError::ApexBase(format!("Result conversion error: {}", e)))?;

        self.current_external_ids_from_batch(&batch)
    }

    /// Query fields and return both IDs and field data in a single SQL query.
    /// Fast path: if the WHERE clause is a simple AND-of-equalities and field_eq_index
    /// is populated, bypasses ApexBase entirely and serves from cache / apex_id_map.
    pub fn query_with_fields(
        &self,
        filter_expr: &str,
    ) -> Result<(Vec<u64>, Vec<HashMap<String, serde_json::Value>>)> {
        // Fast path: in-memory equality index
        if let Some(ext_ids) = self.query_from_index(filter_expr) {
            let fields_list = self.retrieve_many(&ext_ids)?;
            return Ok((ext_ids, fields_list));
        }

        let table = self
            .db
            .table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;

        let sql = format!("SELECT * FROM {} WHERE {}", self.table_name, filter_expr);

        let result_set = table
            .execute(&sql)
            .map_err(|e| LynseError::ApexBase(format!("Query error: {}", e)))?;

        let batch = result_set
            .to_record_batch()
            .map_err(|e| LynseError::ApexBase(format!("Result conversion error: {}", e)))?;

        self.current_fields_from_batch(&batch)
    }

    /// Scan all currently active field rows.
    ///
    /// This is intended for query features that need to inspect text metadata
    /// across rows, such as the baseline BM25 text search path.
    pub fn scan_current_fields(&self) -> Result<Vec<(u64, HashMap<String, serde_json::Value>)>> {
        let table = self
            .db
            .table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;

        let sql = format!("SELECT * FROM {}", self.table_name);
        let batch = table
            .execute(&sql)
            .map_err(|e| LynseError::ApexBase(format!("Query error: {}", e)))?
            .to_record_batch()
            .map_err(|e| LynseError::ApexBase(format!("Result conversion error: {}", e)))?;

        let (ids, fields_list) = self.current_fields_from_batch(&batch)?;
        Ok(ids.into_iter().zip(fields_list).collect())
    }

    /// Retrieve a single record by external ID.
    pub fn retrieve(&self, external_id: u64) -> Result<HashMap<String, serde_json::Value>> {
        // Fast path: in-memory cache
        {
            let cache = self.cache.read();
            if let Some(fields) = cache.get(&external_id) {
                return Ok(fields.clone());
            }
        }

        let table = self
            .db
            .table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;

        // Fast path: apex_id mmap point lookup
        let apex_id = {
            let map = self.apex_id_map.read();
            let i = external_id as usize;
            if i < map.len() && map[i] != TOMBSTONE {
                Some(map[i])
            } else {
                None
            }
        };
        if let Some(apex_id) = apex_id {
            if let Some(row) = table
                .retrieve(apex_id)
                .map_err(|e| LynseError::ApexBase(format!("Retrieve error: {}", e)))?
            {
                let fields: HashMap<String, serde_json::Value> = row
                    .into_iter()
                    .filter(|(k, _)| k != "external_id" && k != "_id")
                    .map(|(k, v)| (k, apex_value_to_json(&v)))
                    .collect();
                return Ok(fields);
            }
        }

        // Fallback: SQL query
        let sql = format!(
            "SELECT * FROM {} WHERE external_id = {}",
            self.table_name, external_id
        );
        let rows = table
            .execute(&sql)
            .map_err(|e| LynseError::ApexBase(format!("Query error: {}", e)))?
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
    /// Uses in-memory cache for O(1) lookups; falls back to batched SQL for cache misses.
    pub fn retrieve_many(
        &self,
        external_ids: &[u64],
    ) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        if external_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Fast path: try to serve entirely from cache
        let cache = self.cache.read();
        let mut results = Vec::with_capacity(external_ids.len());
        let mut miss_indices: Vec<usize> = Vec::new();
        let mut miss_ids: Vec<u64> = Vec::new();

        for (i, &id) in external_ids.iter().enumerate() {
            if let Some(fields) = cache.get(&id) {
                results.push(fields.clone());
            } else {
                results.push(HashMap::new());
                miss_indices.push(i);
                miss_ids.push(id);
            }
        }
        drop(cache);

        // If all hits, return immediately (common case after insert)
        if miss_ids.is_empty() {
            return Ok(results);
        }

        // Slow path: use apex_id_map for mmap batch retrieve, fall back to SQL for unknowns
        let table = self
            .db
            .table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;

        // Collect apex_ids for misses that have a known mapping
        let apex_id_map = self.apex_id_map.read();
        let mut mapped: Vec<(usize, u64, u64)> = Vec::new();
        let mut unmapped: Vec<(usize, u64)> = Vec::new();
        for (&idx, &ext_id) in miss_indices.iter().zip(miss_ids.iter()) {
            let i = ext_id as usize;
            if i < apex_id_map.len() && apex_id_map[i] != TOMBSTONE {
                mapped.push((idx, ext_id, apex_id_map[i]));
            } else {
                unmapped.push((idx, ext_id));
            }
        }
        drop(apex_id_map);

        let mut id_to_fields: HashMap<u64, HashMap<String, serde_json::Value>> =
            HashMap::with_capacity(miss_ids.len());

        // Fast path: mmap point lookups for mapped IDs
        if !mapped.is_empty() {
            let apex_ids: Vec<u64> = mapped.iter().map(|(_, _, aid)| *aid).collect();
            let batch = table
                .retrieve_many(&apex_ids)
                .map_err(|e| LynseError::ApexBase(format!("retrieve_many error: {}", e)))?;
            // Build apex_id→ext_id lookup for O(1) result mapping
            let apex_to_ext: HashMap<u64, u64> = mapped
                .iter()
                .map(|&(_, ext_id, apex_id)| (apex_id, ext_id))
                .collect();
            for (apex_id, fields) in fields_by_apex_id_from_batch(&batch)? {
                if let Some(&ext_id) = apex_to_ext.get(&apex_id) {
                    id_to_fields.insert(ext_id, fields);
                }
            }
        }

        // Fallback: SQL IN(...) for unmapped IDs
        if !unmapped.is_empty() {
            let id_list: Vec<String> = unmapped.iter().map(|(_, id)| id.to_string()).collect();
            let sql = format!(
                "SELECT * FROM {} WHERE external_id IN ({})",
                self.table_name,
                id_list.join(", ")
            );
            let batch = table
                .execute(&sql)
                .map_err(|e| LynseError::ApexBase(format!("Query error: {}", e)))?
                .to_record_batch()
                .map_err(|e| LynseError::ApexBase(format!("Result conversion error: {}", e)))?;
            for (ext_id, fields) in fields_by_external_id_from_batch(&batch)? {
                id_to_fields.insert(ext_id, fields);
            }
        }

        // Fill in miss results and update cache
        let mut cache = self.cache.write();
        for (&idx, &id) in miss_indices.iter().zip(miss_ids.iter()) {
            if let Some(fields) = id_to_fields.remove(&id) {
                cache.insert(id, fields.clone());
                results[idx] = fields;
            }
        }

        Ok(results)
    }

    /// List all field names (column names) in the store.
    pub fn list_fields(&self) -> Result<Vec<String>> {
        let table = self
            .db
            .table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;

        let columns = table
            .columns()
            .map_err(|e| LynseError::ApexBase(format!("Schema error: {}", e)))?;

        Ok(columns
            .into_iter()
            .filter(|k| *k != "_id" && *k != "external_id")
            .collect())
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

    /// Drop the field store table and its persistent map file.
    pub fn drop(&self) -> Result<()> {
        self.db
            .drop_table(&self.table_name)
            .map_err(|e| LynseError::ApexBase(format!("Drop table error: {}", e)))?;
        let _ = std::fs::remove_file(&self.map_path);
        Ok(())
    }

    /// Write tombstone records for deleted external IDs.
    /// Calling this removes the entries from the in-memory map and marks them
    /// as deleted in the persistent file (apex_id = TOMBSTONE).
    pub fn delete_map_entries(&self, ext_ids: &[u64]) {
        if ext_ids.is_empty() {
            return;
        }
        {
            let mut map = self.apex_id_map.write();
            let mut cache = self.cache.write();
            for &id in ext_ids {
                let i = id as usize;
                if i < map.len() {
                    map[i] = TOMBSTONE;
                }
                // Remove from cache; skip field_eq_index cleanup (will be filtered on query)
                cache.remove(&id);
            }
        }
        let tombstones: Vec<(u64, u64)> = ext_ids.iter().map(|&id| (id, TOMBSTONE)).collect();
        let _ = self.append_map_records(&tombstones);
    }

    // ── In-memory field index helpers ────────────────────────────────────────

    fn rebuild_in_memory_indexes(&self) -> Result<()> {
        let records = self.scan_current_fields()?;
        let mut idx = FieldIndex::default();
        let mut cache = self.cache.write();
        cache.clear();

        for (ext_id, fields) in records {
            if cache.len() < MAX_CACHE_ENTRIES {
                cache.insert(ext_id, fields.clone());
            }
            for (field, value) in &fields {
                idx.insert_value(field, value, ext_id);
            }
        }

        *self.field_eq_index.write() = idx;
        Ok(())
    }

    /// Try to answer a WHERE expression from the in-memory equality index.
    /// Returns `Some(ext_ids)` on success, `None` if the index is cold or the
    /// expression is too complex (caller should fall back to ApexBase SQL).
    fn query_from_index(&self, filter_expr: &str) -> Option<Vec<u64>> {
        if let Some((field, keys)) = parse_indexed_in(filter_expr)
            .or_else(|| parse_indexed_or_equalities(filter_expr))
        {
            return Some(self.query_index_values(&field, &keys));
        }
        if let Some(leaves) = parse_indexed_or_leaves(filter_expr) {
            let mut ids: HashSet<u64> = HashSet::new();
            for (field, key) in leaves {
                if let Some(set) = self.query_index_single(&field, &key) {
                    ids.extend(set);
                } else {
                    return None;
                }
            }
            let mut result: Vec<u64> = ids.into_iter().collect();
            result.sort_unstable();
            return Some(self.filter_current_external_ids(result));
        }

        let conditions = parse_indexed_where(filter_expr)?;
        let idx = self.field_eq_index.read();

        // Collect result sets
        let mut sets: Vec<Vec<u64>> = Vec::with_capacity(conditions.len());
        for condition in &conditions {
            let mut ids = match condition {
                IndexedCondition::Eq(field, key) => idx.get(field, key)?.clone(),
                IndexedCondition::Range(field, op, key) => idx.get_range(field, *op, key)?,
                IndexedCondition::Contains(field, key) => idx.get_contains(field, key),
            };
            ids.sort_unstable();
            ids.dedup();
            sets.push(ids);
        }

        // Sort by ascending size so we intersect the smallest set first
        sets.sort_unstable_by_key(|v| v.len());

        let result: Vec<u64> = if sets.len() == 1 {
            sets.remove(0)
        } else {
            // Build a HashSet from the smallest set, intersect with the rest
            let mut current: HashSet<u64> = sets[0].iter().copied().collect();
            for set in &sets[1..] {
                let other: HashSet<u64> = set.iter().copied().collect();
                current.retain(|id| other.contains(id));
                if current.is_empty() {
                    break;
                }
            }
            let mut result: Vec<u64> = current.into_iter().collect();
            result.sort_unstable();
            result
        };

        Some(self.filter_current_external_ids(result))
    }

    fn query_index_single(&self, field: &str, key: &IndexKey) -> Option<Vec<u64>> {
        let idx = self.field_eq_index.read();
        idx.get(field, key).cloned()
    }

    fn query_index_values(&self, field: &str, keys: &[IndexKey]) -> Vec<u64> {
        let idx = self.field_eq_index.read();
        let mut ids: HashSet<u64> = HashSet::new();
        for key in keys {
            if let Some(set) = idx.get(field, key) {
                ids.extend(set.iter().copied());
            }
        }
        let mut result: Vec<u64> = ids.into_iter().collect();
        result.sort_unstable();
        self.filter_current_external_ids(result)
    }

    fn filter_current_external_ids(&self, ids: Vec<u64>) -> Vec<u64> {
        let map = self.apex_id_map.read();
        ids.into_iter()
            .filter(|&ext_id| {
                let i = ext_id as usize;
                i < map.len() && map[i] != TOMBSTONE
            })
            .collect()
    }

    fn current_external_ids_from_batch(&self, batch: &RecordBatch) -> Result<Vec<u64>> {
        if batch.num_rows() == 0 && !record_batch_has_column(batch, "external_id") {
            return Ok(Vec::new());
        }
        let apex_ids = u64_column_values(batch, "_id")?;
        let ext_ids = u64_column_values(batch, "external_id")?;
        let map = self.apex_id_map.read();
        let mut ids = Vec::with_capacity(batch.num_rows());

        for (apex_id, ext_id) in apex_ids.into_iter().zip(ext_ids.into_iter()) {
            let i = ext_id as usize;
            if i < map.len() && map[i] == apex_id && map[i] != TOMBSTONE {
                ids.push(ext_id);
            }
        }

        Ok(ids)
    }

    fn current_fields_from_batch(
        &self,
        batch: &RecordBatch,
    ) -> Result<(Vec<u64>, Vec<HashMap<String, serde_json::Value>>)> {
        if batch.num_rows() == 0 && !record_batch_has_column(batch, "external_id") {
            return Ok((Vec::new(), Vec::new()));
        }
        let apex_ids = u64_column_values(batch, "_id")?;
        let ext_ids = u64_column_values(batch, "external_id")?;
        let field_columns = field_column_indices(batch);
        let map = self.apex_id_map.read();

        let mut ids = Vec::with_capacity(batch.num_rows());
        let mut fields_list = Vec::with_capacity(batch.num_rows());
        for row_idx in 0..batch.num_rows() {
            let apex_id = apex_ids[row_idx];
            let ext_id = ext_ids[row_idx];
            let i = ext_id as usize;
            if i >= map.len() || map[i] != apex_id || map[i] == TOMBSTONE {
                continue;
            }

            ids.push(ext_id);
            fields_list.push(fields_from_batch_row(batch, &field_columns, row_idx));
        }

        Ok((ids, fields_list))
    }

    // ── File I/O helpers ─────────────────────────────────────────────────────

    /// Load the apex_id_map from the binary log file.
    /// Format: sequence of 16-byte records [u64 ext_id LE][u64 apex_id LE].
    /// Records are applied in order (last write wins); TOMBSTONE=deleted.
    /// Returns a Vec<u64> directly indexed by external_id.
    fn load_map_file(map_path: &Path) -> Result<Vec<u64>> {
        let data = std::fs::read(map_path)?;
        let n = data.len() / MAP_RECORD_SIZE;
        if n == 0 {
            return Ok(Vec::new());
        }
        // Find the max ext_id to size the Vec
        let mut max_ext = 0u64;
        for i in 0..n {
            let off = i * MAP_RECORD_SIZE;
            let ext_id = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
            if ext_id > max_ext {
                max_ext = ext_id;
            }
        }
        // Apply all records in order (last write wins)
        let mut vec = vec![TOMBSTONE; max_ext as usize + 1];
        for i in 0..n {
            let off = i * MAP_RECORD_SIZE;
            let ext_id = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
            let apex_id = u64::from_le_bytes(data[off + 8..off + 16].try_into().unwrap());
            vec[ext_id as usize] = apex_id;
        }
        Ok(vec)
    }

    /// Write the entire Vec as a clean binary log (used after DB rebuild).
    /// Skips TOMBSTONE entries to keep the file compact.
    fn save_map_file(vec: &[u64], map_path: &Path) -> Result<()> {
        let active = vec.iter().filter(|&&a| a != TOMBSTONE).count();
        let mut buf = Vec::with_capacity(active * MAP_RECORD_SIZE);
        for (ext_id, &apex_id) in vec.iter().enumerate() {
            if apex_id != TOMBSTONE {
                buf.extend_from_slice(&(ext_id as u64).to_le_bytes());
                buf.extend_from_slice(&apex_id.to_le_bytes());
            }
        }
        std::fs::write(map_path, &buf)?;
        Ok(())
    }

    /// Append records to the binary log file (insert) or tombstones (delete).
    fn append_map_records(&self, records: &[(u64, u64)]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.map_path)?;
        let mut buf = Vec::with_capacity(records.len() * MAP_RECORD_SIZE);
        for &(ext_id, apex_id) in records {
            buf.extend_from_slice(&ext_id.to_le_bytes());
            buf.extend_from_slice(&apex_id.to_le_bytes());
        }
        file.write_all(&buf)?;
        Ok(())
    }

    /// Rebuild the apex_id_map from ApexBase (slow path, O(N)).
    /// Returns a Vec<u64> directly indexed by external_id.
    fn rebuild_map_from_db(db: &apexbase::embedded::ApexDB, table_name: &str) -> Result<Vec<u64>> {
        let table = db
            .table(table_name)
            .map_err(|e| LynseError::ApexBase(format!("Table error: {}", e)))?;
        let sql = format!("SELECT _id, external_id FROM {}", table_name);
        let mut pairs: Vec<(u64, u64)> = Vec::new();
        if let Ok(rs) = table.execute(&sql) {
            if let Ok(batch) = rs.to_record_batch() {
                let apex_ids = u64_column_values(&batch, "_id").unwrap_or_default();
                let ext_ids = u64_column_values(&batch, "external_id").unwrap_or_default();
                pairs = Vec::with_capacity(apex_ids.len().min(ext_ids.len()));
                for (apex_id, ext_id) in apex_ids.into_iter().zip(ext_ids.into_iter()) {
                    pairs.push((ext_id, apex_id));
                }
            }
        }
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let max_ext = pairs.iter().map(|&(e, _)| e).max().unwrap_or(0);
        let mut vec = vec![TOMBSTONE; max_ext as usize + 1];
        for (ext_id, apex_id) in pairs {
            vec[ext_id as usize] = apex_id;
        }
        Ok(vec)
    }
}

fn field_column_indices(batch: &RecordBatch) -> Vec<(usize, String)> {
    batch
        .schema()
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(idx, field)| {
            let name = field.name();
            if name == "_id" || name == "external_id" {
                None
            } else {
                Some((idx, name.clone()))
            }
        })
        .collect()
}

fn column_index(batch: &RecordBatch, name: &str) -> Result<usize> {
    batch
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == name)
        .ok_or_else(|| LynseError::ApexBase(format!("Missing ApexBase column: {}", name)))
}

fn record_batch_has_column(batch: &RecordBatch, name: &str) -> bool {
    batch
        .schema()
        .fields()
        .iter()
        .any(|field| field.name() == name)
}

fn u64_column_values(batch: &RecordBatch, name: &str) -> Result<Vec<u64>> {
    let idx = column_index(batch, name)?;
    let array = batch.column(idx);
    let mut values = Vec::with_capacity(batch.num_rows());
    for row_idx in 0..batch.num_rows() {
        if let Some(value) = arrow_u64_at(array.as_ref(), row_idx) {
            values.push(value);
        } else {
            return Err(LynseError::ApexBase(format!(
                "ApexBase column {} contains a non-integer value",
                name
            )));
        }
    }
    Ok(values)
}

fn fields_by_apex_id_from_batch(
    batch: &RecordBatch,
) -> Result<Vec<(u64, HashMap<String, serde_json::Value>)>> {
    let apex_ids = u64_column_values(batch, "_id")?;
    let field_columns = field_column_indices(batch);
    let mut rows = Vec::with_capacity(batch.num_rows());
    for (row_idx, apex_id) in apex_ids.into_iter().enumerate() {
        rows.push((
            apex_id,
            fields_from_batch_row(batch, &field_columns, row_idx),
        ));
    }
    Ok(rows)
}

fn fields_by_external_id_from_batch(
    batch: &RecordBatch,
) -> Result<Vec<(u64, HashMap<String, serde_json::Value>)>> {
    let ext_ids = u64_column_values(batch, "external_id")?;
    let field_columns = field_column_indices(batch);
    let mut rows = Vec::with_capacity(batch.num_rows());
    for (row_idx, ext_id) in ext_ids.into_iter().enumerate() {
        rows.push((
            ext_id,
            fields_from_batch_row(batch, &field_columns, row_idx),
        ));
    }
    Ok(rows)
}

fn fields_from_batch_row(
    batch: &RecordBatch,
    field_columns: &[(usize, String)],
    row_idx: usize,
) -> HashMap<String, serde_json::Value> {
    let mut fields = HashMap::with_capacity(field_columns.len());
    for (col_idx, name) in field_columns {
        fields.insert(
            name.clone(),
            arrow_json_at(batch.column(*col_idx).as_ref(), row_idx),
        );
    }
    fields
}

fn arrow_u64_at(array: &dyn Array, row_idx: usize) -> Option<u64> {
    if array.is_null(row_idx) {
        return None;
    }
    match array.data_type() {
        ArrowDataType::Int64 => array
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|arr| arr.value(row_idx) as u64),
        ArrowDataType::Int32 => array
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|arr| arr.value(row_idx) as u64),
        ArrowDataType::Int16 => array
            .as_any()
            .downcast_ref::<Int16Array>()
            .map(|arr| arr.value(row_idx) as u64),
        ArrowDataType::Int8 => array
            .as_any()
            .downcast_ref::<Int8Array>()
            .map(|arr| arr.value(row_idx) as u64),
        ArrowDataType::UInt64 => array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .map(|arr| arr.value(row_idx)),
        ArrowDataType::UInt32 => array
            .as_any()
            .downcast_ref::<UInt32Array>()
            .map(|arr| arr.value(row_idx) as u64),
        ArrowDataType::UInt16 => array
            .as_any()
            .downcast_ref::<UInt16Array>()
            .map(|arr| arr.value(row_idx) as u64),
        ArrowDataType::UInt8 => array
            .as_any()
            .downcast_ref::<UInt8Array>()
            .map(|arr| arr.value(row_idx) as u64),
        _ => None,
    }
}

fn arrow_json_at(array: &dyn Array, row_idx: usize) -> serde_json::Value {
    if array.is_null(row_idx) {
        return serde_json::Value::Null;
    }
    match array.data_type() {
        ArrowDataType::Boolean => array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .map(|arr| serde_json::Value::Bool(arr.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        ArrowDataType::Int64 => array
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|arr| serde_json::json!(arr.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        ArrowDataType::Int32 => array
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|arr| serde_json::json!(arr.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        ArrowDataType::Int16 => array
            .as_any()
            .downcast_ref::<Int16Array>()
            .map(|arr| serde_json::json!(arr.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        ArrowDataType::Int8 => array
            .as_any()
            .downcast_ref::<Int8Array>()
            .map(|arr| serde_json::json!(arr.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        ArrowDataType::UInt64 => array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .map(|arr| serde_json::json!(arr.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        ArrowDataType::UInt32 => array
            .as_any()
            .downcast_ref::<UInt32Array>()
            .map(|arr| serde_json::json!(arr.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        ArrowDataType::UInt16 => array
            .as_any()
            .downcast_ref::<UInt16Array>()
            .map(|arr| serde_json::json!(arr.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        ArrowDataType::UInt8 => array
            .as_any()
            .downcast_ref::<UInt8Array>()
            .map(|arr| serde_json::json!(arr.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        ArrowDataType::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|arr| serde_json::json!(arr.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        ArrowDataType::Float32 => array
            .as_any()
            .downcast_ref::<Float32Array>()
            .map(|arr| serde_json::json!(arr.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        ArrowDataType::Utf8 => array
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|arr| string_to_json(arr.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        ArrowDataType::LargeUtf8 => array
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .map(|arr| string_to_json(arr.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        ArrowDataType::Binary => array
            .as_any()
            .downcast_ref::<BinaryArray>()
            .map(|arr| serde_json::json!(arr.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        ArrowDataType::LargeBinary => array
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .map(|arr| serde_json::json!(arr.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        _ => serde_json::Value::String(format!("{:?}", array.data_type())),
    }
}

fn string_to_json(value: &str) -> serde_json::Value {
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(value) {
        if parsed.is_array() || parsed.is_object() {
            return parsed;
        }
    }
    serde_json::Value::String(value.to_string())
}

// ── WHERE expression parser ───────────────────────────────────────────────────

enum IndexedCondition {
    Eq(String, IndexKey),
    Range(String, RangeOp, RangeKey),
    Contains(String, IndexKey),
}

/// Parse a WHERE expression as a conjunction of indexed conditions.
///
/// Supports:
/// - `"field" = value`
/// - `"field" <|<=|>|>= value`
/// - `"array_field" CONTAINS value`
///
/// Conditions may be joined by AND. Returns `None` for expressions that should
/// fall back to ApexBase SQL, such as OR, LIKE, and function calls.
fn parse_indexed_in(expr: &str) -> Option<(String, Vec<IndexKey>)> {
    let expr = expr.trim();
    if !expr.starts_with('"') {
        return None;
    }
    let close = expr[1..].find('"')?;
    let field = expr[1..1 + close].to_string();
    let rest = expr[1 + close + 1..].trim();
    if !starts_with_keyword(rest, "IN") {
        return None;
    }
    let values = rest["IN".len()..].trim();
    if !values.starts_with('(') || !values.ends_with(')') {
        return None;
    }
    let inner = values[1..values.len() - 1].trim();
    if inner.is_empty() {
        return None;
    }
    let mut keys = Vec::new();
    for token in split_csv_tokens(inner) {
        keys.push(IndexKey::from_token(token.trim())?);
    }
    Some((field, keys))
}

/// Parse `"field" = v1 OR "field" = v2 OR ...` into a value list.
fn parse_indexed_or_equalities(expr: &str) -> Option<(String, Vec<IndexKey>)> {
    let parts = split_on_or(expr);
    if parts.len() < 2 {
        return None;
    }
    let mut field_name: Option<String> = None;
    let mut keys = Vec::new();
    for part in parts {
        let (field, key) = parse_equality_condition(part.trim())?;
        match &field_name {
            Some(existing) if existing != &field => return None,
            None => field_name = Some(field),
            _ => {}
        }
        keys.push(key);
    }
    Some((field_name?, keys))
}

/// Parse a top-level OR of simple equality predicates on different fields.
fn parse_indexed_or_leaves(expr: &str) -> Option<Vec<(String, IndexKey)>> {
    let parts = split_on_or(expr);
    if parts.len() < 2 {
        return None;
    }
    let mut leaves = Vec::with_capacity(parts.len());
    for part in parts {
        leaves.push(parse_equality_condition(part.trim())?);
    }
    Some(leaves)
}

fn parse_equality_condition(expr: &str) -> Option<(String, IndexKey)> {
    let expr = expr.trim();
    if !expr.starts_with('"') {
        return None;
    }
    let close = expr[1..].find('"')?;
    let field = expr[1..1 + close].to_string();
    let rest = expr[1 + close + 1..].trim();
    if !rest.starts_with('=') || rest.starts_with("!=") {
        return None;
    }
    let value = rest[1..].trim();
    Some((field, IndexKey::from_token(extract_value_literal(value)?)?))
}

fn split_on_or(expr: &str) -> Vec<&str> {
    let bytes = expr.as_bytes();
    let mut parts: Vec<&str> = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    let mut in_dquote = false;
    let mut in_squote = false;

    while i < bytes.len() {
        match bytes[i] {
            b'"' if !in_squote => {
                in_dquote = !in_dquote;
                i += 1;
            }
            b'\'' if !in_dquote => {
                in_squote = !in_squote;
                i += 1;
            }
            _ if !in_dquote && !in_squote => {
                if i + 4 <= bytes.len()
                    && bytes[i] == b' '
                    && bytes[i + 1].to_ascii_uppercase() == b'O'
                    && bytes[i + 2].to_ascii_uppercase() == b'R'
                    && bytes[i + 3] == b' '
                {
                    parts.push(expr[start..i].trim());
                    i += 4;
                    start = i;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }
    parts.push(expr[start..].trim());
    parts
}

fn split_csv_tokens(values: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut start = 0usize;
    let bytes = values.as_bytes();
    let mut i = 0usize;
    let mut in_dquote = false;
    let mut in_squote = false;

    while i < bytes.len() {
        match bytes[i] {
            b'"' if !in_squote => {
                in_dquote = !in_dquote;
                i += 1;
            }
            b'\'' if !in_dquote => {
                in_squote = !in_squote;
                i += 1;
            }
            b',' if !in_dquote && !in_squote => {
                let token = values[start..i].trim();
                if !token.is_empty() {
                    tokens.push(token);
                }
                i += 1;
                start = i;
            }
            _ => i += 1,
        }
    }
    let token = values[start..].trim();
    if !token.is_empty() {
        tokens.push(token);
    }
    tokens
}

fn parse_indexed_where(expr: &str) -> Option<Vec<IndexedCondition>> {
    let parts = split_on_and(expr);
    let mut conditions = Vec::with_capacity(parts.len());
    for part in parts {
        conditions.push(parse_single_indexed_condition(part.trim())?);
    }
    if conditions.is_empty() {
        None
    } else {
        Some(conditions)
    }
}

/// Split `expr` on ` AND ` / ` and ` boundaries that are not inside quotes.
fn split_on_and(expr: &str) -> Vec<&str> {
    let bytes = expr.as_bytes();
    let mut parts: Vec<&str> = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    let mut in_dquote = false;
    let mut in_squote = false;

    while i < bytes.len() {
        match bytes[i] {
            b'"' if !in_squote => {
                in_dquote = !in_dquote;
                i += 1;
            }
            b'\'' if !in_dquote => {
                in_squote = !in_squote;
                i += 1;
            }
            _ if !in_dquote && !in_squote => {
                // Check for " AND " (case-insensitive, 5 bytes)
                if i + 5 <= bytes.len()
                    && bytes[i] == b' '
                    && bytes[i + 1].to_ascii_uppercase() == b'A'
                    && bytes[i + 2].to_ascii_uppercase() == b'N'
                    && bytes[i + 3].to_ascii_uppercase() == b'D'
                    && bytes[i + 4] == b' '
                {
                    parts.push(expr[start..i].trim());
                    i += 5;
                    start = i;
                } else {
                    i += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    }
    parts.push(expr[start..].trim());
    parts
}

fn parse_single_indexed_condition(expr: &str) -> Option<IndexedCondition> {
    let expr = expr.trim();

    // Field name must be double-quoted
    if !expr.starts_with('"') {
        return None;
    }
    let close = expr[1..].find('"')?;
    let field = expr[1..1 + close].to_string();
    let rest = expr[1 + close + 1..].trim();

    if starts_with_keyword(rest, "CONTAINS") {
        let value = rest["CONTAINS".len()..].trim();
        return Some(IndexedCondition::Contains(
            field,
            IndexKey::from_token(extract_value_literal(value)?)?,
        ));
    }

    for (op_token, op) in [
        (">=", RangeOp::Gte),
        ("<=", RangeOp::Lte),
        (">", RangeOp::Gt),
        ("<", RangeOp::Lt),
    ] {
        if let Some(value) = rest.strip_prefix(op_token) {
            return Some(IndexedCondition::Range(
                field,
                op,
                RangeKey::from_token(extract_value_literal(value.trim())?)?,
            ));
        }
    }

    // Operator must be plain `=` for equality (not `!=`, `>=`, `<=`)
    if rest.starts_with('=') && !rest.starts_with("!=") {
        let value = rest[1..].trim();
        return Some(IndexedCondition::Eq(
            field,
            IndexKey::from_token(extract_value_literal(value)?)?,
        ));
    }

    None
}

fn starts_with_keyword(value: &str, keyword: &str) -> bool {
    let value = value.trim_start();
    if value.len() < keyword.len() || !value[..keyword.len()].eq_ignore_ascii_case(keyword) {
        return false;
    }
    value[keyword.len()..]
        .chars()
        .next()
        .map(|c| c.is_whitespace())
        .unwrap_or(true)
}

fn extract_value_literal(value: &str) -> Option<&str> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    if value.starts_with('"') {
        let end = value[1..].find('"')?;
        return Some(&value[..end + 2]);
    }
    if value.starts_with('\'') {
        let end = value[1..].find('\'')?;
        return Some(&value[..end + 2]);
    }
    let end = value
        .find(|c: char| c.is_whitespace())
        .unwrap_or(value.len());
    Some(&value[..end])
}

fn unquote_token(value: &str) -> &str {
    let value = value.trim();
    if value.len() >= 2
        && ((value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\'')))
    {
        &value[1..value.len() - 1]
    } else {
        value
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_int_index_is_not_blacklisted_at_high_cardinality() {
        let mut idx = FieldIndex::default();
        for i in 0..150_000u64 {
            idx.insert_value("order", &serde_json::json!(i), i);
        }

        let key = IndexKey::Int(1);
        assert_eq!(idx.get("order", &key).map(|ids| ids.as_slice()), Some(&[1][..]));

        let high = IndexKey::Int(149_999);
        assert_eq!(
            idx.get("order", &high).map(|ids| ids.as_slice()),
            Some(&[149_999][..])
        );
    }

    #[test]
    fn sparse_index_is_blacklisted_at_high_cardinality() {
        let mut idx = FieldIndex::default();
        for i in 0..=MAX_INDEX_UNIQUE_VALUES as u64 {
            idx.insert_value("uuid", &serde_json::json!(format!("value_{i}")), i);
        }

        assert!(idx.blacklisted.contains("uuid"));
        assert!(idx.get("uuid", &IndexKey::Str("value_0".to_string())).is_none());
    }

    #[test]
    fn parse_indexed_in_and_or_equalities() {
        let (field, keys) = parse_indexed_in("\"order\" IN (1, 2)").unwrap();
        assert_eq!(field, "order");
        assert_eq!(keys.len(), 2);

        let (field, keys) = parse_indexed_or_equalities("\"order\" = 1 OR \"order\" = 2").unwrap();
        assert_eq!(field, "order");
        assert_eq!(keys.len(), 2);

        let leaves =
            parse_indexed_or_leaves("\"order\" = 1 OR test = 'test_1'").unwrap();
        assert_eq!(leaves.len(), 2);
    }

    #[test]
    fn query_from_index_supports_in_and_or() {
        let mut idx = FieldIndex::default();
        for i in 0..10u64 {
            idx.insert_value("order", &serde_json::json!(i), i);
        }

        let mut ids_in = HashSet::new();
        for key in [IndexKey::Int(1), IndexKey::Int(2)] {
            if let Some(set) = idx.get("order", &key) {
                ids_in.extend(set.iter().copied());
            }
        }
        let mut expected_in: Vec<u64> = ids_in.into_iter().collect();
        expected_in.sort_unstable();
        assert_eq!(expected_in, vec![1, 2]);
    }
}
