//! Database engine module.
//!
//! Orchestrates collections, storage, indexing, and search operations.
//! This is the main entry point for all database operations from Python.

use crate::distance::DistanceMetric;
use crate::error::{LynseError, Result};
use crate::index::{self, IndexType, SearchParams, VectorIndex};
use crate::storage::dtype::{
    decode_vector_bytes_to_f32, encode_f32_slice_as_le_bytes, f16_bits_to_f32, VectorDtype,
};
use crate::storage::field_store::FieldStore;
use crate::storage::polarvec_mmap::{
    parse_bits, PolarVecIndex, DEFAULT_BITS as POLARVEC_DEFAULT_BITS,
    DEFAULT_OVERSAMPLE as POLARVEC_OVERSAMPLE,
};
use crate::storage::pq_mmap::{parse_n_subspaces, PQIndex, DEFAULT_OVERSAMPLE as PQ_OVERSAMPLE};
use crate::storage::rabitq_mmap::{RaBitQIndex, DEFAULT_OVERSAMPLE as RABITQ_OVERSAMPLE};
use crate::storage::vector_store::VectorStore;
use crate::storage::wal::WALStorage;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

const STORAGE_MANIFEST_FILE: &str = "storage_manifest.json";
const STABLE_FIELD_IDS_FILE: &str = "fields.stable_ids";
const SNAPSHOT_MANIFEST_FILE: &str = "snapshot_manifest.json";
const DATABASE_SNAPSHOT_MANIFEST_FILE: &str = "database_snapshot_manifest.json";
const COLLECTION_EXPORT_MANIFEST_FILE: &str = "export_manifest.json";
const COLLECTION_EXPORT_VECTORS_FILE: &str = "vectors.f32";
const COLLECTION_EXPORT_METADATA_FILE: &str = "metadata.jsonl";
const VECTOR_FIELDS_DIR: &str = "vector_fields";
const VECTOR_FIELDS_MANIFEST_FILE: &str = "manifest.json";
const DEFAULT_VECTOR_FIELD_NAME: &str = "default";
const EXTERNAL_ID_MAP_FILE: &str = "external_id_map.json";
const EXTERNAL_ID_MAP_DELTA_FILE: &str = "external_id_map.delta.jsonl";
const EXTERNAL_ID_MAP_BIN_FILE: &str = "external_id_map.bin";
const EXTERNAL_ID_MAP_DELTA_BIN_FILE: &str = "external_id_map.delta.bin";
const EXTERNAL_ID_MAP_VERSION: u32 = 1;
const SPARSE_VECTORS_FILE: &str = "sparse_vectors.jsonl";
const TEXT_INDEX_FILE: &str = "text_index.bin";
const LEGACY_TEXT_INDEX_FILE: &str = "text_index.json";
const TOMBSTONE_FILE: &str = "tombstone.bin";
const TEXT_INDEX_MAGIC: &[u8; 4] = b"LTX2";
const TEXT_INDEX_FORMAT_VERSION: u32 = 1;
const STORAGE_FORMAT_NAME: &str = "lynsedb-collection";
const DATABASE_SNAPSHOT_FORMAT_NAME: &str = "lynsedb-database";
const COLLECTION_EXPORT_FORMAT_NAME: &str = "lynsedb-collection-jsonl-binary-export";
const STORAGE_FORMAT_VERSION: u32 = 2;
const WAL_FORMAT_VERSION: u32 = 4;
const PENDING_INGEST_FLUSH_ROWS: usize = 10_000;
const PENDING_INGEST_FLUSH_BYTES: usize = 32 * 1024 * 1024;

#[cfg(test)]
static SNAPSHOT_COPY_DELAY_MS: AtomicU64 = AtomicU64::new(0);

/// Collection metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMeta {
    pub name: String,
    pub dimension: usize,
    pub chunk_size: usize,
    pub index_mode: Option<String>,
    pub dtypes: String,
}

/// A single vector collection, managing vectors + fields + index + WAL.
pub struct Collection {
    pub meta: CollectionMeta,
    path: PathBuf,
    vector_dtype: VectorDtype,
    vector_store: VectorStore,
    field_store: FieldStore,
    /// True when FieldStore records are keyed by stable internal IDs instead of rows.
    fields_use_stable_ids: bool,
    wal: WALStorage,
    pending_ingest: Mutex<PendingIngestBuffer>,
    index: Option<Box<dyn VectorIndex>>,
    index_mode: Option<String>,
    /// Fingerprint of the last index sync (for incremental updates)
    last_sync_fingerprint: Option<String>,
    /// PQ index for FLAT-*-PQ index modes.
    pq_index: Option<PQIndex>,
    /// RaBitQ index for FLAT-*-RABITQ index modes.
    rabitq_index: Option<RaBitQIndex>,
    /// PolarVec index for FLAT-*-POLARVEC index modes.
    polarvec_index: Option<PolarVecIndex>,
    named_vector_fields: HashMap<String, NamedVectorField>,
    sparse_vectors: SparseVectorStore,
    text_index: InvertedTextIndex,
    /// Soft-delete tombstone: IDs marked for deletion, filtered from search results.
    tombstone: Arc<RwLock<HashSet<u64>>>,
    /// User-facing ID map: id_map[row_offset] = user_id.
    /// Persisted to id_map.bin. Backfilled with sequential IDs for backward compat.
    id_map: Vec<u64>,
    /// Reverse map: internal_id → row_offset. Rebuilt from id_map on load.
    reverse_id_map: HashMap<u64, usize>,
    /// Public external ID → internal numeric ID.
    external_to_internal_id: HashMap<ExternalId, u64>,
    /// Internal numeric ID → public external ID.
    internal_to_external_id: HashMap<u64, ExternalId>,
    /// Next automatically assigned internal ID.
    next_internal_id: u64,
    /// Holds the collection-level writer lock until close/drop.
    lock: Option<FileLock>,
    read_only: bool,
}

/// Public user-facing record ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExternalId {
    Int(u64),
    String(String),
}

#[derive(Default)]
struct PendingIngestBuffer {
    encoded_vectors: Vec<u8>,
    vectors: Vec<f32>,
    ids: Vec<u64>,
    row_offsets: Vec<u64>,
    vector_dtype: VectorDtype,
    dim: usize,
}

#[derive(Clone)]
struct PendingIngestSnapshot {
    vectors: Vec<f32>,
    row_offsets: Vec<u64>,
}

impl PendingIngestBuffer {
    fn len(&self) -> usize {
        self.ids.len()
    }

    fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn encoded_bytes(&self) -> usize {
        self.encoded_vectors.len()
    }

    fn append(
        &mut self,
        encoded_vectors: &[u8],
        decoded_vectors: &[f32],
        vector_dtype: VectorDtype,
        dim: usize,
        ids: &[u64],
        row_offsets: &[u64],
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        if self.dim == 0 {
            self.dim = dim;
            self.vector_dtype = vector_dtype;
        } else if self.dim != dim || self.vector_dtype != vector_dtype {
            return Err(LynseError::InvalidArgument(
                "pending ingest buffer cannot mix vector dimensions or dtypes".to_string(),
            ));
        }
        self.encoded_vectors.extend_from_slice(encoded_vectors);
        self.vectors.extend_from_slice(decoded_vectors);
        self.ids.extend_from_slice(ids);
        self.row_offsets.extend_from_slice(row_offsets);
        Ok(())
    }

    fn should_flush(&self) -> bool {
        self.len() >= PENDING_INGEST_FLUSH_ROWS
            || self.encoded_bytes() >= PENDING_INGEST_FLUSH_BYTES
    }

    fn snapshot(&self) -> PendingIngestSnapshot {
        PendingIngestSnapshot {
            vectors: self.vectors.clone(),
            row_offsets: self.row_offsets.clone(),
        }
    }

    fn take(&mut self) -> Option<(Vec<u8>, VectorDtype, usize, Vec<u64>)> {
        if self.is_empty() {
            return None;
        }
        let encoded_vectors = std::mem::take(&mut self.encoded_vectors);
        let ids = std::mem::take(&mut self.ids);
        self.vectors.clear();
        self.row_offsets.clear();
        let dtype = self.vector_dtype;
        let dim = self.dim;
        self.vector_dtype = VectorDtype::default();
        self.dim = 0;
        Some((encoded_vectors, dtype, dim, ids))
    }
}

impl ExternalId {
    fn validate(&self) -> Result<()> {
        match self {
            Self::String(value) if value.is_empty() => Err(LynseError::InvalidArgument(
                "string IDs cannot be empty".to_string(),
            )),
            _ => Ok(()),
        }
    }
}

impl std::fmt::Display for ExternalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(value) => write!(f, "{value}"),
            Self::String(value) => write!(f, "{value}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExternalIdMapFile {
    version: u32,
    next_internal_id: u64,
    entries: Vec<ExternalIdEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExternalIdEntry {
    internal_id: u64,
    external_id: ExternalId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExternalIdMapBinaryFile {
    version: u32,
    next_internal_id: u64,
    entries: Vec<ExternalIdBinaryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExternalIdBinaryEntry {
    internal_id: u64,
    external_id: ExternalIdBinary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ExternalIdBinary {
    Int(u64),
    String(String),
}

impl From<ExternalId> for ExternalIdBinary {
    fn from(value: ExternalId) -> Self {
        match value {
            ExternalId::Int(value) => Self::Int(value),
            ExternalId::String(value) => Self::String(value),
        }
    }
}

impl From<ExternalIdBinary> for ExternalId {
    fn from(value: ExternalIdBinary) -> Self {
        match value {
            ExternalIdBinary::Int(value) => Self::Int(value),
            ExternalIdBinary::String(value) => Self::String(value),
        }
    }
}

impl From<&ExternalIdMapFile> for ExternalIdMapBinaryFile {
    fn from(value: &ExternalIdMapFile) -> Self {
        Self {
            version: value.version,
            next_internal_id: value.next_internal_id,
            entries: value
                .entries
                .iter()
                .cloned()
                .map(|entry| ExternalIdBinaryEntry {
                    internal_id: entry.internal_id,
                    external_id: ExternalIdBinary::from(entry.external_id),
                })
                .collect(),
        }
    }
}

impl From<ExternalIdMapBinaryFile> for ExternalIdMapFile {
    fn from(value: ExternalIdMapBinaryFile) -> Self {
        Self {
            version: value.version,
            next_internal_id: value.next_internal_id,
            entries: value
                .entries
                .into_iter()
                .map(|entry| ExternalIdEntry {
                    internal_id: entry.internal_id,
                    external_id: ExternalId::from(entry.external_id),
                })
                .collect(),
        }
    }
}

#[cfg(unix)]
struct FileLock {
    _file: std::fs::File,
}

#[cfg(unix)]
impl FileLock {
    fn exclusive(path: &Path) -> Result<Self> {
        Self::acquire(path, libc::LOCK_EX)
    }

    fn shared(path: &Path) -> Result<Self> {
        Self::acquire(path, libc::LOCK_SH)
    }

    fn acquire(path: &Path, lock_kind: libc::c_int) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        let rc = unsafe { libc::flock(file.as_raw_fd(), lock_kind | libc::LOCK_NB) };
        if rc == 0 {
            return Ok(Self { _file: file });
        }

        let err = std::io::Error::last_os_error();
        let raw = err.raw_os_error();
        if err.kind() == std::io::ErrorKind::WouldBlock
            || raw == Some(libc::EWOULDBLOCK)
            || raw == Some(libc::EAGAIN)
        {
            return Err(LynseError::Storage(format!(
                "path is already open by another writer; lock file: {}",
                path.display()
            )));
        }

        Err(err.into())
    }
}

#[cfg(unix)]
impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = unsafe { libc::flock(self._file.as_raw_fd(), libc::LOCK_UN) };
    }
}

#[cfg(not(unix))]
struct FileLock;

#[cfg(not(unix))]
impl FileLock {
    fn exclusive(_path: &Path) -> Result<Self> {
        Ok(Self)
    }

    fn shared(_path: &Path) -> Result<Self> {
        Ok(Self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenMode {
    ReadWrite,
    ReadOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StorageManifest {
    format: String,
    version: u32,
    collection_name: String,
    dimension: usize,
    chunk_size: usize,
    vector_dtype: String,
    id_map_encoding: String,
    wal_version: u32,
    field_store: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CollectionSnapshotManifest {
    snapshot_type: String,
    collection_name: String,
    storage_format: String,
    storage_version: u32,
    dimension: usize,
    chunk_size: usize,
}

impl CollectionSnapshotManifest {
    fn current(collection: &Collection) -> Self {
        Self {
            snapshot_type: "lynsedb-collection-snapshot".to_string(),
            collection_name: collection.meta.name.clone(),
            storage_format: STORAGE_FORMAT_NAME.to_string(),
            storage_version: STORAGE_FORMAT_VERSION,
            dimension: collection.meta.dimension,
            chunk_size: collection.meta.chunk_size,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatabaseSnapshotManifest {
    snapshot_type: String,
    database_name: String,
    storage_format: String,
    storage_version: u32,
    collections: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CollectionExportManifest {
    export_type: String,
    collection_name: String,
    export_format: String,
    export_version: u32,
    dimension: usize,
    chunk_size: usize,
    count: usize,
    vector_dtype: String,
    byte_order: String,
    vectors_file: String,
    metadata_file: String,
}

impl CollectionExportManifest {
    fn current(collection: &Collection, count: usize) -> Self {
        let vector_dtype = collection.vector_dtype;
        Self {
            export_type: "lynsedb-collection-export".to_string(),
            collection_name: collection.meta.name.clone(),
            export_format: COLLECTION_EXPORT_FORMAT_NAME.to_string(),
            export_version: if vector_dtype == VectorDtype::F16 {
                STORAGE_FORMAT_VERSION
            } else {
                1
            },
            dimension: collection.meta.dimension,
            chunk_size: collection.meta.chunk_size,
            count,
            vector_dtype: vector_dtype.storage_name().to_string(),
            byte_order: "little-endian".to_string(),
            vectors_file: COLLECTION_EXPORT_VECTORS_FILE.to_string(),
            metadata_file: COLLECTION_EXPORT_METADATA_FILE.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CollectionExportRecord {
    id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    external_id: Option<ExternalId>,
    #[serde(default)]
    field: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFieldConfig {
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    pub index_mode: String,
    #[serde(default = "default_vector_dtype_string")]
    pub dtypes: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct VectorFieldsManifest {
    #[serde(default)]
    fields: Vec<VectorFieldConfig>,
}

struct NamedVectorField {
    config: VectorFieldConfig,
    vector_store: VectorStore,
    index: Option<Box<dyn VectorIndex>>,
    id_map: Vec<u64>,
    reverse_id_map: HashMap<u64, usize>,
    path: PathBuf,
}

fn default_vector_dtype_string() -> String {
    VectorDtype::F32.storage_name().to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SparseVectorRecord {
    id: u64,
    indices: Vec<u32>,
    values: Vec<f32>,
}

impl SparseVectorRecord {
    fn from_entries(id: u64, entries: &[(u32, f32)]) -> Self {
        Self {
            id,
            indices: entries.iter().map(|(idx, _)| *idx).collect(),
            values: entries.iter().map(|(_, value)| *value).collect(),
        }
    }
}

#[derive(Debug, Clone)]
struct SparseVectorStore {
    path: PathBuf,
    vectors: HashMap<u64, Vec<(u32, f32)>>,
}

impl SparseVectorStore {
    fn load(path: PathBuf) -> Result<Self> {
        let mut store = Self {
            path,
            vectors: HashMap::new(),
        };
        if !store.path.exists() {
            return Ok(store);
        }

        let content = std::fs::read_to_string(&store.path)?;
        for (line_no, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let record: SparseVectorRecord = serde_json::from_str(line).map_err(|e| {
                LynseError::Serialization(format!(
                    "failed to parse sparse vector record at line {}: {}",
                    line_no + 1,
                    e
                ))
            })?;
            if record.indices.len() != record.values.len() {
                return Err(LynseError::Storage(format!(
                    "sparse vector record for id {} has {} indices but {} values",
                    record.id,
                    record.indices.len(),
                    record.values.len()
                )));
            }
            let entries: Vec<(u32, f32)> = record
                .indices
                .into_iter()
                .zip(record.values.into_iter())
                .collect();
            let normalized = normalize_sparse_entries(&entries)?;
            if normalized.is_empty() {
                store.vectors.remove(&record.id);
            } else {
                store.vectors.insert(record.id, normalized);
            }
        }

        Ok(store)
    }

    fn upsert_many(&mut self, ids: &[u64], vectors: &[Vec<(u32, f32)>]) -> Result<()> {
        if ids.len() != vectors.len() {
            return Err(LynseError::InvalidArgument(format!(
                "ids length ({}) must match sparse vector count ({})",
                ids.len(),
                vectors.len()
            )));
        }

        let mut next = self.vectors.clone();
        for (&id, vector) in ids.iter().zip(vectors.iter()) {
            let normalized = normalize_sparse_entries(vector)?;
            if normalized.is_empty() {
                next.remove(&id);
            } else {
                next.insert(id, normalized);
            }
        }

        self.write_records(&next)?;
        self.vectors = next;
        Ok(())
    }

    fn remove_ids(&mut self, ids: &HashSet<u64>) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }

        let mut next = self.vectors.clone();
        let before = next.len();
        next.retain(|id, _| !ids.contains(id));
        if before == next.len() {
            return Ok(());
        }

        self.write_records(&next)?;
        self.vectors = next;
        Ok(())
    }

    fn search(
        &self,
        query: &[(u32, f32)],
        k: usize,
        allowed_ids: Option<&HashSet<u64>>,
        tombstones: &HashSet<u64>,
    ) -> Vec<(u64, f32)> {
        if query.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut scored = Vec::new();
        for (&id, vector) in &self.vectors {
            if tombstones.contains(&id) {
                continue;
            }
            if let Some(allowed) = allowed_ids {
                if !allowed.contains(&id) {
                    continue;
                }
            }

            let score = sparse_inner_product(query, vector);
            if score != 0.0 {
                scored.push((id, score));
            }
        }

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(k);
        scored
    }

    fn write_records(&self, vectors: &HashMap<u64, Vec<(u32, f32)>>) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut ids: Vec<u64> = vectors.keys().copied().collect();
        ids.sort_unstable();

        let mut bytes = Vec::new();
        for id in ids {
            let Some(entries) = vectors.get(&id) else {
                continue;
            };
            let record = SparseVectorRecord::from_entries(id, entries);
            serde_json::to_writer(&mut bytes, &record)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            bytes.push(b'\n');
        }

        Collection::atomic_write_file(&self.path, &bytes)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InvertedTextIndexData {
    #[serde(default = "default_text_index_version")]
    version: u32,
    #[serde(default)]
    postings: HashMap<String, HashMap<u64, HashMap<String, u32>>>,
    #[serde(default)]
    doc_lengths: HashMap<u64, HashMap<String, usize>>,
}

#[derive(Serialize)]
struct InvertedTextIndexDataRef<'a> {
    version: u32,
    postings: &'a HashMap<String, HashMap<u64, HashMap<String, u32>>>,
    doc_lengths: &'a HashMap<u64, HashMap<String, usize>>,
    field_names: &'a HashMap<String, usize>,
}

fn default_text_index_version() -> u32 {
    TEXT_INDEX_FORMAT_VERSION
}

#[derive(Debug, Clone)]
struct InvertedTextIndex {
    path: PathBuf,
    postings: HashMap<String, HashMap<u64, HashMap<String, u32>>>,
    doc_lengths: HashMap<u64, HashMap<String, usize>>,
    doc_total_lengths: HashMap<u64, usize>,
    doc_count_by_field: HashMap<String, usize>,
    total_doc_len_by_field: HashMap<String, usize>,
    all_doc_count: usize,
    all_total_doc_len: usize,
    df_cache: Arc<RwLock<HashMap<String, usize>>>,
    loaded_from_disk: bool,
}

impl InvertedTextIndex {
    fn load(path: PathBuf) -> Result<Self> {
        let (index_path, bytes, legacy_json) = if path.exists() {
            (path.clone(), Some(std::fs::read(&path)?), false)
        } else {
            let legacy_path = path.with_file_name(LEGACY_TEXT_INDEX_FILE);
            if legacy_path.exists() {
                (path.clone(), Some(std::fs::read(&legacy_path)?), true)
            } else {
                (path.clone(), None, false)
            }
        };

        let Some(bytes) = bytes else {
            return Ok(Self {
                path: index_path,
                postings: HashMap::new(),
                doc_lengths: HashMap::new(),
                doc_total_lengths: HashMap::new(),
                doc_count_by_field: HashMap::new(),
                total_doc_len_by_field: HashMap::new(),
                all_doc_count: 0,
                all_total_doc_len: 0,
                df_cache: Arc::new(RwLock::new(HashMap::new())),
                loaded_from_disk: false,
            });
        };

        let data: InvertedTextIndexData = if legacy_json {
            serde_json::from_slice(&bytes).map_err(|e| LynseError::Serialization(e.to_string()))?
        } else if bytes.starts_with(TEXT_INDEX_MAGIC) {
            decode_compact_text_index(&bytes)?
        } else {
            bincode::deserialize(&bytes).map_err(|e| LynseError::Serialization(e.to_string()))?
        };
        if data.version > TEXT_INDEX_FORMAT_VERSION {
            return Err(LynseError::Storage(format!(
                "text index format version {} is newer than supported version {}",
                data.version, TEXT_INDEX_FORMAT_VERSION
            )));
        }

        let mut index = Self {
            path: index_path,
            postings: data.postings,
            doc_lengths: data.doc_lengths,
            doc_total_lengths: HashMap::new(),
            doc_count_by_field: HashMap::new(),
            total_doc_len_by_field: HashMap::new(),
            all_doc_count: 0,
            all_total_doc_len: 0,
            df_cache: Arc::new(RwLock::new(HashMap::new())),
            loaded_from_disk: true,
        };
        index.rebuild_length_stats();
        Ok(index)
    }

    fn needs_bootstrap(&self) -> bool {
        !self.loaded_from_disk
    }

    fn is_empty(&self) -> bool {
        self.doc_lengths.is_empty()
    }

    fn rebuild(
        &mut self,
        ids: &[u64],
        fields: &[HashMap<String, serde_json::Value>],
        persist: bool,
    ) -> Result<()> {
        if ids.len() != fields.len() {
            return Err(LynseError::InvalidArgument(format!(
                "ids length ({}) must match field count ({})",
                ids.len(),
                fields.len()
            )));
        }

        self.postings.clear();
        self.doc_lengths.clear();
        self.doc_total_lengths.clear();
        self.clear_length_stats();
        self.clear_df_cache();
        for (&id, field_map) in ids.iter().zip(fields.iter()) {
            self.index_document(id, field_map);
        }
        if persist {
            self.write_to_disk()?;
            self.loaded_from_disk = true;
        }
        Ok(())
    }

    fn upsert_documents(
        &mut self,
        ids: &[u64],
        fields: &[HashMap<String, serde_json::Value>],
        persist: bool,
    ) -> Result<()> {
        if ids.len() != fields.len() {
            return Err(LynseError::InvalidArgument(format!(
                "ids length ({}) must match field count ({})",
                ids.len(),
                fields.len()
            )));
        }

        if !ids.is_empty() {
            self.clear_df_cache();
        }
        for (&id, field_map) in ids.iter().zip(fields.iter()) {
            if self.doc_lengths.contains_key(&id) {
                self.remove_document(id);
            }
            self.index_document(id, field_map);
        }

        if persist {
            self.write_to_disk()?;
            self.loaded_from_disk = true;
        }
        Ok(())
    }

    fn insert_documents(
        &mut self,
        ids: &[u64],
        fields: &[HashMap<String, serde_json::Value>],
        persist: bool,
    ) -> Result<()> {
        if ids.len() != fields.len() {
            return Err(LynseError::InvalidArgument(format!(
                "ids length ({}) must match field count ({})",
                ids.len(),
                fields.len()
            )));
        }

        if !ids.is_empty() {
            self.clear_df_cache();
        }
        for (&id, field_map) in ids.iter().zip(fields.iter()) {
            self.index_document(id, field_map);
        }

        if persist {
            self.write_to_disk()?;
            self.loaded_from_disk = true;
        }
        Ok(())
    }

    fn remove_ids(&mut self, ids: &HashSet<u64>, persist: bool) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }

        self.clear_df_cache();
        for &id in ids {
            self.remove_document(id);
        }

        if persist {
            self.write_to_disk()?;
            self.loaded_from_disk = true;
        }
        Ok(())
    }

    fn search(
        &self,
        query_text: &str,
        text_fields: Option<&[String]>,
        limit: usize,
        allowed_ids: Option<&HashSet<u64>>,
        tombstones: &HashSet<u64>,
    ) -> Vec<(u64, f32)> {
        let query_terms = tokenize_text(query_text);
        if query_terms.is_empty() || limit == 0 {
            return Vec::new();
        }

        let field_filter = TextFieldSelection::from_fields(text_fields);

        let mut query_counts: HashMap<String, usize> = HashMap::new();
        for term in query_terms {
            *query_counts.entry(term).or_insert(0) += 1;
        }

        let Some((corpus_docs, total_doc_len)) =
            self.corpus_stats(&field_filter, allowed_ids, tombstones)
        else {
            return Vec::new();
        };
        let avg_doc_len = (total_doc_len as f32 / corpus_docs as f32).max(1.0);
        let n_docs = corpus_docs as f32;

        let mut document_frequencies = HashMap::new();
        for term in query_counts.keys() {
            let df = self
                .postings
                .get(term)
                .map(|posting| {
                    self.document_frequency(term, posting, &field_filter, allowed_ids, tombstones)
                })
                .unwrap_or(0);
            if df > 0 {
                document_frequencies.insert(term.as_str(), df);
            }
        }

        if document_frequencies.is_empty() {
            return Vec::new();
        }

        let min_df = document_frequencies
            .values()
            .copied()
            .min()
            .unwrap_or(corpus_docs);
        // High-frequency query terms often match nearly the whole corpus. Seed
        // candidates from selective terms when available, then score candidates
        // with the full query below.
        let selective_cutoff = (min_df.saturating_mul(4)).max(limit.saturating_mul(8));
        let candidate_terms: Vec<&String> = query_counts
            .keys()
            .filter(|term| {
                document_frequencies
                    .get(term.as_str())
                    .map(|df| *df < corpus_docs && *df <= selective_cutoff)
                    .unwrap_or(false)
            })
            .collect();
        let candidate_terms = if candidate_terms.is_empty() {
            query_counts.keys().collect()
        } else {
            candidate_terms
        };

        let mut candidate_ids = HashSet::new();
        for term in candidate_terms {
            let Some(posting) = self.postings.get(term) else {
                continue;
            };
            for (&id, tf_by_field) in posting {
                if !text_doc_allowed(id, allowed_ids, tombstones) {
                    continue;
                }
                if selected_term_frequency(tf_by_field, &field_filter) > 0 {
                    candidate_ids.insert(id);
                }
            }
        }

        if candidate_ids.is_empty() {
            return Vec::new();
        }

        let k1 = 1.2f32;
        let b = 0.75f32;

        let mut scored = Vec::new();
        for id in candidate_ids {
            let Some(lengths) = self.doc_lengths.get(&id) else {
                continue;
            };
            let doc_len = selected_doc_length(lengths, &field_filter);
            if doc_len == 0 {
                continue;
            }

            let mut score = 0.0f32;
            for (term, query_count) in &query_counts {
                let tf = self
                    .postings
                    .get(term)
                    .and_then(|posting| posting.get(&id))
                    .map(|tf_by_field| selected_term_frequency(tf_by_field, &field_filter))
                    .unwrap_or(0) as f32;
                if tf == 0.0 {
                    continue;
                }

                let df = *document_frequencies.get(term.as_str()).unwrap_or(&0) as f32;
                let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();
                let denom = tf + k1 * (1.0 - b + b * doc_len as f32 / avg_doc_len);
                score += *query_count as f32 * idf * (tf * (k1 + 1.0)) / denom;
            }

            if score > 0.0 {
                scored.push((id, score));
            }
        }

        sort_truncate_scores_desc(&mut scored, limit);
        scored
    }

    fn document_frequency(
        &self,
        term: &str,
        posting: &HashMap<u64, HashMap<String, u32>>,
        field_filter: &TextFieldSelection<'_>,
        allowed_ids: Option<&HashSet<u64>>,
        tombstones: &HashSet<u64>,
    ) -> usize {
        if allowed_ids.is_none() && tombstones.is_empty() {
            let cache_key = field_filter.cache_key(term);
            if let Some(cached) = self.df_cache.read().get(&cache_key).copied() {
                return cached;
            }
            let df = posting_document_frequency(posting, field_filter, None, tombstones);
            self.df_cache.write().insert(cache_key, df);
            return df;
        }

        posting_document_frequency(posting, field_filter, allowed_ids, tombstones)
    }

    fn index_document(&mut self, id: u64, fields: &HashMap<String, serde_json::Value>) {
        let mut lengths = HashMap::with_capacity(fields.len());
        for (field, value) in fields {
            let mut term_counts = HashMap::with_capacity(8);
            append_searchable_json_terms(value, &mut term_counts);
            let field_len: usize = term_counts.values().map(|count| *count as usize).sum();
            if field_len == 0 {
                continue;
            }

            let field_name = field.clone();
            lengths.insert(field_name.clone(), field_len);
            for (term, tf) in term_counts {
                self.postings
                    .entry(term)
                    .or_default()
                    .entry(id)
                    .or_default()
                    .insert(field_name.clone(), tf);
            }
        }

        if !lengths.is_empty() {
            self.add_length_stats(&lengths);
            let total: usize = lengths.values().copied().sum();
            if total > 0 {
                self.doc_total_lengths.insert(id, total);
            }
            self.doc_lengths.insert(id, lengths);
        }
    }

    fn remove_document(&mut self, id: u64) {
        if let Some(lengths) = self.doc_lengths.remove(&id) {
            self.remove_length_stats(&lengths);
        }
        self.doc_total_lengths.remove(&id);
        self.postings.retain(|_, posting| {
            posting.remove(&id);
            !posting.is_empty()
        });
    }

    fn clear_length_stats(&mut self) {
        self.doc_count_by_field.clear();
        self.total_doc_len_by_field.clear();
        self.all_doc_count = 0;
        self.all_total_doc_len = 0;
    }

    fn clear_df_cache(&self) {
        self.df_cache.write().clear();
    }

    fn rebuild_length_stats(&mut self) {
        self.clear_length_stats();
        self.doc_total_lengths.clear();
        let lengths: Vec<(u64, HashMap<String, usize>)> = self
            .doc_lengths
            .iter()
            .map(|(&id, lengths)| (id, lengths.clone()))
            .collect();
        for (id, item) in lengths {
            let total: usize = item.values().copied().sum();
            if total > 0 {
                self.doc_total_lengths.insert(id, total);
            }
            self.add_length_stats(&item);
        }
    }

    fn add_length_stats(&mut self, lengths: &HashMap<String, usize>) {
        let total: usize = lengths.values().copied().sum();
        if total > 0 {
            self.all_doc_count += 1;
            self.all_total_doc_len += total;
        }
        for (field, len) in lengths {
            if *len == 0 {
                continue;
            }
            *self.doc_count_by_field.entry(field.clone()).or_insert(0) += 1;
            *self
                .total_doc_len_by_field
                .entry(field.clone())
                .or_insert(0) += *len;
        }
    }

    fn remove_length_stats(&mut self, lengths: &HashMap<String, usize>) {
        let total: usize = lengths.values().copied().sum();
        if total > 0 {
            self.all_doc_count = self.all_doc_count.saturating_sub(1);
            self.all_total_doc_len = self.all_total_doc_len.saturating_sub(total);
        }
        for (field, len) in lengths {
            if *len == 0 {
                continue;
            }
            if let Some(count) = self.doc_count_by_field.get_mut(field) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.doc_count_by_field.remove(field);
                }
            }
            if let Some(total_len) = self.total_doc_len_by_field.get_mut(field) {
                *total_len = total_len.saturating_sub(*len);
                if *total_len == 0 {
                    self.total_doc_len_by_field.remove(field);
                }
            }
        }
    }

    fn corpus_stats(
        &self,
        field_filter: &TextFieldSelection<'_>,
        allowed_ids: Option<&HashSet<u64>>,
        tombstones: &HashSet<u64>,
    ) -> Option<(usize, usize)> {
        if allowed_ids.is_none() && tombstones.is_empty() {
            match field_filter {
                TextFieldSelection::All => {
                    return (self.all_doc_count > 0)
                        .then_some((self.all_doc_count, self.all_total_doc_len));
                }
                TextFieldSelection::One(field) => {
                    let count = self.doc_count_by_field.get(*field).copied().unwrap_or(0);
                    let total = self
                        .total_doc_len_by_field
                        .get(*field)
                        .copied()
                        .unwrap_or(0);
                    return (count > 0).then_some((count, total));
                }
                TextFieldSelection::Many(_) => {}
            }
        }

        let mut corpus_docs = 0usize;
        let mut total_doc_len = 0usize;
        if let Some(allowed) = allowed_ids {
            match field_filter {
                TextFieldSelection::All => {
                    for &id in allowed {
                        if tombstones.contains(&id) {
                            continue;
                        }
                        if let Some(&len) = self.doc_total_lengths.get(&id) {
                            if len > 0 {
                                corpus_docs += 1;
                                total_doc_len += len;
                            }
                        }
                    }
                }
                _ => {
                    for &id in allowed {
                        if tombstones.contains(&id) {
                            continue;
                        }
                        let Some(lengths) = self.doc_lengths.get(&id) else {
                            continue;
                        };
                        let len = selected_doc_length(lengths, field_filter);
                        if len > 0 {
                            corpus_docs += 1;
                            total_doc_len += len;
                        }
                    }
                }
            }
        } else {
            match field_filter {
                TextFieldSelection::All => {
                    for (&id, &len) in &self.doc_total_lengths {
                        if tombstones.contains(&id) {
                            continue;
                        }
                        if len > 0 {
                            corpus_docs += 1;
                            total_doc_len += len;
                        }
                    }
                }
                _ => {
                    for (&id, lengths) in &self.doc_lengths {
                        if tombstones.contains(&id) {
                            continue;
                        }
                        let len = selected_doc_length(lengths, field_filter);
                        if len > 0 {
                            corpus_docs += 1;
                            total_doc_len += len;
                        }
                    }
                }
            }
        }
        (corpus_docs > 0).then_some((corpus_docs, total_doc_len))
    }

    fn write_to_disk(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let data = InvertedTextIndexDataRef {
            version: TEXT_INDEX_FORMAT_VERSION,
            postings: &self.postings,
            doc_lengths: &self.doc_lengths,
            field_names: &self.doc_count_by_field,
        };
        let bytes = encode_compact_text_index(&data)?;
        Collection::atomic_write_file(&self.path, &bytes)?;
        let legacy_path = self.path.with_file_name(LEGACY_TEXT_INDEX_FILE);
        if legacy_path != self.path {
            let _ = std::fs::remove_file(legacy_path);
        }
        Ok(())
    }

    fn write_to_disk_if_present(&self) -> Result<()> {
        if self.is_empty() && !self.path.exists() {
            return Ok(());
        }
        self.write_to_disk()
    }
}

fn encode_compact_text_index(data: &InvertedTextIndexDataRef<'_>) -> Result<Vec<u8>> {
    let mut fields: Vec<&str> = data.field_names.keys().map(String::as_str).collect();
    fields.sort_unstable();

    let mut out = Vec::new();
    out.extend_from_slice(TEXT_INDEX_MAGIC);
    write_var_u64(&mut out, data.version as u64);
    write_var_u64(&mut out, fields.len() as u64);
    for field in &fields {
        write_compact_string(&mut out, field);
    }

    write_var_u64(&mut out, data.postings.len() as u64);
    for (term, posting) in data.postings {
        write_compact_string(&mut out, term);
        write_var_u64(&mut out, posting.len() as u64);

        if posting.len() == 1 {
            if let Some((&id, tf_by_field)) = posting.iter().next() {
                write_var_u64(&mut out, id);
                write_compact_u32_field_entries(&mut out, &fields, tf_by_field);
            }
            continue;
        }

        let mut docs: Vec<(u64, &HashMap<String, u32>)> =
            posting.iter().map(|(id, fields)| (*id, fields)).collect();
        docs.sort_by_key(|(id, _)| *id);

        let mut previous_id = 0u64;
        for (id, tf_by_field) in docs {
            write_var_u64(&mut out, id.saturating_sub(previous_id));
            previous_id = id;
            write_compact_u32_field_entries(&mut out, &fields, tf_by_field);
        }
    }

    let mut docs: Vec<(u64, &HashMap<String, usize>)> = data
        .doc_lengths
        .iter()
        .map(|(id, lengths)| (*id, lengths))
        .collect();
    docs.sort_by_key(|(id, _)| *id);
    write_var_u64(&mut out, docs.len() as u64);
    let mut previous_id = 0u64;
    for (id, lengths) in docs {
        write_var_u64(&mut out, id.saturating_sub(previous_id));
        previous_id = id;

        write_compact_usize_field_entries(&mut out, &fields, lengths);
    }

    Ok(out)
}

fn write_compact_u32_field_entries(
    out: &mut Vec<u8>,
    fields: &[&str],
    values: &HashMap<String, u32>,
) {
    let count = fields
        .iter()
        .filter(|field| values.get(**field).copied().unwrap_or(0) > 0)
        .count();
    write_var_u64(out, count as u64);
    for (field_id, field) in fields.iter().enumerate() {
        let Some(value) = values.get(*field).copied().filter(|value| *value > 0) else {
            continue;
        };
        write_var_u64(out, field_id as u64);
        write_var_u64(out, value as u64);
    }
}

fn write_compact_usize_field_entries(
    out: &mut Vec<u8>,
    fields: &[&str],
    values: &HashMap<String, usize>,
) {
    let count = fields
        .iter()
        .filter(|field| values.get(**field).copied().unwrap_or(0) > 0)
        .count();
    write_var_u64(out, count as u64);
    for (field_id, field) in fields.iter().enumerate() {
        let Some(value) = values.get(*field).copied().filter(|value| *value > 0) else {
            continue;
        };
        write_var_u64(out, field_id as u64);
        write_var_u64(out, value as u64);
    }
}

fn decode_compact_text_index(bytes: &[u8]) -> Result<InvertedTextIndexData> {
    if !bytes.starts_with(TEXT_INDEX_MAGIC) {
        return Err(LynseError::Serialization(
            "invalid compact text index magic".to_string(),
        ));
    }

    let mut cursor = TEXT_INDEX_MAGIC.len();
    let version = read_var_u64(bytes, &mut cursor)?;
    if version > u32::MAX as u64 {
        return Err(LynseError::Serialization(
            "compact text index version is too large".to_string(),
        ));
    }

    let field_count = read_compact_len(bytes, &mut cursor)?;
    let mut fields = Vec::with_capacity(field_count);
    for _ in 0..field_count {
        fields.push(read_compact_string(bytes, &mut cursor)?);
    }

    let term_count = read_compact_len(bytes, &mut cursor)?;
    let mut postings = HashMap::with_capacity(term_count);
    for _ in 0..term_count {
        let term = read_compact_string(bytes, &mut cursor)?;
        let posting_count = read_compact_len(bytes, &mut cursor)?;
        let mut posting = HashMap::with_capacity(posting_count);
        let mut previous_id = 0u64;
        for _ in 0..posting_count {
            let delta = read_var_u64(bytes, &mut cursor)?;
            let id = previous_id.checked_add(delta).ok_or_else(|| {
                LynseError::Serialization("compact text index id delta overflow".to_string())
            })?;
            previous_id = id;

            let field_entry_count = read_compact_len(bytes, &mut cursor)?;
            let mut tf_by_field = HashMap::with_capacity(field_entry_count);
            for _ in 0..field_entry_count {
                let field_id = read_compact_len(bytes, &mut cursor)?;
                let field = fields.get(field_id).ok_or_else(|| {
                    LynseError::Serialization(format!(
                        "compact text index field id {} is out of range",
                        field_id
                    ))
                })?;
                let tf = read_var_u64(bytes, &mut cursor)?;
                if tf > u32::MAX as u64 {
                    return Err(LynseError::Serialization(
                        "compact text index term frequency is too large".to_string(),
                    ));
                }
                tf_by_field.insert(field.clone(), tf as u32);
            }
            posting.insert(id, tf_by_field);
        }
        postings.insert(term, posting);
    }

    let doc_count = read_compact_len(bytes, &mut cursor)?;
    let mut doc_lengths = HashMap::with_capacity(doc_count);
    let mut previous_id = 0u64;
    for _ in 0..doc_count {
        let delta = read_var_u64(bytes, &mut cursor)?;
        let id = previous_id.checked_add(delta).ok_or_else(|| {
            LynseError::Serialization("compact text index doc id delta overflow".to_string())
        })?;
        previous_id = id;

        let field_entry_count = read_compact_len(bytes, &mut cursor)?;
        let mut lengths = HashMap::with_capacity(field_entry_count);
        for _ in 0..field_entry_count {
            let field_id = read_compact_len(bytes, &mut cursor)?;
            let field = fields.get(field_id).ok_or_else(|| {
                LynseError::Serialization(format!(
                    "compact text index field id {} is out of range",
                    field_id
                ))
            })?;
            let len = read_var_u64(bytes, &mut cursor)?;
            if len > usize::MAX as u64 {
                return Err(LynseError::Serialization(
                    "compact text index document length is too large".to_string(),
                ));
            }
            lengths.insert(field.clone(), len as usize);
        }
        doc_lengths.insert(id, lengths);
    }

    Ok(InvertedTextIndexData {
        version: version as u32,
        postings,
        doc_lengths,
    })
}

fn write_compact_string(out: &mut Vec<u8>, value: &str) {
    write_var_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

fn read_compact_string(bytes: &[u8], cursor: &mut usize) -> Result<String> {
    let len = read_compact_len(bytes, cursor)?;
    let end = cursor.checked_add(len).ok_or_else(|| {
        LynseError::Serialization("compact text index string length overflow".to_string())
    })?;
    if end > bytes.len() {
        return Err(LynseError::Serialization(
            "compact text index string extends past end of file".to_string(),
        ));
    }
    let value = std::str::from_utf8(&bytes[*cursor..end])
        .map_err(|e| LynseError::Serialization(e.to_string()))?
        .to_string();
    *cursor = end;
    Ok(value)
}

fn read_compact_len(bytes: &[u8], cursor: &mut usize) -> Result<usize> {
    let value = read_var_u64(bytes, cursor)?;
    if value > usize::MAX as u64 {
        return Err(LynseError::Serialization(
            "compact text index length is too large".to_string(),
        ));
    }
    Ok(value as usize)
}

fn write_var_u64(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push(((value as u8) & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn read_var_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        if *cursor >= bytes.len() {
            return Err(LynseError::Serialization(
                "compact text index varint extends past end of file".to_string(),
            ));
        }
        let byte = bytes[*cursor];
        *cursor += 1;
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 64 {
            return Err(LynseError::Serialization(
                "compact text index varint is too large".to_string(),
            ));
        }
    }
}

impl DatabaseSnapshotManifest {
    fn current(database_name: &str, collections: Vec<String>) -> Self {
        Self {
            snapshot_type: "lynsedb-database-snapshot".to_string(),
            database_name: database_name.to_string(),
            storage_format: DATABASE_SNAPSHOT_FORMAT_NAME.to_string(),
            storage_version: STORAGE_FORMAT_VERSION,
            collections,
        }
    }
}

impl StorageManifest {
    fn current(name: &str, dimension: usize, chunk_size: usize, vector_dtype: VectorDtype) -> Self {
        Self {
            format: STORAGE_FORMAT_NAME.to_string(),
            version: STORAGE_FORMAT_VERSION,
            collection_name: name.to_string(),
            dimension,
            chunk_size,
            vector_dtype: vector_dtype.storage_name().to_string(),
            id_map_encoding: "u64-le-row-offset-map".to_string(),
            wal_version: WAL_FORMAT_VERSION,
            field_store: "apexbase".to_string(),
        }
    }
}

impl Collection {
    /// Get the collection path.
    pub fn collection_path(&self) -> &Path {
        &self.path
    }

    /// Get vector store info for zero-copy mmap access from Python.
    /// Returns (path, n_vectors, dimension).
    pub fn vector_store_info(&self) -> Result<(String, usize, usize)> {
        self.flush_pending_ingest()?;
        let (n, dim) = self.shape()?;
        let path = self.vector_store.contiguous_path()?;
        Ok((path.to_string_lossy().to_string(), n as usize, dim))
    }

    pub fn vector_dtype(&self) -> VectorDtype {
        self.vector_dtype
    }

    /// Create or open a collection.
    pub fn open(path: &Path, name: &str, dimension: usize, chunk_size: usize) -> Result<Self> {
        Self::open_with_mode(path, name, dimension, chunk_size, OpenMode::ReadWrite, None)
    }

    pub fn open_with_dtype(
        path: &Path,
        name: &str,
        dimension: usize,
        chunk_size: usize,
        vector_dtype: VectorDtype,
    ) -> Result<Self> {
        Self::open_with_mode(
            path,
            name,
            dimension,
            chunk_size,
            OpenMode::ReadWrite,
            Some(vector_dtype),
        )
    }

    /// Open an existing collection in read-only mode.
    pub fn open_read_only(
        path: &Path,
        name: &str,
        dimension: usize,
        chunk_size: usize,
    ) -> Result<Self> {
        Self::open_with_mode(path, name, dimension, chunk_size, OpenMode::ReadOnly, None)
    }

    fn open_with_mode(
        path: &Path,
        name: &str,
        dimension: usize,
        chunk_size: usize,
        mode: OpenMode,
        requested_dtype: Option<VectorDtype>,
    ) -> Result<Self> {
        let collection_path = path.join(name);
        if mode == OpenMode::ReadWrite {
            std::fs::create_dir_all(&collection_path)?;
        } else if !collection_path.exists() {
            return Err(LynseError::CollectionNotFound(name.to_string()));
        }

        let lock_path = collection_path.join(".writer.lock");
        let lock = if mode == OpenMode::ReadWrite {
            FileLock::exclusive(&lock_path)?
        } else {
            FileLock::shared(&lock_path)?
        };

        if mode == OpenMode::ReadOnly && !Self::storage_manifest_path(&collection_path).exists() {
            return Err(LynseError::Storage(
                "read-only open requires storage_manifest.json; open writable once to create it"
                    .to_string(),
            ));
        }
        let manifest = Self::ensure_storage_manifest(
            &collection_path,
            name,
            dimension,
            chunk_size,
            requested_dtype,
        )?;
        let vector_dtype = VectorDtype::parse(&manifest.vector_dtype)?;
        let dimension = if dimension == 0 {
            manifest.dimension
        } else {
            dimension
        };
        let chunk_size = if chunk_size == 0 {
            manifest.chunk_size
        } else {
            chunk_size
        };

        let vector_store = VectorStore::new(&collection_path, dimension, chunk_size, vector_dtype)?;

        let field_db_path = collection_path.join("fields_db");
        let fields_use_stable_ids = collection_path.join(STABLE_FIELD_IDS_FILE).exists();
        let field_table = if fields_use_stable_ids {
            "fields_v2"
        } else {
            "fields"
        };
        let field_store = FieldStore::new(&field_db_path, field_table)?;

        // Initialize WAL
        let wal = WALStorage::new(name, chunk_size, &collection_path, 5000)?;

        // Try to load existing index
        let index_path = collection_path.join("index");
        let index_meta_path = collection_path.join("index_meta");
        std::fs::create_dir_all(&index_path)?;
        std::fs::create_dir_all(&index_meta_path)?;

        let (index, index_mode) = Self::try_load_index(&index_meta_path, &index_path)?;

        // Load last sync fingerprint
        let last_sync_fingerprint = Self::load_sync_fingerprint(&collection_path);

        let meta = CollectionMeta {
            name: name.to_string(),
            dimension,
            chunk_size,
            index_mode: index_mode.clone(),
            dtypes: vector_dtype.storage_name().to_string(),
        };

        let tombstone_path = collection_path.join("tombstone.bin");
        let tombstone = Arc::new(RwLock::new(Self::load_tombstone_from_disk(&tombstone_path)));
        let sparse_vectors = SparseVectorStore::load(collection_path.join(SPARSE_VECTORS_FILE))?;
        let text_index = InvertedTextIndex::load(collection_path.join(TEXT_INDEX_FILE))?;
        let text_index_needs_bootstrap = text_index.needs_bootstrap();

        let mut coll = Self {
            meta,
            path: collection_path,
            vector_dtype,
            vector_store,
            field_store,
            fields_use_stable_ids,
            wal,
            pending_ingest: Mutex::new(PendingIngestBuffer::default()),
            index,
            index_mode,
            last_sync_fingerprint,
            pq_index: None,
            rabitq_index: None,
            polarvec_index: None,
            named_vector_fields: HashMap::new(),
            sparse_vectors,
            text_index,
            tombstone,
            id_map: Vec::new(),
            reverse_id_map: HashMap::new(),
            external_to_internal_id: HashMap::new(),
            internal_to_external_id: HashMap::new(),
            next_internal_id: 0,
            lock: Some(lock),
            read_only: mode == OpenMode::ReadOnly,
        };

        if mode == OpenMode::ReadWrite {
            // A positional update publishes its replay journal before touching
            // live segment bytes. Reapply it before loading IDs or WAL records.
            let recovered_vector_updates = coll.vector_store.recover_pending_updates()?;

            // Establish the durable row boundary before WAL replay. If a previous
            // process crashed after appending vectors but before appending id_map,
            // id_map is the authoritative set of fully applied rows.
            let id_map_repaired = coll.initialize_id_map_for_open()?;

            coll.migrate_fields_to_stable_ids()?;

            // Load external IDs before WAL replay because public IDs are
            // persisted before the WAL in the record add path.
            coll.load_external_id_map_for_recovery()?;

            // Recover any uncommitted WAL data on startup.
            let recovered_wal = coll.recover_wal()?;

            let external_map_repaired = coll.repair_external_id_map_for_open()?;

            if recovered_vector_updates
                || id_map_repaired
                || recovered_wal
                || external_map_repaired
            {
                if let Some(mode) = coll.index_mode.clone() {
                    coll.build_index(&mode)?;
                }
            }
        } else {
            if coll.vector_store.has_pending_updates() {
                return Err(LynseError::Storage(
                    "collection has pending vector updates; open it writable to recover before using read-only mode"
                        .to_string(),
                ));
            }
            coll.initialize_id_map_for_read_only()?;
            coll.initialize_external_id_map_for_read_only()?;
            if coll.has_uncommitted_data() {
                return Err(LynseError::Storage(
                    "collection has uncommitted WAL data; open it writable to recover before using read-only mode".to_string(),
                ));
            }
        }

        // Load quantizer indices from disk if present
        coll.try_load_pq_rabitq()?;

        coll.named_vector_fields = Self::load_named_vector_fields(&coll.path, chunk_size, mode)?;
        if text_index_needs_bootstrap && !coll.id_map.is_empty() {
            coll.rebuild_text_index(mode == OpenMode::ReadWrite)?;
        }

        Ok(coll)
    }

    fn storage_manifest_path(collection_path: &Path) -> PathBuf {
        collection_path.join(STORAGE_MANIFEST_FILE)
    }

    fn ensure_storage_manifest(
        collection_path: &Path,
        name: &str,
        dimension: usize,
        chunk_size: usize,
        requested_dtype: Option<VectorDtype>,
    ) -> Result<StorageManifest> {
        let manifest_path = Self::storage_manifest_path(collection_path);
        if manifest_path.exists() {
            let bytes = std::fs::read(&manifest_path)?;
            let mut manifest: StorageManifest = serde_json::from_slice(&bytes)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;

            if manifest.format != STORAGE_FORMAT_NAME {
                return Err(LynseError::Storage(format!(
                    "unsupported storage format '{}'",
                    manifest.format
                )));
            }
            if manifest.version > STORAGE_FORMAT_VERSION {
                return Err(LynseError::Storage(format!(
                    "collection uses storage format version {}, but this binary supports up to {}",
                    manifest.version, STORAGE_FORMAT_VERSION
                )));
            }
            if manifest.collection_name != name {
                return Err(LynseError::Storage(format!(
                    "collection manifest name '{}' does not match requested collection '{}'",
                    manifest.collection_name, name
                )));
            }
            if dimension != 0 && manifest.dimension == 0 {
                manifest.dimension = dimension;
                Self::write_storage_manifest(&manifest_path, &manifest)?;
            } else if dimension != 0 && manifest.dimension != dimension {
                return Err(LynseError::DimensionMismatch {
                    expected: manifest.dimension,
                    got: dimension,
                });
            }
            if chunk_size != 0 && manifest.chunk_size != chunk_size {
                return Err(LynseError::Storage(format!(
                    "collection chunk_size {} does not match requested chunk_size {}",
                    manifest.chunk_size, chunk_size
                )));
            }
            let manifest_dtype = VectorDtype::parse(&manifest.vector_dtype)?;
            if let Some(dtype) = requested_dtype {
                if manifest_dtype != dtype {
                    return Err(LynseError::InvalidArgument(format!(
                        "collection vector dtype {} does not match requested dtype {}",
                        manifest_dtype.storage_name(),
                        dtype.storage_name()
                    )));
                }
            }
            return Ok(manifest);
        }

        let vector_dtype = requested_dtype.unwrap_or_default();
        let manifest = StorageManifest::current(name, dimension, chunk_size, vector_dtype);
        Self::write_storage_manifest(&manifest_path, &manifest)?;
        Ok(manifest)
    }

    fn write_storage_manifest(path: &Path, manifest: &StorageManifest) -> Result<()> {
        let json = serde_json::to_vec_pretty(manifest)
            .map_err(|e| LynseError::Serialization(e.to_string()))?;
        Self::atomic_write_file(path, &json)?;
        Ok(())
    }

    fn update_parent_collection_config_dimension(&self, dimension: usize) -> Result<()> {
        let Some(db_path) = self.path.parent() else {
            return Ok(());
        };
        let config_path = db_path.join("collections.json");
        if !config_path.exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(&config_path)?;
        let mut configs: HashMap<String, CollectionConfig> =
            serde_json::from_str(&content).unwrap_or_default();
        let Some(config) = configs.get_mut(&self.meta.name) else {
            return Ok(());
        };
        if config.dim == dimension {
            return Ok(());
        }

        config.dim = dimension;
        let json = serde_json::to_vec_pretty(&configs)
            .map_err(|e| LynseError::Serialization(e.to_string()))?;
        Self::atomic_write_file(&config_path, &json)
    }

    fn initialize_dimension_if_needed(&mut self, dimension: usize) -> Result<()> {
        if dimension == 0 {
            return Err(LynseError::InvalidArgument(
                "vector dimension must be greater than zero".to_string(),
            ));
        }
        if self.meta.dimension == dimension {
            return Ok(());
        }
        if self.meta.dimension != 0 {
            return Err(LynseError::DimensionMismatch {
                expected: self.meta.dimension,
                got: dimension,
            });
        }

        let (stored_rows, _) = self.vector_store.get_shape()?;
        if stored_rows != 0 || self.pending_len() != 0 || !self.id_map.is_empty() {
            return Err(LynseError::Storage(
                "cannot initialize dimension for a collection that already contains data"
                    .to_string(),
            ));
        }

        let manifest = StorageManifest::current(
            &self.meta.name,
            dimension,
            self.meta.chunk_size,
            self.vector_dtype,
        );
        Self::write_storage_manifest(&Self::storage_manifest_path(&self.path), &manifest)?;
        self.meta.dimension = dimension;
        self.vector_store = VectorStore::new(
            &self.path,
            dimension,
            self.meta.chunk_size,
            self.vector_dtype,
        )?;
        self.update_parent_collection_config_dimension(dimension)?;
        Ok(())
    }

    fn infer_dimension_from_f32(&mut self, vectors: &[f32], n_vectors: usize) -> Result<usize> {
        if n_vectors == 0 {
            if !vectors.is_empty() {
                return Err(LynseError::InvalidArgument(
                    "vectors must be empty when n_vectors is zero".to_string(),
                ));
            }
            return Ok(self.meta.dimension);
        }
        if vectors.len() % n_vectors != 0 {
            return Err(LynseError::InvalidArgument(format!(
                "vector value count ({}) must be divisible by n_vectors ({})",
                vectors.len(),
                n_vectors
            )));
        }
        let dimension = vectors.len() / n_vectors;
        self.initialize_dimension_if_needed(dimension)?;
        Ok(dimension)
    }

    fn infer_dimension_from_encoded(
        &mut self,
        encoded_vectors: &[u8],
        vector_dtype: VectorDtype,
        n_vectors: usize,
    ) -> Result<usize> {
        if n_vectors == 0 {
            if !encoded_vectors.is_empty() {
                return Err(LynseError::InvalidArgument(
                    "encoded vectors must be empty when n_vectors is zero".to_string(),
                ));
            }
            return Ok(self.meta.dimension);
        }

        let width = vector_dtype.byte_width();
        if encoded_vectors.len() % width != 0 {
            return Err(LynseError::InvalidArgument(format!(
                "encoded vector byte count ({}) is not aligned to dtype width ({})",
                encoded_vectors.len(),
                width
            )));
        }
        let values = encoded_vectors.len() / width;
        if values % n_vectors != 0 {
            return Err(LynseError::InvalidArgument(format!(
                "encoded vector value count ({}) must be divisible by n_vectors ({})",
                values, n_vectors
            )));
        }
        let dimension = values / n_vectors;
        self.initialize_dimension_if_needed(dimension)?;
        Ok(dimension)
    }

    fn vector_fields_root(collection_path: &Path) -> PathBuf {
        collection_path.join(VECTOR_FIELDS_DIR)
    }

    fn vector_fields_manifest_path(collection_path: &Path) -> PathBuf {
        Self::vector_fields_root(collection_path).join(VECTOR_FIELDS_MANIFEST_FILE)
    }

    fn validate_vector_field_name(name: &str) -> Result<String> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(LynseError::InvalidArgument(
                "vector field name cannot be empty".to_string(),
            ));
        }
        if trimmed == DEFAULT_VECTOR_FIELD_NAME {
            return Err(LynseError::InvalidArgument(
                "'default' is reserved for the primary collection vector".to_string(),
            ));
        }
        if trimmed.len() > 128 {
            return Err(LynseError::InvalidArgument(
                "vector field name cannot exceed 128 characters".to_string(),
            ));
        }
        if !trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(LynseError::InvalidArgument(
                "vector field name may only contain ASCII letters, digits, '_' and '-'".to_string(),
            ));
        }
        Ok(trimmed.to_string())
    }

    fn normalize_field_index_mode(
        index_mode: Option<&str>,
        metric: DistanceMetric,
    ) -> Result<String> {
        let mode = index_mode
            .map(|m| m.trim())
            .filter(|m| !m.is_empty())
            .map(|m| m.to_uppercase())
            .unwrap_or_else(|| match metric {
                DistanceMetric::L2Squared => "FLAT-L2".to_string(),
                DistanceMetric::Cosine => "FLAT-COS".to_string(),
                DistanceMetric::InnerProduct => "FLAT-IP".to_string(),
                DistanceMetric::Hamming => "FLAT-HAMMING".to_string(),
                DistanceMetric::Jaccard => "FLAT-JACCARD".to_string(),
            });
        if index::resolve_index_type(&mode).is_none() {
            return Err(LynseError::InvalidArgument(format!(
                "unknown vector field index mode '{}'",
                mode
            )));
        }
        Ok(mode)
    }

    fn resolve_named_vector_metric(
        metric: Option<&str>,
        index_mode: Option<&str>,
    ) -> Result<DistanceMetric> {
        if let Some(index_mode) = index_mode {
            let index_metric = Self::metric_from_mode_str(index_mode);
            if let Some(metric) = metric {
                let parsed = DistanceMetric::from_str(metric).ok_or_else(|| {
                    LynseError::InvalidArgument(format!(
                        "unsupported vector field metric '{}'",
                        metric
                    ))
                })?;
                if parsed != index_metric {
                    return Err(LynseError::InvalidArgument(format!(
                        "vector field metric '{}' does not match index mode '{}'",
                        metric, index_mode
                    )));
                }
            }
            Ok(index_metric)
        } else if let Some(metric) = metric {
            DistanceMetric::from_str(metric).ok_or_else(|| {
                LynseError::InvalidArgument(format!("unsupported vector field metric '{}'", metric))
            })
        } else {
            Ok(DistanceMetric::InnerProduct)
        }
    }

    fn load_vector_fields_manifest(path: &Path) -> Result<VectorFieldsManifest> {
        if !path.exists() {
            return Ok(VectorFieldsManifest::default());
        }
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| LynseError::Serialization(e.to_string()))
    }

    fn load_named_vector_fields(
        collection_path: &Path,
        chunk_size: usize,
        mode: OpenMode,
    ) -> Result<HashMap<String, NamedVectorField>> {
        let manifest_path = Self::vector_fields_manifest_path(collection_path);
        let manifest = Self::load_vector_fields_manifest(&manifest_path)?;
        let root = Self::vector_fields_root(collection_path);
        let mut fields = HashMap::new();

        for mut config in manifest.fields {
            let name = Self::validate_vector_field_name(&config.name)?;
            let field_path = root.join(&name);
            let vector_dtype = VectorDtype::parse(&config.dtypes)?;
            config.dtypes = vector_dtype.storage_name().to_string();
            let store = VectorStore::new(&field_path, config.dimension, chunk_size, vector_dtype)?;
            let index_path = field_path.join("index");
            let index_meta_path = field_path.join("index_meta");
            if mode == OpenMode::ReadWrite {
                std::fs::create_dir_all(&index_path)?;
                std::fs::create_dir_all(&index_meta_path)?;
            }
            let (index, loaded_index_mode) = Self::try_load_index(&index_meta_path, &index_path)?;
            if let Some(index_mode) = loaded_index_mode {
                config.index_mode = index_mode;
                config.metric = Self::metric_from_mode_str(&config.index_mode)
                    .name()
                    .to_string();
            }
            let (n_vecs, dim) = store.get_shape()?;
            if dim != config.dimension {
                return Err(LynseError::DimensionMismatch {
                    expected: config.dimension,
                    got: dim,
                });
            }

            let id_map_path = store.id_map_path();
            let mut id_map = Self::load_id_map_path(&id_map_path);
            let n_vecs = n_vecs as usize;
            if id_map.len() < n_vecs {
                if mode == OpenMode::ReadOnly {
                    return Err(LynseError::Storage(format!(
                        "named vector field '{}' needs writable recovery before read-only open",
                        name
                    )));
                }
                store.truncate_to_vectors(id_map.len())?;
            } else if id_map.len() > n_vecs {
                if mode == OpenMode::ReadOnly {
                    return Err(LynseError::Storage(format!(
                        "named vector field '{}' id map exceeds vector rows",
                        name
                    )));
                }
                id_map.truncate(n_vecs);
                Self::write_id_map_path(&id_map_path, &id_map)?;
            }

            fields.insert(
                name.clone(),
                NamedVectorField {
                    config: VectorFieldConfig {
                        name: name.clone(),
                        ..config
                    },
                    vector_store: store,
                    index,
                    reverse_id_map: Self::build_reverse_id_map(&id_map),
                    id_map,
                    path: field_path,
                },
            );
        }

        Ok(fields)
    }

    fn save_named_vector_fields_manifest(&self) -> Result<()> {
        let root = Self::vector_fields_root(&self.path);
        std::fs::create_dir_all(&root)?;
        let mut fields: Vec<VectorFieldConfig> = self
            .named_vector_fields
            .values()
            .map(|field| field.config.clone())
            .collect();
        fields.sort_by(|a, b| a.name.cmp(&b.name));
        let manifest = VectorFieldsManifest { fields };
        let bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| LynseError::Serialization(e.to_string()))?;
        Self::atomic_write_file(&root.join(VECTOR_FIELDS_MANIFEST_FILE), &bytes)?;
        Ok(())
    }

    fn atomic_write_file(path: &Path, data: &[u8]) -> Result<()> {
        let tmp_path = path.with_extension("tmp");
        std::fs::write(&tmp_path, data)?;
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    fn atomic_write_file_durable(path: &Path, data: &[u8]) -> Result<()> {
        use std::io::Write;

        let tmp_path = path.with_extension("tmp");
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp_path)?;
        file.write_all(data)?;
        file.sync_all()?;
        std::fs::rename(&tmp_path, path)?;
        if let Some(parent) = path.parent() {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    }

    fn sync_path_recursively(path: &Path) -> Result<()> {
        if !path.exists() {
            return Ok(());
        }

        let metadata = std::fs::metadata(path)?;
        if metadata.is_file() {
            let file = std::fs::OpenOptions::new().read(true).open(path)?;
            file.sync_all()?;
        } else if metadata.is_dir() {
            for entry in std::fs::read_dir(path)? {
                Self::sync_path_recursively(&entry?.path())?;
            }
            if let Ok(dir) = std::fs::File::open(path) {
                let _ = dir.sync_all();
            }
        }

        Ok(())
    }

    fn sync_file_if_exists(path: &Path) -> Result<()> {
        if path.exists() && std::fs::metadata(path)?.is_file() {
            let file = std::fs::OpenOptions::new().read(true).open(path)?;
            file.sync_all()?;
        }
        Ok(())
    }

    fn sync_dir_if_exists(path: &Path) {
        if path.exists() {
            if let Ok(dir) = std::fs::File::open(path) {
                let _ = dir.sync_all();
            }
        }
    }

    fn sync_collection_files(&self) -> Result<()> {
        Self::sync_path_recursively(&self.path)
    }

    fn sync_checkpoint_files(&self) -> Result<()> {
        for name in [
            "vectors.bin",
            "info.json",
            "fingerprint",
            "storage_manifest.json",
            EXTERNAL_ID_MAP_BIN_FILE,
            TOMBSTONE_FILE,
            SPARSE_VECTORS_FILE,
            TEXT_INDEX_FILE,
            LEGACY_TEXT_INDEX_FILE,
            "sync_fingerprint",
            "vector_manifest.json",
        ] {
            Self::sync_file_if_exists(&self.path.join(name))?;
        }
        Self::sync_file_if_exists(&self.vector_store.id_map_path())?;
        Self::sync_path_recursively(&self.path.join("vector_segments"))?;

        Self::sync_path_recursively(&self.path.join("fields_db"))?;
        Self::sync_path_recursively(&self.path.join("index_meta"))?;
        Self::sync_path_recursively(&self.path.join("index"))?;
        Self::sync_path_recursively(&Self::vector_fields_root(&self.path))?;

        Self::sync_dir_if_exists(&self.path);
        Ok(())
    }

    fn persist_vector_store_metadata(&self) -> Result<()> {
        self.vector_store.persist_metadata()?;
        for field in self.named_vector_fields.values() {
            field.vector_store.persist_metadata()?;
        }
        Ok(())
    }

    fn rebuild_text_index(&mut self, persist: bool) -> Result<()> {
        let field_keys = self.field_lookup_keys(&self.id_map);
        let fields = self.field_store.retrieve_many(&field_keys)?;
        let user_ids = self.id_map.clone();
        self.text_index.rebuild(&user_ids, &fields, persist)
    }

    fn make_temp_sibling(path: &Path) -> Result<PathBuf> {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)?;
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("snapshot");
        Ok(parent.join(format!(
            ".{}.tmp-{}",
            name,
            uuid::Uuid::new_v4().to_string().replace("-", "")
        )))
    }

    fn should_skip_snapshot_entry(file_name: &str) -> bool {
        file_name == ".writer.lock"
            || file_name == ".database.lock"
            || file_name == ".manager.lock"
            || file_name == SNAPSHOT_MANIFEST_FILE
            || file_name == DATABASE_SNAPSHOT_MANIFEST_FILE
            || file_name == COLLECTION_EXPORT_MANIFEST_FILE
            || file_name.ends_with(".wal")
            || file_name.ends_with(".tmp")
            || file_name.contains(".tmp-")
    }

    fn copy_dir_for_snapshot(src: &Path, dst: &Path) -> Result<()> {
        std::fs::create_dir_all(dst)?;
        #[cfg(test)]
        {
            let delay_ms = SNAPSHOT_COPY_DELAY_MS.load(Ordering::Relaxed);
            if delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            }
        }
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();
            if Self::should_skip_snapshot_entry(&file_name_str) {
                continue;
            }

            let src_path = entry.path();
            let dst_path = dst.join(&file_name);
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                Self::copy_dir_for_snapshot(&src_path, &dst_path)?;
            } else if metadata.is_file() {
                std::fs::copy(&src_path, &dst_path)?;
            }
        }
        Ok(())
    }

    fn read_storage_manifest_from_dir(path: &Path) -> Result<StorageManifest> {
        let manifest_path = path.join(STORAGE_MANIFEST_FILE);
        let bytes = std::fs::read(&manifest_path)?;
        serde_json::from_slice(&bytes).map_err(|e| LynseError::Serialization(e.to_string()))
    }

    fn read_snapshot_manifest(path: &Path) -> Result<CollectionSnapshotManifest> {
        let manifest_path = path.join(SNAPSHOT_MANIFEST_FILE);
        let bytes = std::fs::read(&manifest_path)?;
        serde_json::from_slice(&bytes).map_err(|e| LynseError::Serialization(e.to_string()))
    }

    fn read_export_manifest(path: &Path) -> Result<CollectionExportManifest> {
        let manifest_path = path.join(COLLECTION_EXPORT_MANIFEST_FILE);
        let bytes = std::fs::read(&manifest_path)?;
        serde_json::from_slice(&bytes).map_err(|e| LynseError::Serialization(e.to_string()))
    }

    fn validate_export_manifest(manifest: &CollectionExportManifest) -> Result<()> {
        if manifest.export_format != COLLECTION_EXPORT_FORMAT_NAME {
            return Err(LynseError::Storage(format!(
                "unsupported collection export format '{}'",
                manifest.export_format
            )));
        }
        if manifest.export_version > STORAGE_FORMAT_VERSION {
            return Err(LynseError::Storage(format!(
                "collection export uses format version {}, but this binary supports up to {}",
                manifest.export_version, STORAGE_FORMAT_VERSION
            )));
        }
        VectorDtype::parse(&manifest.vector_dtype)?;
        if manifest.byte_order != "little-endian" {
            return Err(LynseError::Storage(format!(
                "unsupported export byte order '{}'",
                manifest.byte_order
            )));
        }
        Ok(())
    }

    /// Recover uncommitted WAL segments into main storage.
    fn recover_wal(&mut self) -> Result<bool> {
        if !self.wal.has_uncommitted_data() {
            return Ok(false);
        }

        let segments = self.wal.get_segments()?;
        let mut recovered_any = false;
        for seg in &segments {
            if seg.dim != self.meta.dimension {
                return Err(LynseError::DimensionMismatch {
                    expected: self.meta.dimension,
                    got: seg.dim,
                });
            }

            let ids: Vec<u64> = if seg.ids.len() == seg.n_vectors {
                seg.ids.clone()
            } else {
                let start = self.max_id().map(|id| id + 1).unwrap_or(0);
                (start..start + seg.n_vectors as u64).collect()
            };

            let mut missing_vectors = Vec::new();
            let mut missing_encoded_vectors = Vec::new();
            let mut missing_ids = Vec::new();
            let mut missing_fields = Vec::new();
            let mut existing_vector_rows = Vec::new();
            let mut existing_encoded_vectors = Vec::new();
            let mut existing_decoded_vectors = Vec::new();
            let mut existing_field_rows = Vec::new();
            let mut existing_field_user_ids = Vec::new();
            let mut existing_field_values = Vec::new();
            let encoded_row_width = seg
                .dim
                .checked_mul(seg.dtype.byte_width())
                .ok_or_else(|| LynseError::Storage("WAL row byte size overflows".to_string()))?;

            for (i, &user_id) in ids.iter().enumerate() {
                if let Some(row) = self.reverse_id_map.get(&user_id).copied() {
                    let encoded_start = i * encoded_row_width;
                    let encoded_end = encoded_start + encoded_row_width;
                    if encoded_end > seg.encoded_data.len() {
                        return Err(LynseError::Storage(
                            "WAL segment encoded vector payload is shorter than expected"
                                .to_string(),
                        ));
                    }
                    let decoded_start = i * seg.dim;
                    let decoded_end = decoded_start + seg.dim;
                    if decoded_end > seg.data.len() {
                        return Err(LynseError::Storage(
                            "WAL segment vector payload is shorter than expected".to_string(),
                        ));
                    }
                    existing_vector_rows.push(row as u64);
                    existing_encoded_vectors
                        .extend_from_slice(&seg.encoded_data[encoded_start..encoded_end]);
                    existing_decoded_vectors.extend_from_slice(&seg.data[decoded_start..decoded_end]);
                    if let Some(field) = seg.fields.get(i) {
                        existing_field_rows.push(row as u64);
                        existing_field_user_ids.push(user_id);
                        existing_field_values.push(field.clone());
                    }
                    continue;
                }

                let start = i * seg.dim;
                let end = start + seg.dim;
                if end > seg.data.len() {
                    return Err(LynseError::Storage(
                        "WAL segment vector payload is shorter than expected".to_string(),
                    ));
                }
                missing_vectors.extend_from_slice(&seg.data[start..end]);
                let encoded_start = i * encoded_row_width;
                let encoded_end = encoded_start + encoded_row_width;
                if encoded_end > seg.encoded_data.len() {
                    return Err(LynseError::Storage(
                        "WAL segment encoded vector payload is shorter than expected".to_string(),
                    ));
                }
                missing_encoded_vectors
                    .extend_from_slice(&seg.encoded_data[encoded_start..encoded_end]);
                missing_ids.push(user_id);
                missing_fields.push(seg.fields.get(i).cloned().unwrap_or_default());
            }

            if !missing_ids.is_empty() {
                if seg.dtype == self.vector_dtype {
                    self.append_recovered_encoded_items(
                        &missing_vectors,
                        &missing_encoded_vectors,
                        seg.dtype,
                        &missing_ids,
                        &missing_fields,
                    )?;
                } else {
                    self.append_recovered_items(&missing_vectors, &missing_ids, &missing_fields)?;
                }
                recovered_any = true;
            }
            if !existing_vector_rows.is_empty() {
                let encoded;
                let encoded_vectors = if seg.dtype == self.vector_dtype {
                    existing_encoded_vectors.as_slice()
                } else {
                    encoded = encode_f32_slice_as_le_bytes(
                        &existing_decoded_vectors,
                        self.vector_dtype,
                    );
                    encoded.as_slice()
                };
                self.vector_store.overwrite_encoded_rows(
                    &existing_vector_rows,
                    encoded_vectors,
                    self.vector_dtype,
                )?;
                recovered_any = true;
            }
            if !existing_field_user_ids.is_empty() {
                let field_keys = if self.fields_use_stable_ids {
                    &existing_field_user_ids
                } else {
                    &existing_field_rows
                };
                self.field_store
                    .replace_fields_at_ids(field_keys, &existing_field_values)?;
                self.text_index.upsert_documents(
                    &existing_field_user_ids,
                    &existing_field_values,
                    false,
                )?;
                recovered_any = true;
            }
        }

        // Clean WAL after successful recovery
        if recovered_any {
            self.text_index.write_to_disk_if_present()?;
        }
        self.wal.cleanup()?;
        Ok(recovered_any || !segments.is_empty())
    }

    /// Load PQ / RaBitQ / PolarVec indices from disk if the index_mode requires them.
    fn try_load_pq_rabitq(&mut self) -> Result<()> {
        let upper = self
            .index_mode
            .as_ref()
            .map(|m| m.to_uppercase())
            .unwrap_or_default();

        if upper.contains("PQ") && !upper.contains("IVF") {
            let pq_path = self.path.join("pq_index.bin");
            if pq_path.exists() {
                match PQIndex::load(&pq_path) {
                    Ok(pq) => {
                        self.pq_index = Some(pq);
                    }
                    Err(e) => {
                        log::warn!("Failed to load PQ index: {}", e);
                    }
                }
            }
        } else if upper.contains("RABITQ") {
            let rq_path = self.path.join("rabitq_index.bin");
            if rq_path.exists() {
                match RaBitQIndex::load(&rq_path) {
                    Ok(rq) => {
                        self.rabitq_index = Some(rq);
                    }
                    Err(e) => {
                        log::warn!("Failed to load RaBitQ index: {}", e);
                    }
                }
            }
        } else if upper.contains("POLARVEC") {
            let pv_path = self.path.join("polarvec_index.bin");
            if pv_path.exists() {
                match PolarVecIndex::load(&pv_path) {
                    Ok(pv) => {
                        self.polarvec_index = Some(pv);
                    }
                    Err(e) => {
                        log::warn!("Failed to load PolarVec index: {}", e);
                    }
                }
            }
        }
        Ok(())
    }

    /// Load tombstone set from a binary file (sorted u64 LE values).
    fn load_tombstone_from_disk(path: &Path) -> HashSet<u64> {
        let Ok(bytes) = std::fs::read(path) else {
            return HashSet::new();
        };
        bytes
            .chunks_exact(8)
            .filter_map(|c| c.try_into().ok())
            .map(|b: [u8; 8]| u64::from_le_bytes(b))
            .collect()
    }

    /// Load an ID map. Partial trailing bytes are ignored.
    fn load_id_map_path(id_map_path: &Path) -> Vec<u64> {
        if id_map_path.exists() {
            std::fs::read(id_map_path)
                .ok()
                .map(|bytes| {
                    bytes
                        .chunks_exact(8)
                        .filter_map(|c| c.try_into().ok())
                        .map(|b: [u8; 8]| u64::from_le_bytes(b))
                        .collect()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    fn build_reverse_id_map(id_map: &[u64]) -> HashMap<u64, usize> {
        id_map
            .iter()
            .enumerate()
            .map(|(i, &uid)| (uid, i))
            .collect()
    }

    fn write_id_map_path(path: &Path, ids: &[u64]) -> Result<()> {
        let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        std::fs::write(path, &id_bytes)?;
        Ok(())
    }

    fn external_id_map_path(path: &Path) -> PathBuf {
        path.join(EXTERNAL_ID_MAP_FILE)
    }

    fn external_id_map_delta_path(path: &Path) -> PathBuf {
        path.join(EXTERNAL_ID_MAP_DELTA_FILE)
    }

    fn external_id_map_bin_path(path: &Path) -> PathBuf {
        path.join(EXTERNAL_ID_MAP_BIN_FILE)
    }

    fn external_id_map_delta_bin_path(path: &Path) -> PathBuf {
        path.join(EXTERNAL_ID_MAP_DELTA_BIN_FILE)
    }

    fn serialize_external_id_map_binary(file: &ExternalIdMapFile) -> Result<Vec<u8>> {
        let binary = ExternalIdMapBinaryFile::from(file);
        bincode::serialize(&binary).map_err(|e| LynseError::Serialization(e.to_string()))
    }

    fn deserialize_external_id_map_binary(bytes: &[u8]) -> Result<ExternalIdMapFile> {
        let binary: ExternalIdMapBinaryFile =
            bincode::deserialize(bytes).map_err(|e| LynseError::Serialization(e.to_string()))?;
        Ok(ExternalIdMapFile::from(binary))
    }

    fn validate_external_id_map_version(version: u32) -> Result<()> {
        if version > EXTERNAL_ID_MAP_VERSION {
            return Err(LynseError::Storage(format!(
                "external ID map version {} is newer than supported version {}",
                version, EXTERNAL_ID_MAP_VERSION
            )));
        }
        Ok(())
    }

    fn write_external_id_map(&self) -> Result<()> {
        let mut internal_ids: Vec<u64> = self.internal_to_external_id.keys().copied().collect();
        internal_ids.sort_unstable();
        let entries = internal_ids
            .into_iter()
            .filter_map(|internal_id| {
                self.internal_to_external_id
                    .get(&internal_id)
                    .cloned()
                    .filter(|external_id| !matches!(external_id, ExternalId::Int(value) if *value == internal_id))
                    .map(|external_id| ExternalIdEntry {
                        internal_id,
                        external_id,
                    })
            })
            .collect();
        let file = ExternalIdMapFile {
            version: EXTERNAL_ID_MAP_VERSION,
            next_internal_id: self.next_internal_id,
            entries,
        };
        let bytes = Self::serialize_external_id_map_binary(&file)?;
        Self::atomic_write_file(&Self::external_id_map_bin_path(&self.path), &bytes)?;
        for stale_path in [
            Self::external_id_map_delta_bin_path(&self.path),
            Self::external_id_map_path(&self.path),
            Self::external_id_map_delta_path(&self.path),
        ] {
            if stale_path.exists() {
                std::fs::remove_file(stale_path)?;
            }
        }
        Ok(())
    }

    fn append_external_id_map_delta(&self, entries: Vec<ExternalIdEntry>) -> Result<()> {
        let entries: Vec<ExternalIdEntry> = entries
            .into_iter()
            .filter(|entry| {
                !matches!(entry.external_id, ExternalId::Int(value) if value == entry.internal_id)
            })
            .collect();
        if entries.is_empty() {
            return Ok(());
        }
        let file = ExternalIdMapFile {
            version: EXTERNAL_ID_MAP_VERSION,
            next_internal_id: self.next_internal_id,
            entries,
        };
        let bytes = Self::serialize_external_id_map_binary(&file)?;
        let len = bytes.len() as u64;
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(Self::external_id_map_delta_bin_path(&self.path))?;
        f.write_all(&len.to_le_bytes())?;
        f.write_all(&bytes)?;
        f.flush()?;
        Ok(())
    }

    fn load_external_id_map_file(path: &Path) -> Result<Option<ExternalIdMapFile>> {
        let map_bin_path = Self::external_id_map_bin_path(path);
        let map_json_path = Self::external_id_map_path(path);
        let delta_bin_path = Self::external_id_map_delta_bin_path(path);
        let delta_json_path = Self::external_id_map_delta_path(path);
        if !map_bin_path.exists()
            && !map_json_path.exists()
            && !delta_bin_path.exists()
            && !delta_json_path.exists()
        {
            return Ok(None);
        }
        let mut next_internal_id = 0;
        let mut entries_by_internal_id: BTreeMap<u64, ExternalId> = BTreeMap::new();

        if map_bin_path.exists() {
            let bytes = std::fs::read(&map_bin_path)?;
            let file = Self::deserialize_external_id_map_binary(&bytes)?;
            Self::validate_external_id_map_version(file.version)?;
            next_internal_id = next_internal_id.max(file.next_internal_id);
            for entry in file.entries {
                entries_by_internal_id.insert(entry.internal_id, entry.external_id);
            }
        } else if map_json_path.exists() {
            let bytes = std::fs::read(&map_json_path)?;
            let file: ExternalIdMapFile = serde_json::from_slice(&bytes)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            Self::validate_external_id_map_version(file.version)?;
            next_internal_id = next_internal_id.max(file.next_internal_id);
            for entry in file.entries {
                entries_by_internal_id.insert(entry.internal_id, entry.external_id);
            }
        }

        if delta_bin_path.exists() {
            let bytes = std::fs::read(&delta_bin_path)?;
            let mut cursor = 0usize;
            while cursor < bytes.len() {
                if bytes.len() - cursor < std::mem::size_of::<u64>() {
                    return Err(LynseError::Storage(
                        "external ID map delta is truncated".to_string(),
                    ));
                }
                let mut len_bytes = [0u8; 8];
                len_bytes.copy_from_slice(&bytes[cursor..cursor + 8]);
                cursor += 8;
                let len = u64::from_le_bytes(len_bytes) as usize;
                if bytes.len() - cursor < len {
                    return Err(LynseError::Storage(
                        "external ID map delta frame is truncated".to_string(),
                    ));
                }
                let file = Self::deserialize_external_id_map_binary(&bytes[cursor..cursor + len])?;
                cursor += len;
                Self::validate_external_id_map_version(file.version)?;
                next_internal_id = next_internal_id.max(file.next_internal_id);
                for entry in file.entries {
                    entries_by_internal_id.insert(entry.internal_id, entry.external_id);
                }
            }
        }

        if delta_json_path.exists() {
            let text = std::fs::read_to_string(&delta_json_path)?;
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let file: ExternalIdMapFile = serde_json::from_str(line)
                    .map_err(|e| LynseError::Serialization(e.to_string()))?;
                Self::validate_external_id_map_version(file.version)?;
                next_internal_id = next_internal_id.max(file.next_internal_id);
                for entry in file.entries {
                    entries_by_internal_id.insert(entry.internal_id, entry.external_id);
                }
            }
        }

        let entries = entries_by_internal_id
            .into_iter()
            .map(|(internal_id, external_id)| ExternalIdEntry {
                internal_id,
                external_id,
            })
            .collect();
        Ok(Some(ExternalIdMapFile {
            version: EXTERNAL_ID_MAP_VERSION,
            next_internal_id,
            entries,
        }))
    }

    fn rebuild_external_lookup(
        &mut self,
        file: Option<ExternalIdMapFile>,
        persist_if_repaired: bool,
    ) -> Result<bool> {
        let live_internal_ids: HashSet<u64> = self.id_map.iter().copied().collect();
        let mut external_to_internal = HashMap::new();
        let mut internal_to_external = HashMap::new();
        let mut repaired = false;

        if let Some(file) = file {
            self.next_internal_id = file.next_internal_id;
            for entry in file.entries {
                if !live_internal_ids.contains(&entry.internal_id) {
                    repaired = true;
                    continue;
                }
                entry.external_id.validate()?;
                if internal_to_external.contains_key(&entry.internal_id)
                    || external_to_internal.contains_key(&entry.external_id)
                {
                    repaired = true;
                    continue;
                }
                external_to_internal.insert(entry.external_id.clone(), entry.internal_id);
                internal_to_external.insert(entry.internal_id, entry.external_id);
            }
        }

        for &internal_id in &self.id_map {
            if let std::collections::hash_map::Entry::Vacant(entry) =
                internal_to_external.entry(internal_id)
            {
                let external_id = ExternalId::Int(internal_id);
                entry.insert(external_id.clone());
                external_to_internal.insert(external_id, internal_id);
                repaired = true;
            }
        }

        let min_next = self
            .id_map
            .iter()
            .copied()
            .max()
            .map(|id| id.saturating_add(1))
            .unwrap_or(0);
        if self.next_internal_id < min_next {
            self.next_internal_id = min_next;
            repaired = true;
        }

        self.external_to_internal_id = external_to_internal;
        self.internal_to_external_id = internal_to_external;

        if repaired && persist_if_repaired && !self.read_only {
            self.write_external_id_map()?;
        }
        Ok(repaired)
    }

    fn load_external_id_map_for_recovery(&mut self) -> Result<()> {
        self.external_to_internal_id.clear();
        self.internal_to_external_id.clear();
        self.next_internal_id = self
            .id_map
            .iter()
            .copied()
            .max()
            .map(|id| id.saturating_add(1))
            .unwrap_or(0);

        let Some(file) = Self::load_external_id_map_file(&self.path)? else {
            return Ok(());
        };
        self.next_internal_id = file.next_internal_id;
        for entry in file.entries {
            entry.external_id.validate()?;
            if self
                .external_to_internal_id
                .insert(entry.external_id.clone(), entry.internal_id)
                .is_some()
                || self
                    .internal_to_external_id
                    .insert(entry.internal_id, entry.external_id)
                    .is_some()
            {
                return Err(LynseError::Storage(
                    "external ID map contains duplicate entries".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn repair_external_id_map_for_open(&mut self) -> Result<bool> {
        let entries = self
            .internal_to_external_id
            .iter()
            .map(|(&internal_id, external_id)| ExternalIdEntry {
                internal_id,
                external_id: external_id.clone(),
            })
            .collect();
        let file = ExternalIdMapFile {
            version: EXTERNAL_ID_MAP_VERSION,
            next_internal_id: self.next_internal_id,
            entries,
        };
        self.rebuild_external_lookup(Some(file), true)
    }

    /// Initialize id_map before WAL replay.
    ///
    /// If id_map exists, it is treated as the durable row boundary. Extra vector
    /// rows beyond it are truncated so WAL replay can safely re-apply them.
    /// If id_map is missing, we backfill sequential IDs for legacy data.
    fn initialize_id_map_for_open(&mut self) -> Result<bool> {
        let id_map_path = self.vector_store.id_map_path();
        let id_map_exists = id_map_path.exists();
        let (n_vecs, _) = self.vector_store.get_shape()?;
        let mut id_map = Self::load_id_map_path(&id_map_path);
        let mut repaired = false;

        if id_map_exists {
            let n_vecs = n_vecs as usize;
            let durable_rows = id_map.len().min(n_vecs);
            self.vector_store.truncate_to_vectors(durable_rows)?;
            if id_map.len() < n_vecs {
                repaired = true;
            } else if id_map.len() > n_vecs {
                id_map.truncate(n_vecs);
                Self::write_id_map_path(&id_map_path, &id_map)?;
                repaired = true;
            }
        } else {
            while id_map.len() < n_vecs as usize {
                id_map.push(id_map.len() as u64);
            }
            Self::write_id_map_path(&id_map_path, &id_map)?;
        }

        self.reverse_id_map = Self::build_reverse_id_map(&id_map);
        self.id_map = id_map;

        Ok(repaired)
    }

    fn initialize_id_map_for_read_only(&mut self) -> Result<()> {
        let id_map_path = self.vector_store.id_map_path();
        let id_map_exists = id_map_path.exists();
        let (n_vecs, _) = self.vector_store.get_shape()?;
        let n_vecs = n_vecs as usize;
        let mut id_map = Self::load_id_map_path(&id_map_path);

        if id_map_exists {
            if id_map.len() < n_vecs {
                return Err(LynseError::Storage(
                    "collection needs writable recovery before it can be opened read-only"
                        .to_string(),
                ));
            }
            id_map.truncate(n_vecs);
        } else {
            while id_map.len() < n_vecs {
                id_map.push(id_map.len() as u64);
            }
        }

        self.reverse_id_map = Self::build_reverse_id_map(&id_map);
        self.id_map = id_map;

        Ok(())
    }

    fn initialize_external_id_map_for_read_only(&mut self) -> Result<()> {
        let file = Self::load_external_id_map_file(&self.path)?;
        self.rebuild_external_lookup(file, false)?;
        Ok(())
    }

    fn ensure_writable(&self) -> Result<()> {
        if self.read_only {
            return Err(LynseError::InvalidArgument(
                "collection is opened read-only".to_string(),
            ));
        }
        Ok(())
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    fn allocate_internal_ids(&mut self, n: usize) -> Result<Vec<u64>> {
        let start = self.next_internal_id;
        let end = start
            .checked_add(n as u64)
            .ok_or_else(|| LynseError::InvalidArgument("internal ID allocation overflow".into()))?;
        self.next_internal_id = end;
        Ok((start..end).collect())
    }

    /// Append new stable IDs to the current generation's map.
    fn append_id_map_path(id_map_path: &Path, ids: &[u64]) -> Result<()> {
        use std::io::Write;
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(id_map_path)?;
        f.write_all(&bytes)?;
        Ok(())
    }

    /// Map an internal row offset to a user ID.
    #[inline]
    fn row_to_user_id(&self, row: u64) -> u64 {
        self.id_map.get(row as usize).copied().unwrap_or(row)
    }

    /// Map a user ID to an internal row offset.
    #[inline]
    fn user_id_to_row(&self, user_id: u64) -> Option<usize> {
        self.reverse_id_map.get(&user_id).copied()
    }

    fn migrate_fields_to_stable_ids(&mut self) -> Result<()> {
        if self.fields_use_stable_ids {
            return Ok(());
        }

        let row_offsets: Vec<u64> = (0..self.id_map.len() as u64).collect();
        let fields = self.field_store.retrieve_many(&row_offsets)?;
        let stable_store = FieldStore::new(&self.path.join("fields_db"), "fields_v2")?;
        stable_store.rebuild_at_ids(&self.id_map, &fields)?;
        stable_store.flush()?;
        Self::atomic_write_file_durable(
            &self.path.join(STABLE_FIELD_IDS_FILE),
            b"stable-internal-id-v1\n",
        )?;
        self.field_store = stable_store;
        self.fields_use_stable_ids = true;
        Ok(())
    }

    fn field_keys_to_rows(&self, keys: Vec<u64>) -> Vec<u64> {
        if !self.fields_use_stable_ids {
            return keys;
        }
        keys.into_iter()
            .filter_map(|id| self.user_id_to_row(id).map(|row| row as u64))
            .collect()
    }

    fn field_keys_to_user_ids(&self, keys: Vec<u64>) -> Vec<u64> {
        if self.fields_use_stable_ids {
            keys
        } else {
            keys.into_iter()
                .map(|row| self.row_to_user_id(row))
                .collect()
        }
    }

    fn field_lookup_keys(&self, user_ids: &[u64]) -> Vec<u64> {
        if self.fields_use_stable_ids {
            user_ids.to_vec()
        } else {
            user_ids
                .iter()
                .map(|&id| self.user_id_to_row(id).unwrap_or(id as usize) as u64)
                .collect()
        }
    }

    /// Check if a user ID exists in the collection.
    pub fn is_id_exists(&self, user_id: u64) -> bool {
        self.reverse_id_map.contains_key(&user_id)
    }

    pub fn is_external_id_exists(&self, external_id: &ExternalId) -> bool {
        self.external_to_internal_id.contains_key(external_id)
    }

    pub fn internal_id_for_external_id(&self, external_id: &ExternalId) -> Option<u64> {
        self.external_to_internal_id.get(external_id).copied()
    }

    pub fn internal_ids_for_external_ids(&self, external_ids: &[ExternalId]) -> Result<Vec<u64>> {
        external_ids
            .iter()
            .map(|external_id| {
                self.internal_id_for_external_id(external_id)
                    .ok_or_else(|| {
                        LynseError::InvalidArgument(format!(
                            "external id {} does not exist",
                            external_id
                        ))
                    })
            })
            .collect()
    }

    pub fn external_id_for_internal_id(&self, internal_id: u64) -> ExternalId {
        self.internal_to_external_id
            .get(&internal_id)
            .cloned()
            .unwrap_or(ExternalId::Int(internal_id))
    }

    pub fn external_ids_for_internal_ids(&self, internal_ids: &[u64]) -> Vec<ExternalId> {
        internal_ids
            .iter()
            .map(|&internal_id| self.external_id_for_internal_id(internal_id))
            .collect()
    }

    pub fn external_int_ids_for_internal_ids(&self, internal_ids: &[u64]) -> Option<Vec<i64>> {
        let mut out = Vec::with_capacity(internal_ids.len());
        for &internal_id in internal_ids {
            let value = match self.internal_to_external_id.get(&internal_id) {
                Some(ExternalId::Int(value)) => *value,
                Some(ExternalId::String(_)) => return None,
                None => internal_id,
            };
            out.push(i64::try_from(value).ok()?);
        }
        Some(out)
    }

    pub fn all_internal_ids(&self) -> Vec<u64> {
        self.id_map.clone()
    }

    /// Return the maximum user ID in the collection, or None if empty.
    pub fn max_id(&self) -> Option<u64> {
        self.id_map.iter().copied().max()
    }

    /// Persist tombstone set to disk as sorted u64 LE values.
    fn save_tombstone_to_disk(path: &Path, set: &HashSet<u64>) -> Result<()> {
        let mut ids: Vec<u64> = set.iter().copied().collect();
        ids.sort_unstable();
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Mark vectors as soft-deleted. Deleted IDs are excluded from all search results.
    pub fn delete_items(&self, ids: &[u64]) -> Result<()> {
        self.ensure_writable()?;
        let mut set = self.tombstone.write();
        for &id in ids {
            set.insert(id);
        }
        let path = self.path.join("tombstone.bin");
        Self::save_tombstone_to_disk(&path, &set)
    }

    /// Undelete previously soft-deleted vectors.
    pub fn restore_items(&self, ids: &[u64]) -> Result<()> {
        self.ensure_writable()?;
        let mut set = self.tombstone.write();
        for &id in ids {
            set.remove(&id);
        }
        let path = self.path.join("tombstone.bin");
        Self::save_tombstone_to_disk(&path, &set)
    }

    /// Return all currently soft-deleted IDs (sorted).
    pub fn list_deleted_ids(&self) -> Vec<u64> {
        let set = self.tombstone.read();
        let mut ids: Vec<u64> = set.iter().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Filter tombstoned IDs from (ids, dists) pairs and cap the result length.
    fn filter_tombstoned_limit(
        &self,
        ids: Vec<u64>,
        dists: Vec<f32>,
        limit: usize,
    ) -> (Vec<u64>, Vec<f32>) {
        let set = self.tombstone.read();
        let iter = ids.into_iter().zip(dists);
        let pairs: Vec<(u64, f32)> = if set.is_empty() {
            iter.take(limit).collect()
        } else {
            iter.filter(|(id, _)| !set.contains(id))
                .take(limit)
                .collect()
        };
        let mut out_ids = Vec::with_capacity(pairs.len());
        let mut out_dists = Vec::with_capacity(pairs.len());
        for (id, dist) in pairs {
            out_ids.push(id);
            out_dists.push(dist);
        }
        (out_ids, out_dists)
    }

    fn pending_search(
        &self,
        query: &[f32],
        k: usize,
        subset_indices: Option<&[u64]>,
        metric: DistanceMetric,
    ) -> (Vec<u64>, Vec<f32>) {
        if k == 0 {
            return (Vec::new(), Vec::new());
        }
        let snapshot = {
            let pending = self.pending_ingest.lock();
            if pending.is_empty() {
                return (Vec::new(), Vec::new());
            }
            pending.snapshot()
        };
        let dim = self.meta.dimension;
        if snapshot.vectors.is_empty() || snapshot.row_offsets.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let (data, row_offsets) = if let Some(subset) = subset_indices {
            let subset: HashSet<u64> = subset.iter().copied().collect();
            let mut data = Vec::new();
            let mut row_offsets = Vec::new();
            for (idx, &row) in snapshot.row_offsets.iter().enumerate() {
                if subset.contains(&row) {
                    let start = idx * dim;
                    let end = start + dim;
                    if end <= snapshot.vectors.len() {
                        data.extend_from_slice(&snapshot.vectors[start..end]);
                        row_offsets.push(row);
                    }
                }
            }
            (data, row_offsets)
        } else {
            (snapshot.vectors, snapshot.row_offsets)
        };

        if row_offsets.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let (indices, dists) = crate::distance::top_k_search(query, &data, dim, k, metric);
        let rows = indices
            .iter()
            .filter_map(|&idx| row_offsets.get(idx as usize).copied())
            .collect();
        (rows, dists)
    }

    fn merge_row_results(
        &self,
        left_ids: Vec<u64>,
        left_dists: Vec<f32>,
        right_ids: Vec<u64>,
        right_dists: Vec<f32>,
        limit: usize,
        metric: DistanceMetric,
    ) -> (Vec<u64>, Vec<f32>) {
        if right_ids.is_empty() {
            return (left_ids, left_dists);
        }
        if left_ids.is_empty() {
            return (right_ids, right_dists);
        }

        let ascending = metric.is_ascending();
        let mut best: HashMap<u64, f32> = HashMap::new();
        for (id, dist) in left_ids.into_iter().zip(left_dists) {
            best.insert(id, dist);
        }
        for (id, dist) in right_ids.into_iter().zip(right_dists) {
            match best.get_mut(&id) {
                Some(existing) => {
                    let better = if ascending {
                        dist < *existing
                    } else {
                        dist > *existing
                    };
                    if better {
                        *existing = dist;
                    }
                }
                None => {
                    best.insert(id, dist);
                }
            }
        }

        let mut pairs: Vec<(u64, f32)> = best.into_iter().collect();
        if ascending {
            pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        if pairs.len() > limit {
            pairs.truncate(limit);
        }
        pairs.into_iter().unzip()
    }

    /// Load the last sync fingerprint from disk.
    fn load_sync_fingerprint(collection_path: &Path) -> Option<String> {
        let fp_path = collection_path.join("sync_fingerprint");
        std::fs::read_to_string(&fp_path)
            .ok()
            .map(|s| s.trim().to_string())
    }

    /// Save the sync fingerprint to disk.
    fn save_sync_fingerprint(&self) -> Result<()> {
        if let Some(ref fp) = self.last_sync_fingerprint {
            let fp_path = self.path.join("sync_fingerprint");
            std::fs::write(&fp_path, fp)?;
        }
        Ok(())
    }

    /// Try to load an existing index from disk.
    fn try_load_index(
        meta_path: &Path,
        index_path: &Path,
    ) -> Result<(Option<Box<dyn VectorIndex>>, Option<String>)> {
        let meta_file = meta_path.join("index_metadata.json");
        if !meta_file.exists() {
            return Ok((None, None));
        }

        let content = std::fs::read_to_string(&meta_file)?;
        let meta: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| LynseError::Serialization(e.to_string()))?;

        let index_type = meta["index_type"]
            .as_str()
            .ok_or_else(|| LynseError::Storage("Missing index_type in metadata".into()))?;

        // Flat-family types (PQ/RaBitQ/PolarVec/SQ8) never write index.bin — skip loading
        let index_type_upper = index_type.to_uppercase();
        if index_type_upper.starts_with("FLAT") {
            return Ok((None, Some(index_type.to_string())));
        }

        let index_file = meta["index_file"].as_str().unwrap_or("index.bin");
        let index_data_file = index_path.join(index_file);
        if !index_data_file.exists() {
            return Ok((None, Some(index_type.to_string())));
        }

        let data = std::fs::read(&index_data_file)?;
        let mut idx = index::create_index(index_type)?;
        idx.deserialize(&data)?;

        Ok((Some(idx), Some(index_type.to_string())))
    }

    fn save_index_metadata(meta_path: &Path, meta: &serde_json::Value) -> Result<()> {
        std::fs::create_dir_all(meta_path)?;
        let metadata = serde_json::to_vec_pretty(meta)
            .map_err(|e| LynseError::Serialization(e.to_string()))?;
        Self::atomic_write_file(&meta_path.join("index_metadata.json"), &metadata)
    }

    fn write_generation_index(index_path: &Path, index_data: &[u8]) -> Result<String> {
        std::fs::create_dir_all(index_path)?;
        let generation = uuid::Uuid::new_v4().to_string().replace("-", "");
        let file_name = format!("index-{}.bin", generation);
        Self::atomic_write_file(&index_path.join(&file_name), index_data)?;
        Ok(file_name)
    }

    fn validate_unique_ids(ids: &[u64]) -> Result<()> {
        let mut seen = HashSet::with_capacity(ids.len());
        for &id in ids {
            if !seen.insert(id) {
                return Err(LynseError::InvalidArgument(format!(
                    "duplicate id {} within the same insert batch",
                    id
                )));
            }
        }
        Ok(())
    }

    fn validate_new_ids(&self, ids: &[u64]) -> Result<()> {
        Self::validate_unique_ids(ids)?;
        for &id in ids {
            if self.reverse_id_map.contains_key(&id) {
                return Err(LynseError::InvalidArgument(format!(
                    "id {} already exists; use update/upsert semantics instead",
                    id
                )));
            }
        }
        Ok(())
    }

    fn validate_new_external_ids(&self, external_ids: &[ExternalId]) -> Result<()> {
        let mut seen = HashSet::with_capacity(external_ids.len());
        for external_id in external_ids {
            external_id.validate()?;
            if !seen.insert(external_id.clone()) {
                return Err(LynseError::InvalidArgument(format!(
                    "duplicate external id {} within the same insert batch",
                    external_id
                )));
            }
            if self.external_to_internal_id.contains_key(external_id) {
                return Err(LynseError::InvalidArgument(format!(
                    "external id {} already exists",
                    external_id
                )));
            }
        }
        Ok(())
    }

    fn validate_row_boundary(&self) -> Result<usize> {
        let (n_rows, _) = self.vector_store.get_shape()?;
        let n_rows = n_rows as usize;
        if n_rows != self.id_map.len() {
            return Err(LynseError::Storage(format!(
                "vector row count ({}) does not match id_map length ({})",
                n_rows,
                self.id_map.len()
            )));
        }
        Ok(n_rows)
    }

    fn pending_len(&self) -> usize {
        self.pending_ingest.lock().len()
    }

    fn validate_logical_row_boundary(&self) -> Result<usize> {
        let (n_rows, _) = self.vector_store.get_shape()?;
        let n_rows = n_rows as usize;
        let pending_rows = self.pending_len();
        let logical_rows = n_rows
            .checked_add(pending_rows)
            .ok_or_else(|| LynseError::Storage("logical row count overflows".to_string()))?;
        if logical_rows != self.id_map.len() {
            return Err(LynseError::Storage(format!(
                "logical row count (persisted {} + pending {}) does not match id_map length ({})",
                n_rows,
                pending_rows,
                self.id_map.len()
            )));
        }
        Ok(logical_rows)
    }

    fn flush_pending_ingest(&self) -> Result<()> {
        let snapshot = {
            let pending = self.pending_ingest.lock();
            if pending.is_empty() {
                return Ok(());
            }
            (
                pending.encoded_vectors.clone(),
                pending.vector_dtype,
                pending.dim,
                pending.ids.clone(),
            )
        };
        let (encoded_vectors, vector_dtype, dim, ids) = snapshot;
        self.write_ingested_vectors(&encoded_vectors, vector_dtype, dim, &ids)?;
        let _ = self.pending_ingest.lock().take();
        Ok(())
    }

    fn write_ingested_vectors(
        &self,
        encoded_vectors: &[u8],
        vector_dtype: VectorDtype,
        dim: usize,
        ids: &[u64],
    ) -> Result<()> {
        if dim != self.meta.dimension {
            return Err(LynseError::DimensionMismatch {
                expected: self.meta.dimension,
                got: dim,
            });
        }
        self.vector_store
            .write_encoded_le_bytes(encoded_vectors, ids.len(), vector_dtype)?;
        Self::append_id_map_path(&self.vector_store.id_map_path(), ids)
    }

    fn append_recovered_items(
        &mut self,
        vectors: &[f32],
        ids: &[u64],
        fields: &[HashMap<String, serde_json::Value>],
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }

        let dim = self.meta.dimension;
        let n_vectors = ids.len();
        if vectors.len() != n_vectors * dim {
            return Err(LynseError::DimensionMismatch {
                expected: n_vectors * dim,
                got: vectors.len(),
            });
        }
        if fields.len() != n_vectors {
            return Err(LynseError::InvalidArgument(format!(
                "fields length ({}) must match recovered vector count ({})",
                fields.len(),
                n_vectors
            )));
        }

        self.validate_new_ids(ids)?;
        let start_row = self.validate_row_boundary()?;
        let row_ids: Vec<u64> = (start_row as u64..start_row as u64 + n_vectors as u64).collect();

        self.vector_store.write(vectors)?;
        self.field_store.batch_store_at_ids(ids, fields)?;
        self.text_index.insert_documents(ids, fields, false)?;

        if let Some(ref mut idx) = self.index {
            idx.insert(vectors, n_vectors, dim, &row_ids)?;
        }

        for (i, &user_id) in ids.iter().enumerate() {
            self.id_map.push(user_id);
            self.reverse_id_map.insert(user_id, start_row + i);
            if !self.internal_to_external_id.contains_key(&user_id) {
                let external_id = ExternalId::Int(user_id);
                self.internal_to_external_id
                    .insert(user_id, external_id.clone());
                self.external_to_internal_id.insert(external_id, user_id);
                self.next_internal_id = self.next_internal_id.max(user_id.saturating_add(1));
            }
        }
        Self::append_id_map_path(&self.vector_store.id_map_path(), ids)?;
        self.write_external_id_map()?;

        Ok(())
    }

    fn append_recovered_encoded_items(
        &mut self,
        decoded_vectors: &[f32],
        encoded_vectors: &[u8],
        vector_dtype: VectorDtype,
        ids: &[u64],
        fields: &[HashMap<String, serde_json::Value>],
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        if vector_dtype != self.vector_dtype {
            return Err(LynseError::InvalidArgument(format!(
                "recovered vector dtype {} does not match collection dtype {}",
                vector_dtype.storage_name(),
                self.vector_dtype.storage_name()
            )));
        }

        let dim = self.meta.dimension;
        let n_vectors = ids.len();
        if decoded_vectors.len() != n_vectors * dim {
            return Err(LynseError::DimensionMismatch {
                expected: n_vectors * dim,
                got: decoded_vectors.len(),
            });
        }
        if fields.len() != n_vectors {
            return Err(LynseError::InvalidArgument(format!(
                "fields length ({}) must match recovered vector count ({})",
                fields.len(),
                n_vectors
            )));
        }

        self.validate_new_ids(ids)?;
        let start_row = self.validate_row_boundary()?;
        let row_ids: Vec<u64> = (start_row as u64..start_row as u64 + n_vectors as u64).collect();

        self.vector_store
            .write_encoded_le_bytes(encoded_vectors, n_vectors, vector_dtype)?;
        self.field_store.batch_store_at_ids(ids, fields)?;
        self.text_index.insert_documents(ids, fields, false)?;

        if let Some(ref mut idx) = self.index {
            idx.insert(decoded_vectors, n_vectors, dim, &row_ids)?;
        }

        for (i, &user_id) in ids.iter().enumerate() {
            self.id_map.push(user_id);
            self.reverse_id_map.insert(user_id, start_row + i);
            if !self.internal_to_external_id.contains_key(&user_id) {
                let external_id = ExternalId::Int(user_id);
                self.internal_to_external_id
                    .insert(user_id, external_id.clone());
                self.external_to_internal_id.insert(external_id, user_id);
                self.next_internal_id = self.next_internal_id.max(user_id.saturating_add(1));
            }
        }
        Self::append_id_map_path(&self.vector_store.id_map_path(), ids)?;
        self.write_external_id_map()?;

        Ok(())
    }

    fn buffer_add_items_encoded(
        &mut self,
        encoded_vectors: &[u8],
        vector_dtype: VectorDtype,
        decoded_vectors_for_index: Option<&[f32]>,
        n_vectors: usize,
        ids: &[u64],
        fields: Option<&[HashMap<String, serde_json::Value>]>,
    ) -> Result<()> {
        self.ensure_writable()?;
        if vector_dtype != self.vector_dtype {
            return Err(LynseError::InvalidArgument(format!(
                "encoded vector dtype {} does not match collection dtype {}",
                vector_dtype.storage_name(),
                self.vector_dtype.storage_name()
            )));
        }
        let dim = self.infer_dimension_from_encoded(encoded_vectors, vector_dtype, n_vectors)?;
        let expected_bytes = n_vectors
            .checked_mul(dim)
            .and_then(|values| values.checked_mul(vector_dtype.byte_width()))
            .ok_or_else(|| {
                LynseError::InvalidArgument("encoded vector byte size overflows".to_string())
            })?;
        if encoded_vectors.len() != expected_bytes {
            return Err(LynseError::DimensionMismatch {
                expected: expected_bytes,
                got: encoded_vectors.len(),
            });
        }
        if ids.len() != n_vectors {
            return Err(LynseError::InvalidArgument(format!(
                "ids length ({}) must match n_vectors ({})",
                ids.len(),
                n_vectors
            )));
        }
        if let Some(field_list) = fields {
            if field_list.len() != n_vectors {
                return Err(LynseError::InvalidArgument(format!(
                    "fields length ({}) must match n_vectors ({})",
                    field_list.len(),
                    n_vectors
                )));
            }
        }
        if n_vectors == 0 {
            return Ok(());
        }

        self.validate_new_ids(ids)?;
        let start_row = self.validate_logical_row_boundary()?;
        let row_ids: Vec<u64> = (start_row as u64..start_row as u64 + n_vectors as u64).collect();

        let wal_fields: Vec<HashMap<String, serde_json::Value>> = match fields {
            Some(fl) => fl.to_vec(),
            None => Vec::new(),
        };
        let direct_write = n_vectors >= PENDING_INGEST_FLUSH_ROWS
            || encoded_vectors.len() >= PENDING_INGEST_FLUSH_BYTES;
        let needs_decoded_vectors = self.index.is_some() || !direct_write;
        let decoded;
        let vectors = if needs_decoded_vectors {
            let vectors = match decoded_vectors_for_index {
                Some(vectors) => vectors,
                None => {
                    decoded = decode_vector_bytes_to_f32(encoded_vectors, vector_dtype);
                    decoded.as_slice()
                }
            };
            if vectors.len() != n_vectors * dim {
                return Err(LynseError::DimensionMismatch {
                    expected: n_vectors * dim,
                    got: vectors.len(),
                });
            }
            Some(vectors)
        } else {
            None
        };

        if direct_write {
            self.flush_pending_ingest()?;
            let (wal_result, vector_result) = rayon::join(
                || {
                    self.wal.write_log_encoded_data(
                        encoded_vectors,
                        dim,
                        vector_dtype,
                        ids,
                        &wal_fields,
                    )
                },
                || {
                    self.vector_store.write_encoded_le_bytes(
                        encoded_vectors,
                        n_vectors,
                        vector_dtype,
                    )
                },
            );
            if let Err(error) = wal_result {
                if vector_result.is_ok() {
                    let _ = self.vector_store.truncate_to_vectors(start_row);
                }
                return Err(error);
            }
            vector_result?;
        } else {
            self.wal.write_log_encoded_data(
                encoded_vectors,
                dim,
                vector_dtype,
                ids,
                &wal_fields,
            )?;
        }

        if let Some(field_list) = fields {
            self.field_store.batch_store_at_ids(ids, field_list)?;
            self.text_index.insert_documents(ids, field_list, false)?;
        }

        if let Some(ref mut idx) = self.index {
            let vectors = vectors.ok_or_else(|| {
                LynseError::Storage("decoded vectors are required for index updates".to_string())
            })?;
            idx.insert(vectors, n_vectors, dim, &row_ids)?;
        }

        for (i, &user_id) in ids.iter().enumerate() {
            self.id_map.push(user_id);
            self.reverse_id_map.insert(user_id, start_row + i);
            if !self.internal_to_external_id.contains_key(&user_id) {
                let external_id = ExternalId::Int(user_id);
                self.internal_to_external_id
                    .insert(user_id, external_id.clone());
                self.external_to_internal_id.insert(external_id, user_id);
                self.next_internal_id = self.next_internal_id.max(user_id.saturating_add(1));
            }
        }

        if direct_write {
            Self::append_id_map_path(&self.vector_store.id_map_path(), ids)?;
        } else {
            let should_flush = {
                let mut pending = self.pending_ingest.lock();
                let vectors = vectors.ok_or_else(|| {
                    LynseError::Storage(
                        "decoded vectors are required for pending ingest buffering".to_string(),
                    )
                })?;
                pending.append(encoded_vectors, vectors, vector_dtype, dim, ids, &row_ids)?;
                pending.should_flush()
            };
            if should_flush {
                self.flush_pending_ingest()?;
            }
        }

        Ok(())
    }

    pub(crate) fn add_items_encoded_vectors(
        &mut self,
        encoded_vectors: &[u8],
        vector_dtype: VectorDtype,
        n_vectors: usize,
        ids: &[u64],
        fields: Option<&[HashMap<String, serde_json::Value>]>,
    ) -> Result<()> {
        self.buffer_add_items_encoded(encoded_vectors, vector_dtype, None, n_vectors, ids, fields)
    }

    /// Add vectors with user-specified IDs and optional field metadata.
    /// Data is first written to WAL for crash safety, then to main storage.
    pub fn add_items(
        &mut self,
        vectors: &[f32],
        n_vectors: usize,
        ids: &[u64],
        fields: Option<&[HashMap<String, serde_json::Value>]>,
    ) -> Result<()> {
        self.ensure_writable()?;
        let dim = self.infer_dimension_from_f32(vectors, n_vectors)?;
        if vectors.len() != n_vectors * dim {
            return Err(LynseError::DimensionMismatch {
                expected: dim,
                got: if n_vectors == 0 {
                    vectors.len()
                } else {
                    vectors.len() / n_vectors
                },
            });
        }
        let encoded_vectors = encode_f32_slice_as_le_bytes(vectors, self.vector_dtype);
        self.buffer_add_items_encoded(
            &encoded_vectors,
            self.vector_dtype,
            Some(vectors),
            n_vectors,
            ids,
            fields,
        )
    }

    /// Add records with public string/integer external IDs.
    ///
    /// Internal numeric IDs are assigned by the collection allocator and are the
    /// only IDs used by vector storage, indexes, tombstones, and WAL replay.
    pub fn add_records(
        &mut self,
        vectors: &[f32],
        n_vectors: usize,
        external_ids: &[ExternalId],
        fields: Option<&[HashMap<String, serde_json::Value>]>,
    ) -> Result<Vec<u64>> {
        self.ensure_writable()?;
        let dim = self.infer_dimension_from_f32(vectors, n_vectors)?;
        if vectors.len() != n_vectors * dim {
            return Err(LynseError::DimensionMismatch {
                expected: dim,
                got: if n_vectors == 0 {
                    vectors.len()
                } else {
                    vectors.len() / n_vectors
                },
            });
        }
        if external_ids.len() != n_vectors {
            return Err(LynseError::InvalidArgument(format!(
                "ids length ({}) must match n_vectors ({})",
                external_ids.len(),
                n_vectors
            )));
        }
        if let Some(field_list) = fields {
            if field_list.len() != n_vectors {
                return Err(LynseError::InvalidArgument(format!(
                    "fields length ({}) must match n_vectors ({})",
                    field_list.len(),
                    n_vectors
                )));
            }
        }

        self.validate_new_external_ids(external_ids)?;
        let prev_next_internal_id = self.next_internal_id;
        let internal_ids = self.allocate_internal_ids(n_vectors)?;
        let prev_external_to_internal = self.external_to_internal_id.clone();
        let prev_internal_to_external = self.internal_to_external_id.clone();

        let delta_entries: Vec<ExternalIdEntry> = external_ids
            .iter()
            .cloned()
            .zip(internal_ids.iter().copied())
            .map(|(external_id, internal_id)| ExternalIdEntry {
                internal_id,
                external_id,
            })
            .collect();

        for entry in &delta_entries {
            self.external_to_internal_id
                .insert(entry.external_id.clone(), entry.internal_id);
            self.internal_to_external_id
                .insert(entry.internal_id, entry.external_id.clone());
        }

        self.append_external_id_map_delta(delta_entries)?;

        let encoded_vectors = encode_f32_slice_as_le_bytes(vectors, self.vector_dtype);
        let result = self.buffer_add_items_encoded(
            &encoded_vectors,
            self.vector_dtype,
            Some(vectors),
            n_vectors,
            &internal_ids,
            fields,
        );

        if let Err(err) = result {
            self.next_internal_id = prev_next_internal_id;
            self.external_to_internal_id = prev_external_to_internal;
            self.internal_to_external_id = prev_internal_to_external;
            let _ = self.write_external_id_map();
            return Err(err);
        }

        Ok(internal_ids)
    }

    /// Create a named vector field stored independently from the primary vector.
    pub fn create_vector_field(
        &mut self,
        name: &str,
        dimension: usize,
        metric: Option<&str>,
        index_mode: Option<&str>,
    ) -> Result<VectorFieldConfig> {
        self.create_vector_field_with_dtype(name, dimension, metric, index_mode, None)
    }

    pub fn create_vector_field_with_dtype(
        &mut self,
        name: &str,
        dimension: usize,
        metric: Option<&str>,
        index_mode: Option<&str>,
        dtypes: Option<&str>,
    ) -> Result<VectorFieldConfig> {
        self.ensure_writable()?;
        if dimension == 0 {
            return Err(LynseError::InvalidArgument(
                "vector field dimension must be greater than zero".to_string(),
            ));
        }

        let name = Self::validate_vector_field_name(name)?;
        let metric = Self::resolve_named_vector_metric(metric, index_mode)?;
        let index_mode = Self::normalize_field_index_mode(index_mode, metric)?;
        let vector_dtype = match dtypes {
            Some(dtype) => VectorDtype::parse(dtype)?,
            None => self.vector_dtype,
        };

        if let Some(existing) = self.named_vector_fields.get(&name) {
            if existing.config.dimension == dimension
                && existing.config.metric == metric.name()
                && existing.config.index_mode == index_mode
                && VectorDtype::parse(&existing.config.dtypes)? == vector_dtype
            {
                return Ok(existing.config.clone());
            }
            return Err(LynseError::InvalidArgument(format!(
                "vector field '{}' already exists with a different configuration",
                name
            )));
        }

        let root = Self::vector_fields_root(&self.path);
        std::fs::create_dir_all(&root)?;
        let field_path = root.join(&name);
        let vector_store =
            VectorStore::new(&field_path, dimension, self.meta.chunk_size, vector_dtype)?;
        let config = VectorFieldConfig {
            name: name.clone(),
            dimension,
            metric: metric.name().to_string(),
            index_mode,
            dtypes: vector_dtype.storage_name().to_string(),
        };

        self.named_vector_fields.insert(
            name,
            NamedVectorField {
                config: config.clone(),
                vector_store,
                index: None,
                id_map: Vec::new(),
                reverse_id_map: HashMap::new(),
                path: field_path,
            },
        );
        self.save_named_vector_fields_manifest()?;
        Ok(config)
    }

    /// List vector fields. The primary vector is reported as the reserved
    /// `default` field for API consistency.
    pub fn list_vector_fields(&self) -> Vec<VectorFieldConfig> {
        let mut fields = vec![VectorFieldConfig {
            name: DEFAULT_VECTOR_FIELD_NAME.to_string(),
            dimension: self.meta.dimension,
            metric: self.resolve_metric().name().to_string(),
            index_mode: self
                .index_mode
                .clone()
                .unwrap_or_else(|| "FLAT-IP".to_string()),
            dtypes: self.vector_dtype.storage_name().to_string(),
        }];
        let mut named: Vec<VectorFieldConfig> = self
            .named_vector_fields
            .values()
            .map(|field| field.config.clone())
            .collect();
        named.sort_by(|a, b| a.name.cmp(&b.name));
        fields.extend(named);
        fields
    }

    /// Return the stored vector count and dimension for a vector field.
    pub fn vector_field_shape(&self, field_name: &str) -> Result<(u64, usize)> {
        if field_name == DEFAULT_VECTOR_FIELD_NAME || field_name.trim().is_empty() {
            return self.shape();
        }

        let field_name = Self::validate_vector_field_name(field_name)?;
        let field = self.named_vector_fields.get(&field_name).ok_or_else(|| {
            LynseError::InvalidArgument(format!("vector field '{}' does not exist", field_name))
        })?;
        field.vector_store.get_shape()
    }

    /// Estimate dense vector bytes stored by the primary and named vector fields.
    pub fn estimated_vector_bytes(&self) -> Result<u64> {
        fn field_bytes(n_vectors: u64, dim: usize, dtype: VectorDtype) -> Result<u64> {
            n_vectors
                .checked_mul(dim as u64)
                .and_then(|values| values.checked_mul(dtype.byte_width() as u64))
                .ok_or_else(|| LynseError::Storage("vector byte estimate overflows".to_string()))
        }

        let (primary_n, primary_dim) = self.shape()?;
        let mut total = field_bytes(primary_n, primary_dim, self.vector_store.dtype())?;
        for field in self.named_vector_fields.values() {
            let (n, dim) = field.vector_store.get_shape()?;
            let bytes = field_bytes(n, dim, field.vector_store.dtype())?;
            total = total
                .checked_add(bytes)
                .ok_or_else(|| LynseError::Storage("vector byte estimate overflows".to_string()))?;
        }
        Ok(total)
    }

    /// Attach vectors to an existing named vector field for existing user IDs.
    pub fn add_named_vectors(
        &mut self,
        field_name: &str,
        vectors: &[f32],
        n_vectors: usize,
        ids: &[u64],
    ) -> Result<()> {
        self.ensure_writable()?;
        let field_name = Self::validate_vector_field_name(field_name)?;
        let field = self.named_vector_fields.get(&field_name).ok_or_else(|| {
            LynseError::InvalidArgument(format!("vector field '{}' does not exist", field_name))
        })?;
        let dim = field.config.dimension;

        if vectors.len() != n_vectors * dim {
            return Err(LynseError::DimensionMismatch {
                expected: dim,
                got: if n_vectors == 0 {
                    0
                } else {
                    vectors.len() / n_vectors
                },
            });
        }
        if ids.len() != n_vectors {
            return Err(LynseError::InvalidArgument(format!(
                "ids length ({}) must match n_vectors ({})",
                ids.len(),
                n_vectors
            )));
        }

        let mut seen = HashSet::with_capacity(ids.len());
        for &id in ids {
            if !seen.insert(id) {
                return Err(LynseError::InvalidArgument(format!(
                    "duplicate id {} within named vector batch",
                    id
                )));
            }
            if !self.is_id_exists(id) {
                return Err(LynseError::InvalidArgument(format!(
                    "cannot add named vector for unknown id {}",
                    id
                )));
            }
            if field.reverse_id_map.contains_key(&id) {
                return Err(LynseError::InvalidArgument(format!(
                    "id {} already has a vector for field '{}'",
                    id, field_name
                )));
            }
        }

        let field = self.named_vector_fields.get_mut(&field_name).unwrap();
        let start_row = field.id_map.len();
        let row_ids: Vec<u64> = (start_row as u64..start_row as u64 + n_vectors as u64).collect();
        field.vector_store.write(vectors)?;
        if let Some(ref mut idx) = field.index {
            idx.insert(vectors, n_vectors, dim, &row_ids)?;
        }
        for (i, &id) in ids.iter().enumerate() {
            field.id_map.push(id);
            field.reverse_id_map.insert(id, start_row + i);
        }
        Self::append_id_map_path(&field.vector_store.id_map_path(), ids)?;
        Ok(())
    }

    /// Attach sparse feature vectors to existing user IDs. Sparse vectors are
    /// searched with inner product and stored independently from dense fields.
    pub fn add_sparse_vectors(&mut self, ids: &[u64], vectors: &[Vec<(u32, f32)>]) -> Result<()> {
        self.ensure_writable()?;
        if ids.len() != vectors.len() {
            return Err(LynseError::InvalidArgument(format!(
                "ids length ({}) must match sparse vector count ({})",
                ids.len(),
                vectors.len()
            )));
        }

        let mut seen = HashSet::with_capacity(ids.len());
        for &id in ids {
            if !seen.insert(id) {
                return Err(LynseError::InvalidArgument(format!(
                    "duplicate id {} within sparse vector batch",
                    id
                )));
            }
            if !self.is_id_exists(id) {
                return Err(LynseError::InvalidArgument(format!(
                    "cannot add sparse vector for unknown id {}",
                    id
                )));
            }
        }

        self.sparse_vectors.upsert_many(ids, vectors)
    }

    /// Build or change the index for a named vector field.
    pub fn build_vector_field_index(&mut self, field_name: &str, index_type: &str) -> Result<()> {
        self.build_vector_field_index_with_options(field_name, index_type, None)
    }

    pub fn build_vector_field_index_with_options(
        &mut self,
        field_name: &str,
        index_type: &str,
        n_clusters: Option<usize>,
    ) -> Result<()> {
        self.ensure_writable()?;
        if field_name == DEFAULT_VECTOR_FIELD_NAME || field_name.trim().is_empty() {
            return self.build_index_with_options(index_type, n_clusters);
        }

        let field_name = Self::validate_vector_field_name(field_name)?;
        let metric = Self::metric_from_mode_str(index_type);
        let index_type = Self::normalize_field_index_mode(Some(index_type), metric)?;
        let is_flat = Self::resolve_metric_from_type(&index_type).is_some();
        let index_upper = index_type.to_uppercase();
        if n_clusters.is_some()
            && !(index_upper.starts_with("IVF") || index_upper.starts_with("SPANN"))
        {
            return Err(LynseError::InvalidArgument(
                "n_clusters is only supported for IVF and SPANN indexes".to_string(),
            ));
        }

        {
            let field = self
                .named_vector_fields
                .get_mut(&field_name)
                .ok_or_else(|| {
                    LynseError::InvalidArgument(format!(
                        "vector field '{}' does not exist",
                        field_name
                    ))
                })?;
            let dim = field.config.dimension;
            let n_vectors = field.vector_store.get_shape()?.0 as usize;
            field.index = None;

            if is_flat {
                let index_path = field.path.join("index");
                if index_path.exists() {
                    std::fs::remove_dir_all(&index_path)?;
                }
                let meta_path = field.path.join("index_meta");
                let meta = serde_json::json!({
                    "index_type": index_type,
                    "n_vectors": n_vectors,
                    "dimension": dim,
                });
                Self::save_index_metadata(&meta_path, &meta)?;
            } else {
                let n = field.vector_store.get_shape()?.0 as usize;
                if n == 0 {
                    return Err(LynseError::EmptyDatabase);
                }

                let all_data = field.vector_store.read_all_f32()?;
                let row_ids: Vec<u64> = (0..n as u64).collect();
                let mut idx = index::create_index_with_options(&index_type, n_clusters)?;
                idx.build(&all_data, n, dim, Some(&row_ids))?;

                let index_data = idx.serialize()?;
                let index_path = field.path.join("index");
                let index_file = Self::write_generation_index(&index_path, &index_data)?;
                let meta_path = field.path.join("index_meta");
                let meta = serde_json::json!({
                    "index_type": index_type,
                    "n_vectors": n,
                    "dimension": dim,
                    "index_file": index_file,
                    "n_clusters": n_clusters,
                });
                Self::save_index_metadata(&meta_path, &meta)?;
                field.index = Some(idx);
            }

            field.config.index_mode = index_type.clone();
            field.config.metric = metric.name().to_string();
        }

        self.save_named_vector_fields_manifest()?;
        Ok(())
    }

    /// Remove a named vector field index and return it to flat search.
    pub fn remove_vector_field_index(&mut self, field_name: &str) -> Result<()> {
        self.ensure_writable()?;
        if field_name == DEFAULT_VECTOR_FIELD_NAME || field_name.trim().is_empty() {
            return self.remove_index();
        }

        let field_name = Self::validate_vector_field_name(field_name)?;
        {
            let field = self
                .named_vector_fields
                .get_mut(&field_name)
                .ok_or_else(|| {
                    LynseError::InvalidArgument(format!(
                        "vector field '{}' does not exist",
                        field_name
                    ))
                })?;
            field.index = None;
            let index_path = field.path.join("index");
            let meta_path = field.path.join("index_meta");
            if index_path.exists() {
                std::fs::remove_dir_all(&index_path)?;
            }
            if meta_path.exists() {
                std::fs::remove_dir_all(&meta_path)?;
            }
            field.config.index_mode = match DistanceMetric::from_str(&field.config.metric)
                .unwrap_or(DistanceMetric::InnerProduct)
            {
                DistanceMetric::L2Squared => "FLAT-L2".to_string(),
                DistanceMetric::Cosine => "FLAT-COS".to_string(),
                DistanceMetric::Hamming => "FLAT-HAMMING".to_string(),
                DistanceMetric::Jaccard => "FLAT-JACCARD".to_string(),
                DistanceMetric::InnerProduct => "FLAT-IP".to_string(),
            };
        }

        self.save_named_vector_fields_manifest()?;
        Ok(())
    }

    /// Flush pending WAL bytes and fsync collection files without clearing WAL.
    pub fn flush(&self) -> Result<()> {
        self.ensure_writable()?;
        self.flush_pending_ingest()?;
        self.persist_vector_store_metadata()?;
        self.wal.flush()?;
        self.sync_collection_files()
    }

    /// Checkpoint durable state and clear WAL after data has reached main storage.
    pub fn checkpoint(&self) -> Result<()> {
        self.ensure_writable()?;
        self.text_index.write_to_disk_if_present()?;
        self.flush_pending_ingest()?;
        self.field_store.flush()?;
        self.persist_vector_store_metadata()?;
        self.write_external_id_map()?;
        self.sync_checkpoint_files()?;
        self.wal.cleanup()?;
        Self::sync_dir_if_exists(self.wal.log_dir());
        Self::sync_dir_if_exists(&self.path);
        Ok(())
    }

    /// Lightweight checkpoint for local benchmarks: persist derived metadata and
    /// clear WAL without forcing a recursive fsync.
    pub fn checkpoint_fast(&self) -> Result<()> {
        self.ensure_writable()?;
        self.text_index.write_to_disk_if_present()?;
        self.flush_pending_ingest()?;
        self.persist_vector_store_metadata()?;
        self.write_external_id_map()?;
        self.wal.cleanup()
    }

    /// Close the collection handle from an API perspective.
    ///
    /// The writer lock is released when the `Collection` object itself is
    /// dropped; this method makes outstanding state durable and stops the WAL.
    pub fn close(&mut self) -> Result<()> {
        if !self.read_only {
            self.field_store.flush()?;
            self.checkpoint()?;
            self.wal.stop()?;
        }

        for field in self.named_vector_fields.values_mut() {
            field.index = None;
        }
        self.index = None;
        self.pq_index = None;
        self.rabitq_index = None;
        self.polarvec_index = None;
        self.lock.take();
        Ok(())
    }

    /// Commit: move pending data into main storage and clear WAL without a
    /// recursive fsync. Use `checkpoint()` when an explicit durable fsync
    /// barrier is required.
    pub fn commit(&self) -> Result<()> {
        self.checkpoint_fast()
    }

    /// Check if there is uncommitted WAL data.
    pub fn has_uncommitted_data(&self) -> bool {
        self.wal.has_uncommitted_data()
    }

    /// Build or rebuild the index.
    ///
    /// For Flat types (Flat-IP/L2/Cos): just sets the metric. Data is already
    /// in VectorStore's mmap — search uses it directly with zero-copy.
    ///
    /// For HNSW/IVF/SPANN types: loads data from mmap and builds the index structure.
    pub fn build_index(&mut self, index_type: &str) -> Result<()> {
        self.build_index_with_options(index_type, None)
    }

    pub fn build_index_with_options(
        &mut self,
        index_type: &str,
        n_clusters: Option<usize>,
    ) -> Result<()> {
        self.ensure_writable()?;
        self.flush_pending_ingest()?;
        let dim = self.meta.dimension;
        let upper = index_type.to_uppercase();

        // Clear any previously built quantizer indices
        self.pq_index = None;
        self.rabitq_index = None;
        self.polarvec_index = None;

        let is_flat = Self::resolve_metric_from_type(index_type).is_some();
        if n_clusters.is_some() && !(upper.starts_with("IVF") || upper.starts_with("SPANN")) {
            return Err(LynseError::InvalidArgument(
                "n_clusters is only supported for IVF and SPANN indexes".to_string(),
            ));
        }

        let n_vectors = if is_flat {
            // Flat family: clear any graph/partition index, and remove stale graph index.bin
            self.index = None;
            let stale_index = self.path.join("index").join("index.bin");
            if stale_index.exists() {
                let _ = std::fs::remove_file(&stale_index);
            }
            let n = self
                .vector_store
                .get_shape()
                .map(|(n, _)| n as usize)
                .unwrap_or(0);
            // Build PQ index if requested
            if upper.contains("PQ") {
                if n > 0 {
                    let n_subspaces = parse_n_subspaces(&upper, dim);
                    let all_data = self.vector_store.read_all_f32()?;
                    let pq = PQIndex::build(&all_data, n, dim, n_subspaces);
                    let pq_path = self.path.join("pq_index.bin");
                    pq.save(&pq_path)
                        .map_err(|e| LynseError::Storage(e.to_string()))?;
                    self.pq_index = Some(pq);
                }
            } else if upper.contains("RABITQ") {
                if n > 0 {
                    let all_data = self.vector_store.read_all_f32()?;
                    let rq = RaBitQIndex::build(&all_data, n, dim);
                    let rq_path = self.path.join("rabitq_index.bin");
                    rq.save(&rq_path)
                        .map_err(|e| LynseError::Storage(e.to_string()))?;
                    self.rabitq_index = Some(rq);
                }
            } else if upper.contains("POLARVEC") {
                if n > 0 {
                    let bits = parse_bits(&upper, POLARVEC_DEFAULT_BITS);
                    let metric = self.resolve_metric();
                    let all_data = self.vector_store.read_all_f32()?;
                    let pv = PolarVecIndex::build_for_metric(&all_data, n, dim, bits, metric);
                    let pv_path = self.path.join("polarvec_index.bin");
                    pv.save(&pv_path)
                        .map_err(|e| LynseError::Storage(e.to_string()))?;
                    self.polarvec_index = Some(pv);
                }
            }
            n
        } else {
            // Graph/partition ANN indexes need data to build.
            let n = self.vector_store.get_shape()?.0 as usize;
            if n == 0 {
                return Err(LynseError::EmptyDatabase);
            }

            let all_data = self.vector_store.read_all_f32()?;
            let ids: Vec<u64> = (0..n as u64).collect();

            let mut idx = index::create_index_with_options(index_type, n_clusters)?;
            idx.build(&all_data, n, dim, Some(&ids))?;

            // Save index to disk
            let index_data = idx.serialize()?;
            let index_path = self.path.join("index");
            let index_file = Self::write_generation_index(&index_path, &index_data)?;

            self.index = Some(idx);
            let meta_path = self.path.join("index_meta");
            let meta = serde_json::json!({
                "index_type": index_type,
                "n_vectors": n,
                "dimension": dim,
                "index_file": index_file,
                "n_clusters": n_clusters,
            });
            Self::save_index_metadata(&meta_path, &meta)?;
            n
        };

        // Save metadata
        if is_flat {
            let meta_path = self.path.join("index_meta");
            let meta = serde_json::json!({
                "index_type": index_type,
                "n_vectors": n_vectors,
                "dimension": dim,
            });
            Self::save_index_metadata(&meta_path, &meta)?;
        }

        self.index_mode = Some(index_type.to_string());
        self.meta.index_mode = self.index_mode.clone();

        Ok(())
    }

    /// Map an index type string to a distance metric (for Flat types).
    /// Returns Some(metric) for Flat-family types, None for ANN structures
    /// (HNSW/IVF/SPANN/DiskANN) which build a separate index.
    fn resolve_metric_from_type(index_type: &str) -> Option<DistanceMetric> {
        let upper = index_type.to_uppercase();
        // Flat family: no separate index structure needed. Bare "FLAT" is not
        // accepted; callers must provide an explicit metric suffix.
        if upper.starts_with("FLAT-") {
            Some(Self::metric_from_mode_str(&upper))
        } else {
            // Legacy compat
            match index_type {
                "Flat-IP" | "flat-ip" => Some(DistanceMetric::InnerProduct),
                "Flat-L2" | "flat-l2" => Some(DistanceMetric::L2Squared),
                "Flat-Cos" | "flat-cosine" => Some(DistanceMetric::Cosine),
                _ => None,
            }
        }
    }

    /// Extract the distance metric from an index_mode string.
    /// Supports: FLAT-IP, FLAT-L2, FLAT-COS, FLAT-IP-SQ8, IVF-L2-SQ8, etc.
    fn metric_from_mode_str(mode: &str) -> DistanceMetric {
        let upper = mode.to_uppercase();
        if upper.contains("JACCARD") {
            DistanceMetric::Jaccard
        } else if upper.contains("HAMMING") {
            DistanceMetric::Hamming
        } else if upper.contains("L2") {
            DistanceMetric::L2Squared
        } else if upper.contains("COS") {
            DistanceMetric::Cosine
        } else {
            // Default: Inner Product ("FLAT-IP", "FLAT-IP-SQ8", "IVF-IP", etc.)
            DistanceMetric::InnerProduct
        }
    }

    /// Check if the index_mode string includes SQ8 quantization.
    fn resolve_use_sq8(&self) -> bool {
        self.index_mode
            .as_ref()
            .map(|m| m.to_uppercase().contains("SQ8"))
            .unwrap_or(false)
    }

    /// Search for nearest neighbors.
    ///
    /// Search path:
    /// 1. If an HNSW/IVF/SPANN/DiskANN index exists → use it
    /// 2. Otherwise → zero-copy mmap brute-force via VectorStore's FlatMmap
    /// 3. Filtered search → brute-force with subset filtering on mmap data
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        where_expr: Option<&str>,
        nprobe: usize,
        approx: bool,
        eps: f32,
    ) -> Result<SearchResult> {
        let dim = self.meta.dimension;
        if query.len() != dim {
            return Err(LynseError::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        }

        let subset_indices = if let Some(filter) = where_expr {
            Some(Arc::new(
                self.field_keys_to_rows(self.field_store.query(filter)?),
            ))
        } else {
            None
        };

        self.search_with_precomputed_filter(query, k, subset_indices, nprobe, approx, eps)
    }

    fn search_with_precomputed_filter(
        &self,
        query: &[f32],
        k: usize,
        subset_indices: Option<Arc<Vec<u64>>>,
        nprobe: usize,
        approx: bool,
        eps: f32,
    ) -> Result<SearchResult> {
        let dim = self.meta.dimension;
        if query.len() != dim {
            return Err(LynseError::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        }

        let tombstone_count = self.tombstone.read().len();
        let search_params = SearchParams {
            k: if tombstone_count == 0 {
                k
            } else {
                k.saturating_add(tombstone_count)
            },
            nprobe,
            ef_search: None,
            subset_indices,
        };
        let search_k = search_params.k;

        let collection_metric = self.resolve_metric();
        let approx_cfg = if approx && collection_metric.supports_flat_approx() {
            Some(crate::storage::approx_search::ApproxSearchConfig::new(eps))
        } else {
            None
        };

        let filtered_index_requires_exact = search_params.subset_indices.is_some()
            && self
                .index
                .as_ref()
                .map(|idx| {
                    matches!(
                        idx.config().index_type,
                        IndexType::HNSW | IndexType::DiskANN
                    )
                })
                .unwrap_or(false);

        let (result_ids, result_dists) = if filtered_index_requires_exact {
            // HNSW and DiskANN do not currently apply subset_indices inside
            // their graph searches. Use the exact filtered path to avoid
            // returning rows outside the metadata predicate.
            self.brute_force_search_filtered(query, search_k, &search_params)?
        } else if let Some(ref idx) = self.index {
            // ANN structure index path
            idx.search(query, search_k, &search_params)?
        } else if search_params.subset_indices.is_some() {
            // Filtered flat-family search must evaluate only allowed row IDs.
            // Auxiliary flat quantizers (PQ/RaBitQ/PolarVec) are whole-corpus
            // candidate generators, so filtered queries use the exact filtered
            // mmap path for correctness.
            self.brute_force_search_filtered(query, search_k, &search_params)?
        } else if let Some(ref pq) = self.pq_index {
            // PQ (Product Quantization) two-pass search
            let metric = collection_metric;
            let all_data = self.vector_store.read_all_f32()?;
            let (indices, dists) = pq.search(query, search_k, &all_data, metric, PQ_OVERSAMPLE);
            (indices.into_iter().map(|i| i as u64).collect(), dists)
        } else if let Some(ref rq) = self.rabitq_index {
            // RaBitQ binary two-pass search
            let metric = collection_metric;
            let all_data = self.vector_store.read_all_f32()?;
            let (indices, dists) = rq.search(query, search_k, &all_data, metric, RABITQ_OVERSAMPLE);
            (indices.into_iter().map(|i| i as u64).collect(), dists)
        } else if let Some(ref pv) = self.polarvec_index {
            // PolarVec multi-bit LUT two-pass search
            let metric = collection_metric;
            let all_data = self.vector_store.read_all_f32()?;
            let (indices, dists) =
                pv.search(query, search_k, &all_data, metric, POLARVEC_OVERSAMPLE);
            (indices.into_iter().map(|i| i as u64).collect(), dists)
        } else {
            // Unfiltered: zero-copy mmap + fused parallel topk (~5ms for 1M×128)
            let metric = collection_metric;
            self.vector_store.search(
                query,
                search_k,
                metric,
                self.resolve_use_sq8(),
                approx_cfg,
            )?
        };

        let (pending_ids, pending_dists) = self.pending_search(
            query,
            search_k,
            search_params
                .subset_indices
                .as_ref()
                .map(|subset| subset.as_slice()),
            collection_metric,
        );
        let (result_ids, result_dists) = self.merge_row_results(
            result_ids,
            result_dists,
            pending_ids,
            pending_dists,
            search_k,
            collection_metric,
        );

        let result_ids: Vec<u64> = result_ids
            .iter()
            .map(|&row| self.row_to_user_id(row))
            .collect();
        let (result_ids, result_dists) = self.filter_tombstoned_limit(result_ids, result_dists, k);

        Ok(SearchResult {
            ids: result_ids,
            distances: result_dists,
            fields: Vec::new(),
            index_mode: self.index_mode.clone().unwrap_or("FLAT-IP".into()),
            dimension: dim,
            k,
        })
    }

    /// Search a named vector field. Passing `default` uses the primary vector.
    pub fn search_vector_field(
        &self,
        field_name: &str,
        query: &[f32],
        k: usize,
        where_expr: Option<&str>,
    ) -> Result<SearchResult> {
        self.search_vector_field_with_options(field_name, query, k, where_expr, false, 1e-4)
    }

    /// Search a named vector field with flat-search approximation controls.
    pub fn search_vector_field_with_options(
        &self,
        field_name: &str,
        query: &[f32],
        k: usize,
        where_expr: Option<&str>,
        approx: bool,
        eps: f32,
    ) -> Result<SearchResult> {
        if field_name == DEFAULT_VECTOR_FIELD_NAME || field_name.trim().is_empty() {
            return self.search(query, k, where_expr, 10, approx, eps);
        }

        let field_name = Self::validate_vector_field_name(field_name)?;
        let field = self.named_vector_fields.get(&field_name).ok_or_else(|| {
            LynseError::InvalidArgument(format!("vector field '{}' does not exist", field_name))
        })?;
        let dim = field.config.dimension;
        if query.len() != dim {
            return Err(LynseError::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        }

        let allowed_ids = if let Some(filter) = where_expr {
            let keys = self.field_store.query(filter)?;
            Some(
                self.field_keys_to_user_ids(keys)
                    .into_iter()
                    .collect::<HashSet<u64>>(),
            )
        } else {
            None
        };

        let has_tombstones = !self.tombstone.read().is_empty();
        let use_field_index = field.index.is_some() && allowed_ids.is_none() && !has_tombstones;
        let search_k = if allowed_ids.is_some() || has_tombstones {
            field.id_map.len()
        } else {
            k
        };

        let (rows, dists) = if use_field_index {
            let search_params = SearchParams {
                k,
                nprobe: 10,
                ef_search: None,
                subset_indices: None,
            };
            field
                .index
                .as_ref()
                .unwrap()
                .search(query, k, &search_params)?
        } else {
            let metric = DistanceMetric::from_str(&field.config.metric)
                .unwrap_or_else(|| Self::metric_from_mode_str(&field.config.index_mode));
            let use_sq8 = field.config.index_mode.to_uppercase().contains("SQ8");
            let approx_cfg = if approx && metric.supports_flat_approx() {
                Some(crate::storage::approx_search::ApproxSearchConfig::new(eps))
            } else {
                None
            };
            field
                .vector_store
                .search(query, search_k, metric, use_sq8, approx_cfg)?
        };

        let tombstones = self.tombstone.read();
        let mut ids = Vec::with_capacity(rows.len().min(k));
        let mut distances = Vec::with_capacity(rows.len().min(k));

        for (row, dist) in rows.into_iter().zip(dists.into_iter()) {
            let Some(&user_id) = field.id_map.get(row as usize) else {
                continue;
            };
            if tombstones.contains(&user_id) {
                continue;
            }
            if let Some(ref allowed) = allowed_ids {
                if !allowed.contains(&user_id) {
                    continue;
                }
            }
            ids.push(user_id);
            distances.push(dist);
            if ids.len() >= k {
                break;
            }
        }

        Ok(SearchResult {
            ids,
            distances,
            fields: Vec::new(),
            index_mode: field.config.index_mode.clone(),
            dimension: dim,
            k,
        })
    }

    /// Search sparse vectors with inner product. Optional metadata filters use
    /// the same field expression language as dense vector search.
    pub fn search_sparse(
        &self,
        query: &[(u32, f32)],
        k: usize,
        where_expr: Option<&str>,
    ) -> Result<SearchResult> {
        let query = normalize_sparse_entries(query)?;
        if query.is_empty() || k == 0 {
            return Ok(SearchResult {
                ids: Vec::new(),
                distances: Vec::new(),
                fields: Vec::new(),
                index_mode: "SPARSE-FLAT-IP".to_string(),
                dimension: 0,
                k,
            });
        }

        let allowed_ids = if let Some(filter) = where_expr {
            let keys = self.field_store.query(filter)?;
            Some(self.field_keys_to_user_ids(keys).into_iter().collect())
        } else {
            None
        };

        let tombstones = self.tombstone.read();
        let ranked = self
            .sparse_vectors
            .search(&query, k, allowed_ids.as_ref(), &tombstones);

        Ok(SearchResult {
            ids: ranked.iter().map(|(id, _)| *id).collect(),
            distances: ranked.iter().map(|(_, score)| *score).collect(),
            fields: Vec::new(),
            index_mode: "SPARSE-FLAT-IP".to_string(),
            dimension: 0,
            k,
        })
    }

    /// Search with lightweight profile information for explain/debug tooling.
    pub fn search_with_profile(
        &self,
        query: &[f32],
        k: usize,
        where_expr: Option<&str>,
        nprobe: usize,
        approx: bool,
        eps: f32,
    ) -> Result<(SearchResult, QueryProfile)> {
        let started = Instant::now();
        let mut filter_us = 0u64;
        let mut filter_matches = None;

        let subset_indices = if let Some(expr) = where_expr {
            let filter_started = Instant::now();
            let ids = self.field_keys_to_rows(self.field_store.query(expr)?);
            filter_us = elapsed_micros_u64(filter_started);
            filter_matches = Some(ids.len());
            Some(Arc::new(ids))
        } else {
            None
        };

        let search_started = Instant::now();
        let result =
            self.search_with_precomputed_filter(query, k, subset_indices, nprobe, approx, eps)?;
        let search_us = elapsed_micros_u64(search_started);
        let total_vectors = self.shape()?.0;
        let filtered = where_expr.is_some();
        let index_path = self.profile_index_path(filtered);
        let scanned_vectors = self.estimate_scanned_vectors(total_vectors as usize, filter_matches);
        let result_count = result.ids.len();

        let profile = QueryProfile {
            query_kind: "vector".to_string(),
            vector_field: "default".to_string(),
            index_path,
            total_vectors,
            filter_expression: where_expr.map(|s| s.to_string()),
            filter_matches,
            scanned_vectors,
            result_count,
            filter_us,
            search_us,
            rerank_us: 0,
            total_us: elapsed_micros_u64(started),
        };

        Ok((result, profile))
    }

    /// BM25 text search over stored metadata fields.
    ///
    /// Uses the persistent inverted index when available, with a scan fallback
    /// for legacy collections opened before the text index file exists.
    pub fn text_search(
        &self,
        query_text: &str,
        text_fields: Option<&[String]>,
        k: usize,
        where_expr: Option<&str>,
    ) -> Result<SearchResult> {
        let (scores, index_mode) = self.bm25_text_scores(query_text, text_fields, where_expr, k)?;
        Ok(SearchResult {
            ids: scores.iter().map(|(id, _)| *id).collect(),
            distances: scores.iter().map(|(_, score)| *score).collect(),
            fields: Vec::new(),
            index_mode: index_mode.to_string(),
            dimension: self.meta.dimension,
            k,
        })
    }

    /// Hybrid vector + text search with RRF or weighted-score fusion.
    pub fn hybrid_search(
        &self,
        query: Option<&[f32]>,
        query_text: Option<&str>,
        k: usize,
        where_expr: Option<&str>,
        text_fields: Option<&[String]>,
        fusion: &str,
        vector_weight: f32,
        text_weight: f32,
        rrf_k: f32,
        candidate_limit: usize,
        nprobe: usize,
    ) -> Result<SearchResult> {
        if query.is_none() && query_text.map(|s| s.trim().is_empty()).unwrap_or(true) {
            return Err(LynseError::InvalidArgument(
                "hybrid_search requires a vector, text, or both".to_string(),
            ));
        }

        let candidate_limit = candidate_limit.max(k).max(1);
        let mut fused: HashMap<u64, f32> = HashMap::new();
        let subset_indices = if let Some(filter) = where_expr {
            Some(Arc::new(
                self.field_keys_to_rows(self.field_store.query(filter)?),
            ))
        } else {
            None
        };
        if let Some(vector) = query {
            let vector_result = self.search_with_precomputed_filter(
                vector,
                candidate_limit,
                subset_indices.clone(),
                nprobe,
                false,
                1e-4,
            )?;
            let vector_scores =
                normalize_vector_scores(&vector_result.distances, self.resolve_metric());
            add_fused_scores(
                &mut fused,
                &vector_result.ids,
                &vector_scores,
                fusion,
                vector_weight,
                rrf_k,
            );
        }

        if let Some(text) = query_text {
            if !text.trim().is_empty() {
                let allowed_text_ids = subset_indices.as_ref().map(|rows| {
                    rows.iter()
                        .map(|&row| self.row_to_user_id(row))
                        .collect::<HashSet<u64>>()
                });
                let (text_scores, _) = self.bm25_text_scores_with_allowed_ids(
                    text,
                    text_fields,
                    allowed_text_ids.as_ref(),
                    candidate_limit,
                )?;
                let ids: Vec<u64> = text_scores.iter().map(|(id, _)| *id).collect();
                let scores: Vec<f32> = text_scores.iter().map(|(_, score)| *score).collect();
                let normalized = normalize_scores(&scores, false);
                add_fused_scores(&mut fused, &ids, &normalized, fusion, text_weight, rrf_k);
            }
        }

        let mut ranked: Vec<(u64, f32)> = fused.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);

        let normalized_fusion = if fusion.eq_ignore_ascii_case("weighted") {
            "WEIGHTED"
        } else {
            "RRF"
        };

        Ok(SearchResult {
            ids: ranked.iter().map(|(id, _)| *id).collect(),
            distances: ranked.iter().map(|(_, score)| *score).collect(),
            fields: Vec::new(),
            index_mode: format!("HYBRID-{}", normalized_fusion),
            dimension: self.meta.dimension,
            k,
        })
    }

    fn profile_index_path(&self, filtered: bool) -> String {
        if self.index.is_some() {
            "ann_index".to_string()
        } else if self.pq_index.is_some() {
            "pq_two_pass".to_string()
        } else if self.rabitq_index.is_some() {
            "rabitq_two_pass".to_string()
        } else if self.polarvec_index.is_some() {
            "polarvec_two_pass".to_string()
        } else if filtered {
            "flat_mmap_filtered".to_string()
        } else {
            "flat_mmap".to_string()
        }
    }

    fn estimate_scanned_vectors(
        &self,
        total_vectors: usize,
        filter_matches: Option<usize>,
    ) -> usize {
        if self.index.is_some()
            || self.pq_index.is_some()
            || self.rabitq_index.is_some()
            || self.polarvec_index.is_some()
        {
            filter_matches.unwrap_or(total_vectors)
        } else {
            filter_matches.unwrap_or(total_vectors)
        }
    }

    fn bm25_text_scores(
        &self,
        query_text: &str,
        text_fields: Option<&[String]>,
        where_expr: Option<&str>,
        limit: usize,
    ) -> Result<(Vec<(u64, f32)>, &'static str)> {
        let allowed_ids = if let Some(expr) = where_expr {
            let keys = self.field_store.query(expr)?;
            Some(self.field_keys_to_user_ids(keys).into_iter().collect())
        } else {
            None
        };

        self.bm25_text_scores_with_allowed_ids(query_text, text_fields, allowed_ids.as_ref(), limit)
    }

    fn bm25_text_scores_with_allowed_ids(
        &self,
        query_text: &str,
        text_fields: Option<&[String]>,
        allowed_ids: Option<&HashSet<u64>>,
        limit: usize,
    ) -> Result<(Vec<(u64, f32)>, &'static str)> {
        if self.text_index.is_empty() {
            return self
                .bm25_text_scores_scan_allowed_ids(query_text, text_fields, allowed_ids, limit)
                .map(|scores| (scores, "BM25-SCAN"));
        }

        let tombstones = self.tombstone.read();
        let scores =
            self.text_index
                .search(query_text, text_fields, limit, allowed_ids, &tombstones);
        Ok((scores, "BM25-INVERTED"))
    }

    fn bm25_text_scores_scan_allowed_ids(
        &self,
        query_text: &str,
        text_fields: Option<&[String]>,
        allowed_ids: Option<&HashSet<u64>>,
        limit: usize,
    ) -> Result<Vec<(u64, f32)>> {
        if let Some(allowed) = allowed_ids {
            let field_keys: Vec<u64> = if self.fields_use_stable_ids {
                allowed.iter().copied().collect()
            } else {
                allowed
                    .iter()
                    .filter_map(|&id| self.user_id_to_row(id).map(|row| row as u64))
                    .collect()
            };
            self.bm25_text_scores_scan_rows(query_text, text_fields, &field_keys, limit)
        } else {
            self.bm25_text_scores_scan(query_text, text_fields, None, limit)
        }
    }

    fn bm25_text_scores_scan(
        &self,
        query_text: &str,
        text_fields: Option<&[String]>,
        where_expr: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(u64, f32)>> {
        let field_keys: Vec<u64> = if let Some(expr) = where_expr {
            self.field_store.query(expr)?
        } else if self.fields_use_stable_ids {
            self.id_map.clone()
        } else {
            let (n, _) = self.shape()?;
            (0..n).collect()
        };

        self.bm25_text_scores_scan_rows(query_text, text_fields, &field_keys, limit)
    }

    fn bm25_text_scores_scan_rows(
        &self,
        query_text: &str,
        text_fields: Option<&[String]>,
        row_ids: &[u64],
        limit: usize,
    ) -> Result<Vec<(u64, f32)>> {
        let query_terms = tokenize_text(query_text);
        if query_terms.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let field_filter = TextFieldSelection::from_fields(text_fields);

        let mut docs: Vec<(u64, Vec<String>)> = Vec::new();
        let fields_list = self.field_store.retrieve_many(row_ids)?;
        for (field_key, fields) in row_ids.iter().copied().zip(fields_list.into_iter()) {
            let user_id = if self.fields_use_stable_ids {
                field_key
            } else {
                self.row_to_user_id(field_key)
            };
            if self.tombstone.read().contains(&user_id) {
                continue;
            }

            let text = fields_to_searchable_text(&fields, &field_filter);
            let tokens = tokenize_text(&text);
            if !tokens.is_empty() {
                docs.push((user_id, tokens));
            }
        }

        if docs.is_empty() {
            return Ok(Vec::new());
        }

        let mut query_counts: HashMap<String, usize> = HashMap::new();
        for term in query_terms {
            *query_counts.entry(term).or_insert(0) += 1;
        }

        let mut document_frequencies: HashMap<&str, usize> = HashMap::new();
        let mut term_counts_by_doc: Vec<HashMap<&str, usize>> = Vec::with_capacity(docs.len());
        let mut total_doc_len = 0usize;

        for (_, tokens) in &docs {
            total_doc_len += tokens.len();
            let mut counts: HashMap<&str, usize> = HashMap::new();
            for token in tokens {
                *counts.entry(token.as_str()).or_insert(0) += 1;
            }
            for term in query_counts.keys() {
                if counts.contains_key(term.as_str()) {
                    *document_frequencies.entry(term.as_str()).or_insert(0) += 1;
                }
            }
            term_counts_by_doc.push(counts);
        }

        let avg_doc_len = (total_doc_len as f32 / docs.len() as f32).max(1.0);
        let n_docs = docs.len() as f32;
        let k1 = 1.2f32;
        let b = 0.75f32;
        let mut scored = Vec::new();

        for ((id, tokens), counts) in docs.iter().zip(term_counts_by_doc.iter()) {
            let doc_len = tokens.len() as f32;
            let mut score = 0.0f32;

            for (term, query_count) in &query_counts {
                let tf = counts.get(term.as_str()).copied().unwrap_or(0) as f32;
                if tf == 0.0 {
                    continue;
                }
                let df = document_frequencies
                    .get(term.as_str())
                    .copied()
                    .unwrap_or(0) as f32;
                let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();
                let denom = tf + k1 * (1.0 - b + b * doc_len / avg_doc_len);
                score += *query_count as f32 * idf * (tf * (k1 + 1.0)) / denom;
            }

            if score > 0.0 {
                scored.push((*id, score));
            }
        }

        sort_truncate_scores_desc(&mut scored, limit);
        Ok(scored)
    }

    /// Batch search: search multiple query vectors in parallel.
    /// Returns one SearchResult per query.
    pub fn batch_search(
        &self,
        queries: &[f32],
        n_queries: usize,
        k: usize,
        where_expr: Option<&str>,
        nprobe: usize,
    ) -> Result<Vec<SearchResult>> {
        let dim = self.meta.dimension;
        if queries.len() != n_queries * dim {
            return Err(LynseError::DimensionMismatch {
                expected: dim * n_queries,
                got: queries.len(),
            });
        }

        if self.pending_len() > 0 {
            let mut results = Vec::with_capacity(n_queries);
            for i in 0..n_queries {
                let start = i * dim;
                let query = &queries[start..start + dim];
                results.push(self.search(query, k, where_expr, nprobe, false, 1e-4)?);
            }
            return Ok(results);
        }

        // Pre-compute filter once (shared across all queries)
        let subset_indices = if let Some(filter) = where_expr {
            Some(Arc::new(
                self.field_keys_to_rows(self.field_store.query(filter)?),
            ))
        } else {
            None
        };

        if subset_indices.is_none()
            && self.index.is_none()
            && self.pq_index.is_none()
            && self.rabitq_index.is_none()
            && self.polarvec_index.is_none()
            && !self.resolve_use_sq8()
        {
            let metric = self.resolve_metric();
            if self.vector_dtype == VectorDtype::F16 {
                let tombstone_count = self.tombstone.read().len();
                let search_k = if tombstone_count == 0 {
                    k
                } else {
                    k.saturating_add(tombstone_count)
                };
                let all_data = self.vector_store.read_all_f32()?;
                let n_rows = self.vector_store.get_shape()?.0 as usize;
                let (batch_rows, batch_dists) =
                    crate::storage::flat_mmap::batch_exact_flat_search_f32(
                        queries, n_queries, &all_data, dim, search_k, n_rows, metric,
                    );
                let results = batch_rows
                    .into_iter()
                    .zip(batch_dists)
                    .map(|(rows, dists)| {
                        let result_ids: Vec<u64> = rows
                            .iter()
                            .map(|&row| self.row_to_user_id(row as u64))
                            .collect();
                        let (result_ids, result_dists) =
                            self.filter_tombstoned_limit(result_ids, dists, k);
                        SearchResult {
                            ids: result_ids,
                            distances: result_dists,
                            fields: Vec::new(),
                            index_mode: self.index_mode.clone().unwrap_or("FLAT-IP".into()),
                            dimension: dim,
                            k,
                        }
                    })
                    .collect();
                return Ok(results);
            }
        }

        use rayon::prelude::*;

        let results: Vec<Result<SearchResult>> = (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let start = i * dim;
                let query = &queries[start..start + dim];
                let tombstone_count = self.tombstone.read().len();
                let search_k = if tombstone_count == 0 {
                    k
                } else {
                    k.saturating_add(tombstone_count)
                };

                let search_params = SearchParams {
                    k: search_k,
                    nprobe,
                    ef_search: None,
                    subset_indices: subset_indices.clone(),
                };

                let (result_ids, result_dists) = if let Some(ref idx) = self.index {
                    idx.search(query, search_k, &search_params)?
                } else if let Some(ref pq) = self.pq_index {
                    let metric = self.resolve_metric();
                    let guard = self.vector_store.read_mmap()?;
                    match guard.as_ref() {
                        None => (vec![], vec![]),
                        Some(fm) => {
                            let all_data = fm.as_f32_cow();
                            let (indices, dists) =
                                pq.search(query, search_k, &all_data, metric, PQ_OVERSAMPLE);
                            let ids: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
                            (ids, dists)
                        }
                    }
                } else if let Some(ref rq) = self.rabitq_index {
                    let metric = self.resolve_metric();
                    let guard = self.vector_store.read_mmap()?;
                    match guard.as_ref() {
                        None => (vec![], vec![]),
                        Some(fm) => {
                            let all_data = fm.as_f32_cow();
                            let (indices, dists) =
                                rq.search(query, search_k, &all_data, metric, RABITQ_OVERSAMPLE);
                            let ids: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
                            (ids, dists)
                        }
                    }
                } else if let Some(ref pv) = self.polarvec_index {
                    let metric = self.resolve_metric();
                    let guard = self.vector_store.read_mmap()?;
                    match guard.as_ref() {
                        None => (vec![], vec![]),
                        Some(fm) => {
                            let all_data = fm.as_f32_cow();
                            let (indices, dists) =
                                pv.search(query, search_k, &all_data, metric, POLARVEC_OVERSAMPLE);
                            let ids: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
                            (ids, dists)
                        }
                    }
                } else if search_params.subset_indices.is_some() {
                    self.brute_force_search_filtered(query, search_k, &search_params)?
                } else {
                    let metric = self.resolve_metric();
                    self.vector_store.search(
                        query,
                        search_k,
                        metric,
                        self.resolve_use_sq8(),
                        None,
                    )?
                };

                let result_ids: Vec<u64> = result_ids
                    .iter()
                    .map(|&row| self.row_to_user_id(row))
                    .collect();
                let (result_ids, result_dists) =
                    self.filter_tombstoned_limit(result_ids, result_dists, k);

                Ok(SearchResult {
                    ids: result_ids,
                    distances: result_dists,
                    fields: Vec::new(),
                    index_mode: self.index_mode.clone().unwrap_or("FLAT-IP".into()),
                    dimension: dim,
                    k,
                })
            })
            .collect();

        results.into_iter().collect()
    }

    /// Resolve the distance metric from the index_mode string.
    fn resolve_metric(&self) -> DistanceMetric {
        self.index_mode
            .as_ref()
            .map(|m| Self::metric_from_mode_str(m))
            .unwrap_or(DistanceMetric::InnerProduct)
    }

    /// Filtered brute-force search on mmap data.
    ///
    /// Delegates to FlatMmap::search_filtered() which uses two strategies:
    /// - Few matches (≤ 50K): direct random access — O(matches) not O(n)
    /// - Many matches (> 50K): parallel scan with bitset — same SIMD parallelism as unfiltered
    fn brute_force_search_filtered(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        let metric = self.resolve_metric();

        let subset = match &params.subset_indices {
            Some(s) => s.as_slice(),
            None => {
                // No filter — use unfiltered path
                return self.vector_store.search(
                    query,
                    k,
                    metric,
                    self.resolve_use_sq8(),
                    None,
                );
            }
        };

        if subset.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        self.vector_store.search_filtered(query, k, metric, subset)
    }

    // ─── Filtered search strategies for benchmarking ────────────────────────

    /// Strategy 1: Pre-filter with copy (current approach).
    /// field_store.query() → copy matching vectors → top_k on contiguous subset.
    fn search_prefilter_copy(
        &self,
        query: &[f32],
        k: usize,
        subset: &[u64],
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        let dim = self.meta.dimension;
        let metric = self.resolve_metric();

        let guard = self.vector_store.read_mmap()?;
        let fm = match guard.as_ref() {
            None => return Ok((Vec::new(), Vec::new())),
            Some(fm) => fm,
        };
        let all_data = fm.as_f32_cow();
        let n_vectors = fm.len();

        let subset_set: std::collections::HashSet<u64> = subset.iter().cloned().collect();
        let mut filtered_data: Vec<f32> = Vec::with_capacity(subset.len() * dim);
        let mut filtered_ids: Vec<u64> = Vec::with_capacity(subset.len());

        for id in 0..n_vectors {
            let uid = id as u64;
            if subset_set.contains(&uid) {
                let start = id * dim;
                let end = start + dim;
                if end <= all_data.len() {
                    filtered_data.extend_from_slice(&all_data[start..end]);
                    filtered_ids.push(uid);
                }
            }
        }

        if filtered_ids.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let (top_idx, top_dist) =
            crate::distance::top_k_search(query, &filtered_data, dim, k, metric);
        let result_ids: Vec<u64> = top_idx.iter().map(|&i| filtered_ids[i as usize]).collect();
        Ok((result_ids, top_dist))
    }

    /// Strategy 2: Fused pre-filter (no copy).
    /// Scan mmap data in-place, compute distance only for matching IDs, skip others.
    /// Uses a BitSet for O(1) membership check. No vector data is copied.
    fn search_prefilter_fused(
        &self,
        query: &[f32],
        k: usize,
        subset: &[u64],
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        use crate::distance::simd;
        use crate::distance::{self, DistanceMetric};

        let dim = self.meta.dimension;
        let metric = self.resolve_metric();

        let guard = self.vector_store.read_mmap()?;
        let fm = match guard.as_ref() {
            None => return Ok((Vec::new(), Vec::new())),
            Some(fm) => fm,
        };
        let all_data = fm.as_f32_cow();
        let n_vectors = fm.len();

        if subset.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        // Build BitSet for O(1) membership test
        let max_id = subset.iter().copied().max().unwrap_or(0) as usize;
        let mut bitset = vec![0u64; (max_id / 64) + 1];
        for &id in subset {
            let idx = id as usize;
            bitset[idx / 64] |= 1u64 << (idx % 64);
        }

        // Select distance function
        let dist_fn: Box<dyn Fn(&[f32], &[f32]) -> f32> = match metric {
            DistanceMetric::InnerProduct => {
                Box::new(|a: &[f32], b: &[f32]| simd::inner_product_f32(a, b))
            }
            DistanceMetric::L2Squared => {
                Box::new(|a: &[f32], b: &[f32]| simd::l2_squared_f32(a, b))
            }
            DistanceMetric::Cosine => {
                Box::new(|a: &[f32], b: &[f32]| simd::cosine_distance_f32(a, b))
            }
            _ => Box::new(move |a: &[f32], b: &[f32]| distance::compute_distance_f32(a, b, metric)),
        };
        let ascending = metric.is_ascending();

        // Fused scan: iterate mmap, skip non-matching, compute distance inline
        // Use simple sorted Vec for top-k tracking (no FloatOrd needed)
        let mut top: Vec<(f32, u64)> = Vec::with_capacity(k);
        let mut threshold = if ascending {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };
        let mut filled = false;

        for id in 0..n_vectors {
            // BitSet membership check
            if id <= max_id && (bitset[id / 64] & (1u64 << (id % 64))) != 0 {
                let start = id * dim;
                let cand = &all_data[start..start + dim];
                let dist = dist_fn(query, cand);

                if !filled {
                    top.push((dist, id as u64));
                    if top.len() == k {
                        if ascending {
                            top.sort_by(|a, b| {
                                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                            });
                        } else {
                            top.sort_by(|a, b| {
                                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                            });
                        }
                        threshold = top[k - 1].0;
                        filled = true;
                    }
                } else {
                    let dominated = if ascending {
                        dist < threshold
                    } else {
                        dist > threshold
                    };
                    if dominated {
                        top[k - 1] = (dist, id as u64);
                        // Insertion sort to maintain order
                        let mut j = k - 1;
                        if ascending {
                            while j > 0 && top[j].0 < top[j - 1].0 {
                                top.swap(j, j - 1);
                                j -= 1;
                            }
                        } else {
                            while j > 0 && top[j].0 > top[j - 1].0 {
                                top.swap(j, j - 1);
                                j -= 1;
                            }
                        }
                        threshold = top[k - 1].0;
                    }
                }
            }
        }

        if !filled && !top.is_empty() {
            if ascending {
                top.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        let ids = top.iter().map(|(_, id)| *id).collect();
        let dists = top.iter().map(|(d, _)| *d).collect();
        Ok((ids, dists))
    }

    /// Strategy 3: Post-filter (distance-first).
    /// Run full FlatMmap optimized scan (SIMD, parallel, SQ8) on ALL vectors
    /// with k * oversample_factor, then filter results by matching IDs.
    fn search_postfilter(
        &self,
        query: &[f32],
        k: usize,
        subset: &[u64],
        oversample: usize,
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        let metric = self.resolve_metric();

        let guard = self.vector_store.read_mmap()?;
        let fm = match guard.as_ref() {
            None => return Ok((Vec::new(), Vec::new())),
            Some(fm) => fm,
        };

        if subset.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        // Build BitSet for O(1) membership test
        let max_id = subset.iter().copied().max().unwrap_or(0) as usize;
        let mut bitset = vec![0u64; (max_id / 64) + 1];
        for &id in subset {
            let idx = id as usize;
            bitset[idx / 64] |= 1u64 << (idx % 64);
        }

        // Search with oversample: get more results than needed
        let search_k = (k * oversample).min(fm.len());
        let use_sq8 = self.resolve_use_sq8();
        let (indices, dists) = fm.search(query, search_k, metric, use_sq8, None);

        // Filter by matching IDs
        let mut result_ids = Vec::with_capacity(k);
        let mut result_dists = Vec::with_capacity(k);
        for (&idx, &dist) in indices.iter().zip(dists.iter()) {
            let uid = idx as u64;
            let i = uid as usize;
            if i <= max_id && (bitset[i / 64] & (1u64 << (i % 64))) != 0 {
                result_ids.push(uid);
                result_dists.push(dist);
                if result_ids.len() == k {
                    break;
                }
            }
        }

        Ok((result_ids, result_dists))
    }

    /// Benchmark all three filtered search strategies.
    /// Returns timing in microseconds for each: (prefilter_copy_us, prefilter_fused_us, postfilter_us)
    /// and result counts for verification.
    pub fn benchmark_filtered_search(
        &self,
        query: &[f32],
        k: usize,
        where_expr: &str,
        oversample: usize,
        warmup: usize,
        iterations: usize,
    ) -> Result<Vec<(String, f64, usize)>> {
        let dim = self.meta.dimension;
        if query.len() != dim {
            return Err(LynseError::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        }

        // Get matching IDs once (shared by all strategies)
        let t0 = std::time::Instant::now();
        let subset = self.field_keys_to_rows(self.field_store.query(where_expr)?);
        let field_query_us = t0.elapsed().as_micros() as f64;
        let n_matches = subset.len();

        let mut results = Vec::new();
        results.push(("field_query".to_string(), field_query_us, n_matches));

        // Strategy 1: Pre-filter with copy
        for _ in 0..warmup {
            let _ = self.search_prefilter_copy(query, k, &subset)?;
        }
        let t1 = std::time::Instant::now();
        let mut r1_len = 0;
        for _ in 0..iterations {
            let (ids, _) = self.search_prefilter_copy(query, k, &subset)?;
            r1_len = ids.len();
        }
        let us1 = t1.elapsed().as_micros() as f64 / iterations as f64;
        results.push(("prefilter_copy".to_string(), us1, r1_len));

        // Strategy 2: Fused pre-filter (no copy)
        for _ in 0..warmup {
            let _ = self.search_prefilter_fused(query, k, &subset)?;
        }
        let t2 = std::time::Instant::now();
        let mut r2_len = 0;
        for _ in 0..iterations {
            let (ids, _) = self.search_prefilter_fused(query, k, &subset)?;
            r2_len = ids.len();
        }
        let us2 = t2.elapsed().as_micros() as f64 / iterations as f64;
        results.push(("prefilter_fused".to_string(), us2, r2_len));

        // Strategy 3: Post-filter (distance-first)
        for _ in 0..warmup {
            let _ = self.search_postfilter(query, k, &subset, oversample)?;
        }
        let t3 = std::time::Instant::now();
        let mut r3_len = 0;
        for _ in 0..iterations {
            let (ids, _) = self.search_postfilter(query, k, &subset, oversample)?;
            r3_len = ids.len();
        }
        let us3 = t3.elapsed().as_micros() as f64 / iterations as f64;
        results.push(("postfilter".to_string(), us3, r3_len));

        // Also time unfiltered search for reference
        {
            let metric = self.resolve_metric();
            let guard = self.vector_store.read_mmap()?;
            if let Some(fm) = guard.as_ref() {
                let use_sq8 = self.resolve_use_sq8();
                for _ in 0..warmup {
                    let _ = fm.search(query, k, metric, use_sq8, None);
                }
                let t4 = std::time::Instant::now();
                for _ in 0..iterations {
                    let _ = fm.search(query, k, metric, use_sq8, None);
                }
                let us4 = t4.elapsed().as_micros() as f64 / iterations as f64;
                results.push(("unfiltered_mmap".to_string(), us4, k));
            }
        }

        Ok(results)
    }

    fn update_existing_items_in_place(
        &mut self,
        ids: &[u64],
        vectors: &[f32],
        fields: Option<&[HashMap<String, serde_json::Value>]>,
    ) -> Result<()> {
        let rows: Vec<u64> = ids
            .iter()
            .map(|id| {
                self.reverse_id_map
                    .get(id)
                    .copied()
                    .map(|row| row as u64)
                    .ok_or_else(|| {
                        LynseError::Storage(format!("existing ID {id} has no vector row"))
                    })
            })
            .collect::<Result<_>>()?;
        let encoded = encode_f32_slice_as_le_bytes(vectors, self.vector_dtype);
        self.vector_store
            .apply_journaled_encoded_rows(&rows, &encoded, self.vector_dtype)?;

        if let Some(field_values) = fields {
            let field_rows: Vec<u64> = if self.fields_use_stable_ids {
                ids.to_vec()
            } else {
                rows.clone()
            };
            self.field_store
                .replace_fields_at_ids(&field_rows, field_values)?;
            self.text_index.upsert_documents(ids, field_values, true)?;
        }

        let tombstone_changed = {
            let mut tombstone = self.tombstone.write();
            ids.iter()
                .fold(false, |changed, id| tombstone.remove(id) || changed)
        };
        if tombstone_changed {
            let path = self.path.join("tombstone.bin");
            let tombstone = self.tombstone.read();
            Self::save_tombstone_to_disk(&path, &tombstone)?;
        }

        if let Some(mode) = self.index_mode.clone() {
            self.build_index(&mode)?;
        }
        self.vector_store.finish_pending_updates()?;
        Ok(())
    }

    /// Upsert vectors by user ID.
    ///
    /// Existing IDs are updated in place by rewriting the vector file with the
    /// corresponding row replaced. New IDs are appended. If fields are provided,
    /// each provided field map replaces the row's existing fields; if fields are
    /// omitted, existing fields are preserved and new rows get no fields.
    pub fn upsert_items(
        &mut self,
        ids: &[u64],
        vectors: &[f32],
        n_vectors: usize,
        fields: Option<&[HashMap<String, serde_json::Value>]>,
    ) -> Result<()> {
        self.ensure_writable()?;
        self.flush_pending_ingest()?;
        let dim = self.infer_dimension_from_f32(vectors, n_vectors)?;
        if vectors.len() != n_vectors * dim {
            return Err(LynseError::DimensionMismatch {
                expected: dim,
                got: vectors.len() / n_vectors,
            });
        }
        if ids.len() != n_vectors {
            return Err(LynseError::InvalidArgument(
                "ids length must match n_vectors".to_string(),
            ));
        }
        if let Some(field_list) = fields {
            if field_list.len() != n_vectors {
                return Err(LynseError::InvalidArgument(format!(
                    "fields length ({}) must match n_vectors ({})",
                    field_list.len(),
                    n_vectors
                )));
            }
        }
        Self::validate_unique_ids(ids)?;

        // Existing-only updates do not change row order or the stable ID map.
        // Use a journaled positional write instead of rebuilding every segment.
        if ids
            .iter()
            .all(|user_id| self.reverse_id_map.contains_key(user_id))
        {
            return self.update_existing_items_in_place(ids, vectors, fields);
        }

        let (mut all_vectors, n_total) = {
            let guard = self.vector_store.read_mmap()?;
            match guard.as_ref() {
                None => (Vec::new(), 0usize),
                Some(fm) => (fm.as_f32_cow().into_owned(), fm.len()),
            }
        };
        if all_vectors.len() != n_total * dim {
            return Err(LynseError::Storage(
                "vector store length does not match collection dimension".to_string(),
            ));
        }

        let mut new_id_map = self.id_map.clone();
        let mut new_reverse_id_map = self.reverse_id_map.clone();
        let mut field_rows = Vec::new();
        let mut field_values = Vec::new();
        let mut tombstone_changed = false;

        {
            let mut tombstone = self.tombstone.write();
            for (i, &user_id) in ids.iter().enumerate() {
                let src_start = i * dim;
                let src_end = src_start + dim;
                let row = if let Some(row) = new_reverse_id_map.get(&user_id).copied() {
                    let dst_start = row * dim;
                    let dst_end = dst_start + dim;
                    all_vectors[dst_start..dst_end].copy_from_slice(&vectors[src_start..src_end]);
                    if tombstone.remove(&user_id) {
                        tombstone_changed = true;
                    }
                    row
                } else {
                    let row = new_id_map.len();
                    all_vectors.extend_from_slice(&vectors[src_start..src_end]);
                    new_id_map.push(user_id);
                    new_reverse_id_map.insert(user_id, row);
                    row
                };

                if let Some(field_list) = fields {
                    field_rows.push(if self.fields_use_stable_ids {
                        user_id
                    } else {
                        row as u64
                    });
                    field_values.push(field_list[i].clone());
                }
            }
        }

        self.vector_store
            .replace_data_with_id_map(&all_vectors, &new_id_map)?;
        self.id_map = new_id_map;
        self.reverse_id_map = new_reverse_id_map;

        if fields.is_some() {
            self.field_store
                .replace_fields_at_ids(&field_rows, &field_values)?;
            self.text_index
                .upsert_documents(ids, fields.unwrap(), true)?;
        }
        if tombstone_changed {
            let path = self.path.join("tombstone.bin");
            let set = self.tombstone.read();
            Self::save_tombstone_to_disk(&path, &set)?;
        }

        if let Some(mode) = self.index_mode.clone() {
            self.build_index(&mode)?;
        }

        Ok(())
    }

    /// Update existing vectors by ID.
    ///
    /// All IDs must already exist. For insert-or-update behavior, use
    /// `upsert_items`.
    pub fn update_items(
        &mut self,
        ids: &[u64],
        vectors: &[f32],
        n_vectors: usize,
        fields: Option<&[HashMap<String, serde_json::Value>]>,
    ) -> Result<()> {
        self.ensure_writable()?;
        let dim = self.meta.dimension;
        if vectors.len() != n_vectors * dim {
            return Err(LynseError::DimensionMismatch {
                expected: dim,
                got: vectors.len() / n_vectors,
            });
        }
        if ids.len() != n_vectors {
            return Err(LynseError::InvalidArgument(
                "ids length must match n_vectors".to_string(),
            ));
        }
        Self::validate_unique_ids(ids)?;
        for &id in ids {
            if !self.reverse_id_map.contains_key(&id) {
                return Err(LynseError::InvalidArgument(format!(
                    "id {} does not exist; use upsert_items to insert missing IDs",
                    id
                )));
            }
        }

        self.upsert_items(ids, vectors, n_vectors, fields)
    }

    /// Remove the index.
    pub fn remove_index(&mut self) -> Result<()> {
        self.ensure_writable()?;
        self.index = None;
        self.index_mode = None;
        self.meta.index_mode = None;
        self.last_sync_fingerprint = None;
        self.pq_index = None;
        self.rabitq_index = None;
        self.polarvec_index = None;

        let index_path = self.path.join("index");
        let meta_path = self.path.join("index_meta");
        let pq_path = self.path.join("pq_index.bin");
        let rq_path = self.path.join("rabitq_index.bin");
        let pv_path = self.path.join("polarvec_index.bin");

        if index_path.exists() {
            std::fs::remove_dir_all(&index_path)?;
        }
        if meta_path.exists() {
            std::fs::remove_dir_all(&meta_path)?;
        }
        if pq_path.exists() {
            std::fs::remove_file(&pq_path)?;
        }
        if rq_path.exists() {
            std::fs::remove_file(&rq_path)?;
        }
        if pv_path.exists() {
            std::fs::remove_file(&pv_path)?;
        }

        Ok(())
    }

    /// Get collection shape (n_vectors, dimension).
    pub fn shape(&self) -> Result<(u64, usize)> {
        let (n, dim) = self.vector_store.get_shape()?;
        Ok((n + self.pending_len() as u64, dim))
    }

    /// Get vector dimension.
    pub fn dimension(&self) -> usize {
        self.meta.dimension
    }

    /// Get read access to the vector store's FlatMmap for zero-copy reads.
    pub fn vector_store_read_mmap(
        &self,
    ) -> Result<parking_lot::RwLockReadGuard<'_, Option<crate::storage::flat_mmap::FlatMmap>>> {
        self.vector_store.read_mmap()
    }

    /// Get the current storage fingerprint.
    pub fn fingerprint(&self) -> String {
        self.vector_store.fingerprint().unwrap_or_default()
    }

    /// Check if the index needs to be synced with new data.
    pub fn needs_index_sync(&self) -> bool {
        let current_fp = self.vector_store.fingerprint().unwrap_or_default();
        match &self.last_sync_fingerprint {
            Some(fp) => fp != &current_fp,
            None => self.index.is_some(),
        }
    }

    /// Incrementally sync the index with any new data since last sync.
    /// Only rebuilds if the storage fingerprint has changed.
    ///
    /// Note: For Flat types, no sync is needed — VectorStore's mmap is always
    /// up-to-date (re-mmap'd after every write). Sync only applies to
    /// separate ANN structures such as HNSW/IVF/SPANN/DiskANN.
    pub fn sync_index(&mut self) -> Result<()> {
        self.ensure_writable()?;
        if !self.needs_index_sync() {
            return Ok(());
        }

        let current_fp = self.vector_store.fingerprint().unwrap_or_default();
        let dim = self.meta.dimension;

        if let Some(ref mut idx) = self.index {
            let guard = self.vector_store.read_mmap()?;
            if let Some(fm) = guard.as_ref() {
                let all_data = fm.as_f32_cow();
                let n_vectors = fm.len();

                if n_vectors > 0 {
                    let ids: Vec<u64> = (0..n_vectors as u64).collect();
                    idx.build(&all_data, n_vectors, dim, Some(&ids))?;

                    let index_data = idx.serialize()?;
                    let index_path = self.path.join("index");
                    let index_file = Self::write_generation_index(&index_path, &index_data)?;
                    if let Some(index_type) = self.index_mode.as_deref() {
                        let meta_path = self.path.join("index_meta");
                        let meta = serde_json::json!({
                            "index_type": index_type,
                            "n_vectors": n_vectors,
                            "dimension": dim,
                            "index_file": index_file,
                        });
                        Self::save_index_metadata(&meta_path, &meta)?;
                    }
                }
            }
        }

        self.last_sync_fingerprint = Some(current_fp);
        self.save_sync_fingerprint()?;
        Ok(())
    }

    /// Return the first `n` vectors with their user IDs and field metadata.
    ///
    /// Returns `(flat_f32_data, user_ids, fields)`.
    pub fn head(
        &self,
        n: usize,
    ) -> Result<(Vec<f32>, Vec<u64>, Vec<HashMap<String, serde_json::Value>>)> {
        self.flush_pending_ingest()?;
        let total = self.vector_store.get_shape()?.0 as usize;
        let rows: Vec<u64> = (0..n.min(total) as u64).collect();
        let data = self.vector_store.read_rows(&rows)?;
        let user_ids: Vec<u64> = rows.iter().map(|&row| self.row_to_user_id(row)).collect();

        let field_keys = self.field_lookup_keys(&user_ids);
        let fields = self.field_store.retrieve_many(&field_keys)?;
        Ok((data, user_ids, fields))
    }

    /// Return the last `n` vectors with their user IDs and field metadata.
    pub fn tail(
        &self,
        n: usize,
    ) -> Result<(Vec<f32>, Vec<u64>, Vec<HashMap<String, serde_json::Value>>)> {
        self.flush_pending_ingest()?;
        let total = self.vector_store.get_shape()?.0 as usize;
        let start = total.saturating_sub(n.min(total));
        let rows: Vec<u64> = (start as u64..total as u64).collect();
        let data = self.vector_store.read_rows(&rows)?;
        let user_ids: Vec<u64> = rows.iter().map(|&row| self.row_to_user_id(row)).collect();

        let field_keys = self.field_lookup_keys(&user_ids);
        let fields = self.field_store.retrieve_many(&field_keys)?;
        Ok((data, user_ids, fields))
    }

    /// Query field metadata with a SQL-like filter. Returns matching user IDs.
    pub fn query_fields(&self, where_expr: &str) -> Result<Vec<u64>> {
        let field_keys = self.field_store.query(where_expr)?;
        Ok(self
            .field_keys_to_user_ids(field_keys)
            .into_iter()
            .filter(|id| self.reverse_id_map.contains_key(id))
            .collect())
    }

    /// Query field metadata with a SQL-like filter, returning both IDs and fields
    /// in a single ApexBase query. Eliminates the two-query pattern.
    pub fn query_with_fields(
        &self,
        where_expr: &str,
    ) -> Result<(Vec<u64>, Vec<HashMap<String, serde_json::Value>>)> {
        let (field_keys, fields) = self.field_store.query_with_fields(where_expr)?;
        let mut user_ids = Vec::new();
        let mut live_fields = Vec::new();
        for (id, field) in self
            .field_keys_to_user_ids(field_keys)
            .into_iter()
            .zip(fields)
        {
            if self.reverse_id_map.contains_key(&id) {
                user_ids.push(id);
                live_fields.push(field);
            }
        }
        Ok((user_ids, live_fields))
    }

    /// Retrieve field metadata for specific user IDs.
    pub fn retrieve_fields(&self, ids: &[u64]) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        self.field_store.retrieve_many(&self.field_lookup_keys(ids))
    }

    /// List all field names in the collection.
    pub fn list_fields(&self) -> Result<Vec<String>> {
        self.field_store.list_fields()
    }

    /// Get the current index mode string.
    pub fn get_index_mode(&self) -> Option<&str> {
        self.index_mode.as_deref()
    }

    /// Read vectors by their external IDs.
    /// O(1) positional lookup: id maps directly to mmap offset.
    pub fn read_vectors_by_ids(
        &self,
        ids: &[u64],
    ) -> Result<(Vec<f32>, Vec<HashMap<String, serde_json::Value>>)> {
        let (result_data, _) = self.read_vectors_by_ids_only_with_rows(ids)?;
        let fields = self.field_store.retrieve_many(&self.field_lookup_keys(ids))?;
        Ok((result_data, fields))
    }

    /// Read vectors by their external IDs without fetching fields.
    pub fn read_vectors_by_ids_only(&self, ids: &[u64]) -> Result<Vec<f32>> {
        self.read_vectors_by_ids_only_with_rows(ids)
            .map(|(vectors, _)| vectors)
    }

    fn read_vectors_by_ids_only_with_rows(&self, ids: &[u64]) -> Result<(Vec<f32>, Vec<u64>)> {
        let dim = self.meta.dimension;
        let pending_snapshot = {
            let pending = self.pending_ingest.lock();
            if pending.is_empty() {
                None
            } else {
                Some(pending.snapshot())
            }
        };
        let row_offsets: Vec<u64> = ids
            .iter()
            .map(|&uid| self.user_id_to_row(uid).unwrap_or(uid as usize) as u64)
            .collect();
        let persisted_rows = self.vector_store.get_shape()?.0;
        let mut result_data = self.vector_store.read_rows(&row_offsets)?;
        if let Some(snapshot) = pending_snapshot.as_ref() {
            for (out_pos, &row) in row_offsets.iter().enumerate() {
                if row < persisted_rows {
                    continue;
                }
                if let Some(pending_pos) = snapshot
                    .row_offsets
                    .iter()
                    .position(|&pending_row| pending_row == row)
                {
                    let src_start = pending_pos * dim;
                    let src_end = src_start + dim;
                    if src_end <= snapshot.vectors.len() {
                        let dst = &mut result_data[out_pos * dim..(out_pos + 1) * dim];
                        dst.copy_from_slice(&snapshot.vectors[src_start..src_end]);
                    }
                }
            }
        }

        Ok((result_data, row_offsets))
    }

    /// Range search: return all (non-deleted) vectors within a distance threshold.
    ///
    /// For ascending metrics (L2): returns IDs where distance ≤ threshold.
    /// For descending metrics (IP/Cosine): returns IDs where score ≥ threshold.
    /// Results are sorted by distance/score in the natural order for the metric.
    ///
    /// Returns `(ids, distances)` capped at `max_results`.
    pub fn search_range(
        &self,
        query: &[f32],
        threshold: f32,
        max_results: usize,
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        if max_results == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let dim = self.meta.dimension;
        if query.len() != dim {
            return Err(LynseError::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        }

        let metric = self.resolve_metric();
        let ascending = metric.is_ascending();

        let all_data = self.vector_store.read_all_f32()?;
        let n_vectors = self.vector_store.get_shape()?.0 as usize;
        let tombstone = self.tombstone.read();

        let mut result: Vec<(u64, f32)> = Vec::new();

        for i in 0..n_vectors {
            let user_id = self.row_to_user_id(i as u64);
            if tombstone.contains(&user_id) {
                continue;
            }
            let start = i * dim;
            let vec = &all_data[start..start + dim];
            let dist = crate::distance::compute_distance_f32(query, vec, metric);

            let passes = if ascending {
                dist <= threshold
            } else {
                dist >= threshold
            };

            if passes {
                result.push((user_id, dist));
            }
        }
        drop(tombstone);

        if result.len() > max_results {
            if ascending {
                result.select_nth_unstable_by(max_results - 1, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            } else {
                result.select_nth_unstable_by(max_results - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            result.truncate(max_results);
        }

        if ascending {
            result.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            result.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let (ids, dists) = result.into_iter().unzip();
        Ok((ids, dists))
    }

    /// Compact the collection by physically removing all tombstoned vectors.
    ///
    /// Process:
    /// 1. Rewrite only segments containing tombstoned rows.
    /// 2. Atomically publish the new segments and ID map through the manifest.
    /// 4. Clear the tombstone set.
    /// 5. Rebuild the index (if one exists) on the compacted data.
    ///
    /// Returns the number of vectors physically removed.
    pub fn compact(&mut self) -> Result<usize> {
        self.ensure_writable()?;
        self.flush_pending_ingest()?;
        let tombstoned_user_ids: HashSet<u64> = self.tombstone.read().clone();
        if tombstoned_user_ids.is_empty() {
            return Ok(0);
        }

        // Build set of tombstoned row offsets
        let tombstoned_rows: HashSet<usize> = tombstoned_user_ids
            .iter()
            .filter_map(|&uid| self.user_id_to_row(uid))
            .collect();

        let n_removed = tombstoned_rows.len();
        let live_user_ids: Vec<u64> = self
            .id_map
            .iter()
            .enumerate()
            .filter_map(|(row, &id)| (!tombstoned_rows.contains(&row)).then_some(id))
            .collect();
        self.vector_store
            .compact_rows(&tombstoned_rows, &live_user_ids)?;

        // Rebuild in-memory id_map and persist
        self.reverse_id_map = live_user_ids
            .iter()
            .enumerate()
            .map(|(i, &uid)| (uid, i))
            .collect();
        self.id_map = live_user_ids.clone();

        let live_user_id_set: HashSet<u64> = live_user_ids.iter().copied().collect();
        self.internal_to_external_id
            .retain(|internal_id, _| live_user_id_set.contains(internal_id));
        self.external_to_internal_id
            .retain(|_, internal_id| live_user_id_set.contains(internal_id));
        for &internal_id in &live_user_ids {
            if !self.internal_to_external_id.contains_key(&internal_id) {
                let external_id = ExternalId::Int(internal_id);
                self.internal_to_external_id
                    .insert(internal_id, external_id.clone());
                self.external_to_internal_id.insert(external_id, internal_id);
            }
        }
        self.write_external_id_map()?;

        let mut field_indexes_to_rebuild = Vec::new();
        for field in self.named_vector_fields.values_mut() {
            if field.id_map.is_empty() {
                continue;
            }
            let deleted_field_rows: HashSet<usize> = field
                .id_map
                .iter()
                .enumerate()
                .filter_map(|(row, id)| tombstoned_user_ids.contains(id).then_some(row))
                .collect();
            let live_field_ids: Vec<u64> = field
                .id_map
                .iter()
                .copied()
                .filter(|id| !tombstoned_user_ids.contains(id))
                .collect();
            field
                .vector_store
                .compact_rows(&deleted_field_rows, &live_field_ids)?;
            field.reverse_id_map = Self::build_reverse_id_map(&live_field_ids);
            field.id_map = live_field_ids.clone();
            if field.index.is_some() {
                field.index = None;
                field_indexes_to_rebuild
                    .push((field.config.name.clone(), field.config.index_mode.clone()));
            }
        }

        self.sparse_vectors.remove_ids(&tombstoned_user_ids)?;
        self.text_index.remove_ids(&tombstoned_user_ids, true)?;
        self.field_store
            .delete_map_entries(&tombstoned_user_ids.iter().copied().collect::<Vec<_>>());

        // Clear tombstone set and file
        {
            let mut set = self.tombstone.write();
            set.clear();
        }
        let tombstone_path = self.path.join("tombstone.bin");
        if tombstone_path.exists() {
            std::fs::remove_file(&tombstone_path)?;
        }

        // Rebuild index if one was present
        if let Some(ref mode) = self.index_mode.clone() {
            let mode = mode.clone();
            self.build_index(&mode)?;
        }
        for (field_name, index_mode) in field_indexes_to_rebuild {
            self.build_vector_field_index(&field_name, &index_mode)?;
        }

        Ok(n_removed)
    }

    /// Delete the entire collection from disk.
    pub fn delete(self) -> Result<()> {
        self.ensure_writable()?;
        if self.path.exists() {
            std::fs::remove_dir_all(&self.path)?;
        }
        Ok(())
    }

    /// Export live collection rows to a portable JSONL + binary-vector directory.
    pub fn export_to(&self, export_path: &Path) -> Result<()> {
        if export_path.exists() {
            return Err(LynseError::InvalidArgument(format!(
                "export target already exists: {}",
                export_path.display()
            )));
        }
        self.flush_pending_ingest()?;

        let tmp_path = Self::make_temp_sibling(export_path)?;
        if tmp_path.exists() {
            std::fs::remove_dir_all(&tmp_path)?;
        }

        let export_result = (|| -> Result<()> {
            use std::io::Write;

            std::fs::create_dir_all(&tmp_path)?;
            let dim = self.meta.dimension;
            let guard = self.vector_store.read_mmap()?;
            let tombstone = self.tombstone.read();

            let (row_offsets, user_ids, exported_count) = match guard.as_ref() {
                None => (Vec::new(), Vec::new(), 0usize),
                Some(fm) => {
                    let mut row_offsets = Vec::new();
                    let mut user_ids = Vec::new();
                    for row in 0..fm.len() {
                        let user_id = self.row_to_user_id(row as u64);
                        if tombstone.contains(&user_id) {
                            continue;
                        }
                        row_offsets.push(row as u64);
                        user_ids.push(user_id);
                    }
                    let count = row_offsets.len();
                    (row_offsets, user_ids, count)
                }
            };

            let fields = self.field_store.retrieve_many(&self.field_lookup_keys(&user_ids))?;

            let vectors_path = tmp_path.join(COLLECTION_EXPORT_VECTORS_FILE);
            let vectors_file = std::fs::File::create(&vectors_path)?;
            let mut vectors_writer = std::io::BufWriter::new(vectors_file);
            if let Some(fm) = guard.as_ref() {
                match fm.dtype() {
                    VectorDtype::F32 => {
                        let all_data = fm.as_slice();
                        for &row in &row_offsets {
                            let start = row as usize * dim;
                            let end = start + dim;
                            for value in &all_data[start..end] {
                                vectors_writer.write_all(&value.to_le_bytes())?;
                            }
                        }
                    }
                    VectorDtype::F16 => {
                        let all_data = fm.as_f16_bits_slice();
                        for &row in &row_offsets {
                            let start = row as usize * dim;
                            let end = start + dim;
                            for value in &all_data[start..end] {
                                vectors_writer.write_all(&value.to_le_bytes())?;
                            }
                        }
                    }
                }
            }
            vectors_writer.flush()?;

            let metadata_path = tmp_path.join(COLLECTION_EXPORT_METADATA_FILE);
            let metadata_file = std::fs::File::create(&metadata_path)?;
            let mut metadata_writer = std::io::BufWriter::new(metadata_file);
            for (id, field) in user_ids.iter().zip(fields.iter()) {
                let record = CollectionExportRecord {
                    id: *id,
                    external_id: Some(self.external_id_for_internal_id(*id)),
                    field: field.clone(),
                };
                serde_json::to_writer(&mut metadata_writer, &record)
                    .map_err(|e| LynseError::Serialization(e.to_string()))?;
                metadata_writer.write_all(b"\n")?;
            }
            metadata_writer.flush()?;

            let manifest = CollectionExportManifest::current(self, exported_count);
            let manifest_bytes = serde_json::to_vec_pretty(&manifest)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            Self::atomic_write_file(
                &tmp_path.join(COLLECTION_EXPORT_MANIFEST_FILE),
                &manifest_bytes,
            )?;

            drop(tombstone);
            drop(guard);

            Self::sync_path_recursively(&tmp_path)?;
            std::fs::rename(&tmp_path, export_path)?;
            if let Some(parent) = export_path.parent() {
                if let Ok(dir) = std::fs::File::open(parent) {
                    let _ = dir.sync_all();
                }
            }
            Ok(())
        })();

        if export_result.is_err() {
            let _ = std::fs::remove_dir_all(&tmp_path);
        }
        export_result
    }

    fn import_from_export(
        &mut self,
        export_path: &Path,
        manifest: &CollectionExportManifest,
    ) -> Result<()> {
        use std::io::{BufRead, Read};

        self.ensure_writable()?;
        Self::validate_export_manifest(manifest)?;
        let vector_dtype = VectorDtype::parse(&manifest.vector_dtype)?;
        if manifest.dimension != self.meta.dimension {
            return Err(LynseError::DimensionMismatch {
                expected: self.meta.dimension,
                got: manifest.dimension,
            });
        }
        if self.shape()?.0 != 0 {
            return Err(LynseError::InvalidArgument(
                "import target collection must be empty".to_string(),
            ));
        }

        let vectors_path = export_path.join(&manifest.vectors_file);
        let metadata_path = export_path.join(&manifest.metadata_file);
        let vectors_file = std::fs::File::open(&vectors_path)?;
        let metadata_file = std::fs::File::open(&metadata_path)?;
        let mut vectors_reader = std::io::BufReader::new(vectors_file);
        let metadata_reader = std::io::BufReader::new(metadata_file);

        let row_byte_len = manifest
            .dimension
            .checked_mul(vector_dtype.byte_width())
            .ok_or_else(|| {
                LynseError::InvalidArgument("export dimension is too large".to_string())
            })?;
        let mut row_bytes = vec![0u8; row_byte_len];
        let mut vectors = Vec::with_capacity(manifest.dimension * 1024);
        let mut ids = Vec::with_capacity(1024);
        let mut fields = Vec::with_capacity(1024);
        let mut external_entries: Vec<(u64, ExternalId)> = Vec::with_capacity(manifest.count);
        let mut row_count = 0usize;

        for line in metadata_reader.lines() {
            let line = line?;
            let record: CollectionExportRecord = serde_json::from_str(&line)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            vectors_reader.read_exact(&mut row_bytes)?;
            match vector_dtype {
                VectorDtype::F32 => {
                    for chunk in row_bytes.chunks_exact(4) {
                        vectors.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                    }
                }
                VectorDtype::F16 => {
                    for chunk in row_bytes.chunks_exact(2) {
                        vectors.push(f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
                    }
                }
            }
            ids.push(record.id);
            external_entries.push((
                record.id,
                record
                    .external_id
                    .unwrap_or(ExternalId::Int(record.id)),
            ));
            fields.push(record.field);
            row_count += 1;

            if ids.len() >= 1024 {
                self.add_items(&vectors, ids.len(), &ids, Some(&fields))?;
                vectors.clear();
                ids.clear();
                fields.clear();
            }
        }

        if !ids.is_empty() {
            self.add_items(&vectors, ids.len(), &ids, Some(&fields))?;
        }

        if row_count != manifest.count {
            return Err(LynseError::Storage(format!(
                "export metadata row count {} does not match manifest count {}",
                row_count, manifest.count
            )));
        }

        let mut extra = [0u8; 1];
        if vectors_reader.read(&mut extra)? != 0 {
            return Err(LynseError::Storage(
                "export vectors file contains more bytes than metadata rows describe".to_string(),
            ));
        }

        let mut external_to_internal = HashMap::with_capacity(external_entries.len());
        let mut internal_to_external = HashMap::with_capacity(external_entries.len());
        for (internal_id, external_id) in external_entries {
            external_id.validate()?;
            if !self.reverse_id_map.contains_key(&internal_id) {
                return Err(LynseError::Storage(format!(
                    "export metadata references missing internal id {}",
                    internal_id
                )));
            }
            if internal_to_external.insert(internal_id, external_id.clone()).is_some()
                || external_to_internal.insert(external_id, internal_id).is_some()
            {
                return Err(LynseError::InvalidArgument(
                    "export metadata contains duplicate external IDs".to_string(),
                ));
            }
        }
        self.internal_to_external_id = internal_to_external;
        self.external_to_internal_id = external_to_internal;
        self.write_external_id_map()?;

        self.checkpoint()
    }

    /// Create a filesystem snapshot of this collection.
    ///
    /// Writable handles checkpoint first. Read-only handles can snapshot only
    /// already-clean collections, which read-only open enforces.
    pub fn snapshot_to(&self, snapshot_path: &Path) -> Result<()> {
        if snapshot_path.exists() {
            return Err(LynseError::InvalidArgument(format!(
                "snapshot target already exists: {}",
                snapshot_path.display()
            )));
        }

        if self.read_only {
            if self.has_uncommitted_data() {
                return Err(LynseError::Storage(
                    "cannot snapshot read-only collection with uncommitted WAL data".to_string(),
                ));
            }
        } else {
            self.checkpoint()?;
        }

        let tmp_path = Self::make_temp_sibling(snapshot_path)?;
        if tmp_path.exists() {
            std::fs::remove_dir_all(&tmp_path)?;
        }

        let copy_result = (|| -> Result<()> {
            Self::copy_dir_for_snapshot(&self.path, &tmp_path)?;
            let manifest = CollectionSnapshotManifest::current(self);
            let manifest_bytes = serde_json::to_vec_pretty(&manifest)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            Self::atomic_write_file(&tmp_path.join(SNAPSHOT_MANIFEST_FILE), &manifest_bytes)?;
            Self::sync_path_recursively(&tmp_path)?;
            std::fs::rename(&tmp_path, snapshot_path)?;
            if let Some(parent) = snapshot_path.parent() {
                if let Ok(dir) = std::fs::File::open(parent) {
                    let _ = dir.sync_all();
                }
            }
            Ok(())
        })();

        if copy_result.is_err() {
            let _ = std::fs::remove_dir_all(&tmp_path);
        }
        copy_result
    }
}

/// Search result container, maps to Python's `Result` class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub ids: Vec<u64>,
    pub distances: Vec<f32>,
    pub fields: Vec<HashMap<String, serde_json::Value>>,
    pub index_mode: String,
    pub dimension: usize,
    pub k: usize,
}

/// Lightweight query profile used by explain/profile endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryProfile {
    pub query_kind: String,
    pub vector_field: String,
    pub index_path: String,
    pub total_vectors: u64,
    pub filter_expression: Option<String>,
    pub filter_matches: Option<usize>,
    pub scanned_vectors: usize,
    pub result_count: usize,
    pub filter_us: u64,
    pub search_us: u64,
    pub rerank_us: u64,
    pub total_us: u64,
}

fn elapsed_micros_u64(started: Instant) -> u64 {
    started.elapsed().as_micros().min(u64::MAX as u128) as u64
}

fn normalize_sparse_entries(entries: &[(u32, f32)]) -> Result<Vec<(u32, f32)>> {
    let mut merged: BTreeMap<u32, f32> = BTreeMap::new();
    for &(index, value) in entries {
        if !value.is_finite() {
            return Err(LynseError::InvalidArgument(
                "sparse vector values must be finite".to_string(),
            ));
        }
        if value == 0.0 {
            continue;
        }
        *merged.entry(index).or_insert(0.0) += value;
    }

    Ok(merged
        .into_iter()
        .filter(|(_, value)| *value != 0.0)
        .collect())
}

fn sparse_inner_product(query: &[(u32, f32)], vector: &[(u32, f32)]) -> f32 {
    let mut q = 0usize;
    let mut v = 0usize;
    let mut score = 0.0f32;

    while q < query.len() && v < vector.len() {
        let (q_index, q_value) = query[q];
        let (v_index, v_value) = vector[v];
        if q_index == v_index {
            score += q_value * v_value;
            q += 1;
            v += 1;
        } else if q_index < v_index {
            q += 1;
        } else {
            v += 1;
        }
    }

    score
}

enum TextFieldSelection<'a> {
    All,
    One(&'a str),
    Many(HashSet<String>),
}

impl<'a> TextFieldSelection<'a> {
    fn from_fields(text_fields: Option<&'a [String]>) -> Self {
        match text_fields {
            None => Self::All,
            Some([]) => Self::All,
            Some([field]) => Self::One(field.as_str()),
            Some(fields) => Self::Many(fields.iter().cloned().collect()),
        }
    }

    fn contains(&self, field: &str) -> bool {
        match self {
            Self::All => true,
            Self::One(selected) => *selected == field,
            Self::Many(selected) => selected.contains(field),
        }
    }

    fn cache_key(&self, term: &str) -> String {
        let mut key = String::with_capacity(term.len() + 24);
        key.push_str(term);
        key.push('\u{1f}');
        match self {
            Self::All => key.push('*'),
            Self::One(field) => key.push_str(field),
            Self::Many(fields) => {
                let mut fields: Vec<&str> = fields.iter().map(String::as_str).collect();
                fields.sort_unstable();
                for (idx, field) in fields.iter().enumerate() {
                    if idx > 0 {
                        key.push(',');
                    }
                    key.push_str(field);
                }
            }
        }
        key
    }
}

fn compare_score_desc_then_id(a: &(u64, f32), b: &(u64, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.0.cmp(&b.0))
}

fn sort_truncate_scores_desc(scored: &mut Vec<(u64, f32)>, limit: usize) {
    if limit == 0 {
        scored.clear();
        return;
    }
    if scored.len() > limit {
        scored.select_nth_unstable_by(limit, compare_score_desc_then_id);
        scored.truncate(limit);
    }
    scored.sort_by(compare_score_desc_then_id);
}

fn tokenize_text(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter_map(|token| {
            if token.is_empty() {
                None
            } else {
                Some(normalize_text_token(token))
            }
        })
        .collect()
}

fn normalize_text_token(token: &str) -> String {
    if token.is_ascii() && !token.as_bytes().iter().any(u8::is_ascii_uppercase) {
        token.to_owned()
    } else {
        token.to_lowercase()
    }
}

fn append_searchable_json_terms(value: &serde_json::Value, counts: &mut HashMap<String, u32>) {
    match value {
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
        serde_json::Value::String(v) => {
            append_text_terms(v, counts);
        }
        serde_json::Value::Array(values) => {
            for value in values {
                append_searchable_json_terms(value, counts);
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values() {
                append_searchable_json_terms(value, counts);
            }
        }
    }
}

fn append_text_terms(text: &str, counts: &mut HashMap<String, u32>) {
    for token in text.split(|c: char| !c.is_alphanumeric()) {
        if token.is_empty() {
            continue;
        }
        *counts.entry(normalize_text_token(token)).or_insert(0) += 1;
    }
}

fn text_doc_allowed(
    id: u64,
    allowed_ids: Option<&HashSet<u64>>,
    tombstones: &HashSet<u64>,
) -> bool {
    !tombstones.contains(&id)
        && allowed_ids
            .map(|allowed| allowed.contains(&id))
            .unwrap_or(true)
}

fn posting_document_frequency(
    posting: &HashMap<u64, HashMap<String, u32>>,
    field_filter: &TextFieldSelection<'_>,
    allowed_ids: Option<&HashSet<u64>>,
    tombstones: &HashSet<u64>,
) -> usize {
    if allowed_ids.is_none() && tombstones.is_empty() {
        match field_filter {
            TextFieldSelection::All => return posting.len(),
            TextFieldSelection::One(field) => {
                return posting
                    .values()
                    .filter(|tf_by_field| tf_by_field.get(*field).copied().unwrap_or(0) > 0)
                    .count();
            }
            TextFieldSelection::Many(_) => {}
        }
    }

    posting
        .iter()
        .filter(|(id, tf_by_field)| {
            text_doc_allowed(**id, allowed_ids, tombstones)
                && selected_term_frequency(tf_by_field, field_filter) > 0
        })
        .count()
}

fn selected_doc_length(
    lengths_by_field: &HashMap<String, usize>,
    field_filter: &TextFieldSelection<'_>,
) -> usize {
    match field_filter {
        TextFieldSelection::All => lengths_by_field.values().copied().sum(),
        TextFieldSelection::One(field) => lengths_by_field.get(*field).copied().unwrap_or(0),
        TextFieldSelection::Many(filter) => lengths_by_field
            .iter()
            .filter(|(field, _)| filter.contains(*field))
            .map(|(_, len)| *len)
            .sum(),
    }
}

fn selected_term_frequency(
    tf_by_field: &HashMap<String, u32>,
    field_filter: &TextFieldSelection<'_>,
) -> u32 {
    match field_filter {
        TextFieldSelection::All => tf_by_field.values().copied().sum(),
        TextFieldSelection::One(field) => tf_by_field.get(*field).copied().unwrap_or(0),
        TextFieldSelection::Many(filter) => tf_by_field
            .iter()
            .filter(|(field, _)| filter.contains(*field))
            .map(|(_, tf)| *tf)
            .sum(),
    }
}

fn fields_to_searchable_text(
    fields: &HashMap<String, serde_json::Value>,
    field_filter: &TextFieldSelection<'_>,
) -> String {
    let mut text = String::new();
    for (field, value) in fields {
        if !field_filter.contains(field) {
            continue;
        }
        append_searchable_json_text(value, &mut text);
        text.push(' ');
    }
    text
}

fn append_searchable_json_text(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
        serde_json::Value::String(v) => {
            out.push_str(v);
        }
        serde_json::Value::Array(values) => {
            for value in values {
                append_searchable_json_text(value, out);
                out.push(' ');
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values() {
                append_searchable_json_text(value, out);
                out.push(' ');
            }
        }
    }
}

fn normalize_vector_scores(distances: &[f32], metric: DistanceMetric) -> Vec<f32> {
    normalize_scores(distances, metric.is_ascending())
}

fn normalize_scores(scores: &[f32], ascending: bool) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for score in scores {
        if score.is_finite() {
            min = min.min(*score);
            max = max.max(*score);
        }
    }

    if !min.is_finite() || !max.is_finite() || (max - min).abs() <= f32::EPSILON {
        return vec![1.0; scores.len()];
    }

    scores
        .iter()
        .map(|score| {
            let normalized = ((*score - min) / (max - min)).clamp(0.0, 1.0);
            if ascending {
                1.0 - normalized
            } else {
                normalized
            }
        })
        .collect()
}

fn add_fused_scores(
    fused: &mut HashMap<u64, f32>,
    ids: &[u64],
    scores: &[f32],
    fusion: &str,
    weight: f32,
    rrf_k: f32,
) {
    let weight = weight.max(0.0);
    let use_weighted = fusion.eq_ignore_ascii_case("weighted");
    for (rank, id) in ids.iter().enumerate() {
        let contribution = if use_weighted {
            scores.get(rank).copied().unwrap_or(0.0) * weight
        } else {
            weight / (rrf_k.max(1.0) + rank as f32 + 1.0)
        };
        *fused.entry(*id).or_insert(0.0) += contribution;
    }
}

/// Database manager: manages multiple collections within a single database.
pub struct DatabaseEngine {
    root_path: PathBuf,
    collections: Arc<RwLock<HashMap<String, Arc<RwLock<Collection>>>>>,
    _lock: FileLock,
    read_only: bool,
}

impl DatabaseEngine {
    /// Open or create a database at the given root path.
    pub fn open(root_path: &Path) -> Result<Self> {
        Self::open_with_mode(root_path, OpenMode::ReadWrite)
    }

    /// Open an existing database in read-only mode.
    pub fn open_read_only(root_path: &Path) -> Result<Self> {
        Self::open_with_mode(root_path, OpenMode::ReadOnly)
    }

    fn open_with_mode(root_path: &Path, mode: OpenMode) -> Result<Self> {
        if mode == OpenMode::ReadWrite {
            std::fs::create_dir_all(root_path)?;
        } else if !root_path.exists() {
            return Err(LynseError::DatabaseNotFound(
                root_path.to_string_lossy().to_string(),
            ));
        }
        let lock_path = root_path.join(".database.lock");
        let lock = if mode == OpenMode::ReadWrite {
            FileLock::exclusive(&lock_path)?
        } else {
            FileLock::shared(&lock_path)?
        };

        let engine = Self {
            root_path: root_path.to_path_buf(),
            collections: Arc::new(RwLock::new(HashMap::new())),
            _lock: lock,
            read_only: mode == OpenMode::ReadOnly,
        };

        // Scan for existing collections
        engine.scan_collections()?;

        Ok(engine)
    }

    fn ensure_writable(&self) -> Result<()> {
        if self.read_only {
            return Err(LynseError::InvalidArgument(
                "database is opened read-only".to_string(),
            ));
        }
        Ok(())
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Scan root path for existing collections.
    fn scan_collections(&self) -> Result<()> {
        if let Ok(entries) = std::fs::read_dir(&self.root_path) {
            for entry in entries.flatten() {
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    let name = entry.file_name().to_string_lossy().to_string();
                    let info_path = entry.path().join("info.json");
                    if info_path.exists() {
                        // Try to load collection metadata
                        if let Ok(content) = std::fs::read_to_string(&info_path) {
                            if let Ok(info) = serde_json::from_str::<serde_json::Value>(&content) {
                                if let (Some(_rows), Some(dim)) = (
                                    info["total_shape"].get(0).and_then(|v| v.as_u64()),
                                    info["total_shape"].get(1).and_then(|v| v.as_u64()),
                                ) {
                                    let _ = self.get_or_open_collection(&name, dim as usize, 0);
                                }
                            }
                        }
                    } else {
                        let manifest_path = entry.path().join(STORAGE_MANIFEST_FILE);
                        if manifest_path.exists() {
                            if let Ok(content) = std::fs::read_to_string(&manifest_path) {
                                if let Ok(manifest) =
                                    serde_json::from_str::<StorageManifest>(&content)
                                {
                                    let _ = self.get_or_open_collection(
                                        &name,
                                        manifest.dimension,
                                        manifest.chunk_size,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Get an existing collection or open it.
    pub fn get_or_open_collection(
        &self,
        name: &str,
        dimension: usize,
        chunk_size: usize,
    ) -> Result<Arc<RwLock<Collection>>> {
        self.get_or_open_collection_with_dtype(name, dimension, chunk_size, None)
    }

    pub fn get_or_open_collection_with_dtype(
        &self,
        name: &str,
        dimension: usize,
        chunk_size: usize,
        vector_dtype: Option<VectorDtype>,
    ) -> Result<Arc<RwLock<Collection>>> {
        let mut collections = self.collections.write();

        if let Some(coll) = collections.get(name) {
            if let Some(dtype) = vector_dtype {
                let existing = coll.read().vector_dtype;
                if existing != dtype {
                    return Err(LynseError::InvalidArgument(format!(
                        "collection vector dtype {} does not match requested dtype {}",
                        existing.storage_name(),
                        dtype.storage_name()
                    )));
                }
            }
            return Ok(Arc::clone(coll));
        }

        let collection = if self.read_only {
            Collection::open_read_only(&self.root_path, name, dimension, chunk_size)?
        } else if let Some(dtype) = vector_dtype {
            Collection::open_with_dtype(&self.root_path, name, dimension, chunk_size, dtype)?
        } else {
            Collection::open(&self.root_path, name, dimension, chunk_size)?
        };
        let arc = Arc::new(RwLock::new(collection));
        collections.insert(name.to_string(), Arc::clone(&arc));
        Ok(arc)
    }

    /// Create a new collection (error if exists).
    pub fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        chunk_size: usize,
    ) -> Result<Arc<RwLock<Collection>>> {
        self.create_collection_with_dtype(name, dimension, chunk_size, VectorDtype::F32)
    }

    pub fn create_collection_with_dtype(
        &self,
        name: &str,
        dimension: usize,
        chunk_size: usize,
        vector_dtype: VectorDtype,
    ) -> Result<Arc<RwLock<Collection>>> {
        self.ensure_writable()?;
        let coll_path = self.root_path.join(name);
        if coll_path.exists() {
            return Err(LynseError::CollectionAlreadyExists(name.to_string()));
        }
        self.get_or_open_collection_with_dtype(name, dimension, chunk_size, Some(vector_dtype))
    }

    /// Drop a collection.
    pub fn drop_collection(&self, name: &str) -> Result<()> {
        self.ensure_writable()?;
        let mut collections = self.collections.write();
        let fallback_path = self.root_path.join(name);

        let path = if let Some(coll) = collections.remove(name) {
            let mut coll_guard = coll.write();
            let path = coll_guard.path.clone();
            coll_guard.close()?;
            drop(coll_guard);
            drop(coll);
            path
        } else {
            fallback_path
        };

        if path.exists() {
            std::fs::remove_dir_all(&path)?;
        }
        Ok(())
    }

    /// List all collection names.
    pub fn list_collections(&self) -> Vec<String> {
        self.collections.read().keys().cloned().collect()
    }

    /// Number of currently open collection handles in this database engine.
    pub fn open_collection_count(&self) -> usize {
        self.collections.read().len()
    }

    /// Checkpoint all collections currently open in this engine.
    pub fn checkpoint_open_collections(&self) -> Result<()> {
        if self.read_only {
            return Ok(());
        }

        let collections: Vec<Arc<RwLock<Collection>>> =
            self.collections.read().values().cloned().collect();
        for coll in collections {
            let coll = coll.read();
            if coll.has_uncommitted_data() {
                coll.checkpoint()?;
            }
        }
        Ok(())
    }

    fn external_collection_handle(&self) -> Option<String> {
        let collections = self.collections.read();
        collections.iter().find_map(|(name, coll)| {
            if Arc::strong_count(coll) > 1 {
                Some(name.clone())
            } else {
                None
            }
        })
    }

    fn close_open_collections(&self) -> Result<()> {
        let collections: Vec<Arc<RwLock<Collection>>> =
            self.collections.read().values().cloned().collect();
        for coll in collections {
            coll.write().close()?;
        }
        Ok(())
    }

    /// Create a consistent filesystem snapshot of this database directory.
    ///
    /// All collection read locks are held from checkpoint through directory
    /// copy. Searches can continue because they share read locks, while writes
    /// wait until the snapshot has been atomically moved into place.
    pub fn snapshot_to(&self, database_name: &str, snapshot_path: &Path) -> Result<()> {
        if snapshot_path.exists() {
            return Err(LynseError::InvalidArgument(format!(
                "snapshot target already exists: {}",
                snapshot_path.display()
            )));
        }

        let tmp_path = Collection::make_temp_sibling(snapshot_path)?;
        if tmp_path.exists() {
            std::fs::remove_dir_all(&tmp_path)?;
        }

        let collections = self.collections.read();
        let collection_guards: Vec<_> = collections.values().map(|coll| coll.read()).collect();
        if !self.read_only {
            for coll in &collection_guards {
                coll.checkpoint()?;
            }
        }
        let collection_names: Vec<String> = collections.keys().cloned().collect();

        let copy_result = (|| -> Result<()> {
            Collection::copy_dir_for_snapshot(&self.root_path, &tmp_path)?;
            let manifest = DatabaseSnapshotManifest::current(database_name, collection_names);
            let manifest_bytes = serde_json::to_vec_pretty(&manifest)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            Collection::atomic_write_file(
                &tmp_path.join(DATABASE_SNAPSHOT_MANIFEST_FILE),
                &manifest_bytes,
            )?;
            Collection::sync_path_recursively(&tmp_path)?;
            std::fs::rename(&tmp_path, snapshot_path)?;
            if let Some(parent) = snapshot_path.parent() {
                if let Ok(dir) = std::fs::File::open(parent) {
                    let _ = dir.sync_all();
                }
            }
            Ok(())
        })();

        drop(collection_guards);
        drop(collections);

        if copy_result.is_err() {
            let _ = std::fs::remove_dir_all(&tmp_path);
        }
        copy_result
    }

    /// Create a snapshot for a collection in this database.
    pub fn snapshot_collection(&self, name: &str, snapshot_path: &Path) -> Result<()> {
        let coll = self.get_or_open_collection(name, 0, 0)?;
        let result = coll.read().snapshot_to(snapshot_path);
        result
    }

    /// Export a collection to portable JSONL metadata plus binary vectors.
    pub fn export_collection(&self, name: &str, export_path: &Path) -> Result<()> {
        let coll = self.get_or_open_collection(name, 0, 0)?;
        let result = coll.read().export_to(export_path);
        result
    }

    /// Restore a collection from a snapshot directory.
    pub fn restore_collection_from_snapshot(
        &self,
        name: &str,
        snapshot_path: &Path,
        overwrite: bool,
    ) -> Result<CollectionConfig> {
        self.ensure_writable()?;
        if !snapshot_path.exists() {
            return Err(LynseError::InvalidArgument(format!(
                "snapshot path does not exist: {}",
                snapshot_path.display()
            )));
        }

        let storage_manifest = Collection::read_storage_manifest_from_dir(snapshot_path)?;
        let snapshot_manifest = Collection::read_snapshot_manifest(snapshot_path).ok();
        if let Some(snapshot_manifest) = snapshot_manifest.as_ref() {
            if snapshot_manifest.storage_format != STORAGE_FORMAT_NAME {
                return Err(LynseError::Storage(format!(
                    "unsupported snapshot storage format '{}'",
                    snapshot_manifest.storage_format
                )));
            }
            if snapshot_manifest.storage_version > STORAGE_FORMAT_VERSION {
                return Err(LynseError::Storage(format!(
                    "snapshot uses storage format version {}, but this binary supports up to {}",
                    snapshot_manifest.storage_version, STORAGE_FORMAT_VERSION
                )));
            }
        }

        let target_path = self.root_path.join(name);
        if target_path.exists() && !overwrite {
            return Err(LynseError::CollectionAlreadyExists(name.to_string()));
        }

        let existing = {
            let mut collections = self.collections.write();
            collections.remove(name)
        };

        if let Some(coll) = existing {
            if Arc::strong_count(&coll) > 1 {
                let mut collections = self.collections.write();
                collections.insert(name.to_string(), coll);
                return Err(LynseError::Storage(format!(
                    "collection '{}' is still referenced; close active handles before restore",
                    name
                )));
            }
            coll.write().close()?;
        }

        if target_path.exists() {
            std::fs::remove_dir_all(&target_path)?;
        }

        let tmp_path = Collection::make_temp_sibling(&target_path)?;
        if tmp_path.exists() {
            std::fs::remove_dir_all(&tmp_path)?;
        }

        let restore_result = (|| -> Result<()> {
            Collection::copy_dir_for_snapshot(snapshot_path, &tmp_path)?;

            if storage_manifest.collection_name != name {
                let mut remapped = storage_manifest.clone();
                remapped.collection_name = name.to_string();
                let manifest_bytes = serde_json::to_vec_pretty(&remapped)
                    .map_err(|e| LynseError::Serialization(e.to_string()))?;
                Collection::atomic_write_file(
                    &tmp_path.join(STORAGE_MANIFEST_FILE),
                    &manifest_bytes,
                )?;
            }

            Collection::sync_path_recursively(&tmp_path)?;
            std::fs::rename(&tmp_path, &target_path)?;
            if let Ok(dir) = std::fs::File::open(&self.root_path) {
                let _ = dir.sync_all();
            }
            Ok(())
        })();

        if let Err(err) = restore_result {
            let _ = std::fs::remove_dir_all(&tmp_path);
            return Err(err);
        }

        let restored = Collection::open(
            &self.root_path,
            name,
            storage_manifest.dimension,
            storage_manifest.chunk_size,
        )?;
        self.collections
            .write()
            .insert(name.to_string(), Arc::new(RwLock::new(restored)));

        Ok(CollectionConfig {
            dim: storage_manifest.dimension,
            chunk_size: storage_manifest.chunk_size,
            description: None,
            dtypes: VectorDtype::parse(&storage_manifest.vector_dtype)?
                .storage_name()
                .to_string(),
        })
    }

    /// Import a collection from portable JSONL metadata plus binary vectors.
    pub fn import_collection_from_export(
        &self,
        name: &str,
        export_path: &Path,
        overwrite: bool,
    ) -> Result<CollectionConfig> {
        self.ensure_writable()?;
        if !export_path.exists() {
            return Err(LynseError::InvalidArgument(format!(
                "export path does not exist: {}",
                export_path.display()
            )));
        }

        let export_manifest = Collection::read_export_manifest(export_path)?;
        Collection::validate_export_manifest(&export_manifest)?;

        let target_path = self.root_path.join(name);
        if target_path.exists() && !overwrite {
            return Err(LynseError::CollectionAlreadyExists(name.to_string()));
        }

        let existing = {
            let mut collections = self.collections.write();
            collections.remove(name)
        };

        if let Some(coll) = existing {
            if Arc::strong_count(&coll) > 1 {
                let mut collections = self.collections.write();
                collections.insert(name.to_string(), coll);
                return Err(LynseError::Storage(format!(
                    "collection '{}' is still referenced; close active handles before import",
                    name
                )));
            }
            coll.write().close()?;
        }

        if target_path.exists() {
            std::fs::remove_dir_all(&target_path)?;
        }

        let export_dtype = VectorDtype::parse(&export_manifest.vector_dtype)?;
        let mut collection = Collection::open_with_dtype(
            &self.root_path,
            name,
            export_manifest.dimension,
            export_manifest.chunk_size,
            export_dtype,
        )?;
        let import_result = collection.import_from_export(export_path, &export_manifest);
        if let Err(err) = import_result {
            drop(collection);
            let _ = std::fs::remove_dir_all(&target_path);
            return Err(err);
        }

        self.collections
            .write()
            .insert(name.to_string(), Arc::new(RwLock::new(collection)));

        Ok(CollectionConfig {
            dim: export_manifest.dimension,
            chunk_size: export_manifest.chunk_size,
            description: None,
            dtypes: export_dtype.storage_name().to_string(),
        })
    }

    /// Check if a collection exists.
    pub fn has_collection(&self, name: &str) -> bool {
        self.collections.read().contains_key(name) || self.root_path.join(name).exists()
    }

    /// Get the root path.
    pub fn root_path(&self) -> &Path {
        &self.root_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn duplicate_insert_ids_are_rejected() {
        let tmp = TempDir::new().unwrap();
        let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();

        let vectors = vec![1.0, 0.0, 0.0, 0.0];
        coll.add_items(&vectors, 1, &[42], None).unwrap();

        let err = coll.add_items(&vectors, 1, &[42], None).unwrap_err();
        assert!(err.to_string().contains("already exists"));
    }

    #[test]
    fn external_ids_are_allocated_and_persisted() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll = Collection::open(tmp.path(), "col", 3, 100).unwrap();
            let ids = vec![ExternalId::String("doc-a".into()), ExternalId::Int(42)];
            let vectors = vec![
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0,
            ];
            let internal_ids = coll.add_records(&vectors, 2, &ids, None).unwrap();
            assert_eq!(internal_ids, vec![0, 1]);
            assert_eq!(
                coll.internal_ids_for_external_ids(&ids).unwrap(),
                internal_ids
            );
            coll.checkpoint().unwrap();
            let collection_path = tmp.path().join("col");
            assert!(collection_path.join(EXTERNAL_ID_MAP_BIN_FILE).exists());
            assert!(!collection_path.join(EXTERNAL_ID_MAP_FILE).exists());
        }

        let coll = Collection::open(tmp.path(), "col", 3, 100).unwrap();
        assert!(coll.is_external_id_exists(&ExternalId::String("doc-a".into())));
        assert!(coll.is_external_id_exists(&ExternalId::Int(42)));

        let result = coll
            .search(&[1.0, 0.0, 0.0], 2, None, 10, false, 1e-4)
            .unwrap();
        assert_eq!(
            coll.external_ids_for_internal_ids(&result.ids),
            vec![ExternalId::String("doc-a".into()), ExternalId::Int(42)]
        );
    }

    #[test]
    fn legacy_json_external_id_map_still_loads() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll = Collection::open(tmp.path(), "col", 3, 100).unwrap();
            coll.add_items(&[1.0, 0.0, 0.0], 1, &[0], None).unwrap();
            coll.checkpoint().unwrap();
        }

        let legacy = ExternalIdMapFile {
            version: EXTERNAL_ID_MAP_VERSION,
            next_internal_id: 1,
            entries: vec![ExternalIdEntry {
                internal_id: 0,
                external_id: ExternalId::String("legacy-doc".into()),
            }],
        };
        let collection_path = tmp.path().join("col");
        let _ = std::fs::remove_file(collection_path.join(EXTERNAL_ID_MAP_BIN_FILE));
        let _ = std::fs::remove_file(collection_path.join(EXTERNAL_ID_MAP_DELTA_BIN_FILE));
        std::fs::write(
            collection_path.join(EXTERNAL_ID_MAP_FILE),
            serde_json::to_vec(&legacy).unwrap(),
        )
        .unwrap();

        let coll = Collection::open(tmp.path(), "col", 3, 100).unwrap();
        let external_id = ExternalId::String("legacy-doc".into());
        assert!(coll.is_external_id_exists(&external_id));
        assert_eq!(
            coll.internal_ids_for_external_ids(&[external_id.clone()])
                .unwrap(),
            vec![0]
        );
        assert_eq!(coll.external_ids_for_internal_ids(&[0]), vec![external_id]);
    }

    #[test]
    fn f16_collection_persists_dtype_and_searches_after_reopen() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll =
                Collection::open_with_dtype(tmp.path(), "col", 4, 100, VectorDtype::F16).unwrap();
            coll.add_items(
                &[
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, //
                    0.5, 0.5, 0.0, 0.0,
                ],
                3,
                &[10, 11, 12],
                None,
            )
            .unwrap();
            coll.checkpoint().unwrap();
            assert_eq!(coll.vector_dtype(), VectorDtype::F16);
            assert_eq!(
                std::fs::metadata(coll.vector_store.vectors_path())
                    .unwrap()
                    .len(),
                24
            );
        }

        let coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        assert_eq!(coll.vector_dtype(), VectorDtype::F16);
        assert_eq!(coll.meta.dtypes, "float16");

        let result = coll
            .search(&[1.0, 0.0, 0.0, 0.0], 2, None, 10, false, 1e-4)
            .unwrap();
        assert_eq!(result.ids, vec![10, 12]);
        assert!((result.distances[0] - 1.0).abs() < 1e-6);
        assert!((result.distances[1] - 0.5).abs() < 1e-6);
        drop(coll);

        let err = match Collection::open_with_dtype(tmp.path(), "col", 4, 100, VectorDtype::F32) {
            Ok(_) => panic!("opening an f16 collection as f32 should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("does not match requested dtype"));
    }

    #[test]
    fn f16_encoded_add_writes_half_width_wal_and_vectors() {
        let tmp = TempDir::new().unwrap();
        let mut coll =
            Collection::open_with_dtype(tmp.path(), "col", 4, 100, VectorDtype::F16).unwrap();
        let vectors = vec![1.0, 0.5, 0.0, -0.5, 2.0, 1.5, 1.0, 0.0];
        let encoded = encode_f32_slice_as_le_bytes(&vectors, VectorDtype::F16);

        coll.add_items_encoded_vectors(&encoded, VectorDtype::F16, 2, &[10, 11], None)
            .unwrap();
        coll.flush().unwrap();

        assert_eq!(
            std::fs::metadata(coll.vector_store.vectors_path())
                .unwrap()
                .len(),
            16
        );
        let segments = coll.wal.get_segments().unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].dtype, VectorDtype::F16);
        assert_eq!(segments[0].encoded_data.len(), vectors.len() * 2);
        assert_eq!(segments[0].data, vectors);
    }

    #[test]
    fn f16_collection_batch_search_reuses_decoded_candidates() {
        let tmp = TempDir::new().unwrap();
        let mut coll =
            Collection::open_with_dtype(tmp.path(), "col", 4, 100, VectorDtype::F16).unwrap();
        coll.add_items(
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.5, 0.5, 0.0, 0.0,
            ],
            3,
            &[10, 11, 12],
            None,
        )
        .unwrap();

        let results = coll
            .batch_search(
                &[
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0,
                ],
                2,
                2,
                None,
                10,
            )
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].ids, vec![10, 12]);
        assert_eq!(results[1].ids, vec![11, 12]);
    }

    #[test]
    fn f16_named_vector_field_searches() {
        let tmp = TempDir::new().unwrap();
        let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        coll.add_items(
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0,
            ],
            2,
            &[1, 2],
            None,
        )
        .unwrap();
        let config = coll
            .create_vector_field_with_dtype("image", 2, Some("ip"), None, Some("float16"))
            .unwrap();
        assert_eq!(config.dtypes, "float16");
        coll.add_named_vectors("image", &[1.0, 0.0, 0.0, 1.0], 2, &[1, 2])
            .unwrap();

        let result = coll
            .search_vector_field_with_options("image", &[0.0, 1.0], 1, None, false, 1e-4)
            .unwrap();
        assert_eq!(result.ids, vec![2]);
        assert!((result.distances[0] - 1.0).abs() < 1e-6);

        let field = coll.named_vector_fields.get("image").unwrap();
        assert_eq!(field.vector_store.dtype(), VectorDtype::F16);
        assert_eq!(
            std::fs::metadata(field.vector_store.vectors_path())
                .unwrap()
                .len(),
            8
        );
    }

    #[test]
    fn collection_writer_lock_rejects_second_open_until_drop() {
        let tmp = TempDir::new().unwrap();
        let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();

        let err = match Collection::open(tmp.path(), "col", 4, 100) {
            Ok(_) => panic!("second writer should not acquire the collection lock"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("already open by another writer"));

        coll.close().unwrap();
        Collection::open(tmp.path(), "col", 4, 100).unwrap();
    }

    #[test]
    fn database_engine_writer_lock_rejects_second_open_until_drop() {
        let tmp = TempDir::new().unwrap();
        let engine = DatabaseEngine::open(tmp.path()).unwrap();

        let err = match DatabaseEngine::open(tmp.path()) {
            Ok(_) => panic!("second database engine should not acquire the database lock"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("already open by another writer"));

        drop(engine);
        DatabaseEngine::open(tmp.path()).unwrap();
    }

    #[test]
    fn database_manager_writer_lock_rejects_second_open_until_drop() {
        let tmp = TempDir::new().unwrap();
        let manager = DatabaseManager::new(tmp.path()).unwrap();

        let err = match DatabaseManager::new(tmp.path()) {
            Ok(_) => panic!("second database manager should not acquire the manager lock"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("already open by another writer"));

        drop(manager);
        DatabaseManager::new(tmp.path()).unwrap();
    }

    #[test]
    fn database_engine_read_only_rejects_writes() {
        let tmp = TempDir::new().unwrap();
        {
            let engine = DatabaseEngine::open(tmp.path()).unwrap();
            engine.create_collection("col", 4, 100).unwrap();
        }

        let ro = DatabaseEngine::open_read_only(tmp.path()).unwrap();
        let ro2 = DatabaseEngine::open_read_only(tmp.path()).unwrap();
        assert!(ro.is_read_only());
        assert!(ro2.is_read_only());
        assert!(ro.has_collection("col"));

        let err = match ro.create_collection("other", 4, 100) {
            Ok(_) => panic!("read-only database engine should reject create_collection"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("read-only"));
    }

    #[test]
    fn database_manager_read_only_rejects_writes() {
        let tmp = TempDir::new().unwrap();
        {
            let manager = DatabaseManager::new(tmp.path()).unwrap();
            manager.create_database("db").unwrap();
            manager
                .require_collection("db", "col", 4, 100, false, None)
                .unwrap();
        }

        let ro = DatabaseManager::new_read_only(tmp.path()).unwrap();
        let ro2 = DatabaseManager::new_read_only(tmp.path()).unwrap();
        assert!(ro.is_read_only());
        assert!(ro2.is_read_only());
        assert!(ro.database_exists("db"));
        assert_eq!(ro.show_collections("db").unwrap(), vec!["col".to_string()]);

        let err = ro.create_database("other").unwrap_err();
        assert!(err.to_string().contains("read-only"));
    }

    #[test]
    fn manager_checkpoint_open_collections_clears_wal() {
        let tmp = TempDir::new().unwrap();
        let manager = DatabaseManager::new(tmp.path()).unwrap();
        manager.create_database("db").unwrap();
        manager
            .require_collection("db", "col", 4, 100, false, None)
            .unwrap();

        manager
            .with_database("db", |engine| {
                let coll = engine.get_or_open_collection("col", 4, 100)?;
                let mut coll = coll.write();
                coll.add_items(&[1.0, 0.0, 0.0, 0.0], 1, &[7], None)?;
                assert!(coll.has_uncommitted_data());
                Ok(())
            })
            .unwrap();

        manager.checkpoint_open_collections().unwrap();

        manager
            .with_database("db", |engine| {
                let coll = engine.get_or_open_collection("col", 4, 100)?;
                assert!(!coll.read().has_uncommitted_data());
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn collection_snapshot_restore_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let snapshot_path = tmp.path().join("snapshots").join("col_snapshot");
        let db_path = tmp.path().join("db");

        let engine = DatabaseEngine::open(&db_path).unwrap();
        {
            let coll = engine.create_collection("col", 4, 100).unwrap();
            let fields = vec![HashMap::from([(
                "tag".to_string(),
                serde_json::json!("snapshot"),
            )])];
            coll.write()
                .add_items(&[1.0, 0.0, 0.0, 0.0], 1, &[42], Some(&fields))
                .unwrap();
        }

        engine.snapshot_collection("col", &snapshot_path).unwrap();
        engine.drop_collection("col").unwrap();
        engine
            .restore_collection_from_snapshot("col", &snapshot_path, false)
            .unwrap();

        let coll = engine.get_or_open_collection("col", 4, 100).unwrap();
        let (vectors, fields) = coll.read().read_vectors_by_ids(&[42]).unwrap();
        assert_eq!(vectors, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(fields[0].get("tag"), Some(&serde_json::json!("snapshot")));
        assert!(!snapshot_path.join(".writer.lock").exists());
        assert!(snapshot_path.join(SNAPSHOT_MANIFEST_FILE).exists());
    }

    #[test]
    fn manager_restore_collection_updates_config() {
        let tmp = TempDir::new().unwrap();
        let snapshot_path = tmp.path().join("snapshots").join("col_snapshot");
        let manager = DatabaseManager::new(tmp.path()).unwrap();
        manager.create_database("db").unwrap();
        manager
            .require_collection("db", "col", 4, 100, false, Some("old"))
            .unwrap();
        manager
            .with_database("db", |engine| {
                let coll = engine.get_or_open_collection("col", 4, 100)?;
                coll.write()
                    .add_items(&[1.0, 0.0, 0.0, 0.0], 1, &[7], None)?;
                Ok(())
            })
            .unwrap();
        manager
            .snapshot_collection("db", "col", &snapshot_path)
            .unwrap();
        manager.drop_collection("db", "col").unwrap();

        manager
            .restore_collection_from_snapshot("db", "restored", &snapshot_path, false)
            .unwrap();

        let configs = manager.get_collection_configs("db").unwrap();
        let config = configs.get("restored").unwrap();
        assert_eq!(config.dim, 4);
        assert_eq!(config.chunk_size, 100);
        assert!(manager.collection_exists("db", "restored").unwrap());
    }

    #[test]
    fn manager_export_import_collection_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let export_path = tmp.path().join("exports").join("col_export");
        let manager = DatabaseManager::new(tmp.path()).unwrap();
        manager.create_database("db").unwrap();
        manager
            .require_collection("db", "col", 4, 100, false, Some("source"))
            .unwrap();
        manager
            .with_database("db", |engine| {
                let coll = engine.get_or_open_collection("col", 4, 100)?;
                let fields = vec![
                    HashMap::from([("tag".to_string(), serde_json::json!("keep"))]),
                    HashMap::from([("tag".to_string(), serde_json::json!("drop"))]),
                ];
                let mut coll = coll.write();
                coll.add_items(
                    &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    2,
                    &[10, 20],
                    Some(&fields),
                )?;
                coll.delete_items(&[20])?;
                Ok(())
            })
            .unwrap();

        manager
            .export_collection("db", "col", &export_path)
            .unwrap();
        assert!(export_path.join(COLLECTION_EXPORT_MANIFEST_FILE).exists());
        assert!(export_path.join(COLLECTION_EXPORT_VECTORS_FILE).exists());
        assert!(export_path.join(COLLECTION_EXPORT_METADATA_FILE).exists());

        let manifest = Collection::read_export_manifest(&export_path).unwrap();
        assert_eq!(manifest.dimension, 4);
        assert_eq!(manifest.count, 1);
        assert_eq!(
            std::fs::metadata(export_path.join(COLLECTION_EXPORT_VECTORS_FILE))
                .unwrap()
                .len(),
            16
        );

        manager
            .import_collection_from_export("db", "imported", &export_path, false)
            .unwrap();
        assert!(manager.collection_exists("db", "imported").unwrap());

        let configs = manager.get_collection_configs("db").unwrap();
        assert_eq!(configs.get("imported").unwrap().dim, 4);

        manager
            .with_database("db", |engine| {
                let coll = engine.get_or_open_collection("imported", 4, 100)?;
                let coll = coll.read();
                assert_eq!(coll.shape()?, (1, 4));
                assert!(coll.is_id_exists(10));
                assert!(!coll.is_id_exists(20));

                let (vectors, fields) = coll.read_vectors_by_ids(&[10])?;
                assert_eq!(vectors, vec![1.0, 0.0, 0.0, 0.0]);
                assert_eq!(fields[0].get("tag"), Some(&serde_json::json!("keep")));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn manager_export_import_f16_collection_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let export_path = tmp.path().join("exports").join("col_f16_export");
        let manager = DatabaseManager::new(tmp.path()).unwrap();
        manager.create_database("db").unwrap();
        manager
            .require_collection_with_dtype(
                "db",
                "col",
                4,
                100,
                false,
                Some("source"),
                Some("float16"),
            )
            .unwrap();
        manager
            .with_database("db", |engine| {
                let coll = engine.get_or_open_collection_with_dtype(
                    "col",
                    4,
                    100,
                    Some(VectorDtype::F16),
                )?;
                let fields = vec![HashMap::from([(
                    "tag".to_string(),
                    serde_json::json!("keep"),
                )])];
                coll.write()
                    .add_items(&[1.0, 0.5, 0.0, -0.5], 1, &[10], Some(&fields))?;
                Ok(())
            })
            .unwrap();

        manager
            .export_collection("db", "col", &export_path)
            .unwrap();

        let manifest = Collection::read_export_manifest(&export_path).unwrap();
        assert_eq!(manifest.vector_dtype, "float16");
        assert_eq!(manifest.dimension, 4);
        assert_eq!(manifest.count, 1);
        assert_eq!(
            std::fs::metadata(export_path.join(COLLECTION_EXPORT_VECTORS_FILE))
                .unwrap()
                .len(),
            8
        );

        manager
            .import_collection_from_export("db", "imported_f16", &export_path, false)
            .unwrap();
        let configs = manager.get_collection_configs("db").unwrap();
        assert_eq!(configs.get("imported_f16").unwrap().dtypes, "float16");

        manager
            .with_database("db", |engine| {
                let coll = engine.get_or_open_collection_with_dtype(
                    "imported_f16",
                    4,
                    100,
                    Some(VectorDtype::F16),
                )?;
                let coll = coll.read();
                assert_eq!(coll.vector_dtype(), VectorDtype::F16);
                assert_eq!(coll.shape()?, (1, 4));
                let (vectors, fields) = coll.read_vectors_by_ids(&[10])?;
                assert_eq!(vectors, vec![1.0, 0.5, 0.0, -0.5]);
                assert_eq!(fields[0].get("tag"), Some(&serde_json::json!("keep")));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn manager_snapshot_restore_database_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let snapshot_path = tmp.path().join("snapshots").join("db_snapshot");
        let manager = DatabaseManager::new(tmp.path()).unwrap();
        manager.create_database("db").unwrap();
        manager
            .require_collection("db", "col", 4, 100, false, Some("primary"))
            .unwrap();
        manager
            .require_collection("db", "other", 4, 100, false, Some("secondary"))
            .unwrap();

        manager
            .with_database("db", |engine| {
                let fields = vec![HashMap::from([(
                    "scope".to_string(),
                    serde_json::json!("database"),
                )])];
                let col = engine.get_or_open_collection("col", 4, 100)?;
                col.write()
                    .add_items(&[1.0, 0.0, 0.0, 0.0], 1, &[11], Some(&fields))?;

                let other = engine.get_or_open_collection("other", 4, 100)?;
                other
                    .write()
                    .add_items(&[0.0, 1.0, 0.0, 0.0], 1, &[22], None)?;
                Ok(())
            })
            .unwrap();

        manager.snapshot_database("db", &snapshot_path).unwrap();
        assert!(snapshot_path.join(DATABASE_SNAPSHOT_MANIFEST_FILE).exists());
        assert!(snapshot_path.join("collections.json").exists());
        assert!(!snapshot_path.join(".database.lock").exists());
        assert!(!snapshot_path.join("col").join(".writer.lock").exists());

        manager.drop_database("db").unwrap();
        assert!(!manager.database_exists("db"));

        manager
            .restore_database_from_snapshot("db", &snapshot_path, false)
            .unwrap();
        assert!(manager.database_exists("db"));
        assert!(manager.list_databases().contains(&"db".to_string()));
        assert!(manager.collection_exists("db", "col").unwrap());
        assert!(manager.collection_exists("db", "other").unwrap());

        let configs = manager.get_collection_configs("db").unwrap();
        assert_eq!(
            configs
                .get("col")
                .and_then(|cfg| cfg.description.as_deref()),
            Some("primary")
        );

        manager
            .with_database("db", |engine| {
                let col = engine.get_or_open_collection("col", 4, 100)?;
                let (vectors, fields) = col.read().read_vectors_by_ids(&[11])?;
                assert_eq!(vectors, vec![1.0, 0.0, 0.0, 0.0]);
                assert_eq!(fields[0].get("scope"), Some(&serde_json::json!("database")));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn manager_restore_database_under_new_name_registers_database() {
        let tmp = TempDir::new().unwrap();
        let snapshot_path = tmp.path().join("snapshots").join("db_snapshot");
        let manager = DatabaseManager::new(tmp.path()).unwrap();
        manager.create_database("db").unwrap();
        manager
            .require_collection("db", "col", 4, 100, false, None)
            .unwrap();

        let original_fingerprint =
            std::fs::read_to_string(tmp.path().join("db/.fingerprint")).unwrap();
        manager.snapshot_database("db", &snapshot_path).unwrap();
        manager
            .restore_database_from_snapshot("clone", &snapshot_path, false)
            .unwrap();

        assert!(manager.database_exists("db"));
        assert!(manager.database_exists("clone"));
        assert!(manager.collection_exists("clone", "col").unwrap());
        assert!(manager.list_databases().contains(&"clone".to_string()));

        let clone_fingerprint =
            std::fs::read_to_string(tmp.path().join("clone/.fingerprint")).unwrap();
        assert_ne!(original_fingerprint, clone_fingerprint);
    }

    #[test]
    fn database_snapshot_allows_readers_and_blocks_writers() {
        struct ResetSnapshotDelay;
        impl Drop for ResetSnapshotDelay {
            fn drop(&mut self) {
                SNAPSHOT_COPY_DELAY_MS.store(0, Ordering::Relaxed);
            }
        }

        SNAPSHOT_COPY_DELAY_MS.store(100, Ordering::Relaxed);
        let _reset = ResetSnapshotDelay;

        let tmp = TempDir::new().unwrap();
        let snapshot_path = tmp.path().join("snapshots").join("db_snapshot");
        let manager = Arc::new(DatabaseManager::new(tmp.path()).unwrap());
        manager.create_database("db").unwrap();
        manager
            .require_collection("db", "col", 4, 100, false, None)
            .unwrap();
        let coll_arc = manager
            .with_database("db", |engine| {
                let coll = engine.get_or_open_collection("col", 4, 100)?;
                coll.write()
                    .add_items(&[1.0, 0.0, 0.0, 0.0], 1, &[1], None)?;
                Ok(coll)
            })
            .unwrap();

        let snapshot_for_thread = snapshot_path.clone();
        let manager_for_thread = Arc::clone(&manager);
        let handle = std::thread::spawn(move || {
            manager_for_thread
                .snapshot_database("db", &snapshot_for_thread)
                .unwrap();
        });

        let parent = snapshot_path.parent().unwrap();
        let prefix = format!(
            ".{}.tmp-",
            snapshot_path.file_name().unwrap().to_string_lossy()
        );
        let mut saw_temp_snapshot = false;
        for _ in 0..1000 {
            if parent.exists()
                && std::fs::read_dir(parent)
                    .unwrap()
                    .flatten()
                    .any(|entry| entry.file_name().to_string_lossy().starts_with(&prefix))
            {
                saw_temp_snapshot = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(saw_temp_snapshot);

        let result = coll_arc
            .read()
            .search(&[1.0, 0.0, 0.0, 0.0], 1, None, 10, false, 1e-4)
            .unwrap();
        assert_eq!(result.ids, vec![1]);
        assert!(coll_arc.try_write().is_none());

        handle.join().unwrap();
    }

    #[test]
    fn checkpoint_clears_wal_and_reopen_preserves_rows() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
            coll.add_items(&[1.0, 0.0, 0.0, 0.0], 1, &[7], None)
                .unwrap();
            assert!(coll.has_uncommitted_data());
            coll.checkpoint().unwrap();
            assert!(!coll.has_uncommitted_data());
        }

        let coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        assert_eq!(coll.shape().unwrap(), (1, 4));
        assert!(coll.is_id_exists(7));
        assert!(!coll.has_uncommitted_data());
    }

    #[test]
    fn collection_read_only_open_allows_reads_and_rejects_writes() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
            coll.add_items(&[1.0, 0.0, 0.0, 0.0], 1, &[7], None)
                .unwrap();
            coll.commit().unwrap();
        }

        let mut ro = Collection::open_read_only(tmp.path(), "col", 4, 100).unwrap();
        let ro2 = Collection::open_read_only(tmp.path(), "col", 4, 100).unwrap();
        assert!(ro.is_read_only());
        assert!(ro2.is_read_only());

        let result = ro
            .search(&[1.0, 0.0, 0.0, 0.0], 1, None, 10, false, 1e-4)
            .unwrap();
        assert_eq!(result.ids, vec![7]);

        let err = ro
            .add_items(&[0.0, 1.0, 0.0, 0.0], 1, &[8], None)
            .unwrap_err();
        assert!(err.to_string().contains("read-only"));

        drop(ro2);
        ro.close().unwrap();
        Collection::open(tmp.path(), "col", 4, 100).unwrap();
    }

    #[test]
    fn collection_read_only_open_rejects_uncommitted_wal() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
            coll.add_items(&[1.0, 0.0, 0.0, 0.0], 1, &[7], None)
                .unwrap();
            assert!(coll.has_uncommitted_data());
        }

        let err = match Collection::open_read_only(tmp.path(), "col", 4, 100) {
            Ok(_) => panic!("read-only open should reject collections needing recovery"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("uncommitted WAL"));
    }

    #[test]
    fn vector_only_uncommitted_wal_reopen_preserves_custom_ids() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
            let vectors = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
            coll.add_items(&vectors, 2, &[100, 200], None).unwrap();
            assert!(coll.has_uncommitted_data());
        }

        let coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        assert_eq!(coll.shape().unwrap(), (2, 4));
        assert!(coll.is_id_exists(100));
        assert!(coll.is_id_exists(200));
        assert!(!coll.has_uncommitted_data());
    }

    #[test]
    fn wal_recovery_overwrites_existing_vector_rows() {
        use std::io::{Seek, Write};

        let tmp = TempDir::new().unwrap();
        let expected = vec![1.0, 2.0, 3.0, 4.0];
        {
            let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
            coll.add_items(&expected, 1, &[42], None).unwrap();
            coll.flush_pending_ingest().unwrap();
            assert!(coll.has_uncommitted_data());
        }

        let mut vectors = std::fs::OpenOptions::new()
            .write(true)
            .open(tmp.path().join("col").join("vectors.bin"))
            .unwrap();
        vectors.seek(std::io::SeekFrom::Start(0)).unwrap();
        vectors.write_all(&[0u8; 16]).unwrap();
        vectors.flush().unwrap();
        drop(vectors);

        let coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        assert_eq!(coll.read_vectors_by_ids_only(&[42]).unwrap(), expected);
        assert!(!coll.has_uncommitted_data());
    }

    #[test]
    fn uncommitted_wal_reopen_preserves_string_external_ids() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
            let vectors = vec![1.0, 0.0, 0.0, 0.0];
            let ids = vec![ExternalId::String("doc-a".into())];
            coll.add_records(&vectors, 1, &ids, None).unwrap();
            assert!(coll.has_uncommitted_data());
            let collection_path = tmp.path().join("col");
            assert!(collection_path
                .join(EXTERNAL_ID_MAP_DELTA_BIN_FILE)
                .exists());
            assert!(!collection_path.join(EXTERNAL_ID_MAP_DELTA_FILE).exists());
        }

        let coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        let external_id = ExternalId::String("doc-a".into());
        assert_eq!(coll.shape().unwrap(), (1, 4));
        assert!(coll.is_external_id_exists(&external_id));
        assert_eq!(
            coll.internal_ids_for_external_ids(&[external_id.clone()])
                .unwrap(),
            vec![0]
        );
        assert_eq!(coll.external_ids_for_internal_ids(&[0]), vec![external_id]);
        assert!(!coll.has_uncommitted_data());
    }

    #[test]
    fn fields_remain_aligned_after_vector_only_rows() {
        let tmp = TempDir::new().unwrap();
        let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();

        let first = vec![1.0, 0.0, 0.0, 0.0];
        coll.add_items(&first, 1, &[10], None).unwrap();

        let second = vec![0.0, 1.0, 0.0, 0.0];
        let fields = vec![HashMap::from([(
            "tag".to_string(),
            serde_json::json!("with_fields"),
        )])];
        coll.add_items(&second, 1, &[20], Some(&fields)).unwrap();

        let retrieved = coll.retrieve_fields(&[20]).unwrap();
        assert_eq!(
            retrieved[0].get("tag"),
            Some(&serde_json::json!("with_fields"))
        );
    }

    #[test]
    fn legacy_row_keyed_fields_migrate_to_stable_ids_before_compaction() {
        let tmp = TempDir::new().unwrap();
        let collection_path = tmp.path().join("col");
        {
            let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
            coll.add_items(&[1.0; 8], 2, &[10, 20], None).unwrap();
            coll.checkpoint().unwrap();
        }

        std::fs::remove_file(collection_path.join(STABLE_FIELD_IDS_FILE)).unwrap();
        let legacy = FieldStore::new(&collection_path.join("fields_db"), "fields").unwrap();
        legacy
            .rebuild_at_ids(
                &[0, 1],
                &[
                    HashMap::from([("tag".to_string(), serde_json::json!("first"))]),
                    HashMap::from([("tag".to_string(), serde_json::json!("second"))]),
                ],
            )
            .unwrap();
        drop(legacy);

        let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        assert!(collection_path.join(STABLE_FIELD_IDS_FILE).exists());
        assert_eq!(coll.query_fields("\"tag\" = 'second'").unwrap(), vec![20]);
        coll.delete_items(&[10]).unwrap();
        assert_eq!(coll.compact().unwrap(), 1);
        assert_eq!(coll.retrieve_fields(&[20]).unwrap()[0]["tag"], "second");
    }

    #[test]
    fn update_existing_id_rewrites_vector_without_growing_shape() {
        let tmp = TempDir::new().unwrap();
        let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();

        coll.add_items(&[1.0, 0.0, 0.0, 0.0], 1, &[7], None)
            .unwrap();
        coll.update_items(&[7], &[0.0, 1.0, 0.0, 0.0], 1, None)
            .unwrap();

        assert_eq!(coll.shape().unwrap(), (1, 4));
        let (vectors, _) = coll.read_vectors_by_ids(&[7]).unwrap();
        assert_eq!(vectors, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn writable_open_replays_pending_positional_update_journal() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
            coll.add_items(&[1.0, 0.0, 0.0, 0.0], 1, &[7], None)
                .unwrap();
            coll.commit().unwrap();
            let replacement = encode_f32_slice_as_le_bytes(
                &[0.0, 1.0, 0.0, 0.0],
                VectorDtype::F32,
            );
            coll.vector_store
                .write_update_journal(&[0], &replacement)
                .unwrap();
        }

        let read_only_error = match Collection::open_read_only(tmp.path(), "col", 4, 100) {
            Ok(_) => panic!("read-only open should reject a pending positional update"),
            Err(error) => error,
        };
        assert!(read_only_error.to_string().contains("pending vector updates"));

        let coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        let (vectors, _) = coll.read_vectors_by_ids(&[7]).unwrap();
        assert_eq!(vectors, vec![0.0, 1.0, 0.0, 0.0]);
        assert!(!coll.vector_store.has_pending_updates());
    }

    #[test]
    fn update_missing_id_is_rejected() {
        let tmp = TempDir::new().unwrap();
        let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();

        let err = coll
            .update_items(&[404], &[0.0, 1.0, 0.0, 0.0], 1, None)
            .unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn upsert_updates_existing_and_inserts_missing() {
        let tmp = TempDir::new().unwrap();
        let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();

        let initial_fields = vec![HashMap::from([(
            "tag".to_string(),
            serde_json::json!("old"),
        )])];
        coll.add_items(&[1.0, 0.0, 0.0, 0.0], 1, &[1], Some(&initial_fields))
            .unwrap();

        let upsert_fields = vec![
            HashMap::from([("tag".to_string(), serde_json::json!("new"))]),
            HashMap::from([("tag".to_string(), serde_json::json!("inserted"))]),
        ];
        coll.upsert_items(
            &[1, 2],
            &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            2,
            Some(&upsert_fields),
        )
        .unwrap();

        assert_eq!(coll.shape().unwrap(), (2, 4));
        let (vectors, fields) = coll.read_vectors_by_ids(&[1, 2]).unwrap();
        assert_eq!(vectors, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(fields[0].get("tag"), Some(&serde_json::json!("new")));
        assert_eq!(fields[1].get("tag"), Some(&serde_json::json!("inserted")));
        assert!(coll.query_fields("\"tag\" = 'old'").unwrap().is_empty());
        assert_eq!(coll.query_fields("\"tag\" = 'new'").unwrap(), vec![1]);
    }

    #[test]
    fn upsert_without_fields_preserves_existing_fields() {
        let tmp = TempDir::new().unwrap();
        let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();

        let fields = vec![HashMap::from([(
            "tag".to_string(),
            serde_json::json!("preserved"),
        )])];
        coll.add_items(&[1.0, 0.0, 0.0, 0.0], 1, &[3], Some(&fields))
            .unwrap();
        coll.upsert_items(&[3], &[0.0, 1.0, 0.0, 0.0], 1, None)
            .unwrap();

        let retrieved = coll.retrieve_fields(&[3]).unwrap();
        assert_eq!(
            retrieved[0].get("tag"),
            Some(&serde_json::json!("preserved"))
        );
    }

    #[test]
    fn reopen_after_upsert_preserves_vectors_ids_and_fields() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
            let initial_fields = vec![HashMap::from([(
                "tag".to_string(),
                serde_json::json!("old"),
            )])];
            coll.add_items(&[1.0, 0.0, 0.0, 0.0], 1, &[11], Some(&initial_fields))
                .unwrap();
            coll.commit().unwrap();

            let upsert_fields = vec![
                HashMap::from([("tag".to_string(), serde_json::json!("updated"))]),
                HashMap::from([("tag".to_string(), serde_json::json!("inserted"))]),
            ];
            coll.upsert_items(
                &[11, 22],
                &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                2,
                Some(&upsert_fields),
            )
            .unwrap();
        }

        let coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        assert_eq!(coll.shape().unwrap(), (2, 4));
        assert!(coll.is_id_exists(11));
        assert!(coll.is_id_exists(22));

        let (vectors, fields) = coll.read_vectors_by_ids(&[11, 22]).unwrap();
        assert_eq!(vectors, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(fields[0].get("tag"), Some(&serde_json::json!("updated")));
        assert_eq!(fields[1].get("tag"), Some(&serde_json::json!("inserted")));
        assert!(coll.query_fields("\"tag\" = 'old'").unwrap().is_empty());
        assert_eq!(coll.query_fields("\"tag\" = 'updated'").unwrap(), vec![11]);
    }

    #[test]
    fn tombstones_survive_reopen_and_filter_search() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
            coll.add_items(
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                2,
                &[10, 20],
                None,
            )
            .unwrap();
            coll.commit().unwrap();
            coll.delete_items(&[20]).unwrap();
        }

        let coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        assert_eq!(coll.list_deleted_ids(), vec![20]);

        let result = coll
            .search(&[0.0, 1.0, 0.0, 0.0], 2, None, 10, false, 1e-4)
            .unwrap();
        assert_eq!(result.ids, vec![10]);
    }

    #[test]
    fn query_profile_reports_filter_and_search_path() {
        let tmp = TempDir::new().unwrap();
        let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        let fields = vec![
            HashMap::from([("category".to_string(), serde_json::json!("doc"))]),
            HashMap::from([("category".to_string(), serde_json::json!("code"))]),
        ];
        coll.add_items(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            2,
            &[10, 20],
            Some(&fields),
        )
        .unwrap();

        let (result, profile) = coll
            .search_with_profile(
                &[1.0, 0.0, 0.0, 0.0],
                1,
                Some("\"category\" = 'doc'"),
                10,
                false,
                1e-4,
            )
            .unwrap();
        assert_eq!(result.ids, vec![10]);
        assert_eq!(profile.query_kind, "vector");
        assert_eq!(profile.filter_matches, Some(1));
        assert_eq!(profile.result_count, 1);
        assert!(profile.index_path.contains("flat"));
    }

    #[test]
    fn text_and_hybrid_search_rank_metadata_matches() {
        let tmp = TempDir::new().unwrap();
        let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        let fields = vec![
            HashMap::from([(
                "body".to_string(),
                serde_json::json!("rust vector database"),
            )]),
            HashMap::from([(
                "body".to_string(),
                serde_json::json!("python web framework"),
            )]),
            HashMap::from([(
                "body".to_string(),
                serde_json::json!("hybrid vector search"),
            )]),
        ];
        coll.add_items(
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            3,
            &[10, 20, 30],
            Some(&fields),
        )
        .unwrap();

        let text_fields = vec!["body".to_string()];
        let text_result = coll
            .text_search("vector database", Some(&text_fields), 2, None)
            .unwrap();
        assert!(text_result.ids.contains(&10));
        assert!(text_result.distances[0] > 0.0);
        assert_eq!(text_result.index_mode, "BM25-INVERTED");

        let hybrid = coll
            .hybrid_search(
                Some(&[0.0, 1.0, 0.0, 0.0]),
                Some("vector"),
                3,
                None,
                Some(&text_fields),
                "rrf",
                1.0,
                1.0,
                60.0,
                6,
                10,
            )
            .unwrap();
        assert_eq!(hybrid.ids.len(), 3);
        assert!(hybrid.ids.contains(&20));
        assert!(hybrid.ids.iter().any(|id| *id == 10 || *id == 30));
    }

    #[test]
    fn text_index_survives_reopen_and_tracks_field_updates() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
            let fields = vec![
                HashMap::from([("body".to_string(), serde_json::json!("alpha source"))]),
                HashMap::from([("body".to_string(), serde_json::json!("beta target"))]),
            ];
            coll.add_items(
                &[
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0,
                ],
                2,
                &[10, 20],
                Some(&fields),
            )
            .unwrap();
            coll.checkpoint().unwrap();
        }

        let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        let text_fields = vec!["body".to_string()];
        let result = coll
            .text_search("target", Some(&text_fields), 5, None)
            .unwrap();
        assert_eq!(result.index_mode, "BM25-INVERTED");
        assert_eq!(result.ids, vec![20]);

        let updated_fields = vec![HashMap::from([(
            "body".to_string(),
            serde_json::json!("alpha target"),
        )])];
        coll.update_items(&[10], &[0.5, 0.0, 0.0, 0.0], 1, Some(&updated_fields))
            .unwrap();

        let updated = coll
            .text_search("target", Some(&text_fields), 5, None)
            .unwrap();
        assert!(updated.ids.contains(&10));
        assert!(updated.ids.contains(&20));

        coll.delete_items(&[20]).unwrap();
        let filtered = coll
            .text_search("target", Some(&text_fields), 5, None)
            .unwrap();
        assert_eq!(filtered.ids, vec![10]);
    }

    #[test]
    fn metadata_indexes_cover_range_bool_array_datetime_and_reopen() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
            let fields: Vec<HashMap<String, serde_json::Value>> = (0..5)
                .map(|i| {
                    HashMap::from([
                        ("order".to_string(), serde_json::json!(i)),
                        ("active".to_string(), serde_json::json!(i % 2 == 0)),
                        (
                            "tags".to_string(),
                            serde_json::json!(if i % 2 == 0 {
                                vec!["rust", "vector"]
                            } else {
                                vec!["python"]
                            }),
                        ),
                        (
                            "created_at".to_string(),
                            serde_json::json!(format!("2026-04-{:02}", i + 1)),
                        ),
                    ])
                })
                .collect();
            let vectors: Vec<f32> = (0..5).flat_map(|i| [i as f32, 0.0, 0.0, 0.0]).collect();
            coll.add_items(&vectors, 5, &[10, 11, 12, 13, 14], Some(&fields))
                .unwrap();
            coll.checkpoint().unwrap();
        }

        let coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        assert_eq!(
            coll.query_fields("\"order\" >= 2 AND \"order\" < 4")
                .unwrap(),
            vec![12, 13]
        );
        assert_eq!(
            coll.query_fields("\"active\" = true").unwrap(),
            vec![10, 12, 14]
        );
        assert_eq!(
            coll.query_fields("\"tags\" CONTAINS 'rust'").unwrap(),
            vec![10, 12, 14]
        );
        assert_eq!(
            coll.query_fields("\"created_at\" >= '2026-04-03' AND \"created_at\" <= '2026-04-04'")
                .unwrap(),
            vec![12, 13]
        );
        assert_eq!(
            coll.query_fields("\"order\" IN (1, 2)").unwrap(),
            vec![11, 12]
        );
        assert_eq!(
            coll.query_fields("\"order\" = 1 OR \"order\" = 2").unwrap(),
            vec![11, 12]
        );
    }

    #[test]
    fn named_vector_field_survives_reopen_and_searches_independently() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
            coll.add_items(
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                2,
                &[10, 20],
                None,
            )
            .unwrap();
            coll.create_vector_field("image", 3, Some("l2"), None)
                .unwrap();
            coll.add_named_vectors("image", &[0.0, 0.0, 0.0, 5.0, 0.0, 0.0], 2, &[10, 20])
                .unwrap();
            coll.build_vector_field_index("image", "HNSW-L2").unwrap();
            coll.checkpoint().unwrap();
        }

        let coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        let vector_fields = coll.list_vector_fields();
        assert!(vector_fields.iter().any(|field| field.name == "image"
            && field.dimension == 3
            && field.metric == "l2"
            && field.index_mode == "HNSW-L2"));

        let result = coll
            .search_vector_field("image", &[4.9, 0.0, 0.0], 1, None)
            .unwrap();
        assert_eq!(result.ids, vec![20]);

        let default_result = coll
            .search(&[1.0, 0.0, 0.0, 0.0], 1, None, 10, false, 1e-4)
            .unwrap();
        assert_eq!(default_result.ids, vec![10]);
    }

    #[test]
    fn named_vector_field_approx_search_uses_sampled_dims() {
        let tmp = TempDir::new().unwrap();
        let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        let n = 4096usize;
        let dim = 128usize;
        let true_id = 3000u64;

        let primary = vec![0.0f32; n * 4];
        let ids: Vec<u64> = (0..n as u64).collect();
        coll.add_items(&primary, n, &ids, None).unwrap();
        coll.create_vector_field("image", dim, Some("l2"), None)
            .unwrap();

        let mut named = vec![0.0f32; n * dim];
        let mut query = vec![0.0f32; dim];
        for j in 96..dim {
            query[j] = 1.0;
            named[true_id as usize * dim + j] = 1.0;
        }
        coll.add_named_vectors("image", &named, n, &ids).unwrap();

        let result = coll
            .search_vector_field_with_options("image", &query, 1, None, true, 1e-4)
            .unwrap();
        assert_eq!(result.ids, vec![true_id]);
        assert_eq!(result.distances, vec![0.0]);
    }

    #[test]
    fn sparse_vectors_survive_reopen_filter_and_compact() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
            let fields = vec![
                HashMap::from([("group".to_string(), serde_json::json!("a"))]),
                HashMap::from([("group".to_string(), serde_json::json!("b"))]),
                HashMap::from([("group".to_string(), serde_json::json!("a"))]),
            ];
            coll.add_items(
                &[
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0,
                ],
                3,
                &[10, 20, 30],
                Some(&fields),
            )
            .unwrap();
            coll.add_sparse_vectors(
                &[10, 20, 30],
                &[
                    vec![(1, 1.0), (5, 0.5)],
                    vec![(2, 2.0), (5, 1.0)],
                    vec![(2, 0.5), (7, 1.0)],
                ],
            )
            .unwrap();
            coll.checkpoint().unwrap();
        }

        let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        let result = coll.search_sparse(&[(2, 1.0)], 2, None).unwrap();
        assert_eq!(result.ids, vec![20, 30]);
        assert_eq!(result.index_mode, "SPARSE-FLAT-IP");

        let filtered = coll
            .search_sparse(&[(5, 1.0)], 5, Some("\"group\" = 'b'"))
            .unwrap();
        assert_eq!(filtered.ids, vec![20]);

        coll.delete_items(&[20]).unwrap();
        coll.compact().unwrap();

        let after_compact = coll.search_sparse(&[(2, 1.0)], 5, None).unwrap();
        assert_eq!(after_compact.ids, vec![30]);
    }

    #[test]
    fn storage_manifest_is_created_and_validated_on_open() {
        let tmp = TempDir::new().unwrap();
        Collection::open(tmp.path(), "col", 4, 100).unwrap();

        let manifest_path = tmp.path().join("col").join(STORAGE_MANIFEST_FILE);
        assert!(manifest_path.exists());

        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["format"], STORAGE_FORMAT_NAME);
        assert_eq!(manifest["version"], STORAGE_FORMAT_VERSION);
        assert_eq!(manifest["dimension"], 4);
        assert_eq!(manifest["chunk_size"], 100);

        let err = match Collection::open(tmp.path(), "col", 8, 100) {
            Ok(_) => panic!("opening with a mismatched dimension should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("Dimension mismatch"));
    }

    #[test]
    fn storage_manifest_rejects_future_versions() {
        let tmp = TempDir::new().unwrap();
        Collection::open(tmp.path(), "col", 4, 100).unwrap();

        let manifest_path = tmp.path().join("col").join(STORAGE_MANIFEST_FILE);
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["version"] = serde_json::json!(STORAGE_FORMAT_VERSION + 1);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let err = match Collection::open(tmp.path(), "col", 4, 100) {
            Ok(_) => panic!("opening a future storage format should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("supports up to"));
    }

    #[test]
    fn persisted_hnsw_index_survives_reopen() {
        let tmp = TempDir::new().unwrap();
        {
            let mut coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
            coll.add_items(
                &[
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, //
                    0.0, 0.0, 0.0, 1.0,
                ],
                4,
                &[101, 202, 303, 404],
                None,
            )
            .unwrap();
            coll.commit().unwrap();
            coll.build_index("HNSW-IP").unwrap();

            let meta_path = tmp
                .path()
                .join("col")
                .join("index_meta")
                .join("index_metadata.json");
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
            let index_file = meta["index_file"].as_str().unwrap();
            assert_ne!(index_file, "index.bin");
            assert!(tmp
                .path()
                .join("col")
                .join("index")
                .join(index_file)
                .exists());
        }

        let coll = Collection::open(tmp.path(), "col", 4, 100).unwrap();
        assert_eq!(coll.get_index_mode(), Some("HNSW-IP"));
        let result = coll
            .search(&[0.0, 1.0, 0.0, 0.0], 1, None, 10, false, 1e-4)
            .unwrap();
        assert_eq!(result.ids, vec![202]);
    }
}

// ─── DatabaseManager: manages multiple databases ─────────────────────────────

/// Collection registration info stored in collections.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub dim: usize,
    pub chunk_size: usize,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default = "default_vector_dtype_string")]
    pub dtypes: String,
}

/// Top-level manager for multiple databases.
/// Each database is a directory containing collections.
pub struct DatabaseManager {
    root_path: PathBuf,
    databases: RwLock<HashMap<String, DatabaseEngine>>,
    _lock: FileLock,
    read_only: bool,
}

impl DatabaseManager {
    /// Open the database manager at the given root path.
    pub fn new(root_path: &Path) -> Result<Self> {
        Self::open_with_mode(root_path, OpenMode::ReadWrite)
    }

    /// Open the database manager in read-only mode.
    pub fn new_read_only(root_path: &Path) -> Result<Self> {
        Self::open_with_mode(root_path, OpenMode::ReadOnly)
    }

    fn open_with_mode(root_path: &Path, mode: OpenMode) -> Result<Self> {
        if mode == OpenMode::ReadWrite {
            std::fs::create_dir_all(root_path)?;
        } else if !root_path.exists() {
            return Err(LynseError::DatabaseNotFound(
                root_path.to_string_lossy().to_string(),
            ));
        }
        let lock_path = root_path.join(".manager.lock");
        let lock = if mode == OpenMode::ReadWrite {
            FileLock::exclusive(&lock_path)?
        } else {
            FileLock::shared(&lock_path)?
        };
        let mgr = Self {
            root_path: root_path.to_path_buf(),
            databases: RwLock::new(HashMap::new()),
            _lock: lock,
            read_only: mode == OpenMode::ReadOnly,
        };
        // Scan for existing databases
        mgr.scan_databases();
        Ok(mgr)
    }

    fn ensure_writable(&self) -> Result<()> {
        if self.read_only {
            return Err(LynseError::InvalidArgument(
                "database manager is opened read-only".to_string(),
            ));
        }
        Ok(())
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Scan root path for existing databases (directories with .fingerprint).
    fn scan_databases(&self) {
        if let Ok(entries) = std::fs::read_dir(&self.root_path) {
            for entry in entries.flatten() {
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if entry.path().join(".fingerprint").exists() {
                        let _ = self.get_or_open_database(&name);
                    }
                }
            }
        }
    }

    /// Create a new database.
    pub fn create_database(&self, name: &str) -> Result<()> {
        self.ensure_writable()?;
        let db_path = self.root_path.join(name);
        std::fs::create_dir_all(&db_path)?;

        // Write fingerprint
        let fp_path = db_path.join(".fingerprint");
        if !fp_path.exists() {
            let fp = uuid::Uuid::new_v4().to_string().replace("-", "");
            std::fs::write(&fp_path, &fp)?;
        }

        // Register in databases.json
        self.register_database(name)?;

        // Open the engine
        let _ = self.get_or_open_database(name)?;
        Ok(())
    }

    /// Drop a database and all its collections.
    pub fn drop_database(&self, name: &str) -> Result<()> {
        self.ensure_writable()?;
        let mut dbs = self.databases.write();
        let existing = dbs.remove(name);
        drop(dbs);

        let db_path = self.root_path.join(name);
        if let Some(engine) = existing {
            engine.close_open_collections()?;
        }

        if db_path.exists() {
            std::fs::remove_dir_all(&db_path)?;
        }

        self.deregister_database(name)?;
        Ok(())
    }

    /// List all databases.
    pub fn list_databases(&self) -> Vec<String> {
        // Merge from databases.json and in-memory
        let mut result: Vec<String> = Vec::new();
        let db_json = self.root_path.join("databases.json");
        if let Ok(content) = std::fs::read_to_string(&db_json) {
            if let Ok(list) = serde_json::from_str::<Vec<String>>(&content) {
                for name in &list {
                    if self.root_path.join(name).exists() && !result.contains(name) {
                        result.push(name.clone());
                    }
                }
            }
        }
        // Also check in-memory
        let dbs = self.databases.read();
        for name in dbs.keys() {
            if !result.contains(name) {
                result.push(name.clone());
            }
        }
        result
    }

    /// Check if a database exists.
    pub fn database_exists(&self, name: &str) -> bool {
        self.root_path.join(name).join(".fingerprint").exists()
    }

    /// Get or open a DatabaseEngine for a specific database.
    pub fn get_or_open_database(&self, name: &str) -> Result<()> {
        {
            let dbs = self.databases.read();
            if dbs.contains_key(name) {
                return Ok(());
            }
        }

        let mut dbs = self.databases.write();
        if dbs.contains_key(name) {
            return Ok(());
        }

        let db_path = self.root_path.join(name);
        if !db_path.exists() {
            return Err(LynseError::DatabaseNotFound(name.to_string()));
        }
        let engine = if self.read_only {
            DatabaseEngine::open_read_only(&db_path)?
        } else {
            DatabaseEngine::open(&db_path)?
        };
        dbs.insert(name.to_string(), engine);
        Ok(())
    }

    /// Get a reference to the database engine (caller must hold read lock).
    pub fn with_database<F, T>(&self, name: &str, f: F) -> Result<T>
    where
        F: FnOnce(&DatabaseEngine) -> Result<T>,
    {
        let dbs = self.databases.read();
        let engine = dbs
            .get(name)
            .ok_or_else(|| LynseError::DatabaseNotFound(name.to_string()))?;
        f(engine)
    }

    /// Get a mutable reference to the database engine.
    pub fn with_database_mut<F, T>(&self, name: &str, f: F) -> Result<T>
    where
        F: FnOnce(&DatabaseEngine) -> Result<T>,
    {
        self.ensure_writable()?;
        let dbs = self.databases.read();
        let engine = dbs
            .get(name)
            .ok_or_else(|| LynseError::DatabaseNotFound(name.to_string()))?;
        f(engine)
    }

    /// Register database in databases.json.
    fn register_database(&self, name: &str) -> Result<()> {
        let db_json = self.root_path.join("databases.json");
        let mut list: Vec<String> = if let Ok(content) = std::fs::read_to_string(&db_json) {
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            Vec::new()
        };
        if !list.contains(&name.to_string()) {
            list.push(name.to_string());
        }
        let json =
            serde_json::to_string(&list).map_err(|e| LynseError::Serialization(e.to_string()))?;
        std::fs::write(&db_json, json)?;
        Ok(())
    }

    /// Deregister database from databases.json.
    fn deregister_database(&self, name: &str) -> Result<()> {
        let db_json = self.root_path.join("databases.json");
        if !db_json.exists() {
            return Ok(());
        }
        let mut list: Vec<String> = if let Ok(content) = std::fs::read_to_string(&db_json) {
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            Vec::new()
        };
        list.retain(|n| n != name);
        let json =
            serde_json::to_string(&list).map_err(|e| LynseError::Serialization(e.to_string()))?;
        std::fs::write(&db_json, json)?;
        Ok(())
    }

    /// Require (create or get) a collection within a database.
    pub fn require_collection(
        &self,
        db_name: &str,
        collection_name: &str,
        dim: usize,
        chunk_size: usize,
        drop_if_exists: bool,
        description: Option<&str>,
    ) -> Result<()> {
        self.require_collection_with_dtype(
            db_name,
            collection_name,
            dim,
            chunk_size,
            drop_if_exists,
            description,
            None,
        )
    }

    pub fn require_collection_with_dtype(
        &self,
        db_name: &str,
        collection_name: &str,
        dim: usize,
        chunk_size: usize,
        drop_if_exists: bool,
        description: Option<&str>,
        dtypes: Option<&str>,
    ) -> Result<()> {
        self.ensure_writable()?;
        let vector_dtype = match dtypes {
            Some(dtype) => VectorDtype::parse(dtype)?,
            None => VectorDtype::F32,
        };
        // Ensure database is open
        self.get_or_open_database(db_name)?;

        if drop_if_exists {
            self.with_database(db_name, |engine| {
                let _ = engine.drop_collection(collection_name);
                Ok(())
            })?;
        }

        let effective_dim = self.with_database(db_name, |engine| {
            let coll = engine.get_or_open_collection_with_dtype(
                collection_name,
                dim,
                chunk_size,
                Some(vector_dtype),
            )?;
            let dimension = coll.read().dimension();
            Ok(dimension)
        })?;

        // Save collection config
        self.save_collection_config(
            db_name,
            collection_name,
            effective_dim,
            chunk_size,
            description,
            Some(vector_dtype),
        )?;
        Ok(())
    }

    /// Save collection config to collections.json in the database directory.
    fn save_collection_config(
        &self,
        db_name: &str,
        collection_name: &str,
        dim: usize,
        chunk_size: usize,
        description: Option<&str>,
        vector_dtype: Option<VectorDtype>,
    ) -> Result<()> {
        let db_path = self.root_path.join(db_name);
        let config_path = db_path.join("collections.json");
        let mut configs: HashMap<String, CollectionConfig> =
            if let Ok(content) = std::fs::read_to_string(&config_path) {
                serde_json::from_str(&content).unwrap_or_default()
            } else {
                HashMap::new()
            };

        let entry = configs
            .entry(collection_name.to_string())
            .or_insert(CollectionConfig {
                dim,
                chunk_size,
                description: description.map(|s| s.to_string()),
                dtypes: vector_dtype.unwrap_or_default().storage_name().to_string(),
            });
        entry.dim = dim;
        entry.chunk_size = chunk_size;
        if let Some(dtype) = vector_dtype {
            entry.dtypes = dtype.storage_name().to_string();
        }
        if let Some(desc) = description {
            entry.description = Some(desc.to_string());
        }

        let json = serde_json::to_string_pretty(&configs)
            .map_err(|e| LynseError::Serialization(e.to_string()))?;
        std::fs::write(&config_path, json)?;
        Ok(())
    }

    /// Load collection configs from collections.json.
    pub fn get_collection_configs(
        &self,
        db_name: &str,
    ) -> Result<HashMap<String, CollectionConfig>> {
        let config_path = self.root_path.join(db_name).join("collections.json");
        if !config_path.exists() {
            return Ok(HashMap::new());
        }
        let content = std::fs::read_to_string(&config_path)?;
        let configs: HashMap<String, CollectionConfig> =
            serde_json::from_str(&content).unwrap_or_default();
        Ok(configs)
    }

    /// Update collection description.
    pub fn update_collection_description(
        &self,
        db_name: &str,
        collection_name: &str,
        description: Option<&str>,
    ) -> Result<()> {
        self.ensure_writable()?;
        self.get_or_open_database(db_name)?;
        let dbs = self.databases.read();
        let engine = dbs
            .get(db_name)
            .ok_or_else(|| LynseError::DatabaseNotFound(db_name.to_string()))?;
        let _collections = engine.collections.write();

        let config_path = self.root_path.join(db_name).join("collections.json");
        let mut configs: HashMap<String, CollectionConfig> =
            if let Ok(content) = std::fs::read_to_string(&config_path) {
                serde_json::from_str(&content).unwrap_or_default()
            } else {
                HashMap::new()
            };

        if let Some(cfg) = configs.get_mut(collection_name) {
            cfg.description = description.map(|s| s.to_string());
        }

        let json = serde_json::to_string_pretty(&configs)
            .map_err(|e| LynseError::Serialization(e.to_string()))?;
        std::fs::write(&config_path, json)?;
        Ok(())
    }

    /// Show collections in a database.
    pub fn show_collections(&self, db_name: &str) -> Result<Vec<String>> {
        self.get_or_open_database(db_name)?;
        self.with_database(db_name, |engine| Ok(engine.list_collections()))
    }

    /// Number of currently open database engines in this manager.
    pub fn open_database_count(&self) -> usize {
        self.databases.read().len()
    }

    /// Number of currently open collection handles across open databases.
    pub fn open_collection_count(&self) -> usize {
        let dbs = self.databases.read();
        dbs.values()
            .map(DatabaseEngine::open_collection_count)
            .sum()
    }

    /// Checkpoint all currently open collections across all open databases.
    pub fn checkpoint_open_collections(&self) -> Result<()> {
        if self.read_only {
            return Ok(());
        }

        let dbs = self.databases.read();
        for engine in dbs.values() {
            engine.checkpoint_open_collections()?;
        }
        Ok(())
    }

    fn read_database_snapshot_manifest(path: &Path) -> Result<DatabaseSnapshotManifest> {
        let manifest_path = path.join(DATABASE_SNAPSHOT_MANIFEST_FILE);
        let bytes = std::fs::read(&manifest_path)?;
        serde_json::from_slice(&bytes).map_err(|e| LynseError::Serialization(e.to_string()))
    }

    fn validate_database_snapshot_manifest(manifest: &DatabaseSnapshotManifest) -> Result<()> {
        if manifest.storage_format != DATABASE_SNAPSHOT_FORMAT_NAME {
            return Err(LynseError::Storage(format!(
                "unsupported database snapshot format '{}'",
                manifest.storage_format
            )));
        }
        if manifest.storage_version > STORAGE_FORMAT_VERSION {
            return Err(LynseError::Storage(format!(
                "database snapshot uses storage format version {}, but this binary supports up to {}",
                manifest.storage_version, STORAGE_FORMAT_VERSION
            )));
        }
        Ok(())
    }

    /// Create a filesystem snapshot for a whole database.
    pub fn snapshot_database(&self, db_name: &str, snapshot_path: &Path) -> Result<()> {
        self.get_or_open_database(db_name)?;
        let dbs = self.databases.read();
        let engine = dbs
            .get(db_name)
            .ok_or_else(|| LynseError::DatabaseNotFound(db_name.to_string()))?;
        engine.snapshot_to(db_name, snapshot_path)
    }

    /// Restore a whole database from a filesystem snapshot.
    pub fn restore_database_from_snapshot(
        &self,
        db_name: &str,
        snapshot_path: &Path,
        overwrite: bool,
    ) -> Result<()> {
        self.ensure_writable()?;
        if !snapshot_path.exists() {
            return Err(LynseError::InvalidArgument(format!(
                "snapshot path does not exist: {}",
                snapshot_path.display()
            )));
        }

        let snapshot_manifest = Self::read_database_snapshot_manifest(snapshot_path)?;
        Self::validate_database_snapshot_manifest(&snapshot_manifest)?;

        let target_path = self.root_path.join(db_name);
        if target_path.exists() && !overwrite {
            return Err(LynseError::InvalidArgument(format!(
                "database already exists: {}",
                db_name
            )));
        }

        let existing = {
            let mut dbs = self.databases.write();
            dbs.remove(db_name)
        };

        if let Some(engine) = existing {
            if let Some(collection_name) = engine.external_collection_handle() {
                let mut dbs = self.databases.write();
                dbs.insert(db_name.to_string(), engine);
                return Err(LynseError::Storage(format!(
                    "collection '{}.{}' is still referenced; close active handles before restore",
                    db_name, collection_name
                )));
            }
            if let Err(err) = engine.close_open_collections() {
                let mut dbs = self.databases.write();
                dbs.insert(db_name.to_string(), engine);
                return Err(err);
            }
        }

        if target_path.exists() {
            std::fs::remove_dir_all(&target_path)?;
        }

        let tmp_path = Collection::make_temp_sibling(&target_path)?;
        if tmp_path.exists() {
            std::fs::remove_dir_all(&tmp_path)?;
        }

        let restore_result = (|| -> Result<()> {
            Collection::copy_dir_for_snapshot(snapshot_path, &tmp_path)?;

            let fingerprint_path = tmp_path.join(".fingerprint");
            if snapshot_manifest.database_name != db_name || !fingerprint_path.exists() {
                let fp = uuid::Uuid::new_v4().to_string().replace("-", "");
                Collection::atomic_write_file(&fingerprint_path, fp.as_bytes())?;
            }

            Collection::sync_path_recursively(&tmp_path)?;
            std::fs::rename(&tmp_path, &target_path)?;
            if let Ok(dir) = std::fs::File::open(&self.root_path) {
                let _ = dir.sync_all();
            }
            Ok(())
        })();

        if let Err(err) = restore_result {
            let _ = std::fs::remove_dir_all(&tmp_path);
            return Err(err);
        }

        self.get_or_open_database(db_name)?;
        self.register_database(db_name)
    }

    /// Create a snapshot for a collection within a database.
    pub fn snapshot_collection(
        &self,
        db_name: &str,
        collection_name: &str,
        snapshot_path: &Path,
    ) -> Result<()> {
        self.get_or_open_database(db_name)?;
        self.with_database(db_name, |engine| {
            engine.snapshot_collection(collection_name, snapshot_path)
        })
    }

    /// Export a collection to portable JSONL metadata plus binary vectors.
    pub fn export_collection(
        &self,
        db_name: &str,
        collection_name: &str,
        export_path: &Path,
    ) -> Result<()> {
        self.get_or_open_database(db_name)?;
        self.with_database(db_name, |engine| {
            engine.export_collection(collection_name, export_path)
        })
    }

    /// Restore a collection from a snapshot directory.
    pub fn restore_collection_from_snapshot(
        &self,
        db_name: &str,
        collection_name: &str,
        snapshot_path: &Path,
        overwrite: bool,
    ) -> Result<()> {
        self.ensure_writable()?;
        self.get_or_open_database(db_name)?;
        let config = self.with_database(db_name, |engine| {
            engine.restore_collection_from_snapshot(collection_name, snapshot_path, overwrite)
        })?;
        self.save_collection_config(
            db_name,
            collection_name,
            config.dim,
            config.chunk_size,
            config.description.as_deref(),
            Some(VectorDtype::parse(&config.dtypes)?),
        )
    }

    /// Import a collection from portable JSONL metadata plus binary vectors.
    pub fn import_collection_from_export(
        &self,
        db_name: &str,
        collection_name: &str,
        export_path: &Path,
        overwrite: bool,
    ) -> Result<()> {
        self.ensure_writable()?;
        self.get_or_open_database(db_name)?;
        let config = self.with_database(db_name, |engine| {
            engine.import_collection_from_export(collection_name, export_path, overwrite)
        })?;
        self.save_collection_config(
            db_name,
            collection_name,
            config.dim,
            config.chunk_size,
            config.description.as_deref(),
            Some(VectorDtype::parse(&config.dtypes)?),
        )
    }

    /// Drop a collection from a database.
    pub fn drop_collection(&self, db_name: &str, collection_name: &str) -> Result<()> {
        self.ensure_writable()?;
        self.get_or_open_database(db_name)?;
        self.with_database(db_name, |engine| engine.drop_collection(collection_name))?;

        // Remove from collections.json
        let config_path = self.root_path.join(db_name).join("collections.json");
        if config_path.exists() {
            let mut configs: HashMap<String, CollectionConfig> =
                if let Ok(content) = std::fs::read_to_string(&config_path) {
                    serde_json::from_str(&content).unwrap_or_default()
                } else {
                    HashMap::new()
                };
            configs.remove(collection_name);
            let json = serde_json::to_string_pretty(&configs)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            std::fs::write(&config_path, json)?;
        }
        Ok(())
    }

    /// Check if collection exists in a database.
    pub fn collection_exists(&self, db_name: &str, collection_name: &str) -> Result<bool> {
        self.get_or_open_database(db_name)?;
        self.with_database(db_name, |engine| Ok(engine.has_collection(collection_name)))
    }

    /// Get the root path.
    pub fn root_path(&self) -> &Path {
        &self.root_path
    }
}
