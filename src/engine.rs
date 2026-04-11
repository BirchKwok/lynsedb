//! Database engine module.
//!
//! Orchestrates collections, storage, indexing, and search operations.
//! This is the main entry point for all database operations from Python.

use crate::distance::DistanceMetric;
use crate::error::{LynseError, Result};
use crate::index::{self, SearchParams, VectorIndex};
use crate::storage::field_store::FieldStore;
use crate::storage::polarvec_mmap::{
    parse_bits, PolarVecIndex, DEFAULT_BITS as POLARVEC_DEFAULT_BITS,
    DEFAULT_OVERSAMPLE as POLARVEC_OVERSAMPLE,
};
use crate::storage::pq_mmap::{parse_n_subspaces, PQIndex, DEFAULT_OVERSAMPLE as PQ_OVERSAMPLE};
use crate::storage::rabitq_mmap::{RaBitQIndex, DEFAULT_OVERSAMPLE as RABITQ_OVERSAMPLE};
use crate::storage::vector_store::VectorStore;
use crate::storage::wal::WALStorage;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const STORAGE_MANIFEST_FILE: &str = "storage_manifest.json";
const STORAGE_FORMAT_NAME: &str = "lynsedb-collection";
const STORAGE_FORMAT_VERSION: u32 = 1;
const WAL_FORMAT_VERSION: u32 = 2;

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
    vector_store: VectorStore,
    field_store: FieldStore,
    wal: WALStorage,
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
    /// Soft-delete tombstone: IDs marked for deletion, filtered from search results.
    tombstone: Arc<RwLock<HashSet<u64>>>,
    /// User-facing ID map: id_map[row_offset] = user_id.
    /// Persisted to id_map.bin. Backfilled with sequential IDs for backward compat.
    id_map: Vec<u64>,
    /// Reverse map: user_id → row_offset. Rebuilt from id_map on load.
    reverse_id_map: HashMap<u64, usize>,
    /// Holds the collection-level writer lock until close/drop.
    lock: Option<FileLock>,
}

#[cfg(unix)]
struct FileLock {
    _file: std::fs::File,
}

#[cfg(unix)]
impl FileLock {
    fn exclusive(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
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

impl StorageManifest {
    fn current(name: &str, dimension: usize, chunk_size: usize) -> Self {
        Self {
            format: STORAGE_FORMAT_NAME.to_string(),
            version: STORAGE_FORMAT_VERSION,
            collection_name: name.to_string(),
            dimension,
            chunk_size,
            vector_dtype: "float32".to_string(),
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
        let (n, dim) = self.shape()?;
        let path = self.vector_store.vectors_path();
        Ok((path.to_string_lossy().to_string(), n as usize, dim))
    }

    /// Create or open a collection.
    pub fn open(path: &Path, name: &str, dimension: usize, chunk_size: usize) -> Result<Self> {
        let collection_path = path.join(name);
        std::fs::create_dir_all(&collection_path)?;

        let lock = FileLock::exclusive(&collection_path.join(".writer.lock"))?;

        Self::ensure_storage_manifest(&collection_path, name, dimension, chunk_size)?;

        let vector_store = VectorStore::new(&collection_path, dimension, chunk_size)?;

        let field_db_path = collection_path.join("fields_db");
        let field_store = FieldStore::new(&field_db_path, "fields")?;

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
            dtypes: "float32".to_string(),
        };

        let tombstone_path = collection_path.join("tombstone.bin");
        let tombstone = Arc::new(RwLock::new(Self::load_tombstone_from_disk(&tombstone_path)));

        let mut coll = Self {
            meta,
            path: collection_path,
            vector_store,
            field_store,
            wal,
            index,
            index_mode,
            last_sync_fingerprint,
            pq_index: None,
            rabitq_index: None,
            polarvec_index: None,
            tombstone,
            id_map: Vec::new(),
            reverse_id_map: HashMap::new(),
            lock: Some(lock),
        };

        // Establish the durable row boundary before WAL replay. If a previous
        // process crashed after appending vectors but before appending id_map,
        // id_map is the authoritative set of fully applied rows.
        let id_map_repaired = coll.initialize_id_map_for_open()?;

        // Recover any uncommitted WAL data on startup.
        let recovered_wal = coll.recover_wal()?;

        if id_map_repaired || recovered_wal {
            if let Some(mode) = coll.index_mode.clone() {
                coll.build_index(&mode)?;
            }
        }

        // Load quantizer indices from disk if present
        coll.try_load_pq_rabitq()?;

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
    ) -> Result<StorageManifest> {
        let manifest_path = Self::storage_manifest_path(collection_path);
        if manifest_path.exists() {
            let bytes = std::fs::read(&manifest_path)?;
            let manifest: StorageManifest = serde_json::from_slice(&bytes)
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
            if manifest.dimension != dimension {
                return Err(LynseError::DimensionMismatch {
                    expected: manifest.dimension,
                    got: dimension,
                });
            }
            if manifest.chunk_size != chunk_size {
                return Err(LynseError::Storage(format!(
                    "collection chunk_size {} does not match requested chunk_size {}",
                    manifest.chunk_size, chunk_size
                )));
            }
            return Ok(manifest);
        }

        let manifest = StorageManifest::current(name, dimension, chunk_size);
        Self::write_storage_manifest(&manifest_path, &manifest)?;
        Ok(manifest)
    }

    fn write_storage_manifest(path: &Path, manifest: &StorageManifest) -> Result<()> {
        let json = serde_json::to_vec_pretty(manifest)
            .map_err(|e| LynseError::Serialization(e.to_string()))?;
        Self::atomic_write_file(path, &json)?;
        Ok(())
    }

    fn atomic_write_file(path: &Path, data: &[u8]) -> Result<()> {
        let tmp_path = path.with_extension("tmp");
        std::fs::write(&tmp_path, data)?;
        std::fs::rename(&tmp_path, path)?;
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

    fn sync_collection_files(&self) -> Result<()> {
        Self::sync_path_recursively(&self.path)
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
            let mut missing_ids = Vec::new();
            let mut missing_fields = Vec::new();

            for (i, &user_id) in ids.iter().enumerate() {
                if self.reverse_id_map.contains_key(&user_id) {
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
                missing_ids.push(user_id);
                missing_fields.push(seg.fields.get(i).cloned().unwrap_or_default());
            }

            if !missing_ids.is_empty() {
                self.append_recovered_items(&missing_vectors, &missing_ids, &missing_fields)?;
                recovered_any = true;
            }
        }

        // Clean WAL after successful recovery
        self.wal.cleanup()?;
        Ok(recovered_any)
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

    /// Load id_map from id_map.bin. Partial trailing bytes are ignored.
    fn load_id_map_file(path: &Path) -> Vec<u64> {
        let id_map_path = path.join("id_map.bin");
        if id_map_path.exists() {
            std::fs::read(&id_map_path)
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

    fn write_id_map(path: &Path, ids: &[u64]) -> Result<()> {
        let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        std::fs::write(path.join("id_map.bin"), &id_bytes)?;
        Ok(())
    }

    /// Initialize id_map before WAL replay.
    ///
    /// If id_map exists, it is treated as the durable row boundary. Extra vector
    /// rows beyond it are truncated so WAL replay can safely re-apply them.
    /// If id_map is missing, we backfill sequential IDs for legacy data.
    fn initialize_id_map_for_open(&mut self) -> Result<bool> {
        let id_map_path = self.path.join("id_map.bin");
        let id_map_exists = id_map_path.exists();
        let (n_vecs, _) = self.vector_store.get_shape()?;
        let mut id_map = Self::load_id_map_file(&self.path);
        let mut repaired = false;

        if id_map_exists {
            let n_vecs = n_vecs as usize;
            if id_map.len() < n_vecs {
                self.vector_store.truncate_to_vectors(id_map.len())?;
                repaired = true;
            } else if id_map.len() > n_vecs {
                id_map.truncate(n_vecs);
                Self::write_id_map(&self.path, &id_map)?;
                repaired = true;
            }
        } else {
            while id_map.len() < n_vecs as usize {
                id_map.push(id_map.len() as u64);
            }
            if !id_map.is_empty() {
                Self::write_id_map(&self.path, &id_map)?;
            }
        }

        self.reverse_id_map = Self::build_reverse_id_map(&id_map);
        self.id_map = id_map;

        Ok(repaired)
    }

    /// Append new user IDs to id_map.bin (sequential write, no rewrite).
    fn append_id_map(path: &Path, ids: &[u64]) -> Result<()> {
        use std::io::Write;
        let id_map_path = path.join("id_map.bin");
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&id_map_path)?;
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

    /// Check if a user ID exists in the collection.
    pub fn is_id_exists(&self, user_id: u64) -> bool {
        self.reverse_id_map.contains_key(&user_id)
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
        let mut set = self.tombstone.write();
        for &id in ids {
            set.insert(id);
        }
        let path = self.path.join("tombstone.bin");
        Self::save_tombstone_to_disk(&path, &set)
    }

    /// Undelete previously soft-deleted vectors.
    pub fn restore_items(&self, ids: &[u64]) -> Result<()> {
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

    /// Filter tombstoned IDs from (ids, dists) pairs.
    fn filter_tombstoned(&self, ids: Vec<u64>, dists: Vec<f32>) -> (Vec<u64>, Vec<f32>) {
        let set = self.tombstone.read();
        if set.is_empty() {
            return (ids, dists);
        }
        ids.into_iter()
            .zip(dists)
            .filter(|(id, _)| !set.contains(id))
            .unzip()
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
        self.field_store.batch_store_at_ids(&row_ids, fields)?;

        if let Some(ref mut idx) = self.index {
            idx.insert(vectors, n_vectors, dim, &row_ids)?;
        }

        for (i, &user_id) in ids.iter().enumerate() {
            self.id_map.push(user_id);
            self.reverse_id_map.insert(user_id, start_row + i);
        }
        Self::append_id_map(&self.path, ids)?;

        Ok(())
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
        let dim = self.meta.dimension;
        if vectors.len() != n_vectors * dim {
            return Err(LynseError::DimensionMismatch {
                expected: dim,
                got: vectors.len() / n_vectors,
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
        self.validate_new_ids(ids)?;
        let start_row = self.validate_row_boundary()?;
        let row_ids: Vec<u64> = (start_row as u64..start_row as u64 + n_vectors as u64).collect();

        // Write to WAL first for crash safety
        let wal_fields: Vec<HashMap<String, serde_json::Value>> = match fields {
            Some(fl) => fl.to_vec(),
            None => Vec::new(),
        };
        self.wal.write_log_data(vectors, dim, ids, &wal_fields)?;

        // Write vectors to main storage (single flat file, auto re-mmap)
        self.vector_store.write(vectors)?;

        // Store fields only when provided (skip empty metadata to avoid slow per-row I/O)
        if let Some(field_list) = fields {
            self.field_store.batch_store_at_ids(&row_ids, field_list)?;
        }

        // Insert into index if exists (HNSW/IVF — Flat types use mmap directly)
        if let Some(ref mut idx) = self.index {
            idx.insert(vectors, n_vectors, dim, &row_ids)?;
        }

        // Update in-memory id_map and persist
        for (i, &user_id) in ids.iter().enumerate() {
            self.id_map.push(user_id);
            self.reverse_id_map.insert(user_id, start_row + i);
        }
        Self::append_id_map(&self.path, ids)?;

        Ok(())
    }

    /// Flush pending WAL bytes and fsync collection files without clearing WAL.
    pub fn flush(&self) -> Result<()> {
        self.wal.flush()?;
        self.sync_collection_files()
    }

    /// Checkpoint durable state and clear WAL after data has reached main storage.
    pub fn checkpoint(&self) -> Result<()> {
        self.flush()?;
        self.wal.cleanup()?;
        self.sync_collection_files()
    }

    /// Close the collection handle from an API perspective.
    ///
    /// The writer lock is released when the `Collection` object itself is
    /// dropped; this method makes outstanding state durable and stops the WAL.
    pub fn close(&mut self) -> Result<()> {
        self.checkpoint()?;
        self.wal.stop()?;
        self.lock.take();
        Ok(())
    }

    /// Commit: checkpoint main storage and clear WAL after successful writes.
    pub fn commit(&self) -> Result<()> {
        self.checkpoint()
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
    /// For HNSW/IVF types: loads data from mmap and builds the index structure.
    pub fn build_index(&mut self, index_type: &str) -> Result<()> {
        let dim = self.meta.dimension;
        let upper = index_type.to_uppercase();

        // Clear any previously built quantizer indices
        self.pq_index = None;
        self.rabitq_index = None;
        self.polarvec_index = None;

        let is_flat = Self::resolve_metric_from_type(index_type).is_some();

        let n_vectors = if is_flat {
            // Flat family: clear any HNSW/IVF tree index, and remove stale graph index.bin
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
                    let guard = self.vector_store.read_mmap()?;
                    let fm = guard.as_ref().ok_or(LynseError::EmptyDatabase)?;
                    let pq = PQIndex::build(fm.as_slice(), n, dim, n_subspaces);
                    drop(guard);
                    let pq_path = self.path.join("pq_index.bin");
                    pq.save(&pq_path)
                        .map_err(|e| LynseError::Storage(e.to_string()))?;
                    self.pq_index = Some(pq);
                }
            } else if upper.contains("RABITQ") {
                if n > 0 {
                    let guard = self.vector_store.read_mmap()?;
                    let fm = guard.as_ref().ok_or(LynseError::EmptyDatabase)?;
                    let rq = RaBitQIndex::build(fm.as_slice(), n, dim);
                    drop(guard);
                    let rq_path = self.path.join("rabitq_index.bin");
                    rq.save(&rq_path)
                        .map_err(|e| LynseError::Storage(e.to_string()))?;
                    self.rabitq_index = Some(rq);
                }
            } else if upper.contains("POLARVEC") {
                if n > 0 {
                    let bits = parse_bits(&upper, POLARVEC_DEFAULT_BITS);
                    let guard = self.vector_store.read_mmap()?;
                    let fm = guard.as_ref().ok_or(LynseError::EmptyDatabase)?;
                    let pv = PolarVecIndex::build(fm.as_slice(), n, dim, bits);
                    drop(guard);
                    let pv_path = self.path.join("polarvec_index.bin");
                    pv.save(&pv_path)
                        .map_err(|e| LynseError::Storage(e.to_string()))?;
                    self.polarvec_index = Some(pv);
                }
            }
            n
        } else {
            // HNSW/IVF: need data to build
            let guard = self.vector_store.read_mmap()?;
            let fm = guard.as_ref().ok_or(LynseError::EmptyDatabase)?;
            let n = fm.len();
            if n == 0 {
                return Err(LynseError::EmptyDatabase);
            }

            let all_data = fm.as_slice();
            let ids: Vec<u64> = (0..n as u64).collect();

            let mut idx = index::create_index(index_type)?;
            idx.build(all_data, n, dim, Some(&ids))?;

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
    /// Returns Some(metric) for Flat-family types, None for HNSW/IVF (which build a separate index).
    fn resolve_metric_from_type(index_type: &str) -> Option<DistanceMetric> {
        let upper = index_type.to_uppercase();
        // Flat family: no separate index structure needed
        if upper.starts_with("FLAT") || upper == "FLAT" {
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
    /// Supports: FLAT, FLAT-L2, FLAT-COS, FLAT-IP-SQ8, IVF-L2-SQ8, etc.
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
            // Default: Inner Product ("FLAT", "FLAT-IP-SQ8", "IVF", etc.)
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
    /// 1. If an HNSW/IVF index exists → use it
    /// 2. Otherwise → zero-copy mmap brute-force via VectorStore's FlatMmap
    /// 3. Filtered search → brute-force with subset filtering on mmap data
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        where_expr: Option<&str>,
        nprobe: usize,
    ) -> Result<SearchResult> {
        let dim = self.meta.dimension;
        if query.len() != dim {
            return Err(LynseError::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        }

        // Apply field filter
        let subset_indices = if let Some(filter) = where_expr {
            Some(self.field_store.query(filter)?)
        } else {
            None
        };

        let search_params = SearchParams {
            k,
            nprobe,
            ef_search: None,
            subset_indices,
        };

        let (result_ids, result_dists) = if let Some(ref idx) = self.index {
            // HNSW/IVF index path
            idx.search(query, k, &search_params)?
        } else if let Some(ref pq) = self.pq_index {
            // PQ (Product Quantization) two-pass search
            let metric = self.resolve_metric();
            let guard = self.vector_store.read_mmap()?;
            match guard.as_ref() {
                None => (vec![], vec![]),
                Some(fm) => {
                    let (indices, dists) =
                        pq.search(query, k, fm.as_slice(), metric, PQ_OVERSAMPLE);
                    let ids: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
                    (ids, dists)
                }
            }
        } else if let Some(ref rq) = self.rabitq_index {
            // RaBitQ binary two-pass search
            let metric = self.resolve_metric();
            let guard = self.vector_store.read_mmap()?;
            match guard.as_ref() {
                None => (vec![], vec![]),
                Some(fm) => {
                    let (indices, dists) =
                        rq.search(query, k, fm.as_slice(), metric, RABITQ_OVERSAMPLE);
                    let ids: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
                    (ids, dists)
                }
            }
        } else if let Some(ref pv) = self.polarvec_index {
            // PolarVec multi-bit LUT two-pass search
            let metric = self.resolve_metric();
            let guard = self.vector_store.read_mmap()?;
            match guard.as_ref() {
                None => (vec![], vec![]),
                Some(fm) => {
                    let (indices, dists) =
                        pv.search(query, k, fm.as_slice(), metric, POLARVEC_OVERSAMPLE);
                    let ids: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
                    (ids, dists)
                }
            }
        } else if search_params.subset_indices.is_some() {
            // Filtered brute-force on mmap data
            self.brute_force_search_filtered(query, k, &search_params)?
        } else {
            // Unfiltered: zero-copy mmap + fused parallel topk (~5ms for 1M×128)
            let metric = self.resolve_metric();
            let guard = self.vector_store.read_mmap()?;
            match guard.as_ref() {
                None => {
                    return Ok(SearchResult {
                        ids: Vec::new(),
                        distances: Vec::new(),
                        fields: Vec::new(),
                        index_mode: self.index_mode.clone().unwrap_or("flat-ip".into()),
                        dimension: dim,
                        k,
                    });
                }
                Some(fm) => {
                    let use_sq8 = self.resolve_use_sq8();
                    let (indices, dists) = fm.search(query, k, metric, use_sq8);
                    let ids: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
                    (ids, dists)
                }
            }
        };

        let result_ids: Vec<u64> = result_ids
            .iter()
            .map(|&row| self.row_to_user_id(row))
            .collect();
        let (result_ids, result_dists) = self.filter_tombstoned(result_ids, result_dists);

        Ok(SearchResult {
            ids: result_ids,
            distances: result_dists,
            fields: Vec::new(),
            index_mode: self.index_mode.clone().unwrap_or("FLAT".into()),
            dimension: dim,
            k,
        })
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

        // Pre-compute filter once (shared across all queries)
        let subset_indices = if let Some(filter) = where_expr {
            Some(self.field_store.query(filter)?)
        } else {
            None
        };

        use rayon::prelude::*;

        let results: Vec<Result<SearchResult>> = (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let start = i * dim;
                let query = &queries[start..start + dim];

                let search_params = SearchParams {
                    k,
                    nprobe,
                    ef_search: None,
                    subset_indices: subset_indices.clone(),
                };

                let (result_ids, result_dists) = if let Some(ref idx) = self.index {
                    idx.search(query, k, &search_params)?
                } else if let Some(ref pq) = self.pq_index {
                    let metric = self.resolve_metric();
                    let guard = self.vector_store.read_mmap()?;
                    match guard.as_ref() {
                        None => (vec![], vec![]),
                        Some(fm) => {
                            let (indices, dists) =
                                pq.search(query, k, fm.as_slice(), metric, PQ_OVERSAMPLE);
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
                            let (indices, dists) =
                                rq.search(query, k, fm.as_slice(), metric, RABITQ_OVERSAMPLE);
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
                            let (indices, dists) =
                                pv.search(query, k, fm.as_slice(), metric, POLARVEC_OVERSAMPLE);
                            let ids: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
                            (ids, dists)
                        }
                    }
                } else if search_params.subset_indices.is_some() {
                    self.brute_force_search_filtered(query, k, &search_params)?
                } else {
                    let metric = self.resolve_metric();
                    let guard = self.vector_store.read_mmap()?;
                    match guard.as_ref() {
                        None => (vec![], vec![]),
                        Some(fm) => {
                            let use_sq8 = self.resolve_use_sq8();
                            let (indices, dists) = fm.search(query, k, metric, use_sq8);
                            let ids: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
                            (ids, dists)
                        }
                    }
                };

                let result_ids: Vec<u64> = result_ids
                    .iter()
                    .map(|&row| self.row_to_user_id(row))
                    .collect();
                let (result_ids, result_dists) = self.filter_tombstoned(result_ids, result_dists);

                Ok(SearchResult {
                    ids: result_ids,
                    distances: result_dists,
                    fields: Vec::new(),
                    index_mode: self.index_mode.clone().unwrap_or("FLAT".into()),
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

        let guard = self.vector_store.read_mmap()?;
        let fm = match guard.as_ref() {
            None => return Ok((Vec::new(), Vec::new())),
            Some(fm) => fm,
        };

        let subset = match &params.subset_indices {
            Some(s) => s,
            None => {
                // No filter — use unfiltered path
                let use_sq8 = self.resolve_use_sq8();
                let (indices, dists) = fm.search(query, k, metric, use_sq8);
                let ids: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
                return Ok((ids, dists));
            }
        };

        if subset.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let (indices, dists) = fm.search_filtered(query, k, metric, subset);
        let ids: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
        Ok((ids, dists))
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
        let all_data = fm.as_slice();
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
        let all_data = fm.as_slice();
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
        let (indices, dists) = fm.search(query, search_k, metric, use_sq8);

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
        let subset = self.field_store.query(where_expr)?;
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
                    let _ = fm.search(query, k, metric, use_sq8);
                }
                let t4 = std::time::Instant::now();
                for _ in 0..iterations {
                    let _ = fm.search(query, k, metric, use_sq8);
                }
                let us4 = t4.elapsed().as_micros() as f64 / iterations as f64;
                results.push(("unfiltered_mmap".to_string(), us4, k));
            }
        }

        Ok(results)
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

        let (mut all_vectors, n_total) = {
            let guard = self.vector_store.read_mmap()?;
            match guard.as_ref() {
                None => (Vec::new(), 0usize),
                Some(fm) => (fm.as_slice().to_vec(), fm.len()),
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
                    field_rows.push(row as u64);
                    field_values.push(field_list[i].clone());
                }
            }
        }

        self.vector_store.replace_data(&all_vectors)?;
        Self::write_id_map(&self.path, &new_id_map)?;
        self.id_map = new_id_map;
        self.reverse_id_map = new_reverse_id_map;

        if fields.is_some() {
            self.field_store
                .replace_fields_at_ids(&field_rows, &field_values)?;
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
        self.vector_store.get_shape()
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
    /// up-to-date (re-mmap'd after every write). Sync only applies to HNSW/IVF.
    pub fn sync_index(&mut self) -> Result<()> {
        if !self.needs_index_sync() {
            return Ok(());
        }

        let current_fp = self.vector_store.fingerprint().unwrap_or_default();
        let dim = self.meta.dimension;

        if let Some(ref mut idx) = self.index {
            let guard = self.vector_store.read_mmap()?;
            if let Some(fm) = guard.as_ref() {
                let all_data = fm.as_slice();
                let n_vectors = fm.len();

                if n_vectors > 0 {
                    let ids: Vec<u64> = (0..n_vectors as u64).collect();
                    idx.build(all_data, n_vectors, dim, Some(&ids))?;

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
        let dim = self.meta.dimension;
        let guard = self.vector_store.read_mmap()?;
        let (data, user_ids) = match guard.as_ref() {
            None => (Vec::new(), Vec::new()),
            Some(fm) => {
                let total = fm.len();
                let take = n.min(total);
                let data = fm.as_slice()[..take * dim].to_vec();
                let user_ids: Vec<u64> = (0..take as u64)
                    .map(|row| self.row_to_user_id(row))
                    .collect();
                (data, user_ids)
            }
        };
        drop(guard);

        let row_offsets: Vec<u64> = (0..user_ids.len() as u64).collect();
        let fields = self.field_store.retrieve_many(&row_offsets)?;
        Ok((data, user_ids, fields))
    }

    /// Return the last `n` vectors with their user IDs and field metadata.
    pub fn tail(
        &self,
        n: usize,
    ) -> Result<(Vec<f32>, Vec<u64>, Vec<HashMap<String, serde_json::Value>>)> {
        let dim = self.meta.dimension;
        let guard = self.vector_store.read_mmap()?;
        let (data, user_ids, row_start) = match guard.as_ref() {
            None => (Vec::new(), Vec::new(), 0usize),
            Some(fm) => {
                let total = fm.len();
                let take = n.min(total);
                let start = total - take;
                let data = fm.as_slice()[start * dim..].to_vec();
                let user_ids: Vec<u64> = (start as u64..total as u64)
                    .map(|row| self.row_to_user_id(row))
                    .collect();
                (data, user_ids, start)
            }
        };
        drop(guard);

        let n_actual = user_ids.len();
        let row_offsets: Vec<u64> =
            (row_start as u64..row_start as u64 + n_actual as u64).collect();
        let fields = self.field_store.retrieve_many(&row_offsets)?;
        Ok((data, user_ids, fields))
    }

    /// Query field metadata with a SQL-like filter. Returns matching user IDs.
    pub fn query_fields(&self, where_expr: &str) -> Result<Vec<u64>> {
        let row_ids = self.field_store.query(where_expr)?;
        Ok(row_ids
            .iter()
            .map(|&row| self.row_to_user_id(row))
            .collect())
    }

    /// Query field metadata with a SQL-like filter, returning both IDs and fields
    /// in a single ApexBase query. Eliminates the two-query pattern.
    pub fn query_with_fields(
        &self,
        where_expr: &str,
    ) -> Result<(Vec<u64>, Vec<HashMap<String, serde_json::Value>>)> {
        let (row_ids, fields) = self.field_store.query_with_fields(where_expr)?;
        let user_ids = row_ids
            .iter()
            .map(|&row| self.row_to_user_id(row))
            .collect();
        Ok((user_ids, fields))
    }

    /// Retrieve field metadata for specific user IDs.
    pub fn retrieve_fields(&self, ids: &[u64]) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        let row_offsets: Vec<u64> = ids
            .iter()
            .map(|&uid| self.user_id_to_row(uid).unwrap_or(uid as usize) as u64)
            .collect();
        self.field_store.retrieve_many(&row_offsets)
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
        let dim = self.meta.dimension;
        let guard = self.vector_store.read_mmap()?;

        let mut result_data: Vec<f32> = vec![0.0f32; ids.len() * dim];
        let row_offsets: Vec<u64> = ids
            .iter()
            .map(|&uid| self.user_id_to_row(uid).unwrap_or(uid as usize) as u64)
            .collect();

        if let Some(fm) = guard.as_ref() {
            let all_data = fm.as_slice();
            let n_total = fm.len();

            for (out_pos, &row) in row_offsets.iter().enumerate() {
                let row = row as usize;
                if row < n_total {
                    let src_start = row * dim;
                    let src_end = src_start + dim;
                    if src_end <= all_data.len() {
                        let dst = &mut result_data[out_pos * dim..(out_pos + 1) * dim];
                        dst.copy_from_slice(&all_data[src_start..src_end]);
                    }
                }
            }
        }
        drop(guard);

        let fields = self.field_store.retrieve_many(&row_offsets)?;
        Ok((result_data, fields))
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
        let dim = self.meta.dimension;
        if query.len() != dim {
            return Err(LynseError::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        }

        let metric = self.resolve_metric();
        let ascending = metric.is_ascending();

        let guard = self.vector_store.read_mmap()?;
        let fm = match guard.as_ref() {
            None => return Ok((Vec::new(), Vec::new())),
            Some(fm) => fm,
        };

        let all_data = fm.as_slice();
        let n_vectors = fm.len();
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
        drop(guard);

        if ascending {
            result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        result.truncate(max_results);
        let (ids, dists) = result.into_iter().unzip();
        Ok((ids, dists))
    }

    /// Compact the collection by physically removing all tombstoned vectors.
    ///
    /// Process:
    /// 1. Collect live (non-tombstoned) vectors and their user IDs.
    /// 2. Atomically rewrite `vectors.bin` with only live vectors.
    /// 3. Rebuild `id_map.bin` and in-memory maps.
    /// 4. Clear the tombstone set.
    /// 5. Rebuild the index (if one exists) on the compacted data.
    ///
    /// Returns the number of vectors physically removed.
    pub fn compact(&mut self) -> Result<usize> {
        let dim = self.meta.dimension;

        let tombstoned_user_ids: HashSet<u64> = self.tombstone.read().clone();
        if tombstoned_user_ids.is_empty() {
            return Ok(0);
        }

        // Build set of tombstoned row offsets
        let tombstoned_rows: HashSet<usize> = tombstoned_user_ids
            .iter()
            .filter_map(|&uid| self.user_id_to_row(uid))
            .collect();

        // Read all vectors
        let (all_vectors, n_total) = {
            let guard = self.vector_store.read_mmap()?;
            match guard.as_ref() {
                None => return Ok(0),
                Some(fm) => (fm.as_slice().to_vec(), fm.len()),
            }
        };

        let n_removed = tombstoned_rows.len();

        // Collect live vectors and their user IDs
        let mut live_vectors: Vec<f32> = Vec::with_capacity((n_total - n_removed) * dim);
        let mut live_user_ids: Vec<u64> = Vec::new();

        for row in 0..n_total {
            if tombstoned_rows.contains(&row) {
                continue;
            }
            let start = row * dim;
            live_vectors.extend_from_slice(&all_vectors[start..start + dim]);
            live_user_ids.push(self.row_to_user_id(row as u64));
        }

        // Atomically rewrite vectors.bin
        self.vector_store.replace_data(&live_vectors)?;

        // Rebuild in-memory id_map and persist
        self.reverse_id_map = live_user_ids
            .iter()
            .enumerate()
            .map(|(i, &uid)| (uid, i))
            .collect();
        self.id_map = live_user_ids.clone();

        let id_bytes: Vec<u8> = live_user_ids
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect();
        std::fs::write(self.path.join("id_map.bin"), &id_bytes)?;

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

        Ok(n_removed)
    }

    /// Delete the entire collection from disk.
    pub fn delete(self) -> Result<()> {
        if self.path.exists() {
            std::fs::remove_dir_all(&self.path)?;
        }
        Ok(())
    }
}

/// Search result container, maps to Python's `Result` class.
#[derive(Debug)]
pub struct SearchResult {
    pub ids: Vec<u64>,
    pub distances: Vec<f32>,
    pub fields: Vec<HashMap<String, serde_json::Value>>,
    pub index_mode: String,
    pub dimension: usize,
    pub k: usize,
}

/// Database manager: manages multiple collections within a single database.
pub struct DatabaseEngine {
    root_path: PathBuf,
    collections: Arc<RwLock<HashMap<String, Arc<RwLock<Collection>>>>>,
    _lock: FileLock,
}

impl DatabaseEngine {
    /// Open or create a database at the given root path.
    pub fn open(root_path: &Path) -> Result<Self> {
        std::fs::create_dir_all(root_path)?;
        let lock = FileLock::exclusive(&root_path.join(".database.lock"))?;

        let engine = Self {
            root_path: root_path.to_path_buf(),
            collections: Arc::new(RwLock::new(HashMap::new())),
            _lock: lock,
        };

        // Scan for existing collections
        engine.scan_collections()?;

        Ok(engine)
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
                                    let _ =
                                        self.get_or_open_collection(&name, dim as usize, 100_000);
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
        let mut collections = self.collections.write();

        if let Some(coll) = collections.get(name) {
            return Ok(Arc::clone(coll));
        }

        let collection = Collection::open(&self.root_path, name, dimension, chunk_size)?;
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
        let coll_path = self.root_path.join(name);
        if coll_path.exists() {
            return Err(LynseError::CollectionAlreadyExists(name.to_string()));
        }
        self.get_or_open_collection(name, dimension, chunk_size)
    }

    /// Drop a collection.
    pub fn drop_collection(&self, name: &str) -> Result<()> {
        let mut collections = self.collections.write();

        if let Some(coll) = collections.remove(name) {
            let coll_guard = coll.write();
            let path = coll_guard.path.clone();
            drop(coll_guard);
            drop(coll);
            if path.exists() {
                std::fs::remove_dir_all(&path)?;
            }
        }
        Ok(())
    }

    /// List all collection names.
    pub fn list_collections(&self) -> Vec<String> {
        self.collections.read().keys().cloned().collect()
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

        let result = coll.search(&[0.0, 1.0, 0.0, 0.0], 2, None, 10).unwrap();
        assert_eq!(result.ids, vec![10]);
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
            coll.build_index("HNSW").unwrap();

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
        assert_eq!(coll.get_index_mode(), Some("HNSW"));
        let result = coll.search(&[0.0, 1.0, 0.0, 0.0], 1, None, 10).unwrap();
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
}

/// Top-level manager for multiple databases.
/// Each database is a directory containing collections.
pub struct DatabaseManager {
    root_path: PathBuf,
    databases: RwLock<HashMap<String, DatabaseEngine>>,
    _lock: FileLock,
}

impl DatabaseManager {
    /// Open the database manager at the given root path.
    pub fn new(root_path: &Path) -> Result<Self> {
        std::fs::create_dir_all(root_path)?;
        let lock = FileLock::exclusive(&root_path.join(".manager.lock"))?;
        let mgr = Self {
            root_path: root_path.to_path_buf(),
            databases: RwLock::new(HashMap::new()),
            _lock: lock,
        };
        // Scan for existing databases
        mgr.scan_databases();
        Ok(mgr)
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
        let mut dbs = self.databases.write();
        dbs.remove(name);

        let db_path = self.root_path.join(name);
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
        let mut dbs = self.databases.write();
        if dbs.contains_key(name) {
            return Ok(());
        }
        let db_path = self.root_path.join(name);
        if !db_path.exists() {
            return Err(LynseError::DatabaseNotFound(name.to_string()));
        }
        let engine = DatabaseEngine::open(&db_path)?;
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
        // Ensure database is open
        self.get_or_open_database(db_name)?;

        if drop_if_exists {
            self.with_database(db_name, |engine| {
                let _ = engine.drop_collection(collection_name);
                Ok(())
            })?;
        }

        self.with_database(db_name, |engine| {
            let _ = engine.get_or_open_collection(collection_name, dim, chunk_size)?;
            Ok(())
        })?;

        // Save collection config
        self.save_collection_config(db_name, collection_name, dim, chunk_size, description)?;
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
            });
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

    /// Drop a collection from a database.
    pub fn drop_collection(&self, db_name: &str, collection_name: &str) -> Result<()> {
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
