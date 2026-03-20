//! High-performance vector storage powered by NumPack 0.6.0.
//!
//! Features:
//! - **NumPack ParallelIO**: batch save_arrays, in-place append_rows, mmap load_array
//! - **mmap reads**: automatic LRU cache inside NumPack, invalidated on write
//! - **In-place append**: O(new_data) — no file rewrite for appending rows
//! - **Concurrent R/W**: safe via Mutex<ParallelIO>
//! - Chunk IDs computed arithmetically — no filesystem scan on write path

use crate::error::{LynseError, Result};
use ndarray::{ArrayD, IxDyn};
use numpack::{DataType, ParallelIO};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Chunk metadata for tracking vector storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMeta {
    pub chunk_id: u32,
    pub start_id: u64,
    pub end_id: u64,
    pub n_vectors: u64,
}

/// ID mapping for chunk-based storage, equivalent to Python's `IDMapper`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdMapper {
    /// Map from chunk filename stem → (start_id, end_id)
    entries: BTreeMap<String, (u64, u64)>,
}

impl IdMapper {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    pub fn add_entry(&mut self, filename: &str, start_id: u64, end_id: u64) {
        self.entries.insert(filename.to_string(), (start_id, end_id));
    }

    pub fn get_entry(&self, filename: &str) -> Option<(u64, u64)> {
        self.entries.get(filename).copied()
    }

    /// Generate IDs for a given chunk as a contiguous range.
    pub fn generate_ids(&self, filename: &str) -> Option<Vec<u64>> {
        self.entries.get(filename).map(|&(start, end)| {
            (start..=end).collect()
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let data = bincode::serialize(self)
            .map_err(|e| LynseError::Serialization(e.to_string()))?;
        atomic_write(path, &data)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        let mapper: Self = bincode::deserialize(&data)
            .map_err(|e| LynseError::Serialization(e.to_string()))?;
        Ok(mapper)
    }
}

// ─── Atomic I/O helpers ─────────────────────────────────────────────────────

/// Write data atomically: write to `.tmp`, rename over target.
/// Rename is atomic on all supported filesystems — readers never see a partial file.
/// Call `VectorStore::flush()` explicitly for full durability.
fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    let tmp_path = path.with_extension("tmp");
    std::fs::write(&tmp_path, data)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

// ─── VectorStore ────────────────────────────────────────────────────────────

/// High-performance vector store powered by NumPack 0.6.0.
///
/// - **Writes**: `save_arrays` (batch, parallel) + `append_rows` (in-place O(n))
/// - **Reads**: `load_array` (mmap with automatic LRU cache)
/// - **Concurrent**: Mutex<ParallelIO> for safe single-writer access
pub struct VectorStore {
    collection_path: PathBuf,
    dimension: usize,
    chunk_size: usize,
    id_mapper: Arc<RwLock<IdMapper>>,
    fingerprint: Arc<RwLock<Option<String>>>,
    /// NumPack ParallelIO — handles mmap caching, integrity, and parallel I/O
    npk_io: Arc<Mutex<ParallelIO>>,
}

impl VectorStore {
    /// Create or open a vector store at the given path.
    pub fn new(collection_path: &Path, dimension: usize, chunk_size: usize) -> Result<Self> {
        let chunk_path = collection_path.join("chunk_data");
        std::fs::create_dir_all(&chunk_path)?;

        // Initialize NumPack ParallelIO (handles mmap cache + metadata internally)
        let npk_io = ParallelIO::new(chunk_path.clone())
            .map_err(|e| LynseError::Storage(format!("NumPack init error: {}", e)))?;

        // Load or create ID mapper
        let id_mapper_path = collection_path.join("id_mapper.bin");
        let id_mapper = if id_mapper_path.exists() {
            IdMapper::load(&id_mapper_path)?
        } else {
            IdMapper::new()
        };

        // Load fingerprint
        let fp_path = collection_path.join("fingerprint");
        let fingerprint = if fp_path.exists() {
            let content = std::fs::read_to_string(&fp_path)?;
            content.lines().last().map(|s| s.trim().to_string())
        } else {
            None
        };

        Ok(Self {
            collection_path: collection_path.to_path_buf(),
            dimension,
            chunk_size,
            id_mapper: Arc::new(RwLock::new(id_mapper)),
            fingerprint: Arc::new(RwLock::new(fingerprint)),
            npk_io: Arc::new(Mutex::new(npk_io)),
        })
    }

    /// Get vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get chunk size limit.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get the collection path.
    pub fn collection_path(&self) -> &Path {
        &self.collection_path
    }

    /// Get the current fingerprint.
    pub fn fingerprint(&self) -> Option<String> {
        self.fingerprint.read().clone()
    }

    /// Update the fingerprint after a write operation.
    pub fn update_fingerprint(&self) -> Result<()> {
        let new_fp = uuid::Uuid::new_v4().to_string().replace("-", "");
        let fp_path = self.collection_path.join("fingerprint");

        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&fp_path)?;
        writeln!(f, "{}", new_fp)?;

        *self.fingerprint.write() = Some(new_fp);
        Ok(())
    }

    // ── Chunk file helpers ──────────────────────────────────────────────

    /// Get all chunk filenames (stems, e.g. `"chunk_0"`) sorted by ID.
    /// Uses NumPack's list_arrays for metadata-driven enumeration.
    pub fn get_all_files(&self) -> Vec<String> {
        let io = self.npk_io.lock().unwrap();
        Self::sorted_chunk_names(&io)
    }

    /// Get sorted chunk names from a locked ParallelIO (avoids re-lock).
    fn sorted_chunk_names(io: &ParallelIO) -> Vec<String> {
        let mut names: Vec<String> = io
            .list_arrays()
            .into_iter()
            .filter(|n| n.starts_with("chunk_"))
            .collect();
        names.sort_unstable_by_key(|f| {
            f.strip_prefix("chunk_")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0)
        });
        names
    }

    /// Check if any data files exist.
    pub fn file_exists(&self) -> bool {
        !self.get_all_files().is_empty()
    }

    // ── Core write path (NumPack 0.6.0 batch + append) ─────────────────

    /// Write vector data via NumPack ParallelIO.
    ///
    /// - **Append**: `append_rows` for filling the last chunk (in-place, O(new_data))
    /// - **New chunks**: `save_arrays` batch call (parallel I/O)
    /// - Zero-copy ArrayD from borrowed `&[f32]` via unsafe Vec wrapping
    pub fn write(&self, data: &[f32]) -> Result<()> {
        let n_vectors = data.len() / self.dimension;
        if data.len() % self.dimension != 0 {
            return Err(LynseError::DimensionMismatch {
                expected: self.dimension,
                got: data.len() % self.dimension,
            });
        }

        // Read current total shape
        let info_path = self.collection_path.join("info.json");
        let total_rows: u64 = if info_path.exists() {
            let content = std::fs::read_to_string(&info_path)?;
            let info: serde_json::Value = serde_json::from_str(&content)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            info["total_shape"][0].as_u64().unwrap_or(0)
        } else {
            0
        };

        let chunk_size = self.chunk_size;
        let dim = self.dimension;
        let mut offset = 0usize;

        // Compute current chunk state from total_rows
        let (mut next_chunk_id, space_in_current) = if total_rows == 0 {
            (0u64, 0usize)
        } else {
            let last_id = (total_rows - 1) / chunk_size as u64;
            let used = total_rows as usize - (last_id as usize * chunk_size);
            (last_id, chunk_size - used)
        };

        // Hold the NumPack lock once for all chunk operations
        let io = self.npk_io.lock()
            .map_err(|e| LynseError::Storage(format!("lock error: {}", e)))?;

        // 1) Append to last chunk if space remains — in-place O(new_data)
        if space_in_current > 0 && n_vectors > 0 {
            let take = n_vectors.min(space_in_current);
            let append_data = &data[..take * dim];
            let stem = format!("chunk_{}", next_chunk_id);

            // Zero-copy ArrayD from borrowed slice
            let array = Self::borrow_as_array(append_data, take, dim);
            io.append_rows(&stem, &array)
                .map_err(|e| LynseError::Storage(format!("NumPack append error: {}", e)))?;
            Self::forget_array(array);

            offset += take;
            if offset < n_vectors {
                next_chunk_id += 1;
            }
        } else if total_rows > 0 {
            next_chunk_id += 1;
        }

        // 2) Collect remaining new chunks for batch save_arrays
        let mut arrays_to_save: Vec<(String, ArrayD<f32>, DataType)> = Vec::new();

        while offset < n_vectors {
            let take = (n_vectors - offset).min(chunk_size);
            let chunk_data = &data[offset * dim..(offset + take) * dim];
            let stem = format!("chunk_{}", next_chunk_id);

            let array = Self::borrow_as_array(chunk_data, take, dim);
            arrays_to_save.push((stem, array, DataType::Float32));
            offset += take;
            next_chunk_id += 1;
        }

        // 3) Single batch save_arrays call — enables NumPack parallel writes
        if !arrays_to_save.is_empty() {
            io.save_arrays(&arrays_to_save)
                .map_err(|e| LynseError::Storage(format!("NumPack write error: {}", e)))?;
        }

        // 4) Prevent double-free: forget all borrowed arrays
        for (_, array, _) in arrays_to_save.drain(..) {
            Self::forget_array(array);
        }

        // 5) Build ID mapper while still holding the lock
        //    (sync_metadata deferred — numpack syncs on drop or explicit call)
        Self::rebuild_id_mapper_locked(&io, &self.id_mapper)?;

        // Release NumPack lock
        drop(io);

        // Update total shape (atomic)
        let new_total = total_rows + n_vectors as u64;
        let info = serde_json::json!({
            "total_shape": [new_total, self.dimension]
        });
        atomic_write(
            &info_path,
            serde_json::to_string(&info)
                .map_err(|e| LynseError::Serialization(e.to_string()))?
                .as_bytes(),
        )?;

        // Fingerprint: update in-memory
        let new_fp = uuid::Uuid::new_v4().to_string().replace("-", "");
        *self.fingerprint.write() = Some(new_fp);

        Ok(())
    }

    /// Create a zero-copy `ArrayD<f32>` wrapping borrowed `&[f32]` data.
    ///
    /// SAFETY: The caller MUST call `forget_array()` after the array is no longer
    /// needed, to prevent double-free of the borrowed data.
    #[inline]
    fn borrow_as_array(data: &[f32], n_rows: usize, dim: usize) -> ArrayD<f32> {
        unsafe {
            let vec = Vec::from_raw_parts(data.as_ptr() as *mut f32, data.len(), data.len());
            ArrayD::from_shape_vec_unchecked(IxDyn(&[n_rows, dim]), vec)
        }
    }

    /// Forget the internal Vec of an ArrayD to prevent freeing borrowed data.
    #[inline]
    fn forget_array(array: ArrayD<f32>) {
        let (vec, _) = array.into_raw_vec_and_offset();
        std::mem::forget(vec);
    }

    // ── Core read path (NumPack mmap with LRU cache) ────────────────────

    /// Load a chunk via NumPack's mmap-based `load_array` (automatic LRU cache).
    pub fn load_chunk_npk(&self, stem: &str) -> Result<Vec<f32>> {
        let io = self.npk_io.lock()
            .map_err(|e| LynseError::Storage(format!("lock error: {}", e)))?;
        let array: ArrayD<f32> = io.load_array(stem)
            .map_err(|e| LynseError::Storage(format!("NumPack load error: {}", e)))?;
        let (vec, _) = array.into_raw_vec_and_offset();
        Ok(vec)
    }

    /// Memory-mapped read of a chunk (alias for load_chunk_npk).
    pub fn mmap_read(&self, stem: &str) -> Result<Vec<f32>> {
        self.load_chunk_npk(stem)
    }

    // ── ID mapper ───────────────────────────────────────────────────────

    /// Rebuild ID mapper using a locked ParallelIO (no re-lock).
    fn rebuild_id_mapper_locked(io: &ParallelIO, id_mapper: &RwLock<IdMapper>) -> Result<()> {
        let mut mapper = id_mapper.write();
        *mapper = IdMapper::new();

        let names = Self::sorted_chunk_names(io);
        let mut start_id: u64 = 0;
        for stem in &names {
            let shape = io.get_shape(stem)
                .map_err(|e| LynseError::Storage(format!("NumPack shape error: {}", e)))?;
            let nv = shape[0];
            if nv > 0 {
                mapper.add_entry(stem, start_id, start_id + nv - 1);
                start_id += nv;
            }
        }
        Ok(())
    }

    /// Update the ID mapper from NumPack metadata and persist to disk.
    pub fn update_id_map(&self) -> Result<()> {
        let io = self.npk_io.lock()
            .map_err(|e| LynseError::Storage(format!("lock error: {}", e)))?;
        Self::rebuild_id_mapper_locked(&io, &self.id_mapper)?;
        drop(io);

        let mapper_path = self.collection_path.join("id_mapper.bin");
        self.id_mapper.read().save(&mapper_path)?;
        Ok(())
    }

    /// Get total shape [n_vectors, dimension].
    pub fn get_shape(&self) -> Result<(u64, usize)> {
        let info_path = self.collection_path.join("info.json");
        if info_path.exists() {
            let content = std::fs::read_to_string(&info_path)?;
            let info: serde_json::Value = serde_json::from_str(&content)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            let rows = info["total_shape"][0].as_u64().unwrap_or(0);
            Ok((rows, self.dimension))
        } else {
            Ok((0, self.dimension))
        }
    }

    /// Get a clone of the current ID mapper.
    pub fn id_mapper(&self) -> IdMapper {
        self.id_mapper.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_id_mapper() {
        let mut mapper = IdMapper::new();
        mapper.add_entry("chunk_0", 0, 99);
        mapper.add_entry("chunk_1", 100, 199);

        assert_eq!(mapper.get_entry("chunk_0"), Some((0, 99)));
        let ids = mapper.generate_ids("chunk_0").unwrap();
        assert_eq!(ids.len(), 100);
        assert_eq!(ids[0], 0);
        assert_eq!(ids[99], 99);
    }

    #[test]
    fn test_write_read_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let store = VectorStore::new(tmp.path(), 4, 100).unwrap();

        // Write 10 vectors of dim 4
        let data: Vec<f32> = (0..40).map(|i| i as f32).collect();
        store.write(&data).unwrap();

        assert_eq!(store.get_shape().unwrap(), (10, 4));
        let loaded = store.load_chunk_npk("chunk_0").unwrap();
        assert_eq!(loaded, data);

        // Append 5 more (uses append_rows in-place)
        let data2: Vec<f32> = (40..60).map(|i| i as f32).collect();
        store.write(&data2).unwrap();
        assert_eq!(store.get_shape().unwrap(), (15, 4));

        // Read back — chunk_0 should contain all 15 vectors
        let loaded_all = store.load_chunk_npk("chunk_0").unwrap();
        let expected: Vec<f32> = (0..60).map(|i| i as f32).collect();
        assert_eq!(loaded_all, expected);
    }

    #[test]
    fn test_chunk_boundary() {
        let tmp = TempDir::new().unwrap();
        // chunk_size=10, dim=4 → chunk_0 holds 10 vectors (40 floats)
        let store = VectorStore::new(tmp.path(), 4, 10).unwrap();

        // Write 10 vectors → fills chunk_0
        let data: Vec<f32> = (0..40).map(|i| i as f32).collect();
        store.write(&data).unwrap();
        assert_eq!(store.get_shape().unwrap(), (10, 4));

        // Write 5 more → creates chunk_1
        let data2: Vec<f32> = (40..60).map(|i| i as f32).collect();
        store.write(&data2).unwrap();
        assert_eq!(store.get_shape().unwrap(), (15, 4));

        // Verify two chunks exist
        let files = store.get_all_files();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0], "chunk_0");
        assert_eq!(files[1], "chunk_1");
    }

    #[test]
    fn test_numpack_load_array() {
        let tmp = TempDir::new().unwrap();
        let store = VectorStore::new(tmp.path(), 4, 100).unwrap();

        let data: Vec<f32> = (0..40).map(|i| i as f32).collect();
        store.write(&data).unwrap();

        // NumPack load_array (mmap with automatic cache)
        let loaded = store.load_chunk_npk("chunk_0").unwrap();
        assert_eq!(loaded, data);

        // Second load should use NumPack's internal mmap cache
        let loaded2 = store.load_chunk_npk("chunk_0").unwrap();
        assert_eq!(loaded2, data);
    }
}
