//! High-performance vector storage with single-file mmap.
//!
//! Architecture:
//! - **Single file**: all vectors in `vectors.bin` (raw f32 LE, contiguous)
//! - **Append-only writes**: O(new_data) raw file append — no mmap involved
//! - **Lazy mmap reads**: FlatMmap created on first read, invalidated on write
//! - **No write amplification**: data written once, searched directly via mmap
//!
//! Write path: raw byte append to file (fast, no mmap overhead).
//! Read path: lazy FlatMmap creation (zero-copy mmap on first search).
//! This avoids the costly munmap+mmap cycle on every write batch.

use crate::error::{LynseError, Result};
use crate::storage::flat_mmap::FlatMmap;
use parking_lot::RwLock;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// ─── Atomic I/O helpers ─────────────────────────────────────────────────────

/// Write data atomically: write to `.tmp`, rename over target.
fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    let tmp_path = path.with_extension("tmp");
    std::fs::write(&tmp_path, data)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

// ─── VectorStore ────────────────────────────────────────────────────────────

/// High-performance vector store backed by a single mmap'd binary file.
///
/// - **Writes**: raw f32 byte append to file (no mmap involvement)
/// - **Reads**: lazy FlatMmap creation, zero-copy `&[f32]` slice
/// - **Search**: fused parallel SIMD distance + per-thread heap (via FlatMmap)
/// - **Concurrent**: RwLock for mmap cache — multiple readers, exclusive writer
pub struct VectorStore {
    collection_path: PathBuf,
    dimension: usize,
    /// Path to the single vectors file.
    vectors_path: PathBuf,
    /// Cached FlatMmap for reads. None = stale (needs re-mmap on next read).
    /// Write path sets this to None; read path lazily creates it.
    mmap_cache: RwLock<Option<FlatMmap>>,
    /// In-memory vector count (avoids file stat on shape queries).
    total_vectors: AtomicU64,
    fingerprint: RwLock<Option<String>>,
}

impl VectorStore {
    /// Create or open a vector store at the given path.
    pub fn new(collection_path: &Path, dimension: usize, _chunk_size: usize) -> Result<Self> {
        std::fs::create_dir_all(collection_path)?;

        let vectors_path = collection_path.join("vectors.bin");

        // Compute total vectors from file size (if file exists)
        let total_vectors = if vectors_path.exists() {
            let file_len = std::fs::metadata(&vectors_path)
                .map(|m| m.len())
                .unwrap_or(0);
            file_len / (dimension as u64 * 4)
        } else {
            0
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
            vectors_path,
            mmap_cache: RwLock::new(None), // lazy — created on first read
            total_vectors: AtomicU64::new(total_vectors),
            fingerprint: RwLock::new(fingerprint),
        })
    }

    /// Get vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the collection path.
    pub fn collection_path(&self) -> &Path {
        &self.collection_path
    }

    /// Get the path to the vectors binary file (for zero-copy mmap access).
    pub fn vectors_path(&self) -> &Path {
        &self.vectors_path
    }

    /// Get the current fingerprint.
    pub fn fingerprint(&self) -> Option<String> {
        self.fingerprint.read().clone()
    }

    // ── Core write path (raw file append — NO mmap) ─────────────────────

    /// Append vectors to the flat file. Does NOT re-mmap.
    ///
    /// The mmap cache is invalidated; it will be lazily re-created on next read.
    /// This makes writes fast: just a file append + counter update.
    pub fn write(&self, data: &[f32]) -> Result<()> {
        let n_vectors = data.len() / self.dimension;
        if data.len() % self.dimension != 0 {
            return Err(LynseError::DimensionMismatch {
                expected: self.dimension,
                got: data.len() % self.dimension,
            });
        }
        if n_vectors == 0 {
            return Ok(());
        }

        // Raw byte append — no mmap involved
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        {
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.vectors_path)?;
            file.write_all(bytes)?;
            file.flush()?;
        }

        // Update in-memory counter
        let new_total = self
            .total_vectors
            .fetch_add(n_vectors as u64, Ordering::Relaxed)
            + n_vectors as u64;

        // Invalidate mmap cache (will be re-created on next read)
        *self.mmap_cache.write() = None;

        // Update info.json (atomic) — for backward compat with scan_collections
        let info_path = self.collection_path.join("info.json");
        let info = serde_json::json!({
            "total_shape": [new_total, self.dimension]
        });
        atomic_write(
            &info_path,
            serde_json::to_string(&info)
                .map_err(|e| LynseError::Serialization(e.to_string()))?
                .as_bytes(),
        )?;

        // Update fingerprint in-memory
        let new_fp = uuid::Uuid::new_v4().to_string().replace("-", "");
        *self.fingerprint.write() = Some(new_fp);

        Ok(())
    }

    // ── Core read path (lazy mmap creation) ─────────────────────────────

    /// Ensure the mmap cache is populated. Called before any read operation.
    ///
    /// Uses double-checked locking: fast read-lock check, then write-lock create.
    fn ensure_mmap(&self) -> Result<()> {
        // Fast path: mmap already cached
        {
            let guard = self.mmap_cache.read();
            if guard.is_some() {
                return Ok(());
            }
        }
        // Slow path: create mmap
        let mut guard = self.mmap_cache.write();
        if guard.is_none() {
            let total = self.total_vectors.load(Ordering::Relaxed);
            if total > 0 {
                let fm = FlatMmap::open(&self.vectors_path, self.dimension)
                    .map_err(|e| LynseError::Storage(format!("FlatMmap open error: {}", e)))?;
                *guard = Some(fm);
            }
        }
        Ok(())
    }

    /// Get read access to the underlying FlatMmap.
    ///
    /// Lazily creates the mmap if needed (first read after write).
    /// Returns an RAII guard — multiple readers can hold this concurrently.
    pub fn read_mmap(&self) -> Result<parking_lot::RwLockReadGuard<'_, Option<FlatMmap>>> {
        self.ensure_mmap()?;
        Ok(self.mmap_cache.read())
    }

    /// Replace ALL stored vectors with new data (used by compaction).
    /// Atomically writes new data, resets the vector count, and invalidates the mmap cache.
    pub fn replace_data(&self, data: &[f32]) -> Result<()> {
        let n_vectors = if self.dimension > 0 {
            data.len() / self.dimension
        } else {
            0
        };

        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        atomic_write(&self.vectors_path, bytes)?;

        self.total_vectors
            .store(n_vectors as u64, Ordering::Relaxed);
        *self.mmap_cache.write() = None;

        // Update info.json
        let info_path = self.collection_path.join("info.json");
        let info = serde_json::json!({ "total_shape": [n_vectors, self.dimension] });
        atomic_write(
            &info_path,
            serde_json::to_string(&info)
                .map_err(|e| LynseError::Serialization(e.to_string()))?
                .as_bytes(),
        )?;

        Ok(())
    }

    /// Truncate the vector file to the first `n_vectors` rows.
    ///
    /// Used during WAL recovery when `id_map.bin` proves that a previous write
    /// reached the vector file but did not fully reach the commit boundary.
    pub fn truncate_to_vectors(&self, n_vectors: usize) -> Result<()> {
        let target_len = n_vectors as u64 * self.dimension as u64 * 4;
        let current_len = std::fs::metadata(&self.vectors_path)
            .map(|m| m.len())
            .unwrap_or(0);
        if target_len > current_len {
            return Err(LynseError::Storage(format!(
                "cannot truncate vectors.bin to {} bytes; current length is {} bytes",
                target_len, current_len
            )));
        }

        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(&self.vectors_path)?;
        file.set_len(target_len)?;
        file.sync_all()?;

        self.total_vectors
            .store(n_vectors as u64, Ordering::Relaxed);
        *self.mmap_cache.write() = None;

        let info_path = self.collection_path.join("info.json");
        let info = serde_json::json!({ "total_shape": [n_vectors, self.dimension] });
        atomic_write(
            &info_path,
            serde_json::to_string(&info)
                .map_err(|e| LynseError::Serialization(e.to_string()))?
                .as_bytes(),
        )?;

        let new_fp = uuid::Uuid::new_v4().to_string().replace("-", "");
        *self.fingerprint.write() = Some(new_fp);

        Ok(())
    }

    /// Check if any vector data exists.
    pub fn file_exists(&self) -> bool {
        self.total_vectors.load(Ordering::Relaxed) > 0
    }

    /// Get total shape (n_vectors, dimension).
    pub fn get_shape(&self) -> Result<(u64, usize)> {
        Ok((self.total_vectors.load(Ordering::Relaxed), self.dimension))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_write_read_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let store = VectorStore::new(tmp.path(), 4, 100).unwrap();

        // Write 10 vectors of dim 4
        let data: Vec<f32> = (0..40).map(|i| i as f32).collect();
        store.write(&data).unwrap();

        assert_eq!(store.get_shape().unwrap(), (10, 4));

        // Read via mmap
        let guard = store.read_mmap().unwrap();
        let fm = guard.as_ref().unwrap();
        assert_eq!(fm.as_slice(), &data[..]);
        drop(guard);

        // Append 5 more
        let data2: Vec<f32> = (40..60).map(|i| i as f32).collect();
        store.write(&data2).unwrap();
        assert_eq!(store.get_shape().unwrap(), (15, 4));

        // Read all — single contiguous file
        let guard = store.read_mmap().unwrap();
        let fm = guard.as_ref().unwrap();
        let expected: Vec<f32> = (0..60).map(|i| i as f32).collect();
        assert_eq!(fm.as_slice(), &expected[..]);
    }

    #[test]
    fn test_search_via_mmap() {
        use crate::distance::DistanceMetric;

        let tmp = TempDir::new().unwrap();
        let store = VectorStore::new(tmp.path(), 4, 100).unwrap();

        // Write 5 vectors
        let data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // vec 0
            0.0, 1.0, 0.0, 0.0, // vec 1
            0.5, 0.5, 0.0, 0.0, // vec 2
            0.0, 0.0, 1.0, 0.0, // vec 3
            0.0, 0.0, 0.0, 1.0, // vec 4
        ];
        store.write(&data).unwrap();

        // Search via FlatMmap (zero-copy)
        let guard = store.read_mmap().unwrap();
        let fm = guard.as_ref().unwrap();
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let (ids, dists) = fm.search(&query, 2, DistanceMetric::InnerProduct, false, None);
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], 0); // exact match (IP=1.0)
        assert!((dists[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_many_small_writes() {
        let tmp = TempDir::new().unwrap();
        let store = VectorStore::new(tmp.path(), 4, 100).unwrap();

        // Simulate many small flushes (like insert_session)
        for i in 0..100 {
            let data: Vec<f32> = (0..40).map(|j| (i * 40 + j) as f32).collect();
            store.write(&data).unwrap();
        }

        assert_eq!(store.get_shape().unwrap(), (1000, 4));

        // Read all — single mmap creation
        let guard = store.read_mmap().unwrap();
        let fm = guard.as_ref().unwrap();
        assert_eq!(fm.len(), 1000);
    }
}
