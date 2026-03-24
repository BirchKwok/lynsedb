//! Database engine module.
//!
//! Orchestrates collections, storage, indexing, and search operations.
//! This is the main entry point for all database operations from Python.

use crate::distance::DistanceMetric;
use crate::error::{LynseError, Result};
use crate::index::{self, SearchParams, VectorIndex};
use crate::storage::field_store::FieldStore;
use crate::storage::flat_mmap::FlatMmap;
use crate::storage::vector_store::VectorStore;
use crate::storage::wal::WALStorage;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    /// Ultra-fast mmap-backed flat index for non-quantized Flat types.
    /// When set, search bypasses the in-memory FlatIndex entirely.
    flat_mmap: Option<(FlatMmap, DistanceMetric)>,
}

impl Collection {
    /// Get the collection path.
    pub fn collection_path(&self) -> &Path {
        &self.path
    }

    /// Create or open a collection.
    pub fn open(
        path: &Path,
        name: &str,
        dimension: usize,
        chunk_size: usize,
    ) -> Result<Self> {
        let collection_path = path.join(name);
        std::fs::create_dir_all(&collection_path)?;

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

        // Try to open existing FlatMmap if index is a non-quantized Flat type
        let flat_mmap = Self::try_open_flat_mmap(&collection_path, dimension, index_mode.as_deref());

        // Load last sync fingerprint
        let last_sync_fingerprint = Self::load_sync_fingerprint(&collection_path);

        let meta = CollectionMeta {
            name: name.to_string(),
            dimension,
            chunk_size,
            index_mode: index_mode.clone(),
            dtypes: "float32".to_string(),
        };

        let mut coll = Self {
            meta,
            path: collection_path,
            vector_store,
            field_store,
            wal,
            index,
            index_mode,
            last_sync_fingerprint,
            flat_mmap,
        };

        // Recover any uncommitted WAL data on startup
        coll.recover_wal()?;

        Ok(coll)
    }

    /// Recover uncommitted WAL segments into main storage.
    fn recover_wal(&mut self) -> Result<()> {
        if !self.wal.has_uncommitted_data() {
            return Ok(());
        }

        let segments = self.wal.get_segments()?;
        for seg in &segments {
            self.vector_store.write(&seg.data)?;
            if !seg.fields.is_empty() {
                self.field_store.batch_store(&seg.fields)?;
            }
        }

        // Clean WAL after successful recovery
        self.wal.cleanup()?;
        Ok(())
    }

    /// Load the last sync fingerprint from disk.
    fn load_sync_fingerprint(collection_path: &Path) -> Option<String> {
        let fp_path = collection_path.join("sync_fingerprint");
        std::fs::read_to_string(&fp_path).ok().map(|s| s.trim().to_string())
    }

    /// Save the sync fingerprint to disk.
    fn save_sync_fingerprint(&self) -> Result<()> {
        if let Some(ref fp) = self.last_sync_fingerprint {
            let fp_path = self.path.join("sync_fingerprint");
            std::fs::write(&fp_path, fp)?;
        }
        Ok(())
    }

    /// Check if an index type string is a non-quantized Flat type (eligible for FlatMmap).
    fn is_flat_mmap_type(index_type: &str) -> Option<DistanceMetric> {
        match index_type {
            "Flat-IP" | "flat-ip" | "FLAT" => Some(DistanceMetric::InnerProduct),
            "Flat-L2" | "flat-l2" => Some(DistanceMetric::L2Squared),
            "Flat-Cos" | "flat-cosine" => Some(DistanceMetric::Cosine),
            _ => None,
        }
    }

    /// Path for the consolidated flat mmap vector file.
    fn flat_mmap_path(collection_path: &Path) -> PathBuf {
        collection_path.join("flat_vectors.bin")
    }

    /// Try to open an existing FlatMmap from disk (used during Collection::open).
    fn try_open_flat_mmap(
        collection_path: &Path,
        dimension: usize,
        index_mode: Option<&str>,
    ) -> Option<(FlatMmap, DistanceMetric)> {
        let metric = index_mode.and_then(Self::is_flat_mmap_type)?;
        let fpath = Self::flat_mmap_path(collection_path);
        if !fpath.exists() {
            return None;
        }
        FlatMmap::open(&fpath, dimension).ok().map(|fm| (fm, metric))
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
        let meta: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| LynseError::Serialization(e.to_string()))?;

        let index_type = meta["index_type"]
            .as_str()
            .ok_or_else(|| LynseError::Storage("Missing index_type in metadata".into()))?;

        let index_data_file = index_path.join("index.bin");
        if !index_data_file.exists() {
            return Ok((None, Some(index_type.to_string())));
        }

        let data = std::fs::read(&index_data_file)?;
        let mut idx = index::create_index(index_type)?;
        idx.deserialize(&data)?;

        Ok((Some(idx), Some(index_type.to_string())))
    }

    /// Add vectors with optional field metadata.
    /// Data is first written to WAL for crash safety, then to main storage.
    pub fn add_items(
        &mut self,
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

        // Write to WAL first for crash safety
        let wal_fields: Vec<HashMap<String, serde_json::Value>> = match fields {
            Some(fl) => fl.to_vec(),
            None => Vec::new(),
        };
        self.wal.write_log_data(vectors, dim, &wal_fields)?;

        // Write vectors to main storage
        self.vector_store.write(vectors)?;

        // Store fields only when provided (skip empty metadata to avoid slow per-row I/O)
        if let Some(field_list) = fields {
            self.field_store.batch_store(field_list)?;
        }

        // Append to FlatMmap if active (keeps mmap in sync with storage)
        if let Some((ref mut fm, _)) = self.flat_mmap {
            fm.write(vectors)
                .map_err(|e| LynseError::Storage(format!("FlatMmap append error: {}", e)))?;
        }

        // Insert into index if exists
        if let Some(ref mut idx) = self.index {
            let shape = self.vector_store.get_shape()?;
            let start_id = shape.0 - n_vectors as u64;
            let ids: Vec<u64> = (start_id..start_id + n_vectors as u64).collect();
            idx.insert(vectors, n_vectors, dim, &ids)?;
        }

        Ok(())
    }

    /// Commit: clear WAL after successful writes.
    pub fn commit(&self) -> Result<()> {
        self.wal.cleanup()
    }

    /// Check if there is uncommitted WAL data.
    pub fn has_uncommitted_data(&self) -> bool {
        self.wal.has_uncommitted_data()
    }

    /// Build or rebuild the index.
    pub fn build_index(&mut self, index_type: &str) -> Result<()> {
        let dim = self.meta.dimension;

        // Load all vectors from storage
        let all_files = self.vector_store.get_all_files();
        let mut all_data: Vec<f32> = Vec::new();

        for filename in &all_files {
            let chunk = self.vector_store.load_chunk_npk(filename)?;
            all_data.extend_from_slice(&chunk);
        }

        let n_vectors = all_data.len() / dim;
        if n_vectors == 0 {
            return Err(LynseError::EmptyDatabase);
        }

        // For non-quantized Flat types, use FlatMmap (zero-copy mmap + fused parallel topk).
        // This avoids storing all vectors in memory AND avoids the 512MB clone per search.
        if let Some(metric) = Self::is_flat_mmap_type(index_type) {
            let fpath = Self::flat_mmap_path(&self.path);
            // Remove old file to avoid appending to stale data
            let _ = std::fs::remove_file(&fpath);
            let mut fm = FlatMmap::open(&fpath, dim)
                .map_err(|e| LynseError::Storage(format!("FlatMmap open error: {}", e)))?;
            fm.write(&all_data)
                .map_err(|e| LynseError::Storage(format!("FlatMmap write error: {}", e)))?;

            self.flat_mmap = Some((fm, metric));
            // No in-memory FlatIndex needed
            self.index = None;
        } else {
            // Clear flat_mmap if switching to a non-flat index type
            self.flat_mmap = None;

            let ids: Vec<u64> = (0..n_vectors as u64).collect();

            // Create and build index
            let mut idx = index::create_index(index_type)?;
            idx.build(&all_data, n_vectors, dim, Some(&ids))?;

            // Save index
            let index_data = idx.serialize()?;
            let index_path = self.path.join("index");
            std::fs::create_dir_all(&index_path)?;
            std::fs::write(index_path.join("index.bin"), &index_data)?;

            self.index = Some(idx);
        }

        // Save metadata
        let meta_path = self.path.join("index_meta");
        std::fs::create_dir_all(&meta_path)?;
        let meta = serde_json::json!({
            "index_type": index_type,
            "n_vectors": n_vectors,
            "dimension": dim,
        });
        std::fs::write(
            meta_path.join("index_metadata.json"),
            serde_json::to_string_pretty(&meta)
                .map_err(|e| LynseError::Serialization(e.to_string()))?,
        )?;

        self.index_mode = Some(index_type.to_string());
        self.meta.index_mode = self.index_mode.clone();

        Ok(())
    }

    /// Search for nearest neighbors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        search_filter: Option<&str>,
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
        let subset_indices = if let Some(filter) = search_filter {
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

        // Use FlatMmap (fastest path) → regular index → brute-force fallback
        let (result_ids, result_dists) = if let Some((ref fm, metric)) = self.flat_mmap {
            if search_params.subset_indices.is_some() {
                // Filtered search: fall back to chunk-based brute-force
                self.brute_force_search(query, k, &search_params)?
            } else {
                // Unfiltered FlatMmap: zero-copy mmap + fused parallel topk
                let (indices, dists) = fm.search(query, k, metric);
                let ids: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
                (ids, dists)
            }
        } else if let Some(ref idx) = self.index {
            idx.search(query, k, &search_params)?
        } else {
            // Brute-force search across all chunks
            self.brute_force_search(query, k, &search_params)?
        };

        // Fields are NOT retrieved during search for performance.
        // Use Collection::retrieve_fields() for lazy loading when needed.

        Ok(SearchResult {
            ids: result_ids,
            distances: result_dists,
            fields: Vec::new(),
            index_mode: self.index_mode.clone().unwrap_or("flat-ip".into()),
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
        search_filter: Option<&str>,
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
        let subset_indices = if let Some(filter) = search_filter {
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

                let (result_ids, result_dists) = if let Some((ref fm, metric)) = self.flat_mmap {
                    if search_params.subset_indices.is_some() {
                        self.brute_force_search(query, k, &search_params)?
                    } else {
                        let (indices, dists) = fm.search(query, k, metric);
                        let ids: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
                        (ids, dists)
                    }
                } else if let Some(ref idx) = self.index {
                    idx.search(query, k, &search_params)?
                } else {
                    self.brute_force_search(query, k, &search_params)?
                };

                Ok(SearchResult {
                    ids: result_ids,
                    distances: result_dists,
                    fields: Vec::new(),
                    index_mode: self.index_mode.clone().unwrap_or("flat-ip".into()),
                    dimension: dim,
                    k,
                })
            })
            .collect();

        // Collect results, propagating any errors
        results.into_iter().collect()
    }

    /// Resolve the distance metric from the index_mode string.
    fn resolve_metric(&self) -> DistanceMetric {
        self.index_mode
            .as_ref()
            .and_then(|m| {
                if m.contains("l2") {
                    Some(DistanceMetric::L2Squared)
                } else if m.contains("cosine") || m.contains("cos") {
                    Some(DistanceMetric::Cosine)
                } else if m.contains("hamming") {
                    Some(DistanceMetric::Hamming)
                } else if m.contains("jaccard") {
                    Some(DistanceMetric::Jaccard)
                } else {
                    Some(DistanceMetric::InnerProduct)
                }
            })
            .unwrap_or(DistanceMetric::InnerProduct)
    }

    /// Brute-force search when no index is built.
    ///
    /// Streaming approach: search each chunk independently with batch distances +
    /// quickselect, then merge top-k across chunks. Avoids copying all data into
    /// a single Vec.
    fn brute_force_search(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        let dim = self.meta.dimension;
        let metric = self.resolve_metric();
        let ascending = metric.is_ascending();

        let all_files = self.vector_store.get_all_files();
        let id_mapper = self.vector_store.id_mapper();

        // Pre-compute subset filter HashSet once (if any)
        let subset_set: Option<std::collections::HashSet<u64>> = params
            .subset_indices
            .as_ref()
            .map(|s| s.iter().cloned().collect());

        let mut best_ids: Vec<u64> = Vec::new();
        let mut best_dists: Vec<f32> = Vec::new();

        for filename in &all_files {
            let chunk = self.vector_store.load_chunk_npk(filename)?;

            if let Some(ids) = id_mapper.generate_ids(filename) {
                if let Some(ref sset) = subset_set {
                    // Filtered: extract matching vectors
                    let mut filtered_data: Vec<f32> = Vec::new();
                    let mut filtered_ids: Vec<u64> = Vec::new();
                    for (i, &id) in ids.iter().enumerate() {
                        if sset.contains(&id) {
                            let start = i * dim;
                            filtered_data.extend_from_slice(&chunk[start..start + dim]);
                            filtered_ids.push(id);
                        }
                    }
                    if filtered_ids.is_empty() {
                        continue;
                    }
                    // Search filtered chunk
                    let (top_idx, top_dist) =
                        crate::distance::top_k_search(query, &filtered_data, dim, k, metric);
                    let chunk_ids: Vec<u64> = top_idx.iter().map(|&i| filtered_ids[i as usize]).collect();
                    // Merge with running best
                    let (merged_ids, merged_dists) = crate::distance::top_k_heap_merge(
                        &best_ids, &best_dists, &top_dist, &chunk_ids, k, ascending,
                    );
                    best_ids = merged_ids;
                    best_dists = merged_dists;
                } else {
                    // Unfiltered: search directly on chunk data (zero extra copy)
                    let (top_idx, top_dist) =
                        crate::distance::top_k_search(query, &chunk, dim, k, metric);
                    let chunk_ids: Vec<u64> = top_idx.iter().map(|&i| ids[i as usize]).collect();
                    // Merge with running best
                    let (merged_ids, merged_dists) = crate::distance::top_k_heap_merge(
                        &best_ids, &best_dists, &top_dist, &chunk_ids, k, ascending,
                    );
                    best_ids = merged_ids;
                    best_dists = merged_dists;
                }
            }
        }

        Ok((best_ids, best_dists))
    }

    /// Update vectors: atomic delete + insert for given IDs.
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

        // Delete old vectors from index
        if let Some(ref mut idx) = self.index {
            idx.delete(ids)?;
        }

        // Re-insert with new data
        if let Some(ref mut idx) = self.index {
            idx.insert(vectors, n_vectors, dim, ids)?;
        }

        // Write new vectors to storage
        self.vector_store.write(vectors)?;

        // Store fields if provided
        if let Some(field_list) = fields {
            self.field_store.batch_store(field_list)?;
        }

        Ok(())
    }

    /// Remove the index.
    pub fn remove_index(&mut self) -> Result<()> {
        self.index = None;
        self.flat_mmap = None;
        self.index_mode = None;
        self.meta.index_mode = None;
        self.last_sync_fingerprint = None;

        let index_path = self.path.join("index");
        let meta_path = self.path.join("index_meta");
        let flat_mmap_file = Self::flat_mmap_path(&self.path);
        if index_path.exists() {
            std::fs::remove_dir_all(&index_path)?;
        }
        if meta_path.exists() {
            std::fs::remove_dir_all(&meta_path)?;
        }
        if flat_mmap_file.exists() {
            std::fs::remove_file(&flat_mmap_file)?;
        }

        Ok(())
    }

    /// Get collection shape (n_vectors, dimension).
    pub fn shape(&self) -> Result<(u64, usize)> {
        self.vector_store.get_shape()
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
            None => self.index.is_some() || self.flat_mmap.is_some(),
        }
    }

    /// Incrementally sync the index with any new data since last sync.
    /// Only rebuilds if the storage fingerprint has changed.
    pub fn sync_index(&mut self) -> Result<()> {
        if !self.needs_index_sync() {
            return Ok(());
        }

        let current_fp = self.vector_store.fingerprint().unwrap_or_default();
        let dim = self.meta.dimension;

        // FlatMmap path: rebuild the consolidated flat file
        if let Some((ref mut fm, _metric)) = self.flat_mmap {
            let all_files = self.vector_store.get_all_files();
            let mut all_data: Vec<f32> = Vec::new();
            for filename in &all_files {
                let chunk = self.vector_store.load_chunk_npk(filename)?;
                all_data.extend_from_slice(&chunk);
            }
            if !all_data.is_empty() {
                // Truncate and rewrite
                let fpath = Self::flat_mmap_path(&self.path);
                let _ = std::fs::remove_file(&fpath);
                let mut new_fm = FlatMmap::open(&fpath, dim)
                    .map_err(|e| LynseError::Storage(format!("FlatMmap open error: {}", e)))?;
                new_fm.write(&all_data)
                    .map_err(|e| LynseError::Storage(format!("FlatMmap write error: {}", e)))?;
                *fm = new_fm;
            }
        } else if let Some(ref mut idx) = self.index {
            let all_files = self.vector_store.get_all_files();
            let mut all_data: Vec<f32> = Vec::new();
            for filename in &all_files {
                let chunk = self.vector_store.load_chunk_npk(filename)?;
                all_data.extend_from_slice(&chunk);
            }

            let n_vectors = all_data.len() / dim;
            if n_vectors > 0 {
                let ids: Vec<u64> = (0..n_vectors as u64).collect();
                idx.build(&all_data, n_vectors, dim, Some(&ids))?;

                let index_data = idx.serialize()?;
                let index_path = self.path.join("index");
                std::fs::write(index_path.join("index.bin"), &index_data)?;
            }
        }

        self.last_sync_fingerprint = Some(current_fp);
        self.save_sync_fingerprint()?;
        Ok(())
    }

    /// Return the first `n` vectors with their field metadata.
    ///
    /// Returns `(flat_f32_data, dimension, fields)`.
    pub fn head(&self, n: usize) -> Result<(Vec<f32>, Vec<HashMap<String, serde_json::Value>>)> {
        let dim = self.meta.dimension;
        let all_files = self.vector_store.get_all_files();
        let id_mapper = self.vector_store.id_mapper();

        let mut data: Vec<f32> = Vec::new();
        let mut ids: Vec<u64> = Vec::new();
        let mut count = 0usize;

        for filename in &all_files {
            let chunk = self.vector_store.load_chunk_npk(filename)?;
            let n_vecs = chunk.len() / dim;
            let take = n_vecs.min(n - count);

            data.extend_from_slice(&chunk[..take * dim]);

            if let Some(chunk_ids) = id_mapper.generate_ids(filename) {
                ids.extend_from_slice(&chunk_ids[..take]);
            }

            count += take;
            if count >= n {
                break;
            }
        }

        let fields = self.field_store.retrieve_many(&ids)?;
        Ok((data, fields))
    }

    /// Return the last `n` vectors with their field metadata.
    pub fn tail(&self, n: usize) -> Result<(Vec<f32>, Vec<HashMap<String, serde_json::Value>>)> {
        let dim = self.meta.dimension;
        let all_files = self.vector_store.get_all_files();
        let id_mapper = self.vector_store.id_mapper();

        let mut data: Vec<f32> = Vec::new();
        let mut ids: Vec<u64> = Vec::new();
        let mut count = 0usize;

        for filename in all_files.iter().rev() {
            let chunk = self.vector_store.load_chunk_npk(filename)?;
            let n_vecs = chunk.len() / dim;
            let take = n_vecs.min(n - count);
            let skip = n_vecs - take;

            // Prepend to front (reverse order iteration)
            let mut prefix = chunk[skip * dim..].to_vec();
            prefix.extend_from_slice(&data);
            data = prefix;

            if let Some(chunk_ids) = id_mapper.generate_ids(filename) {
                let mut prefix_ids = chunk_ids[skip..].to_vec();
                prefix_ids.extend_from_slice(&ids);
                ids = prefix_ids;
            }

            count += take;
            if count >= n {
                break;
            }
        }

        let fields = self.field_store.retrieve_many(&ids)?;
        Ok((data, fields))
    }

    /// Query field metadata with a SQL-like filter. Returns matching external IDs.
    pub fn query_fields(&self, filter_expr: &str) -> Result<Vec<u64>> {
        self.field_store.query(filter_expr)
    }

    /// Retrieve field metadata for specific IDs.
    pub fn retrieve_fields(
        &self,
        ids: &[u64],
    ) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        self.field_store.retrieve_many(ids)
    }

    /// List all field names in the collection.
    pub fn list_fields(&self) -> Result<Vec<String>> {
        self.field_store.list_fields()
    }

    /// Get the current index mode string.
    pub fn get_index_mode(&self) -> Option<&str> {
        self.index_mode.as_deref()
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

/// Database manager: manages multiple collections.
pub struct DatabaseEngine {
    root_path: PathBuf,
    collections: Arc<RwLock<HashMap<String, Arc<RwLock<Collection>>>>>,
}

impl DatabaseEngine {
    /// Open or create a database at the given root path.
    pub fn open(root_path: &Path) -> Result<Self> {
        std::fs::create_dir_all(root_path)?;

        let engine = Self {
            root_path: root_path.to_path_buf(),
            collections: Arc::new(RwLock::new(HashMap::new())),
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
                                    let _ = self.get_or_open_collection(
                                        &name,
                                        dim as usize,
                                        100_000,
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
        self.collections.read().contains_key(name)
            || self.root_path.join(name).exists()
    }

    /// Get the root path.
    pub fn root_path(&self) -> &Path {
        &self.root_path
    }
}
