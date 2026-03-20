//! Database engine module.
//!
//! Orchestrates collections, storage, indexing, and search operations.
//! This is the main entry point for all database operations from Python.

use crate::distance::DistanceMetric;
use crate::error::{LynseError, Result};
use crate::index::{self, SearchParams, VectorIndex};
use crate::storage::field_store::FieldStore;
use crate::storage::vector_store::VectorStore;
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

/// A single vector collection, managing vectors + fields + index.
pub struct Collection {
    pub meta: CollectionMeta,
    path: PathBuf,
    vector_store: VectorStore,
    field_store: FieldStore,
    index: Option<Box<dyn VectorIndex>>,
    index_mode: Option<String>,
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

        // Try to load existing index
        let index_path = collection_path.join("index");
        let index_meta_path = collection_path.join("index_meta");
        std::fs::create_dir_all(&index_path)?;
        std::fs::create_dir_all(&index_meta_path)?;

        let (index, index_mode) = Self::try_load_index(&index_meta_path, &index_path)?;

        let meta = CollectionMeta {
            name: name.to_string(),
            dimension,
            chunk_size,
            index_mode: index_mode.clone(),
            dtypes: "float32".to_string(),
        };

        Ok(Self {
            meta,
            path: collection_path,
            vector_store,
            field_store,
            index,
            index_mode,
        })
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

        // Write vectors
        self.vector_store.write(vectors)?;

        // Store fields only when provided (skip empty metadata to avoid slow per-row I/O)
        if let Some(field_list) = fields {
            self.field_store.batch_store(field_list)?;
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

        let ids: Vec<u64> = (0..n_vectors as u64).collect();

        // Create and build index
        let mut idx = index::create_index(index_type)?;
        idx.build(&all_data, n_vectors, dim, Some(&ids))?;

        // Save index
        let index_data = idx.serialize()?;
        let index_path = self.path.join("index");
        std::fs::create_dir_all(&index_path)?;
        std::fs::write(index_path.join("index.bin"), &index_data)?;

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

        self.index = Some(idx);
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

        // Use index if available, otherwise brute-force
        let (result_ids, result_dists) = if let Some(ref idx) = self.index {
            idx.search(query, k, &search_params)?
        } else {
            // Brute-force search across all chunks
            self.brute_force_search(query, k, &search_params)?
        };

        // Retrieve fields for results
        let fields = self.field_store.retrieve_many(&result_ids)?;

        Ok(SearchResult {
            ids: result_ids,
            distances: result_dists,
            fields,
            index_mode: self.index_mode.clone().unwrap_or("flat-ip".into()),
            dimension: dim,
            k,
        })
    }

    /// Brute-force search when no index is built.
    fn brute_force_search(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        let dim = self.meta.dimension;
        let metric = self
            .index_mode
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
            .unwrap_or(DistanceMetric::InnerProduct);

        let all_files = self.vector_store.get_all_files();
        let id_mapper = self.vector_store.id_mapper();

        let mut all_data: Vec<f32> = Vec::new();
        let mut all_ids: Vec<u64> = Vec::new();

        for filename in &all_files {
            let chunk = self.vector_store.load_chunk_npk(filename)?;
            let _n_in_chunk = chunk.len() / dim;

            if let Some(ids) = id_mapper.generate_ids(filename) {
                // Apply subset filter
                if let Some(ref subset) = params.subset_indices {
                    let subset_set: std::collections::HashSet<u64> =
                        subset.iter().cloned().collect();
                    for (i, &id) in ids.iter().enumerate() {
                        if subset_set.contains(&id) {
                            let start = i * dim;
                            all_data.extend_from_slice(&chunk[start..start + dim]);
                            all_ids.push(id);
                        }
                    }
                } else {
                    all_data.extend_from_slice(&chunk);
                    all_ids.extend(ids);
                }
            }
        }

        if all_ids.is_empty() {
            return Ok((vec![], vec![]));
        }

        let (top_indices, top_dists) =
            crate::distance::top_k_search(query, &all_data, dim, k, metric);

        let result_ids: Vec<u64> = top_indices
            .iter()
            .map(|&idx| all_ids[idx as usize])
            .collect();

        Ok((result_ids, top_dists))
    }

    /// Remove the index.
    pub fn remove_index(&mut self) -> Result<()> {
        self.index = None;
        self.index_mode = None;
        self.meta.index_mode = None;

        let index_path = self.path.join("index");
        let meta_path = self.path.join("index_meta");
        if index_path.exists() {
            std::fs::remove_dir_all(&index_path)?;
        }
        if meta_path.exists() {
            std::fs::remove_dir_all(&meta_path)?;
        }

        Ok(())
    }

    /// Get collection shape (n_vectors, dimension).
    pub fn shape(&self) -> Result<(u64, usize)> {
        self.vector_store.get_shape()
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
