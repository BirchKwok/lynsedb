//! PyO3 Python bindings for LynseDB Rust core.
//!
//! Exposes the Rust engine to Python, maintaining API compatibility
//! with the existing LynseDB Python interface.

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::engine::{Collection, DatabaseEngine, DatabaseManager, SearchResult};

/// Register all Python classes and functions into the module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatabaseEngine>()?;
    m.add_class::<PyDatabaseManager>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyFlatIndex>()?;
    m.add_class::<PyIvfFlatIndex>()?;
    m.add_function(wrap_pyfunction!(py_compute_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_top_k_search, m)?)?;
    m.add_function(wrap_pyfunction!(py_start_server, m)?)?;
    m.add_function(wrap_pyfunction!(py_start_server_background, m)?)?;
    Ok(())
}

// ─── DatabaseEngine binding ──────────────────────────────────────────────────

/// Python wrapper for the Rust DatabaseEngine.
#[pyclass(name = "DatabaseEngine")]
pub struct PyDatabaseEngine {
    inner: DatabaseEngine,
}

#[pymethods]
impl PyDatabaseEngine {
    #[new]
    fn new(root_path: &str) -> PyResult<Self> {
        let engine = DatabaseEngine::open(Path::new(root_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: engine })
    }

    /// Create a new collection.
    fn create_collection(&self, name: &str, dimension: usize) -> PyResult<PyCollection> {
        let coll = self
            .inner
            .create_collection(name, dimension, 100_000)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyCollection { inner: coll })
    }

    /// Get or open an existing collection.
    fn get_collection(&self, name: &str, dimension: usize) -> PyResult<PyCollection> {
        let coll = self
            .inner
            .get_or_open_collection(name, dimension, 100_000)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyCollection { inner: coll })
    }

    /// Drop a collection.
    fn drop_collection(&self, name: &str) -> PyResult<()> {
        self.inner
            .drop_collection(name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// List all collections.
    fn list_collections(&self) -> Vec<String> {
        self.inner.list_collections()
    }

    /// Check if a collection exists.
    fn has_collection(&self, name: &str) -> bool {
        self.inner.has_collection(name)
    }

    /// Get root path.
    fn root_path(&self) -> String {
        self.inner.root_path().to_string_lossy().to_string()
    }
}

// ─── Collection binding ──────────────────────────────────────────────────────

/// Python wrapper for a Rust Collection.
#[pyclass(name = "Collection")]
pub struct PyCollection {
    inner: Arc<RwLock<Collection>>,
}

#[pymethods]
impl PyCollection {
    /// Add vectors with user-specified IDs and optional field metadata.
    ///
    /// Args:
    ///     vectors: numpy array of shape (n, dim) with dtype float32
    ///     ids: list of integer user IDs, one per vector
    ///     fields: optional list of dicts, one per vector
    #[pyo3(signature = (vectors, ids, fields=None))]
    fn add_items(
        &self,
        vectors: PyReadonlyArray2<f32>,
        ids: Vec<u64>,
        fields: Option<&Bound<'_, PyList>>,
    ) -> PyResult<()> {
        let array = vectors.as_array();
        let n_vectors = array.nrows();
        let _dim = array.ncols();

        // Zero-copy: get contiguous f32 slice from numpy array
        let flat_data: &[f32] = array
            .as_slice()
            .expect("numpy array must be contiguous (C-order)");

        // Convert Python dicts to Rust HashMaps
        let rust_fields = if let Some(field_list) = fields {
            let mut result: Vec<HashMap<String, serde_json::Value>> = Vec::with_capacity(n_vectors);
            for item in field_list.iter() {
                let dict = item.downcast::<PyDict>()?;
                let mut map = HashMap::new();
                for (key, value) in dict.iter() {
                    let k: String = key.extract()?;
                    let v = py_to_json_value(&value)?;
                    map.insert(k, v);
                }
                result.push(map);
            }
            Some(result)
        } else {
            None
        };

        let mut coll = self.inner.write();
        coll.add_items(&flat_data, n_vectors, &ids, rust_fields.as_deref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Build or rebuild the index.
    ///
    /// Args:
    ///     index_type: e.g. "IVF-IP-SQ8", "HNSW-L2", "Flat-Cos", etc.
    fn build_index(&self, index_type: &str) -> PyResult<()> {
        let mut coll = self.inner.write();
        coll.build_index(index_type)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Remove the current index.
    fn remove_index(&self) -> PyResult<()> {
        let mut coll = self.inner.write();
        coll.remove_index()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Search for nearest neighbors.
    ///
    /// Args:
    ///     vector: query vector (1D numpy array)
    ///     k: number of results
    ///     where: optional SQL-like filter string
    ///     nprobe: number of IVF probes (default 10)
    ///
    /// Returns:
    ///     PySearchResult with ids, distances, fields
    #[pyo3(signature = (vector, k=None, where_expr=None, nprobe=None))]
    fn search(
        &self,
        vector: PyReadonlyArray1<f32>,
        k: Option<usize>,
        where_expr: Option<&str>,
        nprobe: Option<usize>,
    ) -> PyResult<PySearchResult> {
        let query = vector.as_slice()?;
        let k = k.unwrap_or(10);
        let nprobe = nprobe.unwrap_or(10);

        let coll = self.inner.read();
        let result = coll
            .search(query, k, where_expr, nprobe)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PySearchResult { inner: result })
    }

    /// Get collection shape (n_vectors, dimension).
    fn shape(&self) -> PyResult<(u64, usize)> {
        let coll = self.inner.read();
        coll.shape()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get collection name.
    fn name(&self) -> String {
        self.inner.read().meta.name.clone()
    }

    /// Get collection dimension.
    fn dimension(&self) -> usize {
        self.inner.read().meta.dimension
    }

    /// Get current index mode.
    fn index_mode(&self) -> Option<String> {
        self.inner.read().meta.index_mode.clone()
    }

    /// Batch search: search multiple query vectors in parallel.
    ///
    /// Args:
    ///     vectors: numpy array of shape (n_queries, dim) with dtype float32
    ///     k: number of results per query
    ///     where: optional SQL-like filter string
    ///     nprobe: number of IVF probes (default 10)
    ///
    /// Returns:
    ///     list of PySearchResult, one per query
    #[pyo3(signature = (vectors, k=None, where_expr=None, nprobe=None))]
    fn batch_search(
        &self,
        vectors: PyReadonlyArray2<f32>,
        k: Option<usize>,
        where_expr: Option<&str>,
        nprobe: Option<usize>,
    ) -> PyResult<Vec<PySearchResult>> {
        let array = vectors.as_array();
        let n_queries = array.nrows();
        let flat_data: &[f32] = array
            .as_slice()
            .expect("numpy array must be contiguous (C-order)");
        let k = k.unwrap_or(10);
        let nprobe = nprobe.unwrap_or(10);

        let coll = self.inner.read();
        let results = coll
            .batch_search(flat_data, n_queries, k, where_expr, nprobe)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|r| PySearchResult { inner: r })
            .collect())
    }

    /// Update vectors by IDs: atomic delete + insert.
    ///
    /// Args:
    ///     ids: list of vector IDs to update
    ///     vectors: numpy array of shape (n, dim) with dtype float32
    ///     fields: optional list of dicts, one per vector
    #[pyo3(signature = (ids, vectors, fields=None))]
    fn update_items(
        &self,
        ids: Vec<u64>,
        vectors: PyReadonlyArray2<f32>,
        fields: Option<&Bound<'_, PyList>>,
    ) -> PyResult<()> {
        let array = vectors.as_array();
        let n_vectors = array.nrows();
        let flat_data: &[f32] = array
            .as_slice()
            .expect("numpy array must be contiguous (C-order)");

        let rust_fields = py_fields_to_json(fields, n_vectors)?;

        let mut coll = self.inner.write();
        coll.update_items(&ids, flat_data, n_vectors, rust_fields.as_deref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Insert or update vectors by IDs.
    ///
    /// Existing IDs are updated in place. Missing IDs are inserted. If fields
    /// are provided, each field map replaces the row's current fields.
    #[pyo3(signature = (ids, vectors, fields=None))]
    fn upsert_items(
        &self,
        ids: Vec<u64>,
        vectors: PyReadonlyArray2<f32>,
        fields: Option<&Bound<'_, PyList>>,
    ) -> PyResult<()> {
        let array = vectors.as_array();
        let n_vectors = array.nrows();
        let flat_data: &[f32] = array
            .as_slice()
            .expect("numpy array must be contiguous (C-order)");

        let rust_fields = py_fields_to_json(fields, n_vectors)?;

        let mut coll = self.inner.write();
        coll.upsert_items(&ids, flat_data, n_vectors, rust_fields.as_deref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Commit: clear WAL after successful writes.
    fn commit(&self) -> PyResult<()> {
        let coll = self.inner.read();
        coll.commit()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Flush pending WAL bytes and fsync collection files without clearing WAL.
    fn flush(&self) -> PyResult<()> {
        let coll = self.inner.read();
        coll.flush()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Checkpoint durable state and clear WAL.
    fn checkpoint(&self) -> PyResult<()> {
        let coll = self.inner.read();
        coll.checkpoint()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Close the collection handle from an API perspective.
    fn close(&self) -> PyResult<()> {
        let mut coll = self.inner.write();
        coll.close()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Check if there is uncommitted WAL data.
    fn has_uncommitted_data(&self) -> bool {
        self.inner.read().has_uncommitted_data()
    }

    /// Sync the index with any new data since last sync.
    fn sync_index(&self) -> PyResult<()> {
        let mut coll = self.inner.write();
        coll.sync_index()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get the current storage fingerprint.
    fn fingerprint(&self) -> String {
        self.inner.read().fingerprint()
    }

    /// Return first n vectors + user IDs + fields.
    ///
    /// Returns: (flat_f32_numpy_1d, ids: Vec<u64>, list_of_field_dicts)
    #[pyo3(signature = (n=5))]
    fn head<'py>(
        &self,
        py: Python<'py>,
        n: usize,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<u64>, Bound<'py, PyList>)> {
        let coll = self.inner.read();
        let (data, ids, fields) = coll
            .head(n)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let arr = data.into_pyarray_bound(py);
        let field_list = fields_to_pylist(py, &fields)?;
        Ok((arr, ids, field_list))
    }

    /// Return last n vectors + user IDs + fields.
    #[pyo3(signature = (n=5))]
    fn tail<'py>(
        &self,
        py: Python<'py>,
        n: usize,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<u64>, Bound<'py, PyList>)> {
        let coll = self.inner.read();
        let (data, ids, fields) = coll
            .tail(n)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let arr = data.into_pyarray_bound(py);
        let field_list = fields_to_pylist(py, &fields)?;
        Ok((arr, ids, field_list))
    }

    /// Get vector store info for zero-copy mmap access.
    ///
    /// Returns (path, n_vectors, dimension) so Python can create:
    ///   vectors = np.memmap(path, dtype=np.float32, mode='r').reshape(n, dim)
    ///
    /// This is **true zero-copy**: OS page cache ensures the same physical
    /// pages are shared between Rust mmap and Python memmap.
    fn vector_store_info(&self) -> PyResult<(String, u64, usize)> {
        let coll = self.inner.read();
        let (n, dim) = coll
            .shape()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let (path, _, _) = coll
            .vector_store_info()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok((path, n, dim))
    }

    /// Return a zero-copy numpy memmap view of the entire vector store.
    ///
    /// Returns: numpy array of shape (n_vectors, dimension) backed by OS mmap.
    /// No data is copied — reads go directly to the page cache.
    fn vectors_numpy<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let coll = self.inner.read();
        let (path, n, dim) = coll
            .vector_store_info()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        drop(coll);

        if n == 0 {
            // Return empty array
            let np = py.import_bound("numpy")?;
            let empty = np.call_method1("zeros", ((0, dim), "float32"))?;
            return Ok(empty.into());
        }

        let np = py.import_bound("numpy")?;
        let dtype_obj: PyObject = "float32".into_py(py);
        let mode_obj: PyObject = "r".into_py(py);
        let shape_obj: PyObject = (n as usize, dim).into_py(py);
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", dtype_obj)?;
        kwargs.set_item("mode", mode_obj)?;
        kwargs.set_item("shape", shape_obj)?;
        let memmap = np.call_method("memmap", (path,), Some(&kwargs))?;
        Ok(memmap.into())
    }

    /// Retrieve vectors by IDs as a zero-copy-friendly numpy array.
    ///
    /// For contiguous ID ranges, this is nearly zero-copy (backed by mmap).
    /// For scattered IDs, copies only the requested vectors (single allocation).
    ///
    /// Returns: numpy array of shape (len(ids), dimension), dtype float32
    fn get_vectors<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<u64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        let coll = self.inner.read();
        let dim = coll.dimension();
        let n_ids = ids.len();

        let (flat, _fields) = coll
            .read_vectors_by_ids(&ids)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let arr = unsafe { numpy::PyArray2::<f32>::new_bound(py, [n_ids, dim], false) };
        let out_slice = unsafe { arr.as_slice_mut()? };
        let copy_len = flat.len().min(out_slice.len());
        out_slice[..copy_len].copy_from_slice(&flat[..copy_len]);

        Ok(arr)
    }

    /// Query fields with a SQL-like filter. Returns matching IDs.
    fn query_fields(&self, where_expr: &str) -> PyResult<Vec<u64>> {
        let coll = self.inner.read();
        coll.query_fields(where_expr)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Query fields with a SQL-like filter. Returns (ids, fields) in a single
    /// ApexBase query — no second round-trip for field retrieval.
    fn query_with_fields<'py>(
        &self,
        py: Python<'py>,
        where_expr: &str,
    ) -> PyResult<(Vec<u64>, Bound<'py, PyList>)> {
        let coll = self.inner.read();
        let (ids, fields) = coll
            .query_with_fields(where_expr)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let field_list = fields_to_pylist(py, &fields)?;
        Ok((ids, field_list))
    }

    /// Retrieve field metadata for specific IDs.
    fn retrieve_fields<'py>(&self, py: Python<'py>, ids: Vec<u64>) -> PyResult<Bound<'py, PyList>> {
        let coll = self.inner.read();
        let fields = coll
            .retrieve_fields(&ids)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        fields_to_pylist(py, &fields)
    }

    /// List all field names.
    fn list_fields(&self) -> PyResult<Vec<String>> {
        let coll = self.inner.read();
        coll.list_fields()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Soft-delete vectors by ID. Deleted IDs are excluded from all future search results.
    /// The raw data is NOT removed from disk; use vacuum() to physically compact.
    fn delete_items(&self, ids: Vec<u64>) -> PyResult<()> {
        let coll = self.inner.read();
        coll.delete_items(&ids)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Restore previously soft-deleted vectors (remove from tombstone).
    fn restore_items(&self, ids: Vec<u64>) -> PyResult<()> {
        let coll = self.inner.read();
        coll.restore_items(&ids)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Return the sorted list of all currently soft-deleted IDs.
    fn list_deleted_ids(&self) -> PyResult<Vec<u64>> {
        let coll = self.inner.read();
        Ok(coll.list_deleted_ids())
    }

    /// Range search: return all non-deleted vectors within a distance threshold.
    ///
    /// For L2 metric: returns IDs where distance ≤ threshold (ascending).
    /// For IP/Cosine: returns IDs where score ≥ threshold (descending).
    ///
    /// Returns (ids: Vec<u64>, distances: Vec<f32>) capped at max_results.
    #[pyo3(signature = (vector, threshold, max_results=1000))]
    fn search_range(
        &self,
        vector: PyReadonlyArray1<f32>,
        threshold: f32,
        max_results: usize,
    ) -> PyResult<(Vec<u64>, Vec<f32>)> {
        let query = vector.as_slice()?;
        let coll = self.inner.read();
        coll.search_range(query, threshold, max_results)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Benchmark filtered search strategies.
    ///
    /// Compares three approaches:
    /// 1. prefilter_copy: field query → copy matching vectors → top_k
    /// 2. prefilter_fused: field query → scan mmap in-place, skip non-matching
    /// 3. postfilter: full FlatMmap scan with oversample → filter results
    ///
    /// Returns list of (strategy_name, microseconds, result_count) tuples.
    #[pyo3(signature = (vector, k=10, where_expr="", oversample=100, warmup=3, iterations=10))]
    fn benchmark_filtered_search(
        &self,
        vector: PyReadonlyArray1<f32>,
        k: usize,
        where_expr: &str,
        oversample: usize,
        warmup: usize,
        iterations: usize,
    ) -> PyResult<Vec<(String, f64, usize)>> {
        let query = vector.as_slice()?;
        let coll = self.inner.read();
        coll.benchmark_filtered_search(query, k, where_expr, oversample, warmup, iterations)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Compact the collection: physically remove all tombstoned vectors,
    /// rewrite vectors.bin and id_map.bin, rebuild index if present.
    /// Returns the number of vectors removed.
    fn compact(&self) -> PyResult<usize> {
        let mut coll = self.inner.write();
        coll.compact()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Check whether a user ID exists in the collection.
    fn is_id_exists(&self, user_id: u64) -> PyResult<bool> {
        let coll = self.inner.read();
        Ok(coll.is_id_exists(user_id))
    }

    /// Return the maximum user ID stored in the collection, or -1 if empty.
    fn max_id(&self) -> PyResult<i64> {
        let coll = self.inner.read();
        Ok(coll.max_id().map(|id| id as i64).unwrap_or(-1))
    }

    /// Delete the collection from disk.
    fn delete(&self) -> PyResult<()> {
        let coll = self.inner.read();
        let path = coll.collection_path().to_path_buf();
        drop(coll);
        if path.exists() {
            std::fs::remove_dir_all(&path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }
        Ok(())
    }
}

// ─── SearchResult binding ────────────────────────────────────────────────────

/// Python wrapper for search results.
#[pyclass(name = "SearchResult")]
pub struct PySearchResult {
    inner: SearchResult,
}

#[pymethods]
impl PySearchResult {
    /// Get result IDs as numpy array (zero-copy transfer to numpy).
    fn ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let ids: Vec<i64> = self.inner.ids.iter().map(|&id| id as i64).collect();
        ids.into_pyarray_bound(py)
    }

    /// Get result distances as numpy array (zero-copy transfer to numpy).
    fn distances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.distances.clone().into_pyarray_bound(py)
    }

    /// Get result fields as list of dicts.
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty_bound(py);
        for field_map in &self.inner.fields {
            let dict = PyDict::new_bound(py);
            for (k, v) in field_map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            list.append(dict)?;
        }
        Ok(list)
    }

    /// Get the index mode used for this search.
    fn index_mode(&self) -> &str {
        &self.inner.index_mode
    }

    /// Get result count.
    fn __len__(&self) -> usize {
        self.inner.ids.len()
    }

    /// Get (ids, distances, fields) tuple for Python unpacking.
    fn to_tuple<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyList>,
    )> {
        Ok((self.ids(py), self.distances(py), self.fields(py)?))
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResult(n={}, k={}, dim={}, index={})",
            self.inner.ids.len(),
            self.inner.k,
            self.inner.dimension,
            self.inner.index_mode,
        )
    }
}

// ─── FlatIndex (mmap-backed brute-force) ────────────────────────────────────

/// Ultra-fast flat vector index backed by a single mmap'd binary file.
///
/// Usage from Python:
///   idx = lynse_core.FlatIndex("/path/to/vectors.bin", 128)
///   idx.write(np.random.rand(1_000_000, 128).astype(np.float32))
///   ids, dists = idx.search(query_vec, k=10, metric="ip")
#[pyclass(name = "FlatIndex")]
pub struct PyFlatIndex {
    inner: crate::storage::flat_mmap::FlatMmap,
}

#[pymethods]
impl PyFlatIndex {
    #[new]
    fn new(path: &str, dim: usize) -> PyResult<Self> {
        let inner = crate::storage::flat_mmap::FlatMmap::open(std::path::Path::new(path), dim)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of vectors in the index.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Vector dimension.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Write vectors from a contiguous numpy array (n_vectors × dim).
    fn write(&mut self, data: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = data.as_array();
        let flat = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("numpy array must be contiguous (C-order)")
        })?;
        self.inner
            .write(flat)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Brute-force top-k search.
    ///
    /// Args:
    ///   query: 1-D f32 array (dim,)
    ///   k: number of results
    ///   metric: "ip" | "l2" | "cosine"
    ///
    /// Returns:
    ///   (indices: ndarray[u32], distances: ndarray[f32])
    #[pyo3(signature = (query, k=10, metric="ip"))]
    fn search<'py>(
        &mut self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        k: usize,
        metric: &str,
    ) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>)> {
        let metric = crate::distance::DistanceMetric::from_str(metric).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Unknown metric: {}", metric))
        })?;
        let q = query.as_slice()?;
        let (ids, dists) = self.inner.search(q, k, metric, false);
        Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
    }

    /// Batch search: multiple queries against the same data.
    ///
    /// Args:
    ///   queries: 2-D f32 array (n_queries × dim)
    ///   k: number of results per query
    ///   metric: "ip" | "l2" | "cosine"
    ///
    /// Returns:
    ///   list of (indices, distances) tuples
    #[pyo3(signature = (queries, k=10, metric="ip"))]
    fn batch_search<'py>(
        &mut self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        metric: &str,
    ) -> PyResult<Vec<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>)>> {
        let metric_enum = crate::distance::DistanceMetric::from_str(metric).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Unknown metric: {}", metric))
        })?;
        let arr = queries.as_array();
        let dim = arr.ncols();
        let flat = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("numpy array must be contiguous (C-order)")
        })?;
        let n_queries = arr.nrows();

        // Sequential search (each search already uses rayon internally)
        let mut results = Vec::with_capacity(n_queries);
        for q in 0..n_queries {
            let q_start = q * dim;
            let q_end = q_start + dim;
            let (ids, dists) = self
                .inner
                .search(&flat[q_start..q_end], k, metric_enum, false);
            results.push((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)));
        }
        Ok(results)
    }
}

// ─── IvfFlatIndex (IVF + mmap brute-force) ──────────────────────────────────

/// IVF_FLAT index: KMeans partitioning + brute-force within partitions.
///
/// Usage from Python:
///   idx = lynse_core.IvfFlatIndex.build("/path/to/ivf.bin", data, dim=128, n_partitions=256)
///   ids, dists = idx.search(query, k=10, nprobe=10, metric="ip")
#[pyclass(name = "IvfFlatIndex")]
pub struct PyIvfFlatIndex {
    inner: crate::storage::ivf_flat_mmap::IvfFlatMmap,
}

#[pymethods]
impl PyIvfFlatIndex {
    /// Build a new IVF_FLAT index from data.
    ///
    /// Args:
    ///   metric: "ip" | "l2" | "cosine" — KMeans uses matching metric for better recall
    #[staticmethod]
    #[pyo3(signature = (path, data, dim, n_partitions=256, n_iters=20, metric="ip"))]
    fn build(
        path: &str,
        data: PyReadonlyArray2<f32>,
        dim: usize,
        n_partitions: usize,
        n_iters: usize,
        metric: &str,
    ) -> PyResult<Self> {
        let metric_enum = crate::distance::DistanceMetric::from_str(metric).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Unknown metric: {}", metric))
        })?;
        let arr = data.as_array();
        let flat = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("numpy array must be contiguous (C-order)")
        })?;
        let inner = crate::storage::ivf_flat_mmap::IvfFlatMmap::build(
            std::path::Path::new(path),
            flat,
            dim,
            n_partitions,
            n_iters,
            metric_enum,
        )
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Open an existing IVF_FLAT index.
    #[staticmethod]
    fn open(path: &str, dim: usize) -> PyResult<Self> {
        let inner =
            crate::storage::ivf_flat_mmap::IvfFlatMmap::open(std::path::Path::new(path), dim)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn n_partitions(&self) -> usize {
        self.inner.n_partitions()
    }

    /// IVF_FLAT search.
    ///
    /// Args:
    ///   query: 1-D f32 array (dim,)
    ///   k: number of results
    ///   nprobe: number of partitions to scan (higher = better recall, slower)
    ///   metric: "ip" | "l2" | "cosine"
    #[pyo3(signature = (query, k=10, nprobe=10, metric="ip"))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        k: usize,
        nprobe: usize,
        metric: &str,
    ) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>)> {
        let metric = crate::distance::DistanceMetric::from_str(metric).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Unknown metric: {}", metric))
        })?;
        let q = query.as_slice()?;
        let (ids, dists) = self.inner.search(q, k, nprobe, metric);
        Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
    }
}

// ─── Standalone functions ────────────────────────────────────────────────────

/// Compute distance between two vectors.
#[pyfunction]
fn py_compute_distance(
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
    metric: &str,
) -> PyResult<f32> {
    let metric = crate::distance::DistanceMetric::from_str(metric).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("Unknown metric: {}", metric))
    })?;
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    if a_slice.len() != b_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Vector dimensions must match",
        ));
    }
    Ok(crate::distance::compute_distance_f32(
        a_slice, b_slice, metric,
    ))
}

/// Top-k search: find k nearest vectors from candidates.
/// Zero-copy: reads directly from numpy memory without intermediate Vec allocation.
#[pyfunction]
fn py_top_k_search<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<f32>,
    candidates: PyReadonlyArray2<f32>,
    metric: &str,
    k: usize,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>)> {
    let metric = crate::distance::DistanceMetric::from_str(metric).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("Unknown metric: {}", metric))
    })?;

    let q = query.as_slice()?;
    let c_array = candidates.as_array();
    let dim = c_array.ncols();

    // Zero-copy: get contiguous f32 slice directly from numpy memory
    let flat: &[f32] = c_array
        .as_slice()
        .expect("numpy candidates array must be contiguous (C-order)");

    let (ids, dists) = crate::distance::top_k_search(q, flat, dim, k, metric);

    Ok((ids.into_pyarray_bound(py), dists.into_pyarray_bound(py)))
}

// ─── DatabaseManager binding ─────────────────────────────────────────────────

/// Python wrapper for the Rust DatabaseManager.
#[pyclass(name = "DatabaseManager")]
pub struct PyDatabaseManager {
    inner: Arc<DatabaseManager>,
}

#[pymethods]
impl PyDatabaseManager {
    #[new]
    fn new(root_path: &str) -> PyResult<Self> {
        let mgr = DatabaseManager::new(Path::new(root_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(mgr),
        })
    }

    fn create_database(&self, name: &str) -> PyResult<()> {
        self.inner
            .create_database(name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn drop_database(&self, name: &str) -> PyResult<()> {
        self.inner
            .drop_database(name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn list_databases(&self) -> Vec<String> {
        self.inner.list_databases()
    }

    fn database_exists(&self, name: &str) -> bool {
        self.inner.database_exists(name)
    }

    fn show_collections(&self, db_name: &str) -> PyResult<Vec<String>> {
        self.inner
            .show_collections(db_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (db_name, collection_name, dim, drop_if_exists=None, description=None))]
    fn require_collection(
        &self,
        db_name: &str,
        collection_name: &str,
        dim: usize,
        drop_if_exists: Option<bool>,
        description: Option<&str>,
    ) -> PyResult<()> {
        self.inner
            .require_collection(
                db_name,
                collection_name,
                dim,
                100_000,
                drop_if_exists.unwrap_or(false),
                description,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn drop_collection(&self, db_name: &str, collection_name: &str) -> PyResult<()> {
        self.inner
            .drop_collection(db_name, collection_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn collection_exists(&self, db_name: &str, collection_name: &str) -> PyResult<bool> {
        self.inner
            .collection_exists(db_name, collection_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (db_name, collection_name, description=None))]
    fn update_collection_description(
        &self,
        db_name: &str,
        collection_name: &str,
        description: Option<&str>,
    ) -> PyResult<()> {
        self.inner
            .update_collection_description(db_name, collection_name, description)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get a collection from a database, returning a PyCollection for direct access.
    fn get_collection(
        &self,
        db_name: &str,
        collection_name: &str,
        dim: usize,
    ) -> PyResult<PyCollection> {
        self.inner
            .get_or_open_database(db_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let coll = self
            .inner
            .with_database(db_name, |engine| {
                engine.get_or_open_collection(collection_name, dim, 100_000)
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyCollection { inner: coll })
    }

    /// Get collection config from collections.json.
    fn get_collection_config(
        &self,
        db_name: &str,
        collection_name: &str,
    ) -> PyResult<Option<(usize, usize, Option<String>)>> {
        let configs = self
            .inner
            .get_collection_configs(db_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(configs
            .get(collection_name)
            .map(|c| (c.dim, c.chunk_size, c.description.clone())))
    }

    fn root_path(&self) -> String {
        self.inner.root_path().to_string_lossy().to_string()
    }
}

// ─── Server functions ────────────────────────────────────────────────────────

/// Start the HTTP server (blocking). Releases GIL so other Python threads can run.
#[pyfunction]
#[pyo3(signature = (host="127.0.0.1", port=7637, root_path=".", api_key=None))]
fn py_start_server(
    py: Python<'_>,
    host: &str,
    port: u16,
    root_path: &str,
    api_key: Option<String>,
) -> PyResult<()> {
    let host = host.to_string();
    let root_path = root_path.to_string();
    py.allow_threads(move || {
        crate::server::run_server(&host, port, &root_path, api_key)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })
}

/// Start the HTTP server in a background thread.
/// Returns immediately. The server runs until the process exits.
#[pyfunction]
#[pyo3(signature = (host="127.0.0.1", port=7637, root_path=".", api_key=None))]
fn py_start_server_background(
    py: Python<'_>,
    host: &str,
    port: u16,
    root_path: &str,
    api_key: Option<String>,
) -> PyResult<()> {
    let host = host.to_string();
    let root_path = root_path.to_string();
    py.allow_threads(move || {
        crate::server::start_server_background(host, port, root_path, api_key);
    });
    // Give server a moment to start
    std::thread::sleep(std::time::Duration::from_millis(200));
    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Convert a Vec of field HashMaps to a Python list of dicts.
fn fields_to_pylist<'py>(
    py: Python<'py>,
    fields: &[HashMap<String, serde_json::Value>],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty_bound(py);
    for field_map in fields {
        let dict = PyDict::new_bound(py);
        for (k, v) in field_map {
            dict.set_item(k, json_to_py(py, v)?)?;
        }
        list.append(dict)?;
    }
    Ok(list)
}

fn py_fields_to_json(
    fields: Option<&Bound<'_, PyList>>,
    expected_len: usize,
) -> PyResult<Option<Vec<HashMap<String, serde_json::Value>>>> {
    let Some(field_list) = fields else {
        return Ok(None);
    };
    if field_list.len() != expected_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "fields length ({}) must match vector count ({})",
            field_list.len(),
            expected_len
        )));
    }

    let mut result: Vec<HashMap<String, serde_json::Value>> = Vec::with_capacity(expected_len);
    for item in field_list.iter() {
        let dict = item.downcast::<PyDict>()?;
        let mut map = HashMap::new();
        for (key, value) in dict.iter() {
            let k: String = key.extract()?;
            let v = py_to_json_value(&value)?;
            map.insert(k, v);
        }
        result.push(map);
    }
    Ok(Some(result))
}

/// Convert a Python object to serde_json::Value.
fn py_to_json_value(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(serde_json::json!(i))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::json!(f))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let mut arr = Vec::new();
        for item in list.iter() {
            arr.push(py_to_json_value(&item)?);
        }
        Ok(serde_json::Value::Array(arr))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            map.insert(key, py_to_json_value(&v)?);
        }
        Ok(serde_json::Value::Object(map))
    } else {
        // Fallback: convert to string
        let s = obj.str()?.to_string();
        Ok(serde_json::Value::String(s))
    }
}

/// Convert serde_json::Value to a Python object.
fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<pyo3::PyObject> {
    match v {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_py(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_py(py))
            } else {
                Ok(n.to_string().into_py(py))
            }
        }
        serde_json::Value::String(s) => Ok(s.into_py(py)),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(obj) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in obj {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}
