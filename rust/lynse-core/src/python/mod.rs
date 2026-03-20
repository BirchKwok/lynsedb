//! PyO3 Python bindings for LynseDB Rust core.
//!
//! Exposes the Rust engine to Python, maintaining API compatibility
//! with the existing LynseDB Python interface.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use parking_lot::RwLock;

use crate::engine::{Collection, DatabaseEngine, SearchResult};

/// Register all Python classes and functions into the module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatabaseEngine>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PySearchResult>()?;
    m.add_function(wrap_pyfunction!(py_compute_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_top_k_search, m)?)?;
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
    #[pyo3(signature = (name, dimension, chunk_size=None))]
    fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        chunk_size: Option<usize>,
    ) -> PyResult<PyCollection> {
        let chunk_size = chunk_size.unwrap_or(100_000);
        let coll = self
            .inner
            .create_collection(name, dimension, chunk_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyCollection { inner: coll })
    }

    /// Get or open an existing collection.
    #[pyo3(signature = (name, dimension, chunk_size=None))]
    fn get_collection(
        &self,
        name: &str,
        dimension: usize,
        chunk_size: Option<usize>,
    ) -> PyResult<PyCollection> {
        let chunk_size = chunk_size.unwrap_or(100_000);
        let coll = self
            .inner
            .get_or_open_collection(name, dimension, chunk_size)
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
    /// Add vectors with optional field metadata.
    ///
    /// Args:
    ///     vectors: numpy array of shape (n, dim) with dtype float32
    ///     fields: optional list of dicts, one per vector
    #[pyo3(signature = (vectors, fields=None))]
    fn add_items(
        &self,
        vectors: PyReadonlyArray2<f32>,
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
            let mut result: Vec<HashMap<String, serde_json::Value>> =
                Vec::with_capacity(n_vectors);
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
        coll.add_items(
            &flat_data,
            n_vectors,
            rust_fields.as_deref(),
        )
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
    ///     search_filter: optional SQL-like filter string
    ///     nprobe: number of IVF probes (default 10)
    ///
    /// Returns:
    ///     PySearchResult with ids, distances, fields
    #[pyo3(signature = (vector, k=None, search_filter=None, nprobe=None))]
    fn search(
        &self,
        vector: PyReadonlyArray1<f32>,
        k: Option<usize>,
        search_filter: Option<&str>,
        nprobe: Option<usize>,
    ) -> PyResult<PySearchResult> {
        let query = vector.as_slice()?;
        let k = k.unwrap_or(10);
        let nprobe = nprobe.unwrap_or(10);

        let coll = self.inner.read();
        let result = coll
            .search(query, k, search_filter, nprobe)
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
    /// Get result IDs as numpy array.
    fn ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let ids: Vec<i64> = self.inner.ids.iter().map(|&id| id as i64).collect();
        PyArray1::from_vec_bound(py, ids)
    }

    /// Get result distances as numpy array.
    fn distances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec_bound(py, self.inner.distances.clone())
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
    fn to_tuple<'py>(&self, py: Python<'py>) -> PyResult<(
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
    Ok(crate::distance::compute_distance_f32(a_slice, b_slice, metric))
}

/// Top-k search: find k nearest vectors from candidates.
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
    let flat: Vec<f32> = c_array.iter().copied().collect();

    let (ids, dists) = crate::distance::top_k_search(q, &flat, dim, k, metric);

    Ok((
        PyArray1::from_vec_bound(py, ids),
        PyArray1::from_vec_bound(py, dists),
    ))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

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
