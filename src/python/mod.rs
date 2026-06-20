//! PyO3 Python bindings for LynseDB Rust core.
//!
//! Exposes the Rust engine to Python, maintaining API compatibility
//! with the existing LynseDB Python interface.

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use parking_lot::{Mutex, RwLock};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use pyo3::{IntoPyObjectExt, Py, PyAny};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream, ToSocketAddrs};
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::engine::{
    Collection, DatabaseEngine, DatabaseManager, ExternalId, SearchResult, VectorFieldConfig,
};
use crate::storage::dtype::VectorDtype;
use numpy::ndarray::ArrayView2;

/// Register all Python classes and functions into the module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatabaseEngine>()?;
    m.add_class::<PyDatabaseManager>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyFlatIndex>()?;
    m.add_class::<PyIvfFlatIndex>()?;
    m.add_class::<PyClusterReadCoordinator>()?;
    m.add_class::<PyRemoteHttpClient>()?;
    m.add_function(wrap_pyfunction!(py_compute_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_top_k_search, m)?)?;
    m.add_function(wrap_pyfunction!(metadata_rpc_get, m)?)?;
    m.add_function(wrap_pyfunction!(metadata_rpc_cas, m)?)?;
    m.add_function(wrap_pyfunction!(metadata_rpc_close, m)?)?;
    m.add_function(wrap_pyfunction!(py_start_server, m)?)?;
    m.add_function(wrap_pyfunction!(py_start_server_background, m)?)?;
    Ok(())
}

// ─── Cluster read coordinator binding ───────────────────────────────────────

const MAX_METADATA_RPC_FRAME_BYTES: usize = 512 * 1024 * 1024;
const MAX_METADATA_RPC_IDLE_PER_OWNER: usize = 4;
static METADATA_RPC_POOL: OnceLock<Mutex<HashMap<String, Vec<TcpStream>>>> = OnceLock::new();

#[pyclass(name = "RemoteHttpClient")]
pub struct PyRemoteHttpClient {
    agent: ureq::Agent,
    base_url: String,
    api_key: Option<String>,
}

#[pymethods]
impl PyRemoteHttpClient {
    #[new]
    #[pyo3(signature = (base_url, api_key=None))]
    fn new(base_url: &str, api_key: Option<String>) -> PyResult<Self> {
        Ok(Self {
            agent: ureq::Agent::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
        })
    }

    #[pyo3(signature = (path, params=None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        path: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let params = query_params_from_pydict(params)?;
        let url = self.url_with_query(path, &params);
        let (status, raw) = py.detach(|| self.request_raw("GET", &url, None, None))?;
        remote_response_tuple(py, status, raw)
    }

    #[pyo3(signature = (path, json_body=None, params=None))]
    fn post_json<'py>(
        &self,
        py: Python<'py>,
        path: &str,
        json_body: Option<&Bound<'_, PyAny>>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let value = match json_body {
            Some(obj) if !obj.is_none() => py_to_json_value(obj)?,
            _ => serde_json::Value::Object(serde_json::Map::new()),
        };
        let body = serde_json::to_vec(&value)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let params = query_params_from_pydict(params)?;
        let url = self.url_with_query(path, &params);
        let (status, raw) =
            py.detach(|| self.request_raw("POST", &url, Some(&body), Some("application/json")))?;
        remote_response_tuple(py, status, raw)
    }

    #[pyo3(signature = (path, content, params=None, content_type="application/octet-stream"))]
    fn post_binary_raw<'py>(
        &self,
        py: Python<'py>,
        path: &str,
        content: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
        content_type: &str,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let body = content.extract::<Vec<u8>>()?;
        let params = query_params_from_pydict(params)?;
        let url = self.url_with_query(path, &params);
        let (status, raw) =
            py.detach(|| self.request_raw("POST", &url, Some(&body), Some(content_type)))?;
        remote_response_tuple(py, status, raw)
    }

    #[pyo3(signature = (database_name, collection_name, vectors, vector_encoding="float32", return_ids=false))]
    fn bulk_add_binary<'py>(
        &self,
        py: Python<'py>,
        database_name: &str,
        collection_name: &str,
        vectors: PyReadonlyArray2<'_, f32>,
        vector_encoding: &str,
        return_ids: bool,
    ) -> PyResult<Py<PyAny>> {
        let matrix = vectors.as_array();
        let shape = matrix.shape();
        if shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "vectors must be a 2D matrix",
            ));
        }
        let n_vectors = shape[0];
        let dim = shape[1];
        let body = encode_f32_matrix_for_wire(matrix, vector_encoding)?;
        let url = self.url_with_query(
            "/bulk_add_binary",
            &[
                ("database_name", database_name.to_string()),
                ("collection_name", collection_name.to_string()),
                ("dim", dim.to_string()),
                ("n_vectors", n_vectors.to_string()),
                ("vector_encoding", normalize_remote_vector_encoding(vector_encoding)?),
                ("return_ids", return_ids.to_string()),
            ],
        );
        let raw = py.detach(|| self.post_binary(&url, &body))?;
        if !return_ids {
            return Ok(py.None());
        }
        let ids = parse_json_response_ids(&raw)?;
        Ok(ids.into_pyarray(py).into_any().unbind())
    }

    #[pyo3(signature = (endpoint, database_name, collection_name, vectors, ids, vector_encoding="float32"))]
    fn write_binary_ids(
        &self,
        py: Python<'_>,
        endpoint: &str,
        database_name: &str,
        collection_name: &str,
        vectors: PyReadonlyArray2<'_, f32>,
        ids: Vec<u64>,
        vector_encoding: &str,
    ) -> PyResult<()> {
        if endpoint != "add_binary_ids" && endpoint != "upsert_binary" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "endpoint must be add_binary_ids or upsert_binary",
            ));
        }
        let matrix = vectors.as_array();
        let shape = matrix.shape();
        if shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "vectors must be a 2D matrix",
            ));
        }
        let n_vectors = shape[0];
        let dim = shape[1];
        if ids.len() != n_vectors {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ids length must match vector row count",
            ));
        }
        let normalized_encoding = normalize_remote_vector_encoding(vector_encoding)?;
        let mut body = encode_f32_matrix_for_wire(matrix, &normalized_encoding)?;
        let mut params = vec![
            ("database_name", database_name.to_string()),
            ("collection_name", collection_name.to_string()),
            ("dim", dim.to_string()),
            ("n_vectors", n_vectors.to_string()),
            ("vector_encoding", normalized_encoding),
            ("return_ids", "false".to_string()),
        ];
        if let Some(start) = contiguous_id_start(&ids) {
            params.push(("ids_encoding", "range".to_string()));
            params.push(("ids_start", start.to_string()));
        } else {
            params.push(("ids_encoding", "raw".to_string()));
            for id in ids {
                body.extend_from_slice(&id.to_le_bytes());
            }
        }
        let url = self.url_with_query(&format!("/{}", endpoint), &params);
        py.detach(|| self.post_binary(&url, &body)).map(|_| ())
    }

    #[pyo3(signature = (endpoint, database_name, collection_name, vectors, ids, fields=None, vector_encoding="float32"))]
    fn write_records_binary<'py>(
        &self,
        py: Python<'py>,
        endpoint: &str,
        database_name: &str,
        collection_name: &str,
        vectors: PyReadonlyArray2<'_, f32>,
        ids: Vec<u64>,
        fields: Option<&Bound<'_, PyList>>,
        vector_encoding: &str,
    ) -> PyResult<Py<PyAny>> {
        if endpoint != "add_records_binary" && endpoint != "upsert_records_binary" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "endpoint must be add_records_binary or upsert_records_binary",
            ));
        }
        let matrix = vectors.as_array();
        let shape = matrix.shape();
        if shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "vectors must be a 2D matrix",
            ));
        }
        let n_vectors = shape[0];
        let dim = shape[1];
        if ids.len() != n_vectors {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ids length must match vector row count",
            ));
        }
        let normalized_encoding = normalize_remote_vector_encoding(vector_encoding)?;
        let mut body = encode_f32_matrix_for_wire(matrix, &normalized_encoding)?;
        let mut params = vec![
            ("database_name", database_name.to_string()),
            ("collection_name", collection_name.to_string()),
            ("dim", dim.to_string()),
            ("n_vectors", n_vectors.to_string()),
            ("vector_encoding", normalized_encoding),
        ];
        if let Some(start) = contiguous_id_start(&ids) {
            params.push(("ids_encoding", "range".to_string()));
            params.push(("ids_start", start.to_string()));
        } else {
            params.push(("ids_encoding", "raw".to_string()));
            for id in ids {
                body.extend_from_slice(&id.to_le_bytes());
            }
        }
        if let Some(field_maps) = py_fields_to_json(fields, n_vectors)? {
            encode_fields_binary(&field_maps, &mut body)?;
        }
        let url = self.url_with_query(&format!("/{}", endpoint), &params);
        let raw = py.detach(|| self.post_binary(&url, &body))?;
        let ids = parse_json_response_ids(&raw)?;
        Ok(ids.into_pyarray(py).into_any().unbind())
    }

    #[pyo3(signature = (database_name, collection_name, vector, k=10, where_expr=None, return_fields=false, vector_field="default", nprobe=10, approx=false, eps=1e-4, vector_encoding="float32"))]
    fn search_binary<'py>(
        &self,
        py: Python<'py>,
        database_name: &str,
        collection_name: &str,
        vector: PyReadonlyArray1<'_, f32>,
        k: usize,
        where_expr: Option<String>,
        return_fields: bool,
        vector_field: &str,
        nprobe: usize,
        approx: bool,
        eps: f32,
        vector_encoding: &str,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let values = vector.as_array();
        let dim = values.len();
        let normalized_encoding = normalize_remote_vector_encoding(vector_encoding)?;
        let body = if let Some(values) = values.as_slice() {
            encode_f32_slice_for_wire(values, &normalized_encoding)?
        } else {
            encode_f32_values_for_wire(values.iter().copied(), &normalized_encoding)?
        };
        let mut params = vec![
            ("database_name", database_name.to_string()),
            ("collection_name", collection_name.to_string()),
            ("dim", dim.to_string()),
            ("k", k.to_string()),
            ("return_fields", return_fields.to_string()),
            ("vector_encoding", normalized_encoding),
            ("nprobe", nprobe.to_string()),
            ("approx", approx.to_string()),
            ("eps", eps.to_string()),
        ];
        if let Some(expr) = where_expr {
            params.push(("where", expr));
        }
        if vector_field != "default" {
            params.push(("vector_field", vector_field.to_string()));
        }
        let url = self.url_with_query("/search_binary", &params);
        let raw = py.detach(|| self.post_binary(&url, &body))?;
        decode_remote_search_block(py, &raw, 0).map(|(tuple, _)| tuple)
    }

    #[pyo3(signature = (database_name, collection_name, vectors, k=10, where_expr=None, return_fields=false, nprobe=10, vector_encoding="float32"))]
    fn batch_search_binary<'py>(
        &self,
        py: Python<'py>,
        database_name: &str,
        collection_name: &str,
        vectors: PyReadonlyArray2<'_, f32>,
        k: usize,
        where_expr: Option<String>,
        return_fields: bool,
        nprobe: usize,
        vector_encoding: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        let matrix = vectors.as_array();
        let shape = matrix.shape();
        if shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "vectors must be a 2D matrix",
            ));
        }
        let n_queries = shape[0];
        let dim = shape[1];
        let normalized_encoding = normalize_remote_vector_encoding(vector_encoding)?;
        let body = encode_f32_matrix_for_wire(matrix, &normalized_encoding)?;
        let mut params = vec![
            ("database_name", database_name.to_string()),
            ("collection_name", collection_name.to_string()),
            ("dim", dim.to_string()),
            ("n_queries", n_queries.to_string()),
            ("k", k.to_string()),
            ("return_fields", return_fields.to_string()),
            ("vector_encoding", normalized_encoding),
            ("nprobe", nprobe.to_string()),
        ];
        if let Some(expr) = where_expr {
            params.push(("where", expr));
        }
        let url = self.url_with_query("/batch_search_binary", &params);
        let raw = py.detach(|| self.post_binary(&url, &body))?;
        decode_remote_batch_search(py, &raw, n_queries)
    }

    fn close(&self) {}
}

impl PyRemoteHttpClient {
    fn url_with_query<K>(&self, path: &str, params: &[(K, String)]) -> String
    where
        K: AsRef<str>,
    {
        let mut url = if path.starts_with("http://") || path.starts_with("https://") {
            path.to_string()
        } else {
            format!("{}{}", self.base_url, path)
        };
        if !params.is_empty() {
            if url.contains('?') {
                url.push('&');
            } else {
                url.push('?');
            }
            for (idx, (key, value)) in params.iter().enumerate() {
                if idx > 0 {
                    url.push('&');
                }
                url.push_str(&percent_encode(key.as_ref()));
                url.push('=');
                url.push_str(&percent_encode(value));
            }
        }
        url
    }

    fn request_raw(
        &self,
        method: &str,
        url: &str,
        body: Option<&[u8]>,
        content_type: Option<&str>,
    ) -> PyResult<(u16, Vec<u8>)> {
        let mut request = match method {
            "GET" => self.agent.get(url),
            "POST" => self.agent.post(url),
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unsupported HTTP method '{}'",
                    other
                )))
            }
        };
        if let Some(api_key) = &self.api_key {
            request = request.set("Authorization", &format!("Bearer {}", api_key));
        }
        if let Some(content_type) = content_type {
            request = request.set("Content-Type", content_type);
        }

        let result = match body {
            Some(bytes) => request.send_bytes(bytes),
            None => request.call(),
        };
        match result {
            Ok(response) => {
                let status = response.status();
                let raw = read_ureq_response(response)?;
                Ok((status, raw))
            }
            Err(ureq::Error::Status(status, response)) => {
                let raw = read_ureq_response(response)?;
                Ok((status, raw))
            }
            Err(ureq::Error::Transport(err)) => Err(pyo3::exceptions::PyConnectionError::new_err(
                format!("remote HTTP request failed: {}", err),
            )),
        }
    }

    fn post_binary(&self, url: &str, body: &[u8]) -> PyResult<Vec<u8>> {
        let (status, raw) =
            self.request_raw("POST", url, Some(body), Some("application/octet-stream"))?;
        if (200..300).contains(&status) {
            Ok(raw)
        } else {
            Err(remote_http_status_error(status, raw))
        }
    }
}

/// Python wrapper around the Rust-side hot read fan-out coordinator.
#[pyclass(name = "ClusterReadCoordinator")]
pub struct PyClusterReadCoordinator {
    inner: crate::cluster::RustReadCoordinator,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl PyClusterReadCoordinator {
    #[new]
    #[pyo3(signature = (cluster_state_path, timeout_secs=30.0, api_key=None))]
    fn new(cluster_state_path: &str, timeout_secs: f64, api_key: Option<String>) -> PyResult<Self> {
        let timeout = Duration::from_secs_f64(timeout_secs.max(0.001));
        let inner =
            crate::cluster::RustReadCoordinator::from_path(cluster_state_path, timeout, api_key)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner, runtime })
    }

    fn search_binary<'py>(
        &self,
        py: Python<'py>,
        meta_json: &str,
        raw: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let meta = parse_cluster_meta(meta_json)?;
        let raw = raw.as_bytes().to_vec();
        let output = py
            .detach(|| self.runtime.block_on(self.inner.search_binary(meta, raw)))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &output))
    }

    fn batch_search_binary<'py>(
        &self,
        py: Python<'py>,
        meta_json: &str,
        raw: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let meta = parse_cluster_meta(meta_json)?;
        let raw = raw.as_bytes().to_vec();
        let output = py
            .detach(|| {
                self.runtime
                    .block_on(self.inner.batch_search_binary(meta, raw))
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &output))
    }
}

fn parse_cluster_meta(meta_json: &str) -> PyResult<serde_json::Value> {
    serde_json::from_str(meta_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (owner_uri, key, timeout_secs=30.0, api_key=None))]
fn metadata_rpc_get(
    py: Python<'_>,
    owner_uri: String,
    key: String,
    timeout_secs: f64,
    api_key: Option<String>,
) -> PyResult<(u64, Option<String>)> {
    let timeout = Duration::from_secs_f64(timeout_secs.max(0.001));
    let mut meta = serde_json::Map::new();
    meta.insert("key".to_string(), serde_json::Value::String(key));
    if let Some(api_key) = api_key {
        meta.insert("api_key".to_string(), serde_json::Value::String(api_key));
    }
    let meta = serde_json::Value::Object(meta);

    let (response_meta, raw) = py.detach(move || {
        metadata_rpc_request(&owner_uri, crate::rpc::OP_METADATA_GET, meta, &[], timeout)
    })?;
    let version = metadata_response_version(&response_meta)?;
    let exists = response_meta
        .get("exists")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(!raw.is_empty());
    if !exists {
        return Ok((version, None));
    }
    let value = String::from_utf8(raw)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((version, Some(value)))
}

#[pyfunction]
#[pyo3(signature = (owner_uri, key, expected_version, value_json, timeout_secs=30.0, api_key=None))]
fn metadata_rpc_cas(
    py: Python<'_>,
    owner_uri: String,
    key: String,
    expected_version: u64,
    value_json: String,
    timeout_secs: f64,
    api_key: Option<String>,
) -> PyResult<u64> {
    let timeout = Duration::from_secs_f64(timeout_secs.max(0.001));
    let mut meta = serde_json::Map::new();
    meta.insert("key".to_string(), serde_json::Value::String(key));
    meta.insert(
        "expected_version".to_string(),
        serde_json::Value::Number(expected_version.into()),
    );
    if let Some(api_key) = api_key {
        meta.insert("api_key".to_string(), serde_json::Value::String(api_key));
    }
    let meta = serde_json::Value::Object(meta);
    let raw = value_json.into_bytes();

    let (response_meta, _) = py.detach(move || {
        metadata_rpc_request(&owner_uri, crate::rpc::OP_METADATA_CAS, meta, &raw, timeout)
    })?;
    metadata_response_version(&response_meta)
}

#[pyfunction]
#[pyo3(signature = (owner_uri=None))]
fn metadata_rpc_close(owner_uri: Option<String>) -> PyResult<usize> {
    let streams = if let Some(owner_uri) = owner_uri {
        let normalized_owner = owner_uri.trim().trim_end_matches('/');
        let (host, port) = derive_metadata_rpc_target(normalized_owner)?;
        let pool_key = format!("{}|{}:{}", normalized_owner, host, port);
        let mut pool = metadata_rpc_pool().lock();
        pool.remove(&pool_key).unwrap_or_default()
    } else {
        let mut pool = metadata_rpc_pool().lock();
        pool.drain()
            .flat_map(|(_key, streams)| streams)
            .collect::<Vec<_>>()
    };
    let count = streams.len();
    for stream in streams {
        let _ = stream.shutdown(Shutdown::Both);
    }
    Ok(count)
}

fn metadata_rpc_request(
    owner_uri: &str,
    op: u8,
    meta: serde_json::Value,
    raw: &[u8],
    timeout: Duration,
) -> PyResult<(serde_json::Value, Vec<u8>)> {
    let (host, port) = derive_metadata_rpc_target(owner_uri)?;
    let meta_bytes = serde_json::to_vec(&meta)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let payload_len = 1usize
        .checked_add(4)
        .and_then(|len| len.checked_add(meta_bytes.len()))
        .and_then(|len| len.checked_add(raw.len()))
        .ok_or_else(|| py_runtime_err("metadata RPC payload is too large"))?;
    if payload_len > u32::MAX as usize || payload_len > MAX_METADATA_RPC_FRAME_BYTES {
        return Err(py_runtime_err(format!(
            "metadata RPC payload too large: {} bytes",
            payload_len
        )));
    }

    let mut payload = Vec::with_capacity(payload_len);
    payload.push(op);
    payload.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
    payload.extend_from_slice(&meta_bytes);
    payload.extend_from_slice(raw);

    let pool_key = format!("{}|{}:{}", owner_uri, host, port);
    let (mut stream, pooled) = match take_metadata_rpc_stream(&pool_key) {
        Some(stream) => (stream, true),
        None => (
            connect_metadata_rpc(owner_uri, &host, port, timeout)?,
            false,
        ),
    };
    match metadata_rpc_roundtrip(owner_uri, &mut stream, &payload, timeout) {
        Ok(response) => {
            return_metadata_rpc_stream(&pool_key, stream);
            Ok(response)
        }
        Err(_err) if pooled => {
            drop(stream);
            let mut fresh = connect_metadata_rpc(owner_uri, &host, port, timeout)?;
            let response = metadata_rpc_roundtrip(owner_uri, &mut fresh, &payload, timeout)?;
            return_metadata_rpc_stream(&pool_key, fresh);
            Ok(response)
        }
        Err(err) => Err(err),
    }
}

fn metadata_rpc_roundtrip(
    owner_uri: &str,
    stream: &mut TcpStream,
    payload: &[u8],
    timeout: Duration,
) -> PyResult<(serde_json::Value, Vec<u8>)> {
    stream
        .set_nodelay(true)
        .map_err(|e| py_runtime_err(format!("metadata RPC setup failed: {}", e)))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| py_runtime_err(format!("metadata RPC setup failed: {}", e)))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| py_runtime_err(format!("metadata RPC setup failed: {}", e)))?;

    stream
        .write_all(&(payload.len() as u32).to_le_bytes())
        .and_then(|_| stream.write_all(&payload))
        .map_err(|e| {
            py_runtime_err(format!("metadata RPC write to {} failed: {}", owner_uri, e))
        })?;

    let mut header = [0u8; 4];
    stream.read_exact(&mut header).map_err(|e| {
        py_runtime_err(format!(
            "metadata RPC read from {} failed: {}",
            owner_uri, e
        ))
    })?;
    let frame_len = u32::from_le_bytes(header) as usize;
    if frame_len > MAX_METADATA_RPC_FRAME_BYTES {
        return Err(py_runtime_err(format!(
            "metadata RPC response too large: {} bytes",
            frame_len
        )));
    }
    let mut frame = vec![0u8; frame_len];
    stream.read_exact(&mut frame).map_err(|e| {
        py_runtime_err(format!(
            "metadata RPC read from {} failed: {}",
            owner_uri, e
        ))
    })?;

    decode_metadata_rpc_response(&frame)
}

fn metadata_rpc_pool() -> &'static Mutex<HashMap<String, Vec<TcpStream>>> {
    METADATA_RPC_POOL.get_or_init(|| Mutex::new(HashMap::new()))
}

fn take_metadata_rpc_stream(pool_key: &str) -> Option<TcpStream> {
    let mut pool = metadata_rpc_pool().lock();
    let stream = pool.get_mut(pool_key).and_then(Vec::pop);
    if pool.get(pool_key).is_some_and(Vec::is_empty) {
        pool.remove(pool_key);
    }
    stream
}

fn return_metadata_rpc_stream(pool_key: &str, stream: TcpStream) {
    let mut pool = metadata_rpc_pool().lock();
    let streams = pool.entry(pool_key.to_string()).or_default();
    if streams.len() < MAX_METADATA_RPC_IDLE_PER_OWNER {
        streams.push(stream);
    }
}

fn connect_metadata_rpc(
    owner_uri: &str,
    host: &str,
    port: u16,
    timeout: Duration,
) -> PyResult<TcpStream> {
    let addrs = (host, port).to_socket_addrs().map_err(|e| {
        py_runtime_err(format!(
            "metadata RPC resolve failed for {}: {}",
            owner_uri, e
        ))
    })?;
    let mut last_error = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, timeout) {
            Ok(stream) => return Ok(stream),
            Err(err) => last_error = Some(err),
        }
    }
    let message = match last_error {
        Some(err) => format!("metadata RPC connect to {} failed: {}", owner_uri, err),
        None => format!(
            "metadata RPC resolve failed for {}: no socket addresses",
            owner_uri
        ),
    };
    Err(py_runtime_err(message))
}

fn decode_metadata_rpc_response(frame: &[u8]) -> PyResult<(serde_json::Value, Vec<u8>)> {
    if frame.len() < 5 {
        return Err(py_runtime_err("metadata RPC response frame is too short"));
    }
    let status = frame[0];
    let meta_len = u32::from_le_bytes([frame[1], frame[2], frame[3], frame[4]]) as usize;
    if frame.len() < 5 + meta_len {
        return Err(py_runtime_err(
            "metadata RPC response metadata length exceeds frame",
        ));
    }
    let meta = if meta_len == 0 {
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        serde_json::from_slice(&frame[5..5 + meta_len])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };
    if status != 0 {
        let message = meta
            .get("error")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("metadata RPC request failed");
        return Err(py_runtime_err(message.to_string()));
    }
    Ok((meta, frame[5 + meta_len..].to_vec()))
}

fn metadata_response_version(meta: &serde_json::Value) -> PyResult<u64> {
    meta.get("version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| py_runtime_err("metadata RPC response missing version"))
}

fn derive_metadata_rpc_target(uri: &str) -> PyResult<(String, u16)> {
    let trimmed = uri.trim().trim_end_matches('/');
    let (scheme, without_scheme) = if let Some(rest) = trimmed.strip_prefix("http://") {
        ("http", rest)
    } else if let Some(rest) = trimmed.strip_prefix("https://") {
        ("https", rest)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid metadata owner URI '{}'",
            uri
        )));
    };
    let authority = without_scheme.split('/').next().unwrap_or(without_scheme);
    let authority = authority
        .rsplit_once('@')
        .map(|(_, host)| host)
        .unwrap_or(authority);
    let default_port = if scheme == "https" { 443 } else { 80 };

    let (host, http_port) = if let Some(rest) = authority.strip_prefix('[') {
        let end = rest.find(']').ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid metadata owner URI '{}'", uri))
        })?;
        let host = &rest[..end];
        let after_host = &rest[end + 1..];
        let port = if let Some(port_text) = after_host.strip_prefix(':') {
            parse_metadata_uri_port(uri, port_text)?
        } else {
            default_port
        };
        (host.to_string(), port)
    } else if let Some((host, port_text)) = authority.rsplit_once(':') {
        (host.to_string(), parse_metadata_uri_port(uri, port_text)?)
    } else {
        (authority.to_string(), default_port)
    };
    if host.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid metadata owner URI '{}'",
            uri
        )));
    }
    Ok((host, crate::rpc::derive_rpc_port(http_port)))
}

fn parse_metadata_uri_port(uri: &str, port_text: &str) -> PyResult<u16> {
    port_text.parse::<u16>().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "invalid metadata owner URI port in '{}': {}",
            uri, e
        ))
    })
}

fn py_runtime_err<T: ToString>(message: T) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(message.to_string())
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

    #[staticmethod]
    fn open_read_only(root_path: &str) -> PyResult<Self> {
        let engine = DatabaseEngine::open_read_only(Path::new(root_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: engine })
    }

    fn is_read_only(&self) -> bool {
        self.inner.is_read_only()
    }

    /// Create a new collection.
    #[pyo3(signature = (name, dimension, dtypes=None))]
    fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        dtypes: Option<&str>,
    ) -> PyResult<PyCollection> {
        let vector_dtype = parse_py_vector_dtype(dtypes)?;
        let coll = self
            .inner
            .create_collection_with_dtype(name, dimension, 100_000, vector_dtype)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyCollection { inner: coll })
    }

    /// Get or open an existing collection.
    #[pyo3(signature = (name, dimension, dtypes=None))]
    fn get_collection(
        &self,
        name: &str,
        dimension: usize,
        dtypes: Option<&str>,
    ) -> PyResult<PyCollection> {
        let vector_dtype = dtypes
            .map(VectorDtype::parse)
            .transpose()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let coll = self
            .inner
            .get_or_open_collection_with_dtype(name, dimension, 100_000, vector_dtype)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyCollection { inner: coll })
    }

    /// Drop a collection.
    fn drop_collection(&self, name: &str) -> PyResult<()> {
        self.inner
            .drop_collection(name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn snapshot_collection(&self, name: &str, snapshot_path: &str) -> PyResult<()> {
        self.inner
            .snapshot_collection(name, Path::new(snapshot_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn export_collection(&self, name: &str, export_path: &str) -> PyResult<()> {
        self.inner
            .export_collection(name, Path::new(export_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (name, snapshot_path, overwrite=false))]
    fn restore_collection(&self, name: &str, snapshot_path: &str, overwrite: bool) -> PyResult<()> {
        self.inner
            .restore_collection_from_snapshot(name, Path::new(snapshot_path), overwrite)
            .map(|_| ())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (name, export_path, overwrite=false))]
    fn import_collection(&self, name: &str, export_path: &str, overwrite: bool) -> PyResult<()> {
        self.inner
            .import_collection_from_export(name, Path::new(export_path), overwrite)
            .map(|_| ())
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
    /// Add records with public string/integer external IDs.
    #[pyo3(signature = (vectors, ids, fields=None))]
    fn add_records<'py>(
        &self,
        py: Python<'py>,
        vectors: PyReadonlyArray2<f32>,
        ids: &Bound<'_, PyList>,
        fields: Option<&Bound<'_, PyList>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let array = vectors.as_array();
        let n_vectors = array.nrows();
        let flat_data: &[f32] = array
            .as_slice()
            .expect("numpy array must be contiguous (C-order)");
        let external_ids = pylist_to_external_ids(ids)?;
        let rust_fields = py_fields_to_json(fields, n_vectors)?;

        let mut coll = self.inner.write();
        coll.add_records(flat_data, n_vectors, &external_ids, rust_fields.as_deref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        external_ids_to_pylist(py, &external_ids)
    }

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

        let rust_fields = py_fields_to_json(fields, n_vectors)?;

        let mut coll = self.inner.write();
        coll.add_items(&flat_data, n_vectors, &ids, rust_fields.as_deref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Add float16-encoded vectors with user-specified IDs and optional metadata.
    ///
    /// Args:
    ///     vectors: numpy array of shape (n, dim) with dtype uint16 containing
    ///         little-endian IEEE float16 bits.
    ///     ids: list of integer user IDs, one per vector
    ///     fields: optional list of dicts, one per vector
    #[pyo3(signature = (vectors, ids, fields=None))]
    fn add_items_encoded_f16(
        &self,
        vectors: PyReadonlyArray2<u16>,
        ids: Vec<u64>,
        fields: Option<&Bound<'_, PyList>>,
    ) -> PyResult<()> {
        let array = vectors.as_array();
        let n_vectors = array.nrows();
        let flat_data: &[u16] = array
            .as_slice()
            .expect("numpy array must be contiguous (C-order)");
        let encoded = unsafe {
            std::slice::from_raw_parts(
                flat_data.as_ptr().cast::<u8>(),
                flat_data.len() * std::mem::size_of::<u16>(),
            )
        };

        let rust_fields = py_fields_to_json(fields, n_vectors)?;

        let mut coll = self.inner.write();
        coll.add_items_encoded_vectors(
            encoded,
            VectorDtype::F16,
            n_vectors,
            &ids,
            rust_fields.as_deref(),
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Create an additional named vector field for existing records.
    #[pyo3(signature = (name, dimension, metric=None, index_mode=None, dtypes=None))]
    fn create_vector_field(
        &self,
        name: &str,
        dimension: usize,
        metric: Option<&str>,
        index_mode: Option<&str>,
        dtypes: Option<&str>,
    ) -> PyResult<()> {
        let mut coll = self.inner.write();
        coll.create_vector_field_with_dtype(name, dimension, metric, index_mode, dtypes)
            .map(|_| ())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// List vector fields, including the reserved default primary vector field.
    fn list_vector_fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let coll = self.inner.read();
        vector_fields_to_pylist(py, &coll.list_vector_fields())
    }

    /// Add vectors to a named vector field for existing IDs.
    fn add_named_vectors(
        &self,
        field_name: &str,
        vectors: PyReadonlyArray2<f32>,
        ids: Vec<u64>,
    ) -> PyResult<()> {
        let array = vectors.as_array();
        let n_vectors = array.nrows();
        let flat_data: &[f32] = array
            .as_slice()
            .expect("numpy array must be contiguous (C-order)");

        let mut coll = self.inner.write();
        coll.add_named_vectors(field_name, flat_data, n_vectors, &ids)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Build or change the index for a named vector field.
    #[pyo3(signature = (field_name, index_type, n_clusters=None))]
    fn build_vector_field_index(
        &self,
        field_name: &str,
        index_type: &str,
        n_clusters: Option<usize>,
    ) -> PyResult<()> {
        let mut coll = self.inner.write();
        coll.build_vector_field_index_with_options(field_name, index_type, n_clusters)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Remove a named vector field index and return it to flat search.
    fn remove_vector_field_index(&self, field_name: &str) -> PyResult<()> {
        let mut coll = self.inner.write();
        coll.remove_vector_field_index(field_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Add sparse vectors for existing IDs.
    fn add_sparse_vectors(&self, vectors: Vec<Vec<(u32, f32)>>, ids: Vec<u64>) -> PyResult<()> {
        let mut coll = self.inner.write();
        coll.add_sparse_vectors(&ids, &vectors)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Build or rebuild the index.
    ///
    /// Args:
    ///     index_type: e.g. "IVF-IP-SQ8", "HNSW-L2", "Flat-Cos", etc.
    #[pyo3(signature = (index_type, n_clusters=None))]
    fn build_index(&self, index_type: &str, n_clusters: Option<usize>) -> PyResult<()> {
        let mut coll = self.inner.write();
        coll.build_index_with_options(index_type, n_clusters)
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
    ///     nprobe: number of IVF/SPANN probes or HNSW search breadth (default 10)
    ///     approx: flat-search approximation for IP/L2/Cosine only; ignored for Hamming/Jaccard
    ///     eps: distance rounding tolerance when approx applies
    ///
    /// Returns:
    ///     PySearchResult with ids, distances, fields
    #[pyo3(signature = (vector, k=None, where_expr=None, nprobe=None, approx=None, eps=None))]
    fn search(
        &self,
        vector: PyReadonlyArray1<f32>,
        k: Option<usize>,
        where_expr: Option<&str>,
        nprobe: Option<usize>,
        approx: Option<bool>,
        eps: Option<f32>,
    ) -> PyResult<PySearchResult> {
        let query = vector.as_slice()?;
        let k = k.unwrap_or(10);
        let nprobe = nprobe.unwrap_or(10);
        let approx = approx.unwrap_or(false);
        let eps = eps.unwrap_or(1e-4);

        let coll = self.inner.read();
        let result = coll
            .search(query, k, where_expr, nprobe, approx, eps)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PySearchResult { inner: result })
    }

    /// Search a named vector field.
    ///
    /// approx applies only to IP/L2/Cosine fields; Hamming/Jaccard fields always
    /// use exact binary-distance search.
    #[pyo3(signature = (field_name, vector, k=None, where_expr=None, approx=None, eps=None))]
    fn search_vector_field(
        &self,
        field_name: &str,
        vector: PyReadonlyArray1<f32>,
        k: Option<usize>,
        where_expr: Option<&str>,
        approx: Option<bool>,
        eps: Option<f32>,
    ) -> PyResult<PySearchResult> {
        let query = vector.as_slice()?;
        let k = k.unwrap_or(10);
        let approx = approx.unwrap_or(false);
        let eps = eps.unwrap_or(1e-4);

        let coll = self.inner.read();
        let result = coll
            .search_vector_field_with_options(field_name, query, k, where_expr, approx, eps)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PySearchResult { inner: result })
    }

    /// Search sparse vectors with inner product.
    #[pyo3(signature = (vector, k=None, where_expr=None))]
    fn search_sparse(
        &self,
        vector: Vec<(u32, f32)>,
        k: Option<usize>,
        where_expr: Option<&str>,
    ) -> PyResult<PySearchResult> {
        let k = k.unwrap_or(10);

        let coll = self.inner.read();
        let result = coll
            .search_sparse(&vector, k, where_expr)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PySearchResult { inner: result })
    }

    /// Search and return profile/explain metadata as a Python dict.
    #[pyo3(signature = (vector, k=None, where_expr=None, nprobe=None, approx=None, eps=None))]
    fn search_profile<'py>(
        &self,
        py: Python<'py>,
        vector: PyReadonlyArray1<f32>,
        k: Option<usize>,
        where_expr: Option<&str>,
        nprobe: Option<usize>,
        approx: Option<bool>,
        eps: Option<f32>,
    ) -> PyResult<PyObject> {
        let query = vector.as_slice()?;
        let k = k.unwrap_or(10);
        let nprobe = nprobe.unwrap_or(10);
        let approx = approx.unwrap_or(false);
        let eps = eps.unwrap_or(1e-4);

        let coll = self.inner.read();
        let (result, profile) = coll
            .search_with_profile(query, k, where_expr, nprobe, approx, eps)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let value = serde_json::json!({
            "items": {
                "k": result.k,
                "ids": result.ids,
                "scores": result.distances,
                "index": result.index_mode,
            },
            "profile": profile,
        });
        json_to_py(py, &value)
    }

    /// BM25 text search over metadata fields.
    #[pyo3(signature = (text, text_fields=None, k=10, where_expr=None))]
    fn text_search(
        &self,
        text: &str,
        text_fields: Option<Vec<String>>,
        k: usize,
        where_expr: Option<&str>,
    ) -> PyResult<PySearchResult> {
        let coll = self.inner.read();
        let result = coll
            .text_search(text, text_fields.as_deref(), k, where_expr)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PySearchResult { inner: result })
    }

    /// Hybrid vector + BM25 text search with RRF or weighted fusion.
    #[pyo3(signature = (
        vector=None,
        text=None,
        k=10,
        where_expr=None,
        text_fields=None,
        fusion="rrf",
        vector_weight=1.0,
        text_weight=1.0,
        rrf_k=60.0,
        candidate_limit=None,
        nprobe=10
    ))]
    fn hybrid_search(
        &self,
        vector: Option<PyReadonlyArray1<f32>>,
        text: Option<&str>,
        k: usize,
        where_expr: Option<&str>,
        text_fields: Option<Vec<String>>,
        fusion: &str,
        vector_weight: f32,
        text_weight: f32,
        rrf_k: f32,
        candidate_limit: Option<usize>,
        nprobe: usize,
    ) -> PyResult<PySearchResult> {
        let vector_slice = match vector.as_ref() {
            Some(vector) => Some(vector.as_slice()?),
            None => None,
        };
        let candidate_limit = candidate_limit.unwrap_or_else(|| k.saturating_mul(4).max(k));

        let coll = self.inner.read();
        let result = coll
            .hybrid_search(
                vector_slice,
                text,
                k,
                where_expr,
                text_fields.as_deref(),
                fusion,
                vector_weight,
                text_weight,
                rrf_k,
                candidate_limit,
                nprobe,
            )
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

    fn vector_dtype(&self) -> String {
        self.inner.read().vector_dtype().storage_name().to_string()
    }

    /// Get current index mode.
    fn index_mode(&self) -> Option<String> {
        self.inner.read().meta.index_mode.clone()
    }

    /// Check whether the collection handle was opened read-only.
    fn is_read_only(&self) -> bool {
        self.inner.read().is_read_only()
    }

    /// Batch search: search multiple query vectors in parallel.
    ///
    /// Args:
    ///     vectors: numpy array of shape (n_queries, dim) with dtype float32
    ///     k: number of results per query
    ///     where: optional SQL-like filter string
    ///     nprobe: number of IVF/SPANN probes or HNSW search breadth (default 10)
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

    /// Commit: clear WAL after successful writes without forcing recursive fsync.
    /// Use checkpoint() for an explicit durable fsync barrier.
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

    /// Lightweight checkpoint: clear WAL without forcing recursive fsync.
    fn checkpoint_fast(&self) -> PyResult<()> {
        let coll = self.inner.read();
        coll.checkpoint_fast()
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

        let arr = data.into_pyarray(py);
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

        let arr = data.into_pyarray(py);
        let field_list = fields_to_pylist(py, &fields)?;
        Ok((arr, ids, field_list))
    }

    /// Get vector store info for zero-copy mmap access.
    ///
    /// Returns (path, n_vectors, dimension) so Python can create:
    ///   vectors = np.memmap(path, dtype=collection.vector_dtype, mode='r').reshape(n, dim)
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
    fn vectors_numpy<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let coll = self.inner.read();
        let dtype = coll.vector_dtype().numpy_name().to_string();
        let (path, n, dim) = coll
            .vector_store_info()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        drop(coll);

        if n == 0 {
            // Return empty array
            let np = py.import("numpy")?;
            let empty = np.call_method1("zeros", ((0, dim), dtype.as_str()))?;
            return Ok(empty.unbind());
        }

        let np = py.import("numpy")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", dtype)?;
        kwargs.set_item("mode", "r")?;
        kwargs.set_item("shape", (n as usize, dim))?;
        let memmap = np.call_method("memmap", (path,), Some(&kwargs))?;
        Ok(memmap.unbind())
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

        let flat = coll
            .read_vectors_by_ids_only(&ids)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let arr = unsafe { numpy::PyArray2::<f32>::new(py, [n_ids, dim], false) };
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

    /// Query fields and return public external IDs as efficiently as possible.
    ///
    /// Integer-only IDs are returned as a numpy int64 array; mixed/string IDs
    /// fall back to a Python list.
    fn query_external_ids_array<'py>(
        &self,
        py: Python<'py>,
        where_expr: &str,
    ) -> PyResult<Py<PyAny>> {
        let coll = self.inner.read();
        let internal_ids = coll
            .query_fields(where_expr)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        if let Some(values) = coll.external_int_ids_for_internal_ids(&internal_ids) {
            return Ok(values.into_pyarray(py).into_any().unbind());
        }
        Ok(
            external_ids_to_pylist(py, &coll.external_ids_for_internal_ids(&internal_ids))?
                .into_any()
                .unbind(),
        )
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

    /// Convert internal numeric IDs to public external IDs.
    fn external_ids<'py>(&self, py: Python<'py>, ids: Vec<u64>) -> PyResult<Bound<'py, PyList>> {
        let coll = self.inner.read();
        external_ids_to_pylist(py, &coll.external_ids_for_internal_ids(&ids))
    }

    /// Convert internal numeric IDs to public external IDs as efficiently as possible.
    ///
    /// Integer-only IDs are returned as a numpy int64 array; mixed/string IDs
    /// fall back to a Python list for object dtype handling in the Python layer.
    fn external_ids_array<'py>(&self, py: Python<'py>, ids: Vec<u64>) -> PyResult<Py<PyAny>> {
        let coll = self.inner.read();
        if let Some(values) = coll.external_int_ids_for_internal_ids(&ids) {
            return Ok(values.into_pyarray(py).into_any().unbind());
        }
        let external_ids = coll.external_ids_for_internal_ids(&ids);

        Ok(external_ids_to_pylist(py, &external_ids)?
            .into_any()
            .unbind())
    }

    /// Convert public external IDs to internal numeric IDs.
    fn internal_ids(&self, ids: &Bound<'_, PyList>) -> PyResult<Vec<u64>> {
        let external_ids = pylist_to_external_ids(ids)?;
        let coll = self.inner.read();
        coll.internal_ids_for_external_ids(&external_ids)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Check whether a public external ID exists in the collection.
    fn is_external_id_exists(&self, id: &Bound<'_, PyAny>) -> PyResult<bool> {
        let external_id = py_to_external_id(id)?;
        let coll = self.inner.read();
        Ok(coll.is_external_id_exists(&external_id))
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
    /// For ascending distance metrics: returns IDs where distance ≤ threshold.
    /// For descending inner-product scores: returns IDs where score ≥ threshold.
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

    fn snapshot_to(&self, snapshot_path: &str) -> PyResult<()> {
        let coll = self.inner.read();
        coll.snapshot_to(Path::new(snapshot_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn export_to(&self, export_path: &str) -> PyResult<()> {
        let coll = self.inner.read();
        coll.export_to(Path::new(export_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
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
        ids.into_pyarray(py)
    }

    /// Get result distances as numpy array (zero-copy transfer to numpy).
    fn distances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.distances.clone().into_pyarray(py)
    }

    /// Get result fields as list of dicts.
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for field_map in &self.inner.fields {
            let dict = PyDict::new(py);
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
        let inner = crate::storage::flat_mmap::FlatMmap::open(
            std::path::Path::new(path),
            dim,
            VectorDtype::F32,
        )
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
        let (ids, dists) = self.inner.search(q, k, metric, false, None);
        Ok((ids.into_pyarray(py), dists.into_pyarray(py)))
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
            let (ids, dists) =
                self.inner
                    .search(&flat[q_start..q_end], k, metric_enum, false, None);
            results.push((ids.into_pyarray(py), dists.into_pyarray(py)));
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
        Ok((ids.into_pyarray(py), dists.into_pyarray(py)))
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
    if !metric.accepts_dimension(a_slice.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "haversine requires two values in [longitude_degrees, latitude_degrees] order",
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
    if q.len() != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Query dimension must match candidate dimension",
        ));
    }
    if !metric.accepts_dimension(dim) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "haversine requires two values in [longitude_degrees, latitude_degrees] order",
        ));
    }

    // Zero-copy: get contiguous f32 slice directly from numpy memory
    let flat: &[f32] = c_array
        .as_slice()
        .expect("numpy candidates array must be contiguous (C-order)");

    let (ids, dists) = crate::distance::top_k_search(q, flat, dim, k, metric);

    Ok((ids.into_pyarray(py), dists.into_pyarray(py)))
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

    #[staticmethod]
    fn open_read_only(root_path: &str) -> PyResult<Self> {
        let mgr = DatabaseManager::new_read_only(Path::new(root_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(mgr),
        })
    }

    fn is_read_only(&self) -> bool {
        self.inner.is_read_only()
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

    fn snapshot_database(&self, name: &str, snapshot_path: &str) -> PyResult<()> {
        self.inner
            .snapshot_database(name, Path::new(snapshot_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (name, snapshot_path, overwrite=false))]
    fn restore_database(&self, name: &str, snapshot_path: &str, overwrite: bool) -> PyResult<()> {
        self.inner
            .restore_database_from_snapshot(name, Path::new(snapshot_path), overwrite)
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

    #[pyo3(signature = (db_name, collection_name, dim=None, drop_if_exists=None, description=None, dtypes=None))]
    fn require_collection(
        &self,
        db_name: &str,
        collection_name: &str,
        dim: Option<usize>,
        drop_if_exists: Option<bool>,
        description: Option<&str>,
        dtypes: Option<&str>,
    ) -> PyResult<()> {
        self.inner
            .require_collection_with_dtype(
                db_name,
                collection_name,
                dim.unwrap_or(0),
                100_000,
                drop_if_exists.unwrap_or(false),
                description,
                dtypes,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn drop_collection(&self, db_name: &str, collection_name: &str) -> PyResult<()> {
        self.inner
            .drop_collection(db_name, collection_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn snapshot_collection(
        &self,
        db_name: &str,
        collection_name: &str,
        snapshot_path: &str,
    ) -> PyResult<()> {
        self.inner
            .snapshot_collection(db_name, collection_name, Path::new(snapshot_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn export_collection(
        &self,
        db_name: &str,
        collection_name: &str,
        export_path: &str,
    ) -> PyResult<()> {
        self.inner
            .export_collection(db_name, collection_name, Path::new(export_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (db_name, collection_name, snapshot_path, overwrite=false))]
    fn restore_collection(
        &self,
        db_name: &str,
        collection_name: &str,
        snapshot_path: &str,
        overwrite: bool,
    ) -> PyResult<()> {
        self.inner
            .restore_collection_from_snapshot(
                db_name,
                collection_name,
                Path::new(snapshot_path),
                overwrite,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (db_name, collection_name, export_path, overwrite=false))]
    fn import_collection(
        &self,
        db_name: &str,
        collection_name: &str,
        export_path: &str,
        overwrite: bool,
    ) -> PyResult<()> {
        self.inner
            .import_collection_from_export(
                db_name,
                collection_name,
                Path::new(export_path),
                overwrite,
            )
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
    ) -> PyResult<Option<(usize, usize, Option<String>, String)>> {
        let configs = self
            .inner
            .get_collection_configs(db_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(configs
            .get(collection_name)
            .map(|c| (c.dim, c.chunk_size, c.description.clone(), c.dtypes.clone())))
    }

    /// Get all collection configs from collections.json in one read.
    fn get_collection_configs(
        &self,
        db_name: &str,
    ) -> PyResult<HashMap<String, (usize, usize, Option<String>, String)>> {
        let configs = self
            .inner
            .get_collection_configs(db_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(configs
            .into_iter()
            .map(|(name, c)| (name, (c.dim, c.chunk_size, c.description, c.dtypes)))
            .collect())
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

fn normalize_remote_vector_encoding(value: &str) -> PyResult<String> {
    match value.to_ascii_lowercase().as_str() {
        "" | "float32" | "f32" => Ok("float32".to_string()),
        "float16" | "f16" => Ok("float16".to_string()),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported vector_encoding '{}'",
            other
        ))),
    }
}

fn encode_f32_values_for_wire<I>(values: I, vector_encoding: &str) -> PyResult<Vec<u8>>
where
    I: IntoIterator<Item = f32>,
{
    match normalize_remote_vector_encoding(vector_encoding)?.as_str() {
        "float16" => {
            let mut out = Vec::new();
            for value in values {
                out.extend_from_slice(&half::f16::from_f32(value).to_le_bytes());
            }
            Ok(out)
        }
        _ => {
            let mut out = Vec::new();
            for value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
            Ok(out)
        }
    }
}

fn encode_f32_slice_for_wire(values: &[f32], vector_encoding: &str) -> PyResult<Vec<u8>> {
    match normalize_remote_vector_encoding(vector_encoding)?.as_str() {
        "float16" => encode_f32_values_for_wire(values.iter().copied(), "float16"),
        _ => {
            #[cfg(target_endian = "little")]
            {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        values.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(values),
                    )
                };
                Ok(bytes.to_vec())
            }
            #[cfg(not(target_endian = "little"))]
            {
                encode_f32_values_for_wire(values.iter().copied(), "float32")
            }
        }
    }
}

fn encode_f32_matrix_for_wire(
    matrix: ArrayView2<'_, f32>,
    vector_encoding: &str,
) -> PyResult<Vec<u8>> {
    if let Some(values) = matrix.as_slice() {
        encode_f32_slice_for_wire(values, vector_encoding)
    } else {
        encode_f32_values_for_wire(matrix.iter().copied(), vector_encoding)
    }
}

fn contiguous_id_start(ids: &[u64]) -> Option<u64> {
    let (&first, rest) = ids.split_first()?;
    if rest
        .iter()
        .enumerate()
        .all(|(offset, &id)| id == first + offset as u64 + 1)
    {
        Some(first)
    } else {
        None
    }
}

fn percent_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z'
            | b'a'..=b'z'
            | b'0'..=b'9'
            | b'-'
            | b'_'
            | b'.'
            | b'~' => out.push(byte as char),
            _ => {
                out.push('%');
                out.push_str(&format!("{:02X}", byte));
            }
        }
    }
    out
}

fn query_params_from_pydict(
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<(String, String)>> {
    let Some(params) = params else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(params.len());
    for (key, value) in params.iter() {
        if value.is_none() {
            continue;
        }
        let key = key.str()?.to_str()?.to_string();
        let value = if let Ok(value) = value.extract::<bool>() {
            value.to_string()
        } else if let Ok(value) = value.extract::<i64>() {
            value.to_string()
        } else if let Ok(value) = value.extract::<u64>() {
            value.to_string()
        } else if let Ok(value) = value.extract::<f64>() {
            value.to_string()
        } else if let Ok(value) = value.extract::<String>() {
            value
        } else {
            value.str()?.to_str()?.to_string()
        };
        out.push((key, value));
    }
    Ok(out)
}

fn remote_response_tuple<'py>(
    py: Python<'py>,
    status: u16,
    raw: Vec<u8>,
) -> PyResult<Bound<'py, PyTuple>> {
    let (json_ok, json_value) = match serde_json::from_slice::<serde_json::Value>(&raw) {
        Ok(value) => (true, json_to_py(py, &value)?),
        Err(_) => (false, py.None()),
    };
    let items = vec![
        status.into_py_any(py)?,
        PyBytes::new(py, &raw).into_any().unbind(),
        json_ok.into_py_any(py)?,
        json_value,
    ];
    PyTuple::new(py, items)
}

fn remote_http_status_error(status: u16, raw: Vec<u8>) -> PyErr {
    let body = String::from_utf8(raw).unwrap_or_default();
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "remote HTTP request failed with status {}: {}",
        status, body
    ))
}

fn read_ureq_response(response: ureq::Response) -> PyResult<Vec<u8>> {
    let mut reader = response.into_reader();
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(bytes)
}

fn parse_json_response_ids(raw: &[u8]) -> PyResult<Vec<i64>> {
    let value: serde_json::Value = serde_json::from_slice(raw)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    if value
        .get("status")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|status| status != "success")
    {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            value
                .get("error")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("remote API request failed")
                .to_string(),
        ));
    }
    let ids = value
        .get("params")
        .and_then(|params| params.get("ids"))
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(serde_json::Value::as_i64)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok(ids)
}

fn write_u32_le(out: &mut Vec<u8>, value: usize) -> PyResult<()> {
    let value = u32::try_from(value)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("binary payload count exceeds u32"))?;
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_string_binary(out: &mut Vec<u8>, value: &str) -> PyResult<()> {
    let bytes = value.as_bytes();
    write_u32_le(out, bytes.len())?;
    out.extend_from_slice(bytes);
    Ok(())
}

fn encode_json_value_binary(value: &serde_json::Value, out: &mut Vec<u8>) -> PyResult<()> {
    match value {
        serde_json::Value::Null => out.push(0),
        serde_json::Value::Bool(false) => out.push(1),
        serde_json::Value::Bool(true) => out.push(2),
        serde_json::Value::Number(number) => {
            if let Some(value) = number.as_u64() {
                out.push(4);
                out.extend_from_slice(&value.to_le_bytes());
            } else if let Some(value) = number.as_i64() {
                out.push(3);
                out.extend_from_slice(&value.to_le_bytes());
            } else if let Some(value) = number.as_f64() {
                out.push(5);
                out.extend_from_slice(&value.to_le_bytes());
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unsupported JSON number in fields payload",
                ));
            }
        }
        serde_json::Value::String(value) => {
            out.push(6);
            write_string_binary(out, value)?;
        }
        serde_json::Value::Array(items) => {
            out.push(7);
            write_u32_le(out, items.len())?;
            for item in items {
                encode_json_value_binary(item, out)?;
            }
        }
        serde_json::Value::Object(map) => {
            out.push(8);
            write_u32_le(out, map.len())?;
            for (key, item) in map {
                write_string_binary(out, key)?;
                encode_json_value_binary(item, out)?;
            }
        }
    }
    Ok(())
}

fn encode_fields_binary(
    fields: &[HashMap<String, serde_json::Value>],
    out: &mut Vec<u8>,
) -> PyResult<()> {
    out.extend_from_slice(crate::rpc::FIELDS_BINARY_MAGIC);
    write_u32_le(out, fields.len())?;
    for field in fields {
        out.push(1);
        write_u32_le(out, field.len())?;
        for (key, value) in field {
            write_string_binary(out, key)?;
            encode_json_value_binary(value, out)?;
        }
    }
    Ok(())
}

fn read_u32_le(buf: &[u8], offset: &mut usize) -> PyResult<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| py_runtime_err("remote binary offset overflow"))?;
    if end > buf.len() {
        return Err(py_runtime_err("remote binary response is truncated"));
    }
    let value = u32::from_le_bytes([
        buf[*offset],
        buf[*offset + 1],
        buf[*offset + 2],
        buf[*offset + 3],
    ]);
    *offset = end;
    Ok(value)
}

fn decode_i64_le_vec(buf: &[u8]) -> Vec<i64> {
    #[cfg(target_endian = "little")]
    {
        let mut out = Vec::<i64>::with_capacity(buf.len() / 8);
        unsafe {
            out.set_len(buf.len() / 8);
            std::ptr::copy_nonoverlapping(buf.as_ptr(), out.as_mut_ptr().cast::<u8>(), buf.len());
        }
        out
    }
    #[cfg(not(target_endian = "little"))]
    {
        buf.chunks_exact(8)
            .map(|chunk| {
                u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as i64
            })
            .collect()
    }
}

fn decode_f32_le_vec(buf: &[u8]) -> Vec<f32> {
    #[cfg(target_endian = "little")]
    {
        let mut out = Vec::<f32>::with_capacity(buf.len() / 4);
        unsafe {
            out.set_len(buf.len() / 4);
            std::ptr::copy_nonoverlapping(buf.as_ptr(), out.as_mut_ptr().cast::<u8>(), buf.len());
        }
        out
    }
    #[cfg(not(target_endian = "little"))]
    {
        buf.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }
}

fn decode_remote_search_block<'py>(
    py: Python<'py>,
    buf: &[u8],
    mut offset: usize,
) -> PyResult<(Bound<'py, PyTuple>, usize)> {
    let n = read_u32_le(buf, &mut offset)? as usize;
    let id_bytes = n
        .checked_mul(8)
        .ok_or_else(|| py_runtime_err("remote binary id byte size overflow"))?;
    let dist_bytes = n
        .checked_mul(4)
        .ok_or_else(|| py_runtime_err("remote binary distance byte size overflow"))?;
    if offset + id_bytes + dist_bytes > buf.len() {
        return Err(py_runtime_err("remote binary response is truncated"));
    }
    let ids = decode_i64_le_vec(&buf[offset..offset + id_bytes]);
    offset += id_bytes;

    let dists = decode_f32_le_vec(&buf[offset..offset + dist_bytes]);
    offset += dist_bytes;

    let fields_len = read_u32_le(buf, &mut offset)? as usize;
    if offset + fields_len > buf.len() {
        return Err(py_runtime_err("remote binary fields payload is truncated"));
    }
    let fields: Vec<HashMap<String, serde_json::Value>> = if fields_len == 0 {
        Vec::new()
    } else {
        serde_json::from_slice(&buf[offset..offset + fields_len])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };
    offset += fields_len;

    let items = vec![
        ids.into_pyarray(py).into_any().unbind(),
        dists.into_pyarray(py).into_any().unbind(),
        fields_to_pylist(py, &fields)?.into_any().unbind(),
    ];
    Ok((PyTuple::new(py, items)?, offset))
}

fn decode_remote_batch_search<'py>(
    py: Python<'py>,
    buf: &[u8],
    expected_queries: usize,
) -> PyResult<Bound<'py, PyList>> {
    let mut offset = 0usize;
    let count = read_u32_le(buf, &mut offset)? as usize;
    if count != expected_queries {
        return Err(py_runtime_err(format!(
            "batch_search_binary returned {} queries, expected {}",
            count, expected_queries
        )));
    }
    let list = PyList::empty(py);
    for _ in 0..count {
        let (tuple, next_offset) = decode_remote_search_block(py, buf, offset)?;
        offset = next_offset;
        list.append(tuple)?;
    }
    if offset != buf.len() {
        return Err(py_runtime_err("trailing bytes in remote batch search response"));
    }
    Ok(list)
}

/// Convert a Vec of field HashMaps to a Python list of dicts.
fn fields_to_pylist<'py>(
    py: Python<'py>,
    fields: &[HashMap<String, serde_json::Value>],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for field_map in fields {
        let dict = PyDict::new(py);
        for (k, v) in field_map {
            dict.set_item(k, json_to_py(py, v)?)?;
        }
        list.append(dict)?;
    }
    Ok(list)
}

fn vector_fields_to_pylist<'py>(
    py: Python<'py>,
    fields: &[VectorFieldConfig],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for field in fields {
        let dict = PyDict::new(py);
        dict.set_item("name", &field.name)?;
        dict.set_item("dimension", field.dimension)?;
        dict.set_item("metric", &field.metric)?;
        dict.set_item("index_mode", &field.index_mode)?;
        dict.set_item("dtypes", &field.dtypes)?;
        list.append(dict)?;
    }
    Ok(list)
}

fn py_to_external_id(obj: &Bound<'_, PyAny>) -> PyResult<ExternalId> {
    if obj.extract::<bool>().is_ok() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "bool is not a valid LynseDB ID",
        ));
    }
    if let Ok(value) = obj.extract::<u64>() {
        return Ok(ExternalId::Int(value));
    }
    if let Ok(value) = obj.extract::<String>() {
        if value.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "string IDs cannot be empty",
            ));
        }
        return Ok(ExternalId::String(value));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "IDs must be strings or non-negative integers",
    ))
}

fn pylist_to_external_ids(ids: &Bound<'_, PyList>) -> PyResult<Vec<ExternalId>> {
    let mut out = Vec::with_capacity(ids.len());
    for item in ids.iter() {
        out.push(py_to_external_id(&item)?);
    }
    if out.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ids cannot be empty",
        ));
    }
    Ok(out)
}

fn external_ids_to_pylist<'py>(
    py: Python<'py>,
    ids: &[ExternalId],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for id in ids {
        match id {
            ExternalId::Int(value) => list.append(*value)?,
            ExternalId::String(value) => list.append(value)?,
        }
    }
    Ok(list)
}

fn parse_py_vector_dtype(dtypes: Option<&str>) -> PyResult<VectorDtype> {
    dtypes
        .map(VectorDtype::parse)
        .unwrap_or_else(|| Ok(VectorDtype::F32))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
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
fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match v {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => (*b).into_py_any(py),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py_any(py)
            } else if let Some(f) = n.as_f64() {
                f.into_py_any(py)
            } else {
                n.to_string().into_py_any(py)
            }
        }
        serde_json::Value::String(s) => s.into_py_any(py),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(obj) => {
            let dict = PyDict::new(py);
            for (k, v) in obj {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}
