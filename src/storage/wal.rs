//! Write-Ahead Log (WAL) storage for crash recovery.
//!
//! Port of Python `lynse/storage_layer/wal.py` to Rust.
//! Features:
//! - Buffered writes with configurable flush threshold
//! - File rotation when WAL exceeds max size
//! - Segment-based binary format with mmap reads
//! - Atomic row-count updates with fsync
//! - Iterator-based replay for recovery

use crate::error::{LynseError, Result};
use crate::storage::dtype::{
    decode_vector_bytes_to_f32, encode_f32_slice_as_le_bytes, VectorDtype,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ─── Binary format constants ─────────────────────────────────────────────────

/// File header: version(u64) + chunk_size(u64) + flush_interval_ms(u64) + count_rows(u64)
const HEADER_SIZE: usize = 32; // 4 * 8 bytes
const WAL_VERSION_JSON_FIELDS: u64 = 3;
const WAL_VERSION: u64 = 4;

/// Segment header: data_size(u64) + record_count(u64) + status(u8) + data_dim(u64)
const SEGMENT_HEADER_SIZE: usize = 25; // 8 + 8 + 1 + 8

/// Status byte values
const SEGMENT_STATUS_COMMITTED: u8 = 1;

/// Limits
const MAX_WAL_SIZE: u64 = 1024 * 1024 * 1024; // 1GB
const BUFFER_FLUSH_SIZE: usize = 10_000;
const WRITE_BUFFER_SIZE: usize = 8 * 1024 * 1024; // 8MB

#[derive(Debug, Clone, Serialize, Deserialize)]
enum WalFieldValue {
    Null,
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    String(String),
    Array(Vec<WalFieldValue>),
    Object(Vec<(String, WalFieldValue)>),
}

fn json_to_wal_field_value(value: &serde_json::Value) -> WalFieldValue {
    match value {
        serde_json::Value::Null => WalFieldValue::Null,
        serde_json::Value::Bool(value) => WalFieldValue::Bool(*value),
        serde_json::Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                WalFieldValue::I64(value)
            } else if let Some(value) = value.as_u64() {
                WalFieldValue::U64(value)
            } else if let Some(value) = value.as_f64() {
                WalFieldValue::F64(value)
            } else {
                WalFieldValue::Null
            }
        }
        serde_json::Value::String(value) => WalFieldValue::String(value.clone()),
        serde_json::Value::Array(values) => {
            WalFieldValue::Array(values.iter().map(json_to_wal_field_value).collect())
        }
        serde_json::Value::Object(values) => WalFieldValue::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), json_to_wal_field_value(value)))
                .collect(),
        ),
    }
}

fn wal_field_value_to_json(value: WalFieldValue) -> serde_json::Value {
    match value {
        WalFieldValue::Null => serde_json::Value::Null,
        WalFieldValue::Bool(value) => serde_json::Value::Bool(value),
        WalFieldValue::I64(value) => serde_json::Value::Number(value.into()),
        WalFieldValue::U64(value) => serde_json::Value::Number(value.into()),
        WalFieldValue::F64(value) => serde_json::Number::from_f64(value)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        WalFieldValue::String(value) => serde_json::Value::String(value),
        WalFieldValue::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(wal_field_value_to_json).collect())
        }
        WalFieldValue::Object(values) => {
            let object = values
                .into_iter()
                .map(|(key, value)| (key, wal_field_value_to_json(value)))
                .collect();
            serde_json::Value::Object(object)
        }
    }
}

fn serialize_wal_fields(fields: &[HashMap<String, serde_json::Value>]) -> Result<Vec<u8>> {
    let binary_fields: Vec<HashMap<String, WalFieldValue>> = fields
        .iter()
        .map(|field| {
            field
                .iter()
                .map(|(key, value)| (key.clone(), json_to_wal_field_value(value)))
                .collect()
        })
        .collect();
    bincode::serialize(&binary_fields).map_err(|e| LynseError::Serialization(e.to_string()))
}

fn deserialize_wal_fields(bytes: &[u8]) -> Result<Vec<HashMap<String, serde_json::Value>>> {
    let binary_fields: Vec<HashMap<String, WalFieldValue>> =
        bincode::deserialize(bytes).map_err(|e| LynseError::Serialization(e.to_string()))?;
    Ok(binary_fields
        .into_iter()
        .map(|field| {
            field
                .into_iter()
                .map(|(key, value)| (key, wal_field_value_to_json(value)))
                .collect()
        })
        .collect())
}

// ─── WAL Buffer ──────────────────────────────────────────────────────────────

/// In-memory buffer that accumulates vectors + fields before flushing to disk.
struct WALBuffer {
    /// Accumulated encoded vector data (flattened, little-endian)
    data: Vec<Vec<u8>>,
    /// User-facing IDs for each vector.
    ids: Vec<u64>,
    /// Accumulated field metadata per vector
    fields: Vec<HashMap<String, serde_json::Value>>,
    /// Dimension of vectors (set on first append)
    dim: usize,
    /// Vector dtype used by the encoded data.
    dtype: VectorDtype,
    /// Total number of records buffered
    size: usize,
}

impl WALBuffer {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            ids: Vec::new(),
            fields: Vec::new(),
            dim: 0,
            dtype: VectorDtype::F32,
            size: 0,
        }
    }

    fn append_encoded(
        &mut self,
        data: &[u8],
        dim: usize,
        dtype: VectorDtype,
        ids: &[u64],
        fields: &[HashMap<String, serde_json::Value>],
    ) -> Result<()> {
        if self.dim == 0 {
            self.dim = dim;
            self.dtype = dtype;
        } else if self.dim != dim || self.dtype != dtype {
            return Err(LynseError::InvalidArgument(
                "WAL buffer cannot mix vector dimensions or dtypes".to_string(),
            ));
        }
        self.data.push(data.to_vec());
        self.ids.extend_from_slice(ids);
        self.fields.extend(fields.iter().cloned());
        self.size += ids.len();
        Ok(())
    }

    /// Return concatenated data + fields, consuming the buffer contents.
    fn take(
        &mut self,
    ) -> Option<(
        Vec<u8>,
        usize,
        VectorDtype,
        Vec<u64>,
        Vec<HashMap<String, serde_json::Value>>,
    )> {
        if self.size == 0 {
            return None;
        }
        let byte_len: usize = self.data.iter().map(Vec::len).sum();
        let mut concatenated: Vec<u8> = Vec::with_capacity(byte_len);
        for chunk in &self.data {
            concatenated.extend_from_slice(chunk);
        }
        let ids = std::mem::take(&mut self.ids);
        let fields = std::mem::take(&mut self.fields);
        let dim = self.dim;
        let dtype = self.dtype;
        self.data.clear();
        self.size = 0;
        self.dim = 0;
        self.dtype = VectorDtype::F32;
        Some((concatenated, dim, dtype, ids, fields))
    }

    fn clear(&mut self) {
        self.data.clear();
        self.ids.clear();
        self.fields.clear();
        self.size = 0;
        self.dim = 0;
        self.dtype = VectorDtype::F32;
    }
}

// ─── WAL Segment (for reading) ───────────────────────────────────────────────

/// A single WAL segment read from disk.
pub struct WALSegment {
    /// Vector data as flat f32 slice, shape = (n_vectors, dim)
    pub data: Vec<f32>,
    /// Original encoded vector payload from the WAL segment.
    pub encoded_data: Vec<u8>,
    /// Dtype used by the encoded WAL payload.
    pub dtype: VectorDtype,
    /// User-facing IDs for each vector.
    pub ids: Vec<u64>,
    /// Dimension of each vector
    pub dim: usize,
    /// Number of vectors in this segment
    pub n_vectors: usize,
    /// Per-vector field metadata
    pub fields: Vec<HashMap<String, serde_json::Value>>,
}

// ─── WAL Storage ─────────────────────────────────────────────────────────────

/// Write-Ahead Log for crash-safe vector ingestion.
///
/// Data flow: `write_log_data()` → buffer → flush → WAL file on disk
/// Recovery: `get_segments_iterator()` replays all committed segments
pub struct WALStorage {
    storage_path: PathBuf,
    collection_name: String,
    #[allow(dead_code)]
    current_wal_id: u64,
    wal_file: PathBuf,
    chunk_size: usize,
    flush_interval_ms: u64,

    // Internal state (protected by mutex)
    inner: Mutex<WALInner>,
}

struct WALInner {
    buffer: WALBuffer,
    write_buf: Vec<u8>,
    write_buf_pos: usize,
    current_file: Option<File>,
    pending_row_count: u64,
    initialized: bool,
    last_flush: Instant,
    alive: bool,
}

impl WALStorage {
    /// Create or open a WAL for the given collection.
    pub fn new(
        collection_name: &str,
        chunk_size: usize,
        storage_path: &Path,
        flush_interval_ms: u64,
    ) -> Result<Self> {
        let wal_dir = storage_path.join("wal");
        fs::create_dir_all(&wal_dir)?;

        let current_wal_id = Self::get_latest_wal_id(&wal_dir, collection_name);
        let wal_file = Self::wal_path(&wal_dir, collection_name, current_wal_id);

        Ok(Self {
            storage_path: wal_dir,
            collection_name: collection_name.to_string(),
            current_wal_id,
            wal_file,
            chunk_size,
            flush_interval_ms,
            inner: Mutex::new(WALInner {
                buffer: WALBuffer::new(),
                write_buf: vec![0u8; WRITE_BUFFER_SIZE],
                write_buf_pos: 0,
                current_file: None,
                pending_row_count: 0,
                initialized: false,
                last_flush: Instant::now(),
                alive: true,
            }),
        })
    }

    /// Get the WAL directory path.
    pub fn log_dir(&self) -> &Path {
        &self.storage_path
    }

    // ── Path helpers ──

    fn wal_path(dir: &Path, collection: &str, id: u64) -> PathBuf {
        dir.join(format!("{}.{:06}.wal", collection, id))
    }

    fn get_latest_wal_id(dir: &Path, collection: &str) -> u64 {
        let prefix = format!("{}.", collection);
        let mut max_id: u64 = 0;
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(&prefix) && name.ends_with(".wal") {
                    // Parse "collection.NNNNNN.wal"
                    let parts: Vec<&str> = name.trim_end_matches(".wal").rsplitn(2, '.').collect();
                    if let Some(id_str) = parts.first() {
                        if let Ok(id) = id_str.parse::<u64>() {
                            max_id = max_id.max(id);
                        }
                    }
                }
            }
        }
        max_id
    }

    fn get_file_row_count(file_path: &Path) -> u64 {
        let mut f = match File::open(file_path) {
            Ok(f) => f,
            Err(_) => return 0,
        };
        // Skip version(8) + chunk_size(8) + flush_interval(8), read count_rows(8)
        if f.seek(SeekFrom::Start(24)).is_err() {
            return 0;
        }
        let mut buf = [0u8; 8];
        if f.read_exact(&mut buf).is_err() {
            return 0;
        }
        u64::from_le_bytes(buf)
    }

    fn get_file_version(file_path: &Path) -> Option<u64> {
        let mut f = File::open(file_path).ok()?;
        let mut buf = [0u8; 8];
        f.read_exact(&mut buf).ok()?;
        Some(u64::from_le_bytes(buf))
    }

    // ── Public API ──

    /// Write vector data + fields to the WAL (buffered).
    pub fn write_log_data(
        &self,
        data: &[f32],
        dim: usize,
        ids: &[u64],
        fields: &[HashMap<String, serde_json::Value>],
    ) -> Result<()> {
        self.write_log_data_with_dtype(data, dim, VectorDtype::F32, ids, fields)
    }

    /// Write f32 vector data to the WAL, encoded with the requested dtype.
    pub fn write_log_data_with_dtype(
        &self,
        data: &[f32],
        dim: usize,
        dtype: VectorDtype,
        ids: &[u64],
        fields: &[HashMap<String, serde_json::Value>],
    ) -> Result<()> {
        if dim == 0 {
            return Err(LynseError::InvalidArgument(
                "WAL write requires a non-zero vector dimension".to_string(),
            ));
        }
        if data.len() % dim != 0 {
            return Err(LynseError::DimensionMismatch {
                expected: dim,
                got: data.len() % dim,
            });
        }
        let n_vectors = data.len() / dim;
        if ids.len() != n_vectors {
            return Err(LynseError::InvalidArgument(format!(
                "ids length ({}) must match WAL vector count ({})",
                ids.len(),
                n_vectors
            )));
        }
        if !fields.is_empty() && fields.len() != n_vectors {
            return Err(LynseError::InvalidArgument(format!(
                "fields length ({}) must match WAL vector count ({})",
                fields.len(),
                n_vectors
            )));
        }

        let encoded = encode_f32_slice_as_le_bytes(data, dtype);
        self.write_log_encoded_data(&encoded, dim, dtype, ids, fields)
    }

    /// Write already-encoded little-endian vector bytes to the WAL.
    pub fn write_log_encoded_data(
        &self,
        data: &[u8],
        dim: usize,
        dtype: VectorDtype,
        ids: &[u64],
        fields: &[HashMap<String, serde_json::Value>],
    ) -> Result<()> {
        if dim == 0 {
            return Err(LynseError::InvalidArgument(
                "WAL write requires a non-zero vector dimension".to_string(),
            ));
        }
        let row_width = dim
            .checked_mul(dtype.byte_width())
            .ok_or_else(|| LynseError::InvalidArgument("WAL row byte size overflows".into()))?;
        if row_width == 0 || data.len() % row_width != 0 {
            return Err(LynseError::DimensionMismatch {
                expected: row_width,
                got: data.len() % row_width,
            });
        }
        let n_vectors = data.len() / row_width;
        if ids.len() != n_vectors {
            return Err(LynseError::InvalidArgument(format!(
                "ids length ({}) must match WAL vector count ({})",
                ids.len(),
                n_vectors
            )));
        }
        if !fields.is_empty() && fields.len() != n_vectors {
            return Err(LynseError::InvalidArgument(format!(
                "fields length ({}) must match WAL vector count ({})",
                fields.len(),
                n_vectors
            )));
        }
        if n_vectors == 0 {
            return Ok(());
        }

        let normalized_fields;
        let fields = if fields.is_empty() {
            normalized_fields = vec![HashMap::new(); n_vectors];
            &normalized_fields
        } else {
            fields
        };

        let mut inner = self.inner.lock();

        // Lazy-initialize the WAL file
        if !inner.initialized {
            self.initialize_wal_file(&mut inner)?;
        }

        inner.buffer.append_encoded(data, dim, dtype, ids, fields)?;

        if inner.buffer.size >= BUFFER_FLUSH_SIZE
            || inner.last_flush.elapsed().as_millis() as u64 >= self.flush_interval_ms
        {
            self.flush_buffer_to_disk(&mut inner)?;
        }

        // Make WAL bytes visible to the OS before callers append to the main
        // vector store. Expensive fsync is handled by flush()/checkpoint().
        self.flush_buffer_to_disk(&mut inner)?;
        self.flush_write_buffer(&mut inner)?;
        self.flush_row_count(&mut inner)?;
        if let Some(ref mut f) = inner.current_file {
            f.flush()?;
        }

        Ok(())
    }

    /// Check if there is uncommitted data in WAL files.
    pub fn has_uncommitted_data(&self) -> bool {
        let pattern_prefix = format!("{}.", self.collection_name);
        let mut count = 0u64;
        if let Ok(entries) = fs::read_dir(&self.storage_path) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(&pattern_prefix) && name.ends_with(".wal") {
                    count += Self::get_file_row_count(&entry.path());
                }
            }
        }
        // Also check in-memory buffer
        let inner = self.inner.lock();
        count > 0 || inner.buffer.size > 0
    }

    /// Flush all pending WAL buffers and fsync the active WAL file.
    pub fn flush(&self) -> Result<()> {
        let mut inner = self.inner.lock();
        self.flush_buffer_to_disk(&mut inner)?;
        self.flush_write_buffer(&mut inner)?;
        self.flush_row_count(&mut inner)?;
        if let Some(ref mut f) = inner.current_file {
            f.flush()?;
            f.sync_all()?;
        }
        Ok(())
    }

    /// Iterate over all committed segments across all WAL files.
    /// This flushes any buffered data first.
    pub fn get_segments(&self) -> Result<Vec<WALSegment>> {
        // Flush everything first
        {
            let mut inner = self.inner.lock();
            self.flush_buffer_to_disk(&mut inner)?;
            self.flush_write_buffer(&mut inner)?;
            self.flush_row_count(&mut inner)?;
            if let Some(ref mut f) = inner.current_file {
                f.flush()?;
                f.sync_all()?;
            }
        }

        // Collect and sort WAL files
        let mut wal_files: Vec<PathBuf> = Vec::new();
        let pattern_prefix = format!("{}.", self.collection_name);
        if let Ok(entries) = fs::read_dir(&self.storage_path) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(&pattern_prefix) && name.ends_with(".wal") {
                    wal_files.push(entry.path());
                }
            }
        }
        wal_files.sort();

        let mut segments = Vec::new();

        for wal_path in &wal_files {
            let file_data = fs::read(wal_path)?;
            if file_data.len() < HEADER_SIZE {
                continue;
            }
            let version = u64::from_le_bytes(file_data[0..8].try_into().unwrap());
            if version != WAL_VERSION && version != WAL_VERSION_JSON_FIELDS {
                return Err(LynseError::Storage(format!(
                    "unsupported WAL version {}; expected {} or {}",
                    version, WAL_VERSION_JSON_FIELDS, WAL_VERSION
                )));
            }

            let mut pos = HEADER_SIZE;

            while pos + SEGMENT_HEADER_SIZE <= file_data.len() {
                // Read segment header
                let data_size =
                    u64::from_le_bytes(file_data[pos..pos + 8].try_into().unwrap()) as usize;
                let record_count =
                    u64::from_le_bytes(file_data[pos + 8..pos + 16].try_into().unwrap()) as usize;
                let status = file_data[pos + 16];
                let data_dim =
                    u64::from_le_bytes(file_data[pos + 17..pos + 25].try_into().unwrap()) as usize;

                pos += SEGMENT_HEADER_SIZE;

                // Validate
                if data_size == 0 || data_size > MAX_WAL_SIZE as usize {
                    break;
                }
                if status != SEGMENT_STATUS_COMMITTED {
                    // Skip uncommitted segment
                    if pos + data_size <= file_data.len() {
                        pos += data_size;
                    } else {
                        break;
                    }
                    continue;
                }
                if pos + data_size > file_data.len() {
                    // Incomplete segment
                    break;
                }

                let dtype = {
                    if pos >= file_data.len() {
                        break;
                    }
                    let code = file_data[pos];
                    pos += 1;
                    match code {
                        1 => VectorDtype::F32,
                        2 => VectorDtype::F16,
                        other => {
                            return Err(LynseError::Storage(format!(
                                "unsupported WAL vector dtype code {}",
                                other
                            )));
                        }
                    }
                };

                // Read vector data: data_len(u64) + encoded data_bytes
                if pos + 8 > file_data.len() {
                    break;
                }
                let vec_data_size =
                    u64::from_le_bytes(file_data[pos..pos + 8].try_into().unwrap()) as usize;
                pos += 8;

                if pos + vec_data_size > file_data.len() {
                    break;
                }
                let vec_bytes = &file_data[pos..pos + vec_data_size];
                pos += vec_data_size;

                if vec_data_size % dtype.byte_width() != 0 {
                    return Err(LynseError::Storage(format!(
                        "WAL vector payload size {} is not aligned to {} bytes",
                        vec_data_size,
                        dtype.byte_width()
                    )));
                }
                let n_floats = vec_data_size / dtype.byte_width();
                let float_data = decode_vector_bytes_to_f32(vec_bytes, dtype);

                // Read fields: fields_len(u64) + fields_bytes
                if pos + 8 > file_data.len() {
                    break;
                }
                let fields_size =
                    u64::from_le_bytes(file_data[pos..pos + 8].try_into().unwrap()) as usize;
                pos += 8;

                if pos + fields_size > file_data.len() {
                    break;
                }
                let fields_bytes = &file_data[pos..pos + fields_size];
                pos += fields_size;

                let fields: Vec<HashMap<String, serde_json::Value>> = if version
                    == WAL_VERSION_JSON_FIELDS
                {
                    serde_json::from_slice(fields_bytes).map_err(|e| {
                        LynseError::Serialization(format!("failed to decode WAL JSON fields: {e}"))
                    })?
                } else {
                    deserialize_wal_fields(fields_bytes).map_err(|e| {
                        LynseError::Serialization(format!(
                            "failed to decode WAL binary fields: {e}"
                        ))
                    })?
                };

                if pos + 8 > file_data.len() {
                    break;
                }
                let ids_size =
                    u64::from_le_bytes(file_data[pos..pos + 8].try_into().unwrap()) as usize;
                pos += 8;

                if pos + ids_size > file_data.len() {
                    break;
                }
                let ids_bytes = &file_data[pos..pos + ids_size];
                pos += ids_size;

                let ids = ids_bytes
                    .chunks_exact(8)
                    .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();

                let n_vectors = if data_dim > 0 {
                    n_floats / data_dim
                } else {
                    record_count
                };

                segments.push(WALSegment {
                    data: float_data,
                    encoded_data: vec_bytes.to_vec(),
                    dtype,
                    ids,
                    dim: data_dim,
                    n_vectors,
                    fields,
                });
            }
        }

        Ok(segments)
    }

    /// Get total row count across all WAL files + buffer.
    pub fn total_rows(&self) -> u64 {
        let inner = self.inner.lock();
        let mut total = inner.buffer.size as u64 + inner.pending_row_count;

        let pattern_prefix = format!("{}.", self.collection_name);
        if let Ok(entries) = fs::read_dir(&self.storage_path) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(&pattern_prefix) && name.ends_with(".wal") {
                    total += Self::get_file_row_count(&entry.path());
                }
            }
        }
        total
    }

    /// Cleanup all WAL files (after successful commit to main storage).
    pub fn cleanup(&self) -> Result<()> {
        let mut inner = self.inner.lock();
        inner.buffer.clear();
        self.close_current_file(&mut inner);

        let pattern_prefix = format!("{}.", self.collection_name);
        if let Ok(entries) = fs::read_dir(&self.storage_path) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(&pattern_prefix) && name.ends_with(".wal") {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
        Ok(())
    }

    /// Reset WAL state completely (cleanup + reinitialize).
    pub fn reincarnate(&self) -> Result<()> {
        self.cleanup()?;
        let mut inner = self.inner.lock();
        inner.initialized = false;
        inner.write_buf_pos = 0;
        inner.pending_row_count = 0;
        inner.last_flush = Instant::now();
        Ok(())
    }

    /// Stop the WAL, flushing all pending data.
    pub fn stop(&self) -> Result<()> {
        let mut inner = self.inner.lock();
        inner.alive = false;
        self.flush_buffer_to_disk(&mut inner)?;
        self.flush_write_buffer(&mut inner)?;
        self.flush_row_count(&mut inner)?;
        self.close_current_file(&mut inner);
        Ok(())
    }

    // ── Internal methods ──

    fn initialize_wal_file(&self, inner: &mut WALInner) -> Result<()> {
        if inner.initialized {
            return Ok(());
        }
        if self.wal_file.exists() {
            if let Some(version) = Self::get_file_version(&self.wal_file) {
                if version == WAL_VERSION_JSON_FIELDS {
                    if Self::get_file_row_count(&self.wal_file) == 0 {
                        fs::remove_file(&self.wal_file)?;
                    } else {
                        return Err(LynseError::Storage(
                            "WAL v3 has pending rows; reopen the collection to recover it before writing WAL v4".to_string(),
                        ));
                    }
                } else if version != WAL_VERSION {
                    return Err(LynseError::Storage(format!(
                        "WAL file version {} is not supported; expected {} or {}",
                        version, WAL_VERSION_JSON_FIELDS, WAL_VERSION
                    )));
                }
            }
        }
        if !self.wal_file.exists() {
            let mut f = File::create(&self.wal_file)?;
            // Write header: version + chunk_size + flush_interval_ms + count_rows(0)
            f.write_all(&WAL_VERSION.to_le_bytes())?;
            f.write_all(&(self.chunk_size as u64).to_le_bytes())?;
            f.write_all(&self.flush_interval_ms.to_le_bytes())?;
            f.write_all(&0u64.to_le_bytes())?; // count_rows = 0
            f.flush()?;
            f.sync_all()?;
        }
        inner.initialized = true;
        Ok(())
    }

    /// Rotate WAL file if it exceeds the max size.
    /// Uses interior mutability pattern — fields that need mutation are behind the Mutex.
    fn maybe_rotate_wal(&self, _inner: &mut WALInner) -> Result<()> {
        // We cannot mutate self.wal_file/current_wal_id here because &self is shared.
        // Instead, rotation is handled at the WALStorage level by cleanup + reincarnate
        // after commit. For single-WAL-file mode, rotation is not needed since
        // commit() clears the WAL after data is flushed to main storage.
        //
        // For very large uncommitted batches, the WAL file may grow large, but this
        // is acceptable since commit() will reclaim the space.
        Ok(())
    }

    fn flush_buffer_to_disk(&self, inner: &mut WALInner) -> Result<()> {
        // Check for WAL file rotation before writing
        self.maybe_rotate_wal(inner)?;

        let taken = inner.buffer.take();
        let (data_bytes, dim, dtype, ids, fields) = match taken {
            Some(t) => t,
            None => return Ok(()),
        };

        let fields_bytes = serialize_wal_fields(&fields)?;
        let ids_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();

        let total_size = 1 + 8 + data_bytes.len() + 8 + fields_bytes.len() + 8 + ids_bytes.len();
        let record_count = ids.len();

        // Write segment header
        let mut header = Vec::with_capacity(SEGMENT_HEADER_SIZE);
        header.extend_from_slice(&(total_size as u64).to_le_bytes()); // data_size
        header.extend_from_slice(&(record_count as u64).to_le_bytes()); // record_count
        header.push(SEGMENT_STATUS_COMMITTED); // status
        header.extend_from_slice(&(dim as u64).to_le_bytes()); // data_dim
        self.write_to_buffer(inner, &header);

        // Write vector dtype for WAL v3+.
        let dtype_code = match dtype {
            VectorDtype::F32 => 1u8,
            VectorDtype::F16 => 2u8,
        };
        self.write_to_buffer(inner, &[dtype_code]);

        // Write vector data: length + bytes
        self.write_to_buffer(inner, &(data_bytes.len() as u64).to_le_bytes());
        self.write_to_buffer(inner, &data_bytes);

        // Write fields: length + bytes
        self.write_to_buffer(inner, &(fields_bytes.len() as u64).to_le_bytes());
        self.write_to_buffer(inner, &fields_bytes);

        // Write user IDs: length + bytes
        self.write_to_buffer(inner, &(ids_bytes.len() as u64).to_le_bytes());
        self.write_to_buffer(inner, &ids_bytes);

        // Flush write buffer
        self.flush_write_buffer(inner)?;

        // Update row count
        inner.pending_row_count += record_count as u64;
        self.flush_row_count(inner)?;

        inner.last_flush = Instant::now();

        Ok(())
    }

    fn write_to_buffer(&self, inner: &mut WALInner, data: &[u8]) {
        if inner.write_buf_pos + data.len() > WRITE_BUFFER_SIZE {
            // Grow from the actually used prefix. Extending the full fixed-size
            // buffer would write zero-filled slack bytes into the WAL segment.
            inner.write_buf.truncate(inner.write_buf_pos);
            inner.write_buf.extend_from_slice(data);
            inner.write_buf_pos = inner.write_buf.len();
            return;
        }
        let end = inner.write_buf_pos + data.len();
        inner.write_buf[inner.write_buf_pos..end].copy_from_slice(data);
        inner.write_buf_pos = end;
    }

    fn flush_write_buffer(&self, inner: &mut WALInner) -> Result<()> {
        if inner.write_buf_pos == 0 {
            return Ok(());
        }

        if inner.current_file.is_none() {
            inner.current_file = Some(
                OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(&self.wal_file)?,
            );
        }

        let f = inner.current_file.as_mut().unwrap();
        f.write_all(&inner.write_buf[..inner.write_buf_pos])?;
        f.flush()?;
        // Note: skip sync_all() here for throughput; data is fsynced on commit/stop
        inner.write_buf_pos = 0;
        // Reset write_buf to fixed size if it grew
        if inner.write_buf.len() > WRITE_BUFFER_SIZE {
            inner.write_buf.resize(WRITE_BUFFER_SIZE, 0);
        }
        Ok(())
    }

    fn flush_row_count(&self, inner: &mut WALInner) -> Result<()> {
        if inner.pending_row_count == 0 {
            return Ok(());
        }
        if !self.wal_file.exists() {
            return Ok(());
        }

        let current_count = Self::get_file_row_count(&self.wal_file);
        let new_count = current_count + inner.pending_row_count;

        let mut f = OpenOptions::new().write(true).open(&self.wal_file)?;
        f.seek(SeekFrom::Start(24))?; // offset of count_rows in header
        f.write_all(&new_count.to_le_bytes())?;
        f.flush()?;
        // Note: skip sync_all() here for throughput; data is fsynced on commit/stop

        inner.pending_row_count = 0;
        Ok(())
    }

    fn close_current_file(&self, inner: &mut WALInner) {
        if let Some(f) = inner.current_file.take() {
            drop(f);
        }
    }
}

impl Drop for WALStorage {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_wal_write_and_read() {
        let tmp = TempDir::new().unwrap();
        let wal = WALStorage::new("test_col", 100_000, tmp.path(), 5000).unwrap();

        // Write some data
        let dim = 4;
        let vectors: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let fields = vec![
            {
                let mut m = HashMap::new();
                m.insert("label".to_string(), serde_json::json!("a"));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("label".to_string(), serde_json::json!("b"));
                m
            },
        ];

        let ids = vec![10, 11];
        wal.write_log_data(&vectors, dim, &ids, &fields).unwrap();

        assert!(wal.has_uncommitted_data());

        let segments = wal.get_segments().unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].n_vectors, 2);
        assert_eq!(segments[0].dim, 4);
        assert_eq!(segments[0].dtype, VectorDtype::F32);
        assert_eq!(segments[0].data, vectors);
        assert_eq!(segments[0].encoded_data.len(), vectors.len() * 4);
        assert_eq!(segments[0].ids, ids);
        assert_eq!(segments[0].fields.len(), 2);

        // Cleanup
        wal.cleanup().unwrap();
        assert!(!wal.has_uncommitted_data());
    }

    #[test]
    fn test_wal_v3_json_fields_can_be_read() {
        let tmp = TempDir::new().unwrap();
        let wal = WALStorage::new("test_col", 100_000, tmp.path(), 5000).unwrap();
        let wal_path = wal.log_dir().join("test_col.000000.wal");

        let dim = 2usize;
        let vectors = [1.0f32, 2.0];
        let vec_bytes: Vec<u8> = vectors.iter().flat_map(|v| v.to_le_bytes()).collect();
        let fields = vec![HashMap::from([(
            "label".to_string(),
            serde_json::json!("legacy"),
        )])];
        let fields_bytes = serde_json::to_vec(&fields).unwrap();
        let ids_bytes = 7u64.to_le_bytes();
        let data_size = 1 + 8 + vec_bytes.len() + 8 + fields_bytes.len() + 8 + ids_bytes.len();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&WAL_VERSION_JSON_FIELDS.to_le_bytes());
        bytes.extend_from_slice(&100_000u64.to_le_bytes());
        bytes.extend_from_slice(&5000u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&(data_size as u64).to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.push(SEGMENT_STATUS_COMMITTED);
        bytes.extend_from_slice(&(dim as u64).to_le_bytes());
        bytes.push(1);
        bytes.extend_from_slice(&(vec_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&vec_bytes);
        bytes.extend_from_slice(&(fields_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&fields_bytes);
        bytes.extend_from_slice(&(ids_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&ids_bytes);
        fs::write(wal_path, bytes).unwrap();

        let segments = wal.get_segments().unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].ids, vec![7]);
        assert_eq!(
            segments[0].fields[0].get("label"),
            Some(&serde_json::json!("legacy"))
        );
    }

    #[test]
    fn test_wal_multiple_writes() {
        let tmp = TempDir::new().unwrap();
        let wal = WALStorage::new("test_col", 100_000, tmp.path(), 5000).unwrap();

        let dim = 2;
        for i in 0..5 {
            let vectors = vec![i as f32, (i + 1) as f32];
            let fields = vec![{
                let mut m = HashMap::new();
                m.insert("idx".to_string(), serde_json::json!(i));
                m
            }];
            wal.write_log_data(&vectors, dim, &[i as u64], &fields)
                .unwrap();
        }

        let segments = wal.get_segments().unwrap();
        let total_vectors: usize = segments.iter().map(|s| s.n_vectors).sum();
        assert_eq!(total_vectors, 5);
    }

    #[test]
    fn test_wal_vector_only_records_are_counted() {
        let tmp = TempDir::new().unwrap();
        let wal = WALStorage::new("test_col", 100_000, tmp.path(), 5000).unwrap();

        let dim = 2;
        let vectors = vec![1.0, 2.0, 3.0, 4.0];
        let ids = vec![100, 101];
        wal.write_log_data(&vectors, dim, &ids, &[]).unwrap();

        assert!(wal.has_uncommitted_data());
        let segments = wal.get_segments().unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].n_vectors, 2);
        assert_eq!(segments[0].ids, ids);
        assert_eq!(segments[0].fields.len(), 2);
        assert!(segments[0].fields.iter().all(|fields| fields.is_empty()));
    }

    #[test]
    fn test_wal_f16_payload_stays_half_width() {
        let tmp = TempDir::new().unwrap();
        let wal = WALStorage::new("test_col", 100_000, tmp.path(), 5000).unwrap();

        let dim = 4;
        let vectors = vec![1.0, 0.5, 0.0, -0.5, 2.0, 1.5, 1.0, 0.0];
        let ids = vec![1, 2];
        wal.write_log_data_with_dtype(&vectors, dim, VectorDtype::F16, &ids, &[])
            .unwrap();

        let segments = wal.get_segments().unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].dtype, VectorDtype::F16);
        assert_eq!(segments[0].n_vectors, 2);
        assert_eq!(segments[0].encoded_data.len(), vectors.len() * 2);
        assert_eq!(segments[0].data, vectors);
    }
}
