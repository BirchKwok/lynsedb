//! Internal LynseDB cluster RPC.
//!
//! This is intentionally small and private to LynseDB nodes. The external API
//! remains HTTP; the coordinator opportunistically uses this length-prefixed TCP
//! protocol for hot shard calls and falls back to HTTP when unavailable.

use crate::engine::DatabaseManager;
use crate::error::{LynseError, Result};
use crate::storage::dtype::VectorDtype;
use half::f16;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

pub(crate) const OP_PING: u8 = 1;
pub(crate) const OP_SEARCH: u8 = 2;
pub(crate) const OP_BATCH_SEARCH: u8 = 3;
pub(crate) const OP_BULK_ADD_BINARY_IDS: u8 = 4;
pub(crate) const OP_UPSERT_BINARY_IDS: u8 = 5;
pub(crate) const OP_DELETE_ITEMS: u8 = 6;
pub(crate) const OP_RESTORE_ITEMS: u8 = 7;
pub(crate) const OP_COLLECTION_CONTROL: u8 = 8;

const STATUS_OK: u8 = 0;
const STATUS_ERR: u8 = 1;
const MAX_FRAME_BYTES: usize = 512 * 1024 * 1024;
pub(crate) const FIELDS_BINARY_MAGIC: &[u8] = b"LDBF1";

#[derive(Debug, Deserialize)]
struct BaseMeta {
    api_key: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SearchMeta {
    api_key: Option<String>,
    database_name: String,
    collection_name: String,
    dim: usize,
    vector_encoding: Option<String>,
    vector_field: Option<String>,
    k: Option<usize>,
    #[serde(rename = "where")]
    where_expr: Option<String>,
    return_fields: Option<bool>,
    nprobe: Option<usize>,
    approx: Option<bool>,
    eps: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct BatchSearchMeta {
    api_key: Option<String>,
    database_name: String,
    collection_name: String,
    dim: usize,
    n_queries: usize,
    vector_encoding: Option<String>,
    k: Option<usize>,
    #[serde(rename = "where")]
    where_expr: Option<String>,
    return_fields: Option<bool>,
    nprobe: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct BinaryItemsMeta {
    api_key: Option<String>,
    database_name: String,
    collection_name: String,
    dim: usize,
    n_vectors: usize,
    vector_encoding: Option<String>,
    ids: Option<Vec<u64>>,
    ids_encoding: Option<String>,
    ids_start: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct IdsMeta {
    api_key: Option<String>,
    database_name: String,
    collection_name: String,
    ids: Vec<u64>,
}

#[derive(Debug, Deserialize)]
struct ControlMeta {
    api_key: Option<String>,
    database_name: String,
    collection_name: String,
    action: String,
}

#[derive(Debug, Serialize)]
struct RpcOkMeta {
    ok: bool,
}

#[derive(Debug, Serialize)]
struct RpcIdsMeta {
    ids: Vec<u64>,
}

#[derive(Debug, Serialize)]
struct RpcPongMeta {
    protocol: &'static str,
    version: u32,
}

/// Derive the internal RPC port from the public HTTP port.
///
/// The coordinator uses the same rule, so users only configure HTTP addresses.
pub fn derive_rpc_port(http_port: u16) -> u16 {
    if http_port <= 55_535 {
        http_port + 10_000
    } else {
        http_port - 10_000
    }
}

/// Run the internal RPC server until the process exits.
pub async fn run_rpc_server(
    host: String,
    port: u16,
    manager: Arc<DatabaseManager>,
    api_key: Option<String>,
) -> std::io::Result<()> {
    let listener = TcpListener::bind((host.as_str(), port)).await?;
    log::info!("Starting LynseDB internal RPC on {}:{}", host, port);

    loop {
        let (stream, _) = listener.accept().await?;
        let manager = Arc::clone(&manager);
        let api_key = api_key.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, manager, api_key).await {
                log::debug!("internal RPC request failed: {}", e);
            }
        });
    }
}

async fn handle_connection(
    mut stream: TcpStream,
    manager: Arc<DatabaseManager>,
    expected_api_key: Option<String>,
) -> std::io::Result<()> {
    loop {
        let frame = match read_frame(&mut stream).await {
            Ok(frame) => frame,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::UnexpectedEof
                        | std::io::ErrorKind::ConnectionReset
                        | std::io::ErrorKind::BrokenPipe
                ) =>
            {
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        let manager = Arc::clone(&manager);
        let expected_api_key = expected_api_key.clone();
        let response = match tokio::task::spawn_blocking(move || {
            handle_frame(&frame, &manager, expected_api_key.as_deref())
        })
        .await
        {
            Ok(Ok((meta, raw))) => encode_response(STATUS_OK, &meta, &raw),
            Ok(Err(e)) => encode_response(
                STATUS_ERR,
                &serde_json::json!({"error": e.to_string()}),
                &[],
            ),
            Err(e) => encode_response(
                STATUS_ERR,
                &serde_json::json!({"error": e.to_string()}),
                &[],
            ),
        };
        if let Err(e) = write_frame(&mut stream, &response).await {
            if matches!(
                e.kind(),
                std::io::ErrorKind::UnexpectedEof
                    | std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::BrokenPipe
            ) {
                return Ok(());
            }
            return Err(e);
        }
    }
}

async fn read_frame(stream: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "internal RPC frame is too large",
        ));
    }
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload).await?;
    Ok(payload)
}

async fn write_frame(stream: &mut TcpStream, payload: &[u8]) -> std::io::Result<()> {
    let len = payload.len() as u32;
    stream.write_all(&len.to_le_bytes()).await?;
    stream.write_all(payload).await?;
    stream.flush().await
}

fn handle_frame(
    frame: &[u8],
    manager: &DatabaseManager,
    expected_api_key: Option<&str>,
) -> Result<(serde_json::Value, Vec<u8>)> {
    if frame.len() < 5 {
        return Err(LynseError::InvalidArgument(
            "internal RPC frame is too short".to_string(),
        ));
    }
    let op = frame[0];
    let meta_len = u32::from_le_bytes([frame[1], frame[2], frame[3], frame[4]]) as usize;
    if frame.len() < 5 + meta_len {
        return Err(LynseError::InvalidArgument(
            "internal RPC metadata length exceeds frame".to_string(),
        ));
    }
    let meta_bytes = &frame[5..5 + meta_len];
    let raw = &frame[5 + meta_len..];

    match op {
        OP_PING => {
            let meta: BaseMeta = serde_json::from_slice(meta_bytes)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            validate_api_key(expected_api_key, meta.api_key.as_deref())?;
            Ok((
                serde_json::to_value(RpcPongMeta {
                    protocol: "lynsedb-rpc",
                    version: 1,
                })
                .map_err(|e| LynseError::Serialization(e.to_string()))?,
                Vec::new(),
            ))
        }
        OP_SEARCH => {
            let meta: SearchMeta = serde_json::from_slice(meta_bytes)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            validate_api_key(expected_api_key, meta.api_key.as_deref())?;
            let raw = handle_search(manager, meta, raw)?;
            Ok((serde_json::json!({}), raw))
        }
        OP_BATCH_SEARCH => {
            let meta: BatchSearchMeta = serde_json::from_slice(meta_bytes)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            validate_api_key(expected_api_key, meta.api_key.as_deref())?;
            let raw = handle_batch_search(manager, meta, raw)?;
            Ok((serde_json::json!({}), raw))
        }
        OP_BULK_ADD_BINARY_IDS => {
            let meta: BinaryItemsMeta = serde_json::from_slice(meta_bytes)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            validate_api_key(expected_api_key, meta.api_key.as_deref())?;
            let ids = handle_binary_items(manager, meta, raw, false)?;
            Ok((
                serde_json::to_value(RpcIdsMeta { ids })
                    .map_err(|e| LynseError::Serialization(e.to_string()))?,
                Vec::new(),
            ))
        }
        OP_UPSERT_BINARY_IDS => {
            let meta: BinaryItemsMeta = serde_json::from_slice(meta_bytes)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            validate_api_key(expected_api_key, meta.api_key.as_deref())?;
            let ids = handle_binary_items(manager, meta, raw, true)?;
            Ok((
                serde_json::to_value(RpcIdsMeta { ids })
                    .map_err(|e| LynseError::Serialization(e.to_string()))?,
                Vec::new(),
            ))
        }
        OP_DELETE_ITEMS => {
            let meta: IdsMeta = serde_json::from_slice(meta_bytes)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            validate_api_key(expected_api_key, meta.api_key.as_deref())?;
            handle_ids(manager, meta, false)?;
            Ok((
                serde_json::to_value(RpcOkMeta { ok: true })
                    .map_err(|e| LynseError::Serialization(e.to_string()))?,
                Vec::new(),
            ))
        }
        OP_RESTORE_ITEMS => {
            let meta: IdsMeta = serde_json::from_slice(meta_bytes)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            validate_api_key(expected_api_key, meta.api_key.as_deref())?;
            handle_ids(manager, meta, true)?;
            Ok((
                serde_json::to_value(RpcOkMeta { ok: true })
                    .map_err(|e| LynseError::Serialization(e.to_string()))?,
                Vec::new(),
            ))
        }
        OP_COLLECTION_CONTROL => {
            let meta: ControlMeta = serde_json::from_slice(meta_bytes)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            validate_api_key(expected_api_key, meta.api_key.as_deref())?;
            handle_control(manager, meta)?;
            Ok((
                serde_json::to_value(RpcOkMeta { ok: true })
                    .map_err(|e| LynseError::Serialization(e.to_string()))?,
                Vec::new(),
            ))
        }
        _ => Err(LynseError::InvalidArgument(format!(
            "unknown internal RPC op {}",
            op
        ))),
    }
}

fn validate_api_key(expected: Option<&str>, supplied: Option<&str>) -> Result<()> {
    if let Some(expected) = expected {
        if supplied != Some(expected) {
            return Err(LynseError::InvalidArgument(
                "invalid internal RPC api key".to_string(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn raw_f32_cow<'a>(raw: &'a [u8], expected_floats: usize) -> Result<Cow<'a, [f32]>> {
    let expected_bytes = expected_floats
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| LynseError::InvalidArgument("raw vector byte size overflows".into()))?;
    if raw.len() != expected_bytes {
        return Err(LynseError::InvalidArgument(format!(
            "expected {} raw vector bytes, got {}",
            expected_bytes,
            raw.len()
        )));
    }
    if expected_floats == 0 {
        return Ok(Cow::Borrowed(&[]));
    }

    #[cfg(target_endian = "little")]
    {
        if raw.as_ptr().align_offset(std::mem::align_of::<f32>()) == 0 {
            let values =
                unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, expected_floats) };
            return Ok(Cow::Borrowed(values));
        }
    }

    let mut values = Vec::with_capacity(expected_floats);
    for chunk in raw.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(Cow::Owned(values))
}

pub(crate) fn normalized_vector_encoding(vector_encoding: Option<&str>) -> Result<String> {
    let encoding = vector_encoding.unwrap_or("float32").to_ascii_lowercase();
    match encoding.as_str() {
        "float32" | "f32" => Ok("float32".to_string()),
        "float16" | "f16" | "fp16" => Ok("float16".to_string()),
        other => Err(LynseError::InvalidArgument(format!(
            "unsupported vector_encoding '{}'",
            other
        ))),
    }
}

pub(crate) fn vector_dtype_for_encoding(vector_encoding: Option<&str>) -> Result<VectorDtype> {
    let encoding = normalized_vector_encoding(vector_encoding)?;
    if encoding == "float16" {
        Ok(VectorDtype::F16)
    } else {
        Ok(VectorDtype::F32)
    }
}

pub(crate) fn encoded_vector_bytes(
    expected_floats: usize,
    vector_encoding: Option<&str>,
) -> Result<usize> {
    let encoding = normalized_vector_encoding(vector_encoding)?;
    let byte_width = if encoding == "float16" {
        2usize
    } else {
        4usize
    };
    expected_floats
        .checked_mul(byte_width)
        .ok_or_else(|| LynseError::InvalidArgument("encoded vector byte size overflows".into()))
}

pub(crate) fn raw_vector_cow<'a>(
    raw: &'a [u8],
    expected_floats: usize,
    vector_encoding: Option<&str>,
) -> Result<Cow<'a, [f32]>> {
    let encoding = normalized_vector_encoding(vector_encoding)?;
    if encoding == "float32" {
        return raw_f32_cow(raw, expected_floats);
    }

    let expected_bytes = encoded_vector_bytes(expected_floats, Some(&encoding))?;
    if raw.len() != expected_bytes {
        return Err(LynseError::InvalidArgument(format!(
            "expected {} {} vector bytes, got {}",
            expected_bytes,
            encoding,
            raw.len()
        )));
    }
    let mut values = Vec::with_capacity(expected_floats);
    for chunk in raw.chunks_exact(2) {
        values.push(f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
    }
    Ok(Cow::Owned(values))
}

fn handle_search(manager: &DatabaseManager, meta: SearchMeta, raw: &[u8]) -> Result<Vec<u8>> {
    let query = raw_vector_cow(raw, meta.dim, meta.vector_encoding.as_deref())?;
    let k = meta.k.unwrap_or(10);
    let nprobe = meta.nprobe.unwrap_or(10);
    let approx = meta.approx.unwrap_or(false);
    let eps = meta.eps.unwrap_or(1e-4);
    let return_fields = meta.return_fields.unwrap_or(false);
    let vector_field = meta.vector_field.as_deref().unwrap_or("default");

    manager.get_or_open_database(&meta.database_name)?;
    let (ids, distances, fields) = manager.with_database(&meta.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&meta.collection_name, 0, 100_000)?;
        let coll = coll_arc.read();
        let sr = if vector_field == "default" {
            coll.search(
                query.as_ref(),
                k,
                meta.where_expr.as_deref(),
                nprobe,
                approx,
                eps,
            )?
        } else {
            coll.search_vector_field_with_options(
                vector_field,
                query.as_ref(),
                k,
                meta.where_expr.as_deref(),
                approx,
                eps,
            )?
        };
        let fields = if return_fields && !sr.ids.is_empty() {
            coll.retrieve_fields(&sr.ids)?
        } else {
            Vec::new()
        };
        Ok((sr.ids, sr.distances, fields))
    })?;
    Ok(encode_search_result_binary(&ids, &distances, &fields))
}

fn handle_batch_search(
    manager: &DatabaseManager,
    meta: BatchSearchMeta,
    raw: &[u8],
) -> Result<Vec<u8>> {
    let flat = raw_vector_cow(
        raw,
        meta.n_queries * meta.dim,
        meta.vector_encoding.as_deref(),
    )?;
    let k = meta.k.unwrap_or(10);
    let nprobe = meta.nprobe.unwrap_or(10);
    let return_fields = meta.return_fields.unwrap_or(false);

    manager.get_or_open_database(&meta.database_name)?;
    manager.with_database(&meta.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&meta.collection_name, 0, 100_000)?;
        let coll = coll_arc.read();
        let results = coll.batch_search(
            flat.as_ref(),
            meta.n_queries,
            k,
            meta.where_expr.as_deref(),
            nprobe,
        )?;

        let mut buf = Vec::with_capacity(4 + results.len() * (4 + k * 12 + 4));
        buf.extend_from_slice(&(results.len() as u32).to_le_bytes());
        for sr in &results {
            let fields = if return_fields && !sr.ids.is_empty() {
                coll.retrieve_fields(&sr.ids)?
            } else {
                Vec::new()
            };
            let block = encode_search_result_binary(&sr.ids, &sr.distances, &fields);
            buf.extend_from_slice(&block);
        }
        Ok(buf)
    })
}

fn handle_binary_items(
    manager: &DatabaseManager,
    meta: BinaryItemsMeta,
    raw: &[u8],
    upsert: bool,
) -> Result<Vec<u64>> {
    let vector_dtype = vector_dtype_for_encoding(meta.vector_encoding.as_deref())?;
    let (encoded_vectors, ids, fields) = if let Some(ids) = meta.ids.clone() {
        let (encoded_vectors, fields) = split_encoded_binary_item_payload(
            raw,
            meta.n_vectors * meta.dim,
            meta.vector_encoding.as_deref(),
        )?;
        (encoded_vectors, ids, fields)
    } else {
        split_encoded_binary_item_payload_with_ids(
            raw,
            meta.n_vectors * meta.dim,
            meta.n_vectors,
            meta.vector_encoding.as_deref(),
            meta.ids_encoding.as_deref(),
            meta.ids_start,
        )?
    };
    if ids.len() != meta.n_vectors {
        return Err(LynseError::InvalidArgument(format!(
            "ids length ({}) must match n_vectors ({})",
            ids.len(),
            meta.n_vectors
        )));
    }
    if let Some(fields) = fields.as_ref() {
        if fields.len() != meta.n_vectors {
            return Err(LynseError::InvalidArgument(format!(
                "fields length ({}) must match n_vectors ({})",
                fields.len(),
                meta.n_vectors
            )));
        }
    }
    manager.get_or_open_database(&meta.database_name)?;
    let ids_for_write = ids.clone();
    manager.with_database(&meta.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&meta.collection_name, 0, 100_000)?;
        let mut coll = coll_arc.write();
        if upsert {
            let vectors = raw_vector_cow(
                encoded_vectors,
                meta.n_vectors * meta.dim,
                meta.vector_encoding.as_deref(),
            )?;
            upsert_binary_items(
                &mut coll,
                &ids_for_write,
                meta.dim,
                vectors.as_ref(),
                fields.as_ref(),
            )
        } else if vector_dtype == VectorDtype::F16 && coll.vector_dtype() == VectorDtype::F16 {
            let fields = add_fields_from_payload(fields.as_ref());
            coll.add_items_encoded_vectors(
                encoded_vectors,
                VectorDtype::F16,
                meta.n_vectors,
                &ids_for_write,
                fields.as_ref().map(|items| items.as_slice()),
            )
        } else {
            let vectors = raw_vector_cow(
                encoded_vectors,
                meta.n_vectors * meta.dim,
                meta.vector_encoding.as_deref(),
            )?;
            let fields = add_fields_from_payload(fields.as_ref());
            coll.add_items(
                vectors.as_ref(),
                meta.n_vectors,
                &ids_for_write,
                fields.as_ref().map(|items| items.as_slice()),
            )
        }
    })?;
    Ok(ids)
}

pub(crate) fn split_encoded_binary_item_payload<'a>(
    raw: &'a [u8],
    expected_vector_floats: usize,
    vector_encoding: Option<&str>,
) -> Result<(
    &'a [u8],
    Option<Vec<Option<HashMap<String, serde_json::Value>>>>,
)> {
    let vector_bytes = encoded_vector_bytes(expected_vector_floats, vector_encoding)?;
    if raw.len() < vector_bytes {
        return Err(LynseError::InvalidArgument(format!(
            "expected at least {} raw vector bytes, got {}",
            vector_bytes,
            raw.len()
        )));
    }
    let encoded_vectors = &raw[..vector_bytes];
    let fields_raw = &raw[vector_bytes..];
    if fields_raw.is_empty() {
        return Ok((encoded_vectors, None));
    }
    let fields = decode_fields_payload(fields_raw)?;
    Ok((encoded_vectors, Some(fields)))
}

pub(crate) fn split_binary_item_payload_with_ids<'a>(
    raw: &'a [u8],
    expected_vector_floats: usize,
    n_vectors: usize,
    vector_encoding: Option<&str>,
    ids_encoding: Option<&str>,
    ids_start: Option<u64>,
) -> Result<(
    Cow<'a, [f32]>,
    Vec<u64>,
    Option<Vec<Option<HashMap<String, serde_json::Value>>>>,
)> {
    let (encoded_vectors, ids, fields) = split_encoded_binary_item_payload_with_ids(
        raw,
        expected_vector_floats,
        n_vectors,
        vector_encoding,
        ids_encoding,
        ids_start,
    )?;
    let vectors = raw_vector_cow(encoded_vectors, expected_vector_floats, vector_encoding)?;
    Ok((vectors, ids, fields))
}

pub(crate) fn split_encoded_binary_item_payload_with_ids<'a>(
    raw: &'a [u8],
    expected_vector_floats: usize,
    n_vectors: usize,
    vector_encoding: Option<&str>,
    ids_encoding: Option<&str>,
    ids_start: Option<u64>,
) -> Result<(
    &'a [u8],
    Vec<u64>,
    Option<Vec<Option<HashMap<String, serde_json::Value>>>>,
)> {
    let vector_bytes = encoded_vector_bytes(expected_vector_floats, vector_encoding)?;
    if raw.len() < vector_bytes {
        return Err(LynseError::InvalidArgument(format!(
            "expected at least {} raw vector bytes, got {}",
            vector_bytes,
            raw.len()
        )));
    }
    let encoded_vectors = &raw[..vector_bytes];
    let after_vectors = &raw[vector_bytes..];
    let encoding = ids_encoding.unwrap_or("raw").to_ascii_lowercase();
    let (ids, fields_raw) = match encoding.as_str() {
        "raw" | "" => {
            let id_bytes = n_vectors
                .checked_mul(std::mem::size_of::<u64>())
                .ok_or_else(|| LynseError::InvalidArgument("raw id byte size overflows".into()))?;
            if after_vectors.len() < id_bytes {
                return Err(LynseError::InvalidArgument(format!(
                    "expected at least {} raw id bytes after vectors, got {}",
                    id_bytes,
                    after_vectors.len()
                )));
            }
            let ids_raw = &after_vectors[..id_bytes];
            let mut ids = Vec::with_capacity(n_vectors);
            for chunk in ids_raw.chunks_exact(8) {
                ids.push(u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]));
            }
            (ids, &after_vectors[id_bytes..])
        }
        "range" => {
            let start = ids_start.ok_or_else(|| {
                LynseError::InvalidArgument("ids_start is required for range id encoding".into())
            })?;
            let count = u64::try_from(n_vectors)
                .map_err(|_| LynseError::InvalidArgument("n_vectors does not fit in u64".into()))?;
            let end = start
                .checked_add(count)
                .ok_or_else(|| LynseError::InvalidArgument("range ids overflow u64".into()))?;
            ((start..end).collect(), after_vectors)
        }
        other => {
            return Err(LynseError::InvalidArgument(format!(
                "unsupported ids_encoding '{}'",
                other
            )));
        }
    };

    if fields_raw.is_empty() {
        return Ok((encoded_vectors, ids, None));
    }
    let fields = decode_fields_payload(fields_raw)?;
    Ok((encoded_vectors, ids, Some(fields)))
}

pub(crate) fn decode_fields_payload(
    raw: &[u8],
) -> Result<Vec<Option<HashMap<String, serde_json::Value>>>> {
    if let Some(binary) = raw.strip_prefix(FIELDS_BINARY_MAGIC) {
        return decode_binary_fields(binary);
    }
    Err(LynseError::Serialization(
        "invalid RPC fields payload magic".to_string(),
    ))
}

struct BinaryCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BinaryCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(LynseError::Serialization(
                "unexpected end of binary fields payload".to_string(),
            ));
        }
        let value = self.data[self.pos];
        self.pos += 1;
        Ok(value)
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| LynseError::Serialization("binary fields offset overflows".into()))?;
        if end > self.data.len() {
            return Err(LynseError::Serialization(
                "unexpected end of binary fields payload".to_string(),
            ));
        }
        let out = &self.data[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i64(&mut self) -> Result<i64> {
        let bytes = self.read_exact(8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let bytes = self.read_exact(8)?;
        Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u32()? as usize;
        let bytes = self.read_exact(len)?;
        std::str::from_utf8(bytes)
            .map(|s| s.to_string())
            .map_err(|e| {
                LynseError::Serialization(format!("invalid utf-8 in fields payload: {}", e))
            })
    }

    fn ensure_finished(&self) -> Result<()> {
        if self.pos != self.data.len() {
            return Err(LynseError::Serialization(
                "trailing bytes in binary fields payload".to_string(),
            ));
        }
        Ok(())
    }
}

fn decode_binary_fields(raw: &[u8]) -> Result<Vec<Option<HashMap<String, serde_json::Value>>>> {
    let mut cursor = BinaryCursor::new(raw);
    let count = cursor.read_u32()? as usize;
    let mut fields = Vec::with_capacity(count);
    for _ in 0..count {
        let present = cursor.read_u8()?;
        match present {
            0 => fields.push(None),
            1 => fields.push(Some(decode_binary_object_hashmap(&mut cursor, 0)?)),
            other => {
                return Err(LynseError::Serialization(format!(
                    "invalid field presence tag {}",
                    other
                )));
            }
        }
    }
    cursor.ensure_finished()?;
    Ok(fields)
}

fn decode_binary_object_hashmap(
    cursor: &mut BinaryCursor<'_>,
    depth: usize,
) -> Result<HashMap<String, serde_json::Value>> {
    if depth > 128 {
        return Err(LynseError::Serialization(
            "binary fields payload is nested too deeply".to_string(),
        ));
    }
    let count = cursor.read_u32()? as usize;
    let mut out = HashMap::with_capacity(count);
    for _ in 0..count {
        let key = cursor.read_string()?;
        let value = decode_binary_value(cursor, depth + 1)?;
        out.insert(key, value);
    }
    Ok(out)
}

fn decode_binary_object_value(
    cursor: &mut BinaryCursor<'_>,
    depth: usize,
) -> Result<serde_json::Value> {
    let map = decode_binary_object_hashmap(cursor, depth)?;
    let object = map.into_iter().collect();
    Ok(serde_json::Value::Object(object))
}

fn decode_binary_value(cursor: &mut BinaryCursor<'_>, depth: usize) -> Result<serde_json::Value> {
    if depth > 128 {
        return Err(LynseError::Serialization(
            "binary fields payload is nested too deeply".to_string(),
        ));
    }
    let tag = cursor.read_u8()?;
    match tag {
        0 => Ok(serde_json::Value::Null),
        1 => Ok(serde_json::Value::Bool(false)),
        2 => Ok(serde_json::Value::Bool(true)),
        3 => Ok(serde_json::Value::Number(cursor.read_i64()?.into())),
        4 => Ok(serde_json::Value::Number(cursor.read_u64()?.into())),
        5 => serde_json::Number::from_f64(cursor.read_f64()?)
            .map(serde_json::Value::Number)
            .ok_or_else(|| LynseError::Serialization("non-finite float in fields payload".into())),
        6 => Ok(serde_json::Value::String(cursor.read_string()?)),
        7 => {
            let count = cursor.read_u32()? as usize;
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(decode_binary_value(cursor, depth + 1)?);
            }
            Ok(serde_json::Value::Array(values))
        }
        8 => decode_binary_object_value(cursor, depth + 1),
        other => Err(LynseError::Serialization(format!(
            "unknown binary value tag {}",
            other
        ))),
    }
}

pub(crate) fn add_fields_from_payload(
    fields: Option<&Vec<Option<HashMap<String, serde_json::Value>>>>,
) -> Option<Vec<HashMap<String, serde_json::Value>>> {
    let fields = fields?;
    let normalized: Vec<HashMap<String, serde_json::Value>> = fields
        .iter()
        .map(|field| field.clone().unwrap_or_default())
        .collect();
    if normalized.iter().any(|field| !field.is_empty()) {
        Some(normalized)
    } else {
        None
    }
}

pub(crate) fn upsert_binary_items(
    coll: &mut crate::engine::Collection,
    ids: &[u64],
    dim: usize,
    vectors: &[f32],
    fields: Option<&Vec<Option<HashMap<String, serde_json::Value>>>>,
) -> Result<()> {
    let Some(fields) = fields else {
        return coll.upsert_items(ids, vectors, ids.len(), None);
    };

    let mut vectors_without_fields = Vec::new();
    let mut ids_without_fields = Vec::new();
    let mut vectors_with_fields = Vec::new();
    let mut ids_with_fields = Vec::new();
    let mut fields_with_fields = Vec::new();

    for (idx, maybe_field) in fields.iter().enumerate() {
        let start = idx * dim;
        let end = start + dim;
        if let Some(field) = maybe_field {
            ids_with_fields.push(ids[idx]);
            vectors_with_fields.extend_from_slice(&vectors[start..end]);
            fields_with_fields.push(field.clone());
        } else {
            ids_without_fields.push(ids[idx]);
            vectors_without_fields.extend_from_slice(&vectors[start..end]);
        }
    }

    if !ids_without_fields.is_empty() {
        coll.upsert_items(
            &ids_without_fields,
            &vectors_without_fields,
            ids_without_fields.len(),
            None,
        )?;
    }
    if !ids_with_fields.is_empty() {
        coll.upsert_items(
            &ids_with_fields,
            &vectors_with_fields,
            ids_with_fields.len(),
            Some(&fields_with_fields),
        )?;
    }
    Ok(())
}

fn handle_ids(manager: &DatabaseManager, meta: IdsMeta, restore: bool) -> Result<()> {
    manager.get_or_open_database(&meta.database_name)?;
    manager.with_database(&meta.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&meta.collection_name, 0, 100_000)?;
        let coll = coll_arc.read();
        if restore {
            coll.restore_items(&meta.ids)
        } else {
            coll.delete_items(&meta.ids)
        }
    })
}

fn handle_control(manager: &DatabaseManager, meta: ControlMeta) -> Result<()> {
    manager.get_or_open_database(&meta.database_name)?;
    manager.with_database(&meta.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&meta.collection_name, 0, 100_000)?;
        match meta.action.as_str() {
            "commit" => {
                let coll = coll_arc.read();
                coll.commit()
            }
            "flush" => {
                let coll = coll_arc.read();
                coll.flush()
            }
            "checkpoint" => {
                let coll = coll_arc.read();
                coll.checkpoint()
            }
            "close" | "close_collection" => {
                let mut coll = coll_arc.write();
                coll.close()
            }
            other => Err(LynseError::InvalidArgument(format!(
                "unsupported internal RPC control action '{}'",
                other
            ))),
        }
    })
}

fn encode_response(status: u8, meta: &serde_json::Value, raw: &[u8]) -> Vec<u8> {
    let meta_bytes = serde_json::to_vec(meta).unwrap_or_else(|_| b"{}".to_vec());
    let mut buf = Vec::with_capacity(1 + 4 + meta_bytes.len() + raw.len());
    buf.push(status);
    buf.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(&meta_bytes);
    buf.extend_from_slice(raw);
    buf
}

pub(crate) fn encode_search_result_binary(
    ids: &[u64],
    distances: &[f32],
    fields: &[HashMap<String, serde_json::Value>],
) -> Vec<u8> {
    let fields_json = if fields.is_empty() {
        Vec::new()
    } else {
        serde_json::to_vec(fields).unwrap_or_default()
    };
    let mut buf = Vec::with_capacity(4 + ids.len() * 12 + 4 + fields_json.len());
    buf.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    for &id in ids {
        buf.extend_from_slice(&id.to_le_bytes());
    }
    for &distance in distances {
        buf.extend_from_slice(&distance.to_le_bytes());
    }
    buf.extend_from_slice(&(fields_json.len() as u32).to_le_bytes());
    buf.extend_from_slice(&fields_json);
    buf
}
