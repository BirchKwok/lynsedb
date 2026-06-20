//! Rust-side cluster read coordinator primitives.
//!
//! This module is intentionally scoped to hot read fan-out first. It can be
//! wired into Actix handlers without moving Python's write-routing metadata
//! logic all at once.

use crate::error::{LynseError, Result};
use crate::rpc::{self, OP_BATCH_SEARCH, OP_SEARCH};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time;

const MAX_CLUSTER_RPC_FRAME_BYTES: usize = 512 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct RustReadCoordinator {
    groups: Arc<Vec<ShardGroup>>,
    collections: Arc<HashMap<String, CollectionConfig>>,
    timeout: Duration,
    api_key: Option<String>,
    rpc_pool: Arc<Mutex<HashMap<String, Vec<TcpStream>>>>,
    rpc_max_idle_per_uri: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ClusterConfig {
    #[serde(default)]
    pub shard_groups: Vec<ShardGroup>,
    #[serde(default)]
    pub shards: Vec<ShardGroup>,
    #[serde(default)]
    pub collections: HashMap<String, CollectionConfig>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ShardGroup {
    pub name: String,
    pub primary: String,
    #[serde(default)]
    pub replicas: Vec<Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CollectionConfig {
    pub index_mode: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchBlock {
    pub ids: Vec<u64>,
    pub distances: Vec<f32>,
    pub fields: Vec<HashMap<String, Value>>,
}

impl RustReadCoordinator {
    pub fn from_config(
        config: ClusterConfig,
        timeout: Duration,
        api_key: Option<String>,
    ) -> Result<Self> {
        let groups = if !config.shard_groups.is_empty() {
            config.shard_groups
        } else {
            config.shards
        };
        if groups.is_empty() {
            return Err(LynseError::InvalidArgument(
                "cluster config requires at least one shard group".to_string(),
            ));
        }
        Ok(Self {
            groups: Arc::new(groups),
            collections: Arc::new(config.collections),
            timeout,
            api_key,
            rpc_pool: Arc::new(Mutex::new(HashMap::new())),
            rpc_max_idle_per_uri: 8,
        })
    }

    pub fn from_path(
        path: impl AsRef<Path>,
        timeout: Duration,
        api_key: Option<String>,
    ) -> Result<Self> {
        let raw = fs::read_to_string(path)?;
        let config: ClusterConfig =
            serde_json::from_str(&raw).map_err(|e| LynseError::Serialization(e.to_string()))?;
        Self::from_config(config, timeout, api_key)
    }

    pub async fn search_binary(&self, mut meta: Value, raw: Vec<u8>) -> Result<Vec<u8>> {
        normalize_rpc_meta(&mut meta, self.api_key.as_deref())?;
        let k = json_usize(&meta, "k", 10);
        let return_fields = json_bool(&meta, "return_fields", false);
        let ascending = self.ascending_for_meta(&meta);
        let shard_raw = self.fanout_rpc(OP_SEARCH, meta, raw).await?;
        let mut blocks = Vec::with_capacity(shard_raw.len());
        for buf in &shard_raw {
            let (block, offset) = decode_search_result_binary(buf, 0)?;
            if offset != buf.len() {
                return Err(LynseError::Serialization(
                    "trailing bytes in shard search result".to_string(),
                ));
            }
            blocks.push(block);
        }
        let merged = merge_search_blocks(&blocks, k, ascending, return_fields);
        Ok(rpc::encode_search_result_binary(
            &merged.ids,
            &merged.distances,
            &merged.fields,
        ))
    }

    pub async fn batch_search_binary(&self, mut meta: Value, raw: Vec<u8>) -> Result<Vec<u8>> {
        normalize_rpc_meta(&mut meta, self.api_key.as_deref())?;
        let k = json_usize(&meta, "k", 10);
        let n_queries = json_usize(&meta, "n_queries", 0);
        let return_fields = json_bool(&meta, "return_fields", false);
        let ascending = self.ascending_for_meta(&meta);
        let shard_raw = self.fanout_rpc(OP_BATCH_SEARCH, meta, raw).await?;
        let mut per_query: Vec<Vec<SearchBlock>> = vec![Vec::new(); n_queries];

        for buf in &shard_raw {
            let mut offset = 0usize;
            if buf.len() < 4 {
                return Err(LynseError::Serialization(
                    "batch search result frame is too short".to_string(),
                ));
            }
            let count = read_u32(buf, &mut offset)? as usize;
            if count != n_queries {
                return Err(LynseError::InvalidArgument(format!(
                    "batch_search_binary returned {} queries, expected {}",
                    count, n_queries
                )));
            }
            for bucket in per_query.iter_mut().take(count) {
                let (block, next) = decode_search_result_binary(buf, offset)?;
                offset = next;
                bucket.push(block);
            }
            if offset != buf.len() {
                return Err(LynseError::Serialization(
                    "trailing bytes in shard batch search result".to_string(),
                ));
            }
        }

        let mut out = Vec::new();
        out.extend_from_slice(&(n_queries as u32).to_le_bytes());
        for blocks in &per_query {
            let merged = merge_search_blocks(blocks, k, ascending, return_fields);
            out.extend_from_slice(&rpc::encode_search_result_binary(
                &merged.ids,
                &merged.distances,
                &merged.fields,
            ));
        }
        Ok(out)
    }

    async fn fanout_rpc(&self, op: u8, meta: Value, raw: Vec<u8>) -> Result<Vec<Vec<u8>>> {
        let meta_bytes =
            serde_json::to_vec(&meta).map_err(|e| LynseError::Serialization(e.to_string()))?;
        let mut payload = Vec::with_capacity(1 + 4 + meta_bytes.len() + raw.len());
        payload.push(op);
        payload.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
        payload.extend_from_slice(&meta_bytes);
        payload.extend_from_slice(&raw);
        let payload = Arc::new(payload);

        if self.groups.len() == 1 {
            let (_meta, raw) = self
                .rpc_request_payload(&self.groups[0].primary, Arc::clone(&payload))
                .await?;
            return Ok(vec![raw]);
        }
        if self.groups.len() == 2 {
            let left_uri = self.groups[0].primary.clone();
            let right_uri = self.groups[1].primary.clone();
            let ((_, left_raw), (_, right_raw)) = tokio::try_join!(
                self.rpc_request_payload(&left_uri, Arc::clone(&payload)),
                self.rpc_request_payload(&right_uri, Arc::clone(&payload)),
            )?;
            return Ok(vec![left_raw, right_raw]);
        }

        let mut handles = Vec::with_capacity(self.groups.len());
        for group in self.groups.iter() {
            let uri = group.primary.clone();
            let payload = Arc::clone(&payload);
            let coordinator = self.clone();
            handles.push(tokio::spawn(async move {
                coordinator.rpc_request_payload(&uri, payload).await
            }));
        }

        let mut out = Vec::with_capacity(handles.len());
        for handle in handles {
            let (_meta, raw) = handle
                .await
                .map_err(|e| LynseError::Storage(format!("cluster RPC task failed: {}", e)))??;
            out.push(raw);
        }
        Ok(out)
    }

    #[cfg(test)]
    async fn rpc_request(
        &self,
        http_uri: &str,
        op: u8,
        meta: Value,
        raw: Vec<u8>,
    ) -> Result<(Value, Vec<u8>)> {
        let meta_bytes =
            serde_json::to_vec(&meta).map_err(|e| LynseError::Serialization(e.to_string()))?;
        let mut payload = Vec::with_capacity(1 + 4 + meta_bytes.len() + raw.len());
        payload.push(op);
        payload.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
        payload.extend_from_slice(&meta_bytes);
        payload.extend_from_slice(&raw);
        self.rpc_request_payload(http_uri, Arc::new(payload)).await
    }

    async fn rpc_request_payload(
        &self,
        http_uri: &str,
        payload: Arc<Vec<u8>>,
    ) -> Result<(Value, Vec<u8>)> {
        let mut last_error = None;
        for _ in 0..2 {
            let mut stream = match self.take_rpc_stream(http_uri).await {
                Ok(stream) => stream,
                Err(e) => {
                    last_error = Some(e);
                    continue;
                }
            };
            match rpc_roundtrip(&mut stream, payload.as_slice(), self.timeout).await {
                Ok(response) => {
                    self.return_rpc_stream(http_uri, stream);
                    return Ok(response);
                }
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| LynseError::Storage("cluster RPC request failed".into())))
    }

    async fn take_rpc_stream(&self, http_uri: &str) -> Result<TcpStream> {
        let uri = normalize_uri(http_uri);
        let pooled = {
            let mut pool = self.rpc_pool.lock();
            let mut remove_bucket = false;
            let stream = if let Some(bucket) = pool.get_mut(&uri) {
                let stream = bucket.pop();
                remove_bucket = bucket.is_empty();
                stream
            } else {
                None
            };
            if remove_bucket {
                pool.remove(&uri);
            }
            stream
        };
        if let Some(stream) = pooled {
            return Ok(stream);
        }

        let (host, port) = derive_rpc_target(&uri)?;
        let stream = time::timeout(self.timeout, TcpStream::connect((host.as_str(), port)))
            .await
            .map_err(|_| LynseError::Storage("cluster RPC connect timed out".to_string()))?
            .map_err(LynseError::Io)?;
        let _ = stream.set_nodelay(true);
        Ok(stream)
    }

    fn return_rpc_stream(&self, http_uri: &str, stream: TcpStream) {
        let uri = normalize_uri(http_uri);
        let mut pool = self.rpc_pool.lock();
        let bucket = pool.entry(uri).or_default();
        if bucket.len() < self.rpc_max_idle_per_uri {
            bucket.push(stream);
        }
    }

    fn ascending_for_meta(&self, meta: &Value) -> bool {
        if let Some(index_mode) = meta.get("index_mode").and_then(Value::as_str) {
            return is_ascending_index(Some(index_mode));
        }
        let db_name = json_string(meta, "database_name");
        let coll_name = json_string(meta, "collection_name");
        let key = format!("{}/{}", db_name, coll_name);
        let index_mode = self
            .collections
            .get(&key)
            .and_then(|coll| coll.index_mode.as_deref());
        is_ascending_index(index_mode)
    }
}

fn normalize_uri(uri: &str) -> String {
    uri.trim().trim_end_matches('/').to_string()
}

pub fn is_ascending_index(index_mode: Option<&str>) -> bool {
    crate::distance::DistanceMetric::from_index_mode(index_mode.unwrap_or("FLAT-IP"))
        .is_some_and(|metric| metric.is_ascending())
}

pub fn merge_search_blocks(
    blocks: &[SearchBlock],
    k: usize,
    ascending: bool,
    return_fields: bool,
) -> SearchBlock {
    if k == 0 {
        return SearchBlock {
            ids: Vec::new(),
            distances: Vec::new(),
            fields: Vec::new(),
        };
    }

    if !return_fields {
        let total = blocks.len().saturating_mul(k);
        let mut merged: Vec<(u64, f32)> = Vec::with_capacity(total);
        for block in blocks {
            for (&id, &distance) in block.ids.iter().zip(&block.distances) {
                merged.push((id, distance));
            }
        }

        if merged.len() > k {
            merged.select_nth_unstable_by(k - 1, |left, right| {
                compare_distance(left.1, right.1, ascending)
            });
            merged.truncate(k);
        }
        merged.sort_unstable_by(|left, right| compare_distance(left.1, right.1, ascending));

        return SearchBlock {
            ids: merged.iter().map(|(id, _)| *id).collect(),
            distances: merged.iter().map(|(_, distance)| *distance).collect(),
            fields: Vec::new(),
        };
    }

    let mut merged: Vec<(u64, f32, Option<HashMap<String, Value>>)> = Vec::new();
    for block in blocks {
        for (idx, (&id, &distance)) in block.ids.iter().zip(&block.distances).enumerate() {
            let field = block.fields.get(idx).cloned();
            merged.push((id, distance, field));
        }
    }

    if merged.len() > k {
        merged.select_nth_unstable_by(k, |left, right| {
            compare_distance(left.1, right.1, ascending)
        });
        merged.truncate(k);
    }
    merged.sort_unstable_by(|left, right| compare_distance(left.1, right.1, ascending));

    SearchBlock {
        ids: merged.iter().map(|(id, _, _)| *id).collect(),
        distances: merged.iter().map(|(_, distance, _)| *distance).collect(),
        fields: if return_fields {
            merged
                .into_iter()
                .map(|(_, _, field)| field.unwrap_or_default())
                .collect()
        } else {
            Vec::new()
        },
    }
}

fn compare_distance(left: f32, right: f32, ascending: bool) -> Ordering {
    let ord = left.partial_cmp(&right).unwrap_or(Ordering::Equal);
    if ascending {
        ord
    } else {
        ord.reverse()
    }
}

pub fn decode_search_result_binary(buf: &[u8], mut offset: usize) -> Result<(SearchBlock, usize)> {
    let n = read_u32(buf, &mut offset)? as usize;
    let mut ids = Vec::with_capacity(n);
    for _ in 0..n {
        ids.push(read_u64(buf, &mut offset)?);
    }
    let mut distances = Vec::with_capacity(n);
    for _ in 0..n {
        distances.push(read_f32(buf, &mut offset)?);
    }
    let fields_len = read_u32(buf, &mut offset)? as usize;
    if buf.len() < offset + fields_len {
        return Err(LynseError::Serialization(
            "search result fields length exceeds frame".to_string(),
        ));
    }
    let fields = if fields_len == 0 {
        Vec::new()
    } else {
        serde_json::from_slice(&buf[offset..offset + fields_len])
            .map_err(|e| LynseError::Serialization(e.to_string()))?
    };
    offset += fields_len;
    Ok((
        SearchBlock {
            ids,
            distances,
            fields,
        },
        offset,
    ))
}

async fn rpc_roundtrip(
    stream: &mut TcpStream,
    payload: &[u8],
    timeout: Duration,
) -> Result<(Value, Vec<u8>)> {
    time::timeout(
        timeout,
        stream.write_all(&(payload.len() as u32).to_le_bytes()),
    )
    .await
    .map_err(|_| LynseError::Storage("cluster RPC write timed out".to_string()))?
    .map_err(LynseError::Io)?;
    time::timeout(timeout, stream.write_all(payload))
        .await
        .map_err(|_| LynseError::Storage("cluster RPC write timed out".to_string()))?
        .map_err(LynseError::Io)?;

    let mut header = [0u8; 4];
    time::timeout(timeout, stream.read_exact(&mut header))
        .await
        .map_err(|_| LynseError::Storage("cluster RPC read timed out".to_string()))?
        .map_err(LynseError::Io)?;
    let frame_len = u32::from_le_bytes(header) as usize;
    if frame_len > MAX_CLUSTER_RPC_FRAME_BYTES {
        return Err(LynseError::InvalidArgument(format!(
            "cluster RPC frame too large: {} bytes",
            frame_len
        )));
    }
    let mut frame = vec![0u8; frame_len];
    time::timeout(timeout, stream.read_exact(&mut frame))
        .await
        .map_err(|_| LynseError::Storage("cluster RPC read timed out".to_string()))?
        .map_err(LynseError::Io)?;

    decode_rpc_response(&frame)
}

fn decode_rpc_response(frame: &[u8]) -> Result<(Value, Vec<u8>)> {
    if frame.len() < 5 {
        return Err(LynseError::Serialization(
            "cluster RPC response frame is too short".to_string(),
        ));
    }
    let status = frame[0];
    let meta_len = u32::from_le_bytes([frame[1], frame[2], frame[3], frame[4]]) as usize;
    if frame.len() < 5 + meta_len {
        return Err(LynseError::Serialization(
            "cluster RPC response metadata length exceeds frame".to_string(),
        ));
    }
    let meta: Value = if meta_len == 0 {
        Value::Object(Map::new())
    } else {
        serde_json::from_slice(&frame[5..5 + meta_len])
            .map_err(|e| LynseError::Serialization(e.to_string()))?
    };
    if status != 0 {
        let message = meta
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("cluster RPC request failed");
        return Err(LynseError::Storage(message.to_string()));
    }
    Ok((meta, frame[5 + meta_len..].to_vec()))
}

fn derive_rpc_target(uri: &str) -> Result<(String, u16)> {
    let trimmed = normalize_uri(uri);
    let without_scheme = trimmed
        .strip_prefix("http://")
        .or_else(|| trimmed.strip_prefix("https://"))
        .ok_or_else(|| LynseError::InvalidArgument(format!("invalid shard URI '{}'", uri)))?;
    let authority = without_scheme.split('/').next().unwrap_or(without_scheme);
    let (host, port) = authority
        .rsplit_once(':')
        .ok_or_else(|| LynseError::InvalidArgument(format!("shard URI has no port '{}'", uri)))?;
    let http_port = port
        .parse::<u16>()
        .map_err(|e| LynseError::InvalidArgument(format!("invalid shard URI port: {}", e)))?;
    Ok((host.to_string(), rpc::derive_rpc_port(http_port)))
}

fn normalize_rpc_meta(meta: &mut Value, api_key: Option<&str>) -> Result<()> {
    let object = meta.as_object_mut().ok_or_else(|| {
        LynseError::InvalidArgument("cluster RPC metadata must be an object".to_string())
    })?;
    normalize_usize_field(object, "dim")?;
    normalize_usize_field(object, "k")?;
    normalize_usize_field(object, "n_queries")?;
    normalize_usize_field(object, "nprobe")?;
    normalize_bool_field(object, "return_fields")?;
    normalize_bool_field(object, "approx")?;
    normalize_f32_field(object, "eps")?;
    if let Some(api_key) = api_key {
        object.insert("api_key".to_string(), Value::String(api_key.to_string()));
    }
    Ok(())
}

fn normalize_usize_field(object: &mut Map<String, Value>, key: &str) -> Result<()> {
    if let Some(value) = object.get(key) {
        if let Some(text) = value.as_str() {
            let parsed = text.parse::<u64>().map_err(|e| {
                LynseError::InvalidArgument(format!("invalid {} value '{}': {}", key, text, e))
            })?;
            object.insert(key.to_string(), Value::Number(parsed.into()));
        }
    }
    Ok(())
}

fn normalize_bool_field(object: &mut Map<String, Value>, key: &str) -> Result<()> {
    if let Some(value) = object.get(key) {
        if let Some(text) = value.as_str() {
            let parsed = matches!(
                text.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
            object.insert(key.to_string(), Value::Bool(parsed));
        }
    }
    Ok(())
}

fn normalize_f32_field(object: &mut Map<String, Value>, key: &str) -> Result<()> {
    if let Some(value) = object.get(key) {
        if let Some(text) = value.as_str() {
            let parsed = text.parse::<f64>().map_err(|e| {
                LynseError::InvalidArgument(format!("invalid {} value '{}': {}", key, text, e))
            })?;
            let number = serde_json::Number::from_f64(parsed)
                .ok_or_else(|| LynseError::InvalidArgument(format!("invalid {} value", key)))?;
            object.insert(key.to_string(), Value::Number(number));
        }
    }
    Ok(())
}

fn json_usize(meta: &Value, key: &str, default: usize) -> usize {
    meta.get(key)
        .and_then(|value| {
            value
                .as_u64()
                .map(|v| v as usize)
                .or_else(|| value.as_str().and_then(|text| text.parse::<usize>().ok()))
        })
        .unwrap_or(default)
}

fn json_bool(meta: &Value, key: &str, default: bool) -> bool {
    meta.get(key)
        .and_then(|value| {
            value.as_bool().or_else(|| {
                value.as_str().map(|text| {
                    matches!(
                        text.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
            })
        })
        .unwrap_or(default)
}

fn json_string(meta: &Value, key: &str) -> String {
    meta.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn read_u32(buf: &[u8], offset: &mut usize) -> Result<u32> {
    if buf.len() < *offset + 4 {
        return Err(LynseError::Serialization(
            "unexpected end of binary search result".to_string(),
        ));
    }
    let out = u32::from_le_bytes([
        buf[*offset],
        buf[*offset + 1],
        buf[*offset + 2],
        buf[*offset + 3],
    ]);
    *offset += 4;
    Ok(out)
}

fn read_u64(buf: &[u8], offset: &mut usize) -> Result<u64> {
    if buf.len() < *offset + 8 {
        return Err(LynseError::Serialization(
            "unexpected end of binary search result".to_string(),
        ));
    }
    let out = u64::from_le_bytes([
        buf[*offset],
        buf[*offset + 1],
        buf[*offset + 2],
        buf[*offset + 3],
        buf[*offset + 4],
        buf[*offset + 5],
        buf[*offset + 6],
        buf[*offset + 7],
    ]);
    *offset += 8;
    Ok(out)
}

fn read_f32(buf: &[u8], offset: &mut usize) -> Result<f32> {
    if buf.len() < *offset + 4 {
        return Err(LynseError::Serialization(
            "unexpected end of binary search result".to_string(),
        ));
    }
    let out = f32::from_le_bytes([
        buf[*offset],
        buf[*offset + 1],
        buf[*offset + 2],
        buf[*offset + 3],
    ]);
    *offset += 4;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(ids: &[u64], distances: &[f32]) -> SearchBlock {
        SearchBlock {
            ids: ids.to_vec(),
            distances: distances.to_vec(),
            fields: Vec::new(),
        }
    }

    #[test]
    fn merge_search_blocks_obeys_metric_order() {
        let merged = merge_search_blocks(
            &[block(&[1, 2], &[0.4, 0.9]), block(&[3], &[0.8])],
            2,
            false,
            false,
        );
        assert_eq!(merged.ids, vec![2, 3]);
        assert_eq!(merged.distances, vec![0.9, 0.8]);

        let merged = merge_search_blocks(
            &[block(&[1, 2], &[4.0, 1.0]), block(&[3], &[2.0])],
            2,
            true,
            false,
        );
        assert_eq!(merged.ids, vec![2, 3]);
        assert_eq!(merged.distances, vec![1.0, 2.0]);
    }

    #[test]
    fn domain_index_order_is_ascending() {
        for mode in [
            "FLAT-L1",
            "HNSW-HAVERSINE",
            "FLAT-CORRELATION",
            "FLAT-HELLINGER",
            "FLAT-WASSERSTEIN",
            "FLAT-TANIMOTO-BINARY",
            "FLAT-DICE-BINARY",
        ] {
            assert!(is_ascending_index(Some(mode)), "{mode}");
        }
        assert!(!is_ascending_index(Some("FLAT-IP")));
    }

    #[test]
    fn decode_search_result_binary_roundtrips() {
        let fields = vec![HashMap::from([(
            "tag".to_string(),
            Value::String("x".into()),
        )])];
        let encoded = rpc::encode_search_result_binary(&[7], &[0.5], &fields);
        let (decoded, offset) = decode_search_result_binary(&encoded, 0).unwrap();
        assert_eq!(offset, encoded.len());
        assert_eq!(decoded.ids, vec![7]);
        assert_eq!(decoded.distances, vec![0.5]);
        assert_eq!(decoded.fields, fields);
    }

    #[test]
    fn normalize_rpc_meta_converts_query_strings() {
        let mut meta = serde_json::json!({
            "database_name": "db",
            "collection_name": "docs",
            "dim": "64",
            "n_queries": "2",
            "k": "10",
            "return_fields": "true",
            "eps": "0.5"
        });
        normalize_rpc_meta(&mut meta, Some("secret")).unwrap();
        assert_eq!(meta["dim"], serde_json::json!(64));
        assert_eq!(meta["n_queries"], serde_json::json!(2));
        assert_eq!(meta["return_fields"], serde_json::json!(true));
        assert_eq!(meta["eps"], serde_json::json!(0.5));
        assert_eq!(meta["api_key"], serde_json::json!("secret"));
    }

    #[test]
    fn derives_rpc_target_from_http_uri() {
        assert_eq!(
            derive_rpc_target("http://127.0.0.1:7638").unwrap(),
            ("127.0.0.1".to_string(), 17638)
        );
    }

    #[tokio::test]
    async fn rpc_request_reuses_idle_stream() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::net::TcpListener;

        let listener = loop {
            let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
            if listener.local_addr().unwrap().port() > 10_000 {
                break listener;
            }
        };
        let rpc_port = listener.local_addr().unwrap().port();
        let http_uri = format!("http://127.0.0.1:{}", rpc_port - 10_000);
        let accepted = Arc::new(AtomicUsize::new(0));
        let accepted_task = Arc::clone(&accepted);

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            accepted_task.fetch_add(1, Ordering::SeqCst);
            for _ in 0..2 {
                let mut header = [0u8; 4];
                stream.read_exact(&mut header).await.unwrap();
                let frame_len = u32::from_le_bytes(header) as usize;
                let mut frame = vec![0u8; frame_len];
                stream.read_exact(&mut frame).await.unwrap();

                let meta = br#"{"ok":true}"#;
                let mut response = Vec::new();
                response.push(0);
                response.extend_from_slice(&(meta.len() as u32).to_le_bytes());
                response.extend_from_slice(meta);
                stream
                    .write_all(&(response.len() as u32).to_le_bytes())
                    .await
                    .unwrap();
                stream.write_all(&response).await.unwrap();
            }
        });

        let coordinator = RustReadCoordinator::from_config(
            ClusterConfig {
                shard_groups: vec![ShardGroup {
                    name: "sg0".to_string(),
                    primary: http_uri.clone(),
                    replicas: Vec::new(),
                }],
                shards: Vec::new(),
                collections: HashMap::new(),
            },
            Duration::from_secs(2),
            None,
        )
        .unwrap();

        coordinator
            .rpc_request(&http_uri, OP_SEARCH, serde_json::json!({}), Vec::new())
            .await
            .unwrap();
        coordinator
            .rpc_request(&http_uri, OP_SEARCH, serde_json::json!({}), Vec::new())
            .await
            .unwrap();
        server.await.unwrap();

        assert_eq!(accepted.load(Ordering::SeqCst), 1);
    }
}
