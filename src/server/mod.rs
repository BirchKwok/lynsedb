//! HTTP server module for LynseDB.
//!
//! Provides a RESTful API using actix-web, replacing the Python Flask server.
//! All endpoints are compatible with the existing HTTPClient Python class.

use std::collections::{HashMap, HashSet};
use std::future::{ready, Future, Ready};
use std::io::Write;
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use actix_web::body::EitherBody;
use actix_web::dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform};
use actix_web::http::header::{HeaderMap, HeaderName, HeaderValue};
use actix_web::{web, App, HttpResponse, HttpServer};
use serde::{Deserialize, Serialize};

use crate::engine::DatabaseManager;
use crate::error::LynseError;

// ─── Shared application state ────────────────────────────────────────────────

struct AppState {
    manager: Arc<DatabaseManager>,
    start_time_unix_seconds: u64,
    metrics: Arc<HttpMetrics>,
    limits: ServerLimits,
}

const DEFAULT_SLOW_QUERY_WARN_MS: u64 = 1000;
const REQUEST_ID_HEADER: &str = "x-request-id";
static LOGGER_INIT: Once = Once::new();
const DEFAULT_MAX_TOP_K: usize = 10_000;
const DEFAULT_MAX_BATCH_VECTORS: usize = 100_000;
const DEFAULT_MAX_COLLECTION_VECTORS: u64 = 10_000_000;
const DEFAULT_MAX_COLLECTION_VECTOR_BYTES: u64 = 1_099_511_627_776; // 1 TiB

#[derive(Clone, Copy, Debug)]
struct ServerRuntimeConfig {
    workers: usize,
    keep_alive_secs: u64,
    client_request_timeout_secs: u64,
    json_limit_bytes: usize,
    payload_limit_bytes: usize,
    slow_query_warn_ms: u64,
    limits: ServerLimits,
    audit_log_enabled: bool,
}

#[derive(Clone, Copy, Debug)]
struct ServerLimits {
    /// 0 disables this guard.
    max_top_k: usize,
    /// 0 disables this guard.
    max_batch_vectors: usize,
    /// 0 disables this guard.
    max_collection_vectors: u64,
    /// 0 disables this guard.
    max_collection_vector_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
enum RequestOutcome {
    Normal,
    Unauthorized,
    HandlerFailure,
}

struct HttpMetrics {
    request_id_seq: AtomicU64,
    request_total: AtomicU64,
    request_error_total: AtomicU64,
    request_duration_sum_nanos: AtomicU64,
    status_2xx: AtomicU64,
    status_3xx: AtomicU64,
    status_4xx: AtomicU64,
    status_5xx: AtomicU64,
    status_other: AtomicU64,
    error_client_4xx_total: AtomicU64,
    error_server_5xx_total: AtomicU64,
    error_unauthorized_total: AtomicU64,
    error_handler_failure_total: AtomicU64,
    histogram_bucket_bounds: Vec<f64>,
    histogram_bucket_counts: Vec<AtomicU64>,
    index_build_started_total: AtomicU64,
    index_build_completed_total: AtomicU64,
    index_build_failed_total: AtomicU64,
    index_build_in_progress: AtomicU64,
    index_build_current_vectors: AtomicU64,
    index_build_last_vectors: AtomicU64,
    index_build_duration_sum_nanos: AtomicU64,
    index_build_last_duration_nanos: AtomicU64,
    index_build_current_progress_ppm: AtomicU64,
    index_build_last_progress_ppm: AtomicU64,
}

impl HttpMetrics {
    fn new() -> Self {
        // Bucket boundaries in seconds for Prometheus-style latency histogram.
        let bounds = vec![
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ];
        let mut bucket_counts = Vec::with_capacity(bounds.len() + 1);
        for _ in 0..=bounds.len() {
            bucket_counts.push(AtomicU64::new(0));
        }

        Self {
            request_id_seq: AtomicU64::new(0),
            request_total: AtomicU64::new(0),
            request_error_total: AtomicU64::new(0),
            request_duration_sum_nanos: AtomicU64::new(0),
            status_2xx: AtomicU64::new(0),
            status_3xx: AtomicU64::new(0),
            status_4xx: AtomicU64::new(0),
            status_5xx: AtomicU64::new(0),
            status_other: AtomicU64::new(0),
            error_client_4xx_total: AtomicU64::new(0),
            error_server_5xx_total: AtomicU64::new(0),
            error_unauthorized_total: AtomicU64::new(0),
            error_handler_failure_total: AtomicU64::new(0),
            histogram_bucket_bounds: bounds,
            histogram_bucket_counts: bucket_counts,
            index_build_started_total: AtomicU64::new(0),
            index_build_completed_total: AtomicU64::new(0),
            index_build_failed_total: AtomicU64::new(0),
            index_build_in_progress: AtomicU64::new(0),
            index_build_current_vectors: AtomicU64::new(0),
            index_build_last_vectors: AtomicU64::new(0),
            index_build_duration_sum_nanos: AtomicU64::new(0),
            index_build_last_duration_nanos: AtomicU64::new(0),
            index_build_current_progress_ppm: AtomicU64::new(0),
            index_build_last_progress_ppm: AtomicU64::new(0),
        }
    }

    fn next_request_id(&self) -> u64 {
        self.request_id_seq.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn observe(&self, status_code: u16, elapsed: Duration, outcome: RequestOutcome) {
        self.request_total.fetch_add(1, Ordering::Relaxed);

        let nanos_u64 = elapsed.as_nanos().min(u64::MAX as u128) as u64;
        self.request_duration_sum_nanos
            .fetch_add(nanos_u64, Ordering::Relaxed);

        match status_code {
            200..=299 => {
                self.status_2xx.fetch_add(1, Ordering::Relaxed);
            }
            300..=399 => {
                self.status_3xx.fetch_add(1, Ordering::Relaxed);
            }
            400..=499 => {
                self.status_4xx.fetch_add(1, Ordering::Relaxed);
            }
            500..=599 => {
                self.status_5xx.fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                self.status_other.fetch_add(1, Ordering::Relaxed);
            }
        }

        if status_code >= 400 {
            self.request_error_total.fetch_add(1, Ordering::Relaxed);
        }
        if (400..=499).contains(&status_code) {
            self.error_client_4xx_total.fetch_add(1, Ordering::Relaxed);
        }
        if (500..=599).contains(&status_code) {
            self.error_server_5xx_total.fetch_add(1, Ordering::Relaxed);
        }

        match outcome {
            RequestOutcome::Normal => {}
            RequestOutcome::Unauthorized => {
                self.error_unauthorized_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            RequestOutcome::HandlerFailure => {
                self.error_handler_failure_total
                    .fetch_add(1, Ordering::Relaxed);
            }
        }

        let elapsed_secs = elapsed.as_secs_f64();
        let mut bucket_index = self.histogram_bucket_bounds.len();
        for (idx, bound) in self.histogram_bucket_bounds.iter().enumerate() {
            if elapsed_secs <= *bound {
                bucket_index = idx;
                break;
            }
        }
        self.histogram_bucket_counts[bucket_index].fetch_add(1, Ordering::Relaxed);
    }

    fn request_total(&self) -> u64 {
        self.request_total.load(Ordering::Relaxed)
    }

    fn request_duration_sum_seconds(&self) -> f64 {
        self.request_duration_sum_nanos.load(Ordering::Relaxed) as f64 / 1_000_000_000.0
    }

    fn request_latency_quantile_seconds(&self, quantile: f64) -> f64 {
        if !(0.0..=1.0).contains(&quantile) {
            return 0.0;
        }

        let total = self.request_total();
        if total == 0 {
            return 0.0;
        }

        let target_rank = (quantile * total as f64).ceil().max(1.0);
        let mut cumulative = 0u64;

        for idx in 0..self.histogram_bucket_counts.len() {
            let bucket_count = self.histogram_bucket_counts[idx].load(Ordering::Relaxed);
            cumulative += bucket_count;

            if (cumulative as f64) < target_rank {
                continue;
            }

            let lower_bound = if idx == 0 {
                0.0
            } else {
                self.histogram_bucket_bounds[idx - 1]
            };

            if idx >= self.histogram_bucket_bounds.len() {
                return lower_bound;
            }

            let upper_bound = self.histogram_bucket_bounds[idx];
            if bucket_count == 0 {
                return upper_bound;
            }

            let prev_cumulative = cumulative - bucket_count;
            let in_bucket_rank =
                (target_rank - prev_cumulative as f64).clamp(0.0, bucket_count as f64);
            let fraction = in_bucket_rank / bucket_count as f64;
            return lower_bound + (upper_bound - lower_bound) * fraction;
        }

        self.histogram_bucket_bounds.last().copied().unwrap_or(0.0)
    }

    fn track_index_build<T, E, F>(&self, total_vectors: u64, build: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        self.index_build_started_total
            .fetch_add(1, Ordering::Relaxed);
        self.index_build_in_progress.fetch_add(1, Ordering::Relaxed);
        self.index_build_current_vectors
            .fetch_add(total_vectors, Ordering::Relaxed);
        self.index_build_current_progress_ppm
            .store(0, Ordering::Relaxed);

        let started = Instant::now();
        let result = build();
        let elapsed = started.elapsed();
        let elapsed_nanos = elapsed.as_nanos().min(u64::MAX as u128) as u64;

        atomic_saturating_sub(&self.index_build_in_progress, 1);
        atomic_saturating_sub(&self.index_build_current_vectors, total_vectors);
        self.index_build_last_vectors
            .store(total_vectors, Ordering::Relaxed);
        self.index_build_duration_sum_nanos
            .fetch_add(elapsed_nanos, Ordering::Relaxed);
        self.index_build_last_duration_nanos
            .store(elapsed_nanos, Ordering::Relaxed);

        if result.is_ok() {
            self.index_build_completed_total
                .fetch_add(1, Ordering::Relaxed);
            self.index_build_current_progress_ppm
                .store(0, Ordering::Relaxed);
            self.index_build_last_progress_ppm
                .store(1_000_000, Ordering::Relaxed);
        } else {
            self.index_build_failed_total
                .fetch_add(1, Ordering::Relaxed);
            self.index_build_current_progress_ppm
                .store(0, Ordering::Relaxed);
            self.index_build_last_progress_ppm
                .store(0, Ordering::Relaxed);
        }

        result
    }

    fn index_build_duration_sum_seconds(&self) -> f64 {
        self.index_build_duration_sum_nanos.load(Ordering::Relaxed) as f64 / 1_000_000_000.0
    }

    fn index_build_last_duration_seconds(&self) -> f64 {
        self.index_build_last_duration_nanos.load(Ordering::Relaxed) as f64 / 1_000_000_000.0
    }
}

fn atomic_saturating_sub(counter: &AtomicU64, amount: u64) {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        let next = current.saturating_sub(amount);
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(value) => current = value,
        }
    }
}

fn parse_positive_usize_env(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    match raw.parse::<usize>() {
        Ok(v) if v > 0 => Some(v),
        _ => {
            log::warn!(
                "Ignoring invalid {}='{}' (must be a positive integer)",
                name,
                raw
            );
            None
        }
    }
}

fn parse_u64_env(name: &str) -> Option<u64> {
    let raw = std::env::var(name).ok()?;
    match raw.parse::<u64>() {
        Ok(v) => Some(v),
        _ => {
            log::warn!(
                "Ignoring invalid {}='{}' (must be a non-negative integer)",
                name,
                raw
            );
            None
        }
    }
}

fn parse_bool_env(name: &str) -> Option<bool> {
    let raw = std::env::var(name).ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => {
            log::warn!(
                "Ignoring invalid {}='{}' (must be true/false or 1/0)",
                name,
                raw
            );
            None
        }
    }
}

fn load_server_runtime_config() -> ServerRuntimeConfig {
    let workers =
        parse_positive_usize_env("LYNSE_SERVER_WORKERS").unwrap_or_else(|| num_cpus::get().max(2));
    let keep_alive_secs = parse_positive_usize_env("LYNSE_KEEP_ALIVE_SECS")
        .map(|v| v as u64)
        .unwrap_or(75);
    let client_request_timeout_secs = parse_positive_usize_env("LYNSE_CLIENT_REQUEST_TIMEOUT_SECS")
        .map(|v| v as u64)
        .unwrap_or(300);

    let json_limit_mb = parse_positive_usize_env("LYNSE_JSON_LIMIT_MB").unwrap_or(256);
    let payload_limit_mb = parse_positive_usize_env("LYNSE_PAYLOAD_LIMIT_MB").unwrap_or(512);
    let slow_query_warn_ms =
        parse_u64_env("LYNSE_SLOW_QUERY_WARN_MS").unwrap_or(DEFAULT_SLOW_QUERY_WARN_MS);
    let limits = ServerLimits {
        max_top_k: parse_u64_env("LYNSE_MAX_TOP_K")
            .map(|v| v.min(usize::MAX as u64) as usize)
            .unwrap_or(DEFAULT_MAX_TOP_K),
        max_batch_vectors: parse_u64_env("LYNSE_MAX_BATCH_VECTORS")
            .map(|v| v.min(usize::MAX as u64) as usize)
            .unwrap_or(DEFAULT_MAX_BATCH_VECTORS),
        max_collection_vectors: parse_u64_env("LYNSE_MAX_COLLECTION_VECTORS")
            .unwrap_or(DEFAULT_MAX_COLLECTION_VECTORS),
        max_collection_vector_bytes: parse_u64_env("LYNSE_MAX_COLLECTION_VECTOR_BYTES")
            .unwrap_or(DEFAULT_MAX_COLLECTION_VECTOR_BYTES),
    };
    let audit_log_enabled = parse_bool_env("LYNSE_AUDIT_LOG").unwrap_or(true);

    ServerRuntimeConfig {
        workers,
        keep_alive_secs,
        client_request_timeout_secs,
        json_limit_bytes: json_limit_mb.saturating_mul(1024 * 1024),
        payload_limit_bytes: payload_limit_mb.saturating_mul(1024 * 1024),
        slow_query_warn_ms,
        limits,
        audit_log_enabled,
    }
}

fn checked_vector_bytes(n_vectors: usize, dim: usize) -> Result<u64, LynseError> {
    let floats = n_vectors
        .checked_mul(dim)
        .ok_or_else(|| LynseError::InvalidArgument("vector count × dimension overflows".into()))?;
    let bytes = floats
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| LynseError::InvalidArgument("vector byte estimate overflows".into()))?;
    Ok(bytes as u64)
}

fn validate_top_k(limits: &ServerLimits, value: usize, field: &str) -> Result<(), LynseError> {
    if limits.max_top_k > 0 && value > limits.max_top_k {
        return Err(LynseError::InvalidArgument(format!(
            "{} {} exceeds server max_top_k {}",
            field, value, limits.max_top_k
        )));
    }
    Ok(())
}

fn validate_batch_vectors(
    limits: &ServerLimits,
    n_vectors: usize,
    field: &str,
) -> Result<(), LynseError> {
    if limits.max_batch_vectors > 0 && n_vectors > limits.max_batch_vectors {
        return Err(LynseError::InvalidArgument(format!(
            "{} {} exceeds server max_batch_vectors {}",
            field, n_vectors, limits.max_batch_vectors
        )));
    }
    Ok(())
}

fn validate_request_vector_bytes(
    limits: &ServerLimits,
    n_vectors: usize,
    dim: usize,
    field: &str,
) -> Result<(), LynseError> {
    let bytes = checked_vector_bytes(n_vectors, dim)?;
    if limits.max_collection_vector_bytes > 0 && bytes > limits.max_collection_vector_bytes {
        return Err(LynseError::InvalidArgument(format!(
            "{} estimated vector bytes {} exceed server max_collection_vector_bytes {}",
            field, bytes, limits.max_collection_vector_bytes
        )));
    }
    Ok(())
}

fn validate_collection_vector_count(
    limits: &ServerLimits,
    current_vectors: u64,
    additional_vectors: u64,
) -> Result<(), LynseError> {
    if limits.max_collection_vectors == 0 {
        return Ok(());
    }

    let target = current_vectors
        .checked_add(additional_vectors)
        .ok_or_else(|| LynseError::InvalidArgument("collection vector count overflows".into()))?;
    if target > limits.max_collection_vectors {
        return Err(LynseError::InvalidArgument(format!(
            "collection vector count {} would exceed server max_collection_vectors {}",
            target, limits.max_collection_vectors
        )));
    }
    Ok(())
}

fn validate_collection_vector_bytes(
    limits: &ServerLimits,
    current_bytes: u64,
    additional_bytes: u64,
) -> Result<(), LynseError> {
    if limits.max_collection_vector_bytes == 0 {
        return Ok(());
    }

    let target = current_bytes
        .checked_add(additional_bytes)
        .ok_or_else(|| LynseError::InvalidArgument("collection vector bytes overflow".into()))?;
    if target > limits.max_collection_vector_bytes {
        return Err(LynseError::InvalidArgument(format!(
            "collection vector bytes {} would exceed server max_collection_vector_bytes {}",
            target, limits.max_collection_vector_bytes
        )));
    }
    Ok(())
}

fn validate_collection_insert(
    limits: &ServerLimits,
    coll: &crate::engine::Collection,
    additional_vectors: u64,
    additional_vector_bytes: u64,
) -> Result<(), LynseError> {
    let (current_vectors, _) = coll.shape()?;
    validate_collection_vector_count(limits, current_vectors, additional_vectors)?;
    validate_collection_vector_bytes(
        limits,
        coll.estimated_vector_bytes()?,
        additional_vector_bytes,
    )
}

#[derive(Debug, Default)]
struct StorageUsage {
    disk_bytes: u64,
    wal_bytes: u64,
    index_bytes: u64,
}

fn collect_storage_usage(root: &Path) -> StorageUsage {
    let mut usage = StorageUsage::default();
    collect_storage_usage_inner(root, &mut usage);
    usage
}

fn collect_storage_usage_inner(path: &Path, usage: &mut StorageUsage) {
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return;
    };

    if metadata.is_file() {
        let len = metadata.len();
        usage.disk_bytes = usage.disk_bytes.saturating_add(len);

        if path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.ends_with(".wal"))
            .unwrap_or(false)
        {
            usage.wal_bytes = usage.wal_bytes.saturating_add(len);
        }
        if is_index_file(path) {
            usage.index_bytes = usage.index_bytes.saturating_add(len);
        }
        return;
    }

    if metadata.is_dir() {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            collect_storage_usage_inner(&entry.path(), usage);
        }
    }
}

fn is_index_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };

    matches!(
        name,
        "pq_index.bin" | "rabitq_index.bin" | "polarvec_index.bin" | "index_metadata.json"
    ) || (name.starts_with("index-") && name.ends_with(".bin"))
}

#[cfg(target_os = "linux")]
fn process_resident_memory_bytes() -> u64 {
    if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
        if let Some(rss_pages) = statm
            .split_whitespace()
            .nth(1)
            .and_then(|v| v.parse::<u64>().ok())
        {
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if page_size > 0 {
                return rss_pages.saturating_mul(page_size as u64);
            }
        }
    }
    process_resident_memory_bytes_from_rusage()
}

#[cfg(all(unix, not(target_os = "linux")))]
fn process_resident_memory_bytes() -> u64 {
    process_resident_memory_bytes_from_rusage()
}

#[cfg(unix)]
fn process_resident_memory_bytes_from_rusage() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return 0;
    }
    let usage = unsafe { usage.assume_init() };
    let raw = usage.ru_maxrss.max(0) as u64;
    #[cfg(target_os = "macos")]
    {
        raw
    }
    #[cfg(not(target_os = "macos"))]
    {
        raw.saturating_mul(1024)
    }
}

#[cfg(not(unix))]
fn process_resident_memory_bytes() -> u64 {
    0
}

fn request_elapsed_ms(elapsed: Duration) -> f64 {
    elapsed.as_secs_f64() * 1000.0
}

fn is_query_endpoint(path: &str) -> bool {
    matches!(
        path,
        "/search"
            | "/search_profile"
            | "/text_search"
            | "/sparse_search"
            | "/hybrid_search"
            | "/search_binary"
            | "/batch_search"
            | "/batch_search_binary"
            | "/query"
            | "/query_vectors"
            | "/search_range"
    )
}

fn audit_action_for_path(path: &str) -> Option<&'static str> {
    match path {
        "/create_database" => Some("create_database"),
        "/drop_database" | "/delete_database" => Some("drop_database"),
        "/snapshot_database" => Some("snapshot_database"),
        "/restore_database" => Some("restore_database"),
        "/required_collection" => Some("require_collection"),
        "/drop_collection" => Some("drop_collection"),
        "/snapshot_collection" => Some("snapshot_collection"),
        "/export_collection" => Some("export_collection"),
        "/restore_collection" => Some("restore_collection"),
        "/import_collection" => Some("import_collection"),
        "/add_item" => Some("add_item"),
        "/bulk_add_items" => Some("bulk_add_items"),
        "/upsert_items" => Some("upsert_items"),
        "/bulk_add_binary" => Some("bulk_add_binary"),
        "/create_vector_field" => Some("create_vector_field"),
        "/add_named_vectors" => Some("add_named_vectors"),
        "/add_sparse_vectors" => Some("add_sparse_vectors"),
        "/build_vector_field_index" => Some("build_vector_field_index"),
        "/remove_vector_field_index" => Some("remove_vector_field_index"),
        "/commit" => Some("commit"),
        "/flush" => Some("flush"),
        "/checkpoint" => Some("checkpoint"),
        "/close" => Some("close_collection"),
        "/update_collection_description" | "/update_description" => Some("update_description"),
        "/build_index" => Some("build_index"),
        "/remove_index" => Some("remove_index"),
        "/delete_items" => Some("delete_items"),
        "/restore_items" => Some("restore_items"),
        "/compact" => Some("compact"),
        _ => None,
    }
}

fn log_request_event(
    request_id: u64,
    method: &str,
    path: &str,
    status_code: u16,
    elapsed: Duration,
    outcome: &str,
) {
    log::info!(
        "{}",
        serde_json::json!({
            "event": "http_request",
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "elapsed_ms": request_elapsed_ms(elapsed),
            "outcome": outcome,
        })
    );
}

fn attach_request_id_header(headers: &mut HeaderMap, request_id: u64) {
    if let Ok(value) = HeaderValue::from_str(&request_id.to_string()) {
        headers.insert(HeaderName::from_static(REQUEST_ID_HEADER), value);
    }
}

fn maybe_log_audit_event(
    audit_log_enabled: bool,
    request_id: u64,
    method: &str,
    path: &str,
    status_code: u16,
    outcome: &str,
) {
    if !audit_log_enabled {
        return;
    }

    let action = audit_action_for_path(path).or_else(|| {
        if outcome == "unauthorized" && !is_public_endpoint(path) {
            Some("unauthorized_request")
        } else {
            None
        }
    });
    let Some(action) = action else {
        return;
    };

    log::info!(
        "{}",
        serde_json::json!({
            "event": "audit",
            "request_id": request_id,
            "action": action,
            "method": method,
            "path": path,
            "status_code": status_code,
            "outcome": outcome,
        })
    );
}

fn maybe_warn_slow_query(
    request_id: u64,
    method: &str,
    path: &str,
    status_code: u16,
    elapsed: Duration,
    threshold_ms: u64,
) {
    if threshold_ms == 0 || !is_query_endpoint(path) {
        return;
    }

    let elapsed_ms = request_elapsed_ms(elapsed);
    if elapsed_ms < threshold_ms as f64 {
        return;
    }

    log::warn!(
        "{}",
        serde_json::json!({
            "event": "slow_query",
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "elapsed_ms": elapsed_ms,
            "threshold_ms": threshold_ms,
        })
    );
}

fn init_rust_logger() {
    LOGGER_INIT.call_once(|| {
        let default_filter = std::env::var("LYNSE_LOG_LEVEL")
            .unwrap_or_else(|_| "info".to_string())
            .to_lowercase();
        let env = env_logger::Env::default().default_filter_or(default_filter);
        let _ = env_logger::Builder::from_env(env)
            .format(|buf, record| {
                let mut event = serde_json::Map::new();
                event.insert(
                    "timestamp".to_string(),
                    serde_json::json!(buf.timestamp_millis().to_string()),
                );
                event.insert(
                    "level".to_string(),
                    serde_json::json!(record.level().to_string()),
                );
                event.insert("target".to_string(), serde_json::json!(record.target()));

                let message = record.args().to_string();
                match serde_json::from_str::<serde_json::Value>(&message) {
                    Ok(serde_json::Value::Object(fields)) => {
                        for (key, value) in fields {
                            event.insert(key, value);
                        }
                    }
                    _ => {
                        event.insert("message".to_string(), serde_json::json!(message));
                    }
                }

                writeln!(buf, "{}", serde_json::Value::Object(event))
            })
            .try_init();
    });
}

// ─── Basic Auth / Bearer Token Middleware ─────────────────────────────────────

/// Middleware factory. Wraps every request with an optional API key check.
/// Accepts both `Authorization: Bearer <key>` and `Authorization: Basic base64(:key)` headers.
struct ApiKeyAuth {
    api_key: Option<String>,
    metrics: Arc<HttpMetrics>,
    slow_query_warn_ms: u64,
    audit_log_enabled: bool,
}

impl<S, B> Transform<S, ServiceRequest> for ApiKeyAuth
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = actix_web::Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<EitherBody<B>>;
    type Error = actix_web::Error;
    type InitError = ();
    type Transform = ApiKeyAuthMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(ApiKeyAuthMiddleware {
            service,
            api_key: self.api_key.clone(),
            metrics: Arc::clone(&self.metrics),
            slow_query_warn_ms: self.slow_query_warn_ms,
            audit_log_enabled: self.audit_log_enabled,
        }))
    }
}

struct ApiKeyAuthMiddleware<S> {
    service: S,
    api_key: Option<String>,
    metrics: Arc<HttpMetrics>,
    slow_query_warn_ms: u64,
    audit_log_enabled: bool,
}

fn decode_basic(b64: &str) -> Option<String> {
    use base64::{engine::general_purpose::STANDARD, Engine};
    let bytes = STANDARD.decode(b64).ok()?;
    let s = String::from_utf8(bytes).ok()?;
    // Format is "username:password" — take only the password part
    s.splitn(2, ':').nth(1).map(|p| p.to_string())
}

fn is_authorized(req: &ServiceRequest, key: &str) -> bool {
    req.headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            if let Some(bearer) = v.strip_prefix("Bearer ") {
                bearer == key
            } else if let Some(b64) = v.strip_prefix("Basic ") {
                decode_basic(b64).as_deref() == Some(key)
            } else {
                false
            }
        })
        .unwrap_or(false)
}

fn is_public_endpoint(path: &str) -> bool {
    matches!(path, "/" | "/healthz" | "/readyz")
}

impl<S, B> Service<ServiceRequest> for ApiKeyAuthMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = actix_web::Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<EitherBody<B>>;
    type Error = actix_web::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let started = Instant::now();
        let request_id = self.metrics.next_request_id();
        let method = req.method().as_str().to_string();
        let path = req.path().to_string();
        let slow_query_warn_ms = self.slow_query_warn_ms;
        let audit_log_enabled = self.audit_log_enabled;
        if let Some(ref key) = self.api_key {
            if !is_public_endpoint(req.path()) && !is_authorized(&req, key) {
                let metrics = Arc::clone(&self.metrics);
                return Box::pin(async move {
                    let mut response = HttpResponse::Unauthorized()
                        .insert_header(("WWW-Authenticate", "Basic realm=\"LynseDB\""))
                        .json(serde_json::json!({"error": "Unauthorized"}));
                    attach_request_id_header(response.headers_mut(), request_id);
                    let elapsed = started.elapsed();
                    metrics.observe(401, elapsed, RequestOutcome::Unauthorized);
                    log_request_event(request_id, &method, &path, 401, elapsed, "unauthorized");
                    maybe_log_audit_event(
                        audit_log_enabled,
                        request_id,
                        &method,
                        &path,
                        401,
                        "unauthorized",
                    );
                    maybe_warn_slow_query(
                        request_id,
                        &method,
                        &path,
                        401,
                        elapsed,
                        slow_query_warn_ms,
                    );
                    Ok(req.into_response(response.map_into_right_body()))
                });
            }
        }

        let metrics = Arc::clone(&self.metrics);
        let fut = self.service.call(req);
        Box::pin(async move {
            match fut.await {
                Ok(res) => {
                    let mut res = res;
                    let status_code = res.status().as_u16();
                    let elapsed = started.elapsed();
                    metrics.observe(status_code, elapsed, RequestOutcome::Normal);
                    attach_request_id_header(res.headers_mut(), request_id);
                    log_request_event(request_id, &method, &path, status_code, elapsed, "normal");
                    maybe_log_audit_event(
                        audit_log_enabled,
                        request_id,
                        &method,
                        &path,
                        status_code,
                        "normal",
                    );
                    maybe_warn_slow_query(
                        request_id,
                        &method,
                        &path,
                        status_code,
                        elapsed,
                        slow_query_warn_ms,
                    );
                    Ok(res.map_into_left_body())
                }
                Err(err) => {
                    let elapsed = started.elapsed();
                    metrics.observe(500, elapsed, RequestOutcome::HandlerFailure);
                    log_request_event(request_id, &method, &path, 500, elapsed, "handler_failure");
                    maybe_log_audit_event(
                        audit_log_enabled,
                        request_id,
                        &method,
                        &path,
                        500,
                        "handler_failure",
                    );
                    maybe_warn_slow_query(
                        request_id,
                        &method,
                        &path,
                        500,
                        elapsed,
                        slow_query_warn_ms,
                    );
                    Err(err)
                }
            }
        })
    }
}

// ─── Request/Response types ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct DatabaseRequest {
    database_name: Option<String>,
}

#[derive(Deserialize)]
struct CreateDatabaseRequest {
    database_name: String,
    drop_if_exists: Option<bool>,
}

#[derive(Deserialize)]
struct DropDatabaseRequest {
    database_name: String,
}

#[derive(Deserialize)]
struct DatabaseExistsRequest {
    database_name: String,
}

#[derive(Deserialize)]
struct SnapshotDatabaseRequest {
    database_name: String,
    snapshot_path: String,
}

#[derive(Deserialize)]
struct RestoreDatabaseRequest {
    database_name: String,
    snapshot_path: String,
    overwrite: Option<bool>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct RequireCollectionRequest {
    database_name: String,
    collection_name: String,
    dim: usize,
    drop_if_exists: Option<bool>,
    description: Option<String>,
    // Accepted for backward API compat but ignored
    #[allow(dead_code)]
    n_threads: Option<usize>,
    #[allow(dead_code)]
    warm_up: Option<bool>,
}

#[derive(Deserialize)]
struct CollectionRequest {
    database_name: String,
    collection_name: String,
}

#[derive(Deserialize)]
struct SnapshotCollectionRequest {
    database_name: String,
    collection_name: String,
    snapshot_path: String,
}

#[derive(Deserialize)]
struct ExportCollectionRequest {
    database_name: String,
    collection_name: String,
    export_path: String,
}

#[derive(Deserialize)]
struct RestoreCollectionRequest {
    database_name: String,
    collection_name: String,
    snapshot_path: String,
    overwrite: Option<bool>,
}

#[derive(Deserialize)]
struct ImportCollectionRequest {
    database_name: String,
    collection_name: String,
    export_path: String,
    overwrite: Option<bool>,
}

#[derive(Deserialize)]
struct AddItemRequest {
    database_name: String,
    collection_name: String,
    item: AddItemData,
}

#[derive(Deserialize)]
struct AddItemData {
    vector: Vec<f32>,
    id: Option<u64>,
    field: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Deserialize)]
struct BulkAddItemsRequest {
    database_name: String,
    collection_name: String,
    items: Vec<BulkItemData>,
}

#[derive(Deserialize)]
struct BulkItemData {
    vector: Vec<f32>,
    id: Option<u64>,
    field: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Deserialize)]
struct CreateVectorFieldRequest {
    database_name: String,
    collection_name: String,
    field_name: String,
    dim: usize,
    metric: Option<String>,
    index_mode: Option<String>,
}

#[derive(Deserialize)]
struct AddNamedVectorsRequest {
    database_name: String,
    collection_name: String,
    field_name: String,
    vectors: Vec<Vec<f32>>,
    ids: Vec<u64>,
}

#[derive(Deserialize)]
struct SparseVectorData {
    indices: Vec<u32>,
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct AddSparseVectorsRequest {
    database_name: String,
    collection_name: String,
    vectors: Vec<SparseVectorData>,
    ids: Vec<u64>,
}

#[derive(Deserialize)]
struct BuildVectorFieldIndexRequest {
    database_name: String,
    collection_name: String,
    field_name: String,
    index_mode: Option<String>,
}

#[derive(Deserialize)]
struct SearchRequest {
    database_name: String,
    collection_name: String,
    vector: Vec<f32>,
    vector_field: Option<String>,
    k: Option<usize>,
    #[serde(rename = "where")]
    where_expr: Option<String>,
    return_fields: Option<bool>,
    nprobe: Option<usize>,
    approx: Option<bool>,
    eps: Option<f32>,
}

#[derive(Deserialize)]
struct SearchProfileRequest {
    database_name: String,
    collection_name: String,
    vector: Vec<f32>,
    k: Option<usize>,
    #[serde(rename = "where")]
    where_expr: Option<String>,
    return_fields: Option<bool>,
    nprobe: Option<usize>,
    approx: Option<bool>,
    eps: Option<f32>,
}

#[derive(Deserialize)]
struct TextSearchRequest {
    database_name: String,
    collection_name: String,
    text: String,
    text_fields: Option<Vec<String>>,
    k: Option<usize>,
    #[serde(rename = "where")]
    where_expr: Option<String>,
    return_fields: Option<bool>,
}

#[derive(Deserialize)]
struct SparseSearchRequest {
    database_name: String,
    collection_name: String,
    vector: SparseVectorData,
    k: Option<usize>,
    #[serde(rename = "where")]
    where_expr: Option<String>,
    return_fields: Option<bool>,
}

#[derive(Deserialize)]
struct HybridSearchRequest {
    database_name: String,
    collection_name: String,
    vector: Option<Vec<f32>>,
    text: Option<String>,
    text_fields: Option<Vec<String>>,
    k: Option<usize>,
    #[serde(rename = "where")]
    where_expr: Option<String>,
    fusion: Option<String>,
    vector_weight: Option<f32>,
    text_weight: Option<f32>,
    rrf_k: Option<f32>,
    candidate_limit: Option<usize>,
    return_fields: Option<bool>,
    nprobe: Option<usize>,
}

#[derive(Deserialize)]
struct BatchSearchRequest {
    database_name: String,
    collection_name: String,
    vectors: Vec<Vec<f32>>,
    k: Option<usize>,
    #[serde(rename = "where")]
    where_expr: Option<String>,
    return_fields: Option<bool>,
    nprobe: Option<usize>,
}

#[derive(Deserialize)]
struct CommitRequest {
    database_name: String,
    collection_name: String,
    items: Option<Vec<BulkItemData>>,
}

#[derive(Deserialize)]
struct BuildIndexRequest {
    database_name: String,
    collection_name: String,
    index_mode: Option<String>,
}

#[derive(Deserialize)]
struct HeadTailRequest {
    database_name: String,
    collection_name: String,
    n: Option<usize>,
}

#[derive(Deserialize)]
struct QueryRequest {
    database_name: String,
    collection_name: String,
    #[serde(rename = "where")]
    where_expr: Option<String>,
    filter_ids: Option<Vec<u64>>,
    return_ids_only: Option<bool>,
}

#[derive(Deserialize)]
struct QueryVectorsRequest {
    database_name: String,
    collection_name: String,
    #[serde(rename = "where")]
    where_expr: Option<String>,
    filter_ids: Option<Vec<u64>>,
}

#[derive(Deserialize)]
struct DeleteItemsRequest {
    database_name: String,
    collection_name: String,
    ids: Vec<u64>,
}

#[derive(Deserialize)]
struct SearchRangeRequest {
    database_name: String,
    collection_name: String,
    vector: Vec<f32>,
    threshold: f32,
    #[serde(default = "default_max_results")]
    max_results: usize,
}

fn default_max_results() -> usize {
    1000
}

#[derive(Deserialize)]
struct UpdateDescriptionRequest {
    database_name: String,
    collection_name: String,
    description: String,
}

#[derive(Deserialize)]
struct BulkAddBinaryQuery {
    database_name: String,
    collection_name: String,
    dim: usize,
    n_vectors: usize,
}

#[derive(Deserialize)]
struct SearchBinaryQuery {
    database_name: String,
    collection_name: String,
    dim: usize,
    vector_field: Option<String>,
    k: Option<usize>,
    #[serde(rename = "where")]
    where_expr: Option<String>,
    return_fields: Option<bool>,
    nprobe: Option<usize>,
    approx: Option<bool>,
    eps: Option<f32>,
}

#[derive(Deserialize)]
struct BatchSearchBinaryQuery {
    database_name: String,
    collection_name: String,
    dim: usize,
    n_queries: usize,
    k: Option<usize>,
    #[serde(rename = "where")]
    where_expr: Option<String>,
    return_fields: Option<bool>,
    nprobe: Option<usize>,
}

#[derive(Deserialize)]
struct HeadTailBinaryQuery {
    database_name: String,
    collection_name: String,
    n: Option<usize>,
}

#[derive(Deserialize)]
struct IsIdExistsRequest {
    database_name: String,
    collection_name: String,
    id: u64,
}

#[derive(Deserialize)]
struct MaxIdRequest {
    database_name: String,
    collection_name: String,
}

#[derive(Deserialize)]
struct ReadByIdRequest {
    database_name: String,
    collection_name: String,
    id: serde_json::Value, // can be int or list of ints
}

#[derive(Serialize)]
struct ApiResponse<T: Serialize> {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    fn success(params: T) -> HttpResponse {
        HttpResponse::Ok().json(ApiResponse {
            status: "success".to_string(),
            params: Some(params),
            error: None,
        })
    }
}

fn error_response(msg: &str) -> HttpResponse {
    HttpResponse::InternalServerError().json(serde_json::json!({"error": msg}))
}

fn bad_request(msg: &str) -> HttpResponse {
    HttpResponse::BadRequest().json(serde_json::json!({"error": msg}))
}

fn limit_bad_request(err: LynseError) -> HttpResponse {
    bad_request(&err.to_string())
}

// ─── Helper: get collection from manager ─────────────────────────────────────

fn with_collection<F, T>(
    manager: &DatabaseManager,
    db_name: &str,
    coll_name: &str,
    f: F,
) -> Result<T, LynseError>
where
    F: FnOnce(&crate::engine::Collection) -> Result<T, LynseError>,
{
    manager.get_or_open_database(db_name)?;
    manager.with_database(db_name, |engine| {
        let coll_arc = engine.get_or_open_collection(coll_name, 0, 100_000)?;
        let coll = coll_arc.read();
        f(&coll)
    })
}

fn with_collection_mut<F, T>(
    manager: &DatabaseManager,
    db_name: &str,
    coll_name: &str,
    f: F,
) -> Result<T, LynseError>
where
    F: FnOnce(&mut crate::engine::Collection) -> Result<T, LynseError>,
{
    manager.get_or_open_database(db_name)?;
    manager.with_database(db_name, |engine| {
        let coll_arc = engine.get_or_open_collection(coll_name, 0, 100_000)?;
        let mut coll = coll_arc.write();
        f(&mut coll)
    })
}

fn sparse_vector_entries(vector: &SparseVectorData) -> Result<Vec<(u32, f32)>, LynseError> {
    if vector.indices.len() != vector.values.len() {
        return Err(LynseError::InvalidArgument(format!(
            "sparse vector has {} indices but {} values",
            vector.indices.len(),
            vector.values.len()
        )));
    }
    Ok(vector
        .indices
        .iter()
        .copied()
        .zip(vector.values.iter().copied())
        .collect())
}

// ─── Database operation handlers ─────────────────────────────────────────────

async fn index() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "success",
        "message": "LynseDB HTTP API"
    }))
}

async fn healthz() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "ok"
    }))
}

async fn readyz(state: web::Data<AppState>) -> HttpResponse {
    let root_exists = state.manager.root_path().exists();
    let ready = root_exists;
    if ready {
        HttpResponse::Ok().json(serde_json::json!({
            "status": "ready",
            "root_path": state.manager.root_path().to_string_lossy().to_string()
        }))
    } else {
        HttpResponse::ServiceUnavailable().json(serde_json::json!({
            "status": "not_ready",
            "root_path": state.manager.root_path().to_string_lossy().to_string()
        }))
    }
}

#[derive(Clone, Copy)]
struct OpenApiRoute {
    method: &'static str,
    path: &'static str,
    tag: &'static str,
    summary: &'static str,
    request_schema: Option<&'static str>,
}

fn openapi_routes() -> &'static [OpenApiRoute] {
    &[
        OpenApiRoute {
            method: "get",
            path: "/",
            tag: "system",
            summary: "API root",
            request_schema: None,
        },
        OpenApiRoute {
            method: "get",
            path: "/healthz",
            tag: "system",
            summary: "Liveness probe",
            request_schema: None,
        },
        OpenApiRoute {
            method: "get",
            path: "/readyz",
            tag: "system",
            summary: "Readiness probe",
            request_schema: None,
        },
        OpenApiRoute {
            method: "get",
            path: "/metrics",
            tag: "system",
            summary: "Prometheus metrics",
            request_schema: None,
        },
        OpenApiRoute {
            method: "get",
            path: "/openapi.json",
            tag: "system",
            summary: "OpenAPI specification",
            request_schema: None,
        },
        OpenApiRoute {
            method: "post",
            path: "/create_database",
            tag: "database",
            summary: "Create database",
            request_schema: Some("CreateDatabaseRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/drop_database",
            tag: "database",
            summary: "Drop database",
            request_schema: Some("DropDatabaseRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/snapshot_database",
            tag: "database",
            summary: "Snapshot database",
            request_schema: Some("SnapshotDatabaseRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/restore_database",
            tag: "database",
            summary: "Restore database",
            request_schema: Some("RestoreDatabaseRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/database_exists",
            tag: "database",
            summary: "Check database existence",
            request_schema: Some("DatabaseExistsRequest"),
        },
        OpenApiRoute {
            method: "get",
            path: "/list_databases",
            tag: "database",
            summary: "List databases",
            request_schema: None,
        },
        OpenApiRoute {
            method: "post",
            path: "/delete_database",
            tag: "database",
            summary: "Delete database",
            request_schema: Some("DropDatabaseRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/required_collection",
            tag: "collection",
            summary: "Create or require collection",
            request_schema: Some("RequireCollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/drop_collection",
            tag: "collection",
            summary: "Drop collection",
            request_schema: Some("CollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/snapshot_collection",
            tag: "collection",
            summary: "Snapshot collection",
            request_schema: Some("SnapshotCollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/export_collection",
            tag: "collection",
            summary: "Export collection",
            request_schema: Some("ExportCollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/restore_collection",
            tag: "collection",
            summary: "Restore collection",
            request_schema: Some("RestoreCollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/import_collection",
            tag: "collection",
            summary: "Import collection",
            request_schema: Some("ImportCollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/show_collections",
            tag: "collection",
            summary: "Show collections",
            request_schema: Some("DatabaseRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/add_item",
            tag: "vectors",
            summary: "Add one vector",
            request_schema: Some("AddItemRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/bulk_add_items",
            tag: "vectors",
            summary: "Add vectors in bulk",
            request_schema: Some("BulkAddItemsRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/create_vector_field",
            tag: "vectors",
            summary: "Create named vector field",
            request_schema: Some("CreateVectorFieldRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/list_vector_fields",
            tag: "vectors",
            summary: "List vector fields",
            request_schema: Some("CollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/add_named_vectors",
            tag: "vectors",
            summary: "Add named vectors",
            request_schema: Some("AddNamedVectorsRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/add_sparse_vectors",
            tag: "vectors",
            summary: "Add sparse vectors",
            request_schema: Some("AddSparseVectorsRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/build_vector_field_index",
            tag: "vectors",
            summary: "Build named vector field index",
            request_schema: Some("BuildVectorFieldIndexRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/remove_vector_field_index",
            tag: "vectors",
            summary: "Remove named vector field index",
            request_schema: Some("BuildVectorFieldIndexRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/upsert_items",
            tag: "vectors",
            summary: "Upsert vectors",
            request_schema: Some("BulkAddItemsRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/bulk_add_binary",
            tag: "vectors",
            summary: "Add vectors with binary body",
            request_schema: None,
        },
        OpenApiRoute {
            method: "post",
            path: "/search",
            tag: "query",
            summary: "Vector search",
            request_schema: Some("SearchRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/search_profile",
            tag: "query",
            summary: "Vector search with query profile",
            request_schema: Some("SearchProfileRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/text_search",
            tag: "query",
            summary: "BM25 metadata text search",
            request_schema: Some("TextSearchRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/sparse_search",
            tag: "query",
            summary: "Sparse vector inner-product search",
            request_schema: Some("SparseSearchRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/hybrid_search",
            tag: "query",
            summary: "Hybrid vector and text search",
            request_schema: Some("HybridSearchRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/search_binary",
            tag: "query",
            summary: "Vector search with binary body",
            request_schema: None,
        },
        OpenApiRoute {
            method: "post",
            path: "/batch_search",
            tag: "query",
            summary: "Batch vector search",
            request_schema: Some("BatchSearchRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/batch_search_binary",
            tag: "query",
            summary: "Batch vector search with binary body",
            request_schema: None,
        },
        OpenApiRoute {
            method: "post",
            path: "/commit",
            tag: "collection",
            summary: "Commit collection WAL",
            request_schema: Some("CommitRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/flush",
            tag: "collection",
            summary: "Flush collection",
            request_schema: Some("CollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/checkpoint",
            tag: "collection",
            summary: "Checkpoint collection",
            request_schema: Some("CollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/close_collection",
            tag: "collection",
            summary: "Close collection handle",
            request_schema: Some("CollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/collection_shape",
            tag: "collection",
            summary: "Get collection shape",
            request_schema: Some("CollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/is_collection_exists",
            tag: "collection",
            summary: "Check collection existence",
            request_schema: Some("CollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/get_collection_config",
            tag: "collection",
            summary: "Get collection config",
            request_schema: Some("CollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/update_collection_description",
            tag: "collection",
            summary: "Update collection description",
            request_schema: Some("UpdateDescriptionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/update_description",
            tag: "collection",
            summary: "Update collection description",
            request_schema: Some("UpdateDescriptionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/show_collections_details",
            tag: "collection",
            summary: "Show collection details",
            request_schema: Some("DatabaseRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/build_index",
            tag: "index",
            summary: "Build index",
            request_schema: Some("BuildIndexRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/remove_index",
            tag: "index",
            summary: "Remove index",
            request_schema: Some("CollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/head",
            tag: "vectors",
            summary: "Read first vectors",
            request_schema: Some("HeadTailRequest"),
        },
        OpenApiRoute {
            method: "get",
            path: "/head_binary",
            tag: "vectors",
            summary: "Read first vectors as binary",
            request_schema: None,
        },
        OpenApiRoute {
            method: "post",
            path: "/tail",
            tag: "vectors",
            summary: "Read last vectors",
            request_schema: Some("HeadTailRequest"),
        },
        OpenApiRoute {
            method: "get",
            path: "/tail_binary",
            tag: "vectors",
            summary: "Read last vectors as binary",
            request_schema: None,
        },
        OpenApiRoute {
            method: "post",
            path: "/get_collection_path",
            tag: "collection",
            summary: "Get collection path",
            request_schema: Some("CollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/query",
            tag: "query",
            summary: "Query metadata fields",
            request_schema: Some("QueryRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/query_vectors",
            tag: "query",
            summary: "Query vectors by metadata or IDs",
            request_schema: Some("QueryVectorsRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/list_fields",
            tag: "collection",
            summary: "List metadata fields",
            request_schema: Some("CollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/index_mode",
            tag: "index",
            summary: "Get index mode",
            request_schema: Some("CollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/is_id_exists",
            tag: "vectors",
            summary: "Check vector ID existence",
            request_schema: Some("IsIdExistsRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/max_id",
            tag: "vectors",
            summary: "Get maximum vector ID",
            request_schema: Some("MaxIdRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/read_by_only_id",
            tag: "vectors",
            summary: "Read vectors by ID",
            request_schema: Some("ReadByIdRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/delete_items",
            tag: "vectors",
            summary: "Delete vectors",
            request_schema: Some("DeleteItemsRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/restore_items",
            tag: "vectors",
            summary: "Restore deleted vectors",
            request_schema: Some("DeleteItemsRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/list_deleted_ids",
            tag: "vectors",
            summary: "List deleted vector IDs",
            request_schema: Some("CollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/search_range",
            tag: "query",
            summary: "Range search",
            request_schema: Some("SearchRangeRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/compact",
            tag: "collection",
            summary: "Compact collection",
            request_schema: Some("CollectionRequest"),
        },
        OpenApiRoute {
            method: "post",
            path: "/stats",
            tag: "collection",
            summary: "Collection stats",
            request_schema: Some("CollectionRequest"),
        },
    ]
}

fn openapi_spec() -> serde_json::Value {
    let mut paths = serde_json::Map::new();
    for route in openapi_routes() {
        let mut operation = serde_json::json!({
            "tags": [route.tag],
            "summary": route.summary,
            "responses": {
                "200": { "description": "Success" },
                "400": { "description": "Bad request" },
                "401": { "description": "Unauthorized" },
                "500": { "description": "Server error" }
            }
        });

        if let Some(schema_name) = route.request_schema {
            operation["requestBody"] = serde_json::json!({
                "required": true,
                "content": {
                    "application/json": {
                        "schema": { "$ref": format!("#/components/schemas/{schema_name}") }
                    }
                }
            });
        }

        let entry = paths
            .entry(route.path.to_string())
            .or_insert_with(|| serde_json::json!({}));
        if let Some(obj) = entry.as_object_mut() {
            obj.insert(route.method.to_string(), operation);
        }
    }

    serde_json::json!({
        "openapi": "3.0.3",
        "info": {
            "title": "LynseDB HTTP API",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "Embedded-first vector database HTTP API."
        },
        "servers": [{"url": "http://127.0.0.1:7637"}],
        "security": [{"ApiKeyAuth": []}],
        "paths": paths,
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "API key"
                }
            },
            "schemas": openapi_schemas()
        }
    })
}

fn openapi_schemas() -> serde_json::Value {
    serde_json::json!({
        "ApiResponse": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "params": {"type": "object", "additionalProperties": true},
                "error": {"type": "string"}
            }
        },
        "DatabaseRequest": object_schema(&["database_name"]),
        "CreateDatabaseRequest": object_schema(&["database_name", "drop_if_exists"]),
        "DropDatabaseRequest": object_schema(&["database_name"]),
        "DatabaseExistsRequest": object_schema(&["database_name"]),
        "SnapshotDatabaseRequest": object_schema(&["database_name", "snapshot_path"]),
        "RestoreDatabaseRequest": object_schema(&["database_name", "snapshot_path", "overwrite"]),
        "RequireCollectionRequest": object_schema(&["database_name", "collection_name", "dim", "drop_if_exists", "description"]),
        "CollectionRequest": object_schema(&["database_name", "collection_name"]),
        "SnapshotCollectionRequest": object_schema(&["database_name", "collection_name", "snapshot_path"]),
        "ExportCollectionRequest": object_schema(&["database_name", "collection_name", "export_path"]),
        "RestoreCollectionRequest": object_schema(&["database_name", "collection_name", "snapshot_path", "overwrite"]),
        "ImportCollectionRequest": object_schema(&["database_name", "collection_name", "export_path", "overwrite"]),
        "AddItemRequest": object_schema(&["database_name", "collection_name", "item"]),
        "BulkAddItemsRequest": object_schema(&["database_name", "collection_name", "items"]),
        "CreateVectorFieldRequest": object_schema(&["database_name", "collection_name", "field_name", "dim", "metric", "index_mode"]),
        "AddNamedVectorsRequest": object_schema(&["database_name", "collection_name", "field_name", "vectors", "ids"]),
        "AddSparseVectorsRequest": object_schema(&["database_name", "collection_name", "vectors", "ids"]),
        "BuildVectorFieldIndexRequest": object_schema(&["database_name", "collection_name", "field_name", "index_mode"]),
        "SearchRequest": object_schema(&["database_name", "collection_name", "vector", "vector_field", "k", "where", "return_fields", "nprobe"]),
        "SearchProfileRequest": object_schema(&["database_name", "collection_name", "vector", "k", "where", "return_fields", "nprobe"]),
        "TextSearchRequest": object_schema(&["database_name", "collection_name", "text", "text_fields", "k", "where", "return_fields"]),
        "SparseSearchRequest": object_schema(&["database_name", "collection_name", "vector", "k", "where", "return_fields"]),
        "HybridSearchRequest": object_schema(&["database_name", "collection_name", "vector", "text", "text_fields", "k", "where", "fusion", "vector_weight", "text_weight", "rrf_k", "candidate_limit", "return_fields", "nprobe"]),
        "BatchSearchRequest": object_schema(&["database_name", "collection_name", "vectors", "k", "where", "return_fields", "nprobe"]),
        "CommitRequest": object_schema(&["database_name", "collection_name", "items"]),
        "UpdateDescriptionRequest": object_schema(&["database_name", "collection_name", "description"]),
        "BuildIndexRequest": object_schema(&["database_name", "collection_name", "index_mode"]),
        "HeadTailRequest": object_schema(&["database_name", "collection_name", "n"]),
        "QueryRequest": object_schema(&["database_name", "collection_name", "where", "filter_ids", "return_ids_only"]),
        "QueryVectorsRequest": object_schema(&["database_name", "collection_name", "where", "filter_ids"]),
        "IsIdExistsRequest": object_schema(&["database_name", "collection_name", "id"]),
        "MaxIdRequest": object_schema(&["database_name", "collection_name"]),
        "ReadByIdRequest": object_schema(&["database_name", "collection_name", "id"]),
        "DeleteItemsRequest": object_schema(&["database_name", "collection_name", "ids"]),
        "SearchRangeRequest": object_schema(&["database_name", "collection_name", "vector", "threshold", "max_results"])
    })
}

fn object_schema(properties: &[&str]) -> serde_json::Value {
    let props: serde_json::Map<String, serde_json::Value> = properties
        .iter()
        .map(|name| {
            (
                (*name).to_string(),
                serde_json::json!({
                    "description": format!("{name} field"),
                }),
            )
        })
        .collect();
    serde_json::json!({
        "type": "object",
        "properties": props,
        "additionalProperties": true
    })
}

async fn openapi_json() -> HttpResponse {
    HttpResponse::Ok().json(openapi_spec())
}

async fn metrics(state: web::Data<AppState>) -> HttpResponse {
    let database_names = state.manager.list_databases();
    let databases_total = database_names.len();
    let collections_total: usize = database_names
        .iter()
        .filter_map(|db| state.manager.show_collections(db).ok())
        .map(|collections| collections.len())
        .sum();

    let open_databases_total = state.manager.open_database_count();
    let open_collections_total = state.manager.open_collection_count();
    let storage_usage = collect_storage_usage(state.manager.root_path());
    let process_memory_bytes = process_resident_memory_bytes();
    let request_total = state.metrics.request_total();
    let request_duration_sum_seconds = state.metrics.request_duration_sum_seconds();
    let request_error_total = state.metrics.request_error_total.load(Ordering::Relaxed);
    let status_2xx = state.metrics.status_2xx.load(Ordering::Relaxed);
    let status_3xx = state.metrics.status_3xx.load(Ordering::Relaxed);
    let status_4xx = state.metrics.status_4xx.load(Ordering::Relaxed);
    let status_5xx = state.metrics.status_5xx.load(Ordering::Relaxed);
    let status_other = state.metrics.status_other.load(Ordering::Relaxed);
    let error_client_4xx_total = state.metrics.error_client_4xx_total.load(Ordering::Relaxed);
    let error_server_5xx_total = state.metrics.error_server_5xx_total.load(Ordering::Relaxed);
    let error_unauthorized_total = state
        .metrics
        .error_unauthorized_total
        .load(Ordering::Relaxed);
    let error_handler_failure_total = state
        .metrics
        .error_handler_failure_total
        .load(Ordering::Relaxed);
    let p50 = state.metrics.request_latency_quantile_seconds(0.50);
    let p90 = state.metrics.request_latency_quantile_seconds(0.90);
    let p99 = state.metrics.request_latency_quantile_seconds(0.99);
    let index_build_started_total = state
        .metrics
        .index_build_started_total
        .load(Ordering::Relaxed);
    let index_build_completed_total = state
        .metrics
        .index_build_completed_total
        .load(Ordering::Relaxed);
    let index_build_failed_total = state
        .metrics
        .index_build_failed_total
        .load(Ordering::Relaxed);
    let index_build_in_progress = state
        .metrics
        .index_build_in_progress
        .load(Ordering::Relaxed);
    let index_build_current_vectors = state
        .metrics
        .index_build_current_vectors
        .load(Ordering::Relaxed);
    let index_build_last_vectors = state
        .metrics
        .index_build_last_vectors
        .load(Ordering::Relaxed);
    let index_build_duration_sum_seconds = state.metrics.index_build_duration_sum_seconds();
    let index_build_last_duration_seconds = state.metrics.index_build_last_duration_seconds();
    let index_build_current_progress = state
        .metrics
        .index_build_current_progress_ppm
        .load(Ordering::Relaxed) as f64
        / 1_000_000.0;
    let index_build_last_progress = state
        .metrics
        .index_build_last_progress_ppm
        .load(Ordering::Relaxed) as f64
        / 1_000_000.0;
    let index_build_finished_total =
        index_build_completed_total.saturating_add(index_build_failed_total);

    let mut histogram = String::new();
    let mut cumulative = 0u64;
    for (idx, bound) in state.metrics.histogram_bucket_bounds.iter().enumerate() {
        cumulative += state.metrics.histogram_bucket_counts[idx].load(Ordering::Relaxed);
        histogram.push_str(&format!(
            "lynsedb_http_request_duration_seconds_bucket{{le=\"{}\"}} {}\n",
            bound, cumulative
        ));
    }
    cumulative += state.metrics.histogram_bucket_counts
        [state.metrics.histogram_bucket_bounds.len()]
    .load(Ordering::Relaxed);
    histogram.push_str(&format!(
        "lynsedb_http_request_duration_seconds_bucket{{le=\"+Inf\"}} {}\n",
        cumulative
    ));

    let body = format!(
        concat!(
            "# HELP lynsedb_up LynseDB server availability.\n",
            "# TYPE lynsedb_up gauge\n",
            "lynsedb_up 1\n",
            "# HELP lynsedb_server_start_time_seconds LynseDB server start time in unix seconds.\n",
            "# TYPE lynsedb_server_start_time_seconds gauge\n",
            "lynsedb_server_start_time_seconds {start_time}\n",
            "# HELP lynsedb_databases_total Number of databases visible on disk.\n",
            "# TYPE lynsedb_databases_total gauge\n",
            "lynsedb_databases_total {db_total}\n",
            "# HELP lynsedb_collections_total Number of collections visible on disk across databases.\n",
            "# TYPE lynsedb_collections_total gauge\n",
            "lynsedb_collections_total {coll_total}\n",
            "# HELP lynsedb_open_databases_total Number of currently opened database engines.\n",
            "# TYPE lynsedb_open_databases_total gauge\n",
            "lynsedb_open_databases_total {open_db_total}\n",
            "# HELP lynsedb_open_collections_total Number of currently opened collection handles.\n",
            "# TYPE lynsedb_open_collections_total gauge\n",
            "lynsedb_open_collections_total {open_coll_total}\n",
            "# HELP lynsedb_storage_disk_bytes Total bytes used by the LynseDB data directory.\n",
            "# TYPE lynsedb_storage_disk_bytes gauge\n",
            "lynsedb_storage_disk_bytes {disk_usage_bytes}\n",
            "# HELP lynsedb_storage_wal_bytes Total bytes used by WAL files under the LynseDB data directory.\n",
            "# TYPE lynsedb_storage_wal_bytes gauge\n",
            "lynsedb_storage_wal_bytes {wal_bytes}\n",
            "# HELP lynsedb_storage_index_bytes Total bytes used by vector index files under the LynseDB data directory.\n",
            "# TYPE lynsedb_storage_index_bytes gauge\n",
            "lynsedb_storage_index_bytes {index_bytes}\n",
            "# HELP lynsedb_process_resident_memory_bytes Resident memory currently attributable to the LynseDB process when available.\n",
            "# TYPE lynsedb_process_resident_memory_bytes gauge\n",
            "lynsedb_process_resident_memory_bytes {process_memory_bytes}\n",
            "# HELP lynsedb_index_builds_total Total number of vector index build attempts by status.\n",
            "# TYPE lynsedb_index_builds_total counter\n",
            "lynsedb_index_builds_total{{status=\"started\"}} {index_build_started_total}\n",
            "lynsedb_index_builds_total{{status=\"completed\"}} {index_build_completed_total}\n",
            "lynsedb_index_builds_total{{status=\"failed\"}} {index_build_failed_total}\n",
            "# HELP lynsedb_index_builds_in_progress Number of vector index builds currently running.\n",
            "# TYPE lynsedb_index_builds_in_progress gauge\n",
            "lynsedb_index_builds_in_progress {index_build_in_progress}\n",
            "# HELP lynsedb_index_build_current_vectors Number of vectors in currently running index builds.\n",
            "# TYPE lynsedb_index_build_current_vectors gauge\n",
            "lynsedb_index_build_current_vectors {index_build_current_vectors}\n",
            "# HELP lynsedb_index_build_last_vectors Number of vectors processed by the most recent index build.\n",
            "# TYPE lynsedb_index_build_last_vectors gauge\n",
            "lynsedb_index_build_last_vectors {index_build_last_vectors}\n",
            "# HELP lynsedb_index_build_progress_ratio Current and last vector index build progress ratio.\n",
            "# TYPE lynsedb_index_build_progress_ratio gauge\n",
            "lynsedb_index_build_progress_ratio{{scope=\"current\"}} {index_build_current_progress}\n",
            "lynsedb_index_build_progress_ratio{{scope=\"last\"}} {index_build_last_progress}\n",
            "# HELP lynsedb_index_build_duration_seconds Duration of vector index builds in seconds.\n",
            "# TYPE lynsedb_index_build_duration_seconds summary\n",
            "lynsedb_index_build_duration_seconds_sum {index_build_duration_sum_seconds}\n",
            "lynsedb_index_build_duration_seconds_count {index_build_finished_total}\n",
            "# HELP lynsedb_index_build_last_duration_seconds Duration of the most recent vector index build in seconds.\n",
            "# TYPE lynsedb_index_build_last_duration_seconds gauge\n",
            "lynsedb_index_build_last_duration_seconds {index_build_last_duration_seconds}\n",
            "# HELP lynsedb_http_requests_total Total number of HTTP requests handled by LynseDB.\n",
            "# TYPE lynsedb_http_requests_total counter\n",
            "lynsedb_http_requests_total {request_total}\n",
            "# HELP lynsedb_http_requests_by_status_total Total number of HTTP requests by status class.\n",
            "# TYPE lynsedb_http_requests_by_status_total counter\n",
            "lynsedb_http_requests_by_status_total{{status_class=\"2xx\"}} {status_2xx}\n",
            "lynsedb_http_requests_by_status_total{{status_class=\"3xx\"}} {status_3xx}\n",
            "lynsedb_http_requests_by_status_total{{status_class=\"4xx\"}} {status_4xx}\n",
            "lynsedb_http_requests_by_status_total{{status_class=\"5xx\"}} {status_5xx}\n",
            "lynsedb_http_requests_by_status_total{{status_class=\"other\"}} {status_other}\n",
            "# HELP lynsedb_http_request_errors_total Total number of HTTP requests resulting in error responses.\n",
            "# TYPE lynsedb_http_request_errors_total counter\n",
            "lynsedb_http_request_errors_total {request_error_total}\n",
            "# HELP lynsedb_http_request_errors_by_kind_total Error counters split by semantic reason.\n",
            "# TYPE lynsedb_http_request_errors_by_kind_total counter\n",
            "lynsedb_http_request_errors_by_kind_total{{kind=\"client_4xx\"}} {error_client_4xx_total}\n",
            "lynsedb_http_request_errors_by_kind_total{{kind=\"server_5xx\"}} {error_server_5xx_total}\n",
            "lynsedb_http_request_errors_by_kind_total{{kind=\"unauthorized\"}} {error_unauthorized_total}\n",
            "lynsedb_http_request_errors_by_kind_total{{kind=\"handler_failure\"}} {error_handler_failure_total}\n",
            "# HELP lynsedb_http_request_duration_seconds HTTP request latency histogram in seconds.\n",
            "# TYPE lynsedb_http_request_duration_seconds histogram\n",
            "{histogram}",
            "lynsedb_http_request_duration_seconds_sum {request_duration_sum_seconds}\n",
            "lynsedb_http_request_duration_seconds_count {request_total}\n",
            "# HELP lynsedb_http_request_duration_seconds_quantile Estimated HTTP request latency quantiles.\n",
            "# TYPE lynsedb_http_request_duration_seconds_quantile gauge\n",
            "lynsedb_http_request_duration_seconds_quantile{{quantile=\"0.5\"}} {p50}\n",
            "lynsedb_http_request_duration_seconds_quantile{{quantile=\"0.9\"}} {p90}\n",
            "lynsedb_http_request_duration_seconds_quantile{{quantile=\"0.99\"}} {p99}\n"
        ),
        start_time = state.start_time_unix_seconds,
        db_total = databases_total,
        coll_total = collections_total,
        open_db_total = open_databases_total,
        open_coll_total = open_collections_total,
        disk_usage_bytes = storage_usage.disk_bytes,
        wal_bytes = storage_usage.wal_bytes,
        index_bytes = storage_usage.index_bytes,
        process_memory_bytes = process_memory_bytes,
        index_build_started_total = index_build_started_total,
        index_build_completed_total = index_build_completed_total,
        index_build_failed_total = index_build_failed_total,
        index_build_finished_total = index_build_finished_total,
        index_build_in_progress = index_build_in_progress,
        index_build_current_vectors = index_build_current_vectors,
        index_build_last_vectors = index_build_last_vectors,
        index_build_current_progress = index_build_current_progress,
        index_build_last_progress = index_build_last_progress,
        index_build_duration_sum_seconds = index_build_duration_sum_seconds,
        index_build_last_duration_seconds = index_build_last_duration_seconds,
        request_total = request_total,
        request_error_total = request_error_total,
        status_2xx = status_2xx,
        status_3xx = status_3xx,
        status_4xx = status_4xx,
        status_5xx = status_5xx,
        status_other = status_other,
        error_client_4xx_total = error_client_4xx_total,
        error_server_5xx_total = error_server_5xx_total,
        error_unauthorized_total = error_unauthorized_total,
        error_handler_failure_total = error_handler_failure_total,
        histogram = histogram,
        request_duration_sum_seconds = request_duration_sum_seconds,
        p50 = p50,
        p90 = p90,
        p99 = p99
    );

    HttpResponse::Ok()
        .content_type("text/plain; version=0.0.4; charset=utf-8")
        .body(body)
}

async fn create_database(
    state: web::Data<AppState>,
    body: web::Json<CreateDatabaseRequest>,
) -> HttpResponse {
    let drop_if_exists = body.drop_if_exists.unwrap_or(false);

    if drop_if_exists && state.manager.database_exists(&body.database_name) {
        if let Err(e) = state.manager.drop_database(&body.database_name) {
            return error_response(&e.to_string());
        }
    }

    match state.manager.create_database(&body.database_name) {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn drop_database(
    state: web::Data<AppState>,
    body: web::Json<DropDatabaseRequest>,
) -> HttpResponse {
    match state.manager.drop_database(&body.database_name) {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn snapshot_database(
    state: web::Data<AppState>,
    body: web::Json<SnapshotDatabaseRequest>,
) -> HttpResponse {
    match state.manager.snapshot_database(
        &body.database_name,
        std::path::Path::new(&body.snapshot_path),
    ) {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "snapshot_path": body.snapshot_path
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn restore_database(
    state: web::Data<AppState>,
    body: web::Json<RestoreDatabaseRequest>,
) -> HttpResponse {
    match state.manager.restore_database_from_snapshot(
        &body.database_name,
        std::path::Path::new(&body.snapshot_path),
        body.overwrite.unwrap_or(false),
    ) {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "snapshot_path": body.snapshot_path
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn database_exists(
    state: web::Data<AppState>,
    body: web::Json<DatabaseExistsRequest>,
) -> HttpResponse {
    let exists = state.manager.database_exists(&body.database_name);
    ApiResponse::success(serde_json::json!({
        "database_name": body.database_name,
        "exists": exists
    }))
}

async fn list_databases(state: web::Data<AppState>) -> HttpResponse {
    let databases = state.manager.list_databases();
    ApiResponse::success(serde_json::json!({
        "databases": databases
    }))
}

async fn delete_database(
    state: web::Data<AppState>,
    body: web::Json<DropDatabaseRequest>,
) -> HttpResponse {
    match state.manager.drop_database(&body.database_name) {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

// ─── Collection operation handlers ───────────────────────────────────────────

async fn required_collection(
    state: web::Data<AppState>,
    body: web::Json<RequireCollectionRequest>,
) -> HttpResponse {
    let drop_if_exists = body.drop_if_exists.unwrap_or(false);

    match state.manager.require_collection(
        &body.database_name,
        &body.collection_name,
        body.dim,
        100_000,
        drop_if_exists,
        body.description.as_deref(),
    ) {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn drop_collection(
    state: web::Data<AppState>,
    body: web::Json<CollectionRequest>,
) -> HttpResponse {
    match state
        .manager
        .drop_collection(&body.database_name, &body.collection_name)
    {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn snapshot_collection(
    state: web::Data<AppState>,
    body: web::Json<SnapshotCollectionRequest>,
) -> HttpResponse {
    match state.manager.snapshot_collection(
        &body.database_name,
        &body.collection_name,
        std::path::Path::new(&body.snapshot_path),
    ) {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "snapshot_path": body.snapshot_path
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn export_collection(
    state: web::Data<AppState>,
    body: web::Json<ExportCollectionRequest>,
) -> HttpResponse {
    match state.manager.export_collection(
        &body.database_name,
        &body.collection_name,
        std::path::Path::new(&body.export_path),
    ) {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "export_path": body.export_path
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn restore_collection(
    state: web::Data<AppState>,
    body: web::Json<RestoreCollectionRequest>,
) -> HttpResponse {
    match state.manager.restore_collection_from_snapshot(
        &body.database_name,
        &body.collection_name,
        std::path::Path::new(&body.snapshot_path),
        body.overwrite.unwrap_or(false),
    ) {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "snapshot_path": body.snapshot_path
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn import_collection(
    state: web::Data<AppState>,
    body: web::Json<ImportCollectionRequest>,
) -> HttpResponse {
    match state.manager.import_collection_from_export(
        &body.database_name,
        &body.collection_name,
        std::path::Path::new(&body.export_path),
        body.overwrite.unwrap_or(false),
    ) {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "export_path": body.export_path
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn show_collections(
    state: web::Data<AppState>,
    body: web::Json<DatabaseRequest>,
) -> HttpResponse {
    let db_name = match &body.database_name {
        Some(name) => name.as_str(),
        None => return bad_request("Missing required parameter: database_name"),
    };

    match state.manager.show_collections(db_name) {
        Ok(collections) => ApiResponse::success(serde_json::json!({
            "database_name": db_name,
            "collections": collections
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn add_item(state: web::Data<AppState>, body: web::Json<AddItemRequest>) -> HttpResponse {
    let dim = body.item.vector.len();
    let vectors = body.item.vector.clone();
    let fields = body.item.field.clone().map(|f| vec![f]);
    let limits = state.limits;
    let vector_bytes = match checked_vector_bytes(1, dim) {
        Ok(bytes) => bytes,
        Err(e) => return limit_bad_request(e),
    };

    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }
    let result = state.manager.with_database(&body.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&body.collection_name, dim, 100_000)?;
        let mut coll = coll_arc.write();
        let user_id = body.item.id.unwrap_or_else(|| {
            coll.max_id()
                .map(|max_id| max_id + 1)
                .unwrap_or_else(|| coll.shape().map(|(n, _)| n).unwrap_or(0))
        });
        validate_collection_insert(&limits, &coll, 1, vector_bytes)?;
        coll.add_items(&vectors, 1, &[user_id], fields.as_deref())?;
        Ok(user_id)
    });

    match result {
        Ok(user_id) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "item": { "id": user_id }
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn bulk_add_binary(
    state: web::Data<AppState>,
    query: web::Query<BulkAddBinaryQuery>,
    body: web::Bytes,
) -> HttpResponse {
    let dim = query.dim;
    let n_vectors = query.n_vectors;
    if let Err(e) = validate_batch_vectors(&state.limits, n_vectors, "n_vectors") {
        return limit_bad_request(e);
    }
    let expected_bytes_u64 = match checked_vector_bytes(n_vectors, dim) {
        Ok(bytes) => bytes,
        Err(e) => return limit_bad_request(e),
    };
    if let Err(e) = validate_request_vector_bytes(&state.limits, n_vectors, dim, "binary payload") {
        return limit_bad_request(e);
    }
    let expected_bytes = match usize::try_from(expected_bytes_u64) {
        Ok(bytes) => bytes,
        Err(_) => return bad_request("Expected binary payload size exceeds usize"),
    };

    if body.len() != expected_bytes {
        return bad_request(&format!(
            "Expected {} bytes ({} vectors × {} dim × 4), got {}",
            expected_bytes,
            n_vectors,
            dim,
            body.len()
        ));
    }

    // Zero-copy reinterpret bytes as f32 slice
    let float_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(body.as_ptr() as *const f32, n_vectors * dim) };

    if let Err(e) = state.manager.get_or_open_database(&query.database_name) {
        return error_response(&e.to_string());
    }
    let limits = state.limits;
    // bulk_add_binary has no user IDs in the binary protocol — assign sequential from current max
    let result = state.manager.with_database(&query.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&query.collection_name, dim, 100_000)?;
        let mut coll = coll_arc.write();
        let start_id = coll
            .max_id()
            .map(|m| m + 1)
            .unwrap_or_else(|| coll.shape().map(|(n, _)| n).unwrap_or(0));
        let seq_ids: Vec<u64> = (start_id..start_id + n_vectors as u64).collect();
        validate_collection_insert(&limits, &coll, n_vectors as u64, expected_bytes_u64)?;
        coll.add_items(float_slice, n_vectors, &seq_ids, None)?;
        Ok(())
    });

    match result {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": query.database_name,
            "collection_name": query.collection_name,
            "n_vectors": n_vectors
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn bulk_add_items(
    state: web::Data<AppState>,
    body: web::Json<BulkAddItemsRequest>,
) -> HttpResponse {
    if body.items.is_empty() {
        return bad_request("No items provided");
    }

    let dim = body.items[0].vector.len();
    let n_vectors = body.items.len();
    if let Err(e) = validate_batch_vectors(&state.limits, n_vectors, "items") {
        return limit_bad_request(e);
    }
    let vector_bytes = match checked_vector_bytes(n_vectors, dim) {
        Ok(bytes) => bytes,
        Err(e) => return limit_bad_request(e),
    };
    if let Err(e) = validate_request_vector_bytes(&state.limits, n_vectors, dim, "items") {
        return limit_bad_request(e);
    }
    let mut flat_vectors: Vec<f32> = Vec::with_capacity(n_vectors * dim);
    let mut fields: Vec<HashMap<String, serde_json::Value>> = Vec::with_capacity(n_vectors);
    let mut ids: Vec<Option<u64>> = Vec::with_capacity(n_vectors);

    for item in &body.items {
        if item.vector.len() != dim {
            return bad_request("all vectors in a batch must have the same dimension");
        }
        flat_vectors.extend_from_slice(&item.vector);
        fields.push(item.field.clone().unwrap_or_default());
        ids.push(item.id);
    }

    let has_fields = fields.iter().any(|f| !f.is_empty());

    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }
    let limits = state.limits;
    let result = state.manager.with_database(&body.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&body.collection_name, dim, 100_000)?;
        let mut coll = coll_arc.write();
        let mut next_id = coll.max_id().map(|max_id| max_id + 1).unwrap_or(0);
        let mut reserved = HashSet::with_capacity(n_vectors);
        let mut resolved_ids = Vec::with_capacity(n_vectors);
        for opt_id in &ids {
            if let Some(id) = opt_id {
                reserved.insert(*id);
                resolved_ids.push(*id);
            } else {
                while coll.is_id_exists(next_id) || reserved.contains(&next_id) {
                    next_id += 1;
                }
                resolved_ids.push(next_id);
                reserved.insert(next_id);
                next_id += 1;
            }
        }
        validate_collection_insert(&limits, &coll, n_vectors as u64, vector_bytes)?;
        if has_fields {
            coll.add_items(&flat_vectors, n_vectors, &resolved_ids, Some(&fields))?;
        } else {
            coll.add_items(&flat_vectors, n_vectors, &resolved_ids, None)?;
        }
        Ok(resolved_ids)
    });

    match result {
        Ok(resolved_ids) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "ids": resolved_ids
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn create_vector_field(
    state: web::Data<AppState>,
    body: web::Json<CreateVectorFieldRequest>,
) -> HttpResponse {
    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }

    let result = state.manager.with_database(&body.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&body.collection_name, 0, 100_000)?;
        let mut coll = coll_arc.write();
        coll.create_vector_field(
            &body.field_name,
            body.dim,
            body.metric.as_deref(),
            body.index_mode.as_deref(),
        )
    });

    match result {
        Ok(config) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "field": config
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn list_vector_fields(
    state: web::Data<AppState>,
    body: web::Json<CollectionRequest>,
) -> HttpResponse {
    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }

    let result = state.manager.with_database(&body.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&body.collection_name, 0, 100_000)?;
        let coll = coll_arc.read();
        Ok(coll.list_vector_fields())
    });

    match result {
        Ok(fields) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "fields": fields
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn add_named_vectors(
    state: web::Data<AppState>,
    body: web::Json<AddNamedVectorsRequest>,
) -> HttpResponse {
    if body.vectors.is_empty() {
        return bad_request("No vectors provided");
    }
    if body.vectors.len() != body.ids.len() {
        return bad_request("vectors length must match ids length");
    }

    let dim = body.vectors[0].len();
    let n_vectors = body.vectors.len();
    if let Err(e) = validate_batch_vectors(&state.limits, n_vectors, "vectors") {
        return limit_bad_request(e);
    }
    let vector_bytes = match checked_vector_bytes(n_vectors, dim) {
        Ok(bytes) => bytes,
        Err(e) => return limit_bad_request(e),
    };
    if let Err(e) = validate_request_vector_bytes(&state.limits, n_vectors, dim, "vectors") {
        return limit_bad_request(e);
    }
    let mut flat_vectors = Vec::with_capacity(n_vectors * dim);
    for vector in &body.vectors {
        if vector.len() != dim {
            return bad_request("all named vectors in a batch must have the same dimension");
        }
        flat_vectors.extend_from_slice(vector);
    }

    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }

    let limits = state.limits;
    let result = state.manager.with_database(&body.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&body.collection_name, 0, 100_000)?;
        let mut coll = coll_arc.write();
        validate_collection_vector_bytes(&limits, coll.estimated_vector_bytes()?, vector_bytes)?;
        coll.add_named_vectors(&body.field_name, &flat_vectors, n_vectors, &body.ids)
    });

    match result {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "field_name": body.field_name,
            "ids": body.ids,
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn add_sparse_vectors(
    state: web::Data<AppState>,
    body: web::Json<AddSparseVectorsRequest>,
) -> HttpResponse {
    if body.vectors.len() != body.ids.len() {
        return bad_request("vectors length must match ids length");
    }
    if let Err(e) = validate_batch_vectors(&state.limits, body.vectors.len(), "vectors") {
        return limit_bad_request(e);
    }

    let vectors = match body
        .vectors
        .iter()
        .map(sparse_vector_entries)
        .collect::<std::result::Result<Vec<_>, LynseError>>()
    {
        Ok(vectors) => vectors,
        Err(e) => return bad_request(&e.to_string()),
    };

    let result = with_collection_mut(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| coll.add_sparse_vectors(&body.ids, &vectors),
    );

    match result {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "ids": body.ids,
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn build_vector_field_index(
    state: web::Data<AppState>,
    body: web::Json<BuildVectorFieldIndexRequest>,
) -> HttpResponse {
    let index_mode = body.index_mode.as_deref().unwrap_or("FLAT");
    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }

    let metrics = Arc::clone(&state.metrics);
    let result = state.manager.with_database(&body.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&body.collection_name, 0, 100_000)?;
        let mut coll = coll_arc.write();
        let n_vectors = coll
            .vector_field_shape(&body.field_name)
            .map(|(n, _)| n)
            .unwrap_or(0);
        metrics.track_index_build(n_vectors, || {
            coll.build_vector_field_index(&body.field_name, index_mode)
        })
    });

    match result {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "field_name": body.field_name,
            "index_mode": index_mode,
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn remove_vector_field_index(
    state: web::Data<AppState>,
    body: web::Json<BuildVectorFieldIndexRequest>,
) -> HttpResponse {
    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }

    let result = state.manager.with_database(&body.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&body.collection_name, 0, 100_000)?;
        let mut coll = coll_arc.write();
        coll.remove_vector_field_index(&body.field_name)
    });

    match result {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "field_name": body.field_name,
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn upsert_items(
    state: web::Data<AppState>,
    body: web::Json<BulkAddItemsRequest>,
) -> HttpResponse {
    if body.items.is_empty() {
        return bad_request("No items provided");
    }

    let dim = body.items[0].vector.len();
    let n_vectors = body.items.len();
    if let Err(e) = validate_batch_vectors(&state.limits, n_vectors, "items") {
        return limit_bad_request(e);
    }
    if let Err(e) = validate_request_vector_bytes(&state.limits, n_vectors, dim, "items") {
        return limit_bad_request(e);
    }
    for item in &body.items {
        if item.vector.len() != dim {
            return bad_request("all vectors in a batch must have the same dimension");
        }
    }

    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }

    let limits = state.limits;
    let result = state.manager.with_database(&body.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&body.collection_name, dim, 100_000)?;
        let mut coll = coll_arc.write();

        let mut vectors_with_fields = Vec::new();
        let mut ids_with_fields = Vec::new();
        let mut fields_with_fields = Vec::new();
        let mut vectors_without_fields = Vec::new();
        let mut ids_without_fields = Vec::new();
        let mut returned_ids = Vec::with_capacity(body.items.len());
        let mut seen_ids = HashSet::with_capacity(body.items.len());

        for item in &body.items {
            let user_id = item.id.ok_or_else(|| {
                LynseError::InvalidArgument("upsert_items requires every item to include id".into())
            })?;
            if !seen_ids.insert(user_id) {
                return Err(LynseError::InvalidArgument(format!(
                    "duplicate id {} within the same upsert batch",
                    user_id
                )));
            }
            returned_ids.push(user_id);
            if let Some(field) = item.field.clone() {
                vectors_with_fields.extend_from_slice(&item.vector);
                ids_with_fields.push(user_id);
                fields_with_fields.push(field);
            } else {
                vectors_without_fields.extend_from_slice(&item.vector);
                ids_without_fields.push(user_id);
            }
        }

        let additional_vectors = returned_ids
            .iter()
            .filter(|&&id| !coll.is_id_exists(id))
            .count() as u64;
        let additional_bytes = checked_vector_bytes(additional_vectors as usize, dim)?;
        validate_collection_insert(&limits, &coll, additional_vectors, additional_bytes)?;

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

        Ok(returned_ids)
    });

    match result {
        Ok(ids) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "ids": ids
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn search(state: web::Data<AppState>, body: web::Json<SearchRequest>) -> HttpResponse {
    let k = body.k.unwrap_or(10);
    if let Err(e) = validate_top_k(&state.limits, k, "k") {
        return limit_bad_request(e);
    }
    if let Err(e) = validate_request_vector_bytes(&state.limits, 1, body.vector.len(), "vector") {
        return limit_bad_request(e);
    }
    let nprobe = body.nprobe.unwrap_or(10);
    let approx = body.approx.unwrap_or(false);
    let eps = body.eps.unwrap_or(1e-4);
    let return_fields = body.return_fields.unwrap_or(false);
    let filter = body.where_expr.as_deref();
    let vector_field = body.vector_field.as_deref().unwrap_or("default");

    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }
    let result = state.manager.with_database(&body.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&body.collection_name, 0, 100_000)?;
        let coll = coll_arc.read();
        let sr = if vector_field == "default" {
            coll.search(&body.vector, k, filter, nprobe, approx, eps)?
        } else {
            coll.search_vector_field(vector_field, &body.vector, k, filter)?
        };

        let fields_data = if return_fields && !sr.ids.is_empty() {
            let retrieved = coll.retrieve_fields(&sr.ids)?;
            retrieved
        } else {
            Vec::new()
        };

        Ok((sr, fields_data))
    });

    match result {
        Ok((sr, fields_data)) => {
            let ids: Vec<i64> = sr.ids.iter().map(|&id| id as i64).collect();
            let scores = sr.distances;
            let fields: Vec<HashMap<String, serde_json::Value>> = if return_fields {
                fields_data
            } else {
                Vec::new()
            };

            ApiResponse::success(serde_json::json!({
                "database_name": body.database_name,
                "collection_name": body.collection_name,
                "items": {
                    "k": k,
                    "ids": ids,
                    "scores": scores,
                    "vector_field": vector_field,
                    "fields": fields,
                }
            }))
        }
        Err(e) => error_response(&e.to_string()),
    }
}

async fn search_profile(
    state: web::Data<AppState>,
    body: web::Json<SearchProfileRequest>,
) -> HttpResponse {
    let k = body.k.unwrap_or(10);
    if let Err(e) = validate_top_k(&state.limits, k, "k") {
        return limit_bad_request(e);
    }
    if let Err(e) = validate_request_vector_bytes(&state.limits, 1, body.vector.len(), "vector") {
        return limit_bad_request(e);
    }
    let nprobe = body.nprobe.unwrap_or(10);
    let approx = body.approx.unwrap_or(false);
    let eps = body.eps.unwrap_or(1e-4);
    let return_fields = body.return_fields.unwrap_or(false);
    let filter = body.where_expr.as_deref();

    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| {
            let (sr, profile) =
                coll.search_with_profile(&body.vector, k, filter, nprobe, approx, eps)?;
            let fields_data = if return_fields && !sr.ids.is_empty() {
                coll.retrieve_fields(&sr.ids)?
            } else {
                Vec::new()
            };
            Ok((sr, fields_data, profile))
        },
    );

    match result {
        Ok((sr, fields_data, profile)) => {
            let ids: Vec<i64> = sr.ids.iter().map(|&id| id as i64).collect();
            ApiResponse::success(serde_json::json!({
                "database_name": body.database_name,
                "collection_name": body.collection_name,
                "items": {
                    "k": k,
                    "ids": ids,
                    "scores": sr.distances,
                    "fields": if return_fields { fields_data } else { Vec::new() },
                    "index": sr.index_mode,
                },
                "profile": profile,
            }))
        }
        Err(e) => error_response(&e.to_string()),
    }
}

async fn text_search(
    state: web::Data<AppState>,
    body: web::Json<TextSearchRequest>,
) -> HttpResponse {
    let k = body.k.unwrap_or(10);
    if let Err(e) = validate_top_k(&state.limits, k, "k") {
        return limit_bad_request(e);
    }
    let return_fields = body.return_fields.unwrap_or(false);
    let filter = body.where_expr.as_deref();

    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| {
            let sr = coll.text_search(&body.text, body.text_fields.as_deref(), k, filter)?;
            let fields_data = if return_fields && !sr.ids.is_empty() {
                coll.retrieve_fields(&sr.ids)?
            } else {
                Vec::new()
            };
            Ok((sr, fields_data))
        },
    );

    match result {
        Ok((sr, fields_data)) => {
            let ids: Vec<i64> = sr.ids.iter().map(|&id| id as i64).collect();
            ApiResponse::success(serde_json::json!({
                "database_name": body.database_name,
                "collection_name": body.collection_name,
                "items": {
                    "k": k,
                    "ids": ids,
                    "scores": sr.distances,
                    "fields": if return_fields { fields_data } else { Vec::new() },
                    "index": sr.index_mode,
                }
            }))
        }
        Err(e) => error_response(&e.to_string()),
    }
}

async fn sparse_search(
    state: web::Data<AppState>,
    body: web::Json<SparseSearchRequest>,
) -> HttpResponse {
    let k = body.k.unwrap_or(10);
    if let Err(e) = validate_top_k(&state.limits, k, "k") {
        return limit_bad_request(e);
    }
    let return_fields = body.return_fields.unwrap_or(false);
    let filter = body.where_expr.as_deref();
    let query = match sparse_vector_entries(&body.vector) {
        Ok(query) => query,
        Err(e) => return bad_request(&e.to_string()),
    };

    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| {
            let sr = coll.search_sparse(&query, k, filter)?;
            let fields_data = if return_fields && !sr.ids.is_empty() {
                coll.retrieve_fields(&sr.ids)?
            } else {
                Vec::new()
            };
            Ok((sr, fields_data))
        },
    );

    match result {
        Ok((sr, fields_data)) => {
            let ids: Vec<i64> = sr.ids.iter().map(|&id| id as i64).collect();
            ApiResponse::success(serde_json::json!({
                "database_name": body.database_name,
                "collection_name": body.collection_name,
                "items": {
                    "k": k,
                    "ids": ids,
                    "scores": sr.distances,
                    "fields": if return_fields { fields_data } else { Vec::new() },
                    "index": sr.index_mode,
                }
            }))
        }
        Err(e) => error_response(&e.to_string()),
    }
}

async fn hybrid_search(
    state: web::Data<AppState>,
    body: web::Json<HybridSearchRequest>,
) -> HttpResponse {
    let k = body.k.unwrap_or(10);
    if let Err(e) = validate_top_k(&state.limits, k, "k") {
        return limit_bad_request(e);
    }
    let nprobe = body.nprobe.unwrap_or(10);
    let return_fields = body.return_fields.unwrap_or(false);
    let filter = body.where_expr.as_deref();
    let fusion = body.fusion.as_deref().unwrap_or("rrf");
    let candidate_limit = body
        .candidate_limit
        .unwrap_or_else(|| k.saturating_mul(4).max(k));
    if let Err(e) = validate_top_k(&state.limits, candidate_limit, "candidate_limit") {
        return limit_bad_request(e);
    }
    if let Some(ref vector) = body.vector {
        if let Err(e) = validate_request_vector_bytes(&state.limits, 1, vector.len(), "vector") {
            return limit_bad_request(e);
        }
    }

    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| {
            let sr = coll.hybrid_search(
                body.vector.as_deref(),
                body.text.as_deref(),
                k,
                filter,
                body.text_fields.as_deref(),
                fusion,
                body.vector_weight.unwrap_or(1.0),
                body.text_weight.unwrap_or(1.0),
                body.rrf_k.unwrap_or(60.0),
                candidate_limit,
                nprobe,
            )?;
            let fields_data = if return_fields && !sr.ids.is_empty() {
                coll.retrieve_fields(&sr.ids)?
            } else {
                Vec::new()
            };
            Ok((sr, fields_data))
        },
    );

    match result {
        Ok((sr, fields_data)) => {
            let ids: Vec<i64> = sr.ids.iter().map(|&id| id as i64).collect();
            ApiResponse::success(serde_json::json!({
                "database_name": body.database_name,
                "collection_name": body.collection_name,
                "items": {
                    "k": k,
                    "ids": ids,
                    "scores": sr.distances,
                    "fields": if return_fields { fields_data } else { Vec::new() },
                    "index": sr.index_mode,
                    "fusion": fusion,
                }
            }))
        }
        Err(e) => error_response(&e.to_string()),
    }
}

async fn batch_search(
    state: web::Data<AppState>,
    body: web::Json<BatchSearchRequest>,
) -> HttpResponse {
    let k = body.k.unwrap_or(10);
    if let Err(e) = validate_top_k(&state.limits, k, "k") {
        return limit_bad_request(e);
    }
    if let Err(e) = validate_batch_vectors(&state.limits, body.vectors.len(), "queries") {
        return limit_bad_request(e);
    }
    let query_dim = body.vectors.first().map(|v| v.len()).unwrap_or(0);
    for vector in &body.vectors {
        if vector.len() != query_dim {
            return bad_request("all batch query vectors must have the same dimension");
        }
    }
    if let Err(e) =
        validate_request_vector_bytes(&state.limits, body.vectors.len(), query_dim, "queries")
    {
        return limit_bad_request(e);
    }
    let nprobe = body.nprobe.unwrap_or(10);
    let return_fields = body.return_fields.unwrap_or(false);
    let filter = body.where_expr.clone();

    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }
    let result = state.manager.with_database(&body.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&body.collection_name, 0, 100_000)?;
        let coll = coll_arc.read();

        // Flatten vectors for batch_search
        let dim = query_dim;
        let mut flat: Vec<f32> = Vec::with_capacity(body.vectors.len() * dim);
        for v in &body.vectors {
            flat.extend_from_slice(v);
        }

        let results = coll.batch_search(&flat, body.vectors.len(), k, filter.as_deref(), nprobe)?;

        let mut all_results = Vec::with_capacity(results.len());
        for sr in &results {
            let ids: Vec<i64> = sr.ids.iter().map(|&id| id as i64).collect();
            let fields_data = if return_fields && !sr.ids.is_empty() {
                coll.retrieve_fields(&sr.ids)?
            } else {
                Vec::new()
            };
            all_results.push(serde_json::json!({
                "ids": ids,
                "scores": sr.distances,
                "fields": fields_data,
            }));
        }

        Ok(all_results)
    });

    match result {
        Ok(results) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "results": results
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn commit(state: web::Data<AppState>, body: web::Json<CommitRequest>) -> HttpResponse {
    if let Some(ref items) = body.items {
        if let Err(e) = validate_batch_vectors(&state.limits, items.len(), "items") {
            return limit_bad_request(e);
        }
        if let Some(first) = items.first() {
            let dim = first.vector.len();
            if let Err(e) = validate_request_vector_bytes(&state.limits, items.len(), dim, "items")
            {
                return limit_bad_request(e);
            }
            for item in items {
                if item.vector.len() != dim {
                    return bad_request("all vectors in a batch must have the same dimension");
                }
            }
        }
    }

    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }
    let limits = state.limits;
    let result = state.manager.with_database(&body.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&body.collection_name, 0, 100_000)?;

        // Add pending items if any
        if let Some(ref items) = body.items {
            if !items.is_empty() {
                let dim = items[0].vector.len();
                let n_vectors = items.len();
                let vector_bytes = checked_vector_bytes(n_vectors, dim)?;
                let mut flat_vectors: Vec<f32> = Vec::with_capacity(n_vectors * dim);
                let mut fields: Vec<HashMap<String, serde_json::Value>> =
                    Vec::with_capacity(n_vectors);

                let mut requested_ids: Vec<Option<u64>> = Vec::with_capacity(n_vectors);
                for item in items.iter() {
                    flat_vectors.extend_from_slice(&item.vector);
                    fields.push(item.field.clone().unwrap_or_default());
                    requested_ids.push(item.id);
                }

                let has_fields = fields.iter().any(|f| !f.is_empty());
                let mut coll = coll_arc.write();
                let mut next_id = coll.max_id().map(|max_id| max_id + 1).unwrap_or(0);
                let mut reserved = HashSet::with_capacity(n_vectors);
                let mut item_ids: Vec<u64> = Vec::with_capacity(n_vectors);
                for opt_id in &requested_ids {
                    if let Some(id) = opt_id {
                        reserved.insert(*id);
                        item_ids.push(*id);
                    } else {
                        while coll.is_id_exists(next_id) || reserved.contains(&next_id) {
                            next_id += 1;
                        }
                        item_ids.push(next_id);
                        reserved.insert(next_id);
                        next_id += 1;
                    }
                }
                validate_collection_insert(&limits, &coll, n_vectors as u64, vector_bytes)?;
                if has_fields {
                    coll.add_items(&flat_vectors, n_vectors, &item_ids, Some(&fields))?;
                } else {
                    coll.add_items(&flat_vectors, n_vectors, &item_ids, None)?;
                }
            }
        }

        let coll = coll_arc.read();
        coll.commit()?;
        Ok(())
    });

    match result {
        Ok(()) => {
            // Return 200 directly (no async task needed in Rust)
            ApiResponse::success(serde_json::json!({
                "status": "Success",
                "result": {
                    "database_name": body.database_name,
                    "collection_name": body.collection_name
                }
            }))
        }
        Err(e) => error_response(&e.to_string()),
    }
}

async fn flush(state: web::Data<AppState>, body: web::Json<CollectionRequest>) -> HttpResponse {
    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| coll.flush(),
    );

    match result {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn checkpoint(
    state: web::Data<AppState>,
    body: web::Json<CollectionRequest>,
) -> HttpResponse {
    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| coll.checkpoint(),
    );

    match result {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn close_collection(
    state: web::Data<AppState>,
    body: web::Json<CollectionRequest>,
) -> HttpResponse {
    let result = with_collection_mut(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| coll.close(),
    );

    match result {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn collection_shape(
    state: web::Data<AppState>,
    body: web::Json<CollectionRequest>,
) -> HttpResponse {
    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| coll.shape(),
    );

    match result {
        Ok((rows, dim)) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "shape": [rows, dim]
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn is_collection_exists(
    state: web::Data<AppState>,
    body: web::Json<CollectionRequest>,
) -> HttpResponse {
    match state
        .manager
        .collection_exists(&body.database_name, &body.collection_name)
    {
        Ok(exists) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "exists": exists
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn get_collection_config(
    state: web::Data<AppState>,
    body: web::Json<CollectionRequest>,
) -> HttpResponse {
    match state.manager.get_collection_configs(&body.database_name) {
        Ok(configs) => {
            if let Some(config) = configs.get(&body.collection_name) {
                ApiResponse::success(serde_json::json!({
                    "database_name": body.database_name,
                    "collection_name": body.collection_name,
                    "config": config
                }))
            } else {
                bad_request(&format!(
                    "Collection '{}' not found in config",
                    body.collection_name
                ))
            }
        }
        Err(e) => error_response(&e.to_string()),
    }
}

async fn update_collection_description(
    state: web::Data<AppState>,
    body: web::Json<UpdateDescriptionRequest>,
) -> HttpResponse {
    match state.manager.update_collection_description(
        &body.database_name,
        &body.collection_name,
        Some(&body.description),
    ) {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "description": body.description
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn update_description(
    state: web::Data<AppState>,
    body: web::Json<UpdateDescriptionRequest>,
) -> HttpResponse {
    // Same as update_collection_description (backward compat)
    update_collection_description(state, body).await
}

async fn show_collections_details(
    state: web::Data<AppState>,
    body: web::Json<DatabaseRequest>,
) -> HttpResponse {
    let db_name = match &body.database_name {
        Some(name) => name.as_str(),
        None => return bad_request("Missing required parameter: database_name"),
    };

    let configs = match state.manager.get_collection_configs(db_name) {
        Ok(c) => c,
        Err(e) => return error_response(&e.to_string()),
    };

    let collections = match state.manager.show_collections(db_name) {
        Ok(c) => c,
        Err(e) => return error_response(&e.to_string()),
    };

    if let Err(e) = state.manager.get_or_open_database(db_name) {
        return error_response(&e.to_string());
    }

    let mut details: HashMap<String, serde_json::Value> = HashMap::new();
    for name in &collections {
        // Single lock acquisition per collection for both shape and index_mode
        let info = state.manager.with_database(db_name, |engine| {
            let coll_arc = engine.get_or_open_collection(name, 0, 100_000)?;
            let coll = coll_arc.read();
            let shape = coll.shape()?;
            let idx_mode = coll.get_index_mode().map(|s| s.to_string());
            Ok((shape, idx_mode))
        });

        let (shape, index_mode) = match info {
            Ok(((rows, dim), idx)) => (Ok((rows, dim)), Ok(idx)),
            Err(e) => (Err(e.to_string()), Err(String::new())),
        };

        let config = configs.get(name);
        let mut detail = serde_json::json!({
            "name": name,
        });

        if let Ok((rows, dim)) = shape {
            detail["shape"] = serde_json::json!([rows, dim]);
        }
        if let Ok(Some(mode)) = index_mode {
            detail["index_mode"] = serde_json::json!(mode);
        }
        if let Some(cfg) = config {
            detail["dim"] = serde_json::json!(cfg.dim);
            if let Some(ref desc) = cfg.description {
                detail["description"] = serde_json::json!(desc);
            }
        }

        details.insert(name.clone(), detail);
    }

    ApiResponse::success(serde_json::json!({
        "database_name": db_name,
        "collections": details
    }))
}

async fn build_index(
    state: web::Data<AppState>,
    body: web::Json<BuildIndexRequest>,
) -> HttpResponse {
    let index_mode = body.index_mode.as_deref().unwrap_or("FLAT");
    let metrics = Arc::clone(&state.metrics);

    let result = with_collection_mut(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| {
            let n_vectors = coll.shape().map(|(n, _)| n).unwrap_or(0);
            metrics.track_index_build(n_vectors, || coll.build_index(index_mode))
        },
    );

    match result {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "index_mode": index_mode
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn remove_index(
    state: web::Data<AppState>,
    body: web::Json<CollectionRequest>,
) -> HttpResponse {
    let result = with_collection_mut(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| coll.remove_index(),
    );

    match result {
        Ok(()) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn head(state: web::Data<AppState>, body: web::Json<HeadTailRequest>) -> HttpResponse {
    let n = body.n.unwrap_or(5);
    if let Err(e) = validate_top_k(&state.limits, n, "n") {
        return limit_bad_request(e);
    }

    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| {
            let (data, user_ids, fields) = coll.head(n)?;
            let dim_ = coll.meta.dimension;
            let n_vecs = if dim_ > 0 { data.len() / dim_ } else { 0 };

            let vectors: Vec<Vec<f32>> = (0..n_vecs)
                .map(|i| data[i * dim_..(i + 1) * dim_].to_vec())
                .collect();
            let ids: Vec<i64> = user_ids.iter().map(|&id| id as i64).collect();

            Ok((vectors, ids, fields))
        },
    );

    match result {
        Ok((vectors, ids, fields)) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "head": [vectors, ids, fields]
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn tail(state: web::Data<AppState>, body: web::Json<HeadTailRequest>) -> HttpResponse {
    let n = body.n.unwrap_or(5);
    if let Err(e) = validate_top_k(&state.limits, n, "n") {
        return limit_bad_request(e);
    }

    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| {
            let (data, user_ids, fields) = coll.tail(n)?;
            let dim_ = coll.meta.dimension;
            let n_vecs = if dim_ > 0 { data.len() / dim_ } else { 0 };

            let vectors: Vec<Vec<f32>> = (0..n_vecs)
                .map(|i| data[i * dim_..(i + 1) * dim_].to_vec())
                .collect();
            let ids: Vec<i64> = user_ids.iter().map(|&id| id as i64).collect();

            Ok((vectors, ids, fields))
        },
    );

    match result {
        Ok((vectors, ids, fields)) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "tail": [vectors, ids, fields]
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn get_collection_path(
    state: web::Data<AppState>,
    body: web::Json<CollectionRequest>,
) -> HttpResponse {
    let path = state
        .manager
        .root_path()
        .join(&body.database_name)
        .join(&body.collection_name);

    ApiResponse::success(serde_json::json!({
        "database_name": body.database_name,
        "collection_name": body.collection_name,
        "collection_path": path.to_string_lossy()
    }))
}

async fn query(state: web::Data<AppState>, body: web::Json<QueryRequest>) -> HttpResponse {
    let return_ids_only = body.return_ids_only.unwrap_or(false);
    if let Some(ref filter_ids) = body.filter_ids {
        if let Err(e) = validate_batch_vectors(&state.limits, filter_ids.len(), "filter_ids") {
            return limit_bad_request(e);
        }
    }

    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| {
            let filter_expr = body.where_expr.as_deref();

            // Fast path: single query for both IDs + fields when filtering by expression
            if let Some(expr) = filter_expr {
                if return_ids_only {
                    let ids = coll.query_fields(expr)?;
                    return Ok(serde_json::json!(ids
                        .iter()
                        .map(|&id| id as i64)
                        .collect::<Vec<i64>>()));
                }
                let (ids, fields) = coll.query_with_fields(expr)?;
                let records: Vec<serde_json::Value> = ids
                    .iter()
                    .zip(fields.iter())
                    .map(|(&id, f)| {
                        let mut rec = f.clone();
                        rec.insert("id".to_string(), serde_json::json!(id as i64));
                        serde_json::json!(rec)
                    })
                    .collect();
                return Ok(serde_json::json!(records));
            }

            let ids = if let Some(ref filter_ids) = body.filter_ids {
                filter_ids.clone()
            } else {
                // Return all IDs
                let (n, _) = coll.shape()?;
                (0..n).collect()
            };

            if return_ids_only {
                Ok(serde_json::json!(ids
                    .iter()
                    .map(|&id| id as i64)
                    .collect::<Vec<i64>>()))
            } else {
                let fields = coll.retrieve_fields(&ids)?;
                let records: Vec<serde_json::Value> = ids
                    .iter()
                    .zip(fields.iter())
                    .map(|(&id, f)| {
                        let mut rec = f.clone();
                        rec.insert("id".to_string(), serde_json::json!(id as i64));
                        serde_json::json!(rec)
                    })
                    .collect();
                Ok(serde_json::json!(records))
            }
        },
    );

    match result {
        Ok(result_data) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "result": result_data
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn query_vectors(
    state: web::Data<AppState>,
    body: web::Json<QueryVectorsRequest>,
) -> HttpResponse {
    if let Some(ref filter_ids) = body.filter_ids {
        if let Err(e) = validate_batch_vectors(&state.limits, filter_ids.len(), "filter_ids") {
            return limit_bad_request(e);
        }
    }

    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| {
            let filter_expr = body.where_expr.as_deref();
            let dim = coll.meta.dimension;

            let ids = if let Some(expr) = filter_expr {
                coll.query_fields(expr)?
            } else if let Some(ref filter_ids) = body.filter_ids {
                filter_ids.clone()
            } else {
                let (n, _) = coll.shape()?;
                (0..n).collect()
            };

            let (flat_data, fields) = coll.read_vectors_by_ids(&ids)?;
            let n_vecs = if dim > 0 { flat_data.len() / dim } else { 0 };
            let vectors: Vec<Vec<f32>> = (0..n_vecs)
                .map(|i| flat_data[i * dim..(i + 1) * dim].to_vec())
                .collect();
            let id_list: Vec<i64> = ids.iter().map(|&id| id as i64).collect();

            Ok((vectors, id_list, fields))
        },
    );

    match result {
        Ok((vectors, ids, fields)) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "result": [vectors, ids, fields]
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn read_by_only_id(
    state: web::Data<AppState>,
    body: web::Json<ReadByIdRequest>,
) -> HttpResponse {
    // Parse id field: can be single int or list of ints
    let ids: Vec<u64> = match &body.id {
        serde_json::Value::Number(n) => {
            if let Some(id) = n.as_u64() {
                vec![id]
            } else {
                return bad_request("Invalid id");
            }
        }
        serde_json::Value::Array(arr) => {
            let mut ids = Vec::with_capacity(arr.len());
            for v in arr {
                if let Some(id) = v.as_u64() {
                    ids.push(id);
                } else {
                    return bad_request("Invalid id in list");
                }
            }
            ids
        }
        _ => return bad_request("id must be an integer or list of integers"),
    };
    if let Err(e) = validate_batch_vectors(&state.limits, ids.len(), "ids") {
        return limit_bad_request(e);
    }

    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| {
            let dim = coll.meta.dimension;
            let (flat_data, fields) = coll.read_vectors_by_ids(&ids)?;
            let n_vecs = if dim > 0 { flat_data.len() / dim } else { 0 };
            let vectors: Vec<Vec<f32>> = (0..n_vecs)
                .map(|i| flat_data[i * dim..(i + 1) * dim].to_vec())
                .collect();
            let id_list: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
            Ok((vectors, id_list, fields))
        },
    );

    match result {
        Ok((vectors, ids, fields)) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "item": [vectors, ids, fields]
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn delete_items(
    state: web::Data<AppState>,
    body: web::Json<DeleteItemsRequest>,
) -> HttpResponse {
    if let Err(e) = validate_batch_vectors(&state.limits, body.ids.len(), "ids") {
        return limit_bad_request(e);
    }
    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| coll.delete_items(&body.ids),
    );
    match result {
        Ok(_) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "status": "ok"
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn restore_items(
    state: web::Data<AppState>,
    body: web::Json<DeleteItemsRequest>,
) -> HttpResponse {
    if let Err(e) = validate_batch_vectors(&state.limits, body.ids.len(), "ids") {
        return limit_bad_request(e);
    }
    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| coll.restore_items(&body.ids),
    );
    match result {
        Ok(_) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "status": "ok"
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn list_deleted_ids(
    state: web::Data<AppState>,
    body: web::Json<CollectionRequest>,
) -> HttpResponse {
    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| Ok(coll.list_deleted_ids()),
    );
    match result {
        Ok(ids) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "ids": ids
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn search_range(
    state: web::Data<AppState>,
    body: web::Json<SearchRangeRequest>,
) -> HttpResponse {
    if let Err(e) = validate_top_k(&state.limits, body.max_results, "max_results") {
        return limit_bad_request(e);
    }
    if let Err(e) = validate_request_vector_bytes(&state.limits, 1, body.vector.len(), "vector") {
        return limit_bad_request(e);
    }
    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| coll.search_range(&body.vector, body.threshold, body.max_results),
    );
    match result {
        Ok((ids, dists)) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "result": {"ids": ids, "distances": dists}
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn list_fields(
    state: web::Data<AppState>,
    body: web::Json<CollectionRequest>,
) -> HttpResponse {
    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| coll.list_fields(),
    );

    match result {
        Ok(fields) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "fields": fields
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn index_mode(
    state: web::Data<AppState>,
    body: web::Json<CollectionRequest>,
) -> HttpResponse {
    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| Ok(coll.get_index_mode().map(|s| s.to_string())),
    );

    match result {
        Ok(mode) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "index_mode": mode
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn is_id_exists(
    state: web::Data<AppState>,
    body: web::Json<IsIdExistsRequest>,
) -> HttpResponse {
    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| Ok(coll.is_id_exists(body.id)),
    );

    match result {
        Ok(exists) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "is_id_exists": exists
        })),
        Err(_) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "is_id_exists": false
        })),
    }
}

async fn max_id(state: web::Data<AppState>, body: web::Json<MaxIdRequest>) -> HttpResponse {
    let result = with_collection(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| Ok(coll.max_id().map(|id| id as i64).unwrap_or(-1)),
    );

    match result {
        Ok(max) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "max_id": max
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn compact(state: web::Data<AppState>, body: web::Json<CollectionRequest>) -> HttpResponse {
    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }
    let result = state.manager.with_database(&body.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&body.collection_name, 0, 100_000)?;
        let mut coll = coll_arc.write();
        let removed = coll.compact()?;
        Ok(removed)
    });

    match result {
        Ok(removed) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "vectors_removed": removed
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

async fn collection_stats(
    state: web::Data<AppState>,
    body: web::Json<CollectionRequest>,
) -> HttpResponse {
    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }
    let result = state.manager.with_database(&body.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&body.collection_name, 0, 100_000)?;
        let coll = coll_arc.read();
        let (n_vectors, dimension) = coll.shape()?;
        let n_tombstoned = coll.list_deleted_ids().len() as u64;
        let n_live = n_vectors.saturating_sub(n_tombstoned);
        let index_mode = coll.get_index_mode().unwrap_or("none").to_string();
        let max_id = coll.max_id().map(|id| id as i64).unwrap_or(-1);
        Ok(serde_json::json!({
            "n_vectors": n_vectors,
            "n_live": n_live,
            "n_tombstoned": n_tombstoned,
            "dimension": dimension,
            "index_mode": index_mode,
            "max_id": max_id
        }))
    });

    match result {
        Ok(stats) => ApiResponse::success(serde_json::json!({
            "database_name": body.database_name,
            "collection_name": body.collection_name,
            "stats": stats
        })),
        Err(e) => error_response(&e.to_string()),
    }
}

// ─── Compact binary helpers ──────────────────────────────────────────────────

/// Encode search result as compact binary:
/// [4B n_results (u32 LE)]
/// [n × 8B ids (u64 LE)]
/// [n × 4B distances (f32 LE)]
/// [4B fields_json_len (u32 LE)]
/// [fields_json_len bytes UTF-8 JSON]
fn encode_search_result_binary(
    ids: &[u64],
    distances: &[f32],
    fields: &[HashMap<String, serde_json::Value>],
) -> Vec<u8> {
    let n = ids.len();
    let fields_json = if fields.is_empty() {
        Vec::new()
    } else {
        serde_json::to_vec(fields).unwrap_or_default()
    };
    // Pre-allocate: 4 + n*8 + n*4 + 4 + fields_json.len()
    let cap = 4 + n * 8 + n * 4 + 4 + fields_json.len();
    let mut buf = Vec::with_capacity(cap);
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    for &id in ids {
        buf.extend_from_slice(&id.to_le_bytes());
    }
    for &d in distances {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&(fields_json.len() as u32).to_le_bytes());
    buf.extend_from_slice(&fields_json);
    buf
}

/// Encode head/tail result as compact binary:
/// [4B n_vectors (u32 LE)]
/// [4B dim (u32 LE)]
/// [n × dim × 4B vectors (f32 LE)]
/// [n × 8B ids (u64 LE)]
/// [4B fields_json_len (u32 LE)]
/// [fields_json_len bytes UTF-8 JSON]
fn encode_vectors_binary(
    data: &[f32],
    dim: usize,
    ids: &[u64],
    fields: &[HashMap<String, serde_json::Value>],
) -> Vec<u8> {
    let n = ids.len();
    let fields_json = if fields.is_empty() {
        Vec::new()
    } else {
        serde_json::to_vec(fields).unwrap_or_default()
    };
    let cap = 4 + 4 + data.len() * 4 + n * 8 + 4 + fields_json.len();
    let mut buf = Vec::with_capacity(cap);
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    // vectors as raw f32 LE
    for &v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for &id in ids {
        buf.extend_from_slice(&id.to_le_bytes());
    }
    buf.extend_from_slice(&(fields_json.len() as u32).to_le_bytes());
    buf.extend_from_slice(&fields_json);
    buf
}

fn binary_error(msg: &str) -> HttpResponse {
    HttpResponse::InternalServerError()
        .content_type("text/plain")
        .body(msg.to_string())
}

fn binary_bad_request(msg: &str) -> HttpResponse {
    HttpResponse::BadRequest()
        .content_type("text/plain")
        .body(msg.to_string())
}

// ─── Binary search endpoint ─────────────────────────────────────────────────

async fn search_binary(
    state: web::Data<AppState>,
    query: web::Query<SearchBinaryQuery>,
    body: web::Bytes,
) -> HttpResponse {
    let dim = query.dim;
    let expected = match checked_vector_bytes(1, dim).and_then(|bytes| {
        usize::try_from(bytes).map_err(|_| {
            LynseError::InvalidArgument("Expected binary query size exceeds usize".into())
        })
    }) {
        Ok(bytes) => bytes,
        Err(e) => return binary_bad_request(&e.to_string()),
    };
    if body.len() != expected {
        return binary_bad_request(&format!("Expected {} bytes, got {}", expected, body.len()));
    }

    let vector: &[f32] = unsafe { std::slice::from_raw_parts(body.as_ptr() as *const f32, dim) };

    let k = query.k.unwrap_or(10);
    if let Err(e) = validate_top_k(&state.limits, k, "k") {
        return binary_bad_request(&e.to_string());
    }
    if let Err(e) = validate_request_vector_bytes(&state.limits, 1, dim, "query") {
        return binary_bad_request(&e.to_string());
    }
    let nprobe = query.nprobe.unwrap_or(10);
    let approx = query.approx.unwrap_or(false);
    let eps = query.eps.unwrap_or(1e-4);
    let return_fields = query.return_fields.unwrap_or(false);
    let filter = query.where_expr.as_deref();
    let vector_field = query.vector_field.as_deref().unwrap_or("default");

    if let Err(e) = state.manager.get_or_open_database(&query.database_name) {
        return binary_error(&e.to_string());
    }
    let result = state.manager.with_database(&query.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&query.collection_name, 0, 100_000)?;
        let coll = coll_arc.read();
        let sr = if vector_field == "default" {
            coll.search(vector, k, filter, nprobe, approx, eps)?
        } else {
            coll.search_vector_field(vector_field, vector, k, filter)?
        };

        let fields_data = if return_fields && !sr.ids.is_empty() {
            coll.retrieve_fields(&sr.ids)?
        } else {
            Vec::new()
        };

        Ok(encode_search_result_binary(
            &sr.ids,
            &sr.distances,
            &fields_data,
        ))
    });

    match result {
        Ok(buf) => HttpResponse::Ok()
            .content_type("application/octet-stream")
            .body(buf),
        Err(e) => binary_error(&e.to_string()),
    }
}

// ─── Binary batch search endpoint ───────────────────────────────────────────

async fn batch_search_binary(
    state: web::Data<AppState>,
    query: web::Query<BatchSearchBinaryQuery>,
    body: web::Bytes,
) -> HttpResponse {
    let dim = query.dim;
    let nq = query.n_queries;
    if let Err(e) = validate_batch_vectors(&state.limits, nq, "n_queries") {
        return binary_bad_request(&e.to_string());
    }
    let expected = match checked_vector_bytes(nq, dim).and_then(|bytes| {
        usize::try_from(bytes).map_err(|_| {
            LynseError::InvalidArgument("Expected binary payload size exceeds usize".into())
        })
    }) {
        Ok(bytes) => bytes,
        Err(e) => return binary_bad_request(&e.to_string()),
    };
    if body.len() != expected {
        return binary_bad_request(&format!(
            "Expected {} bytes ({} queries × {} dim × 4), got {}",
            expected,
            nq,
            dim,
            body.len()
        ));
    }

    let flat: &[f32] = unsafe { std::slice::from_raw_parts(body.as_ptr() as *const f32, nq * dim) };

    let k = query.k.unwrap_or(10);
    if let Err(e) = validate_top_k(&state.limits, k, "k") {
        return binary_bad_request(&e.to_string());
    }
    if let Err(e) = validate_request_vector_bytes(&state.limits, nq, dim, "queries") {
        return binary_bad_request(&e.to_string());
    }
    let nprobe = query.nprobe.unwrap_or(10);
    let return_fields = query.return_fields.unwrap_or(false);
    let filter = query.where_expr.clone();

    if let Err(e) = state.manager.get_or_open_database(&query.database_name) {
        return binary_error(&e.to_string());
    }
    let result = state.manager.with_database(&query.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&query.collection_name, 0, 100_000)?;
        let coll = coll_arc.read();

        let results = coll.batch_search(flat, nq, k, filter.as_deref(), nprobe)?;

        // Encode: [4B n_queries][per-query binary block]
        let mut buf = Vec::with_capacity(4 + results.len() * (4 + k * 12 + 4));
        buf.extend_from_slice(&(results.len() as u32).to_le_bytes());

        for sr in &results {
            let fields_data = if return_fields && !sr.ids.is_empty() {
                coll.retrieve_fields(&sr.ids)?
            } else {
                Vec::new()
            };
            let block = encode_search_result_binary(&sr.ids, &sr.distances, &fields_data);
            buf.extend_from_slice(&block);
        }

        Ok(buf)
    });

    match result {
        Ok(buf) => HttpResponse::Ok()
            .content_type("application/octet-stream")
            .body(buf),
        Err(e) => binary_error(&e.to_string()),
    }
}

// ─── Binary head/tail endpoints ─────────────────────────────────────────────

async fn head_binary(
    state: web::Data<AppState>,
    query: web::Query<HeadTailBinaryQuery>,
) -> HttpResponse {
    let n = query.n.unwrap_or(5);
    if let Err(e) = validate_top_k(&state.limits, n, "n") {
        return binary_bad_request(&e.to_string());
    }

    let result = with_collection(
        &state.manager,
        &query.database_name,
        &query.collection_name,
        |coll| {
            let (data, ids, fields) = coll.head(n)?;
            let dim = coll.meta.dimension;
            Ok(encode_vectors_binary(&data, dim, &ids, &fields))
        },
    );

    match result {
        Ok(buf) => HttpResponse::Ok()
            .content_type("application/octet-stream")
            .body(buf),
        Err(e) => binary_error(&e.to_string()),
    }
}

async fn tail_binary(
    state: web::Data<AppState>,
    query: web::Query<HeadTailBinaryQuery>,
) -> HttpResponse {
    let n = query.n.unwrap_or(5);
    if let Err(e) = validate_top_k(&state.limits, n, "n") {
        return binary_bad_request(&e.to_string());
    }

    let result = with_collection(
        &state.manager,
        &query.database_name,
        &query.collection_name,
        |coll| {
            let (data, ids, fields) = coll.tail(n)?;
            let dim = coll.meta.dimension;
            Ok(encode_vectors_binary(&data, dim, &ids, &fields))
        },
    );

    match result {
        Ok(buf) => HttpResponse::Ok()
            .content_type("application/octet-stream")
            .body(buf),
        Err(e) => binary_error(&e.to_string()),
    }
}

// ─── Server configuration ────────────────────────────────────────────────────

fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg
        // Root
        .route("/", web::get().to(index))
        .route("/healthz", web::get().to(healthz))
        .route("/readyz", web::get().to(readyz))
        .route("/metrics", web::get().to(metrics))
        .route("/openapi.json", web::get().to(openapi_json))
        // Database operations
        .route("/create_database", web::post().to(create_database))
        .route("/drop_database", web::post().to(drop_database))
        .route("/snapshot_database", web::post().to(snapshot_database))
        .route("/restore_database", web::post().to(restore_database))
        .route("/database_exists", web::post().to(database_exists))
        .route("/list_databases", web::get().to(list_databases))
        .route("/delete_database", web::post().to(delete_database))
        // Collection operations
        .route("/required_collection", web::post().to(required_collection))
        .route("/drop_collection", web::post().to(drop_collection))
        .route("/snapshot_collection", web::post().to(snapshot_collection))
        .route("/export_collection", web::post().to(export_collection))
        .route("/restore_collection", web::post().to(restore_collection))
        .route("/import_collection", web::post().to(import_collection))
        .route("/show_collections", web::post().to(show_collections))
        .route("/add_item", web::post().to(add_item))
        .route("/bulk_add_items", web::post().to(bulk_add_items))
        .route("/create_vector_field", web::post().to(create_vector_field))
        .route("/list_vector_fields", web::post().to(list_vector_fields))
        .route("/add_named_vectors", web::post().to(add_named_vectors))
        .route("/add_sparse_vectors", web::post().to(add_sparse_vectors))
        .route(
            "/build_vector_field_index",
            web::post().to(build_vector_field_index),
        )
        .route(
            "/remove_vector_field_index",
            web::post().to(remove_vector_field_index),
        )
        .route("/upsert_items", web::post().to(upsert_items))
        .route("/bulk_add_binary", web::post().to(bulk_add_binary))
        .route("/search", web::post().to(search))
        .route("/search_profile", web::post().to(search_profile))
        .route("/text_search", web::post().to(text_search))
        .route("/sparse_search", web::post().to(sparse_search))
        .route("/hybrid_search", web::post().to(hybrid_search))
        .route("/search_binary", web::post().to(search_binary))
        .route("/batch_search", web::post().to(batch_search))
        .route("/batch_search_binary", web::post().to(batch_search_binary))
        .route("/commit", web::post().to(commit))
        .route("/flush", web::post().to(flush))
        .route("/checkpoint", web::post().to(checkpoint))
        .route("/close_collection", web::post().to(close_collection))
        .route("/collection_shape", web::post().to(collection_shape))
        .route(
            "/is_collection_exists",
            web::post().to(is_collection_exists),
        )
        .route(
            "/get_collection_config",
            web::post().to(get_collection_config),
        )
        .route(
            "/update_collection_description",
            web::post().to(update_collection_description),
        )
        .route("/update_description", web::post().to(update_description))
        .route(
            "/show_collections_details",
            web::post().to(show_collections_details),
        )
        .route("/build_index", web::post().to(build_index))
        .route("/remove_index", web::post().to(remove_index))
        .route("/head", web::post().to(head))
        .route("/head_binary", web::get().to(head_binary))
        .route("/tail", web::post().to(tail))
        .route("/tail_binary", web::get().to(tail_binary))
        .route("/get_collection_path", web::post().to(get_collection_path))
        .route("/query", web::post().to(query))
        .route("/query_vectors", web::post().to(query_vectors))
        .route("/list_fields", web::post().to(list_fields))
        .route("/index_mode", web::post().to(index_mode))
        .route("/is_id_exists", web::post().to(is_id_exists))
        .route("/max_id", web::post().to(max_id))
        .route("/read_by_only_id", web::post().to(read_by_only_id))
        .route("/delete_items", web::post().to(delete_items))
        .route("/restore_items", web::post().to(restore_items))
        .route("/list_deleted_ids", web::post().to(list_deleted_ids))
        .route("/search_range", web::post().to(search_range))
        .route("/compact", web::post().to(compact))
        .route("/stats", web::post().to(collection_stats));
}

/// Start the HTTP server. This function blocks until the server is stopped.
/// Pass `api_key = Some("secret")` to enable HTTP Basic Auth / Bearer token auth.
pub fn run_server(
    host: &str,
    port: u16,
    root_path: &str,
    api_key: Option<String>,
) -> std::io::Result<()> {
    init_rust_logger();

    let manager = Arc::new(
        DatabaseManager::new(std::path::Path::new(root_path))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?,
    );
    let api_key = Arc::new(api_key);
    let metrics = Arc::new(HttpMetrics::new());
    let start_time_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let runtime_cfg = load_server_runtime_config();

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        log::info!(
            "Starting LynseDB server on {}:{} (workers={}, keep_alive={}s, request_timeout={}s, json_limit={}B, payload_limit={}B, slow_query_warn={}ms, max_top_k={}, max_batch_vectors={}, max_collection_vectors={}, max_collection_vector_bytes={}, audit_log={})",
            host,
            port,
            runtime_cfg.workers,
            runtime_cfg.keep_alive_secs,
            runtime_cfg.client_request_timeout_secs,
            runtime_cfg.json_limit_bytes,
            runtime_cfg.payload_limit_bytes,
            runtime_cfg.slow_query_warn_ms,
            runtime_cfg.limits.max_top_k,
            runtime_cfg.limits.max_batch_vectors,
            runtime_cfg.limits.max_collection_vectors,
            runtime_cfg.limits.max_collection_vector_bytes,
            runtime_cfg.audit_log_enabled
        );

        let app_manager = Arc::clone(&manager);
        let app_metrics = Arc::clone(&metrics);
        let app_runtime_cfg = runtime_cfg;
        let server = HttpServer::new(move || {
            let key_clone: Option<String> = (*api_key).clone();
            let metrics_clone = Arc::clone(&app_metrics);
            App::new()
                .app_data(web::Data::new(AppState {
                    manager: Arc::clone(&app_manager),
                    start_time_unix_seconds,
                    metrics: Arc::clone(&metrics_clone),
                    limits: app_runtime_cfg.limits,
                }))
                .app_data(web::JsonConfig::default().limit(app_runtime_cfg.json_limit_bytes))
                .app_data(web::PayloadConfig::default().limit(app_runtime_cfg.payload_limit_bytes))
                .wrap(ApiKeyAuth {
                    api_key: key_clone,
                    metrics: metrics_clone,
                    slow_query_warn_ms: app_runtime_cfg.slow_query_warn_ms,
                    audit_log_enabled: app_runtime_cfg.audit_log_enabled,
                })
                .configure(configure_routes)
        })
        .workers(runtime_cfg.workers)
        .keep_alive(std::time::Duration::from_secs(runtime_cfg.keep_alive_secs))
        .client_request_timeout(std::time::Duration::from_secs(
            runtime_cfg.client_request_timeout_secs,
        ))
        .bind((host, port))?
        .disable_signals()
        .run();

        let handle = server.handle();
        tokio::spawn(async move {
            wait_for_shutdown_signal().await;
            handle.stop(true).await;
        });

        let result = server.await;
        if let Err(e) = manager.checkpoint_open_collections() {
            log::error!("Failed to checkpoint collections during shutdown: {}", e);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ));
        }
        result
    })
}

async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        let mut sigterm = signal(SignalKind::terminate()).ok();
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = async {
                if let Some(signal) = sigterm.as_mut() {
                    signal.recv().await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {}
        }
    }

    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Start the HTTP server in a background thread. Returns the thread handle.
pub fn start_server_background(
    host: String,
    port: u16,
    root_path: String,
    api_key: Option<String>,
) -> std::thread::JoinHandle<std::io::Result<()>> {
    std::thread::spawn(move || run_server(&host, port, &root_path, api_key))
}

#[cfg(test)]
mod tests {
    use super::{
        audit_action_for_path, collect_storage_usage, is_public_endpoint, is_query_endpoint,
        openapi_spec, validate_batch_vectors, validate_collection_insert, validate_top_k, AppState,
        HttpMetrics, RequestOutcome, ServerLimits,
    };
    use actix_web::{body::to_bytes, http::StatusCode, web};
    use std::fs;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::TempDir;

    use crate::engine::DatabaseManager;

    fn test_limits() -> ServerLimits {
        ServerLimits {
            max_top_k: 10,
            max_batch_vectors: 10,
            max_collection_vectors: 10,
            max_collection_vector_bytes: 1024,
        }
    }

    #[test]
    fn public_endpoint_whitelist_matches_probe_routes() {
        assert!(is_public_endpoint("/"));
        assert!(is_public_endpoint("/healthz"));
        assert!(is_public_endpoint("/readyz"));
        assert!(!is_public_endpoint("/metrics"));
        assert!(!is_public_endpoint("/openapi.json"));
        assert!(!is_public_endpoint("/search"));
    }

    #[test]
    fn openapi_spec_includes_server_and_query_routes() {
        let spec = openapi_spec();
        let paths = spec["paths"].as_object().expect("paths object");
        assert!(paths.contains_key("/openapi.json"));
        assert!(paths.contains_key("/search"));
        assert!(paths.contains_key("/search_profile"));
        assert!(paths.contains_key("/text_search"));
        assert!(paths.contains_key("/sparse_search"));
        assert!(paths.contains_key("/hybrid_search"));
        assert!(paths.contains_key("/add_sparse_vectors"));
    }

    #[test]
    fn http_metrics_records_request_counts_and_latency() {
        let metrics = HttpMetrics::new();
        metrics.observe(200, Duration::from_millis(10), RequestOutcome::Normal);
        metrics.observe(401, Duration::from_millis(20), RequestOutcome::Unauthorized);
        metrics.observe(404, Duration::from_millis(50), RequestOutcome::Normal);
        metrics.observe(
            503,
            Duration::from_millis(200),
            RequestOutcome::HandlerFailure,
        );

        assert_eq!(metrics.request_total(), 4);
        assert_eq!(metrics.request_error_total.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.status_2xx.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.status_4xx.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.status_5xx.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.error_client_4xx_total.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.error_server_5xx_total.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.error_unauthorized_total.load(Ordering::Relaxed), 1);
        assert_eq!(
            metrics.error_handler_failure_total.load(Ordering::Relaxed),
            1
        );
        assert!(metrics.request_duration_sum_seconds() >= 0.28);

        let p50 = metrics.request_latency_quantile_seconds(0.50);
        let p90 = metrics.request_latency_quantile_seconds(0.90);
        let p99 = metrics.request_latency_quantile_seconds(0.99);
        assert!(p50 > 0.0);
        assert!(p90 >= p50);
        assert!(p99 >= p90);
    }

    #[test]
    fn storage_usage_counts_disk_wal_and_index_bytes() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("plain.bin"), vec![1u8; 3]).unwrap();
        fs::create_dir_all(tmp.path().join("db").join("col").join("wal")).unwrap();
        fs::write(
            tmp.path()
                .join("db")
                .join("col")
                .join("wal")
                .join("col.000000.wal"),
            vec![2u8; 5],
        )
        .unwrap();
        fs::create_dir_all(tmp.path().join("db").join("col").join("index")).unwrap();
        fs::write(
            tmp.path()
                .join("db")
                .join("col")
                .join("index")
                .join("index-test.bin"),
            vec![3u8; 7],
        )
        .unwrap();

        let usage = collect_storage_usage(tmp.path());
        assert!(usage.disk_bytes >= 15);
        assert_eq!(usage.wal_bytes, 5);
        assert_eq!(usage.index_bytes, 7);
    }

    #[test]
    fn http_metrics_track_index_builds() {
        let metrics = HttpMetrics::new();
        let ok: std::result::Result<(), ()> =
            metrics.track_index_build(42, || std::result::Result::Ok(()));
        assert!(ok.is_ok());
        assert_eq!(metrics.index_build_started_total.load(Ordering::Relaxed), 1);
        assert_eq!(
            metrics.index_build_completed_total.load(Ordering::Relaxed),
            1
        );
        assert_eq!(metrics.index_build_in_progress.load(Ordering::Relaxed), 0);
        assert_eq!(
            metrics.index_build_current_vectors.load(Ordering::Relaxed),
            0
        );
        assert_eq!(metrics.index_build_last_vectors.load(Ordering::Relaxed), 42);
        assert_eq!(
            metrics
                .index_build_last_progress_ppm
                .load(Ordering::Relaxed),
            1_000_000
        );

        let err: std::result::Result<(), ()> =
            metrics.track_index_build(7, || std::result::Result::Err(()));
        assert!(err.is_err());
        assert_eq!(metrics.index_build_started_total.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.index_build_failed_total.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.index_build_in_progress.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.index_build_last_vectors.load(Ordering::Relaxed), 7);
    }

    #[test]
    fn query_endpoint_classifier_covers_search_routes() {
        assert!(is_query_endpoint("/search"));
        assert!(is_query_endpoint("/batch_search_binary"));
        assert!(is_query_endpoint("/query_vectors"));
        assert!(!is_query_endpoint("/metrics"));
        assert!(!is_query_endpoint("/build_index"));
    }

    #[actix_rt::test]
    async fn metrics_endpoint_exports_resource_and_index_build_metrics() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("db").join("col").join("wal")).unwrap();
        fs::write(
            tmp.path()
                .join("db")
                .join("col")
                .join("wal")
                .join("col.000000.wal"),
            vec![4u8; 5],
        )
        .unwrap();

        let manager = Arc::new(DatabaseManager::new(tmp.path()).unwrap());
        let http_metrics = Arc::new(HttpMetrics::new());
        let ok: std::result::Result<(), ()> =
            http_metrics.track_index_build(3, || std::result::Result::Ok(()));
        assert!(ok.is_ok());

        let response = super::metrics(web::Data::new(AppState {
            manager,
            start_time_unix_seconds: 123,
            metrics: http_metrics,
            limits: test_limits(),
        }))
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body()).await.unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(body.contains("lynsedb_storage_wal_bytes 5"));
        assert!(body.contains("lynsedb_process_resident_memory_bytes"));
        assert!(body.contains("lynsedb_index_builds_total{status=\"started\"} 1"));
        assert!(body.contains("lynsedb_index_build_last_vectors 3"));
        assert!(body.contains("lynsedb_index_build_progress_ratio{scope=\"last\"} 1"));
    }

    #[test]
    fn server_limits_reject_oversized_top_k_batch_and_collection_growth() {
        let limits = ServerLimits {
            max_top_k: 2,
            max_batch_vectors: 3,
            max_collection_vectors: 2,
            max_collection_vector_bytes: 32,
        };

        assert!(validate_top_k(&limits, 3, "k").is_err());
        assert!(validate_top_k(&limits, 2, "k").is_ok());
        assert!(validate_batch_vectors(&limits, 4, "items").is_err());
        assert!(validate_batch_vectors(&limits, 3, "items").is_ok());

        let tmp = TempDir::new().unwrap();
        let mut coll = crate::engine::Collection::open(tmp.path(), "col", 4, 100).unwrap();
        coll.add_items(
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0,
            ],
            2,
            &[1, 2],
            None,
        )
        .unwrap();

        assert!(validate_collection_insert(&limits, &coll, 1, 16).is_err());
    }

    #[test]
    fn audit_action_classifier_only_marks_mutations() {
        assert_eq!(audit_action_for_path("/build_index"), Some("build_index"));
        assert_eq!(audit_action_for_path("/add_item"), Some("add_item"));
        assert_eq!(audit_action_for_path("/search"), None);
        assert_eq!(audit_action_for_path("/metrics"), None);
    }
}
