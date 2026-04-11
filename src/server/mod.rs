//! HTTP server module for LynseDB.
//!
//! Provides a RESTful API using actix-web, replacing the Python Flask server.
//! All endpoints are compatible with the existing HTTPClient Python class.

use std::collections::{HashMap, HashSet};
use std::future::{ready, Future, Ready};
use std::pin::Pin;
use std::sync::Arc;

use actix_web::body::EitherBody;
use actix_web::dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform};
use actix_web::{web, App, HttpResponse, HttpServer};
use serde::{Deserialize, Serialize};

use crate::engine::DatabaseManager;
use crate::error::LynseError;

// ─── Shared application state ────────────────────────────────────────────────

pub struct AppState {
    pub manager: Arc<DatabaseManager>,
    pub api_key: Option<String>,
}

// ─── Basic Auth / Bearer Token Middleware ─────────────────────────────────────

/// Middleware factory. Wraps every request with an optional API key check.
/// Accepts both `Authorization: Bearer <key>` and `Authorization: Basic base64(:key)` headers.
pub struct ApiKeyAuth {
    pub api_key: Option<String>,
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
        }))
    }
}

pub struct ApiKeyAuthMiddleware<S> {
    service: S,
    api_key: Option<String>,
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
        if let Some(ref key) = self.api_key {
            if !is_authorized(&req, key) {
                return Box::pin(async move {
                    let response = HttpResponse::Unauthorized()
                        .insert_header(("WWW-Authenticate", "Basic realm=\"LynseDB\""))
                        .json(serde_json::json!({"error": "Unauthorized"}));
                    Ok(req.into_response(response.map_into_right_body()))
                });
            }
        }
        let fut = self.service.call(req);
        Box::pin(async move {
            let res = fut.await?;
            Ok(res.map_into_left_body())
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
struct SearchRequest {
    database_name: String,
    collection_name: String,
    vector: Vec<f32>,
    k: Option<usize>,
    #[serde(rename = "where")]
    where_expr: Option<String>,
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
    k: Option<usize>,
    #[serde(rename = "where")]
    where_expr: Option<String>,
    return_fields: Option<bool>,
    nprobe: Option<usize>,
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

// ─── Database operation handlers ─────────────────────────────────────────────

async fn index() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "success",
        "message": "LynseDB HTTP API"
    }))
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
    let expected_bytes = n_vectors * dim * 4;

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
    // bulk_add_binary has no user IDs in the binary protocol — assign sequential from current max
    let result = state.manager.with_database(&query.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&query.collection_name, dim, 100_000)?;
        let mut coll = coll_arc.write();
        let start_id = coll
            .max_id()
            .map(|m| m + 1)
            .unwrap_or_else(|| coll.shape().map(|(n, _)| n).unwrap_or(0));
        let seq_ids: Vec<u64> = (start_id..start_id + n_vectors as u64).collect();
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
    let mut flat_vectors: Vec<f32> = Vec::with_capacity(n_vectors * dim);
    let mut fields: Vec<HashMap<String, serde_json::Value>> = Vec::with_capacity(n_vectors);
    let mut ids: Vec<Option<u64>> = Vec::with_capacity(n_vectors);

    for item in &body.items {
        flat_vectors.extend_from_slice(&item.vector);
        fields.push(item.field.clone().unwrap_or_default());
        ids.push(item.id);
    }

    let has_fields = fields.iter().any(|f| !f.is_empty());

    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }
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

async fn upsert_items(
    state: web::Data<AppState>,
    body: web::Json<BulkAddItemsRequest>,
) -> HttpResponse {
    if body.items.is_empty() {
        return bad_request("No items provided");
    }

    let dim = body.items[0].vector.len();

    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }

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
    let nprobe = body.nprobe.unwrap_or(10);
    let return_fields = body.return_fields.unwrap_or(false);
    let filter = body.where_expr.as_deref();

    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }
    let result = state.manager.with_database(&body.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&body.collection_name, 0, 100_000)?;
        let coll = coll_arc.read();
        let sr = coll.search(&body.vector, k, filter, nprobe)?;

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
                    "fields": fields,
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
        let dim = if body.vectors.is_empty() {
            0
        } else {
            body.vectors[0].len()
        };
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
    if let Err(e) = state.manager.get_or_open_database(&body.database_name) {
        return error_response(&e.to_string());
    }
    let result = state.manager.with_database(&body.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&body.collection_name, 0, 100_000)?;

        // Add pending items if any
        if let Some(ref items) = body.items {
            if !items.is_empty() {
                let dim = items[0].vector.len();
                let n_vectors = items.len();
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

    let result = with_collection_mut(
        &state.manager,
        &body.database_name,
        &body.collection_name,
        |coll| coll.build_index(index_mode),
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

// ─── Binary search endpoint ─────────────────────────────────────────────────

async fn search_binary(
    state: web::Data<AppState>,
    query: web::Query<SearchBinaryQuery>,
    body: web::Bytes,
) -> HttpResponse {
    let dim = query.dim;
    let expected = dim * 4;
    if body.len() != expected {
        return HttpResponse::BadRequest()
            .content_type("text/plain")
            .body(format!("Expected {} bytes, got {}", expected, body.len()));
    }

    let vector: &[f32] = unsafe { std::slice::from_raw_parts(body.as_ptr() as *const f32, dim) };

    let k = query.k.unwrap_or(10);
    let nprobe = query.nprobe.unwrap_or(10);
    let return_fields = query.return_fields.unwrap_or(false);
    let filter = query.where_expr.as_deref();

    if let Err(e) = state.manager.get_or_open_database(&query.database_name) {
        return binary_error(&e.to_string());
    }
    let result = state.manager.with_database(&query.database_name, |engine| {
        let coll_arc = engine.get_or_open_collection(&query.collection_name, 0, 100_000)?;
        let coll = coll_arc.read();
        let sr = coll.search(vector, k, filter, nprobe)?;

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
    let expected = nq * dim * 4;
    if body.len() != expected {
        return HttpResponse::BadRequest()
            .content_type("text/plain")
            .body(format!(
                "Expected {} bytes ({} queries × {} dim × 4), got {}",
                expected,
                nq,
                dim,
                body.len()
            ));
    }

    let flat: &[f32] = unsafe { std::slice::from_raw_parts(body.as_ptr() as *const f32, nq * dim) };

    let k = query.k.unwrap_or(10);
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
        // Database operations
        .route("/create_database", web::post().to(create_database))
        .route("/drop_database", web::post().to(drop_database))
        .route("/database_exists", web::post().to(database_exists))
        .route("/list_databases", web::get().to(list_databases))
        .route("/delete_database", web::post().to(delete_database))
        // Collection operations
        .route("/required_collection", web::post().to(required_collection))
        .route("/drop_collection", web::post().to(drop_collection))
        .route("/show_collections", web::post().to(show_collections))
        .route("/add_item", web::post().to(add_item))
        .route("/bulk_add_items", web::post().to(bulk_add_items))
        .route("/upsert_items", web::post().to(upsert_items))
        .route("/bulk_add_binary", web::post().to(bulk_add_binary))
        .route("/search", web::post().to(search))
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
    let manager = Arc::new(
        DatabaseManager::new(std::path::Path::new(root_path))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?,
    );
    let api_key = Arc::new(api_key);

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        log::info!("Starting LynseDB server on {}:{}", host, port);

        HttpServer::new(move || {
            let key_clone: Option<String> = (*api_key).clone();
            App::new()
                .app_data(web::Data::new(AppState {
                    manager: Arc::clone(&manager),
                    api_key: key_clone.clone(),
                }))
                .app_data(web::JsonConfig::default().limit(1024 * 1024 * 256)) // 256MB JSON limit
                .app_data(web::PayloadConfig::default().limit(1024 * 1024 * 512)) // 512MB raw payload limit
                .wrap(ApiKeyAuth { api_key: key_clone })
                .configure(configure_routes)
        })
        .workers(num_cpus::get().max(2))
        .keep_alive(std::time::Duration::from_secs(75))
        .client_request_timeout(std::time::Duration::from_secs(300))
        .bind((host, port))?
        .run()
        .await
    })
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
