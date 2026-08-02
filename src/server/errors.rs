//! HTTP error mapping shared by JSON and binary endpoints.

use actix_web::{http::StatusCode, HttpResponse};

use crate::error::LynseError;

pub(super) fn internal_error(msg: &str) -> HttpResponse {
    HttpResponse::InternalServerError()
        .json(serde_json::json!({"error": msg, "code": "internal_error"}))
}

fn status_and_code(error: &LynseError) -> (StatusCode, &'static str) {
    match error {
        LynseError::InvalidArgument(_) | LynseError::DimensionMismatch { .. } => {
            (StatusCode::BAD_REQUEST, "invalid_argument")
        }
        LynseError::EmptyDatabase => (StatusCode::BAD_REQUEST, "empty_database"),
        LynseError::CollectionNotFound(_) | LynseError::DatabaseNotFound(_) => {
            (StatusCode::NOT_FOUND, "not_found")
        }
        LynseError::CollectionAlreadyExists(_) => (StatusCode::CONFLICT, "already_exists"),
        LynseError::IndexNotBuilt => (StatusCode::CONFLICT, "index_not_built"),
        LynseError::QuantizerNotTrained => (StatusCode::CONFLICT, "quantizer_not_trained"),
        LynseError::Io(_)
        | LynseError::Storage(_)
        | LynseError::Index(_)
        | LynseError::Serialization(_)
        | LynseError::ApexBase(_)
        | LynseError::NumPack(_)
        | LynseError::Python(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
    }
}

pub(super) fn lynse_error(error: &LynseError) -> HttpResponse {
    let (status, code) = status_and_code(error);
    HttpResponse::build(status).json(serde_json::json!({
        "error": error.to_string(),
        "code": code,
    }))
}

pub(super) fn bad_request(msg: &str) -> HttpResponse {
    HttpResponse::BadRequest().json(serde_json::json!({
        "error": msg,
        "code": "invalid_argument"
    }))
}

pub(super) fn limit_bad_request(error: LynseError) -> HttpResponse {
    bad_request(&error.to_string())
}

pub(super) fn binary_lynse_error(error: &LynseError) -> HttpResponse {
    let (status, _) = status_and_code(error);
    HttpResponse::build(status)
        .content_type("text/plain")
        .body(error.to_string())
}

pub(super) fn binary_bad_request(msg: &str) -> HttpResponse {
    HttpResponse::BadRequest()
        .content_type("text/plain")
        .body(msg.to_string())
}
