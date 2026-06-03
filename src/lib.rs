//! LynseDB Rust Core - High-performance vector database engine
//!
//! This crate provides the core engine for LynseDB, a vector database designed
//! for billion-scale vector search. It uses:
//! - ApexBase for structured metadata storage
//! - NumPack for high-performance vector storage
//! - Custom SIMD-accelerated distance computations
//! - Multiple index types: Flat, IVF, HNSW, DiskANN

#[cfg(target_os = "windows")]
compile_error!("Native Windows builds are not supported. Use WSL 2 or Docker on Windows.");

pub mod distance;
pub mod engine;
pub mod error;
pub mod index;
pub mod python;
pub mod quantizer;
pub mod server;
pub mod storage;

use pyo3::prelude::*;

/// Python module definition
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(m)?;
    Ok(())
}
