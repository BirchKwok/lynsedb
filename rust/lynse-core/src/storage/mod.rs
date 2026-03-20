//! Storage layer module.
//!
//! - `vector_store`: NumPack-based vector storage for billion-scale datasets
//! - `field_store`: ApexBase-based structured metadata storage
//! - `wal`: Write-Ahead Log for crash-safe vector ingestion
//! - `flat_mmap`: Ultra-fast flat binary storage with persistent mmap

pub mod vector_store;
pub mod field_store;
pub mod wal;
pub mod bitset;
pub mod flat_mmap;
pub mod ivf_flat_mmap;
