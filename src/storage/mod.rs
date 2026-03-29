//! Storage layer module.
//!
//! - `vector_store`: Single-file mmap vector storage (powered by `flat_mmap`)
//! - `field_store`: ApexBase-based structured metadata storage
//! - `wal`: Write-Ahead Log for crash-safe vector ingestion
//! - `flat_mmap`: Ultra-fast flat binary storage with persistent mmap

pub mod vector_store;
pub mod field_store;
pub mod wal;
pub mod bitset;
pub mod flat_mmap;
pub mod ivf_flat_mmap;
pub mod pq_mmap;
pub mod rabitq_mmap;
pub mod polarvec_mmap;
