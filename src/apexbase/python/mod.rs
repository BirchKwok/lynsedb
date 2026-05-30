//! Python bindings via PyO3
//!
//! ApexStorage is the storage implementation using on-demand reading.

mod bindings;

pub use bindings::ApexStorageImpl as ApexStorage;
