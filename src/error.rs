//! Error types for LynseDB core.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LynseError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    #[error("Collection already exists: {0}")]
    CollectionAlreadyExists(String),

    #[error("Database not found: {0}")]
    DatabaseNotFound(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Empty database")]
    EmptyDatabase,

    #[error("Index not built")]
    IndexNotBuilt,

    #[error("Quantizer not trained")]
    QuantizerNotTrained,

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("ApexBase error: {0}")]
    ApexBase(String),

    #[error("NumPack error: {0}")]
    NumPack(String),

    #[error("Python error: {0}")]
    Python(String),
}

pub type Result<T> = std::result::Result<T, LynseError>;

impl From<LynseError> for pyo3::PyErr {
    fn from(err: LynseError) -> pyo3::PyErr {
        match &err {
            LynseError::DimensionMismatch { .. } => {
                pyo3::exceptions::PyValueError::new_err(err.to_string())
            }
            LynseError::CollectionNotFound(_) | LynseError::DatabaseNotFound(_) => {
                pyo3::exceptions::PyValueError::new_err(err.to_string())
            }
            LynseError::InvalidArgument(_) => {
                pyo3::exceptions::PyValueError::new_err(err.to_string())
            }
            LynseError::EmptyDatabase => pyo3::exceptions::PyValueError::new_err(err.to_string()),
            LynseError::Io(_) => pyo3::exceptions::PyIOError::new_err(err.to_string()),
            _ => pyo3::exceptions::PyRuntimeError::new_err(err.to_string()),
        }
    }
}
