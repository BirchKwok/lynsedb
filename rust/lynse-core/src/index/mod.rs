//! Vector index module.
//!
//! Implements: Flat, IVF, HNSW, DiskANN indices with multiple distance metrics
//! and quantization support. Designed for billion-scale datasets.

pub mod flat;
pub mod hnsw;
pub mod diskann;
pub mod ivf;

use crate::distance::DistanceMetric;
use crate::error::{LynseError, Result};
use crate::quantizer::QuantizerType;
use serde::{Deserialize, Serialize};

/// Common index configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub index_type: IndexType,
    pub distance_metric: DistanceMetric,
    pub quantizer_type: QuantizerType,
    pub dimension: usize,
    pub params: IndexParams,
}

/// Index type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexType {
    Flat,
    HNSW,
    DiskANN,
    IVF,
}

/// Index-specific parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexParams {
    Flat,
    HNSW {
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        max_level: Option<usize>,
    },
    DiskANN {
        r: usize,
        l: usize,
        alpha: f32,
        max_degree: usize,
    },
    IVF {
        n_centroids: usize,
        nprobe: usize,
    },
}

impl Default for IndexParams {
    fn default() -> Self {
        IndexParams::Flat
    }
}

/// Trait for all vector indices.
pub trait VectorIndex: Send + Sync {
    /// Build the index from vectors and optional IDs.
    fn build(
        &mut self,
        vectors: &[f32],
        n_vectors: usize,
        dim: usize,
        ids: Option<&[u64]>,
    ) -> Result<()>;

    /// Search for k nearest neighbors.
    /// Returns (ids, distances).
    fn search(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>)>;

    /// Delete vectors by IDs.
    fn delete(&mut self, ids: &[u64]) -> Result<()>;

    /// Insert additional vectors (incremental).
    fn insert(&mut self, vectors: &[f32], n_vectors: usize, dim: usize, ids: &[u64]) -> Result<()>;

    /// Get the number of indexed vectors.
    fn len(&self) -> usize;

    /// Check if the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the index has been built/trained.
    fn is_trained(&self) -> bool;

    /// Get index configuration.
    fn config(&self) -> &IndexConfig;

    /// Serialize the index to bytes.
    fn serialize(&self) -> Result<Vec<u8>>;

    /// Deserialize the index from bytes.
    fn deserialize(&mut self, data: &[u8]) -> Result<()>;

    /// Get the index name string (matches Python API naming).
    fn name(&self) -> String;
}

/// Search parameters.
#[derive(Debug, Clone)]
pub struct SearchParams {
    pub k: usize,
    pub nprobe: usize,           // for IVF
    pub ef_search: Option<usize>, // for HNSW
    pub subset_indices: Option<Vec<u64>>, // filter IDs
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            k: 10,
            nprobe: 10,
            ef_search: None,
            subset_indices: None,
        }
    }
}

/// Index alias map matching Python's `Indexer._INDEX_ALIAS`.
pub fn resolve_index_type(alias: &str) -> Option<(IndexType, DistanceMetric, QuantizerType)> {
    let _lower = alias.to_lowercase();
    match alias {
        // Flat indices
        "FLAT" | "Flat-IP" | "flat-ip" => {
            Some((IndexType::Flat, DistanceMetric::InnerProduct, QuantizerType::None))
        }
        "Flat-L2" | "flat-l2" => {
            Some((IndexType::Flat, DistanceMetric::L2Squared, QuantizerType::None))
        }
        "Flat-Cos" | "flat-cosine" => {
            Some((IndexType::Flat, DistanceMetric::Cosine, QuantizerType::None))
        }
        "Flat-IP-SQ8" | "flat-ip-sq8" => {
            Some((IndexType::Flat, DistanceMetric::InnerProduct, QuantizerType::Scalar))
        }
        "Flat-L2-SQ8" | "flat-l2-sq8" => {
            Some((IndexType::Flat, DistanceMetric::L2Squared, QuantizerType::Scalar))
        }
        "Flat-Cos-SQ8" | "flat-cosine-sq8" => {
            Some((IndexType::Flat, DistanceMetric::Cosine, QuantizerType::Scalar))
        }
        "Flat-Jaccard-Binary" | "flat-jaccard" => {
            Some((IndexType::Flat, DistanceMetric::Jaccard, QuantizerType::Binary))
        }
        "Flat-Hamming-Binary" | "flat-hamming" => {
            Some((IndexType::Flat, DistanceMetric::Hamming, QuantizerType::Binary))
        }

        // HNSW indices
        "HNSW" | "HNSW-IP" | "hnsw-ip" => {
            Some((IndexType::HNSW, DistanceMetric::InnerProduct, QuantizerType::None))
        }
        "HNSW-L2" | "hnsw-l2" => {
            Some((IndexType::HNSW, DistanceMetric::L2Squared, QuantizerType::None))
        }
        "HNSW-Cos" | "hnsw-cosine" => {
            Some((IndexType::HNSW, DistanceMetric::Cosine, QuantizerType::None))
        }
        "HNSW-IP-SQ8" | "hnsw-ip-sq8" => {
            Some((IndexType::HNSW, DistanceMetric::InnerProduct, QuantizerType::Scalar))
        }
        "HNSW-L2-SQ8" | "hnsw-l2-sq8" => {
            Some((IndexType::HNSW, DistanceMetric::L2Squared, QuantizerType::Scalar))
        }
        "HNSW-Cos-SQ8" | "hnsw-cosine-sq8" => {
            Some((IndexType::HNSW, DistanceMetric::Cosine, QuantizerType::Scalar))
        }

        // DiskANN indices
        "DiskANN" | "DiskANN-IP" | "diskann-ip" => {
            Some((IndexType::DiskANN, DistanceMetric::InnerProduct, QuantizerType::None))
        }
        "DiskANN-L2" | "diskann-l2" => {
            Some((IndexType::DiskANN, DistanceMetric::L2Squared, QuantizerType::None))
        }
        "DiskANN-Cos" | "diskann-cosine" => {
            Some((IndexType::DiskANN, DistanceMetric::Cosine, QuantizerType::None))
        }
        "DiskANN-IP-SQ8" | "diskann-ip-sq8" => {
            Some((IndexType::DiskANN, DistanceMetric::InnerProduct, QuantizerType::Scalar))
        }
        "DiskANN-L2-SQ8" | "diskann-l2-sq8" => {
            Some((IndexType::DiskANN, DistanceMetric::L2Squared, QuantizerType::Scalar))
        }
        "DiskANN-Cos-SQ8" | "diskann-cosine-sq8" => {
            Some((IndexType::DiskANN, DistanceMetric::Cosine, QuantizerType::Scalar))
        }

        // IVF indices
        "IVF" | "IVF-IP" | "ivf-ip" => {
            Some((IndexType::IVF, DistanceMetric::InnerProduct, QuantizerType::None))
        }
        "IVF-L2" | "ivf-l2" => {
            Some((IndexType::IVF, DistanceMetric::L2Squared, QuantizerType::None))
        }
        "IVF-Cos" | "ivf-cosine" => {
            Some((IndexType::IVF, DistanceMetric::Cosine, QuantizerType::None))
        }
        "IVF-IP-SQ8" | "ivf-ip-sq8" => {
            Some((IndexType::IVF, DistanceMetric::InnerProduct, QuantizerType::Scalar))
        }
        "IVF-L2-SQ8" | "ivf-l2-sq8" => {
            Some((IndexType::IVF, DistanceMetric::L2Squared, QuantizerType::Scalar))
        }
        "IVF-Cos-SQ8" | "ivf-cosine-sq8" => {
            Some((IndexType::IVF, DistanceMetric::Cosine, QuantizerType::Scalar))
        }
        "IVF-Jaccard-Binary" | "ivf-jaccard" => {
            Some((IndexType::IVF, DistanceMetric::Jaccard, QuantizerType::Binary))
        }
        "IVF-Hamming-Binary" | "ivf-hamming" => {
            Some((IndexType::IVF, DistanceMetric::Hamming, QuantizerType::Binary))
        }

        _ => None,
    }
}

/// Create an index from a type alias string.
pub fn create_index(alias: &str) -> Result<Box<dyn VectorIndex>> {
    let (index_type, metric, quant) = resolve_index_type(alias).ok_or_else(|| {
        LynseError::InvalidArgument(format!("Unknown index type: {}", alias))
    })?;

    match index_type {
        IndexType::Flat => Ok(Box::new(flat::FlatIndex::new(metric, quant))),
        IndexType::HNSW => Ok(Box::new(hnsw::HNSWIndex::new(
            metric,
            quant,
            16,   // M
            128,  // ef_construction (matches usearch default)
            50,   // ef_search
            None, // max_level
        ))),
        IndexType::DiskANN => Ok(Box::new(diskann::DiskANNIndex::new(
            metric,
            quant,
            64,   // R
            100,  // L
            1.2,  // alpha
            128,  // max_degree
        ))),
        IndexType::IVF => Ok(Box::new(ivf::IVFIndex::new(
            metric,
            quant,
            256, // n_centroids
            32,  // nprobe
        ))),
    }
}
