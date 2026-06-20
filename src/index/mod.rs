//! Vector index module.
//!
//! Implements: Flat, IVF, SPANN, HNSW, DiskANN indices with multiple distance
//! metrics and quantization support. Designed for billion-scale datasets.

pub mod diskann;
pub mod flat;
pub mod hnsw;
pub mod ivf;
pub(crate) mod kmeans;
pub mod spann;

use crate::distance::DistanceMetric;
use crate::error::{LynseError, Result};
use crate::quantizer::QuantizerType;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
    SPANN,
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
    SPANN {
        n_centroids: usize,
        nprobe: usize,
        replica_count: usize,
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
    pub nprobe: usize,                         // for IVF/SPANN
    pub ef_search: Option<usize>,              // for HNSW
    pub subset_indices: Option<Arc<Vec<u64>>>, // filter row IDs
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
/// Case-insensitive: all inputs are normalised to uppercase before matching.
pub fn resolve_index_type(alias: &str) -> Option<(IndexType, DistanceMetric, QuantizerType)> {
    let u = alias.to_uppercase();
    match u.as_str() {
        // ── Flat ────────────────────────────────────────────────────────────
        "FLAT-IP" => Some((
            IndexType::Flat,
            DistanceMetric::InnerProduct,
            QuantizerType::None,
        )),
        "FLAT-L2" => Some((
            IndexType::Flat,
            DistanceMetric::L2Squared,
            QuantizerType::None,
        )),
        "FLAT-COS" | "FLAT-COSINE" => {
            Some((IndexType::Flat, DistanceMetric::Cosine, QuantizerType::None))
        }
        "FLAT-IP-SQ8" => Some((
            IndexType::Flat,
            DistanceMetric::InnerProduct,
            QuantizerType::Scalar,
        )),
        "FLAT-L2-SQ8" => Some((
            IndexType::Flat,
            DistanceMetric::L2Squared,
            QuantizerType::Scalar,
        )),
        "FLAT-COS-SQ8" | "FLAT-COSINE-SQ8" => Some((
            IndexType::Flat,
            DistanceMetric::Cosine,
            QuantizerType::Scalar,
        )),
        "FLAT-JACCARD-BINARY" | "FLAT-JACCARD" => Some((
            IndexType::Flat,
            DistanceMetric::Jaccard,
            QuantizerType::Binary,
        )),
        "FLAT-HAMMING-BINARY" | "FLAT-HAMMING" => Some((
            IndexType::Flat,
            DistanceMetric::Hamming,
            QuantizerType::Binary,
        )),

        // ── HNSW ────────────────────────────────────────────────────────────
        "HNSW-IP" => Some((
            IndexType::HNSW,
            DistanceMetric::InnerProduct,
            QuantizerType::None,
        )),
        "HNSW-L2" => Some((
            IndexType::HNSW,
            DistanceMetric::L2Squared,
            QuantizerType::None,
        )),
        "HNSW-COS" | "HNSW-COSINE" => {
            Some((IndexType::HNSW, DistanceMetric::Cosine, QuantizerType::None))
        }
        "HNSW-IP-SQ8" => Some((
            IndexType::HNSW,
            DistanceMetric::InnerProduct,
            QuantizerType::Scalar,
        )),
        "HNSW-L2-SQ8" => Some((
            IndexType::HNSW,
            DistanceMetric::L2Squared,
            QuantizerType::Scalar,
        )),
        "HNSW-COS-SQ8" | "HNSW-COSINE-SQ8" => Some((
            IndexType::HNSW,
            DistanceMetric::Cosine,
            QuantizerType::Scalar,
        )),

        // ── DiskANN ─────────────────────────────────────────────────────────
        "DISKANN-IP" => Some((
            IndexType::DiskANN,
            DistanceMetric::InnerProduct,
            QuantizerType::None,
        )),
        "DISKANN-L2" => Some((
            IndexType::DiskANN,
            DistanceMetric::L2Squared,
            QuantizerType::None,
        )),
        "DISKANN-COS" | "DISKANN-COSINE" => Some((
            IndexType::DiskANN,
            DistanceMetric::Cosine,
            QuantizerType::None,
        )),
        "DISKANN-IP-SQ8" => Some((
            IndexType::DiskANN,
            DistanceMetric::InnerProduct,
            QuantizerType::Scalar,
        )),
        "DISKANN-L2-SQ8" => Some((
            IndexType::DiskANN,
            DistanceMetric::L2Squared,
            QuantizerType::Scalar,
        )),
        "DISKANN-COS-SQ8" | "DISKANN-COSINE-SQ8" => Some((
            IndexType::DiskANN,
            DistanceMetric::Cosine,
            QuantizerType::Scalar,
        )),

        // ── IVF ─────────────────────────────────────────────────────────────
        "IVF-IP" => Some((
            IndexType::IVF,
            DistanceMetric::InnerProduct,
            QuantizerType::None,
        )),
        "IVF-L2" => Some((
            IndexType::IVF,
            DistanceMetric::L2Squared,
            QuantizerType::None,
        )),
        "IVF-COS" | "IVF-COSINE" => {
            Some((IndexType::IVF, DistanceMetric::Cosine, QuantizerType::None))
        }
        "IVF-IP-SQ8" => Some((
            IndexType::IVF,
            DistanceMetric::InnerProduct,
            QuantizerType::Scalar,
        )),
        "IVF-L2-SQ8" => Some((
            IndexType::IVF,
            DistanceMetric::L2Squared,
            QuantizerType::Scalar,
        )),
        "IVF-COS-SQ8" | "IVF-COSINE-SQ8" => Some((
            IndexType::IVF,
            DistanceMetric::Cosine,
            QuantizerType::Scalar,
        )),
        "IVF-JACCARD-BINARY" | "IVF-JACCARD" => Some((
            IndexType::IVF,
            DistanceMetric::Jaccard,
            QuantizerType::Binary,
        )),
        "IVF-HAMMING-BINARY" | "IVF-HAMMING" => Some((
            IndexType::IVF,
            DistanceMetric::Hamming,
            QuantizerType::Binary,
        )),

        // ── SPANN ───────────────────────────────────────────────────────────
        "SPANN-IP" => Some((
            IndexType::SPANN,
            DistanceMetric::InnerProduct,
            QuantizerType::None,
        )),
        "SPANN-L2" => Some((
            IndexType::SPANN,
            DistanceMetric::L2Squared,
            QuantizerType::None,
        )),
        "SPANN-COS" | "SPANN-COSINE" => Some((
            IndexType::SPANN,
            DistanceMetric::Cosine,
            QuantizerType::None,
        )),
        "SPANN-IP-SQ8" => Some((
            IndexType::SPANN,
            DistanceMetric::InnerProduct,
            QuantizerType::Scalar,
        )),
        "SPANN-L2-SQ8" => Some((
            IndexType::SPANN,
            DistanceMetric::L2Squared,
            QuantizerType::Scalar,
        )),
        "SPANN-COS-SQ8" | "SPANN-COSINE-SQ8" => Some((
            IndexType::SPANN,
            DistanceMetric::Cosine,
            QuantizerType::Scalar,
        )),

        _ => resolve_domain_index_type(&u),
    }
}

/// Domain-oriented metrics intentionally start with exact flat search and
/// HNSW. Partition and quantized indexes need metric-specific recall studies
/// before they can be advertised safely.
fn resolve_domain_index_type(alias: &str) -> Option<(IndexType, DistanceMetric, QuantizerType)> {
    let (family, suffix) = alias.split_once('-')?;
    let index_type = match family {
        "FLAT" => IndexType::Flat,
        "HNSW" => IndexType::HNSW,
        _ => return None,
    };

    let metric = DistanceMetric::from_index_mode(alias)?;
    let binary_suffix = suffix.ends_with("-BINARY");
    if metric.is_binary() {
        if index_type != IndexType::Flat {
            return None;
        }
        let accepted = matches!(
            suffix,
            "HAMMING"
                | "HAMMING-BINARY"
                | "JACCARD"
                | "JACCARD-BINARY"
                | "TANIMOTO"
                | "TANIMOTO-BINARY"
                | "DICE"
                | "DICE-BINARY"
                | "SORENSEN"
                | "SORENSEN-BINARY"
                | "SORENSEN-DICE"
                | "SORENSEN-DICE-BINARY"
        );
        return accepted.then_some((index_type, metric, QuantizerType::Binary));
    }

    if binary_suffix {
        return None;
    }
    let accepted = matches!(
        suffix,
        "L1" | "MANHATTAN"
            | "CITYBLOCK"
            | "HAVERSINE"
            | "HAVERSINE-M"
            | "GEO"
            | "CORRELATION"
            | "PEARSON"
            | "HELLINGER"
            | "WASSERSTEIN"
            | "WASSERSTEIN-1D"
            | "WASSERSTEIN1D"
            | "EMD"
    );
    accepted.then_some((index_type, metric, QuantizerType::None))
}

/// Create an index from a type alias string.
pub fn create_index(alias: &str) -> Result<Box<dyn VectorIndex>> {
    create_index_with_options(alias, None)
}

pub fn create_index_with_options(
    alias: &str,
    n_centroids: Option<usize>,
) -> Result<Box<dyn VectorIndex>> {
    let (index_type, metric, quant) = resolve_index_type(alias)
        .ok_or_else(|| LynseError::InvalidArgument(format!("Unknown index type: {}", alias)))?;

    if n_centroids == Some(0) {
        return Err(LynseError::InvalidArgument(
            "n_clusters must be greater than 0".to_string(),
        ));
    }
    if n_centroids.is_some() && !matches!(index_type, IndexType::IVF | IndexType::SPANN) {
        return Err(LynseError::InvalidArgument(
            "n_clusters is only supported for IVF and SPANN indexes".to_string(),
        ));
    }

    match index_type {
        IndexType::Flat => Ok(Box::new(flat::FlatIndex::new(metric, quant))),
        IndexType::HNSW => Ok(Box::new(hnsw::HNSWIndex::new(
            metric, quant, 16,   // M
            128,  // ef_construction (matches usearch default)
            50,   // ef_search
            None, // max_level
        ))),
        IndexType::DiskANN => Ok(Box::new(diskann::DiskANNIndex::new(
            metric, quant, 64,  // R
            100, // L
            1.2, // alpha
            128, // max_degree
        ))),
        IndexType::IVF => Ok(Box::new(ivf::IVFIndex::new(
            metric,
            quant,
            n_centroids.unwrap_or(256),
            32, // nprobe
        ))),
        IndexType::SPANN => Ok(Box::new(spann::SPANNIndex::new(
            metric,
            quant,
            n_centroids.unwrap_or(256),
            32, // nprobe
            spann::DEFAULT_REPLICA_COUNT,
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_spann_aliases() {
        assert_eq!(
            resolve_index_type("SPANN-L2").unwrap(),
            (
                IndexType::SPANN,
                DistanceMetric::L2Squared,
                QuantizerType::None
            )
        );
        assert_eq!(
            resolve_index_type("SPANN-COS-SQ8").unwrap(),
            (
                IndexType::SPANN,
                DistanceMetric::Cosine,
                QuantizerType::Scalar
            )
        );
    }

    #[test]
    fn spann_accepts_n_clusters() {
        let idx = create_index_with_options("SPANN-L2", Some(8)).unwrap();
        assert_eq!(idx.config().index_type, IndexType::SPANN);
    }

    #[test]
    fn bare_index_family_names_are_rejected() {
        for alias in ["FLAT", "HNSW", "DISKANN", "IVF", "SPANN"] {
            assert!(
                resolve_index_type(alias).is_none(),
                "{alias} should require an explicit metric suffix"
            );
            assert!(create_index(alias).is_err());
        }
    }

    #[test]
    fn resolves_domain_metrics_with_explicit_capabilities() {
        assert_eq!(
            resolve_index_type("flat-manhattan"),
            Some((
                IndexType::Flat,
                DistanceMetric::Manhattan,
                QuantizerType::None
            ))
        );
        assert_eq!(
            resolve_index_type("HNSW-PEARSON"),
            Some((
                IndexType::HNSW,
                DistanceMetric::Correlation,
                QuantizerType::None
            ))
        );
        assert_eq!(
            resolve_index_type("FLAT-TANIMOTO-BINARY"),
            Some((
                IndexType::Flat,
                DistanceMetric::Tanimoto,
                QuantizerType::Binary
            ))
        );
        assert!(resolve_index_type("HNSW-TANIMOTO-BINARY").is_none());
        assert!(resolve_index_type("FLAT-HELLINGER-SQ8").is_none());
    }

    #[test]
    fn domain_hnsw_serialization_roundtrip() {
        let dim = 4;
        let vectors = vec![
            1.0, 2.0, 3.0, 4.0, // row 0
            4.0, 3.0, 2.0, 1.0, // row 1
            1.0, 3.0, 5.0, 8.0, // row 2, correlated with row 0
        ];
        let mut built = create_index("HNSW-CORRELATION").unwrap();
        built.build(&vectors, 3, dim, Some(&[10, 11, 12])).unwrap();
        let bytes = built.serialize().unwrap();

        let mut loaded = create_index("HNSW-CORRELATION").unwrap();
        loaded.deserialize(&bytes).unwrap();
        let (ids, distances) = loaded
            .search(
                &vectors[..dim],
                1,
                &SearchParams {
                    ef_search: Some(32),
                    ..SearchParams::default()
                },
            )
            .unwrap();

        assert_eq!(ids, vec![10]);
        assert!(distances[0].abs() < 1e-6);
    }
}
