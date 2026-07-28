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
use std::path::Path;
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

    /// Build from an owned vector buffer (avoids an extra copy for large N).
    /// Default clones into [`build`](Self::build); DiskANN layered overrides.
    fn build_owned(
        &mut self,
        vectors: Vec<f32>,
        n_vectors: usize,
        dim: usize,
        ids: Option<Vec<u64>>,
    ) -> Result<()> {
        self.build(&vectors, n_vectors, dim, ids.as_deref())
    }

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

    /// Delete vectors by IDs, optionally supplying their f32 vectors (row-major).
    ///
    /// DiskANN IP-delete uses the vectors for in-place edge repair when the
    /// index no longer keeps full-precision data (layered mode). Default
    /// ignores `vectors` and calls [`delete`](Self::delete).
    fn delete_with_vectors(&mut self, ids: &[u64], vectors: &[f32]) -> Result<()> {
        let _ = vectors;
        self.delete(ids)
    }

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

    /// Bind collection data directory (DiskANN loads `diskann/` sidecars here).
    fn attach_data_dir(&mut self, _dir: &Path) -> Result<()> {
        Ok(())
    }

    /// When true, engine should call [`search_candidates`] then exact-rescore via VectorStore.
    fn uses_store_rescore(&self) -> bool {
        false
    }

    /// Oversampled candidate row indices for store-side exact rescore.
    fn search_candidates(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<Vec<u32>> {
        let (ids, _) = self.search(query, k, params)?;
        Ok(ids.into_iter().map(|id| id as u32).collect())
    }

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
            // 0 = use index build-time default (IVF/SPANN nprobe, HNSW ef_search).
            nprobe: 0,
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
        "DISKANN-IP-PQ" | "DISKANN-IP-PQ8" | "DISKANN-IP-PQ16" => Some((
            IndexType::DiskANN,
            DistanceMetric::InnerProduct,
            QuantizerType::Product,
        )),
        "DISKANN-L2-PQ" | "DISKANN-L2-PQ8" | "DISKANN-L2-PQ16" => Some((
            IndexType::DiskANN,
            DistanceMetric::L2Squared,
            QuantizerType::Product,
        )),
        "DISKANN-COS-PQ" | "DISKANN-COSINE-PQ" => Some((
            IndexType::DiskANN,
            DistanceMetric::Cosine,
            QuantizerType::Product,
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
    // Canberra and Bray-Curtis are exposed as exact metrics until their ANN
    // recall characteristics have been validated independently.
    if index_type == IndexType::HNSW
        && matches!(
            metric,
            DistanceMetric::Canberra | DistanceMetric::BrayCurtis
        )
    {
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
            | "JENSEN-SHANNON"
            | "JENSENSHANNON"
            | "JS"
            | "CHEBYSHEV"
            | "CHEBYCHEV"
            | "LINF"
            | "CANBERRA"
            | "BRAY-CURTIS"
            | "BRAYCURTIS"
    );
    accepted.then_some((index_type, metric, QuantizerType::None))
}

/// Tunable parameters for [`create_index_with_build_options`], exposed as
/// Python/HTTP `build_index(**kwargs)` / `params`.
///
/// Defaults applied when a field is `None`:
/// - IVF/SPANN: `n_clusters=256`, `nprobe=32`, SPANN `replica_count=1`
/// - HNSW: `m=16`, `ef_construction=128`, `ef_search=50`, `max_level` uncapped
/// - DiskANN: `r=16`, `l=64`, `alpha=1.2`, `max_degree=r`
///
/// Unknown keys are rejected; keys that do not apply to the selected index
/// family are ignored (so shared kwargs dicts work across modes).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexBuildOptions {
    /// IVF / SPANN coarse centroids (`n_centroids` accepted as alias). Default: 256.
    #[serde(default, alias = "n_centroids")]
    pub n_clusters: Option<usize>,
    /// HNSW max neighbors per layer (M). Default: 16.
    #[serde(default)]
    pub m: Option<usize>,
    /// HNSW construction beam width. Default: 128.
    #[serde(default)]
    pub ef_construction: Option<usize>,
    /// HNSW default search beam (overridable per-query via nprobe/ef). Default: 50.
    #[serde(default)]
    pub ef_search: Option<usize>,
    /// Optional HNSW max level cap (omit for uncapped / internal default).
    #[serde(default)]
    pub max_level: Option<usize>,
    /// DiskANN / Vamana target out-degree (R). Default: 16.
    #[serde(default)]
    pub r: Option<usize>,
    /// DiskANN search/build beam (L); build may further cap L_build. Default: 64.
    #[serde(default)]
    pub l: Option<usize>,
    /// DiskANN robust-prune alpha (≥ 1.0). Default: 1.2.
    #[serde(default)]
    pub alpha: Option<f32>,
    /// DiskANN hard degree cap (defaults to R when omitted).
    #[serde(default)]
    pub max_degree: Option<usize>,
    /// IVF / SPANN default nprobe stored on the index. Default: 32.
    #[serde(default)]
    pub nprobe: Option<usize>,
    /// SPANN boundary replica count. Default: 1.
    #[serde(default)]
    pub replica_count: Option<usize>,
}

impl IndexBuildOptions {
    /// Parse from a JSON object (Python kwargs / HTTP `params`).
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| {
            LynseError::InvalidArgument("index build options must be a JSON object".into())
        })?;
        for key in obj.keys() {
            if !Self::is_known_key(key) {
                return Err(LynseError::InvalidArgument(format!(
                    "unknown index build parameter '{}'; supported keys: {}",
                    key,
                    Self::known_keys_csv()
                )));
            }
        }
        serde_json::from_value(value.clone()).map_err(|e| {
            LynseError::InvalidArgument(format!("invalid index build parameter value: {}", e))
        })
    }

    pub fn from_n_clusters(n_clusters: Option<usize>) -> Self {
        Self {
            n_clusters,
            ..Self::default()
        }
    }

    fn is_known_key(key: &str) -> bool {
        matches!(
            key,
            "n_clusters"
                | "n_centroids"
                | "m"
                | "ef_construction"
                | "ef_search"
                | "max_level"
                | "r"
                | "l"
                | "alpha"
                | "max_degree"
                | "nprobe"
                | "replica_count"
        )
    }

    fn known_keys_csv() -> &'static str {
        "n_clusters, n_centroids, m, ef_construction, ef_search, max_level, r, l, alpha, max_degree, nprobe, replica_count"
    }

    /// Drop fields that do not apply to `index_type` (keeps shared-kwargs loops safe).
    pub fn filtered_for(&self, index_type: IndexType) -> Self {
        match index_type {
            IndexType::Flat => Self::default(),
            IndexType::HNSW => Self {
                m: self.m,
                ef_construction: self.ef_construction,
                ef_search: self.ef_search,
                max_level: self.max_level,
                ..Self::default()
            },
            IndexType::DiskANN => Self {
                r: self.r,
                l: self.l,
                alpha: self.alpha,
                max_degree: self.max_degree,
                ..Self::default()
            },
            IndexType::IVF => Self {
                n_clusters: self.n_clusters,
                nprobe: self.nprobe,
                ..Self::default()
            },
            IndexType::SPANN => Self {
                n_clusters: self.n_clusters,
                nprobe: self.nprobe,
                replica_count: self.replica_count,
                ..Self::default()
            },
        }
    }

    fn validate_positive(name: &str, value: Option<usize>) -> Result<()> {
        if value == Some(0) {
            return Err(LynseError::InvalidArgument(format!(
                "{} must be greater than 0",
                name
            )));
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        Self::validate_positive("n_clusters", self.n_clusters)?;
        Self::validate_positive("m", self.m)?;
        Self::validate_positive("ef_construction", self.ef_construction)?;
        Self::validate_positive("ef_search", self.ef_search)?;
        Self::validate_positive("r", self.r)?;
        Self::validate_positive("l", self.l)?;
        Self::validate_positive("max_degree", self.max_degree)?;
        Self::validate_positive("nprobe", self.nprobe)?;
        Self::validate_positive("replica_count", self.replica_count)?;
        if let Some(alpha) = self.alpha {
            if !(alpha.is_finite() && alpha >= 1.0) {
                return Err(LynseError::InvalidArgument(
                    "alpha must be a finite value >= 1.0".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Create an index from a type alias string.
pub fn create_index(alias: &str) -> Result<Box<dyn VectorIndex>> {
    create_index_with_build_options(alias, &IndexBuildOptions::default())
}

pub fn create_index_with_options(
    alias: &str,
    n_centroids: Option<usize>,
) -> Result<Box<dyn VectorIndex>> {
    create_index_with_build_options(alias, &IndexBuildOptions::from_n_clusters(n_centroids))
}

pub fn create_index_with_build_options(
    alias: &str,
    opts: &IndexBuildOptions,
) -> Result<Box<dyn VectorIndex>> {
    let (index_type, metric, quant) = resolve_index_type(alias)
        .ok_or_else(|| LynseError::InvalidArgument(format!("Unknown index type: {}", alias)))?;

    opts.validate()?;
    let opts = opts.filtered_for(index_type);

    match index_type {
        IndexType::Flat => Ok(Box::new(flat::FlatIndex::new(metric, quant))),
        IndexType::HNSW => Ok(Box::new(hnsw::HNSWIndex::new(
            metric,
            quant,
            opts.m.unwrap_or(16),
            opts.ef_construction.unwrap_or(128),
            opts.ef_search.unwrap_or(50),
            opts.max_level,
        ))),
        IndexType::DiskANN => {
            let r = opts.r.unwrap_or(16);
            Ok(Box::new(diskann::DiskANNIndex::with_alias(
                metric,
                quant,
                r,
                opts.l.unwrap_or(64),
                opts.alpha.unwrap_or(1.2),
                opts.max_degree.unwrap_or(r),
                alias,
            )))
        }
        IndexType::IVF => Ok(Box::new(ivf::IVFIndex::new(
            metric,
            quant,
            opts.n_clusters.unwrap_or(256),
            opts.nprobe.unwrap_or(32),
        ))),
        IndexType::SPANN => Ok(Box::new(spann::SPANNIndex::new(
            metric,
            quant,
            opts.n_clusters.unwrap_or(256),
            opts.nprobe.unwrap_or(32),
            opts.replica_count
                .unwrap_or(spann::DEFAULT_REPLICA_COUNT),
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
        match &idx.config().params {
            IndexParams::SPANN { n_centroids, .. } => assert_eq!(*n_centroids, 8),
            _ => panic!("expected SPANN params"),
        }
    }

    #[test]
    fn hnsw_build_options_apply_m_and_ef() {
        let opts = IndexBuildOptions {
            m: Some(8),
            ef_construction: Some(64),
            ef_search: Some(40),
            n_clusters: Some(999), // ignored for HNSW
            ..IndexBuildOptions::default()
        };
        let idx = create_index_with_build_options("HNSW-IP", &opts).unwrap();
        match &idx.config().params {
            IndexParams::HNSW {
                m,
                ef_construction,
                ef_search,
                ..
            } => {
                assert_eq!(*m, 8);
                assert_eq!(*ef_construction, 64);
                assert_eq!(*ef_search, 40);
            }
            _ => panic!("expected HNSW params"),
        }
    }

    #[test]
    fn diskann_build_options_apply_r_l_alpha() {
        let opts = IndexBuildOptions {
            r: Some(24),
            l: Some(80),
            alpha: Some(1.5),
            ..IndexBuildOptions::default()
        };
        let idx = create_index_with_build_options("DISKANN-L2", &opts).unwrap();
        match &idx.config().params {
            IndexParams::DiskANN {
                r,
                l,
                alpha,
                max_degree,
            } => {
                assert_eq!(*r, 24);
                assert_eq!(*l, 80);
                assert!((*alpha - 1.5).abs() < 1e-6);
                assert_eq!(*max_degree, 24);
            }
            _ => panic!("expected DiskANN params"),
        }
    }

    #[test]
    fn unknown_build_option_key_errors() {
        let v = serde_json::json!({"m": 16, "typo_param": 1});
        let err = IndexBuildOptions::from_json(&v).unwrap_err().to_string();
        assert!(err.contains("unknown index build parameter"), "{err}");
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
        assert!(resolve_index_type("HNSW-JENSEN-SHANNON").is_some());
        assert!(resolve_index_type("HNSW-CHEBYSHEV").is_some());
        assert!(resolve_index_type("FLAT-CANBERRA").is_some());
        assert!(resolve_index_type("FLAT-BRAY-CURTIS").is_some());
        assert!(resolve_index_type("HNSW-CANBERRA").is_none());
        assert!(resolve_index_type("HNSW-BRAY-CURTIS").is_none());
        assert!(resolve_index_type("FLAT-CHEBYSHEV-SQ8").is_none());
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
