//! IVF (Inverted File) index implementation.
//!
//! Partitions the vector space into clusters using K-means, then searches
//! only the nearest clusters. Good balance of speed and recall for large datasets.

use super::{kmeans, IndexConfig, IndexParams, IndexType, SearchParams, VectorIndex};
use crate::distance::{compute_distance_f32, DistanceMetric};
use crate::error::{LynseError, Result};
use crate::quantizer::{self, Quantizer, QuantizerType};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

/// IVF index with inverted file structure.
pub struct IVFIndex {
    config: IndexConfig,
    quantizer: Box<dyn Quantizer>,
    data: Vec<f32>,
    encoded_data: Vec<f32>,
    ids: Vec<u64>,
    /// Cluster centroids (flattened: n_centroids * dim)
    centroids: Vec<f32>,
    /// Inverted lists: cluster_id → list of vector indices
    inverted_lists: HashMap<usize, Vec<usize>>,
    n_centroids: usize,
    nprobe: usize,
    trained: bool,
}

impl IVFIndex {
    pub fn new(
        metric: DistanceMetric,
        quant_type: QuantizerType,
        n_centroids: usize,
        nprobe: usize,
    ) -> Self {
        let quantizer = quantizer::create_quantizer(match quant_type {
            QuantizerType::None => "none",
            QuantizerType::Scalar => "sq8",
            QuantizerType::Binary => "binary",
            QuantizerType::Product => "pq",
        })
        .unwrap();

        Self {
            config: IndexConfig {
                index_type: IndexType::IVF,
                distance_metric: metric,
                quantizer_type: quant_type,
                dimension: 0,
                params: IndexParams::IVF {
                    n_centroids,
                    nprobe,
                },
            },
            quantizer,
            data: Vec::new(),
            encoded_data: Vec::new(),
            ids: Vec::new(),
            centroids: Vec::new(),
            inverted_lists: HashMap::new(),
            n_centroids,
            nprobe,
            trained: false,
        }
    }
}

impl VectorIndex for IVFIndex {
    fn build(
        &mut self,
        vectors: &[f32],
        n_vectors: usize,
        dim: usize,
        ids: Option<&[u64]>,
    ) -> Result<()> {
        self.config.dimension = dim;
        self.ids = match ids {
            Some(id_slice) => id_slice.to_vec(),
            None => (0..n_vectors as u64).collect(),
        };

        self.data = vectors.to_vec();

        if self.config.quantizer_type != QuantizerType::None {
            self.quantizer.fit(vectors, n_vectors, dim)?;
            let bytes = self.quantizer.encode(vectors, n_vectors, dim)?;
            self.encoded_data = self.quantizer.decode(&bytes, n_vectors, dim)?;
        } else {
            self.encoded_data = vectors.to_vec();
        }

        if n_vectors == 0 {
            self.trained = true;
            return Ok(());
        }

        // Train centroids and assign vectors without cloning the encoded matrix.
        let trained = kmeans::train_l2(&self.encoded_data, n_vectors, dim, self.n_centroids, 20);
        self.centroids = trained.centroids;
        self.n_centroids = trained.n_centroids;
        self.inverted_lists =
            kmeans::inverted_lists_from_assignments(&trained.assignments, self.n_centroids);

        self.trained = true;
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        if !self.trained || self.ids.is_empty() {
            return Err(LynseError::IndexNotBuilt);
        }

        let dim = self.config.dimension;
        let nprobe = params.nprobe.max(1);

        let encoded_query = if self.config.quantizer_type != QuantizerType::None {
            let bytes = self.quantizer.encode(query, 1, dim)?;
            self.quantizer.decode(&bytes, 1, dim)?
        } else {
            query.to_vec()
        };

        let n_centroids = self.centroids.len() / dim;

        // Find nearest centroids
        let mut centroid_dists: Vec<(f32, usize)> = (0..n_centroids)
            .map(|c| {
                let dist = compute_distance_f32(
                    &encoded_query,
                    &self.centroids[c * dim..(c + 1) * dim],
                    DistanceMetric::L2Squared,
                );
                (dist, c)
            })
            .collect();
        centroid_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Collect candidates from nearest clusters
        let mut candidates: Vec<usize> = Vec::new();
        for &(_, centroid_id) in centroid_dists.iter().take(nprobe) {
            if let Some(list) = self.inverted_lists.get(&centroid_id) {
                candidates.extend_from_slice(list);
            }
        }

        // Apply subset filter
        if let Some(ref subset) = params.subset_indices {
            let subset_set: HashSet<u64> = subset.iter().cloned().collect();
            candidates.retain(|&c| subset_set.contains(&self.ids[c]));
        }

        if candidates.is_empty() {
            // Fallback: search all
            candidates = (0..self.ids.len()).collect();
        }

        // Limit candidates
        candidates.truncate(k * 100);

        // Score candidates
        let mut scored: Vec<(f32, usize)> = candidates
            .iter()
            .map(|&c| {
                let dist = compute_distance_f32(
                    &encoded_query,
                    &self.encoded_data[c * dim..(c + 1) * dim],
                    self.config.distance_metric,
                );
                (dist, c)
            })
            .collect();

        // Sort by distance metric
        if self.config.distance_metric.is_ascending() {
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        } else {
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        }
        scored.truncate(k);

        let result_ids: Vec<u64> = scored.iter().map(|(_, idx)| self.ids[*idx]).collect();
        let result_dists: Vec<f32> = scored.iter().map(|(d, _)| *d).collect();

        Ok((result_ids, result_dists))
    }

    fn delete(&mut self, ids: &[u64]) -> Result<()> {
        let id_set: HashSet<u64> = ids.iter().cloned().collect();

        let dim = self.config.dimension;
        let mut new_data = Vec::new();
        let mut new_encoded = Vec::new();
        let mut new_ids = Vec::new();

        for (i, &id) in self.ids.iter().enumerate() {
            if !id_set.contains(&id) {
                let start = i * dim;
                new_data.extend_from_slice(&self.data[start..start + dim]);
                new_encoded.extend_from_slice(&self.encoded_data[start..start + dim]);
                new_ids.push(id);
            }
        }

        self.data = new_data;
        self.encoded_data = new_encoded;
        self.ids = new_ids;

        // Reassign to clusters
        if !self.encoded_data.is_empty() {
            let n = self.ids.len();
            let assignments = kmeans::assign_l2(
                &self.encoded_data[..n * dim],
                &self.centroids,
                dim,
                self.n_centroids,
            );
            self.inverted_lists =
                kmeans::inverted_lists_from_assignments(&assignments, self.n_centroids);
        } else {
            self.inverted_lists.clear();
        }

        Ok(())
    }

    fn insert(&mut self, vectors: &[f32], n_vectors: usize, dim: usize, ids: &[u64]) -> Result<()> {
        if dim != self.config.dimension {
            return Err(LynseError::DimensionMismatch {
                expected: self.config.dimension,
                got: dim,
            });
        }

        let old_count = self.ids.len();
        self.data.extend_from_slice(vectors);
        self.ids.extend_from_slice(ids);

        let encoded_new = if self.config.quantizer_type != QuantizerType::None {
            let bytes = self.quantizer.encode(vectors, n_vectors, dim)?;
            self.quantizer.decode(&bytes, n_vectors, dim)?
        } else {
            vectors.to_vec()
        };
        self.encoded_data.extend_from_slice(&encoded_new);

        // Assign new vectors to clusters
        let n_centroids = self.centroids.len() / dim;
        let half_norms = kmeans::centroid_half_norms(&self.centroids, dim, n_centroids);
        for i in 0..n_vectors {
            let vec_idx = old_count + i;
            let best_c = kmeans::nearest_l2_centroid(
                &encoded_new[i * dim..(i + 1) * dim],
                &self.centroids,
                &half_norms,
                dim,
                n_centroids,
            );
            self.inverted_lists
                .entry(best_c)
                .or_insert_with(Vec::new)
                .push(vec_idx);
        }

        Ok(())
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn config(&self) -> &IndexConfig {
        &self.config
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        let state = IVFState {
            data: self.data.clone(),
            encoded_data: self.encoded_data.clone(),
            ids: self.ids.clone(),
            centroids: self.centroids.clone(),
            inverted_lists: self.inverted_lists.clone(),
            config: self.config.clone(),
            n_centroids: self.n_centroids,
            nprobe: self.nprobe,
            trained: self.trained,
        };
        bincode::serialize(&state).map_err(|e| LynseError::Serialization(e.to_string()))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let state: IVFState =
            bincode::deserialize(data).map_err(|e| LynseError::Serialization(e.to_string()))?;
        self.data = state.data;
        self.encoded_data = state.encoded_data;
        self.ids = state.ids;
        self.centroids = state.centroids;
        self.inverted_lists = state.inverted_lists;
        self.config = state.config;
        self.n_centroids = state.n_centroids;
        self.nprobe = state.nprobe;
        self.trained = state.trained;
        Ok(())
    }

    fn name(&self) -> String {
        let metric = self.config.distance_metric.name();
        let quant = match self.config.quantizer_type {
            QuantizerType::None => "",
            QuantizerType::Scalar => "-sq8",
            QuantizerType::Binary => "-binary",
            QuantizerType::Product => "-pq",
        };
        format!("ivf-{}{}", metric, quant)
    }
}

#[derive(Serialize, Deserialize)]
struct IVFState {
    data: Vec<f32>,
    encoded_data: Vec<f32>,
    ids: Vec<u64>,
    centroids: Vec<f32>,
    inverted_lists: HashMap<usize, Vec<usize>>,
    config: IndexConfig,
    n_centroids: usize,
    nprobe: usize,
    trained: bool,
}
