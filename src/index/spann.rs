//! SPANN (Space Partitioning Approximate Nearest Neighbor) index.
//!
//! This implementation uses a memory-resident centroid routing layer with
//! posting lists and lightweight boundary replication. Build time trains L2
//! coarse partitions; search probes the nearest partitions and reranks all
//! collected candidates with the requested metric.

use super::{kmeans, IndexConfig, IndexParams, IndexType, SearchParams, VectorIndex};
use crate::distance::{compute_distance_f32, quickselect_k_pub, DistanceMetric};
use crate::error::{LynseError, Result};
use crate::quantizer::{self, Quantizer, QuantizerType};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;

pub(crate) const DEFAULT_REPLICA_COUNT: usize = 1;
const REPLICA_DISTANCE_FACTOR: f32 = 1.35;

/// SPANN index with centroid posting lists and boundary replicas.
pub struct SPANNIndex {
    config: IndexConfig,
    quantizer: Box<dyn Quantizer>,
    data: Vec<f32>,
    encoded_data: Vec<f32>,
    ids: Vec<u64>,
    centroids: Vec<f32>,
    postings: Vec<Vec<usize>>,
    primary_assignments: Vec<usize>,
    n_centroids: usize,
    nprobe: usize,
    replica_count: usize,
    trained: bool,
}

impl SPANNIndex {
    pub fn new(
        metric: DistanceMetric,
        quant_type: QuantizerType,
        n_centroids: usize,
        nprobe: usize,
        replica_count: usize,
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
                index_type: IndexType::SPANN,
                distance_metric: metric,
                quantizer_type: quant_type,
                dimension: 0,
                params: IndexParams::SPANN {
                    n_centroids,
                    nprobe,
                    replica_count,
                },
            },
            quantizer,
            data: Vec::new(),
            encoded_data: Vec::new(),
            ids: Vec::new(),
            centroids: Vec::new(),
            postings: Vec::new(),
            primary_assignments: Vec::new(),
            n_centroids,
            nprobe,
            replica_count,
            trained: false,
        }
    }

    fn quantizer_name(quant_type: QuantizerType) -> &'static str {
        match quant_type {
            QuantizerType::None => "none",
            QuantizerType::Scalar => "sq8",
            QuantizerType::Binary => "binary",
            QuantizerType::Product => "pq",
        }
    }

    fn rebuild_postings(&mut self) {
        let dim = self.config.dimension;
        let n_vectors = self.ids.len();
        if n_vectors == 0 || self.n_centroids == 0 || dim == 0 {
            self.postings = vec![Vec::new(); self.n_centroids];
            self.primary_assignments.clear();
            return;
        }

        let per_vector_centroids: Vec<Vec<usize>> = (0..n_vectors)
            .into_par_iter()
            .map(|i| {
                let vector = &self.encoded_data[i * dim..(i + 1) * dim];
                Self::posting_centroids_for_vector(
                    vector,
                    &self.centroids,
                    dim,
                    self.n_centroids,
                    self.replica_count,
                )
            })
            .collect();

        let mut postings = vec![Vec::new(); self.n_centroids];
        let mut primary_assignments = Vec::with_capacity(n_vectors);
        for (idx, centroids) in per_vector_centroids.into_iter().enumerate() {
            if let Some(&primary) = centroids.first() {
                primary_assignments.push(primary);
            }
            for centroid in centroids {
                postings[centroid].push(idx);
            }
        }

        self.postings = postings;
        self.primary_assignments = primary_assignments;
    }

    fn posting_centroids_for_vector(
        vector: &[f32],
        centroids: &[f32],
        dim: usize,
        n_centroids: usize,
        replica_count: usize,
    ) -> Vec<usize> {
        if n_centroids == 0 {
            return Vec::new();
        }

        let keep = (replica_count + 1).min(n_centroids);
        let mut best: Vec<(f32, usize)> = vec![(f32::INFINITY, usize::MAX); keep];

        for c in 0..n_centroids {
            let centroid = &centroids[c * dim..(c + 1) * dim];
            let dist = compute_distance_f32(vector, centroid, DistanceMetric::L2Squared);
            if dist >= best[keep - 1].0 {
                continue;
            }

            let mut pos = keep - 1;
            while pos > 0 && dist < best[pos - 1].0 {
                best[pos] = best[pos - 1];
                pos -= 1;
            }
            best[pos] = (dist, c);
        }

        let mut selected = Vec::with_capacity(keep);
        if best[0].1 == usize::MAX {
            selected.push(0);
            return selected;
        }

        selected.push(best[0].1);
        if replica_count == 0 {
            return selected;
        }

        let primary_dist = best[0].0;
        let threshold = if primary_dist <= f32::EPSILON {
            primary_dist + f32::EPSILON
        } else {
            primary_dist * REPLICA_DISTANCE_FACTOR
        };

        for &(dist, centroid) in best.iter().skip(1) {
            if centroid == usize::MAX {
                continue;
            }
            if selected.len() <= replica_count && dist <= threshold {
                selected.push(centroid);
            }
        }
        selected
    }

    fn nearest_centroids_for_query(&self, encoded_query: &[f32], nprobe: usize) -> Vec<usize> {
        let dim = self.config.dimension;
        let n_centroids = self.centroids.len() / dim;
        let nprobe = nprobe.max(1).min(n_centroids);
        if nprobe == 0 {
            return Vec::new();
        }

        let mut centroid_dists: Vec<(f32, u32)> = (0..n_centroids)
            .map(|c| {
                let centroid = &self.centroids[c * dim..(c + 1) * dim];
                (
                    compute_distance_f32(encoded_query, centroid, DistanceMetric::L2Squared),
                    c as u32,
                )
            })
            .collect();

        quickselect_k_pub(&mut centroid_dists, nprobe, true);
        let top = &mut centroid_dists[..nprobe];
        top.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        top.iter().map(|&(_, c)| c as usize).collect()
    }

    fn collect_candidates(
        &self,
        probed_centroids: &[usize],
        subset: Option<&HashSet<u64>>,
    ) -> Vec<usize> {
        let mut seen = vec![false; self.ids.len()];
        let estimated = probed_centroids
            .iter()
            .filter_map(|&c| self.postings.get(c).map(Vec::len))
            .sum::<usize>()
            .min(self.ids.len());
        let mut candidates = Vec::with_capacity(estimated);

        for &centroid in probed_centroids {
            let Some(list) = self.postings.get(centroid) else {
                continue;
            };
            for &idx in list {
                if seen[idx] {
                    continue;
                }
                if let Some(subset) = subset {
                    if !subset.contains(&self.ids[idx]) {
                        continue;
                    }
                }
                seen[idx] = true;
                candidates.push(idx);
            }
        }

        candidates
    }

    fn fallback_candidates(&self, subset: Option<&HashSet<u64>>) -> Vec<usize> {
        match subset {
            None => (0..self.ids.len()).collect(),
            Some(subset) => self
                .ids
                .iter()
                .enumerate()
                .filter_map(|(idx, id)| subset.contains(id).then_some(idx))
                .collect(),
        }
    }
}

impl VectorIndex for SPANNIndex {
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
            self.centroids.clear();
            self.postings.clear();
            self.primary_assignments.clear();
            self.trained = true;
            return Ok(());
        }

        let trained = kmeans::train_l2(&self.encoded_data, n_vectors, dim, self.n_centroids, 20);
        self.centroids = trained.centroids;
        self.n_centroids = trained.n_centroids;
        self.primary_assignments = trained.assignments;
        self.postings = vec![Vec::new(); self.n_centroids];

        if self.replica_count == 0 {
            for (idx, &centroid) in self.primary_assignments.iter().enumerate() {
                self.postings[centroid].push(idx);
            }
        } else {
            self.rebuild_postings();
        }

        self.config.params = IndexParams::SPANN {
            n_centroids: self.n_centroids,
            nprobe: self.nprobe,
            replica_count: self.replica_count,
        };
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
        if k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let dim = self.config.dimension;
        let encoded_query = if self.config.quantizer_type != QuantizerType::None {
            let bytes = self.quantizer.encode(query, 1, dim)?;
            self.quantizer.decode(&bytes, 1, dim)?
        } else {
            query.to_vec()
        };

        let subset_set = params
            .subset_indices
            .as_ref()
            .map(|subset| subset.iter().copied().collect::<HashSet<u64>>());
        let subset = subset_set.as_ref();

        let probed = self.nearest_centroids_for_query(&encoded_query, params.nprobe);
        let mut candidates = self.collect_candidates(&probed, subset);
        if candidates.len() < k {
            candidates = self.fallback_candidates(subset);
        }
        if candidates.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let limit = k.min(candidates.len());
        let ascending = self.config.distance_metric.is_ascending();
        let mut scored: Vec<(f32, u32)> = candidates
            .iter()
            .enumerate()
            .map(|(local_idx, &row_idx)| {
                let start = row_idx * dim;
                let dist = compute_distance_f32(
                    &encoded_query,
                    &self.encoded_data[start..start + dim],
                    self.config.distance_metric,
                );
                (dist, local_idx as u32)
            })
            .collect();

        quickselect_k_pub(&mut scored, limit, ascending);
        let top = &mut scored[..limit];
        if ascending {
            top.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        } else {
            top.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        }

        let mut result_ids = Vec::with_capacity(limit);
        let mut result_dists = Vec::with_capacity(limit);
        for &(dist, local_idx) in top.iter() {
            let row_idx = candidates[local_idx as usize];
            result_ids.push(self.ids[row_idx]);
            result_dists.push(dist);
        }

        Ok((result_ids, result_dists))
    }

    fn delete(&mut self, ids: &[u64]) -> Result<()> {
        let id_set: HashSet<u64> = ids.iter().copied().collect();
        let dim = self.config.dimension;
        let mut new_data = Vec::new();
        let mut new_encoded = Vec::new();
        let mut new_ids = Vec::new();

        for (i, &id) in self.ids.iter().enumerate() {
            if id_set.contains(&id) {
                continue;
            }
            let start = i * dim;
            new_data.extend_from_slice(&self.data[start..start + dim]);
            new_encoded.extend_from_slice(&self.encoded_data[start..start + dim]);
            new_ids.push(id);
        }

        self.data = new_data;
        self.encoded_data = new_encoded;
        self.ids = new_ids;
        self.rebuild_postings();
        Ok(())
    }

    fn insert(&mut self, vectors: &[f32], n_vectors: usize, dim: usize, ids: &[u64]) -> Result<()> {
        if n_vectors == 0 {
            return Ok(());
        }
        if !self.trained || self.centroids.is_empty() {
            return self.build(vectors, n_vectors, dim, Some(ids));
        }
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

        if self.postings.len() != self.n_centroids {
            self.postings.resize_with(self.n_centroids, Vec::new);
        }

        for i in 0..n_vectors {
            let vec_idx = old_count + i;
            let vector = &encoded_new[i * dim..(i + 1) * dim];
            let centroids = Self::posting_centroids_for_vector(
                vector,
                &self.centroids,
                dim,
                self.n_centroids,
                self.replica_count,
            );
            if let Some(&primary) = centroids.first() {
                self.primary_assignments.push(primary);
            }
            for centroid in centroids {
                self.postings[centroid].push(vec_idx);
            }
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
        let state = SPANNState {
            data: self.data.clone(),
            encoded_data: self.encoded_data.clone(),
            ids: self.ids.clone(),
            centroids: self.centroids.clone(),
            postings: self.postings.clone(),
            primary_assignments: self.primary_assignments.clone(),
            config: self.config.clone(),
            quantizer_state: self.quantizer.serialize()?,
            n_centroids: self.n_centroids,
            nprobe: self.nprobe,
            replica_count: self.replica_count,
            trained: self.trained,
        };
        bincode::serialize(&state).map_err(|e| LynseError::Serialization(e.to_string()))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let state: SPANNState =
            bincode::deserialize(data).map_err(|e| LynseError::Serialization(e.to_string()))?;
        self.data = state.data;
        self.encoded_data = state.encoded_data;
        self.ids = state.ids;
        self.centroids = state.centroids;
        self.postings = state.postings;
        self.primary_assignments = state.primary_assignments;
        self.config = state.config;
        self.n_centroids = state.n_centroids;
        self.nprobe = state.nprobe;
        self.replica_count = state.replica_count;
        self.trained = state.trained;
        let mut quantizer =
            quantizer::create_quantizer(Self::quantizer_name(self.config.quantizer_type))?;
        quantizer.deserialize(&state.quantizer_state)?;
        self.quantizer = quantizer;
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
        format!("spann-{}{}", metric, quant)
    }
}

#[derive(Serialize, Deserialize)]
struct SPANNState {
    data: Vec<f32>,
    encoded_data: Vec<f32>,
    ids: Vec<u64>,
    centroids: Vec<f32>,
    postings: Vec<Vec<usize>>,
    primary_assignments: Vec<usize>,
    config: IndexConfig,
    quantizer_state: Vec<u8>,
    n_centroids: usize,
    nprobe: usize,
    replica_count: usize,
    trained: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spann_probe_all_matches_expected_l2_top1() {
        let data = vec![
            0.0, 0.0, //
            1.0, 0.0, //
            0.0, 1.0, //
            5.0, 5.0, //
            5.2, 5.0, //
            5.0, 5.2, //
        ];
        let ids = vec![10, 11, 12, 13, 14, 15];
        let mut idx = SPANNIndex::new(DistanceMetric::L2Squared, QuantizerType::None, 2, 2, 1);
        idx.build(&data, 6, 2, Some(&ids)).unwrap();

        let params = SearchParams {
            k: 2,
            nprobe: 2,
            ef_search: None,
            subset_indices: None,
        };
        let (result_ids, dists) = idx.search(&[5.1, 5.0], 2, &params).unwrap();

        assert!(matches!(result_ids[0], 13 | 14));
        assert!(dists[0] <= dists[1]);
    }

    #[test]
    fn spann_subset_filter_uses_index_ids() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 5.0, 5.0, 5.1, 5.0];
        let ids = vec![0, 1, 2, 3];
        let mut idx = SPANNIndex::new(DistanceMetric::L2Squared, QuantizerType::None, 2, 2, 1);
        idx.build(&data, 4, 2, Some(&ids)).unwrap();

        let params = SearchParams {
            k: 2,
            nprobe: 1,
            ef_search: None,
            subset_indices: Some(vec![0, 1]),
        };
        let (result_ids, _) = idx.search(&[5.1, 5.0], 2, &params).unwrap();

        assert!(result_ids.iter().all(|id| *id <= 1));
    }

    #[test]
    fn spann_sq8_serialization_keeps_quantizer_state() {
        let data = vec![
            0.0, 0.0, //
            0.2, 0.0, //
            0.0, 0.2, //
            5.0, 5.0, //
            5.2, 5.0, //
            5.0, 5.2, //
        ];
        let ids = vec![10, 11, 12, 13, 14, 15];
        let mut idx = SPANNIndex::new(DistanceMetric::L2Squared, QuantizerType::Scalar, 2, 2, 1);
        idx.build(&data, 6, 2, Some(&ids)).unwrap();

        let bytes = idx.serialize().unwrap();
        let mut loaded = SPANNIndex::new(DistanceMetric::L2Squared, QuantizerType::Scalar, 2, 2, 1);
        loaded.deserialize(&bytes).unwrap();

        let params = SearchParams {
            k: 2,
            nprobe: 2,
            ef_search: None,
            subset_indices: None,
        };
        let (result_ids, dists) = loaded.search(&[5.1, 5.0], 2, &params).unwrap();

        assert_eq!(result_ids.len(), 2);
        assert!(dists.iter().all(|d| d.is_finite()));
    }
}
