//! IVF (Inverted File) index implementation.
//!
//! Partitions the vector space into clusters using K-means, then searches
//! only the nearest clusters. Good balance of speed and recall for large datasets.

use super::{kmeans, IndexConfig, IndexParams, IndexType, SearchParams, VectorIndex};
use crate::distance::simd::{
    pack_binary_f32, packed_dice_u64, packed_hamming_u64, packed_jaccard_u64,
};
use crate::distance::{compute_distance_f32, DistanceMetric};
use crate::error::{LynseError, Result};
use crate::quantizer::{self, Quantizer, QuantizerType};
use rayon::prelude::*;
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
    /// Packed one-bit codes for binary search metrics (`words_per_vector` u64s/row).
    packed_codes: Vec<u64>,
    words_per_vector: usize,
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
            packed_codes: Vec::new(),
            words_per_vector: 0,
        }
    }

    /// Coarse quantizer metric. Binary search metrics route with L2 so continuous
    /// centroids keep soft bit-density information (Hamming on thresholded means
    /// discards that and yields weak probe order + very slow training).
    #[inline]
    fn routing_metric(&self) -> DistanceMetric {
        if self.config.distance_metric.is_binary() {
            DistanceMetric::L2Squared
        } else {
            self.config.distance_metric
        }
    }

    fn rebuild_packed_codes(&mut self) {
        self.packed_codes.clear();
        self.words_per_vector = 0;
        if !self.config.distance_metric.is_binary() || self.encoded_data.is_empty() {
            return;
        }
        let dim = self.config.dimension;
        let n = self.ids.len();
        if dim == 0 || n == 0 {
            return;
        }
        let words = dim.div_ceil(64);
        let mut packed = vec![0u64; n * words];
        packed
            .par_chunks_mut(words)
            .zip(self.encoded_data.par_chunks(dim))
            .for_each(|(dst, src)| pack_binary_f32(src, dst));
        self.words_per_vector = words;
        self.packed_codes = packed;
    }

    #[inline]
    fn packed_distance_fn(metric: DistanceMetric) -> fn(&[u64], &[u64]) -> f32 {
        match metric {
            DistanceMetric::Hamming => packed_hamming_u64,
            DistanceMetric::Jaccard | DistanceMetric::Tanimoto => packed_jaccard_u64,
            DistanceMetric::Dice => packed_dice_u64,
            _ => packed_hamming_u64,
        }
    }

    fn pack_query_binary(&self, query: &[f32]) -> Result<Vec<u64>> {
        let dim = self.config.dimension;
        let bytes = self.quantizer.encode(query, 1, dim)?;
        let encoded = self.quantizer.decode(&bytes, 1, dim)?;
        let words = dim.div_ceil(64);
        let mut packed = vec![0u64; words];
        pack_binary_f32(&encoded, &mut packed);
        Ok(packed)
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
            self.packed_codes.clear();
            self.words_per_vector = 0;
            self.trained = true;
            return Ok(());
        }

        // Train centroids with the routing metric (L2 for binary search metrics).
        let trained = kmeans::train_for_metric(
            &self.encoded_data,
            n_vectors,
            dim,
            self.n_centroids,
            20,
            self.routing_metric(),
        );
        self.centroids = trained.centroids;
        self.n_centroids = trained.n_centroids;
        self.inverted_lists =
            kmeans::inverted_lists_from_assignments(&trained.assignments, self.n_centroids);
        self.rebuild_packed_codes();

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
        let nprobe = if params.nprobe == 0 {
            self.nprobe.max(1)
        } else {
            params.nprobe.max(1)
        };

        // SQ8/PQ benefit from oversample + full-precision re-rank. Binary codes are
        // already the search representation — skip the slow float re-rank path.
        let use_exact_rerank = matches!(
            self.config.quantizer_type,
            QuantizerType::Scalar | QuantizerType::Product
        ) && self.data.len() == self.ids.len() * dim;

        let metric = self.config.distance_metric;
        let ascending = metric.is_ascending();
        let binary_metric = metric.is_binary();
        let routing = self.routing_metric();
        let routing_ascending = routing.is_ascending();

        let encoded_query = if self.config.quantizer_type != QuantizerType::None {
            let bytes = self.quantizer.encode(query, 1, dim)?;
            self.quantizer.decode(&bytes, 1, dim)?
        } else {
            query.to_vec()
        };

        let packed_query = if binary_metric && self.words_per_vector > 0 {
            Some(self.pack_query_binary(query)?)
        } else {
            None
        };

        let n_centroids = self.centroids.len() / dim;

        // Rank centroids with the routing metric (L2 for binary indexes).
        let mut centroid_dists: Vec<(f32, usize)> = (0..n_centroids)
            .map(|c| {
                let dist = compute_distance_f32(
                    &encoded_query,
                    &self.centroids[c * dim..(c + 1) * dim],
                    routing,
                );
                (dist, c)
            })
            .collect();
        if routing_ascending {
            centroid_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        } else {
            centroid_dists.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        }

        // Collect *all* vectors from the nprobe nearest lists (Faiss-style).
        let mut candidates: Vec<usize> = Vec::new();
        for &(_, centroid_id) in centroid_dists.iter().take(nprobe.min(n_centroids)) {
            if let Some(list) = self.inverted_lists.get(&centroid_id) {
                candidates.extend_from_slice(list);
            }
        }

        // Apply subset filter. Empty probed lists must fall back to the filtered
        // corpus — never to an unfiltered full scan.
        let subset = params.subset.as_ref();
        if let Some(subset) = subset {
            candidates.retain(|&c| subset.contains(self.ids[c] as usize));
        }

        if candidates.is_empty() {
            candidates = match subset {
                Some(subset) => (0..self.ids.len())
                    .filter(|&c| subset.contains(self.ids[c] as usize))
                    .collect(),
                None => (0..self.ids.len()).collect(),
            };
        }

        if candidates.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let pool = if use_exact_rerank {
            (k.saturating_mul(10)).max(k).min(candidates.len())
        } else {
            k.min(candidates.len())
        };

        let mut scored: Vec<(f32, u32)> = if let Some(ref pq) = packed_query {
            let words = self.words_per_vector;
            let dist_fn = Self::packed_distance_fn(metric);
            candidates
                .iter()
                .enumerate()
                .map(|(local_idx, &c)| {
                    let start = c * words;
                    let dist = dist_fn(pq, &self.packed_codes[start..start + words]);
                    (dist, local_idx as u32)
                })
                .collect()
        } else {
            candidates
                .iter()
                .enumerate()
                .map(|(local_idx, &c)| {
                    let dist = compute_distance_f32(
                        &encoded_query,
                        &self.encoded_data[c * dim..(c + 1) * dim],
                        metric,
                    );
                    (dist, local_idx as u32)
                })
                .collect()
        };

        crate::distance::quickselect_k_pub(&mut scored, pool, ascending);
        let top = &mut scored[..pool];
        if ascending {
            top.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        } else {
            top.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        }

        if use_exact_rerank {
            let mut rescored: Vec<(f32, u32)> = top
                .iter()
                .map(|(_, local_idx)| {
                    let c = candidates[*local_idx as usize];
                    let dist =
                        compute_distance_f32(query, &self.data[c * dim..(c + 1) * dim], metric);
                    (dist, c as u32)
                })
                .collect();
            let limit = k.min(rescored.len());
            crate::distance::quickselect_k_pub(&mut rescored, limit, ascending);
            let exact_top = &mut rescored[..limit];
            if ascending {
                exact_top.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            } else {
                exact_top.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
            }
            return Ok((
                exact_top
                    .iter()
                    .map(|(_, c)| self.ids[*c as usize])
                    .collect(),
                exact_top.iter().map(|(d, _)| *d).collect(),
            ));
        }

        let limit = k.min(top.len());
        let mut result_ids = Vec::with_capacity(limit);
        let mut result_dists = Vec::with_capacity(limit);
        for &(dist, local_idx) in top.iter().take(limit) {
            result_ids.push(self.ids[candidates[local_idx as usize]]);
            result_dists.push(dist);
        }

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

        // Reassign to clusters with the same routing metric used at build time.
        if !self.encoded_data.is_empty() {
            let n = self.ids.len();
            let assignments = kmeans::assign_metric(
                &self.encoded_data[..n * dim],
                &self.centroids,
                dim,
                self.n_centroids,
                self.routing_metric(),
            );
            self.inverted_lists =
                kmeans::inverted_lists_from_assignments(&assignments, self.n_centroids);
        } else {
            self.inverted_lists.clear();
        }
        self.rebuild_packed_codes();

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

        // Assign new vectors to clusters with the routing metric.
        let routing = self.routing_metric();
        let ascending = routing.is_ascending();
        let n_centroids = self.centroids.len() / dim;
        let words = if self.config.distance_metric.is_binary() {
            dim.div_ceil(64)
        } else {
            0
        };
        if words > 0 && self.words_per_vector == 0 {
            self.words_per_vector = words;
        }
        for i in 0..n_vectors {
            let vec_idx = old_count + i;
            let vector = &encoded_new[i * dim..(i + 1) * dim];
            let mut best_c = 0usize;
            let mut best_rank = f32::MAX;
            for c in 0..n_centroids {
                let centroid = &self.centroids[c * dim..(c + 1) * dim];
                let raw = compute_distance_f32(vector, centroid, routing);
                let rank = if ascending { raw } else { -raw };
                if rank < best_rank {
                    best_rank = rank;
                    best_c = c;
                }
            }
            self.inverted_lists
                .entry(best_c)
                .or_insert_with(Vec::new)
                .push(vec_idx);

            if words > 0 {
                let mut packed = vec![0u64; words];
                pack_binary_f32(vector, &mut packed);
                self.packed_codes.extend_from_slice(&packed);
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
            quantizer_state: self.quantizer.serialize()?,
            packed_codes: self.packed_codes.clone(),
            words_per_vector: self.words_per_vector,
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
        self.packed_codes = state.packed_codes;
        self.words_per_vector = state.words_per_vector;
        if !state.quantizer_state.is_empty() {
            self.quantizer.deserialize(&state.quantizer_state)?;
        }
        // Older checkpoints omit packed codes — rebuild lazily for binary metrics.
        if self.config.distance_metric.is_binary()
            && (self.packed_codes.is_empty() || self.words_per_vector == 0)
            && !self.encoded_data.is_empty()
        {
            self.rebuild_packed_codes();
        }
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
    #[serde(default)]
    quantizer_state: Vec<u8>,
    #[serde(default)]
    packed_codes: Vec<u64>,
    #[serde(default)]
    words_per_vector: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn filtered_search_empty_probe_does_not_leak_unfiltered_ids() {
        let mut idx = IVFIndex::new(DistanceMetric::L2Squared, QuantizerType::None, 2, 1);
        // Two well-separated clusters.
        let vectors = vec![
            0.0, 0.0, // id 1
            0.1, 0.0, // id 2
            10.0, 10.0, // id 3
            10.1, 10.0, // id 4
        ];
        let ids = [1u64, 2, 3, 4];
        idx.build(&vectors, 4, 2, Some(&ids)).unwrap();

        // Only keep the far cluster in the filter, but probe the near centroid
        // of the query so the probed lists become empty after filtering.
        let params = SearchParams {
            k: 2,
            nprobe: 1,
            ef_search: None,
            subset: Some(Arc::new(crate::storage::bitset::BitSet::from_ids([
                3u64, 4,
            ]))),
        };
        let (result_ids, _) = idx.search(&[0.0, 0.0], 2, &params).unwrap();
        assert!(!result_ids.is_empty(), "expected filtered fallback hits");
        assert!(
            result_ids.iter().all(|id| *id == 3 || *id == 4),
            "unfiltered ids leaked: {:?}",
            result_ids
        );
    }

    #[test]
    fn ivf_ip_recall_improves_with_nprobe() {
        let mut idx = IVFIndex::new(DistanceMetric::InnerProduct, QuantizerType::None, 32, 4);
        let n = 800usize;
        let dim = 32usize;
        let mut vectors = Vec::with_capacity(n * dim);
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            for j in 0..dim {
                let x = (((i * 131 + j * 17 + 1) % 997) as f32 / 997.0) + 0.01;
                vectors.push(x);
            }
            ids.push(i as u64);
        }
        idx.build(&vectors, n, dim, Some(&ids)).unwrap();

        let q = &vectors[..dim];
        // Brute-force top-10 by IP
        let mut scored: Vec<(f32, u64)> = (0..n)
            .map(|i| {
                let v = &vectors[i * dim..(i + 1) * dim];
                let ip: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                (ip, i as u64)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let exact: std::collections::HashSet<u64> =
            scored.iter().take(10).map(|(_, id)| *id).collect();

        let low = SearchParams {
            k: 10,
            nprobe: 2,
            ef_search: None,
            subset: None,
        };
        let high = SearchParams {
            k: 10,
            nprobe: 32,
            ef_search: None,
            subset: None,
        };
        let (ids_low, _) = idx.search(q, 10, &low).unwrap();
        let (ids_high, _) = idx.search(q, 10, &high).unwrap();
        let rec_low = ids_low.iter().filter(|id| exact.contains(id)).count() as f32 / 10.0;
        let rec_high = ids_high.iter().filter(|id| exact.contains(id)).count() as f32 / 10.0;
        assert!(
            rec_high >= rec_low,
            "higher nprobe should not hurt recall: low={rec_low} high={rec_high}"
        );
        // nprobe == n_centroids → full scan of all inverted lists → exact top-k.
        assert!(
            (rec_high - 1.0).abs() < f32::EPSILON,
            "IVF-IP with nprobe=n_centroids must be exact, got recall={rec_high} ids={:?}",
            ids_high
        );
        // Raising nprobe past tiny values must change or improve results vs nprobe=2
        // (guards against the old k*100 early-stop that froze probe depth).
        assert!(
            rec_high > rec_low || ids_low != ids_high || rec_low >= 0.9,
            "nprobe increase should improve exploration: low={rec_low} high={rec_high}"
        );
    }

    #[test]
    fn ivf_hamming_binary_full_probe_matches_flat_distances() {
        let mut idx = IVFIndex::new(DistanceMetric::Hamming, QuantizerType::Binary, 16, 4);
        let n = 256usize;
        let dim = 32usize;
        let mut vectors = Vec::with_capacity(n * dim);
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            for j in 0..dim {
                vectors.push(if ((i * 17 + j * 3) % 2) == 0 {
                    1.0
                } else {
                    0.0
                });
            }
            ids.push(i as u64);
        }
        idx.build(&vectors, n, dim, Some(&ids)).unwrap();
        assert!(!idx.packed_codes.is_empty());

        let q = &vectors[..dim];
        let mut exact: Vec<(f32, u64)> = (0..n)
            .map(|i| {
                let v = &vectors[i * dim..(i + 1) * dim];
                let dist = compute_distance_f32(q, v, DistanceMetric::Hamming);
                (dist, i as u64)
            })
            .collect();
        exact.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        let exact_dists: Vec<f32> = exact.iter().take(10).map(|(d, _)| *d).collect();

        let params = SearchParams {
            k: 10,
            nprobe: 16,
            ef_search: None,
            subset: None,
        };
        let (_, got_dists) = idx.search(q, 10, &params).unwrap();
        assert_eq!(got_dists, exact_dists);
    }
}
