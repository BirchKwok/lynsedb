//! DiskANN index implementation.
//!
//! Graph-based approximate nearest neighbor search optimized for disk-resident data.
//! Supports billion-scale datasets with good recall-latency tradeoff.

use crate::distance::{self, compute_distance_f32, DistanceMetric};
use crate::error::{LynseError, Result};
use crate::quantizer::{self, Quantizer, QuantizerType};
use super::{IndexConfig, IndexParams, IndexType, SearchParams, VectorIndex};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;

/// DiskANN graph-based index.
pub struct DiskANNIndex {
    config: IndexConfig,
    quantizer: Box<dyn Quantizer>,
    data: Vec<f32>,
    encoded_data: Vec<f32>,
    ids: Vec<u64>,
    /// Adjacency list graph
    graph: Vec<Vec<usize>>,
    entry_point: Option<usize>,
    r: usize,
    l: usize,
    alpha: f32,
    max_degree: usize,
    trained: bool,
}

impl DiskANNIndex {
    pub fn new(
        metric: DistanceMetric,
        quant_type: QuantizerType,
        r: usize,
        l: usize,
        alpha: f32,
        max_degree: usize,
    ) -> Self {
        let quantizer = quantizer::create_quantizer(
            match quant_type {
                QuantizerType::None => "none",
                QuantizerType::Scalar => "sq8",
                QuantizerType::Binary => "binary",
                QuantizerType::Product => "pq",
            },
        )
        .unwrap();

        Self {
            config: IndexConfig {
                index_type: IndexType::DiskANN,
                distance_metric: metric,
                quantizer_type: quant_type,
                dimension: 0,
                params: IndexParams::DiskANN {
                    r,
                    l,
                    alpha,
                    max_degree,
                },
            },
            quantizer,
            data: Vec::new(),
            encoded_data: Vec::new(),
            ids: Vec::new(),
            graph: Vec::new(),
            entry_point: None,
            r,
            l,
            alpha,
            max_degree,
            trained: false,
        }
    }

    fn distance_single(&self, a_idx: usize, b: &[f32]) -> f32 {
        let dim = self.config.dimension;
        let start = a_idx * dim;
        compute_distance_f32(
            &self.encoded_data[start..start + dim],
            b,
            self.config.distance_metric,
        )
    }

    fn search_graph(
        &self,
        query: &[f32],
        ef: usize,
        subset: Option<&HashSet<usize>>,
    ) -> Vec<usize> {
        let entry = match self.entry_point {
            Some(ep) => ep,
            None => return vec![],
        };

        let mut visited = HashSet::new();
        visited.insert(entry);

        let entry_dist = self.distance_single(entry, query);
        let mut candidates: Vec<(f32, usize)> = vec![(entry_dist, entry)];

        while !candidates.is_empty() {
            let (_, current) = candidates.remove(0);

            if current < self.graph.len() {
                for &neighbor in &self.graph[current] {
                    if !visited.contains(&neighbor) {
                        if let Some(sub) = subset {
                            if !sub.contains(&neighbor) {
                                continue;
                            }
                        }
                        visited.insert(neighbor);
                        let n_dist = self.distance_single(neighbor, query);
                        candidates.push((n_dist, neighbor));
                    }
                }
            }

            candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            candidates.truncate(ef);
        }

        candidates.into_iter().map(|(_, id)| id).collect()
    }

    fn prune_edges(&mut self, node_id: usize) {
        if self.graph[node_id].len() <= self.max_degree {
            return;
        }

        let dim = self.config.dimension;
        let node_start = node_id * dim;
        let node_vec: Vec<f32> = self.encoded_data[node_start..node_start + dim].to_vec();

        let mut distances: Vec<(f32, usize)> = self.graph[node_id]
            .iter()
            .map(|&n| {
                let n_start = n * dim;
                let dist = compute_distance_f32(
                    &node_vec,
                    &self.encoded_data[n_start..n_start + dim],
                    self.config.distance_metric,
                );
                (dist, n)
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        self.graph[node_id] = distances
            .into_iter()
            .take(self.max_degree)
            .map(|(_, n)| n)
            .collect();
    }
}

impl VectorIndex for DiskANNIndex {
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

        // Initialize graph
        self.graph = vec![Vec::new(); n_vectors];
        let mut rng = rand::thread_rng();
        self.entry_point = Some(rng.gen_range(0..n_vectors));

        // Build graph incrementally
        for i in 0..n_vectors {
            if Some(i) == self.entry_point {
                continue;
            }

            let i_start = i * dim;
            let point_vec: Vec<f32> = self.encoded_data[i_start..i_start + dim].to_vec();

            let neighbors = self.search_graph(&point_vec, self.l, None);

            for &neighbor in neighbors.iter().take(self.r) {
                if self.graph[i].len() < self.max_degree {
                    self.graph[i].push(neighbor);
                }
                if self.graph[neighbor].len() < self.max_degree {
                    self.graph[neighbor].push(i);
                }
            }

            if self.graph[i].len() > self.max_degree {
                self.prune_edges(i);
            }
        }

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

        let encoded_query = if self.config.quantizer_type != QuantizerType::None {
            let bytes = self.quantizer.encode(query, 1, dim)?;
            self.quantizer.decode(&bytes, 1, dim)?
        } else {
            query.to_vec()
        };

        let subset = params.subset_indices.as_ref().map(|ids| {
            let id_set: HashSet<u64> = ids.iter().cloned().collect();
            self.ids
                .iter()
                .enumerate()
                .filter(|(_, id)| id_set.contains(id))
                .map(|(i, _)| i)
                .collect::<HashSet<usize>>()
        });

        let ef = (k * 10).max(self.l);
        let candidates = self.search_graph(&encoded_query, ef, subset.as_ref());

        if candidates.is_empty() {
            // Fallback: brute force
            let (top_idx, top_dist) = distance::top_k_search(
                &encoded_query,
                &self.encoded_data,
                dim,
                k,
                self.config.distance_metric,
            );
            let result_ids: Vec<u64> = top_idx.iter().map(|&i| self.ids[i as usize]).collect();
            return Ok((result_ids, top_dist));
        }

        let mut scored: Vec<(f32, usize)> = candidates
            .iter()
            .map(|&c| (self.distance_single(c, &encoded_query), c))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        scored.truncate(k);

        let result_ids: Vec<u64> = scored.iter().map(|(_, idx)| self.ids[*idx]).collect();
        let result_dists: Vec<f32> = scored.iter().map(|(d, _)| *d).collect();

        Ok((result_ids, result_dists))
    }

    fn delete(&mut self, ids: &[u64]) -> Result<()> {
        let id_set: HashSet<u64> = ids.iter().cloned().collect();
        let indices_to_delete: HashSet<usize> = self
            .ids
            .iter()
            .enumerate()
            .filter(|(_, id)| id_set.contains(id))
            .map(|(i, _)| i)
            .collect();

        if indices_to_delete.is_empty() {
            return Ok(());
        }

        // Build remap
        let mut index_map: HashMap<usize, usize> = HashMap::new();
        let mut new_idx = 0usize;
        for old_idx in 0..self.ids.len() {
            if !indices_to_delete.contains(&old_idx) {
                index_map.insert(old_idx, new_idx);
                new_idx += 1;
            }
        }

        // Rebuild graph
        let mut new_graph = Vec::new();
        for (i, neighbors) in self.graph.iter().enumerate() {
            if !indices_to_delete.contains(&i) {
                let new_neighbors: Vec<usize> = neighbors
                    .iter()
                    .filter(|n| !indices_to_delete.contains(n))
                    .map(|&n| index_map[&n])
                    .collect();
                new_graph.push(new_neighbors);
            }
        }
        // Update entry point before moving new_graph
        if let Some(ep) = self.entry_point {
            if indices_to_delete.contains(&ep) {
                self.entry_point = if new_graph.is_empty() {
                    None
                } else {
                    Some(0)
                };
            } else {
                self.entry_point = Some(index_map[&ep]);
            }
        }
        self.graph = new_graph;

        // Remove data
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

        Ok(())
    }

    fn insert(
        &mut self,
        vectors: &[f32],
        n_vectors: usize,
        dim: usize,
        ids: &[u64],
    ) -> Result<()> {
        if dim != self.config.dimension {
            return Err(LynseError::DimensionMismatch {
                expected: self.config.dimension,
                got: dim,
            });
        }

        let old_count = self.ids.len();
        self.data.extend_from_slice(vectors);
        self.ids.extend_from_slice(ids);

        if self.config.quantizer_type != QuantizerType::None {
            let bytes = self.quantizer.encode(vectors, n_vectors, dim)?;
            let decoded = self.quantizer.decode(&bytes, n_vectors, dim)?;
            self.encoded_data.extend_from_slice(&decoded);
        } else {
            self.encoded_data.extend_from_slice(vectors);
        }

        // Extend graph
        for _ in 0..n_vectors {
            self.graph.push(Vec::new());
        }

        // Add new points to graph
        for i in 0..n_vectors {
            let point_idx = old_count + i;
            let p_start = point_idx * dim;
            let point_vec: Vec<f32> = self.encoded_data[p_start..p_start + dim].to_vec();

            let neighbors = self.search_graph(&point_vec, self.l, None);
            for &neighbor in neighbors.iter().take(self.r) {
                if self.graph[point_idx].len() < self.max_degree {
                    self.graph[point_idx].push(neighbor);
                }
                if self.graph[neighbor].len() < self.max_degree {
                    self.graph[neighbor].push(point_idx);
                }
            }

            if self.graph[point_idx].len() > self.max_degree {
                self.prune_edges(point_idx);
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
        let state = DiskANNState {
            data: self.data.clone(),
            encoded_data: self.encoded_data.clone(),
            ids: self.ids.clone(),
            graph: self.graph.clone(),
            entry_point: self.entry_point,
            config: self.config.clone(),
            r: self.r,
            l: self.l,
            alpha: self.alpha,
            max_degree: self.max_degree,
            trained: self.trained,
        };
        bincode::serialize(&state).map_err(|e| LynseError::Serialization(e.to_string()))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let state: DiskANNState =
            bincode::deserialize(data).map_err(|e| LynseError::Serialization(e.to_string()))?;
        self.data = state.data;
        self.encoded_data = state.encoded_data;
        self.ids = state.ids;
        self.graph = state.graph;
        self.entry_point = state.entry_point;
        self.config = state.config;
        self.r = state.r;
        self.l = state.l;
        self.alpha = state.alpha;
        self.max_degree = state.max_degree;
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
        format!("diskann-{}{}", metric, quant)
    }
}

#[derive(Serialize, Deserialize)]
struct DiskANNState {
    data: Vec<f32>,
    encoded_data: Vec<f32>,
    ids: Vec<u64>,
    graph: Vec<Vec<usize>>,
    entry_point: Option<usize>,
    config: IndexConfig,
    r: usize,
    l: usize,
    alpha: f32,
    max_degree: usize,
    trained: bool,
}
