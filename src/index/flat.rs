//! Flat (brute-force) index implementation.
//!
//! Exact nearest neighbor search by computing distances to all vectors.
//! Best for small-to-medium datasets or as a baseline for accuracy.

use crate::distance::{self, DistanceMetric};
use crate::error::{LynseError, Result};
use crate::quantizer::{self, Quantizer, QuantizerType};
use super::{IndexConfig, IndexParams, IndexType, SearchParams, VectorIndex};
use serde::{Deserialize, Serialize};

/// Flat index: exhaustive search over all vectors.
pub struct FlatIndex {
    config: IndexConfig,
    quantizer: Box<dyn Quantizer>,
    /// Original f32 data (flattened: n_vectors * dim)
    data: Vec<f32>,
    /// Encoded data for distance computation
    encoded_data: Vec<f32>,
    /// Vector IDs
    ids: Vec<u64>,
    /// Whether index has been built
    trained: bool,
}

impl FlatIndex {
    pub fn new(metric: DistanceMetric, quant_type: QuantizerType) -> Self {
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
                index_type: IndexType::Flat,
                distance_metric: metric,
                quantizer_type: quant_type,
                dimension: 0,
                params: IndexParams::Flat,
            },
            quantizer,
            data: Vec::new(),
            encoded_data: Vec::new(),
            ids: Vec::new(),
            trained: false,
        }
    }
}

impl VectorIndex for FlatIndex {
    fn build(
        &mut self,
        vectors: &[f32],
        n_vectors: usize,
        dim: usize,
        ids: Option<&[u64]>,
    ) -> Result<()> {
        self.config.dimension = dim;

        // Assign IDs
        self.ids = match ids {
            Some(id_slice) => id_slice.to_vec(),
            None => (0..n_vectors as u64).collect(),
        };

        // Store original data
        self.data = vectors.to_vec();

        // Train quantizer and encode
        if self.config.quantizer_type != QuantizerType::None {
            self.quantizer.fit(vectors, n_vectors, dim)?;
            let encoded_bytes = self.quantizer.encode(vectors, n_vectors, dim)?;
            self.encoded_data = self.quantizer.decode(&encoded_bytes, n_vectors, dim)?;
        } else {
            self.encoded_data = vectors.to_vec();
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

        // Encode query if quantizer is used
        let encoded_query = if self.config.quantizer_type != QuantizerType::None {
            let encoded_bytes = self.quantizer.encode(query, 1, dim)?;
            self.quantizer.decode(&encoded_bytes, 1, dim)?
        } else {
            query.to_vec()
        };

        // Apply subset filter if provided
        let (search_data, search_ids): (Vec<f32>, Vec<u64>) =
            if let Some(ref subset) = params.subset_indices {
                let mut filtered_data = Vec::new();
                let mut filtered_ids = Vec::new();
                for (i, &id) in self.ids.iter().enumerate() {
                    if subset.contains(&id) {
                        let start = i * dim;
                        filtered_data.extend_from_slice(&self.encoded_data[start..start + dim]);
                        filtered_ids.push(id);
                    }
                }
                (filtered_data, filtered_ids)
            } else {
                (self.encoded_data.clone(), self.ids.clone())
            };

        if search_ids.is_empty() {
            return Ok((vec![], vec![]));
        }

        // Compute distances and get top-k
        let (top_indices, top_dists) = distance::top_k_search(
            &encoded_query,
            &search_data,
            dim,
            k,
            self.config.distance_metric,
        );

        // Map local indices to actual IDs
        let result_ids: Vec<u64> = top_indices
            .iter()
            .map(|&idx| search_ids[idx as usize])
            .collect();

        Ok((result_ids, top_dists))
    }

    fn delete(&mut self, ids: &[u64]) -> Result<()> {
        let dim = self.config.dimension;
        let mut new_data = Vec::new();
        let mut new_encoded = Vec::new();
        let mut new_ids = Vec::new();

        for (i, &id) in self.ids.iter().enumerate() {
            if !ids.contains(&id) {
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

        self.data.extend_from_slice(vectors);
        self.ids.extend_from_slice(ids);

        // Encode new vectors
        if self.config.quantizer_type != QuantizerType::None {
            let encoded_bytes = self.quantizer.encode(vectors, n_vectors, dim)?;
            let decoded = self.quantizer.decode(&encoded_bytes, n_vectors, dim)?;
            self.encoded_data.extend_from_slice(&decoded);
        } else {
            self.encoded_data.extend_from_slice(vectors);
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
        let state = FlatState {
            data: self.data.clone(),
            encoded_data: self.encoded_data.clone(),
            ids: self.ids.clone(),
            config: self.config.clone(),
            trained: self.trained,
        };
        bincode::serialize(&state).map_err(|e| LynseError::Serialization(e.to_string()))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let state: FlatState =
            bincode::deserialize(data).map_err(|e| LynseError::Serialization(e.to_string()))?;
        self.data = state.data;
        self.encoded_data = state.encoded_data;
        self.ids = state.ids;
        self.config = state.config;
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
        format!("flat-{}{}", metric, quant)
    }
}

#[derive(Serialize, Deserialize)]
struct FlatState {
    data: Vec<f32>,
    encoded_data: Vec<f32>,
    ids: Vec<u64>,
    config: IndexConfig,
    trained: bool,
}
