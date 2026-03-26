//! Vector quantization module.
//!
//! Implements: NoQuantizer, ScalarQuantizer (SQ8), BinaryQuantizer, ProductQuantizer (PQ).
//! Designed for billion-scale datasets with 700-3500 dimensional vectors.

use crate::error::{LynseError, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Quantizer type enum for configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizerType {
    None,
    Scalar,  // SQ8
    Binary,
    Product, // PQ
}

impl QuantizerType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(Self::None),
            "sq" | "sq8" | "scalar" => Some(Self::Scalar),
            "binary" => Some(Self::Binary),
            "pq" | "product" => Some(Self::Product),
            _ => None,
        }
    }
}

/// Trait for all quantizers.
pub trait Quantizer: Send + Sync {
    /// Train the quantizer on training data.
    fn fit(&mut self, data: &[f32], n_vectors: usize, dim: usize) -> Result<()>;

    /// Encode vectors into quantized form.
    /// Returns the encoded bytes.
    fn encode(&self, data: &[f32], n_vectors: usize, dim: usize) -> Result<Vec<u8>>;

    /// Decode quantized vectors back to f32.
    fn decode(&self, codes: &[u8], n_vectors: usize, dim: usize) -> Result<Vec<f32>>;

    /// Whether the quantizer has been trained.
    fn is_trained(&self) -> bool;

    /// The encoded size per vector in bytes.
    fn encoded_size(&self, dim: usize) -> usize;

    /// Serialize quantizer state.
    fn serialize(&self) -> Result<Vec<u8>>;

    /// Deserialize quantizer state.
    fn deserialize(&mut self, data: &[u8]) -> Result<()>;

    /// Get quantizer type.
    fn quantizer_type(&self) -> QuantizerType;
}

// ─── No Quantizer ────────────────────────────────────────────────────────────

/// Pass-through quantizer that performs no compression.
#[derive(Debug, Clone, Default)]
pub struct NoQuantizer;

impl Quantizer for NoQuantizer {
    fn fit(&mut self, _data: &[f32], _n_vectors: usize, _dim: usize) -> Result<()> {
        Ok(())
    }

    fn encode(&self, data: &[f32], _n_vectors: usize, _dim: usize) -> Result<Vec<u8>> {
        // Store f32 data as raw bytes
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        Ok(bytes)
    }

    fn decode(&self, codes: &[u8], _n_vectors: usize, _dim: usize) -> Result<Vec<f32>> {
        if codes.len() % 4 != 0 {
            return Err(LynseError::InvalidArgument(
                "Invalid byte length for f32 decode".into(),
            ));
        }
        let floats: Vec<f32> = codes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Ok(floats)
    }

    fn is_trained(&self) -> bool {
        true
    }

    fn encoded_size(&self, dim: usize) -> usize {
        dim * 4 // f32 = 4 bytes
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        Ok(vec![])
    }

    fn deserialize(&mut self, _data: &[u8]) -> Result<()> {
        Ok(())
    }

    fn quantizer_type(&self) -> QuantizerType {
        QuantizerType::None
    }
}

// ─── Scalar Quantizer (SQ8) ─────────────────────────────────────────────────

/// 8-bit scalar quantizer: maps each dimension to [0, 255].
/// Reduces memory by 4x compared to f32 while preserving distance ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizer {
    bits: u8,
    min_val: Option<Vec<f32>>,
    max_val: Option<Vec<f32>>,
    scale: Option<Vec<f32>>,
    trained: bool,
}

impl ScalarQuantizer {
    pub fn new(bits: u8) -> Self {
        Self {
            bits,
            min_val: None,
            max_val: None,
            scale: None,
            trained: false,
        }
    }
}

impl Default for ScalarQuantizer {
    fn default() -> Self {
        Self::new(8)
    }
}

impl Quantizer for ScalarQuantizer {
    fn fit(&mut self, data: &[f32], n_vectors: usize, dim: usize) -> Result<()> {
        if n_vectors == 0 || dim == 0 {
            return Err(LynseError::InvalidArgument("Empty training data".into()));
        }

        let max_val_int = (1u32 << self.bits) - 1;

        // Compute min/max per dimension using parallel reduction
        let mut min_val = vec![f32::MAX; dim];
        let mut max_val = vec![f32::MIN; dim];

        for i in 0..n_vectors {
            let base = i * dim;
            for d in 0..dim {
                let v = data[base + d];
                if v < min_val[d] {
                    min_val[d] = v;
                }
                if v > max_val[d] {
                    max_val[d] = v;
                }
            }
        }

        // Compute scale factors
        let scale: Vec<f32> = min_val
            .iter()
            .zip(max_val.iter())
            .map(|(&mn, &mx)| {
                let range = mx - mn;
                if range == 0.0 {
                    1.0
                } else {
                    range / max_val_int as f32
                }
            })
            .collect();

        self.min_val = Some(min_val);
        self.max_val = Some(max_val);
        self.scale = Some(scale);
        self.trained = true;

        Ok(())
    }

    fn encode(&self, data: &[f32], n_vectors: usize, dim: usize) -> Result<Vec<u8>> {
        let min_val = self
            .min_val
            .as_ref()
            .ok_or_else(|| LynseError::QuantizerNotTrained)?;
        let scale = self
            .scale
            .as_ref()
            .ok_or_else(|| LynseError::QuantizerNotTrained)?;

        let max_code = ((1u32 << self.bits) - 1) as f32;

        let mut codes = vec![0u8; n_vectors * dim];

        // Parallel encode for large datasets
        if n_vectors > 1000 {
            codes
                .par_chunks_mut(dim)
                .enumerate()
                .for_each(|(i, chunk)| {
                    let base = i * dim;
                    for d in 0..dim {
                        let normalized = (data[base + d] - min_val[d]) / scale[d];
                        chunk[d] = normalized.clamp(0.0, max_code) as u8;
                    }
                });
        } else {
            for i in 0..n_vectors {
                let base = i * dim;
                for d in 0..dim {
                    let normalized = (data[base + d] - min_val[d]) / scale[d];
                    codes[i * dim + d] = normalized.clamp(0.0, max_code) as u8;
                }
            }
        }

        Ok(codes)
    }

    fn decode(&self, codes: &[u8], n_vectors: usize, dim: usize) -> Result<Vec<f32>> {
        let min_val = self
            .min_val
            .as_ref()
            .ok_or_else(|| LynseError::QuantizerNotTrained)?;
        let scale = self
            .scale
            .as_ref()
            .ok_or_else(|| LynseError::QuantizerNotTrained)?;

        let mut data = vec![0.0f32; n_vectors * dim];

        if n_vectors > 1000 {
            data.par_chunks_mut(dim)
                .enumerate()
                .for_each(|(i, chunk)| {
                    let base = i * dim;
                    for d in 0..dim {
                        chunk[d] = codes[base + d] as f32 * scale[d] + min_val[d];
                    }
                });
        } else {
            for i in 0..n_vectors {
                let base = i * dim;
                for d in 0..dim {
                    data[base + d] = codes[base + d] as f32 * scale[d] + min_val[d];
                }
            }
        }

        Ok(data)
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn encoded_size(&self, dim: usize) -> usize {
        dim // 1 byte per dimension for SQ8
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| LynseError::Serialization(e.to_string()))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let loaded: ScalarQuantizer = bincode::deserialize(data)
            .map_err(|e| LynseError::Serialization(e.to_string()))?;
        *self = loaded;
        Ok(())
    }

    fn quantizer_type(&self) -> QuantizerType {
        QuantizerType::Scalar
    }
}

// ─── Binary Quantizer ────────────────────────────────────────────────────────

/// Binary quantizer: converts each dimension to a single bit.
/// Extremely compact: reduces memory by 32x compared to f32.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryQuantizer {
    threshold: f32,
}

impl BinaryQuantizer {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl Default for BinaryQuantizer {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Quantizer for BinaryQuantizer {
    fn fit(&mut self, _data: &[f32], _n_vectors: usize, _dim: usize) -> Result<()> {
        Ok(()) // No training needed
    }

    fn encode(&self, data: &[f32], n_vectors: usize, dim: usize) -> Result<Vec<u8>> {
        // Pack bits: each byte holds 8 dimensions
        let bytes_per_vector = (dim + 7) / 8;
        let mut codes = vec![0u8; n_vectors * bytes_per_vector];

        for i in 0..n_vectors {
            let base_in = i * dim;
            let base_out = i * bytes_per_vector;
            for d in 0..dim {
                if data[base_in + d] > self.threshold {
                    codes[base_out + d / 8] |= 1 << (d % 8);
                }
            }
        }

        Ok(codes)
    }

    fn decode(&self, codes: &[u8], n_vectors: usize, dim: usize) -> Result<Vec<f32>> {
        let bytes_per_vector = (dim + 7) / 8;
        let mut data = vec![0.0f32; n_vectors * dim];

        for i in 0..n_vectors {
            let base_in = i * bytes_per_vector;
            let base_out = i * dim;
            for d in 0..dim {
                if codes[base_in + d / 8] & (1 << (d % 8)) != 0 {
                    data[base_out + d] = 1.0;
                }
            }
        }

        Ok(data)
    }

    fn is_trained(&self) -> bool {
        true
    }

    fn encoded_size(&self, dim: usize) -> usize {
        (dim + 7) / 8 // Packed bits
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| LynseError::Serialization(e.to_string()))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let loaded: BinaryQuantizer = bincode::deserialize(data)
            .map_err(|e| LynseError::Serialization(e.to_string()))?;
        *self = loaded;
        Ok(())
    }

    fn quantizer_type(&self) -> QuantizerType {
        QuantizerType::Binary
    }
}

// ─── Product Quantizer (PQ) ─────────────────────────────────────────────────

/// Product quantizer: splits vector into sub-spaces and quantizes each independently.
/// Good balance between compression ratio and recall for high-dimensional vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    n_subspaces: usize,
    n_clusters: usize,
    subspace_size: usize,
    /// Codebooks: [n_subspaces][n_clusters][subspace_size] stored flat
    codebooks: Option<Vec<f32>>,
    trained: bool,
}

impl ProductQuantizer {
    pub fn new(n_subspaces: usize, n_clusters: usize) -> Self {
        Self {
            n_subspaces,
            n_clusters,
            subspace_size: 0,
            codebooks: None,
            trained: false,
        }
    }
}

impl Default for ProductQuantizer {
    fn default() -> Self {
        Self::new(8, 256)
    }
}

impl ProductQuantizer {
    /// Run K-means on a subspace of the data.
    fn kmeans_subspace(
        &self,
        data: &[f32],
        n_vectors: usize,
        dim: usize,
        subspace_idx: usize,
    ) -> Vec<f32> {
        let sub_dim = self.subspace_size;
        let start_dim = subspace_idx * sub_dim;
        let n_clusters = self.n_clusters.min(n_vectors);

        // Extract subspace data
        let mut sub_data = vec![0.0f32; n_vectors * sub_dim];
        for i in 0..n_vectors {
            for d in 0..sub_dim {
                sub_data[i * sub_dim + d] = data[i * dim + start_dim + d];
            }
        }

        // Simple K-means (Lloyd's algorithm)
        let max_iter = 100;
        let mut centroids = vec![0.0f32; n_clusters * sub_dim];

        // Initialize centroids with K-means++ style
        // First centroid: random
        let first = rand::random::<usize>() % n_vectors;
        centroids[..sub_dim].copy_from_slice(&sub_data[first * sub_dim..(first + 1) * sub_dim]);

        // Remaining centroids: proportional to distance²
        for c in 1..n_clusters {
            let mut max_dist = 0.0f32;
            let mut best_idx = 0;
            for i in 0..n_vectors {
                let mut min_dist = f32::MAX;
                for prev_c in 0..c {
                    let mut dist = 0.0f32;
                    for d in 0..sub_dim {
                        let diff =
                            sub_data[i * sub_dim + d] - centroids[prev_c * sub_dim + d];
                        dist += diff * diff;
                    }
                    min_dist = min_dist.min(dist);
                }
                if min_dist > max_dist {
                    max_dist = min_dist;
                    best_idx = i;
                }
            }
            let src_start = best_idx * sub_dim;
            let dst_start = c * sub_dim;
            centroids[dst_start..dst_start + sub_dim]
                .copy_from_slice(&sub_data[src_start..src_start + sub_dim]);
        }

        // Iterate
        let mut assignments = vec![0u32; n_vectors];
        for _iter in 0..max_iter {
            // Assign
            let mut changed = false;
            for i in 0..n_vectors {
                let mut best_c = 0u32;
                let mut best_dist = f32::MAX;
                for c in 0..n_clusters {
                    let mut dist = 0.0f32;
                    for d in 0..sub_dim {
                        let diff =
                            sub_data[i * sub_dim + d] - centroids[c * sub_dim + d];
                        dist += diff * diff;
                    }
                    if dist < best_dist {
                        best_dist = dist;
                        best_c = c as u32;
                    }
                }
                if assignments[i] != best_c {
                    assignments[i] = best_c;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            let mut counts = vec![0u32; n_clusters];
            let mut new_centroids = vec![0.0f32; n_clusters * sub_dim];

            for i in 0..n_vectors {
                let c = assignments[i] as usize;
                counts[c] += 1;
                for d in 0..sub_dim {
                    new_centroids[c * sub_dim + d] += sub_data[i * sub_dim + d];
                }
            }

            for c in 0..n_clusters {
                if counts[c] > 0 {
                    for d in 0..sub_dim {
                        new_centroids[c * sub_dim + d] /= counts[c] as f32;
                    }
                }
            }

            centroids = new_centroids;
        }

        centroids
    }
}

impl Quantizer for ProductQuantizer {
    fn fit(&mut self, data: &[f32], n_vectors: usize, dim: usize) -> Result<()> {
        if n_vectors == 0 || dim == 0 {
            return Err(LynseError::InvalidArgument("Empty training data".into()));
        }

        self.subspace_size = dim / self.n_subspaces;
        if dim % self.n_subspaces != 0 {
            return Err(LynseError::InvalidArgument(format!(
                "Dimension {} not divisible by n_subspaces {}",
                dim, self.n_subspaces
            )));
        }

        let n_clusters = self.n_clusters.min(n_vectors);
        let sub_dim = self.subspace_size;

        // Train each subspace's codebook (parallel across subspaces)
        let codebooks_parts: Vec<Vec<f32>> = (0..self.n_subspaces)
            .into_par_iter()
            .map(|s| self.kmeans_subspace(data, n_vectors, dim, s))
            .collect();

        // Flatten codebooks: [n_subspaces * n_clusters * subspace_size]
        let mut codebooks = Vec::with_capacity(self.n_subspaces * n_clusters * sub_dim);
        for part in codebooks_parts {
            codebooks.extend_from_slice(&part);
        }

        self.codebooks = Some(codebooks);
        self.trained = true;

        Ok(())
    }

    fn encode(&self, data: &[f32], n_vectors: usize, dim: usize) -> Result<Vec<u8>> {
        let codebooks = self
            .codebooks
            .as_ref()
            .ok_or_else(|| LynseError::QuantizerNotTrained)?;

        let sub_dim = self.subspace_size;
        let n_clusters = self.n_clusters.min(codebooks.len() / (self.n_subspaces * sub_dim));

        // Each vector encodes to n_subspaces bytes (cluster indices)
        let mut codes = vec![0u8; n_vectors * self.n_subspaces];

        let encode_fn = |i: usize, codes_row: &mut [u8]| {
            for s in 0..self.n_subspaces {
                let data_offset = i * dim + s * sub_dim;
                let cb_offset = s * n_clusters * sub_dim;

                let mut best_c = 0u8;
                let mut best_dist = f32::MAX;

                for c in 0..n_clusters {
                    let mut dist = 0.0f32;
                    for d in 0..sub_dim {
                        let diff = data[data_offset + d]
                            - codebooks[cb_offset + c * sub_dim + d];
                        dist += diff * diff;
                    }
                    if dist < best_dist {
                        best_dist = dist;
                        best_c = c as u8;
                    }
                }
                codes_row[s] = best_c;
            }
        };

        if n_vectors > 1000 {
            codes
                .par_chunks_mut(self.n_subspaces)
                .enumerate()
                .for_each(|(i, row)| encode_fn(i, row));
        } else {
            for i in 0..n_vectors {
                let start = i * self.n_subspaces;
                let end = start + self.n_subspaces;
                encode_fn(i, &mut codes[start..end]);
            }
        }

        Ok(codes)
    }

    fn decode(&self, codes: &[u8], n_vectors: usize, dim: usize) -> Result<Vec<f32>> {
        let codebooks = self
            .codebooks
            .as_ref()
            .ok_or_else(|| LynseError::QuantizerNotTrained)?;

        let sub_dim = self.subspace_size;
        let n_clusters = self.n_clusters.min(codebooks.len() / (self.n_subspaces * sub_dim));

        let mut data = vec![0.0f32; n_vectors * dim];

        for i in 0..n_vectors {
            for s in 0..self.n_subspaces {
                let c = codes[i * self.n_subspaces + s] as usize;
                let cb_offset = s * n_clusters * sub_dim + c * sub_dim;
                let data_offset = i * dim + s * sub_dim;

                data[data_offset..data_offset + sub_dim]
                    .copy_from_slice(&codebooks[cb_offset..cb_offset + sub_dim]);
            }
        }

        Ok(data)
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn encoded_size(&self, _dim: usize) -> usize {
        self.n_subspaces // 1 byte per subspace
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| LynseError::Serialization(e.to_string()))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let loaded: ProductQuantizer = bincode::deserialize(data)
            .map_err(|e| LynseError::Serialization(e.to_string()))?;
        *self = loaded;
        Ok(())
    }

    fn quantizer_type(&self) -> QuantizerType {
        QuantizerType::Product
    }
}

/// Factory function to create a quantizer from type string.
pub fn create_quantizer(quantizer_type: &str) -> Result<Box<dyn Quantizer>> {
    match QuantizerType::from_str(quantizer_type) {
        Some(QuantizerType::None) => Ok(Box::new(NoQuantizer)),
        Some(QuantizerType::Scalar) => Ok(Box::new(ScalarQuantizer::default())),
        Some(QuantizerType::Binary) => Ok(Box::new(BinaryQuantizer::default())),
        Some(QuantizerType::Product) => Ok(Box::new(ProductQuantizer::default())),
        None => Err(LynseError::InvalidArgument(format!(
            "Unknown quantizer type: {}",
            quantizer_type
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_quantizer() {
        let q = NoQuantizer;
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let encoded = q.encode(&data, 2, 3).unwrap();
        let decoded = q.decode(&encoded, 2, 3).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_scalar_quantizer() {
        let mut q = ScalarQuantizer::default();
        let data = vec![0.0f32, 0.5, 1.0, 0.2, 0.8, 0.4];
        q.fit(&data, 2, 3).unwrap();
        let encoded = q.encode(&data, 2, 3).unwrap();
        assert_eq!(encoded.len(), 6); // 2 vectors * 3 dims
        let decoded = q.decode(&encoded, 2, 3).unwrap();
        // Check approximate reconstruction
        for (a, b) in data.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.01, "Expected {} ≈ {}", a, b);
        }
    }

    #[test]
    fn test_binary_quantizer() {
        let q = BinaryQuantizer::default();
        let data = vec![0.1f32, 0.9, 0.3, 0.7];
        let encoded = q.encode(&data, 1, 4).unwrap();
        assert_eq!(encoded.len(), 1); // 4 bits packed in 1 byte
        // 0.1 < 0.5 → 0, 0.9 > 0.5 → 1, 0.3 < 0.5 → 0, 0.7 > 0.5 → 1
        assert_eq!(encoded[0] & 0x0F, 0b1010);
    }
}
