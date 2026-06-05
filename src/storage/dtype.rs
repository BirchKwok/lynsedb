use crate::error::{LynseError, Result};
use half::f16;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VectorDtype {
    F32,
    F16,
}

impl VectorDtype {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" | "float" => Ok(Self::F32),
            "f16" | "float16" | "half" | "fp16" => Ok(Self::F16),
            other => Err(LynseError::InvalidArgument(format!(
                "unsupported vector dtype '{}'; expected float32/f32 or float16/f16",
                other
            ))),
        }
    }

    #[inline]
    pub fn byte_width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }

    pub fn storage_name(self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::F16 => "float16",
        }
    }

    pub fn short_name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }

    pub fn numpy_name(self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::F16 => "float16",
        }
    }
}

impl Default for VectorDtype {
    fn default() -> Self {
        Self::F32
    }
}

#[inline]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

#[inline]
pub fn f32_to_f16_bits(value: f32) -> u16 {
    f16::from_f32(value).to_bits()
}

pub fn encode_f32_slice_as_le_bytes(data: &[f32], dtype: VectorDtype) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * dtype.byte_width());
    match dtype {
        VectorDtype::F32 => {
            for value in data {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        VectorDtype::F16 => {
            for value in data {
                bytes.extend_from_slice(&f32_to_f16_bits(*value).to_le_bytes());
            }
        }
    }
    bytes
}

pub fn decode_f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect()
}

pub fn decode_f32_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
