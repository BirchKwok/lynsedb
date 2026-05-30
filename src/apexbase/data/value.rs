//! Value representation for database records

use super::DataType;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// A value that can be stored in the database
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Bool(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    String(String),
    Binary(Vec<u8>),
    /// Fixed-size float32 vector — auto-detected from np.ndarray on store().
    /// Bytes are raw little-endian f32: len = dim * 4.
    FixedList(Vec<u8>),
    Json(serde_json::Value),
    Timestamp(i64),
    Date(i32),
    Array(Vec<Value>),
}

impl Value {
    /// Get the data type of this value
    pub fn data_type(&self) -> DataType {
        match self {
            Value::Null => DataType::Null,
            Value::Bool(_) => DataType::Bool,
            Value::Int8(_) => DataType::Int8,
            Value::Int16(_) => DataType::Int16,
            Value::Int32(_) => DataType::Int32,
            Value::Int64(_) => DataType::Int64,
            Value::UInt8(_) => DataType::UInt8,
            Value::UInt16(_) => DataType::UInt16,
            Value::UInt32(_) => DataType::UInt32,
            Value::UInt64(_) => DataType::UInt64,
            Value::Float32(_) => DataType::Float32,
            Value::Float64(_) => DataType::Float64,
            Value::String(_) => DataType::String,
            Value::Binary(_) => DataType::Binary,
            Value::FixedList(_) => DataType::Binary,
            Value::Json(_) => DataType::Json,
            Value::Timestamp(_) => DataType::Timestamp,
            Value::Date(_) => DataType::Date,
            Value::Array(_) => DataType::Array,
        }
    }

    /// Check if this value is null
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Try to convert to i64
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int8(v) => Some(*v as i64),
            Value::Int16(v) => Some(*v as i64),
            Value::Int32(v) => Some(*v as i64),
            Value::Int64(v) => Some(*v),
            Value::UInt8(v) => Some(*v as i64),
            Value::UInt16(v) => Some(*v as i64),
            Value::UInt32(v) => Some(*v as i64),
            Value::UInt64(v) => {
                if *v <= i64::MAX as u64 {
                    Some(*v as i64)
                } else {
                    None
                }
            }
            Value::Timestamp(v) => Some(*v),
            Value::Date(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Try to convert to f64
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float32(v) => Some(*v as f64),
            Value::Float64(v) => Some(*v),
            Value::Int8(v) => Some(*v as f64),
            Value::Int16(v) => Some(*v as f64),
            Value::Int32(v) => Some(*v as f64),
            Value::Int64(v) => Some(*v as f64),
            Value::UInt8(v) => Some(*v as f64),
            Value::UInt16(v) => Some(*v as f64),
            Value::UInt32(v) => Some(*v as f64),
            Value::UInt64(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Try to convert to string
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to convert to bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Convert to string representation (for display purposes)
    pub fn to_string_value(&self) -> String {
        match self {
            Value::Null => "null".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Int8(v) => v.to_string(),
            Value::Int16(v) => v.to_string(),
            Value::Int32(v) => v.to_string(),
            Value::Int64(v) => v.to_string(),
            Value::UInt8(v) => v.to_string(),
            Value::UInt16(v) => v.to_string(),
            Value::UInt32(v) => v.to_string(),
            Value::UInt64(v) => v.to_string(),
            Value::Float32(v) => v.to_string(),
            Value::Float64(v) => v.to_string(),
            Value::String(s) => s.clone(),
            Value::Binary(b) => {
                if let Ok(s) = std::str::from_utf8(b) {
                    s.to_string()
                } else {
                    format!("<binary_data_{}bytes>", b.len())
                }
            }
            Value::FixedList(b) => format!("<vector_{}dims>", b.len() / 4),
            Value::Json(j) => j.to_string(),
            Value::Timestamp(t) => t.to_string(),
            Value::Date(d) => d.to_string(),
            Value::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_string_value()).collect();
                format!("[{}]", items.join(", "))
            }
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        bincode::deserialize(bytes).ok()
    }

    /// Infer data type from a Python-like value (for compatibility)
    pub fn infer_from_python_value(val: &serde_json::Value) -> Self {
        match val {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int64(i)
                } else if let Some(f) = n.as_f64() {
                    Value::Float64(f)
                } else {
                    Value::String(n.to_string())
                }
            }
            serde_json::Value::String(s) => Value::String(s.clone()),
            serde_json::Value::Array(arr) => {
                Value::Array(arr.iter().map(Value::infer_from_python_value).collect())
            }
            serde_json::Value::Object(_) => Value::Json(val.clone()),
        }
    }

    /// Convert to serde_json::Value
    pub fn to_json_value(&self) -> serde_json::Value {
        match self {
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(*b),
            Value::Int8(v) => serde_json::json!(*v),
            Value::Int16(v) => serde_json::json!(*v),
            Value::Int32(v) => serde_json::json!(*v),
            Value::Int64(v) => serde_json::json!(*v),
            Value::UInt8(v) => serde_json::json!(*v),
            Value::UInt16(v) => serde_json::json!(*v),
            Value::UInt32(v) => serde_json::json!(*v),
            Value::UInt64(v) => serde_json::json!(*v),
            Value::Float32(v) => serde_json::json!(*v),
            Value::Float64(v) => serde_json::json!(*v),
            Value::String(s) => serde_json::Value::String(s.clone()),
            Value::Binary(b) => serde_json::json!(b),
            Value::FixedList(b) => serde_json::json!(b),
            Value::Json(j) => j.clone(),
            Value::Timestamp(t) => serde_json::json!(*t),
            Value::Date(d) => serde_json::json!(*d),
            Value::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(|v| v.to_json_value()).collect())
            }
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Value::Null, Value::Null) => Some(Ordering::Equal),
            (Value::Null, _) => Some(Ordering::Less),
            (_, Value::Null) => Some(Ordering::Greater),
            (Value::Bool(a), Value::Bool(b)) => a.partial_cmp(b),
            (Value::Int64(a), Value::Int64(b)) => a.partial_cmp(b),
            (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
            (Value::String(a), Value::String(b)) => a.partial_cmp(b),
            // Try numeric comparison
            (a, b) => {
                if let (Some(fa), Some(fb)) = (a.as_f64(), b.as_f64()) {
                    fa.partial_cmp(&fb)
                } else if let (Some(sa), Some(sb)) = (a.as_str(), b.as_str()) {
                    sa.partial_cmp(sb)
                } else {
                    None
                }
            }
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::Null
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string_value())
    }
}

// Conversion implementations
impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::Int32(v)
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int64(v)
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Float64(v)
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::String(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::String(v.to_string())
    }
}

impl From<Vec<u8>> for Value {
    fn from(v: Vec<u8>) -> Self {
        Value::Binary(v)
    }
}

impl From<serde_json::Value> for Value {
    fn from(v: serde_json::Value) -> Self {
        Value::infer_from_python_value(&v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_type() {
        assert_eq!(Value::Int64(42).data_type(), DataType::Int64);
        assert_eq!(Value::String("hello".into()).data_type(), DataType::String);
    }

    #[test]
    fn test_value_conversion() {
        assert_eq!(Value::Int64(42).as_i64(), Some(42));
        assert_eq!(Value::Float64(3.14).as_f64(), Some(3.14));
        assert_eq!(Value::String("hello".into()).as_str(), Some("hello"));
    }

    #[test]
    fn test_value_serialization() {
        let val = Value::Int64(42);
        let bytes = val.to_bytes();
        let restored = Value::from_bytes(&bytes);
        assert_eq!(restored, Some(val));
    }
}
