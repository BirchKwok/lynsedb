//! Data type definitions

use serde::{Deserialize, Serialize};

/// Supported data types in ApexBase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum DataType {
    Null = 0,
    Bool = 1,
    Int8 = 2,
    Int16 = 3,
    Int32 = 4,
    Int64 = 5,
    UInt8 = 6,
    UInt16 = 7,
    UInt32 = 8,
    UInt64 = 9,
    Float32 = 10,
    Float64 = 11,
    String = 12,
    Binary = 13,
    Json = 14,
    Timestamp = 15,
    Date = 16,
    Array = 17,
    Decimal = 18,
    Float16Vector = 19,
}

impl DataType {
    /// Get the fixed size of this data type in bytes, or None for variable-length types
    pub fn fixed_size(&self) -> Option<usize> {
        match self {
            DataType::Null => Some(0),
            DataType::Bool => Some(1),
            DataType::Int8 | DataType::UInt8 => Some(1),
            DataType::Int16 | DataType::UInt16 => Some(2),
            DataType::Int32 | DataType::UInt32 | DataType::Float32 => Some(4),
            DataType::Int64 | DataType::UInt64 | DataType::Float64 | DataType::Timestamp => Some(8),
            DataType::Date => Some(4),
            DataType::Decimal => Some(16), // i128 for decimal storage
            DataType::String | DataType::Binary | DataType::Json | DataType::Array => None,
            DataType::Float16Vector => None,
        }
    }

    /// Check if this type is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64
                | DataType::Decimal
        )
    }

    /// Check if this type is variable-length
    pub fn is_variable_length(&self) -> bool {
        matches!(
            self,
            DataType::String | DataType::Binary | DataType::Json | DataType::Array
        )
    }

    /// Check if this type is a decimal type
    pub fn is_decimal(&self) -> bool {
        matches!(self, DataType::Decimal)
    }

    /// Convert from SQL type string (for compatibility with DuckDB types)
    pub fn from_sql_type(sql_type: &str) -> Self {
        let sql_type = sql_type.to_uppercase();
        match sql_type.as_str() {
            "BOOLEAN" | "BOOL" => DataType::Bool,
            "TINYINT" | "INT1" => DataType::Int8,
            "SMALLINT" | "INT2" => DataType::Int16,
            "INTEGER" | "INT" | "INT4" => DataType::Int32,
            "BIGINT" | "INT8" | "LONG" => DataType::Int64,
            "UTINYINT" => DataType::UInt8,
            "USMALLINT" => DataType::UInt16,
            "UINTEGER" => DataType::UInt32,
            "UBIGINT" => DataType::UInt64,
            "FLOAT" | "REAL" | "FLOAT4" => DataType::Float32,
            "DOUBLE" | "FLOAT8" => DataType::Float64,
            "VARCHAR" | "TEXT" | "STRING" | "CHAR" => DataType::String,
            "BLOB" | "BYTEA" | "BINARY" | "VARBINARY" => DataType::Binary,
            "JSON" => DataType::Json,
            "DECIMAL" | "NUMERIC" => DataType::Decimal,
            "TIMESTAMP" | "DATETIME" => DataType::Timestamp,
            "DATE" => DataType::Date,
            _ => DataType::String, // Default to String for unknown types
        }
    }

    /// Convert to SQL type string
    pub fn to_sql_type(&self) -> &'static str {
        match self {
            DataType::Null => "NULL",
            DataType::Bool => "BOOLEAN",
            DataType::Int8 => "TINYINT",
            DataType::Int16 => "SMALLINT",
            DataType::Int32 => "INTEGER",
            DataType::Int64 => "BIGINT",
            DataType::UInt8 => "UTINYINT",
            DataType::UInt16 => "USMALLINT",
            DataType::UInt32 => "UINTEGER",
            DataType::UInt64 => "UBIGINT",
            DataType::Float32 => "FLOAT",
            DataType::Float64 => "DOUBLE",
            DataType::String => "VARCHAR",
            DataType::Binary => "BLOB",
            DataType::Json => "JSON",
            DataType::Timestamp => "TIMESTAMP",
            DataType::Date => "DATE",
            DataType::Array => "ARRAY",
            DataType::Decimal => "DECIMAL",
            DataType::Float16Vector => "FLOAT16_VECTOR",
        }
    }
}

impl Default for DataType {
    fn default() -> Self {
        DataType::Null
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sql_type())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_sql_type() {
        assert_eq!(DataType::from_sql_type("BIGINT"), DataType::Int64);
        assert_eq!(DataType::from_sql_type("VARCHAR"), DataType::String);
        assert_eq!(DataType::from_sql_type("DOUBLE"), DataType::Float64);
        assert_eq!(DataType::from_sql_type("BOOLEAN"), DataType::Bool);
    }

    #[test]
    fn test_fixed_size() {
        assert_eq!(DataType::Int64.fixed_size(), Some(8));
        assert_eq!(DataType::String.fixed_size(), None);
    }
}
