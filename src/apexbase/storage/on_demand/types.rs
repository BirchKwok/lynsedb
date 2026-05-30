// Core types: ColumnType, ColumnValue, ColumnDef, FileSchema, ColumnData

/// Column data type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ColumnType {
    Null = TYPE_NULL,
    Bool = TYPE_BOOL,
    Int8 = TYPE_INT8,
    Int16 = TYPE_INT16,
    Int32 = TYPE_INT32,
    Int64 = TYPE_INT64,
    UInt8 = TYPE_UINT8,
    UInt16 = TYPE_UINT16,
    UInt32 = TYPE_UINT32,
    UInt64 = TYPE_UINT64,
    Float32 = TYPE_FLOAT32,
    Float64 = TYPE_FLOAT64,
    String = TYPE_STRING,
    Binary = TYPE_BINARY,
    StringDict = TYPE_STRING_DICT,  // Dictionary-encoded string for low-cardinality columns
    Timestamp = TYPE_TIMESTAMP,      // Timestamp (i64 microseconds since Unix epoch)
    Date = TYPE_DATE,                // Date (i64 days since Unix epoch, stored as i64 for alignment)
    /// Fixed-size list of f32 — stored as contiguous raw bytes (dim * 4 per row, no offset array).
    /// Semantically equivalent to Arrow FixedSizeList<Float32>.
    FixedList = TYPE_FIXED_LIST,
    /// Fixed-size list of f16 — stored as contiguous raw bytes (dim * 2 per row, no offset array).
    /// Half the storage of FixedList; decoded to f32 on read/distance computation.
    Float16List = TYPE_FLOAT16_LIST,
}

impl ColumnType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            TYPE_NULL => Some(ColumnType::Null),
            TYPE_BOOL => Some(ColumnType::Bool),
            TYPE_INT8 => Some(ColumnType::Int8),
            TYPE_INT16 => Some(ColumnType::Int16),
            TYPE_INT32 => Some(ColumnType::Int32),
            TYPE_INT64 => Some(ColumnType::Int64),
            TYPE_UINT8 => Some(ColumnType::UInt8),
            TYPE_UINT16 => Some(ColumnType::UInt16),
            TYPE_UINT32 => Some(ColumnType::UInt32),
            TYPE_UINT64 => Some(ColumnType::UInt64),
            TYPE_FLOAT32 => Some(ColumnType::Float32),
            TYPE_FLOAT64 => Some(ColumnType::Float64),
            TYPE_STRING => Some(ColumnType::String),
            TYPE_BINARY => Some(ColumnType::Binary),
            TYPE_STRING_DICT => Some(ColumnType::StringDict),
            TYPE_TIMESTAMP => Some(ColumnType::Timestamp),
            TYPE_DATE => Some(ColumnType::Date),
            TYPE_FIXED_LIST => Some(ColumnType::FixedList),
            TYPE_FLOAT16_LIST => Some(ColumnType::Float16List),
            _ => None,
        }
    }

    /// Fixed size in bytes (0 for variable-length types)
    pub fn fixed_size(&self) -> usize {
        match self {
            ColumnType::Null => 0,
            ColumnType::Bool => 1,
            ColumnType::Int8 | ColumnType::UInt8 => 1,
            ColumnType::Int16 | ColumnType::UInt16 => 2,
            ColumnType::Int32 | ColumnType::UInt32 | ColumnType::Float32 => 4,
            ColumnType::Int64 | ColumnType::UInt64 | ColumnType::Float64 | ColumnType::Timestamp | ColumnType::Date => 8,
            ColumnType::String | ColumnType::Binary | ColumnType::StringDict | ColumnType::FixedList | ColumnType::Float16List => 0,
        }
    }

    pub fn is_variable_length(&self) -> bool {
        matches!(self, ColumnType::String | ColumnType::Binary | ColumnType::StringDict | ColumnType::FixedList | ColumnType::Float16List)
    }
}

/// Generic column value for API
#[derive(Debug, Clone)]
pub enum ColumnValue {
    Null,
    Bool(bool),
    Int64(i64),
    Float64(f64),
    String(String),
    Binary(Vec<u8>),
    /// Raw f32 bytes for a FixedList (vector) column
    FixedList(Vec<u8>),
}

/// Column definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub dtype: ColumnType,
}

/// Schema definition (for API compatibility)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileSchema {
    pub columns: Vec<ColumnDef>,
    name_to_idx: HashMap<String, usize>,
}

impl FileSchema {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            name_to_idx: HashMap::new(),
        }
    }

    pub fn add_column(&mut self, name: &str, dtype: ColumnType) -> usize {
        if let Some(&idx) = self.name_to_idx.get(name) {
            return idx;
        }
        let idx = self.columns.len();
        self.columns.push(ColumnDef {
            name: name.to_string(),
            dtype,
        });
        self.name_to_idx.insert(name.to_string(), idx);
        idx
    }

    pub fn get_index(&self, name: &str) -> Option<usize> {
        self.name_to_idx.get(name).copied()
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }
}

// ============================================================================
// Column Data Storage
// ============================================================================

/// Efficient column data storage
#[derive(Debug, Clone)]
pub enum ColumnData {
    Bool {
        data: Vec<u8>,  // Packed bits
        len: usize,
    },
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    String {
        offsets: Vec<u32>,  // Offset into data
        data: Vec<u8>,      // UTF-8 bytes
    },
    Binary {
        offsets: Vec<u32>,  // Offset into data
        data: Vec<u8>,      // Raw bytes
    },
    /// Dictionary-encoded string column (DuckDB-style optimization)
    /// - indices: u32 index per row pointing into dictionary
    /// - dict_offsets: offset into dict_data for each unique string
    /// - dict_data: concatenated unique string bytes
    /// 
    /// Benefits:
    /// - GROUP BY/DISTINCT work on integer indices instead of string hashing
    /// - Much smaller storage for low-cardinality columns
    /// - Faster comparisons (integer vs string)
    StringDict {
        indices: Vec<u32>,      // Per-row dictionary index (0 = NULL)
        dict_offsets: Vec<u32>, // Offsets into dict_data
        dict_data: Vec<u8>,     // Unique string bytes
    },
    /// Fixed-size list of f32 — contiguous raw bytes, no offset array.
    /// data.len() == row_count * dim * 4
    FixedList {
        data: Vec<u8>,  // raw little-endian f32 bytes
        dim: u32,       // number of f32 elements per row
    },
    /// Fixed-size list of f16 — contiguous raw LE u16 bytes, no offset array.
    /// data.len() == row_count * dim * 2
    Float16List {
        data: Vec<u8>,  // raw little-endian f16 (u16) bytes
        dim: u32,       // number of f16 elements per row
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// f16 ↔ f32 conversion utilities (software, no external crate)
// ─────────────────────────────────────────────────────────────────────────────

/// Encode f32 as IEEE 754 half-precision bits (u16 LE).
#[inline]
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_f32 = ((bits >> 23) & 0xFF) as i32;
    let mant_f32 = bits & 0x7FFFFF;
    if exp_f32 == 0xFF {
        let mant_f16 = if mant_f32 != 0 { 0x0200u16 } else { 0u16 };
        return sign | 0x7C00 | mant_f16;
    }
    let exp = exp_f32 - 127 + 15;
    if exp >= 31 { return sign | 0x7C00; }
    if exp <= 0 {
        if exp < -10 { return sign; }
        let mant = (mant_f32 | 0x800000) >> (1 - exp);
        return sign | (mant >> 13) as u16;
    }
    sign | ((exp as u16) << 10) | ((mant_f32 >> 13) as u16)
}

/// Decode IEEE 754 half-precision bits (u16) to f32.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp  = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 { sign }
        else {
            let mut e = 127u32 - 14;
            let mut m = mant;
            while m & 0x400 == 0 { m <<= 1; e = e.wrapping_sub(1); }
            sign | (e << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 31 {
        sign | (0xFF << 23) | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Convert f32 bytes (LE) to f16 bytes (LE).
/// Input length must be a multiple of 4; output is half the length.
pub fn f32_bytes_to_f16_bytes(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() / 2);
    for chunk in src.chunks_exact(4) {
        let f = f32::from_le_bytes(chunk.try_into().unwrap());
        let h = f32_to_f16(f);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// Convert f16 bytes (LE) to f32 bytes (LE) in-place into dst.
pub fn f16_bytes_to_f32_into(src: &[u8], dst: &mut [f32]) {
    for (i, chunk) in src.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes(chunk.try_into().unwrap());
        dst[i] = f16_to_f32(bits);
    }
}

/// Decode raw little-endian f32 bytes into owned f32 values.
pub fn f32_le_bytes_to_values(src: &[u8]) -> Vec<f32> {
    src.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Decode raw little-endian f16 bytes into owned f32 values.
pub fn f16_bytes_to_f32_values(src: &[u8]) -> Vec<f32> {
    src.chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes(chunk.try_into().unwrap());
            f16_to_f32(bits)
        })
        .collect()
}

impl ColumnData {
    pub fn new(dtype: ColumnType) -> Self {
        match dtype {
            ColumnType::Bool => ColumnData::Bool { data: Vec::new(), len: 0 },
            ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 | ColumnType::Int64 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 |
            ColumnType::Timestamp | ColumnType::Date => {
                ColumnData::Int64(Vec::new())
            }
            ColumnType::Float32 | ColumnType::Float64 => ColumnData::Float64(Vec::new()),
            ColumnType::String => ColumnData::String {
                offsets: vec![0],
                data: Vec::new(),
            },
            ColumnType::Binary => ColumnData::Binary {
                offsets: vec![0],
                data: Vec::new(),
            },
            ColumnType::StringDict => ColumnData::StringDict {
                indices: Vec::new(),
                dict_offsets: vec![0],
                dict_data: Vec::new(),
            },
            ColumnType::FixedList => ColumnData::FixedList { data: Vec::new(), dim: 0 },
            ColumnType::Float16List => ColumnData::Float16List { data: Vec::new(), dim: 0 },
            ColumnType::Null => ColumnData::Int64(Vec::new()),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Bool { len, .. } => *len,
            ColumnData::Int64(v) => v.len(),
            ColumnData::Float64(v) => v.len(),
            ColumnData::String { offsets, .. } => offsets.len().saturating_sub(1),
            ColumnData::Binary { offsets, .. } => offsets.len().saturating_sub(1),
            ColumnData::StringDict { indices, .. } => indices.len(),
            ColumnData::FixedList { data, dim } => if *dim == 0 { 0 } else { data.len() / (*dim as usize * 4) },
            ColumnData::Float16List { data, dim } => if *dim == 0 { 0 } else { data.len() / (*dim as usize * 2) },
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn column_type(&self) -> ColumnType {
        match self {
            ColumnData::Bool { .. } => ColumnType::Bool,
            ColumnData::Int64(_) => ColumnType::Int64,
            ColumnData::Float64(_) => ColumnType::Float64,
            ColumnData::String { .. } => ColumnType::String,
            ColumnData::Binary { .. } => ColumnType::Binary,
            ColumnData::StringDict { .. } => ColumnType::StringDict,
            ColumnData::FixedList { .. } => ColumnType::FixedList,
            ColumnData::Float16List { .. } => ColumnType::Float16List,
        }
    }

    #[inline]
    pub fn push_i64(&mut self, value: i64) {
        if let ColumnData::Int64(v) = self {
            v.push(value);
        }
    }

    #[inline]
    pub fn push_f64(&mut self, value: f64) {
        if let ColumnData::Float64(v) = self {
            v.push(value);
        }
    }

    #[inline]
    pub fn push_bool(&mut self, value: bool) {
        if let ColumnData::Bool { data, len } = self {
            let byte_idx = *len / 8;
            let bit_idx = *len % 8;
            if byte_idx >= data.len() {
                data.push(0);
            }
            if value {
                data[byte_idx] |= 1 << bit_idx;
            }
            *len += 1;
        }
    }

    #[inline]
    pub fn push_string(&mut self, value: &str) {
        if let ColumnData::String { offsets, data } = self {
            data.extend_from_slice(value.as_bytes());
            offsets.push(data.len() as u32);
        }
    }

    #[inline]
    pub fn push_bytes(&mut self, value: &[u8]) {
        if let ColumnData::Binary { offsets, data } = self {
            data.extend_from_slice(value);
            offsets.push(data.len() as u32);
        }
    }

    pub fn extend_i64(&mut self, values: &[i64]) {
        if let ColumnData::Int64(v) = self {
            v.extend_from_slice(values);
        }
    }

    pub fn extend_f64(&mut self, values: &[f64]) {
        if let ColumnData::Float64(v) = self {
            v.extend_from_slice(values);
        }
    }

    /// Batch extend strings - much faster than individual push_string calls
    #[inline]
    pub fn extend_strings(&mut self, values: &[String]) {
        if let ColumnData::String { offsets, data } = self {
            // Pre-calculate total size needed
            let total_len: usize = values.iter().map(|s| s.len()).sum();
            data.reserve(total_len);
            offsets.reserve(values.len());
            
            for s in values {
                data.extend_from_slice(s.as_bytes());
                offsets.push(data.len() as u32);
            }
        }
    }

    /// Batch extend binary data
    #[inline]
    pub fn extend_bytes(&mut self, values: &[Vec<u8>]) {
        if let ColumnData::Binary { offsets, data } = self {
            let total_len: usize = values.iter().map(|b| b.len()).sum();
            data.reserve(total_len);
            offsets.reserve(values.len());
            
            for b in values {
                data.extend_from_slice(b);
                offsets.push(data.len() as u32);
            }
        }
    }

    /// Batch extend bools
    /// OPTIMIZATION: pre-allocate all needed bytes upfront, skip branch on data.len()
    #[inline]
    pub fn extend_bools(&mut self, values: &[bool]) {
        if let ColumnData::Bool { data, len } = self {
            if values.is_empty() { return; }
            let new_len = *len + values.len();
            let needed_bytes = (new_len + 7) / 8;
            data.resize(needed_bytes, 0);
            for &value in values {
                if value {
                    let byte_idx = *len / 8;
                    let bit_idx = *len % 8;
                    data[byte_idx] |= 1 << bit_idx;
                }
                *len += 1;
            }
        }
    }

    /// Serialize to bytes
    /// OPTIMIZED: Uses bulk memcpy for numeric columns instead of per-element loops
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            ColumnData::Bool { data, len } => {
                // Write exactly (len + 7) / 8 bytes — data.len() may exceed this
                // from push_bool's incremental Vec growth
                let byte_len = (*len + 7) / 8;
                let mut buf = Vec::with_capacity(8 + byte_len);
                buf.extend_from_slice(&(*len as u64).to_le_bytes());
                buf.extend_from_slice(&data[..byte_len.min(data.len())]);
                // Pad if data is shorter than expected (shouldn't happen normally)
                if data.len() < byte_len {
                    buf.resize(8 + byte_len, 0);
                }
                buf
            }
            ColumnData::Int64(v) => {
                // OPTIMIZATION: Bulk memcpy instead of per-element loop
                // ~10x faster for large arrays
                let mut buf = Vec::with_capacity(8 + v.len() * 8);
                buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                // SAFETY: i64 slice can be safely viewed as bytes on all platforms
                let bytes = unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8)
                };
                buf.extend_from_slice(bytes);
                buf
            }
            ColumnData::Float64(v) => {
                // OPTIMIZATION: Bulk memcpy instead of per-element loop
                let mut buf = Vec::with_capacity(8 + v.len() * 8);
                buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                // SAFETY: f64 slice can be safely viewed as bytes on all platforms
                let bytes = unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8)
                };
                buf.extend_from_slice(bytes);
                buf
            }
            ColumnData::String { offsets, data } | ColumnData::Binary { offsets, data } => {
                // OPTIMIZATION: Pre-allocate and use bulk memcpy for offsets
                let count = offsets.len().saturating_sub(1);
                let mut buf = Vec::with_capacity(8 + offsets.len() * 4 + 8 + data.len());
                buf.extend_from_slice(&(count as u64).to_le_bytes());
                // Bulk copy offsets (u32 array)
                let offset_bytes = unsafe {
                    std::slice::from_raw_parts(offsets.as_ptr() as *const u8, offsets.len() * 4)
                };
                buf.extend_from_slice(offset_bytes);
                buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
                buf.extend_from_slice(data);
                buf
            }
            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                // Format: [row_count][dict_size][indices...][dict_offsets...][dict_data_len][dict_data]
                // OPTIMIZATION: Pre-allocate and use bulk memcpy
                let mut buf = Vec::with_capacity(
                    16 + indices.len() * 4 + dict_offsets.len() * 4 + 8 + dict_data.len()
                );
                buf.extend_from_slice(&(indices.len() as u64).to_le_bytes());
                buf.extend_from_slice(&(dict_offsets.len() as u64).to_le_bytes());
                // Bulk copy indices (u32 array)
                let indices_bytes = unsafe {
                    std::slice::from_raw_parts(indices.as_ptr() as *const u8, indices.len() * 4)
                };
                buf.extend_from_slice(indices_bytes);
                // Bulk copy dict_offsets (u32 array)
                let offsets_bytes = unsafe {
                    std::slice::from_raw_parts(dict_offsets.as_ptr() as *const u8, dict_offsets.len() * 4)
                };
                buf.extend_from_slice(offsets_bytes);
                buf.extend_from_slice(&(dict_data.len() as u64).to_le_bytes());
                buf.extend_from_slice(dict_data);
                buf
            }
            ColumnData::FixedList { data, dim } => {
                let count = if *dim == 0 { 0usize } else { data.len() / (*dim as usize * 4) };
                let mut buf = Vec::with_capacity(12 + data.len());
                buf.extend_from_slice(&(count as u64).to_le_bytes());
                buf.extend_from_slice(&dim.to_le_bytes());
                buf.extend_from_slice(data);
                buf
            }
            ColumnData::Float16List { data, dim } => {
                let count = if *dim == 0 { 0usize } else { data.len() / (*dim as usize * 2) };
                let mut buf = Vec::with_capacity(12 + data.len());
                buf.extend_from_slice(&(count as u64).to_le_bytes());
                buf.extend_from_slice(&dim.to_le_bytes());
                buf.extend_from_slice(data);
                buf
            }
        }
    }

    /// Push a single f32 vector (as raw bytes) into a FixedList column.
    #[inline]
    pub fn push_fixed_list(&mut self, bytes: &[u8]) {
        if let ColumnData::FixedList { data, dim } = self {
            let row_bytes = bytes.len();
            if *dim == 0 && row_bytes % 4 == 0 {
                *dim = (row_bytes / 4) as u32;
            }
            data.extend_from_slice(bytes);
        }
    }

    /// Push a single f16 vector (as raw LE u16 bytes) into a Float16List column.
    #[inline]
    pub fn push_float16_list(&mut self, bytes: &[u8]) {
        if let ColumnData::Float16List { data, dim } = self {
            let row_bytes = bytes.len();
            if *dim == 0 && row_bytes % 2 == 0 {
                *dim = (row_bytes / 2) as u32;
            }
            data.extend_from_slice(bytes);
        }
    }

    /// Push f32 bytes (LE) into a Float16List column (converts f32→f16 on the fly).
    #[inline]
    pub fn push_float16_list_from_f32(&mut self, f32_bytes: &[u8]) {
        if let ColumnData::Float16List { data, dim } = self {
            let n = f32_bytes.len() / 4;
            if *dim == 0 && n > 0 { *dim = n as u32; }
            for chunk in f32_bytes.chunks_exact(4) {
                let f = f32::from_le_bytes(chunk.try_into().unwrap());
                data.extend_from_slice(&f32_to_f16(f).to_le_bytes());
            }
        }
    }

    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        match self {
            ColumnData::Bool { data, len } => {
                let byte_len = (*len + 7) / 8;
                writer.write_all(&(*len as u64).to_le_bytes())?;
                writer.write_all(&data[..byte_len.min(data.len())])?;
                // Pad if data is shorter than expected
                if data.len() < byte_len {
                    let pad = byte_len - data.len();
                    writer.write_all(&vec![0u8; pad])?;
                }
            }
            ColumnData::Int64(v) => {
                writer.write_all(&(v.len() as u64).to_le_bytes())?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8)
                };
                writer.write_all(bytes)?;
            }
            ColumnData::Float64(v) => {
                writer.write_all(&(v.len() as u64).to_le_bytes())?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8)
                };
                writer.write_all(bytes)?;
            }
            ColumnData::String { offsets, data } | ColumnData::Binary { offsets, data } => {
                let count = offsets.len().saturating_sub(1);
                writer.write_all(&(count as u64).to_le_bytes())?;
                let offset_bytes = unsafe {
                    std::slice::from_raw_parts(offsets.as_ptr() as *const u8, offsets.len() * 4)
                };
                writer.write_all(offset_bytes)?;
                writer.write_all(&(data.len() as u64).to_le_bytes())?;
                writer.write_all(data)?;
            }
            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                writer.write_all(&(indices.len() as u64).to_le_bytes())?;
                writer.write_all(&(dict_offsets.len() as u64).to_le_bytes())?;
                let indices_bytes = unsafe {
                    std::slice::from_raw_parts(indices.as_ptr() as *const u8, indices.len() * 4)
                };
                writer.write_all(indices_bytes)?;
                let offsets_bytes = unsafe {
                    std::slice::from_raw_parts(dict_offsets.as_ptr() as *const u8, dict_offsets.len() * 4)
                };
                writer.write_all(offsets_bytes)?;
                writer.write_all(&(dict_data.len() as u64).to_le_bytes())?;
                writer.write_all(dict_data)?;
            }
            ColumnData::FixedList { data, dim } => {
                let count = if *dim == 0 { 0 } else { data.len() / (*dim as usize * 4) };
                writer.write_all(&(count as u64).to_le_bytes())?;
                writer.write_all(&dim.to_le_bytes())?;
                writer.write_all(data)?;
            }
            ColumnData::Float16List { data, dim } => {
                let count = if *dim == 0 { 0 } else { data.len() / (*dim as usize * 2) };
                writer.write_all(&(count as u64).to_le_bytes())?;
                writer.write_all(&dim.to_le_bytes())?;
                writer.write_all(data)?;
            }
        }
        Ok(())
    }

    /// Deserialize from to_bytes() output, given the known column type.
    /// Returns (ColumnData, bytes_consumed).
    pub fn from_bytes_typed(bytes: &[u8], col_type: ColumnType) -> io::Result<(Self, usize)> {
        let mut pos = 0;
        
        macro_rules! read_u64 {
            () => {{
                if pos + 8 > bytes.len() {
                    return Err(err_data("ColumnData::from_bytes_typed: unexpected EOF reading u64"));
                }
                let v = u64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap());
                pos += 8;
                v
            }};
        }
        
        match col_type {
            ColumnType::Bool => {
                let len = read_u64!() as usize;
                let byte_len = (len + 7) / 8;
                if pos + byte_len > bytes.len() {
                    return Err(err_data("Bool column data truncated"));
                }
                let data = bytes[pos..pos + byte_len].to_vec();
                pos += byte_len;
                Ok((ColumnData::Bool { data, len }, pos))
            }
            ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 |
            ColumnType::Timestamp | ColumnType::Date => {
                let count = read_u64!() as usize;
                let byte_len = count * 8;
                if pos + byte_len > bytes.len() {
                    return Err(err_data("Int64 column data truncated"));
                }
                let mut v = vec![0i64; count];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes[pos..].as_ptr(), v.as_mut_ptr() as *mut u8, byte_len,
                    );
                }
                pos += byte_len;
                Ok((ColumnData::Int64(v), pos))
            }
            ColumnType::Float64 | ColumnType::Float32 => {
                let count = read_u64!() as usize;
                let byte_len = count * 8;
                if pos + byte_len > bytes.len() {
                    return Err(err_data("Float64 column data truncated"));
                }
                let mut v = vec![0f64; count];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes[pos..].as_ptr(), v.as_mut_ptr() as *mut u8, byte_len,
                    );
                }
                pos += byte_len;
                Ok((ColumnData::Float64(v), pos))
            }
            ColumnType::String => {
                let count = read_u64!() as usize;
                let offsets_len = (count + 1) * 4;
                if pos + offsets_len > bytes.len() {
                    return Err(err_data("String offsets truncated"));
                }
                let mut offsets = vec![0u32; count + 1];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes[pos..].as_ptr(), offsets.as_mut_ptr() as *mut u8, offsets_len,
                    );
                }
                pos += offsets_len;
                let data_len = read_u64!() as usize;
                if pos + data_len > bytes.len() {
                    return Err(err_data("String data truncated"));
                }
                let data = bytes[pos..pos + data_len].to_vec();
                pos += data_len;
                Ok((ColumnData::String { offsets, data }, pos))
            }
            ColumnType::Binary => {
                let count = read_u64!() as usize;
                let offsets_len = (count + 1) * 4;
                if pos + offsets_len > bytes.len() {
                    return Err(err_data("Binary offsets truncated"));
                }
                let mut offsets = vec![0u32; count + 1];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes[pos..].as_ptr(), offsets.as_mut_ptr() as *mut u8, offsets_len,
                    );
                }
                pos += offsets_len;
                let data_len = read_u64!() as usize;
                if pos + data_len > bytes.len() {
                    return Err(err_data("Binary data truncated"));
                }
                let data = bytes[pos..pos + data_len].to_vec();
                pos += data_len;
                Ok((ColumnData::Binary { offsets, data }, pos))
            }
            ColumnType::StringDict => {
                let row_count = read_u64!() as usize;
                let dict_size = read_u64!() as usize;
                let indices_len = row_count * 4;
                if pos + indices_len > bytes.len() {
                    return Err(err_data("StringDict indices truncated"));
                }
                let mut indices = vec![0u32; row_count];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes[pos..].as_ptr(), indices.as_mut_ptr() as *mut u8, indices_len,
                    );
                }
                pos += indices_len;
                let dict_offsets_len = dict_size * 4;
                if pos + dict_offsets_len > bytes.len() {
                    return Err(err_data("StringDict dict_offsets truncated"));
                }
                let mut dict_offsets = vec![0u32; dict_size];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes[pos..].as_ptr(), dict_offsets.as_mut_ptr() as *mut u8, dict_offsets_len,
                    );
                }
                pos += dict_offsets_len;
                let dict_data_len = read_u64!() as usize;
                if pos + dict_data_len > bytes.len() {
                    return Err(err_data("StringDict dict_data truncated"));
                }
                let dict_data = bytes[pos..pos + dict_data_len].to_vec();
                pos += dict_data_len;
                Ok((ColumnData::StringDict { indices, dict_offsets, dict_data }, pos))
            }
            ColumnType::FixedList => {
                let count = read_u64!() as usize;
                if pos + 4 > bytes.len() {
                    return Err(err_data("FixedList: dim truncated"));
                }
                let dim = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap());
                pos += 4;
                let byte_len = count * dim as usize * 4;
                if pos + byte_len > bytes.len() {
                    return Err(err_data("FixedList: data truncated"));
                }
                let data = bytes[pos..pos + byte_len].to_vec();
                pos += byte_len;
                Ok((ColumnData::FixedList { data, dim }, pos))
            }
            ColumnType::Float16List => {
                let count = read_u64!() as usize;
                if pos + 4 > bytes.len() {
                    return Err(err_data("Float16List: dim truncated"));
                }
                let dim = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap());
                pos += 4;
                let byte_len = count * dim as usize * 2;
                if pos + byte_len > bytes.len() {
                    return Err(err_data("Float16List: data truncated"));
                }
                let data = bytes[pos..pos + byte_len].to_vec();
                pos += byte_len;
                Ok((ColumnData::Float16List { data, dim }, pos))
            }
            ColumnType::Null => {
                let count = read_u64!() as usize;
                let byte_len = count * 8;
                pos += byte_len.min(bytes.len() - pos);
                Ok((ColumnData::Int64(vec![0i64; count]), pos))
            }
        }
    }
    
    /// Skip over serialized column data without allocating memory.
    /// Returns the number of bytes consumed (same as from_bytes_typed would report).
    /// Used by mmap-based on-demand reading to skip unrequested columns.
    pub fn skip_bytes_typed(bytes: &[u8], col_type: ColumnType) -> io::Result<usize> {
        let mut pos = 0;

        macro_rules! read_u64 {
            () => {{
                if pos + 8 > bytes.len() {
                    return Err(err_data("skip_bytes_typed: unexpected EOF reading u64"));
                }
                let v = u64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap());
                pos += 8;
                v
            }};
        }

        match col_type {
            ColumnType::Bool => {
                let len = read_u64!() as usize;
                let byte_len = (len + 7) / 8;
                pos += byte_len;
            }
            ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 |
            ColumnType::Timestamp | ColumnType::Date => {
                let count = read_u64!() as usize;
                pos += count * 8;
            }
            ColumnType::Float64 | ColumnType::Float32 => {
                let count = read_u64!() as usize;
                pos += count * 8;
            }
            ColumnType::String | ColumnType::Binary => {
                let count = read_u64!() as usize;
                let offsets_len = (count + 1) * 4;
                pos += offsets_len;
                let data_len = read_u64!() as usize;
                pos += data_len;
            }
            ColumnType::StringDict => {
                let row_count = read_u64!() as usize;
                let dict_size = read_u64!() as usize;
                pos += row_count * 4; // indices
                pos += dict_size * 4; // dict_offsets
                let dict_data_len = read_u64!() as usize;
                pos += dict_data_len;
            }
            ColumnType::FixedList => {
                let count = read_u64!() as usize;
                if pos + 4 > bytes.len() {
                    return Err(err_data("skip FixedList: dim truncated"));
                }
                let dim = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap());
                pos += 4;
                pos += count * dim as usize * 4;
            }
            ColumnType::Float16List => {
                let count = read_u64!() as usize;
                if pos + 4 > bytes.len() {
                    return Err(err_data("skip Float16List: dim truncated"));
                }
                let dim = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap());
                pos += 4;
                pos += count * dim as usize * 2;
            }
            ColumnType::Null => {
                let count = read_u64!() as usize;
                pos += count * 8;
            }
        }

        if pos > bytes.len() {
            return Err(err_data(format!("skip_bytes_typed: would skip past EOF ({} > {})", pos, bytes.len())));
        }
        Ok(pos)
    }

    /// Create an empty column with the same type
    pub fn clone_empty(&self) -> Self {
        match self {
            ColumnData::Bool { .. } => ColumnData::Bool { data: Vec::new(), len: 0 },
            ColumnData::Int64(_) => ColumnData::Int64(Vec::new()),
            ColumnData::Float64(_) => ColumnData::Float64(Vec::new()),
            ColumnData::String { .. } => ColumnData::String { offsets: vec![0], data: Vec::new() },
            ColumnData::Binary { .. } => ColumnData::Binary { offsets: vec![0], data: Vec::new() },
            ColumnData::StringDict { .. } => ColumnData::StringDict { 
                indices: Vec::new(), 
                dict_offsets: vec![0], 
                dict_data: Vec::new() 
            },
            ColumnData::FixedList { dim, .. } => ColumnData::FixedList { data: Vec::new(), dim: *dim },
            ColumnData::Float16List { dim, .. } => ColumnData::Float16List { data: Vec::new(), dim: *dim },
        }
    }
    
    /// Append another column's data to this column.
    /// OPTIMIZATION: bulk copy for byte-aligned Bool, pre-allocated offsets for String/Binary.
    pub fn append(&mut self, other: &Self) {
        match (self, other) {
            (ColumnData::Bool { data, len }, ColumnData::Bool { data: other_data, len: other_len }) => {
                if *other_len == 0 { return; }
                // OPTIMIZATION: byte-aligned → bulk copy
                if *len % 8 == 0 {
                    let other_byte_len = (*other_len + 7) / 8;
                    let copy_len = other_byte_len.min(other_data.len());
                    data.extend_from_slice(&other_data[..copy_len]);
                    if copy_len < other_byte_len {
                        data.resize(data.len() + (other_byte_len - copy_len), 0);
                    }
                    *len += *other_len;
                } else {
                    for i in 0..*other_len {
                        let byte_idx = i / 8;
                        let bit_idx = i % 8;
                        let val = byte_idx < other_data.len() && (other_data[byte_idx] >> bit_idx) & 1 == 1;
                        let new_byte = *len / 8;
                        let new_bit = *len % 8;
                        if new_byte >= data.len() {
                            data.push(0);
                        }
                        if val {
                            data[new_byte] |= 1 << new_bit;
                        }
                        *len += 1;
                    }
                }
            }
            (ColumnData::Int64(v), ColumnData::Int64(other_v)) => {
                v.extend_from_slice(other_v);
            }
            (ColumnData::Float64(v), ColumnData::Float64(other_v)) => {
                v.extend_from_slice(other_v);
            }
            (ColumnData::String { offsets, data }, ColumnData::String { offsets: other_offsets, data: other_data }) |
            (ColumnData::Binary { offsets, data }, ColumnData::Binary { offsets: other_offsets, data: other_data }) => {
                let base_offset = *offsets.last().unwrap_or(&0);
                // OPTIMIZATION: pre-allocate and batch push
                offsets.reserve(other_offsets.len() - 1);
                for i in 1..other_offsets.len() {
                    offsets.push(base_offset + other_offsets[i]);
                }
                data.extend_from_slice(other_data);
            }
            (ColumnData::StringDict { indices, dict_offsets, dict_data },
             ColumnData::StringDict { indices: other_indices, dict_offsets: other_offsets, dict_data: other_data }) => {
                let existing_dict_size = dict_offsets.len();
                let base = *dict_offsets.last().unwrap_or(&0);
                dict_offsets.reserve(other_offsets.len() - 1);
                for i in 1..other_offsets.len() {
                    dict_offsets.push(base + other_offsets[i]);
                }
                dict_data.extend_from_slice(other_data);
                let offset = if existing_dict_size > 0 { existing_dict_size as u32 - 1 } else { 0 };
                indices.reserve(other_indices.len());
                for &idx in other_indices {
                    indices.push(idx + offset);
                }
            }
            (ColumnData::FixedList { data, dim }, ColumnData::FixedList { data: other_data, dim: other_dim }) => {
                if *dim == 0 && *other_dim > 0 { *dim = *other_dim; }
                data.extend_from_slice(other_data);
            }
            (ColumnData::Float16List { data, dim }, ColumnData::Float16List { data: other_data, dim: other_dim }) => {
                if *dim == 0 && *other_dim > 0 { *dim = *other_dim; }
                data.extend_from_slice(other_data);
            }
            _ => {} // Type mismatch - ignore
        }
    }

    /// Apply a null bitmap: for rows marked as null (bit set), replace string data
    /// with empty strings (clearing the NULL marker sentinel). For numeric types,
    /// null rows get zeroed. This cleans up the `\x00__NULL__\x00` sentinel used
    /// by insert_typed_with_nulls so downstream consumers see proper empty/zero values.
    pub fn apply_null_bitmap(&mut self, null_bitmap: &[u8]) {
        match self {
            ColumnData::String { offsets, data } => {
                if offsets.len() < 2 { return; }
                let row_count = offsets.len() - 1;
                let mut new_offsets = Vec::with_capacity(row_count + 1);
                let mut new_data = Vec::new();
                new_offsets.push(0u32);
                for i in 0..row_count {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    let is_null = byte_idx < null_bitmap.len()
                        && (null_bitmap[byte_idx] >> bit_idx) & 1 != 0;
                    if is_null {
                        new_offsets.push(new_data.len() as u32);
                    } else {
                        let s = offsets[i] as usize;
                        let e = offsets[i + 1] as usize;
                        new_data.extend_from_slice(&data[s..e]);
                        new_offsets.push(new_data.len() as u32);
                    }
                }
                *offsets = new_offsets;
                *data = new_data;
            }
            ColumnData::Int64(v) => {
                for i in 0..v.len() {
                    let b = i / 8; let bit = i % 8;
                    if b < null_bitmap.len() && (null_bitmap[b] >> bit) & 1 != 0 {
                        v[i] = 0;
                    }
                }
            }
            ColumnData::Float64(v) => {
                for i in 0..v.len() {
                    let b = i / 8; let bit = i % 8;
                    if b < null_bitmap.len() && (null_bitmap[b] >> bit) & 1 != 0 {
                        v[i] = 0.0;
                    }
                }
            }
            ColumnData::Bool { data: bits, len } => {
                for i in 0..*len {
                    let b = i / 8; let bit = i % 8;
                    if b < null_bitmap.len() && (null_bitmap[b] >> bit) & 1 != 0 {
                        // Clear the bit for null rows
                        if i / 8 < bits.len() {
                            bits[i / 8] &= !(1 << (i % 8));
                        }
                    }
                }
            }
            ColumnData::Binary { offsets, data } => {
                if offsets.len() < 2 { return; }
                let row_count = offsets.len() - 1;
                let mut new_offsets = Vec::with_capacity(row_count + 1);
                let mut new_data = Vec::new();
                new_offsets.push(0u32);
                for i in 0..row_count {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    let is_null = byte_idx < null_bitmap.len()
                        && (null_bitmap[byte_idx] >> bit_idx) & 1 != 0;
                    if is_null {
                        new_offsets.push(new_data.len() as u32);
                    } else {
                        let s = offsets[i] as usize;
                        let e = offsets[i + 1] as usize;
                        new_data.extend_from_slice(&data[s..e]);
                        new_offsets.push(new_data.len() as u32);
                    }
                }
                *offsets = new_offsets;
                *data = new_data;
            }
            _ => {}
        }
    }

    /// Filter column data to only include rows at specified indices.
    /// OPTIMIZATION: uses pre-allocation and unchecked indexing for hot paths.
    pub fn filter_by_indices(&self, indices: &[usize]) -> Self {
        match self {
            ColumnData::Bool { data, len: _ } => {
                let new_len = indices.len();
                let mut new_data = vec![0u8; (new_len + 7) / 8];
                for (new_idx, &idx) in indices.iter().enumerate() {
                    let old_byte = idx / 8;
                    let old_bit = idx % 8;
                    if old_byte < data.len() && (data[old_byte] >> old_bit) & 1 == 1 {
                        new_data[new_idx / 8] |= 1 << (new_idx % 8);
                    }
                }
                ColumnData::Bool { data: new_data, len: new_len }
            }
            ColumnData::FixedList { data, dim } => {
                let stride = *dim as usize * 4;
                let mut new_data = Vec::with_capacity(indices.len() * stride);
                for &i in indices {
                    if stride > 0 && i * stride + stride <= data.len() {
                        new_data.extend_from_slice(&data[i * stride .. i * stride + stride]);
                    } else {
                        new_data.extend(std::iter::repeat(0u8).take(stride));
                    }
                }
                ColumnData::FixedList { data: new_data, dim: *dim }
            }
            ColumnData::Float16List { data, dim } => {
                let stride = *dim as usize * 2;
                let mut new_data = Vec::with_capacity(indices.len() * stride);
                for &i in indices {
                    if stride > 0 && i * stride + stride <= data.len() {
                        new_data.extend_from_slice(&data[i * stride .. i * stride + stride]);
                    } else {
                        new_data.extend(std::iter::repeat(0u8).take(stride));
                    }
                }
                ColumnData::Float16List { data: new_data, dim: *dim }
            }
            ColumnData::Int64(v) => {
                // OPTIMIZATION: pre-allocate exact size, use unchecked indexing
                let mut result = Vec::with_capacity(indices.len());
                for &i in indices {
                    // Safety: caller guarantees indices are in-range (built from 0..ids.len())
                    result.push(if i < v.len() { unsafe { *v.get_unchecked(i) } } else { 0 });
                }
                ColumnData::Int64(result)
            }
            ColumnData::Float64(v) => {
                let mut result = Vec::with_capacity(indices.len());
                for &i in indices {
                    result.push(if i < v.len() { unsafe { *v.get_unchecked(i) } } else { 0.0 });
                }
                ColumnData::Float64(result)
            }
            ColumnData::String { offsets, data } | ColumnData::Binary { offsets, data } => {
                let mut new_offsets = Vec::with_capacity(indices.len() + 1);
                new_offsets.push(0u32);
                // Estimate average string length for pre-allocation
                let avg_len = if offsets.len() > 1 {
                    data.len() / (offsets.len() - 1)
                } else { 0 };
                let mut new_data = Vec::with_capacity(indices.len() * avg_len);
                for &idx in indices {
                    if idx + 1 < offsets.len() {
                        let start = offsets[idx] as usize;
                        let end = offsets[idx + 1] as usize;
                        new_data.extend_from_slice(&data[start..end]);
                        new_offsets.push(new_data.len() as u32);
                    }
                }
                if matches!(self, ColumnData::String { .. }) {
                    ColumnData::String { offsets: new_offsets, data: new_data }
                } else {
                    ColumnData::Binary { offsets: new_offsets, data: new_data }
                }
            }
            ColumnData::StringDict { indices: row_indices, dict_offsets, dict_data } => {
                // Just filter the indices array, dictionary stays the same
                let mut new_indices = Vec::with_capacity(indices.len());
                for &i in indices {
                    if i < row_indices.len() {
                        new_indices.push(unsafe { *row_indices.get_unchecked(i) });
                    }
                }
                ColumnData::StringDict { 
                    indices: new_indices, 
                    dict_offsets: dict_offsets.clone(), 
                    dict_data: dict_data.clone() 
                }
            }
        }
    }
    
    /// Check if dictionary encoding would be beneficial for this column
    /// Returns true if cardinality is low relative to row count
    pub fn should_dict_encode(&self) -> bool {
        if let ColumnData::String { offsets, data } = self {
            use ahash::AHashSet;
            
            let row_count = offsets.len().saturating_sub(1);
            if row_count < 100 {
                return false; // Too few rows to benefit
            }
            
            // Sample up to 1000 rows to estimate cardinality
            let sample_size = row_count.min(1000);
            let mut unique_strings: AHashSet<&[u8]> = AHashSet::with_capacity(sample_size / 10);
            
            let step = if row_count > sample_size { row_count / sample_size } else { 1 };
            let mut i = 0;
            while i < row_count && unique_strings.len() < sample_size / 5 {
                let start = offsets[i] as usize;
                let end = offsets[i + 1] as usize;
                if end <= data.len() {
                    unique_strings.insert(&data[start..end]);
                }
                i += step;
            }
            
            // Dictionary encoding is beneficial if cardinality < 20% of sampled rows
            // or if there are fewer than 10000 unique values
            let estimated_cardinality = unique_strings.len();
            estimated_cardinality < sample_size / 5 || estimated_cardinality < 10000
        } else {
            false
        }
    }
    
    /// Convert regular String column to dictionary-encoded StringDict
    /// This is beneficial for low-cardinality columns (e.g., category, status)
    pub fn to_dict_encoded(&self) -> Option<Self> {
        if let ColumnData::String { offsets, data } = self {
            use ahash::AHashMap;
            
            let row_count = offsets.len().saturating_sub(1);
            if row_count == 0 {
                return Some(ColumnData::StringDict {
                    indices: Vec::new(),
                    dict_offsets: vec![0],
                    dict_data: Vec::new(),
                });
            }
            
            // Build dictionary: string -> dict_index
            let mut dict_map: AHashMap<&[u8], u32> = AHashMap::with_capacity(1000);
            let mut dict_offsets_new = vec![0u32];
            let mut dict_data_new = Vec::new();
            let mut row_indices = Vec::with_capacity(row_count);
            let mut next_dict_idx = 1u32; // 0 reserved for NULL
            
            for i in 0..row_count {
                let start = offsets[i] as usize;
                let end = offsets[i + 1] as usize;
                let str_bytes = &data[start..end];
                
                let dict_idx = *dict_map.entry(str_bytes).or_insert_with(|| {
                    let idx = next_dict_idx;
                    next_dict_idx += 1;
                    dict_data_new.extend_from_slice(str_bytes);
                    dict_offsets_new.push(dict_data_new.len() as u32);
                    idx
                });
                row_indices.push(dict_idx);
            }
            
            Some(ColumnData::StringDict {
                indices: row_indices,
                dict_offsets: dict_offsets_new,
                dict_data: dict_data_new,
            })
        } else {
            None
        }
    }
    
    /// Decode StringDict back to plain String column.
    /// Used during streaming compaction to normalize types before merging with delta data.
    pub fn decode_string_dict(self) -> Self {
        if let ColumnData::StringDict { indices, dict_offsets, dict_data } = self {
            let mut offsets = Vec::with_capacity(indices.len() + 1);
            let mut data = Vec::new();
            offsets.push(0u32);
            
            for &idx in &indices {
                if idx == 0 || (idx as usize) >= dict_offsets.len() {
                    offsets.push(data.len() as u32);
                } else {
                    let start = dict_offsets[(idx - 1) as usize] as usize;
                    let end = dict_offsets[idx as usize] as usize;
                    if end <= dict_data.len() && start <= end {
                        data.extend_from_slice(&dict_data[start..end]);
                    }
                    offsets.push(data.len() as u32);
                }
            }
            
            ColumnData::String { offsets, data }
        } else {
            self
        }
    }
    
    /// Try to convert to dictionary encoding if beneficial, otherwise return self
    pub fn maybe_dict_encode(self) -> Self {
        if self.should_dict_encode() {
            self.to_dict_encoded().unwrap_or(self)
        } else {
            self
        }
    }
    
    /// Get dictionary index for a row (for StringDict columns)
    #[inline]
    pub fn get_dict_index(&self, row: usize) -> Option<u32> {
        if let ColumnData::StringDict { indices, .. } = self {
            indices.get(row).copied()
        } else {
            None
        }
    }
    
    /// Extract a contiguous row range [start, end).
    /// More efficient than filter_by_indices for contiguous ranges (uses memcpy).
    pub fn slice_range(&self, start: usize, end: usize) -> Self {
        match self {
            ColumnData::FixedList { data, dim } => {
                let stride = *dim as usize * 4;
                let row_count = if stride == 0 { 0 } else { data.len() / stride };
                let s = start.min(row_count);
                let e = end.min(row_count);
                ColumnData::FixedList {
                    data: data[s * stride .. e * stride].to_vec(),
                    dim: *dim,
                }
            }
            ColumnData::Float16List { data, dim } => {
                let stride = *dim as usize * 2;
                let row_count = if stride == 0 { 0 } else { data.len() / stride };
                let s = start.min(row_count);
                let e = end.min(row_count);
                ColumnData::Float16List {
                    data: data[s * stride .. e * stride].to_vec(),
                    dim: *dim,
                }
            }
            ColumnData::Bool { data, len } => {
                let s = start.min(*len);
                let e = end.min(*len);
                let count = e.saturating_sub(s);
                let mut new_data = vec![0u8; (count + 7) / 8];
                for i in 0..count {
                    let ob = (s + i) / 8;
                    let obit = (s + i) % 8;
                    if ob < data.len() && (data[ob] >> obit) & 1 == 1 {
                        new_data[i / 8] |= 1 << (i % 8);
                    }
                }
                ColumnData::Bool { data: new_data, len: count }
            }
            ColumnData::Int64(v) => {
                ColumnData::Int64(v[start.min(v.len())..end.min(v.len())].to_vec())
            }
            ColumnData::Float64(v) => {
                ColumnData::Float64(v[start.min(v.len())..end.min(v.len())].to_vec())
            }
            ColumnData::String { offsets, data } => {
                let row_count = offsets.len().saturating_sub(1);
                let s = start.min(row_count);
                let e = end.min(row_count);
                if e <= s {
                    return ColumnData::String { offsets: vec![0], data: Vec::new() };
                }
                let data_start = offsets[s] as usize;
                let data_end = offsets[e] as usize;
                let new_data = data[data_start..data_end.min(data.len())].to_vec();
                let base = offsets[s];
                let new_offsets: Vec<u32> = offsets[s..=e].iter().map(|&o| o - base).collect();
                ColumnData::String { offsets: new_offsets, data: new_data }
            }
            ColumnData::Binary { offsets, data } => {
                let row_count = offsets.len().saturating_sub(1);
                let s = start.min(row_count);
                let e = end.min(row_count);
                if e <= s {
                    return ColumnData::Binary { offsets: vec![0], data: Vec::new() };
                }
                let data_start = offsets[s] as usize;
                let data_end = offsets[e] as usize;
                let new_data = data[data_start..data_end.min(data.len())].to_vec();
                let base = offsets[s];
                let new_offsets: Vec<u32> = offsets[s..=e].iter().map(|&o| o - base).collect();
                ColumnData::Binary { offsets: new_offsets, data: new_data }
            }
            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                let s = start.min(indices.len());
                let e = end.min(indices.len());
                ColumnData::StringDict {
                    indices: indices[s..e].to_vec(),
                    dict_offsets: dict_offsets.clone(),
                    dict_data: dict_data.clone(),
                }
            }
        }
    }
    
    /// Estimate memory usage in bytes
    pub fn estimate_memory_bytes(&self) -> usize {
        match self {
            ColumnData::Bool { data, .. } => data.len(),
            ColumnData::Int64(v) => v.len() * 8,
            ColumnData::Float64(v) => v.len() * 8,
            ColumnData::String { offsets, data } => offsets.len() * 4 + data.len(),
            ColumnData::Binary { offsets, data } => offsets.len() * 4 + data.len(),
            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                indices.len() * 4 + dict_offsets.len() * 4 + dict_data.len()
            }
            ColumnData::FixedList { data, .. } => data.len(),
            ColumnData::Float16List { data, .. } => data.len(),
        }
    }
}
