// Header types: OnDemandHeader, RowGroupMeta, RgColumnZoneMap, V4Footer, ColumnIndexEntry, OnDemandSchema

// ============================================================================
// File Header (256 bytes)
// ============================================================================

#[derive(Debug, Clone)]
pub struct OnDemandHeader {
    pub version: u32,
    pub flags: u32,
    pub row_count: u64,
    pub column_count: u32,
    pub row_group_size: u32,
    pub schema_offset: u64,
    pub column_index_offset: u64,
    pub id_column_offset: u64,
    pub created_at: i64,
    pub modified_at: i64,
    pub checksum: u32,
    /// Byte offset to V4Footer
    pub footer_offset: u64,
    /// Number of Row Groups
    pub row_group_count: u32,
}

impl OnDemandHeader {
    pub fn new() -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            version: FORMAT_VERSION_V4,
            flags: 0,
            row_count: 0,
            column_count: 0,
            row_group_size: DEFAULT_ROW_GROUP_SIZE,
            schema_offset: HEADER_SIZE as u64,
            column_index_offset: 0,
            id_column_offset: 0,
            footer_offset: 0,
            row_group_count: 0,
            created_at: now,
            modified_at: now,
            checksum: 0,
        }
    }

    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        let mut pos = 0;

        // Magic (8 bytes)
        buf[pos..pos + 8].copy_from_slice(MAGIC);
        pos += 8;

        // Version (4 bytes)
        buf[pos..pos + 4].copy_from_slice(&self.version.to_le_bytes());
        pos += 4;

        // Flags (4 bytes)
        buf[pos..pos + 4].copy_from_slice(&self.flags.to_le_bytes());
        pos += 4;

        // Row count (8 bytes)
        buf[pos..pos + 8].copy_from_slice(&self.row_count.to_le_bytes());
        pos += 8;

        // Column count (4 bytes)
        buf[pos..pos + 4].copy_from_slice(&self.column_count.to_le_bytes());
        pos += 4;

        // Row group size (4 bytes)
        buf[pos..pos + 4].copy_from_slice(&self.row_group_size.to_le_bytes());
        pos += 4;

        // Schema offset (8 bytes)
        buf[pos..pos + 8].copy_from_slice(&self.schema_offset.to_le_bytes());
        pos += 8;

        // Column index offset (8 bytes)
        buf[pos..pos + 8].copy_from_slice(&self.column_index_offset.to_le_bytes());
        pos += 8;

        // ID column offset (8 bytes)
        buf[pos..pos + 8].copy_from_slice(&self.id_column_offset.to_le_bytes());
        pos += 8;

        // Created timestamp (8 bytes)
        buf[pos..pos + 8].copy_from_slice(&self.created_at.to_le_bytes());
        pos += 8;

        // Modified timestamp (8 bytes)
        buf[pos..pos + 8].copy_from_slice(&self.modified_at.to_le_bytes());
        pos += 8;

        // Checksum (4 bytes) - computed over previous bytes
        let checksum = crc32fast::hash(&buf[0..pos]);
        buf[pos..pos + 4].copy_from_slice(&checksum.to_le_bytes());
        pos += 4;

        // V4 fields (in reserved space, after checksum)
        buf[pos..pos + 8].copy_from_slice(&self.footer_offset.to_le_bytes());
        pos += 8;
        buf[pos..pos + 4].copy_from_slice(&self.row_group_count.to_le_bytes());

        buf
    }

    pub fn from_bytes(bytes: &[u8; HEADER_SIZE]) -> io::Result<Self> {
        let mut pos = 0;

        // Verify magic
        if &bytes[pos..pos + 8] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid file magic",
            ));
        }
        pos += 8;

        let version = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let flags = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let row_count = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let column_count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let row_group_size = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let schema_offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let column_index_offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let id_column_offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let created_at = i64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let modified_at = i64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let checksum = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());

        // Verify checksum (covers core header fields)
        let computed = crc32fast::hash(&bytes[0..pos]);
        if computed != checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Header checksum mismatch",
            ));
        }
        pos += 4;

        // V4 fields (from reserved space)
        let footer_offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let row_group_count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());

        Ok(Self {
            version,
            flags,
            row_count,
            column_count,
            row_group_size,
            schema_offset,
            column_index_offset,
            id_column_offset,
            created_at,
            modified_at,
            checksum,
            footer_offset,
            row_group_count,
        })
    }
}

// ============================================================================
// V4 Row Group Metadata (40 bytes per Row Group in footer)
// ============================================================================

/// Metadata for a single Row Group stored in the V4 footer.
/// Each Row Group is a self-contained chunk of rows with its own columns + nulls.
#[derive(Debug, Clone, Copy)]
pub struct RowGroupMeta {
    /// Byte offset from file start where this Row Group's data begins
    pub offset: u64,
    /// Total byte size of this Row Group's data (IDs + deletion + columns)
    pub data_size: u64,
    /// Number of rows in this Row Group
    pub row_count: u32,
    /// Minimum row ID in this Row Group (for predicate pushdown)
    pub min_id: u64,
    /// Maximum row ID in this Row Group
    pub max_id: u64,
    /// Number of deleted rows (soft-deleted via deletion vector)
    pub deletion_count: u32,
}

impl RowGroupMeta {
    pub fn to_bytes(&self) -> [u8; ROW_GROUP_META_SIZE] {
        let mut buf = [0u8; ROW_GROUP_META_SIZE];
        buf[0..8].copy_from_slice(&self.offset.to_le_bytes());
        buf[8..16].copy_from_slice(&self.data_size.to_le_bytes());
        buf[16..20].copy_from_slice(&self.row_count.to_le_bytes());
        buf[20..28].copy_from_slice(&self.min_id.to_le_bytes());
        buf[28..36].copy_from_slice(&self.max_id.to_le_bytes());
        buf[36..40].copy_from_slice(&self.deletion_count.to_le_bytes());
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            offset: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            data_size: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            row_count: u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            min_id: u64::from_le_bytes(bytes[20..28].try_into().unwrap()),
            max_id: u64::from_le_bytes(bytes[28..36].try_into().unwrap()),
            deletion_count: u32::from_le_bytes(bytes[36..40].try_into().unwrap()),
        }
    }
    
    /// Active (non-deleted) row count
    pub fn active_rows(&self) -> u32 {
        self.row_count.saturating_sub(self.deletion_count)
    }
}

/// V4 file footer: stored at end of file, contains schema + Row Group directory.
///
/// Layout:
/// ```text
/// [schema_bytes_len: u64][schema_bytes]
/// [rg_count: u32]
/// [RowGroupMeta × rg_count]
/// [footer_size: u64]       ← total footer bytes (for seeking from EOF)
/// [MAGIC_V4_FOOTER: 8 bytes]
/// ```
/// Per-column zone map (min/max) for a single Row Group.
/// Used to skip entire RGs during scans when the filter range doesn't overlap.
#[derive(Debug, Clone)]
pub struct RgColumnZoneMap {
    /// Column index in the schema
    pub col_idx: u16,
    /// Min value (i64 bits for Int64, f64 bits for Float64)
    pub min_bits: i64,
    /// Max value (i64 bits for Int64, f64 bits for Float64)
    pub max_bits: i64,
    /// True if the column contains any NULL values in this RG
    pub has_nulls: bool,
    /// True if this is a Float64 column (min_bits/max_bits are f64::to_bits)
    pub is_float: bool,
}

impl RgColumnZoneMap {
    /// Check if a filter value might match rows in this zone map.
    /// `op` is one of: "=", ">", ">=", "<", "<="
    pub fn may_contain_int(&self, op: &str, val: i64) -> bool {
        match op {
            "=" | "==" => val >= self.min_bits && val <= self.max_bits,
            ">" => self.max_bits > val,
            ">=" => self.max_bits >= val,
            "<" => self.min_bits < val,
            "<=" => self.min_bits <= val,
            _ => true,
        }
    }

    pub fn may_contain_float(&self, op: &str, val: f64) -> bool {
        let min_f = f64::from_bits(self.min_bits as u64);
        let max_f = f64::from_bits(self.max_bits as u64);
        match op {
            "=" | "==" => val >= min_f && val <= max_f,
            ">" => max_f > val,
            ">=" => max_f >= val,
            "<" => min_f < val,
            "<=" => min_f <= val,
            _ => true,
        }
    }

    /// Check if a BETWEEN range might overlap this zone map (Int64)
    pub fn may_overlap_int_range(&self, low: i64, high: i64) -> bool {
        self.max_bits >= low && self.min_bits <= high
    }

    /// Check if a BETWEEN range might overlap this zone map (Float64)
    pub fn may_overlap_float_range(&self, low: f64, high: f64) -> bool {
        let min_f = f64::from_bits(self.min_bits as u64);
        let max_f = f64::from_bits(self.max_bits as u64);
        max_f >= low && min_f <= high
    }
}

/// Zone maps for all RGs: outer Vec is per-RG, inner Vec is per-column.
/// Only numeric columns (Int64, Float64) have zone maps.
pub type RgZoneMaps = Vec<Vec<RgColumnZoneMap>>;

/// Row Group Column Offset Index (RCIX): per-RG, per-column byte offsets
/// of each column's null bitmap start within the uncompressed RG body.
/// Enables O(1) direct seeks for cold-start LIMIT reads without sequential scan.
pub type RgColumnOffsets = Vec<Vec<u32>>;

const ZONE_MAP_MAGIC: &[u8; 4] = b"ZMAP";
const RCIX_MAGIC: &[u8; 4] = b"RCIX";

#[derive(Debug, Clone)]
pub struct V4Footer {
    pub schema: OnDemandSchema,
    pub row_groups: Vec<RowGroupMeta>,
    /// Per-RG per-column zone maps (min/max for numeric columns)
    pub zone_maps: RgZoneMaps,
    /// Per-RG per-column byte offsets of null bitmaps within the uncompressed RG body.
    /// Empty when not present (backward compat with older files).
    pub col_offsets: RgColumnOffsets,
}

impl V4Footer {
    pub fn to_bytes(&self) -> Vec<u8> {
        let schema_bytes = self.schema.to_bytes();
        let rg_count = self.row_groups.len() as u32;
        
        let mut buf = Vec::with_capacity(1024);
        
        // Schema
        buf.extend_from_slice(&(schema_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(&schema_bytes);
        
        // Row Group directory
        buf.extend_from_slice(&rg_count.to_le_bytes());
        for rg in &self.row_groups {
            buf.extend_from_slice(&rg.to_bytes());
        }
        
        // Zone maps section (optional, backward-compat: old readers skip to footer_size)
        if !self.zone_maps.is_empty() {
            buf.extend_from_slice(ZONE_MAP_MAGIC);
            buf.extend_from_slice(&(self.zone_maps.len() as u32).to_le_bytes()); // rg count
            for rg_zmaps in &self.zone_maps {
                buf.extend_from_slice(&(rg_zmaps.len() as u16).to_le_bytes()); // cols in this RG
                for zm in rg_zmaps {
                    buf.extend_from_slice(&zm.col_idx.to_le_bytes());
                    buf.extend_from_slice(&zm.min_bits.to_le_bytes());
                    buf.extend_from_slice(&zm.max_bits.to_le_bytes());
                    let flags: u8 = (zm.has_nulls as u8) | ((zm.is_float as u8) << 1);
                    buf.push(flags);
                }
            }
        }
        
        // RCIX section: per-RG per-column body offsets for direct-seek reads
        if !self.col_offsets.is_empty() {
            buf.extend_from_slice(RCIX_MAGIC);
            buf.extend_from_slice(&(self.col_offsets.len() as u32).to_le_bytes());
            for rg_offsets in &self.col_offsets {
                buf.extend_from_slice(&(rg_offsets.len() as u16).to_le_bytes());
                for &off in rg_offsets {
                    buf.extend_from_slice(&off.to_le_bytes());
                }
            }
        }

        // Footer size (everything before this field + 8 bytes for size + 8 bytes for magic)
        let footer_size = buf.len() as u64 + 8 + 8;
        buf.extend_from_slice(&footer_size.to_le_bytes());
        
        // Magic
        buf.extend_from_slice(MAGIC_V4_FOOTER);
        
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < 20 {
            return Err(err_data("V4 footer too small"));
        }
        
        let mut pos = 0;
        
        // Schema
        let schema_len = u64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap()) as usize;
        pos += 8;
        if pos + schema_len > bytes.len() {
            return Err(err_data("V4 footer: schema overflow"));
        }
        let schema = OnDemandSchema::from_bytes(&bytes[pos..pos+schema_len])?;
        pos += schema_len;
        
        // Row Group directory
        let rg_count = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap()) as usize;
        pos += 4;
        
        let mut row_groups = Vec::with_capacity(rg_count);
        for _ in 0..rg_count {
            if pos + ROW_GROUP_META_SIZE > bytes.len() {
                return Err(err_data("V4 footer: RG meta overflow"));
            }
            row_groups.push(RowGroupMeta::from_bytes(&bytes[pos..pos+ROW_GROUP_META_SIZE]));
            pos += ROW_GROUP_META_SIZE;
        }
        
        // Try reading zone maps section (backward compat: may not exist)
        let mut zone_maps: RgZoneMaps = Vec::new();
        if pos + 4 <= bytes.len() && &bytes[pos..pos+4] == ZONE_MAP_MAGIC {
            pos += 4;
            let zm_rg_count = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            for _ in 0..zm_rg_count {
                if pos + 2 > bytes.len() { break; }
                let col_count = u16::from_le_bytes(bytes[pos..pos+2].try_into().unwrap()) as usize;
                pos += 2;
                let mut rg_zmaps = Vec::with_capacity(col_count);
                for _ in 0..col_count {
                    if pos + 19 > bytes.len() { break; } // 2+8+8+1 = 19 bytes per entry
                    let col_idx = u16::from_le_bytes(bytes[pos..pos+2].try_into().unwrap());
                    pos += 2;
                    let min_bits = i64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap());
                    pos += 8;
                    let max_bits = i64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap());
                    pos += 8;
                    let flags = bytes[pos];
                    pos += 1;
                    rg_zmaps.push(RgColumnZoneMap {
                        col_idx,
                        min_bits,
                        max_bits,
                        has_nulls: (flags & 1) != 0,
                        is_float: (flags & 2) != 0,
                    });
                }
                zone_maps.push(rg_zmaps);
            }
        }
        
        // Try reading RCIX section (backward compat: may not exist)
        let mut col_offsets: RgColumnOffsets = Vec::new();
        if pos + 4 <= bytes.len() && &bytes[pos..pos+4] == RCIX_MAGIC {
            pos += 4;
            if pos + 4 <= bytes.len() {
                let rcix_rg_count = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap()) as usize;
                pos += 4;
                for _ in 0..rcix_rg_count {
                    if pos + 2 > bytes.len() { break; }
                    let ncols = u16::from_le_bytes(bytes[pos..pos+2].try_into().unwrap()) as usize;
                    pos += 2;
                    let mut rg_offs = Vec::with_capacity(ncols);
                    for _ in 0..ncols {
                        if pos + 4 > bytes.len() { break; }
                        rg_offs.push(u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap()));
                        pos += 4;
                    }
                    col_offsets.push(rg_offs);
                }
            }
        }

        Ok(Self { schema, row_groups, zone_maps, col_offsets })
    }
    
    /// Total active rows across all Row Groups
    pub fn total_active_rows(&self) -> u64 {
        self.row_groups.iter().map(|rg| rg.active_rows() as u64).sum()
    }
    
    /// Total rows (including deleted) across all Row Groups
    pub fn total_rows(&self) -> u64 {
        self.row_groups.iter().map(|rg| rg.row_count as u64).sum()
    }
}

// ============================================================================
// Column Index Entry (32 bytes per column)
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct ColumnIndexEntry {
    pub data_offset: u64,
    pub data_length: u64,
    pub null_offset: u64,
    pub null_length: u64,
}

impl ColumnIndexEntry {
    pub fn to_bytes(&self) -> [u8; COLUMN_INDEX_ENTRY_SIZE] {
        let mut buf = [0u8; COLUMN_INDEX_ENTRY_SIZE];
        buf[0..8].copy_from_slice(&self.data_offset.to_le_bytes());
        buf[8..16].copy_from_slice(&self.data_length.to_le_bytes());
        buf[16..24].copy_from_slice(&self.null_offset.to_le_bytes());
        buf[24..32].copy_from_slice(&self.null_length.to_le_bytes());
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            data_offset: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            data_length: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            null_offset: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            null_length: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        }
    }
}

// ============================================================================
// Schema (bincode-free serialization)
// ============================================================================

/// A default value for a column constraint (serializable)
#[derive(Debug, Clone, PartialEq)]
pub enum DefaultValue {
    Int64(i64),
    Float64(f64),
    String(String),
    Bool(bool),
    Null,
}

/// Per-column constraint flags stored in schema
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ColumnConstraints {
    pub not_null: bool,
    pub primary_key: bool,
    pub unique: bool,
    pub default_value: Option<DefaultValue>,
    /// CHECK constraint expression stored as SQL text (re-parsed at enforcement time)
    pub check_expr_sql: Option<String>,
    /// FOREIGN KEY: references (table_name, column_name) in another table
    pub foreign_key: Option<(String, String)>,
    /// AUTOINCREMENT: auto-generate sequential integer values on INSERT
    pub autoincrement: bool,
}

#[derive(Debug, Clone, Default)]
pub struct OnDemandSchema {
    pub columns: Vec<(String, ColumnType)>,
    name_to_idx: HashMap<String, usize>,
    /// Per-column constraints (indexed same as columns)
    pub constraints: Vec<ColumnConstraints>,
}

impl OnDemandSchema {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            name_to_idx: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    pub fn add_column(&mut self, name: &str, dtype: ColumnType) -> usize {
        if let Some(&idx) = self.name_to_idx.get(name) {
            return idx;
        }
        let idx = self.columns.len();
        self.columns.push((name.to_string(), dtype));
        self.name_to_idx.insert(name.to_string(), idx);
        self.constraints.push(ColumnConstraints::default());
        idx
    }

    /// Add a column with constraints
    pub fn add_column_with_constraints(&mut self, name: &str, dtype: ColumnType, cons: ColumnConstraints) -> usize {
        if let Some(&idx) = self.name_to_idx.get(name) {
            if idx < self.constraints.len() {
                self.constraints[idx] = cons;
            }
            return idx;
        }
        let idx = self.columns.len();
        self.columns.push((name.to_string(), dtype));
        self.name_to_idx.insert(name.to_string(), idx);
        self.constraints.push(cons);
        idx
    }

    /// Get constraints for a column by name
    pub fn get_constraints(&self, name: &str) -> Option<&ColumnConstraints> {
        self.name_to_idx.get(name).and_then(|&idx| self.constraints.get(idx))
    }

    pub fn get_index(&self, name: &str) -> Option<usize> {
        self.name_to_idx.get(name).copied()
    }

    /// Rename a column in-place. Returns `true` if the column was found and renamed.
    pub fn rename_column(&mut self, old_name: &str, new_name: &str) -> bool {
        if let Some(&idx) = self.name_to_idx.get(old_name) {
            self.columns[idx].0 = new_name.to_string();
            let old_idx = self.name_to_idx.remove(old_name);
            if let Some(i) = old_idx {
                self.name_to_idx.insert(new_name.to_string(), i);
            }
            true
        } else {
            false
        }
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Serialize schema to bytes (no bincode)
    /// Layout: [col_count:u32][ [name_len:u16][name:bytes][type:u8] ... ][ [flags:u8] ... ]
    /// Constraint flags are appended AFTER all column data for backward compatibility.
    /// Constraint flags byte: bit0=not_null, bit1=primary_key, bit2=unique, bit4=autoincrement
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        
        // Column count
        buf.extend_from_slice(&(self.columns.len() as u32).to_le_bytes());
        
        // Each column: [name_len:u16][name:bytes][type:u8]
        for (name, dtype) in &self.columns {
            let name_bytes = name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            buf.push(*dtype as u8);
        }
        
        // Append constraint flags after all columns (1 byte per column)
        let has_any = self.constraints.iter().any(|c| c.not_null || c.primary_key || c.unique || c.default_value.is_some() || c.autoincrement);
        if has_any || !self.constraints.is_empty() {
            for i in 0..self.columns.len() {
                let cons = self.constraints.get(i).cloned().unwrap_or_default();
                let flags = (cons.not_null as u8)
                    | ((cons.primary_key as u8) << 1)
                    | ((cons.unique as u8) << 2)
                    | (if cons.default_value.is_some() { 0x08 } else { 0 })
                    | ((cons.autoincrement as u8) << 4);
                buf.push(flags);
            }
            // Serialize default values for columns that have them (bit3 set in flags)
            let has_defaults = self.constraints.iter().any(|c| c.default_value.is_some());
            if has_defaults {
                buf.push(0xDF); // marker byte for default values section
                for i in 0..self.columns.len() {
                    let cons = self.constraints.get(i);
                    match cons.and_then(|c| c.default_value.as_ref()) {
                        Some(DefaultValue::Int64(v)) => {
                            buf.push(1); // type tag
                            buf.extend_from_slice(&v.to_le_bytes());
                        }
                        Some(DefaultValue::Float64(v)) => {
                            buf.push(2);
                            buf.extend_from_slice(&v.to_le_bytes());
                        }
                        Some(DefaultValue::String(v)) => {
                            buf.push(3);
                            let sb = v.as_bytes();
                            buf.extend_from_slice(&(sb.len() as u16).to_le_bytes());
                            buf.extend_from_slice(sb);
                        }
                        Some(DefaultValue::Bool(v)) => {
                            buf.push(4);
                            buf.push(*v as u8);
                        }
                        Some(DefaultValue::Null) => {
                            buf.push(5);
                        }
                        None => {
                            buf.push(0); // no default
                        }
                    }
                }
            }
            
            // Serialize CHECK expressions (marker 0xCE)
            let has_checks = self.constraints.iter().any(|c| c.check_expr_sql.is_some());
            if has_checks {
                buf.push(0xCE); // marker byte for CHECK expressions section
                for i in 0..self.columns.len() {
                    let cons = self.constraints.get(i);
                    match cons.and_then(|c| c.check_expr_sql.as_ref()) {
                        Some(sql) => {
                            let sb = sql.as_bytes();
                            buf.extend_from_slice(&(sb.len() as u16).to_le_bytes());
                            buf.extend_from_slice(sb);
                        }
                        None => {
                            buf.extend_from_slice(&0u16.to_le_bytes()); // len=0 means no CHECK
                        }
                    }
                }
            }
            
            // Serialize FOREIGN KEY references (marker 0xFC)
            let has_fks = self.constraints.iter().any(|c| c.foreign_key.is_some());
            if has_fks {
                buf.push(0xFC); // marker byte for FOREIGN KEY section
                for i in 0..self.columns.len() {
                    let cons = self.constraints.get(i);
                    match cons.and_then(|c| c.foreign_key.as_ref()) {
                        Some((ref_table, ref_col)) => {
                            // Format: u16(table_len) + table_bytes + u16(col_len) + col_bytes
                            let tb = ref_table.as_bytes();
                            let cb = ref_col.as_bytes();
                            buf.extend_from_slice(&(tb.len() as u16).to_le_bytes());
                            buf.extend_from_slice(tb);
                            buf.extend_from_slice(&(cb.len() as u16).to_le_bytes());
                            buf.extend_from_slice(cb);
                        }
                        None => {
                            buf.extend_from_slice(&0u16.to_le_bytes()); // len=0 means no FK
                        }
                    }
                }
            }
        }
        
        buf
    }

    /// Deserialize schema from bytes (no bincode)
    /// Backward compatible: parses columns first, then checks if trailing bytes
    /// contain constraint flags (exactly column_count bytes remaining = constraints).
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut pos = 0;
        
        if bytes.len() < 4 {
            return Err(err_data("Schema too short"));
        }
        
        let column_count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        
        let mut schema = Self::new();
        
        // Parse columns: [name_len:u16][name:bytes][type:u8]
        for _ in 0..column_count {
            if pos + 2 > bytes.len() {
                return Err(err_data("Truncated schema"));
            }
            
            let name_len = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            
            if pos + name_len + 1 > bytes.len() {
                return Err(err_data("Truncated column"));
            }
            
            let name = std::str::from_utf8(&bytes[pos..pos + name_len])
                .map_err(|e| err_data(e.to_string()))?
                .to_string();
            pos += name_len;
            
            let dtype = ColumnType::from_u8(bytes[pos])
                .ok_or_else(|| err_data("Invalid column type"))?;
            pos += 1;
            
            schema.add_column(&name, dtype);
        }
        
        // Check for trailing constraint flags (at least column_count bytes remaining)
        let remaining = bytes.len() - pos;
        if remaining >= column_count && column_count > 0 {
            for i in 0..column_count {
                let flags = bytes[pos + i];
                if i < schema.constraints.len() {
                    schema.constraints[i] = ColumnConstraints {
                        not_null: flags & 1 != 0,
                        primary_key: flags & 2 != 0,
                        unique: flags & 4 != 0,
                        default_value: None, // filled below if marker present
                        check_expr_sql: None, // filled below if marker present
                        foreign_key: None, // filled below if marker present
                        autoincrement: flags & 0x10 != 0,
                    };
                }
            }
            pos += column_count;

            // Check for default values marker (0xDF) after flags
            if pos < bytes.len() && bytes[pos] == 0xDF {
                pos += 1;
                for i in 0..column_count {
                    if pos >= bytes.len() { break; }
                    let tag = bytes[pos];
                    pos += 1;
                    let dv = match tag {
                        1 => { // Int64
                            if pos + 8 > bytes.len() { break; }
                            let v = i64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap());
                            pos += 8;
                            Some(DefaultValue::Int64(v))
                        }
                        2 => { // Float64
                            if pos + 8 > bytes.len() { break; }
                            let v = f64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap());
                            pos += 8;
                            Some(DefaultValue::Float64(v))
                        }
                        3 => { // String
                            if pos + 2 > bytes.len() { break; }
                            let slen = u16::from_le_bytes(bytes[pos..pos+2].try_into().unwrap()) as usize;
                            pos += 2;
                            if pos + slen > bytes.len() { break; }
                            let s = std::str::from_utf8(&bytes[pos..pos+slen]).unwrap_or("").to_string();
                            pos += slen;
                            Some(DefaultValue::String(s))
                        }
                        4 => { // Bool
                            if pos >= bytes.len() { break; }
                            let v = bytes[pos] != 0;
                            pos += 1;
                            Some(DefaultValue::Bool(v))
                        }
                        5 => Some(DefaultValue::Null),
                        _ => None, // 0 = no default
                    };
                    if i < schema.constraints.len() {
                        schema.constraints[i].default_value = dv;
                    }
                }
            }
            
            // Check for CHECK expressions marker (0xCE) after defaults
            if pos < bytes.len() && bytes[pos] == 0xCE {
                pos += 1;
                for i in 0..column_count {
                    if pos + 2 > bytes.len() { break; }
                    let slen = u16::from_le_bytes(bytes[pos..pos+2].try_into().unwrap()) as usize;
                    pos += 2;
                    if slen > 0 {
                        if pos + slen > bytes.len() { break; }
                        let sql = std::str::from_utf8(&bytes[pos..pos+slen]).unwrap_or("").to_string();
                        pos += slen;
                        if i < schema.constraints.len() {
                            schema.constraints[i].check_expr_sql = Some(sql);
                        }
                    }
                }
            }
            
            // Check for FOREIGN KEY marker (0xFC) after CHECK
            if pos < bytes.len() && bytes[pos] == 0xFC {
                pos += 1;
                for i in 0..column_count {
                    if pos + 2 > bytes.len() { break; }
                    let tlen = u16::from_le_bytes(bytes[pos..pos+2].try_into().unwrap()) as usize;
                    pos += 2;
                    if tlen > 0 {
                        if pos + tlen > bytes.len() { break; }
                        let ref_table = std::str::from_utf8(&bytes[pos..pos+tlen]).unwrap_or("").to_string();
                        pos += tlen;
                        if pos + 2 > bytes.len() { break; }
                        let clen = u16::from_le_bytes(bytes[pos..pos+2].try_into().unwrap()) as usize;
                        pos += 2;
                        if pos + clen > bytes.len() { break; }
                        let ref_col = std::str::from_utf8(&bytes[pos..pos+clen]).unwrap_or("").to_string();
                        pos += clen;
                        if i < schema.constraints.len() {
                            schema.constraints[i].foreign_key = Some((ref_table, ref_col));
                        }
                    }
                }
            }
        }
        
        Ok(schema)
    }
}


