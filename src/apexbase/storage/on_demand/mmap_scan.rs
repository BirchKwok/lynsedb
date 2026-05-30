// Mmap scanning, footer loading, column range readers, filter+group+order fast path

/// Safe: cast byte slice to &[i64]. Falls back to owned Vec when pointer is not 8-byte aligned.
#[inline(always)]
fn bytes_as_i64_slice(bytes: &[u8], n: usize) -> std::borrow::Cow<'_, [i64]> {
    let ptr = bytes.as_ptr();
    if ptr as usize % 8 == 0 && bytes.len() >= n * 8 {
        std::borrow::Cow::Borrowed(unsafe { std::slice::from_raw_parts(ptr as *const i64, n) })
    } else {
        std::borrow::Cow::Owned((0..n).map(|i| i64::from_le_bytes(bytes[i*8..i*8+8].try_into().unwrap())).collect())
    }
}

/// Safe: cast byte slice to &[f64]. Falls back to owned Vec when pointer is not 8-byte aligned.
#[inline(always)]
fn bytes_as_f64_slice(bytes: &[u8], n: usize) -> std::borrow::Cow<'_, [f64]> {
    let ptr = bytes.as_ptr();
    if ptr as usize % 8 == 0 && bytes.len() >= n * 8 {
        std::borrow::Cow::Borrowed(unsafe { std::slice::from_raw_parts(ptr as *const f64, n) })
    } else {
        std::borrow::Cow::Owned((0..n).map(|i| f64::from_le_bytes(bytes[i*8..i*8+8].try_into().unwrap())).collect())
    }
}

/// Safe: cast byte slice to &[u64]. Falls back to owned Vec when pointer is not 8-byte aligned.
#[inline(always)]
fn bytes_as_u64_slice(bytes: &[u8], n: usize) -> std::borrow::Cow<'_, [u64]> {
    let ptr = bytes.as_ptr();
    if ptr as usize % 8 == 0 && bytes.len() >= n * 8 {
        std::borrow::Cow::Borrowed(unsafe { std::slice::from_raw_parts(ptr as *const u64, n) })
    } else {
        std::borrow::Cow::Owned((0..n).map(|i| u64::from_le_bytes(bytes[i*8..i*8+8].try_into().unwrap())).collect())
    }
}

/// Safe: cast byte slice to &[u32]. Falls back to owned Vec when pointer is not 4-byte aligned.
#[inline(always)]
fn bytes_as_u32_slice(bytes: &[u8], n: usize) -> std::borrow::Cow<'_, [u32]> {
    let ptr = bytes.as_ptr();
    if ptr as usize % 4 == 0 && bytes.len() >= n * 4 {
        std::borrow::Cow::Borrowed(unsafe { std::slice::from_raw_parts(ptr as *const u32, n) })
    } else {
        std::borrow::Cow::Owned((0..n).map(|i| u32::from_le_bytes(bytes[i*4..i*4+4].try_into().unwrap())).collect())
    }
}

#[inline(always)]
fn bitpack_value_at(packed: &[u8], bit_width: usize, min_val: i64, idx: usize) -> Option<i64> {
    if bit_width == 0 {
        return Some(min_val);
    }
    if bit_width >= 64 {
        return None;
    }
    let bit_pos = idx.checked_mul(bit_width)?;
    let byte_off = bit_pos / 8;
    let bit_shift = bit_pos % 8;
    let mask = (1u64 << bit_width) - 1;

    if bit_shift + bit_width <= 64 && byte_off + 8 <= packed.len() {
        let raw = unsafe { std::ptr::read_unaligned(packed.as_ptr().add(byte_off) as *const u64) };
        return Some(min_val.wrapping_add(((raw >> bit_shift) & mask) as i64));
    }

    let bytes_needed = (bit_shift + bit_width + 7) / 8;
    if byte_off + bytes_needed > packed.len() {
        return None;
    }
    let mut raw = 0u64;
    for j in 0..bytes_needed {
        raw |= (packed[byte_off + j] as u64) << (j * 8);
    }
    Some(min_val.wrapping_add(((raw >> bit_shift) & mask) as i64))
}

pub(crate) enum MmapBatchColumn {
    I64(Vec<Option<i64>>),
    F64(Vec<Option<f64>>),
    Str(Vec<Option<String>>),
    Bool(Vec<Option<bool>>),
    Bin(Vec<Option<Vec<u8>>>),
}

pub(crate) struct MmapBatchColumns {
    pub(crate) row_count: usize,
    pub(crate) columns: Vec<(String, MmapBatchColumn)>,
}

// ─── MULTI-PREDICATE PARALLEL SCAN ──────────────────────────────────────────

/// Scan predicate for `scan_multi_predicates_parallel`.
/// Each variant targets a single column and can be scanned independently.
pub enum MmapScanPred<'a> {
    NumericRange { col: &'a str, low: f64, high: f64 },
    StringEq { col: &'a str, value: &'a str },
    NumericIn { col: &'a str, values: &'a [i64] },
    StringIn { col: &'a str, values: &'a [String] },
}

// ─── LIKE PATTERN SUPPORT ────────────────────────────────────────────────────

/// Pre-classified LIKE pattern for zero-alloc byte-level matching.
/// Owned pattern bytes allow thread-safe sharing across Rayon parallel tasks.
#[derive(Clone)]
pub(crate) enum LikeKind {
    /// 'prefix%' — match strings starting with prefix bytes
    Prefix(Vec<u8>),
    /// '%suffix' — match strings ending with suffix bytes
    Suffix(Vec<u8>),
    /// '%substr%' — memmem scan within string bytes
    Contains(Vec<u8>),
    /// '%' — match all non-null rows
    Any,
    /// Complex pattern with '_' or multiple '%' — compiled regex
    Regex(regex::Regex),
}

/// Pre-compiled finder for contains patterns (much faster than on-the-fly)
/// Uses memchr's precompilation which caches SIMD state
pub struct PrecompiledFinder {
    finder: memchr::memmem::Finder<'static>,
}

/// Test whether a raw byte slice matches a LikeKind pattern.
/// Must not allocate — called inside Rayon parallel closures.
#[inline(always)]
pub(crate) fn like_matches_bytes(kind: &LikeKind, s: &[u8]) -> bool {
    match kind {
        LikeKind::Prefix(p)   => s.len() >= p.len() && fast_eq(p, s),
        LikeKind::Suffix(p)   => s.len() >= p.len() && fast_eq(p, &s[s.len()-p.len()..]),
        LikeKind::Contains(p) => memchr::memmem::find(s, p).is_some(),
        LikeKind::Any         => true,
        LikeKind::Regex(re)   => std::str::from_utf8(s).map(|st| re.is_match(st)).unwrap_or(false),
    }
}

/// Fast equality check using memchr's optimized comparison
/// Uses word-at-a-time comparison for longer prefixes
#[inline(always)]
fn fast_eq(pattern: &[u8], s: &[u8]) -> bool {
    if pattern.len() != s.len() {
        if pattern.len() > s.len() {
            return false;
        }
        // For prefix match: just check first pattern.len() bytes
        if s.len() < pattern.len() {
            return false;
        }
    }
    pattern == &s[..pattern.len()]
}

/// Classify a SQL LIKE pattern into a LikeKind for fast byte-level matching.
/// Returns None when the pattern has no wildcards (exact match → use scan_string_filter_mmap).
pub(crate) fn classify_like_pattern(pattern: &str) -> Option<LikeKind> {
    let pb = pattern.as_bytes();
    let plen = pb.len();
    if plen == 0 { return None; }
    if !pb.contains(&b'%') && !pb.contains(&b'_') { return None; }
    if pattern == "%" { return Some(LikeKind::Any); }
    let sw = pb[0] == b'%';
    let ew = pb[plen - 1] == b'%';
    if !sw && ew {
        let prefix = &pattern[..plen - 1];
        if !prefix.contains('%') && !prefix.contains('_') {
            return Some(LikeKind::Prefix(prefix.as_bytes().to_vec()));
        }
    }
    if sw && !ew {
        let suffix = &pattern[1..];
        if !suffix.contains('%') && !suffix.contains('_') {
            return Some(LikeKind::Suffix(suffix.as_bytes().to_vec()));
        }
    }
    if sw && ew && plen > 2 {
        let middle = &pattern[1..plen - 1];
        if !middle.contains('%') && !middle.contains('_') {
            return Some(LikeKind::Contains(middle.as_bytes().to_vec()));
        }
    }
    // Complex pattern → compile regex once, reuse across all rows
    let mut re_str = String::with_capacity(plen * 2 + 2);
    re_str.push('^');
    for c in pattern.chars() {
        match c {
            '%' => re_str.push_str(".*"),
            '_' => re_str.push('.'),
            '.' | '+' | '*' | '?' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\' => {
                re_str.push('\\'); re_str.push(c);
            }
            _ => re_str.push(c),
        }
    }
    re_str.push('$');
    regex::Regex::new(&re_str).ok().map(LikeKind::Regex)
}

// ─────────────────────────────────────────────────────────────────────────────

impl OnDemandStorage {
    /// Get the V4 footer, using cached version when file hasn't changed.
    /// Cache is invalidated when file size changes (another instance appended data)
    /// or explicitly via `invalidate_footer_cache()` after writes.
    pub(crate) fn get_or_load_footer(&self) -> io::Result<Option<V4Footer>> {
        // Fast path: if mmap is valid (file_size > 0) and footer is cached, return immediately
        // without any syscall. mmap_cache is always invalidated after writes, so file_size == 0
        // when stale. This avoids a metadata() syscall on every query — especially costly on Windows
        // where NtQueryAttributesFile requires a kernel transition + security descriptor check.
        {
            let mc = self.mmap_cache.read();
            if mc.file_size > 0 {
                let cached = self.v4_footer.read();
                if cached.is_some() {
                    return Ok(cached.clone());
                }
            }
        }

        let file_len = std::fs::metadata(&self.path)
            .map(|m| m.len())
            .unwrap_or(0);
        if file_len < HEADER_SIZE as u64 {
            return Ok(None);
        }

        // Secondary check: return cached footer if mmap size still matches
        {
            let cached = self.v4_footer.read();
            if let Some(ref footer) = *cached {
                let mc = self.mmap_cache.read();
                if mc.file_size == file_len {
                    return Ok(Some(footer.clone()));
                }
            }
        }

        // Invalidate mmap if file size changed (another instance appended data)
        {
            let mut mc = self.mmap_cache.write();
            if mc.file_size != 0 && mc.file_size != file_len {
                mc.invalidate();
            }
        }

        // Ensure file handle is open (reopen if needed after save_v4 replaced file)
        {
            let fg = self.file.read();
            if fg.is_none() {
                drop(fg);
                if let Ok(f) = open_for_sequential_read(&self.path) {
                    *self.file.write() = Some(f);
                }
            }
        }

        let file_guard = self.file.read();
        let file_handle = match file_guard.as_ref() {
            Some(f) => f,
            None => return Ok(None),
        };
        let mut mmap = self.mmap_cache.write();

        // Read the on-disk header fresh to get current footer_offset
        let mut header_bytes = [0u8; HEADER_SIZE];
        mmap.read_at(file_handle, &mut header_bytes, 0)?;
        let on_disk_header = OnDemandHeader::from_bytes(&header_bytes)?;

        if on_disk_header.footer_offset == 0 || on_disk_header.version != FORMAT_VERSION_V4 {
            return Ok(None);
        }
        let footer_offset = on_disk_header.footer_offset;
        if footer_offset >= file_len {
            return Ok(None);
        }

        let footer_byte_count = (file_len - footer_offset) as usize;
        if footer_byte_count < 16 {
            // Too small to hold even footer_size + magic trailer
            return Ok(None);
        }
        let mut footer_bytes = vec![0u8; footer_byte_count];
        mmap.read_at(file_handle, &mut footer_bytes, footer_offset)?;
        drop(mmap);
        drop(file_guard);

        // Validate footer magic before parsing.
        // During concurrent append_row_group, the header may still reference the
        // old footer_offset after the old footer has been overwritten with RG data.
        // In that case the bytes here are not a valid footer — return None so the
        // caller gracefully retries or falls back.
        if footer_byte_count < 8
            || &footer_bytes[footer_byte_count - 8..] != MAGIC_V4_FOOTER
        {
            return Ok(None);
        }

        let footer = match V4Footer::from_bytes(&footer_bytes) {
            Ok(f) => f,
            Err(_) => return Ok(None), // transient inconsistency during concurrent write
        };
        // Cache the footer for subsequent reads
        *self.v4_footer.write() = Some(footer.clone());
        Ok(Some(footer))
    }

    /// Invalidate the cached V4 footer (call after writes that change the footer).
    fn invalidate_footer_cache(&self) {
        *self.v4_footer.write() = None;
    }

    /// Read columns from on-disk V4 Row Groups directly via mmap → Arrow RecordBatch.
    /// Only materializes the requested columns; skips others with zero allocation.
    /// This is the core on-demand reading function that avoids loading all data into memory.
    ///
    /// # Arguments
    /// * `column_names` - Which columns to read (None = all)
    /// * `include_id` - Whether to include the _id column
    /// * `row_limit` - Maximum number of active rows to return (None = all)
    /// * `dict_encode_strings` - Whether to produce DictionaryArray for low-cardinality strings
    pub fn to_arrow_batch_mmap(
        &self,
        column_names: Option<&[&str]>,
        include_id: bool,
        row_limit: Option<usize>,
        dict_encode_strings: bool,
    ) -> io::Result<Option<RecordBatch>> {
        self.to_arrow_batch_mmap_range(column_names, include_id, 0, row_limit, dict_encode_strings)
    }

    /// Read active row ids and string columns directly from V4 mmap for FTS backfill.
    /// The caller falls back to the general Arrow path when this returns None.
    pub(crate) fn read_fts_string_columns_mmap(
        &self,
        column_names: &[String],
    ) -> io::Result<Option<(Vec<u32>, Vec<(String, ColumnData)>)>> {
        if column_names.is_empty() || self.has_delta() || self.has_v4_in_memory_data() {
            return Ok(None);
        }

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(Some((Vec::new(), Vec::new()))),
        };
        let schema = &footer.schema;

        let mut col_indices = Vec::with_capacity(column_names.len());
        for name in column_names {
            let Some(idx) = schema.get_index(name) else {
                return Ok(None);
            };
            match schema.columns[idx].1 {
                ColumnType::String | ColumnType::StringDict => col_indices.push(idx),
                _ => return Ok(None),
            }
        }

        let total_active = footer.total_active_rows() as usize;
        let file_guard = self.file.read();
        let file = file_guard
            .as_ref()
            .ok_or_else(|| err_not_conn("File not open for FTS mmap read"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        let mut doc_ids: Vec<u32> = Vec::with_capacity(total_active);
        let mut columns: Vec<(String, ColumnData)> = column_names
            .iter()
            .map(|name| (name.clone(), ColumnData::new(ColumnType::String)))
            .collect();

        for (rg_idx, rg_meta) in footer.row_groups.iter().enumerate() {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 || rg_meta.active_rows() == 0 {
                continue;
            }

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() {
                return Err(err_data("RG extends past EOF"));
            }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
            if rg_bytes.len() < 32 {
                return Ok(None);
            }

            let compress_flag = rg_bytes[28];
            let encoding_version = rg_bytes[29];
            if compress_flag != RG_COMPRESS_NONE
                || encoding_version < 1
                || rg_idx >= footer.col_offsets.len()
                || footer.col_offsets[rg_idx].is_empty()
            {
                return Ok(None);
            }

            let body = &rg_bytes[32..];
            let id_byte_len = rg_rows * 8;
            let del_vec_len = (rg_rows + 7) / 8;
            if id_byte_len + del_vec_len > body.len() {
                return Err(err_data("RG body truncated"));
            }

            let has_deletes = rg_meta.deletion_count > 0;
            let del_bytes = &body[id_byte_len..id_byte_len + del_vec_len];
            let active_indices: Option<Vec<usize>> = if has_deletes {
                Some(
                    (0..rg_rows)
                        .filter(|&i| (del_bytes[i / 8] >> (i % 8)) & 1 == 0)
                        .collect(),
                )
            } else {
                None
            };
            let active_len = active_indices.as_ref().map(|v| v.len()).unwrap_or(rg_rows);

            if let Some(indices) = active_indices.as_ref() {
                let ids = bytes_as_u64_slice(&body[..id_byte_len], rg_rows);
                doc_ids.extend(indices.iter().map(|&i| ids[i] as u32));
            } else if rg_meta.max_id == rg_meta.min_id + rg_rows as u64 - 1 {
                doc_ids.extend((0..rg_rows).map(|i| (rg_meta.min_id + i as u64) as u32));
            } else {
                let ids = bytes_as_u64_slice(&body[..id_byte_len], rg_rows);
                doc_ids.extend(ids.iter().map(|&id| id as u32));
            }

            let rg_col_offsets = &footer.col_offsets[rg_idx];
            let null_bitmap_len = (rg_rows + 7) / 8;
            for (out_pos, &col_idx) in col_indices.iter().enumerate() {
                let Some(&col_start_u32) = rg_col_offsets.get(col_idx) else {
                    let default_col = Self::create_default_column(ColumnType::String, active_len);
                    columns[out_pos].1.append(&default_col);
                    continue;
                };
                let col_start = col_start_u32 as usize;
                if col_start + null_bitmap_len > body.len() {
                    return Err(err_data("RG column null bitmap truncated"));
                }
                let null_bytes = &body[col_start..col_start + null_bitmap_len];
                let data_start = col_start + null_bitmap_len;
                if data_start > body.len() {
                    return Err(err_data("RG column data truncated"));
                }

                let col_type = schema.columns[col_idx].1;
                let (col_data, _) = read_column_encoded(&body[data_start..], col_type)?;
                let mut col_data = if matches!(col_data, ColumnData::StringDict { .. }) {
                    col_data.decode_string_dict()
                } else {
                    col_data
                };

                if let Some(indices) = active_indices.as_ref() {
                    col_data = col_data.filter_by_indices(indices);
                    if null_bytes.iter().any(|&b| b != 0) {
                        let mut active_nulls = vec![0u8; (indices.len() + 7) / 8];
                        for (new_idx, &old_idx) in indices.iter().enumerate() {
                            if (null_bytes[old_idx / 8] >> (old_idx % 8)) & 1 == 1 {
                                active_nulls[new_idx / 8] |= 1 << (new_idx % 8);
                            }
                        }
                        col_data.apply_null_bitmap(&active_nulls);
                    }
                } else if null_bytes.iter().any(|&b| b != 0) {
                    col_data.apply_null_bitmap(null_bytes);
                }

                columns[out_pos].1.append(&col_data);
            }
        }

        Ok(Some((doc_ids, columns)))
    }

    /// Read an active-row window from on-disk V4 Row Groups directly via mmap.
    /// `row_offset` is counted after delete filtering, matching `row_limit` semantics.
    pub fn to_arrow_batch_mmap_range(
        &self,
        column_names: Option<&[&str]>,
        include_id: bool,
        row_offset: usize,
        row_limit: Option<usize>,
        dict_encode_strings: bool,
    ) -> io::Result<Option<RecordBatch>> {
        use arrow::array::{Int64Array, StringArray, BooleanArray, PrimitiveArray};
        use arrow::buffer::{Buffer, NullBuffer, BooleanBuffer, ScalarBuffer};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType, Int64Type, Float64Type};
        use std::sync::Arc;

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None), // footer not yet written (empty/new file)
        };

        let schema = &footer.schema;
        let col_count = schema.column_count();

        // Determine which columns to read (indices into schema)
        let col_indices: Vec<usize> = if let Some(names) = column_names {
            names.iter()
                .filter(|&&n| n != "_id")
                .filter_map(|&name| schema.get_index(name))
                .collect()
        } else {
            (0..col_count).collect()
        };

        // Compute total active rows across all RGs
        let total_active: usize = footer.row_groups.iter()
            .map(|rg| rg.active_rows() as usize)
            .sum();

        if total_active == 0 {
            // Build empty schema and return empty batch
            let mut fields: Vec<Field> = Vec::new();
            if include_id {
                fields.push(Field::new("_id", ArrowDataType::Int64, false));
            }
            for &ci in &col_indices {
                let (name, ct) = &schema.columns[ci];
                let dt = match ct {
                    ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
                    ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 |
                    ColumnType::UInt32 | ColumnType::UInt64 => ArrowDataType::Int64,
                    ColumnType::Float64 | ColumnType::Float32 => ArrowDataType::Float64,
                    ColumnType::Bool => ArrowDataType::Boolean,
                    _ => ArrowDataType::Utf8,
                };
                fields.push(Field::new(name, dt, true));
            }
            let arrow_schema = Arc::new(Schema::new(fields));
            return Ok(Some(RecordBatch::new_empty(arrow_schema)));
        }

        let effective_start = row_offset.min(total_active);
        let effective_limit = row_limit
            .unwrap_or_else(|| total_active.saturating_sub(effective_start))
            .min(total_active.saturating_sub(effective_start));

        // Get mmap for the file
        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for mmap read"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        // Accumulators for each output column + _id
        let mut all_ids: Vec<i64> = Vec::with_capacity(effective_limit);
        // For each requested column, accumulate ColumnData across RGs
        let mut col_accumulators: Vec<ColumnData> = col_indices.iter()
            .map(|&ci| {
                let ct = schema.columns[ci].1;
                // StringDict is decoded to String before accumulation,
                // so accumulator must be String type
                let acc_type = if ct == ColumnType::StringDict { ColumnType::String } else { ct };
                ColumnData::new(acc_type)
            })
            .collect();
        let mut null_accumulators: Vec<Vec<bool>> = vec![Vec::new(); col_indices.len()];
        let mut rows_collected: usize = 0;
        let mut active_rows_seen: usize = 0;

        for (rg_idx, rg_meta) in footer.row_groups.iter().enumerate() {
            if rows_collected >= effective_limit {
                break;
            }
            if rg_meta.row_count == 0 {
                continue;
            }

            let rg_rows = rg_meta.row_count as usize;
            let rg_active = rg_meta.active_rows() as usize;
            if active_rows_seen + rg_active <= effective_start {
                active_rows_seen += rg_active;
                continue;
            }
            let active_skip = effective_start.saturating_sub(active_rows_seen).min(rg_active);
            let rows_to_take = (effective_limit - rows_collected).min(rg_active - active_skip);
            if rows_to_take == 0 {
                active_rows_seen += rg_active;
                continue;
            }

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() {
                return Err(err_data("RG extends past EOF"));
            }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];

            // Check compression flag at RG header byte 28
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let has_deletes = rg_meta.deletion_count > 0;
            let null_bitmap_len = (rg_rows + 7) / 8;

            // === RCIX fast path: O(1) direct seeks for no-compression, no-deletes ===
            // Skips sequential column scanning — jumps directly to each column via footer index.
            // For LIMIT 100 with 65536-row RG: touches ~800B of IDs + targeted column pages
            // instead of scanning 512KB IDs + full column sequence.
            if compress_flag == RG_COMPRESS_NONE
                && !has_deletes
                && encoding_version >= 1
                && rg_idx < footer.col_offsets.len()
                && !footer.col_offsets[rg_idx].is_empty()
            {
                let rg_body_abs = (rg_meta.offset + 32) as usize;
                let col_offsets = &footer.col_offsets[rg_idx];

                // Read only first rows_to_take IDs directly from mmap (avoids touching rest)
                {
                    let id_start = rg_body_abs + active_skip * 8;
                    let id_end = id_start + rows_to_take * 8;
                    if id_end <= mmap_ref.len() {
                        let id_bytes = &mmap_ref[id_start..id_end];
                        for i in 0..rows_to_take {
                            let id = u64::from_le_bytes(
                                id_bytes[i * 8..(i + 1) * 8].try_into().unwrap()
                            );
                            all_ids.push(id as i64);
                        }
                    }
                }

                // Direct column reads via RCIX — no sequential scan of preceding columns
                // OPTIMIZATION: parallelize column reads for large RGs with multiple columns
                if rows_to_take >= 50_000 && col_indices.len() >= 2 {
                    use rayon::prelude::*;
                    let create_default = Self::create_default_column;
                    let rg_col_results: Vec<io::Result<(ColumnData, Vec<bool>)>> = col_indices.par_iter()
                        .map(|&col_idx| {
                            if col_idx >= col_offsets.len() {
                                let col_type = schema.columns[col_idx].1;
                                let default_col = create_default(col_type, rows_to_take);
                                let nulls = vec![true; rows_to_take];
                                return Ok((default_col, nulls));
                            }
                            let col_abs = rg_body_abs + col_offsets[col_idx] as usize;
                            if col_abs + null_bitmap_len > mmap_ref.len() {
                                let col_type = schema.columns[col_idx].1;
                                return Ok((create_default(col_type, rows_to_take), vec![true; rows_to_take]));
                            }
                            let null_bytes = &mmap_ref[col_abs..col_abs + null_bitmap_len];
                            let data_abs = col_abs + null_bitmap_len;
                            if data_abs >= mmap_ref.len() {
                                let col_type = schema.columns[col_idx].1;
                                return Ok((create_default(col_type, rows_to_take), vec![true; rows_to_take]));
                            }
                            let col_type = schema.columns[col_idx].1;
                            let (col_data, _) = if active_skip == 0 && rows_to_take < rg_rows {
                                read_column_encoded_partial(&mmap_ref[data_abs..], col_type, rows_to_take)?
                            } else {
                                read_column_encoded(&mmap_ref[data_abs..], col_type)?
                            };
                            let col_data = if matches!(&col_data, ColumnData::StringDict { .. }) {
                                col_data.decode_string_dict()
                            } else {
                                col_data
                            };
                            let col_data = if active_skip > 0 || rows_to_take < col_data.len() {
                                col_data.slice_range(active_skip, active_skip + rows_to_take)
                            } else {
                                col_data
                            };
                            let mut nulls = Vec::with_capacity(rows_to_take);
                            for i in 0..rows_to_take {
                                let row = active_skip + i;
                                nulls.push((null_bytes[row / 8] >> (row % 8)) & 1 == 1);
                            }
                            Ok((col_data, nulls))
                        }).collect();
                    for (out_pos, result) in rg_col_results.into_iter().enumerate() {
                        let (col_data, nulls) = result?;
                        col_accumulators[out_pos].append(&col_data);
                        null_accumulators[out_pos].extend(nulls);
                    }
                } else {
                    for (out_pos, &col_idx) in col_indices.iter().enumerate() {
                        if col_idx >= col_offsets.len() {
                            let col_type = schema.columns[col_idx].1;
                            let default_col = Self::create_default_column(col_type, rows_to_take);
                            col_accumulators[out_pos].append(&default_col);
                            null_accumulators[out_pos].extend(std::iter::repeat(true).take(rows_to_take));
                            continue;
                        }
                        let col_abs = rg_body_abs + col_offsets[col_idx] as usize;
                        if col_abs + null_bitmap_len > mmap_ref.len() {
                            continue;
                        }
                        let null_bytes = &mmap_ref[col_abs..col_abs + null_bitmap_len];
                        let data_abs = col_abs + null_bitmap_len;
                        if data_abs >= mmap_ref.len() {
                            continue;
                        }
                        let col_type = schema.columns[col_idx].1;
                        let (col_data, _) = if active_skip == 0 && rows_to_take < rg_rows {
                            read_column_encoded_partial(&mmap_ref[data_abs..], col_type, rows_to_take)?
                        } else {
                            read_column_encoded(&mmap_ref[data_abs..], col_type)?
                        };
                        let col_data = if matches!(&col_data, ColumnData::StringDict { .. }) {
                            col_data.decode_string_dict()
                        } else {
                            col_data
                        };
                        let col_data = if active_skip > 0 || rows_to_take < col_data.len() {
                            col_data.slice_range(active_skip, active_skip + rows_to_take)
                        } else {
                            col_data
                        };
                        col_accumulators[out_pos].append(&col_data);
                        for i in 0..rows_to_take {
                            let row = active_skip + i;
                            null_accumulators[out_pos].push((null_bytes[row / 8] >> (row % 8)) & 1 == 1);
                        }
                    }
                }

                rows_collected += rows_to_take;
                active_rows_seen += rg_active;
                continue; // skip sequential scan path below
            }
            // === End RCIX fast path ===

            // Get the body bytes (after 32-byte RG header), decompressing if needed
            let decompressed_buf = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed_buf.as_deref().unwrap_or(&rg_bytes[32..]);
            let mut pos: usize = 0;

            // Read IDs
            let id_byte_len = rg_rows * 8;
            if pos + id_byte_len > body.len() {
                return Err(err_data("RG IDs truncated"));
            }
            let id_slice = &body[pos..pos + id_byte_len];
            pos += id_byte_len;

            // Read deletion vector
            let del_vec_len = (rg_rows + 7) / 8;
            if pos + del_vec_len > body.len() {
                return Err(err_data("RG deletion vector truncated"));
            }
            let del_bytes = &body[pos..pos + del_vec_len];
            pos += del_vec_len;

            // Always collect active IDs from this RG (needed for DeltaMerger overlay)
            {
                let mut skipped = 0usize;
                let mut taken = 0;
                for i in 0..rg_rows {
                    if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                        continue; // deleted
                    }
                    if skipped < active_skip {
                        skipped += 1;
                        continue;
                    }
                    let id = u64::from_le_bytes(
                        id_slice[i * 8..(i + 1) * 8].try_into().unwrap()
                    );
                    all_ids.push(id as i64);
                    taken += 1;
                    if taken >= rows_to_take { break; }
                }
            }

            // Parse columns — read requested, skip others
            // Build mapping: disk col_idx → output position in col_accumulators
            // This ensures correct data placement regardless of column ordering
            // between the footer schema and the requested column list.
            let col_idx_to_out: HashMap<usize, usize> = col_indices.iter()
                .enumerate()
                .map(|(out_pos, &col_idx)| (col_idx, out_pos))
                .collect();
            // Track which output columns got data from this RG
            let mut rg_filled: Vec<bool> = vec![false; col_indices.len()];
            for col_idx in 0..col_count {
                // Schema evolution: RG may have fewer columns than footer schema.
                // If we've exhausted the RG data, remaining columns get defaults.
                if pos + null_bitmap_len > body.len() {
                    break;
                }
                let null_bytes = &body[pos..pos + null_bitmap_len];
                pos += null_bitmap_len;

                let col_type = schema.columns[col_idx].1;

                if let Some(&out_pos) = col_idx_to_out.get(&col_idx) {
                    // OPTIMIZATION: For LIMIT queries without deletes, use partial column read
                    // to avoid allocating/copying full column data (e.g., 1M rows → only 100)
                    if !has_deletes && active_skip == 0 && rows_to_take < rg_rows && encoding_version >= 1 {
                        let (col_data, consumed) = read_column_encoded_partial(&body[pos..], col_type, rows_to_take)?;
                        pos += consumed;
                        let col_data = if matches!(&col_data, ColumnData::StringDict { .. }) {
                            col_data.decode_string_dict()
                        } else {
                            col_data
                        };
                        col_accumulators[out_pos].append(&col_data);
                        for i in 0..rows_to_take {
                            let is_null = (null_bytes[i / 8] >> (i % 8)) & 1 == 1;
                            null_accumulators[out_pos].push(is_null);
                        }
                    } else {
                        // Full column read path
                        let (col_data, consumed) = if encoding_version >= 1 {
                            read_column_encoded(&body[pos..], col_type)?
                        } else {
                            ColumnData::from_bytes_typed(&body[pos..], col_type)?
                        };
                        pos += consumed;

                        let col_data = if matches!(&col_data, ColumnData::StringDict { .. }) {
                            col_data.decode_string_dict()
                        } else {
                            col_data
                        };

                        if has_deletes {
                            let active_indices: Vec<usize> = (0..rg_rows)
                                .filter(|&i| (del_bytes[i / 8] >> (i % 8)) & 1 == 0)
                                .skip(active_skip)
                                .take(rows_to_take)
                                .collect();
                            let filtered = col_data.filter_by_indices(&active_indices);
                            col_accumulators[out_pos].append(&filtered);

                            for &old_idx in &active_indices {
                                let ob = old_idx / 8;
                                let obit = old_idx % 8;
                                let is_null = ob < null_bytes.len() && (null_bytes[ob] >> obit) & 1 == 1;
                                null_accumulators[out_pos].push(is_null);
                            }
                        } else {
                            if active_skip > 0 || rows_to_take < rg_rows {
                                let range_data = col_data.slice_range(active_skip, active_skip + rows_to_take);
                                col_accumulators[out_pos].append(&range_data);
                                for i in 0..rows_to_take {
                                    let row = active_skip + i;
                                    let is_null = (null_bytes[row / 8] >> (row % 8)) & 1 == 1;
                                    null_accumulators[out_pos].push(is_null);
                                }
                            } else {
                                col_accumulators[out_pos].append(&col_data);
                                for i in 0..rg_rows {
                                    let is_null = (null_bytes[i / 8] >> (i % 8)) & 1 == 1;
                                    null_accumulators[out_pos].push(is_null);
                                }
                            }
                        }
                    }
                    rg_filled[out_pos] = true;
                } else {
                    // Skip this column (no allocation, encoding-aware)
                    let consumed = if encoding_version >= 1 {
                        skip_column_encoded(&body[pos..], col_type)?
                    } else {
                        ColumnData::skip_bytes_typed(&body[pos..], col_type)?
                    };
                    pos += consumed;
                }
            }
            // Fill default values for columns that weren't in this RG (schema evolution)
            for (out_pos, filled) in rg_filled.iter().enumerate() {
                if !filled {
                    let col_type = schema.columns[col_indices[out_pos]].1;
                    let default_col = Self::create_default_column(col_type, rows_to_take);
                    col_accumulators[out_pos].append(&default_col);
                    // All rows are null for this missing column
                    null_accumulators[out_pos].extend(std::iter::repeat(true).take(rows_to_take));
                }
            }
            rows_collected += rows_to_take;
            active_rows_seen += rg_active;
        }

        drop(mmap_guard);
        drop(file_guard);

        // Build Arrow RecordBatch from accumulated data
        let active_count = rows_collected;
        let mut fields: Vec<Field> = Vec::with_capacity(col_indices.len() + 1);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(col_indices.len() + 1);

        // Save row IDs for potential DeltaMerger overlay
        let row_ids_for_delta: Vec<u64> = all_ids.iter().map(|&id| id as u64).collect();

        // _id column
        if include_id {
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(all_ids)));
        }

        // Data columns — parallel conversion for large tables
        use arrow::array::ArrayRef;
        let convert_mmap_column = |out_idx: usize, col_idx: usize| -> io::Result<(Field, ArrayRef)> {
            let (col_name, schema_col_type_ref) = &schema.columns[col_idx];
            let schema_col_type = *schema_col_type_ref;
            let col_data = &col_accumulators[out_idx];
            let null_bitmap = &null_accumulators[out_idx];

            // Build Arrow null buffer from per-bit bool accumulator
            let null_buf: Option<NullBuffer> = if !null_bitmap.is_empty() && null_bitmap.iter().any(|&b| b) {
                let mut validity_bytes = vec![0xFFu8; (active_count + 7) / 8];
                for (i, &is_null) in null_bitmap.iter().enumerate() {
                    if is_null {
                        // Clear validity bit (Arrow: 1=valid, 0=null)
                        validity_bytes[i / 8] &= !(1u8 << (i % 8));
                    }
                }
                Some(NullBuffer::new(BooleanBuffer::new(Buffer::from(validity_bytes), 0, active_count)))
            } else {
                None
            };

            let (arrow_dt, array): (ArrowDataType, ArrayRef) = match col_data {
                ColumnData::Int64(values) => {
                    match schema_col_type {
                        ColumnType::Timestamp => {
                            use arrow::datatypes::TimestampMicrosecondType;
                            let arr = PrimitiveArray::<TimestampMicrosecondType>::new(
                                ScalarBuffer::from(values.clone()), null_buf,
                            );
                            (ArrowDataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None), Arc::new(arr) as ArrayRef)
                        }
                        ColumnType::Date => {
                            use arrow::datatypes::Date32Type;
                            let data_i32: Vec<i32> = values.iter().map(|&v| v as i32).collect();
                            let arr = PrimitiveArray::<Date32Type>::new(
                                ScalarBuffer::from(data_i32), null_buf,
                            );
                            (ArrowDataType::Date32, Arc::new(arr) as ArrayRef)
                        }
                        _ => {
                            let arr = PrimitiveArray::<Int64Type>::new(
                                ScalarBuffer::from(values.clone()), null_buf,
                            );
                            (ArrowDataType::Int64, Arc::new(arr) as ArrayRef)
                        }
                    }
                }
                ColumnData::Float64(values) => {
                    let arr = PrimitiveArray::<Float64Type>::new(
                        ScalarBuffer::from(values.clone()), null_buf,
                    );
                    (ArrowDataType::Float64, Arc::new(arr) as ArrayRef)
                }
                ColumnData::String { offsets, data } => {
                    let count = offsets.len().saturating_sub(1);
                    if null_buf.is_some() {
                        let strings: Vec<Option<&str>> = (0..count.min(active_count)).map(|i| {
                            if i < null_bitmap.len() && null_bitmap[i] {
                                None
                            } else {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&data[start..end]).ok()
                            }
                        }).collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    } else {
                        // OPTIMIZATION: build StringArray directly from u32 offsets + data bytes
                        let row_count = count.min(active_count);
                        let data_end = offsets[row_count] as usize;
                        let mut offsets_i32: Vec<i32> = Vec::with_capacity(row_count + 1);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                offsets[..row_count + 1].as_ptr() as *const i32,
                                offsets_i32.as_mut_ptr(),
                                row_count + 1,
                            );
                            offsets_i32.set_len(row_count + 1);
                        }
                        let offset_buf = unsafe { arrow::buffer::OffsetBuffer::new_unchecked(ScalarBuffer::from(offsets_i32)) };
                        let data_buf = Buffer::from_slice_ref(&data[..data_end]);
                        (ArrowDataType::Utf8, Arc::new(unsafe { StringArray::new_unchecked(offset_buf, data_buf, None) }) as ArrayRef)
                    }
                }
                ColumnData::Bool { data: bool_data, len: bool_len } => {
                    let bools: Vec<Option<bool>> = (0..*bool_len.min(&active_count)).map(|i| {
                        if null_buf.is_some() {
                            if i < null_bitmap.len() && null_bitmap[i] {
                                return None;
                            }
                        }
                        let byte_idx = i / 8;
                        let bit_idx = i % 8;
                        let val = byte_idx < bool_data.len() && (bool_data[byte_idx] >> bit_idx) & 1 == 1;
                        Some(val)
                    }).collect();
                    (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                }
                ColumnData::Binary { offsets, data } => {
                    use arrow::array::BinaryArray;
                    let count = offsets.len().saturating_sub(1);
                    if null_buf.is_some() {
                        let bins: Vec<Option<&[u8]>> = (0..count.min(active_count)).map(|i| {
                            if i < null_bitmap.len() && null_bitmap[i] {
                                None
                            } else {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&data[start..end])
                            }
                        }).collect();
                        (ArrowDataType::Binary, Arc::new(BinaryArray::from(bins)))
                    } else {
                        let bins: Vec<&[u8]> = (0..count.min(active_count)).map(|i| {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            &data[start..end]
                        }).collect();
                        (ArrowDataType::Binary, Arc::new(BinaryArray::from(bins)))
                    }
                }
                ColumnData::FixedList { data, dim } => {
                    use arrow::array::{FixedSizeListArray, Float32Array};
                    let dim_usize = *dim as usize;
                    let row_count = if dim_usize == 0 { 0 } else { data.len() / (dim_usize * 4) }
                        .min(active_count);
                    let byte_len = row_count * dim_usize * 4;
                    let float_arr = Float32Array::from(
                        crate::storage::on_demand::f32_le_bytes_to_values(&data[..byte_len]),
                    );
                    let list_dt = ArrowDataType::FixedSizeList(
                        Arc::new(Field::new("item", ArrowDataType::Float32, false)),
                        dim_usize as i32,
                    );
                    let arr = FixedSizeListArray::new(
                        Arc::new(Field::new("item", ArrowDataType::Float32, false)),
                        dim_usize as i32,
                        Arc::new(float_arr),
                        None,
                    );
                    (list_dt, Arc::new(arr) as ArrayRef)
                }
                ColumnData::Float16List { data, dim } => {
                    use arrow::array::{FixedSizeListArray, Float32Array};
                    let dim_usize = *dim as usize;
                    let row_count = if dim_usize == 0 { 0 } else { data.len() / (dim_usize * 2) }
                        .min(active_count);
                    let mut f32_values: Vec<f32> = Vec::with_capacity(row_count * dim_usize);
                    for chunk in data[..row_count * dim_usize * 2].chunks_exact(2) {
                        let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                        f32_values.push(crate::storage::on_demand::f16_to_f32(bits));
                    }
                    let float_arr = Float32Array::from(f32_values);
                    let list_dt = ArrowDataType::FixedSizeList(
                        Arc::new(Field::new("item", ArrowDataType::Float32, false)),
                        dim_usize as i32,
                    );
                    let arr = FixedSizeListArray::new(
                        Arc::new(Field::new("item", ArrowDataType::Float32, false)),
                        dim_usize as i32,
                        Arc::new(float_arr),
                        None,
                    );
                    (list_dt, Arc::new(arr) as ArrayRef)
                }
                _ => {
                    // Fallback: null array
                    let arr = arrow::array::new_null_array(&ArrowDataType::Utf8, active_count);
                    (ArrowDataType::Utf8, arr)
                }
            };

            Ok((Field::new(col_name, arrow_dt, true), array))
        };

        if active_count >= 50_000 && col_indices.len() >= 2 {
            use rayon::prelude::*;
            let results: Vec<io::Result<(Field, ArrayRef)>> = col_indices.iter().enumerate()
                .collect::<Vec<_>>()
                .par_iter()
                .map(|&(out_idx, &col_idx)| convert_mmap_column(out_idx, col_idx))
                .collect();
            for r in results {
                let (f, a) = r?;
                fields.push(f);
                arrays.push(a);
            }
        } else {
            for (out_idx, &col_idx) in col_indices.iter().enumerate() {
                let (f, a) = convert_mmap_column(out_idx, col_idx)?;
                fields.push(f);
                arrays.push(a);
            }
        }

        let arrow_schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(arrow_schema, arrays)
            .map_err(|e| err_data(e.to_string()))?;

        // Apply DeltaStore overlay if there are pending cell-level updates
        let ds = self.delta_store.read();
        if !ds.is_empty() && batch.num_rows() > 0 {
            let merged = super::delta::DeltaMerger::merge(&batch, &ds, &row_ids_for_delta)?;
            return Ok(Some(merged));
        }

        Ok(Some(batch))
    }
    
    /// Read a single column for a contiguous row range (V4 in-memory path).
    fn read_column_auto(
        &self,
        col_idx: usize,
        col_type: ColumnType,
        start: usize,
        count: usize,
        _total_rows: usize,
        _is_v4: bool,
    ) -> io::Result<ColumnData> {
        let columns = self.columns.read();
        if col_idx < columns.len() && columns[col_idx].len() > 0 {
            Ok(columns[col_idx].slice_range(start, start + count))
        } else {
            Ok(Self::create_default_column(col_type, count))
        }
    }
    
    /// Read a single column for scattered row indices (V4 in-memory path).
    fn read_column_scattered_auto(
        &self,
        col_idx: usize,
        col_type: ColumnType,
        indices: &[usize],
        _total_rows: usize,
        _is_v4: bool,
    ) -> io::Result<ColumnData> {
        let columns = self.columns.read();
        if col_idx < columns.len() && columns[col_idx].len() > 0 {
            Ok(columns[col_idx].filter_by_indices(indices))
        } else {
            Ok(Self::create_default_column(col_type, indices.len()))
        }
    }

    /// Ensure IDs are loaded into memory (lazy loading optimization).
    /// Loads IDs from V4 Row Groups via mmap (lightweight — only IDs, not column data).
    fn ensure_ids_loaded(&self) -> io::Result<()> {
        // Quick check without write lock
        if !self.ids.read().is_empty() {
            return Ok(());
        }
        
        let header = self.header.read();
        let id_count = header.row_count as usize;
        
        if id_count == 0 {
            return Ok(());
        }
        
        drop(header);
        self.ensure_ids_loaded_v4()
    }

    /// V4-specific: Load IDs + deletion vector from Row Groups via mmap.
    /// This is lightweight — only reads IDs and deletion bitmaps, NOT column data.
    /// Enables delete/exists/id_index operations without loading full table into memory.
    fn ensure_ids_loaded_v4(&self) -> io::Result<()> {
        // Double-check under write lock
        let mut ids = self.ids.write();
        if !ids.is_empty() {
            return Ok(());
        }

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(()),
        };

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for V4 ID load"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        let total_active: usize = footer.row_groups.iter()
            .map(|rg| rg.row_count as usize)
            .sum();
        ids.reserve(total_active);
        let mut deleted_acc: Vec<u8> = Vec::new();
        let mut max_id: u64 = 0;

        for rg_meta in &footer.row_groups {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { continue; }

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() {
                return Err(err_data("RG extends past EOF"));
            }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            
            // Check compression flag at RG header byte 28
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            
            // Get the body bytes (after 32-byte RG header), decompressing if needed
            let decompressed_buf = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed_buf.as_deref().unwrap_or(&rg_bytes[32..]);
            let mut pos: usize = 0;

            // Read IDs
            let id_byte_len = rg_rows * 8;
            if pos + id_byte_len > body.len() {
                return Err(err_data("RG IDs truncated"));
            }
            let id_slice = &body[pos..pos + id_byte_len];
            pos += id_byte_len;

            for i in 0..rg_rows {
                let id = u64::from_le_bytes(
                    id_slice[i * 8..(i + 1) * 8].try_into().unwrap()
                );
                ids.push(id);
                if id > max_id { max_id = id; }
            }

            // Read deletion vector
            let del_vec_len = (rg_rows + 7) / 8;
            if pos + del_vec_len > body.len() {
                return Err(err_data("RG deletion vector truncated"));
            }
            deleted_acc.extend_from_slice(&body[pos..pos + del_vec_len]);
        }

        drop(mmap_guard);
        drop(file_guard);

        // Update deletion bitmap — always overwrite on first load (ids were empty above).
        // The pre-allocated zeros in self.deleted must not shadow the real on-disk state.
        if !deleted_acc.is_empty() {
            *self.deleted.write() = deleted_acc;
        }

        // Update next_id
        let current_next = self.next_id.load(Ordering::SeqCst);
        if max_id + 1 > current_next {
            self.next_id.store(max_id + 1, Ordering::SeqCst);
        }

        Ok(())
    }

    /// Scan specific columns from V4 Row Groups via mmap WITHOUT loading all data.
    /// Returns (Vec<ColumnData>, deletion_bitmap, Vec<null_bitmap_per_output_col>).
    /// This is the core building block for mmap-based fast paths (agg, filter, GROUP BY).
    fn scan_columns_mmap(
        &self,
        col_indices: &[usize],
        footer: &V4Footer,
    ) -> io::Result<(Vec<ColumnData>, Vec<u8>)> {
        let (cols, del, _nulls) = self.scan_columns_mmap_with_nulls(col_indices, footer)?;
        Ok((cols, del))
    }

    /// Same as scan_columns_mmap but also returns per-column null bitmaps.
    fn scan_columns_mmap_with_nulls(
        &self,
        col_indices: &[usize],
        footer: &V4Footer,
    ) -> io::Result<(Vec<ColumnData>, Vec<u8>, Vec<Vec<u8>>)> {
        let schema = &footer.schema;
        let col_count = schema.column_count();

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for mmap scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        // Pre-allocate accumulators to total row count — eliminates 16+ reallocations during RG iteration
        let total_rows: usize = footer.row_groups.iter().map(|rg| rg.row_count as usize).sum();
        let total_del_bytes = (total_rows + 7) / 8;
        let mut col_accumulators: Vec<ColumnData> = col_indices.iter()
            .map(|&ci| {
                let ct = schema.columns[ci].1;
                let acc_type = if ct == ColumnType::StringDict { ColumnType::String } else { ct };
                ColumnData::new(acc_type)
            })
            .collect();
        let mut all_del_bytes: Vec<u8> = Vec::with_capacity(total_del_bytes);
        let mut col_null_bitmaps: Vec<Vec<u8>> = col_indices.iter()
            .map(|_| Vec::with_capacity(total_del_bytes))
            .collect();

        let col_idx_to_out: HashMap<usize, usize> = col_indices.iter()
            .enumerate()
            .map(|(out_pos, &col_idx)| (col_idx, out_pos))
            .collect();

        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { continue; }

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() {
                return Err(err_data("RG extends past EOF"));
            }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            
            // Check compression flag at RG header byte 28, encoding version at byte 29
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            
            let null_bitmap_len = (rg_rows + 7) / 8;
            let del_vec_len = (rg_rows + 7) / 8;

            // RCIX FAST PATH: uncompressed + RCIX available → jump directly to each column
            // Eliminates sequential skip of unneeded columns (O(1) per column seek).
            let rcix = footer.col_offsets.get(rg_i).filter(|v| !v.is_empty());
            if compress_flag == RG_COMPRESS_NONE && encoding_version >= 1 && rcix.is_some() {
                let body = &rg_bytes[32..];
                let rg_col_offsets = rcix.unwrap();

                // Deletion vector starts after IDs
                let del_start = rg_rows * 8;
                if del_start + del_vec_len <= body.len() {
                    all_del_bytes.extend_from_slice(&body[del_start..del_start + del_vec_len]);
                }

                // Read only requested columns using RCIX offsets
                for (&col_idx, &out_pos) in &col_idx_to_out {
                    if col_idx >= rg_col_offsets.len() { continue; }
                    let col_start = rg_col_offsets[col_idx] as usize;
                    if col_start + null_bitmap_len > body.len() { continue; }

                    col_null_bitmaps[out_pos]
                        .extend_from_slice(&body[col_start..col_start + null_bitmap_len]);

                    let data_start = col_start + null_bitmap_len;
                    if data_start > body.len() { continue; }
                    let col_type = schema.columns[col_idx].1;
                    let (col_data, _) = read_column_encoded(&body[data_start..], col_type)?;
                    let col_data = if matches!(&col_data, ColumnData::StringDict { .. }) {
                        col_data.decode_string_dict()
                    } else {
                        col_data
                    };
                    col_accumulators[out_pos].append(&col_data);
                }
                continue; // Skip sequential path
            }

            // SEQUENTIAL PATH: compressed or no RCIX — decompress and scan all columns
            let decompressed_buf = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed_buf.as_deref().unwrap_or(&rg_bytes[32..]);
            let mut pos: usize = 0;

            // Skip IDs
            pos += rg_rows * 8;

            // Read deletion vector
            if pos + del_vec_len > body.len() {
                return Err(err_data("RG deletion vector truncated"));
            }
            all_del_bytes.extend_from_slice(&body[pos..pos + del_vec_len]);
            pos += del_vec_len;

            // Parse columns — read requested, skip others
            for col_idx in 0..col_count {
                if pos + null_bitmap_len > body.len() { break; }
                let null_bytes = &body[pos..pos + null_bitmap_len];
                pos += null_bitmap_len;

                let col_type = schema.columns[col_idx].1;

                if let Some(&out_pos) = col_idx_to_out.get(&col_idx) {
                    col_null_bitmaps[out_pos].extend_from_slice(null_bytes);

                    let (col_data, consumed) = if encoding_version >= 1 {
                        read_column_encoded(&body[pos..], col_type)?
                    } else {
                        ColumnData::from_bytes_typed(&body[pos..], col_type)?
                    };
                    pos += consumed;
                    let col_data = if matches!(&col_data, ColumnData::StringDict { .. }) {
                        col_data.decode_string_dict()
                    } else {
                        col_data
                    };
                    col_accumulators[out_pos].append(&col_data);
                } else {
                    let consumed = if encoding_version >= 1 {
                        skip_column_encoded(&body[pos..], col_type)?
                    } else {
                        ColumnData::skip_bytes_typed(&body[pos..], col_type)?
                    };
                    pos += consumed;
                }
            }
        }

        Ok((col_accumulators, all_del_bytes, col_null_bitmaps))
    }

    /// Determine which Row Groups can be skipped based on per-RG zone maps.
    /// Returns a set of RG indices that definitely won't match the filter.
    fn zone_map_prune_rgs(
        footer: &V4Footer,
        filter_col_idx: usize,
        filter_op: &str,
        filter_value: f64,
    ) -> HashSet<usize> {
        let mut skip: HashSet<usize> = HashSet::new();
        let filter_val_i64 = filter_value as i64;
        for (rg_idx, rg_zmaps) in footer.zone_maps.iter().enumerate() {
            for zm in rg_zmaps {
                if zm.col_idx as usize != filter_col_idx { continue; }
                let dominated = if zm.is_float {
                    let mn = f64::from_bits(zm.min_bits as u64);
                    let mx = f64::from_bits(zm.max_bits as u64);
                    match filter_op {
                        ">"  => mx <= filter_value,
                        ">=" => mx < filter_value,
                        "<"  => mn >= filter_value,
                        "<=" => mn > filter_value,
                        "=" | "==" => filter_value < mn || filter_value > mx,
                        _ => false,
                    }
                } else {
                    let mn = zm.min_bits;
                    let mx = zm.max_bits;
                    match filter_op {
                        ">"  => mx <= filter_val_i64,
                        ">=" => mx < filter_val_i64,
                        "<"  => mn >= filter_val_i64,
                        "<=" => mn > filter_val_i64,
                        "=" | "==" => filter_val_i64 < mn || filter_val_i64 > mx,
                        "!=" | "<>" => mn == mx && mn == filter_val_i64,
                        _ => false,
                    }
                };
                if dominated { skip.insert(rg_idx); }
                break; // only one zone map per column per RG
            }
        }
        skip
    }

    /// Like scan_columns_mmap but skips Row Groups in the `skip_rgs` set.
    /// Used for zone-map-pruned filtered scans.
    fn scan_columns_mmap_skip_rgs(
        &self,
        col_indices: &[usize],
        footer: &V4Footer,
        skip_rgs: &HashSet<usize>,
    ) -> io::Result<(Vec<ColumnData>, Vec<u8>)> {
        if skip_rgs.is_empty() {
            // No pruning needed — delegate to normal scan
            return self.scan_columns_mmap(col_indices, footer);
        }

        let schema = &footer.schema;
        let col_count = schema.column_count();

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for mmap scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        let mut col_accumulators: Vec<ColumnData> = col_indices.iter()
            .map(|&ci| {
                let ct = schema.columns[ci].1;
                let acc_type = if ct == ColumnType::StringDict { ColumnType::String } else { ct };
                ColumnData::new(acc_type)
            })
            .collect();
        let mut all_del_bytes: Vec<u8> = Vec::new();

        let col_idx_to_out: HashMap<usize, usize> = col_indices.iter()
            .enumerate()
            .map(|(out_pos, &col_idx)| (col_idx, out_pos))
            .collect();

        for (rg_idx, rg_meta) in footer.row_groups.iter().enumerate() {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { continue; }
            if skip_rgs.contains(&rg_idx) { continue; } // Zone map pruned!

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() {
                return Err(err_data("RG extends past EOF"));
            }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed_buf = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed_buf.as_deref().unwrap_or(&rg_bytes[32..]);
            let mut pos: usize = 0;

            pos += rg_rows * 8; // skip IDs

            let del_vec_len = (rg_rows + 7) / 8;
            if pos + del_vec_len > body.len() {
                return Err(err_data("RG deletion vector truncated"));
            }
            all_del_bytes.extend_from_slice(&body[pos..pos + del_vec_len]);
            pos += del_vec_len;

            let null_bitmap_len = (rg_rows + 7) / 8;
            for col_idx in 0..col_count {
                if pos + null_bitmap_len > body.len() { break; }
                pos += null_bitmap_len; // skip null bitmap

                let col_type = schema.columns[col_idx].1;
                if let Some(&out_pos) = col_idx_to_out.get(&col_idx) {
                    let (col_data, consumed) = if encoding_version >= 1 {
                        read_column_encoded(&body[pos..], col_type)?
                    } else {
                        ColumnData::from_bytes_typed(&body[pos..], col_type)?
                    };
                    pos += consumed;
                    let col_data = if matches!(&col_data, ColumnData::StringDict { .. }) {
                        col_data.decode_string_dict()
                    } else {
                        col_data
                    };
                    col_accumulators[out_pos].append(&col_data);
                } else {
                    let consumed = if encoding_version >= 1 {
                        skip_column_encoded(&body[pos..], col_type)?
                    } else {
                        ColumnData::skip_bytes_typed(&body[pos..], col_type)?
                    };
                    pos += consumed;
                }
            }
        }

        Ok((col_accumulators, all_del_bytes))
    }

    /// Read IDs for a row range (lazy loads IDs if not already loaded)
    pub fn read_ids(&self, start_row: usize, row_count: Option<usize>) -> io::Result<Vec<u64>> {
        // Ensure IDs are loaded (lazy loading)
        self.ensure_ids_loaded()?;

        let ids = self.ids.read();
        let base_total = ids.len();
        let delta_rows = self.delta_row_count();
        let total = base_total + delta_rows;
        let start = start_row.min(total);
        let count = row_count.map(|c| c.min(total - start)).unwrap_or(total - start);
        let end = start + count;

        let mut result = Vec::with_capacity(count);
        if start < base_total {
            let base_end = end.min(base_total);
            result.extend_from_slice(&ids[start..base_end]);
        }
        drop(ids);

        if end > base_total {
            if let Some((delta_ids, _)) = self.read_delta_data()? {
                let delta_start = start.saturating_sub(base_total);
                let delta_end = end
                    .saturating_sub(base_total)
                    .min(delta_ids.len());
                if delta_start < delta_end {
                    result.extend_from_slice(&delta_ids[delta_start..delta_end]);
                }
            }
        }

        Ok(result)
    }

    /// Read IDs for specific row indices (optimized for scattered access, lazy loads)
    pub fn read_ids_by_indices(&self, row_indices: &[usize]) -> io::Result<Vec<i64>> {
        // Ensure IDs are loaded (lazy loading)
        self.ensure_ids_loaded()?;
        
        let ids = self.ids.read();
        let total = ids.len();
        Ok(row_indices.iter()
            .map(|&i| if i < total { ids[i] as i64 } else { 0 })
            .collect())
    }

    /// Execute Complex (Filter+Group+Order) query with single-pass optimization.
    /// V4 streaming mmap path for:
    ///   SELECT group_col, AGG(agg_col) FROM t WHERE filter_col = 'val'
    ///   GROUP BY group_col ORDER BY agg DESC/ASC LIMIT n OFFSET m
    ///
    /// Single pass per RG: filter→group→aggregate via RCIX zero-copy.
    /// Returns None if prerequisites not met (compressed RGs, missing RCIX, etc.)
    /// so the caller falls back to the generic query path.
    pub fn execute_filter_group_order(
        &self,
        filter_col: &str,
        filter_val: &str,
        group_col: &str,
        agg_col: Option<&str>,
        agg_func: crate::query::AggregateFunc,
        descending: bool,
        limit: usize,
        offset: usize,
    ) -> io::Result<Option<RecordBatch>> {
        use crate::query::AggregateFunc;
        use std::collections::HashMap;

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;

        // Resolve column indices
        let filter_idx = match schema.get_index(filter_col) {
            Some(i) => i,
            None => return Ok(None),
        };
        let group_idx = match schema.get_index(group_col) {
            Some(i) => i,
            None => return Ok(None),
        };
        let agg_idx = agg_col.and_then(|ac| schema.get_index(ac));

        let filter_type = schema.columns[filter_idx].1;
        let group_type = schema.columns[group_idx].1;

        // Only support string-typed filter and group columns
        if !matches!(filter_type, ColumnType::String | ColumnType::StringDict) {
            return Ok(None);
        }
        if !matches!(group_type, ColumnType::String | ColumnType::StringDict) {
            return Ok(None);
        }

        // Check agg column is numeric (if present)
        if let Some(ai) = agg_idx {
            let at = schema.columns[ai].1;
            if !matches!(at, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
                ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 |
                ColumnType::UInt32 | ColumnType::UInt64 | ColumnType::Float64 | ColumnType::Float32) {
                return Ok(None);
            }
        }

        // Verify all RGs have RCIX and are uncompressed
        let max_col = [filter_idx, group_idx].iter().copied()
            .chain(agg_idx.iter().copied()).max().unwrap_or(0);
        let all_rcix = footer.row_groups.iter().enumerate().all(|(rg_i, rg_meta)| {
            if rg_meta.row_count == 0 { return true; }
            footer.col_offsets.get(rg_i).map_or(false, |v| v.len() > max_col)
        });
        if !all_rcix { return Ok(None); }

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for filter-group-order"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        let filter_val_bytes = filter_val.as_bytes();
        let filter_val_len = filter_val_bytes.len();

        // Accumulate: group_key → (sum, count)
        let mut groups: HashMap<String, (f64, i64)> = HashMap::new();

        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { continue; }
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Err(err_data("RG past EOF")); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
            if rg_bytes.len() < 32 { continue; }
            let compress_flag = rg_bytes[28];
            let encoding_version = rg_bytes[29];
            if compress_flag != RG_COMPRESS_NONE || encoding_version < 1 {
                return Ok(None); // fallback for compressed RGs
            }

            let body = &rg_bytes[32..];
            let null_bitmap_len = (rg_rows + 7) / 8;
            let has_deleted = rg_meta.deletion_count > 0;
            let del_start = rg_rows * 8;
            let del_vec_len = null_bitmap_len;
            let del_bytes: &[u8] = if has_deleted && del_start + del_vec_len <= body.len() {
                &body[del_start..del_start + del_vec_len]
            } else { &[] };
            let rcix = &footer.col_offsets[rg_i];

            // ── Parse filter column (StringDict or String) ──
            let f_col_off = rcix[filter_idx] as usize;
            let f_data_start = f_col_off + null_bitmap_len;
            if f_data_start >= body.len() { continue; }
            let f_bytes = &body[f_data_start..];
            if f_bytes.is_empty() { continue; }
            let f_encoding = f_bytes[0];
            if f_encoding != COL_ENCODING_PLAIN { return Ok(None); }
            let f_data = &f_bytes[1..];

            // ── Parse group column ──
            let g_col_off = rcix[group_idx] as usize;
            let g_data_start = g_col_off + null_bitmap_len;
            if g_data_start >= body.len() { continue; }
            let g_bytes = &body[g_data_start..];
            if g_bytes.is_empty() { continue; }
            let g_encoding = g_bytes[0];
            if g_encoding != COL_ENCODING_PLAIN { return Ok(None); }
            let g_data = &g_bytes[1..];

            // ── Parse agg column (optional, numeric) ──
            enum AggSlice<'a> { None, F64(&'a [f64]), I64(&'a [i64]), OwnedF64(Vec<f64>), OwnedI64(Vec<i64>) }
            let agg_slice: AggSlice = if let Some(ai) = agg_idx {
                let a_col_off = rcix[ai] as usize;
                let a_data_start = a_col_off + null_bitmap_len;
                if a_data_start + 1 >= body.len() { AggSlice::None } else {
                    let a_enc = body[a_data_start];
                    let a_payload = &body[a_data_start + 1..];
                    if a_enc == COL_ENCODING_PLAIN && a_payload.len() >= 8 {
                        let count = u64::from_le_bytes(a_payload[0..8].try_into().unwrap()) as usize;
                        let col_type = schema.columns[ai].1;
                        let n = count.min(rg_rows);
                        if matches!(col_type, ColumnType::Float64 | ColumnType::Float32) {
                            let cow = bytes_as_f64_slice(&a_payload[8..], n);
                            match cow {
                                std::borrow::Cow::Borrowed(s) => AggSlice::F64(s),
                                std::borrow::Cow::Owned(v) => AggSlice::OwnedF64(v),
                            }
                        } else {
                            let cow = bytes_as_i64_slice(&a_payload[8..], n);
                            match cow {
                                std::borrow::Cow::Borrowed(s) => AggSlice::I64(s),
                                std::borrow::Cow::Owned(v) => AggSlice::OwnedI64(v),
                            }
                        }
                    } else { AggSlice::None }
                }
            } else { AggSlice::None };
            let agg_f64: Option<&[f64]> = match &agg_slice { AggSlice::F64(s) => Some(s), AggSlice::OwnedF64(v) => Some(v.as_slice()), _ => None };
            let agg_i64: Option<&[i64]> = match &agg_slice { AggSlice::I64(s) => Some(s), AggSlice::OwnedI64(v) => Some(v.as_slice()), _ => None };

            // ── Determine matching rows from filter column ──
            let filter_ct = schema.columns[filter_idx].1;
            let group_ct = schema.columns[group_idx].1;

            // ── FAST PATH: Both filter and group are StringDict ──
            // Use flat-array accumulation indexed by group dict_index (zero allocs in hot loop).
            // After this RG, merge O(dict_size) entries into global HashMap.
            if matches!(filter_ct, ColumnType::StringDict) && matches!(group_ct, ColumnType::StringDict) {
                // Parse filter dict
                if f_data.len() < 16 { continue; }
                let f_row_count = u64::from_le_bytes(f_data[0..8].try_into().unwrap()) as usize;
                let f_dict_size = u64::from_le_bytes(f_data[8..16].try_into().unwrap()) as usize;
                if f_dict_size == 0 { continue; }
                let f_indices = bytes_as_u32_slice(&f_data[16..], f_row_count);
                let f_dict_off_start = 16 + f_row_count * 4;
                let f_dict_offsets = bytes_as_u32_slice(&f_data[f_dict_off_start..], f_dict_size);
                let f_dict_data_len_off = f_dict_off_start + f_dict_size * 4;
                if f_dict_data_len_off + 8 > f_data.len() { continue; }
                let f_dict_data_len = u64::from_le_bytes(f_data[f_dict_data_len_off..f_dict_data_len_off+8].try_into().unwrap()) as usize;
                let f_dict_data_start = f_dict_data_len_off + 8;
                let f_raw_end = (f_dict_data_start + f_dict_data_len).min(f_data.len());
                let f_raw_dict = &f_data[f_dict_data_start..f_raw_end];

                // Find target filter dict index
                let mut target_dict_idx: Option<u32> = None;
                for di in 0..f_dict_size {
                    let start = f_dict_offsets[di] as usize;
                    let end = if di + 1 < f_dict_size { f_dict_offsets[di + 1] as usize } else { f_raw_dict.len() };
                    if end - start == filter_val_len && &f_raw_dict[start..end] == filter_val_bytes {
                        target_dict_idx = Some((di + 1) as u32);
                        break;
                    }
                }
                let tdi = match target_dict_idx { Some(v) => v, None => continue };

                // Parse group dict
                if g_data.len() < 16 { continue; }
                let g_row_count = u64::from_le_bytes(g_data[0..8].try_into().unwrap()) as usize;
                let g_dict_size = u64::from_le_bytes(g_data[8..16].try_into().unwrap()) as usize;
                let g_indices = bytes_as_u32_slice(&g_data[16..], g_row_count);
                let g_dict_off_start = 16 + g_row_count * 4;
                let g_dict_offsets = bytes_as_u32_slice(&g_data[g_dict_off_start..], g_dict_size);
                let g_dict_data_len_off = g_dict_off_start + g_dict_size * 4;
                if g_dict_data_len_off + 8 > g_data.len() { continue; }
                let g_dict_data_len = u64::from_le_bytes(g_data[g_dict_data_len_off..g_dict_data_len_off+8].try_into().unwrap()) as usize;
                let g_dict_data_start = g_dict_data_len_off + 8;
                let g_dict_data_end = (g_dict_data_start + g_dict_data_len).min(g_data.len());
                let g_raw_dict = &g_data[g_dict_data_start..g_dict_data_end];

                // Flat accumulation arrays indexed by group dict_index (1-based, slot 0 = null)
                let flat_size = g_dict_size + 1; // +1 for null slot at index 0
                let mut flat_sums = vec![0.0f64; flat_size];
                let mut flat_counts = vec![0i64; flat_size];

                // Hot loop: scan filter indices, accumulate into flat arrays (zero allocations)
                let n = f_row_count.min(rg_rows).min(g_row_count);
                if !has_deleted {
                    if let Some(av) = agg_f64 {
                        let an = av.len();
                        for i in 0..n {
                            if unsafe { *f_indices.get_unchecked(i) } == tdi {
                                let gid = unsafe { *g_indices.get_unchecked(i) } as usize;
                                if gid < flat_size {
                                    unsafe { *flat_counts.get_unchecked_mut(gid) += 1; }
                                    if i < an { unsafe { *flat_sums.get_unchecked_mut(gid) += *av.get_unchecked(i); } }
                                }
                            }
                        }
                    } else if let Some(av) = agg_i64 {
                        let an = av.len();
                        for i in 0..n {
                            if unsafe { *f_indices.get_unchecked(i) } == tdi {
                                let gid = unsafe { *g_indices.get_unchecked(i) } as usize;
                                if gid < flat_size {
                                    unsafe { *flat_counts.get_unchecked_mut(gid) += 1; }
                                    if i < an { unsafe { *flat_sums.get_unchecked_mut(gid) += *av.get_unchecked(i) as f64; } }
                                }
                            }
                        }
                    } else {
                        // COUNT(*) only
                        for i in 0..n {
                            if unsafe { *f_indices.get_unchecked(i) } == tdi {
                                let gid = unsafe { *g_indices.get_unchecked(i) } as usize;
                                if gid < flat_size { unsafe { *flat_counts.get_unchecked_mut(gid) += 1; } }
                            }
                        }
                    }
                } else {
                    // Path with deletion checks
                    for i in 0..n {
                        if !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                        if f_indices[i] == tdi {
                            let gid = g_indices[i] as usize;
                            if gid < flat_size {
                                flat_counts[gid] += 1;
                                if let Some(av) = agg_f64 { if i < av.len() { flat_sums[gid] += av[i]; } }
                                else if let Some(av) = agg_i64 { if i < av.len() { flat_sums[gid] += av[i] as f64; } }
                            }
                        }
                    }
                }

                // Merge flat arrays into global HashMap (O(dict_size), not O(matching_rows))
                for gid in 1..flat_size {
                    if flat_counts[gid] > 0 {
                        let di = gid - 1;
                        let start = g_dict_offsets[di] as usize;
                        let end = if di + 1 < g_dict_size { g_dict_offsets[di + 1] as usize } else { g_raw_dict.len() };
                        let key = std::str::from_utf8(&g_raw_dict[start..end]).unwrap_or("");
                        let entry = groups.entry(key.to_string()).or_insert((0.0, 0));
                        entry.0 += flat_sums[gid];
                        entry.1 += flat_counts[gid];
                    }
                }
                continue; // done with this RG
            }

            // ── FALLBACK: non-StringDict group or filter column ──
            // Parse group column for per-row lookup
            enum GroupResolver<'a> {
                Dict { indices: std::borrow::Cow<'a, [u32]>, strings: Vec<&'a str> },
                Plain { count: usize, offsets: std::borrow::Cow<'a, [u32]>, data: &'a [u8], data_start: usize },
            }
            impl<'a> GroupResolver<'a> {
                fn get(&self, row: usize) -> Option<&str> {
                    match self {
                        GroupResolver::Dict { indices, strings } => {
                            if row >= indices.len() { return None; }
                            let idx = indices[row] as usize;
                            if idx == 0 { return Some(""); }
                            strings.get(idx - 1).copied()
                        }
                        GroupResolver::Plain { count, offsets, data, data_start } => {
                            if row >= *count { return None; }
                            let s = offsets[row] as usize;
                            let e = offsets[row + 1] as usize;
                            if *data_start + e > data.len() { return None; }
                            std::str::from_utf8(&data[*data_start + s..*data_start + e]).ok()
                        }
                    }
                }
            }

            let g_resolver = if matches!(group_ct, ColumnType::StringDict) {
                if g_data.len() < 16 { continue; }
                let row_count = u64::from_le_bytes(g_data[0..8].try_into().unwrap()) as usize;
                let dict_size = u64::from_le_bytes(g_data[8..16].try_into().unwrap()) as usize;
                let indices = bytes_as_u32_slice(&g_data[16..], row_count);
                let dict_off_start = 16 + row_count * 4;
                let dict_offsets = bytes_as_u32_slice(&g_data[dict_off_start..], dict_size);
                let dict_data_len_off = dict_off_start + dict_size * 4;
                if dict_data_len_off + 8 > g_data.len() { continue; }
                let dict_data_len = u64::from_le_bytes(g_data[dict_data_len_off..dict_data_len_off+8].try_into().unwrap()) as usize;
                let dict_data_start = dict_data_len_off + 8;
                let dict_data_end = (dict_data_start + dict_data_len).min(g_data.len());
                let raw_dict = &g_data[dict_data_start..dict_data_end];
                let mut strings: Vec<&str> = Vec::with_capacity(dict_size);
                for di in 0..dict_size {
                    let start = dict_offsets[di] as usize;
                    let end = if di + 1 < dict_size { dict_offsets[di + 1] as usize } else { raw_dict.len() };
                    strings.push(std::str::from_utf8(&raw_dict[start..end]).unwrap_or(""));
                }
                GroupResolver::Dict { indices, strings }
            } else {
                if g_data.len() < 8 { continue; }
                let count = u64::from_le_bytes(g_data[0..8].try_into().unwrap()) as usize;
                let offsets = bytes_as_u32_slice(&g_data[8..], count + 1);
                let data_len_off = 8 + (count + 1) * 4;
                if data_len_off + 8 > g_data.len() { continue; }
                let data_len = u64::from_le_bytes(g_data[data_len_off..data_len_off+8].try_into().unwrap()) as usize;
                let data_start = data_len_off + 8;
                GroupResolver::Plain { count, offsets, data: g_data, data_start }
            };

            macro_rules! accumulate {
                ($row:expr) => {{
                    if let Some(group_str) = g_resolver.get($row) {
                        let entry = groups.entry(group_str.to_string()).or_insert((0.0, 0));
                        entry.1 += 1;
                        if let Some(av) = agg_f64 {
                            if $row < av.len() { entry.0 += av[$row]; }
                        } else if let Some(av) = agg_i64 {
                            if $row < av.len() { entry.0 += av[$row] as f64; }
                        }
                    }
                }};
            }

            // ── Scan filter column for matching rows ──
            if matches!(filter_ct, ColumnType::StringDict) {
                if f_data.len() < 16 { continue; }
                let row_count = u64::from_le_bytes(f_data[0..8].try_into().unwrap()) as usize;
                let dict_size = u64::from_le_bytes(f_data[8..16].try_into().unwrap()) as usize;
                if dict_size == 0 { continue; }
                let indices_cow = bytes_as_u32_slice(&f_data[16..], row_count);
                let indices: &[u32] = &indices_cow;
                let dict_off_start = 16 + row_count * 4;
                let dict_offsets_cow = bytes_as_u32_slice(&f_data[dict_off_start..], dict_size);
                let dict_offsets: &[u32] = &dict_offsets_cow;
                let dict_data_len_off = dict_off_start + dict_size * 4;
                if dict_data_len_off + 8 > f_data.len() { continue; }
                let dict_data_len = u64::from_le_bytes(f_data[dict_data_len_off..dict_data_len_off+8].try_into().unwrap()) as usize;
                let f_dict_data_start = dict_data_len_off + 8;
                let f_raw_end = (f_dict_data_start + dict_data_len).min(f_data.len());
                let f_raw_dict = &f_data[f_dict_data_start..f_raw_end];
                // Find target dict index
                let mut target_dict_idx: Option<u32> = None;
                for di in 0..dict_size {
                    let start = dict_offsets[di] as usize;
                    let end = if di + 1 < dict_size { dict_offsets[di + 1] as usize } else { f_raw_dict.len() };
                    if end - start == filter_val_len && &f_raw_dict[start..end] == filter_val_bytes {
                        target_dict_idx = Some((di + 1) as u32);
                        break;
                    }
                }
                if let Some(tdi) = target_dict_idx {
                    let n = row_count.min(rg_rows);
                    for i in 0..n {
                        if has_deleted && !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                        if indices[i] == tdi { accumulate!(i); }
                    }
                }
            } else {
                // Plain String filter
                if f_data.len() < 8 { continue; }
                let count = u64::from_le_bytes(f_data[0..8].try_into().unwrap()) as usize;
                let offsets_cow = bytes_as_u32_slice(&f_data[8..], count + 1);
                let offsets: &[u32] = &offsets_cow;
                let data_len_off = 8 + (count + 1) * 4;
                if data_len_off + 8 > f_data.len() { continue; }
                let data_start = data_len_off + 8;
                let n = count.min(rg_rows);
                for i in 0..n {
                    if has_deleted && !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                    let s = offsets[i] as usize;
                    let e = offsets[i + 1] as usize;
                    if e - s == filter_val_len && &f_data[data_start + s..data_start + e] == filter_val_bytes {
                        accumulate!(i);
                    }
                }
            }
        } // end RG loop

        if groups.is_empty() {
            return Ok(None);
        }

        // Compute final aggregate values and sort
        let mut results: Vec<(String, f64)> = groups.into_iter().map(|(k, (sum, count))| {
            let val = match agg_func {
                AggregateFunc::Sum => sum,
                AggregateFunc::Count => count as f64,
                AggregateFunc::Avg => if count > 0 { sum / count as f64 } else { 0.0 },
                _ => sum,
            };
            (k, val)
        }).collect();

        if descending {
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        let results: Vec<_> = results.into_iter().skip(offset).take(limit).collect();
        if results.is_empty() {
            return Ok(None);
        }

        // Build Arrow RecordBatch
        use std::sync::Arc;
        use arrow::array::{StringArray, Float64Array};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType};
        let group_values: Vec<&str> = results.iter().map(|(k, _)| k.as_str()).collect();
        let agg_values: Vec<f64> = results.iter().map(|(_, v)| *v).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new(group_col, ArrowDataType::Utf8, false),
            Field::new("agg_result", ArrowDataType::Float64, false),
        ]));
        let arrays: Vec<Arc<dyn arrow::array::Array>> = vec![
            Arc::new(StringArray::from(group_values)),
            Arc::new(Float64Array::from(agg_values)),
        ];
        let batch = RecordBatch::try_new(schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        Ok(Some(batch))
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Ensure id_to_idx AHashMap is built (lazy load)
    /// Called automatically by delete/exists/get_row_idx operations
    fn ensure_id_index(&self) {
        // First ensure IDs are loaded (since we lazy-load them now)
        let _ = self.ensure_ids_loaded();
        
        let mut id_to_idx = self.id_to_idx.write();
        if id_to_idx.is_none() {
            let ids = self.ids.read();
            let mut map = ahash::AHashMap::with_capacity(ids.len());
            for (idx, &id) in ids.iter().enumerate() {
                map.insert(id, idx);
            }
            *id_to_idx = Some(map);
        }
    }

    // ========================================================================
    // Internal read helpers (mmap-based for cross-platform zero-copy reads)
    // ========================================================================

    fn read_column_range_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        dtype: ColumnType,
        start_row: usize,
        row_count: usize,
        _total_rows: usize,
    ) -> io::Result<ColumnData> {
        // ColumnData format has an 8-byte count header for all types
        // Format: [count:u64][data...]
        const HEADER_SIZE: u64 = 8;
        
        match dtype {
            ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 |
            ColumnType::Timestamp | ColumnType::Date => {
                // Format: [count:u64][values:i64*]
                // Zero-copy optimization: read directly into i64 buffer
                let byte_offset = HEADER_SIZE + (start_row * 8) as u64;
                
                let mut values: Vec<i64> = vec![0i64; row_count];
                // SAFETY: i64 has the same memory layout as [u8; 8] on little-endian systems
                // We read directly into the Vec's backing memory to avoid byte-by-byte parsing
                let bytes_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        values.as_mut_ptr() as *mut u8,
                        row_count * 8
                    )
                };
                mmap_cache.read_at(file, bytes_slice, index.data_offset + byte_offset)?;
                
                // Handle endianness: convert from LE if on BE system
                #[cfg(target_endian = "big")]
                for v in &mut values {
                    *v = i64::from_le(*v);
                }
                
                Ok(ColumnData::Int64(values))
            }
            ColumnType::Float64 | ColumnType::Float32 => {
                // Format: [count:u64][values:f64*]
                // Zero-copy optimization: read directly into f64 buffer
                let byte_offset = HEADER_SIZE + (start_row * 8) as u64;
                
                let mut values: Vec<f64> = vec![0f64; row_count];
                // SAFETY: f64 has the same memory layout as [u8; 8] on little-endian systems
                let bytes_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        values.as_mut_ptr() as *mut u8,
                        row_count * 8
                    )
                };
                mmap_cache.read_at(file, bytes_slice, index.data_offset + byte_offset)?;
                
                // Handle endianness: convert from LE if on BE system
                #[cfg(target_endian = "big")]
                for v in &mut values {
                    *v = f64::from_le_bytes(v.to_ne_bytes());
                }
                
                Ok(ColumnData::Float64(values))
            }
            ColumnType::Bool => {
                // Format: [len:u64][packed_bits...]
                let start_byte = start_row / 8;
                let end_byte = (start_row + row_count + 7) / 8;
                let byte_count = end_byte - start_byte;
                
                let mut packed = vec![0u8; byte_count];
                mmap_cache.read_at(file, &mut packed, index.data_offset + HEADER_SIZE + start_byte as u64)?;
                
                Ok(ColumnData::Bool { data: packed, len: row_count })
            }
            ColumnType::String | ColumnType::Binary => {
                // Variable-length type: need to read offsets first
                self.read_variable_column_range_mmap(mmap_cache, file, index, dtype, start_row, row_count)
            }
            ColumnType::StringDict => {
                // Native dictionary-encoded string reading
                self.read_string_dict_column_range_mmap(mmap_cache, file, index, start_row, row_count)
            }
            ColumnType::FixedList => {
                let mut dim_buf = [0u8; 4];
                let _ = mmap_cache.read_at(file, &mut dim_buf, index.data_offset + 8);
                let dim = u32::from_le_bytes(dim_buf);
                let dim_usize = dim as usize;
                let byte_len = row_count * dim_usize * 4;
                let byte_offset = index.data_offset + 12 + (start_row * dim_usize * 4) as u64;
                let mut data = vec![0u8; byte_len];
                if byte_len > 0 {
                    mmap_cache.read_at(file, &mut data, byte_offset)?;
                }
                Ok(ColumnData::FixedList { data, dim })
            }
            ColumnType::Float16List => {
                let mut dim_buf = [0u8; 4];
                let _ = mmap_cache.read_at(file, &mut dim_buf, index.data_offset + 8);
                let dim = u32::from_le_bytes(dim_buf);
                let dim_usize = dim as usize;
                let byte_len = row_count * dim_usize * 2;
                let byte_offset = index.data_offset + 12 + (start_row * dim_usize * 2) as u64;
                let mut data = vec![0u8; byte_len];
                if byte_len > 0 {
                    mmap_cache.read_at(file, &mut data, byte_offset)?;
                }
                Ok(ColumnData::Float16List { data, dim })
            }
            ColumnType::Null => {
                Ok(ColumnData::Int64(vec![0; row_count]))
            }
        }
    }

    fn read_variable_column_range_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        dtype: ColumnType,
        start_row: usize,
        row_count: usize,
    ) -> io::Result<ColumnData> {
        // Variable-length format: [count:u64][offsets:u32*][data_len:u64][data:bytes]
        // Read header to get total count
        let mut header_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut header_buf, index.data_offset)?;
        let total_count = u64::from_le_bytes(header_buf) as usize;
        
        if start_row >= total_count {
            return Ok(ColumnData::String { offsets: vec![0], data: Vec::new() });
        }
        
        let actual_count = row_count.min(total_count - start_row);
        
        // OPTIMIZATION: Read offsets directly into u32 Vec using bulk read
        let offset_start = 8 + start_row * 4; // skip count header
        let offset_count = actual_count + 1;
        let mut offsets: Vec<u32> = vec![0u32; offset_count];
        // SAFETY: u32 slice can be safely viewed as bytes for reading
        let offset_bytes = unsafe {
            std::slice::from_raw_parts_mut(offsets.as_mut_ptr() as *mut u8, offset_count * 4)
        };
        mmap_cache.read_at(file, offset_bytes, index.data_offset + offset_start as u64)?;
        
        // Handle endianness on big-endian systems
        #[cfg(target_endian = "big")]
        for off in &mut offsets {
            *off = u32::from_le(*off);
        }
        
        // Calculate data range
        let data_start = offsets[0];
        let data_end = offsets[actual_count];
        let data_len = (data_end - data_start) as usize;
        
        // Read data portion
        // Data starts after: 8 (count) + (total_count+1)*4 (offsets) + 8 (data_len)
        let data_offset_in_file = index.data_offset + 8 + (total_count + 1) as u64 * 4 + 8 + data_start as u64;
        let mut data = vec![0u8; data_len];
        if data_len > 0 {
            mmap_cache.read_at(file, &mut data, data_offset_in_file)?;
        }
        
        // Normalize offsets to start at 0 using SIMD-friendly subtraction
        let base = offsets[0];
        if base != 0 {
            for off in &mut offsets {
                *off -= base;
            }
        }
        
        match dtype {
            ColumnType::String => Ok(ColumnData::String { offsets, data }),
            ColumnType::Binary => Ok(ColumnData::Binary { offsets, data }),
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Not a variable type")),
        }
    }

    /// Read StringDict column with native format
    /// Format: [row_count:u64][dict_size:u64][indices:u32*row_count][dict_offsets:u32*dict_size][dict_data_len:u64][dict_data]
    /// OPTIMIZED: Uses bulk read for u32 arrays instead of per-element parsing
    fn read_string_dict_column_range_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        start_row: usize,
        row_count: usize,
    ) -> io::Result<ColumnData> {
        let base_offset = index.data_offset;
        
        // Read header: [row_count:u64][dict_size:u64]
        let mut header = [0u8; 16];
        mmap_cache.read_at(file, &mut header, base_offset)?;
        let total_rows = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
        let dict_size = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        
        if start_row >= total_rows {
            return Ok(ColumnData::StringDict {
                indices: Vec::new(),
                dict_offsets: vec![0],
                dict_data: Vec::new(),
            });
        }
        
        let actual_count = row_count.min(total_rows - start_row);
        
        // OPTIMIZATION: Read indices directly into Vec<u32>
        let indices_offset = base_offset + 16 + (start_row * 4) as u64;
        let mut indices: Vec<u32> = vec![0u32; actual_count];
        let indices_bytes = unsafe {
            std::slice::from_raw_parts_mut(indices.as_mut_ptr() as *mut u8, actual_count * 4)
        };
        mmap_cache.read_at(file, indices_bytes, indices_offset)?;
        
        #[cfg(target_endian = "big")]
        for idx in &mut indices {
            *idx = u32::from_le(*idx);
        }
        
        // OPTIMIZATION: Read dict_offsets directly into Vec<u32>
        let dict_offsets_offset = base_offset + 16 + (total_rows * 4) as u64;
        let mut dict_offsets: Vec<u32> = vec![0u32; dict_size];
        let dict_offsets_bytes = unsafe {
            std::slice::from_raw_parts_mut(dict_offsets.as_mut_ptr() as *mut u8, dict_size * 4)
        };
        mmap_cache.read_at(file, dict_offsets_bytes, dict_offsets_offset)?;
        
        #[cfg(target_endian = "big")]
        for off in &mut dict_offsets {
            *off = u32::from_le(*off);
        }
        
        // Read dict_data_len and dict_data
        let dict_data_len_offset = dict_offsets_offset + (dict_size * 4) as u64;
        let mut data_len_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut data_len_buf, dict_data_len_offset)?;
        let dict_data_len = u64::from_le_bytes(data_len_buf) as usize;
        
        let dict_data_offset = dict_data_len_offset + 8;
        let mut dict_data = vec![0u8; dict_data_len];
        if dict_data_len > 0 {
            mmap_cache.read_at(file, &mut dict_data, dict_data_offset)?;
        }
        
        Ok(ColumnData::StringDict {
            indices,
            dict_offsets,
            dict_data,
        })
    }

    /// Read StringDict column with scattered row indices
    /// OPTIMIZED: Only reads the specific indices needed, not all indices
    fn read_string_dict_column_scattered_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        row_indices: &[usize],
    ) -> io::Result<ColumnData> {
        if row_indices.is_empty() {
            return Ok(ColumnData::StringDict {
                indices: Vec::new(),
                dict_offsets: vec![0],
                dict_data: Vec::new(),
            });
        }
        
        let base_offset = index.data_offset;
        
        // Read header: [row_count:u64][dict_size:u64]
        let mut header = [0u8; 16];
        mmap_cache.read_at(file, &mut header, base_offset)?;
        let total_rows = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
        let dict_size = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        
        let all_indices_offset = base_offset + 16;
        let n = row_indices.len();
        
        // OPTIMIZED: Read only the specific indices we need instead of all indices
        // For small scattered reads, read individually; for dense reads, read a range
        let mut indices = Vec::with_capacity(n);
        
        if n <= 256 {
            // Small number of indices - read each one individually using thread-local buffer
            SCATTERED_READ_BUF.with(|buf| {
                let mut buf = buf.borrow_mut();
                buf.resize(4, 0);
                for &row_idx in row_indices {
                    if row_idx < total_rows {
                        mmap_cache.read_at(file, &mut buf[..4], all_indices_offset + (row_idx * 4) as u64)?;
                        indices.push(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]));
                    } else {
                        indices.push(0);
                    }
                }
                Ok::<(), io::Error>(())
            })?;
        } else {
            // For larger reads, find min/max and read that range
            let min_idx = *row_indices.iter().min().unwrap_or(&0);
            let max_idx = *row_indices.iter().max().unwrap_or(&0);
            let range_size = max_idx - min_idx + 1;
            
            // OPTIMIZATION: If range is reasonably dense, read whole range as Vec<u32>
            if range_size <= n * 4 && range_size <= total_rows {
                let mut range_values: Vec<u32> = vec![0u32; range_size];
                let range_bytes = unsafe {
                    std::slice::from_raw_parts_mut(range_values.as_mut_ptr() as *mut u8, range_size * 4)
                };
                mmap_cache.read_at(file, range_bytes, all_indices_offset + (min_idx * 4) as u64)?;
                
                #[cfg(target_endian = "big")]
                for v in &mut range_values {
                    *v = u32::from_le(*v);
                }
                
                for &row_idx in row_indices {
                    if row_idx < total_rows {
                        let local_idx = row_idx - min_idx;
                        indices.push(range_values[local_idx]);
                    } else {
                        indices.push(0);
                    }
                }
            } else {
                // Sparse - read individually using thread-local buffer
                SCATTERED_READ_BUF.with(|buf| {
                    let mut buf = buf.borrow_mut();
                    buf.resize(4, 0);
                    for &row_idx in row_indices {
                        if row_idx < total_rows {
                            mmap_cache.read_at(file, &mut buf[..4], all_indices_offset + (row_idx * 4) as u64)?;
                            indices.push(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]));
                        } else {
                            indices.push(0);
                        }
                    }
                    Ok::<(), io::Error>(())
                })?;
            }
        }
        
        // OPTIMIZATION: Read dict_offsets directly into Vec<u32>
        let dict_offsets_offset = base_offset + 16 + (total_rows * 4) as u64;
        let mut dict_offsets: Vec<u32> = vec![0u32; dict_size];
        let dict_offsets_bytes = unsafe {
            std::slice::from_raw_parts_mut(dict_offsets.as_mut_ptr() as *mut u8, dict_size * 4)
        };
        mmap_cache.read_at(file, dict_offsets_bytes, dict_offsets_offset)?;
        
        #[cfg(target_endian = "big")]
        for off in &mut dict_offsets {
            *off = u32::from_le(*off);
        }
        
        // Read dict_data_len and dict_data
        let dict_data_len_offset = dict_offsets_offset + (dict_size * 4) as u64;
        let mut data_len_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut data_len_buf, dict_data_len_offset)?;
        let dict_data_len = u64::from_le_bytes(data_len_buf) as usize;
        
        let dict_data_offset = dict_data_len_offset + 8;
        let mut dict_data = vec![0u8; dict_data_len];
        if dict_data_len > 0 {
            mmap_cache.read_at(file, &mut dict_data, dict_data_offset)?;
        }
        
        Ok(ColumnData::StringDict {
            indices,
            dict_offsets,
            dict_data,
        })
    }

    /// Optimized scattered read for variable-length columns (String/Binary) using mmap
    fn read_variable_column_scattered_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        dtype: ColumnType,
        row_indices: &[usize],
    ) -> io::Result<ColumnData> {
        if row_indices.is_empty() {
            return Ok(match dtype {
                ColumnType::String => ColumnData::String { offsets: vec![0], data: Vec::new() },
                _ => ColumnData::Binary { offsets: vec![0], data: Vec::new() },
            });
        }

        // Variable-length format: [count:u64][offsets:u32*(count+1)][data_len:u64][data:bytes]
        // Read header to get total count
        let mut header_buf = [0u8; 8];
        mmap_cache.read_at(file, &mut header_buf, index.data_offset)?;
        let total_count = u64::from_le_bytes(header_buf) as usize;

        // Read only the offsets we need (need idx and idx+1 for each row)
        // Collect unique offset indices needed
        let mut offset_indices: Vec<usize> = Vec::with_capacity(row_indices.len() * 2);
        for &idx in row_indices {
            if idx < total_count {
                offset_indices.push(idx);
                offset_indices.push(idx + 1);
            }
        }
        offset_indices.sort_unstable();
        offset_indices.dedup();

        if offset_indices.is_empty() {
            return Ok(match dtype {
                ColumnType::String => ColumnData::String { offsets: vec![0], data: Vec::new() },
                _ => ColumnData::Binary { offsets: vec![0], data: Vec::new() },
            });
        }

        // Read required offsets in batches (optimize for contiguous ranges)
        let mut offset_map: HashMap<usize, u32> = HashMap::with_capacity(offset_indices.len());
        let offset_base = index.data_offset + 8; // skip count header
        
        // For small number of indices, read individually
        // For larger sets, read a range that covers all needed offsets
        let min_idx = *offset_indices.first().unwrap();
        let max_idx = *offset_indices.last().unwrap();
        
        if max_idx - min_idx < offset_indices.len() * 4 {
            // Indices are sparse enough - read range
            let range_count = max_idx - min_idx + 1;
            let mut offset_buf = vec![0u8; range_count * 4];
            mmap_cache.read_at(file, &mut offset_buf, offset_base + (min_idx * 4) as u64)?;
            
            for &idx in &offset_indices {
                let local_idx = idx - min_idx;
                let off = u32::from_le_bytes(offset_buf[local_idx * 4..(local_idx + 1) * 4].try_into().unwrap());
                offset_map.insert(idx, off);
            }
        } else {
            // Very sparse - read individually
            let mut buf = [0u8; 4];
            for &idx in &offset_indices {
                mmap_cache.read_at(file, &mut buf, offset_base + (idx * 4) as u64)?;
                offset_map.insert(idx, u32::from_le_bytes(buf));
            }
        }

        // Calculate data offset base: skip count(8) + offsets((total_count+1)*4) + data_len(8)
        let data_base = index.data_offset + 8 + (total_count + 1) as u64 * 4 + 8;

        // Read data for each requested row and build result
        let mut result_offsets = vec![0u32];
        let mut result_data = Vec::new();

        for &idx in row_indices {
            if idx < total_count {
                let start = *offset_map.get(&idx).unwrap_or(&0);
                let end = *offset_map.get(&(idx + 1)).unwrap_or(&start);
                let len = (end - start) as usize;
                
                if len > 0 {
                    let mut chunk = vec![0u8; len];
                    mmap_cache.read_at(file, &mut chunk, data_base + start as u64)?;
                    result_data.extend_from_slice(&chunk);
                }
                result_offsets.push(result_data.len() as u32);
            } else {
                // Out of bounds - push empty
                result_offsets.push(result_data.len() as u32);
            }
        }

        match dtype {
            ColumnType::String => Ok(ColumnData::String { offsets: result_offsets, data: result_data }),
            ColumnType::Binary => Ok(ColumnData::Binary { offsets: result_offsets, data: result_data }),
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Not a variable type")),
        }
    }
    
    /// Optimized scattered read for numeric types using row-group based I/O
    /// Reads data in larger chunks (row-groups) to reduce number of I/O operations
    fn read_numeric_scattered_optimized<T: Copy + Default + 'static>(
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        row_indices: &[usize],
        header_size: u64,
    ) -> io::Result<Vec<T>> {
        if row_indices.is_empty() {
            return Ok(Vec::new());
        }
        
        let n = row_indices.len();
        let elem_size = std::mem::size_of::<T>();
        
        // For small numbers, simple sequential read without sorting is faster
        // Typical LIMIT queries (100-500 rows) benefit from avoiding sort overhead
        if n <= 256 {
            let mut values = Vec::with_capacity(n);
            SCATTERED_READ_BUF.with(|buf| {
                let mut buf = buf.borrow_mut();
                buf.resize(8, 0);
                for &idx in row_indices {
                    mmap_cache.read_at(file, &mut buf[..elem_size], index.data_offset + header_size + (idx * elem_size) as u64)?;
                    let val: T = unsafe { std::ptr::read(buf.as_ptr() as *const T) };
                    values.push(val);
                }
                Ok::<(), io::Error>(())
            })?;
            return Ok(values);
        }
        
        // ROW-GROUP BASED READING for larger scattered reads
        const ROW_GROUP_SIZE: usize = 8192;
        
        // Sort indices and track original positions
        let mut indexed: Vec<(usize, usize)> = row_indices.iter().enumerate().map(|(i, &idx)| (idx, i)).collect();
        indexed.sort_unstable_by_key(|&(idx, _)| idx);
        
        let mut result: Vec<T> = vec![T::default(); n];
        let mut i = 0;
        
        // Process by row-groups
        while i < indexed.len() {
            let first_idx = indexed[i].0;
            let group_start = (first_idx / ROW_GROUP_SIZE) * ROW_GROUP_SIZE;
            let group_end = group_start + ROW_GROUP_SIZE;
            
            // Find all indices within this row-group
            let mut group_indices = Vec::new();
            while i < indexed.len() && indexed[i].0 < group_end {
                group_indices.push(indexed[i]);
                i += 1;
            }
            
            // Decide read strategy based on density within group
            let indices_in_group = group_indices.len();
            let span = group_indices.last().unwrap().0 - group_indices.first().unwrap().0 + 1;
            
            // If indices are dense enough, read the span; otherwise read full group
            if indices_in_group * 4 >= span || span <= 256 {
                // Dense or small span: read just the span
                let read_start = group_indices.first().unwrap().0;
                let read_len = span;
                let mut buf: Vec<u8> = vec![0u8; read_len * elem_size];
                mmap_cache.read_at(file, &mut buf, index.data_offset + header_size + (read_start * elem_size) as u64)?;
                
                for (idx, orig_pos) in group_indices {
                    let offset = idx - read_start;
                    let val: T = unsafe { std::ptr::read(buf.as_ptr().add(offset * elem_size) as *const T) };
                    result[orig_pos] = val;
                }
            } else {
                // Sparse: read individual values (but they're sorted so still sequential-ish)
                let mut buf = [0u8; 8];
                for (idx, orig_pos) in group_indices {
                    mmap_cache.read_at(file, &mut buf[..elem_size], index.data_offset + header_size + (idx * elem_size) as u64)?;
                    let val: T = unsafe { std::ptr::read(buf.as_ptr() as *const T) };
                    result[orig_pos] = val;
                }
            }
        }
        
        Ok(result)
    }

    /// Optimized scattered read for fixed-size vector types (FixedList/Float16List).
    /// `elem_bytes` is bytes per element: 4 for f32 (FixedList), 2 for f16 (Float16List).
    /// Uses row-group batching for large reads instead of per-row I/O.
    fn read_fixed_scattered_optimized(
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        row_indices: &[usize],
        elem_bytes: usize,
    ) -> io::Result<(Vec<u8>, u32)> {
        // Read dim from header: [count:u64][dim:u32][data...]
        let mut dim_buf = [0u8; 4];
        mmap_cache.read_at(file, &mut dim_buf, index.data_offset + 8)?;
        let dim = u32::from_le_bytes(dim_buf);
        let dim_usize = dim as usize;
        let row_byte_len = dim_usize * elem_bytes;

        if row_indices.is_empty() || dim_usize == 0 {
            return Ok((Vec::new(), dim));
        }

        let n = row_indices.len();
        let data_base = index.data_offset + 12; // skip count(8) + dim(4)
        let mut result = vec![0u8; n * row_byte_len];

        if n <= 256 {
            // Small: sequential reads using thread-local buffer
            SCATTERED_READ_BUF.with(|buf| {
                let mut buf = buf.borrow_mut();
                buf.resize(row_byte_len, 0);
                for (out_i, &row) in row_indices.iter().enumerate() {
                    mmap_cache.read_at(file, &mut buf[..row_byte_len], data_base + (row * row_byte_len) as u64)?;
                    result[out_i * row_byte_len..(out_i + 1) * row_byte_len].copy_from_slice(&buf[..row_byte_len]);
                }
                Ok::<(), io::Error>(())
            })?;
        } else {
            // Large: sort indices, batch into contiguous span reads
            const ROW_GROUP_SIZE: usize = 8192;
            let mut indexed: Vec<(usize, usize)> = row_indices.iter().enumerate().map(|(i, &idx)| (idx, i)).collect();
            indexed.sort_unstable_by_key(|&(idx, _)| idx);

            let mut i = 0;
            while i < indexed.len() {
                let first_idx = indexed[i].0;
                let group_start = (first_idx / ROW_GROUP_SIZE) * ROW_GROUP_SIZE;
                let group_end = group_start + ROW_GROUP_SIZE;

                let mut group_indices = Vec::new();
                while i < indexed.len() && indexed[i].0 < group_end {
                    group_indices.push(indexed[i]);
                    i += 1;
                }

                let span = group_indices.last().unwrap().0 - group_indices.first().unwrap().0 + 1;
                let indices_in_group = group_indices.len();

                if indices_in_group * 4 >= span || span <= 256 {
                    // Dense or small span: read contiguous range
                    let read_start = group_indices.first().unwrap().0;
                    let buf_len = span * row_byte_len;
                    let mut buf = vec![0u8; buf_len];
                    mmap_cache.read_at(file, &mut buf, data_base + (read_start * row_byte_len) as u64)?;
                    for (idx, orig_pos) in group_indices {
                        let offset = (idx - read_start) * row_byte_len;
                        result[orig_pos * row_byte_len..(orig_pos + 1) * row_byte_len]
                            .copy_from_slice(&buf[offset..offset + row_byte_len]);
                    }
                } else {
                    // Sparse: individual reads
                    let mut buf = vec![0u8; row_byte_len];
                    for (idx, orig_pos) in group_indices {
                        mmap_cache.read_at(file, &mut buf, data_base + (idx * row_byte_len) as u64)?;
                        result[orig_pos * row_byte_len..(orig_pos + 1) * row_byte_len]
                            .copy_from_slice(&buf);
                    }
                }
            }
        }

        Ok((result, dim))
    }

    fn read_column_scattered_mmap(
        &self,
        mmap_cache: &mut MmapCache,
        file: &File,
        index: &ColumnIndexEntry,
        dtype: ColumnType,
        row_indices: &[usize],
        _total_rows: usize,
    ) -> io::Result<ColumnData> {
        // ColumnData format has an 8-byte count header
        const HEADER_SIZE: u64 = 8;
        
        match dtype {
            ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 |
            ColumnType::Timestamp | ColumnType::Date => {
                Self::read_numeric_scattered_optimized::<i64>(mmap_cache, file, index, row_indices, HEADER_SIZE)
                    .map(ColumnData::Int64)
            }
            ColumnType::Float64 | ColumnType::Float32 => {
                Self::read_numeric_scattered_optimized::<f64>(mmap_cache, file, index, row_indices, HEADER_SIZE)
                    .map(ColumnData::Float64)
            }
            ColumnType::String | ColumnType::Binary => {
                // Optimized scattered read for variable-length types
                self.read_variable_column_scattered_mmap(mmap_cache, file, index, dtype, row_indices)
            }
            ColumnType::Bool => {
                // Bool is stored as packed bits: [count:u64][packed_bits...]
                // Read the packed bits and extract specific indices
                let packed_len = (index.data_length as usize - 8 + 7) / 8;
                let mut packed = vec![0u8; packed_len.max(1)];
                if packed_len > 0 {
                    mmap_cache.read_at(file, &mut packed, index.data_offset + HEADER_SIZE)?;
                }
                
                // Extract the specific bits for requested indices
                let mut result_packed = vec![0u8; (row_indices.len() + 7) / 8];
                for (result_idx, &src_idx) in row_indices.iter().enumerate() {
                    let src_byte = src_idx / 8;
                    let src_bit = src_idx % 8;
                    let bit_value = if src_byte < packed.len() {
                        (packed[src_byte] >> src_bit) & 1
                    } else {
                        0
                    };
                    
                    let dst_byte = result_idx / 8;
                    let dst_bit = result_idx % 8;
                    if bit_value == 1 {
                        result_packed[dst_byte] |= 1 << dst_bit;
                    }
                }
                
                Ok(ColumnData::Bool { data: result_packed, len: row_indices.len() })
            }
            ColumnType::StringDict => {
                // Native dictionary-encoded string scattered read
                self.read_string_dict_column_scattered_mmap(mmap_cache, file, index, row_indices)
            }
            ColumnType::FixedList => {
                Self::read_fixed_scattered_optimized(mmap_cache, file, index, row_indices, 4)
                    .map(|(data, dim)| ColumnData::FixedList { data, dim })
            }
            ColumnType::Float16List => {
                Self::read_fixed_scattered_optimized(mmap_cache, file, index, row_indices, 2)
                    .map(|(data, dim)| ColumnData::Float16List { data, dim })
            }
            ColumnType::Null => {
                Ok(ColumnData::Int64(vec![0i64; row_indices.len()]))
            }
        }
    }

    /// Mmap-level string equality scan: find rows where col_name = target_value.
    /// Scans raw bytes without creating Arrow arrays. Returns global row indices of matches.
    pub fn scan_string_filter_mmap(&self, col_name: &str, target: &str, limit: Option<usize>) -> io::Result<Option<Vec<usize>>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        if !matches!(col_type, ColumnType::String | ColumnType::StringDict) { return Ok(None); }
        let col_count = schema.column_count();
        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for string scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;
        let target_bytes = target.as_bytes();
        let max_matches = limit.unwrap_or(usize::MAX);
        let mut matches: Vec<usize> = Vec::new();
        let mut global_row_offset: usize = 0;
        // Build the memmem Finder once (precomputes Boyer-Moore table from target_bytes)
        let memmem_finder = memchr::memmem::Finder::new(target_bytes);

        // ── PARALLEL FAST PATH ───────────────────────────────────────────────
        // For no-limit StringDict scans on uncompressed+RCIX data, scan all RGs
        // in parallel using Rayon to exploit all CPU cores simultaneously.
        if limit.is_none() && footer.row_groups.len() > 1
            && matches!(col_type, ColumnType::StringDict)
        {
            // Check whether every RG qualifies for the parallel fast path
            let all_fast = footer.row_groups.iter().enumerate().all(|(rg_i, rg_meta)| {
                let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                if rg_end > mmap_ref.len() { return false; }
                let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
                let compress_flag = rg_bytes.get(28).copied().unwrap_or(RG_COMPRESS_NONE);
                let enc_ver = rg_bytes.get(29).copied().unwrap_or(0);
                compress_flag == RG_COMPRESS_NONE && enc_ver >= 1
                    && footer.col_offsets.get(rg_i).map_or(false, |v| v.len() > col_idx)
            });

            if all_fast {
                // Cast pointer to usize (Send+Sync) — safe because mmap_guard keeps Mmap
                // alive for the entire scope and all parallel tasks are read-only.
                let mmap_ptr: usize = mmap_ref.as_ptr() as usize;
                let mmap_len: usize = mmap_ref.len();

                // Build per-RG scan descriptors upfront
                struct RgDesc {
                    rg_offset: usize, rg_data_size: usize, rg_rows: usize,
                    global_off: usize, col_rcix: usize, has_deletes: bool,
                }
                let mut rg_descs: Vec<RgDesc> = Vec::with_capacity(footer.row_groups.len());
                let target_len_i64 = target_bytes.len() as i64;
                let mut off = 0usize;
                for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
                    let global_off = off;
                    off += rg_meta.row_count as usize;

                    if let Some(zmaps) = footer.zone_maps.get(rg_i) {
                        if let Some(zm) = zmaps
                            .iter()
                            .find(|z| z.col_idx as usize == col_idx && !z.is_float)
                        {
                            if target_len_i64 < zm.min_bits || target_len_i64 > zm.max_bits {
                                continue;
                            }
                        }
                    }

                    rg_descs.push(RgDesc {
                        rg_offset: rg_meta.offset as usize,
                        rg_data_size: rg_meta.data_size as usize,
                        rg_rows: rg_meta.row_count as usize,
                        global_off,
                        col_rcix: footer.col_offsets[rg_i][col_idx] as usize,
                        has_deletes: rg_meta.deletion_count > 0,
                    });
                }

                let target_len = target_bytes.len();
                let all_rg_matches: Vec<Vec<usize>> = rg_descs.par_iter().map(|desc| {
                    let mmap = unsafe { std::slice::from_raw_parts(mmap_ptr as *const u8, mmap_len) };
                    let rg_end = desc.rg_offset + desc.rg_data_size;
                    if rg_end > mmap.len() || rg_end < desc.rg_offset + 32 { return vec![]; }
                    let rg_bytes = &mmap[desc.rg_offset..rg_end];
                    let body = &rg_bytes[32..];
                    let rg_rows = desc.rg_rows;
                    let null_bitmap_len = (rg_rows + 7) / 8;
                    let del_vec_len = null_bitmap_len;
                    let del_start = rg_rows * 8;

                    let col_off = desc.col_rcix;
                    if col_off + null_bitmap_len > body.len() { return vec![]; }
                    let col_bytes = &body[col_off + null_bitmap_len..];
                    if col_bytes.is_empty() { return vec![]; }
                    let encoding = col_bytes[0];
                    if encoding != COL_ENCODING_PLAIN { return vec![]; }
                    let data = &col_bytes[1..];
                    if data.len() < 16 { return vec![]; }

                    let row_count = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0;8])) as usize;
                    let dict_size = u64::from_le_bytes(data[8..16].try_into().unwrap_or([0;8])) as usize;
                    if dict_size == 0 { return vec![]; }
                    let indices_start = 16usize;
                    let indices_len = row_count * 4;
                    let dict_off_start = indices_start + indices_len;
                    let dict_offsets_len = dict_size * 4;
                    let dict_data_len_off = dict_off_start + dict_offsets_len;
                    if dict_data_len_off + 8 > data.len() { return vec![]; }
                    let dict_data_len = u64::from_le_bytes(
                        data[dict_data_len_off..dict_data_len_off+8].try_into().unwrap_or([0;8])
                    ) as usize;
                    let dict_data_start = dict_data_len_off + 8;

                    let dict_offsets_cow = bytes_as_u32_slice(&data[dict_off_start..], dict_size);
                    let dict_offsets: &[u32] = &dict_offsets_cow;
                    let indices_cow = bytes_as_u32_slice(&data[indices_start..], row_count);
                    let indices: &[u32] = &indices_cow;

                    // SIMD search target in raw dict data
                    let raw_end = (dict_data_start + dict_data_len).min(data.len());
                    let raw_dict = &data[dict_data_start..raw_end];
                    let finder = memchr::memmem::Finder::new(target_bytes);
                    let mut target_dict_idx: Option<u32> = None;
                    let mut search_from = 0usize;
                    while let Some(rel) = finder.find(&raw_dict[search_from..]) {
                        let abs = search_from + rel;
                        if let Ok(di) = dict_offsets.binary_search(&(abs as u32)) {
                            let de = if di + 1 < dict_size { dict_offsets[di+1] as usize } else { dict_data_len };
                            if de - abs == target_len {
                                target_dict_idx = Some((di + 1) as u32);
                                break;
                            }
                        }
                        search_from += rel + 1;
                        if search_from >= raw_dict.len() { break; }
                    }

                    let Some(tdi) = target_dict_idx else { return vec![]; };

                    let n = row_count.min(rg_rows);
                    let mut local: Vec<usize> = Vec::new();
                    if !desc.has_deletes {
                        for i in 0..n {
                            if indices[i] == tdi { local.push(desc.global_off + i); }
                        }
                    } else {
                        if del_start + del_vec_len > body.len() { return local; }
                        let del_bytes = &body[del_start..del_start + del_vec_len];
                        for i in 0..n {
                            if (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            if indices[i] == tdi { local.push(desc.global_off + i); }
                        }
                    }
                    local
                }).collect();

                // Merge results in RG order (already ordered since RGs are enumerated in order)
                matches = all_rg_matches.into_iter().flatten().collect();
                return Ok(Some(matches));
            }
        }
        // ── END PARALLEL FAST PATH ───────────────────────────────────────────

        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            if matches.len() >= max_matches { break; }
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { global_row_offset += rg_rows; continue; }
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Err(err_data("RG extends past EOF")); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
            let del_vec_len = (rg_rows + 7) / 8;
            let del_start = rg_rows * 8;
            if del_start + del_vec_len > body.len() { return Err(err_data("RG del vec truncated")); }
            let del_bytes = &body[del_start..del_start + del_vec_len];
            let has_deletes = rg_meta.deletion_count > 0;
            let null_bitmap_len = (rg_rows + 7) / 8;

            // Zone-map skip: if RG has a string-length zone map for this column,
            // skip the entire RG when target_len is outside [min_len, max_len].
            // This eliminates scanning 15/16 RGs for typical queries (e.g. 9-char target
            // in a dataset where only RG0 has 9-char strings).
            if let Some(zmaps) = footer.zone_maps.get(rg_i) {
                if let Some(zm) = zmaps.iter().find(|z| z.col_idx as usize == col_idx && !z.is_float) {
                    let tlen = target_bytes.len() as i64;
                    if tlen < zm.min_bits || tlen > zm.max_bits {
                        global_row_offset += rg_rows;
                        continue;
                    }
                }
            }

            // RCIX fast path: jump directly to target column without scanning all columns
            let rcix = if compress_flag == RG_COMPRESS_NONE && encoding_version >= 1 {
                footer.col_offsets.get(rg_i).filter(|v| v.len() > col_idx)
            } else { None };

            if let Some(rcix) = rcix {
                let col_off = rcix[col_idx] as usize;
                if col_off + null_bitmap_len > body.len() { global_row_offset += rg_rows; continue; }
                let null_bytes = &body[col_off..col_off + null_bitmap_len];
                let ct = schema.columns[col_idx].1;
                let col_bytes = &body[col_off + null_bitmap_len..];
                {
                    let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                    let encoding = if encoding_version >= 1 { col_bytes[0] } else { COL_ENCODING_PLAIN };
                    let data = &col_bytes[enc_offset..];

                    if encoding == COL_ENCODING_PLAIN && matches!(ct, ColumnType::String) && data.len() >= 8 {
                        let count = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
                        let all_offsets_len = (count + 1) * 4;
                        if 8 + all_offsets_len <= data.len() {
                            let data_len_off = 8 + all_offsets_len;
                            if data_len_off + 8 <= data.len() {
                                let data_start = data_len_off + 8;
                                // FAST: cast offset bytes to &[u32] slice (avoids 2M u32::from_le_bytes calls)
                                let offsets_cow = bytes_as_u32_slice(&data[8..], count + 1);
                                let offsets: &[u32] = &offsets_cow;
                                let target_len = target_bytes.len();
                                let n = count.min(rg_rows);
                                // FAST: memmem scan raw string data + binary search boundary check
                                // Replaces O(n) sequential scan with O(scan + k·log n) where k = rare hits
                                let data_len_val = u64::from_le_bytes(
                                    data[data_len_off..data_len_off+8].try_into().unwrap_or([0;8])
                                ) as usize;
                                let raw_end = (data_start + data_len_val).min(data.len());
                                let raw_str = &data[data_start..raw_end];
                                let mut search_from = 0usize;
                                while let Some(rel) = memmem_finder.find(&raw_str[search_from..]) {
                                    if matches.len() >= max_matches { break; }
                                    let abs = search_from + rel;
                                    // Binary search: find if abs is a valid string start offset
                                    if let Ok(di) = offsets[..count].binary_search(&(abs as u32)) {
                                        let end_off = offsets[di + 1] as usize;
                                        if end_off - abs == target_len && di < n {
                                            // Verify not deleted / null
                                            let skip = if has_deletes {
                                                (del_bytes[di / 8] >> (di % 8)) & 1 == 1
                                            } else { false };
                                            if !skip && (null_bytes[di / 8] >> (di % 8)) & 1 == 0 {
                                                matches.push(global_row_offset + di);
                                            }
                                        }
                                    }
                                    search_from += rel + 1;
                                    if search_from >= raw_str.len() { break; }
                                }
                            }
                        }
                    } else if encoding != COL_ENCODING_PLAIN && matches!(ct, ColumnType::String) {
                        // Non-PLAIN encoded String: decode then scan
                        let (col_data, _consumed) = if encoding_version >= 1 {
                            read_column_encoded(col_bytes, ct)?
                        } else {
                            ColumnData::from_bytes_typed(col_bytes, ct)?
                        };
                        if let ColumnData::String { offsets, data: str_data } = &col_data {
                            let count = offsets.len().saturating_sub(1);
                            for i in 0..count.min(rg_rows) {
                                if matches.len() >= max_matches { break; }
                                if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                let s = offsets[i] as usize;
                                let e = offsets[i + 1] as usize;
                                if e - s == target_bytes.len() && &str_data[s..e] == target_bytes {
                                    matches.push(global_row_offset + i);
                                }
                            }
                        }
                    } else if encoding == COL_ENCODING_PLAIN && matches!(ct, ColumnType::StringDict) && data.len() >= 16 {
                        // StringDict: find target in dict, then scan indices
                        let row_count = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
                        let dict_size = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
                        let indices_start = 16usize;
                        let indices_len = row_count * 4;
                        let dict_off_start = indices_start + indices_len;
                        let dict_offsets_len = dict_size * 4;
                        let dict_data_len_off = dict_off_start + dict_offsets_len;
                        if dict_data_len_off + 8 <= data.len() {
                            let dict_data_len = u64::from_le_bytes(data[dict_data_len_off..dict_data_len_off+8].try_into().unwrap()) as usize;
                            let dict_data_start = dict_data_len_off + 8;
                            // FAST: cast to &[u32] slices
                            let dict_offsets_cow = bytes_as_u32_slice(&data[dict_off_start..], dict_size);
                            let dict_offsets: &[u32] = &dict_offsets_cow;
                            let indices_cow = bytes_as_u32_slice(&data[indices_start..], row_count);
                            let indices: &[u32] = &indices_cow;
                            // Find target in dictionary using SIMD memmem + binary search boundary check
                            let target_len = target_bytes.len();
                            let mut target_dict_idx: Option<u32> = None;
                            let raw_end = (dict_data_start + dict_data_len).min(data.len());
                            let raw_dict = &data[dict_data_start..raw_end];
                            let mut search_from = 0usize;
                            while let Some(rel) = memmem_finder.find(&raw_dict[search_from..]) {
                                let abs = search_from + rel;
                                // Verify exact boundary: binary search for abs in dict_offsets
                                if let Ok(di) = dict_offsets.binary_search(&(abs as u32)) {
                                    let de = if di + 1 < dict_size { dict_offsets[di + 1] as usize } else { dict_data_len };
                                    if de - abs == target_len {
                                        target_dict_idx = Some((di + 1) as u32);
                                        break;
                                    }
                                }
                                search_from += rel + 1;
                                if search_from >= raw_dict.len() { break; }
                            }
                            if let Some(tdi) = target_dict_idx {
                                let n = row_count.min(rg_rows);
                                if !has_deletes {
                                    for i in 0..n {
                                        if matches.len() >= max_matches { break; }
                                        if indices[i] == tdi { matches.push(global_row_offset + i); }
                                    }
                                } else {
                                    for i in 0..n {
                                        if matches.len() >= max_matches { break; }
                                        if (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                        if indices[i] == tdi { matches.push(global_row_offset + i); }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                // Fallback: sequential pos scan for compressed or pre-RCIX row groups
                let mut pos = del_start + del_vec_len;
                for ci in 0..col_count {
                    if pos + null_bitmap_len > body.len() { break; }
                    let null_bytes = &body[pos..pos + null_bitmap_len];
                    pos += null_bitmap_len;
                    let ct = schema.columns[ci].1;
                    if ci == col_idx {
                        let col_bytes = &body[pos..];
                        let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                        let encoding = if encoding_version >= 1 && !col_bytes.is_empty() { col_bytes[0] } else { COL_ENCODING_PLAIN };
                        let data = if enc_offset <= col_bytes.len() { &col_bytes[enc_offset..] } else { &[] };
                        if encoding == COL_ENCODING_PLAIN && matches!(ct, ColumnType::String) && data.len() >= 8 {
                            let count = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
                            let all_offsets_len = (count + 1) * 4;
                            if 8 + all_offsets_len <= data.len() {
                                let data_len_off = 8 + all_offsets_len;
                                if data_len_off + 8 <= data.len() {
                                    let data_start = data_len_off + 8;
                                    let offsets = bytes_as_u32_slice(&data[8..], count + 1);
                                    let tlen = target_bytes.len();
                                    for i in 0..count.min(rg_rows) {
                                        if matches.len() >= max_matches { break; }
                                        if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                        let s = offsets[i] as usize; let e = offsets[i + 1] as usize;
                                        if e - s == tlen && data_start + e <= data.len() && &data[data_start + s..data_start + e] == target_bytes {
                                            matches.push(global_row_offset + i);
                                        }
                                    }
                                }
                            }
                        } else if encoding == COL_ENCODING_PLAIN && matches!(ct, ColumnType::StringDict) && data.len() >= 16 {
                            let row_count = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
                            let dict_size = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
                            let dict_off_start = 16 + row_count * 4;
                            let dict_data_len_off = dict_off_start + dict_size * 4;
                            if dict_data_len_off + 8 <= data.len() {
                                let dict_data_len = u64::from_le_bytes(data[dict_data_len_off..dict_data_len_off+8].try_into().unwrap()) as usize;
                                let dict_data_start = dict_data_len_off + 8;
                                let dict_offsets = bytes_as_u32_slice(&data[dict_off_start..], dict_size);
                                let indices = bytes_as_u32_slice(&data[16..], row_count);
                                let tlen = target_bytes.len();
                                let mut tdi: Option<u32> = None;
                                for di in 0..dict_size {
                                    let ds = dict_offsets[di] as usize;
                                    let de = if di + 1 < dict_size { dict_offsets[di + 1] as usize } else { dict_data_len };
                                    if de - ds == tlen && dict_data_start + de <= data.len() && &data[dict_data_start + ds..dict_data_start + de] == target_bytes {
                                        tdi = Some((di + 1) as u32); break;
                                    }
                                }
                                if let Some(tdi) = tdi {
                                    for i in 0..row_count.min(rg_rows) {
                                        if matches.len() >= max_matches { break; }
                                        if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                        if indices[i] == tdi { matches.push(global_row_offset + i); }
                                    }
                                }
                            }
                        }
                    }
                    let consumed = if encoding_version >= 1 {
                        skip_column_encoded(&body[pos..], ct)?
                    } else {
                        ColumnData::skip_bytes_typed(&body[pos..], ct)?
                    };
                    pos += consumed;
                }
            }
            global_row_offset += rg_rows;
        }
        drop(mmap_guard);
        drop(file_guard);
        Ok(Some(matches))
    }

    /// Mmap-level string IN scan: find rows where col_name IN (v1, v2, ...).
    /// Uses a single pass over the target column when all RGs are uncompressed+RCIX.
    /// Falls back to repeated equality scans for correctness on unsupported layouts.
    pub fn scan_string_in_mmap(
        &self,
        col_name: &str,
        values: &[String],
        limit: Option<usize>,
    ) -> io::Result<Option<Vec<usize>>> {
        use rayon::prelude::*;

        if values.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if values.len() == 1 {
            return self.scan_string_filter_mmap(col_name, &values[0], limit);
        }

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        let is_dict = matches!(col_type, ColumnType::StringDict);
        let is_string = matches!(col_type, ColumnType::String);
        if !is_dict && !is_string {
            return Ok(None);
        }

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for string IN scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        let all_fast = footer.row_groups.iter().enumerate().all(|(rg_i, rg_meta)| {
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() {
                return false;
            }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
            let compress_flag = rg_bytes.get(28).copied().unwrap_or(RG_COMPRESS_NONE);
            let enc_ver = rg_bytes.get(29).copied().unwrap_or(0);
            compress_flag == RG_COMPRESS_NONE
                && enc_ver >= 1
                && footer.col_offsets.get(rg_i).map_or(false, |v| v.len() > col_idx)
        });

        if !all_fast {
            drop(mmap_guard);
            drop(file_guard);
            let mut all_indices: Vec<usize> = Vec::new();
            for value in values {
                if let Some(mut idxs) = self.scan_string_filter_mmap(col_name, value, None)? {
                    all_indices.append(&mut idxs);
                }
            }
            all_indices.sort_unstable();
            all_indices.dedup();
            if let Some(lim) = limit {
                all_indices.truncate(lim);
            }
            return Ok(Some(all_indices));
        }

        let target_bytes: Vec<&[u8]> = values.iter().map(|s| s.as_bytes()).collect();
        let min_len = values.iter().map(|s| s.len()).min().unwrap_or(0) as i64;
        let max_len = values.iter().map(|s| s.len()).max().unwrap_or(0) as i64;
        let matches_any = |bytes: &[u8]| -> bool {
            target_bytes
                .iter()
                .any(|target| target.len() == bytes.len() && *target == bytes)
        };

        struct RgDesc {
            rg_idx: usize,
            rg_offset: usize,
            rg_data_size: usize,
            rg_rows: usize,
            global_off: usize,
            col_rcix: usize,
            has_deletes: bool,
        }

        let mut rg_descs: Vec<RgDesc> = Vec::with_capacity(footer.row_groups.len());
        let mut off = 0usize;
        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            rg_descs.push(RgDesc {
                rg_idx: rg_i,
                rg_offset: rg_meta.offset as usize,
                rg_data_size: rg_meta.data_size as usize,
                rg_rows: rg_meta.row_count as usize,
                global_off: off,
                col_rcix: footer.col_offsets[rg_i][col_idx] as usize,
                has_deletes: rg_meta.deletion_count > 0,
            });
            off += rg_meta.row_count as usize;
        }

        let scan_rg = |desc: &RgDesc, mmap: &[u8]| -> Vec<usize> {
            let rg_end = desc.rg_offset + desc.rg_data_size;
            if rg_end > mmap.len() || rg_end < desc.rg_offset + 32 {
                return vec![];
            }
            let body = &mmap[desc.rg_offset + 32..rg_end];
            let bitmap_len = (desc.rg_rows + 7) / 8;
            let del_start = desc.rg_rows * 8;
            if desc.col_rcix + bitmap_len > body.len() {
                return vec![];
            }

            let null_bytes = &body[desc.col_rcix..desc.col_rcix + bitmap_len];
            let col_bytes = &body[desc.col_rcix + bitmap_len..];
            if col_bytes.is_empty() || col_bytes[0] != COL_ENCODING_PLAIN {
                return vec![];
            }
            let payload = &col_bytes[1..];
            let mut local: Vec<usize> = Vec::new();

            if is_string {
                if payload.len() < 8 {
                    return local;
                }
                let count = u64::from_le_bytes(payload[0..8].try_into().unwrap_or([0; 8])) as usize;
                let offsets_len = (count + 1) * 4;
                let data_len_off = 8 + offsets_len;
                if data_len_off + 8 > payload.len() {
                    return local;
                }
                let data_len = u64::from_le_bytes(
                    payload[data_len_off..data_len_off + 8]
                        .try_into()
                        .unwrap_or([0; 8]),
                ) as usize;
                let data_start = data_len_off + 8;
                let data_end = (data_start + data_len).min(payload.len());
                if data_end < data_start {
                    return local;
                }
                let raw = &payload[data_start..data_end];
                let offsets_cow = bytes_as_u32_slice(&payload[8..], count + 1);
                let offsets: &[u32] = &offsets_cow;
                let n = count.min(desc.rg_rows);

                if !desc.has_deletes {
                    for i in 0..n {
                        if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                            continue;
                        }
                        let s = offsets[i] as usize;
                        let e = offsets[i + 1] as usize;
                        if e >= s && e <= raw.len() && matches_any(&raw[s..e]) {
                            local.push(desc.global_off + i);
                        }
                    }
                } else {
                    if del_start + bitmap_len > body.len() {
                        return local;
                    }
                    let del_bytes = &body[del_start..del_start + bitmap_len];
                    for i in 0..n {
                        if (del_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                            continue;
                        }
                        if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                            continue;
                        }
                        let s = offsets[i] as usize;
                        let e = offsets[i + 1] as usize;
                        if e >= s && e <= raw.len() && matches_any(&raw[s..e]) {
                            local.push(desc.global_off + i);
                        }
                    }
                }
            } else {
                if payload.len() < 16 {
                    return local;
                }
                let row_count = u64::from_le_bytes(payload[0..8].try_into().unwrap_or([0; 8])) as usize;
                let dict_size = u64::from_le_bytes(payload[8..16].try_into().unwrap_or([0; 8])) as usize;
                if dict_size == 0 {
                    return local;
                }
                let dict_off_start = 16 + row_count * 4;
                let dict_data_len_off = dict_off_start + dict_size * 4;
                if dict_data_len_off + 8 > payload.len() {
                    return local;
                }
                let dict_data_len = u64::from_le_bytes(
                    payload[dict_data_len_off..dict_data_len_off + 8]
                        .try_into()
                        .unwrap_or([0; 8]),
                ) as usize;
                let dict_data_start = dict_data_len_off + 8;
                let raw_end = (dict_data_start + dict_data_len).min(payload.len());
                let raw_dict = &payload[dict_data_start..raw_end];
                let dict_offsets_cow = bytes_as_u32_slice(&payload[dict_off_start..], dict_size);
                let dict_offsets: &[u32] = &dict_offsets_cow;
                let indices_cow = bytes_as_u32_slice(&payload[16..], row_count);
                let indices: &[u32] = &indices_cow;

                let mut match_flags = vec![false; dict_size + 1];
                for di in 0..dict_size {
                    let ds = dict_offsets[di] as usize;
                    let de = if di + 1 < dict_size {
                        dict_offsets[di + 1] as usize
                    } else {
                        dict_data_len
                    };
                    if ds <= de && de <= raw_dict.len() && matches_any(&raw_dict[ds..de]) {
                        match_flags[di + 1] = true;
                    }
                }

                let n = row_count.min(desc.rg_rows);
                if !desc.has_deletes {
                    for i in 0..n {
                        if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                            continue;
                        }
                        let idx = indices[i] as usize;
                        if idx < match_flags.len() && match_flags[idx] {
                            local.push(desc.global_off + i);
                        }
                    }
                } else {
                    if del_start + bitmap_len > body.len() {
                        return local;
                    }
                    let del_bytes = &body[del_start..del_start + bitmap_len];
                    for i in 0..n {
                        if (del_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                            continue;
                        }
                        if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                            continue;
                        }
                        let idx = indices[i] as usize;
                        if idx < match_flags.len() && match_flags[idx] {
                            local.push(desc.global_off + i);
                        }
                    }
                }
            }

            local
        };

        let max_matches = limit.unwrap_or(usize::MAX);
        let mut matches: Vec<usize> = if limit.is_none() && rg_descs.len() > 1 {
            let mmap_ptr = mmap_ref.as_ptr() as usize;
            let mmap_len = mmap_ref.len();
            rg_descs
                .par_iter()
                .filter_map(|desc| {
                    if let Some(zmaps) = footer.zone_maps.get(desc.rg_idx) {
                        if let Some(zm) = zmaps.iter().find(|z| z.col_idx as usize == col_idx && !z.is_float) {
                            if max_len < zm.min_bits || min_len > zm.max_bits {
                                return None;
                            }
                        }
                    }
                    let mmap = unsafe { std::slice::from_raw_parts(mmap_ptr as *const u8, mmap_len) };
                    Some(scan_rg(desc, mmap))
                })
                .flatten()
                .collect()
        } else {
            let mut out = Vec::new();
            for desc in &rg_descs {
                if out.len() >= max_matches {
                    break;
                }
                if let Some(zmaps) = footer.zone_maps.get(desc.rg_idx) {
                    if let Some(zm) = zmaps.iter().find(|z| z.col_idx as usize == col_idx && !z.is_float) {
                        if max_len < zm.min_bits || min_len > zm.max_bits {
                            continue;
                        }
                    }
                }
                let mut local = scan_rg(desc, mmap_ref);
                out.append(&mut local);
                if out.len() >= max_matches {
                    out.truncate(max_matches);
                    break;
                }
            }
            out
        };

        if let Some(lim) = limit {
            matches.truncate(lim);
        }
        Ok(Some(matches))
    }

    /// Mmap-level boolean column filter: find rows where col = target_value (true/false).
    /// Uses Rayon parallel scan across row groups for maximum performance.
    pub fn scan_bool_filter_mmap(
        &self,
        col_name: &str,
        target_value: bool,
        limit: Option<usize>,
    ) -> io::Result<Option<Vec<usize>>> {
        use rayon::prelude::*;
        
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        if !matches!(col_type, ColumnType::Bool) { return Ok(None); }
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for bool scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;
        
        let max_matches = limit.unwrap_or(usize::MAX);
        let target_bit: u8 = if target_value { 1 } else { 0 };
        
        // Use parallel scan for better performance on multi-core
        if footer.row_groups.len() > 1 {
            let mmap_ptr: usize = mmap_ref.as_ptr() as usize;
            let mmap_len: usize = mmap_ref.len();
            
            struct RgDesc {
                rg_offset: usize, rg_data_size: usize, rg_rows: usize,
                global_off: usize, col_rcix: usize, has_deletes: bool,
            }
            let mut rg_descs: Vec<RgDesc> = Vec::with_capacity(footer.row_groups.len());
            let mut off = 0usize;
            for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
                rg_descs.push(RgDesc {
                    rg_offset: rg_meta.offset as usize,
                    rg_data_size: rg_meta.data_size as usize,
                    rg_rows: rg_meta.row_count as usize,
                    global_off: off,
                    col_rcix: footer.col_offsets[rg_i][col_idx] as usize,
                    has_deletes: rg_meta.deletion_count > 0,
                });
                off += rg_meta.row_count as usize;
            }
            
            let all_rg_matches: Vec<Vec<usize>> = rg_descs.par_iter().map(|desc| {
                let mmap = unsafe { std::slice::from_raw_parts(mmap_ptr as *const u8, mmap_len) };
                let rg_end = desc.rg_offset + desc.rg_data_size;
                if rg_end > mmap.len() || rg_end < desc.rg_offset + 32 { return vec![]; }
                let rg_bytes = &mmap[desc.rg_offset..rg_end];
                let body = &rg_bytes[32..];
                let rg_rows = desc.rg_rows;
                let null_bitmap_len = (rg_rows + 7) / 8;
                let del_vec_len = null_bitmap_len;
                
                let col_off = desc.col_rcix;
                if col_off + null_bitmap_len > body.len() { return vec![]; }
                let bool_data = &body[col_off..col_off + null_bitmap_len];
                
                let mut matches = Vec::new();
                for i in 0..rg_rows {
                    if matches.len() >= max_matches { break; }
                    if desc.has_deletes {
                        let b = i / 8; let bit = i % 8;
                        if b < del_vec_len && (body[b] >> bit) & 1 != 0 { continue; }
                    }
                    let bool_val = (bool_data[i/8] >> (i%8)) & 1;
                    if bool_val == target_bit {
                        matches.push(desc.global_off + i);
                    }
                }
                matches
            }).collect();
            
            let mut result: Vec<usize> = all_rg_matches.into_iter().flatten().collect();
            if let Some(lim) = limit {
                result.truncate(lim);
            }
            return Ok(Some(result));
        }
        
        // Fallback for single RG
        let mut matches: Vec<usize> = Vec::new();
        let mut global_row_offset: usize = 0;
        
        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            if matches.len() >= max_matches { break; }
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { global_row_offset += rg_rows; continue; }
            
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { continue; }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            let body = &rg_bytes[32..];
            let null_bitmap_len = (rg_rows + 7) / 8;
            let del_vec_len = null_bitmap_len;
            
            let col_off = if rg_i < footer.col_offsets.len() && col_idx < footer.col_offsets[rg_i].len() {
                footer.col_offsets[rg_i][col_idx] as usize
            } else {
                continue;
            };
            
            if col_off + null_bitmap_len > body.len() { global_row_offset += rg_rows; continue; }
            let bool_data = &body[col_off..col_off + null_bitmap_len];
            
            let has_deletes = rg_meta.deletion_count > 0;
            for i in 0..rg_rows {
                if matches.len() >= max_matches { break; }
                if has_deletes && (body[i/8] >> (i%8)) & 1 != 0 { continue; }
                let bool_val = (bool_data[i/8] >> (i%8)) & 1;
                if bool_val == target_bit {
                    matches.push(global_row_offset + i);
                }
            }
            global_row_offset += rg_rows;
        }
        
        Ok(Some(matches))
    }

    /// Mmap-level LIKE pattern scan: find rows where `col_name LIKE pattern`.
    /// Handles prefix ('abc%'), suffix ('%abc'), contains ('%abc%'), any ('%'), and
    /// complex patterns via compiled regex. Delegates exact (no-wildcard) patterns to
    /// scan_string_filter_mmap. Uses Rayon parallel scan across row groups.
    pub fn scan_like_filter_mmap(
        &self,
        col_name: &str,
        pattern: &str,
        limit: Option<usize>,
    ) -> io::Result<Option<Vec<usize>>> {
        use rayon::prelude::*;

        // No wildcards → exact equality: delegate to the optimised equality scanner
        if !pattern.contains('%') && !pattern.contains('_') {
            return self.scan_string_filter_mmap(col_name, pattern, limit);
        }
        let like_kind = match classify_like_pattern(pattern) {
            Some(k) => k,
            None => return Ok(None),
        };

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        if !matches!(col_type, ColumnType::String | ColumnType::StringDict) {
            return Ok(None);
        }
        let is_dict = matches!(col_type, ColumnType::StringDict);

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for LIKE scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        let max_matches = limit.unwrap_or(usize::MAX);
        let mut matches: Vec<usize> = Vec::new();
        let mut global_row_offset: usize = 0;

        // ── PARALLEL FAST PATH ───────────────────────────────────────────────
        // Requires: no limit, multiple RGs, all uncompressed+RCIX
        if limit.is_none() && footer.row_groups.len() > 1 {
            let all_fast = footer.row_groups.iter().enumerate().all(|(rg_i, rg_meta)| {
                let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                if rg_end > mmap_ref.len() { return false; }
                let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
                let compress_flag = rg_bytes.get(28).copied().unwrap_or(RG_COMPRESS_NONE);
                let enc_ver = rg_bytes.get(29).copied().unwrap_or(0);
                compress_flag == RG_COMPRESS_NONE && enc_ver >= 1
                    && footer.col_offsets.get(rg_i).map_or(false, |v| v.len() > col_idx)
            });

            if all_fast {
                let mmap_ptr: usize = mmap_ref.as_ptr() as usize;
                let mmap_len: usize = mmap_ref.len();

                struct RgDesc {
                    rg_offset: usize, rg_data_size: usize, rg_rows: usize,
                    global_off: usize, col_rcix: usize, has_deletes: bool,
                }
                let mut rg_descs: Vec<RgDesc> = Vec::with_capacity(footer.row_groups.len());
                let mut off = 0usize;
                for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
                    rg_descs.push(RgDesc {
                        rg_offset:    rg_meta.offset as usize,
                        rg_data_size: rg_meta.data_size as usize,
                        rg_rows:      rg_meta.row_count as usize,
                        global_off:   off,
                        col_rcix:     footer.col_offsets[rg_i][col_idx] as usize,
                        has_deletes:  rg_meta.deletion_count > 0,
                    });
                    off += rg_meta.row_count as usize;
                }

                let like_kind_ref = &like_kind;
                let all_rg_matches: Vec<Vec<usize>> = rg_descs.par_iter().map(|desc| {
                    let mmap = unsafe {
                        std::slice::from_raw_parts(mmap_ptr as *const u8, mmap_len)
                    };
                    let rg_end = desc.rg_offset + desc.rg_data_size;
                    if rg_end > mmap.len() || rg_end < desc.rg_offset + 32 { return vec![]; }
                    let body = &mmap[desc.rg_offset + 32..rg_end];
                    let rg_rows = desc.rg_rows;
                    let bitmap_len = (rg_rows + 7) / 8;
                    let del_start  = rg_rows * 8;

                    let col_off = desc.col_rcix;
                    if col_off + bitmap_len > body.len() { return vec![]; }
                    let null_bytes = &body[col_off..col_off + bitmap_len];
                    let col_bytes  = &body[col_off + bitmap_len..];
                    if col_bytes.is_empty() || col_bytes[0] != COL_ENCODING_PLAIN { return vec![]; }
                    let data = &col_bytes[1..];

                    let del_bytes_opt: Option<&[u8]> = if desc.has_deletes
                        && del_start + bitmap_len <= body.len()
                    {
                        Some(&body[del_start..del_start + bitmap_len])
                    } else { None };

                    let mut local: Vec<usize> = Vec::new();

                    if !is_dict {
                        // ── String PLAIN ──────────────────────────────────────
                        if data.len() < 8 { return local; }
                        let count = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0;8])) as usize;
                        let data_len_off = 8 + (count + 1) * 4;
                        if data_len_off + 8 > data.len() { return local; }
                        let data_str_len = u64::from_le_bytes(
                            data[data_len_off..data_len_off+8].try_into().unwrap_or([0;8])
                        ) as usize;
                        let data_start = data_len_off + 8;
                        let data_end = (data_start + data_str_len).min(data.len());
                        if data_end < data_start { return local; }
                        let data_region = &data[data_start..data_end];
                        let offsets_cow = bytes_as_u32_slice(&data[8..], count + 1);
                        let offsets: &[u32] = &offsets_cow;
                        let n = count.min(rg_rows);

                        // Fast path: no deletions and no nulls - skip bitmap checks entirely
                        let no_deletes = del_bytes_opt.is_none();
                        let no_nulls = null_bytes.iter().all(|&b| b == 0);

                        if no_deletes && no_nulls {
                            // Ultra-fast path: no bitmap checks needed
                            for i in 0..n {
                                let s = offsets[i] as usize;
                                let e = offsets[i+1] as usize;
                                if e > data_region.len() { continue; }
                                if like_matches_bytes(like_kind_ref, &data_region[s..e]) {
                                    local.push(desc.global_off + i);
                                }
                            }
                        } else {
                            // Standard path with deletion/null checks
                            for i in 0..n {
                                if let Some(db) = del_bytes_opt {
                                    if (db[i/8] >> (i%8)) & 1 == 1 { continue; }
                                }
                                if (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                let s = offsets[i] as usize;
                                let e = offsets[i+1] as usize;
                                if e > data_region.len() { continue; }
                                if like_matches_bytes(like_kind_ref, &data_region[s..e]) {
                                    local.push(desc.global_off + i);
                                }
                            }
                        }
                    } else {
                        // ── StringDict PLAIN ──────────────────────────────────
                        if data.len() < 16 { return local; }
                        let row_count = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0;8])) as usize;
                        let dict_size = u64::from_le_bytes(data[8..16].try_into().unwrap_or([0;8])) as usize;
                        if dict_size == 0 { return local; }
                        let dict_off_start    = 16 + row_count * 4;
                        let dict_data_len_off = dict_off_start + dict_size * 4;
                        if dict_data_len_off + 8 > data.len() { return local; }
                        let dict_data_len = u64::from_le_bytes(
                            data[dict_data_len_off..dict_data_len_off+8].try_into().unwrap_or([0;8])
                        ) as usize;
                        let dict_data_start = dict_data_len_off + 8;
                        let raw_end  = (dict_data_start + dict_data_len).min(data.len());
                        let raw_dict = &data[dict_data_start..raw_end];
                        let dict_offsets_cow = bytes_as_u32_slice(&data[dict_off_start..], dict_size);
                        let dict_offsets: &[u32] = &dict_offsets_cow;
                        let indices_cow = bytes_as_u32_slice(&data[16..], row_count);
                        let indices: &[u32] = &indices_cow;

                        // Pre-compute per-dict-entry match flags (O(dict_size), very fast)
                        let mut match_flags = vec![false; dict_size + 1];
                        for di in 0..dict_size {
                            let ds = dict_offsets[di] as usize;
                            let de = if di + 1 < dict_size { dict_offsets[di+1] as usize } else { dict_data_len };
                            if ds <= de && de <= raw_dict.len() {
                                match_flags[di + 1] = like_matches_bytes(like_kind_ref, &raw_dict[ds..de]);
                            }
                        }

                        let n = row_count.min(rg_rows);
                        for i in 0..n {
                            if let Some(db) = del_bytes_opt {
                                if (db[i/8] >> (i%8)) & 1 == 1 { continue; }
                            }
                            let idx = indices[i] as usize;
                            if idx < match_flags.len() && match_flags[idx] {
                                local.push(desc.global_off + i);
                            }
                        }
                    }
                    local
                }).collect();

                matches = all_rg_matches.into_iter().flatten().collect();
                return Ok(Some(matches));
            }
        }
        // ── END PARALLEL FAST PATH ───────────────────────────────────────────

        // SEQUENTIAL PATH: single-RG, limited scans, or compressed files
        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            if matches.len() >= max_matches { break; }
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { global_row_offset += rg_rows; continue; }
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Err(err_data("LIKE scan: RG extends past EOF")); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
            let compress_flag    = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
            let bitmap_len  = (rg_rows + 7) / 8;
            let del_start   = rg_rows * 8;
            let del_bytes: &[u8] = if del_start + bitmap_len <= body.len() {
                &body[del_start..del_start + bitmap_len]
            } else { &[] };
            let has_deletes = rg_meta.deletion_count > 0;

            // RCIX: jump directly to the target column
            let rcix = if compress_flag == RG_COMPRESS_NONE && encoding_version >= 1 {
                footer.col_offsets.get(rg_i).filter(|v| v.len() > col_idx)
            } else { None };

            if let Some(rcix) = rcix {
                let col_off = rcix[col_idx] as usize;
                if col_off + bitmap_len > body.len() { global_row_offset += rg_rows; continue; }
                let null_bytes = &body[col_off..col_off + bitmap_len];
                let col_bytes  = &body[col_off + bitmap_len..];
                if col_bytes.is_empty() { global_row_offset += rg_rows; continue; }
                if col_bytes[0] == COL_ENCODING_PLAIN {
                    let data = &col_bytes[1..];
                    if !is_dict {
                        // String PLAIN
                        if data.len() >= 8 {
                            let count = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0;8])) as usize;
                            let data_len_off = 8 + (count + 1) * 4;
                            if data_len_off + 8 <= data.len() {
                                let data_str_len = u64::from_le_bytes(
                                    data[data_len_off..data_len_off+8].try_into().unwrap_or([0;8])
                                ) as usize;
                                let data_start  = data_len_off + 8;
                                let data_end    = (data_start + data_str_len).min(data.len());
                                let data_region = &data[data_start..data_end];
                                let offsets_cow = bytes_as_u32_slice(&data[8..], count + 1);
                                let offsets: &[u32] = &offsets_cow;
                                let n = count.min(rg_rows);
                                for i in 0..n {
                                    if matches.len() >= max_matches { break; }
                                    if has_deletes && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                    if (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                    let s = offsets[i] as usize;
                                    let e = offsets[i+1] as usize;
                                    if e <= data_region.len() && like_matches_bytes(&like_kind, &data_region[s..e]) {
                                        matches.push(global_row_offset + i);
                                    }
                                }
                            }
                        }
                    } else {
                        // StringDict PLAIN
                        if data.len() >= 16 {
                            let row_count = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0;8])) as usize;
                            let dict_size = u64::from_le_bytes(data[8..16].try_into().unwrap_or([0;8])) as usize;
                            if dict_size > 0 {
                                let dict_off_start    = 16 + row_count * 4;
                                let dict_data_len_off = dict_off_start + dict_size * 4;
                                if dict_data_len_off + 8 <= data.len() {
                                    let dict_data_len = u64::from_le_bytes(
                                        data[dict_data_len_off..dict_data_len_off+8].try_into().unwrap_or([0;8])
                                    ) as usize;
                                    let dict_data_start = dict_data_len_off + 8;
                                    let raw_end  = (dict_data_start + dict_data_len).min(data.len());
                                    let raw_dict = &data[dict_data_start..raw_end];
                                    let dict_offsets = bytes_as_u32_slice(&data[dict_off_start..], dict_size);
                                    let indices      = bytes_as_u32_slice(&data[16..], row_count);
                                    let mut match_flags = vec![false; dict_size + 1];
                                    for di in 0..dict_size {
                                        let ds = dict_offsets[di] as usize;
                                        let de = if di + 1 < dict_size { dict_offsets[di+1] as usize } else { dict_data_len };
                                        if ds <= de && de <= raw_dict.len() {
                                            match_flags[di + 1] = like_matches_bytes(&like_kind, &raw_dict[ds..de]);
                                        }
                                    }
                                    let n = row_count.min(rg_rows);
                                    for i in 0..n {
                                        if matches.len() >= max_matches { break; }
                                        if has_deletes && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                        let idx = indices[i] as usize;
                                        if idx < match_flags.len() && match_flags[idx] {
                                            matches.push(global_row_offset + i);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Non-RCIX or non-PLAIN: skip (caller falls back to executor)
            global_row_offset += rg_rows;
        }
        drop(mmap_guard);
        drop(file_guard);
        Ok(Some(matches))
    }

    /// Mmap-level numeric range scan: find rows where col_name BETWEEN low AND high.
    /// Returns global row indices of matches.
    pub fn scan_numeric_range_mmap(&self, col_name: &str, low: f64, high: f64, limit: Option<usize>) -> io::Result<Option<Vec<usize>>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        let is_bool = matches!(col_type, ColumnType::Bool);
        let is_int = matches!(col_type, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
            ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
            ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date);
        let is_float = matches!(col_type, ColumnType::Float64 | ColumnType::Float32);
        if !is_bool && !is_int && !is_float { return Ok(None); }
        
        // For boolean columns, convert to integer range: false=0, true=1
        let (low, high) = if is_bool {
            let bool_low = if low > 0.5 { 1 } else { 0 };
            let bool_high = if high > 0.5 { 1 } else { 0 };
            (bool_low as f64, bool_high as f64)
        } else {
            (low, high)
        };
        
        let col_count = schema.column_count();
        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for range scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;
        let max_matches = limit.unwrap_or(usize::MAX);
        let mut matches: Vec<usize> = Vec::new();
        let mut global_row_offset: usize = 0;

        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            if matches.len() >= max_matches { break; }
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { global_row_offset += rg_rows; continue; }

            // Zone map pruning: skip RG if filter range can't overlap
            if rg_i < footer.zone_maps.len() {
                if let Some(zm) = footer.zone_maps[rg_i].iter().find(|z| z.col_idx as usize == col_idx) {
                    let skip = if zm.is_float {
                        !zm.may_overlap_float_range(low, high)
                    } else {
                        !zm.may_overlap_int_range(low.ceil() as i64, high.floor() as i64)
                    };
                    if skip { global_row_offset += rg_rows; continue; }
                }
            }

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Err(err_data("RG extends past EOF")); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
            let del_vec_len = (rg_rows + 7) / 8;
            let id_section = rg_rows * 8;
            if id_section + del_vec_len > body.len() { return Err(err_data("RG del vec truncated")); }
            let del_bytes = &body[id_section..id_section + del_vec_len];
            let has_deletes = rg_meta.deletion_count > 0;
            let null_bitmap_len = (rg_rows + 7) / 8;

            // RCIX fast path: jump directly to target column offset
            let rcix_available = rg_i < footer.col_offsets.len()
                && col_idx < footer.col_offsets[rg_i].len()
                && compress_flag == RG_COMPRESS_NONE;
            if rcix_available {
                let col_body_off = footer.col_offsets[rg_i][col_idx] as usize;
                if col_body_off + null_bitmap_len > body.len() { global_row_offset += rg_rows; continue; }
                let null_bytes = &body[col_body_off..col_body_off + null_bitmap_len];
                let data_start = col_body_off + null_bitmap_len;
                let col_bytes = &body[data_start..];
                let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                let encoding = if encoding_version >= 1 && !col_bytes.is_empty() { col_bytes[0] } else { COL_ENCODING_PLAIN };
                
                // Handle boolean columns (packed bits)
                if is_bool {
                    let bool_data_len = (rg_rows + 7) / 8;
                    if col_bytes.len() >= enc_offset + bool_data_len {
                        let bool_data = &col_bytes[enc_offset..enc_offset + bool_data_len];
                        let low_i = low.ceil() as i64;
                        let high_i = high.floor() as i64;
                        for i in 0..rg_rows {
                            if matches.len() >= max_matches { break; }
                            if has_deletes && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            if (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            let bool_val = (bool_data[i/8] >> (i%8)) & 1;
                            if bool_val as i64 >= low_i && bool_val as i64 <= high_i {
                                matches.push(global_row_offset + i);
                            }
                        }
                        global_row_offset += rg_rows;
                        continue;
                    }
                }
                
                if encoding == COL_ENCODING_PLAIN && col_bytes.len() > enc_offset + 8 {
                    let payload = &col_bytes[enc_offset..];
                    let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                    let n = count.min(rg_rows).min((payload.len() - 8) / 8);
                    let no_nulls = !null_bytes.iter().any(|&b| b != 0);
                    let unlimited = max_matches == usize::MAX;
                    if is_int {
                        let low_i = low.ceil() as i64;
                        let high_i = high.floor() as i64;
                        let vals = bytes_as_i64_slice(&payload[8..], n);
                        if !has_deletes && no_nulls && unlimited {
                            for (i, &v) in vals.iter().enumerate() {
                                if v >= low_i && v <= high_i { matches.push(global_row_offset + i); }
                            }
                        } else {
                            for i in 0..n {
                                if matches.len() >= max_matches { break; }
                                if has_deletes && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                if !no_nulls && (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                if vals[i] >= low_i && vals[i] <= high_i { matches.push(global_row_offset + i); }
                            }
                        }
                    } else {
                        let vals = bytes_as_f64_slice(&payload[8..], n);
                        if !has_deletes && no_nulls && unlimited {
                            for (i, &v) in vals.iter().enumerate() {
                                if v >= low && v <= high { matches.push(global_row_offset + i); }
                            }
                        } else {
                            for i in 0..n {
                                if matches.len() >= max_matches { break; }
                                if has_deletes && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                if !no_nulls && (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                if vals[i] >= low && vals[i] <= high { matches.push(global_row_offset + i); }
                            }
                        }
                    }
                    global_row_offset += rg_rows;
                    continue;
                } else if encoding == COL_ENCODING_BITPACK
                    && is_int
                    && col_bytes.len() >= enc_offset + 17
                {
                    let payload = &col_bytes[enc_offset..];
                    let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                    let bit_width = payload[8] as usize;
                    if bit_width < 64 {
                        let min_val = i64::from_le_bytes(payload[9..17].try_into().unwrap());
                        let packed_bytes = (count * bit_width + 7) / 8;
                        if payload.len() >= 17 + packed_bytes {
                            let packed = &payload[17..17 + packed_bytes];
                            let n = count.min(rg_rows);
                            let low_i = low.ceil() as i64;
                            let high_i = high.floor() as i64;
                            let no_nulls = !null_bytes.iter().any(|&b| b != 0);
                            for i in 0..n {
                                if matches.len() >= max_matches {
                                    break;
                                }
                                if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                                    continue;
                                }
                                if !no_nulls && (null_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                                    continue;
                                }
                                if let Some(v) = bitpack_value_at(packed, bit_width, min_val, i) {
                                    if v >= low_i && v <= high_i {
                                        matches.push(global_row_offset + i);
                                    }
                                }
                            }
                            global_row_offset += rg_rows;
                            continue;
                        }
                    }
                }
            }

            // Fallback: sequential column scan (compressed or no RCIX)
            let mut pos = id_section + del_vec_len;
            for ci in 0..col_count {
                if pos + null_bitmap_len > body.len() { break; }
                let null_bytes = &body[pos..pos + null_bitmap_len];
                pos += null_bitmap_len;
                let ct = schema.columns[ci].1;

                if ci == col_idx {
                    let col_bytes = &body[pos..];
                    let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                    let encoding = if encoding_version >= 1 { col_bytes[0] } else { COL_ENCODING_PLAIN };
                    let data_slice = &col_bytes[enc_offset..];

                    if encoding == COL_ENCODING_PLAIN && data_slice.len() >= 8 {
                        let count = u64::from_le_bytes(data_slice[0..8].try_into().unwrap()) as usize;
                        let values_start = 8usize;
                        let n = count.min(rg_rows);
                        if is_int {
                            let low_i = low.ceil() as i64;
                            let high_i = high.floor() as i64;
                            let nn = n.min((data_slice.len() - values_start) / 8);
                            let vals = bytes_as_i64_slice(&data_slice[values_start..], nn);
                            if !has_deletes {
                                for i in 0..vals.len() {
                                    if matches.len() >= max_matches { break; }
                                    if vals[i] >= low_i && vals[i] <= high_i { matches.push(global_row_offset + i); }
                                }
                            } else {
                                for i in 0..vals.len() {
                                    if matches.len() >= max_matches { break; }
                                    if (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if vals[i] >= low_i && vals[i] <= high_i { matches.push(global_row_offset + i); }
                                }
                            }
                        } else {
                            let nn = n.min((data_slice.len() - values_start) / 8);
                            let vals = bytes_as_f64_slice(&data_slice[values_start..], nn);
                            if !has_deletes {
                                for i in 0..vals.len() {
                                    if matches.len() >= max_matches { break; }
                                    if vals[i] >= low && vals[i] <= high { matches.push(global_row_offset + i); }
                                }
                            } else {
                                for i in 0..vals.len() {
                                    if matches.len() >= max_matches { break; }
                                    if (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if vals[i] >= low && vals[i] <= high { matches.push(global_row_offset + i); }
                                }
                            }
                        }
                    } else {
                        // Non-PLAIN encoding (RLE, BITPACK, etc.): decode then scan
                        let (col_data, _consumed) = if encoding_version >= 1 {
                            read_column_encoded(col_bytes, ct)?
                        } else {
                            ColumnData::from_bytes_typed(col_bytes, ct)?
                        };
                        match &col_data {
                            ColumnData::Int64(vals) => {
                                let low_i = low.ceil() as i64;
                                let high_i = high.floor() as i64;
                                for i in 0..vals.len().min(rg_rows) {
                                    if matches.len() >= max_matches { break; }
                                    if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if vals[i] >= low_i && vals[i] <= high_i { matches.push(global_row_offset + i); }
                                }
                            }
                            ColumnData::Float64(vals) => {
                                for i in 0..vals.len().min(rg_rows) {
                                    if matches.len() >= max_matches { break; }
                                    if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if vals[i] >= low && vals[i] <= high { matches.push(global_row_offset + i); }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                let consumed = if encoding_version >= 1 {
                    skip_column_encoded(&body[pos..], ct)?
                } else {
                    ColumnData::skip_bytes_typed(&body[pos..], ct)?
                };
                pos += consumed;
            }
            global_row_offset += rg_rows;
        }
        drop(mmap_guard);
        drop(file_guard);
        Ok(Some(matches))
    }

    /// Mmap-level numeric IN scan: find rows where col_name IN (v1, v2, ...).
    /// Single pass over column data with HashSet lookup. Returns global row indices.
    pub fn scan_numeric_in_mmap(&self, col_name: &str, values: &[i64], limit: Option<usize>) -> io::Result<Option<Vec<usize>>> {
        if values.is_empty() { return Ok(Some(Vec::new())); }
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        let is_int = matches!(col_type, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
            ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
            ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date);
        let is_float = matches!(col_type, ColumnType::Float64 | ColumnType::Float32);
        if !is_int && !is_float { return Ok(None); }

        let mut small_values: Vec<i64> = values.to_vec();
        small_values.sort_unstable();
        small_values.dedup();
        let use_small_values = small_values.len() <= 16;
        let value_set: ahash::AHashSet<i64> = if use_small_values {
            ahash::AHashSet::new()
        } else {
            small_values.iter().copied().collect()
        };
        let matches_value = |v: i64| -> bool {
            if use_small_values {
                small_values.contains(&v)
            } else {
                value_set.contains(&v)
            }
        };
        // For zone map pruning: compute min/max of IN values
        let in_min = *small_values.first().unwrap();
        let in_max = *small_values.last().unwrap();

        let col_count = schema.column_count();
        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for IN scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;
        let max_matches = limit.unwrap_or(usize::MAX);
        let mut matches: Vec<usize> = Vec::new();
        let mut global_row_offset: usize = 0;

        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            if matches.len() >= max_matches { break; }
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { global_row_offset += rg_rows; continue; }

            // Zone map pruning
            if rg_i < footer.zone_maps.len() {
                if let Some(zm) = footer.zone_maps[rg_i].iter().find(|z| z.col_idx as usize == col_idx) {
                    let skip = if zm.is_float {
                        !zm.may_overlap_float_range(in_min as f64, in_max as f64)
                    } else {
                        !zm.may_overlap_int_range(in_min, in_max)
                    };
                    if skip { global_row_offset += rg_rows; continue; }
                }
            }

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Err(err_data("RG extends past EOF")); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
            let del_vec_len = (rg_rows + 7) / 8;
            let id_section = rg_rows * 8;
            if id_section + del_vec_len > body.len() { return Err(err_data("RG del vec truncated")); }
            let del_bytes = &body[id_section..id_section + del_vec_len];
            let has_deletes = rg_meta.deletion_count > 0;
            let null_bitmap_len = (rg_rows + 7) / 8;

            // RCIX fast path
            let rcix_available = rg_i < footer.col_offsets.len()
                && col_idx < footer.col_offsets[rg_i].len()
                && compress_flag == RG_COMPRESS_NONE;
            if rcix_available {
                let col_body_off = footer.col_offsets[rg_i][col_idx] as usize;
                if col_body_off + null_bitmap_len > body.len() { global_row_offset += rg_rows; continue; }
                let null_bytes = &body[col_body_off..col_body_off + null_bitmap_len];
                let data_start = col_body_off + null_bitmap_len;
                let col_bytes = &body[data_start..];
                let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                let encoding = if encoding_version >= 1 && !col_bytes.is_empty() { col_bytes[0] } else { COL_ENCODING_PLAIN };

                if encoding == COL_ENCODING_PLAIN && col_bytes.len() > enc_offset + 8 {
                    let payload = &col_bytes[enc_offset..];
                    let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                    let n = count.min(rg_rows).min((payload.len() - 8) / 8);
                    let no_nulls = !null_bytes.iter().any(|&b| b != 0);
                    let unlimited = max_matches == usize::MAX;
                    if is_int {
                        let vals = bytes_as_i64_slice(&payload[8..], n);
                        if !has_deletes && no_nulls && unlimited {
                            for (i, &v) in vals.iter().enumerate() {
                                if matches_value(v) { matches.push(global_row_offset + i); }
                            }
                        } else {
                            for i in 0..n {
                                if matches.len() >= max_matches { break; }
                                if has_deletes && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                if !no_nulls && (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                if matches_value(vals[i]) { matches.push(global_row_offset + i); }
                            }
                        }
                    } else {
                        let vals = bytes_as_f64_slice(&payload[8..], n);
                        if !has_deletes && no_nulls && unlimited {
                            for (i, &v) in vals.iter().enumerate() {
                                if matches_value(v as i64) { matches.push(global_row_offset + i); }
                            }
                        } else {
                            for i in 0..n {
                                if matches.len() >= max_matches { break; }
                                if has_deletes && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                if !no_nulls && (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                if matches_value(vals[i] as i64) { matches.push(global_row_offset + i); }
                            }
                        }
                    }
                    global_row_offset += rg_rows;
                    continue;
                }
            }

            // Fallback: sequential column scan
            let mut pos = id_section + del_vec_len;
            for ci in 0..col_count {
                if pos + null_bitmap_len > body.len() { break; }
                let null_bytes = &body[pos..pos + null_bitmap_len];
                pos += null_bitmap_len;
                let ct = schema.columns[ci].1;

                if ci == col_idx {
                    let col_bytes = &body[pos..];
                    let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                    let encoding = if encoding_version >= 1 { col_bytes[0] } else { COL_ENCODING_PLAIN };
                    let data_slice = &col_bytes[enc_offset..];

                    if encoding == COL_ENCODING_PLAIN && data_slice.len() >= 8 {
                        let count = u64::from_le_bytes(data_slice[0..8].try_into().unwrap()) as usize;
                        let values_start = 8usize;
                        let n = count.min(rg_rows);
                        if is_int {
                            let nn = n.min((data_slice.len() - values_start) / 8);
                            let vals = bytes_as_i64_slice(&data_slice[values_start..], nn);
                            if !has_deletes {
                                for i in 0..vals.len() {
                                    if matches.len() >= max_matches { break; }
                                    if matches_value(vals[i]) { matches.push(global_row_offset + i); }
                                }
                            } else {
                                for i in 0..vals.len() {
                                    if matches.len() >= max_matches { break; }
                                    if (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if matches_value(vals[i]) { matches.push(global_row_offset + i); }
                                }
                            }
                        } else {
                            let nn = n.min((data_slice.len() - values_start) / 8);
                            let vals = bytes_as_f64_slice(&data_slice[values_start..], nn);
                            if !has_deletes {
                                for i in 0..vals.len() {
                                    if matches.len() >= max_matches { break; }
                                    if matches_value(vals[i] as i64) { matches.push(global_row_offset + i); }
                                }
                            } else {
                                for i in 0..vals.len() {
                                    if matches.len() >= max_matches { break; }
                                    if (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if matches_value(vals[i] as i64) { matches.push(global_row_offset + i); }
                                }
                            }
                        }
                    } else {
                        // Non-PLAIN encoding: decode then scan
                        let (col_data, _consumed) = if encoding_version >= 1 {
                            read_column_encoded(col_bytes, ct)?
                        } else {
                            ColumnData::from_bytes_typed(col_bytes, ct)?
                        };
                        match &col_data {
                            ColumnData::Int64(vals) => {
                                for i in 0..vals.len().min(rg_rows) {
                                    if matches.len() >= max_matches { break; }
                                    if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if matches_value(vals[i]) { matches.push(global_row_offset + i); }
                                }
                            }
                            ColumnData::Float64(vals) => {
                                for i in 0..vals.len().min(rg_rows) {
                                    if matches.len() >= max_matches { break; }
                                    if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if matches_value(vals[i] as i64) { matches.push(global_row_offset + i); }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                let consumed = if encoding_version >= 1 {
                    skip_column_encoded(&body[pos..], ct)?
                } else {
                    ColumnData::skip_bytes_typed(&body[pos..], ct)?
                };
                pos += consumed;
            }
            global_row_offset += rg_rows;
        }
        drop(mmap_guard);
        drop(file_guard);
        Ok(Some(matches))
    }

    /// Scan a numeric WHERE column and write new values to a SET column in-place, in one pass.
    /// Only works when both columns use PLAIN encoding and RGs are uncompressed (RCIX required).
    /// Returns Some(count) of rows updated, or None if conditions not met (caller falls back).
    pub fn scan_and_update_inplace(
        &self,
        where_col: &str,
        low: f64,
        high: f64,
        set_col: &str,
        new_value_bytes: &[u8; 8], // raw little-endian bytes of the new value (f64 or i64)
    ) -> io::Result<Option<i64>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let where_idx = match schema.get_index(where_col) { Some(i) => i, None => return Ok(None) };
        let set_idx   = match schema.get_index(set_col)   { Some(i) => i, None => return Ok(None) };
        let where_type = schema.columns[where_idx].1;
        let is_int = matches!(where_type, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
            ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
            ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date);
        let is_float = matches!(where_type, ColumnType::Float64 | ColumnType::Float32);
        if !is_int && !is_float { return Ok(None); }
        // Require RCIX for both columns in all row groups
        let n_rgs = footer.row_groups.len();
        if footer.col_offsets.len() < n_rgs { return Ok(None); }
        let low_i = low.ceil() as i64;
        let high_i = high.floor() as i64;
        let mut total_updated: i64 = 0;

        // Need read-write access: open separate write handle
        let mut write_file = std::fs::OpenOptions::new().read(true).write(true).open(&self.path)?;

        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| err_not_conn("File not open"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { continue; }
            // Zone map pruning for WHERE column
            if rg_i < footer.zone_maps.len() {
                if let Some(zm) = footer.zone_maps[rg_i].iter().find(|z| z.col_idx as usize == where_idx) {
                    let skip = if zm.is_float { !zm.may_overlap_float_range(low, high) }
                               else { !zm.may_overlap_int_range(low_i, high_i) };
                    if skip { continue; }
                }
            }
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Ok(None); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { 1 };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            // Require uncompressed + RCIX for in-place write
            if compress_flag != RG_COMPRESS_NONE { return Ok(None); }
            if rg_i >= footer.col_offsets.len() { return Ok(None); }
            let rg_col_offsets = &footer.col_offsets[rg_i];
            if where_idx >= rg_col_offsets.len() || set_idx >= rg_col_offsets.len() { return Ok(None); }

            let body = &rg_bytes[32..];
            let del_vec_len = (rg_rows + 7) / 8;
            let null_bitmap_len = del_vec_len;
            let id_section = rg_rows * 8;
            if id_section + del_vec_len > body.len() { continue; }
            let del_bytes = &body[id_section..id_section + del_vec_len];
            let has_deletes = rg_meta.deletion_count > 0;

            // Read WHERE column (any encoding — decode it)
            let where_col_off = rg_col_offsets[where_idx] as usize;
            if where_col_off + null_bitmap_len > body.len() { continue; }
            let where_null = &body[where_col_off..where_col_off + null_bitmap_len];
            let where_col_bytes = &body[where_col_off + null_bitmap_len..];

            // Decode WHERE column values (supports PLAIN, BITPACK, RLE).
            // Keep PLAIN mmap data borrowed; copying an 8MB predicate column
            // dominates idempotent UPDATEs on the OLTP benchmark.
            use std::borrow::Cow;
            enum WhereVals<'a> {
                Int(Cow<'a, [i64]>),
                Flt(Cow<'a, [f64]>),
                BitPackI64 {
                    packed: &'a [u8],
                    bit_width: usize,
                    min_val: i64,
                },
            }
            let where_vals: WhereVals<'_>;
            let n: usize;

            if is_int {
                // Try zero-copy PLAIN path first
                let enc = if encoding_version >= 1 && !where_col_bytes.is_empty() { where_col_bytes[0] } else { COL_ENCODING_PLAIN };
                if enc == COL_ENCODING_PLAIN {
                    let payload = &where_col_bytes[1..];
                    if payload.len() < 8 { continue; }
                    let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                    let nn = count.min(rg_rows).min((payload.len() - 8) / 8);
                    let vals_cow = bytes_as_i64_slice(&payload[8..], nn);
                    n = nn;
                    where_vals = WhereVals::Int(vals_cow);
                } else if enc == COL_ENCODING_BITPACK {
                    let payload = &where_col_bytes[1..];
                    if payload.len() < 17 { continue; }
                    let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                    let bit_width = payload[8] as usize;
                    if bit_width >= 64 { return Ok(None); }
                    let min_val = i64::from_le_bytes(payload[9..17].try_into().unwrap());
                    let packed_bytes = (count * bit_width + 7) / 8;
                    if payload.len() < 17 + packed_bytes { return Ok(None); }
                    n = count.min(rg_rows);
                    where_vals = WhereVals::BitPackI64 {
                        packed: &payload[17..17 + packed_bytes],
                        bit_width,
                        min_val,
                    };
                } else {
                    // Decode (BITPACK, RLE, etc.)
                    let (col_data, _) = read_column_encoded(where_col_bytes, where_type)?;
                    match col_data {
                        ColumnData::Int64(v) => {
                            let nn = v.len().min(rg_rows);
                            n = nn;
                            where_vals = WhereVals::Int(Cow::Owned(v));
                        }
                        _ => continue,
                    }
                }
            } else {
                let enc = if encoding_version >= 1 && !where_col_bytes.is_empty() { where_col_bytes[0] } else { COL_ENCODING_PLAIN };
                if enc == COL_ENCODING_PLAIN {
                    let payload = &where_col_bytes[1..];
                    if payload.len() < 8 { continue; }
                    let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                    let nn = count.min(rg_rows).min((payload.len() - 8) / 8);
                    let vals_cow = bytes_as_f64_slice(&payload[8..], nn);
                    n = nn;
                    where_vals = WhereVals::Flt(vals_cow);
                } else {
                    let (col_data, _) = read_column_encoded(where_col_bytes, where_type)?;
                    match col_data {
                        ColumnData::Float64(v) => {
                            let nn = v.len().min(rg_rows);
                            n = nn;
                            where_vals = WhereVals::Flt(Cow::Owned(v));
                        }
                        _ => continue,
                    }
                }
            };

            // Verify SET column is PLAIN (required for in-place overwrite)
            let set_col_off = rg_col_offsets[set_idx] as usize;
            if set_col_off + null_bitmap_len > body.len() { continue; }
            let set_data = &body[set_col_off + null_bitmap_len..];
            let set_enc = if encoding_version >= 1 && !set_data.is_empty() { set_data[0] } else { COL_ENCODING_PLAIN };
            if set_enc != COL_ENCODING_PLAIN { return Ok(None); }
            if set_data.len() < 9 { return Ok(None); }
            let set_count = u64::from_le_bytes(set_data[1..9].try_into().unwrap()) as usize;
            if set_count < n || set_data.len() < 9 + n * 8 { return Ok(None); }
            let set_values = &set_data[9..9 + n * 8];

            // File offset of SET column's value array:
            // rg_meta.offset (RG start) + 32 (RG header) + set_col_off + null_bitmap_len + 1 (enc byte) + 8 (count)
            let values_file_offset = (rg_meta.offset as usize + 32 + set_col_off + null_bitmap_len + 1 + 8) as u64;

            // Lazily copy and rewrite the SET value array only when at least one
            // matching row would physically change. Repeated idempotent UPDATEs
            // still return the matched-row count without dirtying the file.
            use std::io::{Seek, SeekFrom, Write};
            let mut value_buf: Option<Vec<u8>> = None;
            let mut rg_updated = 0i64;
            match where_vals {
                WhereVals::Int(vals) => {
                    let no_nulls = !where_null.iter().any(|&b| b != 0);
                    if !has_deletes && no_nulls {
                        for i in 0..n {
                            if vals[i] >= low_i && vals[i] <= high_i {
                                let off = i * 8;
                                if &set_values[off..off + 8] != new_value_bytes {
                                    let buf = value_buf.get_or_insert_with(|| set_values.to_vec());
                                    buf[off..off + 8].copy_from_slice(new_value_bytes);
                                }
                                rg_updated += 1;
                            }
                        }
                    } else {
                        for i in 0..n {
                            if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                                continue;
                            }
                            if !no_nulls && (where_null[i / 8] >> (i % 8)) & 1 == 1 {
                                continue;
                            }
                            if vals[i] >= low_i && vals[i] <= high_i {
                                let off = i * 8;
                                if &set_values[off..off + 8] != new_value_bytes {
                                    let buf = value_buf.get_or_insert_with(|| set_values.to_vec());
                                    buf[off..off + 8].copy_from_slice(new_value_bytes);
                                }
                                rg_updated += 1;
                            }
                        }
                    }
                }
                WhereVals::Flt(vals) => {
                    let no_nulls = !where_null.iter().any(|&b| b != 0);
                    if !has_deletes && no_nulls {
                        for i in 0..n {
                            if vals[i] >= low && vals[i] <= high {
                                let off = i * 8;
                                if &set_values[off..off + 8] != new_value_bytes {
                                    let buf = value_buf.get_or_insert_with(|| set_values.to_vec());
                                    buf[off..off + 8].copy_from_slice(new_value_bytes);
                                }
                                rg_updated += 1;
                            }
                        }
                    } else {
                        for i in 0..n {
                            if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                                continue;
                            }
                            if !no_nulls && (where_null[i / 8] >> (i % 8)) & 1 == 1 {
                                continue;
                            }
                            if vals[i] >= low && vals[i] <= high {
                                let off = i * 8;
                                if &set_values[off..off + 8] != new_value_bytes {
                                    let buf = value_buf.get_or_insert_with(|| set_values.to_vec());
                                    buf[off..off + 8].copy_from_slice(new_value_bytes);
                                }
                                rg_updated += 1;
                            }
                        }
                    }
                }
                WhereVals::BitPackI64 { packed, bit_width, min_val } => {
                    let mask = if bit_width == 0 { 0 } else { (1u64 << bit_width) - 1 };
                    let no_nulls = !where_null.iter().any(|&b| b != 0);
                    let handle_match =
                        |i: usize, value_buf: &mut Option<Vec<u8>>, rg_updated: &mut i64| {
                            let off = i * 8;
                            if &set_values[off..off + 8] != new_value_bytes {
                                let buf = value_buf.get_or_insert_with(|| set_values.to_vec());
                                buf[off..off + 8].copy_from_slice(new_value_bytes);
                            }
                            *rg_updated += 1;
                        };
                    if bit_width == 0 {
                        if min_val >= low_i && min_val <= high_i {
                            if !has_deletes && no_nulls {
                                for i in 0..n {
                                    handle_match(i, &mut value_buf, &mut rg_updated);
                                }
                            } else {
                                for i in 0..n {
                                    if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                                        continue;
                                    }
                                    if !no_nulls && (where_null[i / 8] >> (i % 8)) & 1 == 1 {
                                        continue;
                                    }
                                    handle_match(i, &mut value_buf, &mut rg_updated);
                                }
                            }
                        }
                    } else if !has_deletes && no_nulls {
                        for i in 0..n {
                            let bit_pos = i * bit_width;
                            let byte_off = bit_pos / 8;
                            let bit_shift = bit_pos % 8;
                            let bytes_needed = (bit_shift + bit_width + 7) / 8;
                            if byte_off + bytes_needed > packed.len() {
                                continue;
                            }
                            let mut raw = 0u64;
                            for j in 0..bytes_needed {
                                raw |= (packed[byte_off + j] as u64) << (j * 8);
                            }
                            let v = min_val.wrapping_add(((raw >> bit_shift) & mask) as i64);
                            if v >= low_i && v <= high_i {
                                handle_match(i, &mut value_buf, &mut rg_updated);
                            }
                        }
                    } else {
                        for i in 0..n {
                            if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                                continue;
                            }
                            if !no_nulls && (where_null[i / 8] >> (i % 8)) & 1 == 1 {
                                continue;
                            }
                            let bit_pos = i * bit_width;
                            let byte_off = bit_pos / 8;
                            let bit_shift = bit_pos % 8;
                            let bytes_needed = (bit_shift + bit_width + 7) / 8;
                            if byte_off + bytes_needed > packed.len() {
                                continue;
                            }
                            let mut raw = 0u64;
                            for j in 0..bytes_needed {
                                raw |= (packed[byte_off + j] as u64) << (j * 8);
                            }
                            let v = min_val.wrapping_add(((raw >> bit_shift) & mask) as i64);
                            if v >= low_i && v <= high_i {
                                handle_match(i, &mut value_buf, &mut rg_updated);
                            }
                        }
                    }
                }
            }
            if rg_updated > 0 {
                if let Some(value_buf) = value_buf {
                    write_file.seek(SeekFrom::Start(values_file_offset))?;
                    write_file.write_all(&value_buf)?;
                }
                total_updated += rg_updated;
            }
        }
        drop(mmap_guard);
        drop(file_guard);
        Ok(Some(total_updated))
    }

    /// Overwrite one numeric cell by `_id` using the V4 row-group id section.
    /// Returns None unless the target row exists, is active, and the SET column is
    /// PLAIN numeric in an uncompressed RCIX row group.
    pub fn locate_numeric_cell_for_update(
        &self,
        id: u64,
        set_col: &str,
    ) -> io::Result<Option<(u64, u64, u8, u64)>> {
        if set_col == "_id" {
            return Ok(None);
        }

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let footer_offset = self.footer_offset_hint();
        if footer_offset == 0 {
            return Ok(None);
        }

        let schema = &footer.schema;
        let set_idx = match schema.get_index(set_col) {
            Some(i) => i,
            None => return Ok(None),
        };
        let set_type = schema.columns[set_idx].1;
        let is_numeric = matches!(
            set_type,
            ColumnType::Int64
                | ColumnType::Int8
                | ColumnType::Int16
                | ColumnType::Int32
                | ColumnType::UInt8
                | ColumnType::UInt16
                | ColumnType::UInt32
                | ColumnType::UInt64
                | ColumnType::Timestamp
                | ColumnType::Date
                | ColumnType::Float64
                | ColumnType::Float32
        );
        if !is_numeric {
            return Ok(None);
        }

        let (rg_i, rg_meta) = match footer
            .row_groups
            .iter()
            .enumerate()
            .find(|(_, rg)| rg.min_id <= id && id <= rg.max_id && rg.row_count > 0)
        {
            Some(v) => v,
            None => return Ok(Some((footer_offset, 0, 0, 0))),
        };
        if rg_i >= footer.col_offsets.len() || set_idx >= footer.col_offsets[rg_i].len() {
            return Ok(None);
        }

        let rg_rows = rg_meta.row_count as usize;
        let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
        let file_guard = self.file.read();
        let file = file_guard
            .as_ref()
            .ok_or_else(|| err_not_conn("File not open"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;
        if rg_end > mmap_ref.len() {
            return Ok(None);
        }

        let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
        let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { 1 };
        let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
        if compress_flag != RG_COMPRESS_NONE || encoding_version < 1 {
            return Ok(None);
        }

        let body = &rg_bytes[32..];
        let guess = id.saturating_sub(rg_meta.min_id) as usize;
        if guess >= rg_rows {
            return Ok(Some((footer_offset, 0, 0, 0)));
        }
        let id_start = guess * 8;
        if id_start + 8 > body.len() {
            return Ok(None);
        }
        let actual_id = u64::from_le_bytes(body[id_start..id_start + 8].try_into().unwrap());
        if actual_id != id {
            return Ok(None);
        }

        let bitmap_len = (rg_rows + 7) / 8;
        let del_off = rg_rows * 8 + guess / 8;
        if del_off >= body.len() {
            return Ok(None);
        }
        if ((body[del_off] >> (guess % 8)) & 1) == 1 {
            return Ok(Some((footer_offset, 0, 0, 0)));
        }

        let set_col_off = footer.col_offsets[rg_i][set_idx] as usize;
        if set_col_off + bitmap_len + 1 + 8 > body.len() {
            return Ok(None);
        }
        let set_data = &body[set_col_off + bitmap_len..];
        if set_data.is_empty() || set_data[0] != COL_ENCODING_PLAIN {
            return Ok(None);
        }

        let value_file_offset =
            rg_meta.offset + 32 + (set_col_off + bitmap_len + 1 + 8 + guess * 8) as u64;
        let null_byte_file_offset = rg_meta.offset + 32 + (set_col_off + guess / 8) as u64;
        Ok(Some((
            footer_offset,
            null_byte_file_offset,
            1u8 << (guess % 8),
            value_file_offset,
        )))
    }

    pub fn update_by_id_inplace(
        &self,
        id: u64,
        set_col: &str,
        new_value_bytes: &[u8; 8],
    ) -> io::Result<Option<(i64, bool)>> {
        if set_col == "_id" {
            return Ok(None);
        }

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let set_idx = match schema.get_index(set_col) {
            Some(i) => i,
            None => return Ok(None),
        };
        let set_type = schema.columns[set_idx].1;
        let is_numeric = matches!(
            set_type,
            ColumnType::Int64
                | ColumnType::Int8
                | ColumnType::Int16
                | ColumnType::Int32
                | ColumnType::UInt8
                | ColumnType::UInt16
                | ColumnType::UInt32
                | ColumnType::UInt64
                | ColumnType::Timestamp
                | ColumnType::Date
                | ColumnType::Float64
                | ColumnType::Float32
        );
        if !is_numeric {
            return Ok(None);
        }

        let (rg_i, rg_meta) = match footer
            .row_groups
            .iter()
            .enumerate()
            .find(|(_, rg)| rg.min_id <= id && id <= rg.max_id && rg.row_count > 0)
        {
            Some(v) => v,
            None => return Ok(Some((0, false))),
        };
        if rg_i >= footer.col_offsets.len() || set_idx >= footer.col_offsets[rg_i].len() {
            return Ok(None);
        }

        let rg_rows = rg_meta.row_count as usize;
        let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
        let file_guard = self.file.read();
        let file = file_guard
            .as_ref()
            .ok_or_else(|| err_not_conn("File not open"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;
        if rg_end > mmap_ref.len() {
            return Ok(None);
        }

        let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
        let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { 1 };
        let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
        if compress_flag != RG_COMPRESS_NONE || encoding_version < 1 {
            return Ok(None);
        }

        let body = &rg_bytes[32..];
        let guess = id.saturating_sub(rg_meta.min_id) as usize;
        if guess >= rg_rows {
            return Ok(Some((0, false)));
        }
        let id_start = guess * 8;
        if id_start + 8 > body.len() {
            return Ok(None);
        }
        let actual_id = u64::from_le_bytes(body[id_start..id_start + 8].try_into().unwrap());
        if actual_id != id {
            return Ok(None);
        }

        let bitmap_len = (rg_rows + 7) / 8;
        let del_off = rg_rows * 8 + guess / 8;
        if del_off >= body.len() {
            return Ok(None);
        }
        if ((body[del_off] >> (guess % 8)) & 1) == 1 {
            return Ok(Some((0, false)));
        }

        let set_col_off = footer.col_offsets[rg_i][set_idx] as usize;
        if set_col_off + bitmap_len + 1 + 8 > body.len() {
            return Ok(None);
        }
        let set_data = &body[set_col_off + bitmap_len..];
        if set_data.is_empty() || set_data[0] != COL_ENCODING_PLAIN {
            return Ok(None);
        }

        let value_file_offset =
            rg_meta.offset + 32 + (set_col_off + bitmap_len + 1 + 8 + guess * 8) as u64;
        let null_byte_file_offset = rg_meta.offset + 32 + (set_col_off + guess / 8) as u64;
        let value_body_offset = set_col_off + bitmap_len + 1 + 8 + guess * 8;
        let null_byte = body[set_col_off + guess / 8];
        if ((null_byte >> (guess % 8)) & 1) == 0
            && value_body_offset + 8 <= body.len()
            && &body[value_body_offset..value_body_offset + 8] == new_value_bytes
        {
            return Ok(Some((1, false)));
        }

        use std::io::{Seek, SeekFrom, Write};
        let mut write_file_guard = self.write_file.write();
        if write_file_guard.is_none() {
            *write_file_guard = Some(
                std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&self.path)?,
            );
        }
        let write_file = write_file_guard
            .as_mut()
            .ok_or_else(|| err_not_conn("Write file not open"))?;
        let mut null_byte = null_byte;
        null_byte &= !(1u8 << (guess % 8));
        write_file.seek(SeekFrom::Start(null_byte_file_offset))?;
        write_file.write_all(&[null_byte])?;
        write_file.seek(SeekFrom::Start(value_file_offset))?;
        write_file.write_all(new_value_bytes)?;

        drop(mmap_guard);
        drop(file_guard);
        Ok(Some((1, true)))
    }

    pub fn update_numeric_cell_cached(
        &self,
        footer_offset: u64,
        null_byte_file_offset: u64,
        null_mask: u8,
        value_file_offset: u64,
        new_value_bytes: &[u8; 8],
    ) -> io::Result<Option<(i64, bool)>> {
        if footer_offset == 0 || self.footer_offset_hint() != footer_offset {
            return Ok(None);
        }

        let file_guard = self.file.read();
        let file = file_guard
            .as_ref()
            .ok_or_else(|| err_not_conn("File not open"))?;

        let mut null_byte = [0u8; 1];
        pread_fallback(file, &mut null_byte, null_byte_file_offset)?;
        if (null_byte[0] & null_mask) != 0 {
            return Ok(Some((0, false)));
        }

        let mut current = [0u8; 8];
        pread_fallback(file, &mut current, value_file_offset)?;
        if current == *new_value_bytes {
            return Ok(Some((1, false)));
        }

        use std::io::{Seek, SeekFrom, Write};
        let mut write_file_guard = self.write_file.write();
        if write_file_guard.is_none() {
            *write_file_guard = Some(
                std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&self.path)?,
            );
        }
        let write_file = write_file_guard
            .as_mut()
            .ok_or_else(|| err_not_conn("Write file not open"))?;

        let updated_null = null_byte[0] & !null_mask;
        if updated_null != null_byte[0] {
            write_file.seek(SeekFrom::Start(null_byte_file_offset))?;
            write_file.write_all(&[updated_null])?;
        }
        write_file.seek(SeekFrom::Start(value_file_offset))?;
        write_file.write_all(new_value_bytes)?;

        Ok(Some((1, true)))
    }

    /// Check whether `_id` is present and not deleted by reading only the row-group
    /// id/deletion sections. Returns None when the V4 layout is not directly readable.
    pub fn row_id_active_rcix(&self, id: u64) -> io::Result<Option<bool>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let rg_meta = match footer
            .row_groups
            .iter()
            .find(|rg| rg.min_id <= id && id <= rg.max_id && rg.row_count > 0)
        {
            Some(rg) => rg,
            None => return Ok(Some(false)),
        };
        let rg_rows = rg_meta.row_count as usize;
        let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;

        let file_guard = self.file.read();
        let file = file_guard
            .as_ref()
            .ok_or_else(|| err_not_conn("File not open"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;
        if rg_end > mmap_ref.len() {
            return Ok(None);
        }
        let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
        if rg_bytes.len() < 32 || rg_bytes[28] != RG_COMPRESS_NONE {
            return Ok(None);
        }

        let body = &rg_bytes[32..];
        if body.len() < rg_rows * 8 {
            return Ok(None);
        }
        let guess = id.saturating_sub(rg_meta.min_id) as usize;
        let local_idx = if guess < rg_rows {
            let start = guess * 8;
            let actual = u64::from_le_bytes(body[start..start + 8].try_into().unwrap());
            if actual == id {
                guess
            } else {
                let ids_cow = bytes_as_u64_slice(&body[..rg_rows * 8], rg_rows);
                match ids_cow.binary_search(&id) {
                    Ok(i) => i,
                    Err(_) => return Ok(Some(false)),
                }
            }
        } else {
            return Ok(Some(false));
        };

        let del_off = rg_rows * 8 + local_idx / 8;
        if del_off >= body.len() {
            return Ok(None);
        }
        let deleted = ((body[del_off] >> (local_idx % 8)) & 1) == 1;
        Ok(Some(!deleted))
    }

    /// Scan a numeric column for rows in [low, high] and return their row IDs directly.
    /// Unlike scan_numeric_range_mmap which returns global row indices, this reads IDs
    /// from the row group body in the same pass — avoids a separate _id column read.
    /// Also checks the delta store for overridden values on already-updated rows.
    pub fn scan_numeric_range_mmap_with_ids(
        &self,
        col_name: &str,
        low: f64,
        high: f64,
    ) -> io::Result<Option<Vec<u64>>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        let is_int = matches!(col_type, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
            ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
            ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date);
        let is_float = matches!(col_type, ColumnType::Float64 | ColumnType::Float32);
        if !is_int && !is_float { return Ok(None); }

        let col_count = schema.column_count();
        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for range+id scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        let low_i = low.ceil() as i64;
        let high_i = high.floor() as i64;
        let mut result: Vec<u64> = Vec::new();

        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { continue; }
            // Skip fully-deleted row groups — zone maps may still show overlap
            if rg_meta.active_rows() == 0 { continue; }

            // Zone map pruning
            if rg_i < footer.zone_maps.len() {
                if let Some(zm) = footer.zone_maps[rg_i].iter().find(|z| z.col_idx as usize == col_idx) {
                    let skip = if zm.is_float {
                        !zm.may_overlap_float_range(low, high)
                    } else {
                        !zm.may_overlap_int_range(low_i, high_i)
                    };
                    if skip { continue; }
                }
            }

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Err(err_data("RG extends past EOF")); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);

            let id_section = rg_rows * 8;
            let del_vec_len = (rg_rows + 7) / 8;
            let null_bitmap_len = del_vec_len;
            if id_section + del_vec_len > body.len() { continue; }

            // Read IDs in bulk
            let ids_cow = bytes_as_u64_slice(body, rg_rows);
            let ids: &[u64] = &ids_cow;
            let del_bytes = &body[id_section..id_section + del_vec_len];
            let has_deletes = rg_meta.deletion_count > 0;

            // RCIX fast path: jump directly to target column
            let rcix_available = rg_i < footer.col_offsets.len()
                && col_idx < footer.col_offsets[rg_i].len()
                && compress_flag == RG_COMPRESS_NONE;
            if rcix_available {
                let col_body_off = footer.col_offsets[rg_i][col_idx] as usize;
                if col_body_off + null_bitmap_len > body.len() { continue; }
                let null_bytes = &body[col_body_off..col_body_off + null_bitmap_len];
                let data_start = col_body_off + null_bitmap_len;
                let col_bytes = &body[data_start..];
                let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                let encoding = if encoding_version >= 1 && !col_bytes.is_empty() { col_bytes[0] } else { COL_ENCODING_PLAIN };
                if encoding == COL_ENCODING_PLAIN && col_bytes.len() > enc_offset + 8 {
                    let payload = &col_bytes[enc_offset..];
                    let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                    let n = count.min(rg_rows).min((payload.len() - 8) / 8);
                    if is_int {
                        let vals = bytes_as_i64_slice(&payload[8..], n);
                        for i in 0..n {
                            if has_deletes && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            if (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            if vals[i] >= low_i && vals[i] <= high_i { result.push(ids[i]); }
                        }
                    } else {
                        let vals = bytes_as_f64_slice(&payload[8..], n);
                        for i in 0..n {
                            if has_deletes && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            if (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            if vals[i] >= low && vals[i] <= high { result.push(ids[i]); }
                        }
                    }
                    continue;
                }
                // RCIX encoding-aware range pruning: read min/max from encoding header
                // without allocating, skip RG if range can't overlap [low_i, high_i].
                if is_int {
                    let data = &col_bytes[enc_offset..];
                    let can_skip = if encoding == COL_ENCODING_BITPACK && data.len() >= 17 {
                        // Header: [count:u64][bit_width:u8][min_value:i64]
                        let bit_width = data[8] as u32;
                        let min_val = i64::from_le_bytes(data[9..17].try_into().unwrap());
                        let max_val = if bit_width == 0 { min_val }
                                      else { min_val.saturating_add(((1u64 << bit_width) - 1) as i64) };
                        max_val < low_i || min_val > high_i
                    } else if encoding == COL_ENCODING_RLE && data.len() >= 16 {
                        // Header: [count:u64][num_runs:u64][(value:i64,run_len:u32)...]
                        let num_runs = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
                        let mut rle_min = i64::MAX;
                        let mut rle_max = i64::MIN;
                        let mut ok = true;
                        for r in 0..num_runs {
                            let off = 16 + r * 12;
                            if off + 8 > data.len() { ok = false; break; }
                            let v = i64::from_le_bytes(data[off..off+8].try_into().unwrap());
                            if v < rle_min { rle_min = v; }
                            if v > rle_max { rle_max = v; }
                        }
                        ok && (rle_max < low_i || rle_min > high_i)
                    } else {
                        false
                    };
                    if can_skip { continue; }
                }
            }

            // Sequential fallback: scan columns until we reach col_idx
            let mut pos = id_section + del_vec_len;
            for ci in 0..col_count {
                if pos + null_bitmap_len > body.len() { break; }
                let null_bytes = &body[pos..pos + null_bitmap_len];
                pos += null_bitmap_len;
                let ct = schema.columns[ci].1;
                if ci == col_idx {
                    let col_bytes = &body[pos..];
                    let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                    let encoding = if encoding_version >= 1 && !col_bytes.is_empty() { col_bytes[0] } else { COL_ENCODING_PLAIN };
                    let data_slice = &col_bytes[enc_offset..];
                    if encoding == COL_ENCODING_PLAIN && data_slice.len() >= 8 {
                        let count = u64::from_le_bytes(data_slice[0..8].try_into().unwrap()) as usize;
                        let n = count.min(rg_rows);
                        if is_int {
                            let nn = n.min((data_slice.len()-8)/8);
                            let vals = bytes_as_i64_slice(&data_slice[8..], nn);
                            for i in 0..vals.len() {
                                if has_deletes && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                if (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                if vals[i] >= low_i && vals[i] <= high_i { result.push(ids[i]); }
                            }
                        } else {
                            let nn = n.min((data_slice.len()-8)/8);
                            let vals = bytes_as_f64_slice(&data_slice[8..], nn);
                            for i in 0..vals.len() {
                                if has_deletes && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                if (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                if vals[i] >= low && vals[i] <= high { result.push(ids[i]); }
                            }
                        }
                    } else {
                        // Range pruning before full decode: read min/max from encoding
                        // header without allocating to skip RGs with no matching values.
                        if is_int && encoding_version >= 1 {
                            let data = &col_bytes[enc_offset..];
                            let can_skip = if encoding == COL_ENCODING_BITPACK && data.len() >= 17 {
                                let bit_width = data[8] as u32;
                                let min_val = i64::from_le_bytes(data[9..17].try_into().unwrap());
                                let max_val = if bit_width == 0 { min_val }
                                              else { min_val.saturating_add(((1u64 << bit_width) - 1) as i64) };
                                max_val < low_i || min_val > high_i
                            } else if encoding == COL_ENCODING_RLE && data.len() >= 16 {
                                let num_runs = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
                                let mut rle_min = i64::MAX;
                                let mut rle_max = i64::MIN;
                                let mut ok = true;
                                for r in 0..num_runs {
                                    let off = 16 + r * 12;
                                    if off + 8 > data.len() { ok = false; break; }
                                    let v = i64::from_le_bytes(data[off..off+8].try_into().unwrap());
                                    if v < rle_min { rle_min = v; }
                                    if v > rle_max { rle_max = v; }
                                }
                                ok && (rle_max < low_i || rle_min > high_i)
                            } else { false };
                            if can_skip { break; }
                        }
                        let (col_data, _) = if encoding_version >= 1 {
                            read_column_encoded(col_bytes, ct)?
                        } else {
                            ColumnData::from_bytes_typed(col_bytes, ct)?
                        };
                        match &col_data {
                            ColumnData::Int64(vals) => {
                                for i in 0..vals.len().min(rg_rows) {
                                    if has_deletes && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                    if (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                    if vals[i] >= low_i && vals[i] <= high_i { result.push(ids[i]); }
                                }
                            }
                            ColumnData::Float64(vals) => {
                                for i in 0..vals.len().min(rg_rows) {
                                    if has_deletes && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                    if (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                    if vals[i] >= low && vals[i] <= high { result.push(ids[i]); }
                                }
                            }
                            _ => {}
                        }
                    }
                    break; // Found and processed the target column, move to next RG
                }
                let consumed = if encoding_version >= 1 {
                    skip_column_encoded(&body[pos..], ct)?
                } else {
                    ColumnData::skip_bytes_typed(&body[pos..], ct)?
                };
                pos += consumed;
            }
        }
        drop(mmap_guard);
        drop(file_guard);
        Ok(Some(result))
    }

    /// Direct mmap top-K scan: finds top-k row indices by a numeric column without materializing
    /// the full Arrow array. Uses RCIX + zone maps for O(N_rows) with O(k) heap in L1 cache.
    /// Returns Vec<(global_row_idx, value)> sorted in the requested order.
    pub fn scan_top_k_indices_mmap(
        &self,
        col_name: &str,
        k: usize,
        descending: bool,
    ) -> io::Result<Option<Vec<(usize, f64)>>> {
        if k == 0 { return Ok(Some(vec![])); }
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        let is_int = matches!(col_type, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
            ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
            ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date);
        let is_float = matches!(col_type, ColumnType::Float64 | ColumnType::Float32);
        if !is_int && !is_float { return Ok(None); }

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for top-k scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        // heap: sorted Vec<(value, global_idx)>; descending → keep k largest
        let mut heap: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
        let mut global_offset: usize = 0;

        for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { global_offset += rg_rows; continue; }

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Err(err_data("RG extends past EOF")); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
            let id_section = rg_rows * 8;
            let del_vec_len = (rg_rows + 7) / 8;
            let null_bitmap_len = (rg_rows + 7) / 8;
            let has_deletes = rg_meta.deletion_count > 0;
            let del_bytes = if id_section + del_vec_len <= body.len() {
                &body[id_section..id_section + del_vec_len]
            } else { &[] };

            // Get pointer to column data via RCIX if available
            let col_bytes: &[u8] = if rg_i < footer.col_offsets.len()
                && col_idx < footer.col_offsets[rg_i].len()
                && compress_flag == RG_COMPRESS_NONE
            {
                let col_body_off = footer.col_offsets[rg_i][col_idx] as usize;
                let data_start = col_body_off + null_bitmap_len;
                if data_start > body.len() { global_offset += rg_rows; continue; }
                &body[data_start..]
            } else {
                // Fallback: sequential column scan
                let mut pos = id_section + del_vec_len;
                let mut found: &[u8] = &[];
                for ci in 0..schema.column_count() {
                    if pos + null_bitmap_len > body.len() { break; }
                    pos += null_bitmap_len;
                    if ci == col_idx { found = &body[pos..]; break; }
                    let consumed = if encoding_version >= 1 {
                        skip_column_encoded(&body[pos..], schema.columns[ci].1)?
                    } else {
                        ColumnData::skip_bytes_typed(&body[pos..], schema.columns[ci].1)?
                    };
                    pos += consumed;
                }
                found
            };

            if col_bytes.is_empty() { global_offset += rg_rows; continue; }

            let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
            let encoding = if encoding_version >= 1 && !col_bytes.is_empty() { col_bytes[0] } else { COL_ENCODING_PLAIN };

            if encoding == COL_ENCODING_PLAIN && col_bytes.len() > enc_offset + 8 {
                let payload = &col_bytes[enc_offset..];
                let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                let n = count.min(rg_rows).min((payload.len() - 8) / 8);

                macro_rules! topk_scan {
                    ($vals:expr) => {{
                        if descending {
                            // Keep k largest: heap sorted descending, threshold = heap[k-1]
                            for i in 0..n {
                                if has_deletes && !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                let val = $vals[i];
                                if heap.len() < k {
                                    let pos = heap.partition_point(|(v, _)| *v > val);
                                    heap.insert(pos, (val, global_offset + i));
                                } else if val > heap[k - 1].0 {
                                    let pos = heap.partition_point(|(v, _)| *v > val);
                                    heap.insert(pos, (val, global_offset + i));
                                    heap.pop();
                                }
                            }
                        } else {
                            // Keep k smallest: heap sorted ascending, threshold = heap[k-1]
                            for i in 0..n {
                                if has_deletes && !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                let val = $vals[i];
                                if heap.len() < k {
                                    let pos = heap.partition_point(|(v, _)| *v < val);
                                    heap.insert(pos, (val, global_offset + i));
                                } else if val < heap[k - 1].0 {
                                    let pos = heap.partition_point(|(v, _)| *v < val);
                                    heap.insert(pos, (val, global_offset + i));
                                    heap.pop();
                                }
                            }
                        }
                    }};
                }

                if is_float {
                    let ptr = payload[8..].as_ptr();
                    if ptr as usize % std::mem::align_of::<f64>() == 0 {
                        let vals = unsafe { std::slice::from_raw_parts(ptr as *const f64, n) };
                        topk_scan!(vals);
                    } else {
                        let data = &payload[8..8 + n * 8];
                        let vals: Vec<f64> = (0..n).map(|i| f64::from_le_bytes(data[i*8..i*8+8].try_into().unwrap())).collect();
                        topk_scan!(vals);
                    }
                } else {
                    let ptr = payload[8..].as_ptr();
                    if ptr as usize % std::mem::align_of::<i64>() == 0 {
                        let vals = unsafe { std::slice::from_raw_parts(ptr as *const i64, n) };
                        let fvals: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
                        topk_scan!(fvals);
                    } else {
                        let data = &payload[8..8 + n * 8];
                        let fvals: Vec<f64> = (0..n).map(|i| i64::from_le_bytes(data[i*8..i*8+8].try_into().unwrap()) as f64).collect();
                        topk_scan!(fvals);
                    }
                }
            } else {
                // Non-PLAIN: decode and scan
                let (col_data, _) = if encoding_version >= 1 {
                    read_column_encoded(col_bytes, col_type)?
                } else {
                    ColumnData::from_bytes_typed(col_bytes, col_type)?
                };
                let fvals: Vec<f64> = match &col_data {
                    ColumnData::Float64(v) => v.iter().map(|&x| x).collect(),
                    ColumnData::Int64(v) => v.iter().map(|&x| x as f64).collect(),
                    _ => { global_offset += rg_rows; continue; }
                };
                let n = fvals.len().min(rg_rows);
                macro_rules! topk_scan2 {
                    ($vals:expr) => {{
                        if descending {
                            for i in 0..n {
                                if has_deletes && !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                let val = $vals[i];
                                if heap.len() < k {
                                    let pos = heap.partition_point(|(v, _)| *v > val);
                                    heap.insert(pos, (val, global_offset + i));
                                } else if val > heap[k - 1].0 {
                                    let pos = heap.partition_point(|(v, _)| *v > val);
                                    heap.insert(pos, (val, global_offset + i));
                                    heap.pop();
                                }
                            }
                        } else {
                            for i in 0..n {
                                if has_deletes && !del_bytes.is_empty() && (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                let val = $vals[i];
                                if heap.len() < k {
                                    let pos = heap.partition_point(|(v, _)| *v < val);
                                    heap.insert(pos, (val, global_offset + i));
                                } else if val < heap[k - 1].0 {
                                    let pos = heap.partition_point(|(v, _)| *v < val);
                                    heap.insert(pos, (val, global_offset + i));
                                    heap.pop();
                                }
                            }
                        }
                    }};
                }
                topk_scan2!(fvals);
            }
            global_offset += rg_rows;
        }
        Ok(Some(heap.into_iter().map(|(v, i)| (i, v)).collect()))
    }

    /// Compute numeric column aggregates directly from mmap without Arrow arrays.
    /// Returns (count, sum, min, max) for the specified column.
    /// Only works for Int64/Float64 columns in V4 mmap-only mode.
    pub fn compute_column_stats_mmap(&self, col_name: &str) -> io::Result<Option<(u64, f64, f64, f64)>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        let is_int = matches!(col_type, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
            ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
            ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date);
        let is_float = matches!(col_type, ColumnType::Float64 | ColumnType::Float32);
        if !is_int && !is_float { return Ok(None); }

        let col_count = schema.column_count();
        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for mmap agg"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        let mut total_count: u64 = 0;
        let mut total_sum: f64 = 0.0;
        let mut total_min: f64 = f64::INFINITY;
        let mut total_max: f64 = f64::NEG_INFINITY;

        for rg_meta in &footer.row_groups {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 { continue; }
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Err(err_data("RG extends past EOF")); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
            let mut pos = rg_rows * 8; // skip IDs
            let del_vec_len = (rg_rows + 7) / 8;
            if pos + del_vec_len > body.len() { return Err(err_data("RG del vec truncated")); }
            let del_bytes = &body[pos..pos + del_vec_len];
            let has_deletes = rg_meta.deletion_count > 0;
            pos += del_vec_len;

            let null_bitmap_len = (rg_rows + 7) / 8;
            for ci in 0..col_count {
                if pos + null_bitmap_len > body.len() { break; }
                let null_bytes = &body[pos..pos + null_bitmap_len];
                pos += null_bitmap_len;

                let ct = schema.columns[ci].1;
                if ci == col_idx {
                    let col_bytes = &body[pos..];
                    let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                    let encoding = if encoding_version >= 1 { col_bytes[0] } else { COL_ENCODING_PLAIN };

                    if encoding == COL_ENCODING_PLAIN {
                        let data = &col_bytes[enc_offset..];
                        if data.len() >= 8 {
                            let count = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
                            let values_start = 8usize;
                            if is_int {
                                for i in 0..count.min(rg_rows) {
                                    if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    let off = values_start + i * 8;
                                    if off + 8 > data.len() { break; }
                                    let v = i64::from_le_bytes(data[off..off+8].try_into().unwrap()) as f64;
                                    total_count += 1;
                                    total_sum += v;
                                    if v < total_min { total_min = v; }
                                    if v > total_max { total_max = v; }
                                }
                            } else {
                                for i in 0..count.min(rg_rows) {
                                    if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    let off = values_start + i * 8;
                                    if off + 8 > data.len() { break; }
                                    let v = f64::from_le_bytes(data[off..off+8].try_into().unwrap());
                                    if !v.is_nan() {
                                        total_count += 1;
                                        total_sum += v;
                                        if v < total_min { total_min = v; }
                                        if v > total_max { total_max = v; }
                                    }
                                }
                            }
                        }
                    } else {
                        // Encoded column: fallback to full decode
                        let (col_data, _) = if encoding_version >= 1 {
                            read_column_encoded(col_bytes, ct)?
                        } else {
                            ColumnData::from_bytes_typed(col_bytes, ct)?
                        };
                        match &col_data {
                            ColumnData::Int64(vals) => {
                                for (i, &v) in vals.iter().enumerate() {
                                    if has_deletes && i < rg_rows && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if i < rg_rows && (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    let fv = v as f64;
                                    total_count += 1;
                                    total_sum += fv;
                                    if fv < total_min { total_min = fv; }
                                    if fv > total_max { total_max = fv; }
                                }
                            }
                            ColumnData::Float64(vals) => {
                                for (i, &v) in vals.iter().enumerate() {
                                    if has_deletes && i < rg_rows && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if i < rg_rows && (null_bytes[i / 8] >> (i % 8)) & 1 == 1 { continue; }
                                    if !v.is_nan() {
                                        total_count += 1;
                                        total_sum += v;
                                        if v < total_min { total_min = v; }
                                        if v > total_max { total_max = v; }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                // Skip column data to advance pos
                let consumed = if encoding_version >= 1 {
                    skip_column_encoded(&body[pos..], ct)?
                } else {
                    ColumnData::skip_bytes_typed(&body[pos..], ct)?
                };
                pos += consumed;
            }
        }

        if total_count == 0 {
            return Ok(Some((0, 0.0, 0.0, 0.0)));
        }
        Ok(Some((total_count, total_sum, total_min, total_max)))
    }

    /// Scan multiple predicates in parallel on a single shared mmap (one lock acquisition).
    /// Each predicate targets a different column and is scanned independently via Rayon.
    /// Results are merged and deduplicated into a single sorted index vector.
    pub fn scan_multi_predicates_parallel(
        &self,
        predicates: &[MmapScanPred],
    ) -> io::Result<Option<Vec<usize>>> {
        use rayon::prelude::*;

        if predicates.is_empty() { return Ok(Some(Vec::new())); }

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;

        // Pre-resolve column indices and validate types
        struct PredDesc {
            col_idx: usize,
            is_int: bool,
            is_float: bool,
            is_string: bool,
            is_dict: bool,
        }
        let mut descs: Vec<Option<PredDesc>> = Vec::with_capacity(predicates.len());
        for pred in predicates {
            let col_name = match pred {
                MmapScanPred::NumericRange { col, .. } => *col,
                MmapScanPred::StringEq { col, .. } => *col,
                MmapScanPred::NumericIn { col, .. } => *col,
                MmapScanPred::StringIn { col, .. } => *col,
            };
            let col_idx = match schema.get_index(col_name) {
                Some(i) => i,
                None => { descs.push(None); continue; }
            };
            let ct = schema.columns[col_idx].1;
            let is_int = matches!(ct, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
                ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
                ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date);
            let is_float = matches!(ct, ColumnType::Float64 | ColumnType::Float32);
            let is_string = matches!(ct, ColumnType::String);
            let is_dict = matches!(ct, ColumnType::StringDict);
            descs.push(Some(PredDesc { col_idx, is_int, is_float, is_string, is_dict }));
        }

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for multi-pred scan"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        // Cast mmap pointer for safe sharing across rayon threads (read-only, mmap_guard keeps it alive)
        let mmap_ptr: usize = mmap_ref.as_ptr() as usize;
        let mmap_len: usize = mmap_ref.len();

        // Pre-build per-RG descriptors
        struct RgInfo {
            offset: usize, data_size: usize, row_count: usize,
            global_off: usize, deletion_count: u64,
        }
        let mut rg_infos: Vec<RgInfo> = Vec::with_capacity(footer.row_groups.len());
        let mut cumul = 0usize;
        for rg in &footer.row_groups {
            rg_infos.push(RgInfo {
                offset: rg.offset as usize, data_size: rg.data_size as usize,
                row_count: rg.row_count as usize, global_off: cumul,
                deletion_count: rg.deletion_count as u64,
            });
            cumul += rg.row_count as usize;
        }

        // Scan each predicate in parallel
        let results: Vec<Vec<usize>> = predicates.par_iter().enumerate().map(|(pi, pred)| {
            let desc = match &descs[pi] {
                Some(d) => d,
                None => return vec![],
            };
            let mmap = unsafe { std::slice::from_raw_parts(mmap_ptr as *const u8, mmap_len) };
            let col_idx = desc.col_idx;
            let mut matches: Vec<usize> = Vec::new();

            for (rg_i, rg) in rg_infos.iter().enumerate() {
                let rg_rows = rg.row_count;
                if rg_rows == 0 { continue; }

                // Zone map pruning
                if rg_i < footer.zone_maps.len() {
                    if let Some(zm) = footer.zone_maps[rg_i].iter().find(|z| z.col_idx as usize == col_idx) {
                        let skip = match pred {
                            MmapScanPred::NumericRange { low, high, .. } => {
                                if zm.is_float { !zm.may_overlap_float_range(*low, *high) }
                                else { !zm.may_overlap_int_range(low.ceil() as i64, high.floor() as i64) }
                            }
                            MmapScanPred::NumericIn { values, .. } => {
                                if let (Some(&mn), Some(&mx)) = (values.iter().min(), values.iter().max()) {
                                    if zm.is_float { !zm.may_overlap_float_range(mn as f64, mx as f64) }
                                    else { !zm.may_overlap_int_range(mn, mx) }
                                } else { false }
                            }
                            MmapScanPred::StringEq { value, .. } => {
                                if !zm.is_float {
                                    let tlen = value.len() as i64;
                                    tlen < zm.min_bits || tlen > zm.max_bits
                                } else { false }
                            }
                            MmapScanPred::StringIn { values, .. } => {
                                if !zm.is_float && !values.is_empty() {
                                    let min_len = values.iter().map(|s| s.len()).min().unwrap() as i64;
                                    let max_len = values.iter().map(|s| s.len()).max().unwrap() as i64;
                                    max_len < zm.min_bits || min_len > zm.max_bits
                                } else { false }
                            }
                        };
                        if skip { continue; }
                    }
                }

                let rg_end = rg.offset + rg.data_size;
                if rg_end > mmap.len() || rg_end < rg.offset + 32 { continue; }
                let rg_bytes = &mmap[rg.offset..rg_end];
                let compress_flag = rg_bytes[28];
                let encoding_version = rg_bytes[29];
                if compress_flag != RG_COMPRESS_NONE { continue; } // skip compressed RGs in parallel path
                let body = &rg_bytes[32..];
                let null_bitmap_len = (rg_rows + 7) / 8;
                let del_start = rg_rows * 8;
                let has_deletes = rg.deletion_count > 0;

                // RCIX required for parallel path
                let rcix = footer.col_offsets.get(rg_i).filter(|v| v.len() > col_idx);
                let rcix = match rcix { Some(r) => r, None => continue };
                let col_off = rcix[col_idx] as usize;
                if col_off + null_bitmap_len > body.len() { continue; }
                let null_bytes = &body[col_off..col_off + null_bitmap_len];
                let col_bytes = &body[col_off + null_bitmap_len..];
                let enc_offset = if encoding_version >= 1 { 1usize } else { 0 };
                let encoding = if encoding_version >= 1 && !col_bytes.is_empty() { col_bytes[0] } else { COL_ENCODING_PLAIN };
                if encoding != COL_ENCODING_PLAIN { continue; }
                let payload = if enc_offset <= col_bytes.len() { &col_bytes[enc_offset..] } else { continue; };

                match pred {
                    MmapScanPred::NumericRange { low, high, .. } if desc.is_int && payload.len() >= 8 => {
                        let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                        let n = count.min(rg_rows).min((payload.len() - 8) / 8);
                        let vals = bytes_as_i64_slice(&payload[8..], n);
                        let low_i = low.ceil() as i64;
                        let high_i = high.floor() as i64;
                        for i in 0..n {
                            if has_deletes && del_start + null_bitmap_len <= body.len()
                                && (body[del_start + i/8] >> (i%8)) & 1 == 1 { continue; }
                            if (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            if vals[i] >= low_i && vals[i] <= high_i { matches.push(rg.global_off + i); }
                        }
                    }
                    MmapScanPred::NumericRange { low, high, .. } if desc.is_float && payload.len() >= 8 => {
                        let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                        let n = count.min(rg_rows).min((payload.len() - 8) / 8);
                        let vals = bytes_as_f64_slice(&payload[8..], n);
                        for i in 0..n {
                            if has_deletes && del_start + null_bitmap_len <= body.len()
                                && (body[del_start + i/8] >> (i%8)) & 1 == 1 { continue; }
                            if (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            if vals[i] >= *low && vals[i] <= *high { matches.push(rg.global_off + i); }
                        }
                    }
                    MmapScanPred::NumericIn { values, .. } if desc.is_int && payload.len() >= 8 => {
                        let value_set: std::collections::HashSet<i64> = values.iter().copied().collect();
                        let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                        let n = count.min(rg_rows).min((payload.len() - 8) / 8);
                        let vals = bytes_as_i64_slice(&payload[8..], n);
                        for i in 0..n {
                            if has_deletes && del_start + null_bitmap_len <= body.len()
                                && (body[del_start + i/8] >> (i%8)) & 1 == 1 { continue; }
                            if (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            if value_set.contains(&vals[i]) { matches.push(rg.global_off + i); }
                        }
                    }
                    MmapScanPred::StringEq { value, .. } if desc.is_dict && payload.len() >= 16 => {
                        let target_bytes = value.as_bytes();
                        let row_count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                        let dict_size = u64::from_le_bytes(payload[8..16].try_into().unwrap()) as usize;
                        if dict_size == 0 { continue; }
                        let indices_start = 16usize;
                        let dict_off_start = indices_start + row_count * 4;
                        let dict_data_len_off = dict_off_start + dict_size * 4;
                        if dict_data_len_off + 8 > payload.len() { continue; }
                        let dict_data_len = u64::from_le_bytes(payload[dict_data_len_off..dict_data_len_off+8].try_into().unwrap()) as usize;
                        let dict_data_start = dict_data_len_off + 8;

                        let dict_offsets = bytes_as_u32_slice(&payload[dict_off_start..], dict_size);
                        let indices = bytes_as_u32_slice(&payload[indices_start..], row_count);

                        // Find target in dictionary
                        let raw_end = (dict_data_start + dict_data_len).min(payload.len());
                        let raw_dict = &payload[dict_data_start..raw_end];
                        let finder = memchr::memmem::Finder::new(target_bytes);
                        let target_len = target_bytes.len();
                        let mut target_dict_idx: Option<u32> = None;
                        let mut search_from = 0usize;
                        while let Some(rel) = finder.find(&raw_dict[search_from..]) {
                            let abs = search_from + rel;
                            if let Ok(di) = dict_offsets.binary_search(&(abs as u32)) {
                                let de = if di + 1 < dict_size { dict_offsets[di+1] as usize } else { dict_data_len };
                                if de - abs == target_len {
                                    target_dict_idx = Some((di + 1) as u32);
                                    break;
                                }
                            }
                            search_from += rel + 1;
                            if search_from >= raw_dict.len() { break; }
                        }

                        if let Some(tdi) = target_dict_idx {
                            let n = row_count.min(rg_rows);
                            if !has_deletes {
                                for i in 0..n {
                                    if indices[i] == tdi { matches.push(rg.global_off + i); }
                                }
                            } else if del_start + null_bitmap_len <= body.len() {
                                let del_bytes = &body[del_start..del_start + null_bitmap_len];
                                for i in 0..n {
                                    if (del_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                                    if indices[i] == tdi { matches.push(rg.global_off + i); }
                                }
                            }
                        }
                    }
                    MmapScanPred::StringEq { value, .. } if desc.is_string && payload.len() >= 8 => {
                        let target_bytes = value.as_bytes();
                        let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                        let off_start = 8usize;
                        let data_len_off = off_start + (count + 1) * 4;
                        if data_len_off + 8 > payload.len() { continue; }
                        let data_start = data_len_off + 8;
                        let n = count.min(rg_rows);
                        for i in 0..n {
                            if has_deletes && del_start + null_bitmap_len <= body.len()
                                && (body[del_start + i/8] >> (i%8)) & 1 == 1 { continue; }
                            if (null_bytes[i/8] >> (i%8)) & 1 == 1 { continue; }
                            let s_off = off_start + i * 4;
                            let e_off = s_off + 4;
                            if e_off + 4 > payload.len() { continue; }
                            let s = u32::from_le_bytes(payload[s_off..s_off+4].try_into().unwrap()) as usize;
                            let e = u32::from_le_bytes(payload[e_off..e_off+4].try_into().unwrap()) as usize;
                            if data_start + e <= payload.len() && e - s == target_bytes.len()
                                && &payload[data_start+s..data_start+e] == target_bytes
                            {
                                matches.push(rg.global_off + i);
                            }
                        }
                    }
                    MmapScanPred::StringIn { values, .. } if desc.is_dict || desc.is_string => {
                        // Delegate: scan each value and merge
                        for val in *values {
                            let sub = MmapScanPred::StringEq { col: match pred { MmapScanPred::StringIn { col, .. } => col, _ => unreachable!() }, value: val.as_str() };
                            // Inline: we can't recurse, so just note — StringIn with multiple values
                            // is handled by the caller splitting into multiple StringEq predicates.
                            // For now, skip (this path is rarely used in OR decomposition).
                            let _ = sub;
                        }
                    }
                    _ => {} // unsupported predicate/type combo: skip
                }
            }
            matches
        }).collect();

        drop(mmap_guard);
        drop(file_guard);

        let mut all_indices: Vec<usize> = Vec::with_capacity(results.iter().map(|r| r.len()).sum());
        for r in results {
            all_indices.extend(r);
        }
        all_indices.sort_unstable();
        all_indices.dedup();
        Ok(Some(all_indices))
    }

    /// Extract specific rows by global indices from mmap, returning an Arrow RecordBatch.
    /// Navigates each RG body once and extracts only values at target positions.
    /// Uses typed ColBuf: StringBuilder for String/StringDict (zero heap alloc per value),
    /// Vec<Option<i64/f64/bool>> for scalars (no Value enum boxing).
    pub fn extract_rows_by_indices_to_arrow(&self, indices: &[usize], col_refs: Option<&[&str]>) -> io::Result<Option<arrow::record_batch::RecordBatch>> {
        use arrow::array::{ArrayRef, Int64Array, Float64Array, StringBuilder, BooleanArray, StringArray};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType};
        use std::sync::Arc;

        if indices.is_empty() {
            return Ok(Some(arrow::record_batch::RecordBatch::new_empty(Arc::new(Schema::empty()))));
        }

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_count = schema.column_count();

        // Column projection: build a mask of which columns to actually extract
        let col_needed: Vec<bool> = if let Some(refs) = col_refs {
            schema.columns.iter().map(|(name, _)| refs.iter().any(|r| r.eq_ignore_ascii_case(name))).collect()
        } else {
            vec![true; col_count]
        };

        // Build RG cumulative bounds (binary-search friendly)
        let mut cumulative = 0usize;
        let rg_bounds: Vec<(usize, usize)> = footer.row_groups.iter().map(|rg| {
            let s = cumulative; cumulative += rg.row_count as usize; (s, cumulative)
        }).collect();

        // Group indices by RG: (output_position, local_index_within_rg)
        // indices from scan_like_filter_mmap are sorted → out_idx is monotonically increasing
        let mut rg_local_indices: Vec<Vec<(usize, usize)>> = vec![Vec::new(); footer.row_groups.len()];
        for (out_idx, &global_idx) in indices.iter().enumerate() {
            let rg_i = rg_bounds.partition_point(|&(_, end)| end <= global_idx);
            if rg_i < footer.row_groups.len() {
                rg_local_indices[rg_i].push((out_idx, global_idx - rg_bounds[rg_i].0));
            }
        }

        let n_out = indices.len();
        let mut out_ids: Vec<i64> = vec![0i64; n_out];

        // Typed per-column storage — no Value enum boxing, no String heap alloc
        enum ColBuf {
            I64(Vec<Option<i64>>),
            F64(Vec<Option<f64>>),
            Str(StringBuilder),        // String/StringDict: sequential append, zero alloc
            Bool(Vec<Option<bool>>),
            Bin(Vec<Option<Vec<u8>>>),  // Binary columns: preserve raw bytes
            FixedVec(Vec<Option<Vec<u8>>>, u32), // FixedList/Float16List: raw f32 bytes per row + dim
            Other(Vec<Option<crate::data::Value>>), // rare fallback
        }
        let mut col_bufs: Vec<ColBuf> = schema.columns.iter().enumerate().map(|(ci, (_, ct))| {
            if !col_needed[ci] {
                return ColBuf::I64(Vec::new()); // placeholder, never filled
            }
            match ct {
                ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
                ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 |
                ColumnType::Timestamp | ColumnType::Date => ColBuf::I64(vec![None; n_out]),
                ColumnType::Float64 | ColumnType::Float32 => ColBuf::F64(vec![None; n_out]),
                ColumnType::String | ColumnType::StringDict =>
                    ColBuf::Str(StringBuilder::with_capacity(n_out, n_out * 10)),
                ColumnType::Bool => ColBuf::Bool(vec![None; n_out]),
                ColumnType::Binary => ColBuf::Bin(vec![None; n_out]),
                ColumnType::FixedList | ColumnType::Float16List => ColBuf::FixedVec(vec![None; n_out], 0),
                _ => ColBuf::Other(vec![None; n_out]),
            }
        }).collect();

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for batch extract"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        // ── PARALLEL COLUMN EXTRACTION ────────────────────────────────────
        // Pre-compute column offsets for each RG (from RCIX or by scanning).
        // Then process each column independently in parallel via rayon.
        let needed_col_indices: Vec<usize> = (0..col_count).filter(|&ci| col_needed[ci]).collect();
        let mut par_col_offsets: Vec<Option<Vec<u32>>> = Vec::new(); // per-RG col offsets
        let mut par_eligible = needed_col_indices.len() >= 2 && n_out >= 500;
        if par_eligible {
            for (rg_i, local_pairs) in rg_local_indices.iter().enumerate() {
                if local_pairs.is_empty() {
                    par_col_offsets.push(None); // no rows needed
                    continue;
                }
                if rg_i >= footer.row_groups.len() { par_eligible = false; break; }
                let rg_meta = &footer.row_groups[rg_i];
                let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                if rg_end > mmap_ref.len() { par_eligible = false; break; }
                let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
                if rg_bytes.len() < 32 { par_eligible = false; break; }
                let compress_flag = rg_bytes[28];
                let enc_ver = rg_bytes[29];
                if compress_flag != RG_COMPRESS_NONE || enc_ver < 1 {
                    par_eligible = false; break;
                }
                // Try RCIX first
                if rg_i < footer.col_offsets.len() && footer.col_offsets[rg_i].len() >= col_count {
                    par_col_offsets.push(Some(footer.col_offsets[rg_i].clone()));
                } else {
                    // Compute offsets by scanning through columns
                    let body = &rg_bytes[32..];
                    let rg_rows = rg_meta.row_count as usize;
                    let null_bm_len = (rg_rows + 7) / 8;
                    let mut offsets = Vec::with_capacity(col_count);
                    let mut pos = rg_rows * 8 + null_bm_len; // skip ID block + _id null bitmap
                    let mut ok = true;
                    for ci in 0..col_count {
                        offsets.push(pos as u32);
                        // Advance past null bitmap + encoded column data
                        pos += null_bm_len;
                        if pos > body.len() { ok = false; break; }
                        match skip_column_encoded(&body[pos..], schema.columns[ci].1) {
                            Ok(consumed) => pos += consumed,
                            Err(_) => { ok = false; break; }
                        }
                    }
                    if !ok || offsets.len() < col_count { par_eligible = false; break; }
                    par_col_offsets.push(Some(offsets));
                }
            }
        }

        if par_eligible {
            use rayon::prelude::*;
            // Extract IDs
            for (rg_i, local_pairs) in rg_local_indices.iter().enumerate() {
                if local_pairs.is_empty() { continue; }
                let rg_meta = &footer.row_groups[rg_i];
                let rg_rows = rg_meta.row_count as usize;
                let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
                let body = &rg_bytes[32..];
                if rg_rows * 8 <= body.len() {
                    for &(out_idx, local_idx) in local_pairs {
                        let id_off = local_idx * 8;
                        if id_off + 8 <= rg_rows * 8 {
                            out_ids[out_idx] = i64::from_le_bytes(body[id_off..id_off+8].try_into().unwrap());
                        }
                    }
                }
            }

            let mmap_ptr = mmap_ref.as_ptr() as usize;
            let mmap_len = mmap_ref.len();

            // Each parallel task extracts data AND builds the Arrow array in one pass.
            let col_arrays: Vec<(usize, Field, ArrayRef)> = needed_col_indices.par_iter().map(|&ci| {
                let mmap = unsafe { std::slice::from_raw_parts(mmap_ptr as *const u8, mmap_len) };
                let ct = schema.columns[ci].1;
                let col_name = &schema.columns[ci].0;

                macro_rules! for_each_rg {
                    ($handler:expr) => {{
                        for (rg_i, local_pairs) in rg_local_indices.iter().enumerate() {
                            if local_pairs.is_empty() { continue; }
                            let offsets = match &par_col_offsets[rg_i] {
                                Some(o) => o,
                                None => continue,
                            };
                            let rg_meta = &footer.row_groups[rg_i];
                            let rg_rows = rg_meta.row_count as usize;
                            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                            if rg_end > mmap.len() { continue; }
                            let rg_bytes = &mmap[rg_meta.offset as usize..rg_end];
                            let body = &rg_bytes[32..];
                            let null_bitmap_len = (rg_rows + 7) / 8;
                            let col_off = offsets[ci] as usize;
                            if col_off + null_bitmap_len > body.len() { continue; }
                            let null_bytes = &body[col_off..col_off + null_bitmap_len];
                            let col_bytes = &body[col_off + null_bitmap_len..];
                            if col_bytes.is_empty() { continue; }
                            let encoding = col_bytes[0];
                            let payload = &col_bytes[1..];
                            #[allow(clippy::redundant_closure_call)]
                            ($handler)(local_pairs, null_bytes, encoding, payload, rg_rows);
                        }
                    }};
                }

                match ct {
                    ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
                    ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 |
                    ColumnType::Timestamp | ColumnType::Date => {
                        let mut vals: Vec<Option<i64>> = vec![None; n_out];
                        for_each_rg!(|pairs: &[(usize, usize)], null_bytes: &[u8], encoding: u8, payload: &[u8], _rg_rows: usize| {
                            if encoding == COL_ENCODING_PLAIN && payload.len() >= 8 {
                                for &(out_idx, local_idx) in pairs {
                                    if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                    let off = 8 + local_idx * 8;
                                    if off + 8 <= payload.len() {
                                        vals[out_idx] = Some(i64::from_le_bytes(payload[off..off+8].try_into().unwrap()));
                                    }
                                }
                            } else if encoding == 2 {
                                for &(out_idx, local_idx) in pairs {
                                    if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                    if let Some(v) = crate::storage::on_demand::bitpack_decode_at_idx(payload, local_idx) {
                                        vals[out_idx] = Some(v);
                                    }
                                }
                            }
                        });
                        (ci, Field::new(col_name, ArrowDataType::Int64, true),
                         Arc::new(Int64Array::from(vals)) as ArrayRef)
                    }
                    ColumnType::Float64 | ColumnType::Float32 => {
                        let mut vals: Vec<Option<f64>> = vec![None; n_out];
                        for_each_rg!(|pairs: &[(usize, usize)], null_bytes: &[u8], encoding: u8, payload: &[u8], _rg_rows: usize| {
                            if encoding == COL_ENCODING_PLAIN && payload.len() >= 8 {
                                for &(out_idx, local_idx) in pairs {
                                    if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                    let off = 8 + local_idx * 8;
                                    if off + 8 <= payload.len() {
                                        vals[out_idx] = Some(f64::from_le_bytes(payload[off..off+8].try_into().unwrap()));
                                    }
                                }
                            }
                        });
                        (ci, Field::new(col_name, ArrowDataType::Float64, true),
                         Arc::new(Float64Array::from(vals)) as ArrayRef)
                    }
                    ColumnType::StringDict => {
                        let mut builder = StringBuilder::with_capacity(n_out, n_out * 16);
                        // Pre-fill with nulls; we'll overwrite non-null slots
                        for _ in 0..n_out { builder.append_null(); }
                        // Extract: collect (out_idx, start, end) ranges into mmap
                        let mut ranges: Vec<(usize, usize, usize)> = Vec::new();
                        for_each_rg!(|pairs: &[(usize, usize)], null_bytes: &[u8], encoding: u8, payload: &[u8], rg_rows: usize| {
                            if encoding != COL_ENCODING_PLAIN || payload.len() < 16 { return; }
                            let row_count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                            let dict_size = u64::from_le_bytes(payload[8..16].try_into().unwrap()) as usize;
                            let indices_start = 16usize;
                            let dict_off_start = indices_start + row_count * 4;
                            let dict_data_len_off = dict_off_start + dict_size * 4;
                            if dict_data_len_off + 8 > payload.len() { return; }
                            let dict_data_len = u64::from_le_bytes(payload[dict_data_len_off..dict_data_len_off+8].try_into().unwrap()) as usize;
                            let dict_data_start = dict_data_len_off + 8;
                            let payload_abs = payload.as_ptr() as usize - mmap_ptr;
                            for &(out_idx, local_idx) in pairs {
                                if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                if local_idx >= row_count { continue; }
                                let idx_off = indices_start + local_idx * 4;
                                if idx_off + 4 > payload.len() { continue; }
                                let dict_idx = u32::from_le_bytes(payload[idx_off..idx_off+4].try_into().unwrap());
                                if dict_idx == 0 { continue; }
                                let di = (dict_idx - 1) as usize;
                                if di >= dict_size { continue; }
                                let ds_off = dict_off_start + di * 4;
                                if ds_off + 4 > payload.len() { continue; }
                                let ds = u32::from_le_bytes(payload[ds_off..ds_off+4].try_into().unwrap()) as usize;
                                let de = if di + 1 < dict_size {
                                    let de_off = ds_off + 4;
                                    if de_off + 4 <= payload.len() { u32::from_le_bytes(payload[de_off..de_off+4].try_into().unwrap()) as usize } else { dict_data_len }
                                } else { dict_data_len };
                                if dict_data_start + de <= payload.len() {
                                    ranges.push((out_idx, payload_abs + dict_data_start + ds, payload_abs + dict_data_start + de));
                                }
                            }
                        });
                        // Build StringArray: null-prefilled builder, overwrite with values
                        // More efficient: just build from Option<&str> vec
                        let mut strs: Vec<Option<&str>> = vec![None; n_out];
                        for &(idx, s, e) in &ranges {
                            strs[idx] = Some(std::str::from_utf8(&mmap[s..e]).unwrap_or(""));
                        }
                        let arr: StringArray = strs.into_iter().collect();
                        (ci, Field::new(col_name, ArrowDataType::Utf8, true), Arc::new(arr) as ArrayRef)
                    }
                    ColumnType::String => {
                        let mut ranges: Vec<(usize, usize, usize)> = Vec::new();
                        for_each_rg!(|pairs: &[(usize, usize)], null_bytes: &[u8], encoding: u8, payload: &[u8], rg_rows: usize| {
                            if encoding != COL_ENCODING_PLAIN || payload.len() < 8 { return; }
                            let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                            let data_len_off = 8 + (count + 1) * 4;
                            if data_len_off + 8 > payload.len() { return; }
                            let data_start = data_len_off + 8;
                            let n = count.min(rg_rows);
                            let payload_abs = payload.as_ptr() as usize - mmap_ptr;
                            for &(out_idx, local_idx) in pairs {
                                if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                if local_idx >= n { continue; }
                                let s_off = 8 + local_idx * 4;
                                let e_off = s_off + 4;
                                if e_off + 4 > payload.len() { continue; }
                                let s = u32::from_le_bytes(payload[s_off..s_off+4].try_into().unwrap()) as usize;
                                let e = u32::from_le_bytes(payload[e_off..e_off+4].try_into().unwrap()) as usize;
                                if data_start + e <= payload.len() {
                                    ranges.push((out_idx, payload_abs + data_start + s, payload_abs + data_start + e));
                                }
                            }
                        });
                        let mut strs: Vec<Option<&str>> = vec![None; n_out];
                        for &(idx, s, e) in &ranges {
                            strs[idx] = Some(std::str::from_utf8(&mmap[s..e]).unwrap_or(""));
                        }
                        let arr: StringArray = strs.into_iter().collect();
                        (ci, Field::new(col_name, ArrowDataType::Utf8, true), Arc::new(arr) as ArrayRef)
                    }
                    ColumnType::Bool => {
                        let mut vals: Vec<Option<bool>> = vec![None; n_out];
                        for_each_rg!(|pairs: &[(usize, usize)], null_bytes: &[u8], encoding: u8, payload: &[u8], _rg_rows: usize| {
                            if encoding != COL_ENCODING_PLAIN || payload.len() < 8 { return; }
                            for &(out_idx, local_idx) in pairs {
                                if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                let byte_off = 8 + local_idx / 8;
                                if byte_off < payload.len() {
                                    vals[out_idx] = Some((payload[byte_off] >> (local_idx % 8)) & 1 == 1);
                                }
                            }
                        });
                        let arr: BooleanArray = vals.into_iter().collect();
                        (ci, Field::new(col_name, ArrowDataType::Boolean, true), Arc::new(arr) as ArrayRef)
                    }
                    _ => {
                        // Binary / FixedVec / unknown: produce null Utf8 column (fallback path handles these)
                        let arr = StringArray::from(vec![None::<&str>; n_out]);
                        (ci, Field::new(col_name, ArrowDataType::Utf8, true), Arc::new(arr) as ArrayRef)
                    }
                }
            }).collect();

            // Assemble RecordBatch from parallel results
            let mut fields: Vec<Field> = Vec::with_capacity(col_count + 1);
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(col_count + 1);
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(out_ids)));
            for (_, field, arr) in col_arrays {
                fields.push(field);
                arrays.push(arr);
            }
            let schema_ref = Arc::new(Schema::new(fields));
            let batch = arrow::record_batch::RecordBatch::try_new(schema_ref, arrays)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            drop(mmap_guard);
            drop(file_guard);
            return Ok(Some(batch));
        }
        // ── END PARALLEL COLUMN EXTRACTION ─────────────────────────────────

        for (rg_i, local_pairs) in rg_local_indices.iter().enumerate() {
            if local_pairs.is_empty() { continue; }
            let rg_meta = &footer.row_groups[rg_i];
            let rg_rows = rg_meta.row_count as usize;
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Err(err_data("RG extends past EOF")); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize .. rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
            let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
            let null_bitmap_len = (rg_rows + 7) / 8;

            // Extract IDs for target rows (no mmap copy — slice read)
            if rg_rows * 8 <= body.len() {
                for &(out_idx, local_idx) in local_pairs {
                    let id_off = local_idx * 8;
                    if id_off + 8 <= rg_rows * 8 {
                        out_ids[out_idx] = i64::from_le_bytes(body[id_off..id_off+8].try_into().unwrap());
                    }
                }
            }

            // Use RCIX for O(1) per-column access when available
            let rcix = if compress_flag == RG_COMPRESS_NONE && encoding_version >= 1
                && rg_i < footer.col_offsets.len() && footer.col_offsets[rg_i].len() >= col_count
            { Some(&footer.col_offsets[rg_i]) } else { None };

            let mut pos = rg_rows * 8 + null_bitmap_len;

            for ci in 0..col_count {
                let ct = schema.columns[ci].1;
                // Column projection: skip columns not needed
                if !col_needed[ci] {
                    if rcix.is_none() {
                        // Sequential layout: must advance pos past null bitmap + column data
                        if pos + null_bitmap_len > body.len() { break; }
                        pos += null_bitmap_len;
                        let consumed = if encoding_version >= 1 {
                            skip_column_encoded(&body[pos..], ct)?
                        } else {
                            ColumnData::skip_bytes_typed(&body[pos..], ct)?
                        };
                        pos += consumed;
                    }
                    // RCIX: no pos tracking needed, just skip
                    continue;
                }
                let (null_bytes, col_bytes) = if let Some(rcix) = rcix {
                    let col_off = rcix[ci] as usize;
                    if col_off + null_bitmap_len > body.len() {
                        // Column not present: append nulls for Str buffers (maintain length invariant)
                        if let ColBuf::Str(ref mut b) = col_bufs[ci] {
                            for _ in local_pairs { b.append_null(); }
                        }
                        continue;
                    }
                    (&body[col_off..col_off + null_bitmap_len], &body[col_off + null_bitmap_len..])
                } else {
                    if pos + null_bitmap_len > body.len() { break; }
                    let nb = &body[pos..pos + null_bitmap_len];
                    pos += null_bitmap_len;
                    (nb, &body[pos..])
                };
                let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                let encoding = if encoding_version >= 1 && !col_bytes.is_empty() { col_bytes[0] } else { COL_ENCODING_PLAIN };
                let data_bytes = if enc_offset <= col_bytes.len() { &col_bytes[enc_offset..] } else { &[] as &[u8] };

                // Fast typed extraction — no Value boxing
                let extracted = match (encoding, ct, &mut col_bufs[ci]) {
                    // Plain Int64-compatible
                    (COL_ENCODING_PLAIN, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
                     ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
                     ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date, ColBuf::I64(vals))
                    if data_bytes.len() >= 8 => {
                        for &(out_idx, local_idx) in local_pairs {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                            let off = 8 + local_idx * 8;
                            if off + 8 <= data_bytes.len() {
                                vals[out_idx] = Some(i64::from_le_bytes(data_bytes[off..off+8].try_into().unwrap()));
                            }
                        }
                        true
                    }
                    // Plain Float64/Float32
                    (COL_ENCODING_PLAIN, ColumnType::Float64 | ColumnType::Float32, ColBuf::F64(vals))
                    if data_bytes.len() >= 8 => {
                        for &(out_idx, local_idx) in local_pairs {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                            let off = 8 + local_idx * 8;
                            if off + 8 <= data_bytes.len() {
                                vals[out_idx] = Some(f64::from_le_bytes(data_bytes[off..off+8].try_into().unwrap()));
                            }
                        }
                        true
                    }
                    // Plain String — zero-copy slice to StringBuilder
                    (COL_ENCODING_PLAIN, ColumnType::String, ColBuf::Str(b))
                    if data_bytes.len() >= 8 => {
                        let count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
                        let data_len_off = 8 + (count + 1) * 4;
                        if data_len_off + 8 <= data_bytes.len() {
                            let data_start = data_len_off + 8;
                            for &(_, local_idx) in local_pairs {
                                if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 {
                                    b.append_null(); continue;
                                }
                                if local_idx >= count { b.append_null(); continue; }
                                let s_off = 8 + local_idx * 4;
                                let e_off = s_off + 4;
                                if e_off + 4 <= data_bytes.len() {
                                    let s = u32::from_le_bytes(data_bytes[s_off..s_off+4].try_into().unwrap()) as usize;
                                    let e = u32::from_le_bytes(data_bytes[e_off..e_off+4].try_into().unwrap()) as usize;
                                    if data_start + e <= data_bytes.len() {
                                        b.append_value(std::str::from_utf8(&data_bytes[data_start+s..data_start+e]).unwrap_or(""));
                                    } else { b.append_null(); }
                                } else { b.append_null(); }
                            }
                            true
                        } else { false }
                    }
                    // Plain StringDict — zero-copy dict lookup to StringBuilder
                    (COL_ENCODING_PLAIN, ColumnType::StringDict, ColBuf::Str(b))
                    if data_bytes.len() >= 16 => {
                        let row_count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
                        let dict_size = u64::from_le_bytes(data_bytes[8..16].try_into().unwrap()) as usize;
                        let indices_start = 16usize;
                        let dict_off_start = indices_start + row_count * 4;
                        let dict_data_len_off = dict_off_start + dict_size * 4;
                        if dict_data_len_off + 8 <= data_bytes.len() {
                            let dict_data_len = u64::from_le_bytes(data_bytes[dict_data_len_off..dict_data_len_off+8].try_into().unwrap()) as usize;
                            let dict_data_start = dict_data_len_off + 8;
                            for &(_, local_idx) in local_pairs {
                                if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 {
                                    b.append_null(); continue;
                                }
                                if local_idx >= row_count { b.append_null(); continue; }
                                let idx_off = indices_start + local_idx * 4;
                                if idx_off + 4 > data_bytes.len() { b.append_null(); continue; }
                                let dict_idx = u32::from_le_bytes(data_bytes[idx_off..idx_off+4].try_into().unwrap());
                                if dict_idx == 0 { b.append_null(); continue; }
                                let di = (dict_idx - 1) as usize;
                                if di >= dict_size { b.append_null(); continue; }
                                let ds_off = dict_off_start + di * 4;
                                if ds_off + 4 > data_bytes.len() { b.append_null(); continue; }
                                let ds = u32::from_le_bytes(data_bytes[ds_off..ds_off+4].try_into().unwrap()) as usize;
                                let de = if di + 1 < dict_size {
                                    let de_off = ds_off + 4;
                                    if de_off + 4 <= data_bytes.len() {
                                        u32::from_le_bytes(data_bytes[de_off..de_off+4].try_into().unwrap()) as usize
                                    } else { dict_data_len }
                                } else { dict_data_len };
                                if dict_data_start + de <= data_bytes.len() {
                                    b.append_value(std::str::from_utf8(&data_bytes[dict_data_start+ds..dict_data_start+de]).unwrap_or(""));
                                } else { b.append_null(); }
                            }
                            true
                        } else { false }
                    }
                    // Bitpack Int64-compatible
                    (2u8, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
                     ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
                     ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date, ColBuf::I64(vals)) => {
                        for &(out_idx, local_idx) in local_pairs {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                            if let Some(v) = crate::storage::on_demand::bitpack_decode_at_idx(data_bytes, local_idx) {
                                vals[out_idx] = Some(v);
                            }
                        }
                        true
                    }
                    // Plain Bool
                    (COL_ENCODING_PLAIN, ColumnType::Bool, ColBuf::Bool(vals))
                    if data_bytes.len() >= 8 => {
                        for &(out_idx, local_idx) in local_pairs {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                            let byte_off = 8 + local_idx / 8;
                            if byte_off < data_bytes.len() {
                                vals[out_idx] = Some((data_bytes[byte_off] >> (local_idx % 8)) & 1 == 1);
                            }
                        }
                        true
                    }
                    _ => false,
                };

                if !extracted {
                    // Fallback: decode full column and pick values at target indices
                    let (col_data, _) = if encoding_version >= 1 {
                        read_column_encoded(col_bytes, ct)?
                    } else {
                        ColumnData::from_bytes_typed(col_bytes, ct)?
                    };
                    match &mut col_bufs[ci] {
                        ColBuf::I64(vals) => {
                            if let ColumnData::Int64(v) = &col_data {
                                for &(out_idx, local_idx) in local_pairs {
                                    if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                    if local_idx < v.len() { vals[out_idx] = Some(v[local_idx]); }
                                }
                            }
                        }
                        ColBuf::F64(vals) => {
                            if let ColumnData::Float64(v) = &col_data {
                                for &(out_idx, local_idx) in local_pairs {
                                    if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                    if local_idx < v.len() { vals[out_idx] = Some(v[local_idx]); }
                                }
                            }
                        }
                        ColBuf::Bool(vals) => {
                            if let ColumnData::Bool { data, len } = &col_data {
                                for &(out_idx, local_idx) in local_pairs {
                                    if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                    if local_idx < *len { vals[out_idx] = Some((data[local_idx/8] >> (local_idx%8)) & 1 == 1); }
                                }
                            }
                        }
                        ColBuf::Str(b) => {
                            match &col_data {
                                ColumnData::String { offsets, data } => {
                                    let cnt = offsets.len().saturating_sub(1);
                                    for &(_, local_idx) in local_pairs {
                                        if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { b.append_null(); continue; }
                                        if local_idx < cnt {
                                            let s = offsets[local_idx] as usize;
                                            let e = offsets[local_idx + 1] as usize;
                                            b.append_value(std::str::from_utf8(&data[s..e]).unwrap_or(""));
                                        } else { b.append_null(); }
                                    }
                                }
                                ColumnData::StringDict { indices: idx_arr, dict_offsets, dict_data, .. } => {
                                    for &(_, local_idx) in local_pairs {
                                        if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { b.append_null(); continue; }
                                        if local_idx < idx_arr.len() {
                                            let di = idx_arr[local_idx];
                                            if di == 0 { b.append_null(); continue; }
                                            let d = (di - 1) as usize;
                                            if d < dict_offsets.len() {
                                                let s = dict_offsets[d] as usize;
                                                let e = if d + 1 < dict_offsets.len() { dict_offsets[d+1] as usize } else { dict_data.len() };
                                                b.append_value(std::str::from_utf8(&dict_data[s..e.min(dict_data.len())]).unwrap_or(""));
                                            } else { b.append_null(); }
                                        } else { b.append_null(); }
                                    }
                                }
                                _ => { for _ in local_pairs { b.append_null(); } }
                            }
                        }
                        ColBuf::Bin(vals) => {
                            if let ColumnData::Binary { offsets, data } = &col_data {
                                let cnt = offsets.len().saturating_sub(1);
                                for &(out_idx, local_idx) in local_pairs {
                                    if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                    if local_idx < cnt {
                                        let s = offsets[local_idx] as usize;
                                        let e = offsets[local_idx + 1] as usize;
                                        vals[out_idx] = Some(data[s..e].to_vec());
                                    }
                                }
                            }
                        }
                        ColBuf::FixedVec(vals, ref mut dim_out) => {
                            match &col_data {
                                ColumnData::FixedList { data, dim } => {
                                    *dim_out = *dim;
                                    let d = *dim as usize;
                                    let row_bytes = d * 4;
                                    let row_count = if d == 0 { 0 } else { data.len() / row_bytes };
                                    for &(out_idx, local_idx) in local_pairs {
                                        if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                        if local_idx < row_count {
                                            let s = local_idx * row_bytes;
                                            vals[out_idx] = Some(data[s..s + row_bytes].to_vec());
                                        }
                                    }
                                }
                                ColumnData::Float16List { data, dim } => {
                                    *dim_out = *dim;
                                    let d = *dim as usize;
                                    let row_bytes_f16 = d * 2;
                                    let row_count = if d == 0 { 0 } else { data.len() / row_bytes_f16 };
                                    for &(out_idx, local_idx) in local_pairs {
                                        if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                        if local_idx < row_count {
                                            let s = local_idx * row_bytes_f16;
                                            let f16_bytes = &data[s..s + row_bytes_f16];
                                            let f32_bytes: Vec<u8> = f16_bytes.chunks_exact(2)
                                                .flat_map(|c| crate::storage::on_demand::f16_to_f32(
                                                    u16::from_le_bytes(c.try_into().unwrap())
                                                ).to_le_bytes())
                                                .collect();
                                            vals[out_idx] = Some(f32_bytes);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        ColBuf::Other(vals) => {
                            for &(out_idx, local_idx) in local_pairs {
                                if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                let val = match &col_data {
                                    ColumnData::Int64(v) => if local_idx < v.len() { Some(crate::data::Value::Int64(v[local_idx])) } else { None },
                                    ColumnData::Float64(v) => if local_idx < v.len() { Some(crate::data::Value::Float64(v[local_idx])) } else { None },
                                    ColumnData::String { offsets, data } => {
                                        let cnt = offsets.len().saturating_sub(1);
                                        if local_idx < cnt {
                                            let s = offsets[local_idx] as usize; let e = offsets[local_idx+1] as usize;
                                            Some(crate::data::Value::String(std::str::from_utf8(&data[s..e]).unwrap_or("").to_string()))
                                        } else { None }
                                    }
                                    _ => None,
                                };
                                if let Some(v) = val { vals[out_idx] = Some(v); }
                            }
                        }
                    }
                }

                if rcix.is_none() {
                    let consumed = if encoding_version >= 1 {
                        skip_column_encoded(&body[pos..], ct)?
                    } else {
                        ColumnData::skip_bytes_typed(&body[pos..], ct)?
                    };
                    pos += consumed;
                }
            }
        }

        drop(mmap_guard);
        drop(file_guard);

        // Build Arrow RecordBatch directly from typed ColBuf — no extra copies
        let mut fields: Vec<Field> = Vec::with_capacity(col_count + 1);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(col_count + 1);

        fields.push(Field::new("_id", ArrowDataType::Int64, false));
        arrays.push(Arc::new(Int64Array::from(out_ids)));

        for (ci, buf) in col_bufs.into_iter().enumerate() {
            if !col_needed[ci] { continue; }
            let col_name = &schema.columns[ci].0;
            let ct = schema.columns[ci].1;
            match buf {
                ColBuf::I64(vals) => {
                    fields.push(Field::new(col_name, ArrowDataType::Int64, true));
                    arrays.push(Arc::new(Int64Array::from(vals)));
                }
                ColBuf::F64(vals) => {
                    fields.push(Field::new(col_name, ArrowDataType::Float64, true));
                    arrays.push(Arc::new(Float64Array::from(vals)));
                }
                ColBuf::Str(mut b) => {
                    fields.push(Field::new(col_name, ArrowDataType::Utf8, true));
                    arrays.push(Arc::new(b.finish()) as ArrayRef);
                }
                ColBuf::Bool(vals) => {
                    let arr: BooleanArray = vals.into_iter().collect();
                    fields.push(Field::new(col_name, ArrowDataType::Boolean, true));
                    arrays.push(Arc::new(arr));
                }
                ColBuf::Bin(vals) => {
                    use arrow::array::BinaryArray;
                    let bin_data: Vec<Option<&[u8]>> = vals.iter()
                        .map(|v| v.as_deref())
                        .collect();
                    fields.push(Field::new(col_name, ArrowDataType::Binary, true));
                    arrays.push(Arc::new(BinaryArray::from(bin_data)) as ArrayRef);
                }
                ColBuf::FixedVec(vals, dim) => {
                    let d = dim as usize;
                    if d > 0 {
                        let mut all_f32: Vec<f32> = Vec::with_capacity(n_out * d);
                        let mut null_bits: Vec<bool> = Vec::with_capacity(n_out);
                        for v in &vals {
                            match v {
                                Some(bytes) if bytes.len() == d * 4 => {
                                    null_bits.push(true);
                                    for chunk in bytes.chunks_exact(4) {
                                        all_f32.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                                    }
                                }
                                _ => {
                                    null_bits.push(false);
                                    all_f32.extend(std::iter::repeat(0.0f32).take(d));
                                }
                            }
                        }
                        use arrow::array::{FixedSizeListArray, Float32Array};
                        use arrow::datatypes::Field as ArrowField;
                        let float_arr = Float32Array::from(all_f32);
                        let item_field = Arc::new(ArrowField::new("item", ArrowDataType::Float32, false));
                        let null_buf: Option<arrow::buffer::NullBuffer> = if null_bits.iter().any(|b| !b) {
                            Some(arrow::buffer::NullBuffer::from(null_bits.iter().map(|&b| b).collect::<Vec<bool>>()))
                        } else { None };
                        let list_arr = FixedSizeListArray::new(item_field.clone(), d as i32, Arc::new(float_arr), null_buf);
                        let list_dt = ArrowDataType::FixedSizeList(item_field, d as i32);
                        fields.push(Field::new(col_name, list_dt, true));
                        arrays.push(Arc::new(list_arr) as ArrayRef);
                    } else {
                        fields.push(Field::new(col_name, ArrowDataType::Utf8, true));
                        arrays.push(Arc::new(StringArray::from(vec![None::<&str>; n_out])) as ArrayRef);
                    }
                }
                ColBuf::Other(vals) => {
                    let mut b = StringBuilder::with_capacity(n_out, n_out * 8);
                    for v in vals {
                        match v {
                            Some(crate::data::Value::String(s)) => b.append_value(&s),
                            Some(crate::data::Value::Int64(n)) => b.append_value(&n.to_string()),
                            _ => b.append_null(),
                        }
                    }
                    let dt = match ct {
                        _ => ArrowDataType::Utf8,
                    };
                    fields.push(Field::new(col_name, dt, true));
                    arrays.push(Arc::new(b.finish()) as ArrayRef);
                }
            }
        }

        let batch_schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(batch_schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
            .map(Some)
    }

    pub(crate) fn extract_rows_by_indices_mmap_columns(
        &self,
        indices: &[usize],
        col_refs: Option<&[&str]>,
    ) -> io::Result<Option<MmapBatchColumns>> {
        if indices.is_empty() {
            return Ok(Some(MmapBatchColumns {
                row_count: 0,
                columns: Vec::new(),
            }));
        }

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_count = schema.column_count();

        let include_id = col_refs
            .map(|refs| refs.iter().any(|r| r.eq_ignore_ascii_case("_id")))
            .unwrap_or(true);
        let col_needed: Vec<bool> = if let Some(refs) = col_refs {
            schema
                .columns
                .iter()
                .map(|(name, _)| refs.iter().any(|r| r.eq_ignore_ascii_case(name)))
                .collect()
        } else {
            vec![true; col_count]
        };
        if let Some(refs) = col_refs {
            for requested in refs {
                if !requested.eq_ignore_ascii_case("_id")
                    && schema.get_index(requested).is_none()
                {
                    return Ok(None);
                }
            }
        }

        let mut cumulative = 0usize;
        let rg_bounds: Vec<(usize, usize)> = footer
            .row_groups
            .iter()
            .map(|rg| {
                let start = cumulative;
                cumulative += rg.row_count as usize;
                (start, cumulative)
            })
            .collect();

        let n_rows = indices.len();
        let mut rg_local_indices: Vec<Vec<(usize, usize)>> =
            vec![Vec::new(); footer.row_groups.len()];
        for (out_idx, &global_idx) in indices.iter().enumerate() {
            let rg_i = rg_bounds.partition_point(|&(_, end)| end <= global_idx);
            if rg_i < footer.row_groups.len() {
                rg_local_indices[rg_i].push((out_idx, global_idx - rg_bounds[rg_i].0));
            }
        }

        for (rg_i, local_pairs) in rg_local_indices.iter().enumerate() {
            if local_pairs.is_empty() {
                continue;
            }
            if rg_i >= footer.col_offsets.len() || footer.col_offsets[rg_i].len() < col_count {
                return Ok(None);
            }
        }

        enum ColBuf {
            I64(Vec<Option<i64>>),
            F64(Vec<Option<f64>>),
            Str(Vec<Option<String>>),
            Bool(Vec<Option<bool>>),
            Bin(Vec<Option<Vec<u8>>>),
        }

        let mut found_mask = vec![false; n_rows];
        let mut out_ids = vec![0i64; n_rows];
        let mut col_bufs: Vec<ColBuf> = schema
            .columns
            .iter()
            .enumerate()
            .map(|(ci, (_, ct))| {
                if !col_needed[ci] {
                    return ColBuf::I64(Vec::new());
                }
                match ct {
                    ColumnType::Int64
                    | ColumnType::Int8
                    | ColumnType::Int16
                    | ColumnType::Int32
                    | ColumnType::UInt8
                    | ColumnType::UInt16
                    | ColumnType::UInt32
                    | ColumnType::UInt64
                    | ColumnType::Timestamp
                    | ColumnType::Date => ColBuf::I64(vec![None; n_rows]),
                    ColumnType::Float64 | ColumnType::Float32 => ColBuf::F64(vec![None; n_rows]),
                    ColumnType::String | ColumnType::StringDict => {
                        ColBuf::Str(vec![None; n_rows])
                    }
                    ColumnType::Bool => ColBuf::Bool(vec![None; n_rows]),
                    ColumnType::Binary => ColBuf::Bin(vec![None; n_rows]),
                    _ => ColBuf::Str(vec![None; n_rows]),
                }
            })
            .collect();

        let file_guard = self.file.read();
        let file = match file_guard.as_ref() {
            Some(f) => f,
            None => return Ok(None),
        };
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        for (rg_i, local_pairs) in rg_local_indices.iter().enumerate() {
            if local_pairs.is_empty() {
                continue;
            }
            let rg_meta = &footer.row_groups[rg_i];
            let rg_rows = rg_meta.row_count as usize;
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() {
                return Err(err_not_conn("RG extends past EOF"));
            }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
            if rg_bytes.len() < 32 {
                return Ok(None);
            }
            let compress_flag = rg_bytes[28];
            let encoding_version = rg_bytes[29];
            if compress_flag != RG_COMPRESS_NONE || encoding_version < 1 {
                return Ok(None);
            }

            let body = &rg_bytes[32..];
            let null_bitmap_len = (rg_rows + 7) / 8;
            let ids_section_len = rg_rows * 8;
            if ids_section_len + null_bitmap_len > body.len() {
                return Ok(None);
            }
            let del_bytes = &body[ids_section_len..ids_section_len + null_bitmap_len];
            let has_deletes = rg_meta.deletion_count > 0;
            let rcix = &footer.col_offsets[rg_i];

            let mut valid_pairs: Vec<(usize, usize)> = Vec::with_capacity(local_pairs.len());
            for &(out_idx, local_idx) in local_pairs {
                if local_idx >= rg_rows {
                    continue;
                }
                if has_deletes && (del_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 {
                    continue;
                }
                let id_off = local_idx * 8;
                if id_off + 8 > ids_section_len {
                    continue;
                }
                found_mask[out_idx] = true;
                out_ids[out_idx] =
                    u64::from_le_bytes(body[id_off..id_off + 8].try_into().unwrap()) as i64;
                valid_pairs.push((out_idx, local_idx));
            }
            if valid_pairs.is_empty() {
                continue;
            }

            for ci in 0..col_count {
                if !col_needed[ci] {
                    continue;
                }
                let ct = schema.columns[ci].1;
                let col_off = rcix[ci] as usize;
                if col_off + null_bitmap_len > body.len() {
                    return Ok(None);
                }
                let null_bytes = &body[col_off..col_off + null_bitmap_len];
                let col_bytes = &body[col_off + null_bitmap_len..];
                if col_bytes.is_empty() {
                    return Ok(None);
                }
                let encoding = col_bytes[0];
                let data_bytes = &col_bytes[1..];

                let extracted = match (encoding, ct, &mut col_bufs[ci]) {
                    (
                        COL_ENCODING_PLAIN,
                        ColumnType::Int64
                        | ColumnType::Int8
                        | ColumnType::Int16
                        | ColumnType::Int32
                        | ColumnType::UInt8
                        | ColumnType::UInt16
                        | ColumnType::UInt32
                        | ColumnType::UInt64
                        | ColumnType::Timestamp
                        | ColumnType::Date,
                        ColBuf::I64(vals),
                    ) if data_bytes.len() >= 8 => {
                        for &(out_idx, local_idx) in &valid_pairs {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 {
                                continue;
                            }
                            let off = 8 + local_idx * 8;
                            if off + 8 <= data_bytes.len() {
                                vals[out_idx] = Some(i64::from_le_bytes(
                                    data_bytes[off..off + 8].try_into().unwrap(),
                                ));
                            }
                        }
                        true
                    }
                    (
                        COL_ENCODING_PLAIN,
                        ColumnType::Float64 | ColumnType::Float32,
                        ColBuf::F64(vals),
                    ) if data_bytes.len() >= 8 => {
                        for &(out_idx, local_idx) in &valid_pairs {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 {
                                continue;
                            }
                            let off = 8 + local_idx * 8;
                            if off + 8 <= data_bytes.len() {
                                vals[out_idx] = Some(f64::from_le_bytes(
                                    data_bytes[off..off + 8].try_into().unwrap(),
                                ));
                            }
                        }
                        true
                    }
                    (COL_ENCODING_PLAIN, ColumnType::String, ColBuf::Str(vals))
                        if data_bytes.len() >= 8 =>
                    {
                        let count =
                            u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
                        let data_len_off = 8 + (count + 1) * 4;
                        if data_len_off + 8 > data_bytes.len() {
                            return Ok(None);
                        }
                        let data_start = data_len_off + 8;
                        for &(out_idx, local_idx) in &valid_pairs {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1
                                || local_idx >= count
                            {
                                continue;
                            }
                            let s_off = 8 + local_idx * 4;
                            let e_off = s_off + 4;
                            if e_off + 4 <= data_bytes.len() {
                                let s = u32::from_le_bytes(
                                    data_bytes[s_off..s_off + 4].try_into().unwrap(),
                                ) as usize;
                                let e = u32::from_le_bytes(
                                    data_bytes[e_off..e_off + 4].try_into().unwrap(),
                                ) as usize;
                                if data_start + e <= data_bytes.len() {
                                    vals[out_idx] = Some(
                                        std::str::from_utf8(&data_bytes[data_start + s..data_start + e])
                                            .unwrap_or("")
                                            .to_string(),
                                    );
                                }
                            }
                        }
                        true
                    }
                    (COL_ENCODING_PLAIN, ColumnType::StringDict, ColBuf::Str(vals))
                        if data_bytes.len() >= 16 =>
                    {
                        let row_count =
                            u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
                        let dict_size =
                            u64::from_le_bytes(data_bytes[8..16].try_into().unwrap()) as usize;
                        let indices_start = 16usize;
                        let dict_off_start = indices_start + row_count * 4;
                        let dict_data_len_off = dict_off_start + dict_size * 4;
                        if dict_data_len_off + 8 > data_bytes.len() {
                            return Ok(None);
                        }
                        let dict_data_len = u64::from_le_bytes(
                            data_bytes[dict_data_len_off..dict_data_len_off + 8]
                                .try_into()
                                .unwrap(),
                        ) as usize;
                        let dict_data_start = dict_data_len_off + 8;
                        for &(out_idx, local_idx) in &valid_pairs {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1
                                || local_idx >= row_count
                            {
                                continue;
                            }
                            let idx_off = indices_start + local_idx * 4;
                            if idx_off + 4 > data_bytes.len() {
                                continue;
                            }
                            let dict_idx =
                                u32::from_le_bytes(data_bytes[idx_off..idx_off + 4].try_into().unwrap());
                            if dict_idx == 0 {
                                continue;
                            }
                            let di = (dict_idx - 1) as usize;
                            if di >= dict_size {
                                continue;
                            }
                            let ds_off = dict_off_start + di * 4;
                            if ds_off + 4 > data_bytes.len() {
                                continue;
                            }
                            let ds = u32::from_le_bytes(
                                data_bytes[ds_off..ds_off + 4].try_into().unwrap(),
                            ) as usize;
                            let de = if di + 1 < dict_size {
                                let de_off = ds_off + 4;
                                if de_off + 4 <= data_bytes.len() {
                                    u32::from_le_bytes(
                                        data_bytes[de_off..de_off + 4].try_into().unwrap(),
                                    ) as usize
                                } else {
                                    dict_data_len
                                }
                            } else {
                                dict_data_len
                            };
                            if dict_data_start + de <= data_bytes.len() {
                                vals[out_idx] = Some(
                                    std::str::from_utf8(
                                        &data_bytes[dict_data_start + ds..dict_data_start + de],
                                    )
                                    .unwrap_or("")
                                    .to_string(),
                                );
                            }
                        }
                        true
                    }
                    (
                        COL_ENCODING_BITPACK,
                        ColumnType::Int64
                        | ColumnType::Int8
                        | ColumnType::Int16
                        | ColumnType::Int32
                        | ColumnType::UInt8
                        | ColumnType::UInt16
                        | ColumnType::UInt32
                        | ColumnType::UInt64
                        | ColumnType::Timestamp
                        | ColumnType::Date,
                        ColBuf::I64(vals),
                    ) => {
                        for &(out_idx, local_idx) in &valid_pairs {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 {
                                continue;
                            }
                            if let Some(v) = crate::storage::on_demand::bitpack_decode_at_idx(
                                data_bytes,
                                local_idx,
                            ) {
                                vals[out_idx] = Some(v);
                            }
                        }
                        true
                    }
                    (COL_ENCODING_PLAIN, ColumnType::Bool, ColBuf::Bool(vals))
                        if data_bytes.len() >= 8 =>
                    {
                        for &(out_idx, local_idx) in &valid_pairs {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 {
                                continue;
                            }
                            let byte_off = 8 + local_idx / 8;
                            if byte_off < data_bytes.len() {
                                vals[out_idx] =
                                    Some((data_bytes[byte_off] >> (local_idx % 8)) & 1 == 1);
                            }
                        }
                        true
                    }
                    _ => false,
                };
                if !extracted {
                    return Ok(None);
                }
            }
        }

        drop(mmap_guard);
        drop(file_guard);

        let n_out = found_mask.iter().filter(|&&b| b).count();
        if n_out == 0 {
            return Ok(Some(MmapBatchColumns {
                row_count: 0,
                columns: Vec::new(),
            }));
        }

        let mut columns = Vec::with_capacity(col_count + usize::from(include_id));
        if include_id {
            columns.push((
                "_id".to_string(),
                MmapBatchColumn::I64(
                    (0..n_rows)
                        .filter(|&i| found_mask[i])
                        .map(|i| Some(out_ids[i]))
                        .collect(),
                ),
            ));
        }

        for (ci, buf) in col_bufs.into_iter().enumerate() {
            if !col_needed[ci] {
                continue;
            }
            let name = schema.columns[ci].0.clone();
            match buf {
                ColBuf::I64(vals) => columns.push((
                    name,
                    MmapBatchColumn::I64(
                        (0..n_rows)
                            .filter(|&i| found_mask[i])
                            .map(|i| vals[i])
                            .collect(),
                    ),
                )),
                ColBuf::F64(vals) => columns.push((
                    name,
                    MmapBatchColumn::F64(
                        (0..n_rows)
                            .filter(|&i| found_mask[i])
                            .map(|i| vals[i])
                            .collect(),
                    ),
                )),
                ColBuf::Str(vals) => columns.push((
                    name,
                    MmapBatchColumn::Str(
                        (0..n_rows)
                            .filter(|&i| found_mask[i])
                            .map(|i| vals[i].clone())
                            .collect(),
                    ),
                )),
                ColBuf::Bool(vals) => columns.push((
                    name,
                    MmapBatchColumn::Bool(
                        (0..n_rows)
                            .filter(|&i| found_mask[i])
                            .map(|i| vals[i])
                            .collect(),
                    ),
                )),
                ColBuf::Bin(vals) => columns.push((
                    name,
                    MmapBatchColumn::Bin(
                        (0..n_rows)
                            .filter(|&i| found_mask[i])
                            .map(|i| vals[i].clone())
                            .collect(),
                    ),
                )),
            }
        }

        Ok(Some(MmapBatchColumns {
            row_count: n_out,
            columns,
        }))
    }

    /// Batch retrieve multiple rows by IDs using one footer lock + one mmap body slice per RG.
    ///
    /// Compared to N × retrieve_rcix: eliminates N-1 footer lock acquisitions, N-1 RG linear
    /// scans, and all read_cached_bytes overhead (page cache lock + pread per small read).
    /// Returns None to fall back when RCIX is unavailable for any needed RG.
    pub(crate) fn retrieve_many_mmap_columns(&self, ids: &[u64]) -> io::Result<Option<MmapBatchColumns>> {

        if ids.is_empty() {
            return Ok(Some(MmapBatchColumns { row_count: 0, columns: Vec::new() }));
        }

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_count = schema.column_count();
        let n_ids = ids.len();

        // ── Step 1: Map each input ID → rg_i (one footer read, no per-ID lock) ─
        let non_empty_row_groups: Vec<(u64, u64, usize)> = footer
            .row_groups
            .iter()
            .enumerate()
            .filter_map(|(rg_i, rg)| (rg.row_count > 0).then_some((rg.min_id, rg.max_id, rg_i)))
            .collect();
        let mut rg_map: Vec<Vec<(usize, u64)>> = vec![Vec::new(); footer.row_groups.len()];
        if !non_empty_row_groups.is_empty() {
            let ids_are_sorted = ids.windows(2).all(|pair| pair[0] <= pair[1]);
            if ids_are_sorted {
                let mut rg_pos = 0usize;
                for (out_pos, &id) in ids.iter().enumerate() {
                    while rg_pos < non_empty_row_groups.len()
                        && non_empty_row_groups[rg_pos].1 < id
                    {
                        rg_pos += 1;
                    }
                    if let Some(&(min_id, max_id, rg_i)) = non_empty_row_groups.get(rg_pos) {
                        if min_id <= id && id <= max_id {
                            rg_map[rg_i].push((out_pos, id));
                        }
                    }
                }
            } else {
                for (out_pos, &id) in ids.iter().enumerate() {
                    let rg_pos = non_empty_row_groups.partition_point(|(_, max_id, _)| *max_id < id);
                    if let Some(&(min_id, max_id, rg_i)) = non_empty_row_groups.get(rg_pos) {
                        if min_id <= id && id <= max_id {
                            rg_map[rg_i].push((out_pos, id));
                        }
                    }
                }
            }
        }

        // Upfront check: all needed RGs must have RCIX (encoding_version >= 1 + col_offsets)
        for (rg_i, hits) in rg_map.iter().enumerate() {
            if hits.is_empty() { continue; }
            if rg_i >= footer.col_offsets.len() || footer.col_offsets[rg_i].len() < col_count {
                return Ok(None);
            }
        }

        // ── Step 2: Allocate per-column staging buffers indexed by out_pos ──────
        // Vec<Option<T>> allows random-access writes so we can build output in ID order.
        let mut found_mask: Vec<bool> = vec![false; n_ids];
        let mut out_ids: Vec<i64> = vec![0i64; n_ids];

        enum ColBuf {
            I64(Vec<Option<i64>>),
            F64(Vec<Option<f64>>),
            Str(Vec<Option<String>>),
            Bool(Vec<Option<bool>>),
            Bin(Vec<Option<Vec<u8>>>),
        }

        let mut col_bufs: Vec<ColBuf> = schema.columns.iter().map(|(_, ct)| match ct {
            ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 |
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 |
            ColumnType::Timestamp | ColumnType::Date => ColBuf::I64(vec![None; n_ids]),
            ColumnType::Float64 | ColumnType::Float32 => ColBuf::F64(vec![None; n_ids]),
            ColumnType::String | ColumnType::StringDict => ColBuf::Str(vec![None; n_ids]),
            ColumnType::Bool => ColBuf::Bool(vec![None; n_ids]),
            ColumnType::Binary => ColBuf::Bin(vec![None; n_ids]),
            _ => ColBuf::Str(vec![None; n_ids]), // rare types as string fallback
        }).collect();

        // ── Step 3: Acquire mmap once, then process each RG in a single body slice
        let file_guard = self.file.read();
        let file = match file_guard.as_ref() { Some(f) => f, None => return Ok(None) };
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        for (rg_i, hits) in rg_map.iter().enumerate() {
            if hits.is_empty() { continue; }
            let rg_meta = &footer.row_groups[rg_i];
            let rg_rows = rg_meta.row_count as usize;
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return Err(err_not_conn("RG extends past EOF")); }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];

            if rg_bytes.len() < 32 { continue; }
            let compress_flag = rg_bytes[28];
            let encoding_version = rg_bytes[29];
            if compress_flag != RG_COMPRESS_NONE || encoding_version < 1 { return Ok(None); }

            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
            let null_bitmap_len = (rg_rows + 7) / 8;
            let ids_section_len = rg_rows * 8;
            if ids_section_len > body.len() { continue; }

            let rcix = &footer.col_offsets[rg_i]; // already validated above

            // Resolve local_idx for each ID in this RG (O(1) guess + optional binary search)
            let mut valid_hits: Vec<(usize, usize)> = Vec::with_capacity(hits.len()); // (out_pos, local_idx)
            let mut ids_cow_cache = None;
            for &(out_pos, id) in hits {
                let guess = id.saturating_sub(rg_meta.min_id) as usize;
                let local_idx = if guess < rg_rows {
                    let off = guess * 8;
                    if off + 8 <= body.len() {
                        let stored = u64::from_le_bytes(body[off..off+8].try_into().unwrap());
                        if stored == id { guess } else {
                            let ids_cow = ids_cow_cache
                                .get_or_insert_with(|| bytes_as_u64_slice(&body[..ids_section_len], rg_rows));
                            match ids_cow.binary_search(&id) { Ok(i) => i, Err(_) => continue }
                        }
                    } else { continue }
                } else {
                    let ids_cow = ids_cow_cache
                        .get_or_insert_with(|| bytes_as_u64_slice(&body[..ids_section_len], rg_rows));
                    match ids_cow.binary_search(&id) { Ok(i) => i, Err(_) => continue }
                };

                // Deletion check
                let del_off = ids_section_len + local_idx / 8;
                if del_off >= body.len() { continue; }
                if (body[del_off] >> (local_idx % 8)) & 1 == 1 { continue; }

                found_mask[out_pos] = true;
                out_ids[out_pos] = id as i64;
                valid_hits.push((out_pos, local_idx));
            }

            if valid_hits.is_empty() { continue; }

            // Extract column values for all valid hits in this RG
            for ci in 0..col_count {
                let ct = schema.columns[ci].1;
                let col_off = rcix[ci] as usize;
                if col_off + null_bitmap_len > body.len() { continue; }
                let null_bytes = &body[col_off..col_off + null_bitmap_len];
                let col_bytes = &body[col_off + null_bitmap_len..];

                if col_bytes.is_empty() { continue; }
                let encoding = col_bytes[0];
                let data_bytes = &col_bytes[1..];

                let extracted = match (encoding, ct, &mut col_bufs[ci]) {
                    (COL_ENCODING_PLAIN, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
                     ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
                     ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date, ColBuf::I64(vals))
                    if data_bytes.len() >= 8 => {
                        for &(out_pos, local_idx) in &valid_hits {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                            let off = 8 + local_idx * 8;
                            if off + 8 <= data_bytes.len() {
                                vals[out_pos] = Some(i64::from_le_bytes(data_bytes[off..off+8].try_into().unwrap()));
                            }
                        }
                        true
                    }
                    (COL_ENCODING_PLAIN, ColumnType::Float64 | ColumnType::Float32, ColBuf::F64(vals))
                    if data_bytes.len() >= 8 => {
                        for &(out_pos, local_idx) in &valid_hits {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                            let off = 8 + local_idx * 8;
                            if off + 8 <= data_bytes.len() {
                                vals[out_pos] = Some(f64::from_le_bytes(data_bytes[off..off+8].try_into().unwrap()));
                            }
                        }
                        true
                    }
                    (COL_ENCODING_PLAIN, ColumnType::String, ColBuf::Str(vals))
                    if data_bytes.len() >= 8 => {
                        let count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
                        let data_len_off = 8 + (count + 1) * 4;
                        if data_len_off + 8 > data_bytes.len() { continue; }
                        let data_start = data_len_off + 8;
                        for &(out_pos, local_idx) in &valid_hits {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                            if local_idx >= count { continue; }
                            let s_off = 8 + local_idx * 4;
                            let e_off = s_off + 4;
                            if e_off + 4 <= data_bytes.len() {
                                let s = u32::from_le_bytes(data_bytes[s_off..s_off+4].try_into().unwrap()) as usize;
                                let e = u32::from_le_bytes(data_bytes[e_off..e_off+4].try_into().unwrap()) as usize;
                                if data_start + e <= data_bytes.len() {
                                    vals[out_pos] = Some(
                                        std::str::from_utf8(&data_bytes[data_start+s..data_start+e])
                                            .unwrap_or("").to_string()
                                    );
                                }
                            }
                        }
                        true
                    }
                    (COL_ENCODING_PLAIN, ColumnType::StringDict, ColBuf::Str(vals))
                    if data_bytes.len() >= 16 => {
                        let row_count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
                        let dict_size = u64::from_le_bytes(data_bytes[8..16].try_into().unwrap()) as usize;
                        let indices_start = 16usize;
                        let dict_off_start = indices_start + row_count * 4;
                        let dict_data_len_off = dict_off_start + dict_size * 4;
                        if dict_data_len_off + 8 > data_bytes.len() { continue; }
                        let dict_data_len = u64::from_le_bytes(
                            data_bytes[dict_data_len_off..dict_data_len_off+8].try_into().unwrap()
                        ) as usize;
                        let dict_data_start = dict_data_len_off + 8;
                        for &(out_pos, local_idx) in &valid_hits {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                            if local_idx >= row_count { continue; }
                            let idx_off = indices_start + local_idx * 4;
                            if idx_off + 4 > data_bytes.len() { continue; }
                            let dict_idx = u32::from_le_bytes(data_bytes[idx_off..idx_off+4].try_into().unwrap());
                            if dict_idx == 0 { continue; }
                            let di = (dict_idx - 1) as usize;
                            if di >= dict_size { continue; }
                            let ds_off = dict_off_start + di * 4;
                            if ds_off + 4 > data_bytes.len() { continue; }
                            let ds = u32::from_le_bytes(data_bytes[ds_off..ds_off+4].try_into().unwrap()) as usize;
                            let de = if di + 1 < dict_size {
                                let de_off = ds_off + 4;
                                if de_off + 4 <= data_bytes.len() {
                                    u32::from_le_bytes(data_bytes[de_off..de_off+4].try_into().unwrap()) as usize
                                } else { dict_data_len }
                            } else { dict_data_len };
                            if dict_data_start + de <= data_bytes.len() {
                                vals[out_pos] = Some(
                                    std::str::from_utf8(&data_bytes[dict_data_start+ds..dict_data_start+de])
                                        .unwrap_or("").to_string()
                                );
                            }
                        }
                        true
                    }
                    (2u8, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 |
                     ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 |
                     ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date, ColBuf::I64(vals)) => {
                        for &(out_pos, local_idx) in &valid_hits {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                            if let Some(v) = crate::storage::on_demand::bitpack_decode_at_idx(data_bytes, local_idx) {
                                vals[out_pos] = Some(v);
                            }
                        }
                        true
                    }
                    (COL_ENCODING_PLAIN, ColumnType::Bool, ColBuf::Bool(vals))
                    if data_bytes.len() >= 8 => {
                        for &(out_pos, local_idx) in &valid_hits {
                            if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                            let byte_off = 8 + local_idx / 8;
                            if byte_off < data_bytes.len() {
                                vals[out_pos] = Some((data_bytes[byte_off] >> (local_idx % 8)) & 1 == 1);
                            }
                        }
                        true
                    }
                    _ => false,
                };

                if !extracted {
                    // Fallback: full column decode, pick values at target indices
                    let (col_data, _) = read_column_encoded(col_bytes, ct)?;
                    match &mut col_bufs[ci] {
                        ColBuf::I64(vals) => {
                            if let ColumnData::Int64(v) = &col_data {
                                for &(out_pos, local_idx) in &valid_hits {
                                    if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                    if local_idx < v.len() { vals[out_pos] = Some(v[local_idx]); }
                                }
                            }
                        }
                        ColBuf::F64(vals) => {
                            if let ColumnData::Float64(v) = &col_data {
                                for &(out_pos, local_idx) in &valid_hits {
                                    if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                    if local_idx < v.len() { vals[out_pos] = Some(v[local_idx]); }
                                }
                            }
                        }
                        ColBuf::Str(vals) => {
                            match &col_data {
                                ColumnData::String { offsets, data } => {
                                    let cnt = offsets.len().saturating_sub(1);
                                    for &(out_pos, local_idx) in &valid_hits {
                                        if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                        if local_idx < cnt {
                                            let s = offsets[local_idx] as usize;
                                            let e = offsets[local_idx + 1] as usize;
                                            vals[out_pos] = Some(std::str::from_utf8(&data[s..e]).unwrap_or("").to_string());
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        ColBuf::Bin(vals) => {
                            if let ColumnData::Binary { offsets, data } = &col_data {
                                let cnt = offsets.len().saturating_sub(1);
                                for &(out_pos, local_idx) in &valid_hits {
                                    if (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1 { continue; }
                                    if local_idx < cnt {
                                        let s = offsets[local_idx] as usize;
                                        let e = offsets[local_idx + 1] as usize;
                                        vals[out_pos] = Some(data[s..e].to_vec());
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        drop(mmap_guard);
        drop(file_guard);

        // ── Step 4: Build native column buffers in input-ID order ─────────────
        let n_out: usize = found_mask.iter().filter(|&&b| b).count();
        if n_out == 0 {
            return Ok(Some(MmapBatchColumns { row_count: 0, columns: Vec::new() }));
        }

        let mut columns: Vec<(String, MmapBatchColumn)> = Vec::with_capacity(col_count + 1);

        // _id column
        let id_vals: Vec<i64> = (0..n_ids).filter(|&i| found_mask[i]).map(|i| out_ids[i]).collect();
        columns.push(("_id".to_string(), MmapBatchColumn::I64(id_vals.into_iter().map(Some).collect())));

        for (ci, buf) in col_bufs.into_iter().enumerate() {
            let col_name = &schema.columns[ci].0;
            match buf {
                ColBuf::I64(vals) => {
                    let v: Vec<Option<i64>> = (0..n_ids).filter(|&i| found_mask[i]).map(|i| vals[i]).collect();
                    columns.push((col_name.clone(), MmapBatchColumn::I64(v)));
                }
                ColBuf::F64(vals) => {
                    let v: Vec<Option<f64>> = (0..n_ids).filter(|&i| found_mask[i]).map(|i| vals[i]).collect();
                    columns.push((col_name.clone(), MmapBatchColumn::F64(v)));
                }
                ColBuf::Str(vals) => {
                    let v: Vec<Option<String>> = (0..n_ids).filter(|&i| found_mask[i]).map(|i| vals[i].clone()).collect();
                    columns.push((col_name.clone(), MmapBatchColumn::Str(v)));
                }
                ColBuf::Bool(vals) => {
                    let v: Vec<Option<bool>> = (0..n_ids).filter(|&i| found_mask[i]).map(|i| vals[i]).collect();
                    columns.push((col_name.clone(), MmapBatchColumn::Bool(v)));
                }
                ColBuf::Bin(vals) => {
                    let v: Vec<Option<Vec<u8>>> = (0..n_ids).filter(|&i| found_mask[i]).map(|i| vals[i].clone()).collect();
                    columns.push((col_name.clone(), MmapBatchColumn::Bin(v)));
                }
            }
        }

        Ok(Some(MmapBatchColumns { row_count: n_out, columns }))
    }

    pub fn retrieve_many_mmap(&self, ids: &[u64]) -> io::Result<Option<arrow::record_batch::RecordBatch>> {
        use arrow::array::{ArrayRef, BinaryArray, BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        let Some(batch_cols) = self.retrieve_many_mmap_columns(ids)? else {
            return Ok(None);
        };
        if batch_cols.row_count == 0 {
            return Ok(Some(arrow::record_batch::RecordBatch::new_empty(Arc::new(Schema::empty()))));
        }

        let mut fields: Vec<Field> = Vec::with_capacity(batch_cols.columns.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(batch_cols.columns.len());
        for (name, col) in batch_cols.columns {
            match col {
                MmapBatchColumn::I64(vals) => {
                    fields.push(Field::new(&name, ArrowDataType::Int64, true));
                    arrays.push(Arc::new(Int64Array::from(vals)) as ArrayRef);
                }
                MmapBatchColumn::F64(vals) => {
                    fields.push(Field::new(&name, ArrowDataType::Float64, true));
                    arrays.push(Arc::new(Float64Array::from(vals)) as ArrayRef);
                }
                MmapBatchColumn::Str(vals) => {
                    fields.push(Field::new(&name, ArrowDataType::Utf8, true));
                    arrays.push(Arc::new(StringArray::from(vals)) as ArrayRef);
                }
                MmapBatchColumn::Bool(vals) => {
                    let arr: BooleanArray = vals.into_iter().collect();
                    fields.push(Field::new(&name, ArrowDataType::Boolean, true));
                    arrays.push(Arc::new(arr) as ArrayRef);
                }
                MmapBatchColumn::Bin(vals) => {
                    let bin_data: Vec<Option<&[u8]>> = vals.iter().map(|o| o.as_deref()).collect();
                    fields.push(Field::new(&name, ArrowDataType::Binary, true));
                    arrays.push(Arc::new(BinaryArray::from(bin_data)) as ArrayRef);
                }
            }
        }

        let batch_schema = Arc::new(Schema::new(fields));
        arrow::record_batch::RecordBatch::try_new(batch_schema, arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
            .map(Some)
    }

    /// Single-pass parallel LIKE scan + full-row extraction.
    ///
    /// For each RG (Rayon parallel): scans the LIKE column for matches, then
    /// immediately extracts ALL columns for matched rows — no separate scan/extract passes.
    /// Materializes only the matching rows, avoiding the 1M-row Arrow allocation of
    /// the full-scan path. Returns None to fall back on compressed/non-RCIX files.
    pub fn scan_like_and_extract_mmap(
        &self,
        col_name: &str,
        pattern: &str,
        limit: Option<usize>,
    ) -> io::Result<Option<arrow::record_batch::RecordBatch>> {
        use rayon::prelude::*;
        use arrow::array::{ArrayRef, Int64Array, Float64Array, BooleanArray, StringArray};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType};
        use std::sync::Arc;

        if !pattern.contains('%') && !pattern.contains('_') {
            return Ok(None); // exact match → caller uses string equality scanner
        }
        let like_kind = match classify_like_pattern(pattern) {
            Some(k) => k,
            None => return Ok(None),
        };

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        if !matches!(col_type, ColumnType::String | ColumnType::StringDict) {
            return Ok(None);
        }
        let is_dict_col = matches!(col_type, ColumnType::StringDict);
        let col_count = schema.column_count();

        let file_guard = self.file.read();
        let file = file_guard.as_ref()
            .ok_or_else(|| err_not_conn("File not open for LIKE scan+extract"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        // All RGs must be uncompressed + have RCIX
        let all_fast = footer.row_groups.iter().enumerate().all(|(rg_i, rg_meta)| {
            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap_ref.len() { return false; }
            let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
            let compress_flag = rg_bytes.get(28).copied().unwrap_or(RG_COMPRESS_NONE);
            let enc_ver = rg_bytes.get(29).copied().unwrap_or(0);
            compress_flag == RG_COMPRESS_NONE && enc_ver >= 1
                && footer.col_offsets.get(rg_i).map_or(false, |v| v.len() >= col_count)
        });
        if !all_fast { return Ok(None); }

        // Build output Arrow schema: _id + all schema columns
        let schema_col_types: Vec<(String, ColumnType)> =
            schema.columns.iter().map(|(n, ct)| (n.clone(), *ct)).collect();
        let mut out_fields: Vec<Field> = Vec::with_capacity(col_count + 1);
        out_fields.push(Field::new("_id", ArrowDataType::Int64, false));
        for (cn, ct) in &schema_col_types {
            let adt = match ct {
                ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32
                | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64
                | ColumnType::Timestamp | ColumnType::Date => ArrowDataType::Int64,
                ColumnType::Float64 | ColumnType::Float32 => ArrowDataType::Float64,
                ColumnType::String | ColumnType::StringDict => ArrowDataType::Utf8,
                ColumnType::Bool => ArrowDataType::Boolean,
                _ => ArrowDataType::Utf8,
            };
            out_fields.push(Field::new(cn.as_str(), adt, true));
        }
        let out_schema: Arc<Schema> = Arc::new(Schema::new(out_fields));

        // Unsafe raw pointer for Rayon (mmap lives for the duration of this fn)
        let mmap_ptr: usize = mmap_ref.as_ptr() as usize;
        let mmap_len: usize = mmap_ref.len();

        struct RgDesc {
            rg_offset: usize,
            rg_data_size: usize,
            rg_rows: usize,
            has_deletes: bool,
            col_rcix: Vec<u32>,
        }
        let rg_descs: Vec<RgDesc> = footer.row_groups.iter().enumerate().map(|(rg_i, rg_meta)| {
            RgDesc {
                rg_offset:    rg_meta.offset as usize,
                rg_data_size: rg_meta.data_size as usize,
                rg_rows:      rg_meta.row_count as usize,
                has_deletes:  rg_meta.deletion_count > 0,
                col_rcix:     footer.col_offsets[rg_i].clone(),
            }
        }).collect();

        let like_kind_ref = &like_kind;
        let schema_types_ref = &schema_col_types;
        let out_schema_arc = out_schema.clone();

        // ── Parallel per-RG: scan LIKE col → extract all cols ────────────────
        let rg_batches: Vec<Option<arrow::record_batch::RecordBatch>> = rg_descs.par_iter().map(|desc| {
            let mmap: &[u8] = unsafe { std::slice::from_raw_parts(mmap_ptr as *const u8, mmap_len) };
            let rg_end = desc.rg_offset + desc.rg_data_size;
            if rg_end > mmap.len() || desc.rg_data_size < 32 { return None; }
            let body = &mmap[desc.rg_offset + 32..rg_end];
            let rg_rows = desc.rg_rows;
            let bitmap_len = (rg_rows + 7) / 8;
            let del_start = rg_rows * 8;
            let del_bytes_opt: Option<&[u8]> = if desc.has_deletes && del_start + bitmap_len <= body.len() {
                Some(&body[del_start..del_start + bitmap_len])
            } else { None };

            // ── Scan LIKE column ─────────────────────────────────────────────
            let lc_off = desc.col_rcix[col_idx] as usize;
            if lc_off + bitmap_len > body.len() { return None; }
            let lc_null = &body[lc_off..lc_off + bitmap_len];
            let lc_col  = &body[lc_off + bitmap_len..];
            if lc_col.is_empty() || lc_col[0] != COL_ENCODING_PLAIN { return None; }
            let lc_data = &lc_col[1..];

            let mut matched: Vec<usize> = Vec::new();

            if !is_dict_col {
                // String PLAIN
                if lc_data.len() < 8 { return None; }
                let count = u64::from_le_bytes(lc_data[0..8].try_into().ok()?) as usize;
                let off_end = 8 + (count + 1) * 4;
                if off_end + 8 > lc_data.len() { return None; }
                let dsl = u64::from_le_bytes(lc_data[off_end..off_end+8].try_into().ok()?) as usize;
                let ds = off_end + 8;
                let de = (ds + dsl).min(lc_data.len());
                let data_region = &lc_data[ds..de];
                let off_cow = bytes_as_u32_slice(&lc_data[8..], count + 1);
                let offsets: &[u32] = &off_cow;
                let n = count.min(rg_rows);
                for i in 0..n {
                    if let Some(db) = del_bytes_opt { if (db[i/8] >> (i%8)) & 1 == 1 { continue; } }
                    if (lc_null[i/8] >> (i%8)) & 1 == 1 { continue; }
                    let s = offsets[i] as usize; let e = offsets[i+1] as usize;
                    if e <= data_region.len() && like_matches_bytes(like_kind_ref, &data_region[s..e]) {
                        matched.push(i);
                    }
                }
            } else {
                // StringDict PLAIN
                if lc_data.len() < 16 { return None; }
                let row_count = u64::from_le_bytes(lc_data[0..8].try_into().ok()?) as usize;
                let dict_size = u64::from_le_bytes(lc_data[8..16].try_into().ok()?) as usize;
                if dict_size == 0 { return None; }
                let doff_start = 16 + row_count * 4;
                let ddl_off = doff_start + dict_size * 4;
                if ddl_off + 8 > lc_data.len() { return None; }
                let ddl = u64::from_le_bytes(lc_data[ddl_off..ddl_off+8].try_into().ok()?) as usize;
                let dds = ddl_off + 8;
                let raw_dict = &lc_data[dds..(dds + ddl).min(lc_data.len())];
                let doff_cow = bytes_as_u32_slice(&lc_data[doff_start..], dict_size);
                let dict_offsets: &[u32] = &doff_cow;
                let idx_cow = bytes_as_u32_slice(&lc_data[16..], row_count);
                let indices: &[u32] = &idx_cow;
                let mut match_flags = vec![false; dict_size + 1];
                for di in 0..dict_size {
                    let a = dict_offsets[di] as usize;
                    let b = if di + 1 < dict_size { dict_offsets[di+1] as usize } else { ddl };
                    if a <= b && b <= raw_dict.len() {
                        match_flags[di + 1] = like_matches_bytes(like_kind_ref, &raw_dict[a..b]);
                    }
                }
                let n = row_count.min(rg_rows);
                for i in 0..n {
                    if let Some(db) = del_bytes_opt { if (db[i/8] >> (i%8)) & 1 == 1 { continue; } }
                    let idx = indices[i] as usize;
                    if idx < match_flags.len() && match_flags[idx] { matched.push(i); }
                }
            }

            if matched.is_empty() { return None; }
            let n_match = matched.len();

            // ── Extract _id + all columns for matched rows ───────────────────
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(schema_types_ref.len() + 1);

            // _id from the IDs section (first rg_rows * 8 bytes of body)
            let id_vals: Vec<i64> = matched.iter().map(|&li| {
                let off = li * 8;
                if off + 8 <= body.len() {
                    i64::from_le_bytes(body[off..off+8].try_into().unwrap_or([0;8]))
                } else { 0 }
            }).collect();
            arrays.push(Arc::new(Int64Array::from(id_vals)) as ArrayRef);

            for ci in 0..schema_types_ref.len() {
                let ct = schema_types_ref[ci].1;
                let col_off = desc.col_rcix[ci] as usize;

                macro_rules! push_null_arr {
                    ($ct:expr, $n:expr) => {
                        match $ct {
                            ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32
                            | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64
                            | ColumnType::Timestamp | ColumnType::Date =>
                                Arc::new(Int64Array::from(vec![None::<i64>; $n])) as ArrayRef,
                            ColumnType::Float64 | ColumnType::Float32 =>
                                Arc::new(Float64Array::from(vec![None::<f64>; $n])) as ArrayRef,
                            ColumnType::Bool =>
                                Arc::new(BooleanArray::from(vec![None::<bool>; $n])) as ArrayRef,
                            _ =>
                                Arc::new(StringArray::from(vec![None::<&str>; $n])) as ArrayRef,
                        }
                    };
                }

                if col_off + bitmap_len > body.len() {
                    arrays.push(push_null_arr!(ct, n_match)); continue;
                }
                let null_bytes = &body[col_off..col_off + bitmap_len];
                let col_bytes  = &body[col_off + bitmap_len..];
                if col_bytes.is_empty() {
                    arrays.push(push_null_arr!(ct, n_match)); continue;
                }
                let encoding   = col_bytes[0];
                let data_bytes = if col_bytes.len() > 1 { &col_bytes[1..] } else { &[] as &[u8] };

                let arr: ArrayRef = match (encoding, ct) {
                    // ── Plain Int64-family ───────────────────────────────────
                    (COL_ENCODING_PLAIN, ColumnType::Int64 | ColumnType::Int8 | ColumnType::Int16
                     | ColumnType::Int32 | ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32
                     | ColumnType::UInt64 | ColumnType::Timestamp | ColumnType::Date)
                    if data_bytes.len() >= 8 => {
                        let vals: Vec<Option<i64>> = matched.iter().map(|&li| {
                            if (null_bytes[li/8] >> (li%8)) & 1 == 1 { return None; }
                            let off = 8 + li * 8;
                            if off + 8 <= data_bytes.len() {
                                Some(i64::from_le_bytes(data_bytes[off..off+8].try_into().unwrap_or([0;8])))
                            } else { None }
                        }).collect();
                        Arc::new(Int64Array::from(vals))
                    }
                    // ── Plain Float64-family ─────────────────────────────────
                    (COL_ENCODING_PLAIN, ColumnType::Float64 | ColumnType::Float32)
                    if data_bytes.len() >= 8 => {
                        let vals: Vec<Option<f64>> = matched.iter().map(|&li| {
                            if (null_bytes[li/8] >> (li%8)) & 1 == 1 { return None; }
                            let off = 8 + li * 8;
                            if off + 8 <= data_bytes.len() {
                                Some(f64::from_le_bytes(data_bytes[off..off+8].try_into().unwrap_or([0;8])))
                            } else { None }
                        }).collect();
                        Arc::new(Float64Array::from(vals))
                    }
                    // ── Plain String ─────────────────────────────────────────
                    (COL_ENCODING_PLAIN, ColumnType::String) if data_bytes.len() >= 8 => {
                        let count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap_or([0;8])) as usize;
                        let off_end = 8 + (count + 1) * 4;
                        if off_end + 8 > data_bytes.len() {
                            push_null_arr!(ct, n_match)
                        } else {
                            let dsl = u64::from_le_bytes(data_bytes[off_end..off_end+8].try_into().unwrap_or([0;8])) as usize;
                            let dstart = off_end + 8;
                            let dend   = (dstart + dsl).min(data_bytes.len());
                            let dr = &data_bytes[dstart..dend];
                            let oc = bytes_as_u32_slice(&data_bytes[8..], count + 1);
                            let offs: &[u32] = &oc;
                            let vals: Vec<Option<&str>> = matched.iter().map(|&li| {
                                if (null_bytes[li/8] >> (li%8)) & 1 == 1 { return None; }
                                if li >= count { return None; }
                                let s = offs[li] as usize; let e = offs[li+1] as usize;
                                if e <= dr.len() { std::str::from_utf8(&dr[s..e]).ok() } else { None }
                            }).collect();
                            Arc::new(StringArray::from(vals))
                        }
                    }
                    // ── Plain StringDict ─────────────────────────────────────
                    (COL_ENCODING_PLAIN, ColumnType::StringDict) if data_bytes.len() >= 16 => {
                        let row_count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap_or([0;8])) as usize;
                        let dict_size = u64::from_le_bytes(data_bytes[8..16].try_into().unwrap_or([0;8])) as usize;
                        if dict_size == 0 {
                            push_null_arr!(ct, n_match)
                        } else {
                            let doff_s = 16 + row_count * 4;
                            let ddl_o  = doff_s + dict_size * 4;
                            if ddl_o + 8 > data_bytes.len() {
                                push_null_arr!(ct, n_match)
                            } else {
                                let ddl = u64::from_le_bytes(data_bytes[ddl_o..ddl_o+8].try_into().unwrap_or([0;8])) as usize;
                                let dds = ddl_o + 8;
                                let raw_dict = &data_bytes[dds..(dds + ddl).min(data_bytes.len())];
                                let doff_c = bytes_as_u32_slice(&data_bytes[doff_s..], dict_size);
                                let doffs: &[u32] = &doff_c;
                                let idx_c = bytes_as_u32_slice(&data_bytes[16..], row_count);
                                let idxs: &[u32] = &idx_c;
                                let vals: Vec<Option<&str>> = matched.iter().map(|&li| {
                                    if (null_bytes[li/8] >> (li%8)) & 1 == 1 { return None; }
                                    if li >= row_count { return None; }
                                    let dk = idxs[li] as usize;
                                    if dk == 0 || dk > dict_size { return None; }
                                    let di = dk - 1;
                                    let a = doffs[di] as usize;
                                    let b = if di + 1 < dict_size { doffs[di+1] as usize } else { ddl };
                                    if a <= b && b <= raw_dict.len() {
                                        std::str::from_utf8(&raw_dict[a..b]).ok()
                                    } else { None }
                                }).collect();
                                Arc::new(StringArray::from(vals))
                            }
                        }
                    }
                    // ── Plain Bool ───────────────────────────────────────────
                    (COL_ENCODING_PLAIN, ColumnType::Bool) => {
                        let vals: Vec<Option<bool>> = matched.iter().map(|&li| {
                            if (null_bytes[li/8] >> (li%8)) & 1 == 1 { return None; }
                            let off = 8 + li;
                            if off < data_bytes.len() { Some(data_bytes[off] != 0) } else { None }
                        }).collect();
                        Arc::new(BooleanArray::from(vals))
                    }
                    // ── Encoded (non-PLAIN): decode full column, pick rows ───
                    _ => {
                        match read_column_encoded(col_bytes, ct) {
                            Ok((col_data, _)) => {
                                let col_data = if matches!(&col_data, ColumnData::StringDict { .. }) {
                                    col_data.decode_string_dict()
                                } else { col_data };
                                match col_data {
                                    ColumnData::Int64(v) => {
                                        let vals: Vec<Option<i64>> = matched.iter().map(|&li| {
                                            if (null_bytes[li/8] >> (li%8)) & 1 == 1 { return None; }
                                            v.get(li).copied()
                                        }).collect();
                                        Arc::new(Int64Array::from(vals))
                                    }
                                    ColumnData::Float64(v) => {
                                        let vals: Vec<Option<f64>> = matched.iter().map(|&li| {
                                            if (null_bytes[li/8] >> (li%8)) & 1 == 1 { return None; }
                                            v.get(li).copied()
                                        }).collect();
                                        Arc::new(Float64Array::from(vals))
                                    }
                                    ColumnData::String { offsets, data: str_data } => {
                                        let cnt = offsets.len().saturating_sub(1);
                                        let vals: Vec<Option<&str>> = matched.iter().map(|&li| {
                                            if (null_bytes[li/8] >> (li%8)) & 1 == 1 { return None; }
                                            if li >= cnt { return None; }
                                            let s = offsets[li] as usize;
                                            let e = offsets[li + 1] as usize;
                                            if e <= str_data.len() { std::str::from_utf8(&str_data[s..e]).ok() } else { None }
                                        }).collect();
                                        Arc::new(StringArray::from(vals))
                                    }
                                    _ => push_null_arr!(ct, n_match),
                                }
                            }
                            Err(_) => push_null_arr!(ct, n_match),
                        }
                    }
                };
                arrays.push(arr);
            }

            arrow::record_batch::RecordBatch::try_new(out_schema_arc.clone(), arrays).ok()
        }).collect();

        // ── Concatenate per-RG batches ────────────────────────────────────────
        let non_empty: Vec<&arrow::record_batch::RecordBatch> =
            rg_batches.iter().filter_map(|b| b.as_ref()).collect();

        let result = if non_empty.is_empty() {
            arrow::record_batch::RecordBatch::new_empty(out_schema)
        } else if non_empty.len() == 1 {
            non_empty[0].clone()
        } else {
            // Per-column concat (most efficient for Arrow)
            let n_fields = out_schema.fields().len();
            let mut final_arrays: Vec<ArrayRef> = Vec::with_capacity(n_fields);
            for ci in 0..n_fields {
                let cols: Vec<&dyn arrow::array::Array> =
                    non_empty.iter().map(|b| b.column(ci).as_ref()).collect();
                let arr = arrow::compute::concat(&cols)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                final_arrays.push(arr);
            }
            arrow::record_batch::RecordBatch::try_new(out_schema, final_arrays)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?
        };

        // Apply limit if requested
        let result = if let Some(lim) = limit {
            let n = result.num_rows().min(lim);
            result.slice(0, n)
        } else {
            result
        };

        Ok(Some(result))
    }

    /// Zero-copy parallel TopK for a Binary (vector) column.
    ///
    /// For uniform-stride binary data (fixed-size vectors stored as raw f32 bytes),
    /// scans directly on OS mmap with no memcpy.
    /// Returns `Some(Vec<(global_row_idx, distance)>)` or `None` to fall back.
    pub fn topk_binary_direct(
        &self,
        col_name: &str,
        computer: &crate::query::vector_ops::DistanceComputer,
        k: usize,
    ) -> io::Result<Option<Vec<(usize, f32)>>> {
        use crate::query::vector_ops::topk_heap_on_floats;

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        if schema.columns[col_idx].1 != ColumnType::Binary {
            return Ok(None);
        }

        let file_guard = self.file.read();
        let file = match file_guard.as_ref() { Some(f) => f, None => return Ok(None) };
        let mmap_arc = self.mmap_cache.write().get_mmap_arc(file)?;
        drop(file_guard);
        let mmap: &[u8] = &mmap_arc;

        // ── PASS 1: validate all RGs, determine dim, compute total rows ──────
        let query_dim = computer.query.len();
        let total_active: usize = footer.row_groups.iter().map(|rg| rg.active_rows() as usize).sum();
        if total_active == 0 { return Ok(Some(vec![])); }

        struct RgDesc { count: usize, dim: usize, data_start: usize, byte_len: usize }
        let mut rg_descs: Vec<Option<RgDesc>> = Vec::with_capacity(footer.row_groups.len());

        for (rg_idx, rg_meta) in footer.row_groups.iter().enumerate() {
            if rg_meta.row_count == 0 { rg_descs.push(None); continue; }
            let rg_rows = rg_meta.row_count as usize;

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap.len() { return Ok(None); }
            let rg_bytes = &mmap[rg_meta.offset as usize..rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { 1 };
            let encoding_ver  = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };

            if compress_flag != RG_COMPRESS_NONE || encoding_ver < 1
                || rg_meta.deletion_count > 0
                || rg_idx >= footer.col_offsets.len()
                || col_idx >= footer.col_offsets[rg_idx].len()
            { return Ok(None); }

            let rg_body_abs  = (rg_meta.offset + 32) as usize;
            let col_abs      = rg_body_abs + footer.col_offsets[rg_idx][col_idx] as usize;
            let null_bm_len  = (rg_rows + 7) / 8;
            let data_abs     = col_abs + null_bm_len;

            if data_abs + 9 > mmap.len() { return Ok(None); }
            if mmap[data_abs] != COL_ENCODING_PLAIN { return Ok(None); }

            let count = u64::from_le_bytes(mmap[data_abs+1..data_abs+9].try_into().unwrap()) as usize;
            if count == 0 { rg_descs.push(None); continue; }

            let off_base = data_abs + 9;
            if off_base + 8 > mmap.len() { return Ok(None); }
            let off0 = u32::from_le_bytes(mmap[off_base..off_base+4].try_into().unwrap()) as usize;
            let off1 = u32::from_le_bytes(mmap[off_base+4..off_base+8].try_into().unwrap()) as usize;
            if off1 <= off0 || (off1 - off0) % 4 != 0 { return Ok(None); }
            let dim = (off1 - off0) / 4;
            if dim != query_dim { return Ok(None); }

            // Binary column format: [count:u64][(count+1)*u32 offsets][data_len:u64][data bytes]
            // Must skip the 8-byte data_len field between the offsets array and the float data.
            let data_start = off_base + (count + 1) * 4 + 8;
            let byte_len   = count * dim * 4;
            if data_start + byte_len > mmap.len() { return Ok(None); }
            rg_descs.push(Some(RgDesc { count, dim, data_start, byte_len }));
        }

        // ── PASS 2: fill reusable buffer and run ONE topk scan ───────────────
        // scan_buf caches the float data for this column. On repeated queries the
        // data is already present — skip the 512MB mmap→heap copy entirely.
        // Invalidated by invalidate_page_cache() on every write.
        let needed = total_active * query_dim;
        let file_size = mmap.len() as u64;
        let mut buf_guard = self.scan_buf.lock().unwrap();
        let cached_size = self.scan_buf_file_size.load(std::sync::atomic::Ordering::Acquire);
        let col_guard = self.scan_buf_col.lock().unwrap();
        let cache_hit = cached_size == file_size
            && buf_guard.len() >= needed
            && col_guard.as_str() == col_name;
        drop(col_guard);

        if !cache_hit {
            if buf_guard.capacity() < needed {
                let cur = buf_guard.len();
                buf_guard.reserve(needed - cur);
            }
            unsafe { buf_guard.set_len(needed); }

            let buf_ptr = buf_guard.as_mut_ptr();
            let mut filled_floats = 0usize;
            for desc in rg_descs.iter() {
                let Some(d) = desc else { continue };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        mmap.as_ptr().add(d.data_start),
                        (buf_ptr as *mut u8).add(filled_floats * 4),
                        d.byte_len,
                    );
                }
                filled_floats += d.count * d.dim;
            }

            // Mark cache valid
            let mut cg = self.scan_buf_col.lock().unwrap();
            cg.clear(); cg.push_str(col_name);
            drop(cg);
            self.scan_buf_file_size.store(file_size, std::sync::atomic::Ordering::Release);
        }
        drop(mmap_arc);

        // SAFETY: scan_buf holds at least `needed` valid f32 elements.
        let buf_ptr = buf_guard.as_ptr();
        let floats: &[f32] = unsafe { std::slice::from_raw_parts(buf_ptr, needed) };
        let total_rows = needed / query_dim;
        let mut result = topk_heap_on_floats(floats, total_rows, query_dim, computer, k);
        drop(buf_guard);
        result.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result.truncate(k);
        Ok(Some(result))
    }

    /// Zero-copy parallel TopK for a FixedList column.
    ///
    /// Runs directly on the OS mmap — no Arrow construction, no 512MB memcpy.
    /// Returns `Some(Vec<(global_row_idx, distance)>)` sorted ascending, or
    /// `None` if the column is not found / not FixedList / RG requires fallback.
    pub fn topk_fixedlist_direct(
        &self,
        col_name: &str,
        computer: &crate::query::vector_ops::DistanceComputer,
        k: usize,
    ) -> io::Result<Option<Vec<(usize, f32)>>> {
        use crate::query::vector_ops::topk_heap_on_floats;

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let is_f16 = schema.columns[col_idx].1 == ColumnType::Float16List;
        if schema.columns[col_idx].1 != ColumnType::FixedList && !is_f16 {
            return Ok(None);
        }

        let query_dim = computer.query.len();

        // Get Arc<Mmap> and immediately release the write lock.
        let file_guard = self.file.read();
        let file = match file_guard.as_ref() {
            Some(f) => f,
            None => return Ok(None),
        };
        let mmap_arc = self.mmap_cache.write().get_mmap_arc(file)?;
        drop(file_guard);

        let mmap: &[u8] = &mmap_arc;
        let null_bitmap_len_fn = |rg_rows: usize| (rg_rows + 7) / 8;

        // ── PASS 1: validate all RGs, collect descriptors ──────────────────
        struct RgDesc { count: usize, float_abs: usize, byte_len: usize }
        let mut rg_descs: Vec<Option<RgDesc>> = Vec::with_capacity(footer.row_groups.len());
        let mut total_active: usize = 0;
        // co_idx tracks position in footer.col_offsets independently of rg_idx.
        // Empty RGs (row_count==0, e.g. the initial RG from CREATE TABLE) never
        // push a col_offsets entry, so rg_idx and co_idx can diverge.
        let mut co_idx: usize = 0;

        for (_rg_idx, rg_meta) in footer.row_groups.iter().enumerate() {
            let rg_active = rg_meta.active_rows() as usize;
            total_active += rg_active;

            if rg_meta.row_count == 0 { rg_descs.push(None); continue; }
            let rg_rows = rg_meta.row_count as usize;

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap.len() { return Ok(None); }
            let rg_bytes = &mmap[rg_meta.offset as usize..rg_end];

            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { 1 };
            let encoding_ver  = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };

            if compress_flag != RG_COMPRESS_NONE
                || encoding_ver < 1
                || rg_meta.deletion_count > 0
                || co_idx >= footer.col_offsets.len()
                || col_idx >= footer.col_offsets[co_idx].len()
            {
                return Ok(None);
            }

            let rg_body_abs = (rg_meta.offset + 32) as usize;
            let col_abs  = rg_body_abs + footer.col_offsets[co_idx][col_idx] as usize;
            let data_abs = col_abs + null_bitmap_len_fn(rg_rows);

            // FixedList/Float16List layout: [encoding:u8][count:u64][dim:u32][elem * count * dim]
            if data_abs + 13 > mmap.len() { return Ok(None); }
            if mmap[data_abs] != COL_ENCODING_PLAIN { return Ok(None); }

            let count = u64::from_le_bytes(mmap[data_abs+1..data_abs+9].try_into().unwrap()) as usize;
            let dim   = u32::from_le_bytes(mmap[data_abs+9..data_abs+13].try_into().unwrap()) as usize;

            if count == 0 || dim == 0 { rg_descs.push(None); continue; }
            if dim != query_dim { return Ok(None); }

            let elem_bytes = if is_f16 { 2 } else { 4 };
            let float_abs = data_abs + 13;
            let byte_len  = count * dim * elem_bytes;
            if float_abs + byte_len > mmap.len() { return Ok(None); }

            rg_descs.push(Some(RgDesc { count, float_abs, byte_len }));
            co_idx += 1;
        }

        if total_active == 0 { return Ok(Some(vec![])); }

        let file_size = mmap.len() as u64;

        // ── PASS 2 (f16): cache raw f16 bytes, decode per-element during topk ─
        if is_f16 {
            let f16_needed = total_active * query_dim * 2;
            let mut f16_guard = self.scan_buf_f16.lock().unwrap();
            let f16_cached   = self.scan_buf_f16_file_size.load(std::sync::atomic::Ordering::Acquire);
            let f16_cg       = self.scan_buf_f16_col.lock().unwrap();
            let f16_hit      = f16_cached == file_size
                && f16_guard.len() >= f16_needed
                && f16_cg.as_str() == col_name;
            drop(f16_cg);

            if !f16_hit {
                let cur = f16_guard.len();
                if f16_guard.capacity() < f16_needed {
                    f16_guard.reserve(f16_needed - cur);
                }
                unsafe { f16_guard.set_len(f16_needed); }
                let f16_ptr = f16_guard.as_mut_ptr();
                let mut filled = 0usize;
                for desc in rg_descs.iter() {
                    let Some(d) = desc else { continue };
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            mmap.as_ptr().add(d.float_abs),
                            f16_ptr.add(filled),
                            d.byte_len,
                        );
                    }
                    filled += d.byte_len;
                }
                let mut cg = self.scan_buf_f16_col.lock().unwrap();
                cg.clear(); cg.push_str(col_name);
                drop(cg);
                self.scan_buf_f16_file_size.store(file_size, std::sync::atomic::Ordering::Release);
            }
            drop(mmap_arc);

            let f16_ptr   = f16_guard.as_ptr();
            let f16_slice = unsafe { std::slice::from_raw_parts(f16_ptr, f16_needed) };
            let mut result = crate::query::vector_ops::topk_heap_on_f16_bytes(
                f16_slice, total_active, query_dim, computer, k);
            drop(f16_guard);
            result.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            result.truncate(k);
            return Ok(Some(result));
        }

        // ── PASS 2 (f32): fill scan_buf and run ONE topk scan ────────────────
        let needed    = total_active * query_dim;
        let mut buf_guard = self.scan_buf.lock().unwrap();
        let cached_size = self.scan_buf_file_size.load(std::sync::atomic::Ordering::Acquire);
        let col_guard   = self.scan_buf_col.lock().unwrap();
        let cache_hit   = cached_size == file_size
            && buf_guard.len() >= needed
            && col_guard.as_str() == col_name;
        drop(col_guard);

        if !cache_hit {
            let cur = buf_guard.len();
            if buf_guard.capacity() < needed {
                buf_guard.reserve(needed - cur);
            }
            unsafe { buf_guard.set_len(needed); }
            let buf_ptr = buf_guard.as_mut_ptr();
            let mut filled_floats = 0usize;
            for desc in rg_descs.iter() {
                let Some(d) = desc else { continue };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        mmap.as_ptr().add(d.float_abs),
                        (buf_ptr as *mut u8).add(filled_floats * 4),
                        d.byte_len,
                    );
                }
                filled_floats += d.count * query_dim;
            }
            let mut cg = self.scan_buf_col.lock().unwrap();
            cg.clear(); cg.push_str(col_name);
            drop(cg);
            self.scan_buf_file_size.store(file_size, std::sync::atomic::Ordering::Release);
        }
        drop(mmap_arc);

        let buf_ptr = buf_guard.as_ptr();
        let floats: &[f32] = unsafe { std::slice::from_raw_parts(buf_ptr, needed) };
        let total_rows = needed / query_dim;
        let mut result = topk_heap_on_floats(floats, total_rows, query_dim, computer, k);
        drop(buf_guard);
        result.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result.truncate(k);
        Ok(Some(result))
    }

    /// Batch parallel TopK for a FixedList column — N queries in one call.
    ///
    /// Loads `scan_buf` once, then runs all N queries in parallel via Rayon
    /// (outer parallelism over queries, sequential inner scan per query).
    /// This is significantly faster than N sequential `topk_fixedlist_direct` calls
    /// because the mmap→heap copy happens only once regardless of N.
    ///
    /// `queries`: raw LE f32 bytes, `n_queries × dim`, row-major.
    /// Returns `Some(Vec<Vec<(row_idx, dist)>>)` of length `n_queries`,
    /// each sorted ascending, or `None` to fall back to the Arrow path.
    pub fn batch_topk_fixedlist_direct(
        &self,
        col_name: &str,
        queries: &[f32],
        n_queries: usize,
        k: usize,
        metric: crate::query::vector_ops::DistanceMetric,
    ) -> io::Result<Option<Vec<Vec<(usize, f32)>>>> {
        use crate::query::vector_ops::batch_topk_on_floats;

        if n_queries == 0 || queries.len() == 0 {
            return Ok(Some(vec![vec![]; n_queries]));
        }
        let query_dim = queries.len() / n_queries;
        if query_dim == 0 || queries.len() != n_queries * query_dim {
            return Ok(None);
        }

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let is_f16_batch = schema.columns[col_idx].1 == ColumnType::Float16List;
        if schema.columns[col_idx].1 != ColumnType::FixedList && !is_f16_batch {
            return Ok(None);
        }

        let file_guard = self.file.read();
        let file = match file_guard.as_ref() { Some(f) => f, None => return Ok(None) };
        let mmap_arc = self.mmap_cache.write().get_mmap_arc(file)?;
        drop(file_guard);
        let mmap: &[u8] = &mmap_arc;
        let null_bitmap_len_fn = |rg_rows: usize| (rg_rows + 7) / 8;

        // ── PASS 1: validate all RGs, collect descriptors ──────────────────
        struct RgDesc { count: usize, float_abs: usize, byte_len: usize }
        let mut rg_descs: Vec<Option<RgDesc>> = Vec::with_capacity(footer.row_groups.len());
        let mut total_active: usize = 0;
        // co_idx tracks position in footer.col_offsets independently of rg_idx.
        // Empty RGs (row_count==0) never push a col_offsets entry.
        let mut co_idx: usize = 0;

        for (_rg_idx, rg_meta) in footer.row_groups.iter().enumerate() {
            let rg_active = rg_meta.active_rows() as usize;
            total_active += rg_active;

            if rg_meta.row_count == 0 { rg_descs.push(None); continue; }
            let rg_rows = rg_meta.row_count as usize;

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap.len() { return Ok(None); }
            let rg_bytes = &mmap[rg_meta.offset as usize..rg_end];

            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { 1 };
            let encoding_ver  = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };

            if compress_flag != RG_COMPRESS_NONE
                || encoding_ver < 1
                || rg_meta.deletion_count > 0
                || co_idx >= footer.col_offsets.len()
                || col_idx >= footer.col_offsets[co_idx].len()
            { return Ok(None); }

            let rg_body_abs = (rg_meta.offset + 32) as usize;
            let col_abs  = rg_body_abs + footer.col_offsets[co_idx][col_idx] as usize;
            let data_abs = col_abs + null_bitmap_len_fn(rg_rows);

            if data_abs + 13 > mmap.len() { return Ok(None); }
            if mmap[data_abs] != COL_ENCODING_PLAIN { return Ok(None); }

            let count = u64::from_le_bytes(mmap[data_abs+1..data_abs+9].try_into().unwrap()) as usize;
            let dim   = u32::from_le_bytes(mmap[data_abs+9..data_abs+13].try_into().unwrap()) as usize;

            if count == 0 || dim == 0 { rg_descs.push(None); continue; }
            if dim != query_dim { return Ok(None); }

            let elem_bytes_b = if is_f16_batch { 2 } else { 4 };
            let float_abs = data_abs + 13;
            let byte_len  = count * dim * elem_bytes_b;
            if float_abs + byte_len > mmap.len() { return Ok(None); }
            rg_descs.push(Some(RgDesc { count, float_abs, byte_len }));
            co_idx += 1;
        }

        if total_active == 0 {
            return Ok(Some(vec![vec![]; n_queries]));
        }

        let file_size = mmap.len() as u64;

        // ── PASS 2 (f16 batch): cache raw f16 bytes, decode per-row during topk
        if is_f16_batch {
            let f16_needed = total_active * query_dim * 2;
            let mut f16_guard = self.scan_buf_f16.lock().unwrap();
            let f16_cached   = self.scan_buf_f16_file_size.load(std::sync::atomic::Ordering::Acquire);
            let f16_cg       = self.scan_buf_f16_col.lock().unwrap();
            let f16_hit      = f16_cached == file_size
                && f16_guard.len() >= f16_needed
                && f16_cg.as_str() == col_name;
            drop(f16_cg);

            if !f16_hit {
                let cur = f16_guard.len();
                if f16_guard.capacity() < f16_needed {
                    f16_guard.reserve(f16_needed - cur);
                }
                unsafe { f16_guard.set_len(f16_needed); }
                let f16_ptr = f16_guard.as_mut_ptr();
                let mut filled = 0usize;
                for desc in rg_descs.iter() {
                    let Some(d) = desc else { continue };
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            mmap.as_ptr().add(d.float_abs),
                            f16_ptr.add(filled),
                            d.byte_len,
                        );
                    }
                    filled += d.byte_len;
                }
                let mut cg = self.scan_buf_f16_col.lock().unwrap();
                cg.clear(); cg.push_str(col_name);
                drop(cg);
                self.scan_buf_f16_file_size.store(file_size, std::sync::atomic::Ordering::Release);
            }
            drop(mmap_arc);

            let f16_ptr   = f16_guard.as_ptr();
            let f16_slice = unsafe { std::slice::from_raw_parts(f16_ptr, f16_needed) };
            let results = crate::query::vector_ops::batch_topk_on_f16_bytes(
                f16_slice, total_active, query_dim, queries, n_queries, k, metric);
            drop(f16_guard);
            return Ok(Some(results));
        }

        // ── PASS 2 (f32 batch): fill scan_buf once, run all N queries ────────
        let needed    = total_active * query_dim;
        let mut buf_guard = self.scan_buf.lock().unwrap();
        let cached_size = self.scan_buf_file_size.load(std::sync::atomic::Ordering::Acquire);
        let col_guard   = self.scan_buf_col.lock().unwrap();
        let cache_hit   = cached_size == file_size
            && buf_guard.len() >= needed
            && col_guard.as_str() == col_name;
        drop(col_guard);

        if !cache_hit {
            if buf_guard.capacity() < needed {
                let cur = buf_guard.len();
                buf_guard.reserve(needed - cur);
            }
            unsafe { buf_guard.set_len(needed); }
            let buf_ptr = buf_guard.as_mut_ptr();
            let mut filled_floats = 0usize;
            for desc in rg_descs.iter() {
                let Some(d) = desc else { continue };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        mmap.as_ptr().add(d.float_abs),
                        (buf_ptr as *mut u8).add(filled_floats * 4),
                        d.byte_len,
                    );
                }
                filled_floats += d.count * query_dim;
            }
            let mut cg = self.scan_buf_col.lock().unwrap();
            cg.clear(); cg.push_str(col_name);
            drop(cg);
            self.scan_buf_file_size.store(file_size, std::sync::atomic::Ordering::Release);
        }
        drop(mmap_arc);

        let buf_ptr  = buf_guard.as_ptr();
        let floats: &[f32] = unsafe { std::slice::from_raw_parts(buf_ptr, needed) };
        let total_rows = needed / query_dim;

        let results = batch_topk_on_floats(floats, total_rows, query_dim, queries, n_queries, k, metric);
        drop(buf_guard);
        Ok(Some(results))
    }

    /// Batch parallel TopK for a Binary vector column — N queries in one call.
    ///
    /// Mirrors `batch_topk_fixedlist_direct` but parses the Binary column format.
    /// Returns `Some(Vec<Vec<(row_idx, dist)>>)` or `None` to fall back.
    pub fn batch_topk_binary_direct(
        &self,
        col_name: &str,
        queries: &[f32],
        n_queries: usize,
        k: usize,
        metric: crate::query::vector_ops::DistanceMetric,
    ) -> io::Result<Option<Vec<Vec<(usize, f32)>>>> {
        use crate::query::vector_ops::batch_topk_on_floats;

        if n_queries == 0 || queries.len() == 0 {
            return Ok(Some(vec![vec![]; n_queries]));
        }
        let query_dim = queries.len() / n_queries;
        if query_dim == 0 || queries.len() != n_queries * query_dim {
            return Ok(None);
        }

        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        if schema.columns[col_idx].1 != ColumnType::Binary {
            return Ok(None);
        }

        let file_guard = self.file.read();
        let file = match file_guard.as_ref() { Some(f) => f, None => return Ok(None) };
        let mmap_arc = self.mmap_cache.write().get_mmap_arc(file)?;
        drop(file_guard);
        let mmap: &[u8] = &mmap_arc;

        // ── PASS 1: validate all RGs ────────────────────────────────────────
        let total_active: usize = footer.row_groups.iter().map(|rg| rg.active_rows() as usize).sum();
        if total_active == 0 {
            return Ok(Some(vec![vec![]; n_queries]));
        }

        struct RgDesc { count: usize, dim: usize, data_start: usize, byte_len: usize }
        let mut rg_descs: Vec<Option<RgDesc>> = Vec::with_capacity(footer.row_groups.len());

        for (rg_idx, rg_meta) in footer.row_groups.iter().enumerate() {
            if rg_meta.row_count == 0 { rg_descs.push(None); continue; }
            let rg_rows = rg_meta.row_count as usize;

            let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
            if rg_end > mmap.len() { return Ok(None); }
            let rg_bytes = &mmap[rg_meta.offset as usize..rg_end];
            let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { 1 };
            let encoding_ver  = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };

            if compress_flag != RG_COMPRESS_NONE || encoding_ver < 1
                || rg_meta.deletion_count > 0
                || rg_idx >= footer.col_offsets.len()
                || col_idx >= footer.col_offsets[rg_idx].len()
            { return Ok(None); }

            let rg_body_abs  = (rg_meta.offset + 32) as usize;
            let null_bm_len  = (rg_rows + 7) / 8;
            let col_abs      = rg_body_abs + footer.col_offsets[rg_idx][col_idx] as usize;
            let data_abs     = col_abs + null_bm_len;

            if data_abs + 9 > mmap.len() { return Ok(None); }
            if mmap[data_abs] != COL_ENCODING_PLAIN { return Ok(None); }

            let count = u64::from_le_bytes(mmap[data_abs+1..data_abs+9].try_into().unwrap()) as usize;
            if count == 0 { rg_descs.push(None); continue; }

            let off_base = data_abs + 9;
            if off_base + 8 > mmap.len() { return Ok(None); }
            let off0 = u32::from_le_bytes(mmap[off_base..off_base+4].try_into().unwrap()) as usize;
            let off1 = u32::from_le_bytes(mmap[off_base+4..off_base+8].try_into().unwrap()) as usize;
            if off1 <= off0 || (off1 - off0) % 4 != 0 { return Ok(None); }
            let dim = (off1 - off0) / 4;
            if dim != query_dim { return Ok(None); }

            let data_start = off_base + (count + 1) * 4 + 8;
            let byte_len   = count * dim * 4;
            if data_start + byte_len > mmap.len() { return Ok(None); }
            rg_descs.push(Some(RgDesc { count, dim, data_start, byte_len }));
        }

        // ── PASS 2: fill scan_buf once, run all N queries in parallel ───────
        let needed    = total_active * query_dim;
        let file_size = mmap.len() as u64;
        let mut buf_guard = self.scan_buf.lock().unwrap();
        let cached_size = self.scan_buf_file_size.load(std::sync::atomic::Ordering::Acquire);
        let col_guard   = self.scan_buf_col.lock().unwrap();
        let cache_hit   = cached_size == file_size
            && buf_guard.len() >= needed
            && col_guard.as_str() == col_name;
        drop(col_guard);

        if !cache_hit {
            if buf_guard.capacity() < needed {
                let cur = buf_guard.len();
                buf_guard.reserve(needed - cur);
            }
            unsafe { buf_guard.set_len(needed); }
            let buf_ptr = buf_guard.as_mut_ptr();
            let mut filled_floats = 0usize;
            for desc in rg_descs.iter() {
                let Some(d) = desc else { continue };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        mmap.as_ptr().add(d.data_start),
                        (buf_ptr as *mut u8).add(filled_floats * 4),
                        d.byte_len,
                    );
                }
                filled_floats += d.count * d.dim;
            }
            let mut cg = self.scan_buf_col.lock().unwrap();
            cg.clear(); cg.push_str(col_name);
            drop(cg);
            self.scan_buf_file_size.store(file_size, std::sync::atomic::Ordering::Release);
        }
        drop(mmap_arc);

        let buf_ptr  = buf_guard.as_ptr();
        let floats: &[f32] = unsafe { std::slice::from_raw_parts(buf_ptr, needed) };
        let total_rows = needed / query_dim;

        let results = batch_topk_on_floats(floats, total_rows, query_dim, queries, n_queries, k, metric);
        drop(buf_guard);
        Ok(Some(results))
    }

}
