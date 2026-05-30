// Insert, delete, read_row_by_id, column management, save, persist

use std::sync::RwLock as StdRwLock;

static PENDING_DELETES: StdRwLock<Option<HashMap<PathBuf, Vec<u8>>>> = StdRwLock::new(None);

fn global_pending_deletes() -> &'static StdRwLock<Option<HashMap<PathBuf, Vec<u8>>>> {
    let mut guard = PENDING_DELETES.write().unwrap();
    if guard.is_none() {
        *guard = Some(HashMap::new());
    }
    drop(guard);
    &PENDING_DELETES
}

/// Apply pending delete state from global map to file on disk.
/// Called on fresh open so reads see the latest state.
/// Returns Ok(()) even if no pending state exists.
pub fn apply_pending_deletes(path: &std::path::Path) -> io::Result<()> {
    let buf = {
        let pending = global_pending_deletes().read().unwrap();
        pending.as_ref().and_then(|m| m.get(path).cloned())
    };
    let buf = match buf {
        Some(b) => b,
        None => return Ok(()),
    };
    // Parse pending format: "APXP"[rg_count:u32][(rg_i:u32, offset:u64, len:u32, del_bytes)...][footer_off:u64][footer_len:u32][footer_bytes...]
    if buf.len() < 8 || &buf[0..4] != b"APXP" {
        return Ok(());
    }
    let mut pos = 4;
    let rg_count = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let mut rg_writes: Vec<(usize, u64, Vec<u8>)> = Vec::new();
    for _ in 0..rg_count {
        if pos + 12 > buf.len() {
            break;
        }
        let rg_i = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let offset = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + len > buf.len() {
            break;
        }
        rg_writes.push((rg_i, offset, buf[pos..pos + len].to_vec()));
        pos += len;
    }
    if pos + 12 > buf.len() {
        return Ok(());
    }
    let footer_offset = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
    pos += 8;
    let footer_len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    if pos + footer_len > buf.len() {
        return Ok(());
    }
    let footer_bytes = &buf[pos..pos + footer_len];
    // Write deletion vectors, footer, and active row count to disk.
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?;
    for (rg_i, offset, del_bytes) in &rg_writes {
        file.seek(SeekFrom::Start(*offset))?;
        file.write_all(del_bytes)?;
    }
    file.seek(SeekFrom::Start(footer_offset))?;
    file.write_all(footer_bytes)?;
    if let Ok(footer) = V4Footer::from_bytes(footer_bytes) {
        let active_rows: u64 = footer
            .row_groups
            .iter()
            .map(|rg| rg.row_count.saturating_sub(rg.deletion_count) as u64)
            .sum();
        let mut header_bytes = [0u8; HEADER_SIZE];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut header_bytes)?;
        let mut header = OnDemandHeader::from_bytes(&header_bytes)?;
        header.row_count = active_rows;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header.to_bytes())?;
    }
    file.flush()?;
    // Remove from global map after checkpoint
    global_pending_deletes()
        .write()
        .unwrap()
        .as_mut()
        .map(|m| m.remove(path));
    Ok(())
}

impl OnDemandStorage {
    /// Read IDs for specific global row indices from mmap.
    /// Returns Vec<u64> of IDs corresponding to the given indices.
    pub fn get_ids_for_global_indices_mmap(&self, indices: &[usize]) -> io::Result<Vec<u64>> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok(vec![]),
        };
        // Build RG bounds
        let mut rg_bounds: Vec<(usize, usize)> = Vec::new();
        let mut cumulative = 0usize;
        for rg in &footer.row_groups {
            let n = rg.row_count as usize;
            rg_bounds.push((cumulative, cumulative + n));
            cumulative += n;
        }

        let file_guard = self.file.read();
        let file = file_guard
            .as_ref()
            .ok_or_else(|| err_not_conn("File not open for ID read"))?;
        let mut mmap_guard = self.mmap_cache.write();
        let mmap_ref = mmap_guard.get_or_create(file)?;

        let mut result = Vec::with_capacity(indices.len());
        for &idx in indices {
            let mut found = false;
            for (rg_i, &(start, end)) in rg_bounds.iter().enumerate() {
                if idx >= start && idx < end {
                    let local = idx - start;
                    let rg_meta = &footer.row_groups[rg_i];
                    let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                    if rg_end <= mmap_ref.len() {
                        let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
                        let compress_flag = if rg_bytes.len() >= 32 {
                            rg_bytes[28]
                        } else {
                            RG_COMPRESS_NONE
                        };
                        let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
                        let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
                        let off = local * 8;
                        if off + 8 <= body.len() {
                            let id = u64::from_le_bytes(body[off..off + 8].try_into().unwrap());
                            result.push(id);
                            found = true;
                        }
                    }
                    break;
                }
            }
            if !found {
                result.push(0);
            }
        }
        Ok(result)
    }
    // ========================================================================
    // Write APIs
    // ========================================================================

    /// Insert typed columns directly
    pub fn insert_typed(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        // Determine row count as maximum across all columns (for heterogeneous schemas)
        let row_count = int_columns
            .values()
            .map(|v| v.len())
            .max()
            .unwrap_or(0)
            .max(float_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(string_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(binary_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(bool_columns.values().map(|v| v.len()).max().unwrap_or(0));

        if row_count == 0 {
            return Ok(Vec::new());
        }

        // Allocate IDs atomically
        let start_id = self.next_id.fetch_add(row_count as u64, Ordering::SeqCst);
        let ids: Vec<u64> = (start_id..start_id + row_count as u64).collect();

        // Ensure schema has all columns, padding existing rows with defaults for new columns
        {
            let mut schema = self.schema.write();
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();
            let existing_row_count = self.ids.read().len();

            // First, add any new columns to schema
            for name in int_columns.keys() {
                schema.add_column(name, ColumnType::Int64);
            }
            for name in float_columns.keys() {
                schema.add_column(name, ColumnType::Float64);
            }
            for name in string_columns.keys() {
                schema.add_column(name, ColumnType::String);
            }
            for name in binary_columns.keys() {
                schema.add_column(name, ColumnType::Binary);
            }
            for name in bool_columns.keys() {
                schema.add_column(name, ColumnType::Bool);
            }

            // Then, ensure columns vector matches schema (using correct types from schema)
            while columns.len() < schema.column_count() {
                let col_idx = columns.len();
                let (_, col_type) = &schema.columns[col_idx];
                let mut col = ColumnData::new(*col_type);
                // Pad with defaults for existing rows
                if existing_row_count > 0 {
                    match &mut col {
                        ColumnData::Int64(v) => v.resize(existing_row_count, 0),
                        ColumnData::Float64(v) => v.resize(existing_row_count, 0.0),
                        ColumnData::String { offsets, .. } => {
                            for _ in 0..existing_row_count {
                                offsets.push(0);
                            }
                        }
                        ColumnData::Binary { offsets, .. } => {
                            for _ in 0..existing_row_count {
                                offsets.push(0);
                            }
                        }
                        ColumnData::Bool { len, .. } => {
                            *len = existing_row_count;
                        }
                        _ => {}
                    }
                }
                columns.push(col);
                nulls.push(Vec::new());
            }
        }

        // OPTIMIZATION: combine ID append + column append + metadata updates
        // to minimize lock acquire/release overhead
        let col_count_for_header;
        {
            let schema = self.schema.read();
            col_count_for_header = schema.column_count() as u32;
            let mut ids_guard = self.ids.write();
            let start_idx = ids_guard.len();
            ids_guard.extend_from_slice(&ids);
            let total_rows_after = ids_guard.len();
            drop(ids_guard);

            // Append column data (schema read lock still held)
            let mut columns = self.columns.write();
            for (name, values) in int_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_i64(&values);
                }
            }
            for (name, values) in float_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_f64(&values);
                }
            }
            for (name, values) in string_columns {
                if let Some(idx) = schema.get_index(&name) {
                    if idx < columns.len() {
                        match &columns[idx] {
                            ColumnData::String { .. } => {
                                columns[idx].extend_strings(&values);
                            }
                            ColumnData::StringDict {
                                indices,
                                dict_offsets,
                                dict_data,
                            } => {
                                let mut new_offsets = vec![0u32];
                                let mut new_data = Vec::new();
                                for &dict_idx in indices {
                                    if dict_idx == 0 {
                                        new_offsets.push(new_data.len() as u32);
                                    } else {
                                        let actual_idx = (dict_idx - 1) as usize;
                                        if actual_idx + 1 < dict_offsets.len() {
                                            let start = dict_offsets[actual_idx] as usize;
                                            let end = dict_offsets[actual_idx + 1] as usize;
                                            new_data.extend_from_slice(&dict_data[start..end]);
                                        }
                                        new_offsets.push(new_data.len() as u32);
                                    }
                                }
                                columns[idx] = ColumnData::String {
                                    offsets: new_offsets,
                                    data: new_data,
                                };
                                columns[idx].extend_strings(&values);
                            }
                            _ => {
                                columns[idx] = ColumnData::new(ColumnType::String);
                                columns[idx].extend_strings(&values);
                            }
                        }
                    }
                }
            }
            for (name, values) in binary_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_bytes(&values);
                }
            }
            for (name, values) in bool_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_bools(&values);
                }
            }
            drop(columns);
            drop(schema);

            // Update id_to_idx if already built (avoid rebuilding)
            {
                let mut id_to_idx = self.id_to_idx.write();
                if let Some(map) = id_to_idx.as_mut() {
                    for (i, &id) in ids.iter().enumerate() {
                        map.insert(id, start_idx + i);
                    }
                }
            }

            // Extend deleted bitmap
            {
                let mut deleted = self.deleted.write();
                let new_len = (total_rows_after + 7) / 8;
                deleted.resize(new_len, 0);
            }
        }

        // Update header
        {
            let mut header = self.header.write();
            header.row_count += row_count as u64;
            header.column_count = col_count_for_header;
            header.modified_at = chrono::Utc::now().timestamp();
        }

        // Update active count (new rows are not deleted)
        self.active_count
            .fetch_add(row_count as u64, Ordering::Relaxed);

        Ok(ids)
    }

    /// Insert typed columns with EXPLICIT IDs (used during delta compaction)
    /// This preserves the original IDs from delta file instead of generating new ones
    fn insert_typed_with_ids(
        &self,
        ids: &[u64],
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
    ) -> io::Result<()> {
        let row_count = ids.len();
        if row_count == 0 {
            return Ok(());
        }

        // Update next_id to be greater than any provided ID
        for &id in ids {
            let current = self.next_id.load(Ordering::SeqCst);
            if id >= current {
                self.next_id.store(id + 1, Ordering::SeqCst);
            }
        }

        // Ensure schema has all columns
        {
            let mut schema = self.schema.write();
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();
            let existing_row_count = self.ids.read().len();

            for name in int_columns.keys() {
                schema.add_column(name, ColumnType::Int64);
            }
            for name in float_columns.keys() {
                schema.add_column(name, ColumnType::Float64);
            }
            for name in string_columns.keys() {
                schema.add_column(name, ColumnType::String);
            }
            for name in binary_columns.keys() {
                schema.add_column(name, ColumnType::Binary);
            }
            for name in bool_columns.keys() {
                schema.add_column(name, ColumnType::Bool);
            }

            while columns.len() < schema.column_count() {
                let col_idx = columns.len();
                let (_, col_type) = &schema.columns[col_idx];
                let mut col = ColumnData::new(*col_type);
                if existing_row_count > 0 {
                    match &mut col {
                        ColumnData::Int64(v) => v.resize(existing_row_count, 0),
                        ColumnData::Float64(v) => v.resize(existing_row_count, 0.0),
                        ColumnData::String { offsets, .. } => {
                            for _ in 0..existing_row_count {
                                offsets.push(0);
                            }
                        }
                        ColumnData::Binary { offsets, .. } => {
                            for _ in 0..existing_row_count {
                                offsets.push(0);
                            }
                        }
                        ColumnData::Bool { len, .. } => {
                            *len = existing_row_count;
                        }
                        ColumnData::StringDict { indices, .. } => {
                            indices.resize(existing_row_count, 0);
                        }
                        ColumnData::FixedList { .. } => {} // pads implicitly
                        ColumnData::Float16List { .. } => {} // pads implicitly
                    }
                }
                columns.push(col);
                nulls.push(Vec::new());
            }
        }

        // Append IDs
        {
            let mut ids_vec = self.ids.write();
            ids_vec.extend_from_slice(ids);
        }

        // Append column data
        {
            let schema = self.schema.read();
            let mut columns = self.columns.write();

            for (name, values) in int_columns {
                if let Some(idx) = schema.get_index(&name) {
                    if idx < columns.len() {
                        columns[idx].extend_i64(&values);
                    }
                }
            }

            for (name, values) in float_columns {
                if let Some(idx) = schema.get_index(&name) {
                    if idx < columns.len() {
                        columns[idx].extend_f64(&values);
                    }
                }
            }

            for (name, values) in string_columns {
                if let Some(idx) = schema.get_index(&name) {
                    if idx < columns.len() {
                        match &columns[idx] {
                            ColumnData::String { .. } => {
                                columns[idx].extend_strings(&values);
                            }
                            ColumnData::StringDict {
                                indices,
                                dict_offsets,
                                dict_data,
                            } => {
                                let mut new_offsets = vec![0u32];
                                let mut new_data = Vec::new();
                                for &dict_idx in indices {
                                    if dict_idx == 0 {
                                        new_offsets.push(new_data.len() as u32);
                                    } else {
                                        let actual_idx = (dict_idx - 1) as usize;
                                        if actual_idx + 1 < dict_offsets.len() {
                                            let start = dict_offsets[actual_idx] as usize;
                                            let end = dict_offsets[actual_idx + 1] as usize;
                                            new_data.extend_from_slice(&dict_data[start..end]);
                                        }
                                        new_offsets.push(new_data.len() as u32);
                                    }
                                }
                                columns[idx] = ColumnData::String {
                                    offsets: new_offsets,
                                    data: new_data,
                                };
                                columns[idx].extend_strings(&values);
                            }
                            _ => {
                                columns[idx] = ColumnData::new(ColumnType::String);
                                columns[idx].extend_strings(&values);
                            }
                        }
                    }
                }
            }

            for (name, values) in bool_columns {
                if let Some(idx) = schema.get_index(&name) {
                    if idx < columns.len() {
                        columns[idx].extend_bools(&values);
                    }
                }
            }

            // Pad columns that don't have new data
            for col_idx in 0..columns.len() {
                let expected_len = self.ids.read().len();
                let current_len = columns[col_idx].len();
                if current_len < expected_len {
                    let pad_count = expected_len - current_len;
                    match &mut columns[col_idx] {
                        ColumnData::Int64(v) => v.extend(std::iter::repeat(0).take(pad_count)),
                        ColumnData::Float64(v) => v.extend(std::iter::repeat(0.0).take(pad_count)),
                        ColumnData::String { offsets, .. } => {
                            for _ in 0..pad_count {
                                offsets.push(*offsets.last().unwrap_or(&0));
                            }
                        }
                        ColumnData::Binary { offsets, .. } => {
                            for _ in 0..pad_count {
                                offsets.push(*offsets.last().unwrap_or(&0));
                            }
                        }
                        ColumnData::FixedList { .. } => {} // pads implicitly
                        ColumnData::Float16List { .. } => {} // pads implicitly
                        ColumnData::Bool { data, len } => {
                            for _ in 0..pad_count {
                                let byte_idx = *len / 8;
                                if byte_idx >= data.len() {
                                    data.push(0);
                                }
                                *len += 1;
                            }
                        }
                        ColumnData::StringDict { indices, .. } => {
                            indices.extend(std::iter::repeat(0).take(pad_count));
                        }
                    }
                }
            }
        }

        // Update header
        {
            let mut header = self.header.write();
            header.row_count = self.ids.read().len() as u64;
            header.column_count = self.schema.read().column_count() as u32;
        }

        // Extend deleted bitmap
        {
            let mut deleted = self.deleted.write();
            let new_len = (self.ids.read().len() + 7) / 8;
            deleted.resize(new_len, 0);
        }

        self.active_count
            .fetch_add(row_count as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Insert typed columns with explicit NULL tracking for heterogeneous schemas
    pub fn insert_typed_with_nulls(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
        null_positions: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        self.insert_typed_with_nulls_full(
            int_columns,
            float_columns,
            string_columns,
            binary_columns,
            HashMap::new(),
            bool_columns,
            null_positions,
        )
    }

    /// Full version with fixedlist_columns support
    pub fn insert_typed_with_nulls_full(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        binary_columns: HashMap<String, Vec<Vec<u8>>>,
        fixedlist_columns: HashMap<String, Vec<Vec<u8>>>,
        bool_columns: HashMap<String, Vec<bool>>,
        null_positions: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        // Determine row count as maximum across all columns
        let row_count = int_columns
            .values()
            .map(|v| v.len())
            .max()
            .unwrap_or(0)
            .max(float_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(string_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(binary_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(
                fixedlist_columns
                    .values()
                    .map(|v| v.len())
                    .max()
                    .unwrap_or(0),
            )
            .max(bool_columns.values().map(|v| v.len()).max().unwrap_or(0));

        if row_count == 0 {
            return Ok(Vec::new());
        }

        // Allocate IDs atomically
        let start_id = self.next_id.fetch_add(row_count as u64, Ordering::SeqCst);
        let ids: Vec<u64> = (start_id..start_id + row_count as u64).collect();

        // Ensure schema has all columns and track column indices
        // For new columns, pad existing rows with defaults (NULL-like values)
        let mut col_name_to_idx: HashMap<String, usize> = HashMap::new();
        {
            let mut schema = self.schema.write();
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();
            let existing_row_count = self.ids.read().len();

            for name in int_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Int64);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Int64);
                    // Pad with defaults for existing rows
                    if let ColumnData::Int64(v) = &mut col {
                        v.resize(existing_row_count, 0);
                    }
                    columns.push(col);
                    // Mark all existing rows as NULL for new column
                    nulls.push(Vec::new());
                }
            }
            for name in float_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Float64);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Float64);
                    if let ColumnData::Float64(v) = &mut col {
                        v.resize(existing_row_count, 0.0);
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in string_columns.keys() {
                let idx = schema.add_column(name, ColumnType::String);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::String);
                    // Pad with empty strings for existing rows
                    if let ColumnData::String { offsets, .. } = &mut col {
                        for _ in 0..existing_row_count {
                            offsets.push(0);
                        }
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in binary_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Binary);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Binary);
                    if let ColumnData::Binary { offsets, .. } = &mut col {
                        for _ in 0..existing_row_count {
                            offsets.push(0);
                        }
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in fixedlist_columns.keys() {
                let idx = schema.add_column(name, ColumnType::FixedList);
                col_name_to_idx.insert(name.clone(), idx);
                let actual_type = schema
                    .columns
                    .get(idx)
                    .map(|(_, t)| *t)
                    .unwrap_or(ColumnType::FixedList);
                while columns.len() <= idx {
                    let col = match actual_type {
                        ColumnType::Float16List => ColumnData::Float16List {
                            data: Vec::new(),
                            dim: 0,
                        },
                        _ => ColumnData::FixedList {
                            data: Vec::new(),
                            dim: 0,
                        },
                    };
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in bool_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Bool);
                col_name_to_idx.insert(name.clone(), idx);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Bool);
                    if let ColumnData::Bool { len, .. } = &mut col {
                        *len = existing_row_count;
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
        }

        // Append IDs
        self.ids.write().extend_from_slice(&ids);

        // Append column data
        {
            let schema = self.schema.read();
            let mut columns = self.columns.write();

            for (name, values) in int_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_i64(&values);
                }
            }
            for (name, values) in float_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_f64(&values);
                }
            }
            for (name, values) in string_columns {
                if let Some(idx) = schema.get_index(&name) {
                    // StringDict columns are pre-initialized from schema type but push_string
                    // only works on String variant; convert in-place before appending.
                    if matches!(columns[idx], ColumnData::StringDict { .. }) {
                        columns[idx] = ColumnData::new(ColumnType::String);
                    }
                    for v in &values {
                        columns[idx].push_string(v);
                    }
                }
            }
            for (name, values) in binary_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in &values {
                        columns[idx].push_bytes(v);
                    }
                }
            }
            for (name, values) in fixedlist_columns {
                if let Some(idx) = schema.get_index(&name) {
                    let is_f16 = matches!(columns[idx], ColumnData::Float16List { .. });
                    for v in &values {
                        if is_f16 {
                            columns[idx].push_float16_list_from_f32(v);
                        } else {
                            columns[idx].push_fixed_list(v);
                        }
                    }
                }
            }
            for (name, values) in bool_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in values {
                        columns[idx].push_bool(v);
                    }
                }
            }
        }

        // Update null bitmaps for each column
        {
            let mut nulls = self.nulls.write();
            let base_row = self.ids.read().len() - row_count;

            for (col_name, is_null_vec) in null_positions {
                if let Some(&col_idx) = col_name_to_idx.get(&col_name) {
                    if col_idx < nulls.len() {
                        // Extend null bitmap for this column
                        let null_bitmap = &mut nulls[col_idx];
                        for (i, &is_null) in is_null_vec.iter().enumerate() {
                            if is_null {
                                let row_idx = base_row + i;
                                let byte_idx = row_idx / 8;
                                let bit_idx = row_idx % 8;
                                while null_bitmap.len() <= byte_idx {
                                    null_bitmap.push(0);
                                }
                                null_bitmap[byte_idx] |= 1 << bit_idx;
                            }
                        }
                    }
                }
            }
        }

        // Update header
        {
            let mut header = self.header.write();
            header.row_count += row_count as u64;
            header.column_count = self.schema.read().column_count() as u32;
            header.modified_at = chrono::Utc::now().timestamp();
        }

        // Update id_to_idx mapping only if it's already built
        {
            let ids_guard = self.ids.read();
            let mut id_to_idx = self.id_to_idx.write();
            if let Some(map) = id_to_idx.as_mut() {
                let start_idx = ids_guard.len() - ids.len();
                for (i, &id) in ids.iter().enumerate() {
                    map.insert(id, start_idx + i);
                }
            }
        }

        // Extend deleted bitmap with zeros for new rows
        {
            let mut deleted = self.deleted.write();
            let new_len = (self.ids.read().len() + 7) / 8;
            deleted.resize(new_len, 0);
        }

        // Update active count (new rows are not deleted)
        self.active_count
            .fetch_add(row_count as u64, Ordering::Relaxed);

        Ok(ids)
    }

    // ========================================================================
    // Delete/Update APIs
    // ========================================================================

    /// Delete a row by ID (soft delete)
    /// Returns true if the row was found and deleted
    pub fn delete(&self, id: u64) -> bool {
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();
        if let Some(&row_idx) = map.get(&id) {
            drop(id_to_idx); // Release read lock before write
            let mut deleted = self.deleted.write();
            let byte_idx = row_idx / 8;
            let bit_idx = row_idx % 8;

            // Ensure bitmap is large enough
            if byte_idx >= deleted.len() {
                deleted.resize(byte_idx + 1, 0);
            }

            // Only decrement if not already deleted
            let was_deleted = (deleted[byte_idx] >> bit_idx) & 1 == 1;
            if !was_deleted {
                self.active_count.fetch_sub(1, Ordering::Relaxed);
            }

            // Set the deleted bit
            deleted[byte_idx] |= 1 << bit_idx;
            true
        } else {
            false
        }
    }

    /// Delete an unflushed V4 memtable row in-place.
    ///
    /// This keeps the common OLTP pattern `insert -> delete same row` entirely
    /// inside the warm insert backend instead of creating a DeltaStore delete
    /// that the next append must persist before it can continue.
    pub fn delete_pending_v4_in_memory_row(&self, id: u64) -> bool {
        let pending = self.pending_v4_in_memory_rows();
        if pending == 0 {
            return false;
        }

        let ids = self.ids.read();
        let ids_len = ids.len();
        if ids_len == 0 {
            return false;
        }

        let start = ids_len.saturating_sub(pending);
        let Some(row_idx) = ids[start..]
            .iter()
            .position(|&row_id| row_id == id)
            .map(|idx| start + idx)
        else {
            return false;
        };
        drop(ids);

        let mut deleted = self.deleted.write();
        let byte_idx = row_idx / 8;
        let bit_idx = row_idx % 8;
        if byte_idx >= deleted.len() {
            deleted.resize(byte_idx + 1, 0);
        }

        let was_deleted = (deleted[byte_idx] >> bit_idx) & 1 == 1;
        if was_deleted {
            return false;
        }

        deleted[byte_idx] |= 1 << bit_idx;
        self.active_count.fetch_sub(1, Ordering::Relaxed);
        true
    }

    /// Delete multiple rows by IDs (soft delete)
    /// Returns true if all rows were found and deleted
    pub fn delete_batch(&self, ids: &[u64]) -> bool {
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();
        let mut deleted = self.deleted.write();
        let mut all_found = true;
        let mut deleted_count = 0u64;

        for &id in ids {
            if let Some(&row_idx) = map.get(&id) {
                let byte_idx = row_idx / 8;
                let bit_idx = row_idx % 8;

                if byte_idx >= deleted.len() {
                    deleted.resize(byte_idx + 1, 0);
                }

                // Only count if not already deleted
                let was_deleted = (deleted[byte_idx] >> bit_idx) & 1 == 1;
                if !was_deleted {
                    deleted_count += 1;
                }

                deleted[byte_idx] |= 1 << bit_idx;
            } else {
                all_found = false;
            }
        }

        // Update active count
        if deleted_count > 0 {
            self.active_count
                .fetch_sub(deleted_count, Ordering::Relaxed);
        }

        all_found
    }

    /// Check if a row is deleted
    pub fn is_deleted(&self, row_idx: usize) -> bool {
        let deleted = self.deleted.read();
        let byte_idx = row_idx / 8;
        let bit_idx = row_idx % 8;

        if byte_idx < deleted.len() {
            (deleted[byte_idx] >> bit_idx) & 1 == 1
        } else {
            false
        }
    }

    /// Check if an ID exists and is not deleted
    /// Also checks delta file for IDs not yet merged into base
    pub fn exists(&self, id: u64) -> bool {
        // First check base file IDs
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();
        if let Some(&row_idx) = map.get(&id) {
            if !self.is_deleted(row_idx) {
                return true;
            }
        }

        // Check delta file for IDs not yet merged
        if let Ok(Some((delta_ids, _))) = self.read_delta_data() {
            return delta_ids.contains(&id);
        }

        false
    }

    /// Get row index for an ID (None if not found or deleted)
    pub fn get_row_idx(&self, id: u64) -> Option<usize> {
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();
        if let Some(&row_idx) = map.get(&id) {
            if !self.is_deleted(row_idx) {
                Some(row_idx)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// OPTIMIZED: Read a single row by ID using O(1) index lookup.
    /// Returns HashMap of column_name -> ColumnData (single element).
    /// Supports both in-memory and mmap-only paths.
    pub fn read_row_by_id(
        &self,
        id: u64,
        column_names: Option<&[&str]>,
    ) -> io::Result<Option<HashMap<String, ColumnData>>> {
        let is_v4 = self.is_v4_format();

        let schema = self.schema.read();
        let cols_to_read: Vec<(usize, String, ColumnType)> = if let Some(names) = column_names {
            names
                .iter()
                .filter_map(|&name| {
                    if name == "_id" {
                        None
                    } else {
                        schema
                            .get_index(name)
                            .map(|idx| (idx, name.to_string(), schema.columns[idx].1))
                    }
                })
                .collect()
        } else {
            schema
                .columns
                .iter()
                .enumerate()
                .map(|(idx, (name, dtype))| (idx, name.clone(), *dtype))
                .collect()
        };
        let total_rows = self.header.read().row_count as usize;
        drop(schema);

        let mut result = HashMap::new();
        let include_id = column_names
            .map(|cols| cols.contains(&"_id"))
            .unwrap_or(true);
        if include_id {
            result.insert("_id".to_string(), ColumnData::Int64(vec![id as i64]));
        }

        if is_v4 && !self.has_v4_in_memory_data() {
            // MMAP PATH: direct max_id scan + binary search — no full ID array load
            let col_indices: Vec<usize> = cols_to_read.iter().map(|(idx, _, _)| *idx).collect();
            if let Some(footer) = self.get_or_load_footer()? {
                // Find RG via max_id + binary search within sorted RG ID array
                let file_guard = self.file.read();
                let file = file_guard
                    .as_ref()
                    .ok_or_else(|| err_not_conn("File not open"))?;
                let mut mmap_guard = self.mmap_cache.write();
                let mmap_ref = mmap_guard.get_or_create(file)?;
                let mut found_rg_i: Option<usize> = None;
                let mut local_idx_found = 0usize;
                for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
                    if rg_meta.max_id < id || rg_meta.row_count == 0 {
                        continue;
                    }
                    let rg_rows = rg_meta.row_count as usize;
                    let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                    if rg_end > mmap_ref.len() {
                        continue;
                    }
                    let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
                    let cflag = if rg_bytes.len() >= 32 {
                        rg_bytes[28]
                    } else {
                        RG_COMPRESS_NONE
                    };
                    let dec = decompress_rg_body(cflag, &rg_bytes[32..])?;
                    let body = dec.as_deref().unwrap_or(&rg_bytes[32..]);
                    if rg_rows * 8 > body.len() {
                        continue;
                    }
                    let ids_cow = bytes_as_u64_slice(body, rg_rows);
                    if let Ok(idx) = ids_cow.binary_search(&id) {
                        found_rg_i = Some(rg_i);
                        local_idx_found = idx;
                        break;
                    }
                }
                drop(mmap_guard);
                drop(file_guard);
                if let Some(rg_i) = found_rg_i {
                    let local_idx = local_idx_found;
                    // Create a single-RG footer view for scan
                    let single_rg_footer = V4Footer {
                        schema: footer.schema.clone(),
                        row_groups: vec![footer.row_groups[rg_i].clone()],
                        zone_maps: if rg_i < footer.zone_maps.len() {
                            vec![footer.zone_maps[rg_i].clone()]
                        } else {
                            vec![]
                        },
                        col_offsets: if rg_i < footer.col_offsets.len() {
                            vec![footer.col_offsets[rg_i].clone()]
                        } else {
                            vec![]
                        },
                    };
                    let (scanned, _del, col_nulls) =
                        self.scan_columns_mmap_with_nulls(&col_indices, &single_rg_footer)?;
                    let local_indices = vec![local_idx];
                    for (out_pos, (_, col_name, _)) in cols_to_read.iter().enumerate() {
                        if out_pos < scanned.len() {
                            let mut col = scanned[out_pos].clone();
                            if out_pos < col_nulls.len() && !col_nulls[out_pos].is_empty() {
                                col.apply_null_bitmap(&col_nulls[out_pos]);
                            }
                            result.insert(col_name.clone(), col.filter_by_indices(&local_indices));
                        }
                    }
                }
            }
        } else {
            let row_idx = match self.get_row_idx(id) {
                Some(idx) => idx,
                None => return Ok(None),
            };
            let indices = vec![row_idx];
            for (col_idx, col_name, col_type) in cols_to_read {
                let col_data = self
                    .read_column_scattered_auto(col_idx, col_type, &indices, total_rows, is_v4)?;
                result.insert(col_name, col_data);
            }
        }

        Ok(Some(result))
    }

    /// Read bytes from an already-locked page cache (no lock acquisition).
    /// Returns false if any required page is missing (cache miss).
    #[inline]
    fn read_from_locked_cache(
        cache: &std::collections::HashMap<u64, Box<[u8; 4096]>>,
        abs_offset: u64,
        dst: &mut [u8],
    ) -> bool {
        let len = dst.len();
        if len == 0 {
            return true;
        }
        let mut written = 0usize;
        let mut cur_off = abs_offset;
        while written < len {
            let page_num = cur_off / 4096;
            let page_off = (cur_off % 4096) as usize;
            let to_copy = (len - written).min(4096 - page_off);
            match cache.get(&page_num) {
                Some(page) => {
                    dst[written..written + to_copy]
                        .copy_from_slice(&page[page_off..page_off + to_copy]);
                    written += to_copy;
                    cur_off += to_copy as u64;
                }
                None => return false,
            }
        }
        true
    }

    /// Zero-syscall RCIX point lookup using user-space page cache.
    /// Reads file bytes via cached heap pages (pread on miss), avoiding repeated macOS
    /// mmap soft page faults (~21µs each). After warmup, all accesses hit the heap cache
    /// (~50ns each) — no page table involvement.
    pub(crate) fn retrieve_rcix(
        &self,
        id: u64,
    ) -> io::Result<Option<Vec<(String, crate::data::Value)>>> {
        use crate::data::Value;
        // Extract only the fields we need — avoids cloning the full footer
        // (zone_maps, all row_groups, all col_offsets for every RG, schema strings).
        let (rg_offset, rg_rows, rg_min_id, col_offsets_rg, col_schema) = {
            let fg = self.v4_footer.read();
            let footer = match fg.as_ref() {
                Some(f) => f,
                None => return Ok(None),
            };
            let col_count = footer.schema.column_count();
            let mut rg_i_found = None;
            for (i, rg) in footer.row_groups.iter().enumerate() {
                if rg.min_id <= id && id <= rg.max_id && rg.row_count > 0 {
                    rg_i_found = Some(i);
                    break;
                }
            }
            let rg_i = match rg_i_found {
                Some(i) => i,
                None => return Ok(None),
            };
            let rg_meta = &footer.row_groups[rg_i];
            if rg_i >= footer.col_offsets.len() || footer.col_offsets[rg_i].len() < col_count {
                return Ok(None);
            }
            // Clone only the small per-RG slice and per-column schema (names + types)
            let col_offsets_rg: Vec<u32> = footer.col_offsets[rg_i].clone();
            let col_schema: Vec<(String, ColumnType)> = footer.schema.columns.clone();
            (
                rg_meta.offset,
                rg_meta.row_count as usize,
                rg_meta.min_id,
                col_offsets_rg,
                col_schema,
            )
        };

        let col_count = col_schema.len();
        // Read RG header (32 bytes) to check compression + encoding version
        let mut rg_hdr = [0u8; 32];
        self.read_cached_bytes(rg_offset, &mut rg_hdr)?;
        if rg_hdr[28] != RG_COMPRESS_NONE || rg_hdr[29] < 1 {
            return Ok(None);
        }

        let body_base = rg_offset + 32;
        let null_bitmap_len = (rg_rows + 7) / 8;

        // O(1) local_idx: direct guess then verify
        let guess = (id.saturating_sub(rg_min_id)) as usize;
        let local_idx = if guess < rg_rows {
            let mut id_buf = [0u8; 8];
            self.read_cached_bytes(body_base + (guess * 8) as u64, &mut id_buf)?;
            if u64::from_le_bytes(id_buf) == id {
                guess
            } else {
                // Rare: non-contiguous IDs — binary search over full ID array
                let mut ids_buf = vec![0u8; rg_rows * 8];
                self.read_cached_bytes(body_base, &mut ids_buf)?;
                let ids_cow = bytes_as_u64_slice(&ids_buf, rg_rows);
                match ids_cow.binary_search(&id) {
                    Ok(i) => i,
                    Err(_) => return Ok(None),
                }
            }
        } else {
            return Ok(None);
        };

        // Deletion check
        let mut del_buf = [0u8; 1];
        self.read_cached_bytes(
            body_base + (rg_rows * 8 + local_idx / 8) as u64,
            &mut del_buf,
        )?;
        if (del_buf[0] >> (local_idx % 8)) & 1 == 1 {
            return Ok(None);
        }

        let col_offsets = &col_offsets_rg;
        // Use col_schema as schema reference (already cloned above)
        let mut result = Vec::with_capacity(col_count + 1);
        result.push(("_id".to_string(), Value::Int64(id as i64)));

        for col_idx in 0..col_count {
            let col_name = col_schema[col_idx].0.clone();
            let col_type = col_schema[col_idx].1;
            let col_start = col_offsets[col_idx] as usize;

            // Null bit
            let mut null_buf = [0u8; 1];
            self.read_cached_bytes(
                body_base + (col_start + local_idx / 8) as u64,
                &mut null_buf,
            )?;
            if (null_buf[0] >> (local_idx % 8)) & 1 == 1 {
                result.push((col_name, Value::Null));
                continue;
            }

            // Encoding byte
            let enc_off = body_base + (col_start + null_bitmap_len) as u64;
            let mut enc_buf = [0u8; 1];
            self.read_cached_bytes(enc_off, &mut enc_buf)?;
            let encoding = enc_buf[0];

            // data_bytes base = col_start + null_bitmap_len + 1 (skip encoding byte)
            let data_base = body_base + (col_start + null_bitmap_len + 1) as u64;

            let val = match (encoding, col_type) {
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
                ) => {
                    let mut v = [0u8; 8];
                    self.read_cached_bytes(data_base + (8 + local_idx * 8) as u64, &mut v)?;
                    Value::Int64(i64::from_le_bytes(v))
                }
                (COL_ENCODING_PLAIN, ColumnType::Float64 | ColumnType::Float32) => {
                    let mut v = [0u8; 8];
                    self.read_cached_bytes(data_base + (8 + local_idx * 8) as u64, &mut v)?;
                    Value::Float64(f64::from_le_bytes(v))
                }
                (COL_ENCODING_PLAIN, ColumnType::Bool) => {
                    let mut v = [0u8; 1];
                    self.read_cached_bytes(data_base + (8 + local_idx / 8) as u64, &mut v)?;
                    Value::Bool((v[0] >> (local_idx % 8)) & 1 == 1)
                }
                (COL_ENCODING_PLAIN, ColumnType::String) => {
                    let mut count_buf = [0u8; 8];
                    self.read_cached_bytes(data_base, &mut count_buf)?;
                    let count = u64::from_le_bytes(count_buf) as usize;
                    if local_idx >= count {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let mut se_buf = [0u8; 8];
                    self.read_cached_bytes(data_base + (8 + local_idx * 4) as u64, &mut se_buf)?;
                    let s = u32::from_le_bytes(se_buf[0..4].try_into().unwrap()) as usize;
                    let e = u32::from_le_bytes(se_buf[4..8].try_into().unwrap()) as usize;
                    if s > e {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let dd_start = 8 + (count + 1) * 4 + 8;
                    let mut str_buf = vec![0u8; e - s];
                    self.read_cached_bytes(data_base + (dd_start + s) as u64, &mut str_buf)?;
                    Value::String(std::str::from_utf8(&str_buf).unwrap_or("").to_string())
                }
                (COL_ENCODING_PLAIN, ColumnType::Binary) => {
                    let mut count_buf = [0u8; 8];
                    self.read_cached_bytes(data_base, &mut count_buf)?;
                    let count = u64::from_le_bytes(count_buf) as usize;
                    if local_idx >= count {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let mut se_buf = [0u8; 8];
                    self.read_cached_bytes(data_base + (8 + local_idx * 4) as u64, &mut se_buf)?;
                    let s = u32::from_le_bytes(se_buf[0..4].try_into().unwrap()) as usize;
                    let e = u32::from_le_bytes(se_buf[4..8].try_into().unwrap()) as usize;
                    if s > e {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let dd_start = 8 + (count + 1) * 4 + 8;
                    let mut bin_buf = vec![0u8; e - s];
                    self.read_cached_bytes(data_base + (dd_start + s) as u64, &mut bin_buf)?;
                    Value::Binary(bin_buf)
                }
                (COL_ENCODING_PLAIN, ColumnType::StringDict) => {
                    let mut hdr = [0u8; 16];
                    self.read_cached_bytes(data_base, &mut hdr)?;
                    let row_count = u64::from_le_bytes(hdr[0..8].try_into().unwrap()) as usize;
                    let dict_size = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
                    if local_idx >= row_count {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let mut idx_buf = [0u8; 4];
                    self.read_cached_bytes(data_base + (16 + local_idx * 4) as u64, &mut idx_buf)?;
                    let dict_idx = u32::from_le_bytes(idx_buf) as usize;
                    if dict_idx == 0 {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let di = dict_idx - 1;
                    if di >= dict_size {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let dict_off_start = 16 + row_count * 4;
                    let mut se_buf = [0u8; 8];
                    self.read_cached_bytes(
                        data_base + (dict_off_start + di * 4) as u64,
                        &mut se_buf,
                    )?;
                    let s = u32::from_le_bytes(se_buf[0..4].try_into().unwrap()) as usize;
                    let e = u32::from_le_bytes(se_buf[4..8].try_into().unwrap()) as usize;
                    if s > e {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let dd_start = dict_off_start + dict_size * 4 + 8;
                    let mut str_buf = vec![0u8; e - s];
                    self.read_cached_bytes(data_base + (dd_start + s) as u64, &mut str_buf)?;
                    Value::String(std::str::from_utf8(&str_buf).unwrap_or("").to_string())
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
                ) => {
                    // BitPack format: [count:u64][bit_width:u8][min_value:i64][packed_bits...]
                    let mut hdr = [0u8; 17];
                    self.read_cached_bytes(data_base, &mut hdr)?;
                    let count = u64::from_le_bytes(hdr[0..8].try_into().unwrap()) as usize;
                    let bit_width = hdr[8] as usize;
                    let min_val = i64::from_le_bytes(hdr[9..17].try_into().unwrap());
                    if local_idx >= count || bit_width == 0 {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let bit_pos = local_idx * bit_width;
                    let byte_off = bit_pos / 8;
                    let bit_shift = bit_pos % 8;
                    let bytes_needed = ((bit_shift + bit_width + 7) / 8).min(3);
                    let mut bits = [0u8; 3];
                    self.read_cached_bytes(
                        data_base + 17 + byte_off as u64,
                        &mut bits[..bytes_needed],
                    )?;
                    let raw = (bits[0] as u64) | ((bits[1] as u64) << 8) | ((bits[2] as u64) << 16);
                    let mask = if bit_width >= 64 {
                        u64::MAX
                    } else {
                        (1u64 << bit_width) - 1
                    };
                    Value::Int64(min_val + ((raw >> bit_shift) & mask) as i64)
                }
                (
                    COL_ENCODING_RLE,
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
                ) => {
                    // RLE format: [count:u64][num_runs:u64][(value:i64, run_len:u32) × num_runs]
                    let mut hdr = [0u8; 16];
                    self.read_cached_bytes(data_base, &mut hdr)?;
                    let count = u64::from_le_bytes(hdr[0..8].try_into().unwrap()) as usize;
                    let num_runs = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
                    if local_idx >= count {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let mut cumulative = 0usize;
                    let mut found = Value::Null;
                    for run_i in 0..num_runs {
                        let mut rb = [0u8; 12];
                        self.read_cached_bytes(data_base + 16 + (run_i * 12) as u64, &mut rb)?;
                        let run_val = i64::from_le_bytes(rb[0..8].try_into().unwrap());
                        let run_len = u32::from_le_bytes(rb[8..12].try_into().unwrap()) as usize;
                        cumulative += run_len;
                        if local_idx < cumulative {
                            found = Value::Int64(run_val);
                            break;
                        }
                    }
                    found
                }
                (COL_ENCODING_RLE_BOOL, ColumnType::Bool) => {
                    // Bool RLE: [count:u64][num_runs:u64][(value:u8, run_len:u32) × num_runs]
                    let mut hdr = [0u8; 16];
                    self.read_cached_bytes(data_base, &mut hdr)?;
                    let count = u64::from_le_bytes(hdr[0..8].try_into().unwrap()) as usize;
                    let num_runs = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
                    if local_idx >= count {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let mut cumulative = 0usize;
                    let mut found = Value::Null;
                    for run_i in 0..num_runs {
                        let mut rb = [0u8; 5];
                        self.read_cached_bytes(data_base + 16 + (run_i * 5) as u64, &mut rb)?;
                        let run_val = rb[0] != 0;
                        let run_len = u32::from_le_bytes(rb[1..5].try_into().unwrap()) as usize;
                        cumulative += run_len;
                        if local_idx < cumulative {
                            found = Value::Bool(run_val);
                            break;
                        }
                    }
                    found
                }
                _ => {
                    // Unknown encoding: fall back to read_row_by_id_values
                    return Ok(None);
                }
            };
            result.push((col_name, val));
        }
        Ok(Some(result))
    }

    /// Projected RCIX point lookup: reads only the requested columns.
    /// This keeps `WHERE col = 'x' LIMIT 1` fast even when the query projects a subset of fields.
    pub(crate) fn retrieve_rcix_projected(
        &self,
        id: u64,
        column_names: &[&str],
    ) -> io::Result<Option<Vec<(String, crate::data::Value)>>> {
        use crate::data::Value;

        let include_id = column_names.contains(&"_id");
        let requested_cols: Vec<&str> = column_names
            .iter()
            .copied()
            .filter(|name| *name != "_id")
            .collect();

        let (rg_offset, rg_rows, rg_min_id, selected_cols) = {
            let fg = self.v4_footer.read();
            let footer = match fg.as_ref() {
                Some(f) => f,
                None => return Ok(None),
            };
            let mut rg_i_found = None;
            for (i, rg) in footer.row_groups.iter().enumerate() {
                if rg.min_id <= id && id <= rg.max_id && rg.row_count > 0 {
                    rg_i_found = Some(i);
                    break;
                }
            }
            let rg_i = match rg_i_found {
                Some(i) => i,
                None => return Ok(None),
            };
            let rg_meta = &footer.row_groups[rg_i];
            let rg_col_offsets = match footer.col_offsets.get(rg_i) {
                Some(offsets) => offsets,
                None => return Ok(None),
            };
            let mut selected_cols = Vec::with_capacity(requested_cols.len());
            for &col_name in &requested_cols {
                let col_idx = match footer.schema.get_index(col_name) {
                    Some(idx) => idx,
                    None => continue,
                };
                if col_idx >= rg_col_offsets.len() {
                    return Ok(None);
                }
                selected_cols.push((
                    col_name.to_string(),
                    footer.schema.columns[col_idx].1,
                    rg_col_offsets[col_idx] as usize,
                ));
            }
            (
                rg_meta.offset,
                rg_meta.row_count as usize,
                rg_meta.min_id,
                selected_cols,
            )
        };

        let mut rg_hdr = [0u8; 32];
        self.read_cached_bytes(rg_offset, &mut rg_hdr)?;
        if rg_hdr[28] != RG_COMPRESS_NONE || rg_hdr[29] < 1 {
            return Ok(None);
        }

        let body_base = rg_offset + 32;
        let null_bitmap_len = (rg_rows + 7) / 8;

        let guess = (id.saturating_sub(rg_min_id)) as usize;
        let local_idx = if guess < rg_rows {
            let mut id_buf = [0u8; 8];
            self.read_cached_bytes(body_base + (guess * 8) as u64, &mut id_buf)?;
            if u64::from_le_bytes(id_buf) == id {
                guess
            } else {
                let mut ids_buf = vec![0u8; rg_rows * 8];
                self.read_cached_bytes(body_base, &mut ids_buf)?;
                let ids_cow = bytes_as_u64_slice(&ids_buf, rg_rows);
                match ids_cow.binary_search(&id) {
                    Ok(i) => i,
                    Err(_) => return Ok(None),
                }
            }
        } else {
            return Ok(None);
        };

        let mut del_buf = [0u8; 1];
        self.read_cached_bytes(
            body_base + (rg_rows * 8 + local_idx / 8) as u64,
            &mut del_buf,
        )?;
        if (del_buf[0] >> (local_idx % 8)) & 1 == 1 {
            return Ok(None);
        }

        let mut result = Vec::with_capacity(selected_cols.len() + usize::from(include_id));
        if include_id {
            result.push(("_id".to_string(), Value::Int64(id as i64)));
        }

        for (col_name, col_type, col_start) in selected_cols {
            let mut null_buf = [0u8; 1];
            self.read_cached_bytes(
                body_base + (col_start + local_idx / 8) as u64,
                &mut null_buf,
            )?;
            if (null_buf[0] >> (local_idx % 8)) & 1 == 1 {
                result.push((col_name, Value::Null));
                continue;
            }

            let enc_off = body_base + (col_start + null_bitmap_len) as u64;
            let mut enc_buf = [0u8; 1];
            self.read_cached_bytes(enc_off, &mut enc_buf)?;
            let encoding = enc_buf[0];
            let data_base = body_base + (col_start + null_bitmap_len + 1) as u64;

            let val = match (encoding, col_type) {
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
                ) => {
                    let mut v = [0u8; 8];
                    self.read_cached_bytes(data_base + (8 + local_idx * 8) as u64, &mut v)?;
                    Value::Int64(i64::from_le_bytes(v))
                }
                (COL_ENCODING_PLAIN, ColumnType::Float64 | ColumnType::Float32) => {
                    let mut v = [0u8; 8];
                    self.read_cached_bytes(data_base + (8 + local_idx * 8) as u64, &mut v)?;
                    Value::Float64(f64::from_le_bytes(v))
                }
                (COL_ENCODING_PLAIN, ColumnType::Bool) => {
                    let mut v = [0u8; 1];
                    self.read_cached_bytes(data_base + (8 + local_idx / 8) as u64, &mut v)?;
                    Value::Bool((v[0] >> (local_idx % 8)) & 1 == 1)
                }
                (COL_ENCODING_PLAIN, ColumnType::String) => {
                    let mut count_buf = [0u8; 8];
                    self.read_cached_bytes(data_base, &mut count_buf)?;
                    let count = u64::from_le_bytes(count_buf) as usize;
                    if local_idx >= count {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let mut se_buf = [0u8; 8];
                    self.read_cached_bytes(data_base + (8 + local_idx * 4) as u64, &mut se_buf)?;
                    let s = u32::from_le_bytes(se_buf[0..4].try_into().unwrap()) as usize;
                    let e = u32::from_le_bytes(se_buf[4..8].try_into().unwrap()) as usize;
                    if s > e {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let dd_start = 8 + (count + 1) * 4 + 8;
                    let mut str_buf = vec![0u8; e - s];
                    self.read_cached_bytes(data_base + (dd_start + s) as u64, &mut str_buf)?;
                    Value::String(std::str::from_utf8(&str_buf).unwrap_or("").to_string())
                }
                (COL_ENCODING_PLAIN, ColumnType::Binary) => {
                    let mut count_buf = [0u8; 8];
                    self.read_cached_bytes(data_base, &mut count_buf)?;
                    let count = u64::from_le_bytes(count_buf) as usize;
                    if local_idx >= count {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let mut se_buf = [0u8; 8];
                    self.read_cached_bytes(data_base + (8 + local_idx * 4) as u64, &mut se_buf)?;
                    let s = u32::from_le_bytes(se_buf[0..4].try_into().unwrap()) as usize;
                    let e = u32::from_le_bytes(se_buf[4..8].try_into().unwrap()) as usize;
                    if s > e {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let dd_start = 8 + (count + 1) * 4 + 8;
                    let mut bin_buf = vec![0u8; e - s];
                    self.read_cached_bytes(data_base + (dd_start + s) as u64, &mut bin_buf)?;
                    Value::Binary(bin_buf)
                }
                (COL_ENCODING_PLAIN, ColumnType::StringDict) => {
                    let mut hdr = [0u8; 16];
                    self.read_cached_bytes(data_base, &mut hdr)?;
                    let row_count = u64::from_le_bytes(hdr[0..8].try_into().unwrap()) as usize;
                    let dict_size = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
                    if local_idx >= row_count {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let mut idx_buf = [0u8; 4];
                    self.read_cached_bytes(data_base + (16 + local_idx * 4) as u64, &mut idx_buf)?;
                    let dict_idx = u32::from_le_bytes(idx_buf) as usize;
                    if dict_idx == 0 {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let di = dict_idx - 1;
                    if di >= dict_size {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let dict_off_start = 16 + row_count * 4;
                    let mut se_buf = [0u8; 8];
                    self.read_cached_bytes(
                        data_base + (dict_off_start + di * 4) as u64,
                        &mut se_buf,
                    )?;
                    let s = u32::from_le_bytes(se_buf[0..4].try_into().unwrap()) as usize;
                    let e = u32::from_le_bytes(se_buf[4..8].try_into().unwrap()) as usize;
                    if s > e {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let dd_start = dict_off_start + dict_size * 4 + 8;
                    let mut str_buf = vec![0u8; e - s];
                    self.read_cached_bytes(data_base + (dd_start + s) as u64, &mut str_buf)?;
                    Value::String(std::str::from_utf8(&str_buf).unwrap_or("").to_string())
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
                ) => {
                    let mut hdr = [0u8; 17];
                    self.read_cached_bytes(data_base, &mut hdr)?;
                    let count = u64::from_le_bytes(hdr[0..8].try_into().unwrap()) as usize;
                    let bit_width = hdr[8] as usize;
                    let min_val = i64::from_le_bytes(hdr[9..17].try_into().unwrap());
                    if local_idx >= count || bit_width == 0 {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let bit_pos = local_idx * bit_width;
                    let byte_off = bit_pos / 8;
                    let bit_shift = bit_pos % 8;
                    let bytes_needed = ((bit_shift + bit_width + 7) / 8).min(3);
                    let mut bits = [0u8; 3];
                    self.read_cached_bytes(
                        data_base + 17 + byte_off as u64,
                        &mut bits[..bytes_needed],
                    )?;
                    let raw = (bits[0] as u64) | ((bits[1] as u64) << 8) | ((bits[2] as u64) << 16);
                    let mask = if bit_width >= 64 {
                        u64::MAX
                    } else {
                        (1u64 << bit_width) - 1
                    };
                    Value::Int64(min_val + ((raw >> bit_shift) & mask) as i64)
                }
                (
                    COL_ENCODING_RLE,
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
                ) => {
                    let mut hdr = [0u8; 16];
                    self.read_cached_bytes(data_base, &mut hdr)?;
                    let count = u64::from_le_bytes(hdr[0..8].try_into().unwrap()) as usize;
                    let num_runs = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
                    if local_idx >= count {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let mut cumulative = 0usize;
                    let mut found = Value::Null;
                    for run_i in 0..num_runs {
                        let mut rb = [0u8; 12];
                        self.read_cached_bytes(data_base + 16 + (run_i * 12) as u64, &mut rb)?;
                        let run_val = i64::from_le_bytes(rb[0..8].try_into().unwrap());
                        let run_len = u32::from_le_bytes(rb[8..12].try_into().unwrap()) as usize;
                        cumulative += run_len;
                        if local_idx < cumulative {
                            found = Value::Int64(run_val);
                            break;
                        }
                    }
                    found
                }
                (COL_ENCODING_RLE_BOOL, ColumnType::Bool) => {
                    let mut hdr = [0u8; 16];
                    self.read_cached_bytes(data_base, &mut hdr)?;
                    let count = u64::from_le_bytes(hdr[0..8].try_into().unwrap()) as usize;
                    let num_runs = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
                    if local_idx >= count {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let mut cumulative = 0usize;
                    let mut found = Value::Null;
                    for run_i in 0..num_runs {
                        let mut rb = [0u8; 5];
                        self.read_cached_bytes(data_base + 16 + (run_i * 5) as u64, &mut rb)?;
                        let run_val = rb[0] != 0;
                        let run_len = u32::from_le_bytes(rb[1..5].try_into().unwrap()) as usize;
                        cumulative += run_len;
                        if local_idx < cumulative {
                            found = Value::Bool(run_val);
                            break;
                        }
                    }
                    found
                }
                _ => return Ok(None),
            };
            result.push((col_name, val));
        }

        Ok(Some(result))
    }

    /// RCIX pread batch read: builds Arrow RecordBatch for first `rows_to_take` rows
    /// using page cache instead of mmap. Only reads the minimal bytes per column:
    ///   - PLAIN Int64/Float64: 8 + N*8 bytes  (not 8 + rg_rows*8)
    ///   - BITPACK:             17 + ceil(N*bw/8) bytes
    ///   - PLAIN String:        offsets(N+1*4) + string data for first N strings
    ///   - StringDict:          indices(N*4) + full dict (small for low-cardinality)
    pub(crate) fn to_arrow_batch_pread_rcix(
        &self,
        col_indices: &[usize],
        include_id: bool,
        rows_to_take: usize,
    ) -> io::Result<Option<arrow::record_batch::RecordBatch>> {
        use arrow::array::{BooleanArray, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        let footer = match self.v4_footer.read().as_ref() {
            Some(f) => f.clone(),
            None => return Ok(None),
        };
        // Only safe for single-RG or multi-RG where limit fits within first RG.
        // (multi-RG full-scan via to_arrow_batch_with_limit must use mmap path)
        let rg_meta = match footer.row_groups.first() {
            Some(r) if r.row_count > 0 => r,
            _ => return Ok(None),
        };
        if rg_meta.deletion_count > 0 {
            return Ok(None);
        }
        if footer.col_offsets.is_empty() || footer.col_offsets[0].is_empty() {
            return Ok(None);
        }
        let rg_rows = rg_meta.row_count as usize;
        // For multi-RG: only use pread if limit fits within first RG; else mmap handles full scan
        if footer.row_groups.len() > 1 && rows_to_take > rg_rows {
            return Ok(None);
        }
        let n = rows_to_take.min(rg_rows);
        if n == 0 {
            return Ok(None);
        }
        let body_base = rg_meta.offset + 32;
        let null_bitmap_len = (rg_rows + 7) / 8;
        let col_offsets = &footer.col_offsets[0];
        let schema = &footer.schema;

        // Check RG header compression flag
        let mut rg_hdr = [0u8; 32];
        self.read_cached_bytes(rg_meta.offset, &mut rg_hdr)?;
        if rg_hdr[28] != RG_COMPRESS_NONE || rg_hdr[29] < 1 {
            return Ok(None);
        }

        let mut fields: Vec<arrow::datatypes::Field> = Vec::new();
        let mut arrays: Vec<arrow::array::ArrayRef> = Vec::new();

        if include_id {
            let mut id_buf = vec![0u8; n * 8];
            self.read_cached_bytes(body_base, &mut id_buf)?;
            let ids: Vec<i64> = (0..n)
                .map(|i| i64::from_le_bytes(id_buf[i * 8..i * 8 + 8].try_into().unwrap()))
                .collect();
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(ids)));
        }

        for &col_idx in col_indices {
            // Schema evolution: column added after this RG — fall back to mmap path
            if col_idx >= col_offsets.len() {
                return Ok(None);
            }
            let (col_name, col_type) = &schema.columns[col_idx];
            let col_abs = body_base + col_offsets[col_idx] as u64;

            // null bitmap: only read bytes needed for first n rows (not full null_bitmap_len)
            let needed_null_bytes = (n + 7) / 8;
            let mut null_bytes = vec![0u8; needed_null_bytes];
            self.read_cached_bytes(col_abs, &mut null_bytes)?;
            let null_flags: Vec<bool> = (0..n)
                .map(|i| (null_bytes[i / 8] >> (i % 8)) & 1 == 1)
                .collect();
            let has_nulls = null_flags.iter().any(|&b| b);

            // Read encoding byte + data payload in ONE call starting at data_start.
            // enc[0] = encoding, rest = payload. Saves one read_cached_bytes call per column.
            let data_start = col_abs + null_bitmap_len as u64;

            let arrow_dt;
            let arr: arrow::array::ArrayRef = {
                // Peek at encoding byte with a small read covering enc + header
                let mut enc_buf = [0u8; 1];
                self.read_cached_bytes(data_start, &mut enc_buf)?;
                let encoding = enc_buf[0];
                let payload = data_start + 1;
                match (encoding, col_type) {
                    (
                        0,
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
                    ) => {
                        arrow_dt = ArrowDataType::Int64;
                        let mut buf = vec![0u8; 8 + n * 8];
                        self.read_cached_bytes(payload, &mut buf)?;
                        let vals: Vec<Option<i64>> = (0..n)
                            .map(|i| {
                                if null_flags[i] {
                                    None
                                } else {
                                    Some(i64::from_le_bytes(
                                        buf[8 + i * 8..8 + i * 8 + 8].try_into().unwrap(),
                                    ))
                                }
                            })
                            .collect();
                        Arc::new(Int64Array::from(vals))
                    }
                    (0, ColumnType::Float64 | ColumnType::Float32) => {
                        arrow_dt = ArrowDataType::Float64;
                        let mut buf = vec![0u8; 8 + n * 8];
                        self.read_cached_bytes(payload, &mut buf)?;
                        let vals: Vec<Option<f64>> = (0..n)
                            .map(|i| {
                                if null_flags[i] {
                                    None
                                } else {
                                    Some(f64::from_le_bytes(
                                        buf[8 + i * 8..8 + i * 8 + 8].try_into().unwrap(),
                                    ))
                                }
                            })
                            .collect();
                        Arc::new(Float64Array::from(vals))
                    }
                    (0, ColumnType::Bool) => {
                        arrow_dt = ArrowDataType::Boolean;
                        let byte_count = (n + 7) / 8;
                        let mut buf = vec![0u8; 8 + byte_count]; // skip count header
                        self.read_cached_bytes(payload, &mut buf)?;
                        let bools: Vec<Option<bool>> = (0..n)
                            .map(|i| {
                                if null_flags[i] {
                                    None
                                } else {
                                    Some((buf[8 + i / 8] >> (i % 8)) & 1 == 1)
                                }
                            })
                            .collect();
                        Arc::new(BooleanArray::from(bools))
                    }
                    (0, ColumnType::String) => {
                        arrow_dt = ArrowDataType::Utf8;
                        // Read count + first (n+1) offsets
                        let mut hdr = vec![0u8; 8 + (n + 1) * 4];
                        self.read_cached_bytes(payload, &mut hdr)?;
                        let all_off_end_u32 =
                            u32::from_le_bytes(hdr[8 + n * 4..8 + n * 4 + 4].try_into().unwrap())
                                as usize;
                        // data section starts at payload + 8 + (rg_rows+1)*4 + 8
                        // but we can compute it from next col or we read from col structure:
                        // offset 0..rg_rows+1 offsets, then data_len (8), then data
                        // For first n rows, string data is at [0..all_off_end_u32] in the data buffer
                        // data_buf_abs = payload + 8 + (rg_rows+1)*4 + 8
                        // We read just all_off_end_u32 bytes of string data
                        let data_buf_abs = payload + 8 + (rg_rows + 1) as u64 * 4 + 8;
                        let mut str_data = vec![0u8; all_off_end_u32];
                        if all_off_end_u32 > 0 {
                            self.read_cached_bytes(data_buf_abs, &mut str_data)?;
                        }
                        let vals: Vec<Option<&str>> = (0..n)
                            .map(|i| {
                                if null_flags[i] {
                                    return None;
                                }
                                let s = u32::from_le_bytes(
                                    hdr[8 + i * 4..8 + i * 4 + 4].try_into().unwrap(),
                                ) as usize;
                                let e = u32::from_le_bytes(
                                    hdr[8 + i * 4 + 4..8 + i * 4 + 8].try_into().unwrap(),
                                ) as usize;
                                if s <= e && e <= str_data.len() {
                                    std::str::from_utf8(&str_data[s..e]).ok()
                                } else {
                                    Some("")
                                }
                            })
                            .collect();
                        Arc::new(StringArray::from(vals))
                    }
                    (0, ColumnType::StringDict) => {
                        arrow_dt = ArrowDataType::Utf8;
                        let mut hdr = [0u8; 16];
                        self.read_cached_bytes(payload, &mut hdr)?;
                        let row_count = u64::from_le_bytes(hdr[0..8].try_into().unwrap()) as usize;
                        let dict_size = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
                        let mut idx_buf = vec![0u8; n * 4];
                        self.read_cached_bytes(payload + 16, &mut idx_buf)?;
                        // dict section: after all row_count indices
                        let dict_abs = payload + 16 + row_count as u64 * 4;
                        let dict_off_bytes = dict_size * 4;
                        let mut dict_hdr = vec![0u8; dict_off_bytes + 8];
                        if dict_size > 0 {
                            self.read_cached_bytes(dict_abs, &mut dict_hdr)?;
                        }
                        let dict_data_len = if dict_hdr.len() >= dict_off_bytes + 8 {
                            u64::from_le_bytes(
                                dict_hdr[dict_off_bytes..dict_off_bytes + 8]
                                    .try_into()
                                    .unwrap(),
                            ) as usize
                        } else {
                            0
                        };
                        let mut dict_data = vec![0u8; dict_data_len];
                        if dict_data_len > 0 {
                            self.read_cached_bytes(
                                dict_abs + dict_off_bytes as u64 + 8,
                                &mut dict_data,
                            )?;
                        }
                        let vals: Vec<Option<&str>> = (0..n)
                            .map(|i| {
                                if null_flags[i] {
                                    return None;
                                }
                                let di_raw = u32::from_le_bytes(
                                    idx_buf[i * 4..i * 4 + 4].try_into().unwrap(),
                                ) as usize;
                                if di_raw == 0 || di_raw > dict_size {
                                    return Some("");
                                }
                                let di = di_raw - 1;
                                let off_off = di * 4;
                                if off_off + 8 > dict_hdr.len() {
                                    return Some("");
                                }
                                let s = u32::from_le_bytes(
                                    dict_hdr[off_off..off_off + 4].try_into().unwrap(),
                                ) as usize;
                                let e = u32::from_le_bytes(
                                    dict_hdr[off_off + 4..off_off + 8].try_into().unwrap(),
                                ) as usize;
                                if s <= e && e <= dict_data.len() {
                                    std::str::from_utf8(&dict_data[s..e]).ok()
                                } else {
                                    Some("")
                                }
                            })
                            .collect();
                        Arc::new(StringArray::from(vals))
                    }
                    (
                        2, /* BITPACK */
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
                    ) => {
                        arrow_dt = ArrowDataType::Int64;
                        let mut bp_hdr = [0u8; 17];
                        self.read_cached_bytes(payload, &mut bp_hdr)?;
                        let bit_width = bp_hdr[8] as usize;
                        let min_val = i64::from_le_bytes(bp_hdr[9..17].try_into().unwrap());
                        let packed_bytes = if bit_width > 0 {
                            (n * bit_width + 7) / 8
                        } else {
                            0
                        };
                        let mut packed = vec![0u8; packed_bytes];
                        if packed_bytes > 0 {
                            self.read_cached_bytes(payload + 17, &mut packed)?;
                        }
                        let vals: Vec<Option<i64>> = (0..n)
                            .map(|i| {
                                if null_flags[i] || bit_width == 0 {
                                    return None;
                                }
                                let bit_pos = i * bit_width;
                                let byte_off = bit_pos / 8;
                                let bit_shift = bit_pos % 8;
                                let bytes_need = ((bit_shift + bit_width + 7) / 8).min(3);
                                let mut b = [0u8; 3];
                                for k in 0..bytes_need {
                                    if byte_off + k < packed.len() {
                                        b[k] = packed[byte_off + k];
                                    }
                                }
                                let raw =
                                    (b[0] as u64) | ((b[1] as u64) << 8) | ((b[2] as u64) << 16);
                                let mask = if bit_width >= 64 {
                                    u64::MAX
                                } else {
                                    (1u64 << bit_width) - 1
                                };
                                Some(min_val + ((raw >> bit_shift) & mask) as i64)
                            })
                            .collect();
                        Arc::new(Int64Array::from(vals))
                    }
                    (
                        1, /* RLE */
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
                    ) => {
                        arrow_dt = ArrowDataType::Int64;
                        let mut rle_hdr = [0u8; 16];
                        self.read_cached_bytes(payload, &mut rle_hdr)?;
                        let num_runs =
                            u64::from_le_bytes(rle_hdr[8..16].try_into().unwrap()) as usize;
                        let mut run_buf = vec![0u8; num_runs * 12];
                        if num_runs > 0 {
                            self.read_cached_bytes(payload + 16, &mut run_buf)?;
                        }
                        // decode first n values
                        let mut vals: Vec<Option<i64>> = Vec::with_capacity(n);
                        let mut cum = 0usize;
                        let mut run_i = 0usize;
                        for row_i in 0..n {
                            if null_flags[row_i] {
                                vals.push(None);
                                continue;
                            }
                            while run_i < num_runs
                                && cum
                                    + u32::from_le_bytes(
                                        run_buf[run_i * 12 + 8..run_i * 12 + 12]
                                            .try_into()
                                            .unwrap(),
                                    ) as usize
                                    <= row_i
                            {
                                cum += u32::from_le_bytes(
                                    run_buf[run_i * 12 + 8..run_i * 12 + 12].try_into().unwrap(),
                                ) as usize;
                                run_i += 1;
                            }
                            let v = if run_i < num_runs {
                                i64::from_le_bytes(
                                    run_buf[run_i * 12..run_i * 12 + 8].try_into().unwrap(),
                                )
                            } else {
                                0
                            };
                            vals.push(Some(v));
                        }
                        Arc::new(Int64Array::from(vals))
                    }
                    _ => return Ok(None), // unknown encoding — caller falls to mmap path
                }
            };
            fields.push(Field::new(col_name.as_str(), arrow_dt, has_nulls));
            arrays.push(arr);
        }

        let arrow_schema = Arc::new(Schema::new(fields));
        match arrow::record_batch::RecordBatch::try_new(arrow_schema, arrays) {
            Ok(batch) => Ok(Some(batch)),
            Err(_) => Ok(None),
        }
    }

    /// Ultra-fast point lookup: returns Vec<(col_name, Value)> directly from V4 columns
    /// Bypasses Arrow conversion and HashMap overhead
    fn read_in_memory_row_by_id_values(
        &self,
        id: u64,
    ) -> io::Result<Option<Vec<(String, crate::data::Value)>>> {
        use crate::data::Value;

        let ids_guard = self.ids.read();
        let row_idx = match ids_guard.binary_search(&id) {
            Ok(i) => i,
            Err(_) => return Ok(None),
        };
        if self.is_deleted(row_idx) {
            return Ok(None);
        }
        drop(ids_guard);

        let schema = self.schema.read();
        let columns = self.columns.read();
        let nulls = self.nulls.read();

        let mut result = Vec::with_capacity(schema.column_count() + 1);
        result.push(("_id".to_string(), Value::Int64(id as i64)));

        for (col_idx, (col_name, _)) in schema.columns.iter().enumerate() {
            // Check null
            if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                let b = row_idx / 8;
                let bit = row_idx % 8;
                if b < nulls[col_idx].len() && (nulls[col_idx][b] >> bit) & 1 == 1 {
                    result.push((col_name.clone(), Value::Null));
                    continue;
                }
            }

            if col_idx >= columns.len() {
                result.push((col_name.clone(), Value::Null));
                continue;
            }

            let val = match &columns[col_idx] {
                ColumnData::Int64(v) => {
                    if row_idx < v.len() {
                        Value::Int64(v[row_idx])
                    } else {
                        Value::Null
                    }
                }
                ColumnData::Float64(v) => {
                    if row_idx < v.len() {
                        Value::Float64(v[row_idx])
                    } else {
                        Value::Null
                    }
                }
                ColumnData::String { offsets, data } => {
                    let count = offsets.len().saturating_sub(1);
                    if row_idx < count {
                        let s = offsets[row_idx] as usize;
                        let e = offsets[row_idx + 1] as usize;
                        Value::String(std::str::from_utf8(&data[s..e]).unwrap_or("").to_string())
                    } else {
                        Value::Null
                    }
                }
                ColumnData::Bool { data, len } => {
                    if row_idx < *len {
                        let b = row_idx / 8;
                        let bit = row_idx % 8;
                        if b < data.len() {
                            Value::Bool((data[b] >> bit) & 1 == 1)
                        } else {
                            Value::Null
                        }
                    } else {
                        Value::Null
                    }
                }
                ColumnData::Binary { offsets, data } => {
                    let count = offsets.len().saturating_sub(1);
                    if row_idx < count {
                        let s = offsets[row_idx] as usize;
                        let e = offsets[row_idx + 1] as usize;
                        Value::Binary(data[s..e].to_vec())
                    } else {
                        Value::Null
                    }
                }
                ColumnData::StringDict {
                    indices,
                    dict_offsets,
                    dict_data,
                } => {
                    if row_idx < indices.len() {
                        let idx = indices[row_idx];
                        if idx == 0 {
                            Value::Null
                        } else {
                            let di = (idx - 1) as usize;
                            if di + 1 < dict_offsets.len() {
                                let s = dict_offsets[di] as usize;
                                let e = dict_offsets[di + 1] as usize;
                                Value::String(
                                    std::str::from_utf8(&dict_data[s..e])
                                        .unwrap_or("")
                                        .to_string(),
                                )
                            } else {
                                Value::Null
                            }
                        }
                    } else {
                        Value::Null
                    }
                }
                _ => Value::Null,
            };
            result.push((col_name.clone(), val));
        }

        Ok(Some(result))
    }

    pub fn read_row_by_id_values(
        &self,
        id: u64,
    ) -> io::Result<Option<Vec<(String, crate::data::Value)>>> {
        use crate::data::Value;

        let is_v4 = self.is_v4_format();
        if !is_v4 {
            return Ok(None);
        }
        if let Some(row) = self.read_in_memory_row_by_id_values(id)? {
            return Ok(Some(row));
        }
        {
            // RCIX MMAP PATH: always try first, even if data is also in memory.
            // Avoids building the 1M-entry id_to_idx HashMap (~0.76ms).
            // DIRECT MMAP PATH: Find RG via min_id/max_id (no decompression needed).
            let footer = match self.get_or_load_footer()? {
                Some(f) => f,
                None => return Ok(None),
            };
            let file_guard = self.file.read();
            let file = file_guard
                .as_ref()
                .ok_or_else(|| err_not_conn("File not open for point lookup"))?;
            let mut mmap_guard = self.mmap_cache.write();
            let mmap_ref = mmap_guard.get_or_create(file)?;

            // Step 1: Find RG via min_id/max_id — O(N_rgs), zero decompression.
            let mut found_rg_i: Option<usize> = None;
            for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
                if rg_meta.min_id <= id && id <= rg_meta.max_id && rg_meta.row_count > 0 {
                    found_rg_i = Some(rg_i);
                    break;
                }
            }
            let rg_i = match found_rg_i {
                Some(i) => i,
                None => return Ok(None),
            };
            let rg_meta = &footer.row_groups[rg_i];
            let rg_rows = rg_meta.row_count as usize;
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

            // Step 2: RCIX fast path — uncompressed, enc_ver ≥ 1, col_offsets present.
            let col_count = footer.schema.column_count();
            let has_rcix = compress_flag == RG_COMPRESS_NONE
                && encoding_version >= 1
                && rg_i < footer.col_offsets.len()
                && footer.col_offsets[rg_i].len() >= col_count;

            if has_rcix {
                let body = &rg_bytes[32..];

                // Step 3: O(1) local_idx — try direct id-based index, verify, else binary search.
                let guess = (id.saturating_sub(rg_meta.min_id)) as usize;
                let local_idx = if guess < rg_rows && guess * 8 + 8 <= body.len() {
                    let id_at =
                        u64::from_le_bytes(body[guess * 8..guess * 8 + 8].try_into().unwrap());
                    if id_at == id {
                        guess
                    } else {
                        let ids_cow = bytes_as_u64_slice(body, rg_rows);
                        match ids_cow.binary_search(&id) {
                            Ok(i) => i,
                            Err(_) => return Ok(None),
                        }
                    }
                } else {
                    let ids_cow = bytes_as_u64_slice(body, rg_rows);
                    match ids_cow.binary_search(&id) {
                        Ok(i) => i,
                        Err(_) => return Ok(None),
                    }
                };

                // Step 4: Check deletion bit.
                let del_start = rg_rows * 8;
                let del_vec_len = (rg_rows + 7) / 8;
                if del_start + local_idx / 8 >= body.len() {
                    return Ok(None);
                }
                if (body[del_start + local_idx / 8] >> (local_idx % 8)) & 1 == 1 {
                    return Ok(None);
                }

                let null_bitmap_len = (rg_rows + 7) / 8;
                let col_offsets = &footer.col_offsets[rg_i];
                let schema = &footer.schema;
                let mut result = Vec::with_capacity(col_count + 1);
                result.push(("_id".to_string(), Value::Int64(id as i64)));

                // Step 5: RCIX jump to each column — O(1) per column, no sequential scan.
                for col_idx in 0..col_count {
                    let col_name = schema.columns[col_idx].0.clone();
                    let col_type = schema.columns[col_idx].1;
                    let col_start = col_offsets[col_idx] as usize;

                    if col_start + null_bitmap_len > body.len() {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let is_null = (body[col_start + local_idx / 8] >> (local_idx % 8)) & 1 == 1;
                    let data_start = col_start + null_bitmap_len;
                    if is_null || data_start >= body.len() {
                        result.push((col_name, Value::Null));
                        continue;
                    }
                    let col_bytes = &body[data_start..];
                    let encoding = col_bytes[0]; // enc_ver >= 1
                    let data_bytes = &col_bytes[1..];

                    let val = match (encoding, col_type) {
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
                        ) => {
                            // Plain Int64: [count:u64][data: count*8]
                            let off = 8 + local_idx * 8;
                            if off + 8 <= data_bytes.len() {
                                Value::Int64(i64::from_le_bytes(
                                    data_bytes[off..off + 8].try_into().unwrap(),
                                ))
                            } else {
                                Value::Null
                            }
                        }
                        (COL_ENCODING_PLAIN, ColumnType::Float64 | ColumnType::Float32) => {
                            let off = 8 + local_idx * 8;
                            if off + 8 <= data_bytes.len() {
                                Value::Float64(f64::from_le_bytes(
                                    data_bytes[off..off + 8].try_into().unwrap(),
                                ))
                            } else {
                                Value::Null
                            }
                        }
                        (COL_ENCODING_PLAIN, ColumnType::Bool) => {
                            // Plain Bool: [len:u64][packed bits]
                            let byte_off = 8 + local_idx / 8;
                            let bit = local_idx % 8;
                            if byte_off < data_bytes.len() {
                                Value::Bool((data_bytes[byte_off] >> bit) & 1 == 1)
                            } else {
                                Value::Null
                            }
                        }
                        (COL_ENCODING_PLAIN, ColumnType::String) => {
                            // Plain String: [count:u64][offsets:(count+1)*4][data_len:u64][data]
                            if data_bytes.len() >= 8 {
                                let count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap())
                                    as usize;
                                let off_start = 8 + local_idx * 4;
                                let off_end = 8 + (local_idx + 1) * 4;
                                if off_end + 4 <= data_bytes.len() && local_idx < count {
                                    let s = u32::from_le_bytes(
                                        data_bytes[off_start..off_start + 4].try_into().unwrap(),
                                    ) as usize;
                                    let e = u32::from_le_bytes(
                                        data_bytes[off_end..off_end + 4].try_into().unwrap(),
                                    ) as usize;
                                    let offsets_end = 8 + (count + 1) * 4;
                                    let data_len_off = offsets_end;
                                    if data_len_off + 8 <= data_bytes.len() {
                                        let data_start = data_len_off + 8;
                                        if data_start + e <= data_bytes.len() {
                                            Value::String(
                                                std::str::from_utf8(
                                                    &data_bytes[data_start + s..data_start + e],
                                                )
                                                .unwrap_or("")
                                                .to_string(),
                                            )
                                        } else {
                                            Value::Null
                                        }
                                    } else {
                                        Value::Null
                                    }
                                } else {
                                    Value::Null
                                }
                            } else {
                                Value::Null
                            }
                        }
                        (COL_ENCODING_PLAIN, ColumnType::Binary) => {
                            // Plain Binary: same layout as String: [count:u64][offsets:(count+1)*4][data_len:u64][data]
                            if data_bytes.len() >= 8 {
                                let count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap())
                                    as usize;
                                let off_start = 8 + local_idx * 4;
                                let off_end = 8 + (local_idx + 1) * 4;
                                if off_end + 4 <= data_bytes.len() && local_idx < count {
                                    let s = u32::from_le_bytes(
                                        data_bytes[off_start..off_start + 4].try_into().unwrap(),
                                    ) as usize;
                                    let e = u32::from_le_bytes(
                                        data_bytes[off_end..off_end + 4].try_into().unwrap(),
                                    ) as usize;
                                    let offsets_end = 8 + (count + 1) * 4;
                                    let data_len_off = offsets_end;
                                    if data_len_off + 8 <= data_bytes.len() {
                                        let data_start = data_len_off + 8;
                                        if data_start + e <= data_bytes.len() {
                                            Value::Binary(
                                                data_bytes[data_start + s..data_start + e].to_vec(),
                                            )
                                        } else {
                                            Value::Null
                                        }
                                    } else {
                                        Value::Null
                                    }
                                } else {
                                    Value::Null
                                }
                            } else {
                                Value::Null
                            }
                        }
                        (COL_ENCODING_PLAIN, ColumnType::StringDict) => {
                            // StringDict: [row_count:u64][dict_size:u64][indices:row_count*4][dict_offsets:dict_size*4][dict_data_len:u64][dict_data]
                            if data_bytes.len() >= 16 {
                                let row_count =
                                    u64::from_le_bytes(data_bytes[0..8].try_into().unwrap())
                                        as usize;
                                let dict_size =
                                    u64::from_le_bytes(data_bytes[8..16].try_into().unwrap())
                                        as usize;
                                let idx_off = 16 + local_idx * 4;
                                if idx_off + 4 <= data_bytes.len() && local_idx < row_count {
                                    let dict_idx = u32::from_le_bytes(
                                        data_bytes[idx_off..idx_off + 4].try_into().unwrap(),
                                    );
                                    if dict_idx == 0 {
                                        Value::Null
                                    } else {
                                        let di = (dict_idx - 1) as usize;
                                        let dict_off_start = 16 + row_count * 4;
                                        let do_off = dict_off_start + di * 4;
                                        let do_off_next = dict_off_start + (di + 1) * 4;
                                        if do_off_next + 4 <= data_bytes.len() && di < dict_size {
                                            let ds = u32::from_le_bytes(
                                                data_bytes[do_off..do_off + 4].try_into().unwrap(),
                                            )
                                                as usize;
                                            let de = u32::from_le_bytes(
                                                data_bytes[do_off_next..do_off_next + 4]
                                                    .try_into()
                                                    .unwrap(),
                                            )
                                                as usize;
                                            let dict_data_len_off = dict_off_start + dict_size * 4;
                                            if dict_data_len_off + 8 <= data_bytes.len() {
                                                let dict_data_start = dict_data_len_off + 8;
                                                if dict_data_start + de <= data_bytes.len() {
                                                    Value::String(
                                                        std::str::from_utf8(
                                                            &data_bytes[dict_data_start + ds
                                                                ..dict_data_start + de],
                                                        )
                                                        .unwrap_or("")
                                                        .to_string(),
                                                    )
                                                } else {
                                                    Value::Null
                                                }
                                            } else {
                                                Value::Null
                                            }
                                        } else {
                                            Value::Null
                                        }
                                    }
                                } else {
                                    Value::Null
                                }
                            } else {
                                Value::Null
                            }
                        }
                        _ => {
                            // RLE/Bitpack/other: fallback to full decode, extract single value
                            let (col_data, _consumed) = if encoding_version >= 1 {
                                read_column_encoded(col_bytes, col_type)?
                            } else {
                                ColumnData::from_bytes_typed(col_bytes, col_type)?
                            };
                            match &col_data {
                                ColumnData::Int64(v) => {
                                    if local_idx < v.len() {
                                        Value::Int64(v[local_idx])
                                    } else {
                                        Value::Null
                                    }
                                }
                                ColumnData::Float64(v) => {
                                    if local_idx < v.len() {
                                        Value::Float64(v[local_idx])
                                    } else {
                                        Value::Null
                                    }
                                }
                                ColumnData::Bool { data, len } => {
                                    if local_idx < *len {
                                        Value::Bool(
                                            (data[local_idx / 8] >> (local_idx % 8)) & 1 == 1,
                                        )
                                    } else {
                                        Value::Null
                                    }
                                }
                                ColumnData::String { offsets, data } => {
                                    let count = offsets.len().saturating_sub(1);
                                    if local_idx < count {
                                        let s = offsets[local_idx] as usize;
                                        let e = offsets[local_idx + 1] as usize;
                                        Value::String(
                                            std::str::from_utf8(&data[s..e])
                                                .unwrap_or("")
                                                .to_string(),
                                        )
                                    } else {
                                        Value::Null
                                    }
                                }
                                ColumnData::Binary { offsets, data } => {
                                    let count = offsets.len().saturating_sub(1);
                                    if local_idx < count {
                                        let s = offsets[local_idx] as usize;
                                        let e = offsets[local_idx + 1] as usize;
                                        Value::Binary(data[s..e].to_vec())
                                    } else {
                                        Value::Null
                                    }
                                }
                                ColumnData::StringDict {
                                    indices,
                                    dict_offsets,
                                    dict_data,
                                    ..
                                } => {
                                    if local_idx < indices.len() {
                                        let di = indices[local_idx];
                                        if di == 0 {
                                            Value::Null
                                        } else {
                                            let idx = (di - 1) as usize;
                                            if idx < dict_offsets.len() {
                                                let s = dict_offsets[idx] as usize;
                                                let e = if idx + 1 < dict_offsets.len() {
                                                    dict_offsets[idx + 1] as usize
                                                } else {
                                                    dict_data.len()
                                                };
                                                Value::String(
                                                    std::str::from_utf8(&dict_data[s..e])
                                                        .unwrap_or("")
                                                        .to_string(),
                                                )
                                            } else {
                                                Value::Null
                                            }
                                        }
                                    } else {
                                        Value::Null
                                    }
                                }
                                _ => Value::Null,
                            }
                        }
                    };

                    result.push((col_name, val));
                }

                drop(mmap_guard);
                drop(file_guard);
                return Ok(Some(result));
            }

            // Fallback: compressed RG or no RCIX — decompress, binary search, sequential scan.
            let decompressed = decompress_rg_body(compress_flag, &rg_bytes[32..])?;
            let body: &[u8] = decompressed.as_deref().unwrap_or(&rg_bytes[32..]);
            if rg_rows * 8 > body.len() {
                return Ok(None);
            }
            let ids_cow = bytes_as_u64_slice(body, rg_rows);
            let local_idx = match ids_cow.binary_search(&id) {
                Ok(i) => i,
                Err(_) => return Ok(None),
            };
            let del_start = rg_rows * 8;
            let del_vec_len = (rg_rows + 7) / 8;
            if del_start + del_vec_len > body.len() {
                return Ok(None);
            }
            if (body[del_start + local_idx / 8] >> (local_idx % 8)) & 1 == 1 {
                return Ok(None);
            }
            let mut pos = del_start + del_vec_len;
            let null_bitmap_len = (rg_rows + 7) / 8;
            let schema = &footer.schema;
            let mut result = Vec::with_capacity(col_count + 1);
            result.push(("_id".to_string(), Value::Int64(id as i64)));
            for col_idx in 0..col_count {
                if pos + null_bitmap_len > body.len() {
                    break;
                }
                let null_bytes = &body[pos..pos + null_bitmap_len];
                pos += null_bitmap_len;
                let is_null = (null_bytes[local_idx / 8] >> (local_idx % 8)) & 1 == 1;
                let col_type = schema.columns[col_idx].1;
                let col_name = schema.columns[col_idx].0.clone();
                let col_bytes = &body[pos..];
                let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                let encoding = if encoding_version >= 1 && !col_bytes.is_empty() {
                    col_bytes[0]
                } else {
                    COL_ENCODING_PLAIN
                };
                let consumed = if encoding_version >= 1 {
                    skip_column_encoded(col_bytes, col_type)?
                } else {
                    ColumnData::skip_bytes_typed(col_bytes, col_type)?
                };
                if is_null {
                    pos += consumed;
                    result.push((col_name, Value::Null));
                    continue;
                }
                let data_bytes = &col_bytes[enc_offset..];
                let val = match (encoding, col_type) {
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
                    ) => {
                        let off = 8 + local_idx * 8;
                        if off + 8 <= data_bytes.len() {
                            Value::Int64(i64::from_le_bytes(
                                data_bytes[off..off + 8].try_into().unwrap(),
                            ))
                        } else {
                            Value::Null
                        }
                    }
                    (COL_ENCODING_PLAIN, ColumnType::Float64 | ColumnType::Float32) => {
                        let off = 8 + local_idx * 8;
                        if off + 8 <= data_bytes.len() {
                            Value::Float64(f64::from_le_bytes(
                                data_bytes[off..off + 8].try_into().unwrap(),
                            ))
                        } else {
                            Value::Null
                        }
                    }
                    (COL_ENCODING_PLAIN, ColumnType::Bool) => {
                        let byte_off = 8 + local_idx / 8;
                        if byte_off < data_bytes.len() {
                            Value::Bool((data_bytes[byte_off] >> (local_idx % 8)) & 1 == 1)
                        } else {
                            Value::Null
                        }
                    }
                    (COL_ENCODING_PLAIN, ColumnType::String) => {
                        if data_bytes.len() >= 8 {
                            let count =
                                u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
                            let off_s = 8 + local_idx * 4;
                            let off_e = 8 + (local_idx + 1) * 4;
                            if off_e + 4 <= data_bytes.len() && local_idx < count {
                                let s = u32::from_le_bytes(
                                    data_bytes[off_s..off_s + 4].try_into().unwrap(),
                                ) as usize;
                                let e = u32::from_le_bytes(
                                    data_bytes[off_e..off_e + 4].try_into().unwrap(),
                                ) as usize;
                                let ds = 8 + (count + 1) * 4 + 8;
                                if ds + e <= data_bytes.len() {
                                    Value::String(
                                        std::str::from_utf8(&data_bytes[ds + s..ds + e])
                                            .unwrap_or("")
                                            .to_string(),
                                    )
                                } else {
                                    Value::Null
                                }
                            } else {
                                Value::Null
                            }
                        } else {
                            Value::Null
                        }
                    }
                    _ => {
                        let (cd, _) = if encoding_version >= 1 {
                            read_column_encoded(col_bytes, col_type)?
                        } else {
                            ColumnData::from_bytes_typed(col_bytes, col_type)?
                        };
                        match &cd {
                            ColumnData::Int64(v) => {
                                if local_idx < v.len() {
                                    Value::Int64(v[local_idx])
                                } else {
                                    Value::Null
                                }
                            }
                            ColumnData::Float64(v) => {
                                if local_idx < v.len() {
                                    Value::Float64(v[local_idx])
                                } else {
                                    Value::Null
                                }
                            }
                            ColumnData::String { offsets, data } => {
                                let c = offsets.len().saturating_sub(1);
                                if local_idx < c {
                                    let s = offsets[local_idx] as usize;
                                    let e = offsets[local_idx + 1] as usize;
                                    Value::String(
                                        std::str::from_utf8(&data[s..e]).unwrap_or("").to_string(),
                                    )
                                } else {
                                    Value::Null
                                }
                            }
                            _ => Value::Null,
                        }
                    }
                };
                pos += consumed;
                result.push((col_name, val));
            }
            drop(mmap_guard);
            drop(file_guard);
            return Ok(Some(result));
        }
    }

    /// Fast SELECT * LIMIT N: read first N non-deleted rows directly from V4 columns
    /// Returns (column_names, rows) where each row is Vec<Value>
    /// Bypasses SQL parsing and Arrow conversion entirely
    pub fn read_rows_limit_values(
        &self,
        limit: usize,
    ) -> io::Result<Option<(Vec<String>, Vec<Vec<crate::data::Value>>)>> {
        use crate::data::Value;

        let is_v4 = self.is_v4_format();
        if !is_v4 {
            return Ok(None);
        }

        // MMAP PATH: scan from disk if no in-memory data
        if !self.has_v4_in_memory_data() {
            return self.read_rows_limit_values_mmap(limit);
        }

        let schema = self.schema.read();
        let columns = self.columns.read();
        let nulls = self.nulls.read();
        let ids = self.ids.read();
        let deleted = self.deleted.read();
        let total_rows = ids.len();
        let has_deleted = deleted.iter().any(|&b| b != 0);

        // Build column names
        let mut col_names = Vec::with_capacity(schema.column_count() + 1);
        col_names.push("_id".to_string());
        for (name, _) in &schema.columns {
            col_names.push(name.clone());
        }

        let actual_limit = limit.min(total_rows);
        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(actual_limit);
        let mut emitted = 0usize;

        for row_idx in 0..total_rows {
            if emitted >= limit {
                break;
            }
            // Skip deleted
            if has_deleted {
                let b = row_idx / 8;
                let bit = row_idx % 8;
                if b < deleted.len() && (deleted[b] >> bit) & 1 != 0 {
                    continue;
                }
            }

            let mut row = Vec::with_capacity(col_names.len());
            // _id
            row.push(if row_idx < ids.len() {
                Value::Int64(ids[row_idx] as i64)
            } else {
                Value::Null
            });

            for col_idx in 0..schema.column_count() {
                // Null check
                if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                    let b = row_idx / 8;
                    let bit = row_idx % 8;
                    if b < nulls[col_idx].len() && (nulls[col_idx][b] >> bit) & 1 == 1 {
                        row.push(Value::Null);
                        continue;
                    }
                }
                if col_idx >= columns.len() {
                    row.push(Value::Null);
                    continue;
                }

                let val = match &columns[col_idx] {
                    ColumnData::Int64(v) => {
                        if row_idx < v.len() {
                            Value::Int64(v[row_idx])
                        } else {
                            Value::Null
                        }
                    }
                    ColumnData::Float64(v) => {
                        if row_idx < v.len() {
                            Value::Float64(v[row_idx])
                        } else {
                            Value::Null
                        }
                    }
                    ColumnData::String { offsets, data } => {
                        let count = offsets.len().saturating_sub(1);
                        if row_idx < count {
                            let s = offsets[row_idx] as usize;
                            let e = offsets[row_idx + 1] as usize;
                            Value::String(
                                std::str::from_utf8(&data[s..e]).unwrap_or("").to_string(),
                            )
                        } else {
                            Value::Null
                        }
                    }
                    ColumnData::Bool { data, len } => {
                        if row_idx < *len {
                            let b = row_idx / 8;
                            let bit = row_idx % 8;
                            if b < data.len() {
                                Value::Bool((data[b] >> bit) & 1 == 1)
                            } else {
                                Value::Null
                            }
                        } else {
                            Value::Null
                        }
                    }
                    ColumnData::Binary { offsets, data } => {
                        let count = offsets.len().saturating_sub(1);
                        if row_idx < count {
                            let s = offsets[row_idx] as usize;
                            let e = offsets[row_idx + 1] as usize;
                            Value::Binary(data[s..e].to_vec())
                        } else {
                            Value::Null
                        }
                    }
                    ColumnData::StringDict {
                        indices,
                        dict_offsets,
                        dict_data,
                    } => {
                        if row_idx < indices.len() {
                            let idx = indices[row_idx];
                            if idx == 0 {
                                Value::Null
                            } else {
                                let di = (idx - 1) as usize;
                                if di + 1 < dict_offsets.len() {
                                    let s = dict_offsets[di] as usize;
                                    let e = dict_offsets[di + 1] as usize;
                                    Value::String(
                                        std::str::from_utf8(&dict_data[s..e])
                                            .unwrap_or("")
                                            .to_string(),
                                    )
                                } else {
                                    Value::Null
                                }
                            }
                        } else {
                            Value::Null
                        }
                    }
                    _ => Value::Null,
                };
                row.push(val);
            }
            rows.push(row);
            emitted += 1;
        }

        Ok(Some((col_names, rows)))
    }

    /// MMAP PATH: Fast SELECT * LIMIT N
    fn read_rows_limit_values_mmap(
        &self,
        _limit: usize,
    ) -> io::Result<Option<(Vec<String>, Vec<Vec<crate::data::Value>>)>> {
        Ok(None)
    }

    /// OPTIMIZED: Read multiple rows by IDs using O(1) index lookups
    /// Returns Vec of (id, row_data) for found rows
    pub fn read_rows_by_ids(
        &self,
        ids: &[u64],
        column_names: Option<&[&str]>,
    ) -> io::Result<Vec<(u64, HashMap<String, ColumnData>)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let is_v4 = self.is_v4_format();
        if is_v4 && !self.has_v4_in_memory_data() {
            // MMAP PATH: delegate to read_row_by_id per ID
            let mut results = Vec::with_capacity(ids.len());
            for &id in ids {
                if let Some(row) = self.read_row_by_id(id, column_names)? {
                    results.push((id, row));
                }
            }
            return Ok(results);
        }

        // Build id_to_idx if needed
        self.ensure_id_index();
        let id_to_idx = self.id_to_idx.read();
        let map = id_to_idx.as_ref().unwrap();

        // Collect valid row indices
        let mut valid_ids_indices: Vec<(u64, usize)> = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Some(&row_idx) = map.get(&id) {
                if !self.is_deleted(row_idx) {
                    valid_ids_indices.push((id, row_idx));
                }
            }
        }

        if valid_ids_indices.is_empty() {
            return Ok(Vec::new());
        }

        let indices: Vec<usize> = valid_ids_indices.iter().map(|(_, idx)| *idx).collect();
        drop(id_to_idx);

        // Read columns
        let schema = self.schema.read();

        let cols_to_read: Vec<(usize, String, ColumnType)> = if let Some(names) = column_names {
            names
                .iter()
                .filter_map(|&name| {
                    if name == "_id" {
                        None
                    } else {
                        schema
                            .get_index(name)
                            .map(|idx| (idx, name.to_string(), schema.columns[idx].1))
                    }
                })
                .collect()
        } else {
            schema
                .columns
                .iter()
                .enumerate()
                .map(|(idx, (name, dtype))| (idx, name.clone(), *dtype))
                .collect()
        };

        let total_rows = self.header.read().row_count as usize;
        drop(schema);

        // Read all columns for all indices
        let mut column_data: HashMap<String, ColumnData> = HashMap::new();
        let include_id = column_names
            .map(|cols| cols.contains(&"_id"))
            .unwrap_or(true);

        for (col_idx, col_name, col_type) in cols_to_read {
            let col_data =
                self.read_column_scattered_auto(col_idx, col_type, &indices, total_rows, is_v4)?;
            column_data.insert(col_name, col_data);
        }

        // Split into per-row results
        let mut results = Vec::with_capacity(valid_ids_indices.len());
        for (i, (id, _)) in valid_ids_indices.iter().enumerate() {
            let mut row_data = HashMap::new();
            if include_id {
                row_data.insert("_id".to_string(), ColumnData::Int64(vec![*id as i64]));
            }
            for (col_name, col_data) in &column_data {
                let single_val = col_data.filter_by_indices(&[i]);
                row_data.insert(col_name.clone(), single_val);
            }
            results.push((*id, row_data));
        }

        Ok(results)
    }

    /// Get the count of non-deleted rows (includes delta rows)
    pub fn active_row_count(&self) -> u64 {
        let base_active = self.active_count.load(std::sync::atomic::Ordering::Relaxed);
        let delta_rows = self.delta_row_count() as u64;
        let pending_delta_deletes = self.delta_store.read().delete_count() as u64;
        base_active.saturating_sub(pending_delta_deletes) + delta_rows
    }

    /// Drop a column from schema (logical delete - data stays but column is removed from schema)
    /// When save() is called, only columns in schema will be written to file
    pub fn drop_column(&self, name: &str) -> io::Result<()> {
        let mut schema = self.schema.write();

        // Find column index
        let idx = match schema.get_index(name) {
            Some(idx) => idx,
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Column '{}' not found", name),
                ))
            }
        };

        // Remove from schema (logical delete)
        schema.columns.remove(idx);
        schema.name_to_idx.remove(name);

        // Rebuild name_to_idx with updated indices
        // Collect names first to avoid borrow conflict
        let names: Vec<String> = schema.columns.iter().map(|(n, _)| n.clone()).collect();
        schema.name_to_idx.clear();
        for (i, n) in names.into_iter().enumerate() {
            schema.name_to_idx.insert(n, i);
        }

        // Also remove from in-memory structures to keep them in sync with schema
        // This ensures save() writes correct data
        {
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();
            let mut column_index = self.column_index.write();

            if idx < columns.len() {
                columns.remove(idx);
            }
            if idx < nulls.len() {
                nulls.remove(idx);
            }
            if idx < column_index.len() {
                column_index.remove(idx);
            }
        }

        // Update header column count
        {
            let mut header = self.header.write();
            header.column_count = schema.column_count() as u32;
        }

        Ok(())
    }

    /// Add a new column to schema and storage with padding for existing rows
    pub fn add_column_with_padding(
        &self,
        name: &str,
        dtype: crate::data::DataType,
    ) -> io::Result<()> {
        use crate::data::DataType;

        // For V4, schema is updated via footer; data stays on disk (mmap)
        self.load_all_columns_into_memory()?;

        let col_type = match dtype {
            DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                ColumnType::Int64
            }
            DataType::Float64 | DataType::Float32 => ColumnType::Float64,
            DataType::String => ColumnType::String,
            DataType::Bool => ColumnType::Bool,
            DataType::Binary => ColumnType::Binary,
            DataType::Timestamp => ColumnType::Timestamp,
            DataType::Date => ColumnType::Date,
            DataType::Float16Vector => ColumnType::Float16List,
            _ => ColumnType::String,
        };

        let mut schema = self.schema.write();
        let mut columns = self.columns.write();
        let mut nulls = self.nulls.write();
        // Use header.row_count for V4 (IDs may not be loaded in mmap-only mode)
        let existing_row_count = {
            let header = self.header.read();
            let from_header = header.row_count as usize;
            drop(header);
            let ids = self.ids.read();
            let from_ids = ids.len();
            drop(ids);
            from_header.max(from_ids)
        };

        // Add to schema
        let idx = schema.add_column(name, col_type);

        // Ensure columns vector is large enough
        while columns.len() <= idx {
            let mut col = ColumnData::new(col_type);
            // Pad with defaults for existing rows
            match &mut col {
                ColumnData::Int64(v) => v.resize(existing_row_count, 0),
                ColumnData::Float64(v) => v.resize(existing_row_count, 0.0),
                ColumnData::String { offsets, .. } => {
                    for _ in 0..existing_row_count {
                        offsets.push(0);
                    }
                }
                ColumnData::Binary { offsets, .. } => {
                    for _ in 0..existing_row_count {
                        offsets.push(0);
                    }
                }
                ColumnData::Bool { len, .. } => {
                    *len = existing_row_count;
                }
                ColumnData::StringDict { indices, .. } => {
                    indices.resize(existing_row_count, 0);
                }
                ColumnData::FixedList { .. } => {}
                ColumnData::Float16List { .. } => {}
            }
            columns.push(col);
            nulls.push(Vec::new());
        }

        // Update header
        {
            let mut header = self.header.write();
            header.column_count = schema.column_count() as u32;
        }

        Ok(())
    }

    /// Replace a row by ID (delete old row, insert new with SAME ID)
    /// Returns true if successful
    pub fn replace(&self, id: u64, data: &HashMap<String, ColumnValue>) -> io::Result<bool> {
        // Check if ID exists
        if !self.exists(id) {
            return Ok(false);
        }

        // Delete the old row (soft delete)
        self.delete(id);

        // Convert data to typed columns for insert_typed
        let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
        let mut binary_columns: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();

        for (name, val) in data {
            match val {
                ColumnValue::Int64(v) => {
                    int_columns.insert(name.clone(), vec![*v]);
                }
                ColumnValue::Float64(v) => {
                    float_columns.insert(name.clone(), vec![*v]);
                }
                ColumnValue::String(v) => {
                    string_columns.insert(name.clone(), vec![v.clone()]);
                }
                ColumnValue::Binary(v) => {
                    binary_columns.insert(name.clone(), vec![v.clone()]);
                }
                ColumnValue::FixedList(v) => {
                    binary_columns.insert(name.clone(), vec![v.clone()]);
                }
                ColumnValue::Bool(v) => {
                    bool_columns.insert(name.clone(), vec![*v]);
                }
                ColumnValue::Null => {}
            }
        }

        // Use insert_typed but override the ID
        // First, determine row count (should be 1)
        let row_count = 1;

        // Instead of using next_id, we'll use the original ID
        let ids = vec![id];

        // Ensure schema has all columns and pad new columns with defaults
        {
            let mut schema = self.schema.write();
            let mut columns = self.columns.write();
            let mut nulls = self.nulls.write();
            let ids = self.ids.read();
            let existing_row_count = ids.len();
            drop(ids);

            for name in int_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Int64);
                while columns.len() <= idx {
                    // New column - pad with defaults for existing rows
                    let mut col = ColumnData::new(ColumnType::Int64);
                    if let ColumnData::Int64(v) = &mut col {
                        v.resize(existing_row_count, 0);
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in float_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Float64);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Float64);
                    if let ColumnData::Float64(v) = &mut col {
                        v.resize(existing_row_count, 0.0);
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in string_columns.keys() {
                let idx = schema.add_column(name, ColumnType::String);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::String);
                    if let ColumnData::String { offsets, .. } = &mut col {
                        // For strings, push empty string offsets for existing rows
                        for _ in 0..existing_row_count {
                            offsets.push(0);
                        }
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in binary_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Binary);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Binary);
                    if let ColumnData::Binary { offsets, .. } = &mut col {
                        for _ in 0..existing_row_count {
                            offsets.push(0);
                        }
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
            for name in bool_columns.keys() {
                let idx = schema.add_column(name, ColumnType::Bool);
                while columns.len() <= idx {
                    let mut col = ColumnData::new(ColumnType::Bool);
                    if let ColumnData::Bool { len, .. } = &mut col {
                        *len = existing_row_count;
                    }
                    columns.push(col);
                    nulls.push(Vec::new());
                }
            }
        }

        // Append ID
        self.ids.write().extend_from_slice(&ids);

        // Append column data
        {
            let schema = self.schema.read();
            let mut columns = self.columns.write();

            for (name, values) in int_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_i64(&values);
                }
            }
            for (name, values) in float_columns {
                if let Some(idx) = schema.get_index(&name) {
                    columns[idx].extend_f64(&values);
                }
            }
            for (name, values) in string_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in &values {
                        columns[idx].push_string(v);
                    }
                }
            }
            for (name, values) in binary_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in &values {
                        columns[idx].push_bytes(v);
                    }
                }
            }
            for (name, values) in bool_columns {
                if let Some(idx) = schema.get_index(&name) {
                    for v in values {
                        columns[idx].push_bool(v);
                    }
                }
            }

            // Pad any schema columns not in the replacement data with defaults + null
            let expected_len = self.ids.read().len();
            let mut nulls = self.nulls.write();
            for col_idx in 0..schema.column_count() {
                if col_idx < columns.len() && columns[col_idx].len() < expected_len {
                    // This column wasn't in the replacement — pad with default
                    let deficit = expected_len - columns[col_idx].len();
                    for _ in 0..deficit {
                        match &mut columns[col_idx] {
                            ColumnData::Int64(v) => v.push(0),
                            ColumnData::Float64(v) => v.push(0.0),
                            ColumnData::String { offsets, .. } => {
                                offsets.push(*offsets.last().unwrap_or(&0));
                            }
                            ColumnData::Binary { offsets, .. } => {
                                offsets.push(*offsets.last().unwrap_or(&0));
                            }
                            ColumnData::Bool { data, len } => {
                                let byte_idx = *len / 8;
                                if byte_idx >= data.len() {
                                    data.push(0);
                                }
                                *len += 1;
                            }
                            ColumnData::StringDict { indices, .. } => indices.push(0),
                            ColumnData::FixedList { .. } => {} // pads implicitly
                            ColumnData::Float16List { .. } => {} // pads implicitly
                        }
                    }
                    // Mark padded rows as null
                    if col_idx >= nulls.len() {
                        nulls.resize(col_idx + 1, Vec::new());
                    }
                    let total_rows = expected_len;
                    let null_len = (total_rows + 7) / 8;
                    nulls[col_idx].resize(null_len, 0);
                    for row in (total_rows - deficit)..total_rows {
                        nulls[col_idx][row / 8] |= 1 << (row % 8);
                    }
                }
            }
        }

        // Update header
        {
            let mut header = self.header.write();
            header.row_count = self.ids.read().len() as u64;
            header.column_count = self.schema.read().column_count() as u32;
        }

        // Update id_to_idx mapping only if it's already built
        {
            let ids_guard = self.ids.read();
            let mut id_to_idx = self.id_to_idx.write();
            if let Some(map) = id_to_idx.as_mut() {
                let row_idx = ids_guard.len() - 1;
                map.insert(id, row_idx);
            }
        }

        // Extend deleted bitmap
        {
            let mut deleted = self.deleted.write();
            let new_len = (self.ids.read().len() + 7) / 8;
            deleted.resize(new_len, 0);
        }

        Ok(true)
    }

    // ========================================================================
    // Persistence
    // ========================================================================

    /// Check if a string column should use dictionary encoding
    /// Returns true if unique values < 20% of row count and row count > 1000
    fn should_dict_encode(col: &ColumnData) -> bool {
        if let ColumnData::String { offsets, data } = col {
            let row_count = offsets.len().saturating_sub(1);
            if row_count < 1000 {
                return false;
            }
            // Estimate unique values by sampling
            use ahash::AHashSet;
            let sample_size = (row_count / 10).min(1000);
            let mut unique: AHashSet<&[u8]> = AHashSet::with_capacity(sample_size);
            for i in 0..sample_size {
                let idx = i * 10; // Sample every 10th row
                if idx < row_count {
                    let start = offsets[idx] as usize;
                    let end = offsets[idx + 1] as usize;
                    unique.insert(&data[start..end]);
                }
            }
            // Use dictionary if cardinality < 20% of sampled rows
            unique.len() < sample_size / 5
        } else {
            false
        }
    }

    /// Save to file (full rewrite with V4 format)
    ///
    /// MEMORY OPTIMIZED: Processes one column at a time using placeholder + seek-back.
    /// Peak memory = original columns (already in memory) + 1 filtered column copy,
    /// instead of original columns + ALL filtered column copies.
    ///
    /// Automatically converts low-cardinality string columns to dictionary encoding.
    pub fn save(&self) -> io::Result<()> {
        // OPTIMIZATION: For existing V4 files with only deletions (no new rows,
        // no schema changes), update deletion vectors in-place instead of full rewrite.
        // All other cases use the proven save_v4() full-rewrite path.
        // Note: append optimization is handled at engine level (write_typed→append_row_group).
        let header = self.header.read();
        let is_v4 = header.version == FORMAT_VERSION_V4 && header.footer_offset > 0;
        drop(header);

        if is_v4 {
            let on_disk_rows = self.persisted_row_count.load(Ordering::SeqCst) as usize;
            let ids = self.ids.read();
            let in_memory_ids = ids.len();
            let has_new_rows = in_memory_ids > 0;
            let base_loaded = self.v4_base_loaded.load(Ordering::SeqCst);
            let has_unloaded_base = on_disk_rows > 0 && in_memory_ids > 0 && !base_loaded;
            drop(ids);

            // If base data isn't loaded but we have new rows, append the
            // already-buffered rows incrementally. Do not call append_row_group()
            // here: that API is for rows that are not yet in memory and will
            // extend ids/active_count after writing. These rows are the memtable
            // buffer itself, so extending again duplicates them on every flush.
            if has_unloaded_base {
                let ids = self.ids.read();
                let new_ids: Vec<u64> = ids.clone();
                drop(ids);
                let cols = self.columns.read();
                let new_cols: Vec<ColumnData> = cols.clone();
                drop(cols);
                let nulls = self.nulls.read();
                let new_nulls: Vec<Vec<u8>> = nulls.clone();
                drop(nulls);
                self.pending_rows.store(0, Ordering::SeqCst);
                self.write_row_group_to_disk(&new_ids, &new_cols, &new_nulls)?;

                // Keep a warm insert backend as metadata-only after persistence.
                // Future single-row appends can reuse the schema/next_id/footer
                // without treating the already-persisted rows as pending again.
                self.ids.write().clear();
                self.columns.write().clear();
                self.nulls.write().clear();
                self.deleted.write().clear();
                *self.id_to_idx.write() = None;
                if self.has_pending_deltas() {
                    self.save_delta_store()?;
                }
                return Ok(());
            }

            if !has_new_rows && !base_loaded && on_disk_rows > 0 {
                // Schema-only change (add/drop/rename column) on V4 mmap-only.
                // Base data is NOT in memory — must NOT call save_v4() which would
                // rewrite with empty data and lose everything.
                // Instead, update just the footer schema on disk.
                if self.has_pending_deltas() {
                    self.save_delta_store()?;
                }
                return self.update_v4_footer_schema();
            }

            if !has_new_rows {
                let deleted = self.deleted.read();
                let has_deletes = deleted.iter().any(|&b| b != 0);
                if has_deletes {
                    // Count deleted rows for compaction threshold
                    let del_count = (0..on_disk_rows)
                        .filter(|&i| {
                            let byte_idx = i / 8;
                            let bit_idx = i % 8;
                            byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 == 1
                        })
                        .count();
                    drop(deleted);
                    let ratio = if on_disk_rows > 0 {
                        del_count as f64 / on_disk_rows as f64
                    } else {
                        0.0
                    };

                    if ratio <= 0.5 {
                        // Low deletion ratio → update deletion vectors in-place
                        self.pending_rows.store(0, Ordering::SeqCst);
                        // Also persist delta store if it has pending changes
                        if self.has_pending_deltas() {
                            let _ = self.save_delta_store();
                        }
                        return self.save_deletion_vectors();
                    }
                    // High deletion ratio → full rewrite to reclaim space (fall through)
                }
            }
        }

        self.pending_rows.store(0, Ordering::SeqCst);
        let result = self.save_v4();
        // After full rewrite, clear delta store (deltas are now in the base file)
        if result.is_ok() {
            let _ = self.clear_delta_store();
            // WAL checkpoint: all data is persisted, truncate WAL to prevent unbounded growth
            self.checkpoint_wal();
        }
        result
    }

    /// Force a full base-file rewrite from the current in-memory state.
    ///
    /// SQL DML paths use this when later statements in the same call must observe a
    /// fully materialized base table instead of append-only spill sidecars.
    pub fn save_full(&self) -> io::Result<()> {
        self.pending_rows.store(0, Ordering::SeqCst);
        let result = self.save_v4();
        if result.is_ok() {
            let _ = self.clear_delta_store();
            self.checkpoint_wal();
            *self.delta_file.write() = None;
            let delta_path = Self::delta_path(&self.path);
            let _ = std::fs::remove_file(&delta_path);
            let _ = std::fs::remove_file(Self::delta_meta_path(&delta_path));
        }
        result
    }

    /// Checkpoint WAL: truncate the WAL file after a successful save.
    /// All WAL records are now redundant since data is fully persisted to .apex.
    fn checkpoint_wal(&self) {
        let mut wal_writer = self.wal_writer.write();
        if wal_writer.is_some() {
            // Drop the existing writer to release file handle
            *wal_writer = None;
            // Recreate WAL file (truncates old content)
            let wal_path = Self::wal_path(&self.path);
            let next_id = self.next_id.load(Ordering::SeqCst);
            if let Ok(writer) = super::incremental::WalWriter::create(&wal_path, next_id) {
                *wal_writer = Some(writer);
            }
        }
        // Clear WAL buffer
        let mut wal_buffer = self.wal_buffer.write();
        wal_buffer.clear();
    }

    // ========================================================================
    // V4 Row Group Format — Save / Open / Append
    // ========================================================================

    /// Slice a null bitmap for a contiguous row range [start, end).
    /// OPTIMIZATION: uses bulk memcpy when start is byte-aligned.
    fn slice_null_bitmap(nulls: &[u8], start: usize, end: usize) -> Vec<u8> {
        let count = end.saturating_sub(start);
        if count == 0 || nulls.is_empty() {
            return vec![0u8; (count + 7) / 8];
        }
        let result_len = (count + 7) / 8;
        if start % 8 == 0 {
            let src_byte = start / 8;
            let copy_len = result_len.min(nulls.len().saturating_sub(src_byte));
            let mut result = vec![0u8; result_len];
            if copy_len > 0 {
                result[..copy_len].copy_from_slice(&nulls[src_byte..src_byte + copy_len]);
            }
            let tail_bits = count % 8;
            if tail_bits > 0 && result_len > 0 {
                result[result_len - 1] &= (1u8 << tail_bits) - 1;
            }
            return result;
        }
        let mut result = vec![0u8; result_len];
        for i in 0..count {
            let ob = (start + i) / 8;
            let obit = (start + i) % 8;
            if ob < nulls.len() && (nulls[ob] >> obit) & 1 == 1 {
                result[i / 8] |= 1 << (i % 8);
            }
        }
        result
    }

    /// Save in V4 Row Group format.
    /// Splits data into Row Groups of DEFAULT_ROW_GROUP_SIZE rows each.
    /// Each RG is self-contained with IDs, deletion vector, and per-column data.
    ///
    /// V4 File Layout:
    /// ```text
    /// [Header 256B] [RG0] [RG1] ... [V4Footer]
    /// ```
    pub fn save_v4(&self) -> io::Result<()> {
        self.mmap_cache.write().invalidate();
        self.invalidate_page_cache();
        *self.file.write() = None;
        *self.write_file.write() = None;
        // On Windows, active mmaps prevent file truncate/write (OS error 1224).
        // Must invalidate ALL caches (engine cache + insert_cache + schema_cache + executor STORAGE_CACHE).
        // On Unix/Linux, only executor cache needs invalidation (mmaps don't block writes).
        #[cfg(target_os = "windows")]
        super::engine::engine().invalidate(&self.path);
        #[cfg(not(target_os = "windows"))]
        crate::query::ApexExecutor::invalidate_cache_for_path(&self.path);

        // Atomic write: write to .tmp file, then rename over the original.
        // If crash occurs mid-write, only the .tmp file is corrupted; original is intact.
        let tmp_path = self.path.with_extension("apex.tmp");
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        // On Windows, larger write buffers reduce syscall overhead significantly.
        #[cfg(windows)]
        let mut writer = BufWriter::with_capacity(2 * 1024 * 1024, file);
        #[cfg(not(windows))]
        let mut writer = BufWriter::with_capacity(256 * 1024, file);

        // Phase 1: Build filtered (active) data under read guards.
        // This produces clean flat columns/ids/nulls with deleted rows removed
        // and missing columns padded. Used for both disk write and in-memory state.
        let active_ids: Vec<u64>;
        let mut active_columns: Vec<ColumnData>;
        let mut active_nulls: Vec<Vec<u8>>;
        let active_count: usize;
        let col_count: usize;
        let schema_clone: OnDemandSchema;

        {
            let schema = self.schema.read();
            let ids = self.ids.read();
            let columns = self.columns.read();
            let nulls = self.nulls.read();
            let deleted = self.deleted.read();

            col_count = schema.column_count();
            schema_clone = schema.clone();
            let has_deleted = deleted.iter().any(|&b| b != 0);

            if has_deleted {
                let indices: Vec<usize> = (0..ids.len())
                    .filter(|&i| {
                        let byte_idx = i / 8;
                        let bit_idx = i % 8;
                        byte_idx >= deleted.len() || (deleted[byte_idx] >> bit_idx) & 1 == 0
                    })
                    .collect();
                active_ids = indices.iter().map(|&i| ids[i]).collect();
                active_count = indices.len();

                active_columns = Vec::with_capacity(col_count);
                active_nulls = Vec::with_capacity(col_count);
                for col_idx in 0..col_count {
                    // Filter column data
                    if col_idx < columns.len() {
                        active_columns.push(columns[col_idx].filter_by_indices(&indices));
                    } else {
                        active_columns.push(Self::create_default_column(
                            schema.columns[col_idx].1,
                            active_count,
                        ));
                    }
                    // Filter null bitmap
                    let orig_nulls = nulls.get(col_idx).map(|v| v.as_slice()).unwrap_or(&[]);
                    let null_len = (active_count + 7) / 8;
                    let mut nb = vec![0u8; null_len];
                    for (new_idx, &old_idx) in indices.iter().enumerate() {
                        let ob = old_idx / 8;
                        let obit = old_idx % 8;
                        if ob < orig_nulls.len() && (orig_nulls[ob] >> obit) & 1 == 1 {
                            nb[new_idx / 8] |= 1 << (new_idx % 8);
                        }
                    }
                    active_nulls.push(nb);
                }
            } else {
                active_ids = ids.to_vec();
                active_count = ids.len();

                active_columns = Vec::with_capacity(col_count);
                active_nulls = Vec::with_capacity(col_count);
                for col_idx in 0..col_count {
                    if col_idx < columns.len() {
                        active_columns.push(columns[col_idx].clone());
                    } else {
                        active_columns.push(Self::create_default_column(
                            schema.columns[col_idx].1,
                            active_count,
                        ));
                    }
                    active_nulls.push(nulls.get(col_idx).map(|v| v.to_vec()).unwrap_or_default());
                }
            }
        } // All read guards dropped here

        // Phase 2: Write V4 format from active data (no lock contention).
        let adaptive_rg_size =
            Self::compute_adaptive_row_group_size(&schema_clone, &active_columns, active_count);
        {
            let mut header = self.header.write();
            header.row_group_size = adaptive_rg_size;
        }
        let rg_size = adaptive_rg_size as usize;

        // Write placeholder header
        writer.write_all(&[0u8; HEADER_SIZE])?;

        // Write Row Groups
        let mut rg_metas: Vec<RowGroupMeta> = Vec::new();
        let mut all_zone_maps: RgZoneMaps = Vec::new();
        let mut all_rg_col_offsets: Vec<Vec<u32>> = Vec::new();
        let mut actual_col_types: Vec<ColumnType> = Vec::new();
        let mut chunk_start = 0;

        while chunk_start < active_count || (active_count == 0 && rg_metas.is_empty()) {
            let chunk_end = (chunk_start + rg_size).min(active_count);
            let chunk_rows = chunk_end - chunk_start;

            // Handle empty table — write one empty RG
            if active_count == 0 && rg_metas.is_empty() {
                let rg_offset = writer.stream_position()?;
                writer.write_all(MAGIC_ROW_GROUP)?;
                writer.write_all(&0u32.to_le_bytes())?;
                writer.write_all(&(col_count as u32).to_le_bytes())?;
                writer.write_all(&0u64.to_le_bytes())?;
                writer.write_all(&0u64.to_le_bytes())?;
                writer.write_all(&[0u8; 4])?;
                let rg_end = writer.stream_position()?;
                rg_metas.push(RowGroupMeta {
                    offset: rg_offset,
                    data_size: rg_end - rg_offset,
                    row_count: 0,
                    min_id: 0,
                    max_id: 0,
                    deletion_count: 0,
                });
                break;
            }

            let rg_offset = writer.stream_position()?;
            let chunk_ids = &active_ids[chunk_start..chunk_end];
            let min_id = chunk_ids.iter().copied().min().unwrap_or(0);
            let max_id = chunk_ids.iter().copied().max().unwrap_or(0);

            // Serialize RG body to buffer (IDs + deletion vector + columns)
            let is_single_rg = chunk_start == 0 && chunk_end == active_count;
            let null_bitmap_len = (chunk_rows + 7) / 8;
            let mut body_buf: Vec<u8> = Vec::with_capacity(chunk_rows * 8 + chunk_rows * col_count);
            {
                let mut body_writer = std::io::Cursor::new(&mut body_buf);

                // IDs — bulk write via unsafe slice cast
                let id_bytes = unsafe {
                    std::slice::from_raw_parts(chunk_ids.as_ptr() as *const u8, chunk_ids.len() * 8)
                };
                body_writer.write_all(id_bytes)?;

                // Deletion vector (all zeros — fresh save, no deletes)
                let del_vec_len = (chunk_rows + 7) / 8;
                body_writer.write_all(&vec![0u8; del_vec_len])?;

                // Columns
                let mut rg_col_offsets: Vec<u32> = Vec::with_capacity(col_count);
                for col_idx in 0..col_count {
                    let chunk_col_owned;
                    let chunk_col_ref: &ColumnData = if is_single_rg {
                        &active_columns[col_idx]
                    } else {
                        chunk_col_owned =
                            active_columns[col_idx].slice_range(chunk_start, chunk_end);
                        &chunk_col_owned
                    };

                    // Dict-encode low-cardinality string columns for disk
                    let dict_encoded;
                    let processed: &ColumnData = if Self::should_dict_encode(chunk_col_ref) {
                        dict_encoded = chunk_col_ref
                            .to_dict_encoded()
                            .unwrap_or_else(|| chunk_col_ref.clone());
                        &dict_encoded
                    } else {
                        chunk_col_ref
                    };

                    // Track actual type for footer schema
                    if rg_metas.is_empty() {
                        let actual_type = match processed {
                            ColumnData::StringDict { .. } => ColumnType::StringDict,
                            _ => schema_clone.columns[col_idx].1,
                        };
                        actual_col_types.push(actual_type);
                    }

                    // Record body offset of this column's null bitmap for RCIX
                    rg_col_offsets.push(body_writer.position() as u32);

                    // Null bitmap
                    if is_single_rg && active_nulls[col_idx].len() == null_bitmap_len {
                        body_writer.write_all(&active_nulls[col_idx])?;
                    } else {
                        let chunk_nulls =
                            Self::slice_null_bitmap(&active_nulls[col_idx], chunk_start, chunk_end);
                        body_writer.write_all(&chunk_nulls)?;
                    }
                    write_column_encoded(
                        processed,
                        schema_clone.columns[col_idx].1,
                        &mut body_writer,
                    )?
                }
                all_rg_col_offsets.push(rg_col_offsets);
            }

            // Compress body using configured compression algorithm
            let (compress_flag, disk_body) = compress_rg_body(body_buf, self.compression());

            // RG header (32 bytes) — byte 28 = compression flag, byte 29 = encoding version
            writer.write_all(MAGIC_ROW_GROUP)?;
            writer.write_all(&(chunk_rows as u32).to_le_bytes())?;
            writer.write_all(&(col_count as u32).to_le_bytes())?;
            writer.write_all(&min_id.to_le_bytes())?;
            writer.write_all(&max_id.to_le_bytes())?;
            writer.write_all(&[compress_flag, 1, 0, 0])?; // encoding_version=1: per-column encoding prefix

            // RG body (possibly compressed)
            writer.write_all(&disk_body)?;

            // Compute zone maps for this RG's numeric columns
            let mut rg_zmaps: Vec<RgColumnZoneMap> = Vec::new();
            for col_idx in 0..col_count {
                let chunk_col_ref: &ColumnData = if is_single_rg {
                    &active_columns[col_idx]
                } else {
                    // Already sliced above — re-slice for zone map
                    // Use active_columns directly since we only need min/max
                    &active_columns[col_idx]
                };
                match chunk_col_ref {
                    ColumnData::Int64(data) => {
                        if !data.is_empty() {
                            let slice = if is_single_rg {
                                &data[..]
                            } else {
                                &data[chunk_start..chunk_end]
                            };
                            let (mut mn, mut mx) = (i64::MAX, i64::MIN);
                            for &v in slice {
                                mn = mn.min(v);
                                mx = mx.max(v);
                            }
                            rg_zmaps.push(RgColumnZoneMap {
                                col_idx: col_idx as u16,
                                min_bits: mn,
                                max_bits: mx,
                                has_nulls: false,
                                is_float: false,
                            });
                        }
                    }
                    ColumnData::Float64(data) => {
                        if !data.is_empty() {
                            let slice = if is_single_rg {
                                &data[..]
                            } else {
                                &data[chunk_start..chunk_end]
                            };
                            let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
                            for &v in slice {
                                if v < mn {
                                    mn = v;
                                }
                                if v > mx {
                                    mx = v;
                                }
                            }
                            rg_zmaps.push(RgColumnZoneMap {
                                col_idx: col_idx as u16,
                                min_bits: mn.to_bits() as i64,
                                max_bits: mx.to_bits() as i64,
                                has_nulls: false,
                                is_float: true,
                            });
                        }
                    }
                    ColumnData::String { offsets, .. } => {
                        // Store min/max string byte-length for zone-skip in string filter scans
                        let end_idx = chunk_end.min(offsets.len().saturating_sub(1));
                        let start_idx = if is_single_rg { 0 } else { chunk_start };
                        if end_idx > start_idx {
                            let (mut mn, mut mx) = (u32::MAX, 0u32);
                            for i in start_idx..end_idx {
                                let len = offsets[i + 1].saturating_sub(offsets[i]);
                                if len < mn {
                                    mn = len;
                                }
                                if len > mx {
                                    mx = len;
                                }
                            }
                            if mn <= mx {
                                rg_zmaps.push(RgColumnZoneMap {
                                    col_idx: col_idx as u16,
                                    min_bits: mn as i64,
                                    max_bits: mx as i64,
                                    has_nulls: false,
                                    is_float: false,
                                });
                            }
                        }
                    }
                    ColumnData::StringDict { dict_offsets, .. } => {
                        // Store min/max dict-entry byte-length for zone-skip
                        if dict_offsets.len() > 1 {
                            let (mut mn, mut mx) = (u32::MAX, 0u32);
                            for i in 0..dict_offsets.len().saturating_sub(1) {
                                let len = dict_offsets[i + 1].saturating_sub(dict_offsets[i]);
                                if len < mn {
                                    mn = len;
                                }
                                if len > mx {
                                    mx = len;
                                }
                            }
                            if mn <= mx {
                                rg_zmaps.push(RgColumnZoneMap {
                                    col_idx: col_idx as u16,
                                    min_bits: mn as i64,
                                    max_bits: mx as i64,
                                    has_nulls: false,
                                    is_float: false,
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
            all_zone_maps.push(rg_zmaps);

            let rg_end = writer.stream_position()?;
            rg_metas.push(RowGroupMeta {
                offset: rg_offset,
                data_size: rg_end - rg_offset,
                row_count: chunk_rows as u32,
                min_id,
                max_id,
                deletion_count: 0,
            });

            chunk_start = chunk_end;
        }

        // Build modified schema with actual types (StringDict if dict-encoded)
        // IMPORTANT: preserve constraints from the original schema
        let modified_schema = if !actual_col_types.is_empty() {
            let mut ms = OnDemandSchema::new();
            for (col_idx, (col_name, _)) in schema_clone.columns.iter().enumerate() {
                ms.add_column(col_name, actual_col_types[col_idx]);
            }
            // Copy constraints from original schema
            ms.constraints = schema_clone.constraints.clone();
            ms
        } else {
            schema_clone.clone()
        };

        // Write V4 footer
        let footer_offset = writer.stream_position()?;
        let footer = V4Footer {
            schema: modified_schema,
            row_groups: rg_metas.clone(),
            zone_maps: all_zone_maps,
            col_offsets: all_rg_col_offsets,
        };
        writer.write_all(&footer.to_bytes())?;
        writer.flush()?;

        if self.durability == super::DurabilityLevel::Max {
            writer.get_ref().sync_all()?;
        }

        // Seek back to fix header
        {
            let mut header = self.header.write();
            header.version = FORMAT_VERSION_V4;
            header.row_count = active_count as u64;
            header.column_count = col_count as u32;
            header.footer_offset = footer_offset;
            header.row_group_count = rg_metas.len() as u32;
            header.schema_offset = 0;
            header.column_index_offset = 0;
            header.id_column_offset = 0;
        }
        self.cached_footer_offset
            .store(footer_offset, Ordering::Release);
        let header = self.header.read();
        let writer_inner = writer.get_mut();
        writer_inner.seek(SeekFrom::Start(0))?;
        writer_inner.write_all(&header.to_bytes())?;
        writer_inner.flush()?;

        // Ensure all data is on disk before the atomic rename
        if self.durability != super::DurabilityLevel::Fast {
            writer_inner.sync_all()?;
        }

        // Phase 3: Atomic rename .tmp → .apex
        // POSIX rename is atomic; on crash the original file remains intact.
        // On Windows, retry on transient failures from antivirus / Search Indexer / cloud sync
        // that may briefly hold a sharing lock on the destination file.
        drop(header);
        drop(writer);
        #[cfg(windows)]
        {
            let mut last_err = None;
            for attempt in 0u64..5 {
                match std::fs::rename(&tmp_path, &self.path) {
                    Ok(()) => {
                        last_err = None;
                        break;
                    }
                    Err(e) => {
                        last_err = Some(e);
                        if attempt < 4 {
                            std::thread::sleep(std::time::Duration::from_millis(
                                10 * (attempt + 1),
                            ));
                        }
                    }
                }
            }
            if let Some(e) = last_err {
                return Err(e);
            }
        }
        #[cfg(not(windows))]
        std::fs::rename(&tmp_path, &self.path)?;

        // Write column stats sidecar for O(1) aggregation fast path
        self.write_col_stats_sidecar(&schema_clone, &active_columns);

        // OPTIMIZATION: compute max_id BEFORE moving active_ids (avoids re-reading after write)
        let max_active_id = active_ids.iter().max().copied().unwrap_or(0);

        *self.column_index.write() = Vec::new();
        *self.ids.write() = active_ids;
        *self.columns.write() = active_columns;
        *self.nulls.write() = active_nulls;
        let del_len = (active_count + 7) / 8;
        *self.deleted.write() = vec![0u8; del_len];
        *self.id_to_idx.write() = None;
        self.mmap_cache.write().invalidate();
        self.invalidate_page_cache();

        self.active_count
            .store(active_count as u64, Ordering::SeqCst);
        // save_v4 physically removes deleted rows; persisted = active
        self.persisted_row_count
            .store(active_count as u64, Ordering::SeqCst);
        // Mark base as loaded — all data is now in memory after full rewrite
        self.v4_base_loaded.store(true, Ordering::SeqCst);
        let candidate = max_active_id + 1;
        let current = self.next_id.load(Ordering::SeqCst);
        if candidate > current {
            self.next_id.store(candidate, Ordering::SeqCst);
        }

        let file = open_for_sequential_read(&self.path)?;
        *self.file.write() = Some(file);

        if self.durability == super::DurabilityLevel::Fast {
            self.mark_main_sync_pending();
        } else {
            self.clear_main_sync_pending();
        }

        // On Linux and Windows, eagerly create the mmap so the next read avoids lazy-creation
        // overhead. On Linux this eliminates lazy mmap setup; on Windows this triggers the
        // prefault loop in MmapCache::get_or_create so subsequent queries avoid page faults.
        #[cfg(any(target_os = "linux", windows))]
        {
            let file_guard = self.file.read();
            if let Some(f) = file_guard.as_ref() {
                let _ = self.mmap_cache.write().get_or_create(f);
            }
        }

        Ok(())
    }

    /// Open a V4 file: read footer, then load all RG data into flat columns.
    /// Used by write operations (drop_column, etc.) that need full data in memory,
    /// and by tests. Production reads use mmap on-demand reading instead.
    pub fn open_v4_data(&self) -> io::Result<()> {
        let header = self.header.read();
        if header.footer_offset == 0 {
            return Err(err_data("V4 file has no footer"));
        }
        let footer_offset = header.footer_offset;
        drop(header);

        // Read footer from file
        let file_guard = self.file.read();
        let file = file_guard
            .as_ref()
            .ok_or_else(|| err_not_conn("File not open for V4 read"))?;
        let mut mmap = self.mmap_cache.write();

        // Read footer
        let file_len = std::fs::metadata(&self.path)?.len();
        let footer_byte_count = (file_len - footer_offset) as usize;
        let mut footer_bytes = vec![0u8; footer_byte_count];
        mmap.read_at(file, &mut footer_bytes, footer_offset)?;
        let footer = V4Footer::from_bytes(&footer_bytes)?;

        // Update schema from footer
        *self.schema.write() = footer.schema.clone();
        let col_count = footer.schema.column_count();

        // Compute total rows from RG metadata (header.row_count stores active count,
        // but RGs may contain deleted rows that are still physically present)
        let total_rows: usize = footer
            .row_groups
            .iter()
            .map(|rg| rg.row_count as usize)
            .sum();

        // Allocate flat columns
        let mut all_ids: Vec<u64> = Vec::with_capacity(total_rows);
        let mut all_columns: Vec<ColumnData> = (0..col_count)
            .map(|i| ColumnData::new(footer.schema.columns[i].1))
            .collect();
        let mut all_nulls: Vec<Vec<u8>> = vec![Vec::new(); col_count];
        let mut all_deleted: Vec<u8> = Vec::new(); // flat deletion bitmap

        // Read each Row Group as a byte buffer, parse sequentially
        let mut max_id_seen: u64 = 0;
        let mut total_deleted: u64 = 0;
        for rg_meta in &footer.row_groups {
            if rg_meta.row_count == 0 {
                continue;
            }
            let rg_rows = rg_meta.row_count as usize;
            let rg_size = rg_meta.data_size as usize;

            // Read entire RG into buffer
            let mut rg_buf = vec![0u8; rg_size];
            mmap.read_at(file, &mut rg_buf, rg_meta.offset)?;

            // Check compression flag at RG header byte 28, encoding version at byte 29
            let compress_flag = if rg_buf.len() >= 32 {
                rg_buf[28]
            } else {
                RG_COMPRESS_NONE
            };
            let encoding_version = if rg_buf.len() >= 32 { rg_buf[29] } else { 0 };

            // Get the body bytes (after 32-byte RG header), decompressing if needed
            let decompressed_buf = decompress_rg_body(compress_flag, &rg_buf[32..])?;
            let body: &[u8] = decompressed_buf.as_deref().unwrap_or(&rg_buf[32..]);
            let mut pos: usize = 0;

            // Parse IDs — OPTIMIZATION: bulk memcpy instead of per-element loop
            let ids_before = all_ids.len();
            let id_byte_len = rg_rows * 8;
            all_ids.resize(ids_before + rg_rows, 0);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    body[pos..].as_ptr(),
                    all_ids[ids_before..].as_mut_ptr() as *mut u8,
                    id_byte_len,
                );
            }
            if rg_meta.max_id > max_id_seen {
                max_id_seen = rg_meta.max_id;
            }
            pos += id_byte_len;

            // Read deletion vector and merge into flat bitmap
            let del_vec_len = (rg_rows + 7) / 8;
            let del_bytes = &body[pos..pos + del_vec_len];
            let needed_len = (ids_before + rg_rows + 7) / 8;
            if all_deleted.len() < needed_len {
                all_deleted.resize(needed_len, 0);
            }
            if ids_before % 8 == 0 {
                let dest_byte = ids_before / 8;
                let copy_len = del_vec_len.min(all_deleted.len() - dest_byte);
                all_deleted[dest_byte..dest_byte + copy_len]
                    .copy_from_slice(&del_bytes[..copy_len]);
            } else {
                for i in 0..rg_rows {
                    if (del_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                        let flat_idx = ids_before + i;
                        all_deleted[flat_idx / 8] |= 1 << (flat_idx % 8);
                    }
                }
            }
            total_deleted += rg_meta.deletion_count as u64;
            pos += del_vec_len;

            // Parse columns
            let null_bitmap_len = (rg_rows + 7) / 8;
            for col_idx in 0..col_count {
                // Read null bitmap
                let null_bytes = &body[pos..pos + null_bitmap_len];

                // Merge into flat nulls
                let flat_start = ids_before;
                let needed_len = (flat_start + rg_rows + 7) / 8;
                if all_nulls[col_idx].len() < needed_len {
                    all_nulls[col_idx].resize(needed_len, 0);
                }
                // OPTIMIZATION: bulk copy when flat_start is byte-aligned
                if flat_start % 8 == 0 {
                    let dest_byte = flat_start / 8;
                    let copy_len = null_bitmap_len.min(all_nulls[col_idx].len() - dest_byte);
                    all_nulls[col_idx][dest_byte..dest_byte + copy_len]
                        .copy_from_slice(&null_bytes[..copy_len]);
                } else {
                    for i in 0..rg_rows {
                        if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                            let flat_idx = flat_start + i;
                            all_nulls[col_idx][flat_idx / 8] |= 1 << (flat_idx % 8);
                        }
                    }
                }
                pos += null_bitmap_len;

                // Parse column data (encoding-aware for version 1, plain for version 0)
                let col_type = footer.schema.columns[col_idx].1;
                let (col_data, consumed) = if encoding_version >= 1 {
                    read_column_encoded(&body[pos..], col_type)?
                } else {
                    ColumnData::from_bytes_typed(&body[pos..], col_type)?
                };
                pos += consumed;

                // Append to flat column
                all_columns[col_idx].append(&col_data);
            }
        }

        drop(mmap);
        drop(file_guard);

        // Decode StringDict columns back to plain String for in-memory use.
        // Dict encoding is a disk-only optimization; push_string/extend_strings
        // only work on ColumnData::String, so we must normalize here.
        {
            let mut schema_w = self.schema.write();
            for col_idx in 0..all_columns.len() {
                if matches!(&all_columns[col_idx], ColumnData::StringDict { .. }) {
                    let col = std::mem::replace(
                        &mut all_columns[col_idx],
                        ColumnData::new(ColumnType::String),
                    );
                    all_columns[col_idx] = col.decode_string_dict();
                    // Update schema type from StringDict → String
                    if col_idx < schema_w.columns.len() {
                        schema_w.columns[col_idx].1 = ColumnType::String;
                    }
                }
            }
        }

        // OPTIMIZATION: compute next_id from tracked max before moving all_ids
        let next_id = if max_id_seen > 0 {
            max_id_seen + 1
        } else {
            all_ids
                .iter()
                .max()
                .map(|&id| id + 1)
                .unwrap_or(crate::storage::FIRST_ROW_ID)
        };

        // Store flat data
        *self.ids.write() = all_ids;
        *self.columns.write() = all_columns;
        *self.nulls.write() = all_nulls;

        // Use deletion vectors read from disk (not all-zeros)
        let deleted_len = (total_rows + 7) / 8;
        if all_deleted.len() < deleted_len {
            all_deleted.resize(deleted_len, 0);
        }
        *self.deleted.write() = all_deleted;

        self.next_id.store(next_id, Ordering::SeqCst);
        self.active_count
            .store(total_rows as u64 - total_deleted, Ordering::SeqCst);
        // Track actual on-disk row count (total rows in RGs, including deleted)
        self.persisted_row_count
            .store(total_rows as u64, Ordering::SeqCst);
        self.v4_base_loaded.store(true, Ordering::SeqCst);
        *self.id_to_idx.write() = None;

        Ok(())
    }

    /// Update only the V4 footer schema on disk (no data rewrite).
    /// Used for DDL operations (add/drop/rename column) when base data
    /// is not loaded into memory (mmap-only mode).
    pub fn update_v4_footer_schema(&self) -> io::Result<()> {
        let header = self.header.read();
        if header.version != FORMAT_VERSION_V4 || header.footer_offset == 0 {
            return Err(err_data("update_v4_footer_schema requires V4 format file"));
        }
        let footer_offset = header.footer_offset;
        drop(header);

        // Read existing footer from disk
        let file_len = std::fs::metadata(&self.path)?.len();
        let footer_bytes = {
            let file_guard = self.file.read();
            let file = file_guard
                .as_ref()
                .ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap = self.mmap_cache.write();
            let size = (file_len - footer_offset) as usize;
            let mut buf = vec![0u8; size];
            mmap.read_at(file, &mut buf, footer_offset)?;
            buf
        };
        let mut footer = V4Footer::from_bytes(&footer_bytes)?;

        // Update footer schema from current in-memory schema
        let schema = self.schema.read();
        footer.schema = schema.clone();
        drop(schema);

        // Release mmap before writing
        self.mmap_cache.write().invalidate();
        self.invalidate_page_cache();
        *self.file.write() = None;
        *self.write_file.write() = None;
        crate::query::ApexExecutor::invalidate_cache_for_path(&self.path);

        // Write updated footer at same offset (overwrite old footer)
        let mut file = OpenOptions::new().write(true).open(&self.path)?;
        let new_footer_bytes = footer.to_bytes();
        file.seek(SeekFrom::Start(footer_offset))?;
        file.write_all(&new_footer_bytes)?;
        // Write footer size + magic trailer
        file.write_all(&(new_footer_bytes.len() as u64).to_le_bytes())?;
        file.write_all(b"APXFOOT\0")?;
        file.flush()?;

        // Truncate file to remove any trailing data from old (possibly larger) footer
        let new_file_len = footer_offset + new_footer_bytes.len() as u64 + 16;
        file.set_len(new_file_len)?;

        // Update header column count (both in-memory and on-disk)
        {
            let mut header = self.header.write();
            header.column_count = footer.schema.column_count() as u32;
            // Write updated header to disk
            let mut hfile = OpenOptions::new().write(true).open(&self.path)?;
            hfile.write_all(&header.to_bytes())?;
            hfile.flush()?;
        }

        // Reopen file handle
        drop(file);
        let file = open_for_sequential_read(&self.path)?;
        *self.file.write() = Some(file);

        if self.durability == super::DurabilityLevel::Fast {
            self.mark_main_sync_pending();
        } else {
            self.sync_main_file_data()?;
            self.clear_main_sync_pending();
        }

        // Update the in-memory v4_footer cache so that subsequent reads
        // (e.g. to_arrow_batch_mmap via get_or_load_footer) see the new
        // column names immediately, even before the backend is evicted
        // from insert_cache.
        *self.v4_footer.write() = Some(footer);

        Ok(())
    }

    /// Update only the deletion vectors in existing Row Groups on disk.
    /// O(num_RGs) random writes instead of O(all_data) full rewrite.
    /// Also updates the footer's per-RG deletion_count and the header's row_count.
    fn save_deletion_vectors(&self) -> io::Result<()> {
        let header = self.header.read();
        if header.version != FORMAT_VERSION_V4 || header.footer_offset == 0 {
            return Err(err_data("save_deletion_vectors requires V4 format file"));
        }
        let footer_offset = header.footer_offset;
        drop(header);

        // Read existing footer
        let file_len = std::fs::metadata(&self.path)?.len();
        let footer_bytes = {
            let file_guard = self.file.read();
            let file = file_guard
                .as_ref()
                .ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap = self.mmap_cache.write();
            let size = (file_len - footer_offset) as usize;
            let mut buf = vec![0u8; size];
            mmap.read_at(file, &mut buf, footer_offset)?;
            buf
        };
        let mut footer = V4Footer::from_bytes(&footer_bytes)?;

        // Check if any RG is compressed — if so, we cannot do in-place deletion
        // vector updates (the deletion vector is inside the compressed body).
        // Fall back to full rewrite via save_v4().
        {
            let file_guard = self.file.read();
            let file = file_guard
                .as_ref()
                .ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap = self.mmap_cache.write();
            let mmap_ref = mmap.get_or_create(file)?;
            for rg_meta in &footer.row_groups {
                if rg_meta.row_count == 0 {
                    continue;
                }
                let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                if rg_end <= mmap_ref.len() {
                    let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
                    if rg_bytes.len() >= 32 && rg_bytes[28] != RG_COMPRESS_NONE {
                        // Compressed RG detected — must do full rewrite
                        drop(mmap);
                        drop(file_guard);
                        return self.save_v4();
                    }
                }
            }
        }

        // Release mmap before writing
        self.mmap_cache.write().invalidate();
        self.invalidate_page_cache();
        *self.file.write() = None;
        *self.write_file.write() = None;
        crate::query::ApexExecutor::invalidate_cache_for_path(&self.path);

        let deleted = self.deleted.read();
        let mut file = OpenOptions::new().write(true).open(&self.path)?;

        // For each RG, write the updated deletion vector at its known offset
        let mut flat_row_start: usize = 0;
        let mut total_active: u64 = 0;
        for rg_meta in footer.row_groups.iter_mut() {
            let rg_rows = rg_meta.row_count as usize;
            if rg_rows == 0 {
                continue;
            }

            // Deletion vector starts after RG header (32 bytes) + IDs (rg_rows * 8)
            let del_vec_offset = rg_meta.offset + 32 + (rg_rows as u64 * 8);
            let del_vec_len = (rg_rows + 7) / 8;

            // Extract this RG's slice from the flat deleted bitmap
            let rg_del_vec =
                Self::slice_null_bitmap(&deleted, flat_row_start, flat_row_start + rg_rows);

            // Count deleted rows in this RG
            let mut del_count: u32 = 0;
            for i in 0..rg_rows {
                if (rg_del_vec[i / 8] >> (i % 8)) & 1 == 1 {
                    del_count += 1;
                }
            }
            rg_meta.deletion_count = del_count;
            total_active += (rg_rows as u32 - del_count) as u64;

            // Write deletion vector to disk
            file.seek(SeekFrom::Start(del_vec_offset))?;
            file.write_all(&rg_del_vec[..del_vec_len])?;

            flat_row_start += rg_rows;
        }
        drop(deleted);

        // Rewrite footer with updated deletion_counts
        file.seek(SeekFrom::Start(footer_offset))?;
        let new_footer_bytes = footer.to_bytes();
        file.write_all(&new_footer_bytes)?;
        // Truncate in case new footer is shorter (shouldn't happen in deletion-only paths).
        // Skipped on Windows: set_len fails with ERROR_USER_MAPPED_FILE (os error 1224) when
        // any mmap is open. Footer size never changes in deletion-only operations, so this
        // is a safe no-op to skip.
        let new_end = footer_offset + new_footer_bytes.len() as u64;
        #[cfg(not(target_os = "windows"))]
        file.set_len(new_end)?;
        file.flush()?;

        // Update header: row_count = active rows (matches save_v4 convention)
        {
            let mut header = self.header.write();
            header.row_count = total_active;
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&header.to_bytes())?;
        }
        file.flush()?;
        drop(file);

        // Reopen file handle
        *self.file.write() = Some(open_for_sequential_read(&self.path)?);

        if self.durability == super::DurabilityLevel::Fast {
            self.mark_main_sync_pending();
        } else {
            self.sync_main_file_data()?;
            self.clear_main_sync_pending();
        }

        Ok(())
    }

    /// Save after deletion-only operations (no new rows inserted).
    ///
    /// This bypasses the broken logic in `save()` which conflates "IDs loaded from disk
    /// for deletion lookup" with "truly new rows needing append". After a deletion-only
    /// operation on a V4 mmap-only file, `ids` may be populated (by `ensure_ids_loaded_v4`)
    /// even though no new rows exist, causing `save()` to fall through to a full `save_v4()`
    /// rewrite. This method calls `save_deletion_vectors()` directly: O(num_RGs) random
    /// writes instead of a full O(all_data) rewrite.
    ///
    /// For compressed files (where deletion vectors are embedded in the compressed body),
    /// falls back to loading all data + `save_v4()`.
    pub fn save_delete_only(&self) -> io::Result<()> {
        let header = self.header.read();
        let is_v4 = header.version == FORMAT_VERSION_V4 && header.footer_offset > 0;
        drop(header);

        if !is_v4 {
            // Non-V4: fall through to normal save
            return self.save();
        }

        // Check if any RG is compressed — compressed RGs embed the deletion vector inside
        // the compressed body, so we cannot update it in-place.
        // NOTE: get_or_load_footer() acquires mmap_cache.write() internally; call it BEFORE
        // acquiring mmap_cache ourselves to avoid deadlock.
        let footer_opt = self.get_or_load_footer()?;
        let has_compression = if let Some(footer) = footer_opt {
            let file_guard = self.file.read();
            if let Some(file) = file_guard.as_ref() {
                let mut mmap = self.mmap_cache.write();
                if let Ok(mmap_ref) = mmap.get_or_create(file) {
                    footer.row_groups.iter().any(|rg| {
                        if rg.row_count == 0 {
                            return false;
                        }
                        let rg_start = rg.offset as usize;
                        rg_start + 32 <= mmap_ref.len()
                            && mmap_ref[rg_start + 28] != RG_COMPRESS_NONE
                    })
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        if has_compression {
            // Compressed: must load all columns before full rewrite
            self.load_all_columns_into_memory()?;
            self.pending_rows.store(0, Ordering::SeqCst);
            let result = self.save_v4();
            if result.is_ok() {
                let _ = self.clear_delta_store();
                self.checkpoint_wal();
            }
            result
        } else {
            // Uncompressed: fast in-place deletion vector update (O(num_RGs) writes)
            if self.has_pending_deltas() {
                let _ = self.save_delta_store();
            }
            self.save_deletion_vectors()
        }
    }

    /// Single-pass scan + mark deleted + save for numeric predicates.
    /// Returns `Some(newly_deleted_count)` on success, `None` if fast path unavailable
    /// (non-V4, compressed RGs, or column not found).
    ///
    /// Unlike delete_batch() + save_delete_only(), this never builds an id_to_idx HashMap.
    /// Instead it works directly with flat row indices derived from the mmap scan,
    /// merging new deletions with the on-disk deletion vector in one pass.
    pub fn delete_where_numeric_range_inplace(
        &self,
        col_name: &str,
        low: f64,
        high: f64,
    ) -> io::Result<Option<i64>> {
        let header = self.header.read();
        let is_v4 = header.version == FORMAT_VERSION_V4 && header.footer_offset > 0;
        let footer_offset = header.footer_offset;
        drop(header);
        if !is_v4 {
            return Ok(None);
        }

        // Load footer (acquires mmap_cache internally — must be before we take our own lock)
        let footer_opt = self.get_or_load_footer()?;
        let mut footer = match footer_opt {
            Some(f) => f,
            None => return Ok(None),
        };

        let schema = &footer.schema;
        let col_idx = match schema.get_index(col_name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        let is_int = matches!(
            col_type,
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
        );
        let is_float = matches!(col_type, ColumnType::Float64 | ColumnType::Float32);
        if !is_int && !is_float {
            return Ok(None);
        }
        let low_i = low.ceil() as i64;
        let high_i = high.floor() as i64;

        // Per-RG: (del_vec_offset, new_del_bytes) for RGs that had new deletions
        struct RgWrite {
            del_vec_offset: u64,
            del_bytes: Vec<u8>,
            new_del_count: u32,
        }
        // Zone map updates to apply after the scan (can't mutate footer while iterating it)
        struct ZmUpdate {
            rg_i: usize,
            zm_pos: usize,
            new_min: i64,
            new_max: i64,
        }
        let mut rg_writes: Vec<(usize, RgWrite)> = Vec::new(); // (rg_i, write)
        let mut zm_updates: Vec<ZmUpdate> = Vec::new();
        let mut newly_deleted: i64 = 0;
        let mut new_total_active: u64 = 0;
        let mut can_fast_delete = true;

        {
            let file_guard = self.file.read();
            let file = match file_guard.as_ref() {
                Some(f) => f,
                None => return Ok(None),
            };
            let mut mmap = self.mmap_cache.write();
            let mmap_ref = match mmap.get_or_create(file) {
                Ok(m) => m,
                Err(_) => return Ok(None),
            };

            'rg_loop: for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
                let rg_rows = rg_meta.row_count as usize;
                if rg_rows == 0 {
                    continue;
                }
                let del_vec_len = (rg_rows + 7) / 8;
                let null_bitmap_len = del_vec_len;

                // Skip fully-deleted RGs — nothing more can be deleted
                if rg_meta.deletion_count as usize == rg_rows {
                    // all rows already deleted, active_rows = 0
                    continue;
                }

                // Zone map pruning
                if rg_i < footer.zone_maps.len() {
                    if let Some(zm) = footer.zone_maps[rg_i]
                        .iter()
                        .find(|z| z.col_idx as usize == col_idx)
                    {
                        let skip = if zm.is_float {
                            !zm.may_overlap_float_range(low, high)
                        } else {
                            !zm.may_overlap_int_range(low_i, high_i)
                        };
                        if skip {
                            new_total_active += rg_meta.active_rows() as u64;
                            continue;
                        }
                    }
                }

                let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                if rg_end > mmap_ref.len() {
                    can_fast_delete = false;
                    break 'rg_loop;
                }

                // Compressed RGs cannot be in-place updated — fall back
                if mmap_ref.len() >= rg_meta.offset as usize + 29
                    && mmap_ref[rg_meta.offset as usize + 28] != RG_COMPRESS_NONE
                {
                    can_fast_delete = false;
                    break 'rg_loop;
                }

                let body = &mmap_ref[(rg_meta.offset as usize + 32)..rg_end];
                let id_section = rg_rows * 8;
                if id_section + del_vec_len > body.len() {
                    can_fast_delete = false;
                    break 'rg_loop;
                }

                // Require RCIX col_offsets — without them we can't find the target column
                let rcix_ok =
                    rg_i < footer.col_offsets.len() && col_idx < footer.col_offsets[rg_i].len();
                if !rcix_ok {
                    can_fast_delete = false;
                    break 'rg_loop;
                }

                let encoding_version = if mmap_ref.len() > rg_meta.offset as usize + 29 {
                    mmap_ref[rg_meta.offset as usize + 29]
                } else {
                    0
                };

                let col_body_off = footer.col_offsets[rg_i][col_idx] as usize;
                if col_body_off + null_bitmap_len > body.len() {
                    can_fast_delete = false;
                    break 'rg_loop;
                }
                let null_bytes = &body[col_body_off..col_body_off + null_bitmap_len];
                let data_start = col_body_off + null_bitmap_len;
                let col_bytes = &body[data_start..];
                let enc_offset = if encoding_version >= 1 { 1 } else { 0 };
                let encoding = if encoding_version >= 1 && !col_bytes.is_empty() {
                    col_bytes[0]
                } else {
                    COL_ENCODING_PLAIN
                };

                // Handle PLAIN and RLE encodings — fall back for others
                if col_bytes.len() <= enc_offset + 8 {
                    can_fast_delete = false;
                    break 'rg_loop;
                }

                let del_bytes_src = &body[id_section..id_section + del_vec_len];
                let mut del_bytes = del_bytes_src.to_vec();
                let has_deletes = rg_meta.deletion_count > 0;
                let mut rg_newly_deleted: i64 = 0;

                // Decode column data based on encoding
                let payload: Vec<u8> = if encoding == COL_ENCODING_RLE {
                    // RLE format: [count:u64][num_runs:u64][(value:i64, run_len:u32)...]
                    let rle_data = &col_bytes[enc_offset..];
                    if rle_data.len() < 16 {
                        can_fast_delete = false;
                        break 'rg_loop;
                    }
                    let count = u64::from_le_bytes(rle_data[0..8].try_into().unwrap()) as usize;
                    let num_runs = u64::from_le_bytes(rle_data[8..16].try_into().unwrap()) as usize;
                    if rle_data.len() < 16 + num_runs * 12 {
                        can_fast_delete = false;
                        break 'rg_loop;
                    }
                    // Decode RLE to the same payload shape as PLAIN: [count:u64][values...].
                    // The delete scanner below expects the first 8 bytes to be the value count.
                    let mut decoded = Vec::with_capacity(8 + count * 8);
                    decoded.extend_from_slice(&(count as u64).to_le_bytes());
                    let mut pos = 16;
                    for _ in 0..num_runs {
                        let val = i64::from_le_bytes(rle_data[pos..pos + 8].try_into().unwrap());
                        pos += 8;
                        let run_len =
                            u32::from_le_bytes(rle_data[pos..pos + 4].try_into().unwrap()) as usize;
                        pos += 4;
                        for _ in 0..run_len {
                            decoded.extend_from_slice(&val.to_le_bytes());
                        }
                    }
                    decoded
                } else if encoding == COL_ENCODING_PLAIN {
                    col_bytes[enc_offset..].to_vec()
                } else {
                    can_fast_delete = false;
                    break 'rg_loop;
                };

                let count = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                let n = count
                    .min(rg_rows)
                    .min((payload.len().saturating_sub(8)) / 8);
                let vals_raw = &payload[8..];
                if is_int {
                    for i in 0..n {
                        if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                            continue;
                        }
                        if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                            continue;
                        }
                        let v =
                            i64::from_le_bytes(vals_raw[i * 8..(i + 1) * 8].try_into().unwrap());
                        if v >= low_i && v <= high_i {
                            del_bytes[i / 8] |= 1 << (i % 8);
                            rg_newly_deleted += 1;
                        }
                    }
                } else {
                    for i in 0..n {
                        if has_deletes && (del_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                            continue;
                        }
                        if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                            continue;
                        }
                        let v =
                            f64::from_le_bytes(vals_raw[i * 8..(i + 1) * 8].try_into().unwrap());
                        if v >= low && v <= high {
                            del_bytes[i / 8] |= 1 << (i % 8);
                            rg_newly_deleted += 1;
                        }
                    }
                }

                newly_deleted += rg_newly_deleted;
                let new_del_count = if rg_newly_deleted > 0 {
                    del_bytes.iter().map(|b| b.count_ones()).sum::<u32>()
                } else {
                    rg_meta.deletion_count
                };
                new_total_active += (rg_rows as u32 - new_del_count) as u64;

                // Only enqueue a write if this RG actually had new deletions
                if rg_newly_deleted > 0 {
                    let del_vec_offset = rg_meta.offset + 32 + (rg_rows as u64 * 8);
                    rg_writes.push((
                        rg_i,
                        RgWrite {
                            del_vec_offset,
                            del_bytes,
                            new_del_count,
                        },
                    ));

                    // Zone map staleness fix: if deleted values touched the zone map boundary,
                    // rescan remaining active rows so future queries can prune this RG.
                    if rg_i < footer.zone_maps.len() {
                        if let Some(zm_pos) = footer.zone_maps[rg_i]
                            .iter()
                            .position(|z| z.col_idx as usize == col_idx)
                        {
                            let zm = &footer.zone_maps[rg_i][zm_pos];
                            let boundary_hit = if zm.is_float {
                                low <= f64::from_bits(zm.max_bits as u64)
                                    && high >= f64::from_bits(zm.min_bits as u64)
                            } else {
                                low_i <= zm.max_bits && high_i >= zm.min_bits
                            };
                            if boundary_hit {
                                let updated_del = &rg_writes.last().unwrap().1.del_bytes;
                                let mut new_min = i64::MAX;
                                let mut new_max = i64::MIN;
                                let n_scan = count.min(rg_rows).min(vals_raw.len() / 8);
                                if is_int {
                                    for i in 0..n_scan {
                                        if (updated_del[i / 8] >> (i % 8)) & 1 == 1 {
                                            continue;
                                        }
                                        if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                                            continue;
                                        }
                                        let v = i64::from_le_bytes(
                                            vals_raw[i * 8..(i + 1) * 8].try_into().unwrap(),
                                        );
                                        if v < new_min {
                                            new_min = v;
                                        }
                                        if v > new_max {
                                            new_max = v;
                                        }
                                    }
                                } else {
                                    for i in 0..n_scan {
                                        if (updated_del[i / 8] >> (i % 8)) & 1 == 1 {
                                            continue;
                                        }
                                        if (null_bytes[i / 8] >> (i % 8)) & 1 == 1 {
                                            continue;
                                        }
                                        let v = f64::from_le_bytes(
                                            vals_raw[i * 8..(i + 1) * 8].try_into().unwrap(),
                                        );
                                        let v_bits = v.to_bits() as i64;
                                        if v_bits < new_min {
                                            new_min = v_bits;
                                        }
                                        if v_bits > new_max {
                                            new_max = v_bits;
                                        }
                                    }
                                }
                                // new_min > new_max means all rows deleted — use impossible range
                                zm_updates.push(ZmUpdate {
                                    rg_i,
                                    zm_pos,
                                    new_min: if new_min <= new_max { new_min } else { 1 },
                                    new_max: if new_min <= new_max { new_max } else { 0 },
                                });
                            }
                        }
                    }
                }
            }
        }

        if !can_fast_delete {
            return Ok(None);
        }
        if newly_deleted == 0 {
            return Ok(Some(0));
        }

        // Apply zone map updates (stale boundary fix) so future scans can prune these RGs
        for zu in zm_updates {
            if zu.rg_i < footer.zone_maps.len() && zu.zm_pos < footer.zone_maps[zu.rg_i].len() {
                footer.zone_maps[zu.rg_i][zu.zm_pos].min_bits = zu.new_min;
                footer.zone_maps[zu.rg_i][zu.zm_pos].max_bits = zu.new_max;
            }
        }

        // Reuse the already-parsed footer (no re-read from disk needed)
        let mut footer_mut = footer;

        // On Unix: defer all disk writes to global pending map (zero I/O at DELETE time).
        // The next open_with_durability / open_for_read_with_file call applies pending state.
        #[cfg(unix)]
        {
            // Clear user-space row cache only (no mmap invalidation needed)
            self.invalidate_page_cache();

            // Update footer deletion counts
            for (rg_i, wr) in &rg_writes {
                footer_mut.row_groups[*rg_i].deletion_count = wr.new_del_count;
            }

            // Serialize pending state to global map (no file I/O)
            let footer_bytes = footer_mut.to_bytes();
            let mut buf = Vec::with_capacity(8 + rg_writes.len() * 20 + 12 + footer_bytes.len());
            buf.extend_from_slice(b"APXP");
            buf.extend_from_slice(&(rg_writes.len() as u32).to_le_bytes());
            for (rg_i, wr) in &rg_writes {
                let del_vec_len = (footer_mut.row_groups[*rg_i].row_count as usize + 7) / 8;
                buf.extend_from_slice(&(*rg_i as u32).to_le_bytes());
                buf.extend_from_slice(&wr.del_vec_offset.to_le_bytes());
                buf.extend_from_slice(&(del_vec_len as u32).to_le_bytes());
                buf.extend_from_slice(&wr.del_bytes[..del_vec_len]);
            }
            buf.extend_from_slice(&footer_offset.to_le_bytes());
            buf.extend_from_slice(&(footer_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(&footer_bytes);

            // Store in global map — zero file I/O at DELETE time
            global_pending_deletes()
                .write()
                .unwrap()
                .as_mut()
                .map(|m| m.insert(self.path.clone(), buf));

            // Update header in memory only (not on disk)
            self.header.write().row_count = new_total_active;
        }
        #[cfg(not(unix))]
        {
            // Non-Unix: full mmap invalidation + seek-based writes.
            // On Windows, invalidate engine cache first to release mmap handles
            // that would otherwise block file writes (OS error 1224).
            self.mmap_cache.write().invalidate();
            self.invalidate_page_cache();
            *self.file.write() = None;
            *self.write_file.write() = None;
            #[cfg(windows)]
            super::engine::engine().invalidate(&self.path);
            crate::query::ApexExecutor::invalidate_cache_for_path(&self.path);

            let mut file_mut = OpenOptions::new().read(true).write(true).open(&self.path)?;
            for (rg_i, wr) in &rg_writes {
                file_mut.seek(SeekFrom::Start(wr.del_vec_offset))?;
                let rg_rows = footer_mut.row_groups[*rg_i].row_count as usize;
                let del_vec_len = (rg_rows + 7) / 8;
                file_mut.write_all(&wr.del_bytes[..del_vec_len])?;
                footer_mut.row_groups[*rg_i].deletion_count = wr.new_del_count;
            }
            file_mut.seek(SeekFrom::Start(footer_offset))?;
            let new_footer_bytes = footer_mut.to_bytes();
            file_mut.write_all(&new_footer_bytes)?;
            // Footer size is identical for deletion-only ops; skip set_len on Windows
            // to avoid an unnecessary syscall (deletion counts change but entry count doesn't).
            #[cfg(not(windows))]
            {
                let new_end = footer_offset + new_footer_bytes.len() as u64;
                file_mut.set_len(new_end)?;
            }
            {
                let mut hdr = self.header.write();
                hdr.row_count = new_total_active;
                file_mut.seek(SeekFrom::Start(0))?;
                file_mut.write_all(&hdr.to_bytes())?;
            }
            file_mut.flush()?;
            *self.file.write() = Some(file_mut);
        }

        Ok(Some(newly_deleted))
    }

    /// Delete rows by their IDs directly from mmap — bypasses the id_to_idx HashMap.
    /// Reads each RG's ID section via mmap and uses binary search to locate target rows.
    /// Only RGs with new deletions are written back.
    /// Returns `Some(newly_deleted)` on success, `None` if fast path unavailable.
    pub fn delete_ids_inplace_v4(&self, ids: &[u64]) -> io::Result<Option<i64>> {
        if ids.is_empty() {
            return Ok(Some(0));
        }

        let header = self.header.read();
        let is_v4 = header.version == FORMAT_VERSION_V4 && header.footer_offset > 0;
        let footer_offset = header.footer_offset;
        drop(header);
        if !is_v4 {
            return Ok(None);
        }

        let footer_opt = self.get_or_load_footer()?;
        let mut footer = match footer_opt {
            Some(f) => f,
            None => return Ok(None),
        };

        let mut sorted_ids = ids.to_vec();
        sorted_ids.sort_unstable();

        struct RgWrite {
            del_vec_offset: u64,
            del_bytes: Vec<u8>,
            new_del_count: u32,
        }
        let mut rg_writes: Vec<(usize, RgWrite)> = Vec::new();
        let mut newly_deleted: i64 = 0;
        let mut new_total_active: u64 = 0;

        {
            let file_guard = self.file.read();
            let file = match file_guard.as_ref() {
                Some(f) => f,
                None => return Ok(None),
            };
            let mut mmap = self.mmap_cache.write();
            let mmap_ref = match mmap.get_or_create(file) {
                Ok(m) => m,
                Err(_) => return Ok(None),
            };

            for (rg_i, rg_meta) in footer.row_groups.iter().enumerate() {
                let rg_rows = rg_meta.row_count as usize;
                if rg_rows == 0 {
                    continue;
                }
                if rg_meta.deletion_count as usize == rg_rows {
                    continue;
                }

                let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                if rg_end > mmap_ref.len() {
                    return Ok(None);
                }
                if mmap_ref.len() >= rg_meta.offset as usize + 29
                    && mmap_ref[rg_meta.offset as usize + 28] != RG_COMPRESS_NONE
                {
                    return Ok(None);
                }

                let body_start = rg_meta.offset as usize + 32;
                let ids_size = rg_rows * 8;
                let del_vec_len = (rg_rows + 7) / 8;
                if body_start + ids_size + del_vec_len > rg_end {
                    return Ok(None);
                }

                let rg_ids_cow =
                    bytes_as_u64_slice(&mmap_ref[body_start..body_start + ids_size], rg_rows);
                let rg_ids: &[u64] = &rg_ids_cow;

                // Quick range check (IDs are monotonically increasing within each RG)
                let lo = sorted_ids.partition_point(|&x| x < rg_ids[0]);
                let hi = sorted_ids.partition_point(|&x| x <= rg_ids[rg_rows - 1]);
                if lo >= hi {
                    new_total_active += rg_meta.active_rows() as u64;
                    continue;
                }

                let del_bytes_src =
                    &mmap_ref[body_start + ids_size..body_start + ids_size + del_vec_len];
                let mut del_bytes = del_bytes_src.to_vec();
                let mut rg_newly_deleted: i64 = 0;

                for &target_id in &sorted_ids[lo..hi] {
                    if let Ok(pos) = rg_ids.binary_search(&target_id) {
                        if (del_bytes[pos / 8] >> (pos % 8)) & 1 == 0 {
                            del_bytes[pos / 8] |= 1 << (pos % 8);
                            rg_newly_deleted += 1;
                        }
                    }
                }

                newly_deleted += rg_newly_deleted;
                let new_del_count = if rg_newly_deleted > 0 {
                    del_bytes.iter().map(|b| b.count_ones()).sum::<u32>()
                } else {
                    rg_meta.deletion_count
                };
                new_total_active += (rg_rows as u32 - new_del_count) as u64;

                if rg_newly_deleted > 0 {
                    let del_vec_offset = rg_meta.offset + 32 + ids_size as u64;
                    rg_writes.push((
                        rg_i,
                        RgWrite {
                            del_vec_offset,
                            del_bytes,
                            new_del_count,
                        },
                    ));
                }
            }
        }

        if newly_deleted == 0 {
            return Ok(Some(0));
        }

        let mut footer_mut = footer;
        self.mmap_cache.write().invalidate();
        self.invalidate_page_cache();
        *self.file.write() = None;
        *self.write_file.write() = None;
        #[cfg(windows)]
        super::engine::engine().invalidate(&self.path);
        crate::query::ApexExecutor::invalidate_cache_for_path(&self.path);

        let mut file = OpenOptions::new().write(true).open(&self.path)?;

        for (rg_i, wr) in &rg_writes {
            file.seek(SeekFrom::Start(wr.del_vec_offset))?;
            let del_vec_len = (footer_mut.row_groups[*rg_i].row_count as usize + 7) / 8;
            file.write_all(&wr.del_bytes[..del_vec_len])?;
            footer_mut.row_groups[*rg_i].deletion_count = wr.new_del_count;
        }

        file.seek(SeekFrom::Start(footer_offset))?;
        let new_footer_bytes = footer_mut.to_bytes();
        file.write_all(&new_footer_bytes)?;
        // Footer size is identical (deletion counts only); skip set_len on Windows.
        let new_end = footer_offset + new_footer_bytes.len() as u64;
        #[cfg(not(target_os = "windows"))]
        file.set_len(new_end)?;
        file.flush()?;

        {
            let mut hdr = self.header.write();
            hdr.row_count = new_total_active;
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&hdr.to_bytes())?
        }
        file.flush()?;
        drop(file);

        *self.file.write() = Some(open_for_sequential_read(&self.path)?);
        Ok(Some(newly_deleted))
    }

    /// Write a new Row Group to disk without modifying in-memory state.
    /// Called by save() when rows are already in memory and only need persisting.
    /// Also called by append_row_group() which additionally updates memory.
    fn write_row_group_to_disk(
        &self,
        new_ids: &[u64],
        new_columns: &[ColumnData],
        new_nulls: &[Vec<u8>],
    ) -> io::Result<()> {
        let header = self.header.read();
        if header.version != FORMAT_VERSION_V4 || header.footer_offset == 0 {
            return Err(err_data("write_row_group_to_disk requires V4 format file"));
        }
        let footer_offset = header.footer_offset;
        drop(header);

        // Read existing footer
        let file_len = std::fs::metadata(&self.path)?.len();
        let footer_bytes = {
            let file_guard = self.file.read();
            let file = file_guard
                .as_ref()
                .ok_or_else(|| err_not_conn("File not open"))?;
            let mut mmap = self.mmap_cache.write();
            let size = (file_len - footer_offset) as usize;
            let mut buf = vec![0u8; size];
            mmap.read_at(file, &mut buf, footer_offset)?;
            buf
        };
        let mut footer = V4Footer::from_bytes(&footer_bytes)?;

        // Schema evolution: merge any new columns from in-memory schema into footer
        {
            let mem_schema = self.schema.read();
            for (name, ct) in &mem_schema.columns {
                if footer.schema.get_index(name).is_none() {
                    footer.schema.add_column(name, *ct);
                }
            }
        }
        let col_count = footer.schema.column_count();

        // Release mmap before writing
        self.mmap_cache.write().invalidate();
        self.invalidate_page_cache();
        *self.file.write() = None;
        *self.write_file.write() = None;
        crate::query::ApexExecutor::invalidate_cache_for_path(&self.path);
        self.delete_col_stats_sidecar();

        // Open file for append — seek to old footer position (overwrite it)
        let mut file = OpenOptions::new().write(true).open(&self.path)?;
        file.seek(SeekFrom::Start(footer_offset))?;
        let mut writer = BufWriter::with_capacity(64 * 1024, file);

        let rg_rows = new_ids.len();
        let rg_offset = footer_offset;
        let min_id = new_ids.iter().copied().min().unwrap_or(0);
        let max_id = new_ids.iter().copied().max().unwrap_or(0);

        // Serialize RG body to buffer (IDs + deletion vector + columns)
        let null_bitmap_len = (rg_rows + 7) / 8;
        let (delete_bitmap, deletion_count) = {
            let deleted = self.deleted.read();
            let mut bitmap = vec![0u8; null_bitmap_len];
            let copy_len = deleted.len().min(null_bitmap_len);
            if copy_len > 0 {
                bitmap[..copy_len].copy_from_slice(&deleted[..copy_len]);
            }
            let count = (0..rg_rows)
                .filter(|row_idx| {
                    let byte_idx = row_idx / 8;
                    let bit_idx = row_idx % 8;
                    byte_idx < bitmap.len() && (bitmap[byte_idx] >> bit_idx) & 1 == 1
                })
                .count() as u32;
            (bitmap, count)
        };
        let mut body_buf: Vec<u8> = Vec::with_capacity(rg_rows * 8 + rg_rows * col_count);
        {
            let mut body_writer = std::io::Cursor::new(&mut body_buf);

            // IDs
            for &id in new_ids {
                body_writer.write_all(&id.to_le_bytes())?;
            }

            // Deletion vector for rows that were inserted and deleted before flush.
            body_writer.write_all(&delete_bitmap)?;

            // Columns
            let mut new_rg_col_offsets: Vec<u32> = Vec::with_capacity(col_count);
            for col_idx in 0..col_count {
                // Record body offset of this column's null bitmap for RCIX
                new_rg_col_offsets.push(body_writer.position() as u32);

                // Null bitmap
                let col_nulls = new_nulls.get(col_idx).map(|v| v.as_slice()).unwrap_or(&[]);
                let padded = if col_nulls.len() < null_bitmap_len {
                    let mut v = vec![0u8; null_bitmap_len];
                    let copy = col_nulls.len().min(null_bitmap_len);
                    v[..copy].copy_from_slice(&col_nulls[..copy]);
                    v
                } else {
                    col_nulls[..null_bitmap_len].to_vec()
                };
                body_writer.write_all(&padded)?;

                // Column data — dict-encode if footer schema expects StringDict
                if col_idx < new_columns.len() {
                    let col = &new_columns[col_idx];
                    let col_type = if col_idx < footer.schema.columns.len() {
                        footer.schema.columns[col_idx].1
                    } else {
                        ColumnType::Int64
                    };
                    if col_type == ColumnType::StringDict
                        && matches!(col, ColumnData::String { .. })
                    {
                        if let Some(dict) = col.to_dict_encoded() {
                            write_column_encoded(&dict, col_type, &mut body_writer)?;
                        } else {
                            write_column_encoded(col, col_type, &mut body_writer)?;
                        }
                    } else {
                        write_column_encoded(col, col_type, &mut body_writer)?;
                    }
                }
            }
            footer.col_offsets.push(new_rg_col_offsets);
        }

        // Compress body using configured compression algorithm
        let (compress_flag, disk_body) = compress_rg_body(body_buf, self.compression());

        // Write RG header (32 bytes) — byte 28 = compression flag
        writer.write_all(MAGIC_ROW_GROUP)?;
        writer.write_all(&(rg_rows as u32).to_le_bytes())?;
        writer.write_all(&(col_count as u32).to_le_bytes())?;
        writer.write_all(&min_id.to_le_bytes())?;
        writer.write_all(&max_id.to_le_bytes())?;
        writer.write_all(&[compress_flag, 1, 0, 0])?; // encoding_version=1: per-column encoding prefix

        // RG body (possibly compressed)
        writer.write_all(&disk_body)?;

        let rg_end = writer.stream_position()?;

        // Update footer with new RG
        footer.row_groups.push(RowGroupMeta {
            offset: rg_offset,
            data_size: rg_end - rg_offset,
            row_count: rg_rows as u32,
            min_id,
            max_id,
            deletion_count,
        });
        let active_rows_after: u64 = footer
            .row_groups
            .iter()
            .map(|rg| (rg.row_count as u64).saturating_sub(rg.deletion_count as u64))
            .sum();

        // Write updated footer + trailer (footer_size + magic)
        let new_footer_offset = rg_end;
        let footer_bytes = footer.to_bytes();
        writer.write_all(&footer_bytes)?;
        writer.write_all(&(footer_bytes.len() as u64).to_le_bytes())?;
        writer.write_all(MAGIC_V4_FOOTER)?;
        writer.flush()?;

        // Fix header
        let new_persisted = self.persisted_row_count.load(Ordering::SeqCst) + rg_rows as u64;
        let writer_inner = writer.get_mut();
        {
            let mut header = self.header.write();
            header.row_count = active_rows_after;
            header.footer_offset = new_footer_offset;
            header.row_group_count = footer.row_groups.len() as u32;
        }
        self.cached_footer_offset
            .store(new_footer_offset, Ordering::Release);
        let header = self.header.read();
        writer_inner.seek(SeekFrom::Start(0))?;
        writer_inner.write_all(&header.to_bytes())?;
        writer_inner.flush()?;

        drop(header);
        drop(writer);

        // Reopen file
        let new_file = open_for_sequential_read(&self.path)?;
        *self.file.write() = Some(new_file);

        if self.durability == super::DurabilityLevel::Fast {
            self.mark_main_sync_pending();
        } else {
            self.sync_main_file_data()?;
            self.clear_main_sync_pending();
        }

        // Update persisted count (disk now has more rows)
        self.persisted_row_count
            .store(new_persisted, Ordering::SeqCst);

        // Keep this storage instance coherent when it stays warm in the insert cache.
        // Without this, a cached V4 footer from before the append can point at the
        // overwritten old-footer bytes, and later readers/deleters may interpret row data
        // as row-group metadata.
        *self.v4_footer.write() = Some(footer);

        Ok(())
    }

    /// Append a new Row Group to an existing V4 file without rewriting.
    /// Overwrites old footer, writes new RG + updated footer, fixes header.
    /// Also updates in-memory state (IDs, active_count).
    /// Use this when adding NEW data that is NOT already in memory.
    pub fn append_row_group(
        &self,
        new_ids: &[u64],
        new_columns: &[ColumnData],
        new_nulls: &[Vec<u8>],
    ) -> io::Result<()> {
        let rg_rows = new_ids.len();
        self.write_row_group_to_disk(new_ids, new_columns, new_nulls)?;

        // Update in-memory state (caller hasn't added these rows yet)
        {
            let mut ids = self.ids.write();
            ids.extend_from_slice(new_ids);
        }
        let next_id = new_ids
            .iter()
            .max()
            .map(|&id| id + 1)
            .unwrap_or(crate::storage::FIRST_ROW_ID);
        let current_next = self.next_id.load(Ordering::SeqCst);
        if next_id > current_next {
            self.next_id.store(next_id, Ordering::SeqCst);
        }
        self.active_count
            .fetch_add(rg_rows as u64, Ordering::SeqCst);
        *self.id_to_idx.write() = None;

        Ok(())
    }

    /// Explicitly sync data to disk (fsync)
    ///
    /// This ensures all buffered data is written to persistent storage.
    /// For safe/max durability modes, also syncs the WAL file.
    /// Called automatically for Safe/Max durability levels on save().
    /// For Fast durability, call this manually when you need durability guarantees.
    fn sync_open_file_data(file: &File) -> io::Result<()> {
        #[cfg(target_os = "macos")]
        {
            let rc = unsafe { libc::fsync(file.as_raw_fd()) };
            return if rc == 0 {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            };
        }
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            file.sync_data()
        }
        #[cfg(not(unix))]
        {
            file.sync_all()
        }
    }

    fn sync_path_data(path: &Path) -> io::Result<()> {
        let file = OpenOptions::new().write(true).append(true).open(path)?;
        Self::sync_open_file_data(&file)
    }

    fn sync_main_file_data(&self) -> io::Result<()> {
        #[cfg(unix)]
        {
            let file_guard = self.file.read();
            if let Some(file) = file_guard.as_ref() {
                return Self::sync_open_file_data(file);
            }
        }

        Self::sync_path_data(&self.path)
    }

    fn sync_delta_file_data(&self, delta_path: &Path) -> io::Result<()> {
        #[cfg(unix)]
        {
            let file_guard = self.delta_file.read();
            if let Some(file) = file_guard.as_ref() {
                return Self::sync_open_file_data(file);
            }
        }

        Self::sync_path_data(delta_path)
    }

    pub fn sync(&self) -> io::Result<()> {
        // Sync WAL first (for safe/max modes)
        if self.durability != super::DurabilityLevel::Fast {
            let mut wal_writer = self.wal_writer.write();
            if let Some(writer) = wal_writer.as_mut() {
                writer.sync()?;
            }
        }

        if self.main_sync_pending() {
            if self.path.exists() {
                self.sync_main_file_data()?;
            }
            self.clear_main_sync_pending();
        }

        let delta_path = Self::delta_path(&self.path);
        if self.delta_sync_pending() {
            if delta_path.exists() {
                self.sync_delta_file_data(&delta_path)?;
            }
            self.clear_delta_sync_pending();
        }

        let deltastore_path = {
            let mut path = self.path.clone();
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            path.set_file_name(format!("{}.deltastore", name));
            path
        };
        if self.deltastore_sync_pending() {
            if deltastore_path.exists() {
                Self::sync_path_data(&deltastore_path)?;
            }
            self.clear_deltastore_sync_pending();
        }
        Ok(())
    }

    fn stats_sidecar_path(&self) -> PathBuf {
        PathBuf::from(format!("{}.stats", self.path.display()))
    }

    fn compute_adaptive_row_group_size(
        schema: &OnDemandSchema,
        columns: &[ColumnData],
        row_count: usize,
    ) -> u32 {
        if row_count == 0 || schema.columns.is_empty() {
            return DEFAULT_ROW_GROUP_SIZE;
        }
        if row_count < DEFAULT_ROW_GROUP_SIZE as usize {
            return DEFAULT_ROW_GROUP_SIZE;
        }

        let sample_rows = row_count.min(2048);
        let estimated_width: f64 = schema
            .columns
            .iter()
            .enumerate()
            .map(|(col_idx, (_, col_type))| {
                Self::estimate_column_row_width(columns.get(col_idx), *col_type, sample_rows)
            })
            .sum();

        if estimated_width < 20.0 {
            131_072
        } else if estimated_width > 100.0 {
            32_768
        } else {
            DEFAULT_ROW_GROUP_SIZE
        }
    }

    fn estimate_column_row_width(
        column: Option<&ColumnData>,
        col_type: ColumnType,
        sample_rows: usize,
    ) -> f64 {
        match column {
            Some(ColumnData::Bool { .. }) | Some(ColumnData::Int64(_)) | Some(ColumnData::Float64(_)) => {
                match col_type {
                    ColumnType::Bool => 1.0,
                    _ => 8.0,
                }
            }
            Some(ColumnData::String { offsets, .. }) => {
                let rows = offsets.len().saturating_sub(1).min(sample_rows);
                if rows == 0 {
                    16.0
                } else {
                    let total = offsets[rows] as f64 - offsets[0] as f64;
                    4.0 + total / rows as f64
                }
            }
            Some(ColumnData::Binary { offsets, .. }) => {
                let rows = offsets.len().saturating_sub(1).min(sample_rows);
                if rows == 0 {
                    16.0
                } else {
                    let total = offsets[rows] as f64 - offsets[0] as f64;
                    4.0 + total / rows as f64
                }
            }
            Some(ColumnData::StringDict { .. }) => 4.0,
            Some(ColumnData::FixedList { dim, .. }) => (*dim as f64 * 4.0).max(8.0),
            Some(ColumnData::Float16List { dim, .. }) => (*dim as f64 * 2.0).max(4.0),
            None => match col_type {
                ColumnType::Bool => 1.0,
                ColumnType::Binary | ColumnType::String | ColumnType::StringDict => 16.0,
                ColumnType::FixedList => 32.0,
                ColumnType::Float16List => 16.0,
                _ => 8.0,
            },
        }
    }

    fn write_col_stats_sidecar(&self, schema: &OnDemandSchema, columns: &[ColumnData]) {
        let nulls = self.nulls.read();
        // Helper: bit=1 means NULL
        #[inline]
        fn is_null_at(bm: &[u8], i: usize) -> bool {
            let b = i / 8;
            let bit = i % 8;
            b < bm.len() && (bm[b] >> bit) & 1 == 1
        }
        let mut buf: Vec<u8> = Vec::with_capacity(512);
        buf.extend_from_slice(b"APEXSTAT");
        let mut entries: Vec<(String, i64, f64, f64, f64, bool)> = Vec::new();
        for (ci, (name, ctype)) in schema.columns.iter().enumerate() {
            let col = match columns.get(ci) {
                Some(c) => c,
                None => continue,
            };
            let null_bm: &[u8] = if ci < nulls.len() { &nulls[ci] } else { &[] };
            let has_nulls = !null_bm.is_empty() && null_bm.iter().any(|&b| b != 0);
            match (ctype, col) {
                (
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
                    ColumnData::Int64(vals),
                ) if !vals.is_empty() => {
                    let (mut cnt, mut s, mut mn, mut mx) = (0i64, 0i64, i64::MAX, i64::MIN);
                    for (i, &v) in vals.iter().enumerate() {
                        if has_nulls && is_null_at(null_bm, i) {
                            continue;
                        }
                        cnt += 1;
                        s = s.wrapping_add(v);
                        if v < mn {
                            mn = v;
                        }
                        if v > mx {
                            mx = v;
                        }
                    }
                    if cnt > 0 {
                        entries.push((name.clone(), cnt, s as f64, mn as f64, mx as f64, true));
                    }
                }
                (ColumnType::Float64 | ColumnType::Float32, ColumnData::Float64(vals))
                    if !vals.is_empty() =>
                {
                    let (mut cnt, mut s, mut mn, mut mx) =
                        (0i64, 0.0f64, f64::INFINITY, f64::NEG_INFINITY);
                    for (i, &v) in vals.iter().enumerate() {
                        if has_nulls && is_null_at(null_bm, i) {
                            continue;
                        }
                        cnt += 1;
                        s += v;
                        if v < mn {
                            mn = v;
                        }
                        if v > mx {
                            mx = v;
                        }
                    }
                    if cnt > 0 {
                        entries.push((name.clone(), cnt, s, mn, mx, false));
                    }
                }
                _ => {}
            }
        }
        if entries.is_empty() {
            return;
        }
        buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (name, count, sum, min, max, is_int) in &entries {
            let nb = name.as_bytes();
            buf.extend_from_slice(&(nb.len() as u16).to_le_bytes());
            buf.extend_from_slice(nb);
            buf.extend_from_slice(&count.to_le_bytes());
            buf.extend_from_slice(&sum.to_bits().to_le_bytes());
            buf.extend_from_slice(&min.to_bits().to_le_bytes());
            buf.extend_from_slice(&max.to_bits().to_le_bytes());
            buf.push(*is_int as u8);
        }
        let _ = std::fs::write(self.stats_sidecar_path(), &buf);
    }

    fn delete_col_stats_sidecar(&self) {
        let p = self.stats_sidecar_path();
        if p.exists() {
            let _ = std::fs::remove_file(&p);
        }
    }
}
