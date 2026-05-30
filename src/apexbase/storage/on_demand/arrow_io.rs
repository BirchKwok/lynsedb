// Arrow batch generation, read_columns, filtered reads

impl OnDemandStorage {
    /// Build Arrow RecordBatch directly from in-memory V4 columns.
    /// OPTIMIZATION: bypasses read_columns→HashMap→get_null_mask→Vec<bool> pipeline.
    /// - Int64/Float64 without nulls: single memcpy (no per-element Option wrapping)
    /// - String: builds from &str references (no per-element String allocation)
    /// - Null bitmaps: read packed bytes directly (no Vec<bool> expansion)
    pub fn to_arrow_batch(
        &self,
        column_names: Option<&[&str]>,
        include_id: bool,
    ) -> io::Result<RecordBatch> {
        self.to_arrow_batch_inner(column_names, include_id, false)
    }

    /// Build Arrow RecordBatch with optional dictionary encoding for string columns.
    /// When dict_encode_strings=true, low-cardinality string columns produce DictionaryArray
    /// which accelerates GROUP BY and WHERE filters.
    pub fn to_arrow_batch_dict(
        &self,
        column_names: Option<&[&str]>,
        include_id: bool,
    ) -> io::Result<RecordBatch> {
        self.to_arrow_batch_inner(column_names, include_id, true)
    }

    fn to_arrow_batch_inner(
        &self,
        column_names: Option<&[&str]>,
        include_id: bool,
        dict_encode_strings: bool,
    ) -> io::Result<RecordBatch> {
        use arrow::array::{Int64Array, StringArray, BooleanArray, PrimitiveArray};
        use arrow::buffer::{Buffer, NullBuffer, BooleanBuffer, ScalarBuffer};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType, Int64Type, Float64Type};
        use std::sync::Arc;

        // ON-DEMAND MMAP PATH: For V4 files, prefer reading directly from mmap
        // instead of loading all data into memory. This is the key memory optimization.
        // LOCK-FREE: use cached_footer_offset atomic instead of header RwLock.
        {
            let is_v4 = self.cached_footer_offset.load(Ordering::Relaxed) > 0;

            if is_v4 {
                // Check if columns are already loaded in memory (write buffer has data)
                let cols = self.columns.read();
                let has_in_memory_data = !cols.is_empty() && cols.iter().any(|c| c.len() > 0);
                drop(cols);

                let on_disk_rows = self.persisted_row_count.load(Ordering::SeqCst) as usize;
                let base_loaded = self.v4_base_loaded.load(Ordering::SeqCst);
                let pending_rows = if has_in_memory_data {
                    self.pending_v4_in_memory_rows()
                } else {
                    0
                };

                if has_in_memory_data && pending_rows > 0 && on_disk_rows > 0 && !base_loaded {
                    let base_batch = self.to_arrow_batch_mmap(
                        column_names, include_id, None, false,
                    )?;
                    let pending_batch = self.pending_v4_to_arrow_batch(column_names, include_id)?;

                    return match base_batch {
                        Some(base) if base.num_rows() > 0 && pending_batch.num_rows() > 0 => {
                            let schema = base.schema();
                            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(base.num_columns());
                            for idx in 0..base.num_columns() {
                                let pieces = [
                                    base.column(idx).as_ref(),
                                    pending_batch.column(idx).as_ref(),
                                ];
                                arrays.push(arrow::compute::concat(&pieces)
                                    .map_err(|e| err_data(e.to_string()))?);
                            }
                            RecordBatch::try_new(schema, arrays)
                                .map_err(|e| err_data(e.to_string()))
                        }
                        Some(base) if base.num_rows() > 0 => Ok(base),
                        _ => Ok(pending_batch),
                    };
                } else if !has_in_memory_data {
                    // Pure mmap path — read directly from disk every time
                    if let Some(batch) = self.to_arrow_batch_mmap(
                        column_names, include_id, None, dict_encode_strings,
                    )? {
                        return Ok(batch);
                    }
                }
                // If we have in-memory data (write buffer), fall through to legacy path
                // which reads from self.columns/ids/nulls/deleted
            }
        }

        // At this point: V4 with in-memory write buffer.
        // No loading needed — data is already available in self.columns/ids.
        let schema = self.schema.read();
        let ids = self.ids.read();
        let columns = self.columns.read();
        let nulls = self.nulls.read();
        let deleted = self.deleted.read();

        let total_rows = ids.len();
        let col_count = schema.column_count();

        // Check for deleted rows
        let has_deleted = deleted.iter().any(|&b| b != 0);

        // Determine active row indices (skip deleted)
        let active_indices: Option<Vec<usize>> = if has_deleted {
            Some((0..total_rows)
                .filter(|&i| {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    byte_idx >= deleted.len() || (deleted[byte_idx] >> bit_idx) & 1 == 0
                })
                .collect())
        } else {
            None
        };
        let active_count = active_indices.as_ref().map(|v| v.len()).unwrap_or(total_rows);

        // Determine which columns to read
        let col_indices: Vec<usize> = if let Some(names) = column_names {
            names.iter()
                .filter(|&&n| n != "_id")
                .filter_map(|&name| schema.get_index(name))
                .collect()
        } else {
            (0..col_count).collect()
        };

        let mut fields: Vec<Field> = Vec::with_capacity(col_indices.len() + 1);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(col_indices.len() + 1);

        // _id column
        if include_id {
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            if let Some(ref indices) = active_indices {
                let active_ids: Vec<i64> = indices.iter().map(|&i| ids[i] as i64).collect();
                arrays.push(Arc::new(Int64Array::from(active_ids)));
            } else {
                // Zero-copy: reinterpret u64 slice as i64 slice, copy once to Arrow buffer
                let mut ids_copy: Vec<i64> = Vec::with_capacity(total_rows);
                // SAFETY: u64 and i64 have identical memory layout
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ids.as_ptr() as *const i64,
                        ids_copy.as_mut_ptr(),
                        total_rows,
                    );
                    ids_copy.set_len(total_rows);
                }
                arrays.push(Arc::new(Int64Array::from(ids_copy)));
            }
        }

        // Data columns — parallel for large tables with multiple columns
        use arrow::array::ArrayRef;
        let convert_column = |col_idx: usize| -> io::Result<(Field, ArrayRef)> {
            let (col_name, _col_type) = &schema.columns[col_idx];
            let col_data = if col_idx < columns.len() { Some(&columns[col_idx]) } else { None };

            // Build Arrow null buffer from packed bitmap
            let null_buf: Option<NullBuffer> = if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                let null_bitmap = &nulls[col_idx];
                let has_any_null = null_bitmap.iter().any(|&b| b != 0);
                if has_any_null {
                    if active_indices.is_none() {
                        // No deletes: Arrow validity = inverted null bitmap
                        // null_bitmap bit=1 means NULL, Arrow validity bit=1 means VALID
                        let mut validity_bytes = vec![0xFFu8; (active_count + 7) / 8];
                        for byte_idx in 0..null_bitmap.len().min(validity_bytes.len()) {
                            validity_bytes[byte_idx] = !null_bitmap[byte_idx];
                        }
                        // Mask trailing bits
                        let tail = active_count % 8;
                        if tail > 0 {
                            let last = validity_bytes.len() - 1;
                            validity_bytes[last] &= (1u8 << tail) - 1;
                        }
                        Some(NullBuffer::new(BooleanBuffer::new(Buffer::from(validity_bytes), 0, active_count)))
                    } else {
                        // Has deletes: build validity for active rows only
                        let indices = active_indices.as_ref().unwrap();
                        let mut validity_bytes = vec![0xFFu8; (active_count + 7) / 8];
                        for (new_idx, &old_idx) in indices.iter().enumerate() {
                            let ob = old_idx / 8;
                            let obit = old_idx % 8;
                            if ob < null_bitmap.len() && (null_bitmap[ob] >> obit) & 1 == 1 {
                                // This row is NULL → clear validity bit
                                validity_bytes[new_idx / 8] &= !(1u8 << (new_idx % 8));
                            }
                        }
                        Some(NullBuffer::new(BooleanBuffer::new(Buffer::from(validity_bytes), 0, active_count)))
                    }
                } else {
                    None // All valid
                }
            } else {
                None // No null info
            };

            let schema_col_type = *_col_type;
            let (arrow_dt, array): (ArrowDataType, ArrayRef) = match col_data {
                Some(ColumnData::Int64(values)) => {
                    let data_vec: Vec<i64> = if let Some(ref indices) = active_indices {
                        indices.iter().map(|&i| if i < values.len() { values[i] } else { 0 }).collect()
                    } else {
                        values.clone()
                    };
                    // Use schema type to produce proper Arrow type for Timestamp/Date
                    match schema_col_type {
                        ColumnType::Timestamp => {
                            use arrow::datatypes::TimestampMicrosecondType;
                            let arr = PrimitiveArray::<TimestampMicrosecondType>::new(
                                ScalarBuffer::from(data_vec), null_buf,
                            );
                            (ArrowDataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None), Arc::new(arr) as ArrayRef)
                        }
                        ColumnType::Date => {
                            // Date stored as i64 days since epoch, convert to i32 for Arrow Date32
                            use arrow::datatypes::Date32Type;
                            let data_i32: Vec<i32> = data_vec.iter().map(|&v| v as i32).collect();
                            let arr = PrimitiveArray::<Date32Type>::new(
                                ScalarBuffer::from(data_i32), null_buf,
                            );
                            (ArrowDataType::Date32, Arc::new(arr) as ArrayRef)
                        }
                        _ => {
                            let arr = PrimitiveArray::<Int64Type>::new(
                                ScalarBuffer::from(data_vec), null_buf,
                            );
                            (ArrowDataType::Int64, Arc::new(arr) as ArrayRef)
                        }
                    }
                }
                Some(ColumnData::Float64(values)) => {
                    let data_vec = if let Some(ref indices) = active_indices {
                        indices.iter().map(|&i| if i < values.len() { values[i] } else { 0.0 }).collect()
                    } else {
                        values.clone()
                    };
                    let arr = PrimitiveArray::<Float64Type>::new(
                        ScalarBuffer::from(data_vec), null_buf,
                    );
                    (ArrowDataType::Float64, Arc::new(arr) as ArrayRef)
                }
                Some(ColumnData::String { offsets, data }) => {
                    // OPTIMIZATION: build StringArray from &str refs (no per-element String alloc)
                    let count = offsets.len().saturating_sub(1);
                    if let Some(ref indices) = active_indices {
                        let strings: Vec<Option<&str>> = indices.iter().map(|&i| {
                            if i < count {
                                // Check null
                                if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                                    let ob = i / 8;
                                    let obit = i % 8;
                                    if ob < nulls[col_idx].len() && (nulls[col_idx][ob] >> obit) & 1 == 1 {
                                        return None;
                                    }
                                }
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&data[start..end]).ok()
                            } else {
                                None
                            }
                        }).collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    } else if null_buf.is_some() {
                        let null_bitmap = &nulls[col_idx];
                        let strings: Vec<Option<&str>> = (0..count.min(active_count)).map(|i| {
                            let ob = i / 8;
                            let obit = i % 8;
                            if ob < null_bitmap.len() && (null_bitmap[ob] >> obit) & 1 == 1 {
                                None
                            } else {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&data[start..end]).ok()
                            }
                        }).collect();
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    } else {
                        // No nulls, no deletes: fastest path
                        let row_count = count.min(active_count);
                        
                        // OPTIMIZATION: Try to build DictionaryArray for low-cardinality columns
                        // Only when dict_encode_strings=true (GROUP BY queries)
                        let try_dict = dict_encode_strings && row_count >= 100;
                        if try_dict {
                            // Sample first 1000 rows to estimate cardinality
                            let sample_size = row_count.min(1000);
                            let mut sample_unique = ahash::AHashSet::with_capacity(100);
                            let step = if row_count > sample_size { row_count / sample_size } else { 1 };
                            let mut si = 0;
                            while si < row_count && sample_unique.len() <= 1000 {
                                let start = offsets[si] as usize;
                                let end = offsets[si + 1] as usize;
                                sample_unique.insert(&data[start..end]);
                                si += step;
                            }
                            
                            if sample_unique.len() <= 1000 {
                                // Low cardinality → build DictionaryArray<UInt32Type>
                                use arrow::array::{UInt32Array, DictionaryArray};
                                use arrow::datatypes::UInt32Type;
                                
                                let mut dict_map: ahash::AHashMap<&[u8], u32> = ahash::AHashMap::with_capacity(sample_unique.len());
                                let mut dict_strings: Vec<&str> = Vec::with_capacity(sample_unique.len());
                                let mut next_id = 0u32;
                                let mut keys: Vec<u32> = Vec::with_capacity(row_count);
                                
                                for i in 0..row_count {
                                    let start = offsets[i] as usize;
                                    let end = offsets[i + 1] as usize;
                                    let bytes = &data[start..end];
                                    let id = *dict_map.entry(bytes).or_insert_with(|| {
                                        let id = next_id;
                                        next_id += 1;
                                        dict_strings.push(std::str::from_utf8(bytes).unwrap_or(""));
                                        id
                                    });
                                    keys.push(id);
                                }
                                
                                let keys_array = UInt32Array::from(keys);
                                let values_array = StringArray::from_iter_values(dict_strings);
                                let dict_array = DictionaryArray::<UInt32Type>::try_new(
                                    keys_array, Arc::new(values_array),
                                ).map_err(|e| err_data(e.to_string()))?;
                                let arr_ref: ArrayRef = Arc::new(dict_array);
                                (arr_ref.data_type().clone(), arr_ref)
                            } else {
                                // High cardinality → plain StringArray
                                let strings: Vec<&str> = (0..row_count).map(|i| {
                                    let start = offsets[i] as usize;
                                    let end = offsets[i + 1] as usize;
                                    std::str::from_utf8(&data[start..end]).unwrap_or("")
                                }).collect();
                                (ArrowDataType::Utf8, Arc::new(StringArray::from_iter_values(strings)))
                            }
                        } else {
                            // OPTIMIZATION: build StringArray directly from u32 offsets + data bytes
                            // Avoids intermediate Vec<&str> and per-string from_utf8 validation
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
                            // SAFETY: data written by our storage engine is valid UTF-8
                            (ArrowDataType::Utf8, Arc::new(unsafe { StringArray::new_unchecked(offset_buf, data_buf, None) }) as ArrayRef)
                        }
                    }
                }
                Some(ColumnData::Bool { data: packed, len }) => {
                    if let Some(ref indices) = active_indices {
                        let bools: Vec<Option<bool>> = indices.iter().map(|&i| {
                            if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                                let ob = i / 8;
                                let obit = i % 8;
                                if ob < nulls[col_idx].len() && (nulls[col_idx][ob] >> obit) & 1 == 1 {
                                    return None;
                                }
                            }
                            if i < *len {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                            } else {
                                None
                            }
                        }).collect();
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    } else {
                        let bools: Vec<Option<bool>> = (0..*len).map(|i| {
                            if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                                let ob = i / 8;
                                let obit = i % 8;
                                if ob < nulls[col_idx].len() && (nulls[col_idx][ob] >> obit) & 1 == 1 {
                                    return None;
                                }
                            }
                            let byte_idx = i / 8;
                            let bit_idx = i % 8;
                            Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                        }).collect();
                        (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                    }
                }
                Some(ColumnData::Binary { offsets, data }) => {
                    use arrow::array::BinaryArray;
                    let count = offsets.len().saturating_sub(1);
                    let binary_data: Vec<Option<&[u8]>> = if let Some(ref indices) = active_indices {
                        indices.iter().map(|&i| {
                            if i < count {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&data[start..end] as &[u8])
                            } else { None }
                        }).collect()
                    } else {
                        (0..count.min(active_count)).map(|i| {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            Some(&data[start..end] as &[u8])
                        }).collect()
                    };
                    (ArrowDataType::Binary, Arc::new(BinaryArray::from(binary_data)))
                }
                Some(ColumnData::StringDict { indices, dict_offsets, dict_data }) => {
                    let value_at = |row_idx: usize| -> Option<&str> {
                        if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                            let ob = row_idx / 8;
                            let obit = row_idx % 8;
                            if ob < nulls[col_idx].len() && (nulls[col_idx][ob] >> obit) & 1 == 1 {
                                return None;
                            }
                        }

                        let raw_idx = indices.get(row_idx).copied().unwrap_or(0);
                        if raw_idx == 0 {
                            return None;
                        }

                        let dict_idx = (raw_idx - 1) as usize;
                        if dict_idx + 1 >= dict_offsets.len() {
                            return None;
                        }
                        let start = dict_offsets[dict_idx] as usize;
                        let end = dict_offsets[dict_idx + 1] as usize;
                        if end > dict_data.len() {
                            return None;
                        }
                        std::str::from_utf8(&dict_data[start..end]).ok()
                    };

                    let strings: Vec<Option<&str>> = if let Some(ref indices) = active_indices {
                        indices.iter().map(|&i| value_at(i)).collect()
                    } else {
                        (0..active_count).map(value_at).collect()
                    };
                    (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                }
                Some(ColumnData::Float16List { data, dim }) => {
                    use arrow::array::{FixedSizeListArray, Float32Array};
                    let dim_usize = *dim as usize;
                    let row_count_full = if dim_usize == 0 { 0 } else { data.len() / (dim_usize * 2) };
                    let f32_data: Vec<f32> = if let Some(ref indices) = active_indices {
                        let mut out = Vec::with_capacity(indices.len() * dim_usize);
                        for &i in indices {
                            let src_start = i * dim_usize * 2;
                            let src_end = src_start + dim_usize * 2;
                            if src_end <= data.len() {
                                for chunk in data[src_start..src_end].chunks_exact(2) {
                                    let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                                    out.push(crate::storage::on_demand::f16_to_f32(bits));
                                }
                            } else {
                                out.extend(std::iter::repeat(0.0f32).take(dim_usize));
                            }
                        }
                        out
                    } else {
                        let n = row_count_full.min(active_count);
                        let mut out = Vec::with_capacity(n * dim_usize);
                        for chunk in data[..n * dim_usize * 2].chunks_exact(2) {
                            let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                            out.push(crate::storage::on_demand::f16_to_f32(bits));
                        }
                        out
                    };
                    let row_count = if dim_usize == 0 { 0 } else { f32_data.len() / dim_usize };
                    let float_arr = Float32Array::from(f32_data);
                    let list_dt = ArrowDataType::FixedSizeList(
                        Arc::new(Field::new("item", ArrowDataType::Float32, false)), dim_usize as i32,
                    );
                    let arr = FixedSizeListArray::new(
                        Arc::new(Field::new("item", ArrowDataType::Float32, false)), dim_usize as i32,
                        Arc::new(float_arr), null_buf,
                    );
                    (list_dt, Arc::new(arr) as ArrayRef)
                }
                Some(ColumnData::FixedList { data, dim }) => {
                    use arrow::array::{FixedSizeListArray, Float32Array};
                    let dim_usize = *dim as usize;
                    let row_count_full = if dim_usize == 0 { 0 } else { data.len() / (dim_usize * 4) };
                    let selected_data: Vec<f32> = if let Some(ref indices) = active_indices {
                        let mut out = Vec::with_capacity(indices.len() * dim_usize);
                        for &i in indices {
                            let start = i * dim_usize * 4;
                            let end = start + dim_usize * 4;
                            if end <= data.len() {
                                out.extend(crate::storage::on_demand::f32_le_bytes_to_values(&data[start..end]));
                            } else {
                                out.extend(std::iter::repeat(0.0f32).take(dim_usize));
                            }
                        }
                        out
                    } else {
                        let byte_len = row_count_full.min(active_count) * dim_usize * 4;
                        crate::storage::on_demand::f32_le_bytes_to_values(&data[..byte_len])
                    };
                    let row_count = if dim_usize == 0 { 0 } else { selected_data.len() / dim_usize };
                    let float_arr = Float32Array::from(selected_data);
                    let list_dt = ArrowDataType::FixedSizeList(
                        Arc::new(Field::new("item", ArrowDataType::Float32, false)),
                        dim_usize as i32,
                    );
                    let arr = FixedSizeListArray::new(
                        Arc::new(Field::new("item", ArrowDataType::Float32, false)),
                        dim_usize as i32,
                        Arc::new(float_arr),
                        null_buf,
                    );
                    (list_dt, Arc::new(arr) as ArrayRef)
                }
                None => {
                    // Column doesn't exist, create default
                    (ArrowDataType::Int64, Arc::new(Int64Array::from(vec![0i64; active_count])))
                }
            };

            Ok((Field::new(col_name, arrow_dt, true), array))
        };

        if active_count >= 50_000 && col_indices.len() >= 2 {
            use rayon::prelude::*;
            let results: Vec<io::Result<(Field, ArrayRef)>> = col_indices.par_iter()
                .map(|&ci| convert_column(ci)).collect();
            for r in results {
                let (f, a) = r?;
                fields.push(f);
                arrays.push(a);
            }
        } else {
            for &ci in &col_indices {
                let (f, a) = convert_column(ci)?;
                fields.push(f);
                arrays.push(a);
            }
        }

        let arrow_schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(arrow_schema, arrays)
            .map_err(|e| err_data(e.to_string()))
    }

    /// Build an Arrow batch from only the V4 in-memory append area.
    ///
    /// Insert backends for mmap-only tables keep just pending rows in
    /// ids/columns/nulls. The normal in-memory path assumes those buffers are a
    /// complete table, so SQL overlay reads need a pending-only batch that can be
    /// concatenated after the mmap base batch.
    fn pending_v4_to_arrow_batch(
        &self,
        column_names: Option<&[&str]>,
        include_id: bool,
    ) -> io::Result<RecordBatch> {
        use arrow::array::{
            ArrayRef, BinaryArray, BooleanArray, FixedSizeListArray, Float32Array, Int64Array,
            PrimitiveArray, StringArray,
        };
        use arrow::buffer::{Buffer, BooleanBuffer, NullBuffer, ScalarBuffer};
        use arrow::datatypes::{
            DataType as ArrowDataType, Date32Type, Field, Float64Type, Int64Type, Schema,
            TimeUnit, TimestampMicrosecondType,
        };
        use std::sync::Arc;

        let schema = self.schema.read();
        let ids = self.ids.read();
        let columns = self.columns.read();
        let nulls = self.nulls.read();
        let row_count = ids.len();
        let row_ids: Vec<u64> = ids.iter().copied().collect();

        let col_indices: Vec<usize> = if let Some(names) = column_names {
            names.iter()
                .filter(|&&n| n != "_id")
                .filter_map(|&name| schema.get_index(name))
                .collect()
        } else {
            (0..schema.column_count()).collect()
        };

        let mut fields: Vec<Field> = Vec::with_capacity(col_indices.len() + 1);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(col_indices.len() + 1);

        if include_id {
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(
                ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
            )));
        }

        let is_null = |col_idx: usize, row_idx: usize| -> bool {
            if col_idx >= nulls.len() {
                return false;
            }
            let null_bitmap = &nulls[col_idx];
            let byte_idx = row_idx / 8;
            let bit_idx = row_idx % 8;
            byte_idx < null_bitmap.len() && (null_bitmap[byte_idx] >> bit_idx) & 1 == 1
        };

        let make_null_buf = |col_idx: usize| -> Option<NullBuffer> {
            if col_idx >= nulls.len() || row_count == 0 {
                return None;
            }
            let null_bitmap = &nulls[col_idx];
            if null_bitmap.is_empty() || !null_bitmap.iter().any(|&b| b != 0) {
                return None;
            }
            let mut validity_bytes = vec![0xFFu8; (row_count + 7) / 8];
            for row_idx in 0..row_count {
                if is_null(col_idx, row_idx) {
                    validity_bytes[row_idx / 8] &= !(1u8 << (row_idx % 8));
                }
            }
            let tail = row_count % 8;
            if tail > 0 {
                let last = validity_bytes.len() - 1;
                validity_bytes[last] &= (1u8 << tail) - 1;
            }
            Some(NullBuffer::new(BooleanBuffer::new(
                Buffer::from(validity_bytes),
                0,
                row_count,
            )))
        };

        for &col_idx in &col_indices {
            let (col_name, schema_col_type) = &schema.columns[col_idx];
            let schema_col_type = *schema_col_type;
            let col_data = columns.get(col_idx);
            let null_buf = make_null_buf(col_idx);

            let (arrow_dt, array): (ArrowDataType, ArrayRef) = match col_data {
                Some(ColumnData::Int64(values)) => {
                    let data_vec: Vec<i64> = (0..row_count)
                        .map(|i| values.get(i).copied().unwrap_or(0))
                        .collect();
                    match schema_col_type {
                        ColumnType::Timestamp => {
                            let arr = PrimitiveArray::<TimestampMicrosecondType>::new(
                                ScalarBuffer::from(data_vec),
                                null_buf,
                            );
                            (
                                ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
                                Arc::new(arr) as ArrayRef,
                            )
                        }
                        ColumnType::Date => {
                            let data_i32: Vec<i32> = data_vec.iter().map(|&v| v as i32).collect();
                            let arr = PrimitiveArray::<Date32Type>::new(
                                ScalarBuffer::from(data_i32),
                                null_buf,
                            );
                            (ArrowDataType::Date32, Arc::new(arr) as ArrayRef)
                        }
                        _ => {
                            let arr = PrimitiveArray::<Int64Type>::new(
                                ScalarBuffer::from(data_vec),
                                null_buf,
                            );
                            (ArrowDataType::Int64, Arc::new(arr) as ArrayRef)
                        }
                    }
                }
                Some(ColumnData::Float64(values)) => {
                    let data_vec: Vec<f64> = (0..row_count)
                        .map(|i| values.get(i).copied().unwrap_or(0.0))
                        .collect();
                    let arr = PrimitiveArray::<Float64Type>::new(
                        ScalarBuffer::from(data_vec),
                        null_buf,
                    );
                    (ArrowDataType::Float64, Arc::new(arr) as ArrayRef)
                }
                Some(ColumnData::String { offsets, data }) => {
                    let count = offsets.len().saturating_sub(1);
                    let strings: Vec<Option<&str>> = (0..row_count)
                        .map(|i| {
                            if is_null(col_idx, i) || i >= count {
                                return None;
                            }
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            std::str::from_utf8(&data[start..end]).ok()
                        })
                        .collect();
                    (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                }
                Some(ColumnData::Bool { data, len }) => {
                    let bools: Vec<Option<bool>> = (0..row_count)
                        .map(|i| {
                            if is_null(col_idx, i) || i >= *len {
                                return None;
                            }
                            let byte_idx = i / 8;
                            let bit_idx = i % 8;
                            Some(byte_idx < data.len() && (data[byte_idx] >> bit_idx) & 1 == 1)
                        })
                        .collect();
                    (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                }
                Some(ColumnData::Binary { offsets, data }) => {
                    let count = offsets.len().saturating_sub(1);
                    let binary_data: Vec<Option<&[u8]>> = (0..row_count)
                        .map(|i| {
                            if is_null(col_idx, i) || i >= count {
                                return None;
                            }
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            Some(&data[start..end])
                        })
                        .collect();
                    (ArrowDataType::Binary, Arc::new(BinaryArray::from(binary_data)))
                }
                Some(ColumnData::StringDict { indices, dict_offsets, dict_data }) => {
                    let strings: Vec<Option<&str>> = (0..row_count)
                        .map(|i| {
                            if is_null(col_idx, i) {
                                return None;
                            }
                            let raw_idx = indices.get(i).copied().unwrap_or(0);
                            if raw_idx == 0 {
                                return None;
                            }
                            let dict_idx = (raw_idx - 1) as usize;
                            if dict_idx + 1 >= dict_offsets.len() {
                                return None;
                            }
                            let start = dict_offsets[dict_idx] as usize;
                            let end = dict_offsets[dict_idx + 1] as usize;
                            std::str::from_utf8(&dict_data[start..end]).ok()
                        })
                        .collect();
                    (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                }
                Some(ColumnData::FixedList { data, dim }) => {
                    let dim_usize = *dim as usize;
                    if dim_usize == 0 {
                        let arr = arrow::array::new_null_array(&ArrowDataType::Utf8, row_count);
                        (ArrowDataType::Utf8, arr)
                    } else {
                        let byte_len = row_count * dim_usize * 4;
                        let mut selected_data = crate::storage::on_demand::f32_le_bytes_to_values(
                            &data[..data.len().min(byte_len)],
                        );
                        selected_data.resize(row_count * dim_usize, 0.0);
                        let float_arr = Float32Array::from(selected_data);
                        let item = Arc::new(Field::new("item", ArrowDataType::Float32, false));
                        let list_dt = ArrowDataType::FixedSizeList(item.clone(), dim_usize as i32);
                        let arr = FixedSizeListArray::new(
                            item,
                            dim_usize as i32,
                            Arc::new(float_arr),
                            null_buf,
                        );
                        (list_dt, Arc::new(arr) as ArrayRef)
                    }
                }
                Some(ColumnData::Float16List { data, dim }) => {
                    let dim_usize = *dim as usize;
                    if dim_usize == 0 {
                        let arr = arrow::array::new_null_array(&ArrowDataType::Utf8, row_count);
                        (ArrowDataType::Utf8, arr)
                    } else {
                        let mut f32_values = Vec::with_capacity(row_count * dim_usize);
                        let available_rows = data.len() / (dim_usize * 2);
                        for row_idx in 0..row_count {
                            if row_idx < available_rows {
                                let start = row_idx * dim_usize * 2;
                                let end = start + dim_usize * 2;
                                for chunk in data[start..end].chunks_exact(2) {
                                    let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                                    f32_values.push(crate::storage::on_demand::f16_to_f32(bits));
                                }
                            } else {
                                f32_values.extend(std::iter::repeat(0.0f32).take(dim_usize));
                            }
                        }
                        let float_arr = Float32Array::from(f32_values);
                        let item = Arc::new(Field::new("item", ArrowDataType::Float32, false));
                        let list_dt = ArrowDataType::FixedSizeList(item.clone(), dim_usize as i32);
                        let arr = FixedSizeListArray::new(
                            item,
                            dim_usize as i32,
                            Arc::new(float_arr),
                            null_buf,
                        );
                        (list_dt, Arc::new(arr) as ArrayRef)
                    }
                }
                None => {
                    let arr = arrow::array::new_null_array(&ArrowDataType::Utf8, row_count);
                    (ArrowDataType::Utf8, arr)
                }
            };

            fields.push(Field::new(col_name, arrow_dt, true));
            arrays.push(array);
        }

        let arrow_schema = Arc::new(Schema::new(fields));
        let batch =
            RecordBatch::try_new(arrow_schema, arrays).map_err(|e| err_data(e.to_string()))?;
        let delta = self.delta_store.read();
        if delta.is_empty() {
            Ok(batch)
        } else {
            crate::storage::DeltaMerger::merge(&batch, &delta, &row_ids)
        }
    }

    /// Build Arrow RecordBatch with a row LIMIT from in-memory V4 columns.
    /// Much faster than read_columns() for small LIMIT queries (SELECT * LIMIT N).
    pub fn to_arrow_batch_with_limit(
        &self,
        column_names: Option<&[&str]>,
        include_id: bool,
        limit: usize,
    ) -> io::Result<RecordBatch> {
        use arrow::array::{Int64Array, StringArray, BooleanArray, PrimitiveArray};
        use arrow::buffer::{Buffer, NullBuffer, BooleanBuffer, ScalarBuffer};
        use arrow::datatypes::{Schema, Field, DataType as ArrowDataType, Int64Type, Float64Type};
        use std::sync::Arc;

        // ON-DEMAND PATH for LIMIT queries
        {
            let is_v4 = self.cached_footer_offset.load(Ordering::Relaxed) > 0;
            if is_v4 {
                let cols = self.columns.read();
                let has_in_memory_data = !cols.is_empty() && cols.iter().any(|c| c.len() > 0);
                drop(cols);
                if !has_in_memory_data {
                    // PREAD RCIX PATH: reads only the minimal bytes needed (no mmap page faults).
                    // Falls back to mmap path for multi-RG, compressed, deleted rows, or pending deltas.
                    // Skip pread when DeltaStore has pending updates: pread reads raw bytes and does
                    // NOT apply DeltaMerger, so it would return stale (pre-update) values.
                    let has_pending_deltas = !self.delta_store.read().is_empty();
                    if !has_pending_deltas {
                        let footer_opt = self.v4_footer.read().clone();
                        if let Some(ref footer) = footer_opt {
                            let schema = &footer.schema;
                            let col_indices: Vec<usize> = if let Some(names) = column_names {
                                names.iter().filter(|&&n| n != "_id")
                                    .filter_map(|&name| schema.get_index(name))
                                    .collect()
                            } else {
                                (0..schema.column_count()).collect()
                            };
                            if let Ok(Some(batch)) = self.to_arrow_batch_pread_rcix(&col_indices, include_id, limit) {
                                return Ok(batch);
                            }
                        }
                    }
                    // MMAP path: handles multi-RG, deletes, compressed, unknown encodings,
                    // and applies DeltaMerger overlay for pending cell-level updates.
                    if let Some(batch) = self.to_arrow_batch_mmap(
                        column_names, include_id, Some(limit), true,
                    )? {
                        return Ok(batch);
                    }
                }
            }
        }

        // At this point: V4 with in-memory write buffer.
        let schema = self.schema.read();
        let ids = self.ids.read();
        let columns = self.columns.read();
        let nulls = self.nulls.read();
        let deleted = self.deleted.read();

        let total_rows = ids.len();
        let col_count = schema.column_count();
        let has_deleted = deleted.iter().any(|&b| b != 0);

        // Collect first `limit` active row indices
        let actual_limit;
        let row_indices: Option<Vec<usize>> = if has_deleted {
            let mut indices = Vec::with_capacity(limit.min(total_rows));
            for i in 0..total_rows {
                if indices.len() >= limit { break; }
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                if byte_idx >= deleted.len() || (deleted[byte_idx] >> bit_idx) & 1 == 0 {
                    indices.push(i);
                }
            }
            actual_limit = indices.len();
            Some(indices)
        } else {
            actual_limit = limit.min(total_rows);
            None // contiguous range 0..actual_limit
        };

        if actual_limit == 0 {
            let arrow_schema = Arc::new(Schema::empty());
            return Ok(RecordBatch::new_empty(arrow_schema));
        }

        let col_indices: Vec<usize> = if let Some(names) = column_names {
            names.iter()
                .filter(|&&n| n != "_id")
                .filter_map(|&name| schema.get_index(name))
                .collect()
        } else {
            (0..col_count).collect()
        };

        let mut fields: Vec<Field> = Vec::with_capacity(col_indices.len() + 1);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(col_indices.len() + 1);

        // _id column
        if include_id {
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            if let Some(ref indices) = row_indices {
                let active_ids: Vec<i64> = indices.iter().map(|&i| ids[i] as i64).collect();
                arrays.push(Arc::new(Int64Array::from(active_ids)));
            } else {
                // Contiguous: just copy first actual_limit IDs
                let mut ids_copy: Vec<i64> = Vec::with_capacity(actual_limit);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ids.as_ptr() as *const i64,
                        ids_copy.as_mut_ptr(),
                        actual_limit,
                    );
                    ids_copy.set_len(actual_limit);
                }
                arrays.push(Arc::new(Int64Array::from(ids_copy)));
            }
        }

        for &col_idx in &col_indices {
            let (col_name, schema_col_type) = &schema.columns[col_idx];
            let schema_col_type = *schema_col_type;
            let col_data = if col_idx < columns.len() { Some(&columns[col_idx]) } else { None };

            // Build null buffer for this column (critical for IS NULL queries)
            let null_buf: Option<NullBuffer> = if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                let null_bitmap = &nulls[col_idx];
                let has_any_null = null_bitmap.iter().any(|&b| b != 0);
                if has_any_null {
                    let mut validity_bytes = vec![0xFFu8; (actual_limit + 7) / 8];
                    if let Some(ref indices) = row_indices {
                        for (new_idx, &old_idx) in indices.iter().enumerate() {
                            let ob = old_idx / 8;
                            let obit = old_idx % 8;
                            if ob < null_bitmap.len() && (null_bitmap[ob] >> obit) & 1 == 1 {
                                validity_bytes[new_idx / 8] &= !(1u8 << (new_idx % 8));
                            }
                        }
                    } else {
                        for byte_idx in 0..null_bitmap.len().min(validity_bytes.len()) {
                            validity_bytes[byte_idx] = !null_bitmap[byte_idx];
                        }
                    }
                    let tail = actual_limit % 8;
                    if tail > 0 {
                        let last = validity_bytes.len() - 1;
                        validity_bytes[last] &= (1u8 << tail) - 1;
                    }
                    Some(NullBuffer::new(BooleanBuffer::new(Buffer::from(validity_bytes), 0, actual_limit)))
                } else { None }
            } else { None };

            let (arrow_dt, array): (ArrowDataType, ArrayRef) = match col_data {
                Some(ColumnData::Int64(values)) => {
                    let data_vec: Vec<i64> = if let Some(ref indices) = row_indices {
                        indices.iter().map(|&i| if i < values.len() { values[i] } else { 0 }).collect()
                    } else {
                        values[..actual_limit.min(values.len())].to_vec()
                    };
                    match schema_col_type {
                        ColumnType::Timestamp => {
                            use arrow::datatypes::TimestampMicrosecondType;
                            let arr = PrimitiveArray::<TimestampMicrosecondType>::new(
                                ScalarBuffer::from(data_vec), null_buf,
                            );
                            (ArrowDataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None), Arc::new(arr) as ArrayRef)
                        }
                        ColumnType::Date => {
                            use arrow::datatypes::Date32Type;
                            let data_i32: Vec<i32> = data_vec.iter().map(|&v| v as i32).collect();
                            let arr = PrimitiveArray::<Date32Type>::new(
                                ScalarBuffer::from(data_i32), null_buf,
                            );
                            (ArrowDataType::Date32, Arc::new(arr) as ArrayRef)
                        }
                        _ => {
                            let arr = PrimitiveArray::<Int64Type>::new(ScalarBuffer::from(data_vec), null_buf);
                            (ArrowDataType::Int64, Arc::new(arr) as ArrayRef)
                        }
                    }
                }
                Some(ColumnData::Float64(values)) => {
                    let data_vec: Vec<f64> = if let Some(ref indices) = row_indices {
                        indices.iter().map(|&i| if i < values.len() { values[i] } else { 0.0 }).collect()
                    } else {
                        values[..actual_limit.min(values.len())].to_vec()
                    };
                    let arr = PrimitiveArray::<Float64Type>::new(ScalarBuffer::from(data_vec), null_buf);
                    (ArrowDataType::Float64, Arc::new(arr) as ArrayRef)
                }
                Some(ColumnData::String { offsets, data }) => {
                    let count = offsets.len().saturating_sub(1);
                    if null_buf.is_some() {
                        // Has nulls: use Option<&str> path
                        let null_bitmap = &nulls[col_idx];
                        let strings: Vec<Option<&str>> = if let Some(ref indices) = row_indices {
                            indices.iter().map(|&i| {
                                let ob = i / 8;
                                let obit = i % 8;
                                if ob < null_bitmap.len() && (null_bitmap[ob] >> obit) & 1 == 1 {
                                    None
                                } else if i < count {
                                    let start = offsets[i] as usize;
                                    let end = offsets[i + 1] as usize;
                                    std::str::from_utf8(&data[start..end]).ok()
                                } else { None }
                            }).collect()
                        } else {
                            (0..actual_limit.min(count)).map(|i| {
                                let ob = i / 8;
                                let obit = i % 8;
                                if ob < null_bitmap.len() && (null_bitmap[ob] >> obit) & 1 == 1 {
                                    None
                                } else {
                                    let start = offsets[i] as usize;
                                    let end = offsets[i + 1] as usize;
                                    std::str::from_utf8(&data[start..end]).ok()
                                }
                            }).collect()
                        };
                        (ArrowDataType::Utf8, Arc::new(StringArray::from(strings)))
                    } else {
                        let strings: Vec<&str> = if let Some(ref indices) = row_indices {
                            indices.iter().map(|&i| {
                                if i < count {
                                    let start = offsets[i] as usize;
                                    let end = offsets[i + 1] as usize;
                                    std::str::from_utf8(&data[start..end]).unwrap_or("")
                                } else { "" }
                            }).collect()
                        } else {
                            let lim = actual_limit.min(count);
                            (0..lim).map(|i| {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                std::str::from_utf8(&data[start..end]).unwrap_or("")
                            }).collect()
                        };
                        (ArrowDataType::Utf8, Arc::new(StringArray::from_iter_values(strings)))
                    }
                }
                Some(ColumnData::Bool { data: packed, len }) => {
                    let bools: Vec<Option<bool>> = if let Some(ref indices) = row_indices {
                        indices.iter().map(|&i| {
                            // Check null
                            if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                                let ob = i / 8;
                                let obit = i % 8;
                                if ob < nulls[col_idx].len() && (nulls[col_idx][ob] >> obit) & 1 == 1 {
                                    return None;
                                }
                            }
                            if i < *len {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                            } else { None }
                        }).collect()
                    } else {
                        (0..actual_limit.min(*len)).map(|i| {
                            // Check null
                            if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                                let ob = i / 8;
                                let obit = i % 8;
                                if ob < nulls[col_idx].len() && (nulls[col_idx][ob] >> obit) & 1 == 1 {
                                    return None;
                                }
                            }
                            let byte_idx = i / 8;
                            let bit_idx = i % 8;
                            Some(byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1)
                        }).collect()
                    };
                    (ArrowDataType::Boolean, Arc::new(BooleanArray::from(bools)))
                }
                Some(ColumnData::Binary { offsets, data }) => {
                    use arrow::array::BinaryArray;
                    let count = offsets.len().saturating_sub(1);
                    let binary_data: Vec<Option<&[u8]>> = if let Some(ref indices) = row_indices {
                        indices.iter().map(|&i| {
                            if i < count {
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                Some(&data[start..end] as &[u8])
                            } else { None }
                        }).collect()
                    } else {
                        (0..actual_limit.min(count)).map(|i| {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            Some(&data[start..end] as &[u8])
                        }).collect()
                    };
                    (ArrowDataType::Binary, Arc::new(BinaryArray::from(binary_data)))
                }
                _ => {
                    (ArrowDataType::Int64, Arc::new(Int64Array::from(vec![0i64; actual_limit])))
                }
            };

            fields.push(Field::new(col_name, arrow_dt, true));
            arrays.push(array);
        }

        let arrow_schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(arrow_schema, arrays)
            .map_err(|e| err_data(e.to_string()))
    }

    // ========================================================================
    // On-Demand Read APIs (the key feature)
    // ========================================================================

    /// Read specific columns for a row range
    /// 
    /// This is the core on-demand read function:
    /// - Only reads the requested columns from disk
    /// - Only reads the requested row range
    /// - Uses pread for efficient random access
    ///
    /// # Arguments
    /// * `column_names` - Columns to read (None = all columns)
    /// * `start_row` - Starting row index (0-based)
    /// * `row_count` - Number of rows to read (None = to end)
    pub fn read_columns(
        &self,
        column_names: Option<&[&str]>,
        start_row: usize,
        row_count: Option<usize>,
    ) -> io::Result<HashMap<String, ColumnData>> {
        let header = self.header.read();
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        
        let base_rows = header.row_count as usize;
        let delta_rows = self.delta_row_count();
        let total_rows = base_rows + delta_rows;
        
        let actual_start = start_row.min(total_rows);
        let actual_count = row_count
            .map(|c| c.min(total_rows - actual_start))
            .unwrap_or(total_rows - actual_start);
        
        if actual_count == 0 {
            return Ok(HashMap::new());
        }

        // Determine which columns to read (by name for delta merge)
        // When reading all columns (None), include both base and delta columns
        let mut col_names_to_read: Vec<String> = match column_names {
            Some(names) => names.iter().map(|s| s.to_string()).collect(),
            None => schema.columns.iter().map(|(n, _)| n.clone()).collect(),
        };
        
        // If reading all columns and delta exists, also include delta-only columns
        if column_names.is_none() && delta_rows > 0 {
            if let Ok(Some((_delta_ids, delta_columns))) = self.read_delta_data() {
                for col_name in delta_columns.keys() {
                    if !col_names_to_read.contains(col_name) {
                        col_names_to_read.push(col_name.clone());
                    }
                }
            }
        }
        
        let col_indices: Vec<usize> = col_names_to_read
            .iter()
            .filter_map(|name| schema.get_index(name))
            .collect();

        // Calculate how many rows to read from base vs delta
        let base_start = actual_start.min(base_rows);
        let base_count = if actual_start < base_rows {
            actual_count.min(base_rows - actual_start)
        } else {
            0
        };
        
        // V4 fast path: read from in-memory columns (no mmap column index)
        // LOCK-FREE: use cached_footer_offset for V4 detection
        let v4_mode = column_index.is_empty() && self.cached_footer_offset.load(Ordering::Relaxed) > 0;
        drop(column_index);
        drop(schema);
        drop(header);
        
        let mut result = HashMap::new();
        
        if v4_mode && base_count > 0 && self.has_v4_in_memory_data() {
            // Use in-memory columns if available (write buffer path)
            
            let schema = self.schema.read();
            let columns = self.columns.read();
            for &col_idx in &col_indices {
                let (col_name, col_type) = &schema.columns[col_idx];
                if col_idx < columns.len() && columns[col_idx].len() > 0 {
                    result.insert(col_name.clone(),
                        columns[col_idx].slice_range(base_start, base_start + base_count));
                } else {
                    result.insert(col_name.clone(),
                        Self::create_default_column(*col_type, base_count));
                }
            }
        } else if v4_mode && base_count > 0 {
            // V4 MMAP PATH: scan columns from RGs via mmap with null bitmaps, then slice
            if let Some(footer) = self.get_or_load_footer()? {
                let f_schema = &footer.schema;
                let (scanned, _del, col_nulls) = self.scan_columns_mmap_with_nulls(&col_indices, &footer)?;
                for (out_pos, &col_idx) in col_indices.iter().enumerate() {
                    let (col_name, col_type) = &f_schema.columns[col_idx];
                    if out_pos < scanned.len() {
                        let mut col = scanned[out_pos].clone();
                        // Apply null bitmap: replace NULL marker strings with empty strings
                        if out_pos < col_nulls.len() && !col_nulls[out_pos].is_empty() {
                            col.apply_null_bitmap(&col_nulls[out_pos]);
                        }
                        let col_len = col.len();
                        if base_start == 0 && base_count >= col_len {
                            result.insert(col_name.clone(), col);
                        } else {
                            let end = (base_start + base_count).min(col_len);
                            if base_start < end {
                                let indices: Vec<usize> = (base_start..end).collect();
                                result.insert(col_name.clone(), col.filter_by_indices(&indices));
                            } else {
                                result.insert(col_name.clone(),
                                    Self::create_default_column(*col_type, base_count));
                            }
                        }
                    } else {
                        result.insert(col_name.clone(),
                            Self::create_default_column(*col_type, base_count));
                    }
                }
            }
        } else if base_count > 0 {
            // V3: read from mmap via column index
            let file_guard = self.file.read();
            let file = file_guard.as_ref().ok_or_else(|| {
                err_not_conn("File not open")
            })?;
            let mut mmap_cache = self.mmap_cache.write();
            let column_index = self.column_index.read();
            let schema = self.schema.read();
            
            // V4 files don't use column_index — return defaults if we reach here
            // (Primary V4 read path is to_arrow_batch_mmap, this is a safety fallback)
            if column_index.is_empty() {
                for &col_idx in &col_indices {
                    let (col_name, col_type) = &schema.columns[col_idx];
                    result.insert(col_name.clone(),
                        Self::create_default_column(*col_type, base_count));
                }
                return Ok(result);
            }
            
            for &col_idx in &col_indices {
                let (col_name, col_type) = &schema.columns[col_idx];
                let index_entry = &column_index[col_idx];
                
                let col_data = self.read_column_range_mmap(
                    &mut mmap_cache,
                    file,
                    index_entry,
                    *col_type,
                    base_start,
                    base_count,
                    base_rows,
                )?;
                
                result.insert(col_name.clone(), col_data);
            }
        }
        
        // Merge delta data if needed
        if delta_rows > 0 && actual_start + actual_count > base_rows {
            if let Some((_delta_ids, delta_columns)) = self.read_delta_data()? {
                let delta_start = if actual_start > base_rows { actual_start - base_rows } else { 0 };
                let delta_count = actual_count - base_count;
                let actual_delta_count = delta_count.min(delta_rows - delta_start);
                
                // Get schema to determine column types for padding
                let schema = self.schema.read();
                
                for col_name in &col_names_to_read {
                    if let Some(delta_col) = delta_columns.get(col_name) {
                        // Extract the range we need from delta
                        let delta_slice = if delta_start == 0 && delta_count >= delta_col.len() {
                            delta_col.clone()
                        } else {
                            let end = delta_start + delta_count.min(delta_col.len().saturating_sub(delta_start));
                            let indices: Vec<usize> = (delta_start..end).collect();
                            delta_col.filter_by_indices(&indices)
                        };
                        
                        // Check if column exists in base result
                        if let Some(base_col) = result.get_mut(col_name) {
                            // Column exists in base - append delta
                            base_col.append(&delta_slice);
                        } else {
                            // Column only exists in delta - need to pad base rows with defaults first
                            let col_type = delta_slice.column_type();
                            let mut padded = ColumnData::new(col_type);
                            
                            // Pad base rows with defaults (NULL/0/empty)
                            for _ in 0..base_count {
                                match &mut padded {
                                    ColumnData::Int64(v) => v.push(0),
                                    ColumnData::Float64(v) => v.push(0.0),
                                    ColumnData::String { offsets, .. } => offsets.push(*offsets.last().unwrap_or(&0)),
                                    ColumnData::Bool { data, len } => {
                                        let byte_idx = *len / 8;
                                        if byte_idx >= data.len() { data.push(0); }
                                        *len += 1;
                                    }
                                    _ => {}
                                }
                            }
                            
                            // Append delta data after padding
                            padded.append(&delta_slice);
                            result.insert(col_name.clone(), padded);
                        }
                    } else {
                        // Column exists in schema but not in delta - pad with defaults
                        if let Some(col_idx) = schema.get_index(col_name) {
                            let (_, col_type) = &schema.columns[col_idx];
                            let mut padding = ColumnData::new(*col_type);
                            // Pad with default values
                            for _ in 0..actual_delta_count {
                                match &mut padding {
                                    ColumnData::Int64(v) => v.push(0),
                                    ColumnData::Float64(v) => v.push(0.0),
                                    ColumnData::String { offsets, .. } => offsets.push(*offsets.last().unwrap_or(&0)),
                                    ColumnData::Bool { data, len } => {
                                        let byte_idx = *len / 8;
                                        if byte_idx >= data.len() { data.push(0); }
                                        *len += 1;
                                    }
                                    _ => {}
                                }
                            }
                            
                            if let Some(base_col) = result.get_mut(col_name) {
                                base_col.append(&padding);
                            } else {
                                result.insert(col_name.clone(), padding);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }

    /// Check if a specific row/column is NULL
    /// Returns true if the value at (row_idx, col_name) is NULL
    pub fn is_null(&self, row_idx: usize, col_name: &str) -> bool {
        let schema = self.schema.read();
        if let Some(col_idx) = schema.get_index(col_name) {
            let nulls = self.nulls.read();
            if col_idx < nulls.len() {
                let null_bitmap = &nulls[col_idx];
                let byte_idx = row_idx / 8;
                let bit_idx = row_idx % 8;
                if byte_idx < null_bitmap.len() {
                    return (null_bitmap[byte_idx] & (1 << bit_idx)) != 0;
                }
            }
        }
        false
    }

    /// Get null bitmap for a column (for Arrow conversion)
    /// Returns a Vec<bool> where true means the value is NULL
    /// Reads from file via mmap if not loaded in memory
    pub fn get_null_mask(&self, col_name: &str, start_row: usize, row_count: usize) -> Vec<bool> {
        // Only use in-memory nulls if available — mmap path handles nulls separately
        let schema = self.schema.read();
        let mut result = vec![false; row_count];
        
        if let Some(col_idx) = schema.get_index(col_name) {
            // First check in-memory nulls
            let nulls = self.nulls.read();
            if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                let null_bitmap = &nulls[col_idx];
                for i in 0..row_count {
                    let row_idx = start_row + i;
                    let byte_idx = row_idx / 8;
                    let bit_idx = row_idx % 8;
                    if byte_idx < null_bitmap.len() {
                        result[i] = (null_bitmap[byte_idx] & (1 << bit_idx)) != 0;
                    }
                }
            } else {
                drop(nulls);
                // V4 mmap path: read null bitmap from RG data
                let header = self.header.read();
                if header.footer_offset > 0 {
                    drop(header);
                    drop(schema);
                    if let Ok(Some(footer)) = self.get_or_load_footer() {
                        let f_schema = &footer.schema;
                        let f_col_count = f_schema.column_count();
                        if let Some(file_guard) = self.file.try_read() {
                            if let Some(file) = file_guard.as_ref() {
                                let mut mmap_guard = self.mmap_cache.write();
                                if let Ok(mmap_ref) = mmap_guard.get_or_create(file) {
                                    let mut global_row = 0usize;
                                    for rg_meta in &footer.row_groups {
                                        let rg_rows = rg_meta.row_count as usize;
                                        let rg_end = (rg_meta.offset + rg_meta.data_size) as usize;
                                        if rg_end > mmap_ref.len() { break; }
                                        let rg_bytes = &mmap_ref[rg_meta.offset as usize..rg_end];
                                        // Check compression flag at RG header byte 28, encoding version at byte 29
                                        let compress_flag = if rg_bytes.len() >= 32 { rg_bytes[28] } else { RG_COMPRESS_NONE };
                                        let encoding_version = if rg_bytes.len() >= 32 { rg_bytes[29] } else { 0 };
                                        let decompressed_buf: Option<Vec<u8>> = match decompress_rg_body(compress_flag, &rg_bytes[32..]) {
                                            Ok(v) => v,
                                            Err(_) => break,
                                        };
                                        let body: &[u8] = decompressed_buf.as_deref().unwrap_or(&rg_bytes[32..]);
                                        let mut pos: usize = rg_rows * 8; // skip IDs
                                        let del_vec_len = (rg_rows + 7) / 8;
                                        pos += del_vec_len; // skip deletion vector
                                        let null_bitmap_len = (rg_rows + 7) / 8;
                                        // Navigate to the right column's null bitmap
                                        for ci in 0..f_col_count {
                                            if pos + null_bitmap_len > body.len() { break; }
                                            let null_bytes = &body[pos..pos + null_bitmap_len];
                                            pos += null_bitmap_len;
                                            if ci == col_idx {
                                                // Extract null bits for rows in range
                                                for ri in 0..rg_rows {
                                                    let global_ri = global_row + ri;
                                                    if global_ri >= start_row && global_ri < start_row + row_count {
                                                        let out_i = global_ri - start_row;
                                                        let b = ri / 8; let bit = ri % 8;
                                                        if b < null_bytes.len() && (null_bytes[b] >> bit) & 1 != 0 {
                                                            result[out_i] = true;
                                                        }
                                                    }
                                                }
                                            }
                                            // Skip column data (encoding-aware)
                                            let ct = f_schema.columns[ci].1;
                                            let consumed = if encoding_version >= 1 {
                                                skip_column_encoded(&body[pos..], ct)
                                            } else {
                                                ColumnData::skip_bytes_typed(&body[pos..], ct)
                                            };
                                            if let Ok(c) = consumed {
                                                pos += c;
                                            } else { break; }
                                        }
                                        global_row += rg_rows;
                                    }
                                }
                            }
                        }
                    }
                    return result;
                }
                drop(header);
                // V3: Read null bitmap from file via column index
                let column_index = self.column_index.read();
                if col_idx < column_index.len() {
                    let index_entry = &column_index[col_idx];
                    let null_len = index_entry.null_length as usize;
                    if null_len > 0 {
                        if let Some(file) = self.file.read().as_ref() {
                            let mut mmap_cache = self.mmap_cache.write();
                            let mut null_bitmap = vec![0u8; null_len];
                            if mmap_cache.read_at(file, &mut null_bitmap, index_entry.null_offset).is_ok() {
                                for i in 0..row_count {
                                    let row_idx = start_row + i;
                                    let byte_idx = row_idx / 8;
                                    let bit_idx = row_idx % 8;
                                    if byte_idx < null_bitmap.len() {
                                        result[i] = (null_bitmap[byte_idx] & (1 << bit_idx)) != 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        result
    }

    /// Read columns with predicate pushdown - filter rows at storage level
    /// This avoids loading rows that don't match the filter condition
    /// 
    /// # Arguments
    /// * `column_names` - Columns to read (None = all columns)
    /// * `filter_column` - Column name to filter on
    /// * `filter_op` - Comparison operator: ">=", ">", "<=", "<", "=", "!="
    /// * `filter_value` - Value to compare against (i64 or f64)
    pub fn read_columns_filtered(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_op: &str,
        filter_value: f64,
    ) -> io::Result<(HashMap<String, ColumnData>, Vec<usize>)> {
        let is_v4 = self.is_v4_format();
        // V4 without in-memory data: use mmap path
        if is_v4 && !self.has_v4_in_memory_data() {
            return self.read_columns_filtered_mmap(column_names, filter_column, filter_op, filter_value);
        }
        
        let header = self.header.read();
        let schema = self.schema.read();
        
        let total_rows = header.row_count as usize;
        if total_rows == 0 {
            return Ok((HashMap::new(), Vec::new()));
        }

        // First, read the filter column to determine matching rows
        let filter_col_idx = schema.get_index(filter_column).ok_or_else(|| {
            err_not_found(format!("Filter column: {}", filter_column))
        })?;
        
        let (_, filter_col_type) = &schema.columns[filter_col_idx];
        let filter_col_type = *filter_col_type;
        drop(schema);
        drop(header);

        // Read filter column data
        let filter_data = self.read_column_auto(filter_col_idx, filter_col_type, 0, total_rows, total_rows, is_v4)?;
        
        // Apply filter and collect matching row indices
        let matching_indices: Vec<usize> = match &filter_data {
            ColumnData::Int64(values) => {
                let filter_val = filter_value as i64;
                values.iter().enumerate()
                    .filter(|(_, &v)| match filter_op {
                        ">=" => v >= filter_val,
                        ">" => v > filter_val,
                        "<=" => v <= filter_val,
                        "<" => v < filter_val,
                        "=" | "==" => v == filter_val,
                        "!=" | "<>" => v != filter_val,
                        _ => true,
                    })
                    .map(|(i, _)| i)
                    .collect()
            }
            ColumnData::Float64(values) => {
                values.iter().enumerate()
                    .filter(|(_, &v)| match filter_op {
                        ">=" => v >= filter_value,
                        ">" => v > filter_value,
                        "<=" => v <= filter_value,
                        "<" => v < filter_value,
                        "=" | "==" => (v - filter_value).abs() < f64::EPSILON,
                        "!=" | "<>" => (v - filter_value).abs() >= f64::EPSILON,
                        _ => true,
                    })
                    .map(|(i, _)| i)
                    .collect()
            }
            _ => (0..total_rows).collect(),
        };

        if matching_indices.is_empty() {
            return Ok((HashMap::new(), Vec::new()));
        }

        // Determine which columns to read
        let schema = self.schema.read();
        let col_indices: Vec<usize> = match column_names {
            Some(names) => names
                .iter()
                .filter_map(|name| schema.get_index(name))
                .collect(),
            None => (0..schema.column_count()).collect(),
        };

        let mut result = HashMap::new();

        // Read only matching rows for each column
        for &col_idx in &col_indices {
            let (col_name, col_type) = &schema.columns[col_idx];
            
            // OPTIMIZATION: Skip reading filter column again - reuse already-read data
            if col_idx == filter_col_idx {
                let filtered_data = match &filter_data {
                    ColumnData::Int64(values) => {
                        ColumnData::Int64(matching_indices.iter().map(|&i| values[i]).collect())
                    }
                    ColumnData::Float64(values) => {
                        ColumnData::Float64(matching_indices.iter().map(|&i| values[i]).collect())
                    }
                    other => other.clone(),
                };
                result.insert(col_name.clone(), filtered_data);
                continue;
            }
            
            let col_data = self.read_column_scattered_auto(col_idx, *col_type, &matching_indices, total_rows, is_v4)?;
            result.insert(col_name.clone(), col_data);
        }

        Ok((result, matching_indices))
    }

    /// MMAP PATH: numeric filter directly on V4 RG data via mmap
    /// Uses per-RG zone maps to skip Row Groups where filter can't match.
    fn read_columns_filtered_mmap(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_op: &str,
        filter_value: f64,
    ) -> io::Result<(HashMap<String, ColumnData>, Vec<usize>)> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok((HashMap::new(), Vec::new())),
        };
        let schema = &footer.schema;
        
        let filter_col_idx = match schema.get_index(filter_column) {
            Some(idx) => idx,
            None => return Ok((HashMap::new(), Vec::new())),
        };
        
        // Per-RG zone map pruning: build set of RG indices to skip
        let skip_rgs = Self::zone_map_prune_rgs(&footer, filter_col_idx, filter_op, filter_value);
        
        // Scan filter column from mmap (skipping pruned RGs)
        let (scanned, del_bytes) = self.scan_columns_mmap_skip_rgs(&[filter_col_idx], &footer, &skip_rgs)?;
        if scanned.is_empty() { return Ok((HashMap::new(), Vec::new())); }
        
        let has_deleted = del_bytes.iter().any(|&b| b != 0);
        
        let matching_indices: Vec<usize> = match &scanned[0] {
            ColumnData::Int64(values) => {
                let fv = filter_value as i64;
                values.iter().enumerate()
                    .filter(|(i, _)| {
                        if has_deleted { let b = i / 8; let bit = i % 8; if b < del_bytes.len() && (del_bytes[b] >> bit) & 1 != 0 { return false; } }
                        true
                    })
                    .filter(|(_, &v)| match filter_op {
                        ">=" => v >= fv, ">" => v > fv, "<=" => v <= fv, "<" => v < fv,
                        "=" | "==" => v == fv, "!=" | "<>" => v != fv, _ => true,
                    })
                    .map(|(i, _)| i)
                    .collect()
            }
            ColumnData::Float64(values) => {
                values.iter().enumerate()
                    .filter(|(i, _)| {
                        if has_deleted { let b = i / 8; let bit = i % 8; if b < del_bytes.len() && (del_bytes[b] >> bit) & 1 != 0 { return false; } }
                        true
                    })
                    .filter(|(_, &v)| match filter_op {
                        ">=" => v >= filter_value, ">" => v > filter_value,
                        "<=" => v <= filter_value, "<" => v < filter_value,
                        "=" | "==" => (v - filter_value).abs() < f64::EPSILON,
                        "!=" | "<>" => (v - filter_value).abs() >= f64::EPSILON, _ => true,
                    })
                    .map(|(i, _)| i)
                    .collect()
            }
            _ => return Ok((HashMap::new(), Vec::new())),
        };
        
        if matching_indices.is_empty() {
            return Ok((HashMap::new(), Vec::new()));
        }
        
        // Scan result columns (skip same pruned RGs)
        let col_indices: Vec<usize> = match column_names {
            Some(names) => names.iter()
                .filter(|&&n| n != "_id")
                .filter_map(|&name| schema.get_index(name))
                .collect(),
            None => (0..schema.column_count()).collect(),
        };
        
        let (all_cols, _) = self.scan_columns_mmap_skip_rgs(&col_indices, &footer, &skip_rgs)?;
        
        let mut result = HashMap::new();
        for (out_idx, &col_idx) in col_indices.iter().enumerate() {
            let (col_name, _) = &schema.columns[col_idx];
            if out_idx < all_cols.len() {
                result.insert(col_name.clone(), all_cols[out_idx].filter_by_indices(&matching_indices));
            }
        }
        
        Ok((result, matching_indices))
    }

    /// Read columns with STRING predicate pushdown - filter rows at storage level
    /// This is optimized for string equality filters (column = 'value')
    /// Uses bloom filters to skip row groups that definitely don't contain the value
    pub fn read_columns_filtered_string(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_value: &str,
        filter_eq: bool,  // true = equals, false = not equals
    ) -> io::Result<(HashMap<String, ColumnData>, Vec<usize>)> {
        // V4 FAST PATH: scan in-memory columns directly (no disk I/O)
        // LOCK-FREE: use cached_footer_offset for V4 detection
        {
            if self.cached_footer_offset.load(Ordering::Relaxed) > 0 {
                if self.has_v4_in_memory_data() {
                    // In-memory fast path
                    let schema = self.schema.read();
                    let columns = self.columns.read();
                    let deleted = self.deleted.read();
                    let total_rows = self.ids.read().len();
                    
                    let filter_col_idx = match schema.get_index(filter_column) {
                        Some(idx) => idx,
                        None => return Ok((HashMap::new(), Vec::new())),
                    };
                    
                    let filter_bytes = filter_value.as_bytes();
                    let filter_len = filter_bytes.len();
                    let has_deleted = deleted.iter().any(|&b| b != 0);
                    
                    let mut matching_indices: Vec<usize> = Vec::with_capacity(1024);
                    
                    if filter_col_idx < columns.len() {
                        if let ColumnData::String { offsets, data } = &columns[filter_col_idx] {
                            let count = offsets.len().saturating_sub(1).min(total_rows);
                            let first_byte = filter_bytes.first().copied();
                            
                            for i in 0..count {
                                if has_deleted {
                                    let byte_idx = i / 8;
                                    let bit_idx = i % 8;
                                    if byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 != 0 {
                                        continue;
                                    }
                                }
                                
                                let start = offsets[i] as usize;
                                let end = offsets[i + 1] as usize;
                                let str_len = end - start;
                                
                                if str_len != filter_len {
                                    if !filter_eq { matching_indices.push(i); }
                                    continue;
                                }
                                
                                if let Some(fb) = first_byte {
                                    if start < data.len() && data[start] != fb {
                                        if !filter_eq { matching_indices.push(i); }
                                        continue;
                                    }
                                }
                                
                                let matches = end <= data.len() && &data[start..end] == filter_bytes;
                                if (filter_eq && matches) || (!filter_eq && !matches) {
                                    matching_indices.push(i);
                                }
                            }
                        }
                    }
                    
                    if matching_indices.is_empty() {
                        return Ok((HashMap::new(), Vec::new()));
                    }
                    
                    let col_indices: Vec<usize> = match column_names {
                        Some(names) => names.iter()
                            .filter(|&&n| n != "_id")
                            .filter_map(|&name| schema.get_index(name))
                            .collect(),
                        None => (0..schema.column_count()).collect(),
                    };
                    
                    let mut result = HashMap::new();
                    for &col_idx in &col_indices {
                        let (col_name, _) = &schema.columns[col_idx];
                        if col_idx < columns.len() {
                            result.insert(col_name.clone(), columns[col_idx].filter_by_indices(&matching_indices));
                        }
                    }
                    
                    return Ok((result, matching_indices));
                }
                
                // MMAP PATH: scan string filter column from V4 RGs
                return self.read_columns_filtered_string_mmap(
                    column_names, filter_column, filter_value, filter_eq,
                );
            }
        }
        
        // V3 SLOW PATH: read from disk
        let is_v4 = self.is_v4_format();
        
        let header = self.header.read();
        let schema = self.schema.read();
        
        let total_rows = header.row_count as usize;
        if total_rows == 0 {
            return Ok((HashMap::new(), Vec::new()));
        }

        // Find filter column
        let filter_col_idx = schema.get_index(filter_column).ok_or_else(|| {
            err_not_found(format!("Filter column: {}", filter_column))
        })?;
        
        let (_, filter_col_type) = &schema.columns[filter_col_idx];
        let filter_col_type = *filter_col_type;
        
        // Only works for string columns (including dictionary-encoded)
        if !matches!(filter_col_type, ColumnType::String | ColumnType::StringDict) {
            return Err(err_input("String filter requires string column"));
        }
        
        drop(schema);
        drop(header);

        // Read filter column data
        let filter_data = self.read_column_auto(filter_col_idx, filter_col_type, 0, total_rows, total_rows, is_v4)?;
        
        // OPTIMIZATION: Build and use bloom filter for large datasets
        // Build bloom filter on-the-fly and use it to identify candidate row groups
        let filter_bytes = filter_value.as_bytes();
        let use_bloom = filter_eq && total_rows > 10000; // Only use bloom for equality on large datasets
        
        // Apply string filter
        let matching_indices: Vec<usize> = match filter_data {
            ColumnData::String { offsets, data } => {
                let count = offsets.len().saturating_sub(1);
                let filter_len = filter_bytes.len() as u32;
                
                // OPTIMIZATION: Pre-compute first byte and length for fast rejection
                let first_byte = filter_bytes.first().copied();
                
                // Pre-allocate with estimated capacity (assume ~10% match rate)
                let mut result = Vec::with_capacity(count / 10 + 1);
                
                // OPTIMIZATION: Use bloom filter to skip row groups for large datasets
                const ROW_GROUP_SIZE: usize = 8192;  // 8K rows per group for bloom filter
                
                if use_bloom && count > ROW_GROUP_SIZE {
                    // Build bloom filter index and identify candidate groups
                    use crate::storage::bloom::{ColumnBloomIndex, BLOOM_FP_RATE};
                    let bloom_index = ColumnBloomIndex::build_from_strings(
                        filter_column,
                        &offsets,
                        &data,
                        ROW_GROUP_SIZE,
                        BLOOM_FP_RATE,
                    );
                    
                    // Get row ranges that might contain the value
                    let scan_ranges = bloom_index.get_scan_ranges(filter_bytes);
                    
                    // Only scan candidate row groups
                    for (group_start, group_end) in scan_ranges {
                        for i in group_start..group_end.min(count) {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            let str_len = (end - start) as u32;
                            
                            // Fast path: length mismatch rejection
                            if str_len != filter_len {
                                continue;
                            }
                            
                            // Fast path: first byte mismatch
                            if let Some(fb) = first_byte {
                                if start < data.len() && data[start] != fb {
                                    continue;
                                }
                            }
                            
                            // Full comparison
                            let matches = end <= data.len() && &data[start..end] == filter_bytes;
                            if matches {
                                result.push(i);
                            }
                        }
                    }
                } else {
                    // Standard chunked processing for small datasets or != filter
                    const CHUNK_SIZE: usize = 1024;
                    for chunk_start in (0..count).step_by(CHUNK_SIZE) {
                        let chunk_end = (chunk_start + CHUNK_SIZE).min(count);
                        
                        for i in chunk_start..chunk_end {
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            let str_len = (end - start) as u32;
                            
                            // Fast path: length mismatch rejection (most common case)
                            if str_len != filter_len {
                                if !filter_eq {
                                    result.push(i);
                                }
                                continue;
                            }
                            
                            // Fast path: first byte mismatch (catches ~255/256 of remaining)
                            if let Some(fb) = first_byte {
                                if start < data.len() && data[start] != fb {
                                    if !filter_eq {
                                        result.push(i);
                                    }
                                    continue;
                                }
                            }
                            
                            // Full comparison only when length and first byte match
                            let matches = end <= data.len() && &data[start..end] == filter_bytes;
                            if filter_eq == matches {
                                result.push(i);
                            }
                        }
                    }
                }
                result
            }
            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                // OPTIMIZATION: Find matching dictionary index first, then scan indices
                // This is O(dict_size + row_count) vs O(row_count * string_len)
                let filter_bytes = filter_value.as_bytes();
                let filter_len = filter_bytes.len();
                let mut matching_dict_idx: Option<u32> = None;
                
                // Find which dictionary entry matches the filter value (with fast rejection)
                let dict_count = dict_offsets.len().saturating_sub(1);
                for i in 0..dict_count {
                    let start = dict_offsets[i] as usize;
                    let end = if i + 1 < dict_offsets.len() { dict_offsets[i + 1] as usize } else { dict_data.len() };
                    // Fast rejection by length
                    if end - start != filter_len {
                        continue;
                    }
                    if end <= dict_data.len() && &dict_data[start..end] == filter_bytes {
                        matching_dict_idx = Some((i + 1) as u32); // +1 because 0 = NULL
                        break;
                    }
                }
                
                // OPTIMIZATION: Pre-allocate and use pointer-based scan for speed
                let count = indices.len();
                let mut result = Vec::with_capacity(count / 10 + 1);
                
                match (matching_dict_idx, filter_eq) {
                    (Some(target_idx), true) => {
                        // SIMD-friendly: scan in chunks with pointer arithmetic
                        let ptr = indices.as_ptr();
                        for i in 0..count {
                            // Pointer dereference avoids bounds checking
                            if unsafe { *ptr.add(i) } == target_idx {
                                result.push(i);
                            }
                        }
                    }
                    (Some(target_idx), false) => {
                        let ptr = indices.as_ptr();
                        for i in 0..count {
                            let idx = unsafe { *ptr.add(i) };
                            if idx != target_idx && idx != 0 {
                                result.push(i);
                            }
                        }
                    }
                    (None, true) => {
                        // Value not in dictionary, no matches for equality
                        // result stays empty
                    }
                    (None, false) => {
                        // Value not in dictionary, all non-NULL rows match
                        let ptr = indices.as_ptr();
                        for i in 0..count {
                            if unsafe { *ptr.add(i) } != 0 {
                                result.push(i);
                            }
                        }
                    }
                }
                result
            }
            _ => return Err(err_input("Expected string column")),
        };

        if matching_indices.is_empty() {
            return Ok((HashMap::new(), Vec::new()));
        }

        // Read only matching rows for each requested column
        let schema = self.schema.read();
        let col_indices: Vec<usize> = match column_names {
            Some(names) => names
                .iter()
                .filter_map(|name| schema.get_index(name))
                .collect(),
            None => (0..schema.column_count()).collect(),
        };

        let mut result = HashMap::new();
        for &col_idx in &col_indices {
            let (col_name, col_type) = &schema.columns[col_idx];
            let col_data = self.read_column_scattered_auto(col_idx, *col_type, &matching_indices, total_rows, is_v4)?;
            result.insert(col_name.clone(), col_data);
        }

        Ok((result, matching_indices))
    }
    
    /// MMAP PATH: string filter directly on V4 RG data via mmap
    fn read_columns_filtered_string_mmap(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_value: &str,
        filter_eq: bool,
    ) -> io::Result<(HashMap<String, ColumnData>, Vec<usize>)> {
        let footer = match self.get_or_load_footer()? {
            Some(f) => f,
            None => return Ok((HashMap::new(), Vec::new())),
        };
        let schema = &footer.schema;
        
        let filter_col_idx = match schema.get_index(filter_column) {
            Some(idx) => idx,
            None => return Ok((HashMap::new(), Vec::new())),
        };
        
        // Scan filter column from mmap
        let (scanned, del_bytes) = self.scan_columns_mmap(&[filter_col_idx], &footer)?;
        if scanned.is_empty() { return Ok((HashMap::new(), Vec::new())); }
        
        let has_deleted = del_bytes.iter().any(|&b| b != 0);
        let filter_bytes = filter_value.as_bytes();
        let filter_len = filter_bytes.len();
        let first_byte = filter_bytes.first().copied();
        
        let mut matching_indices: Vec<usize> = Vec::with_capacity(1024);
        
        match &scanned[0] {
            ColumnData::String { offsets, data } => {
                let count = offsets.len().saturating_sub(1);
                for i in 0..count {
                    if has_deleted {
                        let b = i / 8; let bit = i % 8;
                        if b < del_bytes.len() && (del_bytes[b] >> bit) & 1 != 0 { continue; }
                    }
                    let start = offsets[i] as usize;
                    let end = offsets[i + 1] as usize;
                    let str_len = end - start;
                    
                    if str_len != filter_len {
                        if !filter_eq { matching_indices.push(i); }
                        continue;
                    }
                    if let Some(fb) = first_byte {
                        if start < data.len() && data[start] != fb {
                            if !filter_eq { matching_indices.push(i); }
                            continue;
                        }
                    }
                    let matches = end <= data.len() && &data[start..end] == filter_bytes;
                    if (filter_eq && matches) || (!filter_eq && !matches) {
                        matching_indices.push(i);
                    }
                }
            }
            ColumnData::StringDict { indices, dict_offsets, dict_data } => {
                // Pre-compute which dict entries (1-based) match the filter value.
                // Entry 0 means null/empty; real entries start at index 1.
                let dict_len = dict_offsets.len().saturating_sub(1);
                let matching_dict: Vec<bool> = (0..dict_len).map(|i| {
                    let start = dict_offsets[i] as usize;
                    let end = dict_offsets[i + 1] as usize;
                    let s = if end <= dict_data.len() { &dict_data[start..end] } else { &[] };
                    let m = s == filter_bytes;
                    if filter_eq { m } else { !m }
                }).collect();
                for (i, &dict_idx) in indices.iter().enumerate() {
                    if has_deleted {
                        let b = i / 8; let bit = i % 8;
                        if b < del_bytes.len() && (del_bytes[b] >> bit) & 1 != 0 { continue; }
                    }
                    // dict_idx is 1-based; 0 = null
                    let matches = dict_idx > 0 && {
                        let actual = (dict_idx - 1) as usize;
                        actual < matching_dict.len() && matching_dict[actual]
                    };
                    if matches { matching_indices.push(i); }
                }
            }
            _ => return Ok((HashMap::new(), Vec::new())),
        }
        
        if matching_indices.is_empty() {
            return Ok((HashMap::new(), Vec::new()));
        }
        
        // Now scan the result columns for matching rows
        let col_indices: Vec<usize> = match column_names {
            Some(names) => names.iter()
                .filter(|&&n| n != "_id")
                .filter_map(|&name| schema.get_index(name))
                .collect(),
            None => (0..schema.column_count()).collect(),
        };
        
        let (all_cols, _) = self.scan_columns_mmap(&col_indices, &footer)?;
        
        let mut result = HashMap::new();
        for (out_idx, &col_idx) in col_indices.iter().enumerate() {
            let (col_name, _) = &schema.columns[col_idx];
            if out_idx < all_cols.len() {
                result.insert(col_name.clone(), all_cols[out_idx].filter_by_indices(&matching_indices));
            }
        }
        
        Ok((result, matching_indices))
    }

    /// Read columns with STRING predicate pushdown and early termination for LIMIT
    /// Stops scanning once we have enough matching rows - much faster for LIMIT queries
    pub fn read_columns_filtered_string_with_limit(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        filter_value: &str,
        filter_eq: bool,
        limit: usize,
        offset: usize,
    ) -> io::Result<(HashMap<String, ColumnData>, Vec<usize>)> {
        // V4: scan in-memory columns directly with early termination
        // LOCK-FREE: use cached_footer_offset for V4 detection
        {
            if self.cached_footer_offset.load(Ordering::Relaxed) > 0 {
                if !self.has_v4_in_memory_data() {
                    // MMAP PATH: delegate to non-limit mmap scan, then apply offset+limit
                    let (mut result, mut indices) = self.read_columns_filtered_string_mmap(
                        column_names, filter_column, filter_value, filter_eq,
                    )?;
                    if indices.len() > offset {
                        let end = (offset + limit).min(indices.len());
                        let sliced_indices: Vec<usize> = indices[offset..end].to_vec();
                        // Re-filter result columns to only keep the offset+limit slice
                        let keep: Vec<usize> = (offset..end).collect();
                        for (_name, col) in result.iter_mut() {
                            *col = col.filter_by_indices(&keep);
                        }
                        indices = sliced_indices;
                    } else {
                        return Ok((HashMap::new(), Vec::new()));
                    }
                    return Ok((result, indices));
                }
                
                let schema = self.schema.read();
                let columns = self.columns.read();
                let deleted = self.deleted.read();
                let total_rows = self.ids.read().len();
                
                let filter_col_idx = match schema.get_index(filter_column) {
                    Some(idx) => idx,
                    None => return Ok((HashMap::new(), Vec::new())),
                };
                
                let filter_bytes = filter_value.as_bytes();
                let filter_len = filter_bytes.len();
                let needed = offset + limit;
                let has_deleted = deleted.iter().any(|&b| b != 0);
                
                // Scan filter column with early termination
                let mut matching_indices: Vec<usize> = Vec::with_capacity(needed.min(1024));
                
                if filter_col_idx < columns.len() {
                    if let ColumnData::String { offsets, data } = &columns[filter_col_idx] {
                        let count = offsets.len().saturating_sub(1).min(total_rows);
                        for i in 0..count {
                            if matching_indices.len() >= needed { break; }
                            
                            // Skip deleted rows
                            if has_deleted {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                if byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 != 0 {
                                    continue;
                                }
                            }
                            
                            let start = offsets[i] as usize;
                            let end = offsets[i + 1] as usize;
                            let str_len = end - start;
                            
                            // Fast rejection by length
                            if str_len != filter_len { 
                                if filter_eq { continue; } 
                                else { matching_indices.push(i); continue; }
                            }
                            
                            let matches = end <= data.len() && &data[start..end] == filter_bytes;
                            if (filter_eq && matches) || (!filter_eq && !matches) {
                                matching_indices.push(i);
                            }
                        }
                    }
                }
                
                // Apply offset
                if offset > 0 && offset < matching_indices.len() {
                    matching_indices = matching_indices[offset..].to_vec();
                } else if offset >= matching_indices.len() {
                    matching_indices.clear();
                }
                if matching_indices.len() > limit {
                    matching_indices.truncate(limit);
                }
                
                if matching_indices.is_empty() {
                    return Ok((HashMap::new(), Vec::new()));
                }
                
                // Read only needed columns for matching indices
                let col_indices: Vec<usize> = match column_names {
                    Some(names) => names.iter()
                        .filter(|&&n| n != "_id")
                        .filter_map(|&name| schema.get_index(name))
                        .collect(),
                    None => (0..schema.column_count()).collect(),
                };
                
                let mut result = HashMap::new();
                for &col_idx in &col_indices {
                    let (col_name, _) = &schema.columns[col_idx];
                    if col_idx < columns.len() {
                        result.insert(col_name.clone(), columns[col_idx].filter_by_indices(&matching_indices));
                    }
                }
                
                return Ok((result, matching_indices));
            }
        }
        
        let header = self.header.read();
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        
        let total_rows = header.row_count as usize;
        if total_rows == 0 {
            return Ok((HashMap::new(), Vec::new()));
        }

        let filter_col_idx = schema.get_index(filter_column).ok_or_else(|| {
            err_not_found(format!("Filter column: {}", filter_column))
        })?;
        
        let (_, filter_col_type) = &schema.columns[filter_col_idx];
        let filter_index = &column_index[filter_col_idx];
        
        if !matches!(filter_col_type, ColumnType::String | ColumnType::StringDict) {
            return Err(err_input("String filter requires string column"));
        }
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            err_not_conn("File not open")
        })?;
        
        let mut mmap_cache = self.mmap_cache.write();
        let needed = offset + limit;

        // For dictionary-encoded strings, use fast integer key scan with early termination
        // Format: [row_count:u64][dict_size:u64][indices:u32*row_count][dict_offsets:u32*dict_size][dict_data_len:u64][dict_data]
        if *filter_col_type == ColumnType::StringDict {
            let base_offset = filter_index.data_offset;
            
            // Read header: [row_count:u64][dict_size:u64]
            let mut header = [0u8; 16];
            mmap_cache.read_at(file, &mut header, base_offset)?;
            let stored_rows = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
            let dict_size = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
            
            if stored_rows == 0 || dict_size == 0 {
                return Ok((HashMap::new(), Vec::new()));
            }
            
            // Read dict_offsets
            let dict_offsets_offset = base_offset + 16 + (stored_rows * 4) as u64;
            let mut dict_offsets_buf = vec![0u8; dict_size * 4];
            mmap_cache.read_at(file, &mut dict_offsets_buf, dict_offsets_offset)?;
            
            let mut dict_offsets = Vec::with_capacity(dict_size);
            for i in 0..dict_size {
                dict_offsets.push(u32::from_le_bytes(dict_offsets_buf[i * 4..(i + 1) * 4].try_into().unwrap()));
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
            
            // Find target key in dictionary
            // dict_offsets[i] gives start of string i, dict_offsets[i+1] or dict_data_len gives end
            let filter_bytes = filter_value.as_bytes();
            let mut target_key: Option<u32> = None;
            let dict_count = dict_size.saturating_sub(1);
            
            // Linear search for dictionary lookup (small dictionaries are common)
            for i in 0..dict_count {
                let start = dict_offsets[i] as usize;
                let end = if i + 1 < dict_size { dict_offsets[i + 1] as usize } else { dict_data_len };
                if end <= dict_data.len() && start <= end && &dict_data[start..end] == filter_bytes {
                    target_key = Some((i + 1) as u32);
                    break;
                }
            }
            
            let target_key = match (target_key, filter_eq) {
                (Some(k), true) => k,
                (None, true) => return Ok((HashMap::new(), Vec::new())),
                _ => return self.read_columns_filtered_string(column_names, filter_column, filter_value, filter_eq),
            };
            
            // Stream through indices with early termination - OPTIMIZED with pointer arithmetic
            let indices_offset = base_offset + 16;
            let mut matching_indices = Vec::with_capacity(needed.min(1000));
            
            // Read indices in larger chunks for better throughput
            const CHUNK_SIZE: usize = 8192;
            let mut chunk_buf = vec![0u32; CHUNK_SIZE];
            let mut row = 0usize;
            
            while row < stored_rows && matching_indices.len() < needed {
                let chunk_rows = CHUNK_SIZE.min(stored_rows - row);
                let chunk_bytes = unsafe {
                    std::slice::from_raw_parts_mut(chunk_buf.as_mut_ptr() as *mut u8, chunk_rows * 4)
                };
                mmap_cache.read_at(file, chunk_bytes, indices_offset + (row * 4) as u64)?;
                
                // OPTIMIZED: Use pointer arithmetic for faster scanning
                let buf_ptr = chunk_buf.as_ptr();
                for i in 0..chunk_rows {
                    // unsafe pointer dereference avoids bounds check
                    if unsafe { *buf_ptr.add(i) } == target_key {
                        matching_indices.push(row + i);
                        if matching_indices.len() >= needed {
                            break;
                        }
                    }
                }
                row += chunk_rows;
            }
            
            // Apply offset
            let final_indices: Vec<usize> = matching_indices.into_iter().skip(offset).take(limit).collect();
            
            if final_indices.is_empty() {
                return Ok((HashMap::new(), Vec::new()));
            }
            
            // Read columns for matching rows - SIMPLIFIED approach without sorting
            let col_indices: Vec<usize> = match column_names {
                Some(names) => names.iter().filter_map(|name| schema.get_index(name)).collect(),
                None => (0..schema.column_count()).collect(),
            };
            
            // OPTIMIZATION: Read columns directly without sorting
            // The overhead of sorting may not be worth it for small result sets
            let mut result: HashMap<String, ColumnData> = HashMap::with_capacity(col_indices.len());
            for &col_idx in &col_indices {
                let (col_name, col_type) = &schema.columns[col_idx];
                let index_entry = &column_index[col_idx];
                let col_data = self.read_column_scattered_mmap(&mut mmap_cache, file, index_entry, *col_type, &final_indices, total_rows)?;
                result.insert(col_name.clone(), col_data);
            }
            
            return Ok((result, final_indices));
        }
        
        // Fallback to regular method for non-dictionary strings
        self.read_columns_filtered_string(column_names, filter_column, filter_value, filter_eq)
    }

    /// Build a lazy cache of `string_value -> first active row id` for high-cardinality
    /// equality lookups, enabling near-random-access performance for `LIMIT 1`.
    pub fn build_first_string_row_id_cache(
        &self,
        col_name: &str,
    ) -> io::Result<Option<ahash::AHashMap<Box<str>, u64>>> {
        if self.has_pending_deltas() || self.delta_row_count() > 0 {
            return Ok(None);
        }

        let schema = self.schema.read();
        let col_idx = match schema.get_index(col_name) {
            Some(idx) => idx,
            None => return Ok(None),
        };
        let col_type = schema.columns[col_idx].1;
        if !matches!(col_type, ColumnType::String | ColumnType::StringDict) {
            return Ok(None);
        }
        drop(schema);

        let row_ids = self.read_ids(0, None)?;
        if row_ids.is_empty() {
            return Ok(Some(ahash::AHashMap::new()));
        }

        let (mut column, deleted, packed_nulls, bool_nulls): (
            ColumnData,
            Vec<u8>,
            Vec<u8>,
            Option<Vec<bool>>,
        ) = if self.is_v4_format() && !self.has_v4_in_memory_data() {
            let footer = match self.get_or_load_footer()? {
                Some(footer) => footer,
                None => return Ok(Some(ahash::AHashMap::new())),
            };
            let (cols, deleted, nulls) = self.scan_columns_mmap_with_nulls(&[col_idx], &footer)?;
            let column = cols
                .into_iter()
                .next()
                .unwrap_or_else(|| ColumnData::new(ColumnType::String));
            let packed_nulls = nulls.into_iter().next().unwrap_or_default();
            (column, deleted, packed_nulls, None)
        } else {
            let deleted = self.deleted.read().clone();
            let packed_nulls = {
                let nulls = self.nulls.read();
                if col_idx < nulls.len() && !nulls[col_idx].is_empty() {
                    Some(nulls[col_idx].clone())
                } else {
                    None
                }
            };
            let column = {
                let columns = self.columns.read();
                if col_idx < columns.len() && columns[col_idx].len() > 0 {
                    Some(columns[col_idx].clone())
                } else {
                    None
                }
            }
            .or_else(|| self.read_columns(Some(&[col_name]), 0, None).ok()?.remove(col_name))
            .unwrap_or_else(|| ColumnData::new(ColumnType::String));
            let bool_nulls = if packed_nulls.is_none() {
                Some(self.get_null_mask(col_name, 0, row_ids.len()))
            } else {
                None
            };
            (column, deleted, packed_nulls.unwrap_or_default(), bool_nulls)
        };

        if matches!(column, ColumnData::StringDict { .. }) {
            column = column.decode_string_dict();
        }

        let ColumnData::String { offsets, data } = column else {
            return Ok(None);
        };

        let count = offsets.len().saturating_sub(1).min(row_ids.len());
        let has_deleted = deleted.iter().any(|&b| b != 0);
        let has_packed_nulls = !packed_nulls.is_empty();
        let bool_nulls = bool_nulls.as_deref();
        let mut cache = ahash::AHashMap::with_capacity(count);

        for row_idx in 0..count {
            if has_deleted {
                let byte_idx = row_idx / 8;
                let bit_idx = row_idx % 8;
                if byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 != 0 {
                    continue;
                }
            }

            if has_packed_nulls {
                let byte_idx = row_idx / 8;
                let bit_idx = row_idx % 8;
                if byte_idx < packed_nulls.len() && (packed_nulls[byte_idx] >> bit_idx) & 1 != 0 {
                    continue;
                }
            } else if let Some(nulls) = bool_nulls {
                if row_idx < nulls.len() && nulls[row_idx] {
                    continue;
                }
            }

            let start = offsets[row_idx] as usize;
            let end = offsets[row_idx + 1] as usize;
            if start > end || end > data.len() {
                continue;
            }

            if let Ok(value) = std::str::from_utf8(&data[start..end]) {
                cache.entry(Box::<str>::from(value)).or_insert(row_ids[row_idx]);
            }
        }

        Ok(Some(cache))
    }

    /// Read columns with numeric range filter and early termination for LIMIT
    /// Optimized for SELECT * WHERE col BETWEEN low AND high LIMIT n
    pub fn read_columns_filtered_range_with_limit(
        &self,
        column_names: Option<&[&str]>,
        filter_column: &str,
        low: f64,
        high: f64,
        limit: usize,
        offset: usize,
    ) -> io::Result<(HashMap<String, ColumnData>, Vec<usize>)> {
        // V4: scan in-memory columns directly with early termination
        // LOCK-FREE: use cached_footer_offset for V4 detection
        {
            if self.cached_footer_offset.load(Ordering::Relaxed) > 0 {
                if !self.has_v4_in_memory_data() {
                    // MMAP PATH: use BETWEEN as >= low AND <= high
                    let (mut result, mut indices) = self.read_columns_filtered_mmap(
                        column_names, filter_column, ">=", low,
                    )?;
                    // Apply <= high filter
                    indices.retain(|&i| {
                        for col in result.values() {
                            match col {
                                ColumnData::Int64(v) => { if i < v.len() && v[i] > high as i64 { return false; } }
                                ColumnData::Float64(v) => { if i < v.len() && v[i] > high { return false; } }
                                _ => {}
                            }
                        }
                        true
                    });
                    // Apply offset+limit
                    if indices.len() > offset {
                        let end = (offset + limit).min(indices.len());
                        let keep: Vec<usize> = (offset..end).collect();
                        for (_name, col) in result.iter_mut() {
                            *col = col.filter_by_indices(&keep);
                        }
                        indices = indices[offset..end].to_vec();
                    } else {
                        return Ok((HashMap::new(), Vec::new()));
                    }
                    return Ok((result, indices));
                }
                
                let schema = self.schema.read();
                let columns = self.columns.read();
                let deleted = self.deleted.read();
                let total_rows = self.ids.read().len();
                let needed = offset + limit;
                
                let filter_col_idx = match schema.get_index(filter_column) {
                    Some(idx) => idx,
                    None => return Ok((HashMap::new(), Vec::new())),
                };
                
                let has_deleted = deleted.iter().any(|&b| b != 0);
                let mut matching_indices: Vec<usize> = Vec::with_capacity(needed.min(1024));
                
                if filter_col_idx < columns.len() {
                    match &columns[filter_col_idx] {
                        ColumnData::Int64(values) => {
                            let low_i = low as i64;
                            let high_i = high as i64;
                            let count = values.len().min(total_rows);
                            for i in 0..count {
                                if matching_indices.len() >= needed { break; }
                                if has_deleted {
                                    let byte_idx = i / 8;
                                    let bit_idx = i % 8;
                                    if byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 != 0 { continue; }
                                }
                                let v = unsafe { *values.get_unchecked(i) };
                                if v >= low_i && v <= high_i {
                                    matching_indices.push(i);
                                }
                            }
                        }
                        ColumnData::Float64(values) => {
                            let count = values.len().min(total_rows);
                            for i in 0..count {
                                if matching_indices.len() >= needed { break; }
                                if has_deleted {
                                    let byte_idx = i / 8;
                                    let bit_idx = i % 8;
                                    if byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 != 0 { continue; }
                                }
                                let v = unsafe { *values.get_unchecked(i) };
                                if v >= low && v <= high {
                                    matching_indices.push(i);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                
                // Apply offset + limit
                if offset > 0 && offset < matching_indices.len() {
                    matching_indices = matching_indices[offset..].to_vec();
                } else if offset >= matching_indices.len() {
                    matching_indices.clear();
                }
                if matching_indices.len() > limit {
                    matching_indices.truncate(limit);
                }
                
                if matching_indices.is_empty() {
                    return Ok((HashMap::new(), Vec::new()));
                }
                
                let col_indices: Vec<usize> = match column_names {
                    Some(names) => names.iter()
                        .filter(|&&n| n != "_id")
                        .filter_map(|&name| schema.get_index(name))
                        .collect(),
                    None => (0..schema.column_count()).collect(),
                };
                
                let mut result = HashMap::new();
                for &col_idx in &col_indices {
                    let (col_name, _) = &schema.columns[col_idx];
                    if col_idx < columns.len() {
                        result.insert(col_name.clone(), columns[col_idx].filter_by_indices(&matching_indices));
                    }
                }
                
                return Ok((result, matching_indices));
            }
        }
        
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        let header = self.header.read();
        let deleted = self.deleted.read();
        
        let filter_col_idx = schema.get_index(filter_column).ok_or_else(|| {
            err_not_found(format!("Column: {}", filter_column))
        })?;
        
        let (_, filter_col_type) = &schema.columns[filter_col_idx];
        let filter_index = &column_index[filter_col_idx];
        let total_rows = header.row_count as usize;
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            err_not_conn("File not open")
        })?;
        
        let mut mmap_cache = self.mmap_cache.write();
        let needed = offset + limit;
        
        // Only works for numeric columns
        if !matches!(filter_col_type, ColumnType::Int64 | ColumnType::Float64 | 
                     ColumnType::Int32 | ColumnType::Int16 | ColumnType::Int8 |
                     ColumnType::UInt64 | ColumnType::UInt32 | ColumnType::UInt16 | ColumnType::UInt8 |
                     ColumnType::Float32) {
            return Err(err_input("Range filter needs numeric columns"));
        }
        
        // Stream through the filter column in chunks with early termination
        const CHUNK_SIZE: usize = 8192;
        let mut matching_indices = Vec::with_capacity(needed);
        let mut row_start = 0;
        
        while row_start < total_rows && matching_indices.len() < needed {
            let chunk_rows = CHUNK_SIZE.min(total_rows - row_start);
            
            // Read chunk of filter column
            let chunk_data = self.read_column_range_mmap(
                &mut mmap_cache, file, filter_index, *filter_col_type, 
                row_start, chunk_rows, total_rows
            )?;
            
            // Evaluate range predicate on chunk
            match &chunk_data {
                ColumnData::Int64(values) => {
                    let low_i = low as i64;
                    let high_i = high as i64;
                    for (i, &v) in values.iter().enumerate() {
                        let row_idx = row_start + i;
                        // Check deleted bitmap
                        let byte_idx = row_idx / 8;
                        let bit_idx = row_idx % 8;
                        let is_deleted = byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 == 1;
                        
                        if !is_deleted && v >= low_i && v <= high_i {
                            matching_indices.push(row_idx);
                            if matching_indices.len() >= needed {
                                break;
                            }
                        }
                    }
                }
                ColumnData::Float64(values) => {
                    for (i, &v) in values.iter().enumerate() {
                        let row_idx = row_start + i;
                        let byte_idx = row_idx / 8;
                        let bit_idx = row_idx % 8;
                        let is_deleted = byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 == 1;
                        
                        if !is_deleted && v >= low && v <= high {
                            matching_indices.push(row_idx);
                            if matching_indices.len() >= needed {
                                break;
                            }
                        }
                    }
                }
                _ => {}
            }
            
            row_start += chunk_rows;
        }
        
        // Apply offset
        let final_indices: Vec<usize> = matching_indices.into_iter().skip(offset).take(limit).collect();
        
        if final_indices.is_empty() {
            return Ok((HashMap::new(), Vec::new()));
        }
        
        // Read columns for matching rows
        let col_indices: Vec<usize> = match column_names {
            Some(names) => names.iter().filter_map(|name| schema.get_index(name)).collect(),
            None => (0..schema.column_count()).collect(),
        };
        
        let mut result = HashMap::new();
        for &col_idx in &col_indices {
            let (col_name, col_type) = &schema.columns[col_idx];
            let index_entry = &column_index[col_idx];
            let col_data = self.read_column_scattered_mmap(&mut mmap_cache, file, index_entry, *col_type, &final_indices, total_rows)?;
            result.insert(col_name.clone(), col_data);
        }
        
        Ok((result, final_indices))
    }

    /// Read columns with combined STRING + NUMERIC filter and early termination
    /// Optimized for SELECT * WHERE string_col = 'value' AND numeric_col > N LIMIT n
    /// Two-stage filter: first string equality (fast dict scan), then numeric comparison
    pub fn read_columns_filtered_string_numeric_with_limit(
        &self,
        column_names: Option<&[&str]>,
        string_column: &str,
        string_value: &str,
        numeric_column: &str,
        numeric_op: &str,  // ">" | ">=" | "<" | "<=" | "="
        numeric_value: f64,
        limit: usize,
        offset: usize,
    ) -> io::Result<(HashMap<String, ColumnData>, Vec<usize>)> {
        // V4: scan in-memory columns directly with early termination
        // LOCK-FREE: use cached_footer_offset for V4 detection
        {
            if self.cached_footer_offset.load(Ordering::Relaxed) > 0 {
                if !self.has_v4_in_memory_data() {
                    // MMAP PATH: scan string + numeric filter columns, then apply limit
                    let (result, indices) = self.read_columns_filtered_string_mmap(
                        column_names, string_column, string_value, true,
                    )?;
                    // TODO: apply numeric filter + offset/limit on top
                    // For now return the string-filtered result (numeric filter applied by caller)
                    return Ok((result, indices));
                }
                
                let schema = self.schema.read();
                let columns = self.columns.read();
                let deleted = self.deleted.read();
                let total_rows = self.ids.read().len();
                let needed = offset + limit;
                
                let str_col_idx = match schema.get_index(string_column) {
                    Some(idx) => idx,
                    None => return Ok((HashMap::new(), Vec::new())),
                };
                let num_col_idx = match schema.get_index(numeric_column) {
                    Some(idx) => idx,
                    None => return Ok((HashMap::new(), Vec::new())),
                };
                
                let filter_bytes = string_value.as_bytes();
                let filter_len = filter_bytes.len();
                let has_deleted = deleted.iter().any(|&b| b != 0);
                let mut matching_indices: Vec<usize> = Vec::with_capacity(needed.min(1024));
                
                // Get string and numeric column references
                if str_col_idx < columns.len() && num_col_idx < columns.len() {
                    if let ColumnData::String { offsets: str_offsets, data: str_data } = &columns[str_col_idx] {
                        let count = str_offsets.len().saturating_sub(1).min(total_rows);
                        
                        let num_compare = |idx: usize| -> bool {
                            match &columns[num_col_idx] {
                                ColumnData::Int64(vals) if idx < vals.len() => {
                                    let v = vals[idx] as f64;
                                    match numeric_op {
                                        ">" => v > numeric_value,
                                        ">=" => v >= numeric_value,
                                        "<" => v < numeric_value,
                                        "<=" => v <= numeric_value,
                                        "=" => (v - numeric_value).abs() < f64::EPSILON,
                                        _ => false,
                                    }
                                }
                                ColumnData::Float64(vals) if idx < vals.len() => {
                                    let v = vals[idx];
                                    match numeric_op {
                                        ">" => v > numeric_value,
                                        ">=" => v >= numeric_value,
                                        "<" => v < numeric_value,
                                        "<=" => v <= numeric_value,
                                        "=" => (v - numeric_value).abs() < f64::EPSILON,
                                        _ => false,
                                    }
                                }
                                _ => false,
                            }
                        };
                        
                        for i in 0..count {
                            if matching_indices.len() >= needed { break; }
                            
                            if has_deleted {
                                let byte_idx = i / 8;
                                let bit_idx = i % 8;
                                if byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 != 0 { continue; }
                            }
                            
                            // String equality check first (usually more selective)
                            let start = str_offsets[i] as usize;
                            let end = str_offsets[i + 1] as usize;
                            if end - start != filter_len { continue; }
                            if end > str_data.len() || &str_data[start..end] != filter_bytes { continue; }
                            
                            // Then numeric check
                            if num_compare(i) {
                                matching_indices.push(i);
                            }
                        }
                    }
                }
                
                // Apply offset + limit
                if offset > 0 && offset < matching_indices.len() {
                    matching_indices = matching_indices[offset..].to_vec();
                } else if offset >= matching_indices.len() {
                    matching_indices.clear();
                }
                if matching_indices.len() > limit {
                    matching_indices.truncate(limit);
                }
                
                if matching_indices.is_empty() {
                    return Ok((HashMap::new(), Vec::new()));
                }
                
                let col_indices: Vec<usize> = match column_names {
                    Some(names) => names.iter()
                        .filter(|&&n| n != "_id")
                        .filter_map(|&name| schema.get_index(name))
                        .collect(),
                    None => (0..schema.column_count()).collect(),
                };
                
                let mut result = HashMap::new();
                for &col_idx in &col_indices {
                    let (col_name, _) = &schema.columns[col_idx];
                    if col_idx < columns.len() {
                        result.insert(col_name.clone(), columns[col_idx].filter_by_indices(&matching_indices));
                    }
                }
                
                return Ok((result, matching_indices));
            }
        }
        
        let header = self.header.read();
        let schema = self.schema.read();
        let column_index = self.column_index.read();
        let deleted = self.deleted.read();
        
        let total_rows = header.row_count as usize;
        if total_rows == 0 {
            return Ok((HashMap::new(), Vec::new()));
        }

        // Get string column info
        let str_col_idx = schema.get_index(string_column).ok_or_else(|| {
            err_not_found(format!("String column: {}", string_column))
        })?;
        let (_, str_col_type) = &schema.columns[str_col_idx];
        let str_index = &column_index[str_col_idx];
        
        // Get numeric column info
        let num_col_idx = schema.get_index(numeric_column).ok_or_else(|| {
            err_not_found(format!("Numeric column: {}", numeric_column))
        })?;
        let (_, num_col_type) = &schema.columns[num_col_idx];
        let num_index = &column_index[num_col_idx];
        
        // Validate column types
        if !matches!(str_col_type, ColumnType::String | ColumnType::StringDict) {
            return Err(err_input("String filter requires string column"));
        }
        if !matches!(num_col_type, ColumnType::Int64 | ColumnType::Float64 | 
                     ColumnType::Int32 | ColumnType::Int16 | ColumnType::Int8 |
                     ColumnType::UInt64 | ColumnType::UInt32 | ColumnType::UInt16 | ColumnType::UInt8 |
                     ColumnType::Float32) {
            return Err(err_input("Numeric filter needs numeric column"));
        }
        
        let file_guard = self.file.read();
        let file = file_guard.as_ref().ok_or_else(|| {
            err_not_conn("File not open")
        })?;
        
        let mut mmap_cache = self.mmap_cache.write();
        let needed = offset + limit;

        // For StringDict, use fast dictionary-based filter
        if *str_col_type == ColumnType::StringDict {
            let base_offset = str_index.data_offset;
            
            // Read dictionary header and find target key
            let mut str_header = [0u8; 16];
            mmap_cache.read_at(file, &mut str_header, base_offset)?;
            let stored_rows = u64::from_le_bytes(str_header[0..8].try_into().unwrap()) as usize;
            let dict_size = u64::from_le_bytes(str_header[8..16].try_into().unwrap()) as usize;
            
            if stored_rows == 0 || dict_size == 0 {
                return Ok((HashMap::new(), Vec::new()));
            }
            
            // Read dictionary
            let dict_offsets_offset = base_offset + 16 + (stored_rows * 4) as u64;
            let mut dict_offsets_buf = vec![0u8; dict_size * 4];
            mmap_cache.read_at(file, &mut dict_offsets_buf, dict_offsets_offset)?;
            
            let mut dict_offsets = Vec::with_capacity(dict_size);
            for i in 0..dict_size {
                dict_offsets.push(u32::from_le_bytes(dict_offsets_buf[i * 4..(i + 1) * 4].try_into().unwrap()));
            }
            
            let dict_data_len_offset = dict_offsets_offset + (dict_size * 4) as u64;
            let mut data_len_buf = [0u8; 8];
            mmap_cache.read_at(file, &mut data_len_buf, dict_data_len_offset)?;
            let dict_data_len = u64::from_le_bytes(data_len_buf) as usize;
            
            let dict_data_offset = dict_data_len_offset + 8;
            let mut dict_data = vec![0u8; dict_data_len];
            if dict_data_len > 0 {
                mmap_cache.read_at(file, &mut dict_data, dict_data_offset)?;
            }
            
            // Find target key
            let filter_bytes = string_value.as_bytes();
            let mut target_key: Option<u32> = None;
            let dict_count = dict_size.saturating_sub(1);
            
            for i in 0..dict_count {
                let start = dict_offsets[i] as usize;
                let end = if i + 1 < dict_size { dict_offsets[i + 1] as usize } else { dict_data_len };
                if end <= dict_data.len() && start <= end && &dict_data[start..end] == filter_bytes {
                    target_key = Some((i + 1) as u32);
                    break;
                }
            }
            
            let target_key = match target_key {
                Some(k) => k,
                None => return Ok((HashMap::new(), Vec::new())),
            };
            
            // Two-stage streaming filter with early termination
            let str_indices_offset = base_offset + 16;
            let mut matching_indices = Vec::with_capacity(needed.min(1000));
            
            const CHUNK_SIZE: usize = 8192;
            let mut row = 0usize;
            
            while row < stored_rows && matching_indices.len() < needed {
                let chunk_rows = CHUNK_SIZE.min(stored_rows - row);
                
                // Read string indices chunk
                let mut str_chunk = vec![0u32; chunk_rows];
                let chunk_bytes = unsafe {
                    std::slice::from_raw_parts_mut(str_chunk.as_mut_ptr() as *mut u8, chunk_rows * 4)
                };
                mmap_cache.read_at(file, chunk_bytes, str_indices_offset + (row * 4) as u64)?;
                
                // Read numeric column chunk
                let num_chunk = self.read_column_range_mmap(
                    &mut mmap_cache, file, num_index, *num_col_type, row, chunk_rows, total_rows
                )?;
                
                // Combined filter
                for i in 0..chunk_rows {
                    let row_idx = row + i;
                    
                    // Check deleted
                    let byte_idx = row_idx / 8;
                    let bit_idx = row_idx % 8;
                    let is_deleted = byte_idx < deleted.len() && (deleted[byte_idx] >> bit_idx) & 1 == 1;
                    if is_deleted {
                        continue;
                    }
                    
                    // Check string match
                    if str_chunk[i] != target_key {
                        continue;
                    }
                    
                    // Check numeric condition
                    let num_match = match &num_chunk {
                        ColumnData::Int64(values) => {
                            let v = values[i] as f64;
                            match numeric_op {
                                ">" => v > numeric_value,
                                ">=" => v >= numeric_value,
                                "<" => v < numeric_value,
                                "<=" => v <= numeric_value,
                                "=" => (v - numeric_value).abs() < f64::EPSILON,
                                _ => false,
                            }
                        }
                        ColumnData::Float64(values) => {
                            let v = values[i];
                            match numeric_op {
                                ">" => v > numeric_value,
                                ">=" => v >= numeric_value,
                                "<" => v < numeric_value,
                                "<=" => v <= numeric_value,
                                "=" => (v - numeric_value).abs() < f64::EPSILON,
                                _ => false,
                            }
                        }
                        _ => false,
                    };
                    
                    if num_match {
                        matching_indices.push(row_idx);
                        if matching_indices.len() >= needed {
                            break;
                        }
                    }
                }
                
                row += chunk_rows;
            }
            
            // Apply offset
            let final_indices: Vec<usize> = matching_indices.into_iter().skip(offset).take(limit).collect();
            
            if final_indices.is_empty() {
                return Ok((HashMap::new(), Vec::new()));
            }
            
            // Read columns for matching rows
            let col_indices: Vec<usize> = match column_names {
                Some(names) => names.iter().filter_map(|name| schema.get_index(name)).collect(),
                None => (0..schema.column_count()).collect(),
            };
            
            let mut result = HashMap::with_capacity(col_indices.len());
            for &col_idx in &col_indices {
                let (col_name, col_type) = &schema.columns[col_idx];
                let index_entry = &column_index[col_idx];
                let col_data = self.read_column_scattered_mmap(&mut mmap_cache, file, index_entry, *col_type, &final_indices, total_rows)?;
                result.insert(col_name.clone(), col_data);
            }
            
            return Ok((result, final_indices));
        }
        
        // Fallback for non-dictionary strings
        Err(err_input("Needs dictionary-encoded string"))
    }

    /// Read a single column for specific row indices
    pub fn read_column_by_indices(
        &self,
        column_name: &str,
        row_indices: &[usize],
    ) -> io::Result<ColumnData> {
        let is_v4 = self.is_v4_format();
        
        let schema = self.schema.read();
        let header = self.header.read();
        
        let col_idx = schema.get_index(column_name).ok_or_else(|| {
            err_not_found(format!("Column: {}", column_name))
        })?;
        
        let (_, col_type) = &schema.columns[col_idx];
        let col_type = *col_type;
        let total_rows = header.row_count as usize;
        drop(schema);
        drop(header);

        self.read_column_scattered_auto(col_idx, col_type, row_indices, total_rows, is_v4)
    }

    /// Check if this is a V4 format file (has footer).
    /// LOCK-FREE: uses cached_footer_offset atomic instead of header RwLock.
    #[inline]
    pub fn is_v4_format(&self) -> bool {
        self.cached_footer_offset.load(Ordering::Relaxed) > 0
    }

    /// Check if V4 column data is currently loaded in memory.
    /// Does NOT trigger any loading — purely a state check.
    #[inline]
    pub fn has_v4_in_memory_data(&self) -> bool {
        if !self.is_v4_format() { return false; }
        let cols = self.columns.read();
        !cols.is_empty() && cols.iter().any(|c| c.len() > 0)
    }

    /// Number of rows currently buffered in memory for a V4 table.
    ///
    /// Insert backends for mmap-only V4 files keep newly appended rows in
    /// `ids/columns` until save()/flush() persists them as a row group. If the
    /// full base table has been loaded into memory, only rows beyond the on-disk
    /// footer count are considered pending.
    #[inline]
    pub fn pending_v4_in_memory_rows(&self) -> usize {
        if !self.is_v4_format() {
            return 0;
        }
        let ids = self.ids.read();
        let ids_len = ids.len();
        if ids_len == 0 || !self.has_v4_in_memory_data() {
            return 0;
        }
        let on_disk_rows = self
            .v4_footer
            .read()
            .as_ref()
            .map(|footer| footer.row_groups.iter().map(|rg| rg.row_count as usize).sum())
            .unwrap_or(0);
        if on_disk_rows == 0 {
            ids_len
        } else if ids.first().copied().unwrap_or(0) != 1 {
            // Insert backends for mmap-only V4 files hold only newly appended
            // IDs, e.g. base has rows 1..N while memory starts at N+1.
            ids_len
        } else if ids_len < on_disk_rows {
            ids_len
        } else {
            ids_len.saturating_sub(on_disk_rows)
        }
    }

    /// Check if in-memory columns contain the FULL base dataset (not just write buffer).
    /// Used by save() to decide between append vs full rewrite.
    #[inline]
    fn has_v4_in_memory_data_with_base(&self, on_disk_rows: usize) -> bool {
        let cols = self.columns.read();
        if cols.is_empty() { return false; }
        // If any column has >= on_disk_rows elements, base data is loaded
        cols.iter().any(|c| c.len() >= on_disk_rows)
    }

}
