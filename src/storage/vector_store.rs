//! Segmented mmap vector storage.
//!
//! Each segment contains only contiguous encoded vector rows. The manifest is
//! changed only when the segment set changes, so normal ingestion remains a
//! sequential append while compaction can replace selected segments.

use crate::distance::DistanceMetric;
use crate::error::{LynseError, Result};
use crate::storage::approx_search::ApproxSearchConfig;
use crate::storage::dtype::{encode_f32_slice_as_le_bytes, VectorDtype};
use crate::storage::flat_mmap::FlatMmap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering as CmpOrdering;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Write};
#[cfg(not(any(unix, windows)))]
use std::io::{Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const VECTOR_MANIFEST_FILE: &str = "vector_manifest.json";
const VECTOR_MANIFEST_VERSION: u32 = 1;
const SEGMENT_DIR: &str = "vector_segments";
const DEFAULT_ID_MAP_FILE: &str = "id_map.bin";
const UPDATE_JOURNAL_FILE: &str = "vector_updates.wal";
const UPDATE_JOURNAL_MAGIC: &[u8; 8] = b"LYNVUPD1";
const UPDATE_JOURNAL_VERSION: u32 = 1;
#[cfg(not(test))]
const DEFAULT_SEGMENT_TARGET_BYTES: u64 = 256 * 1024 * 1024;
#[cfg(test)]
const DEFAULT_SEGMENT_TARGET_BYTES: u64 = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SegmentEntry {
    file: String,
    rows: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorManifest {
    version: u32,
    generation: u64,
    id_map_file: String,
    segments: Vec<SegmentEntry>,
}

impl VectorManifest {
    fn legacy(rows: u64) -> Self {
        Self {
            version: VECTOR_MANIFEST_VERSION,
            generation: 0,
            id_map_file: DEFAULT_ID_MAP_FILE.to_string(),
            segments: if rows == 0 {
                Vec::new()
            } else {
                vec![SegmentEntry {
                    file: "vectors.bin".to_string(),
                    rows,
                }]
            },
        }
    }
}

fn write_atomic_durable(path: &Path, data: &[u8]) -> Result<()> {
    let tmp_path = path.with_extension("tmp");
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&tmp_path)?;
    file.write_all(data)?;
    file.sync_all()?;
    std::fs::rename(&tmp_path, path)?;
    if let Some(parent) = path.parent() {
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    let tmp_path = path.with_extension("tmp");
    std::fs::write(&tmp_path, data)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

fn write_all_at(file: &File, mut data: &[u8], mut offset: u64) -> std::io::Result<()> {
    while !data.is_empty() {
        #[cfg(unix)]
        let written = {
            use std::os::unix::fs::FileExt;
            file.write_at(data, offset)?
        };
        #[cfg(windows)]
        let written = {
            use std::os::windows::fs::FileExt;
            file.seek_write(data, offset)?
        };
        #[cfg(not(any(unix, windows)))]
        let written = {
            let mut cloned = file.try_clone()?;
            cloned.seek(SeekFrom::Start(offset))?;
            cloned.write(data)?
        };

        if written == 0 {
            return Err(std::io::Error::new(
                ErrorKind::WriteZero,
                "failed to write vector row at its file offset",
            ));
        }
        data = &data[written..];
        offset += written as u64;
    }
    Ok(())
}

struct PendingVectorUpdates {
    rows: Vec<u64>,
    encoded: Vec<u8>,
}

pub struct VectorStore {
    collection_path: PathBuf,
    dimension: usize,
    dtype: VectorDtype,
    manifest_path: PathBuf,
    manifest: RwLock<VectorManifest>,
    mmap_cache: RwLock<Vec<Option<FlatMmap>>>,
    compatibility_mmap: RwLock<Option<FlatMmap>>,
    total_vectors: AtomicU64,
    fingerprint: RwLock<Option<String>>,
    segment_target_bytes: u64,
}

impl VectorStore {
    pub fn new(
        collection_path: &Path,
        dimension: usize,
        _chunk_size: usize,
        dtype: VectorDtype,
    ) -> Result<Self> {
        std::fs::create_dir_all(collection_path)?;
        let manifest_path = collection_path.join(VECTOR_MANIFEST_FILE);
        let row_width = dimension as u64 * dtype.byte_width() as u64;

        let mut manifest = if manifest_path.exists() {
            let bytes = std::fs::read(&manifest_path)?;
            let parsed: VectorManifest = serde_json::from_slice(&bytes)
                .map_err(|e| LynseError::Serialization(e.to_string()))?;
            if parsed.version > VECTOR_MANIFEST_VERSION {
                return Err(LynseError::Storage(format!(
                    "vector manifest version {} is newer than supported version {}",
                    parsed.version, VECTOR_MANIFEST_VERSION
                )));
            }
            parsed
        } else {
            let legacy_path = collection_path.join("vectors.bin");
            let bytes = std::fs::metadata(&legacy_path).map(|m| m.len()).unwrap_or(0);
            VectorManifest::legacy(if row_width == 0 { 0 } else { bytes / row_width })
        };

        let mut total_vectors = 0u64;
        for segment in &mut manifest.segments {
            let len = std::fs::metadata(collection_path.join(&segment.file))
                .map(|m| m.len())
                .unwrap_or(0);
            segment.rows = if row_width == 0 { 0 } else { len / row_width };
            total_vectors = total_vectors.saturating_add(segment.rows);
        }

        let fp_path = collection_path.join("fingerprint");
        let fingerprint = if fp_path.exists() {
            std::fs::read_to_string(&fp_path)?
                .lines()
                .last()
                .map(|line| line.trim().to_string())
        } else {
            None
        };
        let cache_len = manifest.segments.len();
        let segment_target_bytes = std::env::var("LYNSE_SEGMENT_TARGET_BYTES")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_SEGMENT_TARGET_BYTES)
            .max(row_width);

        Ok(Self {
            collection_path: collection_path.to_path_buf(),
            dimension,
            dtype,
            manifest_path,
            manifest: RwLock::new(manifest),
            mmap_cache: RwLock::new((0..cache_len).map(|_| None).collect()),
            compatibility_mmap: RwLock::new(None),
            total_vectors: AtomicU64::new(total_vectors),
            fingerprint: RwLock::new(fingerprint),
            segment_target_bytes,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn dtype(&self) -> VectorDtype {
        self.dtype
    }

    pub fn collection_path(&self) -> &Path {
        &self.collection_path
    }

    pub fn id_map_path(&self) -> PathBuf {
        self.collection_path.join(&self.manifest.read().id_map_file)
    }

    pub fn segment_paths(&self) -> Vec<PathBuf> {
        self.manifest
            .read()
            .segments
            .iter()
            .map(|segment| self.collection_path.join(&segment.file))
            .collect()
    }

    pub fn vectors_path(&self) -> PathBuf {
        self.segment_paths()
            .into_iter()
            .next()
            .unwrap_or_else(|| self.collection_path.join("vectors.bin"))
    }

    pub fn fingerprint(&self) -> Option<String> {
        self.fingerprint.read().clone()
    }

    fn row_width(&self) -> u64 {
        self.dimension as u64 * self.dtype.byte_width() as u64
    }

    fn mark_changed(&self, total_vectors: u64) {
        *self.fingerprint.write() = Some(format!(
            "{}:{}:{}",
            total_vectors,
            self.dimension,
            self.dtype.storage_name()
        ));
    }

    fn persist_manifest(&self, manifest: &VectorManifest) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(manifest)
            .map_err(|e| LynseError::Serialization(e.to_string()))?;
        write_atomic_durable(&self.manifest_path, &bytes)
    }

    pub fn persist_metadata(&self) -> Result<()> {
        self.persist_metadata_with(write_atomic_durable)
    }

    /// Persist compatibility metadata without forcing it to stable storage.
    ///
    /// This is used by the lightweight commit path; explicit checkpoints still
    /// use `persist_metadata()` and provide the fsync durability barrier.
    pub fn persist_metadata_fast(&self) -> Result<()> {
        self.persist_metadata_with(write_atomic)
    }

    fn persist_metadata_with(&self, writer: fn(&Path, &[u8]) -> Result<()>) -> Result<()> {
        let total = self.total_vectors.load(Ordering::Relaxed);
        let info = serde_json::json!({ "total_shape": [total, self.dimension] });
        writer(
            &self.collection_path.join("info.json"),
            serde_json::to_string(&info)
                .map_err(|e| LynseError::Serialization(e.to_string()))?
                .as_bytes(),
        )?;
        if let Some(fp) = self.fingerprint() {
            writer(&self.collection_path.join("fingerprint"), fp.as_bytes())?;
        }
        Ok(())
    }

    pub fn write(&self, data: &[f32]) -> Result<()> {
        if data.len() % self.dimension != 0 {
            return Err(LynseError::DimensionMismatch {
                expected: self.dimension,
                got: data.len() % self.dimension,
            });
        }
        let rows = data.len() / self.dimension;
        if rows == 0 {
            return Ok(());
        }
        self.append_encoded_bytes(&encode_f32_slice_as_le_bytes(data, self.dtype), rows)
    }

    pub fn write_encoded_le_bytes(
        &self,
        data: &[u8],
        n_vectors: usize,
        dtype: VectorDtype,
    ) -> Result<()> {
        if dtype != self.dtype {
            return Err(LynseError::InvalidArgument(format!(
                "encoded vector dtype {} does not match store dtype {}",
                dtype.storage_name(),
                self.dtype.storage_name()
            )));
        }
        let expected = n_vectors
            .checked_mul(self.dimension)
            .and_then(|values| values.checked_mul(dtype.byte_width()))
            .ok_or_else(|| LynseError::InvalidArgument("encoded byte size overflows".into()))?;
        if data.len() != expected {
            return Err(LynseError::DimensionMismatch {
                expected,
                got: data.len(),
            });
        }
        if n_vectors == 0 {
            return Ok(());
        }
        self.append_encoded_bytes(data, n_vectors)
    }

    fn next_segment_file(manifest: &VectorManifest) -> String {
        format!(
            "{}/seg-{:020}-{:06}.bin",
            SEGMENT_DIR,
            manifest.generation + 1,
            manifest.segments.len()
        )
    }

    fn append_encoded_bytes(&self, bytes: &[u8], n_vectors: usize) -> Result<()> {
        let mut manifest = self.manifest.write();
        let row_width = self.row_width();
        let append_to_current = manifest.segments.last().is_some_and(|segment| {
            segment.rows.saturating_mul(row_width) + bytes.len() as u64
                <= self.segment_target_bytes
        });

        if manifest.segments.is_empty() || !append_to_current {
            let previous_manifest = manifest.clone();
            let file_name = if manifest.segments.is_empty() && !self.manifest_path.exists() {
                "vectors.bin".to_string()
            } else {
                Self::next_segment_file(&manifest)
            };
            let path = self.collection_path.join(&file_name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&path)?;
            if let Err(error) = file.write_all(bytes) {
                let _ = std::fs::remove_file(&path);
                return Err(error.into());
            }
            if let Err(error) = file.flush() {
                let _ = std::fs::remove_file(&path);
                return Err(error.into());
            }
            manifest.generation += 1;
            manifest.segments.push(SegmentEntry {
                file: file_name,
                rows: n_vectors as u64,
            });
            if manifest.segments.len() > 1 || self.manifest_path.exists() {
                if let Err(error) = self.persist_manifest(&manifest) {
                    *manifest = previous_manifest;
                    let _ = std::fs::remove_file(&path);
                    return Err(error);
                }
            }
            self.mmap_cache.write().push(None);
        } else {
            let segment = manifest.segments.last_mut().expect("segment exists");
            let path = self.collection_path.join(&segment.file);
            let old_len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
            if let Err(error) = file.write_all(bytes).and_then(|_| file.flush()) {
                let _ = file.set_len(old_len);
                return Err(error.into());
            }
            segment.rows += n_vectors as u64;
            if let Some(last) = self.mmap_cache.write().last_mut() {
                *last = None;
            }
        }
        *self.compatibility_mmap.write() = None;

        let total = self
            .total_vectors
            .fetch_add(n_vectors as u64, Ordering::Relaxed)
            + n_vectors as u64;
        self.mark_changed(total);
        Ok(())
    }

    fn ensure_mmaps(&self) -> Result<()> {
        let manifest = self.manifest.read();
        let mut cache = self.mmap_cache.write();
        if cache.len() != manifest.segments.len() {
            cache.resize_with(manifest.segments.len(), || None);
        }
        for (slot, segment) in cache.iter_mut().zip(manifest.segments.iter()) {
            if slot.is_none() && segment.rows > 0 {
                *slot = Some(
                    FlatMmap::open(
                        &self.collection_path.join(&segment.file),
                        self.dimension,
                        self.dtype,
                    )
                    .map_err(|e| LynseError::Storage(format!("FlatMmap open error: {e}")))?,
                );
            }
        }
        Ok(())
    }

    fn ensure_compatibility_mmap(&self) -> Result<()> {
        if self.compatibility_mmap.read().is_some() {
            return Ok(());
        }
        let manifest = self.manifest.read().clone();
        if manifest.segments.is_empty() {
            return Ok(());
        }
        let path = if manifest.segments.len() == 1 {
            self.collection_path.join(&manifest.segments[0].file)
        } else {
            let path = self.collection_path.join("vectors.compat.bin");
            let mut bytes = Vec::with_capacity(
                self.total_vectors.load(Ordering::Relaxed) as usize * self.row_width() as usize,
            );
            for segment in &manifest.segments {
                bytes.extend_from_slice(&std::fs::read(self.collection_path.join(&segment.file))?);
            }
            write_atomic_durable(&path, &bytes)?;
            path
        };
        let mmap = FlatMmap::open(&path, self.dimension, self.dtype)
            .map_err(|e| LynseError::Storage(format!("FlatMmap open error: {e}")))?;
        *self.compatibility_mmap.write() = Some(mmap);
        Ok(())
    }

    pub fn read_mmap(&self) -> Result<parking_lot::RwLockReadGuard<'_, Option<FlatMmap>>> {
        self.ensure_compatibility_mmap()?;
        Ok(self.compatibility_mmap.read())
    }

    pub fn contiguous_path(&self) -> Result<PathBuf> {
        self.ensure_compatibility_mmap()?;
        let manifest = self.manifest.read();
        Ok(if manifest.segments.len() <= 1 {
            manifest
                .segments
                .first()
                .map(|segment| self.collection_path.join(&segment.file))
                .unwrap_or_else(|| self.collection_path.join("vectors.bin"))
        } else {
            self.collection_path.join("vectors.compat.bin")
        })
    }

    pub fn read_all_f32(&self) -> Result<Vec<f32>> {
        self.ensure_mmaps()?;
        let cache = self.mmap_cache.read();
        let mut data = Vec::with_capacity(
            self.total_vectors.load(Ordering::Relaxed) as usize * self.dimension,
        );
        for mmap in cache.iter().flatten() {
            data.extend_from_slice(&mmap.as_f32_cow());
        }
        Ok(data)
    }

    pub fn read_rows(&self, rows: &[u64]) -> Result<Vec<f32>> {
        self.ensure_mmaps()?;
        let manifest = self.manifest.read();
        let cache = self.mmap_cache.read();
        let mut result = vec![0.0f32; rows.len() * self.dimension];
        let mut base = 0u64;
        for (segment, mmap) in manifest.segments.iter().zip(cache.iter()) {
            let end = base + segment.rows;
            if let Some(mmap) = mmap {
                let data = mmap.as_f32_cow();
                for (out, &row) in rows.iter().enumerate() {
                    if row >= base && row < end {
                        let local = (row - base) as usize;
                        result[out * self.dimension..(out + 1) * self.dimension]
                            .copy_from_slice(
                                &data[local * self.dimension..(local + 1) * self.dimension],
                            );
                    }
                }
            }
            base = end;
        }
        Ok(result)
    }

    fn update_journal_path(&self) -> PathBuf {
        self.collection_path.join(UPDATE_JOURNAL_FILE)
    }

    pub fn has_pending_updates(&self) -> bool {
        self.update_journal_path().exists()
    }

    fn grouped_updates(
        &self,
        manifest: &VectorManifest,
        rows: &[u64],
    ) -> Result<Vec<Vec<(u64, usize)>>> {
        let mut groups = vec![Vec::new(); manifest.segments.len()];
        for (input_index, &row) in rows.iter().enumerate() {
            let mut base = 0u64;
            let mut found = false;
            for (segment_index, segment) in manifest.segments.iter().enumerate() {
                let end = base.saturating_add(segment.rows);
                if row >= base && row < end {
                    groups[segment_index].push((row - base, input_index));
                    found = true;
                    break;
                }
                base = end;
            }
            if !found {
                return Err(LynseError::Storage(format!(
                    "cannot overwrite missing vector row {row}"
                )));
            }
        }
        for group in &mut groups {
            group.sort_unstable_by_key(|(local_row, _)| *local_row);
        }
        Ok(groups)
    }

    fn encode_update_journal(&self, rows: &[u64], encoded: &[u8]) -> Result<Vec<u8>> {
        let row_width = self.row_width() as usize;
        if encoded.len() != rows.len().saturating_mul(row_width) {
            return Err(LynseError::DimensionMismatch {
                expected: rows.len().saturating_mul(row_width),
                got: encoded.len(),
            });
        }

        let record_width = 8usize
            .checked_add(row_width)
            .ok_or_else(|| LynseError::Storage("vector update journal size overflows".into()))?;
        let records_size = rows
            .len()
            .checked_mul(record_width)
            .ok_or_else(|| LynseError::Storage("vector update journal size overflows".into()))?;
        let capacity = 37usize
            .checked_add(records_size)
            .and_then(|value| value.checked_add(4))
            .ok_or_else(|| LynseError::Storage("vector update journal size overflows".into()))?;
        let mut journal = Vec::with_capacity(capacity);
        journal.extend_from_slice(UPDATE_JOURNAL_MAGIC);
        journal.extend_from_slice(&UPDATE_JOURNAL_VERSION.to_le_bytes());
        journal.extend_from_slice(&(self.dimension as u64).to_le_bytes());
        journal.push(match self.dtype {
            VectorDtype::F32 => 1,
            VectorDtype::F16 => 2,
        });
        journal.extend_from_slice(&(rows.len() as u64).to_le_bytes());
        journal.extend_from_slice(&(row_width as u64).to_le_bytes());
        for (input_index, &row) in rows.iter().enumerate() {
            journal.extend_from_slice(&row.to_le_bytes());
            journal.extend_from_slice(
                &encoded[input_index * row_width..(input_index + 1) * row_width],
            );
        }
        let checksum = crc32fast::hash(&journal);
        journal.extend_from_slice(&checksum.to_le_bytes());
        Ok(journal)
    }

    fn decode_update_journal(&self, journal: &[u8]) -> Result<PendingVectorUpdates> {
        const HEADER_SIZE: usize = 37;
        if journal.len() < HEADER_SIZE + 4 || &journal[..8] != UPDATE_JOURNAL_MAGIC {
            return Err(LynseError::Storage(
                "invalid vector update journal header".to_string(),
            ));
        }

        let payload_end = journal.len() - 4;
        let expected_checksum =
            u32::from_le_bytes(journal[payload_end..].try_into().unwrap());
        let actual_checksum = crc32fast::hash(&journal[..payload_end]);
        if actual_checksum != expected_checksum {
            return Err(LynseError::Storage(
                "vector update journal checksum mismatch".to_string(),
            ));
        }

        let version = u32::from_le_bytes(journal[8..12].try_into().unwrap());
        if version != UPDATE_JOURNAL_VERSION {
            return Err(LynseError::Storage(format!(
                "unsupported vector update journal version {version}"
            )));
        }
        let dimension = u64::from_le_bytes(journal[12..20].try_into().unwrap()) as usize;
        if dimension != self.dimension {
            return Err(LynseError::Storage(format!(
                "vector update journal dimension {dimension} does not match {}",
                self.dimension
            )));
        }
        let dtype = match journal[20] {
            1 => VectorDtype::F32,
            2 => VectorDtype::F16,
            code => {
                return Err(LynseError::Storage(format!(
                    "unsupported vector update journal dtype code {code}"
                )))
            }
        };
        if dtype != self.dtype {
            return Err(LynseError::Storage(format!(
                "vector update journal dtype {} does not match {}",
                dtype.storage_name(),
                self.dtype.storage_name()
            )));
        }
        let count = u64::from_le_bytes(journal[21..29].try_into().unwrap()) as usize;
        let row_width = u64::from_le_bytes(journal[29..37].try_into().unwrap()) as usize;
        if row_width != self.row_width() as usize {
            return Err(LynseError::Storage(format!(
                "vector update journal row width {row_width} does not match {}",
                self.row_width()
            )));
        }
        let record_width = 8usize
            .checked_add(row_width)
            .ok_or_else(|| LynseError::Storage("vector update journal size overflows".into()))?;
        let expected_len = HEADER_SIZE
            .checked_add(count.checked_mul(record_width).ok_or_else(|| {
                LynseError::Storage("vector update journal size overflows".into())
            })?)
            .and_then(|value| value.checked_add(4))
            .ok_or_else(|| LynseError::Storage("vector update journal size overflows".into()))?;
        if journal.len() != expected_len {
            return Err(LynseError::Storage(format!(
                "vector update journal length {} does not match expected {expected_len}",
                journal.len()
            )));
        }

        let mut rows = Vec::with_capacity(count);
        let mut encoded = Vec::with_capacity(count.saturating_mul(row_width));
        let mut cursor = HEADER_SIZE;
        for _ in 0..count {
            rows.push(u64::from_le_bytes(
                journal[cursor..cursor + 8].try_into().unwrap(),
            ));
            cursor += 8;
            encoded.extend_from_slice(&journal[cursor..cursor + row_width]);
            cursor += row_width;
        }
        Ok(PendingVectorUpdates { rows, encoded })
    }

    pub(crate) fn write_update_journal(&self, rows: &[u64], encoded: &[u8]) -> Result<()> {
        let journal = self.encode_update_journal(rows, encoded)?;
        write_atomic_durable(&self.update_journal_path(), &journal)
    }

    fn clear_update_journal(&self) -> Result<()> {
        let path = self.update_journal_path();
        if path.exists() {
            std::fs::remove_file(&path)?;
            if let Some(parent) = path.parent() {
                if let Ok(dir) = File::open(parent) {
                    let _ = dir.sync_all();
                }
            }
        }
        Ok(())
    }

    fn apply_encoded_updates(&self, rows: &[u64], encoded: &[u8]) -> Result<()> {
        let row_width = self.row_width() as usize;
        if encoded.len() != rows.len().saturating_mul(row_width) {
            return Err(LynseError::DimensionMismatch {
                expected: rows.len().saturating_mul(row_width),
                got: encoded.len(),
            });
        }

        let manifest = self.manifest.write();
        let groups = self.grouped_updates(&manifest, rows)?;
        let mut cache = self.mmap_cache.write();
        let mut compatibility = self.compatibility_mmap.write();
        for (segment_index, group) in groups.iter().enumerate() {
            if group.is_empty() {
                continue;
            }
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(self.collection_path.join(&manifest.segments[segment_index].file))?;
            for &(local_row, input_index) in group {
                write_all_at(
                    &file,
                    &encoded[input_index * row_width..(input_index + 1) * row_width],
                    local_row * row_width as u64,
                )?;
            }
            file.sync_all()?;
            if let Some(slot) = cache.get_mut(segment_index) {
                *slot = None;
            }
        }
        *compatibility = None;
        Ok(())
    }

    pub fn recover_pending_updates(&self) -> Result<bool> {
        let path = self.update_journal_path();
        if !path.exists() {
            return Ok(false);
        }
        let pending = self.decode_update_journal(&std::fs::read(&path)?)?;
        self.apply_encoded_updates(&pending.rows, &pending.encoded)?;
        self.clear_update_journal()?;
        Ok(true)
    }

    pub fn apply_journaled_encoded_rows(
        &self,
        rows: &[u64],
        encoded: &[u8],
        dtype: VectorDtype,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        if dtype != self.dtype {
            return Err(LynseError::InvalidArgument(format!(
                "encoded vector dtype {} does not match store dtype {}",
                dtype.storage_name(),
                self.dtype.storage_name()
            )));
        }

        // Validate every row before publishing the durable replay record.
        self.grouped_updates(&self.manifest.read(), rows)?;
        self.write_update_journal(rows, encoded)?;
        self.apply_encoded_updates(rows, encoded)?;
        Ok(())
    }

    pub fn finish_pending_updates(&self) -> Result<()> {
        self.clear_update_journal()
    }

    pub fn overwrite_encoded_rows(
        &self,
        rows: &[u64],
        encoded: &[u8],
        dtype: VectorDtype,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        self.apply_journaled_encoded_rows(rows, encoded, dtype)?;
        self.clear_update_journal()?;
        Ok(())
    }

    fn merge_results(
        &self,
        mut results: Vec<(u64, f32)>,
        k: usize,
        metric: DistanceMetric,
    ) -> (Vec<u64>, Vec<f32>) {
        results.sort_unstable_by(|a, b| {
            let order = a.1.partial_cmp(&b.1).unwrap_or(CmpOrdering::Equal);
            let order = if metric.is_ascending() { order } else { order.reverse() };
            order.then_with(|| a.0.cmp(&b.0))
        });
        results.truncate(k);
        results.into_iter().unzip()
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
        use_sq8: bool,
        approx: Option<ApproxSearchConfig>,
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        self.ensure_mmaps()?;
        let manifest = self.manifest.read();
        let cache = self.mmap_cache.read();
        let mut merged = Vec::with_capacity(k.saturating_mul(manifest.segments.len()));
        let mut base = 0u64;
        for (segment, mmap) in manifest.segments.iter().zip(cache.iter()) {
            if let Some(mmap) = mmap {
                let (rows, distances) = mmap.search(query, k, metric, use_sq8, approx);
                merged.extend(
                    rows.into_iter()
                        .zip(distances)
                        .map(|(row, distance)| (base + row as u64, distance)),
                );
            }
            base += segment.rows;
        }
        Ok(self.merge_results(merged, k, metric))
    }

    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
        subset_rows: &[u64],
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        self.ensure_mmaps()?;
        let manifest = self.manifest.read();
        let cache = self.mmap_cache.read();
        let mut merged = Vec::with_capacity(k.saturating_mul(manifest.segments.len()));
        let mut base = 0u64;
        for (segment, mmap) in manifest.segments.iter().zip(cache.iter()) {
            let end = base + segment.rows;
            if let Some(mmap) = mmap {
                let local_rows: Vec<u64> = subset_rows
                    .iter()
                    .copied()
                    .filter(|&row| row >= base && row < end)
                    .map(|row| row - base)
                    .collect();
                if !local_rows.is_empty() {
                    let (rows, distances) = mmap.search_filtered(query, k, metric, &local_rows);
                    merged.extend(
                        rows.into_iter()
                            .zip(distances)
                            .map(|(row, distance)| (base + row as u64, distance)),
                    );
                }
            }
            base = end;
        }
        Ok(self.merge_results(merged, k, metric))
    }

    pub fn replace_data(&self, data: &[f32]) -> Result<()> {
        self.replace_data_with_ids(data, None)
    }

    pub fn replace_data_with_id_map(&self, data: &[f32], ids: &[u64]) -> Result<()> {
        self.replace_data_with_ids(data, Some(ids))
    }

    fn replace_data_with_ids(&self, data: &[f32], ids: Option<&[u64]>) -> Result<()> {
        if self.dimension == 0 || data.len() % self.dimension != 0 {
            return Err(LynseError::DimensionMismatch {
                expected: self.dimension,
                got: data.len(),
            });
        }
        let bytes = encode_f32_slice_as_le_bytes(data, self.dtype);
        self.replace_encoded_generation(&bytes, data.len() / self.dimension, ids)
    }

    fn replace_encoded_generation(
        &self,
        bytes: &[u8],
        rows: usize,
        id_map: Option<&[u64]>,
    ) -> Result<()> {
        if id_map.is_some_and(|ids| ids.len() != rows) {
            return Err(LynseError::Storage(
                "replacement vector rows do not match ID-map length".to_string(),
            ));
        }
        let old_manifest = self.manifest.read().clone();
        let generation = old_manifest.generation + 1;
        let file_name = format!("{SEGMENT_DIR}/seg-{generation:020}-000000.bin");
        let path = self.collection_path.join(&file_name);
        std::fs::create_dir_all(path.parent().expect("segment parent"))?;
        write_atomic_durable(&path, bytes)?;

        let id_map_file = if let Some(ids) = id_map {
            let file = format!("id_map-{generation:020}.bin");
            let encoded: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            write_atomic_durable(&self.collection_path.join(&file), &encoded)?;
            file
        } else {
            old_manifest.id_map_file.clone()
        };
        let manifest = VectorManifest {
            version: VECTOR_MANIFEST_VERSION,
            generation,
            id_map_file,
            segments: if rows == 0 {
                Vec::new()
            } else {
                vec![SegmentEntry {
                    file: file_name,
                    rows: rows as u64,
                }]
            },
        };
        self.persist_manifest(&manifest)?;
        *self.manifest.write() = manifest;
        *self.mmap_cache.write() = if rows == 0 { Vec::new() } else { vec![None] };
        *self.compatibility_mmap.write() = None;
        self.total_vectors.store(rows as u64, Ordering::Relaxed);
        self.mark_changed(rows as u64);
        self.persist_metadata()?;
        self.cleanup_superseded(&old_manifest);
        Ok(())
    }

    pub fn compact_rows(&self, deleted_rows: &HashSet<usize>, new_ids: &[u64]) -> Result<()> {
        if deleted_rows.is_empty() {
            return Ok(());
        }
        let old_manifest = self.manifest.read().clone();
        let generation = old_manifest.generation + 1;
        let row_width = self.row_width() as usize;
        let mut new_segments = Vec::new();
        let mut base = 0usize;

        for (index, segment) in old_manifest.segments.iter().enumerate() {
            let end = base + segment.rows as usize;
            let local_deleted: Vec<usize> = deleted_rows
                .iter()
                .copied()
                .filter(|&row| row >= base && row < end)
                .map(|row| row - base)
                .collect();
            if local_deleted.is_empty() {
                new_segments.push(segment.clone());
                base = end;
                continue;
            }

            let deleted: HashSet<usize> = local_deleted.into_iter().collect();
            let source = std::fs::read(self.collection_path.join(&segment.file))?;
            let live_rows = segment.rows as usize - deleted.len();
            if live_rows > 0 {
                let mut compacted = Vec::with_capacity(live_rows * row_width);
                for row in 0..segment.rows as usize {
                    if !deleted.contains(&row) {
                        compacted.extend_from_slice(&source[row * row_width..(row + 1) * row_width]);
                    }
                }
                let file = format!("{SEGMENT_DIR}/seg-{generation:020}-{index:06}.bin");
                let path = self.collection_path.join(&file);
                std::fs::create_dir_all(path.parent().expect("segment parent"))?;
                write_atomic_durable(&path, &compacted)?;
                new_segments.push(SegmentEntry {
                    file,
                    rows: live_rows as u64,
                });
            }
            base = end;
        }

        let compacted_rows: usize = new_segments.iter().map(|segment| segment.rows as usize).sum();
        if compacted_rows != new_ids.len() {
            return Err(LynseError::Storage(format!(
                "compacted vector rows ({compacted_rows}) do not match ID-map length ({})",
                new_ids.len()
            )));
        }

        let id_map_file = format!("id_map-{generation:020}.bin");
        let id_bytes: Vec<u8> = new_ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        write_atomic_durable(&self.collection_path.join(&id_map_file), &id_bytes)?;
        let manifest = VectorManifest {
            version: VECTOR_MANIFEST_VERSION,
            generation,
            id_map_file,
            segments: new_segments,
        };
        self.persist_manifest(&manifest)?;
        *self.manifest.write() = manifest;
        *self.mmap_cache.write() = (0..self.manifest.read().segments.len())
            .map(|_| None)
            .collect();
        *self.compatibility_mmap.write() = None;
        self.total_vectors
            .store(new_ids.len() as u64, Ordering::Relaxed);
        self.mark_changed(new_ids.len() as u64);
        self.persist_metadata()?;
        self.cleanup_superseded(&old_manifest);
        Ok(())
    }

    fn cleanup_superseded(&self, old: &VectorManifest) {
        let current = self.manifest.read();
        let retained: HashSet<&str> = current.segments.iter().map(|s| s.file.as_str()).collect();
        for segment in &old.segments {
            if !retained.contains(segment.file.as_str()) {
                let _ = std::fs::remove_file(self.collection_path.join(&segment.file));
            }
        }
        if old.id_map_file != current.id_map_file && old.id_map_file != DEFAULT_ID_MAP_FILE {
            let _ = std::fs::remove_file(self.collection_path.join(&old.id_map_file));
        }
    }

    pub fn truncate_to_vectors(&self, n_vectors: usize) -> Result<()> {
        let mut manifest = self.manifest.write();
        let available: u64 = manifest.segments.iter().map(|segment| segment.rows).sum();
        if n_vectors as u64 > available {
            return Err(LynseError::Storage(format!(
                "cannot truncate vector store to {n_vectors} rows; current length is {available}"
            )));
        }

        let mut keep = n_vectors as u64;
        let mut retained = Vec::new();
        let mut remove_after_publish = Vec::new();
        for segment in &manifest.segments {
            if keep == 0 {
                remove_after_publish.push(self.collection_path.join(&segment.file));
                continue;
            }
            let rows = keep.min(segment.rows);
            if rows <= segment.rows && keep <= segment.rows {
                let path = self.collection_path.join(&segment.file);
                let file = OpenOptions::new().write(true).open(path)?;
                file.set_len(rows * self.row_width())?;
                file.sync_all()?;
            }
            let mut entry = segment.clone();
            entry.rows = rows;
            retained.push(entry);
            keep -= rows;
        }
        manifest.segments = retained;
        if self.manifest_path.exists() {
            self.persist_manifest(&manifest)?;
        }
        for path in remove_after_publish {
            let _ = std::fs::remove_file(path);
        }
        *self.mmap_cache.write() = (0..manifest.segments.len()).map(|_| None).collect();
        *self.compatibility_mmap.write() = None;
        self.total_vectors
            .store(n_vectors as u64, Ordering::Relaxed);
        self.mark_changed(n_vectors as u64);
        self.persist_metadata()?;
        Ok(())
    }

    pub fn disk_bytes(&self) -> u64 {
        self.segment_paths()
            .iter()
            .filter_map(|path| std::fs::metadata(path).ok().map(|m| m.len()))
            .sum()
    }

    pub fn file_exists(&self) -> bool {
        self.total_vectors.load(Ordering::Relaxed) > 0
    }

    pub fn get_shape(&self) -> Result<(u64, usize)> {
        Ok((self.total_vectors.load(Ordering::Relaxed), self.dimension))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segmented_roundtrip_search_and_filtered_search() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = VectorStore::new(tmp.path(), 4, 100, VectorDtype::F32).unwrap();
        let data: Vec<f32> = (0..400).map(|value| value as f32).collect();
        store.write(&data).unwrap();
        store.write(&data).unwrap();

        assert_eq!(store.get_shape().unwrap(), (200, 4));
        assert_eq!(store.read_all_f32().unwrap().len(), 800);
        let query = [0.0, 1.0, 2.0, 3.0];
        let (ids, _) = store
            .search(&query, 1, DistanceMetric::L2Squared, false, None)
            .unwrap();
        assert_eq!(ids, vec![0]);
        let (ids, _) = store
            .search_filtered(&query, 1, DistanceMetric::L2Squared, &[100])
            .unwrap();
        assert_eq!(ids, vec![100]);
    }

    #[test]
    fn compact_rewrites_only_affected_segments() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = VectorStore::new(tmp.path(), 4, 100, VectorDtype::F32).unwrap();
        let data: Vec<f32> = (0..400).map(|value| value as f32).collect();
        store.write(&data).unwrap();
        store.write(&data).unwrap();
        let before = store.segment_paths();
        let deleted = HashSet::from([0usize]);
        let ids: Vec<u64> = (1..200).collect();
        store.compact_rows(&deleted, &ids).unwrap();
        let after = store.segment_paths();

        assert_eq!(store.get_shape().unwrap().0, 199);
        assert_ne!(before[0], after[0]);
        assert_eq!(before[1], after[1]);
        assert_eq!(store.read_all_f32().unwrap().len(), 199 * 4);
    }

    #[test]
    fn truncate_removes_partial_tail_and_extra_segments() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = VectorStore::new(tmp.path(), 4, 100, VectorDtype::F32).unwrap();
        let data: Vec<f32> = (0..400).map(|value| value as f32).collect();
        store.write(&data).unwrap();
        store.write(&data).unwrap();
        store.truncate_to_vectors(50).unwrap();
        assert_eq!(store.get_shape().unwrap().0, 50);
        assert_eq!(store.segment_paths().len(), 1);
    }

    #[test]
    fn legacy_partial_row_is_trimmed_to_durable_boundary() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("vectors.bin");
        let mut bytes = encode_f32_slice_as_le_bytes(&[1.0; 8], VectorDtype::F32);
        bytes.extend_from_slice(&[1, 2, 3]);
        std::fs::write(&path, bytes).unwrap();

        let store = VectorStore::new(tmp.path(), 4, 100, VectorDtype::F32).unwrap();
        assert_eq!(store.get_shape().unwrap().0, 2);
        store.truncate_to_vectors(2).unwrap();
        assert_eq!(std::fs::metadata(path).unwrap().len(), 2 * 4 * 4);
    }

    #[test]
    fn compact_manifest_publishes_matching_id_map_generation() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = VectorStore::new(tmp.path(), 4, 100, VectorDtype::F32).unwrap();
        let data: Vec<f32> = (0..400).map(|value| value as f32).collect();
        store.write(&data).unwrap();
        std::fs::write(
            store.id_map_path(),
            (0u64..100)
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let ids: Vec<u64> = (1..100).collect();
        store.compact_rows(&HashSet::from([0]), &ids).unwrap();

        let reopened = VectorStore::new(tmp.path(), 4, 100, VectorDtype::F32).unwrap();
        let persisted_ids: Vec<u64> = std::fs::read(reopened.id_map_path())
            .unwrap()
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(persisted_ids, ids);
        assert_eq!(reopened.get_shape().unwrap().0, 99);
    }

    #[test]
    fn journaled_overwrite_updates_rows_across_segments_and_reopens() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = VectorStore::new(tmp.path(), 4, 100, VectorDtype::F32).unwrap();
        let data: Vec<f32> = (0..256).map(|value| value as f32).collect();
        store.write(&data).unwrap();
        store.write(&data).unwrap();

        let replacement = [10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0];
        let encoded = encode_f32_slice_as_le_bytes(&replacement, VectorDtype::F32);
        store
            .overwrite_encoded_rows(&[0, 100], &encoded, VectorDtype::F32)
            .unwrap();

        assert!(!store.has_pending_updates());
        assert_eq!(store.read_rows(&[0, 100]).unwrap(), replacement);
        drop(store);

        let reopened = VectorStore::new(tmp.path(), 4, 100, VectorDtype::F32).unwrap();
        assert_eq!(reopened.read_rows(&[0, 100]).unwrap(), replacement);
    }

    #[test]
    fn pending_update_journal_replays_after_partial_positional_write() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = VectorStore::new(tmp.path(), 4, 100, VectorDtype::F32).unwrap();
        let data: Vec<f32> = (0..256).map(|value| value as f32).collect();
        store.write(&data).unwrap();
        store.write(&data).unwrap();

        let replacement = [30.0, 31.0, 32.0, 33.0, 40.0, 41.0, 42.0, 43.0];
        let encoded = encode_f32_slice_as_le_bytes(&replacement, VectorDtype::F32);
        store.write_update_journal(&[0, 100], &encoded).unwrap();

        // Simulate a crash after the first target row reached its segment.
        let first_segment = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&store.segment_paths()[0])
            .unwrap();
        write_all_at(&first_segment, &encoded[..16], 0).unwrap();
        first_segment.sync_all().unwrap();
        drop(store);

        let reopened = VectorStore::new(tmp.path(), 4, 100, VectorDtype::F32).unwrap();
        assert!(reopened.has_pending_updates());
        assert!(reopened.recover_pending_updates().unwrap());
        assert!(!reopened.has_pending_updates());
        assert_eq!(reopened.read_rows(&[0, 100]).unwrap(), replacement);
    }

    #[test]
    fn journaled_overwrite_preserves_f16_storage_encoding() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = VectorStore::new(tmp.path(), 4, 100, VectorDtype::F16).unwrap();
        store.write(&[0.0; 16]).unwrap();
        let replacement = [1.5, 2.5, 3.5, 4.5];
        let encoded = encode_f32_slice_as_le_bytes(&replacement, VectorDtype::F16);
        store
            .overwrite_encoded_rows(&[2], &encoded, VectorDtype::F16)
            .unwrap();
        assert_eq!(store.read_rows(&[2]).unwrap(), replacement);
    }

    #[test]
    fn corrupt_update_journal_is_rejected_without_touching_vectors() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = VectorStore::new(tmp.path(), 4, 100, VectorDtype::F32).unwrap();
        store.write(&[1.0; 16]).unwrap();
        let encoded = encode_f32_slice_as_le_bytes(&[2.0; 4], VectorDtype::F32);
        store.write_update_journal(&[1], &encoded).unwrap();
        let path = store.update_journal_path();
        let mut journal = std::fs::read(&path).unwrap();
        journal[25] ^= 0xff;
        std::fs::write(&path, journal).unwrap();

        let error = store.recover_pending_updates().unwrap_err();
        assert!(error.to_string().contains("checksum mismatch"));
        assert_eq!(store.read_rows(&[1]).unwrap(), vec![1.0; 4]);
    }
}
