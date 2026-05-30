//! ApexBase On-Demand Columnar Format (V4)
//!
//! A custom binary file format supporting:
//! - Column projection: read only required columns
//! - Row range scan: read only required row ranges  
//! - Zero-copy reads via pread/mmap
//! - No external serialization dependencies (bincode-free)
//!
//! File Format:
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Header (256 bytes)                                          │
//! │   - Magic: "APEXV3\0\0" (8 bytes, retained for compat)      │
//! │   - Version: u32                                            │
//! │   - Flags: u32                                              │
//! │   - Row count: u64                                          │
//! │   - Column count: u32                                       │
//! │   - Row group size: u32 (rows per group, default 65536)     │
//! │   - Schema offset: u64                                      │
//! │   - Column index offset: u64                                │
//! │   - ID column offset: u64                                   │
//! │   - Timestamps, checksum, reserved                          │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Schema Block                                                │
//! │   - For each column: [name_len:u16][name:bytes][type:u8]    │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Column Index (32 bytes per column)                          │
//! │   - data_offset: u64                                        │
//! │   - data_length: u64                                        │
//! │   - null_offset: u64                                        │
//! │   - null_length: u64                                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │ ID Column (contiguous u64 array)                            │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Column Data Blocks                                          │
//! │   Per column: [null_bitmap][column_data]                    │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Footer (24 bytes)                                           │
//! │   - Magic: "APEXEND\0"                                      │
//! │   - Checksum: u32                                           │
//! │   - File size: u64                                          │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};

use arrow::array::ArrayRef;
use arrow::record_batch::RecordBatch;
use memmap2::Mmap;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::delta::DeltaStore;

/// Helper for InvalidData errors
#[inline]
fn err_data(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}
/// Helper for NotFound errors  
#[inline]
fn err_not_found(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::NotFound, msg.into())
}
/// Helper for NotConnected errors
#[inline]
fn err_not_conn(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::NotConnected, msg)
}
/// Helper for InvalidInput errors
#[inline]
fn err_input(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

// Thread-local buffer for scattered reads to avoid repeated allocations
thread_local! {
    static SCATTERED_READ_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(8192));
}

// ============================================================================
// Cross-platform file reading (mmap with pread fallback)
// ============================================================================

/// Memory-mapped file cache for fast repeated reads
/// Uses OS page cache for automatic caching
struct MmapCache {
    mmap: Option<std::sync::Arc<Mmap>>,
    file_size: u64,
}

impl MmapCache {
    fn new() -> Self {
        Self {
            mmap: None,
            file_size: 0,
        }
    }

    /// Get or create mmap Arc for the file, allowing the Arc to be cloned and held outside the lock.
    pub(crate) fn get_mmap_arc(&mut self, file: &File) -> io::Result<std::sync::Arc<Mmap>> {
        let metadata = file.metadata()?;
        let current_size = metadata.len();
        if self.mmap.is_none() || self.file_size != current_size {
            if current_size == 0 {
                return Err(err_data("Empty file"));
            }
            let mmap = unsafe {
                // On Linux, use MAP_POPULATE for files < 64MB to pre-fault pages
                // and eliminate page-fault overhead on first access.
                #[cfg(target_os = "linux")]
                {
                    if current_size < 64 * 1024 * 1024 {
                        memmap2::MmapOptions::new().populate().map(file)?
                    } else {
                        Mmap::map(file)?
                    }
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Mmap::map(file)?
                }
            };
            // On Linux, hint sequential access so the kernel doubles readahead.
            #[cfg(target_os = "linux")]
            {
                let _ = mmap.advise(memmap2::Advice::Sequential);
            }
            // On Windows: pre-fault pages to eliminate first-access page faults.
            // Three strategies based on file size:
            //  - Small files (<4MB): single-thread touch loop (rayon overhead not worthwhile).
            //  - Medium files (4MB-128MB): rayon-parallel prefault — concurrent page faults
            //    let the OS service multiple NVMe queue entries simultaneously (3-5x faster).
            //  - Large files (>=128MB): PrefetchVirtualMemory (async OS prefetch, Win8+).
            #[cfg(windows)]
            {
                let ptr = mmap.as_ptr();
                let len = mmap.len();
                if current_size < 4 * 1024 * 1024 {
                    // Small file: single-threaded touch loop.
                    let mut i = 0usize;
                    while i < len {
                        unsafe {
                            let _ = ptr.add(i).read_volatile();
                        }
                        i += 4096;
                    }
                } else if current_size < 128 * 1024 * 1024 {
                    // Medium file: rayon-parallel prefault.
                    // Each thread touches a contiguous range of pages, generating
                    // concurrent page faults that the OS can batch into parallel I/O.
                    let ptr_usize = ptr as usize;
                    let num_pages = (len + 4095) / 4096;
                    let t = rayon::current_num_threads().max(1);
                    let pages_per_thread = (num_pages + t - 1) / t;
                    rayon::scope(|s| {
                        for tid in 0..t {
                            let start_page = tid * pages_per_thread;
                            if start_page >= num_pages {
                                break;
                            }
                            let end_page = (start_page + pages_per_thread).min(num_pages);
                            let file_len = len;
                            s.spawn(move |_| {
                                for page in start_page..end_page {
                                    let offset = page * 4096;
                                    if offset < file_len {
                                        unsafe {
                                            let _ = (ptr_usize as *const u8)
                                                .add(offset)
                                                .read_volatile();
                                        }
                                    }
                                }
                            });
                        }
                    });
                } else {
                    // Large file: async prefault via PrefetchVirtualMemory (Windows 8+).
                    // Tells the OS to bring pages into the working set without blocking.
                    #[repr(C)]
                    struct MemoryRangeEntry {
                        address: *const u8,
                        size: usize,
                    }
                    extern "system" {
                        fn GetCurrentProcess() -> isize;
                        fn PrefetchVirtualMemory(
                            process: isize,
                            count: usize,
                            ranges: *const MemoryRangeEntry,
                            flags: u32,
                        ) -> i32;
                    }
                    let entry = MemoryRangeEntry {
                        address: ptr,
                        size: len,
                    };
                    unsafe {
                        PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
                    }
                }
            }
            self.mmap = Some(std::sync::Arc::new(mmap));
            self.file_size = current_size;
        }
        Ok(std::sync::Arc::clone(self.mmap.as_ref().unwrap()))
    }

    /// Get or create mmap for the file
    fn get_or_create(&mut self, file: &File) -> io::Result<&Mmap> {
        self.get_mmap_arc(file)?;
        Ok(self.mmap.as_ref().unwrap())
    }

    /// Return mmap slice without any syscalls or staleness checks.
    /// Caller must ensure the mmap is still valid (backend freshness checked by STORAGE_CACHE).
    pub(crate) fn mmap_slice_unchecked(&self) -> Option<&[u8]> {
        self.mmap.as_ref().map(|arc| -> &[u8] { arc.as_ref() })
    }

    /// Read bytes at offset using mmap (zero-copy when possible)
    fn read_at(&mut self, file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
        let mmap = self.get_or_create(file)?;
        let start = offset as usize;
        let end = start + buf.len();

        if end > mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "Read past EOF: offset={}, len={}, file_size={}",
                    offset,
                    buf.len(),
                    mmap.len()
                ),
            ));
        }

        buf.copy_from_slice(&mmap[start..end]);
        Ok(())
    }

    /// Get a slice directly from mmap (true zero-copy)
    fn slice(&mut self, file: &File, offset: u64, len: usize) -> io::Result<&[u8]> {
        let mmap = self.get_or_create(file)?;
        let start = offset as usize;
        let end = start + len;

        if end > mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "Slice past EOF: offset={}, len={}, file_size={}",
                    offset,
                    len,
                    mmap.len()
                ),
            ));
        }

        Ok(&mmap[start..end])
    }

    /// Invalidate cache (call after writes)
    fn invalidate(&mut self) {
        self.mmap = None;
        self.file_size = 0;
    }

    /// Return mmap size without any syscalls.
    pub(crate) fn mmap_len(&self) -> usize {
        self.mmap.as_ref().map(|m| m.len()).unwrap_or(0)
    }
}

/// Open a file optimised for sequential access.
/// On Windows, adds FILE_FLAG_SEQUENTIAL_SCAN (0x08000000) so the OS doubles
/// read-ahead and avoids random-access caching overhead.
/// On Linux/macOS, equivalent to `File::open` (mmap + MADV_SEQUENTIAL handles this).
pub(crate) fn open_for_sequential_read(path: &Path) -> io::Result<File> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;
        OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_SEQUENTIAL_SCAN)
            .open(path)
    }
    #[cfg(not(windows))]
    {
        File::open(path)
    }
}

/// Cross-platform positioned read (fallback for when mmap is not available)
#[cfg(unix)]
fn pread_fallback(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(windows)]
fn pread_fallback(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut total_read = 0;
    while total_read < buf.len() {
        let n = file.seek_read(&mut buf[total_read..], offset + total_read as u64)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
        }
        total_read += n;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn pread_fallback(file: &mut File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    // Generic fallback using seek + read (not thread-safe)
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}

// ============================================================================
// Constants
// ============================================================================

const MAGIC: &[u8; 8] = b"APEXV3\0\0";
const FORMAT_VERSION_V4: u32 = 4;
const HEADER_SIZE: usize = 256;
const COLUMN_INDEX_ENTRY_SIZE: usize = 32;
const DEFAULT_ROW_GROUP_SIZE: u32 = 65536;

// V4 Row Group format constants
const MAGIC_ROW_GROUP: &[u8; 4] = b"APXG";
const MAGIC_V4_FOOTER: &[u8; 8] = b"APXFOOT\0";
/// Size of a serialized RowGroupMeta entry in the footer (8+8+4+8+8+4 = 40 bytes)
const ROW_GROUP_META_SIZE: usize = 40;

// Per-RG compression flags (stored in RG header byte 28)
const RG_COMPRESS_NONE: u8 = 0;
const RG_COMPRESS_LZ4: u8 = 1;
const RG_COMPRESS_ZSTD: u8 = 2;

/// Minimum RG body size (bytes) to bother compressing.
/// Below this threshold, compression overhead exceeds savings.
const COMPRESS_MIN_BODY_SIZE: usize = 512;

// Header flags field: bits 0-1 encode CompressionType
const FLAG_COMPRESS_MASK: u32 = 0b11;
const FLAG_COMPRESS_NONE: u32 = 0;
const FLAG_COMPRESS_LZ4: u32 = 1;
const FLAG_COMPRESS_ZSTD: u32 = 2;

/// Compression algorithm for Row Group bodies.
/// Default is `None` (no compression) for maximum read performance.
/// Can only be changed on empty tables; persisted in header flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    /// No compression (default). Best read performance.
    None,
    /// LZ4 block compression. Fast with moderate ratio.
    Lz4,
    /// Zstd compression (level 1). Better ratio, slower than LZ4.
    Zstd,
}

impl CompressionType {
    fn from_flags(flags: u32) -> Self {
        match flags & FLAG_COMPRESS_MASK {
            FLAG_COMPRESS_LZ4 => CompressionType::Lz4,
            FLAG_COMPRESS_ZSTD => CompressionType::Zstd,
            _ => CompressionType::None,
        }
    }

    fn to_flags_bits(self) -> u32 {
        match self {
            CompressionType::None => FLAG_COMPRESS_NONE,
            CompressionType::Lz4 => FLAG_COMPRESS_LZ4,
            CompressionType::Zstd => FLAG_COMPRESS_ZSTD,
        }
    }

    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(CompressionType::None),
            "lz4" => Some(CompressionType::Lz4),
            "zstd" | "zstandard" => Some(CompressionType::Zstd),
            _ => Option::None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            CompressionType::None => "none",
            CompressionType::Lz4 => "lz4",
            CompressionType::Zstd => "zstd",
        }
    }
}

// Per-column encoding types (stored as 1-byte prefix when encoding_version=1)
//
// Column compression encodings (0-15):
const COL_ENCODING_PLAIN: u8 = 0;
const COL_ENCODING_RLE: u8 = 1;
const COL_ENCODING_BITPACK: u8 = 2;
const COL_ENCODING_RLE_BOOL: u8 = 3;

// Extended encodings for compatibility with other systems (16-31):
const COL_ENCODING_FORWARD: u8 = 16; // Forward encoding (delta)
const COL_ENCODING_DICTIONARY: u8 = 17; // Dictionary encoding for strings
const COL_ENCODING_UNENCODED: u8 = 18; // Unencoded (raw bytes)

// Character encoding hints for string columns (typically stored in schema, not per-column):
// These are not column compression encodings, but we handle them gracefully if encountered.
const CHAR_ENCODING_UTF8: u8 = 1; // UTF-8 (common)
const CHAR_ENCODING_ASCII: u8 = 0; // ASCII (7-bit)
const CHAR_ENCODING_LATIN1: u8 = 208; // ISO-8859-1 (Latin-1)
const CHAR_ENCODING_UTF16: u8 = 209; // UTF-16

/// Encode an Int64 column with RLE (Run-Length Encoding).
/// Format: [count:u64][num_runs:u64][(value:i64, run_len:u32)...]
/// Returns None if RLE is not beneficial (fewer than 30% compression).
fn rle_encode_i64(data: &[i64]) -> Option<Vec<u8>> {
    if data.len() < 16 {
        return None;
    }
    let mut runs: Vec<(i64, u32)> = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let val = data[i];
        let mut run_len: u32 = 1;
        while (i + run_len as usize) < data.len() && data[i + run_len as usize] == val {
            run_len += 1;
        }
        runs.push((val, run_len));
        i += run_len as usize;
    }
    // Only use RLE if it saves space: runs * 12 bytes < original count * 8 bytes
    let rle_size = 16 + runs.len() * 12; // 8 (count) + 8 (num_runs) + runs * (8+4)
    let plain_size = 8 + data.len() * 8; // 8 (count) + data * 8
    if rle_size >= (plain_size * 7 / 10) {
        return None;
    } // Need at least 30% savings
    let mut buf = Vec::with_capacity(rle_size);
    buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(runs.len() as u64).to_le_bytes());
    for (val, len) in &runs {
        buf.extend_from_slice(&val.to_le_bytes());
        buf.extend_from_slice(&len.to_le_bytes());
    }
    Some(buf)
}

/// Decode RLE-encoded Int64 data back to plain Vec<i64>.
fn rle_decode_i64(bytes: &[u8]) -> io::Result<(Vec<i64>, usize)> {
    if bytes.len() < 16 {
        return Err(err_data("RLE Int64: truncated header"));
    }
    let count = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let num_runs = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let body_len = num_runs * 12;
    if bytes.len() < 16 + body_len {
        return Err(err_data("RLE Int64: truncated runs"));
    }
    let mut result = Vec::with_capacity(count);
    let mut pos = 16;
    for _ in 0..num_runs {
        let val = i64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        let run_len = u32::from_le_bytes(bytes[pos + 8..pos + 12].try_into().unwrap()) as usize;
        result.extend(std::iter::repeat(val).take(run_len));
        pos += 12;
    }
    Ok((result, pos))
}

/// Encode an Int64 column with Bit-packing.
/// Format: [count:u64][bit_width:u8][min_value:i64][packed_data...]
/// Stores (value - min_value) in bit_width bits each.
/// Returns None if bit-packing doesn't save enough space (need < 48-bit width).
fn bitpack_encode_i64(data: &[i64]) -> Option<Vec<u8>> {
    if data.len() < 16 {
        return None;
    }
    let min_val = *data.iter().min()?;
    let max_val = *data.iter().max()?;
    if min_val == max_val {
        // All same value — RLE handles this better
        return None;
    }
    let range = (max_val as u128).wrapping_sub(min_val as u128);
    if range > u64::MAX as u128 {
        return None;
    }
    let range_u64 = range as u64;
    let bit_width = 64 - range_u64.leading_zeros(); // bits needed
    if bit_width == 0 || bit_width >= 48 {
        return None;
    } // Need meaningful savings
    let packed_bits = data.len() as u64 * bit_width as u64;
    let packed_bytes = ((packed_bits + 7) / 8) as usize;
    // Header: 8 (count) + 1 (bit_width) + 8 (min_value) = 17 bytes
    let total_size = 17 + packed_bytes;
    let plain_size = 8 + data.len() * 8;
    if total_size >= (plain_size * 7 / 10) {
        return None;
    } // Need at least 30% savings
    let mut buf = Vec::with_capacity(total_size);
    buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
    buf.push(bit_width as u8);
    buf.extend_from_slice(&min_val.to_le_bytes());
    // Pack values
    let mut packed = vec![0u8; packed_bytes];
    let bw = bit_width as usize;
    for (i, &val) in data.iter().enumerate() {
        let delta = (val - min_val) as u64;
        let bit_offset = i * bw;
        for b in 0..bw {
            if (delta >> b) & 1 == 1 {
                let global_bit = bit_offset + b;
                packed[global_bit / 8] |= 1 << (global_bit % 8);
            }
        }
    }
    buf.extend_from_slice(&packed);
    Some(buf)
}

/// Decode Bit-packed Int64 data back to plain Vec<i64>.
fn bitpack_decode_i64(bytes: &[u8]) -> io::Result<(Vec<i64>, usize)> {
    if bytes.len() < 17 {
        return Err(err_data("BitPack Int64: truncated header"));
    }
    let count = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let bit_width = bytes[8] as usize;
    let min_val = i64::from_le_bytes(bytes[9..17].try_into().unwrap());
    let packed_bits = count * bit_width;
    let packed_bytes = (packed_bits + 7) / 8;
    if bytes.len() < 17 + packed_bytes {
        return Err(err_data("BitPack Int64: truncated packed data"));
    }
    let packed = &bytes[17..17 + packed_bytes];
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let bit_offset = i * bit_width;
        let mut delta: u64 = 0;
        for b in 0..bit_width {
            let global_bit = bit_offset + b;
            if (packed[global_bit / 8] >> (global_bit % 8)) & 1 == 1 {
                delta |= 1u64 << b;
            }
        }
        result.push(min_val.wrapping_add(delta as i64));
    }
    Ok((result, 17 + packed_bytes))
}

/// Decode a single value at `idx` from BITPACK bytes without allocating the full Vec.
/// Returns None if out of bounds or malformed.
pub(super) fn bitpack_decode_at_idx(bytes: &[u8], idx: usize) -> Option<i64> {
    if bytes.len() < 17 {
        return None;
    }
    let count = u64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
    let bit_width = bytes[8] as usize;
    let min_val = i64::from_le_bytes(bytes[9..17].try_into().ok()?);
    if idx >= count {
        return None;
    }
    if bit_width == 0 {
        return Some(min_val);
    }
    let packed_bytes = (count * bit_width + 7) / 8;
    if bytes.len() < 17 + packed_bytes {
        return None;
    }
    let packed = &bytes[17..17 + packed_bytes];
    let bit_offset = idx * bit_width;
    let mut delta: u64 = 0;
    for b in 0..bit_width {
        let global_bit = bit_offset + b;
        if (packed[global_bit / 8] >> (global_bit % 8)) & 1 == 1 {
            delta |= 1u64 << b;
        }
    }
    Some(min_val.wrapping_add(delta as i64))
}

/// Compute (sum, min, max, count, bytes_consumed) directly from BITPACK bytes without Vec allocation.
/// Returns None if input is too short or bit_width is unsupported.
pub(super) fn bitpack_agg_i64(bytes: &[u8]) -> Option<(i64, i64, i64, usize, usize)> {
    if bytes.len() < 17 {
        return None;
    }
    let count = u64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
    let bit_width = bytes[8] as usize;
    let min_val = i64::from_le_bytes(bytes[9..17].try_into().ok()?);
    if bit_width == 0 {
        let consumed = 17;
        return Some((count as i64 * min_val, min_val, min_val, count, consumed));
    }
    let packed_bytes = (count * bit_width + 7) / 8;
    if bytes.len() < 17 + packed_bytes {
        return None;
    }
    let packed = &bytes[17..17 + packed_bytes];
    let mut sum_delta: u64 = 0;
    let mut min_delta: u64 = u64::MAX;
    let mut max_delta: u64 = 0;
    let mask = (1u64 << bit_width) - 1;

    if bit_width == 6 {
        // VECTORIZABLE FAST PATH for 6-bit values (age 0-63, packed 8 per 6 bytes).
        let n_full = count / 8;
        let mut sums = [0u64; 8];
        let mut mins = [u64::MAX; 8];
        let mut maxs = [0u64; 8];
        for g in 0..n_full {
            let off = g * 6;
            if off + 6 > packed.len() {
                break;
            }
            let b = &packed[off..];
            let word48 = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], 0, 0]);
            for k in 0..8usize {
                let delta = (word48 >> (k * 6)) & 0x3Fu64;
                sums[k] = sums[k].wrapping_add(delta);
                mins[k] = mins[k].min(delta);
                maxs[k] = maxs[k].max(delta);
            }
        }
        sum_delta = sums.iter().copied().sum();
        min_delta = mins.iter().copied().min().unwrap_or(u64::MAX);
        max_delta = maxs.iter().copied().max().unwrap_or(0);
        for i in (n_full * 8)..count {
            let bit_offset = i * 6;
            let byte_idx = bit_offset / 8;
            let bit_in_byte = bit_offset % 8;
            let b0 = packed.get(byte_idx).copied().unwrap_or(0) as u64;
            let b1 = packed.get(byte_idx + 1).copied().unwrap_or(0) as u64;
            let delta = ((b0 | (b1 << 8)) >> bit_in_byte) & 0x3F;
            sum_delta = sum_delta.wrapping_add(delta);
            min_delta = min_delta.min(delta);
            max_delta = max_delta.max(delta);
        }
    } else if bit_width == 7 {
        // VECTORIZABLE FAST PATH for 7-bit values (most common: age 0-127).
        // Process 8 values per 7-byte chunk using a 56-bit word — inner k-loop has
        // independent accumulators, enabling LLVM to auto-vectorize with SIMD.
        let n_full = count / 8;
        let mut sums = [0u64; 8];
        let mut mins = [u64::MAX; 8];
        let mut maxs = [0u64; 8];
        for g in 0..n_full {
            let off = g * 7;
            if off + 7 > packed.len() {
                break;
            }
            let b = &packed[off..off + 7];
            let word56 = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], 0]);
            for k in 0..8usize {
                let delta = (word56 >> (k * 7)) & 0x7Fu64;
                sums[k] = sums[k].wrapping_add(delta);
                mins[k] = mins[k].min(delta);
                maxs[k] = maxs[k].max(delta);
            }
        }
        sum_delta = sums.iter().copied().sum();
        min_delta = mins.iter().copied().min().unwrap_or(u64::MAX);
        max_delta = maxs.iter().copied().max().unwrap_or(0);
        // Handle remaining values (0-7 leftover)
        for i in (n_full * 8)..count {
            let bit_offset = i * 7;
            let byte_idx = bit_offset / 8;
            let bit_in_byte = bit_offset % 8;
            let b0 = packed[byte_idx] as u64;
            let b1 = if byte_idx + 1 < packed.len() {
                packed[byte_idx + 1] as u64
            } else {
                0
            };
            let delta = ((b0 | (b1 << 8)) >> bit_in_byte) & 0x7F;
            sum_delta = sum_delta.wrapping_add(delta);
            min_delta = min_delta.min(delta);
            max_delta = max_delta.max(delta);
        }
    } else if bit_width == 8 {
        // Trivial: 1 byte per value — vectorizes easily
        for &b in packed.iter().take(count) {
            let delta = b as u64;
            sum_delta = sum_delta.wrapping_add(delta);
            min_delta = min_delta.min(delta);
            max_delta = max_delta.max(delta);
        }
    } else if bit_width <= 16 {
        // General word-level fast path — scalar but avoids inner bit-by-bit loop
        for i in 0..count {
            let bit_offset = i * bit_width;
            let byte_idx = bit_offset / 8;
            let bit_in_byte = bit_offset % 8;
            let word = unsafe {
                let p = packed.as_ptr().add(byte_idx);
                if byte_idx + 3 <= packed.len() {
                    let b0 = *p as u32;
                    let b1 = *p.add(1) as u32;
                    let b2 = *p.add(2) as u32;
                    b0 | (b1 << 8) | (b2 << 16)
                } else if byte_idx + 2 <= packed.len() {
                    let b0 = *p as u32;
                    let b1 = *p.add(1) as u32;
                    b0 | (b1 << 8)
                } else if byte_idx < packed.len() {
                    *p as u32
                } else {
                    break;
                }
            };
            let delta = ((word as u64) >> bit_in_byte) & mask;
            sum_delta = sum_delta.wrapping_add(delta);
            min_delta = min_delta.min(delta);
            max_delta = max_delta.max(delta);
        }
    } else {
        // GENERAL PATH: bit-by-bit extraction for wide values
        for i in 0..count {
            let bit_offset = i * bit_width;
            let mut delta: u64 = 0;
            for b in 0..bit_width {
                let global_bit = bit_offset + b;
                if (packed[global_bit / 8] >> (global_bit % 8)) & 1 == 1 {
                    delta |= 1u64 << b;
                }
            }
            sum_delta = sum_delta.wrapping_add(delta);
            if delta < min_delta {
                min_delta = delta;
            }
            if delta > max_delta {
                max_delta = delta;
            }
        }
    }
    let sum = (count as i64)
        .wrapping_mul(min_val)
        .wrapping_add(sum_delta as i64);
    let mn = min_val.wrapping_add(if count > 0 { min_delta as i64 } else { 0 });
    let mx = min_val.wrapping_add(if count > 0 { max_delta as i64 } else { 0 });
    Some((sum, mn, mx, count, 17 + packed_bytes))
}

/// Encode a Bool column with RLE (Run-Length Encoding).
/// Format: [count:u64][num_runs:u64][(value:u8, run_len:u32)...]
/// Bool columns are stored as packed bits; RLE encodes runs of true/false.
/// Returns None if RLE is not beneficial (fewer than 30% compression).
fn rle_encode_bool(data: &[u8], len: usize) -> Option<Vec<u8>> {
    if len < 16 {
        return None;
    }
    let mut runs: Vec<(u8, u32)> = Vec::new();
    let mut i = 0;
    while i < len {
        let val = if (data[i / 8] >> (i % 8)) & 1 == 1 {
            1u8
        } else {
            0u8
        };
        let mut run_len: u32 = 1;
        while (i + run_len as usize) < len {
            let next_bit = (data[(i + run_len as usize) / 8] >> ((i + run_len as usize) % 8)) & 1;
            if (next_bit == 1) != (val == 1) {
                break;
            }
            run_len += 1;
        }
        runs.push((val, run_len));
        i += run_len as usize;
    }
    // RLE size: 16 header + 5 per run (1 byte val + 4 byte len)
    let rle_size = 16 + runs.len() * 5;
    let plain_size = 8 + (len + 7) / 8; // 8 (count) + packed_bits
    if rle_size >= (plain_size * 7 / 10) {
        return None;
    } // Need ≥30% savings
    let mut buf = Vec::with_capacity(rle_size);
    buf.extend_from_slice(&(len as u64).to_le_bytes());
    buf.extend_from_slice(&(runs.len() as u64).to_le_bytes());
    for (val, run_len) in &runs {
        buf.push(*val);
        buf.extend_from_slice(&run_len.to_le_bytes());
    }
    Some(buf)
}

/// Decode RLE-encoded Bool data back to ColumnData::Bool.
fn rle_decode_bool(bytes: &[u8]) -> io::Result<(ColumnData, usize)> {
    if bytes.len() < 16 {
        return Err(err_data("RLE Bool: truncated header"));
    }
    let count = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let num_runs = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let body_len = num_runs * 5;
    if bytes.len() < 16 + body_len {
        return Err(err_data("RLE Bool: truncated runs"));
    }
    let packed_byte_len = (count + 7) / 8;
    let mut data = vec![0u8; packed_byte_len];
    let mut pos = 16;
    let mut bit_idx = 0;
    for _ in 0..num_runs {
        let val = bytes[pos];
        let run_len = u32::from_le_bytes(bytes[pos + 1..pos + 5].try_into().unwrap()) as usize;
        if val == 1 {
            for j in 0..run_len {
                let bi = bit_idx + j;
                data[bi / 8] |= 1 << (bi % 8);
            }
        }
        bit_idx += run_len;
        pos += 5;
    }
    Ok((ColumnData::Bool { data, len: count }, pos))
}

/// Write a column with encoding prefix: [encoding:u8][encoded_data...]
/// Tries RLE → Bit-pack → Plain, picks the smallest encoding.
fn write_column_encoded<W: Write>(
    col: &ColumnData,
    col_type: ColumnType,
    writer: &mut W,
) -> io::Result<()> {
    match col {
        ColumnData::Int64(data) => {
            // Try RLE first (best for sorted/low-cardinality)
            if let Some(rle_bytes) = rle_encode_i64(data) {
                // Try Bit-pack too and pick the smaller
                if let Some(bp_bytes) = bitpack_encode_i64(data) {
                    if bp_bytes.len() < rle_bytes.len() {
                        writer.write_all(&[COL_ENCODING_BITPACK])?;
                        return writer.write_all(&bp_bytes);
                    }
                }
                writer.write_all(&[COL_ENCODING_RLE])?;
                return writer.write_all(&rle_bytes);
            }
            // Try Bit-pack alone
            if let Some(bp_bytes) = bitpack_encode_i64(data) {
                writer.write_all(&[COL_ENCODING_BITPACK])?;
                return writer.write_all(&bp_bytes);
            }
            // Fallback: plain
            writer.write_all(&[COL_ENCODING_PLAIN])?;
            col.write_to(writer)
        }
        ColumnData::Bool { data, len } => {
            // Try Bool RLE (best for long runs of true/false)
            if let Some(rle_bytes) = rle_encode_bool(data, *len) {
                writer.write_all(&[COL_ENCODING_RLE_BOOL])?;
                return writer.write_all(&rle_bytes);
            }
            writer.write_all(&[COL_ENCODING_PLAIN])?;
            col.write_to(writer)
        }
        ColumnData::FixedList { .. } => {
            // FixedList: always plain — f32 data is already random (no benefit from RLE/bitpack)
            writer.write_all(&[COL_ENCODING_PLAIN])?;
            col.write_to(writer)
        }
        _ => {
            // Other types: always plain
            writer.write_all(&[COL_ENCODING_PLAIN])?;
            col.write_to(writer)
        }
    }
}

/// Read only the first `limit` rows from an encoded column.
/// Returns (partial ColumnData, total bytes consumed for skipping past the full column).
/// For plain encoding, avoids allocating the full column.
/// For RLE/bitpack, falls back to full decode then slice.
fn read_column_encoded_partial(
    bytes: &[u8],
    col_type: ColumnType,
    limit: usize,
) -> io::Result<(ColumnData, usize)> {
    if bytes.is_empty() {
        return Err(err_data("read_column_encoded_partial: empty input"));
    }
    let encoding = bytes[0];
    let data_bytes = &bytes[1..];

    // For BITPACK: partial decode — only decode first `limit` values (avoids decoding full RG)
    if encoding == COL_ENCODING_BITPACK
        && matches!(
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
        )
    {
        if data_bytes.len() >= 17 {
            let count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
            let bit_width = data_bytes[8] as usize;
            let min_val = i64::from_le_bytes(data_bytes[9..17].try_into().unwrap());
            let packed_bytes = (count * bit_width + 7) / 8;
            let total_consumed = 1 + 17 + packed_bytes;
            let actual = count.min(limit);
            let mut result = Vec::with_capacity(actual);
            if bit_width == 0 {
                result.resize(actual, min_val);
            } else if 17 + packed_bytes <= data_bytes.len() {
                let packed = &data_bytes[17..17 + packed_bytes];
                for i in 0..actual {
                    let bit_offset = i * bit_width;
                    let mut delta: u64 = 0;
                    for b in 0..bit_width {
                        let global_bit = bit_offset + b;
                        if (packed[global_bit / 8] >> (global_bit % 8)) & 1 == 1 {
                            delta |= 1u64 << b;
                        }
                    }
                    result.push(min_val.wrapping_add(delta as i64));
                }
            }
            return Ok((ColumnData::Int64(result), total_consumed));
        }
    }

    // For other non-plain encodings, fall back to full decode + slice
    if encoding != COL_ENCODING_PLAIN {
        let (col, consumed) = read_column_encoded(bytes, col_type)?;
        let sliced = if col.len() > limit {
            col.slice_range(0, limit)
        } else {
            col
        };
        return Ok((sliced, consumed));
    }

    // Plain encoding: read only `limit` rows
    let total_consumed = 1 + ColumnData::skip_bytes_typed(data_bytes, col_type)?;

    match col_type {
        ColumnType::Int64
        | ColumnType::Int8
        | ColumnType::Int16
        | ColumnType::Int32
        | ColumnType::UInt8
        | ColumnType::UInt16
        | ColumnType::UInt32
        | ColumnType::UInt64
        | ColumnType::Timestamp
        | ColumnType::Date => {
            if data_bytes.len() < 8 {
                return Err(err_data("partial Int64: truncated"));
            }
            let count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
            let actual = count.min(limit);
            let byte_len = actual * 8;
            if 8 + byte_len > data_bytes.len() {
                return Err(err_data("partial Int64: data truncated"));
            }
            let mut v = vec![0i64; actual];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data_bytes[8..].as_ptr(),
                    v.as_mut_ptr() as *mut u8,
                    byte_len,
                );
            }
            Ok((ColumnData::Int64(v), total_consumed))
        }
        ColumnType::Float64 | ColumnType::Float32 => {
            if data_bytes.len() < 8 {
                return Err(err_data("partial Float64: truncated"));
            }
            let count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
            let actual = count.min(limit);
            let byte_len = actual * 8;
            if 8 + byte_len > data_bytes.len() {
                return Err(err_data("partial Float64: data truncated"));
            }
            let mut v = vec![0f64; actual];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data_bytes[8..].as_ptr(),
                    v.as_mut_ptr() as *mut u8,
                    byte_len,
                );
            }
            Ok((ColumnData::Float64(v), total_consumed))
        }
        ColumnType::Bool => {
            if data_bytes.len() < 8 {
                return Err(err_data("partial Bool: truncated"));
            }
            let len = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
            let actual = len.min(limit);
            let byte_len = (actual + 7) / 8;
            if 8 + byte_len > data_bytes.len() {
                return Err(err_data("partial Bool: data truncated"));
            }
            let data = data_bytes[8..8 + byte_len].to_vec();
            Ok((ColumnData::Bool { data, len: actual }, total_consumed))
        }
        ColumnType::String => {
            if data_bytes.len() < 8 {
                return Err(err_data("partial String: truncated"));
            }
            let count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
            let actual = count.min(limit);
            let all_offsets_len = (count + 1) * 4;
            if 8 + all_offsets_len > data_bytes.len() {
                return Err(err_data("partial String: offsets truncated"));
            }
            // Read only first actual+1 offsets
            let mut offsets = vec![0u32; actual + 1];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data_bytes[8..].as_ptr(),
                    offsets.as_mut_ptr() as *mut u8,
                    (actual + 1) * 4,
                );
            }
            let data_len_off = 8 + all_offsets_len;
            if data_len_off + 8 > data_bytes.len() {
                return Err(err_data("partial String: data_len truncated"));
            }
            let data_start = data_len_off + 8;
            let base = offsets[0] as usize;
            let end = offsets[actual] as usize;
            if data_start + end > data_bytes.len() {
                return Err(err_data("partial String: data truncated"));
            }
            let data = data_bytes[data_start + base..data_start + end].to_vec();
            // Adjust offsets to be zero-based
            if base > 0 {
                for o in offsets.iter_mut() {
                    *o -= base as u32;
                }
            }
            Ok((ColumnData::String { offsets, data }, total_consumed))
        }
        ColumnType::StringDict => {
            if data_bytes.len() < 16 {
                return Err(err_data("partial StringDict: truncated"));
            }
            let row_count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
            let dict_size = u64::from_le_bytes(data_bytes[8..16].try_into().unwrap()) as usize;
            let actual = row_count.min(limit);
            // Read only first `actual` indices
            let all_indices_len = row_count * 4;
            if 16 + all_indices_len > data_bytes.len() {
                return Err(err_data("partial StringDict: indices truncated"));
            }
            let mut indices = vec![0u32; actual];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data_bytes[16..].as_ptr(),
                    indices.as_mut_ptr() as *mut u8,
                    actual * 4,
                );
            }
            // Read full dictionary (small)
            let dict_off_start = 16 + all_indices_len;
            let dict_offsets_len = dict_size * 4;
            if dict_off_start + dict_offsets_len > data_bytes.len() {
                return Err(err_data("partial StringDict: dict_offsets truncated"));
            }
            let mut dict_offsets = vec![0u32; dict_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data_bytes[dict_off_start..].as_ptr(),
                    dict_offsets.as_mut_ptr() as *mut u8,
                    dict_offsets_len,
                );
            }
            let dict_data_len_off = dict_off_start + dict_offsets_len;
            if dict_data_len_off + 8 > data_bytes.len() {
                return Err(err_data("partial StringDict: dict_data_len truncated"));
            }
            let dict_data_len = u64::from_le_bytes(
                data_bytes[dict_data_len_off..dict_data_len_off + 8]
                    .try_into()
                    .unwrap(),
            ) as usize;
            let dict_data_start = dict_data_len_off + 8;
            if dict_data_start + dict_data_len > data_bytes.len() {
                return Err(err_data("partial StringDict: dict_data truncated"));
            }
            let dict_data = data_bytes[dict_data_start..dict_data_start + dict_data_len].to_vec();
            Ok((
                ColumnData::StringDict {
                    indices,
                    dict_offsets,
                    dict_data,
                },
                total_consumed,
            ))
        }
        _ => {
            // Fallback: full decode + slice
            let (col, consumed) = ColumnData::from_bytes_typed(data_bytes, col_type)?;
            let sliced = if col.len() > limit {
                col.slice_range(0, limit)
            } else {
                col
            };
            Ok((sliced, 1 + consumed))
        }
    }
}

/// Read a column with encoding prefix: [encoding:u8][encoded_data...]
/// Returns (ColumnData, bytes_consumed) including the encoding prefix byte.
fn read_column_encoded(bytes: &[u8], col_type: ColumnType) -> io::Result<(ColumnData, usize)> {
    if bytes.is_empty() {
        return Err(err_data("read_column_encoded: empty input"));
    }
    let encoding = bytes[0];
    let data_bytes = &bytes[1..];
    match encoding {
        COL_ENCODING_PLAIN => {
            let (col, consumed) = ColumnData::from_bytes_typed(data_bytes, col_type)?;
            Ok((col, 1 + consumed))
        }
        COL_ENCODING_RLE => {
            match col_type {
                ColumnType::Int64
                | ColumnType::Int8
                | ColumnType::Int16
                | ColumnType::Int32
                | ColumnType::UInt8
                | ColumnType::UInt16
                | ColumnType::UInt32
                | ColumnType::UInt64
                | ColumnType::Timestamp
                | ColumnType::Date => {
                    let (data, consumed) = rle_decode_i64(data_bytes)?;
                    Ok((ColumnData::Int64(data), 1 + consumed))
                }
                _ => {
                    // RLE for non-integer type — fallback to plain
                    let (col, consumed) = ColumnData::from_bytes_typed(data_bytes, col_type)?;
                    Ok((col, 1 + consumed))
                }
            }
        }
        COL_ENCODING_BITPACK => match col_type {
            ColumnType::Int64
            | ColumnType::Int8
            | ColumnType::Int16
            | ColumnType::Int32
            | ColumnType::UInt8
            | ColumnType::UInt16
            | ColumnType::UInt32
            | ColumnType::UInt64
            | ColumnType::Timestamp
            | ColumnType::Date => {
                let (data, consumed) = bitpack_decode_i64(data_bytes)?;
                Ok((ColumnData::Int64(data), 1 + consumed))
            }
            _ => {
                let (col, consumed) = ColumnData::from_bytes_typed(data_bytes, col_type)?;
                Ok((col, 1 + consumed))
            }
        },
        COL_ENCODING_RLE_BOOL => {
            let (col, consumed) = rle_decode_bool(data_bytes)?;
            Ok((col, 1 + consumed))
        }
        // Handle extended encodings - fallback to plain for compatibility
        COL_ENCODING_FORWARD | COL_ENCODING_DICTIONARY | COL_ENCODING_UNENCODED => {
            // Treat as plain encoding for backward compatibility
            let (col, consumed) = ColumnData::from_bytes_typed(data_bytes, col_type)?;
            Ok((col, 1 + consumed))
        }
        // Handle potential character encoding values (not compression encodings)
        // These may appear if data was written by another system or corrupted
        _ => {
            // Check if this might be a character encoding (typically seen in string columns)
            // Character encodings like UTF-8, Latin-1, UTF-16 are not column compression,
            // so we treat the entire byte sequence as plain data
            if matches!(
                col_type,
                ColumnType::String | ColumnType::StringDict | ColumnType::Binary
            ) {
                // For string/binary types, treat unknown byte as raw data (PLAIN fallback)
                let (col, consumed) = ColumnData::from_bytes_typed(data_bytes, col_type)?;
                Ok((col, 1 + consumed))
            } else {
                // For other types, try as plain encoding (might be data corruption or legacy format)
                let (col, consumed) = ColumnData::from_bytes_typed(data_bytes, col_type)?;
                Ok((col, 1 + consumed))
            }
        }
    }
}

/// Skip over an encoded column's data without allocating.
/// Returns bytes consumed including the encoding prefix byte.
fn skip_column_encoded(bytes: &[u8], col_type: ColumnType) -> io::Result<usize> {
    if bytes.is_empty() {
        return Err(err_data("skip_column_encoded: empty input"));
    }
    let encoding = bytes[0];
    let data_bytes = &bytes[1..];
    match encoding {
        COL_ENCODING_PLAIN => {
            let consumed = ColumnData::skip_bytes_typed(data_bytes, col_type)?;
            Ok(1 + consumed)
        }
        COL_ENCODING_RLE => {
            // RLE format: [count:u64][num_runs:u64][(value:i64, run_len:u32)...]
            if data_bytes.len() < 16 {
                return Err(err_data("skip RLE: truncated header"));
            }
            let num_runs = u64::from_le_bytes(data_bytes[8..16].try_into().unwrap()) as usize;
            Ok(1 + 16 + num_runs * 12)
        }
        COL_ENCODING_BITPACK => {
            // BitPack format: [count:u64][bit_width:u8][min_value:i64][packed...]
            if data_bytes.len() < 17 {
                return Err(err_data("skip BitPack: truncated header"));
            }
            let count = u64::from_le_bytes(data_bytes[0..8].try_into().unwrap()) as usize;
            let bit_width = data_bytes[8] as usize;
            let packed_bytes = (count * bit_width + 7) / 8;
            Ok(1 + 17 + packed_bytes)
        }
        COL_ENCODING_RLE_BOOL => {
            // Bool RLE format: [count:u64][num_runs:u64][(value:u8, run_len:u32)...]
            if data_bytes.len() < 16 {
                return Err(err_data("skip RLE Bool: truncated header"));
            }
            let num_runs = u64::from_le_bytes(data_bytes[8..16].try_into().unwrap()) as usize;
            Ok(1 + 16 + num_runs * 5)
        }
        _ => {
            // For unknown encodings (shouldn't happen), fall back to plain skip
            let consumed = ColumnData::skip_bytes_typed(data_bytes, col_type)?;
            Ok(1 + consumed)
        }
    }
}

/// Decompress an RG body based on the compression flag.
/// Returns Ok(None) if uncompressed (caller should use raw bytes),
/// or Ok(Some(Vec<u8>)) with decompressed data.
fn decompress_rg_body(compress_flag: u8, compressed: &[u8]) -> io::Result<Option<Vec<u8>>> {
    match compress_flag {
        RG_COMPRESS_NONE => Ok(None),
        RG_COMPRESS_LZ4 => {
            let decompressed = lz4_flex::decompress_size_prepended(compressed)
                .map_err(|e| err_data(&format!("LZ4 decompress failed: {}", e)))?;
            Ok(Some(decompressed))
        }
        RG_COMPRESS_ZSTD => {
            let decompressed = zstd::bulk::decompress(compressed, 256 * 1024 * 1024)
                .map_err(|e| err_data(&format!("Zstd decompress failed: {}", e)))?;
            Ok(Some(decompressed))
        }
        _ => Err(err_data(&format!(
            "Unknown compression flag: {}",
            compress_flag
        ))),
    }
}

/// Compress an RG body using the specified compression algorithm.
/// Returns (compress_flag, compressed_or_original_bytes).
fn compress_rg_body(body: Vec<u8>, compression: CompressionType) -> (u8, Vec<u8>) {
    if body.len() < COMPRESS_MIN_BODY_SIZE || matches!(compression, CompressionType::None) {
        return (RG_COMPRESS_NONE, body);
    }
    match compression {
        CompressionType::None => (RG_COMPRESS_NONE, body),
        CompressionType::Zstd => {
            if let Ok(compressed) = zstd::bulk::compress(&body, 1) {
                if compressed.len() < body.len() {
                    return (RG_COMPRESS_ZSTD, compressed);
                }
            }
            (RG_COMPRESS_NONE, body)
        }
        CompressionType::Lz4 => {
            let compressed = lz4_flex::compress_prepend_size(&body);
            if compressed.len() < body.len() {
                return (RG_COMPRESS_LZ4, compressed);
            }
            (RG_COMPRESS_NONE, body)
        }
    }
}

// Column type identifiers
const TYPE_NULL: u8 = 0;
const TYPE_BOOL: u8 = 1;
const TYPE_INT8: u8 = 2;
const TYPE_INT16: u8 = 3;
const TYPE_INT32: u8 = 4;
const TYPE_INT64: u8 = 5;
const TYPE_UINT8: u8 = 6;
const TYPE_UINT16: u8 = 7;
const TYPE_UINT32: u8 = 8;
const TYPE_UINT64: u8 = 9;
const TYPE_FLOAT32: u8 = 10;
const TYPE_FLOAT64: u8 = 11;
const TYPE_STRING: u8 = 12;
const TYPE_BINARY: u8 = 13;
const TYPE_STRING_DICT: u8 = 14; // Dictionary-encoded string (DuckDB-style)
const TYPE_TIMESTAMP: u8 = 15; // Timestamp (microseconds since Unix epoch)
const TYPE_DATE: u8 = 16; // Date (days since Unix epoch)
const TYPE_FIXED_LIST: u8 = 17; // Fixed-size list of f32 (no offset array)
const TYPE_FLOAT16_LIST: u8 = 18; // Fixed-size list of f16 (half-precision, dim*2 bytes/row)

// ============================================================================
// Data Types
// ============================================================================

// Type definitions
include!("types.rs");
include!("header.rs");

// OnDemandStorage struct and implementation
include!("storage_core.rs");
include!("arrow_io.rs");
include!("mmap_scan.rs");
include!("read_write.rs");
include!("agg_wal.rs");

#[cfg(test)]
mod tests;
