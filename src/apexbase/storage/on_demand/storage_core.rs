// OnDemandStorage: struct definition, constructors, delta operations, compact

// ============================================================================
// On-Demand Storage Engine
// ============================================================================

const SYNC_PENDING_MAIN: u8 = 0b001;
const SYNC_PENDING_DELTA: u8 = 0b010;
const SYNC_PENDING_DELTASTORE: u8 = 0b100;

struct DeltaStringIndexCache {
    len: u64,
    modified: std::time::SystemTime,
    index: HashMap<String, HashMap<String, Vec<u64>>>,
}

static DELTA_STRING_INDEX_CACHE: once_cell::sync::Lazy<RwLock<HashMap<PathBuf, DeltaStringIndexCache>>> =
    once_cell::sync::Lazy::new(|| RwLock::new(HashMap::new()));
static DELTA_ROW_COUNT_CACHE: once_cell::sync::Lazy<
    RwLock<HashMap<PathBuf, (u64, std::time::SystemTime, usize)>>,
> = once_cell::sync::Lazy::new(|| RwLock::new(HashMap::new()));

/// High-performance on-demand columnar storage
///
/// Key features:
/// - Read only required columns (column projection)
/// - Read only required row ranges  
/// - Uses mmap for zero-copy reads with OS page cache (cross-platform)
/// - Soft delete with deleted bitmap
/// - Update via delete + insert
pub struct OnDemandStorage {
    path: PathBuf,
    file: RwLock<Option<File>>,
    write_file: RwLock<Option<File>>,
    delta_file: RwLock<Option<File>>,
    /// Memory-mapped file cache for fast repeated reads
    mmap_cache: RwLock<MmapCache>,
    header: RwLock<OnDemandHeader>,
    schema: RwLock<OnDemandSchema>,
    column_index: RwLock<Vec<ColumnIndexEntry>>,
    /// In-memory column data (legacy: used as write buffer for pending inserts)
    columns: RwLock<Vec<ColumnData>>,
    /// Row IDs (legacy: used as write buffer for pending inserts)
    ids: RwLock<Vec<u64>>,
    /// Next row ID
    next_id: AtomicU64,
    /// Null bitmaps per column (legacy: used as write buffer for pending inserts)
    nulls: RwLock<Vec<Vec<u8>>>,
    /// Deleted row bitmap (packed bits, 1 = deleted)
    deleted: RwLock<Vec<u8>>,
    /// ID to row index mapping for fast lookups (lazy-loaded)
    /// Only built when needed for delete/exists operations
    /// Uses AHashMap for faster hash computation on u64 keys
    id_to_idx: RwLock<Option<ahash::AHashMap<u64, usize>>>,
    /// Cached count of active (non-deleted) rows for O(1) COUNT(*)
    active_count: AtomicU64,
    /// Durability level for controlling fsync behavior
    durability: super::DurabilityLevel,
    /// WAL writer for safe/max durability modes (None for fast mode)
    wal_writer: RwLock<Option<super::incremental::WalWriter>>,
    /// WAL buffer for pending writes (used for recovery)
    wal_buffer: RwLock<Vec<super::incremental::WalRecord>>,
    /// Auto-flush threshold: number of pending rows (0 = disabled)
    auto_flush_rows: AtomicU64,
    /// Auto-flush threshold: estimated memory bytes (0 = disabled)
    auto_flush_bytes: AtomicU64,
    /// Count of rows inserted since last save (for auto-flush)
    pending_rows: AtomicU64,
    /// Total rows physically on disk (including deleted). Only updated after disk writes.
    /// Used by save() to distinguish in-memory-only rows from persisted rows.
    persisted_row_count: AtomicU64,
    /// Whether V4 base data was bulk-loaded into memory (only in tests via open_v4_data).
    /// Production code never sets this — in-memory data is always just the write buffer.
    v4_base_loaded: AtomicBool,
    /// Lock-free cache of header.footer_offset for V4 detection on the read path.
    /// Avoids acquiring header RwLock on every to_arrow_batch / read call.
    /// Updated atomically whenever header.footer_offset changes (save_v4, open, append_row_group).
    cached_footer_offset: AtomicU64,
    /// Cached V4 footer with Row Group metadata (lazy-loaded from disk).
    /// Enables on-demand mmap reads without loading all data into memory.
    v4_footer: RwLock<Option<V4Footer>>,
    /// Delta store for cell-level update tracking (Phase 4.5).
    /// Tracks pending UPDATE changes without rewriting the base file.
    /// On read, DeltaMerger overlays these changes on top of base data.
    delta_store: RwLock<DeltaStore>,
    /// Bitmask of files that were written directly to disk and still need fsync.
    sync_pending: AtomicU8,
    /// Row Group body compression algorithm. Default: None (no compression).
    /// Persisted in header flags bits 0-1. Can only be set on empty tables.
    compression: std::sync::atomic::AtomicU8,
    /// User-space page cache for retrieve_rcix point lookups.
    /// Caches 4KB file pages as heap memory to avoid mmap page-fault overhead on macOS.
    /// On-demand: only pages actually accessed are cached (~13 pages = ~52KB per backend).
    /// Invalidated after every write (save_v4).
    pub(crate) page_cache: RwLock<HashMap<u64, Box<[u8; 4096]>>>,
    /// Reusable scratch buffer for vector TopK scans.
    /// Pre-allocated on first use; grown as needed; reused to avoid per-query
    /// 512MB allocation + soft-page-fault overhead on the destination pages.
    /// Never shrinks — sized to the largest scan seen so far.
    pub(crate) scan_buf: std::sync::Mutex<Vec<f32>>,
    /// File size when scan_buf was last populated; 0 = cache invalid.
    /// Used to skip re-copying vector data when file hasn't changed.
    pub(crate) scan_buf_file_size: std::sync::atomic::AtomicU64,
    /// Column name whose data is currently in scan_buf (empty = none).
    pub(crate) scan_buf_col: std::sync::Mutex<String>,
    /// Raw f16 byte cache for Float16List TopK scans.
    /// Stores n_rows × dim × 2 raw LE f16 bytes; f32 decode happens per-row
    /// during distance computation — halves memory vs a decoded f32 scan_buf.
    pub(crate) scan_buf_f16: std::sync::Mutex<Vec<u8>>,
    /// File size when scan_buf_f16 was last populated; 0 = cache invalid.
    pub(crate) scan_buf_f16_file_size: std::sync::atomic::AtomicU64,
    /// Column name whose f16 data is currently in scan_buf_f16 (empty = none).
    pub(crate) scan_buf_f16_col: std::sync::Mutex<String>,
    /// Global lock for thread-safe concurrent access to file and mmap.
    /// This prevents "File not open" and "V4 footer: schema overflow" errors
    /// when multiple threads access the storage simultaneously.
    pub(crate) global_lock: parking_lot::RwLock<()>,
}

impl OnDemandStorage {
    /// Create a new storage file with default durability (Fast)
    pub fn create(path: &Path) -> io::Result<Self> {
        Self::create_with_durability(path, super::DurabilityLevel::Fast)
    }

    /// Create a new storage file with specified durability level
    pub fn create_with_durability(
        path: &Path,
        durability: super::DurabilityLevel,
    ) -> io::Result<Self> {
        Self::create_with_schema_and_durability(path, durability, &[])
    }

    /// Create a new storage file with pre-defined schema and durability level.
    /// Pre-defining schema avoids schema inference on the first insert, providing
    /// a performance benefit: columns and null vectors are pre-allocated with
    /// correct types so insert_typed() hits the fast path immediately.
    pub fn create_with_schema_and_durability(
        path: &Path,
        durability: super::DurabilityLevel,
        schema_cols: &[(String, ColumnType)],
    ) -> io::Result<Self> {
        let header = OnDemandHeader::new();
        let mut schema = OnDemandSchema::new();
        let mut columns = Vec::with_capacity(schema_cols.len());
        let mut nulls = Vec::with_capacity(schema_cols.len());

        // Pre-populate schema and empty column vectors
        for (name, dtype) in schema_cols {
            schema.add_column(name, *dtype);
            columns.push(ColumnData::new(*dtype));
            nulls.push(Vec::new());
        }

        // Initialize WAL for safe/max durability modes
        let wal_writer = if durability != super::DurabilityLevel::Fast {
            let wal_path = Self::wal_path(path);
            Some(super::incremental::WalWriter::create(
                &wal_path,
                crate::storage::FIRST_ROW_ID,
            )?)
        } else {
            None
        };

        let storage = Self {
            path: path.to_path_buf(),
            file: RwLock::new(None),
            write_file: RwLock::new(None),
            delta_file: RwLock::new(None),
            mmap_cache: RwLock::new(MmapCache::new()),
            header: RwLock::new(header),
            schema: RwLock::new(schema),
            column_index: RwLock::new(Vec::new()),
            columns: RwLock::new(columns),
            ids: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(crate::storage::FIRST_ROW_ID),
            nulls: RwLock::new(nulls),
            deleted: RwLock::new(Vec::new()),
            id_to_idx: RwLock::new(Some(ahash::AHashMap::new())),
            active_count: AtomicU64::new(0),
            durability,
            wal_writer: RwLock::new(wal_writer),
            wal_buffer: RwLock::new(Vec::new()),
            auto_flush_rows: AtomicU64::new(100000),
            auto_flush_bytes: AtomicU64::new(500 * 1024 * 1024),
            pending_rows: AtomicU64::new(0),
            persisted_row_count: AtomicU64::new(0),
            v4_base_loaded: AtomicBool::new(false),
            cached_footer_offset: AtomicU64::new(0),
            v4_footer: RwLock::new(None),
            delta_store: RwLock::new(DeltaStore::new(path)),
            sync_pending: AtomicU8::new(0),
            compression: std::sync::atomic::AtomicU8::new(CompressionType::None as u8),
            page_cache: RwLock::new(HashMap::new()),
            scan_buf: std::sync::Mutex::new(Vec::new()),
            scan_buf_file_size: std::sync::atomic::AtomicU64::new(0),
            scan_buf_col: std::sync::Mutex::new(String::new()),
            scan_buf_f16: std::sync::Mutex::new(Vec::new()),
            scan_buf_f16_file_size: std::sync::atomic::AtomicU64::new(0),
            scan_buf_f16_col: std::sync::Mutex::new(String::new()),
            global_lock: parking_lot::RwLock::new(()),
        };

        // Write initial file
        storage.save()?;

        Ok(storage)
    }

    /// Get WAL file path for a given data file path
    fn wal_path(main_path: &Path) -> PathBuf {
        let mut wal_path = main_path.to_path_buf();
        let ext = wal_path
            .extension()
            .map(|e| format!("{}.wal", e.to_string_lossy()))
            .unwrap_or_else(|| "wal".to_string());
        wal_path.set_extension(ext);
        wal_path
    }

    /// Open existing storage with default durability (Fast)
    pub fn open(path: &Path) -> io::Result<Self> {
        Self::open_with_durability(path, super::DurabilityLevel::Fast)
    }

    /// Open existing storage with specified durability level
    /// Uses mmap for fast zero-copy reads with OS page cache
    pub fn open_with_durability(
        path: &Path,
        durability: super::DurabilityLevel,
    ) -> io::Result<Self> {
        // Clean up stale .tmp files from crashed atomic writes
        let tmp_path = path.with_extension("apex.tmp");
        if tmp_path.exists() {
            let _ = std::fs::remove_file(&tmp_path);
        }
        // Clean up stale .deltastore.tmp from crashed DeltaStore save
        let ds_tmp = {
            let mut p = path.to_path_buf();
            let name = p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            p.set_file_name(format!("{}.deltastore.tmp", name));
            p
        };
        if ds_tmp.exists() {
            let _ = std::fs::remove_file(&ds_tmp);
        }
        // Apply any deferred delete state before reading the file
        let _ = apply_pending_deletes(path);

        let file = open_for_sequential_read(path)?;

        // Create mmap cache and use it for initial reads
        let mut mmap_cache = MmapCache::new();

        // Read header using mmap (zero-copy)
        let mut header_bytes = [0u8; HEADER_SIZE];
        mmap_cache.read_at(&file, &mut header_bytes, 0)?;
        let header = OnDemandHeader::from_bytes(&header_bytes)?;

        if header.footer_offset == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unsupported legacy file format (V3). Please re-create the table.",
            ));
        }

        let id_count = header.row_count as usize;

        // V4 Row Group format: read schema from footer
        let file_len = file.metadata()?.len();
        let footer_byte_count = (file_len - header.footer_offset) as usize;
        let mut footer_bytes = vec![0u8; footer_byte_count];
        mmap_cache.read_at(&file, &mut footer_bytes, header.footer_offset)?;
        let footer = V4Footer::from_bytes(&footer_bytes)?;
        let schema = footer.schema.clone();
        let column_index: Vec<ColumnIndexEntry> = Vec::new();
        // Use max_id from non-empty RG metadata (row_count may be < max _id after deletes)
        let next_id = footer
            .row_groups
            .iter()
            .filter(|rg| rg.row_count > 0)
            .map(|rg| rg.max_id)
            .max()
            .map(|m| m + 1)
            .unwrap_or(crate::storage::FIRST_ROW_ID);
        let cached_v4_footer: Option<V4Footer> = Some(footer);

        let columns: Vec<ColumnData> = schema
            .columns
            .iter()
            .map(|(_, col_type)| ColumnData::new(*col_type))
            .collect();
        let nulls = vec![Vec::new(); header.column_count as usize];
        let deleted_len = (id_count + 7) / 8;
        let deleted = vec![0u8; deleted_len];

        // Handle WAL recovery and initialization for safe/max durability
        let wal_path = Self::wal_path(path);
        let (wal_writer, wal_buffer, recovered_next_id) =
            if durability != super::DurabilityLevel::Fast {
                if wal_path.exists() {
                    // Replay WAL for crash recovery
                    let mut reader = super::incremental::WalReader::open(&wal_path)?;
                    let all_records = reader.read_all()?;

                    // P0-3: Collect committed txn_ids for recovery filtering
                    let committed_txns: std::collections::HashSet<u64> = all_records
                        .iter()
                        .filter_map(|r| match r {
                            super::incremental::WalRecord::TxnCommit { txn_id } => Some(*txn_id),
                            _ => None,
                        })
                        .collect();

                    // Filter: keep only auto-commit (txn_id=0) and committed txn DML records
                    // ALSO: idempotency guard — skip Insert/BatchInsert records whose IDs
                    // are already in the base file (id < next_id). This prevents duplicate
                    // rows if WAL is replayed after the base file was already saved.
                    let base_next_id = next_id; // next_id from base file before WAL recovery
                    let records: Vec<_> = all_records
                        .into_iter()
                        .filter(|r| {
                            match r {
                                super::incremental::WalRecord::Insert { txn_id, id, .. } => {
                                    (*txn_id == 0 || committed_txns.contains(txn_id))
                                        && *id >= base_next_id // Skip if already persisted
                                }
                                super::incremental::WalRecord::BatchInsert {
                                    txn_id,
                                    start_id,
                                    rows,
                                    ..
                                } => {
                                    let end_id = *start_id + rows.len() as u64;
                                    (*txn_id == 0 || committed_txns.contains(txn_id))
                                        && end_id > base_next_id // Keep if any rows are new
                                }
                                super::incremental::WalRecord::Delete { txn_id, id, .. } => {
                                    (*txn_id == 0 || committed_txns.contains(txn_id))
                                        && *id < base_next_id // Only delete rows that exist in base
                                }
                                _ => true, // Keep checkpoints, txn boundaries
                            }
                        })
                        .collect();

                    // Find max ID from WAL records (handles both Insert and BatchInsert)
                    let max_wal_id = records
                        .iter()
                        .filter_map(|r| match r {
                            super::incremental::WalRecord::Insert { id, .. } => Some(*id),
                            super::incremental::WalRecord::BatchInsert {
                                start_id, rows, ..
                            } => Some(*start_id + rows.len() as u64 - 1),
                            _ => None,
                        })
                        .max();

                    let recovered_id = max_wal_id.map(|id| id + 1).unwrap_or(next_id);

                    // Open for append
                    let writer = super::incremental::WalWriter::open(&wal_path)?;
                    (Some(writer), records, recovered_id)
                } else {
                    // Create new WAL
                    let writer = super::incremental::WalWriter::create(&wal_path, next_id)?;
                    (Some(writer), Vec::new(), next_id)
                }
            } else {
                (None, Vec::new(), next_id)
            };

        let delta_next_id = {
            let delta_path = Self::delta_path(path);
            if delta_path.exists() {
                Self::get_max_id_from_delta_fast(&delta_path)
                    .ok()
                    .map(|id| id.saturating_add(1))
                    .unwrap_or(next_id)
            } else {
                next_id
            }
        };
        let final_next_id = recovered_next_id.max(next_id).max(delta_next_id);
        let cached_fo = header.footer_offset;

        // Read compression type from header flags
        let comp_type = CompressionType::from_flags(header.flags);

        Ok(Self {
            path: path.to_path_buf(),
            file: RwLock::new(Some(file)),
            write_file: RwLock::new(None),
            delta_file: RwLock::new(None),
            mmap_cache: RwLock::new(mmap_cache),
            header: RwLock::new(header),
            schema: RwLock::new(schema),
            column_index: RwLock::new(column_index),
            columns: RwLock::new(columns),
            ids: RwLock::new(Vec::new()), // Empty - lazy loaded when needed
            next_id: AtomicU64::new(final_next_id),
            nulls: RwLock::new(nulls),
            deleted: RwLock::new(deleted),
            id_to_idx: RwLock::new(None), // Lazy loaded when needed
            active_count: AtomicU64::new(if let Some(ref f) = cached_v4_footer {
                // Footer already loaded: derive active count from per-RG metadata.
                // Allows DELETE to skip header pwrite while fresh backends still get
                // the correct count (footer.deletion_count is always kept in sync).
                f.row_groups
                    .iter()
                    .map(|rg| (rg.row_count as u64).saturating_sub(rg.deletion_count as u64))
                    .sum::<u64>()
            } else {
                id_count as u64
            }),
            durability,
            wal_writer: RwLock::new(wal_writer),
            wal_buffer: RwLock::new(wal_buffer),
            auto_flush_rows: AtomicU64::new(10000),
            auto_flush_bytes: AtomicU64::new(500 * 1024 * 1024),
            pending_rows: AtomicU64::new(0),
            persisted_row_count: AtomicU64::new(id_count as u64),
            v4_base_loaded: AtomicBool::new(false),
            cached_footer_offset: AtomicU64::new(cached_fo),
            v4_footer: RwLock::new(cached_v4_footer),
            delta_store: RwLock::new(
                DeltaStore::load(path).unwrap_or_else(|_| DeltaStore::new(path)),
            ),
            sync_pending: AtomicU8::new(0),
            compression: std::sync::atomic::AtomicU8::new(comp_type as u8),
            page_cache: RwLock::new(HashMap::new()),
            scan_buf: std::sync::Mutex::new(Vec::new()),
            scan_buf_file_size: std::sync::atomic::AtomicU64::new(0),
            scan_buf_col: std::sync::Mutex::new(String::new()),
            scan_buf_f16: std::sync::Mutex::new(Vec::new()),
            scan_buf_f16_file_size: std::sync::atomic::AtomicU64::new(0),
            scan_buf_f16_col: std::sync::Mutex::new(String::new()),
            global_lock: parking_lot::RwLock::new(()),
        })
    }

    /// Open for reading only, reusing a pre-opened File and known file_len.
    /// Skips DeltaStore::load (saves 1 stat syscall) and internal File::open (saves 1 open syscall).
    /// For pure read paths only — DeltaStore is initialized empty (no pending updates).
    pub fn open_for_read_with_file(path: &Path, file: File, file_len: u64) -> io::Result<Self> {
        // Apply pending delete state before creating the mmap so reads see fresh data
        let _ = apply_pending_deletes(path);
        let mut mmap_cache = MmapCache::new();

        let mut header_bytes = [0u8; HEADER_SIZE];
        mmap_cache.read_at(&file, &mut header_bytes, 0)?;
        let header = OnDemandHeader::from_bytes(&header_bytes)?;

        if header.footer_offset == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unsupported legacy file format (V3). Please re-create the table.",
            ));
        }

        let id_count = header.row_count as usize;
        let footer_byte_count = (file_len - header.footer_offset) as usize;
        let mut footer_bytes = vec![0u8; footer_byte_count];
        mmap_cache.read_at(&file, &mut footer_bytes, header.footer_offset)?;
        let footer = V4Footer::from_bytes(&footer_bytes)?;
        let schema = footer.schema.clone();
        let column_index: Vec<ColumnIndexEntry> = Vec::new();
        let next_id = footer
            .row_groups
            .iter()
            .filter(|rg| rg.row_count > 0)
            .map(|rg| rg.max_id)
            .max()
            .map(|m| m + 1)
            .unwrap_or(crate::storage::FIRST_ROW_ID);
        let cached_v4_footer: Option<V4Footer> = Some(footer);

        let columns: Vec<ColumnData> = schema
            .columns
            .iter()
            .map(|(_, col_type)| ColumnData::new(*col_type))
            .collect();
        let nulls = vec![Vec::new(); header.column_count as usize];
        let deleted_len = (id_count + 7) / 8;
        let deleted = vec![0u8; deleted_len];
        let cached_fo = header.footer_offset;
        let comp_type = CompressionType::from_flags(header.flags);

        Ok(Self {
            path: path.to_path_buf(),
            file: RwLock::new(Some(file)),
            write_file: RwLock::new(None),
            delta_file: RwLock::new(None),
            mmap_cache: RwLock::new(mmap_cache),
            header: RwLock::new(header),
            schema: RwLock::new(schema),
            column_index: RwLock::new(column_index),
            columns: RwLock::new(columns),
            ids: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(next_id),
            nulls: RwLock::new(nulls),
            deleted: RwLock::new(deleted),
            id_to_idx: RwLock::new(None),
            active_count: AtomicU64::new(if let Some(ref f) = cached_v4_footer {
                f.row_groups
                    .iter()
                    .map(|rg| (rg.row_count as u64).saturating_sub(rg.deletion_count as u64))
                    .sum::<u64>()
            } else {
                id_count as u64
            }),
            durability: super::DurabilityLevel::Fast,
            wal_writer: RwLock::new(None),
            wal_buffer: RwLock::new(Vec::new()),
            auto_flush_rows: AtomicU64::new(10000),
            auto_flush_bytes: AtomicU64::new(500 * 1024 * 1024),
            pending_rows: AtomicU64::new(0),
            persisted_row_count: AtomicU64::new(id_count as u64),
            v4_base_loaded: AtomicBool::new(false),
            cached_footer_offset: AtomicU64::new(cached_fo),
            v4_footer: RwLock::new(cached_v4_footer),
            delta_store: RwLock::new(
                DeltaStore::load(path).unwrap_or_else(|_| DeltaStore::new(path)),
            ),
            sync_pending: AtomicU8::new(0),
            compression: std::sync::atomic::AtomicU8::new(comp_type as u8),
            page_cache: RwLock::new(HashMap::new()),
            scan_buf: std::sync::Mutex::new(Vec::new()),
            scan_buf_file_size: std::sync::atomic::AtomicU64::new(0),
            scan_buf_col: std::sync::Mutex::new(String::new()),
            scan_buf_f16: std::sync::Mutex::new(Vec::new()),
            scan_buf_f16_file_size: std::sync::atomic::AtomicU64::new(0),
            scan_buf_f16_col: std::sync::Mutex::new(String::new()),
            global_lock: parking_lot::RwLock::new(()),
        })
    }

    /// Set auto-flush thresholds for automatic persistence
    /// * `rows` - Auto-flush when pending rows exceed this count (0 = disabled)
    /// * `bytes` - Auto-flush when estimated memory exceeds this size (0 = disabled)
    pub fn set_auto_flush(&self, rows: u64, bytes: u64) {
        self.auto_flush_rows.store(rows, Ordering::SeqCst);
        self.auto_flush_bytes.store(bytes, Ordering::SeqCst);
    }

    /// Get current auto-flush configuration
    pub fn get_auto_flush(&self) -> (u64, u64) {
        (
            self.auto_flush_rows.load(Ordering::SeqCst),
            self.auto_flush_bytes.load(Ordering::SeqCst),
        )
    }

    #[inline]
    pub fn mark_sync_pending(&self) {
        self.mark_main_sync_pending();
    }

    #[inline]
    pub fn mark_main_sync_pending(&self) {
        self.sync_pending
            .fetch_or(SYNC_PENDING_MAIN, Ordering::SeqCst);
    }

    #[inline]
    pub fn mark_delta_sync_pending(&self) {
        self.sync_pending
            .fetch_or(SYNC_PENDING_DELTA, Ordering::SeqCst);
    }

    #[inline]
    pub fn mark_deltastore_sync_pending(&self) {
        self.sync_pending
            .fetch_or(SYNC_PENDING_DELTASTORE, Ordering::SeqCst);
    }

    #[inline]
    pub fn sync_pending(&self) -> bool {
        self.sync_pending_bits() != 0
    }

    #[inline]
    pub fn footer_offset_hint(&self) -> u64 {
        self.cached_footer_offset.load(Ordering::Acquire)
    }

    #[inline]
    pub fn sync_pending_bits(&self) -> u8 {
        self.sync_pending.load(Ordering::SeqCst)
    }

    #[inline]
    pub fn main_sync_pending(&self) -> bool {
        self.sync_pending_bits() & SYNC_PENDING_MAIN != 0
    }

    #[inline]
    pub fn delta_sync_pending(&self) -> bool {
        self.sync_pending_bits() & SYNC_PENDING_DELTA != 0
    }

    #[inline]
    pub fn deltastore_sync_pending(&self) -> bool {
        self.sync_pending_bits() & SYNC_PENDING_DELTASTORE != 0
    }

    #[inline]
    pub fn clear_sync_pending(&self) {
        self.sync_pending.store(0, Ordering::SeqCst);
    }

    #[inline]
    pub fn clear_main_sync_pending(&self) {
        self.sync_pending
            .fetch_and(!SYNC_PENDING_MAIN, Ordering::SeqCst);
    }

    #[inline]
    pub fn clear_delta_sync_pending(&self) {
        self.sync_pending
            .fetch_and(!SYNC_PENDING_DELTA, Ordering::SeqCst);
    }

    #[inline]
    pub fn clear_deltastore_sync_pending(&self) {
        self.sync_pending
            .fetch_and(!SYNC_PENDING_DELTASTORE, Ordering::SeqCst);
    }

    /// Acquire global read lock for thread-safe concurrent reads.
    /// Returns a guard that releases the lock when dropped.
    /// Multiple readers can hold the lock simultaneously.
    #[inline]
    pub fn read_lock(&self) -> parking_lot::RwLockReadGuard<()> {
        self.global_lock.read()
    }

    /// Acquire global write lock for thread-safe writes.
    /// Returns a guard that releases the lock when dropped.
    /// Only one writer can hold the lock; readers are blocked while held.
    #[inline]
    pub fn write_lock(&self) -> parking_lot::RwLockWriteGuard<()> {
        self.global_lock.write()
    }

    /// Estimate current in-memory data size in bytes
    pub fn estimate_memory_bytes(&self) -> u64 {
        let columns = self.columns.read();
        let mut total: u64 = 0;

        for col in columns.iter() {
            total += col.estimate_memory_bytes() as u64;
        }

        // Add overhead for IDs (8 bytes each)
        total += self.ids.read().len() as u64 * 8;

        // Add overhead for null bitmaps
        for null_bitmap in self.nulls.read().iter() {
            total += null_bitmap.len() as u64;
        }

        // Add deleted bitmap
        total += self.deleted.read().len() as u64;

        total
    }

    /// Read bytes from the file using the user-space page cache.
    /// On cache miss, performs a positioned read (pread) and caches the 4KB page.
    /// On cache hit, copies bytes from the cached heap page — zero mmap page faults.
    /// This eliminates repeated soft page faults on macOS for point lookup paths.
    pub(crate) fn read_cached_bytes(&self, abs_offset: u64, dst: &mut [u8]) -> io::Result<()> {
        let len = dst.len();
        if len == 0 {
            return Ok(());
        }
        let mut written = 0usize;
        let mut cur_off = abs_offset;
        while written < len {
            let page_num = cur_off / 4096;
            let page_off = (cur_off % 4096) as usize;
            let to_copy = (len - written).min(4096 - page_off);
            // Fast path: page is in cache
            {
                let cache = self.page_cache.read();
                if let Some(page) = cache.get(&page_num) {
                    dst[written..written + to_copy]
                        .copy_from_slice(&page[page_off..page_off + to_copy]);
                    written += to_copy;
                    cur_off += to_copy as u64;
                    continue;
                }
            }
            // Cache miss: pread from file and cache the page
            let mut buf = [0u8; 4096];
            {
                let file_guard = self.file.read();
                let file = file_guard
                    .as_ref()
                    .ok_or_else(|| io::Error::new(io::ErrorKind::NotConnected, "file not open"))?;
                #[cfg(unix)]
                {
                    use std::os::unix::fs::FileExt;
                    let _ = file.read_at(&mut buf, page_num * 4096);
                }
                #[cfg(windows)]
                {
                    use std::os::windows::fs::FileExt;
                    let _ = file.seek_read(&mut buf, page_num * 4096);
                }
            }
            dst[written..written + to_copy].copy_from_slice(&buf[page_off..page_off + to_copy]);
            written += to_copy;
            cur_off += to_copy as u64;
            self.page_cache.write().insert(page_num, Box::new(buf));
        }
        Ok(())
    }

    /// Invalidate the user-space page cache and raw Arrow batch cache.
    /// Called after every write (save_v4, append_row_group, open_v4_data).
    pub(crate) fn invalidate_page_cache(&self) {
        self.page_cache.write().clear();
        self.scan_buf_file_size
            .store(0, std::sync::atomic::Ordering::Release);
        self.scan_buf_f16_file_size
            .store(0, std::sync::atomic::Ordering::Release);
    }

    /// Check if auto-flush is needed and perform it if so
    /// Returns true if auto-flush was performed
    fn maybe_auto_flush(&self) -> io::Result<bool> {
        let rows_threshold = self.auto_flush_rows.load(Ordering::SeqCst);
        let bytes_threshold = self.auto_flush_bytes.load(Ordering::SeqCst);

        // Check row threshold
        if rows_threshold > 0 {
            let pending = self.pending_rows.load(Ordering::SeqCst);
            if pending >= rows_threshold {
                self.save()?;
                self.pending_rows.store(0, Ordering::SeqCst);
                return Ok(true);
            }
        }

        // Check memory threshold
        if bytes_threshold > 0 {
            let mem_bytes = self.estimate_memory_bytes();
            if mem_bytes >= bytes_threshold {
                self.save()?;
                self.pending_rows.store(0, Ordering::SeqCst);
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get the current compression type.
    pub fn compression(&self) -> CompressionType {
        match self.compression.load(Ordering::Relaxed) {
            1 => CompressionType::Lz4,
            2 => CompressionType::Zstd,
            _ => CompressionType::None,
        }
    }

    /// Set compression type. Only effective on empty tables (row_count == 0).
    /// The setting is persisted in the header flags and survives restarts.
    /// Returns Ok(true) if applied, Ok(false) if table is non-empty (no-op).
    pub fn set_compression(&self, comp: CompressionType) -> io::Result<bool> {
        if self.active_count.load(Ordering::SeqCst) > 0
            || self.persisted_row_count.load(Ordering::SeqCst) > 0
        {
            return Ok(false);
        }
        self.compression.store(comp as u8, Ordering::SeqCst);
        // Persist to header flags
        {
            let mut header = self.header.write();
            header.flags = (header.flags & !FLAG_COMPRESS_MASK) | comp.to_flags_bits();
        }
        // Re-save header to disk
        self.save()?;
        Ok(true)
    }

    /// Helper: Get file reference or return NotConnected error
    /// Reduces boilerplate in read methods
    #[inline]
    fn get_file_ref(&self) -> io::Result<parking_lot::RwLockReadGuard<'_, Option<File>>> {
        let guard = self.file.read();
        if guard.is_none() {
            return Err(err_not_conn("File not open"));
        }
        Ok(guard)
    }

    /// Create or open storage with default durability (Fast)
    pub fn open_or_create(path: &Path) -> io::Result<Self> {
        Self::open_or_create_with_durability(path, super::DurabilityLevel::Fast)
    }

    /// Create or open storage with specified durability level
    pub fn open_or_create_with_durability(
        path: &Path,
        durability: super::DurabilityLevel,
    ) -> io::Result<Self> {
        if path.exists() {
            Self::open_with_durability(path, durability)
        } else {
            Self::create_with_durability(path, durability)
        }
    }

    /// Open for write with default durability (Fast)
    pub fn open_for_write(path: &Path) -> io::Result<Self> {
        Self::open_for_write_with_durability(path, super::DurabilityLevel::Fast)
    }

    /// Open for write with specified durability level
    /// IMPORTANT: For memory efficiency, column data is loaded lazily.
    /// - For INSERT: use open_for_insert() which only loads metadata
    /// - For UPDATE/DELETE: this function loads all column data
    pub fn open_for_write_with_durability(
        path: &Path,
        durability: super::DurabilityLevel,
    ) -> io::Result<Self> {
        if !path.exists() {
            return Self::create_with_durability(path, durability);
        }

        // Open the storage first
        let storage = Self::open_with_durability(path, durability)?;

        // If there are existing rows, load all column data into memory
        // This is required because save() rewrites the entire file from self.columns
        let row_count = storage.header.read().row_count as usize;
        if row_count > 0 {
            storage.load_all_columns_into_memory()?;
        } else {
            // Even with 0 rows, initialize empty columns based on schema
            // This is needed for INSERT after ALTER TABLE (columns defined but no data)
            let schema = storage.schema.read();
            let mut columns = storage.columns.write();
            let mut nulls = storage.nulls.write();

            // Always reinitialize columns with correct types from schema
            // The initial columns vector may have placeholder Int64 types
            if schema.column_count() > 0 {
                columns.clear();
                nulls.clear();
                for (_name, col_type) in schema.columns.iter() {
                    columns.push(ColumnData::new(*col_type));
                    nulls.push(Vec::new());
                }
            }
        }

        Ok(storage)
    }

    /// Open for INSERT operations only - memory efficient!
    /// Only loads metadata (header, schema, ids), NOT column data.
    /// New data is written to a delta file and merged on read or compact.
    pub fn open_for_insert(path: &Path) -> io::Result<Self> {
        Self::open_for_insert_with_durability(path, super::DurabilityLevel::Fast)
    }

    /// Open for INSERT with specified durability - memory efficient!
    pub fn open_for_insert_with_durability(
        path: &Path,
        durability: super::DurabilityLevel,
    ) -> io::Result<Self> {
        if !path.exists() {
            return Self::create_with_durability(path, durability);
        }

        // Just open without loading column data - metadata only
        Self::open_with_durability(path, durability)
    }

    /// Open for SCHEMA changes only - MOST memory efficient!
    /// Only loads header, schema, and column index. Does NOT load IDs or column data.
    /// Use for: ALTER TABLE ADD/DROP/RENAME COLUMN, TRUNCATE
    pub fn open_for_schema_change(path: &Path) -> io::Result<Self> {
        Self::open_for_schema_change_with_durability(path, super::DurabilityLevel::Fast)
    }

    /// Open for SCHEMA changes with specified durability.
    /// Delegates to open_with_durability (V4-only format).
    pub fn open_for_schema_change_with_durability(
        path: &Path,
        durability: super::DurabilityLevel,
    ) -> io::Result<Self> {
        if !path.exists() {
            return Self::create_with_durability(path, durability);
        }
        Self::open_with_durability(path, durability)
    }

    /// Get the delta file path for this storage
    fn delta_path(base_path: &Path) -> PathBuf {
        let mut delta = base_path.to_path_buf();
        let name = delta.file_name().unwrap_or_default().to_string_lossy();
        delta.set_file_name(format!("{}.delta", name));
        delta
    }

    fn delta_meta_path(delta_path: &Path) -> PathBuf {
        let mut meta = delta_path.to_path_buf();
        let name = meta.file_name().unwrap_or_default().to_string_lossy();
        meta.set_file_name(format!("{}.meta", name));
        meta
    }

    // ========================================================================
    // DeltaStore accessors (Phase 4.5)
    // ========================================================================

    /// Record a cell-level update in the delta store.
    /// Used by UPDATE to avoid delete+insert for single-cell changes.
    pub fn delta_update_cell(&self, row_id: u64, column_name: &str, new_value: crate::data::Value) {
        self.delta_store
            .write()
            .update_cell(row_id, column_name, new_value);
    }

    /// Record a full row update in the delta store.
    pub fn delta_update_row(&self, row_id: u64, values: &HashMap<String, crate::data::Value>) {
        self.delta_store.write().update_row(row_id, values);
    }

    fn row_active_for_delta_overlay(&self, row_id: u64) -> io::Result<bool> {
        if row_id >= self.next_id.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(false);
        }
        if self.delta_store.read().is_deleted(row_id) {
            return Ok(false);
        }

        match self.row_id_active_rcix(row_id)? {
            Some(true) => Ok(true),
            Some(false) => {
                if self.pending_v4_in_memory_rows() == 0 {
                    Ok(false)
                } else {
                    Ok(self.exists(row_id))
                }
            }
            None => Ok(self.exists(row_id)),
        }
    }

    /// Record a row deletion in the delta store without rewriting the base file.
    pub fn delta_delete_row(&self, row_id: u64) -> io::Result<bool> {
        if !self.row_active_for_delta_overlay(row_id)? {
            return Ok(false);
        }
        self.delta_store.write().delete_row(row_id);
        Ok(true)
    }

    /// Record a full-row replacement in the delta store for an existing row.
    pub fn delta_update_existing_row(
        &self,
        row_id: u64,
        values: &HashMap<String, crate::data::Value>,
    ) -> io::Result<bool> {
        if !self.row_active_for_delta_overlay(row_id)? {
            return Ok(false);
        }
        self.delta_store.write().update_row(row_id, values);
        Ok(true)
    }

    /// Batch update multiple rows in a single lock acquisition.
    /// `batch` is a slice of (row_id, col_name, new_value) triples.
    pub fn delta_batch_update_rows(&self, batch: &[(u64, &str, crate::data::Value)]) {
        if !batch.is_empty() {
            self.delta_store.write().batch_update_rows(batch);
        }
    }

    /// Scan a numeric column for rows in [low, high] and return their row IDs directly.
    /// Returns None if not applicable (column not found, etc.).
    pub fn scan_numeric_range_with_ids(
        &self,
        col_name: &str,
        low: f64,
        high: f64,
    ) -> io::Result<Option<Vec<u64>>> {
        self.scan_numeric_range_mmap_with_ids(col_name, low, high)
    }

    /// Check if the delta store has any pending changes.
    pub fn has_pending_deltas(&self) -> bool {
        !self.delta_store.read().is_empty()
    }

    /// Get the number of pending delta updates.
    pub fn delta_update_count(&self) -> usize {
        self.delta_store.read().update_count()
    }

    /// Get the number of pending delta deletes.
    pub fn delta_delete_count(&self) -> usize {
        self.delta_store.read().delete_count()
    }

    /// Check whether pending DeltaStore updates modify a specific column.
    pub fn delta_updates_column(&self, column_name: &str) -> bool {
        self.delta_store.read().updates_column(column_name)
    }

    /// Return row IDs whose pending DeltaStore update sets `column_name` to `value`.
    pub fn delta_rows_with_string_update(&self, column_name: &str, value: &str) -> Vec<u64> {
        self.delta_store
            .read()
            .rows_with_string_update(column_name, value)
    }

    /// Save the delta store to disk (called during save path).
    pub fn save_delta_store(&self) -> io::Result<()> {
        let mut delta_store = self.delta_store.write();
        let was_dirty = delta_store.is_dirty();
        delta_store.save()?;
        drop(delta_store);

        if was_dirty {
            if self.durability == super::DurabilityLevel::Max {
                self.clear_deltastore_sync_pending();
            } else {
                self.mark_deltastore_sync_pending();
            }
        }

        Ok(())
    }

    /// Clear the delta store (called after compaction merges deltas into base).
    pub fn clear_delta_store(&self) -> io::Result<()> {
        let mut ds = self.delta_store.write();
        ds.clear();
        ds.save()?;
        ds.remove_file()?;
        self.clear_deltastore_sync_pending();
        Ok(())
    }

    /// Get a read reference to the delta store (for DeltaMerger on read path).
    pub fn delta_store(&self) -> parking_lot::RwLockReadGuard<'_, DeltaStore> {
        self.delta_store.read()
    }

    /// Check if delta compaction is needed based on update/delete count vs base rows.
    pub fn needs_delta_compaction(&self) -> bool {
        let ds = self.delta_store.read();
        let base_rows = self.active_count.load(std::sync::atomic::Ordering::Relaxed);
        ds.needs_compaction(base_rows)
    }

    /// Compact deltas into the base file: load base data, apply updates in-place,
    /// then do a full save_v4 rewrite which clears the delta store.
    pub fn compact_deltas(&self) -> io::Result<()> {
        let ds = self.delta_store.read();
        if ds.is_empty() {
            return Ok(());
        }

        // Collect updates and deletes before releasing the lock
        let all_updates = ds.all_updates().clone();
        let delete_bitmap = ds.delete_bitmap().clone();
        drop(ds);

        // Skip compaction if V4 data isn't in memory — deltas stay in DeltaStore
        // and are applied at read time via DeltaMerger overlay.
        if self.is_v4_format() && !self.has_v4_in_memory_data() {
            return Ok(());
        }

        // Apply deletes: mark deleted rows in the deleted bitmap
        {
            let ids = self.ids.read();
            let mut deleted = self.deleted.write();
            for (idx, id) in ids.iter().enumerate() {
                if delete_bitmap.is_deleted(*id) {
                    let byte_idx = idx / 8;
                    let bit_idx = idx % 8;
                    if byte_idx < deleted.len() {
                        deleted[byte_idx] |= 1 << bit_idx;
                    }
                }
            }
        }

        // Apply cell-level updates to in-memory columns
        {
            let ids = self.ids.read();
            let schema = self.schema.read();
            let mut columns = self.columns.write();

            // Build id→index map for fast lookup
            let id_to_idx: std::collections::HashMap<u64, usize> =
                ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

            for (row_id, col_updates) in &all_updates {
                if let Some(&row_idx) = id_to_idx.get(row_id) {
                    for (col_name, record) in col_updates {
                        if let Some(col_idx) = schema.get_index(col_name) {
                            if col_idx < columns.len() {
                                match &record.new_value {
                                    crate::data::Value::Int64(v) => {
                                        if let ColumnData::Int64(ref mut data) = columns[col_idx] {
                                            if row_idx < data.len() {
                                                data[row_idx] = *v;
                                            }
                                        }
                                    }
                                    crate::data::Value::Float64(v) => {
                                        if let ColumnData::Float64(ref mut data) = columns[col_idx]
                                        {
                                            if row_idx < data.len() {
                                                data[row_idx] = *v;
                                            }
                                        }
                                    }
                                    crate::data::Value::String(s) => {
                                        if let ColumnData::String { offsets, data } =
                                            &mut columns[col_idx]
                                        {
                                            // For strings, we need to rebuild — update in-place is complex
                                            // For compaction (rare), this is acceptable
                                            let mut strings: Vec<String> =
                                                Vec::with_capacity(offsets.len().saturating_sub(1));
                                            for i in 0..offsets.len().saturating_sub(1) {
                                                let start = offsets[i] as usize;
                                                let end = offsets[i + 1] as usize;
                                                if i == row_idx {
                                                    strings.push(s.clone());
                                                } else {
                                                    strings.push(
                                                        String::from_utf8_lossy(&data[start..end])
                                                            .to_string(),
                                                    );
                                                }
                                            }
                                            // Rebuild
                                            data.clear();
                                            offsets.clear();
                                            offsets.push(0);
                                            for st in &strings {
                                                data.extend_from_slice(st.as_bytes());
                                                offsets.push(data.len() as u32);
                                            }
                                        }
                                    }
                                    crate::data::Value::Bool(v) => {
                                        if let ColumnData::Bool { data, .. } = &mut columns[col_idx]
                                        {
                                            let byte_idx = row_idx / 8;
                                            let bit_idx = row_idx % 8;
                                            if byte_idx < data.len() {
                                                if *v {
                                                    data[byte_idx] |= 1 << bit_idx;
                                                } else {
                                                    data[byte_idx] &= !(1 << bit_idx);
                                                }
                                            }
                                        }
                                    }
                                    _ => {} // UInt64, Null, etc. — skip for now
                                }
                            }
                        }
                    }
                }
            }
        }

        // Full rewrite, then clear delta store (updates are now baked into base file)
        self.save_v4()?;
        self.clear_delta_store()
    }

    /// Apply any pending delta store updates/deletes to already-loaded in-memory columns.
    /// Must be called AFTER load_all_columns_into_memory() so self.ids/columns/deleted are populated.
    /// This ensures save_v4() always writes the correct (post-update) values and can safely
    /// clear the delta store afterwards.
    fn apply_pending_deltas_in_place(&self) {
        let ds = self.delta_store.read();
        if ds.is_empty() {
            return;
        }
        let all_updates = ds.all_updates().clone();
        let delete_bitmap = ds.delete_bitmap().clone();
        drop(ds);

        if !delete_bitmap.is_empty() {
            let ids = self.ids.read();
            let mut deleted = self.deleted.write();
            for (idx, id) in ids.iter().enumerate() {
                if delete_bitmap.is_deleted(*id) {
                    let byte_idx = idx / 8;
                    let bit_idx = idx % 8;
                    if byte_idx >= deleted.len() {
                        deleted.resize(byte_idx + 1, 0);
                    }
                    deleted[byte_idx] |= 1 << bit_idx;
                }
            }
        }

        if !all_updates.is_empty() {
            let ids = self.ids.read();
            let schema = self.schema.read();
            let mut columns = self.columns.write();

            let id_to_idx: ahash::AHashMap<u64, usize> =
                ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

            for (row_id, col_updates) in &all_updates {
                if let Some(&row_idx) = id_to_idx.get(row_id) {
                    for (col_name, record) in col_updates {
                        if let Some(col_idx) = schema.get_index(col_name) {
                            if col_idx < columns.len() {
                                match &record.new_value {
                                    crate::data::Value::Int64(v) => {
                                        if let ColumnData::Int64(ref mut data) = columns[col_idx] {
                                            if row_idx < data.len() {
                                                data[row_idx] = *v;
                                            }
                                        }
                                    }
                                    crate::data::Value::Float64(v) => {
                                        if let ColumnData::Float64(ref mut data) = columns[col_idx]
                                        {
                                            if row_idx < data.len() {
                                                data[row_idx] = *v;
                                            }
                                        }
                                    }
                                    crate::data::Value::String(s) => {
                                        if let ColumnData::String { offsets, data } =
                                            &mut columns[col_idx]
                                        {
                                            let mut strings: Vec<String> =
                                                Vec::with_capacity(offsets.len().saturating_sub(1));
                                            for i in 0..offsets.len().saturating_sub(1) {
                                                let start = offsets[i] as usize;
                                                let end = offsets[i + 1] as usize;
                                                if i == row_idx {
                                                    strings.push(s.clone());
                                                } else {
                                                    strings.push(
                                                        String::from_utf8_lossy(&data[start..end])
                                                            .to_string(),
                                                    );
                                                }
                                            }
                                            data.clear();
                                            offsets.clear();
                                            offsets.push(0);
                                            for st in &strings {
                                                data.extend_from_slice(st.as_bytes());
                                                offsets.push(data.len() as u32);
                                            }
                                        }
                                    }
                                    crate::data::Value::Bool(v) => {
                                        if let ColumnData::Bool { data, .. } = &mut columns[col_idx]
                                        {
                                            let byte_idx = row_idx / 8;
                                            let bit_idx = row_idx % 8;
                                            if byte_idx < data.len() {
                                                if *v {
                                                    data[byte_idx] |= 1 << bit_idx;
                                                } else {
                                                    data[byte_idx] &= !(1 << bit_idx);
                                                }
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Get the maximum ID from a delta file (for computing next_id on open)
    fn get_max_id_from_delta(delta_path: &Path) -> io::Result<u64> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = File::open(delta_path)?;
        let mut max_id: u64 = 0;

        loop {
            // Read record count
            let mut count_buf = [0u8; 8];
            match file.read_exact(&mut count_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            let record_count = u64::from_le_bytes(count_buf) as usize;

            // Read IDs and track max
            for _ in 0..record_count {
                let mut id_buf = [0u8; 8];
                file.read_exact(&mut id_buf)?;
                let id = u64::from_le_bytes(id_buf);
                max_id = max_id.max(id);
            }

            // Skip rest of record (int columns)
            let mut count_buf4 = [0u8; 4];
            file.read_exact(&mut count_buf4)?;
            let int_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..int_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                file.seek(SeekFrom::Current(name_len as i64))?;
                file.seek(SeekFrom::Current((record_count * 8) as i64))?;
            }

            // Skip float columns
            file.read_exact(&mut count_buf4)?;
            let float_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..float_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                file.seek(SeekFrom::Current(name_len as i64))?;
                file.seek(SeekFrom::Current((record_count * 8) as i64))?;
            }

            // Skip string columns (variable length - need to read lengths)
            file.read_exact(&mut count_buf4)?;
            let string_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..string_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                file.seek(SeekFrom::Current(name_len as i64))?;
                for _ in 0..record_count {
                    let mut str_len_buf = [0u8; 4];
                    file.read_exact(&mut str_len_buf)?;
                    let str_len = u32::from_le_bytes(str_len_buf) as usize;
                    file.seek(SeekFrom::Current(str_len as i64))?;
                }
            }

            // Skip bool columns
            file.read_exact(&mut count_buf4)?;
            let bool_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..bool_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                file.seek(SeekFrom::Current(name_len as i64))?;
                let skip_bytes = (record_count + 7) / 8;
                file.seek(SeekFrom::Current(skip_bytes as i64))?;
            }

            // Skip binary columns (variable length)
            file.read_exact(&mut count_buf4)?;
            let binary_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..binary_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                file.seek(SeekFrom::Current(name_len as i64))?;
                for _ in 0..record_count {
                    let mut bin_len_buf = [0u8; 4];
                    file.read_exact(&mut bin_len_buf)?;
                    let bin_len = u32::from_le_bytes(bin_len_buf) as usize;
                    file.seek(SeekFrom::Current(bin_len as i64))?;
                }
            }
        }

        Ok(max_id)
    }

    fn get_max_id_from_delta_fast(delta_path: &Path) -> io::Result<u64> {
        let meta_path = Self::delta_meta_path(delta_path);
        if let Ok(bytes) = std::fs::read(&meta_path) {
            if bytes.len() == 8 {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&bytes);
                return Ok(u64::from_le_bytes(buf));
            }
        }

        let max_id = Self::get_max_id_from_delta(delta_path)?;
        let _ = Self::write_delta_max_id(delta_path, max_id);
        Ok(max_id)
    }

    fn write_delta_max_id(delta_path: &Path, max_id: u64) -> io::Result<()> {
        std::fs::write(Self::delta_meta_path(delta_path), max_id.to_le_bytes())
    }

    /// Check if delta file exists
    pub fn has_delta(&self) -> bool {
        Self::delta_path(&self.path).exists()
    }

    /// Load all column data from disk into memory
    /// This is needed before write operations to preserve existing data
    fn load_all_columns_into_memory(&self) -> io::Result<()> {
        let header = self.header.read();
        let total_rows = header.row_count as usize;

        if total_rows == 0 {
            return Ok(());
        }

        // V4 files: load all RG data into memory for write operations
        if header.footer_offset > 0 {
            drop(header);
            self.open_v4_data()?;
            // Apply any pending delta store updates so save_v4() bakes them in correctly.
            // Without this, a subsequent save() would write pre-update values to disk and
            // then clear the delta store, permanently losing the updates.
            self.apply_pending_deltas_in_place();
            return Ok(());
        }

        let schema = self.schema.read();
        let column_index = self.column_index.read();

        // CRITICAL: Load IDs first since they're lazy-loaded
        // Without this, insert operations will think there are 0 existing rows
        drop(header);
        drop(schema);
        drop(column_index);
        self.ensure_ids_loaded()?;
        let header = self.header.read();
        let schema = self.schema.read();
        let column_index = self.column_index.read();

        let file_guard = self.file.read();
        let file = file_guard
            .as_ref()
            .ok_or_else(|| err_not_conn("File not open"))?;

        let mut mmap_cache = self.mmap_cache.write();
        let mut columns = self.columns.write();
        let mut nulls = self.nulls.write();

        let column_index_len = column_index.len();

        // Load each column from disk
        for col_idx in 0..schema.column_count() {
            let (_, col_type) = &schema.columns[col_idx];

            // Handle columns added via ALTER TABLE that don't have disk data yet
            if col_idx >= column_index_len {
                // Column exists in schema but not on disk - create padded column
                let mut col_data = ColumnData::new(*col_type);
                // Pad with defaults for existing rows
                for _ in 0..total_rows {
                    match &mut col_data {
                        ColumnData::Int64(v) => v.push(0),
                        ColumnData::Float64(v) => v.push(0.0),
                        ColumnData::String { offsets, .. } => {
                            offsets.push(*offsets.last().unwrap_or(&0))
                        }
                        ColumnData::Binary { offsets, .. } => {
                            offsets.push(*offsets.last().unwrap_or(&0))
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

                if col_idx < columns.len() {
                    columns[col_idx] = col_data;
                } else {
                    columns.push(col_data);
                }

                // Empty null bitmap for new columns
                if col_idx < nulls.len() {
                    nulls[col_idx] = Vec::new();
                } else {
                    nulls.push(Vec::new());
                }
                continue;
            }

            let index_entry = &column_index[col_idx];

            // Read column data
            let col_data = self.read_column_range_mmap(
                &mut mmap_cache,
                file,
                index_entry,
                *col_type,
                0,
                total_rows,
                total_rows,
            )?;

            // Store in columns array
            if col_idx < columns.len() {
                columns[col_idx] = col_data;
            } else {
                columns.push(col_data);
            }

            // Read null bitmap for this column
            let null_len = index_entry.null_length as usize;
            if null_len > 0 {
                let mut null_bitmap = vec![0u8; null_len];
                mmap_cache.read_at(file, &mut null_bitmap, index_entry.null_offset)?;
                if col_idx < nulls.len() {
                    nulls[col_idx] = null_bitmap;
                } else {
                    nulls.push(null_bitmap);
                }
            }
        }

        Ok(())
    }

    fn append_typed_to_delta_with_ids(
        &self,
        ids: &[u64],
        int_columns: &HashMap<String, Vec<i64>>,
        float_columns: &HashMap<String, Vec<f64>>,
        string_columns: &HashMap<String, Vec<String>>,
        bool_columns: &HashMap<String, Vec<bool>>,
    ) -> io::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }

        let delta_path = Self::delta_path(&self.path);
        let mut delta_file = self.delta_file.write();
        if delta_file.is_none() {
            if let Some(parent) = delta_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .read(true)
                .open(&delta_path)?;
            *delta_file = Some(file);
        }
        let file = delta_file
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "delta file not open"))?;

        // Delta spill is the hot path for explicit flush() on tiny OLTP bursts.
        // Buffer small per-field writes so a 1-row durable flush is not dominated
        // by dozens of tiny append syscalls.
        let mut writer = std::io::BufWriter::with_capacity(64 * 1024, &mut *file);

        writer.write_all(&(ids.len() as u64).to_le_bytes())?;
        for id in ids {
            writer.write_all(&id.to_le_bytes())?;
        }

        let int_col_count = int_columns.len() as u32;
        writer.write_all(&int_col_count.to_le_bytes())?;
        for (name, values) in int_columns {
            let name_bytes = name.as_bytes();
            writer.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
            writer.write_all(name_bytes)?;
            for v in values {
                writer.write_all(&v.to_le_bytes())?;
            }
        }

        let float_col_count = float_columns.len() as u32;
        writer.write_all(&float_col_count.to_le_bytes())?;
        for (name, values) in float_columns {
            let name_bytes = name.as_bytes();
            writer.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
            writer.write_all(name_bytes)?;
            for v in values {
                writer.write_all(&v.to_le_bytes())?;
            }
        }

        let string_col_count = string_columns.len() as u32;
        writer.write_all(&string_col_count.to_le_bytes())?;
        for (name, values) in string_columns {
            let name_bytes = name.as_bytes();
            writer.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
            writer.write_all(name_bytes)?;
            for v in values {
                let v_bytes = v.as_bytes();
                writer.write_all(&(v_bytes.len() as u32).to_le_bytes())?;
                writer.write_all(v_bytes)?;
            }
        }

        let bool_col_count = bool_columns.len() as u32;
        writer.write_all(&bool_col_count.to_le_bytes())?;
        for (name, values) in bool_columns {
            let name_bytes = name.as_bytes();
            writer.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
            writer.write_all(name_bytes)?;
            for v in values {
                writer.write_all(&[if *v { 1u8 } else { 0u8 }])?;
            }
        }

        writer.flush()?;
        drop(writer);
        if self.durability == super::DurabilityLevel::Max {
            file.sync_all()?;
            self.clear_delta_sync_pending();
        } else {
            self.mark_delta_sync_pending();
        }
        if let Some(max_id) = ids.iter().copied().max() {
            let _ = Self::write_delta_max_id(&delta_path, max_id);
        }

        Ok(())
    }

    /// Insert rows to delta file (memory efficient - doesn't load existing data)
    /// Returns the IDs assigned to the inserted rows
    pub fn insert_rows_to_delta(
        &self,
        rows: &[HashMap<String, ColumnValue>],
    ) -> io::Result<Vec<u64>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        let delta_path = Self::delta_path(&self.path);
        let delta_before = std::fs::metadata(&delta_path).ok().map(|metadata| {
            (
                metadata.len(),
                metadata
                    .modified()
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
            )
        });

        // Get schema to handle partial columns correctly
        let schema = self.schema.read();

        // Build column data from rows - ensure all columns have same length
        let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
        let mut binary_columns: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();

        // Initialize column vectors based on schema
        for (col_name, col_type) in &schema.columns {
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
                    int_columns.insert(col_name.clone(), Vec::with_capacity(rows.len()));
                }
                ColumnType::Float64 | ColumnType::Float32 => {
                    float_columns.insert(col_name.clone(), Vec::with_capacity(rows.len()));
                }
                ColumnType::String | ColumnType::StringDict => {
                    string_columns.insert(col_name.clone(), Vec::with_capacity(rows.len()));
                }
                ColumnType::Binary => {
                    binary_columns.insert(col_name.clone(), Vec::with_capacity(rows.len()));
                }
                ColumnType::FixedList | ColumnType::Float16List => {
                    binary_columns.insert(col_name.clone(), Vec::with_capacity(rows.len()));
                }
                ColumnType::Bool => {
                    bool_columns.insert(col_name.clone(), Vec::with_capacity(rows.len()));
                }
                ColumnType::Null => {
                    // Null columns are handled as strings with empty default
                    string_columns.insert(col_name.clone(), Vec::with_capacity(rows.len()));
                }
            }
        }

        // For each row, add values for ALL schema columns (default for missing)
        for row in rows {
            for (col_name, col_type) in &schema.columns {
                let val = row.get(col_name);
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
                        let v = val
                            .and_then(|v| {
                                if let ColumnValue::Int64(n) = v {
                                    Some(*n)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(0);
                        int_columns.get_mut(col_name).unwrap().push(v);
                    }
                    ColumnType::Float64 | ColumnType::Float32 => {
                        let v = val
                            .and_then(|v| {
                                if let ColumnValue::Float64(n) = v {
                                    Some(*n)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(0.0);
                        float_columns.get_mut(col_name).unwrap().push(v);
                    }
                    ColumnType::String | ColumnType::StringDict | ColumnType::Null => {
                        let v = val
                            .and_then(|v| {
                                if let ColumnValue::String(s) = v {
                                    Some(s.clone())
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_default();
                        string_columns.get_mut(col_name).unwrap().push(v);
                    }
                    ColumnType::Binary => {
                        let v = val
                            .and_then(|v| {
                                if let ColumnValue::Binary(b) = v {
                                    Some(b.clone())
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_default();
                        binary_columns.get_mut(col_name).unwrap().push(v);
                    }
                    ColumnType::FixedList | ColumnType::Float16List => {
                        let v = val
                            .and_then(|v| match v {
                                ColumnValue::FixedList(b) | ColumnValue::Binary(b) => {
                                    Some(b.clone())
                                }
                                _ => None,
                            })
                            .unwrap_or_default();
                        binary_columns.get_mut(col_name).unwrap().push(v);
                    }
                    ColumnType::Bool => {
                        let v = val
                            .and_then(|v| {
                                if let ColumnValue::Bool(b) = v {
                                    Some(*b)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(false);
                        bool_columns.get_mut(col_name).unwrap().push(v);
                    }
                }
            }
        }

        drop(schema);

        // Allocate IDs
        let mut ids = Vec::with_capacity(rows.len());
        for _ in 0..rows.len() {
            ids.push(self.next_id.fetch_add(1, Ordering::SeqCst));
        }

        self.append_typed_to_delta_with_ids(
            &ids,
            &int_columns,
            &float_columns,
            &string_columns,
            &bool_columns,
        )?;
        Self::refresh_delta_insert_caches(&delta_path, delta_before, &ids, &string_columns);
        Ok(ids)
    }

    fn refresh_delta_insert_caches(
        delta_path: &Path,
        before: Option<(u64, std::time::SystemTime)>,
        ids: &[u64],
        string_columns: &HashMap<String, Vec<String>>,
    ) {
        if ids.is_empty() {
            return;
        }

        let Ok(metadata) = std::fs::metadata(delta_path) else {
            return;
        };
        let file_len = metadata.len();
        let modified = metadata
            .modified()
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

        {
            let mut cache = DELTA_ROW_COUNT_CACHE.write();
            if cache.len() > 128 {
                cache.clear();
            }
            match before {
                None => {
                    cache.insert(delta_path.to_path_buf(), (file_len, modified, ids.len()));
                }
                Some((before_len, before_modified)) => {
                    if let Some(entry) = cache.get_mut(delta_path) {
                        if entry.0 == before_len && entry.1 >= before_modified {
                            entry.0 = file_len;
                            entry.1 = modified;
                            entry.2 += ids.len();
                        }
                    }
                }
            }
        }

        let append_strings =
            |index: &mut HashMap<String, HashMap<String, Vec<u64>>>| {
                for (column, values) in string_columns {
                    let value_index = index.entry(column.clone()).or_default();
                    for (row_idx, id) in ids.iter().copied().enumerate() {
                        if let Some(value) = values.get(row_idx) {
                            value_index.entry(value.clone()).or_default().push(id);
                        }
                    }
                }
            };

        let mut cache = DELTA_STRING_INDEX_CACHE.write();
        if cache.len() > 128 {
            cache.clear();
        }
        match before {
            None => {
                let mut index = HashMap::new();
                append_strings(&mut index);
                cache.insert(
                    delta_path.to_path_buf(),
                    DeltaStringIndexCache {
                        len: file_len,
                        modified,
                        index,
                    },
                );
            }
            Some((before_len, before_modified)) => {
                if let Some(entry) = cache.get_mut(delta_path) {
                    if entry.len == before_len && entry.modified >= before_modified {
                        append_strings(&mut entry.index);
                        entry.len = file_len;
                        entry.modified = modified;
                    }
                }
            }
        }
    }

    /// Insert typed columns to delta file (memory efficient - doesn't load existing data)
    /// Returns the IDs assigned to the inserted rows
    fn insert_typed_to_delta(
        &self,
        int_columns: HashMap<String, Vec<i64>>,
        float_columns: HashMap<String, Vec<f64>>,
        string_columns: HashMap<String, Vec<String>>,
        _binary_columns: HashMap<String, Vec<Vec<u8>>>, // Not yet implemented in delta
        bool_columns: HashMap<String, Vec<bool>>,
    ) -> io::Result<Vec<u64>> {
        // Determine row count
        let row_count = int_columns
            .values()
            .map(|v| v.len())
            .max()
            .unwrap_or(0)
            .max(float_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(string_columns.values().map(|v| v.len()).max().unwrap_or(0))
            .max(bool_columns.values().map(|v| v.len()).max().unwrap_or(0));

        if row_count == 0 {
            return Ok(Vec::new());
        }

        let delta_path = Self::delta_path(&self.path);

        // Allocate IDs
        let mut ids = Vec::with_capacity(row_count);
        for _ in 0..row_count {
            ids.push(self.next_id.fetch_add(1, Ordering::SeqCst));
        }

        self.append_typed_to_delta_with_ids(
            &ids,
            &int_columns,
            &float_columns,
            &string_columns,
            &bool_columns,
        )?;
        Ok(ids)
    }

    fn discard_pending_v4_rows_from(&self, pending_start: usize) {
        let truncate_bitmap = |bitmap: &mut Vec<u8>, row_count: usize| {
            let new_len = (row_count + 7) / 8;
            bitmap.truncate(new_len);
            if row_count == 0 {
                bitmap.clear();
            } else if row_count % 8 != 0 {
                if let Some(last) = bitmap.last_mut() {
                    *last &= (1u8 << (row_count % 8)) - 1;
                }
            }
        };

        if pending_start == 0 {
            self.ids.write().clear();
            self.columns.write().clear();
            self.nulls.write().clear();
            self.deleted.write().clear();
        } else {
            self.ids.write().truncate(pending_start);
            {
                let mut columns = self.columns.write();
                for column in columns.iter_mut() {
                    *column = column.slice_range(0, pending_start);
                }
            }
            {
                let mut nulls = self.nulls.write();
                for bitmap in nulls.iter_mut() {
                    truncate_bitmap(bitmap, pending_start);
                }
            }
            {
                let mut deleted = self.deleted.write();
                truncate_bitmap(&mut deleted, pending_start);
            }
        }
        *self.id_to_idx.write() = None;
        self.pending_rows.store(0, Ordering::SeqCst);
    }

    /// Spill mmap-only V4 memtable rows to the delta sidecar instead of
    /// rewriting the base file. This keeps explicit `flush()` on small OLTP
    /// bursts fast while preserving cross-process visibility through the
    /// existing delta merge path.
    pub fn spill_pending_v4_rows_to_delta(&self) -> io::Result<bool> {
        if !self.is_v4_format() {
            return Ok(false);
        }

        let pending = self.pending_v4_in_memory_rows();
        if pending == 0 {
            return Ok(false);
        }

        let footer_guard = self.v4_footer.read();
        let Some(footer) = footer_guard.as_ref() else {
            return Ok(false);
        };
        let on_disk_rows: usize = footer
            .row_groups
            .iter()
            .map(|rg| rg.row_count as usize)
            .sum();
        if on_disk_rows == 0 {
            // New/empty tables must write the initial base file so schema/footer metadata
            // is persisted together with the first rows.
            return Ok(false);
        }

        let ids = self.ids.read();
        let ids_len = ids.len();
        if ids_len < pending {
            return Ok(false);
        }
        let pending_start = ids_len - pending;
        if ids.first().copied().unwrap_or(0) == 1 && pending_start < on_disk_rows {
            // Some persisted base rows are mixed into the in-memory prefix. Fall back to a
            // full save so we do not misclassify base rows as pending append-only rows.
            return Ok(false);
        }
        let deleted = self.deleted.read();
        if ids.first().copied().unwrap_or(0) == 1 {
            for row_idx in 0..pending_start.min(on_disk_rows) {
                let byte_idx = row_idx / 8;
                let bit_idx = row_idx % 8;
                if byte_idx < deleted.len() && ((deleted[byte_idx] >> bit_idx) & 1 == 1) {
                    // Persisted-base deletes require a full rewrite or delete-vector update.
                    return Ok(false);
                }
            }
        }
        if pending == 1 {
            let row_idx_abs = pending_start;
            let byte_idx = row_idx_abs / 8;
            let bit_idx = row_idx_abs % 8;
            let is_deleted =
                byte_idx < deleted.len() && ((deleted[byte_idx] >> bit_idx) & 1 == 1);
            if !is_deleted {
                let pending_id = ids[row_idx_abs];
                drop(ids);

                let schema = self.schema.read();
                let columns = self.columns.read();
                let nulls = self.nulls.read();
                if schema.columns != footer.schema.columns {
                    return Ok(false);
                }
                if columns.len() < schema.column_count() {
                    return Ok(false);
                }

                let mut int_columns: HashMap<String, Vec<i64>> =
                    HashMap::with_capacity(schema.column_count());
                let mut float_columns: HashMap<String, Vec<f64>> =
                    HashMap::with_capacity(schema.column_count());
                let mut string_columns: HashMap<String, Vec<String>> =
                    HashMap::with_capacity(schema.column_count());
                let mut bool_columns: HashMap<String, Vec<bool>> =
                    HashMap::with_capacity(schema.column_count());

                for (col_idx, (col_name, col_type)) in schema.columns.iter().enumerate() {
                    if let Some(bitmap) = nulls.get(col_idx) {
                        let byte_idx = row_idx_abs / 8;
                        let bit_idx = row_idx_abs % 8;
                        if byte_idx < bitmap.len() && (bitmap[byte_idx] >> bit_idx) & 1 == 1 {
                            return Ok(false);
                        }
                    }

                    match (&columns[col_idx], col_type) {
                        (
                            ColumnData::Int64(values),
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
                            let Some(&value) = values.get(row_idx_abs) else {
                                return Ok(false);
                            };
                            int_columns.insert(col_name.clone(), vec![value]);
                        }
                        (ColumnData::Float64(values), ColumnType::Float64 | ColumnType::Float32) => {
                            let Some(&value) = values.get(row_idx_abs) else {
                                return Ok(false);
                            };
                            float_columns.insert(col_name.clone(), vec![value]);
                        }
                        (
                            ColumnData::String { offsets, data },
                            ColumnType::String | ColumnType::Null,
                        ) => {
                            let Some((&start, &end)) =
                                offsets.get(row_idx_abs).zip(offsets.get(row_idx_abs + 1))
                            else {
                                return Ok(false);
                            };
                            let start = start as usize;
                            let end = end as usize;
                            if start > end || end > data.len() {
                                return Ok(false);
                            }
                            string_columns.insert(
                                col_name.clone(),
                                vec![std::str::from_utf8(&data[start..end])
                                    .unwrap_or("")
                                    .to_string()],
                            );
                        }
                        (
                            ColumnData::StringDict {
                                indices,
                                dict_offsets,
                                dict_data,
                            },
                            ColumnType::StringDict,
                        ) => {
                            let Some(&dict_idx) = indices.get(row_idx_abs) else {
                                return Ok(false);
                            };
                            if dict_idx == 0 {
                                return Ok(false);
                            }
                            let di = (dict_idx - 1) as usize;
                            let Some((&start, &end)) =
                                dict_offsets.get(di).zip(dict_offsets.get(di + 1))
                            else {
                                return Ok(false);
                            };
                            let start = start as usize;
                            let end = end as usize;
                            if start > end || end > dict_data.len() {
                                return Ok(false);
                            }
                            string_columns.insert(
                                col_name.clone(),
                                vec![std::str::from_utf8(&dict_data[start..end])
                                    .unwrap_or("")
                                    .to_string()],
                            );
                        }
                        (ColumnData::Bool { data, len }, ColumnType::Bool) => {
                            if row_idx_abs >= *len {
                                return Ok(false);
                            }
                            let byte_idx = row_idx_abs / 8;
                            let bit_idx = row_idx_abs % 8;
                            let value =
                                byte_idx < data.len() && ((data[byte_idx] >> bit_idx) & 1 == 1);
                            bool_columns.insert(col_name.clone(), vec![value]);
                        }
                        _ => return Ok(false),
                    }
                }

                drop(deleted);
                drop(nulls);
                drop(columns);
                drop(schema);

                self.append_typed_to_delta_with_ids(
                    &[pending_id],
                    &int_columns,
                    &float_columns,
                    &string_columns,
                    &bool_columns,
                )?;
                self.discard_pending_v4_rows_from(pending_start);
                return Ok(true);
            }
        }
        let mut live_row_indices_abs = Vec::with_capacity(pending);
        let mut live_row_indices_local = Vec::with_capacity(pending);
        for row_idx in pending_start..ids_len {
            let byte_idx = row_idx / 8;
            let bit_idx = row_idx % 8;
            let is_deleted = byte_idx < deleted.len() && ((deleted[byte_idx] >> bit_idx) & 1 == 1);
            if !is_deleted {
                live_row_indices_abs.push(row_idx);
                live_row_indices_local.push(row_idx - pending_start);
            }
        }
        let pending_ids: Vec<u64> = live_row_indices_abs
            .iter()
            .map(|&row_idx| ids[row_idx])
            .collect();
        drop(ids);

        let schema = self.schema.read();
        let columns = self.columns.read();
        let nulls = self.nulls.read();
        if schema.columns != footer.schema.columns {
            // Schema evolution must go through the normal save path so the base footer and
            // column layout stay in sync with what readers expect on reopen.
            return Ok(false);
        }
        if columns.len() < schema.column_count() {
            return Ok(false);
        }

        let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
        let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
        let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();

        for (col_idx, (col_name, col_type)) in schema.columns.iter().enumerate() {
            if let Some(bitmap) = nulls.get(col_idx) {
                for &row_idx in &live_row_indices_abs {
                    let byte_idx = row_idx / 8;
                    let bit_idx = row_idx % 8;
                    if byte_idx < bitmap.len() && (bitmap[byte_idx] >> bit_idx) & 1 == 1 {
                        return Ok(false);
                    }
                }
            }

            let sliced = columns[col_idx].slice_range(pending_start, ids_len);
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
                    let ColumnData::Int64(values) = sliced else {
                        return Ok(false);
                    };
                    if live_row_indices_local
                        .iter()
                        .any(|&row_idx| row_idx >= values.len())
                    {
                        return Ok(false);
                    }
                    let filtered: Vec<i64> = live_row_indices_local
                        .iter()
                        .map(|&row_idx| values[row_idx])
                        .collect();
                    int_columns.insert(col_name.clone(), filtered);
                }
                ColumnType::Float64 | ColumnType::Float32 => {
                    let ColumnData::Float64(values) = sliced else {
                        return Ok(false);
                    };
                    if live_row_indices_local
                        .iter()
                        .any(|&row_idx| row_idx >= values.len())
                    {
                        return Ok(false);
                    }
                    let filtered: Vec<f64> = live_row_indices_local
                        .iter()
                        .map(|&row_idx| values[row_idx])
                        .collect();
                    float_columns.insert(col_name.clone(), filtered);
                }
                ColumnType::String | ColumnType::StringDict | ColumnType::Null => {
                    let normalized = if matches!(sliced, ColumnData::StringDict { .. }) {
                        sliced.decode_string_dict()
                    } else {
                        sliced
                    };
                    let ColumnData::String { offsets, data } = normalized else {
                        return Ok(false);
                    };
                    let mut values = Vec::with_capacity(live_row_indices_local.len());
                    for &row_idx in &live_row_indices_local {
                        if row_idx + 1 >= offsets.len() {
                            return Ok(false);
                        }
                        let start = offsets[row_idx] as usize;
                        let end = offsets[row_idx + 1] as usize;
                        if start > end || end > data.len() {
                            return Ok(false);
                        }
                        values.push(
                            std::str::from_utf8(&data[start..end])
                                .unwrap_or("")
                                .to_string(),
                        );
                    }
                    string_columns.insert(col_name.clone(), values);
                }
                ColumnType::Bool => {
                    let ColumnData::Bool { data, len } = sliced else {
                        return Ok(false);
                    };
                    let mut values = Vec::with_capacity(live_row_indices_local.len());
                    for &row_idx in &live_row_indices_local {
                        if row_idx >= len {
                            return Ok(false);
                        }
                        let byte_idx = row_idx / 8;
                        let bit_idx = row_idx % 8;
                        let value = byte_idx < data.len() && ((data[byte_idx] >> bit_idx) & 1 == 1);
                        values.push(value);
                    }
                    bool_columns.insert(col_name.clone(), values);
                }
                ColumnType::Binary | ColumnType::FixedList | ColumnType::Float16List => {
                    return Ok(false);
                }
            }
        }

        drop(deleted);
        drop(nulls);
        drop(columns);
        drop(schema);

        if !pending_ids.is_empty() {
            self.append_typed_to_delta_with_ids(
                &pending_ids,
                &int_columns,
                &float_columns,
                &string_columns,
                &bool_columns,
            )?;
        }
        self.discard_pending_v4_rows_from(pending_start);
        Ok(true)
    }

    /// Compact: merge delta file into base file
    ///
    /// MEMORY EFFICIENT: Uses column-streaming merge.
    /// Processes one column at a time via mmap, never loading all columns simultaneously.
    /// Peak memory ≈ max(single column) + delta data, instead of ALL columns + delta.
    ///
    /// For a 10M-row × 5-column table, this reduces peak memory from ~800MB to ~160MB.
    pub fn compact(&self) -> io::Result<()> {
        let delta_path = Self::delta_path(&self.path);
        if !delta_path.exists() {
            return Ok(());
        }

        self.load_all_columns_into_memory()?;
        self.merge_delta_file(&delta_path)?;
        self.save()?;

        // Delete delta file
        *self.delta_file.write() = None;
        let _ = std::fs::remove_file(&delta_path);
        let _ = std::fs::remove_file(Self::delta_meta_path(&delta_path));

        Ok(())
    }

    /// Convert an Arrow ArrayRef to ColumnData, preserving nulls.
    fn arrow_array_to_column_data(array: &dyn arrow::array::Array) -> ColumnData {
        use arrow::array::{
            Array, BinaryArray, BooleanArray, Float64Array, Int64Array, StringArray,
        };
        use arrow::datatypes::DataType as ArrowDT;
        match array.data_type() {
            ArrowDT::Int64 => {
                let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
                ColumnData::Int64(arr.values().to_vec())
            }
            ArrowDT::Float64 => {
                let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
                ColumnData::Float64(arr.values().to_vec())
            }
            ArrowDT::Utf8 => {
                let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
                let mut offsets = Vec::with_capacity(arr.len() + 1);
                let mut data = Vec::new();
                offsets.push(0u32);
                for j in 0..arr.len() {
                    if arr.is_null(j) {
                        offsets.push(data.len() as u32);
                    } else {
                        let s = arr.value(j).as_bytes();
                        data.extend_from_slice(s);
                        offsets.push(data.len() as u32);
                    }
                }
                ColumnData::String { offsets, data }
            }
            ArrowDT::Boolean => {
                let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                let n = arr.len();
                let byte_len = (n + 7) / 8;
                let mut bits = vec![0u8; byte_len];
                for j in 0..n {
                    if !arr.is_null(j) && arr.value(j) {
                        bits[j / 8] |= 1 << (j % 8);
                    }
                }
                ColumnData::Bool { data: bits, len: n }
            }
            ArrowDT::Binary => {
                let arr = array.as_any().downcast_ref::<BinaryArray>().unwrap();
                let mut offsets = Vec::with_capacity(arr.len() + 1);
                let mut data = Vec::new();
                offsets.push(0u32);
                for j in 0..arr.len() {
                    if arr.is_null(j) {
                        offsets.push(data.len() as u32);
                    } else {
                        data.extend_from_slice(arr.value(j));
                        offsets.push(data.len() as u32);
                    }
                }
                ColumnData::Binary { offsets, data }
            }
            _ => ColumnData::new(ColumnType::Int64),
        }
    }

    /// Create a column filled with default values (0, 0.0, "", false).
    /// Used for columns added via ALTER TABLE that have no disk data yet.
    fn create_default_column(dtype: ColumnType, count: usize) -> ColumnData {
        if count == 0 {
            return ColumnData::new(dtype);
        }
        match dtype {
            ColumnType::Bool => ColumnData::Bool {
                data: vec![0u8; (count + 7) / 8],
                len: count,
            },
            ColumnType::Int64
            | ColumnType::Int8
            | ColumnType::Int16
            | ColumnType::Int32
            | ColumnType::UInt8
            | ColumnType::UInt16
            | ColumnType::UInt32
            | ColumnType::UInt64
            | ColumnType::Timestamp
            | ColumnType::Date => ColumnData::Int64(vec![0i64; count]),
            ColumnType::Float64 | ColumnType::Float32 => ColumnData::Float64(vec![0.0f64; count]),
            ColumnType::String | ColumnType::StringDict => ColumnData::String {
                offsets: vec![0u32; count + 1],
                data: Vec::new(),
            },
            ColumnType::Binary => ColumnData::Binary {
                offsets: vec![0u32; count + 1],
                data: Vec::new(),
            },
            ColumnType::FixedList => ColumnData::FixedList {
                data: Vec::new(),
                dim: 0,
            },
            ColumnType::Float16List => ColumnData::Float16List {
                data: Vec::new(),
                dim: 0,
            },
            ColumnType::Null => ColumnData::Int64(vec![0i64; count]),
        }
    }

    // compact_column_streaming removed — was legacy dead code (326 lines).
    // save() always produces V4 format; compact() uses in-memory merge path.

    /// Read delta file and merge into in-memory columns
    fn merge_delta_file(&self, delta_path: &Path) -> io::Result<()> {
        let mut file = File::open(delta_path)?;

        loop {
            // Try to read record count
            let mut count_buf = [0u8; 8];
            match file.read_exact(&mut count_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            let record_count = u64::from_le_bytes(count_buf) as usize;

            // Read IDs
            let mut delta_ids = Vec::with_capacity(record_count);
            for _ in 0..record_count {
                let mut id_buf = [0u8; 8];
                file.read_exact(&mut id_buf)?;
                delta_ids.push(u64::from_le_bytes(id_buf));
            }

            // Read int columns
            let mut int_columns: HashMap<String, Vec<i64>> = HashMap::new();
            let mut count_buf = [0u8; 4];
            file.read_exact(&mut count_buf)?;
            let int_col_count = u32::from_le_bytes(count_buf) as usize;
            for _ in 0..int_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();
                let mut values = Vec::with_capacity(record_count);
                for _ in 0..record_count {
                    let mut v_buf = [0u8; 8];
                    file.read_exact(&mut v_buf)?;
                    values.push(i64::from_le_bytes(v_buf));
                }
                int_columns.insert(name, values);
            }

            // Read float columns
            let mut float_columns: HashMap<String, Vec<f64>> = HashMap::new();
            file.read_exact(&mut count_buf)?;
            let float_col_count = u32::from_le_bytes(count_buf) as usize;
            for _ in 0..float_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();
                let mut values = Vec::with_capacity(record_count);
                for _ in 0..record_count {
                    let mut v_buf = [0u8; 8];
                    file.read_exact(&mut v_buf)?;
                    values.push(f64::from_le_bytes(v_buf));
                }
                float_columns.insert(name, values);
            }

            // Read string columns
            let mut string_columns: HashMap<String, Vec<String>> = HashMap::new();
            file.read_exact(&mut count_buf)?;
            let string_col_count = u32::from_le_bytes(count_buf) as usize;
            for _ in 0..string_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();
                let mut values = Vec::with_capacity(record_count);
                for _ in 0..record_count {
                    let mut str_len_buf = [0u8; 4];
                    file.read_exact(&mut str_len_buf)?;
                    let str_len = u32::from_le_bytes(str_len_buf) as usize;
                    let mut str_buf = vec![0u8; str_len];
                    file.read_exact(&mut str_buf)?;
                    let val = String::from_utf8_lossy(&str_buf).to_string();
                    values.push(val);
                }
                string_columns.insert(name, values);
            }

            // Read bool columns
            let mut bool_columns: HashMap<String, Vec<bool>> = HashMap::new();
            file.read_exact(&mut count_buf)?;
            let bool_col_count = u32::from_le_bytes(count_buf) as usize;
            for _ in 0..bool_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();
                let mut values = Vec::with_capacity(record_count);
                for _ in 0..record_count {
                    let mut v_buf = [0u8; 1];
                    file.read_exact(&mut v_buf)?;
                    values.push(v_buf[0] != 0);
                }
                bool_columns.insert(name, values);
            }

            // Merge into in-memory columns PRESERVING original delta IDs
            // This is critical for correct ID assignment after delete operations
            self.insert_typed_with_ids(
                &delta_ids,
                int_columns,
                float_columns,
                string_columns,
                HashMap::new(), // binary columns (not implemented in delta yet)
                bool_columns,
            )?;
        }

        Ok(())
    }

    /// Read delta file and return column data without merging into memory
    /// Returns: (delta_ids, column_data_map) where column_data_map is column_name -> ColumnData
    fn read_delta_data(&self) -> io::Result<Option<(Vec<u64>, HashMap<String, ColumnData>)>> {
        let delta_path = Self::delta_path(&self.path);
        if !delta_path.exists() {
            return Ok(None);
        }

        let mut file = File::open(&delta_path)?;
        let mut all_ids: Vec<u64> = Vec::new();
        let mut all_columns: HashMap<String, ColumnData> = HashMap::new();

        loop {
            // Try to read record count
            let mut count_buf = [0u8; 8];
            match file.read_exact(&mut count_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            let record_count = u64::from_le_bytes(count_buf) as usize;

            // Read IDs
            for _ in 0..record_count {
                let mut id_buf = [0u8; 8];
                file.read_exact(&mut id_buf)?;
                all_ids.push(u64::from_le_bytes(id_buf));
            }

            // Read int columns
            let mut count_buf4 = [0u8; 4];
            file.read_exact(&mut count_buf4)?;
            let int_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..int_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();

                let col_data = all_columns
                    .entry(name)
                    .or_insert_with(|| ColumnData::new(ColumnType::Int64));
                for _ in 0..record_count {
                    let mut v_buf = [0u8; 8];
                    file.read_exact(&mut v_buf)?;
                    col_data.push_i64(i64::from_le_bytes(v_buf));
                }
            }

            // Read float columns
            file.read_exact(&mut count_buf4)?;
            let float_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..float_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();

                let col_data = all_columns
                    .entry(name)
                    .or_insert_with(|| ColumnData::new(ColumnType::Float64));
                for _ in 0..record_count {
                    let mut v_buf = [0u8; 8];
                    file.read_exact(&mut v_buf)?;
                    col_data.push_f64(f64::from_le_bytes(v_buf));
                }
            }

            // Read string columns
            file.read_exact(&mut count_buf4)?;
            let string_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..string_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();

                let col_data = all_columns
                    .entry(name)
                    .or_insert_with(|| ColumnData::new(ColumnType::String));
                for _ in 0..record_count {
                    let mut str_len_buf = [0u8; 4];
                    file.read_exact(&mut str_len_buf)?;
                    let str_len = u32::from_le_bytes(str_len_buf) as usize;
                    let mut str_buf = vec![0u8; str_len];
                    file.read_exact(&mut str_buf)?;
                    let val = String::from_utf8_lossy(&str_buf).to_string();
                    col_data.push_string(&val);
                }
            }

            // Read bool columns
            file.read_exact(&mut count_buf4)?;
            let bool_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..bool_col_count {
                let mut len_buf = [0u8; 2];
                file.read_exact(&mut len_buf)?;
                let name_len = u16::from_le_bytes(len_buf) as usize;
                let mut name_buf = vec![0u8; name_len];
                file.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).to_string();

                let col_data = all_columns
                    .entry(name)
                    .or_insert_with(|| ColumnData::new(ColumnType::Bool));
                for _ in 0..record_count {
                    let mut v_buf = [0u8; 1];
                    file.read_exact(&mut v_buf)?;
                    col_data.push_bool(v_buf[0] != 0);
                }
            }
        }

        if all_ids.is_empty() {
            Ok(None)
        } else {
            Ok(Some((all_ids, all_columns)))
        }
    }

    #[inline]
    fn column_string_at(col: &ColumnData, row_idx: usize) -> Option<&str> {
        match col {
            ColumnData::String { offsets, data } => {
                if row_idx + 1 >= offsets.len() {
                    return None;
                }
                let start = offsets[row_idx] as usize;
                let end = offsets[row_idx + 1] as usize;
                if start <= end && end <= data.len() {
                    std::str::from_utf8(&data[start..end]).ok()
                } else {
                    None
                }
            }
            ColumnData::StringDict {
                indices,
                dict_offsets,
                dict_data,
            } => {
                let dict_idx = *indices.get(row_idx)?;
                if dict_idx == 0 {
                    return None;
                }
                let di = (dict_idx - 1) as usize;
                let start = *dict_offsets.get(di)? as usize;
                let end = if di + 1 < dict_offsets.len() {
                    dict_offsets[di + 1] as usize
                } else {
                    dict_data.len()
                };
                if start <= end && end <= dict_data.len() {
                    std::str::from_utf8(&dict_data[start..end]).ok()
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    #[inline]
    fn column_binary_at(col: &ColumnData, row_idx: usize) -> Option<&[u8]> {
        match col {
            ColumnData::Binary { offsets, data } => {
                if row_idx + 1 >= offsets.len() {
                    return None;
                }
                let start = offsets[row_idx] as usize;
                let end = offsets[row_idx + 1] as usize;
                if start <= end && end <= data.len() {
                    Some(&data[start..end])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    #[inline]
    fn column_bool_at(col: &ColumnData, row_idx: usize) -> Option<bool> {
        match col {
            ColumnData::Bool { data, len } if row_idx < *len => {
                let byte_idx = row_idx / 8;
                let bit_idx = row_idx % 8;
                data.get(byte_idx).map(|b| ((b >> bit_idx) & 1) == 1)
            }
            _ => None,
        }
    }

    /// Return committed append-only delta row IDs whose string column equals `target`.
    /// This lets string equality filters stay mmap-fast without compacting `.delta`.
    pub fn delta_string_match_ids(&self, column_name: &str, target: &str) -> io::Result<Vec<u64>> {
        let delta_path = Self::delta_path(&self.path);
        if !delta_path.exists() {
            DELTA_STRING_INDEX_CACHE.write().remove(&delta_path);
            return Ok(Vec::new());
        };

        #[inline]
        fn take_slice<'a>(bytes: &'a [u8], pos: &mut usize, len: usize) -> io::Result<&'a [u8]> {
            let end = pos
                .checked_add(len)
                .ok_or_else(|| err_data("delta string scan offset overflow"))?;
            if end > bytes.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "delta string scan truncated",
                ));
            }
            let out = &bytes[*pos..end];
            *pos = end;
            Ok(out)
        }

        #[inline]
        fn read_u16(bytes: &[u8], pos: &mut usize) -> io::Result<u16> {
            let raw = take_slice(bytes, pos, 2)?;
            Ok(u16::from_le_bytes(raw.try_into().unwrap()))
        }

        #[inline]
        fn read_u32(bytes: &[u8], pos: &mut usize) -> io::Result<u32> {
            let raw = take_slice(bytes, pos, 4)?;
            Ok(u32::from_le_bytes(raw.try_into().unwrap()))
        }

        #[inline]
        fn read_u64(bytes: &[u8], pos: &mut usize) -> io::Result<u64> {
            let raw = take_slice(bytes, pos, 8)?;
            Ok(u64::from_le_bytes(raw.try_into().unwrap()))
        }

        fn parse_delta_string_index(
            bytes: &[u8],
            index: &mut HashMap<String, HashMap<String, Vec<u64>>>,
        ) -> io::Result<()> {
            let mut pos = 0usize;
            while pos < bytes.len() {
            if bytes.len() - pos < 8 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "delta string scan truncated record count",
                ));
            }
            let record_count = read_u64(&bytes, &mut pos)? as usize;

            let mut ids = Vec::with_capacity(record_count);
            for _ in 0..record_count {
                ids.push(read_u64(&bytes, &mut pos)?);
            }

            let int_col_count = read_u32(&bytes, &mut pos)? as usize;
            for _ in 0..int_col_count {
                let name_len = read_u16(&bytes, &mut pos)? as usize;
                take_slice(&bytes, &mut pos, name_len)?;
                take_slice(&bytes, &mut pos, record_count * 8)?;
            }

            let float_col_count = read_u32(&bytes, &mut pos)? as usize;
            for _ in 0..float_col_count {
                let name_len = read_u16(&bytes, &mut pos)? as usize;
                take_slice(&bytes, &mut pos, name_len)?;
                take_slice(&bytes, &mut pos, record_count * 8)?;
            }

            let string_col_count = read_u32(&bytes, &mut pos)? as usize;
            for _ in 0..string_col_count {
                let name_len = read_u16(&bytes, &mut pos)? as usize;
                let name = take_slice(&bytes, &mut pos, name_len)?;
                let col_name = String::from_utf8_lossy(name).into_owned();
                let col_index = index.entry(col_name).or_default();

                for row_idx in 0..record_count {
                    let str_len = read_u32(&bytes, &mut pos)? as usize;
                    let value = take_slice(&bytes, &mut pos, str_len)?;
                    if let Some(id) = ids.get(row_idx) {
                        let value = String::from_utf8_lossy(value).into_owned();
                        col_index.entry(value).or_default().push(*id);
                    }
                }
            }

            let bool_col_count = read_u32(&bytes, &mut pos)? as usize;
            for _ in 0..bool_col_count {
                let name_len = read_u16(&bytes, &mut pos)? as usize;
                take_slice(&bytes, &mut pos, name_len)?;
                take_slice(&bytes, &mut pos, record_count)?;
            }
        }
            Ok(())
        }

        let metadata = std::fs::metadata(&delta_path)?;
        let file_len = metadata.len();
        let modified = metadata
            .modified()
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

        let mut cache = DELTA_STRING_INDEX_CACHE.write();
        if cache.len() > 128 {
            cache.clear();
        }

        let entry = cache.entry(delta_path.clone()).or_insert_with(|| DeltaStringIndexCache {
            len: 0,
            modified: std::time::SystemTime::UNIX_EPOCH,
            index: HashMap::new(),
        });

        let up_to_date = entry.len == file_len && entry.modified >= modified;
        let can_append = entry.len > 0 && entry.len < file_len && entry.modified <= modified;
        if !up_to_date && !can_append {
            entry.len = 0;
            entry.index.clear();
        }

        if !up_to_date {
            let bytes = std::fs::read(&delta_path)?;
            let start = if can_append { entry.len as usize } else { 0 };
            parse_delta_string_index(&bytes[start..], &mut entry.index)?;
            entry.len = file_len;
            entry.modified = modified;
        }

        Ok(entry
            .index
            .get(column_name)
            .and_then(|values| values.get(target))
            .cloned()
            .unwrap_or_default())
    }

    /// Materialize committed append-only delta rows by ID in caller order.
    pub fn read_delta_rows_by_ids_to_arrow(
        &self,
        ids: &[u64],
    ) -> io::Result<arrow::record_batch::RecordBatch> {
        use arrow::array::{
            ArrayRef, BinaryArray, BooleanArray, Float64Array, Int64Array, StringArray,
        };
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        let schema_cols = self.schema.read().columns.clone();
        let empty_batch = || -> io::Result<arrow::record_batch::RecordBatch> {
            let mut fields = Vec::with_capacity(schema_cols.len() + 1);
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(schema_cols.len() + 1);
            fields.push(Field::new("_id", ArrowDataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef);
            for (name, col_type) in &schema_cols {
                let (dt, array): (ArrowDataType, ArrayRef) = match col_type {
                    ColumnType::Bool => (
                        ArrowDataType::Boolean,
                        Arc::new(BooleanArray::from(Vec::<Option<bool>>::new())),
                    ),
                    ColumnType::Float32 | ColumnType::Float64 => (
                        ArrowDataType::Float64,
                        Arc::new(Float64Array::from(Vec::<Option<f64>>::new())),
                    ),
                    ColumnType::Binary | ColumnType::FixedList | ColumnType::Float16List => (
                        ArrowDataType::Binary,
                        Arc::new(BinaryArray::from(Vec::<Option<&[u8]>>::new())),
                    ),
                    ColumnType::String | ColumnType::StringDict | ColumnType::Null => (
                        ArrowDataType::Utf8,
                        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
                    ),
                    _ => (
                        ArrowDataType::Int64,
                        Arc::new(Int64Array::from(Vec::<Option<i64>>::new())),
                    ),
                };
                fields.push(Field::new(name, dt, true));
                arrays.push(array);
            }
            arrow::record_batch::RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
        };

        if ids.is_empty() {
            return empty_batch();
        }

        let Some((delta_ids, delta_columns)) = self.read_delta_data()? else {
            return empty_batch();
        };
        let delta_pos: HashMap<u64, usize> = delta_ids
            .iter()
            .enumerate()
            .map(|(idx, id)| (*id, idx))
            .collect();
        let positions: Vec<(u64, usize)> = ids
            .iter()
            .filter_map(|id| delta_pos.get(id).copied().map(|pos| (*id, pos)))
            .collect();
        if positions.is_empty() {
            return empty_batch();
        }

        let mut fields = Vec::with_capacity(schema_cols.len() + 1);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(schema_cols.len() + 1);
        let row_ids: Vec<i64> = positions.iter().map(|(id, _)| *id as i64).collect();
        fields.push(Field::new("_id", ArrowDataType::Int64, false));
        arrays.push(Arc::new(Int64Array::from(row_ids)) as ArrayRef);

        for (name, col_type) in &schema_cols {
            let column = delta_columns.get(name);
            let (dt, array): (ArrowDataType, ArrayRef) = match col_type {
                ColumnType::Bool => {
                    let values: Vec<Option<bool>> = positions
                        .iter()
                        .map(|(_, row_idx)| column.and_then(|c| Self::column_bool_at(c, *row_idx)))
                        .collect();
                    (ArrowDataType::Boolean, Arc::new(BooleanArray::from(values)))
                }
                ColumnType::Float32 | ColumnType::Float64 => {
                    let values: Vec<Option<f64>> = positions
                        .iter()
                        .map(|(_, row_idx)| match column {
                            Some(ColumnData::Float64(values)) => values.get(*row_idx).copied(),
                            _ => None,
                        })
                        .collect();
                    (ArrowDataType::Float64, Arc::new(Float64Array::from(values)))
                }
                ColumnType::String | ColumnType::StringDict | ColumnType::Null => {
                    let values: Vec<Option<String>> = positions
                        .iter()
                        .map(|(_, row_idx)| {
                            column
                                .and_then(|c| Self::column_string_at(c, *row_idx))
                                .map(str::to_owned)
                        })
                        .collect();
                    let refs: Vec<Option<&str>> = values.iter().map(|v| v.as_deref()).collect();
                    (ArrowDataType::Utf8, Arc::new(StringArray::from(refs)))
                }
                ColumnType::Binary | ColumnType::FixedList | ColumnType::Float16List => {
                    let values: Vec<Option<Vec<u8>>> = positions
                        .iter()
                        .map(|(_, row_idx)| {
                            column
                                .and_then(|c| Self::column_binary_at(c, *row_idx))
                                .map(|b| b.to_vec())
                        })
                        .collect();
                    let refs: Vec<Option<&[u8]>> = values.iter().map(|v| v.as_deref()).collect();
                    (ArrowDataType::Binary, Arc::new(BinaryArray::from(refs)))
                }
                _ => {
                    let values: Vec<Option<i64>> = positions
                        .iter()
                        .map(|(_, row_idx)| match column {
                            Some(ColumnData::Int64(values)) => values.get(*row_idx).copied(),
                            _ => None,
                        })
                        .collect();
                    (ArrowDataType::Int64, Arc::new(Int64Array::from(values)))
                }
            };
            fields.push(Field::new(name, dt, true));
            arrays.push(array);
        }

        arrow::record_batch::RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Get the total row count including delta rows (for accurate row_count reporting)
    fn delta_row_count(&self) -> usize {
        let delta_path = Self::delta_path(&self.path);
        let Ok(metadata) = std::fs::metadata(&delta_path) else {
            DELTA_ROW_COUNT_CACHE.write().remove(&delta_path);
            return 0;
        };

        let file_len = metadata.len();
        let modified = metadata
            .modified()
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

        {
            let cache = DELTA_ROW_COUNT_CACHE.read();
            if let Some((cached_len, cached_modified, cached_count)) = cache.get(&delta_path) {
                if *cached_len == file_len && *cached_modified >= modified {
                    return *cached_count;
                }
            }
        }

        let (start, mut total) = {
            let cache = DELTA_ROW_COUNT_CACHE.read();
            if let Some((cached_len, cached_modified, cached_count)) = cache.get(&delta_path) {
                if *cached_len < file_len && *cached_modified <= modified {
                    (*cached_len, *cached_count)
                } else {
                    (0, 0)
                }
            } else {
                (0, 0)
            }
        };

        let Ok(mut file) = File::open(&delta_path) else {
            return 0;
        };
        if start > 0 && file.seek(SeekFrom::Start(start)).is_err() {
            total = 0;
            let _ = file.seek(SeekFrom::Start(0));
        }

        loop {
            let mut count_buf = [0u8; 8];
            match file.read_exact(&mut count_buf) {
                Ok(_) => {}
                Err(_) => break,
            }
            let record_count = u64::from_le_bytes(count_buf) as usize;
            total += record_count;

            // Skip the rest of this record block
            // IDs
            if file
                .seek(SeekFrom::Current((record_count * 8) as i64))
                .is_err()
            {
                break;
            }

            // Int columns
            let mut count_buf4 = [0u8; 4];
            if file.read_exact(&mut count_buf4).is_err() {
                break;
            }
            let int_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..int_col_count {
                let mut len_buf = [0u8; 2];
                if file.read_exact(&mut len_buf).is_err() {
                    break;
                }
                let name_len = u16::from_le_bytes(len_buf) as usize;
                if file
                    .seek(SeekFrom::Current(
                        name_len as i64 + (record_count * 8) as i64,
                    ))
                    .is_err()
                {
                    break;
                }
            }

            // Float columns
            if file.read_exact(&mut count_buf4).is_err() {
                break;
            }
            let float_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..float_col_count {
                let mut len_buf = [0u8; 2];
                if file.read_exact(&mut len_buf).is_err() {
                    break;
                }
                let name_len = u16::from_le_bytes(len_buf) as usize;
                if file
                    .seek(SeekFrom::Current(
                        name_len as i64 + (record_count * 8) as i64,
                    ))
                    .is_err()
                {
                    break;
                }
            }

            // String columns - variable length, need to read each
            if file.read_exact(&mut count_buf4).is_err() {
                break;
            }
            let string_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..string_col_count {
                let mut len_buf = [0u8; 2];
                if file.read_exact(&mut len_buf).is_err() {
                    break;
                }
                let name_len = u16::from_le_bytes(len_buf) as usize;
                if file.seek(SeekFrom::Current(name_len as i64)).is_err() {
                    break;
                }
                for _ in 0..record_count {
                    let mut str_len_buf = [0u8; 4];
                    if file.read_exact(&mut str_len_buf).is_err() {
                        break;
                    }
                    let str_len = u32::from_le_bytes(str_len_buf) as usize;
                    if file.seek(SeekFrom::Current(str_len as i64)).is_err() {
                        break;
                    }
                }
            }

            // Bool columns
            if file.read_exact(&mut count_buf4).is_err() {
                break;
            }
            let bool_col_count = u32::from_le_bytes(count_buf4) as usize;
            for _ in 0..bool_col_count {
                let mut len_buf = [0u8; 2];
                if file.read_exact(&mut len_buf).is_err() {
                    break;
                }
                let name_len = u16::from_le_bytes(len_buf) as usize;
                if file
                    .seek(SeekFrom::Current(name_len as i64 + record_count as i64))
                    .is_err()
                {
                    break;
                }
            }
        }

        let mut cache = DELTA_ROW_COUNT_CACHE.write();
        if cache.len() > 128 {
            cache.clear();
        }
        cache.insert(delta_path, (file_len, modified, total));
        total
    }
}
