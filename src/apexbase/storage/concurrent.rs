//! Concurrent Access Statistics and Monitoring
//!
//! Provides thread-safe statistics collection for storage operations
//! without sacrificing performance. Uses lock-free atomic operations
//! for hot path metrics.
//!
//! # Design Principles
//! - Lock-free for all hot path operations (read/write counters)
//! - Minimal overhead: single atomic increment for most operations
//! - Periodic snapshot for detailed analytics
//! - Cache-line aligned counters to avoid false sharing

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Aligned counter to avoid false sharing between atomic counters
/// Uses 64-byte alignment to prevent false sharing in multi-threaded access
#[repr(align(64))]
struct AlignedCounter(AtomicU64);

impl AlignedCounter {
    fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    #[inline]
    fn fetch_add(&self, val: u64, order: Ordering) -> u64 {
        self.0.fetch_add(val, order)
    }

    #[inline]
    fn load(&self, order: Ordering) -> u64 {
        self.0.load(order)
    }

    #[inline]
    fn fetch_max(&self, val: u64, order: Ordering) -> u64 {
        // Use CAS loop for fetch_max (not directly available in atomic)
        let mut current = self.0.load(order);
        while current < val {
            match self
                .0
                .compare_exchange_weak(current, val, order, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(v) => current = v,
            }
        }
        current
    }
}

/// Storage operation statistics
///
/// Thread-safe statistics collection using lock-free atomics
/// for high-performance concurrent access monitoring.
pub struct StorageStats {
    /// Total number of read operations
    read_count: AlignedCounter,
    /// Total bytes read
    read_bytes: AlignedCounter,
    /// Total number of write operations  
    write_count: AlignedCounter,
    /// Total bytes written
    write_bytes: AlignedCounter,
    /// Number of cache hits (mmap reused)
    cache_hits: AlignedCounter,
    /// Number of cache misses (mmap created)
    cache_misses: AlignedCounter,
    /// Number of concurrent readers (peak)
    peak_concurrent_readers: AlignedCounter,
    /// Number of concurrent writers (peak)
    peak_concurrent_writers: AlignedCounter,
    /// Total lock wait time in microseconds (approximate)
    lock_wait_us: AlignedCounter,
    /// Number of WAL append operations
    wal_append_count: AlignedCounter,
    /// Number of flush operations
    flush_count: AlignedCounter,
    /// Current active readers (atomic for fast updates)
    active_readers: AtomicUsize,
    /// Current active writers (atomic for fast updates)
    active_writers: AtomicUsize,
    /// Statistics snapshot timestamp
    last_snapshot: AtomicU64,
    /// Snapshot interval in seconds
    snapshot_interval: u64,
}

impl StorageStats {
    /// Create new storage statistics
    pub fn new() -> Self {
        Self {
            read_count: AlignedCounter::new(),
            read_bytes: AlignedCounter::new(),
            write_count: AlignedCounter::new(),
            write_bytes: AlignedCounter::new(),
            cache_hits: AlignedCounter::new(),
            cache_misses: AlignedCounter::new(),
            peak_concurrent_readers: AlignedCounter::new(),
            peak_concurrent_writers: AlignedCounter::new(),
            lock_wait_us: AlignedCounter::new(),
            wal_append_count: AlignedCounter::new(),
            flush_count: AlignedCounter::new(),
            active_readers: AtomicUsize::new(0),
            active_writers: AtomicUsize::new(0),
            last_snapshot: AtomicU64::new(0),
            snapshot_interval: 60, // Default 60 second interval
        }
    }

    /// Record a read operation
    #[inline]
    pub fn record_read(&self, bytes: u64) {
        self.read_count.fetch_add(1, Ordering::Relaxed);
        self.read_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a write operation
    #[inline]
    pub fn record_write(&self, bytes: u64) {
        self.write_count.fetch_add(1, Ordering::Relaxed);
        self.write_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a cache hit
    #[inline]
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss
    #[inline]
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a WAL append
    #[inline]
    pub fn record_wal_append(&self) {
        self.wal_append_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a flush operation
    #[inline]
    pub fn record_flush(&self) {
        self.flush_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record lock wait time
    #[inline]
    pub fn record_lock_wait(&self, us: u64) {
        self.lock_wait_us.fetch_add(us, Ordering::Relaxed);
    }

    /// Start a read operation, returns guard to auto-release
    #[inline]
    pub fn begin_read(&self) -> ReadGuard<'_> {
        let current = self.active_readers.fetch_add(1, Ordering::Relaxed) + 1;

        // Update peak if necessary
        let peak = self.peak_concurrent_readers.load(Ordering::Relaxed);
        if current as u64 > peak {
            self.peak_concurrent_readers
                .fetch_max(current as u64, Ordering::Relaxed);
        }

        ReadGuard { stats: self }
    }

    /// Start a write operation, returns guard to auto-release
    #[inline]
    pub fn begin_write(&self) -> WriteGuard<'_> {
        let current = self.active_writers.fetch_add(1, Ordering::Relaxed) + 1;

        // Update peak if necessary
        let peak = self.peak_concurrent_writers.load(Ordering::Relaxed);
        if current as u64 > peak {
            self.peak_concurrent_writers
                .fetch_max(current as u64, Ordering::Relaxed);
        }

        WriteGuard { stats: self }
    }

    /// Get current active readers
    #[inline]
    pub fn active_readers_count(&self) -> usize {
        self.active_readers.load(Ordering::Relaxed)
    }

    /// Get current active writers
    #[inline]
    pub fn active_writers_count(&self) -> usize {
        self.active_writers.load(Ordering::Relaxed)
    }

    /// Get cache hit rate as a percentage (0-100)
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            100.0
        } else {
            (hits as f64 / total as f64) * 100.0
        }
    }

    /// Get total read throughput in MB/s since last snapshot
    pub fn read_throughput_mbps(&self) -> f64 {
        let bytes = self.read_bytes.load(Ordering::Relaxed);
        let interval = self.snapshot_interval as f64;
        if interval > 0.0 {
            bytes as f64 / (1024.0 * 1024.0 * interval)
        } else {
            0.0
        }
    }

    /// Get total write throughput in MB/s since last snapshot
    pub fn write_throughput_mbps(&self) -> f64 {
        let bytes = self.write_bytes.load(Ordering::Relaxed);
        let interval = self.snapshot_interval as f64;
        if interval > 0.0 {
            bytes as f64 / (1024.0 * 1024.0 * interval)
        } else {
            0.0
        }
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.read_count.0.store(0, Ordering::Relaxed);
        self.read_bytes.0.store(0, Ordering::Relaxed);
        self.write_count.0.store(0, Ordering::Relaxed);
        self.write_bytes.0.store(0, Ordering::Relaxed);
        self.cache_hits.0.store(0, Ordering::Relaxed);
        self.cache_misses.0.store(0, Ordering::Relaxed);
        self.peak_concurrent_readers.0.store(0, Ordering::Relaxed);
        self.peak_concurrent_writers.0.store(0, Ordering::Relaxed);
        self.lock_wait_us.0.store(0, Ordering::Relaxed);
        self.wal_append_count.0.store(0, Ordering::Relaxed);
        self.flush_count.0.store(0, Ordering::Relaxed);
        self.active_readers.store(0, Ordering::Relaxed);
        self.active_writers.store(0, Ordering::Relaxed);
    }

    /// Take a snapshot of current statistics
    pub fn snapshot(&self) -> StorageSnapshot {
        StorageSnapshot {
            read_count: self.read_count.load(Ordering::Relaxed),
            read_bytes: self.read_bytes.load(Ordering::Relaxed),
            write_count: self.write_count.load(Ordering::Relaxed),
            write_bytes: self.write_bytes.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            peak_concurrent_readers: self.peak_concurrent_readers.load(Ordering::Relaxed),
            peak_concurrent_writers: self.peak_concurrent_writers.load(Ordering::Relaxed),
            wal_append_count: self.wal_append_count.load(Ordering::Relaxed),
            flush_count: self.flush_count.load(Ordering::Relaxed),
            active_readers: self.active_readers.load(Ordering::Relaxed),
            active_writers: self.active_writers.load(Ordering::Relaxed),
            cache_hit_rate: self.cache_hit_rate(),
            read_throughput_mbps: self.read_throughput_mbps(),
            write_throughput_mbps: self.write_throughput_mbps(),
            timestamp: Instant::now(),
        }
    }
}

impl Default for StorageStats {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for read operations - automatically releases on drop
pub struct ReadGuard<'a> {
    stats: &'a StorageStats,
}

impl<'a> Drop for ReadGuard<'a> {
    #[inline]
    fn drop(&mut self) {
        self.stats.active_readers.fetch_sub(1, Ordering::Relaxed);
    }
}

/// RAII guard for write operations - automatically releases on drop
pub struct WriteGuard<'a> {
    stats: &'a StorageStats,
}

impl<'a> Drop for WriteGuard<'a> {
    #[inline]
    fn drop(&mut self) {
        self.stats.active_writers.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Snapshot of storage statistics at a point in time
pub struct StorageSnapshot {
    pub read_count: u64,
    pub read_bytes: u64,
    pub write_count: u64,
    pub write_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub peak_concurrent_readers: u64,
    pub peak_concurrent_writers: u64,
    pub wal_append_count: u64,
    pub flush_count: u64,
    pub active_readers: usize,
    pub active_writers: usize,
    pub cache_hit_rate: f64,
    pub read_throughput_mbps: f64,
    pub write_throughput_mbps: f64,
    pub timestamp: Instant,
}

impl StorageSnapshot {
    /// Format snapshot as human-readable string
    pub fn format(&self) -> String {
        format!(
            "StorageStats {{\n\
            \treads: {} ({} bytes)\n\
            \twrites: {} ({} bytes)\n\
            \tcache: {} hits, {} misses ({:.1}% hit rate)\n\
            \tconcurrent: {} peak readers, {} peak writers\n\
            \tactive: {} readers, {} writers\n\
            \tthroughput: read {:.2} MB/s, write {:.2} MB/s\n\
            \tWAL: {} appends, {} flushes\n\
            }}",
            self.read_count,
            self.read_bytes,
            self.write_count,
            self.write_bytes,
            self.cache_hits,
            self.cache_misses,
            self.cache_hit_rate,
            self.peak_concurrent_readers,
            self.peak_concurrent_writers,
            self.active_readers,
            self.active_writers,
            self.read_throughput_mbps,
            self.write_throughput_mbps,
            self.wal_append_count,
            self.flush_count,
        )
    }
}

// =============================================================================
// Global Storage Statistics
// =============================================================================

use once_cell::sync::Lazy;
use std::sync::Arc;

/// Global storage statistics instance
static GLOBAL_STORAGE_STATS: Lazy<Arc<StorageStats>> = Lazy::new(|| Arc::new(StorageStats::new()));

/// Get the global storage statistics
pub fn global_stats() -> Arc<StorageStats> {
    Arc::clone(&GLOBAL_STORAGE_STATS)
}

/// Per-backend storage statistics (can be used for per-table stats)
pub type BackendStats = StorageStats;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use tempfile::tempdir;

    #[test]
    fn test_concurrent_stats() {
        let stats = Arc::new(StorageStats::new());

        // Spawn multiple threads writing concurrently
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let stats = Arc::clone(&stats);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        stats.record_read(1024);
                        stats.record_write(512);
                        let _guard = stats.begin_read();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.read_count, 4000);
        assert_eq!(snapshot.write_count, 4000);
        assert_eq!(snapshot.read_bytes, 4096000);
        assert_eq!(snapshot.write_bytes, 2048000);
    }

    #[test]
    fn test_cache_hit_rate() {
        let stats = StorageStats::new();

        stats.record_cache_hit();
        stats.record_cache_hit();
        stats.record_cache_miss();

        let rate = stats.cache_hit_rate();
        assert!((rate - 66.66).abs() < 0.1, "Expected ~66.66%, got {}", rate);
    }

    #[test]
    fn test_peak_concurrent() {
        let stats = StorageStats::new();

        let guards: Vec<_> = (0..5).map(|_| stats.begin_read()).collect();

        assert_eq!(stats.active_readers_count(), 5);
        assert!(stats.peak_concurrent_readers.load(Ordering::Relaxed) >= 5);

        drop(guards);

        assert_eq!(stats.active_readers_count(), 0);
    }
}
