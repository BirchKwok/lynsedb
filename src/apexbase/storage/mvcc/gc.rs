//! Garbage Collector - Removes old row versions no longer visible to any snapshot
//!
//! Runs periodically or on-demand to reclaim memory and storage used by
//! superseded row versions that are no longer needed by any active snapshot.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::snapshot::SnapshotManager;
use super::version_store::VersionStore;

// ============================================================================
// GC Configuration
// ============================================================================

/// Default interval between GC runs (in seconds)
const DEFAULT_GC_INTERVAL_SECS: u64 = 60;

/// Default minimum versions before GC triggers
const DEFAULT_MIN_VERSIONS_FOR_GC: u64 = 1000;

/// GC configuration
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Minimum interval between GC runs
    pub interval: Duration,
    /// Minimum total version count before GC triggers
    pub min_versions: u64,
    /// Whether GC is enabled
    pub enabled: bool,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(DEFAULT_GC_INTERVAL_SECS),
            min_versions: DEFAULT_MIN_VERSIONS_FOR_GC,
            enabled: true,
        }
    }
}

// ============================================================================
// GC Statistics
// ============================================================================

/// Statistics from a GC run
#[derive(Debug, Clone, Default)]
pub struct GcStats {
    /// Number of versions removed
    pub versions_removed: usize,
    /// Duration of the GC run
    pub duration: Duration,
    /// Oldest active timestamp used for this GC
    pub oldest_active_ts: u64,
}

// ============================================================================
// Garbage Collector
// ============================================================================

/// Garbage collector for MVCC version store
///
/// Determines which old versions can be safely removed based on
/// the oldest active snapshot, then removes them from the version store.
///
/// Safety guarantee: never removes a version that might still be
/// visible to any active snapshot.
pub struct GarbageCollector {
    /// Configuration
    config: GcConfig,
    /// Last GC run time
    last_run: AtomicU64, // stored as epoch millis
    /// Whether a GC is currently in progress
    running: AtomicBool,
    /// Total versions removed across all GC runs
    total_removed: AtomicU64,
}

impl GarbageCollector {
    /// Create a new garbage collector with default config
    pub fn new() -> Self {
        Self {
            config: GcConfig::default(),
            last_run: AtomicU64::new(0),
            running: AtomicBool::new(false),
            total_removed: AtomicU64::new(0),
        }
    }

    /// Create with custom config
    pub fn with_config(config: GcConfig) -> Self {
        Self {
            config,
            last_run: AtomicU64::new(0),
            running: AtomicBool::new(false),
            total_removed: AtomicU64::new(0),
        }
    }

    /// Check if GC should run based on heuristics
    pub fn should_run(&self, version_store: &VersionStore) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Don't run if already running
        if self.running.load(Ordering::Relaxed) {
            return false;
        }

        // Check minimum version count
        if version_store.total_versions() < self.config.min_versions {
            return false;
        }

        // Check interval
        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let last = self.last_run.load(Ordering::Relaxed);
        let interval_millis = self.config.interval.as_millis() as u64;

        now_millis.saturating_sub(last) >= interval_millis
    }

    /// Run garbage collection
    ///
    /// Returns statistics about the GC run, or None if GC was skipped.
    pub fn run(
        &self,
        version_store: &VersionStore,
        snapshot_manager: &SnapshotManager,
    ) -> Option<GcStats> {
        // Try to acquire the running flag (CAS to prevent concurrent GC)
        if self
            .running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_err()
        {
            return None; // Another GC is already running
        }

        let start = Instant::now();
        let oldest_active_ts = snapshot_manager.oldest_active_timestamp();

        // If there are active snapshots with very old timestamps, GC won't help much
        // but we still run to clean up what we can
        let removed = version_store.gc(oldest_active_ts);

        let duration = start.elapsed();
        self.total_removed
            .fetch_add(removed as u64, Ordering::Relaxed);

        // Update last run time
        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_run.store(now_millis, Ordering::Relaxed);

        // Release running flag
        self.running.store(false, Ordering::SeqCst);

        Some(GcStats {
            versions_removed: removed,
            duration,
            oldest_active_ts,
        })
    }

    /// Try to run GC if conditions are met (non-blocking)
    pub fn maybe_run(
        &self,
        version_store: &VersionStore,
        snapshot_manager: &SnapshotManager,
    ) -> Option<GcStats> {
        if self.should_run(version_store) {
            self.run(version_store, snapshot_manager)
        } else {
            None
        }
    }

    /// Total versions removed across all GC runs
    pub fn total_removed(&self) -> u64 {
        self.total_removed.load(Ordering::Relaxed)
    }

    /// Whether GC is currently running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Update GC configuration
    pub fn set_config(&mut self, config: GcConfig) {
        self.config = config;
    }
}

impl Default for GarbageCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Value;
    use crate::storage::mvcc::version_store::next_timestamp;
    use std::collections::HashMap;

    fn make_row(name: &str) -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), Value::String(name.to_string()));
        m
    }

    #[test]
    fn test_gc_removes_old_versions() {
        let vs = VersionStore::new();
        let sm = SnapshotManager::new();
        let gc = GarbageCollector::with_config(GcConfig {
            min_versions: 0,
            interval: Duration::from_secs(0),
            enabled: true,
        });

        // Insert and update multiple times
        let ts1 = next_timestamp();
        vs.insert(1, ts1, make_row("v1"));

        let ts2 = next_timestamp();
        vs.update(1, ts2, make_row("v2")).unwrap();

        let ts3 = next_timestamp();
        vs.update(1, ts3, make_row("v3")).unwrap();

        assert_eq!(vs.total_versions(), 3);

        // No active snapshots → GC can clean up old versions
        let stats = gc.run(&vs, &sm).unwrap();
        assert!(stats.versions_removed > 0);
    }

    #[test]
    fn test_gc_preserves_visible_versions() {
        let vs = VersionStore::new();
        let sm = SnapshotManager::new();
        let gc = GarbageCollector::with_config(GcConfig {
            min_versions: 0,
            interval: Duration::from_secs(0),
            enabled: true,
        });

        let ts1 = next_timestamp();
        vs.insert(1, ts1, make_row("v1"));

        // Take a snapshot at ts1
        let snap = sm.create_snapshot();

        let ts2 = next_timestamp();
        vs.update(1, ts2, make_row("v2")).unwrap();

        // GC should NOT remove v1 because snap still sees it
        gc.run(&vs, &sm);

        // Snapshot should still see v1
        let data = vs.read(1, snap.read_ts);
        assert!(data.is_some());

        sm.release(snap.id);
    }

    #[test]
    fn test_gc_disabled() {
        let vs = VersionStore::new();
        let gc = GarbageCollector::with_config(GcConfig {
            enabled: false,
            ..Default::default()
        });

        assert!(!gc.should_run(&vs));
    }
}
