//! Full-Text Search Module - Native Rust NanoFTS Integration
//!
//! This module provides high-performance full-text search capabilities
//! by directly integrating the nanofts crate, eliminating Python-Rust
//! boundary crossing overhead.

use nanofts::{EngineConfig, ResultHandle, UnifiedEngine};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// FTS Engine wrapper for ApexBase
///
/// Provides a thread-safe wrapper around nanofts UnifiedEngine
///
/// Note: nanofts uses u32 for doc IDs internally, but ApexBase uses u64
/// We handle the conversion transparently.
pub struct FtsEngine {
    engine: RwLock<Option<UnifiedEngine>>,
    #[allow(dead_code)]
    index_path: PathBuf,
    #[allow(dead_code)]
    config: FtsConfig,
}

/// FTS Configuration
#[derive(Clone, Debug)]
pub struct FtsConfig {
    /// Maximum length for Chinese n-gram tokens
    pub max_chinese_length: usize,
    /// Minimum term length
    pub min_term_length: usize,
    /// Fuzzy search similarity threshold (0.0 - 1.0)
    pub fuzzy_threshold: f64,
    /// Maximum edit distance for fuzzy search
    pub fuzzy_max_distance: usize,
    /// Track document terms (required for deletion support)
    pub track_doc_terms: bool,
    /// Enable lazy loading for large indexes
    pub lazy_load: bool,
    /// LRU cache size
    pub cache_size: usize,
}

impl Default for FtsConfig {
    fn default() -> Self {
        Self {
            max_chinese_length: 4,
            min_term_length: 2,
            fuzzy_threshold: 0.7,
            fuzzy_max_distance: 2,
            track_doc_terms: false,
            lazy_load: false,
            cache_size: 10000,
        }
    }
}

/// Custom error type for FTS operations
#[derive(Debug)]
pub enum FtsError {
    EngineError(String),
    NotInitialized,
}

impl std::fmt::Display for FtsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FtsError::EngineError(e) => write!(f, "FTS Engine error: {}", e),
            FtsError::NotInitialized => write!(f, "FTS Engine not initialized"),
        }
    }
}

impl std::error::Error for FtsError {}

impl From<nanofts::EngineError> for FtsError {
    fn from(e: nanofts::EngineError) -> Self {
        FtsError::EngineError(e.to_string())
    }
}

pub type FtsResult<T> = Result<T, FtsError>;

impl FtsEngine {
    /// Create a new FTS engine with the specified index path
    pub fn new<P: AsRef<Path>>(index_path: P, config: FtsConfig) -> FtsResult<Self> {
        let path = index_path.as_ref().to_path_buf();

        // Create engine config
        let engine_config = EngineConfig::persistent(path.to_str().unwrap_or(""))
            .with_lazy_load(config.lazy_load)
            .with_cache_size(config.cache_size)
            .with_track_doc_terms(config.track_doc_terms);

        let engine = UnifiedEngine::new(engine_config)?;

        Ok(Self {
            engine: RwLock::new(Some(engine)),
            index_path: path,
            config,
        })
    }

    /// Create an in-memory FTS engine (for testing)
    pub fn memory_only(config: FtsConfig) -> FtsResult<Self> {
        let engine_config = EngineConfig::memory_only();
        let engine = UnifiedEngine::new(engine_config)?;

        Ok(Self {
            engine: RwLock::new(Some(engine)),
            index_path: PathBuf::new(),
            config,
        })
    }

    /// Add a document to the index
    ///
    /// # Arguments
    /// * `doc_id` - Unique document identifier (u64, will be truncated to u32)
    /// * `fields` - Document fields to index (field_name -> text)
    pub fn add_document(&self, doc_id: u64, fields: HashMap<String, String>) -> FtsResult<()> {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            engine.add_document(doc_id as u32, fields)?;
        }
        Ok(())
    }

    /// Add multiple documents to the index (batch operation - uses parallel processing)
    ///
    /// # Arguments
    /// * `docs` - Vector of (doc_id, fields) tuples
    pub fn add_documents(&self, docs: Vec<(u64, HashMap<String, String>)>) -> FtsResult<()> {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            // Convert u64 doc_ids to u32 for nanofts
            let docs_u32: Vec<(u32, HashMap<String, String>)> = docs
                .into_iter()
                .map(|(id, fields)| (id as u32, fields))
                .collect();
            // Use nanofts batch method (internally uses parallel processing)
            engine.add_documents(docs_u32)?;
        }
        Ok(())
    }

    /// Add documents with pre-concatenated text (fastest path - no HashMap overhead)
    ///
    /// # Arguments
    /// * `doc_ids` - Vector of document IDs
    /// * `texts` - Vector of text content (same length as doc_ids)
    pub fn add_documents_texts(&self, doc_ids: Vec<u64>, texts: Vec<String>) -> FtsResult<()> {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            let doc_ids_u32: Vec<u32> = doc_ids.into_iter().map(|id| id as u32).collect();
            engine.add_documents_texts(doc_ids_u32, texts)?;
        }
        Ok(())
    }

    /// Add documents with columnar data (owned Strings)
    ///
    /// # Arguments
    /// * `doc_ids` - Vector of document IDs
    /// * `columns` - Vector of (field_name, values) where values has same length as doc_ids
    pub fn add_documents_columnar(
        &self,
        doc_ids: Vec<u64>,
        columns: Vec<(String, Vec<String>)>,
    ) -> FtsResult<()> {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            let doc_ids_u32: Vec<u32> = doc_ids.into_iter().map(|id| id as u32).collect();
            engine.add_documents_columnar(doc_ids_u32, columns)?;
        }
        Ok(())
    }

    /// 🥇 Fastest path: add documents with pre-joined text as &str slices (zero-copy)
    ///
    /// ~3.4M docs/s. Use when text fields are already concatenated.
    ///
    /// # Arguments
    /// * `doc_ids` - Slice of u32 document IDs
    /// * `texts` - Slice of &str text (same length as doc_ids)
    pub fn add_documents_arrow_texts(&self, doc_ids: &[u32], texts: &[&str]) -> FtsResult<()> {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            engine.add_documents_arrow_texts(doc_ids, texts)?;
        }
        Ok(())
    }

    /// 🥈 Second fastest: multi-column &str slices (zero-copy, Arrow-format)
    ///
    /// ~3.3M docs/s. Use when data comes from columnar sources (Arrow, Parquet, DataFrame).
    ///
    /// # Arguments
    /// * `doc_ids` - Slice of u32 document IDs
    /// * `columns` - Vec of (field_name, Vec<&str>) column data
    pub fn add_documents_arrow_str(
        &self,
        doc_ids: &[u32],
        columns: Vec<(String, Vec<&str>)>,
    ) -> FtsResult<()> {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            engine.add_documents_arrow_str(doc_ids, columns)?;
        }
        Ok(())
    }

    /// Remove a document from the index
    pub fn remove_document(&self, doc_id: u64) -> FtsResult<()> {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            engine.remove_document(doc_id as u32)?;
        }
        Ok(())
    }

    /// Remove multiple documents from the index
    pub fn remove_documents(&self, doc_ids: &[u64]) -> FtsResult<()> {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            for &doc_id in doc_ids {
                engine.remove_document(doc_id as u32)?;
            }
        }
        Ok(())
    }

    /// Update a document in the index
    pub fn update_document(&self, doc_id: u64, fields: HashMap<String, String>) -> FtsResult<()> {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            engine.update_document(doc_id as u32, fields)?;
        }
        Ok(())
    }

    /// Search for documents matching the query
    ///
    /// Returns a ResultHandle containing matching document IDs
    pub fn search(&self, query: &str) -> FtsResult<ResultHandle> {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            Ok(engine.search(query)?)
        } else {
            Err(FtsError::NotInitialized)
        }
    }

    /// Fuzzy search for documents matching the query (tolerates typos)
    pub fn fuzzy_search(&self, query: &str, min_results: usize) -> FtsResult<ResultHandle> {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            Ok(engine.fuzzy_search(query, min_results)?)
        } else {
            Err(FtsError::NotInitialized)
        }
    }

    /// Search and return document IDs as a Vec<u64>
    pub fn search_ids(&self, query: &str) -> FtsResult<Vec<u64>> {
        let result = self.search(query)?;
        // Convert u32 to u64
        Ok(result.iter().map(|id| id as u64).collect())
    }

    /// Search and return top N document IDs
    pub fn search_top_n(&self, query: &str, n: usize) -> FtsResult<Vec<u64>> {
        let result = self.search(query)?;
        // Convert u32 to u64
        Ok(result.page(0, n).into_iter().map(|id| id as u64).collect())
    }

    /// Get paginated results (convert u32 to u64)
    pub fn search_page(&self, query: &str, offset: usize, limit: usize) -> FtsResult<Vec<u64>> {
        let result = self.search(query)?;
        Ok(result
            .page(offset, limit)
            .into_iter()
            .map(|id| id as u64)
            .collect())
    }

    /// Flush changes to disk
    pub fn flush(&self) -> FtsResult<()> {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            engine.flush()?;
        }
        Ok(())
    }

    /// Start an asynchronous flush. Returns immediately after making data searchable in memory;
    /// actual disk write (fsync) runs in a background thread.
    /// Use `wait_flush()` if you need to confirm durability.
    pub fn flush_async(&self) -> FtsResult<()> {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            engine.flush_async()?;
        }
        Ok(())
    }

    /// Wait for a previously started `flush_async()` to complete.
    /// Returns the number of terms flushed, or 0 if no background flush is pending.
    pub fn wait_flush(&self) -> FtsResult<usize> {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            Ok(engine.wait_flush()?)
        } else {
            Ok(0)
        }
    }

    /// Compact the index (apply deletions and optimize storage)
    pub fn compact(&self) -> FtsResult<()> {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            engine.compact()?;
        }
        Ok(())
    }

    /// Get index statistics
    pub fn stats(&self) -> HashMap<String, u64> {
        let guard = self.engine.read();
        let mut result_map = HashMap::new();

        if let Some(ref engine) = *guard {
            let stats = engine.stats();
            // stats is HashMap<String, f64>, convert to u64
            for (k, v) in stats {
                result_map.insert(k, v as u64);
            }
        }

        result_map
    }

    /// Set fuzzy search configuration
    pub fn set_fuzzy_config(&self, threshold: f64, max_distance: usize, max_candidates: usize) {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            engine.set_fuzzy_config(threshold, max_distance, max_candidates);
        }
    }

    /// Warmup cache with specific terms (useful for lazy load mode)
    pub fn warmup_terms(&self, terms: &[String]) -> usize {
        let guard = self.engine.read();
        if let Some(ref engine) = *guard {
            engine.warmup_terms(terms.to_vec())
        } else {
            0
        }
    }
}

/// FTS Manager - Manages multiple FTS engines for different tables
pub struct FtsManager {
    base_path: PathBuf,
    engines: RwLock<HashMap<String, Arc<FtsEngine>>>,
    default_config: FtsConfig,
}

impl FtsManager {
    /// Create a new FTS manager
    pub fn new<P: AsRef<Path>>(base_path: P, config: FtsConfig) -> Self {
        let path = base_path.as_ref().to_path_buf();

        // Ensure the directory exists
        if !path.exists() {
            let _ = std::fs::create_dir_all(&path);
        }

        Self {
            base_path: path,
            engines: RwLock::new(HashMap::new()),
            default_config: config,
        }
    }

    /// Get or create an FTS engine for a table
    pub fn get_engine(&self, table_name: &str) -> FtsResult<Arc<FtsEngine>> {
        // Fast path: check if engine exists
        {
            let engines = self.engines.read();
            if let Some(engine) = engines.get(table_name) {
                return Ok(Arc::clone(engine));
            }
        }

        // Slow path: create engine
        let mut engines = self.engines.write();

        // Double-check after acquiring write lock
        if let Some(engine) = engines.get(table_name) {
            return Ok(Arc::clone(engine));
        }

        // Create new engine
        let index_path = self.base_path.join(format!("{}.nfts", table_name));
        let engine = Arc::new(FtsEngine::new(&index_path, self.default_config.clone())?);
        engines.insert(table_name.to_string(), Arc::clone(&engine));

        Ok(engine)
    }

    /// Remove an FTS engine for a table (and optionally delete index files)
    pub fn remove_engine(&self, table_name: &str, delete_files: bool) -> FtsResult<()> {
        let mut engines = self.engines.write();
        engines.remove(table_name);

        if delete_files {
            let index_path = self.base_path.join(format!("{}.nfts", table_name));
            let wal_path = self.base_path.join(format!("{}.nfts.wal", table_name));

            if index_path.exists() {
                let _ = std::fs::remove_file(&index_path);
            }
            if wal_path.exists() {
                let _ = std::fs::remove_file(&wal_path);
            }
        }

        Ok(())
    }

    /// Flush all engines
    pub fn flush_all(&self) -> FtsResult<()> {
        let engines = self.engines.read();
        for engine in engines.values() {
            engine.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_engine() {
        let config = FtsConfig::default();
        let engine = FtsEngine::memory_only(config).unwrap();

        // Add a document
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), "Hello World".to_string());
        fields.insert("content".to_string(), "This is a test document".to_string());
        engine.add_document(1, fields).unwrap();

        // Search
        let result = engine.search("hello").unwrap();
        assert_eq!(result.total_hits(), 1);

        // Get IDs
        let ids: Vec<u64> = result.iter().map(|id| id as u64).collect();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn test_batch_add() {
        let config = FtsConfig::default();
        let engine = FtsEngine::memory_only(config).unwrap();

        // Add multiple documents
        let docs: Vec<(u64, HashMap<String, String>)> = (0..100)
            .map(|i| {
                let mut fields = HashMap::new();
                fields.insert("title".to_string(), format!("Document {}", i));
                fields.insert("content".to_string(), format!("Content for document {}", i));
                (i as u64, fields)
            })
            .collect();

        engine.add_documents(docs).unwrap();

        // Search
        let result = engine.search("Document").unwrap();
        assert_eq!(result.total_hits(), 100);
    }
}
